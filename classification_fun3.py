import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm
import os
import json
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.utils.class_weight import compute_class_weight
import re
from collections import defaultdict
import nlpaug.augmenter.char as nac
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 实验配置
EXPERIMENTS = [
    {"name": "baseline", "use_aug": True, "use_gru": True, "use_shared": True, "model_type": "hierarchical"},
    {"name": "no_aug", "use_aug": False, "use_gru": True, "use_shared": True, "model_type": "hierarchical"},
    {"name": "no_gru", "use_aug": True, "use_gru": False, "use_shared": True, "model_type": "hierarchical"},
    {"name": "no_shared", "use_aug": True, "use_gru": True, "use_shared": False, "model_type": "hierarchical"},
    {"name": "flat_bert", "use_aug": True, "use_gru": False, "use_shared": False, "model_type": "flat"}
]

# 全局配置
CONFIG = {
    "max_len": 128,
    "batch_size": 16,
    "epochs": 20,
    "lr": 2e-5,
    "warmup_steps": 100,
    "val_size": 0.2,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "gradient_accumulation_steps": 1,
    "model_path": "pretrained_models/chinese-roberta-wwm-ext",
    "num_workers": 4,
    "pin_memory": True,
    "results_dir": "experiment_results",
    "tail_threshold": 10  # 长尾类别阈值（样本数少于该值的类别视为长尾类别）
}

# 确保结果目录存在
os.makedirs(CONFIG['results_dir'], exist_ok=True)


# 文本清洗函数
def clean_text(text: str) -> str:
    """基础文本清洗"""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()


# 优化后的数据集类
class OptimizedHierarchyDataset(Dataset):
    def __init__(self, texts, main_labels, sub_labels, tokenizer, max_len, augmenter=None):
        self.texts = texts
        # 将标签转换为长整型张量
        self.main_labels = torch.tensor(main_labels, dtype=torch.long)
        self.sub_labels = torch.tensor(sub_labels, dtype=torch.long)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.augmenter = augmenter

        # 预编码所有文本
        logger.info("Pre-tokenizing texts...")
        self.encodings = []
        for text in tqdm(texts, desc="Tokenizing"):
            # 数据增强
            if self.augmenter:
                text = self.augmenter.augment(text)[0]

            encoding = tokenizer(
                str(text),
                max_length=max_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            self.encodings.append({
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten()
            })

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {
            **self.encodings[idx],
            'main_label': self.main_labels[idx],
            'sub_label': self.sub_labels[idx]
        }


# 文本增强器
def create_text_augmenter():
    return nac.KeyboardAug(aug_char_max=3, aug_word_p=0.3)


# 修复子分类问题的层级BERT模型
class FixedHierarchicalBERT(nn.Module):
    """修复子分类问题的层级分类模型"""

    def __init__(self, num_main: int, num_sub: int, model_path: str,
                 hierarchy_map: Dict[int, List[int]],
                 use_gru: bool = True, use_shared: bool = True):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_path)
        self.num_main = num_main
        self.num_sub = num_sub
        self.hierarchy = hierarchy_map
        self.use_gru = use_gru
        self.use_shared = use_shared

        # 冻结部分层
        for i, layer in enumerate(self.bert.encoder.layer[:8]):
            for param in layer.parameters():
                param.requires_grad = False

        # GRU层
        if use_gru:
            self.rnn = nn.GRU(
                input_size=self.bert.config.hidden_size,
                hidden_size=256,
                num_layers=1,
                bidirectional=True,
                batch_first=True
            )

        # 主分类器
        classifier_input_size = 512 if use_gru else self.bert.config.hidden_size
        self.main_classifier = nn.Sequential(
            nn.Linear(classifier_input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_main)
        )

        # 子分类器和映射
        self.sub_classifiers = nn.ModuleDict()
        self.sub_mappings = {}
        self.main_to_sub = {}  # 主类到子类的映射字典

        # 共享特征提取层
        if use_shared:
            self.shared_feature_extractor = nn.Sequential(
                nn.Linear(classifier_input_size, 128),
                nn.ReLU(),
                nn.Dropout(0.3)
            )

        # 为每个有子类的主类创建子分类器
        for main_id in range(num_main):
            sub_ids = hierarchy_map.get(main_id, [])
            if sub_ids:
                # 创建主类到子类的映射
                self.main_to_sub[main_id] = sub_ids

                # 创建子类ID到本地ID的映射
                self.sub_mappings[main_id] = {
                    local_id: global_id
                    for local_id, global_id in enumerate(sub_ids)
                }

                # 创建子分类器
                if use_shared:
                    self.sub_classifiers[str(main_id)] = nn.Linear(128, len(sub_ids))
                else:
                    self.sub_classifiers[str(main_id)] = nn.Sequential(
                        nn.Linear(classifier_input_size, 128),
                        nn.ReLU(),
                        nn.Dropout(0.3),
                        nn.Linear(128, len(sub_ids))
                    )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                main_labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = self.bert(input_ids, attention_mask)
        sequence_output = outputs.last_hidden_state

        # 特征提取
        if self.use_gru:
            rnn_output, _ = self.rnn(sequence_output)
            pooled = torch.mean(rnn_output, dim=1)  # 平均池化
        else:
            pooled = outputs.pooler_output  # 使用[CLS]标记

        # 主分类
        main_logits = self.main_classifier(pooled)

        # 子分类
        batch_size = input_ids.size(0)
        sub_logits = torch.full((batch_size, self.num_sub), -1e9, device=input_ids.device)

        # 训练和推理统一处理逻辑
        if main_labels is None:
            main_labels = torch.argmax(main_logits, dim=1)

        # 提取共享特征
        if self.use_shared:
            features = self.shared_feature_extractor(pooled)
        else:
            features = pooled

        # 为每个样本生成子类logits
        for i in range(batch_size):
            main_id = main_labels[i].item()

            # 检查主类是否有子分类器
            if main_id in self.main_to_sub:
                # 获取子分类器
                classifier_key = str(main_id)

                # 提取特征
                if self.use_shared:
                    feature = features[i].unsqueeze(0)
                else:
                    feature = pooled[i].unsqueeze(0)

                # 获取子类logits
                local_logits = self.sub_classifiers[classifier_key](feature)

                # 将本地logits映射到全局logits
                for local_id, global_id in self.sub_mappings[main_id].items():
                    sub_logits[i, global_id] = local_logits[0, local_id]

        return main_logits, sub_logits


# 平铺BERT模型（非层级结构）
class FlatBERT(nn.Module):
    """平铺结构BERT模型"""

    def __init__(self, num_classes: int, model_path: str):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_path)
        self.classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

        # 冻结部分层
        for i, layer in enumerate(self.bert.encoder.layer[:8]):
            for param in layer.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask)
        pooled = outputs.pooler_output
        return self.classifier(pooled)


# 修复后的训练函数
def fixed_train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                      optimizer: torch.optim.Optimizer, device: str, epochs: int,
                      num_main: int, num_sub: int,
                      scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                      experiment_name: str = "experiment"):
    """训练模型并修复子分类问题"""
    # 收集所有标签用于类别权重计算
    all_main_labels = []
    all_sub_labels = []
    for batch in train_loader:
        all_main_labels.extend(batch['main_label'].tolist())
        all_sub_labels.extend(batch['sub_label'].tolist())

    # 计算类别权重 - 修复维度不匹配问题
    # 获取训练集中实际存在的类别
    unique_main_labels = np.unique(all_main_labels)
    computed_main_weights = compute_class_weight(
        'balanced',
        classes=unique_main_labels,
        y=all_main_labels
    )
    # 创建全1权重数组，长度等于总类别数
    main_weights = np.ones(num_main)
    # 将计算得到的权重赋给实际存在的类别
    for idx, cls in enumerate(unique_main_labels):
        main_weights[cls] = computed_main_weights[idx]
    main_criterion = nn.CrossEntropyLoss(weight=torch.tensor(main_weights, dtype=torch.float).to(device))

    # 子类权重计算
    valid_sub_labels = [label for label in all_sub_labels if label != -1]
    if valid_sub_labels:
        unique_sub_labels = np.unique(valid_sub_labels)
        computed_sub_weights = compute_class_weight(
            'balanced',
            classes=unique_sub_labels,
            y=valid_sub_labels
        )
        sub_weights = np.ones(num_sub)
        for idx, cls in enumerate(unique_sub_labels):
            sub_weights[cls] = computed_sub_weights[idx]
        sub_criterion = nn.CrossEntropyLoss(weight=torch.tensor(sub_weights, dtype=torch.float).to(device))
    else:
        sub_criterion = None
        logger.warning("No valid sub labels found. Skipping sub classification loss.")

    best_acc = 0.0
    early_stop_counter = 0
    best_model = None

    # 存储训练指标
    train_metrics = {
        'loss': [],
        'main_acc': [],
        'sub_acc': []
    }

    # 存储验证指标
    val_metrics = {
        'main_acc': [],
        'sub_acc': []
    }

    # 判断是否为层级模型
    is_hierarchical = hasattr(model, 'main_to_sub')

    for epoch in range(epochs):
        logger.info(f"\nEpoch {epoch + 1}/{epochs}")

        # 训练
        model.train()
        train_loss = 0.0
        main_correct = 0
        sub_correct = 0
        total_samples = 0
        sub_samples = 0  # 有效的子类样本数

        progress_bar = tqdm(train_loader, desc="Training", leave=False)
        for batch in progress_bar:
            inputs = batch['input_ids'].to(device, non_blocking=True)
            masks = batch['attention_mask'].to(device, non_blocking=True)
            main_labels = batch['main_label'].to(device, non_blocking=True)
            sub_labels = batch['sub_label'].to(device, non_blocking=True)

            optimizer.zero_grad()

            # 修改点：根据模型类型调用不同的forward方法
            if is_hierarchical:
                # 层级模型：传递main_labels参数
                main_logits, sub_logits = model(inputs, masks, main_labels=main_labels)
            else:
                # 平铺模型：不传递main_labels参数
                main_logits = model(inputs, masks)
                sub_logits = None

            main_loss = main_criterion(main_logits, main_labels)

            # 计算子类损失
            sub_loss = 0
            if is_hierarchical and sub_criterion:
                # 确定哪些样本有子类
                has_subclass = torch.tensor([
                    main_id in model.main_to_sub for main_id in main_labels.cpu().numpy()
                ], device=device).bool()

                if has_subclass.any():
                    # 只计算有子类的样本
                    valid_sub_logits = sub_logits[has_subclass]
                    valid_sub_labels = sub_labels[has_subclass]

                    # 计算子类损失
                    sub_loss = sub_criterion(valid_sub_logits, valid_sub_labels)

            # 平衡主类和子类损失
            loss = main_loss + (0.5 * sub_loss if is_hierarchical and sub_loss != 0 else 0)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if scheduler:
                scheduler.step()

            train_loss += loss.item()

            # 计算训练准确率
            main_preds = torch.argmax(main_logits, dim=1)
            main_correct += (main_preds == main_labels).sum().item()

            # 计算子类准确率
            if is_hierarchical and sub_criterion and has_subclass.any():
                sub_preds = torch.argmax(valid_sub_logits, dim=1)
                sub_correct += (sub_preds == valid_sub_labels).sum().item()
                sub_samples += valid_sub_labels.size(0)

            total_samples += inputs.size(0)

            # 更新进度条
            progress_bar.set_postfix({
                'loss': f"{train_loss / (progress_bar.n + 1):.4f}",
                'main_acc': f"{main_correct / total_samples:.4f}",
                'sub_acc': f"{sub_correct / sub_samples:.4f}" if is_hierarchical and sub_samples > 0 else "N/A"
            })

        # 记录训练指标
        avg_loss = train_loss / len(train_loader)
        train_main_acc = main_correct / total_samples
        train_sub_acc = sub_correct / sub_samples if is_hierarchical and sub_samples > 0 else 0.0

        train_metrics['loss'].append(avg_loss)
        train_metrics['main_acc'].append(train_main_acc)
        train_metrics['sub_acc'].append(train_sub_acc)

        # 验证
        model.eval()
        val_main_correct = 0
        val_sub_correct = 0
        val_samples = 0
        val_sub_samples = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating", leave=False):
                inputs = batch['input_ids'].to(device, non_blocking=True)
                masks = batch['attention_mask'].to(device, non_blocking=True)
                main_labels = batch['main_label'].to(device, non_blocking=True)
                sub_labels = batch['sub_label'].to(device, non_blocking=True)

                # 修改点：验证时根据模型类型调用不同的forward方法
                if is_hierarchical:
                    main_logits, sub_logits = model(inputs, masks)
                else:
                    main_logits = model(inputs, masks)
                    sub_logits = None

                main_preds = torch.argmax(main_logits, dim=1)
                val_main_correct += (main_preds == main_labels).sum().item()
                val_samples += inputs.size(0)

                # 计算子类准确率（仅层级模型）
                if is_hierarchical and sub_criterion:
                    # 确定哪些样本有子类
                    has_subclass = torch.tensor([
                        main_id in model.main_to_sub for main_id in main_preds.cpu().numpy()
                    ], device=device).bool()

                    if has_subclass.any():
                        valid_sub_logits = sub_logits[has_subclass]
                        valid_sub_labels = sub_labels[has_subclass]
                        sub_preds = torch.argmax(valid_sub_logits, dim=1)
                        val_sub_correct += (sub_preds == valid_sub_labels).sum().item()
                        val_sub_samples += valid_sub_labels.size(0)

        # 日志指标
        val_main_acc = val_main_correct / val_samples
        val_sub_acc = val_sub_correct / val_sub_samples if is_hierarchical and val_sub_samples > 0 else 0.0

        val_metrics['main_acc'].append(val_main_acc)
        val_metrics['sub_acc'].append(val_sub_acc)

        logger.info(f"Train Loss: {avg_loss:.4f}")
        logger.info(f"Train Main Acc: {train_main_acc:.4f} | Train Sub Acc: {train_sub_acc:.4f}")
        logger.info(f"Val Main Acc: {val_main_acc:.4f} | Val Sub Acc: {val_sub_acc:.4f}")

        # 早停和模型保存
        combined_acc = (val_main_acc + (val_sub_acc if is_hierarchical else 0)) / (2 if is_hierarchical else 1)
        if combined_acc > best_acc:
            best_acc = combined_acc
            early_stop_counter = 0
            best_model = model.state_dict()
            logger.info("Best model found!")
        else:
            early_stop_counter += 1
            if early_stop_counter >= 3:
                logger.info("Early stopping triggered!")
                break

    # 加载最佳模型
    if best_model:
        model.load_state_dict(best_model)

    # 保存训练指标
    metrics = {
        'train': train_metrics,
        'val': val_metrics
    }

    # 创建实验目录
    exp_dir = os.path.join(CONFIG['results_dir'], experiment_name)
    os.makedirs(exp_dir, exist_ok=True)

    # 保存指标
    with open(os.path.join(exp_dir, 'training_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    return model


# 模型评估函数
def evaluate_model(model, data_loader, device, label_encoders, experiment_name, is_flat=False):
    """评估模型并保存结果"""
    model.eval()

    # 创建实验目录
    exp_dir = os.path.join(CONFIG['results_dir'], experiment_name)
    os.makedirs(exp_dir, exist_ok=True)

    # 存储所有预测和真实标签
    all_main_preds = []
    all_main_true = []
    all_sub_preds = []
    all_sub_true = []
    all_texts = []

    # 存储每个样本的预测结果
    sample_results = []

    # 判断是否为层级模型
    is_hierarchical = hasattr(model, 'main_to_sub') and not is_flat

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            inputs = batch['input_ids'].to(device)
            masks = batch['attention_mask'].to(device)
            main_labels = batch['main_label'].numpy()
            sub_labels = batch['sub_label'].numpy()
            texts = batch.get('text', [""] * len(main_labels))  # 如果没有文本，使用空字符串

            if is_flat:
                # 平铺模型处理
                logits = model(inputs, masks)
                flat_preds = torch.argmax(logits, dim=1).cpu().numpy()

                # 将平铺预测转换为层级预测（这里简化处理，实际需要根据任务设计）
                main_preds = flat_preds
                sub_preds = [-1] * len(main_preds)  # 平铺模型没有子类预测
            else:
                # 层级模型处理
                main_logits, sub_logits = model(inputs, masks)
                main_preds = torch.argmax(main_logits, dim=1).cpu().numpy()
                sub_preds = torch.argmax(sub_logits, dim=1).cpu().numpy()

            all_main_preds.extend(main_preds)
            all_main_true.extend(main_labels)
            all_sub_preds.extend(sub_preds)
            all_sub_true.extend(sub_labels)
            all_texts.extend(texts)

            # 保存每个样本的结果
            for i in range(len(main_labels)):
                sample_results.append({
                    'text': texts[i],
                    'true_main': label_encoders['main'].inverse_transform([main_labels[i]])[0],
                    'pred_main': label_encoders['main'].inverse_transform([main_preds[i]])[0],
                    'true_sub': label_encoders['sub'].inverse_transform([sub_labels[i]])[0] if sub_labels[
                                                                                                   i] != -1 else -1,
                    'pred_sub': label_encoders['sub'].inverse_transform([sub_preds[i]])[0] if sub_preds[i] != -1 else -1
                })

    # 计算主类别指标
    main_metrics = {
        'accuracy': accuracy_score(all_main_true, all_main_preds),
        'precision_macro': precision_score(all_main_true, all_main_preds, average='macro', zero_division=0),
        'recall_macro': recall_score(all_main_true, all_main_preds, average='macro', zero_division=0),
        'f1_macro': f1_score(all_main_true, all_main_preds, average='macro', zero_division=0),
        'precision_micro': precision_score(all_main_true, all_main_preds, average='micro', zero_division=0),
        'recall_micro': recall_score(all_main_true, all_main_preds, average='micro', zero_division=0),
        'f1_micro': f1_score(all_main_true, all_main_preds, average='micro', zero_division=0)
    }

    # 计算子类别指标（只考虑有效子类样本）
    valid_sub_indices = [i for i, label in enumerate(all_sub_true) if label != -1]
    if valid_sub_indices:
        valid_sub_true = [all_sub_true[i] for i in valid_sub_indices]
        valid_sub_preds = [all_sub_preds[i] for i in valid_sub_indices]

        sub_metrics = {
            'accuracy': accuracy_score(valid_sub_true, valid_sub_preds),
            'precision_macro': precision_score(valid_sub_true, valid_sub_preds, average='macro', zero_division=0),
            'recall_macro': recall_score(valid_sub_true, valid_sub_preds, average='macro', zero_division=0),
            'f1_macro': f1_score(valid_sub_true, valid_sub_preds, average='macro', zero_division=0),
            'precision_micro': precision_score(valid_sub_true, valid_sub_preds, average='micro', zero_division=0),
            'recall_micro': recall_score(valid_sub_true, valid_sub_preds, average='micro', zero_division=0),
            'f1_micro': f1_score(valid_sub_true, valid_sub_preds, average='micro', zero_division=0)
        }
    else:
        sub_metrics = {
            'accuracy': 0,
            'precision_macro': 0,
            'recall_macro': 0,
            'f1_macro': 0,
            'precision_micro': 0,
            'recall_micro': 0,
            'f1_micro': 0
        }

    # 合并指标
    metrics = {
        'main_metrics': main_metrics,
        'sub_metrics': sub_metrics
    }

    # 保存指标
    with open(os.path.join(exp_dir, 'evaluation_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    # 保存样本预测结果
    results_df = pd.DataFrame(sample_results)
    results_df.to_csv(os.path.join(exp_dir, 'predictions.csv'), index=False)

    # 计算并保存混淆矩阵数据 - 修复维度问题
    # 获取实际出现的唯一类别
    unique_main = np.unique(all_main_true)
    main_class_names = label_encoders['main'].inverse_transform(unique_main)

    main_cm = confusion_matrix(all_main_true, all_main_preds, labels=unique_main)
    main_cm_df = pd.DataFrame(main_cm,
                              index=main_class_names,
                              columns=main_class_names)
    main_cm_df.to_csv(os.path.join(exp_dir, 'main_confusion_matrix.csv'))

    if valid_sub_indices:
        unique_sub = np.unique(valid_sub_true)
        sub_class_names = label_encoders['sub'].inverse_transform(unique_sub)

        sub_cm = confusion_matrix(valid_sub_true, valid_sub_preds, labels=unique_sub)
        sub_cm_df = pd.DataFrame(sub_cm,
                                 index=sub_class_names,
                                 columns=sub_class_names)
        sub_cm_df.to_csv(os.path.join(exp_dir, 'sub_confusion_matrix.csv'))

    # 长尾类别评估
    evaluate_tail_classes(all_main_true, all_main_preds,
                          all_sub_true, all_sub_preds,
                          label_encoders, exp_dir)

    return metrics


def evaluate_tail_classes(true_main, pred_main, true_sub, pred_sub, label_encoders, exp_dir):
    """评估长尾类别表现"""
    # 统计每个类别的样本数
    main_class_counts = {int(cls): int(count) for cls, count in zip(*np.unique(true_main, return_counts=True))}
    sub_class_counts = {int(cls): int(count) for cls, count in zip(*np.unique(true_sub, return_counts=True))}

    # 识别长尾类别（样本数少于阈值）
    tail_main_classes = [int(cls) for cls, count in main_class_counts.items()
                         if count < CONFIG['tail_threshold'] and cls != -1]

    tail_sub_classes = [int(cls) for cls, count in sub_class_counts.items()
                        if count < CONFIG['tail_threshold'] and cls != -1]

    # 计算每个类别的指标（主类别）
    main_metrics = {}
    unique_main = np.unique(true_main)
    for cls in unique_main:
        # 只计算长尾类别
        if int(cls) not in tail_main_classes:
            continue

        # 创建二分类标签：当前类别为1，其他类别为0
        binary_true = np.array([1 if x == cls else 0 for x in true_main])
        binary_pred = np.array([1 if x == cls else 0 for x in pred_main])

        # 计算指标
        precision = precision_score(binary_true, binary_pred, zero_division=0)
        recall = recall_score(binary_true, binary_pred, zero_division=0)
        f1 = f1_score(binary_true, binary_pred, zero_division=0)

        main_metrics[int(cls)] = {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'support': int(main_class_counts[int(cls)])
        }

    # 计算每个类别的指标（子类别）
    sub_metrics = {}
    unique_sub = np.unique(true_sub)
    for cls in unique_sub:
        # 只计算长尾类别
        if cls == -1 or int(cls) not in tail_sub_classes:
            continue

        # 创建二分类标签：当前类别为1，其他类别为0
        binary_true = np.array([1 if x == cls else 0 for x in true_sub])
        binary_pred = np.array([1 if x == cls else 0 for x in pred_sub])

        # 计算指标
        precision = precision_score(binary_true, binary_pred, zero_division=0)
        recall = recall_score(binary_true, binary_pred, zero_division=0)
        f1 = f1_score(binary_true, binary_pred, zero_division=0)

        sub_metrics[int(cls)] = {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'support': int(sub_class_counts[int(cls)])
        }

    # 保存结果
    tail_main_results = []
    for cls, metrics in main_metrics.items():
        tail_main_results.append({
            'class': label_encoders['main'].inverse_transform([cls])[0],
            'class_id': cls,
            'support': metrics['support'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1']
        })

    tail_sub_results = []
    for cls, metrics in sub_metrics.items():
        tail_sub_results.append({
            'class': label_encoders['sub'].inverse_transform([cls])[0],
            'class_id': cls,
            'support': metrics['support'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1']
        })

    # 保存结果
    tail_results = {
        'main_tail_classes': tail_main_results,
        'sub_tail_classes': tail_sub_results
    }

    with open(os.path.join(exp_dir, 'tail_class_metrics.json'), 'w') as f:
        json.dump(tail_results, f, indent=2)

    # 可视化长尾类别召回率
    if tail_main_results:
        plot_tail_recall(tail_main_results, os.path.join(exp_dir, 'main_tail_recall.png'), "Main Classes")
    if tail_sub_results:
        plot_tail_recall(tail_sub_results, os.path.join(exp_dir, 'sub_tail_recall.png'), "Sub Classes")

    return tail_results


def plot_tail_recall(tail_results, save_path, title):
    """绘制长尾类别召回率图"""
    classes = [res['class'] for res in tail_results]
    recall = [res['recall'] for res in tail_results]
    support = [res['support'] for res in tail_results]

    # 按召回率排序
    sorted_indices = np.argsort(recall)
    classes = [classes[i] for i in sorted_indices]
    recall = [recall[i] for i in sorted_indices]
    support = [support[i] for i in sorted_indices]

    plt.figure(figsize=(12, 8))
    bars = plt.barh(classes, recall, color='skyblue')
    plt.xlabel('Recall')
    plt.title(f'Tail Class Recall: {title}')
    plt.xlim(0, 1)

    # 添加样本数量标签
    for i, bar in enumerate(bars):
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                 f'n={support[i]}',
                 ha='left', va='center')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# 主函数
def main():
    # 准备数据
    df = pd.read_excel("data/sample_multi_grading.xlsx")
    df["schemaCol"] = df["schemaCol"].apply(clean_text)

    # 填充子类别的NaN值
    df['categoryNameIndex_C4_sub'].fillna("None", inplace=True)

    # 标签编码
    main_le = LabelEncoder()
    sub_le = LabelEncoder()
    df['main_label'] = main_le.fit_transform(df['categoryNameIndex_C4_main'])
    df['sub_label'] = sub_le.fit_transform(df['categoryNameIndex_C4_sub'])

    # 保存标签编码器
    label_encoders = {
        'main': main_le,
        'sub': sub_le
    }

    # 构建层级映射
    hierarchy = defaultdict(list)
    for _, row in df.iterrows():
        if row['categoryNameIndex_C4_sub'] != "None":  # 跳过无子类的样本
            hierarchy[row['main_label']].append(row['sub_label'])
    hierarchy = {k: sorted(list(set(v))) for k, v in hierarchy.items()}

    # 打印层级信息
    logger.info("Hierarchy mapping:")
    for main_id, sub_ids in hierarchy.items():
        logger.info(f"Main class {main_id} has {len(sub_ids)} subclasses: {sub_ids}")

    # 检查类别分布并移除样本不足的类别
    main_class_counts = df['main_label'].value_counts()
    min_samples = 2  # 每个类别至少需要2个样本

    # 找出样本不足的类别
    poor_classes = main_class_counts[main_class_counts < min_samples].index.tolist()
    logger.warning(
        f"Removing underpopulated classes: {poor_classes} with counts: {main_class_counts[poor_classes].tolist()}")

    # 过滤掉样本不足的类别
    df_filtered = df[~df['main_label'].isin(poor_classes)]
    logger.info(f"Original dataset size: {len(df)}, Filtered dataset size: {len(df_filtered)}")

    # 使用分层抽样分割数据集（按主类别）
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    for train_index, temp_index in sss.split(df_filtered, df_filtered['main_label']):
        train_df = df_filtered.iloc[train_index]
        temp_df = df_filtered.iloc[temp_index]

    # 检查临时集中是否有类别样本数不足2
    temp_main_class_counts = temp_df['main_label'].value_counts()
    poor_temp_classes = temp_main_class_counts[temp_main_class_counts < 2].index.tolist()
    if poor_temp_classes:
        logger.warning(
            f"Removing underpopulated classes in temp set: {poor_temp_classes} with counts: {temp_main_class_counts[poor_temp_classes].tolist()}")
        # 将这些类别从临时集中移除
        temp_df = temp_df[~temp_df['main_label'].isin(poor_temp_classes)]
        # 同时从训练集中移除这些类别以保持一致性
        train_df = train_df[~train_df['main_label'].isin(poor_temp_classes)]
        logger.info(f"Temp set size after removal: {len(temp_df)}, Train set size after removal: {len(train_df)}")

    # 然后进行第二次分层抽样
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    # 确保临时集中每个类别至少有2个样本
    if len(temp_df) > 0:
        for val_index, test_index in sss2.split(temp_df, temp_df['main_label']):
            val_df = temp_df.iloc[val_index]
            test_df = temp_df.iloc[test_index]
    else:
        logger.error("Temporary set is empty after removing underpopulated classes!")
        val_df = pd.DataFrame()
        test_df = pd.DataFrame()

    # 创建测试集数据加载器（用于最终评估）
    tokenizer = BertTokenizer.from_pretrained(CONFIG['model_path'])

    test_dataset = OptimizedHierarchyDataset(
        test_df["schemaCol"].values,
        test_df["main_label"].values,
        test_df["sub_label"].values,
        tokenizer,
        CONFIG['max_len']
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=CONFIG['num_workers'],
        pin_memory=CONFIG['pin_memory']
    )

    # 添加文本到测试集数据加载器（用于结果分析）
    test_loader.dataset.texts = test_df["schemaCol"].values.tolist()

    # 运行所有实验
    all_results = {}

    for exp_config in EXPERIMENTS:
        exp_name = exp_config['name']
        logger.info(f"\n{'=' * 50}")
        logger.info(f"Starting experiment: {exp_name}")
        logger.info(f"Config: {exp_config}")
        logger.info(f"{'=' * 50}")

        # 创建实验目录
        exp_dir = os.path.join(CONFIG['results_dir'], exp_name)
        os.makedirs(exp_dir, exist_ok=True)

        # 保存实验配置
        with open(os.path.join(exp_dir, 'config.json'), 'w') as f:
            json.dump(exp_config, f, indent=2)

        # 创建数据增强器
        augmenter = create_text_augmenter() if exp_config['use_aug'] else None

        # 创建数据集
        train_dataset = OptimizedHierarchyDataset(
            train_df["schemaCol"].values,
            train_df["main_label"].values,
            train_df["sub_label"].values,
            tokenizer,
            CONFIG['max_len'],
            augmenter=augmenter
        )

        val_dataset = OptimizedHierarchyDataset(
            val_df["schemaCol"].values,
            val_df["main_label"].values,
            val_df["sub_label"].values,
            tokenizer,
            CONFIG['max_len']
        )

        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=CONFIG['batch_size'],
            shuffle=True,
            num_workers=CONFIG['num_workers'],
            pin_memory=CONFIG['pin_memory']
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=CONFIG['batch_size'],
            shuffle=False,
            num_workers=CONFIG['num_workers'],
            pin_memory=CONFIG['pin_memory']
        )

        # 初始化模型
        if exp_config['model_type'] == 'hierarchical':
            model = FixedHierarchicalBERT(
                num_main=len(main_le.classes_),
                num_sub=len(sub_le.classes_),
                model_path=CONFIG['model_path'],
                hierarchy_map=hierarchy,
                use_gru=exp_config['use_gru'],
                use_shared=exp_config['use_shared']
            ).to(CONFIG['device'])
        else:  # flat_bert
            model = FlatBERT(
                num_classes=len(main_le.classes_),  # 平铺模型只预测主类
                model_path=CONFIG['model_path']
            ).to(CONFIG['device'])

        # 打印模型结构
        logger.info(f"Model architecture for {exp_name}:")
        logger.info(model)

        # 优化器
        optimizer = AdamW(model.parameters(), lr=CONFIG['lr'])

        # 学习率调度器
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=CONFIG['warmup_steps'],
            num_training_steps=len(train_loader) * CONFIG['epochs']
        ) if exp_config['model_type'] == 'hierarchical' else None

        # 训练模型
        trained_model = fixed_train_model(
            model,
            train_loader,
            val_loader,
            optimizer,
            CONFIG['device'],
            CONFIG['epochs'],
            num_main=len(main_le.classes_),
            num_sub=len(sub_le.classes_),
            scheduler=scheduler,
            experiment_name=exp_name
        )

        # 保存模型
        model_path = os.path.join(exp_dir, 'model.bin')
        torch.save(trained_model.state_dict(), model_path)
        logger.info(f"Model saved to {model_path}")

        # 在测试集上评估模型
        test_metrics = evaluate_model(
            trained_model,
            test_loader,
            CONFIG['device'],
            label_encoders,
            exp_name,
            is_flat=(exp_config['model_type'] == 'flat')
        )

        # 保存测试结果
        all_results[exp_name] = test_metrics
        logger.info(f"\nTest metrics for {exp_name}:")
        logger.info(f"Main Accuracy: {test_metrics['main_metrics']['accuracy']:.4f}")
        logger.info(f"Main F1 Macro: {test_metrics['main_metrics']['f1_macro']:.4f}")
        if test_metrics['sub_metrics']['accuracy'] > 0:
            logger.info(f"Sub Accuracy: {test_metrics['sub_metrics']['accuracy']:.4f}")
            logger.info(f"Sub F1 Macro: {test_metrics['sub_metrics']['f1_macro']:.4f}")

    # 保存所有实验结果
    with open(os.path.join(CONFIG['results_dir'], 'all_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)

    # 打印实验总结
    logger.info("\nExperiment Summary:")
    for exp_name, metrics in all_results.items():
        logger.info(f"{exp_name}:")
        logger.info(f"  Main Acc: {metrics['main_metrics']['accuracy']:.4f}")
        logger.info(f"  Main F1 Macro: {metrics['main_metrics']['f1_macro']:.4f}")
        if metrics['sub_metrics']['accuracy'] > 0:
            logger.info(f"  Sub Acc: {metrics['sub_metrics']['accuracy']:.4f}")
            logger.info(f"  Sub F1 Macro: {metrics['sub_metrics']['f1_macro']:.4f}")

    logger.info("All experiments completed!")


if __name__ == "__main__":
    main()