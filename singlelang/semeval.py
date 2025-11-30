import pandas as pd
import torch
import numpy as np
from sklearn.metrics import f1_score, classification_report
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import wandb
import jieba
import random

# Disable wandb logging for this script
wandb.init(mode="disabled")

# 读取数据
train = pd.read_csv('/kaggle/input/semeval/train_zho.csv')
train, val = train_test_split(train, test_size=.2)
# 读取测试集
test = pd.read_csv('/kaggle/input/semeval/dev_zho.csv')

# 定义新的列顺序
new_column_order = ['id', 'political', 'racial/ethnic', 'religious', 'gender/sexual', 'other']

# 调整列的顺序
def reorder_columns(df, new_order):
    existing_columns = [col for col in new_order if col in df.columns]
    other_columns = [col for col in df.columns if col not in new_order]
    final_order = existing_columns + other_columns
    return df[final_order]

train = reorder_columns(train, new_column_order)
val = reorder_columns(val, new_column_order)
test = reorder_columns(test, new_column_order)

print(f"训练集大小: {len(train)}")
print(f"验证集大小: {len(val)}")
print(f"测试集大小: {len(test)}")

# ==================== 简化的数据增强方法 ====================

class DataAugmenter:
    def __init__(self):
        pass
    
    def synonym_replacement(self, text, n=2):
        """同义词替换 - 使用jieba分词和简单的同义词替换"""
        words = list(jieba.cut(text))
        if len(words) <= 1:
            return text
            
        # 简单的同义词替换（这里使用一个简单的同义词字典，实际应用中可以使用更复杂的词典）
        synonym_dict = {
            '好': ['佳', '棒', '优秀', '出色'],
            '坏': ['差', '糟糕', '不好', '劣质'],
            '大': ['巨大', '庞大', '宏大', '广大'],
            '小': ['微小', '细小', '狭小', '迷你'],
            '说': ['讲', '道', '表示', '声称'],
            '看': ['瞧', '观', '注视', '瞥见'],
            '走': ['行', '步', '移动', '前进'],
            '快': ['迅速', '急速', '飞快', '快速'],
            '慢': ['缓慢', '迟缓', '悠悠', '慢吞吞']
        }
        
        new_words = words.copy()
        replaced = 0
        for i, word in enumerate(new_words):
            if word in synonym_dict and random.random() < 0.3 and replaced < n:
                new_words[i] = random.choice(synonym_dict[word])
                replaced += 1
                
        return ''.join(new_words)
    
    def random_swap(self, text, n=2):
        """随机交换词语"""
        words = list(jieba.cut(text))
        if len(words) <= 1:
            return text
            
        new_words = words.copy()
        for _ in range(min(n, len(words)//2)):
            idx1, idx2 = random.sample(range(len(words)), 2)
            new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
            
        return ''.join(new_words)
    
    def random_deletion(self, text, p=0.1):
        """随机删除词语"""
        words = list(jieba.cut(text))
        if len(words) <= 1:
            return text
            
        new_words = []
        for word in words:
            if random.random() > p:
                new_words.append(word)
                
        if len(new_words) == 0:
            return words[random.randint(0, len(words)-1)]
            
        return ''.join(new_words)
    
    def random_insertion(self, text, n=1):
        """随机插入词语"""
        words = list(jieba.cut(text))
        if len(words) == 0:
            return text
            
        new_words = words.copy()
        for _ in range(n):
            random_word = random.choice(words)
            random_idx = random.randint(0, len(new_words))
            new_words.insert(random_idx, random_word)
            
        return ''.join(new_words)

def augment_dataset(df, augmenter, num_augmentations=2):
    """对数据集进行增强"""
    augmented_data = []
    
    for _, row in df.iterrows():
        text = row['text']
        labels = row[['political', 'racial/ethnic', 'religious', 'gender/sexual', 'other']].values
        
        # 原始数据
        augmented_data.append({
            'text': text,
            'political': labels[0],
            'racial/ethnic': labels[1],
            'religious': labels[2],
            'gender/sexual': labels[3],
            'other': labels[4]
        })
        
        # 生成增强数据
        for i in range(num_augmentations):
            # 随机选择一种增强方法
            aug_method = random.choice([
                'synonym_replacement',
                'random_swap',
                'random_deletion',
                'random_insertion'
            ])
            
            if aug_method == 'synonym_replacement':
                augmented_text = augmenter.synonym_replacement(text)
            elif aug_method == 'random_swap':
                augmented_text = augmenter.random_swap(text)
            elif aug_method == 'random_deletion':
                augmented_text = augmenter.random_deletion(text)
            elif aug_method == 'random_insertion':
                augmented_text = augmenter.random_insertion(text)
            
            # 确保增强后的文本与原始文本不同且不为空
            if augmented_text != text and len(augmented_text.strip()) > 0:
                augmented_data.append({
                    'text': augmented_text,
                    'political': labels[0],
                    'racial/ethnic': labels[1],
                    'religious': labels[2],
                    'gender/sexual': labels[3],
                    'other': labels[4]
                })
    
    return pd.DataFrame(augmented_data)

# 应用数据增强
print("开始数据增强...")
augmenter = DataAugmenter()
train_augmented = augment_dataset(train, augmenter, num_augmentations=2)
print(f"增强后训练集大小: {len(train_augmented)}")

# 数据集类
class PolarizationDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, truncation=True, padding=False, max_length=self.max_length, return_tensors='pt')

        item = {key: encoding[key].squeeze() for key in encoding.keys()}
        item['labels'] = torch.tensor(label, dtype=torch.float)
        return item

# 加载中文tokenizer
tokenizer = AutoTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')

# 创建数据集
label_columns = ['political', 'racial/ethnic', 'religious', 'gender/sexual', 'other']
train_dataset = PolarizationDataset(
    train_augmented['text'].tolist(), 
    train_augmented[label_columns].values.tolist(), 
    tokenizer
)
val_dataset = PolarizationDataset(
    val['text'].tolist(), 
    val[label_columns].values.tolist(), 
    tokenizer
)

# 为测试集创建数据集（使用虚拟标签）
test_labels = np.zeros((len(test), 5))
test_dataset = PolarizationDataset(
    test['text'].tolist(), 
    test_labels.tolist(),
    tokenizer
)

# 加载模型
model = AutoModelForSequenceClassification.from_pretrained(
    'hfl/chinese-roberta-wwm-ext',
    num_labels=5, 
    problem_type="multi_label_classification"
)

# 定义评估指标
def compute_metrics_multilabel(p):
    probs = torch.sigmoid(torch.from_numpy(p.predictions))
    preds = (probs > 0.5).int().numpy()
    return {'f1_macro': f1_score(p.label_ids, preds, average='macro')}

# 训练参数
training_args = TrainingArguments(
    output_dir=f"./",
    num_train_epochs=5,
    learning_rate=1e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    eval_strategy="epoch",
    save_strategy="no",
    logging_steps=100,
    disable_tqdm=False,
    load_best_model_at_end=False,
    metric_for_best_model="eval_f1_macro",
    greater_is_better=True
)

# 初始化Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics_multilabel,
    data_collator=DataCollatorWithPadding(tokenizer)
)

# 训练模型
print("开始训练模型...")
trainer.train()

# 评估模型
print("评估模型...")
eval_results = trainer.evaluate()
print(f"验证集上的Macro F1分数: {eval_results['eval_f1_macro']:.4f}")

# 对测试集进行预测
print("\n开始对测试集进行预测...")
test_predictions = trainer.predict(test_dataset)

# 将预测概率转换为二进制标签
test_probs = torch.sigmoid(torch.from_numpy(test_predictions.predictions)).numpy()
test_preds = (test_probs > 0.5).astype(int)

# 创建包含预测结果的DataFrame
results_df = test[['id']].copy()

# 添加预测的标签
for i, col in enumerate(label_columns):
    results_df[col] = test_preds[:, i]

# 显示一些预测样本
print("\n测试集预测结果示例:")
print(results_df.head(10))

# 保存预测结果到CSV文件
output_file = '/kaggle/working/pred_zho.csv'
results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"\n预测结果已保存到: {output_file}")