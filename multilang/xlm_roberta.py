import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import pandas as pd
import torch
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import os
import glob
from transformers import get_linear_schedule_with_warmup

# ==================== 改进的训练模块 ====================

class StableTrainer:
    def __init__(self, model, tokenizer, train_dataset, val_dataset, 
                 learning_rate=1e-5, batch_size=4, accumulation_steps=4):
        """
        参数说明：
        - learning_rate: 降低学习率，XLM-RoBERTa-large需要更小的学习率
        - batch_size: 减小批次大小，提高稳定性
        - accumulation_steps: 梯度累积步数，模拟更大的batch size
        """
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.accumulation_steps = accumulation_steps
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        self.model.to(self.device)
        
        # 优化器 - 使用更小的学习率
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=learning_rate,
            weight_decay=0.01,  # 增加权重衰减
            eps=1e-8
        )
        
        # 学习率调度器 - 线性预热
        total_steps = len(train_dataset) * 3 // (batch_size * accumulation_steps)
        warmup_steps = int(0.1 * total_steps)  # 10%的warmup
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # 记录训练历史
        self.train_losses = []
        self.val_f1_scores = []
        
    def train(self, epochs=3):
        print("开始训练...")
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            train_loader = torch.utils.data.DataLoader(
                self.train_dataset, batch_size=self.batch_size, shuffle=True
            )
            
            # 梯度累积计数器
            accumulation_counter = 0
            self.optimizer.zero_grad()
            
            for batch_idx, batch in enumerate(train_loader):
                # 移动数据到设备
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # 前向传播
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                # 梯度累积：将损失除以累积步数
                loss = outputs.loss / self.accumulation_steps
                loss.backward()
                
                total_loss += outputs.loss.item()
                
                # 梯度累积
                accumulation_counter += 1
                if accumulation_counter % self.accumulation_steps == 0:
                    # 梯度裁剪 - 防止梯度爆炸
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    accumulation_counter = 0
                
                # 更频繁地打印训练信息
                if batch_idx % 10 == 0:  # 每10个batch打印一次
                    current_lr = self.optimizer.param_groups[0]['lr']
                    print(f'Epoch: {epoch+1}/{epochs}, '
                          f'Batch: {batch_idx}/{len(train_loader)}, '
                          f'Loss: {outputs.loss.item():.4f}, '
                          f'LR: {current_lr:.2e}')
            
            # 处理剩余的梯度
            if accumulation_counter > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            avg_loss = total_loss / len(train_loader)
            self.train_losses.append(avg_loss)
            
            print(f'\nEpoch {epoch+1} 完成, 平均损失: {avg_loss:.4f}')
            
            # 每个epoch后验证
            eval_results = self.evaluate()
            self.val_f1_scores.append(eval_results['f1_macro'])
            
            print(f'Epoch {epoch+1} 验证结果: F1-Macro={eval_results["f1_macro"]:.4f}, '
                  f'F1-Micro={eval_results["f1_micro"]:.4f}')
            
            # 保存最佳模型
            if eval_results['f1_macro'] > getattr(self, 'best_f1', 0):
                self.best_f1 = eval_results['f1_macro']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'f1_score': self.best_f1,
                }, 'best_model.pth')
                print(f"✅ 保存最佳模型，F1: {self.best_f1:.4f}")
    
    def evaluate(self):
        self.model.eval()
        val_loader = torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.batch_size * 2  # 验证时可以使用更大的batch
        )
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                probs = torch.sigmoid(outputs.logits)
                preds = (probs > 0.5).int().cpu().numpy()
                labels_np = labels.cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(labels_np)
        
        f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        f1_micro = f1_score(all_labels, all_preds, average='micro', zero_division=0)
        
        return {'f1_macro': f1_macro, 'f1_micro': f1_micro}
    
    def predict(self, test_dataset):
        self.model.eval()
        test_loader = torch.utils.data.DataLoader(
            test_dataset, 
            batch_size=self.batch_size * 2  # 预测时也可以使用更大的batch
        )
        
        all_preds = []
        all_probs = []
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                probs = torch.sigmoid(outputs.logits)
                preds = (probs > 0.5).int().cpu().numpy()
                
                all_preds.extend(preds)
                all_probs.extend(probs.cpu().numpy())
        
        return np.array(all_preds), np.array(all_probs)

# ==================== 改进的数据集类 ====================

class ImprovedPolarizationDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):  # 增加max_length
        """
        改进的数据集类，增加文本长度限制和更好的预处理
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])  # 确保文本是字符串
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text, 
            truncation=True, 
            padding='max_length',
            max_length=self.max_length, 
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.float)
        }

# ==================== 数据检查函数 ====================

def check_data_distribution(df, label_columns):
    """检查数据分布，帮助理解数据不平衡问题"""
    print(f"\n数据分布检查:")
    print(f"总样本数: {len(df)}")
    
    for col in label_columns:
        if col in df.columns:
            positive_count = df[col].sum()
            percentage = (positive_count / len(df)) * 100
            print(f"  {col}: {positive_count} 正样本 ({percentage:.2f}%)")
    
    # 检查多标签情况
    df['num_labels'] = df[label_columns].sum(axis=1)
    print(f"\n多标签分布:")
    for i in range(6):
        count = (df['num_labels'] == i).sum()
        print(f"  {i}个标签: {count} 样本 ({count/len(df)*100:.2f}%)")

# ==================== 预测函数 ====================

def predict_multiple_datasets(trainer, tokenizer, test_files, label_columns, output_dir='./data/'):
    """对多个测试数据集进行预测"""
    
    os.makedirs(output_dir, exist_ok=True)
    all_predictions = {}
    
    for dataset_name, file_path in test_files.items():
        print(f"\n正在处理测试集: {dataset_name}")
        
        try:
            # 读取测试数据
            test_df = pd.read_csv(file_path)
            print(f"成功读取 {len(test_df)} 条测试数据")
            
            # 数据质量检查
            print(f"数据检查:")
            print(f"  文本缺失: {test_df['text'].isnull().sum()}")
            
            # 创建测试数据集
            test_dataset = ImprovedPolarizationDataset(
                test_df['text'].fillna('').tolist(),
                np.zeros((len(test_df), len(label_columns))),
                tokenizer
            )
            
            # 进行预测
            print(f"开始对 {dataset_name} 进行预测...")
            test_preds, test_probs = trainer.predict(test_dataset)
            
            # 创建结果DataFrame
            results_df = test_df[['id']].copy() if 'id' in test_df.columns else pd.DataFrame(index=range(len(test_df)))
            
            # 添加预测结果
            for i, col in enumerate(label_columns):
                results_df[col] = test_preds[:, i].astype(int)  # 确保是整数
            
            # 保存结果
            all_predictions[dataset_name] = results_df
            
            output_filename = f"pred_{dataset_name}.csv"
            output_path = os.path.join(output_dir, output_filename)
            results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"✅ 预测结果已保存到: {output_path}")
            
            # 显示预测统计
            print(f"预测标签统计:")
            for col in label_columns:
                count = int(results_df[col].sum())
                percentage = (count / len(results_df)) * 100
                print(f"  {col}: {count} 条 ({percentage:.1f}%)")
                
        except Exception as e:
            print(f"❌ 处理 {dataset_name} 测试集时出错: {e}")
            continue
    
    return all_predictions

# ==================== 主流程 ====================

def main():
    # 安装必要的库
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
    except ImportError:
        print("安装transformers库...")
        os.system("pip install transformers")
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
    
    # 训练集文件映射
    train_files = {
        'eng': 'data/train_eng.csv',
        'amh': 'data/train_amh.csv',
        'arb': 'data/train_arb.csv',
        'deu': 'data/train_deu.csv',
        'fas': 'data/train_fas.csv',
        'hau': 'data/train_hau.csv',
        'hin': 'data/train_hin.csv',
        'ita': 'data/train_ita.csv',
        'nep': 'data/train_nep.csv',
        'spa': 'data/train_spa.csv',
        'tur': 'data/train_tur.csv',
        'khm': 'data/train_khm.csv',
        'ben': 'data/train_ben.csv',
        'mya': 'data/train_mya.csv',
        'ori': 'data/train_ori.csv',
        'pan': 'data/train_pan.csv',
        'pol': 'data/train_pol.csv',
        'rus': 'data/train_rus.csv',
        'swa': 'data/train_swa.csv',
        'tel': 'data/train_tel.csv',
        'urd': 'data/train_urd.csv',
        'zho': 'data/train_zho.csv'

    }
    
    # 测试集文件映射
    test_files = {
        'zho': 'data/dev_zho.csv',
        'eng': 'data/dev_eng.csv',
        'arb': 'data/dev_arb.csv',
        'deu': 'data/dev_deu.csv',
        'fas': 'data/dev_fas.csv',
        'hin': 'data/dev_hin.csv',
        'spa': 'data/dev_spa.csv',
        'hau': 'data/dev_hau.csv',
        'amh': 'data/dev_amh.csv',
        'ita': 'data/dev_ita.csv',
        'nep': 'data/dev_nep.csv',
        'tur': 'data/dev_tur.csv',
        'urd': 'data/dev_urd.csv',
        'khm': 'data/dev_khm.csv',
        'ben': 'data/dev_ben.csv',
        'mya': 'data/dev_mya.csv',
        'ori': 'data/dev_ori.csv',
        'pan': 'data/dev_pan.csv',
        'pol': 'data/dev_pol.csv',
        'rus': 'data/dev_rus.csv',
        'swa': 'data/dev_swa.csv',
        'tel': 'data/dev_tel.csv'
    }
    
    # 加载数据
    all_data = []
    print("开始加载训练数据...")
    
    for lang, file_path in train_files.items():
        try:
            df = pd.read_csv(file_path)
            print(f"✅ {lang}: {len(df)} 条数据")
            all_data.append(df)
        except Exception as e:
            print(f"❌ {lang}: 加载失败 - {e}")
            continue
    
    if not all_data:
        print("没有成功加载任何训练数据！")
        return
    
    # 合并数据
    combined_train = pd.concat(all_data, ignore_index=True)
    print(f"\n合并后总数据量: {len(combined_train)} 条")
    
    # 定义标签列
    label_columns = ['political', 'racial/ethnic', 'religious', 'gender/sexual', 'other']
    
    # 数据质量检查
    check_data_distribution(combined_train, label_columns)
    
    # 数据清洗
    combined_train = combined_train.dropna(subset=['text'])  # 移除空文本
    print(f"\n清洗后数据量: {len(combined_train)} 条")
    
    # 分割数据集
    train, val = train_test_split(combined_train, test_size=0.2, random_state=42)
    
    print(f"\n数据集大小:")
    print(f"训练集: {len(train)}")
    print(f"验证集: {len(val)}")
    
    # 加载模型和tokenizer
    print("\n加载模型和tokenizer...")
    model_name = "xlm-roberta-large"
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # 重要：正确配置多标签分类
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(label_columns),
            problem_type="multi_label_classification"
        )
        
        print(f"✅ 成功加载模型: {model_name}")
        
    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        return
    
    # 创建数据集
    train_dataset = ImprovedPolarizationDataset(
        train['text'].tolist(),
        train[label_columns].values.astype(float).tolist(),
        tokenizer,
        max_length=256  # 增加文本长度
    )
    
    val_dataset = ImprovedPolarizationDataset(
        val['text'].tolist(),
        val[label_columns].values.astype(float).tolist(),
        tokenizer,
        max_length=256
    )
    
    print(f"\n数据集示例:")
    sample = train_dataset[0]
    print(f"输入ID形状: {sample['input_ids'].shape}")
    print(f"注意力掩码形状: {sample['attention_mask'].shape}")
    print(f"标签: {sample['labels']}")
    
    # 训练模型 - 使用更稳定的配置
    print("\n开始训练模型...")
    trainer = StableTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        learning_rate=2e-5,  # 更小的学习率
        batch_size=8,        # 更小的批次大小
        accumulation_steps=8 # 梯度累积
    )
    
    trainer.train(epochs=3)  
    
    # 对测试集进行预测
    print("\n开始对测试集进行预测...")
    all_predictions = predict_multiple_datasets(trainer, tokenizer, test_files, label_columns)
    
    # 生成汇总报告
    print("\n生成汇总报告...")
    summary_data = []
    for dataset_name, pred_df in all_predictions.items():
        row = {'数据集': dataset_name, '样本数量': len(pred_df)}
        for col in label_columns:
            if col in pred_df.columns:
                count = int(pred_df[col].sum())
                percentage = (count / len(pred_df)) * 100
                row[col] = f"{count} ({percentage:.1f}%)"
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df)
    
    # 保存汇总报告
    summary_file = 'predictions_summary.csv'
    summary_df.to_csv(summary_file, index=False, encoding='utf-8-sig')
    print(f"\n汇总报告保存到: {summary_file}")

if __name__ == "__main__":
    main()