import pandas as pd
import torch
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import os
import glob

# ==================== 简化的训练模块 ====================

class SimpleTrainer:
    def __init__(self, model, tokenizer, train_dataset, val_dataset, learning_rate=2e-5, batch_size=8):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        
    def train(self, epochs=3):
        print("开始训练...")
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            train_loader = torch.utils.data.DataLoader(
                self.train_dataset, batch_size=self.batch_size, shuffle=True
            )
            
            for batch_idx, batch in enumerate(train_loader):
                # 移动数据到设备
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                self.optimizer.zero_grad()
                
                # 前向传播
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                
                if batch_idx % 20 == 0:
                    print(f'Epoch: {epoch+1}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
            
            avg_loss = total_loss / len(train_loader)
            print(f'Epoch {epoch+1} 完成, 平均损失: {avg_loss:.4f}')
            
            # 每个epoch后验证
            eval_results = self.evaluate()
            print(f'Epoch {epoch+1} 验证结果: {eval_results}')
    
    def evaluate(self):
        self.model.eval()
        val_loader = torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.batch_size
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
        
        f1_macro = f1_score(all_labels, all_preds, average='macro')
        f1_micro = f1_score(all_labels, all_preds, average='micro')
        
        return {'f1_macro': f1_macro, 'f1_micro': f1_micro}
    
    def predict(self, test_dataset):
        self.model.eval()
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size)
        
        all_preds = []
        
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
        
        return np.array(all_preds)

# ==================== 数据集类 ====================

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

# ==================== 预测多个数据集的函数 ====================

def predict_multiple_datasets(trainer, tokenizer, test_files, label_columns, output_dir='/kaggle/working/'):
    """对多个测试数据集进行预测，每个数据集输出单独的CSV文件"""
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    all_predictions = {}
    
    for dataset_name, file_path in test_files.items():
        print(f"\n正在处理测试集: {dataset_name}")
        print(f"文件路径: {file_path}")
        
        try:
            # 读取测试数据
            test_df = pd.read_csv(file_path)
            print(f"成功读取 {len(test_df)} 条测试数据")
            
            # 数据预处理 - 确保列顺序正确
            new_column_order = ['id', 'political', 'racial/ethnic', 'religious', 'gender/sexual', 'other']
            
            def reorder_columns(df, new_order):
                existing = [col for col in new_order if col in df.columns]
                others = [col for col in df.columns if col not in new_order]
                return df[existing + others]
            
            test_df = reorder_columns(test_df, new_column_order)
            
            # 创建测试数据集
            test_dataset = PolarizationDataset(
                test_df['text'].tolist(),
                np.zeros((len(test_df), 5)),  # 测试集没有真实标签，用0填充
                tokenizer
            )
            
            # 进行预测
            print(f"开始对 {dataset_name} 进行预测...")
            test_preds = trainer.predict(test_dataset)
            
            # 创建结果DataFrame
            results_df = test_df[['id']].copy()
            for i, col in enumerate(label_columns):
                results_df[col] = test_preds[:, i]
            
            # 保存到字典中
            all_predictions[dataset_name] = results_df
            
            # 为每个数据集生成唯一的输出文件名
            output_filename = f"predictions_{dataset_name}.csv"
            output_path = os.path.join(output_dir, output_filename)
            
            # 保存到独立的CSV文件
            results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"✅ 预测结果已保存到: {output_path}")
            
            # 显示样本结果和统计
            print(f"\n{dataset_name} 预测结果示例:")
            print(results_df.head(3))
            
            print(f"\n{dataset_name} 预测标签统计:")
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
        'eng': '/kaggle/input/semeval/train_eng.csv',
        'amh': '/kaggle/input/semeval/train_amh.csv',
        'arb': '/kaggle/input/semeval/train_arb.csv',
        'deu': '/kaggle/input/semeval/train_deu.csv',
        'fas': '/kaggle/input/semeval/train_fas.csv',
        'hau': '/kaggle/input/semeval/train_hau.csv',
        'hin': '/kaggle/input/semeval/train_hin.csv',
        'ita': '/kaggle/input/semeval/train_ita.csv',
        'nep': '/kaggle/input/semeval/train_nep.csv',
        'spa': '/kaggle/input/semeval/train_spa.csv',
        'tur': '/kaggle/input/semeval/train_tur.csv',
        'urd': '/kaggle/input/semeval/train_urd.csv',
        'zho': '/kaggle/input/semeval/train_zho.csv'
    }
    
    # 测试集文件映射 - 可以添加多个测试集
    test_files = {
        'zho': '/kaggle/input/semeval/dev_zho.csv',
        'eng': '/kaggle/input/semeval/dev_eng.csv',
        'arb': '/kaggle/input/semeval/dev_arb.csv',
        'deu': '/kaggle/input/semeval/dev_deu.csv',
        'fas': '/kaggle/input/semeval/dev_fas.csv',
        'hin': '/kaggle/input/semeval/dev_hin.csv',
        'spa': '/kaggle/input/semeval/dev_spa.csv',
        'hau': '/kaggle/input/semeval/dev_hau.csv',
        'amh': '/kaggle/input/semeval/dev_amh.csv',
        'ita': '/kaggle/input/semeval/dev_ita.csv',
        'nep': '/kaggle/input/semeval/dev_nep.csv',
        'tur': '/kaggle/input/semeval/dev_tur.csv',
        'urd': '/kaggle/input/semeval/dev_urd.csv',
        # 可以根据需要添加更多测试集
    }
    
    # 自动检测测试集文件（可选）
    def auto_detect_test_files(base_path='/kaggle/input/semeval/'):
        """自动检测测试集文件"""
        detected_files = {}
        
        # 查找所有dev_*.csv文件
        dev_files = glob.glob(os.path.join(base_path, 'dev_*.csv'))
        for file_path in dev_files:
            # 提取数据集名称
            filename = os.path.basename(file_path)
            dataset_name = filename.replace('.csv', '')  # 例如: dev_zho
            detected_files[dataset_name] = file_path
            
        # 查找所有test_*.csv文件
        test_files = glob.glob(os.path.join(base_path, 'test_*.csv'))
        for file_path in test_files:
            # 提取数据集名称
            filename = os.path.basename(file_path)
            dataset_name = filename.replace('.csv', '')  # 例如: test_eng
            detected_files[dataset_name] = file_path
            
        return detected_files
    
    # 使用自动检测的测试集文件（取消注释以启用）
    # print("自动检测测试集文件...")
    # test_files = auto_detect_test_files()
    # print(f"检测到 {len(test_files)} 个测试集")
    
    all_data = []
    
    # 处理每种训练语言 - 直接读取不翻译
    for lang, file_path in train_files.items():
        print(f"\n{'='*50}")
        print(f"处理 {lang} 训练数据...")
        print(f"{'='*50}")
        
        try:
            df = pd.read_csv(file_path)
            print(f"原始数据: {len(df)} 条")
            all_data.append(df)
            print(f"完成 {lang}")
            
        except Exception as e:
            print(f"处理 {lang} 时出错: {e}")
            continue
    
    # 合并数据
    if all_data:
        combined_train = pd.concat(all_data, ignore_index=True)
    else:
        print("没有成功加载任何训练数据！")
        return
    
    print(f"\n最终训练集: {len(combined_train)} 条")
    
    # 数据预处理
    new_column_order = ['id', 'political', 'racial/ethnic', 'religious', 'gender/sexual', 'other']
    
    def reorder_columns(df, new_order):
        existing = [col for col in new_order if col in df.columns]
        others = [col for col in df.columns if col not in new_order]
        return df[existing + others]
    
    combined_train = reorder_columns(combined_train, new_column_order)
    
    # 分割数据集
    train, val = train_test_split(combined_train, test_size=0.2, random_state=42)
    
    print(f"\n数据集大小:")
    print(f"训练集: {len(train)}")
    print(f"验证集: {len(val)}")
    print(f"测试集数量: {len(test_files)}")
    
    # 加载多语言模型 - 使用XLM-RoBERTa
    print("\n加载多语言模型...")
    model_name = "xlm-roberta-base"  # 支持100种语言的多语言模型
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=5,
            problem_type="multi_label_classification"
        )
        print(f"成功加载模型: {model_name}")
    except Exception as e:
        print(f"加载多语言模型失败: {e}")
        # 备用方案：使用多语言BERT
        print("尝试加载多语言BERT...")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
        model = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-multilingual-cased",
            num_labels=5,
            problem_type="multi_label_classification"
        )
    
    # 创建训练和验证数据集
    label_columns = ['political', 'racial/ethnic', 'religious', 'gender/sexual', 'other']
    
    train_dataset = PolarizationDataset(
        train['text'].tolist(),
        train[label_columns].values.tolist(),
        tokenizer
    )
    
    val_dataset = PolarizationDataset(
        val['text'].tolist(),
        val[label_columns].values.tolist(),
        tokenizer
    )
    
    # 训练模型
    trainer = SimpleTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        learning_rate=2e-5,
        batch_size=8
    )
    
    trainer.train(epochs=3)
    
    # 对多个测试集进行预测
    print("\n" + "="*60)
    print("开始对多个测试集进行预测...")
    print("="*60)
    
    all_predictions = predict_multiple_datasets(trainer, tokenizer, test_files, label_columns)
    
    # 创建汇总报告
    print("\n" + "="*60)
    print("预测汇总报告")
    print("="*60)
    
    summary_data = []
    for dataset_name, pred_df in all_predictions.items():
        row = {'数据集': dataset_name, '样本数量': len(pred_df)}
        for col in label_columns:
            count = int(pred_df[col].sum())
            percentage = (count / len(pred_df)) * 100
            row[col] = f"{count} ({percentage:.1f}%)"
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df)
    
    # 保存汇总报告
    summary_file = '/kaggle/working/predictions_summary.csv'
    summary_df.to_csv(summary_file, index=False, encoding='utf-8-sig')
    print(f"\n汇总报告保存到: {summary_file}")
    
    # 显示生成的所有预测文件
    print("\n" + "="*60)
    print("生成的预测文件列表")
    print("="*60)
    
    prediction_files = glob.glob('/kaggle/working/pred_*.csv')
    for file_path in prediction_files:
        file_size = os.path.getsize(file_path) / 1024  # KB
        print(f"📄 {os.path.basename(file_path)} ({file_size:.1f} KB)")

if __name__ == "__main__":
    main()