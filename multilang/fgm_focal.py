import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import glob
from transformers import get_linear_schedule_with_warmup, AutoTokenizer, AutoModelForSequenceClassification

# ==================== 1. 对抗训练模块 (FGM) ====================
class FGM:
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1.0, emb_name='word_embeddings'):
        # 在embedding上添加扰动
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='word_embeddings'):
        # 恢复embedding
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

# ==================== 2. 新增 Focal Loss ====================
class MultiLabelFocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, reduction='mean'):
        """
        alpha: 控制正负样本权重的平衡 (0.25 是常用值，表示降低负样本权重)
        gamma: 聚焦参数，越大越关注难分样本 (常用 2.0)
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, labels):
        # 1. 计算二元交叉熵 (不进行 reduction，保留每个样本的 loss)
        bce_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
        
        # 2. 计算预测概率 p_t (对应于真实类别的概率)
        probs = torch.sigmoid(logits)
        p_t = probs * labels + (1 - probs) * (1 - labels)
        
        # 3. 计算 Focal Term: (1 - p_t) ^ gamma
        focal_term = (1 - p_t).pow(self.gamma)
        
        # 4. 应用 Alpha 平衡 (可选)
        if self.alpha is not None:
            alpha_t = self.alpha * labels + (1 - self.alpha) * (1 - labels)
            loss = alpha_t * focal_term * bce_loss
        else:
            loss = focal_term * bce_loss

        # 5. Reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

# ==================== 3. 改进的训练器 ====================

class StableTrainer:
    def __init__(self, model, tokenizer, train_dataset, val_dataset, 
                 learning_rate=1e-5, batch_size=8, accumulation_steps=4):
        
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.accumulation_steps = accumulation_steps
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
            
        self.model.to(self.device)
        
        # 使用 Focal Loss 替代原有的 RobustLoss
        # alpha=0.75, gamma=2.0 是经验上效果较好的默认值
        self.criterion = MultiLabelFocalLoss(alpha=0.75, gamma=2.0)
        
        # 初始化对抗训练
        self.fgm = FGM(model)
        
        # 优化器设置
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        
        self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)
        
        total_steps = len(train_dataset) * 5 // (batch_size * accumulation_steps)
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )
        
        self.best_f1 = 0
        self.best_thresholds = np.array([0.5] * 5)
        
    def train(self, epochs=5):
        print("开始训练...")
        
        for epoch in range(epochs):
            self.model.train()
            train_loader = torch.utils.data.DataLoader(
                self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0
            )
            
            total_loss = 0
            self.optimizer.zero_grad()
            
            for batch_idx, batch in enumerate(train_loader):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # 1. 正常前向传播
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = self.criterion(outputs.logits, labels)
                loss = loss / self.accumulation_steps
                loss.backward()
                
                # 2. 对抗训练 (FGM)
                self.fgm.attack()
                outputs_adv = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss_adv = self.criterion(outputs_adv.logits, labels)
                loss_adv = loss_adv / self.accumulation_steps
                loss_adv.backward()
                self.fgm.restore()
                
                total_loss += loss.item() * self.accumulation_steps
                
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                
                if batch_idx % 50 == 0:
                    print(f'Epoch: {epoch+1}, Batch: {batch_idx}, Loss: {loss.item()*self.accumulation_steps:.4f}')
            
            avg_loss = total_loss / len(train_loader)
            print(f'\nEpoch {epoch+1} 完成, 平均损失: {avg_loss:.4f}')
            
            # 验证并搜索最佳阈值
            self.evaluate_and_tune(epoch)
            
    def evaluate_and_tune(self, epoch):
        self.model.eval()
        val_loader = torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.batch_size * 2, shuffle=False
        )
        
        all_logits = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                all_logits.append(outputs.logits.cpu())
                all_labels.append(labels.cpu())
        
        all_logits = torch.cat(all_logits, dim=0)
        all_probs = torch.sigmoid(all_logits).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()
        
        # 搜索最佳阈值
        best_thresholds = np.array([0.5] * 5)
        
        # 对每个标签独立搜索
        for i in range(5):
            best_t = 0.5
            best_f1_i = 0
            for t in np.arange(0.2, 0.8, 0.05):
                pred_i = (all_probs[:, i] > t).astype(int)
                f1_i = f1_score(all_labels[:, i], pred_i, zero_division=0)
                if f1_i > best_f1_i:
                    best_f1_i = f1_i
                    best_t = t
            best_thresholds[i] = best_t
        
        final_preds = np.zeros_like(all_probs)
        for i in range(5):
            final_preds[:, i] = (all_probs[:, i] > best_thresholds[i]).astype(int)
            
        val_f1 = f1_score(all_labels, final_preds, average='macro', zero_division=0)
        
        print(f"Epoch {epoch+1} 验证 F1: {val_f1:.4f}")
        print(f"最佳阈值: {best_thresholds}")
        
        if val_f1 > self.best_f1:
            self.best_f1 = val_f1
            self.best_thresholds = best_thresholds
            torch.save({
                'model': self.model.state_dict(),
                'thresholds': self.best_thresholds
            }, 'best_model.pth')
            print("✅ 模型已保存 (F1提升)")

    def predict(self, test_dataset):
        print(f"加载最佳模型进行预测...")
        # =========================================================
        # FIX: 添加 weights_only=False 解决 UnpicklingError
        # =========================================================
        checkpoint = torch.load('best_model.pth', weights_only=False)
        
        self.model.load_state_dict(checkpoint['model'])
        thresholds = checkpoint['thresholds']
        print(f"使用加载的最佳阈值: {thresholds}")
        
        self.model.eval()
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size * 2)
        
        all_probs = []
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.sigmoid(outputs.logits)
                all_probs.append(probs.cpu())
        
        all_probs = torch.cat(all_probs, dim=0).numpy()
        
        final_preds = np.zeros_like(all_probs)
        for i in range(len(thresholds)):
            final_preds[:, i] = (all_probs[:, i] > thresholds[i]).astype(int)
            
        return final_preds, all_probs

# ==================== 数据集类 ====================
class ImprovedPolarizationDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text, truncation=True, padding='max_length',
            max_length=self.max_length, return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.float)
        }

# ==================== 主函数 ====================

def main():
    # 1. 准备文件路径
    train_files = {
        'eng': 'data/train_eng.csv', 'amh': 'data/train_amh.csv',
        'arb': 'data/train_arb.csv', 'deu': 'data/train_deu.csv',
        'fas': 'data/train_fas.csv', 'hau': 'data/train_hau.csv',
        'hin': 'data/train_hin.csv', 'ita': 'data/train_ita.csv',
        'nep': 'data/train_nep.csv', 'spa': 'data/train_spa.csv',
        'tur': 'data/train_tur.csv', 'khm': 'data/train_khm.csv',
        'ben': 'data/train_ben.csv', 'mya': 'data/train_mya.csv',
        'ori': 'data/train_ori.csv', 'pan': 'data/train_pan.csv',
        'pol': 'data/train_pol.csv', 'rus': 'data/train_rus.csv',
        'swa': 'data/train_swa.csv', 'tel': 'data/train_tel.csv',
        'urd': 'data/train_urd.csv', 'zho': 'data/train_zho.csv'
    }
    
    test_files = {
        'zho': 'data/dev_zho.csv', 'eng': 'data/dev_eng.csv',
        'arb': 'data/dev_arb.csv', 'deu': 'data/dev_deu.csv',
        'fas': 'data/dev_fas.csv', 'hin': 'data/dev_hin.csv',
        'spa': 'data/dev_spa.csv', 'hau': 'data/dev_hau.csv',
        'amh': 'data/dev_amh.csv', 'ita': 'data/dev_ita.csv',
        'nep': 'data/dev_nep.csv', 'tur': 'data/dev_tur.csv',
        'urd': 'data/dev_urd.csv', 'khm': 'data/dev_khm.csv',
        'ben': 'data/dev_ben.csv', 'mya': 'data/dev_mya.csv',
        'ori': 'data/dev_ori.csv', 'pan': 'data/dev_pan.csv',
        'pol': 'data/dev_pol.csv', 'rus': 'data/dev_rus.csv',
        'swa': 'data/dev_swa.csv', 'tel': 'data/dev_tel.csv'
    }

    # 2. 加载训练数据
    all_data = []
    print("正在加载训练数据...")
    for lang, path in train_files.items():
        if os.path.exists(path):
            try:
                df = pd.read_csv(path, on_bad_lines='skip')
                # 简单的数据增强：对低资源语言复制一遍（过采样）
                if lang in ['amh', 'hau', 'ben', 'ori', 'pan', 'swa', 'tel']:
                    all_data.append(df)
                all_data.append(df)
            except:
                pass
    
    if not all_data:
        print("未找到训练数据，请检查路径。")
        return

    combined_train = pd.concat(all_data, ignore_index=True)
    combined_train = combined_train.dropna(subset=['text'])
    
    label_columns = ['political', 'racial/ethnic', 'religious', 'gender/sexual', 'other']
    
    # 注意：由于使用了 Focal Loss，我们不再需要手动计算 pos_weights 传入 Loss
    # Focal Loss 会自动处理难分类样本
    
    # 4. 数据划分
    train_df, val_df = train_test_split(combined_train, test_size=0.15, random_state=42)
    
    # 5. 模型初始化
    model_name = "xlm-roberta-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=len(label_columns),
        problem_type="multi_label_classification"
    )
    
    train_dataset = ImprovedPolarizationDataset(
        train_df['text'].tolist(), train_df[label_columns].values, tokenizer
    )
    val_dataset = ImprovedPolarizationDataset(
        val_df['text'].tolist(), val_df[label_columns].values, tokenizer
    )
    
    # 6. 训练
    trainer = StableTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        learning_rate=1e-5,
        batch_size=8,
        accumulation_steps=4
    )
    
    trainer.train(epochs=5)
    
    # 7. 预测
    print("\n开始预测...")
    output_dir = './submission/'
    os.makedirs(output_dir, exist_ok=True)
    
    for lang, path in test_files.items():
        if not os.path.exists(path): continue
        
        print(f"预测: {lang}")
        test_df = pd.read_csv(path)
        test_dataset = ImprovedPolarizationDataset(
            test_df['text'].fillna('').tolist(),
            np.zeros((len(test_df), 5)),
            tokenizer
        )
        
        preds, _ = trainer.predict(test_dataset)
        
        # 结果保存
        sub_df = test_df[['id']].copy() if 'id' in test_df.columns else pd.DataFrame(index=range(len(test_df)))
        for i, col in enumerate(label_columns):
            sub_df[col] = preds[:, i]
            
        sub_df.to_csv(os.path.join(output_dir, f'pred_{lang}.csv'), index=False)

if __name__ == "__main__":
    main()