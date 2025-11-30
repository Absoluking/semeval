import os
import pandas as pd
import json
import time
from openai import OpenAI
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
import numpy as np

# 初始化客户端
client = OpenAI(
    api_key="",  # 请在此处填入您的API Key
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

def predict_polarization(text):
    """
    使用千问3 API预测文本的极化程度
    """
    prompt = f"""
    请分析以下文本内容，判断其在五个维度上是否包含极化内容。对于每个维度，如果是极化内容输出1，否则输出0。
    
    五个维度定义：
    - political: 政治极化，涉及政治立场、意识形态对立等
    - racial/ethnic: 种族/民族极化，涉及种族歧视、民族对立等  
    - religious: 宗教极化，涉及宗教冲突、信仰对立等
    - gender/sexual: 性别/性取向极化，涉及性别歧视、性取向对立等
    - other: 其他极化，如地域歧视、社会阶层对立等
    
    文本内容：{text}
    
    请以JSON格式输出结果，格式如下：
    {{
        "political": 0或1,
        "racial/ethnic": 0或1, 
        "religious": 0或1,
        "gender/sexual": 0或1,
        "other": 0或1
    }}
    只输出JSON，不要其他内容。
    """
    
    try:
        completion = client.chat.completions.create(
            model="qwen3-max",
            messages=[
                {"role": "system", "content": "你是一个专业的文本内容分析助手，能够准确识别文本中的极化内容。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1  # 降低随机性，使结果更稳定
        )
        
        response = completion.choices[0].message.content.strip()
        
        # 解析JSON响应
        result = json.loads(response)
        return result
        
    except Exception as e:
        print(f"API调用错误: {e}")
        # 返回默认值
        return {
            "political": 0,
            "racial/ethnic": 0,
            "religious": 0,
            "gender/sexual": 0,
            "other": 0
        }

def evaluate_predictions(true_labels, pred_labels, categories):
    """
    评估预测结果并计算F1分数等指标
    """
    results = {}
    
    for category in categories:
        if category in true_labels.columns and category in pred_labels.columns:
            true = true_labels[category].values
            pred = pred_labels[category].values
            
            # 计算各项指标
            f1 = f1_score(true, pred, zero_division=0)
            precision = precision_score(true, pred, zero_division=0)
            recall = recall_score(true, pred, zero_division=0)
            
            results[category] = {
                'f1': f1,
                'precision': precision,
                'recall': recall,
                'support': np.sum(true)
            }
    
    # 计算宏平均F1
    macro_f1 = np.mean([results[cat]['f1'] for cat in results])
    results['macro_avg'] = {
        'f1': macro_f1,
        'precision': np.mean([results[cat]['precision'] for cat in results if cat != 'macro_avg']),
        'recall': np.mean([results[cat]['recall'] for cat in results if cat != 'macro_avg']),
        'support': np.sum([results[cat]['support'] for cat in results if cat != 'macro_avg'])
    }
    
    return results

def print_evaluation_results(results):
    """
    打印评估结果
    """
    print("\n" + "="*80)
    print("模型性能评估结果")
    print("="*80)
    
    # 打印每个类别的结果
    categories = [cat for cat in results if cat != 'macro_avg']
    print(f"{'类别':<15} {'F1分数':<10} {'精确率':<10} {'召回率':<10} {'支持度':<10}")
    print("-" * 60)
    
    for category in categories:
        metrics = results[category]
        print(f"{category:<15} {metrics['f1']:<10.4f} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f} {metrics['support']:<10}")
    
    # 打印宏平均
    macro_metrics = results['macro_avg']
    print("-" * 60)
    print(f"{'宏平均':<15} {macro_metrics['f1']:<10.4f} {macro_metrics['precision']:<10.4f} {macro_metrics['recall']:<10.4f} {macro_metrics['support']:<10}")
    print("="*80)

def predict_dataset(input_file, output_file, has_labels=False, true_labels_file=None):
    """
    预测整个数据集
    """
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误：输入文件 {input_file} 不存在！")
        return pd.DataFrame()
    
    # 读取数据
    df = pd.read_csv(input_file)
    
    # 存储预测结果
    predictions = []
    
    print(f"开始处理 {len(df)} 条数据...")
    
    for index, row in df.iterrows():
        text = row['text']
        id_val = row['id']
        
        print(f"处理第 {index + 1}/{len(df)} 条: {text[:50]}...")
        
        # 调用API进行预测
        result = predict_polarization(text)
        
        # 添加ID到结果中
        result['id'] = id_val
        
        predictions.append(result)
        
        # 添加延迟以避免API限制
        time.sleep(1)
    
    # 转换为DataFrame并重新排列列顺序
    result_df = pd.DataFrame(predictions)
    result_df = result_df[['id', 'political', 'racial/ethnic', 'religious', 'gender/sexual', 'other']]
    
    # 保存结果
    result_df.to_csv(output_file, index=False)
    print(f"预测完成！结果已保存到 {output_file}")
    
    # 打印统计信息
    print("\n预测结果统计:")
    for col in ['political', 'racial/ethnic', 'religious', 'gender/sexual', 'other']:
        count = result_df[col].sum()
        print(f"{col}: {count} 条极化内容 ({count/len(result_df)*100:.1f}%)")
    
    return result_df

def main():
    # 配置参数
    train_file = '111.csv'  # 训练集文件（需要包含真实标签）
    test_file = 'dev_urd.csv'        # 测试集文件
    train_output = 'pred_train_zho.csv'  # 训练集预测结果
    test_output = 'pred_urd.csv'         # 测试集预测结果
    
    # 检查训练集是否存在（用于评估）
    if os.path.exists(train_file):
        print("检测到训练集文件，开始预测训练集并评估模型性能...")
        
        # 读取训练集真实标签
        train_df = pd.read_csv(train_file)
        
        # 预测训练集
        train_predictions = predict_dataset(train_file, train_output, has_labels=True)
        
        # 提取真实标签和预测标签
        true_labels = train_df[['political', 'racial/ethnic', 'religious', 'gender/sexual', 'other']]
        pred_labels = train_predictions[['political', 'racial/ethnic', 'religious', 'gender/sexual', 'other']]
        
        # 评估模型性能
        categories = ['political', 'racial/ethnic', 'religious', 'gender/sexual', 'other']
        evaluation_results = evaluate_predictions(true_labels, pred_labels, categories)
        
        # 打印评估结果
        print_evaluation_results(evaluation_results)
        
        # 保存评估结果到文件
        eval_df = pd.DataFrame(evaluation_results).T
        eval_df.to_csv('evaluation_results.csv', index=True)
        print("评估结果已保存到 evaluation_results.csv")
    
    else:
        print(f"未找到训练集文件 {train_file}，跳过模型评估步骤")
    
    # 预测测试集
    print("\n开始预测测试集...")
    predict_dataset(test_file, test_output)

if __name__ == "__main__":
    main()