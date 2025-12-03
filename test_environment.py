import torch
import pandas as pd
import os

# 检查PyTorch版本和GPU可用性
print(f"PyTorch版本: {torch.__version__}")
print(f"GPU可用: {torch.cuda.is_available()}")

# 检查数据集路径
dataset_path = os.path.join("ml-20m", "ml-20m", "ratings.csv")
print(f"数据集路径存在: {os.path.exists(dataset_path)}")

# 读取数据示例
if os.path.exists(dataset_path):
    df = pd.read_csv(dataset_path, nrows=5)
    print("数据示例:\n", df.head())
else:
    print("无法读取数据集，请检查路径是否正确")