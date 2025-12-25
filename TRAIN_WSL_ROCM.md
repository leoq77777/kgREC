# WSL2 + ROCm 训练指南

## 新增功能

✅ **WSL2 + ROCm 训练支持**
- 自动检测WSL环境
- 自动检测和安装ROCm PyTorch
- 智能设备选择（GPU/CPU自动切换）
- 完整的训练流程

## 快速开始

### 方法1: 使用自动化脚本（推荐）

在WSL中执行：
```bash
# 方式1: 使用Shell脚本
chmod +x train_wsl_rocm.sh
./train_wsl_rocm.sh [epochs] [batch_size] [lr] [dim]

# 示例: 使用默认参数
./train_wsl_rocm.sh

# 示例: 自定义参数
./train_wsl_rocm.sh 50 512 0.0001 64
```

### 方法2: 使用Python脚本（推荐）

在WSL中执行：
```bash
# 自动检测和安装ROCm
python3 train_with_rocm.py --auto_setup \
    --dataset ml-20m \
    --data_path ml-20m/ml-20m/ \
    --epoch 50 \
    --batch_size 512 \
    --lr 1e-4 \
    --dim 64
```

### 方法3: 手动安装ROCm后训练

```bash
# 1. 安装ROCm PyTorch
pip3 uninstall torch torchvision torchaudio
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7

# 2. 验证安装
python3 check_gpu.py

# 3. 开始训练
python3 train_with_rocm.py \
    --dataset ml-20m \
    --data_path ml-20m/ml-20m/ \
    --epoch 50 \
    --batch_size 512
```

## 功能特性

### 1. 自动环境检测
- ✅ 检测WSL环境
- ✅ 检测PyTorch版本（CPU/ROCm）
- ✅ 检测GPU可用性
- ✅ 显示GPU信息（名称、显存）

### 2. 自动ROCm安装
- ✅ 检测到CPU版本时提示安装
- ✅ 自动卸载CPU版本
- ✅ 自动安装ROCm版本
- ✅ 安装后自动验证

### 3. 智能设备选择
- ✅ 自动选择最佳设备（GPU优先）
- ✅ GPU不可用时自动降级到CPU
- ✅ 显示设备信息和使用建议

### 4. 训练参数
- `--dataset`: 数据集名称（默认: ml-20m）
- `--data_path`: 数据路径（默认: ml-20m/ml-20m/）
- `--epoch`: 训练轮数（默认: 50）
- `--batch_size`: 批次大小（默认: 512）
- `--lr`: 学习率（默认: 1e-4）
- `--dim`: 嵌入维度（默认: 64）
- `--auto_setup`: 自动设置ROCm环境
- `--force_cpu`: 强制使用CPU

## 训练输出

训练过程中会显示：
- 设备信息（GPU/CPU）
- 训练进度条
- 每个epoch的损失和评估指标
- 最佳模型保存位置

## 性能对比

| 配置 | 每轮时间 | 50轮总时间 | 速度提升 |
|------|---------|-----------|---------|
| CPU  | ~45分钟 | ~37.5小时 | 1x      |
| GPU  | ~5分钟  | ~4小时    | 9x      |

## 故障排除

### 问题1: 显示"未检测到GPU支持"
**解决方案**:
1. 检查GPU是否支持ROCm（RX 6000/7000系列）
2. 运行 `python3 train_with_rocm.py --auto_setup` 安装ROCm
3. 检查WSL版本（必须是WSL2）

### 问题2: 安装ROCm后仍然不可用
**解决方案**:
1. 检查ROCm驱动是否安装
2. 查看 `README_ROCm.md` 获取详细配置指南
3. 尝试重启WSL

### 问题3: 训练时内存不足
**解决方案**:
1. 减小 `--batch_size`（如256或128）
2. 减小 `--dim`（如32）
3. 使用CPU训练（添加 `--force_cpu`）

## 文件说明

- `train_with_rocm.py`: 主训练脚本（Python）
- `train_wsl_rocm.sh`: Shell训练脚本
- `check_gpu.py`: GPU检测脚本
- `install_rocm_pytorch.sh`: ROCm安装脚本
- `README_ROCm.md`: ROCm配置详细文档

## 下一步

1. 在WSL中运行 `python3 train_with_rocm.py --auto_setup`
2. 等待ROCm安装完成
3. 开始GPU训练！

