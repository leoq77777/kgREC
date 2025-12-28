#!/bin/bash
# 云环境设置脚本
# 用于在云服务器上快速设置训练环境

set -e

echo "=========================================="
echo "KGRec 云训练环境设置"
echo "=========================================="

# 1. 更新系统
echo "更新系统包..."
sudo apt-get update -y
sudo apt-get upgrade -y

# 2. 安装Python和基础工具
echo "安装Python和工具..."
sudo apt-get install -y python3 python3-pip git wget curl

# 3. 安装CUDA（如果需要GPU）
if command -v nvidia-smi &> /dev/null; then
    echo "检测到NVIDIA GPU，安装CUDA工具包..."
    # 这里可以根据需要安装CUDA
    # 大多数云平台已经预装了CUDA
    nvidia-smi
fi

# 4. 创建Python虚拟环境
echo "创建虚拟环境..."
python3 -m venv venv
source venv/bin/activate

# 5. 升级pip
echo "升级pip..."
pip install --upgrade pip

# 6. 安装依赖
echo "安装Python依赖..."
pip install -r requirements.txt

# 7. 验证安装
echo "验证安装..."
python3 -c "import torch; print(f'PyTorch版本: {torch.__version__}')"
if command -v nvidia-smi &> /dev/null; then
    python3 -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}')"
    if torch.cuda.is_available():
        print(f'GPU数量: {torch.cuda.device_count()}')
        print(f'GPU名称: {torch.cuda.get_device_name(0)}')
fi

echo "=========================================="
echo "环境设置完成！"
echo "=========================================="
echo "激活虚拟环境: source venv/bin/activate"
echo "开始训练: python train_with_rocm.py ..."

