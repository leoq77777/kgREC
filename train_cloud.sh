#!/bin/bash
# 云训练启动脚本
# 用于在云服务器上启动训练

set -e

# 激活虚拟环境
source venv/bin/activate

# 训练参数配置
DATASET="ml-20m"
DATA_PATH="ml-20m/ml-20m/"
EPOCHS=50
BATCH_SIZE=512
LR=1e-4
DIM=64
SAVE_INTERVAL=5

# 检查数据文件
if [ ! -f "${DATA_PATH}/train.txt" ]; then
    echo "错误: 训练文件不存在，请先准备数据"
    echo "运行: python prepare_data_for_kgrec.py"
    exit 1
fi

# 创建必要的目录
mkdir -p weights
mkdir -p logs/${DATASET}
mkdir -p plots

# 启动训练
echo "=========================================="
echo "开始训练"
echo "=========================================="
echo "数据集: ${DATASET}"
echo "训练轮数: ${EPOCHS}"
echo "批次大小: ${BATCH_SIZE}"
echo "学习率: ${LR}"
echo "嵌入维度: ${DIM}"
echo "检查点间隔: ${SAVE_INTERVAL}"
echo "=========================================="

# 检测是否使用GPU
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    echo "检测到GPU，使用GPU训练"
    GPU_FLAG=""
else
    echo "未检测到GPU，使用CPU训练"
    GPU_FLAG="--force_cpu"
fi

python train_with_rocm.py \
    --dataset ${DATASET} \
    --data_path ${DATA_PATH} \
    --epoch ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --lr ${LR} \
    --dim ${DIM} \
    ${GPU_FLAG} \
    --save_interval ${SAVE_INTERVAL} \
    2>&1 | tee train_output.log

echo "=========================================="
echo "训练完成！"
echo "=========================================="

