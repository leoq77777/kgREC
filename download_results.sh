#!/bin/bash
# 从云服务器下载训练结果
# 使用方法: ./download_results.sh user@host:/path/to/source

if [ $# -eq 0 ]; then
    echo "使用方法: $0 user@host:/path/to/source"
    echo "示例: $0 ubuntu@123.45.67.89:/home/ubuntu/kgREC"
    exit 1
fi

SOURCE=$1
LOCAL_DIR="./cloud_results"

echo "=========================================="
echo "从云服务器下载训练结果"
echo "源: $SOURCE"
echo "本地目录: $LOCAL_DIR"
echo "=========================================="

# 创建本地目录
mkdir -p "$LOCAL_DIR"

# 需要下载的文件和目录
ITEMS=(
    "weights/"
    "logs/"
    "plots/"
    "train_output.log"
    "*.ckpt"
)

# 使用rsync下载
if command -v rsync &> /dev/null; then
    echo "使用rsync下载..."
    for item in "${ITEMS[@]}"; do
        echo "下载: $item"
        rsync -avz --progress "$SOURCE/$item" "$LOCAL_DIR/" 2>/dev/null || true
    done
else
    echo "使用scp下载..."
    for item in "${ITEMS[@]}"; do
        echo "下载: $item"
        scp -r "$SOURCE/$item" "$LOCAL_DIR/" 2>/dev/null || true
    done
fi

echo "=========================================="
echo "下载完成！"
echo "结果保存在: $LOCAL_DIR"
echo "=========================================="

