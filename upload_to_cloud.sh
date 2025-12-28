#!/bin/bash
# 上传文件到云服务器
# 使用方法: ./upload_to_cloud.sh user@host:/path/to/destination

if [ $# -eq 0 ]; then
    echo "使用方法: $0 user@host:/path/to/destination"
    echo "示例: $0 ubuntu@123.45.67.89:/home/ubuntu/kgREC"
    exit 1
fi

DEST=$1

echo "=========================================="
echo "上传文件到云服务器"
echo "目标: $DEST"
echo "=========================================="

# 需要上传的文件和目录
FILES=(
    "run_kgrec.py"
    "train_with_rocm.py"
    "prepare_data_for_kgrec.py"
    "requirements.txt"
    "setup_cloud_env.sh"
    "train_cloud.sh"
    "modules/"
    "utils/"
    ".gitignore"
)

# 使用rsync上传（推荐，支持断点续传）
if command -v rsync &> /dev/null; then
    echo "使用rsync上传..."
    for item in "${FILES[@]}"; do
        if [ -e "$item" ]; then
            echo "上传: $item"
            rsync -avz --progress "$item" "$DEST/"
        fi
    done
else
    echo "使用scp上传（如果没有rsync）..."
    for item in "${FILES[@]}"; do
        if [ -e "$item" ]; then
            echo "上传: $item"
            scp -r "$item" "$DEST/"
        fi
    done
fi

echo "=========================================="
echo "上传完成！"
echo "=========================================="
echo "下一步:"
echo "1. SSH连接到服务器: ssh ${DEST%%:*}"
echo "2. 运行设置脚本: bash setup_cloud_env.sh"
echo "3. 准备数据（如果需要）: python prepare_data_for_kgrec.py"
echo "4. 开始训练: bash train_cloud.sh"

