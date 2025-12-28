# KGRec 云训练部署指南

## 🚀 快速开始

### 1. 准备云服务器

推荐平台：
- **Vast.ai** (最便宜，$0.5-2/小时)
- **RunPod** (简单易用)
- **AWS EC2** (稳定可靠)
- **Google Cloud** (机器学习友好)

### 2. 上传代码

```bash
# 使用提供的上传脚本
chmod +x upload_to_cloud.sh
./upload_to_cloud.sh user@your-server:/path/to/kgREC

# 或使用git（如果服务器有git）
git clone https://github.com/leoq77777/kgREC.git
cd kgREC
```

### 3. 设置环境

```bash
# SSH连接到服务器
ssh user@your-server

# 进入项目目录
cd /path/to/kgREC

# 运行环境设置脚本
chmod +x setup_cloud_env.sh
bash setup_cloud_env.sh
```

### 4. 准备数据

```bash
# 如果数据已上传，直接使用
# 如果需要重新生成数据
python prepare_data_for_kgrec.py
```

### 5. 开始训练

```bash
# 使用提供的训练脚本（自动检测GPU）
chmod +x train_cloud.sh
bash train_cloud.sh

# 或直接运行
python train_with_rocm.py \
    --dataset ml-20m \
    --data_path ml-20m/ml-20m/ \
    --epoch 50 \
    --batch_size 512 \
    --lr 1e-4 \
    --dim 64 \
    --save_interval 5
```

### 6. 监控训练

```bash
# 查看实时输出
tail -f train_output.log

# 检查GPU使用
watch -n 1 nvidia-smi

# 检查检查点
python check_checkpoints_now.py
```

### 7. 下载结果

在本地执行：

```bash
chmod +x download_results.sh
./download_results.sh user@your-server:/path/to/kgREC
```

## 📋 文件说明

- `setup_cloud_env.sh` - 自动设置云环境
- `train_cloud.sh` - 启动训练脚本
- `upload_to_cloud.sh` - 上传文件到服务器
- `download_results.sh` - 下载训练结果
- `requirements-cloud.txt` - 云环境依赖（GPU优化）
- `check_checkpoints_now.py` - 检查检查点状态

## ⚙️ 配置说明

### GPU训练

脚本会自动检测GPU，如果可用则使用GPU训练。

### 批次大小调整

GPU内存更大，可以增加批次大小：

```bash
# 在 train_cloud.sh 中修改
BATCH_SIZE=1024  # 或更大
```

### 使用screen/tmux保持训练

```bash
# 使用screen
screen -S training
bash train_cloud.sh
# 按 Ctrl+A 然后 D 退出，用 screen -r training 恢复

# 使用tmux
tmux new -s training
bash train_cloud.sh
# 按 Ctrl+B 然后 D 退出，用 tmux attach -t training 恢复
```

## 💰 成本估算

- **RTX 3090**: ~$0.5-1/小时，训练4小时 = $2-4
- **A100**: ~$1-2/小时，训练4小时 = $4-8
- 比本地CPU训练快10-40倍

## 🆘 故障排除

### GPU不可用

```bash
# 检查GPU
nvidia-smi

# 如果不可用，强制使用CPU
python train_with_rocm.py ... --force_cpu
```

### 内存不足

减小批次大小：
```bash
--batch_size 256  # 或更小
```

### 连接中断

使用screen/tmux保持会话，或使用nohup：
```bash
nohup bash train_cloud.sh > train.log 2>&1 &
```

## 📚 更多信息

查看 `云训练迁移指南.md` 获取详细说明。

