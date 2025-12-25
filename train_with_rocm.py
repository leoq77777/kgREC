"""
WSL2 + ROCm 训练脚本
自动检测和配置ROCm环境，支持AMD GPU训练
"""
import os
import sys
import subprocess
import torch
import argparse
from pathlib import Path

def check_wsl():
    """检查是否在WSL环境中"""
    return os.path.exists('/proc/version') and 'microsoft' in open('/proc/version').read().lower()

def check_rocm_installation():
    """检查ROCm是否已安装"""
    rocm_available = torch.cuda.is_available()
    rocm_version = None
    
    if rocm_available:
        try:
            # 尝试获取ROCm版本信息
            if hasattr(torch.version, 'hip'):
                rocm_version = torch.version.hip
            elif hasattr(torch.version, 'cuda'):
                # 在ROCm中，cuda实际上指向HIP
                rocm_version = torch.version.cuda
        except:
            pass
    
    return rocm_available, rocm_version

def install_rocm_pytorch():
    """安装ROCm版本的PyTorch"""
    print("=" * 60)
    print("检测到需要安装ROCm版本的PyTorch")
    print("=" * 60)
    
    response = input("是否现在安装? (y/n): ").strip().lower()
    if response != 'y':
        print("取消安装，将使用CPU训练")
        return False
    
    print("\n正在卸载CPU版本的PyTorch...")
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', 
                       'torch', 'torchvision', 'torchaudio'], 
                      check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"卸载时出错: {e}")
        return False
    
    print("\n正在安装ROCm 5.7版本的PyTorch...")
    print("这可能需要几分钟，请耐心等待...")
    
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'install', 'torch', 'torchvision', 'torchaudio',
             '--index-url', 'https://download.pytorch.org/whl/rocm5.7'],
            check=True,
            capture_output=True,
            text=True
        )
        print("安装成功！")
        
        # 重新导入torch以获取新版本
        import importlib
        importlib.reload(torch)
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"安装失败: {e}")
        print(f"错误输出: {e.stderr}")
        return False

def setup_rocm_environment():
    """设置ROCm环境"""
    print("\n" + "=" * 60)
    print("ROCm环境检查")
    print("=" * 60)
    
    # 检查WSL
    is_wsl = check_wsl()
    print(f"WSL环境: {'是' if is_wsl else '否'}")
    
    if not is_wsl:
        print("警告: 未检测到WSL环境，ROCm可能无法正常工作")
    
    # 检查PyTorch版本
    print(f"\n当前PyTorch版本: {torch.__version__}")
    is_cpu_version = '+cpu' in torch.__version__ or 'cpu' in torch.__version__.lower()
    
    if is_cpu_version:
        print("检测到CPU版本PyTorch")
        if install_rocm_pytorch():
            # 重新检查
            rocm_available, rocm_version = check_rocm_installation()
        else:
            return False, None
    else:
        rocm_available, rocm_version = check_rocm_installation()
    
    # 显示ROCm状态
    print(f"\nROCm可用: {rocm_available}")
    if rocm_available:
        print(f"ROCm版本: {rocm_version or '未知'}")
        print(f"GPU设备数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  设备 {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"    显存: {props.total_memory / 1024**3:.2f} GB")
    else:
        print("未检测到ROCm支持")
        print("可能原因:")
        print("  1. GPU不支持ROCm")
        print("  2. 需要安装ROCm驱动")
        print("  3. WSL配置问题")
    
    return rocm_available, rocm_version

def main():
    parser = argparse.ArgumentParser(description='WSL2 + ROCm训练脚本')
    parser.add_argument('--dataset', type=str, default='ml-20m', help='数据集名称')
    parser.add_argument('--data_path', type=str, default='ml-20m/ml-20m/', help='数据路径')
    parser.add_argument('--epoch', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=512, help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--dim', type=int, default=64, help='嵌入维度')
    parser.add_argument('--auto_setup', action='store_true', help='自动设置ROCm环境')
    parser.add_argument('--force_cpu', action='store_true', help='强制使用CPU')
    parser.add_argument('--resume', type=str, default=None, help='从检查点恢复训练（检查点文件路径）')
    parser.add_argument('--save_interval', type=int, default=5, help='每N个epoch保存一次检查点（用于恢复训练）')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("KGRec WSL2 + ROCm 训练")
    print("=" * 60)
    
    # 设置ROCm环境
    if args.auto_setup and not args.force_cpu:
        rocm_available, rocm_version = setup_rocm_environment()
    else:
        rocm_available, _ = check_rocm_installation()
    
    # 确定使用的设备
    use_gpu = rocm_available and not args.force_cpu
    device_type = "GPU (ROCm)" if use_gpu else "CPU"
    
    print("\n" + "=" * 60)
    print("训练配置")
    print("=" * 60)
    print(f"设备: {device_type}")
    print(f"数据集: {args.dataset}")
    print(f"数据路径: {args.data_path}")
    print(f"训练轮数: {args.epoch}")
    print(f"批次大小: {args.batch_size}")
    print(f"学习率: {args.lr}")
    print(f"嵌入维度: {args.dim}")
    print("=" * 60)
    
    # 检查数据文件
    data_path = Path(args.data_path)
    train_file = data_path / "train.txt"
    test_file = data_path / "test.txt"
    kg_file = data_path / "kg_final.txt"
    
    if not train_file.exists():
        print(f"\n错误: 训练文件不存在: {train_file}")
        print("请先运行: python prepare_data_for_kgrec.py")
        sys.exit(1)
    
    # 构建训练命令
    cmd = [
        sys.executable, 'run_kgrec.py',
        '--dataset', args.dataset,
        '--data_path', str(data_path) + '/',
        '--epoch', str(args.epoch),
        '--batch_size', str(args.batch_size),
        '--lr', str(args.lr),
        '--dim', str(args.dim),
        '--gpu_id', '0',
        '--cuda', '1' if use_gpu else '0',
        '--save',
        '--out_dir', './weights/',
        '--log',  # 启用日志保存
        '--log_fn', 'train',  # 日志文件名
        '--save_interval', str(args.save_interval)  # 定期保存检查点
    ]
    
    # 如果指定了恢复训练，添加resume参数
    if args.resume:
        cmd.extend(['--resume', args.resume])
        print(f"\n将从检查点恢复训练: {args.resume}")
    
    print(f"\n执行命令:")
    print(' '.join(cmd))
    print("\n" + "=" * 60)
    print("开始训练...")
    print("=" * 60 + "\n")
    
    # 执行训练
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n训练失败，错误代码: {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n训练被用户中断")
        sys.exit(0)

if __name__ == '__main__':
    main()

