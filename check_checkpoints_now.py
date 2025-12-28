"""
实时检查检查点文件
用于确认训练是否正在保存检查点
"""
import os
import sys
from pathlib import Path
from datetime import datetime
import time

# 设置输出编码
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def check_checkpoints():
    """检查检查点文件"""
    print("=" * 70)
    print("检查点状态检查")
    print("=" * 70)
    print()
    
    weights_dir = Path('./weights/')
    
    if not weights_dir.exists():
        print("❌ weights目录不存在")
        print("   说明: 训练可能刚开始，还没有保存检查点")
        print("   建议: 等待几个epoch后再次检查")
        print()
        return False
    
    print("✅ weights目录存在")
    print()
    
    # 查找所有检查点文件
    ckpt_files = list(weights_dir.glob('*.ckpt'))
    
    if not ckpt_files:
        print("⚠️  未找到检查点文件")
        print("   可能原因:")
        print("   1. 训练刚开始，还没有到保存时机")
        print("   2. 还没有找到更好的模型（最佳模型检查点）")
        print("   3. 还没有到定期保存的epoch（如果设置了save_interval）")
        print()
        print("   建议:")
        print("   - 检查训练日志，确认训练进度")
        print("   - 确认是否设置了 --save_interval 参数")
        print("   - 等待几个epoch后再次检查")
        print()
        return False
    
    print(f"✅ 找到 {len(ckpt_files)} 个检查点文件:")
    print()
    
    # 按修改时间排序
    ckpt_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    for i, ckpt_file in enumerate(ckpt_files[:10], 1):  # 只显示最新的10个
        stat = ckpt_file.stat()
        size_mb = stat.st_size / (1024 * 1024)
        mtime = datetime.fromtimestamp(stat.st_mtime)
        time_ago = datetime.now() - mtime
        
        # 判断文件是否最近更新
        if time_ago.total_seconds() < 300:  # 5分钟内
            status = "🟢 最近更新"
        elif time_ago.total_seconds() < 3600:  # 1小时内
            status = "🟡 较新"
        else:
            status = "⚪ 较旧"
        
        print(f"  [{i}] {ckpt_file.name}")
        print(f"      大小: {size_mb:.2f} MB")
        print(f"      保存时间: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"      距离现在: {time_ago.total_seconds()/60:.1f} 分钟前")
        print(f"      状态: {status}")
        print()
    
    if len(ckpt_files) > 10:
        print(f"  ... 还有 {len(ckpt_files) - 10} 个检查点文件")
        print()
    
    # 检查最新文件
    latest = ckpt_files[0]
    latest_time = datetime.fromtimestamp(latest.stat().st_mtime)
    time_since = (datetime.now() - latest_time).total_seconds()
    
    print("=" * 70)
    print("最新检查点信息:")
    print("=" * 70)
    print(f"文件: {latest.name}")
    print(f"大小: {latest.stat().st_size / (1024 * 1024):.2f} MB")
    print(f"保存时间: {latest_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    if time_since < 600:  # 10分钟内
        print(f"✅ 状态: 最近更新（{time_since/60:.1f} 分钟前）")
        print("   说明: 训练正在正常保存检查点")
    elif time_since < 3600:  # 1小时内
        print(f"⚠️  状态: 较新（{time_since/60:.1f} 分钟前）")
        print("   说明: 检查点存在，但可能训练较慢或暂停")
    else:
        print(f"⚠️  状态: 较旧（{time_since/3600:.1f} 小时前）")
        print("   说明: 检查点较旧，训练可能已停止或很慢")
    
    print()
    print("=" * 70)
    print("恢复训练命令:")
    print("=" * 70)
    print(f"python train_with_rocm.py \\")
    print(f"    --dataset ml-20m \\")
    print(f"    --data_path ml-20m/ml-20m/ \\")
    print(f"    --epoch 50 \\")
    print(f"    --batch_size 512 \\")
    print(f"    --lr 1e-4 \\")
    print(f"    --dim 64 \\")
    print(f"    --resume ./weights/{latest.name} \\")
    print(f"    --save_interval 5")
    print()
    
    return True

def check_training_logs():
    """检查训练日志"""
    print("=" * 70)
    print("训练日志检查")
    print("=" * 70)
    print()
    
    log_dir = Path('./logs/ml-20m/')
    
    if not log_dir.exists():
        print("⚠️  日志目录不存在")
        print("   说明: 可能没有启用日志功能")
        print()
        return
    
    log_files = list(log_dir.glob('*.log'))
    
    if not log_files:
        print("⚠️  未找到日志文件")
        print()
        return
    
    latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
    stat = latest_log.stat()
    mtime = datetime.fromtimestamp(stat.st_mtime)
    
    print(f"最新日志文件: {latest_log.name}")
    print(f"最后更新: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 读取最后几行
    try:
        with open(latest_log, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            if lines:
                print("最后5行日志:")
                print("-" * 70)
                for line in lines[-5:]:
                    print(line.rstrip())
                print("-" * 70)
    except Exception as e:
        print(f"读取日志失败: {e}")
    
    print()

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("训练检查点实时检查工具")
    print("=" * 70)
    print()
    
    has_checkpoints = check_checkpoints()
    check_training_logs()
    
    print("=" * 70)
    print("总结")
    print("=" * 70)
    
    if has_checkpoints:
        print("✅ 检查点文件存在，可以恢复训练")
        print("   使用上面的恢复训练命令即可")
    else:
        print("⚠️  当前没有检查点文件")
        print("   建议:")
        print("   1. 确认训练是否设置了 --save_interval 参数")
        print("   2. 等待几个epoch后再次检查")
        print("   3. 检查训练日志确认训练进度")
    
    print()

