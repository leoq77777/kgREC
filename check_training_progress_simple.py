"""
简化的训练进度检查脚本
快速查看训练状态
"""
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime

def main():
    print("\n" + "=" * 70)
    print("训练进度快速检查")
    print("=" * 70 + "\n")
    
    # 1. 检查进程
    print("【1】检查运行中的Python进程:")
    print("-" * 70)
    try:
        if sys.platform == 'win32':
            result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq python.exe'], 
                                   capture_output=True, text=True)
            lines = [l for l in result.stdout.split('\n') if 'python.exe' in l]
            if len(lines) > 0:
                print(f"找到 {len(lines)} 个Python进程")
                for line in lines:
                    parts = line.split()
                    if len(parts) >= 5:
                        pid = parts[1]
                        mem_kb = parts[4].replace(',', '')
                        try:
                            mem_mb = int(mem_kb) / 1024
                            if mem_mb > 500:  # 大于500MB可能是训练进程
                                print(f"  ⚠️  进程 {pid}: {mem_mb:.0f} MB (可能是训练进程)")
                            else:
                                print(f"  进程 {pid}: {mem_mb:.0f} MB")
                        except:
                            print(f"  进程 {pid}: {parts[4]}")
            else:
                print("  未找到Python进程")
        else:
            result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
            training_procs = [l for l in result.stdout.split('\n') 
                            if 'python' in l.lower() and ('train' in l.lower() or 'run_kgrec' in l.lower())]
            if training_procs:
                print(f"找到 {len(training_procs)} 个训练进程")
                for proc in training_procs:
                    print(f"  {proc[:100]}")
            else:
                print("  未找到训练进程")
    except Exception as e:
        print(f"  检查失败: {e}")
    
    print()
    
    # 2. 检查检查点文件
    print("【2】检查保存的模型文件:")
    print("-" * 70)
    weights_dir = Path('./weights/')
    if weights_dir.exists():
        ckpt_files = list(weights_dir.glob('*.ckpt'))
        if ckpt_files:
            latest = max(ckpt_files, key=lambda x: x.stat().st_mtime)
            stat = latest.stat()
            size_mb = stat.st_size / (1024 * 1024)
            mtime = datetime.fromtimestamp(stat.st_mtime)
            print(f"  找到 {len(ckpt_files)} 个检查点文件")
            print(f"  最新文件: {latest.name}")
            print(f"  大小: {size_mb:.2f} MB")
            print(f"  保存时间: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            print("  未找到检查点文件 (.ckpt)")
    else:
        print("  权重目录不存在 (./weights/)")
        print("  提示: 检查点会在找到更好的模型时自动保存")
    
    print()
    
    # 3. 检查日志文件
    print("【3】检查日志文件:")
    print("-" * 70)
    log_dir = Path('./logs/ml-20m/')
    if log_dir.exists():
        log_files = list(log_dir.glob('*.log'))
        if log_files:
            latest = max(log_files, key=lambda x: x.stat().st_mtime)
            stat = latest.stat()
            mtime = datetime.fromtimestamp(stat.st_mtime)
            print(f"  找到 {len(log_files)} 个日志文件")
            print(f"  最新文件: {latest.name}")
            print(f"  最后更新: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # 读取最后几行
            try:
                with open(latest, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                    if lines:
                        print(f"\n  最后3行日志:")
                        for line in lines[-3:]:
                            print(f"    {line.rstrip()}")
            except:
                pass
        else:
            print("  未找到日志文件")
            print("  提示: 训练时需要使用 --log --log_fn <name> 参数才会保存日志")
    else:
        print("  日志目录不存在")
    
    print()
    
    # 4. 建议
    print("【4】如何查看实时训练进度:")
    print("-" * 70)
    print("  方法1: 如果训练在终端中运行，直接查看终端输出")
    print("  方法2: 如果使用nohup后台运行:")
    print("         tail -f nohup.out  (在WSL/Linux中)")
    print("  方法3: 查看进程输出 (如果重定向了输出)")
    print("  方法4: 定期检查检查点文件的时间戳")
    print("  方法5: 下次训练时添加 --log --log_fn train 参数保存日志")
    
    print()
    print("=" * 70)
    print("检查完成")
    print("=" * 70)
    print()

if __name__ == '__main__':
    main()

