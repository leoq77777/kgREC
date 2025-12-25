"""
安全停止训练脚本
在停止训练前尝试保存当前状态
"""
import os
import sys
import subprocess
import signal
import time
from pathlib import Path

def find_training_process():
    """查找训练进程"""
    try:
        if sys.platform == 'win32':
            # Windows
            result = subprocess.run(
                ['wmic', 'process', 'where', "name='python.exe'", 'get', 'ProcessId,CommandLine,WorkingSetSize'],
                capture_output=True,
                text=True
            )
            lines = [l.strip() for l in result.stdout.split('\n') if l.strip()]
            
            training_processes = []
            for line in lines[1:]:  # 跳过标题
                if 'run_kgrec' in line.lower() or 'train' in line.lower():
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            pid = int(parts[-1])
                            mem_kb = parts[-2] if parts[-2].isdigit() else '0'
                            mem_mb = int(mem_kb) / (1024 * 1024) if mem_kb != '0' else 0
                            if mem_mb > 500:  # 大于500MB可能是训练进程
                                training_processes.append((pid, mem_mb, line))
                        except:
                            pass
            return training_processes
        else:
            # Linux/WSL
            result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
            lines = result.stdout.split('\n')
            training_processes = []
            for line in lines:
                if 'python' in line.lower() and ('run_kgrec' in line.lower() or 'train' in line.lower()):
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            pid = int(parts[1])
                            mem_mb = float(parts[5]) if len(parts) > 5 else 0
                            training_processes.append((pid, mem_mb, line))
                        except:
                            pass
            return training_processes
    except Exception as e:
        print(f"查找进程时出错: {e}")
        return []

def check_checkpoints():
    """检查是否有检查点文件"""
    weights_dir = Path('./weights/')
    if not weights_dir.exists():
        return []
    
    ckpt_files = list(weights_dir.glob('*.ckpt'))
    return sorted(ckpt_files, key=lambda x: x.stat().st_mtime, reverse=True)

def main():
    print("=" * 70)
    print("安全停止训练工具")
    print("=" * 70)
    print()
    
    # 查找训练进程
    print("【1】查找训练进程...")
    processes = find_training_process()
    
    if not processes:
        print("  未找到训练进程")
        print("  训练可能已经停止或不在运行")
        return
    
    print(f"  找到 {len(processes)} 个可能的训练进程:")
    for i, (pid, mem_mb, cmd) in enumerate(processes, 1):
        print(f"    [{i}] PID: {pid}, 内存: {mem_mb:.0f} MB")
        print(f"        命令: {cmd[:80]}...")
    print()
    
    # 检查检查点
    print("【2】检查已保存的检查点...")
    checkpoints = check_checkpoints()
    if checkpoints:
        print(f"  找到 {len(checkpoints)} 个检查点文件:")
        for ckpt in checkpoints[:5]:  # 只显示最新的5个
            stat = ckpt.stat()
            size_mb = stat.st_size / (1024 * 1024)
            mtime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(stat.st_mtime))
            print(f"    - {ckpt.name} ({size_mb:.2f} MB, {mtime})")
    else:
        print("  未找到检查点文件")
        print("  警告: 停止训练可能会丢失当前进度！")
    print()
    
    # 确认
    print("【3】确认停止训练")
    print("  注意:")
    print("  - 如果训练代码支持，会尝试保存当前状态")
    print("  - 但无法保证能完全保存当前epoch的进度")
    print("  - 建议等待当前epoch完成后再停止")
    print()
    
    response = input("是否继续停止训练? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("已取消")
        return
    
    # 停止进程
    print()
    print("【4】停止训练进程...")
    for pid, mem_mb, cmd in processes:
        try:
            if sys.platform == 'win32':
                print(f"  正在停止进程 {pid}...")
                subprocess.run(['taskkill', '/PID', str(pid), '/F'], 
                             capture_output=True)
            else:
                print(f"  正在发送SIGTERM信号到进程 {pid}...")
                os.kill(pid, signal.SIGTERM)
                time.sleep(2)
                # 如果还在运行，强制终止
                try:
                    os.kill(pid, 0)  # 检查进程是否还在
                    print(f"  进程仍在运行，发送SIGKILL...")
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    print(f"  进程 {pid} 已停止")
        except Exception as e:
            print(f"  停止进程 {pid} 时出错: {e}")
    
    print()
    print("=" * 70)
    print("训练已停止")
    print("=" * 70)
    print()
    print("提示:")
    print("1. 检查最新的检查点文件在 ./weights/ 目录")
    print("2. 下次训练时使用 --resume <检查点路径> 恢复训练")
    print("3. 例如: python train_with_rocm.py --resume ./weights/train_ml-20m.ckpt")
    print()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n操作被用户取消")
        sys.exit(0)

