"""
检查训练进度脚本
用于查看当前训练的状态和进度
"""
import os
import sys
import subprocess
import re
from pathlib import Path
from datetime import datetime
import glob

def check_running_processes():
    """检查是否有正在运行的训练进程"""
    print("=" * 60)
    print("检查运行中的训练进程")
    print("=" * 60)
    
    try:
        # 在Windows/WSL中查找Python训练进程
        if sys.platform == 'win32':
            result = subprocess.run(
                ['wmic', 'process', 'where', "name='python.exe'", 'get', 'ProcessId,CommandLine,WorkingSetSize'],
                capture_output=True,
                text=True
            )
            lines = [line.strip() for line in result.stdout.split('\n') if line.strip()]
            
            python_processes = []
            for line in lines[1:]:  # 跳过标题行
                if 'python.exe' in line.lower() or 'python' in line.lower():
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            pid = parts[-1]
                            mem_kb = parts[-2] if parts[-2].isdigit() else 'N/A'
                            cmd = ' '.join(parts[:-2])
                            python_processes.append((pid, mem_kb, cmd))
                        except:
                            python_processes.append(('N/A', 'N/A', line))
            
            if python_processes:
                print(f"找到 {len(python_processes)} 个Python进程:")
                training_found = False
                for pid, mem, cmd in python_processes:
                    mem_mb = int(mem) / (1024 * 1024) if mem != 'N/A' and mem.isdigit() else 0
                    is_training = 'train' in cmd.lower() or 'run_kgrec' in cmd.lower() or mem_mb > 500
                    if is_training:
                        training_found = True
                        print(f"\n  [可能是训练进程]")
                    print(f"  PID: {pid}")
                    print(f"  内存: {mem_mb:.2f} MB" if mem_mb > 0 else f"  内存: {mem}")
                    print(f"  命令: {cmd[:100]}..." if len(cmd) > 100 else f"  命令: {cmd}")
                
                if not training_found:
                    print("\n提示: 如果训练进程在后台运行，可能无法直接看到命令")
            else:
                print("未找到运行中的Python进程")
        else:
            # Linux/WSL
            result = subprocess.run(
                ['ps', 'aux'],
                capture_output=True,
                text=True
            )
            lines = result.stdout.split('\n')
            training_processes = [
                line for line in lines 
                if 'python' in line.lower() and ('train' in line.lower() or 'run_kgrec' in line.lower())
            ]
            
            if training_processes:
                print(f"找到 {len(training_processes)} 个训练相关进程:")
                for proc in training_processes:
                    print(f"  {proc}")
            else:
                print("未找到运行中的训练进程")
    except Exception as e:
        print(f"检查进程时出错: {e}")
        print("提示: 可以手动运行 'tasklist | findstr python' 查看进程")
    
    print()

def check_log_files(dataset='ml-20m'):
    """检查日志文件"""
    print("=" * 60)
    print("检查日志文件")
    print("=" * 60)
    
    log_dir = Path(f'./logs/{dataset}/')
    
    if not log_dir.exists():
        print(f"日志目录不存在: {log_dir}")
        print("提示: 如果训练时没有使用 --log 参数，日志不会保存到文件")
        print()
        return
    
    log_files = list(log_dir.glob('*.log'))
    
    if not log_files:
        print(f"在 {log_dir} 中未找到日志文件")
        print("提示: 训练时需要使用 --log 和 --log_fn 参数才会保存日志")
        print()
        return
    
    print(f"找到 {len(log_files)} 个日志文件:")
    for log_file in sorted(log_files, key=lambda x: x.stat().st_mtime, reverse=True):
        stat = log_file.stat()
        size = stat.st_size
        mtime = datetime.fromtimestamp(stat.st_mtime)
        print(f"\n  文件: {log_file.name}")
        print(f"  大小: {size / 1024:.2f} KB")
        print(f"  最后修改: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 读取最后几行
        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                if lines:
                    print(f"  总行数: {len(lines)}")
                    print(f"  最后5行:")
                    for line in lines[-5:]:
                        print(f"    {line.rstrip()}")
        except Exception as e:
            print(f"  读取文件时出错: {e}")
    
    print()

def check_checkpoint_files():
    """检查检查点文件"""
    print("=" * 60)
    print("检查检查点文件")
    print("=" * 60)
    
    weights_dir = Path('./weights/')
    
    if not weights_dir.exists():
        print(f"权重目录不存在: {weights_dir}")
        print("提示: 检查点会在训练过程中保存，格式为 .ckpt 文件")
        print()
        return
    
    ckpt_files = list(weights_dir.glob('*.ckpt'))
    
    if not ckpt_files:
        print(f"在 {weights_dir} 中未找到检查点文件")
        print("提示: 检查点会在找到更好的模型时保存")
        print()
        return
    
    print(f"找到 {len(ckpt_files)} 个检查点文件:")
    for ckpt_file in sorted(ckpt_files, key=lambda x: x.stat().st_mtime, reverse=True):
        stat = ckpt_file.stat()
        size = stat.st_size / (1024 * 1024)  # MB
        mtime = datetime.fromtimestamp(stat.st_mtime)
        print(f"\n  文件: {ckpt_file.name}")
        print(f"  大小: {size:.2f} MB")
        print(f"  最后修改: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
    
    print()

def extract_progress_from_logs(dataset='ml-20m'):
    """从日志文件中提取训练进度信息"""
    print("=" * 60)
    print("提取训练进度信息")
    print("=" * 60)
    
    log_dir = Path(f'./logs/{dataset}/')
    
    if not log_dir.exists():
        print("日志目录不存在，无法提取进度信息")
        print()
        return
    
    log_files = list(log_dir.glob('*.log'))
    
    if not log_files:
        print("未找到日志文件，无法提取进度信息")
        print()
        return
    
    # 读取最新的日志文件
    latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
    print(f"读取最新日志文件: {latest_log.name}\n")
    
    try:
        with open(latest_log, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
            # 提取epoch信息
            epoch_pattern = r'epoch\s+(\d+)'
            epochs = re.findall(epoch_pattern, content, re.IGNORECASE)
            
            if epochs:
                max_epoch = max([int(e) for e in epochs])
                print(f"最新训练轮数: Epoch {max_epoch}")
            
            # 提取recall信息
            recall_pattern = r'recall.*?\[([\d.]+)'
            recalls = re.findall(recall_pattern, content, re.IGNORECASE)
            
            if recalls:
                latest_recall = recalls[-1]
                print(f"最新Recall@20: {latest_recall}")
            
            # 提取最后几行包含epoch的信息
            lines = content.split('\n')
            epoch_lines = [line for line in lines if 'epoch' in line.lower() or 'Epoch' in line]
            
            if epoch_lines:
                print(f"\n最近的训练记录 (最后3条):")
                for line in epoch_lines[-3:]:
                    print(f"  {line.strip()}")
            
    except Exception as e:
        print(f"读取日志文件时出错: {e}")
    
    print()

def check_output_files():
    """检查可能的输出文件"""
    print("=" * 60)
    print("检查其他相关文件")
    print("=" * 60)
    
    # 检查是否有nohup.out或其他输出文件
    output_files = []
    for pattern in ['nohup.out', '*.out', 'train_*.log', 'output*.txt']:
        output_files.extend(glob.glob(pattern))
    
    if output_files:
        print(f"找到 {len(output_files)} 个可能的输出文件:")
        for out_file in sorted(output_files, key=lambda x: os.path.getmtime(x), reverse=True):
            stat = os.stat(out_file)
            size = stat.st_size / 1024  # KB
            mtime = datetime.fromtimestamp(stat.st_mtime)
            print(f"\n  文件: {out_file}")
            print(f"  大小: {size:.2f} KB")
            print(f"  最后修改: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # 读取最后几行
            try:
                with open(out_file, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                    if lines:
                        print(f"  最后3行:")
                        for line in lines[-3:]:
                            print(f"    {line.rstrip()}")
            except:
                pass
    else:
        print("未找到其他输出文件")
    
    print()

def main():
    print("\n" + "=" * 60)
    print("训练进度检查工具")
    print("=" * 60 + "\n")
    
    # 检查运行中的进程
    check_running_processes()
    
    # 检查日志文件
    check_log_files()
    
    # 检查检查点文件
    check_checkpoint_files()
    
    # 从日志提取进度
    extract_progress_from_logs()
    
    # 检查其他输出文件
    check_output_files()
    
    print("=" * 60)
    print("检查完成")
    print("=" * 60)
    print("\n提示:")
    print("1. 如果训练进程正在运行，可以使用以下命令查看实时输出:")
    print("   - 在WSL中: tail -f nohup.out (如果使用了nohup)")
    print("   - 或者直接查看进程的标准输出")
    print("2. 如果训练时使用了 --log 参数，日志会保存在 ./logs/{dataset}/ 目录")
    print("3. 检查点文件会保存在 ./weights/ 目录，格式为 .ckpt")
    print("4. 训练进度信息包括: Epoch数、Loss、Recall@20、NDCG等指标")
    print()

if __name__ == '__main__':
    main()

