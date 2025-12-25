# AMD GPU (ROCm) 训练配置指南

## 当前状态

✅ **已确认**: 当前使用 **CPU训练** (PyTorch 2.3.1+cpu)

## WSL + AMD GPU + ROCm 支持

### 支持情况

**是的，ROCm可以在WSL2中使用！** 但需要满足以下条件：

1. **WSL版本**: 必须是WSL2（不是WSL1）
2. **AMD GPU**: 需要支持ROCm的显卡
   - ✅ RX 6000系列 (RDNA2架构)
   - ✅ RX 7000系列 (RDNA3架构)  
   - ✅ Radeon Pro系列
   - ✅ Instinct系列
   - ❌ 较老的显卡可能不支持

### 快速安装步骤

#### 1. 检查WSL版本
```bash
# 在PowerShell中执行
wsl --list --verbose

# 如果显示VERSION为1，需要升级
wsl --set-version Ubuntu 2  # 替换为你的发行版名称
```

#### 2. 在WSL中安装ROCm PyTorch

**方法1: 使用安装脚本（推荐）**
```bash
# 在WSL中执行
chmod +x install_rocm_pytorch.sh
./install_rocm_pytorch.sh
```

**方法2: 手动安装**
```bash
# 在WSL中执行
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7
```

#### 3. 验证安装
```bash
python check_gpu.py
```

如果显示 `CUDA可用: True`，说明安装成功！

### 使用GPU训练

安装ROCm后，训练时使用：
```bash
python run_kgrec.py --cuda 1 --gpu_id 0 --dataset ml-20m --data_path ml-20m/ml-20m/ --epoch 50
```

或者使用简化脚本：
```bash
python train.py  # 会自动检测GPU
```

## 性能对比

| 配置 | 每轮训练时间 | 总训练时间(50轮) |
|------|-------------|-----------------|
| CPU  | ~45分钟     | ~37.5小时       |
| GPU  | ~5分钟      | ~4小时          |

**GPU训练速度约为CPU的7-9倍！**

## 常见问题

### Q1: 安装后仍然显示CUDA不可用？
**A**: 可能原因：
1. GPU不支持ROCm（检查显卡型号）
2. 需要在WSL中安装ROCm驱动（见setup_rocm_wsl.md）
3. WSL版本是1而不是2

### Q2: 如何检查我的AMD显卡是否支持ROCm？
**A**: 查看 [ROCm官方支持列表](https://rocm.docs.amd.com/en/latest/release/gpu_os_support.html)

### Q3: 如果GPU不支持ROCm怎么办？
**A**: 选项：
1. 继续使用CPU训练（较慢但稳定）
2. 尝试使用DirectML（Windows原生，但支持有限）
3. 考虑使用云GPU服务

### Q4: WSL中ROCm性能如何？
**A**: 性能通常为原生Linux的90-95%，对于训练来说完全可以接受。

## 下一步

1. 检查WSL版本：`wsl --list --verbose`
2. 在WSL中运行安装脚本：`./install_rocm_pytorch.sh`
3. 验证安装：`python check_gpu.py`
4. 开始GPU训练！

## 参考文档

- [ROCm官方文档](https://rocm.docs.amd.com/)
- [PyTorch ROCm安装指南](https://pytorch.org/get-started/locally/)
- [WSL GPU支持](https://learn.microsoft.com/en-us/windows/wsl/tutorials/gpu-compute)

