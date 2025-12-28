# 云部署检查清单

## ✅ 部署前检查

- [ ] 代码已测试，本地可以运行
- [ ] 所有依赖已列出在 requirements-cloud.txt
- [ ] 数据文件已准备或可以重新生成
- [ ] 已选择云平台并创建账号

## 📦 需要上传的文件

### 核心代码
- [x] run_kgrec.py
- [x] train_with_rocm.py
- [x] prepare_data_for_kgrec.py
- [x] modules/ (所有模型代码)
- [x] utils/ (所有工具函数)

### 配置文件
- [x] requirements-cloud.txt
- [x] .gitignore

### 部署脚本
- [x] setup_cloud_env.sh
- [x] train_cloud.sh
- [x] upload_to_cloud.sh
- [x] download_results.sh

### 文档
- [x] README_CLOUD.md
- [x] 云训练迁移指南.md

### 数据（可选）
- [ ] ml-20m/ (如果数据很大，可以在云上重新生成)

## 🚀 部署步骤

1. **上传代码到GitHub**
   ```bash
   git add .
   git commit -m "feat: 添加云训练部署支持"
   git push origin master
   ```

2. **在云服务器上克隆**
   ```bash
   git clone https://github.com/leoq77777/kgREC.git
   cd kgREC
   ```

3. **设置环境**
   ```bash
   bash setup_cloud_env.sh
   ```

4. **准备数据**
   ```bash
   python prepare_data_for_kgrec.py
   ```

5. **开始训练**
   ```bash
   bash train_cloud.sh
   ```

## 📝 注意事项

- 确保云服务器有足够的存储空间（至少50GB）
- 使用screen/tmux保持训练会话
- 定期检查检查点文件
- 训练完成后及时停止实例以节省费用

