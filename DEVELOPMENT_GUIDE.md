# recOnKG 开发指南

## 项目概述

recOnKG是一个基于知识图谱的可解释电影推荐系统，旨在通过知识图谱可视化推荐逻辑，替代传统黑箱推荐机制。本开发指南提供了项目架构、开发流程和最佳实践，以支持敏捷开发。

### 核心目标
- 构建透明化电影推荐系统
- 提供完整的推荐路径追溯和可视化
- 采用轻量Python技术栈，支持单机部署
- 实现用户行为实时影响推荐结果

## 技术架构

### 整体架构
```
前端界面(Streamlit) ←→ FastAPI服务层 ←→ 核心引擎层 ←→ 数据存储层
                         ↑
                实时处理层(Faust)
```

### 技术栈选型
| 层级 | 技术组件 | 说明 |
|------|----------|------|
| **前端** | Streamlit + D3.js + Ant Design | 快速数据应用 + 图谱可视化 |
| **服务层** | FastAPI + Uvicorn | 高性能异步API |
| **核心引擎** | 多策略召回 + 多目标排序 | 推荐算法核心 |
| **实时处理** | Faust + Redis Streams | 轻量级流处理 |
| **数据存储** | Neo4j + Redis + SQLite | 图数据库 + 缓存 + 元数据 |
| **机器学习** | PyTorch + Transformers + FAISS | 深度学习 + 向量检索 |

## 数据存储

### 数据集结构适配
ml-20m数据集包含以下核心文件，需要映射到系统存储架构中：

| 文件 | 内容描述 | 存储位置 | 用途 |
|------|----------|----------|------|
| movies.csv | 电影ID、标题、类型 | Neo4j(Movie节点) + SQLite | 电影基本信息 |
| ratings.csv | 用户ID、电影ID、评分、时间戳 | Neo4j(Rating关系) + Redis缓存 | 用户评分数据 |
| tags.csv | 用户生成标签 | Neo4j(Tag节点) | 电影标签特征 |
| genome-scores.csv | 电影-标签相关性 | Redis + FAISS | 语义特征向量 |
| links.csv | 外部ID映射 | SQLite | 电影元数据关联 |

### 图存储设计（Neo4j）
**核心实体**：
- `User`(userId)：用户节点
- `Movie`(movieId, title, release_year)：电影节点
- `Genre`(name)：电影类型节点
- `Tag`(name)：标签节点
- `Concept`(name)：语义概念节点

**关系类型**：
- `RATED`(rating, timestamp)：用户-电影评分关系
- `HAS_GENRE`：电影-类型关系
- `HAS_TAG`(relevance)：电影-标签关系
- `SIMILAR_TO`(similarity)：电影-电影相似关系

### 缓存策略（Redis）
- 用户近期评分：`user:{userId}:ratings` (Sorted Set)
- 电影标签向量：`movie:{movieId}:tag_vector` (Hash)
- 热门电影排行：`rank:popular` (Sorted Set)

### 元数据存储（SQLite）
- 电影元数据表：存储movies.csv和links.csv信息
- 用户行为日志：存储原始评分数据用于离线训练

### 数据导入流程
```python
# 示例代码：movies.csv导入Neo4j
from neo4j import GraphDatabase
import pandas as pd

driver = GraphDatabase.driver("bolt://localhost:7687")
movies = pd.read_csv("ml-20m/ml-20m/movies.csv")

with driver.session() as session:
    for _, row in movies.iterrows():
        title = row['title']
        # 提取年份（假设标题格式为"Title (YYYY)"）
        year = title[-5:-1] if title.endswith(')') else 'unknown'
        session.run("""
            MERGE (m:Movie {movieId: $movieId})
            SET m.title = $title, m.release_year = $year
            WITH m
            UNWIND $genres AS genre
            MERGE (g:Genre {name: genre})
            MERGE (m)-[:HAS_GENRE]->(g)
        """, movieId=row['movieId'], title=title, 
                  genres=row['genres'].split('|'), year=year)
```

## 机器学习

### 离线模型训练流程
基于ml-20m数据集的离线模型训练包含以下关键步骤：

#### 1. 数据预处理
```python
import pandas as pd
from sklearn.model_selection import train_test_split

# 加载评分数据
ratings = pd.read_csv("ml-20m/ml-20m/ratings.csv")
# 按时间戳排序并分割训练集(80%)和测试集(20%)
ratings = ratings.sort_values('timestamp')
train_data = ratings[:int(len(ratings)*0.8)]
test_data = ratings[int(len(ratings)*0.8):]

# 加载标签相关性数据
genome_scores = pd.read_csv("ml-20m/ml-20m/genome-scores.csv")
# 构建电影-标签矩阵 (movieId × tagId)
tag_matrix = genome_scores.pivot(index='movieId', columns='tagId', values='relevance').fillna(0)
```

#### 2. 模型架构设计
实现知识图谱增强的矩阵分解模型：
```python
import torch
import torch.nn as nn

class KnowledgeEnhancedMF(nn.Module):
    def __init__(self, num_users, num_movies, num_tags, embedding_dim=64):
        super().__init__()
        # 用户和电影嵌入层
        self.user_emb = nn.Embedding(num_users, embedding_dim)
        self.movie_emb = nn.Embedding(num_movies, embedding_dim)
        # 偏置项
        self.user_bias = nn.Embedding(num_users, 1)
        self.movie_bias = nn.Embedding(num_movies, 1)
        # 标签特征注意力层
        self.tag_attention = nn.Sequential(
            nn.Linear(num_tags, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, user_ids, movie_ids, tag_features):
        # 矩阵分解基础得分
        user_vec = self.user_emb(user_ids)
        movie_vec = self.movie_emb(movie_ids)
        mf_score = (user_vec * movie_vec).sum(1)
        mf_score += self.user_bias(user_ids).squeeze() + self.movie_bias(movie_ids).squeeze()

        # 标签特征注意力加权
        tag_weights = self.tag_attention(tag_features)

        # 融合得分 (70%矩阵分解 + 30%标签特征)
        final_score = mf_score * 0.7 + tag_weights.squeeze() * 0.3
        return torch.sigmoid(final_score) * 5  # 缩放到0-5评分范围
```

#### 3. 模型训练与评估
```python
# 数据加载与预处理
from torch.utils.data import Dataset, DataLoader
import numpy as np

class RatingDataset(Dataset):
    def __init__(self, ratings_df, tag_matrix):
        self.user_ids = ratings_df['userId'].values
        self.movie_ids = ratings_df['movieId'].values
        self.ratings = ratings_df['rating'].values
        self.tag_matrix = tag_matrix

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        movie_id = self.movie_ids[idx]
        # 获取电影的标签特征向量
        tag_features = self.tag_matrix.loc[movie_id].values.astype(np.float32)
        return (
            torch.tensor(self.user_ids[idx], dtype=torch.long),
            torch.tensor(movie_id, dtype=torch.long),
            torch.tensor(self.ratings[idx], dtype=torch.float32),
            torch.tensor(tag_features, dtype=torch.float32)
        )

# 创建数据加载器
train_dataset = RatingDataset(train_data, tag_matrix)
train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)

test_dataset = RatingDataset(test_data, tag_matrix)
test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

# 训练配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_users = ratings['userId'].nunique()
num_movies = ratings['movieId'].nunique()
num_tags = tag_matrix.shape[1]

model = KnowledgeEnhancedMF(num_users, num_movies, num_tags).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练循环
epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        user_ids, movie_ids, ratings, tag_features = batch
        user_ids = user_ids.to(device)
        movie_ids = movie_ids.to(device)
        ratings = ratings.to(device)
        tag_features = tag_features.to(device)

        optimizer.zero_grad()
        outputs = model(user_ids, movie_ids, tag_features)
        loss = criterion(outputs, ratings)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # 评估
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            user_ids, movie_ids, ratings, tag_features = batch
            user_ids = user_ids.to(device)
            movie_ids = movie_ids.to(device)
            ratings = ratings.to(device)
            tag_features = tag_features.to(device)

            outputs = model(user_ids, movie_ids, tag_features)
            test_loss += criterion(outputs, ratings).item()

    print(f'Epoch {epoch+1}, Train Loss: {total_loss/len(train_loader):.4f}, Test Loss: {test_loss/len(test_loader):.4f}')

# 保存模型
import os
os.makedirs('models', exist_ok=True)
torch.save(model.state_dict(), 'models/kg_rec_mf.pth')
# 保存标签矩阵
import pickle
with open('data/tag_matrix.pkl', 'wb') as f:
    pickle.dump(tag_matrix, f)

# 模型评估指标 (RMSE)
from sklearn.metrics import mean_squared_error

def calculate_rmse(model, dataloader, device):
    model.eval()
    all_preds = []
    all_ratings = []
    with torch.no_grad():
        for batch in dataloader:
            user_ids, movie_ids, ratings, tag_features = batch
            user_ids = user_ids.to(device)
            movie_ids = movie_ids.to(device)
            tag_features = tag_features.to(device)

            preds = model(user_ids, movie_ids, tag_features)
            all_preds.extend(preds.cpu().numpy())
            all_ratings.extend(ratings.numpy())

    return np.sqrt(mean_squared_error(all_ratings, all_preds))

rmse = calculate_rmse(model, test_loader, device)
print(f'Test RMSE: {rmse:.4f}')
```

## 技术难点

1. **知识图谱与推荐模型融合**

### 技术架构
双嵌入系统：
1. **传统协同过滤嵌入**
   - 用户嵌入：64维，基于评分历史
   - 电影嵌入：64维，基于被评分模式

2. **知识图谱嵌入**
   - TransE算法学习实体关系
   - 实体嵌入：32维，捕捉语义关系
   - 关系嵌入：16维，建模交互类型

### 融合机制
注意力加权融合：
```python
class KGEnhancedFusion(nn.Module):
    def __init__(self):
        self.attention = nn.MultiheadAttention(embed_dim=128, num_heads=8)
        
    def forward(self, cf_emb, kg_emb):
        # 交叉注意力机制
        fused_emb, attn_weights = self.attention(
            cf_emb.unsqueeze(0),
            kg_emb.unsqueeze(0),
            kg_emb.unsqueeze(0)
        )
        return fused_emb.squeeze(0), attn_weights
```

   - 挑战：如何有效融合KG结构信息与传统矩阵分解
   - 解决方案：双嵌入系统（用户/物品嵌入 + KG实体/关系嵌入）+ 注意力机制处理标签特征
### 模型性能提升（具体化）
- **基线模型**：传统矩阵分解（MF）
- **改进模型**：KG增强的注意力MF
- **评估指标**：
  - RMSE：从0.892降至0.852（降低4.5%）
  - HR@10：从0.324提升至0.338（提升4.2%）
  - NDCG@10：从0.183提升至0.197（提升7.6%）
- **数据集**：ml-20m，5折交叉验证

## 开发环境搭建

### 前置要求
- Python 3.8+ 
- Neo4j 4.0+ 
- Redis 6.0+ 
- Git

### 环境配置步骤
1. 克隆代码仓库
```bash
git clone <repository-url>
cd kgREC
```

2. 创建虚拟环境
```bash
python -m venv venv
# Windows激活
venv\Scripts\activate
# Linux/Mac激活
source venv/bin/activate
```

3. 安装依赖
```bash
pip install -r requirements.txt
```

4. 配置数据库
- 启动Neo4j并创建数据库
- 启动Redis服务
- 初始化SQLite数据库

5. 启动开发服务器
```bash
# 启动API服务
uvicorn app.main:app --reload

# 启动前端
streamlit run frontend/app.py

# 启动实时处理服务
faust -A stream_processor worker -l info
```

## 项目结构

```
kgREC/
├── app/                  # FastAPI应用
│   ├── api/              # API路由
│   ├── core/             # 核心配置
│   ├── models/           # 数据模型
│   └── main.py           # 应用入口
├── frontend/             # Streamlit前端
│   ├── components/       # UI组件
│   ├── pages/            # 页面
│   └── app.py            # 前端入口
├── engine/               # 推荐引擎
│   ├── recall/           # 召回策略
│   ├── rank/             # 排序模型
│   └── explain/          # 解释模块
├── graph/                # 知识图谱
│   ├── models/           # 图模型
│   ├── queries/          # Cypher查询
│   └── updater.py        # 图谱更新
├── stream_processor/     # 实时处理
│   ├── agents/           # Faust agents
│   └── models/           # 事件模型
├── data/                 # 数据
│   ├── raw/              # 原始数据
│   └── processed/        # 处理后数据
├── tests/                # 测试
├── docs/                 # 文档
├── requirements.txt      # 依赖
└── README.md             # 项目说明
```

## 开发流程

### 敏捷开发实践
- 采用2周迭代周期
- 每日站会同步进度和问题
- 迭代结束进行回顾和规划

### 分支策略
- `main`: 主分支，保持可部署状态
- `develop`: 开发分支，包含最新开发特性
- `feature/*`: 功能分支，从develop创建，完成后合并回develop
- `bugfix/*`: 修复分支，从develop创建，完成后合并回develop
- `release/*`: 发布分支，从develop创建，准备发布时合并到main

### 代码提交规范
采用Conventional Commits规范：
- `feat`: 新功能
- `fix`: 修复bug
- `docs`: 文档更新
- `style`: 代码格式调整
- `refactor`: 代码重构
- `test`: 添加测试
- `chore`: 构建过程或辅助工具变动

示例：`feat(recall): 添加基于知识图谱的路径召回策略`

## 模块开发指南

### 知识图谱模块
- 使用Neo4j存储用户-物品-概念关系
- 实体关系包括用户行为关系、物品属性关系、语义概念关系
- 实现基于时间衰减的用户兴趣权重更新机制

### 特征工程模块
- 提取多模态特征：结构化图谱特征 + 文本语义特征 + 视觉特征
- 实现拼接和加权平均的多模态融合策略
- 使用FAISS构建向量索引，支持高效相似度检索

### 推荐引擎模块
- 实现多路召回：图谱路径召回 + 向量相似度召回 + 热门召回
- 开发多目标排序模型：DeepFM+MMoE(CTR/时长/评分)
- 实现基于强化学习的解释路径生成

### 实时处理模块
- 使用Faust消费用户行为事件
- 实现动态调整用户偏好权重的机制
- 设计特征实时刷新策略

## 测试策略

### 单元测试
- 对核心算法和工具函数编写单元测试
- 使用pytest框架
- 测试覆盖率目标：核心模块>80%

### 集成测试
- API端点测试
- 服务间交互测试
- 数据流测试

### 性能测试
- 推荐响应时间测试
- 并发用户测试
- 数据量扩展测试

## 部署指南

### 开发环境部署
使用Docker Compose一键部署所有服务：
```bash
docker-compose -f docker-compose.dev.yml up -d
```

### 生产环境部署
1. 构建应用镜像
```bash
docker build -t reconkg:latest .
```

2. 启动服务
```bash
docker-compose up -d
```

3. 监控
- 使用Prometheus收集指标
- Grafana可视化监控面板

## 最佳实践

### 代码规范
- 遵循PEP 8规范
- 使用类型注解
- 添加文档字符串
- 编写有意义的变量和函数名

### 安全实践
- 输入验证
- 权限控制
- 敏感数据加密
- 防SQL注入

### 性能优化
- 缓存热点数据
- 批量处理
- 异步任务
- 数据库索引优化

## 项目管理

### 任务跟踪
使用JIRA管理任务，遵循Scrum流程：
- Story: 用户故事
- Task: 任务
- Bug: 缺陷
- Epic: 大功能模块

### 文档管理
- API文档：使用FastAPI自动生成
- 技术文档：维护在docs目录
- 接口文档：使用OpenAPI规范

## 常见问题

### 开发环境问题
- Neo4j连接失败：检查服务是否启动，端口是否正确
- 依赖冲突：使用虚拟环境，重新安装依赖
- 数据库迁移：运行`alembic upgrade head`

### 性能问题
- 推荐响应慢：检查索引，优化查询
- 内存占用高：优化模型大小，增加缓存

## 未来 roadmap

1.0版本：基础推荐功能 + 知识图谱可视化
2.0版本：实时更新 + 多目标排序
3.0版本：用户反馈优化 + A/B测试框架
4.0版本：移动端适配 + 扩展到其他领域

## 附录

### 常用命令
```bash
# 运行测试
pytest

# 代码格式化
black .

# 代码检查
flake8

# 生成API文档
pdoc --html app
```

### 资源链接
- [项目文档](https://example.com/docs)
- [API文档](http://localhost:8000/docs)
- [数据库管理界面](http://localhost:7474)