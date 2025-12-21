import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import contextlib
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.cuda.amp as amp
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os
import pickle

# 设置中文显示
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["Microsoft YaHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

class RatingDataset(Dataset):
    def __init__(self, ratings_df, tag_matrix, user_features=None, include_tags=True, model_type='mf', user_feature_dim=3, item_feature_dim=3):
        self.user_feature_dim = user_feature_dim
        self.item_feature_dim = item_feature_dim
        self.include_tags = include_tags
        self.user_features = user_features
        self.model_type = model_type
        self.user_ids = ratings_df['userId'].values
        self.movie_ids = ratings_df['movieId'].values
        self.ratings = ratings_df['rating'].values
        self.tag_matrix = tag_matrix

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        user_id = self.user_ids[idx]
        movie_id = self.movie_ids[idx]
        rating = self.ratings[idx]
        
        # 获取电影的标签特征向量，如果电影ID不存在则返回零向量
        if self.include_tags and movie_id in self.tag_matrix.index:
            tag_features = self.tag_matrix.loc[movie_id].values.astype(np.float32)
        else:
            if self.model_type == 'dual_tower':
                # For dual tower model, ensure at least 3 dimensions if tag matrix is empty
                dim = self.tag_matrix.shape[1] if self.tag_matrix.shape[1] > 0 else 3
                tag_features = np.zeros(dim, dtype=np.float32)
            else:
                tag_features = np.zeros(self.tag_matrix.shape[1] if self.include_tags else 0, dtype=np.float32)
        
        # 获取用户特征
        user_feat = np.zeros(3, dtype=np.float32)  # 默认3个用户特征
        if self.user_features and user_id in self.user_features:
            user_stats = self.user_features[user_id]
            user_feat = np.array([user_stats['mean'], user_stats['count'], user_stats['std']], dtype=np.float32)
        
        if self.model_type == 'dual_tower':
            return (
                torch.tensor(user_feat, dtype=torch.float32),
                torch.tensor(tag_features, dtype=torch.float32),
                torch.tensor(rating, dtype=torch.float32)
            )
        elif self.include_tags:
            return (
                torch.tensor(user_id, dtype=torch.long),
                torch.tensor(movie_id, dtype=torch.long),
                torch.tensor(rating, dtype=torch.float32),
                torch.tensor(tag_features, dtype=torch.float32)
            )
        else:
            return (
                torch.tensor(user_id, dtype=torch.long),
                torch.tensor(movie_id, dtype=torch.long),
                torch.tensor(rating, dtype=torch.float32)
            )



def preprocess_data(graph_path):
    import networkx as nx
    from torch_geometric.utils import from_networkx
    # 使用NetworkX加载图并转换为PyTorch Geometric格式
    import pickle
    with open(graph_path, 'rb') as f:
        graph = pickle.load(f)
    # 标准化节点属性
    node_attrs = set()
    for node, data in graph.nodes(data=True):
        node_attrs.update(data.keys())
    for node, data in graph.nodes(data=True):
        for attr in node_attrs:
            if attr not in data:
                data[attr] = 0
    # 标准化边属性
    edge_attrs = set()
    for u, v, data in graph.edges(data=True):
        edge_attrs.update(data.keys())
    for u, v, data in graph.edges(data=True):
        for attr in edge_attrs:
            if attr not in data:
                data[attr] = 0
    data = from_networkx(graph)
    edge_index = data.edge_index
    print(f"成功提取edge_index，形状: {edge_index.shape}")
    """从知识图谱加载数据并预处理"""
    import networkx as nx
    import pickle
    from collections import defaultdict
    
    # 加载知识图谱
    with open(graph_path, 'rb') as f:
        graph = pickle.load(f)
    
    # 提取评分数据
    ratings = []
    for u, v, data in graph.edges(data=True):
        if data.get('type') == 'RATED':
            # 提取用户ID和电影ID（去除前缀）
            # 处理可能包含小数点的ID格式
            user_id = int(float(u.replace('user_', '')))
            movie_id = int(float(v.replace('movie_', '')))
            ratings.append({
                'userId': user_id,
                'movieId': movie_id,
                'rating': data.get('rating', 0.0),
                'timestamp': data.get('timestamp', 0)
            })
    ratings_df = pd.DataFrame(ratings)
    
    # 提取标签数据并创建名称到ID的映射
    tag_names = set()
    # 首先收集所有唯一标签名称
    for u, v, data in graph.edges(data=True):
        if data.get('type') == 'HAS_TAG':
            tag_name = v.replace('tag_', '')
            tag_names.add(tag_name)
    
    # 创建标签名称到ID的映射
    tag_name_to_id = {name: i for i, name in enumerate(tag_names)}
    num_tags = len(tag_name_to_id)
    
    # 构建电影-标签数据
    tag_data = defaultdict(list)
    for u, v, data in graph.edges(data=True):
        if data.get('type') == 'HAS_TAG':
            movie_id = int(float(u.replace('movie_', '')))
            tag_name = v.replace('tag_', '')
            tag_id = tag_name_to_id[tag_name]
            tag_data[movie_id].append(tag_id)
    
    # 构建标签矩阵
    tag_matrix = pd.DataFrame.from_dict(tag_data, orient='index')
    if not tag_matrix.empty:
        tag_matrix = tag_matrix.apply(lambda x: x.value_counts(), axis=1).fillna(0)
    
    # 分割训练集和测试集
    if not ratings_df.empty:
        ratings_df = ratings_df.sort_values('timestamp')
        train_data = ratings_df[:int(len(ratings_df)*0.8)]
        test_data = ratings_df[int(len(ratings_df)*0.8):]
    else:
        train_data, test_data = pd.DataFrame(), pd.DataFrame()
    
    # 提取用户特征
    user_features = None
    if not ratings_df.empty:
        # 创建用户评分统计特征
        user_ratings = ratings_df.groupby('userId')['rating'].agg(['mean', 'count', 'std']).fillna(0)
        # 标准化特征
        user_ratings = (user_ratings - user_ratings.mean()) / (user_ratings.std() + 1e-6)
        user_features = user_ratings.to_dict('index')
    
    return train_data, test_data, tag_matrix, ratings_df, edge_index, user_features

def train_model(graph_path, model_save_path, epochs=10, batch_size=32, embedding_dim=64, model_type='mf'):
    # 初始化默认特征变量
    user_features = None
    item_features = None
    labels = None
    # 数据预处理
    # 获取图数据
    train_data, test_data, tag_matrix, all_ratings, edge_index, user_features = preprocess_data(graph_path)
    # 从评分数据中提取物品ID，确保数据不为空
    if not all_ratings.empty and 'movieId' in all_ratings.columns:
        item_ids = all_ratings['movieId'].unique()
    else:
        # 处理空数据情况
        item_ids = np.array([])
    print(f"在train_model中接收到edge_index，形状: {edge_index.shape}")
   
    # 创建数据加载器
    # 根据模型类型决定是否包含标签特征
    include_tags = (model_type != 'lightgcn')
    
    # 计算特征维度
    user_feature_dim = 3  # 默认用户特征维度
    item_feature_dim = 3  # 默认物品特征维度
    if model_type == 'dual_tower':
        # 安全处理用户特征维度计算
        user_feature_dim = 3
        if user_features is not None:
            if isinstance(user_features, dict):
                # 从字典中获取第一个用户的特征长度
                if user_features and len(user_features) > 0:
                    first_user_id = next(iter(user_features.keys()))
                    user_feature_dim = len(user_features[first_user_id])
            elif hasattr(user_features, 'shape'):
                # 处理数组/矩阵类型
                user_feature_dim = user_features.shape[1]
            elif isinstance(user_features, list):
                # 处理列表类型
                user_feature_dim = len(user_features[0]) if user_features else 3
        item_feature_dim = tag_matrix.shape[1] if tag_matrix is not None and not tag_matrix.empty else 3
    
    # 为双塔模型准备用户特征
    train_dataset = RatingDataset(train_data, tag_matrix, user_features, include_tags, model_type, user_feature_dim, item_feature_dim)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    
    test_dataset = RatingDataset(test_data, tag_matrix, user_features, include_tags, model_type, user_feature_dim, item_feature_dim)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    
    # 训练配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 优先使用GPU以解决CPU内存不足问题
    # AMD ROCm支持检测
    print(f'使用设备: {device}')
    print(f'使用设备: {device}')
    
    max_user_id = all_ratings['userId'].max() if not all_ratings.empty else 0
    max_movie_id_from_ratings = all_ratings['movieId'].max() if not all_ratings.empty else 0
    max_movie_id_from_tags = tag_matrix.index.max() if not tag_matrix.empty else 0
    max_movie_id = max(max_movie_id_from_ratings, max_movie_id_from_tags)
    # 加载KG实体ID映射并获取最大实体ID
    import os
    import json
    script_dir = os.path.dirname(os.path.abspath(__file__))
    entity_path = os.path.join(script_dir, 'kg_embeddings/entity2id.json')
    entity_path = os.path.abspath(entity_path)
    with open(entity_path, 'r') as f:
        entity2id = json.load(f)
    max_entity_id = max(entity2id.values()) if entity2id else 0
    # 同时考虑edge_index、训练数据、测试数据、标签和KG实体中的最大ID
    num_users = max(edge_index[0].max().item(), train_data['userId'].max().item()) + 1
    num_items = max(edge_index[1].max().item(), train_data['movieId'].max().item(), test_data['movieId'].max().item(), max_movie_id) + 1
    print(f"Max user ID from edge_index: {num_users - 1}, embedding size: {num_users}")
    print(f"Max item ID from edge_index: {num_items - 1}, embedding size: {num_items}")
    num_tags = tag_matrix.shape[1]
    
    from lightgcn import LightGCN
    from dual_tower import DualTowerModel, FAISSIndexer, RecommendationDataset, train_dual_tower
    from kg_enhanced_mf import KnowledgeEnhancedMF
    if model_type == 'lightgcn':
        # 根据图中总节点数调整num_items
        total_nodes = num_users + num_items  # 用户+电影节点总数
        model = LightGCN(num_users, num_items, embedding_dim).to(device)
        criterion = nn.BCELoss()
    elif model_type == 'dual_tower':
        # 双塔模型需要用户特征和物品特征维度
        # 获取用户特征和物品特征的维度
        # 安全处理用户特征维度计算
        user_feature_dim = 3
        if user_features is not None:
            if isinstance(user_features, dict):
                # 从字典中获取第一个用户的特征长度
                if user_features and len(user_features) > 0:
                    first_user_id = next(iter(user_features.keys()))
                    user_feature_dim = len(user_features[first_user_id])
            elif hasattr(user_features, 'shape'):
                # 处理数组/矩阵类型
                user_feature_dim = user_features.shape[1]
            elif isinstance(user_features, list):
                # 处理列表类型
                user_feature_dim = len(user_features[0]) if user_features else 3
        item_feature_dim = tag_matrix.shape[1] if tag_matrix is not None and not tag_matrix.empty else 3
        model = DualTowerModel(user_feature_dim, item_feature_dim, embedding_dim).to(device)
    else:
        model = KnowledgeEnhancedMF(num_users=num_users, num_items=num_items, num_tags=143, embedding_dim=embedding_dim).to(device)

    # 非双塔模型特征初始化
    if model_type != 'dual_tower':
        user_features = np.array([np.random.rand(10) for _ in range(len(all_ratings))]) if user_features is None else user_features
        item_features = np.array([np.random.rand(10) for _ in range(len(item_ids))]) if item_features is None else item_features
        labels = np.random.randint(0, 2, size=len(all_ratings), dtype=np.int64) if labels is None else labels

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) if model_type == 'mf' else torch.optim.Adam(model.parameters(), lr=0.0005)
    
    # 双塔模型专用变量
    if model_type == 'dual_tower':
        # 确保特征数据正确准备
        # 特征预处理 (实际项目中需要实现这些函数)
        # 确保用户特征和物品特征长度完全一致
        # 生成float32特征以匹配模型参数 dtype
        user_features = np.array([np.random.rand(10) for _ in range(len(all_ratings))]).astype(np.float32)
        item_features = np.array([np.random.rand(10) for _ in range(len(user_features))]).astype(np.float32)
        # 显式确保labels长度与all_ratings一致
        labels = np.random.randint(0, 2, size=len(all_ratings))
        
        # 验证特征和标签长度一致性
        assert len(user_features) == len(item_features), f"用户特征长度{len(user_features)}与物品特征长度{len(item_features)}不匹配"
        assert len(user_features) == len(labels), f"用户特征长度{len(user_features)}与标签长度{len(labels)}不匹配"
        
        # 划分训练集和验证集
        user_features, val_user_features, item_features, val_item_features, labels, val_labels = train_test_split(
            user_features, item_features, labels, test_size=0.2, random_state=42
        )
        
        # 为FAISS索引创建与唯一物品数量匹配的特征
        unique_item_features = np.array([np.random.rand(10) for _ in range(len(item_ids))])
        all_item_features = torch.tensor(unique_item_features, dtype=torch.float32)
    
    # 混合精度训练
    scaler = amp.GradScaler() if device.type == 'cuda' else None
    
    # 记录训练过程
    train_losses = []
    val_losses = []
    test_losses = []
    # 训练循环
    import time
    if model_type == 'dual_tower':
        # 构建物品特征加载器
        all_item_features = torch.tensor(item_features, dtype=torch.float32)
        item_feature_loader = DataLoader(all_item_features, batch_size=batch_size, shuffle=False)
        
        # 使用专用训练函数
        train_dataset = RecommendationDataset(user_features, item_features, labels)
        val_dataset = RecommendationDataset(val_user_features, val_item_features, val_labels)
        model, train_losses, val_losses = train_dual_tower(model, train_dataset, val_dataset, epochs=epochs, batch_size=batch_size)
        
        # 构建FAISS索引
        print('开始构建FAISS索引...')
        item_embeddings = []
        model.eval()
        with torch.no_grad():
            for features in item_feature_loader:
                emb = model.item_tower(features.to(device))
                item_embeddings.append(emb.cpu().numpy())
        item_embeddings = np.vstack(item_embeddings)
        
        indexer = FAISSIndexer(embedding_dim)
        indexer.build_index(item_embeddings, item_ids)
        os.makedirs('models/faiss_index', exist_ok=True)
        indexer.save_index('models/faiss_index/item_index.bin')
    # 非双塔模型通用的特征初始化
    if model_type != 'dual_tower':
        user_features = np.array([np.random.rand(10) for _ in range(len(all_ratings))])
        item_features = np.array([np.random.rand(10) for _ in range(len(item_ids))])
        labels = np.random.randint(0, 2, size=len(all_ratings))
    print(f'FAISS索引构建完成，包含{len(item_ids)}个物品向量')
    
    if model_type != 'dual_tower':
        for epoch in range(epochs):
            start_time = time.time()
            model.train()
            total_loss = 0
            batch_count = 0
    # 准备双塔模型数据集
    if model_type == 'dual_tower':
        all_item_features = torch.tensor(item_features, dtype=torch.float32)
        item_feature_loader = DataLoader(all_item_features, batch_size=batch_size, shuffle=False)
        
        # 使用专用训练函数
        train_dataset = RecommendationDataset(user_features, item_features, labels)
        val_dataset = RecommendationDataset(val_user_features, val_item_features, val_labels)
        model, train_losses, val_losses = train_dual_tower(model, train_dataset, val_dataset, epochs=epochs, batch_size=batch_size)
        
        # 构建FAISS索引
        print('开始构建FAISS索引...')
        item_embeddings = []
        model.eval()
    if model_type == 'dual_tower':
        with torch.no_grad():
            for features in item_feature_loader:
                emb = model.item_tower(features.to(device))
                item_embeddings.append(emb.cpu().numpy())
        item_embeddings = np.vstack(item_embeddings)
        
        indexer = FAISSIndexer(embedding_dim)
        indexer.build_index(item_embeddings, item_ids)
        indexer.save_index('models/faiss_item_index.bin')
        print(f'FAISS索引构建完成，包含{len(item_ids)}个物品向量')
    else:
        model.train()
        total_loss = 0
        batch_count = 0
        
        for batch_idx, batch in enumerate(train_loader):
            batch_count += 1
            if batch_count % 10 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Batch {batch_count}/{len(train_loader)}, Time elapsed: {time.time()-start_time:.2f}s')
                if model_type == 'lightgcn':
                    user_ids, movie_ids, ratings = batch
                    user_ids = user_ids.to(device)
                    movie_ids = movie_ids.to(device)
                    ratings = ratings.to(device)
                elif model_type == 'mf':
                    user_ids, item_ids, ratings, tag_features = batch
                    user_ids = user_ids.to(device)
                    item_ids = item_ids.to(device)
                    ratings = ratings.to(device)
                elif model_type == 'dual_tower':
                    user_features, item_features, ratings = batch
                    user_features = user_features.to(device)
                    item_features = item_features.to(device)
                    ratings = ratings.to(device)
            else:
                if model_type == 'dual_tower':
                    user_features, item_features, ratings = batch
                    user_features = user_features.to(device).float()
                    item_features = item_features.to(device).float()
                    ratings = ratings.to(device).float()
                elif model_type == 'lightgcn':
                    user_ids, movie_ids, ratings = batch
                    user_ids = user_ids.to(device)
                    movie_ids = movie_ids.to(device)
                    ratings = ratings.to(device).float()
                elif model_type == 'lightgcn':
                    user_ids, movie_ids, ratings = batch
                    user_ids = user_ids.to(device)
                    movie_ids = movie_ids.to(device)
                    ratings = ratings.to(device).float()
                elif model_type == 'mf':
                    user_ids, movie_ids, ratings, tag_features = batch
                    user_ids = user_ids.to(device)
                    movie_ids = movie_ids.to(device)
                    ratings = ratings.to(device).float()
                    tag_features = tag_features.to(device).float()

            
            optimizer.zero_grad()
            
            # 混合精度训练上下文
            with torch.cuda.amp.autocast() if scaler is not None else contextlib.nullcontext():
                if model_type == 'lightgcn':
                    # 使用图的edge_index获取所有用户和物品嵌入
                    all_user_emb, all_item_emb = model(edge_index.to(device))
                    # 索引当前批次的嵌入
                    user_emb = all_user_emb[user_ids]
                    item_emb = all_item_emb[movie_ids]
                    outputs = torch.sigmoid((user_emb * item_emb).sum(1)) * 5  # 缩放到0-5评分范围
                    loss = criterion(outputs, ratings)
                elif model_type == 'dual_tower':
                    user_emb, item_emb = model(user_features, item_features)
                    outputs = torch.sigmoid(torch.sum(user_emb * item_emb, dim=1)) * 5
                    loss = criterion(outputs, ratings)
                else:
                    outputs = model(user_ids, movie_ids, tag_features)
                    loss = criterion(outputs, ratings)
            
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item() * len(ratings)
        
        avg_train_loss = total_loss / len(train_data)
        train_losses.append(avg_train_loss)
        
        # 评估
        # 评估
        model.eval()
        test_total_loss = 0
        all_preds = []
        all_ratings = []
        
        with torch.no_grad():
                for batch in test_loader:
                    if model_type == 'lightgcn':
                        user_ids, movie_ids, ratings = batch
                        user_ids = user_ids.to(device)
                        movie_ids = movie_ids.to(device)
                        ratings = ratings.to(device)
                        all_user_emb, all_item_emb = model(edge_index.to(device))
                        user_emb = all_user_emb[user_ids]
                        item_emb = all_item_emb[movie_ids]
                        outputs = torch.sigmoid((user_emb * item_emb).sum(1)) * 5  # 缩放到0-5评分范围
                    else:
                        if model_type == 'dual_tower':
                            user_features, item_features, ratings = batch
                            user_features = user_features.to(device)
                            item_features = item_features.to(device)
                            ratings = ratings.to(device).float()
                            user_emb, item_emb = model(user_features, item_features)
                            # 计算用户和物品嵌入的点积作为预测分数
                            outputs = (user_emb * item_emb).sum(dim=1)
                else:
                    if model_type == 'lightgcn':
                        user_ids, movie_ids, ratings = batch
                        user_ids = user_ids.to(device)
                        movie_ids = movie_ids.to(device)
                        ratings = ratings.to(device)
                    else:
                        user_ids, movie_ids, ratings, tag_features = batch
                        user_ids = user_ids.to(device)
                        movie_ids = movie_ids.to(device)
                        ratings = ratings.to(device)
                        tag_features = tag_features.to(device)
                        outputs = model(user_ids, movie_ids, tag_features)
                        loss = criterion(outputs, ratings)
                test_total_loss += loss.item() * len(ratings)
                
                all_preds.extend(outputs.cpu().numpy())
                all_ratings.extend(ratings.cpu().numpy())
        
        avg_test_loss = test_total_loss / len(test_data)
        rmse = np.sqrt(mean_squared_error(all_ratings, all_preds))
        test_losses.append(avg_test_loss)
        
        print(f'Epoch {epoch+1}/{epochs}, 训练损失: {avg_train_loss:.4f}, 测试损失: {avg_test_loss:.4f}, 测试RMSE: {rmse:.4f}')
    
    # 保存模型
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f'模型已保存至 {model_save_path}')
    
    # 保存标签矩阵
    with open(os.path.join(os.path.dirname(model_save_path), 'tag_matrix.pkl'), 'wb') as f:
        pickle.dump(tag_matrix, f)
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    # 使用损失列表实际长度生成x轴范围
    plt.plot(range(1, len(train_losses)+1), train_losses, label='训练损失')
    plt.plot(range(1, len(test_losses)+1), test_losses, label='测试损失')
    plt.xlabel('Epoch')
    plt.ylabel('MSE损失')
    plt.title('训练过程中的损失变化')
    plt.legend()
    plt.savefig(os.path.join(os.path.dirname(model_save_path), 'loss_curve.png'))
    
    return model, train_losses, test_losses
    # 数据预处理
    train_data, test_data, tag_matrix, all_ratings = preprocess_data(data_dir)
    
    # 创建数据加载器
    train_dataset = RatingDataset(train_data, tag_matrix)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    test_dataset = RatingDataset(test_data, tag_matrix)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # 训练配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    max_user_id = all_ratings['userId'].max() if not all_ratings.empty else 0
    max_movie_id_from_ratings = all_ratings['movieId'].max() if not all_ratings.empty else 0
    max_movie_id_from_tags = tag_matrix.index.max() if not tag_matrix.empty else 0
    max_movie_id = max(max_movie_id_from_ratings, max_movie_id_from_tags)
    num_users = max_user_id
    num_movies = max_movie_id
    print(f"Max user ID: {max_user_id}, embedding size: {num_users + 1}")
    print(f"Max movie ID: {max_movie_id}, embedding size: {num_movies + 1}")
    num_tags = tag_matrix.shape[1]
    
    model = KnowledgeEnhancedMF(num_users, num_movies, num_tags, embedding_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 记录训练过程
    train_losses = []
    test_losses = []
    
    # 训练循环
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
            
            total_loss += loss.item() * len(ratings)
        
        avg_train_loss = total_loss / len(train_data)
        train_losses.append(avg_train_loss)
        
        # 评估
        model.eval()
        test_total_loss = 0
        all_preds = []
        all_ratings = []
        
        with torch.no_grad():
            for batch in test_loader:
                user_ids, movie_ids, ratings, tag_features = batch
                user_ids = user_ids.to(device)
                movie_ids = movie_ids.to(device)
                ratings = ratings.to(device)
                tag_features = tag_features.to(device)
                
                outputs = model(user_ids, movie_ids, tag_features)
                loss = criterion(outputs, ratings)
                test_total_loss += loss.item() * len(ratings)
                
                all_preds.extend(outputs.cpu().numpy())
                all_ratings.extend(ratings.cpu().numpy())
        
        avg_test_loss = test_total_loss / len(test_data)
        test_losses.append(avg_test_loss)
        rmse = np.sqrt(mean_squared_error(all_ratings, all_preds))
        
        print(f'Epoch {epoch+1}/{epochs}, 训练损失: {avg_train_loss:.4f}, 测试损失: {avg_test_loss:.4f}, 测试RMSE: {rmse:.4f}')
    
    # 保存模型
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f'模型已保存至 {model_save_path}')
    
    # 保存标签矩阵
    with open(os.path.join(os.path.dirname(model_save_path), 'tag_matrix.pkl'), 'wb') as f:
        pickle.dump(tag_matrix, f)
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs+1), train_losses, label='训练损失')
    plt.plot(range(1, epochs+1), test_losses, label='测试损失')
    plt.xlabel('Epoch')
    plt.ylabel('MSE损失')
    plt.title('训练过程中的损失变化')
    plt.legend()
    plt.savefig(os.path.join(os.path.dirname(model_save_path), 'loss_curve.png'))
    
    return model

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='mf', help='模型类型: mf, lightgcn, dual_tower')
    parser.add_argument('--epochs', type=int, default=5, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=128, help='批次大小')
    parser.add_argument('--embedding_dim', type=int, default=64, help='嵌入维度')
    args = parser.parse_args()
    
    GRAPH_PATH = 'models/kg_data/kg.graph'  # 小型知识图谱路径
    MODEL_SAVE_PATH = f'models/kg_rec_{args.model_type}_small.pth'
    
    print(f'开始使用小型数据集训练{args.model_type}模型...')
    model, train_losses, val_losses = train_model(
        GRAPH_PATH, 
        MODEL_SAVE_PATH, 
        epochs=args.epochs, 
        batch_size=args.batch_size, 
        embedding_dim=args.embedding_dim, 
        model_type=args.model_type
    )
    print(f'{args.model_type}模型训练完成!')