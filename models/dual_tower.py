import torch
import torch.nn as nn
import torch.nn.functional as F
import faiss
import numpy as np
from torch.utils.data import DataLoader, Dataset

class DualTowerModel(nn.Module):
    def __init__(self, user_feature_dim, item_feature_dim, embedding_dim=64):
        super(DualTowerModel, self).__init__()
        
        # 用户塔
        self.user_tower = nn.Sequential(
            nn.Linear(10, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim)
        )
        
        # 物品塔
        self.item_tower = nn.Sequential(
            nn.Linear(10, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim)
        )
        
        # 温度参数
        self.temperature = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, user_features, item_features):
        # 获取用户和物品嵌入
        user_emb = self.user_tower(user_features)
        item_emb = self.item_tower(item_features)
        
        # L2归一化
        user_emb = F.normalize(user_emb, p=2, dim=1)
        item_emb = F.normalize(item_emb, p=2, dim=1)
        
        return user_emb, item_emb
        
    def calculate_loss(self, user_emb, item_emb, labels):
        # 计算余弦相似度
        # 计算用户-物品对相似度（1D向量）
        cos_sim = F.cosine_similarity(user_emb, item_emb, dim=1) * torch.exp(self.temperature)  # 点积相似度
        # 二元分类使用BCEWithLogitsLoss，确保输入输出形状匹配
        loss = F.binary_cross_entropy_with_logits(cos_sim, labels.to(torch.float))
        return loss

class RecommendationDataset(Dataset):
    def __init__(self, user_features, item_features, labels):
        self.user_features = user_features
        self.item_features = item_features
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.user_features[idx], self.item_features[idx], self.labels[idx]

class FAISSIndexer:
    def __init__(self, dimension):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.item_ids = None
        
    def build_index(self, item_embeddings, item_ids):
        # 构建FAISS索引
        self.index.add(item_embeddings.astype(np.float32))
        self.item_ids = item_ids
        
    def search(self, user_embedding, top_k=10):
        # 搜索最近邻
        distances, indices = self.index.search(user_embedding.astype(np.float32), top_k)
        
        # 返回物品ID和相似度分数
        return [self.item_ids[i] for i in indices[0]], distances[0]
        
    def save_index(self, path):
        # 保存索引
        faiss.write_index(self.index, path)
        
    def load_index(self, path):
        # 加载索引
        self.index = faiss.read_index(path)

# 训练函数
def train_dual_tower(model, train_dataset, val_dataset, epochs=50, batch_size=256, lr=0.001):
    train_losses = []
    val_losses = []
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for user_features, item_features, labels in train_loader:
            optimizer.zero_grad()
            
            user_emb, item_emb = model(user_features, item_features)
            loss = model.calculate_loss(user_emb, item_emb, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * user_features.size(0)
        
        train_loss /= len(train_loader.dataset)
        
        # 验证
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for user_features, item_features, labels in val_loader:
                user_emb, item_emb = model(user_features, item_features)
                loss = model.calculate_loss(user_emb, item_emb, labels)
                val_loss += loss.item() * user_features.size(0)
            
        val_loss /= len(val_loader.dataset)
        
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # 学习率调度
        scheduler.step(val_loss)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'dual_tower_best.pth')
            print(f'Saved best model with val loss: {best_val_loss:.4f}')
    
    return model, train_losses, val_losses