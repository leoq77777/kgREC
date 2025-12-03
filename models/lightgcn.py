import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import numpy as np

class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64, num_layers=3):
        super(LightGCN, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        
        # 用户和物品嵌入
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # GCN层
        self.convs = nn.ModuleList([GCNConv(embedding_dim, embedding_dim) for _ in range(num_layers)])
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        
    def forward(self, edge_index):
        # 合并用户和物品嵌入
        x = torch.cat([self.user_embedding.weight, self.item_embedding.weight])
        
        # 保存每一层的嵌入
        embeddings = [x]
        
        # 图卷积传播
        for conv in self.convs:
            x = conv(x, edge_index)
            embeddings.append(x)
        
        # 加权平均各层嵌入
        embeddings = torch.stack(embeddings, dim=1)
        embeddings = torch.mean(embeddings, dim=1)
        
        # 分离用户和物品嵌入
        user_embeddings = embeddings[:self.num_users]
        item_embeddings = embeddings[self.num_users:]
        
        return user_embeddings, item_embeddings
        
    def predict(self, user_ids, item_ids):
        # 获取用户和物品嵌入
        user_embeddings, item_embeddings = self.forward(self.edge_index)
        
        # 计算预测分数 (内积)
        user_emb = user_embeddings[user_ids]
        item_emb = item_embeddings[item_ids]
        scores = torch.sum(user_emb * item_emb, dim=1)
        
        return scores
        
    def bpr_loss(self, user_ids, pos_item_ids, neg_item_ids):
        # 计算正样本和负样本的分数
        user_embeddings, item_embeddings = self.forward(self.edge_index)
        
        user_emb = user_embeddings[user_ids]
        pos_item_emb = item_embeddings[pos_item_ids]
        neg_item_emb = item_embeddings[neg_item_ids]
        
        pos_scores = torch.sum(user_emb * pos_item_emb, dim=1)
        neg_scores = torch.sum(user_emb * neg_item_emb, dim=1)
        
        # BPR损失
        loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores)))
        
        return loss

# 训练函数示例
def train_lightgcn(model, train_data, epochs=50, lr=0.001, batch_size=1024):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        
        # 打乱训练数据
        indices = np.random.permutation(len(train_data['user_ids']))
        user_ids = train_data['user_ids'][indices]
        pos_item_ids = train_data['pos_item_ids'][indices]
        neg_item_ids = train_data['neg_item_ids'][indices]
        
        for i in range(0, len(train_data['user_ids']), batch_size):
            batch_user = user_ids[i:i+batch_size]
            batch_pos = pos_item_ids[i:i+batch_size]
            batch_neg = neg_item_ids[i:i+batch_size]
            
            optimizer.zero_grad()
            loss = model.bpr_loss(batch_user, batch_pos, batch_neg)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_data):.4f}')
    
    return model