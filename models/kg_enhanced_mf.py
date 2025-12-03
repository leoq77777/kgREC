import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader

class KnowledgeEnhancedMF(nn.Module):
    def __init__(self, num_users, num_movies, num_tags, embedding_dim=64):
        super().__init__()
        # 用户和电影嵌入层
        self.user_emb = nn.Embedding(num_users + 1, embedding_dim)
        self.movie_emb = nn.Embedding(num_movies + 1, embedding_dim)
        # 偏置项
        self.user_bias = nn.Embedding(num_users + 1, 1)
        self.movie_bias = nn.Embedding(num_movies + 1, 1)
        # 标签特征注意力层
        self.tag_attention = nn.Sequential(
            nn.Linear(num_tags, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        # 初始化权重
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.movie_emb.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.movie_bias.weight)

    def forward(self, user_ids, movie_ids, tag_features):
        # 矩阵分解基础得分
        user_vec = self.user_emb(user_ids)
        movie_vec = self.movie_emb(movie_ids)
        mf_score = (user_vec * movie_vec).sum(1)
        mf_score += self.user_bias(user_ids).squeeze() + self.movie_bias(movie_ids).squeeze()

        # 标签特征注意力加权
        tag_weights = self.tag_attention(tag_features).squeeze()

        # 融合得分 (70%矩阵分解 + 30%标签特征)
        final_score = mf_score * 0.7 + tag_weights * 0.3
        return torch.sigmoid(final_score) * 5  # 缩放到0-5评分范围

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