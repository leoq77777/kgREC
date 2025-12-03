import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import networkx as nx
import os
import logging
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import roc_auc_score, precision_score, recall_score

class KnowledgeGraphEmbeddingModel:
    def __init__(self,
                 graph: nx.MultiDiGraph,
                 embedding_dim: int = 100,
                 model_type: str = "TransE",
                 margin: float = 1.0,
                 learning_rate: float = 0.001):
        self.logger = logging.getLogger(__name__)
        self.logger.info("=== 初始化KnowledgeGraphEmbeddingModel ===")
        self.logger.info(f"模型类型: {model_type}, 嵌入维度: {embedding_dim}")
        """
        知识图谱嵌入模型基类
        :param graph: NetworkX知识图谱对象
        :param embedding_dim: 嵌入维度
        :param model_type: 嵌入模型类型 (TransE, DistMult)
        :param margin: TransE模型中的margin参数
        :param learning_rate: 学习率
        """
        self.graph = graph
        self.embedding_dim = embedding_dim
        self.model_type = model_type
        self.margin = margin
        self.learning_rate = learning_rate
        self.logger = logging.getLogger("KnowledgeGraphEmbeddingModel")
        logging.basicConfig(level=logging.INFO)

        # 获取所有实体和关系
        self.entities = list(graph.nodes())
        self.relations = list(set([graph.edges[u, v, k].get("type", "DEFAULT") 
                                 for u, v, k in graph.edges(keys=True)]))

        # 创建实体和关系到ID的映射
        self.entity2id = {entity: i for i, entity in enumerate(self.entities)}
        self.relation2id = {relation: i for i, relation in enumerate(self.relations)}

        # 初始化嵌入
        self.entity_embeddings = None
        self.relation_embeddings = None
        self.logger.info("开始初始化嵌入向量...")
        self._initialize_embeddings()
        self.logger.info("嵌入向量初始化完成")

        # 准备训练数据
        self.logger.info("开始准备训练三元组...")
        triples_start = time.time()
        self.triples = self._prepare_training_triples()
        triples_end = time.time()
        self.logger.info(f"训练三元组准备完成，耗时: {triples_end - triples_start:.2f}秒, 共 {len(self.triples)} 个三元组")
        self.logger.info(f"已初始化{model_type}模型，实体数: {len(self.entities)}, 关系数: {len(self.relations)}, 三元组数: {len(self.triples)}")

        # 定义模型、损失函数和优化器
        self.model = self._build_model()
        self.loss_fn = self._get_loss_function()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def _initialize_embeddings(self) -> None:
        """初始化实体和关系嵌入"""
        # 实体嵌入
        self.entity_embeddings = nn.Embedding(len(self.entities), self.embedding_dim)
        nn.init.xavier_uniform_(self.entity_embeddings.weight.data)

        # 关系嵌入
        self.relation_embeddings = nn.Embedding(len(self.relations), self.embedding_dim)
        nn.init.xavier_uniform_(self.relation_embeddings.weight.data)

        # 如果是TransE模型，归一化实体嵌入
        if self.model_type == "TransE":
            self.entity_embeddings.weight.data = nn.functional.normalize(
                self.entity_embeddings.weight.data, p=2, dim=1
            )

    def _build_model(self) -> nn.Module:
        """构建嵌入模型"""
        if self.model_type == "TransE":
            return TransEModel(self.entity_embeddings, self.relation_embeddings)
        elif self.model_type == "DistMult":
            return DistMultModel(self.entity_embeddings, self.relation_embeddings)
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")

    def _get_loss_function(self) -> nn.Module:
        """获取损失函数"""
        if self.model_type == "TransE":
            return nn.MarginRankingLoss(margin=self.margin)
        elif self.model_type == "DistMult":
            return nn.BCELoss()
        else:
            return nn.MSELoss()

    def _prepare_training_triples(self) -> List[Tuple[str, str, str]]:
        self.logger.info("开始准备训练三元组...")
        start_time = time.time()
        """准备训练三元组 (头实体, 关系, 尾实体)"""
        triples = []
        total_edges = len(self.graph.edges(keys=True))
        self.logger.info(f"总边数: {total_edges}, 开始转换为三元组...")
        for i, (u, v, k) in enumerate(self.graph.edges(keys=True)):
            if i % 1000000 == 0 and i > 0:
                self.logger.info(f"已处理 {i}/{total_edges} 条边 ({i/total_edges:.2%})")
            relation = self.graph.edges[u, v, k].get("type", "DEFAULT")
            triples.append((u, relation, v))
        end_time = time.time()
        self.logger.info(f"三元组准备完成，共处理 {len(triples)} 个三元组，耗时 {end_time - start_time:.2f}秒")
        return triples

    def _generate_negative_triples(self, positive_triples: List[Tuple], num_negatives: int = 1) -> List[Tuple]:
        """生成负采样三元组"""
        negative_triples = []
        for h, r, t in positive_triples:
            for _ in range(num_negatives):
                # 随机替换头实体或尾实体
                if np.random.rand() < 0.5:
                    # 替换头实体
                    h_neg = np.random.choice(self.entities)
                    while h_neg == h:
                        h_neg = np.random.choice(self.entities)
                    negative_triples.append((h_neg, r, t))
                else:
                    # 替换尾实体
                    t_neg = np.random.choice(self.entities)
                    while t_neg == t:
                        t_neg = np.random.choice(self.entities)
                    negative_triples.append((h, r, t_neg))
        return negative_triples

    def train(self, epochs: int = 100, batch_size: int = 128, num_negatives: int = 1) -> Dict:
        """
        训练嵌入模型
        :param epochs: 训练轮数
        :param batch_size: 批次大小
        :param num_negatives: 每个正样本的负样本数
        :return: 训练指标
        """
        self.model.train()
        metrics = {"loss": []}
        self.logger.info("=== 开始模型训练 ===")
        self.logger.info(f"总 epochs: {epochs}, 批次大小: {batch_size}, 负样本数: {num_negatives}")
        training_start_time = time.time()

        for epoch in range(epochs):
            epoch_start_time = time.time()
            total_loss = 0
            self.logger.info(f"Epoch {epoch+1}/{epochs} 开始处理")
            # 打乱训练数据
            np.random.shuffle(self.triples)
            num_batches = len(self.triples) // batch_size

            for i in range(num_batches):
                if i % 10 == 0:  # 每10个批次输出一次进度
                    self.logger.info(f"  Batch {i+1}/{num_batches} 处理中")
                # 获取批次正样本
                batch_positive = self.triples[i*batch_size : (i+1)*batch_size]
                # 生成负样本
                batch_negative = self._generate_negative_triples(batch_positive, num_negatives)

                # 转换为ID
                pos_h = torch.tensor([self.entity2id[h] for h, r, t in batch_positive], dtype=torch.long)
                pos_r = torch.tensor([self.relation2id[r] for h, r, t in batch_positive], dtype=torch.long)
                pos_t = torch.tensor([self.entity2id[t] for h, r, t in batch_positive], dtype=torch.long)

                neg_h = torch.tensor([self.entity2id[h] for h, r, t in batch_negative], dtype=torch.long)
                neg_r = torch.tensor([self.relation2id[r] for h, r, t in batch_negative], dtype=torch.long)
                neg_t = torch.tensor([self.entity2id[t] for h, r, t in batch_negative], dtype=torch.long)

                # 前向传播
                pos_scores = self.model(pos_h, pos_r, pos_t)
                neg_scores = self.model(neg_h, neg_r, neg_t)

                # 计算损失
                if self.model_type == "TransE":
                    # MarginRankingLoss需要目标标签 (1表示正样本分数应高于负样本)
                    target = torch.tensor([1], dtype=torch.float)
                    loss = self.loss_fn(pos_scores, neg_scores, target)
                else:
                    # DistMult使用二分类损失
                    labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])
                    scores = torch.cat([pos_scores, neg_scores])
                    loss = self.loss_fn(scores, labels)

                # 反向传播和优化
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            # 记录平均损失
            epoch_end_time = time.time()
            avg_loss = total_loss / num_batches
            metrics["loss"].append(avg_loss)
            self.logger.info(f"Epoch {epoch+1}/{epochs} 完成, 平均损失: {avg_loss:.4f}, 耗时: {epoch_end_time - epoch_start_time:.2f}秒")

            if (epoch + 1) % 10 == 0:
                self.logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        training_end_time = time.time()
        self.logger.info(f"=== 模型训练完成 ===")
        self.logger.info(f"总训练耗时: {training_end_time - training_start_time:.2f}秒")
        return metrics

    def evaluate(self, test_triples: Optional[List[Tuple]] = None, k: int = 10) -> Dict:
        """
        评估模型性能
        :param test_triples: 测试三元组列表，如果为None则使用训练数据的一部分
        :param k: Hits@k评估指标中的k值
        :return: 评估指标字典
        """
        self.model.eval()
        if test_triples is None:
            # 从训练数据中划分10%作为测试集
            split_idx = int(len(self.triples) * 0.9)
            test_triples = self.triples[split_idx:]
            self.triples = self.triples[:split_idx]

        metrics = {"hits@1": 0, "hits@3": 0, "hits@10": 0, "mean_rank": 0}
        num_triples = len(test_triples)

        with torch.no_grad():
            for h, r, t in test_triples:
                # 获取实体和关系ID
                h_id = self.entity2id[h]
                r_id = self.relation2id[r]
                t_id = self.entity2id[t]

                # 计算所有实体作为尾实体的分数
                h_tensor = torch.tensor([h_id], dtype=torch.long).repeat(len(self.entities))
                r_tensor = torch.tensor([r_id], dtype=torch.long).repeat(len(self.entities))
                t_tensor = torch.tensor(range(len(self.entities)), dtype=torch.long)

                scores = self.model(h_tensor, r_tensor, t_tensor)
                scores = scores.cpu().numpy()

                # 获取正确答案的排名
                target_score = scores[t_id]
                # 计算有多少实体的分数高于目标实体
                rank = np.sum(scores > target_score) + 1  # +1是因为排名从1开始

                # 更新指标
                metrics["mean_rank"] += rank
                if rank <= 1:
                    metrics["hits@1"] += 1
                if rank <= 3:
                    metrics["hits@3"] += 1
                if rank <= k:
                    metrics["hits@10"] += 1

        # 计算平均指标
        metrics["mean_rank"] /= num_triples
        metrics["hits@1"] /= num_triples
        metrics["hits@3"] /= num_triples
        metrics["hits@10"] /= num_triples

        self.logger.info(f"评估结果 - Hits@1: {metrics['hits@1']:.4f}, Hits@3: {metrics['hits@3']:.4f}, Hits@10: {metrics['hits@10']:.4f}, Mean Rank: {metrics['mean_rank']:.2f}")
        return metrics

    def save_embeddings(self, save_dir: str = "models/kg_embeddings/") -> None:
        """
        保存实体和关系嵌入
        :param save_dir: 保存目录
        """
        os.makedirs(save_dir, exist_ok=True)

        # 保存嵌入权重
        np.save(os.path.join(save_dir, "entity_embeddings.npy"), 
                self.entity_embeddings.weight.data.cpu().numpy())
        np.save(os.path.join(save_dir, "relation_embeddings.npy"), 
                self.relation_embeddings.weight.data.cpu().numpy())

        # 保存映射
        import json
        with open(os.path.join(save_dir, "entity2id.json"), "w") as f:
            json.dump(self.entity2id, f, indent=2)
        with open(os.path.join(save_dir, "relation2id.json"), "w") as f:
            json.dump(self.relation2id, f, indent=2)

        self.logger.info(f"嵌入已保存到: {save_dir}")

    @classmethod
    def load_embeddings(cls, load_dir: str = "models/kg_embeddings/") -> Tuple[np.ndarray, np.ndarray, Dict, Dict]:
        """
        加载实体和关系嵌入
        :param load_dir: 加载目录
        :return: (实体嵌入, 关系嵌入, entity2id, relation2id)
        """
        # 加载嵌入权重
        entity_embeddings = np.load(os.path.join(load_dir, "entity_embeddings.npy"))
        relation_embeddings = np.load(os.path.join(load_dir, "relation_embeddings.npy"))

        # 加载映射
        import json
        with open(os.path.join(load_dir, "entity2id.json"), "r") as f:
            entity2id = json.load(f)
        with open(os.path.join(load_dir, "relation2id.json"), "r") as f:
            relation2id = json.load(f)

        return entity_embeddings, relation_embeddings, entity2id, relation2id

class TransEModel(nn.Module):
    """TransE模型实现"""
    def __init__(self, entity_embeddings, relation_embeddings):
        super(TransEModel, self).__init__()
        self.entity_embeddings = entity_embeddings
        self.relation_embeddings = relation_embeddings

    def forward(self, h, r, t):
        # 获取嵌入
        h_emb = self.entity_embeddings(h)
        r_emb = self.relation_embeddings(r)
        t_emb = self.entity_embeddings(t)

        # 计算h + r - t的L1范数
        score = torch.norm(h_emb + r_emb - t_emb, p=1, dim=1)
        # 返回负分数，因为我们希望正样本的分数更小
        return -score

class DistMultModel(nn.Module):
    """DistMult模型实现"""
    def __init__(self, entity_embeddings, relation_embeddings):
        super(DistMultModel, self).__init__()
        self.entity_embeddings = entity_embeddings
        self.relation_embeddings = relation_embeddings
        self.sigmoid = nn.Sigmoid()

    def forward(self, h, r, t):
        # 获取嵌入
        h_emb = self.entity_embeddings(h)
        r_emb = self.relation_embeddings(r)
        t_emb = self.entity_embeddings(t)

        # 计算h^T * diag(r) * t
        score = torch.sum(h_emb * r_emb * t_emb, dim=1)
        # 使用sigmoid将分数映射到[0,1]区间
        return self.sigmoid(score)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='知识图谱嵌入模型训练脚本')
    parser.add_argument('--graph_path', type=str, default='models/kg_data/kg.graph', help='知识图谱文件路径')
    parser.add_argument('--embedding_dim', type=int, default=50, help='嵌入维度 (小型数据集推荐50)')
    parser.add_argument('--epochs', type=int, default=10, help='训练轮数 (小型数据集推荐5-10)')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小 (小型数据集推荐32-64)')
    parser.add_argument('--debug', action='store_true', help='启用调试模式，输出详细进度信息')
    args = parser.parse_args()
    # 示例用法
    from kg_construction import KnowledgeGraphConstructor
    import time

    # 加载知识图谱
    kg_path = args.graph_path
    if os.path.exists(kg_path):
        kg_constructor = KnowledgeGraphConstructor()
        kg_constructor.load_graph(kg_path)
        graph = kg_constructor.graph
        logging.info(f"成功加载知识图谱: {kg_path}, 包含 {len(graph.nodes)} 个实体和 {len(graph.edges)} 条关系")
        logging.info("开始图谱数据预处理...")
        preprocess_start = time.time()
        
        # 创建并训练嵌入模型
        preprocess_end = time.time()
        logging.info(f"图谱数据预处理完成，耗时: {preprocess_end - preprocess_start:.2f}秒")
        start_time = time.time()
        logging.info("开始初始化知识图谱嵌入模型...")
        kg_emb_model = KnowledgeGraphEmbeddingModel(
            graph=graph,
            embedding_dim=args.embedding_dim,
            model_type="TransE",
            margin=1.0,
            learning_rate=0.001
        )
        if args.debug:
            logging.info(f"调试模式启用 - 嵌入维度: {args.embedding_dim}, 训练轮数: {args.epochs}, 批次大小: {args.batch_size}")
        
        # 训练模型
        logging.info("开始模型训练...")
        metrics = kg_emb_model.train(epochs=args.epochs, batch_size=args.batch_size)
        
        # 评估模型
        logging.info("开始模型评估...")
        kg_emb_model.evaluate(k=10)
        logging.info("模型评估完成")
        
        # 保存嵌入
        logging.info("开始保存嵌入向量...")
        kg_emb_model.save_embeddings()
        logging.info("嵌入向量保存完成")
        
        end_time = time.time()
        print(f"嵌入模型训练和评估完成，耗时: {end_time - start_time:.2f}秒")
    else:
        print(f"""知识图谱文件未找到: {kg_path}
请先运行data_to_kg.py生成知识图谱""")