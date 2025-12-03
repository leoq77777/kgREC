import os
import unittest
# 设置测试环境变量
os.environ["TESTING"] = "True"
import tempfile
import shutil
import pandas as pd
import numpy as np
from kg_inference_engine import KnowledgeGraphInferenceEngine
from kg_construction import KnowledgeGraphConstructor

class TestKnowledgeGraphInferenceEngine(unittest.TestCase):
    """知识图谱推理引擎测试套件"""
    @classmethod
    def setUpClass(cls):
        """测试前的准备工作，创建临时目录和示例数据"""
        # 设置测试环境变量
        os.environ["TESTING"] = "True"
        # 创建临时目录
        cls.temp_dir = tempfile.mkdtemp()
        cls.embeddings_dir = os.path.join(cls.temp_dir, "embeddings")
        cls.index_dir = os.path.join(cls.temp_dir, "faiss_index")
        cls.kg_save_path = os.path.join(cls.temp_dir, "test_kg.graph")

        # 创建示例知识图谱
        cls._create_sample_kg()

        # 创建示例嵌入
        cls._create_sample_embeddings()

        print(f"测试环境准备完成，临时目录: {cls.temp_dir}")

    @classmethod
    def tearDownClass(cls):
        """测试结束后清理临时文件"""
        shutil.rmtree(cls.temp_dir)
        print(f"测试环境清理完成")

    @classmethod
    def _create_sample_kg(cls):
        """创建测试用的知识图谱"""
        # 创建示例数据
        movies_data = {
            "movieId": [1, 2, 3, 4, 5],
            "title": [
                "Toy Story (1995)",
                "Jumanji (1995)",
                "Grumpier Old Men (1995)",
                "Waiting to Exhale (1995)",
                "Father of the Bride Part II (1995)"
            ],
            "genres": [
                "Adventure|Animation|Children|Comedy|Fantasy",
                "Adventure|Children|Fantasy",
                "Comedy|Romance",
                "Comedy|Drama|Romance",
                "Comedy"
            ]
        }
        movies_df = pd.DataFrame(movies_data)

        ratings_data = {
            "userId": [1, 1, 1, 2, 2],
            "movieId": [1, 2, 3, 4, 5],
            "rating": [4.0, 3.5, 5.0, 2.5, 4.5],
            "timestamp": [1112486027, 1112484676, 1112484819, 1112484727, 1112484580]
        }
        ratings_df = pd.DataFrame(ratings_data)

        tags_data = {
            "userId": [1, 2, 3],
            "movieId": [1, 3, 1],
            "tag": ["funny", "romantic", "animated"],
            "timestamp": [1112486027, 1112484819, 1112484580]
        }
        tags_df = pd.DataFrame(tags_data)

        # 构建知识图谱
        constructor = KnowledgeGraphConstructor()
        constructor.build_graph_from_data(movies_df, ratings_df, tags_df)
        constructor.save_graph(cls.kg_save_path)

    @classmethod
    def _create_sample_embeddings(cls):
        """创建测试用的嵌入文件"""
        os.makedirs(cls.embeddings_dir, exist_ok=True)

        # 创建示例实体嵌入
        entities = [
            "user_1", "user_2", "user_3",
            "movie_1", "movie_2", "movie_3", "movie_4", "movie_5",
            "genre_adventure", "genre_animation", "genre_children",
            "genre_comedy", "genre_fantasy", "genre_romance", "genre_drama",
            "tag_funny", "tag_romantic", "tag_animated"
        ]
        entity2id = {entity: i for i, entity in enumerate(entities)}
        # 为电影实体创建确定性嵌入，确保movie_1与movie_2、movie_3相似
        entity_embeddings = np.zeros((len(entities), 50))
        for i, entity in enumerate(entities):
            if entity.startswith('movie_'):
                movie_num = int(entity.split('_')[1])
                entity_embeddings[i] = np.ones(50) * movie_num
            else:
                entity_embeddings[i] = np.random.randn(50)

        # 创建示例关系嵌入
        relations = ["RATED", "HAS_GENRE", "HAS_TAG", "TAGGED_BY"]
        relation2id = {relation: i for i, relation in enumerate(relations)}
        # 为关系创建确定性嵌入
        relation_embeddings = np.zeros((len(relations), 50))
        for i, relation in enumerate(relations):
            relation_embeddings[i] = np.ones(50) * (i + 1)

        # 保存嵌入
        np.save(os.path.join(cls.embeddings_dir, "entity_embeddings.npy"), entity_embeddings)
        np.save(os.path.join(cls.embeddings_dir, "relation_embeddings.npy"), relation_embeddings)

        # 保存映射
        import json
        with open(os.path.join(cls.embeddings_dir, "entity2id.json"), "w") as f:
            json.dump(entity2id, f)
        with open(os.path.join(cls.embeddings_dir, "relation2id.json"), "w") as f:
            json.dump(relation2id, f)

    def setUp(self):
        """每个测试前初始化推理引擎"""
        os.environ["TESTING"] = "True"
        self.inference_engine = KnowledgeGraphInferenceEngine(
            embeddings_dir=self.embeddings_dir,
            index_dir=self.index_dir
        )
        self.inference_engine.load_graph(self.kg_save_path)

    def test_find_similar_entities(self):
        """测试查找相似实体功能"""
        # 获取movie_1的ID
        self.assertIn("movie_1", self.inference_engine.entity2id, "movie_1实体不存在")
        similar_entities = self.inference_engine.find_similar_entities("movie_1", top_k=5)
        self.assertEqual(len(similar_entities), 5, "返回的相似实体数量不正确")
        self.assertIsInstance(similar_entities[0][1], float, "相似度分数应该是浮点数")

        # 验证结果中不应包含自身
        entity_names = [entity for entity, _ in similar_entities]
        self.assertNotIn("movie_1", entity_names[:-1], "相似实体结果不应包含自身")

    def test_predict_relation_score(self):
        """测试三元组合理性预测功能"""
        # 测试合理的三元组
        score = self.inference_engine.predict_relation_score("user_1", "RATED", "movie_1")
        self.assertIsInstance(score, float, "预测分数应该是浮点数")
        self.assertGreater(score, 0, "预测分数应该大于0")

        # 测试不合理的三元组
        bad_score = self.inference_engine.predict_relation_score("user_1", "RATED", "genre_comedy")
        self.assertIsInstance(bad_score, float, "预测分数应该是浮点数")

    def test_get_entity_neighbors(self):
        """测试获取实体邻居功能"""
        # 获取movie_1的邻居
        neighbors = self.inference_engine.get_entity_neighbors("movie_1")
        self.assertIn("Genre", neighbors, "movie_1应该有关联的类型实体")
        self.assertGreater(len(neighbors["Genre"]), 0, "movie_1应该有关联的类型实体")

        # 测试指定关系类型
        genre_neighbors = self.inference_engine.get_entity_neighbors("movie_1", relation_type="HAS_GENRE")
        self.assertIn("Genre", genre_neighbors, "应该返回类型邻居")
        self.assertNotIn("user", genre_neighbors, "不应该返回用户邻居")

    def test_recommend_items(self):
        """测试物品推荐功能"""
        # 为user_1推荐电影
        recommendations = self.inference_engine.recommend_items("user_1", top_k=5)
        self.assertEqual(len(recommendations), 5, "应该返回5个推荐结果")
        self.assertIsInstance(recommendations[0][1], float, "推荐分数应该是浮点数")

        # 验证推荐结果中不应包含已交互物品
        user1_items = set()
        for neighbor, _ in self.inference_engine.graph["user_1"].items():
            if self.inference_engine.graph.nodes[neighbor].get("type") == "movie":
                user1_items.add(neighbor)

        recommended_items = [item for item, _ in recommendations]
        for item in recommended_items:
            self.assertNotIn(item, user1_items, "推荐结果不应包含用户已交互物品")

    def test_recommend_for_new_user(self):
        """测试为新用户推荐热门物品"""
        # 为新用户(无交互)推荐电影
        recommendations = self.inference_engine.recommend_items("user_3", top_k=5)
        self.assertEqual(len(recommendations), 5, "应该返回5个推荐结果")
        self.assertIsInstance(recommendations[0][1], float, "推荐分数应该是浮点数")

if __name__ == "__main__":
    unittest.main(verbosity=2)