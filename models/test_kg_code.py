import os
import unittest
import tempfile
import shutil
import pandas as pd
import networkx as nx
from kg_construction import KnowledgeGraphConstructor
from data_to_kg import DataToKGConverter
from kg_embeddings import KnowledgeGraphEmbeddingModel

class TestKnowledgeGraphCode(unittest.TestCase):
    """知识图谱代码测试套件"""
    @classmethod
    def setUpClass(cls):
        """测试前的准备工作，创建临时目录和示例数据"""
        # 创建临时目录
        cls.temp_dir = tempfile.mkdtemp()
        cls.dataset_dir = os.path.join(cls.temp_dir, "ml-latest-small")
        os.makedirs(cls.dataset_dir, exist_ok=True)
        cls.kg_save_path = os.path.join(cls.temp_dir, "test_kg.graph")

        # 创建示例数据
        cls._create_sample_data()
        print(f"测试环境准备完成，临时目录: {cls.temp_dir}")

    @classmethod
    def tearDownClass(cls):
        """测试结束后清理临时文件"""
        shutil.rmtree(cls.temp_dir)
        print(f"测试环境清理完成")

    @classmethod
    def _create_sample_data(cls):
        """创建测试用的MovieLens示例数据"""
        # 创建movies.csv
        movies_data = {
            "movieId": [1, 2, 3],
            "title": [
                "Toy Story (1995)",
                "Jumanji (1995)",
                "Grumpier Old Men (1995)"
            ],
            "genres": [
                "Adventure|Animation|Children|Comedy|Fantasy",
                "Adventure|Children|Fantasy",
                "Comedy|Romance"
            ]
        }
        pd.DataFrame(movies_data).to_csv(os.path.join(cls.dataset_dir, "movies.csv"), index=False)

        # 创建ratings.csv
        ratings_data = {
            "userId": [1, 1, 2, 2, 3],
            "movieId": [1, 2, 1, 3, 2],
            "rating": [4.0, 3.5, 5.0, 2.5, 4.5],
            "timestamp": [1112486027, 1112484676, 1112484819, 1112484727, 1112484580]
        }
        pd.DataFrame(ratings_data).to_csv(os.path.join(cls.dataset_dir, "ratings.csv"), index=False)

        # 创建tags.csv
        tags_data = {
            "userId": [1, 2, 3],
            "movieId": [1, 3, 1],
            "tag": ["funny", "romantic", "animated"],
            "timestamp": [1112486027, 1112484819, 1112484580]
        }
        pd.DataFrame(tags_data).to_csv(os.path.join(cls.dataset_dir, "tags.csv"), index=False)

    def test_01_kg_construction(self):
        """测试知识图谱构建功能"""
        print("\n=== 测试知识图谱构建 ===")
        constructor = KnowledgeGraphConstructor()

        # 加载测试数据
        movies_df = pd.read_csv(os.path.join(self.dataset_dir, "movies.csv"))
        ratings_df = pd.read_csv(os.path.join(self.dataset_dir, "ratings.csv"))
        tags_df = pd.read_csv(os.path.join(self.dataset_dir, "tags.csv"))

        # 构建图谱
        constructor.build_graph_from_data(movies_df, ratings_df, tags_df)

        # 验证节点和关系数量
        self.assertGreater(len(constructor.graph.nodes), 0, "图谱中没有节点被创建")
        self.assertGreater(len(constructor.graph.edges), 0, "图谱中没有关系被创建")

        # 验证特定实体和关系是否存在
        has_user = any("user" in str(node).lower() for node in constructor.graph.nodes)
        has_movie = any("movie" in str(node).lower() for node in constructor.graph.nodes)
        self.assertTrue(has_user, "图谱中没有用户实体")
        self.assertTrue(has_movie, "图谱中没有电影实体")

        # 保存图谱
        constructor.save_graph(self.kg_save_path)
        self.assertTrue(os.path.exists(self.kg_save_path), "图谱保存失败")
        print("知识图谱构建测试通过")

    def test_02_data_to_kg_conversion(self):
        """测试数据到知识图谱的转换功能"""
        print("\n=== 测试数据到知识图谱转换 ===")
        converter = DataToKGConverter(
            dataset_dir=self.dataset_dir,
            kg_save_path=self.kg_save_path,
            neo4j_config=None  # 不使用Neo4j
        )

        # 执行转换
        success, message = converter.convert_to_kg(sample_size=10)
        self.assertTrue(success, f"数据转换失败: {message}")

        # 验证输出
        self.assertTrue(os.path.exists(self.kg_save_path), "转换后未生成图谱文件")

        # 加载并验证转换后的图谱
        loaded_constructor = KnowledgeGraphConstructor()
        loaded_constructor.load_graph(self.kg_save_path)
        self.assertIsNotNone(loaded_constructor, "图谱加载失败")
        self.assertGreater(len(loaded_constructor.graph.nodes), 0, "加载的图谱中没有节点")
        print("数据到知识图谱转换测试通过")

    def test_03_kg_embeddings(self):
        """测试知识图谱嵌入生成功能"""
        print("\n=== 测试知识图谱嵌入 ===")
        if not os.path.exists(self.kg_save_path):
            self.skipTest("知识图谱文件不存在，跳过嵌入测试")

        # 加载图谱
        constructor = KnowledgeGraphConstructor()
        constructor.load_graph(self.kg_save_path)

        # 创建嵌入模型
        try:
            embedding_model = KnowledgeGraphEmbeddingModel(
                graph=constructor.graph,
                embedding_dim=50,
                model_type="TransE",
                margin=1.0,
                learning_rate=0.001
            )
        except Exception as e:
            self.fail(f"嵌入模型初始化失败: {str(e)}")

        # 测试训练（少量epochs）
        try:
            metrics = embedding_model.train(epochs=10, batch_size=16)
            self.assertIsInstance(metrics, dict, "训练未返回指标")
            self.assertIn("loss", metrics, "训练指标中没有损失值")
        except Exception as e:
            self.fail(f"嵌入模型训练失败: {str(e)}")

        # 测试保存嵌入
        try:
            temp_emb_dir = tempfile.mkdtemp()
            embedding_model.save_embeddings(save_dir=temp_emb_dir)
            self.assertTrue(os.path.exists(os.path.join(temp_emb_dir, "entity_embeddings.npy")), "实体嵌入保存失败")
            self.assertTrue(os.path.exists(os.path.join(temp_emb_dir, "relation_embeddings.npy")), "关系嵌入保存失败")
            shutil.rmtree(temp_emb_dir)
        except Exception as e:
            self.fail(f"嵌入保存失败: {str(e)}")
        print("知识图谱嵌入测试通过")

if __name__ == "__main__":
    # 运行所有测试
    unittest.main(verbosity=2)

# 使用说明:
# 1. 确保已安装所有依赖: pip install pandas networkx torch
# 2. 运行测试脚本: python test_kg_code.py
# 3. 查看输出结果，确认所有测试通过