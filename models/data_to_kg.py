import pandas as pd
import os
import logging
from typing import Optional, Tuple
from kg_construction import KnowledgeGraphConstructor
from neo4j_adapter import Neo4jAdapter

class DataToKGConverter:
    def __init__(self,
                 dataset_dir: str = "data/ml-latest-small",
                 kg_save_path: str = "models/kg_data/kg.graph",
                 neo4j_config: Optional[dict] = None):
        """
        初始化数据到知识图谱转换器
        :param dataset_dir: MovieLens数据集目录
        :param kg_save_path: 知识图谱保存路径
        :param neo4j_config: Neo4j数据库配置字典，格式: {"uri": "...", "user": "...", "password": "..."}
        """
        self.dataset_dir = dataset_dir
        self.kg_save_path = kg_save_path
        self.neo4j_config = neo4j_config
        self.kg_constructor = KnowledgeGraphConstructor()
        self.neo4j_adapter = Neo4jAdapter(**neo4j_config) if neo4j_config else None
        self.logger = logging.getLogger("DataToKGConverter")
        logging.basicConfig(level=logging.INFO)

        # 创建保存目录
        os.makedirs(os.path.dirname(kg_save_path), exist_ok=True)

    def load_movies_data(self) -> Optional[pd.DataFrame]:
        """加载电影数据"""
        movies_path = os.path.join(self.dataset_dir, "movies.csv")
        try:
            df = pd.read_csv(movies_path)
            self.logger.info(f"已加载电影数据: {len(df)}条记录")
            # 提取电影类别
            df["genres_list"] = df["genres"].str.split("|")
            return df
        except FileNotFoundError:
            self.logger.error(f"电影数据文件未找到: {movies_path}")
            return None
        except Exception as e:
            self.logger.error(f"加载电影数据失败: {str(e)}")
            return None

    def load_ratings_data(self) -> Optional[pd.DataFrame]:
        """加载评分数据"""
        ratings_path = os.path.join(self.dataset_dir, "ratings.csv")
        try:
            df = pd.read_csv(ratings_path)
            self.logger.info(f"已加载评分数据: {len(df)}条记录")
            return df
        except FileNotFoundError:
            self.logger.error(f"评分数据文件未找到: {ratings_path}")
            return None
        except Exception as e:
            self.logger.error(f"加载评分数据失败: {str(e)}")
            return None

    def load_tags_data(self) -> Optional[pd.DataFrame]:
        """加载标签数据"""
        tags_path = os.path.join(self.dataset_dir, "tags.csv")
        try:
            df = pd.read_csv(tags_path)
            self.logger.info(f"已加载标签数据: {len(df)}条记录")
            return df
        except FileNotFoundError:
            self.logger.warning(f"标签数据文件未找到: {tags_path}，将跳过标签处理")
            return None
        except Exception as e:
            self.logger.error(f"加载标签数据失败: {str(e)}")
            return None

    def convert_to_kg(self,
                     include_ratings: bool = True,
                     include_tags: bool = True,
                     sample_size: Optional[int] = None) -> Tuple[bool, str]:
        """
        将MovieLens数据集转换为知识图谱
        :param include_ratings: 是否包含评分关系
        :param include_tags: 是否包含标签关系
        :param sample_size: 采样大小，None表示使用全部数据
        :return: (转换成功与否, 消息)
        """
        try:
            # 1. 加载数据
            movies_df = self.load_movies_data()
            if movies_df is None:
                return False, "无法加载电影数据，转换失败"

            ratings_df = self.load_ratings_data() if include_ratings else None
            tags_df = self.load_tags_data() if include_tags else None

            # 2. 采样数据（如果需要）
            if sample_size and sample_size > 0:
                movies_df = movies_df.sample(min(sample_size, len(movies_df)))
                if ratings_df is not None:
                    # 只保留采样电影的评分
                    sampled_movie_ids = set(movies_df["movieId"])
                    ratings_df = ratings_df[ratings_df["movieId"].isin(sampled_movie_ids)]
                    # 进一步采样评分数据
                    ratings_df = ratings_df.sample(min(sample_size * 10, len(ratings_df)))

            # 3. 构建知识图谱
            self.logger.info("开始构建知识图谱...")
            self.kg_constructor.build_graph_from_data(
                movies_df=movies_df,
                ratings_df=ratings_df,
                tags_df=tags_df
            )

            # 4. 保存知识图谱
            self.kg_constructor.save_graph(self.kg_save_path)
            self.logger.info(f"知识图谱已保存到: {self.kg_save_path}")

            # 5. 如果配置了Neo4j，将图谱导入数据库
            if self.neo4j_adapter:
                self.logger.info("开始将知识图谱导入Neo4j数据库...")
                if self.neo4j_adapter.connect():
                    stats = self.neo4j_adapter.batch_import_from_networkx(
                        self.kg_constructor.graph
                    )
                    self.neo4j_adapter.close()
                    self.logger.info(f"Neo4j导入统计: {stats}")
                    if stats["nodes_created"] == 0:
                        return False, "知识图谱导入Neo4j失败，未创建任何节点"
                else:
                    return False, "无法连接到Neo4j数据库"

            return True, f"知识图谱构建成功，包含{self.kg_constructor.graph.number_of_nodes()}个节点和{self.kg_constructor.graph.number_of_edges()}条关系"

        except Exception as e:
            self.logger.error(f"知识图谱转换失败: {str(e)}", exc_info=True)
            return False, f"转换失败: {str(e)}"

if __name__ == "__main__":
    # 示例用法
    converter = DataToKGConverter(
        dataset_dir="data/ml-latest-small",
        kg_save_path="models/kg_data/kg.graph",
        # 取消注释以下行以启用Neo4j导入
        # neo4j_config={"uri": "bolt://localhost:7687", "user": "neo4j", "password": "password"}
    )
    
    # 转换为知识图谱（使用1000部电影的采样数据进行测试）
    success, message = converter.convert_to_kg(sample_size=1000)
    print(f"转换结果: {message}")
    
    if success:
        # 加载并显示图谱基本信息
        kg = KnowledgeGraphConstructor.load_graph("models/kg_data/kg.graph")
        print(f"加载的知识图谱: {kg.number_of_nodes()}个节点, {kg.number_of_edges()}条关系")
        
        # 显示一些统计信息
        print("节点类型统计:")
        node_types = [kg.nodes[n].get("type", "Unknown") for n in kg.nodes()]
        for type_name, count in pd.Series(node_types).value_counts().items():
            print(f"- {type_name}: {count}个")
        
        print("关系类型统计:")
        edge_types = [kg.edges[e].get("type", "Unknown") for e in kg.edges()]
        for type_name, count in pd.Series(edge_types).value_counts().items():
            print(f"- {type_name}: {count}条")