import networkx as nx
import pandas as pd
import pickle
from typing import Dict, List, Tuple

class KnowledgeGraphConstructor:
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.entity_types = {
            'user': 'User',
            'movie': 'Movie',
            'genre': 'Genre',
            'tag': 'Tag'
        }
        self.relation_types = {
            'rating': 'RATED',
            'has_genre': 'HAS_GENRE',
            'has_tag': 'HAS_TAG',
            'tagged_by': 'TAGGED_BY'
        }

    def load_movie_data(self, movies_path: str, sample_frac: float = None, sample_size: int = None) -> pd.DataFrame:
        """加载电影数据并提取类型实体"""
        movies_df = pd.read_csv(movies_path)
        # 强制使用极小采样率进行测试
        sample_frac = 0.001  # 0.1%采样率
        movies_df = movies_df.sample(frac=sample_frac, random_state=42)
        print(f"强制应用电影采样率: {sample_frac}, 采样后电影数量: {len(movies_df)}")
        # 提取电影ID、标题和类型
        movies_df['genres'] = movies_df['genres'].str.split('|')
        return movies_df

    def load_rating_data(self, ratings_path: str, movie_ids: set = None, sample_frac: float = None, sample_size: int = None) -> pd.DataFrame:
        """加载评分数据并按电影ID过滤"""
        # 分块读取以减少内存占用
        chunk_iter = pd.read_csv(ratings_path, chunksize=10000, usecols=['userId', 'movieId', 'rating', 'timestamp'])
        ratings_df = pd.concat([chunk[chunk['movieId'].isin(movie_ids)] for chunk in chunk_iter]) if movie_ids else pd.concat(chunk_iter)
        
        # 应用采样
        if sample_frac is not None:
            ratings_df = ratings_df.sample(frac=sample_frac, random_state=42)
        elif sample_size is not None:
            ratings_df = ratings_df.sample(n=sample_size, random_state=42)
        return ratings_df

    def load_tag_data(self, tags_path: str, movie_ids: set = None, sample_frac: float = None, sample_size: int = None) -> pd.DataFrame:
        """加载标签数据并按电影ID过滤"""
        # 分块读取以减少内存占用
        chunk_iter = pd.read_csv(tags_path, chunksize=10000, usecols=['userId', 'movieId', 'tag', 'timestamp'])
        tags_df = pd.concat([chunk[chunk['movieId'].isin(movie_ids)] for chunk in chunk_iter]) if movie_ids else pd.concat(chunk_iter)
        
        # 将tag列转换为字符串并处理NaN值
        tags_df['tag'] = tags_df['tag'].astype(str).fillna('')
        
        # 应用采样
        if sample_frac is not None:
            tags_df = tags_df.sample(frac=sample_frac, random_state=42)
        elif sample_size is not None:
            tags_df = tags_df.sample(n=sample_size, random_state=42)
        return tags_df

    def add_user_entities(self, user_ids: List[int]) -> None:
        """添加用户实体"""
        for user_id in user_ids:
            self.graph.add_node(
                f"user_{user_id}",
                type='user',
                id=user_id
            )

    def add_movie_entities(self, movies_df: pd.DataFrame) -> None:
        """添加电影实体及类型关系"""
        for _, row in movies_df.iterrows():
            movie_id = row['movieId']
            # 添加电影实体
            self.graph.add_node(
                f"movie_{movie_id}",
                type='movie',
                id=movie_id,
                title=row['title'],
                release_year=self._extract_release_year(row['title'])
            )
            
            # 添加类型实体及关系
            for genre in row['genres']:
                genre_id = genre.lower().replace(' ', '_')
                # 添加类型实体（如果不存在）
                if f"genre_{genre_id}" not in self.graph.nodes:
                    self.graph.add_node(
                        f"genre_{genre_id}",
                        type=self.entity_types['genre'],
                        name=genre
                    )
                # 添加电影-类型关系
                self.graph.add_edge(
                    f"movie_{movie_id}",
                    f"genre_{genre_id}",
                    type=self.relation_types['has_genre'],
                    weight=1.0
                )

    def add_rating_relations(self, ratings_df: pd.DataFrame) -> None:
        """添加用户-电影评分关系"""
        for _, row in ratings_df.iterrows():
            user_node = f"user_{row['userId']}"
            movie_node = f"movie_{row['movieId']}"
            
            # 如果用户节点不存在则创建
            if user_node not in self.graph.nodes:
                self.graph.add_node(
                    user_node,
                    type=self.entity_types['user'],
                    id=row['userId']
                )
            
            # 添加评分关系
            self.graph.add_edge(
                user_node,
                movie_node,
                type="RATED",
                rating=row['rating'],
                timestamp=row['timestamp']
            )

    def add_tag_entities_and_relations(self, tags_df: pd.DataFrame) -> None:
        """添加标签实体及关系"""
        for _, row in tags_df.iterrows():
            user_node = f"user_{row['userId']}"
            movie_node = f"movie_{row['movieId']}"
            tag_text = row['tag'].lower().replace(' ', '_')
            tag_node = f"tag_{tag_text}"
            
            # 创建标签实体（如果不存在）
            if tag_node not in self.graph.nodes:
                self.graph.add_node(
                    tag_node,
                    type=self.entity_types['tag'],
                    name=row['tag']
                )
            
            # 如果用户节点不存在则创建
            if user_node not in self.graph.nodes:
                self.graph.add_node(
                    user_node,
                    type=self.entity_types['user'],
                    id=row['userId']
                )
            
            # 添加电影-标签关系
            self.graph.add_edge(
                movie_node,
                tag_node,
                type=self.relation_types['has_tag'],
                timestamp=row['timestamp']
            )
            
            # 添加用户-标签关系
            self.graph.add_edge(
                user_node,
                tag_node,
                type=self.relation_types['tagged_by'],
                timestamp=row['timestamp']
            )

    def _extract_release_year(self, title: str) -> int:
        """从电影标题中提取发行年份"""
        try:
            if title.endswith(')'):
                year = int(title[-5:-1])
                return year if 1900 <= year <= 2025 else None
        except ValueError:
            pass
        return None

    def build_from_files(self, data_dir: str, 
                         movie_sample_frac: float = None, movie_sample_size: int = None,
                         rating_sample_frac: float = None, rating_sample_size: int = None,
                         tag_sample_frac: float = None, tag_sample_size: int = None) -> nx.MultiDiGraph:
        """从文件构建知识图谱"""
        import time
        import os
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 开始从{data_dir}加载数据...", flush=True)
        start_time = time.time()
        print(f"[DEBUG] build_from_files参数: movie_sample_frac={movie_sample_frac}, movie_sample_size={movie_sample_size}")
        
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 加载电影数据...", flush=True)
        print(f"[DEBUG] 调用load_movie_data: sample_frac={movie_sample_frac}, sample_size={movie_sample_size}")
        movies_df = self.load_movie_data(
            os.path.join(data_dir, 'movies.csv'),
            sample_frac=movie_sample_frac,
            sample_size=movie_sample_size
        )
        print(f"[DEBUG] 电影数据加载完成: {len(movies_df)}部电影")
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 已加载{len(movies_df)}部电影数据", flush=True)
        
        # 获取采样后的电影ID集合
        sampled_movie_ids = set(movies_df['movieId'].unique())
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 采样后电影ID数量: {len(sampled_movie_ids)}", flush=True)
        
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 加载评分数据...", flush=True)
        # 先加载全部评分再过滤采样电影
        ratings_df = self.load_rating_data(
            os.path.join(data_dir, 'ratings.csv'),
            movie_ids=sampled_movie_ids,
            sample_frac=rating_sample_frac,
            sample_size=rating_sample_size
        )
        # 应用评分采样
        if rating_sample_frac is not None:
            ratings_df = ratings_df.sample(frac=rating_sample_frac, random_state=42)
        elif rating_sample_size is not None:
            ratings_df = ratings_df.sample(n=rating_sample_size, random_state=42)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 过滤采样后评分数量: {len(ratings_df)}", flush=True)
        
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 加载标签数据...", flush=True)
        # 先加载全部标签再过滤采样电影
        tags_df = self.load_tag_data(
            os.path.join(data_dir, 'tags.csv'),
            movie_ids=sampled_movie_ids,
            sample_frac=tag_sample_frac,
            sample_size=tag_sample_size
        )
        # 应用标签采样
        if tag_sample_frac is not None:
            tags_df = tags_df.sample(frac=tag_sample_frac, random_state=42)
        elif tag_sample_size is not None:
            tags_df = tags_df.sample(n=tag_sample_size, random_state=42)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 过滤采样后标签数量: {len(tags_df)}", flush=True)
        
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 开始构建知识图谱...", flush=True)
        graph = self.build_graph_from_data(movies_df, ratings_df, tags_df)
        
        end_time = time.time()
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 知识图谱构建完成，耗时{end_time - start_time:.2f}秒", flush=True)
        return graph

    def save_graph(self, output_path: str) -> None:
        """保存知识图谱"""
        with open(output_path, 'wb') as f:
            pickle.dump(self.graph, f)
        print(f"知识图谱已保存至: {output_path}")

    def load_graph(self, input_path: str) -> nx.MultiDiGraph:
        """加载知识图谱"""
        with open(input_path, 'rb') as f:
            self.graph = pickle.load(f)
        print(f"知识图谱已从{input_path}加载: {len(self.graph.nodes)}个实体, {len(self.graph.edges)}条关系")
        return self.graph

    def build_graph_from_data(self, movies_df: pd.DataFrame, ratings_df: pd.DataFrame, tags_df: pd.DataFrame) -> nx.MultiDiGraph:
        """从DataFrame构建知识图谱"""
        # 添加实体
        self.add_movie_entities(movies_df)
        self.add_user_entities(ratings_df['userId'].unique().tolist())

        # 添加关系
        self.add_rating_relations(ratings_df)
        self.add_tag_entities_and_relations(tags_df)

        print(f"知识图谱构建完成: {len(self.graph.nodes)}个实体, {len(self.graph.edges)}条关系")
        return self.graph

if __name__ == "__main__":
    # 示例用法
    from pathlib import Path
    kg_constructor = KnowledgeGraphConstructor()
    # 动态获取项目根目录并构建数据路径
    data_dir = Path(__file__).parent.parent / "ml-20m/ml-20m"
    kg = kg_constructor.build_from_files(data_dir=data_dir)
    # 创建kg_data目录（如果不存在）
    import os
    os.makedirs("models/kg_data", exist_ok=True)
    kg_constructor.save_graph("models/kg_data/kg.graph")