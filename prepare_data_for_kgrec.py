"""
数据准备脚本：将MovieLens数据集转换为KGRec训练所需的格式
"""
import pandas as pd
import numpy as np
import os
import pickle
import networkx as nx
from collections import defaultdict
from tqdm import tqdm
import sys

def prepare_train_test_split(data_path, output_path):
    """准备训练/测试数据分割"""
    print("正在加载评分数据...")
    ratings_file = os.path.join(data_path, 'ratings.csv')
    df = pd.read_csv(ratings_file)
    
    # 过滤正样本（评分 >= 4.0）
    df = df[df['rating'] >= 4.0]
    print(f"过滤后正样本数量: {len(df)}")
    
    # 按用户分组，每个用户保留最后一个交互作为测试集
    train_data = []
    test_data = []
    
    print("正在创建训练/测试分割...")
    for user_id in tqdm(df['userId'].unique()):
        user_ratings = df[df['userId'] == user_id].sort_values('timestamp')
        if len(user_ratings) > 1:
            train_data.append(user_ratings.iloc[:-1])
            test_data.append(user_ratings.iloc[-1:])
        elif len(user_ratings) == 1:
            # 如果用户只有一个交互，放入训练集
            train_data.append(user_ratings)
    
    train_df = pd.concat(train_data)
    test_df = pd.concat(test_data)
    
    print(f"训练集大小: {len(train_df)}, 测试集大小: {len(test_df)}")
    
    # 保存为KGRec格式（每行：user_id item1 item2 ...）
    train_file = os.path.join(output_path, 'train.txt')
    test_file = os.path.join(output_path, 'test.txt')
    
    print("正在保存训练/测试文件...")
    with open(train_file, 'w') as f:
        for user_id in tqdm(train_df['userId'].unique()):
            items = train_df[train_df['userId'] == user_id]['movieId'].unique().tolist()
            f.write(f"{user_id} {' '.join(map(str, items))}\n")
    
    with open(test_file, 'w') as f:
        for user_id in tqdm(test_df['userId'].unique()):
            items = test_df[test_df['userId'] == user_id]['movieId'].unique().tolist()
            f.write(f"{user_id} {' '.join(map(str, items))}\n")
    
    print(f"训练/测试文件已保存到: {output_path}")
    return train_df, test_df

def prepare_kg_from_graph(kg_graph_path, output_path):
    """从NetworkX图文件准备知识图谱三元组"""
    print(f"正在从图文件加载知识图谱: {kg_graph_path}")
    
    if not os.path.exists(kg_graph_path):
        print(f"警告: 知识图谱文件不存在: {kg_graph_path}")
        print("将创建一个简单的知识图谱...")
        return create_simple_kg(output_path)
    
    with open(kg_graph_path, 'rb') as f:
        graph = pickle.load(f)
    
    print(f"图包含 {len(graph.nodes)} 个节点, {len(graph.edges)} 条边")
    
    # 转换图为三元组格式
    triplets = []
    relation_map = {}
    relation_id = 1  # 从1开始（0保留给交互关系）
    
    print("正在提取三元组...")
    for u, v, data in tqdm(graph.edges(data=True)):
        # 获取关系类型
        if 'relation' in data:
            rel_name = str(data['relation'])
        elif 'key' in data:
            rel_name = str(data['key'])
        elif 'type' in data:
            rel_name = str(data['type'])
        else:
            rel_name = 'unknown'
        
        # 映射关系到ID
        if rel_name not in relation_map:
            relation_map[rel_name] = relation_id
            relation_id += 1
        
        rel_id = relation_map[rel_name]
        
        # 确保节点ID是整数
        try:
            u_id = int(float(str(u).replace('movie_', '').replace('user_', '').replace('genre_', '').replace('tag_', '')))
            v_id = int(float(str(v).replace('movie_', '').replace('user_', '').replace('genre_', '').replace('tag_', '')))
        except:
            continue
        
        triplets.append([u_id, rel_id, v_id])
    
    triplets = np.array(triplets, dtype=np.int32)
    triplets = np.unique(triplets, axis=0)
    
    # 保存三元组文件
    kg_file = os.path.join(output_path, 'kg_final.txt')
    np.savetxt(kg_file, triplets, fmt='%d', delimiter='\t')
    
    print(f"知识图谱三元组已保存到: {kg_file}")
    print(f"三元组数量: {len(triplets)}, 关系类型数: {len(relation_map)}")
    
    return triplets

def create_simple_kg(output_path):
    """创建简单的知识图谱（如果图文件不存在）"""
    print("创建简单的知识图谱...")
    
    # 从movies.csv创建电影-类型关系
    movies_file = os.path.join('ml-20m', 'ml-20m', 'movies.csv')
    if os.path.exists(movies_file):
        movies_df = pd.read_csv(movies_file)
        triplets = []
        entity_id = max(movies_df['movieId'].max(), 100000)  # 确保实体ID不冲突
        
        genre_map = {}
        genre_id = 1
        
        for _, row in tqdm(movies_df.iterrows(), total=len(movies_df)):
            movie_id = int(row['movieId'])
            genres = str(row['genres']).split('|')
            
            for genre in genres:
                if genre and genre != '(no genres listed)':
                    if genre not in genre_map:
                        genre_map[genre] = entity_id
                        entity_id += 1
                    
                    genre_entity_id = genre_map[genre]
                    # 关系ID 1 表示 HAS_GENRE
                    triplets.append([movie_id, 1, genre_entity_id])
        
        triplets = np.array(triplets, dtype=np.int32)
        triplets = np.unique(triplets, axis=0)
        
        kg_file = os.path.join(output_path, 'kg_final.txt')
        np.savetxt(kg_file, triplets, fmt='%d', delimiter='\t')
        
        print(f"简单知识图谱已创建: {kg_file}")
        print(f"三元组数量: {len(triplets)}")
        
        return triplets
    else:
        print("警告: 无法创建知识图谱，movies.csv不存在")
        return None

def main():
    data_path = 'ml-20m/ml-20m'
    output_path = 'ml-20m/ml-20m'  # 输出到同一目录
    
    if not os.path.exists(data_path):
        print(f"错误: 数据路径不存在: {data_path}")
        sys.exit(1)
    
    print("=" * 50)
    print("准备KGRec训练数据")
    print("=" * 50)
    
    # 1. 准备训练/测试分割
    train_df, test_df = prepare_train_test_split(data_path, output_path)
    
    # 2. 准备知识图谱
    kg_graph_path = 'models/kg_data/kg.graph'
    prepare_kg_from_graph(kg_graph_path, output_path)
    
    print("=" * 50)
    print("数据准备完成！")
    print("=" * 50)
    print(f"训练文件: {os.path.join(output_path, 'train.txt')}")
    print(f"测试文件: {os.path.join(output_path, 'test.txt')}")
    print(f"知识图谱: {os.path.join(output_path, 'kg_final.txt')}")

if __name__ == '__main__':
    main()

