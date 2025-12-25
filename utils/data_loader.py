"""
Data loader for MovieLens dataset, adapted from KGRec reference implementation
"""
import numpy as np
from tqdm import tqdm
import networkx as nx
import scipy.sparse as sp
import pandas as pd
import os
import random
from time import time
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

n_users = 0
n_items = 0
n_entities = 0
n_relations = 0
n_nodes = 0
train_user_set = defaultdict(list)
test_user_set = defaultdict(list)


def read_cf_movielens(file_name):
    """Read MovieLens ratings file and convert to interaction matrix format"""
    df = pd.read_csv(file_name)
    # Filter positive interactions (rating >= 4)
    df = df[df['rating'] >= 4.0]
    inter_mat = []
    for _, row in df.iterrows():
        inter_mat.append([int(row['userId']), int(row['movieId'])])
    return np.array(inter_mat)


def read_cf(file_name):
    """Read interaction file (compatible with both formats)"""
    if file_name.endswith('.csv'):
        return read_cf_movielens(file_name)
    else:
        # Original format: space-separated, first element is user_id, rest are item_ids
        inter_mat = list()
        lines = open(file_name, "r").readlines()
        for l in lines:
            tmps = l.strip()
            inters = [int(i) for i in tmps.split(" ")]
            u_id, pos_ids = inters[0], inters[1:]
            pos_ids = list(set(pos_ids))
            for i_id in pos_ids:
                inter_mat.append([u_id, i_id])
        return np.array(inter_mat)


def remap_item(train_data, test_data):
    global n_users, n_items
    n_users = max(max(train_data[:, 0]), max(test_data[:, 0])) + 1
    n_items = max(max(train_data[:, 1]), max(test_data[:, 1])) + 1

    for u_id, i_id in train_data:
        train_user_set[int(u_id)].append(int(i_id))
    for u_id, i_id in test_data:
        test_user_set[int(u_id)].append(int(i_id))


def read_triplets_movielens(kg_file, movies_file=None):
    """Read knowledge graph triplets from MovieLens data"""
    global n_entities, n_relations, n_nodes
    
    # If kg_file is a pickle file (NetworkX graph), load it
    if kg_file.endswith('.graph') or kg_file.endswith('.pkl'):
        import pickle
        with open(kg_file, 'rb') as f:
            graph = pickle.load(f)
        
        # Convert NetworkX graph to triplets format
        triplets = []
        for u, v, data in graph.edges(data=True):
            if 'relation' in data:
                r = data['relation']
            elif 'key' in data:
                r = data['key']
            else:
                r = 0
            triplets.append([u, r, v])
        
        triplets = np.array(triplets, dtype=np.int32)
    else:
        # Original format: text file with triplets
        can_triplets_np = np.loadtxt(kg_file, dtype=np.int32)
        triplets = can_triplets_np
    
    can_triplets_np = np.unique(triplets, axis=0)
    
    # Add inverse relations if needed
    global args_global
    if 'args_global' in globals() and hasattr(args_global, 'inverse_r') and args_global.inverse_r:
        inv_triplets_np = can_triplets_np.copy()
        inv_triplets_np[:, 0] = can_triplets_np[:, 2]
        inv_triplets_np[:, 2] = can_triplets_np[:, 0]
        inv_triplets_np[:, 1] = can_triplets_np[:, 1] + max(can_triplets_np[:, 1]) + 1
        can_triplets_np[:, 1] = can_triplets_np[:, 1] + 1
        inv_triplets_np[:, 1] = inv_triplets_np[:, 1] + 1
        triplets = np.concatenate((can_triplets_np, inv_triplets_np), axis=0)
    else:
        can_triplets_np[:, 1] = can_triplets_np[:, 1] + 1
        triplets = can_triplets_np.copy()

    n_entities = max(max(triplets[:, 0]), max(triplets[:, 2])) + 1
    n_nodes = n_entities + n_users
    n_relations = max(triplets[:, 1]) + 1

    return triplets


def read_triplets(file_name):
    """Read triplets file (compatible with both formats)"""
    if file_name.endswith('.graph') or file_name.endswith('.pkl'):
        return read_triplets_movielens(file_name)
    else:
        return read_triplets_movielens(file_name)


def build_graph(train_data, triplets):
    ckg_graph = nx.MultiDiGraph()
    rd = defaultdict(list)

    print("Begin to load interaction triples ...")
    for u_id, i_id in tqdm(train_data, ascii=True):
        rd[0].append([u_id, i_id])

    print("\nBegin to load knowledge graph triples ...")
    for h_id, r_id, t_id in tqdm(triplets, ascii=True):
        ckg_graph.add_edge(h_id, t_id, key=r_id)

    return ckg_graph, rd


def build_sparse_relational_graph(relation_dict):
    def _bi_norm_lap(adj):
        # D^{-1/2}AD^{-1/2}
        rowsum = np.array(adj.sum(1))

        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

        # 修复维度匹配问题：确保矩阵维度正确
        # bi_lap = D^{-1/2} A D^{-1/2}
        bi_lap = d_mat_inv_sqrt.dot(adj)
        # 需要转置d_mat_inv_sqrt以匹配列维度
        if adj.shape[0] == adj.shape[1]:
            # 如果是方阵，使用相同的d_mat_inv_sqrt
            bi_lap = bi_lap.dot(d_mat_inv_sqrt)
        else:
            # 如果不是方阵，需要为列创建单独的归一化
            colsum = np.array(adj.sum(0))
            d_col_inv_sqrt = np.power(colsum, -0.5).flatten()
            d_col_inv_sqrt[np.isinf(d_col_inv_sqrt)] = 0.
            d_mat_col_inv_sqrt = sp.diags(d_col_inv_sqrt)
            bi_lap = bi_lap.dot(d_mat_col_inv_sqrt)
        return bi_lap.tocoo()

    def _si_norm_lap(adj):
        # D^{-1}A
        rowsum = np.array(adj.sum(1))

        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)

        norm_adj = d_mat_inv.dot(adj)
        return norm_adj.tocoo()

    adj_mat_list = []
    norm_mat_list = []
    mean_mat_list = []

    print("building adjacency matrix")
    for r_id in relation_dict.keys():
        np_mat = np.array(relation_dict[r_id])
        if len(np_mat) == 0:
            continue
        adj_mat = sp.coo_matrix((np.ones(len(np_mat)), (np_mat[:, 0], np_mat[:, 1])),
                                shape=(n_users, n_items), dtype=np.float32)
        adj_mat_list.append(adj_mat)

        # normalized adj matrix
        norm_mat = _bi_norm_lap(adj_mat)
        norm_mat_list.append(norm_mat)

        # mean adj matrix
        mean_mat = _si_norm_lap(adj_mat)
        mean_mat_list.append(mean_mat)

    return adj_mat_list, norm_mat_list, mean_mat_list


def load_data(args):
    """
    Load MovieLens dataset and convert to KGRec format
    """
    global n_users, n_items, n_entities, n_relations, n_nodes
    global train_user_set, test_user_set
    
    # Store args globally for use in other functions
    global args_global
    args_global = args
    
    data_path = args.data_path
    dataset = args.dataset
    
    print('reading train and test user-item set ...')
    train_file = os.path.join(data_path, 'train.txt')
    test_file = os.path.join(data_path, 'test.txt')
    
    # If train.txt/test.txt don't exist, try to create them from MovieLens format
    if not os.path.exists(train_file) or not os.path.exists(test_file):
        if dataset == 'ml-20m':
            # Try to load from MovieLens format
            ratings_file = os.path.join(data_path, 'ratings.csv')
            if os.path.exists(ratings_file):
                print('Creating train/test split from MovieLens ratings...')
                df = pd.read_csv(ratings_file)
                df = df[df['rating'] >= 4.0]  # Filter positive interactions
                
                # Split by user (leave-one-out)
                train_data = []
                test_data = []
                for user_id in df['userId'].unique():
                    user_ratings = df[df['userId'] == user_id].sort_values('timestamp')
                    if len(user_ratings) > 1:
                        train_data.append(user_ratings.iloc[:-1])
                        test_data.append(user_ratings.iloc[-1:])
                
                train_df = pd.concat(train_data)
                test_df = pd.concat(test_data)
                
                # Save in required format
                os.makedirs(data_path, exist_ok=True)
                with open(train_file, 'w') as f:
                    for user_id in train_df['userId'].unique():
                        items = train_df[train_df['userId'] == user_id]['movieId'].tolist()
                        f.write(f"{user_id} {' '.join(map(str, items))}\n")
                
                with open(test_file, 'w') as f:
                    for user_id in test_df['userId'].unique():
                        items = test_df[test_df['userId'] == user_id]['movieId'].tolist()
                        f.write(f"{user_id} {' '.join(map(str, items))}\n")
    
    train_cf = read_cf(train_file)
    test_cf = read_cf(test_file)
    
    remap_item(train_cf, test_cf)
    
    print('reading KG ...')
    kg_file = os.path.join(data_path, 'kg_final.txt')
    if not os.path.exists(kg_file):
        # Try to load from graph file
        kg_file = os.path.join('models', 'kg_data', 'kg.graph')
        if not os.path.exists(kg_file):
            raise FileNotFoundError(f"Knowledge graph file not found. Please create it first.")
    
    triplets = read_triplets(kg_file)
    
    print('building graph ...')
    graph, relation_dict = build_graph(train_cf, triplets)
    
    n_params = {
        'n_users': n_users,
        'n_items': n_items,
        'n_entities': n_entities,
        'n_relations': n_relations,
        'n_nodes': n_nodes
    }
    
    print('building sparse matrix ...')
    adj_mat_list, norm_mat_list, mean_mat_list = build_sparse_relational_graph(relation_dict)
    
    user_dict = {
        'train_user_set': train_user_set,
        'test_user_set': test_user_set,
    }
    
    return train_cf, test_cf, user_dict, n_params, graph, (adj_mat_list, norm_mat_list, mean_mat_list)

