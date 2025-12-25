"""
Evaluation functions for KGRec
Adapted from reference implementation
"""
from .metrics import *
from .parser import parse_args_kgsr
import torch
import numpy as np
import multiprocessing
import heapq
from time import time
from .data_loader import train_user_set, test_user_set, n_items

cores = multiprocessing.cpu_count() // 2

# Parse args for evaluation
try:
    args = parse_args_kgsr()
    Ks = eval(args.Ks)
    device = torch.device("cuda:" + str(args.gpu_id)) if args.cuda else torch.device("cpu")
    BATCH_SIZE = args.test_batch_size
    batch_test_flag = args.batch_test_flag
except:
    # Default values if args not available
    Ks = [20]
    device = torch.device("cpu")
    BATCH_SIZE = 1024
    batch_test_flag = True


def ranklist_by_heapq(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = 0.
    return r, auc

def get_auc(item_score, user_pos_test):
    item_score = sorted(item_score.items(), key=lambda kv: kv[1])
    item_score.reverse()
    item_sort = [x[0] for x in item_score]
    posterior = [x[1] for x in item_score]

    r = []
    for i in item_sort:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = AUC(ground_truth=r, prediction=posterior)
    return auc

def ranklist_by_sorted(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = get_auc(item_score, user_pos_test)
    return r, auc

def get_performance(user_pos_test, r, auc, Ks):
    precision, recall, ndcg, hit_ratio = [], [], [], []

    for K in Ks:
        precision.append(precision_at_k(r, K))
        recall.append(recall_at_k(r, K, len(user_pos_test)))
        ndcg.append(ndcg_at_k(r, K, user_pos_test))
        hit_ratio.append(hit_at_k(r, K))

    return {'recall': np.array(recall), 'precision': np.array(precision),
            'ndcg': np.array(ndcg), 'hit_ratio': np.array(hit_ratio), 'auc': auc}


def test_one_user(x):
    # user u's ratings for user u
    rating = x[0]
    # uid
    u = x[1]
    # user u's items in the training set
    try:
        training_items = train_user_set[u]
    except Exception:
        training_items = []
    # user u's items in the test set
    user_pos_test = test_user_set[u]

    all_items = set(range(0, n_items))

    test_items = list(all_items - set(training_items))

    if batch_test_flag:
        r, auc = ranklist_by_heapq(user_pos_test, test_items, rating, Ks)
    else:
        r, auc = ranklist_by_sorted(user_pos_test, test_items, rating, Ks)

    return get_performance(user_pos_test, r, auc, Ks)


def test(model, user_dict, n_params):
    result = {'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks)),
              'precision': np.zeros(len(Ks)), 'hit_ratio': np.zeros(len(Ks))}

    global train_user_set, test_user_set, n_items
    train_user_set = user_dict['train_user_set']
    test_user_set = user_dict['test_user_set']
    n_items = n_params['n_items']

    u_g_embeddings, i_g_embeddings = model.generate()

    test_users = list(test_user_set.keys())
    test_users = test_users[:BATCH_SIZE] if batch_test_flag else test_users

    user_batch = torch.LongTensor(test_users).to(device)
    u_batch_embeddings = u_g_embeddings[user_batch]

    # batch-item test
    all_items = set(range(n_items))
    test_items = []
    for u in test_users:
        training_items = train_user_set.get(u, [])
        test_items.append(list(all_items - set(training_items)))

    item_batch = torch.LongTensor([item for sublist in test_items for item in sublist]).to(device)
    i_batch_embeddings = i_g_embeddings[item_batch]

    rating_batch = model.rating(u_batch_embeddings, i_batch_embeddings).cpu().detach().numpy()

    user_batch_rating_uid = zip(rating_batch, test_users)
    batch_result = []
    for x in user_batch_rating_uid:
        batch_result.append(test_one_user(x))

    for re in batch_result:
        result['recall'] += re['recall'] / len(test_users)
        result['ndcg'] += re['ndcg'] / len(test_users)
        result['precision'] += re['precision'] / len(test_users)
        result['hit_ratio'] += re['hit_ratio'] / len(test_users)

    return result

