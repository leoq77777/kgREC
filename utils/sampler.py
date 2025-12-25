import numpy as np
import random

class UniformSampler:
    """Uniform negative sampler for recommendation"""
    def __init__(self, seed=2020):
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
    
    def sample_negative(self, user_ids, n_items, train_user_dict, num_negatives=1):
        """
        Sample negative items for users
        :param user_ids: array of user IDs
        :param n_items: total number of items
        :param train_user_dict: dict mapping user_id to list of positive items
        :param num_negatives: number of negative samples per user
        :return: numpy array of negative item IDs (shape: [len(user_ids), num_negatives])
        """
        negatives = []
        for u_id in user_ids:
            u_id = int(u_id)
            pos_items = set(train_user_dict.get(u_id, []))
            neg_items = []
            
            for _ in range(num_negatives):
                neg_item = random.randint(0, n_items - 1)
                while neg_item in pos_items:
                    neg_item = random.randint(0, n_items - 1)
                neg_items.append(neg_item)
            
            # 返回单个值或列表，确保格式一致
            if num_negatives == 1:
                negatives.append(neg_items[0])
            else:
                negatives.append(neg_items)
        
        # 返回numpy数组，确保是2D
        result = np.array(negatives)
        if result.ndim == 1:
            result = result.reshape(-1, 1)
        return result

