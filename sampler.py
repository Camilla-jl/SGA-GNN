# MENTOR/src/utils_package/sampler.py

import numpy as np
import torch
import torch.nn.functional as F

class HardNegativeSampler:
    def __init__(self, train_matrix_csr, item_features, device, top_k=20):
        print("Initializing Hard Negative Sampler...")
        self.train_matrix_csr = train_matrix_csr
        self.device = device
        self.top_k = top_k
        
        self.item_similarity = self._calculate_item_similarity(item_features)
        self.candidate_items = self._get_candidate_items()
        
        print(f"Hard Negative Sampler initialized (top_k={self.top_k}).")

    def _calculate_item_similarity(self, item_features):
        features = item_features.to(self.device)
        norm_features = F.normalize(features, p=2, dim=1)
        sim_matrix = torch.matmul(norm_features, norm_features.t())
        sim_matrix.fill_diagonal_(-torch.inf)
        return sim_matrix

    def _get_candidate_items(self):
        _, top_k_indices = torch.topk(self.item_similarity, k=self.top_k, dim=1)
        return top_k_indices.cpu().numpy()

    def sample(self, user_ids, pos_item_ids):
        batch_size = user_ids.size(0)
        neg_item_ids = torch.zeros(batch_size, dtype=torch.long)
        
        user_ids_cpu = user_ids.cpu().numpy()
        pos_item_ids_cpu = pos_item_ids.cpu().numpy()

        for i in range(batch_size):
            user = user_ids_cpu[i]
            pos_item = pos_item_ids_cpu[i]
            user_pos_history = self.train_matrix_csr.getrow(user).indices
            
            hard_candidates = self.candidate_items[pos_item]
            
            # 从困难候选集中不断随机采样，直到找到一个真正的负样本
            while True:
                neg_candidate = np.random.choice(hard_candidates)
                if neg_candidate not in user_pos_history:
                    neg_item_ids[i] = neg_candidate
                    break
        
        return neg_item_ids.to(self.device)