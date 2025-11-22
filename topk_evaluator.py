# --- START OF NEW topk_evaluator.py ---

import os
import numpy as np
import pandas as pd
import torch
from utils_package.metrics import metrics_dict
from torch.nn.utils.rnn import pad_sequence
from utils_package.utils import get_local_time
from collections import defaultdict

# These metrics are typical in topk recommendations
topk_metrics = {metric.lower(): metric for metric in ['Recall', 'Precision', 'NDCG', 'MAP']}


class TopKEvaluator(object):
    r"""TopK Evaluator for both overall and group-based evaluation in ranking tasks."""

    def __init__(self, config, dataset):
        self.config = config
        self.dataset = dataset
        self.metrics = config['metrics']
        self.topk = config['topk']
        self.save_recom_result = config['save_recommended_topk']
        
        self._check_args()
        self._prepare_item_groups()

    def _prepare_item_groups(self):
        """Calculates item popularity and assigns each item to a group."""
        print("Preparing item groups for cold-start evaluation...")
        
        # 1. Get training interaction matrix to calculate popularity
        train_mat = self.dataset.inter_matrix(form='csr').astype(np.float32)
        
        # 2. Calculate popularity (number of interactions) for each item
        item_pop = np.array(train_mat.sum(axis=0)).squeeze()
        
        self.item_group_dict = {}
        # Define group thresholds based on your paper
        # Note: item IDs are 0-indexed
        for item_id, pop in enumerate(item_pop):
            if pop >= 5 and pop <= 10:
                self.item_group_dict[item_id] = 'Cold'
            elif pop >= 11 and pop <= 50:
                self.item_group_dict[item_id] = 'Medium'
            elif pop > 50:
                self.item_group_dict[item_id] = 'Hot'
            # Items with < 5 interactions are not in 5-core training set, 
            # but might appear in valid/test, we can label them or ignore.
            # For simplicity, we only focus on the defined groups.
        
        print(f"Item groups prepared. Found "
              f"{len([g for g in self.item_group_dict.values() if g == 'Cold'])} Cold items, "
              f"{len([g for g in self.item_group_dict.values() if g == 'Medium'])} Medium items, "
              f"{len([g for g in self.item_group_dict.values() if g == 'Hot'])} Hot items.")

    def evaluate(self, batch_matrix_list, eval_data, is_test=False, idx=0):
        """
        Calculates overall and group-based metrics.
        
        Returns:
            dict: A nested dictionary with overall and grouped results.
                  e.g., {'Overall': {'NDCG@20': 0.1}, 'Cold': {'NDCG@20': 0.05}}
        """
        # 1. Get ground truth and top-k prediction matrix
        pos_items_list = eval_data.get_eval_items()
        pos_len_list = eval_data.get_eval_len_list()
        topk_index = torch.cat(batch_matrix_list, dim=0).cpu().numpy()
        
        # 2. Get the boolean matrix indicating hit or not (for all users)
        bool_rec_matrix = []
        for gt_items, topk_preds in zip(pos_items_list, topk_index):
            bool_rec_matrix.append([True if pred_item in gt_items else False for pred_item in topk_preds])
        bool_rec_matrix = np.asarray(bool_rec_matrix)

        # 3. Calculate Overall performance
        overall_result = self._calculate_metrics_for_group(pos_len_list, bool_rec_matrix)
        
        final_metric_dict = {'Overall': overall_result}

        # 4. Calculate Group-based performance
        eval_users = eval_data.get_eval_users() # Needed to get the correct ground truth items
        
        for group_name in ['Cold', 'Medium', 'Hot']:
            # a. Filter users and ground truth based on the item group
            group_pos_len_list = []
            group_bool_rec_rows = []
            
            for i, user_id in enumerate(eval_users):
                # Get the ground truth items for this user
                user_gt_items = pos_items_list[i]
                
                # Find which of these ground truth items belong to the current group
                items_in_group = [item for item in user_gt_items if self.item_group_dict.get(item) == group_name]
                
                if not items_in_group:
                    continue # This user has no ground truth items in this group
                
                # This user contributes to this group's evaluation
                # The length of relevant items is the count of items in this group
                group_pos_len_list.append(len(items_in_group))
                # The hit matrix row remains the same, as we evaluate on the full top-k list
                group_bool_rec_rows.append(bool_rec_matrix[i])

            if not group_pos_len_list:
                print(f"Warning: No items found in test set for group '{group_name}'. Skipping.")
                continue

            # b. Calculate metrics for the filtered group
            group_pos_len_array = np.array(group_pos_len_list, dtype=np.int64)
            group_bool_rec_matrix = np.asarray(group_bool_rec_rows)
            
            group_result = self._calculate_metrics_for_group(group_pos_len_array, group_bool_rec_matrix)
            final_metric_dict[group_name] = group_result

        # Optional: Save recommendation result (code unchanged)
        if self.save_recom_result and is_test:
            self._save_topk_results(topk_index, eval_data, idx)

        return final_metric_dict

    def _calculate_metrics_for_group(self, pos_len, bool_rec_matrix):
        """A helper function to calculate metrics for a given group of data."""
        metric_dict = {}
        result_list = self._calculate_metrics(pos_len, bool_rec_matrix)
        
        for metric, value in zip(self.metrics, result_list):
            for k in self.topk:
                key = '{}@{}'.format(metric.capitalize(), k)
                metric_dict[key] = round(value[k - 1], 4)
        return metric_dict

    def _calculate_metrics(self, pos_len_list, topk_index):
        """Original metric calculation function (unchanged)."""
        result_list = []
        for metric in self.metrics:
            metric_fuc = metrics_dict[metric.lower()]
            result = metric_fuc(topk_index, pos_len_list)
            result_list.append(result)
        return np.stack(result_list, axis=0)
        
    def _save_topk_results(self, topk_index, eval_data, idx):
        """Saves the recommendation top-k results to a file."""
        dataset_name = self.config['dataset']
        model_name = self.config['model']
        max_k = max(self.topk)
        dir_name = os.path.abspath(self.config.get('recommend_topk', 'saved_topk'))
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        file_path = os.path.join(dir_name, '{}-{}-idx{}-top{}-{}.csv'.format(
            model_name, dataset_name, idx, max_k, get_local_time()))
        
        df_data = {'user_id': eval_data.get_eval_users()}
        for i in range(max_k):
            df_data[f'top_{i+1}'] = topk_index[:, i]
        
        x_df = pd.DataFrame(df_data)
        x_df.to_csv(file_path, sep='\t', index=False)
        print(f"Top-K results saved to {file_path}")

    def _check_args(self):
        """Original argument checking (unchanged)."""
        if isinstance(self.metrics, (str, list)):
            if isinstance(self.metrics, str):
                self.metrics = [self.metrics]
        else:
            raise TypeError('metrics must be str or list')
        for m in self.metrics:
            if m.lower() not in topk_metrics:
                raise ValueError(f"There is no topk metric named {m}!")
        self.metrics = [metric.lower() for metric in self.metrics]

        if isinstance(self.topk, (int, list)):
            if isinstance(self.topk, int):
                self.topk = [self.topk]
            for k in self.topk:
                if k <= 0:
                    raise ValueError(f'topk must be a positive integer, but get `{k}`')
        else:
            raise TypeError('topk must be an integer or a list')

    def __str__(self):
        return 'TopK Evaluator: Metrics=[{}], TopK=[{}]'.format(
            ', '.join(self.metrics), ', '.join(map(str, self.topk))
        )