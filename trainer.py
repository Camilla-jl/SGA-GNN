import os
import itertools
import torch
import torch.optim as optim
from torch.nn.utils.clip_grad import clip_grad_norm_
import numpy as np
import matplotlib.pyplot as plt
import math
from time import time
from logging import getLogger
from tqdm import tqdm

from utils_package.utils import get_local_time, early_stopping, dict2str
from utils_package.topk_evaluator import TopKEvaluator

class AbstractTrainer(object):
    def __init__(self, config, model):
        self.config = config
        self.model = model
    def fit(self, train_data):
        raise NotImplementedError('Method [fit] should be implemented.')
    def evaluate(self, eval_data):
        raise NotImplementedError('Method [evaluate] should be implemented.')

class Trainer(AbstractTrainer):
    def __init__(self, config, model):
        super(Trainer, self).__init__(config, model)
        self.logger = getLogger()
        self.learner = config['learner']
        self.learning_rate = config['learning_rate']
        self.epochs = config['epochs']
        self.eval_step = min(config['eval_step'], self.epochs)
        self.stopping_step = config['stopping_step']
        self.clip_grad_norm = config['clip_grad_norm']
        self.valid_metric = config['valid_metric'].lower()
        self.valid_metric_bigger = config['valid_metric_bigger']
        self.test_batch_size = config['eval_batch_size']
        self.device = config['device']
        self.weight_decay = 0.0
        if config['weight_decay'] is not None:
            wd = config['weight_decay']
            self.weight_decay = eval(wd) if isinstance(wd, str) else wd
        self.req_training = config['req_training']
        self.start_epoch = 0
        self.cur_step = 0
        tmp_dd = {}
        for j, k in list(itertools.product(config['metrics'], config['topk'])):
            tmp_dd[f'{j.lower()}@{k}'] = 0.0
        self.best_valid_score = -1
        self.best_valid_result = tmp_dd
        self.best_test_upon_valid = tmp_dd
        self.train_loss_dict = dict()
        self.optimizer = self._build_optimizer()
        lr_scheduler_config = config['learning_rate_scheduler']
        if lr_scheduler_config:
            fac = lambda epoch: lr_scheduler_config[0] ** (epoch / lr_scheduler_config[1])
            self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=fac)
        else:
            self.lr_scheduler = None
        self.evaluator = TopKEvaluator(config, self.model.dataset)
        self.item_tensor = None
        self.tot_item_num = None

    def _build_optimizer(self):
        if self.learner.lower() == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'adagrad':
            optimizer = optim.Adagrad(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'rmsprop':
            optimizer = optim.RMSprop(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            self.logger.warning('Received unrecognized optimizer, set default Adam optimizer')
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer

    def _train_epoch(self, train_data, epoch_idx, loss_func=None):
        if not getattr(self.config, 'req_training', True):
            return 0.0, []
            
        self.model.train()
        loss_func = loss_func or self.model.calculate_loss
        total_loss = 0.0
        
        # Use tqdm for a nice progress bar
        for batch_idx, interaction_data in enumerate(tqdm(train_data, desc=f"Epoch {epoch_idx} Training")):
            self.optimizer.zero_grad()
            
            # =======================================================================
            # >>>>>>>>>>>>>>>>>>>>>>>> 核心修改：处理不同采样逻辑 <<<<<<<<<<<<<<<<<<<<<
            # =======================================================================
            final_interaction = None
            # Check if we are in hard negative sampling mode
            if isinstance(interaction_data, tuple) and len(interaction_data) == 3 and interaction_data[2] is None:
                user, pos_item = interaction_data[0].to(self.device), interaction_data[1].to(self.device)
                
                # Call the hard negative sampler from the model
                # Assuming it samples 1 negative item per positive item
                neg_item = self.model.hard_negative_sampler.sample(user, pos_item)
                
                final_interaction = (user, pos_item, neg_item)
            else:
                # This is the original random sampling case where data is a single tensor
                final_interaction = interaction_data
            # =======================================================================

            losses = loss_func(final_interaction)
            
            if isinstance(losses, tuple):
                loss = sum(losses)
            else:
                loss = losses
            '''
            if self._check_nan(loss):
                self.logger.info(f'Loss is nan at epoch: {epoch_idx}, batch index: {batch_idx}. Exiting.')
                return torch.tensor(float('nan')), []
            '''
            if self._check_nan(loss):
                self.logger.error(f'Loss is NaN at epoch {epoch_idx}, batch {batch_idx}. Stopping epoch.')
                # 返回一个 float('nan') 作为信号
                return float('nan'), []

            loss.backward()
            if self.clip_grad_norm:
                clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)
            self.optimizer.step()
            
            total_loss += loss.item()

        return total_loss / len(train_data), []

    def _valid_epoch(self, valid_data):
        valid_result = self.evaluate(valid_data)
        
        metric_key = self.valid_metric
        # The new evaluator may return grouped results. We use 'Overall' for early stopping.
        if isinstance(valid_result, dict) and 'Overall' in valid_result:
             valid_result_overall = valid_result['Overall']
        else:
             valid_result_overall = valid_result
        
        valid_score = valid_result_overall.get(metric_key, list(valid_result_overall.values())[0])
        return valid_score, valid_result

    def _check_nan(self, loss):
        return torch.isnan(loss) or torch.isinf(loss)

    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, loss):
        return f"epoch {epoch_idx} training [time: {e_time - s_time:.2f}s, train loss: {loss:.4f}]"

    def fit(self, train_data, valid_data=None, test_data=None, saved=False, verbose=True):
        for epoch_idx in range(self.start_epoch, self.epochs):
            training_start_time = time()
            if hasattr(self.model, 'pre_epoch_processing'):
                self.model.pre_epoch_processing()
            
            train_loss, _ = self._train_epoch(train_data, epoch_idx)
            
            # 使用 math.isnan 来检查 float 类型的 train_loss
            if math.isnan(train_loss):
                self.logger.error("Training stopped due to NaN loss.")
                break

            if self.lr_scheduler:
                self.lr_scheduler.step()

            self.train_loss_dict[epoch_idx] = train_loss
            training_end_time = time()
            train_loss_output = self._generate_train_loss_output(epoch_idx, training_start_time, training_end_time, train_loss)
            
            if verbose:
                self.logger.info(train_loss_output)

            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                valid_score, valid_result = self._valid_epoch(valid_data)
                
                self.best_valid_score, self.cur_step, stop_flag, update_flag = early_stopping(
                    valid_score, self.best_valid_score, self.cur_step,
                    max_step=self.stopping_step, bigger=self.valid_metric_bigger)
                
                valid_end_time = time()
                
                if verbose:
                    valid_score_output = f"epoch {epoch_idx} evaluating [time: {valid_end_time - valid_start_time:.2f}s, valid_score: {valid_score:.4f}]"
                    self.logger.info(valid_score_output)
                    self.logger.info('valid result: \n' + dict2str(valid_result))
                
                if update_flag:
                    update_output = f"██ {self.config['model']}--Best validation results updated!!!"
                    if verbose:
                        self.logger.info(update_output)
                    self.best_valid_result = valid_result
                    
                    if test_data:
                        # Rerun evaluation on test data when best validation is found
                        _, self.best_test_upon_valid = self._valid_epoch(test_data)
                        self.logger.info('test result: \n' + dict2str(self.best_test_upon_valid))

                if stop_flag:
                    stop_output = f'+++++Finished training, best eval result in epoch {epoch_idx - self.cur_step * self.eval_step}'
                    if verbose:
                        self.logger.info(stop_output)
                    break


        # ======================= DEBUGGING CODE =======================
        print("="*30)
        print(f"DEBUG: Checking model type inside Trainer.fit()")
        print(f"Type of self.model is: {type(self.model)}")
        print(f"Does it have the method? {hasattr(self.model, 'get_final_item_embeddings')}")
        print("="*30)
        # =============================================================
        # ==============================================================================
        # >>>>>>>>>>>>>>>>>>>>>>>> MODIFICATION FOR VISUALIZATION <<<<<<<<<<<<<<<<<<<<<
        # ==============================================================================
        
        # After training is complete, save the final item embeddings for visualization
        # Check if the model has the desired method to prevent errors with other models
        if hasattr(self.model, 'get_final_item_embeddings'):
            self.logger.info("="*20 + " SAVING EMBEDDINGS FOR VISUALIZATION " + "="*20)

            # 1. Get final item embeddings from the trained model
            final_item_embeds = self.model.get_final_item_embeddings()

            # 2. Define path and filename using info from the config
            model_name = self.config['model']
            dataset_name = self.config['dataset']
            #output_dir = './saved_embeddings/'
            DRIVE_PROJECT_PATH = '/content/drive/MyDrive/MoToRec_Project/' # MyDrive是你的主云盘
            output_dir = os.path.join(DRIVE_PROJECT_PATH, 'saved_embeddings') # 新的保存路径

            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{model_name}_{dataset_name}_item_embeddings.npy")

            # 3. Save the embeddings file
            np.save(output_path, final_item_embeds)
            
            self.logger.info(f"Final item embeddings shape: {final_item_embeds.shape}")
            self.logger.info(f"Item embeddings successfully saved to: {output_path}")

        # ==============================================================================
        # >>>>>>>>>>>>>>>>>>>>>>>> MODIFICATION END <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # ==============================================================================
        
                    
        return self.best_valid_score, self.best_valid_result, self.best_test_upon_valid

    @torch.no_grad()
    def evaluate(self, eval_data, is_test=False, idx=0):
        self.model.eval()
        batch_matrix_list = []
        for batch_idx, batched_data in enumerate(eval_data):
            scores = self.model.full_sort_predict(batched_data)
            masked_items = batched_data[1]
            scores[masked_items[0], masked_items[1]] = -1e10
            _, topk_index = torch.topk(scores, max(self.config['topk']), dim=-1)
            batch_matrix_list.append(topk_index)
        return self.evaluator.evaluate(batch_matrix_list, eval_data, is_test=is_test, idx=idx)

    def plot_train_loss(self, show=True, save_path=None):
        epochs = sorted(self.train_loss_dict.keys())
        values = [float(self.train_loss_dict[epoch]) for epoch in epochs]
        plt.plot(epochs, values)
        plt.xticks(epochs)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        if show:
            plt.show()
        if save_path:
            plt.savefig(save_path)