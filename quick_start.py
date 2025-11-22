# --- START OF NEW quick_start.py ---

from logging import getLogger
from itertools import product
from utils_package.dataset import RecDataset
from utils_package.dataloader import TrainDataLoader, EvalDataLoader
from utils_package.logger import init_logger
from utils_package.configurator import Config
from utils_package.utils import init_seed, get_model, get_trainer, dict2str
import platform
import os


def quick_start(model, dataset, config_dict, save_model=True):
    # merge config dict
    config = Config(model, dataset, config_dict)
    
    # Init logger
    init_logger(config)
    logger = getLogger()
    
    # Log config info
    logger.info('██Server: \t' + platform.node())
    logger.info('██Dir: \t' + os.getcwd() + '\n')
    logger.info(config)

    # Load data
    dataset = RecDataset(config)
    logger.info(str(dataset))

    # Split data
    train_dataset, valid_dataset, test_dataset = dataset.split()
    logger.info('\n====Training====\n' + str(train_dataset))
    logger.info('\n====Validation====\n' + str(valid_dataset))
    logger.info('\n====Testing====\n' + str(test_dataset))

    # Wrap into dataloaders
    train_data = TrainDataLoader(config, train_dataset, batch_size=config['train_batch_size'], shuffle=True)
    valid_data = EvalDataLoader(config, valid_dataset, additional_dataset=train_dataset, batch_size=config['eval_batch_size'])
    test_data = EvalDataLoader(config, test_dataset, additional_dataset=train_dataset, batch_size=config['eval_batch_size'])

    ############ Dataset loaded, run model ############
    
    hyper_ret = []
    best_test_value = 0.0
    best_test_idx = 0

    logger.info('\n\n=================================\n\n')

    # --- MODIFICATION START: Prepare the metric key for comparison ---
    # Get the validation metric key from config and format it correctly
    # e.g., 'recall@10' -> 'Recall@10'
    val_metric_lower = config['valid_metric'].lower()
    if val_metric_lower:
        parts = val_metric_lower.split('@')
        formatted_val_metric = f"{parts[0].capitalize()}@{parts[1]}"
    else:
        # Fallback to a default metric if not specified
        formatted_val_metric = 'NDCG@20'
    logger.info(f"Using '{formatted_val_metric}' from the 'Overall' group to determine the best model.")
    # --- MODIFICATION END ---
    
    # Hyper-parameter loops
    hyper_ls = []
    if "seed" not in config['hyper_parameters']:
        config['hyper_parameters'] = ['seed'] + config['hyper_parameters']
    for i in config['hyper_parameters']:
        hyper_ls.append(config[i] or [None])
    
    combinators = list(product(*hyper_ls))
    total_loops = len(combinators)
    
    for idx, hyper_tuple in enumerate(combinators):
        # Set random seed for reproducibility
        for param_name, param_value in zip(config['hyper_parameters'], hyper_tuple):
            config[param_name] = param_value
        init_seed(config['seed'])

        logger.info('========={}/{}: Parameters:{}={}======='.format(
            idx + 1, total_loops, config['hyper_parameters'], hyper_tuple))

        # Set random state of dataloader
        train_data.pretrain_setup()
        
        # Model loading and initialization
        model = get_model(config['model'])(config, train_data).to(config['device'])
        logger.info(model)

        # Trainer loading and initialization
        trainer = get_trainer()(config, model)
        
        # Model training
        best_valid_score, best_valid_result, best_test_upon_valid = trainer.fit(
            train_data, valid_data=valid_data, test_data=test_data, saved=save_model
        )
        
        hyper_ret.append((hyper_tuple, best_valid_result, best_test_upon_valid))

        # --- MODIFICATION START: Safely access the nested dictionary for comparison ---
        # Save the best test result based on the validation metric of the 'Overall' group
        current_test_value = 0.0
        if best_test_upon_valid and 'Overall' in best_test_upon_valid and formatted_val_metric in best_test_upon_valid['Overall']:
            current_test_value = best_test_upon_valid['Overall'][formatted_val_metric]
        else:
            logger.warning(f"Metric '{formatted_val_metric}' not found in the 'Overall' group of test results. Cannot compare.")

        if current_test_value > best_test_value:
            best_test_value = current_test_value
            best_test_idx = idx
            logger.info(f"██ New best test score for '{formatted_val_metric}' found: {best_test_value:.4f}")
        # --- MODIFICATION END ---

        logger.info('best valid result: \n' + dict2str(best_valid_result))
        logger.info('test result: \n' + dict2str(best_test_upon_valid))
        logger.info('████Current BEST████:\nParameters: {}={},\n'
                    'Valid: {},\nTest: {}\n\n\n'.format(config['hyper_parameters'],
            hyper_ret[best_test_idx][0], dict2str(hyper_ret[best_test_idx][1]), dict2str(hyper_ret[best_test_idx][2])))

    # Log final summary
    logger.info('\n============All Over=====================')
    for i, (p, k, v) in enumerate(hyper_ret):
        logger.info('Loop {} -- Parameters: {}={},\n best valid: {},\n best test: {}'.format(
            i + 1, config['hyper_parameters'], p, dict2str(k), dict2str(v)
        ))

    logger.info('\n\n█████████████ BEST HYPER-PARAMETER SETTING ████████████████')
    if hyper_ret:
        logger.info('\tParameters: {}={},\nValid: {},\nTest: {}\n\n'.format(
            config['hyper_parameters'],
            hyper_ret[best_test_idx][0],
            dict2str(hyper_ret[best_test_idx][1]),
            dict2str(hyper_ret[best_test_idx][2])
        ))
    else:
        logger.warning("No hyper-parameter loops were run.")