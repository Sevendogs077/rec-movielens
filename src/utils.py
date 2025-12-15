import os
import sys
import json
import logging
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='MovieLens Recommender System')

    # ============ Environment ============
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--seed', type=int, default=77)
    parser.add_argument('--save_dir', type=str, default='./output')

    # ============ Data ============
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--max_history_len', type=int, default=50,
                        help='max length of user history sequence')

    # ============ Model ============
    parser.add_argument('--model_type', type=str, default='gmf',
                        choices=['mf', 'gmf', 'neumf', 'fm', 'deepfm'],
                        help='type of model')

    parser.add_argument('--embedding_dim', type=int, default=64,
                        help='Dimension of embedding vectors')

    # mlp
    parser.add_argument('--mlp_layers', type=int, nargs='+', default=[64, 32, 16])
    parser.add_argument('--dropout', type=float, default=0.2)

    # ============ Train ============
    parser.add_argument('--num_epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--loss_type', type=str, default='mse', choices=['mse', 'l1'])

    # ============ Optim & Scheduler ============
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-4)

    # scheduler
    parser.add_argument('--scheduler', type=str, default='plateau',
                        choices=['none', 'step', 'cosine', 'plateau'],
                        help='LR scheduler type')

    parser.add_argument('--lr_step', type=int, default=5,
                        help='StepLR decay step')

    parser.add_argument('--lr_gamma', type=float, default=0.5,
                        help='LR decay factor')

    parser.add_argument('--lr_min', type=float, default=1e-6,
                        help='Minimum learning rate')

    parser.add_argument('--lr_patience', type=int, default=5,
                        help='Patience for Plateau')

    args = parser.parse_args()
    return args

def save_args(args, save_dir, logger):
    param_path = os.path.join(save_dir, 'config.json')
    with open(param_path, 'w') as f:
        json.dump(vars(args), f, indent=4)
    logger.info(f"Config saved to {os.path.abspath(param_path)}")

def load_args(save_dir):
    param_path = os.path.join(save_dir, 'config.json')
    with open(param_path, 'r') as f:
        args_dict = json.load(f)
    return argparse.Namespace(**args_dict)

def get_logger(save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    log_file = os.path.join(save_dir, 'train.log')

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # File: detailed info
    file_fmt = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    # Console: message only
    console_fmt = logging.Formatter('%(message)s')

    if not logger.handlers:
        # Handler 1
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_fmt)
        logger.addHandler(console_handler)

        # Handler 2
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setFormatter(file_fmt)
        logger.addHandler(file_handler)

    return logger