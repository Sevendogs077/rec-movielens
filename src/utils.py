import argparse
import torch

def parse_args():
    parser = argparse.ArgumentParser(description='MovieLens Recommender System')

    # ============ data ============
    parser.add_argument('--data_path', type=str, default='./data/ratings.dat')
    parser.add_argument('--num_features', type=int, default=64)

    # ============ model ============
    parser.add_argument('--model_type', type=str, default='gmf', choices=['mf', 'gmf', 'ncf'])

    # ============ train ============
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--loss_type', type=str, default='mse',choices=['mse', 'l1'])
    parser.add_argument('--seed', type=int, default=77)

    # ============ save ============
    parser.add_argument('--save_dir', type=str, default='./log')

    # ============ device ============
    parser.add_argument('--device', type=str, default='cuda',choices=['cuda', 'cpu'])

    # ============ parse =============
    args = parser.parse_args()

    # String -> torch.device
    #args.device = torch.device('cuda') if args.device == 'cuda' and torch.cuda.is_available() else torch.device('cpu')

    return args



