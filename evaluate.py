import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

from src.datasets import get_dataset
from src.models import all_models
from src.utils import parse_args, load_args, get_logger

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def evaluate(input_args):
    # Configure device
    device = input_args.device
    print(f"Using device: {device}")

    # Confirm save_dir
    save_dir = input_args.save_dir
    if not os.path.exists(save_dir):
        raise FileNotFoundError(f"Experiment directory not found: {save_dir}")

    # Load train_args
    train_args = load_args(save_dir)

    # Load train seed
    set_seed(train_args.seed)

    # Load dataset
    dataset = get_dataset(train_args)

    # Recreate dataset
    data_size = len(dataset)
    train_size = int(train_args.train_ratio * data_size)
    test_size = data_size - train_size

    _, test_set = random_split(dataset, [train_size, test_size])

    # Create dataloader
    test_loader = DataLoader(test_set, batch_size=train_args.batch_size, shuffle=False)

    # Select model
    if train_args.model_type not in all_models:
        raise ValueError(f"Model {train_args.model_type} not found.")

    model_class = all_models[train_args.model_type]

    # Select model params
    # Select model params
    if hasattr(model_class, 'REQUIRED_FEATURES'):
        req_features = model_class.REQUIRED_FEATURES
    else:
        req_features = None

    if isinstance(req_features, list):
        selected_feature_dims = {
            k: v for k, v in dataset.feature_dims.items() if k in req_features
        }
    else:
        selected_feature_dims = dataset.feature_dims.copy()

    model_params = {
        'feature_dims': selected_feature_dims,
        'embedding_dim': train_args.embedding_dim,
        'mlp_layers': train_args.mlp_layers,
        'dropout': train_args.dropout
    }

    # Initialize model architecture
    net = model_class(**model_params)
    net = net.to(device)

    # Load trained model weights
    model_path = os.path.join(save_dir, 'best_model.pth')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Weights not found: {model_path}")

    net.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Loaded weights from: {os.path.abspath(model_path)}")

    # Start inference
    net.eval()
    all_preds = []
    all_ratings = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)

            output = net(inputs)

            all_preds.extend(output.cpu().numpy())
            all_ratings.extend(labels.cpu().numpy())

    y_pred = np.array(all_preds)
    y_true = np.array(all_ratings)

    # Clipping to [1, 5]
    y_pred = np.clip(y_pred, 1.0, 5.0)

    # Metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)

    print(" Evaluation Results ".center(60, "="))
    print(f"Model: {train_args.model_type.upper()}")
    print(f"Seed:  {train_args.seed} (Restored)")
    print(f"Test Set Size: {len(y_true)}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print("=" * 60)

def main():
    args = parse_args()
    evaluate(args)

if __name__ == '__main__':
    main()