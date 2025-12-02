import os
import sys
from tqdm import tqdm
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from src.dataset import MovieLensDataset
from src import model
from src.utils import parse_args

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

def train(args):
    # Select device
    if args.device == 'cuda' and torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')

    print(f"Using Device: {args.device}")

    # Setup save_dir
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Load dataset
    dataset = MovieLensDataset(args.data_path)

    # Split dataset into train_set & test_set
    data_size = len(dataset)
    train_size = int(args.train_ratio * data_size)
    test_size = data_size - train_size

    train_set, test_set = random_split(dataset, [train_size, test_size])

    # Create dataloader
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    # Select model
    model_mapping = {
        'mf': model.MatrixFactorization,
        'gmf': model.GeneralizedMF,
        'ncf': model.NeuralCF,
    }
    if args.model_type not in model_mapping:
        raise ValueError(f"Model '{args.model_type}' not found. Choices: {list(model_mapping.keys())}")
    model_class = model_mapping[args.model_type]

    # Initialize net
    net = model_class(dataset.num_users, dataset.num_items, args.num_features)
    net = net.to(args.device)

    # Define loss func & optimizer
    loss_choices = {
        'mse': nn.MSELoss(),
        'l1':  nn.L1Loss()
    }
    criterion = loss_choices[args.loss_type]
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    # Initialize best_loss
    best_loss = float('inf')

    # Initialize loss_history
    train_loss_history = []
    test_loss_history = []

    # Start training loop
    for epoch in range(args.num_epochs):

        # ============ Train ============
        net.train()
        total_train_loss = 0

        train_pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{args.num_epochs} [Train]",
            leave=False,
            ncols=80,
            mininterval=1.0,
            file = sys.stdout
        )

        for user_ids, item_ids, ratings in train_pbar:
            user_ids = user_ids.to(args.device)
            item_ids = item_ids.to(args.device)
            ratings = ratings.to(args.device)

            # Zero gradient
            optimizer.zero_grad()

            # Forward
            output = net(user_ids, item_ids)

            # Compute loss
            loss = criterion(output, ratings)

            # Backward & Optimization
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_train_loss = total_train_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)

        # ============ Eval ============
        net.eval()
        total_test_loss = 0

        with torch.no_grad():
            for user_ids, item_ids, ratings in test_loader:
                user_ids = user_ids.to(args.device)
                item_ids = item_ids.to(args.device)
                ratings = ratings.to(args.device)

                output = net(user_ids, item_ids)
                loss = criterion(output, ratings)

                total_test_loss += loss.item()

        avg_test_loss = total_test_loss / len(test_loader)
        test_loss_history.append(avg_test_loss)

        # ============ Log & Save ============
        print(f"Epoch {epoch + 1}/{args.num_epochs} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Test Loss: {avg_test_loss:.4f}")

        print("-" * 60)

        # Save best model
        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            torch.save(net.state_dict(), os.path.join(args.save_dir, 'best_model.pth'))

    # Save final model
    torch.save(net.state_dict(), os.path.join(args.save_dir, 'last_model.pth'))
    print(f"Training Done! Best Test Loss: {best_loss:.4f}")

    # Save loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_history, label='Training Loss', color='blue')
    plt.plot(test_loss_history, label='Validation Loss', color='orange')

    plt.title(f'Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plot_path = os.path.join(args.save_dir, 'loss_curve.png')
    plt.savefig(plot_path)
    print(f"Loss curve saved to: {plot_path}")

def main():
    args = parse_args()
    set_seed(args.seed)
    train(args)

if __name__ == '__main__':
    main()