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
import torch.optim.lr_scheduler as lr_scheduler

from src.datasets import get_dataset
from src.models import all_models
from src.utils import parse_args, save_args, get_logger

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

def train(args):
    # Setup save_dir
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Initialize logger
    logger = get_logger(args.save_dir)

    # Select device
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    logger.info(f"Using Device: {device}")

    # Save args
    save_args(args, args.save_dir, logger)

    # Load dataset
    dataset = get_dataset(args)

    # Split dataset
    data_size = len(dataset)
    train_size = int(args.train_ratio * data_size)
    test_size = data_size - train_size

    train_set, test_set = random_split(dataset, [train_size, test_size])

    # Create dataloader
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    # Select model type
    if args.model_type not in all_models:
        raise ValueError(f"Model {args.model_type} not found.")

    model_class = all_models[args.model_type]

    # Initialize model param
    model_params = {
        'num_users': dataset.num_users,
        'num_items': dataset.num_items,
        'num_features': args.num_features,
        'feature_dims': getattr(dataset, 'feature_dims', None),
        'mlp_layers': args.mlp_layers,
        'dropout': args.dropout
    }

    # Initialize model architecture
    net = model_class(**model_params)
    net = net.to(device)

    # Define loss function
    loss_choices = {
        'mse': nn.MSELoss(),
        'l1':  nn.L1Loss()
    }
    criterion = loss_choices[args.loss_type]

    # Define optimizer
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)
    # Select lr scheduler
    if args.scheduler == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
    elif args.scheduler == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=args.lr_min)
    elif args.scheduler == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.lr_gamma, patience=args.lr_patience)
    else:
        scheduler = None

    # Initialize best_loss
    best_loss = float('inf')

    # Initialize loss_history
    train_loss_history = []
    test_loss_history = []

    # Start training
    logger.info(" Start Training ".center(60, "="))
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

        for inputs, labels in train_pbar:

            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)

            # Zero gradient
            optimizer.zero_grad()

            # Forward
            output = net(inputs)

            # Compute loss
            loss = criterion(output, labels)

            # Backward & Optimization
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_train_loss = total_train_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)

        # ============ Validation ============
        net.eval()
        total_test_loss = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = {k: v.to(device) for k, v in inputs.items()}
                labels = labels.to(device)

                output = net(inputs)
                loss = criterion(output, labels)

                total_test_loss += loss.item()

        avg_test_loss = total_test_loss / len(test_loader)
        test_loss_history.append(avg_test_loss)

        # ============ Lr scheduler step ============
        if scheduler is not None:
            last_lr = optimizer.param_groups[0]['lr']

            if args.scheduler == 'plateau':
                scheduler.step(avg_test_loss)
            else:
                scheduler.step()

            current_lr = optimizer.param_groups[0]['lr']

            if current_lr != last_lr:
                logger.info(f"Learning Rate changed: {last_lr} -> {current_lr}")

        # ============ Log & Save ============
        # Save best model
        is_best = avg_test_loss < best_loss
        if is_best:
            best_loss = avg_test_loss
            torch.save(net.state_dict(), os.path.join(args.save_dir, 'best_model.pth'))

        mark = " *" if is_best else ""
        logger.info(f"Epoch {epoch + 1}/{args.num_epochs} | "
                    f"Train Loss: {avg_train_loss:.4f} | "
                    f"Test Loss: {avg_test_loss:.4f}{mark}")

    # Save final model
    torch.save(net.state_dict(), os.path.join(args.save_dir, 'last_model.pth'))
    logger.info(f"Training Done! Best Test Loss: {best_loss:.4f}")

    # Draw & Save loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss_history, label='Training Loss', color='blue')
    plt.plot(test_loss_history, label='Validation Loss', color='orange')

    plt.title(f'Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plot_path = os.path.join(args.save_dir, 'loss_curve.png')
    plt.savefig(plot_path)
    logger.info(f"Loss curve saved to: {os.path.abspath(plot_path)}")

def main():
    args = parse_args()
    set_seed(args.seed)
    train(args)

if __name__ == '__main__':
    main()