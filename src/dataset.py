import os
import pandas as pd
import logging
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

class MovieLensDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()

        abs_path = os.path.abspath(data_path)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"Data file not found.\nExpected path: {abs_path}\n")

        logger.info(f"Successfully loaded dataset: {abs_path}")

        names = ['user_id', 'item_id', 'rating', 'timestamp']
        df = pd.read_csv(abs_path, sep='::', header=None, engine='python', names=names)

        # Adjust IDs for PyTorch Embedding (0-based)
        df['user_id'] -= 1
        df['item_id'] -= 1

        self.users = torch.tensor(df['user_id'].values, dtype=torch.long)
        self.items = torch.tensor(df['item_id'].values, dtype=torch.long)
        self.ratings = torch.tensor(df['rating'].values, dtype=torch.float32)
        self.timestamps = torch.tensor(df['timestamp'].values, dtype=torch.long)

        self.num_users = int(df['user_id'].max()) + 1
        self.num_items = int(df['item_id'].max()) + 1

    def __len__(self):
        return self.ratings.shape[0]

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]
