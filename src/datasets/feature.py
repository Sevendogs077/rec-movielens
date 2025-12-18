import torch
from .base import BaseDataset

class FeatureDataset(BaseDataset):
    def __init__(self, data_path, max_history_len=50, **kwargs):
        super().__init__(data_path)

        self.feature_dims['history'] = self.num_items + 1 # register 'history'

        # ========== Collect labels (rating) ==========
        self.labels = torch.tensor(self.df['rating'].values, dtype=torch.float32)

        # ========== Collect features ==========
        self.fields = {}
        for col_name in self.feature_names:
            if col_name in self.df:
                self.fields[col_name] = torch.tensor(self.df[col_name].values, dtype=torch.long)

        # ========== Preparation for slice ==========
        self.max_history_len = max_history_len
        self.history_lens_list = self.df['history_len'].values
        self.user_ids_list = self.df['user_id'].values

    def __getitem__(self, idx):
        # ========== Get fields ==========
        x = {key: tensor[idx] for key, tensor in self.fields.items()}

        # ========== Get histories ==========
        # Get user_id
        user_id = self.user_ids_list[idx]

        # Get history len
        seq_len = self.history_lens_list[idx]

        # All history of this user
        full_history = self.user_history_dict.get(user_id, [])

        # slicing
        seq_history = full_history[:seq_len]

        # Truncate
        if len(seq_history) > self.max_history_len:
            seq_history = seq_history[-self.max_history_len:]

        # Padding
        pad_len = self.max_history_len - len(seq_history)
        if pad_len > 0:
            padding_token = self.padding_token
            seq_history = seq_history + [padding_token] * pad_len # After

        # Transfer to tensor
        x['history'] = torch.tensor(seq_history, dtype=torch.long)

        return x, self.labels[idx]


