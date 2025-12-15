import torch
from torch import nn

class FactorizationMachine(nn.Module):
    def __init__(self, num_users, num_items, feature_dims, num_features, **kwargs):
        super().__init__()