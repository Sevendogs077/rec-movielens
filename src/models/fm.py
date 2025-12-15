import torch
from torch import nn

class FactorizationMachine(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()