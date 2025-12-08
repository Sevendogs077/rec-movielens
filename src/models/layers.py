import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=None, dropout=0.2, output_layer=True):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [64, 32]

        layers = []
        current_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_dim = h_dim

        if output_layer:
            layers.append(nn.Linear(current_dim, 1))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)