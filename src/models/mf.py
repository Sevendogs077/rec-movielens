import torch
import torch.nn as nn

from .base import BaseModel

class MatrixFactorization(BaseModel):

    REQUIRED_FEATURES = ['user_id', 'item_id']

    def __init__(self, feature_dims, embedding_dim, **kwargs):
        super().__init__(feature_dims)

        # Feature selection
        self.feature_names = self.REQUIRED_FEATURES

        # Initial embedding
        self.num_embeddings = int(sum(feature_dims.values()))
        self.embedding_dim = int(embedding_dim)

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)

        # Compute offsets
        # Shift indices to separate User IDs and Item IDs in the global table
        # e.g., User 0 -> Index 0; Item 0 -> Index N_users
        feature_sizes = [feature_dims[name] for name in self.feature_names]

        # [0, num_users]
        offsets = torch.tensor((0, *feature_sizes[:-1]), dtype=torch.long)

        # Not trained
        self.register_buffer('offsets', torch.cumsum(offsets, dim=0))

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, mean=0, std=0.01)

    def forward(self, inputs):

        # Stack inputs
        x = [inputs[name] for name in self.feature_names]
        x = torch.stack(x, dim=1)

        # Add offsets (Broadcast)
        # x: [B, 2] + offsets: [2] -> x: [B, 2]
        x = x + self.offsets

        # Embedding lookup
        # [B, 2] -> [B, 2, Dim]
        emb = self.embedding(x)

        # Split
        # [B, 2, Dim] -> [B, Dim]
        user_emb = emb[:, 0, :]
        item_emb = emb[:, 1, :]

        # Dot product
        output = torch.sum(user_emb * item_emb, dim=1)

        return output