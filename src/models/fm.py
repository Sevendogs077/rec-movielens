import torch
from torch import nn
from .base import BaseModel

class FactorizationMachine(BaseModel):

    REQUIRED_FEATURES = '__all__'

    def __init__(self, feature_dims, embedding_dim, **kwargs):
        super().__init__(feature_dims)

        # Features
        self.feature_names = list(feature_dims.keys())

        # Embedding
        self.num_embeddings = int(sum(feature_dims.values()))
        self.embedding_dim = int(embedding_dim+1) # Plus 1 for 1st rank weights
        self.embedding = nn.Embedding(num_embeddings=self.num_embeddings, embedding_dim=self.embedding_dim)

        # Global bias
        self.global_bias = nn.Parameter(torch.zeros(1))

        # Index offsets
        feature_sizes = [feature_dims[name] for name in self.feature_names]
        offsets = torch.tensor((0, *feature_sizes[:-1]), dtype=torch.long)
        self.register_buffer('offsets', torch.cumsum(offsets, dim=0))

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, mean=0, std=0.01)
        nn.init.constant_(self.global_bias, 0.0)

    def forward(self, inputs):
        feature_ids = [inputs[name] for name in self.feature_names]
        feature_ids = torch.stack(feature_ids, dim=1)
        feature_ids = feature_ids + self.offsets # [B, F]

        emb = self.embedding(feature_ids) # [B, F, D]

        emb_w = emb[:,:,0] # [B, F]
        emb_v = emb[:,:,1:] # [B, F, D]

        first_rank = torch.sum(emb_w, dim=1)
        second_rank = 0.5 * torch.sum((torch.sum(emb_v, dim=1))**2 - torch.sum(emb_v**2, dim=1), dim=1)

        logits = self.global_bias + first_rank + second_rank

        output = logits.view(-1)
        return output


