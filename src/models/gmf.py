import torch
import torch.nn as nn
from .base import BaseModel

class GeneralizedMF(BaseModel):

    REQUIRED_FEATURES = ['user_id', 'item_id']

    def __init__(self, feature_dims, embedding_dim, **kwargs):
        super().__init__(feature_dims)

        # Features
        self.feature_names = self.REQUIRED_FEATURES
        self.user_idx = self.feature_names.index('user_id')
        self.item_idx = self.feature_names.index('item_id')

        # Embedding
        self.num_embeddings = int(sum(feature_dims.values()))
        self.embedding_dim = int(embedding_dim)
        self.embedding = nn.Embedding(num_embeddings=self.num_embeddings, embedding_dim=self.embedding_dim)

        # Offsets
        feature_sizes = [feature_dims[name] for name in self.feature_names]
        offsets = torch.tensor((0, *feature_sizes[:-1]), dtype=torch.long)
        self.register_buffer('offsets', torch.cumsum(offsets, dim=0))

        # Predict layer
        self.predict_layer = nn.Linear(embedding_dim, 1, bias=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, mean=0, std=0.01)
        nn.init.constant_(self.predict_layer.weight, 1.0) # simulate MF at the beginning

    def forward(self, inputs):
        # Stack inputs
        feature_ids = [inputs[name] for name in self.feature_names]
        feature_ids = torch.stack(feature_ids, dim=1)

        feature_ids = feature_ids + self.offsets

        emb = self.embedding(feature_ids)

        user_emb = emb[:, self.user_idx, :]
        item_emb = emb[:, self.item_idx, :]

        # Element-wise product
        element_product = user_emb * item_emb

        # Apply learnable weights to each feature dimension
        logits = self.predict_layer(element_product)

        # Squeeze to batch size
        output = logits.view(-1)
        return output