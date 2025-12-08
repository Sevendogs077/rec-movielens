import torch
import torch.nn as nn

class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, num_features, **kwargs):
        super().__init__()

        # The user and item embeddings must have the same dimension for the dot product interaction
        self.user_embedding = nn.Embedding(num_embeddings=num_users, embedding_dim=num_features)
        self.item_embedding = nn.Embedding(num_embeddings=num_items, embedding_dim=num_features)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.user_embedding.weight, mean=0, std=0.01)
        nn.init.normal_(self.item_embedding.weight, mean=0, std=0.01)

    def forward(self, user_ids, item_ids):
        u_emb = self.user_embedding(user_ids)
        i_emb = self.item_embedding(item_ids)

        element_product = u_emb * i_emb
        output = torch.sum(element_product, dim=1)
        return output