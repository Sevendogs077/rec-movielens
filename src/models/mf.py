import torch
import torch.nn as nn

class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, **kwargs):
        super().__init__()

        # The user and item embeddings must have the same dimension for the dot product interaction
        self.user_embedding = nn.Embedding(num_embeddings=num_users, embedding_dim=embedding_dim)
        self.item_embedding = nn.Embedding(num_embeddings=num_items, embedding_dim=embedding_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.user_embedding.weight, mean=0, std=0.01)
        nn.init.normal_(self.item_embedding.weight, mean=0, std=0.01)

    def forward(self, inputs):
        user_ids = inputs['user_id']
        item_ids = inputs['item_id']

        user_emb = self.user_embedding(user_ids)  # shape: [batch_size, embed_dim]
        item_emb = self.item_embedding(item_ids)  # shape: [batch_size, embed_dim]

        # Dot Product
        element_product = user_emb * item_emb
        output = torch.sum(element_product, dim=1)  # shape: [batch_size]
        return output