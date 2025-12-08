import torch
import torch.nn as nn
from .layers import MLP

class NeuralCF(nn.Module):
    def __init__(self, num_users, num_items, num_features, mlp_layers=None, dropout=0.2, **kwargs):
        super().__init__()
        self.user_embedding = nn.Embedding(num_embeddings=num_users, embedding_dim=num_features)
        self.item_embedding = nn.Embedding(num_embeddings=num_items, embedding_dim=num_features)

        self.mlp = MLP(
            input_dim=num_features * 2,
            hidden_dims=mlp_layers,
            dropout=dropout
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.user_embedding.weight, mean=0, std=0.01)
        nn.init.normal_(self.item_embedding.weight, mean=0, std=0.01)

        # Kaiming Normal Initialization
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, user_ids, item_ids):
        u_emb = self.user_embedding(user_ids)
        i_emb = self.item_embedding(item_ids)

        logits = self.mlp(torch.cat([u_emb,i_emb], dim=1))

        output = logits.view(-1)
        return output