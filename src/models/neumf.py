import torch
import torch.nn as nn
from .layers import MLP

class NeuMF(nn.Module):
    def __init__(self, num_users, num_items, num_features, mlp_layers=None, dropout=0.2, **kwargs):
        super().__init__()
        self.user_embedding_mlp = nn.Embedding(num_embeddings=num_users, embedding_dim=num_features)
        self.item_embedding_mlp = nn.Embedding(num_embeddings=num_items, embedding_dim=num_features)
        self.user_embedding_gmf = nn.Embedding(num_embeddings=num_users, embedding_dim=num_features)
        self.item_embedding_gmf = nn.Embedding(num_embeddings=num_items, embedding_dim=num_features)

        predict_input_dim = num_features + mlp_layers[-1]
        self.predict_layer = nn.Linear(predict_input_dim, 1, bias=False)

        self.mlp = MLP(
            input_dim=num_features * 2,
            hidden_dims=mlp_layers,
            dropout=dropout,
            output_layer=False,
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.user_embedding_mlp.weight, mean=0, std=0.01)
        nn.init.normal_(self.item_embedding_mlp.weight, mean=0, std=0.01)
        nn.init.normal_(self.user_embedding_gmf.weight, mean=0, std=0.01)
        nn.init.normal_(self.item_embedding_gmf.weight, mean=0, std=0.01)

        nn.init.constant_(self.predict_layer.weight, 1.0)

        # Kaiming Normal Initialization
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, inputs):
        user_ids = inputs['user_id']
        item_ids = inputs['item_id']

        u_emb_mlp = self.user_embedding_mlp(user_ids)
        i_emb_mlp = self.item_embedding_mlp(item_ids)
        u_emb_gmf = self.user_embedding_gmf(user_ids)
        i_emb_gmf = self.item_embedding_gmf(item_ids)

        # Left: GMF
        gmf_output = u_emb_gmf * i_emb_gmf

        # Right: MLP
        mlp_input = torch.cat([u_emb_mlp, i_emb_mlp], dim=1)
        mlp_output = self.mlp(mlp_input)

        # Concatenation
        fusion_vector = torch.cat([gmf_output, mlp_output], dim=1)
        logits = self.predict_layer(fusion_vector)

        # Squeeze
        output = logits.view(-1)
        return output