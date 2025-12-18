import torch
import torch.nn as nn
from .base import BaseModel
from .layers import MLP

class NeuMF(BaseModel):

    REQUIRED_FEATURES = ['user_id', 'item_id']

    def __init__(self, feature_dims, embedding_dim, mlp_layers, dropout):
        super().__init__(feature_dims)

        self.feature_names = self.REQUIRED_FEATURES
        self.user_idx = self.feature_names.index('user_id')
        self.item_idx = self.feature_names.index('item_id')

        self.num_embeddings = int(sum(feature_dims.values()))
        self.embedding_dim = int(embedding_dim * 2) # NeuMF needs 2 independent embeddings
        self.embedding = nn.Embedding(num_embeddings=self.num_embeddings, embedding_dim=self.embedding_dim)

        # Offsets
        feature_sizes = list(feature_dims[name] for name in self.feature_names)
        offsets = torch.tensor(data=(0, *feature_sizes[:-1]), dtype=torch.long)
        self.register_buffer('offsets', torch.cumsum(offsets, dim=0))

        self.mlp = MLP(
                    input_dim=self.embedding_dim,
                    hidden_dims=mlp_layers,
                    dropout=dropout,
                    output_layer=False,
                )

        predict_input_dim = embedding_dim + mlp_layers[-1]
        self.predict_layer = nn.Linear(predict_input_dim, 1, bias=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, std=0.01)

        nn.init.constant_(self.predict_layer.weight, 1.0)

        # Kaiming Normal Initialization
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, inputs):
        """
        :param inputs: Dict[F * Tensor[B]]
        """

        feature_ids = [inputs[name] for name in self.feature_names] # List[F * Tensor[B]]
        feature_ids = torch.stack(feature_ids, dim=1) # Tensor[B, F]

        # offsets
        feature_ids = feature_ids + self.offsets # [B, F] + [F] = [B, F]

        # Split emb
        emb = self.embedding(feature_ids) # [B, F, 2 * D] (2: NeuMF needs 2 independent embeddings)
        user_emb = emb[:, self.user_idx, :] # [B, 2 * D]
        item_emb = emb[:, self.item_idx, :] # [B, 2 * D]

        # Divide into 2-tower: MLP & GMF
        user_mlp_emb = user_emb[:, :self.embedding_dim//2] # [B, D]
        user_gmf_emb = user_emb[:, self.embedding_dim//2:] # [B, D]

        item_mlp_emb = item_emb[:, :self.embedding_dim//2] # [B, D]
        item_gmf_emb = item_emb[:, self.embedding_dim//2:] # [B, D]

        # Left: MLP
        mlp_input = torch.cat([user_mlp_emb, item_mlp_emb], dim=1) # [B, 2D]
        mlp_output = self.mlp(mlp_input) # [B, mlp_layers[-1]]

        # Right: GMF
        gmf_output = user_gmf_emb * item_gmf_emb # [B, D]

        # Concatenation
        fusion_vector = torch.cat([gmf_output, mlp_output], dim=1)
        logits = self.predict_layer(fusion_vector)

        # Squeeze
        output = logits.view(-1)

        return output