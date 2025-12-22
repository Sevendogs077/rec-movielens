import torch
from torch import nn

from .base import BaseModel

class LogisticRegression(BaseModel):
    # Typically, LR uses Sigmoid for CTR prediction.
    # In this project, Scaled Sigmoid is used to regress the explicit rating (1-5) directly.
    REQUIRED_FEATURES = '__all__'
    def __init__(self, feature_dims, embedding_dim, **kwargs):
        super().__init__(feature_dims)

        # feature names
        self.feature_names = list(feature_dims.keys())

        # emb
        self.num_embeddings = int(sum(feature_dims.values()))
        self.embedding_dim = 1 # LR only needs 1st rank weights
        self.embedding = nn.Embedding(num_embeddings=self.num_embeddings, embedding_dim=self.embedding_dim)

        # global bias
        self.global_bias = nn.Parameter(torch.zeros(1))

        # offsets
        feature_sizes = [feature_dims[name] for name in self.feature_names]
        offsets = torch.tensor((0, *feature_sizes[:-1]), dtype=torch.long)
        self.register_buffer('offsets', torch.cumsum(offsets, dim=0))

        # init
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, mean=0, std=0.01)

    def forward(self, inputs):
        # inputs: Dict[Str: List[int]]
        feature_ids = [inputs[name] for name in self.feature_names] # [Tensor 1,...,Tensor F], each Tensor's dim = [B]

        embedded_features = []

        for feature_id, offset in zip(feature_ids, self.offsets):
            feature_offset = feature_id + offset
            emb = self.embedding(feature_offset) # [B, D]

            if emb.ndim == 3:
                emb = emb.mean(dim=1)  # pooling for sequential features

            embedded_features.append(emb.unsqueeze(1))  # [B, 1, D]

        emb = torch.cat(embedded_features, dim=1)

        logits = self.global_bias + torch.sum(emb, dim=1) # [B, 1]

        output = torch.sigmoid(logits) * 4.0 + 1.0

        output = output.view(-1)

        return output








