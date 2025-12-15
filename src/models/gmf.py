import torch.nn as nn

class GeneralizedMF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, **kwargs):
        super().__init__()

        self.user_embedding = nn.Embedding(num_embeddings=num_users, embedding_dim=embedding_dim)
        self.item_embedding = nn.Embedding(num_embeddings=num_items, embedding_dim=embedding_dim)

        self.predict_layer = nn.Linear(embedding_dim, 1, bias=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.user_embedding.weight, mean=0, std=0.01)
        nn.init.normal_(self.item_embedding.weight, mean=0, std=0.01)

        # simulate MF at the beginning
        nn.init.constant_(self.predict_layer.weight, 1.0)

    def forward(self, inputs):
        user_ids = inputs['user_id']
        item_ids = inputs['item_id']

        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)

        # Element-wise product
        element_product = user_emb * item_emb

        # Apply learnable weights to each feature dimension
        logits = self.predict_layer(element_product)

        # Squeeze to batch size
        output = logits.view(-1)
        return output