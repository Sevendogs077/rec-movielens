import torch
import torch.nn as nn

class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, num_features):
        super().__init__()

        # The user and item embeddings must have the same dimension for the dot product interaction
        self.user_embedding = nn.Embedding(num_embeddings=num_users, embedding_dim=num_features)
        self.item_embedding = nn.Embedding(num_embeddings=num_items, embedding_dim=num_features)

        # Initialize weights
        nn.init.normal_(self.user_embedding.weight, mean=0, std=0.01)
        nn.init.normal_(self.item_embedding.weight, mean=0, std=0.01)

    def forward(self, user_ids, item_ids):
        user_embedding = self.user_embedding(user_ids)
        item_embedding = self.item_embedding(item_ids)

        element_product = user_embedding * item_embedding
        output = torch.sum(element_product, dim=1)
        return output

class GeneralizedMF(nn.Module):
    def __init__(self, num_users, num_items, num_features):
        super().__init__()
        self.user_embedding = nn.Embedding(num_embeddings=num_users, embedding_dim=num_features)
        self.item_embedding = nn.Embedding(num_embeddings=num_items, embedding_dim=num_features)

        self.predict_layer = nn.Linear(num_features, 1, bias=False)

        # Initialize weights
        nn.init.normal_(self.user_embedding.weight, mean=0, std=0.01)
        nn.init.normal_(self.item_embedding.weight, mean=0, std=0.01)
        nn.init.constant_(self.predict_layer.weight, 1.0) # simulate MF at the begining

    def forward(self, user_ids, item_ids):
        user_embedding = self.user_embedding(user_ids)
        item_embedding = self.item_embedding(item_ids)

        # Element-wise product
        element_product = user_embedding * item_embedding

        # Apply learnable weights to each feature dimension
        logits = self.predict_layer(element_product)

        # Squeeze to batch size
        output = logits.view(-1)
        return output

class NeuralCF(nn.Module):
    def __init__(self, num_users, num_items, num_features):
        super().__init__()
        self.user_embedding = nn.Embedding(num_embeddings=num_users, embedding_dim=num_features)
        self.item_embedding = nn.Embedding(num_embeddings=num_items, embedding_dim=num_features)
        nn.init.normal_(self.user_embedding.weight, mean=0, std=0.01)
        nn.init.normal_(self.item_embedding.weight, mean=0, std=0.01)

    def forward(self, user_ids, item_ids):
        user_embedding = self.user_embedding(user_ids)
        item_embedding = self.item_embedding(item_ids)

        output = torch.sum(user_embedding * item_embedding, dim=1)
        return output
