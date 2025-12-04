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
        u_emb = self.user_embedding(user_ids)
        i_emb = self.item_embedding(item_ids)

        element_product = u_emb * i_emb
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
        nn.init.constant_(self.predict_layer.weight, 1.0) # simulate MF at the beginning

    def forward(self, user_ids, item_ids):
        u_emb = self.user_embedding(user_ids)
        i_emb = self.item_embedding(item_ids)

        # Element-wise product
        element_product = u_emb * i_emb

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

        self.mlp_layers = nn.Sequential(nn.Linear(num_features*2, 32),
                                        nn.ReLU(),
                                        nn.Dropout(0.2),

                                        nn.Linear(32, 16),
                                        nn.ReLU(),
                                        nn.Dropout(0.1),

                                        nn.Linear(16, 8),
                                        nn.ReLU(),

                                        nn.Linear(8, 1),
                                        )

        nn.init.normal_(self.user_embedding.weight, mean=0, std=0.01)
        nn.init.normal_(self.item_embedding.weight, mean=0, std=0.01)

        # Kaiming Normal Initialization
        for m in self.mlp_layers:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    def forward(self, user_ids, item_ids):
        u_emb = self.user_embedding(user_ids)
        i_emb = self.item_embedding(item_ids)

        logits = self.mlp_layers(torch.cat([u_emb,i_emb], dim=1))

        output = logits.view(-1)
        return output
