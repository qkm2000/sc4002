from torch import nn
import torch

class CNNModel(nn.Module):
    """
    A Convolutional Neural Network (CNN) model to produce
    sentence representations and perform sentiment classification.

    Args:
        embedding_dim (int):
            Dimension of the embeddings.
        embedding_matrix (numpy.ndarray):
            Pre-trained embedding matrix.
        num_filters (int, optional):
            Number of convolutional filters. Default is 100.
        filter_sizes (list of int, optional):
            List of filter sizes for convolution layers. Default is [3, 4, 5].
        freeze_embeddings (bool, optional):
            Whether to freeze the embedding weights. Default is True.
    """

    def __init__(
        self,
        embedding_dim,
        embedding_matrix,
        num_filters=100,
        filter_sizes=[3, 4, 5],
        freeze_embeddings=True
    ):
        super(CNNModel, self).__init__()

        # Create the embedding layer using the pre-trained matrix
        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix, dtype=torch.float32),
            freeze=freeze_embeddings  # Freeze the embedding weights
        )

        # Create convolutional layers for each filter size
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (fs, embedding_dim)) for fs in filter_sizes
        ])

        # Fully connected layer to output sentiment score
        self.fc = nn.Linear(num_filters * len(filter_sizes), 1)
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Embedding lookup and reshape for convolution
        # Shape: (batch_size, max_seq_len, embedding_dim)
        x = self.embedding(x)
        # Add channel dimension for convolution,
        # shape: (batch_size, 1, max_seq_len, embedding_dim)
        x = x.unsqueeze(1)

        # Apply convolution + ReLU + max pooling for each filter size
        conv_outputs = [
            torch.relu(conv(x)).squeeze(3) for conv in self.convs
        ]
        pooled_outputs = [
            torch.max(output, dim=2)[0] for output in conv_outputs
        ]

        # Concatenate all pooled outputs
        out = torch.cat(pooled_outputs, dim=1)
        out = self.dropout(out)  # Apply dropout for regularization
        out = self.fc(out)       # Fully connected layer

        return self.sigmoid(out)  # Binary classification
