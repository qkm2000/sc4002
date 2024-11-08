import torch
from torch import nn


class RNNModel(nn.Module):
    """
    A Recurrent Neural Network (RNN) model with support
    for different RNN types (RNN, LSTM, GRU).
    Args:
        embedding_dim (int):
            Dimension of the embeddings.
        hidden_size (int):
            Number of features in the hidden state.
        embedding_matrix (numpy.ndarray):
            Pre-trained embedding matrix.
        rnn_type (str, optional):
            Type of RNN to use ('rnn', 'lstm', 'gru'). Default is 'rnn'.
        freeze_embeddings (bool, optional):
            Whether to freeze the embedding weights. Default is True.
        bidirectional (bool, optional):
            If True, becomes a bidirectional RNN. Default is True.
        num_layers (int, optional):
            Number of recurrent layers. Default is 1.
    """
    def __init__(
        self,
        embedding_dim,
        hidden_size,
        embedding_matrix,
        rnn_type='rnn',
        freeze_embeddings=True,
        bidirectional=True,
        num_layers=1,
    ):
        super(RNNModel, self).__init__()

        # Create the embedding layer using Word2Vec
        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix, dtype=torch.float32),
            freeze=freeze_embeddings  # Freeze the embedding weights
        )
        self.rnn_type = rnn_type.lower()

        # Initialize the chosen RNN type
        if self.rnn_type == 'rnn':
            self.rnn = nn.RNN(
                input_size=embedding_dim,
                hidden_size=hidden_size,
                batch_first=True,
                num_layers=num_layers,
                bidirectional=bidirectional
            )
        elif self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size=embedding_dim,
                hidden_size=hidden_size,
                batch_first=True,
                num_layers=num_layers,
                bidirectional=bidirectional
            )
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(
                input_size=embedding_dim,
                hidden_size=hidden_size,
                batch_first=True,
                num_layers=num_layers,
                bidirectional=bidirectional
            )
        else:
            raise ValueError(
                "Invalid RNN type. Choose from 'rnn', 'lstm', gru'.")

        # Define the fully connected layer based on bidirectional setting
        if bidirectional and self.rnn_type in ['rnn', 'lstm', 'gru']:
            self.fc = nn.Linear(hidden_size * 2, 1)
        else:
            self.fc = nn.Linear(hidden_size, 1)

        self.dropout = nn.Dropout(0.5)  # Dropout for regularization
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)

        # Forward pass through RNN
        if self.rnn_type == 'rnn':
            _, hn = self.rnn(x)
        elif self.rnn_type == 'lstm':
            _, (hn, _) = self.rnn(x)
        elif self.rnn_type == 'gru':
            _, hn = self.rnn(x)

        # Combine forward and backward hidden states if bidirectional
        if self.rnn.bidirectional:
            hn = self.dropout(
                hn[-2:].transpose(0, 1).contiguous().view(x.size(0), -1)
            )
        else:
            hn = self.dropout(hn[-1].view(x.size(0), -1))

        # Final output layer
        out = self.fc(hn)
        return self.sigmoid(out)

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
    
    def __init__(self, embedding_dim, embedding_matrix, num_filters=100, filter_sizes=[3, 4, 5], freeze_embeddings=True):
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
        x = self.embedding(x)  # Shape: (batch_size, max_seq_len, embedding_dim)
        x = x.unsqueeze(1)  # Add channel dimension for convolution, shape: (batch_size, 1, max_seq_len, embedding_dim)

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

        return self.sigmoid(out) # Binary classification