import torch
from torch import nn


class RNNModel(nn.Module):
    """
    A Recurrent Neural Network (RNN) model with support
    for different RNN types (RNN, LSTM, GRU, CNN).
    Args:
        embedding_dim (int):
            Dimension of the embeddings.
        hidden_size (int):
            Number of features in the hidden state.
        embedding_matrix (numpy.ndarray):
            Pre-trained embedding matrix.
        rnn_type (str, optional):
            Type of RNN to use ('rnn', 'lstm', 'gru', 'cnn'). Default is 'rnn'.
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
        elif self.rnn_type == 'cnn':
                self.conv1 = nn.Conv1d(in_channels=embedding_dim, out_channels=hidden_size, kernel_size=3, padding=1)
                self.conv2 = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, padding=1)
                self.pool = nn.MaxPool1d(kernel_size=2)
                self.fc = nn.Linear(hidden_size, 1)
        else:
            raise ValueError(
                "Invalid RNN type. Choose from 'rnn', 'lstm', gru', or 'cnn'.")

        # Define the fully connected layer based on bidirectional setting
        if bidirectional:
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
        elif self.rnn_type == 'cnn':
            x = x.permute(0, 2, 1)  # Change to (batch_size, embedding_dim, seq_len) for Conv1d
            x = self.conv1(x)        # Apply first convolution layer
            x = torch.relu(x)
            x = self.conv2(x)        # Apply second convolution layer
            x = torch.relu(x)
            x = self.pool(x)         # Apply max pooling
            x = torch.max(x, 2)[0]   # Global max pooling over sequence length
            x = self.dropout(x)
            out = self.fc(x)

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
