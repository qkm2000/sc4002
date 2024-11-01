import torch
from torch import nn


class RNNModel(nn.Module):
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
                "Invalid RNN type. Choose from 'rnn', 'lstm', or 'gru'.")

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
