import torch
import torch.nn as nn


class CNNRNNHybridModel(nn.Module):
    def __init__(
        self,
        embed_dim,
        embedding_matrix,
        rnn_hidden_dim,
        num_classes,
        kernel_size=3,
        conv_out_channels=128,
        cnn_layers=2,
        rnn_layers=2,
        fc_hidden_dim=256,
        freeze_embeddings=True,
        dropout_prob=0.5,
        pooling_mode="last_state"
    ):
        super(CNNRNNHybridModel, self).__init__()

        # Store pooling mode
        self.pooling_mode = pooling_mode

        # Embedding layer
        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix, dtype=torch.float32),
            freeze=freeze_embeddings
        )

        # CNN layers: stack multiple Conv1d and MaxPool1d layers
        cnn_modules = []
        in_channels = embed_dim
        for _ in range(cnn_layers):
            cnn_modules.append(nn.Conv1d(
                in_channels,
                conv_out_channels,
                kernel_size=kernel_size,
                padding=1
            ))
            cnn_modules.append(nn.ReLU())
            cnn_modules.append(nn.MaxPool1d(kernel_size=2))
            cnn_modules.append(nn.Dropout(dropout_prob))
            in_channels = conv_out_channels  # update in_channels for next layer
        self.cnn = nn.Sequential(*cnn_modules)

        # RNN layers: stack multiple LSTM and GRU layers
        rnn_modules = []
        rnn_input_size = conv_out_channels
        for i in range(rnn_layers):
            if i % 2 == 0:
                rnn_modules.append(nn.LSTM(
                    input_size=rnn_input_size,
                    hidden_size=rnn_hidden_dim,
                    num_layers=1,
                    bidirectional=True,
                    batch_first=True
                ))
            else:
                rnn_modules.append(nn.GRU(
                    input_size=rnn_hidden_dim * 2,
                    hidden_size=rnn_hidden_dim,
                    num_layers=1,
                    bidirectional=True,
                    batch_first=True
                ))
            rnn_modules.append(nn.Dropout(dropout_prob))
            # update input size for next layer (bidirectional)
            rnn_input_size = rnn_hidden_dim * 2

        self.rnn = nn.Sequential(*rnn_modules)

        # Fully connected layers
        rnn_output_dim = rnn_hidden_dim * 2  # bidirectional
        self.fc1 = nn.Linear(rnn_output_dim, fc_hidden_dim)
        self.fc2 = nn.Linear(fc_hidden_dim, fc_hidden_dim)
        self.fc3 = nn.Linear(fc_hidden_dim, num_classes)

        self.dropout_fc = nn.Dropout(dropout_prob)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Embedding
        x = self.embedding(x)    # shape: (batch_size, seq_len, embed_dim)

        # CNN
        x = x.permute(0, 2, 1)   # shape: (batch_size, embed_dim, seq_len)
        # shape: (batch_size, conv_out_channels, reduced_seq_len)
        x = self.cnn(x)
        # shape: (batch_size, reduced_seq_len, conv_out_channels)
        x = x.permute(0, 2, 1)

        # RNN
        for rnn_layer in self.rnn:
            if isinstance(rnn_layer, (nn.LSTM, nn.GRU)):
                x, _ = rnn_layer(x)  # Pass through LSTM or GRU layer

        # Pooling options
        if self.pooling_mode == "last_state":
            x = x[:, -1, :]  # Use the last hidden state of the sequence
        elif self.pooling_mode == "mean_max":
            mean_pool = torch.mean(x, dim=1)
            max_pool, _ = torch.max(x, dim=1)
            x = (mean_pool + max_pool) / 2

        # Fully connected layers with activation and dropout
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout_fc(x)

        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout_fc(x)

        x = self.fc3(x)
        return torch.sigmoid(x)  # Apply sigmoid for binary classification
