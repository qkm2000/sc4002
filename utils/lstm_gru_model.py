import torch
import torch.nn as nn


class LSTMGRUHybridModel(nn.Module):
    def __init__(
        self,
        embed_dim,
        embedding_matrix,
        lstm_hidden_dim,
        gru_hidden_dim,
        num_classes,
        num_layers=1,
        freeze_embeddings=True,
        dropout_prob=0.5,
        pooling_method="last_state",
    ):
        super(LSTMGRUHybridModel, self).__init__()
        self.pooling_method = pooling_method

        # Embedding layer
        # Create the embedding layer using Word2Vec
        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix, dtype=torch.float32),
            freeze=freeze_embeddings  # Freeze the embedding weights
        )

        # LSTM layer
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=lstm_hidden_dim, batch_first=True, num_layers=num_layers, bidirectional=True)
        self.dropout_lstm = nn.Dropout(dropout_prob)  # Dropout after LSTM

        # GRU layer
        self.gru = nn.GRU(input_size=2 * lstm_hidden_dim, hidden_size=gru_hidden_dim, batch_first=True, num_layers=num_layers, bidirectional=True)
        self.dropout_gru = nn.Dropout(dropout_prob)  # Dropout after GRU

        # Fully connected layer
        self.fc1 = nn.Linear(2 * gru_hidden_dim, 128)
        self.fc2 = nn.Linear(128, 64)

        self.fc_out = nn.Linear(64, num_classes)
        self.sigmoid = nn.Sigmoid()  # Assuming binary classification

        # Attention layer if pooling_method is 'attention'
        if self.pooling_method == 'attention':
            self.attention_weights = nn.Linear(gru_hidden_dim * 2, 1)

    def forward(self, x):
        x = self.embedding(x)

        # LSTM
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout_lstm(lstm_out)  # Apply dropout after LSTM

        # GRU
        gru_out, _ = self.gru(lstm_out)
        gru_out = self.dropout_gru(gru_out)  # Apply dropout after GRU

        # Pooling method to obtain sentence representation
        if self.pooling_method == 'last_state':
            sentence_representation = gru_out[:, -1, :]

        elif self.pooling_method == 'mean_pool':
            # Apply mean pooling over time
            sentence_representation = torch.mean(gru_out, dim=1)

        elif self.pooling_method == 'max_pool':
            # Apply max pooling over time
            sentence_representation, _ = torch.max(gru_out, dim=1)

        elif self.pooling_method == 'mean_max':
            # Combine mean and max pooling
            max_pooled, _ = torch.max(gru_out, dim=1)
            mean_pooled = torch.mean(gru_out, dim=1)
            sentence_representation = (mean_pooled + max_pooled) / 2

        elif self.pooling_method == 'attention':
            # Attention-based pooling
            attn_weights = torch.tanh(self.attention_weights(gru_out)).squeeze(-1)  # (batch_size, seq_len)
            attn_weights = torch.softmax(attn_weights, dim=1)  # Softmax over time dimension
            sentence_representation = torch.sum(gru_out * attn_weights.unsqueeze(-1), dim=1)  # Weighted sum

        else:
            raise ValueError("Invalid pooling method. Choose from 'last_state', 'max_pool', 'mean_pool', 'mean_max', 'attention'.")

        # Pass through the fully connected layers
        out = self.fc1(sentence_representation)
        out = torch.relu(out)  # Activation after the first fully connected layer
        out = self.fc2(out)
        out = torch.relu(out)  # Activation after the second fully connected layer

        # Final output layer for prediction
        out = self.fc_out(out)

        return torch.sigmoid(out)
