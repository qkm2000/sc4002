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
        pooling_method="last_state",  # Method for sentence representation
    ):
        super(RNNModel, self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix, dtype=torch.float32),
            freeze=freeze_embeddings
        )
        self.rnn_type = rnn_type.lower()
        self.pooling_method = pooling_method

        # RNN layer
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
            raise ValueError("Invalid RNN type. Choose from 'rnn', 'lstm', 'gru'.")

        self.bidirectional = bidirectional
        direction_factor = 2 if bidirectional else 1

        # Fully connected layer
        self.fc = nn.Linear(hidden_size * direction_factor, 1)
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()

        # Attention layer if pooling_method is 'attention'
        if self.pooling_method == 'attention':
            self.attention_weights = nn.Linear(hidden_size * direction_factor, 1)

    def forward(self, x):
        x = self.embedding(x)
        rnn_output, _ = self.rnn(x)

        if self.pooling_method == 'last_state':
            sentence_representation = rnn_output[:, -1, :]

        elif self.pooling_method == 'mean_pool':
            # Apply mean pooling over time
            sentence_representation = torch.mean(rnn_output, dim=1)

        elif self.pooling_method == 'max_pool':
            # Apply max pooling over time
            sentence_representation, _ = torch.max(rnn_output, dim=1)

        elif self.pooling_method == 'mean_max':
            # Combine mean and max pooling
            max_pooled, _ = torch.max(rnn_output, dim=1)
            mean_pooled = torch.mean(rnn_output, dim=1)
            sentence_representation = (mean_pooled + max_pooled) / 2

        elif self.pooling_method == 'attention':
            # Compute attention scores and apply them
            attn_weights = torch.tanh(self.attention_weights(rnn_output)).squeeze(-1)
            attn_weights = torch.softmax(attn_weights, dim=1)
            sentence_representation = torch.sum(rnn_output * attn_weights.unsqueeze(-1), dim=1)

        else:
            raise ValueError("Invalid pooling method. Choose from 'last_state', 'max_pool', 'mean_pool', 'meanmax_pool', 'attention'.")

        sentence_representation = self.dropout(sentence_representation)
        out = self.fc(sentence_representation)
        return self.sigmoid(out)
