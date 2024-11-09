import torch
import torch.nn as nn

class CNNRNNHybridModel(nn.Module):
    def __init__(self, embed_dim, embedding_matrix, rnn_hidden_dim, num_classes, kernel_size=3, conv_out_channels=128, num_layers=1, freeze_embeddings=True, dropout_prob=0.5):
        super(CNNRNNHybridModel, self).__init__()
        
        # Embedding layer
        # Create the embedding layer using Word2Vec
        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix, dtype=torch.float32),
            freeze=freeze_embeddings  # Freeze the embedding weights
        )
        
        # CNN layer
        self.conv1d = nn.Conv1d(in_channels=embed_dim, out_channels=conv_out_channels, kernel_size=kernel_size)
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        self.dropout_cnn = nn.Dropout(dropout_prob)  # Dropout after CNN
        
        # RNN layer
        self.lstm = nn.LSTM(input_size=conv_out_channels, hidden_size=rnn_hidden_dim, batch_first=True, bidirectional=True, num_layers=num_layers)
        self.dropout_rnn = nn.Dropout(dropout_prob)  # Dropout after LSTM
        
        # Fully connected layer
        self.fc = nn.Linear(rnn_hidden_dim * 2, num_classes)  # rnn_hidden_dim * 2 for bidirectional LSTM
    
    def forward(self, x):
        # Embedding
        x = self.embedding(x)  # shape: (batch_size, seq_len, embed_dim)
        
        # CNN
        x = x.permute(0, 2, 1)  # change to (batch_size, embed_dim, seq_len) for Conv1d
        x = torch.relu(self.conv1d(x))  # apply convolution
        x = self.maxpool(x)  # apply max pooling
        x = x.permute(0, 2, 1)  # back to (batch_size, seq_len, conv_out_channels)
        x = self.dropout_cnn(x)  # Apply dropout after CNN
        
        # RNN
        lstm_out, _ = self.lstm(x)  # shape: (batch_size, seq_len, rnn_hidden_dim * 2)
        lstm_out = self.dropout_rnn(lstm_out)  # Apply dropout after LSTM   
        
        # Take the last hidden state for classification
        final_out = lstm_out[:, -1, :]  # shape: (batch_size, rnn_hidden_dim * 2)
        
        # Fully connected layer
        out = self.fc(final_out)  # shape: (batch_size, num_classes)
        return torch.sigmoid(out)  # Apply sigmoid for binary classification
