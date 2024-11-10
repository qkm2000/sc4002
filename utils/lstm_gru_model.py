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
        dropout_prob=0.5
    ):

        super(LSTMGRUHybridModel, self).__init__()
        
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
        #self.fc = nn.Linear(2 * gru_hidden_dim, num_classes)
        self.fc1 = nn.Linear(2 * gru_hidden_dim, 128) 
        self.fc2 = nn.Linear(128, 64) 

        self.fc_out = nn.Linear(64, num_classes)
        #self.dropout = nn.Dropout(dropout_prob)
        self.sigmoid = nn.Sigmoid()  # Assuming binary classification
    
    def forward(self, x):
        x = self.embedding(x)
        
        # LSTM
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout_lstm(lstm_out)  # Apply dropout after LSTM  
        
        # GRU
        gru_out, _ = self.gru(lstm_out)
        gru_out = self.dropout_gru(gru_out)  # Apply dropout after GRU
        
        # Take the last hidden state for classification
        out = gru_out[:, -1, :]  # Taking the last time step
        #out = self.dropout(out)

        # Pass through the fully connected layers
        out = self.fc1(out)
        out = torch.relu(out)    # Activation after the first fully connected layer
        out = self.fc2(out)
        out = torch.relu(out)    # Activation after the second fully connected layer
    

        # Final output layer for prediction
        out = self.fc_out(out)
        
        return torch.sigmoid(out)
