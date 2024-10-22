import gensim.downloader as api
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from utils.utils import *


class SentimentAnalysis:
    def __init__(
        self,
        sentences_train,
        labels_train,
        sentences_val,
        labels_val,
        sentences_test,
        labels_test,
        version,
        embedding_dim=300,  # 300 for Google News Word2Vec
        batch_size=32,
        lr=0.001,
        rnn_type='LSTM',
        early_stopping_patience=3,
        model_save_path="modelfiles/",
        freeze_embeddings=True
    ):
        self.sentences_train = sentences_train
        self.labels_train = labels_train
        self.sentences_val = sentences_val
        self.labels_val = labels_val
        self.sentences_test = sentences_test
        self.labels_test = labels_test
        self.version = version
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.lr = lr
        self.rnn_type = rnn_type
        self.early_stopping_patience = early_stopping_patience
        self.model_save_path = model_save_path
        self.freeze_embeddings = freeze_embeddings

        # Prepare data
        print("Preparing data...")
        self.word2vec_model = self._load_word2vec()
        self.word_index = {
            word: i for i, word in enumerate(
                self.word2vec_model.index_to_key
            )
        }
        self.embedding_matrix = self.word2vec_model.vectors
        self.X_train, self.y_train = self._prepare_data(
            self.sentences_train, self.labels_train)
        self.X_val, self.y_val = self._prepare_data(
            self.sentences_val, self.labels_val)
        self.X_test, self.y_test = self._prepare_data(
            self.sentences_test, self.labels_test)

        # Build PyTorch Dataset and DataLoader
        self.train_loader = self._create_dataloader(self.X_train, self.y_train)
        self.val_loader = self._create_dataloader(self.X_val, self.y_val)
        self.test_loader = self._create_dataloader(self.X_test, self.y_test)

        print("Data preparation complete.")

        # Define the model
        self.model = self._build_model()

        # Loss function and optimizer
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # Early stopping variables
        self.best_accuracy = 0
        self.patience_counter = 0

    def _load_word2vec(self):
        """Load a pretrained Word2Vec model from Gensim."""
        return api.load('word2vec-google-news-300')

    def _prepare_data(self, sentences, labels):
        """Convert sentences to sequences and pad them."""
        X = [[
            self.word_index[word]
            for word in sentence if word in self.word_index]
            for sentence in sentences]
        # Convert to tensors and pad the sequences
        X = [torch.tensor(seq, dtype=torch.long) for seq in X]
        X_padded = pad_sequence(X, batch_first=True,
                                padding_value=0)  # Use 0 for padding
        y = torch.tensor(labels, dtype=torch.float32).unsqueeze(
            1)  # Reshape to match the output shape
        return X_padded, y

    def _create_dataloader(self, X, y):
        """Create DataLoader from datasets."""
        dataset = SentimentDataset(X, y)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def _build_model(self):
        """Create the RNN model using PyTorch."""
        return RNNModel(
            self.embedding_dim,
            self.embedding_matrix,
            rnn_type=self.rnn_type,
            freeze_embeddings=self.freeze_embeddings
        )

    def train(self, epochs=10):
        """Train the model with early stopping."""
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for X_batch, y_batch in self.train_loader:
                self.optimizer.zero_grad()
                output = self.model(X_batch)
                loss = self.criterion(output, y_batch)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch + 1:>3}/{epochs:>3},", end=" ")
            print(f"Loss: {total_loss / len(self.train_loader):.4f},", end=" ")

            accuracy = self.validate(dataset="val")
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                self.patience_counter = 0  # Reset patience counter
                # model is improving, save the model
                self.save_model()
            else:
                self.patience_counter += 1

            # Check for early stopping
            if self.patience_counter >= self.early_stopping_patience:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break

        # load the best model and evaluate on the test set
        self.load_model()
        accuracy = self.validate(dataset="test")

    def validate(self, dataset='val'):
        """Evaluate the model on the validation set."""
        self.model.eval()
        correct = 0
        total = 0
        if dataset == 'val':
            dataloader = self.val_loader
            mode = "Validation"
        elif dataset == 'test':
            dataloader = self.test_loader
            mode = "Test"
        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                output = self.model(X_batch)
                predicted = (output > 0.5).float()
                correct += (predicted == y_batch).sum().item()
                total += y_batch.size(0)
        accuracy = correct / total
        print(f"{mode} Accuracy: {accuracy:.4f}")
        return accuracy

    def save_model(self):
        torch.save(
            self.model.state_dict(),
            f"{self.model_save_path}sentiment_rnn_{self.rnn_type}_{self.version}.pth",
        )
        print("Model saved.")

    def load_model(self, version=None):
        if version:
            self.version = version
        state_dict = torch.load(
            f"{self.model_save_path}sentiment_rnn_{self.rnn_type}_{self.version}.pth",
            weights_only=True
        )
        self.model.load_state_dict(state_dict)
        print("Model loaded.")


class SentimentDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class RNNModel(nn.Module):
    def __init__(
        self,
        embedding_dim,
        embedding_matrix,
        rnn_type='LSTM',
        freeze_embeddings=True
    ):
        super(RNNModel, self).__init__()
        # Create the embedding layer using Word2Vec
        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix, dtype=torch.float32),
            freeze=freeze_embeddings  # Freeze the embedding weights
        )
        self.rnn_type = rnn_type

        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(
                embedding_dim,
                128,  # Hidden size for each direction
                batch_first=True,
                num_layers=1,
                bidirectional=True  # Bidirectional LSTM
            )
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(
                embedding_dim,
                128,  # Hidden size for each direction
                batch_first=True,
                num_layers=1,
                bidirectional=True  # Bidirectional GRU
            )

        # Adjust output size for bidirectional (128 * 2 = 256)
        self.fc = nn.Linear(128 * 2, 1)
        self.dropout = nn.Dropout(0.5)  # Dropout for regularization
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        if self.rnn_type == 'LSTM':
            # hn: (num_layers * 2, batch_size, hidden_size)
            _, (hn, _) = self.rnn(x)
        elif self.rnn_type == 'GRU':
            # hn: (num_layers * 2, batch_size, hidden_size)
            _, hn = self.rnn(x)

        # Combine forward and backward hidden states
        hn = self.dropout(
            hn[-2:].transpose(0, 1).contiguous().view(x.size(0), -1))
        out = self.fc(hn)
        return self.sigmoid(out)
