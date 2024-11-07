from torch.utils.data import dataloader, Dataset
from torch.nn.utils.rnn import pad_sequence
from torch import nn
import torch
from gensim.utils import simple_preprocess
from gensim.models import KeyedVectors
import gensim.downloader as api
import gensim
import nltk
import os
import matplotlib.pyplot as plt
import numpy as np

nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')


class SentimentDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_directory(directory_path):
    """
    Create a directory if it does not exist.
    Args:
        directory_path (str):
            The path to the directory to be created.
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


def load_word2vec(
    vocab=None,
    filepath=None,
    word2vec_api=None,
):
    """
    Load a Word2Vec model from a file or a pre-trained model from the
    gensim API.

    Args:
        vocab (list of str, optional):
            List of words in the vocabulary.
            This is only required when loading a model from a .npy file.
        filepath (str, optional):
            The path to the Word2Vec model file. Defaults to None.
        word2vec_api (str, optional):
            The name of the pre-trained model to load from the gensim API.
            Defaults to None.

    Returns:
        gensim.models.KeyedVectors:
            The loaded Word2Vec model.

    Notes:
    - If both `filepath` and `word2vec_api` are provided,
      `word2vec_api` will take precedence.
    - If neither `filepath` nor `word2vec_api` are provided, the function
      will load the "word2vec-google-news-300" model from the
      gensim API.
    """
    if word2vec_api is not None:
        return api.load(word2vec_api)
    if filepath is not None:
        embedding_matrix = np.load(filepath)
        if vocab is None:
            raise ValueError("Vocabulary must be provided when loading .npy")
        vector_size = embedding_matrix.shape[1]
        word2vec_model = KeyedVectors(vector_size=vector_size)
        word2vec_model.add_vectors(vocab, embedding_matrix)
        return word2vec_model
    return api.load("word2vec-google-news-300")


def prepare_data(
    sentences,
    labels,
    word_index,
    max_seq_len=15
):
    """
    Prepares data for RNN input by tokenizing, padding,
    and converting to tensors.

    Args:
        sentences (list of str):
            List of sentences to be processed.
        labels (list of int/float):
            List of labels corresponding to the sentences.
        word_index (dict):
            Dictionary mapping words to their respective indices.
        max_seq_len (int, optional):
            Maximum sequence length for padding/truncating. Defaults to 15.
    Returns:
        tuple: A tuple containing:
            - X_padded (torch.Tensor):
                Tensor of tokenized and padded sentences.
            - y (torch.Tensor):
                Tensor of labels reshaped to match the output shape.
    """
    X = [
        [word_index[word]
            for word in simple_preprocess(sentence) if word in word_index]
        for sentence in sentences
    ]

    # Clip sequences that are longer than max_seq_len
    X = [seq[:max_seq_len] if len(seq) > max_seq_len else seq for seq in X]

    # Convert to tensors and pad the sequences
    X = [torch.tensor(seq, dtype=torch.long) for seq in X]
    X_padded = pad_sequence(
        X,
        batch_first=True,
        padding_value=0  # Use 0 for padding
    )

    # Further clip or pad to ensure all are exactly max_seq_len
    X_padded = X_padded[:, :max_seq_len] if \
        X_padded.size(1) > max_seq_len else \
        torch.nn.functional.pad(
            X_padded, (0, max_seq_len - X_padded.size(1)), value=0)

    y = torch.tensor(labels, dtype=torch.float32).unsqueeze(
        1)  # Reshape to match the output shape
    return X_padded, y


def create_dataloader(X, y, batch_size, shuffle=True):
    """
    Create a PyTorch DataLoader for a dataset.
    Args:
        X (np.ndarray):
            The input data.
        y (np.ndarray):
            The target data.
        batch_size (int):
            The batch size for the DataLoader.
    Returns:
        torch.utils.data.DataLoader:
            The DataLoader for the dataset.
    """
    dataset = SentimentDataset(X, y)
    return dataloader.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )


def save_model(model, model_save_path):
    """
    Save the state dictionary of a PyTorch model to a specified file path.
    Args:
        model (torch.nn.Module):
            The PyTorch model to be saved.
        model_save_path (str):
            The file path where the model's state dictionary will be saved.
    """

    # Check if the file exists and remove it
    if os.path.exists(model_save_path):
        os.remove(model_save_path)

    torch.save(model.state_dict(), model_save_path)
    print("Model saved.")


def load_model(model, model_save_path):
    """
    Load the state dictionary of a model from a specified file path.
    This loading is inplace, meaning that the model will have to be
    instantiated before calling this function.
    Args:
        model (torch.nn.Module):
            The model instance to load the state dictionary into.
        model_save_path (str):
            The file path to the saved state dictionary.
    """
    state_dict = torch.load(model_save_path, weights_only=True)
    model.load_state_dict(state_dict)
    print("Model loaded.")


def validate(model, val_dataloader):
    """
    Evaluate the performance of a model on a validation dataset.
    Args:
        model (torch.nn.Module):
            The neural network model to be evaluated.
        val_dataloader (torch.utils.data.DataLoader):
            DataLoader for the validation dataset.
    Returns:
        float:
            The accuracy of the model on the validation dataset,
            calculated as the ratio of correctly predicted samples
            to the total number of samples.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in val_dataloader:
            inputs, labels = data
            outputs = model(inputs)
            predicted = torch.round(outputs)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    print(f"Accuracy: {accuracy:.4f}")
    return accuracy


def last_hidden_state(outputs):
    """Extract the last hidden state for classification."""
    return outputs[:, -1]


def mean_pooling(outputs):
    """Apply mean pooling on the RNN outputs for classification."""
    return torch.mean(outputs, dim=1)


def max_pooling(outputs):
    """Apply max pooling on the RNN outputs for classification."""
    return torch.max(outputs, dim=1).values


def train(
    model,
    trn_dataloader,
    val_dataloader,
    optimizer,
    version,
    model_save_path,
    model_type,
    epochs=10,
    criterion=nn.BCELoss(),
    early_stopping_patience=5,
    load_best_model_at_end=True,
    train_mode=None,
):
    """
    Train a given model using the provided training and validation dataloaders.
    Args:
        model (torch.nn.Module):
            The model to be trained.
        trn_dataloader (torch.utils.data.DataLoader):
            DataLoader for the training data.
        val_dataloader (torch.utils.data.DataLoader):
            DataLoader for the validation data.
        optimizer (torch.optim.Optimizer):
            Optimizer for updating the model parameters.
        version (str):
            Version identifier for saving the model.
        model_save_path (str):
            Path to save the trained model.
        model_type (str):
            Type of the model, used in the filename when saving.
        epochs (int, optional):
            Number of epochs to train the model. Default is 10.
        criterion (torch.nn.Module, optional):
            Loss function. Default is nn.BCELoss().
        early_stopping_patience (int, optional):
            Number of epochs with no improvement after which training will be
            stopped. Default is 5.
        load_best_model_at_end (bool, optional):
            Whether to load the best model at the end of training.
            Default is True.
        train_mode (str, optional):
            Mode to process RNN outputs.
            Options are "last_state", "mean_pool", "max_pool".
            Default is None, which uses the original output
            without modification.
    Returns:
        tuple: A tuple containing:
            - losses (list of float):
                List of loss values for each epoch.
            - accuracies (list of float):
                List of accuracy values for each epoch.
    """
    create_directory(model_save_path)
    losses = []  # List to store loss values
    accuracies = []  # List to store accuracy values
    best_accuracy = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in trn_dataloader:
            optimizer.zero_grad()

            # Forward pass through the model
            outputs = model(X_batch)

            # Select output based on train_mode
            if train_mode == "last_state":
                output = last_hidden_state(outputs)
            elif train_mode == "mean_pool":
                output = mean_pooling(outputs)
            elif train_mode == "max_pool":
                output = max_pooling(outputs)
            else:
                output = outputs

            if train_mode:
                output = output.view(output.size(0), -1)

            # Calculate the loss
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1:>3}/{epochs:>3},", end=" ")
        print(f"Loss: {total_loss / len(trn_dataloader):.4f},", end=" ")

        if train_mode:
            filepath = f"{model_save_path}{
                model_type}_{train_mode}_v{version}.pth"
        else:
            filepath = f"{model_save_path}{model_type}_v{version}.pth"

        accuracy = validate(model, val_dataloader)

        # record avg loss and accuracy
        accuracies.append(accuracy)
        losses.append(total_loss / len(trn_dataloader))

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            patience_counter = 0  # Reset patience counter
            # model is improving, save the model
            save_model(model, filepath)
        else:
            patience_counter += 1

        # Check for early stopping
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

    # Load the best model if specified
    if load_best_model_at_end:
        print("Training ended, loading best model...")
        load_model(model, filepath)

    return losses, accuracies


def plot_loss_accuracy(losses, accuracies):
    epochs = range(1, len(losses) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, losses, 'b-', label="Training Loss")
    plt.plot(epochs, accuracies, 'r-', label="Validation Accuracy")

    plt.title("Training Loss and Validation Accuracy over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss / Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()


def save_embeddings(word_index, word2vec_model, save_path):
    """
    Save the embeddings for the words in the vocabulary to a file.
    Args:
        word_index (dict):
            Dictionary mapping words to their respective indices.
        word2vec_model (gensim.models.KeyedVectors):
            The Word2Vec model containing the embeddings.
        save_path (str):
            The file path where the embeddings will be saved.
    """
    # Get the vector size dynamically if `vector_size` is not available
    vector_size = word2vec_model.vector_size if hasattr(word2vec_model, 'vector_size') else len(
        word2vec_model[word_index.keys().__iter__().__next__()])

    embedding_matrix = np.zeros((len(word_index), vector_size))
    for word, i in word_index.items():
        if word in word2vec_model.key_to_index:
            embedding_matrix[i] = word2vec_model[word]
    np.save(save_path, embedding_matrix)
    print("Embeddings saved.")
