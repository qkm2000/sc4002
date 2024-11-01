from torch.utils.data import dataloader, Dataset
from torch.nn.utils.rnn import pad_sequence
from torch import nn
import torch
from gensim.utils import simple_preprocess
import gensim.downloader as api
import gensim
import nltk
import os


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


def load_word2vec(
    filepath=None,
    word2vec_api=None,
):
    """
    Load a Word2Vec model from a file or a pre-trained model from the
    gensim API.

    Args:
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
      will load the "fasttext-wiki-news-subwords-300" model from the
      gensim API.
    """
    if word2vec_api is not None:
        return api.load(word2vec_api)
    if filepath is not None:
        return gensim.models.KeyedVectors.load(filepath)
    return api.load("fasttext-wiki-news-subwords-300")


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
    """
    best_accuracy = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in trn_dataloader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1:>3}/{epochs:>3},", end=" ")
        print(f"Loss: {total_loss / len(trn_dataloader):.4f},", end=" ")

        accuracy = validate(model, val_dataloader)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            patience_counter = 0  # Reset patience counter
            # model is improving, save the model
            save_model(model, f"{model_save_path}{model_type}_v{version}.pth")
        else:
            patience_counter += 1

        # Check for early stopping
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

    # load the best model and evaluate on the test set
    if load_best_model_at_end:
        print("Training ended, loading best model...")
        load_model(model, f"{model_save_path}{model_type}_v{version}.pth")
