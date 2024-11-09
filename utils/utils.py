from torch.utils.data import dataloader, Dataset
from torch.nn.utils.rnn import pad_sequence
from gensim.utils import simple_preprocess
from gensim.models import KeyedVectors
import gensim.downloader as api
import numpy as np
import torch
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
