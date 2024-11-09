from torch import nn
import torch
import os
import matplotlib.pyplot as plt
from utils.utils import create_directory


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
            Default is None, which uses the original output
            without modification.
            Options are:
                None,
                "last_state",
                "mean_pool",
                "max_pool",
                "mean_max",
                "attention".
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
            elif train_mode == "mean_max":
                mean_pooled = mean_pooling(outputs)
                max_pooled = max_pooling(outputs)
                output = (mean_pooled + max_pooled) / 2
            elif train_mode == "attention":
                output = apply_attention(outputs)
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
            filepath = f"{model_save_path}{model_type}_{train_mode}_v{version}.pth"
        else:
            filepath = f"{model_save_path}{model_type}_v{version}.pth"

        # Validation accuracy
        accuracy = validate(model, val_dataloader)

        # record avg loss and accuracy
        accuracies.append(accuracy)
        losses.append(total_loss / len(trn_dataloader))

        # Save the best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            patience_counter = 0
            save_model(model, filepath)
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

    # Load the best model if specified
    if load_best_model_at_end:
        print("Training ended, loading best model...")
        load_model(model, filepath)

    return losses, accuracies


# Helper functions for the new modes
def apply_attention(outputs):
    # Attention mechanism based on a simple weighted mask
    # Customize this function based on the specific attention mask logic
    weights = torch.softmax(outputs, dim=1)  # Simple example, adjust as needed
    weighted_output = (weights * outputs).sum(dim=1)
    return weighted_output


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
