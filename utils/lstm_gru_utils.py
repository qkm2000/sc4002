import os
import torch
import torch.nn as nn
from tqdm import tqdm
from utils.utils import create_directory
import matplotlib.pyplot as plt


device = 'cuda' if torch.cuda.is_available() else 'cpu'


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


def validate(
    model,
    val_dataloader,
    device=device,
):
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
            inputs, labels = inputs.to(device), labels.to(device)
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


def apply_attention(outputs):
    # Attention mechanism based on a simple weighted mask
    # Customize this function based on the specific attention mask logic
    weights = torch.softmax(outputs, dim=1)  # Simple example, adjust as needed
    weighted_output = (weights * outputs).sum(dim=1)
    return weighted_output


def train_lstm_gru_model(
    model,
    trn_dataloader,
    val_dataloader,
    optimizer,
    criterion,
    epochs,
    model_save_path,
    early_stopping_patience=5,
    load_best_model_at_end=True,
    device=device,
    train_mode=None,
):
    model = model.to(device)
    best_val_accuracy = 0
    patience_counter = 0
    train_losses = []
    val_accuracies = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for inputs, labels in trn_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

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

            loss = criterion(outputs.squeeze(), labels.squeeze())
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        train_losses.append(epoch_loss / len(trn_dataloader))

        print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {train_losses[-1]:.4f}")

        # Validation
        val_accuracy = validate(model, val_dataloader, device)
        val_accuracies.append(val_accuracy)

        # Early Stopping
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path + "best_model.pth")
            print("Model saved")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered")
                break

    if load_best_model_at_end:
        model.load_state_dict(torch.load(model_save_path + "best_model.pth"))

    return train_losses, val_accuracies


def plot_training_progress(train_losses, val_accuracies):
    """
    Plot the training progress of the CNN model,
    including training losses and validation accuracies on the same graph.

    Args:
        train_losses (list of float):
            List of training losses for each epoch.
        val_accuracies (list of float):
            List of validation accuracies for each epoch.
    """
    epochs = range(1, len(train_losses) + 1)  # Number of epochs

    # Create a figure and a set of subplots
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot training loss on the first y-axis
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Training Loss', color='blue')
    ax1.plot(epochs, train_losses, label='Training Loss', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Create a second y-axis to plot validation accuracy
    ax2 = ax1.twinx()
    ax2.set_ylabel('Validation Accuracy', color='green')
    ax2.plot(
        epochs,
        val_accuracies,
        label='Validation Accuracy',
        color='green'
    )
    ax2.tick_params(axis='y', labelcolor='green')

    # Add a title and grid
    plt.title('Training Loss and Validation Accuracy Over Epochs')
    ax1.grid(True)

    # Show legend for both y-axes
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # Display the plot
    plt.tight_layout()
    plt.show()
