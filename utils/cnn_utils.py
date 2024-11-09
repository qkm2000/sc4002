import os
import torch
import torch.nn as nn
from tqdm import tqdm
from utils.utils import create_directory
import matplotlib.pyplot as plt


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
    device='cuda' if torch.cuda.is_available() else 'cpu',
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


def train_cnn(
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
    device='cuda' if torch.cuda.is_available() else 'cpu',
):
    """
    Train a CNN model using the provided training and validation dataloaders.
    Args:
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
            Number of epochs with no improvement
            after which training will be stopped. Default is 5.
        load_best_model_at_end (bool, optional):
            Whether to load the best model at the end of training.
            Default is True.
    Returns:
        tuple: A tuple containing:
            - losses (list of float):
                List of loss values for each epoch.
            - accuracies (list of float):
                List of accuracy values for each epoch.
    """
    create_directory(model_save_path)

    model.to(device)
    train_losses = []
    val_accuracies = []
    best_accuracy = 0
    patience_counter = 0  # Counter for early stopping

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        # Training loop
        for X_batch, y_batch in tqdm(trn_dataloader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch"):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()

            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(trn_dataloader)
        train_losses.append(avg_loss)

        # Validation step
        val_accuracy = validate(model, val_dataloader)
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {
              avg_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        # Save model if validation accuracy improves
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            patience_counter = 0  # Reset patience counter
            filepath = f"{model_save_path}{model_type}_v{version}.pth"
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
