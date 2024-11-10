import os
import torch
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


def train_hybrid_model(
    model,
    trn_dataloader,
    val_dataloader,
    optimizer,
    criterion,
    epochs=10,
    model_save_path='saved_models/',
    early_stopping_patience=5,
    load_best_model_at_end=True,
    device='cuda' if torch.cuda.is_available() else 'cpu',
):
    create_directory(model_save_path)
    # Move model to specified device
    model.to(device)

    best_accuracy = 0
    patience_counter = 0  # for early stopping

    # Lists to store losses and accuracies for each epoch
    train_losses = []
    val_accuracies = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for X_batch, y_batch in trn_dataloader:
            # Move data to the specified device
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            output = model(X_batch)

            # Calculate loss
            loss = criterion(output, y_batch.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Calculate average loss for this epoch
        avg_loss = total_loss / len(trn_dataloader)
        train_losses.append(avg_loss)  # Store training loss for the epoch
        print(f"Epoch {epoch+1:>3}/{epochs:>3}, Loss: {avg_loss:.4f}", end=", ")

        # Validate model
        accuracy = validate(model, val_dataloader, device)
        val_accuracies.append(accuracy)  # Store validation accuracy for the epoch

        # Save the best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            patience_counter = 0
            best_model_path = f"{model_save_path}cnnrnn_hybrid_model_best.pth"
            save_model(model, best_model_path)
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= early_stopping_patience:
            print("Early stopping triggered.")
            break

    # Load the best model at the end if specified
    if load_best_model_at_end:
        print("Loading the best model from saved checkpoint...")
        load_model(model, best_model_path)

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
