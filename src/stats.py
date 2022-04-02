"""Module holds method to compute loss and accuracy for provided network and dataloader.
"""
import torch


@torch.no_grad()
def comp_loss_accuracy(model, criterion, dataloader, device) -> tuple[float, float]:
    """Compute loss and accuracy for provided model and dataloader.

    Args:
        model: PyTorch neural network.
        criterion: Loss function.
        dataloader: PyTorch dataloader.
        device: CPU / GPU.

    Returns:
        Loss and accuracy.

    """
    running_loss = 0.0
    running_accuracy = 0.0
    running_counter = 0

    model.train(False)

    for x_data, y_data in dataloader:
        inputs, labels = x_data.to(device), y_data.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels).item()
        predictions = (torch.argmax(outputs, dim=1) == labels).float().sum().item()
        running_loss += loss
        running_accuracy += predictions
        running_counter += labels.size(0)

    model.train(True)

    loss = running_loss / running_counter
    accuracy = running_accuracy / running_counter

    return loss, accuracy
