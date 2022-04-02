"""Method to train neural network.
"""
from torch import nn
from torch import optim

from src.models import MLP
from src.stats import comp_loss_accuracy


def train(dataloader: tuple, config: dict) -> dict:
    """Trains model according to provided configuration.

    Args:
        dataloader: PyTorch dataloader.
        config: Dictionary holding configuration.

    Returns:
        Dictionary holding metrics from training.

    """
    device = config["device"]
    n_epochs = config["n_epochs"]
    learning_rate = config["hparam"]["learning_rate"]["val"]
    weight_decay = config["hparam"]["weight_decay"]["val"]

    model = MLP(config=config)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    trainloader, testloader = dataloader

    for _ in range(n_epochs):

        for x_data, y_data in trainloader:

            # Move data to device.
            inputs, labels = x_data.to(device), y_data.to(device)

            # Zero network parameters.
            optimizer.zero_grad()

            # Forward pass.
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass.
            loss.backward()

            # Gradient descent.
            optimizer.step()

    test_loss, test_accuracy = comp_loss_accuracy(model=model,
                                                  criterion=criterion,
                                                  dataloader=testloader,
                                                  device=device)

    train_loss, train_accuracy = comp_loss_accuracy(model=model,
                                                    criterion=criterion,
                                                    dataloader=trainloader,
                                                    device=device)

    stats = {
        "test_loss": test_loss,
        "train_loss": train_loss,
        "test_accuracy": test_accuracy,
        "train_accuracy": train_accuracy
    }

    return stats
