"""Method to train neural network.

"""
from src.models import MLP
from src.utils import comp_loss_accuracy

import torch
import torch.nn as nn
import torch.optim as optim


def train(dataloader: tuple, config: dict) -> dict:

    n_epochs = config["n_epochs"]
    learning_rate = config["hparam"]["learning_rate"]["val"]
    weight_decay = config["hparam"]["weight_decay"]["val"]
    stats = dict()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = MLP(config=config)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    trainloader, testloader = dataloader

    for epoch in range(n_epochs):

        for x_data, y_data in trainloader:

            # get data
            inputs, labels = x_data.to(device), y_data.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + gradient descent
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    test_loss, test_accuracy = comp_loss_accuracy(model=model,
                                                  criterion=criterion,
                                                  dataloader=testloader,
                                                  device=device)

    train_loss, train_accuracy = comp_loss_accuracy(model=model,
                                                    criterion=criterion,
                                                    dataloader=trainloader,
                                                    device=device)

    stats["test_loss"] = test_loss
    stats["train_loss"] = train_loss
    stats["test_accuracy"] = test_accuracy
    stats["train_accuracy"] = train_accuracy

    return stats
