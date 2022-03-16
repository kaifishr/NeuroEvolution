import numpy as np
import random
import torch
import yaml
import copy
import math
# from scipy.stats import loguniform


def get_hparams() -> list:
    """Lookup table with names of hyperparameters.

    Returns:
        List of hyperparameters.

    """
    hparams = [
        "batch_size",
        "learning_rate",
        "dropout_rate",
        "weight_decay",
        "n_dims_hidden",
        "n_layers_hidden",
    ]
    return hparams


@torch.no_grad()
def comp_loss_accuracy(model, criterion, dataloader, device) -> tuple[float, float]:

    running_loss = 0.0
    running_accuracy = 0.0
    running_counter = 0

    model.eval()

    for x_data, y_data in dataloader:
        inputs, labels = x_data.to(device), y_data.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels).item()
        pred = (torch.argmax(outputs, dim=1) == labels).float().sum().item()
        running_loss += loss
        running_accuracy += pred
        running_counter += labels.size(0)

    model.train()

    loss = running_loss / running_counter
    accuracy = running_accuracy / running_counter

    return loss, accuracy




# def set_random_seeds(random_seed: int) -> None:
#     """Seeds random number generators of PyTorch, Numpy, and Random module.
#
#     Args:
#         random_seed: Initial random seed.
#
#     """
#     torch.manual_seed(random_seed)
#     np.random.seed(random_seed)
#     # random.seed(random_seed)