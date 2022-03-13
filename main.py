"""Evolution of neural networks with genetic algorithms.

todo:
    - add hyperparameters to tensorboard to see correlations between parameters
    - mutate according to magnitude
    - move dataloader to main()
"""
from src.mutate import mutate_config

import copy

import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import time

from torch.utils.tensorboard import SummaryWriter

from itertools import permutations

from src.models import MLP
from src.utils import (
    comp_test_stats,
    load_yaml,
    # set_random_seeds,
    get_hparams,
)
from src.data import get_dataloader


def train(dataloader: tuple, config: dict, writer) -> float:

    n_epochs = config["n_epochs"]
    learning_rate = config["learning_rate"]
    step_size = config["step_size"]
    gamma = config["gamma"]
    weight_decay = config["weight_decay"]
    stats_every_n_epochs = config["stats_every_n_epochs"]

    # random_seed = config["random_seed"]
    # set_random_seeds(random_seed=random_seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = MLP(config=config)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    trainloader, testloader = dataloader

    for epoch in range(n_epochs):

        running_loss = 0.0
        running_accuracy = 0.0
        running_counter = 0

        model.train()
        for x_data, y_data in trainloader:

            # get the inputs; data is a list of [inputs, lables]
            inputs, labels = x_data.to(device), y_data.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + gradient descent
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # keeping track of statistics
            running_loss += loss.item()
            running_accuracy += (torch.argmax(outputs, dim=1) == labels).float().sum()
            running_counter += labels.size(0)

        # if (epoch % stats_every_n_epochs == 0) or (epoch + 1 == n_epochs):  # print every n epochs
        #     running_loss = running_loss / running_counter
        #     running_accuracy = running_accuracy / running_counter

        #     writer.add_scalar("train_loss", running_loss, epoch)
        #     writer.add_scalar("train_accuracy", running_accuracy, epoch)

        #     test_loss, test_accuracy = comp_test_stats(model=model,
        #                                                criterion=criterion,
        #                                                test_loader=testloader,
        #                                                device=device)
        #     writer.add_scalar("test_loss", test_loss, epoch)
        #     writer.add_scalar("test_accuracy", test_accuracy, epoch)
        #     print(f"{epoch:04d} {running_loss:.3f} {running_accuracy:.4f} {test_loss:.3f} {test_accuracy:.3f}")

        # scheduler.step()

    test_loss, test_accuracy = comp_test_stats(model=model,
                                               criterion=criterion,
                                               test_loader=testloader,
                                               device=device)

    train_loss = running_loss / running_counter
    train_accuracy = running_accuracy / running_counter

    return train_loss, train_accuracy, test_loss, test_accuracy


def main():

    n_iterations = 9999999999
    n_agents = 4
    increase_epochs_every_n = 999999999

    file_path = "config.yml"
    base_config = load_yaml(file_path=file_path)
    print(yaml.dump(base_config))

    # Use same config initially
    configs = [copy.deepcopy(base_config) for _ in range(n_agents)]

    # Parameters to track
    hparams = get_hparams()

    dataset = base_config['dataset']
    writer = SummaryWriter(comment=f"_{dataset}_evo")
    model_writer = None  # SummaryWriter(comment=f"_{dataset}_model")

    batch_size = base_config["batch_size"]
    num_workers = base_config["n_workers"]
    dataloader = get_dataloader(dataset=dataset, batch_size=batch_size, num_workers=num_workers)

    for i in range(n_iterations):
        print(f"Iteration {i:05d}")

        # Test agents of current iteration
        train_losses = list()
        train_accuracies = list()
        test_losses = list()
        test_accuracies = list()
        for config in configs:
            # if (i+1) % increase_epochs_every_n == 0:
            #     config["n_epochs"] += 1
            stats = train(dataloader=dataloader, config=config, writer=model_writer)
            train_loss, train_accuracy, test_loss, test_accuracy = stats
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            test_losses.append(test_loss)
            test_accuracies.append(test_accuracy)

        # Get best agent
        best_agent_idx = np.argmin(test_losses)
        # best_agent_idx = np.argmin(train_losses)
        best_config = configs[best_agent_idx]

        # Write current values of hyperparameters to Tensorboard
        for hparam_name, value in best_config.items():
            if hparam_name in hparams:
                writer.add_scalar(f"hparam_{hparam_name}", value, global_step=i)

        print(yaml.dump(best_config))

        configs = [mutate_config(config=best_config) for _ in range(n_agents)]
        writer.add_scalar("metric_global_train_loss", train_losses[best_agent_idx], global_step=i)
        writer.add_scalar("metric_global_train_accuracy", train_accuracies[best_agent_idx], global_step=i)
        writer.add_scalar("metric_global_test_loss", test_losses[best_agent_idx], global_step=i)
        writer.add_scalar("metric_global_test_accuracy", test_accuracies[best_agent_idx], global_step=i)

    writer.close()
    # model_writer.close()


if __name__ == "__main__":
    main()
