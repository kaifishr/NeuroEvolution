"""Evolution of neural networks with genetic algorithms.

todo:
    - mutate according to magnitude
"""
from src.mutate import mutate_config
from src.models import MLP
from src.utils import (
    comp_loss_accuracy,
    # set_random_seeds,
    get_hparams,
)
from src.data import get_dataloader
from src.config import load_config

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from torch.utils.tensorboard import SummaryWriter


def train(dataloader: tuple, config: dict) -> dict:

    n_epochs = config["n_epochs"]
    learning_rate = config["learning_rate"]
    weight_decay = config["weight_decay"]
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


def main():

    file_path = "config.yml"
    base_config = load_config(file_path=file_path)
    print(yaml.dump(base_config))

    n_agents = base_config["n_agents"]
    n_generations = base_config["n_generations"]

    # Use same config initially
    configs = [copy.deepcopy(base_config) for _ in range(n_agents)]

    # Parameters to track
    hparams = get_hparams()

    dataset = base_config['dataset']
    writer = SummaryWriter(comment=f"_{dataset}_evo")

    dataloader = get_dataloader(config=base_config)

    for i in range(n_generations):
        print(f"Iteration {i:05d}")

        # Test agents of current iteration
        train_losses = list()
        train_accuracies = list()
        test_losses = list()
        test_accuracies = list()

        for config in configs:
            # if (i+1) % increase_epochs_every_n == 0:
            #     config["n_epochs"] += 1

            stats = train(dataloader=dataloader, config=config)

            train_losses.append(stats["train_loss"])
            train_accuracies.append(stats["train_accuracy"])
            test_losses.append(stats["test_loss"])
            test_accuracies.append(stats["test_accuracy"])

        # Get the best agent of current iteration
        best_agent_idx = np.argmin(test_losses)
        best_config = configs[best_agent_idx]
        configs = [mutate_config(config=best_config) for _ in range(n_agents)]

        # Write current values of hyperparameters to Tensorboard
        for hparam_name, value in best_config.items():
            if hparam_name in hparams:
                if hparam_name == "n_dims_hidden":
                    for idx, val in enumerate(value):
                        writer.add_scalar(f"time_series_layer_{idx}_{hparam_name}",
                                          val, global_step=i)
                else:
                    writer.add_scalar(f"time_series_{hparam_name}", value, global_step=i)

        # Add scalars to tensorboard
        train_loss = train_losses[best_agent_idx]
        train_accuracy = train_accuracies[best_agent_idx]
        test_loss = test_losses[best_agent_idx]
        test_accuracy = test_accuracies[best_agent_idx]

        writer.add_scalar("time_series_train_loss", train_loss, global_step=i)
        writer.add_scalar("time_series_train_accuracy", train_accuracy, global_step=i)
        writer.add_scalar("time_series_test_loss", test_loss, global_step=i)
        writer.add_scalar("time_series_test_accuracy", test_accuracy, global_step=i)

        # Add hyperparameters and metrics to tensorboard
        hparam_dict = dict()
        metric_dict = dict()

        for hparam_name, value in best_config.items():
            if hparam_name in hparams:
                if hparam_name == "n_dims_hidden":
                    for idx, val in enumerate(value):
                        hparam_dict[f"hparam_layer_{idx}_{hparam_name}"] = val
                else:
                    hparam_dict[f"hparam_{hparam_name}"] = value

        metric_dict["metric_train_loss"] = train_loss
        metric_dict["metric_train_accuracy"] = train_accuracy
        metric_dict["metric_test_loss"] = test_loss
        metric_dict["metric_test_accuracy"] = test_accuracy

        writer.add_hparams(hparam_dict=hparam_dict, metric_dict=metric_dict, run_name=f"gen_{i}")

    writer.close()


if __name__ == "__main__":
    main()
