"""Evolution of neural networks with genetic algorithms.

"""
from src.trainer import train
from src.mutate import mutate_hparams
from src.data import get_dataloader
from src.config import load_config

import copy
import numpy as np
import json

from torch.utils.tensorboard import SummaryWriter


def main():

    config_path = "config.yml"
    hparam_path = "hparams.yml"

    base_config = load_config(config_path=config_path, hparam_path=hparam_path)
    print(json.dumps(base_config, indent=4))

    n_agents = base_config["n_agents"]
    n_generations = base_config["n_generations"]

    # Use same config initially
    configs = [copy.deepcopy(base_config) for _ in range(n_agents)]

    dataset = base_config['dataset']
    writer = SummaryWriter(comment=f"_{dataset}")

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

            dataloader = get_dataloader(config=config)
            stats = train(dataloader=dataloader, config=config)

            train_losses.append(stats["train_loss"])
            train_accuracies.append(stats["train_accuracy"])
            test_losses.append(stats["test_loss"])
            test_accuracies.append(stats["test_accuracy"])

        # Get the best agent of current iteration
        best_agent_idx = np.argmin(test_losses)
        best_config = configs[best_agent_idx]
        configs = [mutate_hparams(config=best_config) for _ in range(n_agents)]

        # Write current values of hyperparameters to Tensorboard
        for hparam_name, hparam in best_config["hparam"].items():
            if hparam_name == "n_dims_hidden":
                for idx, val in enumerate(hparam["val"]):
                    writer.add_scalar(f"time_series_layer_{idx}_{hparam_name}",
                                      val, global_step=i)
            else:
                writer.add_scalar(f"time_series_{hparam_name}", hparam["val"], global_step=i)

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

        for hparam_name, hparam in best_config["hparam"].items():
            if hparam_name == "n_dims_hidden":
                for idx, val in enumerate(hparam["val"]):
                    hparam_dict[f"hparam_layer_{idx}_{hparam_name}"] = val
            else:
                hparam_dict[f"hparam_{hparam_name}"] = hparam["val"]

        metric_dict["metric_train_loss"] = train_loss
        metric_dict["metric_train_accuracy"] = train_accuracy
        metric_dict["metric_test_loss"] = test_loss
        metric_dict["metric_test_accuracy"] = test_accuracy

        writer.add_hparams(hparam_dict=hparam_dict, metric_dict=metric_dict, run_name=f"gen_{i}")

    writer.close()


if __name__ == "__main__":
    main()
