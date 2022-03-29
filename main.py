"""Evolution of neural networks with genetic optimization.

"""
from src.trainer import train
from src.mutate import mutate_hparams
from src.data import get_dataloader
from src.config import load_config

from copy import deepcopy
from datetime import datetime
import numpy as np
import time

from torch.utils.tensorboard import SummaryWriter


def main():

    config_path = "config/config.yml"
    hparam_path = "config/hparams.yml"

    base_config = load_config(config_path=config_path, hparam_path=hparam_path)

    n_agents = base_config["n_agents"]
    n_generations = base_config["n_generations"]

    # Use same config initially
    configs = [deepcopy(base_config) for _ in range(n_agents)]

    # Tensorboard writer
    dataset = base_config['dataset']
    writer = SummaryWriter(log_dir=f"./runs/{dataset}_{datetime.now()}")

    # Get data
    dataloader = get_dataloader(**base_config)

    # Create dictionary for running statistics
    stats_name = ("train_loss", "train_accuracy", "test_loss", "test_accuracy")
    stats_dict = {name: [] for name in stats_name}

    # Main optimization loop
    for i in range(n_generations):
        t0 = time.time()
        print(f"Iteration {i:05d}")

        # Reset all values in stats dictionary
        for k in stats_dict.keys():
            stats_dict[k] = list()

        # Loop over all agents
        for config in configs:

            stats = train(dataloader=dataloader, config=config)

            for key, value in stats.items():
                stats_dict[key].append(value)

        # Get the best agent of current iteration
        best_agent_idx = np.argmin(stats_dict["train_loss"])
        best_config = configs[best_agent_idx]
        configs = [mutate_hparams(config=best_config) for _ in range(n_agents)]

        # Track average time per generation
        time_per_generation = (time.time()-t0)/n_agents
        writer.add_scalar("time_series/time_per_generation", time_per_generation, global_step=i)

        # Add scalars to Tensorboard
        for key, value in stats_dict.items():
            writer.add_scalar(f"time_series/{key}", value[best_agent_idx], global_step=i)

        # Add hyperparameters to Tensorboard
        for hparam_name, hparam in best_config["hparam"].items():
            if hparam_name == "n_dims_hidden":
                for idx, val in enumerate(hparam["val"]):
                    writer.add_scalar(f"time_series/network/layer_{idx}_{hparam_name}",
                                      val, global_step=i)
            else:
                writer.add_scalar(f"time_series/{hparam_name}", hparam["val"], global_step=i)

        # Add hyperparameters and metrics of the best agent to tensorboard
        if base_config["add_hyperparameters"]:
            hparam_dict = dict()
            metric_dict = dict()

            for hparam_name, hparam in best_config["hparam"].items():
                if hparam_name == "n_dims_hidden":
                    for idx, val in enumerate(hparam["val"]):
                        hparam_dict[f"hparam_layer_{idx}_{hparam_name}"] = val
                else:
                    hparam_dict[f"hparam/{hparam_name}"] = hparam["val"]

            for key, value in stats_dict.items():
                writer.add_scalar(f"metric/{key}", value[best_agent_idx], global_step=i)

            writer.add_hparams(hparam_dict=hparam_dict, metric_dict=metric_dict, run_name=f"{i}")

    writer.close()


if __name__ == "__main__":
    main()
