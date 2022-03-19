import copy
import math
import numpy as np
import random


def rand_sign() -> float:
    """Random sign.

    Returns:
        Randomly generated -1 or 1.

    """
    return 1.0 if random.random() < 0.5 else -1.0


def num(x: float) -> float:
    """Computes magnitude of number.

    Args:
        x: Scalar value.

    Returns:
        Magnitude of scalar value.

    """
    order = math.floor(math.log10(x))
    return math.pow(10.0, order)


def comp_eta(value, scale, dtype, step_size, **config) -> float:
    """Computes magnitude of number.

    Args:
        x: Scalar value.

    Returns:
        Magnitude of scalar value.

    """
    if scale == "linear":
        eta = step_size
    elif scale == "log":
        eta = step_size * num(value)
    else:
        raise NotImplementedError

    return eta


def mutate_hparams(config: dict) -> dict:
    """Randomly mutates hyperparameters according to specification in configuration.

    Args:
        config: Dictionary holding current network configuration.

    Returns:
        Dictionary holding mutated hyperparameters.

    """
    config = copy.deepcopy(config)

    mutation_rate = config["mutation_rate"]

    # Mutate parameters
    for hparam_name, hparam in config["hparam"].items():

        if hparam["mutate"]:

            if mutation_rate > random.random():

                if hparam["ptype"] == "scalar":
                    val = hparam["val"]
                    eta = comp_eta(val, **hparam)
                    val = val + rand_sign() * eta
                    val = np.clip(val, hparam["val_min"], hparam["val_max"])
                    hparam["val"] = val

                    if hparam["dtype"] == "int":
                        hparam["val"] = int(val)
                    elif hparam["dtype"] == "float":
                        hparam["val"] = float(val)

                elif hparam["ptype"] == "vector":

                    val = list()
                    for value in hparam["val"]:
                        eta = comp_eta(value, **hparam)
                        value = value + rand_sign() * eta
                        value = np.clip(value, hparam["val_min"], hparam["val_max"])
                        val.append(value)

                    if hparam["dtype"] == "int":
                        val = list(map(int, val))
                    elif hparam["dtype"] == "float":
                        val = list(map(float, val))

                    hparam["val"] = val

                else:
                    raise NotImplementedError("Parameter type must be 'scalar' or 'vector'")

    return config


if __name__ == "__main__":

    # Quick sanity check
    import datetime
    from src.config import load_yaml
    from torch.utils.tensorboard import SummaryWriter

    for _ in range(10):
        config = load_yaml(file_path="../hparams.yml")
        config["local_mutation_rate"] = 0.01
        config["global_mutation_rate"] = 1.0
        config["hparam"]["n_dims_hidden"]["val"] = 4*[4, ]

        writer = SummaryWriter(f"../runs/{datetime.datetime.now()}")

        n_iterations = 1000
        for i in range(n_iterations):
            for hparam_name, hparam in config["hparam"].items():
                # print(hparam_name, hparam["val"])
                if isinstance(hparam["val"], list):
                    for j, v in enumerate(hparam["val"]):
                        writer.add_scalar(f"{hparam_name}_{j}", v, global_step=i)
                else:
                    writer.add_scalar(hparam_name, hparam["val"], global_step=i)
            # print()
            config = mutate_hparams(config=config)

        writer.close()
