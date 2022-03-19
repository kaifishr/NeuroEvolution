import numpy as np
import random

from copy import deepcopy
from typing import Union


def mutate_hparams(config: dict) -> dict:
    """Randomly mutates hyperparameters according to specification in configuration.

    Args:
        config: Dictionary holding current network configuration.

    Returns:
        Dictionary holding mutated hyperparameters.

    """
    config = deepcopy(config)

    global_mutation_rate = config["global_mutation_rate"]
    local_mutation_rate = config["local_mutation_rate"]

    # Mutate parameters
    for hparam_name, hparam in config["hparam"].items():

        if hparam["mutate"]:

            if random.random() < global_mutation_rate:

                if hparam["ptype"] == "scalar":
                    value = hparam["val"]

                    value = mutate_value(value, local_mutation_rate, **hparam)

                    hparam["val"] = value

                elif hparam["ptype"] == "vector":

                    val = list()
                    for value in hparam["val"]:
                        value = mutate_value(value, local_mutation_rate, **hparam)
                        val.append(value)

                    hparam["val"] = val

                else:
                    raise NotImplementedError("Parameter type must be 'scalar' or 'vector'")

    return config


def mutate_value(value, local_mutation_rate, dtype, **hparam) -> Union[int, float]:
    """Mutates single value of hyperparameter.

    Args:
        value:
        local_mutation_rate:
        dtype:
        **hparam:

    Returns:
        Mutated value.

    """
    # Compute step size based on values magnitude
    eta = comp_step_size(value, local_mutation_rate, dtype)

    # Update value
    value = value + eta

    # Ensure value is within desired bounds
    value = np.clip(value, hparam["val_min"], hparam["val_max"])

    # Cast to correct datatype
    if dtype == "int":
        val = int(value)
    elif dtype == "float":
        val = float(value)
    else:
        raise NotImplementedError(f"'Type' must be 'int' or 'float', got {dtype = } instead.")

    return val


def comp_step_size(value, local_mutation_rate, dtype) -> Union[int, float]:
    """Computes step size.

    Args:
        value:
        local_mutation_rate:
        dtype:

    Returns:
        Step size for value.

    """
    # Compute step size
    eta = value * local_mutation_rate * random.random()

    # Ensure change of parameter
    if dtype == "int":
        eta = eta if eta > 1 else 1

    return rand_sign() * eta


def rand_sign() -> int:
    """Random sign.

    Returns:
        Randomly generated -1 or 1.

    """
    return 1 if random.random() < 0.5 else -1
