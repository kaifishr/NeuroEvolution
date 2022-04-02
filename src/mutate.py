"""Methods for hyperparameter mutation.
"""

import random
from copy import deepcopy
from typing import Union

import numpy as np


def mutate_hparams(config: dict) -> dict:
    """Randomly mutates hyperparameters according to specification in configuration.

    Args:
        config: Dictionary holding current network configuration.

    Returns:
        Dictionary holding mutated hyperparameters.

    """
    config = deepcopy(config)

    global_mutation_rate = config["global_mutation_rate"]

    # Mutate parameters
    for hparam in config["hparam"].values():

        if hparam["mutate"]:

            if random.random() < global_mutation_rate:
                dtype = hparam["dtype"]

                if hparam["ptype"] == "scalar":
                    value = hparam["val"]
                    value = mutate_value(value, dtype, config, hparam)
                    hparam["val"] = value

                elif hparam["ptype"] == "vector":
                    val = []

                    for value in hparam["val"]:
                        value = mutate_value(value, dtype, config, hparam)
                        val.append(value)

                    hparam["val"] = val

                else:
                    raise NotImplementedError("Parameter type must be 'scalar' or 'vector'")

    return config


def mutate_value(value: Union[float, int],
                 dtype: str,
                 config: dict,
                 hparam: dict) -> Union[float, int]:
    """Mutates single value of hyperparameter.

    Args:
        value: Scalar value of type float or int.
        dtype: Datatype.
        config: Dictionary holding configuration.
        hparam:

    Returns:
        Mutated value.

    """
    local_mutation_rate = config["local_mutation_rate"]
    mutation_operator = config["mutation_operator"]

    # Compute step size based on values magnitude
    if mutation_operator == "proportional":
        eta = comp_proportional_step_size(value, local_mutation_rate, dtype)
    elif mutation_operator == "discrete":
        eta = comp_discrete_step_size(hparam)
    else:
        raise NotImplementedError(f"Mutation operator '{mutation_operator}' not implemented.")

    # Update value
    value += eta

    # Ensure value is within desired bounds
    value = np.clip(value, hparam["val_min"], hparam["val_max"])

    # Cast to correct datatype
    if dtype == "int":
        value = int(value)
    else:  # dtype == "float":
        value = float(value)

    return value


def comp_discrete_step_size(hparam: dict) -> Union[int, float]:
    """Computes discrete step size.

    Args:
        hparam: Configuration for hyperparameter.

    Returns:
        Step size for value.

    """
    dtype = hparam["dtype"]
    step_size = hparam["step_size"]

    eta = step_size
    if dtype == "float":
        eta = step_size * random.random()

    return rand_sign() * eta


def comp_proportional_step_size(value: Union[float, int],
                                local_mutation_rate: float,
                                dtype: str) -> Union[int, float]:
    """Computes step size proportional to value.

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
