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


# def sign() -> float:  # +-1
#     return random.uniform(-1.0, 1.0)


# Compute magnitude of number
def num(x: float) -> float:
    """Computes magnitude of number.

    Args:
        x: Scalar value.

    Returns:
        Magnitude of scalar value.

    """
    order = math.floor(math.log10(x))
    return math.pow(10.0, order)


def mutate_config(config: dict) -> dict:  # todo: better mutate_hparams()
    """Randomly mutates hyperparameters according to configuration.

    todo:
        * simplify logic

    Args:
        config: Dictionary holding current network configuration.

    Returns:
        Dictionary holding mutated hyperparameters.

    """
    config = copy.deepcopy(config)

    local_mutation_rate = config["local_mutation_rate"]
    global_mutation_rate = config["global_mutation_rate"]

    # Mutate parameters
    for hparam_name, hparam in config["hparam"].items():

        if hparam["mutate"]:

            if global_mutation_rate > random.random():

                if hparam["dtype"] == "int":

                    if hparam["ptype"] == "scalar":
                        val = hparam["val"]
                        val = val + rand_sign() * local_mutation_rate * num(val)
                        val = np.clip(val, hparam["val_min"], hparam["val_max"])
                        hparam["val"] = int(val)

                    elif hparam["ptype"] == "vector":
                        values = list()
                        for val in hparam["val"]:
                            # val = val + rand_sign() * local_mutation_rate * num(val)
                            val = val + rand_sign() * num(val)
                            val = np.clip(val, hparam["val_min"], hparam["val_max"])
                            values.append(int(val))
                        hparam["val"] = values

                elif hparam["dtype"] == "float":

                    if hparam["ptype"] == "scalar":
                        val = hparam["val"]
                        val = val + rand_sign() * local_mutation_rate * num(val)
                        val = np.clip(val, hparam["val_min"], hparam["val_max"])
                        hparam["val"] = float(val)

                    elif hparam["ptype"] == "vector":
                        values = list()
                        for val in hparam["val"]:
                            val = val + rand_sign() * local_mutation_rate * num(val)
                            val = np.clip(val, hparam["val_min"], hparam["val_max"])
                            values.append(float(val))
                        hparam["val"] = values
                else:
                    raise NotImplementedError("Datatype must be 'float' or 'int'.")

    return config
