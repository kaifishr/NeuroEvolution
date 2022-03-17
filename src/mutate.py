import copy
import math
import numpy as np
import random


def mutate_config(config: dict) -> dict:  # todo: better mutate_hparams()
    """Generates random configuration for network parameters.

    Args:
        config: Dictionary holding current network configuration.

    """
    config = copy.deepcopy(config)

    # Setting parameters boundaries
    # batch_size_min = 4
    # batch_size_max = 128
    # learning_rate_min = 1e-6
    # learning_rate_max = 1e-1
    # weight_decay_min = 1e-6
    # weight_decay_max = 1e-2
    # dropout_rate_min = 0.01
    # dropout_rate_max = 0.5
    # n_dims_hidden_min = 4
    # n_dims_hidden_max = 128
    # # n_layers_hidden_min = 2
    # # n_layers_hidden_max = 16

    # todo: move these two bois to config
    local_mutation_rate = 0.1  # 0.05, 0.1, 0.2
    global_mutation_rate = 0.2

    def sign() -> float:
        return 1.0 if random.random() < 0.5 else -1.0

    # def sign() -> float:  # +-1
    #     return random.uniform(-1.0, 1.0)

    # Compute magnitude of number
    def num(x) -> float:
        mag = math.floor(math.log10(x))
        return math.pow(10.0, mag)

    # def num(x) -> float:
    #     return x

    # Mutate parameters
    for hparam_name, hparam in config["hparams"].items():
        print(hparam_name)
        if hparam["mutate"]:
            if global_mutation_rate > random.random():
                if hparam["dtype"] == "int":
                    val = hparam["val"]
                    val = val + sign() * local_mutation_rate
                    val = np.clip(val, hparam["val_min"], hparam["val_max"])
                    hparam["val"] = int(val)
                elif hparam["dtype"] == "float":
                    val = hparam["val"]
                    val = val + sign() * num(val) * local_mutation_rate
                    val = np.clip(val, hparam["val_min"], hparam["val_max"])
                    hparam["val"] = float(val)
                else:
                    raise NotImplementedError("Datatype must be 'float' or 'int'.")

        # if param_name == "batch_size":
        #     if global_mutation_rate > random.random():
        #         # Mutate parameter
        #         new_value = value + sign()
        #         # Sanity check
        #         new_value = np.clip(new_value, batch_size_min, batch_size_max)
        #         config[param_name] = int(new_value)

        # elif param_name == "learning_rate":
        #     if global_mutation_rate > random.random():
        #         # Mutate parameter
        #         new_value = value + sign() * num(value) * local_mutation_rate
        #         # Sanity check
        #         new_value = np.clip(new_value, learning_rate_min, learning_rate_max)
        #         config[param_name] = float(new_value)

        # elif param_name == "weight_decay":
        #     if global_mutation_rate > random.random():
        #         # Mutate parameter
        #         new_value = value + sign() * num(value) * local_mutation_rate
        #         # Sanity check
        #         new_value = np.clip(new_value, weight_decay_min, weight_decay_max)
        #         config[param_name] = float(new_value)

        # elif param_name == "dropout_rate":
        #     if global_mutation_rate > random.random():
        #         # Mutate parameter
        #         new_value = value + sign() * 0.01
        #         # Sanity check
        #         new_value = np.clip(new_value, dropout_rate_min, dropout_rate_max)
        #         config[param_name] = float(new_value)

        # elif param_name == "n_dims_hidden":
        #     for i, val in enumerate(value):
        #         if global_mutation_rate > random.random():
        #             # Mutate parameter
        #             new_value = val + sign()
        #             # Sanity check
        #             new_value = np.clip(new_value, n_dims_hidden_min, n_dims_hidden_max)
        #             value[i] = int(new_value)
        #     config[param_name] = value

        # elif param_name == "n_layers_hidden":
        #     if global_mutation_rate > random.random():
        #         # Mutate parameter
        #         new_value = value + sign()
        #         # Sanity check
        #         new_value = np.clip(new_value, n_layers_hidden_min, n_layers_hidden_max)
        #         config[param_name] = int(new_value)

    return config
