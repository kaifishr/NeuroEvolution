import yaml


def load_yaml(file_path: str) -> dict:
    """Loads YAML file.

    Args:
        file_path: Path to yaml file.

    Returns:
        Dictionary holding content of yaml file.

    """
    with open(file_path, "r") as fp:
        try:
            return yaml.safe_load(fp)
        except yaml.YAMLError as exc:
            print(exc)


def load_config(file_path: str) -> dict:
    """Loads and enriches configuration file with additional information.

    Args:
        file_path: Path to configuration file.

    Returns:
        Configuration as Python dictionary.

    """
    # Load yaml configuration file.
    config = load_yaml(file_path=file_path)

    # Modify config to allow evolving each layer size individually
    n_dims_hidden = config["hparams"]["n_dims_hidden"]["val_init"]
    n_layers_hidden = config["hparams"]["n_layers_hidden"]["val_init"]
    config["hparams"]["n_dims_hidden"]["val_init"] = n_layers_hidden * [n_dims_hidden]

    return config
