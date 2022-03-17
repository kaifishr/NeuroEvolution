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


def load_config(config_path: str, hparam_path: str) -> dict:
    """Loads and enriches configuration file with additional information.

    Args:
        config_path: Path to configuration file.
        hparam_path: Path to file with hyperparameters.

    Returns:
        Configuration as Python dictionary.

    """
    configuration = dict()

    # Load configuration files
    config = load_yaml(file_path=config_path)
    hparam = load_yaml(file_path=hparam_path)

    configuration.update(config)
    configuration.update(hparam)

    # Modify config to allow evolving each layer size individually
    n_dims_hidden = configuration["hparam"]["n_dims_hidden"]["val"]
    n_layers_hidden = configuration["hparam"]["n_layers_hidden"]["val"]
    configuration["hparam"]["n_dims_hidden"]["val"] = n_layers_hidden * [n_dims_hidden]

    return configuration
