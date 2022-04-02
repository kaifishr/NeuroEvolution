"""Module provides methods for datasets and data loaders.
"""
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

from src.random_subset import RandomSubset


def get_cifar10(n_workers: int, subset_ratio: float, **config: dict) -> tuple:
    """

    Args:
        n_workers: Number of workers.
        subset_ratio: Ratio defining size of subset.
        **config: Dictionary holding configuration.

    Returns:
        Train and test dataloader.

    """
    batch_size = config["hparam"]["batch_size"]["val"]
    batch_size_test = config["batch_size_test"]

    # Define transforms for dataset
    stats = {"mean": (0.4914, 0.4822, 0.4465), "std": (0.2023, 0.1994, 0.2010)}

    transform_train = transforms.Compose([
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(degrees=10),
        # transforms.RandomCrop(32, padding=5),
        transforms.ToTensor(),
        transforms.Normalize(**stats)
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(**stats)
    ])

    # Create dataset
    trainset_config = dict(root="./data", train=True, download=True, transform=transform_train)
    trainset = torchvision.datasets.CIFAR10(**trainset_config)

    testset_config = dict(root="./data", train=False, download=True, transform=transform_test)
    testset = torchvision.datasets.CIFAR10(**testset_config)

    # Use random subsets during training
    trainset = RandomSubset(dataset=trainset, subset_ratio=subset_ratio)
    testset = RandomSubset(dataset=testset, subset_ratio=subset_ratio)

    # Create dataloader
    trainloader_config = dict(dataset=trainset, batch_size=batch_size, shuffle=True,
                              num_workers=n_workers, pin_memory=True)
    trainloader = DataLoader(**trainloader_config)

    testloader_config = dict(dataset=testset, batch_size=batch_size_test, shuffle=False,
                             num_workers=n_workers, pin_memory=True)
    testloader = DataLoader(**testloader_config)

    return trainloader, testloader


def get_fashion_mnist(n_workers: int, subset_ratio: float, **config: dict) -> tuple:
    """

    Args:
        n_workers: Number of workers.
        subset_ratio: Ratio defining size of subset.
        **config: Dictionary holding configuration.

    Returns:
        Train and test dataloader.

    """
    batch_size = config["hparam"]["batch_size"]["val"]
    batch_size_test = config["batch_size_test"]

    # Fashion-MNIST
    stats = {"mean": (0.2859,), "std": (0.3530,)}

    # Define transforms for dataset
    transform_train = transforms.Compose([
        # transforms.RandomRotation(degrees=10),
        # transforms.RandomCrop(28, padding=2),
        transforms.ToTensor(),
        transforms.Normalize(**stats)
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(**stats)
    ])

    # Create dataset
    trainset_config = dict(root="./data", train=True, download=True, transform=transform_train)
    trainset = torchvision.datasets.FashionMNIST(**trainset_config)

    testset_config = dict(root="./data", train=False, download=True, transform=transform_test)
    testset = torchvision.datasets.FashionMNIST(**testset_config)

    # Use random subsets during training
    trainset = RandomSubset(dataset=trainset, subset_ratio=subset_ratio)
    testset = RandomSubset(dataset=testset, subset_ratio=subset_ratio)

    # Create dataloader
    trainloader_config = dict(dataset=trainset, batch_size=batch_size, shuffle=True,
                              num_workers=n_workers, pin_memory=True)

    testloader_config = dict(dataset=testset, batch_size=batch_size_test, shuffle=False,
                             num_workers=n_workers, pin_memory=True)

    trainloader = DataLoader(**trainloader_config)
    testloader = DataLoader(**testloader_config)

    return trainloader, testloader


def get_dataloader(dataset, **config: dict) -> tuple[DataLoader, DataLoader]:
    """Method returns dataloader for desired dataset.

    Args:
        dataset:
        **config:

    Returns:
        Training and test data loaders.

    """
    if dataset == "cifar10":
        trainloader, testloader = get_cifar10(**config)

    elif dataset == "fashion_mnist":
        trainloader, testloader = get_fashion_mnist(**config)

    else:
        raise NotImplementedError(f"Dataset {dataset} not implemented.")

    return trainloader, testloader
