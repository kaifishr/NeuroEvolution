import math
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from sklearn.datasets import make_blobs


def get_dataloader(dataset: str, batch_size: int, num_workers: int) \
        -> tuple[DataLoader, DataLoader]:

    trainloader = None
    testloader = None

    if dataset == "cifar10":

        avg = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)

        train_transforms = [
            transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=45),
            transforms.RandomCrop(32, padding=5),
            transforms.ToTensor(),
            transforms.Normalize(avg, std)
        ]

        test_transforms = [
            transforms.ToTensor(),
            transforms.Normalize(avg, std)
        ]

        transform_train = transforms.Compose(train_transforms)
        transform_test = transforms.Compose(test_transforms)

        trainset_config = dict(root="./data", train=True, download=True, transform=transform_train)
        trainset = torchvision.datasets.CIFAR10(**trainset_config)

        trainloader_config = dict(dataset=trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False)
        trainloader = DataLoader(**trainloader_config)


        testset_config = dict(root="./data", train=False, download=True, transform=transform_test)
        testset = torchvision.datasets.CIFAR10(**testset_config)

        testloader_config = dict(dataset=testset, batch_size=2*batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)
        testloader = DataLoader(**testloader_config)

    elif dataset == "cifar100":

        avg = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)

        train_transforms = [
            transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=45),
            transforms.RandomCrop(32, padding=5),
            transforms.ToTensor(),
            transforms.Normalize(avg, std)
        ]

        test_transforms = [
            transforms.ToTensor(),
            transforms.Normalize(avg, std)
        ]

        transform_train = transforms.Compose(train_transforms)
        transform_test = transforms.Compose(test_transforms)

        trainset_config = dict(root="./data", train=True, download=True, transform=transform_train)
        trainset = torchvision.datasets.CIFAR100(**trainset_config)

        trainloader_config = dict(dataset=trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False)
        trainloader = DataLoader(**trainloader_config)

        testset_config = dict(root="./data", train=False, download=True, transform=transform_test)
        testset = torchvision.datasets.CIFAR100(**testset_config)

        testloader_config = dict(dataset=testset, batch_size=2*batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)
        testloader = DataLoader(**testloader_config)

    elif dataset == "fashion_mnist":

        # Fashion-MNIST
        avg = (0.2859, )
        std = (0.3530, )

        transform_train = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                # transforms.RandomRotation(degrees=30),
                # transforms.RandomCrop(28, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(avg, std)
            ]
        )

        transform_test = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(avg, std)]
        )

        trainset = torchvision.datasets.FashionMNIST(root="./data",
                                                     train=True,
                                                     download=True,
                                                     transform=transform_train)

        subset_length = int(len(trainset) * 0.05)
        trainset = Subset(trainset, range(subset_length))

        trainloader = DataLoader(trainset,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=num_workers)

        testset = torchvision.datasets.FashionMNIST(root="./data",
                                                    train=False,
                                                    download=True,
                                                    transform=transform_test)

        testset = Subset(testset, range(subset_length))

        testloader = DataLoader(testset,
                                batch_size=2*batch_size,
                                shuffle=False,
                                num_workers=num_workers)

    elif dataset == "blobs":

        def _normalize(x, a: int = -1.0, b: int = 1.0):
            x_min = np.min(x, axis=0, keepdims=True)
            x_max = np.max(x, axis=0, keepdims=True)
            return (b - a) * (x - x_min) / (x_max - x_min) - a

        def get_blobs(n_classes: int, n_samples: int, n_features: int = 2, random_seed: int = 42):
            cluster_std = list()
            centers = list()

            m = int(math.sqrt(n_classes))
            n = 0
            for i in range(m):
                for j in range(m):
                    n += 1
                    centers.append((i, j))
                    cluster_std.append(0.3 * float(n) / float(m ** 2))

            x, y = make_blobs(n_samples=n_samples,
                              centers=centers,
                              cluster_std=cluster_std,
                              n_features=n_features,
                              random_state=random_seed)

            x = _normalize(x)

            x = torch.Tensor(x)
            y = torch.Tensor(y).type(torch.LongTensor)

            return x, y

        x, y = get_blobs(n_classes=64, n_samples=2000, random_seed=420)
        # Sanity check
        # import matplotlib.pyplot as plt
        # plt.scatter(x[:, 0], x[:, 1], c=y, s=1.0)
        # plt.show()
        trainset = torch.utils.data.TensorDataset(x, y)

        x, y = get_blobs(n_classes=64, n_samples=500, random_seed=69)
        testset = torch.utils.data.TensorDataset(x, y)

        trainloader = torch.utils.data.DataLoader(dataset=trainset,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  pin_memory=True,
                                                  num_workers=num_workers)

        testloader = torch.utils.data.DataLoader(dataset=testset,
                                                   batch_size=batch_size,
                                                   shuffle=False,
                                                   pin_memory=True,
                                                   num_workers=num_workers)

    return trainloader, testloader

