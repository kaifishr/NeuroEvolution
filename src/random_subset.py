"""Random subset dataset.
"""
import random

import numpy as np
import torch

from torch.utils.data import Dataset
from PIL import Image


class RandomSubset(Dataset):
    """Class allows to iterate every epoch through a different random subset of the original
    dataset.

    The intention behind this class is to speed up genetic optimization by using only a small subset
    of the original dataset every epoch. The subset is randomly created every epoch from the
    original dataset. Therefore, all samples from the original dataset are used at some point during
    the training.

    """

    def __init__(self, dataset: Dataset, subset_ratio: float) -> None:
        """

        Args:
            dataset: PyTorch dataset.
            subset_ratio: Defines size of subset.

        """

        super().__init__()

        self.dataset = dataset

        if isinstance(dataset.data, np.ndarray):
            self.data = dataset.data
        elif isinstance(dataset.data, list):
            self.data = np.array(dataset.data)
        elif isinstance(dataset.data, torch.Tensor):
            self.data = dataset.data.numpy()
        else:
            raise TypeError(f"Targets must be of type 'list' or 'tensor', "
                            f"but got {type(dataset.data)}.")

        if isinstance(dataset.targets, np.ndarray):
            self.data = dataset.data
        elif isinstance(dataset.targets, list):
            self.targets = np.array(dataset.targets)
        elif isinstance(dataset.targets, torch.Tensor):
            self.targets = dataset.targets.numpy()
        else:
            raise TypeError(f"Targets must be of type 'list' or 'tensor', "
                            f"but got {type(dataset.targets)}.")

        self.subset_length = int(len(self.data) * subset_ratio)

        self.counter = 0
        self._random_subset()

    def _random_subset(self) -> None:
        """Creates random mappings.
        """
        self.rand_map = random.sample(list(range(len(self.data))), self.subset_length)

    def __len__(self) -> int:
        return self.subset_length

    def __getitem__(self, index: int) -> tuple:

        self.counter += 1
        if self.counter > self.subset_length:
            self._random_subset()
            self.counter = 0

        rand_index = self.rand_map[index]
        img, target = self.data[rand_index], int(self.targets[rand_index])

        # Cast to PIL Image required for transformations.
        if len(img.shape) == 2:
            img = Image.fromarray(img, mode="L")
        elif (len(img.shape) == 3) and (img.shape[-1] == 3):
            img = Image.fromarray(img, mode="RGB")

        if self.dataset.transform is not None:
            img = self.dataset.transform(img)

        return img, target
