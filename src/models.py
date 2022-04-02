"""Module holds class definition of fully-connected neural network.

Class allows to build multilayer perceptron of arbitrary depth and width.

"""
from functools import reduce

import torch
from torch import nn


class MLP(nn.Module):
    """Fully connected neural network.

    Network uses Tanh activation functions to ensure that gradients exist.
    Very small networks with ReLU activation function might not learn at all.

    """

    def __init__(self, config: dict) -> None:
        super().__init__()

        self.config = config

        input_shape = config["input_shape"]
        self._n_dims_in = reduce(lambda x, y: x*y, input_shape)
        self._n_dims_out = config["n_classes"]

        self.layers = self._make_layers()
        self._init_parameters()

        self.classifier = nn.Sequential(*self.layers)

    def _make_layers(self) -> list:
        """Creates list with layers of network.

        Returns:
            List with network layers.

        """
        dropout_rate = self.config["hparam"]["dropout_rate"]["val"]
        n_dims_hidden = self.config["hparam"]["n_dims_hidden"]["val"]

        layers = []

        # Input layer
        layers.append(nn.Linear(self._n_dims_in, n_dims_hidden[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p=dropout_rate))

        # Hidden layer
        for n_dims_in, n_dims_out in zip(n_dims_hidden[:-1], n_dims_hidden[1:]):
            layers.append(nn.Linear(n_dims_in, n_dims_out))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_rate))

        # Output layer
        layers.append(nn.Linear(n_dims_hidden[-1], self._n_dims_out))

        return layers

    @torch.no_grad()
    def _init_parameters(self) -> None:
        """Initializes weights and biases.
        """
        # Standard Xavier initialization
        gain = nn.init.calculate_gain('relu')
        for layer in self.layers:
            if hasattr(layer, "weight"):
                torch.nn.init.xavier_uniform_(layer.weight.data, gain=gain)
            elif hasattr(layer, "bias"):
                layer.weight.data.fill_(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method.
        """
        x = x.view(-1, self._n_dims_in)
        x = self.classifier(x)
        return x
