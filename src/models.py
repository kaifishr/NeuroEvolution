"""Module holds class definition of fully-connected neural network.

Class allows to build multilayer perceptron of arbitrary depth and width.

"""
import torch
import torch.nn as nn
from functools import reduce


class MLP(nn.Module):
    """Fully connected neural network.

    Network uses Tanh activation functions to ensure that gradients exist.
    Very small networks with ReLU activation function might not learn at all.

    Attributes:
        ...
    """

    def __init__(self, config: dict) -> None:
        super().__init__()

        self._dropout_rate = config["hparam"]["dropout_rate"]["val"]
        self._n_dims_hidden = config["hparam"]["n_dims_hidden"]["val"]
        self._dataset = config["dataset"]
        self._input_shape = config["input_shape"]
        self._n_dims_out = config["n_classes"]

        self.n_dims_in = reduce(lambda x, y: x*y, self._input_shape)

        self.layers = list()
        self._make_layers()

        self._init_parameters()

        self.classifier = nn.Sequential(*self.layers)

    def _make_layers(self) -> None:
        # Input layer
        self.layers.append(nn.Linear(self.n_dims_in, self._n_dims_hidden[0]))
        self.layers.append(nn.Tanh())
        self.layers.append(nn.Dropout(p=self._dropout_rate))

        # Hidden layer
        for n_dims_in, n_dims_out in zip(self._n_dims_hidden[:-1], self._n_dims_hidden[1:]):
            self.layers.append(nn.Linear(n_dims_in, n_dims_out))
            self.layers.append(nn.Tanh())
            self.layers.append(nn.Dropout(p=self._dropout_rate))

        # Output layer
        self.layers.append(nn.Linear(self._n_dims_hidden[-1], self._n_dims_out))

    @torch.no_grad()
    def _init_parameters(self) -> None:
        # Standard Xavier initialization
        gain = nn.init.calculate_gain('tanh')
        for layer in self.layers:
            if hasattr(layer, "weight"):
                torch.nn.init.xavier_uniform_(layer.weight.data, gain=gain)
            elif hasattr(layer, "bias"):
                layer.weight.data.fill_(0.0)

    def forward(self, x):

        x = x.view(-1, self.n_dims_in)
        x = self.classifier(x)

        return x
