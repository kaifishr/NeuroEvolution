import torch
import torch.nn as nn
# import torch.nn.functional as F


class MLP(nn.Module):

    def __init__(self, config: dict) -> None:
        super().__init__()

        self.dropout_rate = config["dropout_rate"]
        self.n_dims_hidden = config["n_dims_hidden"]
        self.n_layers_hidden = config["n_layers_hidden"]
        self._dataset = config["dataset"]

        if self._dataset == "blobs":
            self.IMAGE_WIDTH = 1
            self.IMAGE_HEIGHT = 1
            self.COLOR_CHANNELS = 2
            self.n_dims_out = 64
        elif self._dataset == "fashion_mnist":
            self.IMAGE_WIDTH = 28
            self.IMAGE_HEIGHT = 28
            self.COLOR_CHANNELS = 1
            self.n_dims_out = 10
        elif self._dataset == "cifar10":
            self.IMAGE_WIDTH = 32
            self.IMAGE_HEIGHT = 32
            self.COLOR_CHANNELS = 3
            self.n_dims_out = 10
        elif self._dataset == "cifar100":
            self.IMAGE_WIDTH = 32
            self.IMAGE_HEIGHT = 32
            self.COLOR_CHANNELS = 3
            self.n_dims_out = 100
        else:
            raise NotImplementedError(f"Dataset '{self._dataset}' not defined.")

        self.n_dims_in = self.IMAGE_HEIGHT * self.IMAGE_WIDTH * self.COLOR_CHANNELS

        self.layers = list()
        self._make_layers()

        self._init_parameters()

        self.classifier = nn.Sequential(*self.layers)

    def _make_layers(self) -> None:
        # Input layer
        self.layers.append(nn.Linear(self.n_dims_in, self.n_dims_hidden))
        self.layers.append(nn.Tanh())
        self.layers.append(nn.Dropout(p=self.dropout_rate))

        # Hidden layer
        for i in range(self.n_layers_hidden):
            self.layers.append(nn.Linear(self.n_dims_hidden, self.n_dims_hidden))
            self.layers.append(nn.Tanh())
            self.layers.append(nn.Dropout(p=self.dropout_rate))

        # Output layer
        self.layers.append(nn.Linear(self.n_dims_hidden, self.n_dims_out))

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