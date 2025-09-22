import numpy as np
import torch
from torch import nn

class SineLayer(nn.Module):
    """
    A single SIREN sine activation layer.
    """
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                b = np.sqrt(6 / self.in_features) / self.omega_0
                self.linear.weight.uniform_(-b, b)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))

class ResidualSineLayer(nn.Module):
    """
    A residual SIREN block with two sine-activated linear layers.
    """
    def __init__(self, features, bias=True, ave_first=False, ave_second=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.features = features
        self.l1 = nn.Linear(features, features, bias=bias)
        self.l2 = nn.Linear(features, features, bias=bias)
        self.w1 = 0.5 if ave_first else 1.0
        self.w2 = 0.5 if ave_second else 1.0
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            b = np.sqrt(6 / self.features) / self.omega_0
            self.l1.weight.uniform_(-b, b)
            self.l2.weight.uniform_(-b, b)

    def forward(self, x):
        s1 = torch.sin(self.omega_0 * self.l1(self.w1 * x))
        s2 = torch.sin(self.omega_0 * self.l2(s1))
        return self.w2 * (x + s2)

class ResidualSirenNet(nn.Module):
    """
    Residual SIREN network with multiple layers.
    """
    def __init__(self, num_hidden_layers=11, neurons_per_layer=120, omega_0=30):
        super().__init__()
        self.layers = nn.ModuleList()

        # First layer: from 3D coords to hidden features
        self.layers.append(SineLayer(3, neurons_per_layer, is_first=True, omega_0=omega_0))

        # Residual sine blocks
        for i in range(num_hidden_layers - 1):
            ave_first = i >= 1
            ave_second = i == (num_hidden_layers - 2)
            self.layers.append(
                ResidualSineLayer(neurons_per_layer,
                                  ave_first=ave_first,
                                  ave_second=ave_second,
                                  omega_0=omega_0)
            )

        # Final linear layer
        self.final = nn.Linear(neurons_per_layer, 1)
        with torch.no_grad():
            b = np.sqrt(6 / neurons_per_layer) / omega_0
            self.final.weight.uniform_(-b, b)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.final(x)
