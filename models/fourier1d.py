import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from collections import defaultdict

sys.path.append("../")
from .basics import SpectralConv1d
from .utils import _get_act, add_padding, remove_padding


class FNN1d(nn.Module):
    def __init__(
        self,
        modes,
        width=32,
        layers=None,
        fc_dim=128,
        in_dim=2,
        out_dim=1,
        act="gelu",
        pad_ratio=0,
    ):
        super(FNN1d, self).__init__()

        """
        The overall network. It contains several layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """
        self.config = defaultdict(lambda: None, **config)
        self.config = dict(self.config)
        all_attr = list(self.config.keys())
        for key in all_attr:
            setattr(self, key, self.config[key])
        self.modes1 = self.FNO_modes

        self.modes1 = modes
        self.width = width
        if layers is None:
            layers = [width] * 4
        else:
            self.layers = layers
        self.pad_ratio = pad_ratio
        self.fc_dim = fc_dim

        self.fc0 = nn.Linear(in_dim, layers[0])  # input channel is 2: (a(x), x)

        self.sp_convs = nn.ModuleList(
            [
                SpectralConv1d(in_size, out_size, num_modes)
                for in_size, out_size, num_modes in zip(layers, layers[1:], self.modes1)
            ]
        )

        self.ws = nn.ModuleList(
            [
                nn.Conv1d(in_size, out_size, 1)
                for in_size, out_size in zip(layers, layers[1:])
            ]
        )

        # if fc_dim = 0, we do not have nonlinear layer
        if fc_dim > 0:
            self.fc1 = nn.Linear(layers[-1], fc_dim)
            self.fc2 = nn.Linear(fc_dim, out_dim)
        else:
            self.fc2 = nn.Linear(layers[-1], out_dim)

        self.act = _get_act(act)

    def forward(self, x):
        """
        Input shape (of x):     (batch, nx_in,  channels_in)
        Output shape:           (batch, nx_out, channels_out)

        The input resolution is determined by x.shape[-1]
        The output resolution is determined by self.s_outputspace
        """

        length = len(self.ws)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)
        pad_nums = [math.floor(self.pad_ratio * x.shape[-1])]

        # add padding
        x = add_padding(x, pad_nums=pad_nums)

        for i, (speconv, w) in enumerate(zip(self.sp_convs, self.ws)):
            x1 = speconv(x)
            x2 = w(x)
            x = x1 + x2
            if self.act is not None and i != length - 1:
                x = self.act(x)

        # remove padding
        x = remove_padding(x, pad_nums=pad_nums)

        x = x.permute(0, 2, 1)

        # if fc_dim = 0, we do not have nonlinear layer
        fc_dim = self.fc_dim if hasattr(self, "fc_dim") else 1

        if fc_dim > 0:
            x = self.fc1(x)
            if self.act is not None:
                x = self.act(x)

        x = self.fc2(x)

        return x
