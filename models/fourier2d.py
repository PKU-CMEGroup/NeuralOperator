import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .basics import SpectralConv2d
from .utils import _get_act, add_padding, remove_padding


class FNN2d(nn.Module):
    def __init__(
        self,
        modes1,
        modes2,
        width=64,
        layers=None,
        fc_dim=128,
        in_dim=3,
        out_dim=1,
        act="gelu",
        pad_ratio=0,
    ):
        super(FNN2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        # input channel is 3: (a(x, y), x, y)
        if layers is None:
            self.layers = [width] * 4
        else:
            self.layers = layers
        self.pad_ratio = pad_ratio
        self.fc_dim = fc_dim

        self.fc0 = nn.Linear(in_dim, layers[0])

        self.sp_convs = nn.ModuleList(
            [
                SpectralConv2d(in_size, out_size, mode1_num, mode2_num)
                for in_size, out_size, mode1_num, mode2_num in zip(
                    self.layers, self.layers[1:], self.modes1, self.modes2
                )
            ]
        )

        self.ws = nn.ModuleList(
            [
                nn.Conv1d(in_size, out_size, 1)
                for in_size, out_size in zip(self.layers, self.layers[1:])
            ]
        )

        if fc_dim > 0:
            self.fc1 = nn.Linear(layers[-1], fc_dim)
            self.fc2 = nn.Linear(fc_dim, out_dim)
        else:
            self.fc2 = nn.Linear(layers[-1], out_dim)

        self.act = _get_act(act)

    def forward(self, x):
        """
        Args:
            - x : (batch size, x_grid, y_grid, 2)
        Returns:
            - x: (batch size, x_grid, y_grid, 1)
        """
        length = len(self.ws)
        batchsize = x.shape[0]

        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        pad_nums = [
            math.floor(self.pad_ratio * x.shape[-2]),
            math.floor(self.pad_ratio * x.shape[-1]),
        ]
        x = add_padding(x, pad_nums=pad_nums)

        size_x, size_y = x.shape[-2], x.shape[-1]

        for i, (speconv, w) in enumerate(zip(self.sp_convs, self.ws)):
            x1 = speconv(x)
            x2 = w(x.view(batchsize, self.layers[i], -1)).view(
                batchsize, self.layers[i + 1], size_x, size_y
            )
            x = x1 + x2
            if self.act is not None and i != length - 1:
                x = self.act(x)

        x = remove_padding(x, pad_nums=pad_nums)

        x = x.permute(0, 2, 3, 1)

        if self.fc_dim > 0:
            x = self.fc1(x)
            if self.act is not None:
                x = self.act(x)

        x = self.fc2(x)
        return x
