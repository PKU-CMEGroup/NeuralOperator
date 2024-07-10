import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .basics import compl_mul1d
from .utils import _get_act, add_padding, remove_padding
from .basics import SpectralConv1d, SimpleAttention


class GalerkinConv(nn.Module):
    def __init__(self, in_channels, out_channels, modes, bases, wbases):
        super(GalerkinConv, self).__init__()

        """
        1D Spectral layer. It avoids FFT, but utilizes low rank approximation. 
        
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes = modes
        self.bases = bases
        self.wbases = wbases

        self.scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            self.scale
            * torch.rand(in_channels, out_channels, self.modes, dtype=torch.float)
        )

    def forward(self, x):
        bases, wbases = self.bases, self.wbases
        # Compute coeffcients

        x_hat = torch.einsum("bcx,xk->bck", x, wbases)

        # Multiply relevant Fourier modes
        x_hat = compl_mul1d(x_hat, self.weights)

        # Return to physical space
        x = torch.real(torch.einsum("bck,xk->bcx", x_hat, bases))

        return x


class GkNN(nn.Module):
    def __init__(
        self,
        pad_ratio=-1,
        layer_configs=None,
        layers=None,
        fc_dim=128,
        in_dim=2,
        out_dim=1,
        act="gelu",
    ):
        super(GkNN, self).__init__()

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

        self.fc0 = nn.Linear(in_dim, layers[0])  # input channel is 2: (a(x), x)
        self.pad_ratio = pad_ratio

        self.layer_configs = layer_configs
        self.sp_layers = nn.ModuleList(
            [
                self._choose_layer(in_size, out_size, layer_config)
                for in_size, out_size, layer_config in zip(
                    layers, layers[1:], self.layer_configs
                )
            ]
        )

        self.ws = nn.ModuleList(
            [
                nn.Conv1d(in_size, out_size, 1)
                for in_size, out_size in zip(layers, layers[1:])
            ]
        )

        self.fc_dim = fc_dim
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

        # add padding
        if self.pad_ratio > 0:
            pad_nums = [math.floor(self.pad_ratio * x.shape[-1])]
            x = add_padding(x, pad_nums=pad_nums)

        for i, (layer, w) in enumerate(zip(self.sp_layers, self.ws)):
            x1 = layer(x)
            x2 = w(x)
            x = x1 + x2
            if self.act is not None and i != length - 1:
                x = self.act(x)
                
        if self.pad_ratio>0:
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

    @staticmethod
    def _choose_layer(in_channels, out_channels, layer_config):
        type = layer_config["type"]
        if type == "GalerkinConv":
            num_modes = layer_config["num_modes"]
            bases = layer_config["bases"]
            wbases = layer_config["wbases"]
            return GalerkinConv(in_channels, out_channels, num_modes, bases, wbases)
        elif type == "FourierConv":
            num_modes = layer_config["num_modes"]
            return SpectralConv1d(in_channels, out_channels, num_modes)
        elif type == "Attention":
            num_heads = layer_config["num_heads"]
            attention_type = layer_config["attention_type"]
            return SimpleAttention(in_channels, out_channels, num_heads, attention_type)
        else:
            raise ValueError("Layer Type Undefined.")
