import torch
import torch.nn as nn
from collections import defaultdict
from myutils.basics import (
    compl_mul1d,
    _get_act,
)  #  SpectralConv1d, SpectralConv2d_shape,
import time
import sys


# class GalerkinConv(nn.Module):
#     def __init__(self, in_channels, out_channels, modes, bases, wbases):
#         super(GalerkinConv, self).__init__()

#         """
#         1D Spectral layer. It avoids FFT, but utilizes low rank approximation.

#         """

#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         # Number of Fourier modes to multiply, at most floor(N/2) + 1
#         self.modes = modes
#         self.bases = bases
#         self.wbases = wbases

#         self.scale = 1 / (in_channels * out_channels)
#         self.weights = nn.Parameter(
#             self.scale
#             * torch.rand(in_channels, out_channels, self.modes, dtype=torch.float)
#         )

#     def forward(self, x):
#         bases, wbases = self.bases, self.wbases

#         self.print_memory_usage("1")
#         # Compute coeffcients
#         start_time = time.time()
#         x_hat = torch.einsum("bcx,xk->bck", x, wbases)
#         end_time = time.time()
#         print(f"coeff{end_time-start_time}")

#         self.print_memory_usage("2")
#         # Multiply relevant Fourier modes
#         start_time = time.time()
#         x_hat = compl_mul1d(x_hat, self.weights)
#         end_time = time.time()

#         print(f"green{end_time-start_time}")
#         self.print_memory_usage("3")
#         # Return to physical space
#         start_time = time.time()
#         x = torch.real(torch.einsum("bck,xk->bcx", x_hat, bases))
#         end_time = time.time()
#         print(f"back{end_time-start_time}")

#         # sys.exit()
#         return x

#     def print_memory_usage(self, message):
#         allocated = torch.cuda.memory_allocated()
#         reserved = torch.cuda.memory_reserved()
#         print(
#             f"{message}: Allocated = {allocated / 1024**2:.2f} MB, Reserved = {reserved / 1024**2:.2f} MB"
#         )


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


class GkNO(nn.Module):
    def __init__(self, bases_list, **config):
        super(GkNO, self).__init__()

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
        self.bases_fourier = bases_list[0]
        self.wbases_fourier = bases_list[1]
        self.bases_pca = bases_list[2]
        self.wbases_pca = bases_list[3]

        self.config = defaultdict(lambda: None, **config)
        self.config = dict(self.config)
        all_attr = list(self.config.keys())
        for key in all_attr:
            setattr(self, key, self.config[key])

        self.fc0 = nn.Linear(
            self.in_dim, self.layers_dim[0]
        )  # input channel is 2: (a(x), x)

        self.sp_layers = nn.ModuleList(
            [
                self._choose_layer(index, in_size, out_size, layer_type)
                for index, (in_size, out_size, layer_type) in enumerate(
                    zip(self.layers_dim, self.layers_dim[1:], self.layer_types)
                )
            ]
        )
        #######
        self.ws = nn.ModuleList(
            [
                nn.Conv1d(in_size, out_size, 1)
                for in_size, out_size in zip(self.layers_dim, self.layers_dim[1:])
            ]
        )

        # if fc_dim = 0, we do not have nonlinear layer
        if self.fc_dim > 0:
            self.fc1 = nn.Linear(self.layers_dim[-1], self.fc_dim)
            self.fc2 = nn.Linear(self.fc_dim, self.out_dim)
        else:
            self.fc2 = nn.Linear(self.layers_dim[-1], self.out_dim)

        self.act = _get_act(self.act)

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

        # # add padding
        # if self.pad_ratio > 0:
        #     pad_nums = [math.floor(self.pad_ratio * x.shape[-1])]
        #     x = add_padding(x, pad_nums=pad_nums)

        for i, (layer, w) in enumerate(zip(self.sp_layers, self.ws)):
            x1 = layer(x)
            x2 = w(x)
            res = x1 + x2
            if self.act is not None and i != length - 1:
                res = self.act(res)
            if self.residual[i] == True:
                x = x + res
            else:
                x = res

        # if self.pad_ratio > 0:
        #     x = remove_padding(x, pad_nums=pad_nums)

        x = x.permute(0, 2, 1)

        # if fc_dim = 0, we do not have nonlinear layer
        fc_dim = self.fc_dim

        if fc_dim > 0:
            x = self.fc1(x)
            if self.act is not None:
                x = self.act(x)

        x = self.fc2(x)

        return x

    def _choose_layer(self, index, in_channels, out_channels, layer_type):
        if layer_type == "GalerkinConv_fourier":
            num_modes = self.GkNN_modes[index]
            bases = self.bases_fourier
            wbases = self.wbases_fourier
            return GalerkinConv(in_channels, out_channels, num_modes, bases, wbases)
        elif layer_type == "GalerkinConv_pca":
            num_modes = self.GkNN_modes[index]
            bases = self.bases_pca
            wbases = self.wbases_pca
            return GalerkinConv(in_channels, out_channels, num_modes, bases, wbases)

        else:
            raise KeyError("Layer Type Undefined.")
