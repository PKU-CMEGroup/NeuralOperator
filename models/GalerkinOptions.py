import torch
import torch.nn as nn
from collections import defaultdict

from .basics import (
    compl_mul1d,
    compl_mul1d_matrix,
    SpectralConv1d,
    SpectralConv2d_shape,
    SimpleAttention,
)
from .utils import _get_act


class GalerkinConvOptions(nn.Module):

    def __init__(
        self, in_channels, out_channels, modes, bases, wbases, diag_width, is_adjust
    ):
        super(GalerkinConvOptions, self).__init__()

        """
        1D Spectral layer. It avoids FFT, but utilizes low rank approximation. 
        Options:
            1. Adjust bases by learning a linear transformation W.
            2. Set the width of the weights to enable interaction between coefficients of different modes.
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.bases = bases[:, : self.modes]
        self.wbases = wbases[:, : self.modes]

        self.is_adjust = is_adjust
        if is_adjust:
            # Initialize W as an identity matrix
            self.W_hidden_bases = nn.Parameter(torch.eye(self.modes, dtype=torch.float))
            self.W_hidden_wbases = nn.Parameter(
                torch.eye(self.modes, dtype=torch.float)
            )

        self.diag_width = diag_width
        self.scale = 1 / (in_channels * out_channels)
        if self.diag_width > 0:
            # The main diagonal
            self.weights_list = nn.ParameterList(
                [
                    nn.Parameter(
                        self.scale
                        * torch.rand(
                            in_channels, out_channels, self.modes, dtype=torch.float
                        )
                    )
                ]
            )
            for i in range(self.diag_width - 1):
                # The i-th and the -i-th diagonal
                self.weights_list.append(
                    nn.Parameter(
                        self.scale
                        * torch.rand(
                            in_channels,
                            out_channels,
                            self.modes - i - 1,
                            dtype=torch.float,
                        )
                    )
                )
                self.weights_list.append(
                    nn.Parameter(
                        self.scale
                        * torch.rand(
                            in_channels,
                            out_channels,
                            self.modes - i - 1,
                            dtype=torch.float,
                        )
                    )
                )
        elif self.diag_width == -1:
            # Using the full rank matrix
            self.weights_matrix = nn.Parameter(
                self.scale
                * torch.rand(
                    in_channels, out_channels, self.modes, self.modes, dtype=torch.float
                )
            )
        else:
            raise ValueError("Value Of Diagonal Width Is Invalid!")

    def forward(self, x):
        bases, wbases = self.bases, self.wbases

        # Adjust bases
        if self.is_adjust:
            bases = torch.einsum("xk,kl->xl", bases, self.W_hidden_bases)
            wbases = torch.einsum("xk,kl->xl", wbases, self.W_hidden_wbases)

        # Compute coeffcients
        x_co = torch.einsum("bcx,xk->bck", x, wbases)

        # Multiply relevant modes
        if self.diag_width >= 0:
            x_hat = compl_mul1d(x_co, self.weights_list[0])
            for i in range(self.diag_width - 1):
                x1 = compl_mul1d(x_hat[..., i + 1 :], self.weights_list[2 * i + 1])
                x2 = compl_mul1d(x_hat[..., : -(i + 1)], self.weights_list[2 * i + 2])
                x_hat_clone = x_hat.clone()
                x_hat_clone[..., : -(i + 1)] += x1
                x_hat_clone[..., (i + 1) :] += x2
                x_hat = x_hat_clone
        elif self.diag_width == -1:
            x_hat = compl_mul1d_matrix(x_co, self.weights_matrix)

        # Return to physical space
        x = torch.real(torch.einsum("bck,xk->bcx", x_hat, bases))

        return x


class GkNNOptions(nn.Module):
    def __init__(self, bases_list, **config):
        super(GkNNOptions, self).__init__()
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
        Options:
            1. Type of spectral convolution layer.
            2. Whether to include the f==1 in the input.
            3. Nomalizations
            4. Dropouts
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

        # input channel is 2: (a(x), x) or 3: (F(x), a(x), x)
        if self.is_includeF:
            self.in_dim = self.in_dim + 1
        self.fc0 = nn.Linear(self.in_dim, self.layers_dim[0])

        self.sp_layers = nn.ModuleList(
            [
                self._choose_layer(index, in_size, out_size, layer_type)
                for index, (in_size, out_size, layer_type) in enumerate(
                    zip(self.layers_dim, self.layers_dim[1:], self.layer_types)
                )
            ]
        )
        self.ws = nn.ModuleList(
            [
                nn.Conv1d(in_size, out_size, 1)
                for in_size, out_size in zip(self.layers_dim, self.layers_dim[1:])
            ]
        )
        self.length = len(self.ws)

        # if fc_dim = 0, we do not have nonlinear layer
        if self.fc_dim > 0:
            self.fc1 = nn.Linear(self.layers_dim[-1], self.fc_dim)
            self.fc2 = nn.Linear(self.fc_dim, self.out_dim)
        else:
            self.fc2 = nn.Linear(self.layers_dim[-1], self.out_dim)

        self.act = _get_act(self.act)

    def forward(self, x):

        if self.is_includeF:
            F = torch.ones(x.shape[:-1] + (1,), device=x.device)
            x = torch.cat((F, x), dim=-1)

        x = self.fc0(x)

        x = x.permute(0, 2, 1)

        for i, (layer, w) in enumerate(zip(self.sp_layers, self.ws)):
            x1 = layer(x)
            x2 = w(x)
            res = x1 + x2

            # activation function
            if self.act is not None and i != self.length - 1:
                res = self.act(res)

            # skipping connection
            if self.residual[i] == True:
                x = x + res
            else:
                x = res

        x = x.permute(0, 2, 1)

        if self.fc_dim > 0:
            x = self.fc1(x)
            if self.act is not None:
                x = self.act(x)
        x = self.fc2(x)

        return x

    def _choose_layer(self, index, in_channels, out_channels, layer_type):
        if layer_type in ["GalerkinConv_fourier", "GalerkinConv_pca"]:
            num_modes = self.GkNN_modes[index]
            diag_width = self.diag_width_list[index]
            is_adjust = self.is_adjust_list[index]
            if layer_type == "GalerkinConv_fourier":
                bases = self.bases_fourier
                wbases = self.wbases_fourier
            elif layer_type == "GalerkinConv_pca":
                bases = self.bases_pca
                wbases = self.wbases_pca
            return GalerkinConvOptions(
                in_channels,
                out_channels,
                num_modes,
                bases,
                wbases,
                diag_width,
                is_adjust,
            )

        elif layer_type == "FourierConv1d":
            num_modes = self.FNO_modes[index]
            return SpectralConv1d(in_channels, out_channels, num_modes)

        elif layer_type == "FourierConv2d":
            num_modes1 = self.FNO_modes[index]
            num_modes2 = self.FNO_modes[index]
            return SpectralConv2d_shape(
                in_channels, out_channels, num_modes1, num_modes2
            )

        elif layer_type == "SimpleAttention":
            num_heads = self.num_heads[index]
            attention_type = self.attention_types[index]
            return SimpleAttention(in_channels, out_channels, num_heads, attention_type)
        else:
            raise ValueError("Layer Type Undefined.")
