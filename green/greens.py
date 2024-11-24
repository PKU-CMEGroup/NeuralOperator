import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from myutils.basics import compl_mul1d, _get_act
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict


@torch.jit.script
def mul_lowrank(U1, U2, M, x):
    x = torch.einsum("dmks,bidms->bidmk", U2, x)
    x = torch.einsum("dmkl,bidmk->bidml", M, x)
    x = torch.einsum("dmsl,bidml->bidms", U1, x)
    return x


class PointHierGreen(nn.Module):
    def __init__(self, modes, r_list):
        super(PointHierGreen, self).__init__()
        self.type = "hierarchicalGreen"

        self.modes = modes
        self.L = 3 + len(r_list)

        self.M = nn.ParameterList()
        self.U1 = nn.ParameterList()
        self.U2 = nn.ParameterList()

        for l, r in enumerate(r_list):
            l = l + 3
            num_blocks = self.modes // 2**l
            size_part = 2 ** (l - 1)
            self.U1.append(
                nn.Parameter(torch.rand(
                    2, num_blocks, size_part, r, dtype=torch.float))
            )
            self.M.append(
                nn.Parameter(
                    (2 * torch.rand(2, num_blocks, r, r, dtype=torch.float) - 1)
                )
            )
            self.U2.append(
                nn.Parameter(torch.rand(2, num_blocks, r,
                             size_part, dtype=torch.float))
            )

        self.diag = nn.Parameter(torch.ones(self.modes, dtype=torch.float))

    def forward(self, x):
        x_diag = x * self.diag
        x_add = torch.zeros_like(x)
        for l in range(3, self.L):
            num_blocks = self.modes // 2**l
            size_part = 2 ** (l - 1)
            x_blocks = x.reshape(*x.shape[:-1], 2, num_blocks, size_part)
            x_blocks = x_blocks[..., [1, 0], :, :]
            x_add += mul_lowrank(
                self.U1[l - 3], self.U2[l - 3], self.M[l - 3], x_blocks
            ).reshape(*x_add.shape[:-1], self.modes)
        return x_diag + x_add


class FcGreen(nn.Module):
    def __init__(self, modes):
        super(FcGreen, self).__init__()
        self.type = "fcGreen"

        self.w1 = nn.Parameter(torch.eye(modes, modes, dtype=torch.float))
        self.act = F.gelu
        self.w2 = nn.Parameter(torch.eye(modes, modes, dtype=torch.float))

    def forward(self, x):
        x = torch.einsum("bck,kl->bcl", x, self.w1)
        x = self.act(x)
        x = torch.einsum("bcl,lk->bck", x, self.w2)
        return x


class Conv1dGreen(nn.Module):
    def __init__(self, channels, kernel_size):
        super(Conv1dGreen, self).__init__()
        """
        similar to width=2
        """
        self.type = "convGreen"

        assert kernel_size % 2 == 1
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2

        self.conv = nn.Conv1d(
            channels, channels, kernel_size=self.kernel_size, padding=self.padding
        )

    def forward(self, x):
        return self.conv(x)


class GreenGalerkinConv(nn.Module):
    def __init__(self, in_channels, out_channels, modes, bases, wbases):
        super(GreenGalerkinConv, self).__init__()

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
        self.weights_channels = nn.Parameter(
            self.scale
            * torch.rand(in_channels, out_channels, self.modes, dtype=torch.float)
        )

    def forward(self, x, green=None):
        bases, wbases = self.bases, self.wbases

        # Compute coeffcients
        x_hat = torch.einsum("bcx,xk->bck", x, wbases)

        # Multiply relevant Fourier modes
        if green is not None:
            x_hat = green(x_hat)
        x_hat = compl_mul1d(x_hat, self.weights_channels)

        # Return to physical space
        x = torch.real(torch.einsum("bck,xk->bcx", x_hat, bases))

        return x


class GreenGkNO(nn.Module):
    def __init__(self, bases, **config):
        super(GreenGkNO, self).__init__()

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
        self.bases = bases

        self.config = defaultdict(lambda: None, **config)
        self.config = dict(self.config)
        all_attr = list(self.config.keys())
        for key in all_attr:
            setattr(self, key, self.config[key])

        self.fc0 = nn.Linear(self.in_dim, self.layer_channels[0])

        self.sp_layers = nn.ModuleList(
            [
                self._choose_layer(index, in_channels, out_channels, layer_type)
                for index, (in_channels, out_channels, layer_type) in enumerate(
                    zip(self.layer_channels,
                        self.layer_channels[1:], self.layer_types)
                )
            ]
        )
        self.greens = nn.ModuleList(
            [
                self._choose_green(channels, green_type)
                for channels, green_type in zip(self.layer_channels, self.green_types)
            ]
        )
        self.ws = nn.ModuleList(
            [
                nn.Conv1d(in_channels, out_channels, 1)
                for in_channels, out_channels in zip(
                    self.layer_channels, self.layer_channels[1:]
                )
            ]
        )
        self.length = len(self.ws)

        if self.fc_channels > 0:
            self.fc1 = nn.Linear(self.layer_channels[-1], self.fc_channels)
            self.fc2 = nn.Linear(self.fc_channels, self.out_dim)
        else:
            self.fc2 = nn.Linear(self.layer_channels[-1], self.out_dim)

        self.act = _get_act(self.act)

    def forward(self, x):
        """
        Input shape (of x):     (batch, nx_in,  channels_in)
        Output shape:           (batch, nx_out, channels_out)

        The input resolution is determined by x.shape[-1]
        The output resolution is determined by self.s_outputspace
        """

        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        for i, (layer, w) in enumerate(zip(self.sp_layers, self.ws)):
            x1 = layer(x, self.greens[i])
            x2 = w(x)
            res = x1 + x2

            # activation function
            if self.act is not None and i != self.length - 1:
                res = self.act(res)

            # skipping connection
            if self.should_residual[i] == True:
                x = x + res
            else:
                x = res

        x = x.permute(0, 2, 1)

        if self.fc_channels > 0:
            x = self.fc1(x)
            if self.act is not None:
                x = self.act(x)
        x = self.fc2(x)

        return x

    def _choose_layer(self, index, in_channels, out_channels, layer_type):
        if layer_type == "GalerkinConv":
            modes = self.modes_gk[index]
            bases = self.bases[self.bases_types[index]][0]
            wbases = self.bases[self.wbases_types[index]][1]
            return GreenGalerkinConv(in_channels, out_channels, modes, bases, wbases)
        else:
            raise KeyError("Layer Type Undefined.")

    def _choose_green(self, channels, green_type):
        if green_type == "hierGreen":
            return PointHierGreen(self.modes_gk[0], self.r_list)
        elif green_type == "fcGreen":
            return FcGreen(self.modes_gk[0])
        elif green_type == "convGreen":
            return Conv1dGreen(channels, self.kernel_size)
        elif green_type == "None":
            return None
        else:
            return KeyError("Green Function Undefined.")
