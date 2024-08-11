import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import defaultdict
import sys

sys.path.append("../")
from .basics import (
    compl_mul1d,
    SpectralConv1d,
    SpectralConv2d_shape,
    SimpleAttention,
)
from .utils import _get_act, add_padding, remove_padding


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

class myGalerkinConv(nn.Module):
    def __init__(self, in_channels, out_channels, modes, bases_in, wbases_in, bases_out, wbases_out):
        super(myGalerkinConv, self).__init__()

        """
        1D Spectral layer. It avoids FFT, but utilizes low rank approximation. 
        
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes = modes
        self.bases_in = bases_in
        self.wbases_in = wbases_in
        self.bases_out = bases_out
        self.wbases_out = wbases_out

        self.scale = 1 / (in_channels * out_channels)
        self.weights_in = nn.Parameter(
            self.scale
            * torch.rand(in_channels, out_channels, self.modes, dtype=torch.float)
        )
        self.weights_out = nn.Parameter(
            self.scale
            * torch.rand(in_channels, out_channels, self.modes, dtype=torch.float)
        )

    def forward(self, x):
        bases_in, wbases_in, bases_out, wbases_out = self.bases_in, self.wbases_in, self.bases_out, self.wbases_out
        
        # Compute coeffcients

        x_hat_in = torch.einsum("bcx,xk->bck", x, wbases_in)

        # Multiply relevant Fourier modes
        x_hat_in = compl_mul1d(x_hat_in, self.weights_in)

        # Return to physical space
        x_in = torch.real(torch.einsum("bck,xk->bcx", x_hat_in, bases_in))

        # Compute coeffcients

        x_hat_out = torch.einsum("bcx,xk->bck", x, wbases_out)

        # Multiply relevant Fourier modes
        x_hat_out = compl_mul1d(x_hat_out, self.weights_out)

        # Return to physical space
        x_out = torch.real(torch.einsum("bck,xk->bcx", x_hat_out, bases_out))

        x = x_in + x_out

        return x

class myGkNN3(nn.Module):
    def __init__(self, bases_list,**config):
        super(myGkNN3, self).__init__()

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
        self.bases_fourier=bases_list[0]
        self.wbases_fourier=bases_list[1]
        self.bases_pca_in=bases_list[2]
        self.wbases_pca_in=bases_list[3]
        self.bases_pca_out=bases_list[4]
        self.wbases_pca_out=bases_list[5]
        
        self.config = defaultdict(lambda: None, **config)
        self.config = dict(self.config)
        all_attr = list(self.config.keys())
        for key in all_attr:
            setattr(self, key, self.config[key])

        if not self.input_fc:
            self.extra_dim = [self.in_dim]*len(self.layer_types)
        self.fc0 = nn.Linear(
            self.in_dim, self.layers_dim[0]
        )  # input channel is 2: (a(x), x)

        self.sp_layers = nn.ModuleList(
            [
                self._choose_layer(index, in_size + extra_size, out_size, layer_type)
                for index, (extra_size,in_size, out_size, layer_type) in enumerate(
                    zip(self.extra_dim, self.layers_dim, self.layers_dim[1:], self.layer_types)
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
        if self.input_fc:
            self.extra_fcs = nn.ModuleList([nn.Linear(self.in_dim,dim) for dim in self.extra_dim])

    def forward(self, x):
        """
        Input shape (of x):     (batch, nx_in,  channels_in)
        Output shape:           (batch, nx_out, channels_out)

        The input resolution is determined by x.shape[-1]
        The output resolution is determined by self.s_outputspace
        """

        length = len(self.ws)
        x_extra = x[:,:,:self.in_dim]


        x = x[:,:,1:]

        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        # # add padding
        # if self.pad_ratio > 0:
        #     pad_nums = [math.floor(self.pad_ratio * x.shape[-1])]
        #     x = add_padding(x, pad_nums=pad_nums)

        for i, (layer, w) in enumerate(zip(self.sp_layers, self.ws)):
            if self.input_fc:
                x_extra_i = self.extra_fcs[i](x_extra)
                x_extra_i = x_extra_i.permute(0, 2, 1)
                x0 = torch.cat((x,x_extra_i),dim=1)
            else:
                x_extra_i = x_extra.permute(0, 2, 1)
                x0 = torch.cat((x,x_extra_i),dim=1) 
            x1 = layer(x0)
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
        elif layer_type == "myGalerkinConv_pca":
            num_modes = self.GkNN_modes[index]
            bases_in = self.bases_pca_in
            wbases_in = self.wbases_pca_in
            bases_out = self.bases_pca_out
            wbases_out = self.wbases_pca_out
            return myGalerkinConv(in_channels, out_channels, num_modes, bases_in, wbases_in, bases_out, wbases_out)
        elif layer_type == "FourierConv1d":
            num_modes = self.FNO_modes[index]
            return SpectralConv1d(in_channels, out_channels, num_modes)
        elif layer_type == "FourierConv2d":
            num_modes1 = self.FNO_modes[index]
            num_modes2 = self.FNO_modes[index]
            return SpectralConv2d_shape(
                in_channels, out_channels, num_modes1, num_modes2)
        elif layer_type == "Attention":
            num_heads = self.num_heads[index]
            attention_type = self.attention_types[index]
            return SimpleAttention(in_channels, out_channels, num_heads, attention_type)
        else:
            raise ValueError("Layer Type Undefined.")