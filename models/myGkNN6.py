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


        
def mycompl_mul1d(weights, H , x_hat):
    x_hat1 = torch.einsum('jkl,bil -> bijk', H , x_hat)
    y = torch.einsum('ioj,bijk -> bok', weights , x_hat1)
    return y

# def compute_H_vector(vector,wbases,grid,modes):

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

class HGalerkinConv(nn.Module):
    def __init__(self, in_channels, out_channels, modes, kernel_modes,  bases, wbases, H):
        super(HGalerkinConv, self).__init__()

        """
        1D Spectral layer. It avoids FFT, but utilizes low rank approximation. 
        
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes = modes
        self.bases = bases
        self.wbases = wbases
        self.H = H
        self.dtype = H.dtype

        self.scale = 1 / (in_channels * out_channels)
        self.kernel_modes = kernel_modes

        self.weights = nn.Parameter(
            self.scale
            * torch.randn(in_channels, out_channels, self.kernel_modes, dtype=self.dtype)
        )
        
        
        

    def forward(self, x):
        bases, wbases = self.bases, self.wbases

        # Compute coeffcients

        x_hat = torch.einsum("bcx,xk->bck", x, wbases)
        

        # Multiply relevant Fourier modes
        x_hat = x_hat.to(dtype=self.dtype)
        x_hat = mycompl_mul1d(self.weights, self.H , x_hat)
        x_hat = x_hat.real

        # Return to physical space
        x = torch.real(torch.einsum("bck,xk->bcx", x_hat, bases))

        return x


class myGkNN6(nn.Module):
    def __init__(self, bases_list ,H ,**config):
        super(myGkNN6, self).__init__()

        self.bases_fourier=bases_list[0]
        self.wbases_fourier=bases_list[1]
        self.bases_pca=bases_list[2]
        self.wbases_pca=bases_list[3]

        
        
        self.config = defaultdict(lambda: None, **config)
        self.config = dict(self.config)
        all_attr = list(self.config.keys())
        for key in all_attr:
            setattr(self, key, self.config[key])
        if self.learning_H == 'complex':
            self.scale = 1/(self.GkNN_modes[0]*self.GkNN_modes[0])
            self.H = nn.Parameter(
                self.scale
                * torch.rand(self.kernel_modes[0], self.GkNN_modes[0], self.GkNN_modes[0], dtype=torch.complex64)
            )
        elif self.learning_H == 'real':
            self.scale = 1/(self.GkNN_modes[0]*self.GkNN_modes[0])
            self.H = nn.Parameter(
                self.scale
                * torch.rand(self.kernel_modes[0], self.GkNN_modes[0], self.GkNN_modes[0], dtype=torch.float)
            )
        else:
            self.H = H

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
        if self.layer_types_plus:
            self.sp_layers_plus = nn.ModuleList(
                [
                    self._choose_layer(index, in_size, out_size, layer_type)
                    for index, (in_size, out_size, layer_type) in enumerate(
                        zip(self.layers_dim, self.layers_dim[1:], self.layer_types_plus)
                    )
                ]
            )
        else:
            self.sp_layers_plus = self.sp_layers
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
        self.dropout_layers = nn.ModuleList(
            [nn.Dropout(p=dropout)
             for dropout in self.dropout]
        )
        

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

        for i, (layer, layer_plus , w, dplayer) in enumerate(zip(self.sp_layers,self.sp_layers_plus, self.ws, self.dropout_layers)):
            x1 = layer(x)
            if self.layer_types_plus:
                x_plus = layer_plus(x)
            else:
                x_plus = 0
            x2 = w(x)
            res = x1 + x_plus + x2
            res = dplayer(res)
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
        elif layer_type == "HGalerkinConv_pca":
            num_modes = self.GkNN_modes[index]
            kernel_modes = self.kernel_modes[index]
            bases = self.bases_pca
            wbases = self.wbases_pca
            H = self.H
            return HGalerkinConv(in_channels, out_channels, num_modes, kernel_modes, bases, wbases,H)
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