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
)
from .utils import _get_act, add_padding, remove_padding


        
def mycompl_mul1d_D(weights, D , x_hat):
    x_hat_expanded = x_hat.unsqueeze(2)  # shape: (bsz, input channel, 1, modes)
    D_expanded = D.unsqueeze(0).unsqueeze(0)  # shape: (1, 1, kernel_modes, modes)
    x_hat1 = x_hat_expanded * D_expanded  # shape: (bsz, input channel, kernel_modes, modes)
    y = torch.einsum('bijk,ioj -> bok', x_hat1 , weights )
    return y


class DGalerkinConv(nn.Module):
    def __init__(self, in_channels, out_channels, modes, kernel_modes,  bases, wbases, D):
        super(DGalerkinConv, self).__init__()

        """
        1D Spectral layer. It avoids FFT, but utilizes low rank approximation. 
        
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes = modes
        self.bases = bases
        self.wbases = wbases
        self.scale1 = 1 / kernel_modes
        self.D = D
            

        self.scale2 = 1 / (in_channels * out_channels)
        self.kernel_modes = kernel_modes

        self.weights = nn.Parameter(
            self.scale2
            * torch.randn(in_channels, out_channels, self.kernel_modes, dtype=torch.float)
        )
                
    def forward(self, x):
        bases, wbases = self.bases, self.wbases

        # Compute coeffcients

        x_hat = torch.einsum("bcx,xk->bck", x, wbases)
        

        # Multiply relevant Fourier modes
        x_hat = mycompl_mul1d_D(self.weights, self.D , x_hat)

        # Return to physical space
        x = torch.einsum("bck,xk->bcx", x_hat, bases)

        return x
 
class DGalerkinConv_double(nn.Module):
    def __init__(self, in_channels, out_channels, modes, kernel_modes,  bases_in, wbases_in, bases_out, wbases_out, D_in , D_out):
        super(DGalerkinConv_double, self).__init__()

        """
        1D Spectral layer. It avoids FFT, but utilizes low rank approximation. 
        
        """

        self.layer_in = DGalerkinConv(in_channels, out_channels, modes, kernel_modes,  bases_in, wbases_in, D_in)
        self.layer_out = DGalerkinConv(in_channels, out_channels, modes, kernel_modes,  bases_out, wbases_out, D_out)
        
        
    def forward(self, x):

        x_in = self.layer_in(x)
        x_out = self.layer_out(x)
        x = x_in + x_out

        return x



class DGalerkin(nn.Module):
    def __init__(self, bases_list  ,**config):
        super(DGalerkin, self).__init__()

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
        if self.D_type == 'same':
            self.scale = 1/self.GkNN_modes[0]
            self.D_in = nn.Parameter(
                self.scale
                * torch.rand(self.kernel_modes[0], self.GkNN_modes[0], dtype=torch.float)
            )
            self.D_out = nn.Parameter(
                self.scale
                * torch.rand(self.kernel_modes[0], self.GkNN_modes[0], dtype=torch.float)
            )
        elif self.D_type == 'different':
            self.D_in = False
            self.D_out = False
        else:
            raise ValueError("H_Type Undefined.")

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


        for i, (layer , w, dplayer) in enumerate(zip(self.sp_layers, self.ws, self.dropout_layers)):
            x1 = layer(x)
            x2 = w(x)
            x = x1 + x2
            x = dplayer(x)
            if self.act is not None and i != length - 1:
                x = self.act(x)



        x = x.permute(0, 2, 1)

        fc_dim = self.fc_dim 
        
        if fc_dim > 0:
            x = self.fc1(x)
            if self.act is not None:
                x = self.act(x)

        x = self.fc2(x)

        return x

    def _choose_layer(self, index, in_channels, out_channels, layer_type):
        if layer_type == "DGalerkinConv_pca":
            num_modes = self.GkNN_modes[index]
            kernel_modes = self.kernel_modes[index]
            bases = self.bases_pca_out
            wbases = self.wbases_pca_out
            D = self.D_out
            return DGalerkinConv(in_channels, out_channels, num_modes, kernel_modes, bases, wbases, D)
        elif layer_type == "DGalerkinConv_doublepca":
            num_modes = self.GkNN_modes[index]
            kernel_modes = self.kernel_modes[index]
            bases_in = self.bases_pca_in
            wbases_in = self.wbases_pca_in
            bases_out = self.bases_pca_out
            wbases_out = self.wbases_pca_out
            D_in = self.D_in
            D_out = self.D_out
            return DGalerkinConv_double(in_channels, out_channels, num_modes, kernel_modes, bases_in, wbases_in, bases_out, wbases_out, D_in, D_out)
        elif layer_type == "DGalerkinConv_fourier":
            num_modes = self.GkNN_modes[index]
            kernel_modes = self.kernel_modes[index]
            bases = self.bases_fourier
            wbases = self.wbases_fourier
            D = self.D_out
            return DGalerkinConv(in_channels, out_channels, num_modes, kernel_modes, bases, wbases, D)
        else:
            raise ValueError("Layer Type Undefined.")