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
        #x.shape: bsz,channel,N
        bases, wbases = self.bases, self.wbases
        H = self.H

        # Compute coeffcients

        x_hat = torch.einsum("bcx,xk->bck", x, wbases)
        

        # Multiply relevant Fourier modes
        x_hat = mycompl_mul1d(self.weights, H , x_hat)

        # Return to physical space
        x = torch.real(torch.einsum("bck,xk->bcx", x_hat, bases))

        return x
 
class HGalerkinConv_double(nn.Module):
    def __init__(self, in_channels, out_channels, modes, kernel_modes,  bases1, wbases1, bases2, wbases2, H1 , H2):
        super(HGalerkinConv_double, self).__init__()

        self.layer1 = HGalerkinConv(in_channels, out_channels, modes, kernel_modes,  bases1, wbases1, H1)
        self.layer2 = HGalerkinConv(in_channels, out_channels, modes, kernel_modes,  bases2, wbases2, H2)
        
        
    def forward(self, x):

        x1 = self.layer1(x)
        x2 = self.layer2(x)
        x = x1 + x2

        return x
class interp_GalerkinConv(nn.Module):
    def __init__(self, in_channels, out_channels, modes, index, bases):
        super(interp_GalerkinConv, self).__init__()

        """
        1D Spectral layer. It avoids FFT, but utilizes low rank approximation. 
        
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes = modes
        self.bases = bases
        self.index = index

        self.scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            self.scale
            * torch.rand(in_channels, out_channels, self.modes, dtype=torch.float)
        )

    def forward(self, x):
        #shape: bsz,channel,N
        bases = self.bases
        index = self.index
        k_max = self.modes
        # Compute coeffcients

        # x_hat = x[:,:,index[:k_max]]  #shape: bsz,channel,k
        x_hat = torch.einsum("bcx,xk->bck", x, bases)

        # Multiply relevant Fourier modes
        x_hat = compl_mul1d(x_hat, self.weights)

        # Return to physical space
        x = torch.real(torch.einsum("bck,xk->bcx", x_hat, bases))

        return x    

class interp_HGalerkinConv(nn.Module):
    def __init__(self, in_channels, out_channels, modes, kernel_modes, index, bases, H,):
        super(interp_HGalerkinConv, self).__init__()


        self.in_channels = in_channels
        self.out_channels = out_channels
        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes = modes
        self.bases = bases
        self.index = index
        self.H = H
        self.dtype = H.dtype

        self.scale = 1 / (in_channels * out_channels)
        self.kernel_modes = kernel_modes

        self.weights = nn.Parameter(
            self.scale
            * torch.randn(in_channels, out_channels, self.kernel_modes, dtype=self.dtype)
        )
        
        
        

    def forward(self, x):
        #x.shape: bsz,channel,N
        bases = self.bases
        index = self.index
        k_max = self.modes
        H = self.H

        # Compute coeffcients

        x_hat = x[:,:,index[:k_max]]  #shape: bsz,channel,k
        

        # Multiply relevant Fourier modes
        x_hat = mycompl_mul1d(self.weights, H , x_hat)

        # Return to physical space
        x = torch.real(torch.einsum("bck,xk->bcx", x_hat, bases))

        return x      

class interp_HGkNN(nn.Module):
    def __init__(self, bases_list ,index,**config):
        super(interp_HGkNN, self).__init__()

        self.bases1=bases_list[0]
        self.wbases1=bases_list[1]
        self.bases2=bases_list[2]
        self.wbases2=bases_list[3]

        self.index = index
        
        
        
        self.config = defaultdict(lambda: None, **config)
        self.config = dict(self.config)
        all_attr = list(self.config.keys())
        for key in all_attr:
            setattr(self, key, self.config[key])
        
    

        if self.coeff_compute == 'interpolation':
            self.GkNN_mode = len(self.index)
        self.scale = 1/(self.GkNN_mode*self.GkNN_mode)
        if 'HGalerkinConv' in self.layer_types:
            if self.double_bases:
                self.H2 = nn.Parameter(
                    self.scale
                    * torch.rand(self.kernel_mode, self.GkNN_mode, self.GkNN_mode, dtype=torch.float)
                )
            self.H1 = nn.Parameter(
                self.scale
                * torch.rand(self.kernel_mode, self.GkNN_mode, self.GkNN_mode, dtype=torch.float)
            )

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
        if layer_type == "GalerkinConv":
            if self.coeff_compute == "dot_product":
                num_modes = self.GkNN_mode
                bases = self.bases1
                wbases = self.wbases1
                return GalerkinConv(in_channels, out_channels, num_modes, bases, wbases)
            elif self.coeff_compute == "interpolation":
                bases = self.bases1
                index = self.index
                num_modes = len(self.index)
                return interp_GalerkinConv(in_channels, out_channels, num_modes, index, bases)
        elif layer_type == "HGalerkinConv":
            if self.coeff_compute == "dot_product":
                if not self.double_bases:
                    num_modes = self.GkNN_mode
                    kernel_modes = self.kernel_mode
                    bases = self.bases1
                    wbases = self.wbases1
                    H1 = self.H1
                    return HGalerkinConv(in_channels, out_channels, num_modes, kernel_modes, bases, wbases, H1)
                else:
                    num_modes = self.GkNN_mode
                    kernel_modes = self.kernel_mode
                    bases1 = self.bases1
                    wbases1 = self.wbases1
                    bases2 = self.bases2
                    wbases2 = self.wbases2
                    H1 = self.H1
                    H2 = self.H2
                    return HGalerkinConv_double(in_channels, out_channels, num_modes, kernel_modes, bases1, wbases1, bases2, wbases2, H1 , H2)
            elif self.coeff_compute == "interpolation":
                kernel_modes = self.kernel_mode
                bases = self.bases1
                index = self.index
                num_modes = len(self.index)
                H1 = self.H1
                return interp_HGalerkinConv(in_channels, out_channels, num_modes, kernel_modes, index, bases, H1)
        else:
            raise ValueError("Layer Type Undefined.")