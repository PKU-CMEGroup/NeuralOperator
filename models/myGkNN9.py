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
        bases, wbases = self.bases, self.wbases
        H = self.H

        # Compute coeffcients

        x_hat = torch.einsum("bcx,xk->bck", x, wbases)
        

        # Multiply relevant Fourier modes
        x_hat = x_hat.to(dtype=self.dtype)
        x_hat = mycompl_mul1d(self.weights, H , x_hat)
        x_hat = x_hat.real

        # Return to physical space
        x = torch.real(torch.einsum("bck,xk->bcx", x_hat, bases))

        return x
 
class HGalerkinConv_double(nn.Module):
    def __init__(self, in_channels, out_channels, modes, kernel_modes,  bases_in, wbases_in, bases_out, wbases_out, H_in , H_out):
        super(HGalerkinConv_double, self).__init__()

        self.layer_in = HGalerkinConv(in_channels, out_channels, modes, kernel_modes,  bases_in, wbases_in, H_in)
        self.layer_out = HGalerkinConv(in_channels, out_channels, modes, kernel_modes,  bases_out, wbases_out, H_out)
        
        
    def forward(self, x):

        x_in = self.layer_in(x)
        x_out = self.layer_out(x)
        x = x_in + x_out

        return x
    
class HGalerkinConv_triple(nn.Module):
    def __init__(self, in_channels, out_channels, modes, kernel_modes,bases_in1, wbases_in1,  bases_in2, wbases_in2, bases_out, wbases_out, H_in , H_out):
        super(HGalerkinConv_triple, self).__init__()

        self.layer_in1 = HGalerkinConv(in_channels, out_channels, modes, kernel_modes,  bases_in1, wbases_in1, H_in)
        self.layer_in2 = HGalerkinConv(in_channels, out_channels, modes, kernel_modes,  bases_in2, wbases_in2, H_in)
        self.layer_out = HGalerkinConv(in_channels, out_channels, modes, kernel_modes,  bases_out, wbases_out, H_out)
        
        
    def forward(self, x):

        x_in1 = self.layer_in1(x)
        x_in2 = self.layer_in2(x)
        x_out = self.layer_out(x)
        x = x_in1 + x_in2 + x_out

        return x
        
class wHGalerkinConv(nn.Module):
    def __init__(self, in_channels, out_channels, modes, kernel_modes,  bases, grid_weight, H):
        super(wHGalerkinConv, self).__init__()



        self.in_channels = in_channels
        self.out_channels = out_channels
        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes = modes
        self.bases = bases
        self.grid_weight = grid_weight.to(bases.device)
        self.H = H
        self.dtype = H.dtype

        self.scale = 1 / (in_channels * out_channels)
        self.kernel_modes = kernel_modes

        self.weights = nn.Parameter(
            self.scale
            * torch.randn(in_channels, out_channels, self.kernel_modes, dtype=self.dtype)
        )
        
    def forward(self, x):
        bases = self.bases
        grid_weight_softmax = F.softmax(self.grid_weight, dim=0).unsqueeze(1)  #shape: N,1
        wbases = bases * grid_weight_softmax
        H = self.H

        # Compute coeffcients

        x_hat = torch.einsum("bcx,xk->bck", x, wbases)
        

        # Multiply relevant Fourier modes
        x_hat = x_hat.to(dtype=self.dtype)
        x_hat = mycompl_mul1d(self.weights, H , x_hat)
        x_hat = x_hat.real

        # Return to physical space
        x = torch.real(torch.einsum("bck,xk->bcx", x_hat, bases))

        return x

class wHGalerkinConv_double(nn.Module):
    def __init__(self, in_channels, out_channels, modes, kernel_modes,  bases_in, grid_weight_in, bases_out, grid_weight_out, H_in , H_out):
        super(wHGalerkinConv_double, self).__init__()

        self.layer_in = wHGalerkinConv(in_channels, out_channels, modes, kernel_modes,  bases_in, grid_weight_in, H_in)
        self.layer_out = wHGalerkinConv(in_channels, out_channels, modes, kernel_modes,  bases_out, grid_weight_out, H_out)
        
        
    def forward(self, x):

        x_in = self.layer_in(x)
        x_out = self.layer_out(x)
        x = x_in + x_out

        return x

class myGkNN9(nn.Module):
    def __init__(self, bases_list ,H_in, H_out ,**config):
        super(myGkNN9, self).__init__()

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

        N = self.bases_pca_out.shape[0]
        # self.grid_weight_out = nn.Parameter(torch.ones(N))
        if self.grid_weight_init == 'ones':
            self.grid_weight_out = nn.Parameter(torch.ones(N))
            if self.double_bases:
                self.grid_weight_in = nn.Parameter(torch.ones(N))
        elif self.grid_weight_init == 'random':
            self.grid_weight_out = nn.Parameter(torch.rand(N, dtype=torch.float))
            if self.double_bases:
                self.grid_weight_in = nn.Parameter(torch.rand(N, dtype=torch.float))
        
        if self.H_init == 'random':
            self.scale = 1/(self.GkNN_modes[0]*self.GkNN_modes[0])
            if self.double_bases:
                self.H_in = nn.Parameter(
                    self.scale
                    * torch.rand(self.kernel_modes[0], self.GkNN_modes[0], self.GkNN_modes[0], dtype=torch.float)
                )
            self.H_out = nn.Parameter(
                self.scale
                * torch.rand(self.kernel_modes[0], self.GkNN_modes[0], self.GkNN_modes[0], dtype=torch.float)
            )
        elif self.H_init == 'Galerkin':
            if self.double_bases:
                H_in = torch.zeros(self.kernel_modes[0], self.GkNN_modes[0], self.GkNN_modes[0]) 
                for i in range(min(self.kernel_modes[0], self.GkNN_modes[0])):
                    H_in[i,i,i] = 1
                self.H_in = nn.Parameter(H_in)

            H_out = torch.zeros(self.kernel_modes[0], self.GkNN_modes[0], self.GkNN_modes[0]) 
            for i in range(min(self.kernel_modes[0], self.GkNN_modes[0])):
                H_out[i,i,i] = 1
            self.H_out = nn.Parameter(H_out)
            

        else:
            if self.double_bases:
                self.H_in = H_in
            self.H_out = H_out

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
        if layer_type == "GalerkinConv_fourier":
            num_modes = self.GkNN_modes[index]
            bases = self.bases_fourier
            wbases = self.wbases_fourier
            return GalerkinConv(in_channels, out_channels, num_modes, bases, wbases)
        elif layer_type == "GalerkinConv_pca":
            num_modes = self.GkNN_modes[index]
            bases = self.bases_pca_out
            wbases = self.wbases_pca_out
            return GalerkinConv(in_channels, out_channels, num_modes, bases, wbases)
        elif layer_type == "HGalerkinConv_fourier":
            num_modes = self.GkNN_modes[index]
            kernel_modes = self.kernel_modes[index]
            bases = self.bases_fourier
            wbases = self.wbases_fourier
            H = self.H_out
            return HGalerkinConv(in_channels, out_channels, num_modes, kernel_modes,  bases, wbases, H)
        elif layer_type == "HGalerkinConv_pca":
            if not self.double_bases:
                num_modes = self.GkNN_modes[index]
                kernel_modes = self.kernel_modes[index]
                bases = self.bases_pca_out
                wbases = self.wbases_pca_out
                H = self.H_out
                return HGalerkinConv(in_channels, out_channels, num_modes, kernel_modes, bases, wbases, H)
            else:
                num_modes = self.GkNN_modes[index]
                kernel_modes = self.kernel_modes[index]
                bases_in = self.bases_pca_in
                wbases_in = self.wbases_pca_in
                bases_out = self.bases_pca_out
                wbases_out = self.wbases_pca_out
                H_in = self.H_in
                H_out = self.H_out
                return HGalerkinConv_double(in_channels, out_channels, num_modes, kernel_modes, bases_in, wbases_in, bases_out, wbases_out, H_out , H_in)
        elif layer_type == "wHGalerkinConv_pca":
            if not self.double_bases:
                num_modes = self.GkNN_modes[index]
                kernel_modes = self.kernel_modes[index]
                bases = self.bases_pca_out
                grid_weight = self.grid_weight_out
                H = self.H_out
                return wHGalerkinConv(in_channels, out_channels, num_modes, kernel_modes, bases, grid_weight, H)
            else:
                num_modes = self.GkNN_modes[index]
                kernel_modes = self.kernel_modes[index]
                bases_in = self.bases_pca_in
                grid_weight_in = self.grid_weight_in
                bases_out = self.bases_pca_out
                grid_weight_out = self.grid_weight_out
                H_in = self.H_in
                H_out = self.H_out
                return wHGalerkinConv_double(in_channels, out_channels, num_modes, kernel_modes, bases_in, grid_weight_in, bases_out, grid_weight_out, H_out , H_in)
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