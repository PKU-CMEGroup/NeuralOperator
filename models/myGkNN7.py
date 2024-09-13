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


        
def mycompl_mul1d(weights, H , x_hat):
    x_hat1 = torch.einsum('jkl,bil -> bijk', H , x_hat)
    y = torch.einsum('ioj,bijk -> bok', weights , x_hat1)
    return y


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

class HGalerkinConv_rank1(nn.Module):
    def __init__(self, in_channels, out_channels, modes, kernel_modes,  bases, wbases, U, A , B):
        super(HGalerkinConv_rank1, self).__init__()

        """
        1D Spectral layer. It avoids FFT, but utilizes low rank approximation. 
        
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes = modes
        self.bases = bases
        self.wbases = wbases
        
        self.A = A    # shape : T_modes, mode
        self.B = B    #shape: T_modes, mode
        self.U = U      #shape: kernel_modes, T_modes


        self.dtype = self.A.dtype
        self.scale = 1 / (in_channels * out_channels)
        self.kernel_modes = kernel_modes

        self.weights = nn.Parameter(
            self.scale
            * torch.randn(in_channels, out_channels, self.kernel_modes, dtype=self.dtype)
        )
        
        
        

    def forward(self, x):
        bases, wbases = self.bases, self.wbases
        H = torch.einsum('ji,ik,il -> jkl',self.U, self.A, self.B)   #shape: kernel_mode, mode, mode

        # Compute coeffcients

        x_hat = torch.einsum("bcx,xk->bck", x, wbases)
        

        # Multiply relevant Fourier modes
        x_hat = x_hat.to(dtype=self.dtype)
        x_hat = mycompl_mul1d(self.weights, H , x_hat)
        x_hat = x_hat.real

        # Return to physical space
        x = torch.real(torch.einsum("bck,xk->bcx", x_hat, bases))

        return x
    
class HGalerkinConv_rank1_double(nn.Module):
    def __init__(self, in_channels, out_channels, modes, kernel_modes, bases_in, wbases_in, bases_out, wbases_out,U_in, A_in, B_in,U_out, A_out, B_out ):
        super(HGalerkinConv_rank1_double, self).__init__()

        self.layer_in = HGalerkinConv_rank1(in_channels, out_channels, modes, kernel_modes,  bases_in, wbases_in, U_in, A_in, B_in)
        self.layer_out = HGalerkinConv_rank1(in_channels, out_channels, modes, kernel_modes,  bases_out, wbases_out, U_out, A_out, B_out)
        
        
    def forward(self, x):

        x_in = self.layer_in(x)
        x_out = self.layer_out(x)
        x = x_in + x_out

        return x

class HGalerkinConv_diagonal(nn.Module):
    def __init__(self, in_channels, out_channels, modes, kernel_modes,  bases, wbases, D):
        super(HGalerkinConv_diagonal, self).__init__()

        """
        1D Spectral layer. It avoids FFT, but utilizes low rank approximation. 
        
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes = modes
        self.bases = bases
        self.wbases = wbases
        self.D = D
        self.dtype = D.dtype

        self.scale = 1 / (in_channels * out_channels)
        self.kernel_modes = kernel_modes

        self.weights = nn.Parameter(
            self.scale
            * torch.randn(in_channels, out_channels, self.kernel_modes, dtype=self.dtype)
        )
                
    def forward(self, x):
        bases, wbases = self.bases, self.wbases
        D = self.D
        H = torch.diag_embed(D).to(D.device)

        # Compute coeffcients

        x_hat = torch.einsum("bcx,xk->bck", x, wbases)
        

        # Multiply relevant Fourier modes
        x_hat = x_hat.to(dtype=self.dtype)
        x_hat = mycompl_mul1d(self.weights, H , x_hat)
        x_hat = x_hat.real

        # Return to physical space
        x = torch.real(torch.einsum("bck,xk->bcx", x_hat, bases))

        return x

class HGalerkinConv_diagonal_double(nn.Module):
    def __init__(self, in_channels, out_channels, modes, kernel_modes,  bases_in, wbases_in, bases_out, wbases_out, D_in , D_out):
        super(HGalerkinConv_diagonal_double, self).__init__()

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
        self.D_in = D_in
        self.D_out = D_out
        
        self.dtype = D_out.dtype

        self.scale = 1 / (in_channels * out_channels)
        self.kernel_modes = kernel_modes

        self.weights_in = nn.Parameter(
            self.scale
            * torch.rand(in_channels, out_channels, self.kernel_modes, dtype=self.dtype)
        )
        self.weights_out = nn.Parameter(
            self.scale
            * torch.rand(in_channels, out_channels, self.kernel_modes, dtype=self.dtype)
        )
        
        

    def forward(self, x):
        bases_in, wbases_in, bases_out, wbases_out = self.bases_in, self.wbases_in, self.bases_out, self.wbases_out
        D_in = self.D_in
        H_in = torch.diag_embed(D_in).to(D_in.device)
        D_out = self.D_out
        H_out = torch.diag_embed(D_out).to(D_out.device)

        # Compute coeffcients
        
        x_hat_in = torch.einsum("bcx,xk->bck", x, wbases_in)


        # Multiply relevant Fourier modes
        x_hat_in = x_hat_in.to(dtype=self.dtype)
        x_hat_in = mycompl_mul1d(self.weights_in, H_in , x_hat_in)
        x_hat_in = x_hat_in.real

        # Return to physical space
        x_in = torch.real(torch.einsum("bck,xk->bcx", x_hat_in, bases_in))

        # Compute coeffcients

        x_hat_out = torch.einsum("bcx,xk->bck", x, wbases_out)


        # Multiply relevant Fourier modes
        x_hat_out = x_hat_out.to(dtype=self.dtype)
        x_hat_out = mycompl_mul1d(self.weights_out, H_out , x_hat_out)
        x_hat_out = x_hat_out.real

        # Return to physical space
        x_out = torch.real(torch.einsum("bck,xk->bcx", x_hat_out, bases_out))

        x = x_in + x_out

        return x
    

class myGkNN7(nn.Module):
    def __init__(self, bases_list  ,**config):
        super(myGkNN7, self).__init__()

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
        if self.H_type == 'full_complex':
            self.scale = 1/(self.GkNN_modes[0]*self.GkNN_modes[0])
            if self.double_bases == True:
                self.H_in = nn.Parameter(
                    self.scale
                    * torch.rand(self.kernel_modes[0], self.GkNN_modes[0], self.GkNN_modes[0], dtype=torch.complex64)
                )
            self.H_out = nn.Parameter(
                self.scale
                * torch.rand(self.kernel_modes[0], self.GkNN_modes[0], self.GkNN_modes[0], dtype=torch.complex64)
            )
        elif self.H_type == 'full_real':
            self.scale = 1/(self.GkNN_modes[0]*self.GkNN_modes[0])
            if self.double_bases == True:
                self.H_in = nn.Parameter(
                    self.scale
                    * torch.rand(self.kernel_modes[0], self.GkNN_modes[0], self.GkNN_modes[0], dtype=torch.float)
                )
            self.H_out = nn.Parameter(
                self.scale
                * torch.rand(self.kernel_modes[0], self.GkNN_modes[0], self.GkNN_modes[0], dtype=torch.float)
            )

        elif self.H_type == "rank1_real":
            self.scale = 1/self.GkNN_modes[0]
            self.scale_T = 1/self.T_modes
            if self.double_bases == True:
                self.A_in = nn.Parameter(
                    self.scale
                    * torch.rand(self.T_modes, self.GkNN_modes[0], dtype=torch.float)
                )
                self.B_in = nn.Parameter(
                    self.scale
                    * torch.rand(self.T_modes, self.GkNN_modes[0], dtype=torch.float)
                )
                self.U_in = nn.Parameter(
                    self.scale_T
                    * torch.rand(self.kernel_modes[0], self.T_modes, dtype=torch.float)
                )
            self.A_out = nn.Parameter(
                self.scale
                * torch.rand(self.T_modes, self.GkNN_modes[0], dtype=torch.float)
            )
            self.B_out = nn.Parameter(
                self.scale
                * torch.rand(self.T_modes, self.GkNN_modes[0], dtype=torch.float)
            )
            self.U_out = nn.Parameter(
                self.scale_T
                * torch.rand(self.kernel_modes[0], self.T_modes, dtype=torch.float)
            )
            
        elif self.H_type == "diagonal":
            self.scale = 1/self.GkNN_modes[0]
            if self.double_bases == True:
                self.D_in = nn.Parameter(
                    self.scale
                    * torch.rand(self.kernel_modes[0], self.GkNN_modes[0], dtype=torch.float)
                )
            self.D_out = nn.Parameter(
                self.scale
                * torch.rand(self.kernel_modes[0], self.GkNN_modes[0], dtype=torch.float)
            )

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
        if layer_type == "HGalerkinConv_pca":
            if self.double_bases ==True:
                num_modes = self.GkNN_modes[index]
                kernel_modes = self.kernel_modes[index]
                bases_in = self.bases_pca_in
                wbases_in = self.wbases_pca_in
                bases_out = self.bases_pca_out
                wbases_out = self.wbases_pca_out
                if self.H_type == "rank1_real":
                    A_in = self.A_in
                    B_in = self.B_in
                    U_in = self.U_in
                    A_out = self.A_out
                    B_out = self.B_out
                    U_out = self.U_out
                    return HGalerkinConv_rank1_double(in_channels, out_channels, num_modes, kernel_modes, bases_in, wbases_in, bases_out, wbases_out,U_in, A_in, B_in, U_out,A_out, B_out )
                elif self.H_type == 'diagonal':
                    D_in = self.D_in
                    D_out = self.D_out
                    return HGalerkinConv_diagonal_double(in_channels, out_channels, num_modes, kernel_modes, bases_in, wbases_in, bases_out, wbases_out, D_in, D_out)
                else:
                    H_in = self.H_in
                    H_out = self.H_out
                    return HGalerkinConv_double(in_channels, out_channels, num_modes, kernel_modes, bases_in, wbases_in, bases_out, wbases_out, H_out , H_in)
            else:
                num_modes = self.GkNN_modes[index]
                kernel_modes = self.kernel_modes[index]
                bases = self.bases_pca_out
                wbases = self.wbases_pca_out
                if self.H_type == "rank1_real":
                    A = self.A_out
                    B = self.B_out
                    U = self.U_out
                    return HGalerkinConv_rank1(in_channels, out_channels, num_modes, kernel_modes, bases, wbases, U, A, B)
                elif self.H_type == 'diagonal':
                    D = self.D_out
                    return HGalerkinConv_diagonal(in_channels, out_channels, num_modes, kernel_modes, bases, wbases, D )
                else:
                    H = self.H_out
                    return HGalerkinConv(in_channels, out_channels, num_modes, kernel_modes, bases, wbases, H) 
        elif layer_type == "HGalerkinConv_fourier":
            num_modes = self.GkNN_modes[index]
            kernel_modes = self.kernel_modes[index]
            bases = self.bases_fourier
            wbases = self.wbases_fourier
            if self.H_type == "rank1_real":
                A = self.A_out
                B = self.B_out
                return HGalerkinConv_rank1(in_channels, out_channels, num_modes, kernel_modes, bases, wbases, A, B )
            elif self.H_type == 'diagonal':
                D = self.D_out
                return HGalerkinConv_diagonal(in_channels, out_channels, num_modes, kernel_modes, bases, wbases, D )
            else:
                H = self.H_out
                return HGalerkinConv(in_channels, out_channels, num_modes, kernel_modes, bases, wbases, H)
        else:
            raise ValueError("Layer Type Undefined.")