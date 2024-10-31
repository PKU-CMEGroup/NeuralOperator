import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from collections import defaultdict
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

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

def mycompl_mul1d_D(weights, D , x_hat):
    x_hat_expanded = x_hat.unsqueeze(2)  # shape: (bsz, input channel, 1, modes)
    D_expanded = D.unsqueeze(0).unsqueeze(0)  # shape: (1, 1, kernel_modes, modes)
    x_hat1 = x_hat_expanded * D_expanded  # shape: (bsz, input channel, kernel_modes, modes)
    y = torch.einsum('bijk,ioj -> bok', x_hat1 , weights )
    return y

def compute_H_D(D, product):
    #D.shape: kernel_mode,modes_out
    #product.shape: modes_out,modes_in
    J = D.shape[0]
    #product.shape: bsz,K,L
    K = product.shape[0]
    L = product.shape[1]
    # print(f'bsz={bsz}')
    H = D.reshape(J,K,1)*product.reshape(1,K,L)
    return H

class PhyGalerkinConv(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super(PhyGalerkinConv, self).__init__()

        """
        1D Spectral layer. It avoids FFT, but utilizes low rank approximation. 
        
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes = modes
        self.dtype = torch.float
        self.scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            self.scale
            * torch.rand(in_channels, out_channels, self.modes, dtype=self.dtype)
        )

    def forward(self, x, wbases, bases):

        # Compute coeffcients
        x_hat = torch.einsum("bcx,bxk->bck", x, wbases)
        # x_hat = x_hat.to(self.dtype)
        # Multiply relevant Fourier modes
        x_hat = compl_mul1d(x_hat, self.weights)

        # Return to physical space
        x = torch.real(torch.einsum("bck,bxk->bcx", x_hat, bases))

        return x

class PhyHGalerkinConv(nn.Module):
    def __init__(self, in_channels, out_channels, modes_in, modes_out, kernel_modes, H):
        super(PhyHGalerkinConv, self).__init__()


        self.in_channels = in_channels
        self.out_channels = out_channels
        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes_in = modes_in
        self.modes_out = modes_out
        self.H = H
        self.dtype = H.dtype

        self.scale = 1 / (in_channels * out_channels)
        self.kernel_modes = kernel_modes

        self.weights = nn.Parameter(
            self.scale
            * torch.randn(in_channels, out_channels, self.kernel_modes, dtype=self.dtype)
        )   
        

    def forward(self, x, wbases, bases):
        #x.shape: bsz,channel,N
        H = self.H
        # Compute coeffcients
        x = x.to(wbases.dtype)
        x_hat = torch.einsum("bcx,bxk->bck", x, wbases[:,:,:self.modes_in])
        x_hat = x_hat.to(self.dtype)
        # Multiply relevant Fourier modes
        x_hat = mycompl_mul1d(self.weights, H , x_hat)
        # Return to physical space
        x = torch.real(torch.einsum("bck,bxk->bcx", x_hat, bases[:,:,:self.modes_out]))
        return x

class PhyDGalerkinConv(nn.Module):
    def __init__(self, in_channels, out_channels, modes, kernel_modes, D):
        super(PhyDGalerkinConv, self).__init__()

        """
        1D Spectral layer. It avoids FFT, but utilizes low rank approximation. 
        
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes = modes
        self.scale1 = 1 / kernel_modes
        self.D = D
        self.dtype = D.dtype
            

        self.scale2 = 1 / (in_channels * out_channels)
        self.kernel_modes = kernel_modes

        self.weights = nn.Parameter(
            self.scale2
            * torch.randn(in_channels, out_channels, self.kernel_modes, dtype=self.dtype)
        )
                
    def forward(self, x, wbases, bases):

        # Compute coeffcients
        x = x.to(wbases.dtype)
        x_hat = torch.einsum("bcx,bxk->bck", x, wbases[:,:,:self.modes])
        x_hat = x_hat.to(self.dtype)


        # Multiply relevant Fourier modes
        x_hat = mycompl_mul1d_D(self.weights, self.D , x_hat)

        # Return to physical space
        x = torch.real(torch.einsum("bck,bxk->bcx", x_hat, bases[:,:,:self.modes]))

        return x

class PhyHGkNN3(nn.Module):
    def __init__(self,base_para_list,**config):

        super(PhyHGkNN3, self).__init__()

        self.base_para_list = base_para_list
        self.config = defaultdict(lambda: None, **config)
        self.config = dict(self.config)
        all_attr = list(self.config.keys())
        for key in all_attr:
            setattr(self, key, self.config[key])
        
        self.construct_base_parameter()
        self.construct_H()
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
        self.fc0 = nn.Linear(self.in_dim,self.layers_dim[0])
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
        self.ln_layers = nn.ModuleList([nn.LayerNorm(dim) for dim in self.layers_dim[1:]])
        self.ln0 = nn.LayerNorm(self.layers_dim[0])


    def forward(self, x):
        """
        Input shape (of x):     (batch, nx_in,  channels_in)
        Output shape:           (batch, nx_out, channels_out)

        The input resolution is determined by x.shape[-1]
        The output resolution is determined by self.s_outputspace
        """

        grid = x[:,:,self.in_dim-self.phy_dim:]
        bases, wbases = self.compute_base(grid)

        length = len(self.ws)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)
        x = self.ln0(x.transpose(1,2)).transpose(1,2)


        for i, (layer , w, dplayer,ln) in enumerate(zip(self.sp_layers, self.ws, self.dropout_layers,self.ln_layers)):
            x1 = layer(x , wbases , bases)
            x2 = w(x)
            x = x1 + x2
            x = dplayer(x)
            x = ln(x.transpose(1,2)).transpose(1,2)
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
        if layer_type == "HGalerkinConv":
            num_modes_in = self.num_modes_in
            num_modes_out = self.num_modes_out
            kernel_modes = self.kernel_mode
            return PhyHGalerkinConv(in_channels, out_channels, num_modes_in, num_modes_out, kernel_modes, self.H1)
        elif layer_type == "DGalerkinConv":
            num_modes = self.num_modes_out
            kernel_modes = self.kernel_mode
            return PhyDGalerkinConv(in_channels, out_channels,  num_modes, kernel_modes, self.H1)
        elif layer_type == "GalerkinConv":
            num_modes = self.num_modes_out
            kernel_modes = self.kernel_mode
            return PhyGalerkinConv(in_channels, out_channels,  num_modes)
        else:
            raise ValueError("Layer Type Undefined.")


    def construct_base_parameter(self):
        if self.base_type == 'Gauss':
            if self.base_para_list[0] != None:
                self.basepts_Gauss = self.base_para_list[0].to(self.device)
            else:
                self.basepts_Gauss = self.uniform_points(self.range_pts_Gauss,self.num_basepts_Gauss, self.phy_dim).to(self.device)
            if self.base_para_list[1] != None:
                self.baseweight_Gauss = self.base_para_list[1].to(self.device)
                self.minweight_Gauss = torch.tensor(0,device = self.device)
            else:
                if self.baseweight_type == 'trained':
                    self.baseweight_Gauss = nn.Parameter(self.basicwt_Gauss*torch.rand(self.num_basepts_Gauss, self.phy_dim, dtype = torch.float))
                else:
                    self.baseweight_Gauss = self.basicwt_Gauss/2*torch.ones(self.num_basepts_Gauss, self.phy_dim, dtype = torch.float, device = self.device)
        elif self.base_type =='Fourier':
            if self.base_para_list[0] != None:
                self.baseweight_Fourier = self.base_para_list[0].to(self.device)
            else:
                self.baseweight_Fourier = nn.Parameter(self.basicwt_Fourier*(torch.rand(self.num_fourierbases, self.phy_dim, dtype = torch.float)-1/2))
        else:
            raise TypeError('base type not defined') 
    
    def compute_base(self,grid):
        N = grid.shape[1]
        if self.base_type == 'Gauss':
            bases = self.compute_bases_Gauss(grid)
            wbases = bases/N
            return bases,wbases
        elif self.base_type == 'Fourier':
            bases = self.compute_bases_Fourier(grid)
            wbases = bases/N
            return bases,wbases
        else:
            raise TypeError('base type not defined')



    def compute_bases_Gauss(self,grid):
        #grid.shape:  bsz,n,phy_in_channel
        grid = grid.unsqueeze(2) #bsz,n,1,phy_in_channel

        basepts = self.basepts_Gauss.unsqueeze(0).unsqueeze(0) # 1,1,phy_out_channel,phy_in_channel
        baseweight = torch.abs(self.baseweight_Gauss).unsqueeze(0).unsqueeze(0)  #1,1,phy_out_channel,phy_in_channel
        baseweight = baseweight + self.minweight_Gauss
        bases = torch.sqrt(torch.prod(baseweight, dim=3))*torch.exp(-1*torch.sum(baseweight*(grid-basepts)**2,dim=3))  #bsz,n,phy_out_channel,phy_in_channel-->bsz,n,phy_out_channel
        bases = bases*math.sqrt(bases.shape[1])/torch.norm(bases, p=2, dim=1, keepdim=True)
        return bases
    

    def compute_bases_Fourier(self,grid):
        #grid.shape:  bsz,n,phy_in_channel
        grid = grid.unsqueeze(2) #bsz,n,1,phy_in_channel
        baseweight = self.baseweight_Fourier.unsqueeze(0).unsqueeze(0)  #1,1,phy_out_channel,phy_in_channel
        bases_complex = torch.exp(torch.sum(1j*baseweight*grid,dim=3))  #bsz,n,phy_out_channel,phy_in_channel-->bsz,n,phy_out_channel
        bases = torch.cat((torch.real(bases_complex),torch.imag(bases_complex)),dim=-1)
        return bases      


    def uniform_points(self,range_pts,num_pts,dim):
        a = round(math.pow(num_pts, 1 / dim))
        index_tensors = []
        for k in range(dim):
            xmin,xmax = range_pts[k][0],range_pts[k][1]
            idx = xmin + (xmax-xmin)*torch.arange(a).float().add(0.5).div(a)
            idx = idx.view((1,) * k+ (-1,) + (1,) * (dim - k - 1))
            index_tensors.append(idx.expand(a, *([a] * (dim - 1))))
        num_pts1 = int(torch.pow(torch.tensor(a),dim))
        x = torch.stack(index_tensors, dim=dim).reshape(num_pts1,dim)
        return x
    

    def construct_H(self):
        if 'HGalerkinConv' in self.layer_types:
            self.scale1 = 1/(self.num_modes_in*self.num_modes_out)
            self.H1 = nn.Parameter(
                    self.scale1
                    * torch.rand(self.kernel_mode, self.num_modes_out, self.num_modes_in, dtype=torch.float)
                )
        if 'DGalerkinConv' in self.layer_types:
            self.scale1 = 1/self.num_modes_in
            self.H1 = nn.Parameter(
                    self.scale1
                    * torch.rand(self.kernel_mode, self.num_modes_in, dtype=torch.float)
                )

        




    def plot_hidden_layer(self,x,y,Nx,Ny,save_figure_hidden,epoch):
        fig, axs = plt.subplots(1, 8, figsize=(24, 5))
        length = len(self.ws)

        x0 = x[0,:,0].cpu().reshape(Nx,Ny)
        im = axs[0].imshow(x0, cmap='viridis')
        axs[0].set_title('input')
        fig.colorbar(im, ax=axs[0])

        grid = x[:,:,self.in_dim-self.phy_dim:]
        bases, wbases = self.compute_base(grid)

        length = len(self.ws)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)
        x = self.ln0(x.transpose(1,2)).transpose(1,2)


        x0 = x[0,0,:].cpu().reshape(Nx,Ny)
        im = axs[1].imshow(x0, cmap='viridis')
        axs[1].set_title(f'x{0}')
        fig.colorbar(im, ax=axs[1])

        for i, (layer , w, dplayer,ln) in enumerate(zip(self.sp_layers, self.ws, self.dropout_layers,self.ln_layers)):
            x1 = layer(x , wbases , bases)
            x2 = w(x)
            x = x1 + x2
            x = dplayer(x)
            x = ln(x.transpose(1,2)).transpose(1,2)
            if self.act is not None and i != length - 1:
                x = self.act(x)
            x0 = x[0,0,:].cpu().reshape(Nx,Ny)
            im = axs[i+2].imshow(x0, cmap='viridis')
            axs[i+2].set_title(f'x{i+1}')
            fig.colorbar(im, ax=axs[i+2])

        x = x.permute(0, 2, 1)

        fc_dim = self.fc_dim 
        
        if fc_dim > 0:
            x = self.fc1(x)
            if self.act is not None:
                x = self.act(x)

        x = self.fc2(x)

        x0 = x[0,:,:].cpu().reshape(Nx,Ny)
        im = axs[6].imshow(x0, cmap='viridis')
        axs[6].set_title('output')
        fig.colorbar(im, ax=axs[6])

        y0 = y[0,:,:].cpu().reshape(Nx,Ny)
        im = axs[7].imshow(y0, cmap='viridis')
        axs[7].set_title('truth_y')
        fig.colorbar(im, ax=axs[7])


        plt.tight_layout()
        plt.savefig(save_figure_hidden + 'ep'+str(epoch).zfill(3)+'.png', format='png')
        plt.close()
    

    def plot_bases(self,num_bases,grid,Nx,Ny,save_figure_bases,epoch):
        bases = self.compute_base(grid)  #bsz,N,k
        row, col = decompose_integer(num_bases)
        fig, axs = plt.subplots(row, col, figsize=(3*col, 3*row)) 
        K = bases.shape[-1]
        ratio = K//num_bases
        for i in range(row): 
            for j in range(col):  
                k = (i*col+j)*ratio
                base_k = bases[0,:,k].detach().cpu().reshape(Nx,Ny)
                if self.base_type == 'Gauss':
                    weight = ['{:.1f}'.format(w.item()) for w in self.baseweight_Gauss[k,:].detach().cpu().numpy()]
                elif self.base_type == 'Fourier':
                    weight = ['{:.1f}'.format(w.item()) for w in self.baseweight_Fourier[k,:].detach().cpu().numpy()]
                im = axs[i,j].imshow(base_k, cmap='viridis')
                axs[i,j].set_title(f'base{k},{weight}')
                fig.colorbar(im, ax=axs[i,j])

        plt.tight_layout()
        plt.savefig(save_figure_bases + 'bases_ep'+str(epoch).zfill(3)+'.png', format='png')
        plt.close() 
    
def decompose_integer(n):
    val = int(math.sqrt(n))
    for i in range(val, 0, -1):
        if n % i == 0:
            return (i, n // i)
    return (1, n)