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
import numpy as np
from scipy.linalg import block_diag
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
        

    def forward(self, x, wbases, bases, base_mask= False):
        #x.shape: bsz,channel,N
        H = self.H
        # base_mask = base_mask.unsqueeze(0).repeat(self.kernel_modes,1,1)
        # H = H*base_mask
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

        self.D = D
        self.dtype = D.dtype
            

        self.scale = 1 / (in_channels * out_channels)
        self.kernel_modes = kernel_modes

        self.weights = nn.Parameter(
            self.scale
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
        x = torch.einsum("bck,bxk->bcx", x_hat, bases[:,:,:self.modes])

        return x

class PhyHGkNN4(nn.Module):
    def __init__(self,base_para_list,**config):

        super(PhyHGkNN4, self).__init__()

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

        
        if self.input_with_weight:
            grid = x[:,:,self.in_dim-self.phy_dim-1:-1]
            grid_weight = x[:,:,-1]
        else:
            grid = x[:,:,self.in_dim-self.phy_dim:]
            grid_weight = None
        bases, wbases = self.compute_bases(grid, grid_weight)

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
            num_modes_in = self.num_modes
            num_modes_out = self.num_modes
            kernel_modes = self.kernel_mode
            return PhyHGalerkinConv(in_channels, out_channels, num_modes_in, num_modes_out, kernel_modes, self.H1)
        if layer_type == "DGalerkinConv":
            num_modes = self.num_modes
            kernel_modes = self.kernel_mode
            return PhyDGalerkinConv(in_channels, out_channels,  num_modes, kernel_modes, self.H1)
        # elif layer_type == "GalerkinConv":
        #     num_modes = self.num_modes_out
        #     kernel_modes = self.kernel_mode
        #     return PhyGalerkinConv(in_channels, out_channels,  num_modes)
        else:
            raise ValueError("Layer Type Undefined.")


    def construct_base_parameter(self):
        self.baseweight_Fourier = self.base_para_list[0].to(self.device)  #shape: K1,phy_dim
        self.basepts_Gauss = self.base_para_list[1].to(self.device)  #shape: K2,phy_dim
        self.baseweight_Gauss = self.base_para_list[2].to(self.device)  #shape: K2,phy_dim
        k1 = self.baseweight_Fourier.shape[0]
        k2 = self.baseweight_Gauss.shape[0]
        if self.Fourier_only:
            k2 = 0
        elif self.Gauss_only:
            k1 = 0
        self.num_modes = 2*k1 + k2
        # self.base_mask = self.create_mask(k1,k2).to(self.device)

    def compute_bases(self, grid, gridweight):
        N = grid.shape[1]
        bases_Fourier = self.compute_bases_Fourier(grid)   #shape: bsz,N,2*K1
        bases_Gauss = self.compute_bases_Gauss(grid) #shape: bsz,N,K2
        if gridweight != None:
            gridweight = gridweight.unsqueeze(-1)
            wbases_Fourier = bases_Fourier * gridweight
            wbases_Gauss = bases_Gauss *gridweight
        else:
            wbases_Fourier = bases_Fourier/N
            wbases_Gauss = bases_Gauss/N

        if self.Fourier_only:
            return bases_Fourier,wbases_Fourier
        if self.Gauss_only:
            return bases_Gauss,wbases_Gauss
        bases = torch.cat((bases_Fourier,bases_Gauss),dim=-1)  # shape: bsz,N, 2*K1 + K2
        wbases = torch.cat((wbases_Fourier,wbases_Gauss),dim=-1)
        return bases,wbases


    def compute_bases_Fourier(self,grid):
        #grid.shape:  bsz,N,phy_dim
        grid = grid.unsqueeze(2) #bsz,N,1,phy_dim
        baseweight = self.baseweight_Fourier.unsqueeze(0).unsqueeze(0)  #1,1,K1,phy_dim
        bases_complex = torch.exp(torch.sum(1j*baseweight*grid,dim=3))  #bsz,N,K1,phy_dim-->bsz,N,K1
        bases = torch.cat((torch.real(bases_complex),torch.imag(bases_complex)),dim=-1)  #bsz,N,2*K1
        bases = bases*math.sqrt(bases.shape[1])/(torch.norm(bases, p=2, dim=1, keepdim=True) + 1e-5)
        return bases     
    
    def compute_bases_Gauss(self,grid):
        #grid.shape:  bsz,N,phy_dim
        grid = grid.unsqueeze(2) #bsz,N,1,phy_dim
        basepts = self.basepts_Gauss.unsqueeze(0).unsqueeze(0) # 1,1,K2,phy_dim
        baseweight = torch.abs(self.baseweight_Gauss).unsqueeze(0).unsqueeze(0)  #1,1,K2,phy_dim
        bases = torch.sqrt(torch.prod(baseweight, dim=3))*torch.exp(-1*torch.sum(baseweight*(grid-basepts)**2,dim=3))  #bsz,N,K2,phy_dim-->bsz,N,K2 
        bases = bases*math.sqrt(bases.shape[1])/torch.norm(bases, p=2, dim=1, keepdim=True)
        return bases
    

    def construct_H(self):
        if 'HGalerkinConv' in self.layer_types:
            self.scale1 = 1/(self.num_modes*self.num_modes)
            self.H1 = nn.Parameter(
                    self.scale1
                    * torch.rand(self.kernel_mode, self.num_modes, self.num_modes, dtype=torch.float)
                )
        if 'DGalerkinConv' in self.layer_types:
            self.scale1 = 1/self.num_modes
            self.H1 = nn.Parameter(
                    self.scale1
                    * torch.rand(self.kernel_mode, self.num_modes, dtype=torch.float)
                )

    def create_mask(self, k1, k2):
        E = np.eye(k1).astype(int)
        E1 = np.block([
            [E, E],
            [E, E]  
        ])
        E2 = np.ones((k2,k2)).astype(int)
        row_idx, col_idx = np.arange(k2), np.arange(k2)
        not_diagonals = np.abs(row_idx[:, None] - col_idx) > 5
        E2[not_diagonals] = 0
        mask = block_diag(E1,E2)
        mask = torch.from_numpy(mask)
        return mask

    def plot_hidden_layer(self,x,y,Nx,Ny,save_figure_hidden,epoch,plot_hidden_layers_num):
        fig, axs = plt.subplots(plot_hidden_layers_num, 8, figsize=(24, 3*plot_hidden_layers_num))
        length = len(self.ws)
        for j in range(plot_hidden_layers_num):
            if self.phy_dim==2:
                x0 = x[j,:,0].cpu().reshape(Nx,Ny)
                im = axs[j,0].imshow(x0, cmap='viridis')
                fig.colorbar(im, ax=axs[j,0])
            elif self.phy_dim==1:
                x0 = x[j, :, 0].cpu()
                im = axs[j,0].plot(x0)
            else:
                raise TypeError('dim is too high to plot')
            axs[j,0].set_title('input')


        if self.input_with_weight:
            grid = x[:,:,self.in_dim-self.phy_dim-1:-1]
            grid_weight = x[:,:,-1]
        else:
            grid = x[:,:,self.in_dim-self.phy_dim:]
            grid_weight = None
        bases, wbases = self.compute_bases(grid, grid_weight)

        length = len(self.ws)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)
        x = self.ln0(x.transpose(1,2)).transpose(1,2)
        
        for j in range(plot_hidden_layers_num):
            if self.phy_dim==2:
                x0 = x[j,0,:].cpu().reshape(Nx,Ny)
                im = axs[j,1].imshow(x0, cmap='viridis')
                fig.colorbar(im, ax=axs[j,1])
            elif self.phy_dim==1:
                x0 = x[j,0,:].cpu()
                im = axs[j,1].plot(x0)

            axs[j,1].set_title(f'x{0}')


        for i, (layer , w, dplayer,ln) in enumerate(zip(self.sp_layers, self.ws, self.dropout_layers,self.ln_layers)):
            x1 = layer(x , wbases , bases)
            x2 = w(x)
            x = x1 + x2
            x = dplayer(x)
            x = ln(x.transpose(1,2)).transpose(1,2)
            if self.act is not None and i != length - 1:
                x = self.act(x)
            for j in range(plot_hidden_layers_num):
                if self.phy_dim==2:
                    x0 = x[j,0,:].cpu().reshape(Nx,Ny)
                    im = axs[j,i+2].imshow(x0, cmap='viridis')
                    fig.colorbar(im, ax=axs[i+2])
                elif self.phy_dim==1:
                    x0 = x[j,0,:].cpu()
                    im = axs[j,i+2].plot(x0)
                axs[j,i+2].set_title(f'x{i+1}')


        x = x.permute(0, 2, 1)

        fc_dim = self.fc_dim 
        
        if fc_dim > 0:
            x = self.fc1(x)
            if self.act is not None:
                x = self.act(x)

        x = self.fc2(x)
        for j in range(plot_hidden_layers_num):
            if self.phy_dim==2:
                x0 = x[j,:,:].cpu().reshape(Nx,Ny)
                im = axs[j,6].imshow(x0, cmap='viridis')
                fig.colorbar(im, ax=axs[j,6])
            elif self.phy_dim==1:
                x0 = x[j, :, 0].cpu()
                im = axs[j,6].plot(x0)
            axs[j,6].set_title('output')

            if self.phy_dim==2:
                y0 = y[j,:,0].cpu().reshape(Nx,Ny)
                im = axs[j,7].imshow(y0, cmap='viridis')
                fig.colorbar(im, ax=axs[j,7])
            elif self.phy_dim==1:
                y0 = y[j,:,0].cpu()
                im = axs[j,7].plot(y0)
            loss = torch.norm(x[j,:,0]-y[j,:,0])/torch.norm(y[j,:,0])
            axs[j,7].set_title(f'truth_y, loss {round(loss.item(),4)}')



        plt.tight_layout()
        plt.savefig(save_figure_hidden + 'ep'+str(epoch).zfill(3)+'.png', format='png')
        plt.close()
    

    # def plot_bases(self,num_bases,grid,Nx,Ny,save_figure_bases,epoch):
    #     bases = self.compute_base(grid)  #bsz,N,k
    #     row, col = decompose_integer(num_bases)
    #     fig, axs = plt.subplots(row, col, figsize=(3*col, 3*row)) 
    #     K = bases.shape[-1]
    #     ratio = K//num_bases
    #     for i in range(row): 
    #         for j in range(col):  
    #             k = (i*col+j)*ratio
    #             base_k = bases[0,:,k].detach().cpu().reshape(Nx,Ny)
    #             if self.base_type == 'Gauss':
    #                 weight = ['{:.1f}'.format(w.item()) for w in self.baseweight_Gauss[k,:].detach().cpu().numpy()]
    #             elif self.base_type == 'Fourier':
    #                 weight = ['{:.1f}'.format(w.item()) for w in self.baseweight_Fourier[k,:].detach().cpu().numpy()]
    #             im = axs[i,j].imshow(base_k, cmap='viridis')
    #             axs[i,j].set_title(f'base{k},{weight}')
    #             fig.colorbar(im, ax=axs[i,j])

    #     plt.tight_layout()
    #     plt.savefig(save_figure_bases + 'bases_ep'+str(epoch).zfill(3)+'.png', format='png')
    #     plt.close() 
    
def decompose_integer(n):
    val = int(math.sqrt(n))
    for i in range(val, 0, -1):
        if n % i == 0:
            return (i, n // i)
    return (1, n)

