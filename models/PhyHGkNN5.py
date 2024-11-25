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
from mayavi import mlab
import imageio
from PIL import Image, ImageDraw, ImageFont
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

class PhyHGkNN5(nn.Module):
    def __init__(self,base_para_list,**config):

        super(PhyHGkNN5, self).__init__()

        self.base_para_list = base_para_list
        self.config = defaultdict(lambda: None, **config)
        self.config = dict(self.config)
        all_attr = list(self.config.keys())
        for key in all_attr:
            setattr(self, key, self.config[key])

        self.construct_base_parameter()
        self.construct_H()
        if not self.local_only:
            self.sp_layers_global = nn.ModuleList(
                [
                    self._choose_layer(index, in_size, out_size, layer_type, True)
                    for index, (in_size, out_size, layer_type) in enumerate(
                        zip(self.layers_dim, self.layers_dim[1:], self.layer_types_global)
                    )
                ]
            )
        if not self.global_only:
            self.sp_layers_local = nn.ModuleList(
                [
                    self._choose_layer(index, in_size, out_size, layer_type, False)
                    for index, (in_size, out_size, layer_type) in enumerate(
                        zip(self.layers_dim, self.layers_dim[1:], self.layer_types_local)
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
            grid = x[:,:,self.in_dim-self.phy_dim:-1]
            grid_weight = x[:,:,-1]
            x = x[:,:,:-1]
        else:
            grid = x[:,:,self.in_dim-self.phy_dim:]
            grid_weight = None
        if not self.local_only:
            bases_g, wbases_g = self.compute_bases_global(grid, grid_weight)
        if not self.global_only:
            bases_l, wbases_l = self.compute_bases_local(grid, grid_weight)

        length = len(self.ws)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)
        x = self.ln0(x.transpose(1,2)).transpose(1,2)

        if self.global_only:
            sequence = zip(self.sp_layers_global, self.ws, self.dropout_layers, self.ln_layers)
        elif self.local_only:
            sequence = zip(self.sp_layers_local, self.ws, self.dropout_layers, self.ln_layers)
        else:
            sequence = zip(self.sp_layers_global, self.sp_layers_local, self.ws, self.dropout_layers, self.ln_layers)

        for i, items  in enumerate(sequence):
            if self.global_only:
                layer_g, w, dplayer, ln = items
                x1 = layer_g(x , wbases_g , bases_g)
            elif self.local_only:
                layer_l, w, dplayer, ln = items
                x1 = layer_l(x , wbases_l , bases_l)
            else:
                layer_g, layer_l, w, dplayer, ln = items
                x1 = layer_g(x , wbases_g , bases_g) + layer_l(x , wbases_l , bases_l)   
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
    

    def _choose_layer(self, index, in_channels, out_channels, layer_type, if_global):
        if layer_type == "HGalerkinConv":
            if if_global:
                num_modes_in = self.num_modes_global
                num_modes_out = self.num_modes_global
                H = self.H_global
            else:
                num_modes_in = self.num_modes_local
                num_modes_out = self.num_modes_local
                H = self.H_local
            kernel_modes = self.kernel_mode
            return PhyHGalerkinConv(in_channels, out_channels, num_modes_in, num_modes_out, kernel_modes, H)
        if layer_type == "DGalerkinConv":
            if if_global:
                num_modes = self.num_modes_global
                H = self.H_global
            else:
                num_modes = self.num_modes_local   
                H = self.H_local        
            kernel_modes = self.kernel_mode
            return PhyDGalerkinConv(in_channels, out_channels,  num_modes, kernel_modes,H)
        else:
            raise ValueError("Layer Type Undefined.")


    def construct_base_parameter(self):
        if not self.local_only:
            self.basefreq_Fourier = self.base_para_list[0].to(self.device)  #shape: K1,phy_dim
            k1 = self.basefreq_Fourier.shape[0]
            self.num_modes_global = 2*k1
        if not self.global_only:
            if self.local_bases_type =='Gauss':
                self.basepts_Gauss_in = self.base_para_list[1].to(self.device)  #shape: K2,phy_dim
                self.baseweight_Gauss_in = self.base_para_list[2].to(self.device)  #shape: K2,phy_dim
                if self.train_local_out:
                    self.basepts_Gauss_out = nn.Parameter(self.base_para_list[1].to(self.device))  #shape: K2,phy_dim
                    self.baseweight_Gauss_out = nn.Parameter(self.base_para_list[2].to(self.device))  #shape: K2,phy_dim
                else:
                    self.basepts_Gauss_out = self.basepts_Gauss_in
                    self.baseweight_Gauss_out = self.baseweight_Gauss_in
                k2 = self.baseweight_Gauss_in.shape[0]
                self.num_modes_local = k2
            elif self.local_bases_type =='Morlet':
                self.basepts_Morlet = self.base_para_list[1].to(self.device)  #shape: K2,phy_dim
                self.baseweight_Morlet = self.base_para_list[2].to(self.device)  #shape: K2,phy_dim
                self.basefreq_Morlet = self.base_para_list[3].to(self.device)  #shape: K2,phy_dim
                k2 = self.basepts_Morlet.shape[0]
                self.num_modes_local = 2*k2


    def compute_bases_local(self, grid, gridweight):
        N = grid.shape[1]
        if gridweight != None:
            gridweight = gridweight.unsqueeze(-1)
            if self.local_bases_type =='Gauss':
                bases_Gauss = self.compute_basesout_Gauss(grid) #shape: bsz,N,K2
                if not self.train_local_out:
                        wbases_Gauss = bases_Gauss * gridweight
                else:
                    wbases_Gauss = self.compute_basesin_Gauss(grid) * gridweight
                return bases_Gauss, wbases_Gauss
            elif self.local_bases_type =='Morlet':
                bases_Morlet = self.compute_bases_Morlet(grid) #shape: bsz,N,K2
                wbases_Morlet = bases_Morlet * gridweight
                return bases_Morlet, wbases_Morlet
        else:
            if self.local_bases_type =='Gauss':
                bases_Gauss = self.compute_basesout_Gauss(grid) #shape: bsz,N,K2
                if not self.train_local_out:
                    wbases_Gauss = bases_Gauss/N
                else:
                    wbases_Gauss = self.compute_basesin_Gauss(grid) * gridweight
                return bases_Gauss, wbases_Gauss
            elif self.local_bases_type =='Morlet':
                bases_Morlet = self.compute_bases_Morlet(grid) #shape: bsz,N,K2
                wbases_Morlet = bases_Morlet/N
                return bases_Morlet, wbases_Morlet
            
        
    def compute_bases_global(self, grid, gridweight):
        N = grid.shape[1]
        bases_Fourier = self.compute_bases_Fourier(grid)   #shape: bsz,N,2*K1
        if gridweight != None:
            gridweight = gridweight.unsqueeze(-1)
            wbases_Fourier = bases_Fourier * gridweight
        else:
            wbases_Fourier = bases_Fourier/N
        return bases_Fourier ,wbases_Fourier 


    def compute_bases_Fourier(self,grid):
        #grid.shape:  bsz,N,phy_dim
        grid = grid.unsqueeze(2) #bsz,N,1,phy_dim
        basefreq = self.basefreq_Fourier.unsqueeze(0).unsqueeze(0)  #1,1,K1,phy_dim
        bases_complex = torch.exp(torch.sum(1j*basefreq*grid,dim=3))  #bsz,N,K1,phy_dim-->bsz,N,K1
        bases = torch.cat((torch.real(bases_complex),torch.imag(bases_complex)),dim=-1)  #bsz,N,2*K1
        bases = bases*math.sqrt(bases.shape[1])/(torch.norm(bases, p=2, dim=1, keepdim=True) + 1e-5)
        return bases     
    
    def compute_basesout_Gauss(self,grid):
        #grid.shape:  bsz,N,phy_dim
        grid = grid.unsqueeze(2) #bsz,N,1,phy_dim
        basepts = self.basepts_Gauss_out.unsqueeze(0).unsqueeze(0) # 1,1,K2,phy_dim
        baseweight = torch.abs(self.baseweight_Gauss_out).unsqueeze(0).unsqueeze(0)  #1,1,K2,phy_dim
        # bases = torch.sqrt(torch.prod(baseweight, dim=3))*torch.sum((grid-basepts)**2,dim=3)*torch.exp(-1*torch.sum(baseweight*(grid-basepts)**2,dim=3))  #bsz,N,K2,phy_dim-->bsz,N,K2
        bases = torch.sqrt(torch.prod(baseweight, dim=3))*torch.exp(-1*torch.sum(baseweight*(grid-basepts)**2,dim=3))  #bsz,N,K2,phy_dim-->bsz,N,K2  
        bases = bases*math.sqrt(bases.shape[1])/(torch.norm(bases, p=2, dim=1, keepdim=True)+1e-5)
        return bases
    
    def compute_basesin_Gauss(self,grid):
        grid = grid.unsqueeze(2) #bsz,N,1,phy_dim
        basepts = self.basepts_Gauss_in.unsqueeze(0).unsqueeze(0) # 1,1,K2,phy_dim
        baseweight = torch.abs(self.baseweight_Gauss_in).unsqueeze(0).unsqueeze(0)  #1,1,K2,phy_dim
        bases = torch.sqrt(torch.prod(baseweight, dim=3))*torch.exp(-1*torch.sum(baseweight*(grid-basepts)**2,dim=3))  #bsz,N,K2,phy_dim-->bsz,N,K2 
        bases = bases*math.sqrt(bases.shape[1])/(torch.norm(bases, p=2, dim=1, keepdim=True)+1e-5)
        return bases
    
    def compute_bases_Morlet(self,grid):
        grid = grid.unsqueeze(2) #bsz,N,1,phy_dim
        basepts = self.basepts_Morlet.unsqueeze(0).unsqueeze(0) # 1,1,K2,phy_dim
        baseweight = torch.abs(self.baseweight_Morlet).unsqueeze(0).unsqueeze(0)  #1,1,K2,phy_dim
        basefreq = self.basefreq_Morlet.unsqueeze(0).unsqueeze(0) #1,1,K2,phy_dim
        b1 = torch.exp(torch.sum(1j*basefreq*grid,dim=3))  #bsz,N,K1,phy_dim-->bsz,N,K1
        b = torch.sqrt(torch.prod(baseweight, dim=3))*torch.exp(-1*torch.sum(baseweight*(grid-basepts)**2,dim=3))  #bsz,N,K1,phy_dim-->bsz,N,K1
        bases_complex = b1*b  #bsz,N,K1
        bases = torch.cat((torch.real(bases_complex),torch.imag(bases_complex)),dim=-1)  #bsz,N,2*K1
        bases = bases*math.sqrt(bases.shape[1])/(torch.norm(bases, p=2, dim=1, keepdim=True) + 1e-5)
        return bases 
    
    def construct_H(self):
        if not self.local_only:
            if 'HGalerkinConv' in self.layer_types_global:
                scale = 1/(self.num_modes_global*self.num_modes_global)
                self.H_global = nn.Parameter(
                        scale
                        * torch.rand(self.kernel_mode, self.num_modes_global, self.num_modes_global, dtype=torch.float)
                    )
            if 'DGalerkinConv' in self.layer_types_global:
                scale = 1/self.num_modes_global
                self.H_global = nn.Parameter(
                        scale
                        * torch.rand(self.kernel_mode, self.num_modes_global, dtype=torch.float)
                    )
        if not self.global_only:
            if 'HGalerkinConv' in self.layer_types_local:
                scale = 1/(self.num_modes_local*self.num_modes_local)
                self.H_local = nn.Parameter(
                        scale
                        * torch.rand(self.kernel_mode, self.num_modes_local, self.num_modes_local, dtype=torch.float)
                    )
            if 'DGalerkinConv' in self.layer_types_local:
                scale = 1/self.num_modes_local
                self.H_local = nn.Parameter(
                        scale
                        * torch.rand(self.kernel_mode, self.num_modes_local, dtype=torch.float)
                    )

    def plot_hidden_layer(self,x,y,Nx,Ny,save_figure_hidden,epoch,plot_hidden_layers_num):
        fig, axs = plt.subplots(plot_hidden_layers_num, 9, figsize=(27, 3*plot_hidden_layers_num))
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

        if not self.local_only:
            bases_g, wbases_g = self.compute_bases_global(grid, grid_weight)
        if not self.global_only:
            bases_l, wbases_l = self.compute_bases_local(grid, grid_weight)


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


        if self.global_only:
            sequence = zip(self.sp_layers_global, self.ws, self.dropout_layers, self.ln_layers)
        elif self.local_only:
            sequence = zip(self.sp_layers_local, self.ws, self.dropout_layers, self.ln_layers)
        else:
            sequence = zip(self.sp_layers_global, self.sp_layers_local, self.ws, self.dropout_layers, self.ln_layers)

        for i, items  in enumerate(sequence):
            if self.global_only:
                layer_g, w, dplayer, ln = items
                x1 = layer_g(x , wbases_g , bases_g)
            elif self.local_only:
                layer_l, w, dplayer, ln = items
                x1 = layer_l(x , wbases_l , bases_l)
            else:
                layer_g, layer_l, w, dplayer, ln = items
                x1 = layer_g(x , wbases_g , bases_g) + layer_l(x , wbases_l , bases_l)   
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

            e = y-x.reshape(y.shape)
            if self.phy_dim==2:
                e0 = e[j,:,0].cpu().reshape(Nx,Ny)
                im = axs[j,8].imshow(e0, cmap='viridis')
                fig.colorbar(im, ax=axs[j,8])
            elif self.phy_dim==1:
                e0 = e[j,:,0].cpu()
                im = axs[j,8].plot(e0)
            norm = torch.norm(e[j,:,0])
            axs[j,8].set_title(f'error, norm {round(norm.item(),4)}')



        plt.tight_layout()
        plt.savefig(save_figure_hidden + 'ep'+str(epoch).zfill(3)+'.png', format='png')
        plt.close()
    
    def plot_hidden_layer_3d(self,x,y,save_figure_hidden,epoch,plot_hidden_layers_num):
        assert self.phy_dim ==3
        mlab.options.offscreen = True  # 设置为无头模式
        out = self.forward(x)
        images = []
        font_path = 'arial.ttf'  # 字体文件路径，或者使用系统字体
        text_color = (0, 0, 0)  # 文本颜色为黑色
        text_position = (10, 400)  # 文本的位置，在图像底部
        if self.input_with_weight:
            x = x[:,:,:-1]
        for j in range(plot_hidden_layers_num):

            x1, y1, z1 = x[j].transpose(0,1).detach().cpu().numpy()
            u1, v1, w1 = y[j].transpose(0,1).detach().cpu().numpy()
            fig1 = mlab.figure(size=(500,500), bgcolor=(1, 1, 1))
            mlab.view(azimuth=0, elevation=70, focalpoint=[0, 0, 0], distance=10, roll=0, figure=fig1)
            mlab.quiver3d(x1, y1, z1, u1, v1, w1, mode='arrow', color=(0, 1, 0), scale_factor=0.075, figure=fig1)
            img1 = mlab.screenshot(figure=fig1)
            img1_with_text = add_label(img1, f"Truth{j}", font_path, text_color, text_position)

            u2, v2, w2 = out[j].transpose(0,1).detach().cpu().numpy()
            fig2 = mlab.figure(size=(500,500), bgcolor=(1, 1, 1))
            mlab.view(azimuth=0, elevation=70, focalpoint=[0, 0, 0], distance=10, roll=0, figure=fig2)
            mlab.quiver3d(x1, y1, z1, u2, v2, w2, mode='arrow', color=(0, 1, 0), scale_factor=0.075, figure=fig2)
            img2 = mlab.screenshot(figure=fig2)
            img2_with_text = add_label(img2,f"Output{j}", font_path, text_color, text_position)

            
            fig3 = mlab.figure(size=(500,500), bgcolor=(1, 1, 1))
            e1 = torch.sum((out[j]-y[j])**2,dim=-1).detach().cpu().numpy()
            mlab.view(azimuth=0, elevation=70, focalpoint=[0, 0, 0], distance=10, roll=0, figure=fig3)
            mlab.points3d(x1, y1, z1, e1, color=(1, 0, 0),mode = 'sphere', scale_factor=0.075, figure=fig3)
            img3 = mlab.screenshot(figure=fig3)
            loss = round(torch.norm(out[j]-y[j]).item()/torch.norm(y[j]).item(),4)
            img3_with_text = add_label(img3,f"Error{j}, loss{loss}", font_path, text_color, text_position)

            combined_image = np.hstack([img1_with_text, img2_with_text,img3_with_text])
            images.append(combined_image)

            mlab.close(fig1)
            mlab.close(fig2)
            mlab.close(fig3)


        final_image = np.vstack(images)
        filename = save_figure_hidden+ 'ep'+str(epoch).zfill(3)+ '.png'
        imageio.imwrite(filename, final_image)
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

def add_label(image, text, font_path, text_color, position):
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)
    font = ImageFont.truetype(font_path, 25)  # 调整字体大小
    
    # # # 计算文本宽度以居中文本
    # text_width, text_height = draw.textsize(text, font=font)
    position = ((pil_image.width - 250) // 2, position[1])
    
    draw.text(position, text, fill=text_color, font=font)
    return np.array(pil_image)