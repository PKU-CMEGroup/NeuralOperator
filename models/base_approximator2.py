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

class base_approximator2(nn.Module):
    def __init__(self, construct_func , para_list , para_if_train, device = 'cuda'):

        super(base_approximator2, self).__init__()  

        self.all_parameters = [para.to(device) for para in para_list]
        self.trainable_indices = {index: idx for idx, index in enumerate(i for i, trainable in enumerate(para_if_train) if trainable)}
        self.trainable_parameters = nn.ParameterList()
        
        for index in self.trainable_indices:
            self.trainable_parameters.append(nn.Parameter(para_list[index]))

        self.construct_fun = construct_func
    def get_para_list(self):
        para_list = [self.trainable_parameters[self.trainable_indices.get(i, None)] if i in self.trainable_indices else param for i, param in enumerate(self.all_parameters)]
        return para_list
    def forward(self,x,grid):
        # x = Phi*c + y,  Phi^T*y=0
        # Phi^T*x = Phi^T*Phi*c
        # c = (Phi^T*Phi)^-1*Phi^T*x 
        # x_proj = Phi*(Phi^T*Phi)^-1*Phi^T*x 
        #x.shape: bsz,N,1
        para_list = self.get_para_list()
        N = x.shape[1]
        bases_Fourier ,bases_Gauss = self.construct_fun(para_list,grid)
        Phi =  torch.cat((bases_Fourier ,bases_Gauss),dim=-1) #shape: bsz,N,k

        x = x.to(Phi.dtype)

        Q, R = torch.linalg.qr(Phi, mode='reduced')  # Q (bsz, N, k), R (bsz, k, k)

        # alpha = 1e-5 
        # identity = torch.eye(R.size(-1), device=R.device, dtype=R.dtype)[None, :, :]
        # R_reg = R + alpha * identity
        R_reg = R

        Qt_x = torch.bmm(Q.permute(0, 2, 1), x)  # (bsz, k, 1)

        c = torch.linalg.solve_triangular(R_reg, Qt_x, upper=True) 

        x_proj = torch.bmm(Phi,c)
        res = x-x_proj
        return res
    
    def plotGaussbases(self,num_bases,grid,Nx,Ny,save_figure_bases,epoch,additional_plotbases=False):
        para_list = self.get_para_list()
        _ ,bases_Gauss = self.construct_fun(para_list,grid)
        bases = bases_Gauss
        if additional_plotbases:
            row, col = decompose_integer(num_bases+1) 
        else:
            row, col = decompose_integer(num_bases)
        fig, axs = plt.subplots(row, col, figsize=(3*col, 3*row))             
        K = bases.shape[-1]
        ratio = K//num_bases
        for i in range(row): 
            for j in range(col): 
                if additional_plotbases and i==row-1 and j == col-1:
                    k = additional_plotbases-1
                else:
                    k = (i*col+j)*ratio
                base_k = bases[0,:,k].detach().cpu().reshape(Nx,Ny)
                weight = ['{:.1f}'.format(w.item()) for w in para_list[2][k,:].detach().cpu().numpy()]

                im = axs[i,j].imshow(base_k, cmap='viridis')
                axs[i,j].set_title(f'base{k},{weight}')
                fig.colorbar(im, ax=axs[i,j])

        plt.tight_layout()
        plt.savefig(save_figure_bases + 'bases_ep'+str(epoch).zfill(3)+'.png', format='png')
        plt.close() 

    def plotx(self,x,grid,Nx,Ny,save_figure_x,epoch):
        fig, axs = plt.subplots(1, 3, figsize=(15, 5)) 
        res = self.forward(x,grid)
        x_proj = x-res
        x,res,x_proj = x[0,:,0].detach().cpu().reshape(Nx,Ny),res[0,:,0].detach().cpu().reshape(Nx,Ny),x_proj[0,:,0].detach().cpu().reshape(Nx,Ny)
        
        im = axs[0].imshow(x, cmap='viridis')
        axs[0].set_title(f'input')
        fig.colorbar(im, ax=axs[0])

        im = axs[1].imshow(x_proj, cmap='viridis')
        axs[1].set_title(f'proj')
        fig.colorbar(im, ax=axs[1])

        im = axs[2].imshow(res, cmap='viridis')
        axs[2].set_title(f'res')
        fig.colorbar(im, ax=axs[2])

        plt.tight_layout()
        plt.savefig(save_figure_x + 'proj_ep'+str(epoch).zfill(3)+'.png', format='png')
        plt.close()  
    
    def plotGaussbasepts(self,range_pts,save_figure_basespts,epoch):
        para_list = self.get_para_list()
        basepts = para_list[1].detach().cpu().numpy()

        xmin, xmax = range_pts[0][0],range_pts[0][1]
        ymin, ymax = range_pts[1][0],range_pts[1][1]

        in_range_points = basepts[(basepts[:, 0] >= xmin) & (basepts[:, 0] <= xmax) &
                                (basepts[:, 1] >= ymin) & (basepts[:, 1] <= ymax)]
        
        plt.scatter(in_range_points[:, 0], in_range_points[:, 1])
        plt.title(f'Gaussbasepts in [{xmin},{xmax}]*[{ymin},{ymax}]')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_figure_basespts + 'Gaussbasepts_ep' + str(epoch).zfill(3)+ '.png', format='png')
        plt.close()

def decompose_integer(n):
    val = int(math.sqrt(n))
    for i in range(val, 0, -1):
        if n % i == 0:
            return (i, n // i)
    return (1, n)