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

class base_approximator(nn.Module):
    def __init__(self, construct_func , init_parameter1, init_parameter2,if_train_para1,if_train_para2):

        super(base_approximator, self).__init__()  
        if if_train_para1:
            self.base_para1 = nn.Parameter(init_parameter1)
        else:
            self.base_para1 = init_parameter1.to('cuda')
        if if_train_para2:
            self.base_para2 = nn.Parameter(init_parameter2)
        else:
            self.base_para2 = init_parameter2.to('cuda')
        self.construct_fun = construct_func

    def forward(self,x,grid):
        # x = Phi*c + y,  Phi^T*y=0
        # Phi^T*x = Phi^T*Phi*c
        # c = (Phi^T*Phi)^-1*Phi^T*x 
        # x_proj = Phi*(Phi^T*Phi)^-1*Phi^T*x 
        #x.shape: bsz,N,1
        Phi = self.construct_fun(self.base_para1,self.base_para2,grid)  #shape: bsz,N,k

        x = x.to(Phi.dtype)

        coeff = torch.bmm(Phi.transpose(1,2),x)  # bsz,k,1
        P = torch.bmm(Phi.transpose(1,2),Phi) # bsz,k,k

        alpha = 1e-5

        identity = torch.eye(P.size(-1), device=P.device, dtype=P.dtype)[None, :, :]
        P_reg = P + alpha * identity
        P_reg = torch.linalg.inv(P_reg)
        
        c = torch.bmm(P_reg,coeff)  #bsz,k,1



        # # QR 分解 Phi
        # Q, R = torch.linalg.qr(Phi, mode='reduced')  # Q 形状: (bsz, N, k), R 形状: (bsz, k, k)

        # alpha = 1e-5  # 正则化参数
        # identity = torch.eye(R.size(-1), device=R.device, dtype=R.dtype)[None, :, :]
        # R_reg = R + alpha * identity
        # # 计算 Q^T * x
        # Qt_x = torch.bmm(Q.permute(0, 2, 1), x)  # 结果形状: (bsz, k, 1)

        # # 使用 linalg.solve_triangular 来解决上三角矩阵的线性方程组
        # c = torch.linalg.solve_triangular(R_reg, Qt_x, upper=True)  # 解决最小二乘问题得到 c

        x_proj = torch.bmm(Phi,c)
        res = x-x_proj
        return res
    
    def plotbases(self,num_bases,weight_pos,grid,Nx,Ny,save_figure_bases,epoch,additional_plotbases=False):
        base = self.construct_fun(self.base_para1,self.base_para2,grid)  #bsz,N,k
        if additional_plotbases:
            row, col = decompose_integer(num_bases+1) 
        else:
            row, col = decompose_integer(num_bases)
        fig, axs = plt.subplots(row, col, figsize=(3*col, 3*row))             
        K = base.shape[-1]
        ratio = K//num_bases
        for i in range(row): 
            for j in range(col): 
                if additional_plotbases and i==row-1 and j == col-1:
                    k = additional_plotbases-1
                else:
                    k = (i*col+j)*ratio
                base_k = base[0,:,k].detach().cpu().reshape(Nx,Ny)
                if weight_pos==1:
                    weight = ['{:.1f}'.format(w.item()) for w in self.base_para1[k%self.base_para1.shape[0],:].detach().cpu().numpy()]
                if weight_pos==2:
                    weight = ['{:.1f}'.format(w.item()) for w in self.base_para2[k%self.base_para2.shape[0],:].detach().cpu().numpy()]
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
        basepts = self.base_para1.detach().cpu().numpy()

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