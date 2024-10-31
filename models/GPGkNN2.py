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

sys.path.append("../")
from mayavi import mlab
import imageio
from PIL import Image, ImageDraw, ImageFont

from .utils import _get_act



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

class GraphGaussconv(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, init_weight,init_freq, device='cuda'):
        super(GraphGaussconv, self).__init__()

        self.fc1 = nn.Linear(in_channels,hidden_channels)
        self.fc2 = nn.Linear(hidden_channels,out_channels)
        self.freq = nn.Parameter(init_freq)  #phy_dim,hidden_channels
        self.weight = nn.Parameter(init_weight)  #hidden_channels
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.device = device


    def forward(self, x, grid, grid_weight, edge_index):
        '''
        x:                          (bsz, in_channel, N)

        grid:                       (bsz, N, phy_dim)

        grid_weight:                (bsz, N)

        edge_index:                 (2, \sum_{n< bsz*N}{k_n})
        '''
        x = x.permute(0,2,1)
        # x = x.to(torch.cfloat)
        x = self.fc1(x)

        bsz,N,phy_dim = grid.shape
        baseweight = self.weight
        basefreq = self.freq.unsqueeze(0)   #1,phy_dim,hidden_channels
        edge_src, edge_dst = edge_index  #num_total_edges

        grid = grid.reshape(-1,phy_dim)   # bsz*N,phy_dim
        grid_weight = grid_weight.reshape(-1)  # bsz*N

        vec_edge = grid[edge_src] - grid[edge_dst]  # num_total_edges,phy_dim
        dist = torch.sqrt(torch.sum((vec_edge)**2,dim=1,keepdim=True)).repeat(1,self.hidden_channels)  # num_total_edges, hidden_channels


        Gauss_edge = torch.sqrt((baseweight/torch.pi)**phy_dim)*torch.exp(-baseweight*dist**2)  # num_total_edges, hidden_channels
        Fourier_edge = torch.imag(torch.exp(torch.sum(1j*basefreq * vec_edge.unsqueeze(-1),dim=1)))  # num_total_edges, hidden_channels
        Morlet_edge = Gauss_edge.to(Fourier_edge.dtype)*Fourier_edge
        # Morlet_edge = Fourier_edge

        weight_edge = grid_weight[edge_src] # num_total_edges
        x_edge = x.reshape(-1, self.hidden_channels)[edge_src]  # num_total_edges,hidden_channels
        value_edge = Morlet_edge * x_edge * weight_edge.unsqueeze(-1) # num_total_edges, hidden_channels

        edge_dst = edge_dst.unsqueeze(-1).repeat(1,self.hidden_channels)
        out = torch.zeros(bsz*N,self.hidden_channels,device = x.device, dtype = value_edge.dtype).scatter_add_(0,edge_dst,value_edge) 
        out = out.reshape(bsz,N,self.hidden_channels)

        # out = torch.real(self.fc2(out)).to(torch.float)
        out = self.fc2(out)

        out = out.permute(0,2,1)

        return out

class GPGkNN2(nn.Module):
    def __init__(self,base_para_list,**config):

        super(GPGkNN2, self).__init__()

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


    def forward(self, x, edge_index):
        '''
        x:                          (bsz, N, in_channel)

        grid:                       (bsz, N, phy_dim)

        grid_weight:                (bsz, N)

        edge_index:                 (2, \sum_{n< bsz*N}{k_n})  with elements in {0,1,...,bsz*N - 1}

        out:                        (bsz, N, out_channel)
        '''

        
        if self.input_with_weight:
            grid = x[:,:,self.in_dim-self.phy_dim:-1]
            grid_weight = x[:,:,-1]
            x = x[:,:,:-1]
        else:
            grid = x[:,:,self.in_dim-self.phy_dim:]
            grid_weight = None
        if not self.local_only:
            bases_g, wbases_g = self.compute_bases_global(grid, grid_weight)
        

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
                x1 = layer_l(x, grid , grid_weight , edge_index)
            else:
                layer_g, layer_l, w, dplayer, ln = items
                x1 = layer_g(x , wbases_g , bases_g) + layer_l(x, grid , grid_weight , edge_index)   
            x2 = w(x)
            x = x1 + x2
            # x = x1
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
        if layer_type == "DGalerkinConv":
            if if_global:
                num_modes = self.num_modes_global
                H = self.H_global
            else:
                num_modes = self.num_modes_local   
                H = self.H_local        
            kernel_modes = self.kernel_mode
            return PhyDGalerkinConv(in_channels, out_channels,  num_modes, kernel_modes,H)
        # elif layer_type == "GPDconv":
        #     assert if_global ==False
        #     num_modes = self.num_modes_local   
        #     H = self.H_local        
        #     kernel_modes = self.kernel_mode
        #     return GPDconv(in_channels, out_channels,  num_modes, kernel_modes,H, self.basepts_Gauss_in, self.baseweight_Gauss_in)
        elif layer_type == "GraphGaussconv":

            return GraphGaussconv(in_channels,self.layer_hidden_dim[index], out_channels, self.init_weight, self.init_freq)
        else:
            raise ValueError("Layer Type Undefined.")


    def construct_base_parameter(self):
        if not self.local_only:
            self.basefreq_Fourier = self.base_para_list[0].to(self.device)  #shape: K1,phy_dim
            k1 = self.basefreq_Fourier.shape[0]
            self.num_modes_global = 2*k1
        if not self.global_only:
            if "GraphGaussconv" in self.layer_types_local:
                if self.base_para_list[1]!=None:
                    self.init_weight = self.base_para_list[1].to(self.device)
                else:
                    self.init_weight = self.init_weight_local*torch.ones(self.layer_hidden_dim[0],device=self.device)
                if self.base_para_list[2]!=None:
                    self.init_freq = self.base_para_list[2].to(self.device)
                else:
                    self.init_freq = self.init_freq_local* 2*torch.pi*first_k_modes(self.phy_dim,self.layer_hidden_dim[0],device=self.device)
            
        
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


    def plot_hidden_layer(self, x, edge_index,y,Nx,Ny,save_figure_hidden,epoch,plot_hidden_layers_num):
        fig, axs = plt.subplots(plot_hidden_layers_num, 9, figsize=(27, 3*plot_hidden_layers_num))
        length = len(self.ws)


        for j in range(plot_hidden_layers_num):
            self.plot_2d(fig,axs, row = j,col = 0,tensor = x[j,:,0],title='input',Nx = Nx,Ny = Ny)


        if self.input_with_weight:
            grid = x[:,:,self.in_dim-self.phy_dim:-1]
            grid_weight = x[:,:,-1]
            x = x[:,:,:-1]
        else:
            grid = x[:,:,self.in_dim-self.phy_dim:]
            grid_weight = None
        if not self.local_only:
            bases_g, wbases_g = self.compute_bases_global(grid, grid_weight)

        

        length = len(self.ws)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)
        x = self.ln0(x.transpose(1,2)).transpose(1,2)
        
        for j in range(plot_hidden_layers_num):
            self.plot_2d(fig,axs,row = j, col = 1,tensor = x[j,0,:],title= 'x0',Nx = Nx,Ny = Ny)


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
                x1 = layer_l(x, grid , grid_weight , edge_index)
            else:
                layer_g, layer_l, w, dplayer, ln = items
                x1 = layer_g(x , wbases_g , bases_g) + layer_l(x, grid , grid_weight , edge_index)   
            x2 = w(x)
            x = x1 + x2
            x = dplayer(x)
            x = ln(x.transpose(1,2)).transpose(1,2)
            if self.act is not None and i != length - 1:
                x = self.act(x)

            for j in range(plot_hidden_layers_num):
                self.plot_2d(fig,axs,row = j, col = i+2,tensor = x[j,0,:],title= f'x{i+1}',Nx = Nx,Ny = Ny)


        x = x.permute(0, 2, 1)

        fc_dim = self.fc_dim 
        
        if fc_dim > 0:
            x = self.fc1(x)
            if self.act is not None:
                x = self.act(x)

        x = self.fc2(x)

        e = y-x.reshape(y.shape)
        for j in range(plot_hidden_layers_num):
            self.plot_2d(fig,axs,row = j, col = 6,tensor = x[j, :, 0],title= 'output',Nx = Nx,Ny = Ny)
            self.plot_2d(fig,axs,row = j, col = 7,tensor = y[j, :, 0],title= 'truth_y',Nx = Nx,Ny = Ny)
            loss = torch.norm(x[j,:,0]-y[j,:,0])/torch.norm(y[j,:,0])
            loss = round(loss.item(),4)
            self.plot_2d(fig,axs,row = j, col = 8,tensor = e[j, :, 0],title= f'error, loss{loss}',Nx = Nx,Ny = Ny)

        plt.tight_layout()
        plt.savefig(save_figure_hidden + 'ep'+str(epoch).zfill(3)+'.png', format='png')
        plt.close()
    
    def plot_2d(self,fig,axs,row,col,tensor,title,Nx=False,Ny=False):
        if self.phy_dim==2:
            assert Nx & Ny
            tensor = tensor.cpu().reshape(Nx,Ny)
            im = axs[row,col].imshow(tensor, cmap='viridis')
            fig.colorbar(im, ax=axs[row,col])
            axs[row,col].set_title(title)
        elif self.phy_dim==1:
            tensor = tensor.cpu()
            im = axs[row,col].plot(tensor)
            axs[row,col].set_title(title)
    
    def plot_hidden_layer_3d(self,x, edge_index,y,save_figure_hidden,epoch,plot_hidden_layers_num):

        assert self.phy_dim ==3
        mlab.options.offscreen = True  # 设置为无头模式
        out = self.forward(x,edge_index)
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

def first_k_modes(dim, k,device = 'cuda'):

    ranges = [torch.arange(-k//dim-1, k//dim + 1,device = device) for _ in range(dim)]
    meshgrid_tensors = torch.meshgrid(*ranges, indexing='ij')
    flat_tensors = [t.flatten() for t in meshgrid_tensors]
    all_pairs = torch.stack(flat_tensors, dim=1)


    abs_sums = torch.sum(torch.abs(all_pairs), dim=1)
    _, indices = torch.topk(-abs_sums, k=k, largest=True, sorted=True)
    selected_pairs = all_pairs[indices]

    result = selected_pairs.T

    return result