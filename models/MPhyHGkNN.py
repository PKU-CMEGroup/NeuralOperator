import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
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



class PhyHGalerkinConv(nn.Module):
    def __init__(self, in_channels, out_channels, modes_in, modes_out, kernel_modes,  bases, wbases, H):
        super(PhyHGalerkinConv, self).__init__()


        self.in_channels = in_channels
        self.out_channels = out_channels
        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes_in = modes_in
        self.modes_out = modes_out
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

        x_hat = torch.einsum("bcx,bxk->bck", x, wbases[:,:,:self.modes_in])

        # Multiply relevant Fourier modes
        x_hat = mycompl_mul1d(self.weights, H , x_hat)

        # Return to physical space
        x = torch.einsum("bck,xk->bcx", x_hat, bases[:,:self.modes_out])
        # x = torch.einsum("bck,bxk->bcx", x_hat, bases[:,:,:self.modes_out])
        return x


class MPhyHGalerkinConv(nn.Module):
    def __init__(self, in_channels_list, out_channels_list, modes_in_list, modes_out_list, kernel_modes_list,  bases_list, wbases_list, H_list):
        super(MPhyHGalerkinConv, self).__init__()

        self.in_channels_list = in_channels_list
        self.out_channels_list = out_channels_list
        self.modes_in_list = modes_in_list
        self.modes_out_list = modes_out_list
        self.kernel_modes_list = kernel_modes_list
        self.bases_list = bases_list
        self.wbases_list = wbases_list
        self.H_list = H_list
        self.depth = len(in_channels_list)
        self.layers = nn.ModuleList(
            [
                PhyHGalerkinConv(in_channels_list[i], out_channels_list[i], modes_in_list[i], modes_out_list[i], kernel_modes_list[i],  bases_list[i], wbases_list[i], H_list[i])
                for i in range(self.depth)
            ]
        )

    def forward(self, x):
        x_slices = []
        start_idx = 0
        for i in range(self.depth):
            end_idx = start_idx + self.in_channels_list[i]
            x_slices.append(x[:, start_idx:end_idx, :])
            start_idx = end_idx
        outputs = [layer(xi) for (layer,xi) in zip(self.layers,x_slices)]
        x = torch.cat(outputs, dim=1)
        return x



class MPhyHGkNN(nn.Module):
    def __init__(self,bases_out,**config):

        super(MPhyHGkNN, self).__init__()
        self.bases_out=bases_out
        self.device = bases_out.device
        self.config = defaultdict(lambda: None, **config)
        self.config = dict(self.config)
        all_attr = list(self.config.keys())
        for key in all_attr:
            setattr(self, key, self.config[key])

        all_attr_layer = list(self.layer.keys())
        for key in all_attr_layer:
            setattr(self, key, self.layer[key])

        self.bases_list = [self.bases_out for i in range(self.depth)]
        self.wbases_list = [self.bases_out.clone() for i in range(self.depth)]
        self.num_basepts = torch.round(torch.pow(torch.tensor(self.num_basepts_per_dim),self.phy_dim)).long()
        self.construst_bases()
        self.construct_H()

        self.sp_layers = nn.ModuleList(
            [
                self._choose_layer(layer_type)
                for layer_type in self.layer_types
                
            ]
        )
        #######
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
        self.update_bases_list(x)


        length = len(self.ws)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)
        x = self.ln0(x.transpose(1,2)).transpose(1,2)


        for i, (layer , w, dplayer,ln) in enumerate(zip(self.sp_layers, self.ws, self.dropout_layers,self.ln_layers)):
            x1 = layer(x)
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


    def _choose_layer(self, layer_type):
        if layer_type == "MPhyHGalerkinConv":
            in_channels_list = self.dim_list
            out_channels_list = self.dim_list
            num_modes_in_list = self.num_basepts
            num_modes_out_list = self.GkNN_mode_out_list
            kernel_modes_list = self.kernel_mode_list
            bases_list = self.bases_list
            wbases_list = self.wbases_list
            
            return MPhyHGalerkinConv(in_channels_list, out_channels_list, num_modes_in_list, num_modes_out_list, kernel_modes_list, bases_list, wbases_list, self.H_list)
        else:
            raise ValueError("Layer Type Undefined.")
    


    def construst_bases(self):
        #create self.basepts_list,self.baseweight_list,self.bases_threhold_list
        depth = self.depth
        basepts_list = []
        baseweight_list = []
        bases_threhold_list = []
        for k in range(depth):
            num_basepts = self.num_basepts[k]
            # basepts = nn.Parameter(torch.rand(num_basepts, self.phy_dim, dtype = torch.float))        
            basepts = self.uniform_points(num_basepts, self.phy_dim,self.range_pts_list[k])
            basepts = basepts.to(self.device)
            baseweight = nn.Parameter(2*self.averageweight_list[k]*torch.rand(num_basepts, self.phy_dim, dtype = torch.float))
            # self.bases_threhold = nn.Parameter(1*torch.rand(self.num_basepts, dtype = torch.float))
            bases_threhold = torch.ones(num_basepts).to(self.device)    
            basepts_list.append(basepts)
            baseweight_list.append(baseweight)
            bases_threhold_list.append(bases_threhold)
        self.basepts_list = basepts_list
        self.baseweight_list = baseweight_list
        self.bases_threhold_list = bases_threhold_list


    def construct_H(self):
        #create H_list
        depth = self.depth
        H_list = []
        for k in range(depth):
            num_basepts = self.num_basepts[k]
            scale = 1/(num_basepts*self.GkNN_mode_out_list[k])
            H = nn.Parameter(
                    scale
                    * torch.rand(self.kernel_mode_list[k],self.GkNN_mode_out_list[k], num_basepts,  dtype=torch.float)
                )
            H_list.append(H)
        self.H_list = H_list


    def update_bases_list(self,x):
        bases_list = self.compute_bases_list(x[:,:,self.in_dim-self.phy_dim:])
        bases_list = [F.relu(bases-bases_threhold.unsqueeze(0).unsqueeze(0)) for (bases,bases_threhold) in zip(bases_list,self.bases_threhold_list)]
        # bases = bases-self.bases_threhold.unsqueeze(0).unsqueeze(0)
        # bases = bases - torch.sum(bases,dim=1).unsqueeze(1)/n
        bases_list = [bases/torch.sqrt(torch.sum(bases**2,dim=1)).unsqueeze(1) for bases in bases_list]
        for sp_layer in self.sp_layers:
            self.wbases_list = [bases/(x.shape[1]) for bases in bases_list]
            for k in range(self.depth):
                sp_layer.layers[k].wbases = self.wbases_list[k]



    def compute_bases_list(self,x_phy_in):
        #x_phy_in.shape:  bsz,n,phy_in_channel
        x_phy_in = x_phy_in.unsqueeze(2) #bsz,n,1,phy_in_channel
        depth = self.depth
        x_phy_out_list = []
        for k in range(depth):
            num_pts = self.basepts_list[k].shape[0] 
            dim = self.basepts_list[k].shape[1]
            range_pts = torch.tensor(self.range_pts_list[k]).to(self.basepts_list[k].device)
            range_pts_min = range_pts[:, 0].unsqueeze(0).expand(num_pts, dim) 
            range_pts_max = range_pts[:, 1].unsqueeze(0).expand(num_pts, dim) 
            basepts = torch.clamp(self.basepts_list[k], min=range_pts_min, max=range_pts_max)
            basepts = basepts.unsqueeze(0).unsqueeze(0) # 1,1,phy_out_channel,phy_in_channel
            baseweight = self.baseweight_list[k].unsqueeze(0).unsqueeze(0)  #1,1,phy_out_channel,phy_in_channel
            baseweight = baseweight + self.minweight_list[k]
            baseweight = baseweight.to(self.device)
            x_phy_out = torch.sqrt(torch.prod(baseweight, dim=3))*torch.exp(-1*torch.sum(baseweight*(x_phy_in-basepts)**2,dim=3))  
            #bsz,n,phy_out_channel,phy_in_channel-->bsz,n,phy_out_channel
            x_phy_out_list.append(x_phy_out)
            # print(x_phy_out.shape)
        return x_phy_out_list


    def uniform_points(self,num_pts,dim,range_pts):
        a = int(math.pow(num_pts, 1 / dim))
        index_tensors = []
        for k in range(dim):
            xmin,xmax = range_pts[k][0],range_pts[k][1]
            idx = xmin + (xmax-xmin)*torch.arange(a).float().add(0.5).div(a)
            idx = idx.view((1,) * k+ (-1,) + (1,) * (dim - k - 1))
            index_tensors.append(idx.expand(a, *([a] * (dim - 1))))
        num_pts1 = int(torch.pow(torch.tensor(a),dim))
        x = torch.stack(index_tensors, dim=dim).reshape(num_pts1,dim)
        return x