import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from collections import defaultdict
import sys
import time

sys.path.append("../")
from .basics import (
    compl_mul1d,
    SpectralConv1d,
    SpectralConv2d_shape,
    SimpleAttention,
)
from .utils import _get_act, add_padding, remove_padding


class TimeTracker:
    def __init__(self):
        self.timestamps = []

    def record(self, label=""):
        timestamp = time.time()
        self.timestamps.append((timestamp, label))
        if len(self.timestamps) > 1:
            prev_timestamp, _ = self.timestamps[-2]
            current_timestamp, _ = self.timestamps[-1]
            print(f"Time between {label} and previous point: {current_timestamp - prev_timestamp:.4f} seconds")
        
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
    #D.shaoe: kernel_mode,modes_out
    #product.shape: modes_out,modes_in
    J = D.shape[0]
    #product.shape: bsz,K,L
    K = product.shape[0]
    L = product.shape[1]
    # print(f'bsz={bsz}')
    H = D.reshape(J,K,1)*product.reshape(1,K,L)
    return H


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
        if len(bases.shape)==2:
            x = torch.einsum("bck,xk->bcx", x_hat, bases[:,:self.modes_out])
        elif len(bases.shape)==3:
            x = torch.einsum("bck,bxk->bcx", x_hat, bases[:,:,:self.modes_out])
        return x

class PhyDGalerkinConv(nn.Module):
    def __init__(self, in_channels, out_channels, modes, kernel_modes,  bases, wbases, D):
        super(PhyDGalerkinConv, self).__init__()

        """
        1D Spectral layer. It avoids FFT, but utilizes low rank approximation. 
        
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes = modes
        self.bases = bases
        self.wbases = wbases
        self.scale1 = 1 / kernel_modes
        self.D = D
            

        self.scale2 = 1 / (in_channels * out_channels)
        self.kernel_modes = kernel_modes

        self.weights = nn.Parameter(
            self.scale2
            * torch.randn(in_channels, out_channels, self.kernel_modes, dtype=torch.float)
        )
                
    def forward(self, x):
        bases, wbases = self.bases, self.wbases

        # Compute coeffcients

        x_hat = torch.einsum("bcx,bxk->bck", x, wbases)
        

        # Multiply relevant Fourier modes
        x_hat = mycompl_mul1d_D(self.weights, self.D , x_hat)

        # Return to physical space
        if len(bases.shape)==2:
            x = torch.einsum("bck,xk->bcx", x_hat, bases[:,:self.modes])
        elif len(bases.shape)==3:
            x = torch.einsum("bck,bxk->bcx", x_hat, bases[:,:,:self.modes])

        return x

class HGalerkinConv(nn.Module):
    def __init__(self, in_channels, out_channels, modes_in, modes_out, kernel_modes,  bases, wbases, H):
        super(HGalerkinConv, self).__init__()


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

        x_hat = torch.einsum("bcx,xk->bck", x, wbases[:,:self.modes_in])
        

        # Multiply relevant Fourier modes
        x_hat = mycompl_mul1d(self.weights, H , x_hat)

        # Return to physical space
        x = torch.einsum("bck,xk->bcx", x_hat, bases[:,:self.modes_out])

        return x

class DGalerkinConv(nn.Module):
    def __init__(self, in_channels, out_channels, modes, kernel_modes,  bases, wbases, D):
        super(DGalerkinConv, self).__init__()

        """
        1D Spectral layer. It avoids FFT, but utilizes low rank approximation. 
        
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes = modes
        self.bases = bases
        self.wbases = wbases
        self.scale1 = 1 / kernel_modes
        self.D = D
            

        self.scale2 = 1 / (in_channels * out_channels)
        self.kernel_modes = kernel_modes

        self.weights = nn.Parameter(
            self.scale2
            * torch.randn(in_channels, out_channels, self.kernel_modes, dtype=torch.float)
        )
                
    def forward(self, x):
        bases, wbases = self.bases, self.wbases

        # Compute coeffcients

        x_hat = torch.einsum("bcx,xk->bck", x, wbases)
        

        # Multiply relevant Fourier modes
        x_hat = mycompl_mul1d_D(self.weights, self.D , x_hat)

        # Return to physical space
        x = torch.einsum("bck,xk->bcx", x_hat, bases)

        return x

class newDGalerkinConv(nn.Module):
    def __init__(self, in_channels, out_channels, modes, kernel_modes,  bases, wbases, D):
        super(newDGalerkinConv, self).__init__()

        """
        1D Spectral layer. It avoids FFT, but utilizes low rank approximation. 
        
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes = modes
        self.bases = bases
        self.wbases = wbases
        self.scale1 = 1 / kernel_modes
        self.D = D
        self.product = torch.mm(self.bases.transpose(0,1),self.wbases)  #shape: K,L
            

        self.scale2 = 1 / (in_channels * out_channels)
        self.kernel_modes = kernel_modes

        self.weights = nn.Parameter(
            self.scale2
            * torch.randn(in_channels, out_channels, self.kernel_modes, dtype=torch.float)
        )
                
    def forward(self, x):
        bases, wbases,product= self.bases, self.wbases,self.product

        # Compute coeffcients

        x_hat = torch.einsum("bcx,xk->bck", x, wbases)
        
        H = compute_H_D(self.D,product)
        # Multiply relevant Fourier modes
        x_hat = mycompl_mul1d(self.weights, H , x_hat)

        # Return to physical space
        x = torch.einsum("bck,xk->bcx", x_hat, bases)

        return x
class compositeHGalerkinConv(nn.Module):
    def __init__(self, type1, type2, in_channels, out_channels, modes_in1, modes_out1,kernel_modes1,modes_in2, modes_out2, kernel_modes2,  bases1, wbases1, H1,bases2, wbases2,H2):
        super(compositeHGalerkinConv, self).__init__()

        if type1 =='D':
            self.layer_HGkNN = DGalerkinConv(in_channels, out_channels, modes_in1, kernel_modes1,  bases1, wbases1, H1)
        elif type1 == 'H':
            self.layer_HGkNN = HGalerkinConv(in_channels, out_channels, modes_in1, modes_out1, kernel_modes1,  bases1, wbases1, H1)
        elif type1 == 'newD':
            self.layer_HGkNN = newDGalerkinConv(in_channels, out_channels, modes_in1, kernel_modes1,  bases1, wbases1, H1)
        else:
            raise ValueError("Layer Type Undefined.")
        if type2 =='D':
            self.layer_PhyHGkNN = PhyDGalerkinConv(in_channels, out_channels, modes_in2, kernel_modes2,  bases2, wbases2, H2)
        elif type2 =='H':
            self.layer_PhyHGkNN = PhyHGalerkinConv(in_channels, out_channels, modes_in2, modes_out2, kernel_modes2,  bases2, wbases2, H2)
        else:
            raise ValueError("Layer Type Undefined.")

    def forward(self, x):
        #x.shape: bsz,channel,N
        x1 = self.layer_HGkNN(x)
        x2 = self.layer_PhyHGkNN(x)
        x = x1 + x2
        return x

class PhyHGkNN2(nn.Module):
    def __init__(self,bases_list,**config):

        super(PhyHGkNN2, self).__init__()
        self.bases1 = bases_list[0]
        self.wbases1 = bases_list[1]
        self.bases2 = bases_list[2]
        self.wbases2 = bases_list[3]

            
        self.config = defaultdict(lambda: None, **config)
        self.config = dict(self.config)
        all_attr = list(self.config.keys())
        for key in all_attr:
            setattr(self, key, self.config[key])

        all_attr_layer = list(self.layer.keys())
        for key in all_attr_layer:
            setattr(self, key, self.layer[key])
        
        
        # self.basepts = nn.Parameter(torch.rand(self.num_basepts, self.phy_dim, dtype = torch.float))        
        basepts = self.uniform_points(self.num_basepts, self.phy_dim)
        if self.pts_type == 'trained':
            self.basepts = nn.Parameter(basepts)
        elif self.pts_type == 'fixed':
            self.basepts = basepts.to('cuda')

        if self.baseweight_type =='parameter':
            self.baseweight = nn.Parameter(self.baseradius*torch.rand(self.num_basepts, self.phy_dim, dtype = torch.float))
            if self.if_bases_threhold:
                self.bases_threhold = nn.Parameter(1*torch.rand(self.num_basepts, dtype = torch.float))
            else:
                self.bases_threhold = torch.zeros(self.num_basepts, dtype = torch.float,device = self.bases1.device)
        elif self.baseweight_type =='same':
            self.baseweight = torch.zeros(self.num_basepts, self.phy_dim, dtype = torch.float,  device = self.bases1.device)
            if self.if_bases_threhold:
                self.bases_threhold = torch.ones(self.num_basepts, dtype = torch.float,device = self.bases1.device)
            else:
                self.bases_threhold = torch.zeros(self.num_basepts, dtype = torch.float,device = self.bases1.device)
        self.if_update_bases =True
        self.construct_H()
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
        if self.print_time: 
            tracker = TimeTracker()
            tracker.record("Start of forward pass")
        if self.if_update_bases:
            self.update_bases(x)
        if self.print_time:
            tracker.record("update_bases")

        length = len(self.ws)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)
        x = self.ln0(x.transpose(1,2)).transpose(1,2)
        if self.print_time:
            tracker.record("fc0 and ln0")

        for i, (layer , w, dplayer,ln) in enumerate(zip(self.sp_layers, self.ws, self.dropout_layers,self.ln_layers)):
            x1 = layer(x)
            x2 = w(x)
            x = x1 + x2
            x = dplayer(x)
            x = ln(x.transpose(1,2)).transpose(1,2)
            if self.act is not None and i != length - 1:
                x = self.act(x)
            if self.print_time:
                tracker.record(f"layer{i}")
        x = x.permute(0, 2, 1)

        fc_dim = self.fc_dim 
        
        if fc_dim > 0:
            x = self.fc1(x)
            if self.act is not None:
                x = self.act(x)

        x = self.fc2(x)
        if self.print_time:
            tracker.record("end of forward pass")
        return x

    def _choose_layer(self, index, in_channels, out_channels, layer_type):
        if layer_type == "compositeHGalerkinConv":
            type1 = self.composite_type1
            type2 = self.composite_type2
            modes_in1 = self.GkNN_mode_in
            modes_out1 = self.GkNN_mode_out
            kernel_modes1 = self.kernel_mode
            modes_in2 = self.num_basepts
            if self.phybases_out =='pca':
                modes_out2 = self.GkNN_mode_out_phybases
            elif self.phybases_out =='phy':
                modes_out2 = self.num_basepts
            kernel_modes2 = self.kernel_mode
            bases1 = self.bases1
            wbases1 = self.wbases1
            bases2 = self.bases2
            wbases2 = self.wbases2
            H1 = self.H1
            H2 = self.H2
            return compositeHGalerkinConv(type1,type2,in_channels, out_channels, modes_in1, modes_out1,kernel_modes1,modes_in2, 
                                          modes_out2, kernel_modes2,  bases1, wbases1, H1,bases2, wbases2,H2)
        else:
            raise ValueError("Layer Type Undefined.")
    
    def construct_H(self):
        if "compositeHGalerkinConv" in self.layer_types:
            
            if self.composite_type1 =='H':
                self.scale1 = 1/(self.GkNN_mode_in*self.GkNN_mode_out)
                self.H1 = nn.Parameter(
                        self.scale1
                        * torch.rand(self.kernel_mode, self.GkNN_mode_out, self.GkNN_mode_in, dtype=torch.float)
                    )
            elif self.composite_type1 =='D':
                assert self.GkNN_mode_in == self.GkNN_mode_out
                self.scale1 = 1/self.GkNN_mode_in
                self.H1 = nn.Parameter(
                        self.scale1
                        * torch.rand(self.kernel_mode, self.GkNN_mode_out, dtype=torch.float)
                    )
            elif self.composite_type1 =='newD':
                self.scale1 = 1/self.GkNN_mode_out
                self.H1 = nn.Parameter(
                        self.scale1
                        * torch.rand(self.kernel_mode, self.GkNN_mode_out, dtype=torch.float)
                    )
            if self.phybases_out =='pca':
                assert self.composite_type2 == 'H'
                self.scale2 = 1/(self.num_basepts*self.GkNN_mode_out_phybases)
                self.H2 = nn.Parameter(
                        self.scale2
                        * torch.rand(self.kernel_mode, self.GkNN_mode_out_phybases, self.num_basepts, dtype=torch.float)
                    )

            elif self.phybases_out =='phy':
                if self.composite_type2 == 'H':
                    self.scale2 = 1/(self.num_basepts*self.num_basepts)
                    self.H2 = nn.Parameter(
                            self.scale2
                            * torch.rand(self.kernel_mode, self.num_basepts, self.num_basepts, dtype=torch.float)
                        )
                elif self.composite_type2 == 'D':
                    self.scale2 = 1/self.num_basepts
                    self.H2 = nn.Parameter(
                            self.scale2
                            * torch.rand(self.kernel_mode, self.num_basepts, dtype=torch.float)
                        )

        

    def update_bases(self,x):
        bases = self.compute_bases(x[:,:,self.in_dim-self.phy_dim:])
        bases = F.relu(bases-self.bases_threhold.unsqueeze(0).unsqueeze(0))
        # bases = bases-self.bases_threhold.unsqueeze(0).unsqueeze(0)
        n = bases.shape[1]
        
        # bases = bases - torch.sum(bases,dim=1).unsqueeze(1)/n
        bases = bases/torch.sqrt(torch.sum(bases**2,dim=1)).unsqueeze(1)
        for sp_layer in self.sp_layers:
            self.wbases2 = bases/(x.shape[1])
            sp_layer.layer_PhyHGkNN.wbases = self.wbases2
            if self.phybases_out =='phy':
                sp_layer.layer_PhyHGkNN.bases = bases


    def compute_bases(self,x_phy_in):
        #x_phy_in.shape:  bsz,n,phy_in_channel
        x_phy_in = x_phy_in.unsqueeze(2) #bsz,n,1,phy_in_channel

        num_pts = self.basepts.shape[0]
        dim = self.basepts.shape[1]
        range_pts = torch.tensor(self.range_pts).to(self.basepts.device)
        range_pts_min = range_pts[:, 0].unsqueeze(0).expand(num_pts, dim) 
        range_pts_max = range_pts[:, 1].unsqueeze(0).expand(num_pts, dim) 
        basepts = torch.clamp(self.basepts, min=range_pts_min, max=range_pts_max)
        basepts = basepts.unsqueeze(0).unsqueeze(0) # 1,1,phy_out_channel,phy_in_channel
        baseweight = self.baseweight.unsqueeze(0).unsqueeze(0)  #1,1,phy_out_channel,phy_in_channel
        baseweight = baseweight + self.minweight
        x_phy_out = torch.sqrt(torch.prod(baseweight, dim=3))*torch.exp(-1*torch.sum(baseweight*(x_phy_in-basepts)**2,dim=3))  #bsz,n,phy_out_channel,phy_in_channel-->bsz,n,phy_out_channel
        return x_phy_out

    def uniform_points(self,num_pts,dim):
        a = round(math.pow(num_pts, 1 / dim))
        index_tensors = []
        for k in range(dim):
            xmin,xmax = self.range_pts[k][0],self.range_pts[k][1]
            idx = xmin + (xmax-xmin)*torch.arange(a).float().add(0.5).div(a)
            idx = idx.view((1,) * k+ (-1,) + (1,) * (dim - k - 1))
            index_tensors.append(idx.expand(a, *([a] * (dim - 1))))
        num_pts1 = int(torch.pow(torch.tensor(a),dim))
        x = torch.stack(index_tensors, dim=dim).reshape(num_pts1,dim)
        return x