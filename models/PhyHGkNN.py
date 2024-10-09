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


class Phylift(nn.Module):
    def __init__(self,phy_in_channel,in_channel,out_channel,liftpts,liftweight,minweight):
        super(Phylift, self).__init__()
        self.phy_in_channel = phy_in_channel
        self.phy_out_channel = liftpts.shape[0]
        self.dim = liftpts.shape[1]
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.liftpts = liftpts  #shape: phy_out_channel, dim
        self.liftweight = liftweight  #shape: phy_out_channel, dim
        self.minweight = minweight
        if liftpts.shape[0]==0:
            self.fc = nn.Linear(in_channel, out_channel)
        else:
            self.fc = nn.Linear(self.phy_out_channel - phy_in_channel + in_channel, out_channel)
    
    def forward(self,x):
        if self.liftpts.shape[0]==0:
            x = self.fc(x)
        else:
            x_phy_in = x[:,:,self.in_channel-self.phy_in_channel:]
            x_phy_out = self.compute_bases(x_phy_in)
            x = torch.cat((x[:,:,:self.in_channel-self.phy_in_channel],x_phy_out),dim=2)
            x = self.fc(x) 
        return x
    
    def compute_bases(self,x_phy_in):
        #x_phy_in.shape:  bsz,n,phy_in_channel
        x_phy_in = x_phy_in.unsqueeze(2) #bsz,n,1,phy_in_channel
        liftpts = torch.clamp(self.liftpts, min=0, max=1)
        liftpts = liftpts.unsqueeze(0).unsqueeze(0) # 1,1,phy_out_channel,phy_in_channel
        liftweight = self.liftweight.unsqueeze(0).unsqueeze(0)  #1,1,phy_out_channel,phy_in_channel
        liftweight = liftweight + self.minweight
        # print(liftweight.shape,x_phy_in.shape,liftpts.shape)
        # print(y.shape)
        x_phy_out = torch.sqrt(torch.prod(liftweight, dim=3))*torch.exp(-1*torch.sum(liftweight*(x_phy_in-liftpts)**2,dim=3))  #bsz,n,phy_out_channel,phy_in_channel-->bsz,n,phy_out_channel
        return x_phy_out

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


def compute_H(D_out,D_in, A, B, product):
    J = A.shape[0]
    #product.shape: bsz,K,L
    K = product.shape[-2]
    L = product.shape[-1]
    bsz = product.shape[0]
    # print(f'bsz={bsz}')
    H = D_out.reshape(1,J,K,1)*D_in.reshape(1,J,1,L)*product.reshape(bsz,1,K,L)
    # H = D_out.reshape(J,K,1)*product.reshape(1,K,L)
    Q = torch.bmm(A.transpose(1,2), B)
    mask_mode1 = Q.shape[1]
    mask_mode2 = Q.shape[2]
    H[:,:,:mask_mode1,:mask_mode2] +=Q.repeat(bsz,1,1,1)
    #H.shape: J,K,L
    return H

def mycompl_mul1d_bsz(weights, H , x_hat):
    x_hat1 = torch.einsum('bjkl,bil -> bijk', H , x_hat)
    y = torch.einsum('ioj,bijk -> bok', weights , x_hat1)
    return y

def compute_H_model(model):
    return compute_H(model.D_out1,model.D_in1, model.A1, model.B1, model.product)

class lowrankPhyHGalerkinConv(nn.Module):
    def __init__(self, bases, wbases,product,D_out,D_in, A,B,config):
        super(lowrankPhyHGalerkinConv, self).__init__()

        all_attr = list(config.keys())
        for key in all_attr:
            setattr(self, key, config[key])

        self.bases = bases
        self.wbases = wbases
        self.dtype = bases.dtype

        self.scale = 1 / (self.GkNN_mode_in * self.GkNN_mode_out)


        self.weights = nn.Parameter(
            self.scale
            * torch.randn(self.dim, self.dim, self.kernel_mode, dtype=self.dtype)
        )
        self.D_out = D_out
        self.D_in = D_in
        self.A = A
        self.B = B
        self.product = product


        # init.xavier_uniform_(self.weights)
        

    def forward(self, x):
        #x.shape: bsz,channel,N
        bases, wbases = self.bases, self.wbases
        D_out,D_in,A,B,product = self.D_out ,self.D_in,self.A, self.B,self.product

        # Compute coeffcients

        x_hat = torch.einsum("bcx,bxk->bck", x, wbases[:,:,:self.GkNN_mode_in])

        # Multiply relevant Fourier modes
        H = compute_H(D_out,D_in,A, B, product)

        x_hat = mycompl_mul1d_bsz(self.weights, H, x_hat)

        # Return to physical space
        x = torch.einsum("bck,xk->bcx", x_hat, bases[:,:self.GkNN_mode_out])
        # x = torch.einsum("bck,bxk->bcx", x_hat, bases[:,:,:self.GkNN_mode_out])
        return x
    
class PhyHGkNN(nn.Module):
    def __init__(self,bases_list,**config):

        super(PhyHGkNN, self).__init__()
        self.bases1=bases_list[0]
        self.wbases1=bases_list[1]
        self.bases2=bases_list[2]
        self.wbases2=bases_list[3]

            
        self.config = defaultdict(lambda: None, **config)
        self.config = dict(self.config)
        all_attr = list(self.config.keys())
        for key in all_attr:
            setattr(self, key, self.config[key])

        all_attr_layer = list(self.layer.keys())
        for key in all_attr_layer:
            setattr(self, key, self.layer[key])
        
        

        # self.liftpts = nn.Parameter(torch.rand(self.num_liftpts, self.phy_dim, dtype = torch.float))
        # self.basepts = nn.Parameter(torch.rand(self.num_basepts, self.phy_dim, dtype = torch.float))        
        liftpts = self.uniform_points(self.num_liftpts, self.phy_dim)
        basepts = self.uniform_points(self.num_basepts, self.phy_dim)
        if self.pts_type == 'trained':
            self.liftpts = nn.Parameter(liftpts)
            self.basepts = nn.Parameter(basepts)
        elif self.pts_type == 'fixed':
            self.liftpts = liftpts.to('cuda')
            self.basepts = basepts.to('cuda')

        self.liftweight = nn.Parameter(self.liftradius*torch.rand(self.num_liftpts, self.phy_dim, dtype = torch.float))
        if self.baseweight_type =='parameter':
            self.baseweight = nn.Parameter(self.baseradius*torch.rand(self.num_basepts, self.phy_dim, dtype = torch.float))
            self.bases_threhold = nn.Parameter(1*torch.rand(self.num_basepts, dtype = torch.float))
        elif self.baseweight_type =='same':
            self.baseweight = torch.zeros(self.num_basepts, self.phy_dim, dtype = torch.float,  device = self.bases1.device)
            self.bases_threhold = torch.ones(self.num_basepts, dtype = torch.float,device = self.bases1.device)
        self.if_update_bases =True
        

        self.lift_layer = Phylift(self.phy_dim,self.in_dim,self.layers_dim[0],self.liftpts,self.liftweight,self.minweight)

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
        if self.if_update_bases:
            self.update_bases(x)

        length = len(self.ws)
        x = self.lift_layer(x)
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

    def _choose_layer(self, index, in_channels, out_channels, layer_type):
        if layer_type == "HGalerkinConv":
            num_modes_in = self.GkNN_mode_in
            if self.phybases_out =='pca':
                num_modes_out = self.GkNN_mode_out_phybases
            elif self.phybases_out =='phy':
                num_modes_out = self.num_basepts
            kernel_modes = self.kernel_mode
            bases = self.bases1
            wbases = self.wbases1
            
            return PhyHGalerkinConv(in_channels, out_channels, num_modes_in, num_modes_out, kernel_modes, bases, wbases, self.H1)
        elif layer_type == "lowrankHGalerkinConv":
            bases = self.bases1
            wbases = self.wbases1
            D_out = self.D_out1
            D_in = self.D_in1
            A = self.A1
            B = self.B1
            product = 0
            return lowrankPhyHGalerkinConv(bases, wbases,product,D_out,D_in, A,B,self.layer)
        else:
            raise ValueError("Layer Type Undefined.")
    
    def construct_H(self):
        if "HGalerkinConv" in self.layer_types:
            if self.phybases_out =='pca':
                self.scale = 1/(self.GkNN_mode_in*self.GkNN_mode_out_phybases)
                self.H1 = nn.Parameter(
                        self.scale
                        * torch.rand(self.kernel_mode, self.GkNN_mode_out_phybases, self.num_basepts, dtype=torch.float)
                    )
            elif self.phybases_out =='phy':
                self.scale = 1/(self.num_basepts*self.num_basepts)
                self.H1 = nn.Parameter(
                        self.scale
                        * torch.rand(self.kernel_mode, self.num_basepts, self.num_basepts, dtype=torch.float)
                    )
        elif "lowrankHGalerkinConv" in self.layer_types:
            self.scale_in = 1/self.GkNN_mode_in
            self.scale_out = 1/self.GkNN_mode_out_phybases
            self.D_out1 = nn.Parameter(
                math.sqrt(self.scale_out)
                * torch.rand(self.kernel_mode,self.GkNN_mode_out_phybases, dtype=torch.float)
            )
            self.D_in1 = nn.Parameter(
                math.sqrt(self.scale_out)
                * torch.rand(self.kernel_mode,self.GkNN_mode_out_phybases, dtype=torch.float)
            )
            # self.D_in1 = 0
            if self.mask_mode:
                self.A1 = nn.Parameter(
                    self.scale_out
                    * torch.rand(self.kernel_mode, self.rank, self.mask_mode, dtype=torch.float)
                )
                self.B1 = nn.Parameter(
                    self.scale_in
                    * torch.rand(self.kernel_mode, self.rank, self.mask_mode, dtype=torch.float)
                )
            else:
                self.A1 = nn.Parameter(
                    self.scale_out
                    * torch.rand(self.kernel_mode, self.rank, self.GkNN_mode_out, dtype=torch.float)
                )
                self.B1 = nn.Parameter(
                    self.scale_in
                    * torch.rand(self.kernel_mode, self.rank, self.GkNN_mode_in, dtype=torch.float)
                )           

    def update_bases(self,x):
        bases = self.compute_bases(x[:,:,self.in_dim-self.phy_dim:])
        bases = F.relu(bases-self.bases_threhold.unsqueeze(0).unsqueeze(0))
        # bases = bases-self.bases_threhold.unsqueeze(0).unsqueeze(0)
        n = bases.shape[1]
        
        # bases = bases - torch.sum(bases,dim=1).unsqueeze(1)/n
        bases = bases/torch.sqrt(torch.sum(bases**2,dim=1)).unsqueeze(1)
        for sp_layer in self.sp_layers:
            self.wbases = bases/(x.shape[1])
            sp_layer.wbases = self.wbases
            if self.phybases_out =='phy':
                sp_layer.bases = bases
            # sp_layer.bases = bases
        if "lowrankHGalerkinConv" in self.layer_types:
            product = torch.bmm(self.bases1.transpose(0,1).unsqueeze(0).repeat(x.shape[0],1,1),self.wbases)
            self.product = product
            for sp_layer in self.sp_layers:
                sp_layer.product = product

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
        a = int(torch.pow(torch.tensor(num_pts),1/dim))
        index_tensors = []
        for k in range(dim):
            xmin,xmax = self.range_pts[k][0],self.range_pts[k][1]
            idx = xmin + (xmax-xmin)*torch.arange(a).float().add(0.5).div(a)
            idx = idx.view((1,) * k+ (-1,) + (1,) * (dim - k - 1))
            index_tensors.append(idx.expand(a, *([a] * (dim - 1))))
        num_pts1 = int(torch.pow(torch.tensor(a),dim))
        x = torch.stack(index_tensors, dim=dim).reshape(num_pts1,dim)
        return x
        
