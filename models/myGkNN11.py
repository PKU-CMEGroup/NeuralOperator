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

def compute_H(D_out,D_in, A, B, product):
    J = A.shape[0]
    K = product.shape[0]
    L = product.shape[1]
    H = D_out.reshape(J,K,1)*D_in.reshape(J,1,L)*product.reshape(1,K,L)
    # H = D_out.reshape(J,K,1)*product.reshape(1,K,L)
    Q = torch.bmm(A.transpose(1,2), B)
    mask_mode1 = Q.shape[1]
    mask_mode2 = Q.shape[2]
    H[:,:mask_mode1,:mask_mode2] +=Q
    #H.shape: J,K,L
    return H

def compute_H_model(model):
    return compute_H(model.D_out1,model.D_in1, model.A1, model.B1, model.product1)

class newHGalerkinConv(nn.Module):
    def __init__(self, bases, wbases,product,D_out,D_in, A,B,config):
        super(newHGalerkinConv, self).__init__()

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

        x_hat = torch.einsum("bcx,xk->bck", x, wbases[:,:self.GkNN_mode_in])

        # Multiply relevant Fourier modes
        H = compute_H(D_out,D_in,A, B, product)

        x_hat = mycompl_mul1d(self.weights, H, x_hat)

        # Return to physical space
        x = torch.einsum("bck,xk->bcx", x_hat, bases[:,:self.GkNN_mode_out])

        return x
 
class newHGalerkinConv_double(nn.Module):
    def __init__(self, in_channels, out_channels, modes_in, modes_out, kernel_modes,rank, bases1, wbases1,product1, bases2, wbases2,product2,D1, A1,B1,D2,A2,B2):
        super(newHGalerkinConv_double, self).__init__()

        self.layer1 = newHGalerkinConv(in_channels, out_channels, modes_in, modes_out, kernel_modes,rank,  bases1, wbases1,product1, D1,A1,B1)
        self.layer2 = newHGalerkinConv(in_channels, out_channels, modes_in, modes_out, kernel_modes, rank, bases2, wbases2,product2, D2,A2,B2)
        
        
    def forward(self, x):

        x1 = self.layer1(x)
        x2 = self.layer2(x)
        x = x1 + x2

        return x
class newHGalerkinConv_symdouble(nn.Module):
    def __init__(self, bases1, wbases1,bases2, wbases2,product,D_out,D_in, A,B,config):
        super(newHGalerkinConv_symdouble, self).__init__()

        self.layer1 = newHGalerkinConv(bases1,wbases1,product,D_out,D_in, A,B,config)
        config2 = config.copy()
        config2['GkNN_mode_in'],config2['GkNN_mode_out'] = config['GkNN_mode_out'],config['GkNN_mode_in']
        self.layer2 = newHGalerkinConv(bases2,wbases2,product.transpose(0,1),D_in,D_out, B,A,config2)
        
        
    def forward(self, x):

        x1 = self.layer1(x)
        x2 = self.layer2(x)
        x = x1 + x2

        return x

class newHGkNN(nn.Module):
    def __init__(self, bases_list ,**config):

        super(newHGkNN, self).__init__()


            
        self.config = defaultdict(lambda: None, **config)
        self.config = dict(self.config)
        all_attr = list(self.config.keys())
        for key in all_attr:
            setattr(self, key, self.config[key])

        all_attr_layer = list(self.layer.keys())
        for key in all_attr_layer:
            setattr(self, key, self.layer[key])

        self.bases1=bases_list[0][:,:self.GkNN_mode_out]
        self.wbases1=bases_list[1][:,:self.GkNN_mode_in]
        

        self.product1 = torch.mm(self.bases1.transpose(0,1),self.wbases1)
        if self.double_bases == 'sym':
            self.bases2=bases_list[2][:,:self.GkNN_mode_in]
            self.wbases2=bases_list[3][:,:self.GkNN_mode_out]
            self.product2 = torch.mm(self.bases2.transpose(0,1),self.wbases2)

        
        self.scale_in = 1/self.GkNN_mode_in
        self.scale_out = 1/self.GkNN_mode_out
        # indices = torch.arange(1, self.kernel_mode+ 1)  
        # self.scale = (1/self.GkNN_mode) / (indices ** 2).reshape(self.kernel_mode,1,1)
    


        self.D_out1 = nn.Parameter(
            math.sqrt(self.scale_out)
            * torch.rand(self.kernel_mode,self.GkNN_mode_out, dtype=torch.float)
        )
        self.D_in1 = nn.Parameter(
            math.sqrt(self.scale_out)
            * torch.rand(self.kernel_mode,self.GkNN_mode_in, dtype=torch.float)
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
        if layer_type == "newHGalerkinConv":
            if not self.double_bases:
                bases = self.bases1
                wbases = self.wbases1
                D_out = self.D_out1
                D_in = self.D_in1
                A = self.A1
                B = self.B1
                product = self.product1
                return newHGalerkinConv(bases, wbases,product,D_out,D_in, A,B,self.layer)
            elif self.double_bases == 'sym':
                bases1 = self.bases1
                wbases1 = self.wbases1
                bases2 = self.bases2
                wbases2 = self.wbases2
                D_out = self.D_out1
                D_in = self.D_in1
                A = self.A1
                B = self.B1
                product = self.product1
                return newHGalerkinConv_symdouble(bases1, wbases1,bases2, wbases2,product,D_out,D_in, A,B,self.layer)                
            else:
                raise ValueError("Not yet.")
                # bases1 = self.bases1
                # wbases1 = self.wbases1
                # bases2 = self.bases2
                # wbases2 = self.wbases2
                # D_out1 = self.D_out1
                # D_in1 = self.D_in1
                # A1 = self.A1
                # B1 = self.B1
                # product1 = self.product1
                # D_out2 = self.D_in1
                # D_in2 = self.D_out1
                # A2 = self.B1
                # B2 = self.A1
                # product2 = self.product1.transpose(1,2)
                # return newHGalerkinConv_double(bases1, wbases1, bases2, wbases2,product1,product2,D_out1,D_in1, A1,B1,D_out2,D_in2,A2,B2,self.layer)
        else:
            raise ValueError("Layer Type Undefined.")
        
