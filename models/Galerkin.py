import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import sys

sys.path.append('../')
from models.basics import compl_mul1d
from models.utils import _get_act




class GalerkinConv(nn.Module):
    def __init__(self, in_channels, out_channels, modes, bases, wbases):
        super(GalerkinConv, self).__init__()

        """
        1D Spectral layer. It avoids FFT, but utilizes low rank approximation. 
        
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes = modes
        self.bases = bases
        self.wbases = wbases

        self.scale = (1 / (in_channels*out_channels))
        self.weights = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes, dtype=torch.float))


    def forward(self, x):
        bases, wbases = self.bases, self.wbases
        # Compute coeffcients


        x_hat = torch.einsum('bcx,xk->bck', x, wbases)


        # Multiply relevant Fourier modes
        x_hat = compl_mul1d(x_hat, self.weights)

        # Return to physical space
        x = torch.real(torch.einsum('bck,xk->bcx', x_hat, bases))
        
        return x
    
class GkNN(nn.Module):
    def __init__(self,
                 bases,
                 wbases,
                 **config
                ):
        super(GkNN, self).__init__()

        """
        The overall network. It contains several layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.config = defaultdict(lambda: None, **config)
        # print(kwargs)
        self.config = dict(self.config)
        # print(self.config)
        all_attr = list(self.config.keys())
        for key in all_attr:
            setattr(self, key, self.config[key])
            
        self.modes = self.GkNN_modes

        if len(bases) == 1:
            bases = [bases[0][:,0:i] for i in self.modes]
        if len(wbases) == 1:

            wbases = [wbases[0][:,0:i] for i in self.modes]


        self.bases = bases
        self.wbases = wbases
        

        
        self.fc0 = nn.Linear(self.in_dim, self.layers[0])  # input channel is 2: (a(x), x)

        self.sp_convs = nn.ModuleList([GalerkinConv(
            in_size, out_size, num_modes, bases, wbases) for in_size, out_size,
              num_modes, bases, wbases in zip(self.layers, self.layers[1:], self.modes, self.bases, self.wbases)])

        self.ws = nn.ModuleList([nn.Conv1d(in_size, out_size, 1)
                                 for in_size, out_size in zip(self.layers, self.layers[1:])])
        
        # if fc_dim = 0, we do not have nonlinear layer
        if self.fc_dim > 0:
            self.fc1 = nn.Linear(self.layers[-1],self.fc_dim)
            self.fc2 = nn.Linear(self.fc_dim, self.out_dim) 
        else:
            self.fc2 = nn.Linear(self.layers[-1], self.out_dim)
            
        self.act = _get_act(self.act)

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

        for i, (speconv, w) in enumerate(zip(self.sp_convs, self.ws)):
            x1 = speconv(x)
            x2 = w(x)
            x = x1 + x2
            if self.act is not None and i != length - 1:
                x = self.act(x)
                
      
        x = x.permute(0, 2, 1)
        
        # if fc_dim = 0, we do not have nonlinear layer
        fc_dim = self.fc_dim if hasattr(self, 'fc_dim') else 1
        
        if fc_dim > 0:
            x = self.fc1(x)
            if self.act is not None:
                x = self.act(x)
            
        x = self.fc2(x)
        
        
        return x









  