import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .basics import SpectralConv4d
from .utils import add_padding, remove_padding, _get_act


class FNN4d(nn.Module):
    def __init__(self, 
                 modes1, modes2, modes3, modes4, width=16, 
                 layers=None, fc_dim=128,
                 in_dim=4, out_dim=1,
                 act='gelu', 
                 pad_ratio=0):
        '''
        Args:
            modes1: list of int, first dimension maximal modes for each layer
            modes2: list of int, second dimension maximal modes for each layer
            modes3: list of int, third dimension maximal modes for each layer
            layers: list of int, channels for each layer
            fc_dim: dimension of fully connected layers
            in_dim: int, input dimension
            out_dim: int, output dimension
            act: {tanh, gelu, relu, leaky_relu}, activation function
            pad_ratio: the ratio of the extended domain
        '''
        super(FNN4d, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.modes4 = modes4
        self.width = width
        if layers is None:
            self.layers = [width] * 4
        else:
            self.layers = layers
        self.pad_ratio = pad_ratio
        self.fc_dim = fc_dim
        
        self.fc0 = nn.Linear(in_dim, layers[0])

        self.sp_convs = nn.ModuleList([SpectralConv4d(
            in_size, out_size, mode1_num, mode2_num, mode3_num, mode4_num)
            for in_size, out_size, mode1_num, mode2_num, mode3_num, mode4_num
            in zip(self.layers, self.layers[1:], self.modes1, self.modes2, self.modes3, self.modes4)])

        self.ws = nn.ModuleList([nn.Conv1d(in_size, out_size, 1)
                                 for in_size, out_size in zip(self.layers, self.layers[1:])])
        if fc_dim > 0:
            self.fc1 = nn.Linear(layers[-1], fc_dim)
            self.fc2 = nn.Linear(fc_dim, out_dim)
        else:
            self.fc2 = nn.Linear(layers[-1], out_dim)
        
        self.act = _get_act(act)

    def forward(self, x):
        '''
        Args:
            x: (batchsize, x_grid, y_grid, t_grid, 3)

        Returns:
            u: (batchsize, x_grid, y_grid, t_grid, 1)

        '''
        length = len(self.ws)
        batchsize = x.shape[0]
        
        x = self.fc0(x)
        x = x.permute(0, 5, 1, 2, 3, 4)
        pad_nums = [math.floor(self.pad_ratio * x.shape[-4]), math.floor(self.pad_ratio * x.shape[-3]), math.floor(self.pad_ratio * x.shape[-2]), math.floor(self.pad_ratio * x.shape[-1])]
        x = add_padding(x, pad_nums=pad_nums)
        
        size_x, size_y, size_z, size_t = x.shape[-4], x.shape[-3], x.shape[-2], x.shape[-1]

        for i, (speconv, w) in enumerate(zip(self.sp_convs, self.ws)):
            x1 = speconv(x)
            x2 = w(x.view(batchsize, self.layers[i], -1)).view(batchsize, self.layers[i+1], size_x, size_y, size_z, size_t)
            x = x1 + x2
            if self.act is not None and i != length - 1:
                x = self.act(x)
                
        x = remove_padding(x, pad_nums=pad_nums)

        x = x.permute(0, 2, 3, 4, 5, 1)
        
        if self.fc_dim > 0:
            x = self.fc1(x)
            if self.act is not None:
                x = self.act(x)
                
        x = self.fc2(x)
        return x