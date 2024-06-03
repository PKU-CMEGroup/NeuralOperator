import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import operator
from functools import reduce
import numpy as np



def add_padding(x, pad_nums):

    if x.ndim == 3: #fourier1d
        res = F.pad(x, [0, pad_nums[0]], 'constant', 0)
    elif x.ndim == 4: #fourier2d
        res = F.pad(x, [0, pad_nums[1], 0, pad_nums[0]], 'constant', 0)
    elif x.ndim == 5: #fourier3d
        res = F.pad(x, [0, pad_nums[2], 0, pad_nums[1], 0, pad_nums[0]], 'constant', 0)
    elif x.ndim == 6: #fourier4d
        res = F.pad(x, [0, pad_nums[3], 0, pad_nums[2], 0, pad_nums[1], 0, pad_nums[0]], 'constant', 0)
    else:
        print("error : x.ndim = ", x.ndim)
            
    return res


def remove_padding(x, pad_nums):
    
    if x.ndim == 3: #fourier1d
        res = x[..., :(None if pad_nums[0] == 0 else -pad_nums[0])]
        
    elif x.ndim == 4: #fourier2d
        res = x[..., :(None if pad_nums[0] == 0 else -pad_nums[0]), :(None if pad_nums[1] == 0 else -pad_nums[1])]

    elif x.ndim == 5: #fourier3d
        res = x[..., :(None if pad_nums[0] == 0 else -pad_nums[0]), :(None if pad_nums[1] == 0 else -pad_nums[1]), :(None if pad_nums[2] == 0 else -pad_nums[2])]
    
    elif x.ndim == 6: #fourier4d
        res = x[..., :(None if pad_nums[0] == 0 else -pad_nums[0]), :(None if pad_nums[1] == 0 else -pad_nums[1]), :(None if pad_nums[2] == 0 else -pad_nums[2]), :(None if pad_nums[3] == 0 else -pad_nums[3])]
        
    else:
        print("error : x.ndim = ", x.ndim)
    
    return res


def _get_act(act):
    if act == 'tanh':
        func = F.tanh
    elif act == 'gelu':
        func = F.gelu
    elif act == 'relu':
        func = F.relu_
    elif act == 'elu':
        func = F.elu_
    elif act == 'leaky_relu':
        func = F.leaky_relu_
    elif act == 'none':
        func = None
    else:
        raise ValueError(f'{act} is not supported')
    return func

def count_params(model):
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul, 
                    list(p.size()+(2,) if p.is_complex() else p.size()))
    return c




def compute_1dFourier_bases(nx, k, Lx):
    grid = np.linspace(0, Lx, nx+1)[:-1]
    bases = np.zeros((nx, k))
    bases[:, 0] = 1/np.sqrt(Lx)
    weights = np.ones(nx)*Lx/nx
    for i in range(k//2):
        bases[:, 2*i+1] = np.sqrt(2/Lx)*np.cos(2*np.pi*(i+1)*grid/Lx)
        if 2*i + 2 <= k-1:
            bases[:, 2*i+2] = np.sqrt(2/Lx)*np.sin(2*np.pi*(i+1)*grid/Lx)
    return grid, bases, weights
