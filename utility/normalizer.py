import numpy as np
import torch
import torch.nn.functional as F

class UnitGaussianNormalizer(object):
    def __init__(self, x, normalization_dim = [], non_normalized_dim = 0, eps=1.0e-5):
        super(UnitGaussianNormalizer, self).__init__()
        '''
        Normalize the input

            Parameters:  
                x : float[..., nchannels]
                normalization_dim  : list, which dimension to normalize
                                  when normalization_dim = [], global normalization 
                                  when normalization_dim = [0,1,...,len(x.shape)-2], and channel-by-channel normalization 
                non_normalized_dim  : last non_normalized_dim channels are not normalized

            Return :
                UnitGaussianNormalizer : class 
        '''
        self.non_normalized_dim = non_normalized_dim
        self.mean = torch.mean(x[...,0:x.shape[-1] - non_normalized_dim],  dim=normalization_dim)
        self.std  = torch.std(x[...,0:x.shape[-1]  - non_normalized_dim],  dim=normalization_dim)
        self.eps  = eps

    def encode(self, x, inplace: bool = False):
        std = self.std + self.eps 
        mean = self.mean
        y = x if inplace else x.clone()
        y[...,0:x.shape[-1] - self.non_normalized_dim] = (x[...,0:x.shape[-1]-self.non_normalized_dim] - mean) / std
        return y
    

    def decode(self, x, inplace: bool = False):
        std = self.std + self.eps 
        mean = self.mean
        y = x if inplace else x.clone()
        y[...,0:x.shape[-1] - self.non_normalized_dim] = (x[...,0:x.shape[-1]-self.non_normalized_dim] * std) + mean
        return y
    
    
    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        

