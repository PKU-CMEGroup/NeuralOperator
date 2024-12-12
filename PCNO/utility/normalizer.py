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
                non_normalized_dim  : last non_normalized_dim channels are note normalized

            Return :
                UnitGaussianNormalizer : class 
        '''
        self.non_normalized_dim = non_normalized_dim
        self.mean = torch.mean(x[...,0:x.shape[-1] - non_normalized_dim],  dim=normalization_dim)
        self.std  = torch.std(x[...,0:x.shape[-1]  - non_normalized_dim],  dim=normalization_dim)
        self.eps  = eps

    def encode(self, x):
        x[...,0:x.shape[-1] - self.non_normalized_dim] = (x[...,0:x.shape[-1]-self.non_normalized_dim] - self.mean) / (self.std + self.eps)
        return x
    

    def decode(self, x):
        std = self.std + self.eps 
        mean = self.mean
        x[...,0:x.shape[-1] - self.non_normalized_dim] = (x[...,0:x.shape[-1]-self.non_normalized_dim] * std) + mean
        return x
    
    
    def to(self, device):
        if device == torch.device('cuda:0'):
            self.mean = self.mean.cuda()
            self.std = self.std.cuda()
        else:
            self.mean = self.mean.cpu()
            self.std = self.std.cpu()



