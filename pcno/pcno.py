import math
import numpy as np
import torch
import sys
import torch.nn as nn
import torch.nn.functional as F
from timeit import default_timer
from utility.adam import Adam
from utility.losses import LpLoss
from utility.normalizer import UnitGaussianNormalizer

    

def _get_act(act):
    if act == "tanh":
        func = F.tanh
    elif act == "gelu":
        func = F.gelu
    elif act == "relu":
        func = F.relu_
    elif act == "elu":
        func = F.elu_
    elif act == "leaky_relu":
        func = F.leaky_relu_
    elif act == "none":
        func = None
    else:
        raise ValueError(f"{act} is not supported")
    return func


def scaled_sigmoid(x: torch.Tensor, min_val: float, max_val: float) -> torch.Tensor:
    """
    Applies a sigmoid function scaled to output values in the range [min_val, max_val].
    This transformation maps any real-valued input to a specified bounded interval,
    maintaining gradient flow for backpropagation. Useful for constraining network outputs.
    
    Math:
        output = min_val + (max_val - min_val) * σ(x)
        where σ(x) = 1/(1 + exp(-x)) is the standard sigmoid function
    
    Require:
        max_val >= min_val
    """
    return min_val + (max_val - min_val) * torch.sigmoid(x)


def scaled_logit(y: torch.Tensor, min_val: float, max_val: float) -> torch.Tensor:
    """
    Inverse of scaled_sigmoid - maps values from [min_val, max_val] back to unbounded space.
    
    Also known as the generalized logit transform. Handles numerical stability at boundaries.
    
    Args:
        y: Input tensor (values must be in (min_val, max_val) range)
        min_val: Lower bound of input range (exclusive)
        max_val: Upper bound of input range (exclusive)
        
    Returns:
        Tensor of same shape as input with unbounded real values
    
    Math:
        output = log( (y - min_val) / (max_val - y) )
        This is the inverse operation of scaled_sigmoid()
  
    Require:
        min_val < y <  max_val
    """
    return torch.log((y - min_val)/(max_val - y))


def compute_Fourier_modes_helper(ndims, nks, Ls):
    '''
    Compute Fourier modes number k
    Fourier bases are cos(kx), sin(kx), 1
    * We cannot have both k and -k, cannot have 0

        Parameters:  
            ndims : int
            nks   : int[ndims]
            Ls    : float[ndims]

        Return :
            k_pairs : float[nmodes, ndims]
    '''
    assert(len(nks) == len(Ls) == ndims)    
    if ndims == 1:
        nk, Lx = nks[0], Ls[0]
        k_pairs    = np.zeros((nk, ndims))
        k_pair_mag = np.zeros(nk)
        i = 0
        for kx in range(1, nk + 1):
            k_pairs[i, :] = 2*np.pi/Lx*kx
            k_pair_mag[i] = np.linalg.norm(k_pairs[i, :])
            i += 1

    elif ndims == 2:
        nx, ny = nks
        Lx, Ly = Ls
        nk = 2*nx*ny + nx + ny
        k_pairs    = np.zeros((nk, ndims))
        k_pair_mag = np.zeros(nk)
        i = 0
        for kx in range(-nx, nx + 1):
            for ky in range(0, ny + 1):
                if (ky==0 and kx<=0): 
                    continue

                k_pairs[i, :] = 2*np.pi/Lx*kx, 2*np.pi/Ly*ky
                k_pair_mag[i] = np.linalg.norm(k_pairs[i, :])
                i += 1

    elif ndims == 3:
        nx, ny, nz = nks
        Lx, Ly, Lz = Ls
        nk = 4*nx*ny*nz + 2*(nx*ny + nx*nz + ny*nz) + nx + ny + nz
        k_pairs    = np.zeros((nk, ndims))
        k_pair_mag = np.zeros(nk)
        i = 0
        for kx in range(-nx, nx + 1):
            for ky in range(-ny, ny + 1):
                for kz in range(0, nz + 1):
                    if (kz==0 and (ky<0  or (ky==0 and kx<=0))): 
                        continue

                    k_pairs[i, :] = 2*np.pi/Lx*kx, 2*np.pi/Ly*ky, 2*np.pi/Lz*kz
                    k_pair_mag[i] = np.linalg.norm(k_pairs[i, :])
                    i += 1
    else:
        raise ValueError(f"{ndims} in compute_Fourier_modes is not supported")
    
    k_pairs = k_pairs[np.argsort(k_pair_mag, kind='stable'), :]
    return k_pairs


def compute_Fourier_modes(ndims, nks, Ls):
    '''
    Compute `nmeasures` sets of Fourier modes number k
    Fourier bases are cos(kx), sin(kx), 1
    * We cannot have both k and -k

        Parameters:  
            ndims : int
            nks   : int[ndims * nmeasures]
            Ls    : float[ndims * nmeasures]

        Return :
            k_pairs : float[nmodes, ndims, nmeasures]
    '''
    assert(len(nks) == len(Ls))
    nmeasures = len(nks) // ndims
    k_pairs = np.stack([compute_Fourier_modes_helper(ndims, nks[i*ndims:(i+1)*ndims], Ls[i*ndims:(i+1)*ndims]) for i in range(nmeasures)], axis=-1)
    
    return k_pairs


def compute_Fourier_bases(nodes, modes):
    '''
    Compute Fourier bases for the whole space
    Fourier bases are cos(kx), sin(kx), 1

        Parameters:  
            nodes        : float[batch_size, nnodes, ndims]
            modes        : float[nmodes, ndims, nmeasures]
            
        Return :
            bases_c, bases_s : float[batch_size, nnodes, nmodes, nmeasures]
            bases_0 : float[batch_size, nnodes, 1, nmeasures]
    '''
    # temp : float[batch_size, nnodes, nmodes, nmeasures]
    temp  = torch.einsum("bxd,kdw->bxkw", nodes, modes) 
    
    bases_c = torch.cos(temp) 
    bases_s = torch.sin(temp) 
    batch_size, nnodes, _, nmeasures = temp.shape
    bases_0 = torch.ones(batch_size, nnodes, 1, nmeasures, dtype=temp.dtype, device=temp.device)
    return bases_c, bases_s, bases_0

################################################################
# Fourier layer
################################################################
class SpectralConv(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super(SpectralConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        nmodes, ndims, nmeasures = modes.shape
        self.modes = modes
        self.nmeasures = nmeasures
        self.scale = 1 / (in_channels * out_channels)

        self.weights_c = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels, out_channels, nmodes, nmeasures, dtype=torch.float
            )
        )
        self.weights_s = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels, out_channels, nmodes, nmeasures, dtype=torch.float
            )
        )
        self.weights_0 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels, out_channels, 1, nmeasures, dtype=torch.float
            )
        )


    def forward(self, x, bases_c, bases_s, bases_0, wbases_c, wbases_s, wbases_0):
        '''
        Compute Fourier neural layer
            Parameters:  
                x                   : float[batch_size, in_channels, nnodes]
                bases_c, bases_s    : float[batch_size, nnodes, nmodes, nmeasures]
                bases_0             : float[batch_size, nnodes, 1, nmeasures]
                wbases_c, wbases_s  : float[batch_size, nnodes, nmodes, nmeasures]
                wbases_0            : float[batch_size, nnodes, 1, nmeasures]

            Return :
                x                   : float[batch_size, out_channels, nnodes]
        '''    
        x_c_hat =  torch.einsum("bix,bxkw->bikw", x, wbases_c)
        x_s_hat = -torch.einsum("bix,bxkw->bikw", x, wbases_s)
        x_0_hat =  torch.einsum("bix,bxkw->bikw", x, wbases_0)

        weights_c, weights_s, weights_0 = self.weights_c, self.weights_s, self.weights_0
        
        f_c_hat = torch.einsum("bikw,iokw->bokw", x_c_hat, weights_c) - torch.einsum("bikw,iokw->bokw", x_s_hat, weights_s)
        f_s_hat = torch.einsum("bikw,iokw->bokw", x_s_hat, weights_c) + torch.einsum("bikw,iokw->bokw", x_c_hat, weights_s)
        f_0_hat = torch.einsum("bikw,iokw->bokw", x_0_hat, weights_0) 

        x = torch.einsum("bokw,bxkw->box", f_0_hat, bases_0)  + 2*torch.einsum("bokw,bxkw->box", f_c_hat, bases_c) -  2*torch.einsum("bokw,bxkw->box", f_s_hat, bases_s) 
        
        return x
    

def compute_gradient(f, directed_edges, edge_gradient_weights):
    '''
    Compute gradient of field f at each node
    The gradient is computed by least square.
    Node x has neighbors x1, x2, ..., xj

    x1 - x                        f(x1) - f(x)
    x2 - x                        f(x2) - f(x)
       :      gradient f(x)   =         :
       :                                :
    xj - x                        f(xj) - f(x)
    
    in matrix form   dx  nable f(x)   = df.
    
    The pseudo-inverse of dx is pinvdx.
    Then gradient f(x) for any function f, is pinvdx * df
    directed_edges stores directed edges (x, x1), (x, x2), ..., (x, xj)
    edge_gradient_weights stores its associated weight pinvdx[:,1], pinvdx[:,2], ..., pinvdx[:,j]

    Then the gradient can be computed 
    gradient f(x) = sum_i pinvdx[:,i] * [f(xi) - f(x)] 
    with scatter_add for each edge
    
    
        Parameters: 
            f : float[batch_size, in_channels, nnodes]
            directed_edges : int[batch_size, max_nedges, 2] 
            edge_gradient_weights : float[batch_size, max_nedges, ndims]
            
        Returns:
            x_gradients : float Tensor[batch_size, in_channels*ndims, max_nnodes]
            * in_channels*ndims dimension is gradient[x_1], gradient[x_2], gradient[x_3]......
    '''

    f = f.permute(0,2,1)
    batch_size, max_nnodes, in_channels = f.shape
    _, max_nedges, ndims = edge_gradient_weights.shape
    # Message passing : compute message = edge_gradient_weights * (f_source - f_target) for each edge
    # target\source : int Tensor[batch_size, max_nedges]
    # message : float Tensor[batch_size, max_nedges, in_channels*ndims]

    target, source = directed_edges[...,0], directed_edges[...,1]  # source and target nodes of edges
    message = torch.einsum('bed,bec->becd', edge_gradient_weights, f[torch.arange(batch_size).unsqueeze(1), source] - f[torch.arange(batch_size).unsqueeze(1), target]).reshape(batch_size, max_nedges, in_channels*ndims)
    
    # f_gradients : float Tensor[batch_size, max_nnodes, in_channels*ndims]
    f_gradients = torch.zeros(batch_size, max_nnodes, in_channels*ndims, dtype=message.dtype, device=message.device)
    f_gradients.scatter_add_(dim=1, src=message, index=target.unsqueeze(2).repeat(1,1,in_channels*ndims))
    
    return f_gradients.permute(0,2,1)
    

class SpectralConvLocal(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super(SpectralConvLocal, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        nmodes, ndims, nmeasures = modes.shape
        self.modes = modes
        self.nmeasures = nmeasures
        self.scale = 1 / (in_channels * out_channels)

        self.weights_c = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels, out_channels, nmodes, nmeasures, dtype=torch.float
            )
        )
        self.weights_s = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels, out_channels, nmodes, nmeasures, dtype=torch.float
            )
        )
        self.weights_0 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels, out_channels, 1, nmeasures, dtype=torch.float
            )
        )
        
    
    def forward(self, f, bases_c, bases_s, bases_0, directed_edges, node_weights):
        '''
        Assemble local force f at each node
            u(x)  =  int K(x,y) f(y) dS(y)
        here K(x,y) = sum_k (wc_k + ws_k i) e^{2 k pi (x - y)/L}. 
        To ensure the result is real, we have wc_k = wc_{-k}, ws_k = -ws_{-k}
        
        directed_edges stores directed edges (x, y1), (x, y2), ..., (x, yj)
        send from xj to x:
            u(x)  =  sum K(x,yj) f(yj) S(yj)
                  =  sum_j sum_k [wc_k + ws_k i] [cos(2pikx/L) + sin(2pikx/L)i] [cos(2pikyj/L) - sin(2pikyj/L)i] f(yj) S(yj)   
                  =  sum_j sum_k [wc_k + ws_k i] [cos(2pikx/L) + sin(2pikx/L)i] [cos(2pikyj/L) - sin(2pikyj/L)i] f(yj) S(yj)    
                  =  sum_j sum_k [wc_k cos(2pikx/L) - ws_k sin(2pikx/L) + ws_k cos(2pikx/L)i + wc_k sin(2pikx/L)i] [cos(2pikyj/L) - sin(2pikyj/L)i] f(yj) S(yj)    
                  =  sum_j { wc_0 + 2sum_k [wc_k cos(2pikx/L)cos(2pikyj/L) - ws_k sin(2pikx/L)cos(2pikyj/L) + ws_k cos(2pikx/L)sin(2pikyj/L) + wc_k sin(2pikx/L)sin(2pikyj/L)] } f(yj) S(yj)    
       
            Parameters: 
                f                   : float[batch_size, in_channels, nnodes]
                bases_c, bases_s    : float[batch_size, nnodes, nmodes, nmeasures]
                bases_0             : float[batch_size, nnodes, 1, nmeasures]
                directed_edges : int[batch_size, max_nedges, 2, nmeasures] 
                node_weights   : float[batch_size, max_nedges, nmeasures]
                
            Returns:
                f_out : float[batch_size, out_channels, max_nnodes]
        '''

        f = f.permute(0,2,1)
        batch_size, max_nnodes, in_channels = f.shape
        f_out = torch.zeros(batch_size, max_nnodes, in_channels, dtype=f.dtype, device=f.device)

        # Message passing : compute message = edge_gradient_weights * (f_source - f_target) for each edge
        # target\source : int Tensor[batch_size, max_nedges]
        # message : float Tensor[batch_size, max_nedges, in_channels*ndims]
        for m in range(self.nmeasures):
            target, source = directed_edges[...,0,m], directed_edges[...,1,m]  # target and source nodes of edges

            # diff_nodes = target - source
            # log_r = torch.log(torch.sum(diff_nodes**2, dim=-1) + 1e-10).unsqueeze(-1)  # (bsz, max_nedges,1)
            
            weights_c, weights_s, weights_0 = self.weights_c[...,m], self.weights_s[...,m], self.weights_0[...,0,m]
            
            # edge_local_weights = weights_0 + 2*torch.einsum('iok, bek->beio', weights_c, bases_c[...,m][torch.arange(batch_size).unsqueeze(1),target] * bases_c[...,m][torch.arange(batch_size).unsqueeze(1),source] + bases_s[...,m][torch.arange(batch_size).unsqueeze(1),target] * bases_s[...,m][torch.arange(batch_size).unsqueeze(1),source]) \
            #                                + 2*torch.einsum('iok, bek->beio', weights_s, bases_c[...,m][torch.arange(batch_size).unsqueeze(1),target] * bases_s[...,m][torch.arange(batch_size).unsqueeze(1),source] - bases_s[...,m][torch.arange(batch_size).unsqueeze(1),target] * bases_c[...,m][torch.arange(batch_size).unsqueeze(1),source])
            # message = torch.einsum('beio, bei, be->beo', edge_local_weights, f[torch.arange(batch_size).unsqueeze(1),source], node_weights[...,m])

            f_source = torch.einsum('bei, be->bei', f[torch.arange(batch_size).unsqueeze(1),source], node_weights[...,m])
        
            message  = torch.einsum('io, bei->beo', weights_0, f_source)
            
            edge_local_coeffs = torch.einsum('bek, bei->beki', bases_c[...,m][torch.arange(batch_size).unsqueeze(1),target] * bases_c[...,m][torch.arange(batch_size).unsqueeze(1),source] + bases_s[...,m][torch.arange(batch_size).unsqueeze(1),target] * bases_s[...,m][torch.arange(batch_size).unsqueeze(1),source], f_source)
            message += 2*torch.einsum('iok, beki->beo', weights_c, edge_local_coeffs)#*log_r
            
            edge_local_coeffs = torch.einsum('bek, bei->beki', bases_c[...,m][torch.arange(batch_size).unsqueeze(1),target] * bases_s[...,m][torch.arange(batch_size).unsqueeze(1),source] - bases_s[...,m][torch.arange(batch_size).unsqueeze(1),target] * bases_c[...,m][torch.arange(batch_size).unsqueeze(1),source], f_source)
            message += 2*torch.einsum('iok, beki->beo', weights_s, edge_local_coeffs)#*log_r
        
            
            f_out.scatter_add_(dim=1, src=message, index=target.unsqueeze(2).repeat(1,1,in_channels))
        
        return f_out.permute(0,2,1)


# class SpectralConvLocalSimp(nn.Module):
#     def __init__(self, in_channels, out_channels, modes):
#         super(SpectralConvLocalSimp, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         nmodes, ndims, nmeasures = modes.shape
#         self.modes = modes
#         self.nmeasures = nmeasures
#         self.scale = 1 / (in_channels * out_channels)

#         self.weights_c = nn.Parameter(
#             self.scale
#             * torch.rand(nmodes, nmeasures, dtype=torch.float
#             )
#         )
#         self.weights_s = nn.Parameter(
#             self.scale
#             * torch.rand(nmodes, nmeasures, dtype=torch.float
#             )
#         )
#         self.weights_0 = nn.Parameter(
#             self.scale
#             * torch.rand(1, nmeasures, dtype=torch.float
#             )
#         )
#         self.w = nn.Conv1d(in_channels, out_channels, 1)

        
    
#     def forward(self, f, bases_c, bases_s, bases_0, directed_edges, node_weights):
#         '''
#         Assemble local force f at each node
#             u(x)  =  int K(x,y) f(y) dS(y)
#         here K(x,y) = sum_k (wc_k + ws_k i) e^{2 k pi (x - y)/L}. 
#         To ensure the result is real, we have wc_k = wc_{-k}, ws_k = -ws_{-k}
        
#         directed_edges stores directed edges (x, y1), (x, y2), ..., (x, yj)
#         send from xj to x:
#             u(x)  =  sum K(x,yj) f(yj) S(yj)
#                   =  sum_j sum_k [wc_k + ws_k i] [cos(2pikx/L) + sin(2pikx/L)i] [cos(2pikyj/L) - sin(2pikyj/L)i] f(yj) S(yj)   
#                   =  sum_j sum_k [wc_k + ws_k i] [cos(2pikx/L) + sin(2pikx/L)i] [cos(2pikyj/L) - sin(2pikyj/L)i] f(yj) S(yj)    
#                   =  sum_j sum_k [wc_k cos(2pikx/L) - ws_k sin(2pikx/L) + ws_k cos(2pikx/L)i + wc_k sin(2pikx/L)i] [cos(2pikyj/L) - sin(2pikyj/L)i] f(yj) S(yj)    
#                   =  sum_j { wc_0 + 2sum_k [wc_k cos(2pikx/L)cos(2pikyj/L) - ws_k sin(2pikx/L)cos(2pikyj/L) + ws_k cos(2pikx/L)sin(2pikyj/L) + wc_k sin(2pikx/L)sin(2pikyj/L)] } f(yj) S(yj)    
       
#             Parameters: 
#                 f                   : float[batch_size, in_channels, nnodes]
#                 bases_c, bases_s    : float[batch_size, nnodes, nmodes, nmeasures]
#                 bases_0             : float[batch_size, nnodes, 1, nmeasures]
#                 directed_edges : int[batch_size, max_nedges, 2, nmeasures] 
#                 node_weights   : float[batch_size, max_nedges, nmeasures]
                
#             Returns:
#                 f_out : float[batch_size, out_channels, max_nnodes]
#         '''

#         f = f.permute(0,2,1)
#         batch_size, max_nnodes, in_channels = f.shape
#         f_out = torch.zeros(batch_size, max_nnodes, in_channels, dtype=f.dtype, device=f.device)

#         # Message passing : compute message = edge_gradient_weights * (f_source - f_target) for each edge
#         # target\source : int Tensor[batch_size, max_nedges]
#         # message : float Tensor[batch_size, max_nedges, in_channels*ndims]
#         for m in range(self.nmeasures):
#             target, source = directed_edges[...,0,m], directed_edges[...,1,m]  # target and source nodes of edges
            
#             weights_c, weights_s, weights_0 = self.weights_c[...,m], self.weights_s[...,m], self.weights_0[...,0,m]
            
#             edge_local_weights = weights_0 + 2*torch.einsum('k, bek->be', weights_c, bases_c[...,m][torch.arange(batch_size).unsqueeze(1),target] * bases_c[...,m][torch.arange(batch_size).unsqueeze(1),source] + bases_s[...,m][torch.arange(batch_size).unsqueeze(1),target] * bases_s[...,m][torch.arange(batch_size).unsqueeze(1),source]) \
#                                            + 2*torch.einsum('k, bek->be', weights_s, bases_c[...,m][torch.arange(batch_size).unsqueeze(1),target] * bases_s[...,m][torch.arange(batch_size).unsqueeze(1),source] - bases_s[...,m][torch.arange(batch_size).unsqueeze(1),target] * bases_c[...,m][torch.arange(batch_size).unsqueeze(1),source])
#             message = torch.einsum('be, bei, be->bei', edge_local_weights, f[torch.arange(batch_size).unsqueeze(1),source], node_weights[...,m])

#             f_out.scatter_add_(dim=1, src=message, index=target.unsqueeze(2).repeat(1,1,in_channels))
        
#         return self.w(f_out.permute(0,2,1))
    
class SpectralConvLocalSimp(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super(SpectralConvLocalSimp, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        nmodes, ndims, nmeasures = modes.shape
        self.modes = modes
        self.nmeasures = nmeasures
        self.scale = 1 / (in_channels * out_channels)

        self.weights_c = nn.Parameter(
            self.scale
            * torch.rand(in_channels, nmodes, nmeasures, dtype=torch.float
            )
        )
        self.weights_s = nn.Parameter(
            self.scale
            * torch.rand(in_channels, nmodes, nmeasures, dtype=torch.float
            )
        )
        self.weights_0 = nn.Parameter(
            self.scale
            * torch.rand(in_channels, 1, nmeasures, dtype=torch.float
            )
        )
        self.w = nn.Conv1d(in_channels, out_channels, 1)
        self.scale = nn.Linear(1, 1)
        self.kernel_linear = nn.Linear(2, 1)
    
    def forward(self, f, bases_c, bases_s, bases_0, nodes, normal_vectors, directed_edges, node_weights):
        '''
        Assemble local force f at each node
            u(x)  =  int K(x,y) f(y) dS(y)
        here K(x,y) = sum_k (wc_k + ws_k i) e^{2 k pi (x - y)/L}. 
        To ensure the result is real, we have wc_k = wc_{-k}, ws_k = -ws_{-k}
        
        directed_edges stores directed edges (x, y1), (x, y2), ..., (x, yj)
        send from xj to x:
            u(x)  =  sum K(x,yj) f(yj) S(yj)
                  =  sum_j sum_k [wc_k + ws_k i] [cos(2pikx/L) + sin(2pikx/L)i] [cos(2pikyj/L) - sin(2pikyj/L)i] f(yj) S(yj)   
                  =  sum_j sum_k [wc_k + ws_k i] [cos(2pikx/L) + sin(2pikx/L)i] [cos(2pikyj/L) - sin(2pikyj/L)i] f(yj) S(yj)    
                  =  sum_j sum_k [wc_k cos(2pikx/L) - ws_k sin(2pikx/L) + ws_k cos(2pikx/L)i + wc_k sin(2pikx/L)i] [cos(2pikyj/L) - sin(2pikyj/L)i] f(yj) S(yj)    
                  =  sum_j { wc_0 + 2sum_k [wc_k cos(2pikx/L)cos(2pikyj/L) - ws_k sin(2pikx/L)cos(2pikyj/L) + ws_k cos(2pikx/L)sin(2pikyj/L) + wc_k sin(2pikx/L)sin(2pikyj/L)] } f(yj) S(yj)    
       
            Parameters: 
                f                   : float[batch_size, in_channels, nnodes]
                bases_c, bases_s    : float[batch_size, nnodes, nmodes, nmeasures]
                bases_0             : float[batch_size, nnodes, 1, nmeasures]
                directed_edges : int[batch_size, max_nedges, 2, nmeasures] 
                node_weights   : float[batch_size, max_nedges, nmeasures]
                nodes:   float[batch_size, nnodes, ndims]
                
            Returns:
                f_out : float[batch_size, out_channels, max_nnodes]
        '''

        f = f.permute(0,2,1)
        batch_size, max_nnodes, in_channels = f.shape
        f_out = torch.zeros(batch_size, max_nnodes, in_channels, dtype=f.dtype, device=f.device)
        weight_per_node = torch.zeros(batch_size, max_nnodes, 1, dtype=f.dtype, device=f.device)

        # Message passing : compute message = edge_gradient_weights * (f_source - f_target) for each edge
        # target\source : int Tensor[batch_size, max_nedges]
        # message : float Tensor[batch_size, max_nedges, in_channels*ndims]
        for m in range(self.nmeasures):
            target, source = directed_edges[...,0,m], directed_edges[...,1,m]  # target and source nodes of edges  (bsz, max_nedges)

            diff_nodes = nodes[torch.arange(batch_size).unsqueeze(1),target] - nodes[torch.arange(batch_size).unsqueeze(1),source]  # (bsz, max_nedges, ndims)
            normal_vectors_source = normal_vectors[torch.arange(batch_size).unsqueeze(1),source]  # (bsz, max_nedges, ndims)
            r_square = torch.sum(diff_nodes**2, dim=-1, keepdim=True) + 1e-6
            gradn_logr = torch.sum(diff_nodes*normal_vectors_source, dim = -1, keepdim = True)/r_square  # (bsz, max_nedges,1)
            # logr = torch.log(r_square)
            # kernel = self.kernel_linear(torch.cat([gradn_logr, logr], dim=-1))  # (bsz, max_nedges,1)

            weights_c, weights_s, weights_0 = self.weights_c[...,m], self.weights_s[...,m], self.weights_0[...,0,m]
            
            edge_local_weights = weights_0 + 2*torch.einsum('ik, bek->bei', weights_c, bases_c[...,m][torch.arange(batch_size).unsqueeze(1),target] * bases_c[...,m][torch.arange(batch_size).unsqueeze(1),source] + bases_s[...,m][torch.arange(batch_size).unsqueeze(1),target] * bases_s[...,m][torch.arange(batch_size).unsqueeze(1),source]) \
                                           + 2*torch.einsum('ik, bek->bei', weights_s, bases_c[...,m][torch.arange(batch_size).unsqueeze(1),target] * bases_s[...,m][torch.arange(batch_size).unsqueeze(1),source] - bases_s[...,m][torch.arange(batch_size).unsqueeze(1),target] * bases_c[...,m][torch.arange(batch_size).unsqueeze(1),source])
            message = torch.einsum('bei, bei, be->bei', edge_local_weights, f[torch.arange(batch_size).unsqueeze(1),source], node_weights[...,m])
            
            f_out.scatter_add_(dim=1, src=message*gradn_logr, index=target.unsqueeze(2).repeat(1,1,in_channels))
            weight_per_node.scatter_add_(dim=1, src=node_weights[...,m:m+1], index=target.unsqueeze(2).repeat(1,1,1)) # (bsz, max_nnodes, 1)
            
            # scale = self.scale(weight_per_node)  # (bsz, max_nnodes, 1)
            # f_out = f_out*scale
            message_exact = torch.einsum('bei, be->bei', f[torch.arange(batch_size).unsqueeze(1),source], node_weights[...,m])
            f_out.scatter_add_(dim=1, src=message_exact*gradn_logr, index=target.unsqueeze(2).repeat(1,1,in_channels))
        return self.w(f_out.permute(0,2,1))
    

class PCNO(nn.Module):
    def __init__(
        self,
        ndims,
        modes,
        nmeasures,
        layers,
        local_modes = None,
        fc_dim=128,
        in_dim=3,
        out_dim=1,
        inv_L_scale_hyper = ['independently', 0.5, 2.0],
        act="gelu",
    ):
        super(PCNO, self).__init__()

        """
        The overall network. 
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. len(layers)-1 layers of the point cloud neural layers u' = (W + K + D)(u).
           linear functions  W: parameterized by self.ws; 
           integral operator K: parameterized by self.sp_convs with nmeasures different integrals
           differential operator D: parameterized by self.gws
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
            
            Parameters: 
                ndims : int 
                    Dimensionality of the problem
                modes : float[nmodes, ndims, nmeasures]
                    It contains nmodes modes k, and Fourier bases include : cos(k x), sin(k x), 1  
                    * We cannot have both k and -k
                    * k is not integer, and it has the form 2pi*K/L0  (K in Z)
                modes : float[nmodes, ndims, nmeasures]
                    It contains nmodes modes k, for local spectral convolution
                    * We cannot have both k and -k
                    * k is not integer, and it has the form 2pi*K/L0  (K in Z)
                nmeasures : int
                    Number of measures
                    There might be different integrals with different measures
                layers : list of int
                    number of channels of each layer
                    The lifting layer first lifts to layers[0]
                    The first Fourier layer maps between layers[0] and layers[1]
                    ...
                    The nlayers Fourier layer maps between layers[nlayers-1] and layers[nlayers]
                    The number of Fourier layers is len(layers) - 1
                fc_dim : int 
                    hidden layers for the projection layer, when fc_dim > 0, otherwise there is no hidden layer
                in_dim : int 
                    The number of channels for the input function
                    For example, when the coefficient function and locations (a(x, y), x, y) are inputs, in_dim = 3
                out_dim : int 
                    The number of channels for the output function

                inv_L_scale_hyper: 3 element hyperparameter list
                    Controls the update behavior of the length scale (L) for Fourier modes. The modes are scaled elementwise as:
                    k = k * inv_L_scale (where each spatial direction or measure may be scaled differently).
                    since k = 2pi K /L0, inv_L_scale is the inverse scale, 1/L = inv_L_scale * 1/L0 
                    Hyperparameters: 
                        train_inv_L_scale (bool or str): Update policy for inv_L_scale:
                            False: Disable training (fixed scaling).
                            'together': Train jointly with other parameters (shared optimizer).
                            'independently': Train with a separate optimizer.

                        inv_L_scale_min (float): Lower bound for scaling factor.

                        inv_L_scale_max (float): Upper bound for scaling factor.

                    Implementation Notes:
                        The effective scaling factor is computed via a sigmoid constraint:
                        inv_L_scale = inv_L_scale_min + (inv_L_scale_max - inv_L_scale_min) * sigmoid(inv_L_scale_latent)
                        This ensures inv_L_scale stays within [inv_L_scale_min, inv_L_scale_max] during optimization.

                act : string (default gelu)
                    The activation function

            
            Returns:
                Point cloud neural operator

        """
        
        self.register_buffer('modes', modes) 
        self.register_buffer('local_modes', local_modes) 
        self.nmeasures = nmeasures
        

        self.layers = layers
        self.fc_dim = fc_dim

        self.ndims = ndims
        self.in_dim = in_dim

        self.fc0 = nn.Linear(in_dim, layers[0])

        self.sp_convs = nn.ModuleList(
            [
                SpectralConv(in_size, out_size, modes)
                for in_size, out_size in zip(
                    self.layers, self.layers[1:]
                )
            ]
        )

        self.train_inv_L_scale, self.inv_L_scale_min, self.inv_L_scale_max  = inv_L_scale_hyper[0], inv_L_scale_hyper[1], inv_L_scale_hyper[2]
        # latent variable for inv_L_scale = inv_L_scale_min + (inv_L_scale_max - inv_L_scale_min) * sigmoid(inv_L_scale_latent)
        self.inv_L_scale_latent = nn.Parameter(torch.full((ndims, nmeasures), scaled_logit(torch.tensor(1.0), self.inv_L_scale_min, self.inv_L_scale_max)), requires_grad = bool(self.train_inv_L_scale))

        if local_modes is not None:
            self.sp_conv_locals = nn.ModuleList(
                SpectralConvLocalSimp(in_size, out_size, local_modes)
                    for in_size, out_size in zip(
                        self.layers, self.layers[1:]
                    )
            )
        else:
            self.sp_conv_locals = [None for _ in range(len(self.layers)-1)]
        
        self.ws = nn.ModuleList(
            [
                nn.Conv1d(in_size, out_size, 1)
                for in_size, out_size in zip(self.layers, self.layers[1:])
            ]
        )

        self.gws = nn.ModuleList(
            [
                nn.Conv1d(ndims*in_size, out_size, 1)
                for in_size, out_size in zip(self.layers, self.layers[1:])
            ]
        )

        if fc_dim > 0:
            self.fc1 = nn.Linear(layers[-1], fc_dim)
            self.fc2 = nn.Linear(fc_dim, out_dim)
        else:
            self.fc2 = nn.Linear(layers[-1], out_dim)

        self.act = _get_act(act)
        self.softsign = F.softsign

        self.normal_params = []  #  group of params which will be trained normally
        self.inv_L_scale_params = []    #  group of params which may be trained specially
        for _, param in self.named_parameters():
            if param is not self.inv_L_scale_latent :
                self.normal_params.append(param)
            else:
                if self.train_inv_L_scale == 'together':
                    self.normal_params.append(param)
                elif self.train_inv_L_scale == 'independently':
                    self.inv_L_scale_params.append(param)
                elif self.train_inv_L_scale == False:
                    continue
                else:
                    raise ValueError(f"{self.train_inv_L_scale} is not supported")
        

    def forward(self, x, aux):
        """
        Forward evaluation. 
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. len(layers)-1 layers of the point cloud neural layers u' = (W + K + D)(u).
           linear functions  W: parameterized by self.ws; 
           integral operator K: parameterized by self.sp_convs with nmeasures different integrals
           differential operator D: parameterized by self.gws
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
            
            Parameters: 
                x : Tensor float[batch_size, max_nnomdes, in_dim] 
                    Input data
                aux : list of Tensor, containing
                    node_mask : Tensor int[batch_size, max_nnomdes, 1]  
                                1: node; otherwise 0

                    nodes : Tensor float[batch_size, max_nnomdes, ndim]  
                            nodal coordinate; padding with 0

                    node_weights  : Tensor float[batch_size, max_nnomdes, nmeasures]  
                                    rho(x)dx used for nmeasures integrations; padding with 0

                    directed_edges : Tensor int[batch_size, max_nedges, 2]  
                                     direted edge pairs; padding with 0  
                                     gradient f(x) = sum_i pinvdx[:,i] * [f(xi) - f(x)] 

                    edge_gradient_weights      : Tensor float[batch_size, max_nedges, ndim] 
                                                 pinvdx on each directed edge 
                                                 gradient f(x) = sum_i pinvdx[:,i] * [f(xi) - f(x)] 

            
            Returns:
                G(x) : Tensor float[batch_size, max_nnomdes, out_dim] 
                       Output data

        """
        length = len(self.ws)
        normal_vectors = x[...,1:3]

        # nodes: float[batch_size, nnodes, ndims]
        node_mask, nodes, node_weights, directed_edges, edge_gradient_weights, close_directed_edges,  scaled_node_weights = aux
        # bases: float[batch_size, nnodes, nmodes, nmeasures]
        # scale the modes k  = k * ( inv_L_scale_min + (inv_L_scale_max - inv_L_scale_min)/(1 + exp(-self.inv_L_scale_latent) ))
        bases_c,  bases_s,  bases_0  = compute_Fourier_bases(nodes, self.modes * (scaled_sigmoid(self.inv_L_scale_latent, self.inv_L_scale_min , self.inv_L_scale_max))) 
        # node_weights: float[batch_size, nnodes, nmeasures]
        # wbases: float[batch_size, nnodes, nmodes, nmeasures]
        # set nodes with zero measure to 0
        wbases_c = torch.einsum("bxkw,bxw->bxkw", bases_c, node_weights)
        wbases_s = torch.einsum("bxkw,bxw->bxkw", bases_s, node_weights)
        wbases_0 = torch.einsum("bxkw,bxw->bxkw", bases_0, node_weights)
        
        # local bases
        # scale the modes k  = k * ( inv_L_scale_min + (inv_L_scale_max - inv_L_scale_min)/(1 + exp(-self.inv_L_scale_latent) ))
        if self.local_modes is not None:
            loc_bases_c,  loc_bases_s,  loc_bases_0  = compute_Fourier_bases(nodes, self.local_modes) 

        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        for i, (speconv, speconvlocal, w, gw) in enumerate(zip(self.sp_convs, self.sp_conv_locals, self.ws, self.gws)):
            x1 = speconv(x, bases_c, bases_s, bases_0, wbases_c, wbases_s, wbases_0)
            x2 = w(x)
            x3 = gw(self.softsign(compute_gradient(x, directed_edges, edge_gradient_weights)))
            if self.local_modes is not None:
                x4 = speconvlocal(x, loc_bases_c,  loc_bases_s,  loc_bases_0, nodes, normal_vectors, close_directed_edges, scaled_node_weights)
            else:
                x4 = 0
            
            if self.act is not None and i != length - 1:
                x = x + self.act(x1 + x2 + x3 + x4) 
            else:
                x  = x1 + x2 + x3 + x4

        x = x.permute(0, 2, 1)

        if self.fc_dim > 0:
            x = self.fc1(x)
            if self.act is not None:
                x = self.act(x)

        x = self.fc2(x)

       
        return x 
    


################################################################
# Training (Optimization)
################################################################

class CombinedOptimizer:
    '''
    CombinedOptimizer.
    train two param groups independently.
    the learning rates of two optimizers are lr, lr_ratio*lr respectively
    '''
    def __init__(self, params1, params2, betas, lr, lr_ratio, weight_decay):
        self.optimizer1 = Adam(
        params1,
        betas=betas,
        lr=lr,
        weight_decay=weight_decay,
        )
        if params2 == []:
            self.optimizer2 = None
        else:
            self.optimizer2 = Adam(
            params2,
            betas=betas,
            lr= lr_ratio*lr,
            weight_decay=weight_decay,
        )

    def step(self):
        self.optimizer1.step()
        if self.optimizer2:
            self.optimizer2.step()
    
    def zero_grad(self):
        self.optimizer1.zero_grad()
        if self.optimizer2:
            self.optimizer2.zero_grad()

    def state_dict(self):
        # Initialize an empty dictionary to store the state
        state = {}
        # Save the state of the first optimizer
        state['optimizer1'] = self.optimizer1.state_dict()
        if self.optimizer2:
            # Save the state of the second optimizer
            state['optimizer2'] = self.optimizer2.state_dict()
        return state

    def load_state_dict(self, state_dict):
        # Load the state of the first optimizer
        self.optimizer1.load_state_dict(state_dict['optimizer1'])
        if self.optimizer2:
            # Load the state of the second optimizer
            self.optimizer2.load_state_dict(state_dict['optimizer2'])

        


class Combinedscheduler_OneCycleLR:
    '''
    Combinedscheduler.
    scheduler two optimizers independently.
    '''
    def __init__(self, Combinedoptimizer,  max_lr, lr_ratio,
            div_factor, final_div_factor, pct_start,
            steps_per_epoch, epochs):

        self.scheduler1 = torch.optim.lr_scheduler.OneCycleLR(
            Combinedoptimizer.optimizer1, max_lr=max_lr,
            div_factor=div_factor, final_div_factor=final_div_factor, pct_start=pct_start,
            steps_per_epoch=steps_per_epoch, epochs=epochs)
        if Combinedoptimizer.optimizer2:
            self.scheduler2 = torch.optim.lr_scheduler.OneCycleLR(
                Combinedoptimizer.optimizer2, max_lr=max_lr*lr_ratio,
                div_factor=div_factor, final_div_factor=final_div_factor, pct_start=pct_start,
                steps_per_epoch=steps_per_epoch, epochs=epochs)
        else:
            self.scheduler2 = None

    def step(self):
        self.scheduler1.step()
        if self.scheduler2:
            self.scheduler2.step()

    def state_dict(self):
        # Initialize an empty dictionary to store the state
        state = {}

        # Save the state of the first scheduler
        state['scheduler1'] = self.scheduler1.state_dict()
        if self.scheduler2:
            # Save the state of the second scheduler
            state['scheduler2'] = self.scheduler2.state_dict()

        return state

    def load_state_dict(self, state_dict):
        # Load the state of the first scheduler
        self.scheduler1.load_state_dict(state_dict['scheduler1'])
        if self.scheduler2:
            # Load the state of the second scheduler
            self.scheduler2.load_state_dict(state_dict['scheduler2'])

        




# x_train, y_train, x_test, y_test are [n_data, n_x, n_channel] arrays
def PCNO_train(x_train, aux_train, y_train, x_test, aux_test, y_test, config, model, save_model_name="./PCNO_model", checkpoint_path=None):
    n_train, n_test = x_train.shape[0], x_test.shape[0]
    train_rel_l2_losses = []
    test_rel_l2_losses = []
    test_l2_losses = []
    normalization_x, normalization_y = config["train"]["normalization_x"], config["train"]["normalization_y"]
    normalization_dim_x, normalization_dim_y = config["train"]["normalization_dim_x"], config["train"]["normalization_dim_y"]
    non_normalized_dim_x, non_normalized_dim_y = config["train"]["non_normalized_dim_x"], config["train"]["non_normalized_dim_y"]
    
    ndims = model.ndims # n_train, size, n_channel
    print("In PCNO_train, ndims = ", ndims)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    if normalization_x:
        x_normalizer = UnitGaussianNormalizer(x_train, non_normalized_dim = non_normalized_dim_x, normalization_dim=normalization_dim_x)
        x_train = x_normalizer.encode(x_train)
        x_test = x_normalizer.encode(x_test)
        x_normalizer.to(device)
        
    if normalization_y:
        y_normalizer = UnitGaussianNormalizer(y_train, non_normalized_dim = non_normalized_dim_y, normalization_dim=normalization_dim_y)
        y_train = y_normalizer.encode(y_train)
        y_test = y_normalizer.encode(y_test)
        y_normalizer.to(device)

    if model.local_modes is None:
        if len(aux_train) < 7:
            aux_train = (aux_train + (torch.zeros(aux_train[0].shape[0]),) * 7)[:7]
            aux_test = (aux_test + (torch.zeros(aux_test[0].shape[0]),) * 7)[:7]


    node_mask_train, nodes_train, node_weights_train, directed_edges_train, edge_gradient_weights_train, close_directed_edges_train,  scaled_node_weights_train = aux_train
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train, node_mask_train, nodes_train, node_weights_train, directed_edges_train, edge_gradient_weights_train, close_directed_edges_train,  scaled_node_weights_train), 
                                               batch_size=config['train']['batch_size'], shuffle=True)
    
    node_mask_test, nodes_test, node_weights_test, directed_edges_test, edge_gradient_weights_test, close_directed_edges_test,  scaled_node_weights_test = aux_test
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test, node_mask_test, nodes_test, node_weights_test, directed_edges_test, edge_gradient_weights_test, close_directed_edges_test,  scaled_node_weights_test), 
                                               batch_size=config['train']['batch_size'], shuffle=False)
    
    myloss = LpLoss(d=1, p=2, size_average=False)

    optimizer = CombinedOptimizer(model.normal_params, model.inv_L_scale_params,
        betas=(0.9, 0.999),
        lr=config["train"]["base_lr"],
        lr_ratio = config["train"]["lr_ratio"],
        weight_decay=config["train"]["weight_decay"],
        )
    
    scheduler = Combinedscheduler_OneCycleLR(
        optimizer, max_lr=config['train']['base_lr'], lr_ratio = config["train"]["lr_ratio"],
        div_factor=2, final_div_factor=100,pct_start=0.2,
        steps_per_epoch=1, epochs=config['train']['epochs'])
    
    current_epoch, epochs = 0, config['train']['epochs']
    
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        # retrieve epoch and loss
        current_epoch = checkpoint['current_epoch'] + 1
        print("resetart from epoch : ", current_epoch)




    for ep in range(current_epoch, epochs):
        t1 = default_timer()
        train_rel_l2 = 0

        model.train()
        for x, y, node_mask, nodes, node_weights, directed_edges, edge_gradient_weights, close_directed_edges,  scaled_node_weights in train_loader:
            x, y, node_mask, nodes, node_weights, directed_edges, edge_gradient_weights, close_directed_edges,  scaled_node_weights = x.to(device), y.to(device), node_mask.to(device), nodes.to(device), node_weights.to(device), directed_edges.to(device), edge_gradient_weights.to(device), close_directed_edges.to(device), scaled_node_weights.to(device)

            batch_size_ = x.shape[0]
            optimizer.zero_grad()
            out = model(x, (node_mask, nodes, node_weights, directed_edges, edge_gradient_weights, close_directed_edges,  scaled_node_weights)) #.reshape(batch_size_,  -1)
            if normalization_y:
                out = y_normalizer.decode(out)
                y = y_normalizer.decode(y)
            out = out * node_mask #mask the padded value with 0,(1 for node, 0 for padding)
            loss = myloss(out.view(batch_size_,-1), y.view(batch_size_,-1))
            loss.backward()

            optimizer.step()
            train_rel_l2 += loss.item()

        test_l2 = 0
        test_rel_l2 = 0


        model.eval()
        with torch.no_grad():
            for x, y, node_mask, nodes, node_weights, directed_edges, edge_gradient_weights, close_directed_edges,  scaled_node_weights in test_loader:
                x, y, node_mask, nodes, node_weights, directed_edges, edge_gradient_weights, close_directed_edges,  scaled_node_weights = x.to(device), y.to(device), node_mask.to(device), nodes.to(device), node_weights.to(device), directed_edges.to(device), edge_gradient_weights.to(device), close_directed_edges.to(device), scaled_node_weights.to(device)

                batch_size_ = x.shape[0]
                out = model(x, (node_mask, nodes, node_weights, directed_edges, edge_gradient_weights, close_directed_edges, scaled_node_weights)) #.reshape(batch_size_,  -1)

                if normalization_y:
                    out = y_normalizer.decode(out)
                    y = y_normalizer.decode(y)
                out = out*node_mask #mask the padded value with 0,(1 for node, 0 for padding)
                test_rel_l2 += myloss(out.view(batch_size_,-1), y.view(batch_size_,-1)).item()
                test_l2 += myloss.abs(out.view(batch_size_,-1), y.view(batch_size_,-1)).item()




        scheduler.step()

        train_rel_l2/= n_train
        test_l2 /= n_test
        test_rel_l2/= n_test
        train_rel_l2_losses.append(train_rel_l2)
        test_rel_l2_losses.append(test_rel_l2)
        test_l2_losses.append(test_l2)
    

        t2 = default_timer()
        print("Epoch : ", ep, " Time: ", round(t2-t1,3), " Rel. Train L2 Loss : ", train_rel_l2, " Rel. Test L2 Loss : ", test_rel_l2, " Test L2 Loss : ", test_l2,
              " inv_L_scale: ",[round(float(x[0]), 3) for x in (scaled_sigmoid(model.inv_L_scale_latent, model.inv_L_scale_min, model.inv_L_scale_max)).cpu().tolist()],
              flush=True)
        if (ep %100 == 99) or (ep == epochs -1):    
            if save_model_name:
                torch.save(model.state_dict(), save_model_name + ".pth")

                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'current_epoch': ep,  # optional: to track training progress
                }, "checkpoint.pth")

            
    
    
    return train_rel_l2_losses, test_rel_l2_losses, test_l2_losses









