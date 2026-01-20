import math
import numpy as np
import torch
import sys
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import Dataset, Dataset, DistributedSampler

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
    
class Geo_emb(nn.Module):
    def __init__(self, geo_size, in_size, out_size):
        super(Geo_emb, self).__init__()
        self.geo_wx = nn.Conv1d(geo_size, out_size, 1, bias=False)
        self.wx = nn.Conv1d(in_size, out_size, 1, bias=False)
        self.w = nn.Conv1d(out_size, out_size, 1, bias=False)
    def forward(self, geo, x):
        '''
        geo: float[batch_size, geo_size, nnodes]
        x:  float[batch_size, in_size, nnodes]
        return: 
            float[batch_size, out_size, nnodes]
        '''

        return self.w(F.softsign(self.geo_wx(geo)) * self.wx(x))

    
class PCNO(nn.Module):
    def __init__(
        self,
        ndims,
        modes,
        nmeasures,
        layers,
        layer_selection = {'grad': True, 'geo': True, 'geointegral': True},
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
        self.nmeasures = nmeasures
        
        self.layer_selection = layer_selection
        self.layers = layers
        self.fc_dim = fc_dim


        self.ndims = ndims
        self.in_dim = in_dim

        # Lifting layer
        self.fc0 = nn.Linear(in_dim, layers[0])

        # Long-range spectral convolution layer
        self.sp_convs = nn.ModuleList(
            [
                SpectralConv(in_size, out_size, modes)
                for in_size, out_size in zip(
                    self.layers, self.layers[1:]
                )
            ]
        )
        # Linear operator outside the spectral convolution layer
        self.sp_ws = nn.ModuleList(
            [
                nn.Conv1d(out_size, out_size, 1, bias = False)
                for in_size, out_size in zip(self.layers, self.layers[1:])
            ]
        )
                
        # Cheap implementation for long-range spectral convolution layer with ny
        # Combine information with ny
        # [x, x\otimes n]    ->    \tilde{x} = W[x, x\otimes n]
        self.sp_convs_nws = nn.ModuleList(
            [
                nn.Conv1d(in_size * (ndims + 1), in_size, 1, bias=False)
                for in_size, out_size in zip(self.layers[1:], self.layers[1:])
            ]
        ) if layer_selection['geointegral'] else [None]*len(layers[1:])


        # Cheap implementation for long-range spectral convolution layer with nx
        # Combine information with nx
        # f    ->    \tilde{f} = W[f \otimes n]
        self.sp_convs_adj_nws = nn.ModuleList(
            [
                nn.Conv1d(out_size * ndims, out_size, 1, bias = False)
                for in_size, out_size in zip(self.layers, self.layers[1:])
            ]
        ) if layer_selection['geointegral'] else [None]*len(layers[1:])
        
        # Linear operator
        self.ws = nn.ModuleList(
            [
                nn.Conv1d(in_size, out_size, 1)
                for in_size, out_size in zip(self.layers, self.layers[1:])
            ]
        )

        # Gradient operator
        self.gws = nn.ModuleList(
            [
                nn.Conv1d(ndims*in_size, out_size, 1, bias = False)
                for in_size, out_size in zip(self.layers, self.layers[1:])
            ]
        ) if layer_selection['grad'] else [None]*len(layers[1:])

        # Short-range geo layer, geo includes nx, d^(1)(nx)
        self.geo_embs = nn.ModuleList([Geo_emb(ndims*(ndims+1), in_size, out_size) for in_size, out_size in zip(self.layers, self.layers[1:])]) if layer_selection['geo'] else [None]*len(layers[1:])
        
        # Projection layer
        if fc_dim > 0:
            self.fc1 = nn.Linear(layers[-1], fc_dim)
            self.fc2 = nn.Linear(fc_dim, out_dim)
        else:
            self.fc2 = nn.Linear(layers[-1], out_dim)


        self.train_inv_L_scale, self.inv_L_scale_min, self.inv_L_scale_max  = inv_L_scale_hyper[0], inv_L_scale_hyper[1], inv_L_scale_hyper[2]
        # latent variable for inv_L_scale = inv_L_scale_min + (inv_L_scale_max - inv_L_scale_min) * sigmoid(inv_L_scale_latent)
        self.inv_L_scale_latent = nn.Parameter(torch.full((ndims, nmeasures), scaled_logit(torch.tensor(1.0), self.inv_L_scale_min, self.inv_L_scale_max)), requires_grad = bool(self.train_inv_L_scale))
        
        
        num_branches = 2 
        if layer_selection['grad']: num_branches += 1
        if layer_selection['geo']: num_branches += 1
        self.scale_factor = 1.0 / num_branches  # 1/num or 1/sqrt(num) ??

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

        # nodes: float[batch_size, nnodes, ndims]
        node_mask, nodes, node_weights, directed_edges, edge_gradient_weights, outward_normals = aux
        # bases: float[batch_size, nnodes, nmodes]
        # scale the modes k  = k * ( inv_L_scale_min + (inv_L_scale_max - inv_L_scale_min)/(1 + exp(-self.inv_L_scale_latent) ))
        bases_c,  bases_s,  bases_0  = compute_Fourier_bases(nodes, self.modes * (scaled_sigmoid(self.inv_L_scale_latent, self.inv_L_scale_min , self.inv_L_scale_max))) 
        # node_weights: float[batch_size, nnodes, nmeasures]
        # wbases: float[batch_size, nnodes, nmodes, nmeasures]
        # set nodes with zero measure to 0
        wbases_c = torch.einsum("bxkw,bxw->bxkw", bases_c, node_weights)
        wbases_s = torch.einsum("bxkw,bxw->bxkw", bases_s, node_weights)
        wbases_0 = torch.einsum("bxkw,bxw->bxkw", bases_0, node_weights)

        geo = torch.cat([outward_normals,compute_gradient(outward_normals, directed_edges, edge_gradient_weights)], dim=1) if self.layer_selection['geo'] else None

        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        for i, (speconv, spw, spconvnw, spconvadjnw, w, gw, geo_emb) in enumerate(zip(self.sp_convs, self.sp_ws, self.sp_convs_nws, self.sp_convs_adj_nws, self.ws, self.gws, self.geo_embs)):
            
            if self.layer_selection['geointegral']:
                x1 = speconv( spconvnw(  torch.cat([x] + [x * outward_normals[:, i:i+1, :] for i in range(outward_normals.size(1))], dim=1)  ), bases_c, bases_s, bases_0, wbases_c, wbases_s, wbases_0)
                x1 = spw(x1) + spconvadjnw(torch.cat([x1 * outward_normals[:, i:i+1, :] for i in range(outward_normals.size(1))], dim=1))
            else:
                x1 = speconv(x, bases_c, bases_s, bases_0, wbases_c, wbases_s, wbases_0)
                x1 = spw(x1)
                
            x2 = w(x)


            if self.layer_selection['grad']:
                x_grad = gw(self.softsign(compute_gradient(x, directed_edges, edge_gradient_weights)))
            else:
                x_grad = 0

            if self.layer_selection['geo']:
                x_geo = geo_emb(geo,  x)
            else:
                x_geo = 0

            
            if self.act is not None and i != length - 1:
                x = x + self.act(self.scale_factor*(x1 + x2 + x_grad + x_geo))
            else:
                x = self.scale_factor*(x1 + x2 + x_grad + x_geo)

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
def PCNO_train_multidist(x_train, aux_train, y_train, x_test_list, aux_test_list, y_test_list,  config, model, label_test_list = None, save_model_name="./PCNO_model", checkpoint_path=None):
    assert len(x_test_list) == len(y_test_list) == len(aux_test_list), "The length of x_test_list, y_test_list and aux_test_list should be the same"
    n_distributions = len(x_test_list)
    n_train= x_train.shape[0]
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
        for i in range(n_distributions):
            x_test_list[i] = x_normalizer.encode(x_test_list[i])
        x_normalizer.to(device)
        
    if normalization_y:
        y_normalizer = UnitGaussianNormalizer(y_train, non_normalized_dim = non_normalized_dim_y, normalization_dim=normalization_dim_y)
        y_train = y_normalizer.encode(y_train)
        for i in range(n_distributions):
            y_test_list[i] = y_normalizer.encode(y_test_list[i])
        y_normalizer.to(device)


    node_mask_train, nodes_train, node_weights_train, directed_edges_train, edge_gradient_weights_train, geo_train = aux_train
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train, node_mask_train, nodes_train, node_weights_train, directed_edges_train, edge_gradient_weights_train, geo_train), 
                                               batch_size=config['train']['batch_size'], shuffle=True)
    
    test_loaders = []

    for i in range(n_distributions):
        node_mask_test, nodes_test, node_weights_test, directed_edges_test, edge_gradient_weights_test, geo_test = aux_test_list[i]
        sub_dataset = torch.utils.data.TensorDataset(
            x_test_list[i], 
            y_test_list[i], 
            node_mask_test, 
            nodes_test, 
            node_weights_test, 
            directed_edges_test, 
            edge_gradient_weights_test,
            geo_test
        )
        sub_loader = torch.utils.data.DataLoader(sub_dataset, batch_size=config['train']['batch_size'], shuffle=False)
        try:
            name = label_test_list[i]
        except:
            name = f"Distribution_{i}"
        test_loaders.append((name, sub_loader))
  
    
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
        steps_per_epoch=len(train_loader), epochs=config['train']['epochs'])
    
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
        for x, y, node_mask, nodes, node_weights, directed_edges, edge_gradient_weights, geo in train_loader:
            x, y, node_mask, nodes, node_weights, directed_edges, edge_gradient_weights, geo = x.to(device), y.to(device), node_mask.to(device), nodes.to(device), node_weights.to(device), directed_edges.to(device), edge_gradient_weights.to(device), geo.to(device)

            batch_size_ = x.shape[0]
            optimizer.zero_grad()
            out = model(x, (node_mask, nodes, node_weights, directed_edges, edge_gradient_weights, geo)) #.reshape(batch_size_,  -1)
            if normalization_y:
                out = y_normalizer.decode(out)
                y = y_normalizer.decode(y)
            out = out * node_mask #mask the padded value with 0,(1 for node, 0 for padding)
            loss = myloss(out.view(batch_size_,-1), y.view(batch_size_,-1))
            
            loss.backward()

            optimizer.step()
            
            scheduler.step()
            
            train_rel_l2 += loss.item()


        test_rel_l2_dict = {}
        test_l2_dict = {}

        model.eval()
        with torch.no_grad():
            for name, loader in test_loaders:
                test_l2 = 0
                test_rel_l2 = 0

                for x, y, node_mask, nodes, node_weights, directed_edges, edge_gradient_weights, geo in loader:
                    x, y, node_mask, nodes, node_weights, directed_edges, edge_gradient_weights, geo = x.to(device), y.to(device), node_mask.to(device), nodes.to(device), node_weights.to(device), directed_edges.to(device), edge_gradient_weights.to(device), geo.to(device)

                    batch_size_ = x.shape[0]
                    out = model(x, (node_mask, nodes, node_weights, directed_edges, edge_gradient_weights, geo)) #.reshape(batch_size_,  -1)

                    if normalization_y:
                        out = y_normalizer.decode(out)
                        y = y_normalizer.decode(y)
                    out = out*node_mask #mask the padded value with 0,(1 for node, 0 for padding)
                    test_rel_l2 += myloss(out.view(batch_size_,-1), y.view(batch_size_,-1)).item()
                    test_l2 += myloss.abs(out.view(batch_size_,-1), y.view(batch_size_,-1)).item()
                test_l2 /= len(loader.dataset)
                test_rel_l2 /= len(loader.dataset)
                test_rel_l2_dict[name] = test_rel_l2
                test_l2_dict[name] = test_l2
    

        

        train_rel_l2/= n_train

        train_rel_l2_losses.append(train_rel_l2)
        test_rel_l2_losses.append(test_rel_l2_dict)
        test_l2_losses.append(test_l2_dict)
    

        t2 = default_timer()
        print("Epoch : ", ep, " Time: ", round(t2-t1,3), " Rel. Train L2 Loss : ", train_rel_l2, " Rel. Test L2 Loss : ", test_rel_l2_dict, " Test L2 Loss : ", test_l2_dict,
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


    node_mask_train, nodes_train, node_weights_train, directed_edges_train, edge_gradient_weights_train, geo_train = aux_train
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train, node_mask_train, nodes_train, node_weights_train, directed_edges_train, edge_gradient_weights_train, geo_train), 
                                               batch_size=config['train']['batch_size'], shuffle=True)
    
    node_mask_test, nodes_test, node_weights_test, directed_edges_test, edge_gradient_weights_test, geo_test = aux_test
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test, node_mask_test, nodes_test, node_weights_test, directed_edges_test, edge_gradient_weights_test, geo_test), 
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
        steps_per_epoch=len(train_loader), epochs=config['train']['epochs'])
    
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
        for x, y, node_mask, nodes, node_weights, directed_edges, edge_gradient_weights, geo in train_loader:
            x, y, node_mask, nodes, node_weights, directed_edges, edge_gradient_weights, geo = x.to(device), y.to(device), node_mask.to(device), nodes.to(device), node_weights.to(device), directed_edges.to(device), edge_gradient_weights.to(device), geo.to(device)

            batch_size_ = x.shape[0]
            optimizer.zero_grad()
            out = model(x, (node_mask, nodes, node_weights, directed_edges, edge_gradient_weights, geo)) #.reshape(batch_size_,  -1)
            if normalization_y:
                out = y_normalizer.decode(out)
                y = y_normalizer.decode(y)
            out = out * node_mask #mask the padded value with 0,(1 for node, 0 for padding)
            loss = myloss(out.view(batch_size_,-1), y.view(batch_size_,-1))
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            train_rel_l2 += loss.item()

        test_l2 = 0
        test_rel_l2 = 0


        model.eval()
        with torch.no_grad():
            for x, y, node_mask, nodes, node_weights, directed_edges, edge_gradient_weights, geo in test_loader:
                x, y, node_mask, nodes, node_weights, directed_edges, edge_gradient_weights, geo = x.to(device), y.to(device), node_mask.to(device), nodes.to(device), node_weights.to(device), directed_edges.to(device), edge_gradient_weights.to(device), geo.to(device)

                batch_size_ = x.shape[0]
                out = model(x, (node_mask, nodes, node_weights, directed_edges, edge_gradient_weights, geo)) #.reshape(batch_size_,  -1)

                if normalization_y:
                    out = y_normalizer.decode(out)
                    y = y_normalizer.decode(y)
                out = out*node_mask #mask the padded value with 0,(1 for node, 0 for padding)
                test_rel_l2 += myloss(out.view(batch_size_,-1), y.view(batch_size_,-1)).item()
                test_l2 += myloss.abs(out.view(batch_size_,-1), y.view(batch_size_,-1)).item()




        

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






        
def PCNO_train_parallel(x_train, aux_train, y_train, x_test, aux_test, y_test, config, model, rank=0, local_rank = 0, world_size=1, save_model_name="./PCNO_parallel_model", checkpoint_path=None):
    """
    PCNO_train function for parallel GPU training
    Assumes:
      - dist.init_process_group already called
      - model is already wrapped by DistributedDataParallel
      - torch.cuda.set_device(local_rank) already called
    Args:
        model: DDP-wrapped model (should already be wrapped with DistributedDataParallel)
        rank: rank of the current process (0 is the main process)
        local_rank: local_rank of the current process on the device 
        world_size: total number of processes
    """
    assert(world_size > 1)
    device = torch.device(f'cuda:{local_rank}')
    
    n_train, n_test = x_train.shape[0], x_test.shape[0]
    
    if rank == 0:
        print(f"[DDP] world_size={world_size}")
        print(f"[DDP] n_train={n_train}, n_test={n_test}")
        print(f"[DDP] batch_size per GPU={config['train']['batch_size']}")
        print(f"[DDP] effective batch_size={config['train']['batch_size'] * world_size}")



    normalization_x, normalization_y = config["train"]["normalization_x"], config["train"]["normalization_y"]
    normalization_dim_x, normalization_dim_y = config["train"]["normalization_dim_x"], config["train"]["normalization_dim_y"]
    non_normalized_dim_x, non_normalized_dim_y = config["train"]["non_normalized_dim_x"], config["train"]["non_normalized_dim_y"]
    
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

    train_rel_l2_losses = []
    test_rel_l2_losses = []
    test_l2_losses = []
    
    # ------------------------
    # Dataset / Sampler / Loader
    # ------------------------
    node_mask_train, nodes_train, node_weights_train, directed_edges_train, edge_gradient_weights_train, geo_train = aux_train
    train_dataset = torch.utils.data.TensorDataset(x_train, y_train, node_mask_train, nodes_train, node_weights_train, directed_edges_train, edge_gradient_weights_train, geo_train)
    train_sampler = torch.utils.data.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=42, drop_last=False)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['train']['batch_size'], sampler=train_sampler, shuffle=False)
    
    
    node_mask_test, nodes_test, node_weights_test, directed_edges_test, edge_gradient_weights_test, geo_test = aux_test
    test_dataset = torch.utils.data.TensorDataset(x_test, y_test, node_mask_test, nodes_test, node_weights_test, directed_edges_test, edge_gradient_weights_test, geo_test)
    test_sampler = torch.utils.data.DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False, seed=42, drop_last=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config['train']['batch_size'], sampler=test_sampler, shuffle=False)
    


    myloss = LpLoss(d=1, p=2, size_average=False)

    optimizer = CombinedOptimizer(model.module.normal_params, model.module.inv_L_scale_params,
        betas=(0.9, 0.999),
        lr=config["train"]["base_lr"],
        lr_ratio = config["train"]["lr_ratio"],
        weight_decay=config["train"]["weight_decay"],
        )
    
    scheduler = Combinedscheduler_OneCycleLR(
        optimizer, max_lr=config['train']['base_lr'], lr_ratio = config["train"]["lr_ratio"],
        div_factor=2, final_div_factor=100,pct_start=0.2,
        steps_per_epoch=len(train_loader), epochs=config['train']['epochs'])
    
    current_epoch, epochs = 0, config['train']['epochs']
    
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.module.load_state_dict(checkpoint['model_state_dict'])
        
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        # retrieve epoch and loss
        current_epoch = checkpoint['current_epoch'] + 1
        if rank == 0:
            print("resetart from epoch : ", current_epoch)


       
    if rank == 0:
        print(f"n_train = ", n_train, " n_test = ", n_test)
        print(f"Batch size per GPU: {config['train']['batch_size']}")
        print(f"Effective batch size: {config['train']['batch_size'] * world_size}")
        print(f"Training batches per GPU: {len(train_loader)}")
        print(f"Test batches per GPU: {len(test_loader)}")
        
        


    for ep in range(current_epoch, epochs):
        train_sampler.set_epoch(ep)
        
        t1 = default_timer()
        train_rel_l2 = 0.0
        num_train_samples = 0
        
        model.train()
        for x, y, node_mask, nodes, node_weights, directed_edges, edge_gradient_weights, geo in train_loader:
            x, y, node_mask, nodes, node_weights, directed_edges, edge_gradient_weights, geo = x.to(device, non_blocking=True), y.to(device, non_blocking=True), node_mask.to(device, non_blocking=True), nodes.to(device, non_blocking=True), node_weights.to(device, non_blocking=True), directed_edges.to(device, non_blocking=True), edge_gradient_weights.to(device, non_blocking=True), geo.to(device, non_blocking=True)

            batch_size_ = x.shape[0]
            
            # Forward pass
            optimizer.zero_grad()
            out = model(x, (node_mask, nodes, node_weights, directed_edges, edge_gradient_weights, geo)) #.reshape(batch_size_,  -1)
            if normalization_y:
                out = y_normalizer.decode(out)
                y = y_normalizer.decode(y)
            out = out * node_mask #mask the padded value with 0,(1 for node, 0 for padding)
            loss = myloss(out.view(batch_size_,-1), y.view(batch_size_,-1))
            
            # Backward pass (gradients are automatically synchronized)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_rel_l2 += loss.item()
            num_train_samples += batch_size_
            
        # synchronize train error
        train_rel_l2_tensor = torch.tensor(train_rel_l2, device=device)
        train_count_tensor = torch.tensor(num_train_samples, device=device)
        dist.all_reduce(train_rel_l2_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(train_count_tensor, op=dist.ReduceOp.SUM)
        train_rel_l2 = train_rel_l2_tensor.item() / train_count_tensor.item()
        
        
        # TEST
        model.eval()
        test_l2 = 0.0
        test_rel_l2 = 0.0
        num_test_samples = 0
        with torch.no_grad():
            for x, y, node_mask, nodes, node_weights, directed_edges, edge_gradient_weights, geo in test_loader:
                x, y, node_mask, nodes, node_weights, directed_edges, edge_gradient_weights, geo = x.to(device, non_blocking=True), y.to(device, non_blocking=True), node_mask.to(device, non_blocking=True), nodes.to(device, non_blocking=True), node_weights.to(device, non_blocking=True), directed_edges.to(device, non_blocking=True), edge_gradient_weights.to(device, non_blocking=True), geo.to(device, non_blocking=True)

                batch_size_ = x.shape[0]
                out = model(x, (node_mask, nodes, node_weights, directed_edges, edge_gradient_weights, geo)) #.reshape(batch_size_,  -1)

                if normalization_y:
                    out = y_normalizer.decode(out)
                    y = y_normalizer.decode(y)
                out = out*node_mask #mask the padded value with 0,(1 for node, 0 for padding)
                test_rel_l2 += myloss(out.view(batch_size_,-1), y.view(batch_size_,-1)).item()
                test_l2 += myloss.abs(out.view(batch_size_,-1), y.view(batch_size_,-1)).item()
                num_test_samples += batch_size_
                
            # synchronize train error
            test_l2_tensor = torch.tensor(test_l2, device=device)
            test_rel_l2_tensor = torch.tensor(test_rel_l2, device=device)
            test_count_tensor = torch.tensor(num_test_samples, device=device)
            dist.all_reduce(test_rel_l2_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(test_l2_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(test_count_tensor, op=dist.ReduceOp.SUM)

            test_l2 = test_l2_tensor.item() / test_count_tensor.item()
            test_rel_l2 = test_rel_l2_tensor.item() / test_count_tensor.item()



        
        
        train_rel_l2_losses.append(train_rel_l2)
        test_rel_l2_losses.append(test_rel_l2)
        test_l2_losses.append(test_l2)
    

        t2 = default_timer()
        
        if rank == 0:
            print("Epoch : ", ep, " Time: ", round(t2-t1,3), " Rel. Train L2 Loss : ", train_rel_l2, " Rel. Test L2 Loss : ", test_rel_l2, " Test L2 Loss : ", test_l2, flush=True)
        
        
            if ((ep %100 == 99) or (ep == epochs -1)) and save_model_name:    
                torch.save({
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'current_epoch': ep,  # optional: to track training progress
                }, "checkpoint_parallel.pth")

        # Synchronize all processes
        dist.barrier()
    
    
    return train_rel_l2_losses, test_rel_l2_losses, test_l2_losses