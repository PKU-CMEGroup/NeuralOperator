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
from pcno.geo_utility import compute_edge_gradient_weights

    

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


def compute_Fourier_modes_helper(ndims, nks, Ls):
    '''
    Compute Fourier modes number k
    Fourier bases are cos(kx), sin(kx), 1
    * We cannot have both k and -k

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
    
    k_pairs = k_pairs[np.argsort(k_pair_mag), :]
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
    
def normfix(wbases_c, wbases_s, wbases_0, nx, modes, sigma):

    temp = torch.einsum("bxd,kdw->bxkw", nx, modes) 
    correction = pow(2*np.pi*sigma**2, 1/2)*torch.exp(-sigma**2/2*temp**2)
    wbases_c = wbases_c * correction
    wbases_s = wbases_s * correction
    wbases_0 = wbases_0 * pow(2*torch.pi*sigma**2, 1/2)

    return wbases_c, wbases_s, wbases_0

def normfix_multi(wbases_c, wbases_s, wbases_0, nx, modes, sigma_list, k_list):
    n_dim = nx.shape[-1]
    wbases_c_list, wbases_s_list, wbases_0_list = [] ,[], []
    for (sigma,k) in zip( sigma_list, k_list):
        num_modes = ((2*k+1)**n_dim -1) //2
        wbases_c1, wbases_s1, wbases_01 = normfix(wbases_c[:,:,:num_modes,:], wbases_s[:,:,:num_modes,:], wbases_0, nx, modes[:num_modes], sigma)
        wbases_c_list.append(wbases_c1)
        wbases_s_list.append(wbases_s1)
        wbases_0_list.append(wbases_01)

    return wbases_c_list, wbases_s_list, wbases_0_list
    
def truncate(nx, x, wbases_c, wbases_s, wbases_0, bases_c, bases_s, bases_0, modes, sigma):
    temp = torch.einsum("bxd,kdw->bxkw", nx, modes) 
    correction = pow(2*torch.pi*sigma**2, 1/2)*torch.exp(-sigma**2/2*temp**2)
    wbases_c_fix = wbases_c * correction
    wbases_s_fix = wbases_s * correction
    wbases_0_fix = wbases_0 * pow(2*torch.pi*sigma**2, 1/2)
    x_c_hat =  torch.einsum("bix,bxkw->bikw", x, wbases_c_fix)
    x_s_hat = -torch.einsum("bix,bxkw->bikw", x, wbases_s_fix)
    x_0_hat =  torch.einsum("bix,bxkw->bikw", x, wbases_0_fix)

    x = torch.einsum("bokw,bxkw->box", x_0_hat, bases_0)  + 2*torch.einsum("bokw,bxkw->box", x_c_hat, bases_c) -  2*torch.einsum("bokw,bxkw->box", x_s_hat, bases_s) 
    
    return x

def direct(x, wbases_c, wbases_s, wbases_0, bases_c, bases_s, bases_0):

    x_c_hat =  torch.einsum("bix,bxkw->bikw", x, wbases_c)
    x_s_hat = -torch.einsum("bix,bxkw->bikw", x, wbases_s)
    x_0_hat =  torch.einsum("bix,bxkw->bikw", x, wbases_0)

    x = torch.einsum("bokw,bxkw->box", x_0_hat, bases_0)  + 2*torch.einsum("bokw,bxkw->box", x_c_hat, bases_c) -  2*torch.einsum("bokw,bxkw->box", x_s_hat, bases_s) 
    
    return x

def normgradient_bases(wbases_c, wbases_s, wbases_0, nx, modes):
    temp = torch.einsum("bxd,kdw->bxkw", nx, modes) 
    wbases_c = wbases_c * temp
    wbases_s = wbases_s * temp
    wbases_0 = wbases_0 * 0
    return wbases_c, wbases_s, wbases_0   

def truncate_onestep(x, nx, wbases_c_fix, wbases_s_fix, wbases_0_fix, bases_c, bases_s, bases_0, k_max):
    '''
    x: (b,i,n)

    return: (b,i,n)
    '''
    n_dim = nx.shape[-1]
    num_modes = ((2*k_max+1)**n_dim- 1 )//2
    x_c_hat =  torch.einsum("bix,bxkw->bikw", x, wbases_c_fix[:,:,:num_modes,:])
    x_s_hat = -torch.einsum("bix,bxkw->bikw", x, wbases_s_fix[:,:,:num_modes,:])
    x_0_hat =  torch.einsum("bix,bxkw->bikw", x, wbases_0_fix)

    x_trun = torch.einsum("bokw,bxkw->box", x_0_hat, bases_0)  + 2*torch.einsum("bokw,bxkw->box", x_c_hat, bases_c[:,:,:num_modes,:]) -  2*torch.einsum("bokw,bxkw->box", x_s_hat, bases_s[:,:,:num_modes,:]) 

    coffe = torch.norm(x, dim = -1, keepdim = True)/(torch.norm(x_trun, dim = -1, keepdim = True) + 1e-5)
    x_trun = x_trun * coffe
    return x_trun

def truncate_multistep(x, nx, wbases_c_fix_list, wbases_s_fix_list, wbases_0_fix_list, bases_c, bases_s, bases_0, k_max_list):
    '''
    x: (b,i,n)

    return: (b,i,o,n)
    '''
    x_trun_list = []
    for i,(wbases_c_fix, wbases_s_fix, wbases_0_fix, k_max) in enumerate(zip(wbases_c_fix_list, wbases_s_fix_list, wbases_0_fix_list, k_max_list)):
        if i==0:
            x_trun = truncate_onestep(x, nx, wbases_c_fix, wbases_s_fix, wbases_0_fix, bases_c, bases_s, bases_0, k_max)
            x_trun_list.append(x_trun.unsqueeze(2))
        else:
            x_trun_new = truncate_onestep(x - x_trun, nx, wbases_c_fix, wbases_s_fix, wbases_0_fix, bases_c, bases_s, bases_0, k_max)
            x_trun = x_trun + x_trun_new
            x_trun_list.append(x_trun.unsqueeze(2))
        x_trun_tensor = torch.cat(x_trun_list, dim = 2)

    return x_trun_tensor

class PCNO_NORMFIX(nn.Module):
    def __init__(
        self,
        ndims,
        modes,
        nmeasures,
        layers,
        g_dim = 16,
        fc_dim=128,
        in_dim=3,
        out_dim=1,
        train_sp_L = 'independently',
        act="gelu",
        sigma_list = [0.8,0.6,0.4,0.2],
        k_list = [4,4,4,4]
    ):
        super(PCNO_NORMFIX, self).__init__()

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
                nmeasures : int
                    Number of measures
                    There might be different integrals with different measures
                train_sp_L: bool or str
                    The way to train sp_L.
                    False: means we dont train sp_L
                    'together': means sp_L will be trained with other params together.
                    'independently' : means sp_L will be trained in another optimizer independently.
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
                act : string (default gelu)
                    The activation function

            
            Returns:
                Point cloud neural operator

        """
        self.modes = modes
        self.nmeasures = nmeasures
        

        self.layers = layers
        self.fc_dim = fc_dim

        self.ndims = ndims
        self.in_dim = in_dim

        self.fc0 = nn.Linear(in_dim, layers[0])
        
        self.sigma_list = sigma_list
        self.k_list = k_list
        self.sp_convs = nn.ModuleList(
            [
                SpectralConv(in_size, out_size, modes)
                for in_size, out_size in zip(
                    self.layers, self.layers[1:]
                )
            ]
        )

        self.train_sp_L = train_sp_L
        self.sp_L = nn.Parameter(torch.ones(ndims, nmeasures), requires_grad = bool(train_sp_L))

        self.ws = nn.ModuleList(
            [
                nn.Conv1d(in_size, out_size, 1)
                for in_size, out_size in zip(self.layers, self.layers[1:])
            ]
        )
        self.ws_res = nn.ModuleList(
            [
                nn.Conv1d(len(k_list)*in_size, out_size, 1)
                for in_size, out_size in zip(self.layers, self.layers[1:])
            ]
        )

        self.gw1s = nn.ModuleList(
            [
                nn.Conv1d(in_size, g_dim,  1)
                for in_size in self.layers[:-1]
            ]
        )

        self.gw2s = nn.ModuleList(
            [
                nn.Conv1d(g_dim * ndims, out_size, 1)
                for out_size in self.layers[1:]
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
        self.sp_L_params = []    #  group of params which may be trained specially
        for _, param in self.named_parameters():
            if param is not self.sp_L :
                self.normal_params.append(param)
            else:
                if self.train_sp_L == 'together':
                    self.normal_params.append(param)
                elif self.train_sp_L == 'independently':
                    self.sp_L_params.append(param)
                elif self.train_sp_L == False:
                    continue
                else:
                    raise ValueError(f"{self.train_sp_L} is not supported")
        

    def forward(self, x, aux, nx):
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
        node_mask, nodes, node_weights, directed_edges, edge_gradient_weights = aux
        # bases: float[batch_size, nnodes, nmodes]
        bases_c,  bases_s,  bases_0  = compute_Fourier_bases(nodes, self.modes * self.sp_L)
        # node_weights: float[batch_size, nnodes, nmeasures]
        # wbases: float[batch_size, nnodes, nmodes, nmeasures]
        # set nodes with zero measure to 0
        wbases_c = torch.einsum("bxkw,bxw->bxkw", bases_c, node_weights)
        wbases_s = torch.einsum("bxkw,bxw->bxkw", bases_s, node_weights)
        wbases_0 = torch.einsum("bxkw,bxw->bxkw", bases_0, node_weights)

        wbases_c_fix_list, wbases_s_fix_list, wbases_0_fix_list = normfix_multi(wbases_c, wbases_s, wbases_0, nx, self.modes * self.sp_L, self.sigma_list, self.k_list)

        
        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        for i, (speconv, w, gw1, gw2, w_res) in enumerate(zip(self.sp_convs, self.ws, self.gw1s, self.gw2s, self.ws_res)):
            x1 = speconv(x, bases_c, bases_s, bases_0, wbases_c, wbases_s, wbases_0)
            x2 = w(x)
            x3 = gw2(self.softsign(compute_gradient(gw1(x), directed_edges, edge_gradient_weights)))
            res = x.unsqueeze(2) - truncate_multistep(x, nx, wbases_c_fix_list, wbases_s_fix_list, wbases_0_fix_list, bases_c, bases_s, bases_0, self.k_list)
            res = res.reshape(res.shape[0], -1, res.shape[-1])
            x = x1 + x2 + x3  + w_res(self.act(res))
            # x = x2 + x3  + w_res(self.act(res))

            if self.act is not None and i != length - 1:
                x = self.act(x) 
            

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
def PCNO_NORMFIX_train(x_train, nx_train, aux_train, y_train, x_test, nx_test, aux_test, y_test, config, model, save_model_name="./PCNO_NORMFIX_model", checkpoint_path=None):
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


    node_mask_train, nodes_train, node_weights_train, directed_edges_train, edge_gradient_weights_train = aux_train
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train, nx_train, node_mask_train, nodes_train, node_weights_train, directed_edges_train, edge_gradient_weights_train), 
                                               batch_size=config['train']['batch_size'], shuffle=True)
    
    node_mask_test, nodes_test, node_weights_test, directed_edges_test, edge_gradient_weights_test = aux_test
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test, nx_test, node_mask_test, nodes_test, node_weights_test, directed_edges_test, edge_gradient_weights_test), 
                                               batch_size=config['train']['batch_size'], shuffle=False)
    
    myloss = LpLoss(d=1, p=2, size_average=False)

    optimizer = CombinedOptimizer(model.normal_params,model.sp_L_params,
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
        for x, y, nx, node_mask, nodes, node_weights, directed_edges, edge_gradient_weights in train_loader:
            x, y, nx, node_mask, nodes, node_weights, directed_edges, edge_gradient_weights = x.to(device), y.to(device), nx.to(device),node_mask.to(device), nodes.to(device), node_weights.to(device), directed_edges.to(device), edge_gradient_weights.to(device)

            batch_size_ = x.shape[0]
            optimizer.zero_grad()
            out = model(x, (node_mask, nodes, node_weights, directed_edges, edge_gradient_weights), nx) #.reshape(batch_size_,  -1)
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
            for x, y, nx, node_mask, nodes, node_weights, directed_edges, edge_gradient_weights in test_loader:
                x, y, nx, node_mask, nodes, node_weights, directed_edges, edge_gradient_weights = x.to(device), y.to(device), nx.to(device),node_mask.to(device), nodes.to(device), node_weights.to(device), directed_edges.to(device), edge_gradient_weights.to(device)

                batch_size_ = x.shape[0]
                out = model(x, (node_mask, nodes, node_weights, directed_edges, edge_gradient_weights), nx) #.reshape(batch_size_,  -1)

                if normalization_y:
                    out = y_normalizer.decode(out)
                    y = y_normalizer.decode(y)
                out=out*node_mask #mask the padded value with 0,(1 for node, 0 for padding)
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
              ' 1/sp_L: ',[round(float(x[0]), 3) for x in model.sp_L.cpu().tolist()],
                  flush=True)
        if (ep %100 == 99) or (ep == epochs -1):    
            torch.save(model.state_dict(), save_model_name + ".pth")

            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'current_epoch': ep,  # optional: to track training progress
            }, "checkpoint.pth")

            
    
    
    return train_rel_l2_losses, test_rel_l2_losses, test_l2_losses

