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
from .mpcno import MPCNO

    

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
    elif act == "softsign":
        func = F.softsign
    elif act == 'soft_identity':
        func = lambda x: x/(1+0.01*x**2)
    elif act == 'piecewise':
        def piecewise_smooth(x):
            return torch.where(
                x > 10, 
                10 + 5 * torch.tanh((x - 10) / 5), 
                torch.where(x < -10, -10 + 5 * torch.tanh((x + 10) / 5), x)
            )
        func = piecewise_smooth
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

class HyperSpectralConv(nn.Module):
    """
    Beta-conditioned SpectralConv via a **Tucker-style hyper-network**.

    The weight offset is a rank-r Tucker-1 decomposition along ALL four axes:

        ΔW_c[i,o,k,w] = Σ_r  u_c^r(β) · A_c[i,r] · B_c[o,r] · V_c[k·w, r]

    In einsum notation:
        ΔW_c = einsum('br, ir, or, kr -> iokr', u_c(β), A_c, B_c, V_c)

    where:
        u_*(β)  ∈ R^{B × r}   — β-dependent mixing coefficients (from hyper_net)
        A_*     ∈ R^{ic × r}  — static in-channel factor
        B_*     ∈ R^{oc × r}  — static out-channel factor
        V_c, V_s ∈ R^{km × r} — static mode factor
        V_0     ∈ R^{1  × r}

    Hyper-MLP output dim = **3r** (one scalar per rank per weight type),
    independent of ic, oc, and km.  This is the minimal possible β-dependent
    parameterisation for a rank-r perturbation.

    For a typical layer (ic=oc=64, km=577, r=8):
        direct output:  ~4.7 M   →   Tucker hyper output:  3×8 = 24 scalars
        total extra params per layer:  3×8 + (64+64+577)×8×3 ≈ 16 k

    All static factors are orthonormally initialised.
    The hyper-MLP output layer is zero-initialised → ΔW=0 at init.

    Args:
        in_channels  : int
        out_channels : int
        modes        : float Tensor [nmodes, ndims, nmeasures]
        hyper_net    : nn.Module   β [B, beta_dim] → [B, 3*r]
        rank         : int   Tucker rank r (default 8)
    """
    def __init__(self, in_channels, out_channels, modes, net_c, net_s, net_0, rank=8):
        super(HyperSpectralConv, self).__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        nmodes, ndims, nmeasures = modes.shape
        self.nmodes    = nmodes
        self.nmeasures = nmeasures
        self.rank      = rank
        self.scale     = 1 / (in_channels * out_channels)

        ic, oc, km, w = in_channels, out_channels, nmodes, nmeasures

        # ---- Base (β-independent) weights — identical to SpectralConv -------
        self.weights_c = nn.Parameter(self.scale * torch.rand(ic, oc, km, w))
        self.weights_s = nn.Parameter(self.scale * torch.rand(ic, oc, km, w))
        self.weights_0 = nn.Parameter(self.scale * torch.rand(ic, oc,  1, w))

        # ---- Static Tucker factors (orthonormally initialised) ---------------
        def _ortho(rows, cols):
            Q, _ = torch.linalg.qr(torch.randn(max(rows, cols), min(rows, cols)))
            return Q[:rows, :cols]

        # in-channel factors  A ∈ R^{ic × r}   (shared across c/s/0)
        self.A_c = nn.Parameter(_ortho(ic, rank))
        self.A_s = nn.Parameter(_ortho(ic, rank))
        self.A_0 = nn.Parameter(_ortho(ic, rank))

        # out-channel factors  B ∈ R^{oc × r}
        self.B_c = nn.Parameter(_ortho(oc, rank))
        self.B_s = nn.Parameter(_ortho(oc, rank))
        self.B_0 = nn.Parameter(_ortho(oc, rank))

        # mode factors  V_c, V_s ∈ R^{km × r};  V_0 ∈ R^{1 × r}
        self.V_c = nn.Parameter(_ortho(km, rank))
        self.V_s = nn.Parameter(_ortho(km, rank))
        self.V_0 = nn.Parameter(_ortho(max(1, w), rank)[:1, :])   # [1, r]

        # ---- Hyper-network (passed in, owned here) ---------------------------
        # output dim = rank,rank,rank   (u_c, u_s, u_0 stacked)
        self.net_c = net_c
        self.net_s = net_s
        self.net_0 = net_0

    def forward(self, x, beta, bases_c, bases_s, bases_0, wbases_c, wbases_s, wbases_0):
        """
        Parameters:
            x      : float [B, in_channels, N]
            beta   : float [B, beta_dim]
            bases_*: same as SpectralConv.forward
        Returns:
            float [B, out_channels, N]
        """
        B  = x.shape[0]
        ic, oc, km, w, r = (self.in_channels, self.out_channels,
                             self.nmodes, self.nmeasures, self.rank)

        # ---- hyper-net: β → mixing vectors  u ∈ R^{B × r}  -----------------
        u_c = self.net_c(beta)                     # [B, r]
        u_s = self.net_s(beta)
        u_0 = self.net_0(beta)

        # ---- Tucker offsets ------------------------------------------------
        # ΔW_c[b,i,o,k] = Σ_r  u_c[b,r] · A_c[i,r] · B_c[o,r] · V_c[k,r]
        # einsum contracts over r → output [B, ic, oc, km], then unsqueeze w

        dW_c = torch.einsum('br,ir,or,kr->biok', u_c, self.A_c, self.B_c, self.V_c
                            ).unsqueeze(-1).expand(B, ic, oc, km, w)
        dW_s = torch.einsum('br,ir,or,kr->biok', u_s, self.A_s, self.B_s, self.V_s
                            ).unsqueeze(-1).expand(B, ic, oc, km, w)
        dW_0 = torch.einsum('br,ir,or,kr->biok', u_0, self.A_0, self.B_0, self.V_0
                            ).unsqueeze(-1)                              # [B, ic, oc, 1, w=1]

        # ---- Effective weights = base + Tucker offset ------------------------
        weights_c = self.weights_c.unsqueeze(0) + dW_c   # [B, ic, oc, km, w]
        weights_s = self.weights_s.unsqueeze(0) + dW_s
        weights_0 = self.weights_0.unsqueeze(0) + dW_0

        # ---- Fourier transform (batch-wise weights) --------------------------
        x_c_hat =  torch.einsum("bix,bxkw->bikw", x, wbases_c)
        x_s_hat = -torch.einsum("bix,bxkw->bikw", x, wbases_s)
        x_0_hat =  torch.einsum("bix,bxkw->bikw", x, wbases_0)

        f_c_hat = (torch.einsum("bikw,biokw->bokw", x_c_hat, weights_c)
                 - torch.einsum("bikw,biokw->bokw", x_s_hat, weights_s))
        f_s_hat = (torch.einsum("bikw,biokw->bokw", x_s_hat, weights_c)
                 + torch.einsum("bikw,biokw->bokw", x_c_hat, weights_s))
        f_0_hat =  torch.einsum("bikw,biokw->bokw", x_0_hat, weights_0)

        return (  torch.einsum("bokw,bxkw->box", f_0_hat, bases_0)
                + 2 * torch.einsum("bokw,bxkw->box", f_c_hat, bases_c)
                - 2 * torch.einsum("bokw,bxkw->box", f_s_hat, bases_s))


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

################################################################
# MPCNO_Beta: beta-conditioned MPCNO via HyperSpectralConv
################################################################

class MPCNO_Beta(MPCNO):
    """
    MPCNO fully conditioned on β via two complementary mechanisms:

    1. **Tucker HyperNetwork** on SpectralConv (long-range branch):
       ΔW(β) = einsum('br,ir,or,kr->iokr', u(β), A, B, V)
       MLP output dim = 3r (independent of ic, oc, km).

    All β-dependent parameters are zero-initialised so the model starts
    identical to a plain MPCNO.

    Extra constructor args (beyond MPCNO's):
        beta_dim     : int   dimension of β input (1 for scalar β)
        hyper_hidden : int   hidden width of per-layer Tucker hyper-MLPs (default 64)
        rank         : int   Tucker rank r for SpectralConv offsets (default 8)
    """

    def __init__(
        self,
        ndims,
        modes,
        nmeasures,
        layers,
        beta_dim,
        hyper_hidden=64,
        rank=8,
        layer_selection={'grad': True, 'geo': True, 'geointegral': True},
        fc_dim=128,
        in_dim=3,
        out_dim=1,
        inv_L_scale_hyper=['independently', 0.5, 2.0],
        scaling_mode='inv',
        act='gelu',
    ):
        # Build the full parent MPCNO (creates plain SpectralConvs)
        super(MPCNO_Beta, self).__init__(
            ndims=ndims, modes=modes, nmeasures=nmeasures, layers=layers,
            layer_selection=layer_selection, fc_dim=fc_dim,
            in_dim=in_dim, out_dim=out_dim,
            inv_L_scale_hyper=inv_L_scale_hyper,
            scaling_mode=scaling_mode, act=act)

        self.beta_dim = beta_dim
        self.rank     = rank
        nmodes_val    = modes.shape[0]

        # Replace every SpectralConv with a HyperSpectralConv (Tucker low-rank).
        # Each layer gets its own small hyper-MLP.
        # MLP output dim = 3 * rank   (u_c, u_s, u_0 — just r scalars each)
        # The ic, oc, km structure lives entirely in the static A, B, V factors.
        new_sp_convs = nn.ModuleList()

        def get_net(beta_dim, hyper_hidden, rank):
            net = nn.Sequential(
                nn.Linear(beta_dim, hyper_hidden),
                nn.GELU(),
                nn.Linear(hyper_hidden, hyper_hidden),
                nn.GELU(),
                nn.Linear(hyper_hidden, rank),
            )
            # Zero-init the output layer → all offsets start at 0 → model == MPCNO at init
            nn.init.zeros_(net[-1].weight)
            nn.init.zeros_(net[-1].bias) 
            return net
            
        for in_size, out_size in zip(layers, layers[1:]):

            net_c = get_net(beta_dim, hyper_hidden, rank)
            net_s = get_net(beta_dim, hyper_hidden, rank)
            net_0 = get_net(beta_dim, hyper_hidden, rank)

            new_sp_convs.append(
                HyperSpectralConv(in_size, out_size, modes, net_c, net_s, net_0, rank=rank))

        self.sp_convs = new_sp_convs  # overwrite the plain SpectralConvs

        # Rebuild param groups because sp_convs changed
        self.normal_params      = []
        self.inv_L_scale_params = []
        for _, param in self.named_parameters():
            if param is not self.inv_L_scale_latent:
                self.normal_params.append(param)
            else:
                if self.train_inv_L_scale == 'together':
                    self.normal_params.append(param)
                elif self.train_inv_L_scale == 'independently':
                    self.inv_L_scale_params.append(param)

    def forward(self, x, beta, aux):
        """
        Parameters:
            x    : float Tensor [B, N, in_dim]
            beta : float Tensor [B, beta_dim]
            aux  : same tuple as MPCNO.forward
        Returns:
            float Tensor [B, N, out_dim]
        """
        length = len(self.ws)

        node_mask, nodes, node_weights, directed_edges, edge_gradient_weights, outward_normals = aux

        bases_c, bases_s, bases_0 = compute_Fourier_bases(
            nodes,
            self.modes * scaled_sigmoid(self.inv_L_scale_latent,
                                        self.inv_L_scale_min,
                                        self.inv_L_scale_max))
        wbases_c = torch.einsum("bxkw,bxw->bxkw", bases_c, node_weights)
        wbases_s = torch.einsum("bxkw,bxw->bxkw", bases_s, node_weights)
        wbases_0 = torch.einsum("bxkw,bxw->bxkw", bases_0, node_weights)

        x = self.fc0(x)
        x = x.permute(0, 2, 1)   # [B, channels, N]

        for i, (speconv, spw, spconvnw, spconvadjnw, w) in enumerate(
                zip(self.sp_convs, self.sp_ws,
                    self.sp_convs_nws, self.sp_convs_adj_nws,
                    self.ws)):

            if self.layer_selection['geointegral']:
                x_in = spconvnw(torch.cat(
                    [x] + [x * outward_normals[:, j:j+1, :]
                           for j in range(outward_normals.size(1))], dim=1))
                x1 = speconv(x_in, beta,
                             bases_c, bases_s, bases_0,
                             wbases_c, wbases_s, wbases_0)
                x1 = spw(x1) + spconvadjnw(torch.cat(
                    [x1 * outward_normals[:, j:j+1, :]
                     for j in range(outward_normals.size(1))], dim=1))
            else:
                x1 = speconv(x, beta,
                             bases_c, bases_s, bases_0,
                             wbases_c, wbases_s, wbases_0)
                x1 = spw(x1)

            x2 = w(x)

            if self.act is not None and i != length - 1:
                x = x + self.act(self.scale_factor * (x1 + x2))
            else:
                x = self.scale_factor * (x1 + x2)

        x = x.permute(0, 2, 1)   # [B, N, channels]

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
def MPCNO_train_multidist(x_train, aux_train, y_train, x_test_list, aux_test_list, y_test_list,  config, model, label_test_list = None, save_model_name="./MPCNO_model", checkpoint_path=None):
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
    print("In MPCNO_train, ndims = ", ndims)
    
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
def MPCNO_train(x_train, aux_train, y_train, x_test, aux_test, y_test, config, model, save_model_name="./MPCNO_model", checkpoint_path=None):
    n_train, n_test = x_train.shape[0], x_test.shape[0]
    train_rel_l2_losses = []
    test_rel_l2_losses = []
    test_l2_losses = []
    normalization_x, normalization_y = config["train"]["normalization_x"], config["train"]["normalization_y"]
    normalization_dim_x, normalization_dim_y = config["train"]["normalization_dim_x"], config["train"]["normalization_dim_y"]
    non_normalized_dim_x, non_normalized_dim_y = config["train"]["non_normalized_dim_x"], config["train"]["non_normalized_dim_y"]
    
    ndims = model.ndims # n_train, size, n_channel
    print("In MPCNO_train, ndims = ", ndims)
    
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



        
def MPCNO_train_parallel(x_train, aux_train, y_train, x_test, aux_test, y_test, config, model, rank=0, local_rank = 0, world_size=1, save_model_name="./MPCNO_parallel_model", checkpoint_path=None):
    """
    MPCNO_train function for parallel GPU training
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

def MPCNO_train_multidist_beta(
        x_train, aux_train, y_train, beta_train,
        x_test_list, aux_test_list, y_test_list, beta_test_list,
        config, model,
        label_test_list=None,
        save_model_name="./MPCNO_model",
        checkpoint_path=None):
    """
    Training loop for MPCNO_Beta.  Identical to MPCNO_train_multidist except:
      - beta_train / beta_test_list are added to every DataLoader batch.
      - model.forward(x, beta, aux) is called instead of model.forward(x, aux).

    Args:
        x_train        : float Tensor [n_train, N, in_dim]
        aux_train      : tuple (node_mask, nodes, node_weights,
                                directed_edges, edge_gradient_weights, geo)
        y_train        : float Tensor [n_train, N, out_dim]
        beta_train     : float Tensor [n_train, beta_dim]
        x_test_list    : list of float Tensor
        aux_test_list  : list of aux tuples
        y_test_list    : list of float Tensor
        beta_test_list : list of float Tensor  [n_test_i, beta_dim]
        config         : same config dict as MPCNO_train_multidist
        model          : MPCNO_Beta instance
    """
    assert (len(x_test_list) == len(y_test_list)
            == len(aux_test_list) == len(beta_test_list)), \
        "x/y/aux/beta test lists must have equal length"

    n_distributions = len(x_test_list)
    n_train = x_train.shape[0]

    train_rel_l2_losses = []
    test_rel_l2_losses  = []
    test_l2_losses      = []

    normalization_x      = config["train"]["normalization_x"]
    normalization_y      = config["train"]["normalization_y"]
    normalization_dim_x  = config["train"]["normalization_dim_x"]
    normalization_dim_y  = config["train"]["normalization_dim_y"]
    non_normalized_dim_x = config["train"]["non_normalized_dim_x"]
    non_normalized_dim_y = config["train"]["non_normalized_dim_y"]

    ndims  = model.ndims
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("In MPCNO_train_multidist_beta, ndims =", ndims)

    # ---- Normalisation (beta is NOT normalised — it's already in [0.5, 1]) --
    if normalization_x:
        x_normalizer = UnitGaussianNormalizer(
            x_train,
            non_normalized_dim=non_normalized_dim_x,
            normalization_dim=normalization_dim_x)
        x_train = x_normalizer.encode(x_train)
        for i in range(n_distributions):
            x_test_list[i] = x_normalizer.encode(x_test_list[i])
        x_normalizer.to(device)

    if normalization_y:
        y_normalizer = UnitGaussianNormalizer(
            y_train,
            non_normalized_dim=non_normalized_dim_y,
            normalization_dim=normalization_dim_y)
        y_train = y_normalizer.encode(y_train)
        for i in range(n_distributions):
            y_test_list[i] = y_normalizer.encode(y_test_list[i])
        y_normalizer.to(device)

    # ---- DataLoaders --------------------------------------------------------
    node_mask_tr, nodes_tr, nw_tr, de_tr, egw_tr, geo_tr = aux_train
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            x_train, y_train, beta_train,
            node_mask_tr, nodes_tr, nw_tr, de_tr, egw_tr, geo_tr),
        batch_size=config['train']['batch_size'],
        shuffle=True)

    test_loaders = []
    for i in range(n_distributions):
        nm_t, nd_t, nw_t, de_t, egw_t, geo_t = aux_test_list[i]
        sub_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                x_test_list[i], y_test_list[i], beta_test_list[i],
                nm_t, nd_t, nw_t, de_t, egw_t, geo_t),
            batch_size=config['train']['batch_size'],
            shuffle=False)
        name = (label_test_list[i] if label_test_list
                else f"Distribution_{i}")
        test_loaders.append((name, sub_loader))

    # ---- Optimiser / Scheduler ----------------------------------------------
    myloss = LpLoss(d=1, p=2, size_average=False)

    optimizer = CombinedOptimizer(
        model.normal_params, model.inv_L_scale_params,
        betas=(0.9, 0.999),
        lr=config["train"]["base_lr"],
        lr_ratio=config["train"]["lr_ratio"],
        weight_decay=config["train"]["weight_decay"])

    scheduler = Combinedscheduler_OneCycleLR(
        optimizer,
        max_lr=config['train']['base_lr'],
        lr_ratio=config["train"]["lr_ratio"],
        div_factor=2, final_div_factor=100, pct_start=0.2,
        steps_per_epoch=len(train_loader),
        epochs=config['train']['epochs'])

    current_epoch = 0
    epochs = config['train']['epochs']

    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        current_epoch = checkpoint['current_epoch'] + 1
        print("Restarting from epoch:", current_epoch)

    # ---- Training loop ------------------------------------------------------
    for ep in range(current_epoch, epochs):
        t1 = default_timer()
        train_rel_l2 = 0.0

        model.train()
        for (x, y, beta,
             node_mask, nodes, node_weights,
             directed_edges, edge_gradient_weights, geo) in train_loader:

            x, y, beta = x.to(device), y.to(device), beta.to(device)
            node_mask             = node_mask.to(device)
            nodes                 = nodes.to(device)
            node_weights          = node_weights.to(device)
            directed_edges        = directed_edges.to(device)
            edge_gradient_weights = edge_gradient_weights.to(device)
            geo                   = geo.to(device)

            batch_size_ = x.shape[0]
            optimizer.zero_grad()

            out = model(x, beta,
                        (node_mask, nodes, node_weights,
                         directed_edges, edge_gradient_weights, geo))

            if normalization_y:
                out = y_normalizer.decode(out)
                y   = y_normalizer.decode(y)

            out  = out * node_mask
            loss = myloss(out.view(batch_size_, -1), y.view(batch_size_, -1))
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_rel_l2 += loss.item()

        # ---- Evaluation -----------------------------------------------------
        test_rel_l2_dict = {}
        test_l2_dict     = {}

        model.eval()
        with torch.no_grad():
            for name, loader in test_loaders:
                t_l2     = 0.0
                t_rel_l2 = 0.0

                for (x, y, beta,
                     node_mask, nodes, node_weights,
                     directed_edges, edge_gradient_weights, geo) in loader:

                    x, y, beta = x.to(device), y.to(device), beta.to(device)
                    node_mask             = node_mask.to(device)
                    nodes                 = nodes.to(device)
                    node_weights          = node_weights.to(device)
                    directed_edges        = directed_edges.to(device)
                    edge_gradient_weights = edge_gradient_weights.to(device)
                    geo                   = geo.to(device)

                    batch_size_ = x.shape[0]
                    out = model(x, beta,
                                (node_mask, nodes, node_weights,
                                 directed_edges, edge_gradient_weights, geo))

                    if normalization_y:
                        out = y_normalizer.decode(out)
                        y   = y_normalizer.decode(y)

                    out      = out * node_mask
                    t_rel_l2 += myloss(out.view(batch_size_, -1),
                                       y.view(batch_size_, -1)).item()
                    t_l2     += myloss.abs(out.view(batch_size_, -1),
                                           y.view(batch_size_, -1)).item()

                test_l2_dict[name]     = t_l2     / len(loader.dataset)
                test_rel_l2_dict[name] = t_rel_l2 / len(loader.dataset)

        train_rel_l2 /= n_train
        train_rel_l2_losses.append(train_rel_l2)
        test_rel_l2_losses.append(test_rel_l2_dict)
        test_l2_losses.append(test_l2_dict)

        t2 = default_timer()
        print("Epoch:", ep,
              " Time:", round(t2 - t1, 3),
              " Rel.Train L2:", train_rel_l2,
              " Rel.Test L2:", test_rel_l2_dict,
              " Test L2:", test_l2_dict,
              " inv_L_scale:", [round(float(v[0]), 3) for v in
                                scaled_sigmoid(model.inv_L_scale_latent,
                                               model.inv_L_scale_min,
                                               model.inv_L_scale_max).cpu().tolist()],
              flush=True)

        if (ep % 100 == 99) or (ep == epochs - 1):
            if save_model_name:
                torch.save(model.state_dict(), save_model_name + ".pth")
                torch.save({
                    'model_state_dict':      model.state_dict(),
                    'optimizer_state_dict':  optimizer.state_dict(),
                    'scheduler_state_dict':  scheduler.state_dict(),
                    'current_epoch':         ep,
                }, "checkpoint_beta.pth")

    return train_rel_l2_losses, test_rel_l2_losses, test_l2_losses