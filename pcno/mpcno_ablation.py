import torch
import torch.nn as nn
from timeit import default_timer
from utility.losses import LpLoss
from utility.normalizer import UnitGaussianNormalizer

from .mpcno import (
    _get_act,
    scaled_sigmoid,
    scaled_logit,
    compute_Fourier_modes_helper,
    compute_Fourier_modes,
    compute_Fourier_bases,
    SpectralConv,
    GradientLayer,
    GeoEmbedding,
    compute_gradient,
    CombinedOptimizer,
    Combinedscheduler_OneCycleLR,
    MPCNO_train_multidist
)


class StructuredNN(nn.Module):
    def __init__(self, in_dim, mid_layers, out_dim, act=None):
        """
        Network structure: input -> mid_layers[0] -> ... -> mid_layers[-1] -> output

        Args:
            in_dim: Input dimension
            mid_layers: List of middle layer dimensions
            out_dim: Output dimension
            act: Activation function (callable)
        """
        super(StructuredNN, self).__init__()

        self.act = act
        self.layers = nn.ModuleList()

        prev_dim = in_dim
        for hidden_dim in mid_layers:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim
        self.layers.append(nn.Linear(prev_dim, out_dim))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != len(self.layers) - 1 and self.act is not None:
                x = self.act(x)
        return x


class MPCNO(nn.Module):
    def __init__(
        self,
        ndims,
        modes,
        nmeasures,
        layers,
        layer_selection = {'grad': True, 'geo': True, 'geointegral': True},
        fc_dim=128,
        proj_layers=None,
        in_dim=3,
        out_dim=1,
        inv_L_scale_hyper = ['independently', 0.5, 2.0],
        scaling_mode = 'inv',
        act="gelu",
        geo_act='softsign',
        proj_act="gelu",
    ):
        super(MPCNO, self).__init__()

        """
        The overall network. 
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. len(layers)-1 layers of the point cloud neural layers u' = (W + K + D)(u).
           linear functions  W: parameterized by self.ws; 
           integral operator K: parameterized by self.sp_convs with nmeasures different integrals
           differential operator D: parameterized by self.grad_embs
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
                layer_selection : dict
                    A configuration dictionary to toggle specific operational branches:
                        'grad' (bool): If True, includes the differential operator branch.
                        'geo' (bool): If True, includes the short-range geometric embedding branch.
                        'geointegral' (bool): If True, enables geometry-aware weights in the spectral 
                                                convolution to handle non-trivial domains.
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
                scaling_mode : string (default "inv")
                    Strategy to scale the combined output of multiple branches.
                    Options:
                        "inv": Scales by 1/n, maintaining the empirical mean of the branches.
                        "sqrt_inv": Scales by 1/sqrt(n), preserving the variance of the combined signal.
                        "none": No scaling (scale factor = 1.0).
                act : string (default gelu)
                    The activation function
                geo_act : string (default "softsign")
                    The activation function used for geometry-related features, applied to 
                    both the gradient operator and the short-range geometric embedding

            
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
        self.grad_layers = nn.ModuleList(
            [
                GradientLayer(ndims, in_size, out_size, geo_act=geo_act)
                for in_size, out_size in zip(self.layers, self.layers[1:])
            ]
        ) if layer_selection['grad'] else [None]*len(layers[1:])

        # Short-range geo layer, geo includes nx, d^(1)(nx)
        self.geo_embs = nn.ModuleList(
            [
                GeoEmbedding(ndims*(ndims+1), in_size, out_size, geo_act=geo_act)
                for in_size, out_size in zip(self.layers, self.layers[1:])
            ]
        ) if layer_selection['geo'] else [None]*len(layers[1:])
        
        self.proj_act = _get_act(proj_act)
        if proj_layers is None:
            proj_layers = [fc_dim] if fc_dim > 0 else []
        elif isinstance(proj_layers, int):
            proj_layers = [proj_layers] if proj_layers > 0 else []
        else:
            proj_layers = [int(width) for width in proj_layers if int(width) > 0]
        self.proj = StructuredNN(
            in_dim=layers[-1],
            mid_layers=proj_layers,
            out_dim=out_dim,
            act=self.proj_act,
        )


        self.train_inv_L_scale, self.inv_L_scale_min, self.inv_L_scale_max  = inv_L_scale_hyper[0], inv_L_scale_hyper[1], inv_L_scale_hyper[2]
        # latent variable for inv_L_scale = inv_L_scale_min + (inv_L_scale_max - inv_L_scale_min) * sigmoid(inv_L_scale_latent)
        self.inv_L_scale_latent = nn.Parameter(torch.full((ndims, nmeasures), scaled_logit(torch.tensor(1.0), self.inv_L_scale_min, self.inv_L_scale_max)), requires_grad = bool(self.train_inv_L_scale))
        
        
        self.scaling_mode = scaling_mode
        num_branches = 2 
        if layer_selection['grad']: num_branches += 1
        if layer_selection['geo']: num_branches += 1
        if scaling_mode == 'inv':
            self.scale_factor = 1.0 / num_branches
        elif scaling_mode == 'sqrt_inv':
            self.scale_factor = 1.0 / (num_branches ** 0.5)
        elif scaling_mode == 'none':
            self.scale_factor = 1.0
        else:
            raise ValueError(f"{scaling_mode} is not supported")

        self.act = _get_act(act)

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
        2. len(layers)-1 layers of the point cloud neural layers u' = u + act((W + K + D + G)(u)).
           linear functions  W: parameterized by self.ws; 
           integral operator K: parameterized by self.sp_convs with nmeasures different integrals
           differential operator D: parameterized by self.grad_embs
           short-range geo layer G: parameterized by self.geo_embs
           
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
                    outward_normals :   Tensor float[batch_size, ndim, max_nnodes]
                                        Surface outward unit normals at each node.

            
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

        for i, (speconv, spw, spconvnw, spconvadjnw, w, grad_layer, geo_emb) in enumerate(zip(self.sp_convs, self.sp_ws, self.sp_convs_nws, self.sp_convs_adj_nws, self.ws, self.grad_layers, self.geo_embs)):
            
            if self.layer_selection['geointegral']:
                x1 = speconv( spconvnw(  torch.cat([x] + [x * outward_normals[:, i:i+1, :] for i in range(outward_normals.size(1))], dim=1)  ), bases_c, bases_s, bases_0, wbases_c, wbases_s, wbases_0)
                x1 = spw(x1) + spconvadjnw(torch.cat([x1 * outward_normals[:, i:i+1, :] for i in range(outward_normals.size(1))], dim=1))
            else:
                x1 = speconv(x, bases_c, bases_s, bases_0, wbases_c, wbases_s, wbases_0)
                x1 = spw(x1)
                
            x2 = w(x)


            if self.layer_selection['grad']:
                x_grad = grad_layer(x, directed_edges, edge_gradient_weights)
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

        x = self.proj(x)
        return x 

    def forward_ablation(self, x, aux):
        """
        Ablation forward used at test time:
        remove grad/geo branches and keep only K + W update path.
        """
        length = len(self.ws)
        node_mask, nodes, node_weights, directed_edges, edge_gradient_weights, outward_normals = aux

        bases_c, bases_s, bases_0 = compute_Fourier_bases(
            nodes,
            self.modes * (scaled_sigmoid(self.inv_L_scale_latent, self.inv_L_scale_min, self.inv_L_scale_max)),
        )
        wbases_c = torch.einsum("bxkw,bxw->bxkw", bases_c, node_weights)
        wbases_s = torch.einsum("bxkw,bxw->bxkw", bases_s, node_weights)
        wbases_0 = torch.einsum("bxkw,bxw->bxkw", bases_0, node_weights)

        if self.scaling_mode == "inv":
            ablation_scale = 0.5
        elif self.scaling_mode == "sqrt_inv":
            ablation_scale = 1.0 / (2 ** 0.5)
        else:
            ablation_scale = 1.0

        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        for i, (speconv, spw, spconvnw, spconvadjnw, w, _, _) in enumerate(
            zip(self.sp_convs, self.sp_ws, self.sp_convs_nws, self.sp_convs_adj_nws, self.ws, self.grad_layers, self.geo_embs)
        ):
            if self.layer_selection["geointegral"]:
                x1 = speconv(
                    spconvnw(torch.cat([x] + [x * outward_normals[:, j : j + 1, :] for j in range(outward_normals.size(1))], dim=1)),
                    bases_c,
                    bases_s,
                    bases_0,
                    wbases_c,
                    wbases_s,
                    wbases_0,
                )
                x1 = spw(x1) + spconvadjnw(
                    torch.cat([x1 * outward_normals[:, j : j + 1, :] for j in range(outward_normals.size(1))], dim=1)
                )
            else:
                x1 = speconv(x, bases_c, bases_s, bases_0, wbases_c, wbases_s, wbases_0)
                x1 = spw(x1)

            x2 = w(x)
            if self.act is not None and i != length - 1:
                x = x + self.act(ablation_scale * (x1 + x2))
            else:
                x = ablation_scale * (x1 + x2)

        x = x.permute(0, 2, 1)
        x = self.proj(x)
        return x
