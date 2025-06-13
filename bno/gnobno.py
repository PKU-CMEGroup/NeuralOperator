################################################################################################
# Still under construction !!!
################################################################################################

import torch
import torch.nn as nn
from pcno.pcno import _get_act, SpectralConv, compute_Fourier_bases


class LinearMLP(nn.Module):
    def __init__(self, in_channels, out_channels, fc_dim=128, n_layers=2, act="gelu", p=0.0):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fc_channels = fc_dim

        self.n_layers = n_layers
        hidden_channels = [in_channels] + [fc_dim] * (n_layers - 1) + [out_channels]

        self.act = _get_act(act)
        self.dropout = (
            nn.ModuleList([nn.Dropout(p) for _ in range(self.n_layers)])
            if p > 0.0 else None
        )

        self.fcs = nn.ModuleList()
        for j in range(self.n_layers):
            self.fcs.append(
                nn.Linear(hidden_channels[j], hidden_channels[j + 1]))

    def forward(self, x):
        for i, fc in enumerate(self.fcs):
            x = fc(x)
            if i < self.n_layers - 1:
                x = self.act(x)
            if self.dropout is not None:
                x = self.dropout[i](x)
        return x


class GNOBlock(nn.Module):
    """
    A simplified GNO block from `neuralop` package that computes:
        g(y) = int_{N(x)} k(x, y) * u(x) dx

    - This naive implementation supports only a linear kernel k(x, y). 
    - k(x,y) is really simple now since it is assumed to be diagonal !!!
    - The input `u` has the same number of channels as the output.
    - The neighbors of the boundary are computed as preprocessed data, rather than being dynamically computed within the current layer.
    """

    def __init__(self, channels, ndims):
        super().__init__()

        self.channels = channels
        self.ndims = ndims

        self.kernel = LinearMLP(2 * self.ndims, self.channels)  # should be self.channels**2

    def forward(self, nodes_x, nodes_y, x, directed_edges, weights):
        """
            Inputs:
                nodes_x         : float(batch_size, max_nnodes, phy_dim)
                nodes_y         : float(batch_size, max_nnodes, phy_dim)
                x               : float(batch_size, max_nnodes, channels)
                weights         : float(batch_size, max_nedges)
                directed_edges  : int64(batch_size, max_nedges, 2)

            Returns:
                out             : float(batch_size, max_nnodes, channels)
        """
        x = x.permute(0, 2, 1)

        batch_size, max_nnodes, _ = nodes_y.shape
        src, tgt = directed_edges[..., 0], directed_edges[..., 1]

        coord_pairs = torch.cat([nodes_x[torch.arange(batch_size).unsqueeze(1), src],
                                 nodes_y[torch.arange(batch_size).unsqueeze(1), tgt]], dim=-1)
        ker = self.kernel(coord_pairs)

        ker_weighted = torch.einsum('be,bec->bec', weights, ker)
        features = ker_weighted * x[torch.arange(batch_size).unsqueeze(1), src]

        out = torch.zeros(batch_size, max_nnodes, self.channels,
                          dtype=features.dtype, device=features.device)
        out.scatter_add_(dim=1, src=features,
                         index=tgt.unsqueeze(-1).expand_as(features))

        out = out.permute(0, 2, 1)

        return out


class ExtGNOBNO(nn.Module):
    def __init__(self,
                 ndims,
                 modes,
                 nmeasures,
                 layers,
                 radius,
                 fc_dim=128,
                 in_dim_x=2,
                 in_dim_y=3,
                 out_dim=1,
                 inv_L_scale_hyper = ['independently', 0.5, 2.0],
                 act="gelu"
                 ):
        super(ExtGNOBNO, self).__init__()
        """ 
        A naive implementation of ExtBNO.
        The local operator is replaced by a global one. 
        """

        self.model_type = "ExtBNO"

        self.modes = modes
        self.nmeasures = nmeasures

        self.layers = layers
        self.radius = radius
        self.fc_dim = fc_dim

        self.ndims = ndims
        self.in_dim_x = in_dim_x
        self.in_dim_y = in_dim_y

        self.fc0_x = nn.Linear(in_dim_x, layers[0], fc_dim)
        self.fc0_y = nn.Linear(in_dim_y, layers[0], fc_dim)

        self.sp_convs_ext = nn.ModuleList(
            [
                SpectralConv(in_size, out_size, modes)
                for in_size, out_size in zip(
                    self.layers, self.layers[1:]
                )
            ]
        )
        self.sp_convs_y = nn.ModuleList(
            [
                SpectralConv(in_size, out_size, modes)
                for in_size, out_size in zip(
                    self.layers, self.layers[1:]
                )
            ]
        )

        self.train_inv_L_scale, self.inv_L_scale_min, self.inv_L_scale_max  = inv_L_scale_hyper[0], inv_L_scale_hyper[1], inv_L_scale_hyper[2]
        # latent variable for inv_L_scale = inv_L_scale_min + (inv_L_scale_max - inv_L_scale_min) / (1 + exp(inv_L_scale_latent)) 
        self.inv_L_scale_latent = nn.Parameter(torch.full((ndims, nmeasures), np.log((self.inv_L_scale_max - 1)/(1.0 - self.inv_L_scale_min)), device='cuda'), requires_grad = bool(self.train_inv_L_scale))


        self.gnos_re = nn.ModuleList(
            [
                GNOBlock(size, ndims)
                for size in self.layers[1:]
            ]
        )

        self.ws_x = nn.ModuleList(
            [
                nn.Conv1d(in_size, out_size, 1)
                for in_size, out_size in zip(self.layers, self.layers[1:])
            ]
        )
        self.ws_y = nn.ModuleList(
            [
                nn.Conv1d(in_size, out_size, 1)
                for in_size, out_size in zip(self.layers, self.layers[1:])
            ]
        )

        if fc_dim > 0:
            self.fc1 = nn.Linear(layers[-1], fc_dim)
            self.fc2 = nn.Linear(fc_dim, out_dim)
        else:
            self.fc2 = nn.Linear(layers[-1], out_dim)

        self.act = _get_act(act)

        self.normal_params = []  # group of params which will be trained normally
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
                

    def forward(self, x, y, aux):
        """
        Forward evaluation. 
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. len(layers)-1 layers of the boundary neural layers 
                        u' = W1(u) + E(v)
                        v' = (W2 + K)(u)
           linear functions  W1, W2: parameterized by self.ws_x and self.ws_y; 
           integral operator K: parameterized by self.sp_convs_y with nmeasures different integrals
           extension operator E: 
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

            Parameters: 
                x   : Tensor float[batch_size, max_nnomdes_x, in_dim] 
                      Input data in the entire domain
                y   : Tensor float[batch_size, max_nnomdes_x, in_dim] 
                      Input data on the boundary
                aux : list of Tensor, containing
                      node_mask_x : Tensor int[batch_size, max_nnomdes_x, 1]  
                                1: node; otherwise 0

                      nodes_x : Tensor float[batch_size, max_nnomdes_x, ndim]  
                      nodes_y : Tensor float[batch_size, max_nnomdes_y, ndim] 
                            nodal coordinate; padding with 0

                      node_weights_y  : Tensor float[batch_size, max_nnomdes_y, nmeasures_y]  
                                    rho(x)dx used for nmeasures integrations; padding with 0
                                    Currently, we assume nmeasures_x = nmeasures_y = nmeasures = 1 for simplicity.  
                                    The case where x or y has various measures is not yet supported and will be addressed in future updates.

                      neighbor_edges  : Tensor int64[batch_size, max_nedges, 2] 

                      edge_weights  : Tensor float[batch_size, max_nedges]                                    

            Returns:
                G(x) : Tensor float[batch_size, max_nnomdes, out_dim] 
                       Output data

        """
        length = len(self.ws_x)

        mask_x, nodes_x, nodes_y, node_weights_y, neighbor_edges, edge_weights = aux

        bases_c_x, bases_s_x, bases_0_x = compute_Fourier_bases(nodes_x, self.modes * (self.inv_L_scale_min + (self.inv_L_scale_max - self.inv_L_scale_min)/(1.0 + torch.exp(self.inv_L_scale_latent))))
        bases_c_y, bases_s_y, bases_0_y = compute_Fourier_bases(nodes_y, self.modes * (self.inv_L_scale_min + (self.inv_L_scale_max - self.inv_L_scale_min)/(1.0 + torch.exp(self.inv_L_scale_latent))))

        wbases_c_y = torch.einsum("bxkw,bxw->bxkw", bases_c_y, node_weights_y)
        wbases_s_y = torch.einsum("bxkw,bxw->bxkw", bases_s_y, node_weights_y)
        wbases_0_y = torch.einsum("bxkw,bxw->bxkw", bases_0_y, node_weights_y)

        x = self.fc0_x(x)
        y = self.fc0_y(y)

        x = x.permute(0, 2, 1)
        y = y.permute(0, 2, 1)

        for i, (speconv_ext, speconv_y, gno_re, w_x, w_y) in enumerate(zip(self.sp_convs_ext, self.sp_convs_y, self.gnos_re, self.ws_x, self.ws_y)):

            x1 = speconv_ext(y, bases_c_x, bases_s_x, bases_0_x, wbases_c_y, wbases_s_y, wbases_0_y)  # extension operator: boundary to the entire domain
            x2 = w_x(x)
            x = x1 + x2
            if self.act is not None and i != length - 1:
                x = self.act(x)

                # a simple evolution of boundary
                # y1 = speconv_y(y, bases_c_y, bases_s_y, bases_0_y, wbases_c_y, wbases_s_y, wbases_0_y)
                y2 = gno_re(nodes_x, nodes_y, x, neighbor_edges, edge_weights)  # reduction operator: neighbors of the boundary to the boundary
                y3 = w_y(y)

                y = y2 + y3
                y = self.act(y)

        x = x.permute(0, 2, 1)

        if self.fc_dim > 0:
            x = self.fc1(x)
            if self.act is not None:
                x = self.act(x)

        x = self.fc2(x)

        return x
