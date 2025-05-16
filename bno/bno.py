################################################################################################
# Still under construction !!!
###############################################################################################

import torch
import torch.nn as nn
from pcno.pcno import _get_act, SpectralConv, compute_Fourier_bases


class BNO(nn.Module):
    def __init__(self,
                 ndims,
                 modes,
                 nmeasures,
                 layers,
                 fc_dim=128,
                 in_dim_x=2,
                 in_dim_y=3,
                 out_dim=1,
                 train_sp_L=False,
                 act="gelu"
                 ):
        super(BNO, self).__init__()
        """ 
        A naive implementation of BNO.
        The local operator is replaced by a global one. 
        """

        self.model_type = "BNO"

        self.modes = modes
        self.nmeasures = nmeasures

        self.layers = layers
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
        self.sp_convs_x = nn.ModuleList(
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
        self.sp_L = nn.Parameter(torch.ones(ndims, nmeasures), requires_grad=bool(train_sp_L))
        self.train_sp_L = train_sp_L

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
        self.sp_L_params = []  # group of params which may be trained specially
        for _, param in self.named_parameters():
            if param is not self.sp_L:
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

    def forward(self, x, y, aux):
        """
        Forward evaluation. 
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. len(layers)-1 layers of the boundary neural layers 
                        u' = (W1 + K1)(u) + E(v)
                        v' = (W2 + K2)(u)
           linear functions  W1, W2: parameterized by self.ws_x and self.ws_y; 
           integral operator K1, K2: parameterized by self.sp_convs_x and self.sp_convs_y with nmeasures different integrals
           extension operator E: 
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

            Parameters: 
                x : Tensor float[batch_size, max_nnomdes_x, in_dim] 
                    Input data in the entire domain
                y : Tensor float[batch_size, max_nnomdes_x, in_dim] 
                    Input data on the boundary
                aux : list of Tensor, containing
                    node_mask_x : Tensor int[batch_size, max_nnomdes_x, 1]  
                    node_mask_y : Tensor int[batch_size, max_nnomdes_y, 1] 
                                1: node; otherwise 0

                    nodes_x : Tensor float[batch_size, max_nnomdes_x, ndim]  
                    nodes_y : Tensor float[batch_size, max_nnomdes_y, ndim] 
                            nodal coordinate; padding with 0

                    node_weights_x  : Tensor float[batch_size, max_nnomdes_x, nmeasures_x]  
                    node_weights_y  : Tensor float[batch_size, max_nnomdes_y, nmeasures_y]  
                                    rho(x)dx used for nmeasures integrations; padding with 0
                                    Currently, we assume nmeasures_x = nmeasures_y = nmeasures = 1 for simplicity.  
                                    The case where x or y has various measures is not yet supported and will be addressed in future updates.                                   

            Returns:
                G(x) : Tensor float[batch_size, max_nnomdes, out_dim] 
                       Output data

        """
        length = len(self.ws_x)
        node_mask_x, nodes_x, nodes_y, node_weights_x, node_weights_y = aux

        bases_c_x, bases_s_x, bases_0_x = compute_Fourier_bases(nodes_x, self.modes * self.sp_L)
        bases_c_y, bases_s_y, bases_0_y = compute_Fourier_bases(nodes_y, self.modes * self.sp_L)

        wbases_c_x = torch.einsum("bxkw,bxw->bxkw", bases_c_x, node_weights_x)
        wbases_s_x = torch.einsum("bxkw,bxw->bxkw", bases_s_x, node_weights_x)
        wbases_0_x = torch.einsum("bxkw,bxw->bxkw", bases_0_x, node_weights_x)

        wbases_c_y = torch.einsum("bxkw,bxw->bxkw", bases_c_y, node_weights_y)
        wbases_s_y = torch.einsum("bxkw,bxw->bxkw", bases_s_y, node_weights_y)
        wbases_0_y = torch.einsum("bxkw,bxw->bxkw", bases_0_y, node_weights_y)

        x = self.fc0_x(x)
        y = self.fc0_y(y)

        x = x.permute(0, 2, 1)
        y = y.permute(0, 2, 1)

        for i, (speconv_ext, speconv_x, speconv_y, w_x, w_y) in enumerate(zip(self.sp_convs_ext, self.sp_convs_x, self.sp_convs_y, self.ws_x, self.ws_y)):

            x1 = speconv_ext(y, bases_c_x, bases_s_x, bases_0_x, wbases_c_y, wbases_s_y, wbases_0_y)  # extend operator: boundary to the entire domain
            x2 = speconv_x(x, bases_c_x, bases_s_x, bases_0_x, wbases_c_x, wbases_s_x, wbases_0_x)  # global opertor
            x3 = w_x(x)
            x = x1 + x2 + x3
            if self.act is not None and i != length - 1:
                x = self.act(x)

                # a simple evolution of boundary
                y1 = speconv_y(y, bases_c_y, bases_s_y, bases_0_y, wbases_c_y, wbases_s_y, wbases_0_y)  # this should be a local operator: neighbors of boundary to boundary
                y2 = w_y(y)
                y = y1 + y2
                y = self.act(y)

        x = x.permute(0, 2, 1)

        if self.fc_dim > 0:
            x = self.fc1(x)
            if self.act is not None:
                x = self.act(x)

        x = self.fc2(x)

        return x


class ExtBNO(nn.Module):
    def __init__(self,
                 ndims,
                 modes,
                 nmeasures,
                 layers,
                 fc_dim=128,
                 in_dim_x=2,
                 in_dim_y=3,
                 out_dim=1,
                 train_sp_L=False,
                 act="gelu"
                 ):
        super(ExtBNO, self).__init__()
        """ 
        A naive implementation of ExtBNO.
        The local operator is replaced by a global one. 
        """

        self.model_type = "ExtBNO"

        self.modes = modes
        self.nmeasures = nmeasures

        self.layers = layers
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
        self.sp_L = nn.Parameter(torch.ones(ndims, nmeasures), requires_grad=bool(train_sp_L))
        self.train_sp_L = train_sp_L

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
        self.sp_L_params = []  # group of params which may be trained specially
        for _, param in self.named_parameters():
            if param is not self.sp_L:
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
                x : Tensor float[batch_size, max_nnomdes_x, in_dim] 
                    Input data in the entire domain
                y : Tensor float[batch_size, max_nnomdes_x, in_dim] 
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

            Returns:
                G(x) : Tensor float[batch_size, max_nnomdes, out_dim] 
                       Output data

        """
        length = len(self.ws_x)
        node_mask_x, nodes_x, nodes_y, node_weights_y = aux

        bases_c_x, bases_s_x, bases_0_x = compute_Fourier_bases(nodes_x, self.modes * self.sp_L)
        bases_c_y, bases_s_y, bases_0_y = compute_Fourier_bases(nodes_y, self.modes * self.sp_L)

        wbases_c_y = torch.einsum("bxkw,bxw->bxkw", bases_c_y, node_weights_y)
        wbases_s_y = torch.einsum("bxkw,bxw->bxkw", bases_s_y, node_weights_y)
        wbases_0_y = torch.einsum("bxkw,bxw->bxkw", bases_0_y, node_weights_y)

        x = self.fc0_x(x)
        y = self.fc0_y(y)

        x = x.permute(0, 2, 1)
        y = y.permute(0, 2, 1)

        for i, (speconv_ext, speconv_y, w_x, w_y) in enumerate(zip(self.sp_convs_ext, self.sp_convs_y, self.ws_x, self.ws_y)):

            x1 = speconv_ext(y, bases_c_x, bases_s_x, bases_0_x, wbases_c_y, wbases_s_y, wbases_0_y)  # extend operator: boundary to the entire domain
            x2 = w_x(x)
            x = x1 + x2
            if self.act is not None and i != length - 1:
                x = self.act(x)

                # a simple evolution of boundary
                y1 = speconv_y(y, bases_c_y, bases_s_y, bases_0_y, wbases_c_y, wbases_s_y, wbases_0_y)  # this should be a local operator: neighbors of boundary to boundary
                y2 = w_y(y)
                y = y1 + y2
                y = self.act(y)

        x = x.permute(0, 2, 1)

        if self.fc_dim > 0:
            x = self.fc1(x)
            if self.act is not None:
                x = self.act(x)

        x = self.fc2(x)

        return x
