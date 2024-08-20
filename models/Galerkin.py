import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import defaultdict
import sys

sys.path.append("../")
from .basics import (
    compl_mul1d,
    compl_mul1d_matrix,
    SpectralConv1d,
    SpectralConv2d_shape,
    SimpleAttention,
)
from .utils import _get_act, add_padding, remove_padding


# class HiddenBases(nn.Module):
#     def __init__(self, modes, bases, wbases):
#         super(HiddenBases, self).__init__()

#         self.modes = modes
#         self.bases = bases
#         self.wbases = wbases

#         self.fc_bases = nn.Linear(modes, modes)
#         self.fc_wbases = nn.Linear(modes, modes)

#     def forward(self):
#         bases = self.fc_bases(self.bases)
#         wbases = self.fc_wbases(self.wbases)
# return bases,wbases


class GalerkinConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        modes,
        bases,
        wbases,
        diag_width,
        if_hidden_channels,
    ):
        super(GalerkinConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.bases = bases[:, : self.modes]
        self.wbases = wbases[:, : self.modes]
        self.if_hidden_channels = if_hidden_channels
        self.diag_width = diag_width

        if if_hidden_channels == True:
            self.weight_hidden_bases = nn.Parameter(
                (self.modes**2) * torch.eye(self.modes, dtype=torch.float)
            )
            self.weight_hidden_wbases = nn.Parameter(
                (self.modes**2) * torch.eye(self.modes, dtype=torch.float)
            )

        self.scale = 1 / (in_channels * out_channels)
        if self.diag_width > 0:
            self.weights_list = nn.ParameterList(
                [
                    nn.Parameter(
                        self.scale
                        * torch.rand(
                            in_channels, out_channels, self.modes, dtype=torch.float
                        )
                    )
                ]
            )
            for i in range(self.diag_width - 1):
                self.weights_list.append(
                    nn.Parameter(
                        self.scale
                        * torch.rand(
                            in_channels,
                            out_channels,
                            self.modes - i - 1,
                            dtype=torch.float,
                        )
                    )
                )
                self.weights_list.append(
                    nn.Parameter(
                        self.scale
                        * torch.rand(
                            in_channels,
                            out_channels,
                            self.modes - i - 1,
                            dtype=torch.float,
                        )
                    )
                )
        elif self.diag_width == -1:
            self.weights_matrix = nn.Parameter(
                self.scale
                * torch.rand(
                    in_channels, out_channels, self.modes, self.modes, dtype=torch.float
                )
            )
        else:
            raise ValueError("diag width error!")

    def forward(self, x):
        bases = self.bases
        wbases = self.wbases
        if self.if_hidden_channels == True:
            bases = torch.einsum("xk,kl->xl", bases, self.weight_hidden_bases)
            wbases = torch.einsum("xk,kl->xl", wbases, self.weight_hidden_wbases)

        x_co = torch.einsum("bcx,xk->bck", x, wbases)

        if self.diag_width >= 0:
            x_hat = compl_mul1d(x_co, self.weights_list[0])
            for i in range(self.diag_width - 1):
                x1 = compl_mul1d(x_hat[..., i + 1 :], self.weights_list[2 * i + 1])
                x2 = compl_mul1d(x_hat[..., : -(i + 1)], self.weights_list[2 * i + 2])
                x_hat1 = x_hat.clone()
                x_hat1[..., : -(i + 1)] += x1
                x_hat1[..., (i + 1) :] += x2
                x_hat = x_hat1
        elif self.diag_width == -1:
            x_hat = compl_mul1d_matrix(x_co, self.weights_matrix)

        if self.if_hidden_channels == False:
            x = torch.real(torch.einsum("bck,xk->bcx", x_hat, bases))

        return x


class GkNN(nn.Module):
    def __init__(self, bases_list, **config):
        super(GkNN, self).__init__()

        self.bases_fourier = bases_list[0]
        self.wbases_fourier = bases_list[1]
        self.bases_pca = bases_list[2]
        self.wbases_pca = bases_list[3]

        self.config = defaultdict(lambda: None, **config)
        self.config = dict(self.config)
        all_attr = list(self.config.keys())
        for key in all_attr:
            setattr(self, key, self.config[key])

        self.fc0 = nn.Linear(self.in_dim + 1, self.layers_dim[0])

        self.sp_layers = nn.ModuleList(
            [
                self._choose_layer(
                    index,
                    in_size,
                    out_size,
                    layer_type,
                    diag_width,
                    if_hidden_bases,
                )
                for index, (
                    in_size,
                    out_size,
                    layer_type,
                    diag_width,
                    if_hidden_bases,
                ) in enumerate(
                    zip(
                        self.layers_dim,
                        self.layers_dim[1:],
                        self.layer_types,
                        self.diag_width_list,
                        self.if_hidden_bases_list,
                    )
                )
            ]
        )
        self.ws = nn.ModuleList(
            [
                nn.Conv1d(in_size, out_size, 1)
                for in_size, out_size in zip(self.layers_dim, self.layers_dim[1:])
            ]
        )
        self.length = len(self.ws)

        self.dropout_layer_list = nn.ModuleList()
        for _ in range(self.length):
            self.dropout_layer_list.append(nn.Dropout(p=0.5))

        if self.fc_dim > 0:
            self.fc1 = nn.Linear(self.layers_dim[-1], self.fc_dim)
            if self.if_dropout_decoder == True:
                self.dropout_fc1 = nn.Dropout(0.05)
            self.fc2 = nn.Linear(self.fc_dim, self.out_dim)
        else:
            self.fc2 = nn.Linear(self.layers_dim[-1], self.out_dim)

        self.act = _get_act(self.act)

    def forward(self, x):
        f = torch.ones_like(x[..., 0]).unsqueeze(-1)
        x = torch.cat((x, f), dim=-1)

        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        for i, (layer, w) in enumerate(zip(self.sp_layers, self.ws)):
            x1 = layer(x)
            x2 = w(x)
            res = x1 + x2
            if self.act is not None and i != self.length - 1:
                res = self.act(res)
            if self.if_dropout_layer_list[i] == True:
                res = self.dropout_layer_list[i](x)
            if self.residual[i] == True:
                x = x + res
            else:
                x = res

        x = x.permute(0, 2, 1)

        if self.fc_dim > 0:
            x = self.fc1(x)
            if self.act is not None:
                x = self.act(x)
            if self.if_dropout_decoder == True:
                x = self.dropout_fc1(x)

        x = self.fc2(x)

        return x

    def _choose_layer(
        self,
        index,
        in_channels,
        out_channels,
        layer_type,
        diag_width,
        if_hidden_bases,
    ):
        if layer_type == "GalerkinConv_fourier":
            num_modes = self.GkNN_modes[index]
            bases = self.bases_fourier
            wbases = self.wbases_fourier
            return GalerkinConv(
                in_channels,
                out_channels,
                num_modes,
                bases,
                wbases,
                diag_width,
                if_hidden_bases,
            )
        elif layer_type == "GalerkinConv_pca":
            num_modes = self.GkNN_modes[index]
            bases = self.bases_pca
            wbases = self.wbases_pca
            return GalerkinConv(
                in_channels,
                out_channels,
                num_modes,
                bases,
                wbases,
                diag_width,
                if_hidden_bases,
            )
        elif layer_type == "FourierConv1d":
            num_modes = self.FNO_modes[index]
            return SpectralConv1d(in_channels, out_channels, num_modes)
        elif layer_type == "FourierConv2d":
            num_modes1 = self.FNO_modes[index]
            num_modes2 = self.FNO_modes[index]
            return SpectralConv2d_shape(
                in_channels, out_channels, num_modes1, num_modes2
            )
        elif layer_type == "Attention":
            num_heads = self.num_heads[index]
            attention_type = self.attention_types[index]
            return SimpleAttention(in_channels, out_channels, num_heads, attention_type)
        else:
            raise ValueError("Layer Type Undefined.")
