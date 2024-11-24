import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_

from collections import defaultdict
import sys

sys.path.append("../")
from models.basics import (
    compl_mul1d,
    SpectralConv1d,
    SpectralConv2d_shape,
    SimpleAttention,
)
from models.utils import _get_act


class GalerkinConv(nn.Module):
    def __init__(self, in_channels, out_channels, modes, bases, wbases):
        super(GalerkinConv, self).__init__()

        """
        1D Spectral layer. It avoids FFT, but utilizes low rank approximation. 
        
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes = modes
        self.bases = bases
        self.wbases = wbases

        self.scale = 1 / (in_channels * out_channels)
        self.fc = nn.Sequential(
            nn.Conv1d(in_channels, 128, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(128, out_channels, kernel_size=1),
            nn.Linear(modes, modes),
        )

    def forward(self, x):
        bases, wbases = self.bases, self.wbases
        # Compute coeffcients

        x_hat = torch.einsum("bcx,xk->bck", x, wbases)

        # Multiply relevant Fourier modes
        # x_hat = compl_mul1d(x_hat, self.weights)
        x_hat = self.fc(x_hat)

        # Return to physical space
        x = torch.real(torch.einsum("bck,xk->bcx", x_hat, bases))

        return x


class GkNN(nn.Module):
    def __init__(self, bases_list, weights, **config):
        super(GkNN, self).__init__()

        """
        The overall network. It contains several layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """
        self.bases_fourier = bases_list[0]
        self.wbases_fourier = bases_list[1]
        self.bases_pca_in = bases_list[2]
        self.wbases_pca_in = bases_list[3]
        self.bases_pca_out = bases_list[3]
        self.wbases_pca_out = bases_list[3]
        self.weights = weights

        self.config = defaultdict(lambda: None, **config)
        self.config = dict(self.config)
        all_attr = list(self.config.keys())
        for key in all_attr:
            setattr(self, key, self.config[key])

        self.fc0 = nn.Sequential(
            nn.Linear(self.in_dim, self.fc_dim),
            nn.GELU(),
            nn.Linear(self.fc_dim, self.layers_dim[0]),
        )
        # input channel is 2: (a(x), x)

        self.sp_layers = nn.ModuleList(
            [
                self._choose_layer(index, in_size, out_size, layer_type)
                for index, (in_size, out_size, layer_type) in enumerate(
                    zip(self.layers_dim, self.layers_dim[1:], self.layer_types)
                )
            ]
        )
        #######
        self.ws = nn.ModuleList(
            [
                nn.Conv1d(in_size, out_size, 1)
                for in_size, out_size in zip(self.layers_dim, self.layers_dim[1:])
            ]
        )

        # if fc_dim = 0, we do not have nonlinear layer
        if self.fc_dim > 0:
            self.fc1 = nn.Linear(self.layers_dim[-1], self.fc_dim)
            self.fc2 = nn.Linear(self.fc_dim, self.out_dim)
        else:
            self.fc2 = nn.Linear(self.layers_dim[-1], self.out_dim)

        self.act = _get_act(self.act)

    def forward(self, x):
        """
        Input shape (of x):     (batch, nx_in,  channels_in)
        Output shape:           (batch, nx_out, channels_out)

        The input resolution is determined by x.shape[-1]
        The output resolution is determined by self.s_outputspace
        """

        length = len(self.ws)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        for i, (layer, w) in enumerate(zip(self.sp_layers, self.ws)):
            x1 = layer(x)
            x2 = w(x)
            res = x1 + x2
            if self.act is not None and i != length - 1:
                res = self.act(res)
            if self.residual[i] == True:
                x = x + res
            else:
                x = res

        x = x.permute(0, 2, 1)

        fc_dim = self.fc_dim

        if fc_dim > 0:
            x = self.fc1(x)
            if self.act is not None:
                x = self.act(x)

        x = self.fc2(x)

        return x

    def _choose_layer(self, index, in_channels, out_channels, layer_type):
        if layer_type == "GalerkinConv_fourier":
            num_modes = self.GkNN_modes[index]
            bases = self.bases_fourier
            wbases = self.wbases_fourier
            return GalerkinConv(in_channels, out_channels, num_modes, bases, wbases)
        elif layer_type == "GalerkinConv_pca":
            num_modes = self.GkNN_modes[index]
            bases = self.bases_pca_out
            wbases = self.wbases_pca_out
            return GalerkinConv(in_channels, out_channels, num_modes, bases, wbases)
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
        elif layer_type == "DoubleAttention":
            if self.test_types[index] == "encoder":
                bases_V = self.bases_pca_in
                bases_Q = self.bases_fourier
            elif self.test_types[index] == "decoder":
                bases_V = self.bases_fourier
                bases_Q = self.bases_pca_out
            return DoubleGalerkinAttention(
                in_channels,
                out_channels,
                bases_V[:, :128],
                bases_Q[:, :128],
                self.weights,
            )
        else:
            raise ValueError("Layer Type Undefined.")


class DoubleGalerkinAttention(nn.Module):
    def __init__(self, in_channels, out_channels, bases_V, bases_Q, weights):
        super(DoubleGalerkinAttention, self).__init__()

        """
        One Head Galerkin Type Transformer 
        fix the hidden space of query and value to the spcaces spanned by selected bases    
        attn = Q*(layernorm(K)^T*layernorm(V))    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.bases_V = bases_V
        self.bases_Q = bases_Q
        self.weights = weights

        self.modes_V = bases_V.size(-1)
        self.modes_Q = bases_Q.size(-1)
        self.dim_K = self.modes_Q

        # key=xW^k (batchsize, n, in_channels) -> (batchsize, n, dim_k)
        self.linear_K = nn.Linear(self.in_channels, self.dim_K)
        # self.xavier_init = 1e-2
        # self.diagonal_weight = 1e-2
        # self._reset_parameters()

        # layernorm
        self.norm_K = nn.LayerNorm(self.dim_K, eps=1e-05)
        self.norm_V = nn.LayerNorm(self.modes_Q, eps=1e-05)

        # output (batchsize, n, modes_v) -> (batchsize, n, out_channels)
        self.fc = nn.Linear(self.modes_V, out_channels)

    def _reset_parameters(self):
        for param in self.linear_K.parameters():
            if param.ndim > 1:
                xavier_uniform_(param, gain=self.xavier_init)
                param.data += self.diagonal_weight * torch.diag(
                    torch.ones(param.size(-1), dtype=torch.float)
                )
            else:
                constant_(param, 0)

    def _attention1(self, query, key, value):
        co = torch.einsum("bnk,nv->bkv", key, value)  # / math.sqrt(self.dim_K)
        # co = F.dropout(co)
        z = torch.einsum("nk,bkv->bnv", query, co)
        return z

    @staticmethod
    def _attention2(query, key, value):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        seq_len = scores.size(-1)
        p_attn = scores / seq_len
        p_attn = F.dropout(p_attn)
        return torch.matmul(p_attn, value)

    def forward(self, x):
        """
        Input Shape: (batchsize,in_channels,n)
        Output Shape : (batchsize,out_channels,n)
        """
        x = x.permute(0, 2, 1)

        query = self.bases_Q
        key = self.norm_K(self.linear_K(x))
        value = self.norm_V(self.bases_V)  # try 1: norm without weights
        value_weighted = torch.einsum("nv,n->nv", value, self.weights)

        x = self._attention1(query, key, value_weighted)
        x = self.fc(x)

        x = x.permute(0, 2, 1)
        return x
