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

        self.type = "GalerkinConv"
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes = modes
        self.bases = bases
        self.wbases = wbases

        self.scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            self.scale
            * torch.rand(in_channels, out_channels, self.modes, dtype=torch.float)
        )

    def forward(self, x):
        bases, wbases = self.bases, self.wbases
        # Compute coeffcients

        x_hat = torch.einsum("bcx,xk->bck", x, wbases)

        # Multiply relevant Fourier modes
        x_hat = compl_mul1d(x_hat, self.weights)
        # x_hat = self.fc(x_hat)

        # Return to physical space
        x = torch.real(torch.einsum("bck,xk->bcx", x_hat, bases))

        return x


class GkNN(nn.Module):
    def __init__(self, bases_list, **config):
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
            if layer.type in ["SimpleAttention", "DoubleAttention"]:
                # when using Attention layer
                # ignore activation function and skipping connection because they are included in the layer
                x = layer(x)

            else:
                x1 = layer(x)
                x2 = w(x)
                res = x1 + x2

                # activation function
                if self.act is not None and i != length - 1:
                    res = self.act(res)

                # skipping connection
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
        elif layer_type == "SimpleAttention":
            num_heads = self.num_heads[index]
            attention_type = self.attention_types[index]
            return SimpleAttention(in_channels, out_channels, num_heads, attention_type)
        elif layer_type == "DoubleAttention":
            if self.test_types[index] == "encoder":
                wbases_V = self.wbases_pca_in
                bases_Q = self.bases_pca_out
            elif self.test_types[index] == "decoder":
                wbases_V = self.wbases_pca_out
                bases_Q = self.bases_pca_out
            return DoubleAttention(in_channels, out_channels, wbases_V, bases_Q)
        else:
            raise ValueError("Layer Type Undefined.")


class DoubleAttention(nn.Module):
    def __init__(self, in_channels, out_channels, wbases_V, bases_Q):
        super(DoubleAttention, self).__init__()

        """
        Galerkin Type Transformer with One Head
        fix the hidden space of query and value to the spcaces spanned by selected bases    
        attn = Q*(K^T*V) without normlization
        """

        self.type = "DoubleAttention"
        self.dim_k = in_channels

        self.wbases_V = wbases_V[:, : self.dim_k]
        self.bases_Q = bases_Q[:, : self.dim_k]

        # find key in the space spanned by learnable hidden bases
        self.linear_K = nn.Linear(self.dim_k, self.dim_k)
        self._reset_parameters_K()

        # ffn layer
        self.fc = nn.Sequential(
            nn.Linear(self.dim_k, self.dim_k),
            nn.GELU(),
            nn.Linear(self.dim_k, out_channels),
        )
        self.act = _get_act("gelu")

    def forward(self, x):
        """
        Input Shape: (batchsize,in_channels,n)
        Output Shape : (batchsize,out_channels,n)
        """
        x = x.permute(0, 2, 1)

        # try 2: no norm, using weighted bases
        query = self.bases_Q
        key = self.linear_K(x)
        value_weighted = self.wbases_V
        x = self.act(self._attention1(query, key, value_weighted)) + x

        x = x + self.fc(x)
        x = x.permute(0, 2, 1)

        return x

    def _reset_parameters_K(self, xavier_init=1e-2, diagonal_weight=1e-02):
        for param in self.linear_K.parameters():
            if param.ndim > 1:
                xavier_uniform_(param, gain=xavier_init)
                param.data += diagonal_weight * torch.diag(
                    torch.ones(param.size(-1), dtype=torch.float)
                )
            else:
                constant_(param, 0)

    @staticmethod
    def _attention1(query, key, value_weighted):
        """
        galerkin type attention
        O(nd^2) computational complexity
        """
        score = torch.einsum("bnk,nv->bkv", key, value_weighted)
        score = F.dropout(score)

        attn = torch.einsum("nk,bkv->bnv", query, score)
        return attn

    @staticmethod
    def _attention2(query, key, value):
        """
        fourier type attention
        O(n^2d) computational complexity would lead to insufficient memory.
        """
        d_k = query.size(-1)
        score = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        len = score.size(-1)
        p_attn = score / len
        p_attn = F.dropout(p_attn)
        return torch.matmul(p_attn, value)
