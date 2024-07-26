import math
import numpy as np
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_, xavier_normal_


@torch.jit.script
def compl_mul1d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
    res = torch.einsum("bix,iox->box", a, b)
    return res


@torch.jit.script
def compl_mul2d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
    res = torch.einsum("bixy,ioxy->boxy", a, b)
    return res


@torch.jit.script
def compl_mul3d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    res = torch.einsum("bixyz,ioxyz->boxyz", a, b)
    return res


@torch.jit.script
def compl_mul4d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
    res = torch.einsum("bixyzt,ioxyzt->boxyzt", a, b)
    return res


################################################################
# 1d fourier layer
################################################################


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes1 = modes1

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale
            * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat)
        )

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[2])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize,
            self.in_channels,
            x.size(-1) // 2 + 1,
            device=x.device,
            dtype=torch.cfloat,
        )

        out_ft[:, :, : self.modes1] = compl_mul1d(
            x_ft[:, :, : self.modes1], self.weights1
        )

        # Return to physical space
        x = torch.fft.irfftn(out_ft, s=[x.size(-1)], dim=[2])
        return x


################################################################
# 2d fourier layer
################################################################


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat
            )
        )
        self.weights2 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat
            )
        )

    def forward(self, x, gridy=None):
        batchsize = x.shape[0]
        size1 = x.shape[-2]
        size2 = x.shape[-1]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[2, 3])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-2),
            x.size(-1) // 2 + 1,
            device=x.device,
            dtype=torch.cfloat,
        )
        out_ft[:, :, : self.modes1, : self.modes2] = compl_mul2d(
            x_ft[:, :, : self.modes1, : self.modes2], self.weights1
        )
        out_ft[:, :, -self.modes1 :, : self.modes2] = compl_mul2d(
            x_ft[:, :, -self.modes1 :, : self.modes2], self.weights2
        )

        # Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-2), x.size(-1)), dim=[2, 3])
        return x


class SpectralConv2d_shape(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d_shape, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat
            )
        )
        self.weights2 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat
            )
        )

    def forward(self, x):
        batchsize = x.shape[0]
        num1 = math.floor(np.sqrt(x.shape[-1]))
        x = x.reshape((batchsize, self.in_channels, num1, num1))
        size1 = x.shape[-2]
        size2 = x.shape[-1]
        # Compute Fourier coeffcients up to factor of e^(- something constant)

        x_ft = torch.fft.rfftn(x, dim=[2, 3])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-2),
            x.size(-1) // 2 + 1,
            device=x.device,
            dtype=torch.cfloat,
        )
        out_ft[:, :, : self.modes1, : self.modes2] = compl_mul2d(
            x_ft[:, :, : self.modes1, : self.modes2], self.weights1
        )
        out_ft[:, :, -self.modes1 :, : self.modes2] = compl_mul2d(
            x_ft[:, :, -self.modes1 :, : self.modes2], self.weights2
        )

        # Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-2), x.size(-1)), dim=[2, 3])
        return x.reshape((batchsize, self.out_channels, -1))


class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = (
            modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        )
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                dtype=torch.cfloat,
            )
        )
        self.weights2 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                dtype=torch.cfloat,
            )
        )
        self.weights3 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                dtype=torch.cfloat,
            )
        )
        self.weights4 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                dtype=torch.cfloat,
            )
        )

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[2, 3, 4])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(2),
            x.size(3),
            x.size(4) // 2 + 1,
            device=x.device,
            dtype=torch.cfloat,
        )

        out_ft[:, :, : self.modes1, : self.modes2, : self.modes3] = compl_mul3d(
            x_ft[:, :, : self.modes1, : self.modes2, : self.modes3], self.weights1
        )
        out_ft[:, :, -self.modes1 :, : self.modes2, : self.modes3] = compl_mul3d(
            x_ft[:, :, -self.modes1 :, : self.modes2, : self.modes3], self.weights2
        )
        out_ft[:, :, : self.modes1, -self.modes2 :, : self.modes3] = compl_mul3d(
            x_ft[:, :, : self.modes1, -self.modes2 :, : self.modes3], self.weights3
        )
        out_ft[:, :, -self.modes1 :, -self.modes2 :, : self.modes3] = compl_mul3d(
            x_ft[:, :, -self.modes1 :, -self.modes2 :, : self.modes3], self.weights4
        )

        # Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(2), x.size(3), x.size(4)), dim=[2, 3, 4])
        return x


class SpectralConv4d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3, modes4):
        super(SpectralConv4d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = (
            modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        )
        self.modes2 = modes2
        self.modes3 = modes3
        self.modes4 = modes4

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                self.modes4,
                dtype=torch.cfloat,
            )
        )
        self.weights2 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                self.modes4,
                dtype=torch.cfloat,
            )
        )
        self.weights3 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                self.modes4,
                dtype=torch.cfloat,
            )
        )
        self.weights4 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                self.modes4,
                dtype=torch.cfloat,
            )
        )
        self.weights5 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                self.modes4,
                dtype=torch.cfloat,
            )
        )
        self.weights6 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                self.modes4,
                dtype=torch.cfloat,
            )
        )
        self.weights7 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                self.modes4,
                dtype=torch.cfloat,
            )
        )
        self.weights8 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                self.modes4,
                dtype=torch.cfloat,
            )
        )

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[2, 3, 4, 5])
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(2),
            x.size(3),
            x.size(4),
            x.size(5) // 2 + 1,
            device=x.device,
            dtype=torch.cfloat,
        )
        out_ft[:, :, : self.modes1, : self.modes2, : self.modes3, : self.modes4] = (
            compl_mul4d(
                x_ft[:, :, : self.modes1, : self.modes2, : self.modes3, : self.modes4],
                self.weights1,
            )
        )
        out_ft[:, :, -self.modes1 :, : self.modes2, : self.modes3, : self.modes4] = (
            compl_mul4d(
                x_ft[:, :, -self.modes1 :, : self.modes2, : self.modes3, : self.modes4],
                self.weights2,
            )
        )
        out_ft[:, :, : self.modes1, -self.modes2 :, : self.modes3, : self.modes4] = (
            compl_mul4d(
                x_ft[:, :, : self.modes1, -self.modes2 :, : self.modes3, : self.modes4],
                self.weights3,
            )
        )
        out_ft[:, :, : self.modes1, : self.modes2, -self.modes3 :, : self.modes4] = (
            compl_mul4d(
                x_ft[:, :, : self.modes1, : self.modes2, -self.modes3 :, : self.modes4],
                self.weights4,
            )
        )
        out_ft[:, :, -self.modes1 :, -self.modes2 :, : self.modes3, : self.modes4] = (
            compl_mul4d(
                x_ft[
                    :, :, -self.modes1 :, -self.modes2 :, : self.modes3, : self.modes4
                ],
                self.weights5,
            )
        )
        out_ft[:, :, -self.modes1 :, : self.modes2, -self.modes3 :, : self.modes4] = (
            compl_mul4d(
                x_ft[
                    :, :, -self.modes1 :, : self.modes2, -self.modes3 :, : self.modes4
                ],
                self.weights6,
            )
        )
        out_ft[:, :, : self.modes1, -self.modes2 :, -self.modes3 :, : self.modes4] = (
            compl_mul4d(
                x_ft[
                    :, :, : self.modes1, -self.modes2 :, -self.modes3 :, : self.modes4
                ],
                self.weights7,
            )
        )
        out_ft[:, :, -self.modes1 :, -self.modes2 :, -self.modes3 :, : self.modes4] = (
            compl_mul4d(
                x_ft[
                    :, :, -self.modes1 :, -self.modes2 :, -self.modes3 :, : self.modes4
                ],
                self.weights8,
            )
        )

        # Return to physical space
        x = torch.fft.irfftn(
            out_ft, s=(x.size(2), x.size(3), x.size(4), x.size(5)), dim=[2, 3, 4, 5]
        )
        return x


class FourierBlock(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3, act="tanh"):
        super(FourierBlock, self).__init__()
        self.in_channel = in_channels
        self.out_channel = out_channels
        self.speconv = SpectralConv3d(in_channels, out_channels, modes1, modes2, modes3)
        self.linear = nn.Conv1d(in_channels, out_channels, 1)
        if act == "tanh":
            self.act = torch.tanh_
        elif act == "gelu":
            self.act = nn.GELU
        elif act == "none":
            self.act = None
        else:
            raise ValueError(f"{act} is not supported")

    def forward(self, x):
        """
        input x: (batchsize, channel width, x_grid, y_grid, t_grid)
        """
        x1 = self.speconv(x)
        x2 = self.linear(x.view(x.shape[0], self.in_channel, -1))
        out = x1 + x2.view(
            x.shape[0], self.out_channel, x.shape[2], x.shape[3], x.shape[4]
        )
        if self.act is not None:
            out = self.act(out)
        return out


################################################################
# attention layer
################################################################
class SimpleAttention(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_heads,
        attention_type,
        eps=1e-5,
    ):
        super(SimpleAttention, self).__init__()

        assert in_channels % num_heads == 0
        self.in_channels = in_channels
        self.d_k = in_channels // num_heads
        self.num_heads = num_heads
        self.attention_type = attention_type

        self.linears = nn.ModuleList(
            [copy.deepcopy(nn.Linear(in_channels, in_channels)) for _ in range(3)]
        )
        self.xavier_init = 1e-2
        self.diagonal_weight = 1e-2
        self._reset_parameters()

        self.fc = nn.Linear(in_channels, out_channels)

        self._get_norm(eps=eps)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        batchsize = x.size(0)

        query, key, value = [
            layer(x).view(batchsize, -1, self.num_heads, self.d_k).transpose(1, 2)
            for layer, x in zip(self.linears, (x, x, x))
        ]

        if self.attention_type == "galerkin":
            key = torch.stack(
                [
                    norm(x)
                    for norm, x in zip(
                        self.norm_K, (key[:, i, ...] for i in range(self.num_heads))
                    )
                ],
                dim=1,
            )
            value = torch.stack(
                [
                    norm(x)
                    for norm, x in zip(
                        self.norm_V, (value[:, i, ...] for i in range(self.num_heads))
                    )
                ],
                dim=1,
            )

        elif self.attention_type == "fourier":
            key = torch.stack(
                [
                    norm(x)
                    for norm, x in zip(
                        self.norm_K, (key[:, i, ...] for i in range(self.num_heads))
                    )
                ],
                dim=1,
            )
            query = torch.stack(
                [
                    norm(x)
                    for norm, x in zip(
                        self.norm_Q, (query[:, i, ...] for i in range(self.num_heads))
                    )
                ],
                dim=1,
            )

        x = self._attention(query, key, value)
        x = x.transpose(1, 2).contiguous().view(batchsize, -1, self.in_channels)

        x = self.fc(x)
        x = x.permute(0, 2, 1)

        return x

    @staticmethod
    def _attention(query, key, value):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        seq_len = scores.size(-1)
        p_attn = scores / seq_len
        p_attn = F.dropout(p_attn)
        return torch.matmul(p_attn, value)

    def _get_norm(self, eps):
        if self.attention_type == "galerkin":
            self.norm_K = self._get_layernorm(self.d_k, self.num_heads, eps=eps)
            self.norm_V = self._get_layernorm(self.d_k, self.num_heads, eps=eps)
        elif self.attention_type == "fourier":
            self.norm_K = self._get_layernorm(self.d_k, self.num_heads, eps=eps)
            self.norm_Q = self._get_layernorm(self.d_k, self.num_heads, eps=eps)

    @staticmethod
    def _get_layernorm(normalized_dim, num_heads, **kwargs):
        return nn.ModuleList(
            [
                copy.deepcopy(nn.LayerNorm(normalized_dim, **kwargs))
                for _ in range(num_heads)
            ]
        )

    def _reset_parameters(self):
        for param in self.linears.parameters():
            if param.ndim > 1:
                xavier_uniform_(param, gain=self.xavier_init)
                param.data += self.diagonal_weight * torch.diag(
                    torch.ones(param.size(-1), dtype=torch.float)
                )
            else:
                constant_(param, 0)
