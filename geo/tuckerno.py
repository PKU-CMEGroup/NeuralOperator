import torch
import sys
import time
import numpy as np
import torch.nn as nn
import math

sys.path.append("../")

from models import _get_act

################################################################################################
# Compute Bases
################################################################################################
def compute_fourier_pairs(ndim, k, Ls):

    # 1d
    if ndim == 1:
        trunc_k = k // 2 + 1

        k_pairs = np.arange(-trunc_k, trunc_k + 1)
        k_pair_mag = np.abs(k_pairs)
        k_pairs = k_pairs / Ls
        k_pairs = k_pairs[np.argsort(k_pair_mag), : np.newaxis]

    # 2d
    elif ndim == 2:
        Lx, Ly = Ls
        trunc_k = np.int64(np.sqrt(k)) + 1

        k_pairs = np.zeros(((2 * trunc_k + 1) ** 2, 2))
        k_pair_mag = np.zeros((2 * trunc_k + 1) ** 2)

        i = 0
        for kx in range(-trunc_k, trunc_k + 1):
            for ky in range(-trunc_k, trunc_k + 1):
                k_pairs[i, :] = 2 * np.pi * kx / Lx, 2 * np.pi * ky / Ly
                k_pair_mag[i] = kx**2 + ky**2
                i += 1
        k_pairs = k_pairs[np.argsort(k_pair_mag), :]

    return k_pairs[0:k, :]


def compute_fourier_bases2d(grid, k_pairs, mask=None):
    temp = torch.einsum("bdx,kd->bkx", grid, k_pairs)

    bases = torch.zeros_like(temp)

    mask_zero = 1
    mask_cos = (k_pairs[:, 1] > 0) | (
        (k_pairs[:, 1] == 0) & (k_pairs[:, 0] > 0))

    bases[:, mask_zero, :] = 1
    bases[:, mask_cos, :] = math.sqrt(2) * torch.cos(temp)[:, mask_cos, :]
    bases[:, ~mask_zero & ~mask_cos, :] = (
        math.sqrt(2) * torch.sin(temp)[:, ~mask_zero & ~mask_cos, :]
    )

    return bases


################################################################################################
# Layers
################################################################################################


@torch.jit.script
def mul_tucker_all(
    x: torch.Tensor,
    G: torch.Tensor,
    I: torch.Tensor,
    O: torch.Tensor,
    K: torch.Tensor,
    L: torch.Tensor,
) -> torch.Tensor:
    x = torch.einsum("bik,ir,ks->brs", x, I, K)
    x = torch.einsum("brs,rstq->btq", x, G)
    x = torch.einsum("btq,ot,lq->bol", x, O, L)
    return x


class TuckerGalerkinConv(nn.Module):
    def __init__(self, in_channels, out_channels, k_pairs, rank):
        super(TuckerGalerkinConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes, _ = k_pairs.shape

        self.w_I = nn.Parameter(
            torch.rand(in_channels, rank[0], dtype=torch.float) / in_channels
        )
        self.w_O = nn.Parameter(
            torch.rand(out_channels, rank[1], dtype=torch.float) / out_channels
        )
        self.w_K = nn.Parameter(
            torch.rand(self.modes, rank[2], dtype=torch.float) / self.modes
        )
        self.w_L = nn.Parameter(
            torch.rand(self.modes, rank[3], dtype=torch.float) / self.modes
        )
        self.w_G = nn.Parameter(
            torch.rand(rank[0], rank[1], rank[2], rank[3], dtype=torch.float)
        )

    def forward(self, x, wbases, bases):
        size = x.shape[-1]

        x_hat = torch.einsum("bix,bkx->bik", x, wbases)
        x_hat = mul_tucker_all(x_hat, self.w_G, self.w_I,
                               self.w_O, self.w_K, self.w_L)
        x = torch.einsum("bok,bkx->box", x_hat, bases)
        return x


################################################################################
# Model
################################################################################
class TuckerGeoGkNO(nn.Module):
    def __init__(
        self,
        ndim,
        k_pairs,
        layers,
        rank,
        fc_dim=128,
        in_dim=3,
        out_dim=1,
        act="gelu",
    ):
        super(TuckerGeoGkNO, self).__init__()

        self.k_pairs = k_pairs
        self.layers = layers
        self.fc_dim = fc_dim
        self.ndim = ndim
        self.in_dim = in_dim
        self.rank = rank

        self.fc0 = nn.Linear(in_dim, layers[0])

        self.sp_convs = nn.ModuleList(
            [
                TuckerGalerkinConv(in_size, out_size, k_pairs, self.rank)
                for in_size, out_size in zip(self.layers, self.layers[1:])
            ]
        )
        self.ws = nn.ModuleList(
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

    def forward(self, x):
        length = len(self.ws)

        aux = x[..., -2 - self.ndim:].permute(0, 2, 1)
        grid, weights, mask = aux[:, 0: self.ndim,
                                  :], aux[:, -2:-1, :], aux[:, -1:, :]

        bases = compute_fourier_bases2d(grid, self.k_pairs)
        wbases = bases * weights

        x = self.fc0(x[..., 0: self.in_dim])
        x = x.permute(0, 2, 1)

        for i, (speconv, w) in enumerate(zip(self.sp_convs, self.ws)):
            x1 = speconv(x, wbases, bases)
            x2 = w(x)
            x = x1 + x2
            if self.act is not None and i != length - 1:
                x = self.act(x)

        x = x.permute(0, 2, 1)

        if self.fc_dim > 0:
            x = self.fc1(x)
            if self.act is not None:
                x = self.act(x)

        x = self.fc2(x)
        return x


################################################################################################
# Training
################################################################################################
class UnitGaussianNormalizer(object):
    def __init__(self, x, aux_dim=0, eps=1.0e-5):
        super(UnitGaussianNormalizer, self).__init__()
        # x: ndata, nx, nchannels
        # when dim = [], mean and std are both scalars
        self.aux_dim = aux_dim
        self.mean = torch.mean(x[..., 0: x.shape[-1] - aux_dim])
        self.std = torch.std(x[..., 0: x.shape[-1] - aux_dim])
        self.eps = eps

    def encode(self, x):
        x[..., 0: x.shape[-1] - self.aux_dim] = (
            x[..., 0: x.shape[-1] - self.aux_dim] - self.mean
        ) / (self.std + self.eps)
        return x

    def decode(self, x):
        std = self.std + self.eps  # n
        mean = self.mean
        x[..., 0: x.shape[-1] - self.aux_dim] = (
            x[..., 0: x.shape[-1] - self.aux_dim] * std
        ) + mean
        return x

    def to(self, device):
        if device == torch.device("cuda:0"):
            self.mean = self.mean.cuda()
            self.std = self.std.cuda()
        else:
            self.mean = self.mean.cpu()
            self.std = self.std.cpu()
