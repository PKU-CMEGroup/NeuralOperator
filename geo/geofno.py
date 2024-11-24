import torch
import sys
import time
import numpy as np
import torch.nn as nn
import math

sys.path.append("../")
from models import Adam, LpLoss, _get_act


################################################################################################
# Compute Bases
################################################################################################
def compute_Fourier_modes(ndim, nks, Ls):

    # 1d
    if ndim == 1:
        k_pairs = np.arange(0, nks + 1)
        k_pairs = 2 * np.pi * k_pairs / Ls
        k_pairs = k_pairs[:, np.newaxis]

    # 2d
    if ndim == 2:
        nx, ny = nks
        Lx, Ly = Ls
        nk = 2 * nx * ny + nx + ny
        k_pairs = np.zeros((nk, ndim))
        k_pair_mag = np.zeros(nk)
        i = 0
        for kx in range(-nx, nx + 1):
            for ky in range(0, ny + 1):
                if ky == 0 and kx <= 0:
                    continue

                k_pairs[i, :] = 2 * np.pi / Lx * kx, 2 * np.pi / Ly * ky
                k_pair_mag[i] = np.linalg.norm(k_pairs[i, :])
                i += 1
        k_pairs = k_pairs[np.argsort(k_pair_mag), :]

    return k_pairs


def compute_Fourier_bases(grid, modes, mask, should_cat=False):
    # grid : batchsize, ndim, nx
    # modes: nk, ndim
    # mask : batchsize, 1, nx
    temp = torch.einsum("bdx,kd->bkx", grid, modes)
    # temp: batchsize, nx, nk
    bases_c = torch.cos(temp) * mask
    bases_s = torch.sin(temp) * mask
    bases_0 = mask
    return bases_c, bases_s, bases_0


class AdjustComputeFourierBases(nn.Module):
    def __init__(self, k_pairs, Ls, should_learn_L=False, should_cat=False):
        super(AdjustComputeFourierBases, self).__init__()

        nmode, self.ndim = k_pairs.shape
        self.Ls = nn.Parameter(
            torch.tensor(Ls, dtype=torch.float), requires_grad=should_learn_L
        )
        # Lx, Ly = L
        # self.Lx = nn.Parameter(
        #     torch.tensor(Lx, dtype=torch.float), requires_grad=should_learn_L
        # )
        # self.Ly = nn.Parameter(
        #     torch.tensor(Ly, dtype=torch.float), requires_grad=should_learn_L
        # )
        self.k_pairs = k_pairs
        self.shoud_learn_L = should_learn_L
        self.should_cat = should_cat

    def forward(self, grid, mask):
        # grid(batchsize, ndim, nx)  mask(batchsize, 1, nx)
        modes = self.k_pairs / self.Ls
        # modes = torch.zeros_like(self.k_pairs)  # (nk, ndim)
        # modes[:, 0] = self.k_pairs[:, 0] / self.Lx
        # modes[:, 1] = self.k_pairs[:, 1] / self.Ly

        temp = torch.einsum("bdx,kd->bkx", grid, modes)  # (batchsize, nx, nk)

        bases_c = torch.cos(temp) * mask
        bases_s = torch.sin(temp) * mask
        bases_0 = mask
        if self.should_cat:
            return torch.cat((bases_c, bases_s, bases_0), dim=1)
        else:
            return bases_c, bases_s, bases_0


################################################################################################
# Layers
################################################################################################
class SpectralConv(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super(SpectralConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        nmode, ndim = modes.shape
        self.modes = modes

        self.scale = 1 / (in_channels * out_channels)

        self.weights_c = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels,
                                    nmode, dtype=torch.float)
        )
        self.weights_s = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels,
                                    nmode, dtype=torch.float)
        )
        self.weights_0 = nn.Parameter(
            self.scale * torch.rand(in_channels,
                                    out_channels, 1, dtype=torch.float)
        )

    def forward(self, x, wbases_c, wbases_s, wbases_0, bases_c, bases_s, bases_0):
        size = x.shape[-1]

        x_c_hat = torch.einsum("bix,bkx->bik", x, wbases_c)
        x_s_hat = -torch.einsum("bix,bkx->bik", x, wbases_s)
        x_0_hat = torch.einsum("bix,bkx->bik", x, wbases_0)

        weights_c, weights_s, weights_0 = (
            self.weights_c / (size),
            self.weights_s / (size),
            self.weights_0 / (size),
        )

        f_c_hat = torch.einsum("bik,iok->bok", x_c_hat, weights_c) - torch.einsum(
            "bik,iok->bok", x_s_hat, weights_s
        )
        f_s_hat = torch.einsum("bik,iok->bok", x_s_hat, weights_c) + torch.einsum(
            "bik,iok->bok", x_c_hat, weights_s
        )
        f_0_hat = torch.einsum("bik,iok->bok", x_0_hat, weights_0)

        x = (
            torch.einsum("bok,bkx->box", f_0_hat, bases_0)
            + 2 * torch.einsum("bok,bkx->box", f_c_hat, bases_c)
            - 2 * torch.einsum("bok,bkx->box", f_s_hat, bases_s)
        )

        return x


class SimpleConv1d(nn.Module):
    def __init__(
        self,
        input_shape,
        depth,
        f_channels,
        u_channels,
        padding_mode="zeros",
        bias=False,
    ):
        super().__init__()

        self.depth = depth
        self.u_channels = u_channels
        self.f_channels = f_channels
        self.padding_mode = padding_mode

        self.Prelayer = nn.Conv1d(
            f_channels,
            u_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias,
            padding_mode=padding_mode,
        )

        # downblock
        self.Slayers = nn.ModuleList()
        for _ in range(depth):
            self.Slayers.append(
                nn.Conv1d(
                    u_channels,
                    u_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=bias,
                    padding_mode=padding_mode,
                )
            )
        self.Pilayers = nn.ModuleList()
        for _ in range(depth):
            self.Pilayers.append(
                nn.Conv1d(
                    u_channels,
                    u_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=bias,
                    padding_mode=padding_mode,
                )
            )
        self.Alayers = nn.ModuleList()
        for _ in range(depth):
            self.Alayers.append(
                nn.Conv1d(
                    u_channels,
                    u_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=bias,
                    padding_mode=padding_mode,
                )
            )

        # upblock
        self.RTlayers = nn.ModuleList()
        for _ in range(depth):
            kernel_size = 4 - input_shape % 2
            self.RTlayers.append(
                nn.ConvTranspose1d(
                    u_channels,
                    u_channels,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=1,
                    bias=False,
                )
            )
            input_shape = (input_shape + 2 - 1) // 2
        self.Postlayers = nn.ModuleList()
        for _ in range(depth):
            self.Postlayers.append(
                nn.Conv1d(
                    u_channels,
                    u_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=bias,
                    padding_mode=padding_mode,
                )
            )

    def forward(self, x):
        depth = self.depth
        out_list = [0] * (depth + 1)
        x = self.Prelayer(x)
        out_list[0] = x

        # downblock
        for l in range(self.depth):
            x = x + self.Slayers[l](x)
            x = self.Pilayers[l](x)
            out_list[l + 1] = x
            x = x + self.Alayers[l](x)

        # upblock
        for j in range(self.depth - 1, -1, -1):
            x = out_list[j] + self.RTlayers[j](x)
            x = x + self.Postlayers[j](x)

        return x


@torch.jit.script
def compute_f_hat(
    x_c_hat: torch.Tensor,
    x_s_hat: torch.Tensor,
    x_0_hat: torch.Tensor,
    weights_c1: torch.Tensor,
    weights_c2: torch.Tensor,
    weights_s1: torch.Tensor,
    weights_s2: torch.Tensor,
    weights_01: torch.Tensor,
    weights_02: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    f_c_hat = torch.einsum(
        "brk,rok->bok",
        torch.einsum("bik,irk->brk", x_c_hat, weights_c1),
        weights_c2,
    ) - torch.einsum(
        "brk,rok->bok",
        torch.einsum("bik,irk->brk", x_s_hat, weights_s1),
        weights_s2,
    )

    f_s_hat = torch.einsum(
        "brk,rok->bok",
        torch.einsum("bik,irk->brk", x_s_hat, weights_c1),
        weights_c2,
    ) + torch.einsum(
        "brk,rok->bok",
        torch.einsum("bik,irk->brk", x_c_hat, weights_s1),
        weights_s2,
    )

    f_0_hat = torch.einsum(
        "brk,rok->bok",
        torch.einsum("bik,irk->brk", x_0_hat, weights_01),
        weights_02,
    )

    return f_c_hat, f_s_hat, f_0_hat


class LowrankSpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes, rank):
        super(LowrankSpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        nmode, ndim = modes.shape
        self.modes = modes

        self.scale = 1 / (in_channels * out_channels)

        self.scale1 = 1 / (in_channels * math.sqrt(rank))
        self.scale2 = 1 / (out_channels * math.sqrt(rank))
        self.weights_c1 = nn.Parameter(
            self.scale1 * torch.rand(in_channels, rank,
                                     nmode, dtype=torch.float)
        )
        self.weights_c2 = nn.Parameter(
            self.scale2 * torch.rand(rank, out_channels,
                                     nmode, dtype=torch.float)
        )
        self.weights_s1 = nn.Parameter(
            self.scale1 * torch.rand(in_channels, rank,
                                     nmode, dtype=torch.float)
        )
        self.weights_s2 = nn.Parameter(
            self.scale2 * torch.rand(rank, out_channels,
                                     nmode, dtype=torch.float)
        )
        self.weights_01 = nn.Parameter(
            self.scale1 * torch.rand(in_channels, rank, 1, dtype=torch.float)
        )
        self.weights_02 = nn.Parameter(
            self.scale2 * torch.rand(rank, out_channels, 1, dtype=torch.float)
        )

    def forward(self, x, wbases_c, wbases_s, wbases_0, bases_c, bases_s, bases_0):

        size_sqrt = math.sqrt(x.shape[-1])

        x_c_hat = torch.einsum("bix,bkx->bik", x, wbases_c)
        x_s_hat = -torch.einsum("bix,bkx->bik", x, wbases_s)
        x_0_hat = torch.einsum("bix,bkx->bik", x, wbases_0)

        weights_c1, weights_c2, weights_s1, weights_s2, weights_01, weights_02 = (
            self.weights_c1 / size_sqrt,
            self.weights_c2 / size_sqrt,
            self.weights_s1 / size_sqrt,
            self.weights_s2 / size_sqrt,
            self.weights_01 / size_sqrt,
            self.weights_02 / size_sqrt,
        )

        f_c_hat, f_s_hat, f_0_hat = compute_f_hat(
            x_c_hat,
            x_s_hat,
            x_0_hat,
            weights_c1,
            weights_c2,
            weights_s1,
            weights_s2,
            weights_01,
            weights_02,
        )

        x = (
            torch.einsum("bok,bkx->box", f_0_hat, bases_0)
            + 2 * torch.einsum("bok,bkx->box", f_c_hat, bases_c)
            - 2 * torch.einsum("bok,bkx->box", f_s_hat, bases_s)
        )

        return x


@torch.jit.script
def compl_mul1d_lowrank(
    x_hat: torch.Tensor, weights_1: torch.Tensor, weights_2: torch.Tensor
) -> torch.Tensor:
    x_hat = torch.einsum("bik,irk->brk", x_hat, weights_1)
    x_hat = torch.einsum("brk,rok->bok", x_hat, weights_2)
    return x_hat


class LowrankGalerkinConv(nn.Module):
    def __init__(self, in_channels, out_channels, modes, rank):
        super(LowrankGalerkinConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        nmode, _ = modes.shape
        self.modes = 2 * nmode + 1

        self.scale1 = 1 / (in_channels * math.sqrt(rank))
        self.scale2 = 1 / (out_channels * math.sqrt(rank))

        self.weights_1 = nn.Parameter(
            self.scale1 * torch.rand(in_channels, rank,
                                     self.modes, dtype=torch.float)
        )
        self.weights_2 = nn.Parameter(
            self.scale2 * torch.rand(rank, out_channels,
                                     self.modes, dtype=torch.float)
        )

    def forward(self, x, wbases, bases):

        size_sqrt = math.sqrt(x.shape[-1])
        weights_1, weights_2 = (self.weights_1 / size_sqrt,
                                self.weights_2 / size_sqrt)

        x_hat = torch.einsum("bix,bkx->bik", x, wbases)
        x_hat = compl_mul1d_lowrank(x_hat, weights_1, weights_2)
        x = torch.einsum("bok,bkx->box", x_hat, bases)
        return x


################################################################################
# Model
################################################################################
class GeoFNO(nn.Module):
    def __init__(
        self,
        ndim,
        modes,
        layers,
        fc_dim=128,
        in_dim=3,
        out_dim=1,
        act="gelu",
    ):
        super(GeoFNO, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """
        self.modes = modes
        self.layers = layers
        self.fc_dim = fc_dim
        self.ndim = ndim
        self.in_dim = in_dim

        self.fc0 = nn.Linear(in_dim, layers[0])

        self.sp_convs = nn.ModuleList(
            [
                SpectralConv(in_size, out_size, modes)
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
        """
        Args:
            - x : (batch size, x_grid, y_grid, 2)
        Returns:
            - x: (batch size, x_grid, y_grid, 1)
        """
        length = len(self.ws)

        aux = x[..., -2 - self.ndim:].permute(0, 2, 1)  # coord, weights, mask

        grid, weights, mask = aux[:, 0: self.ndim,
                                  :], aux[:, -2:-1, :], aux[:, -1:, :]

        size = grid.shape[-1]
        bases_c, bases_s, bases_0 = compute_Fourier_bases(
            grid, self.modes, mask)
        wbases_c, wbases_s, wbases_0 = (
            bases_c * (weights * size),
            bases_s * (weights * size),
            bases_0 * (weights * size),
        )

        x = self.fc0(x[..., 0: self.in_dim])
        x = x.permute(0, 2, 1)

        for i, (speconv, w) in enumerate(zip(self.sp_convs, self.ws)):
            x1 = speconv(x, wbases_c, wbases_s, wbases_0,
                         bases_c, bases_s, bases_0)
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


class DoubleGeoFNO(nn.Module):
    def __init__(
        self,
        ndim,
        modes,
        layers,
        depth=4,
        fc_dim=128,
        in_dim=3,
        out_dim=1,
        act="gelu",
    ):
        super(DoubleGeoFNO, self).__init__()

        self.modes = modes
        self.layers = layers
        self.depth = depth
        self.fc_dim = fc_dim
        self.ndim = ndim
        self.in_dim = in_dim

        self.fc0 = nn.Linear(in_dim, layers[0])

        self.sp_convs = nn.ModuleList(
            [
                SpectralConv(in_size, out_size, modes)
                for in_size, out_size in zip(self.layers, self.layers[1:])
            ]
        )
        self.simple_convs = nn.ModuleList(
            [
                SimpleConv1d(
                    512,
                    self.depth,
                    in_size,
                    out_size,
                )
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
        """
        Args:
            - x : (batch size, x_grid, y_grid, 2)
        Returns:
            - x: (batch size, x_grid, y_grid, 1)
        """
        length = len(self.ws)

        aux = x[..., -2 - self.ndim:].permute(0, 2, 1)  # coord, weights, mask

        grid, weights, mask = aux[:, 0: self.ndim,
                                  :], aux[:, -2:-1, :], aux[:, -1:, :]

        size = grid.shape[-1]
        bases_c, bases_s, bases_0 = compute_Fourier_bases(
            grid, self.modes, mask)
        wbases_c, wbases_s, wbases_0 = (
            bases_c * (weights * size),
            bases_s * (weights * size),
            bases_0 * (weights * size),
        )

        x = self.fc0(x[..., 0: self.in_dim])
        x = x.permute(0, 2, 1)

        for i, (speconv, w, simpleconv) in enumerate(
            zip(self.sp_convs, self.ws, self.simple_convs)
        ):
            x1 = speconv(x, wbases_c, wbases_s, wbases_0,
                         bases_c, bases_s, bases_0)
            x2 = w(x)
            x3 = simpleconv(x)
            x = (x1 + x2 + x3) / 3
            # x=x1+x3
            if self.act is not None and i != length - 1:
                x = self.act(x)

        x = x.permute(0, 2, 1)

        if self.fc_dim > 0:
            x = self.fc1(x)
            if self.act is not None:
                x = self.act(x)

        x = self.fc2(x)
        return x


class AdjustGeoFNO(nn.Module):
    def __init__(
        self,
        ndim,
        k_pairs,
        L,
        should_learn_L,
        layers,
        fc_dim=128,
        in_dim=3,
        out_dim=1,
        act="gelu",
    ):
        super(AdjustGeoFNO, self).__init__()

        self.layers = layers
        self.fc_dim = fc_dim
        self.ndim = ndim
        self.in_dim = in_dim

        self.fc0 = nn.Linear(in_dim, layers[0])

        self.bases = AdjustComputeFourierBases(k_pairs, L, should_learn_L)
        self.sp_convs = nn.ModuleList(
            [
                SpectralConv(in_size, out_size, k_pairs)
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
        print(f"initial L={L},should_learn_L={should_learn_L}")

    def forward(self, x):

        length = len(self.ws)

        aux = x[..., -2 - self.ndim:].permute(0, 2, 1)  # coord, weights, mask

        grid, weights, mask = aux[:, 0: self.ndim,
                                  :], aux[:, -2:-1, :], aux[:, -1:, :]

        size = grid.shape[-1]
        bases_c, bases_s, bases_0 = self.bases(grid, mask)
        wbases_c, wbases_s, wbases_0 = (
            bases_c * (weights * size),
            bases_s * (weights * size),
            bases_0 * (weights * size),
        )

        x = self.fc0(x[..., 0: self.in_dim])
        x = x.permute(0, 2, 1)

        for i, (speconv, w) in enumerate(zip(self.sp_convs, self.ws)):
            x1 = speconv(x, wbases_c, wbases_s, wbases_0,
                         bases_c, bases_s, bases_0)
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


class LowrankGeoFNO(nn.Module):
    def __init__(
        self,
        ndim,
        k_pairs,
        L,
        rank,
        layers,
        fc_dim=128,
        in_dim=3,
        out_dim=1,
        act="gelu",
        should_learn_L=False,
    ):
        super(LowrankGeoFNO, self).__init__()

        self.layers = layers
        self.fc_dim = fc_dim
        self.ndim = ndim
        self.in_dim = in_dim

        self.fc0 = nn.Linear(in_dim, layers[0])

        self.bases = AdjustComputeFourierBases(k_pairs, L, should_learn_L)
        self.sp_convs = nn.ModuleList(
            [
                LowrankSpectralConv2d(in_size, out_size, k_pairs, rank)
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
        print(f"initial L={L},should_learn_L={should_learn_L},rank={rank}")

    def forward(self, x):

        length = len(self.ws)

        aux = x[..., -2 - self.ndim:].permute(0, 2, 1)  # coord, weights, mask

        grid, weights, mask = aux[:, 0: self.ndim,
                                  :], aux[:, -2:-1, :], aux[:, -1:, :]

        size = grid.shape[-1]
        bases_c, bases_s, bases_0 = self.bases(grid, mask)
        wbases_c, wbases_s, wbases_0 = (
            bases_c * (weights * size),
            bases_s * (weights * size),
            bases_0 * (weights * size),
        )

        x = self.fc0(x[..., 0: self.in_dim])
        x = x.permute(0, 2, 1)

        for i, (speconv, w) in enumerate(zip(self.sp_convs, self.ws)):
            x1 = speconv(x, wbases_c, wbases_s, wbases_0,
                         bases_c, bases_s, bases_0)
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


class LowrankGeoGkNO(nn.Module):
    def __init__(
        self,
        ndim,
        k_pairs,
        L,
        should_learn_L,
        rank,
        layers,
        fc_dim=128,
        in_dim=3,
        out_dim=1,
        act="gelu",
    ):
        super(LowrankGeoGkNO, self).__init__()

        self.layers = layers
        self.fc_dim = fc_dim
        self.ndim = ndim
        self.in_dim = in_dim

        self.fc0 = nn.Linear(in_dim, layers[0])

        self.bases = AdjustComputeFourierBases(
            k_pairs, L, should_learn_L, should_cat=True
        )
        self.sp_convs = nn.ModuleList(
            [
                LowrankGalerkinConv(in_size, out_size, k_pairs, rank)
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
        print(f"initial L={L},should_learn_L={should_learn_L},rank={rank}")

    def forward(self, x):

        length = len(self.ws)

        aux = x[..., -2 - self.ndim:].permute(0, 2, 1)  # coord, weights, mask

        grid, weights, mask = aux[:, 0: self.ndim,
                                  :], aux[:, -2:-1, :], aux[:, -1:, :]

        size = grid.shape[-1]
        bases = self.bases(grid, mask)
        wbases = bases * (weights * size)

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


# x_train, y_train, x_test, y_test are [n_data, n_x, n_channel] arrays
def GeoFNO_train(
    x_train,
    y_train,
    x_test,
    y_test,
    config,
    model,
    should_print_L=False,
    should_print_final=False,
):
    n_train, n_test = x_train.shape[0], x_test.shape[0]
    train_rel_l2_losses = []
    test_rel_l2_losses = []
    test_l2_losses = []
    normalization_x, normalization_y, normalization_dim = (
        config["train"]["normalization_x"],
        config["train"]["normalization_y"],
        config["train"]["normalization_dim"],
    )

    ndim = model.ndim  # n_train, size, n_channel
    print("In GeoFNO_train, ndim = ", ndim)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if normalization_x:
        x_normalizer = UnitGaussianNormalizer(x_train, aux_dim=ndim + 2)
        x_train = x_normalizer.encode(x_train)
        x_test = x_normalizer.encode(x_test)
        x_normalizer.to(device)

    if normalization_y:
        y_normalizer = UnitGaussianNormalizer(y_train, aux_dim=0)
        y_train = y_normalizer.encode(y_train)
        y_test = y_normalizer.encode(y_test)
        y_normalizer.to(device)

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_train, y_train),
        batch_size=config["train"]["batch_size"],
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_test, y_test),
        batch_size=config["train"]["batch_size"],
        shuffle=False,
    )

    # Load from checkpoint
    optimizer = Adam(
        model.parameters(),
        betas=(0.9, 0.999),
        lr=config["train"]["base_lr"],
        weight_decay=config["train"]["weight_decay"],
    )

    if config["train"]["scheduler"] == "MultiStepLR":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=config["train"]["milestones"],
            gamma=config["train"]["scheduler_gamma"],
        )
    elif config["train"]["scheduler"] == "CosineAnnealingLR":
        T_max = (config["train"]["epochs"] // 10) * (
            n_train // config["train"]["batch_size"]
        )
        eta_min = 0.0
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=T_max, eta_min=eta_min
        )
    elif config["train"]["scheduler"] == "OneCycleLR":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config["train"]["base_lr"],
            div_factor=2,
            final_div_factor=100,
            pct_start=0.2,
            steps_per_epoch=1,
            epochs=config["train"]["epochs"],
        )
    else:
        print("Scheduler ", config["train"]
              ["scheduler"], " has not implemented.")

    model.train()
    myloss = LpLoss(d=1, p=1, size_average=False)
    testloss = LpLoss(d=1, p=2, size_average=False)

    epochs = config["train"]["epochs"]

    for ep in range(epochs):
        time_start = time.time()
        train_rel_l2 = 0
        train_t_rel_l2 = 0

        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            batch_size_ = x.shape[0]
            optimizer.zero_grad()
            out = model(x)  # .reshape(batch_size_,  -1)
            if normalization_y:
                out = y_normalizer.decode(out)
                y = y_normalizer.decode(y)

            loss = myloss(out.view(batch_size_, -1), y.view(batch_size_, -1))
            loss.backward()

            optimizer.step()
            train_rel_l2 += loss.item()
            train_t_rel_l2 += myloss(
                out[..., -1].view(batch_size_, -1), y[...,
                                                      - 1].view(batch_size_, -1)
            ).item()

        test_l2 = 0
        test_rel_l2 = 0
        test_rel_l1 = 0
        test_t_rel_l2 = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                batch_size_ = x.shape[0]
                out = model(x)  # .reshape(batch_size_,  -1)

                if normalization_y:
                    out = y_normalizer.decode(out)
                    y = y_normalizer.decode(y)

                test_rel_l2 += myloss(
                    out.view(batch_size_, -1), y.view(batch_size_, -1)
                ).item()
                test_l2 += myloss.abs(
                    out.view(batch_size_, -1), y.view(batch_size_, -1)
                ).item()
                test_rel_l1 += testloss(
                    out.view(batch_size_, -1), y.view(batch_size_, -1)
                ).item()
                test_t_rel_l2 += myloss(
                    out[..., -1].view(batch_size_,
                                      - 1), y[..., -1].view(batch_size_, -1)
                ).item()

        scheduler.step()

        train_rel_l2 /= n_train
        train_t_rel_l2 /= n_train
        test_l2 /= n_test
        test_rel_l2 /= n_test
        test_rel_l1 /= n_train
        test_t_rel_l2 /= n_test

        train_rel_l2_losses.append(train_rel_l2)
        test_rel_l2_losses.append(test_rel_l2)
        test_l2_losses.append(test_l2)

        time_end = time.time()

        if should_print_L == True:
            print(
                "ep:",
                ep,
                " rel.L1 train:",
                train_rel_l2,
                " rel.L1 test:",
                test_rel_l2,
                " rel.L2 test:",
                test_rel_l1,
                " time:",
                time_end - time_start,
                " L:",
                [l.item() for l in model.bases.Ls],
                flush=True,
            )
        else:
            print(
                "ep:",
                ep,
                " rel.L1 train:",
                train_rel_l2,
                " rel.L1 test:",
                test_rel_l2,
                " rel.L2 test:",
                test_rel_l1,
                " time:",
                time_end - time_start,
                flush=True,
            )
        if should_print_final:
            print(
                " rel.final train:",
                train_t_rel_l2,
                " rel.final test:",
                test_t_rel_l2,
            )
    return train_rel_l2_losses, test_rel_l2_losses, test_l2_losses
