import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import operator
from functools import reduce
import numpy as np


def add_padding(x, pad_nums):

    if x.ndim == 3:  # fourier1d
        res = F.pad(x, [0, pad_nums[0]], "constant", 0)
    elif x.ndim == 4:  # fourier2d
        res = F.pad(x, [0, pad_nums[1], 0, pad_nums[0]], "constant", 0)
    elif x.ndim == 5:  # fourier3d
        res = F.pad(x, [0, pad_nums[2], 0, pad_nums[1], 0, pad_nums[0]], "constant", 0)
    elif x.ndim == 6:  # fourier4d
        res = F.pad(
            x,
            [0, pad_nums[3], 0, pad_nums[2], 0, pad_nums[1], 0, pad_nums[0]],
            "constant",
            0,
        )
    else:
        print("error : x.ndim = ", x.ndim)

    return res


def remove_padding(x, pad_nums):

    if x.ndim == 3:  # fourier1d
        res = x[..., : (None if pad_nums[0] == 0 else -pad_nums[0])]

    elif x.ndim == 4:  # fourier2d
        res = x[
            ...,
            : (None if pad_nums[0] == 0 else -pad_nums[0]),
            : (None if pad_nums[1] == 0 else -pad_nums[1]),
        ]

    elif x.ndim == 5:  # fourier3d
        res = x[
            ...,
            : (None if pad_nums[0] == 0 else -pad_nums[0]),
            : (None if pad_nums[1] == 0 else -pad_nums[1]),
            : (None if pad_nums[2] == 0 else -pad_nums[2]),
        ]

    elif x.ndim == 6:  # fourier4d
        res = x[
            ...,
            : (None if pad_nums[0] == 0 else -pad_nums[0]),
            : (None if pad_nums[1] == 0 else -pad_nums[1]),
            : (None if pad_nums[2] == 0 else -pad_nums[2]),
            : (None if pad_nums[3] == 0 else -pad_nums[3]),
        ]

    else:
        print("error : x.ndim = ", x.ndim)

    return res


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
    elif act == "none":
        func = None
    else:
        raise ValueError(f"{act} is not supported")
    return func


def count_params(model):
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul, list(p.size() + (2,) if p.is_complex() else p.size()))
    return c


def compute_1dFourier_bases(nx, k, Lx):
    grid = np.linspace(0, Lx, nx + 1)[:-1]
    bases = np.zeros((nx, k))
    bases[:, 0] = 1 / np.sqrt(Lx)
    weights = np.ones(nx) * Lx / nx
    for i in range(k // 2):
        bases[:, 2 * i + 1] = np.sqrt(2 / Lx) * np.cos(2 * np.pi * (i + 1) * grid / Lx)
        if 2 * i + 2 <= k - 1:
            bases[:, 2 * i + 2] = np.sqrt(2 / Lx) * np.sin(
                2 * np.pi * (i + 1) * grid / Lx
            )
    return grid, bases, weights


def compute_1dWeights(grid):
    weight_start = 1 / 2 * (grid[1] - grid[0]).unsqueeze(0)
    weight_end = 1 / 2 * (grid[-1] - grid[-2]).unsqueeze(0)
    weights_mid = 1 / 2 * (grid[2:] - grid[:-2])
    weights = torch.cat((weight_start, weights_mid, weight_end))
    return weights


def compute_1dFourier_bases_arbitrary(s, k_max, grid, weights):
    L = grid[-1] - grid[0]
    fbases = torch.zeros((s, k_max))
    fbases[:, 0] = 1 / np.sqrt(L)
    for i in range(k_max // 2):
        fbases[:, 2 * i + 1] = torch.sqrt(2 / L) * torch.cos(
            2 * torch.pi * (i + 1) * grid / L
        )
        if 2 * i + 2 <= k_max - 1:
            fbases[:, 2 * i + 2] = torch.sqrt(2 / L) * torch.sin(
                2 * torch.pi * (i + 1) * grid / L
            )
    bases = fbases
    wbases = torch.einsum("sm,s->sm", fbases, weights)
    return bases, wbases


def compute_2dFourier_modes(k):
    trunc_k = np.int64(np.sqrt(k)) + 1

    k_pairs = np.zeros(((2 * trunc_k + 1) ** 2, 2))
    k_pair_mag = np.zeros((2 * trunc_k + 1) ** 2)

    i = 0
    for kx in range(-trunc_k, trunc_k + 1):
        for ky in range(-trunc_k, trunc_k + 1):

            k_pairs[i, :] = kx, ky
            k_pair_mag[i] = kx**2 + ky**2
            i += 1

    k_pairs = k_pairs[np.argsort(k_pair_mag), :]
    return k_pairs[0:k, :]


def compute_2dFourier_bases(nx, ny, k, Lx, Ly):
    gridx, gridy = np.meshgrid(
        np.linspace(0, Lx, nx + 1)[:-1], np.linspace(0, Ly, ny + 1)[:-1]
    )
    bases = np.zeros((nx, ny, k))

    weights = np.ones((nx, ny)) * Lx * Ly / (nx * ny)

    k_pairs = compute_2dFourier_modes(k)
    for i in range(k):
        kx, ky = k_pairs[i, :]
        if kx == 0 and ky == 0:
            bases[:, :, i] = np.sqrt(1 / (Lx * Ly))
        elif ky > 0 or ky == 0 and kx > 0:
            bases[:, :, i] = np.sqrt(2 / (Lx * Ly)) * np.sin(
                2 * np.pi * (kx * gridx / Lx + ky * gridy / Ly)
            )
        else:
            bases[:, :, i] = np.sqrt(2 / (Lx * Ly)) * np.cos(
                2 * np.pi * (kx * gridx / Lx + ky * gridy / Ly)
            )
    return gridx, gridy, bases, weights


def compute_1dpca_bases (Ne , k_max , L,  pca_data):
    U, S, VT = np.linalg.svd(pca_data.T)
    # the integration of the basis is 1.
    fbases = U[:, 0:k_max] / np.sqrt(L / Ne)
    wfbases = L / Ne * fbases
    bases_pca = torch.from_numpy(fbases.astype(np.float32))
    wbases_pca = torch.from_numpy(wfbases.astype(np.float32))
    return bases_pca,wbases_pca

def compute_2dpca_bases (Np , k_max , L,  pca_data):
    U, S, VT = np.linalg.svd(pca_data.T, full_matrices=False)
    fbases = U[:, 0:k_max] / np.sqrt(L * L / Np**2)
    wfbases = L * L / Np**2 * fbases
    bases_pca = torch.from_numpy(fbases.astype(np.float32))
    wbases_pca = torch.from_numpy(wfbases.astype(np.float32))
    return bases_pca,wbases_pca