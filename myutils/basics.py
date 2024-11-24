import torch
import operator
import numpy as np
import torch.nn.functional as F
import scipy.linalg.interpolative as sli
from functools import reduce


######################################################################
# multiplication
######################################################################
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


@torch.jit.script
def compl_mul1d_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # (batch, in_channel, x ), (in_channel, out_channel, x,y) -> (batch, out_channel, x)
    res = torch.einsum("bix,ioxy->boy", a, b)
    return res


# @torch.jit.script
# def mul_lowrank(
#     U1: torch.Tensor, U2: torch.Tensor, M: torch.Tensor, x: torch.Tensor
# ) -> torch.Tensor:
#     x = torch.einsum("dmks,bidms->bidmk", U2, x)
#     x = torch.einsum("dmkl,bidmk->bidml", M, x)
#     x = torch.einsum("dmsl,bidml->bidms", U1, x)
#     return x


@torch.jit.script
def mul_index1(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # (batch, n, heads, d), (batch, n, index, heads, d) -> (batch, n, index, heads)
    res = torch.einsum("bnhd,bnihd->bnih", a, b)
    return res


@torch.jit.script
def mul_index2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # (batch, n, index, heads), (batch, n, index, heads, d) -> (batch, n, heads, d)
    res = torch.einsum("bnih,bnihd->bnhd", a, b)
    return res


######################################################################
# compute bases
######################################################################
def generate_1dGrid_arbitrary(gridsize, s, seed):
    index_selected = np.random.choice(range(gridsize), s, replace=False, seed=seed)
    index_selected.sort()
    index_selected[0], index_selected[s - 1] = 0, gridsize - 1
    grid_selected = torch.tensor(index_selected) / gridsize
    return index_selected, grid_selected


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


def compute_1dpca_bases(Ne, k_max, L, pca_data):
    U, _, _ = np.linalg.svd(pca_data.T)
    fbases = U[:, 0:k_max] / np.sqrt(L / Ne)
    wfbases = L / Ne * fbases
    bases_pca = torch.from_numpy(fbases.astype(np.float32))
    wbases_pca = torch.from_numpy(wfbases.astype(np.float32))
    return bases_pca, wbases_pca


def compute_2dpca_bases(Np, k_max, L, pca_data):
    U, _, _ = np.linalg.svd(pca_data.T, full_matrices=False)
    bases = U[:, 0:k_max] / np.sqrt(L * L / Np**2)
    wbases = L * L / Np**2 * bases
    bases_pca = torch.from_numpy(bases.astype(np.float32))
    wbases_pca = torch.from_numpy(wbases.astype(np.float32))
    return bases_pca, wbases_pca


# def compute_fourier2d_bases(nx, ny, k, Lx, Ly, should_flatten):

#     gridx, gridy = np.meshgrid(
#         np.linspace(0, Ly, ny + 1)[:-1], np.linspace(0, Lx, nx + 1)[:-1]
#     )
#     k_pairs = compute_2dFourier_modes(k)

#     bases = np.zeros((nx, ny, k))
#     for i in range(k):
#         kx, ky = k_pairs[i, :]
#         if kx == 0 and ky == 0:
#             bases[:, :, i] = np.sqrt(1 / (Lx * Ly))
#         elif ky > 0 or ky == 0 and kx > 0:
#             bases[:, :, i] = np.sqrt(2 / (Lx * Ly)) * np.sin(
#                 2 * np.pi * (kx * gridx / Lx + ky * gridy / Ly)
#             )
#         else:
#             bases[:, :, i] = np.sqrt(2 / (Lx * Ly)) * np.cos(
#                 2 * np.pi * (kx * gridx / Lx + ky * gridy / Ly)
#             )

#     weights = np.ones((nx, ny, 1)) * Lx * Ly / (nx * ny)
#     wbases = bases * weights
#     if should_flatten:
#         bases = bases.reshape(-1, k)
#         wbases = wbases.reshape(-1, k)

#     return bases, wbases


def compute_fourier2d_bases(nx, ny, k, Lx, Ly, should_flatten):

    gridx, gridy = np.meshgrid(
        np.linspace(0, Ly, ny + 1)[:-1], np.linspace(0, Lx, nx + 1)[:-1]
    )
    k_pairs = compute_2dFourier_modes(k)

    bases = np.zeros((nx, ny, k))
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

    weights = np.ones((nx, ny, 1)) * Lx * Ly / (nx * ny)
    wbases = bases * weights
    if should_flatten:
        bases = bases.reshape(-1, k)
        wbases = wbases.reshape(-1, k)

    return bases, wbases


def compute_pca_bases(dX, k_max, data, should_flatten):
    # when dX is fixed
    n = data.shape[0]
    shape = data.shape[1:]
    data = data.reshape(n, -1)
    U, _, _ = np.linalg.svd(data.T, full_matrices=False)
    bases = U[:, 0:k_max] / np.sqrt(dX)
    wbases = dX * bases
    if not should_flatten:
        bases = bases.reshape(shape + (k_max,))
        wbases = wbases.reshape(shape + (k_max,))
    return bases, wbases


def compute_id_bases(dX, k_max, data, should_flatten):
    data = data.astype(np.float64)

    n = data.shape[0]
    shape = data.shape[1:]
    data = data.reshape(n, -1)

    idx, proj = sli.interp_decomp(data, k_max)
    bases = sli.reconstruct_interp_matrix(idx, proj).T
    wbases = dX * bases

    if not should_flatten:
        bases = bases.reshape(shape + (k_max,))
        wbases = wbases.reshape(shape + (k_max,))

    return bases, wbases


######################################################################
# padding
######################################################################
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


######################################################################
# distance & index
######################################################################
def pairwise_dist(res1x, res1y, res2x, res2y):
    gridx1 = torch.linspace(0, 1, res1x + 1)[:-1].view(1, -1, 1).repeat(res1y, 1, 1)
    gridy1 = torch.linspace(0, 1, res1y + 1)[:-1].view(-1, 1, 1).repeat(1, res1x, 1)
    grid1 = torch.cat([gridx1, gridy1], dim=-1).view(res1x * res1y, 2)

    gridx2 = torch.linspace(0, 1, res2x + 1)[:-1].view(1, -1, 1).repeat(res2y, 1, 1)
    gridy2 = torch.linspace(0, 1, res2y + 1)[:-1].view(-1, 1, 1).repeat(1, res2x, 1)
    grid2 = torch.cat([gridx2, gridy2], dim=-1).view(res2x * res2y, 2)

    grid1 = grid1.unsqueeze(1).repeat(1, grid2.shape[0], 1)
    grid2 = grid2.unsqueeze(0).repeat(grid1.shape[0], 1, 1)

    dist = torch.norm(grid1 - grid2, dim=-1)
    return (dist**2 / 2.0).float()


def indices_dist2d(res_x, res_y, tol):

    dist_matrix = pairwise_dist(res_x, res_y, res_x, res_y)
    index = [
        (dist_matrix[i] < tol).nonzero(as_tuple=True)[0]
        for i in range(dist_matrix.size(0))
    ]

    return index


def indices_m2v(A, n2):
    i = A[..., 0]
    j = A[..., 1]
    idx = i * n2 + j
    return idx


def indices_neighbor2d(n1, n2, k1, k2, should_flatten=False, offset=None):
    i, j = torch.arange(n1), torch.arange(n2)
    i, j = torch.meshgrid(i, j, indexing="ij")
    center = torch.stack([i, j], dim=-1)

    k = (2 * k1 + 1) * (2 * k2 + 1)
    if offset == None:
        offset_i = torch.arange(-k1, k1 + 1)
        offset_j = torch.arange(-k2, k2 + 1)
        offset_i, offset_j = torch.meshgrid(offset_i, offset_j, indexing="ij")
        offset = torch.stack([offset_i.reshape(-1), offset_j.reshape(-1)], dim=-1)

    center = center.unsqueeze(2).expand(n1, n2, k, 2)
    neighbor = center + offset
    neighbor[..., 0] = neighbor[..., 0] % n1
    neighbor[..., 1] = neighbor[..., 1] % n2

    if should_flatten:
        neighbor = indices_m2v(neighbor, n2)
        neighbor = neighbor.reshape(-1, k)

    return neighbor


######################################################################
# others
######################################################################
# def count_params(model):
#     c = 0
#     for p in list(model.parameters()):
#         if p.numel() > 0:
#             c += reduce(operator.mul, list(p.size() + (2,) if p.is_complex() else p.size()))
#     return c


def count_params(model):
    c = 0
    for p in model.parameters():
        if p.numel() > 0:
            size = p.size()
            c += reduce(operator.mul, size, 1)
    return c


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
