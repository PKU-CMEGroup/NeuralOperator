import gc
import torch
import numpy as np
from scipy.io import loadmat
import torch.utils
import torch.utils.data
from .normalizer import UnitGaussianNormalizer
from .basics import (
    compute_pca_bases,
    compute_fourier2d_bases,
    compute_id_bases,
    indices_neighbor2d,
)

from collections import defaultdict
import copy


def check_config(config):
    layer_channels = config["model"]["layer_channels"]
    should_residual = config["model"]["should_residual"]
    for in_channels, out_channels, should_res in zip(
        layer_channels, layer_channels[1:], should_residual
    ):
        if should_res:
            assert in_channels == out_channels, (
                f"residual connection requires in_channels ({in_channels}) "
                f"to be equal to out_channels ({out_channels})"
            )


###################################################################
# init Data and Bases
###################################################################
def init_darcy2d(config):

    # check_config(config)
    data_path1 = "../data/darcy_2d/piececonst_r421_N1024_smooth1"
    data_path2 = "../data/darcy_2d/piececonst_r421_N1024_smooth2"
    device = config["train"]["device"]
    should_flatten = config["data"]["should_flatten"]
    should_compute_bases = config["data"]["should_compute_bases"]

    # initialize data
    L = config["data"]["L"]
    downsample_ratio = config["data"]["downsample_ratio"]
    n_train = config["data"]["n_train"]
    n_test = config["data"]["n_test"]
    data1 = loadmat(data_path1)
    coeff1 = data1["coeff"]
    sol1 = data1["sol"]
    del data1
    data2 = loadmat(data_path2)
    coeff2 = data2["coeff"][:300, ...]
    sol2 = data2["sol"][:300, ...]
    del data2
    gc.collect()

    data_in = np.vstack((coeff1, coeff2))
    data_out = np.vstack((sol1, sol2))
    print(f"original data: in {data_in.shape} out {data_out.shape}", flush=True)

    Np_ref = data_in.shape[1]
    grid_1d = np.linspace(0, L, Np_ref)
    grid_x, grid_y = np.meshgrid(grid_1d, grid_1d)
    grid_x_ds = grid_x[0::downsample_ratio, 0::downsample_ratio]
    grid_y_ds = grid_y[0::downsample_ratio, 0::downsample_ratio]

    data_in_ds = data_in[0:n_train, 0::downsample_ratio, 0::downsample_ratio]
    data_out_ds = data_out[0:n_train, 0::downsample_ratio, 0::downsample_ratio]

    x_train = torch.from_numpy(
        np.stack(
            (
                data_in_ds,
                np.tile(grid_x_ds, (n_train, 1, 1)),
                np.tile(grid_y_ds, (n_train, 1, 1)),
            ),
            axis=-1,
        ).astype(np.float32)
    )
    y_train = torch.from_numpy(
        data_out_ds[:, :, :, np.newaxis].astype(np.float32))
    x_test = torch.from_numpy(
        np.stack(
            (
                data_in[-n_test:, 0::downsample_ratio, 0::downsample_ratio],
                np.tile(
                    grid_x[0::downsample_ratio,
                           0::downsample_ratio], (n_test, 1, 1)
                ),
                np.tile(
                    grid_y[0::downsample_ratio,
                           0::downsample_ratio], (n_test, 1, 1)
                ),
            ),
            axis=-1,
        ).astype(np.float32)
    )
    y_test = torch.from_numpy(
        data_out[-n_test:, 0::downsample_ratio, 0::downsample_ratio, np.newaxis].astype(
            np.float32
        )
    )

    # grid information
    grid_dict = {}
    grid_dict["L"] = L
    grid_dict["n1"] = x_train.shape[1]
    grid_dict["n2"] = x_train.shape[2]

    # compute bases
    bases_dict = {}
    if should_compute_bases:
        bases_req = list(
            set(config["model"]["wbases_types"]
                + config["model"]["bases_types"])
        )
        k_max = max(config["model"]["modes_gk"])
        Np = (Np_ref + downsample_ratio - 1) // downsample_ratio
        dX = L**2 / Np**2
        if "fourier" in bases_req:
            bases, wbases = compute_fourier2d_bases(
                Np, Np, k_max, L, L, should_flatten)
            bases_dict["fourier"] = [
                torch.from_numpy(bases.astype(np.float32)).to(device),
                torch.from_numpy(wbases.astype(np.float32)).to(device),
            ]
        if "pca_in" in bases_req:
            bases, wbases = compute_pca_bases(
                dX, k_max, data_in_ds, should_flatten)
            bases_dict["pca_in"] = [
                torch.from_numpy(bases.astype(np.float32)).to(device),
                torch.from_numpy(wbases.astype(np.float32)).to(device),
            ]
        if "pca_out" in bases_req:
            bases, wbases = compute_pca_bases(
                dX, k_max, data_out_ds, should_flatten)
            bases_dict["pca_out"] = [
                torch.from_numpy(bases.astype(np.float32)).to(device),
                torch.from_numpy(wbases.astype(np.float32)).to(device),
            ]
        if "id_out" in bases_req:
            bases, wbases = compute_id_bases(
                dX, k_max, data_out_ds, should_flatten)
            bases_dict["id_out"] = [
                torch.from_numpy(bases.astype(np.float32)).to(device),
                torch.from_numpy(wbases.astype(np.float32)).to(device),
            ]

    data_dict = {}
    if should_flatten:
        x_train = x_train.reshape(x_train.shape[0], -1, x_train.shape[-1])
        y_train = y_train.reshape(y_train.shape[0], -1, y_train.shape[-1])
        x_test = x_test.reshape(x_test.shape[0], -1, x_test.shape[-1])
        y_test = y_test.reshape(y_test.shape[0], -1, y_test.shape[-1])

    data_type = config["data"]["type"]
    if data_type == "transpose_in":
        x_train = x_train[..., 0].unsqueeze(-1)
        y_train = x_train.transpose(2, 1)
        x_test = x_test[..., 0].unsqueeze(-1)
        y_test = x_test.transpose(2, 1)
    elif data_type == "transpose_out":
        x_train = y_train.transpose(2, 1)
        x_test = y_test.transpose(2, 1)

    print(
        f"used data: x_train {tuple(x_train.shape)} y_train {tuple(y_train.shape)} ",
        f"x_test {tuple(x_test.shape)} y_test {tuple(y_test.shape)}",
        flush=True,
    )
    data_dict["xtrain"] = x_train
    data_dict["ytrain"] = y_train
    data_dict["xtest"] = x_test
    data_dict["ytest"] = y_test

    return data_dict, bases_dict, grid_dict


def init_airfoil(config):

    data_path = "../data/airfoil/"
    device = torch.device(config["train"]["device"])
    should_flatten = config["data"]["should_flatten"]
    should_compute_bases = config["data"]["should_compute_bases"]

    # initialize data
    L = config["data"]["L"]
    downsample_ratio = config["data"]["downsample_ratio"]
    n_train = config["data"]["n_train"]
    n_test = config["data"]["n_test"]
    coordx = np.load(data_path + "NACA_Cylinder_X.npy")
    coordy = np.load(data_path + "NACA_Cylinder_Y.npy")
    data_in = np.stack((coordx, coordy), axis=3)
    data_out = np.load(data_path + "NACA_Cylinder_Q.npy")[
        :, 4, :, :
    ]  # density, velocity 2d, pressure, mach number
    print(f"original data: in {data_in.shape} out {data_out.shape}", flush=True)

    _, nx, ny, _ = data_in.shape

    data_in_ds = data_in[:, 0::downsample_ratio, 0::downsample_ratio, :]
    data_out_ds = data_out[:, 0::downsample_ratio,
                           0::downsample_ratio, np.newaxis]

    L = 1.0
    grid_x, grid_y = np.meshgrid(np.linspace(0, L, nx), np.linspace(0, L, ny))
    grid_x, grid_y = grid_x.T, grid_y.T
    grid_x_ds = grid_x[0::downsample_ratio, 0::downsample_ratio]
    grid_y_ds = grid_y[0::downsample_ratio, 0::downsample_ratio]

    weights = np.ones(grid_x_ds.shape) / \
        (grid_x_ds.shape[0] * grid_x_ds.shape[1])
    mask = np.ones(grid_x_ds.shape)

    should_aux = config["data"]["should_aux"]
    x_train = np.concatenate(
        (
            data_in_ds[0:n_train, :, :, :],
            np.tile(grid_x_ds, (n_train, 1, 1))[:, :, :, np.newaxis],
            np.tile(grid_y_ds, (n_train, 1, 1))[:, :, :, np.newaxis],
        ),
        axis=3,
    )
    x_test = np.concatenate(
        (
            data_in_ds[-n_test:, :, :, :],
            np.tile(grid_x_ds, (n_test, 1, 1))[:, :, :, np.newaxis],
            np.tile(grid_y_ds, (n_test, 1, 1))[:, :, :, np.newaxis],
        ),
        axis=3,
    )
    if should_aux:
        x_train = np.concatenate(
            (
                x_train,
                np.tile(weights, (n_train, 1, 1))[:, :, :, np.newaxis],
                np.tile(mask, (n_train, 1, 1))[:, :, :, np.newaxis],
            ),
            axis=3,
        )
        x_test = np.concatenate(
            (
                x_test,
                np.tile(weights, (n_test, 1, 1))[:, :, :, np.newaxis],
                np.tile(mask, (n_test, 1, 1))[:, :, :, np.newaxis],
            ),
            axis=3,
        )
    x_train = torch.from_numpy(x_train.astype(np.float32))
    x_test = torch.from_numpy(x_test.astype(np.float32))
    y_train = torch.from_numpy(
        data_out_ds[0:n_train, :, :, :].astype(np.float32))
    y_test = torch.from_numpy(data_out_ds[-n_test:, :, :, :].astype(np.float32))

    # grid information
    grid_dict = {}
    grid_dict["L"] = L
    grid_dict["n1"] = x_train.shape[1]
    grid_dict["n2"] = x_train.shape[2]

    # compute bases
    bases_dict = {}
    if should_compute_bases:
        bases_req = set(
            config["model"]["wbases_types"] + config["model"]["bases_types"]
        )

        k_max = max(config["model"]["modes_gk"])
        Nx = (nx + downsample_ratio - 1) // downsample_ratio
        Ny = (ny + downsample_ratio - 1) // downsample_ratio
        dX = L * L / (Nx * Ny)

        if "fourier" in bases_req:
            bases, wbases = compute_fourier2d_bases(
                Nx, Ny, k_max, L, L, should_flatten)
            bases_dict["fourier"] = [
                torch.from_numpy(bases.astype(np.float32)).to(device),
                torch.from_numpy(wbases.astype(np.float32)).to(device),
            ]
        if "pca_in_x" in bases_req:
            data_pca = data_in_ds[..., 0]
            bases, wbases = compute_pca_bases(
                dX, k_max, data_pca, should_flatten)
            bases_dict["pca_in_x"] = [
                torch.from_numpy(bases.astype(np.float32)).to(device),
                torch.from_numpy(wbases.astype(np.float32)).to(device),
            ]
        if "pca_in_y" in bases_req:
            data_pca = data_in_ds[..., 1]
            bases, wbases = compute_pca_bases(
                dX, k_max, data_pca, should_flatten)
            bases_dict["pca_in_y"] = [
                torch.from_numpy(bases.astype(np.float32)).to(device),
                torch.from_numpy(wbases.astype(np.float32)).to(device),
            ]
        if "pca_out" in bases_req:
            data_pca = data_out_ds[0:n_train, ...]
            bases, wbases = compute_pca_bases(
                dX, k_max, data_pca, should_flatten)
            bases_dict["pca_out"] = [
                torch.from_numpy(bases.astype(np.float32)).to(device),
                torch.from_numpy(wbases.astype(np.float32)).to(device),
            ]
        if "id_out" in bases_req:
            bases, wbases = compute_id_bases(
                dX, k_max, data_out_ds, should_flatten)
            bases_dict["id_out"] = [
                torch.from_numpy(bases.astype(np.float32)).to(device),
                torch.from_numpy(wbases.astype(np.float32)).to(device),
            ]

    data_dict = {}
    if should_flatten:
        x_train = x_train.reshape(x_train.shape[0], -1, x_train.shape[-1])
        y_train = y_train.reshape(y_train.shape[0], -1, y_train.shape[-1])
        x_test = x_test.reshape(x_test.shape[0], -1, x_test.shape[-1])
        y_test = y_test.reshape(y_test.shape[0], -1, y_test.shape[-1])

    print(
        f"used data: x_train {tuple(x_train.shape)} y_train {tuple(y_train.shape)} ",
        f"x_test {tuple(x_test.shape)} y_test {tuple(y_test.shape)}",
        flush=True,
    )
    data_dict["xtrain"] = x_train
    data_dict["ytrain"] = y_train
    data_dict["xtest"] = x_test
    data_dict["ytest"] = y_test

    return data_dict, bases_dict, grid_dict


def init_darcy2d_graph(config):
    import dgl
    # check_config(config)
    data_path1 = "../data/darcy_2d/piececonst_r421_N1024_smooth1"
    data_path2 = "../data/darcy_2d/piececonst_r421_N1024_smooth2"
    device = config["train"]["device"]
    should_flatten = config["data"]["should_flatten"]
    should_compute_bases = config["data"]["should_compute_bases"]

    # initialize data
    L = config["data"]["L"]
    downsample_ratio = config["data"]["downsample_ratio"]
    n_train = config["data"]["n_train"]
    n_test = config["data"]["n_test"]
    data1 = loadmat(data_path1)
    coeff1 = data1["coeff"]
    sol1 = data1["sol"]
    del data1
    data2 = loadmat(data_path2)
    coeff2 = data2["coeff"][:300, ...]
    sol2 = data2["sol"][:300, ...]
    del data2
    gc.collect()

    data_in = np.vstack((coeff1, coeff2))
    data_out = np.vstack((sol1, sol2))
    print(f"original data: in {data_in.shape} out {data_out.shape}", flush=True)

    Np_ref = data_in.shape[1]
    grid_1d = np.linspace(0, L, Np_ref)
    grid_x, grid_y = np.meshgrid(grid_1d, grid_1d)
    grid_x_ds = grid_x[0::downsample_ratio, 0::downsample_ratio]
    grid_y_ds = grid_y[0::downsample_ratio, 0::downsample_ratio]

    data_in_ds = data_in[0:n_train, 0::downsample_ratio, 0::downsample_ratio]
    data_out_ds = data_out[0:n_train, 0::downsample_ratio, 0::downsample_ratio]

    x_train = torch.from_numpy(
        np.stack(
            (
                data_in_ds,
                np.tile(grid_x_ds, (n_train, 1, 1)),
                np.tile(grid_y_ds, (n_train, 1, 1)),
            ),
            axis=-1,
        ).astype(np.float32)
    )
    y_train = torch.from_numpy(
        data_out_ds[:, :, :, np.newaxis].astype(np.float32))
    x_test = torch.from_numpy(
        np.stack(
            (
                data_in[-n_test:, 0::downsample_ratio, 0::downsample_ratio],
                np.tile(
                    grid_x[0::downsample_ratio,
                           0::downsample_ratio], (n_test, 1, 1)
                ),
                np.tile(
                    grid_y[0::downsample_ratio,
                           0::downsample_ratio], (n_test, 1, 1)
                ),
            ),
            axis=-1,
        ).astype(np.float32)
    )
    y_test = torch.from_numpy(
        data_out[-n_test:, 0::downsample_ratio, 0::downsample_ratio, np.newaxis].astype(
            np.float32
        )
    )

    # grid information
    grid_dict = {}
    grid_dict["L"] = L
    grid_dict["n1"] = x_train.shape[1]
    grid_dict["n2"] = x_train.shape[2]

    # compute bases
    bases_dict = {}
    Np = (Np_ref + downsample_ratio - 1) // downsample_ratio
    dX = L**2 / Np**2
    if should_compute_bases:
        bases_req = list(
            set(config["model"]["wbases_types"]
                + config["model"]["bases_types"])
        )
        k_max = max(config["model"]["modes_gk"])
        if "fourier" in bases_req:
            bases, wbases = compute_fourier2d_bases(
                Np, Np, k_max, L, L, should_flatten)
            bases_dict["fourier"] = [
                torch.from_numpy(bases.astype(np.float32)).to(device),
                torch.from_numpy(wbases.astype(np.float32)).to(device),
            ]
        if "pca_in" in bases_req:
            bases, wbases = compute_pca_bases(
                dX, k_max, data_in_ds, should_flatten)
            bases_dict["pca_in"] = [
                torch.from_numpy(bases.astype(np.float32)).to(device),
                torch.from_numpy(wbases.astype(np.float32)).to(device),
            ]
        if "pca_out" in bases_req:
            bases, wbases = compute_pca_bases(
                dX, k_max, data_out_ds, should_flatten)
            bases_dict["pca_out"] = [
                torch.from_numpy(bases.astype(np.float32)).to(device),
                torch.from_numpy(wbases.astype(np.float32)).to(device),
            ]
        if "id_out" in bases_req:
            bases, wbases = compute_id_bases(
                dX, k_max, data_out_ds, should_flatten)
            bases_dict["id_out"] = [
                torch.from_numpy(bases.astype(np.float32)).to(device),
                torch.from_numpy(wbases.astype(np.float32)).to(device),
            ]

    if should_flatten:
        x_train = x_train.reshape(x_train.shape[0], -1, x_train.shape[-1])
        y_train = y_train.reshape(y_train.shape[0], -1, y_train.shape[-1])
        x_test = x_test.reshape(x_test.shape[0], -1, x_test.shape[-1])
        y_test = y_test.reshape(y_test.shape[0], -1, y_test.shape[-1])

    data_type = config["data"]["type"]
    if data_type == "transpose_in":
        x_train = x_train[..., 0].unsqueeze(-1)
        y_train = x_train.transpose(2, 1)
        x_test = x_test[..., 0].unsqueeze(-1)
        y_test = x_test.transpose(2, 1)
    elif data_type == "transpose_out":
        x_train = y_train.transpose(2, 1)
        x_test = y_test.transpose(2, 1)

    normalization_x, normalization_y, normalization_dim = (
        config["train"]["normalization_x"],
        config["train"]["normalization_y"],
        config["train"]["normalization_dim"],
    )
    if normalization_x:
        x_normalizer = UnitGaussianNormalizer(x_train, dim=normalization_dim)
        x_train = x_normalizer.encode(x_train)
        x_test = x_normalizer.encode(x_test)
        x_normalizer.to(device)
    if normalization_y:
        y_normalizer = UnitGaussianNormalizer(y_train, dim=normalization_dim)
        y_train = y_normalizer.encode(y_train)
        y_test = y_normalizer.encode(y_test)
        y_normalizer.to(device)

    print(
        f"used data: x_train {tuple(x_train.shape)} y_train {tuple(y_train.shape)} ",
        f"x_test {tuple(x_test.shape)} y_test {tuple(y_test.shape)}",
        flush=True,
    )

    k = config["data"]["neighbor_size"]
    indices_src = np.repeat(np.arange(Np * Np), (2 * k + 1) * (2 * k + 1))
    indices_dst = indices_neighbor2d(
        Np, Np, k, k, should_flatten=True).reshape(-1)
    G0 = dgl.graph((indices_src, indices_dst))

    # data_dict = {}
    # data_dict["xtrain"] = x_train
    # data_dict["ytrain"] = y_train
    # data_dict["xtest"] = x_test
    # data_dict["ytest"] = y_test
    should_include_grid = config["data"]["should_include_grid"]
    dataset_train = GraphDateset(x_train, y_train, G0, should_include_grid)
    dataset_test = GraphDateset(x_test, y_test, G0, should_include_grid)

    def graph_collate(samples):
        graphs, x, y = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        return batched_graph, torch.stack(x), torch.stack(y)

    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=config["train"]["batch_size"],
        collate_fn=graph_collate,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=config["train"]["batch_size"],
        collate_fn=graph_collate,
        shuffle=False,
    )

    return train_loader, test_loader, y_normalizer, G0


def init_airfoil_graph(config):
    import dgl
    data_path = "../data/airfoil/"
    device = torch.device(config["train"]["device"])
    should_flatten = config["data"]["should_flatten"]
    should_compute_bases = config["data"]["should_compute_bases"]

    # initialize data
    L = config["data"]["L"]
    downsample_ratio = config["data"]["downsample_ratio"]
    n_train = config["data"]["n_train"]
    n_test = config["data"]["n_test"]
    coordx = np.load(data_path + "NACA_Cylinder_X.npy")
    coordy = np.load(data_path + "NACA_Cylinder_Y.npy")
    data_in = np.stack((coordx, coordy), axis=3)
    data_out = np.load(data_path + "NACA_Cylinder_Q.npy")[
        :, 4, :, :
    ]  # density, velocity 2d, pressure, mach number
    print(f"original data: in {data_in.shape} out {data_out.shape}", flush=True)

    _, nx, ny, _ = data_in.shape

    data_in_ds = data_in[:, 0::downsample_ratio, 0::downsample_ratio, :]
    data_out_ds = data_out[:, 0::downsample_ratio,
                           0::downsample_ratio, np.newaxis]

    L = 1.0
    grid_x, grid_y = np.meshgrid(np.linspace(0, L, nx), np.linspace(0, L, ny))
    grid_x, grid_y = grid_x.T, grid_y.T
    grid_x_ds = grid_x[0::downsample_ratio, 0::downsample_ratio]
    grid_y_ds = grid_y[0::downsample_ratio, 0::downsample_ratio]

    x_train = torch.from_numpy(
        np.concatenate(
            (
                data_in_ds[0:n_train, :, :, :],
                np.tile(grid_x_ds, (n_train, 1, 1))[:, :, :, np.newaxis],
                np.tile(grid_y_ds, (n_train, 1, 1))[:, :, :, np.newaxis],
            ),
            axis=3,
        ).astype(np.float32)
    )
    y_train = torch.from_numpy(
        data_out_ds[0:n_train, :, :, :].astype(np.float32))
    x_test = torch.from_numpy(
        np.concatenate(
            (
                data_in_ds[-n_test:, :, :, :],
                np.tile(grid_x_ds, (n_test, 1, 1))[:, :, :, np.newaxis],
                np.tile(grid_y_ds, (n_test, 1, 1))[:, :, :, np.newaxis],
            ),
            axis=3,
        ).astype(np.float32)
    )
    y_test = torch.from_numpy(data_out_ds[-n_test:, :, :, :].astype(np.float32))

    # grid information
    grid_dict = {}
    grid_dict["L"] = L
    grid_dict["n1"] = x_train.shape[1]
    grid_dict["n2"] = x_train.shape[2]

    # compute bases
    bases_dict = {}
    Nx = (nx + downsample_ratio - 1) // downsample_ratio
    Ny = (ny + downsample_ratio - 1) // downsample_ratio
    if should_compute_bases:
        bases_req = set(
            config["model"]["wbases_types"] + config["model"]["bases_types"]
        )

        k_max = max(config["model"]["modes_gk"])
        dX = L * L / (Nx * Ny)

        if "fourier" in bases_req:
            bases, wbases = compute_fourier2d_bases(
                Nx, Ny, k_max, L, L, should_flatten)
            bases_dict["fourier"] = [
                torch.from_numpy(bases.astype(np.float32)).to(device),
                torch.from_numpy(wbases.astype(np.float32)).to(device),
            ]
        if "pca_in_x" in bases_req:
            data_pca = data_in_ds[..., 0]
            bases, wbases = compute_pca_bases(
                dX, k_max, data_pca, should_flatten)
            bases_dict["pca_in_x"] = [
                torch.from_numpy(bases.astype(np.float32)).to(device),
                torch.from_numpy(wbases.astype(np.float32)).to(device),
            ]
        if "pca_in_y" in bases_req:
            data_pca = data_in_ds[..., 1]
            bases, wbases = compute_pca_bases(
                dX, k_max, data_pca, should_flatten)
            bases_dict["pca_in_y"] = [
                torch.from_numpy(bases.astype(np.float32)).to(device),
                torch.from_numpy(wbases.astype(np.float32)).to(device),
            ]
        if "pca_out" in bases_req:
            data_pca = data_out_ds[0:n_train, ...]
            bases, wbases = compute_pca_bases(
                dX, k_max, data_pca, should_flatten)
            bases_dict["pca_out"] = [
                torch.from_numpy(bases.astype(np.float32)).to(device),
                torch.from_numpy(wbases.astype(np.float32)).to(device),
            ]
        if "id_out" in bases_req:
            bases, wbases = compute_id_bases(
                dX, k_max, data_out_ds, should_flatten)
            bases_dict["id_out"] = [
                torch.from_numpy(bases.astype(np.float32)).to(device),
                torch.from_numpy(wbases.astype(np.float32)).to(device),
            ]

    data_dict = {}
    if should_flatten:
        x_train = x_train.reshape(x_train.shape[0], -1, x_train.shape[-1])
        y_train = y_train.reshape(y_train.shape[0], -1, y_train.shape[-1])
        x_test = x_test.reshape(x_test.shape[0], -1, x_test.shape[-1])
        y_test = y_test.reshape(y_test.shape[0], -1, y_test.shape[-1])

    print(
        f"used data: x_train {tuple(x_train.shape)} y_train {tuple(y_train.shape)} ",
        f"x_test {tuple(x_test.shape)} y_test {tuple(y_test.shape)}",
        flush=True,
    )
    normalization_x, normalization_y, normalization_dim = (
        config["train"]["normalization_x"],
        config["train"]["normalization_y"],
        config["train"]["normalization_dim"],
    )
    if normalization_x:
        x_normalizer = UnitGaussianNormalizer(x_train, dim=normalization_dim)
        x_train = x_normalizer.encode(x_train)
        x_test = x_normalizer.encode(x_test)
        x_normalizer.to(device)
    if normalization_y:
        y_normalizer = UnitGaussianNormalizer(y_train, dim=normalization_dim)
        y_train = y_normalizer.encode(y_train)
        y_test = y_normalizer.encode(y_test)
        y_normalizer.to(device)

    k = config["data"]["neighbor_size"]
    indices_src = np.repeat(np.arange(Nx * Ny), (2 * k + 1) * (2 * k + 1))
    indices_dst = indices_neighbor2d(
        Nx, Ny, k, k, should_flatten=True).reshape(-1)
    G0 = dgl.graph((indices_src, indices_dst))

    should_include_grid = config["data"]["should_include_grid"]
    dataset_train = GraphDateset(x_train, y_train, G0, should_include_grid)
    dataset_test = GraphDateset(x_test, y_test, G0, should_include_grid)

    def graph_collate(samples):
        graphs, x, y = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        return batched_graph, torch.stack(x), torch.stack(y)

    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=config["train"]["batch_size"],
        collate_fn=graph_collate,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=config["train"]["batch_size"],
        collate_fn=graph_collate,
        shuffle=False,
    )

    return train_loader, test_loader, y_normalizer, G0


###################################################################
# Dataset Classes
###################################################################


class GraphDateset(torch.utils.data.Dataset):
    def __init__(self, x, y, G0, should_include_grid):

        self.x = x
        self.y = y
        self.G0 = G0
        self.n = x.shape[0]
        self.should_include_grid = should_include_grid

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        G = copy.deepcopy(self.G0)
        return G, self.x[index], self.y[index]
