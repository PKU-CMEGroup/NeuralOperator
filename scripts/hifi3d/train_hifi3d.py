from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


def _make_tensors(data: np.lib.npyio.NpzFile, indices: np.ndarray, model_name: str, node_weights_array: np.ndarray):
    node_mask = torch.from_numpy(data["node_mask"][indices].astype(np.float32))
    nodes = torch.from_numpy(data["nodes"][indices].astype(np.float32))
    node_weights = torch.from_numpy(node_weights_array[indices].astype(np.float32))
    features = torch.from_numpy(data["features"][indices].astype(np.float32))
    directed_edges = torch.from_numpy(data["directed_edges"][indices].astype(np.int64))
    edge_gradient_weights = torch.from_numpy(data["edge_gradient_weights"][indices].astype(np.float32))

    normals = features[..., :3]
    x = torch.cat([normals, nodes], dim=-1)
    y = features[..., -1:]
    if model_name == "mpcno":
        aux = (node_mask, nodes, node_weights, directed_edges, edge_gradient_weights, normals.permute(0, 2, 1))
    else:
        aux = (node_mask, nodes, node_weights, directed_edges, edge_gradient_weights)
    return x, y, aux


def _make_node_weights(data: np.lib.npyio.NpzFile, weight_mode: str, weight_factor: float) -> np.ndarray:
    if weight_mode == "normalized":
        return data["node_weights"]
    if weight_mode == "measure":
        node_measures = data["node_measures"]
        factor = weight_factor if weight_factor > 0 else float(np.max(np.sum(node_measures, axis=(1, 2))))
        if factor <= 0:
            raise ValueError("Cannot build measure weights from non-positive node measures")
        print(f"Using node_measures / factor with factor={factor}", flush=True)
        return node_measures / factor
    raise ValueError(f"Unsupported weight_mode: {weight_mode}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PCNO or M-PCNO on preprocessed HiFi 3D data.")
    parser.add_argument("--data_npz", type=Path, default=Path("data/hifi3d_processed/smoke_cell_centered.npz"))
    parser.add_argument("--names", type=Path, default=Path("data/hifi3d_processed/smoke_names.npy"))
    parser.add_argument("--model", type=str, default="mpcno", choices=["pcno", "mpcno"])
    parser.add_argument("--n_train", type=int, default=16)
    parser.add_argument("--n_test", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--k_max", type=int, default=4)
    parser.add_argument("--layer_sizes", type=str, default="16,16,16")
    parser.add_argument("--fc_dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5.0e-4)
    parser.add_argument("--grad", type=str, default="True", choices=["True", "False"])
    parser.add_argument("--geo", type=str, default="True", choices=["True", "False"])
    parser.add_argument("--geointegral", type=str, default="True", choices=["True", "False"])
    parser.add_argument("--weight_mode", type=str, default="normalized", choices=["normalized", "measure"])
    parser.add_argument("--weight_factor", type=float, default=0.0)
    parser.add_argument("--Ls", type=str, default="")
    parser.add_argument("--train_inv_L_scale", type=str, default="False", choices=["False", "together", "independently"])
    parser.add_argument("--save_model_name", type=str, default="")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data = np.load(args.data_npz)
    names = np.load(args.names, allow_pickle=True) if args.names.exists() else None
    ndata = data["nodes"].shape[0]
    if args.n_train + args.n_test > ndata:
        raise ValueError(f"Need n_train+n_test <= {ndata}, got {args.n_train + args.n_test}")

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(ndata)
    train_idx = order[: args.n_train]
    test_idx = order[args.n_train : args.n_train + args.n_test]

    if names is not None:
        print("Train datasets:", _count_datasets(names[train_idx]), flush=True)
        print("Test datasets:", _count_datasets(names[test_idx]), flush=True)

    node_weights_array = _make_node_weights(data, args.weight_mode, args.weight_factor)
    x_train, y_train, aux_train = _make_tensors(data, train_idx, args.model, node_weights_array)
    x_test, y_test, aux_test = _make_tensors(data, test_idx, args.model, node_weights_array)
    print(
        f"x_train={tuple(x_train.shape)} y_train={tuple(y_train.shape)} "
        f"x_test={tuple(x_test.shape)} y_test={tuple(y_test.shape)}",
        flush=True,
    )

    layers = [int(size) for size in args.layer_sizes.split(",") if size]
    ndim = 3
    if args.Ls:
        Ls = [float(value) for value in args.Ls.split(",")]
        if len(Ls) != ndim:
            raise ValueError(f"Expected {ndim} values in --Ls, got {Ls}")
    else:
        lengths = torch.amax(x_train[..., 3:6], dim=(0, 1)) - torch.amin(x_train[..., 3:6], dim=(0, 1))
        Ls = [max(2.0, float(length.item())) for length in lengths]
    print(f"Using Ls={Ls}, k_max={args.k_max}, layers={layers}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device={device}", flush=True)

    if args.model == "mpcno":
        from pcno.mpcno import MPCNO, MPCNO_train, compute_Fourier_modes

        modes = torch.tensor(compute_Fourier_modes(ndim, [args.k_max] * ndim, Ls), dtype=torch.float32).to(device)
        layer_selection = {
            "grad": args.grad.lower() == "true",
            "geo": args.geo.lower() == "true",
            "geointegral": args.geointegral.lower() == "true",
        }
        train_inv_L_scale = False if args.train_inv_L_scale == "False" else args.train_inv_L_scale
        model = MPCNO(
            ndim,
            modes,
            nmeasures=1,
            layer_selection=layer_selection,
            layers=layers,
            fc_dim=args.fc_dim,
            in_dim=x_train.shape[-1],
            out_dim=y_train.shape[-1],
            inv_L_scale_hyper=[train_inv_L_scale, 0.5, 2.0],
            act="gelu",
            geo_act="softsign",
            scaling_mode="sqrt_inv",
        ).to(device)
        train_fn = MPCNO_train
    else:
        from pcno.pcno import PCNO, PCNO_train, compute_Fourier_modes

        modes = torch.tensor(compute_Fourier_modes(ndim, [args.k_max] * ndim, Ls), dtype=torch.float32).to(device)
        model = PCNO(
            ndim,
            modes,
            nmeasures=1,
            layers=layers,
            fc_dim=args.fc_dim,
            in_dim=x_train.shape[-1],
            out_dim=y_train.shape[-1],
            inv_L_scale_hyper=["together", 0.5, 2.0],
            act="gelu",
        ).to(device)
        train_fn = PCNO_train

    config = {
        "train": {
            "base_lr": args.lr,
            "lr_ratio": 10,
            "weight_decay": 1.0e-4,
            "epochs": args.epochs,
            "scheduler": "OneCycleLR",
            "batch_size": args.batch_size,
            "normalization_x": False,
            "normalization_y": True,
            "normalization_dim_x": [],
            "normalization_dim_y": [],
            "non_normalized_dim_x": 4,
            "non_normalized_dim_y": 0,
        }
    }

    save_model_name = args.save_model_name if args.save_model_name else None
    train_rel_l2, test_rel_l2, test_l2 = train_fn(
        x_train,
        aux_train,
        y_train,
        x_test,
        aux_test,
        y_test,
        config,
        model,
        save_model_name=save_model_name,
    )
    print("Final train_rel_l2:", train_rel_l2[-1], flush=True)
    print("Final test_rel_l2:", test_rel_l2[-1], flush=True)
    print("Final test_l2:", test_l2[-1], flush=True)


def _count_datasets(names: np.ndarray) -> dict[str, int]:
    counts: dict[str, int] = {}
    for name in names:
        dataset = str(name).rsplit("-", 1)[0]
        counts[dataset] = counts.get(dataset, 0) + 1
    return counts


if __name__ == "__main__":
    main()
