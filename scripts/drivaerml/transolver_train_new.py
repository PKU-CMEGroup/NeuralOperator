import os
import torch
import sys
import argparse

from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from timeit import default_timer
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from pcno.geo_utility import preprocess_data_mesh, compute_node_weights
from pcno.mpcno import compute_Fourier_modes, MPCNO, MPCNO_train, MPCNO_train_multidist
torch.set_printoptions(precision=16)

from baselines.transolver_plus import Model  # noqa: E402
from scripts.drivaerml.train_utils import (  # noqa: E402
    _make_node_weights,
    train_baseline
)


torch.manual_seed(0)
np.random.seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

###################################
# load parameters
###################################

parser = argparse.ArgumentParser(description="Train Transolver++ on preprocessed HiFi3D point data.")
parser.add_argument("--data_npz", type=Path, required=True)
parser.add_argument("--n_train", type=int, default=16)
parser.add_argument("--n_test", type=int, default=4)
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--k_max", type=int, default=4)
parser.add_argument("--layer_sizes", type=str, default="64,64,64,64")
parser.add_argument("--fc_dim", type=int, default=128)
parser.add_argument("--lr", type=float, default=5.0e-4)
parser.add_argument("--weight_decay", type=float, default=1.0e-4)
parser.add_argument("--weight_mode", type=str, default="measure", choices=["normalized", "measure"])
parser.add_argument("--weight_factor", type=float, default=0.0)
parser.add_argument("--Ls", type=str, default="")
parser.add_argument("--save_model_name", type=str, default="")
parser.add_argument("--normalization_y", type=str, default="True", choices=["True", "False"])
parser.add_argument("--scheduler_step", type=str, default="batch", choices=["epoch", "batch"])
parser.add_argument("--max_nodes", type=int, default=0)
parser.add_argument("--transolver_nhead", type=int, default=8)
parser.add_argument("--transolver_slice_num", type=int, default=32)
parser.add_argument("--transolver_dropout", type=float, default=0.0)
parser.add_argument("--transolver_mlp_ratio", type=int, default=2)
parser.add_argument("--transolver_ref", type=int, default=8)
args = parser.parse_args()

n_train = args.n_train
n_test = args.n_test


def gen_data_tensors(
    data: np.lib.npyio.NpzFile,
    indices: np.ndarray,
    node_weights: np.ndarray,
    max_nodes: int,
):
    node_slice = slice(None if max_nodes <= 0 else max_nodes)
    node_mask = torch.from_numpy(data["node_mask"][indices, node_slice].astype(np.float32))
    nodes = torch.from_numpy(data["nodes"][indices, node_slice].astype(np.float32))
    node_weights = torch.from_numpy(node_weights[indices, node_slice].astype(np.float32))
    features = torch.from_numpy(data["features"][indices, node_slice].astype(np.float32))
    normals = features[..., :3]
    y = features[..., -1:] * node_mask
    x = torch.cat([nodes, normals], dim=-1)
    x = x * node_mask
    condition = torch.empty((len(indices), 0), dtype=torch.float32)
    aux = (node_mask, nodes, node_weights, condition)
    return x, y, aux


data = np.load(args.data_npz)
ndata = data["nodes"].shape[0]
model_name = "transolver"

rng = np.random.default_rng(0)
order = rng.permutation(ndata)
train_idx = order[:n_train]
test_idx = order[n_train:n_train + n_test]

node_weights_array = _make_node_weights(data, args.weight_mode, args.weight_factor)
x_train, y_train, aux_train = gen_data_tensors(data, train_idx, node_weights_array, args.max_nodes)
x_test, y_test, aux_test = gen_data_tensors(data, test_idx, node_weights_array, args.max_nodes)

layers = [int(size) for size in args.layer_sizes.split(",") if size]
ndim = 3
if args.Ls:
    Ls = [float(value) for value in args.Ls.split(",")]
    if len(Ls) != ndim:
        raise ValueError(f"Expected {ndim} values in --Ls, got {Ls}")
else:
    lengths = torch.amax(aux_train[1], dim=(0, 1)) - torch.amin(aux_train[1], dim=(0, 1))
    Ls = [float(length.item())*2+0.2 for length in lengths]

print(f"Using model= Transolver++", flush=True)
print(f"Using Ls={Ls}, k_max={args.k_max}, layers={layers}", flush=True)
print(f"Using device={device}", flush=True)
print(
        f"x_train={tuple(x_train.shape)} y_train={tuple(y_train.shape)} "
        f"x_test={tuple(x_test.shape)} y_test={tuple(y_test.shape)}",
        flush=True,
)

model = Model(
        space_dim=3,
        fun_dim=x_train.shape[-1] - 3,
        out_dim=y_train.shape[-1],
        n_layers=len(layers),
        n_hidden=layers[0],
        n_head=args.transolver_nhead,
        dropout=args.transolver_dropout,
        mlp_ratio=args.transolver_mlp_ratio,
        slice_num=args.transolver_slice_num,
        ref=args.transolver_ref,
        unified_pos=False,
    ).to(device)

config = {
        "base_lr": args.lr,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "normalization_y": args.normalization_y.lower() == "true",
        "scheduler_step": args.scheduler_step,
}

save_model_name = args.save_model_name if args.save_model_name else None
train_rel_l2, test_rel_l2, test_l2 = train_baseline(
        model_name,
        x_train,
        aux_train,
        y_train,
        [(None, x_test, aux_test, y_test)],
        config,
        model,
        save_model_name=save_model_name,
)