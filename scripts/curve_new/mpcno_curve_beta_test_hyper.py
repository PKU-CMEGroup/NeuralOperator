import os
import sys
import argparse
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).parent.parent))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from pcno.mpcno_hyper_nograd import (
    compute_Fourier_modes,
    MPCNO_Beta,
    MPCNO_train_multidist_beta,
)


torch.set_printoptions(precision=16)
torch.manual_seed(0)
np.random.seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data_to_torch(data_file_path, to_divide=None, factor=1.0):
    data = np.load(data_file_path)
    nnodes = data["nnodes"]
    node_mask = data["node_mask"]
    nodes = data["nodes"]
    node_measures_raw = data["node_measures_raw"]

    print(f"Loaded {nodes.shape[0]} samples from {data_file_path}", flush=True)

    if to_divide is None:
        to_divide = factor * np.amax(np.sum(node_measures_raw, axis=1))
    node_weights = node_measures_raw / to_divide

    node_measures = data["node_measures"]
    directed_edges = data["directed_edges"]
    edge_gradient_weights = data["edge_gradient_weights"]
    features = data["features"]
    betas = data["betas"].astype(np.float32)

    indices = np.isfinite(node_measures_raw)
    node_rhos = np.copy(node_weights)
    node_rhos[indices] = node_rhos[indices] / node_measures[indices]

    nnodes = torch.from_numpy(nnodes)
    node_mask = torch.from_numpy(node_mask)
    nodes = torch.from_numpy(nodes.astype(np.float32))
    node_weights = torch.from_numpy(node_weights.astype(np.float32))
    node_rhos = torch.from_numpy(node_rhos.astype(np.float32))
    features = torch.from_numpy(features.astype(np.float32))
    directed_edges = torch.from_numpy(directed_edges.astype(np.int64))
    edge_gradient_weights = torch.from_numpy(edge_gradient_weights.astype(np.float32))
    betas = torch.from_numpy(betas)

    return (
        nnodes,
        node_mask,
        nodes,
        node_weights,
        node_rhos,
        features,
        directed_edges,
        edge_gradient_weights,
        betas,
        to_divide,
    )


def gen_data_tensors(
    data_indices,
    f_in_dim,
    f_out_dim,
    nodes,
    features,
    node_mask,
    node_weights,
    directed_edges,
    edge_gradient_weights,
    betas,
):
    # x does NOT concatenate beta; beta is fed as separate model input.
    nodes_input = nodes.clone()
    y = features[data_indices][..., -f_out_dim:]
    nx = features[data_indices][..., f_in_dim : f_in_dim + 2]
    aux = (
        node_mask[data_indices],
        nodes[data_indices],
        node_weights[data_indices],
        directed_edges[data_indices],
        edge_gradient_weights[data_indices],
        nx.permute(0, 2, 1),
    )
    beta_batch = betas[data_indices]# .unsqueeze(-1)  # [B,1]
    x = torch.cat(
        [
            features[data_indices][..., :f_in_dim],  # f
            features[data_indices][..., f_in_dim : f_in_dim + 2],  # normals
            nodes_input[data_indices],  # coords
        ],
        dim=-1,
    )
    return x, y, beta_batch, aux


def main():
    parser = argparse.ArgumentParser(description="Train MPCNO_Beta (lowrank) for curve_beta.")
    parser.add_argument("--kernel_type", type=str, default="sp_laplace")
    parser.add_argument("--geointegral", type=str, default="True", choices=["True", "False"])
    parser.add_argument("--to_divide_factor", type=float, default=20.0)
    parser.add_argument("--k_max", type=int, default=16)
    parser.add_argument("--bsz", type=int, default=32)
    parser.add_argument("--ep", type=int, default=500)
    parser.add_argument("--n_train", type=int, default=8000)
    parser.add_argument("--n_test", type=int, default=1000)
    parser.add_argument("--n_two_circles_test", type=int, default=0)
    parser.add_argument("--act", type=str, default="gelu")
    parser.add_argument("--layer_sizes", type=str, default="64,64,64,64,64,64")
    parser.add_argument("--beta_dim", type=int, default=1)
    parser.add_argument("--hyper_hidden", type=int, default=32)
    parser.add_argument("--rank", type=int, default=5)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--two_circles_data_path", type=str, default="")
    args = parser.parse_args()

    if args.n_two_circles_test > 0 and not args.two_circles_data_path:
        raise ValueError("n_two_circles_test > 0 requires --two_circles_data_path")

    f_in_dim, f_out_dim = 1, 1
    data_base_path = "../../data/curve_beta/"
    n_train, n_test, n_two = args.n_train, args.n_test, args.n_two_circles_test

    single_file = os.path.join(data_base_path, args.data_path)
    (
        _nnodes,
        node_mask,
        nodes,
        node_weights,
        _node_rhos,
        features,
        directed_edges,
        edge_gradient_weights,
        betas,
        to_divide,
    ) = load_data_to_torch(single_file, to_divide=None, factor=args.to_divide_factor)

    x_train, y_train, beta_train, aux_train = gen_data_tensors(
        np.arange(n_train),
        f_in_dim,
        f_out_dim,
        nodes,
        features,
        node_mask,
        node_weights,
        directed_edges,
        edge_gradient_weights,
        betas,
    )
    x_test, y_test, beta_test, aux_test = gen_data_tensors(
        np.arange(-n_test, 0),
        f_in_dim,
        f_out_dim,
        nodes,
        features,
        node_mask,
        node_weights,
        directed_edges,
        edge_gradient_weights,
        betas,
    )

    x_test_list, y_test_list, beta_test_list, aux_test_list = [x_test], [y_test], [beta_test], [aux_test]
    label_list = ["Single"]

    if n_two > 0:
        two_file = os.path.join(data_base_path, args.two_circles_data_path)
        (
            _nnodes2,
            node_mask2,
            nodes2,
            node_weights2,
            _node_rhos2,
            features2,
            directed_edges2,
            edge_gradient_weights2,
            betas2,
            _,
        ) = load_data_to_torch(two_file, to_divide=to_divide)

        x_two, y_two, beta_two, aux_two = gen_data_tensors(
            np.arange(n_two),
            f_in_dim,
            f_out_dim,
            nodes2,
            features2,
            node_mask2,
            node_weights2,
            directed_edges2,
            edge_gradient_weights2,
            betas2,
        )
        x_test_list.append(x_two)
        y_test_list.append(y_two)
        beta_test_list.append(beta_two)
        aux_test_list.append(aux_two)
        label_list.append("Two Curves")

    print(f"x_train {x_train.shape}, beta_train {beta_train.shape}, y_train {y_train.shape}", flush=True)
    print(f"x_test {[tuple(x.shape) for x in x_test_list]}", flush=True)

    layers = [int(size) for size in args.layer_sizes.split(",")]
    layer_selection = {
        "grad": False,  # as requested
        "geo": False,   # as requested
        "geointegral": args.geointegral.lower() == "true",
    }

    modes = compute_Fourier_modes(2, [args.k_max, args.k_max], [10, 10])
    modes = torch.tensor(modes, dtype=torch.float).to(device)

    model = MPCNO_Beta(
        ndims=2,
        modes=modes,
        nmeasures=1,
        layers=layers,
        beta_dim=args.beta_dim,
        hyper_hidden=args.hyper_hidden,
        rank=args.rank,
        layer_selection=layer_selection,
        fc_dim=128,
        in_dim=x_train.shape[-1],
        out_dim=y_train.shape[-1],
        inv_L_scale_hyper=[False, 0.5, 2.0],
        scaling_mode="inv",
        act=args.act,
    ).to(device)

    print("------Parameters------")
    print(f"layer_selection={layer_selection}")
    print(f"layers={layers}, beta_dim={args.beta_dim}, hyper_hidden={args.hyper_hidden}, rank={args.rank}")
    print(f"model params={sum(p.numel() for p in model.parameters()):,}")

    config = {
        "train": {
            "base_lr": 5e-4,
            "lr_ratio": 10,
            "weight_decay": 1e-4,
            "epochs": args.ep,
            "scheduler": "OneCycleLR",
            "batch_size": args.bsz,
            "normalization_x": False,
            "normalization_y": True,
            "normalization_dim_x": [],
            "normalization_dim_y": [],
            "non_normalized_dim_x": 4,
            "non_normalized_dim_y": 0,
        }
    }

    MPCNO_train_multidist_beta(
        x_train=x_train,
        aux_train=aux_train,
        y_train=y_train,
        beta_train=beta_train,
        x_test_list=x_test_list,
        aux_test_list=aux_test_list,
        y_test_list=y_test_list,
        beta_test_list=beta_test_list,
        config=config,
        model=model,
        label_test_list=label_list,
        save_model_name=None,
    )


if __name__ == "__main__":
    main()

