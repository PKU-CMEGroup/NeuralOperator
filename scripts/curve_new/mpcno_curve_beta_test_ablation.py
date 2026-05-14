import os
import sys
import argparse
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).parent.parent))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from utility.losses import LpLoss
from pcno.mpcno_structured import compute_Fourier_modes, MPCNO as MPCNO_Structured, MPCNO_train_multidist


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
    # Keep normals in x, as requested:
    # x = [f, beta, normal_x, normal_y, coord_x, coord_y]
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

    beta_batch = betas[data_indices].unsqueeze(-1)
    m_nodes = features.shape[1]
    beta_expanded = beta_batch.unsqueeze(1).expand(-1, m_nodes, -1)

    x = torch.cat(
        [
            features[data_indices][..., :f_in_dim],  # f
            beta_expanded,  # beta
            features[data_indices][..., f_in_dim : f_in_dim + 2],  # normals
            nodes_input[data_indices],  # coordinates
        ],
        dim=-1,
    )
    return x, y, aux


@torch.no_grad()
def evaluate_multidist(model, x_list, y_list, aux_list, labels, batch_size=32, use_ablation=False):
    myloss = LpLoss(d=1, p=2, size_average=False)
    model.eval()
    rel_dict = {}
    abs_dict = {}

    for name, x_test, y_test, aux_test in zip(labels, x_list, y_list, aux_list):
        node_mask, nodes, node_weights, directed_edges, edge_gradient_weights, geo = aux_test
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                x_test,
                y_test,
                node_mask,
                nodes,
                node_weights,
                directed_edges,
                edge_gradient_weights,
                geo,
            ),
            batch_size=batch_size,
            shuffle=False,
        )

        t_rel_l2 = 0.0
        t_l2 = 0.0
        for (
            x,
            y,
            node_mask_b,
            nodes_b,
            node_weights_b,
            directed_edges_b,
            edge_gradient_weights_b,
            geo_b,
        ) in loader:
            x = x.to(device)
            y = y.to(device)
            node_mask_b = node_mask_b.to(device)
            nodes_b = nodes_b.to(device)
            node_weights_b = node_weights_b.to(device)
            directed_edges_b = directed_edges_b.to(device)
            edge_gradient_weights_b = edge_gradient_weights_b.to(device)
            geo_b = geo_b.to(device)

            if use_ablation:
                out = model.forward_ablation(
                    x,
                    (node_mask_b, nodes_b, node_weights_b, directed_edges_b, edge_gradient_weights_b, geo_b),
                )
            else:
                out = model(
                    x,
                    (node_mask_b, nodes_b, node_weights_b, directed_edges_b, edge_gradient_weights_b, geo_b),
                )
            out = out * node_mask_b
            batch_size_ = x.shape[0]
            t_rel_l2 += myloss(out.view(batch_size_, -1), y.view(batch_size_, -1)).item()
            t_l2 += myloss.abs(out.view(batch_size_, -1), y.view(batch_size_, -1)).item()

        rel_dict[name] = t_rel_l2 / len(loader.dataset)
        abs_dict[name] = t_l2 / len(loader.dataset)

    return rel_dict, abs_dict


def main():
    parser = argparse.ArgumentParser(
        description="Train mpcno_structured, then evaluate with forward_ablation (K+W only)."
    )
    parser.add_argument("--kernel_type", type=str, default="sp_laplace", choices=["sp_laplace", "dp_laplace"])
    parser.add_argument("--geo", type=str, default="True", choices=["True", "False"])
    parser.add_argument("--geointegral", type=str, default="True", choices=["True", "False"])
    parser.add_argument("--to_divide_factor", type=float, default=20.0)
    parser.add_argument("--k_max", type=int, default=16)
    parser.add_argument("--bsz", type=int, default=32)
    parser.add_argument("--ep", type=int, default=500)
    parser.add_argument("--n_train", type=int, default=2000)
    parser.add_argument("--n_test", type=int, default=1000)
    parser.add_argument("--n_two_circles_test", type=int, default=0)
    parser.add_argument("--act", type=str, default="gelu")
    parser.add_argument("--geo_act", type=str, default="softsign")
    parser.add_argument("--proj_act", type=str, default="gelu")
    parser.add_argument("--layer_sizes", type=str, default="64,64,64,64,64,64")
    parser.add_argument("--proj_layer_sizes", type=str, default="128,128,128")
    parser.add_argument("--fc_dim", type=int, default=128)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--two_circles_data_path", type=str, default="")
    parser.add_argument(
        "--report_full_test",
        type=str,
        default="True",
        choices=["True", "False"],
        help="Whether to evaluate full model before dropping to K+W mode.",
    )
    args = parser.parse_args()

    if args.n_two_circles_test > 0 and not args.two_circles_data_path:
        raise ValueError("n_two_circles_test > 0 requires --two_circles_data_path")

    f_in_dim = 1
    f_out_dim = 1
    data_base_path = "../../data/curve_beta/"

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

    x_train, y_train, aux_train = gen_data_tensors(
        np.arange(args.n_train),
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
    x_test, y_test, aux_test = gen_data_tensors(
        np.arange(-args.n_test, 0),
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

    x_test_list = [x_test]
    y_test_list = [y_test]
    aux_test_list = [aux_test]
    label_list = ["Single"]

    if args.n_two_circles_test > 0:
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

        x_two, y_two, aux_two = gen_data_tensors(
            np.arange(args.n_two_circles_test),
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
        aux_test_list.append(aux_two)
        label_list.append("Two Curves")

    print(f"x_train {x_train.shape}, y_train {y_train.shape}")
    print(f"x_test shapes {[tuple(x.shape) for x in x_test_list]}")
    print("Domain range per dimension:", torch.amax(nodes, dim=[0, 1]) - torch.amin(nodes, dim=[0, 1]))

    layer_selection = {
        "grad": True,
        "geo": args.geo.lower() == "true",
        "geointegral": args.geointegral.lower() == "true",
    }
    layers = [int(size) for size in args.layer_sizes.split(",")]
    proj_layers = [int(size) for size in args.proj_layer_sizes.split(",") if int(size) > 0]
    modes = compute_Fourier_modes(2, [args.k_max, args.k_max], [10, 10])
    modes = torch.tensor(modes, dtype=torch.float).to(device)

    model = MPCNO_Structured(
        ndims=2,
        modes=modes,
        nmeasures=1,
        layer_selection=layer_selection,
        layers=layers,
        fc_dim=args.fc_dim,
        proj_layers=proj_layers,
        in_dim=x_train.shape[-1],
        out_dim=y_train.shape[-1],
        inv_L_scale_hyper=[False, 0.5, 2.0],
        scaling_mode="inv",
        act=args.act,
        geo_act=args.geo_act,
        proj_act=args.proj_act,
    ).to(device)

    print("\n------Training Setup------")
    print(f"train layer_selection = {layer_selection}")
    print(f"proj_layers = {proj_layers}, proj_act = {args.proj_act}")
    print(f"total parameters = {sum(p.numel() for p in model.parameters()):,}")
    print(f"device = {device}")

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

    MPCNO_train_multidist(
        x_train,
        aux_train,
        y_train,
        x_test_list,
        aux_test_list,
        y_test_list,
        config,
        model,
        label_test_list=label_list,
        save_model_name=None,
    )

    print("\n------Post-Training Evaluation------")
    if args.report_full_test.lower() == "true":
        rel_full, abs_full = evaluate_multidist(
            model=model,
            x_list=x_test_list,
            y_list=y_test_list,
            aux_list=aux_test_list,
            labels=label_list,
            batch_size=args.bsz,
        )
        print(f"mpcno_structured (full) -> RelL2={rel_full}, L2={abs_full}")

    rel_kw, abs_kw = evaluate_multidist(
        model=model,
        x_list=x_test_list,
        y_list=y_test_list,
        aux_list=aux_test_list,
        labels=label_list,
        batch_size=args.bsz,
        use_ablation=True,
    )

    print(f"mpcno_structured forward_ablation (K+W only) -> RelL2={rel_kw}, L2={abs_kw}")
    print("\nDone.")


if __name__ == "__main__":
    main()
