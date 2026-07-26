import argparse
import csv
import os
import re
import sys
from pathlib import Path

import numpy as np
import pyvista as pv
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from pcno.mpcno import MPCNO, compute_Fourier_modes
from utility.normalizer import UnitGaussianNormalizer


def str_to_bool(value):
    return str(value).lower() == "true"


def load_data_to_torch(data_file_path, to_divide=None, factor=1.0):
    data = np.load(data_file_path)
    nnodes, node_mask, nodes = data["nnodes"], data["node_mask"], data["nodes"]
    print(f"Loaded {nodes.shape[0]} samples from {data_file_path}", flush=True)

    node_weights = data["node_measures"]
    if to_divide is None:
        to_divide = factor * np.amax(np.sum(node_weights, axis=1))
    node_weights = node_weights / to_divide

    node_measures = data["node_measures"]
    directed_edges = data["directed_edges"]
    edge_gradient_weights = data["edge_gradient_weights"]
    features = data["features"]

    node_measures_raw = data["node_measures_raw"]
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
    edge_gradient_weights = edge_gradient_weights / 10

    return (
        nnodes,
        node_mask,
        nodes,
        node_weights,
        node_rhos,
        features,
        directed_edges,
        edge_gradient_weights,
        to_divide,
    )


def gen_data_tensors(data_indices, nodes, features, node_mask, node_weights, directed_edges, edge_gradient_weights):
    nodes_input = nodes.clone()
    normals = features[data_indices][..., :3]
    x = torch.cat((normals, nodes_input[data_indices, ...]), -1)
    y = features[data_indices][..., -1:]
    aux = (
        node_mask[data_indices],
        nodes[data_indices],
        node_weights[data_indices],
        directed_edges[data_indices],
        edge_gradient_weights[data_indices],
        normals.permute(0, 2, 1),
    )
    return x, y, aux


def split_indices(ndata, n_train, n_test, seed):
    if seed:
        rng = np.random.default_rng(seed)
        order = rng.permutation(ndata)
        train_idx = order[:n_train]
        test_idx = order[n_train : n_train + n_test]
    else:
        train_idx = np.arange(n_train)
        test_idx = np.arange(ndata - n_test, ndata)
    if len(test_idx) != n_test:
        raise ValueError(f"Expected {n_test} test samples, got {len(test_idx)}")
    return train_idx, test_idx


def load_state_dict(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    if any(key.startswith("module.") for key in checkpoint):
        checkpoint = {key.removeprefix("module."): value for key, value in checkpoint.items()}
    return checkpoint


def original_index_from_stem(stem):
    match = re.search(r"boundary_(\d+)_20k$", stem)
    if match:
        return int(match.group(1))
    return None


def build_sorted_file_index(raw_data_dir):
    files = sorted(Path(raw_data_dir).glob("*.vtp"))
    return {path.stem: i for i, path in enumerate(files)}


def build_numeric_file_index(raw_data_dir):
    files = sorted(
        Path(raw_data_dir).glob("*.vtp"),
        key=lambda path: (
            original_index_from_stem(path.stem) is None,
            original_index_from_stem(path.stem) if original_index_from_stem(path.stem) is not None else path.stem,
        ),
    )
    return {path.stem: i for i, path in enumerate(files)}


def stem_from_name(name):
    name = str(name)
    return name.split("-", 1)[1] if "-" in name else Path(name).stem


def write_prediction_vtp(raw_data_dir, stem, prediction, output_suffix):
    source_path = Path(raw_data_dir) / f"{stem}.vtp"
    if not source_path.exists():
        raise FileNotFoundError(f"Missing source VTP file: {source_path}")

    mesh = pv.read(source_path)
    if mesh.n_points != prediction.shape[0]:
        raise ValueError(
            f"{source_path} has {mesh.n_points} points, but prediction has {prediction.shape[0]}"
        )
    mesh.point_data["Cp"] = prediction.astype(np.float32)
    if "Cp" in mesh.cell_data:
        del mesh.cell_data["Cp"]
    output_path = source_path.with_name(f"{stem}{output_suffix}.vtp")
    mesh.save(output_path)
    return output_path


def format_summary_row(label, row):
    return (
        f"{label}: "
        f"test_local_index={row['test_local_index']}, "
        f"processed_index={row['processed_index']}, "
        f"original_boundary_index={row['original_boundary_index']}, "
        f"name={row['sample_name']}, "
        f"relative_l2_error={row['relative_l2_error']:.8e}, "
        f"prediction_vtp={row['prediction_vtp']}"
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained DrivAerML MPCNO model on test samples.")
    parser.add_argument("--data_path", type=Path, default=Path("../../data/hifi3d_processed/test/drivaerml_vertex_centered.npz"))
    parser.add_argument("--names_path", type=Path, default=Path("../../data/hifi3d_processed/test/drivaerml_names.npy"))
    parser.add_argument("--raw_data_dir", type=Path, default=Path("../../data/HiFi3D/DrivAerML_20000"))
    parser.add_argument("--model_path", type=Path, default=Path("model/mpcno.pth"))
    parser.add_argument("--output_csv", type=Path, default=Path("model/mpcno_test_errors.csv"))
    parser.add_argument("--output_suffix", type=str, default="_test")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_train", type=int, default=400)
    parser.add_argument("--n_test", type=int, default=80)
    parser.add_argument("--to_divide_factor", type=float, default=20.0)
    parser.add_argument("--k_max", type=int, default=12)
    parser.add_argument("--Ls", type=str, default="")
    parser.add_argument("--grad", type=str, default="True", choices=["True", "False"])
    parser.add_argument("--geo", type=str, default="True", choices=["True", "False"])
    parser.add_argument("--geointegral", type=str, default="True", choices=["True", "False"])
    parser.add_argument("--act", type=str, default="gelu")
    parser.add_argument("--geo_act", type=str, default="softsign")
    parser.add_argument("--layer_sizes", type=str, default="64,64,64,64")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--skip_vtp", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ndim = 3
    layer_selection = {
        "grad": str_to_bool(args.grad),
        "geo": str_to_bool(args.geo),
        "geointegral": str_to_bool(args.geointegral),
    }
    layers = [int(size) for size in args.layer_sizes.split(",")]

    (
        nnodes,
        node_mask,
        nodes,
        node_weights,
        _node_rhos,
        features,
        directed_edges,
        edge_gradient_weights,
        _to_divide,
    ) = load_data_to_torch(args.data_path, to_divide=None, factor=args.to_divide_factor)
    names = np.load(args.names_path, allow_pickle=True)

    ndata = int(nnodes.shape[0])
    if len(names) != ndata:
        raise ValueError(f"names length {len(names)} does not match data length {ndata}")

    train_idx, test_idx = split_indices(ndata, args.n_train, args.n_test, args.seed)

    x_train, y_train, _aux_train = gen_data_tensors(
        train_idx, nodes, features, node_mask, node_weights, directed_edges, edge_gradient_weights
    )
    x_test, y_test, aux_test = gen_data_tensors(
        test_idx, nodes, features, node_mask, node_weights, directed_edges, edge_gradient_weights
    )

    if args.Ls:
        Ls = [float(value) for value in args.Ls.split(",")]
        if len(Ls) != ndim:
            raise ValueError(f"Expected {ndim} values in --Ls, got {Ls}")
    else:
        lengths = torch.amax(nodes[..., 0:3], dim=(0, 1)) - torch.amin(nodes[..., 0:3], dim=(0, 1))
        Ls = [float(length.item()) * 2 + 0.2 for length in lengths]

    modes = compute_Fourier_modes(ndim, [args.k_max, args.k_max, args.k_max], Ls)
    modes = torch.tensor(modes, dtype=torch.float32).to(device)
    model = MPCNO(
        ndim,
        modes,
        nmeasures=1,
        layer_selection=layer_selection,
        layers=layers,
        fc_dim=128,
        in_dim=x_train.shape[-1],
        out_dim=y_train.shape[-1],
        inv_L_scale_hyper=[False, 0.5, 2.0],
        scaling_mode="sqrt_inv",
        act=args.act,
        geo_act=args.geo_act,
    ).to(device)
    model.load_state_dict(load_state_dict(args.model_path, device))
    print(f"Load model from {args.model_path}",flush=True)
    model.eval()

    y_normalizer = UnitGaussianNormalizer(y_train, non_normalized_dim=0, normalization_dim=[])
    y_normalizer.to(device)
    sorted_file_index = build_sorted_file_index(args.raw_data_dir)
    numeric_file_index = build_numeric_file_index(args.raw_data_dir)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    rel_l2_errors = []

    node_mask_test, nodes_test, node_weights_test, directed_edges_test, edge_gradient_weights_test, geo_test = aux_test
    with torch.no_grad():
        for start in range(0, args.n_test, args.batch_size):
            end = min(start + args.batch_size, args.n_test)
            x = x_test[start:end].to(device)
            y = y_test[start:end].to(device)
            batch_node_mask = node_mask_test[start:end].to(device)
            batch_nodes = nodes_test[start:end].to(device)
            batch_node_weights = node_weights_test[start:end].to(device)
            batch_directed_edges = directed_edges_test[start:end].to(device)
            batch_edge_gradient_weights = edge_gradient_weights_test[start:end].to(device)
            batch_geo = geo_test[start:end].to(device)

            out = model(
                x,
                (
                    batch_node_mask,
                    batch_nodes,
                    batch_node_weights,
                    batch_directed_edges,
                    batch_edge_gradient_weights,
                    batch_geo,
                ),
            )
            out = y_normalizer.decode(out) * batch_node_mask
            y = y * batch_node_mask

            for local_offset in range(end - start):
                test_local_index = start + local_offset
                processed_index = int(test_idx[test_local_index])
                sample_name = str(names[processed_index])
                stem = stem_from_name(sample_name)
                valid_mask = batch_node_mask[local_offset, :, 0].bool()
                pred_valid = out[local_offset, valid_mask, 0]
                y_valid = y[local_offset, valid_mask, 0]
                rel_l2 = torch.norm(pred_valid - y_valid, p=2) / torch.norm(y_valid, p=2)
                rel_l2_value = float(rel_l2.cpu().item())
                rel_l2_errors.append(rel_l2_value)

                output_vtp = ""
                if not args.skip_vtp:
                    output_vtp = str(
                        write_prediction_vtp(
                            args.raw_data_dir,
                            stem,
                            pred_valid.detach().cpu().numpy(),
                            args.output_suffix,
                        )
                    )

                rows.append(
                    {
                        "test_local_index": test_local_index,
                        "processed_index": processed_index,
                        "sample_name": sample_name,
                        "stem": stem,
                        "original_boundary_index": original_index_from_stem(stem),
                        "numeric_file_index_0based": numeric_file_index.get(stem, ""),
                        "numeric_file_index_1based": numeric_file_index.get(stem, "") + 1
                        if stem in numeric_file_index
                        else "",
                        "sorted_file_index_0based": sorted_file_index.get(stem, ""),
                        "sorted_file_index_1based": sorted_file_index.get(stem, "") + 1
                        if stem in sorted_file_index
                        else "",
                        "relative_l2_error": rel_l2_value,
                        "prediction_vtp": output_vtp,
                    }
                )
                print(
                    f"test_local_index={test_local_index:03d} "
                    f"processed_npz_index={processed_index} "
                    f"original_boundary_index={original_index_from_stem(stem)} "
                    f"name={sample_name} "
                    f"rel_l2={rel_l2_value:.8e}",
                    flush=True,
                )

    fieldnames = [
        "test_local_index",
        "processed_index",
        "sample_name",
        "stem",
        "original_boundary_index",
        "numeric_file_index_0based",
        "numeric_file_index_1based",
        "sorted_file_index_0based",
        "sorted_file_index_1based",
        "relative_l2_error",
        "prediction_vtp",
    ]
    with args.output_csv.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    rows_by_error = sorted(rows, key=lambda row: row["relative_l2_error"])
    max_error_row = rows_by_error[-1]
    median_start = (len(rows_by_error) - 1) // 2
    median_end = len(rows_by_error) // 2
    summary_lines = [
        format_summary_row("max_error", max_error_row),
        format_summary_row("median_error_lower", rows_by_error[median_start]),
    ]
    if median_end != median_start:
        summary_lines.append(format_summary_row("median_error_upper", rows_by_error[median_end]))

    summary_path = args.output_csv.with_name(args.output_csv.stem + "_summary.txt")
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    for line in summary_lines:
        print(line, flush=True)

    print(f"Saved per-sample errors to {args.output_csv}", flush=True)
    print(f"Saved error summary to {summary_path}", flush=True)
    print(f"Mean relative L2 error: {np.mean(rel_l2_errors):.8e}", flush=True)


if __name__ == "__main__":
    main()
