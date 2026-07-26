import argparse
import csv
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import meshio
import numpy as np
import torch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from pcno.geo_utility import compute_node_weights, preprocess_data_mesh
from pcno.mpcno import MPCNO, compute_Fourier_modes
from utility.normalizer import UnitGaussianNormalizer


SCRIPT_DIR = Path(__file__).resolve().parent


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def resolve_user_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def get_median_index(values):
    values = np.asarray(values)
    return int(np.argsort(values)[len(values) // 2])


def load_raw_data(data_path, ndata):
    nodes_list, elems_list, features_list = [], [], []
    for index in range(ndata):
        nodes_list.append(np.load(data_path / f"nodes_{index:05d}.npy"))
        elems_list.append(np.load(data_path / f"elems_{index:05d}.npy"))
        features_list.append(np.load(data_path / f"features_{index:05d}.npy"))
    return nodes_list, elems_list, features_list


def load_preprocessed_data(data_path, ndata, preprocess_data):
    preprocessed_path = data_path / "pcno_triangle_data.npz"
    if preprocess_data or not preprocessed_path.exists():
        print("Loading raw mesh data", flush=True)
        nodes_list, elems_list, features_list = load_raw_data(data_path, ndata)
        print("Preprocessing mesh data", flush=True)
        (
            nnodes,
            node_mask,
            nodes,
            node_measures_raw,
            features,
            directed_edges,
            edge_gradient_weights,
        ) = preprocess_data_mesh(
            nodes_list,
            elems_list,
            features_list,
            mesh_type="vertex_centered",
            adjacent_type="element",
        )
        node_measures, node_weights = compute_node_weights(
            nnodes, node_measures_raw, equal_measure=False
        )
        node_equal_measures, node_equal_weights = compute_node_weights(
            nnodes, node_measures_raw, equal_measure=True
        )
        np.savez_compressed(
            preprocessed_path,
            nnodes=nnodes,
            node_mask=node_mask,
            nodes=nodes,
            node_measures_raw=node_measures_raw,
            node_measures=node_measures,
            node_weights=node_weights,
            node_equal_measures=node_equal_measures,
            node_equal_weights=node_equal_weights,
            features=features,
            directed_edges=directed_edges,
            edge_gradient_weights=edge_gradient_weights,
        )
    return np.load(preprocessed_path)


def load_state_dict(model_path, device):
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    if any(key.startswith("module.") for key in checkpoint):
        checkpoint = {
            key.removeprefix("module."): value for key, value in checkpoint.items()
        }
    return checkpoint


def prepare_tensors(data, data_path, n_train, n_test, equal_weights):
    nnodes = torch.from_numpy(data["nnodes"])
    node_mask = torch.from_numpy(data["node_mask"])
    nodes = torch.from_numpy(data["nodes"].astype(np.float32))
    node_weights_np = data["node_equal_weights"] if equal_weights else data["node_weights"]
    node_weights = torch.from_numpy(node_weights_np.astype(np.float32))
    node_measures = data["node_measures"]
    node_measures_raw = data["node_measures_raw"]
    features = torch.from_numpy(data["features"].astype(np.float32))
    directed_edges = torch.from_numpy(data["directed_edges"].astype(np.int64))
    edge_gradient_weights = torch.from_numpy(data["edge_gradient_weights"].astype(np.float32))

    indices = np.isfinite(node_measures_raw)
    node_rhos_np = np.copy(node_weights_np)
    node_rhos_np[indices] = node_rhos_np[indices] / node_measures[indices]
    node_rhos = torch.from_numpy(node_rhos_np.astype(np.float32))

    normals_path = data_path / "car_shapenet_normals.npz"
    normals = torch.from_numpy(np.load(normals_path)["normals"].astype(np.float32))

    nodes_input = nodes.clone()
    x_train = torch.cat((nodes_input[:n_train], node_rhos[:n_train]), -1)
    x_test = torch.cat((nodes_input[-n_test:], node_rhos[-n_test:]), -1)
    y_train = features[:n_train, :, 0:1]
    y_test = features[-n_test:, :, 0:1]
    aux_test = (
        node_mask[-n_test:],
        nodes[-n_test:],
        node_weights[-n_test:],
        directed_edges[-n_test:],
        edge_gradient_weights[-n_test:],
        normals[-n_test:].permute(0, 2, 1),
    )
    nnodes_test = nnodes[-n_test:]
    return x_train, x_test, y_train, y_test, aux_test, nnodes_test


def build_model(args, x_train, y_train, device):
    layer_selection = {
        "grad": args.grad,
        "geo": args.geo,
        "geointegral": args.geointegral,
    }
    layers = [int(size) for size in args.layer_sizes.split(",")]
    length_scales = [float(value) for value in args.Ls.split(",")]
    if len(length_scales) != 3:
        raise ValueError(f"Expected exactly three length scales in --Ls, got {args.Ls!r}")
    print(f"Ls = {length_scales}", flush=True)
    modes = compute_Fourier_modes(
        3,
        [args.k_max, args.k_max, args.k_max],
        length_scales,
    )
    modes = torch.tensor(modes, dtype=torch.float32, device=device)
    model = MPCNO(
        3,
        modes,
        nmeasures=1,
        layers=layers,
        layer_selection=layer_selection,
        fc_dim=128,
        in_dim=x_train.shape[-1],
        out_dim=y_train.shape[-1],
        inv_L_scale_hyper=[False, 0.5, 2.0],
        scaling_mode="sqrt_inv",
        act=args.act,
        geo_act=args.geo_act,
    ).to(device)
    model.load_state_dict(load_state_dict(args.model_path, device))
    model.eval()
    return model


def evaluate_one(model, tensors, y_normalizer, device, index):
    x_test, y_test, aux_test = tensors
    x = x_test[[index]].to(device)
    y = y_test[[index]].to(device)
    aux = tuple(item[[index]].to(device) for item in aux_test)

    with torch.no_grad():
        pred = model(x, aux)
        pred = y_normalizer.decode(pred)
        y = y_normalizer.decode(y)

    mask = aux[0].to(pred.dtype)
    pred = pred * mask
    y = y * mask
    diff = pred - y
    denominator = torch.linalg.vector_norm(y.reshape(1, -1), dim=1).clamp_min(1.0e-12)
    rel_l2 = torch.linalg.vector_norm(diff.reshape(1, -1), dim=1) / denominator
    abs_l2 = torch.linalg.vector_norm(diff.reshape(1, -1), dim=1)
    return (
        float(rel_l2.item()),
        float(abs_l2.item()),
        pred.cpu().numpy()[0, :, 0],
        y.cpu().numpy()[0, :, 0],
        aux[1].cpu().numpy()[0],
    )


def write_vtk(data_path, output_dir, local_index, data_index, nnodes, prediction, reference):
    elems = np.load(data_path / f"elems_{data_index:05d}.npy")
    points = np.load(data_path / f"nodes_{data_index:05d}.npy")
    nvalid = int(nnodes)
    cells = [("triangle", elems[:, 1:])]
    mesh = meshio.Mesh(
        points=points[:nvalid],
        cells=cells,
        point_data={
            "pressure_ref": reference[:nvalid],
            "pressure_pred": prediction[:nvalid],
            "pressure_error": prediction[:nvalid] - reference[:nvalid],
            "pressure_abs_error": np.abs(prediction[:nvalid] - reference[:nvalid]),
        },
    )
    vtk_path = output_dir / f"sample_{local_index:03d}_data_{data_index:05d}.vtk"
    meshio.write(vtk_path, mesh)

    reference_mesh = meshio.Mesh(
        points=points[:nvalid],
        cells=cells,
        point_data={"pressure": reference[:nvalid]},
    )
    reference_vtk_path = output_dir / f"sample_{local_index:03d}_data_{data_index:05d}_reference.vtk"
    meshio.write(reference_vtk_path, reference_mesh)

    prediction_mesh = meshio.Mesh(
        points=points[:nvalid],
        cells=cells,
        point_data={"pressure": prediction[:nvalid]},
    )
    prediction_vtk_path = output_dir / f"sample_{local_index:03d}_data_{data_index:05d}_prediction.vtk"
    meshio.write(prediction_vtk_path, prediction_mesh)

    return vtk_path, reference_vtk_path, prediction_vtk_path


def plot_sample(data_path, output_dir, local_index, data_index, nnodes, prediction, reference, title):
    elems = np.load(data_path / f"elems_{data_index:05d}.npy")
    points = np.load(data_path / f"nodes_{data_index:05d}.npy")
    nvalid = int(nnodes)
    points = points[:nvalid]
    triangles = elems[:, 1:].astype(int)
    triangles = triangles[np.all(triangles < nvalid, axis=1)]

    values = [
        ("Reference", reference[:nvalid]),
        ("Prediction", prediction[:nvalid]),
        ("Absolute Error", np.abs(prediction[:nvalid] - reference[:nvalid])),
    ]
    ref_pred_min = min(np.nanmin(values[0][1]), np.nanmin(values[1][1]))
    ref_pred_max = max(np.nanmax(values[0][1]), np.nanmax(values[1][1]))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), subplot_kw={"projection": "3d"})
    for axis, (name, node_values) in zip(axes, values):
        face_values = node_values[triangles].mean(axis=1)
        vertices = points[triangles]
        collection = Poly3DCollection(vertices, linewidths=0.0)
        collection.set_array(face_values)
        collection.set_cmap("viridis" if name != "Absolute Error" else "magma")
        if name != "Absolute Error":
            collection.set_clim(ref_pred_min, ref_pred_max)
        axis.add_collection3d(collection, zs=vertices[:, :, 2], zdir="z")
        axis.auto_scale_xyz(points[:, 0], points[:, 1], points[:, 2])
        axis.view_init(elev=18, azim=-65)
        axis.set_title(name)
        axis.set_axis_off()
        fig.colorbar(collection, ax=axis, shrink=0.7, pad=0.02)

    fig.suptitle(title)
    fig.tight_layout()
    png_path = output_dir / f"sample_{local_index:03d}_data_{data_index:05d}.png"
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    return png_path


def write_error_files(output_dir, rows):
    csv_path = output_dir / "test_relative_l2_errors.csv"
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["test_local_index", "data_index", "relative_l2_error", "absolute_l2_error"],
        )
        writer.writeheader()
        writer.writerows(rows)

    rel_l2 = np.array([row["relative_l2_error"] for row in rows], dtype=np.float64)
    np.save(output_dir / "test_relative_l2_errors.npy", rel_l2)

    log_path = output_dir / "test_relative_l2_errors.log"
    with log_path.open("w") as file:
        for row in rows:
            file.write(
                f"test_local_index={row['test_local_index']:03d} "
                f"data_index={row['data_index']:05d} "
                f"rel_l2={row['relative_l2_error']:.8e} "
                f"abs_l2={row['absolute_l2_error']:.8e}\n"
            )

    return csv_path, output_dir / "test_relative_l2_errors.npy", log_path, rel_l2


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate an MPCNO car_shapenet model and plot selected test cases."
    )
    parser.add_argument("--model_path", type=Path, default=SCRIPT_DIR / "model" / "mpcno.pth")
    parser.add_argument("--data_path", type=Path, default=SCRIPT_DIR / "../../data/car_shapenet")
    parser.add_argument("--output_dir", type=Path, default=SCRIPT_DIR / "eval_mpcno")
    parser.add_argument("--n_train", type=int, default=500)
    parser.add_argument("--n_test", type=int, default=111)
    parser.add_argument("--ndata", type=int, default=611)
    parser.add_argument("--k_max", type=int, default=16)
    parser.add_argument("--Ls", type=str, default="4.0,4.0,12.0")
    parser.add_argument("--layer_sizes", type=str, default="64,64,64,64,64,64,64")
    parser.add_argument("--act", type=str, default="gelu")
    parser.add_argument("--geo_act", type=str, default="softsign")
    parser.add_argument("--grad", type=str_to_bool, default=True)
    parser.add_argument("--geo", type=str_to_bool, default=True)
    parser.add_argument("--geointegral", type=str_to_bool, default=True)
    parser.add_argument("--equal_weights", action="store_true")
    parser.add_argument("--preprocess_data", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    args.model_path = resolve_user_path(args.model_path)
    args.data_path = resolve_user_path(args.data_path)
    args.output_dir = resolve_user_path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data = load_preprocessed_data(args.data_path, args.ndata, args.preprocess_data)
    x_train, x_test, y_train, y_test, aux_test, nnodes_test = prepare_tensors(
        data, args.data_path, args.n_train, args.n_test, args.equal_weights
    )
    y_normalizer = UnitGaussianNormalizer(
        y_train,
        non_normalized_dim=0,
        normalization_dim=[],
    )
    y_test = y_normalizer.encode(y_test)
    y_normalizer.to(device)
    model = build_model(args, x_train, y_train, device)

    rows = []
    eval_tensors = (x_test, y_test, aux_test)
    for local_index in range(args.n_test):
        rel_l2, abs_l2, _, _, _ = evaluate_one(
            model, eval_tensors, y_normalizer, device, local_index
        )
        data_index = args.n_train + local_index
        rows.append(
            {
                "test_local_index": local_index,
                "data_index": data_index,
                "relative_l2_error": rel_l2,
                "absolute_l2_error": abs_l2,
            }
        )
        print(
            f"test_local_index={local_index:03d} data_index={data_index:05d} "
            f"rel_l2={rel_l2:.8e}",
            flush=True,
        )

    csv_path, npy_path, log_path, rel_l2 = write_error_files(args.output_dir, rows)
    max_index = int(np.argmax(rel_l2))
    median_index = get_median_index(rel_l2)

    selected_outputs = {}
    for label, local_index in [("max", max_index), ("median", median_index)]:
        rel_error, _, prediction, reference, _ = evaluate_one(
            model, eval_tensors, y_normalizer, device, local_index
        )
        data_index = args.n_train + local_index
        title = (
            f"{label}: test {local_index}, data {data_index}, "
            f"relative L2 = {rel_error:.6e}"
        )
        png_path = plot_sample(
            args.data_path,
            args.output_dir,
            local_index,
            data_index,
            nnodes_test[local_index],
            prediction,
            reference,
            title,
        )
        vtk_path, reference_vtk_path, prediction_vtk_path = write_vtk(
            args.data_path,
            args.output_dir,
            local_index,
            data_index,
            nnodes_test[local_index],
            prediction,
            reference,
        )
        selected_outputs[label] = {
            "test_local_index": local_index,
            "data_index": data_index,
            "relative_l2_error": rel_error,
            "png": str(png_path),
            "vtk": str(vtk_path),
            "reference_vtk": str(reference_vtk_path),
            "prediction_vtk": str(prediction_vtk_path),
        }

    summary = {
        "model_path": str(args.model_path),
        "Ls": args.Ls,
        "csv": str(csv_path),
        "npy": str(npy_path),
        "log": str(log_path),
        "mean_relative_l2_error": float(np.mean(rel_l2)),
        "median_relative_l2_error": float(np.median(rel_l2)),
        "max_relative_l2_error": float(np.max(rel_l2)),
        "selected_outputs": selected_outputs,
    }
    summary_path = args.output_dir / "summary.json"
    with summary_path.open("w") as file:
        json.dump(summary, file, indent=2)

    print(f"Wrote per-sample errors to {csv_path}", flush=True)
    print(f"Wrote per-sample error log to {log_path}", flush=True)
    print(f"Wrote summary to {summary_path}", flush=True)
    print(
        "Max sample: "
        f"test_local_index={max_index}, data_index={args.n_train + max_index}, "
        f"rel_l2={rel_l2[max_index]:.8e}",
        flush=True,
    )
    print(
        "Median sample: "
        f"test_local_index={median_index}, data_index={args.n_train + median_index}, "
        f"rel_l2={rel_l2[median_index]:.8e}",
        flush=True,
    )


if __name__ == "__main__":
    main()
