import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from mpcno_eval_errors import (
    SCRIPT_DIR,
    build_model,
    load_preprocessed_data,
    prepare_tensors,
    resolve_user_path,
    str_to_bool,
)
from pcno.geo_utility import compute_node_weights, preprocess_data_mesh
from utility.normalizer import UnitGaussianNormalizer


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed_call(fn, device=None):
    if device is not None:
        synchronize(device)
    start = time.perf_counter()
    result = fn()
    if device is not None:
        synchronize(device)
    elapsed = time.perf_counter() - start
    return result, elapsed


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean_seconds": float(np.mean(values)),
        "median_seconds": float(np.median(values)),
        "min_seconds": float(np.min(values)),
        "max_seconds": float(np.max(values)),
        "std_seconds": float(np.std(values)),
    }


def load_raw_sample(data_path, data_index):
    nodes = np.load(data_path / f"nodes_{data_index:05d}.npy")
    elems = np.load(data_path / f"elems_{data_index:05d}.npy")
    features = np.load(data_path / f"features_{data_index:05d}.npy")
    normals = np.load(data_path / "car_shapenet_normals.npz")["normals"][data_index]
    return nodes, elems, features, normals


def preprocess_raw_sample(raw_sample):
    nodes_raw, elems_raw, features_raw, normals_raw = raw_sample
    (
        nnodes,
        node_mask,
        nodes,
        node_measures_raw,
        features,
        directed_edges,
        edge_gradient_weights,
    ) = preprocess_data_mesh(
        [nodes_raw],
        [elems_raw],
        [features_raw],
        mesh_type="vertex_centered",
        adjacent_type="element",
    )
    node_measures, node_weights = compute_node_weights(
        nnodes,
        node_measures_raw,
        equal_measure=False,
    )

    indices = np.isfinite(node_measures_raw)
    node_rhos = np.copy(node_weights)
    node_rhos[indices] = node_rhos[indices] / node_measures[indices]

    node_mask = torch.from_numpy(node_mask)
    nodes = torch.from_numpy(nodes.astype(np.float32))
    node_weights = torch.from_numpy(node_weights.astype(np.float32))
    node_rhos = torch.from_numpy(node_rhos.astype(np.float32))
    features = torch.from_numpy(features.astype(np.float32))
    directed_edges = torch.from_numpy(directed_edges.astype(np.int64))
    edge_gradient_weights = torch.from_numpy(edge_gradient_weights.astype(np.float32))
    normals = torch.from_numpy(normals_raw[np.newaxis].astype(np.float32))

    x = torch.cat((nodes, node_rhos), -1)
    y = features[:, :, 0:1]
    aux = (
        node_mask,
        nodes,
        node_weights,
        directed_edges,
        edge_gradient_weights,
        normals.permute(0, 2, 1),
    )
    return x, y, aux


def to_device_sample(sample, device):
    x, y, aux = sample
    return x.to(device), y.to(device), tuple(item.to(device) for item in aux)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure single-sample M-PCNO preprocessing and prediction time."
    )
    parser.add_argument("--model_path", type=Path, default=SCRIPT_DIR / "model" / "mpcno.pth")
    parser.add_argument("--data_path", type=Path, default=SCRIPT_DIR / "../../data/car_shapenet")
    parser.add_argument("--output", type=Path, default=SCRIPT_DIR / "mpcno_single_sample_timing.json")
    parser.add_argument("--data_index", type=int, default=500)
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
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()
    args.model_path = resolve_user_path(args.model_path)
    args.data_path = resolve_user_path(args.data_path)
    args.output = resolve_user_path(args.output)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data = load_preprocessed_data(args.data_path, args.ndata, preprocess_data=False)
    x_train, _, y_train, _, _, _ = prepare_tensors(
        data,
        args.data_path,
        args.n_train,
        args.n_test,
        equal_weights=False,
    )
    y_normalizer = UnitGaussianNormalizer(
        y_train,
        non_normalized_dim=0,
        normalization_dim=[],
    )
    y_normalizer.to(device)
    model = build_model(args, x_train, y_train, device)

    raw_sample, raw_load_time = timed_call(
        lambda: load_raw_sample(args.data_path, args.data_index)
    )
    sample, preprocess_time = timed_call(lambda: preprocess_raw_sample(raw_sample))
    device_sample, host_to_device_time = timed_call(
        lambda: to_device_sample(sample, device),
        device=device,
    )
    x, _, aux = device_sample

    with torch.no_grad():
        for _ in range(args.warmup):
            _ = model(x, aux)
        synchronize(device)

        forward_times = []
        prediction_times = []
        for _ in range(args.repeats):
            _, forward_time = timed_call(lambda: model(x, aux), device=device)
            forward_times.append(forward_time)

            def predict_with_decode():
                pred = model(x, aux)
                pred = y_normalizer.decode(pred)
                pred = pred * aux[0].to(pred.dtype)
                return pred

            _, prediction_time = timed_call(predict_with_decode, device=device)
            prediction_times.append(prediction_time)

    summary = {
        "device": str(device),
        "data_index": args.data_index,
        "raw_load_seconds": raw_load_time,
        "preprocess_seconds": preprocess_time,
        "host_to_device_seconds": host_to_device_time,
        "forward_only": summarize(forward_times),
        "prediction_with_decode_and_mask": summarize(prediction_times),
        "warmup": args.warmup,
        "repeats": args.repeats,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as file:
        json.dump(summary, file, indent=2)

    print(f"Device: {device}")
    print(f"Data index: {args.data_index}")
    print(f"Raw load time: {raw_load_time:.8e} s")
    print(f"Preprocess time: {preprocess_time:.8e} s")
    print(f"Host-to-device time: {host_to_device_time:.8e} s")
    print(
        "Forward-only time: "
        f"{summary['forward_only']['mean_seconds']:.8e} s "
        f"(median {summary['forward_only']['median_seconds']:.8e} s)"
    )
    print(
        "Prediction time with decode/mask: "
        f"{summary['prediction_with_decode_and_mask']['mean_seconds']:.8e} s "
        f"(median {summary['prediction_with_decode_and_mask']['median_seconds']:.8e} s)"
    )
    print(f"Wrote timing summary to {args.output}")


if __name__ == "__main__":
    main()
