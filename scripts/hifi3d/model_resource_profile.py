from __future__ import annotations

import csv
import gc
import inspect
import os
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from train_utils import (  # noqa: E402
    _build_mu_array,
    _forward as _baseline_forward,
    _make_baseline_tensors,
    _make_node_weights,
    _make_tensors,
    _normalize_mu,
    _split_balanced_by_metadata_field,
    _load_metadata,
)
from geofno_train import _build_model as _build_geofno_model  # noqa: E402
from transolver_train import _build_model as _build_transolver_model  # noqa: E402
from utility.adam import Adam  # noqa: E402
from utility.losses import LpLoss  # noqa: E402
from utility.normalizer import UnitGaussianNormalizer  # noqa: E402


ROOT = Path("/root/autodl-tmp/NeuralOperator")
DATA_NPZ = ROOT / "data/HiFi3D_processed/aircraft_mixed_medium/AircraftMixed_A150_N149_B500_vertex_centered.npz"
NAMES = ROOT / "data/HiFi3D_processed/aircraft_mixed_medium/AircraftMixed_A150_N149_B500_names.npy"
METADATA = ROOT / "data/HiFi3D_processed/aircraft_mixed_medium/AircraftMixed_A150_N149_B500_metadata.tsv"
OUT_CSV = ROOT / "logs/hifi3d_model_resource_profile_aircraft_mixed.csv"

MU_FIELDS = [
    "mach",
    "alpha",
    "beta",
    "log10_Re",
    "has_mach",
    "has_alpha",
    "has_beta",
    "has_log10_Re",
]


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for memory profiling.")

    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")

    data = np.load(DATA_NPZ)
    names = np.load(NAMES, allow_pickle=True)
    metadata = _load_metadata(METADATA)
    train_idx, test_idx = _split_balanced_by_metadata_field(
        names,
        metadata,
        field="dataset",
        n_train_per_group=100,
        n_test_per_group=30,
        seed=0,
    )

    node_weights_array = _make_node_weights(data, "measure", 0.0)
    mu_array = _build_mu_array(names, metadata, MU_FIELDS, missing="error")
    mu_array, _, _ = _normalize_mu(mu_array, train_idx)

    rows = []
    for spec in model_specs():
        print(f"Profiling {spec['label']}...", flush=True)
        rows.append(profile_one(spec, data, train_idx, test_idx, node_weights_array, mu_array, device))

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "input_dim",
                "batch_size",
                "trainable_params",
                "total_params",
                "train_peak_allocated_mib",
                "train_peak_reserved_mib",
                "test_peak_allocated_mib",
                "test_peak_reserved_mib",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {OUT_CSV}", flush=True)
    for row in rows:
        print(row, flush=True)


def model_specs() -> list[dict[str, object]]:
    return [
        {
            "label": "PCNO",
            "family": "pcno",
            "model_name": "pcno",
            "batch_size": 8,
            "layers": [64, 64, 64, 64],
            "fc_dim": 128,
            "k_max": 12,
            "Ls": [4.1, 4.1, 4.1],
            "lr": 5.0e-4,
        },
        {
            "label": "M-PCNO",
            "family": "pcno",
            "model_name": "mpcno",
            "batch_size": 8,
            "layers": [64, 64, 64, 64],
            "fc_dim": 128,
            "k_max": 12,
            "Ls": [4.1, 4.1, 4.1],
            "lr": 5.0e-4,
        },
        {
            "label": "GeoFNO",
            "family": "baseline",
            "model_name": "geofno",
            "batch_size": 8,
            "layers": [64, 64, 64, 64],
            "fc_dim": 128,
            "k_max": 12,
            "Ls": [4.1, 4.1, 4.1],
            "lr": 5.0e-4,
        },
        {
            "label": "Transolver++",
            "family": "baseline",
            "model_name": "transolver",
            "batch_size": 1,
            "layers": [384, 384, 384, 384],
            "fc_dim": 128,
            "k_max": 12,
            "Ls": [4.1, 4.1, 4.1],
            "lr": 1.0e-3,
        },
    ]


def profile_one(
    spec: dict[str, object],
    data: np.lib.npyio.NpzFile,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    node_weights_array: np.ndarray,
    mu_array: np.ndarray,
    device: torch.device,
) -> dict[str, object]:
    batch_size = int(spec["batch_size"])
    family = str(spec["family"])
    model_name = str(spec["model_name"])

    if family == "pcno":
        x_train, y_train, aux_train = _make_tensors(data, train_idx, model_name, node_weights_array, mu_array)
        x_test, y_test, aux_test = _make_tensors(data, test_idx, model_name, node_weights_array, mu_array)
        build_fn = lambda: build_pcno_model(spec, x_train.shape[-1], y_train.shape[-1], device)
        forward_fn = lambda model, batch: model(batch[0], batch[2])
        train_batch = make_pcno_batch(x_train, y_train, aux_train, batch_size)
        test_batch = make_pcno_batch(x_test, y_test, aux_test, batch_size)
    else:
        x_train, y_train, aux_train = make_baseline_tensors_compat(
            data, train_idx, model_name, node_weights_array, mu_array
        )
        x_test, y_test, aux_test = make_baseline_tensors_compat(
            data, test_idx, model_name, node_weights_array, mu_array
        )
        build_fn = lambda: build_baseline_model(spec, x_train.shape[-1], y_train.shape[-1], device)
        forward_fn = lambda model, batch: baseline_forward_compat(model_name, model, batch)
        train_batch = make_baseline_batch(x_train, y_train, aux_train, batch_size)
        test_batch = make_baseline_batch(x_test, y_test, aux_test, batch_size)

    y_normalizer = UnitGaussianNormalizer(y_train, non_normalized_dim=0, normalization_dim=[])
    y_train_encoded = y_normalizer.encode(train_batch[1])
    y_test_encoded = y_normalizer.encode(test_batch[1])
    train_batch = (train_batch[0], y_train_encoded, train_batch[2])
    test_batch = (test_batch[0], y_test_encoded, test_batch[2])

    model_for_count = build_fn()
    trainable_params = sum(p.numel() for p in model_for_count.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model_for_count.parameters())
    del model_for_count
    cleanup()

    train_alloc, train_reserved = measure_train(
        build_fn, forward_fn, train_batch, y_normalizer, float(spec["lr"]), device
    )
    cleanup()
    test_alloc, test_reserved = measure_test(build_fn, forward_fn, test_batch, y_normalizer, device)
    cleanup()

    return {
        "model": spec["label"],
        "input_dim": int(x_train.shape[-1]),
        "batch_size": batch_size,
        "trainable_params": int(trainable_params),
        "total_params": int(total_params),
        "train_peak_allocated_mib": round(train_alloc, 1),
        "train_peak_reserved_mib": round(train_reserved, 1),
        "test_peak_allocated_mib": round(test_alloc, 1),
        "test_peak_reserved_mib": round(test_reserved, 1),
    }


def build_pcno_model(spec: dict[str, object], in_dim: int, out_dim: int, device: torch.device) -> torch.nn.Module:
    model_name = str(spec["model_name"])
    if model_name == "mpcno":
        from pcno.mpcno import MPCNO, compute_Fourier_modes

        modes = torch.tensor(
            compute_Fourier_modes(3, [int(spec["k_max"])] * 3, list(spec["Ls"])),
            dtype=torch.float32,
            device=device,
        )
        return MPCNO(
            3,
            modes,
            nmeasures=1,
            layer_selection={"grad": True, "geo": True, "geointegral": True},
            layers=list(spec["layers"]),
            fc_dim=int(spec["fc_dim"]),
            in_dim=in_dim,
            out_dim=out_dim,
            inv_L_scale_hyper=[False, 0.5, 2.0],
            act="gelu",
            geo_act="softsign",
            scaling_mode="sqrt_inv",
        ).to(device)

    from pcno.pcno import PCNO, compute_Fourier_modes

    modes = torch.tensor(
        compute_Fourier_modes(3, [int(spec["k_max"])] * 3, list(spec["Ls"])),
        dtype=torch.float32,
        device=device,
    )
    return PCNO(
        3,
        modes,
        nmeasures=1,
        layers=list(spec["layers"]),
        fc_dim=int(spec["fc_dim"]),
        in_dim=in_dim,
        out_dim=out_dim,
        inv_L_scale_hyper=["together", 0.5, 2.0],
        act="gelu",
    ).to(device)


def build_baseline_model(spec: dict[str, object], in_dim: int, out_dim: int, device: torch.device) -> torch.nn.Module:
    model_name = str(spec["model_name"])
    args = Namespace(
        k_max=int(spec["k_max"]),
        fc_dim=int(spec["fc_dim"]),
        transolver_nhead=8,
        transolver_slice_num=32,
        transolver_dropout=0.0,
        transolver_mlp_ratio=2,
        transolver_ref=8,
    )
    if model_name == "geofno":
        return _build_geofno_model(args, list(spec["layers"]), list(spec["Ls"]), in_dim, out_dim, device)
    if model_name == "transolver":
        return _build_transolver_model(args, list(spec["layers"]), list(spec["Ls"]), in_dim, out_dim, device)
    raise ValueError(f"Unsupported baseline model for profiling: {model_name}")


def make_pcno_batch(x: torch.Tensor, y: torch.Tensor, aux: tuple[torch.Tensor, ...], batch_size: int):
    return x[:batch_size], y[:batch_size], tuple(item[:batch_size] for item in aux)


def make_baseline_batch(x: torch.Tensor, y: torch.Tensor, aux: tuple[torch.Tensor, ...], batch_size: int):
    return x[:batch_size], y[:batch_size], tuple(item[:batch_size] for item in aux)


def make_baseline_tensors_compat(
    data: np.lib.npyio.NpzFile,
    indices: np.ndarray,
    model_name: str,
    node_weights_array: np.ndarray,
    mu_array: np.ndarray,
):
    if len(inspect.signature(_make_baseline_tensors).parameters) == 7:
        return _make_baseline_tensors(data, indices, model_name, node_weights_array, mu_array, 0, "repeat")
    return _make_baseline_tensors(data, indices, model_name, node_weights_array, mu_array, 0)


def baseline_forward_compat(model_name: str, model: torch.nn.Module, batch):
    x, _, aux = batch
    if len(aux) == 4:
        return _baseline_forward(model_name, model, x, aux[0], aux[1], aux[2], aux[3])
    return _baseline_forward(model_name, model, x, aux[0], aux[1], aux[2])


def to_device_batch(batch, device: torch.device):
    x, y, aux = batch
    return x.to(device), y.to(device), tuple(item.to(device) for item in aux)


def measure_train(build_fn, forward_fn, batch, y_normalizer, lr: float, device: torch.device) -> tuple[float, float]:
    model = build_fn()
    y_normalizer.to(device)
    optimizer = Adam(model.parameters(), betas=(0.9, 0.999), lr=lr, weight_decay=1.0e-4)
    loss_fn = LpLoss(d=1, p=2, size_average=False)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)

    model.train()
    x, y, aux = to_device_batch(batch, device)
    optimizer.zero_grad()
    out = forward_fn(model, (x, y, aux))
    out = y_normalizer.decode(out)
    y = y_normalizer.decode(y)
    node_mask = aux[0]
    out = out * node_mask
    y = y * node_mask
    loss = loss_fn(out.reshape(x.shape[0], -1), y.reshape(x.shape[0], -1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    torch.cuda.synchronize(device)

    allocated = torch.cuda.max_memory_allocated(device) / 1024**2
    reserved = torch.cuda.max_memory_reserved(device) / 1024**2
    return allocated, reserved


def measure_test(build_fn, forward_fn, batch, y_normalizer, device: torch.device) -> tuple[float, float]:
    model = build_fn()
    y_normalizer.to(device)
    loss_fn = LpLoss(d=1, p=2, size_average=False)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)

    model.eval()
    with torch.no_grad():
        x, y, aux = to_device_batch(batch, device)
        out = forward_fn(model, (x, y, aux))
        out = y_normalizer.decode(out)
        y = y_normalizer.decode(y)
        node_mask = aux[0]
        out = out * node_mask
        y = y * node_mask
        _ = loss_fn(out.reshape(x.shape[0], -1), y.reshape(x.shape[0], -1))
        torch.cuda.synchronize(device)

    allocated = torch.cuda.max_memory_allocated(device) / 1024**2
    reserved = torch.cuda.max_memory_reserved(device) / 1024**2
    return allocated, reserved


def cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
