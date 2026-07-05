"""Autoregressive PCNO rollouts from preprocessed Euler ``.npz`` data.

This is the corrected PCNO evaluation path for the 2D Euler bump data:

    HDF5 -> per-trajectory npy -> reconstructed cells -> pcno_Euler_forward_data.npz

It intentionally does not build PCNO auxiliary tensors directly from raw HDF5
edges. The output HDF5 files use the same ``predicteds`` / ``targets`` contract
as the CPGNet rollout artifacts so existing visualization/diagnostic scripts can
consume them.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FEATURE_NAMES = ("rho", "v1", "v2", "pres")
EPS = 1.0e-12


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    date = datetime.now().strftime("%Y%m%d")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(f"artifacts/time_dependent_no/pcno_corrected_rollout_{date}"),
    )
    parser.add_argument("--mapping-file", type=Path)
    parser.add_argument("--sample-indices", nargs="+", type=int)
    parser.add_argument("--n-time", type=int, default=79)
    parser.add_argument("--start-time", type=int, default=0)
    parser.add_argument("--k-max", type=int, default=8)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--equal-weights", action="store_true")
    return parser.parse_args(argv)


def load_rollout_visualization_module() -> Any:
    module_path = ROOT / "scripts" / "2d_Euler_eq" / "rollout_visualization.py"
    spec = importlib.util.spec_from_file_location("pcno_euler_rollout_visualization", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def select_device(args: argparse.Namespace) -> torch.device:
    if args.device == "cpu":
        return torch.device("cpu")
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.set_device(args.gpu)
        return torch.device("cuda")
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        return torch.device("cuda")
    return torch.device("cpu")


def load_sample_mapping(path: Path | None) -> dict[int, int]:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    for key in ("sample_to_trajectory", "sample_to_original", "sample_to_original_index"):
        if isinstance(payload, dict) and key in payload:
            payload = payload[key]
            break
    if (
        isinstance(payload, dict)
        and "sample_indices" in payload
        and "test_trajectory_indices" in payload
    ):
        return {
            int(sample): int(trajectory)
            for sample, trajectory in zip(
                payload["sample_indices"],
                payload["test_trajectory_indices"],
            )
        }
    if isinstance(payload, dict) and "trajectory_indices" in payload:
        payload = payload["trajectory_indices"]

    if isinstance(payload, dict):
        mapping: dict[int, int] = {}
        for key, value in payload.items():
            if isinstance(value, dict):
                value = (
                    value.get("trajectory_index")
                    or value.get("original_trajectory")
                    or value.get("original_index")
                )
            mapping[int(key)] = int(value)
        return mapping

    if isinstance(payload, list):
        if all(isinstance(item, int) for item in payload):
            return {sample: int(original) for sample, original in enumerate(payload)}
        mapping = {}
        for item in payload:
            if not isinstance(item, dict):
                continue
            sample = item.get("sample_index", item.get("sample"))
            original = (
                item.get("trajectory_index")
                or item.get("original_trajectory")
                or item.get("original_index")
            )
            if sample is not None and original is not None:
                mapping[int(sample)] = int(original)
        return mapping

    raise ValueError(f"unsupported mapping file format in {path}")


def relative_l2_by_time(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    error = prediction - truth
    numerator = np.linalg.norm(error, axis=1)
    denominator = np.linalg.norm(truth, axis=1)
    relative = np.full(numerator.shape, np.nan, dtype=np.float64)
    valid = denominator > EPS
    relative[valid] = numerator[valid] / denominator[valid]
    relative[~valid & (numerator <= EPS)] = 0.0
    return relative


def summarize_rollout(
    *,
    sample_index: int,
    trajectory_index: int,
    truth: np.ndarray,
    prediction: np.ndarray,
    elapsed_seconds: float,
    result_file: Path,
) -> dict[str, Any]:
    rollout_truth = truth[1:]
    rollout_prediction = prediction[1:]
    error = rollout_prediction - rollout_truth
    rmse = np.sqrt(np.mean(error**2, axis=(0, 1)))
    final_rmse = np.sqrt(np.mean(error[-1] ** 2, axis=0))
    rel = relative_l2_by_time(rollout_truth, rollout_prediction)
    summary: dict[str, Any] = {
        "sample_index": int(sample_index),
        "trajectory_index": int(trajectory_index),
        "result_file": str(result_file),
        "num_steps": int(rollout_truth.shape[0]),
        "num_nodes": int(rollout_truth.shape[1]),
        "elapsed_seconds": float(elapsed_seconds),
        "finite": bool(np.isfinite(rollout_prediction).all()),
        "min_pred_density": float(np.nanmin(rollout_prediction[..., 0])),
        "min_pred_pressure": float(np.nanmin(rollout_prediction[..., 3])),
        "max_abs_prediction": float(np.nanmax(np.abs(rollout_prediction))),
        "rmse": {name: float(value) for name, value in zip(FEATURE_NAMES, rmse)},
        "final_rmse": {name: float(value) for name, value in zip(FEATURE_NAMES, final_rmse)},
        "relative_l2_mean": {
            name: float(value) for name, value in zip(FEATURE_NAMES, np.nanmean(rel, axis=0))
        },
        "relative_l2_final": {
            name: float(value) for name, value in zip(FEATURE_NAMES, rel[-1])
        },
    }
    return summary


def write_result_file(
    path: Path,
    *,
    truth: np.ndarray,
    prediction: np.ndarray,
    sample_index: int,
    trajectory_index: int,
    start_time: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("predicteds", data=prediction[1:], compression="gzip")
        handle.create_dataset("targets", data=truth[1:], compression="gzip")
        handle.attrs["sample_index"] = int(sample_index)
        handle.attrs["trajectory_index"] = int(trajectory_index)
        handle.attrs["start_time"] = int(start_time)
        handle.attrs["num_steps"] = int(prediction.shape[0] - 1)
        handle.attrs["source"] = "pcno_preprocessed_rollout"


def write_summary(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "rollout_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    rows = []
    for item in payload["trajectories"]:
        row = {
            "sample_index": item["sample_index"],
            "trajectory_index": item["trajectory_index"],
            "num_steps": item["num_steps"],
            "num_nodes": item["num_nodes"],
            "finite": item["finite"],
            "min_pred_density": item["min_pred_density"],
            "min_pred_pressure": item["min_pred_pressure"],
        }
        for group in ("rmse", "final_rmse", "relative_l2_mean", "relative_l2_final"):
            for name in FEATURE_NAMES:
                row[f"{group}_{name}"] = item[group][name]
        rows.append(row)
    if rows:
        with (output_dir / "per_trajectory_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    lines = [
        "# Corrected PCNO Preprocessed Rollout",
        "",
        f"Generated: {payload['generated_at']}",
        f"Checkpoint: `{payload['checkpoint_name']}`",
        f"Data dir: `{payload['data_dir']}`",
        f"Device: `{payload['device']}`",
        "",
        "## Aggregate RMSE",
        "",
        "| Variable | Mean RMSE | Mean final RMSE |",
        "| --- | ---: | ---: |",
    ]
    for name in FEATURE_NAMES:
        mean_rmse = np.mean([item["rmse"][name] for item in payload["trajectories"]])
        mean_final = np.mean([item["final_rmse"][name] for item in payload["trajectories"]])
        lines.append(f"| {name} | {mean_rmse:.6g} | {mean_final:.6g} |")
    lines.extend(["", "## Trajectories", ""])
    for item in payload["trajectories"]:
        lines.append(
            f"- sample {item['sample_index']} -> trajectory {item['trajectory_index']}: "
            f"pres RMSE {item['rmse']['pres']:.6g}, final {item['final_rmse']['pres']:.6g}, "
            f"finite={item['finite']}"
        )
    (output_dir / "rollout_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    start = time.time()
    viz = load_rollout_visualization_module()
    device = select_device(args)
    data = viz._load_euler_forward_data(args.data_dir, equal_weights=args.equal_weights)
    model = viz.build_euler2d_pcno_from_data(
        data,
        model_path=args.checkpoint,
        device=device,
        k_max=args.k_max,
    )
    nsamples = int(data["features"].shape[0])
    sample_indices = args.sample_indices if args.sample_indices is not None else list(range(nsamples))
    mapping = load_sample_mapping(args.mapping_file)

    summaries: list[dict[str, Any]] = []
    result_dir = args.output_dir / "result"
    for sample_index in sample_indices:
        if sample_index < 0 or sample_index >= nsamples:
            raise IndexError(f"sample index {sample_index} outside [0, {nsamples})")
        trajectory_index = int(mapping.get(sample_index, sample_index))
        step_start = time.time()
        truth, prediction = viz.rollout_euler2d_sample(
            model,
            data,
            index=sample_index,
            n_time=args.n_time,
            start_time=args.start_time,
            device=device,
        )
        result_file = result_dir / f"{trajectory_index + 1}.h5"
        write_result_file(
            result_file,
            truth=truth,
            prediction=prediction,
            sample_index=sample_index,
            trajectory_index=trajectory_index,
            start_time=args.start_time,
        )
        summary = summarize_rollout(
            sample_index=sample_index,
            trajectory_index=trajectory_index,
            truth=truth,
            prediction=prediction,
            elapsed_seconds=time.time() - step_start,
            result_file=result_file,
        )
        summaries.append(summary)
        print(
            json.dumps(
                {
                    "sample_index": sample_index,
                    "trajectory_index": trajectory_index,
                    "result_file": str(result_file),
                    "pres_rmse": summary["rmse"]["pres"],
                    "pres_final_rmse": summary["final_rmse"]["pres"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    payload = {
        "mode": "pcno_preprocessed_rollout",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_seconds": float(time.time() - start),
        "checkpoint_name": args.checkpoint.name,
        "data_dir": str(args.data_dir),
        "mapping_file": str(args.mapping_file) if args.mapping_file else None,
        "device": str(device),
        "sample_indices": [int(value) for value in sample_indices],
        "n_time": int(args.n_time),
        "start_time": int(args.start_time),
        "k_max": int(args.k_max),
        "equal_weights": bool(args.equal_weights),
        "trajectories": summaries,
    }
    write_summary(args.output_dir, payload)
    print(json.dumps({"summary": str(args.output_dir / "rollout_summary.json")}, indent=2))


if __name__ == "__main__":
    main()
