
"""Autoregressive PCNO rollouts on CPG-style Euler HDF5 trajectories."""

from __future__ import annotations

import argparse
import csv
import json
import math
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

from pcno.pcno import PCNO  # noqa: E402
try:  # noqa: E402
    from pcno.pcno import euler2d_PCNO  # type: ignore[attr-defined]  # noqa: E402
except ImportError:  # pragma: no cover - compatibility with older cloud checkout
    class euler2d_PCNO(PCNO):
        """PCNO with positive Euler density and pressure output channels."""

        def forward(self, x: torch.Tensor, aux: tuple[torch.Tensor, ...]) -> torch.Tensor:
            out = super().forward(x, aux)
            if out.shape[-1] < 2:
                raise ValueError("euler2d_PCNO requires at least two output channels")
            return torch.cat([torch.exp(out[..., :1]), out[..., 1:-1], torch.exp(out[..., -1:])], dim=-1)
from utility.time_dependent_no.euler2d import (  # noqa: E402
    EulerNodeType,
    PRIMITIVE_NAMES,
    load_cpg_primitive_sequence,
    make_cpg_graph_frame,
)
from utility.time_dependent_no.pcno_adapter import make_pcno_euler7_frame_batch  # noqa: E402

EPS = 1.0e-12


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    date = datetime.now().strftime("%Y%m%d")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-file", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path(f"artifacts/time_dependent_no/pcno_cpg_rollout_{date}"))
    parser.add_argument("--trajectory-indices", nargs="+", type=int, default=[0, 6, 11, 13, 17])
    parser.add_argument("--total-steps", type=int, default=80)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--boundary-mode", choices=("official-clamped", "free"), default="official-clamped")
    parser.add_argument("--rollout-mode", choices=("autoregressive", "teacher-forced"), default="autoregressive")
    parser.add_argument("--node-rho-policy", choices=("ones", "node_weights"), default="ones")
    parser.add_argument("--rcond", type=float, default=1.0e-3)
    return parser.parse_args(argv)


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


def load_state_dict(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    if not isinstance(state, dict):
        raise TypeError(f"checkpoint must contain a state_dict, got {type(state).__name__}")
    return {str(k).removeprefix("module."): v for k, v in state.items()}


def build_model(checkpoint: Path, device: torch.device) -> torch.nn.Module:
    state = load_state_dict(checkpoint, device)
    if "modes" not in state:
        raise KeyError("checkpoint state_dict is missing 'modes'")
    modes = state["modes"].to(device=device, dtype=torch.float32)
    in_dim = int(state["fc0.weight"].shape[1])
    out_dim = int(state["fc2.weight"].shape[0])
    fc_dim = int(state["fc1.weight"].shape[0]) if "fc1.weight" in state else 0
    layer_count = 1 + len([key for key in state if key.startswith("ws.") and key.endswith(".weight")])
    width = int(state["fc0.weight"].shape[0])
    layers = [width] * layer_count
    model = euler2d_PCNO(
        ndims=int(modes.shape[1]),
        modes=modes,
        nmeasures=int(modes.shape[2]),
        layers=layers,
        fc_dim=fc_dim,
        in_dim=in_dim,
        out_dim=out_dim,
        act="gelu",
    ).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def to_tensors(batch: Any, device: torch.device) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    x = torch.as_tensor(batch.x, dtype=torch.float32, device=device)
    aux = (
        torch.as_tensor(batch.node_mask, dtype=torch.float32, device=device),
        torch.as_tensor(batch.nodes, dtype=torch.float32, device=device),
        torch.as_tensor(batch.node_weights, dtype=torch.float32, device=device),
        torch.as_tensor(batch.directed_edges, dtype=torch.long, device=device),
        torch.as_tensor(batch.edge_gradient_weights, dtype=torch.float32, device=device),
    )
    return x, aux


def trajectory_keys(handle: h5py.File) -> list[str]:
    return sorted(handle.keys())


def normal_mask(frame: dict[str, np.ndarray]) -> np.ndarray:
    return np.asarray(frame["node_type"], dtype=np.int64) == int(EulerNodeType.NORMAL)


def rollout_trajectory(
    model: torch.nn.Module,
    group: Any,
    trajectory_index: int,
    key: str,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    current_override: np.ndarray | None = None
    first_frame: dict[str, np.ndarray] | None = None
    nonfinite_step: int | None = None
    available_steps = int(load_cpg_primitive_sequence(group).shape[0]) - int(args.start_frame) - 1
    if available_steps <= 0:
        raise ValueError(
            f"no valid one-step transitions for start_frame={args.start_frame}; "
            f"available_steps={available_steps}"
        )
    step_count = min(int(args.total_steps), available_steps)
    if step_count < int(args.total_steps):
        print(
            f"trajectory {trajectory_index}: requested {args.total_steps} steps, "
            f"capping to {step_count} valid transitions",
            flush=True,
        )
    with torch.no_grad():
        for offset in range(step_count):
            frame_index = args.start_frame + offset
            frame = make_cpg_graph_frame(group, frame_index, num_steps=1)
            if first_frame is None:
                first_frame = frame
            current = frame["current_primitives"].copy() if current_override is None else current_override.copy()
            target = np.asarray(frame["target_primitives"], dtype=np.float32)
            mask = normal_mask(frame)
            boundary = ~mask
            if args.boundary_mode == "official-clamped":
                current[boundary] = target[boundary]
            batch = make_pcno_euler7_frame_batch(
                frame,
                current_primitives=current,
                node_rho_policy=args.node_rho_policy,
                rcond=args.rcond,
            )
            x, aux = to_tensors(batch, device)
            pred = model(x, aux).detach().cpu().numpy()[0].astype(np.float32)
            if args.boundary_mode == "official-clamped":
                pred[boundary] = target[boundary]
            if not np.all(np.isfinite(pred)) and nonfinite_step is None:
                nonfinite_step = int(offset)
            predictions.append(pred)
            targets.append(target)
            current_override = pred if args.rollout_mode == "autoregressive" else None
    if first_frame is None:
        raise RuntimeError("no frames were rolled out")
    pred_arr = np.stack(predictions, axis=0)
    target_arr = np.stack(targets, axis=0)
    geometry = {
        "pos": np.asarray(first_frame["pos"], dtype=np.float32),
        "edges": np.asarray(first_frame["edges"], dtype=np.int64),
        "node_type": np.asarray(first_frame["node_type"], dtype=np.int64),
    }
    summary = summarize_rollout(
        pred_arr,
        target_arr,
        geometry["node_type"],
        trajectory_index=trajectory_index,
        trajectory_key=key,
        nonfinite_step=nonfinite_step,
    )
    summary["requested_step_count"] = int(args.total_steps)
    summary["available_step_count"] = int(available_steps)
    return summary, pred_arr, target_arr, geometry


def summarize_rollout(
    pred: np.ndarray,
    target: np.ndarray,
    node_type: np.ndarray,
    *,
    trajectory_index: int,
    trajectory_key: str,
    nonfinite_step: int | None,
) -> dict[str, Any]:
    normal = node_type == int(EulerNodeType.NORMAL)
    boundary = ~normal
    err = pred[:, normal, :] - target[:, normal, :]
    rmse_time = np.sqrt(np.nanmean(err * err, axis=1))
    rel_time = relative_l2_by_time(pred[:, normal, :], target[:, normal, :])
    out = {
        "trajectory_index": int(trajectory_index),
        "trajectory_key": trajectory_key,
        "step_count": int(pred.shape[0]),
        "normal_node_count": int(np.count_nonzero(normal)),
        "boundary_node_count": int(np.count_nonzero(boundary)),
        "nonfinite_step": nonfinite_step,
        "all_finite": bool(np.all(np.isfinite(pred))),
        "overall_rmse": by_var(np.sqrt(np.nanmean(err * err, axis=(0, 1)))),
        "final_rmse": by_var(rmse_time[-1]),
        "overall_relative_l2": by_var(np.nanmean(rel_time, axis=0)),
        "final_relative_l2": by_var(rel_time[-1]),
        "min_density": float(np.nanmin(pred[:, normal, 0])),
        "min_pressure": float(np.nanmin(pred[:, normal, 3])),
        "nonpositive_density_count": int(np.count_nonzero(pred[:, normal, 0] <= 0.0)),
        "nonpositive_pressure_count": int(np.count_nonzero(pred[:, normal, 3] <= 0.0)),
    }
    if np.any(boundary):
        b_err = pred[:, boundary, :] - target[:, boundary, :]
        out["boundary_max_abs_error"] = float(np.nanmax(np.abs(b_err)))
        out["boundary_rmse"] = by_var(np.sqrt(np.nanmean(b_err * b_err, axis=(0, 1))))
    else:
        out["boundary_max_abs_error"] = 0.0
        out["boundary_rmse"] = by_var(np.zeros(4))
    return out


def relative_l2_by_time(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    err = pred - target
    numerator = np.sqrt(np.nansum(err * err, axis=1))
    denominator = np.sqrt(np.nansum(target * target, axis=1))
    return numerator / np.maximum(denominator, EPS)


def by_var(values: Sequence[float] | np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {name: float(arr[index]) for index, name in enumerate(PRIMITIVE_NAMES)}


def flatten_summary(row: dict[str, Any]) -> dict[str, Any]:
    out = {
        "trajectory_index": row["trajectory_index"],
        "trajectory_key": row["trajectory_key"],
        "step_count": row["step_count"],
        "normal_node_count": row["normal_node_count"],
        "boundary_node_count": row["boundary_node_count"],
        "nonfinite_step": row["nonfinite_step"],
        "all_finite": row["all_finite"],
        "boundary_max_abs_error": row["boundary_max_abs_error"],
        "min_density": row["min_density"],
        "min_pressure": row["min_pressure"],
        "nonpositive_density_count": row["nonpositive_density_count"],
        "nonpositive_pressure_count": row["nonpositive_pressure_count"],
    }
    for section in ("overall_rmse", "final_rmse", "overall_relative_l2", "final_relative_l2", "boundary_rmse"):
        for name in PRIMITIVE_NAMES:
            out[f"{section}_{name}"] = row[section][name]
    return out


def write_rollout_file(path: Path, pred: np.ndarray, target: np.ndarray, geometry: dict[str, np.ndarray], summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("predicteds", data=pred, compression="gzip", compression_opts=4)
        handle.create_dataset("targets", data=target, compression="gzip", compression_opts=4)
        handle.create_dataset("pos", data=geometry["pos"])
        handle.create_dataset("edges", data=geometry["edges"])
        handle.create_dataset("node_type", data=geometry["node_type"])
        for key, value in summary.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                handle.attrs[key] = "" if value is None else value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"trajectory_count": int(len(summaries))}
    for section in ("overall_rmse", "final_rmse", "overall_relative_l2", "final_relative_l2"):
        out[section] = {}
        for name in PRIMITIVE_NAMES:
            values = np.array([item[section][name] for item in summaries], dtype=np.float64)
            out[section][name] = float(np.nanmean(values))
    out["all_finite"] = bool(all(item["all_finite"] for item in summaries))
    out["min_density"] = float(np.nanmin([item["min_density"] for item in summaries]))
    out["min_pressure"] = float(np.nanmin([item["min_pressure"] for item in summaries]))
    out["max_boundary_abs_error"] = float(np.nanmax([item["boundary_max_abs_error"] for item in summaries]))
    return out


def write_report(path: Path, payload: dict[str, Any]) -> None:
    agg = payload["aggregate"]
    lines = [
        "# PCNO CPG Autoregressive Rollout",
        "",
        f"Generated: {payload['generated_at']}",
        f"Checkpoint: `{payload['checkpoint_name']}`",
        f"Boundary mode: `{payload['boundary_mode']}`",
        f"Rollout mode: `{payload['rollout_mode']}`",
        f"Node rho policy: `{payload['node_rho_policy']}`",
        "",
        "## Aggregate RMSE",
        "| Metric | rho | v1 | v2 | pres |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for section in ("overall_rmse", "final_rmse", "overall_relative_l2", "final_relative_l2"):
        vals = agg[section]
        lines.append(
            f"| {section} | {vals['rho']:.6g} | {vals['v1']:.6g} | {vals['v2']:.6g} | {vals['pres']:.6g} |"
        )
    lines += [
        "",
        "## Stability",
        f"- All finite: `{agg['all_finite']}`",
        f"- Minimum predicted density: `{agg['min_density']:.6g}`",
        f"- Minimum predicted pressure: `{agg['min_pressure']:.6g}`",
        f"- Max boundary absolute error: `{agg['max_boundary_abs_error']:.6g}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Path):
        return value.name
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.total_steps <= 0:
        raise SystemExit("--total-steps must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_dir = args.output_dir / "result"
    device = select_device(args)
    model = build_model(args.checkpoint, device)
    start = time.time()
    summaries: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []

    with h5py.File(args.dataset_file, "r") as handle:
        keys = trajectory_keys(handle)
        for trajectory_index in args.trajectory_indices:
            if trajectory_index < 0 or trajectory_index >= len(keys):
                raise IndexError(f"trajectory index {trajectory_index} outside [0, {len(keys)})")
            key = keys[trajectory_index]
            summary, pred, target, geometry = rollout_trajectory(
                model,
                handle[key],
                trajectory_index,
                key,
                args,
                device,
            )
            result_path = result_dir / f"{trajectory_index + 1}.h5"
            write_rollout_file(result_path, pred, target, geometry, summary)
            summary["result_file"] = str(result_path)
            summaries.append(summary)
            rows.append(flatten_summary(summary))
            print(
                f"trajectory {trajectory_index}: pres_rmse={summary['overall_rmse']['pres']:.6g} "
                f"final_pres={summary['final_rmse']['pres']:.6g} finite={summary['all_finite']}",
                flush=True,
            )

    payload = {
        "mode": "pcno_cpg_rollout",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_seconds": float(time.time() - start),
        "checkpoint_name": args.checkpoint.name,
        "dataset_file_name": args.dataset_file.name,
        "device": str(device),
        "boundary_mode": args.boundary_mode,
        "rollout_mode": args.rollout_mode,
        "node_rho_policy": args.node_rho_policy,
        "rcond": float(args.rcond),
        "start_frame": int(args.start_frame),
        "total_steps": int(args.total_steps),
        "trajectory_indices": [int(value) for value in args.trajectory_indices],
        "aggregate": aggregate(summaries),
        "trajectories": summaries,
    }
    (args.output_dir / "rollout_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=json_default),
        encoding="utf-8",
    )
    write_csv(args.output_dir / "per_trajectory_metrics.csv", rows)
    write_report(args.output_dir / "rollout_report.md", payload)
    print(json.dumps({"summary": str(args.output_dir / "rollout_summary.json"), "result_dir": str(result_dir)}, indent=2))


if __name__ == "__main__":
    main()
