"""State-distribution drift diagnostics for CPG-style Euler rollouts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utility.time_dependent_no.euler2d import PRIMITIVE_NAMES, EulerNodeType  # noqa: E402

EPS = 1.0e-12


def parse_named_path(raw: str) -> tuple[str, Path]:
    if "=" in raw:
        name, value = raw.split("=", 1)
        return name.strip(), Path(value)
    path = Path(raw)
    return path.name, path


def by_var(values: Sequence[float] | np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {name: float(arr[index]) for index, name in enumerate(PRIMITIVE_NAMES)}


def read_frame_or_static(dataset: h5py.Dataset, frame: int) -> np.ndarray:
    if len(dataset.shape) >= 3:
        return np.asarray(dataset[frame])
    return np.asarray(dataset)


def squeeze_node_column(array: np.ndarray) -> np.ndarray:
    if array.ndim > 1 and array.shape[-1] == 1:
        return np.squeeze(array, axis=-1)
    return array


def result_files(path: Path) -> list[tuple[int, Path]]:
    if path.is_file():
        index = int(path.stem) - 1 if path.stem.isdigit() else 0
        return [(index, path)]
    directory = path / "result" if (path / "result").is_dir() else path
    files = sorted(
        directory.glob("*.h5"),
        key=lambda p: int(p.stem) if p.stem.isdigit() else p.name,
    )
    out = []
    for offset, file in enumerate(files):
        index = int(file.stem) - 1 if file.stem.isdigit() else offset
        out.append((index, file))
    return out


def stack_frame(group: h5py.Group, frame: int) -> np.ndarray:
    return np.stack(
        [
            np.asarray(group[name][frame], dtype=np.float64).reshape(-1)
            for name in PRIMITIVE_NAMES
        ],
        axis=-1,
    )


def normal_mask(group: h5py.Group, frame: int, nodes: int) -> np.ndarray:
    node_type = squeeze_node_column(
        read_frame_or_static(group["node_type"], frame)
    ).astype(np.int64)
    if node_type.shape[0] != nodes:
        return np.ones(nodes, dtype=bool)
    normal = node_type == int(EulerNodeType.NORMAL)
    return normal if np.any(normal) else np.ones(nodes, dtype=bool)


def train_stats(
    train_file: Path, max_trajectories: int, frame_stride: int
) -> dict[str, Any]:
    total = np.zeros(len(PRIMITIVE_NAMES), dtype=np.float64)
    total_sq = np.zeros(len(PRIMITIVE_NAMES), dtype=np.float64)
    min_value = np.full(len(PRIMITIVE_NAMES), np.inf, dtype=np.float64)
    max_value = np.full(len(PRIMITIVE_NAMES), -np.inf, dtype=np.float64)
    count = 0
    sampled = []
    with h5py.File(train_file, "r") as handle:
        keys = list(handle.keys())[:max_trajectories]
        for key in keys:
            group = handle[key]
            steps = int(group[PRIMITIVE_NAMES[0]].shape[0])
            for frame in range(0, steps, frame_stride):
                values = stack_frame(group, frame)
                mask = normal_mask(group, frame, values.shape[0])
                selected = values[mask]
                total += np.sum(selected, axis=0)
                total_sq += np.sum(selected * selected, axis=0)
                min_value = np.minimum(min_value, np.min(selected, axis=0))
                max_value = np.maximum(max_value, np.max(selected, axis=0))
                count += selected.shape[0]
                sampled.append(
                    {
                        "trajectory": key,
                        "frame": int(frame),
                        "nodes": int(selected.shape[0]),
                    }
                )
    mean = total / max(count, 1)
    variance = total_sq / max(count, 1) - mean * mean
    std = np.sqrt(np.maximum(variance, 0.0))
    return {
        "normal_node_count": int(count),
        "mean": mean,
        "std": std,
        "min": min_value,
        "max": max_value,
        "sampled_frames": sampled,
    }


def load_rollout(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        return (
            np.asarray(handle["predicteds"], dtype=np.float64),
            np.asarray(handle["targets"], dtype=np.float64),
        )


def evaluate_sequence(
    values: np.ndarray, stats: dict[str, Any], mask: np.ndarray
) -> dict[str, np.ndarray]:
    selected = values[:, mask, :]
    mean = np.asarray(stats["mean"], dtype=np.float64)
    std = np.asarray(stats["std"], dtype=np.float64)
    min_value = np.asarray(stats["min"], dtype=np.float64)
    max_value = np.asarray(stats["max"], dtype=np.float64)
    z = np.abs(
        (selected - mean.reshape(1, 1, -1)) / np.maximum(std.reshape(1, 1, -1), EPS)
    )
    below = selected < min_value.reshape(1, 1, -1)
    above = selected > max_value.reshape(1, 1, -1)
    outside = below | above
    return {
        "mean": np.mean(selected, axis=1),
        "std": np.std(selected, axis=1),
        "max_abs_z": np.max(z, axis=1),
        "mean_abs_z": np.mean(z, axis=1),
        "outside_range_fraction": np.mean(outside, axis=1),
        "below_range_fraction": np.mean(below, axis=1),
        "above_range_fraction": np.mean(above, axis=1),
    }


def finite_stats(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"mean": math.nan, "max": math.nan, "final": math.nan}
    return {
        "mean": float(np.mean(finite)),
        "max": float(np.max(finite)),
        "final": float(arr[-1]),
    }


def summarize_eval(prefix: str, metrics: dict[str, np.ndarray]) -> dict[str, Any]:
    out = {}
    for key in (
        "max_abs_z",
        "mean_abs_z",
        "outside_range_fraction",
        "below_range_fraction",
        "above_range_fraction",
    ):
        out[f"{prefix}_{key}"] = {
            name: finite_stats(metrics[key][:, index])
            for index, name in enumerate(PRIMITIVE_NAMES)
        }
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument(
        "--run", action="append", required=True, help="NAME=run_dir_or_result_h5"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-max-trajectories", type=int, default=30)
    parser.add_argument("--train-frame-stride", type=int, default=10)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    stats = train_stats(
        args.train_file, args.train_max_trajectories, args.train_frame_stride
    )
    summary: dict[str, Any] = {
        "train_stats": {
            "normal_node_count": stats["normal_node_count"],
            "mean": by_var(stats["mean"]),
            "std": by_var(stats["std"]),
            "min": by_var(stats["min"]),
            "max": by_var(stats["max"]),
            "num_sampled_frames": len(stats["sampled_frames"]),
            "sample_policy": {
                "train_max_trajectories": args.train_max_trajectories,
                "train_frame_stride": args.train_frame_stride,
            },
        },
        "runs": {},
        "trajectories": {},
    }
    rows: list[dict[str, Any]] = []
    with h5py.File(args.test_file, "r") as test:
        test_keys = list(test.keys())
        for raw in args.run:
            run_name, run_path = parse_named_path(raw)
            run_files = result_files(run_path)
            traj_summaries = []
            for trajectory_index, file in run_files:
                pred, target = load_rollout(file)
                group = test[test_keys[trajectory_index]]
                mask = normal_mask(group, 0, pred.shape[1])
                pred_eval = evaluate_sequence(pred, stats, mask)
                target_eval = evaluate_sequence(target, stats, mask)
                item = {
                    "trajectory_index": int(trajectory_index),
                    "file": str(file),
                    "normal_node_count": int(np.count_nonzero(mask)),
                    **summarize_eval("prediction", pred_eval),
                    **summarize_eval("target", target_eval),
                }
                traj_summaries.append(item)
                for step in range(pred.shape[0]):
                    row = {
                        "run": run_name,
                        "trajectory_index": int(trajectory_index),
                        "step": int(step + 1),
                    }
                    for index, name in enumerate(PRIMITIVE_NAMES):
                        row[f"prediction_outside_{name}"] = float(
                            pred_eval["outside_range_fraction"][step, index]
                        )
                        row[f"target_outside_{name}"] = float(
                            target_eval["outside_range_fraction"][step, index]
                        )
                        row[f"prediction_max_abs_z_{name}"] = float(
                            pred_eval["max_abs_z"][step, index]
                        )
                        row[f"target_max_abs_z_{name}"] = float(
                            target_eval["max_abs_z"][step, index]
                        )
                    rows.append(row)
            summary["trajectories"][run_name] = traj_summaries
            summary["runs"][run_name] = {
                "trajectory_count": len(traj_summaries),
                "mean_prediction_final_outside_range_fraction": {
                    name: float(
                        np.mean(
                            [
                                t["prediction_outside_range_fraction"][name]["final"]
                                for t in traj_summaries
                            ]
                        )
                    )
                    for name in PRIMITIVE_NAMES
                },
                "mean_target_final_outside_range_fraction": {
                    name: float(
                        np.mean(
                            [
                                t["target_outside_range_fraction"][name]["final"]
                                for t in traj_summaries
                            ]
                        )
                    )
                    for name in PRIMITIVE_NAMES
                },
                "mean_prediction_final_max_abs_z": {
                    name: float(
                        np.mean(
                            [
                                t["prediction_max_abs_z"][name]["final"]
                                for t in traj_summaries
                            ]
                        )
                    )
                    for name in PRIMITIVE_NAMES
                },
                "mean_target_final_max_abs_z": {
                    name: float(
                        np.mean(
                            [
                                t["target_max_abs_z"][name]["final"]
                                for t in traj_summaries
                            ]
                        )
                    )
                    for name in PRIMITIVE_NAMES
                },
            }

    summary_path = args.output_dir / "state_drift_summary.json"
    per_time_path = args.output_dir / "state_drift_per_time.csv"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=json_default),
        encoding="utf-8",
    )
    write_csv(per_time_path, rows)
    print(
        json.dumps(
            {"summary": str(summary_path), "per_time": str(per_time_path)}, indent=2
        )
    )


if __name__ == "__main__":
    main()
