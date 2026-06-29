from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utility.time_dependent_no.euler2d import make_cpg_graph_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load one real CPG Euler HDF5 frame through the branch reader."
    )
    parser.add_argument("path", type=Path, help="Path to train.h5 or test.h5")
    parser.add_argument("--trajectory", default=None, help="Trajectory group name")
    parser.add_argument("--frame", type=int, default=0, help="Input frame index")
    parser.add_argument(
        "--num-steps",
        type=int,
        default=3,
        help="Future steps to materialize in future_primitives",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import h5py  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise RuntimeError("h5py is required for real HDF5 smoke checks") from exc

    with h5py.File(args.path, "r") as handle:
        trajectory = args.trajectory or sorted(handle.keys())[0]
        frame = make_cpg_graph_frame(
            handle[trajectory],
            args.frame,
            num_steps=args.num_steps,
        )
    payload = summarize_frame(args.path, trajectory, args.frame, args.num_steps, frame)

    print(f"file: {payload['file']}")
    print(f"trajectory: {payload['trajectory']}")
    print(f"frame: {payload['frame']}  num_steps: {payload['num_steps']}")
    print(f"nodes: {payload['num_nodes']}  edges: {payload['num_edges']}")
    print(f"x: {payload['arrays']['x']['shape']}  y: {payload['arrays']['y']['shape']}")
    print(f"future_primitives: {payload['arrays']['future_primitives']['shape']}")
    print(f"node_type_counts: {payload['node_type_counts']}")
    print(f"current rho/pres min: {payload['current_min']['rho']:.6g}, {payload['current_min']['pres']:.6g}")
    print(f"target rho/pres min: {payload['target_min']['rho']:.6g}, {payload['target_min']['pres']:.6g}")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True))
        print(f"saved: {args.output}")


def summarize_frame(
    path: Path,
    trajectory: str,
    frame_index: int,
    num_steps: int,
    frame: dict[str, np.ndarray],
) -> dict[str, Any]:
    node_type = np.asarray(frame["node_type"], dtype=np.int64)
    unique, counts = np.unique(node_type, return_counts=True)
    current = np.asarray(frame["current_primitives"], dtype=np.float64)
    target = np.asarray(frame["target_primitives"], dtype=np.float64)
    return {
        "kind": "cpg_graph_frame_smoke",
        "file": str(path),
        "trajectory": trajectory,
        "frame": int(frame_index),
        "num_steps": int(num_steps),
        "num_nodes": int(frame["pos"].shape[0]),
        "num_edges": int(frame["edges"].shape[0]),
        "arrays": {name: _array_summary(value) for name, value in frame.items()},
        "node_type_counts": {
            int(key): int(value) for key, value in zip(unique, counts, strict=True)
        },
        "current_min": _primitive_min(current),
        "current_max": _primitive_max(current),
        "target_min": _primitive_min(target),
        "target_max": _primitive_max(target),
    }


def _primitive_min(value: np.ndarray) -> dict[str, float]:
    return {
        "rho": float(np.min(value[:, 0])),
        "v1": float(np.min(value[:, 1])),
        "v2": float(np.min(value[:, 2])),
        "pres": float(np.min(value[:, 3])),
    }


def _primitive_max(value: np.ndarray) -> dict[str, float]:
    return {
        "rho": float(np.max(value[:, 0])),
        "v1": float(np.max(value[:, 1])),
        "v2": float(np.max(value[:, 2])),
        "pres": float(np.max(value[:, 3])),
    }


def _array_summary(value: np.ndarray) -> dict[str, Any]:
    array = np.asarray(value)
    summary: dict[str, Any] = {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
    }
    if np.issubdtype(array.dtype, np.number) and array.size:
        finite = np.isfinite(array.astype(np.float64, copy=False))
        summary.update(
            {
                "finite_fraction": float(np.mean(finite)),
                "min": float(np.min(array)),
                "max": float(np.max(array)),
                "mean": float(np.mean(array)),
            }
        )
    return summary


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


if __name__ == "__main__":
    main()
