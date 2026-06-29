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
from utility.time_dependent_no.pcno_adapter import make_pcno_frame_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a no-training PCNO tensor batch from one CPG Euler frame."
    )
    parser.add_argument("path", type=Path, help="Path to train.h5 or test.h5")
    parser.add_argument("--trajectory", default=None, help="Trajectory group name")
    parser.add_argument("--frame", type=int, default=0, help="Input frame index")
    parser.add_argument("--num-steps", type=int, default=1)
    parser.add_argument("--rcond", type=float, default=1.0e-3)
    parser.add_argument(
        "--exclude-positions-from-x",
        action="store_true",
        help="Use only [node_type, rho, v1, v2, pres, Mach] as PCNO input features.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import h5py  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise RuntimeError("h5py is required for real HDF5 adapter smoke checks") from exc

    with h5py.File(args.path, "r") as handle:
        trajectory = args.trajectory or sorted(handle.keys())[0]
        frame = make_cpg_graph_frame(handle[trajectory], args.frame, num_steps=args.num_steps)
    batch = make_pcno_frame_batch(
        frame,
        include_positions_in_x=not args.exclude_positions_from_x,
        rcond=args.rcond,
    )
    payload = {
        "kind": "cpg_pcno_adapter_smoke",
        "file": str(args.path),
        "trajectory": trajectory,
        "frame": int(args.frame),
        "num_steps": int(args.num_steps),
        "metadata": batch.metadata,
        "arrays": {
            "x": _array_summary(batch.x),
            "y": _array_summary(batch.y),
            "node_mask": _array_summary(batch.node_mask),
            "nodes": _array_summary(batch.nodes),
            "node_weights": _array_summary(batch.node_weights),
            "directed_edges": _array_summary(batch.directed_edges),
            "edge_gradient_weights": _array_summary(batch.edge_gradient_weights),
        },
        "node_weight_sum": float(np.sum(batch.node_weights)),
        "edge_gradient_abs_max": float(np.max(np.abs(batch.edge_gradient_weights)))
        if batch.edge_gradient_weights.size
        else 0.0,
    }

    print(f"file: {payload['file']}")
    print(f"trajectory: {payload['trajectory']}  frame: {payload['frame']}")
    print(f"x: {payload['arrays']['x']['shape']}  y: {payload['arrays']['y']['shape']}")
    print(f"nodes: {payload['arrays']['nodes']['shape']}")
    print(f"node_weights: {payload['arrays']['node_weights']['shape']} sum={payload['node_weight_sum']:.6f}")
    print(f"directed_edges: {payload['arrays']['directed_edges']['shape']}")
    print(f"edge_gradient_weights: {payload['arrays']['edge_gradient_weights']['shape']}")
    print(f"metadata: {payload['metadata']}")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True))
        print(f"saved: {args.output}")


def _array_summary(value: np.ndarray) -> dict[str, Any]:
    array = np.asarray(value)
    summary: dict[str, Any] = {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
    }
    if np.issubdtype(array.dtype, np.number) and array.size:
        sample = array.reshape(-1)
        if sample.size > 500_000:
            sample = sample[:500_000]
        sample = sample.astype(np.float64, copy=False)
        finite = np.isfinite(sample)
        summary.update(
            {
                "finite_fraction": float(np.mean(finite)),
                "min": float(np.min(sample)),
                "max": float(np.max(sample)),
                "mean": float(np.mean(sample)),
            }
        )
    return summary


if __name__ == "__main__":
    main()
