from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utility.time_dependent_no.euler2d import inspect_cpg_hdf5_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect a CPG-style 2D Euler HDF5 dataset file."
    )
    parser.add_argument("path", type=Path, help="Path to train.h5/test.h5")
    parser.add_argument(
        "--max-trajectories",
        type=int,
        default=3,
        help="Number of trajectory groups to inspect; use -1 for all.",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Skip numeric min/max sampling and report only shapes/dtypes.",
    )
    parser.add_argument(
        "--max-values-per-array",
        type=int,
        default=500_000,
        help="Maximum sampled values per array for range statistics.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    max_trajectories = None if args.max_trajectories < 0 else args.max_trajectories
    summary = inspect_cpg_hdf5_file(
        args.path,
        max_trajectories=max_trajectories,
        sample_values=not args.metadata_only,
        max_values_per_array=args.max_values_per_array,
    )
    payload = summary.to_dict()

    print(f"file: {args.path}")
    print(f"trajectories: {summary.num_trajectories}")
    print(f"inspected: {summary.inspected_trajectories}")
    for trajectory in summary.trajectories:
        print(
            f"- {trajectory.name}: T={trajectory.num_time_steps}, "
            f"N={trajectory.num_nodes}, E={trajectory.num_edges}, "
            f"missing={trajectory.missing_keys or 'none'}"
        )
        if trajectory.node_type_counts:
            print(f"  node_type_counts: {trajectory.node_type_counts}")
        for warning in trajectory.warnings:
            print(f"  warning: {warning}")
    if summary.warnings:
        print("warnings:")
        for warning in summary.warnings:
            print(f"- {warning}")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True))
        print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
