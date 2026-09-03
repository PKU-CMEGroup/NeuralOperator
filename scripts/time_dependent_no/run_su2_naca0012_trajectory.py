"""Run the provenance-bound solver-only SU2 NACA0012 trajectory pilot."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utility.time_dependent_no.su2_naca_trajectory import (
    NACA_TRAJECTORY_DEFAULT_FINAL_TIME_ITER,
    TrajectoryRunError,
    run_su2_naca0012_trajectory,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resource-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--su2-executable", type=Path, required=True)
    parser.add_argument("--expected-executable-sha256", required=True)
    parser.add_argument(
        "--final-time-iter",
        type=int,
        default=NACA_TRAJECTORY_DEFAULT_FINAL_TIME_ITER,
    )
    parser.add_argument("--timeout-seconds", type=float, default=86400.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        receipt = run_su2_naca0012_trajectory(
            resource_dir=args.resource_dir,
            manifest_path=args.manifest,
            case_dir=args.case_dir,
            target_path=args.target,
            executable_path=args.su2_executable,
            expected_executable_sha256=args.expected_executable_sha256,
            final_time_iter=args.final_time_iter,
            timeout_seconds=args.timeout_seconds,
        )
    except (TrajectoryRunError, OSError, TypeError, ValueError) as error:
        receipt_path = getattr(error, "receipt_path", None)
        location = f" Receipt: {receipt_path}" if receipt_path is not None else ""
        print(f"NACA0012 trajectory failed: {error}.{location}", file=sys.stderr)
        return 1
    print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
