"""Execute and evaluate one frozen SU2 NACA0012 native replay case."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utility.time_dependent_no.su2_native_replay import (
    NativeReplayError,
    run_native_naca0012_replay,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--su2-executable", type=Path, required=True)
    parser.add_argument("--expected-executable-sha256", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        receipt = run_native_naca0012_replay(
            case_dir=args.case_dir,
            target_path=args.target,
            executable_path=args.su2_executable,
            expected_executable_sha256=args.expected_executable_sha256,
            timeout_seconds=args.timeout_seconds,
        )
    except NativeReplayError as error:
        location = f" Receipt: {error.receipt_path}" if error.receipt_path else ""
        print(f"Native replay failed: {error}.{location}", file=sys.stderr)
        return 1
    print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
