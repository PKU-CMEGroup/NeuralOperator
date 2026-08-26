#!/usr/bin/env python3
"""Run the bounded CPU/FP64 P0-A1 restart-sufficiency preflight."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utility.time_dependent_no.p0_restart_sufficiency import run_a1_preflight


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs-root",
        type=Path,
        required=True,
        help="Immutable recovered A0 inputs root.",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Frozen A1 source root containing every manifested Python file.",
    )
    parser.add_argument("--a1-source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--owner-authorized-a1",
        action="store_true",
        help="Required acknowledgement of the owner's explicit P0-A1 authorization.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary, summary_path = run_a1_preflight(
        inputs_root=args.inputs_root,
        source_root=args.source_root,
        a1_source_manifest_path=args.a1_source_manifest,
        output_dir=args.output_dir,
        owner_authorized_a1=args.owner_authorized_a1,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "stage": summary["stage"],
                "payload_sha256": summary["payload_sha256"],
                "summary": summary_path.name,
                "solver_input_hashes": {
                    case["trajectory_id"]: case["prepared_solver_input_sha256"]
                    for case in summary["cases"]
                },
                "activity": summary["activity"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
