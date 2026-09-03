#!/usr/bin/env python3
"""Analyze the frozen solver-only SU2 NACA phase pilot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utility.time_dependent_no.su2_naca_phase_pilot import (
    analyze_su2_naca0012_phase_pilot,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and analyze one accumulated SU2 NACA trajectory prefix."
    )
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--trajectory-receipt", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    receipt = analyze_su2_naca0012_phase_pilot(
        contract_path=arguments.contract,
        trajectory_receipt_path=arguments.trajectory_receipt,
        mesh_path=arguments.mesh,
        history_path=arguments.history,
        output_path=arguments.output,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))
    return 2 if receipt["outcome"] == "INVALID_ARTIFACT" else 0


if __name__ == "__main__":
    sys.exit(main())
