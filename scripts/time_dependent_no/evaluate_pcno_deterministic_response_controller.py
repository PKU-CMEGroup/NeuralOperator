#!/usr/bin/env python3
"""D078: repeat D077 under deterministic CUDA execution without changing gates."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.time_dependent_no.evaluate_pcno_response_gain_controller as d076
import scripts.time_dependent_no.evaluate_pcno_strength_grouped_response_controller as d077

SCHEMA = "pcno_deterministic_response_controller_diagnostic_v1"
EXPERIMENT_CONTRACT = "d078_deterministic_strength_grouped_response_probe"

D078_SPEC = replace(
    d077.D077_SPEC,
    experiment_id="D078",
    schema=SCHEMA,
    experiment_contract=EXPERIMENT_CONTRACT,
    description=__doc__ or "D078 deterministic response-controller replay",
    method_claim=(
        "D077 strength-grouped response controller under deterministic CUDA "
        "execution; no tolerance or correction change; not data assimilation"
    ),
    extra_source_paths=(*d077.D077_SPEC.extra_source_paths, Path(__file__)),
    deterministic_algorithms=True,
)


def parse_args(argv: Sequence[str] | None = None):
    return d076.parse_args_for_experiment(argv, D078_SPEC)


def run(args):
    return d076.run(args, spec=D078_SPEC)


def main(argv: Sequence[str] | None = None) -> int:
    summary = run(parse_args(argv))
    print(
        f"D078 dynamic_fv status={summary['status']} "
        f"calibration_qualified={summary['calibration_qualification']['passed']} "
        f"promotion={summary['promotion'].get('passed')}",
        flush=True,
    )
    return 0 if summary["contract_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
