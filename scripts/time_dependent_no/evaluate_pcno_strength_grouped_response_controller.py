#!/usr/bin/env python3
"""D077: qualify strength-grouped response control for native PCNO correction."""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.time_dependent_no.evaluate_pcno_response_gain_controller as d076

SCHEMA = "pcno_strength_grouped_response_controller_diagnostic_v1"
EXPERIMENT_CONTRACT = "d077_strength_grouped_response_probe"
CALIBRATION_ENERGY_INDICES = (1, 2, 3, 4, 5, 7, 8, 9, 10)
EXPECTED_GROUP_MEMBERS = {
    f"sv_e{energy:02d}": (
        f"sv_e{energy:02d}_y00",
        f"sv_e{energy:02d}_y08",
    )
    for energy in CALIBRATION_ENERGY_INDICES
}


def _strength_group_id(case_id: str) -> str:
    try:
        prefix, position = case_id.rsplit("_", maxsplit=1)
    except ValueError as error:
        raise ValueError(f"D077 case_id has no position suffix: {case_id}") from error
    if prefix not in EXPECTED_GROUP_MEMBERS or position not in {"y00", "y08"}:
        raise ValueError(f"D077 case lies outside frozen calibration groups: {case_id}")
    return prefix


def build_strength_group_map(
    cases: Sequence[d076.CaseData],
) -> Mapping[str, str]:
    """Bind complete y00/y08 pairs to immutable physical-strength provenance."""

    case_by_id = {str(case.case_id): case for case in cases}
    if not case_by_id or len(case_by_id) != len(cases):
        raise ValueError("D077 requires nonempty unique calibration cases")
    observed_ids = set(case_by_id)
    if not observed_ids <= set(d076.parent.DYNAMIC_CALIBRATION_CASES):
        raise ValueError("D077 grouping received a non-calibration case")

    group_by_case = {
        case_id: _strength_group_id(case_id) for case_id in sorted(case_by_id)
    }
    observed_groups = sorted(set(group_by_case.values()))
    for group_id in observed_groups:
        expected_members = set(EXPECTED_GROUP_MEMBERS[group_id])
        members = {
            case_id
            for case_id, observed_group in group_by_case.items()
            if observed_group == group_id
        }
        if members != expected_members:
            raise ValueError(
                f"D077 strength group is not the complete y00/y08 pair: {group_id}"
            )
        strengths = []
        positions = []
        for case_id in sorted(members):
            provenance = case_by_id[case_id].provenance
            if (
                provenance.get("case_id") != case_id
                or provenance.get("split") != "validation"
            ):
                raise ValueError("D077 group does not bind validation provenance")
            parameters = provenance.get("parameters")
            if not isinstance(parameters, dict):
                raise TypeError("D077 group lacks manifest parameter provenance")
            try:
                strengths.append(float(parameters["vortex_epsilon"]))
                positions.append(float(parameters["vortex_y"]))
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("D077 group lacks finite vortex provenance") from error
        if (
            not np.isfinite(strengths).all()
            or not np.allclose(strengths, strengths[0], rtol=1.0e-12, atol=0.0)
            or not np.isfinite(positions).all()
            or len(set(positions)) != 2
        ):
            raise ValueError("D077 group mixes strength or duplicates position")
    return group_by_case


D077_SPEC = d076.ResponseExperimentSpec(
    experiment_id="D077",
    schema=SCHEMA,
    experiment_contract=EXPERIMENT_CONTRACT,
    description=__doc__ or "D077 strength-grouped response controller",
    smoke_calibration_case_count=6,
    group_builder=build_strength_group_map,
    group_contract="paired_physical_vortex_strength",
    require_group_amplitude_match=True,
    require_highest_group_abstention=True,
    null_not_run_evaluation=True,
    method_claim=(
        "offline rank-8 bias and five-call response selector cross-fit by paired "
        "physical strength with target-free upper-support abstention; not data "
        "assimilation"
    ),
    extra_source_paths=(Path(__file__),),
)


def parse_args(argv: Sequence[str] | None = None):
    return d076.parse_args_for_experiment(argv, D077_SPEC)


def run(args):
    return d076.run(args, spec=D077_SPEC)


def main(argv: Sequence[str] | None = None) -> int:
    summary = run(parse_args(argv))
    print(
        f"D077 dynamic_fv status={summary['status']} "
        f"calibration_qualified={summary['calibration_qualification']['passed']} "
        f"promotion={summary['promotion'].get('passed')}"
    )
    return 0 if summary["contract_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
