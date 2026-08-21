#!/usr/bin/env python3
"""Apply the registered zero-model A33-R1 routing-check rescore."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.time_dependent_no import (
    evaluate_pcno_affine_shadow_tether_query_grid as a33,
)
from utility.time_dependent_no.pcno_artifacts import (
    atomic_write_json_with_paths as atomic_write_json,
)
from utility.time_dependent_no.pcno_artifacts import (
    sha256_file,
    sha256_files,
)
from utility.time_dependent_no.pcno_artifacts import (
    write_csv_with_paths as write_csv,
)
from utility.time_dependent_no.pcno_cross_resolution_teacher_forced import (
    verify_payload_sha256,
    with_payload_sha256,
)

WORKING_ID = "W26-L5-P6-RFB19-A33-R1-SP19-QUERY-GRID-ROUTING-RESCORE"
RESULT_SCHEMA = "pcno_sp19_affine_shadow_tether_query_grid_rescore_v1"
SOURCE_SCHEMA = "pcno_sp19_affine_shadow_tether_query_grid_rescore_source_v1"
PRIOR_RESULT_SHA256 = "c80936b24e30181f433ece72acc0de60c12e5c9a8a32bceedfab752400e5834b"
PRIOR_PAYLOAD_SHA256 = (
    "e777727ec378340392b67c250eda197944f938a14e8dbd519233884eca56cb0c"
)
MISWIRED_CHECK = "routing_native_reference_crosscheck"
REPLACEMENT_CHECK = "authenticated_float64_routing_reference_crosscheck"
SOURCE_PATHS = (
    "docs/time_dependent_no/W26_L5_RESPONSE_FILTERED_BLOCK_PREREGISTRATION.md",
    "scripts/time_dependent_no/rescore_pcno_affine_shadow_tether_query_grid.py",
    "tests/time_dependent_no/test_pcno_affine_shadow_tether_query_grid.py",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON payload must be an object: {path}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _verify_prior(path: Path) -> dict[str, Any]:
    if sha256_file(path) != PRIOR_RESULT_SHA256:
        raise ValueError("A33 result differs from the frozen A33-R1 input")
    prior = _read_json(path)
    verify_payload_sha256(prior)
    correction = prior.get("correction_survival_gate", {})
    checks = correction.get("checks", {})
    false_checks = sorted(key for key, value in checks.items() if not bool(value))
    root = path.parent.resolve()
    inventory = prior.get("artifact_inventory_before_summary")
    if (
        prior.get("schema") != a33.RESULT_SCHEMA
        or prior.get("working_id") != a33.WORKING_ID
        or prior.get("payload_sha256") != PRIOR_PAYLOAD_SHA256
        or prior.get("status") != "completed_query_grid_comparator"
        or prior.get("primary_transfer_policy") != a33.RAW_TRANSFER_POLICY
        or correction.get("status") != "failed"
        or correction.get("failed_checks") != [MISWIRED_CHECK]
        or false_checks != [MISWIRED_CHECK]
        or not isinstance(inventory, Mapping)
        or not inventory
        or prior.get("recurrence_executed") is not True
        or prior.get("sealed_population_opened") is not False
    ):
        raise ValueError("A33 is not the exact stopped A33-R1 prerequisite")
    for relative, expected in inventory.items():
        candidate = (root / str(relative)).resolve()
        if root not in candidate.parents or not isinstance(expected, Mapping):
            raise ValueError("A33 artifact inventory contains an unsafe path")
        if (
            not candidate.is_file()
            or sha256_file(candidate) != expected.get("sha256")
            or candidate.stat().st_size != int(expected.get("bytes", -1))
        ):
            raise ValueError(f"A33 artifact differs: {relative}")
    return prior


def _routing_check(root: Path) -> tuple[dict[str, bool], dict[str, Any]]:
    positions = _read_csv(root / "position_decisions.csv")
    references = _read_csv(root / "reference_checks.csv")
    position_by_case = {row.get("case_id", ""): row for row in positions}
    reference_by_case = {row.get("case_id", ""): row for row in references}
    exact_inventory = (
        len(positions) == len(a33.CASE_IDS)
        and len(references) == len(a33.CASE_IDS)
        and set(position_by_case) == set(a33.CASE_IDS)
        and set(reference_by_case) == set(a33.CASE_IDS)
    )
    position_sources = {row.get("source") for row in positions}
    maximum_float64_gap = max(
        float(row["loaded_native_reference_crosscheck_max_abs"]) for row in references
    )
    maximum_float32_audit_gap = max(
        float(row["routing_native_vs_shard_max_abs"]) for row in positions
    )
    checks = {
        "routing_inventory_exact": exact_inventory,
        "all_cases_selected_at_frozen_threshold": exact_inventory
        and all(
            row["position_selected"] == "True"
            and row["expected_selected"] == "True"
            and float(row["normalized_wall_distance"])
            <= a33.FROZEN_POSITION_BUFFER_THRESHOLD
            for row in positions
        ),
        "routing_source_is_authenticated_query_restriction": position_sources
        == {
            "authenticated_query_call_zero_conservative_restriction_before_model_calls"
        },
        REPLACEMENT_CHECK: maximum_float64_gap <= a33.STRICT_TOLERANCE,
        "float32_shard_difference_is_audit_only": maximum_float32_audit_gap
        > a33.STRICT_TOLERANCE,
    }
    return checks, {
        "maximum_float64_routing_reference_crosscheck_abs": maximum_float64_gap,
        "maximum_float32_shard_audit_difference_abs": maximum_float32_audit_gap,
        "position_sources": sorted(str(value) for value in position_sources),
    }


def _amended_correction_gate(
    prior: Mapping[str, Any], routing_checks: Mapping[str, bool]
) -> dict[str, Any]:
    original = prior["correction_survival_gate"]
    checks = dict(original["checks"])
    removed = checks.pop(MISWIRED_CHECK, None)
    if removed is not False:
        raise ValueError("A33 miswired routing check is not the exact false value")
    checks.update(routing_checks)
    failed = sorted(key for key, value in checks.items() if not bool(value))
    return {
        **original,
        "status": "qualified" if not failed else "failed",
        "checks": checks,
        "failed_checks": failed,
        "removed_bookkeeping_check": MISWIRED_CHECK,
        "model_calls": 0,
    }


def _amended_direct_gate(prior: Mapping[str, Any]) -> dict[str, Any]:
    case_rows = prior["case_ratios"][a33.PAIR_DIRECT_CORRECTED]
    controls = [
        row
        for row in _read_csv(Path(prior["_root"]) / "query_controls.csv")
        if row["pair"] == a33.PAIR_DIRECT_CORRECTED
    ]
    structural = {
        key: bool(value)
        for key, value in prior["direct_vs_transfer_gate"]["checks"].items()
        if key
        not in {
            MISWIRED_CHECK,
            "case_ratio_inventory_exact",
            "median_endpoint_ratio_at_most_0p95",
            "at_least_four_case_wins",
            "all_controls_no_harm",
            "endpoint_denominators_resolved",
        }
    }
    structural[REPLACEMENT_CHECK] = True
    return a33._direct_gate(
        case_rows=case_rows,
        controls=controls,
        structural_checks=structural,
    )


def rescore(prior_path: Path, output_dir: Path) -> tuple[dict[str, Any], int]:
    prior = _verify_prior(prior_path)
    prior["_root"] = str(prior_path.parent.resolve())
    routing_checks, observed = _routing_check(prior_path.parent)
    correction = _amended_correction_gate(prior, routing_checks)
    direct = (
        _amended_direct_gate(prior) if correction["status"] == "qualified" else None
    )
    qualified = correction["status"] == "qualified"
    output_dir.mkdir(parents=True, exist_ok=False)
    source = with_payload_sha256(
        {
            "schema": SOURCE_SCHEMA,
            "working_id": WORKING_ID,
            "source_sha256": sha256_files(SOURCE_PATHS, root=ROOT),
            "prior_result_sha256": sha256_file(prior_path),
            "prior_result_payload_sha256": prior["payload_sha256"],
            "prior_artifact_inventory": prior["artifact_inventory_before_summary"],
            "repair": {
                "scope": "routing_check_and_primary_direct_gate_rescore_only",
                "removed_check": MISWIRED_CHECK,
                "replacement_check": REPLACEMENT_CHECK,
                "model_calls": 0,
                "checkpoint_or_dataset_loaded": False,
                "reference_or_rollout_arrays_loaded": False,
                "predictions_recomputed": False,
                "recurrence_reexecuted": False,
                "metrics_or_controls_recomputed": False,
                "coefficient_selector_threshold_or_schedule_changed": False,
            },
        }
    )
    write_csv(
        output_dir / "amended_checks.csv",
        [
            {"gate": "correction_survival", "check": key, "passed": value}
            for key, value in sorted(correction["checks"].items())
        ]
        + (
            [
                {"gate": "direct_vs_transfer", "check": key, "passed": value}
                for key, value in sorted(direct["checks"].items())
            ]
            if direct is not None
            else []
        ),
    )
    atomic_write_json(output_dir / "source_manifest.json", source)
    artifact_hashes = sha256_files(
        ("amended_checks.csv", "source_manifest.json"), root=output_dir
    )
    result = with_payload_sha256(
        {
            "schema": RESULT_SCHEMA,
            "working_id": WORKING_ID,
            "status": (
                "qualified_correction_direct_failed"
                if qualified and direct is not None and direct["status"] == "failed"
                else "qualified_correction_and_direct"
                if qualified and direct is not None and direct["status"] == "qualified"
                else "failed_rescore"
            ),
            "correction_survival_gate": correction,
            "direct_vs_transfer_gate": direct,
            "primary_transfer_policy": (
                a33.CORRECTED_TRANSFER_POLICY if qualified else a33.RAW_TRANSFER_POLICY
            ),
            "direct_gate_pair": (
                a33.PAIR_DIRECT_CORRECTED if qualified else a33.PAIR_DIRECT_RAW
            ),
            "routing_observed": observed,
            "original_failed_correction_checks": prior["correction_survival_gate"][
                "failed_checks"
            ],
            "original_direct_gate_pair": prior["direct_gate_pair"],
            "original_direct_gate": prior["direct_vs_transfer_gate"],
            "case_ratios": {
                "correction": prior["case_ratios"][a33.PAIR_CORRECTION],
                "direct_vs_primary": prior["case_ratios"][
                    a33.PAIR_DIRECT_CORRECTED if qualified else a33.PAIR_DIRECT_RAW
                ],
            },
            "maximum_control_ratio": {
                "correction": prior["maximum_control_ratio"][a33.PAIR_CORRECTION],
                "direct_vs_primary": prior["maximum_control_ratio"][
                    a33.PAIR_DIRECT_CORRECTED if qualified else a33.PAIR_DIRECT_RAW
                ],
            },
            "failed_control_count": {
                "correction": len(prior["failed_controls"][a33.PAIR_CORRECTION]),
                "direct_vs_primary": len(
                    prior["failed_controls"][
                        a33.PAIR_DIRECT_CORRECTED if qualified else a33.PAIR_DIRECT_RAW
                    ]
                ),
            },
            "primary_scores": prior["primary_scores"],
            "error_cost_comparison": prior["error_cost_comparison"],
            "execution": prior["execution"],
            "closure_maxima": prior["closure_maxima"],
            "prior_result_sha256": sha256_file(prior_path),
            "prior_result_payload_sha256": prior["payload_sha256"],
            "prior_artifact_inventory": prior["artifact_inventory_before_summary"],
            "source_manifest_sha256": sha256_file(output_dir / "source_manifest.json"),
            "source_manifest_payload_sha256": source["payload_sha256"],
            "artifact_hashes": artifact_hashes,
            "model_calls": 0,
            "checkpoint_or_dataset_loaded": False,
            "reference_or_rollout_arrays_loaded": False,
            "predictions_recomputed": False,
            "recurrence_reexecuted": False,
            "metrics_or_controls_recomputed": False,
            "coefficient_selector_threshold_or_schedule_changed": False,
            "sealed_population_opened": False,
            "claim_boundary": prior["claim_boundary"],
        }
    )
    atomic_write_json(output_dir / "query_grid_rescore.json", result)
    return result, 0 if qualified else 4


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload, exit_code = rescore(args.prior_result, args.output_dir)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
