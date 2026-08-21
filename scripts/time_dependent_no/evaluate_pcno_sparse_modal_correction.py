#!/usr/bin/env python3
"""Evaluate the W26-L5 SP19 fine-discrepancy correction.

This entry point extends the provenance-bound P2 evaluator without modifying
it. Calibration creates an immutable sparse-mask contract. Teacher-forced
evaluation must pass before the synchronized H30 rollout command is accepted.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.time_dependent_no import (
    evaluate_pcno_cross_resolution_teacher_forced as collection,
)
from scripts.time_dependent_no import (
    evaluate_pcno_fine_discrepancy_correction as parent,
)
from utility.time_dependent_no.pcno_artifacts import (
    atomic_write_json_with_paths as atomic_write_json,
)
from utility.time_dependent_no.pcno_artifacts import (
    git_state,
    sha256_file,
    sha256_files,
)
from utility.time_dependent_no.pcno_artifacts import (
    write_csv_with_paths as write_csv,
)
from utility.time_dependent_no.pcno_cross_resolution_teacher_forced import (
    ALL_INPUT_CALLS,
    EVALUATION_CASE_IDS,
    error_relation_rows,
    verify_payload_sha256,
    with_payload_sha256,
)
from utility.time_dependent_no.pcno_fine_discrepancy_correction import (
    MAX_CORRECTION_TO_NATIVE_INCREMENT,
    modal_coordinates,
    score_candidate_inventory,
)
from utility.time_dependent_no.pcno_sparse_modal_correction import (
    FROZEN_ACTIVE_CELLS,
    SPARSE_POLICY,
    masked_snapshot,
    positive_half_skill_cells,
    synchronized_sparse_modal_step,
)

WORKING_ID = "W26-L5-P3-SP19-A2"
MASK_SCHEMA = "pcno_sparse_modal_mask_contract_v1"
SOURCE_SCHEMA = "pcno_sparse_modal_source_manifest_v1"
TEACHER_SCHEMA = "pcno_sparse_modal_teacher_evaluation_v1"
ROLLOUT_SCHEMA = "pcno_sparse_modal_rollout_v1"
INTERNAL_POLICY = "rank7_fine_away_half"
EARLY_CALLS = tuple(range(15))
LATE_CALLS = tuple(range(15, 30))
RESOLUTION_CONTRACT = parent.RESOLUTION_CONTRACT
NATIVE_RESOLUTION = parent.NATIVE_RESOLUTION

EXPECTED_PARENT_CALIBRATION_SHA256 = (
    "d9533285d89e70e61c16833ceeb27cd542cd65f44925a746c94786df26faf8cd"
)
EXPECTED_MODAL_SUMMARY_SHA256 = (
    "732e3f4e224c6c232dba059d820434075a94a1cd9e50da7bad75a5224fffb097"
)
EXPECTED_MODAL_RECORDS_SHA256 = (
    "c16df1dca5b77fc3b760a0e92a2e35dbd4adaa1910fc1d0998ded12b8d1103bb"
)
EXPECTED_PARENT_SOURCE_SHA256 = (
    "8b3e193b1f2424991bc943009c042e14fdf758f85f170500df7963be0e442b93"
)

SOURCE_PATHS = (
    "docs/time_dependent_no/W26_L5_SPARSE_MODAL_RECURRENCE_PREREGISTRATION.md",
    "utility/time_dependent_no/pcno_sparse_modal_correction.py",
    "scripts/time_dependent_no/evaluate_pcno_sparse_modal_correction.py",
    "tests/time_dependent_no/test_pcno_sparse_modal_correction.py",
)


def _teacher_gate(checks: Mapping[str, bool]) -> dict[str, Any]:
    expected = {
        "mask_contract_exact",
        "parent_calibration_qualified",
        "inventory_exact",
        "full_skill_nonnegative",
        "rank8_skill_at_least_0p05",
        "minimum_four_case_wins",
        "both_half_horizons_positive",
        "all_controls_no_harm",
        "all_proposals_finite_admissible",
        "all_audits_resolved",
        "mask_closure_passed",
        "closure_passed",
        "source_and_artifacts_exact",
    }
    if set(checks) != expected or any(
        type(value) is not bool for value in checks.values()
    ):
        raise ValueError("SP19 teacher gate evidence inventory differs")
    passed = all(checks.values())
    return {
        "status": "recurrent_pilot_authorized" if passed else "stopped",
        "recurrent_pilot_authorized": passed,
        "fresh_confirmation": False,
        "sealed_population_authorized": False,
        "checks": dict(checks),
    }


def _recurrent_gate(checks: Mapping[str, bool]) -> dict[str, Any]:
    expected = {
        "teacher_gate_passed",
        "mask_contract_exact",
        "inventory_exact",
        "all_rollouts_complete_finite_admissible",
        "median_endpoint_ratio_at_most_0p98",
        "minimum_four_endpoint_wins",
        "maximum_endpoint_ratio_at_most_1p02",
        "aggregate_state_rms_ratio_at_most_0p99",
        "increment_and_cumulative_ratios_at_most_one",
        "all_controls_no_harm",
        "two_call_common_source_closure",
        "correction_audits_pass",
        "mask_closure_passed",
        "deterministic_prefix_exact",
        "source_and_artifacts_exact",
    }
    if set(checks) != expected or any(
        type(value) is not bool for value in checks.values()
    ):
        raise ValueError("SP19 recurrent gate evidence inventory differs")
    passed = all(checks.values())
    return {
        "status": "adaptive_recurrent_pass" if passed else "adaptive_recurrent_failed",
        "adaptive_recurrent_pass": passed,
        "fresh_confirmation": False,
        "sealed_population_authorized": False,
        "checks": dict(checks),
    }


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON payload must be an object: {path}")
    return value


def _git_status_short(paths: Sequence[str]) -> list[str]:
    try:
        result = subprocess.run(
            ["git", "status", "--short", "--", *paths],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return ["unknown"]
    return result.stdout.splitlines() if result.returncode == 0 else ["unknown"]


def _relabel_policy(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            **dict(row),
            "policy": (
                SPARSE_POLICY
                if row.get("policy") == INTERNAL_POLICY
                else row.get("policy")
            ),
        }
        for row in rows
    ]


def _modal_contract_statistics(
    summary_rows: Sequence[Mapping[str, Any]],
    modal_records_path: Path,
    active_cells: Sequence[tuple[int, int]],
) -> dict[str, float]:
    active = set(active_cells)
    unmasked_skill = 0.0
    masked_skill = 0.0
    removed_target_energy = 0.0
    for row in summary_rows:
        mode = int(row["mode_index"])
        component = int(row["component"])
        if mode == 0:
            continue
        energy = float(row["target_energy_fraction"])
        contribution = energy * float(row["fixed_half_skill"])
        unmasked_skill += contribution
        if (mode, component) in active:
            masked_skill += contribution
        else:
            removed_target_energy += energy

    total_feature_square = 0.0
    retained_feature_square = 0.0
    with modal_records_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            mode = int(row["mode_index"])
            component = int(row["component"])
            if mode == 0:
                continue
            square = float(row["fine_coordinate"]) ** 2
            total_feature_square += square
            if (mode, component) in active:
                retained_feature_square += square
    if total_feature_square <= 0.0:
        raise ValueError("calibration fine-discrepancy energy is unresolved")
    return {
        "unmasked_rank8_skill": unmasked_skill,
        "masked_rank8_skill": masked_skill,
        "removed_target_energy_fraction": removed_target_energy,
        "retained_feature_energy_fraction": (
            retained_feature_square / total_feature_square
        ),
    }


def create_mask_contract(args: argparse.Namespace) -> dict[str, Any]:
    calibration = parent._require_calibration(args.parent_calibration)
    if sha256_file(args.parent_calibration) != EXPECTED_PARENT_CALIBRATION_SHA256:
        raise ValueError("parent calibration file differs from the preregistration")
    if calibration.get("selection", {}).get("selected_policy") != INTERNAL_POLICY:
        raise ValueError("parent calibration did not select the rank-7 half-gain rule")
    if sha256_file(args.modal_summary) != EXPECTED_MODAL_SUMMARY_SHA256:
        raise ValueError("calibration modal summary differs from the preregistration")
    if sha256_file(args.modal_records) != EXPECTED_MODAL_RECORDS_SHA256:
        raise ValueError("calibration modal records differ from the preregistration")
    if (
        calibration.get("artifact_hashes", {}).get("calibration_modal_summary.json")
        != EXPECTED_MODAL_SUMMARY_SHA256
        or calibration.get("artifact_hashes", {}).get("calibration_modal_records.csv")
        != EXPECTED_MODAL_RECORDS_SHA256
    ):
        raise ValueError("parent calibration does not authenticate the modal inputs")

    summary = _read_json(args.modal_summary)
    verify_payload_sha256(summary)
    rows = summary.get("mode_component_rows")
    if not isinstance(rows, list):
        raise TypeError("modal summary has no mode-component table")
    active_cells = positive_half_skill_cells(rows)
    if active_cells != FROZEN_ACTIVE_CELLS:
        raise ValueError("derived positive-skill mask differs from SP19")
    statistics = _modal_contract_statistics(
        rows,
        args.modal_records,
        active_cells,
    )
    checks = {
        "parent_calibration_qualified": calibration.get("status") == "qualified",
        "exactly_19_active_cells": len(active_cells) == 19,
        "constant_mode_excluded": all(mode > 0 for mode, _ in active_cells),
        "masked_skill_exceeds_unmasked": (
            statistics["masked_rank8_skill"] > statistics["unmasked_rank8_skill"]
        ),
        "retained_energy_strictly_between_zero_and_one": (
            0.0 < statistics["retained_feature_energy_fraction"] < 1.0
        ),
    }
    payload = with_payload_sha256(
        {
            "schema": MASK_SCHEMA,
            "working_id": WORKING_ID,
            "status": "passed" if all(checks.values()) else "failed",
            "population_status": "adaptive_open_validation",
            "policy": SPARSE_POLICY,
            "gain": -0.5,
            "maximum_relative_norm": MAX_CORRECTION_TO_NATIVE_INCREMENT,
            "active_cells": [list(cell) for cell in active_cells],
            "statistics": statistics,
            "checks": checks,
            "parent_calibration_sha256": sha256_file(args.parent_calibration),
            "parent_calibration_payload_sha256": calibration["payload_sha256"],
            "modal_summary_sha256": sha256_file(args.modal_summary),
            "modal_summary_payload_sha256": summary["payload_sha256"],
            "modal_records_sha256": sha256_file(args.modal_records),
            "evaluation_data_used_for_selection": False,
            "claim_boundary": (
                "Calibration-derived family-local structural mask; no fresh, "
                "cross-family, conservation, or sealed claim."
            ),
        }
    )
    atomic_write_json(args.output, payload)
    return payload


def _require_mask_contract(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    verify_payload_sha256(payload)
    active = tuple(
        tuple(int(value) for value in cell) for cell in payload.get("active_cells", [])
    )
    if (
        payload.get("schema") != MASK_SCHEMA
        or payload.get("status") != "passed"
        or payload.get("policy") != SPARSE_POLICY
        or active != FROZEN_ACTIVE_CELLS
        or payload.get("parent_calibration_sha256")
        != EXPECTED_PARENT_CALIBRATION_SHA256
        or payload.get("modal_summary_sha256") != EXPECTED_MODAL_SUMMARY_SHA256
        or payload.get("modal_records_sha256") != EXPECTED_MODAL_RECORDS_SHA256
    ):
        raise ValueError("sparse modal mask contract differs from the preregistration")
    return payload


def _extended_source_manifest(
    args: argparse.Namespace,
    *,
    parent_source: Mapping[str, Any],
    mask_contract: Mapping[str, Any],
) -> dict[str, Any]:
    return with_payload_sha256(
        {
            "schema": SOURCE_SCHEMA,
            "working_id": WORKING_ID,
            "git": git_state(ROOT),
            "source_hashes": sha256_files(SOURCE_PATHS, root=ROOT),
            "source_status": _git_status_short(SOURCE_PATHS),
            "parent_source_manifest_sha256": sha256_file(args.parent_source_manifest),
            "parent_source_payload_sha256": parent_source["payload_sha256"],
            "mask_contract_sha256": sha256_file(args.mask_contract),
            "mask_contract_payload_sha256": mask_contract["payload_sha256"],
        }
    )


def _verify_extended_source(
    path: Path,
    args: argparse.Namespace,
    *,
    parent_source: Mapping[str, Any],
    mask_contract: Mapping[str, Any],
) -> dict[str, Any]:
    recorded = _read_json(path)
    verify_payload_sha256(recorded)
    current = _extended_source_manifest(
        args,
        parent_source=parent_source,
        mask_contract=mask_contract,
    )
    if recorded != current:
        raise ValueError("current SP19 source differs from the frozen source manifest")
    return recorded


def _verify_parent_inputs(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    calibration = parent._require_calibration(args.parent_calibration)
    if sha256_file(args.parent_calibration) != EXPECTED_PARENT_CALIBRATION_SHA256:
        raise ValueError("parent calibration file differs from the preregistration")
    if sha256_file(args.parent_source_manifest) != EXPECTED_PARENT_SOURCE_SHA256:
        raise ValueError("parent source manifest differs from the preregistration")
    parent_source = parent._verify_source_manifest(args.parent_source_manifest, args)
    if calibration["source_manifest_payload_sha256"] != parent_source["payload_sha256"]:
        raise ValueError("parent calibration and source identities differ")
    return calibration, parent_source


def _masked_snapshots(snapshots: Sequence[Any], runtime: Any) -> list[Any]:
    return [
        masked_snapshot(
            snapshot,
            runtime.native_projector,
            active_cells=FROZEN_ACTIVE_CELLS,
        )
        for snapshot in snapshots
    ]


def _mask_closure(
    field: np.ndarray,
    projector: Any,
    *,
    component_scale: np.ndarray,
) -> dict[str, Any]:
    coordinates = modal_coordinates(
        field,
        projector,
        component_scale=component_scale,
    )
    active = set(FROZEN_ACTIVE_CELLS)
    inactive_values = [
        abs(float(coordinates[mode, component]))
        for mode in range(coordinates.shape[0])
        for component in range(coordinates.shape[1])
        if (mode, component) not in active
    ]
    return {
        "maximum_inactive_coordinate_abs": max(inactive_values, default=0.0),
        "maximum_constant_coordinate_abs": float(np.max(np.abs(coordinates[0]))),
        "active_cell_count": len(active),
    }


def run_teacher(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    mask_contract = _require_mask_contract(args.mask_contract)
    calibration, frozen_parent_source = _verify_parent_inputs(args)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=False)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    runtime, runtime_parent_source = parent._build_runtime(args)
    try:
        if runtime_parent_source != frozen_parent_source:
            raise ValueError("teacher runtime differs from the parent source manifest")
        source = _extended_source_manifest(
            args,
            parent_source=frozen_parent_source,
            mask_contract=mask_contract,
        )
        atomic_write_json(output_dir / "source_manifest.json", source)
        collected = collection._collect(
            runtime,
            case_ids=EVALUATION_CASE_IDS,
            input_calls=ALL_INPUT_CALLS,
            coefficients=None,
            phase="sparse_modal_teacher_evaluation",
        )
        snapshots = _masked_snapshots(collected.snapshots, runtime)
        mask_closure_rows = [
            {
                "case_id": snapshot.case_id,
                "input_call": int(snapshot.input_call),
                **_mask_closure(
                    snapshot.basis.fine_minus_native,
                    runtime.native_projector,
                    component_scale=snapshot.component_scale,
                ),
            }
            for snapshot in snapshots
        ]
        controls = parent._reconstruct_controls(
            runtime,
            snapshots,
            policies=(INTERNAL_POLICY,),
        )
        all_payload = score_candidate_inventory(
            snapshots,
            label="evaluation_all",
            expected_case_ids=EVALUATION_CASE_IDS,
            expected_input_calls=ALL_INPUT_CALLS,
            policy=INTERNAL_POLICY,
            resolution=NATIVE_RESOLUTION,
            projector=runtime.native_projector,
        )
        early_payload = score_candidate_inventory(
            [row for row in snapshots if row.input_call in EARLY_CALLS],
            label="evaluation_early",
            expected_case_ids=EVALUATION_CASE_IDS,
            expected_input_calls=EARLY_CALLS,
            policy=INTERNAL_POLICY,
            resolution=NATIVE_RESOLUTION,
            projector=runtime.native_projector,
        )
        late_payload = score_candidate_inventory(
            [row for row in snapshots if row.input_call in LATE_CALLS],
            label="evaluation_late",
            expected_case_ids=EVALUATION_CASE_IDS,
            expected_input_calls=LATE_CALLS,
            policy=INTERNAL_POLICY,
            resolution=NATIVE_RESOLUTION,
            projector=runtime.native_projector,
        )
        control_rows = parent._build_teacher_controls(
            policy=INTERNAL_POLICY,
            score_payload=all_payload,
            controls=controls,
            case_ids=EVALUATION_CASE_IDS,
            input_calls=ALL_INPUT_CALLS,
        )
        closure_pass, closure = parent._candidate_closure(
            policy=INTERNAL_POLICY,
            score_payloads=(all_payload, early_payload, late_payload),
            audit_rows=controls["audit"],
            collection_maxima=collected.maxima,
        )
        full = parent._population_row(all_payload, "full")
        rank8 = parent._population_row(all_payload, "rank8_parallel")
        early_rank8 = parent._population_row(early_payload, "rank8_parallel")
        late_rank8 = parent._population_row(late_payload, "rank8_parallel")
        full_cases = {
            row["case_id"]: row for row in parent._case_rows(all_payload, "full")
        }
        rank8_cases = {
            row["case_id"]: row
            for row in parent._case_rows(all_payload, "rank8_parallel")
        }
        case_wins = sum(
            full_cases[case_id]["skill_status"] == "ok"
            and rank8_cases[case_id]["skill_status"] == "ok"
            and float(full_cases[case_id]["skill_vs_zero"]) >= 0.0
            and float(rank8_cases[case_id]["skill_vs_zero"]) > 0.0
            for case_id in EVALUATION_CASE_IDS
        )
        proposals = controls["proposal"]
        audits = controls["audit"]
        checks = {
            "mask_contract_exact": mask_contract["status"] == "passed",
            "parent_calibration_qualified": calibration["status"] == "qualified",
            "inventory_exact": (
                len(snapshots) == len(EVALUATION_CASE_IDS) * len(ALL_INPUT_CALLS)
                and len(proposals) == len(EVALUATION_CASE_IDS) * len(ALL_INPUT_CALLS)
                and len(audits) == len(EVALUATION_CASE_IDS) * len(ALL_INPUT_CALLS)
            ),
            "full_skill_nonnegative": full["skill_status"] == "ok"
            and float(full["skill_vs_zero"]) >= 0.0,
            "rank8_skill_at_least_0p05": rank8["skill_status"] == "ok"
            and float(rank8["skill_vs_zero"]) >= 0.05,
            "minimum_four_case_wins": case_wins >= 4,
            "both_half_horizons_positive": (
                early_rank8["skill_status"] == "ok"
                and late_rank8["skill_status"] == "ok"
                and float(early_rank8["skill_vs_zero"]) > 0.0
                and float(late_rank8["skill_vs_zero"]) > 0.0
            ),
            "all_controls_no_harm": bool(control_rows)
            and all(parent._control_passed(row) for row in control_rows),
            "all_proposals_finite_admissible": bool(proposals)
            and all(row["finite"] and row["admissible"] for row in proposals),
            "all_audits_resolved": bool(audits)
            and all(row["status"] == "ok" for row in audits),
            "mask_closure_passed": len(mask_closure_rows)
            == len(EVALUATION_CASE_IDS) * len(ALL_INPUT_CALLS)
            and all(
                row["maximum_inactive_coordinate_abs"] <= 1.0e-10
                and row["maximum_constant_coordinate_abs"] <= 1.0e-10
                and row["active_cell_count"] == 19
                for row in mask_closure_rows
            ),
            "closure_passed": closure_pass,
            "source_and_artifacts_exact": runtime_parent_source == frozen_parent_source,
        }
        gate = _teacher_gate(checks)
        modal_path, modal_summary_path, modal_summary = parent._write_modal_outputs(
            output_dir,
            snapshots,
            runtime.native_projector,
            prefix="masked_evaluation",
        )
        score_rows = _relabel_policy(
            [
                *({"policy": INTERNAL_POLICY, **row} for row in all_payload["rows"]),
                *({"policy": INTERNAL_POLICY, **row} for row in early_payload["rows"]),
                *({"policy": INTERNAL_POLICY, **row} for row in late_payload["rows"]),
            ]
        )
        control_rows = _relabel_policy(control_rows)
        front_rows = _relabel_policy(controls["front"])
        integral_rows = _relabel_policy(controls["integral"])
        proposals = _relabel_policy(proposals)
        audits = _relabel_policy(audits)
        write_csv(output_dir / "teacher_scores.csv", score_rows)
        write_csv(output_dir / "teacher_controls.csv", control_rows)
        write_csv(output_dir / "teacher_front_rows.csv", front_rows)
        write_csv(output_dir / "teacher_integral_rows.csv", integral_rows)
        write_csv(output_dir / "teacher_proposals.csv", proposals)
        write_csv(output_dir / "teacher_audits.csv", audits)
        write_csv(output_dir / "teacher_mask_closure.csv", mask_closure_rows)
        write_csv(
            output_dir / "teacher_error_relations_early.csv",
            error_relation_rows(
                [row for row in snapshots if row.input_call < 20],
                cell="case_only",
                resolution=NATIVE_RESOLUTION,
                projector=runtime.native_projector,
            ),
        )
        write_csv(
            output_dir / "teacher_error_relations_late.csv",
            error_relation_rows(
                [row for row in snapshots if row.input_call >= 20],
                cell="joint_held_out",
                resolution=NATIVE_RESOLUTION,
                projector=runtime.native_projector,
            ),
        )
        write_csv(output_dir / "transfer_floors.csv", collected.floor_rows)
        write_csv(output_dir / "reference_checks.csv", collected.reference_rows)
        files = (
            "source_manifest.json",
            "teacher_scores.csv",
            "teacher_controls.csv",
            "teacher_front_rows.csv",
            "teacher_integral_rows.csv",
            "teacher_proposals.csv",
            "teacher_audits.csv",
            "teacher_mask_closure.csv",
            "teacher_error_relations_early.csv",
            "teacher_error_relations_late.csv",
            "masked_evaluation_modal_records.csv",
            "masked_evaluation_modal_summary.json",
            "transfer_floors.csv",
            "reference_checks.csv",
        )
        artifact_hashes = sha256_files(files, root=output_dir)
        payload = with_payload_sha256(
            {
                "schema": TEACHER_SCHEMA,
                "working_id": WORKING_ID,
                "status": (
                    "qualified" if gate["recurrent_pilot_authorized"] else "stopped"
                ),
                "population_status": "adaptive_open_validation",
                "selected_policy": SPARSE_POLICY,
                "teacher_gate": gate,
                "population_scores": {
                    "all_full": dict(full),
                    "all_rank8": dict(rank8),
                    "early_rank8": dict(early_rank8),
                    "late_rank8": dict(late_rank8),
                    "case_win_count": case_wins,
                },
                "failed_controls": [
                    row for row in control_rows if not parent._control_passed(row)
                ],
                "closure": closure,
                "mask_contract_sha256": sha256_file(args.mask_contract),
                "mask_contract_payload_sha256": mask_contract["payload_sha256"],
                "masked_modal_summary_payload_sha256": modal_summary["payload_sha256"],
                "masked_modal_records_sha256": sha256_file(modal_path),
                "masked_modal_summary_sha256": sha256_file(modal_summary_path),
                "parent_calibration_sha256": sha256_file(args.parent_calibration),
                "parent_calibration_payload_sha256": calibration["payload_sha256"],
                "parent_source_manifest_sha256": sha256_file(
                    args.parent_source_manifest
                ),
                "parent_source_payload_sha256": frozen_parent_source["payload_sha256"],
                "source_manifest_sha256": sha256_file(
                    output_dir / "source_manifest.json"
                ),
                "source_manifest_payload_sha256": source["payload_sha256"],
                "execution": collected.execution,
                "collection_maxima": collected.maxima,
                "artifact_hashes": artifact_hashes,
                "recurrence_executed": False,
                "claim_boundary": (
                    "Adaptive teacher-forced dynamic-FV mechanism result; no fresh, "
                    "cross-family, conservation, or sealed claim."
                ),
            }
        )
        atomic_write_json(output_dir / "teacher_evaluation.json", payload)
        return payload, 0 if gate["recurrent_pilot_authorized"] else 4
    finally:
        collection._close_runtime(runtime)


def _require_teacher(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    verify_payload_sha256(payload)
    if (
        payload.get("schema") != TEACHER_SCHEMA
        or payload.get("status") != "qualified"
        or payload.get("selected_policy") != SPARSE_POLICY
        or payload.get("teacher_gate", {}).get("recurrent_pilot_authorized") is not True
    ):
        raise ValueError("SP19 teacher gate did not authorize recurrence")
    expected = payload.get("artifact_hashes")
    if (
        not isinstance(expected, Mapping)
        or sha256_files(tuple(expected), root=path.parent) != expected
    ):
        raise ValueError("SP19 teacher artifact inventory or hashes differ")
    return payload


@contextmanager
def sparse_rollout_driver(
    closure_rows: list[dict[str, Any]] | None = None,
) -> Iterator[None]:
    """Temporarily inject SP19 into the frozen metric-complete rollout driver."""

    original = parent.synchronized_fine_discrepancy_step

    def stepper(
        native_state,
        *,
        contract,
        projector,
        predictor,
        policy,
        volumes,
        component_scale,
    ):
        if policy != INTERNAL_POLICY:
            raise ValueError("SP19 adapter accepts only the rank-7 driver token")
        step = synchronized_sparse_modal_step(
            native_state,
            contract=contract,
            projector=projector,
            predictor=predictor,
            policy=SPARSE_POLICY,
            volumes=volumes,
            component_scale=component_scale,
            active_cells=FROZEN_ACTIVE_CELLS,
        )
        if closure_rows is not None:
            closure_rows.append(
                _mask_closure(
                    step.correction,
                    projector,
                    component_scale=np.asarray(component_scale, dtype=np.float64),
                )
            )
        return step

    parent.synchronized_fine_discrepancy_step = stepper
    try:
        yield
    finally:
        parent.synchronized_fine_discrepancy_step = original


def _relabel_rollout(rollout: dict[str, Any]) -> dict[str, Any]:
    rollout["policy"] = SPARSE_POLICY
    for row in rollout["rows"]:
        row["policy"] = SPARSE_POLICY
    return rollout


def _rollout_cost(
    rows: Sequence[Mapping[str, Any]],
    rollouts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    raw_execution = [row["execution"] for row in rollouts if row["policy"] == "zero"]
    corrected_execution = [
        row["execution"] for row in rollouts if row["policy"] == SPARSE_POLICY
    ]
    cost = {
        "raw_logical_model_calls": sum(
            row["logical_model_calls"] for row in raw_execution
        ),
        "corrected_logical_model_calls": sum(
            row["logical_model_calls"] for row in corrected_execution
        ),
        "raw_native_logical_calls": sum(
            row["native_logical_calls"] for row in raw_execution
        ),
        "corrected_native_logical_calls": sum(
            row["native_logical_calls"] for row in corrected_execution
        ),
        "corrected_fine_logical_calls": sum(
            row["fine_logical_calls"] for row in corrected_execution
        ),
        "raw_forward_seconds": sum(
            row["total_forward_seconds"] for row in raw_execution
        ),
        "corrected_forward_seconds": sum(
            row["total_forward_seconds"] for row in corrected_execution
        ),
        "raw_native_forward_seconds": sum(
            row["native_forward_seconds"] for row in raw_execution
        ),
        "corrected_native_forward_seconds": sum(
            row["native_forward_seconds"] for row in corrected_execution
        ),
        "corrected_fine_forward_seconds": sum(
            row["fine_forward_seconds"] for row in corrected_execution
        ),
        "raw_wall_seconds": sum(row["wall_seconds"] for row in raw_execution),
        "corrected_wall_seconds": sum(
            row["wall_seconds"] for row in corrected_execution
        ),
        "raw_peak_memory_bytes": max(
            row["maximum_peak_gpu_memory_bytes"] for row in raw_execution
        ),
        "corrected_peak_memory_bytes": max(
            row["maximum_peak_gpu_memory_bytes"] for row in corrected_execution
        ),
    }
    cost["logical_model_call_ratio"] = (
        cost["corrected_logical_model_calls"] / cost["raw_logical_model_calls"]
    )
    cost["forward_time_ratio"] = (
        cost["corrected_forward_seconds"] / cost["raw_forward_seconds"]
    )
    cost["wall_time_ratio"] = cost["corrected_wall_seconds"] / cost["raw_wall_seconds"]
    cost["peak_memory_ratio"] = (
        cost["corrected_peak_memory_bytes"] / cost["raw_peak_memory_bytes"]
    )
    cost["corrected_fine_forward_time_fraction"] = (
        cost["corrected_fine_forward_seconds"] / cost["corrected_forward_seconds"]
    )
    raw_endpoint = math.sqrt(
        np.mean(
            [
                float(row["state_error"]) ** 2
                for row in rows
                if row["policy"] == "zero"
                and int(row["input_call"]) == ALL_INPUT_CALLS[-1]
            ]
        )
    )
    corrected_endpoint = math.sqrt(
        np.mean(
            [
                float(row["state_error"]) ** 2
                for row in rows
                if row["policy"] == SPARSE_POLICY
                and int(row["input_call"]) == ALL_INPUT_CALLS[-1]
            ]
        )
    )
    cost["raw_endpoint_state_rms"] = raw_endpoint
    cost["corrected_endpoint_state_rms"] = corrected_endpoint
    raw_dominates = (
        raw_endpoint <= corrected_endpoint
        and cost["raw_wall_seconds"] <= cost["corrected_wall_seconds"]
        and cost["raw_peak_memory_bytes"] <= cost["corrected_peak_memory_bytes"]
    )
    raw_strict = (
        raw_endpoint < corrected_endpoint
        or cost["raw_wall_seconds"] < cost["corrected_wall_seconds"]
        or cost["raw_peak_memory_bytes"] < cost["corrected_peak_memory_bytes"]
    )
    cost["corrected_pareto_dominated_by_raw"] = bool(raw_dominates and raw_strict)
    return cost


def run_rollout(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    mask_contract = _require_mask_contract(args.mask_contract)
    teacher = _require_teacher(args.teacher_evaluation)
    calibration, frozen_parent_source = _verify_parent_inputs(args)
    frozen_source = _verify_extended_source(
        args.source_manifest,
        args,
        parent_source=frozen_parent_source,
        mask_contract=mask_contract,
    )
    if teacher["mask_contract_payload_sha256"] != mask_contract["payload_sha256"]:
        raise ValueError("teacher and SP19 mask identities differ")
    if teacher["source_manifest_payload_sha256"] != frozen_source["payload_sha256"]:
        raise ValueError("teacher and SP19 source identities differ")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=False)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    runtime, runtime_parent_source = parent._build_runtime(args)
    try:
        if runtime_parent_source != frozen_parent_source:
            raise ValueError("rollout runtime differs from the parent source manifest")
        with sparse_rollout_driver():
            prefix_a = parent._rollout_arm(
                runtime,
                case_id=EVALUATION_CASE_IDS[0],
                policy=INTERNAL_POLICY,
                horizon=2,
            )
            prefix_b = parent._rollout_arm(
                runtime,
                case_id=EVALUATION_CASE_IDS[0],
                policy=INTERNAL_POLICY,
                horizon=2,
            )
        prefix_abs = max(
            parent._maximum_absolute(left - right)
            for left, right in zip(prefix_a["states"], prefix_b["states"], strict=True)
        )
        rollouts: list[dict[str, Any]] = []
        mask_closure_rows: list[dict[str, Any]] = []
        for case_id in EVALUATION_CASE_IDS:
            print(f"SP19 rollout: {case_id} raw", flush=True)
            rollouts.append(
                parent._rollout_arm(runtime, case_id=case_id, policy="zero", horizon=30)
            )
            print(f"SP19 rollout: {case_id} corrected", flush=True)
            case_mask_closure: list[dict[str, Any]] = []
            with sparse_rollout_driver(case_mask_closure):
                corrected = parent._rollout_arm(
                    runtime,
                    case_id=case_id,
                    policy=INTERNAL_POLICY,
                    horizon=30,
                )
            mask_closure_rows.extend(
                {
                    "case_id": case_id,
                    "input_call": input_call,
                    **row,
                }
                for input_call, row in enumerate(case_mask_closure)
            )
            rollouts.append(_relabel_rollout(corrected))
        rows = [row for rollout in rollouts for row in rollout["rows"]]
        case_rows, control_rows, population = parent._paired_rollout_controls(rows)
        completed = all(
            rollout["execution"]["completed_calls"] == 30
            and rollout["first_invalid_call"] is None
            and rollout["first_nonfinite_call"] is None
            for rollout in rollouts
        )
        corrected_rollouts = [row for row in rollouts if row["policy"] == SPARSE_POLICY]
        two_call_closure = all(
            row["execution"]["native_logical_calls"] == 30
            and row["execution"]["fine_logical_calls"] == 30
            and row["maxima"]["pre_model_nesting"] == 0.0
            and row["maxima"]["recurrence"] <= 1.0e-12
            for row in corrected_rollouts
        )
        audit_pass = all(
            row["maxima"]["correction_excluded"] == 0.0
            and row["maxima"]["correction_modal_reconstruction"] <= 1.0e-10
            and row["maxima"]["correction_ratio"]
            <= MAX_CORRECTION_TO_NATIVE_INCREMENT + 1.0e-12
            and row["maxima"]["correction_mean"] <= 1.0e-12
            for row in corrected_rollouts
        )
        checks = {
            "teacher_gate_passed": teacher["teacher_gate"]["recurrent_pilot_authorized"]
            is True,
            "mask_contract_exact": mask_contract["status"] == "passed",
            "inventory_exact": len(rows)
            == len(EVALUATION_CASE_IDS) * len(ALL_INPUT_CALLS) * 2,
            "all_rollouts_complete_finite_admissible": completed,
            "median_endpoint_ratio_at_most_0p98": population[
                "median_endpoint_state_ratio"
            ]
            <= 0.98,
            "minimum_four_endpoint_wins": population["endpoint_win_count"] >= 4,
            "maximum_endpoint_ratio_at_most_1p02": population[
                "maximum_endpoint_state_ratio"
            ]
            <= 1.02,
            "aggregate_state_rms_ratio_at_most_0p99": population[
                "aggregate_state_rms_ratio"
            ]
            <= 0.99,
            "increment_and_cumulative_ratios_at_most_one": (
                population["aggregate_increment_defect_rms_ratio"] <= 1.0
                and population["median_endpoint_cumulative_defect_ratio"] <= 1.0
            ),
            "all_controls_no_harm": bool(control_rows)
            and all(parent._control_passed(row) for row in control_rows),
            "two_call_common_source_closure": two_call_closure,
            "correction_audits_pass": audit_pass,
            "mask_closure_passed": len(mask_closure_rows)
            == len(EVALUATION_CASE_IDS) * len(ALL_INPUT_CALLS)
            and all(
                row["maximum_inactive_coordinate_abs"] <= 1.0e-10
                and row["maximum_constant_coordinate_abs"] <= 1.0e-10
                and row["active_cell_count"] == 19
                for row in mask_closure_rows
            ),
            "deterministic_prefix_exact": prefix_abs == 0.0,
            "source_and_artifacts_exact": runtime_parent_source == frozen_parent_source,
        }
        gate = _recurrent_gate(checks)
        write_csv(output_dir / "rollout_call_metrics.csv", rows)
        write_csv(output_dir / "rollout_case_metrics.csv", case_rows)
        write_csv(output_dir / "rollout_controls.csv", control_rows)
        write_csv(output_dir / "rollout_mask_closure.csv", mask_closure_rows)
        write_csv(
            output_dir / "rollout_execution.csv",
            [
                {
                    "case_id": row["case_id"],
                    "policy": row["policy"],
                    **row["execution"],
                }
                for row in rollouts
            ],
        )
        write_csv(
            output_dir / "reference_checks.csv",
            [
                {
                    "case_id": row["case_id"],
                    "policy": row["policy"],
                    **row["reference_check"],
                }
                for row in rollouts
            ],
        )
        files = (
            "rollout_call_metrics.csv",
            "rollout_case_metrics.csv",
            "rollout_controls.csv",
            "rollout_mask_closure.csv",
            "rollout_execution.csv",
            "reference_checks.csv",
        )
        artifact_hashes = sha256_files(files, root=output_dir)
        cost = _rollout_cost(rows, rollouts)
        payload = with_payload_sha256(
            {
                "schema": ROLLOUT_SCHEMA,
                "working_id": WORKING_ID,
                "status": "complete",
                "population_status": "adaptive_open_validation",
                "selected_policy": SPARSE_POLICY,
                "recurrent_gate": gate,
                "population": population,
                "failed_controls": [
                    row for row in control_rows if not parent._control_passed(row)
                ],
                "deterministic_prefix_max_abs": prefix_abs,
                "cost": cost,
                "teacher_evaluation_sha256": sha256_file(args.teacher_evaluation),
                "teacher_evaluation_payload_sha256": teacher["payload_sha256"],
                "mask_contract_sha256": sha256_file(args.mask_contract),
                "mask_contract_payload_sha256": mask_contract["payload_sha256"],
                "parent_calibration_sha256": sha256_file(args.parent_calibration),
                "parent_calibration_payload_sha256": calibration["payload_sha256"],
                "source_manifest_sha256": sha256_file(args.source_manifest),
                "source_manifest_payload_sha256": frozen_source["payload_sha256"],
                "artifact_hashes": artifact_hashes,
                "claim_boundary": (
                    "Adaptive dynamic-FV H30 mechanism result on reused open "
                    "validation; no fresh, cross-family, conservation, or sealed claim."
                ),
            }
        )
        atomic_write_json(output_dir / "rollout.json", payload)
        return payload, 0 if gate["adaptive_recurrent_pass"] else 5
    finally:
        collection._close_runtime(runtime)


def synthetic_summary() -> dict[str, Any]:
    rows = []
    active = set(FROZEN_ACTIVE_CELLS)
    for mode in range(8):
        for component in range(4):
            rows.append(
                {
                    "mode_index": mode,
                    "component": component,
                    "fixed_half_skill": (0.2 if (mode, component) in active else -0.2),
                    "fixed_half_skill_status": "ok",
                }
            )
    original = parent.synchronized_fine_discrepancy_step
    with sparse_rollout_driver():
        installed = parent.synchronized_fine_discrepancy_step is not original
    checks = {
        "mask_exact": positive_half_skill_cells(rows) == FROZEN_ACTIVE_CELLS,
        "exactly_19_active_cells": len(FROZEN_ACTIVE_CELLS) == 19,
        "constant_mode_excluded": all(mode > 0 for mode, _ in FROZEN_ACTIVE_CELLS),
        "scoped_adapter_installed": installed,
        "scoped_adapter_restored": parent.synchronized_fine_discrepancy_step
        is original,
    }
    return with_payload_sha256(
        {
            "schema": "pcno_sparse_modal_synthetic_v1",
            "working_id": WORKING_ID,
            "status": "passed" if all(checks.values()) else "failed",
            "checks": checks,
        }
    )


def _add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parent._add_external_arguments(parser)
    parent._add_runtime_arguments(parser)
    parser.add_argument("--mask-contract", type=Path, required=True)
    parser.add_argument("--parent-calibration", type=Path, required=True)
    parser.add_argument("--parent-source-manifest", type=Path, required=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("synthetic")

    contract = commands.add_parser("contract")
    contract.add_argument("--parent-calibration", type=Path, required=True)
    contract.add_argument("--modal-summary", type=Path, required=True)
    contract.add_argument("--modal-records", type=Path, required=True)
    contract.add_argument("--output", type=Path, required=True)

    teacher = commands.add_parser("teacher")
    _add_runtime_arguments(teacher)
    teacher.add_argument("--output-dir", type=Path, required=True)

    rollout = commands.add_parser("rollout")
    _add_runtime_arguments(rollout)
    rollout.add_argument("--teacher-evaluation", type=Path, required=True)
    rollout.add_argument("--source-manifest", type=Path, required=True)
    rollout.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "synthetic":
        payload = synthetic_summary()
        exit_code = 0 if payload["status"] == "passed" else 2
    elif args.command == "contract":
        payload = create_mask_contract(args)
        exit_code = 0 if payload["status"] == "passed" else 2
    elif args.command == "teacher":
        if args.repeat_forward < 1:
            raise ValueError("repeat-forward must be positive")
        payload, exit_code = run_teacher(args)
    elif args.command == "rollout":
        if args.repeat_forward < 1:
            raise ValueError("repeat-forward must be positive")
        payload, exit_code = run_rollout(args)
    else:  # pragma: no cover
        raise AssertionError(f"unsupported command: {args.command}")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
