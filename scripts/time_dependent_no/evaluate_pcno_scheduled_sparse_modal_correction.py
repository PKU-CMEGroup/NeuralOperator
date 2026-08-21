#!/usr/bin/env python3
"""Evaluate the fixed W26-L5 late-window SP19 recurrent protocol."""

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
from scripts.time_dependent_no import (
    evaluate_pcno_sparse_modal_correction as sp19,
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
from utility.time_dependent_no.pcno_cross_resolution_correction import (
    ResolutionContract,
)
from utility.time_dependent_no.pcno_cross_resolution_teacher_forced import (
    ALL_INPUT_CALLS,
    EVALUATION_CASE_IDS,
    verify_payload_sha256,
    with_payload_sha256,
)
from utility.time_dependent_no.pcno_fine_discrepancy_correction import (
    MAX_CORRECTION_TO_NATIVE_INCREMENT,
)
from utility.time_dependent_no.pcno_scheduled_sparse_modal_correction import (
    ACTIVE_START_INPUT_CALL,
    HORIZON,
    SCHEDULED_POLICY,
    correction_active,
    synchronized_scheduled_sparse_modal_step,
)
from utility.time_dependent_no.pcno_sparse_modal_correction import (
    FROZEN_ACTIVE_CELLS,
    SPARSE_POLICY,
)

WORKING_ID = "W26-L5-P4-L15-SP19-A2"
SOURCE_SCHEMA = "pcno_scheduled_sparse_modal_source_manifest_v1"
ROLLOUT_SCHEMA = "pcno_scheduled_sparse_modal_rollout_v1"
INTERNAL_POLICY = sp19.INTERNAL_POLICY
RESOLUTION_CONTRACT = parent.RESOLUTION_CONTRACT
NATIVE_RESOLUTION = parent.NATIVE_RESOLUTION

EXPECTED_TEACHER_SHA256 = (
    "8368d321ccc62691d7ec2efbcf785dc6dc1ea9103af3399827328121b58731b3"
)
EXPECTED_TEACHER_PAYLOAD_SHA256 = (
    "cb22d2875317b538230007073dfb72c519a1b76d428c10ffea4f42e4daedd371"
)
EXPECTED_SP19_SOURCE_SHA256 = (
    "30ba40d5d874a2f11216cbe660c7d7f08cde4ebaa15f5db1f7c69f4a3e20dc33"
)
EXPECTED_SP19_SOURCE_PAYLOAD_SHA256 = (
    "b277cb670122a5b041da832aae76b84b8bf07d0754e96ef0f2cdd87e6922ecff"
)
EXPECTED_SP19_ROLLOUT_SHA256 = (
    "13ca462bae291c5c9ac2c5ac169b6e2a664f05108a0cbff767977f30dd5cebfa"
)
EXPECTED_SP19_ROLLOUT_PAYLOAD_SHA256 = (
    "70377e141a74c2c8e0ad21fa2aee17d9c1c9aca12e16e7367a54500346f2bff8"
)

SOURCE_PATHS = (
    "docs/time_dependent_no/W26_L5_LATE_WINDOW_RECURRENCE_PREREGISTRATION.md",
    "utility/time_dependent_no/pcno_scheduled_sparse_modal_correction.py",
    "scripts/time_dependent_no/evaluate_pcno_scheduled_sparse_modal_correction.py",
    "tests/time_dependent_no/test_pcno_scheduled_sparse_modal_correction.py",
)


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


def _require_sp19_rollout(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    verify_payload_sha256(payload)
    if sha256_file(path) != EXPECTED_SP19_ROLLOUT_SHA256:
        raise ValueError("SP19 recurrent result file differs from the preregistration")
    if (
        payload.get("payload_sha256") != EXPECTED_SP19_ROLLOUT_PAYLOAD_SHA256
        or payload.get("schema") != sp19.ROLLOUT_SCHEMA
        or payload.get("status") != "complete"
        or payload.get("selected_policy") != SPARSE_POLICY
        or payload.get("recurrent_gate", {}).get("adaptive_recurrent_pass") is not False
    ):
        raise ValueError("SP19 recurrent design evidence differs")
    expected = payload.get("artifact_hashes")
    if (
        not isinstance(expected, Mapping)
        or sha256_files(tuple(expected), root=path.parent) != expected
    ):
        raise ValueError("SP19 recurrent artifact inventory or hashes differ")
    checks = payload.get("recurrent_gate", {}).get("checks", {})
    if checks.get("all_controls_no_harm") is not False or any(
        value is not True
        for key, value in checks.items()
        if key != "all_controls_no_harm"
    ):
        raise ValueError("SP19 recurrent failure signature differs")
    return payload


def _late_teacher_evidence(teacher_path: Path) -> dict[str, float]:
    rows = list(
        csv.DictReader(
            (teacher_path.parent / "teacher_scores.csv").open(
                "r", encoding="utf-8", newline=""
            )
        )
    )
    selected: dict[str, dict[str, str]] = {}
    for row in rows:
        if (
            row["policy"] == SPARSE_POLICY
            and row["cell"] == "evaluation_late"
            and row["scope"] == "population"
            and row["view"] in {"full", "rank8_parallel"}
        ):
            if row["view"] in selected:
                raise ValueError("duplicate SP19 late teacher score")
            selected[row["view"]] = row
    if set(selected) != {"full", "rank8_parallel"}:
        raise ValueError("SP19 late teacher score inventory differs")
    evidence = {
        "full_skill": float(selected["full"]["skill_vs_zero"]),
        "rank8_skill": float(selected["rank8_parallel"]["skill_vs_zero"]),
    }
    if (
        not all(np.isfinite(value) for value in evidence.values())
        or evidence["full_skill"] <= 0.0
        or evidence["rank8_skill"] < 0.05
    ):
        raise ValueError("SP19 late teacher evidence does not qualify")
    return evidence


def _require_predecessors(
    args: argparse.Namespace,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, float],
]:
    mask_contract = sp19._require_mask_contract(args.mask_contract)
    teacher = sp19._require_teacher(args.sp19_teacher_evaluation)
    calibration, parent_source = sp19._verify_parent_inputs(args)
    sp19_source = sp19._verify_extended_source(
        args.sp19_source_manifest,
        args,
        parent_source=parent_source,
        mask_contract=mask_contract,
    )
    rollout = _require_sp19_rollout(args.sp19_rollout)
    if (
        sha256_file(args.sp19_teacher_evaluation) != EXPECTED_TEACHER_SHA256
        or teacher["payload_sha256"] != EXPECTED_TEACHER_PAYLOAD_SHA256
        or sha256_file(args.sp19_source_manifest) != EXPECTED_SP19_SOURCE_SHA256
        or sp19_source["payload_sha256"] != EXPECTED_SP19_SOURCE_PAYLOAD_SHA256
    ):
        raise ValueError("SP19 teacher/source identity differs")
    if (
        teacher["source_manifest_payload_sha256"] != sp19_source["payload_sha256"]
        or teacher["mask_contract_payload_sha256"] != mask_contract["payload_sha256"]
        or rollout["teacher_evaluation_payload_sha256"] != teacher["payload_sha256"]
        or rollout["source_manifest_payload_sha256"] != sp19_source["payload_sha256"]
    ):
        raise ValueError("SP19 predecessor artifact chain differs")
    late_evidence = _late_teacher_evidence(args.sp19_teacher_evaluation)
    return (
        mask_contract,
        teacher,
        calibration,
        parent_source,
        rollout,
        late_evidence,
    )


def _source_manifest(
    args: argparse.Namespace,
    *,
    mask_contract: Mapping[str, Any],
    teacher: Mapping[str, Any],
    calibration: Mapping[str, Any],
    parent_source: Mapping[str, Any],
    rollout: Mapping[str, Any],
    late_evidence: Mapping[str, float],
) -> dict[str, Any]:
    return with_payload_sha256(
        {
            "schema": SOURCE_SCHEMA,
            "working_id": WORKING_ID,
            "git": git_state(ROOT),
            "source_hashes": sha256_files(SOURCE_PATHS, root=ROOT),
            "source_status": _git_status_short(SOURCE_PATHS),
            "candidate": {
                "policy": SCHEDULED_POLICY,
                "active_input_calls": list(range(ACTIVE_START_INPUT_CALL, HORIZON)),
                "inactive_input_calls": list(range(ACTIVE_START_INPUT_CALL)),
                "gain": -0.5,
                "active_modal_cells": [list(cell) for cell in FROZEN_ACTIVE_CELLS],
                "relative_native_increment_cap": MAX_CORRECTION_TO_NATIVE_INCREMENT,
                "native_calls_per_trajectory": HORIZON,
                "fine_calls_per_trajectory": HORIZON - ACTIVE_START_INPUT_CALL,
                "true_error_at_inference": False,
            },
            "late_teacher_evidence": dict(late_evidence),
            "mask_contract_sha256": sha256_file(args.mask_contract),
            "mask_contract_payload_sha256": mask_contract["payload_sha256"],
            "teacher_evaluation_sha256": sha256_file(args.sp19_teacher_evaluation),
            "teacher_evaluation_payload_sha256": teacher["payload_sha256"],
            "sp19_source_manifest_sha256": sha256_file(args.sp19_source_manifest),
            "sp19_source_manifest_payload_sha256": (
                EXPECTED_SP19_SOURCE_PAYLOAD_SHA256
            ),
            "sp19_rollout_sha256": sha256_file(args.sp19_rollout),
            "sp19_rollout_payload_sha256": rollout["payload_sha256"],
            "parent_calibration_sha256": sha256_file(args.parent_calibration),
            "parent_calibration_payload_sha256": calibration["payload_sha256"],
            "parent_source_manifest_sha256": sha256_file(args.parent_source_manifest),
            "parent_source_manifest_payload_sha256": parent_source["payload_sha256"],
        }
    )


def _verify_source_manifest(
    path: Path,
    args: argparse.Namespace,
    *,
    predecessors: tuple[
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        dict[str, float],
    ],
) -> dict[str, Any]:
    recorded = _read_json(path)
    verify_payload_sha256(recorded)
    current = _source_manifest(
        args,
        mask_contract=predecessors[0],
        teacher=predecessors[1],
        calibration=predecessors[2],
        parent_source=predecessors[3],
        rollout=predecessors[4],
        late_evidence=predecessors[5],
    )
    if recorded != current:
        raise ValueError("current late-window source differs from the manifest")
    return recorded


def _recurrent_gate(checks: Mapping[str, bool]) -> dict[str, Any]:
    expected = {
        "teacher_gate_passed",
        "predecessor_identity_exact",
        "inventory_exact",
        "all_rollouts_complete_finite_admissible",
        "median_endpoint_ratio_at_most_0p98",
        "minimum_four_endpoint_wins",
        "maximum_endpoint_ratio_at_most_1p02",
        "aggregate_state_rms_ratio_at_most_0p99",
        "increment_and_cumulative_ratios_at_most_one",
        "all_controls_no_harm",
        "scheduled_common_source_closure",
        "correction_audits_pass",
        "schedule_and_mask_closure_passed",
        "deterministic_prefix_exact",
        "source_and_artifacts_exact",
    }
    if set(checks) != expected or any(
        type(value) is not bool for value in checks.values()
    ):
        raise ValueError("late-window recurrent gate evidence inventory differs")
    passed = all(checks.values())
    return {
        "status": "adaptive_recurrent_pass" if passed else "adaptive_recurrent_failed",
        "adaptive_recurrent_pass": passed,
        "fresh_confirmation": False,
        "sealed_population_authorized": False,
        "checks": dict(checks),
    }


@contextmanager
def late_window_rollout_driver(
    closure_rows: list[dict[str, Any]] | None = None,
) -> Iterator[None]:
    """Temporarily inject the fixed schedule into the frozen rollout driver."""

    original = parent.synchronized_fine_discrepancy_step
    next_input_call = 0

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
        nonlocal next_input_call
        if policy != INTERNAL_POLICY:
            raise ValueError("late-window adapter accepts only the rank-7 token")
        input_call = next_input_call
        call_order = []

        def tracked_predictor(resolution, value):
            call_order.append(resolution)
            return predictor(resolution, value)

        step = synchronized_scheduled_sparse_modal_step(
            native_state,
            input_call=input_call,
            contract=contract,
            projector=projector,
            predictor=tracked_predictor,
            volumes=volumes,
            component_scale=component_scale,
            active_cells=FROZEN_ACTIVE_CELLS,
        )
        active = correction_active(input_call)
        expected_order = (
            [contract.native, contract.fine] if active else [contract.native]
        )
        if call_order != expected_order:
            raise ValueError("scheduled model-call order differs")
        if closure_rows is not None:
            closure_rows.append(
                {
                    "input_call": input_call,
                    "active": active,
                    "logical_model_calls": len(call_order),
                    "native_model_calls": call_order.count(contract.native),
                    "fine_model_calls": call_order.count(contract.fine),
                    "correction_max_abs": float(
                        np.max(np.abs(step.correction), initial=0.0)
                    ),
                    **sp19._mask_closure(
                        step.correction,
                        projector,
                        component_scale=np.asarray(component_scale, dtype=np.float64),
                    ),
                }
            )
        next_input_call += 1
        return step

    parent.synchronized_fine_discrepancy_step = stepper
    try:
        yield
    finally:
        parent.synchronized_fine_discrepancy_step = original


def _relabel_rollout(rollout: dict[str, Any]) -> dict[str, Any]:
    rollout["policy"] = SCHEDULED_POLICY
    for row in rollout["rows"]:
        row["policy"] = SCHEDULED_POLICY
    return rollout


def _rollout_cost(
    rows: Sequence[Mapping[str, Any]],
    rollouts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    raw = [row["execution"] for row in rollouts if row["policy"] == "zero"]
    corrected = [
        row["execution"] for row in rollouts if row["policy"] == SCHEDULED_POLICY
    ]
    cost = {
        "raw_logical_model_calls": sum(row["logical_model_calls"] for row in raw),
        "corrected_logical_model_calls": sum(
            row["logical_model_calls"] for row in corrected
        ),
        "raw_native_logical_calls": sum(row["native_logical_calls"] for row in raw),
        "corrected_native_logical_calls": sum(
            row["native_logical_calls"] for row in corrected
        ),
        "corrected_fine_logical_calls": sum(
            row["fine_logical_calls"] for row in corrected
        ),
        "raw_forward_seconds": sum(row["total_forward_seconds"] for row in raw),
        "corrected_forward_seconds": sum(
            row["total_forward_seconds"] for row in corrected
        ),
        "raw_native_forward_seconds": sum(row["native_forward_seconds"] for row in raw),
        "corrected_native_forward_seconds": sum(
            row["native_forward_seconds"] for row in corrected
        ),
        "corrected_fine_forward_seconds": sum(
            row["fine_forward_seconds"] for row in corrected
        ),
        "raw_wall_seconds": sum(row["wall_seconds"] for row in raw),
        "corrected_wall_seconds": sum(row["wall_seconds"] for row in corrected),
        "raw_peak_memory_bytes": max(
            row["maximum_peak_gpu_memory_bytes"] for row in raw
        ),
        "corrected_peak_memory_bytes": max(
            row["maximum_peak_gpu_memory_bytes"] for row in corrected
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
                if row["policy"] == SCHEDULED_POLICY
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


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    predecessors = _require_predecessors(args)
    manifest = _source_manifest(
        args,
        mask_contract=predecessors[0],
        teacher=predecessors[1],
        calibration=predecessors[2],
        parent_source=predecessors[3],
        rollout=predecessors[4],
        late_evidence=predecessors[5],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite source manifest: {args.output}")
    atomic_write_json(args.output, manifest)
    return manifest


def run_rollout(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    predecessors = _require_predecessors(args)
    frozen_source = _verify_source_manifest(
        args.protocol_source_manifest,
        args,
        predecessors=predecessors,
    )
    teacher = predecessors[1]
    parent_source = predecessors[3]
    predecessor_rollout = predecessors[4]

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=False)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    runtime, runtime_parent_source = parent._build_runtime(args)
    try:
        if runtime_parent_source != parent_source:
            raise ValueError("rollout runtime differs from the parent source manifest")
        with late_window_rollout_driver():
            prefix_a = parent._rollout_arm(
                runtime,
                case_id=EVALUATION_CASE_IDS[0],
                policy=INTERNAL_POLICY,
                horizon=2,
            )
        with late_window_rollout_driver():
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
        schedule_rows: list[dict[str, Any]] = []
        for case_id in EVALUATION_CASE_IDS:
            print(f"late-window rollout: {case_id} raw", flush=True)
            rollouts.append(
                parent._rollout_arm(
                    runtime, case_id=case_id, policy="zero", horizon=HORIZON
                )
            )
            print(f"late-window rollout: {case_id} corrected", flush=True)
            case_schedule: list[dict[str, Any]] = []
            with late_window_rollout_driver(case_schedule):
                corrected = parent._rollout_arm(
                    runtime,
                    case_id=case_id,
                    policy=INTERNAL_POLICY,
                    horizon=HORIZON,
                )
            schedule_rows.extend({"case_id": case_id, **row} for row in case_schedule)
            rollouts.append(_relabel_rollout(corrected))

        rows = [row for rollout in rollouts for row in rollout["rows"]]
        case_rows, control_rows, population = parent._paired_rollout_controls(rows)
        completed = all(
            rollout["execution"]["completed_calls"] == HORIZON
            and rollout["first_invalid_call"] is None
            and rollout["first_nonfinite_call"] is None
            for rollout in rollouts
        )
        corrected_rollouts = [
            row for row in rollouts if row["policy"] == SCHEDULED_POLICY
        ]
        common_source = all(
            row["execution"]["native_logical_calls"] == HORIZON
            and row["execution"]["fine_logical_calls"]
            == HORIZON - ACTIVE_START_INPUT_CALL
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
        schedule_pass = len(schedule_rows) == len(
            EVALUATION_CASE_IDS
        ) * HORIZON and all(
            bool(row["active"]) == correction_active(int(row["input_call"]))
            and int(row["logical_model_calls"]) == (2 if bool(row["active"]) else 1)
            and int(row["native_model_calls"]) == 1
            and int(row["fine_model_calls"]) == (1 if bool(row["active"]) else 0)
            and float(row["maximum_inactive_coordinate_abs"]) <= 1.0e-10
            and float(row["maximum_constant_coordinate_abs"]) <= 1.0e-10
            and int(row["active_cell_count"]) == len(FROZEN_ACTIVE_CELLS)
            and (bool(row["active"]) or float(row["correction_max_abs"]) == 0.0)
            for row in schedule_rows
        )
        checks = {
            "teacher_gate_passed": teacher["teacher_gate"]["recurrent_pilot_authorized"]
            is True,
            "predecessor_identity_exact": predecessor_rollout["payload_sha256"]
            == EXPECTED_SP19_ROLLOUT_PAYLOAD_SHA256,
            "inventory_exact": len(rows) == len(EVALUATION_CASE_IDS) * HORIZON * 2,
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
            "scheduled_common_source_closure": common_source,
            "correction_audits_pass": audit_pass,
            "schedule_and_mask_closure_passed": schedule_pass,
            "deterministic_prefix_exact": prefix_abs == 0.0,
            "source_and_artifacts_exact": runtime_parent_source == parent_source,
        }
        gate = _recurrent_gate(checks)

        write_csv(output_dir / "rollout_call_metrics.csv", rows)
        write_csv(output_dir / "rollout_case_metrics.csv", case_rows)
        write_csv(output_dir / "rollout_controls.csv", control_rows)
        write_csv(output_dir / "rollout_schedule_closure.csv", schedule_rows)
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
            "rollout_schedule_closure.csv",
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
                "selected_policy": SCHEDULED_POLICY,
                "recurrent_gate": gate,
                "population": population,
                "failed_controls": [
                    row for row in control_rows if not parent._control_passed(row)
                ],
                "deterministic_prefix_max_abs": prefix_abs,
                "cost": cost,
                "protocol_source_manifest_sha256": sha256_file(
                    args.protocol_source_manifest
                ),
                "protocol_source_manifest_payload_sha256": frozen_source[
                    "payload_sha256"
                ],
                "teacher_evaluation_sha256": sha256_file(args.sp19_teacher_evaluation),
                "teacher_evaluation_payload_sha256": teacher["payload_sha256"],
                "sp19_rollout_sha256": sha256_file(args.sp19_rollout),
                "sp19_rollout_payload_sha256": predecessor_rollout["payload_sha256"],
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
    volumes, projector = parent._synthetic_projector()
    contract = ResolutionContract(coarse=(4, 2), native=(8, 4), fine=(16, 8))
    scale = np.asarray((0.5, 0.75, 1.0, 1.25), dtype=np.float64)
    state = np.zeros((32, 4), dtype=np.float64)
    early_calls = []
    late_calls = []

    def early_predictor(resolution, value):
        early_calls.append(resolution)
        return value + 0.1

    def late_predictor(resolution, value):
        late_calls.append(resolution)
        return value + 0.1

    early = synchronized_scheduled_sparse_modal_step(
        state,
        input_call=ACTIVE_START_INPUT_CALL - 1,
        contract=contract,
        projector=projector,
        predictor=early_predictor,
        volumes=volumes,
        component_scale=scale,
    )
    late = synchronized_scheduled_sparse_modal_step(
        state,
        input_call=ACTIVE_START_INPUT_CALL,
        contract=contract,
        projector=projector,
        predictor=late_predictor,
        volumes=volumes,
        component_scale=scale,
    )
    original = parent.synchronized_fine_discrepancy_step
    closure_rows: list[dict[str, Any]] = []
    with late_window_rollout_driver(closure_rows):
        injected = parent.synchronized_fine_discrepancy_step
    checks = {
        "early_native_only": early_calls == [contract.native],
        "early_correction_zero": bool(np.count_nonzero(early.correction) == 0),
        "late_native_then_fine": late_calls == [contract.native, contract.fine],
        "late_one_native_state": np.array_equal(
            late.next_native_state,
            late.predictions[contract.native] + late.correction,
        ),
        "driver_injected": injected is not original,
        "driver_restored": parent.synchronized_fine_discrepancy_step is original,
        "fixed_call_budget": HORIZON + (HORIZON - ACTIVE_START_INPUT_CALL) == 45,
    }
    return with_payload_sha256(
        {
            "schema": "pcno_scheduled_sparse_modal_synthetic_v1",
            "working_id": WORKING_ID,
            "status": "passed" if all(checks.values()) else "failed",
            "checks": checks,
        }
    )


def _add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    sp19._add_runtime_arguments(parser)
    parser.add_argument("--sp19-teacher-evaluation", type=Path, required=True)
    parser.add_argument("--sp19-source-manifest", type=Path, required=True)
    parser.add_argument("--sp19-rollout", type=Path, required=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("synthetic")

    preflight = commands.add_parser("preflight")
    _add_runtime_arguments(preflight)
    preflight.add_argument("--output", type=Path, required=True)

    rollout = commands.add_parser("rollout")
    _add_runtime_arguments(rollout)
    rollout.add_argument("--protocol-source-manifest", type=Path, required=True)
    rollout.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "synthetic":
        payload = synthetic_summary()
        exit_code = 0 if payload["status"] == "passed" else 2
    elif args.command == "preflight":
        payload = run_preflight(args)
        exit_code = 0
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
