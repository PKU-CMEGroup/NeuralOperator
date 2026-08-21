#!/usr/bin/env python3
"""Evaluate the W26-L5 fixed fine-discrepancy correction.

The runner reuses the verified projected W26-L5 checkpoint/data runtime.  It
keeps calibration, teacher-forced evaluation, and conditional recurrence as
separate hash-bound commands.  All scientific data are reused open validation;
sealed populations and training are unavailable from this entry point.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from itertools import pairwise
from pathlib import Path
from time import perf_counter
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.time_dependent_no import (
    evaluate_pcno_cross_resolution_teacher_forced as base,
)
from scripts.time_dependent_no import (
    evaluate_pcno_projected_cross_resolution_teacher_forced as parent,
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
    DiagnosticSnapshot,
    ResolutionContract,
    prepare_common_native_inputs,
)
from utility.time_dependent_no.pcno_cross_resolution_teacher_forced import (
    ALL_INPUT_CALLS,
    CALIBRATION_CASE_IDS,
    CALIBRATION_GROUPS,
    EVALUATION_CASE_IDS,
    FRONT_CONTROL_KEYS,
    INTEGRAL_COMPONENT_NAMES,
    build_fixed_cosine_projector,
    error_relation_rows,
    verify_payload_sha256,
    with_payload_sha256,
)
from utility.time_dependent_no.pcno_fine_discrepancy_correction import (
    DENOMINATOR_FLOOR,
    MAX_CORRECTION_TO_NATIVE_INCREMENT,
    NONZERO_POLICIES,
    POLICIES,
    CorrectionPolicy,
    fine_discrepancy_correction,
    modal_coordinates,
    recurrent_gate,
    score_candidate_inventory,
    select_calibration_policy,
    synchronized_fine_discrepancy_step,
    teacher_forced_gate,
)
from utility.time_dependent_no.pcno_residual_structure import shock_vortex_regions
from utility.time_dependent_no.pcno_resolution_transfer import (
    conservative_admissibility_summary,
    load_resolution_reference,
    reference_at_resolution,
    weighted_scaled_rms,
)

WORKING_ID = "W26-L5-P2-FD05-A2"
READINESS_SCHEMA = "pcno_fine_discrepancy_readiness_v1"
SOURCE_MANIFEST_SCHEMA = "pcno_fine_discrepancy_source_manifest_v1"
PREFLIGHT_SCHEMA = "pcno_fine_discrepancy_preflight_v1"
SMOKE_SCHEMA = "pcno_fine_discrepancy_smoke_v1"
CALIBRATION_SCHEMA = "pcno_fine_discrepancy_calibration_v1"
TEACHER_SCHEMA = "pcno_fine_discrepancy_teacher_evaluation_v1"
ROLLOUT_SCHEMA = "pcno_fine_discrepancy_rollout_v1"

RESOLUTION_CONTRACT = base.RESOLUTION_CONTRACT
NATIVE_RESOLUTION = RESOLUTION_CONTRACT.native
EARLY_CALLS = tuple(range(15))
LATE_CALLS = tuple(range(15, 30))
SOURCE_PATHS = (
    "DERIVATION_PACKAGE.md",
    "docs/time_dependent_no/W26_L5_FINE_DISCREPANCY_ROLLOUT_PREREGISTRATION.md",
    "utility/time_dependent_no/pcno_fine_discrepancy_correction.py",
    "scripts/time_dependent_no/evaluate_pcno_fine_discrepancy_correction.py",
    "tests/time_dependent_no/test_pcno_fine_discrepancy_correction.py",
)
FIELD_CONTROL_VIEWS = (
    "full",
    "band_large",
    "band_transition",
    "band_local",
    "region_boundary",
    "region_shock",
    "region_vortex",
    "region_smooth",
    "smooth_local",
    "component_density",
    "component_x_momentum",
    "component_y_momentum",
    "component_energy",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON payload must be an object: {path}")
    return value


def _git_status_short(paths: Sequence[str] | None = None) -> list[str]:
    command = ["git", "status", "--short"]
    if paths is not None:
        command.extend(("--", *paths))
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return ["unknown"]
    return result.stdout.splitlines() if result.returncode == 0 else ["unknown"]


def _source_hashes() -> dict[str, str]:
    return sha256_files(SOURCE_PATHS, root=ROOT)


def _parent_args(args: argparse.Namespace) -> argparse.Namespace:
    values = vars(args).copy()
    values["readiness"] = args.parent_readiness
    return argparse.Namespace(**values)


def _maximum_absolute(value: np.ndarray) -> float:
    array = np.asarray(value, dtype=np.float64)
    return 0.0 if array.size == 0 else float(np.max(np.abs(array)))


def _population_row(payload: Mapping[str, Any], view: str) -> Mapping[str, Any]:
    matches = [
        row
        for row in payload.get("rows", [])
        if row.get("scope") == "population" and row.get("view") == view
    ]
    if len(matches) != 1:
        raise ValueError(f"missing population row for view {view!r}")
    return matches[0]


def _case_rows(payload: Mapping[str, Any], view: str) -> list[Mapping[str, Any]]:
    rows = [
        row
        for row in payload.get("rows", [])
        if row.get("scope") == "case" and row.get("view") == view
    ]
    if len(rows) != len(payload.get("case_ids", [])):
        raise ValueError(f"incomplete case rows for view {view!r}")
    return rows


def _rms_control(
    *,
    key: str,
    scope: str,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not rows or any(
        row.get("zero_error") is None
        or row.get("corrected_error") is None
        or not np.isfinite(float(row["zero_error"]))
        or not np.isfinite(float(row["corrected_error"]))
        for row in rows
    ):
        return {
            "key": key,
            "scope": scope,
            "zero_rms": 0.0,
            "corrected_rms": 0.0,
            "ratio": None,
            "status": "incomplete",
        }
    zero_rms = float(np.sqrt(np.mean([float(row["zero_error"]) ** 2 for row in rows])))
    corrected_rms = float(
        np.sqrt(np.mean([float(row["corrected_error"]) ** 2 for row in rows]))
    )
    if zero_rms <= DENOMINATOR_FLOOR:
        if corrected_rms <= DENOMINATOR_FLOOR:
            ratio = 1.0
            status = "exact_zero_no_change"
        else:
            ratio = None
            status = "small_denominator_harm"
    else:
        ratio = corrected_rms / zero_rms
        status = "ok" if np.isfinite(ratio) else "nonfinite"
    return {
        "key": key,
        "scope": scope,
        "zero_rms": zero_rms,
        "corrected_rms": corrected_rms,
        "ratio": ratio,
        "status": status,
    }


def _control_passed(row: Mapping[str, Any]) -> bool:
    return bool(
        row.get("status") in {"ok", "exact_zero_no_change"}
        and row.get("ratio") is not None
        and np.isfinite(float(row["ratio"]))
        and float(row["ratio"]) <= 1.05
    )


def _modal_records(
    snapshots: Sequence[DiagnosticSnapshot],
    projector: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    matrices: dict[str, list[np.ndarray]] = {"coarse": [], "fine": [], "target": []}
    owners: list[tuple[str, str, int]] = []
    for snapshot in sorted(snapshots, key=lambda row: (row.case_id, row.input_call)):
        fields = {
            "coarse": snapshot.basis.native_minus_coarse,
            "fine": snapshot.basis.fine_minus_native,
            "target": snapshot.target_correction,
        }
        coordinates = {
            name: modal_coordinates(
                value,
                projector,
                component_scale=snapshot.component_scale,
            )
            for name, value in fields.items()
        }
        for name, values in matrices.items():
            values.append(coordinates[name].reshape(-1))
        owners.append((snapshot.case_id, snapshot.group_id, int(snapshot.input_call)))
        for mode_index, mode in enumerate(projector.modes):
            for component in range(coordinates["target"].shape[1]):
                rows.append(
                    {
                        "case_id": snapshot.case_id,
                        "group_id": snapshot.group_id,
                        "input_call": int(snapshot.input_call),
                        "mode_index": mode_index,
                        "mode_x": int(mode[0]),
                        "mode_y": int(mode[1]),
                        "component": component,
                        "coarse_coordinate": float(
                            coordinates["coarse"][mode_index, component]
                        ),
                        "fine_coordinate": float(
                            coordinates["fine"][mode_index, component]
                        ),
                        "target_coordinate": float(
                            coordinates["target"][mode_index, component]
                        ),
                    }
                )

    x_c = np.stack(matrices["coarse"])
    x_f = np.stack(matrices["fine"])
    target = np.stack(matrices["target"])

    def relation(indices: np.ndarray) -> dict[str, Any]:
        coarse = x_c[indices].reshape(-1)
        fine = x_f[indices].reshape(-1)
        truth = target[indices].reshape(-1)
        coarse_error = -truth - coarse
        native_error = -truth
        fine_error = -truth + fine
        cc = float(coarse @ coarse)
        ff = float(fine @ fine)
        yy = float(truth @ truth)
        cf = float(coarse @ fine)
        cy = float(coarse @ truth)
        fy = float(fine @ truth)
        feature_cosine = cf / math.sqrt(cc * ff) if cc > 0.0 and ff > 0.0 else None
        fine_target_cosine = fy / math.sqrt(ff * yy) if ff > 0.0 and yy > 0.0 else None
        beta = fy / ff if ff > DENOMINATOR_FLOOR**2 else None
        fixed_error = yy + 0.25 * ff + fy if beta is not None else None
        fixed_skill = (
            1.0 - fixed_error / yy if fixed_error is not None and yy > 0.0 else None
        )
        lam = cf / ff if ff > DENOMINATOR_FLOOR**2 else None
        innovation = coarse - lam * fine if lam is not None else None
        innovation_square = (
            float(innovation @ innovation) if innovation is not None else None
        )
        innovation_cross = float(innovation @ truth) if innovation is not None else None
        incremental_skill = (
            innovation_cross**2 / (innovation_square * yy)
            if innovation_square is not None
            and innovation_square > DENOMINATOR_FLOOR**2
            and yy > DENOMINATOR_FLOOR**2
            else None
        )

        def pair_metrics(
            left: np.ndarray,
            right: np.ndarray,
            *,
            prefix: str,
        ) -> dict[str, Any]:
            left_square = float(left @ left)
            right_square = float(right @ right)
            cross = float(left @ right)
            resolved = (
                left_square > DENOMINATOR_FLOOR**2
                and right_square > DENOMINATOR_FLOOR**2
            )
            return {
                f"{prefix}_cosine": (
                    cross / math.sqrt(left_square * right_square) if resolved else None
                ),
                f"{prefix}_left_to_right_norm_ratio": (
                    math.sqrt(left_square / right_square) if resolved else None
                ),
                f"{prefix}_status": "ok" if resolved else "small_denominator",
            }

        grid_error_matrix = np.stack((coarse_error, native_error, fine_error))
        grid_error_singular_values = np.linalg.svd(
            grid_error_matrix,
            compute_uv=False,
        )
        grid_error_energy = float(np.sum(np.square(grid_error_singular_values)))
        output = {
            "snapshot_count": int(indices.size),
            "feature_cosine": feature_cosine,
            "feature_relation_status": (
                "ok"
                if cc > DENOMINATOR_FLOOR**2 and ff > DENOMINATOR_FLOOR**2
                else "small_denominator"
            ),
            "coarse_to_fine_norm_ratio": (
                math.sqrt(cc / ff) if cc > 0.0 and ff > 0.0 else None
            ),
            "coarse_target_cosine": (
                cy / math.sqrt(cc * yy) if cc > 0.0 and yy > 0.0 else None
            ),
            "fine_target_cosine": fine_target_cosine,
            "fine_target_status": (
                "ok"
                if ff > DENOMINATOR_FLOOR**2 and yy > DENOMINATOR_FLOOR**2
                else "small_denominator"
            ),
            "fine_only_optimal_beta": beta,
            "fixed_half_skill": fixed_skill,
            "fine_fit_status": (
                "ok"
                if beta is not None and yy > DENOMINATOR_FLOOR**2
                else "small_denominator"
            ),
            "coarse_on_fine_projection": lam,
            "coarse_innovation_energy_fraction": (
                innovation_square / cc
                if innovation_square is not None and cc > 0.0
                else None
            ),
            "coarse_innovation_incremental_skill": incremental_skill,
            "coarse_innovation_status": (
                "ok" if incremental_skill is not None else "small_denominator"
            ),
            "coarse_square": cc,
            "fine_square": ff,
            "target_square": yy,
            "coarse_error_square": float(coarse_error @ coarse_error),
            "native_error_square": float(native_error @ native_error),
            "fine_error_square": float(fine_error @ fine_error),
            "common_grid_error_rank1_energy_fraction": (
                float(grid_error_singular_values[0] ** 2 / grid_error_energy)
                if grid_error_energy > DENOMINATOR_FLOOR**2
                else None
            ),
            "common_grid_error_status": (
                "ok"
                if grid_error_energy > DENOMINATOR_FLOOR**2
                else "small_denominator"
            ),
        }
        output.update(
            pair_metrics(
                coarse_error,
                native_error,
                prefix="coarse_native_error",
            )
        )
        output.update(
            pair_metrics(
                native_error,
                fine_error,
                prefix="native_fine_error",
            )
        )
        output.update(
            pair_metrics(
                coarse_error,
                fine_error,
                prefix="coarse_fine_error",
            )
        )
        return output

    scopes: dict[str, np.ndarray] = {
        "all": np.arange(len(owners), dtype=np.int64),
        "calls_0_14": np.asarray([i for i, row in enumerate(owners) if row[2] <= 14]),
        "calls_15_29": np.asarray([i for i, row in enumerate(owners) if row[2] >= 15]),
    }
    for group_id in sorted({row[1] for row in owners}):
        scopes[f"group::{group_id}"] = np.asarray(
            [i for i, row in enumerate(owners) if row[1] == group_id],
            dtype=np.int64,
        )
    for input_call in ALL_INPUT_CALLS:
        scopes[f"call::{input_call}"] = np.asarray(
            [i for i, row in enumerate(owners) if row[2] == input_call],
            dtype=np.int64,
        )
    relation_rows = [
        {"scope": scope, **relation(indices)}
        for scope, indices in scopes.items()
        if indices.size
    ]

    mode_component_rows = []
    for column in range(target.shape[1]):
        mode_index, component = divmod(column, 4)
        coarse = x_c[:, column]
        fine = x_f[:, column]
        truth = target[:, column]
        ff = float(fine @ fine)
        yy = float(truth @ truth)
        fy = float(fine @ truth)
        beta = fy / ff if ff > DENOMINATOR_FLOOR**2 else None
        fixed_error = float(np.sum(np.square(truth + 0.5 * fine)))
        coarse_error = -truth - coarse
        native_error = -truth
        fine_error = -truth + fine

        def error_cosine(left: np.ndarray, right: np.ndarray) -> float | None:
            denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
            return (
                float(left @ right / denominator)
                if denominator > DENOMINATOR_FLOOR**2
                else None
            )

        mode_component_rows.append(
            {
                "mode_index": mode_index,
                "mode_x": int(projector.modes[mode_index][0]),
                "mode_y": int(projector.modes[mode_index][1]),
                "component": component,
                "target_energy_fraction": yy / float(np.sum(np.square(target)))
                if yy > 0.0
                else 0.0,
                "fine_only_optimal_beta": beta,
                "fine_fit_status": "ok" if beta is not None else "small_denominator",
                "fixed_half_skill": 1.0 - fixed_error / yy
                if yy > DENOMINATOR_FLOOR**2
                else None,
                "fixed_half_skill_status": (
                    "ok" if yy > DENOMINATOR_FLOOR**2 else "small_denominator"
                ),
                "fine_target_cosine": (
                    fy / math.sqrt(ff * yy)
                    if ff > DENOMINATOR_FLOOR**2 and yy > DENOMINATOR_FLOOR**2
                    else None
                ),
                "coarse_native_error_cosine": error_cosine(
                    coarse_error,
                    native_error,
                ),
                "native_fine_error_cosine": error_cosine(
                    native_error,
                    fine_error,
                ),
                "coarse_fine_error_cosine": error_cosine(
                    coarse_error,
                    fine_error,
                ),
            }
        )

    singular_values = np.linalg.svd(target, compute_uv=False)
    singular_energy = np.square(singular_values)
    total_singular_energy = float(singular_energy.sum())
    cumulative = (
        np.cumsum(singular_energy) / total_singular_energy
        if total_singular_energy > DENOMINATOR_FLOOR**2
        else np.zeros_like(singular_energy)
    )
    singular_tolerance = (
        max(target.shape) * np.finfo(np.float64).eps * float(singular_values[0])
        if singular_values.size
        else 0.0
    )
    temporal_rows = []
    for case_id in sorted({row[0] for row in owners}):
        indices = [i for i, row in enumerate(owners) if row[0] == case_id]
        indices.sort(key=lambda i: owners[i][2])
        for left, right in pairwise(indices):
            values = {"case_id": case_id, "input_call": owners[right][2]}
            for name, matrix in (("fine", x_f), ("target", target)):
                denominator = float(
                    np.linalg.norm(matrix[left]) * np.linalg.norm(matrix[right])
                )
                values[f"{name}_consecutive_cosine"] = (
                    float(matrix[left] @ matrix[right] / denominator)
                    if denominator > DENOMINATOR_FLOOR**2
                    else None
                )
            temporal_rows.append(values)
    fine_temporal = [
        row["fine_consecutive_cosine"]
        for row in temporal_rows
        if row["fine_consecutive_cosine"] is not None
    ]
    target_temporal = [
        row["target_consecutive_cosine"]
        for row in temporal_rows
        if row["target_consecutive_cosine"] is not None
    ]
    summary = {
        "relation_rows": relation_rows,
        "mode_component_rows": mode_component_rows,
        "target_singular_values": singular_values.tolist(),
        "target_cumulative_modal_energy": cumulative.tolist(),
        "target_modal_matrix_rank": int(
            np.count_nonzero(singular_values > singular_tolerance)
        ),
        "target_modal_rank_status": (
            "ok"
            if total_singular_energy > DENOMINATOR_FLOOR**2
            else "small_denominator"
        ),
        "target_modes_for_95_percent_energy": (
            int(np.searchsorted(cumulative, 0.95) + 1)
            if total_singular_energy > DENOMINATOR_FLOOR**2
            else None
        ),
        "median_fine_consecutive_cosine": (
            float(np.median(fine_temporal)) if fine_temporal else None
        ),
        "fine_temporal_status": "ok" if fine_temporal else "small_denominator",
        "median_target_consecutive_cosine": (
            float(np.median(target_temporal)) if target_temporal else None
        ),
        "target_temporal_status": ("ok" if target_temporal else "small_denominator"),
        "temporal_rows": temporal_rows,
    }
    return rows, summary


def _synthetic_projector():
    nx, ny = (8, 4)
    x = (np.arange(nx, dtype=np.float64) + 0.5) * (2.0 / nx)
    y = (np.arange(ny, dtype=np.float64) + 0.5) * (1.0 / ny)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    nodes = np.column_stack((xx.reshape(-1), yy.reshape(-1)))
    volumes = np.full(nx * ny, 2.0 / (nx * ny), dtype=np.float64)
    node_type = np.zeros(nx * ny, dtype=np.int64)
    node_type[:nx] = 1
    projector = build_fixed_cosine_projector(
        nodes,
        volumes,
        node_type,
        rank=8,
    )
    return volumes, projector


def synthetic_summary(seed: int = 0) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    volumes, projector = _synthetic_projector()
    scale = np.asarray((0.5, 0.75, 1.0, 1.25), dtype=np.float64)
    native_state = rng.normal(size=(8 * 4, 4))
    calls: list[tuple[tuple[int, int], np.ndarray]] = []

    def predictor(resolution, state):
        calls.append((resolution, np.array(state, copy=True)))
        return state + 0.1

    step = synchronized_fine_discrepancy_step(
        native_state,
        contract=ResolutionContract(coarse=(4, 2), native=(8, 4), fine=(16, 8)),
        projector=projector,
        predictor=predictor,
        policy="rank7_fine_away_half",
        volumes=volumes,
        component_scale=scale,
    )
    expected = prepare_common_native_inputs(
        native_state,
        contract=ResolutionContract(coarse=(4, 2), native=(8, 4), fine=(16, 8)),
    )
    checks = {
        "exactly_two_predictions": [resolution for resolution, _ in calls]
        == [(8, 4), (16, 8)],
        "common_native_inputs": all(
            np.array_equal(value, expected.model_inputs[resolution])
            for resolution, value in calls
        ),
        "one_native_state": step.next_native_state.shape == native_state.shape,
        "excluded_support_zero": step.audit.maximum_excluded_abs == 0.0,
        "rank7_mean_neutral": step.audit.maximum_scaled_component_mean_abs <= 1.0e-12,
        "modal_reconstruction": step.audit.maximum_modal_reconstruction_abs <= 1.0e-10,
        "checkpoint_closed": True,
        "dataset_closed": True,
        "reference_closed": True,
    }
    return with_payload_sha256(
        {
            "schema": SMOKE_SCHEMA,
            "working_id": WORKING_ID,
            "status": "passed" if all(checks.values()) else "failed",
            "seed": int(seed),
            "checks": checks,
            "checkpoint_loaded": False,
            "dataset_loaded": False,
            "reference_loaded": False,
            "source_sha256": _source_hashes(),
        }
    )


def create_readiness(args: argparse.Namespace) -> dict[str, Any]:
    parent_payload = parent._verify_readiness(args.parent_readiness)
    prior_calibration = _read_json(args.prior_calibration)
    prior_evaluation = _read_json(args.prior_evaluation)
    verify_payload_sha256(prior_calibration)
    verify_payload_sha256(prior_evaluation)
    synthetic = synthetic_summary(args.synthetic_seed)
    verify_payload_sha256(synthetic)
    source_hashes = _source_hashes()
    checks = {
        "source_inventory_exact": set(source_hashes) == set(SOURCE_PATHS),
        "focused_cpu_tests_passed": args.focused_test_result.startswith("passed"),
        "synthetic_passed": synthetic["status"] == "passed",
        "synthetic_scientific_inputs_closed": (
            synthetic["checkpoint_loaded"] is False
            and synthetic["dataset_loaded"] is False
            and synthetic["reference_loaded"] is False
        ),
        "parent_readiness_passed": parent_payload.get("status") == "passed",
        "prior_calibration_schema": prior_calibration.get("schema")
        == parent.CALIBRATION_SCHEMA,
        "prior_evaluation_schema": prior_evaluation.get("schema")
        == parent.EVALUATION_SCHEMA,
        "prior_evaluation_stopped": prior_evaluation.get("adaptive_gate", {}).get(
            "status"
        )
        == "stopped",
    }
    payload = with_payload_sha256(
        {
            "schema": READINESS_SCHEMA,
            "working_id": WORKING_ID,
            "status": "passed" if all(checks.values()) else "failed",
            "population_status": "adaptive_open_validation",
            "checks": checks,
            "git": git_state(ROOT),
            "global_status_short": _git_status_short(),
            "relevant_status_short": _git_status_short(SOURCE_PATHS),
            "source_sha256": source_hashes,
            "parent_readiness_sha256": sha256_file(args.parent_readiness),
            "parent_readiness_payload_sha256": parent_payload["payload_sha256"],
            "prior_calibration_sha256": sha256_file(args.prior_calibration),
            "prior_calibration_payload_sha256": prior_calibration["payload_sha256"],
            "prior_evaluation_sha256": sha256_file(args.prior_evaluation),
            "prior_evaluation_payload_sha256": prior_evaluation["payload_sha256"],
            "synthetic_payload_sha256": synthetic["payload_sha256"],
            "focused_test_command": args.focused_test_command,
            "focused_test_result": args.focused_test_result,
            "authorization": {
                "adaptive_open_teacher_forced": True,
                "conditional_adaptive_h30_recurrence": True,
                "training": False,
                "sealed_population": False,
                "bump": False,
                "reference_generation": False,
            },
        }
    )
    atomic_write_json(args.output, payload)
    return payload


def _verify_readiness(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    verify_payload_sha256(payload)
    checks = payload.get("checks")
    if (
        payload.get("schema") != READINESS_SCHEMA
        or payload.get("status") != "passed"
        or payload.get("population_status") != "adaptive_open_validation"
        or not isinstance(checks, Mapping)
        or not checks
        or not all(value is True for value in checks.values())
    ):
        raise ValueError("fine-discrepancy readiness is absent or did not pass")
    if payload.get("source_sha256") != _source_hashes():
        raise ValueError("fine-discrepancy source changed after readiness")
    return payload


def _source_manifest(
    args: argparse.Namespace,
    readiness: Mapping[str, Any],
    parent_source: Mapping[str, Any],
) -> dict[str, Any]:
    parent_readiness = parent._verify_readiness(args.parent_readiness)
    if readiness.get("parent_readiness_payload_sha256") != parent_readiness.get(
        "payload_sha256"
    ):
        raise ValueError("parent readiness identity differs from the new readiness")
    return with_payload_sha256(
        {
            "schema": SOURCE_MANIFEST_SCHEMA,
            "working_id": WORKING_ID,
            "population_status": "adaptive_open_validation",
            "readiness_sha256": sha256_file(args.readiness),
            "readiness_payload_sha256": readiness["payload_sha256"],
            "source_sha256": _source_hashes(),
            "parent_readiness_sha256": sha256_file(args.parent_readiness),
            "parent_source_manifest": parent_source,
            "prior_calibration_sha256": readiness["prior_calibration_sha256"],
            "prior_evaluation_sha256": readiness["prior_evaluation_sha256"],
            "population": {
                "calibration_cases": list(CALIBRATION_CASE_IDS),
                "evaluation_cases": list(EVALUATION_CASE_IDS),
                "input_calls": list(ALL_INPUT_CALLS),
                "strength_ood_and_test": "sealed",
            },
            "candidate": {
                "policies": list(POLICIES),
                "gain": -0.5,
                "relative_native_increment_cap": MAX_CORRECTION_TO_NATIVE_INCREMENT,
                "teacher_predictions": ["coarse", "native", "fine"],
                "recurrent_predictions": ["native", "fine"],
                "true_error_at_inference": False,
            },
            "deterministic_runtime": {
                "torch_deterministic_algorithms": True,
                "cudnn_benchmark": False,
                "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            },
        }
    )


def _open_contract(args: argparse.Namespace):
    readiness = _verify_readiness(args.readiness)
    checkpoint, manifest, store, parent_source = parent._open_contract(
        _parent_args(args)
    )
    try:
        source = _source_manifest(args, readiness, parent_source)
    except Exception:
        store.close()
        raise
    return checkpoint, manifest, store, source


def _build_runtime(args: argparse.Namespace):
    readiness = _verify_readiness(args.readiness)
    runtime, parent_source = parent._build_runtime(_parent_args(args))
    try:
        source = _source_manifest(args, readiness, parent_source)
    except Exception:
        base._close_runtime(runtime)
        raise
    return runtime, source


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    _, manifest, store, source = _open_contract(args)
    try:
        splits = {
            case_id: base.family_case_provenance(manifest, case_id)["split"]
            for case_id in (*CALIBRATION_CASE_IDS, *EVALUATION_CASE_IDS)
        }
        payload = with_payload_sha256(
            {
                "schema": PREFLIGHT_SCHEMA,
                "working_id": WORKING_ID,
                "status": "passed",
                "population_status": "adaptive_open_validation",
                "source_manifest": source,
                "case_inventory_present": all(
                    case_id in store.keys
                    for case_id in (*CALIBRATION_CASE_IDS, *EVALUATION_CASE_IDS)
                ),
                "all_cases_open_validation": all(
                    value == "validation" for value in splits.values()
                ),
                "checkpoint_or_reference_arrays_loaded": False,
                "evaluation_targets_loaded": False,
            }
        )
    finally:
        store.close()
    atomic_write_json(args.output, payload)
    return payload


def _reconstruct_controls(
    runtime: Any,
    snapshots: Sequence[DiagnosticSnapshot],
    *,
    policies: Sequence[CorrectionPolicy],
) -> dict[str, list[dict[str, Any]]]:
    lookup = {(row.case_id, int(row.input_call)): row for row in snapshots}
    if len(lookup) != len(snapshots):
        raise ValueError("control reconstruction received duplicate snapshots")
    front_rows: list[dict[str, Any]] = []
    integral_rows: list[dict[str, Any]] = []
    proposal_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    native_geometry = runtime.geometry_by_resolution[NATIVE_RESOLUTION]
    maximum_target_reconstruction = 0.0
    for case_id in sorted({row.case_id for row in snapshots}):
        reference, _ = load_resolution_reference(
            runtime.args.family_root,
            runtime.args.multires_reference_root,
            runtime.store,
            runtime.manifest,
            case_id,
            training_resolution=NATIVE_RESOLUTION,
        )
        reference_resolution = tuple(
            int(value) for value in reference["retained_resolution"]
        )
        for input_call in sorted(
            row.input_call for row in snapshots if row.case_id == case_id
        ):
            snapshot = lookup[(case_id, int(input_call))]
            target = reference_at_resolution(
                reference["conservative_states"][(int(input_call) + 1) * 2],
                reference_resolution=reference_resolution,
                target_resolution=NATIVE_RESOLUTION,
            )
            if target is None:
                raise ValueError(
                    "native target is unavailable during control reconstruction"
                )
            target = np.asarray(target, dtype=np.float64)
            raw_prediction = target - snapshot.target_correction
            maximum_target_reconstruction = max(
                maximum_target_reconstruction,
                _maximum_absolute(target - raw_prediction - snapshot.target_correction),
            )
            raw_front = base._front_errors(raw_prediction, target, runtime)
            raw_integral = base._physical_integral(
                raw_prediction - target,
                native_geometry.node_measures,
            )
            for policy in policies:
                correction, audit = fine_discrepancy_correction(
                    snapshot.basis,
                    runtime.native_projector,
                    policy=policy,
                    volumes=snapshot.volumes,
                    component_scale=snapshot.component_scale,
                )
                corrected = raw_prediction + correction
                corrected_front = base._front_errors(corrected, target, runtime)
                corrected_integral = base._physical_integral(
                    corrected - target,
                    native_geometry.node_measures,
                )
                for key in FRONT_CONTROL_KEYS:
                    front_rows.append(
                        {
                            "policy": policy,
                            "case_id": case_id,
                            "input_call": int(input_call),
                            "key": key,
                            "zero_error": raw_front[key],
                            "corrected_error": corrected_front[key],
                        }
                    )
                for component, name in enumerate(INTEGRAL_COMPONENT_NAMES):
                    integral_rows.append(
                        {
                            "policy": policy,
                            "case_id": case_id,
                            "input_call": int(input_call),
                            "component": name,
                            "zero_error": float(raw_integral[component]),
                            "corrected_error": float(corrected_integral[component]),
                        }
                    )
                admissibility = conservative_admissibility_summary(
                    corrected,
                    gamma=runtime.normalization.gamma,
                )
                proposal_rows.append(
                    {
                        "policy": policy,
                        "case_id": case_id,
                        "input_call": int(input_call),
                        "finite": bool(np.isfinite(corrected).all()),
                        "admissible": bool(admissibility["admissible"]),
                        "minimum_density": admissibility["minimum_density"],
                        "minimum_pressure": admissibility["minimum_pressure"],
                        "minimum_internal_energy": admissibility[
                            "minimum_internal_energy"
                        ],
                    }
                )
                audit_rows.append(
                    {
                        "policy": policy,
                        "case_id": case_id,
                        "input_call": int(input_call),
                        **asdict(audit),
                    }
                )
        del reference
    return {
        "front": front_rows,
        "integral": integral_rows,
        "proposal": proposal_rows,
        "audit": audit_rows,
        "closure": [
            {
                "maximum_target_reconstruction_abs": maximum_target_reconstruction,
                "model_calls": 0,
            }
        ],
    }


def _build_teacher_controls(
    *,
    policy: CorrectionPolicy,
    score_payload: Mapping[str, Any],
    controls: Mapping[str, Sequence[Mapping[str, Any]]],
    case_ids: Sequence[str],
    input_calls: Sequence[int],
) -> list[dict[str, Any]]:
    case_set = set(case_ids)
    call_set = {int(value) for value in input_calls}
    output: list[dict[str, Any]] = []
    score_rows = score_payload.get("rows", [])
    for view in FIELD_CONTROL_VIEWS:
        for scope in ("population", *sorted(case_set)):
            matches = [
                row
                for row in score_rows
                if row.get("view") == view
                and (
                    (scope == "population" and row.get("scope") == "population")
                    or (row.get("scope") == "case" and row.get("case_id") == scope)
                )
            ]
            if len(matches) != 1:
                output.append(
                    {
                        "key": f"field::{view}",
                        "scope": scope,
                        "zero_rms": 0.0,
                        "corrected_rms": 0.0,
                        "ratio": None,
                        "status": "incomplete",
                    }
                )
                continue
            row = matches[0]
            target_rms = float(row["target_rms"])
            ratio = row.get("rms_ratio_vs_zero")
            output.append(
                {
                    "key": f"field::{view}",
                    "scope": scope,
                    "zero_rms": target_rms,
                    "corrected_rms": target_rms * float(ratio)
                    if ratio is not None
                    else 0.0,
                    "ratio": None if ratio is None else float(ratio),
                    "status": "ok" if row.get("skill_status") == "ok" else "unresolved",
                }
            )

    front = [
        row
        for row in controls["front"]
        if row["policy"] == policy
        and row["case_id"] in case_set
        and int(row["input_call"]) in call_set
    ]
    integral = [
        row
        for row in controls["integral"]
        if row["policy"] == policy
        and row["case_id"] in case_set
        and int(row["input_call"]) in call_set
    ]
    for key in FRONT_CONTROL_KEYS:
        for scope in ("population", *sorted(case_set)):
            output.append(
                _rms_control(
                    key=key,
                    scope=scope,
                    rows=[
                        row
                        for row in front
                        if row["key"] == key
                        and (scope == "population" or row["case_id"] == scope)
                    ],
                )
            )
    endpoint = max(call_set)
    for component in INTEGRAL_COMPONENT_NAMES:
        for scope in ("population", *sorted(case_set)):
            rows = [
                row
                for row in integral
                if row["component"] == component
                and (scope == "population" or row["case_id"] == scope)
            ]
            output.append(
                _rms_control(
                    key=f"integral_rms::{component}",
                    scope=scope,
                    rows=rows,
                )
            )
            output.append(
                _rms_control(
                    key=f"integral_endpoint::{component}",
                    scope=scope,
                    rows=[row for row in rows if int(row["input_call"]) == endpoint],
                )
            )
    for row in output:
        row["policy"] = policy
        row["cell"] = score_payload["cell"]
    return output


def _candidate_closure(
    *,
    policy: CorrectionPolicy,
    score_payloads: Sequence[Mapping[str, Any]],
    audit_rows: Sequence[Mapping[str, Any]],
    collection_maxima: Mapping[str, Any],
) -> tuple[bool, dict[str, Any]]:
    selected_audits = [row for row in audit_rows if row["policy"] == policy]
    maximum_mean = max(
        float(row["maximum_scaled_component_mean_abs"]) for row in selected_audits
    )
    maximum_excluded = max(
        float(row["maximum_excluded_abs"]) for row in selected_audits
    )
    maximum_reconstruction = max(
        float(row["maximum_modal_reconstruction_abs"]) for row in selected_audits
    )
    maximum_ratio = max(
        float(row["correction_to_native_increment"])
        for row in selected_audits
        if row["correction_to_native_increment"] is not None
    )
    maximum_band = max(
        max(float(value) for value in payload["maximum_closure"].values())
        for payload in score_payloads
    )
    payload = {
        "maximum_scaled_component_mean_abs": maximum_mean,
        "maximum_excluded_abs": maximum_excluded,
        "maximum_modal_reconstruction_abs": maximum_reconstruction,
        "maximum_correction_to_native_increment": maximum_ratio,
        "maximum_metric_closure": maximum_band,
        "maximum_pre_model_nesting": float(collection_maxima["pre_model_nesting"]),
        "maximum_sign_closure": float(collection_maxima["sign_closure"]),
        "maximum_increment_integral_closure": float(
            collection_maxima["increment_integral_closure"]
        ),
        "maximum_transfer_floor_closure": float(
            collection_maxima["transfer_floor_closure"]
        ),
        "maximum_region_partition_error": int(
            collection_maxima["region_partition_error"]
        ),
    }
    passed = bool(
        selected_audits
        and maximum_excluded == 0.0
        and maximum_reconstruction <= 1.0e-10
        and maximum_ratio <= MAX_CORRECTION_TO_NATIVE_INCREMENT + 1.0e-12
        and maximum_band <= 1.0e-10
        and float(collection_maxima["pre_model_nesting"]) == 0.0
        and float(collection_maxima["sign_closure"]) <= 1.0e-12
        and float(collection_maxima["increment_integral_closure"]) <= 1.0e-12
        and float(collection_maxima["transfer_floor_closure"]) <= 1.0e-12
        and int(collection_maxima["region_partition_error"]) == 0
        and (policy != "rank7_fine_away_half" or maximum_mean <= 1.0e-12)
    )
    return passed, payload


def _group_win_count(payload: Mapping[str, Any]) -> tuple[int, list[dict[str, Any]]]:
    case_rows = {row["case_id"]: row for row in _case_rows(payload, "rank8_parallel")}
    rows = []
    for group_id, members in sorted(CALIBRATION_GROUPS.items()):
        selected = [case_rows[case_id] for case_id in members]
        zero = sum(float(row["zero_sse"]) for row in selected)
        corrected = sum(float(row["corrected_sse"]) for row in selected)
        skill = 1.0 - corrected / zero if zero > DENOMINATOR_FLOOR**2 else None
        rows.append(
            {
                "group_id": group_id,
                "case_ids": list(members),
                "zero_sse_sum": zero,
                "corrected_sse_sum": corrected,
                "skill_vs_zero": skill,
                "status": "ok"
                if skill is not None and np.isfinite(skill)
                else "unresolved",
            }
        )
    return sum(
        row["skill_vs_zero"] is not None and row["skill_vs_zero"] > 0.0 for row in rows
    ), rows


def _write_modal_outputs(
    output_dir: Path,
    snapshots: Sequence[DiagnosticSnapshot],
    projector: Any,
    *,
    prefix: str,
) -> tuple[Path, Path, dict[str, Any]]:
    modal_rows, modal_summary = _modal_records(snapshots, projector)
    modal_path = output_dir / f"{prefix}_modal_records.csv"
    summary_path = output_dir / f"{prefix}_modal_summary.json"
    write_csv(modal_path, modal_rows)
    summary_payload = with_payload_sha256(
        {
            "schema": "pcno_fine_discrepancy_modal_summary_v1",
            "working_id": WORKING_ID,
            "population_status": "adaptive_open_validation",
            **modal_summary,
        }
    )
    atomic_write_json(summary_path, summary_payload)
    return modal_path, summary_path, summary_payload


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    runtime, source = _build_runtime(args)
    try:
        collected = base._collect(
            runtime,
            case_ids=(CALIBRATION_CASE_IDS[0],),
            input_calls=(0,),
            coefficients=None,
            phase="fine_discrepancy_smoke",
        )
        snapshot = collected.snapshots[0]
        scores = {
            policy: score_candidate_inventory(
                collected.snapshots,
                label="smoke",
                expected_case_ids=(CALIBRATION_CASE_IDS[0],),
                expected_input_calls=(0,),
                policy=policy,
                resolution=NATIVE_RESOLUTION,
                projector=runtime.native_projector,
            )
            for policy in NONZERO_POLICIES
        }
        prepared = prepare_common_native_inputs(
            np.zeros_like(snapshot.basis.native_increment),
            contract=RESOLUTION_CONTRACT,
        )
        call_log = []

        def predictor(resolution, state):
            call_log.append((resolution, np.array(state, copy=True)))
            return state + 0.1

        synthetic_step = synchronized_fine_discrepancy_step(
            prepared.model_inputs[NATIVE_RESOLUTION],
            contract=RESOLUTION_CONTRACT,
            projector=runtime.native_projector,
            predictor=predictor,
            policy="rank7_fine_away_half",
            volumes=snapshot.volumes,
            component_scale=snapshot.component_scale,
        )
        checks = {
            "source_manifest": source["schema"] == SOURCE_MANIFEST_SCHEMA,
            "one_snapshot": len(collected.snapshots) == 1,
            "three_teacher_predictions": collected.execution["logical_model_calls"]
            == 3,
            "both_policies_scored": set(scores) == set(NONZERO_POLICIES),
            "two_recurrent_predictions": len(call_log) == 2,
            "recurrent_call_order": [row[0] for row in call_log]
            == [RESOLUTION_CONTRACT.native, RESOLUTION_CONTRACT.fine],
            "rank7_mean_neutral": synthetic_step.audit.maximum_scaled_component_mean_abs
            <= 1.0e-12,
            "evaluation_targets_closed": True,
        }
        payload = with_payload_sha256(
            {
                "schema": SMOKE_SCHEMA,
                "working_id": WORKING_ID,
                "status": "passed" if all(checks.values()) else "failed",
                "checks": checks,
                "source_manifest_payload_sha256": source["payload_sha256"],
                "execution": collected.execution,
                "closure_maxima": collected.maxima,
                "evaluation_targets_loaded": False,
            }
        )
    finally:
        base._close_runtime(runtime)
    atomic_write_json(args.output, payload)
    return payload


def run_calibration(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=False)
    runtime, source = _build_runtime(args)
    try:
        collected = base._collect(
            runtime,
            case_ids=CALIBRATION_CASE_IDS,
            input_calls=ALL_INPUT_CALLS,
            coefficients=None,
            phase="fine_discrepancy_calibration",
        )
        controls = _reconstruct_controls(
            runtime,
            collected.snapshots,
            policies=NONZERO_POLICIES,
        )
        score_payloads: dict[str, dict[str, Any]] = {}
        early_payloads: dict[str, dict[str, Any]] = {}
        late_payloads: dict[str, dict[str, Any]] = {}
        evidence: dict[str, dict[str, Any]] = {}
        group_rows: list[dict[str, Any]] = []
        all_control_rows: list[dict[str, Any]] = []
        closure_by_policy: dict[str, Any] = {}
        for policy in NONZERO_POLICIES:
            all_payload = score_candidate_inventory(
                collected.snapshots,
                label="calibration_all",
                expected_case_ids=CALIBRATION_CASE_IDS,
                expected_input_calls=ALL_INPUT_CALLS,
                policy=policy,
                resolution=NATIVE_RESOLUTION,
                projector=runtime.native_projector,
            )
            early_payload = score_candidate_inventory(
                [row for row in collected.snapshots if row.input_call in EARLY_CALLS],
                label="calibration_early",
                expected_case_ids=CALIBRATION_CASE_IDS,
                expected_input_calls=EARLY_CALLS,
                policy=policy,
                resolution=NATIVE_RESOLUTION,
                projector=runtime.native_projector,
            )
            late_payload = score_candidate_inventory(
                [row for row in collected.snapshots if row.input_call in LATE_CALLS],
                label="calibration_late",
                expected_case_ids=CALIBRATION_CASE_IDS,
                expected_input_calls=LATE_CALLS,
                policy=policy,
                resolution=NATIVE_RESOLUTION,
                projector=runtime.native_projector,
            )
            score_payloads[policy] = all_payload
            early_payloads[policy] = early_payload
            late_payloads[policy] = late_payload
            policy_controls = _build_teacher_controls(
                policy=policy,
                score_payload=all_payload,
                controls=controls,
                case_ids=CALIBRATION_CASE_IDS,
                input_calls=ALL_INPUT_CALLS,
            )
            all_control_rows.extend(policy_controls)
            wins, current_group_rows = _group_win_count(all_payload)
            for row in current_group_rows:
                group_rows.append({"policy": policy, **row})
            closure_pass, closure = _candidate_closure(
                policy=policy,
                score_payloads=(all_payload, early_payload, late_payload),
                audit_rows=controls["audit"],
                collection_maxima=collected.maxima,
            )
            closure_by_policy[policy] = closure
            full = _population_row(all_payload, "full")
            rank8 = _population_row(all_payload, "rank8_parallel")
            policy_proposals = [
                row for row in controls["proposal"] if row["policy"] == policy
            ]
            policy_audits = [
                row for row in controls["audit"] if row["policy"] == policy
            ]
            evidence[policy] = {
                "inventory_exact": (
                    len(policy_proposals)
                    == len(CALIBRATION_CASE_IDS) * len(ALL_INPUT_CALLS)
                    and len(policy_audits)
                    == len(CALIBRATION_CASE_IDS) * len(ALL_INPUT_CALLS)
                ),
                "full_skill_nonnegative": full["skill_status"] == "ok"
                and float(full["skill_vs_zero"]) >= 0.0,
                "rank8_skill_at_least_0p05": rank8["skill_status"] == "ok"
                and float(rank8["skill_vs_zero"]) >= 0.05,
                "minimum_eight_group_wins": wins >= 8,
                "all_controls_no_harm": bool(policy_controls)
                and all(_control_passed(row) for row in policy_controls),
                "all_proposals_finite_admissible": bool(policy_proposals)
                and all(
                    row["finite"] and row["admissible"] for row in policy_proposals
                ),
                "all_audits_resolved": bool(policy_audits)
                and all(row["status"] == "ok" for row in policy_audits),
                "closure_passed": closure_pass,
                "rank8_rms_ratio": rank8["rms_ratio_vs_zero"],
            }
        selection = select_calibration_policy(evidence)
        modal_path, modal_summary_path, modal_summary = _write_modal_outputs(
            output_dir,
            collected.snapshots,
            runtime.native_projector,
            prefix="calibration",
        )
        score_rows = [
            {"policy": policy, **row}
            for policy, payload in score_payloads.items()
            for row in payload["rows"]
        ]
        score_rows.extend(
            {"policy": policy, **row}
            for policy, payload in early_payloads.items()
            for row in payload["rows"]
        )
        score_rows.extend(
            {"policy": policy, **row}
            for policy, payload in late_payloads.items()
            for row in payload["rows"]
        )
        write_csv(output_dir / "calibration_scores.csv", score_rows)
        write_csv(output_dir / "calibration_controls.csv", all_control_rows)
        write_csv(output_dir / "calibration_group_scores.csv", group_rows)
        write_csv(output_dir / "calibration_proposals.csv", controls["proposal"])
        write_csv(output_dir / "calibration_audits.csv", controls["audit"])
        write_csv(
            output_dir / "calibration_error_relations_fit.csv",
            error_relation_rows(
                [row for row in collected.snapshots if row.input_call < 20],
                cell="calibration_fit",
                resolution=NATIVE_RESOLUTION,
                projector=runtime.native_projector,
            ),
        )
        write_csv(
            output_dir / "calibration_error_relations_time.csv",
            error_relation_rows(
                [row for row in collected.snapshots if row.input_call >= 20],
                cell="time_only",
                resolution=NATIVE_RESOLUTION,
                projector=runtime.native_projector,
            ),
        )
        write_csv(output_dir / "transfer_floors.csv", collected.floor_rows)
        write_csv(output_dir / "reference_checks.csv", collected.reference_rows)
        atomic_write_json(output_dir / "source_manifest.json", source)
        files = [
            "calibration_scores.csv",
            "calibration_controls.csv",
            "calibration_group_scores.csv",
            "calibration_proposals.csv",
            "calibration_audits.csv",
            "calibration_error_relations_fit.csv",
            "calibration_error_relations_time.csv",
            "calibration_modal_records.csv",
            "calibration_modal_summary.json",
            "transfer_floors.csv",
            "reference_checks.csv",
            "source_manifest.json",
        ]
        artifact_hashes = sha256_files(files, root=output_dir)
        payload = with_payload_sha256(
            {
                "schema": CALIBRATION_SCHEMA,
                "working_id": WORKING_ID,
                "status": "qualified"
                if selection["status"] == "qualified"
                else "stopped",
                "population_status": "adaptive_open_validation",
                "selection": selection,
                "evidence": evidence,
                "closure_by_policy": closure_by_policy,
                "modal_summary_payload_sha256": modal_summary["payload_sha256"],
                "modal_records_sha256": sha256_file(modal_path),
                "modal_summary_sha256": sha256_file(modal_summary_path),
                "source_manifest_payload_sha256": source["payload_sha256"],
                "source_manifest_sha256": sha256_file(
                    output_dir / "source_manifest.json"
                ),
                "execution": collected.execution,
                "collection_maxima": collected.maxima,
                "evaluation_targets_loaded": False,
                "artifact_hashes": artifact_hashes,
                "claim_boundary": (
                    "Adaptive dynamic-FV open-validation calibration; fixed gain, no "
                    "training, recurrence, bump, sealed population, or transfer claim."
                ),
            }
        )
        atomic_write_json(output_dir / "calibration.json", payload)
        return payload, 0 if selection["status"] == "qualified" else 3
    finally:
        base._close_runtime(runtime)


def _require_calibration(
    path: Path, *, output_dir: Path | None = None
) -> dict[str, Any]:
    payload = _read_json(path)
    verify_payload_sha256(payload)
    if (
        payload.get("schema") != CALIBRATION_SCHEMA
        or payload.get("status") != "qualified"
        or payload.get("selection", {}).get("status") != "qualified"
        or payload.get("selection", {}).get("selected_policy") not in NONZERO_POLICIES
    ):
        raise ValueError("calibration did not qualify; evaluation stays closed")
    root = path.parent if output_dir is None else output_dir
    expected = payload.get("artifact_hashes")
    if (
        not isinstance(expected, Mapping)
        or sha256_files(tuple(expected), root=root) != expected
    ):
        raise ValueError("calibration artifact inventory or hashes differ")
    if payload.get("source_manifest_payload_sha256") is None:
        raise ValueError("calibration source-manifest identity is missing")
    return payload


def _verify_source_manifest(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    payload = _read_json(path)
    verify_payload_sha256(payload)
    if payload.get("schema") != SOURCE_MANIFEST_SCHEMA:
        raise ValueError("unsupported fine-discrepancy source manifest")
    readiness = _verify_readiness(args.readiness)
    if payload.get("source_sha256") != _source_hashes():
        raise ValueError("source differs from the frozen calibration manifest")
    if payload.get("readiness_payload_sha256") != readiness["payload_sha256"]:
        raise ValueError("readiness differs from the frozen calibration manifest")
    return payload


def run_teacher_evaluation(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    calibration = _require_calibration(args.calibration)
    frozen_source = _verify_source_manifest(args.source_manifest, args)
    if sha256_file(args.source_manifest) != calibration["source_manifest_sha256"]:
        raise ValueError("calibration source-manifest file differs")
    if frozen_source["payload_sha256"] != calibration["source_manifest_payload_sha256"]:
        raise ValueError("calibration source-manifest payload differs")
    selected_policy = calibration["selection"]["selected_policy"]
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=False)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    runtime, source = _build_runtime(args)
    try:
        if source != frozen_source:
            raise ValueError("runtime source manifest differs from calibration")
        collected = base._collect(
            runtime,
            case_ids=EVALUATION_CASE_IDS,
            input_calls=ALL_INPUT_CALLS,
            coefficients=None,
            phase="fine_discrepancy_teacher_evaluation",
        )
        controls = _reconstruct_controls(
            runtime,
            collected.snapshots,
            policies=(selected_policy,),
        )
        all_payload = score_candidate_inventory(
            collected.snapshots,
            label="evaluation_all",
            expected_case_ids=EVALUATION_CASE_IDS,
            expected_input_calls=ALL_INPUT_CALLS,
            policy=selected_policy,
            resolution=NATIVE_RESOLUTION,
            projector=runtime.native_projector,
        )
        early_payload = score_candidate_inventory(
            [row for row in collected.snapshots if row.input_call in EARLY_CALLS],
            label="evaluation_early",
            expected_case_ids=EVALUATION_CASE_IDS,
            expected_input_calls=EARLY_CALLS,
            policy=selected_policy,
            resolution=NATIVE_RESOLUTION,
            projector=runtime.native_projector,
        )
        late_payload = score_candidate_inventory(
            [row for row in collected.snapshots if row.input_call in LATE_CALLS],
            label="evaluation_late",
            expected_case_ids=EVALUATION_CASE_IDS,
            expected_input_calls=LATE_CALLS,
            policy=selected_policy,
            resolution=NATIVE_RESOLUTION,
            projector=runtime.native_projector,
        )
        control_rows = _build_teacher_controls(
            policy=selected_policy,
            score_payload=all_payload,
            controls=controls,
            case_ids=EVALUATION_CASE_IDS,
            input_calls=ALL_INPUT_CALLS,
        )
        closure_pass, closure = _candidate_closure(
            policy=selected_policy,
            score_payloads=(all_payload, early_payload, late_payload),
            audit_rows=controls["audit"],
            collection_maxima=collected.maxima,
        )
        full = _population_row(all_payload, "full")
        rank8 = _population_row(all_payload, "rank8_parallel")
        early_rank8 = _population_row(early_payload, "rank8_parallel")
        late_rank8 = _population_row(late_payload, "rank8_parallel")
        full_cases = {row["case_id"]: row for row in _case_rows(all_payload, "full")}
        rank8_cases = {
            row["case_id"]: row for row in _case_rows(all_payload, "rank8_parallel")
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
            "calibration_qualified": calibration["status"] == "qualified",
            "inventory_exact": (
                len(collected.snapshots)
                == len(EVALUATION_CASE_IDS) * len(ALL_INPUT_CALLS)
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
            and all(_control_passed(row) for row in control_rows),
            "all_proposals_finite_admissible": bool(proposals)
            and all(row["finite"] and row["admissible"] for row in proposals),
            "all_audits_resolved": bool(audits)
            and all(row["status"] == "ok" for row in audits),
            "closure_passed": closure_pass,
            "source_and_artifacts_exact": source == frozen_source,
        }
        gate = teacher_forced_gate(checks)
        modal_path, modal_summary_path, modal_summary = _write_modal_outputs(
            output_dir,
            collected.snapshots,
            runtime.native_projector,
            prefix="evaluation",
        )
        score_rows = [
            *({"policy": selected_policy, **row} for row in all_payload["rows"]),
            *({"policy": selected_policy, **row} for row in early_payload["rows"]),
            *({"policy": selected_policy, **row} for row in late_payload["rows"]),
        ]
        write_csv(output_dir / "teacher_scores.csv", score_rows)
        write_csv(output_dir / "teacher_controls.csv", control_rows)
        write_csv(output_dir / "teacher_front_rows.csv", controls["front"])
        write_csv(output_dir / "teacher_integral_rows.csv", controls["integral"])
        write_csv(output_dir / "teacher_proposals.csv", proposals)
        write_csv(output_dir / "teacher_audits.csv", audits)
        write_csv(
            output_dir / "teacher_error_relations_early.csv",
            error_relation_rows(
                [row for row in collected.snapshots if row.input_call < 20],
                cell="case_only",
                resolution=NATIVE_RESOLUTION,
                projector=runtime.native_projector,
            ),
        )
        write_csv(
            output_dir / "teacher_error_relations_late.csv",
            error_relation_rows(
                [row for row in collected.snapshots if row.input_call >= 20],
                cell="joint_held_out",
                resolution=NATIVE_RESOLUTION,
                projector=runtime.native_projector,
            ),
        )
        write_csv(output_dir / "transfer_floors.csv", collected.floor_rows)
        write_csv(output_dir / "reference_checks.csv", collected.reference_rows)
        files = [
            "teacher_scores.csv",
            "teacher_controls.csv",
            "teacher_front_rows.csv",
            "teacher_integral_rows.csv",
            "teacher_proposals.csv",
            "teacher_audits.csv",
            "teacher_error_relations_early.csv",
            "teacher_error_relations_late.csv",
            "evaluation_modal_records.csv",
            "evaluation_modal_summary.json",
            "transfer_floors.csv",
            "reference_checks.csv",
        ]
        artifact_hashes = sha256_files(files, root=output_dir)
        payload = with_payload_sha256(
            {
                "schema": TEACHER_SCHEMA,
                "working_id": WORKING_ID,
                "status": "qualified"
                if gate["recurrent_pilot_authorized"]
                else "stopped",
                "population_status": "adaptive_open_validation",
                "selected_policy": selected_policy,
                "teacher_gate": gate,
                "population_scores": {
                    "all_full": dict(full),
                    "all_rank8": dict(rank8),
                    "early_rank8": dict(early_rank8),
                    "late_rank8": dict(late_rank8),
                    "case_win_count": case_wins,
                },
                "failed_controls": [
                    row for row in control_rows if not _control_passed(row)
                ],
                "closure": closure,
                "modal_summary_payload_sha256": modal_summary["payload_sha256"],
                "modal_records_sha256": sha256_file(modal_path),
                "modal_summary_sha256": sha256_file(modal_summary_path),
                "calibration_sha256": sha256_file(args.calibration),
                "calibration_payload_sha256": calibration["payload_sha256"],
                "source_manifest_sha256": sha256_file(args.source_manifest),
                "source_manifest_payload_sha256": source["payload_sha256"],
                "execution": collected.execution,
                "collection_maxima": collected.maxima,
                "artifact_hashes": artifact_hashes,
                "recurrence_executed": False,
                "claim_boundary": (
                    "Adaptive teacher-forced dynamic-FV result only; passing authorizes "
                    "the registered open H30 pilot, not a fresh or cross-family claim."
                ),
            }
        )
        atomic_write_json(output_dir / "teacher_evaluation.json", payload)
        return payload, 0 if gate["recurrent_pilot_authorized"] else 4
    finally:
        base._close_runtime(runtime)


def _require_teacher(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    verify_payload_sha256(payload)
    if (
        payload.get("schema") != TEACHER_SCHEMA
        or payload.get("status") != "qualified"
        or payload.get("teacher_gate", {}).get("recurrent_pilot_authorized") is not True
        or payload.get("selected_policy") not in NONZERO_POLICIES
    ):
        raise ValueError("teacher-forced gate did not authorize the recurrent pilot")
    expected = payload.get("artifact_hashes")
    if (
        not isinstance(expected, Mapping)
        or sha256_files(tuple(expected), root=path.parent) != expected
    ):
        raise ValueError("teacher artifact inventory or hashes differ")
    return payload


def _account_execution(
    execution: dict[str, Any],
    timing: Mapping[str, Any],
    *,
    resolution: tuple[int, int],
) -> None:
    forward = timing["forward_seconds"]
    if resolution == NATIVE_RESOLUTION:
        label = "native"
    elif resolution == RESOLUTION_CONTRACT.fine:
        label = "fine"
    else:
        raise ValueError("rollout execution received an unregistered resolution")
    execution["logical_model_calls"] += 1
    execution["actual_forward_passes"] += len(forward)
    execution["total_forward_seconds"] += float(sum(forward))
    execution[f"{label}_logical_calls"] += 1
    execution[f"{label}_actual_forward_passes"] += len(forward)
    execution[f"{label}_forward_seconds"] += float(sum(forward))
    peak = timing.get("peak_gpu_memory_bytes")
    if peak is not None:
        execution["maximum_peak_gpu_memory_bytes"] = max(
            execution["maximum_peak_gpu_memory_bytes"],
            int(peak),
        )


def _rollout_arm(
    runtime: Any,
    *,
    case_id: str,
    policy: CorrectionPolicy,
    horizon: int,
) -> dict[str, Any]:
    if policy not in {"zero", *NONZERO_POLICIES}:
        raise ValueError("unsupported rollout policy")
    reference, reference_check = load_resolution_reference(
        runtime.args.family_root,
        runtime.args.multires_reference_root,
        runtime.store,
        runtime.manifest,
        case_id,
        training_resolution=NATIVE_RESOLUTION,
    )
    reference_resolution = tuple(
        int(value) for value in reference["retained_resolution"]
    )
    initial = reference_at_resolution(
        reference["conservative_states"][0],
        reference_resolution=reference_resolution,
        target_resolution=NATIVE_RESOLUTION,
    )
    if initial is None:
        raise ValueError("native initial reference is unavailable")
    state = np.asarray(initial, dtype=np.float64)
    states = [np.array(state, copy=True)]
    rows: list[dict[str, Any]] = []
    native_geometry = runtime.geometry_by_resolution[NATIVE_RESOLUTION]
    volumes = np.asarray(native_geometry.node_measures, dtype=np.float64)
    state_scale = np.asarray(runtime.normalization.state_scale, dtype=np.float64)
    residual_scale = np.asarray(runtime.normalization.residual_scale, dtype=np.float64)
    cumulative_defect = np.zeros_like(state)
    execution = {
        "logical_model_calls": 0,
        "actual_forward_passes": 0,
        "total_forward_seconds": 0.0,
        "maximum_peak_gpu_memory_bytes": 0,
        "native_logical_calls": 0,
        "fine_logical_calls": 0,
        "native_actual_forward_passes": 0,
        "fine_actual_forward_passes": 0,
        "native_forward_seconds": 0.0,
        "fine_forward_seconds": 0.0,
    }
    maxima = {
        "pre_model_nesting": 0.0,
        "post_fp32_nesting": 0.0,
        "recurrence": 0.0,
        "correction_mean": 0.0,
        "correction_excluded": 0.0,
        "correction_modal_reconstruction": 0.0,
        "correction_ratio": 0.0,
    }
    first_invalid_call = None
    first_nonfinite_call = None
    started = perf_counter()

    for input_call in range(horizon):
        current_reference = reference_at_resolution(
            reference["conservative_states"][input_call * 2],
            reference_resolution=reference_resolution,
            target_resolution=NATIVE_RESOLUTION,
        )
        target = reference_at_resolution(
            reference["conservative_states"][(input_call + 1) * 2],
            reference_resolution=reference_resolution,
            target_resolution=NATIVE_RESOLUTION,
        )
        if current_reference is None or target is None:
            raise ValueError("native rollout reference is unavailable")
        current_reference = np.asarray(current_reference, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)

        if policy == "zero":
            prediction, timing = base.predict_resolution_sample(
                runtime.model,
                runtime.sample_by_resolution[NATIVE_RESOLUTION],
                state,
                device=runtime.device,
                amp="none",
                repeats=1,
            )
            _account_execution(
                execution,
                timing,
                resolution=NATIVE_RESOLUTION,
            )
            next_state = np.asarray(prediction, dtype=np.float64)
            applied_increment = next_state - state
            correction = np.zeros_like(state)
            audit = {
                "status": "zero",
                "cap_active": False,
                "correction_to_native_increment": 0.0,
                "maximum_scaled_component_mean_abs": 0.0,
                "maximum_excluded_abs": 0.0,
                "maximum_modal_reconstruction_abs": 0.0,
                "constant_mode_energy_fraction": 0.0,
            }
        else:

            def predictor(resolution, value):
                prediction, timing = base.predict_resolution_sample(
                    runtime.model,
                    runtime.sample_by_resolution[resolution],
                    value,
                    device=runtime.device,
                    amp="none",
                    repeats=1,
                )
                _account_execution(execution, timing, resolution=resolution)
                return prediction

            step = synchronized_fine_discrepancy_step(
                state,
                contract=RESOLUTION_CONTRACT,
                projector=runtime.native_projector,
                predictor=predictor,
                policy=policy,
                volumes=volumes,
                component_scale=residual_scale,
            )
            next_state = step.next_native_state
            applied_increment = next_state - state
            correction = step.correction
            audit = asdict(step.audit)
            maxima["pre_model_nesting"] = max(
                maxima["pre_model_nesting"],
                float(
                    step.prepared_inputs.nesting_floors[
                        "pre_model_fine_to_native_max_abs"
                    ]
                ),
            )
            maxima["post_fp32_nesting"] = max(
                maxima["post_fp32_nesting"],
                float(
                    step.prepared_inputs.nesting_floors[
                        "post_fp32_fine_to_native_max_abs"
                    ]
                ),
            )
            maxima["recurrence"] = max(
                maxima["recurrence"],
                _maximum_absolute(
                    next_state
                    - np.asarray(step.predictions[NATIVE_RESOLUTION], dtype=np.float64)
                    - correction
                ),
            )
            maxima["correction_mean"] = max(
                maxima["correction_mean"],
                float(audit["maximum_scaled_component_mean_abs"]),
            )
            maxima["correction_excluded"] = max(
                maxima["correction_excluded"],
                float(audit["maximum_excluded_abs"]),
            )
            maxima["correction_modal_reconstruction"] = max(
                maxima["correction_modal_reconstruction"],
                float(audit["maximum_modal_reconstruction_abs"]),
            )
            if audit["correction_to_native_increment"] is not None:
                maxima["correction_ratio"] = max(
                    maxima["correction_ratio"],
                    float(audit["correction_to_native_increment"]),
                )

        true_increment = target - current_reference
        defect = applied_increment - true_increment
        cumulative_defect += defect
        state_error = next_state - target
        state_parts, _ = runtime.native_projector.split(state_error)
        masks, _, _ = shock_vortex_regions(
            target,
            native_geometry.nodes,
            resolution=NATIVE_RESOLUTION,
            gamma=runtime.normalization.gamma,
        )
        admissibility = conservative_admissibility_summary(
            next_state,
            gamma=runtime.normalization.gamma,
        )
        finite = bool(np.isfinite(next_state).all())
        admissible = bool(admissibility["admissible"])
        if not finite and first_nonfinite_call is None:
            first_nonfinite_call = input_call
        if (not finite or not admissible) and first_invalid_call is None:
            first_invalid_call = input_call
        front = base._front_errors(next_state, target, runtime)
        integral = base._physical_integral(state_error, volumes)
        correction_integral = base._physical_integral(correction, volumes)
        correction_coordinates = modal_coordinates(
            correction,
            runtime.native_projector,
            component_scale=residual_scale,
        )
        row = {
            "case_id": case_id,
            "policy": policy,
            "input_call": input_call,
            "output_call": input_call + 1,
            "state_error": weighted_scaled_rms(
                state_error,
                volumes=volumes,
                component_scale=state_scale,
            ),
            "rank8_state_error": weighted_scaled_rms(
                state_parts["parallel"],
                volumes=volumes,
                component_scale=state_scale,
            ),
            "increment_defect": weighted_scaled_rms(
                defect,
                volumes=volumes,
                component_scale=residual_scale,
            ),
            "cumulative_defect": weighted_scaled_rms(
                cumulative_defect,
                volumes=volumes,
                component_scale=residual_scale,
            ),
            "correction_rms": weighted_scaled_rms(
                correction,
                volumes=volumes,
                component_scale=residual_scale,
            ),
            "finite": finite,
            "admissible": admissible,
            "minimum_density": admissibility["minimum_density"],
            "minimum_pressure": admissibility["minimum_pressure"],
            "minimum_internal_energy": admissibility["minimum_internal_energy"],
            "cap_active": bool(audit["cap_active"]),
            "correction_status": audit["status"],
            "correction_to_native_increment": audit["correction_to_native_increment"],
            "constant_mode_energy_fraction": audit["constant_mode_energy_fraction"],
            "correction_modal_coordinate_square": float(
                np.sum(np.square(correction_coordinates))
            ),
            "front_position": front["front_position"],
            "front_strength_log_ratio": front["front_strength_log_ratio"],
            "front_thickness_log_ratio": front["front_thickness_log_ratio"],
        }
        for component, name in enumerate(INTEGRAL_COMPONENT_NAMES):
            row[f"component_{name}_state_error"] = weighted_scaled_rms(
                state_error[:, component : component + 1],
                volumes=volumes,
                component_scale=state_scale[component : component + 1],
            )
            row[f"integral_{name}_error"] = float(integral[component])
            row[f"correction_integral_{name}"] = float(correction_integral[component])
        for mode_index in range(correction_coordinates.shape[0]):
            row[f"correction_mode_{mode_index}_coordinate_square"] = float(
                np.sum(np.square(correction_coordinates[mode_index]))
            )
        for key, mask_name in (
            ("boundary_state_error", "boundary_le_0.05"),
            ("shock_state_error", "partition_shock"),
            ("vortex_state_error", "partition_vortex"),
            ("smooth_state_error", "partition_smooth"),
        ):
            row[key] = weighted_scaled_rms(
                state_error,
                volumes=volumes,
                component_scale=state_scale,
                mask=masks[mask_name],
            )
        rows.append(row)
        state = next_state
        states.append(np.array(state, copy=True))
        if not finite or not admissible:
            break

    execution["wall_seconds"] = perf_counter() - started
    execution["completed_calls"] = len(rows)
    return {
        "case_id": case_id,
        "policy": policy,
        "rows": rows,
        "states": states,
        "execution": execution,
        "maxima": maxima,
        "first_invalid_call": first_invalid_call,
        "first_nonfinite_call": first_nonfinite_call,
        "reference_check": reference_check,
    }


def _paired_rollout_controls(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    raw = [row for row in rows if row["policy"] == "zero"]
    corrected = [row for row in rows if row["policy"] != "zero"]
    raw_lookup = {(row["case_id"], row["input_call"]): row for row in raw}
    corrected_lookup = {(row["case_id"], row["input_call"]): row for row in corrected}
    if set(raw_lookup) != set(corrected_lookup):
        raise ValueError("raw and corrected rollout inventories differ")
    case_rows = []
    for case_id in EVALUATION_CASE_IDS:
        raw_case = sorted(
            [row for row in raw if row["case_id"] == case_id],
            key=lambda row: row["input_call"],
        )
        corrected_case = sorted(
            [row for row in corrected if row["case_id"] == case_id],
            key=lambda row: row["input_call"],
        )
        if len(raw_case) != len(ALL_INPUT_CALLS) or len(corrected_case) != len(
            ALL_INPUT_CALLS
        ):
            raise ValueError("one rollout case did not reach H30")

        def rms_ratio(
            key: str,
            raw_rows: Sequence[Mapping[str, Any]],
            corrected_rows: Sequence[Mapping[str, Any]],
        ) -> float:
            zero = math.sqrt(np.mean([float(row[key]) ** 2 for row in raw_rows]))
            candidate = math.sqrt(
                np.mean([float(row[key]) ** 2 for row in corrected_rows])
            )
            if zero <= DENOMINATOR_FLOOR:
                return 1.0 if candidate <= DENOMINATOR_FLOOR else float("inf")
            return candidate / zero

        raw_endpoint = raw_case[-1]
        corrected_endpoint = corrected_case[-1]
        endpoint_ratio = corrected_endpoint["state_error"] / raw_endpoint["state_error"]
        case_rows.append(
            {
                "case_id": case_id,
                "endpoint_state_ratio": endpoint_ratio,
                "trajectory_state_rms_ratio": rms_ratio(
                    "state_error", raw_case, corrected_case
                ),
                "increment_defect_rms_ratio": rms_ratio(
                    "increment_defect", raw_case, corrected_case
                ),
                "endpoint_cumulative_defect_ratio": (
                    corrected_endpoint["cumulative_defect"]
                    / raw_endpoint["cumulative_defect"]
                ),
                "raw_endpoint_state_error": raw_endpoint["state_error"],
                "corrected_endpoint_state_error": corrected_endpoint["state_error"],
            }
        )

    controls: list[dict[str, Any]] = []
    endpoint_keys = (
        "rank8_state_error",
        "boundary_state_error",
        "shock_state_error",
        "vortex_state_error",
        "smooth_state_error",
        "front_position",
        "front_strength_log_ratio",
        "front_thickness_log_ratio",
        *(f"component_{name}_state_error" for name in INTEGRAL_COMPONENT_NAMES),
    )
    for key in endpoint_keys:
        for scope in ("population", *EVALUATION_CASE_IDS):
            cases = EVALUATION_CASE_IDS if scope == "population" else (scope,)
            trajectory_rows = [
                {
                    "zero_error": raw_lookup[(case_id, input_call)][key],
                    "corrected_error": corrected_lookup[(case_id, input_call)][key],
                }
                for case_id in cases
                for input_call in ALL_INPUT_CALLS
            ]
            endpoint_rows = [
                {
                    "zero_error": raw_lookup[(case_id, ALL_INPUT_CALLS[-1])][key],
                    "corrected_error": corrected_lookup[(case_id, ALL_INPUT_CALLS[-1])][
                        key
                    ],
                }
                for case_id in cases
            ]
            controls.append(
                _rms_control(
                    key=f"trajectory::{key}",
                    scope=scope,
                    rows=trajectory_rows,
                )
            )
            controls.append(
                _rms_control(
                    key=f"endpoint::{key}",
                    scope=scope,
                    rows=endpoint_rows,
                )
            )
    for name in INTEGRAL_COMPONENT_NAMES:
        key = f"integral_{name}_error"
        for scope in ("population", *EVALUATION_CASE_IDS):
            cases = EVALUATION_CASE_IDS if scope == "population" else (scope,)
            all_rows = [
                {
                    "zero_error": raw_lookup[(case_id, input_call)][key],
                    "corrected_error": corrected_lookup[(case_id, input_call)][key],
                }
                for case_id in cases
                for input_call in ALL_INPUT_CALLS
            ]
            endpoint_rows = [
                {
                    "zero_error": raw_lookup[(case_id, ALL_INPUT_CALLS[-1])][key],
                    "corrected_error": corrected_lookup[(case_id, ALL_INPUT_CALLS[-1])][
                        key
                    ],
                }
                for case_id in cases
            ]
            controls.append(
                _rms_control(key=f"integral_rms::{name}", scope=scope, rows=all_rows)
            )
            controls.append(
                _rms_control(
                    key=f"integral_endpoint::{name}",
                    scope=scope,
                    rows=endpoint_rows,
                )
            )
    population = {
        "median_endpoint_state_ratio": float(
            np.median([row["endpoint_state_ratio"] for row in case_rows])
        ),
        "maximum_endpoint_state_ratio": max(
            row["endpoint_state_ratio"] for row in case_rows
        ),
        "endpoint_win_count": sum(
            row["endpoint_state_ratio"] <= 1.0 for row in case_rows
        ),
        "aggregate_state_rms_ratio": math.sqrt(
            sum(float(row["state_error"]) ** 2 for row in corrected)
            / sum(float(row["state_error"]) ** 2 for row in raw)
        ),
        "aggregate_increment_defect_rms_ratio": math.sqrt(
            sum(float(row["increment_defect"]) ** 2 for row in corrected)
            / sum(float(row["increment_defect"]) ** 2 for row in raw)
        ),
        "median_endpoint_cumulative_defect_ratio": float(
            np.median([row["endpoint_cumulative_defect_ratio"] for row in case_rows])
        ),
    }
    return case_rows, controls, population


def run_rollout(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    teacher = _require_teacher(args.teacher_evaluation)
    calibration = _require_calibration(args.calibration)
    frozen_source = _verify_source_manifest(args.source_manifest, args)
    if teacher["calibration_payload_sha256"] != calibration["payload_sha256"]:
        raise ValueError("teacher and calibration identities differ")
    if teacher["source_manifest_payload_sha256"] != frozen_source["payload_sha256"]:
        raise ValueError("teacher and source-manifest identities differ")
    policy = teacher["selected_policy"]
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=False)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    runtime, source = _build_runtime(args)
    try:
        if source != frozen_source:
            raise ValueError("rollout runtime source differs from teacher source")
        prefix_a = _rollout_arm(
            runtime,
            case_id=EVALUATION_CASE_IDS[0],
            policy=policy,
            horizon=2,
        )
        prefix_b = _rollout_arm(
            runtime,
            case_id=EVALUATION_CASE_IDS[0],
            policy=policy,
            horizon=2,
        )
        prefix_abs = max(
            _maximum_absolute(left - right)
            for left, right in zip(prefix_a["states"], prefix_b["states"], strict=True)
        )
        rollouts = []
        for case_id in EVALUATION_CASE_IDS:
            print(f"fine-discrepancy rollout: {case_id} raw", flush=True)
            rollouts.append(
                _rollout_arm(runtime, case_id=case_id, policy="zero", horizon=30)
            )
            print(f"fine-discrepancy rollout: {case_id} {policy}", flush=True)
            rollouts.append(
                _rollout_arm(runtime, case_id=case_id, policy=policy, horizon=30)
            )
        rows = [row for rollout in rollouts for row in rollout["rows"]]
        case_rows, control_rows, population = _paired_rollout_controls(rows)
        completed = all(
            rollout["execution"]["completed_calls"] == 30
            and rollout["first_invalid_call"] is None
            and rollout["first_nonfinite_call"] is None
            for rollout in rollouts
        )
        corrected_rollouts = [row for row in rollouts if row["policy"] == policy]
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
            and (
                policy != "rank7_fine_away_half"
                or row["maxima"]["correction_mean"] <= 1.0e-12
            )
            for row in corrected_rollouts
        )
        checks = {
            "teacher_gate_passed": teacher["teacher_gate"]["recurrent_pilot_authorized"]
            is True,
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
            and all(_control_passed(row) for row in control_rows),
            "two_call_common_source_closure": two_call_closure,
            "correction_audits_pass": audit_pass,
            "deterministic_prefix_exact": prefix_abs == 0.0,
            "source_and_artifacts_exact": source == frozen_source,
        }
        gate = recurrent_gate(checks)
        write_csv(output_dir / "rollout_call_metrics.csv", rows)
        write_csv(output_dir / "rollout_case_metrics.csv", case_rows)
        write_csv(output_dir / "rollout_controls.csv", control_rows)
        execution_rows = [
            {
                "case_id": row["case_id"],
                "policy": row["policy"],
                **row["execution"],
            }
            for row in rollouts
        ]
        write_csv(output_dir / "rollout_execution.csv", execution_rows)
        reference_rows = [
            {
                "case_id": row["case_id"],
                "policy": row["policy"],
                **row["reference_check"],
            }
            for row in rollouts
        ]
        write_csv(output_dir / "reference_checks.csv", reference_rows)
        files = [
            "rollout_call_metrics.csv",
            "rollout_case_metrics.csv",
            "rollout_controls.csv",
            "rollout_execution.csv",
            "reference_checks.csv",
        ]
        artifact_hashes = sha256_files(files, root=output_dir)
        raw_execution = [
            row["execution"] for row in rollouts if row["policy"] == "zero"
        ]
        corrected_execution = [row["execution"] for row in corrected_rollouts]
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
        cost["wall_time_ratio"] = (
            cost["corrected_wall_seconds"] / cost["raw_wall_seconds"]
        )
        cost["peak_memory_ratio"] = (
            cost["corrected_peak_memory_bytes"] / cost["raw_peak_memory_bytes"]
        )
        cost["corrected_fine_forward_time_fraction"] = (
            cost["corrected_fine_forward_seconds"] / cost["corrected_forward_seconds"]
        )
        raw_endpoint_rms = math.sqrt(
            np.mean(
                [
                    float(row["state_error"]) ** 2
                    for row in rows
                    if row["policy"] == "zero"
                    and int(row["input_call"]) == ALL_INPUT_CALLS[-1]
                ]
            )
        )
        corrected_endpoint_rms = math.sqrt(
            np.mean(
                [
                    float(row["state_error"]) ** 2
                    for row in rows
                    if row["policy"] == policy
                    and int(row["input_call"]) == ALL_INPUT_CALLS[-1]
                ]
            )
        )
        cost["raw_endpoint_state_rms"] = raw_endpoint_rms
        cost["corrected_endpoint_state_rms"] = corrected_endpoint_rms
        raw_dominates_or_ties = (
            raw_endpoint_rms <= corrected_endpoint_rms
            and cost["raw_wall_seconds"] <= cost["corrected_wall_seconds"]
            and cost["raw_peak_memory_bytes"] <= cost["corrected_peak_memory_bytes"]
        )
        raw_strictly_better = (
            raw_endpoint_rms < corrected_endpoint_rms
            or cost["raw_wall_seconds"] < cost["corrected_wall_seconds"]
            or cost["raw_peak_memory_bytes"] < cost["corrected_peak_memory_bytes"]
        )
        cost["corrected_pareto_dominated_by_raw"] = bool(
            raw_dominates_or_ties and raw_strictly_better
        )
        payload = with_payload_sha256(
            {
                "schema": ROLLOUT_SCHEMA,
                "working_id": WORKING_ID,
                "status": "complete",
                "population_status": "adaptive_open_validation",
                "selected_policy": policy,
                "recurrent_gate": gate,
                "population": population,
                "failed_controls": [
                    row for row in control_rows if not _control_passed(row)
                ],
                "deterministic_prefix_max_abs": prefix_abs,
                "cost": cost,
                "teacher_evaluation_sha256": sha256_file(args.teacher_evaluation),
                "teacher_evaluation_payload_sha256": teacher["payload_sha256"],
                "calibration_sha256": sha256_file(args.calibration),
                "calibration_payload_sha256": calibration["payload_sha256"],
                "source_manifest_sha256": sha256_file(args.source_manifest),
                "source_manifest_payload_sha256": source["payload_sha256"],
                "artifact_hashes": artifact_hashes,
                "claim_boundary": (
                    "Adaptive dynamic-FV H30 result on reused open validation; no "
                    "fresh confirmation, family transfer, conservation, or sealed claim."
                ),
            }
        )
        atomic_write_json(output_dir / "rollout.json", payload)
        return payload, 0 if gate["adaptive_recurrent_pass"] else 5
    finally:
        base._close_runtime(runtime)


def _add_external_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--readiness", type=Path, required=True)
    parser.add_argument("--parent-readiness", type=Path, required=True)
    parser.add_argument("--base-readiness", type=Path, required=True)
    parser.add_argument("--family-root", type=Path, required=True)
    parser.add_argument("--multires-reference-root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--normalization-file", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--d063-run-contract", type=Path, required=True)
    parser.add_argument("--d063-summary", type=Path, required=True)


def _add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--repeat-forward", type=int, default=2)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    synthetic = commands.add_parser("synthetic")
    synthetic.add_argument("--seed", type=int, default=0)

    readiness = commands.add_parser("readiness")
    readiness.add_argument("--output", type=Path, required=True)
    readiness.add_argument("--parent-readiness", type=Path, required=True)
    readiness.add_argument("--prior-calibration", type=Path, required=True)
    readiness.add_argument("--prior-evaluation", type=Path, required=True)
    readiness.add_argument("--synthetic-seed", type=int, default=0)
    readiness.add_argument("--focused-test-command", required=True)
    readiness.add_argument("--focused-test-result", required=True)

    preflight = commands.add_parser("preflight")
    _add_external_arguments(preflight)
    preflight.add_argument("--output", type=Path, required=True)

    smoke = commands.add_parser("smoke")
    _add_external_arguments(smoke)
    _add_runtime_arguments(smoke)
    smoke.add_argument("--output", type=Path, required=True)

    calibration = commands.add_parser("calibrate")
    _add_external_arguments(calibration)
    _add_runtime_arguments(calibration)
    calibration.add_argument("--output-dir", type=Path, required=True)

    evaluation = commands.add_parser("evaluate")
    _add_external_arguments(evaluation)
    _add_runtime_arguments(evaluation)
    evaluation.add_argument("--source-manifest", type=Path, required=True)
    evaluation.add_argument("--calibration", type=Path, required=True)
    evaluation.add_argument("--output-dir", type=Path, required=True)

    rollout = commands.add_parser("rollout")
    _add_external_arguments(rollout)
    _add_runtime_arguments(rollout)
    rollout.add_argument("--source-manifest", type=Path, required=True)
    rollout.add_argument("--calibration", type=Path, required=True)
    rollout.add_argument("--teacher-evaluation", type=Path, required=True)
    rollout.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "synthetic":
        payload = synthetic_summary(args.seed)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if payload["status"] == "passed" else 2
    if args.command == "readiness":
        payload = create_readiness(args)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if payload["status"] == "passed" else 2
    if args.command == "preflight":
        payload = run_preflight(args)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if payload["status"] == "passed" else 2
    if args.repeat_forward < 1:
        raise ValueError("repeat-forward must be positive")
    if args.command == "smoke":
        payload = run_smoke(args)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if payload["status"] == "passed" else 2
    if args.command == "calibrate":
        payload, exit_code = run_calibration(args)
    elif args.command == "evaluate":
        payload, exit_code = run_teacher_evaluation(args)
    elif args.command == "rollout":
        payload, exit_code = run_rollout(args)
    else:  # pragma: no cover
        raise AssertionError(f"unsupported command: {args.command}")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
