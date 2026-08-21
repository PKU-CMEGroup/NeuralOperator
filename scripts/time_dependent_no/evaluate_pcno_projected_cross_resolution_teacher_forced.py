#!/usr/bin/env python3
"""Run the calibration-first W26-L5 rank-8 projected A2 diagnostic.

The evaluation firewall verifies a qualified calibration artifact before any
evaluation reference or model runtime is constructed.  Every scientific case
is from the already reused open validation population, so outputs are adaptive
mechanism evidence and can never authorize recurrence or sealed-data access.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.time_dependent_no import (
    evaluate_pcno_cross_resolution_teacher_forced as base,
)
from scripts.time_dependent_no.evaluate_pcno_projected_cross_resolution_correction import (
    synthetic_smoke_summary as a1_synthetic_smoke_summary,
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
from utility.time_dependent_no.pcno_cross_resolution_correction import MODEL_NAMES
from utility.time_dependent_no.pcno_cross_resolution_teacher_forced import (
    ALL_INPUT_CALLS,
    CALIBRATION_CASE_IDS,
    CALIBRATION_GROUPS,
    CELL_CALLS,
    EVALUATION_CASE_IDS,
    FIT_INPUT_CALLS,
    HELD_OUT_INPUT_CALLS,
    ControlRatio,
    ProposalEvidence,
    error_relation_rows,
    verify_payload_sha256,
    with_payload_sha256,
)
from utility.time_dependent_no.pcno_projected_cross_resolution_correction import (
    projected_coefficient_stability,
    projected_native_prediction,
    score_projected_cell,
)
from utility.time_dependent_no.pcno_projected_cross_resolution_teacher_forced import (
    ProjectedCalibrationClosureEvidence,
    ProjectedStatisticRecord,
    adaptive_projected_gate,
    all_open_strength_groups,
    projected_grouped_crossfit_from_records,
    projected_statistic_record_from_dict,
    projected_statistic_record_to_dict,
    projected_statistic_records,
    qualify_projected_calibration,
    validate_projected_statistic_inventory,
)
from utility.time_dependent_no.pcno_resolution_transfer import (
    conservative_admissibility_summary,
    load_resolution_reference,
    reference_at_resolution,
)

WORKING_ID = "W26-L5-P1-R8-A2"
READINESS_SCHEMA = "pcno_projected_cross_resolution_teacher_forced_readiness_v1"
SOURCE_MANIFEST_SCHEMA = (
    "pcno_projected_cross_resolution_teacher_forced_source_manifest_v1"
)
STATISTICS_SCHEMA = "pcno_projected_cross_resolution_statistics_v1"
CALIBRATION_SCHEMA = "pcno_projected_cross_resolution_teacher_forced_calibration_v1"
EVALUATION_SCHEMA = "pcno_projected_cross_resolution_teacher_forced_evaluation_v1"
SMOKE_SCHEMA = "pcno_projected_cross_resolution_teacher_forced_smoke_v1"
PREFLIGHT_SCHEMA = "pcno_projected_cross_resolution_teacher_forced_preflight_v1"

EXPECTED_PROJECTED_A1_SHA256 = {
    "docs/time_dependent_no/W26_L5_RANK8_PROJECTED_CORRECTION_PREREGISTRATION.md": (
        "a4e80a1482b0c650ed3cda679ef5f71eaa0bef96e2d2ade07bf7de981d3a4664"
    ),
    "utility/time_dependent_no/pcno_projected_cross_resolution_correction.py": (
        "88b3697f597a692f70c3ca67a99b254a988c64959c5918a70d7548d3961bb4a3"
    ),
    "scripts/time_dependent_no/evaluate_pcno_projected_cross_resolution_correction.py": (
        "2eaaa00e0df4ce458e1089bcff0fb95ac2d72f966eba45457bc02e16c8af967b"
    ),
    "tests/time_dependent_no/test_pcno_projected_cross_resolution_correction.py": (
        "dca9d73a723de07fb398b882a3013a2a5c5807064c167d57965fd950205002c0"
    ),
}
EXPECTED_FROZEN_A2_SHA256 = {
    "docs/time_dependent_no/W26_L5_CROSS_RESOLUTION_PREREGISTRATION.md": (
        "157f02824b05eb346576fc90845edbd7fcb8b4fe2247b48e53d9168bc13fc2b7"
    ),
    "utility/time_dependent_no/pcno_cross_resolution_correction.py": (
        "1b5038b7b52f6eeb810263e194779a85039f97bef29220fcce49f60d7e04509d"
    ),
    "utility/time_dependent_no/pcno_cross_resolution_teacher_forced.py": (
        "b245aa3b9940a2966cc35a415c217f2076a8d4f42b4c19e95599891fee01450d"
    ),
    "scripts/time_dependent_no/evaluate_pcno_cross_resolution_correction.py": (
        "3cf3b13b3312ece83f2163b67efd30e433ca2fa2e68dd416bbd0d7a762fcab70"
    ),
    "scripts/time_dependent_no/evaluate_pcno_cross_resolution_teacher_forced.py": (
        "c741985291391fd04a0890b29ebfe62f775b99f154c34b51d29d7376c76d7030"
    ),
    "tests/time_dependent_no/test_pcno_cross_resolution_correction.py": (
        "9d2aa04cf8c955e4bfc02507e31a416239198dce126bdcb36595b6e93fbae9af"
    ),
    "tests/time_dependent_no/test_pcno_cross_resolution_teacher_forced.py": (
        "526ce7b783c1a6764e37d215480bd719b2fdde8e3c1a2d94c0da918684525b0e"
    ),
}
EXPECTED_PRIOR_A2_ARTIFACT_SHA256 = {
    "base_readiness": "3c9e4576deddaa0e5186d79369a16c73d48875948421f540f0aa3f43d2159549",
    "calibration": "00c37737eefb332695aae18d25c5f0641cdd5c4f9526852b9eadce547be02a29",
    "evaluation": "f02e76a8ee8b00db74d322922b4fd63d412042b26908825854b8b5cac7127c8f",
}
PROJECTED_A2_SOURCE_PATHS = (
    "utility/time_dependent_no/pcno_projected_cross_resolution_teacher_forced.py",
    "scripts/time_dependent_no/evaluate_pcno_projected_cross_resolution_teacher_forced.py",
    "tests/time_dependent_no/test_pcno_projected_cross_resolution_teacher_forced.py",
)
SOURCE_PATHS = (
    *EXPECTED_PROJECTED_A1_SHA256,
    *EXPECTED_FROZEN_A2_SHA256,
    *PROJECTED_A2_SOURCE_PATHS,
)


@dataclass(frozen=True)
class ReconstructedControls:
    front_rows: list[dict[str, Any]]
    integral_rows: list[dict[str, Any]]
    proposal_rows: list[dict[str, Any]]
    maxima: dict[str, float | int]
    wall_seconds: float


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


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON payload must be an object: {path.name}")
    return payload


def _source_hashes() -> dict[str, str]:
    return sha256_files(SOURCE_PATHS, root=ROOT)


def create_readiness(args: argparse.Namespace) -> dict[str, Any]:
    synthetic = a1_synthetic_smoke_summary(args.synthetic_seed)
    verify_payload_sha256(synthetic)
    base_readiness = base._verify_readiness(args.base_readiness)
    source_hashes = _source_hashes()
    prior_hashes = {
        "base_readiness": sha256_file(args.base_readiness),
        "calibration": sha256_file(args.prior_calibration),
        "evaluation": sha256_file(args.prior_evaluation),
    }
    checks = {
        "projected_a1_source_identity_exact": {
            path: source_hashes.get(path) for path in EXPECTED_PROJECTED_A1_SHA256
        }
        == EXPECTED_PROJECTED_A1_SHA256,
        "frozen_a2_source_identity_exact": {
            path: source_hashes.get(path) for path in EXPECTED_FROZEN_A2_SHA256
        }
        == EXPECTED_FROZEN_A2_SHA256,
        "projected_a2_source_inventory_exact": set(source_hashes) == set(SOURCE_PATHS),
        "prior_a2_artifact_identity_exact": prior_hashes
        == EXPECTED_PRIOR_A2_ARTIFACT_SHA256,
        "base_readiness_passed": base_readiness.get("status") == "passed",
        "focused_cpu_tests_passed": args.focused_test_result.startswith("passed"),
        "a1_synthetic_passed": synthetic.get("status") == "passed",
        "a1_synthetic_scientific_inputs_closed": (
            synthetic.get("checkpoint_loaded") is False
            and synthetic.get("dataset_loaded") is False
            and synthetic.get("reference_loaded") is False
        ),
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
            "focused_cpu_test_command": args.focused_test_command,
            "focused_cpu_test_result": args.focused_test_result,
            "a1_synthetic_payload_sha256": synthetic["payload_sha256"],
            "a1_synthetic_seed": int(args.synthetic_seed),
            "prior_a2_artifact_sha256": prior_hashes,
            "base_readiness_payload_sha256": base_readiness["payload_sha256"],
            "authorization": {
                "adaptive_open_teacher_forced": True,
                "training": False,
                "sealed_population": False,
                "recurrence": False,
                "d073_b": False,
                "bump": False,
                "gradient_change": False,
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
        raise ValueError("projected A2 readiness is absent or did not pass")
    if payload.get("source_sha256") != _source_hashes():
        raise ValueError("projected A2 source changed after readiness was recorded")
    return payload


def _base_args(args: argparse.Namespace) -> argparse.Namespace:
    values = vars(args).copy()
    values["readiness"] = args.base_readiness
    return argparse.Namespace(**values)


def _build_source_manifest(
    args: argparse.Namespace,
    readiness: Mapping[str, Any],
    base_source_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    base_readiness = base._verify_readiness(args.base_readiness)
    if (
        sha256_file(args.base_readiness)
        != EXPECTED_PRIOR_A2_ARTIFACT_SHA256["base_readiness"]
    ):
        raise ValueError("frozen base-A2 readiness file identity mismatch")
    if readiness.get("base_readiness_payload_sha256") != base_readiness.get(
        "payload_sha256"
    ):
        raise ValueError("projected readiness and base readiness differ")
    return with_payload_sha256(
        {
            "schema": SOURCE_MANIFEST_SCHEMA,
            "working_id": WORKING_ID,
            "population_status": "adaptive_open_validation",
            "readiness_file_sha256": sha256_file(args.readiness),
            "readiness_payload_sha256": readiness["payload_sha256"],
            "base_readiness_file_sha256": sha256_file(args.base_readiness),
            "base_readiness_payload_sha256": base_readiness["payload_sha256"],
            "source_sha256": _source_hashes(),
            "prior_a2_artifact_sha256": readiness["prior_a2_artifact_sha256"],
            "base_a2_source_manifest": base_source_manifest,
            "population": {
                "calibration_cases": list(CALIBRATION_CASE_IDS),
                "evaluation_cases": list(EVALUATION_CASE_IDS),
                "calibration_calls": list(FIT_INPUT_CALLS),
                "evaluation_calls": list(ALL_INPUT_CALLS),
                "all_open_lopo_groups": {
                    key: list(value)
                    for key, value in all_open_strength_groups().items()
                },
                "strength_ood_and_test": "sealed",
            },
            "candidate": {
                "projection": "rank8_parallel",
                "excluded_node_types": [1, 2, 3],
                "coefficients": "fixed calibration-only no-intercept scalars",
                "all_open_lopo_coefficients": "diagnostic_only_not_deployed",
                "true_error_at_inference": False,
            },
        }
    )


def _open_contract(args: argparse.Namespace):
    readiness = _verify_readiness(args.readiness)
    _, checkpoint, manifest, store, base_source = base._open_contract(_base_args(args))
    try:
        source_manifest = _build_source_manifest(args, readiness, base_source)
    except Exception:
        store.close()
        raise
    return checkpoint, manifest, store, source_manifest


def _build_runtime(args: argparse.Namespace):
    readiness = _verify_readiness(args.readiness)
    runtime = base._build_runtime(_base_args(args))
    try:
        source_manifest = _build_source_manifest(
            args,
            readiness,
            runtime.source_manifest,
        )
    except Exception:
        base._close_runtime(runtime)
        raise
    return runtime, source_manifest


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint, manifest, store, source_manifest = _open_contract(args)
    try:
        payload = with_payload_sha256(
            {
                "schema": PREFLIGHT_SCHEMA,
                "working_id": WORKING_ID,
                "status": "passed",
                "population_status": "adaptive_open_validation",
                "model_constructed": False,
                "reference_trajectory_arrays_loaded": False,
                "evaluation_targets_loaded": False,
                "source_manifest": source_manifest,
                "case_inventory_present": {
                    "calibration": all(
                        case_id in store.keys for case_id in CALIBRATION_CASE_IDS
                    ),
                    "evaluation": all(
                        case_id in store.keys for case_id in EVALUATION_CASE_IDS
                    ),
                },
                "family_splits": {
                    case_id: base.family_case_provenance(manifest, case_id)["split"]
                    for case_id in (*CALIBRATION_CASE_IDS, *EVALUATION_CASE_IDS)
                },
                "checkpoint_schema_version": checkpoint["checkpoint_schema_version"],
            }
        )
    finally:
        store.close()
    atomic_write_json(args.output, payload)
    return payload


def _statistics_payload(
    records: Sequence[ProjectedStatisticRecord],
    *,
    source_manifest_path: Path,
    source_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    return with_payload_sha256(
        {
            "schema": STATISTICS_SCHEMA,
            "working_id": WORKING_ID,
            "population_status": "adaptive_open_validation",
            "source_manifest_sha256": sha256_file(source_manifest_path),
            "source_manifest_payload_sha256": source_manifest["payload_sha256"],
            "case_ids": list(CALIBRATION_CASE_IDS),
            "input_calls": list(FIT_INPUT_CALLS),
            "record_count": len(records),
            "records": [
                projected_statistic_record_to_dict(record) for record in records
            ],
        }
    )


def _read_calibration_statistics(
    path: Path,
    *,
    expected_source_manifest_sha256: str,
    expected_payload_sha256: str,
) -> tuple[ProjectedStatisticRecord, ...]:
    payload = _read_json(path)
    verify_payload_sha256(payload)
    if (
        payload.get("schema") != STATISTICS_SCHEMA
        or payload.get("source_manifest_sha256") != expected_source_manifest_sha256
        or payload.get("payload_sha256") != expected_payload_sha256
        or payload.get("case_ids") != list(CALIBRATION_CASE_IDS)
        or payload.get("input_calls") != list(FIT_INPUT_CALLS)
    ):
        raise ValueError("calibration statistic artifact contract mismatch")
    values = payload.get("records")
    if not isinstance(values, list) or payload.get("record_count") != len(values):
        raise ValueError("calibration statistic record inventory is malformed")
    records = tuple(projected_statistic_record_from_dict(value) for value in values)
    return validate_projected_statistic_inventory(
        records,
        expected_groups=CALIBRATION_GROUPS,
        expected_input_calls=FIT_INPUT_CALLS,
    )


def _require_qualified_calibration(
    path: Path,
    *,
    source_manifest_path: Path,
    statistics_path: Path,
) -> dict[str, Any]:
    payload = _read_json(path)
    verify_payload_sha256(payload)
    if payload.get("schema") != CALIBRATION_SCHEMA:
        raise ValueError("unsupported projected calibration artifact schema")
    if payload.get("source_manifest_sha256") != sha256_file(source_manifest_path):
        raise ValueError("projected calibration source-manifest identity mismatch")
    if payload.get("statistics_file_sha256") != sha256_file(statistics_path):
        raise ValueError("projected calibration statistic-file identity mismatch")
    qualification = payload.get("qualification")
    checks = qualification.get("checks") if isinstance(qualification, Mapping) else None
    coefficients = (
        qualification.get("selected_coefficients")
        if isinstance(qualification, Mapping)
        else None
    )
    if (
        not isinstance(qualification, Mapping)
        or qualification.get("status") != "qualified"
        or not isinstance(checks, Mapping)
        or not checks
        or not all(value is True for value in checks.values())
        or qualification.get("selected_model") not in MODEL_NAMES[1:]
        or not isinstance(coefficients, list)
        or len(coefficients) != 2
        or not all(np.isfinite(float(value)) for value in coefficients)
    ):
        raise ValueError("projected calibration did not qualify; targets stay closed")
    return payload


def _verify_frozen_source_manifest(
    path: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    payload = _read_json(path)
    verify_payload_sha256(payload)
    if payload.get("schema") != SOURCE_MANIFEST_SCHEMA:
        raise ValueError("unsupported projected source-manifest schema")
    if payload.get("source_sha256") != _source_hashes():
        raise ValueError("projected evaluator source differs from calibration")
    readiness = _verify_readiness(args.readiness)
    if payload.get("readiness_payload_sha256") != readiness["payload_sha256"]:
        raise ValueError("projected readiness identity differs from calibration")
    if payload.get("base_readiness_file_sha256") != sha256_file(args.base_readiness):
        raise ValueError("base readiness identity differs from calibration")
    return payload


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    runtime, source_manifest = _build_runtime(args)
    try:
        collected = base._collect(
            runtime,
            case_ids=CALIBRATION_CASE_IDS[:1],
            input_calls=FIT_INPUT_CALLS[:1],
            coefficients=None,
            phase="projected_smoke",
        )
        records = projected_statistic_records(
            collected.snapshots,
            resolution=base.RESOLUTION_CONTRACT.native,
            projector=runtime.native_projector,
        )
        checks = {
            "one_snapshot": len(collected.snapshots) == 1,
            "three_logical_model_calls": collected.execution["logical_model_calls"]
            == 3,
            "one_statistic_record": len(records) == 1,
            "evaluation_targets_closed": True,
            "projection_closure": all(
                value <= 1.0e-10
                for key, value in asdict(records[0].projection_closure).items()
                if key != "maximum_excluded_parallel_abs"
            )
            and records[0].projection_closure.maximum_excluded_parallel_abs == 0.0,
        }
        payload = with_payload_sha256(
            {
                "schema": SMOKE_SCHEMA,
                "working_id": WORKING_ID,
                "status": "passed" if all(checks.values()) else "failed",
                "population_status": "adaptive_open_validation",
                "checks": checks,
                "case_ids": list(CALIBRATION_CASE_IDS[:1]),
                "input_calls": list(FIT_INPUT_CALLS[:1]),
                "source_manifest_payload_sha256": source_manifest["payload_sha256"],
                "execution": collected.execution,
                "closure_maxima": collected.maxima,
                "recurrence_executed": False,
            }
        )
    finally:
        base._close_runtime(runtime)
    atomic_write_json(args.output, payload)
    return payload


def run_calibration(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    runtime, source_manifest = _build_runtime(args)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=False)
    source_manifest_path = base._write_source_manifest(output_dir, source_manifest)
    try:
        collected = base._collect(
            runtime,
            case_ids=CALIBRATION_CASE_IDS,
            input_calls=FIT_INPUT_CALLS,
            coefficients=None,
            phase="projected_calibration",
        )
        records = projected_statistic_records(
            collected.snapshots,
            resolution=base.RESOLUTION_CONTRACT.native,
            projector=runtime.native_projector,
        )
        crossfit = projected_grouped_crossfit_from_records(
            records,
            projector=runtime.native_projector,
            expected_groups=CALIBRATION_GROUPS,
            expected_input_calls=FIT_INPUT_CALLS,
        )
        coefficients = tuple(crossfit["selected_coefficients"] or (0.0, 0.0))
        fit_cell = score_projected_cell(
            collected.snapshots,
            cell="calibration_fit",
            coefficients=coefficients,
            resolution=base.RESOLUTION_CONTRACT.native,
            projector=runtime.native_projector,
        )
        relations = error_relation_rows(
            collected.snapshots,
            cell="calibration_fit",
            resolution=base.RESOLUTION_CONTRACT.native,
            projector=runtime.native_projector,
        )
        closure = ProjectedCalibrationClosureEvidence(
            checkpoint_contract=True,
            reference_contract=len(collected.reference_rows)
            == len(CALIBRATION_CASE_IDS),
            common_source_inventory=len(collected.snapshots)
            == len(CALIBRATION_CASE_IDS) * len(FIT_INPUT_CALLS),
            prediction_inventory=collected.execution["logical_model_calls"]
            == 3 * len(collected.snapshots),
            exact_source_identity=True,
            maximum_pre_model_nesting_floor=float(
                collected.maxima["pre_model_nesting"]
            ),
            maximum_post_fp32_nesting_floor=float(
                collected.maxima["post_fp32_nesting"]
            ),
            maximum_sign_closure=float(collected.maxima["sign_closure"]),
            maximum_increment_integral_closure=float(
                collected.maxima["increment_integral_closure"]
            ),
            maximum_transfer_floor_closure=float(
                collected.maxima["transfer_floor_closure"]
            ),
            maximum_band_closure=max(
                float(fit_cell["maximum_closure"]["band"]),
                float(fit_cell["maximum_closure"]["subspace_reconstruction"]),
                float(fit_cell["maximum_closure"]["subspace_orthogonality"]),
            ),
            maximum_region_partition_error=int(
                collected.maxima["region_partition_error"]
            ),
            maximum_repeat_abs_difference=float(
                collected.execution["maximum_repeat_abs_difference"]
            ),
        )
        qualification = qualify_projected_calibration(crossfit, closure)
        statistics_path = output_dir / "projected_statistics.json"
        statistics_payload = _statistics_payload(
            records,
            source_manifest_path=source_manifest_path,
            source_manifest=source_manifest,
        )
        atomic_write_json(statistics_path, statistics_payload)
        write_csv(output_dir / "transfer_floors.csv", collected.floor_rows)
        write_csv(output_dir / "fit_cell_scores.csv", fit_cell["rows"])
        write_csv(output_dir / "error_relations.csv", relations)
        write_csv(output_dir / "reference_checks.csv", collected.reference_rows)
        artifacts = {
            path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in sorted(output_dir.iterdir())
            if path.is_file()
        }
        payload = with_payload_sha256(
            {
                "schema": CALIBRATION_SCHEMA,
                "working_id": WORKING_ID,
                "status": (
                    "complete_qualified"
                    if qualification["status"] == "qualified"
                    else "complete_not_qualified"
                ),
                "population_status": "adaptive_open_validation",
                "evaluation_targets_loaded": False,
                "source_manifest_sha256": sha256_file(source_manifest_path),
                "source_manifest_payload_sha256": source_manifest["payload_sha256"],
                "statistics_file_sha256": sha256_file(statistics_path),
                "statistics_payload_sha256": statistics_payload["payload_sha256"],
                "crossfit": crossfit,
                "qualification": qualification,
                "calibration_fit_cell": fit_cell,
                "execution": collected.execution,
                "transfer_floor_maxima": collected.maxima,
                "artifact_inventory_before_summary": artifacts,
                "claim_boundary": (
                    "Adaptive open-validation calibration evidence only. True error "
                    "is an offline label; no recurrent, sealed, bump, gradient, or "
                    "Richardson claim."
                ),
            }
        )
        atomic_write_json(output_dir / "calibration.json", payload)
        return payload, 0 if qualification["status"] == "qualified" else 3
    finally:
        base._close_runtime(runtime)


def _maximum_absolute(value: np.ndarray) -> float:
    array = np.asarray(value, dtype=np.float64)
    return 0.0 if array.size == 0 else float(np.max(np.abs(array)))


def _reconstruct_controls(
    runtime: base.RuntimeContext,
    snapshots: Sequence[base.DiagnosticSnapshot],
    *,
    case_ids: Sequence[str],
    input_calls: Sequence[int],
    coefficients: tuple[float, float],
) -> ReconstructedControls:
    """Build physical controls from stored increments with no new model call."""

    started = perf_counter()
    expected = {
        (case_id, int(input_call)) for case_id in case_ids for input_call in input_calls
    }
    by_key = {(row.case_id, int(row.input_call)): row for row in snapshots}
    if len(by_key) != len(snapshots) or set(by_key) != expected:
        raise ValueError("control reconstruction snapshot inventory mismatch")
    front_rows: list[dict[str, Any]] = []
    integral_rows: list[dict[str, Any]] = []
    proposal_rows: list[dict[str, Any]] = []
    maxima: dict[str, float | int] = {
        "target_reconstruction": 0.0,
        "excluded_correction": 0.0,
        "model_calls": 0,
    }
    native_geometry = runtime.geometry_by_resolution[base.RESOLUTION_CONTRACT.native]
    for case_id in case_ids:
        reference, _ = load_resolution_reference(
            runtime.args.family_root,
            runtime.args.multires_reference_root,
            runtime.store,
            runtime.manifest,
            case_id,
            training_resolution=base.RESOLUTION_CONTRACT.native,
        )
        reference_resolution = tuple(
            int(value) for value in reference["retained_resolution"]
        )
        if reference_resolution != base.RESOLUTION_CONTRACT.fine:
            raise ValueError(
                "projected controls require the retained 500x200 reference"
            )
        for input_call in input_calls:
            snapshot = by_key[(case_id, int(input_call))]
            input_frame = int(input_call) * 2
            output_frame = (int(input_call) + 1) * 2
            native_input = reference_at_resolution(
                reference["conservative_states"][input_frame],
                reference_resolution=reference_resolution,
                target_resolution=base.RESOLUTION_CONTRACT.native,
            )
            native_target = reference_at_resolution(
                reference["conservative_states"][output_frame],
                reference_resolution=reference_resolution,
                target_resolution=base.RESOLUTION_CONTRACT.native,
            )
            if native_input is None or native_target is None:
                raise ValueError("native control reference is unavailable")
            prepared = base.prepare_common_native_inputs(
                native_input,
                contract=base.RESOLUTION_CONTRACT,
            )
            raw_prediction = (
                prepared.model_inputs[base.RESOLUTION_CONTRACT.native]
                + snapshot.basis.native_increment
            )
            reconstructed_target = np.asarray(
                native_target, dtype=np.float64
            ) - np.asarray(raw_prediction, dtype=np.float64)
            maxima["target_reconstruction"] = max(
                float(maxima["target_reconstruction"]),
                _maximum_absolute(reconstructed_target - snapshot.target_correction),
            )
            corrected = projected_native_prediction(
                raw_prediction,
                snapshot.basis,
                runtime.native_projector,
                alpha=coefficients[0],
                beta=coefficients[1],
            )
            correction = np.asarray(corrected) - np.asarray(raw_prediction)
            maxima["excluded_correction"] = max(
                float(maxima["excluded_correction"]),
                _maximum_absolute(correction[~runtime.native_projector.interior_mask]),
            )
            raw_front = base._front_errors(raw_prediction, native_target, runtime)
            corrected_front = base._front_errors(corrected, native_target, runtime)
            for key in base.FRONT_CONTROL_KEYS:
                front_rows.append(
                    {
                        "case_id": case_id,
                        "input_call": int(input_call),
                        "key": key,
                        "zero_error": raw_front[key],
                        "corrected_error": corrected_front[key],
                    }
                )
            raw_integral = base._physical_integral(
                raw_prediction - native_target,
                native_geometry.node_measures,
            )
            corrected_integral = base._physical_integral(
                corrected - native_target,
                native_geometry.node_measures,
            )
            for component, name in enumerate(base.INTEGRAL_COMPONENT_NAMES):
                integral_rows.append(
                    {
                        "case_id": case_id,
                        "input_call": int(input_call),
                        "component": name,
                        "zero_error": float(raw_integral[component]),
                        "corrected_error": float(corrected_integral[component]),
                    }
                )
            raw_admissibility = conservative_admissibility_summary(
                raw_prediction,
                gamma=runtime.normalization.gamma,
            )
            corrected_admissibility = conservative_admissibility_summary(
                corrected,
                gamma=runtime.normalization.gamma,
            )
            proposal_rows.append(
                {
                    "case_id": case_id,
                    "input_call": int(input_call),
                    "raw_finite": bool(np.isfinite(raw_prediction).all()),
                    "raw_admissible": bool(raw_admissibility["admissible"]),
                    "finite": bool(np.isfinite(corrected).all()),
                    "admissible": bool(corrected_admissibility["admissible"]),
                    "minimum_density": corrected_admissibility["minimum_density"],
                    "minimum_pressure": corrected_admissibility["minimum_pressure"],
                    "minimum_internal_energy": corrected_admissibility[
                        "minimum_internal_energy"
                    ],
                }
            )
        del reference
    return ReconstructedControls(
        front_rows=front_rows,
        integral_rows=integral_rows,
        proposal_rows=proposal_rows,
        maxima=maxima,
        wall_seconds=perf_counter() - started,
    )


def run_evaluation(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    # Immutable firewall first: no checkpoint, shard, or reference loader before
    # all three calibration-bound artifacts have been verified.
    frozen_source = _verify_frozen_source_manifest(args.source_manifest, args)
    calibration = _require_qualified_calibration(
        args.calibration,
        source_manifest_path=args.source_manifest,
        statistics_path=args.calibration_statistics,
    )
    calibration_records = _read_calibration_statistics(
        args.calibration_statistics,
        expected_source_manifest_sha256=sha256_file(args.source_manifest),
        expected_payload_sha256=calibration["statistics_payload_sha256"],
    )
    coefficients = tuple(
        float(value) for value in calibration["qualification"]["selected_coefficients"]
    )
    runtime, live_source = _build_runtime(args)
    if live_source != frozen_source:
        base._close_runtime(runtime)
        raise ValueError("live external/source contract differs from calibration")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=False)
    try:
        time_only = base._collect(
            runtime,
            case_ids=CALIBRATION_CASE_IDS,
            input_calls=HELD_OUT_INPUT_CALLS,
            coefficients=None,
            phase="projected_time_only",
        )
        evaluation = base._collect(
            runtime,
            case_ids=EVALUATION_CASE_IDS,
            input_calls=ALL_INPUT_CALLS,
            coefficients=None,
            phase="projected_evaluation",
        )
        current_snapshots = [*time_only.snapshots, *evaluation.snapshots]
        current_records = projected_statistic_records(
            current_snapshots,
            resolution=base.RESOLUTION_CONTRACT.native,
            projector=runtime.native_projector,
        )
        all_open_crossfit = projected_grouped_crossfit_from_records(
            (*calibration_records, *current_records),
            projector=runtime.native_projector,
            expected_groups=all_open_strength_groups(),
            expected_input_calls=ALL_INPUT_CALLS,
        )
        all_open_stability = projected_coefficient_stability(
            all_open_crossfit,
            minimum_same_sign_folds=11,
            maximum_relative_iqr=0.5,
        )
        cell_payloads = {}
        relation_rows: list[dict[str, Any]] = []
        for cell in (
            "time_only",
            "case_only",
            "joint_held_out",
            "joint_late_1",
            "joint_late_2",
        ):
            selected = base._select_cell(current_snapshots, cell)
            cell_payloads[cell] = score_projected_cell(
                selected,
                cell=cell,
                coefficients=coefficients,
                resolution=base.RESOLUTION_CONTRACT.native,
                projector=runtime.native_projector,
            )
            relation_rows.extend(
                error_relation_rows(
                    selected,
                    cell=cell,
                    resolution=base.RESOLUTION_CONTRACT.native,
                    projector=runtime.native_projector,
                )
            )
        time_controls = _reconstruct_controls(
            runtime,
            time_only.snapshots,
            case_ids=CALIBRATION_CASE_IDS,
            input_calls=HELD_OUT_INPUT_CALLS,
            coefficients=coefficients,
        )
        evaluation_controls = _reconstruct_controls(
            runtime,
            evaluation.snapshots,
            case_ids=EVALUATION_CASE_IDS,
            input_calls=ALL_INPUT_CALLS,
            coefficients=coefficients,
        )
        front_rows = [*time_controls.front_rows, *evaluation_controls.front_rows]
        integral_rows = [
            *time_controls.integral_rows,
            *evaluation_controls.integral_rows,
        ]
        proposal_rows = [
            *time_controls.proposal_rows,
            *evaluation_controls.proposal_rows,
        ]
        controls: list[ControlRatio] = base._build_controls(
            cell_payloads["joint_held_out"],
            front_rows,
            integral_rows,
        )
        proposal_evidence = [
            ProposalEvidence(
                case_id=str(row["case_id"]),
                input_call=int(row["input_call"]),
                finite=bool(row["finite"]),
                admissible=bool(row["admissible"]),
            )
            for row in evaluation_controls.proposal_rows
            if int(row["input_call"]) in CELL_CALLS["joint_held_out"]
        ]
        gate = adaptive_projected_gate(
            cell_payloads=cell_payloads,
            controls=controls,
            proposals=proposal_evidence,
            all_open_crossfit=all_open_crossfit,
            calibration_qualification=calibration["qualification"],
            control_reconstruction_closure={
                "target_reconstruction": max(
                    float(time_controls.maxima["target_reconstruction"]),
                    float(evaluation_controls.maxima["target_reconstruction"]),
                ),
                "excluded_correction": max(
                    float(time_controls.maxima["excluded_correction"]),
                    float(evaluation_controls.maxima["excluded_correction"]),
                ),
                "model_calls": int(time_controls.maxima["model_calls"])
                + int(evaluation_controls.maxima["model_calls"]),
            },
        )
        score_rows = [
            row for payload in cell_payloads.values() for row in payload["rows"]
        ]
        oracle_rows = [
            row for payload in cell_payloads.values() for row in payload["oracle_rows"]
        ]
        floor_rows = [*time_only.floor_rows, *evaluation.floor_rows]
        reference_rows = [*time_only.reference_rows, *evaluation.reference_rows]
        write_csv(output_dir / "cell_scores.csv", score_rows)
        write_csv(output_dir / "error_relations.csv", relation_rows)
        write_csv(output_dir / "front_controls.csv", front_rows)
        write_csv(output_dir / "integral_controls.csv", integral_rows)
        write_csv(output_dir / "proposal_admissibility.csv", proposal_rows)
        write_csv(output_dir / "transfer_floors.csv", floor_rows)
        write_csv(output_dir / "reference_checks.csv", reference_rows)
        write_csv(
            output_dir / "adaptive_control_ratios.csv",
            [asdict(row) for row in controls],
        )
        artifacts = {
            path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in sorted(output_dir.iterdir())
            if path.is_file()
        }
        payload = with_payload_sha256(
            {
                "schema": EVALUATION_SCHEMA,
                "working_id": WORKING_ID,
                "status": (
                    "complete_continuation_supported"
                    if gate["continuation_request_supported"]
                    else "complete_stopped"
                ),
                "population_status": "adaptive_open_validation",
                "fresh_confirmation": False,
                "calibration_file_sha256": sha256_file(args.calibration),
                "calibration_statistics_file_sha256": sha256_file(
                    args.calibration_statistics
                ),
                "source_manifest_sha256": sha256_file(args.source_manifest),
                "selected_model": calibration["qualification"]["selected_model"],
                "frozen_calibration_coefficients": list(coefficients),
                "adaptive_gate": gate,
                "cell_payloads": cell_payloads,
                "oracle_rows": oracle_rows,
                "controls": [asdict(row) for row in controls],
                "all_open_lopo_crossfit": all_open_crossfit,
                "all_open_lopo_stability_diagnostic": all_open_stability,
                "all_open_lopo_coefficients_deployable": False,
                "execution": {
                    "time_only": time_only.execution,
                    "evaluation": evaluation.execution,
                    "control_reconstruction": {
                        "logical_model_calls": 0,
                        "time_only_wall_seconds": time_controls.wall_seconds,
                        "evaluation_wall_seconds": evaluation_controls.wall_seconds,
                    },
                },
                "closure_maxima": {
                    "time_only": time_only.maxima,
                    "evaluation": evaluation.maxima,
                    "time_control_reconstruction": time_controls.maxima,
                    "evaluation_control_reconstruction": evaluation_controls.maxima,
                    "cell_scoring": {
                        cell: {
                            "base": payload["maximum_closure"],
                            "projection": payload["projection_closure"],
                        }
                        for cell, payload in cell_payloads.items()
                    },
                },
                "artifact_inventory_before_summary": artifacts,
                "recurrence_executed": False,
                "recurrence_authorized": False,
                "sealed_population_authorized": False,
                "claim_boundary": (
                    "Adaptive open-validation teacher-forced fixed-checkpoint evidence "
                    "only. True error was used only for offline labels and diagnostics. "
                    "No fresh confirmation, recurrent benefit, resolution invariance, "
                    "conservation-by-construction, cross-family, bump, gradient-fix, or "
                    "Richardson claim."
                ),
            }
        )
        atomic_write_json(output_dir / "evaluation.json", payload)
        return payload, 0 if gate["continuation_request_supported"] else 4
    finally:
        base._close_runtime(runtime)


def _add_external_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--readiness", type=Path, required=True)
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

    readiness = commands.add_parser("readiness")
    readiness.add_argument("--output", type=Path, required=True)
    readiness.add_argument("--base-readiness", type=Path, required=True)
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
    evaluation.add_argument("--calibration-statistics", type=Path, required=True)
    evaluation.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "readiness":
        payload = create_readiness(args)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if payload["status"] == "passed" else 2
    if args.command == "preflight":
        payload = run_preflight(args)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.repeat_forward < 1:
        raise ValueError("repeat-forward must be positive")
    if args.command == "smoke":
        payload = run_smoke(args)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if payload["status"] == "passed" else 2
    if args.command == "calibrate":
        payload, exit_code = run_calibration(args)
    elif args.command == "evaluate":
        payload, exit_code = run_evaluation(args)
    else:  # pragma: no cover
        raise AssertionError(f"unsupported command: {args.command}")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
