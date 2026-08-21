#!/usr/bin/env python3
"""Evaluate the frozen A28 two-call affine modal map on open E12/E14 truth."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.time_dependent_no import (
    evaluate_pcno_cross_resolution_teacher_forced as base,
)
from scripts.time_dependent_no import (
    evaluate_pcno_fine_discrepancy_correction as fine_eval,
)
from utility.time_dependent_no.pcno_artifacts import (
    atomic_write_json_with_paths as atomic_write_json,
)
from utility.time_dependent_no.pcno_artifacts import (
    digest_array,
    git_state,
    runtime_environment,
    sha256_file,
    sha256_files,
)
from utility.time_dependent_no.pcno_artifacts import (
    write_csv_with_paths as write_csv,
)
from utility.time_dependent_no.pcno_cross_resolution_correction import (
    DiagnosticSnapshot,
    ResolutionContract,
)
from utility.time_dependent_no.pcno_cross_resolution_teacher_forced import (
    FRONT_CONTROL_KEYS,
    INTEGRAL_COMPONENT_NAMES,
    verify_payload_sha256,
    with_payload_sha256,
)
from utility.time_dependent_no.pcno_fine_discrepancy_correction import (
    score_precomputed_correction_inventory,
)
from utility.time_dependent_no.pcno_residual_structure import shock_vortex_regions
from utility.time_dependent_no.pcno_resolution_transfer import (
    conservative_admissibility_summary,
    load_resolution_reference,
    predict_resolution_sample,
)
from utility.time_dependent_no.pcno_sparse_modal_correction import (
    AFFINE_MODAL_POLICY,
    FROZEN_ACTIVE_CELLS,
    synchronized_affine_modal_step,
)
from utility.time_dependent_no.shock_vortex_family import family_case_provenance

WORKING_ID = "W26-L5-P6-RFB19-A29-SP19-AFFINE-TRANSFER-E12-E14"
PREFLIGHT_SCHEMA = "pcno_sp19_affine_transfer_preflight_v1"
RESULT_SCHEMA = "pcno_sp19_affine_transfer_evaluation_v1"
SOURCE_MANIFEST_SCHEMA = "pcno_sp19_affine_transfer_source_manifest_v1"
NATIVE_RESOLUTION = (250, 100)
FINE_RESOLUTION = (500, 200)
CONTRACT = ResolutionContract(
    coarse=(125, 50), native=NATIVE_RESOLUTION, fine=FINE_RESOLUTION
)
INPUT_CALLS = tuple(range(30))
BANDS = ((0, 7), (8, 14), (15, 21), (22, 29))
GROUP_CASES = {
    "e12": tuple(f"sv_e12_y{index:02d}" for index in range(1, 8)),
    "e14": tuple(f"sv_e14_y{index:02d}" for index in range(1, 8)),
}
CASE_IDS = (*GROUP_CASES["e12"], *GROUP_CASES["e14"])
GROUP_IDS = {"e12": "strength_ood_e12", "e14": "strength_ood_e14"}
SOURCE_PATHS = (
    "docs/time_dependent_no/W26_L5_RESPONSE_FILTERED_BLOCK_PREREGISTRATION.md",
    "utility/time_dependent_no/pcno_sparse_modal_correction.py",
    "utility/time_dependent_no/pcno_fine_discrepancy_correction.py",
    "scripts/time_dependent_no/evaluate_pcno_modal_affine_transfer.py",
    "scripts/time_dependent_no/evaluate_pcno_sparse_modal_correction.py",
    "tests/time_dependent_no/test_pcno_sparse_modal_correction.py",
    "tests/time_dependent_no/test_pcno_fine_discrepancy_correction.py",
)
EXPECTED_LINEAGE = {
    "a28": {
        "file": "dfe21dd7cb97e7b7fde4a97707cca2acf811f618d43e5100bbb439c9cfcb9d2d",
        "payload": "4ddf57745878058171bef82b2055373a57e1e4a5612e3430b05b1cc225a36c62",
        "schema": "pcno_sp19_affine_phase_map_v1",
        "status": "completed_candidate_selected",
    },
    "a22": {
        "file": "eb964b01dbb07da16a4ca95fe96f5497f1a87c2a68b137e7ecc2b75def521fc7",
        "payload": "b3f798ece9662ae54678fe59bef99b01ecaf6985b6e350ea082a865447f51125",
        "schema": "pcno_response_filtered_block_buffered_relaxed_tether_replay_v1",
        "status": "qualified_calibration_replay",
    },
    "a23": {
        "file": "aecc8a23c9b7c5e198b2948134ec2f901642684df9c66a04954308f31c9e195c",
        "payload": "8180306144be92f4cc059ba06dddcdf92fa674437fddee22da69b61ff46413ad",
        "schema": "pcno_response_filtered_block_buffered_relaxed_tether_e14_rollout_v1",
        "status": "stopped_retrospective_transfer",
    },
    "a22_reference_checks": {
        "file": "cf4b5292e182b5e1fb9eedde2773cfe73cea5f64fd6d204a9e8022d9413c1a84"
    },
    "a23_reference_checks": {
        "file": "3dca6a2d65502e5a0ac0321a38cf94a35b07ab2c16d1d196f1b1551cd0324af5"
    },
}
SHARD_REFERENCE_SCHEMA = "pcno_shard_native_reference_v1"
SHARD_AUDIT_SHA256 = "5917345022d25c9e81a2a18f13f05d0c3cfe55b85945fb0d536e5f4b82174af3"
SHARD_AUDIT_PAYLOAD_SHA256 = (
    "02986c4f4596726448aab62aab24be1ca73e2aa1df96a58c3d75baf53b69767b"
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON payload must be an object: {path.name}")
    return value


def _source_hashes() -> dict[str, str]:
    return sha256_files(SOURCE_PATHS, root=ROOT)


def _base_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        readiness=args.base_readiness,
        family_root=args.family_root,
        multires_reference_root=args.multires_reference_root,
        data_dir=args.data_dir,
        checkpoint=args.checkpoint,
        normalization_file=args.normalization_file,
        split_file=args.split_file,
        d063_run_contract=args.d063_run_contract,
        d063_summary=args.d063_summary,
        device=getattr(args, "device", "cuda"),
        allow_cpu=getattr(args, "allow_cpu", False),
    )


def _verify_result(path: Path, key: str) -> dict[str, Any]:
    expected = EXPECTED_LINEAGE[key]
    if sha256_file(path) != expected["file"]:
        raise ValueError(f"{key} file digest differs from the frozen contract")
    payload = _read_json(path)
    verify_payload_sha256(payload)
    if (
        payload.get("payload_sha256") != expected["payload"]
        or payload.get("schema") != expected["schema"]
        or payload.get("status") != expected["status"]
    ):
        raise ValueError(f"{key} payload differs from the frozen contract")
    return payload


def _verify_reference_csv(path: Path, key: str) -> list[dict[str, str]]:
    if sha256_file(path) != EXPECTED_LINEAGE[key]["file"]:
        raise ValueError(f"{key} digest differs from the frozen contract")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected_prefix = "sv_e12" if key.startswith("a22") else "sv_e14"
    expected_ids = {f"{expected_prefix}_y{index:02d}" for index in range(9)}
    if len(rows) != 9 or {row.get("case_id") for row in rows} != expected_ids:
        raise ValueError(f"{key} case inventory differs from the frozen contract")
    return rows


def _verify_affine_map(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    payload = _verify_result(path, "a28")
    if payload.get("decision", {}).get("selected") != "cost_two_call":
        raise ValueError("A28 did not select the registered two-call map")
    fit = payload.get("candidates", {}).get("cost_two_call", {}).get("full_fit", {})
    start = np.asarray(fit.get("start_coefficients"), dtype=np.float64)
    end = np.asarray(fit.get("end_coefficients"), dtype=np.float64)
    intercept = np.asarray(fit.get("intercept"), dtype=np.float64)
    active = tuple(tuple(int(value) for value in cell) for cell in fit.get("active_cells", ()))
    if (
        fit.get("family") != "fine_full_affine"
        or int(fit.get("last_input_call", -1)) != 29
        or active != FROZEN_ACTIVE_CELLS
        or start.shape != (19, 19)
        or end.shape != (19, 19)
        or intercept.shape != (19,)
        or not np.array_equal(intercept, np.zeros(19, dtype=np.float64))
        or not np.isfinite(start).all()
        or not np.isfinite(end).all()
    ):
        raise ValueError("A28 coefficient contract differs")
    return start, end, payload


def _lineage(args: argparse.Namespace) -> dict[str, Any]:
    _, _, a28 = _verify_affine_map(args.a28_map)
    a22 = _verify_result(args.a22_result, "a22")
    a23 = _verify_result(args.a23_result, "a23")
    _verify_reference_csv(args.a22_reference_checks, "a22_reference_checks")
    _verify_reference_csv(args.a23_reference_checks, "a23_reference_checks")
    return {
        "a28": {
            "file_sha256": sha256_file(args.a28_map),
            "payload_sha256": a28["payload_sha256"],
        },
        "a22": {
            "file_sha256": sha256_file(args.a22_result),
            "payload_sha256": a22["payload_sha256"],
        },
        "a23": {
            "file_sha256": sha256_file(args.a23_result),
            "payload_sha256": a23["payload_sha256"],
        },
        "a22_reference_checks_sha256": sha256_file(args.a22_reference_checks),
        "a23_reference_checks_sha256": sha256_file(args.a23_reference_checks),
    }


def _build_source_manifest(
    args: argparse.Namespace,
    *,
    base_source_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    return with_payload_sha256(
        {
            "schema": SOURCE_MANIFEST_SCHEMA,
            "working_id": WORKING_ID,
            "source_sha256": _source_hashes(),
            "lineage": _lineage(args),
            "base_runtime_source_manifest": dict(base_source_manifest),
            "population": {
                "case_ids": list(CASE_IDS),
                "groups": {key: list(value) for key, value in GROUP_CASES.items()},
                "input_calls": list(INPUT_CALLS),
                "bands": [list(value) for value in BANDS],
                "status": "already_open_e12_e14_interiors",
            },
            "inference_contract": {
                "logical_calls_per_state": 2,
                "call_order": ["native", "fine"],
                "common_source": "one_native_physical_truth_state",
                "fine_input": "piecewise_constant_prolongation_then_fp32",
                "fine_output_map": "physical_volume_increment_restriction",
                "fine_truth_loaded": False,
                "phase": "input_call_over_29",
                "coefficient_refit": False,
                "maximum_correction_to_native_increment": 0.05,
                "recurrence": False,
            },
        }
    )


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    _, checkpoint, manifest, store, base_source = base._open_contract(_base_args(args))
    try:
        provenance = {
            case_id: family_case_provenance(manifest, case_id) for case_id in CASE_IDS
        }
        groups_ok = all(
            provenance[case_id].get("split") == "test"
            and provenance[case_id].get("split_group_id") == GROUP_IDS[group]
            for group, cases in GROUP_CASES.items()
            for case_id in cases
        )
        source = _build_source_manifest(args, base_source_manifest=base_source)
        checks = {
            "source_inventory_exact": set(source["source_sha256"]) == set(SOURCE_PATHS),
            "lineage_exact": set(source["lineage"])
            == {
                "a28",
                "a22",
                "a23",
                "a22_reference_checks_sha256",
                "a23_reference_checks_sha256",
            },
            "case_inventory_present": all(case_id in store.keys for case_id in CASE_IDS),
            "case_inventory_exact": set(provenance) == set(CASE_IDS),
            "opened_groups_exact": groups_ok,
            "checkpoint_stride_exact": base.checkpoint_step_stride(checkpoint) == 2,
        }
        payload = with_payload_sha256(
            {
                "schema": PREFLIGHT_SCHEMA,
                "working_id": WORKING_ID,
                "status": "passed" if all(checks.values()) else "failed",
                "checks": checks,
                "source_manifest": source,
                "checkpoint_model_built": False,
                "reference_arrays_loaded": False,
                "fine_reference_loaded": False,
                "model_calls": 0,
                "recurrence_executed": False,
                "population": {case_id: provenance[case_id] for case_id in CASE_IDS},
            }
        )
    finally:
        store.close()
    if payload["status"] != "passed":
        raise ValueError("A29 preflight checks did not all pass")
    atomic_write_json(args.output, payload)
    return payload


def _verify_preflight(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    payload = _read_json(path)
    verify_payload_sha256(payload)
    source = payload.get("source_manifest", {})
    if (
        payload.get("schema") != PREFLIGHT_SCHEMA
        or payload.get("working_id") != WORKING_ID
        or payload.get("status") != "passed"
        or payload.get("checkpoint_model_built") is not False
        or payload.get("reference_arrays_loaded") is not False
        or payload.get("fine_reference_loaded") is not False
        or payload.get("model_calls") != 0
        or payload.get("recurrence_executed") is not False
        or not payload.get("checks")
        or not all(payload["checks"].values())
        or source.get("source_sha256") != _source_hashes()
        or source.get("lineage") != _lineage(args)
    ):
        raise ValueError("A29 preflight differs from the frozen contract")
    return payload


def _normalize_csv_value(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return json.dumps(list(value), separators=(",", ":"))
    if value is None:
        return ""
    return str(value)


def _require_reference_row(
    check: Mapping[str, Any], expected: Mapping[str, str]
) -> None:
    actual = {key: _normalize_csv_value(check.get(key)) for key in expected}
    if actual != dict(expected):
        differences = {
            key: {"expected": expected[key], "actual": actual[key]}
            for key in expected
            if actual[key] != expected[key]
        }
        raise ValueError(f"reference check differs: {differences}")


def _load_shard_native_reference(
    runtime: Any, case_id: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    entry = runtime.store.entry(case_id)
    expected_arrays = entry.get("array_sha256")
    required = ("states_conservative", "physical_times", "nodes", "node_measures")
    if not isinstance(expected_arrays, Mapping) or any(
        name not in expected_arrays for name in required
    ):
        raise ValueError("shard-native reference lacks required array digests")
    arrays = {}
    for name in required:
        path = runtime.store.root / str(entry["folder"]) / f"{name}.npy"
        if sha256_file(path) != expected_arrays[name]:
            raise ValueError(f"shard-native reference array mismatch: {name}")
        arrays[name] = np.asarray(runtime.store.array(case_id, name))
    states = arrays["states_conservative"]
    times = arrays["physical_times"]
    nodes = arrays["nodes"]
    measures = arrays["node_measures"]
    expected_nodes = NATIVE_RESOLUTION[0] * NATIVE_RESOLUTION[1]
    if states.dtype != np.float32 or states.shape != (61, expected_nodes, 4):
        raise ValueError("shard-native state contract differs")
    if (
        times.shape != (61,)
        or not np.all(np.diff(times) > 0.0)
        or not np.allclose(
            times,
            np.arange(61, dtype=np.float64) * 0.01,
            rtol=0.0,
            atol=1.0e-15,
        )
    ):
        raise ValueError("shard-native time contract differs")
    if digest_array(states) != entry.get("state_digest"):
        raise ValueError("shard-native state digest mismatch")
    geometry = runtime.geometry_by_resolution[NATIVE_RESOLUTION]
    if not np.array_equal(nodes, np.asarray(geometry.nodes, dtype=nodes.dtype)) or not np.array_equal(
        measures.reshape(-1),
        np.asarray(geometry.node_measures).reshape(-1).astype(measures.dtype, copy=False),
    ):
        raise ValueError("shard-native and runtime geometry differ")
    return {
        "conservative_states": np.asarray(states, dtype=np.float64),
        "physical_times": np.asarray(times, dtype=np.float64),
        "retained_resolution": NATIVE_RESOLUTION,
    }, {
        "case_id": case_id,
        "reference_schema": SHARD_REFERENCE_SCHEMA,
        "reference_source": "checkpoint_bound_shard_states_conservative",
        "source_reference_sha256": entry.get("source_reference_sha256"),
        "shard_manifest_sha256": runtime.store.manifest_digest,
        "shard_states_array_sha256": expected_arrays["states_conservative"],
        "shard_state_digest": entry.get("state_digest"),
        "retained_resolution": "250x100",
        "state_dtype": str(states.dtype),
        "state_shape": list(states.shape),
        "restriction_crosscheck_max_abs": 0.0,
        "serialization_floor": "exact_original_reference_after_float32_cast",
        "shard_native_reference_audit_sha256": SHARD_AUDIT_SHA256,
        "shard_native_reference_audit_payload_sha256": SHARD_AUDIT_PAYLOAD_SHA256,
    }


def _physical_integral(value: np.ndarray, volumes: np.ndarray) -> np.ndarray:
    field = np.asarray(value, dtype=np.float64)
    mass = np.asarray(volumes, dtype=np.float64).reshape(-1)
    if field.shape != (mass.size, 4):
        raise ValueError("field and physical volumes do not align")
    return np.einsum("n,nc->c", mass, field, optimize=True)


def _population_specs() -> list[tuple[str, tuple[str, ...], tuple[int, ...]]]:
    specs = [("overall", CASE_IDS, INPUT_CALLS)]
    specs.extend((f"group_{group}", cases, INPUT_CALLS) for group, cases in GROUP_CASES.items())
    for start, end in BANDS:
        calls = tuple(range(start, end + 1))
        specs.append((f"band_{start}_{end}", CASE_IDS, calls))
        specs.extend(
            (f"group_{group}_band_{start}_{end}", cases, calls)
            for group, cases in GROUP_CASES.items()
        )
    return specs


def _score_inventory(
    snapshots: Sequence[DiagnosticSnapshot],
    corrections: Sequence[np.ndarray],
    *,
    projector: Any,
) -> dict[str, dict[str, Any]]:
    payloads = {}
    aligned = list(zip(snapshots, corrections, strict=True))
    for label, cases, calls in _population_specs():
        case_set, call_set = set(cases), set(calls)
        selected = [
            (snapshot, correction)
            for snapshot, correction in aligned
            if snapshot.case_id in case_set and snapshot.input_call in call_set
        ]
        payloads[label] = score_precomputed_correction_inventory(
            [row[0] for row in selected],
            [row[1] for row in selected],
            label=label,
            expected_case_ids=cases,
            expected_input_calls=calls,
            resolution=NATIVE_RESOLUTION,
            projector=projector,
        )
    return payloads


def _skill_row(
    payload: Mapping[str, Any], view: str, *, case_id: str | None = None
) -> Mapping[str, Any]:
    scope = "population" if case_id is None else "case"
    rows = [
        row
        for row in payload.get("rows", ())
        if row.get("view") == view
        and row.get("scope") == scope
        and row.get("case_id") == case_id
    ]
    if len(rows) != 1:
        raise ValueError(f"score row is incomplete for {view}/{scope}/{case_id}")
    return rows[0]


def _positive_skill(row: Mapping[str, Any]) -> bool:
    value = row.get("skill_vs_zero")
    return bool(
        row.get("skill_status") == "ok"
        and value is not None
        and math.isfinite(float(value))
        and float(value) > 0.0
    )


def _prospective_gate(
    *,
    score_payloads: Mapping[str, Mapping[str, Any]],
    controls: Sequence[Mapping[str, Any]],
    proposal_rows: Sequence[Mapping[str, Any]],
    audit_rows: Sequence[Mapping[str, Any]],
    nesting_rows: Sequence[Mapping[str, Any]],
    execution: Mapping[str, Any],
    structural_checks: Mapping[str, bool],
) -> dict[str, Any]:
    required_views = ("full", "rank8_parallel")
    score_checks = {}
    for label, payload in score_payloads.items():
        score_checks[f"{label}_population"] = all(
            _positive_skill(_skill_row(payload, view)) for view in required_views
        )
    for case_id in CASE_IDS:
        score_checks[f"case_{case_id}"] = all(
            _positive_skill(_skill_row(score_payloads["overall"], view, case_id=case_id))
            for view in required_views
        )
    audit_ok = bool(audit_rows) and all(
        row.get("status") == "ok"
        and row.get("correction_to_native_increment") is not None
        and float(row["correction_to_native_increment"]) <= 0.05 + 1.0e-12
        and float(row["maximum_scaled_component_mean_abs"]) <= 1.0e-12
        and float(row["maximum_excluded_abs"]) <= 1.0e-12
        and float(row["maximum_inactive_coordinate_abs"]) <= 1.0e-12
        and float(row["maximum_modal_reconstruction_abs"]) <= 1.0e-12
        for row in audit_rows
    )
    nesting_ok = bool(nesting_rows) and all(
        float(row["pre_model_coarse_from_native_max_abs"]) <= 1.0e-12
        and float(row["pre_model_fine_to_native_max_abs"]) <= 1.0e-12
        and all(math.isfinite(float(value)) for key, value in row.items() if "max_abs" in key)
        for row in nesting_rows
    )
    checks = {
        **dict(structural_checks),
        **score_checks,
        "controls_no_harm": bool(controls) and all(fine_eval._control_passed(row) for row in controls),
        "proposals_finite_admissible": bool(proposal_rows)
        and all(row["finite"] and row["admissible"] for row in proposal_rows),
        "correction_closure": audit_ok,
        "common_source_nesting": nesting_ok,
        "deterministic_repeat": float(execution["maximum_repeat_abs_difference"]) <= 1.0e-6,
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "failed_checks": sorted(key for key, value in checks.items() if not value),
        "strict_positive_skill_equality_fails": True,
        "control_ratio_limit": 1.05,
        "maximum_correction_ratio": 0.05,
    }


def synthetic_summary() -> dict[str, Any]:
    fake = {
        "skill_status": "ok",
        "skill_vs_zero": 1.0e-6,
    }
    zero = {**fake, "skill_vs_zero": 0.0}
    checks = {
        "case_inventory_exact": len(CASE_IDS) == 14 and len(set(CASE_IDS)) == 14,
        "call_inventory_exact": INPUT_CALLS == tuple(range(30)),
        "population_cells_exact": len(_population_specs()) == 15,
        "positive_skill_passes": _positive_skill(fake),
        "zero_skill_fails": not _positive_skill(zero),
        "source_inventory_exact": len(SOURCE_PATHS) == 7,
        "no_model_or_reference_loader_called": True,
    }
    return with_payload_sha256(
        {
            "schema": "pcno_sp19_affine_transfer_synthetic_v1",
            "working_id": WORKING_ID,
            "status": "passed" if all(checks.values()) else "failed",
            "checks": checks,
            "model_calls": 0,
            "reference_arrays_loaded": False,
        }
    )


def run_evaluation(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    preflight = _verify_preflight(args.preflight, args)
    start_coefficients, end_coefficients, _ = _verify_affine_map(args.a28_map)
    expected_reference = {
        "e12": {
            row["case_id"]: row
            for row in _verify_reference_csv(
                args.a22_reference_checks, "a22_reference_checks"
            )
        },
        "e14": {
            row["case_id"]: row
            for row in _verify_reference_csv(
                args.a23_reference_checks, "a23_reference_checks"
            )
        },
    }
    torch.use_deterministic_algorithms(True)
    runtime = base._build_runtime(_base_args(args))
    started = perf_counter()
    try:
        if runtime.source_manifest != preflight["source_manifest"]["base_runtime_source_manifest"]:
            raise ValueError("base runtime source changed after A29 preflight")
        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=False)
        native_geometry = runtime.geometry_by_resolution[NATIVE_RESOLUTION]
        volumes = np.asarray(native_geometry.node_measures, dtype=np.float64)
        scale = np.asarray(runtime.normalization.residual_scale, dtype=np.float64)
        snapshots: list[DiagnosticSnapshot] = []
        corrections: list[np.ndarray] = []
        audit_rows: list[dict[str, Any]] = []
        nesting_rows: list[dict[str, Any]] = []
        front_rows: list[dict[str, Any]] = []
        integral_rows: list[dict[str, Any]] = []
        proposal_rows: list[dict[str, Any]] = []
        reference_rows: list[dict[str, Any]] = []
        execution = {
            "logical_model_calls": 0,
            "actual_forward_passes_including_repeats": 0,
            "total_forward_seconds": 0.0,
            "native_forward_seconds": 0.0,
            "fine_forward_seconds": 0.0,
            "maximum_repeat_abs_difference": 0.0,
            "maximum_peak_gpu_memory_bytes": 0,
            "call_order_violations": 0,
        }

        def account(resolution: tuple[int, int], timing: Mapping[str, Any]) -> None:
            forwards = [float(value) for value in timing["forward_seconds"]]
            execution["logical_model_calls"] += 1
            execution["actual_forward_passes_including_repeats"] += len(forwards)
            execution["total_forward_seconds"] += sum(forwards)
            key = "native_forward_seconds" if resolution == NATIVE_RESOLUTION else "fine_forward_seconds"
            execution[key] += sum(forwards)
            execution["maximum_repeat_abs_difference"] = max(
                execution["maximum_repeat_abs_difference"], float(timing["repeat_max_abs"])
            )
            peak = timing.get("peak_gpu_memory_bytes")
            if peak is not None:
                execution["maximum_peak_gpu_memory_bytes"] = max(
                    execution["maximum_peak_gpu_memory_bytes"], int(peak)
                )

        for case_index, case_id in enumerate(CASE_IDS, start=1):
            group = "e12" if case_id.startswith("sv_e12") else "e14"
            print(f"A29 case {case_index}/{len(CASE_IDS)}: {case_id}", flush=True)
            if group == "e12":
                reference, reference_check = load_resolution_reference(
                    runtime.args.family_root,
                    None,
                    runtime.store,
                    runtime.manifest,
                    case_id,
                    training_resolution=NATIVE_RESOLUTION,
                )
            else:
                reference, reference_check = _load_shard_native_reference(runtime, case_id)
            _require_reference_row(reference_check, expected_reference[group][case_id])
            reference_rows.append(reference_check)
            states = np.asarray(reference["conservative_states"], dtype=np.float64)
            if states.shape != (61, NATIVE_RESOLUTION[0] * NATIVE_RESOLUTION[1], 4):
                raise ValueError("native teacher-forced reference shape differs")
            for input_call in INPUT_CALLS:
                native_state = states[2 * input_call]
                native_target = states[2 * (input_call + 1)]
                call_order: list[tuple[int, int]] = []

                def predictor(
                    resolution: tuple[int, int],
                    state: np.ndarray,
                    *,
                    _call_order: list[tuple[int, int]] = call_order,
                    _input_call: int = input_call,
                ) -> np.ndarray:
                    _call_order.append(resolution)
                    repeats = args.repeat_forward if _input_call == 0 else 1
                    prediction, timing = predict_resolution_sample(
                        runtime.model,
                        runtime.sample_by_resolution[resolution],
                        state,
                        device=runtime.device,
                        amp="none",
                        repeats=repeats,
                    )
                    account(resolution, timing)
                    return prediction

                step = synchronized_affine_modal_step(
                    native_state,
                    input_call=input_call,
                    contract=CONTRACT,
                    projector=runtime.native_projector,
                    predictor=predictor,
                    start_coefficients=start_coefficients,
                    end_coefficients=end_coefficients,
                    volumes=volumes,
                    component_scale=scale,
                )
                if call_order != [NATIVE_RESOLUTION, FINE_RESOLUTION]:
                    execution["call_order_violations"] += 1
                raw_prediction = np.asarray(step.predictions[NATIVE_RESOLUTION], dtype=np.float64)
                corrected_prediction = raw_prediction + step.correction
                target_correction = native_target - raw_prediction
                masks, _, _ = shock_vortex_regions(
                    native_target,
                    native_geometry.nodes,
                    resolution=NATIVE_RESOLUTION,
                    gamma=runtime.normalization.gamma,
                )
                snapshots.append(
                    DiagnosticSnapshot(
                        case_id=case_id,
                        group_id=group,
                        input_call=input_call,
                        basis=step.basis,
                        target_correction=target_correction,
                        volumes=volumes,
                        component_scale=scale,
                        masks=masks,
                    )
                )
                corrections.append(np.asarray(step.correction, dtype=np.float64))
                audit_rows.append(
                    {"case_id": case_id, "group_id": group, **asdict(step.audit)}
                )
                nesting_rows.append(
                    {
                        "case_id": case_id,
                        "group_id": group,
                        "input_call": input_call,
                        **step.prepared_inputs.nesting_floors,
                    }
                )
                raw_front = base._front_errors(raw_prediction, native_target, runtime)
                corrected_front = base._front_errors(
                    corrected_prediction, native_target, runtime
                )
                for key in FRONT_CONTROL_KEYS:
                    front_rows.append(
                        {
                            "policy": AFFINE_MODAL_POLICY,
                            "case_id": case_id,
                            "group_id": group,
                            "input_call": input_call,
                            "key": key,
                            "zero_error": raw_front[key],
                            "corrected_error": corrected_front[key],
                        }
                    )
                raw_integral = _physical_integral(
                    raw_prediction - native_target, volumes
                )
                corrected_integral = _physical_integral(
                    corrected_prediction - native_target, volumes
                )
                for component, name in enumerate(INTEGRAL_COMPONENT_NAMES):
                    integral_rows.append(
                        {
                            "policy": AFFINE_MODAL_POLICY,
                            "case_id": case_id,
                            "group_id": group,
                            "input_call": input_call,
                            "component": name,
                            "zero_error": float(raw_integral[component]),
                            "corrected_error": float(corrected_integral[component]),
                        }
                    )
                admissibility = conservative_admissibility_summary(
                    corrected_prediction, gamma=runtime.normalization.gamma
                )
                proposal_rows.append(
                    {
                        "case_id": case_id,
                        "group_id": group,
                        "input_call": input_call,
                        **admissibility,
                    }
                )
            del reference, states
            if runtime.device.type == "cuda":
                torch.cuda.empty_cache()

        score_payloads = _score_inventory(
            snapshots, corrections, projector=runtime.native_projector
        )
        controls = fine_eval._build_teacher_controls(
            policy=AFFINE_MODAL_POLICY,
            score_payload=score_payloads["overall"],
            controls={"front": front_rows, "integral": integral_rows},
            case_ids=CASE_IDS,
            input_calls=INPUT_CALLS,
        )
        execution["wall_seconds"] = perf_counter() - started
        execution["snapshot_count"] = len(snapshots)
        execution["logical_raw_native_calls"] = len(CASE_IDS) * len(INPUT_CALLS)
        execution["logical_corrected_two_call_calls"] = 2 * len(CASE_IDS) * len(INPUT_CALLS)
        execution["logical_cost_ratio_vs_raw_native"] = 2.0
        execution["observed_forward_time_ratio_vs_native"] = (
            execution["total_forward_seconds"] / execution["native_forward_seconds"]
        )
        execution["environment"] = runtime_environment(runtime.device)
        structural_checks = {
            "preflight_source_exact": preflight["source_manifest"]["source_sha256"]
            == _source_hashes(),
            "rectangular_inventory_exact": len(snapshots)
            == len(CASE_IDS) * len(INPUT_CALLS),
            "logical_call_inventory_exact": execution["logical_model_calls"]
            == 2 * len(snapshots),
            "native_then_fine_call_order": execution["call_order_violations"] == 0,
            "reference_inventory_exact": len(reference_rows) == len(CASE_IDS),
            "fine_truth_not_loaded": True,
            "coefficient_refit_not_run": True,
            "recurrence_not_run": True,
        }
        gate = _prospective_gate(
            score_payloads=score_payloads,
            controls=controls,
            proposal_rows=proposal_rows,
            audit_rows=audit_rows,
            nesting_rows=nesting_rows,
            execution=execution,
            structural_checks=structural_checks,
        )

        score_rows = [
            {"population_cell": label, **row}
            for label, payload in score_payloads.items()
            for row in payload["rows"]
        ]
        write_csv(output_dir / "score_rows.csv", score_rows)
        write_csv(output_dir / "control_rows.csv", controls)
        write_csv(output_dir / "correction_audits.csv", audit_rows)
        write_csv(output_dir / "nesting_floors.csv", nesting_rows)
        write_csv(output_dir / "front_controls.csv", front_rows)
        write_csv(output_dir / "integral_controls.csv", integral_rows)
        write_csv(output_dir / "proposal_rows.csv", proposal_rows)
        write_csv(output_dir / "reference_checks.csv", reference_rows)
        atomic_write_json(output_dir / "source_manifest.json", preflight["source_manifest"])
        artifacts = {
            path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in sorted(output_dir.iterdir())
            if path.is_file()
        }
        result = with_payload_sha256(
            {
                "schema": RESULT_SCHEMA,
                "working_id": WORKING_ID,
                "status": (
                    "qualified_teacher_forced_transfer"
                    if gate["status"] == "passed"
                    else "stopped_teacher_forced_transfer"
                ),
                "preflight_file_sha256": sha256_file(args.preflight),
                "preflight_payload_sha256": preflight["payload_sha256"],
                "frozen_a28_map_sha256": sha256_file(args.a28_map),
                "prospective_gate": gate,
                "score_payloads": score_payloads,
                "execution": execution,
                "artifact_inventory_before_summary": artifacts,
                "state_arrays_saved": False,
                "animations_generated": False,
                "fine_reference_loaded": False,
                "recurrence_executed": False,
                "claim_boundary": (
                    "Already-open E12/E14 teacher-forced native-truth evidence only. "
                    "Fine is an off-grid query feature and has no truth label. No "
                    "recurrent, cross-family, conservation, convergence, direct-off-grid "
                    "deployment, asymptotic-order, or Richardson claim."
                ),
                "git": git_state(ROOT),
            }
        )
        atomic_write_json(output_dir / "affine_transfer_evaluation.json", result)
        return result, 0 if gate["status"] == "passed" else 4
    finally:
        base._close_runtime(runtime)


def _add_external_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base-readiness", type=Path, required=True)
    parser.add_argument("--family-root", type=Path, required=True)
    parser.add_argument("--multires-reference-root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--normalization-file", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--d063-run-contract", type=Path, required=True)
    parser.add_argument("--d063-summary", type=Path, required=True)
    parser.add_argument("--a28-map", type=Path, required=True)
    parser.add_argument("--a22-result", type=Path, required=True)
    parser.add_argument("--a22-reference-checks", type=Path, required=True)
    parser.add_argument("--a23-result", type=Path, required=True)
    parser.add_argument("--a23-reference-checks", type=Path, required=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("synthetic")
    preflight = commands.add_parser("preflight")
    _add_external_arguments(preflight)
    preflight.add_argument("--output", type=Path, required=True)
    evaluate = commands.add_parser("evaluate")
    _add_external_arguments(evaluate)
    evaluate.add_argument("--preflight", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    evaluate.add_argument("--allow-cpu", action="store_true")
    evaluate.add_argument("--repeat-forward", type=int, default=2)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "synthetic":
        payload = synthetic_summary()
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if payload["status"] == "passed" else 2
    if args.command == "preflight":
        payload = run_preflight(args)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.repeat_forward < 2:
        raise ValueError("A29 repeat-forward must be at least two")
    payload, exit_code = run_evaluation(args)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
