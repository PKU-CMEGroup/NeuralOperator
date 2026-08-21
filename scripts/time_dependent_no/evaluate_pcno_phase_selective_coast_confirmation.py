#!/usr/bin/env python3
"""Confirm frozen A43 on disjoint E00/E11 interior trajectories."""

from __future__ import annotations

import argparse
import json
import os
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
    evaluate_pcno_binary_position_phase_rollout as a31,
)
from scripts.time_dependent_no import (
    evaluate_pcno_phase_selective_coast_replay as a43,
)
from scripts.time_dependent_no import evaluate_pcno_response_filtered_block as response
from utility.time_dependent_no.pcno_artifacts import (
    atomic_write_json_with_paths as atomic_write_json,
)
from utility.time_dependent_no.pcno_artifacts import (
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
    NativeIncrementBasis,
    case_first_statistics,
    score_scalar_correction,
)
from utility.time_dependent_no.pcno_cross_resolution_teacher_forced import (
    CALIBRATION_CASE_IDS,
    EVALUATION_CASE_IDS,
    VIEW_SPECS,
    _statistics_for_view,
    verify_payload_sha256,
    with_payload_sha256,
)
from utility.time_dependent_no.pcno_resolution_transfer import reference_at_resolution
from utility.time_dependent_no.pcno_response_filtered_block import (
    FROZEN_POSITION_BUFFER_THRESHOLD,
)
from utility.time_dependent_no.shock_vortex_family import family_case_provenance

WORKING_ID = "W26-L5-P6-RFB19-A44-R1-NATIVE-SHARD-CONFIRMATION"
PREFLIGHT_SCHEMA = "pcno_phase_selective_coast_confirmation_preflight_v2"
RESULT_SCHEMA = "pcno_phase_selective_coast_confirmation_v2"
SOURCE_SCHEMA = "pcno_phase_selective_coast_confirmation_source_v2"
ANIMATION_SCHEMA = "pcno_phase_selective_coast_confirmation_animation_bundles_v2"

A43_RESULT_SHA256 = "69eeb489609637beadad49ba65706c6392d61b29cfd55153093243f9d1e4560b"
A43_PAYLOAD_SHA256 = "ebfe93a7dad4cba3f45b1bcbf01d54f58de6b041b25a5283c4f59803253a9730"
REQUIRED_CUBLAS_WORKSPACE_CONFIG = ":4096:8"

NATIVE_RESOLUTION = a43.NATIVE_RESOLUTION
INPUT_CALLS = tuple(range(30))
BANDS = {
    "overall": INPUT_CALLS,
    "band_0_7": tuple(range(8)),
    "band_8_14": tuple(range(8, 15)),
    "band_15_21": tuple(range(15, 22)),
    "band_22_29": tuple(range(22, 30)),
    "endpoint_29": (29,),
}
GROUP_CASES = {
    group: tuple(f"sv_{group}_y{position:02d}" for position in range(1, 8))
    for group in ("e00", "e11")
}
GROUP_PROVENANCE_IDS = {"e00": "train_e00", "e11": "train_e11"}
CASE_IDS = tuple(case_id for cases in GROUP_CASES.values() for case_id in cases)
CASE_TO_GROUP = {
    case_id: group for group, cases in GROUP_CASES.items() for case_id in cases
}
EXPECTED_ACTIVE_CASES = (
    "sv_e00_y01",
    "sv_e00_y07",
    "sv_e11_y01",
    "sv_e11_y07",
)
ACTIVE_GROUP_CASES = {
    group: tuple(case_id for case_id in cases if case_id in EXPECTED_ACTIVE_CASES)
    for group, cases in GROUP_CASES.items()
}
INACTIVE_CASES = tuple(
    case_id for case_id in CASE_IDS if case_id not in EXPECTED_ACTIVE_CASES
)
ANIMATION_CASE_IDS = (
    "sv_e00_y01",
    "sv_e00_y04",
    "sv_e00_y07",
    "sv_e11_y01",
    "sv_e11_y07",
)
ANIMATION_CALLS = tuple(range(0, 31, 2))
ANIMATION_CONTRACT = {
    **a43.ANIMATION_CONTRACT,
    "case_ids": list(ANIMATION_CASE_IDS),
    "candidate_label": "frozen A43 phase-selective coast confirmation",
    "output_stem": "phase_selective_coast_confirmation_h30",
}
CONTROL_LIMIT = 1.05
STRICT_TOLERANCE = 1.0e-12
RAW_CALLS = 420
CANDIDATE_CALLS = 548

OWNED_SOURCE_PATHS = (
    "docs/time_dependent_no/W26_L5_A43_INDEPENDENT_CONFIRMATION_PREREGISTRATION.md",
    "scripts/time_dependent_no/evaluate_pcno_phase_selective_coast_confirmation.py",
    "scripts/time_dependent_no/visualize_pcno_phase_selective_coast_confirmation.py",
    "tests/time_dependent_no/test_pcno_phase_selective_coast_confirmation.py",
)
DEPENDENCY_PATHS = tuple(
    dict.fromkeys(
        (
            "scripts/time_dependent_no/evaluate_pcno_phase_selective_coast_replay.py",
            "scripts/time_dependent_no/visualize_pcno_phase_selective_coast_replay.py",
            *a43.DEPENDENCY_PATHS,
            "utility/time_dependent_no/shock_vortex_family.py",
        )
    )
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON payload must be an object: {path}")
    return value


def _verify_a43(path: Path) -> dict[str, Any]:
    if sha256_file(path) != A43_RESULT_SHA256:
        raise ValueError("A43 result file SHA-256 differs")
    payload = _read_json(path)
    verify_payload_sha256(payload)
    if (
        payload.get("schema") != a43.RESULT_SCHEMA
        or payload.get("working_id") != a43.WORKING_ID
        or payload.get("payload_sha256") != A43_PAYLOAD_SHA256
        or payload.get("status") != "qualified_phase_selective_replay"
        or payload.get("gate", {}).get("status") != "qualified"
        or payload.get("gate", {}).get("failed_checks") != []
        or not all(payload.get("gate", {}).get("checks", {}).values())
        or payload.get("recurrence_executed") is not True
    ):
        raise ValueError("A43 is not the exact qualified frozen result")
    inventory = payload.get("artifact_inventory_before_summary")
    if not isinstance(inventory, Mapping) or not inventory:
        raise ValueError("A43 artifact inventory is unavailable")
    root = path.parent.resolve()
    for relative, expected in inventory.items():
        candidate = (root / str(relative)).resolve()
        if root not in candidate.parents or not isinstance(expected, Mapping):
            raise ValueError("A43 artifact inventory contains an unsafe path")
        if (
            not candidate.is_file()
            or sha256_file(candidate) != expected.get("sha256")
            or candidate.stat().st_size != int(expected.get("bytes", -1))
        ):
            raise ValueError(f"A43 artifact differs: {relative}")
    return payload


def _source_hashes() -> dict[str, Any]:
    return {
        "owned": sha256_files(OWNED_SOURCE_PATHS, root=ROOT),
        "dependencies": sha256_files(DEPENDENCY_PATHS, root=ROOT),
    }


def _native_truth_inventory(
    args: argparse.Namespace, store: Any
) -> dict[str, dict[str, Any]]:
    root = args.data_dir.resolve()
    output: dict[str, dict[str, Any]] = {}
    for case_id in CASE_IDS:
        entry = store.entry(case_id)
        folder = entry.get("folder")
        expected_arrays = entry.get("array_sha256")
        source_sha = entry.get("source_reference_sha256")
        state_digest = entry.get("state_digest")
        if (
            not isinstance(folder, str)
            or not folder
            or not isinstance(expected_arrays, Mapping)
            or not isinstance(source_sha, str)
            or not isinstance(state_digest, str)
        ):
            raise ValueError(f"A44 shard manifest entry is incomplete: {case_id}")
        case_root = (root / folder).resolve()
        if root not in case_root.parents or not case_root.is_dir():
            raise ValueError(f"A44 native truth folder is unavailable: {case_id}")
        arrays: dict[str, dict[str, Any]] = {}
        for name in ("states_conservative", "physical_times"):
            expected_sha = expected_arrays.get(name)
            path = (case_root / f"{name}.npy").resolve()
            if (
                case_root not in path.parents
                or not path.is_file()
                or not isinstance(expected_sha, str)
            ):
                raise ValueError(
                    f"A44 native truth array is unavailable: {case_id}/{name}"
                )
            actual_sha = sha256_file(path)
            if actual_sha != expected_sha:
                raise ValueError(f"A44 native truth digest differs: {case_id}/{name}")
            arrays[name] = {
                "relative_path": str(path.relative_to(root)).replace("\\", "/"),
                "sha256": actual_sha,
                "bytes": path.stat().st_size,
            }
        metadata_path = (case_root / "metadata.json").resolve()
        if case_root not in metadata_path.parents or not metadata_path.is_file():
            raise ValueError(f"A44 shard metadata is unavailable: {case_id}")
        metadata = _read_json(metadata_path)
        serialization = metadata.get("float32_serialization")
        tolerances = metadata.get("float32_tolerances")
        try:
            state_global_error = float(serialization["state_global_relative_l2"])
            state_max_error = float(serialization["state_max_frame_relative_l2"])
            state_tolerance = float(tolerances["state_relative_l2"])
        except (KeyError, TypeError, ValueError):
            state_global_error = state_max_error = float("inf")
            state_tolerance = -1.0
        if (
            metadata.get("source_key") != case_id
            or metadata.get("source_reference_sha256") != source_sha
            or not isinstance(serialization, Mapping)
            or not isinstance(tolerances, Mapping)
            or not all(
                np.isfinite(value)
                for value in (state_global_error, state_max_error, state_tolerance)
            )
            or state_global_error < 0.0
            or state_max_error < 0.0
            or state_tolerance <= 0.0
            or state_max_error > state_tolerance
        ):
            raise ValueError(f"A44 native truth serialization is unverified: {case_id}")
        output[case_id] = {
            "truth_kind": "checkpoint_bound_native_shard",
            "folder": folder,
            "source_reference_sha256": source_sha,
            "state_digest": state_digest,
            "arrays": arrays,
            "metadata": {
                "relative_path": str(metadata_path.relative_to(root)).replace(
                    "\\", "/"
                ),
                "sha256": sha256_file(metadata_path),
                "bytes": metadata_path.stat().st_size,
            },
            "float32_serialization": {
                "state_global_relative_l2": state_global_error,
                "state_max_frame_relative_l2": state_max_error,
                "state_relative_l2_tolerance": state_tolerance,
            },
        }
    return output


def _source_manifest(
    args: argparse.Namespace,
    *,
    base_source_manifest: Mapping[str, Any],
    truth_inventory: Mapping[str, Any],
) -> dict[str, Any]:
    a43_payload = _verify_a43(args.a43_result)
    _, _, a28_payload = a31._verify_a28(args.a28_map)
    return with_payload_sha256(
        {
            "schema": SOURCE_SCHEMA,
            "working_id": WORKING_ID,
            "source_sha256": _source_hashes(),
            "lineage": {
                "a43_result_sha256": sha256_file(args.a43_result),
                "a43_payload_sha256": a43_payload["payload_sha256"],
                "a43_source_manifest_sha256": a43_payload["source_manifest_sha256"],
                "a28_result_sha256": sha256_file(args.a28_map),
                "a28_payload_sha256": a28_payload["payload_sha256"],
            },
            "base_runtime_source_manifest": dict(base_source_manifest),
            "native_truth_artifacts": dict(truth_inventory),
            "population": {
                "case_ids": list(CASE_IDS),
                "groups": {key: list(value) for key, value in GROUP_CASES.items()},
                "expected_active_cases": list(EXPECTED_ACTIVE_CASES),
                "family_split": "train",
                "status": "new_disjoint_correction_confirmation_training_support",
                "independent_pcno_model_validation": False,
            },
            "inference_contract": {
                "frozen_method": a43.POLICY,
                "position_descriptor": "call_zero_transverse_velocity_centroid",
                "position_threshold": FROZEN_POSITION_BUFFER_THRESHOLD,
                "exact_active_calls": [
                    call for call in INPUT_CALLS if call not in a43.COAST_CALLS
                ],
                "coast_calls": list(a43.COAST_CALLS),
                "exact_call_order": [
                    "shadow_native",
                    "accepted_native",
                    "accepted_fine",
                ],
                "coast_call_order": ["shadow_native"],
                "candidate_logical_calls": CANDIDATE_CALLS,
                "raw_logical_calls": RAW_CALLS,
                "one_accepted_native_state_per_call": True,
                "fine_truth_loaded": False,
                "coefficient_refit": False,
                "truth_or_case_id_in_decision": False,
                "route_audit_before_first_model_call": True,
                "cublas_workspace_config": REQUIRED_CUBLAS_WORKSPACE_CONFIG,
            },
            "animation_contract": ANIMATION_CONTRACT,
        }
    )


def synthetic_summary() -> dict[str, Any]:
    checks = {
        "population_inventory_14": len(CASE_IDS) == 14,
        "two_complete_groups": all(len(cases) == 7 for cases in GROUP_CASES.values()),
        "active_inventory_4": len(EXPECTED_ACTIVE_CASES) == 4,
        "inactive_inventory_10": len(INACTIVE_CASES) == 10,
        "disjoint_from_a43_cases": not set(CASE_IDS).intersection(a43.CASE_IDS),
        "disjoint_from_correction_fit": not set(CASE_IDS).intersection(
            CALIBRATION_CASE_IDS
        ),
        "disjoint_from_a3_evaluation": not set(CASE_IDS).intersection(
            EVALUATION_CASE_IDS
        ),
        "coast_inventory_14": a43.COAST_CALLS == tuple(range(8, 22)),
        "route_inventory": 64 + 56 + 300 == len(CASE_IDS) * len(INPUT_CALLS),
        "candidate_calls_548": 300 + 120 + 64 + 64 == CANDIDATE_CALLS,
        "raw_calls_420": len(CASE_IDS) * len(INPUT_CALLS) == RAW_CALLS,
        "animation_inventory_5": len(ANIMATION_CASE_IDS) == 5,
        "owned_source_inventory_4": len(OWNED_SOURCE_PATHS) == 4,
    }
    return {
        "schema": "pcno_phase_selective_coast_confirmation_synthetic_v2",
        "working_id": WORKING_ID,
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
    }


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    _, checkpoint, manifest, store, base_source = a43.base._open_contract(
        a43.a29._base_args(args)
    )
    try:
        provenance = {
            case_id: family_case_provenance(manifest, case_id) for case_id in CASE_IDS
        }
        references = _native_truth_inventory(args, store)
        source = _source_manifest(
            args,
            base_source_manifest=base_source,
            truth_inventory=references,
        )
        checks = {
            "source_inventory_exact": set(source["source_sha256"]["owned"])
            == set(OWNED_SOURCE_PATHS)
            and set(source["source_sha256"]["dependencies"]) == set(DEPENDENCY_PATHS),
            "lineage_exact": source["lineage"]["a43_result_sha256"] == A43_RESULT_SHA256
            and source["lineage"]["a43_payload_sha256"] == A43_PAYLOAD_SHA256
            and source["lineage"]["a28_result_sha256"] == a31.A28_RESULT_SHA256
            and source["lineage"]["a28_payload_sha256"] == a31.A28_PAYLOAD_SHA256,
            "case_inventory_present": all(
                case_id in store.keys for case_id in CASE_IDS
            ),
            "population_exact": set(provenance) == set(CASE_IDS),
            "groups_exact": all(
                provenance[case_id].get("split") == "train"
                and provenance[case_id].get("split_group_id")
                == GROUP_PROVENANCE_IDS[group]
                for group, cases in GROUP_CASES.items()
                for case_id in cases
            ),
            "native_truth_inventory_exact": set(references) == set(CASE_IDS),
            "checkpoint_stride_exact": a43.base.checkpoint_step_stride(checkpoint) == 2,
            "synthetic_contract_passed": synthetic_summary()["status"] == "passed",
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
                "truth_arrays_loaded": False,
                "model_calls": 0,
                "recurrence_executed": False,
                "population": provenance,
            }
        )
    finally:
        store.close()
    if payload["status"] != "passed":
        raise ValueError("A44 preflight checks did not all pass")
    atomic_write_json(args.output, payload)
    return payload


def _verify_preflight(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    payload = _read_json(path)
    verify_payload_sha256(payload)
    source = payload.get("source_manifest", {})
    references = source.get("native_truth_artifacts", {})
    if (
        payload.get("schema") != PREFLIGHT_SCHEMA
        or payload.get("working_id") != WORKING_ID
        or payload.get("status") != "passed"
        or not all(payload.get("checks", {}).values())
        or payload.get("checkpoint_model_built") is not False
        or payload.get("reference_arrays_loaded") is not False
        or payload.get("truth_arrays_loaded") is not False
        or payload.get("model_calls") != 0
        or payload.get("recurrence_executed") is not False
        or source.get("source_sha256") != _source_hashes()
        or source.get("lineage", {}).get("a43_result_sha256") != A43_RESULT_SHA256
        or source.get("lineage", {}).get("a43_payload_sha256") != A43_PAYLOAD_SHA256
        or set(references) != set(CASE_IDS)
    ):
        raise ValueError("A44 preflight differs from the frozen contract")
    _verify_a43(args.a43_result)
    a31._verify_a28(args.a28_map)
    root = args.data_dir.resolve()
    for case_id, expected in references.items():
        artifacts = [expected.get("metadata", {})]
        arrays = expected.get("arrays", {})
        artifacts.extend(
            arrays.get(name, {}) for name in ("states_conservative", "physical_times")
        )
        for artifact in artifacts:
            path = (root / str(artifact.get("relative_path", ""))).resolve()
            if (
                root not in path.parents
                or not path.is_file()
                or path.stat().st_size != int(artifact.get("bytes", -1))
                or sha256_file(path) != artifact.get("sha256")
            ):
                raise ValueError(f"A44 native truth changed after preflight: {case_id}")
    return payload


def _load_reference(
    runtime: Any,
    case_id: str,
    expected: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    entry = runtime.store.entry(case_id)
    expected_arrays = expected.get("arrays", {})
    if (
        expected.get("truth_kind") != "checkpoint_bound_native_shard"
        or expected.get("folder") != entry.get("folder")
        or expected.get("source_reference_sha256")
        != entry.get("source_reference_sha256")
        or expected.get("state_digest") != entry.get("state_digest")
        or any(
            expected_arrays.get(name, {}).get("sha256")
            != entry.get("array_sha256", {}).get(name)
            for name in ("states_conservative", "physical_times")
        )
    ):
        raise ValueError(f"A44 native truth identity differs: {case_id}")
    stored_states = runtime.store.states(case_id)
    stored_times = runtime.store.array(case_id, "physical_times")
    state_dtype = str(stored_states.dtype)
    states = np.asarray(stored_states, dtype=np.float64)
    times = np.asarray(stored_times, dtype=np.float64)
    if states.shape != (61, NATIVE_RESOLUTION[0] * NATIVE_RESOLUTION[1], 4):
        raise ValueError("A44 native reference shape differs")
    if times.shape != (61,) or not np.allclose(
        times, np.arange(61) / 100.0, rtol=0.0, atol=1.0e-15
    ):
        raise ValueError("A44 physical-time contract differs")
    reference = {
        "conservative_states": states,
        "physical_times": times,
        "retained_resolution": NATIVE_RESOLUTION,
    }
    check = {
        "case_id": case_id,
        "truth_kind": expected["truth_kind"],
        "source_reference_sha256": expected["source_reference_sha256"],
        "states_array_sha256": expected_arrays["states_conservative"]["sha256"],
        "physical_times_array_sha256": expected_arrays["physical_times"]["sha256"],
        "state_digest": expected["state_digest"],
        "retained_resolution": "250x100",
        "state_dtype": state_dtype,
        "state_shape": list(states.shape),
        **expected["float32_serialization"],
    }
    return reference, check


def _pair_view_statistics(
    runtime: Any,
    *,
    case_id: str,
    reference: Mapping[str, Any],
    raw_states: Sequence[np.ndarray],
    candidate_states: Sequence[np.ndarray],
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    if len(raw_states) != 31 or len(candidate_states) != 31:
        raise ValueError("A44 paired trajectory is incomplete")
    reference_resolution = tuple(
        int(value) for value in reference["retained_resolution"]
    )
    geometry = runtime.geometry_by_resolution[NATIVE_RESOLUTION]
    volumes = np.asarray(geometry.node_measures, dtype=np.float64)
    state_scale = np.asarray(runtime.normalization.state_scale, dtype=np.float64)
    output: list[dict[str, Any]] = []
    maxima = {"band": 0.0, "subspace_reconstruction": 0.0, "orthogonality": 0.0}
    for input_call in INPUT_CALLS:
        target = reference_at_resolution(
            reference["conservative_states"][2 * (input_call + 1)],
            reference_resolution=reference_resolution,
            target_resolution=NATIVE_RESOLUTION,
        )
        if target is None:
            raise ValueError("A44 view target is unavailable")
        target = np.asarray(target, dtype=np.float64)
        raw = np.asarray(raw_states[input_call + 1], dtype=np.float64)
        candidate = np.asarray(candidate_states[input_call + 1], dtype=np.float64)
        correction = candidate - raw
        zero = np.zeros_like(correction)
        masks, _, _ = a43.a29.shock_vortex_regions(
            target,
            geometry.nodes,
            resolution=NATIVE_RESOLUTION,
            gamma=runtime.normalization.gamma,
        )
        snapshot = DiagnosticSnapshot(
            case_id=case_id,
            group_id=CASE_TO_GROUP[case_id],
            input_call=input_call,
            basis=NativeIncrementBasis(
                native_increment=zero,
                coarse_on_native=zero,
                fine_on_native=correction,
                native_minus_coarse=zero,
                fine_minus_native=correction,
            ),
            target_correction=target - raw,
            volumes=volumes,
            component_scale=state_scale,
            masks=masks,
        )
        for view in VIEW_SPECS:
            statistics, closure = _statistics_for_view(
                snapshot,
                view,
                resolution=NATIVE_RESOLUTION,
                projector=runtime.native_projector,
            )
            maxima["band"] = max(
                maxima["band"],
                float(closure["maximum_reconstruction_abs_residual_scaled"]),
                float(closure["maximum_instantaneous_energy_relative_closure"]),
            )
            maxima["subspace_reconstruction"] = max(
                maxima["subspace_reconstruction"],
                float(closure["maximum_subspace_reconstruction_abs"]),
            )
            maxima["orthogonality"] = max(
                maxima["orthogonality"],
                abs(float(closure["maximum_parallel_orthogonal_weighted_inner"])),
            )
            output.append(
                {
                    "case_id": case_id,
                    "group_id": CASE_TO_GROUP[case_id],
                    "input_call": input_call,
                    "view": view.key,
                    "statistics": statistics,
                }
            )
    return output, maxima


def _population_specs() -> dict[str, tuple[str, ...]]:
    return {
        "population": CASE_IDS,
        "active_population": EXPECTED_ACTIVE_CASES,
        "group_e00": GROUP_CASES["e00"],
        "group_e11": GROUP_CASES["e11"],
        "active_group_e00": ACTIVE_GROUP_CASES["e00"],
        "active_group_e11": ACTIVE_GROUP_CASES["e11"],
    }


def _score_views(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cell, calls in BANDS.items():
        call_set = set(calls)
        for view in VIEW_SPECS:
            view_records = [
                row
                for row in records
                if row["view"] == view.key and row["input_call"] in call_set
            ]
            for scope, cases in _population_specs().items():
                selected = [
                    (str(row["case_id"]), row["statistics"])
                    for row in view_records
                    if row["case_id"] in cases
                ]
                score = score_scalar_correction(
                    case_first_statistics(selected), (0.0, 1.0)
                )
                rows.append(
                    {
                        "cell": cell,
                        "view": view.key,
                        "scope": scope,
                        "case_id": None,
                        **asdict(score),
                    }
                )
            for case_id in CASE_IDS:
                selected = [
                    (str(row["case_id"]), row["statistics"])
                    for row in view_records
                    if row["case_id"] == case_id
                ]
                score = score_scalar_correction(
                    case_first_statistics(selected), (0.0, 1.0)
                )
                rows.append(
                    {
                        "cell": cell,
                        "view": view.key,
                        "scope": "case",
                        "case_id": case_id,
                        **asdict(score),
                    }
                )
    return rows


def _field_controls(score_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    scopes = (
        "population",
        "active_population",
        "active_group_e00",
        "active_group_e11",
    )
    for cell in ("overall", "endpoint_29"):
        for view in VIEW_SPECS:
            for scope in scopes:
                row = a31._score_row(score_rows, cell=cell, view=view.key, scope=scope)
                output.append(
                    {
                        "key": f"field::{view.key}",
                        "scope": scope,
                        "cell": cell,
                        "zero_rms": row["target_rms"],
                        "corrected_rms": (
                            float(row["target_rms"]) * float(row["rms_ratio_vs_zero"])
                            if row["rms_ratio_vs_zero"] is not None
                            else None
                        ),
                        "ratio": row["rms_ratio_vs_zero"],
                        "status": (
                            "ok" if row["skill_status"] == "ok" else row["skill_status"]
                        ),
                        "policy": a43.POLICY,
                    }
                )
            for case_id in CASE_IDS:
                row = a31._score_row(
                    score_rows,
                    cell=cell,
                    view=view.key,
                    scope="case",
                    case_id=case_id,
                )
                output.append(
                    {
                        "key": f"field::{view.key}",
                        "scope": case_id,
                        "cell": cell,
                        "zero_rms": row["target_rms"],
                        "corrected_rms": (
                            float(row["target_rms"]) * float(row["rms_ratio_vs_zero"])
                            if row["rms_ratio_vs_zero"] is not None
                            else None
                        ),
                        "ratio": row["rms_ratio_vs_zero"],
                        "status": (
                            "ok" if row["skill_status"] == "ok" else row["skill_status"]
                        ),
                        "policy": a43.POLICY,
                    }
                )
    return output


def _aggregate_paired_controls(
    all_metric_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    case_rows, controls, population = response._paired_rollout_controls_for_cases(
        all_metric_rows, case_ids=CASE_IDS
    )
    scoped_controls = list(controls)
    scoped_populations: dict[str, Any] = {"population": population}
    for scope, cases in (
        ("active_population", EXPECTED_ACTIVE_CASES),
        ("active_group_e00", ACTIVE_GROUP_CASES["e00"]),
        ("active_group_e11", ACTIVE_GROUP_CASES["e11"]),
    ):
        _, local_controls, local_population = (
            response._paired_rollout_controls_for_cases(
                [row for row in all_metric_rows if row["case_id"] in cases],
                case_ids=cases,
            )
        )
        for row in local_controls:
            if row["scope"] == "population":
                scoped_controls.append({**row, "scope": scope})
        scoped_populations[scope] = local_population
    return case_rows, scoped_controls, scoped_populations


def _write_animation_bundle(
    output_dir: Path,
    *,
    runtime: Any,
    case_id: str,
    reference: Mapping[str, Any],
    raw_states: Sequence[np.ndarray],
    candidate_states: Sequence[np.ndarray],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    reference_resolution = tuple(
        int(value) for value in reference["retained_resolution"]
    )
    truth_states = []
    for output_call in range(31):
        value = reference_at_resolution(
            reference["conservative_states"][2 * output_call],
            reference_resolution=reference_resolution,
            target_resolution=NATIVE_RESOLUTION,
        )
        if value is None:
            raise ValueError("A44 animation truth is unavailable")
        truth_states.append(np.asarray(value, dtype=np.float64))
    indices = np.asarray(ANIMATION_CALLS, dtype=np.int64)
    geometry = runtime.geometry_by_resolution[NATIVE_RESOLUTION]
    path = output_dir / f"{case_id}_phase_selective_coast_confirmation_h30.npz"
    np.savez_compressed(
        path,
        case_id=np.asarray(case_id),
        split_group_id=np.asarray(GROUP_PROVENANCE_IDS[CASE_TO_GROUP[case_id]]),
        output_calls=indices,
        physical_times=np.asarray(
            ANIMATION_CONTRACT["physical_times"], dtype=np.float64
        ),
        native_resolution=np.asarray(NATIVE_RESOLUTION, dtype=np.int64),
        nodes=np.asarray(geometry.nodes, dtype=np.float32),
        volumes=np.asarray(geometry.node_measures, dtype=np.float32).reshape(-1),
        state_scale=np.asarray(runtime.normalization.state_scale, dtype=np.float32),
        residual_scale=np.asarray(
            runtime.normalization.residual_scale, dtype=np.float32
        ),
        gamma=np.asarray(runtime.normalization.gamma, dtype=np.float64),
        truth_conservative=np.asarray(truth_states, dtype=np.float64)[indices].astype(
            np.float32
        ),
        raw_shadow_conservative=np.asarray(raw_states, dtype=np.float64)[
            indices
        ].astype(np.float32),
        corrected_conservative=np.asarray(candidate_states, dtype=np.float64)[
            indices
        ].astype(np.float32),
    )
    return {
        "case_id": case_id,
        "split_group_id": GROUP_PROVENANCE_IDS[CASE_TO_GROUP[case_id]],
        "path": path.name,
        "sha256": sha256_file(path),
        "frames": len(indices),
        "storage_dtype": "float32_visualization_only",
    }


def _gate(
    *,
    score_rows: Sequence[Mapping[str, Any]],
    case_rows: Sequence[Mapping[str, Any]],
    controls: Sequence[Mapping[str, Any]],
    rollouts: Sequence[Mapping[str, Any]],
    audit_rows: Sequence[Mapping[str, Any]],
    inactive_max_abs: Mapping[str, float],
    shadow_raw_max_abs: Mapping[str, float],
    prefix_repeat_abs: float,
    main_execution: Mapping[str, Any],
    structural_checks: Mapping[str, bool],
) -> dict[str, Any]:
    checks = dict(structural_checks)
    trajectory_scopes = (
        "population",
        "active_population",
        "active_group_e00",
        "active_group_e11",
    )
    for scope in trajectory_scopes:
        checks[f"{scope}_full_rank8_trajectory_strict"] = all(
            a31._score_row(score_rows, cell="overall", view=view, scope=scope)[
                "skill_status"
            ]
            == "ok"
            and float(
                a31._score_row(score_rows, cell="overall", view=view, scope=scope)[
                    "rms_ratio_vs_zero"
                ]
            )
            < 1.0
            for view in ("full", "rank8_parallel")
        )
    for scope in ("active_population", "active_group_e00", "active_group_e11"):
        row = a31._score_row(score_rows, cell="endpoint_29", view="full", scope=scope)
        checks[f"{scope}_full_endpoint_strict"] = (
            row["skill_status"] == "ok" and float(row["rms_ratio_vs_zero"]) < 1.0
        )
    paired = {str(row["case_id"]): row for row in case_rows}
    active_wins = 0
    active_bounds = True
    active_case_ratios: dict[str, dict[str, float]] = {}
    for case_id in EXPECTED_ACTIVE_CASES:
        trajectory = float(
            a31._score_row(
                score_rows,
                cell="overall",
                view="full",
                scope="case",
                case_id=case_id,
            )["rms_ratio_vs_zero"]
        )
        endpoint = float(
            a31._score_row(
                score_rows,
                cell="endpoint_29",
                view="full",
                scope="case",
                case_id=case_id,
            )["rms_ratio_vs_zero"]
        )
        active_case_ratios[case_id] = {
            "trajectory": trajectory,
            "endpoint": endpoint,
        }
        active_wins += trajectory < 1.0 and endpoint < 1.0
        active_bounds = active_bounds and all(
            float(value) <= CONTROL_LIMIT
            for value in (
                trajectory,
                endpoint,
                paired[case_id]["increment_defect_rms_ratio"],
                paired[case_id]["endpoint_cumulative_defect_ratio"],
            )
        )
    checks["at_least_three_active_case_trajectory_endpoint_wins"] = active_wins >= 3
    checks["every_active_case_primary_no_harm"] = active_bounds
    checks["all_controls_no_harm"] = bool(controls) and all(
        a31._control_passed(row) for row in controls
    )
    checks["inactive_states_bitwise_raw"] = set(inactive_max_abs) == set(
        INACTIVE_CASES
    ) and all(value == 0.0 for value in inactive_max_abs.values())
    checks["active_shadow_byte_equal_raw"] = set(shadow_raw_max_abs) == set(
        EXPECTED_ACTIVE_CASES
    ) and all(value == 0.0 for value in shadow_raw_max_abs.values())
    checks["all_rollouts_complete_finite_admissible"] = len(rollouts) == 2 * len(
        CASE_IDS
    ) and all(
        row["execution"]["completed_calls"] == 30
        and all(metric["finite"] and metric["admissible"] for metric in row["rows"])
        for row in rollouts
    )
    exact = [row for row in audit_rows if row["route"] == "exact_a32_window"]
    coast = [row for row in audit_rows if row["route"] == "surrogate_coast"]
    inactive = [row for row in audit_rows if row["route"] == "inactive_raw"]
    checks["route_audit_inventory_exact"] = (
        len(exact) == 64 and len(coast) == 56 and len(inactive) == 300
    )
    checks["exact_a32_window_closure"] = all(
        bool(row["call_order_exact"])
        and int(row["logical_call_count"]) == 3
        and bool(row["correction_active"])
        and float(row["correction_to_native_increment"]) <= 0.05 + STRICT_TOLERANCE
        and max(
            float(row["pre_model_fine_to_native_max_abs"]),
            float(row["maximum_scaled_component_mean_abs"]),
            float(row["maximum_excluded_abs"]),
            float(row["maximum_inactive_coordinate_abs"]),
            float(row["maximum_modal_reconstruction_abs"]),
            float(row["maximum_proposal_update_abs"]),
            float(row["maximum_integral_difference_abs"]),
            float(row["maximum_boundary_difference_abs"]),
            float(row["maximum_projection_idempotence_abs"]),
            float(row["maximum_update_identity_abs"]),
        )
        <= STRICT_TOLERANCE
        and float(row["post_fp32_fine_to_native_max_abs"]) <= 1.0e-6
        for row in exact
    )
    checks["surrogate_coast_one_native_closure"] = all(
        bool(row["call_order_exact"])
        and int(row["logical_call_count"]) == 1
        and not bool(row["correction_active"])
        and max(
            float(row["pre_model_fine_to_native_max_abs"]),
            float(row["post_fp32_fine_to_native_max_abs"]),
            float(row["maximum_proposal_update_abs"]),
            float(row["maximum_integral_difference_abs"]),
            float(row["maximum_boundary_difference_abs"]),
            float(row["maximum_projection_idempotence_abs"]),
            float(row["maximum_update_identity_abs"]),
        )
        <= STRICT_TOLERANCE
        for row in coast
    )
    checks["inactive_exact_raw_route"] = all(
        bool(row["call_order_exact"])
        and int(row["logical_call_count"]) == 1
        and float(row["retained_displacement_rms"]) == 0.0
        for row in inactive
    )
    checks["deterministic_prefix"] = prefix_repeat_abs <= 1.0e-6
    checks["logical_cost_exact"] = (
        main_execution["raw"]["logical_model_calls"] == RAW_CALLS
        and main_execution["candidate"]["logical_model_calls"] == CANDIDATE_CALLS
    )
    failed = sorted(key for key, value in checks.items() if not bool(value))
    return {
        "status": "qualified" if not failed else "failed",
        "checks": checks,
        "failed_checks": failed,
        "active_case_joint_win_count": active_wins,
        "active_case_ratios": active_case_ratios,
        "strict_improvement_equality_fails": True,
        "control_ratio_limit": CONTROL_LIMIT,
    }


def run_rollout(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != REQUIRED_CUBLAS_WORKSPACE_CONFIG:
        raise ValueError(
            f"A44 requires CUBLAS_WORKSPACE_CONFIG={REQUIRED_CUBLAS_WORKSPACE_CONFIG}"
        )
    preflight = _verify_preflight(args.preflight, args)
    a43_payload = _verify_a43(args.a43_result)
    start_coefficients, end_coefficients, _ = a31._verify_a28(args.a28_map)
    expected_references = preflight["source_manifest"]["native_truth_artifacts"]
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=False)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    runtime = a43.base._build_runtime(a43.a29._base_args(args))
    started = perf_counter()
    try:
        if (
            runtime.source_manifest
            != preflight["source_manifest"]["base_runtime_source_manifest"]
        ):
            raise ValueError("A44 runtime source differs from preflight")

        references: dict[str, dict[str, Any]] = {}
        reference_rows: list[dict[str, Any]] = []
        descriptor_rows: list[dict[str, Any]] = []
        descriptors: dict[str, Any] = {}
        for case_id in CASE_IDS:
            reference, check = _load_reference(
                runtime, case_id, expected_references[case_id]
            )
            initial = np.asarray(reference["conservative_states"][0], dtype=np.float64)
            descriptor = a31._descriptor(runtime, initial)
            selected = bool(
                descriptor.status == "ok"
                and descriptor.normalized_wall_distance is not None
                and descriptor.normalized_wall_distance
                <= FROZEN_POSITION_BUFFER_THRESHOLD
            )
            references[case_id] = reference
            descriptors[case_id] = descriptor
            reference_rows.append({"case_id": case_id, **check})
            descriptor_rows.append(
                {
                    "case_id": case_id,
                    "group_id": CASE_TO_GROUP[case_id],
                    **asdict(descriptor),
                    "position_threshold": FROZEN_POSITION_BUFFER_THRESHOLD,
                    "position_selected": selected,
                    "expected_selected": case_id in EXPECTED_ACTIVE_CASES,
                }
            )
        selected_cases = tuple(
            row["case_id"] for row in descriptor_rows if row["position_selected"]
        )
        if set(selected_cases) != set(EXPECTED_ACTIVE_CASES):
            raise ValueError("A44 route inventory differs before the first model call")

        prefix_case = EXPECTED_ACTIVE_CASES[0]
        prefix_runs = [
            a43._run_candidate_arm(
                runtime,
                case_id=prefix_case,
                reference=references[prefix_case],
                position_selected=True,
                descriptor=descriptors[prefix_case],
                start_coefficients=start_coefficients,
                end_coefficients=end_coefficients,
                horizon=2,
            )
            for _ in range(2)
        ]
        prefix_repeat_abs = max(
            a31._maximum_abs(left - right)
            for name in ("states", "shadow_states")
            for left, right in zip(
                prefix_runs[0][name], prefix_runs[1][name], strict=True
            )
        )
        prefix_execution = {
            "first": prefix_runs[0]["execution"],
            "second": prefix_runs[1]["execution"],
            "maximum_state_abs_difference": prefix_repeat_abs,
            "excluded_from_main_scientific_cost": True,
        }

        rollouts: list[dict[str, Any]] = []
        view_records: list[dict[str, Any]] = []
        audit_rows: list[dict[str, Any]] = []
        inactive_max_abs: dict[str, float] = {}
        shadow_raw_max_abs: dict[str, float] = {}
        view_maxima = {
            "band": 0.0,
            "subspace_reconstruction": 0.0,
            "orthogonality": 0.0,
        }
        animation_payloads: dict[str, tuple[Any, Any, Any]] = {}
        for case_id in CASE_IDS:
            reference = references[case_id]
            descriptor = descriptors[case_id]
            selected = case_id in EXPECTED_ACTIVE_CASES
            print(f"A44 case {case_id}: raw comparator H30", flush=True)
            raw = a31._run_arm(
                runtime,
                case_id=case_id,
                reference=reference,
                policy="zero",
                position_selected=selected,
                descriptor=descriptor,
                start_coefficients=start_coefficients,
                end_coefficients=end_coefficients,
                horizon=30,
            )
            print(
                f"A44 case {case_id}: frozen A43 H30 selected={selected}",
                flush=True,
            )
            candidate = a43._run_candidate_arm(
                runtime,
                case_id=case_id,
                reference=reference,
                position_selected=selected,
                descriptor=descriptor,
                start_coefficients=start_coefficients,
                end_coefficients=end_coefficients,
                horizon=30,
            )
            statistics, maxima = _pair_view_statistics(
                runtime,
                case_id=case_id,
                reference=reference,
                raw_states=raw["states"],
                candidate_states=candidate["states"],
            )
            view_records.extend(statistics)
            for key, value in maxima.items():
                view_maxima[key] = max(view_maxima[key], value)
            if selected:
                shadow_raw_max_abs[case_id] = max(
                    a31._maximum_abs(left - right)
                    for left, right in zip(
                        raw["states"], candidate["shadow_states"], strict=True
                    )
                )
            else:
                inactive_max_abs[case_id] = max(
                    a31._maximum_abs(left - right)
                    for left, right in zip(
                        raw["states"], candidate["states"], strict=True
                    )
                )
            if case_id in ANIMATION_CASE_IDS:
                animation_payloads[case_id] = (
                    reference,
                    tuple(raw["states"]),
                    tuple(candidate["states"]),
                )
            audit_rows.extend(candidate["audit_rows"])
            for rollout in (raw, candidate):
                rollout.pop("states")
                rollout.pop("audit_rows")
                if rollout["policy"] == a43.POLICY:
                    rollout.pop("shadow_states")
                rollouts.append(rollout)
            if runtime.device.type == "cuda":
                torch.cuda.empty_cache()

        score_rows = _score_views(view_records)
        all_metric_rows = [metric for rollout in rollouts for metric in rollout["rows"]]
        case_rows, paired_controls, paired_populations = _aggregate_paired_controls(
            all_metric_rows
        )
        controls = [*_field_controls(score_rows), *paired_controls]
        for row in controls:
            row["cell"] = row.get("cell", "overall")
            row["policy"] = a43.POLICY

        animation_dir = output_dir / "animation_bundles"
        animation_rows = []
        for case_id in ANIMATION_CASE_IDS:
            reference, raw_states, candidate_states = animation_payloads[case_id]
            animation_rows.append(
                _write_animation_bundle(
                    animation_dir,
                    runtime=runtime,
                    case_id=case_id,
                    reference=reference,
                    raw_states=raw_states,
                    candidate_states=candidate_states,
                )
            )
        animation_manifest = with_payload_sha256(
            {
                "schema": ANIMATION_SCHEMA,
                "working_id": WORKING_ID,
                "contract": ANIMATION_CONTRACT,
                "bundles": animation_rows,
                "inference_or_gate_input": False,
            }
        )
        atomic_write_json(animation_dir / "bundle_manifest.json", animation_manifest)

        common_keys = (
            "logical_model_calls",
            "actual_forward_passes",
            "total_forward_seconds",
            "native_logical_calls",
            "fine_logical_calls",
            "native_actual_forward_passes",
            "fine_actual_forward_passes",
            "native_forward_seconds",
            "fine_forward_seconds",
            "wall_seconds",
        )
        candidate_keys = (
            *common_keys,
            "inactive_raw_native_logical_calls",
            "inactive_raw_native_actual_forward_passes",
            "inactive_raw_native_forward_seconds",
            "shadow_native_logical_calls",
            "shadow_native_actual_forward_passes",
            "shadow_native_forward_seconds",
            "candidate_native_logical_calls",
            "candidate_native_actual_forward_passes",
            "candidate_native_forward_seconds",
            "candidate_fine_logical_calls",
            "candidate_fine_actual_forward_passes",
            "candidate_fine_forward_seconds",
            "inactive_raw_steps",
            "exact_a32_steps",
            "surrogate_coast_steps",
        )
        main_execution = {
            "raw": a43._sum_execution(rollouts, "zero", common_keys),
            "candidate": a43._sum_execution(rollouts, a43.POLICY, candidate_keys),
            "maximum_peak_gpu_memory_bytes": max(
                int(row["execution"]["maximum_peak_gpu_memory_bytes"])
                for row in rollouts
            ),
            "wall_seconds": perf_counter() - started,
            "environment": runtime_environment(runtime.device),
        }
        main_execution["candidate_logical_cost_ratio_vs_raw"] = (
            main_execution["candidate"]["logical_model_calls"]
            / main_execution["raw"]["logical_model_calls"]
        )
        main_execution["candidate_forward_time_ratio_vs_raw"] = (
            main_execution["candidate"]["total_forward_seconds"]
            / main_execution["raw"]["total_forward_seconds"]
        )
        structural_checks = {
            "source_exact": preflight["source_manifest"]["source_sha256"]
            == _source_hashes(),
            "population_exact": set(selected_cases) == set(EXPECTED_ACTIVE_CASES),
            "route_audit_before_first_model_call": True,
            "descriptor_inventory_exact": len(descriptor_rows) == len(CASE_IDS),
            "native_truth_inventory_exact": len(reference_rows) == len(CASE_IDS),
            "paired_metric_inventory_exact": len(all_metric_rows)
            == 2 * len(CASE_IDS) * len(INPUT_CALLS),
            "main_logical_call_inventory_exact": (
                main_execution["raw"]["native_logical_calls"] == 420
                and main_execution["raw"]["fine_logical_calls"] == 0
                and main_execution["candidate"]["logical_model_calls"]
                == CANDIDATE_CALLS
                and main_execution["candidate"]["native_logical_calls"] == 484
                and main_execution["candidate"]["fine_logical_calls"] == 64
                and main_execution["candidate"]["inactive_raw_native_logical_calls"]
                == 300
                and main_execution["candidate"]["shadow_native_logical_calls"] == 120
                and main_execution["candidate"]["candidate_native_logical_calls"] == 64
                and main_execution["candidate"]["candidate_fine_logical_calls"] == 64
            ),
            "route_step_inventory_exact": (
                main_execution["candidate"]["inactive_raw_steps"] == 300
                and main_execution["candidate"]["exact_a32_steps"] == 64
                and main_execution["candidate"]["surrogate_coast_steps"] == 56
            ),
            "view_closure": view_maxima["band"] <= 1.0e-10
            and view_maxima["subspace_reconstruction"] <= 1.0e-10
            and view_maxima["orthogonality"] <= 1.0e-10,
            "animation_inventory_exact": len(animation_rows) == len(ANIMATION_CASE_IDS),
            "fine_truth_not_loaded": True,
            "coefficient_refit_not_run": True,
            "one_accepted_native_state_per_call": True,
        }
        gate = _gate(
            score_rows=score_rows,
            case_rows=case_rows,
            controls=controls,
            rollouts=rollouts,
            audit_rows=audit_rows,
            inactive_max_abs=inactive_max_abs,
            shadow_raw_max_abs=shadow_raw_max_abs,
            prefix_repeat_abs=prefix_repeat_abs,
            main_execution=main_execution,
            structural_checks=structural_checks,
        )

        write_csv(output_dir / "rollout_call_metrics.csv", all_metric_rows)
        write_csv(output_dir / "rollout_case_metrics.csv", case_rows)
        write_csv(output_dir / "rollout_controls.csv", controls)
        write_csv(output_dir / "view_scores.csv", score_rows)
        write_csv(output_dir / "route_audits.csv", audit_rows)
        write_csv(output_dir / "position_decisions.csv", descriptor_rows)
        write_csv(output_dir / "reference_checks.csv", reference_rows)
        write_csv(
            output_dir / "rollout_execution.csv",
            [
                {
                    "case_id": row["case_id"],
                    "policy": row["policy"],
                    "position_selected": row["position_selected"],
                    **row["execution"],
                }
                for row in rollouts
            ],
        )
        atomic_write_json(
            output_dir / "source_manifest.json", preflight["source_manifest"]
        )
        artifacts = {
            str(path.relative_to(output_dir)).replace("\\", "/"): {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in sorted(output_dir.rglob("*"))
            if path.is_file()
        }
        primary_scopes = (
            "population",
            "active_population",
            "active_group_e00",
            "active_group_e11",
        )
        result = with_payload_sha256(
            {
                "schema": RESULT_SCHEMA,
                "working_id": WORKING_ID,
                "status": (
                    "qualified_disjoint_case_confirmation"
                    if gate["status"] == "qualified"
                    else "stopped_disjoint_case_confirmation"
                ),
                "population_status": (
                    "new_disjoint_correction_confirmation_training_support"
                ),
                "gate": gate,
                "population": paired_populations,
                "primary_scores": {
                    scope: {
                        view: dict(
                            a31._score_row(
                                score_rows,
                                cell="overall",
                                view=view,
                                scope=scope,
                            )
                        )
                        for view in ("full", "rank8_parallel")
                    }
                    for scope in primary_scopes
                },
                "failed_controls": [
                    dict(row) for row in controls if not a31._control_passed(row)
                ],
                "maximum_control_ratio": max(
                    float(row["ratio"])
                    for row in controls
                    if row.get("ratio") is not None
                ),
                "inactive_case_maximum_state_abs_difference": inactive_max_abs,
                "active_shadow_raw_maximum_state_abs_difference": shadow_raw_max_abs,
                "view_closure_maxima": view_maxima,
                "main_execution": main_execution,
                "prefix_replay": prefix_execution,
                "a43_result_sha256": sha256_file(args.a43_result),
                "a43_payload_sha256": a43_payload["payload_sha256"],
                "preflight_sha256": sha256_file(args.preflight),
                "preflight_payload_sha256": preflight["payload_sha256"],
                "source_manifest_sha256": sha256_file(
                    output_dir / "source_manifest.json"
                ),
                "source_manifest_payload_sha256": preflight["source_manifest"][
                    "payload_sha256"
                ],
                "animation_bundle_manifest": animation_manifest,
                "animation_bundle_manifest_sha256": sha256_file(
                    animation_dir / "bundle_manifest.json"
                ),
                "artifact_inventory_before_summary": artifacts,
                "recurrence_executed": True,
                "animations_rendered": False,
                "fine_reference_loaded": False,
                "new_population_opened": True,
                "test_population_opened": False,
                "checkpoint_training_support_population": True,
                "claim_boundary": (
                    "Disjoint correction-confirmation trajectories inside the "
                    "checkpoint training-support split only; no independent PCNO "
                    "model validation, test/OOD, cross-family, conservation, "
                    "convergence, direct-off-grid, asymptotic-order, Richardson, "
                    "or practical-latency claim."
                ),
                "git": git_state(ROOT),
            }
        )
        atomic_write_json(
            output_dir / "phase_selective_coast_confirmation.json", result
        )
        return result, 0 if gate["status"] == "qualified" else 4
    finally:
        a43.base._close_runtime(runtime)


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
    parser.add_argument("--a43-result", type=Path, required=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("synthetic")
    preflight = commands.add_parser("preflight")
    _add_external_arguments(preflight)
    preflight.add_argument("--output", type=Path, required=True)
    rollout = commands.add_parser("rollout")
    _add_external_arguments(rollout)
    rollout.add_argument("--preflight", type=Path, required=True)
    rollout.add_argument("--output-dir", type=Path, required=True)
    rollout.add_argument("--device", choices=("cuda",), default="cuda")
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
    payload, exit_code = run_rollout(args)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
