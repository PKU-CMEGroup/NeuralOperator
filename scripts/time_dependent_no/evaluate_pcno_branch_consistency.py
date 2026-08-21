#!/usr/bin/env python3
"""Run the W26-L5 A21 truth-free branch-consistency diagnostic.

The command replays native/fine predictions from immutable A19 visualization
states. It freezes every branch decision before loading any truth array.
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
from utility.time_dependent_no.pcno_artifacts import (
    atomic_write_json_with_paths as atomic_write_json,
)
from utility.time_dependent_no.pcno_artifacts import (
    git_state,
    json_safe_with_paths,
    runtime_environment,
    sha256_file,
    sha256_files,
)
from utility.time_dependent_no.pcno_artifacts import (
    write_csv_with_paths as write_csv,
)
from utility.time_dependent_no.pcno_branch_consistency import (
    DENOMINATOR_FLOOR,
    branch_consistency_score,
    score_branch_population,
    select_branch,
    synthetic_summary,
)
from utility.time_dependent_no.pcno_cross_resolution_correction import (
    prepare_common_native_inputs,
)
from utility.time_dependent_no.pcno_cross_resolution_teacher_forced import (
    verify_payload_sha256,
    with_payload_sha256,
)
from utility.time_dependent_no.pcno_resolution_transfer import (
    conservative_admissibility_summary,
    predict_resolution_sample,
    restrict_nested_state,
    weighted_scaled_rms,
)
from utility.time_dependent_no.pcno_sparse_modal_correction import (
    FROZEN_ACTIVE_CELLS,
)

WORKING_ID = "W26-L5-P6-RFB19-A21-BRANCH-CONSISTENCY"
PREFLIGHT_SCHEMA = "pcno_branch_consistency_preflight_v1"
SOURCE_MANIFEST_SCHEMA = "pcno_branch_consistency_source_manifest_v1"
RESULT_SCHEMA = "pcno_branch_consistency_result_v1"

RESOLUTION_CONTRACT = base.RESOLUTION_CONTRACT
NATIVE_RESOLUTION = RESOLUTION_CONTRACT.native
FINE_RESOLUTION = RESOLUTION_CONTRACT.fine
CASE_IDS = ("sv_e12_y00", "sv_e12_y01", "sv_e12_y04", "sv_e12_y08")
OUTPUT_CALLS = tuple(range(0, 31, 2))
TRANSITION_CALLS = OUTPUT_CALLS[:-1]
EXPECTED_LOGICAL_CALLS = 360
EXPECTED_NATIVE_CALLS = 240
EXPECTED_FINE_CALLS = 120
SOURCE_PATHS = (
    "docs/time_dependent_no/W26_L5_RESPONSE_FILTERED_BLOCK_PREREGISTRATION.md",
    "utility/time_dependent_no/pcno_branch_consistency.py",
    "scripts/time_dependent_no/evaluate_pcno_branch_consistency.py",
    "tests/time_dependent_no/test_pcno_branch_consistency.py",
)

A19_RESULT_SHA256 = "c6675dbb5481aea31a16fbd6aef8b7a71d7403a0fb199cd0120ff2cf343b868f"
A19_RESULT_PAYLOAD_SHA256 = "bbe9abffbf1b4b699499eab72add357acf4ee6955a0fae9f1603579f0cd7f5dd"
A19_SOURCE_SHA256 = "6be32d4f6ead4578b357847ffd0912c63a8738eec68c3bd53446b473c1f062c2"
A19_SOURCE_PAYLOAD_SHA256 = "194abb45a3588a5dd1319797106ae961f94ff8494538a7d4de75fa1f91cca7c4"
BUNDLE_MANIFEST_SHA256 = "b7bd0cf7f60a81b2676752e59adc6fed4c0cff77ca28774979d96b7583008e6c"
BUNDLE_MANIFEST_PAYLOAD_SHA256 = "1816271fdb7da13d3eed35654d83dd7c86524d312f38f28d568737f1ac328a17"
EXPECTED_BUNDLES = {
    "sv_e12_y00": (
        "sv_e12_y00_shadow_anchored_h30.npz",
        "eec6ccd7cd80d9499e184c2d4a01c75f7b176a992eb2b02aa168f73afa047b8d",
    ),
    "sv_e12_y01": (
        "sv_e12_y01_shadow_anchored_h30.npz",
        "f8aff117c082ce9cd7dc5999a46df3c8fe9430b2fcc988cb4c375ffe8e9c4079",
    ),
    "sv_e12_y04": (
        "sv_e12_y04_shadow_anchored_h30.npz",
        "e58b95b276fdf44eda2a27dfd4729382ac2b02a919f3c725277cc8c4884de4e4",
    ),
    "sv_e12_y08": (
        "sv_e12_y08_shadow_anchored_h30.npz",
        "501da1fba73ae879d4ac6f0b595fa0da58890da77141488675dcb1390ee1b691",
    ),
}
NPZ_KEYS = {
    "case_id",
    "split_group_id",
    "output_calls",
    "physical_times",
    "native_resolution",
    "nodes",
    "volumes",
    "state_scale",
    "residual_scale",
    "gamma",
    "truth_conservative",
    "raw_shadow_conservative",
    "corrected_conservative",
}
COMPONENT_NAMES = ("density", "x_momentum", "y_momentum", "energy")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def _source_hashes() -> dict[str, str]:
    return sha256_files(SOURCE_PATHS, root=ROOT)


def _git_status_short(paths: Sequence[str]) -> list[str]:
    result = subprocess.run(
        ["git", "status", "--short", "--", *paths],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return [line for line in result.stdout.splitlines() if line]


def _validate_parent_inputs(
    a19_result_path: Path,
    a19_source_path: Path,
    bundle_manifest_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, str]]:
    if sha256_file(a19_result_path) != A19_RESULT_SHA256:
        raise ValueError("A19 result file hash mismatch")
    if sha256_file(a19_source_path) != A19_SOURCE_SHA256:
        raise ValueError("A19 source-manifest file hash mismatch")
    if sha256_file(bundle_manifest_path) != BUNDLE_MANIFEST_SHA256:
        raise ValueError("A19 bundle-manifest file hash mismatch")
    result = _read_json(a19_result_path)
    source = _read_json(a19_source_path)
    manifest = _read_json(bundle_manifest_path)
    for payload in (result, source, manifest):
        verify_payload_sha256(payload)
    if (
        result.get("schema")
        != "pcno_response_filtered_block_relaxed_tether_rollout_v1"
        or result.get("working_id") != "W26-L5-P6-RFB19-A19-RELAX-TETHER-E12"
        or result.get("payload_sha256") != A19_RESULT_PAYLOAD_SHA256
        or result.get("source_manifest_sha256") != A19_SOURCE_SHA256
        or result.get("source_manifest_payload_sha256") != A19_SOURCE_PAYLOAD_SHA256
        or result.get("animation_bundle_manifest_sha256")
        != BUNDLE_MANIFEST_SHA256
    ):
        raise ValueError("A19 result identity mismatch")
    if source.get("payload_sha256") != A19_SOURCE_PAYLOAD_SHA256:
        raise ValueError("A19 source payload mismatch")
    if (
        manifest.get("schema")
        != "pcno_response_filtered_block_animation_bundles_v1"
        or manifest.get("payload_sha256") != BUNDLE_MANIFEST_PAYLOAD_SHA256
        or manifest.get("inference_or_gate_input") is not False
        or result.get("animation_bundle_manifest") != manifest
    ):
        raise ValueError("A19 bundle manifest contract mismatch")
    contract = manifest.get("contract", {})
    if (
        tuple(contract.get("case_ids", ())) != CASE_IDS
        or tuple(contract.get("output_calls", ())) != OUTPUT_CALLS
        or tuple(contract.get("native_resolution", ())) != NATIVE_RESOLUTION
        or contract.get("visualization_only_frame_stride") != 2
    ):
        raise ValueError("A19 bundle population contract mismatch")
    records = manifest.get("bundles")
    if not isinstance(records, list) or len(records) != len(CASE_IDS):
        raise ValueError("A19 bundle inventory mismatch")
    actual = {
        str(record.get("case_id")): (record.get("path"), record.get("sha256"))
        for record in records
        if isinstance(record, Mapping)
    }
    if actual != EXPECTED_BUNDLES:
        raise ValueError("A19 bundle records differ from the frozen inventory")
    bundle_hashes = {}
    for case_id, (name, expected_hash) in EXPECTED_BUNDLES.items():
        path = bundle_manifest_path.parent / name
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise ValueError(f"A19 bundle hash mismatch: {case_id}")
        bundle_hashes[name] = actual_hash
    return result, source, manifest, bundle_hashes


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    result, source, manifest, bundle_hashes = _validate_parent_inputs(
        args.a19_result,
        args.a19_source_manifest,
        args.bundle_manifest,
    )
    source_hashes = _source_hashes()
    synthetic = synthetic_summary()
    checks = {
        "a19_result_identity_exact": result["payload_sha256"]
        == A19_RESULT_PAYLOAD_SHA256,
        "a19_source_identity_exact": source["payload_sha256"]
        == A19_SOURCE_PAYLOAD_SHA256,
        "bundle_manifest_identity_exact": manifest["payload_sha256"]
        == BUNDLE_MANIFEST_PAYLOAD_SHA256,
        "bundle_inventory_exact": set(bundle_hashes)
        == {value[0] for value in EXPECTED_BUNDLES.values()},
        "source_inventory_exact": set(source_hashes) == set(SOURCE_PATHS),
        "focused_cpu_tests_passed": args.focused_test_result.startswith("passed"),
        "synthetic_closure_passed": synthetic["passed"] is True,
        "checkpoint_not_built": True,
        "truth_not_loaded": True,
        "recurrence_not_executed": True,
    }
    payload = with_payload_sha256(
        {
            "schema": PREFLIGHT_SCHEMA,
            "working_id": WORKING_ID,
            "status": "passed" if all(checks.values()) else "failed",
            "checks": checks,
            "source_sha256": source_hashes,
            "relevant_status_short": _git_status_short(SOURCE_PATHS),
            "git": git_state(ROOT),
            "a19_result_sha256": A19_RESULT_SHA256,
            "a19_result_payload_sha256": A19_RESULT_PAYLOAD_SHA256,
            "a19_source_manifest_sha256": A19_SOURCE_SHA256,
            "a19_source_manifest_payload_sha256": A19_SOURCE_PAYLOAD_SHA256,
            "bundle_manifest_sha256": BUNDLE_MANIFEST_SHA256,
            "bundle_manifest_payload_sha256": BUNDLE_MANIFEST_PAYLOAD_SHA256,
            "bundle_sha256": bundle_hashes,
            "focused_test_command": args.focused_test_command,
            "focused_test_result": args.focused_test_result,
            "synthetic": synthetic,
            "authorization": {
                "retrospective_model_replay": True,
                "new_recurrence": False,
                "training": False,
                "sealed_population": False,
                "bump": False,
                "reference_generation": False,
            },
        }
    )
    atomic_write_json(args.output, payload)
    return payload


def _verify_preflight(args: argparse.Namespace) -> dict[str, Any]:
    payload = _read_json(args.preflight)
    verify_payload_sha256(payload)
    if (
        payload.get("schema") != PREFLIGHT_SCHEMA
        or payload.get("working_id") != WORKING_ID
        or payload.get("status") != "passed"
        or payload.get("source_sha256") != _source_hashes()
        or sha256_file(args.a19_result) != payload.get("a19_result_sha256")
        or sha256_file(args.a19_source_manifest)
        != payload.get("a19_source_manifest_sha256")
        or sha256_file(args.bundle_manifest)
        != payload.get("bundle_manifest_sha256")
    ):
        raise ValueError("A21 preflight identity mismatch")
    checks = payload.get("checks")
    if not isinstance(checks, Mapping) or not all(value is True for value in checks.values()):
        raise ValueError("A21 preflight checks are incomplete")
    _validate_parent_inputs(
        args.a19_result,
        args.a19_source_manifest,
        args.bundle_manifest,
    )
    return payload


def _load_branch_inputs(bundle_manifest_path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for case_id in CASE_IDS:
        name, _ = EXPECTED_BUNDLES[case_id]
        path = bundle_manifest_path.parent / name
        with np.load(path, allow_pickle=False) as data:
            if set(data.files) != NPZ_KEYS:
                raise ValueError(f"unexpected bundle keys: {case_id}")
            metadata = {
                "path": path,
                "case_id": str(data["case_id"].item()),
                "split_group_id": str(data["split_group_id"].item()),
                "output_calls": np.array(data["output_calls"], copy=True),
                "physical_times": np.array(data["physical_times"], copy=True),
                "native_resolution": np.array(data["native_resolution"], copy=True),
                "nodes": np.array(data["nodes"], copy=True),
                "volumes": np.array(data["volumes"], copy=True),
                "state_scale": np.array(data["state_scale"], copy=True),
                "residual_scale": np.array(data["residual_scale"], copy=True),
                "gamma": float(data["gamma"].item()),
                "raw": np.array(data["raw_shadow_conservative"], copy=True),
                "corrected": np.array(data["corrected_conservative"], copy=True),
            }
        if (
            metadata["case_id"] != case_id
            or metadata["split_group_id"] != "strength_ood_e12"
            or not np.array_equal(metadata["output_calls"], np.asarray(OUTPUT_CALLS))
            or not np.allclose(
                metadata["physical_times"],
                0.04 * np.arange(len(OUTPUT_CALLS)),
                atol=0.0,
                rtol=0.0,
            )
            or not np.array_equal(metadata["native_resolution"], NATIVE_RESOLUTION)
            or metadata["raw"].shape != (16, 25000, 4)
            or metadata["corrected"].shape != (16, 25000, 4)
            or metadata["raw"].dtype != np.float32
            or metadata["corrected"].dtype != np.float32
            or not np.isfinite(metadata["raw"]).all()
            or not np.isfinite(metadata["corrected"]).all()
            or not np.array_equal(metadata["raw"][0], metadata["corrected"][0])
        ):
            raise ValueError(f"bundle state contract mismatch: {case_id}")
        result[case_id] = metadata
    first = result[CASE_IDS[0]]
    for case_id in CASE_IDS[1:]:
        for key in ("nodes", "volumes", "state_scale", "residual_scale"):
            if not np.array_equal(result[case_id][key], first[key]):
                raise ValueError(f"bundle geometry/scale mismatch: {case_id}/{key}")
        if result[case_id]["gamma"] != first["gamma"]:
            raise ValueError(f"bundle gamma mismatch: {case_id}")
    return result


def _validate_runtime_bundle(
    runtime: Any, bundles: Mapping[str, Mapping[str, Any]]
) -> dict[str, dict[str, float]]:
    bundle = bundles[CASE_IDS[0]]
    geometry = runtime.geometry_by_resolution[NATIVE_RESOLUTION]
    comparisons = (
        (bundle["nodes"], np.asarray(geometry.nodes, dtype=np.float64), "nodes"),
        (
            bundle["volumes"],
            np.asarray(geometry.node_measures, dtype=np.float64).reshape(-1),
            "volumes",
        ),
        (
            bundle["state_scale"],
            np.asarray(runtime.normalization.state_scale, dtype=np.float64),
            "state_scale",
        ),
        (
            bundle["residual_scale"],
            np.asarray(runtime.normalization.residual_scale, dtype=np.float64),
            "residual_scale",
        ),
    )
    floors: dict[str, dict[str, float]] = {}
    for left, right, name in comparisons:
        bundle_value = np.asarray(left, dtype=np.float64)
        runtime_value = np.asarray(right, dtype=np.float64)
        if bundle_value.shape != runtime_value.shape:
            raise ValueError(f"runtime/bundle {name} shape mismatch")
        difference = bundle_value - runtime_value
        scale = max(float(np.max(np.abs(runtime_value))), np.finfo(np.float64).tiny)
        maximum_absolute = float(np.max(np.abs(difference)))
        tolerance = float(np.finfo(np.float32).eps * scale)
        floors[name] = {
            "maximum_absolute": maximum_absolute,
            "rms": float(np.sqrt(np.mean(np.square(difference)))),
            "relative_to_maximum_runtime_absolute": maximum_absolute / scale,
            "one_float32_epsilon_tolerance": tolerance,
        }
        if maximum_absolute > tolerance:
            raise ValueError(f"runtime/bundle {name} exceeds its float32 floor")
    if bundle["gamma"] != float(runtime.normalization.gamma):
        raise ValueError("runtime/bundle gamma mismatch")
    return floors


def _physical_integral(value: np.ndarray, volumes: np.ndarray) -> np.ndarray:
    return np.einsum(
        "n,nc->c",
        np.asarray(volumes, dtype=np.float64),
        np.asarray(value, dtype=np.float64),
        optimize=True,
    )


def _account(execution: dict[str, Any], timing: Mapping[str, Any], resolution: str) -> None:
    execution["logical_model_calls"] += 1
    execution[f"{resolution}_logical_calls"] += 1
    forwards = timing["forward_seconds"]
    execution["actual_forward_passes_including_repeats"] += len(forwards)
    execution["total_forward_seconds"] += float(sum(forwards))
    repeat = float(timing["repeat_max_abs"])
    if math.isfinite(repeat):
        execution["maximum_repeat_abs_difference"] = max(
            execution["maximum_repeat_abs_difference"], repeat
        )
    peak = timing.get("peak_gpu_memory_bytes")
    if peak is not None:
        execution["maximum_peak_gpu_memory_bytes"] = max(
            execution["maximum_peak_gpu_memory_bytes"], int(peak)
        )


def _branch_replay(
    runtime: Any,
    state: np.ndarray,
    execution: dict[str, Any],
) -> dict[str, Any]:
    prepared = prepare_common_native_inputs(state, contract=RESOLUTION_CONTRACT)
    native_current = prepared.model_inputs[NATIVE_RESOLUTION]
    fine_current = prepared.model_inputs[FINE_RESOLUTION]
    native_first, native_timing = predict_resolution_sample(
        runtime.model,
        runtime.sample_by_resolution[NATIVE_RESOLUTION],
        native_current,
        device=runtime.device,
        amp="none",
        repeats=1,
    )
    _account(execution, native_timing, "native")
    fine_first, fine_timing = predict_resolution_sample(
        runtime.model,
        runtime.sample_by_resolution[FINE_RESOLUTION],
        fine_current,
        device=runtime.device,
        amp="none",
        repeats=1,
    )
    _account(execution, fine_timing, "fine")
    native_second, second_timing = predict_resolution_sample(
        runtime.model,
        runtime.sample_by_resolution[NATIVE_RESOLUTION],
        native_first,
        device=runtime.device,
        amp="none",
        repeats=1,
    )
    _account(execution, second_timing, "native")

    native_increment = native_first - native_current
    fine_increment = fine_first - fine_current
    fine_on_native = restrict_nested_state(
        fine_increment,
        fine_resolution=FINE_RESOLUTION,
        coarse_resolution=NATIVE_RESOLUTION,
    )
    score = branch_consistency_score(
        native_increment,
        fine_on_native,
        runtime.native_projector,
        volumes=runtime.geometry_by_resolution[NATIVE_RESOLUTION].node_measures,
        component_scale=runtime.normalization.residual_scale,
    )
    admissibility = [
        conservative_admissibility_summary(value, gamma=runtime.normalization.gamma)
        for value in (native_first, fine_first, native_second)
    ]
    return {
        "score": score,
        "lookahead": native_second,
        "nesting_floors": dict(prepared.nesting_floors),
        "all_predictions_finite": all(item["finite"] for item in admissibility),
        "all_predictions_admissible": all(item["admissible"] for item in admissibility),
        "minimum_density": min(item["minimum_density"] for item in admissibility),
        "minimum_pressure": min(item["minimum_pressure"] for item in admissibility),
    }


def _error_metrics(
    value: np.ndarray,
    truth: np.ndarray,
    *,
    runtime: Any,
    state_scale: np.ndarray,
    volumes: np.ndarray,
) -> dict[str, Any]:
    error = np.asarray(value, dtype=np.float64) - np.asarray(truth, dtype=np.float64)
    boundary_mask = ~runtime.native_projector.interior_mask
    integral = _physical_integral(error, volumes)
    front = base._front_errors(value, truth, runtime)
    admissibility = conservative_admissibility_summary(
        value,
        gamma=runtime.normalization.gamma,
    )
    result: dict[str, Any] = {
        "rms": weighted_scaled_rms(
            error,
            volumes=volumes,
            component_scale=state_scale,
        ),
        "boundary_rms": weighted_scaled_rms(
            error,
            volumes=volumes,
            component_scale=state_scale,
            mask=boundary_mask,
        ),
        "finite": admissibility["finite"],
        "admissible": admissibility["admissible"],
        "minimum_density": admissibility["minimum_density"],
        "minimum_pressure": admissibility["minimum_pressure"],
        **front,
    }
    for index, name in enumerate(COMPONENT_NAMES):
        result[f"integral_{name}"] = float(integral[index])
    return result


def _append_metrics(row: dict[str, Any], prefix: str, metrics: Mapping[str, Any]) -> None:
    row.update({f"{prefix}_{key}": value for key, value in metrics.items()})


def _distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    if finite.size == 0:
        return {"count": 0, "minimum": None, "median": None, "maximum": None}
    return {
        "count": int(finite.size),
        "minimum": float(finite.min()),
        "median": float(np.median(finite)),
        "maximum": float(finite.max()),
    }


def _source_manifest(
    runtime: Any,
    preflight: Mapping[str, Any],
    args: argparse.Namespace,
    serialization_floors: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    return with_payload_sha256(
        {
            "schema": SOURCE_MANIFEST_SCHEMA,
            "working_id": WORKING_ID,
            "source_sha256": _source_hashes(),
            "preflight_sha256": sha256_file(args.preflight),
            "preflight_payload_sha256": preflight["payload_sha256"],
            "a19_result_sha256": A19_RESULT_SHA256,
            "a19_result_payload_sha256": A19_RESULT_PAYLOAD_SHA256,
            "a19_source_manifest_sha256": A19_SOURCE_SHA256,
            "bundle_manifest_sha256": BUNDLE_MANIFEST_SHA256,
            "bundle_sha256": preflight["bundle_sha256"],
            "bundle_runtime_float32_serialization_floors": serialization_floors,
            "base_runtime_source_manifest": runtime.source_manifest,
            "population": {
                "case_ids": list(CASE_IDS),
                "transition_input_calls": list(TRANSITION_CALLS),
                "next_output_calls": list(OUTPUT_CALLS[1:]),
                "serialization": "float32_visualization_only",
                "truth_use": "retrospective_scoring_after_all_decisions_only",
            },
            "score_contract": {
                "active_cells": [list(cell) for cell in FROZEN_ACTIVE_CELLS],
                "denominator_floor": DENOMINATOR_FLOOR,
                "tie_absolute_tolerance": 1.0e-12,
                "tie_relative_tolerance": 1.0e-12,
                "abstention_fallback": "raw",
                "coefficient_fit": False,
                "threshold_fit": False,
            },
            "call_contract": {
                "native_per_branch_transition": 2,
                "fine_per_branch_transition": 1,
                "branches": 2,
                "transitions": len(CASE_IDS) * len(TRANSITION_CALLS),
                "logical_model_calls": EXPECTED_LOGICAL_CALLS,
            },
            "git": git_state(ROOT),
            "relevant_status_short": _git_status_short(SOURCE_PATHS),
        }
    )


def run_evaluation(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    preflight = _verify_preflight(args)
    bundles = _load_branch_inputs(args.bundle_manifest)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    runtime = base._build_runtime(args)
    started = perf_counter()
    try:
        serialization_floors = _validate_runtime_bundle(runtime, bundles)
        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=False)
        execution: dict[str, Any] = {
            "logical_model_calls": 0,
            "native_logical_calls": 0,
            "fine_logical_calls": 0,
            "actual_forward_passes_including_repeats": 0,
            "total_forward_seconds": 0.0,
            "maximum_repeat_abs_difference": 0.0,
            "maximum_peak_gpu_memory_bytes": 0,
        }
        if runtime.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(runtime.device)

        # Phase one has no truth access. It freezes all 60 branch decisions.
        cached_rows: list[dict[str, Any]] = []
        maximum_pre_model_floor = 0.0
        maximum_post_model_floor = 0.0
        all_predictions_finite = True
        all_predictions_admissible = True
        for case_id in CASE_IDS:
            bundle = bundles[case_id]
            for frame_index, input_call in enumerate(TRANSITION_CALLS):
                raw_state = np.asarray(bundle["raw"][frame_index], dtype=np.float64)
                corrected_state = np.asarray(
                    bundle["corrected"][frame_index], dtype=np.float64
                )
                raw_replay = _branch_replay(runtime, raw_state, execution)
                corrected_replay = _branch_replay(runtime, corrected_state, execution)
                decision = select_branch(raw_replay["score"], corrected_replay["score"])
                for replay in (raw_replay, corrected_replay):
                    floors = replay["nesting_floors"]
                    maximum_pre_model_floor = max(
                        maximum_pre_model_floor,
                        float(floors["pre_model_coarse_from_native_max_abs"]),
                        float(floors["pre_model_fine_to_native_max_abs"]),
                    )
                    maximum_post_model_floor = max(
                        maximum_post_model_floor,
                        float(floors["post_fp32_coarse_from_native_max_abs"]),
                        float(floors["post_fp32_fine_to_native_max_abs"]),
                    )
                    all_predictions_finite &= bool(replay["all_predictions_finite"])
                    all_predictions_admissible &= bool(
                        replay["all_predictions_admissible"]
                    )
                cached_rows.append(
                    {
                        "case_id": case_id,
                        "input_call": int(input_call),
                        "next_output_call": int(OUTPUT_CALLS[frame_index + 1]),
                        "physical_time": float(bundle["physical_times"][frame_index]),
                        "next_physical_time": float(
                            bundle["physical_times"][frame_index + 1]
                        ),
                        "raw_state": raw_state,
                        "corrected_state": corrected_state,
                        "raw_lookahead": raw_replay["lookahead"],
                        "corrected_lookahead": corrected_replay["lookahead"],
                        "raw_replay": raw_replay,
                        "corrected_replay": corrected_replay,
                        "decision": decision,
                    }
                )

        if execution["logical_model_calls"] != EXPECTED_LOGICAL_CALLS:
            raise RuntimeError("A21 logical call inventory changed")
        decisions_frozen_at = perf_counter()

        # Phase two begins only after every decision above is immutable.
        truth_by_case: dict[str, np.ndarray] = {}
        for case_id in CASE_IDS:
            with np.load(bundles[case_id]["path"], allow_pickle=False) as data:
                truth = np.array(data["truth_conservative"], copy=True)
            if (
                truth.shape != (16, 25000, 4)
                or truth.dtype != np.float32
                or not np.isfinite(truth).all()
            ):
                raise ValueError(f"truth bundle contract mismatch: {case_id}")
            truth_by_case[case_id] = truth

        volumes = np.asarray(
            runtime.geometry_by_resolution[NATIVE_RESOLUTION].node_measures,
            dtype=np.float64,
        ).reshape(-1)
        state_scale = np.asarray(runtime.normalization.state_scale, dtype=np.float64)
        rows: list[dict[str, Any]] = []
        for cached in cached_rows:
            case_id = cached["case_id"]
            frame_index = int(cached["input_call"]) // 2
            current_truth = truth_by_case[case_id][frame_index]
            lookahead_truth = truth_by_case[case_id][frame_index + 1]
            decision = cached["decision"]
            selected_branch = decision.selected_branch
            current_values = {
                "raw": cached["raw_state"],
                "corrected": cached["corrected_state"],
            }
            lookahead_values = {
                "raw": cached["raw_lookahead"],
                "corrected": cached["corrected_lookahead"],
            }
            current_metrics = {
                name: _error_metrics(
                    value,
                    current_truth,
                    runtime=runtime,
                    state_scale=state_scale,
                    volumes=volumes,
                )
                for name, value in current_values.items()
            }
            lookahead_metrics = {
                name: _error_metrics(
                    value,
                    lookahead_truth,
                    runtime=runtime,
                    state_scale=state_scale,
                    volumes=volumes,
                )
                for name, value in lookahead_values.items()
            }
            current_oracle = min(
                ("raw", "corrected"), key=lambda name: current_metrics[name]["rms"]
            )
            lookahead_oracle = min(
                ("raw", "corrected"), key=lambda name: lookahead_metrics[name]["rms"]
            )
            current_metrics["selected"] = current_metrics[selected_branch]
            current_metrics["oracle"] = current_metrics[current_oracle]
            lookahead_metrics["selected"] = lookahead_metrics[selected_branch]
            lookahead_metrics["oracle"] = lookahead_metrics[lookahead_oracle]
            raw_score = cached["raw_replay"]["score"]
            corrected_score = cached["corrected_replay"]["score"]
            row: dict[str, Any] = {
                "case_id": case_id,
                "input_call": cached["input_call"],
                "next_output_call": cached["next_output_call"],
                "physical_time": cached["physical_time"],
                "next_physical_time": cached["next_physical_time"],
                "selected_branch": selected_branch,
                "selector_status": decision.status,
                "selector_resolved": decision.resolved,
                "score_difference_corrected_minus_raw": decision.score_difference_corrected_minus_raw,
                "current_oracle_branch": current_oracle,
                "lookahead_oracle_branch": lookahead_oracle,
                **{f"raw_{key}": value for key, value in asdict(raw_score).items()},
                **{
                    f"corrected_{key}": value
                    for key, value in asdict(corrected_score).items()
                },
            }
            for horizon, metrics_by_name in (
                ("current", current_metrics),
                ("lookahead", lookahead_metrics),
            ):
                for name, metrics in metrics_by_name.items():
                    _append_metrics(row, f"{horizon}_{name}", metrics)
            rows.append(row)

        truth_scoring_seconds = perf_counter() - decisions_frozen_at
        current_score = score_branch_population(rows, error_prefix="current")
        lookahead_score = score_branch_population(rows, error_prefix="lookahead")
        case_rows = [
            {"horizon": horizon, **case}
            for horizon, score in (
                ("current", current_score),
                ("lookahead", lookahead_score),
            )
            for case in score["case_scores"]
        ]
        checks = {
            "source_identity_exact": preflight["source_sha256"] == _source_hashes(),
            "bundle_inventory_exact": len(bundles) == len(CASE_IDS),
            "bundle_runtime_float32_floor_bounded": all(
                row["maximum_absolute"] <= row["one_float32_epsilon_tolerance"]
                for row in serialization_floors.values()
            ),
            "transition_inventory_exact": len(rows)
            == len(CASE_IDS) * len(TRANSITION_CALLS),
            "truth_loaded_after_all_decisions": decisions_frozen_at > started,
            "selector_truth_input_absent": True,
            "no_recurrent_state_committed": True,
            "pre_model_nesting_exact": maximum_pre_model_floor == 0.0,
            "post_fp32_nesting_finite": np.isfinite(maximum_post_model_floor),
            "all_predictions_finite": all_predictions_finite,
            "all_predictions_admissible": all_predictions_admissible,
            "all_scored_states_finite": all(
                bool(row[f"{horizon}_{name}_finite"])
                for row in rows
                for horizon in ("current", "lookahead")
                for name in ("raw", "corrected", "selected")
            ),
            "all_scored_states_admissible": all(
                bool(row[f"{horizon}_{name}_admissible"])
                for row in rows
                for horizon in ("current", "lookahead")
                for name in ("raw", "corrected", "selected")
            ),
            "logical_calls_exact": execution["logical_model_calls"]
            == EXPECTED_LOGICAL_CALLS,
            "native_calls_exact": execution["native_logical_calls"]
            == EXPECTED_NATIVE_CALLS,
            "fine_calls_exact": execution["fine_logical_calls"]
            == EXPECTED_FINE_CALLS,
            "repeat_difference_zero": execution["maximum_repeat_abs_difference"]
            == 0.0,
            "call_zero_duplicate_tie_closure": all(
                row["selector_status"] == "tie_abstain_raw"
                and not row["selector_resolved"]
                and row["score_difference_corrected_minus_raw"] is not None
                and abs(row["score_difference_corrected_minus_raw"]) <= 1.0e-12
                for row in rows
                if row["input_call"] == 0
            ),
        }
        relation = lookahead_score["score_error_relation"]
        prospective_gate = {
            "all_validity_checks_pass": all(checks.values()),
            "at_least_48_comparisons_resolve": lookahead_score["resolved_count"] >= 48,
            "current_selected_beats_raw": (
                current_score["selected_to_raw_rms_ratio"] is not None
                and current_score["selected_to_raw_rms_ratio"] < 1.0
            ),
            "current_selected_beats_corrected": (
                current_score["selected_to_corrected_rms_ratio"] is not None
                and current_score["selected_to_corrected_rms_ratio"] < 1.0
            ),
            "lookahead_selected_beats_raw": (
                lookahead_score["selected_to_raw_rms_ratio"] is not None
                and lookahead_score["selected_to_raw_rms_ratio"] < 1.0
            ),
            "lookahead_selected_beats_corrected": (
                lookahead_score["selected_to_corrected_rms_ratio"] is not None
                and lookahead_score["selected_to_corrected_rms_ratio"] < 1.0
            ),
            "lookahead_accuracy_above_half": (
                lookahead_score["selector_accuracy"] is not None
                and lookahead_score["selector_accuracy"] > 0.5
            ),
            "lookahead_sign_agreement_above_half": (
                relation["sign_agreement"] is not None
                and relation["sign_agreement"] > 0.5
            ),
            "lookahead_signed_cosine_positive": (
                relation["cosine_status"] == "ok" and relation["cosine"] > 0.0
            ),
            "casewise_lookahead_no_harm_0p1_percent": lookahead_score[
                "maximum_selected_to_raw_case_rms_ratio"
            ]
            <= 1.001,
        }
        gate_passed = all(prospective_gate.values())
        execution["wall_seconds"] = perf_counter() - started
        execution["truth_scoring_seconds"] = truth_scoring_seconds
        execution["device"] = str(runtime.device)
        execution["amp"] = "none"
        execution["repeat_forward"] = 1
        execution["runtime_environment"] = runtime_environment(runtime.device)
        source_manifest = _source_manifest(
            runtime,
            preflight,
            args,
            serialization_floors,
        )
        source_path = output_dir / "source_manifest.json"
        rows_path = output_dir / "branch_rows.csv"
        cases_path = output_dir / "case_metrics.csv"
        atomic_write_json(source_path, source_manifest)
        write_csv(rows_path, rows)
        write_csv(cases_path, case_rows)
        artifact_hashes = {
            path.name: sha256_file(path) for path in (source_path, rows_path, cases_path)
        }
        raw_scores = [
            row["raw_score"] for row in rows if row["raw_score"] is not None
        ]
        corrected_scores = [
            row["corrected_score"]
            for row in rows
            if row["corrected_score"] is not None
        ]
        score_distributions = {
            "raw_score": _distribution(raw_scores),
            "corrected_score": _distribution(corrected_scores),
            "raw_commutator_rms": _distribution(
                [row["raw_commutator_rms"] for row in rows]
            ),
            "corrected_commutator_rms": _distribution(
                [row["corrected_commutator_rms"] for row in rows]
            ),
            "raw_native_increment_rms": _distribution(
                [row["raw_native_increment_rms"] for row in rows]
            ),
            "corrected_native_increment_rms": _distribution(
                [row["corrected_native_increment_rms"] for row in rows]
            ),
        }
        payload = with_payload_sha256(
            json_safe_with_paths(
                {
                "schema": RESULT_SCHEMA,
                "working_id": WORKING_ID,
                "status": (
                    "qualified_retrospective_selector"
                    if gate_passed
                    else "stopped_retrospective_selector"
                ),
                "population_status": "already_open_e12_visualization_replay",
                "checks": checks,
                "prospective_gate": prospective_gate,
                "current_population": current_score,
                "lookahead_population": lookahead_score,
                "score_distributions": score_distributions,
                "transfer_floor_maxima": {
                    "pre_model": maximum_pre_model_floor,
                    "post_fp32": maximum_post_model_floor,
                },
                "bundle_runtime_float32_serialization_floors": serialization_floors,
                "execution": execution,
                "preflight_sha256": sha256_file(args.preflight),
                "preflight_payload_sha256": preflight["payload_sha256"],
                "source_manifest_sha256": artifact_hashes["source_manifest.json"],
                "source_manifest_payload_sha256": source_manifest["payload_sha256"],
                "a19_result_sha256": A19_RESULT_SHA256,
                "bundle_manifest_sha256": BUNDLE_MANIFEST_SHA256,
                "artifact_hashes": artifact_hashes,
                "claim_boundary": (
                    "Retrospective truth-free ranking diagnostic on four float32 A19 "
                    "visualization trajectories. It is not exact replay, prospective "
                    "validation, family transfer, conservation, convergence, off-grid "
                    "superiority, or Richardson extrapolation."
                ),
                }
            )
        )
        atomic_write_json(output_dir / "branch_consistency.json", payload)
        return payload, 0 if gate_passed else 4
    finally:
        base._close_runtime(runtime)


def _add_parent_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--a19-result", type=Path, required=True)
    parser.add_argument("--a19-source-manifest", type=Path, required=True)
    parser.add_argument("--bundle-manifest", type=Path, required=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("synthetic")
    preflight = commands.add_parser("preflight")
    _add_parent_arguments(preflight)
    preflight.add_argument("--output", type=Path, required=True)
    preflight.add_argument("--focused-test-command", required=True)
    preflight.add_argument("--focused-test-result", required=True)
    evaluation = commands.add_parser("evaluate")
    _add_parent_arguments(evaluation)
    base._add_external_arguments(evaluation)
    evaluation.add_argument("--preflight", type=Path, required=True)
    evaluation.add_argument("--output-dir", type=Path, required=True)
    evaluation.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    evaluation.add_argument("--allow-cpu", action="store_true")
    evaluation.add_argument("--repeat-forward", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "synthetic":
        payload = synthetic_summary()
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if payload["passed"] else 2
    if args.command == "preflight":
        payload = run_preflight(args)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if payload["status"] == "passed" else 2
    if args.repeat_forward != 1:
        raise ValueError("A21 freezes exactly one forward pass per logical call")
    payload, exit_code = run_evaluation(args)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
