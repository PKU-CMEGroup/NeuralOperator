"""Benchmark exact paired-native PCNO inference for the qualified A32 tether."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.time_dependent_no import (
    evaluate_pcno_affine_shadow_tether_rollout as a32,
)
from scripts.time_dependent_no import (
    evaluate_pcno_cross_resolution_teacher_forced as base,
)
from scripts.time_dependent_no import evaluate_pcno_modal_affine_transfer as a29
from scripts.time_dependent_no import (
    visualize_pcno_affine_shadow_tether_rollout as a32_visualize,
)
from utility.time_dependent_no.pcno_artifacts import (
    atomic_write_json_with_paths as atomic_write_json,
)
from utility.time_dependent_no.pcno_artifacts import (
    runtime_environment,
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
from utility.time_dependent_no.pcno_resolution_transfer import (
    predict_resolution_batch,
    predict_resolution_sample,
)

WORKING_ID = "W26-L5-P6-RFB19-A40-PAIRED-NATIVE-BATCH"
PREFLIGHT_SCHEMA = "pcno_paired_native_batch_preflight_v1"
RESULT_SCHEMA = "pcno_paired_native_batch_benchmark_v1"
SOURCE_SCHEMA = "pcno_paired_native_batch_source_v1"

A32_RESULT_SHA256 = "58eb364fc2e1dfd8ea00a3e4312ae4ace4c03dd9234bfb92fc48da737180cc0f"
A32_PAYLOAD_SHA256 = "208ac069aee27d021e2506377dbd586c072c9a64353d866dc8e296b71b123e89"
A32_BUNDLE_MANIFEST_SHA256 = (
    "6b56bd44a2fa538a3807b72d7cb5986ec1114ad500b06ab935b2081e6e24e12e"
)
NATIVE_RESOLUTION = (250, 100)
BENCHMARK_OUTPUT_CALLS = tuple(range(0, 30, 2))
EXPECTED_PAIR_COUNT = len(a32.ANIMATION_CASE_IDS) * len(BENCHMARK_OUTPUT_CALLS)
TIMED_REPEATS = 3
WARMUP_REPEATS = 2
EQUIVALENCE_TOLERANCE = 1.0e-6
MAX_MEMORY_FRACTION = 0.90
MAX_BATCH_TIME_RATIO = 0.90
REQUIRED_CUBLAS_WORKSPACE_CONFIG = ":4096:8"

OWNED_SOURCE_PATHS = (
    "docs/time_dependent_no/W26_L5_RESPONSE_FILTERED_BLOCK_PREREGISTRATION.md",
    "utility/time_dependent_no/pcno_runtime.py",
    "utility/time_dependent_no/pcno_resolution_transfer.py",
    "scripts/time_dependent_no/benchmark_pcno_paired_native_batch.py",
    "tests/time_dependent_no/test_pcno_runtime.py",
)
DEPENDENCY_PATHS = (
    "scripts/time_dependent_no/evaluate_pcno_affine_shadow_tether_rollout.py",
    "scripts/time_dependent_no/evaluate_pcno_cross_resolution_teacher_forced.py",
    "scripts/time_dependent_no/evaluate_pcno_modal_affine_transfer.py",
    "scripts/time_dependent_no/visualize_pcno_affine_shadow_tether_rollout.py",
    "utility/time_dependent_no/pcno_artifacts.py",
    "utility/time_dependent_no/pcno_cross_resolution_teacher_forced.py",
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON payload must be an object: {path}")
    return payload


def _verify_a32_result(path: Path) -> dict[str, Any]:
    if sha256_file(path) != A32_RESULT_SHA256:
        raise ValueError("A32 result file SHA-256 mismatch")
    payload = _read_json(path)
    verify_payload_sha256(payload)
    gate = payload.get("gate", {})
    if (
        payload.get("schema") != a32.RESULT_SCHEMA
        or payload.get("working_id") != a32.WORKING_ID
        or payload.get("payload_sha256") != A32_PAYLOAD_SHA256
        or payload.get("status") != "qualified_adaptive_open_rollout"
        or gate.get("status") != "qualified"
        or gate.get("failed_checks") != []
        or not all(gate.get("checks", {}).values())
    ):
        raise ValueError("A32 is not the exact qualified result")
    inventory = payload.get("artifact_inventory_before_summary")
    if not isinstance(inventory, Mapping) or not inventory:
        raise ValueError("A32 artifact inventory is unavailable")
    root = path.parent.resolve()
    for relative, expected in inventory.items():
        candidate = (root / str(relative)).resolve()
        if root not in candidate.parents or not isinstance(expected, Mapping):
            raise ValueError("A32 artifact inventory contains an unsafe path")
        if (
            not candidate.is_file()
            or sha256_file(candidate) != expected.get("sha256")
            or candidate.stat().st_size != int(expected.get("bytes", -1))
        ):
            raise ValueError(f"A32 artifact differs: {relative}")
    return payload


def _verify_a32_contract(
    result_path: Path, manifest_path: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    result = _verify_a32_result(result_path)
    if sha256_file(manifest_path) != A32_BUNDLE_MANIFEST_SHA256:
        raise ValueError("A32 bundle-manifest SHA-256 mismatch")
    _, manifest = a32_visualize._load_contract(result_path, manifest_path)
    source_path = result_path.parent / "source_manifest.json"
    source = _read_json(source_path)
    verify_payload_sha256(source)
    if (
        result.get("source_manifest_sha256") != sha256_file(source_path)
        or result.get("source_manifest_payload_sha256") != source["payload_sha256"]
        or manifest.get("contract") != a32.ANIMATION_CONTRACT
        or tuple(manifest["contract"]["case_ids"]) != a32.ANIMATION_CASE_IDS
        or tuple(manifest["contract"]["output_calls"]) != a32.ANIMATION_CALLS
        or len(manifest.get("bundles", ())) != len(a32.ANIMATION_CASE_IDS)
    ):
        raise ValueError("A32 source or animation contract differs")
    return result, manifest, source


def _source_hashes() -> dict[str, dict[str, str]]:
    return {
        "owned": sha256_files(OWNED_SOURCE_PATHS, root=ROOT),
        "dependencies": sha256_files(DEPENDENCY_PATHS, root=ROOT),
    }


def _source_manifest(
    args: argparse.Namespace,
    *,
    base_source: Mapping[str, Any],
    a32_source: Mapping[str, Any],
    bundle_payload_sha256: str,
) -> dict[str, Any]:
    return with_payload_sha256(
        {
            "schema": SOURCE_SCHEMA,
            "working_id": WORKING_ID,
            "source_sha256": _source_hashes(),
            "lineage": {
                "a32_result_sha256": A32_RESULT_SHA256,
                "a32_payload_sha256": A32_PAYLOAD_SHA256,
                "a32_source_manifest_sha256": sha256_file(
                    args.a32_result.parent / "source_manifest.json"
                ),
                "a32_source_manifest_payload_sha256": a32_source["payload_sha256"],
                "a32_bundle_manifest_sha256": A32_BUNDLE_MANIFEST_SHA256,
                "a32_bundle_manifest_payload_sha256": bundle_payload_sha256,
            },
            "base_runtime_source_manifest": dict(base_source),
            "input_contract": {
                "case_ids": list(a32.ANIMATION_CASE_IDS),
                "output_calls": list(BENCHMARK_OUTPUT_CALLS),
                "pair_count": EXPECTED_PAIR_COUNT,
                "state_fields": [
                    "raw_shadow_conservative",
                    "corrected_conservative",
                ],
                "storage_dtype": "float32_visualization_only",
                "truth_array_indexed": False,
                "accuracy_or_selector_input": False,
            },
            "benchmark_contract": {
                "native_resolution": list(NATIVE_RESOLUTION),
                "warmups_per_method": WARMUP_REPEATS,
                "timed_repeats": TIMED_REPEATS,
                "alternating_method_order": True,
                "reverse_batch_isolation": True,
                "amp": "none",
                "equivalence_tolerance": EQUIVALENCE_TOLERANCE,
                "maximum_memory_fraction": MAX_MEMORY_FRACTION,
                "maximum_batch_time_ratio_strict": MAX_BATCH_TIME_RATIO,
            },
        }
    )


def synthetic_summary() -> dict[str, Any]:
    checks = {
        "case_inventory_5": len(a32.ANIMATION_CASE_IDS) == 5,
        "input_call_inventory_15": BENCHMARK_OUTPUT_CALLS == tuple(range(0, 30, 2)),
        "pair_inventory_75": EXPECTED_PAIR_COUNT == 75,
        "timed_repeats_3": TIMED_REPEATS == 3,
        "warmups_2": WARMUP_REPEATS == 2,
        "owned_source_inventory_5": len(OWNED_SOURCE_PATHS) == 5,
    }
    return {
        "schema": "pcno_paired_native_batch_synthetic_v1",
        "working_id": WORKING_ID,
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
    }


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    _, manifest, a32_source = _verify_a32_contract(
        args.a32_result, args.bundle_manifest
    )
    _, _, _, store, base_source = base._open_contract(a29._base_args(args))
    try:
        source = _source_manifest(
            args,
            base_source=base_source,
            a32_source=a32_source,
            bundle_payload_sha256=manifest["payload_sha256"],
        )
        checks = {
            "source_inventory_exact": set(source["source_sha256"]["owned"])
            == set(OWNED_SOURCE_PATHS)
            and set(source["source_sha256"]["dependencies"]) == set(DEPENDENCY_PATHS),
            "a32_base_runtime_exact": a32_source["base_runtime_source_manifest"]
            == base_source,
            "a32_lineage_exact": source["lineage"]["a32_result_sha256"]
            == A32_RESULT_SHA256
            and source["lineage"]["a32_payload_sha256"] == A32_PAYLOAD_SHA256,
            "bundle_inventory_exact": tuple(manifest["contract"]["case_ids"])
            == a32.ANIMATION_CASE_IDS
            and tuple(manifest["contract"]["output_calls"]) == a32.ANIMATION_CALLS,
            "pair_inventory_exact": EXPECTED_PAIR_COUNT == 75,
        }
        payload = with_payload_sha256(
            {
                "schema": PREFLIGHT_SCHEMA,
                "working_id": WORKING_ID,
                "status": "passed" if all(checks.values()) else "failed",
                "checks": checks,
                "source_manifest": source,
                "checkpoint_model_built": False,
                "bundle_state_arrays_loaded": False,
                "truth_arrays_loaded": False,
                "model_calls": 0,
                "recurrence_executed": False,
            }
        )
    finally:
        store.close()
    if payload["status"] != "passed":
        raise ValueError("A40 preflight checks did not all pass")
    atomic_write_json(args.output, payload)
    return payload


def _verify_preflight(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    verify_payload_sha256(payload)
    if (
        payload.get("schema") != PREFLIGHT_SCHEMA
        or payload.get("working_id") != WORKING_ID
        or payload.get("status") != "passed"
        or not all(payload.get("checks", {}).values())
        or payload.get("checkpoint_model_built") is not False
        or payload.get("bundle_state_arrays_loaded") is not False
        or payload.get("truth_arrays_loaded") is not False
        or payload.get("model_calls") != 0
        or payload.get("recurrence_executed") is not False
        or payload.get("source_manifest", {}).get("source_sha256") != _source_hashes()
    ):
        raise ValueError("A40 preflight differs from the frozen contract")
    return payload


def _state_arrays_from_bundle(
    bundle: Any,
) -> tuple[str, np.ndarray, np.ndarray, np.ndarray]:
    """Read only A32 inference states; never index the bundled truth array."""

    required = {
        "case_id",
        "output_calls",
        "native_resolution",
        "raw_shadow_conservative",
        "corrected_conservative",
        "truth_conservative",
    }
    if set(bundle.files) != {
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
    } or not required.issubset(bundle.files):
        raise ValueError("A32 bundle field inventory differs")
    case_id = str(np.asarray(bundle["case_id"]).item())
    output_calls = np.asarray(bundle["output_calls"], dtype=np.int64)
    resolution = tuple(
        int(value) for value in np.asarray(bundle["native_resolution"]).tolist()
    )
    raw = np.asarray(bundle["raw_shadow_conservative"])
    corrected = np.asarray(bundle["corrected_conservative"])
    expected_shape = (len(a32.ANIMATION_CALLS), np.prod(NATIVE_RESOLUTION), 4)
    if (
        resolution != NATIVE_RESOLUTION
        or tuple(output_calls) != a32.ANIMATION_CALLS
        or raw.shape != expected_shape
        or corrected.shape != expected_shape
        or raw.dtype != np.float32
        or corrected.dtype != np.float32
        or not np.isfinite(raw).all()
        or not np.isfinite(corrected).all()
    ):
        raise ValueError(f"A32 state bundle differs for {case_id}")
    return case_id, output_calls, raw, corrected


def _load_state_pairs(
    manifest_path: Path, manifest: Mapping[str, Any]
) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for row in manifest["bundles"]:
        path = (manifest_path.parent / str(row["path"])).resolve()
        if (
            path.parent != manifest_path.parent.resolve()
            or sha256_file(path) != row["sha256"]
            or row.get("storage_dtype") != "float32_visualization_only"
        ):
            raise ValueError("A32 bundle path, hash, or storage dtype differs")
        with np.load(path, allow_pickle=False) as bundle:
            case_id, output_calls, raw, corrected = _state_arrays_from_bundle(bundle)
        if case_id != row["case_id"]:
            raise ValueError("A32 bundle case ID differs")
        for index, output_call in enumerate(output_calls[:-1]):
            if int(output_call) not in BENCHMARK_OUTPUT_CALLS:
                continue
            pairs.append(
                {
                    "case_id": case_id,
                    "output_call": int(output_call),
                    "shadow": np.array(raw[index], copy=True),
                    "accepted": np.array(corrected[index], copy=True),
                }
            )
    if (
        len(pairs) != EXPECTED_PAIR_COUNT
        or tuple(dict.fromkeys(pair["case_id"] for pair in pairs))
        != a32.ANIMATION_CASE_IDS
    ):
        raise ValueError("A40 paired-state inventory differs")
    return pairs


def _maximum_abs(value: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(value, dtype=np.float64))))


def _sequential_prediction(runtime: Any, states: tuple[np.ndarray, np.ndarray]):
    predictions = []
    timings = []
    for state in states:
        prediction, timing = predict_resolution_sample(
            runtime.model,
            runtime.sample_by_resolution[NATIVE_RESOLUTION],
            state,
            device=runtime.device,
            amp="none",
            repeats=TIMED_REPEATS,
        )
        predictions.append(prediction)
        timings.append(timing)
    return np.stack(predictions), timings


def _batched_prediction(runtime: Any, states: tuple[np.ndarray, np.ndarray]):
    return predict_resolution_batch(
        runtime.model,
        runtime.sample_by_resolution[NATIVE_RESOLUTION],
        states,
        device=runtime.device,
        amp="none",
        repeats=TIMED_REPEATS,
    )


def _benchmark_pair(
    runtime: Any, pair: Mapping[str, Any], pair_index: int
) -> dict[str, Any]:
    states = (pair["shadow"], pair["accepted"])
    if pair_index % 2 == 0:
        sequential, sequential_timings = _sequential_prediction(runtime, states)
        batched, batch_timing = _batched_prediction(runtime, states)
        method_order = "sequential>batch"
    else:
        batched, batch_timing = _batched_prediction(runtime, states)
        sequential, sequential_timings = _sequential_prediction(runtime, states)
        method_order = "batch>sequential"
    reversed_batch, reverse_timing = predict_resolution_batch(
        runtime.model,
        runtime.sample_by_resolution[NATIVE_RESOLUTION],
        states[::-1],
        device=runtime.device,
        amp="none",
        repeats=1,
    )
    sequential_seconds = sum(
        sum(float(value) for value in timing["forward_seconds"])
        for timing in sequential_timings
    )
    batch_seconds = sum(float(value) for value in batch_timing["forward_seconds"])
    sequential_peak = max(
        int(timing["peak_gpu_memory_bytes"]) for timing in sequential_timings
    )
    batch_peak = max(
        int(batch_timing["peak_gpu_memory_bytes"]),
        int(reverse_timing["peak_gpu_memory_bytes"]),
    )
    return {
        "case_id": pair["case_id"],
        "output_call": pair["output_call"],
        "method_order": method_order,
        "state_pair_max_abs_difference": _maximum_abs(states[0] - states[1]),
        "batch_vs_sequential_max_abs": _maximum_abs(batched - sequential),
        "reverse_batch_isolation_max_abs": _maximum_abs(reversed_batch[::-1] - batched),
        "sequential_repeat_max_abs": max(
            float(timing["repeat_max_abs"]) for timing in sequential_timings
        ),
        "batch_repeat_max_abs": float(batch_timing["repeat_max_abs"]),
        "finite": bool(
            np.isfinite(sequential).all()
            and np.isfinite(batched).all()
            and np.isfinite(reversed_batch).all()
        ),
        "sequential_seconds_3_repeats": sequential_seconds,
        "batch_seconds_3_repeats": batch_seconds,
        "batch_time_ratio": batch_seconds / sequential_seconds,
        "sequential_peak_gpu_memory_bytes": sequential_peak,
        "batch_peak_gpu_memory_bytes": batch_peak,
    }


def _gate(rows: Sequence[Mapping[str, Any]], total_memory: int) -> dict[str, Any]:
    sequential_total = sum(float(row["sequential_seconds_3_repeats"]) for row in rows)
    batch_total = sum(float(row["batch_seconds_3_repeats"]) for row in rows)
    batch_peak = max(
        (int(row["batch_peak_gpu_memory_bytes"]) for row in rows), default=0
    )
    checks = {
        "pair_inventory_exact": len(rows) == EXPECTED_PAIR_COUNT,
        "all_outputs_finite": bool(rows) and all(bool(row["finite"]) for row in rows),
        "batch_sequential_equivalence": bool(rows)
        and max(float(row["batch_vs_sequential_max_abs"]) for row in rows)
        <= EQUIVALENCE_TOLERANCE,
        "reverse_batch_isolation": bool(rows)
        and max(float(row["reverse_batch_isolation_max_abs"]) for row in rows)
        <= EQUIVALENCE_TOLERANCE,
        "repeatability": bool(rows)
        and max(
            max(
                float(row["sequential_repeat_max_abs"]),
                float(row["batch_repeat_max_abs"]),
            )
            for row in rows
        )
        <= EQUIVALENCE_TOLERANCE,
        "memory_resolved_and_bounded": total_memory > 0
        and batch_peak <= MAX_MEMORY_FRACTION * total_memory,
        "material_native_speedup": sequential_total > 0.0
        and batch_total / sequential_total < MAX_BATCH_TIME_RATIO,
    }
    return {
        "status": "qualified" if all(checks.values()) else "failed",
        "checks": checks,
        "failed_checks": [key for key, value in checks.items() if not value],
    }


def run_benchmark(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    preflight = _verify_preflight(args.preflight)
    _, manifest, _ = _verify_a32_contract(args.a32_result, args.bundle_manifest)
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != REQUIRED_CUBLAS_WORKSPACE_CONFIG:
        raise ValueError("A40 requires CUBLAS_WORKSPACE_CONFIG=:4096:8")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=False)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    runtime = base._build_runtime(a29._base_args(args))
    try:
        if runtime.device.type != "cuda":
            raise ValueError("A40 benchmark requires CUDA")
        if (
            runtime.source_manifest
            != preflight["source_manifest"]["base_runtime_source_manifest"]
        ):
            raise ValueError("A40 runtime source differs from preflight")
        pairs = _load_state_pairs(args.bundle_manifest, manifest)
        warmup_states = (pairs[0]["shadow"], pairs[0]["accepted"])
        for _ in range(WARMUP_REPEATS):
            for state in warmup_states:
                predict_resolution_sample(
                    runtime.model,
                    runtime.sample_by_resolution[NATIVE_RESOLUTION],
                    state,
                    device=runtime.device,
                    amp="none",
                    repeats=1,
                )
            predict_resolution_batch(
                runtime.model,
                runtime.sample_by_resolution[NATIVE_RESOLUTION],
                warmup_states,
                device=runtime.device,
                amp="none",
                repeats=1,
            )
        rows = []
        for index, pair in enumerate(pairs):
            print(
                f"A40 pair {index + 1}/{len(pairs)}: "
                f"{pair['case_id']} call {pair['output_call']}",
                flush=True,
            )
            rows.append(_benchmark_pair(runtime, pair, index))
        total_memory = int(
            torch.cuda.get_device_properties(runtime.device).total_memory
        )
        gate = _gate(rows, total_memory)
        sequential_total = sum(
            float(row["sequential_seconds_3_repeats"]) for row in rows
        )
        batch_total = sum(float(row["batch_seconds_3_repeats"]) for row in rows)
        summary = {
            "pairs": len(rows),
            "maximum_batch_vs_sequential_abs": max(
                float(row["batch_vs_sequential_max_abs"]) for row in rows
            ),
            "maximum_reverse_batch_isolation_abs": max(
                float(row["reverse_batch_isolation_max_abs"]) for row in rows
            ),
            "maximum_repeat_abs": max(
                max(
                    float(row["sequential_repeat_max_abs"]),
                    float(row["batch_repeat_max_abs"]),
                )
                for row in rows
            ),
            "sequential_seconds_3_repeats_total": sequential_total,
            "batch_seconds_3_repeats_total": batch_total,
            "batch_time_ratio": batch_total / sequential_total,
            "native_speedup": sequential_total / batch_total,
            "median_pair_batch_time_ratio": float(
                np.median([float(row["batch_time_ratio"]) for row in rows])
            ),
            "maximum_sequential_peak_gpu_memory_bytes": max(
                int(row["sequential_peak_gpu_memory_bytes"]) for row in rows
            ),
            "maximum_batch_peak_gpu_memory_bytes": max(
                int(row["batch_peak_gpu_memory_bytes"]) for row in rows
            ),
            "total_gpu_memory_bytes": total_memory,
            "batch_peak_memory_fraction": max(
                int(row["batch_peak_gpu_memory_bytes"]) for row in rows
            )
            / total_memory,
        }
        write_csv(output_dir / "paired_native_benchmark.csv", rows)
        atomic_write_json(
            output_dir / "source_manifest.json", preflight["source_manifest"]
        )
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
                    "qualified_batching_feasibility"
                    if gate["status"] == "qualified"
                    else "stopped_batching_feasibility"
                ),
                "gate": gate,
                "summary": summary,
                "execution": {
                    "warmup_sequential_forward_passes": 2 * WARMUP_REPEATS,
                    "warmup_batch_forward_passes": WARMUP_REPEATS,
                    "timed_sequential_forward_passes": 2
                    * EXPECTED_PAIR_COUNT
                    * TIMED_REPEATS,
                    "timed_batch_forward_passes": EXPECTED_PAIR_COUNT * TIMED_REPEATS,
                    "reverse_isolation_forward_passes": EXPECTED_PAIR_COUNT,
                    "logical_native_predictions_per_a32_active_step": 2,
                    "paired_native_kernel_invocations_per_a32_active_step": 1,
                    "a32_all_case_physical_forward_projection": 484,
                    "a33_corrected_arm_physical_forward_projection": 276,
                    "projection_includes_fine_calls": True,
                    "projection_is_not_measured_full_rollout_cost": True,
                    "environment": runtime_environment(runtime.device),
                },
                "preflight_sha256": sha256_file(args.preflight),
                "preflight_payload_sha256": preflight["payload_sha256"],
                "source_manifest_sha256": sha256_file(
                    output_dir / "source_manifest.json"
                ),
                "source_manifest_payload_sha256": preflight["source_manifest"][
                    "payload_sha256"
                ],
                "artifact_inventory_before_summary": artifacts,
                "bundle_state_arrays_loaded": True,
                "truth_arrays_loaded": False,
                "fine_predictions_executed": False,
                "recurrence_executed": False,
                "accuracy_scored": False,
                "new_population_opened": False,
                "claim_boundary": (
                    "Float32 A32-state kernel feasibility only; no accuracy, full-rollout "
                    "cost, independent-data, coefficient-transfer, conservation, "
                    "convergence, direct-off-grid, or Richardson claim."
                ),
            }
        )
        atomic_write_json(output_dir / "paired_native_batch_benchmark.json", result)
        return result, 0 if gate["status"] == "qualified" else 4
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
    parser.add_argument("--a32-result", type=Path, required=True)
    parser.add_argument("--bundle-manifest", type=Path, required=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("synthetic")
    preflight = commands.add_parser("preflight")
    _add_external_arguments(preflight)
    preflight.add_argument("--output", type=Path, required=True)
    benchmark = commands.add_parser("benchmark")
    _add_external_arguments(benchmark)
    benchmark.add_argument("--preflight", type=Path, required=True)
    benchmark.add_argument("--output-dir", type=Path, required=True)
    benchmark.add_argument("--device", choices=("cuda",), default="cuda")
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
    payload, exit_code = run_benchmark(args)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
