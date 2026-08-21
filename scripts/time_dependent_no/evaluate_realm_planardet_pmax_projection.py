"""Evaluate a causal cumulative-pMax projection on the frozen PlanarDet PCNO.

This diagnostic keeps the D092-R1 step-950 checkpoint, open-validation case,
decoder, and learned calls fixed.  After residual next-state reconstruction it
decodes pMax on CPU through the metric path and replaces only pMax by
``max(current_pMax, proposed_pMax)``.  The projected state is scored and, for
free recurrence, fed back into the next learned call.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.time_dependent_no.evaluate_realm_planardet_pcno import (
    DEVICE_NAME,
    DTYPE,
    _configure_determinism,
    _free_minus_teacher,
    _is_within,
    _load_json,
    _prepare_output_directory,
    _runtime_manifest,
    _structure_summary,
    _validate_best_checkpoint,
    _validate_closed_training_tree,
    _view_summary,
    _write_json,
    _write_npy,
)
from utility.time_dependent_no.realm_benchmark import (
    canonical_json_sha256,
    parse_manifest_payload,
    predict_one_call,
)
from utility.time_dependent_no.realm_pcno import (
    RealmPCNOConfig,
    RealmRegularGridPCNO,
    build_realm_regular_grid_geometry,
)
from utility.time_dependent_no.realm_planardet import (
    CANONICAL_COORDINATE_ORDER,
    DOMAIN_LENGTHS_XY,
    PLANARDET_OPEN_MANIFEST_SHA256,
    PLANARDET_TRAIN_GROUPS,
    PLANARDET_VAL_GROUPS,
    load_planardet_metadata,
    sha256_file,
    validate_local_open_tree,
    validate_planardet_open_manifest,
)
from utility.time_dependent_no.realm_planardet_artifacts import (
    EXECUTABLE_ENTRYPOINTS,
    build_source_manifest,
)
from utility.time_dependent_no.realm_planardet_runtime import (
    CHANNELS,
    EXPECTED_PARAMETER_COUNTS,
    FC_DIM,
    MODE_COUNTS_XY,
    VALIDATION_HORIZON,
    competence_gate,
    load_normalized_trajectories,
    load_normalizer_bundle,
    planardet_boundary_mask,
    validate_frozen_controls,
    validation_control_summary,
)

EXPERIMENT_ID = "w26_l4_pd0_pmax_projection_p0b_d092r1_step950_20260817a"
PREREGISTRATION_SCHEMA = "w26_l4_planardet_pmax_projection_preregistration_v2"
PROJECTION_SOURCE_SCHEMA = "w26_l4_planardet_pmax_projection_source_v2"
RUNTIME_SCHEMA = "w26_l4_planardet_pmax_projection_cuda_runtime_v2"
RESULT_SCHEMA = "w26_l4_planardet_pmax_projection_evaluation_v2"
FINAL_MANIFEST_SCHEMA = "w26_l4_planardet_pmax_projection_final_hash_manifest_v2"
SUMMARY_SCHEMA = "w26_l4_planardet_pmax_projection_summary_v2"

FAILED_P0_EXPERIMENT_ID = "w26_l4_pd0_pmax_projection_p0_d092r1_step950_20260817a"
FAILED_P0_RESULT_SHA256 = (
    "1734f4af369b3686ae66956912999b324d184c783a4c46b686292626a6bc9ca0"
)
FAILED_P0_FINAL_MANIFEST_SHA256 = (
    "ca7edb87f25c6409b21255eeab31c3b43cdc4bf32b8981896e9acb56da41093b"
)

BASELINE_RUN_ID = "d092_r1_planardet_residual_pcno_seed0_width96_5000_20260815a"
BASELINE_SOURCE_DIGEST = (
    "1c48afecdde31d5998695438a39b19ddb328f91b63e272c26ecaed0c30e6f502"
)
BASELINE_CHECKPOINT_SHA256 = (
    "3ecb4ef90c9800812323f388b699ea639fc283ac8da15b801ae6283442ab9924"
)
BASELINE_RESULT_SHA256 = (
    "81c1ba154d6149a2288db46fc5ffad56c86852976fae60ee40813f7c0a9c19c6"
)
BASELINE_RESULT_PAYLOAD_SHA256 = (
    "82efc4030c5942844946ef6176401cc107c696a75ca8dcf0c31097de23e11fdb"
)
BASELINE_FINAL_MANIFEST_SHA256 = (
    "5597bf8579744b8a84ce258bf71e63c9e3d27f1f467091b76dba7d0acaef3772"
)
BASELINE_SUMMARY_SHA256 = (
    "56a717aaef1759bb307ea326f311116f6e4d93faf846e3728251d209bfd47a3a"
)
BASELINE_EVALUATION_FILES = {
    "teacher_prediction_normalized.npy": (
        "75ba3f96d8b48d2a8fd820f438be4b8096573f14071a46654422f25af795d557"
    ),
    "free_prediction_normalized.npy": (
        "d3005a85dfe01d7c43b134f5cb910647fdd98e04e10eb335847318ebd34e5887"
    ),
    "result.json": BASELINE_RESULT_SHA256,
    "source_manifest.json": (
        "8f6db8b68d4ddf71bee1e91a46ac3a78e57628a44fbd84b0214d907e4b2c0b42"
    ),
    "runtime_manifest.json": (
        "4890961875b17d489346f31e48d0d939ad3a20c36484d84c8602d0189af86208"
    ),
}
BASELINE_EVALUATION_SCHEMA = "w26_l4_planardet_pd0_a3_open_validation_evaluation_v1"
BASELINE_FINAL_MANIFEST_SCHEMA = (
    "w26_l4_planardet_pd0_a3_evaluation_final_hash_manifest_v1"
)
BASELINE_SUMMARY_SCHEMA = "w26_l4_planardet_pd0_a3_evaluation_summary_v1"
BASELINE_BEST_STEP = 950
PMAX_CHANNEL = 12
EXPECTED_ARRAY_SHAPE = (VALIDATION_HORIZON, CHANNELS, 384, 832)
MATERIAL_EFFECT_RATIO = 0.10
NEAR_NULL_RATIO = 0.05
TRUTH_INPUT_NO_HARM_RATIO = 1.01


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--baseline-evaluation-dir", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--normalizer-arrays", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def build_projection_source_manifest() -> dict[str, Any]:
    relative = "scripts/time_dependent_no/evaluate_realm_planardet_pmax_projection.py"
    path = REPO_ROOT.joinpath(*relative.split("/"))
    payload: dict[str, Any] = {
        "schema": PROJECTION_SOURCE_SCHEMA,
        "baseline_training_source_manifest_digest": BASELINE_SOURCE_DIGEST,
        "files": [
            {
                "path": relative,
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        ],
    }
    payload["canonical_payload_sha256"] = canonical_json_sha256(payload)
    return payload


def build_preregistration() -> dict[str, Any]:
    source = build_projection_source_manifest()
    payload: dict[str, Any] = {
        "schema": PREREGISTRATION_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "status": "frozen_before_gpu_execution",
        "baseline": {
            "run_id": BASELINE_RUN_ID,
            "best_step": BASELINE_BEST_STEP,
            "checkpoint_sha256": BASELINE_CHECKPOINT_SHA256,
            "training_source_manifest_digest": BASELINE_SOURCE_DIGEST,
            "evaluation_result_sha256": BASELINE_RESULT_SHA256,
            "evaluation_result_payload_sha256": BASELINE_RESULT_PAYLOAD_SHA256,
            "evaluation_final_manifest_sha256": BASELINE_FINAL_MANIFEST_SHA256,
            "teacher_array_sha256": BASELINE_EVALUATION_FILES[
                "teacher_prediction_normalized.npy"
            ],
            "free_array_sha256": BASELINE_EVALUATION_FILES[
                "free_prediction_normalized.npy"
            ],
        },
        "projection_source_manifest_digest": source["canonical_payload_sha256"],
        "corrects_failed_attempt": {
            "experiment_id": FAILED_P0_EXPERIMENT_ID,
            "result_sha256": FAILED_P0_RESULT_SHA256,
            "final_hash_manifest_sha256": FAILED_P0_FINAL_MANIFEST_SHA256,
            "failure": (
                "GPU-side decoded comparison did not close the exact CPU metric "
                "decoder against the physical current pMax"
            ),
            "scientific_interpretation_allowed": False,
        },
        "intervention": {
            "field": "pMax",
            "channel": PMAX_CHANNEL,
            "location": "after residual next-state reconstruction and before scoring or feedback",
            "map": "projected_pMax=max(current_pMax, proposed_pMax)",
            "space": (
                "decode proposed pMax on CPU with the exact float32 metric path, "
                "compare against the exact physical current pMax in Pa, then encode "
                "that physical floor with upward nextafter closure only where the "
                "decoded proposal is lower"
            ),
            "changed_channels": [PMAX_CHANNEL],
            "truth_input_current": "released current pMax",
            "free_current": "previous accepted projected pMax",
            "other_model_outputs": "bitwise unchanged before subsequent feedback",
            "optimization": "none",
        },
        "views": ["projected_truth_input", "projected_free_recurrence"],
        "primary_causal_readout": (
            "mean grouped normalized prediction error excluding the pMax group; "
            "this can change only through projected-state feedback"
        ),
        "effect_bands": {
            "materially_helpful_ratio_at_most": 1.0 - MATERIAL_EFFECT_RATIO,
            "materially_harmful_ratio_at_least": 1.0 + MATERIAL_EFFECT_RATIO,
            "near_null_ratio_interval": [
                1.0 - NEAR_NULL_RATIO,
                1.0 + NEAR_NULL_RATIO,
            ],
            "otherwise": "small_or_inconclusive",
        },
        "gates": {
            "truth_input_npe_no_harm_ratio_at_most": TRUTH_INPUT_NO_HARM_RATIO,
            "projected_truth_input_competence_required": True,
            "teacher_raw_replay_must_match_all_49_baseline_calls_bitwise": True,
            "free_raw_call_1_must_match_baseline_bitwise": True,
            "projection_must_change_only_pMax_bitwise": True,
            "projected_pMax_must_be_nondecreasing": True,
            "complete_finite_H49_required_for_effect_classification": True,
            "teacher_projection_count_must_match_baseline_decrease_count": True,
        },
        "runtime": {
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8 before Python startup",
        },
        "population": "one released open-validation PlanarDet trajectory",
        "sealed_test_object_opened": False,
        "anti_claims": [
            "no new training, checkpoint selection, architecture, or data",
            "no sealed-test, seed-robustness, architecture-level, or REALM-wide claim",
            "projection evidence cannot establish physical conservation",
            "a pMax-channel score improvement alone is not causal recurrence evidence",
        ],
    }
    payload["canonical_payload_sha256"] = canonical_json_sha256(payload)
    return payload


def _validate_projection_preregistration(payload: Mapping[str, Any]) -> None:
    if payload != build_preregistration():
        raise ValueError("projection preregistration differs from frozen contract")


def _validate_baseline_evaluation(
    directory: Path,
    *,
    current_source_manifest: Mapping[str, Any],
) -> Mapping[str, Any]:
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError(
            "baseline evaluation directory must be a non-symlink directory"
        )
    expected_inventory = set(BASELINE_EVALUATION_FILES) | {
        "final_hash_manifest.json",
        "summary.json",
    }
    actual: set[str] = set()
    for path in directory.iterdir():
        if path.is_symlink() or not path.is_file():
            raise ValueError("baseline evaluation contains a non-regular object")
        if any(part.casefold() == "test" for part in path.relative_to(directory).parts):
            raise ValueError("baseline evaluation contains a sealed test path")
        actual.add(path.name)
    if actual != expected_inventory:
        raise ValueError("baseline evaluation inventory differs")

    final_manifest = _load_json(directory / "final_hash_manifest.json")
    if (
        final_manifest.get("schema") != BASELINE_FINAL_MANIFEST_SCHEMA
        or final_manifest.get("self_hash_excluded") is not True
        or final_manifest.get("test_object_opened") is not False
        or final_manifest.get("files") != BASELINE_EVALUATION_FILES
        or sha256_file(directory / "final_hash_manifest.json")
        != BASELINE_FINAL_MANIFEST_SHA256
    ):
        raise ValueError("baseline evaluation final manifest differs")
    for name, expected in BASELINE_EVALUATION_FILES.items():
        if sha256_file(directory / name) != expected:
            raise ValueError(f"baseline evaluation hash differs: {name}")

    source = _load_json(directory / "source_manifest.json")
    if source != current_source_manifest:
        raise ValueError("baseline evaluation source differs from closed training")
    summary = _load_json(directory / "summary.json")
    if (
        sha256_file(directory / "summary.json") != BASELINE_SUMMARY_SHA256
        or summary.get("schema") != BASELINE_SUMMARY_SCHEMA
        or summary.get("result_sha256") != BASELINE_RESULT_SHA256
        or summary.get("final_hash_manifest_sha256") != BASELINE_FINAL_MANIFEST_SHA256
        or summary.get("test_object_opened") is not False
    ):
        raise ValueError("baseline evaluation summary differs")

    result = _load_json(directory / "result.json")
    unsigned = {
        key: value for key, value in result.items() if key != "canonical_payload_sha256"
    }
    gate = result.get("truth_input_competence_gate")
    closure = result.get("evaluator_closure")
    if (
        result.get("schema") != BASELINE_EVALUATION_SCHEMA
        or result.get("run_id") != BASELINE_RUN_ID
        or result.get("canonical_payload_sha256") != BASELINE_RESULT_PAYLOAD_SHA256
        or canonical_json_sha256(unsigned) != BASELINE_RESULT_PAYLOAD_SHA256
        or result.get("checkpoint", {}).get("sha256") != BASELINE_CHECKPOINT_SHA256
        or result.get("checkpoint", {}).get("completed_step") != BASELINE_BEST_STEP
        or result.get("prediction_arrays", {})
        .get("teacher_prediction_normalized.npy", {})
        .get("sha256")
        != BASELINE_EVALUATION_FILES["teacher_prediction_normalized.npy"]
        or result.get("prediction_arrays", {})
        .get("free_prediction_normalized.npy", {})
        .get("sha256")
        != BASELINE_EVALUATION_FILES["free_prediction_normalized.npy"]
        or not isinstance(gate, Mapping)
        or gate.get("all_gates_pass") is not False
        or gate.get("all_pMax_nondecreasing_from_input") is not False
        or any(
            value is not True
            for key, value in gate.items()
            if key not in {"all_gates_pass", "all_pMax_nondecreasing_from_input"}
        )
        or not isinstance(closure, Mapping)
        or closure.get("all_gates_pass") is not True
        or result.get("mechanism_interpretation_allowed") is not False
        or result.get("test_object_opened") is not False
    ):
        raise ValueError("baseline evaluation identity or gate surface differs")

    for name in (
        "teacher_prediction_normalized.npy",
        "free_prediction_normalized.npy",
    ):
        array = np.load(directory / name, allow_pickle=False, mmap_mode="r")
        if array.shape != EXPECTED_ARRAY_SHAPE or array.dtype != np.dtype("float32"):
            raise ValueError(f"baseline prediction array contract differs: {name}")
    return result


def _validate_startup_environment() -> None:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError(
            "CUBLAS_WORKSPACE_CONFIG must equal :4096:8 before Python startup"
        )


def project_pmax_nondecreasing(
    proposal: torch.Tensor,
    current_pmax_pa: torch.Tensor,
    *,
    pmax_mean_pa: float,
    pmax_scale_pa: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if (
        proposal.ndim != 4
        or proposal.shape[1] != CHANNELS
        or current_pmax_pa.shape
        != (proposal.shape[0], proposal.shape[2], proposal.shape[3])
        or proposal.dtype != DTYPE
        or current_pmax_pa.dtype != DTYPE
        or proposal.device.type != "cpu"
        or current_pmax_pa.device.type != "cpu"
    ):
        raise ValueError(
            "projection requires CPU float32 proposal [batch,13,y,x] and "
            "physical current pMax [batch,y,x]"
        )
    if not bool(torch.isfinite(proposal).all()) or not bool(
        torch.isfinite(current_pmax_pa).all()
    ):
        raise ValueError("projection states must be finite")
    if (
        isinstance(pmax_mean_pa, bool)
        or not isinstance(pmax_mean_pa, (int, float))
        or not math.isfinite(float(pmax_mean_pa))
    ):
        raise ValueError("pMax mean must be finite")
    if (
        isinstance(pmax_scale_pa, bool)
        or not isinstance(pmax_scale_pa, (int, float))
        or not math.isfinite(float(pmax_scale_pa))
        or float(pmax_scale_pa) <= 0.0
    ):
        raise ValueError("pMax scale must be finite and positive")

    proposed_pmax = proposal[:, PMAX_CHANNEL]
    mean = proposal.new_tensor(float(pmax_mean_pa))
    scale = proposal.new_tensor(float(pmax_scale_pa))
    proposed_pa = proposed_pmax * scale + mean
    physical_correction = torch.clamp(current_pmax_pa - proposed_pa, min=0.0)
    active = physical_correction > 0.0

    encoded_floor = (current_pmax_pa - mean) / scale
    positive_infinity = torch.full_like(encoded_floor, torch.inf)
    nextafter_updates = 0
    max_nextafter_steps = 0
    for step in range(1, 5):
        below = active & (encoded_floor * scale + mean < current_pmax_pa)
        count = int(below.sum().item())
        if count == 0:
            break
        encoded_floor = torch.where(
            below,
            torch.nextafter(encoded_floor, positive_infinity),
            encoded_floor,
        )
        nextafter_updates += count
        max_nextafter_steps = step
    if bool((active & (encoded_floor * scale + mean < current_pmax_pa)).any()):
        raise RuntimeError("physical pMax floor did not close after four ulps")

    projected = proposal.clone()
    projected[:, PMAX_CHANNEL] = torch.where(active, encoded_floor, proposed_pmax)
    other_channels_unchanged = torch.equal(
        projected[:, :PMAX_CHANNEL], proposal[:, :PMAX_CHANNEL]
    )
    projected_pa = projected[:, PMAX_CHANNEL] * scale + mean
    pmax_nondecreasing = bool((projected_pa >= current_pmax_pa).all())
    if not other_channels_unchanged or not pmax_nondecreasing:
        raise RuntimeError("pMax projection closure failed")

    corrected_cells = int(active.sum().item())
    total_cells = active.numel()
    correction_sum_pa = float(physical_correction.sum().item())
    correction_max_pa = float(physical_correction.max().item())
    return projected, {
        "corrected_cells": corrected_cells,
        "total_cells": total_cells,
        "corrected_fraction": corrected_cells / total_cells,
        "correction_sum_pa": correction_sum_pa,
        "correction_max_pa": correction_max_pa,
        "correction_mean_active_pa": (
            correction_sum_pa / corrected_cells if corrected_cells else 0.0
        ),
        "encoded_floor_nextafter_updates": nextafter_updates,
        "encoded_floor_max_nextafter_steps": max_nextafter_steps,
        "other_channels_bitwise_unchanged": other_channels_unchanged,
        "projected_pMax_nondecreasing": pmax_nondecreasing,
    }


def _predict_projected_truth_inputs(
    model: RealmRegularGridPCNO,
    truth_normalized: torch.Tensor,
    truth_native: torch.Tensor,
    coordinates: torch.Tensor,
    baseline_teacher: np.ndarray,
    *,
    device: torch.device,
    pmax_mean_pa: float,
    pmax_scale_pa: float,
) -> tuple[torch.Tensor, list[dict[str, Any]], list[float]]:
    predictions = torch.empty_like(truth_normalized[1:])
    records: list[dict[str, Any]] = []
    call_seconds: list[float] = []
    model.eval()
    with torch.no_grad():
        for frame in range(VALIDATION_HORIZON):
            current = truth_normalized[frame].unsqueeze(0).to(device)
            started = time.monotonic()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                raw = predict_one_call(
                    model,
                    current,
                    coordinates,
                    parameterization="residual",
                )
            if not bool(torch.isfinite(raw).all()):
                raise RuntimeError("projected truth-input raw proposal is nonfinite")
            raw_cpu = raw[0].to(device="cpu", dtype=DTYPE)
            expected = torch.from_numpy(np.array(baseline_teacher[frame], copy=True))
            if not torch.equal(raw_cpu, expected):
                raise RuntimeError(
                    f"truth-input raw replay differs from baseline at call {frame + 1}"
                )
            projected, stats = project_pmax_nondecreasing(
                raw_cpu.unsqueeze(0),
                truth_native[frame, PMAX_CHANNEL].unsqueeze(0),
                pmax_mean_pa=pmax_mean_pa,
                pmax_scale_pa=pmax_scale_pa,
            )
            torch.cuda.synchronize(device)
            call_seconds.append(time.monotonic() - started)
            predictions[frame].copy_(projected[0])
            records.append({"call": frame + 1, **stats})
    return predictions, records, call_seconds


def _predict_projected_free_recurrence(
    model: RealmRegularGridPCNO,
    initial: torch.Tensor,
    initial_pmax_pa: torch.Tensor,
    coordinates: torch.Tensor,
    baseline_free: np.ndarray,
    *,
    device: torch.device,
    pmax_mean_pa: float,
    pmax_scale_pa: float,
) -> tuple[torch.Tensor, int | None, list[dict[str, Any]], list[float]]:
    predictions = torch.empty(
        (VALIDATION_HORIZON, *initial.shape), dtype=DTYPE, device="cpu"
    )
    current = initial.unsqueeze(0).to(device)
    current_pmax_pa = initial_pmax_pa.unsqueeze(0).to(device="cpu", dtype=DTYPE)
    nonfinite_call: int | None = None
    records: list[dict[str, Any]] = []
    call_seconds: list[float] = []
    model.eval()
    with torch.no_grad():
        for frame in range(VALIDATION_HORIZON):
            started = time.monotonic()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                raw = predict_one_call(
                    model,
                    current,
                    coordinates,
                    parameterization="residual",
                )
            if not bool(torch.isfinite(raw).all()):
                nonfinite_call = frame + 1
                predictions = predictions[:frame]
                break
            raw_cpu = raw[0].to(device="cpu", dtype=DTYPE)
            if frame == 0:
                expected = torch.from_numpy(np.array(baseline_free[0], copy=True))
                if not torch.equal(raw_cpu, expected):
                    raise RuntimeError("free raw call 1 differs from baseline")
            projected, stats = project_pmax_nondecreasing(
                raw_cpu.unsqueeze(0),
                current_pmax_pa,
                pmax_mean_pa=pmax_mean_pa,
                pmax_scale_pa=pmax_scale_pa,
            )
            torch.cuda.synchronize(device)
            call_seconds.append(time.monotonic() - started)
            predictions[frame].copy_(projected[0])
            records.append({"call": frame + 1, **stats})
            current = projected.to(device=device, dtype=DTYPE)
            current_pmax_pa = projected[:, PMAX_CHANNEL] * projected.new_tensor(
                float(pmax_scale_pa)
            ) + projected.new_tensor(float(pmax_mean_pa))
    return predictions, nonfinite_call, records, call_seconds


def _aggregate_projection(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    corrected = sum(int(record["corrected_cells"]) for record in records)
    total = sum(int(record["total_cells"]) for record in records)
    correction_sum = sum(float(record["correction_sum_pa"]) for record in records)
    return {
        "call_count": len(records),
        "calls_with_correction": sum(
            int(record["corrected_cells"] > 0) for record in records
        ),
        "corrected_cells": corrected,
        "total_pMax_cell_calls": total,
        "corrected_fraction": corrected / total if total else 0.0,
        "correction_sum_pa": correction_sum,
        "correction_mean_active_pa": correction_sum / corrected if corrected else 0.0,
        "correction_max_pa": max(
            (float(record["correction_max_pa"]) for record in records), default=0.0
        ),
        "encoded_floor_nextafter_updates": sum(
            int(record["encoded_floor_nextafter_updates"]) for record in records
        ),
        "encoded_floor_max_nextafter_steps": max(
            (int(record["encoded_floor_max_nextafter_steps"]) for record in records),
            default=0,
        ),
        "all_other_channels_bitwise_unchanged": all(
            bool(record["other_channels_bitwise_unchanged"]) for record in records
        ),
        "all_projected_pMax_nondecreasing": all(
            bool(record["projected_pMax_nondecreasing"]) for record in records
        ),
        "by_call": list(records),
    }


def mean_non_pmax_grouped_npe(summary: Mapping[str, Any]) -> float:
    grouped = summary.get("npe_group_by_call")
    if not isinstance(grouped, Mapping) or set(grouped) != {
        "T",
        "chem",
        "p",
        "rho",
        "u",
    }:
        raise ValueError("view summary lacks the frozen PlanarDet groups")
    values = [grouped[name] for name in ("T", "chem", "rho", "u")]
    if not all(isinstance(value, list) and value for value in values):
        raise ValueError("non-pMax group values must be nonempty lists")
    lengths = {len(value) for value in values}
    if len(lengths) != 1:
        raise ValueError("non-pMax group histories must have equal length")
    per_call: list[float] = []
    for items in zip(*values, strict=True):
        if not all(isinstance(value, (int, float)) for value in items):
            raise ValueError("non-pMax group histories must be finite numeric values")
        total = sum(float(value) for value in items)
        if not math.isfinite(total):
            raise ValueError("non-pMax grouped error must be finite")
        per_call.append(total)
    return sum(per_call) / len(per_call)


def classify_non_pmax_effect(ratio: float, *, complete: bool) -> str:
    if not complete or not math.isfinite(ratio) or ratio < 0.0:
        return "not_interpretable"
    if ratio <= 1.0 - MATERIAL_EFFECT_RATIO:
        return "materially_helpful"
    if ratio >= 1.0 + MATERIAL_EFFECT_RATIO:
        return "materially_harmful"
    if 1.0 - NEAR_NULL_RATIO <= ratio <= 1.0 + NEAR_NULL_RATIO:
        return "near_null"
    return "small_or_inconclusive"


def _view_comparison(
    baseline: Mapping[str, Any], projected: Mapping[str, Any]
) -> dict[str, Any]:
    raw_npe = float(baseline["realm_npe_mean"])
    projected_npe = float(projected["realm_npe_mean"])
    raw_non_pmax = mean_non_pmax_grouped_npe(baseline)
    projected_non_pmax = mean_non_pmax_grouped_npe(projected)
    return {
        "realm_npe_mean": {
            "baseline": raw_npe,
            "projected": projected_npe,
            "ratio": projected_npe / raw_npe,
            "delta": projected_npe - raw_npe,
        },
        "non_pMax_grouped_npe_mean": {
            "baseline": raw_non_pmax,
            "projected": projected_non_pmax,
            "ratio": projected_non_pmax / raw_non_pmax,
            "delta": projected_non_pmax - raw_non_pmax,
        },
        "decoded_correlation_case_first": {
            "baseline": baseline["decoded_correlation_case_first"],
            "projected": projected["decoded_correlation_case_first"],
        },
        "admissible_call_count": {
            "baseline": baseline["admissible_call_count"],
            "projected": projected["admissible_call_count"],
        },
        "bounded_call_count": {
            "baseline": baseline["bounded_call_count"],
            "projected": projected["bounded_call_count"],
        },
        "pMax_decrease_from_input_count": {
            "baseline": baseline["pMax_decrease_from_input_count"],
            "projected": projected["pMax_decrease_from_input_count"],
        },
    }


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    _validate_startup_environment()
    training_source = build_source_manifest(
        REPO_ROOT,
        entrypoints=EXECUTABLE_ENTRYPOINTS,
    )
    if training_source["canonical_payload_sha256"] != BASELINE_SOURCE_DIGEST:
        raise ValueError("current frozen training source digest differs")
    projection_source = build_projection_source_manifest()
    preregistration = _load_json(args.preregistration)
    _validate_projection_preregistration(preregistration)
    baseline_result = _validate_baseline_evaluation(
        args.baseline_evaluation_dir,
        current_source_manifest=training_source,
    )
    contract, training_status, training_inputs, training_config = (
        _validate_closed_training_tree(
            args.training_dir,
            current_source_manifest=training_source,
        )
    )
    if contract.run_id != BASELINE_RUN_ID:
        raise ValueError("closed training run differs from projection baseline")

    manifest_payload = _load_json(args.manifest)
    if canonical_json_sha256(manifest_payload) != contract.open_manifest_payload_sha256:
        raise ValueError("open manifest payload differs from preregistration")
    repository, revision, entries = parse_manifest_payload(manifest_payload)
    manifest_summary = validate_planardet_open_manifest(repository, revision, entries)
    inventory = validate_local_open_tree(args.data_root, entries)
    metadata = load_planardet_metadata(args.data_root / "data" / "data.npz")
    if (
        manifest_summary["manifest_sha256"] != PLANARDET_OPEN_MANIFEST_SHA256
        or metadata.train_groups != PLANARDET_TRAIN_GROUPS
        or metadata.val_groups != PLANARDET_VAL_GROUPS
        or training_inputs.get("open_manifest_payload_sha256")
        != canonical_json_sha256(manifest_payload)
        or training_inputs.get("normalizer_arrays_sha256")
        != contract.normalizer_arrays_sha256
        or training_inputs.get("test_object_opened") is not False
    ):
        raise ValueError("projection inputs differ from closed training inputs")
    bundle = load_normalizer_bundle(
        args.normalizer_arrays,
        expected_sha256=contract.normalizer_arrays_sha256,
        expected_coordinates_yx=metadata.canonical_coords_yx,
    )
    normalizer = bundle.normalizer(1, dtype=DTYPE)
    validation_normalized, validation_native_batched = load_normalized_trajectories(
        args.data_root,
        split="val",
        groups=PLANARDET_VAL_GROUPS,
        normalizer=normalizer,
        retain_native=True,
    )
    if validation_native_batched is None:
        raise RuntimeError("validation native truth was not retained")
    controls = validation_control_summary(validation_normalized[0])
    validate_frozen_controls(controls, contract)
    if controls != baseline_result["controls"]:
        raise ValueError("projection controls differ from baseline evaluation")

    checkpoint = torch.load(
        args.training_dir / "best.pt", map_location="cpu", weights_only=False
    )
    if not isinstance(checkpoint, Mapping):
        raise TypeError("retained checkpoint root must be a mapping")
    if sha256_file(args.training_dir / "best.pt") != BASELINE_CHECKPOINT_SHA256:
        raise ValueError("retained checkpoint file hash differs")
    model_state = _validate_best_checkpoint(
        checkpoint,
        contract=contract,
        status=training_status,
        input_manifest=training_inputs,
        config=training_config,
        training_dir=args.training_dir,
        bundle=bundle,
    )

    _configure_determinism(contract.seed)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("projection evaluation requires one visible CUDA device")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("projection evaluation requires bfloat16 autocast support")
    device = torch.device(DEVICE_NAME)
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    runtime = {
        key: value
        for key, value in _runtime_manifest(device, seed=contract.seed).items()
        if key != "canonical_payload_sha256"
    }
    runtime.update(
        {
            "schema": RUNTIME_SCHEMA,
            "experiment_id": EXPERIMENT_ID,
            "projection_source_manifest_digest": projection_source[
                "canonical_payload_sha256"
            ],
        }
    )
    runtime["canonical_payload_sha256"] = canonical_json_sha256(runtime)

    geometry = build_realm_regular_grid_geometry(
        metadata.canonical_coords_yx,
        domain_lengths_xy=DOMAIN_LENGTHS_XY,
        released_coordinate_order=CANONICAL_COORDINATE_ORDER,
    )
    model = RealmRegularGridPCNO(
        config=RealmPCNOConfig(
            channels=CHANNELS,
            mode_counts_xy=MODE_COUNTS_XY,
            layers=(contract.width,) * 5,
            fc_dim=FC_DIM,
            zero_initialize_head=True,
        ),
        geometry=geometry,
    ).to(device=device, dtype=DTYPE)
    model.load_state_dict(model_state, strict=True)
    parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if parameter_count != EXPECTED_PARAMETER_COUNTS[contract.width]:
        raise RuntimeError("projection PCNO parameter count differs")
    coordinates = bundle.canonical_coordinates_yx.unsqueeze(0).to(
        device=device, dtype=DTYPE
    )
    validation_truth = validation_normalized[0]
    validation_native = validation_native_batched[0]
    boundary_mask = planardet_boundary_mask(
        torch.from_numpy(metadata.x.copy()), torch.from_numpy(metadata.y.copy())
    )
    baseline_teacher = np.load(
        args.baseline_evaluation_dir / "teacher_prediction_normalized.npy",
        allow_pickle=False,
        mmap_mode="r",
    )
    baseline_free = np.load(
        args.baseline_evaluation_dir / "free_prediction_normalized.npy",
        allow_pickle=False,
        mmap_mode="r",
    )
    pmax_mean_pa = float(bundle.mean[PMAX_CHANNEL].item())
    pmax_scale_pa = float(bundle.scale[PMAX_CHANNEL].item())

    started_at = time.monotonic()
    teacher_predictions, teacher_records, teacher_seconds = (
        _predict_projected_truth_inputs(
            model,
            validation_truth,
            validation_native,
            coordinates,
            baseline_teacher,
            device=device,
            pmax_mean_pa=pmax_mean_pa,
            pmax_scale_pa=pmax_scale_pa,
        )
    )
    teacher_summary, teacher_decoded = _view_summary(
        teacher_predictions,
        validation_truth[1:],
        validation_native[1:],
        validation_native[:-1],
        bundle=bundle,
        boundary_mask=boundary_mask,
    )
    if teacher_summary is None:
        raise RuntimeError("projected truth-input summary is absent")
    teacher_gate = competence_gate(teacher_summary, contract)

    free_predictions, free_nonfinite_call, free_records, free_seconds = (
        _predict_projected_free_recurrence(
            model,
            validation_truth[0],
            validation_native[0, PMAX_CHANNEL],
            coordinates,
            baseline_free,
            device=device,
            pmax_mean_pa=pmax_mean_pa,
            pmax_scale_pa=pmax_scale_pa,
        )
    )
    free_decoded_for_current = normalizer.decode(
        free_predictions, inverse_domain_policy="nan"
    )
    free_current_native = torch.cat(
        (validation_native[:1], free_decoded_for_current[:-1]), dim=0
    )
    free_summary, free_decoded = _view_summary(
        free_predictions,
        validation_truth[1 : free_predictions.shape[0] + 1],
        validation_native[1 : free_predictions.shape[0] + 1],
        free_current_native,
        bundle=bundle,
        boundary_mask=boundary_mask,
    )
    if free_summary is None:
        raise RuntimeError("projected free recurrence has no finite accepted call")

    teacher_projection = _aggregate_projection(teacher_records)
    free_projection = _aggregate_projection(free_records)
    baseline_teacher_summary = baseline_result["views"]["ordered_teacher_forced"][
        "summary"
    ]
    baseline_free_summary = baseline_result["views"]["free_recurrence"]["summary"]
    teacher_comparison = _view_comparison(baseline_teacher_summary, teacher_summary)
    free_comparison = _view_comparison(baseline_free_summary, free_summary)
    full_horizon = free_predictions.shape[0] == VALIDATION_HORIZON
    effect_class = classify_non_pmax_effect(
        float(free_comparison["non_pMax_grouped_npe_mean"]["ratio"]),
        complete=full_horizon,
    )
    structure = {
        "projected_truth_input": _structure_summary(
            teacher_decoded,
            validation_native,
            metadata_times=metadata.times,
            x=metadata.x,
            y=metadata.y,
        ),
        "projected_free_recurrence": _structure_summary(
            free_decoded,
            validation_native,
            metadata_times=metadata.times,
            x=metadata.x,
            y=metadata.y,
        ),
        "baseline": baseline_result["structure"],
    }

    if _is_within(args.output_dir, args.baseline_evaluation_dir) or _is_within(
        args.baseline_evaluation_dir, args.output_dir
    ):
        raise ValueError("projection output and baseline evaluation must be disjoint")
    _prepare_output_directory(
        args.output_dir,
        training_dir=args.training_dir,
        data_root=args.data_root,
    )
    teacher_path = args.output_dir / "projected_teacher_prediction_normalized.npy"
    free_path = args.output_dir / "projected_free_prediction_normalized.npy"
    _write_npy(teacher_path, teacher_predictions)
    _write_npy(free_path, free_predictions)
    arrays = {
        teacher_path.name: {
            "sha256": sha256_file(teacher_path),
            "shape": list(teacher_predictions.shape),
            "dtype": str(teacher_predictions.numpy().dtype),
        },
        free_path.name: {
            "sha256": sha256_file(free_path),
            "shape": list(free_predictions.shape),
            "dtype": str(free_predictions.numpy().dtype),
        },
    }

    closure = {
        "closed_training_verified": True,
        "frozen_training_source_verified": True,
        "projection_source_verified": True,
        "preregistration_verified": True,
        "baseline_evaluation_verified": True,
        "open_tree_verified": len(inventory) == len(entries),
        "teacher_raw_replay_exact_49_calls": len(teacher_records) == VALIDATION_HORIZON,
        "free_raw_call_1_replay_exact": len(free_records) >= 1,
        "projection_changes_only_pMax": bool(
            teacher_projection["all_other_channels_bitwise_unchanged"]
            and free_projection["all_other_channels_bitwise_unchanged"]
        ),
        "teacher_projection_count_matches_baseline_decrease_count": (
            teacher_projection["corrected_cells"]
            == baseline_teacher_summary["pMax_decrease_from_input_count"]
        ),
        "projected_teacher_pMax_nondecreasing": bool(
            teacher_summary["all_pMax_nondecreasing_from_input"]
        ),
        "projected_free_pMax_nondecreasing": bool(
            free_summary["all_pMax_nondecreasing_from_input"]
        ),
        "projected_teacher_complete_finite_H49": bool(
            teacher_summary["call_count"] == VALIDATION_HORIZON
            and teacher_summary["all_normalized_finite"]
            and teacher_summary["all_decoded_finite"]
        ),
        "projected_free_censoring_recorded": free_predictions.shape[0]
        == (
            VALIDATION_HORIZON
            if free_nonfinite_call is None
            else free_nonfinite_call - 1
        ),
        "projected_free_complete_finite_H49": bool(
            free_predictions.shape[0] == VALIDATION_HORIZON
            and free_summary["all_normalized_finite"]
            and free_summary["all_decoded_finite"]
        ),
        "prediction_arrays_persisted": True,
        "test_object_opened": False,
    }
    closure["all_gates_pass"] = (
        all(value for key, value in closure.items() if key != "test_object_opened")
        and closure["test_object_opened"] is False
    )
    truth_no_harm = bool(
        teacher_comparison["realm_npe_mean"]["ratio"] <= TRUTH_INPUT_NO_HARM_RATIO
    )
    diagnostic_interpretation_allowed = bool(
        teacher_gate["all_gates_pass"]
        and truth_no_harm
        and closure["all_gates_pass"]
        and full_horizon
        and free_summary["all_normalized_finite"]
        and free_summary["all_decoded_finite"]
    )
    result: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "baseline_run_id": BASELINE_RUN_ID,
        "checkpoint": {
            "relative_path": "best.pt",
            "sha256": BASELINE_CHECKPOINT_SHA256,
            "completed_step": BASELINE_BEST_STEP,
            "model_state_sha256": checkpoint["model_state_sha256"],
        },
        "bindings": {
            "training_source_manifest_digest": BASELINE_SOURCE_DIGEST,
            "projection_source_manifest_digest": projection_source[
                "canonical_payload_sha256"
            ],
            "projection_preregistration_payload_digest": preregistration[
                "canonical_payload_sha256"
            ],
            "projection_preregistration_file_sha256": sha256_file(args.preregistration),
            "baseline_result_sha256": BASELINE_RESULT_SHA256,
            "baseline_final_manifest_sha256": BASELINE_FINAL_MANIFEST_SHA256,
            "normalizer_arrays_sha256": bundle.source_sha256,
            "open_manifest_payload_sha256": canonical_json_sha256(manifest_payload),
        },
        "intervention": preregistration["intervention"],
        "controls": controls,
        "views": {
            "projected_truth_input": {
                "aggregation": "49 truth-input calls with causal pMax projection",
                "valid_length": teacher_predictions.shape[0],
                "call_seconds": teacher_seconds,
                "summary": teacher_summary,
            },
            "projected_free_recurrence": {
                "aggregation": "one projected-feedback trajectory from released frame zero",
                "valid_length": free_predictions.shape[0],
                "nonfinite_proposal_call": free_nonfinite_call,
                "call_seconds": free_seconds,
                "summary": free_summary,
            },
        },
        "projection_activity": {
            "truth_input": teacher_projection,
            "free_recurrence": free_projection,
        },
        "comparison_to_unprojected_baseline": {
            "truth_input": teacher_comparison,
            "free_recurrence": free_comparison,
            "projected_free_minus_projected_truth_npe_by_call": (
                _free_minus_teacher(free_summary, teacher_summary)
            ),
        },
        "structure": structure,
        "projected_truth_input_competence_gate": teacher_gate,
        "truth_input_npe_no_harm": truth_no_harm,
        "evaluator_closure": closure,
        "projection_diagnostic_interpretation_allowed": (
            diagnostic_interpretation_allowed
        ),
        "non_pMax_recurrence_effect_classification": (
            effect_class if diagnostic_interpretation_allowed else "not_interpretable"
        ),
        "prediction_arrays": arrays,
        "timing_seconds": {
            "evaluation": time.monotonic() - started_at,
            "teacher_total": sum(teacher_seconds),
            "free_total": sum(free_seconds),
        },
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "parameter_count": parameter_count,
        "test_object_opened": False,
        "anti_claims": preregistration["anti_claims"],
    }
    result["canonical_payload_sha256"] = canonical_json_sha256(result)
    _write_json(args.output_dir / "result.json", result)
    _write_json(args.output_dir / "training_source_manifest.json", training_source)
    _write_json(args.output_dir / "projection_source_manifest.json", projection_source)
    _write_json(args.output_dir / "runtime_manifest.json", runtime)
    _write_json(args.output_dir / "preregistration.json", preregistration)
    hashed_files = (
        teacher_path.name,
        free_path.name,
        "result.json",
        "training_source_manifest.json",
        "projection_source_manifest.json",
        "runtime_manifest.json",
        "preregistration.json",
    )
    final_manifest = {
        "schema": FINAL_MANIFEST_SCHEMA,
        "files": {name: sha256_file(args.output_dir / name) for name in hashed_files},
        "self_hash_excluded": True,
        "test_object_opened": False,
    }
    _write_json(args.output_dir / "final_hash_manifest.json", final_manifest)
    summary = {
        "schema": SUMMARY_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "projected_truth_input_competence_pass": teacher_gate["all_gates_pass"],
        "truth_input_npe_no_harm": truth_no_harm,
        "evaluator_closure_pass": closure["all_gates_pass"],
        "projection_diagnostic_interpretation_allowed": (
            diagnostic_interpretation_allowed
        ),
        "non_pMax_recurrence_effect_classification": result[
            "non_pMax_recurrence_effect_classification"
        ],
        "projected_teacher_valid_length": teacher_predictions.shape[0],
        "projected_free_valid_length": free_predictions.shape[0],
        "result_sha256": sha256_file(args.output_dir / "result.json"),
        "final_hash_manifest_sha256": sha256_file(
            args.output_dir / "final_hash_manifest.json"
        ),
        "test_object_opened": False,
    }
    _write_json(args.output_dir / "summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary = run_evaluation(args)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
