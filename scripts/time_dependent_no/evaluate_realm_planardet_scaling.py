"""Evaluate retained D093 PlanarDet scaling checkpoints on open validation.

This entry point accepts one D093 training directory at a time.  A training
attempt may be complete or may have stopped at the registered decoded-finiteness
gate, but its retained ``best.pt`` must close against the history, frozen
configuration, source snapshot, normalizer, and run provenance.  Evaluation
uses the released validation trajectory only; no test-path argument exists.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
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
    _accepted_prefix,
    _configure_determinism,
    _free_minus_teacher,
    _predict_free_recurrence,
    _predict_truth_inputs,
    _prepare_output_directory,
    _structure_summary,
    _view_summary,
    _write_json,
    _write_npy,
)
from scripts.time_dependent_no.train_realm_planardet_scaling import (
    BEST_CHECKPOINT_SCHEMA,
    DTYPE,
    _build_model,
    _coordinates_for_architecture,
    _load_json,
    _validate_release_and_normalizer,
)
from utility.time_dependent_no.realm_benchmark import canonical_json_sha256
from utility.time_dependent_no.realm_planardet import (
    PLANARDET_VAL_GROUPS,
    sha256_file,
)
from utility.time_dependent_no.realm_planardet_artifacts import (
    build_source_manifest,
    structured_state_sha256,
)
from utility.time_dependent_no.realm_planardet_runtime import (
    VALIDATION_HORIZON,
    competence_gate,
    load_normalized_trajectories,
    planardet_boundary_mask,
    validate_frozen_controls,
    validation_control_summary,
)
from utility.time_dependent_no.realm_planardet_scaling import (
    EXPECTED_PARAMETER_COUNTS,
    SCALING_SOURCE_PATHS,
    PlanarDetScalingContract,
)

DEVICE_NAME = "cuda:0"
EVALUATOR_PATH = "scripts/time_dependent_no/evaluate_realm_planardet_scaling.py"
BASE_EVALUATOR_PATH = "scripts/time_dependent_no/evaluate_realm_planardet_pcno.py"
EVALUATION_SOURCE_PATHS = tuple(
    dict.fromkeys((*SCALING_SOURCE_PATHS, BASE_EVALUATOR_PATH, EVALUATOR_PATH))
)
COMPLETE_STATUS_SCHEMA = "w26_l4_planardet_arch_data_scaling_status_v1"
COMPLETE_FINAL_MANIFEST_SCHEMA = (
    "w26_l4_planardet_arch_data_scaling_final_hash_manifest_v1"
)
DECODE_FAILURE_TEXT = "teacher validation produced a nonfinite decoded proposal"

COMMON_TRAINING_FILES = frozenset(
    {
        "best.pt",
        "config.json",
        "history.json",
        "input_manifest.json",
        "last.pt",
        "preregistration.json",
        "runtime_manifest.json",
        "source_manifest.json",
        "status.json",
    }
)
COMPLETE_TRAINING_FILES = COMMON_TRAINING_FILES | {
    "final_hash_manifest.json",
    "summary.json",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--training-log", type=Path, required=True)
    parser.add_argument("--training-exit-file", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--normalizer-arrays", type=Path, required=True)
    parser.add_argument("--data-audit-final-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _validate_signed_payload(payload: Mapping[str, Any], *, name: str) -> None:
    digest = payload.get("canonical_payload_sha256")
    unsigned = {
        key: value for key, value in payload.items() if key != "canonical_payload_sha256"
    }
    if digest != canonical_json_sha256(unsigned):
        raise ValueError(f"{name} canonical payload digest differs")


def _regular_files(directory: Path) -> set[str]:
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("training directory must be an existing regular directory")
    names: set[str] = set()
    for path in directory.iterdir():
        if path.is_symlink() or not path.is_file():
            raise ValueError("training directory contains a non-regular object")
        if any(part.casefold() == "test" for part in path.relative_to(directory).parts):
            raise ValueError("training directory contains a sealed test path")
        names.add(path.name)
    return names


def _history_best(history: Mapping[str, Any]) -> tuple[list[Mapping[str, Any]], Mapping[str, Any]]:
    rows = history.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("training history must contain rows")
    if any(not isinstance(row, Mapping) for row in rows):
        raise TypeError("training history rows must be mappings")
    steps = [row.get("completed_step") for row in rows]
    if (
        steps[0] != 1
        or steps != sorted(set(steps))
        or any(not isinstance(step, int) for step in steps)
    ):
        raise ValueError("training history step sequence differs")
    eligible = [
        row
        for row in rows
        if row.get("checkpoint_eligible") is True
        and isinstance((row.get("validation") or {}).get("realm_npe_mean"), (int, float))
    ]
    if not eligible:
        raise ValueError("training history has no eligible checkpoint")
    best = min(eligible, key=lambda row: float(row["validation"]["realm_npe_mean"]))
    return rows, best


def _validate_complete_manifest(training_dir: Path) -> None:
    payload = _load_json(training_dir / "final_hash_manifest.json")
    files = payload.get("files")
    expected = COMPLETE_TRAINING_FILES - {"final_hash_manifest.json", "summary.json"}
    if (
        payload.get("schema") != COMPLETE_FINAL_MANIFEST_SCHEMA
        or payload.get("self_hash_excluded") is not True
        or payload.get("test_object_opened") is not False
        or not isinstance(files, Mapping)
        or set(files) != expected
    ):
        raise ValueError("completed training final manifest differs")
    for name, digest in files.items():
        if sha256_file(training_dir / name) != digest:
            raise ValueError(f"completed training hash differs: {name}")


def _validate_training_attempt(
    args: argparse.Namespace,
    *,
    bundle_state_sha256: str,
) -> tuple[
    PlanarDetScalingContract,
    Mapping[str, Any],
    Mapping[str, Any],
    Mapping[str, Any],
    Mapping[str, Any],
    dict[str, Any],
]:
    files = _regular_files(args.training_dir)
    complete = files == COMPLETE_TRAINING_FILES
    if not complete and files != COMMON_TRAINING_FILES:
        raise ValueError("training directory inventory differs from D093 schemas")

    exit_text = args.training_exit_file.read_text(encoding="utf-8").strip()
    log_text = args.training_log.read_text(encoding="utf-8")
    if complete:
        if exit_text != "0":
            raise ValueError("complete training attempt does not have exit code zero")
        _validate_complete_manifest(args.training_dir)
    elif exit_text != "2" or DECODE_FAILURE_TEXT not in log_text:
        raise ValueError("incomplete training attempt lacks the registered decode failure")

    preregistration = _load_json(args.training_dir / "preregistration.json")
    contract = PlanarDetScalingContract.from_payload(preregistration)
    config = _load_json(args.training_dir / "config.json")
    input_manifest = _load_json(args.training_dir / "input_manifest.json")
    source_manifest = _load_json(args.training_dir / "source_manifest.json")
    runtime_manifest = _load_json(args.training_dir / "runtime_manifest.json")
    status = _load_json(args.training_dir / "status.json")
    history = _load_json(args.training_dir / "history.json")
    for name, payload in (
        ("training config", config),
        ("training inputs", input_manifest),
        ("training source", source_manifest),
        ("training runtime", runtime_manifest),
    ):
        _validate_signed_payload(payload, name=name)
    if config != contract.frozen_training_config():
        raise ValueError("training config differs from preregistration")
    expected_training_source = build_source_manifest(
        REPO_ROOT, entrypoints=SCALING_SOURCE_PATHS
    )
    if source_manifest != expected_training_source:
        raise ValueError("current training sources differ from D093 snapshot")
    if input_manifest.get("test_object_opened") is not False:
        raise ValueError("training input manifest does not preserve the sealed test")

    rows, best = _history_best(history)
    best_step = int(best["completed_step"])
    best_score = float(best["validation"]["realm_npe_mean"])
    checkpoint = torch.load(
        args.training_dir / "best.pt", map_location="cpu", weights_only=False
    )
    if not isinstance(checkpoint, Mapping):
        raise TypeError("best checkpoint root must be a mapping")
    provenance = {
        "config_digest": str(config["canonical_payload_sha256"]),
        "input_digest": str(input_manifest["canonical_payload_sha256"]),
        "source_digest": str(source_manifest["canonical_payload_sha256"]),
        "runtime_digest": str(runtime_manifest["canonical_payload_sha256"]),
    }
    run_signature = canonical_json_sha256(provenance)
    model_state = checkpoint.get("model_state")
    normalizer_state = checkpoint.get("normalizer_state")
    if (
        checkpoint.get("schema") != BEST_CHECKPOINT_SCHEMA
        or checkpoint.get("run_id") != contract.run_id
        or checkpoint.get("run_signature") != run_signature
        or checkpoint.get("architecture") != contract.architecture
        or checkpoint.get("completed_step") != best_step
        or checkpoint.get("selection_value") != best_score
        or checkpoint.get("selection_metric") != "teacher_forced_realm_npe_mean"
        or checkpoint.get("model_config") != config.get("model")
        or checkpoint.get("provenance") != provenance
        or checkpoint.get("resume_supported") is not False
        or checkpoint.get("test_object_opened") is not False
        or not isinstance(model_state, Mapping)
        or structured_state_sha256(model_state) != checkpoint.get("model_state_sha256")
        or not isinstance(normalizer_state, Mapping)
        or structured_state_sha256(normalizer_state)
        != checkpoint.get("normalizer_state_sha256")
        or checkpoint.get("normalizer_state_sha256") != bundle_state_sha256
    ):
        raise ValueError("retained best checkpoint identity or provenance differs")

    if complete:
        summary = _load_json(args.training_dir / "summary.json")
        if (
            status.get("schema") != COMPLETE_STATUS_SCHEMA
            or status.get("complete") is not True
            or status.get("completed_step") != contract.total_steps
            or status.get("best_step") != best_step
            or status.get("best_teacher_forced_realm_npe_mean") != best_score
            or status.get("test_object_opened") is not False
            or summary.get("final_hash_manifest_sha256")
            != sha256_file(args.training_dir / "final_hash_manifest.json")
        ):
            raise ValueError("completed training status differs from history")
    elif (
        status.get("complete") is not False
        or status.get("test_object_opened") is not False
        or int(rows[-1]["completed_step"]) >= contract.total_steps
    ):
        raise ValueError("failed training status or history differs")

    attempt = {
        "complete": complete,
        "exit_code": int(exit_text),
        "termination": "completed" if complete else "decoded_validation_nonfinite",
        "history_last_step": int(rows[-1]["completed_step"]),
        "history_row_count": len(rows),
        "best_step": best_step,
        "best_teacher_forced_realm_npe_mean": best_score,
    }
    return contract, input_manifest, checkpoint, model_state, history, attempt


def _runtime_manifest(device: torch.device, *, seed: int) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(device)
    payload: dict[str, Any] = {
        "schema": "w26_l4_d093_best_open_validation_runtime_v1",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "device_name": properties.name,
        "device_total_memory": properties.total_memory,
        "bf16_supported": torch.cuda.is_bf16_supported(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "seed": seed,
    }
    payload["canonical_payload_sha256"] = canonical_json_sha256(payload)
    return payload


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    manifest_payload = _load_json(args.manifest)
    provisional_contract = PlanarDetScalingContract.from_payload(
        _load_json(args.training_dir / "preregistration.json")
    )
    (
        _manifest,
        _repository,
        _revision,
        inventory,
        metadata,
        bundle,
    ) = _validate_release_and_normalizer(
        manifest_path=args.manifest,
        data_root=args.data_root,
        normalizer_arrays_path=args.normalizer_arrays,
        data_audit_final_manifest_path=args.data_audit_final_manifest,
        expected_open_manifest_payload_sha256=(
            provisional_contract.open_manifest_payload_sha256
        ),
        expected_normalizer_arrays_sha256=(
            provisional_contract.normalizer_arrays_sha256
        ),
        expected_data_audit_final_manifest_sha256=(
            provisional_contract.data_audit_final_manifest_sha256
        ),
    )
    if canonical_json_sha256(manifest_payload) != provisional_contract.open_manifest_payload_sha256:
        raise ValueError("open manifest payload differs from preregistration")
    bundle_state_sha256 = structured_state_sha256(bundle.checkpoint_state())
    contract, training_inputs, checkpoint, model_state, _history, attempt = (
        _validate_training_attempt(args, bundle_state_sha256=bundle_state_sha256)
    )
    if (
        training_inputs.get("open_manifest_payload_sha256")
        != canonical_json_sha256(manifest_payload)
        or training_inputs.get("normalizer_arrays_sha256")
        != contract.normalizer_arrays_sha256
        or training_inputs.get("test_object_opened") is not False
    ):
        raise ValueError("evaluation inputs differ from training inputs")

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

    _configure_determinism(contract.seed)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("evaluation requires exactly one visible CUDA device")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("evaluation requires bfloat16 autocast support")
    device = torch.device(DEVICE_NAME)
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    runtime = _runtime_manifest(device, seed=contract.seed)

    model = _build_model(contract.architecture, metadata).to(device=device, dtype=DTYPE)
    model.load_state_dict(model_state, strict=True)
    parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if parameter_count != EXPECTED_PARAMETER_COUNTS[contract.architecture]:
        raise RuntimeError("evaluation parameter count differs")
    coordinates = _coordinates_for_architecture(
        contract.architecture, bundle, device=device
    )
    validation_truth = validation_normalized[0]
    validation_native = validation_native_batched[0]
    boundary_mask = planardet_boundary_mask(
        torch.from_numpy(metadata.x.copy()), torch.from_numpy(metadata.y.copy())
    )

    started = time.monotonic()
    teacher_predictions, teacher_finite, teacher_seconds = _predict_truth_inputs(
        model, validation_truth, coordinates, device=device
    )
    teacher_prefix = _accepted_prefix(teacher_finite)
    fresh_summary, _ = _view_summary(
        teacher_predictions,
        validation_truth[1:],
        validation_native[1:],
        validation_native[:-1],
        bundle=bundle,
        boundary_mask=boundary_mask,
    )
    ordered_summary, teacher_decoded = _view_summary(
        teacher_predictions[:teacher_prefix],
        validation_truth[1 : teacher_prefix + 1],
        validation_native[1 : teacher_prefix + 1],
        validation_native[:teacher_prefix],
        bundle=bundle,
        boundary_mask=boundary_mask,
    )
    free_predictions, free_nonfinite_call, free_seconds = _predict_free_recurrence(
        model, validation_truth[0], coordinates, device=device
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
    if fresh_summary is None:
        raise RuntimeError("fresh-pair evaluation has no attempted calls")
    truth_gate = competence_gate(fresh_summary, contract)
    structure = {
        "ordered_teacher_forced": _structure_summary(
            teacher_decoded,
            validation_native,
            metadata_times=metadata.times,
            x=metadata.x,
            y=metadata.y,
        ),
        "free_recurrence": _structure_summary(
            free_decoded,
            validation_native,
            metadata_times=metadata.times,
            x=metadata.x,
            y=metadata.y,
        ),
    }

    _prepare_output_directory(
        args.output_dir, training_dir=args.training_dir, data_root=args.data_root
    )
    teacher_path = args.output_dir / "teacher_prediction_normalized.npy"
    free_path = args.output_dir / "free_prediction_normalized.npy"
    _write_npy(teacher_path, teacher_predictions)
    _write_npy(free_path, free_predictions)
    source_manifest = build_source_manifest(
        REPO_ROOT, entrypoints=EVALUATION_SOURCE_PATHS
    )
    closure = {
        "training_attempt_verified": True,
        "training_source_manifest_matches": True,
        "best_checkpoint_verified": True,
        "open_tree_verified": len(inventory) > 0,
        "fresh_pairs_all_attempted": len(teacher_finite) == VALIDATION_HORIZON,
        "teacher_censoring_recorded": teacher_prefix == _accepted_prefix(teacher_finite),
        "free_censoring_recorded": free_predictions.shape[0]
        == (VALIDATION_HORIZON if free_nonfinite_call is None else free_nonfinite_call - 1),
        "prediction_arrays_persisted": True,
        "test_object_opened": False,
    }
    closure["all_gates_pass"] = all(
        value for key, value in closure.items() if key != "test_object_opened"
    ) and closure["test_object_opened"] is False
    result: dict[str, Any] = {
        "schema": "w26_l4_d093_best_open_validation_evaluation_v1",
        "run_id": contract.run_id,
        "architecture": contract.architecture,
        "train_trajectory_count": contract.train_trajectory_count,
        "training_attempt": attempt,
        "checkpoint": {
            "relative_path": "best.pt",
            "sha256": sha256_file(args.training_dir / "best.pt"),
            "completed_step": checkpoint["completed_step"],
            "model_state_sha256": checkpoint["model_state_sha256"],
            "selection_value": checkpoint["selection_value"],
        },
        "validation_case": PLANARDET_VAL_GROUPS[0],
        "controls": controls,
        "views": {
            "fresh_pairs": {
                "aggregation": "49 independent truth-input adjacent pairs",
                "attempted_calls": len(teacher_finite),
                "finite_call_mask": teacher_finite,
                "call_seconds": teacher_seconds,
                "summary": fresh_summary,
            },
            "ordered_teacher_forced": {
                "aggregation": "truth restored before every call",
                "valid_length": teacher_prefix,
                "censored_from_call": (
                    teacher_prefix + 1 if teacher_prefix < VALIDATION_HORIZON else None
                ),
                "summary": ordered_summary,
            },
            "free_recurrence": {
                "aggregation": "initialized only from released frame zero",
                "valid_length": free_predictions.shape[0],
                "nonfinite_proposal_call": free_nonfinite_call,
                "call_seconds": free_seconds,
                "summary": free_summary,
            },
        },
        "free_minus_teacher_npe_by_call": _free_minus_teacher(
            free_summary, ordered_summary
        ),
        "structure": structure,
        "truth_input_competence_gate": truth_gate,
        "evaluator_closure": closure,
        "mechanism_interpretation_allowed": bool(
            truth_gate["all_gates_pass"] and closure["all_gates_pass"]
        ),
        "prediction_arrays": {
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
        },
        "timing_seconds": {
            "evaluation": time.monotonic() - started,
            "teacher_total": sum(teacher_seconds),
            "free_total": sum(free_seconds),
        },
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "parameter_count": parameter_count,
        "test_object_opened": False,
        "anti_claims": [
            "open validation is not sealed test evidence",
            "retaining a best checkpoint does not make an interrupted training attempt complete",
            "fresh-versus-free differences do not identify a training or architecture cause",
            "PCFNO has no direct REALM-paper counterpart",
            "the residual FFNO contract is not the paper's direct-state training contract",
        ],
    }
    result["canonical_payload_sha256"] = canonical_json_sha256(result)
    _write_json(args.output_dir / "result.json", result)
    _write_json(args.output_dir / "source_manifest.json", source_manifest)
    _write_json(args.output_dir / "runtime_manifest.json", runtime)
    final_manifest = {
        "schema": "w26_l4_d093_best_open_validation_final_hash_manifest_v1",
        "files": {
            name: sha256_file(args.output_dir / name)
            for name in (
                teacher_path.name,
                free_path.name,
                "result.json",
                "source_manifest.json",
                "runtime_manifest.json",
            )
        },
        "self_hash_excluded": True,
        "test_object_opened": False,
    }
    _write_json(args.output_dir / "final_hash_manifest.json", final_manifest)
    summary = {
        "schema": "w26_l4_d093_best_open_validation_summary_v1",
        "run_id": contract.run_id,
        "architecture": contract.architecture,
        "train_trajectory_count": contract.train_trajectory_count,
        "training_complete": attempt["complete"],
        "best_step": attempt["best_step"],
        "teacher_realm_npe_sum": (
            ordered_summary and ordered_summary["realm_npe_sum_source"]
        ),
        "free_realm_npe_sum": free_summary and free_summary["realm_npe_sum_source"],
        "teacher_valid_length": teacher_prefix,
        "free_valid_length": free_predictions.shape[0],
        "truth_input_competence_pass": truth_gate["all_gates_pass"],
        "evaluator_closure_pass": closure["all_gates_pass"],
        "result_sha256": sha256_file(args.output_dir / "result.json"),
        "final_hash_manifest_sha256": sha256_file(
            args.output_dir / "final_hash_manifest.json"
        ),
        "test_object_opened": False,
    }
    _write_json(args.output_dir / "summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_evaluation(args)
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
