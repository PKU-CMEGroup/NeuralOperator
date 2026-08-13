"""Capture the missing D089 step-100 validation row before its stop gate.

This is a narrow deterministic diagnostic replay, not a training or resume
interface. It accepts only the exact D089 step-50 parent artifacts, open
IgnitHIT inputs, the P1b normalizer, and a new isolated output directory. It
replays optimizer steps 51--100, writes the full H29 validation row atomically,
applies D089's unchanged eligibility check, and stops in every outcome.

Real-data, checkpoint, CUDA, or remote execution requires separate approval.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.time_dependent_no import (
    diagnose_realm_ignithit_decode_failure as replay_support,
)
from scripts.time_dependent_no import (
    train_realm_ignithit_domain_linked_ffno as d089,
)
from utility.time_dependent_no.realm_benchmark import canonical_json_sha256
from utility.time_dependent_no.realm_ffno import (
    normalize_realm_coordinates,
    trainable_parameter_count,
)
from utility.time_dependent_no.realm_ignithit import (
    TRAIN_GROUPS,
    VAL_GROUPS,
    sha256_file,
)

parent = d089.parent

RUN_ID = "d090_realm_ignithit_p1d_step100_failure_capture_20260813a"
STABLE_ID = "D090"
PARENT_RUN_ID = d089.RUN_ID
PARENT_COMPLETED_STEP = 50
PARENT_BEST_STEP = 1
TARGET_COMPLETED_STEP = 100
REPLAY_STEPS = tuple(range(PARENT_COMPLETED_STEP + 1, TARGET_COMPLETED_STEP + 1))

PARENT_LAST_SHA256 = "e0dc58273727fb07e5cb698df805e27ea3f43e68a6a259c1c1581649ebd047bf"
PARENT_BEST_SHA256 = "c25f519f294ec8a11b8b863d7d00b8724b18ef67304cfa632ea954344973ef56"
PARENT_LAST_MODEL_STATE_SHA256 = (
    "6bb34b88b082e7e6d5668c651703707949a4a7d8b2ef4334ed903655c0deb7c0"
)
PARENT_BEST_MODEL_STATE_SHA256 = (
    "efcbdac2e5d85f653ac0bee8452a39435f857f2ba1b5eedd176f12e0e19f8c1f"
)
PARENT_RUN_SIGNATURE = (
    "d5e0acb53973ec9c89ad51aeb718bd91e84a4f192971ff8d47f1235a3efbd5e1"
)
PARENT_BEST_SCORE = 4.547765731811523
PARENT_LAST_SCORE = 4.924855709075928
PARENT_PROVENANCE = {
    "config_digest": (
        "9e1edc140c29ec79fbc99e7c65dffa295e045e01c526b077664a4314ccb7a161"
    ),
    "input_digest": (
        "08d80fc972ae5ace8f4c3c5392c4fd36b53187f2d2a7655fff820257a2ec0074"
    ),
    "source_digest": (
        "b301cab59e53870f46a5ac2ae0252646e51939e3c567d0ce85f3910b919e31ec"
    ),
    "runtime_digest": (
        "e92f794237558837f4d0cef5c23aed535213dc47245092a01208703e8dc9e0f2"
    ),
}
PARENT_FILE_SHA256 = {
    "best.pt": PARENT_BEST_SHA256,
    "config.json": ("593741db58503260f0826146b8ea3551f4be8358369dfd1d7607183c2bb16df2"),
    "history.json": (
        "33226d55fced48ce644257e090e638c35e5f47f4edfe442654db3d9b62431eec"
    ),
    "input_manifest.json": (
        "87e212a322790d78de480b4b338fbf5f3c34ae3247b2062d273899e1179758f4"
    ),
    "last.pt": PARENT_LAST_SHA256,
    "runtime_manifest.json": (
        "3067ee9f3d954460f3cab96d7820466ed23723591269c1237d247f822cf33265"
    ),
    "source_manifest.json": (
        "0b553d2a979894f7fff441dd784604ab0ab7c417eea8386fb7c4c1ef10794bcf"
    ),
}

CAPTURE_CHECKPOINT_SCHEMA = "d090_ignithit_failure_capture_model_v1"
VALIDATION_ROW_SCHEMA = "d090_ignithit_failure_capture_validation_row_v1"
ELIGIBILITY_SCHEMA = "d090_ignithit_failure_capture_eligibility_v1"
ELIGIBILITY_FLAGS = (
    "all_normalized_finite",
    "all_decoded_finite",
    "all_released_state_admissible",
    "all_bounded_10x_train_max",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--normalizer-arrays", type=Path, required=True)
    parser.add_argument("--parent-output-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def diagnostic_contract() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": "d090_ignithit_failure_capture_contract_v1",
        "stable_id": STABLE_ID,
        "run_id": RUN_ID,
        "parent_run_id": PARENT_RUN_ID,
        "parent_completed_step": PARENT_COMPLETED_STEP,
        "parent_best_step": PARENT_BEST_STEP,
        "target_completed_step": TARGET_COMPLETED_STEP,
        "replay_steps": list(REPLAY_STEPS),
        "parent_last_sha256": PARENT_LAST_SHA256,
        "parent_best_sha256": PARENT_BEST_SHA256,
        "parent_last_model_state_sha256": PARENT_LAST_MODEL_STATE_SHA256,
        "parent_best_model_state_sha256": PARENT_BEST_MODEL_STATE_SHA256,
        "parent_run_signature": PARENT_RUN_SIGNATURE,
        "parent_provenance": dict(PARENT_PROVENANCE),
        "validation_start_frame": 0,
        "validation_horizon": parent.VALIDATION_HORIZON,
        "ordered_validation_groups": list(VAL_GROUPS),
        "recurrence": "domain_linked_direct_normalized_proposal",
        "validation_implementation": (
            "d089 raw parent validation plus domain-link diagnostics"
        ),
        "eligibility_implementation": "d089.require_eligible_validation",
        "write_order": [
            "validation_row.json",
            "d089 eligibility check",
            "eligibility.json",
        ],
        "selection": "none",
        "output_repair": "none",
        "retry": "forbidden",
        "stop_after_one_validation_regardless_of_outcome": True,
        "step100_checkpoint_role": "nonresumable diagnostic identity only",
        "test_object_available": False,
        "residual_arm_authorized": False,
    }
    payload["canonical_payload_sha256"] = canonical_json_sha256(payload)
    return payload


def prepare_new_output_directory(
    output_dir: Path,
    *,
    data_root: Path,
    parent_output_dir: Path,
) -> None:
    replay_support.prepare_new_output_directory(
        output_dir,
        data_root=data_root,
        parent_output_dir=parent_output_dir,
    )


def validate_parent_files(root: Path) -> dict[str, str]:
    if root.is_symlink() or not root.is_dir():
        raise FileNotFoundError("D089 parent output directory is missing")
    actual: set[str] = set()
    for path in root.iterdir():
        if path.is_symlink() or not path.is_file():
            raise ValueError("D089 parent output must contain only regular files")
        actual.add(path.name)
    if actual != set(PARENT_FILE_SHA256):
        missing = sorted(set(PARENT_FILE_SHA256) - actual)
        unexpected = sorted(actual - set(PARENT_FILE_SHA256))
        raise ValueError(
            "D089 parent output inventory differs; "
            f"missing={missing}, unexpected={unexpected}"
        )
    return replay_support.validate_file_hashes(root, PARENT_FILE_SHA256)


def validate_restored_parent_identity(
    checkpoint: Mapping[str, Any],
    *,
    completed_step: int,
    best_step: int,
    best_score: float,
    best_model_state_sha256: str | None,
    checkpoint_history: Sequence[Mapping[str, Any]],
    history_file: Mapping[str, Any],
) -> None:
    last_validation = (
        checkpoint_history[-1].get("validation") if checkpoint_history else None
    )
    last_score = (
        last_validation.get("realm_npe_mean")
        if isinstance(last_validation, Mapping)
        else None
    )
    checks = (
        checkpoint.get("schema") == d089.LAST_CHECKPOINT_SCHEMA,
        checkpoint.get("run_id") == PARENT_RUN_ID,
        checkpoint.get("run_signature") == PARENT_RUN_SIGNATURE,
        checkpoint.get("provenance") == PARENT_PROVENANCE,
        completed_step == PARENT_COMPLETED_STEP,
        best_step == PARENT_BEST_STEP,
        best_score == PARENT_BEST_SCORE,
        best_model_state_sha256 == PARENT_BEST_MODEL_STATE_SHA256,
        checkpoint.get("model_state_sha256") == PARENT_LAST_MODEL_STATE_SHA256,
        history_file.get("rows") == list(checkpoint_history),
        [row.get("completed_step") for row in checkpoint_history] == [1, 50],
        last_score == PARENT_LAST_SCORE,
    )
    if not all(checks):
        raise ValueError("restored D089 step-50 parent identity differs")


def replay_registered_steps(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    order_generator: random.Random,
    train_normalized: torch.Tensor,
    coordinates: torch.Tensor,
    *,
    device: torch.device,
) -> dict[str, Any]:
    if REPLAY_STEPS != replay_support.REPLAY_STEPS:
        raise RuntimeError("D090 and maintained replay step ranges differ")
    trace = replay_support.replay_registered_steps(
        model,
        optimizer,
        scheduler,
        order_generator,
        train_normalized,
        coordinates,
        device=device,
    )
    result = dict(trace)
    result.update(
        {
            "schema": "d090_ignithit_failure_capture_replay_trace_v1",
            "run_id": RUN_ID,
            "parent_run_id": PARENT_RUN_ID,
            "parent_last_sha256": PARENT_LAST_SHA256,
        }
    )
    return result


def build_capture_checkpoint(
    model: nn.Module,
    *,
    normalizer_state: Mapping[str, Any],
    diagnostic_run_signature: str,
    diagnostic_source_digest: str,
    runtime_digest: str,
    replay_trace_digest: str,
) -> dict[str, Any]:
    model_state = parent._recursive_to_cpu(model.state_dict())
    normalizer_state_cpu = parent._recursive_to_cpu(normalizer_state)
    return {
        "schema": CAPTURE_CHECKPOINT_SCHEMA,
        "run_id": RUN_ID,
        "parent_run_id": PARENT_RUN_ID,
        "parent_last_sha256": PARENT_LAST_SHA256,
        "parent_run_signature": PARENT_RUN_SIGNATURE,
        "diagnostic_run_signature": diagnostic_run_signature,
        "completed_step": TARGET_COMPLETED_STEP,
        "model_state": model_state,
        "model_state_sha256": parent.structured_state_sha256(model_state),
        "normalizer_state": normalizer_state_cpu,
        "normalizer_state_sha256": parent.structured_state_sha256(normalizer_state_cpu),
        "diagnostic_source_digest": diagnostic_source_digest,
        "runtime_digest": runtime_digest,
        "replay_trace_digest": replay_trace_digest,
        "diagnostic_only": True,
        "selection_eligible": False,
        "resume_supported": False,
        "test_object_opened": False,
    }


def run_raw_validation_capture(
    model: nn.Module,
    validation_normalized: torch.Tensor,
    validation_native: torch.Tensor,
    coordinates: torch.Tensor,
    trajectory_normalizer: Any,
    train_max_abs: torch.Tensor,
    *,
    case_keys: Sequence[str] = VAL_GROUPS,
    calls: int = parent.VALIDATION_HORIZON,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run D089's pre-gate validation and retain non-array stage identities."""

    if not isinstance(model, d089.DomainLinkedMap):
        raise TypeError("D090 validation requires the exact domain-linked model")
    if tuple(case_keys) != tuple(VAL_GROUPS) and calls == parent.VALIDATION_HORIZON:
        raise ValueError("scientific H29 capture requires the frozen cases")
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            prediction = parent.direct_rollout(
                model,
                validation_normalized[:, 0],
                coordinates,
                calls=calls,
            )
            decoded = trajectory_normalizer.decode(
                prediction,
                inverse_domain_policy="nan",
            )
            summary = parent.summarize_validation_predictions(
                prediction,
                validation_normalized[:, 1 : calls + 1],
                decoded,
                validation_native[:, 1 : calls + 1],
                case_keys=case_keys,
                train_max_abs=train_max_abs,
            )
    finally:
        model.train(was_training)
    summary.update(d089.domain_link_diagnostics(model))
    stages: dict[str, Any] = {
        "schema": "d090_ignithit_failure_capture_stage_manifest_v1",
        "frame0_normalized_sha256": parent.structured_state_sha256(
            validation_normalized[:, 0]
        ),
        "truth_normalized_sha256": parent.structured_state_sha256(
            validation_normalized[:, 1 : calls + 1]
        ),
        "truth_native_sha256": parent.structured_state_sha256(
            validation_native[:, 1 : calls + 1]
        ),
        "prediction_normalized_sha256": parent.structured_state_sha256(prediction),
        "prediction_decoded_sha256": parent.structured_state_sha256(decoded),
        "coordinates_sha256": parent.structured_state_sha256(coordinates),
        "prediction_shape": list(prediction.shape),
        "dtype": str(prediction.dtype),
        "device": str(prediction.device),
        "calls": calls,
        "recurrence": "current=domain_linked_proposal",
        "raw_output_repair": False,
    }
    stages["canonical_payload_sha256"] = canonical_json_sha256(stages)
    return summary, stages


def capture_validation_before_gate(
    path: Path,
    row: Mapping[str, Any],
    *,
    eligibility_check: Callable[[Mapping[str, Any]], Mapping[str, Any]] = (
        d089.require_eligible_validation
    ),
) -> dict[str, Any]:
    validation = row.get("validation")
    if not isinstance(validation, Mapping):
        raise TypeError("captured row requires a validation mapping")
    missing = [flag for flag in ELIGIBILITY_FLAGS if flag not in validation]
    if missing:
        raise ValueError(f"captured validation is missing flags: {missing}")
    non_boolean = [
        flag for flag in ELIGIBILITY_FLAGS if not isinstance(validation[flag], bool)
    ]
    if non_boolean:
        raise TypeError(f"captured validation flags must be booleans: {non_boolean}")
    payload = dict(row)
    payload["schema"] = VALIDATION_ROW_SCHEMA
    json.dumps(payload, allow_nan=False)
    d089._PARENT_WRITE_JSON_ATOMIC(path, payload)
    if not path.is_file():
        raise RuntimeError("validation row was not persisted before eligibility")
    if parent._load_json(path) != payload:
        raise RuntimeError("persisted validation row differs before eligibility")
    try:
        eligibility_check(validation)
    except RuntimeError as exc:
        result = {
            "schema": ELIGIBILITY_SCHEMA,
            "eligible": False,
            "status": "captured_gate_failure",
            "failure_message": str(exc),
        }
    else:
        result = {
            "schema": ELIGIBILITY_SCHEMA,
            "eligible": True,
            "status": "unexpected_eligible",
            "failure_message": None,
        }
    result.update(
        {
            "flags": {flag: validation[flag] for flag in ELIGIBILITY_FLAGS},
            "failed_flags": [
                flag for flag in ELIGIBILITY_FLAGS if validation[flag] is False
            ],
            "realm_npe_mean": validation.get("realm_npe_mean"),
            "validation_row_sha256": sha256_file(path),
            "stopped_after_gate": True,
        }
    )
    json.dumps(result, allow_nan=False)
    return result


def _source_manifest() -> dict[str, Any]:
    paths = (
        REPO_ROOT / "utility" / "time_dependent_no" / "realm_benchmark.py",
        REPO_ROOT / "utility" / "time_dependent_no" / "realm_ignithit.py",
        REPO_ROOT / "utility" / "time_dependent_no" / "realm_ffno.py",
        REPO_ROOT / "utility" / "time_dependent_no" / "realm_domain_link.py",
        REPO_ROOT / "scripts" / "time_dependent_no" / "train_realm_ignithit_ffno.py",
        REPO_ROOT
        / "scripts"
        / "time_dependent_no"
        / "train_realm_ignithit_domain_linked_ffno.py",
        REPO_ROOT
        / "scripts"
        / "time_dependent_no"
        / "diagnose_realm_ignithit_decode_failure.py",
        Path(__file__).resolve(),
    )
    payload: dict[str, Any] = {
        "schema": "d090_ignithit_failure_capture_source_v1",
        "files": [
            {
                "path": path.relative_to(REPO_ROOT).as_posix(),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in paths
        ],
    }
    payload["canonical_payload_sha256"] = canonical_json_sha256(payload)
    return payload


def _require_exact_payload(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    name: str,
) -> None:
    if actual != expected:
        raise ValueError(f"current {name} differs from exact D089 parent payload")


def run_capture(args: argparse.Namespace) -> dict[str, Any]:
    validate_parent_files(args.parent_output_dir)
    prepare_new_output_directory(
        args.output_dir,
        data_root=args.data_root,
        parent_output_dir=args.parent_output_dir,
    )
    parent_config = parent._load_json(args.parent_output_dir / "config.json")
    parent_input = parent._load_json(args.parent_output_dir / "input_manifest.json")
    parent_source = parent._load_json(args.parent_output_dir / "source_manifest.json")
    parent_runtime = parent._load_json(args.parent_output_dir / "runtime_manifest.json")
    parent_history = parent._load_json(args.parent_output_dir / "history.json")

    manifest_payload = parent._load_json(args.manifest)
    metadata, current_input = replay_support._current_input_manifest(
        manifest_payload,
        data_root=args.data_root,
    )
    parent._configure_determinism()
    device = torch.device(parent.DEVICE)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("the exact D090 replay requires one visible CUDA device")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)

    with d089.activated_contract():
        (
            state_normalizer,
            trajectory_normalizer,
            train_max_abs,
            normalizer_state,
        ) = parent._normalizers_from_arrays(args.normalizer_arrays)
        current_config = parent.frozen_training_contract()
        current_parent_source = parent._source_manifest()
        current_runtime = parent._runtime_manifest(device)
        _require_exact_payload(current_config, parent_config, name="config")
        _require_exact_payload(current_input, parent_input, name="input manifest")
        _require_exact_payload(current_parent_source, parent_source, name="source")
        _require_exact_payload(current_runtime, parent_runtime, name="runtime")
        observed_provenance = {
            "config_digest": parent_config["canonical_payload_sha256"],
            "input_digest": parent_input["canonical_payload_sha256"],
            "source_digest": parent_source["canonical_payload_sha256"],
            "runtime_digest": parent_runtime["canonical_payload_sha256"],
        }
        if observed_provenance != PARENT_PROVENANCE:
            raise ValueError("D089 parent provenance differs")
        if canonical_json_sha256(PARENT_PROVENANCE) != PARENT_RUN_SIGNATURE:
            raise RuntimeError("frozen D089 parent signature is inconsistent")

        train_normalized, _ = parent._load_normalized_trajectories(
            args.data_root,
            "train",
            TRAIN_GROUPS,
            state_normalizer,
            retain_native=False,
        )
        validation_normalized, validation_native = parent._load_normalized_trajectories(
            args.data_root,
            "val",
            VAL_GROUPS,
            state_normalizer,
            retain_native=True,
        )
        if validation_native is None:
            raise RuntimeError("validation native truth was not retained")
        validation_normalized_device = validation_normalized.to(device)
        validation_native_device = validation_native.to(device)
        coordinates = normalize_realm_coordinates(
            torch.from_numpy(metadata.coords).unsqueeze(0).to(dtype=parent.DTYPE)
        ).to(device)

        model = parent.RealmFFNO2d().to(device=device, dtype=parent.DTYPE)
        if trainable_parameter_count(model) != 8_936_460:
            raise RuntimeError("D089 model parameter count differs")
        optimizer, scheduler = parent.build_optimizer_and_scheduler(model)
        order_generator = random.Random(parent.SEED)
        parent_last = torch.load(
            args.parent_output_dir / "last.pt",
            map_location="cpu",
            weights_only=False,
        )
        if not isinstance(parent_last, Mapping):
            raise TypeError("D089 last checkpoint root must be a mapping")
        completed_step, best_step, best_score, best_digest, history = (
            parent.restore_last_checkpoint(
                parent_last,
                model,
                optimizer,
                scheduler,
                order_generator,
                expected_run_signature=PARENT_RUN_SIGNATURE,
            )
        )
        validate_restored_parent_identity(
            parent_last,
            completed_step=completed_step,
            best_step=best_step,
            best_score=best_score,
            best_model_state_sha256=best_digest,
            checkpoint_history=history,
            history_file=parent_history,
        )
        parent_best = torch.load(
            args.parent_output_dir / "best.pt",
            map_location="cpu",
            weights_only=False,
        )
        if not isinstance(parent_best, Mapping) or not parent._best_checkpoint_matches(
            parent_best,
            run_signature=PARENT_RUN_SIGNATURE,
            best_step=best_step,
            best_score=best_score,
            best_model_state_sha256=best_digest,
            normalizer_state_sha256=parent.structured_state_sha256(normalizer_state),
            provenance=PARENT_PROVENANCE,
        ):
            raise ValueError("D089 best checkpoint identity differs")

        contract = diagnostic_contract()
        source = _source_manifest()
        provenance = {
            "contract_digest": contract["canonical_payload_sha256"],
            "diagnostic_source_digest": source["canonical_payload_sha256"],
            "parent_run_signature": PARENT_RUN_SIGNATURE,
            "parent_last_sha256": PARENT_LAST_SHA256,
            "input_digest": PARENT_PROVENANCE["input_digest"],
            "runtime_digest": PARENT_PROVENANCE["runtime_digest"],
        }
        run_signature = canonical_json_sha256(provenance)
        parent_identity = {
            "schema": "d090_ignithit_failure_capture_parent_v1",
            "run_id": PARENT_RUN_ID,
            "run_signature": PARENT_RUN_SIGNATURE,
            "completed_step": PARENT_COMPLETED_STEP,
            "best_step": PARENT_BEST_STEP,
            "best_score": PARENT_BEST_SCORE,
            "last_score": PARENT_LAST_SCORE,
            "last_model_state_sha256": PARENT_LAST_MODEL_STATE_SHA256,
            "best_model_state_sha256": PARENT_BEST_MODEL_STATE_SHA256,
            "provenance": dict(PARENT_PROVENANCE),
            "files": dict(PARENT_FILE_SHA256),
        }
        for name, payload in (
            ("contract.json", contract),
            ("parent_identity.json", parent_identity),
            ("input_manifest.json", current_input),
            ("source_manifest.json", source),
            ("runtime_manifest.json", current_runtime),
        ):
            d089._PARENT_WRITE_JSON_ATOMIC(args.output_dir / name, payload)

        started_at = time.monotonic()
        replay_trace = replay_registered_steps(
            model,
            optimizer,
            scheduler,
            order_generator,
            train_normalized,
            coordinates,
            device=device,
        )
        d089._PARENT_WRITE_JSON_ATOMIC(
            args.output_dir / "replay_trace.json", replay_trace
        )
        replay_trace_digest = canonical_json_sha256(replay_trace)
        capture_checkpoint = build_capture_checkpoint(
            model,
            normalizer_state=normalizer_state,
            diagnostic_run_signature=run_signature,
            diagnostic_source_digest=source["canonical_payload_sha256"],
            runtime_digest=current_runtime["canonical_payload_sha256"],
            replay_trace_digest=replay_trace_digest,
        )
        parent._torch_save_atomic(
            args.output_dir / "step100_model.pt", capture_checkpoint
        )

        validation_started = time.monotonic()
        validation, stage_manifest = run_raw_validation_capture(
            model,
            validation_normalized_device,
            validation_native_device,
            coordinates,
            trajectory_normalizer,
            train_max_abs,
        )
        validation_seconds = time.monotonic() - validation_started
        d089._PARENT_WRITE_JSON_ATOMIC(
            args.output_dir / "stage_manifest.json", stage_manifest
        )
        last_training_row = replay_trace["rows"][-1]
        validation_row = {
            key: value for key, value in last_training_row.items() if key != "schema"
        }
        validation_row.update(
            {
                "validation_seconds": validation_seconds,
                "validation": validation,
                "model_state_sha256": capture_checkpoint["model_state_sha256"],
                "stage_manifest_digest": stage_manifest["canonical_payload_sha256"],
            }
        )
        eligibility = capture_validation_before_gate(
            args.output_dir / "validation_row.json",
            validation_row,
        )
        d089._PARENT_WRITE_JSON_ATOMIC(
            args.output_dir / "eligibility.json", eligibility
        )

        metric = validation.get("realm_npe_mean")
        metric_value = float(metric) if metric is not None else math.nan
        summary: dict[str, Any] = {
            "schema": "d090_ignithit_failure_capture_summary_v1",
            "stable_id": STABLE_ID,
            "run_id": RUN_ID,
            "run_signature": run_signature,
            "provenance": provenance,
            "parent_run_id": PARENT_RUN_ID,
            "parent_completed_step": PARENT_COMPLETED_STEP,
            "target_completed_step": TARGET_COMPLETED_STEP,
            "optimizer_steps_replayed": len(REPLAY_STEPS),
            "step100_model_state_sha256": capture_checkpoint["model_state_sha256"],
            "replay_trace_digest": replay_trace_digest,
            "stage_manifest_digest": stage_manifest["canonical_payload_sha256"],
            "eligibility_status": eligibility["status"],
            "eligible": eligibility["eligible"],
            "failed_flags": eligibility["failed_flags"],
            "step100_realm_npe_mean": metric_value
            if math.isfinite(metric_value)
            else None,
            "passes_pilot_informed_accuracy_cutoff": (
                metric_value < 4.5625491142 if math.isfinite(metric_value) else None
            ),
            "elapsed_seconds": time.monotonic() - started_at,
            "stopped_after_one_validation": True,
            "retry_executed": False,
            "selection_executed": False,
            "test_object_opened": False,
            "raw_output_repair_used": False,
            "residual_arm_executed": False,
            "anti_claims": [
                "failure capture is not a causal explanation of training dynamics",
                "validation is a model-selection population, not untouched test evidence",
                "a captured row does not authorize continuation, retry, or selection",
                "this diagnostic is not a completed baseline or residual comparison",
            ],
        }
        d089._PARENT_WRITE_JSON_ATOMIC(args.output_dir / "summary.json", summary)
        retained_names = (
            "contract.json",
            "parent_identity.json",
            "input_manifest.json",
            "source_manifest.json",
            "runtime_manifest.json",
            "replay_trace.json",
            "step100_model.pt",
            "stage_manifest.json",
            "validation_row.json",
            "eligibility.json",
            "summary.json",
        )
        final_manifest = {
            "schema": "d090_ignithit_failure_capture_final_hash_manifest_v1",
            "files": {
                name: sha256_file(args.output_dir / name) for name in retained_names
            },
            "self_hash_excluded": True,
        }
        d089._PARENT_WRITE_JSON_ATOMIC(
            args.output_dir / "final_hash_manifest.json", final_manifest
        )
        return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary = run_capture(args)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
