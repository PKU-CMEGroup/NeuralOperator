"""Replay and localize the frozen D088 IgnitHIT step-100 decode failure.

This is a narrow diagnostic, not a general resume or training interface. It
accepts only the exact parent artifacts, open IgnitHIT inputs, P1b normalizer,
and a new isolated output directory. It restores step 50, executes steps
51--100, localizes the H29 decode result, and stops in every outcome.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.time_dependent_no import train_realm_ignithit_ffno as parent_trainer
from utility.time_dependent_no.realm_benchmark import (
    IGNITHIT_FIELDS,
    IGNITHIT_OPEN_MANIFEST_SHA256,
    RealmNormalizer,
    canonical_json_sha256,
    parse_manifest_payload,
    validate_ignithit_open_manifest,
)
from utility.time_dependent_no.realm_ffno import (
    RealmFFNO2d,
    grouped_next_state_mse,
    normalize_realm_coordinates,
    parameter_count_within_reported_tolerance,
    trainable_parameter_count,
)
from utility.time_dependent_no.realm_ignithit import (
    TRAIN_GROUPS,
    VAL_GROUPS,
    load_ignithit_metadata,
    sha256_file,
    validate_local_open_tree,
)

RUN_ID = "d088_realm_ignithit_p1c_decode_localization_20260812a"
PARENT_RUN_ID = parent_trainer.RUN_ID
PARENT_COMPLETED_STEP = 50
TARGET_COMPLETED_STEP = 100
REPLAY_STEPS = tuple(range(PARENT_COMPLETED_STEP + 1, TARGET_COMPLETED_STEP + 1))

PARENT_LAST_SHA256 = "437e304b0c280488c08dcb727ea7de0431bee363df805821785bc09f8fe13832"
PARENT_BEST_SHA256 = "7a94eee88d3ed903b610bdeb3888b144d294cbcc9f90eec807cd005c43480fe4"
PARENT_RUN_SIGNATURE = (
    "2c08721a2b769ca30d25717eea4ee5027e78bc03a73c95f55921d3cf6c25d76f"
)
PARENT_PROVENANCE = {
    "config_digest": (
        "9949335070d23ecd8719a94c67d327e0aa53ce421ea1dffa27892b7fba599fd7"
    ),
    "input_digest": (
        "08d80fc972ae5ace8f4c3c5392c4fd36b53187f2d2a7655fff820257a2ec0074"
    ),
    "source_digest": (
        "a6f71c1c906ebbe8ab36566a8dcc28dd09d1feb1ac0b887bb548ec66570725ee"
    ),
    "runtime_digest": (
        "35e3f3f4bf8098eb73fd6b51fc42828a7ba59cee2c3200b0fea28b0bbd4371aa"
    ),
}
PARENT_FILE_SHA256 = {
    "config.json": ("cc40b001c341aa4d9fcd97c686cd3ca0e2b43aa439f4b4f58a7fb8a5f9887dd6"),
    "input_manifest.json": (
        "87e212a322790d78de480b4b338fbf5f3c34ae3247b2062d273899e1179758f4"
    ),
    "source_manifest.json": (
        "c5f7e4c136115eb59e0d482961ab1a4c363a9515267d1154433fa20562cc47b0"
    ),
    "runtime_manifest.json": (
        "7ea508c45fa1a2a74a9dc5fd0f131686373c3024651fbb2d383ce5f199849f43"
    ),
    "history.json": (
        "24876f3313fcdc1bfe68d8d86b496fcadd6286a99dfcd2a9283ddc5c87db4641"
    ),
    "best.pt": PARENT_BEST_SHA256,
    "last.pt": PARENT_LAST_SHA256,
}

LOCALIZATION_SCHEMA = "d088_ignithit_decode_localization_v1"
MODEL_CHECKPOINT_SCHEMA = "d088_ignithit_decode_localization_model_v1"


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
        "schema": "d088_ignithit_decode_localization_contract_v1",
        "run_id": RUN_ID,
        "parent_run_id": PARENT_RUN_ID,
        "parent_completed_step": PARENT_COMPLETED_STEP,
        "target_completed_step": TARGET_COMPLETED_STEP,
        "replay_steps": list(REPLAY_STEPS),
        "parent_last_sha256": PARENT_LAST_SHA256,
        "parent_best_sha256": PARENT_BEST_SHA256,
        "parent_run_signature": PARENT_RUN_SIGNATURE,
        "parent_provenance": dict(PARENT_PROVENANCE),
        "validation_start_frame": 0,
        "validation_horizon": parent_trainer.VALIDATION_HORIZON,
        "ordered_validation_groups": list(VAL_GROUPS),
        "recurrence": "direct_normalized_proposal",
        "decode_policy": "primary_p1b_inverse_domain_nan",
        "first_event_order": ["case", "call", "channel", "row", "column"],
        "stop_after_localization_regardless_of_reproduction": True,
        "test_object_available": False,
        "residual_arm_authorized": False,
    }
    payload["canonical_payload_sha256"] = canonical_json_sha256(payload)
    return payload


def _is_within(child: Path, parent: Path) -> bool:
    child_resolved = child.resolve(strict=False)
    parent_resolved = parent.resolve(strict=False)
    return (
        child_resolved == parent_resolved or parent_resolved in child_resolved.parents
    )


def prepare_new_output_directory(
    output_dir: Path,
    *,
    data_root: Path,
    parent_output_dir: Path,
) -> None:
    if _is_within(output_dir, data_root) or _is_within(data_root, output_dir):
        raise ValueError("diagnostic output and data root must not overlap")
    if _is_within(output_dir, parent_output_dir) or _is_within(
        parent_output_dir, output_dir
    ):
        raise ValueError("diagnostic output and parent output must not overlap")
    if output_dir.exists():
        if not output_dir.is_dir() or any(output_dir.iterdir()):
            raise ValueError("diagnostic output directory must be absent or empty")
    else:
        output_dir.mkdir(parents=True)


def validate_file_hashes(
    root: Path,
    expected: Mapping[str, str],
) -> dict[str, str]:
    if not root.is_dir():
        raise FileNotFoundError("parent output directory is missing")
    observed: dict[str, str] = {}
    for relative, expected_digest in expected.items():
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"required parent artifact is missing: {relative}")
        actual = sha256_file(path)
        if actual != expected_digest:
            raise ValueError(f"parent artifact SHA-256 differs: {relative}")
        observed[relative] = actual
    return observed


def validate_restored_parent_identity(
    checkpoint: Mapping[str, Any],
    *,
    completed_step: int,
    best_step: int,
    checkpoint_history: Sequence[Mapping[str, Any]],
    history_file: Mapping[str, Any],
) -> None:
    if checkpoint.get("run_id") != PARENT_RUN_ID:
        raise ValueError("parent checkpoint run ID differs")
    if completed_step != PARENT_COMPLETED_STEP or best_step != PARENT_COMPLETED_STEP:
        raise ValueError("parent checkpoint is not the exact step-50 recovery state")
    if checkpoint.get("provenance") != PARENT_PROVENANCE:
        raise ValueError("parent checkpoint provenance differs")
    if history_file.get("rows") != list(checkpoint_history):
        raise ValueError("parent history file and checkpoint history differ")


def _scalar_class(value: float) -> str:
    if math.isnan(value):
        return "nan"
    if value == math.inf:
        return "positive_infinity"
    if value == -math.inf:
        return "negative_infinity"
    return "finite"


def _scalar_record(value: torch.Tensor | float) -> dict[str, float | str | None]:
    number = float(value.item()) if isinstance(value, torch.Tensor) else float(value)
    value_class = _scalar_class(number)
    return {
        "value": number if value_class == "finite" else None,
        "class": value_class,
    }


def _range_record(values: torch.Tensor) -> dict[str, Any]:
    return {
        "minimum": _scalar_record(values.amin()),
        "maximum": _scalar_record(values.amax()),
    }


def _transformed_values(
    prediction_normalized: torch.Tensor,
    normalizer: RealmNormalizer,
) -> torch.Tensor:
    if normalizer.channel_axis != 2:
        raise ValueError("D088 localization requires channel axis 2")
    if normalizer.mean.numel() != len(IGNITHIT_FIELDS):
        raise ValueError("normalizer field count differs from IgnitHIT")
    shape = [1] * prediction_normalized.ndim
    shape[2] = len(IGNITHIT_FIELDS)
    mean = normalizer.mean.to(
        device=prediction_normalized.device,
        dtype=prediction_normalized.dtype,
    ).reshape(shape)
    scale = normalizer.scale.to(
        device=prediction_normalized.device,
        dtype=prediction_normalized.dtype,
    ).reshape(shape)
    return prediction_normalized * scale + mean


def _validate_localization_normalizer(normalizer: RealmNormalizer) -> None:
    if normalizer.channel_axis != 2:
        raise ValueError("D088 localization requires channel axis 2")
    if normalizer.mean.numel() != len(IGNITHIT_FIELDS):
        raise ValueError("normalizer field count differs from IgnitHIT")
    if normalizer.transformed_channels != tuple(range(8)):
        raise ValueError("transformed fields differ from the primary P1b normalizer")
    if (
        normalizer.box_cox_lambda != parent_trainer.BOX_COX_LAMBDA
        or normalizer.box_cox_epsilon != parent_trainer.PRIMARY_BOX_COX_EPSILON
        or normalizer.std_correction != parent_trainer.STD_CORRECTION
        or normalizer.scale_stabilizer != parent_trainer.SCALE_STABILIZER
    ):
        raise ValueError("normalizer transform parameters differ from primary P1b")


def _validate_localization_inputs(
    prediction_normalized: torch.Tensor,
    case_keys: Sequence[str],
) -> None:
    if not isinstance(prediction_normalized, torch.Tensor) or not (
        prediction_normalized.is_floating_point()
    ):
        raise TypeError("prediction_normalized must be a floating tensor")
    if prediction_normalized.ndim != 5 or prediction_normalized.shape[2] != len(
        IGNITHIT_FIELDS
    ):
        raise ValueError("prediction must have shape [case, call, 12, row, column]")
    if any(size == 0 for size in prediction_normalized.shape):
        raise ValueError("prediction axes must be nonempty")
    if prediction_normalized.shape[0] != len(case_keys) or len(set(case_keys)) != len(
        case_keys
    ):
        raise ValueError("case keys must be unique and match the prediction")
    if not all(isinstance(key, str) and key for key in case_keys):
        raise ValueError("case keys must be nonempty strings")
    if not bool(torch.isfinite(prediction_normalized).all()):
        raise RuntimeError("localization requires finite normalized proposals")


def _location_record(
    index: tuple[int, int, int, int, int],
    *,
    case_keys: Sequence[str],
    normalized: torch.Tensor,
    transformed: torch.Tensor,
    decoded: torch.Tensor,
    normalizer: RealmNormalizer,
) -> dict[str, Any]:
    case, call, channel, row, column = index
    transformed_value = transformed[index]
    decoded_value = decoded[index]
    transformed_record = _scalar_record(transformed_value)
    decoded_record = _scalar_record(decoded_value)
    transformed_channel = channel in normalizer.transformed_channels
    base_record: dict[str, float | str | None] | None = None
    if transformed_channel:
        base = normalizer.box_cox_lambda * transformed_value + 1.0
        base_record = _scalar_record(base)
        if transformed_record["class"] != "finite" or base_record["class"] != "finite":
            mechanism = "transformed_overflow"
        elif float(base_record["value"]) < 0.0:
            mechanism = "inverse_domain_violation"
        else:
            mechanism = "inverse_power_overflow"
    else:
        mechanism = "linear_decode_overflow"
    return {
        "case_index": case,
        "case_key": case_keys[case],
        "call": call + 1,
        "channel_index": channel,
        "field": IGNITHIT_FIELDS[channel],
        "row": row,
        "column": column,
        "normalized_value": _scalar_record(normalized[index]),
        "transformed_value": transformed_record,
        "inverse_box_cox_base": base_record,
        "inverse_domain_margin": base_record,
        "decoded_value": decoded_record,
        "mechanism": mechanism,
    }


def _first_invalid_index(
    invalid: torch.Tensor,
    *,
    case_index: int | None = None,
) -> tuple[int, int, int, int, int]:
    if invalid.ndim != 5 or invalid.dtype != torch.bool:
        raise ValueError("invalid mask must be boolean [case, call, C, Y, X]")
    if case_index is None:
        case_hits = invalid.flatten(start_dim=1).any(dim=1)
        case_index = int(torch.nonzero(case_hits, as_tuple=False)[0, 0].item())
    case_mask = invalid[case_index]
    call_hits = case_mask.flatten(start_dim=1).any(dim=1)
    call = int(torch.nonzero(call_hits, as_tuple=False)[0, 0].item())
    channel_hits = case_mask[call].flatten(start_dim=1).any(dim=1)
    channel = int(torch.nonzero(channel_hits, as_tuple=False)[0, 0].item())
    row_hits = case_mask[call, channel].any(dim=1)
    row = int(torch.nonzero(row_hits, as_tuple=False)[0, 0].item())
    column = int(
        torch.nonzero(case_mask[call, channel, row], as_tuple=False)[0, 0].item()
    )
    return case_index, call, channel, row, column


def localize_decode_nonfiniteness(
    prediction_normalized: torch.Tensor,
    normalizer: RealmNormalizer,
    *,
    case_keys: Sequence[str],
) -> dict[str, Any]:
    """Return a JSON-finite localization summary for ``[case, call, C, Y, X]``."""

    _validate_localization_inputs(prediction_normalized, case_keys)
    _validate_localization_normalizer(normalizer)
    transformed = _transformed_values(prediction_normalized, normalizer)
    decoded = normalizer.decode(
        prediction_normalized,
        inverse_domain_policy="nan",
    )
    invalid = ~torch.isfinite(decoded)
    total_invalid = int(invalid.sum().item())
    result: dict[str, Any] = {
        "schema": LOCALIZATION_SCHEMA,
        "case_count": prediction_normalized.shape[0],
        "call_count": prediction_normalized.shape[1],
        "all_normalized_finite": True,
        "reproduced_decoded_nonfinite": total_invalid > 0,
        "total_decoded_nonfinite_points": total_invalid,
        "first_failure": None,
        "first_failure_per_case": [],
        "affected_cases": [],
        "affected_calls": [],
        "affected_channels": [],
        "first_failure_trace": [],
    }
    if total_invalid == 0:
        json.dumps(result, allow_nan=False)
        return result

    first = _first_invalid_index(invalid)
    result["first_failure"] = _location_record(
        first,
        case_keys=case_keys,
        normalized=prediction_normalized,
        transformed=transformed,
        decoded=decoded,
        normalizer=normalizer,
    )
    affected_case_indices = (
        torch.nonzero(
            invalid.any(dim=(1, 2, 3, 4)),
            as_tuple=False,
        )
        .flatten()
        .detach()
        .cpu()
        .tolist()
    )
    result["affected_cases"] = [
        {"case_index": index, "case_key": case_keys[index]}
        for index in affected_case_indices
    ]
    result["affected_calls"] = [
        int(index) + 1
        for index in torch.nonzero(
            invalid.any(dim=(0, 2, 3, 4)),
            as_tuple=False,
        )
        .flatten()
        .detach()
        .cpu()
        .tolist()
    ]
    result["affected_channels"] = [
        {"channel_index": index, "field": IGNITHIT_FIELDS[index]}
        for index in torch.nonzero(
            invalid.any(dim=(0, 1, 3, 4)),
            as_tuple=False,
        )
        .flatten()
        .detach()
        .cpu()
        .tolist()
    ]
    for case in affected_case_indices:
        first_for_case = _first_invalid_index(invalid, case_index=int(case))
        result["first_failure_per_case"].append(
            _location_record(
                first_for_case,
                case_keys=case_keys,
                normalized=prediction_normalized,
                transformed=transformed,
                decoded=decoded,
                normalizer=normalizer,
            )
        )

    first_case, first_call, first_channel, _, _ = first
    for call in range(first_call + 1):
        normalized_slice = prediction_normalized[first_case, call, first_channel]
        transformed_slice = transformed[first_case, call, first_channel]
        decoded_slice = decoded[first_case, call, first_channel]
        trace_row: dict[str, Any] = {
            "call": call + 1,
            "normalized": _range_record(normalized_slice),
            "transformed": _range_record(transformed_slice),
            "inverse_box_cox_base": None,
            "decoded_nonfinite_count": int(
                (~torch.isfinite(decoded_slice)).sum().item()
            ),
        }
        if first_channel in normalizer.transformed_channels:
            base = normalizer.box_cox_lambda * transformed_slice + 1.0
            trace_row["inverse_box_cox_base"] = _range_record(base)
        result["first_failure_trace"].append(trace_row)

    json.dumps(result, allow_nan=False)
    return result


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
    """Execute only the frozen parent operations for optimizer steps 51--100."""

    if train_normalized.shape[:3] != (len(TRAIN_GROUPS), 30, len(IGNITHIT_FIELDS)):
        raise ValueError("training tensor differs from the frozen IgnitHIT contract")
    if train_normalized.dtype != parent_trainer.DTYPE:
        raise ValueError("training tensor must use frozen float32 precision")
    if any(parameter.device != device for parameter in model.parameters()):
        raise ValueError("model parameters and replay device differ")

    model.train()
    rows: list[dict[str, Any]] = []
    for step in REPLAY_STEPS:
        sample = parent_trainer.draw_step_sample(order_generator)
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        group_sums: Counter[str] = Counter()
        learning_rate_used = float(optimizer.param_groups[0]["lr"])
        for case_index in sample.case_indices:
            input_state = (
                train_normalized[case_index, sample.frame_start].unsqueeze(0).to(device)
            )
            target_state = (
                train_normalized[case_index, sample.frame_start + 1]
                .unsqueeze(0)
                .to(device)
            )
            prediction = model(input_state, coordinates)
            loss, by_group = grouped_next_state_mse(prediction, target_state)
            if not bool(torch.isfinite(prediction).all()) or not bool(
                torch.isfinite(loss)
            ):
                raise RuntimeError("replay returned a nonfinite proposal or loss")
            loss_sum += float(loss.detach().item())
            for name, value in by_group.items():
                group_sums[name] += float(value.detach().item())
            parent_trainer.scaled_microbatch_loss(
                loss,
                effective_batch_size=parent_trainer.EFFECTIVE_BATCH_SIZE,
            ).backward()
        if not parent_trainer._all_finite_gradients(model):
            raise RuntimeError("replay returned a missing or nonfinite gradient")
        optimizer.step()
        if not all(
            bool(torch.isfinite(parameter).all()) for parameter in model.parameters()
        ):
            raise RuntimeError("replay optimizer produced a nonfinite parameter")
        scheduler.step()
        rows.append(
            {
                "completed_step": step,
                "frame_start": sample.frame_start,
                "case_order": list(sample.case_indices),
                "learning_rate_used": learning_rate_used,
                "learning_rate_after_scheduler": float(optimizer.param_groups[0]["lr"]),
                "mean_train_grouped_loss": (
                    loss_sum / parent_trainer.EFFECTIVE_BATCH_SIZE
                ),
                "mean_train_loss_by_group": {
                    name: value / parent_trainer.EFFECTIVE_BATCH_SIZE
                    for name, value in sorted(group_sums.items())
                },
            }
        )
    return {
        "schema": "d088_ignithit_decode_localization_replay_trace_v1",
        "parent_completed_step": PARENT_COMPLETED_STEP,
        "target_completed_step": TARGET_COMPLETED_STEP,
        "optimizer_step_count": len(rows),
        "rows": rows,
        "final_model_state_sha256": parent_trainer.structured_state_sha256(
            model.state_dict()
        ),
    }


def build_step100_model_checkpoint(
    model: nn.Module,
    *,
    normalizer_state: Mapping[str, Any],
    diagnostic_run_signature: str,
    diagnostic_source_digest: str,
    runtime_digest: str,
    replay_trace_digest: str,
) -> dict[str, Any]:
    model_state = parent_trainer._recursive_to_cpu(model.state_dict())
    normalizer_state_cpu = parent_trainer._recursive_to_cpu(normalizer_state)
    return {
        "schema": MODEL_CHECKPOINT_SCHEMA,
        "run_id": RUN_ID,
        "parent_run_id": PARENT_RUN_ID,
        "parent_last_sha256": PARENT_LAST_SHA256,
        "parent_run_signature": PARENT_RUN_SIGNATURE,
        "diagnostic_run_signature": diagnostic_run_signature,
        "completed_step": TARGET_COMPLETED_STEP,
        "model_state": model_state,
        "model_state_sha256": parent_trainer.structured_state_sha256(model_state),
        "normalizer_state": normalizer_state_cpu,
        "normalizer_state_sha256": parent_trainer.structured_state_sha256(
            normalizer_state_cpu
        ),
        "diagnostic_source_digest": diagnostic_source_digest,
        "runtime_digest": runtime_digest,
        "replay_trace_digest": replay_trace_digest,
        "inference_only": True,
        "resume_supported": False,
        "test_object_opened": False,
    }


def _diagnostic_source_manifest() -> dict[str, Any]:
    paths = (
        REPO_ROOT / "utility" / "time_dependent_no" / "realm_benchmark.py",
        REPO_ROOT / "utility" / "time_dependent_no" / "realm_ignithit.py",
        REPO_ROOT / "utility" / "time_dependent_no" / "realm_ffno.py",
        REPO_ROOT / "scripts" / "time_dependent_no" / "train_realm_ignithit_ffno.py",
        Path(__file__).resolve(),
    )
    payload: dict[str, Any] = {
        "schema": "d088_ignithit_decode_localization_source_v1",
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


def _current_input_manifest(
    manifest_payload: Mapping[str, Any],
    *,
    data_root: Path,
) -> tuple[Any, dict[str, Any]]:
    repository, revision, entries = parse_manifest_payload(manifest_payload)
    summary = validate_ignithit_open_manifest(repository, revision, entries)
    inventory = validate_local_open_tree(data_root, entries)
    metadata = load_ignithit_metadata(data_root / "data" / "data.npz")
    if (
        tuple(metadata.train_groups) != TRAIN_GROUPS
        or tuple(metadata.val_groups) != VAL_GROUPS
    ):
        raise ValueError("metadata split order differs from the parent contract")
    payload: dict[str, Any] = {
        "schema": "d088_ignithit_ffno_direct_inputs_v1",
        "repository": repository,
        "revision": revision,
        "open_manifest_sha256": summary["manifest_sha256"],
        "open_manifest_payload_sha256": canonical_json_sha256(manifest_payload),
        "open_entry_count": len(inventory),
        "open_total_bytes": sum(entry.size for entry in entries),
        "normalizer_arrays_sha256": parent_trainer.NORMALIZER_ARRAYS_SHA256,
        "ordered_train_groups": list(TRAIN_GROUPS),
        "ordered_validation_groups": list(VAL_GROUPS),
        "test_object_opened": False,
    }
    payload["canonical_payload_sha256"] = canonical_json_sha256(payload)
    if payload["open_manifest_sha256"] != IGNITHIT_OPEN_MANIFEST_SHA256:
        raise RuntimeError("validated open manifest differs from the parent identity")
    return metadata, payload


def _require_exact_payload(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    name: str,
) -> None:
    if actual != expected:
        raise ValueError(f"current {name} differs from the exact parent payload")


def run_localization(args: argparse.Namespace) -> dict[str, Any]:
    validate_file_hashes(args.parent_output_dir, PARENT_FILE_SHA256)
    prepare_new_output_directory(
        args.output_dir,
        data_root=args.data_root,
        parent_output_dir=args.parent_output_dir,
    )
    parent_config = parent_trainer._load_json(args.parent_output_dir / "config.json")
    parent_input = parent_trainer._load_json(
        args.parent_output_dir / "input_manifest.json"
    )
    parent_source = parent_trainer._load_json(
        args.parent_output_dir / "source_manifest.json"
    )
    parent_runtime = parent_trainer._load_json(
        args.parent_output_dir / "runtime_manifest.json"
    )
    parent_history = parent_trainer._load_json(args.parent_output_dir / "history.json")

    manifest_payload = parent_trainer._load_json(args.manifest)
    metadata, current_input = _current_input_manifest(
        manifest_payload,
        data_root=args.data_root,
    )
    (
        state_normalizer,
        trajectory_normalizer,
        _,
        normalizer_state,
    ) = parent_trainer._normalizers_from_arrays(args.normalizer_arrays)

    parent_trainer._configure_determinism()
    device = torch.device(parent_trainer.DEVICE)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("the exact replay requires one visible CUDA device")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    current_config = parent_trainer.frozen_training_contract()
    current_parent_source = parent_trainer._source_manifest()
    current_runtime = parent_trainer._runtime_manifest(device)
    _require_exact_payload(current_config, parent_config, name="training config")
    _require_exact_payload(current_input, parent_input, name="input manifest")
    _require_exact_payload(current_parent_source, parent_source, name="parent source")
    _require_exact_payload(current_runtime, parent_runtime, name="runtime")
    observed_parent_provenance = {
        "config_digest": parent_config["canonical_payload_sha256"],
        "input_digest": parent_input["canonical_payload_sha256"],
        "source_digest": parent_source["canonical_payload_sha256"],
        "runtime_digest": parent_runtime["canonical_payload_sha256"],
    }
    if observed_parent_provenance != PARENT_PROVENANCE:
        raise ValueError("parent canonical provenance differs from the frozen identity")
    if canonical_json_sha256(PARENT_PROVENANCE) != PARENT_RUN_SIGNATURE:
        raise RuntimeError("frozen parent signature is internally inconsistent")

    train_normalized, _ = parent_trainer._load_normalized_trajectories(
        args.data_root,
        "train",
        TRAIN_GROUPS,
        state_normalizer,
        retain_native=False,
    )
    validation_normalized, validation_native = (
        parent_trainer._load_normalized_trajectories(
            args.data_root,
            "val",
            VAL_GROUPS,
            state_normalizer,
            retain_native=True,
        )
    )
    if validation_native is None:
        raise RuntimeError("validation native truth was not retained")
    validation_normalized_device = validation_normalized.to(device)
    validation_native_device = validation_native.to(device)
    coordinates = normalize_realm_coordinates(
        torch.from_numpy(metadata.coords).unsqueeze(0).to(dtype=parent_trainer.DTYPE)
    ).to(device)

    model = RealmFFNO2d().to(device=device, dtype=parent_trainer.DTYPE)
    if not parameter_count_within_reported_tolerance(trainable_parameter_count(model)):
        raise RuntimeError("FFNO-M parameter count is outside the frozen tolerance")
    optimizer, scheduler = parent_trainer.build_optimizer_and_scheduler(model)
    order_generator = random.Random(parent_trainer.SEED)
    parent_last = torch.load(
        args.parent_output_dir / "last.pt",
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(parent_last, Mapping):
        raise TypeError("parent last checkpoint root must be a mapping")
    completed_step, best_step, best_score, best_digest, history = (
        parent_trainer.restore_last_checkpoint(
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
        checkpoint_history=history,
        history_file=parent_history,
    )
    parent_best = torch.load(
        args.parent_output_dir / "best.pt",
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(
        parent_best, Mapping
    ) or not parent_trainer._best_checkpoint_matches(
        parent_best,
        run_signature=PARENT_RUN_SIGNATURE,
        best_step=best_step,
        best_score=best_score,
        best_model_state_sha256=best_digest,
        normalizer_state_sha256=parent_trainer.structured_state_sha256(
            normalizer_state
        ),
        provenance=PARENT_PROVENANCE,
    ):
        raise ValueError("parent best checkpoint identity differs")

    diagnostic_source = _diagnostic_source_manifest()
    contract = diagnostic_contract()
    diagnostic_provenance = {
        "contract_digest": contract["canonical_payload_sha256"],
        "diagnostic_source_digest": diagnostic_source["canonical_payload_sha256"],
        "parent_run_signature": PARENT_RUN_SIGNATURE,
        "parent_last_sha256": PARENT_LAST_SHA256,
        "input_digest": PARENT_PROVENANCE["input_digest"],
        "runtime_digest": PARENT_PROVENANCE["runtime_digest"],
    }
    diagnostic_run_signature = canonical_json_sha256(diagnostic_provenance)
    parent_identity = {
        "schema": "d088_ignithit_decode_localization_parent_v1",
        "run_id": PARENT_RUN_ID,
        "run_signature": PARENT_RUN_SIGNATURE,
        "completed_step": PARENT_COMPLETED_STEP,
        "best_step": best_step,
        "best_score": best_score,
        "model_state_sha256": parent_last["model_state_sha256"],
        "provenance": dict(PARENT_PROVENANCE),
        "files": dict(PARENT_FILE_SHA256),
    }
    for name, payload in (
        ("contract.json", contract),
        ("parent_identity.json", parent_identity),
        ("source_manifest.json", diagnostic_source),
        ("runtime_manifest.json", current_runtime),
    ):
        parent_trainer._write_json_atomic(args.output_dir / name, payload)

    started_at = time.monotonic()
    _ = validation_native_device
    replay_trace = replay_registered_steps(
        model,
        optimizer,
        scheduler,
        order_generator,
        train_normalized,
        coordinates,
        device=device,
    )
    parent_trainer._write_json_atomic(
        args.output_dir / "replay_trace.json", replay_trace
    )
    replay_trace_digest = canonical_json_sha256(replay_trace)
    model_checkpoint = build_step100_model_checkpoint(
        model,
        normalizer_state=normalizer_state,
        diagnostic_run_signature=diagnostic_run_signature,
        diagnostic_source_digest=diagnostic_source["canonical_payload_sha256"],
        runtime_digest=current_runtime["canonical_payload_sha256"],
        replay_trace_digest=replay_trace_digest,
    )
    parent_trainer._torch_save_atomic(
        args.output_dir / "step100_model.pt", model_checkpoint
    )

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            prediction = parent_trainer.direct_rollout(
                model,
                validation_normalized_device[:, 0],
                coordinates,
                calls=parent_trainer.VALIDATION_HORIZON,
            )
            localization = localize_decode_nonfiniteness(
                prediction,
                trajectory_normalizer,
                case_keys=VAL_GROUPS,
            )
    finally:
        model.train(was_training)
    parent_trainer._write_json_atomic(
        args.output_dir / "localization.json", localization
    )
    summary: dict[str, Any] = {
        "schema": "d088_ignithit_decode_localization_summary_v1",
        "run_id": RUN_ID,
        "diagnostic_run_signature": diagnostic_run_signature,
        "parent_run_id": PARENT_RUN_ID,
        "parent_run_signature": PARENT_RUN_SIGNATURE,
        "parent_completed_step": PARENT_COMPLETED_STEP,
        "target_completed_step": TARGET_COMPLETED_STEP,
        "optimizer_steps_replayed": len(REPLAY_STEPS),
        "status": "reproduced"
        if localization["reproduced_decoded_nonfinite"]
        else "not_reproduced",
        "all_normalized_finite": True,
        "reproduced_decoded_nonfinite": localization["reproduced_decoded_nonfinite"],
        "first_failure": localization["first_failure"],
        "step100_model_state_sha256": model_checkpoint["model_state_sha256"],
        "replay_trace_digest": replay_trace_digest,
        "elapsed_seconds": time.monotonic() - started_at,
        "stopped_after_localization": True,
        "test_object_opened": False,
        "residual_arm_executed": False,
        "anti_claims": [
            "decode localization is not a causal explanation of training dynamics",
            "validation is a model-selection population, not untouched test evidence",
            "nonreproduction does not authorize continued training or a retry",
        ],
    }
    parent_trainer._write_json_atomic(args.output_dir / "summary.json", summary)
    retained_names = (
        "contract.json",
        "parent_identity.json",
        "source_manifest.json",
        "runtime_manifest.json",
        "replay_trace.json",
        "step100_model.pt",
        "localization.json",
        "summary.json",
    )
    final_manifest = {
        "schema": "d088_ignithit_decode_localization_final_hash_manifest_v1",
        "files": {name: sha256_file(args.output_dir / name) for name in retained_names},
        "self_hash_excluded": True,
    }
    parent_trainer._write_json_atomic(
        args.output_dir / "final_hash_manifest.json", final_manifest
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary = run_localization(args)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
