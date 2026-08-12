"""Audit the paired D088 step-50/step-100 fresh-map decode margins.

This is an inference-only comparison on the five registered open validation
cases.  Each checkpoint receives the same normalized frame-0 truth exactly
once.  The script does not recur, train, clip, project, access test data, or
construct a residual model.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.time_dependent_no import (
    diagnose_realm_ignithit_decode_failure as localization,
)
from scripts.time_dependent_no import train_realm_ignithit_ffno as parent_trainer
from utility.time_dependent_no.realm_benchmark import (
    IGNITHIT_FIELDS,
    RealmNormalizer,
    canonical_json_sha256,
    grouped_normalized_prediction_error,
)
from utility.time_dependent_no.realm_ffno import (
    RealmFFNO2d,
    RealmFFNOConfig,
    normalize_realm_coordinates,
    parameter_count_within_reported_tolerance,
    trainable_parameter_count,
)
from utility.time_dependent_no.realm_ignithit import (
    VAL_GROUPS,
    sha256_file,
)

RUN_ID = "d088_realm_ignithit_p1c_fresh_margin_audit_20260813a"
FRAME_START = 0
FRAME_TARGET = 1
STEP50_MODEL_STATE_SHA256 = (
    "e75ee39478cceca1a159218d75af8a42d323df5c57de3dae67ffec5d406bbf2b"
)
STEP100_MODEL_STATE_SHA256 = (
    "61df05c24aaf0a0b4b813ca11b69e8a3579e47da17209e827b25100bb2c610fd"
)
LOCALIZATION_FINAL_MANIFEST_SHA256 = (
    "9810e2ecb591bf9b5c45f0069d727e1520f495f2ab1357e39cb26523b1674f7f"
)
LOCALIZATION_FILE_SHA256 = {
    "contract.json": (
        "6325f0cb99af2a1fe487a8f880ddc88914015230c28ef1586fb6a1275c1cb90d"
    ),
    "parent_identity.json": (
        "26bc558b8070a64533dcf3a4eb208b570e24f107f7973173f6c4a6bdce2d1d6a"
    ),
    "source_manifest.json": (
        "5ef9ae44fd047136c071de88833e25aa5d7ea93903d73062c113d618a2e78c67"
    ),
    "runtime_manifest.json": (
        "7ea508c45fa1a2a74a9dc5fd0f131686373c3024651fbb2d383ce5f199849f43"
    ),
    "replay_trace.json": (
        "7cb90640791614a6596c2a3101f5933547257cc744b328e5ebb6247ee5ca6f26"
    ),
    "step100_model.pt": (
        "44b09c34a0db4dce518e71caa0d82fc08056e368b4f33768003fe489c1c0c039"
    ),
    "localization.json": (
        "163a7bfc2f6fc5d4e5b3fd45b47bd3453c4be495d565fb9d78e4acd6455a05b0"
    ),
    "summary.json": (
        "2be657e0e50f26b36e937d15804dfc7e2bb8699d5b0fb4ed671fc283b1d20cb8"
    ),
    "final_hash_manifest.json": LOCALIZATION_FINAL_MANIFEST_SHA256,
}
AUDIT_SCHEMA = "d088_ignithit_fresh_map_margin_audit_v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--normalizer-arrays", type=Path, required=True)
    parser.add_argument("--parent-output-dir", type=Path, required=True)
    parser.add_argument("--localization-output-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def audit_contract() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": "d088_ignithit_fresh_map_margin_contract_v1",
        "run_id": RUN_ID,
        "stable_result_id": "D088",
        "comparison": [
            {
                "label": "step50",
                "completed_step": localization.PARENT_COMPLETED_STEP,
                "checkpoint": "parent_output_dir/last.pt",
                "checkpoint_sha256": localization.PARENT_LAST_SHA256,
                "model_state_sha256": STEP50_MODEL_STATE_SHA256,
            },
            {
                "label": "step100",
                "completed_step": localization.TARGET_COMPLETED_STEP,
                "checkpoint": "localization_output_dir/step100_model.pt",
                "checkpoint_sha256": LOCALIZATION_FILE_SHA256["step100_model.pt"],
                "model_state_sha256": STEP100_MODEL_STATE_SHA256,
            },
        ],
        "ordered_validation_groups": list(VAL_GROUPS),
        "frame_start": FRAME_START,
        "frame_target": FRAME_TARGET,
        "calls_per_checkpoint": 1,
        "input_policy": "same_exact_normalized_frame0_truth",
        "target": "normalized_frame1_truth",
        "normalizer": "unchanged_primary_p1b",
        "normalizer_arrays_sha256": parent_trainer.NORMALIZER_ARRAYS_SHA256,
        "precision": "float32_autocast_disabled",
        "runtime_policy": (
            "both checkpoints_share_one_current_runtime; historical runtime "
            "remains provenance and known-result reproduction is mandatory"
        ),
        "checkpoint_selection": (
            "fixed step50 last state and fixed retained step100 replay state; "
            "no posthoc metric selection"
        ),
        "inverse_domain_criterion": "1 + 0.1 * transformed_value < 0",
        "spatial_support": "count_fraction_lexicographic_first_and_bbox",
        "margin_summaries": ["minimum", "p01_linear_interpolation", "mean"],
        "aggregation": "case_first_with_per_case_per_field_rows_retained",
        "known_result_equivalence": {
            "first_step100_event": "exact case/channel/row/column",
            "normalized_value_absolute_tolerance": 1.0e-5,
            "inverse_margin_absolute_tolerance": 1.0e-5,
            "first_case_channel_invalid_count": 66,
        },
        "recurrence": "none",
        "training": "none",
        "output_repair": "none",
        "test_object_available": False,
        "residual_arm_authorized": False,
    }
    payload["canonical_payload_sha256"] = _canonical_payload_sha256(payload)
    return payload


def _canonical_payload_sha256(payload: Mapping[str, Any]) -> str:
    return canonical_json_sha256(
        {
            key: value
            for key, value in payload.items()
            if key != "canonical_payload_sha256"
        }
    )


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
    localization_output_dir: Path,
) -> None:
    for protected, name in (
        (data_root, "data root"),
        (parent_output_dir, "parent output"),
        (localization_output_dir, "localization output"),
    ):
        if _is_within(output_dir, protected) or _is_within(protected, output_dir):
            raise ValueError(f"audit output and {name} must not overlap")
    if output_dir.exists():
        if not output_dir.is_dir() or any(output_dir.iterdir()):
            raise ValueError("audit output directory must be absent or empty")
    else:
        output_dir.mkdir(parents=True)


def _validate_primary_state_normalizer(normalizer: RealmNormalizer) -> None:
    if normalizer.channel_axis != 1:
        raise ValueError("fresh-map audit requires state channel axis 1")
    if normalizer.mean.numel() != len(IGNITHIT_FIELDS):
        raise ValueError("normalizer field count differs from IgnitHIT")
    if normalizer.transformed_channels != tuple(range(8)):
        raise ValueError("transformed fields differ from primary P1b")
    if (
        normalizer.box_cox_lambda != parent_trainer.BOX_COX_LAMBDA
        or normalizer.box_cox_epsilon != parent_trainer.PRIMARY_BOX_COX_EPSILON
        or normalizer.std_correction != parent_trainer.STD_CORRECTION
        or normalizer.scale_stabilizer != parent_trainer.SCALE_STABILIZER
    ):
        raise ValueError("normalizer transform parameters differ from primary P1b")


def _transformed_values(
    normalized: torch.Tensor,
    normalizer: RealmNormalizer,
) -> torch.Tensor:
    _validate_primary_state_normalizer(normalizer)
    shape = [1] * normalized.ndim
    shape[1] = len(IGNITHIT_FIELDS)
    mean = normalizer.mean.to(
        device=normalized.device,
        dtype=normalized.dtype,
    ).reshape(shape)
    scale = normalizer.scale.to(
        device=normalized.device,
        dtype=normalized.dtype,
    ).reshape(shape)
    return normalized * scale + mean


def _finite_float(value: torch.Tensor | float) -> float:
    number = float(value.item()) if isinstance(value, torch.Tensor) else float(value)
    if not math.isfinite(number):
        raise RuntimeError("fresh-map audit metric is nonfinite")
    return number


def _support_record(mask: torch.Tensor) -> dict[str, Any]:
    if mask.ndim != 2 or mask.dtype != torch.bool:
        raise ValueError("spatial support mask must be boolean [row, column]")
    count = int(mask.sum().item())
    result: dict[str, Any] = {
        "count": count,
        "fraction": count / mask.numel(),
        "first": None,
        "bbox_inclusive": None,
    }
    if count == 0:
        return result
    locations = torch.nonzero(mask, as_tuple=False)
    first = locations[0]
    result["first"] = {
        "row": int(first[0].item()),
        "column": int(first[1].item()),
    }
    result["bbox_inclusive"] = {
        "row_min": int(locations[:, 0].amin().item()),
        "row_max": int(locations[:, 0].amax().item()),
        "column_min": int(locations[:, 1].amin().item()),
        "column_max": int(locations[:, 1].amax().item()),
    }
    return result


def _margin_record(base: torch.Tensor) -> dict[str, Any]:
    if base.ndim != 2 or not base.is_floating_point():
        raise ValueError("inverse-domain base must be floating [row, column]")
    if not bool(torch.isfinite(base).all()):
        raise RuntimeError("inverse-domain base must be finite")
    flat = base.reshape(-1)
    return {
        "minimum": _finite_float(flat.amin()),
        "p01": _finite_float(torch.quantile(flat, 0.01, interpolation="linear")),
        "mean": _finite_float(flat.mean()),
        "invalid_support": _support_record(base < 0.0),
    }


def _validate_audit_inputs(
    step50_prediction: torch.Tensor,
    step100_prediction: torch.Tensor,
    truth_normalized: torch.Tensor,
    normalizer: RealmNormalizer,
    case_keys: Sequence[str],
) -> None:
    tensors = (step50_prediction, step100_prediction, truth_normalized)
    if any(not isinstance(value, torch.Tensor) for value in tensors):
        raise TypeError("predictions and truth must be tensors")
    if any(not value.is_floating_point() for value in tensors):
        raise TypeError("predictions and truth must be floating tensors")
    if any(value.shape != truth_normalized.shape for value in tensors):
        raise ValueError("step-50, step-100, and truth shapes must match")
    if (
        truth_normalized.ndim != 4
        or truth_normalized.shape[1] != len(IGNITHIT_FIELDS)
        or any(size == 0 for size in truth_normalized.shape)
    ):
        raise ValueError("audit tensors require nonempty [case, 12, row, column]")
    if any(value.dtype != truth_normalized.dtype for value in tensors) or any(
        value.device != truth_normalized.device for value in tensors
    ):
        raise ValueError("audit tensors must share dtype and device")
    if not all(bool(torch.isfinite(value).all()) for value in tensors):
        raise RuntimeError("fresh-map audit requires finite normalized tensors")
    if len(case_keys) != truth_normalized.shape[0] or len(set(case_keys)) != len(
        case_keys
    ):
        raise ValueError("case keys must be unique and match the case axis")
    if not all(isinstance(key, str) and key for key in case_keys):
        raise ValueError("case keys must be nonempty strings")
    _validate_primary_state_normalizer(normalizer)


def _normalized_error_summary(
    prediction: torch.Tensor,
    truth: torch.Tensor,
    *,
    case_keys: Sequence[str],
) -> dict[str, Any]:
    errors = grouped_normalized_prediction_error(
        prediction.unsqueeze(1),
        truth.unsqueeze(1),
    )
    difference = prediction - truth
    per_case_field: list[dict[str, Any]] = []
    for case, case_key in enumerate(case_keys):
        for channel, field in enumerate(IGNITHIT_FIELDS):
            values = difference[case, channel]
            mse = values.square().mean()
            per_case_field.append(
                {
                    "case_index": case,
                    "case_key": case_key,
                    "channel_index": channel,
                    "field": field,
                    "mse": _finite_float(mse),
                    "rmse": _finite_float(torch.sqrt(mse)),
                    "maximum_absolute_error": _finite_float(values.abs().amax()),
                }
            )
    per_field_case_first: list[dict[str, Any]] = []
    for channel, field in enumerate(IGNITHIT_FIELDS):
        rows = [row for row in per_case_field if row["channel_index"] == channel]
        per_field_case_first.append(
            {
                "channel_index": channel,
                "field": field,
                "mean_case_mse": sum(float(row["mse"]) for row in rows) / len(rows),
                "mean_case_rmse": sum(float(row["rmse"]) for row in rows) / len(rows),
                "maximum_case_absolute_error": max(
                    float(row["maximum_absolute_error"]) for row in rows
                ),
            }
        )
    per_case = []
    for case, case_key in enumerate(case_keys):
        per_case.append(
            {
                "case_index": case,
                "case_key": case_key,
                "realm_npe_mean": _finite_float(errors.per_case_mean[case]),
                "realm_npe_sum_source": _finite_float(errors.per_case_sum[case]),
                "group_mse": {
                    name: _finite_float(values[case, 0])
                    for name, values in sorted(errors.grouped_per_call.items())
                },
            }
        )
    return {
        "realm_npe_mean": errors.realm_npe_mean,
        "realm_npe_sum_source": errors.realm_npe_sum_source,
        "group_case_first_mse": {
            name: _finite_float(values[:, 0].mean())
            for name, values in sorted(errors.grouped_per_call.items())
        },
        "per_case": per_case,
        "per_case_field": per_case_field,
        "per_field_case_first": per_field_case_first,
    }


def _margin_rows(
    normalized: torch.Tensor,
    normalizer: RealmNormalizer,
    *,
    case_keys: Sequence[str],
) -> tuple[list[dict[str, Any]], torch.Tensor, torch.Tensor]:
    transformed = _transformed_values(normalized, normalizer)
    channels = tuple(normalizer.transformed_channels)
    base = normalizer.box_cox_lambda * transformed[:, channels] + 1.0
    rows: list[dict[str, Any]] = []
    for case, case_key in enumerate(case_keys):
        for local_channel, channel in enumerate(channels):
            rows.append(
                {
                    "case_index": case,
                    "case_key": case_key,
                    "channel_index": channel,
                    "field": IGNITHIT_FIELDS[channel],
                    **_margin_record(base[case, local_channel]),
                }
            )
    return rows, transformed, base


def _first_violation_record(
    invalid: torch.Tensor,
    *,
    normalized: torch.Tensor,
    transformed: torch.Tensor,
    base: torch.Tensor,
    transformed_channels: Sequence[int],
    case_keys: Sequence[str],
) -> dict[str, Any] | None:
    if not bool(invalid.any()):
        return None
    location = torch.nonzero(invalid, as_tuple=False)[0]
    case, local_channel, row, column = (int(value.item()) for value in location)
    channel = int(transformed_channels[local_channel])
    return {
        "case_index": case,
        "case_key": case_keys[case],
        "channel_index": channel,
        "field": IGNITHIT_FIELDS[channel],
        "row": row,
        "column": column,
        "normalized_value": _finite_float(normalized[case, channel, row, column]),
        "transformed_value": _finite_float(transformed[case, channel, row, column]),
        "inverse_domain_margin": _finite_float(base[case, local_channel, row, column]),
    }


def _paired_margin_rows(
    step50_base: torch.Tensor,
    step100_base: torch.Tensor,
    truth_base: torch.Tensor,
    *,
    case_keys: Sequence[str],
    transformed_channels: Sequence[int],
) -> tuple[list[dict[str, Any]], Counter[str]]:
    rows: list[dict[str, Any]] = []
    statuses: Counter[str] = Counter()
    for case, case_key in enumerate(case_keys):
        for local_channel, channel in enumerate(transformed_channels):
            base50 = step50_base[case, local_channel]
            base100 = step100_base[case, local_channel]
            base_truth = truth_base[case, local_channel]
            invalid50 = base50 < 0.0
            invalid100 = base100 < 0.0
            invalid_truth = base_truth < 0.0
            count50 = int(invalid50.sum().item())
            count100 = int(invalid100.sum().item())
            count_truth = int(invalid_truth.sum().item())
            if count_truth:
                status = "truth_inverse_domain_violation"
            elif count50 == 0 and count100 > 0:
                status = "new_step100_inverse_domain_violation"
            elif count50 > 0 and count100 > 0:
                status = "invalid_at_both_checkpoints"
            elif count50 > 0:
                status = "step100_no_longer_invalid"
            else:
                status = "valid_at_both_checkpoints"
            statuses[status] += 1
            intersection = int((invalid50 & invalid100).sum().item())
            union = int((invalid50 | invalid100).sum().item())
            delta = base100 - base50
            rows.append(
                {
                    "case_index": case,
                    "case_key": case_key,
                    "channel_index": channel,
                    "field": IGNITHIT_FIELDS[channel],
                    "status": status,
                    "step50_invalid_count": count50,
                    "step100_invalid_count": count100,
                    "truth_invalid_count": count_truth,
                    "new_step100_support_count": int(
                        (invalid100 & ~invalid50).sum().item()
                    ),
                    "removed_step50_support_count": int(
                        (invalid50 & ~invalid100).sum().item()
                    ),
                    "invalid_support_intersection_count": intersection,
                    "invalid_support_union_count": union,
                    "invalid_support_jaccard": intersection / union if union else None,
                    "invalid_support_jaccard_status": "ok" if union else "empty_union",
                    "minimum_margin_step50": _finite_float(base50.amin()),
                    "minimum_margin_step100": _finite_float(base100.amin()),
                    "minimum_margin_truth": _finite_float(base_truth.amin()),
                    "minimum_margin_shift_step100_minus_step50": _finite_float(
                        base100.amin() - base50.amin()
                    ),
                    "base_delta_minimum": _finite_float(delta.amin()),
                    "base_delta_mean": _finite_float(delta.mean()),
                    "base_delta_maximum": _finite_float(delta.amax()),
                }
            )
    return rows, statuses


def _proposal_delta_rows(
    step50_prediction: torch.Tensor,
    step100_prediction: torch.Tensor,
    truth: torch.Tensor,
    *,
    case_keys: Sequence[str],
) -> list[dict[str, Any]]:
    proposal_delta = step100_prediction - step50_prediction
    error50 = step50_prediction - truth
    error100 = step100_prediction - truth
    rows: list[dict[str, Any]] = []
    for case, case_key in enumerate(case_keys):
        for channel, field in enumerate(IGNITHIT_FIELDS):
            delta = proposal_delta[case, channel]
            mse50 = error50[case, channel].square().mean()
            mse100 = error100[case, channel].square().mean()
            rows.append(
                {
                    "case_index": case,
                    "case_key": case_key,
                    "channel_index": channel,
                    "field": field,
                    "proposal_delta_rmse": _finite_float(
                        torch.sqrt(delta.square().mean())
                    ),
                    "proposal_delta_maximum_absolute": _finite_float(
                        delta.abs().amax()
                    ),
                    "normalized_mse_step50": _finite_float(mse50),
                    "normalized_mse_step100": _finite_float(mse100),
                    "normalized_mse_change_step100_minus_step50": _finite_float(
                        mse100 - mse50
                    ),
                }
            )
    return rows


def summarize_fresh_map_margin_audit(
    step50_prediction: torch.Tensor,
    step100_prediction: torch.Tensor,
    truth_normalized: torch.Tensor,
    normalizer: RealmNormalizer,
    *,
    case_keys: Sequence[str],
) -> dict[str, Any]:
    """Summarize two one-call normalized maps against matching frame-1 truth."""

    _validate_audit_inputs(
        step50_prediction,
        step100_prediction,
        truth_normalized,
        normalizer,
        case_keys,
    )
    rows50, transformed50, base50 = _margin_rows(
        step50_prediction,
        normalizer,
        case_keys=case_keys,
    )
    rows100, transformed100, base100 = _margin_rows(
        step100_prediction,
        normalizer,
        case_keys=case_keys,
    )
    truth_rows, transformed_truth, truth_base = _margin_rows(
        truth_normalized,
        normalizer,
        case_keys=case_keys,
    )
    transformed_channels = tuple(normalizer.transformed_channels)
    paired_rows, statuses = _paired_margin_rows(
        base50,
        base100,
        truth_base,
        case_keys=case_keys,
        transformed_channels=transformed_channels,
    )
    truth_invalid = int((truth_base < 0.0).sum().item())
    step50_invalid = int((base50 < 0.0).sum().item())
    step100_invalid = int((base100 < 0.0).sum().item())
    new_step100 = int(((base100 < 0.0) & ~(base50 < 0.0)).sum().item())
    if truth_invalid:
        classification = "truth_or_normalizer_domain_failure"
    elif step50_invalid == 0 and step100_invalid > 0:
        classification = "checkpoint_history_associated_margin_regression"
    elif new_step100 > 0:
        classification = "mixed_new_step100_margin_regression"
    elif step100_invalid > 0:
        classification = "step100_violation_without_new_support"
    else:
        classification = "no_step100_inverse_domain_violation"

    result: dict[str, Any] = {
        "schema": AUDIT_SCHEMA,
        "case_count": len(case_keys),
        "frame_start": FRAME_START,
        "frame_target": FRAME_TARGET,
        "calls_per_checkpoint": 1,
        "all_normalized_finite": True,
        "checkpoints": {
            "step50": {
                "normalized_error": _normalized_error_summary(
                    step50_prediction,
                    truth_normalized,
                    case_keys=case_keys,
                ),
                "inverse_domain_margin": rows50,
                "total_inverse_domain_violation_points": step50_invalid,
                "first_inverse_domain_violation": _first_violation_record(
                    base50 < 0.0,
                    normalized=step50_prediction,
                    transformed=transformed50,
                    base=base50,
                    transformed_channels=transformed_channels,
                    case_keys=case_keys,
                ),
            },
            "step100": {
                "normalized_error": _normalized_error_summary(
                    step100_prediction,
                    truth_normalized,
                    case_keys=case_keys,
                ),
                "inverse_domain_margin": rows100,
                "total_inverse_domain_violation_points": step100_invalid,
                "first_inverse_domain_violation": _first_violation_record(
                    base100 < 0.0,
                    normalized=step100_prediction,
                    transformed=transformed100,
                    base=base100,
                    transformed_channels=transformed_channels,
                    case_keys=case_keys,
                ),
            },
        },
        "truth": {
            "inverse_domain_margin": truth_rows,
            "total_inverse_domain_violation_points": truth_invalid,
            "first_inverse_domain_violation": _first_violation_record(
                truth_base < 0.0,
                normalized=truth_normalized,
                transformed=transformed_truth,
                base=truth_base,
                transformed_channels=transformed_channels,
                case_keys=case_keys,
            ),
        },
        "paired": {
            "classification": classification,
            "checkpoint_history_associated_margin_regression": (
                truth_invalid == 0 and step50_invalid == 0 and step100_invalid > 0
            ),
            "new_step100_inverse_domain_violation_points": new_step100,
            "outcome_counts": dict(sorted(statuses.items())),
            "per_case_transformed_field": paired_rows,
            "per_case_field_normalized_delta": _proposal_delta_rows(
                step50_prediction,
                step100_prediction,
                truth_normalized,
                case_keys=case_keys,
            ),
        },
        "anti_claims": [
            "checkpoint association is not a general optimizer or architecture cause",
            "one fresh call is not a rollout-accuracy or stability result",
            "validation cases are a model-selection population, not test evidence",
        ],
    }
    json.dumps(result, allow_nan=False)
    return result


def infer_fresh_map_pair(
    step50_model: nn.Module,
    step100_model: nn.Module,
    initial_state: torch.Tensor,
    coordinates: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Call each deployed map once on the same immutable input tensors."""

    if (
        initial_state.ndim != 4
        or coordinates.ndim != 4
        or not initial_state.is_floating_point()
        or not coordinates.is_floating_point()
    ):
        raise ValueError("fresh-map inputs must be floating rank-four tensors")
    if initial_state.dtype != coordinates.dtype or initial_state.device != (
        coordinates.device
    ):
        raise ValueError("fresh-map state and coordinates must share dtype and device")
    if not bool(torch.isfinite(initial_state).all()) or not bool(
        torch.isfinite(coordinates).all()
    ):
        raise RuntimeError("fresh-map inputs must be finite")
    input_before = initial_state.detach().clone()
    coordinates_before = coordinates.detach().clone()
    training_states = (step50_model.training, step100_model.training)
    step50_model.eval()
    step100_model.eval()
    try:
        with torch.inference_mode():
            prediction50 = step50_model(initial_state, coordinates)
            if not torch.equal(initial_state, input_before) or not torch.equal(
                coordinates, coordinates_before
            ):
                raise RuntimeError("step-50 fresh-map call mutated its input")
            prediction100 = step100_model(initial_state, coordinates)
    finally:
        step50_model.train(training_states[0])
        step100_model.train(training_states[1])
    if prediction50.shape != initial_state.shape or prediction100.shape != (
        initial_state.shape
    ):
        raise ValueError("fresh proposal shape differs from frame-0 state")
    if not torch.equal(initial_state, input_before) or not torch.equal(
        coordinates, coordinates_before
    ):
        raise RuntimeError("step-100 fresh-map call mutated its input")
    if not bool(torch.isfinite(prediction50).all()) or not bool(
        torch.isfinite(prediction100).all()
    ):
        raise RuntimeError("fresh normalized proposal is nonfinite")
    return prediction50, prediction100


def validate_step50_checkpoint(checkpoint: Mapping[str, Any]) -> Mapping[str, Any]:
    model_state = checkpoint.get("model_state")
    if not isinstance(model_state, Mapping):
        raise TypeError("step-50 checkpoint model state is missing")
    observed_digest = parent_trainer.structured_state_sha256(model_state)
    if (
        checkpoint.get("schema") != parent_trainer.LAST_CHECKPOINT_SCHEMA
        or checkpoint.get("run_id") != localization.PARENT_RUN_ID
        or checkpoint.get("run_signature") != localization.PARENT_RUN_SIGNATURE
        or checkpoint.get("provenance") != localization.PARENT_PROVENANCE
        or checkpoint.get("completed_step") != localization.PARENT_COMPLETED_STEP
        or checkpoint.get("best_step") != localization.PARENT_COMPLETED_STEP
        or checkpoint.get("resume_supported") is not True
        or checkpoint.get("model_state_sha256") != STEP50_MODEL_STATE_SHA256
        or observed_digest != STEP50_MODEL_STATE_SHA256
    ):
        raise ValueError("step-50 checkpoint identity differs")
    return model_state


def validate_step100_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    normalizer_state_sha256: str,
    localization_summary: Mapping[str, Any],
    localization_source: Mapping[str, Any],
    localization_runtime: Mapping[str, Any],
    replay_trace: Mapping[str, Any],
) -> Mapping[str, Any]:
    model_state = checkpoint.get("model_state")
    normalizer_state = checkpoint.get("normalizer_state")
    if not isinstance(model_state, Mapping) or not isinstance(
        normalizer_state, Mapping
    ):
        raise TypeError("step-100 checkpoint state is missing")
    observed_model_digest = parent_trainer.structured_state_sha256(model_state)
    observed_normalizer_digest = parent_trainer.structured_state_sha256(
        normalizer_state
    )
    if (
        checkpoint.get("schema") != localization.MODEL_CHECKPOINT_SCHEMA
        or checkpoint.get("run_id") != localization.RUN_ID
        or checkpoint.get("parent_run_id") != localization.PARENT_RUN_ID
        or checkpoint.get("parent_last_sha256") != localization.PARENT_LAST_SHA256
        or checkpoint.get("parent_run_signature") != localization.PARENT_RUN_SIGNATURE
        or checkpoint.get("completed_step") != localization.TARGET_COMPLETED_STEP
        or checkpoint.get("diagnostic_run_signature")
        != localization_summary.get("diagnostic_run_signature")
        or checkpoint.get("diagnostic_source_digest")
        != localization_source.get("canonical_payload_sha256")
        or checkpoint.get("runtime_digest")
        != localization_runtime.get("canonical_payload_sha256")
        or checkpoint.get("replay_trace_digest") != canonical_json_sha256(replay_trace)
        or checkpoint.get("model_state_sha256") != STEP100_MODEL_STATE_SHA256
        or observed_model_digest != STEP100_MODEL_STATE_SHA256
        or checkpoint.get("normalizer_state_sha256") != normalizer_state_sha256
        or observed_normalizer_digest != normalizer_state_sha256
        or checkpoint.get("inference_only") is not True
        or checkpoint.get("resume_supported") is not False
        or checkpoint.get("test_object_opened") is not False
    ):
        raise ValueError("step-100 checkpoint identity differs")
    return model_state


def validate_known_step100_reproduction(
    audit: Mapping[str, Any],
    retained_localization: Mapping[str, Any],
    *,
    tolerance: float = 1.0e-5,
) -> None:
    step50 = audit["checkpoints"]["step50"]
    step100 = audit["checkpoints"]["step100"]
    truth = audit["truth"]
    if step50["total_inverse_domain_violation_points"] != 0:
        raise RuntimeError("step-50 fresh map no longer reproduces its valid margin")
    if truth["total_inverse_domain_violation_points"] != 0:
        raise RuntimeError("matching truth crosses the inverse domain")
    observed = step100["first_inverse_domain_violation"]
    expected = retained_localization.get("first_failure")
    if not isinstance(observed, Mapping) or not isinstance(expected, Mapping):
        raise TypeError("step-100 fresh failure record is missing")
    exact_fields = ("case_key", "channel_index", "field", "row", "column")
    if any(observed.get(name) != expected.get(name) for name in exact_fields) or (
        expected.get("call") != 1
    ):
        raise RuntimeError("step-100 first fresh failure location differs")
    expected_normalized = expected.get("normalized_value", {}).get("value")
    expected_margin = expected.get("inverse_domain_margin", {}).get("value")
    if not isinstance(expected_normalized, (int, float)) or not isinstance(
        expected_margin, (int, float)
    ):
        raise TypeError("retained first-failure scalar is missing")
    if (
        abs(float(observed["normalized_value"]) - float(expected_normalized))
        > tolerance
        or abs(float(observed["inverse_domain_margin"]) - float(expected_margin))
        > tolerance
    ):
        raise RuntimeError("step-100 first fresh failure value differs")
    target_row = next(
        row
        for row in step100["inverse_domain_margin"]
        if row["case_key"] == expected["case_key"]
        and row["channel_index"] == expected["channel_index"]
    )
    if target_row["invalid_support"]["count"] != 66:
        raise RuntimeError("step-100 first case/channel support count differs")


def _source_manifest() -> dict[str, Any]:
    paths = (
        REPO_ROOT / "utility" / "time_dependent_no" / "realm_benchmark.py",
        REPO_ROOT / "utility" / "time_dependent_no" / "realm_ignithit.py",
        REPO_ROOT / "utility" / "time_dependent_no" / "realm_ffno.py",
        REPO_ROOT / "scripts" / "time_dependent_no" / "train_realm_ignithit_ffno.py",
        REPO_ROOT
        / "scripts"
        / "time_dependent_no"
        / "diagnose_realm_ignithit_decode_failure.py",
        Path(__file__).resolve(),
    )
    payload: dict[str, Any] = {
        "schema": "d088_ignithit_fresh_map_margin_source_v1",
        "files": [
            {
                "path": path.relative_to(REPO_ROOT).as_posix(),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in paths
        ],
    }
    payload["canonical_payload_sha256"] = _canonical_payload_sha256(payload)
    return payload


def _load_exact_localization_payloads(
    output_dir: Path,
) -> dict[str, Mapping[str, Any]]:
    localization.validate_file_hashes(output_dir, LOCALIZATION_FILE_SHA256)
    final_manifest = parent_trainer._load_json(output_dir / "final_hash_manifest.json")
    if final_manifest.get("files") != {
        key: value
        for key, value in LOCALIZATION_FILE_SHA256.items()
        if key != "final_hash_manifest.json"
    }:
        raise ValueError("localization final hash manifest differs")
    return {
        name: parent_trainer._load_json(output_dir / f"{name}.json")
        for name in (
            "contract",
            "parent_identity",
            "source_manifest",
            "runtime_manifest",
            "replay_trace",
            "localization",
            "summary",
        )
    }


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    localization.validate_file_hashes(
        args.parent_output_dir,
        localization.PARENT_FILE_SHA256,
    )
    localization_payloads = _load_exact_localization_payloads(
        args.localization_output_dir
    )
    prepare_new_output_directory(
        args.output_dir,
        data_root=args.data_root,
        parent_output_dir=args.parent_output_dir,
        localization_output_dir=args.localization_output_dir,
    )
    parent_config = parent_trainer._load_json(args.parent_output_dir / "config.json")
    parent_input = parent_trainer._load_json(
        args.parent_output_dir / "input_manifest.json"
    )
    parent_source = parent_trainer._load_json(
        args.parent_output_dir / "source_manifest.json"
    )
    manifest_payload = parent_trainer._load_json(args.manifest)
    metadata, current_input = localization._current_input_manifest(
        manifest_payload,
        data_root=args.data_root,
    )
    state_normalizer, _, _, normalizer_state = parent_trainer._normalizers_from_arrays(
        args.normalizer_arrays
    )
    normalizer_state_sha256 = parent_trainer.structured_state_sha256(normalizer_state)

    parent_trainer._configure_determinism()
    device = torch.device(parent_trainer.DEVICE)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("the paired audit requires one visible CUDA device")
    torch.cuda.set_device(device)
    current_parent_source = parent_trainer._source_manifest()
    current_localization_source = localization._diagnostic_source_manifest()
    current_runtime = parent_trainer._runtime_manifest(device)
    localization._require_exact_payload(
        parent_trainer.frozen_training_contract(),
        parent_config,
        name="training config",
    )
    localization._require_exact_payload(
        current_input,
        parent_input,
        name="input manifest",
    )
    localization._require_exact_payload(
        current_parent_source,
        parent_source,
        name="parent source",
    )
    localization._require_exact_payload(
        current_localization_source,
        localization_payloads["source_manifest"],
        name="localization source",
    )

    parent_last = torch.load(
        args.parent_output_dir / "last.pt",
        map_location="cpu",
        weights_only=False,
    )
    step100_checkpoint = torch.load(
        args.localization_output_dir / "step100_model.pt",
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(parent_last, Mapping) or not isinstance(
        step100_checkpoint, Mapping
    ):
        raise TypeError("checkpoint root must be a mapping")
    step50_state = validate_step50_checkpoint(parent_last)
    step100_state = validate_step100_checkpoint(
        step100_checkpoint,
        normalizer_state_sha256=normalizer_state_sha256,
        localization_summary=localization_payloads["summary"],
        localization_source=localization_payloads["source_manifest"],
        localization_runtime=localization_payloads["runtime_manifest"],
        replay_trace=localization_payloads["replay_trace"],
    )

    validation_normalized, _ = parent_trainer._load_normalized_trajectories(
        args.data_root,
        "val",
        VAL_GROUPS,
        state_normalizer,
        retain_native=False,
    )
    initial_state_cpu = validation_normalized[:, FRAME_START].contiguous()
    truth_normalized_cpu = validation_normalized[:, FRAME_TARGET].contiguous()
    frame0_normalized_sha256 = parent_trainer.structured_state_sha256(initial_state_cpu)
    frame1_truth_normalized_sha256 = parent_trainer.structured_state_sha256(
        truth_normalized_cpu
    )
    initial_state = initial_state_cpu.to(device)
    truth_normalized = truth_normalized_cpu.to(device)
    coordinates = normalize_realm_coordinates(
        torch.from_numpy(metadata.coords).unsqueeze(0).to(dtype=parent_trainer.DTYPE)
    ).to(device)
    model50 = RealmFFNO2d().to(device=device, dtype=parent_trainer.DTYPE)
    model100 = RealmFFNO2d().to(device=device, dtype=parent_trainer.DTYPE)
    for model in (model50, model100):
        if not parameter_count_within_reported_tolerance(
            trainable_parameter_count(model)
        ):
            raise RuntimeError("FFNO-M parameter count is outside frozen tolerance")
    model50.load_state_dict(step50_state, strict=True)
    model100.load_state_dict(step100_state, strict=True)

    started_at = time.monotonic()
    prediction50, prediction100 = infer_fresh_map_pair(
        model50,
        model100,
        initial_state,
        coordinates,
    )
    audit = summarize_fresh_map_margin_audit(
        prediction50,
        prediction100,
        truth_normalized,
        state_normalizer,
        case_keys=VAL_GROUPS,
    )
    validate_known_step100_reproduction(
        audit,
        localization_payloads["localization"],
    )

    contract = audit_contract()
    source = _source_manifest()
    identity: dict[str, Any] = {
        "schema": "d088_ignithit_fresh_map_margin_identity_v1",
        "run_id": RUN_ID,
        "parent_run_id": localization.PARENT_RUN_ID,
        "localization_run_id": localization.RUN_ID,
        "parent_run_signature": localization.PARENT_RUN_SIGNATURE,
        "parent_provenance": dict(localization.PARENT_PROVENANCE),
        "localization_run_signature": localization_payloads["summary"][
            "diagnostic_run_signature"
        ],
        "localization_final_hash_manifest_sha256": (LOCALIZATION_FINAL_MANIFEST_SHA256),
        "localization_source_digest": localization_payloads["source_manifest"][
            "canonical_payload_sha256"
        ],
        "localization_runtime_digest": localization_payloads["runtime_manifest"][
            "canonical_payload_sha256"
        ],
        "localization_replay_trace_digest": canonical_json_sha256(
            localization_payloads["replay_trace"]
        ),
        "step50_checkpoint_sha256": localization.PARENT_LAST_SHA256,
        "step50_model_state_sha256": STEP50_MODEL_STATE_SHA256,
        "step100_checkpoint_sha256": LOCALIZATION_FILE_SHA256["step100_model.pt"],
        "step100_model_state_sha256": STEP100_MODEL_STATE_SHA256,
        "normalizer_arrays_sha256": parent_trainer.NORMALIZER_ARRAYS_SHA256,
        "normalizer_state_sha256": normalizer_state_sha256,
        "ordered_validation_groups": list(VAL_GROUPS),
        "model_config": asdict(RealmFFNOConfig()),
    }
    identity["canonical_payload_sha256"] = _canonical_payload_sha256(identity)
    provenance = {
        "contract_digest": contract["canonical_payload_sha256"],
        "identity_digest": identity["canonical_payload_sha256"],
        "input_digest": current_input["canonical_payload_sha256"],
        "source_digest": source["canonical_payload_sha256"],
        "runtime_digest": current_runtime["canonical_payload_sha256"],
    }
    run_signature = canonical_json_sha256(provenance)
    stage_manifest: dict[str, Any] = {
        "schema": "d088_ignithit_fresh_map_margin_stage_v1",
        "frame0_normalized_sha256": frame0_normalized_sha256,
        "frame1_truth_normalized_sha256": frame1_truth_normalized_sha256,
        "step50_prediction_normalized_sha256": (
            parent_trainer.structured_state_sha256(prediction50)
        ),
        "step100_prediction_normalized_sha256": (
            parent_trainer.structured_state_sha256(prediction100)
        ),
        "shape": list(initial_state.shape),
        "dtype": str(initial_state.dtype),
        "device": str(initial_state.device),
        "calls_per_checkpoint": 1,
        "same_input_for_both_checkpoints": True,
        "input_mutation_detected": False,
    }
    stage_manifest["canonical_payload_sha256"] = _canonical_payload_sha256(
        stage_manifest
    )
    summary: dict[str, Any] = {
        "schema": "d088_ignithit_fresh_map_margin_summary_v1",
        "run_id": RUN_ID,
        "run_signature": run_signature,
        "provenance": provenance,
        "classification": audit["paired"]["classification"],
        "checkpoint_history_associated_margin_regression": audit["paired"][
            "checkpoint_history_associated_margin_regression"
        ],
        "step50_realm_npe_mean": audit["checkpoints"]["step50"]["normalized_error"][
            "realm_npe_mean"
        ],
        "step100_realm_npe_mean": audit["checkpoints"]["step100"]["normalized_error"][
            "realm_npe_mean"
        ],
        "step50_inverse_domain_violation_points": audit["checkpoints"]["step50"][
            "total_inverse_domain_violation_points"
        ],
        "step100_inverse_domain_violation_points": audit["checkpoints"]["step100"][
            "total_inverse_domain_violation_points"
        ],
        "truth_inverse_domain_violation_points": audit["truth"][
            "total_inverse_domain_violation_points"
        ],
        "first_step100_inverse_domain_violation": audit["checkpoints"]["step100"][
            "first_inverse_domain_violation"
        ],
        "known_step100_result_reproduced": True,
        "elapsed_seconds": time.monotonic() - started_at,
        "stopped_after_one_fresh_call_per_checkpoint": True,
        "test_object_opened": False,
        "residual_arm_executed": False,
        "training_executed": False,
        "anti_claims": list(audit["anti_claims"]),
    }
    retained = {
        "contract.json": contract,
        "identity.json": identity,
        "input_manifest.json": current_input,
        "source_manifest.json": source,
        "runtime_manifest.json": current_runtime,
        "stage_manifest.json": stage_manifest,
        "audit.json": audit,
        "summary.json": summary,
    }
    for name, payload in retained.items():
        parent_trainer._write_json_atomic(args.output_dir / name, payload)
    final_manifest = {
        "schema": "d088_ignithit_fresh_map_margin_final_hash_manifest_v1",
        "files": {name: sha256_file(args.output_dir / name) for name in retained},
        "self_hash_excluded": True,
    }
    parent_trainer._write_json_atomic(
        args.output_dir / "final_hash_manifest.json",
        final_manifest,
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary = run_audit(args)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
