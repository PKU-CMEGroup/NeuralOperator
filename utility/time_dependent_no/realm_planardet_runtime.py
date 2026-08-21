"""Training and evaluation contracts for the REALM PlanarDet residual PCNO.

The module is deliberately narrow: architecture, parameterization, precision,
optimizer family, sampling policy, and split membership are fixed.  A small
preregistration payload supplies only the resource-dependent width and budget
that cannot be frozen until the full-grid GPU smoke has completed.
"""

from __future__ import annotations

import gc
import math
import random
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import torch
from torch import nn

from utility.time_dependent_no.realm_benchmark import (
    MagnitudeEnvelope,
    RealmNormalizer,
    canonical_json_sha256,
    decoded_admissibility,
    decoded_boundedness,
    decoded_spatial_pearson,
    grouped_normalized_prediction_error,
    predict_one_call,
    predict_two_call_final,
)
from utility.time_dependent_no.realm_planardet import (
    BOX_COX_LAMBDA,
    CANONICAL_SPATIAL_SHAPE_YX,
    PLANARDET_FIELDS,
    PLANARDET_GROUPS,
    PRIMARY_BOX_COX_EPSILON,
    SCALE_STABILIZER,
    STD_CORRECTION,
    TRAJECTORY_CANONICAL_SHAPE,
    TRAJECTORY_DTYPE,
    load_planardet_trajectory,
    planardet_normalizer,
    sha256_file,
    trajectory_relative_path,
)

PREREGISTRATION_SCHEMA = "w26_l4_planardet_pd0_a3_preregistration_v1"
NORMALIZER_ARRAY_KEYS = frozenset(
    {
        "primary_mean",
        "primary_std",
        "primary_scale",
        "source_sensitivity_mean",
        "source_sensitivity_std",
        "source_sensitivity_scale",
        "raw_train_mean",
        "raw_train_std",
        "train_max_abs",
        "canonical_coordinates_yx",
    }
)
MODE_COUNTS_XY = (8, 8)
FC_DIM = 128
CHANNELS = len(PLANARDET_FIELDS)
ALLOWED_WIDTHS = frozenset({96, 128})
EXPECTED_PARAMETER_COUNTS = {96: 10_780_401, 128: 19_157_393}
MICROBATCH_SIZE = 1
EFFECTIVE_BATCH_SIZE = 7
PARAMETERIZATION = "residual"
MAX_LR = 1.0e-3
WEIGHT_DECAY = 0.0
ADAM_BETAS = (0.9, 0.999)
ADAM_EPS = 1.0e-8
ONECYCLE_PCT_START = 0.3
ONECYCLE_DIV_FACTOR = 25.0
ONECYCLE_FINAL_DIV_FACTOR = 10_000.0
VALIDATION_HORIZON = TRAJECTORY_CANONICAL_SHAPE[0] - 1
# Float32 reductions may differ by a few ulps across CPU kernels.  This still
# rejects any scientifically meaningful preprocessing or split drift.
CONTROL_TOLERANCE = 1.0e-6
BOUNDEDNESS_EXPANSION_FACTOR = 10.0
BOUNDARY_BAND_WIDTH_M = 5.0e-4
_HEX64 = re.compile(r"[0-9a-f]{64}")
_RUN_ID = re.compile(r"[a-z0-9][a-z0-9_-]{2,95}")


def _require_plain_int(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _require_finite_float(
    value: object,
    name: str,
    *,
    positive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "positive " if positive else ""
        raise ValueError(f"{name} must be a finite {qualifier}number")
    return result


def _require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True)
class PlanarDetTrainingContract:
    """The resource-dependent fields frozen after the full-grid A2 smoke."""

    run_id: str
    seed: int
    width: int
    total_steps: int
    one_call_steps: int
    validation_interval: int
    checkpoint_eligible_from_step: int
    competence_npe_ceiling: float
    max_wall_seconds: float
    smoke_result_sha256: str
    normalizer_arrays_sha256: str
    data_audit_final_manifest_sha256: str
    open_manifest_payload_sha256: str
    persistence_npe: float
    linear_extrapolation_npe: float

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or _RUN_ID.fullmatch(self.run_id) is None:
            raise ValueError("run_id must be a safe lowercase artifact identifier")
        _require_plain_int(self.seed, "seed")
        if self.width not in ALLOWED_WIDTHS:
            raise ValueError("width must be one of the two registered smoke envelopes")
        _require_plain_int(self.total_steps, "total_steps", minimum=97)
        _require_plain_int(self.one_call_steps, "one_call_steps", minimum=49)
        if self.total_steps - self.one_call_steps < 48:
            raise ValueError("two-call phase must cover every legal two-call window")
        _require_plain_int(self.validation_interval, "validation_interval", minimum=1)
        _require_plain_int(
            self.checkpoint_eligible_from_step,
            "checkpoint_eligible_from_step",
            minimum=1,
        )
        if self.checkpoint_eligible_from_step not in validation_steps(self):
            raise ValueError("checkpoint eligibility must begin at a validation step")
        if self.checkpoint_eligible_from_step - self.one_call_steps < 48:
            raise ValueError(
                "checkpoint eligibility must follow one complete two-call window cycle"
            )
        competence = _require_finite_float(
            self.competence_npe_ceiling,
            "competence_npe_ceiling",
            positive=True,
        )
        persistence = _require_finite_float(
            self.persistence_npe, "persistence_npe", positive=True
        )
        linear = _require_finite_float(
            self.linear_extrapolation_npe,
            "linear_extrapolation_npe",
            positive=True,
        )
        if competence >= min(persistence, linear):
            raise ValueError(
                "competence ceiling must strictly beat both frozen controls"
            )
        _require_finite_float(self.max_wall_seconds, "max_wall_seconds", positive=True)
        for name in (
            "smoke_result_sha256",
            "normalizer_arrays_sha256",
            "data_audit_final_manifest_sha256",
            "open_manifest_payload_sha256",
        ):
            _require_sha256(getattr(self, name), name)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> PlanarDetTrainingContract:
        expected = {field.name for field in cls.__dataclass_fields__.values()}
        expected.update({"schema", "canonical_payload_sha256"})
        if set(payload) != expected:
            raise ValueError("preregistration keys do not match the frozen schema")
        if payload.get("schema") != PREREGISTRATION_SCHEMA:
            raise ValueError("preregistration schema differs")
        digest = payload.get("canonical_payload_sha256")
        unsigned = {
            key: value
            for key, value in payload.items()
            if key != "canonical_payload_sha256"
        }
        if digest != canonical_json_sha256(unsigned):
            raise ValueError("preregistration canonical digest differs")
        return cls(**{name: payload[name] for name in asdict_fields(cls)})

    def payload(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": PREREGISTRATION_SCHEMA,
            **asdict(self),
        }
        result["canonical_payload_sha256"] = canonical_json_sha256(result)
        return result

    def frozen_training_config(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": "w26_l4_planardet_pd0_a3_training_config_v1",
            "run": asdict(self),
            "model": {
                "channels": CHANNELS,
                "mode_counts_xy": list(MODE_COUNTS_XY),
                "layers": [self.width] * 5,
                "fc_dim": FC_DIM,
                "activation": "gelu",
                "zero_initialize_head": True,
                "parameterization": PARAMETERIZATION,
                "expected_trainable_parameters": EXPECTED_PARAMETER_COUNTS[self.width],
            },
            "precision": {
                "parameters_geometry_optimizer": "float32",
                "autocast": "bfloat16",
            },
            "training": {
                "microbatch_size": MICROBATCH_SIZE,
                "effective_batch_size": EFFECTIVE_BATCH_SIZE,
                "accumulation_steps": EFFECTIVE_BATCH_SIZE,
                "one_call_steps": self.one_call_steps,
                "two_call_steps": self.total_steps - self.one_call_steps,
                "two_call_first_proposal": "detached",
                "loss": "sum_of_five_group_mean_squared_errors",
                "sampling": "phase_local_shuffled_cycles_over_all_legal_windows",
                "all_seven_cases_per_optimizer_step": True,
            },
            "optimizer": {
                "name": "AdamW",
                "max_lr": MAX_LR,
                "weight_decay": WEIGHT_DECAY,
                "betas": list(ADAM_BETAS),
                "eps": ADAM_EPS,
                "gradient_clip": None,
                "gradient_norm_monitoring": "global_l2_before_optimizer_step",
            },
            "scheduler": {
                "name": "OneCycleLR",
                "total_steps": self.total_steps,
                "pct_start": ONECYCLE_PCT_START,
                "anneal_strategy": "cos",
                "cycle_momentum": False,
                "div_factor": ONECYCLE_DIV_FACTOR,
                "final_div_factor": ONECYCLE_FINAL_DIV_FACTOR,
            },
            "validation": {
                "horizon": VALIDATION_HORIZON,
                "steps": list(validation_steps(self)),
                "selection_metric": "teacher_forced_realm_npe_mean",
                "eligible_from_step": self.checkpoint_eligible_from_step,
                "strict_improvement": True,
                "competence_npe_ceiling": self.competence_npe_ceiling,
                "must_strictly_beat": ["persistence", "linear_extrapolation"],
                "population": "one_released_validation_trajectory",
            },
            "scope": {
                "train_trajectories": 7,
                "validation_trajectories": 1,
                "test_objects_allowed": False,
                "finish_budget_even_if_validation_is_poor": True,
            },
        }
        payload["canonical_payload_sha256"] = canonical_json_sha256(payload)
        return payload


def asdict_fields(cls: type[PlanarDetTrainingContract]) -> tuple[str, ...]:
    return tuple(cls.__dataclass_fields__)


def validation_steps(contract: PlanarDetTrainingContract) -> tuple[int, ...]:
    result = {1, contract.total_steps}
    result.update(
        range(
            contract.validation_interval,
            contract.total_steps + 1,
            contract.validation_interval,
        )
    )
    return tuple(sorted(result))


@dataclass(frozen=True)
class StepWindow:
    phase: str
    calls: int
    frame_start: int
    case_indices: tuple[int, ...]
    phase_cycle: int


def scheduled_step_window(
    step: int,
    *,
    one_call_steps: int,
    seed: int,
    frame_count: int = TRAJECTORY_CANONICAL_SHAPE[0],
    case_count: int = EFFECTIVE_BATCH_SIZE,
) -> StepWindow:
    """Return a resumable, uniformly cycling window and all-case presentation."""

    _require_plain_int(step, "step", minimum=1)
    _require_plain_int(one_call_steps, "one_call_steps", minimum=1)
    _require_plain_int(seed, "seed")
    _require_plain_int(frame_count, "frame_count", minimum=3)
    _require_plain_int(case_count, "case_count", minimum=1)
    if step <= one_call_steps:
        phase = "one_call"
        calls = 1
        phase_step = step - 1
    else:
        phase = "two_call"
        calls = 2
        phase_step = step - one_call_steps - 1
    window_count = frame_count - calls
    phase_cycle, offset = divmod(phase_step, window_count)
    phase_code = 1 if calls == 1 else 2
    frame_order = list(range(window_count))
    random.Random(seed + phase_code * 1_000_003 + phase_cycle).shuffle(frame_order)
    case_order = list(range(case_count))
    random.Random(seed + 10_000_019 + step * 97).shuffle(case_order)
    return StepWindow(
        phase=phase,
        calls=calls,
        frame_start=frame_order[offset],
        case_indices=tuple(case_order),
        phase_cycle=phase_cycle,
    )


def grouped_planardet_mse(
    prediction: torch.Tensor,
    truth: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return the released five-group sum used for optimization."""

    if prediction.shape != truth.shape or prediction.ndim != 4:
        raise ValueError("prediction and truth must share [batch, channel, y, x]")
    slices = PLANARDET_GROUPS.slices(expected_channels=prediction.shape[1])
    total = prediction.new_zeros(())
    by_group: dict[str, torch.Tensor] = {}
    for name, channel_slice in slices.items():
        value = (prediction[:, channel_slice] - truth[:, channel_slice]).square().mean()
        by_group[name] = value
        total = total + value
    return total, by_group


def scaled_microbatch_loss(
    loss: torch.Tensor,
    *,
    effective_batch_size: int = EFFECTIVE_BATCH_SIZE,
) -> torch.Tensor:
    if loss.ndim != 0 or not loss.is_floating_point():
        raise ValueError("microbatch loss must be a floating scalar")
    _require_plain_int(effective_batch_size, "effective_batch_size", minimum=1)
    return loss / effective_batch_size


def training_prediction(
    model: nn.Module,
    current: torch.Tensor,
    coordinates: torch.Tensor,
    *,
    calls: int,
) -> torch.Tensor:
    if calls == 1:
        return predict_one_call(
            model,
            current,
            coordinates,
            parameterization=PARAMETERIZATION,
        )
    if calls == 2:
        return predict_two_call_final(
            model,
            current,
            coordinates,
            parameterization=PARAMETERIZATION,
        )
    raise ValueError("training calls must be exactly one or two")


@dataclass(frozen=True)
class PlanarDetNormalizerBundle:
    mean: torch.Tensor
    scale: torch.Tensor
    train_max_abs: torch.Tensor
    canonical_coordinates_yx: torch.Tensor
    source_sha256: str

    def normalizer(self, channel_axis: int, *, dtype: torch.dtype) -> RealmNormalizer:
        return planardet_normalizer(
            self.mean.to(dtype=dtype),
            self.scale.to(dtype=dtype),
            channel_axis=channel_axis,
            dtype=dtype,
        )

    def checkpoint_state(self) -> dict[str, Any]:
        return {
            "mean": self.mean.detach().cpu().clone(),
            "scale": self.scale.detach().cpu().clone(),
            "train_max_abs": self.train_max_abs.detach().cpu().clone(),
            "transformed_channels": tuple(range(8)),
            "box_cox_lambda": BOX_COX_LAMBDA,
            "box_cox_epsilon": PRIMARY_BOX_COX_EPSILON,
            "std_correction": STD_CORRECTION,
            "scale_stabilizer": SCALE_STABILIZER,
            "source_sha256": self.source_sha256,
        }


def load_normalizer_bundle(
    path: Path,
    *,
    expected_sha256: str,
    expected_coordinates_yx: np.ndarray,
) -> PlanarDetNormalizerBundle:
    if sha256_file(path) != _require_sha256(expected_sha256, "expected_sha256"):
        raise ValueError("normalizer arrays SHA-256 differs from preregistration")
    with np.load(path, allow_pickle=False) as arrays:
        if set(arrays.files) != NORMALIZER_ARRAY_KEYS:
            raise ValueError("normalizer array keys differ from the audited contract")
        mean = np.asarray(arrays["primary_mean"])
        scale = np.asarray(arrays["primary_scale"])
        train_max_abs = np.asarray(arrays["train_max_abs"])
        coordinates = np.asarray(arrays["canonical_coordinates_yx"])
    for name, value in (
        ("mean", mean),
        ("scale", scale),
        ("train_max_abs", train_max_abs),
    ):
        if value.shape != (CHANNELS,) or value.dtype != np.dtype("float64"):
            raise ValueError(f"normalizer {name} must be a float64 13-vector")
        if not np.isfinite(value).all():
            raise ValueError(f"normalizer {name} must be finite")
    if not (scale > 0.0).all() or not (train_max_abs >= 0.0).all():
        raise ValueError("normalizer scale or train envelope has an invalid domain")
    expected_coordinates = np.asarray(expected_coordinates_yx)
    if (
        coordinates.dtype != np.dtype("float32")
        or coordinates.shape != (2, *CANONICAL_SPATIAL_SHAPE_YX)
        or expected_coordinates.shape != coordinates.shape
        or expected_coordinates.dtype != coordinates.dtype
        or not np.array_equal(coordinates, expected_coordinates)
    ):
        raise ValueError("audited normalizer coordinates differ from release metadata")
    return PlanarDetNormalizerBundle(
        mean=torch.from_numpy(mean.copy()),
        scale=torch.from_numpy(scale.copy()),
        train_max_abs=torch.from_numpy(train_max_abs.copy()),
        canonical_coordinates_yx=torch.from_numpy(coordinates.copy()),
        source_sha256=expected_sha256,
    )


def load_normalized_trajectories(
    data_root: Path,
    *,
    split: str,
    groups: Sequence[str],
    normalizer: RealmNormalizer,
    retain_native: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Load one split into CPU tensors, never accepting a sealed split name."""

    if split not in {"train", "val"}:
        raise ValueError("only train and validation trajectories may be loaded")
    shape = (len(groups), *TRAJECTORY_CANONICAL_SHAPE)
    normalized = torch.empty(shape, dtype=torch.float32)
    native_out = torch.empty(shape, dtype=torch.float32) if retain_native else None
    for index, group in enumerate(groups):
        relative = trajectory_relative_path(split, group)
        native_array = load_planardet_trajectory(
            data_root.joinpath(*PurePosixPath(relative).parts), canonical=True
        )
        if native_array.dtype != TRAJECTORY_DTYPE:
            raise ValueError("trajectory dtype differs from the audited contract")
        native = torch.from_numpy(native_array)
        encoded = normalizer.encode(native)
        if not bool(torch.isfinite(encoded).all()):
            raise ValueError("normalizer produced a nonfinite trajectory")
        normalized[index].copy_(encoded)
        if native_out is not None:
            native_out[index].copy_(native)
        del encoded, native, native_array
        gc.collect()
    return normalized, native_out


def inverse_box_cox_min_margin(
    normalized: torch.Tensor,
    bundle: PlanarDetNormalizerBundle,
    *,
    channel_axis: int,
) -> float:
    """Minimum ``1 + lambda * transformed`` over the eight species channels."""

    axis = channel_axis if channel_axis >= 0 else normalized.ndim + channel_axis
    if axis < 0 or axis >= normalized.ndim or normalized.shape[axis] != CHANNELS:
        raise ValueError("channel_axis does not identify the 13 PlanarDet channels")
    shape = [1] * normalized.ndim
    shape[axis] = CHANNELS
    mean = bundle.mean.to(device=normalized.device, dtype=normalized.dtype).reshape(
        shape
    )
    scale = bundle.scale.to(device=normalized.device, dtype=normalized.dtype).reshape(
        shape
    )
    transformed = normalized * scale + mean
    selection = [slice(None)] * normalized.ndim
    selection[axis] = slice(0, 8)
    base = 1.0 + BOX_COX_LAMBDA * transformed[tuple(selection)]
    return float(base.min().item())


def validation_control_summary(normalized_sequence: torch.Tensor) -> dict[str, Any]:
    """Frozen truth-input persistence and normalized-linear controls."""

    if (
        normalized_sequence.ndim != 4
        or normalized_sequence.shape[0] < 3
        or normalized_sequence.shape[1] != CHANNELS
    ):
        raise ValueError("normalized sequence must have [time>=3, 13, y, x]")
    if not bool(torch.isfinite(normalized_sequence).all()):
        raise ValueError("normalized control truth must be finite")
    slices = PLANARDET_GROUPS.slices(expected_channels=CHANNELS)
    group_sums = {
        kind: {name: 0.0 for name in slices}
        for kind in ("persistence", "linear_normalized")
    }
    channel_sums = {
        kind: torch.zeros(CHANNELS, dtype=torch.float64) for kind in group_sums
    }
    calls = normalized_sequence.shape[0] - 1
    for frame in range(calls):
        current = normalized_sequence[frame]
        truth = normalized_sequence[frame + 1]
        predictions = {
            "persistence": current,
            "linear_normalized": (
                current
                if frame == 0
                else 2.0 * current - normalized_sequence[frame - 1]
            ),
        }
        for kind, prediction in predictions.items():
            square = (prediction - truth).to(torch.float64).square()
            channel_sums[kind] += square.mean(dim=(1, 2))
            for name, channel_slice in slices.items():
                group_sums[kind][name] += float(square[channel_slice].mean().item())
    result: dict[str, Any] = {
        "schema": "w26_l4_planardet_truth_input_controls_v1",
        "call_count": calls,
        "linear_control_space": "transformed_normalized_state",
    }
    for kind, group_values in group_sums.items():
        grouped = {name: value / calls for name, value in group_values.items()}
        result[kind] = {
            "realm_npe_mean": sum(grouped.values()),
            "group_mse_mean": grouped,
            "per_channel_mse_mean": (channel_sums[kind] / calls).tolist(),
        }
    result["canonical_payload_sha256"] = canonical_json_sha256(result)
    return result


def validate_frozen_controls(
    summary: Mapping[str, Any],
    contract: PlanarDetTrainingContract,
) -> None:
    for key, expected in (
        ("persistence", contract.persistence_npe),
        ("linear_normalized", contract.linear_extrapolation_npe),
    ):
        row = summary.get(key)
        if not isinstance(row, Mapping):
            raise TypeError("validation control summary is malformed")
        actual = row.get("realm_npe_mean")
        if not isinstance(actual, (int, float)) or not math.isclose(
            float(actual), expected, rel_tol=0.0, abs_tol=CONTROL_TOLERANCE
        ):
            raise ValueError(f"{key} control differs from preregistration")


def planardet_boundary_mask(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    band_width: float = BOUNDARY_BAND_WIDTH_M,
) -> torch.Tensor:
    """Return a fixed physical-width mask for monotone axes in either direction."""

    if x.ndim != 1 or y.ndim != 1 or min(x.numel(), y.numel()) < 2:
        raise ValueError("x and y must be one-dimensional axes with at least two cells")
    if not x.is_floating_point() or not y.is_floating_point():
        raise TypeError("x and y must be floating tensors")
    if not bool(torch.isfinite(x).all()) or not bool(torch.isfinite(y).all()):
        raise ValueError("x and y must be finite")
    if not (
        bool((torch.diff(x) > 0.0).all()) or bool((torch.diff(x) < 0.0).all())
    ) or not (bool((torch.diff(y) > 0.0).all()) or bool((torch.diff(y) < 0.0).all())):
        raise ValueError("x and y must be strictly monotone")
    if not math.isfinite(band_width) or band_width <= 0.0:
        raise ValueError("band_width must be finite and positive")
    x_edge = (x - x.min() <= band_width) | (x.max() - x <= band_width)
    y_edge = (y - y.min() <= band_width) | (y.max() - y <= band_width)
    mask = y_edge[:, None] | x_edge[None, :]
    if not bool(mask.any()) or bool(mask.all()):
        raise ValueError("boundary band must leave a nonempty interior")
    return mask


def _finite_json_list(values: torch.Tensor) -> list[float | None]:
    return [
        float(value) if math.isfinite(float(value)) else None
        for value in values.detach().cpu().reshape(-1).tolist()
    ]


def _finite_json_nested(values: torch.Tensor) -> list[list[float | None]]:
    if values.ndim != 2:
        raise ValueError("nested JSON conversion requires a two-dimensional tensor")
    return [_finite_json_list(row) for row in values]


def _finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None


def summarize_planardet_predictions(
    prediction_normalized: torch.Tensor,
    truth_normalized: torch.Tensor,
    prediction_decoded: torch.Tensor,
    truth_decoded: torch.Tensor,
    current_decoded: torch.Tensor,
    *,
    bundle: PlanarDetNormalizerBundle,
    boundary_mask: torch.Tensor,
) -> dict[str, Any]:
    """Summarize an accepted contiguous sequence of PlanarDet proposals."""

    expected = prediction_normalized.shape
    if (
        prediction_normalized.ndim != 4
        or expected[0] == 0
        or expected[1] != CHANNELS
        or truth_normalized.shape != expected
        or prediction_decoded.shape != expected
        or truth_decoded.shape != expected
        or current_decoded.shape != expected
    ):
        raise ValueError("prediction summaries require matching [call, 13, y, x]")
    if boundary_mask.dtype != torch.bool or tuple(boundary_mask.shape) != tuple(
        expected[-2:]
    ):
        raise ValueError("boundary mask must be bool with the prediction grid shape")
    if not bool(torch.isfinite(truth_normalized).all()) or not bool(
        torch.isfinite(truth_decoded).all()
    ):
        raise ValueError("truth sequences must be finite")

    normalized_finite = bool(torch.isfinite(prediction_normalized).all())
    decoded_finite = bool(torch.isfinite(prediction_decoded).all())
    errors = grouped_normalized_prediction_error(
        prediction_normalized.unsqueeze(0),
        truth_normalized.unsqueeze(0),
        groups=PLANARDET_GROUPS,
    )
    correlation = decoded_spatial_pearson(
        prediction_decoded.unsqueeze(0), truth_decoded.unsqueeze(0)
    )
    status_counts: Counter[str] = Counter()
    for case in correlation.statuses:
        for call in case:
            status_counts.update(call)

    normalized_square = (prediction_normalized - truth_normalized).square()
    normalized_channel_mse = normalized_square.mean(dim=(-2, -1))
    decoded_difference = prediction_decoded - truth_decoded
    decoded_numerator = torch.linalg.vector_norm(decoded_difference.flatten(2), dim=2)
    decoded_denominator = torch.linalg.vector_norm(truth_decoded.flatten(2), dim=2)
    decoded_relative_l2 = torch.where(
        decoded_denominator > 0.0,
        decoded_numerator / decoded_denominator,
        torch.full_like(decoded_numerator, torch.nan),
    )

    envelope = MagnitudeEnvelope(
        max_abs=bundle.train_max_abs.to(
            dtype=prediction_decoded.dtype, device=prediction_decoded.device
        ),
        quantile=1.0,
        channel_axis=0,
    )
    admissible_calls = 0
    bounded_calls = 0
    violations: Counter[str] = Counter()
    boundedness_max_ratio = torch.zeros(CHANNELS, dtype=torch.float64)
    inverse_margins: list[float] = []
    for call in range(expected[0]):
        admissibility = decoded_admissibility(
            prediction_decoded[call],
            groups=PLANARDET_GROUPS,
            channel_axis=0,
        )
        boundedness = decoded_boundedness(
            prediction_decoded[call],
            envelope,
            expansion_factor=BOUNDEDNESS_EXPANSION_FACTOR,
            channel_axis=0,
        )
        admissible_calls += int(admissibility.admissible)
        bounded_calls += int(boundedness.bounded)
        violations.update(
            "nonpositive_pMax" if value == "nonpositive_pressure" else value
            for value in admissibility.violations
        )
        boundedness_max_ratio = torch.maximum(
            boundedness_max_ratio,
            torch.as_tensor(boundedness.max_ratio_by_channel, dtype=torch.float64),
        )
        inverse_margins.append(
            inverse_box_cox_min_margin(
                prediction_normalized[call], bundle, channel_axis=0
            )
        )

    pmax_increment = prediction_decoded[:, 12] - current_decoded[:, 12]
    finite_pmax_increment = torch.isfinite(pmax_increment)
    pmax_decrease = finite_pmax_increment & (pmax_increment < 0.0)
    finite_decrements = pmax_increment[finite_pmax_increment]
    maximum_pmax_decrease = (
        max(0.0, -float(finite_decrements.min().item()))
        if finite_decrements.numel()
        else math.nan
    )
    boundary_square = decoded_difference.square().reshape(-1, *expected[-2:])
    boundary_square = torch.where(
        torch.isfinite(boundary_square),
        boundary_square,
        torch.full_like(boundary_square, torch.inf),
    )
    boundary_mse = float(boundary_square[:, boundary_mask].mean().item())
    interior_mse = float(boundary_square[:, ~boundary_mask].mean().item())
    boundary_ratio = (
        boundary_mse / interior_mse
        if interior_mse > 0.0
        else (1.0 if boundary_mse == 0.0 else math.inf)
    )
    correlation_by_call = torch.nanmean(correlation.values[0], dim=1)
    minimum_inverse_margin = (
        min(inverse_margins)
        if all(math.isfinite(value) for value in inverse_margins)
        else math.nan
    )
    return {
        "call_count": expected[0],
        "realm_npe_mean": _finite_or_none(errors.realm_npe_mean),
        "realm_npe_sum_source": _finite_or_none(errors.realm_npe_sum_source),
        "npe_total_by_call": _finite_json_list(errors.total_per_call[0]),
        "npe_group_by_call": {
            name: _finite_json_list(value[0])
            for name, value in sorted(errors.grouped_per_call.items())
        },
        "normalized_mse_by_call_channel": _finite_json_nested(normalized_channel_mse),
        "decoded_relative_l2_by_call_channel": _finite_json_nested(decoded_relative_l2),
        "decoded_correlation_case_first": (
            correlation.population_case_first_mean
            if math.isfinite(correlation.population_case_first_mean)
            else None
        ),
        "decoded_correlation_by_call": _finite_json_list(correlation_by_call),
        "decoded_correlation_by_call_channel": _finite_json_nested(
            correlation.values[0]
        ),
        "pMax_decoded_correlation_by_call": _finite_json_list(
            correlation.values[0, :, 12]
        ),
        "correlation_status_counts": dict(sorted(status_counts.items())),
        "all_normalized_finite": normalized_finite,
        "all_decoded_finite": decoded_finite,
        "admissible_call_count": admissible_calls,
        "all_released_state_admissible": admissible_calls == expected[0],
        "admissibility_violation_counts": dict(sorted(violations.items())),
        "bounded_call_count": bounded_calls,
        "all_bounded_10x_train_max": bounded_calls == expected[0],
        "max_boundedness_ratio_by_channel": _finite_json_list(boundedness_max_ratio),
        "inverse_box_cox_min_margin_by_call": [
            value if math.isfinite(value) else None for value in inverse_margins
        ],
        "inverse_box_cox_min_margin": _finite_or_none(minimum_inverse_margin),
        "pMax_decrease_from_input_count": int(pmax_decrease.sum().item()),
        "pMax_nonfinite_increment_count": int(
            torch.count_nonzero(~finite_pmax_increment).item()
        ),
        "pMax_max_decrease_from_input_pa": _finite_or_none(maximum_pmax_decrease),
        "all_pMax_nondecreasing_from_input": bool(finite_pmax_increment.all())
        and not bool(pmax_decrease.any()),
        "decoded_boundary_band": {
            "width_m": BOUNDARY_BAND_WIDTH_M,
            "mse": _finite_or_none(boundary_mse),
            "interior_mse": _finite_or_none(interior_mse),
            "boundary_to_interior_ratio": (
                boundary_ratio if math.isfinite(boundary_ratio) else None
            ),
        },
    }


def competence_gate(
    summary: Mapping[str, Any],
    contract: PlanarDetTrainingContract,
) -> dict[str, bool]:
    score = summary.get("realm_npe_mean")
    numeric_score = (
        float(score)
        if isinstance(score, (int, float)) and math.isfinite(float(score))
        else math.inf
    )
    gates = {
        "below_preregistered_npe_ceiling": numeric_score
        <= contract.competence_npe_ceiling,
        "strictly_beats_persistence": numeric_score < contract.persistence_npe,
        "strictly_beats_linear_extrapolation": numeric_score
        < contract.linear_extrapolation_npe,
        "all_normalized_finite": bool(summary.get("all_normalized_finite")),
        "all_decoded_finite": bool(summary.get("all_decoded_finite")),
        "all_released_state_admissible": bool(
            summary.get("all_released_state_admissible")
        ),
        "all_bounded_10x_train_max": bool(summary.get("all_bounded_10x_train_max")),
        "all_pMax_nondecreasing_from_input": bool(
            summary.get("all_pMax_nondecreasing_from_input")
        ),
    }
    gates["all_gates_pass"] = all(gates.values())
    return gates


__all__ = [
    "ADAM_BETAS",
    "ADAM_EPS",
    "ALLOWED_WIDTHS",
    "BOUNDARY_BAND_WIDTH_M",
    "BOUNDEDNESS_EXPANSION_FACTOR",
    "CHANNELS",
    "CONTROL_TOLERANCE",
    "EFFECTIVE_BATCH_SIZE",
    "EXPECTED_PARAMETER_COUNTS",
    "FC_DIM",
    "MAX_LR",
    "MICROBATCH_SIZE",
    "MODE_COUNTS_XY",
    "NORMALIZER_ARRAY_KEYS",
    "ONECYCLE_DIV_FACTOR",
    "ONECYCLE_FINAL_DIV_FACTOR",
    "ONECYCLE_PCT_START",
    "PARAMETERIZATION",
    "PREREGISTRATION_SCHEMA",
    "VALIDATION_HORIZON",
    "WEIGHT_DECAY",
    "PlanarDetNormalizerBundle",
    "PlanarDetTrainingContract",
    "StepWindow",
    "competence_gate",
    "grouped_planardet_mse",
    "inverse_box_cox_min_margin",
    "load_normalized_trajectories",
    "load_normalizer_bundle",
    "planardet_boundary_mask",
    "scaled_microbatch_loss",
    "scheduled_step_window",
    "summarize_planardet_predictions",
    "training_prediction",
    "validate_frozen_controls",
    "validation_control_summary",
    "validation_steps",
]
