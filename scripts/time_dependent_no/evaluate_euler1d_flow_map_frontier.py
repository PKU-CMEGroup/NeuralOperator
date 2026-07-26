"""Evaluate fixed-stride Euler flow maps at common physical endpoints.

The evaluator is intentionally training-free. It restores frozen residual FNO
checkpoints, composes their raw feed-forward maps, and compares every path with
the serialized solver truth at the same saved frame. Failed paths are reported
at both the requested horizon and their last endpoint shared with the paired
path; another learned model is never used as ground truth.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch

if __package__:
    from scripts.time_dependent_no.train_euler1d_target_ladder import (
        PrimitiveNormalizer,
        build_model,
        json_ready,
        pressure_front_top2_metrics_np,
        sha256_file,
    )
else:
    from train_euler1d_target_ladder import (
        PrimitiveNormalizer,
        build_model,
        json_ready,
        pressure_front_top2_metrics_np,
        sha256_file,
    )

from utility.time_dependent_no.euler1d import make_euler1d_batch
from utility.time_dependent_no.euler1d_data import (
    Euler1DNPZ,
    conservative_to_primitive_np,
    load_euler1d_npz,
    primitive_to_conservative_np,
)
from utility.time_dependent_no.euler1d_targets import make_target_adapter

EPS = 1.0e-12
PHYSICAL_SCALES = np.array([1.0, 1.0, 2.5], dtype=np.float64)


def _checkpoint_normalizer(
    checkpoint: dict[str, Any],
    prefix: str,
    coordinates: str,
    normalization: str,
) -> PrimitiveNormalizer:
    mean = torch.as_tensor(checkpoint[f"{prefix}_mean"], dtype=torch.float32).reshape(
        1, 1, 3
    )
    std = torch.as_tensor(checkpoint[f"{prefix}_std"], dtype=torch.float32).reshape(
        1, 1, 3
    )
    return PrimitiveNormalizer(
        mean=mean,
        std=std,
        coordinates=coordinates,
        normalization=normalization,
    )


def load_frozen_residual_checkpoint(
    path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, torch.nn.Module, PrimitiveNormalizer, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("model") != "fno" or checkpoint.get("target") != "residual":
        raise ValueError("resolution gate requires an FNO residual checkpoint")
    checkpoint_args = dict(checkpoint.get("args", {}))
    required_args = (
        "fno_modes",
        "fno_width",
        "fno_layers",
        "fno_fc_dim",
        "fno_pad_ratio",
        "input_coordinates",
        "input_normalization",
        "loss_coordinates",
        "loss_normalization",
        "recurrent_coordinates",
        "step_stride",
        "target_supervision",
    )
    missing = [key for key in required_args if key not in checkpoint_args]
    if missing:
        raise ValueError(f"checkpoint is missing required arguments: {missing}")
    expected_contract = {
        "input_coordinates": "conservative",
        "input_normalization": "fixed_physical",
        "loss_coordinates": "conservative",
        "loss_normalization": "fixed_physical",
        "recurrent_coordinates": "conservative",
        "target_supervision": "state",
    }
    for key, expected in expected_contract.items():
        if checkpoint_args[key] != expected:
            raise ValueError(
                f"checkpoint {key}={checkpoint_args[key]!r}, expected {expected!r}"
            )
    namespace = argparse.Namespace(**checkpoint_args)
    input_normalizer = _checkpoint_normalizer(
        checkpoint,
        "input_normalizer",
        checkpoint_args["input_coordinates"],
        checkpoint_args["input_normalization"],
    )
    loss_normalizer = _checkpoint_normalizer(
        checkpoint,
        "normalizer",
        checkpoint_args["loss_coordinates"],
        checkpoint_args["loss_normalization"],
    )
    model = build_model("fno", "residual", namespace, input_normalizer).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != int(checkpoint["parameter_count"]):
        raise ValueError("restored parameter count differs from checkpoint metadata")
    adapter = make_target_adapter("residual").to(device)
    return model, adapter, loss_normalizer, checkpoint


def _validate_checkpoint_contract(
    checkpoint: dict[str, Any],
    *,
    stride: int,
    expected_cases: np.ndarray,
    data_sha256: str,
    saved_time_sha256: str,
    split: str,
) -> dict[str, Any]:
    args = checkpoint["args"]
    expected = {
        "model": "fno",
        "target": "residual",
        "fno_width": 64,
        "fno_modes": 24,
        "fno_layers": 4,
        "input_coordinates": "conservative",
        "loss_coordinates": "conservative",
        "recurrent_coordinates": "conservative",
        "input_normalization": "fixed_physical",
        "loss_normalization": "fixed_physical",
        "step_stride": stride,
    }
    mismatches = {
        name: {"actual": args.get(name), "expected": value}
        for name, value in expected.items()
        if args.get(name) != value
    }
    if mismatches:
        raise ValueError(f"stride {stride} violates the frozen contract: {mismatches}")
    split_key = "val_cases" if split == "validation" else "test_cases"
    if not np.array_equal(
        np.asarray(checkpoint[split_key], dtype=np.int64),
        expected_cases,
    ):
        raise ValueError(f"stride {stride} uses a different frozen split")
    if checkpoint.get("data_sha256") != data_sha256:
        raise ValueError(f"stride {stride} uses a different dataset")
    if checkpoint.get("saved_time_sha256") != saved_time_sha256:
        raise ValueError(f"stride {stride} uses different saved times")
    return expected


@dataclass(frozen=True)
class PathRollout:
    states: dict[int, np.ndarray]
    completed_frame: int
    requested_frame: int
    termination_reason: str | None

    @property
    def completed(self) -> bool:
        return self.completed_frame == self.requested_frame


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        nargs=2,
        action="append",
        metavar=("STRIDE", "PATH"),
        default=[],
        help="Saved-frame stride and frozen checkpoint path; repeat per stride.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--evaluation-regime",
        choices=("matched_id", "ood"),
        default="matched_id",
    )
    parser.add_argument(
        "--training-data-path",
        type=Path,
        help="Original checkpoint dataset; required only for explicit OOD evaluation.",
    )
    parser.add_argument(
        "--ood-label",
        help="Predeclared OOD suite label; required only in OOD mode.",
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=[32, 64, 96],
    )
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--split-seed", type=int, default=20260707)
    parser.add_argument("--train-cases", type=int, default=384)
    parser.add_argument("--val-cases", type=int, default=64)
    parser.add_argument("--test-cases", type=int, default=64)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument(
        "--wave-audit-only",
        action="store_true",
        help="Validate the dataset and write truth-only wave diagnostics.",
    )
    parser.add_argument("--boundary-margin-cells", type=int, default=8)
    parser.add_argument("--min-front-peak-fraction", type=float, default=0.05)
    parser.add_argument(
        "--perturbation-cases",
        type=int,
        default=16,
        help="Number of frozen-split cases used for local amplification; zero disables.",
    )
    parser.add_argument(
        "--perturbation-start-frames",
        type=int,
        nargs="+",
        default=[0, 32, 64],
    )
    parser.add_argument(
        "--semigroup-start-frames",
        type=int,
        nargs="+",
        default=[0, 32, 64],
        help="Truth-state starts for one-large-step versus composed-map defects.",
    )
    parser.add_argument("--perturbation-relative-norm", type=float, default=1.0e-4)
    parser.add_argument("--perturbation-seed", type=int, default=20260718)
    parser.add_argument("--timing-warmup", type=int, default=20)
    parser.add_argument(
        "--timing-repeats",
        type=int,
        default=100,
        help="Synchronized repeats per learned-map timing measurement; zero disables.",
    )
    parser.add_argument("--throughput-batch-size", type=int, default=64)
    parser.add_argument(
        "--direct-curve-max-horizon",
        type=int,
        default=0,
        help=(
            "Maximum saved frame for dense direct-path reliability curves; "
            "zero disables the audit."
        ),
    )
    parser.add_argument(
        "--direct-curves-only",
        action="store_true",
        help="Skip compositions and supporting diagnostics after direct curves.",
    )
    parser.add_argument(
        "--error-budgets",
        type=float,
        nargs="+",
        default=[0.01, 0.02, 0.05, 0.1],
        help="Fixed-scale conservative relative-L2 reliability thresholds.",
    )
    parser.add_argument(
        "--checkpoint-set-label",
        default="primary_selected",
        help="Provenance label for selected or validation-only checkpoint sets.",
    )
    return parser.parse_args(argv)


def build_composition_paths(
    target_stride: int,
    horizon: int,
    available_strides: Sequence[int] = (1, 2, 4, 8),
) -> dict[str, tuple[int, ...]]:
    """Return direct and smaller-stride paths ending at the same frame."""

    if target_stride < 1 or horizon < target_stride:
        raise ValueError("target stride and horizon must be positive and compatible")
    if horizon % target_stride:
        raise ValueError("horizon must be divisible by target stride")
    available = sorted(set(int(stride) for stride in available_strides))
    if target_stride not in available:
        raise ValueError(f"missing direct stride {target_stride}")
    paths: dict[str, tuple[int, ...]] = {}
    for stride in sorted(
        (
            stride
            for stride in available
            if stride <= target_stride and target_stride % stride == 0
        ),
        reverse=True,
    ):
        label = (
            f"s{target_stride}_direct"
            if stride == target_stride
            else f"s{target_stride}_from_s{stride}"
        )
        paths[label] = (stride,) * (horizon // stride)
    return paths


def run_call_path(
    initial_state: np.ndarray,
    call_strides: Sequence[int],
    advance: Callable[[np.ndarray, int, int], np.ndarray],
    is_admissible: Callable[[np.ndarray], bool] | None = None,
) -> PathRollout:
    """Advance one path until its endpoint or the first raw invalid state."""

    requested_frame = int(sum(call_strides))
    frame = 0
    current = np.asarray(initial_state).copy()
    states = {0: current.copy()}
    termination_reason: str | None = None
    for stride_value in call_strides:
        stride = int(stride_value)
        if stride < 1:
            raise ValueError("call strides must be positive")
        proposed = np.asarray(advance(current, stride, frame))
        if proposed.shape != current.shape:
            raise ValueError("advance changed the state shape")
        if not np.all(np.isfinite(proposed)):
            termination_reason = "nonfinite_raw_state"
            break
        if is_admissible is not None and not is_admissible(proposed):
            termination_reason = "nonpositive_raw_state"
            break
        frame += stride
        current = proposed.copy()
        states[frame] = current
    return PathRollout(
        states=states,
        completed_frame=frame,
        requested_frame=requested_frame,
        termination_reason=termination_reason,
    )


def _euler_state_is_admissible(state: np.ndarray) -> bool:
    return bool(
        state.shape[-1] == 3
        and np.all(state[..., 0] > 0.0)
        and np.all(state[..., 2] > 0.0)
    )


def last_common_frame(
    first: PathRollout,
    second: PathRollout,
    requested_frame: int,
) -> int:
    common = set(first.states).intersection(second.states)
    valid = [frame for frame in common if frame <= requested_frame]
    if not valid:
        raise ValueError("paths do not share an initial frame")
    return max(valid)


def semigroup_error_bounds_hold(
    defect: float,
    direct_truth_error: float,
    composed_truth_error: float,
    *,
    tolerance: float = 1.0e-12,
) -> bool:
    lower = abs(direct_truth_error - composed_truth_error)
    upper = direct_truth_error + composed_truth_error
    return lower - tolerance <= defect <= upper + tolerance


def normalized_amplification_rate(amplification: float, physical_dt: float) -> float:
    if amplification <= 0.0 or physical_dt <= 0.0:
        raise ValueError("amplification and physical_dt must be positive")
    return math.log(amplification) / physical_dt


def fixed_norm_conservative_perturbation(
    primitive: np.ndarray,
    gamma: float,
    relative_norm: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    """Draw one isotropic perturbation with an exact fixed-scale relative norm."""

    if relative_norm <= 0.0:
        raise ValueError("relative_norm must be positive")
    conservative = primitive_to_conservative_np(
        np.asarray(primitive, dtype=np.float64),
        gamma,
    )
    scaled = conservative / PHYSICAL_SCALES
    state_norm = float(np.linalg.norm(scaled.reshape(-1)))
    if state_norm <= 0.0:
        raise ValueError("cannot perturb a zero-norm state")
    direction = rng.standard_normal(scaled.shape)
    direction_norm = float(np.linalg.norm(direction.reshape(-1)))
    if direction_norm <= 0.0:
        raise RuntimeError("sampled a zero perturbation direction")
    delta_scaled = direction * (relative_norm * state_norm / direction_norm)
    perturbed = conservative_to_primitive_np(
        (scaled + delta_scaled) * PHYSICAL_SCALES,
        gamma,
    )
    if not _euler_state_is_admissible(perturbed):
        raise ValueError(
            "fixed-norm perturbation is not admissible; lower the perturbation norm"
        )
    realized = float(np.linalg.norm(delta_scaled.reshape(-1)) / max(state_norm, EPS))
    return perturbed, realized


def fixed_scale_perturbation_amplification(
    input_base: np.ndarray,
    input_perturbed: np.ndarray,
    output_base: np.ndarray,
    output_perturbed: np.ndarray,
    gamma: float,
    physical_dt: float,
) -> dict[str, float]:
    """Measure one local amplification factor in fixed-scale conservative norm."""

    def distance(first: np.ndarray, second: np.ndarray) -> float:
        first_scaled = primitive_to_conservative_np(first, gamma) / PHYSICAL_SCALES
        second_scaled = primitive_to_conservative_np(second, gamma) / PHYSICAL_SCALES
        return float(np.linalg.norm((first_scaled - second_scaled).reshape(-1)))

    input_norm = distance(input_perturbed, input_base)
    output_norm = distance(output_perturbed, output_base)
    if input_norm <= 0.0:
        raise ValueError("input perturbation norm must be positive")
    amplification = max(output_norm, EPS) / input_norm
    return {
        "input_perturbation_l2": input_norm,
        "output_perturbation_l2": output_norm,
        "amplification": amplification,
        "log_amplification_per_physical_time": normalized_amplification_rate(
            amplification,
            physical_dt,
        ),
    }


def _top2_fronts(
    pressure: np.ndarray,
    *,
    separation_cells: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    gradients = np.abs(np.diff(pressure, axis=-1))
    positions = np.empty((gradients.shape[0], 2), dtype=np.int64)
    strengths = np.empty((gradients.shape[0], 2), dtype=np.float64)
    for row_id, row in enumerate(gradients):
        chosen: list[int] = []
        for face in np.argsort(row)[::-1]:
            candidate = int(face)
            if all(
                abs(candidate - previous) >= separation_cells for previous in chosen
            ):
                chosen.append(candidate)
            if len(chosen) == 2:
                break
        if len(chosen) < 2:
            chosen.extend([chosen[-1] if chosen else 0] * (2 - len(chosen)))
        positions[row_id] = chosen
        strengths[row_id] = row[chosen]
    return positions, strengths


def truth_wave_audit(
    source: Euler1DNPZ,
    case_ids: np.ndarray,
    horizons: Sequence[int],
    *,
    boundary_margin_cells: int = 8,
    min_peak_fraction: float = 0.05,
) -> dict[str, Any]:
    """Measure whether strong, changing truth structure remains in the domain."""

    ids = np.asarray(case_ids, dtype=np.int64)
    if ids.size == 0:
        raise ValueError("wave audit requires at least one case")
    pressure = source.data[ids, ..., 2].astype(np.float64)
    case_peak = np.max(np.abs(np.diff(pressure, axis=-1)), axis=(1, 2))
    result: dict[str, Any] = {
        "num_cases": int(ids.size),
        "boundary_margin_cells": boundary_margin_cells,
        "min_peak_fraction": min_peak_fraction,
        "horizons": {},
    }
    for horizon in sorted(set(int(value) for value in horizons)):
        if horizon < 1 or horizon >= source.num_frames:
            raise ValueError(f"invalid wave-audit horizon: {horizon}")
        positions, strengths = _top2_fronts(pressure[:, horizon])
        interior = (positions >= boundary_margin_cells) & (
            positions < source.num_cells - boundary_margin_cells - 1
        )
        strong = strengths >= min_peak_fraction * np.maximum(case_peak[:, None], EPS)
        active = np.any(interior & strong, axis=1)
        previous = max(0, horizon - 8)
        current_state = source.data[ids, horizon].astype(np.float64)
        previous_state = source.data[ids, previous].astype(np.float64)
        change = np.linalg.norm(
            (current_state - previous_state).reshape(ids.size, -1), axis=1
        ) / np.maximum(np.linalg.norm(current_state.reshape(ids.size, -1), axis=1), EPS)
        result["horizons"][str(horizon)] = {
            "interior_active_fraction": float(np.mean(active)),
            "front_peak_fraction_median": float(
                np.median(strengths[:, 0] / np.maximum(case_peak, EPS))
            ),
            "primary_front_face_median": float(np.median(positions[:, 0])),
            "state_change_from_frame": previous,
            "state_change_to_frame": horizon,
            "state_change_from_minus8_relative_l2_median": float(np.median(change)),
            # Retained as a legacy alias for existing D032/D039 artifacts.
            "state_change_relative_l2_median": float(np.median(change)),
        }
    return result


def _relative_l2(prediction: np.ndarray, truth: np.ndarray) -> float:
    return float(
        np.linalg.norm((prediction - truth).reshape(-1))
        / max(np.linalg.norm(truth.reshape(-1)), EPS)
    )


def _fixed_scale_conservative_relative_l2(
    prediction: np.ndarray,
    truth: np.ndarray,
    gamma: float,
) -> float:
    pred = primitive_to_conservative_np(prediction, gamma) / PHYSICAL_SCALES
    target = primitive_to_conservative_np(truth, gamma) / PHYSICAL_SCALES
    return _relative_l2(pred, target)


def _state_metrics(
    prediction: np.ndarray,
    truth: np.ndarray,
    initial: np.ndarray,
    x: np.ndarray,
    gamma: float,
) -> dict[str, float]:
    primitive_error = _relative_l2(prediction, truth)
    conservative_error = _fixed_scale_conservative_relative_l2(prediction, truth, gamma)
    top2 = pressure_front_top2_metrics_np(
        prediction[None],
        truth[None],
        x,
        min_separation_cells=8,
    )
    truth_gradient = np.abs(np.diff(truth[:, 2]))
    shock_faces = np.argsort(truth_gradient)[-2:]
    smooth = np.ones(truth.shape[0], dtype=bool)
    for face in shock_faces:
        smooth[max(0, int(face) - 4) : min(truth.shape[0], int(face) + 6)] = False
    smooth_error = (
        _relative_l2(prediction[smooth], truth[smooth])
        if np.any(smooth)
        else float("nan")
    )
    boundary = np.r_[0:8, truth.shape[0] - 8 : truth.shape[0]]
    dx = float(np.mean(np.diff(x)))
    pred_total = primitive_to_conservative_np(prediction, gamma).sum(axis=0) * dx
    truth_total = primitive_to_conservative_np(truth, gamma).sum(axis=0) * dx
    _ = initial  # Kept in the public helper signature for artifact compatibility.
    total_mismatch_error = float(
        np.linalg.norm(pred_total - truth_total) / max(np.linalg.norm(truth_total), EPS)
    )
    return {
        "primitive_relative_l2": primitive_error,
        "fixed_scale_conservative_relative_l2": conservative_error,
        "smooth_region_relative_l2": smooth_error,
        "boundary_relative_l2": _relative_l2(prediction[boundary], truth[boundary]),
        "shock_top2_position_mae": float(top2["position_assignment_mae"][0]),
        "shock_top2_strength_relative_l1": float(top2["strength_relative_l1"][0]),
        "global_conserved_total_mismatch_relative_l2": total_mismatch_error,
        # Legacy alias: this is not a boundary-flux conservation-closure residual.
        "conservative_budget_relative_l2": total_mismatch_error,
        "min_density": float(np.min(prediction[:, 0])),
        "min_pressure": float(np.min(prediction[:, 2])),
    }


def trajectory_increment_errors(
    predicted_previous: np.ndarray,
    predicted_current: np.ndarray,
    truth_previous: np.ndarray,
    truth_current: np.ndarray,
    gamma: float,
) -> dict[str, float]:
    """Compare an on-policy learned increment with the stored truth increment."""

    pred_previous = (
        primitive_to_conservative_np(predicted_previous, gamma) / PHYSICAL_SCALES
    )
    pred_current = (
        primitive_to_conservative_np(predicted_current, gamma) / PHYSICAL_SCALES
    )
    target_previous = (
        primitive_to_conservative_np(truth_previous, gamma) / PHYSICAL_SCALES
    )
    target_current = (
        primitive_to_conservative_np(truth_current, gamma) / PHYSICAL_SCALES
    )
    residual = (pred_current - pred_previous) - (target_current - target_previous)
    residual_norm = float(np.linalg.norm(residual.reshape(-1)))
    truth_increment_norm = float(
        np.linalg.norm((target_current - target_previous).reshape(-1))
    )
    truth_state_norm = float(np.linalg.norm(target_current.reshape(-1)))
    return {
        "trajectory_increment_relative_l2": (
            residual_norm / max(truth_increment_norm, EPS)
        ),
        "trajectory_increment_state_normalized_l2": (
            residual_norm / max(truth_state_norm, EPS)
        ),
    }


def evaluate_direct_curves(
    source: Euler1DNPZ,
    case_ids: np.ndarray,
    models: dict[int, tuple[torch.nn.Module, torch.nn.Module]],
    max_horizon: int,
    device: torch.device,
    *,
    checkpoint_set_label: str,
) -> list[dict[str, Any]]:
    """Roll each frozen stride once and retain every direct-path call endpoint."""

    if max_horizon < 1 or max_horizon >= source.num_frames:
        raise ValueError("direct-curve horizon must be a saved frame")
    rows: list[dict[str, Any]] = []
    ids = np.asarray(case_ids, dtype=np.int64)
    for stride, (model, adapter) in sorted(models.items()):
        if max_horizon % stride:
            raise ValueError(
                f"direct-curve horizon {max_horizon} is not divisible by stride {stride}"
            )
        current = np.asarray(source.data[ids, 0], dtype=np.float64).copy()
        current_conservative = primitive_to_conservative_np(current, source.gamma)
        active = np.ones(ids.size, dtype=bool)
        cumulative_integral = np.zeros(ids.size, dtype=np.float64)
        previous_error = np.zeros(ids.size, dtype=np.float64)
        previous_time = np.asarray(source.t[ids, 0], dtype=np.float64)
        for target_frame in range(stride, max_horizon + 1, stride):
            active_positions = np.flatnonzero(active)
            if not active_positions.size:
                break
            start_frame = target_frame - stride
            active_ids = ids[active_positions]
            proposed, proposed_conservative = _predict_model_batch_state(
                source,
                active_ids,
                current[active_positions],
                start_frame,
                stride,
                model,
                adapter,
                device,
                conservative=current_conservative[active_positions],
            )
            for local_position, case_position in enumerate(active_positions):
                case_id = int(ids[case_position])
                prediction = np.asarray(proposed[local_position], dtype=np.float64)
                finite = bool(np.all(np.isfinite(prediction)))
                admissible = finite and _euler_state_is_admissible(prediction)
                physical_time = float(source.t[case_id, target_frame])
                row: dict[str, Any] = {
                    "checkpoint_set_label": checkpoint_set_label,
                    "case_id": case_id,
                    "stride": stride,
                    "path": f"s{stride}_direct",
                    "call_index": target_frame // stride,
                    "start_frame": start_frame,
                    "frame": target_frame,
                    "physical_time": physical_time,
                    "physical_elapsed_time": (
                        physical_time - float(source.t[case_id, 0])
                    ),
                    "raw_state_finite": finite,
                    "raw_state_admissible": admissible,
                    "completed_step": admissible,
                    "termination_reason": (
                        None
                        if admissible
                        else (
                            "nonfinite_raw_state"
                            if not finite
                            else "nonpositive_raw_state"
                        )
                    ),
                }
                if not admissible:
                    row.update(
                        {
                            key: float("nan")
                            for key in (
                                "primitive_relative_l2",
                                "fixed_scale_conservative_relative_l2",
                                "smooth_region_relative_l2",
                                "boundary_relative_l2",
                                "shock_top2_position_mae",
                                "shock_top2_strength_relative_l1",
                                "global_conserved_total_mismatch_relative_l2",
                                "conservative_budget_relative_l2",
                                "min_density",
                                "min_pressure",
                                "trajectory_increment_relative_l2",
                                "trajectory_increment_state_normalized_l2",
                                "time_integrated_fixed_scale_conservative_relative_l2",
                                "time_mean_fixed_scale_conservative_relative_l2",
                            )
                        }
                    )
                    active[case_position] = False
                    rows.append(row)
                    continue
                truth_current = np.asarray(
                    source.data[case_id, target_frame], dtype=np.float64
                )
                truth_previous = np.asarray(
                    source.data[case_id, start_frame], dtype=np.float64
                )
                metrics = _state_metrics(
                    prediction,
                    truth_current,
                    np.asarray(source.data[case_id, 0], dtype=np.float64),
                    np.asarray(source.x[case_id], dtype=np.float64),
                    source.gamma,
                )
                metrics.update(
                    trajectory_increment_errors(
                        current[case_position],
                        prediction,
                        truth_previous,
                        truth_current,
                        source.gamma,
                    )
                )
                elapsed = physical_time - float(previous_time[case_position])
                cumulative_integral[case_position] += (
                    0.5
                    * elapsed
                    * (
                        previous_error[case_position]
                        + metrics["fixed_scale_conservative_relative_l2"]
                    )
                )
                metrics["time_integrated_fixed_scale_conservative_relative_l2"] = float(
                    cumulative_integral[case_position]
                )
                metrics["time_mean_fixed_scale_conservative_relative_l2"] = float(
                    cumulative_integral[case_position]
                    / max(row["physical_elapsed_time"], EPS)
                )
                row.update(metrics)
                rows.append(row)
                current[case_position] = prediction
                current_conservative[case_position] = proposed_conservative[
                    local_position
                ]
                previous_error[case_position] = metrics[
                    "fixed_scale_conservative_relative_l2"
                ]
                previous_time[case_position] = physical_time
    return rows


def summarize_direct_curves(
    rows: list[dict[str, Any]],
    horizons: Sequence[int],
    *,
    num_cases: int,
) -> list[dict[str, Any]]:
    """Aggregate direct curves at declared common saved-frame endpoints."""

    summary: list[dict[str, Any]] = []
    path_keys = sorted(
        {
            (
                str(row["checkpoint_set_label"]),
                str(row["path"]),
                int(row["stride"]),
            )
            for row in rows
        }
    )
    metric_names = (
        "primitive_relative_l2",
        "fixed_scale_conservative_relative_l2",
        "smooth_region_relative_l2",
        "boundary_relative_l2",
        "shock_top2_position_mae",
        "shock_top2_strength_relative_l1",
        "global_conserved_total_mismatch_relative_l2",
        "conservative_budget_relative_l2",
        "trajectory_increment_relative_l2",
        "trajectory_increment_state_normalized_l2",
        "time_integrated_fixed_scale_conservative_relative_l2",
        "time_mean_fixed_scale_conservative_relative_l2",
    )
    for label, path, stride in path_keys:
        stride_rows = [
            row
            for row in rows
            if str(row["checkpoint_set_label"]) == label
            and str(row["path"]) == path
            and int(row["stride"]) == stride
        ]
        if not stride_rows:
            continue
        for horizon in sorted(set(int(value) for value in horizons)):
            if horizon % stride:
                raise ValueError(
                    f"summary horizon {horizon} is not divisible by stride {stride}"
                )
            endpoint = [
                row
                for row in stride_rows
                if int(row["frame"]) == horizon and row["completed_step"]
            ]
            terminated = {
                int(row["case_id"])
                for row in stride_rows
                if int(row["frame"]) <= horizon and not row["completed_step"]
            }
            aggregate: dict[str, Any] = {
                "checkpoint_set_label": label,
                "path": path,
                "stride": stride,
                "horizon": horizon,
                "num_cases": num_cases,
                "num_completed": len(endpoint),
                "completion_fraction": len(endpoint) / max(num_cases, 1),
                "raw_admissibility_failures_by_horizon": len(terminated),
            }
            for name in metric_names:
                values = np.asarray(
                    [float(row[name]) for row in endpoint], dtype=np.float64
                )
                finite_values = values[np.isfinite(values)]
                aggregate[f"{name}_mean"] = (
                    float(np.mean(finite_values))
                    if finite_values.size
                    else float("nan")
                )
                if name == "fixed_scale_conservative_relative_l2":
                    aggregate[f"{name}_median"] = (
                        float(np.median(finite_values))
                        if finite_values.size
                        else float("nan")
                    )
                    aggregate[f"{name}_p95"] = (
                        float(np.quantile(finite_values, 0.95))
                        if finite_values.size
                        else float("nan")
                    )
            summary.append(aggregate)
    return summary


def error_budget_survival_rows(
    rows: list[dict[str, Any]],
    error_budgets: Sequence[float],
    *,
    max_horizon: int,
) -> list[dict[str, Any]]:
    """Record first observed budget exceedance or raw admissibility failure."""

    budgets = sorted(set(float(value) for value in error_budgets))
    if not budgets or any(value <= 0.0 for value in budgets):
        raise ValueError("error budgets must be positive")
    groups: dict[tuple[str, str, int, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["checkpoint_set_label"]),
            str(row["path"]),
            int(row["stride"]),
            int(row["case_id"]),
        )
        groups.setdefault(key, []).append(row)
    output: list[dict[str, Any]] = []
    for (label, path, stride, case_id), group in sorted(groups.items()):
        ordered = sorted(group, key=lambda row: int(row["frame"]))
        invalid = next(
            (row for row in ordered if not row["completed_step"]),
            None,
        )
        last_valid = next(
            (row for row in reversed(ordered) if row["completed_step"]),
            None,
        )
        for budget in budgets:
            exceed = next(
                (
                    row
                    for row in ordered
                    if row["completed_step"]
                    and float(row["fixed_scale_conservative_relative_l2"]) > budget
                ),
                None,
            )
            candidates = [row for row in (exceed, invalid) if row is not None]
            event = (
                min(candidates, key=lambda row: int(row["frame"]))
                if candidates
                else None
            )
            if event is None:
                event_kind = "right_censored"
                survival_frame = (
                    int(last_valid["frame"]) if last_valid is not None else 0
                )
                survival_time = (
                    float(last_valid["physical_elapsed_time"])
                    if last_valid is not None
                    else 0.0
                )
            else:
                event_kind = (
                    str(event["termination_reason"])
                    if not event["completed_step"]
                    else "error_budget_exceeded"
                )
                survival_frame = int(event["frame"])
                survival_time = float(event["physical_elapsed_time"])
            output.append(
                {
                    "checkpoint_set_label": label,
                    "path": path,
                    "stride": stride,
                    "case_id": case_id,
                    "error_budget": budget,
                    "event_observed": event is not None,
                    "event_kind": event_kind,
                    "event_frame": (
                        int(event["frame"]) if event is not None else float("nan")
                    ),
                    "event_physical_time": (
                        float(event["physical_elapsed_time"])
                        if event is not None
                        else float("nan")
                    ),
                    "survival_frame": survival_frame,
                    "survival_physical_time": survival_time,
                    "model_calls_until_event_or_censoring": (survival_frame // stride),
                    "first_error_exceed_frame": (
                        int(exceed["frame"]) if exceed is not None else float("nan")
                    ),
                    "raw_admissibility_failure_frame": (
                        int(invalid["frame"]) if invalid is not None else float("nan")
                    ),
                    "right_censored": event is None,
                    "completed_max_horizon": (
                        last_valid is not None
                        and int(last_valid["frame"]) == max_horizon
                    ),
                    "max_horizon": max_horizon,
                    "endpoint_observation_only": True,
                }
            )
    return output


def _frozen_split(
    source: Euler1DNPZ,
    *,
    split_seed: int,
    train_cases: int,
    val_cases: int,
    test_cases: int,
    split: str,
) -> np.ndarray:
    total = train_cases + val_cases + test_cases
    if total > source.num_cases:
        raise ValueError("requested split exceeds the available cases")
    permutation = np.random.default_rng(split_seed).permutation(source.num_cases)
    validation = np.sort(permutation[train_cases : train_cases + val_cases])
    test = np.sort(permutation[train_cases + val_cases : total])
    return validation if split == "validation" else test


def _saved_time_sha256(source: Euler1DNPZ) -> str:
    return hashlib.sha256(np.ascontiguousarray(source.t).view(np.uint8)).hexdigest()


def _checkpoint_epoch(checkpoint: dict[str, Any]) -> Any:
    """Read current trainer metadata while preserving legacy checkpoint support."""

    return checkpoint.get("checkpoint_epoch", checkpoint.get("epoch"))


def _validate_ood_compatibility(
    training: Euler1DNPZ,
    evaluation: Euler1DNPZ,
) -> None:
    if training.num_cells != evaluation.num_cells:
        raise ValueError("OOD data must preserve the checkpoint grid size")
    if training.gamma != evaluation.gamma:
        raise ValueError("OOD data must preserve gamma")
    if not np.array_equal(training.t[0], evaluation.t[0]):
        raise ValueError("OOD data must preserve every saved physical time")
    training_dx = np.diff(training.x.astype(np.float64), axis=1)
    evaluation_dx = np.diff(evaluation.x.astype(np.float64), axis=1)
    reference_dx = float(np.median(training_dx))
    tolerance = 2.0e-6 * max(1.0, abs(reference_dx))
    if (
        float(np.max(np.abs(training_dx - reference_dx))) > tolerance
        or float(np.max(np.abs(evaluation_dx - reference_dx))) > tolerance
    ):
        raise ValueError("OOD data must preserve the uniform grid spacing")


def _validated_ood_metadata(
    source: Euler1DNPZ,
    expected_label: str,
) -> tuple[dict[str, Any], tuple[str, ...], str, dict[str, int]]:
    """Validate and expose the immutable per-case OOD provenance contract."""

    try:
        raw_contract = source.metadata["ood_contract_json"]
        raw_labels = np.asarray(source.metadata["ood_regime"])
        generator_sha256 = str(source.metadata["ood_generator_source_sha256"])
    except KeyError as exc:
        raise ValueError(
            f"OOD dataset is missing provenance key {exc.args[0]}"
        ) from exc
    if not isinstance(raw_contract, str):
        raise ValueError("ood_contract_json must be a scalar JSON string")
    try:
        contract = json.loads(raw_contract)
    except json.JSONDecodeError as exc:
        raise ValueError("ood_contract_json is not valid JSON") from exc
    if not isinstance(contract, dict) or contract.get("label") != expected_label:
        raise ValueError("--ood-label does not match the dataset OOD contract")
    regimes = contract.get("regimes")
    if not isinstance(regimes, dict) or not regimes:
        raise ValueError("OOD contract must declare at least one regime")
    if raw_labels.shape != (source.num_cases,):
        raise ValueError("ood_regime must contain exactly one label per case")
    labels = tuple(
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in raw_labels.tolist()
    )
    if any(not label for label in labels):
        raise ValueError("ood_regime labels must be nonempty")
    unknown = set(labels) - set(regimes)
    if unknown:
        raise ValueError(f"ood_regime contains undeclared labels: {sorted(unknown)}")
    counts = {label: labels.count(label) for label in sorted(regimes)}
    expected_per_regime = int(contract.get("cases_per_regime", 0))
    if expected_per_regime < 1 or any(
        count != expected_per_regime for count in counts.values()
    ):
        raise ValueError("OOD regime counts do not match cases_per_regime")
    if len(generator_sha256) != 64:
        raise ValueError("ood_generator_source_sha256 must be a SHA-256 digest")
    try:
        int(generator_sha256, 16)
    except ValueError as exc:
        raise ValueError("ood_generator_source_sha256 must be hexadecimal") from exc
    return contract, labels, generator_sha256, counts


def _attach_ood_regimes(
    rows: list[dict[str, Any]],
    labels: tuple[str, ...] | None,
) -> None:
    if labels is None:
        return
    for row in rows:
        if "case_id" not in row:
            continue
        case_id = int(row["case_id"])
        if case_id not in range(len(labels)):
            raise ValueError(f"row case_id is outside the OOD dataset: {case_id}")
        row["ood_regime"] = labels[case_id]


def _summarize_paths_by_ood_regime(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for regime in sorted({str(row["ood_regime"]) for row in rows}):
        summaries = _summarize_paths(
            [row for row in rows if str(row["ood_regime"]) == regime]
        )
        for summary in summaries:
            summary["ood_regime"] = regime
        output.extend(summaries)
    return output


def _summarize_direct_curves_by_ood_regime(
    rows: list[dict[str, Any]],
    horizons: Sequence[int],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for regime in sorted({str(row["ood_regime"]) for row in rows}):
        regime_rows = [row for row in rows if str(row["ood_regime"]) == regime]
        num_cases = len({int(row["case_id"]) for row in regime_rows})
        summaries = summarize_direct_curves(
            regime_rows,
            horizons,
            num_cases=num_cases,
        )
        for summary in summaries:
            summary["ood_regime"] = regime
        output.extend(summaries)
    return output


def _select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _checkpoint_paths(entries: Sequence[Sequence[str]]) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for raw_stride, raw_path in entries:
        stride = int(raw_stride)
        if stride < 1:
            raise ValueError("checkpoint strides must be positive")
        if stride in result:
            raise ValueError(f"duplicate checkpoint stride: {stride}")
        result[stride] = Path(raw_path)
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(json_ready(row), sort_keys=True))
            handle.write("\n")


def _summarize_paths(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["path"]), int(row["horizon"])), []).append(row)
    summary: list[dict[str, Any]] = []
    for (path, horizon), group in sorted(groups.items()):
        complete = [row for row in group if row["completed_horizon"]]
        summary.append(
            {
                "path": path,
                "horizon": horizon,
                "target_stride": int(group[0]["target_stride"]),
                "call_stride": int(group[0]["call_stride"]),
                "num_calls": int(group[0]["num_calls"]),
                "num_cases": len(group),
                "num_completed": len(complete),
                "completion_fraction": len(complete) / len(group),
                "primitive_relative_l2_mean": (
                    float(np.mean([row["primitive_relative_l2"] for row in complete]))
                    if complete
                    else float("nan")
                ),
                "fixed_scale_conservative_relative_l2_mean": (
                    float(
                        np.mean(
                            [
                                row["fixed_scale_conservative_relative_l2"]
                                for row in complete
                            ]
                        )
                    )
                    if complete
                    else float("nan")
                ),
                "fixed_scale_conservative_relative_l2_median": (
                    float(
                        np.median(
                            [
                                row["fixed_scale_conservative_relative_l2"]
                                for row in complete
                            ]
                        )
                    )
                    if complete
                    else float("nan")
                ),
                "fixed_scale_conservative_relative_l2_p95": (
                    float(
                        np.quantile(
                            [
                                row["fixed_scale_conservative_relative_l2"]
                                for row in complete
                            ],
                            0.95,
                        )
                    )
                    if complete
                    else float("nan")
                ),
                "shock_top2_position_mae_mean": (
                    float(np.mean([row["shock_top2_position_mae"] for row in complete]))
                    if complete
                    else float("nan")
                ),
            }
        )
    return summary


def _normalized_state_distance(
    first: np.ndarray,
    second: np.ndarray,
    truth: np.ndarray,
    gamma: float,
) -> float:
    first_conservative = primitive_to_conservative_np(first, gamma) / PHYSICAL_SCALES
    second_conservative = primitive_to_conservative_np(second, gamma) / PHYSICAL_SCALES
    truth_conservative = primitive_to_conservative_np(truth, gamma) / PHYSICAL_SCALES
    return float(
        np.linalg.norm((first_conservative - second_conservative).reshape(-1))
        / max(np.linalg.norm(truth_conservative.reshape(-1)), EPS)
    )


def _make_model_batch(
    source: Euler1DNPZ,
    case_ids: np.ndarray,
    primitive: np.ndarray,
    start_frame: int,
    stride: int,
    device: torch.device,
) -> Any:
    ids = np.asarray(case_ids, dtype=np.int64)
    states = np.asarray(primitive, dtype=np.float32)
    if states.shape != (ids.size, source.num_cells, 3):
        raise ValueError("primitive batch shape does not match the selected cases")
    target_frame = start_frame + stride
    if start_frame < 0 or target_frame >= source.num_frames:
        raise ValueError("model call exceeds the serialized trajectory")
    current = torch.from_numpy(np.ascontiguousarray(states)).to(device)
    x = torch.from_numpy(np.ascontiguousarray(source.x[ids], dtype=np.float32)).to(
        device
    )
    dt = torch.from_numpy(
        np.ascontiguousarray(
            source.t[ids, target_frame] - source.t[ids, start_frame],
            dtype=np.float32,
        )
    ).to(device)
    left = torch.from_numpy(
        np.ascontiguousarray(source.left_states[ids], dtype=np.float32)
    ).to(device)
    right = torch.from_numpy(
        np.ascontiguousarray(source.right_states[ids], dtype=np.float32)
    ).to(device)
    return make_euler1d_batch(
        current,
        x,
        dt,
        gamma=source.gamma,
        left_boundary_primitive=left,
        right_initial_primitive=right,
    )


def _predict_model_batch(
    source: Euler1DNPZ,
    case_ids: np.ndarray,
    primitive: np.ndarray,
    start_frame: int,
    stride: int,
    model: torch.nn.Module,
    adapter: Any,
    device: torch.device,
) -> np.ndarray:
    prediction, _conservative = _predict_model_batch_state(
        source,
        case_ids,
        primitive,
        start_frame,
        stride,
        model,
        adapter,
        device,
    )
    return prediction


def _predict_model_batch_state(
    source: Euler1DNPZ,
    case_ids: np.ndarray,
    primitive: np.ndarray,
    start_frame: int,
    stride: int,
    model: torch.nn.Module,
    adapter: Any,
    device: torch.device,
    *,
    conservative: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return raw primitive and conservative outputs for exact recurrence."""

    batch = _make_model_batch(
        source,
        case_ids,
        primitive,
        start_frame,
        stride,
        device,
    )
    if conservative is not None:
        values = np.asarray(conservative, dtype=np.float32)
        if values.shape != np.asarray(primitive).shape:
            raise ValueError("conservative recurrence shape mismatch")
        batch = replace(
            batch,
            current_conservative_state=torch.from_numpy(
                np.ascontiguousarray(values)
            ).to(device),
        )
    with torch.inference_mode():
        decoded = adapter(model(batch), batch)
    return (
        decoded.primitive.detach().cpu().numpy().astype(np.float64),
        decoded.conservative.detach().cpu().numpy().astype(np.float64),
    )


def run_model_call_path(
    source: Euler1DNPZ,
    case_id: int,
    initial_primitive: np.ndarray,
    call_strides: Sequence[int],
    models: dict[int, tuple[torch.nn.Module, Any]],
    device: torch.device,
    *,
    absolute_start_frame: int = 0,
) -> PathRollout:
    """Compose frozen maps while retaining decoded conservative state exactly."""

    requested_frame = int(sum(call_strides))
    local_frame = 0
    current_primitive = np.asarray(initial_primitive, dtype=np.float64).copy()
    current_conservative = primitive_to_conservative_np(current_primitive, source.gamma)
    states = {0: current_primitive.copy()}
    termination_reason: str | None = None
    for stride_value in call_strides:
        stride = int(stride_value)
        if stride not in models:
            raise ValueError(f"missing model for stride {stride}")
        model, adapter = models[stride]
        primitive_batch, conservative_batch = _predict_model_batch_state(
            source,
            np.array([case_id], dtype=np.int64),
            current_primitive[None],
            absolute_start_frame + local_frame,
            stride,
            model,
            adapter,
            device,
            conservative=current_conservative[None],
        )
        proposed = primitive_batch[0]
        if not np.all(np.isfinite(proposed)):
            termination_reason = "nonfinite_raw_state"
            break
        if not _euler_state_is_admissible(proposed):
            termination_reason = "nonpositive_raw_state"
            break
        local_frame += stride
        current_primitive = proposed.copy()
        current_conservative = conservative_batch[0].copy()
        states[local_frame] = current_primitive.copy()
    return PathRollout(
        states=states,
        completed_frame=local_frame,
        requested_frame=requested_frame,
        termination_reason=termination_reason,
    )


def evaluate_perturbation_amplification(
    source: Euler1DNPZ,
    case_ids: np.ndarray,
    models: dict[int, tuple[torch.nn.Module, Any]],
    start_frames: Sequence[int],
    device: torch.device,
    *,
    relative_norm: float,
    seed: int,
) -> list[dict[str, Any]]:
    """Evaluate local map amplification for a frozen perturbation distribution."""

    rng = np.random.default_rng(seed)
    frames = sorted(set(int(value) for value in start_frames))
    perturbations: dict[tuple[int, int], tuple[np.ndarray, float]] = {}
    for start_frame in frames:
        for case_id_value in case_ids:
            case_id = int(case_id_value)
            perturbations[(start_frame, case_id)] = (
                fixed_norm_conservative_perturbation(
                    np.asarray(
                        source.data[case_id, start_frame],
                        dtype=np.float64,
                    ),
                    source.gamma,
                    relative_norm,
                    rng,
                )
            )
    rows: list[dict[str, Any]] = []
    for stride, (model, adapter) in sorted(models.items()):
        for start_frame in frames:
            target_frame = start_frame + stride
            if start_frame < 0 or target_frame >= source.num_frames:
                raise ValueError(
                    f"perturbation frame {start_frame} is invalid for stride {stride}"
                )
            for case_id_value in case_ids:
                case_id = int(case_id_value)
                base = np.asarray(
                    source.data[case_id, start_frame],
                    dtype=np.float64,
                )
                perturbed, realized_relative_norm = perturbations[
                    (start_frame, case_id)
                ]
                pair = np.stack((base, perturbed), axis=0)
                pair_case_ids = np.array([case_id, case_id], dtype=np.int64)
                outputs = _predict_model_batch(
                    source,
                    pair_case_ids,
                    pair,
                    start_frame,
                    stride,
                    model,
                    adapter,
                    device,
                )
                base_output, perturbed_output = outputs
                base_admissible = _euler_state_is_admissible(base_output)
                perturbed_admissible = _euler_state_is_admissible(perturbed_output)
                physical_dt = float(
                    source.t[case_id, target_frame] - source.t[case_id, start_frame]
                )
                if base_admissible and perturbed_admissible:
                    amplification = fixed_scale_perturbation_amplification(
                        base,
                        perturbed,
                        base_output,
                        perturbed_output,
                        source.gamma,
                        physical_dt,
                    )
                else:
                    amplification = {
                        "input_perturbation_l2": float("nan"),
                        "output_perturbation_l2": float("nan"),
                        "amplification": float("nan"),
                        "log_amplification_per_physical_time": float("nan"),
                    }
                rows.append(
                    {
                        "case_id": case_id,
                        "stride": stride,
                        "start_frame": start_frame,
                        "target_frame": target_frame,
                        "physical_dt": physical_dt,
                        "requested_relative_norm": relative_norm,
                        "realized_relative_norm": realized_relative_norm,
                        "direction_distribution": (
                            "iid_standard_normal_normalized_in_fixed_scale_"
                            "conservative_l2"
                        ),
                        "base_output_admissible": base_admissible,
                        "perturbed_output_admissible": perturbed_admissible,
                        "base_output_min_density": float(np.min(base_output[..., 0])),
                        "base_output_min_pressure": float(np.min(base_output[..., 2])),
                        "perturbed_output_min_density": float(
                            np.min(perturbed_output[..., 0])
                        ),
                        "perturbed_output_min_pressure": float(
                            np.min(perturbed_output[..., 2])
                        ),
                        **amplification,
                    }
                )
    return rows


def _summarize_amplification(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(
            (int(row["stride"]), int(row["start_frame"])),
            [],
        ).append(row)
    summary: list[dict[str, Any]] = []
    for (stride, start_frame), group in sorted(groups.items()):
        valid = [
            row
            for row in group
            if row["base_output_admissible"]
            and row["perturbed_output_admissible"]
            and np.isfinite(row["amplification"])
        ]
        factors = np.asarray(
            [row["amplification"] for row in valid],
            dtype=np.float64,
        )
        rates = np.asarray(
            [row["log_amplification_per_physical_time"] for row in valid],
            dtype=np.float64,
        )
        summary.append(
            {
                "stride": stride,
                "start_frame": start_frame,
                "target_frame": start_frame + stride,
                "num_cases": len(group),
                "admissible_pair_fraction": len(valid) / len(group),
                "amplification_median": (
                    float(np.median(factors)) if factors.size else float("nan")
                ),
                "amplification_p95": (
                    float(np.quantile(factors, 0.95)) if factors.size else float("nan")
                ),
                "amplification_max": (
                    float(np.max(factors)) if factors.size else float("nan")
                ),
                "log_amplification_per_physical_time_median": (
                    float(np.median(rates)) if rates.size else float("nan")
                ),
                "log_amplification_per_physical_time_p95": (
                    float(np.quantile(rates, 0.95)) if rates.size else float("nan")
                ),
                "log_amplification_per_physical_time_max": (
                    float(np.max(rates)) if rates.size else float("nan")
                ),
            }
        )
    return summary


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _measure_synchronized_calls(
    callback: Callable[[], Any],
    *,
    warmup: int,
    repeats: int,
    device: torch.device,
) -> dict[str, float]:
    for _ in range(warmup):
        callback()
    _synchronize(device)
    elapsed_ms = np.empty(repeats, dtype=np.float64)
    for repeat in range(repeats):
        _synchronize(device)
        started = time.perf_counter()
        callback()
        _synchronize(device)
        elapsed_ms[repeat] = (time.perf_counter() - started) * 1000.0
    return {
        "median_ms": float(np.median(elapsed_ms)),
        "mean_ms": float(np.mean(elapsed_ms)),
        "p10_ms": float(np.quantile(elapsed_ms, 0.10)),
        "p90_ms": float(np.quantile(elapsed_ms, 0.90)),
        "min_ms": float(np.min(elapsed_ms)),
    }


def benchmark_learned_maps(
    source: Euler1DNPZ,
    case_ids: np.ndarray,
    models: dict[int, tuple[torch.nn.Module, Any]],
    device: torch.device,
    *,
    warmup: int,
    repeats: int,
    throughput_batch_size: int,
) -> list[dict[str, Any]]:
    """Time device-resident and host-to-host learned-map calls separately."""

    batch_sizes = sorted(
        {
            1,
            min(int(throughput_batch_size), int(case_ids.size)),
        }
    )
    rows: list[dict[str, Any]] = []
    for stride, (model, adapter) in sorted(models.items()):
        for batch_size in batch_sizes:
            ids = np.asarray(case_ids[:batch_size], dtype=np.int64)
            host_states = np.ascontiguousarray(
                source.data[ids, 0],
                dtype=np.float32,
            )
            resident_batch = _make_model_batch(
                source,
                ids,
                host_states,
                0,
                stride,
                device,
            )

            def resident_call() -> torch.Tensor:
                with torch.inference_mode():
                    return adapter(model(resident_batch), resident_batch).primitive

            def end_to_end_call() -> np.ndarray:
                batch = _make_model_batch(
                    source,
                    ids,
                    host_states,
                    0,
                    stride,
                    device,
                )
                with torch.inference_mode():
                    prediction = adapter(model(batch), batch).primitive
                return prediction.detach().cpu().numpy()

            resident = _measure_synchronized_calls(
                resident_call,
                warmup=warmup,
                repeats=repeats,
                device=device,
            )
            end_to_end = _measure_synchronized_calls(
                end_to_end_call,
                warmup=warmup,
                repeats=repeats,
                device=device,
            )
            rows.append(
                {
                    "stride": stride,
                    "batch_size": batch_size,
                    "warmup": warmup,
                    "repeats": repeats,
                    **{
                        f"device_resident_{key}": value
                        for key, value in resident.items()
                    },
                    **{
                        f"host_to_host_{key}": value
                        for key, value in end_to_end.items()
                    },
                    "device_resident_samples_per_second": (
                        1000.0 * batch_size / resident["median_ms"]
                    ),
                    "host_to_host_samples_per_second": (
                        1000.0 * batch_size / end_to_end["median_ms"]
                    ),
                }
            )
    return rows


def learned_path_pareto_rows(
    summary_rows: list[dict[str, Any]],
    timing_rows: list[dict[str, Any]],
    saved_times: np.ndarray,
) -> list[dict[str, Any]]:
    """Attach synchronized batch-1 call costs to each evaluated rollout path."""

    batch_one = {
        int(row["stride"]): row for row in timing_rows if int(row["batch_size"]) == 1
    }
    result: list[dict[str, Any]] = []
    for summary in summary_rows:
        stride = int(summary["call_stride"])
        timing = batch_one.get(stride)
        if timing is None:
            continue
        calls = int(summary["num_calls"])
        horizon = int(summary["horizon"])
        result.append(
            {
                **summary,
                "physical_horizon": float(saved_times[horizon] - saved_times[0]),
                "estimated_device_resident_wall_ms": (
                    calls * float(timing["device_resident_median_ms"])
                ),
                "estimated_host_to_host_wall_ms": (
                    calls * float(timing["host_to_host_median_ms"])
                ),
                "wall_clock_accounting": (
                    "number_of_calls_times_synchronized_batch1_median"
                ),
            }
        )
    return result


def evaluate_local_semigroup(
    source: Euler1DNPZ,
    case_ids: np.ndarray,
    models: dict[int, tuple[torch.nn.Module, Any]],
    start_frames: Sequence[int],
    device: torch.device,
) -> list[dict[str, Any]]:
    """Compare one large call with exact smaller-stride compositions."""

    available_strides = sorted(models)
    rows: list[dict[str, Any]] = []
    for target_stride in [value for value in available_strides if value > 1]:
        definitions = build_composition_paths(
            target_stride,
            target_stride,
            available_strides,
        )
        direct_label = f"s{target_stride}_direct"
        if len(definitions) < 2:
            continue
        for start_frame in sorted(set(int(value) for value in start_frames)):
            target_frame = start_frame + target_stride
            if start_frame < 0 or target_frame >= source.num_frames:
                raise ValueError(
                    f"semigroup frame {start_frame} is invalid for stride "
                    f"{target_stride}"
                )
            for case_id_value in case_ids:
                case_id = int(case_id_value)
                initial = np.asarray(
                    source.data[case_id, start_frame],
                    dtype=np.float64,
                )

                rollouts = {
                    label: run_model_call_path(
                        source,
                        case_id,
                        initial,
                        calls,
                        models,
                        device,
                        absolute_start_frame=start_frame,
                    )
                    for label, calls in definitions.items()
                }
                direct = rollouts[direct_label]
                target_truth = np.asarray(
                    source.data[case_id, target_frame],
                    dtype=np.float64,
                )
                for composed_label, composed in rollouts.items():
                    if composed_label == direct_label:
                        continue
                    common_local_frame = last_common_frame(
                        direct,
                        composed,
                        target_stride,
                    )
                    common_absolute_frame = start_frame + common_local_frame
                    common_truth = np.asarray(
                        source.data[case_id, common_absolute_frame],
                        dtype=np.float64,
                    )
                    direct_state = direct.states[common_local_frame]
                    composed_state = composed.states[common_local_frame]
                    direct_common_error = _normalized_state_distance(
                        direct_state,
                        common_truth,
                        common_truth,
                        source.gamma,
                    )
                    composed_common_error = _normalized_state_distance(
                        composed_state,
                        common_truth,
                        common_truth,
                        source.gamma,
                    )
                    defect = _normalized_state_distance(
                        direct_state,
                        composed_state,
                        common_truth,
                        source.gamma,
                    )
                    both_completed = direct.completed and composed.completed
                    direct_target_error = (
                        _normalized_state_distance(
                            direct.states[target_stride],
                            target_truth,
                            target_truth,
                            source.gamma,
                        )
                        if direct.completed
                        else float("nan")
                    )
                    composed_target_error = (
                        _normalized_state_distance(
                            composed.states[target_stride],
                            target_truth,
                            target_truth,
                            source.gamma,
                        )
                        if composed.completed
                        else float("nan")
                    )
                    rows.append(
                        {
                            "case_id": case_id,
                            "target_stride": target_stride,
                            "start_frame": start_frame,
                            "target_frame": target_frame,
                            "direct_path": direct_label,
                            "composed_path": composed_label,
                            "direct_calls": len(definitions[direct_label]),
                            "composed_calls": len(definitions[composed_label]),
                            "both_completed_target": both_completed,
                            "direct_termination_reason": (direct.termination_reason),
                            "composed_termination_reason": (
                                composed.termination_reason
                            ),
                            "last_common_frame": common_absolute_frame,
                            "direct_truth_error_at_common": direct_common_error,
                            "composed_truth_error_at_common": (composed_common_error),
                            "semigroup_defect_at_common": defect,
                            "defect_bounds_hold": semigroup_error_bounds_hold(
                                defect,
                                direct_common_error,
                                composed_common_error,
                                tolerance=1.0e-9,
                            ),
                            "direct_truth_error_at_target": (direct_target_error),
                            "composed_truth_error_at_target": (composed_target_error),
                            "direct_to_composed_truth_error_ratio_at_target": (
                                direct_target_error / max(composed_target_error, EPS)
                                if both_completed
                                else float("nan")
                            ),
                        }
                    )
    return rows


def evaluate_frontier(
    source: Euler1DNPZ,
    case_ids: np.ndarray,
    models: dict[int, tuple[torch.nn.Module, torch.nn.Module]],
    horizons: Sequence[int],
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Evaluate all direct/composed paths against serialized truth."""

    available_strides = sorted(models)
    path_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    for horizon in sorted(set(int(value) for value in horizons)):
        if horizon < 1 or horizon >= source.num_frames:
            raise ValueError(f"invalid evaluation horizon: {horizon}")
        target_strides = [
            stride
            for stride in available_strides
            if stride > 1 and horizon % stride == 0
        ]
        for target_stride in target_strides:
            definitions = build_composition_paths(
                target_stride,
                horizon,
                available_strides,
            )
            direct_label = f"s{target_stride}_direct"
            if len(definitions) < 2:
                continue
            for case_id_value in case_ids:
                case_id = int(case_id_value)
                initial = np.asarray(source.data[case_id, 0], dtype=np.float64)
                x = np.asarray(source.x[case_id], dtype=np.float64)

                rollouts = {
                    label: run_model_call_path(
                        source,
                        case_id,
                        initial,
                        calls,
                        models,
                        device,
                    )
                    for label, calls in definitions.items()
                }
                truth = np.asarray(source.data[case_id, horizon], dtype=np.float64)
                for label, rollout in rollouts.items():
                    row: dict[str, Any] = {
                        "case_id": case_id,
                        "target_stride": target_stride,
                        "call_stride": int(definitions[label][0]),
                        "path": label,
                        "horizon": horizon,
                        "num_calls": len(definitions[label]),
                        "completed_horizon": rollout.completed,
                        "completed_frame": rollout.completed_frame,
                        "termination_reason": rollout.termination_reason,
                    }
                    if rollout.completed:
                        row.update(
                            _state_metrics(
                                rollout.states[horizon],
                                truth,
                                initial,
                                x,
                                source.gamma,
                            )
                        )
                    else:
                        row.update(
                            {
                                key: float("nan")
                                for key in (
                                    "primitive_relative_l2",
                                    "fixed_scale_conservative_relative_l2",
                                    "smooth_region_relative_l2",
                                    "boundary_relative_l2",
                                    "shock_top2_position_mae",
                                    "shock_top2_strength_relative_l1",
                                    "global_conserved_total_mismatch_relative_l2",
                                    "conservative_budget_relative_l2",
                                    "min_density",
                                    "min_pressure",
                                )
                            }
                        )
                    path_rows.append(row)

                direct = rollouts[direct_label]
                for composed_label, composed in rollouts.items():
                    if composed_label == direct_label:
                        continue
                    common_frame = last_common_frame(direct, composed, horizon)
                    common_truth = np.asarray(
                        source.data[case_id, common_frame], dtype=np.float64
                    )
                    direct_state = direct.states[common_frame]
                    composed_state = composed.states[common_frame]
                    direct_error = _normalized_state_distance(
                        direct_state,
                        common_truth,
                        common_truth,
                        source.gamma,
                    )
                    composed_error = _normalized_state_distance(
                        composed_state,
                        common_truth,
                        common_truth,
                        source.gamma,
                    )
                    defect = _normalized_state_distance(
                        direct_state,
                        composed_state,
                        common_truth,
                        source.gamma,
                    )
                    comparison_rows.append(
                        {
                            "case_id": case_id,
                            "target_stride": target_stride,
                            "direct_path": direct_label,
                            "composed_path": composed_label,
                            "requested_horizon": horizon,
                            "both_completed_requested_horizon": (
                                direct.completed and composed.completed
                            ),
                            "last_common_frame": common_frame,
                            "direct_truth_error_at_common": direct_error,
                            "composed_truth_error_at_common": composed_error,
                            "semigroup_defect_at_common": defect,
                            "defect_bounds_hold": semigroup_error_bounds_hold(
                                defect,
                                direct_error,
                                composed_error,
                                tolerance=1.0e-9,
                            ),
                        }
                    )
    return path_rows, comparison_rows


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.torch_threads < 1:
        raise ValueError("--torch-threads must be positive")
    if args.boundary_margin_cells < 1:
        raise ValueError("--boundary-margin-cells must be positive")
    if not 0.0 < args.min_front_peak_fraction <= 1.0:
        raise ValueError("--min-front-peak-fraction must be in (0, 1]")
    if args.perturbation_cases < 0:
        raise ValueError("--perturbation-cases must be nonnegative")
    if args.perturbation_cases and args.perturbation_relative_norm <= 0.0:
        raise ValueError("--perturbation-relative-norm must be positive")
    if args.timing_warmup < 0 or args.timing_repeats < 0:
        raise ValueError("timing warmup and repeats must be nonnegative")
    if args.throughput_batch_size < 1:
        raise ValueError("--throughput-batch-size must be positive")
    if args.direct_curve_max_horizon < 0:
        raise ValueError("--direct-curve-max-horizon must be nonnegative")
    if not args.horizons or any(value < 1 for value in args.horizons):
        raise ValueError("--horizons must contain positive saved-frame indices")
    if (
        args.direct_curve_max_horizon
        and max(args.horizons) > args.direct_curve_max_horizon
    ):
        raise ValueError(
            "--direct-curve-max-horizon must cover every declared summary horizon"
        )
    if args.direct_curves_only and not args.direct_curve_max_horizon:
        raise ValueError("--direct-curves-only requires a positive curve horizon")
    if not args.checkpoint_set_label.strip():
        raise ValueError("--checkpoint-set-label must be nonempty")
    if not args.error_budgets or any(value <= 0.0 for value in args.error_budgets):
        raise ValueError("--error-budgets must contain positive values")
    if args.evaluation_regime == "ood":
        if args.training_data_path is None or not args.ood_label:
            raise ValueError(
                "OOD evaluation requires --training-data-path and --ood-label"
            )
    elif args.training_data_path is not None or args.ood_label is not None:
        raise ValueError("training data and OOD label are only valid in OOD mode")
    torch.set_num_threads(args.torch_threads)
    device = _select_device(args.device)
    source = load_euler1d_npz(args.data_path)
    source.validate()
    if not np.array_equal(
        source.t,
        np.broadcast_to(source.t[:1], source.t.shape),
    ):
        raise ValueError("frontier evaluation requires common saved times")

    training_source: Euler1DNPZ | None = None
    ood_contract: dict[str, Any] | None = None
    ood_regime_labels: tuple[str, ...] | None = None
    ood_generator_sha256: str | None = None
    ood_regime_counts: dict[str, int] | None = None
    if args.evaluation_regime == "matched_id":
        expected_cases = _frozen_split(
            source,
            split_seed=args.split_seed,
            train_cases=args.train_cases,
            val_cases=args.val_cases,
            test_cases=args.test_cases,
            split=args.split,
        )
    else:
        assert args.training_data_path is not None
        training_source = load_euler1d_npz(args.training_data_path)
        training_source.validate()
        if not np.array_equal(
            training_source.t,
            np.broadcast_to(training_source.t[:1], training_source.t.shape),
        ):
            raise ValueError("training data must use common saved times")
        _validate_ood_compatibility(training_source, source)
        assert args.ood_label is not None
        (
            ood_contract,
            ood_regime_labels,
            ood_generator_sha256,
            ood_regime_counts,
        ) = _validated_ood_metadata(source, args.ood_label)
        expected_cases = np.arange(source.num_cases, dtype=np.int64)
    audit_horizons = sorted(
        set(args.horizons)
        | ({args.direct_curve_max_horizon} if args.direct_curve_max_horizon else set())
    )
    wave_audit = truth_wave_audit(
        source,
        expected_cases,
        audit_horizons,
        boundary_margin_cells=args.boundary_margin_cells,
        min_peak_fraction=args.min_front_peak_fraction,
    )
    contract: dict[str, Any] = {
        "data_path": str(args.data_path),
        "data_sha256": sha256_file(args.data_path),
        "saved_time_sha256": _saved_time_sha256(source),
        "data_shape": list(source.data.shape),
        "gamma": source.gamma,
        "saved_times": source.t[0].tolist(),
        "saved_times_common_across_cases": True,
        "evaluation_regime": args.evaluation_regime,
        "split": (args.split if args.evaluation_regime == "matched_id" else "all_ood"),
        "ood_label": args.ood_label,
        "case_ids": expected_cases.tolist(),
        "horizons": sorted(set(args.horizons)),
        "checkpoint_set_label": args.checkpoint_set_label,
        "device": str(device),
        "evaluator_source_sha256": sha256_file(Path(__file__).resolve()),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "metric_semantics": {
            "global_conserved_total_mismatch_relative_l2": (
                "predicted_vs_truth_domain_integrated_conserved_total_mismatch"
            ),
            "conservative_budget_relative_l2": (
                "legacy_alias_not_a_boundary_flux_conservation_closure_residual"
            ),
        },
    }
    if training_source is not None:
        assert args.training_data_path is not None
        contract["checkpoint_training_data_path"] = str(args.training_data_path)
        contract["checkpoint_training_data_sha256"] = sha256_file(
            args.training_data_path
        )
        contract["checkpoint_training_saved_time_sha256"] = _saved_time_sha256(
            training_source
        )
        contract["ood_contract"] = ood_contract
        contract["ood_generator_source_sha256"] = ood_generator_sha256
        contract["ood_regime_counts"] = ood_regime_counts
    if args.wave_audit_only and args.checkpoint:
        raise ValueError("--wave-audit-only does not accept checkpoints")
    if not args.wave_audit_only and not args.checkpoint:
        raise ValueError("at least one --checkpoint is required for model evaluation")
    if not args.wave_audit_only and args.perturbation_cases > expected_cases.size:
        raise ValueError("--perturbation-cases exceeds the frozen split size")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "wave_occupancy.json").write_text(
        json.dumps(json_ready(wave_audit), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if args.wave_audit_only:
        (args.output_dir / "contract.json").write_text(
            json.dumps(json_ready(contract), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(json.dumps(json_ready(wave_audit), indent=2, sort_keys=True))
        return

    checkpoint_paths = _checkpoint_paths(args.checkpoint)
    models: dict[int, tuple[torch.nn.Module, torch.nn.Module]] = {}
    checkpoint_contract: dict[str, Any] = {}
    for stride, path in checkpoint_paths.items():
        model, adapter, _normalizer, checkpoint = load_frozen_residual_checkpoint(
            path,
            device,
        )
        actual_stride = int(checkpoint["args"]["step_stride"])
        if actual_stride != stride:
            raise ValueError(
                f"checkpoint {path} has stride {actual_stride}, expected {stride}"
            )
        frozen_cases = np.asarray(
            checkpoint["val_cases" if args.split == "validation" else "test_cases"],
            dtype=np.int64,
        )
        if args.evaluation_regime == "matched_id" and not np.array_equal(
            frozen_cases, expected_cases
        ):
            raise ValueError(f"checkpoint stride {stride} uses a different split")
        recorded_hash = checkpoint.get("data_sha256")
        expected_training_hash = (
            contract["data_sha256"]
            if args.evaluation_regime == "matched_id"
            else contract["checkpoint_training_data_sha256"]
        )
        if recorded_hash != expected_training_hash:
            raise ValueError(
                f"checkpoint stride {stride} has missing or mismatched training data"
            )
        recorded_time_hash = checkpoint.get("saved_time_sha256")
        expected_training_time_hash = (
            contract["saved_time_sha256"]
            if args.evaluation_regime == "matched_id"
            else contract["checkpoint_training_saved_time_sha256"]
        )
        if recorded_time_hash != expected_training_time_hash:
            raise ValueError(f"checkpoint stride {stride} uses different saved times")
        models[stride] = (model, adapter)
        checkpoint_contract[str(stride)] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "best_epoch": checkpoint.get("best_epoch"),
            "checkpoint_epoch": _checkpoint_epoch(checkpoint),
            "step_stride": actual_stride,
            "data_sha256": recorded_hash,
            "saved_time_sha256": recorded_time_hash,
        }
    contract["checkpoints"] = checkpoint_contract
    contract["direct_curve_audit"] = {
        "enabled": bool(args.direct_curve_max_horizon),
        "max_horizon": args.direct_curve_max_horizon,
        "summary_horizons": sorted(set(args.horizons)),
        "error_budgets": sorted(set(args.error_budgets)),
        "error_metric": "fixed_scale_conservative_relative_l2",
        "survival_observation": "first_observed_call_endpoint",
        "direct_curves_only": args.direct_curves_only,
        "trajectory_increment": (
            "on_policy_predicted_increment_vs_stored_truth_trajectory_increment"
        ),
        "checkpoint_role": (
            "immutable_primary_selection"
            if args.checkpoint_set_label == "primary_selected"
            else "validation_only_checkpoint_candidate"
        ),
    }
    contract["local_semigroup_diagnostic"] = {
        "start_frames": sorted(set(args.semigroup_start_frames)),
        "comparison": (
            "one_direct_large_call_versus_exact_smaller_stride_compositions"
        ),
        "truth_errors_reported_beside_every_defect": True,
    }
    contract["perturbation_diagnostic"] = {
        "case_ids": expected_cases[: args.perturbation_cases].tolist(),
        "relative_norm": args.perturbation_relative_norm,
        "seed": args.perturbation_seed,
        "start_frames": sorted(set(args.perturbation_start_frames)),
        "norm": "fixed_physical_scale_conservative_l2",
        "direction_distribution": (
            "iid_standard_normal_normalized_to_exact_relative_norm"
        ),
        "directions_shared_across_strides": True,
        "metadata_and_boundary_states_held_fixed": True,
    }
    contract["timing_protocol"] = {
        "warmup": args.timing_warmup,
        "repeats": args.timing_repeats,
        "throughput_batch_size": args.throughput_batch_size,
        "synchronization": (
            "torch.cuda.synchronize_before_and_after_each_call"
            if device.type == "cuda"
            else "not_applicable_cpu"
        ),
        "device_resident_boundary": (
            "prebuilt_device_batch_to_device_prediction_no_host_transfer"
        ),
        "host_to_host_boundary": (
            "numpy_primitive_and_metadata_through_tensor_construction_h2d_"
            "model_adapter_and_prediction_d2h"
        ),
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
        ),
    }

    started = time.perf_counter()
    direct_curve_rows = (
        evaluate_direct_curves(
            source,
            expected_cases,
            models,
            args.direct_curve_max_horizon,
            device,
            checkpoint_set_label=args.checkpoint_set_label,
        )
        if args.direct_curve_max_horizon
        else []
    )
    _attach_ood_regimes(direct_curve_rows, ood_regime_labels)
    direct_summary_rows = (
        summarize_direct_curves(
            direct_curve_rows,
            args.horizons,
            num_cases=expected_cases.size,
        )
        if direct_curve_rows
        else []
    )
    survival_rows = (
        error_budget_survival_rows(
            direct_curve_rows,
            args.error_budgets,
            max_horizon=args.direct_curve_max_horizon,
        )
        if direct_curve_rows
        else []
    )
    if args.direct_curves_only:
        path_rows: list[dict[str, Any]] = []
        comparison_rows: list[dict[str, Any]] = []
        summary_rows: list[dict[str, Any]] = []
        local_semigroup_rows: list[dict[str, Any]] = []
        perturbation_rows: list[dict[str, Any]] = []
    else:
        frontier_started = time.perf_counter()
        path_rows, comparison_rows = evaluate_frontier(
            source,
            expected_cases,
            models,
            args.horizons,
            device,
        )
        _attach_ood_regimes(path_rows, ood_regime_labels)
        _attach_ood_regimes(comparison_rows, ood_regime_labels)
        contract["frontier_rollout_runtime_seconds"] = (
            time.perf_counter() - frontier_started
        )
        summary_rows = _summarize_paths(path_rows)
        local_semigroup_rows = evaluate_local_semigroup(
            source,
            expected_cases,
            models,
            args.semigroup_start_frames,
            device,
        )
        _attach_ood_regimes(local_semigroup_rows, ood_regime_labels)
        perturbation_rows = (
            evaluate_perturbation_amplification(
                source,
                expected_cases[: args.perturbation_cases],
                models,
                args.perturbation_start_frames,
                device,
                relative_norm=args.perturbation_relative_norm,
                seed=args.perturbation_seed,
            )
            if args.perturbation_cases
            else []
        )
        _attach_ood_regimes(perturbation_rows, ood_regime_labels)
    perturbation_summary = _summarize_amplification(perturbation_rows)
    ood_path_summary_rows = (
        _summarize_paths_by_ood_regime(path_rows)
        if ood_regime_labels is not None and path_rows
        else []
    )
    ood_direct_summary_rows = (
        _summarize_direct_curves_by_ood_regime(direct_curve_rows, args.horizons)
        if ood_regime_labels is not None and direct_curve_rows
        else []
    )
    timing_rows = (
        benchmark_learned_maps(
            source,
            expected_cases,
            models,
            device,
            warmup=args.timing_warmup,
            repeats=args.timing_repeats,
            throughput_batch_size=args.throughput_batch_size,
        )
        if args.timing_repeats and not args.direct_curves_only
        else []
    )
    pareto_rows = learned_path_pareto_rows(
        summary_rows,
        timing_rows,
        source.t[0],
    )
    contract["total_evaluation_runtime_seconds"] = time.perf_counter() - started
    _write_jsonl(args.output_dir / "paths.jsonl", path_rows)
    _write_jsonl(args.output_dir / "direct_curves.jsonl", direct_curve_rows)
    _write_csv(args.output_dir / "direct_summary.csv", direct_summary_rows)
    _write_csv(args.output_dir / "error_budget_survival.csv", survival_rows)
    _write_csv(args.output_dir / "comparisons.csv", comparison_rows)
    _write_csv(
        args.output_dir / "local_semigroup.csv",
        local_semigroup_rows,
    )
    _write_csv(args.output_dir / "summary.csv", summary_rows)
    _write_csv(
        args.output_dir / "summary_by_ood_regime.csv",
        ood_path_summary_rows,
    )
    _write_csv(
        args.output_dir / "direct_summary_by_ood_regime.csv",
        ood_direct_summary_rows,
    )
    _write_csv(
        args.output_dir / "perturbation_amplification.csv",
        perturbation_rows,
    )
    _write_csv(
        args.output_dir / "perturbation_summary.csv",
        perturbation_summary,
    )
    _write_csv(args.output_dir / "learned_timing.csv", timing_rows)
    _write_csv(args.output_dir / "learned_pareto.csv", pareto_rows)
    (args.output_dir / "contract.json").write_text(
        json.dumps(json_ready(contract), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(
            json_ready(
                {
                    "contract": contract,
                    "wave_occupancy": wave_audit,
                    "summary": summary_rows,
                    "direct_summary": direct_summary_rows,
                    "summary_by_ood_regime": ood_path_summary_rows,
                    "direct_summary_by_ood_regime": ood_direct_summary_rows,
                    "error_budget_survival": survival_rows,
                    "perturbation_summary": perturbation_summary,
                    "learned_timing": timing_rows,
                    "learned_pareto": pareto_rows,
                    "comparison_count": len(comparison_rows),
                    "local_semigroup_comparison_count": len(local_semigroup_rows),
                }
            ),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    printed_summary = direct_summary_rows if args.direct_curves_only else summary_rows
    print(json.dumps(json_ready(printed_summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
