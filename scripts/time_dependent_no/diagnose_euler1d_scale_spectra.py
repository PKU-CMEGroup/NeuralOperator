"""Frozen-checkpoint scale and failure-precursor diagnostics for 1D Euler.

The script evaluates each checkpoint at its native saved-frame stride. It keeps
teacher-forced update spectra separate from autoregressive state-error spectra:
solver fluxes and trajectory increments are valid labels only on truth states.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence, cast

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.time_dependent_no.train_euler1d_target_ladder import (  # noqa: E402
    PrimitiveNormalizer,
    build_model,
    json_ready,
    pressure_front_top2_metrics_np,
)
from utility.time_dependent_no.euler1d import (  # noqa: E402
    conservative_to_primitive,
    make_euler1d_batch,
    primitive_to_conservative,
)
from utility.time_dependent_no.euler1d_data import (  # noqa: E402
    Euler1DNPZ,
    load_euler1d_npz,
)
from utility.time_dependent_no.euler1d_models import Euler1DTarget  # noqa: E402
from utility.time_dependent_no.euler1d_targets import make_target_adapter  # noqa: E402


EPS = 1.0e-12
SPECTRAL_BANDS = {
    "low_1_4": (1, 5),
    "mid_5_16": (5, 17),
    "resolved_17_24": (17, 25),
    "high_25_64": (25, 65),
    "tail_65_nyquist": (65, None),
}


@dataclass
class FrozenModel:
    name: str
    path: Path
    model: nn.Module
    adapter: nn.Module
    args: argparse.Namespace
    test_cases: np.ndarray
    parameter_count: int
    checkpoint_sha256: str

    @property
    def stride(self) -> int:
        return int(self.args.step_stride)


def parse_assignment(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected NAME=VALUE")
    name, assigned = value.split("=", 1)
    name = name.strip()
    assigned = assigned.strip()
    if not name or not assigned:
        raise argparse.ArgumentTypeError("expected nonempty NAME=VALUE")
    return name, assigned


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        action="append",
        type=parse_assignment,
        required=True,
        metavar="NAME=PATH",
    )
    parser.add_argument(
        "--rollout-calls",
        action="append",
        type=parse_assignment,
        default=[],
        metavar="NAME=CALLS",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--teacher-frame-step", type=int, default=5)
    parser.add_argument(
        "--rollout-batch-size",
        type=int,
        default=1,
        help="Use one for bitwise agreement with the canonical per-case evaluator.",
    )
    parser.add_argument("--shock-radius-cells", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args(argv)
    names = [name for name, _ in args.checkpoint]
    if len(set(names)) != len(names):
        parser.error("checkpoint names must be unique")
    call_names = [name for name, _ in args.rollout_calls]
    if len(set(call_names)) != len(call_names):
        parser.error("rollout-call names must be unique")
    if not set(call_names).issubset(names):
        parser.error("--rollout-calls contains an unknown checkpoint name")
    if args.teacher_frame_step < 1:
        parser.error("--teacher-frame-step must be positive")
    if args.rollout_batch_size < 1:
        parser.error("--rollout-batch-size must be positive")
    if args.shock_radius_cells < 0:
        parser.error("--shock-radius-cells must be nonnegative")
    return args


def select_device(name: str, gpu: int) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda" or (name == "auto" and torch.cuda.is_available()):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.set_device(gpu)
        return torch.device("cuda", gpu)
    return torch.device("cpu")


def checkpoint_metadata(
    checkpoint: dict[str, Any],
    args: argparse.Namespace,
    key: str,
    default: str,
) -> str:
    value = checkpoint.get(key)
    if value is None:
        value = getattr(args, key, default)
    return str(value)


def load_frozen_model(
    name: str,
    path: Path,
    device: torch.device,
) -> FrozenModel:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    args = argparse.Namespace(**checkpoint["args"])
    target = cast(Euler1DTarget, str(checkpoint["target"]))
    normalizer = PrimitiveNormalizer(
        mean=checkpoint["input_normalizer_mean"],
        std=checkpoint["input_normalizer_std"],
        coordinates=checkpoint_metadata(
            checkpoint,
            args,
            "input_coordinates",
            "primitive",
        ),
        normalization=checkpoint_metadata(
            checkpoint,
            args,
            "input_normalization",
            "empirical",
        ),
    ).to(device)
    model = build_model(
        str(checkpoint["model"]),
        target,
        args,
        normalizer,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    adapter = make_target_adapter(
        target,
        positive_transform=str(getattr(args, "positive_transform", "none")),
        flux_correction_scale=float(getattr(args, "flux_correction_scale", 1.0)),
        flux_correction_scale_floor=float(
            getattr(args, "flux_correction_scale_floor", 1.0e-6)
        ),
        flux_gauge_mode=str(getattr(args, "flux_gauge_mode", "raw")),
        interface_flux_mode=str(
            getattr(args, "interface_flux_mode", "rusanov")
        ),
    ).to(device)
    return FrozenModel(
        name=name,
        path=path,
        model=model,
        adapter=adapter,
        args=args,
        test_cases=np.asarray(checkpoint["test_cases"], dtype=np.int64),
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def decode_batch(
    frozen: FrozenModel,
    primitive: np.ndarray,
    conservative: np.ndarray,
    x: np.ndarray,
    dt: np.ndarray,
    left_state: np.ndarray,
    right_state: np.ndarray,
    gamma: float,
    device: torch.device,
) -> dict[str, np.ndarray | None]:
    primitive_t = torch.as_tensor(primitive, dtype=torch.float32, device=device)
    conservative_t = torch.as_tensor(
        conservative,
        dtype=torch.float32,
        device=device,
    )
    recurrent_coordinates = str(
        getattr(frozen.args, "recurrent_coordinates", "primitive")
    )
    if recurrent_coordinates == "conservative":
        primitive_t = conservative_to_primitive(conservative_t, gamma=gamma)
    batch = make_euler1d_batch(
        primitive_t,
        torch.as_tensor(x, dtype=torch.float32, device=device),
        torch.as_tensor(dt, dtype=torch.float32, device=device),
        current_conservative_state=conservative_t,
        gamma=gamma,
        left_boundary_primitive=torch.as_tensor(
            left_state,
            dtype=torch.float32,
            device=device,
        ),
        right_initial_primitive=torch.as_tensor(
            right_state,
            dtype=torch.float32,
            device=device,
        ),
    )
    with torch.no_grad():
        raw = frozen.model(batch)
        decoded = frozen.adapter(raw, batch)
    face_flux = decoded.aux.get("face_flux")
    return {
        "primitive": decoded.primitive.detach().cpu().numpy().astype(np.float64),
        "conservative": decoded.conservative.detach()
        .cpu()
        .numpy()
        .astype(np.float64),
        "face_flux": (
            face_flux.detach().cpu().numpy().astype(np.float64)
            if isinstance(face_flux, torch.Tensor)
            else None
        ),
    }


def primitive_to_conservative_np(
    primitive: np.ndarray,
    gamma: float,
) -> np.ndarray:
    tensor = torch.as_tensor(np.ascontiguousarray(primitive))
    if tensor.dtype not in (torch.float32, torch.float64):
        tensor = tensor.to(torch.float64)
    return primitive_to_conservative(tensor, gamma=gamma).numpy()


def conservative_scale(gamma: float) -> np.ndarray:
    return np.array((1.0, 1.0, 1.0 / (gamma - 1.0)), dtype=np.float64)


def _band_slice(num_modes: int, bounds: tuple[int, int | None]) -> slice:
    start, stop = bounds
    return slice(min(start, num_modes), min(stop or num_modes, num_modes))


def spectral_metrics(
    field: np.ndarray,
    *,
    prefix: str,
) -> dict[str, np.ndarray]:
    """Return Hann-windowed, mean-free index-wavenumber diagnostics."""

    values = np.asarray(field, dtype=np.float64)
    if values.ndim != 3 or values.shape[-1] != 3:
        raise ValueError("spectral field must have shape [batch, cells, 3]")
    num_cells = values.shape[1]
    window = np.hanning(num_cells).reshape(1, num_cells, 1)
    centered = values - np.nanmean(values, axis=1, keepdims=True)
    transformed = np.fft.rfft(centered * window, axis=1, norm="ortho")
    energy = np.sum(np.abs(transformed) ** 2, axis=-1)
    finite_rows = np.isfinite(values).all(axis=(1, 2))
    energy[~finite_rows] = np.nan
    nonzero = energy[:, 1:]
    total = np.sum(nonzero, axis=1)
    normalization = max(float(np.sum(window**2)) * values.shape[-1], EPS)
    result: dict[str, np.ndarray] = {
        f"{prefix}_spectral_rms": np.sqrt(total / normalization),
    }
    mode_ids = np.arange(energy.shape[1], dtype=np.float64)
    result[f"{prefix}_spectral_centroid_fraction"] = np.sum(
        energy * mode_ids.reshape(1, -1),
        axis=1,
    ) / np.maximum(total * max(energy.shape[1] - 1, 1), EPS)
    for band, bounds in SPECTRAL_BANDS.items():
        band_energy = np.sum(energy[:, _band_slice(energy.shape[1], bounds)], axis=1)
        result[f"{prefix}_{band}_fraction"] = band_energy / np.maximum(total, EPS)
        result[f"{prefix}_{band}_rms"] = np.sqrt(band_energy / normalization)
    return result


def spectral_pair_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    prefix: str,
) -> dict[str, np.ndarray]:
    """Return bandwise projection gain and error for two update fields."""

    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    num_cells = prediction.shape[1]
    window = np.hanning(num_cells).reshape(1, num_cells, 1)

    def transform(values: np.ndarray) -> np.ndarray:
        centered = values - np.mean(values, axis=1, keepdims=True)
        return np.fft.rfft(centered * window, axis=1, norm="ortho")

    pred_hat = transform(prediction)
    target_hat = transform(target)
    error_hat = pred_hat - target_hat
    result: dict[str, np.ndarray] = {}
    for band, bounds in SPECTRAL_BANDS.items():
        band_slice = _band_slice(pred_hat.shape[1], bounds)
        pred_band = pred_hat[:, band_slice]
        target_band = target_hat[:, band_slice]
        error_band = error_hat[:, band_slice]
        target_energy = np.sum(np.abs(target_band) ** 2, axis=(1, 2))
        pred_energy = np.sum(np.abs(pred_band) ** 2, axis=(1, 2))
        error_energy = np.sum(np.abs(error_band) ** 2, axis=(1, 2))
        cross = np.real(
            np.sum(pred_band * np.conj(target_band), axis=(1, 2))
        )
        result[f"{prefix}_{band}_projection_gain"] = cross / np.maximum(
            target_energy,
            EPS,
        )
        result[f"{prefix}_{band}_amplitude_ratio"] = np.sqrt(
            pred_energy / np.maximum(target_energy, EPS)
        )
        result[f"{prefix}_{band}_relative_error"] = np.sqrt(
            error_energy / np.maximum(target_energy, EPS)
        )
        result[f"{prefix}_{band}_cosine"] = cross / np.maximum(
            np.sqrt(pred_energy * target_energy),
            EPS,
        )
    return result


def shock_masks(
    truth_primitive: np.ndarray,
    *,
    radius_cells: int,
    min_separation_cells: int = 4,
) -> np.ndarray:
    masks = np.zeros(truth_primitive.shape[:2], dtype=bool)
    for sample_id, sample in enumerate(truth_primitive):
        pressure_jump = np.abs(np.diff(sample[:, 2]))
        selected: list[int] = []
        for face_id in np.argsort(-pressure_jump, kind="stable"):
            if all(
                abs(int(face_id) - previous) > min_separation_cells
                for previous in selected
            ):
                selected.append(int(face_id))
                if len(selected) == 2:
                    break
        for face_id in selected:
            lo = max(0, face_id - radius_cells + 1)
            hi = min(sample.shape[0], face_id + radius_cells + 1)
            masks[sample_id, lo:hi] = True
    return masks


def masked_rms(values: np.ndarray, masks: np.ndarray) -> np.ndarray:
    result = np.full(values.shape[0], np.nan, dtype=np.float64)
    for sample_id, mask in enumerate(masks):
        selected = values[sample_id, mask]
        if selected.size and np.isfinite(selected).all():
            result[sample_id] = float(np.sqrt(np.mean(selected**2)))
    return result


def characteristic_metrics(
    conservative_error: np.ndarray,
    base_primitive: np.ndarray,
    gamma: float,
) -> dict[str, np.ndarray]:
    d_rho, d_momentum, d_energy = np.moveaxis(conservative_error, -1, 0)
    rho, velocity, pressure = np.moveaxis(base_primitive, -1, 0)
    sound_speed = np.sqrt(gamma * pressure / rho)
    d_velocity = (d_momentum - velocity * d_rho) / rho
    d_pressure = (gamma - 1.0) * (
        d_energy - velocity * d_momentum + 0.5 * velocity**2 * d_rho
    )
    characteristic = {
        "char_minus_rms": (d_pressure - rho * sound_speed * d_velocity)
        / (2.0 * sound_speed**2),
        "char_contact_rms": d_rho - d_pressure / sound_speed**2,
        "char_plus_rms": (d_pressure + rho * sound_speed * d_velocity)
        / (2.0 * sound_speed**2),
    }
    return {
        name: np.sqrt(np.mean(value**2, axis=1))
        for name, value in characteristic.items()
    }


def difference_metrics(
    error: np.ndarray,
    prediction: np.ndarray,
    target: np.ndarray,
    dx: np.ndarray,
) -> dict[str, np.ndarray]:
    dx1 = dx.reshape(-1, 1, 1)
    first = np.diff(error, axis=1) / dx1
    second = np.diff(error, n=2, axis=1) / dx1**2
    pred_tv = np.sum(np.abs(np.diff(prediction, axis=1)), axis=(1, 2))
    truth_tv = np.sum(np.abs(np.diff(target, axis=1)), axis=(1, 2))
    return {
        "error_d1_rms": np.sqrt(np.mean(first**2, axis=(1, 2))),
        "error_d2_rms": np.sqrt(np.mean(second**2, axis=(1, 2))),
        "state_tv_ratio": pred_tv / np.maximum(truth_tv, EPS),
    }


def state_metrics(
    prediction_primitive: np.ndarray,
    prediction_conservative: np.ndarray,
    truth_primitive: np.ndarray,
    truth_conservative: np.ndarray,
    x: np.ndarray,
    gamma: float,
    *,
    shock_radius_cells: int,
) -> dict[str, np.ndarray]:
    scale = conservative_scale(gamma).reshape(1, 1, 3)
    error = (prediction_conservative - truth_conservative) / scale
    target_scaled = truth_conservative / scale
    flat_error = error.reshape(error.shape[0], -1)
    flat_target = target_scaled.reshape(target_scaled.shape[0], -1)
    masks = shock_masks(truth_primitive, radius_cells=shock_radius_cells)
    dx = np.mean(np.diff(x, axis=1), axis=1)
    metrics: dict[str, np.ndarray] = {
        "state_cons_scaled_rmse": np.sqrt(np.mean(error**2, axis=(1, 2))),
        "state_cons_scaled_rel_l2": np.linalg.norm(flat_error, axis=1)
        / np.maximum(np.linalg.norm(flat_target, axis=1), EPS),
        "state_primitive_rel_l2": np.linalg.norm(
            (prediction_primitive - truth_primitive).reshape(error.shape[0], -1),
            axis=1,
        )
        / np.maximum(
            np.linalg.norm(truth_primitive.reshape(error.shape[0], -1), axis=1),
            EPS,
        ),
        "shock_cons_scaled_rmse": masked_rms(error, masks),
        "smooth_cons_scaled_rmse": masked_rms(error, ~masks),
        "min_density": np.nanmin(prediction_primitive[..., 0], axis=1),
        "min_pressure": np.nanmin(prediction_primitive[..., 2], axis=1),
        "max_abs_primitive": np.nanmax(np.abs(prediction_primitive), axis=(1, 2)),
    }
    metrics.update(spectral_metrics(error, prefix="error"))
    metrics.update(
        difference_metrics(
            error,
            prediction_conservative / scale,
            target_scaled,
            dx,
        )
    )
    metrics.update(characteristic_metrics(prediction_conservative - truth_conservative, truth_primitive, gamma))
    front_position = np.full(error.shape[0], np.nan, dtype=np.float64)
    front_strength = np.full_like(front_position, np.nan)
    for sample_id in range(error.shape[0]):
        front = pressure_front_top2_metrics_np(
            prediction_primitive[sample_id : sample_id + 1],
            truth_primitive[sample_id : sample_id + 1],
            x[sample_id],
        )
        front_position[sample_id] = front["position_assignment_mae"][0]
        front_strength[sample_id] = front["strength_relative_l1"][0]
    metrics["front_position_mae"] = front_position
    metrics["front_strength_relative_l1"] = front_strength
    return metrics


def update_metrics(
    prediction_conservative: np.ndarray,
    current_conservative: np.ndarray,
    truth_conservative: np.ndarray,
    gamma: float,
) -> dict[str, np.ndarray]:
    scale = conservative_scale(gamma).reshape(1, 1, 3)
    prediction_delta = (prediction_conservative - current_conservative) / scale
    truth_delta = (truth_conservative - current_conservative) / scale
    result = spectral_pair_metrics(
        prediction_delta,
        truth_delta,
        prefix="update",
    )
    result.update(spectral_metrics(prediction_delta - truth_delta, prefix="update_error"))
    return result


def face_flux_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    gamma: float,
) -> dict[str, np.ndarray]:
    scale = conservative_scale(gamma).reshape(1, 1, 3)
    normalized_error = (prediction - target) / scale
    normal = np.ones((1, prediction.shape[1], 1), dtype=np.float64)
    normal[:, 0] = -1.0
    coefficient = np.sum(normalized_error * normal, axis=1, keepdims=True) / np.sum(
        normal**2,
        axis=1,
        keepdims=True,
    )
    gauge = normal * coefficient
    active = normalized_error - gauge
    positive_x_active = active * normal
    result = {
        "flux_raw_mse": np.mean(normalized_error**2, axis=(1, 2)),
        "flux_gauge_mse": np.mean(gauge**2, axis=(1, 2)),
        "flux_active_mse": np.mean(active**2, axis=(1, 2)),
    }
    result.update(spectral_metrics(positive_x_active, prefix="flux_active_error"))
    return result


def batch_rows(
    common: dict[str, Any],
    case_ids: np.ndarray,
    metrics: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample_id, case_id in enumerate(case_ids):
        row = {**common, "case_id": int(case_id)}
        row.update(
            {
                key: float(value[sample_id])
                for key, value in metrics.items()
            }
        )
        rows.append(row)
    return rows


def target_face_flux(
    source: Euler1DNPZ,
    case_ids: np.ndarray,
    frame: int,
    stride: int,
) -> np.ndarray:
    if source.face_flux_integral is None:
        raise RuntimeError("D013 requires solver-exported face-flux impulses")
    impulse = np.sum(
        source.face_flux_integral[case_ids, frame : frame + stride],
        axis=1,
        dtype=np.float64,
    )
    interval_dt = source.t[case_ids, frame + stride] - source.t[case_ids, frame]
    return impulse / interval_dt.reshape(-1, 1, 1)


def prediction_validity(primitive: np.ndarray) -> tuple[np.ndarray, list[str]]:
    finite = np.isfinite(primitive).all(axis=(1, 2))
    positive_density = np.all(primitive[..., 0] > 0.0, axis=1)
    positive_pressure = np.all(primitive[..., 2] > 0.0, axis=1)
    valid = finite & positive_density & positive_pressure
    reasons: list[str] = []
    for is_finite, rho_ok, pressure_ok in zip(
        finite,
        positive_density,
        positive_pressure,
        strict=True,
    ):
        if not is_finite:
            reasons.append("nonfinite_state")
        elif not rho_ok and not pressure_ok:
            reasons.append("nonpositive_density_and_pressure")
        elif not rho_ok:
            reasons.append("nonpositive_density")
        elif not pressure_ok:
            reasons.append("nonpositive_pressure")
        else:
            reasons.append("")
    return valid, reasons


def evaluate_teacher_forced(
    frozen: FrozenModel,
    source: Euler1DNPZ,
    device: torch.device,
    *,
    frame_step: int,
    shock_radius_cells: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    case_ids = frozen.test_cases
    stride = frozen.stride
    for frame in range(0, source.num_frames - stride, frame_step):
        target_frame = frame + stride
        current_primitive = source.data[case_ids, frame]
        current_conservative = primitive_to_conservative_np(
            current_primitive,
            source.gamma,
        )
        truth_primitive = source.data[case_ids, target_frame].astype(np.float64)
        truth_conservative = primitive_to_conservative_np(
            truth_primitive,
            source.gamma,
        )
        dt = source.t[case_ids, target_frame] - source.t[case_ids, frame]
        decoded = decode_batch(
            frozen,
            current_primitive,
            current_conservative,
            source.x[case_ids],
            dt,
            source.left_states[case_ids],
            source.right_states[case_ids],
            source.gamma,
            device,
        )
        prediction_primitive = cast(np.ndarray, decoded["primitive"])
        prediction_conservative = cast(np.ndarray, decoded["conservative"])
        metrics = state_metrics(
            prediction_primitive,
            prediction_conservative,
            truth_primitive,
            truth_conservative,
            source.x[case_ids],
            source.gamma,
            shock_radius_cells=shock_radius_cells,
        )
        metrics.update(
            update_metrics(
                prediction_conservative,
                current_conservative,
                truth_conservative,
                source.gamma,
            )
        )
        predicted_flux = decoded["face_flux"]
        if isinstance(predicted_flux, np.ndarray):
            metrics.update(
                face_flux_metrics(
                    predicted_flux,
                    target_face_flux(source, case_ids, frame, stride),
                    source.gamma,
                )
            )
        valid, reasons = prediction_validity(prediction_primitive)
        frame_rows = batch_rows(
            {
                "mode": "teacher_forced",
                "model": frozen.name,
                "stride": stride,
                "source_frame": frame,
                "target_frame": target_frame,
                "call": float("nan"),
                "steps_to_failure": float("nan"),
                "completed_horizon": True,
            },
            case_ids,
            metrics,
        )
        for sample_id, row in enumerate(frame_rows):
            row["proposal_valid"] = bool(valid[sample_id])
            row["termination_reason"] = reasons[sample_id]
        rows.extend(frame_rows)
    return rows


def evaluate_rollout(
    frozen: FrozenModel,
    source: Euler1DNPZ,
    device: torch.device,
    *,
    max_calls: int,
    shock_radius_cells: int,
    rollout_batch_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    case_ids = frozen.test_cases
    if rollout_batch_size < case_ids.size:
        rows: list[dict[str, Any]] = []
        for start in range(0, case_ids.size, rollout_batch_size):
            chunk = case_ids[start : start + rollout_batch_size]
            chunk_model = replace(frozen, test_cases=chunk)
            chunk_rows, _ = evaluate_rollout(
                chunk_model,
                source,
                device,
                max_calls=max_calls,
                shock_radius_cells=shock_radius_cells,
                rollout_batch_size=chunk.size,
            )
            rows.extend(chunk_rows)
        return rows, summarize_rollout_population(
            rows,
            frozen.name,
            case_ids,
            max_calls,
        )
    stride = frozen.stride
    max_available = (source.num_frames - 1) // stride
    if max_calls < 1 or max_calls > max_available:
        raise ValueError(
            f"rollout calls for {frozen.name} must be in [1, {max_available}]"
        )
    current_primitive = torch.from_numpy(
        np.ascontiguousarray(source.data[case_ids, 0])
    ).to(device)
    current_conservative = primitive_to_conservative(
        current_primitive,
        gamma=source.gamma,
    )
    x_t = torch.from_numpy(np.ascontiguousarray(source.x[case_ids])).to(device)
    left_t = torch.from_numpy(
        np.ascontiguousarray(source.left_states[case_ids])
    ).to(device)
    right_t = torch.from_numpy(
        np.ascontiguousarray(source.right_states[case_ids])
    ).to(device)
    active = np.ones(case_ids.size, dtype=bool)
    failure_calls: dict[int, int] = {}
    failure_reasons: dict[int, str] = {}
    rows: list[dict[str, Any]] = []
    row_ids_by_case: dict[int, list[int]] = {
        int(case_id): [] for case_id in case_ids
    }

    for call in range(1, max_calls + 1):
        active_indices = np.flatnonzero(active)
        if active_indices.size == 0:
            break
        active_cases = case_ids[active_indices]
        source_frame = (call - 1) * stride
        target_frame = call * stride
        dt = source.t[active_cases, target_frame] - source.t[
            active_cases,
            source_frame,
        ]
        all_active = active_indices.size == case_ids.size
        if all_active:
            active_primitive = current_primitive
            active_conservative = current_conservative
            active_x = x_t
            active_left = left_t
            active_right = right_t
        else:
            active_index_t = torch.as_tensor(
                active_indices,
                dtype=torch.long,
                device=device,
            )
            active_primitive = current_primitive.index_select(0, active_index_t)
            active_conservative = current_conservative.index_select(
                0,
                active_index_t,
            )
            active_x = x_t.index_select(0, active_index_t)
            active_left = left_t.index_select(0, active_index_t)
            active_right = right_t.index_select(0, active_index_t)
        recurrent_coordinates = str(
            getattr(frozen.args, "recurrent_coordinates", "primitive")
        )
        if recurrent_coordinates == "conservative":
            model_primitive = conservative_to_primitive(
                active_conservative,
                gamma=source.gamma,
            )
        else:
            model_primitive = active_primitive
        batch = make_euler1d_batch(
            model_primitive,
            active_x,
            torch.as_tensor(dt, dtype=current_primitive.dtype, device=device),
            current_conservative_state=(
                active_conservative
                if recurrent_coordinates == "conservative"
                else None
            ),
            gamma=source.gamma,
            left_boundary_primitive=active_left,
            right_initial_primitive=active_right,
        )
        with torch.no_grad():
            raw = frozen.model(batch)
            decoded = frozen.adapter(raw, batch)
        proposed_primitive = decoded.primitive.detach()
        proposed_conservative = decoded.conservative.detach()
        prediction_primitive = (
            proposed_primitive.cpu().numpy().astype(np.float64)
        )
        prediction_conservative = (
            proposed_conservative.cpu().numpy().astype(np.float64)
        )
        truth_primitive = source.data[active_cases, target_frame].astype(np.float64)
        truth_conservative = primitive_to_conservative_np(
            truth_primitive,
            source.gamma,
        )
        metrics = state_metrics(
            prediction_primitive,
            prediction_conservative,
            truth_primitive,
            truth_conservative,
            source.x[active_cases],
            source.gamma,
            shock_radius_cells=shock_radius_cells,
        )
        valid, reasons = prediction_validity(prediction_primitive)
        call_rows = batch_rows(
            {
                "mode": "autoregressive",
                "model": frozen.name,
                "stride": stride,
                "source_frame": source_frame,
                "target_frame": target_frame,
                "call": call,
                "steps_to_failure": float("nan"),
                "completed_horizon": False,
            },
            active_cases,
            metrics,
        )
        for sample_id, row in enumerate(call_rows):
            row["proposal_valid"] = bool(valid[sample_id])
            row["termination_reason"] = reasons[sample_id]
            row_id = len(rows)
            rows.append(row)
            row_ids_by_case[int(active_cases[sample_id])].append(row_id)
            if not valid[sample_id]:
                case_id = int(active_cases[sample_id])
                failure_calls[case_id] = call
                failure_reasons[case_id] = reasons[sample_id]

        valid_global = active_indices[valid]
        invalid_global = active_indices[~valid]
        if case_ids.size == 1 and bool(valid[0]):
            current_primitive = proposed_primitive
            current_conservative = proposed_conservative
        elif valid_global.size:
            valid_local_t = torch.as_tensor(
                np.flatnonzero(valid),
                dtype=torch.long,
                device=device,
            )
            valid_global_t = torch.as_tensor(
                valid_global,
                dtype=torch.long,
                device=device,
            )
            next_primitive = current_primitive.clone()
            next_conservative = current_conservative.clone()
            next_primitive.index_copy_(
                0,
                valid_global_t,
                proposed_primitive.index_select(0, valid_local_t),
            )
            next_conservative.index_copy_(
                0,
                valid_global_t,
                proposed_conservative.index_select(0, valid_local_t),
            )
            current_primitive = next_primitive
            current_conservative = next_conservative
        active[invalid_global] = False

    for case_id in case_ids:
        case = int(case_id)
        failure_call = failure_calls.get(case)
        completed = failure_call is None
        for row_id in row_ids_by_case[case]:
            rows[row_id]["completed_horizon"] = completed
            if failure_call is not None:
                rows[row_id]["steps_to_failure"] = failure_call - int(
                    rows[row_id]["call"]
                )

    return rows, summarize_rollout_population(
        rows,
        frozen.name,
        case_ids,
        max_calls,
    )


def summarize_rollout_population(
    rows: list[dict[str, Any]],
    model_name: str,
    case_ids: np.ndarray,
    max_calls: int,
) -> dict[str, Any]:
    invalid_rows = [row for row in rows if not bool(row["proposal_valid"])]
    failure_calls = {
        int(row["case_id"]): int(row["call"])
        for row in invalid_rows
    }
    failure_reasons = {
        int(row["case_id"]): str(row["termination_reason"])
        for row in invalid_rows
    }
    reason_counts: dict[str, int] = {}
    for reason in failure_reasons.values():
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    valid_lengths = [failure_calls.get(int(case), max_calls + 1) - 1 for case in case_ids]
    num_completed = int(case_ids.size - len(failure_calls))
    return {
        "model": model_name,
        "max_calls": max_calls,
        "num_cases": int(case_ids.size),
        "num_completed": num_completed,
        "completion_fraction": num_completed / int(case_ids.size),
        "failure_reason_counts": reason_counts,
        "valid_calls_mean": float(np.mean(valid_lengths)),
        "valid_calls_median": float(np.median(valid_lengths)),
        "first_failure_call": (
            min(failure_calls.values()) if failure_calls else None
        ),
        "last_failure_call": max(failure_calls.values()) if failure_calls else None,
    }


NON_METRIC_FIELDS = {
    "mode",
    "model",
    "case_id",
    "stride",
    "source_frame",
    "target_frame",
    "call",
    "steps_to_failure",
    "completed_horizon",
    "proposal_valid",
    "termination_reason",
    "time_band",
}


def summarize_subset(
    subset: list[dict[str, Any]],
    common: dict[str, Any],
) -> dict[str, Any]:
    result = {
        **common,
        "count": len(subset),
        "case_count": len({int(row["case_id"]) for row in subset}),
        "valid_fraction": float(
            np.mean([bool(row["proposal_valid"]) for row in subset])
        ),
    }
    metric_keys = sorted(set(subset[0]) - NON_METRIC_FIELDS)
    for key in metric_keys:
        try:
            values = np.asarray([float(row[key]) for row in subset], dtype=np.float64)
        except (TypeError, ValueError):
            continue
        finite = values[np.isfinite(values)]
        result[f"{key}_mean"] = (
            float(np.mean(finite)) if finite.size else float("nan")
        )
        result[f"{key}_median"] = (
            float(np.median(finite)) if finite.size else float("nan")
        )
    return result


def time_band(target_frame: int) -> str:
    if target_frame <= 20:
        return "early_1_20"
    if target_frame <= 50:
        return "middle_21_50"
    return "late_51_100"


def build_summaries(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    teacher_summary: list[dict[str, Any]] = []
    rollout_summary: list[dict[str, Any]] = []
    precursor_summary: list[dict[str, Any]] = []
    model_names = sorted({str(row["model"]) for row in rows})
    for model_name in model_names:
        teacher = [
            row
            for row in rows
            if row["model"] == model_name and row["mode"] == "teacher_forced"
        ]
        if teacher:
            teacher_summary.append(
                summarize_subset(teacher, {"model": model_name, "time_band": "all"})
            )
            for band in ("early_1_20", "middle_21_50", "late_51_100"):
                subset = [
                    row
                    for row in teacher
                    if time_band(int(row["target_frame"])) == band
                ]
                if subset:
                    teacher_summary.append(
                        summarize_subset(
                            subset,
                            {"model": model_name, "time_band": band},
                        )
                    )
        rollout = [
            row
            for row in rows
            if row["model"] == model_name and row["mode"] == "autoregressive"
        ]
        for call in sorted({int(row["call"]) for row in rollout}):
            subset = [row for row in rollout if int(row["call"]) == call]
            rollout_summary.append(
                summarize_subset(subset, {"model": model_name, "call": call})
            )
        for offset in (8, 4, 2, 1, 0):
            subset = [
                row
                for row in rollout
                if math.isfinite(float(row["steps_to_failure"]))
                and int(row["steps_to_failure"]) == offset
            ]
            if subset:
                precursor_summary.append(
                    summarize_subset(
                        subset,
                        {"model": model_name, "steps_to_failure": offset},
                    )
                )
    return teacher_summary, rollout_summary, precursor_summary


def mean_metric(
    rows: list[dict[str, Any]],
    *,
    model: str,
    mode: str,
    metric: str,
    call: int | None = None,
    steps_to_failure: int | None = None,
) -> float:
    subset = [
        row
        for row in rows
        if row["model"] == model
        and row["mode"] == mode
        and (call is None or int(row["call"]) == call)
        and (
            steps_to_failure is None
            or (
                math.isfinite(float(row["steps_to_failure"]))
                and int(row["steps_to_failure"]) == steps_to_failure
            )
        )
    ]
    values = np.asarray([float(row[metric]) for row in subset], dtype=np.float64)
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if finite.size else float("nan")


def safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / max(denominator, EPS)


def model_ratio(
    rows: list[dict[str, Any]],
    numerator: str,
    denominator: str,
    *,
    mode: str,
    metric: str,
    call: int | None = None,
) -> float:
    return safe_ratio(
        mean_metric(
            rows,
            model=numerator,
            mode=mode,
            metric=metric,
            call=call,
        ),
        mean_metric(
            rows,
            model=denominator,
            mode=mode,
            metric=metric,
            call=call,
        ),
    )


def matched_failure_contrast(
    rows: list[dict[str, Any]],
    failing_model: str,
    control_model: str,
    metrics: Sequence[str],
) -> dict[str, Any]:
    failing_rows = [
        row
        for row in rows
        if row["model"] == failing_model
        and row["mode"] == "autoregressive"
        and float(row["steps_to_failure"]) == 0.0
    ]
    control = {
        (int(row["case_id"]), int(row["call"])): row
        for row in rows
        if row["model"] == control_model and row["mode"] == "autoregressive"
    }
    pairs = [
        (row, control[(int(row["case_id"]), int(row["call"]))])
        for row in failing_rows
        if (int(row["case_id"]), int(row["call"])) in control
    ]
    result: dict[str, Any] = {"pair_count": len(pairs)}
    for metric in metrics:
        numerator = np.asarray([float(pair[0][metric]) for pair in pairs])
        denominator = np.asarray([float(pair[1][metric]) for pair in pairs])
        finite = np.isfinite(numerator) & np.isfinite(denominator)
        result[f"{metric}_mean_ratio"] = (
            safe_ratio(float(np.mean(numerator[finite])), float(np.mean(denominator[finite])))
            if np.any(finite)
            else float("nan")
        )
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path.name}")
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def paired_model_ratio(
    rows: list[dict[str, Any]],
    numerator: str,
    denominator: str,
    *,
    metric: str,
    call: int,
) -> dict[str, Any]:
    def indexed(model: str) -> dict[int, dict[str, Any]]:
        return {
            int(row["case_id"]): row
            for row in rows
            if row["model"] == model
            and row["mode"] == "autoregressive"
            and int(row["call"]) == call
            and bool(row["proposal_valid"])
        }

    num = indexed(numerator)
    den = indexed(denominator)
    common = sorted(set(num) & set(den))
    num_values = np.asarray([float(num[case][metric]) for case in common])
    den_values = np.asarray([float(den[case][metric]) for case in common])
    finite = np.isfinite(num_values) & np.isfinite(den_values)
    return {
        "common_cases": int(np.sum(finite)),
        "mean_ratio": (
            safe_ratio(
                float(np.mean(num_values[finite])),
                float(np.mean(den_values[finite])),
            )
            if np.any(finite)
            else float("nan")
        ),
        "wins": int(np.sum(num_values[finite] < den_values[finite])),
    }


def build_report(
    rows: list[dict[str, Any]],
    models: dict[str, FrozenModel],
    rollout_summaries: list[dict[str, Any]],
    source: Euler1DNPZ,
    runtime_seconds: float,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "experiment": "D013_euler1d_scale_spectral_diagnostic",
        "runtime_seconds": runtime_seconds,
        "spectral_contract": {
            "transform": (
                "cell-index rFFT of a spatial-mean-subtracted field after a "
                "Hann window; nonperiodic boundary leakage is therefore not "
                "interpreted as physical high-frequency energy"
            ),
            "bands": SPECTRAL_BANDS,
            "primary_high_band": "mode indices 25 through 64, above the 24-mode FNO cutoff",
            "tail_band": "mode indices 65 through Nyquist",
            "teacher_forced_semantics": (
                "update and solver-flux labels are evaluated only from truth states"
            ),
            "autoregressive_semantics": (
                "state-error spectra only; original-trajectory flux is not used "
                "as a label from generated states"
            ),
        },
        "dataset": {
            "cases": source.num_cases,
            "frames": source.num_frames,
            "cells": source.num_cells,
            "has_face_flux_integral": source.face_flux_integral is not None,
        },
        "models": {
            name: {
                "model": str(frozen.args.model),
                "target": str(frozen.args.target),
                "stride": frozen.stride,
                "test_cases": int(frozen.test_cases.size),
                "test_case_hash": hashlib.sha256(
                    frozen.test_cases.tobytes()
                ).hexdigest()[:16],
                "parameter_count": frozen.parameter_count,
                "checkpoint_sha256": frozen.checkpoint_sha256,
            }
            for name, frozen in models.items()
        },
        "rollout_summaries": rollout_summaries,
    }

    if {"full_flux", "full_residual"}.issubset(models):
        teacher_metrics = (
            "state_cons_scaled_rmse",
            "shock_cons_scaled_rmse",
            "smooth_cons_scaled_rmse",
            "error_high_25_64_rms",
            "error_tail_65_nyquist_rms",
            "error_d1_rms",
            "error_d2_rms",
            "front_position_mae",
            "front_strength_relative_l1",
            "update_high_25_64_projection_gain",
            "update_high_25_64_relative_error",
        )
        report["fullscale_teacher_flux_over_residual"] = {
            metric: model_ratio(
                rows,
                "full_flux",
                "full_residual",
                mode="teacher_forced",
                metric=metric,
            )
            for metric in teacher_metrics
        }
        precursor_metrics = (
            "state_cons_scaled_rmse",
            "shock_cons_scaled_rmse",
            "smooth_cons_scaled_rmse",
            "error_high_25_64_rms",
            "error_tail_65_nyquist_rms",
            "error_spectral_centroid_fraction",
            "error_d1_rms",
            "error_d2_rms",
            "state_tv_ratio",
            "min_density",
            "min_pressure",
        )
        report["full_flux_failure_precursor_offset0_over_offset8"] = {
            metric: safe_ratio(
                mean_metric(
                    rows,
                    model="full_flux",
                    mode="autoregressive",
                    metric=metric,
                    steps_to_failure=0,
                ),
                mean_metric(
                    rows,
                    model="full_flux",
                    mode="autoregressive",
                    metric=metric,
                    steps_to_failure=8,
                ),
            )
            for metric in precursor_metrics
        }
        report["full_flux_failure_vs_matched_residual"] = matched_failure_contrast(
            rows,
            "full_flux",
            "full_residual",
            precursor_metrics,
        )
        report["fullscale_call20_flux_over_residual"] = {
            metric: paired_model_ratio(
                rows,
                "full_flux",
                "full_residual",
                metric=metric,
                call=20,
            )
            for metric in precursor_metrics[:9]
        }

    if {"d022_generated", "d022_teacher", "d022_clean"}.issubset(models):
        d022_metrics = (
            "state_cons_scaled_rmse",
            "shock_cons_scaled_rmse",
            "smooth_cons_scaled_rmse",
            "error_high_25_64_rms",
            "error_tail_65_nyquist_rms",
            "error_spectral_centroid_fraction",
            "error_d1_rms",
            "error_d2_rms",
            "state_tv_ratio",
            "front_position_mae",
            "front_strength_relative_l1",
        )
        report["d022_generated_over_teacher"] = {
            str(call): {
                metric: paired_model_ratio(
                    rows,
                    "d022_generated",
                    "d022_teacher",
                    metric=metric,
                    call=call,
                )
                for metric in d022_metrics
            }
            for call in (20, 50, 100)
        }
        report["d022_generated_over_clean"] = {
            str(call): {
                metric: paired_model_ratio(
                    rows,
                    "d022_generated",
                    "d022_clean",
                    metric=metric,
                    call=call,
                )
                for metric in d022_metrics
            }
            for call in (20, 50, 100)
        }

    teacher_contrast = report.get("fullscale_teacher_flux_over_residual", {})
    precursor_growth = report.get(
        "full_flux_failure_precursor_offset0_over_offset8",
        {},
    )
    matched_failure = report.get("full_flux_failure_vs_matched_residual", {})
    teacher_high = bool(
        teacher_contrast
        and (
            teacher_contrast["error_high_25_64_rms"] >= 1.25
            or teacher_contrast["error_d2_rms"] >= 1.25
        )
    )
    failure_high_growth = bool(
        precursor_growth
        and (
            precursor_growth["error_high_25_64_rms"] >= 2.0
            or precursor_growth["error_d2_rms"] >= 2.0
        )
    )
    matched_high = bool(
        matched_failure
        and (
            matched_failure["error_high_25_64_rms_mean_ratio"] >= 1.5
            or matched_failure["error_d2_rms_mean_ratio"] >= 1.5
        )
    )
    if teacher_high and failure_high_growth and matched_high:
        classification = "high_frequency_precursor_supports_local_dissipation_probe"
    elif failure_high_growth and matched_high:
        classification = "recurrent_high_frequency_growth_not_one_step_fit"
    elif teacher_contrast and teacher_contrast["shock_cons_scaled_rmse"] >= 1.25:
        classification = "shock_local_error_without_declared_high_frequency_precursor"
    else:
        classification = "mixed_or_low_frequency_failure"
    report["predeclared_flags"] = {
        "teacher_flux_high_frequency_excess": teacher_high,
        "flux_failure_high_frequency_growth": failure_high_growth,
        "flux_failure_high_frequency_excess_vs_matched_residual": matched_high,
    }
    report["classification"] = classification
    return report


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    source = load_euler1d_npz(args.data_path)
    if source.face_flux_integral is None:
        raise RuntimeError("D013 requires a dataset with face_flux_integral")
    device = select_device(args.device, args.gpu)
    models = {
        name: load_frozen_model(name, Path(path), device)
        for name, path in args.checkpoint
    }
    for name, frozen in models.items():
        if np.any(frozen.test_cases < 0) or np.any(
            frozen.test_cases >= source.num_cases
        ):
            raise RuntimeError(f"test split for {name} is incompatible with dataset")
        if frozen.stride < 1 or frozen.stride >= source.num_frames:
            raise RuntimeError(f"invalid native stride for {name}")
    call_overrides = {name: int(value) for name, value in args.rollout_calls}
    for name, calls in call_overrides.items():
        if calls < 1:
            raise ValueError(f"rollout calls for {name} must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    rollout_summaries: list[dict[str, Any]] = []
    for model_id, (name, frozen) in enumerate(models.items(), start=1):
        print(
            f"D013 model={name} ({model_id}/{len(models)}) "
            f"cases={frozen.test_cases.size} stride={frozen.stride}",
            flush=True,
        )
        teacher_rows = evaluate_teacher_forced(
            frozen,
            source,
            device,
            frame_step=args.teacher_frame_step,
            shock_radius_cells=args.shock_radius_cells,
        )
        max_available = (source.num_frames - 1) // frozen.stride
        max_calls = call_overrides.get(name, max_available)
        rollout_rows, rollout_summary = evaluate_rollout(
            frozen,
            source,
            device,
            max_calls=max_calls,
            shock_radius_cells=args.shock_radius_cells,
            rollout_batch_size=args.rollout_batch_size,
        )
        all_rows.extend(teacher_rows)
        all_rows.extend(rollout_rows)
        rollout_summaries.append(rollout_summary)
        print(
            f"completed model={name} teacher_rows={len(teacher_rows)} "
            f"rollout_rows={len(rollout_rows)} completion="
            f"{rollout_summary['num_completed']}/{rollout_summary['num_cases']}",
            flush=True,
        )

    teacher_summary, rollout_summary, precursor_summary = build_summaries(all_rows)
    report = build_report(
        all_rows,
        models,
        rollout_summaries,
        source,
        time.perf_counter() - started,
    )
    write_csv(args.output_dir / "per_sample_metrics.csv", all_rows)
    write_csv(args.output_dir / "teacher_summary.csv", teacher_summary)
    write_csv(args.output_dir / "rollout_summary.csv", rollout_summary)
    if precursor_summary:
        write_csv(args.output_dir / "failure_precursor_summary.csv", precursor_summary)
    (args.output_dir / "report.json").write_text(
        json.dumps(json_ready(report), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(json_ready(report["predeclared_flags"]), indent=2), flush=True)
    print(f"classification={report['classification']}", flush=True)
    return report


def main(argv: Sequence[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
