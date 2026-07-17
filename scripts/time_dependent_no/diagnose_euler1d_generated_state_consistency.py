"""Diagnose whether burn-in training learns Euler continuation off trajectory.

The diagnostic constructs states with one frozen generated-burn-in checkpoint,
then compares several frozen learned maps against a WENO-HLLC-ADER reference
advance initialized from exactly the same state. This separates learning the
PDE continuation map from learning a correction toward one stored trajectory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence, cast

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.time_dependent_no.euler1d_weno_hllc_ader_dataset import (  # noqa: E402
    apply_boundary_conditions,
    conservative_to_primitive as conservative_to_primitive_np,
    primitive_to_conservative as primitive_to_conservative_np,
    stable_dt,
    step_first_order_hllc,
    step_weno_hllc_ader2,
)
from scripts.time_dependent_no.train_euler1d_target_ladder import (  # noqa: E402
    PrimitiveNormalizer,
    build_model,
    json_ready,
    pressure_front_top2_metrics_np,
)
from utility.time_dependent_no.euler1d import (  # noqa: E402
    make_euler1d_batch,
    primitive_to_conservative as primitive_to_conservative_torch,
)
from utility.time_dependent_no.euler1d_data import (  # noqa: E402
    Euler1DNPZ,
    load_euler1d_npz,
)
from utility.time_dependent_no.euler1d_models import Euler1DTarget  # noqa: E402
from utility.time_dependent_no.euler1d_targets import make_target_adapter  # noqa: E402


EPS = 1.0e-12
MODEL_NAMES = ("clean_control", "teacher_offset8", "generated_burnin8")


def model_conservative_from_primitive(
    primitive: np.ndarray,
    gamma: float,
) -> np.ndarray:
    primitive_t = torch.as_tensor(primitive, dtype=torch.float32)
    return (
        primitive_to_conservative_torch(primitive_t, gamma=gamma)
        .numpy()
        .astype(np.float64)
    )


@dataclass(frozen=True)
class SolverReplayConfig:
    gamma: float
    cfl: float
    ng: int = 3
    rho_floor: float = 1.0e-10
    p_floor: float = 1.0e-10
    use_shock_flattening: bool = True
    use_hlle_on_troubled_faces: bool = True
    shock_sensor_threshold: float = 0.05
    shock_flatten_radius: int = 4


@dataclass
class LoadedCheckpoint:
    name: str
    model: nn.Module
    adapter: nn.Module
    args: argparse.Namespace
    test_cases: np.ndarray
    checkpoint_sha256: str


def parse_integer_list(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not parsed:
        raise argparse.ArgumentTypeError("expected a comma-separated integer list")
    if len(set(parsed)) != len(parsed):
        raise argparse.ArgumentTypeError("integer list contains duplicates")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--clean-checkpoint", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--generated-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--start-frames",
        type=parse_integer_list,
        default=parse_integer_list("0,10,20,30,40,50,60,70,80"),
    )
    parser.add_argument(
        "--prefix-depths",
        type=parse_integer_list,
        default=parse_integer_list("0,2,4,8"),
    )
    parser.add_argument("--solver-workers", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--rho-floor", type=float, default=1.0e-10)
    parser.add_argument("--p-floor", type=float, default=1.0e-10)
    parser.add_argument("--ghost-cells", type=int, default=3)
    parser.add_argument("--shock-radius-cells", type=int, default=4)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260716)
    args = parser.parse_args(argv)
    if min(args.start_frames) < 0:
        parser.error("--start-frames must be nonnegative")
    if min(args.prefix_depths) < 0:
        parser.error("--prefix-depths must be nonnegative")
    if args.solver_workers < 1:
        parser.error("--solver-workers must be positive")
    if args.ghost_cells < 3:
        parser.error("--ghost-cells must be at least 3 for WENO5")
    if args.shock_radius_cells < 0:
        parser.error("--shock-radius-cells must be nonnegative")
    if args.bootstrap_replicates < 0:
        parser.error("--bootstrap-replicates must be nonnegative")
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


def scalar_metadata(source: Euler1DNPZ, key: str, default: Any) -> Any:
    value = source.metadata.get(key, default)
    if isinstance(value, np.ndarray) and value.shape == ():
        return value.item()
    return value


def solver_config_from_source(
    source: Euler1DNPZ,
    *,
    ng: int,
    rho_floor: float,
    p_floor: float,
) -> SolverReplayConfig:
    return SolverReplayConfig(
        gamma=source.gamma,
        cfl=float(scalar_metadata(source, "cfl", 0.35)),
        ng=ng,
        rho_floor=rho_floor,
        p_floor=p_floor,
        use_shock_flattening=bool(
            scalar_metadata(source, "use_shock_flattening", True)
        ),
        use_hlle_on_troubled_faces=bool(
            scalar_metadata(source, "use_hlle_on_troubled_faces", True)
        ),
        shock_sensor_threshold=float(
            scalar_metadata(source, "shock_sensor_threshold", 0.05)
        ),
        shock_flatten_radius=int(
            scalar_metadata(source, "shock_flatten_radius", 4)
        ),
    )


def advance_reference_conservative(
    conservative: np.ndarray,
    left_state: np.ndarray,
    dx: float,
    interval_dt: float,
    config: SolverReplayConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Advance one saved interval from arbitrary admissible cell averages."""

    conservative = np.asarray(conservative, dtype=np.float64)
    if conservative.ndim != 2 or conservative.shape[-1] != 3:
        raise ValueError("conservative must have shape [cells, 3]")
    if interval_dt <= 0.0:
        raise ValueError("interval_dt must be positive")
    nx = conservative.shape[0]
    U = np.zeros((nx + 2 * config.ng, 3), dtype=np.float64)
    U[config.ng : config.ng + nx] = conservative
    apply_boundary_conditions(U, left_state, config.gamma, config.ng)

    elapsed = 0.0
    substeps = 0
    retry_halvings = 0
    fallback_steps = 0
    while elapsed < interval_dt - 1.0e-14:
        dt = min(
            stable_dt(U, dx, config.cfl, config.gamma, config.ng),
            interval_dt - elapsed,
        )
        success = False
        dt_try = dt
        for retry in range(12):
            U_candidate, ok = step_weno_hllc_ader2(
                U,
                left_state,
                dx,
                dt_try,
                config.gamma,
                config.ng,
                config.rho_floor,
                config.p_floor,
                config.use_shock_flattening,
                config.use_hlle_on_troubled_faces,
                config.shock_sensor_threshold,
                config.shock_flatten_radius,
            )
            if ok:
                U = U_candidate
                elapsed += dt_try
                substeps += 1
                retry_halvings += retry
                success = True
                break
            dt_try *= 0.5

        if not success:
            dt_try = min(0.25 * dt, interval_dt - elapsed)
            U_candidate, ok = step_first_order_hllc(
                U,
                left_state,
                dx,
                dt_try,
                config.gamma,
                config.ng,
                config.rho_floor,
                config.p_floor,
            )
            if not ok:
                raise RuntimeError("reference replay failed positivity fallback")
            U = U_candidate
            elapsed += dt_try
            substeps += 1
            fallback_steps += 1

    result_cons = U[config.ng : config.ng + nx].copy()
    result_primitive = conservative_to_primitive_np(
        result_cons,
        config.gamma,
        config.rho_floor,
        config.p_floor,
    )
    return result_primitive, result_cons, {
        "substeps": substeps,
        "retry_halvings": retry_halvings,
        "fallback_steps": fallback_steps,
    }


def load_checkpoint(
    name: str,
    path: Path,
    device: torch.device,
) -> LoadedCheckpoint:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    args = argparse.Namespace(**checkpoint["args"])
    target = cast(Euler1DTarget, str(checkpoint["target"]))
    normalizer = PrimitiveNormalizer(
        mean=checkpoint["input_normalizer_mean"],
        std=checkpoint["input_normalizer_std"],
        coordinates=str(checkpoint["input_coordinates"]),
        normalization=str(checkpoint["input_normalization"]),
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
    return LoadedCheckpoint(
        name=name,
        model=model,
        adapter=adapter,
        args=args,
        test_cases=np.asarray(checkpoint["test_cases"], dtype=np.int64),
        checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def predict_states(
    checkpoint: LoadedCheckpoint,
    primitive: np.ndarray,
    conservative: np.ndarray,
    x: np.ndarray,
    dt: np.ndarray,
    left_state: np.ndarray,
    right_state: np.ndarray,
    gamma: float,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    primitive_t = torch.as_tensor(primitive, dtype=torch.float32, device=device)
    conservative_t = torch.as_tensor(
        conservative,
        dtype=torch.float32,
        device=device,
    )
    x_t = torch.as_tensor(x, dtype=torch.float32, device=device)
    dt_t = torch.as_tensor(dt, dtype=torch.float32, device=device)
    left_t = torch.as_tensor(left_state, dtype=torch.float32, device=device)
    right_t = torch.as_tensor(right_state, dtype=torch.float32, device=device)
    batch = make_euler1d_batch(
        primitive_t,
        x_t,
        dt_t,
        current_conservative_state=conservative_t,
        gamma=gamma,
        left_boundary_primitive=left_t,
        right_initial_primitive=right_t,
    )
    with torch.no_grad():
        decoded = checkpoint.adapter(checkpoint.model(batch), batch)
    return (
        decoded.primitive.detach().cpu().numpy().astype(np.float64),
        decoded.conservative.detach().cpu().numpy().astype(np.float64),
    )


def conservative_scale(gamma: float) -> np.ndarray:
    return np.array((1.0, 1.0, 1.0 / (gamma - 1.0)), dtype=np.float64)


def scaled_error(
    prediction: np.ndarray,
    target: np.ndarray,
    scale: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[float, float]:
    diff = (np.asarray(prediction) - np.asarray(target)) / scale
    target_scaled = np.asarray(target) / scale
    if mask is not None:
        diff = diff[mask]
        target_scaled = target_scaled[mask]
    absolute = float(np.sqrt(np.mean(np.square(diff))))
    relative = float(
        np.linalg.norm(diff.reshape(-1))
        / max(np.linalg.norm(target_scaled.reshape(-1)), EPS)
    )
    return absolute, relative


def shock_region_mask(
    truth_primitive: np.ndarray,
    *,
    radius_cells: int,
    num_fronts: int = 2,
    min_separation_cells: int = 4,
) -> np.ndarray:
    pressure_jump = np.abs(np.diff(np.asarray(truth_primitive)[:, 2]))
    selected: list[int] = []
    for face_id in np.argsort(-pressure_jump, kind="stable"):
        if all(
            abs(int(face_id) - previous) > min_separation_cells
            for previous in selected
        ):
            selected.append(int(face_id))
            if len(selected) == num_fronts:
                break
    mask = np.zeros(truth_primitive.shape[0], dtype=bool)
    for face_id in selected:
        lo = max(0, face_id - radius_cells + 1)
        hi = min(mask.size, face_id + radius_cells + 1)
        mask[lo:hi] = True
    return mask


def characteristic_rms(
    conservative_error: np.ndarray,
    base_primitive: np.ndarray,
    gamma: float,
    mask: np.ndarray | None = None,
) -> tuple[float, float, float]:
    d_rho, d_momentum, d_energy = np.moveaxis(conservative_error, -1, 0)
    rho, velocity, pressure = np.moveaxis(base_primitive, -1, 0)
    sound_speed = np.sqrt(gamma * pressure / rho)
    d_velocity = (d_momentum - velocity * d_rho) / rho
    d_pressure = (gamma - 1.0) * (
        d_energy - velocity * d_momentum + 0.5 * velocity**2 * d_rho
    )
    alpha_minus = (d_pressure - rho * sound_speed * d_velocity) / (
        2.0 * sound_speed**2
    )
    alpha_plus = (d_pressure + rho * sound_speed * d_velocity) / (
        2.0 * sound_speed**2
    )
    alpha_contact = d_rho - d_pressure / sound_speed**2
    if mask is not None:
        alpha_minus = alpha_minus[mask]
        alpha_contact = alpha_contact[mask]
        alpha_plus = alpha_plus[mask]
    return tuple(
        float(np.sqrt(np.mean(np.square(value))))
        for value in (alpha_minus, alpha_contact, alpha_plus)
    )


def state_error_metrics(
    prediction_primitive: np.ndarray,
    prediction_conservative: np.ndarray,
    target_primitive: np.ndarray,
    target_conservative: np.ndarray,
    x: np.ndarray,
    gamma: float,
    *,
    prefix: str,
    shock_radius_cells: int,
) -> dict[str, float]:
    scale = conservative_scale(gamma)
    absolute, relative = scaled_error(
        prediction_conservative,
        target_conservative,
        scale,
    )
    primitive_relative = float(
        np.linalg.norm((prediction_primitive - target_primitive).reshape(-1))
        / max(np.linalg.norm(target_primitive.reshape(-1)), EPS)
    )
    shock_mask = shock_region_mask(
        target_primitive,
        radius_cells=shock_radius_cells,
    )
    shock_absolute, shock_relative = scaled_error(
        prediction_conservative,
        target_conservative,
        scale,
        shock_mask,
    )
    smooth_absolute, smooth_relative = scaled_error(
        prediction_conservative,
        target_conservative,
        scale,
        ~shock_mask,
    )
    char_minus, char_contact, char_plus = characteristic_rms(
        prediction_conservative - target_conservative,
        target_primitive,
        gamma,
    )
    front = pressure_front_top2_metrics_np(
        prediction_primitive[None],
        target_primitive[None],
        x,
    )
    return {
        f"{prefix}_cons_scaled_rmse": absolute,
        f"{prefix}_cons_scaled_rel_l2": relative,
        f"{prefix}_primitive_rel_l2": primitive_relative,
        f"{prefix}_shock_cons_scaled_rmse": shock_absolute,
        f"{prefix}_shock_cons_scaled_rel_l2": shock_relative,
        f"{prefix}_smooth_cons_scaled_rmse": smooth_absolute,
        f"{prefix}_smooth_cons_scaled_rel_l2": smooth_relative,
        f"{prefix}_char_minus_rms": char_minus,
        f"{prefix}_char_contact_rms": char_contact,
        f"{prefix}_char_plus_rms": char_plus,
        f"{prefix}_front_position_mae": float(front["position_assignment_mae"][0]),
        f"{prefix}_front_strength_relative_l1": float(
            front["strength_relative_l1"][0]
        ),
    }


def cancellation_metrics(
    prediction_conservative: np.ndarray,
    reference_conservative: np.ndarray,
    truth_conservative: np.ndarray,
    gamma: float,
) -> dict[str, float]:
    scale = conservative_scale(gamma)
    reference_error = ((reference_conservative - truth_conservative) / scale).reshape(
        -1
    )
    correction = ((prediction_conservative - reference_conservative) / scale).reshape(
        -1
    )
    reference_norm = float(np.linalg.norm(reference_error))
    correction_norm = float(np.linalg.norm(correction))
    if reference_norm <= EPS or correction_norm <= EPS:
        cosine = float("nan")
    else:
        cosine = float(
            np.dot(correction, -reference_error)
            / (correction_norm * reference_norm)
        )
    return {
        "correction_toward_truth_cosine": cosine,
        "correction_norm_over_reference_defect": correction_norm
        / max(reference_norm, EPS),
    }


GROUP_METRICS = (
    "current_truth_cons_scaled_rmse",
    "reference_truth_cons_scaled_rmse",
    "model_truth_cons_scaled_rmse",
    "model_reference_cons_scaled_rmse",
    "model_truth_primitive_rel_l2",
    "model_reference_primitive_rel_l2",
    "model_truth_shock_cons_scaled_rmse",
    "model_truth_smooth_cons_scaled_rmse",
    "model_reference_shock_cons_scaled_rmse",
    "model_reference_smooth_cons_scaled_rmse",
    "model_truth_char_minus_rms",
    "model_truth_char_contact_rms",
    "model_truth_char_plus_rms",
    "model_reference_char_minus_rms",
    "model_reference_char_contact_rms",
    "model_reference_char_plus_rms",
    "model_truth_front_position_mae",
    "model_truth_front_strength_relative_l1",
    "model_reference_front_position_mae",
    "model_reference_front_strength_relative_l1",
    "correction_toward_truth_cosine",
    "correction_norm_over_reference_defect",
    "truth_error_ratio_to_reference",
    "truth_error_reduction_from_reference",
    "reference_substeps",
    "reference_retry_halvings",
    "reference_fallback_steps",
)


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: list[dict[str, Any]] = []
    for variant in MODEL_NAMES:
        depths = sorted({int(row["prefix_depth"]) for row in rows})
        for depth in depths:
            subset = [
                row
                for row in rows
                if row["variant"] == variant and int(row["prefix_depth"]) == depth
            ]
            summary: dict[str, Any] = {
                "variant": variant,
                "prefix_depth": depth,
                "count": len(subset),
                "case_count": len({int(row["case_id"]) for row in subset}),
            }
            for metric in GROUP_METRICS:
                values = np.asarray(
                    [float(row[metric]) for row in subset],
                    dtype=np.float64,
                )
                finite = values[np.isfinite(values)]
                summary[f"{metric}_mean"] = (
                    float(np.mean(finite)) if finite.size else float("nan")
                )
                summary[f"{metric}_median"] = (
                    float(np.median(finite)) if finite.size else float("nan")
                )
            grouped.append(summary)
    return grouped


def clustered_ratio(
    rows: list[dict[str, Any]],
    numerator: str,
    denominator: str,
    depth: int,
    metric: str,
    *,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    def case_means(variant: str) -> dict[int, float]:
        result: dict[int, float] = {}
        case_ids = {
            int(row["case_id"])
            for row in rows
            if row["variant"] == variant and int(row["prefix_depth"]) == depth
        }
        for case_id in case_ids:
            values = np.asarray(
                [
                    float(row[metric])
                    for row in rows
                    if row["variant"] == variant
                    and int(row["prefix_depth"]) == depth
                    and int(row["case_id"]) == case_id
                ],
                dtype=np.float64,
            )
            finite = values[np.isfinite(values)]
            if finite.size:
                result[case_id] = float(np.mean(finite))
        return result

    num = case_means(numerator)
    den = case_means(denominator)
    case_ids = np.asarray(sorted(set(num) & set(den)), dtype=np.int64)
    if case_ids.size == 0:
        return {
            "ratio": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
        }
    num_values = np.asarray([num[int(case)] for case in case_ids])
    den_values = np.asarray([den[int(case)] for case in case_ids])
    ratio = float(np.mean(num_values) / max(np.mean(den_values), EPS))
    if replicates == 0:
        return {"ratio": ratio, "ci_low": float("nan"), "ci_high": float("nan")}
    rng = np.random.default_rng(seed)
    sample_ids = rng.integers(0, case_ids.size, size=(replicates, case_ids.size))
    ratios = np.mean(num_values[sample_ids], axis=1) / np.maximum(
        np.mean(den_values[sample_ids], axis=1),
        EPS,
    )
    return {
        "ratio": ratio,
        "ci_low": float(np.quantile(ratios, 0.025)),
        "ci_high": float(np.quantile(ratios, 0.975)),
    }


def build_contrasts(
    rows: list[dict[str, Any]],
    depths: tuple[int, ...],
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    metrics = (
        "model_truth_cons_scaled_rmse",
        "model_reference_cons_scaled_rmse",
        "model_truth_shock_cons_scaled_rmse",
        "model_reference_shock_cons_scaled_rmse",
        "model_truth_front_position_mae",
        "model_truth_front_strength_relative_l1",
    )
    pairs = (
        ("generated_vs_teacher", "generated_burnin8", "teacher_offset8"),
        ("generated_vs_clean", "generated_burnin8", "clean_control"),
        ("teacher_vs_clean", "teacher_offset8", "clean_control"),
    )
    contrasts: dict[str, Any] = {}
    for depth in depths:
        depth_result: dict[str, Any] = {}
        for pair_id, numerator, denominator in pairs:
            depth_result[pair_id] = {
                metric: clustered_ratio(
                    rows,
                    numerator,
                    denominator,
                    depth,
                    metric,
                    replicates=replicates,
                    seed=seed + 1009 * depth + 37 * metric_id,
                )
                for metric_id, metric in enumerate(metrics)
            }
        contrasts[str(depth)] = depth_result
    return contrasts


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def validate_checkpoints(
    checkpoints: dict[str, LoadedCheckpoint],
    source: Euler1DNPZ,
) -> np.ndarray:
    generated = checkpoints["generated_burnin8"]
    expected = generated.test_cases
    for name, checkpoint in checkpoints.items():
        if not np.array_equal(checkpoint.test_cases, expected):
            raise RuntimeError(f"test split mismatch for {name}")
        if str(checkpoint.args.model) != "fno":
            raise RuntimeError(f"{name} is not an FNO checkpoint")
        if str(checkpoint.args.target) != "residual":
            raise RuntimeError(f"{name} is not a residual checkpoint")
        if int(checkpoint.args.step_stride) != 1:
            raise RuntimeError(f"{name} does not use saved-frame stride one")
    if np.any(expected < 0) or np.any(expected >= source.num_cases):
        raise RuntimeError("checkpoint test split is incompatible with the dataset")
    return expected


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    source = load_euler1d_npz(args.data_path)
    max_required_frame = max(args.start_frames) + max(args.prefix_depths) + 1
    if max_required_frame >= source.num_frames:
        raise ValueError(
            f"requested frame {max_required_frame}, but dataset ends at "
            f"{source.num_frames - 1}"
        )
    device = select_device(args.device, args.gpu)
    checkpoint_paths = {
        "clean_control": args.clean_checkpoint,
        "teacher_offset8": args.teacher_checkpoint,
        "generated_burnin8": args.generated_checkpoint,
    }
    checkpoints = {
        name: load_checkpoint(name, path, device)
        for name, path in checkpoint_paths.items()
    }
    case_ids = validate_checkpoints(checkpoints, source)
    generated = checkpoints["generated_burnin8"]
    solver_config = solver_config_from_source(
        source,
        ng=args.ghost_cells,
        rho_floor=args.rho_floor,
        p_floor=args.p_floor,
    )
    rows: list[dict[str, Any]] = []
    depths = tuple(sorted(args.prefix_depths))
    max_depth = max(depths)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"D023 cases={case_ids.size} starts={len(args.start_frames)} "
        f"depths={depths} device={device} solver_workers={args.solver_workers}",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=args.solver_workers) as executor:
        for start_id, start_frame in enumerate(args.start_frames, start=1):
            current_primitive = source.data[case_ids, start_frame].astype(np.float64)
            current_conservative = model_conservative_from_primitive(
                current_primitive,
                source.gamma,
            )
            state_bank: dict[int, tuple[np.ndarray, np.ndarray]] = {
                0: (current_primitive.copy(), current_conservative.copy())
            }
            for step in range(1, max_depth + 1):
                frame = start_frame + step - 1
                dt = source.t[case_ids, frame + 1] - source.t[case_ids, frame]
                current_primitive, current_conservative = predict_states(
                    generated,
                    current_primitive,
                    current_conservative,
                    source.x[case_ids],
                    dt,
                    source.left_states[case_ids],
                    source.right_states[case_ids],
                    source.gamma,
                    device,
                )
                if not np.isfinite(current_primitive).all():
                    raise RuntimeError(
                        f"generated state became nonfinite at start={start_frame}, "
                        f"depth={step}"
                    )
                if np.any(current_primitive[..., 0] <= 0.0) or np.any(
                    current_primitive[..., 2] <= 0.0
                ):
                    raise RuntimeError(
                        f"generated state became nonpositive at start={start_frame}, "
                        f"depth={step}"
                    )
                if step in depths:
                    state_bank[step] = (
                        current_primitive.copy(),
                        current_conservative.copy(),
                    )

            for depth in depths:
                frame = start_frame + depth
                generated_primitive, generated_conservative = state_bank[depth]
                truth_current_primitive = source.data[case_ids, frame].astype(np.float64)
                truth_current_conservative = primitive_to_conservative_np(
                    truth_current_primitive,
                    source.gamma,
                )
                truth_next_primitive = source.data[case_ids, frame + 1].astype(
                    np.float64
                )
                truth_next_conservative = primitive_to_conservative_np(
                    truth_next_primitive,
                    source.gamma,
                )
                dt = (source.t[case_ids, frame + 1] - source.t[case_ids, frame]).astype(
                    np.float64
                )
                dx = np.mean(np.diff(source.x[case_ids], axis=1), axis=1).astype(
                    np.float64
                )
                futures = [
                    executor.submit(
                        advance_reference_conservative,
                        generated_conservative[case_index],
                        source.left_states[case_id],
                        float(dx[case_index]),
                        float(dt[case_index]),
                        solver_config,
                    )
                    for case_index, case_id in enumerate(case_ids)
                ]
                reference = [future.result() for future in futures]
                reference_primitive = np.stack([item[0] for item in reference])
                reference_conservative = np.stack([item[1] for item in reference])
                reference_diagnostics = [item[2] for item in reference]

                model_predictions = {
                    name: predict_states(
                        checkpoint,
                        generated_primitive,
                        generated_conservative,
                        source.x[case_ids],
                        dt,
                        source.left_states[case_ids],
                        source.right_states[case_ids],
                        source.gamma,
                        device,
                    )
                    for name, checkpoint in checkpoints.items()
                }

                for case_index, case_id in enumerate(case_ids):
                    current_metrics = state_error_metrics(
                        generated_primitive[case_index],
                        generated_conservative[case_index],
                        truth_current_primitive[case_index],
                        truth_current_conservative[case_index],
                        source.x[case_id],
                        source.gamma,
                        prefix="current_truth",
                        shock_radius_cells=args.shock_radius_cells,
                    )
                    reference_metrics = state_error_metrics(
                        reference_primitive[case_index],
                        reference_conservative[case_index],
                        truth_next_primitive[case_index],
                        truth_next_conservative[case_index],
                        source.x[case_id],
                        source.gamma,
                        prefix="reference_truth",
                        shock_radius_cells=args.shock_radius_cells,
                    )
                    for name, (prediction_primitive, prediction_conservative) in (
                        model_predictions.items()
                    ):
                        model_truth_metrics = state_error_metrics(
                            prediction_primitive[case_index],
                            prediction_conservative[case_index],
                            truth_next_primitive[case_index],
                            truth_next_conservative[case_index],
                            source.x[case_id],
                            source.gamma,
                            prefix="model_truth",
                            shock_radius_cells=args.shock_radius_cells,
                        )
                        model_reference_metrics = state_error_metrics(
                            prediction_primitive[case_index],
                            prediction_conservative[case_index],
                            reference_primitive[case_index],
                            reference_conservative[case_index],
                            source.x[case_id],
                            source.gamma,
                            prefix="model_reference",
                            shock_radius_cells=args.shock_radius_cells,
                        )
                        cancellation = cancellation_metrics(
                            prediction_conservative[case_index],
                            reference_conservative[case_index],
                            truth_next_conservative[case_index],
                            source.gamma,
                        )
                        model_truth_abs = model_truth_metrics[
                            "model_truth_cons_scaled_rmse"
                        ]
                        reference_truth_abs = reference_metrics[
                            "reference_truth_cons_scaled_rmse"
                        ]
                        row = {
                            "variant": name,
                            "case_id": int(case_id),
                            "start_frame": int(start_frame),
                            "prefix_depth": int(depth),
                            "state_frame": int(frame),
                            "target_frame": int(frame + 1),
                            "dt": float(dt[case_index]),
                            **current_metrics,
                            **reference_metrics,
                            **model_truth_metrics,
                            **model_reference_metrics,
                            **cancellation,
                            "truth_error_ratio_to_reference": model_truth_abs
                            / max(reference_truth_abs, EPS),
                            "truth_error_reduction_from_reference": 1.0
                            - model_truth_abs / max(reference_truth_abs, EPS),
                            "reference_substeps": int(
                                reference_diagnostics[case_index]["substeps"]
                            ),
                            "reference_retry_halvings": int(
                                reference_diagnostics[case_index]["retry_halvings"]
                            ),
                            "reference_fallback_steps": int(
                                reference_diagnostics[case_index]["fallback_steps"]
                            ),
                        }
                        rows.append(row)
            print(
                f"completed start_frame={start_frame} "
                f"({start_id}/{len(args.start_frames)}) rows={len(rows)}",
                flush=True,
            )

    grouped = summarize_rows(rows)
    contrasts = build_contrasts(
        rows,
        depths,
        replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed,
    )
    decision_depth = max(depths)
    decision = contrasts[str(decision_depth)]["generated_vs_teacher"]
    truth_ratio = decision["model_truth_cons_scaled_rmse"]["ratio"]
    reference_ratio = decision["model_reference_cons_scaled_rmse"]["ratio"]
    generated_depth_rows = [
        row
        for row in rows
        if row["variant"] == "generated_burnin8"
        and int(row["prefix_depth"]) == decision_depth
    ]
    cancellation_values = np.asarray(
        [row["correction_toward_truth_cosine"] for row in generated_depth_rows],
        dtype=np.float64,
    )
    cancellation_values = cancellation_values[np.isfinite(cancellation_values)]
    cancellation_mean = (
        float(np.mean(cancellation_values))
        if cancellation_values.size
        else float("nan")
    )
    if truth_ratio <= 0.90 and reference_ratio <= 0.90:
        classification = "off_manifold_pde_map_improvement"
    elif truth_ratio <= 0.90 and reference_ratio >= 0.95 and cancellation_mean > 0.0:
        classification = "trajectory_correction_or_denoising"
    elif reference_ratio <= 0.90 and truth_ratio >= 0.95:
        classification = "pde_consistency_without_trajectory_recovery"
    else:
        classification = "mixed_or_inconclusive"

    prefix_zero = [
        row
        for row in rows
        if row["variant"] == "generated_burnin8" and row["prefix_depth"] == 0
    ]
    replay_baseline = float(
        np.mean([row["reference_truth_cons_scaled_rmse"] for row in prefix_zero])
    )
    prefix_max = [
        row
        for row in rows
        if row["variant"] == "generated_burnin8"
        and row["prefix_depth"] == decision_depth
    ]
    perturbed_reference_gap = float(
        np.mean([row["reference_truth_cons_scaled_rmse"] for row in prefix_max])
    )
    report = {
        "experiment": "D023_generated_state_solver_consistency",
        "classification": classification,
        "predeclared_thresholds": {
            "pde_map": "generated/teacher truth and reference error ratios <= 0.90",
            "trajectory_correction": (
                "truth ratio <= 0.90, reference ratio >= 0.95, and positive "
                "generated correction-to-truth cosine"
            ),
            "pde_consistency_only": "reference ratio <= 0.90 and truth ratio >= 0.95",
        },
        "decision_depth": decision_depth,
        "decision_values": {
            "generated_vs_teacher_truth_error_ratio": truth_ratio,
            "generated_vs_teacher_reference_error_ratio": reference_ratio,
            "generated_correction_toward_truth_cosine_mean": cancellation_mean,
        },
        "reference_replay": {
            "prefix_zero_cons_scaled_rmse_mean": replay_baseline,
            "max_prefix_reference_truth_cons_scaled_rmse_mean": (
                perturbed_reference_gap
            ),
            "max_prefix_over_replay_baseline": perturbed_reference_gap
            / max(replay_baseline, EPS),
        },
        "sample_count": len(rows),
        "state_count": len(rows) // len(MODEL_NAMES),
        "case_ids": case_ids,
        "start_frames": args.start_frames,
        "prefix_depths": depths,
        "checkpoints": {
            name: {
                "sha256": checkpoint.checkpoint_sha256,
                "parameter_count": sum(
                    parameter.numel() for parameter in checkpoint.model.parameters()
                ),
            }
            for name, checkpoint in checkpoints.items()
        },
        "solver_replay": {
            **solver_config.__dict__,
            "rho_floor_and_p_floor_source": (
                "explicit diagnostic arguments; dataset metadata does not store floors"
            ),
        },
        "contrasts": contrasts,
        "grouped_summary": grouped,
        "runtime_seconds": time.perf_counter() - started,
    }
    write_csv(args.output_dir / "per_sample.csv", rows)
    write_csv(args.output_dir / "grouped_summary.csv", grouped)
    (args.output_dir / "report.json").write_text(
        json.dumps(json_ready(report), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(json_ready(report["decision_values"]), indent=2), flush=True)
    print(f"classification={classification}", flush=True)
    return report


def main(argv: Sequence[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
