"""Decompose frozen large-step Euler flow maps against same-state solver replay.

This D033 diagnostic is training-free. It first verifies that replaying the
documented reference solver from serialized truth reproduces the next stored
state. Only after that gate passes does it compare each frozen learned map
with a reference macro advance initialized from the identical truth or
on-policy state. A one-call reference replacement is also used as an oracle
diagnostic; it is not a proposed inference method.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from scripts.time_dependent_no.diagnose_euler1d_generated_state_consistency import (  # noqa: E402
    SolverReplayConfig,
    advance_reference_conservative,
    cancellation_metrics,
    solver_config_from_source,
    state_error_metrics,
)
from scripts.time_dependent_no.evaluate_euler1d_flow_map_frontier import (  # noqa: E402
    _euler_state_is_admissible,
    _fixed_scale_conservative_relative_l2,
    _frozen_split,
    _predict_model_batch_state,
    _saved_time_sha256,
    _state_metrics,
    _write_csv,
    _write_jsonl,
    error_budget_survival_rows,
    summarize_direct_curves,
    trajectory_increment_errors,
)
from scripts.time_dependent_no.evaluate_euler1d_resolution_transfer import (  # noqa: E402
    load_frozen_residual_checkpoint,
)
from scripts.time_dependent_no.train_euler1d_target_ladder import (  # noqa: E402
    json_ready,
    sha256_file,
)
from utility.time_dependent_no.euler1d_data import (  # noqa: E402
    Euler1DNPZ,
    load_euler1d_npz,
    primitive_to_conservative_np,
)

EPS = 1.0e-12
REQUIRED_STRIDES = (1, 2, 4, 8)
SAME_STATE_METRICS = (
    "current_truth_cons_scaled_rel_l2",
    "reference_truth_cons_scaled_rel_l2",
    "model_truth_cons_scaled_rel_l2",
    "model_reference_cons_scaled_rel_l2",
    "model_truth_shock_cons_scaled_rel_l2",
    "model_truth_smooth_cons_scaled_rel_l2",
    "model_reference_shock_cons_scaled_rel_l2",
    "model_reference_smooth_cons_scaled_rel_l2",
    "model_truth_front_position_mae",
    "model_reference_front_position_mae",
    "correction_toward_truth_cosine",
    "correction_norm_over_reference_defect",
    "model_reference_cons_scaled_rel_l2_per_physical_time",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        nargs=2,
        action="append",
        metavar=("STRIDE", "PATH"),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-frames", type=int, nargs="+", default=[0, 16, 32, 64])
    parser.add_argument("--rescue-frames", type=int, nargs="+", default=[32, 64])
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument(
        "--summary-horizons",
        type=int,
        nargs="+",
        default=[32, 64, 96],
    )
    parser.add_argument(
        "--error-budgets",
        type=float,
        nargs="+",
        default=[0.02, 0.05, 0.1],
    )
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--split-seed", type=int, default=20260707)
    parser.add_argument("--train-cases", type=int, default=384)
    parser.add_argument("--val-cases", type=int, default=64)
    parser.add_argument("--test-cases", type=int, default=64)
    parser.add_argument(
        "--case-count",
        type=int,
        default=0,
        help="Leading frozen-split cases to evaluate; zero uses the full split.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--solver-workers", type=int, default=8)
    parser.add_argument("--ghost-cells", type=int, default=3)
    parser.add_argument("--rho-floor", type=float, default=1.0e-10)
    parser.add_argument("--p-floor", type=float, default=1.0e-10)
    parser.add_argument("--shock-radius-cells", type=int, default=4)
    parser.add_argument("--truth-replay-tolerance", type=float, default=5.0e-5)
    parser.add_argument(
        "--truth-replay-mean-tolerance",
        type=float,
        default=1.0e-6,
    )
    parser.add_argument(
        "--truth-replay-p99-tolerance",
        type=float,
        default=1.0e-6,
    )
    return parser.parse_args(argv)


def require_fresh_output_dir(path: Path) -> None:
    """Refuse to mix a new diagnostic with stale artifacts from an older run."""

    if not path.exists():
        return
    if not path.is_dir():
        raise ValueError(f"--output-dir is not a directory: {path}")
    if any(path.iterdir()):
        raise FileExistsError(
            f"--output-dir must be absent or empty to prevent stale artifacts: {path}"
        )


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
        if stride in result:
            raise ValueError(f"duplicate checkpoint stride {stride}")
        result[stride] = Path(raw_path)
    if tuple(sorted(result)) != REQUIRED_STRIDES:
        raise ValueError(f"D033 requires frozen strides {REQUIRED_STRIDES}")
    return result


def advance_reference_macro(
    primitive: np.ndarray,
    left_state: np.ndarray,
    dx: float,
    interval_dts: Sequence[float],
    config: SolverReplayConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Replay consecutive saved intervals, preserving generator finalization."""

    dts = tuple(float(value) for value in interval_dts)
    if not dts or any(value <= 0.0 for value in dts):
        raise ValueError("reference macro requires positive saved-interval steps")
    conservative = primitive_to_conservative_np(
        np.asarray(primitive, dtype=np.float64),
        config.gamma,
    )
    totals = {"substeps": 0, "retry_halvings": 0, "fallback_steps": 0}
    current_primitive = np.asarray(primitive, dtype=np.float64)
    for interval_dt in dts:
        current_primitive, conservative, diagnostics = advance_reference_conservative(
            conservative,
            np.asarray(left_state, dtype=np.float64),
            float(dx),
            interval_dt,
            config,
        )
        for name in totals:
            totals[name] += int(diagnostics[name])
    totals["saved_intervals"] = len(dts)
    return current_primitive, conservative, totals


def _reference_batch(
    source: Euler1DNPZ,
    case_ids: np.ndarray,
    primitive: np.ndarray,
    start_frame: int,
    stride: int,
    config: SolverReplayConfig,
    executor: ThreadPoolExecutor,
) -> list[tuple[np.ndarray, np.ndarray, dict[str, int]]]:
    futures = []
    for position, case_id_value in enumerate(case_ids):
        case_id = int(case_id_value)
        interval_dts = np.diff(
            np.asarray(
                source.t[case_id, start_frame : start_frame + stride + 1],
                dtype=np.float64,
            )
        )
        dx = float(np.mean(np.diff(source.x[case_id].astype(np.float64))))
        futures.append(
            executor.submit(
                advance_reference_macro,
                primitive[position],
                source.left_states[case_id],
                dx,
                interval_dts,
                config,
            )
        )
    return [future.result() for future in futures]


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


def _truth_replay(
    source: Euler1DNPZ,
    case_ids: np.ndarray,
    start_frames: Sequence[int],
    strides: Sequence[int],
    config: SolverReplayConfig,
    executor: ThreadPoolExecutor,
) -> tuple[
    list[dict[str, Any]],
    dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray, dict[str, int]]],
]:
    rows: list[dict[str, Any]] = []
    cache: dict[
        tuple[int, int, int],
        tuple[np.ndarray, np.ndarray, dict[str, int]],
    ] = {}
    for stride in strides:
        for start_frame in start_frames:
            current = np.asarray(source.data[case_ids, start_frame], dtype=np.float64)
            reference = _reference_batch(
                source,
                case_ids,
                current,
                start_frame,
                stride,
                config,
                executor,
            )
            target_frame = start_frame + stride
            for position, case_id_value in enumerate(case_ids):
                case_id = int(case_id_value)
                reference_primitive, reference_conservative, diagnostics = reference[
                    position
                ]
                cache[(stride, start_frame, case_id)] = (
                    reference_primitive,
                    reference_conservative,
                    diagnostics,
                )
                truth = np.asarray(source.data[case_id, target_frame], dtype=np.float64)
                rows.append(
                    {
                        "case_id": case_id,
                        "stride": stride,
                        "start_frame": start_frame,
                        "target_frame": target_frame,
                        "fixed_scale_conservative_relative_l2": (
                            _fixed_scale_conservative_relative_l2(
                                reference_primitive,
                                truth,
                                source.gamma,
                            )
                        ),
                        **{
                            f"reference_{key}": value
                            for key, value in diagnostics.items()
                        },
                    }
                )
    return rows, cache


def _on_policy_state_bank(
    source: Euler1DNPZ,
    case_ids: np.ndarray,
    stride: int,
    model: torch.nn.Module,
    adapter: torch.nn.Module,
    max_frame: int,
    device: torch.device,
) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    current = np.asarray(source.data[case_ids, 0], dtype=np.float64).copy()
    current_conservative = primitive_to_conservative_np(current, source.gamma)
    active = np.ones(case_ids.size, dtype=bool)
    bank = {0: (current.copy(), current_conservative.copy(), active.copy())}
    for target_frame in range(stride, max_frame + 1, stride):
        positions = np.flatnonzero(active)
        if positions.size:
            proposed, proposed_conservative = _predict_model_batch_state(
                source,
                case_ids[positions],
                current[positions],
                target_frame - stride,
                stride,
                model,
                adapter,
                device,
                conservative=current_conservative[positions],
            )
            for local, position in enumerate(positions):
                candidate = proposed[local]
                if np.all(np.isfinite(candidate)) and _euler_state_is_admissible(
                    candidate
                ):
                    current[position] = candidate
                    current_conservative[position] = proposed_conservative[local]
                else:
                    active[position] = False
                    current[position] = np.nan
        bank[target_frame] = (
            current.copy(),
            current_conservative.copy(),
            active.copy(),
        )
    return bank


def _same_state_rows(
    source: Euler1DNPZ,
    case_ids: np.ndarray,
    models: dict[int, tuple[torch.nn.Module, torch.nn.Module]],
    start_frames: Sequence[int],
    truth_cache: dict[
        tuple[int, int, int],
        tuple[np.ndarray, np.ndarray, dict[str, int]],
    ],
    config: SolverReplayConfig,
    executor: ThreadPoolExecutor,
    device: torch.device,
    shock_radius_cells: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for stride, (model, adapter) in sorted(models.items()):
        bank = _on_policy_state_bank(
            source,
            case_ids,
            stride,
            model,
            adapter,
            max(start_frames),
            device,
        )
        for state_source in ("truth", "on_policy"):
            for start_frame in start_frames:
                if state_source == "truth":
                    current_all = np.asarray(
                        source.data[case_ids, start_frame], dtype=np.float64
                    )
                    current_conservative_all = primitive_to_conservative_np(
                        current_all, source.gamma
                    )
                    active = np.ones(case_ids.size, dtype=bool)
                else:
                    current_all, current_conservative_all, active = bank[start_frame]
                positions = np.flatnonzero(active)
                if not positions.size:
                    continue
                active_ids = case_ids[positions]
                current = current_all[positions]
                current_conservative_batch = current_conservative_all[positions]
                model_prediction, model_prediction_conservative = (
                    _predict_model_batch_state(
                        source,
                        active_ids,
                        current,
                        start_frame,
                        stride,
                        model,
                        adapter,
                        device,
                        conservative=current_conservative_batch,
                    )
                )
                if state_source == "truth":
                    reference = [
                        truth_cache[(stride, start_frame, int(case_id))]
                        for case_id in active_ids
                    ]
                else:
                    reference = _reference_batch(
                        source,
                        active_ids,
                        current,
                        start_frame,
                        stride,
                        config,
                        executor,
                    )
                target_frame = start_frame + stride
                physical_dt = float(
                    source.t[0, target_frame] - source.t[0, start_frame]
                )
                for local, case_id_value in enumerate(active_ids):
                    case_id = int(case_id_value)
                    prediction = np.asarray(model_prediction[local], dtype=np.float64)
                    finite = bool(np.all(np.isfinite(prediction)))
                    admissible = finite and _euler_state_is_admissible(prediction)
                    row: dict[str, Any] = {
                        "state_source": state_source,
                        "case_id": case_id,
                        "stride": stride,
                        "start_frame": start_frame,
                        "target_frame": target_frame,
                        "physical_dt": physical_dt,
                        "model_raw_state_finite": finite,
                        "model_raw_state_admissible": admissible,
                    }
                    if not finite:
                        rows.append(row)
                        continue
                    current_primitive = current[local]
                    truth_current = np.asarray(
                        source.data[case_id, start_frame], dtype=np.float64
                    )
                    truth_target = np.asarray(
                        source.data[case_id, target_frame], dtype=np.float64
                    )
                    reference_primitive, reference_conservative, diagnostics = (
                        reference[local]
                    )
                    current_conservative = current_conservative_batch[local]
                    truth_current_conservative = primitive_to_conservative_np(
                        truth_current, source.gamma
                    )
                    truth_target_conservative = primitive_to_conservative_np(
                        truth_target, source.gamma
                    )
                    prediction_conservative = model_prediction_conservative[local]
                    row.update(
                        state_error_metrics(
                            current_primitive,
                            current_conservative,
                            truth_current,
                            truth_current_conservative,
                            source.x[case_id],
                            source.gamma,
                            prefix="current_truth",
                            shock_radius_cells=shock_radius_cells,
                        )
                    )
                    row.update(
                        state_error_metrics(
                            reference_primitive,
                            reference_conservative,
                            truth_target,
                            truth_target_conservative,
                            source.x[case_id],
                            source.gamma,
                            prefix="reference_truth",
                            shock_radius_cells=shock_radius_cells,
                        )
                    )
                    row.update(
                        state_error_metrics(
                            prediction,
                            prediction_conservative,
                            truth_target,
                            truth_target_conservative,
                            source.x[case_id],
                            source.gamma,
                            prefix="model_truth",
                            shock_radius_cells=shock_radius_cells,
                        )
                    )
                    row.update(
                        state_error_metrics(
                            prediction,
                            prediction_conservative,
                            reference_primitive,
                            reference_conservative,
                            source.x[case_id],
                            source.gamma,
                            prefix="model_reference",
                            shock_radius_cells=shock_radius_cells,
                        )
                    )
                    row.update(
                        cancellation_metrics(
                            prediction_conservative,
                            reference_conservative,
                            truth_target_conservative,
                            source.gamma,
                        )
                    )
                    row["model_reference_cons_scaled_rel_l2_per_physical_time"] = (
                        row["model_reference_cons_scaled_rel_l2"] / physical_dt
                    )
                    row.update(
                        {
                            f"reference_{key}": value
                            for key, value in diagnostics.items()
                        }
                    )
                    rows.append(row)
    return rows


def _summarize_same_state(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    keys = sorted(
        {
            (
                str(row["state_source"]),
                int(row["stride"]),
                int(row["start_frame"]),
            )
            for row in rows
        }
    )
    for state_source, stride, start_frame in keys:
        group = [
            row
            for row in rows
            if row["state_source"] == state_source
            and int(row["stride"]) == stride
            and int(row["start_frame"]) == start_frame
        ]
        valid = [row for row in group if row["model_raw_state_admissible"]]
        summary: dict[str, Any] = {
            "state_source": state_source,
            "stride": stride,
            "start_frame": start_frame,
            "target_frame": start_frame + stride,
            "num_states": len(group),
            "num_raw_admissible": len(valid),
            "raw_admissibility_fraction": len(valid) / max(len(group), 1),
        }
        for name in SAME_STATE_METRICS:
            values = np.asarray(
                [float(row[name]) for row in valid if name in row],
                dtype=np.float64,
            )
            values = values[np.isfinite(values)]
            summary[f"{name}_mean"] = (
                float(np.mean(values)) if values.size else float("nan")
            )
        output.append(summary)
    return output


def truth_replay_gate(
    errors: Sequence[float],
    *,
    max_tolerance: float,
    mean_tolerance: float,
    p99_tolerance: float,
) -> dict[str, Any]:
    """Gate replay by its bulk distribution and serialization-scale maximum."""

    values = np.asarray(errors, dtype=np.float64)
    if (
        not values.size
        or not np.all(np.isfinite(values))
        or min(max_tolerance, mean_tolerance, p99_tolerance) <= 0.0
    ):
        raise ValueError("truth replay gate requires finite errors and tolerances")
    mean = float(np.mean(values))
    p99 = float(np.quantile(values, 0.99))
    maximum = float(np.max(values))
    return {
        "mean": mean,
        "p99": p99,
        "max": maximum,
        "mean_tolerance": mean_tolerance,
        "p99_tolerance": p99_tolerance,
        "max_tolerance": max_tolerance,
        "passed": (
            mean <= mean_tolerance and p99 <= p99_tolerance and maximum <= max_tolerance
        ),
    }


def _rollout_variant(
    source: Euler1DNPZ,
    case_ids: np.ndarray,
    stride: int,
    model: torch.nn.Module,
    adapter: torch.nn.Module,
    horizon: int,
    rescue_frame: int | None,
    config: SolverReplayConfig,
    executor: ThreadPoolExecutor,
    device: torch.device,
) -> list[dict[str, Any]]:
    path = (
        f"s{stride}_raw"
        if rescue_frame is None
        else f"s{stride}_oracle_rescue_f{rescue_frame}"
    )
    current = np.asarray(source.data[case_ids, 0], dtype=np.float64).copy()
    current_conservative = primitive_to_conservative_np(current, source.gamma)
    active = np.ones(case_ids.size, dtype=bool)
    cumulative_integral = np.zeros(case_ids.size, dtype=np.float64)
    previous_error = np.zeros(case_ids.size, dtype=np.float64)
    previous_time = np.asarray(source.t[case_ids, 0], dtype=np.float64)
    rows: list[dict[str, Any]] = []
    metric_names = (
        "primitive_relative_l2",
        "fixed_scale_conservative_relative_l2",
        "smooth_region_relative_l2",
        "boundary_relative_l2",
        "shock_top2_position_mae",
        "shock_top2_strength_relative_l1",
        "conservative_budget_relative_l2",
        "min_density",
        "min_pressure",
        "trajectory_increment_relative_l2",
        "trajectory_increment_state_normalized_l2",
        "time_integrated_fixed_scale_conservative_relative_l2",
        "time_mean_fixed_scale_conservative_relative_l2",
    )
    for target_frame in range(stride, horizon + 1, stride):
        positions = np.flatnonzero(active)
        if not positions.size:
            break
        active_ids = case_ids[positions]
        start_frame = target_frame - stride
        reference_diagnostics: list[dict[str, int] | None]
        if rescue_frame is not None and start_frame == rescue_frame:
            reference = _reference_batch(
                source,
                active_ids,
                current[positions],
                start_frame,
                stride,
                config,
                executor,
            )
            proposed = np.stack([item[0] for item in reference])
            proposed_conservative = np.stack([item[1] for item in reference])
            reference_diagnostics = [item[2] for item in reference]
            call_kind = "oracle_reference_macro"
        else:
            proposed, proposed_conservative = _predict_model_batch_state(
                source,
                active_ids,
                current[positions],
                start_frame,
                stride,
                model,
                adapter,
                device,
                conservative=current_conservative[positions],
            )
            reference_diagnostics = [None] * positions.size
            call_kind = "learned_map"
        for local, position in enumerate(positions):
            case_id = int(case_ids[position])
            prediction = np.asarray(proposed[local], dtype=np.float64)
            finite = bool(np.all(np.isfinite(prediction)))
            admissible = finite and _euler_state_is_admissible(prediction)
            physical_time = float(source.t[case_id, target_frame])
            row: dict[str, Any] = {
                "checkpoint_set_label": "primary_selected",
                "path": path,
                "case_id": case_id,
                "stride": stride,
                "call_index": target_frame // stride,
                "start_frame": start_frame,
                "frame": target_frame,
                "physical_time": physical_time,
                "physical_elapsed_time": (physical_time - float(source.t[case_id, 0])),
                "call_kind": call_kind,
                "raw_state_finite": finite,
                "raw_state_admissible": admissible,
                "completed_step": admissible,
                "termination_reason": (
                    None
                    if admissible
                    else (
                        "nonfinite_raw_state" if not finite else "nonpositive_raw_state"
                    )
                ),
            }
            diagnostics = reference_diagnostics[local]
            if diagnostics is not None:
                row.update(
                    {
                        f"rescue_reference_{key}": value
                        for key, value in diagnostics.items()
                    }
                )
            if not admissible:
                row.update({name: float("nan") for name in metric_names})
                active[position] = False
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
                    current[position],
                    prediction,
                    truth_previous,
                    truth_current,
                    source.gamma,
                )
            )
            elapsed = physical_time - float(previous_time[position])
            cumulative_integral[position] += (
                0.5
                * elapsed
                * (
                    previous_error[position]
                    + metrics["fixed_scale_conservative_relative_l2"]
                )
            )
            metrics["time_integrated_fixed_scale_conservative_relative_l2"] = float(
                cumulative_integral[position]
            )
            metrics["time_mean_fixed_scale_conservative_relative_l2"] = float(
                cumulative_integral[position] / max(row["physical_elapsed_time"], EPS)
            )
            row.update(metrics)
            rows.append(row)
            current[position] = prediction
            current_conservative[position] = proposed_conservative[local]
            previous_error[position] = metrics["fixed_scale_conservative_relative_l2"]
            previous_time[position] = physical_time
    return rows


def _paired_rescue_comparisons(
    rows: list[dict[str, Any]],
    *,
    horizon: int,
    num_cases: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for stride in sorted({int(row["stride"]) for row in rows}):
        stride_rows = [row for row in rows if int(row["stride"]) == stride]
        raw = {
            int(row["case_id"]): row
            for row in stride_rows
            if row["path"] == f"s{stride}_raw"
            and int(row["frame"]) == horizon
            and row["completed_step"]
        }
        rescue_paths = sorted(
            {
                str(row["path"])
                for row in stride_rows
                if "oracle_rescue" in str(row["path"])
            }
        )
        for path in rescue_paths:
            rescue = {
                int(row["case_id"]): row
                for row in stride_rows
                if row["path"] == path
                and int(row["frame"]) == horizon
                and row["completed_step"]
            }
            paired = sorted(set(raw).intersection(rescue))
            raw_errors = np.asarray(
                [raw[case]["fixed_scale_conservative_relative_l2"] for case in paired]
            )
            rescue_errors = np.asarray(
                [
                    rescue[case]["fixed_scale_conservative_relative_l2"]
                    for case in paired
                ]
            )
            ratios = rescue_errors / np.maximum(raw_errors, EPS)
            output.append(
                {
                    "stride": stride,
                    "horizon": horizon,
                    "raw_path": f"s{stride}_raw",
                    "rescue_path": path,
                    "num_cases": num_cases,
                    "raw_num_completed": len(raw),
                    "rescue_num_completed": len(rescue),
                    "completion_fraction_delta": (
                        (len(rescue) - len(raw)) / max(num_cases, 1)
                    ),
                    "num_paired_completed": len(paired),
                    "paired_rescue_to_raw_error_ratio_mean": (
                        float(np.mean(ratios)) if ratios.size else float("nan")
                    ),
                    "paired_rescue_to_raw_error_ratio_median": (
                        float(np.median(ratios)) if ratios.size else float("nan")
                    ),
                    "paired_rescue_improvement_fraction": (
                        float(np.mean(rescue_errors < raw_errors))
                        if ratios.size
                        else float("nan")
                    ),
                }
            )
    return output


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    if args.torch_threads < 1 or args.solver_workers < 1:
        raise ValueError("thread counts must be positive")
    if args.ghost_cells < 3:
        raise ValueError("--ghost-cells must be at least 3")
    if (
        min(
            args.truth_replay_tolerance,
            args.truth_replay_mean_tolerance,
            args.truth_replay_p99_tolerance,
        )
        <= 0.0
    ):
        raise ValueError("truth replay tolerances must be positive")
    require_fresh_output_dir(args.output_dir)
    torch.set_num_threads(args.torch_threads)
    device = _select_device(args.device)
    source = load_euler1d_npz(args.data_path)
    source.validate()
    if not np.array_equal(source.t, np.broadcast_to(source.t[:1], source.t.shape)):
        raise ValueError("D033 requires common saved times")
    checkpoint_paths = _checkpoint_paths(args.checkpoint)
    declared_frames = (
        *args.start_frames,
        *args.rescue_frames,
        *args.summary_horizons,
    )
    for value in declared_frames:
        if value < 0 or value > args.horizon or value % 8:
            raise ValueError(
                "all declared frames must be nonnegative multiples of eight"
            )
    if any(value >= args.horizon for value in args.rescue_frames):
        raise ValueError("rescue frames must be strictly earlier than --horizon")
    if args.horizon >= source.num_frames or args.horizon % 8:
        raise ValueError("--horizon must be a valid saved-frame multiple of eight")
    if max(args.start_frames) + max(REQUIRED_STRIDES) > args.horizon:
        raise ValueError("same-state target exceeds the declared horizon")
    expected_cases = _frozen_split(
        source,
        split_seed=args.split_seed,
        train_cases=args.train_cases,
        val_cases=args.val_cases,
        test_cases=args.test_cases,
        split=args.split,
    )
    data_sha256 = sha256_file(args.data_path)
    saved_time_sha256 = _saved_time_sha256(source)
    models: dict[int, tuple[torch.nn.Module, torch.nn.Module]] = {}
    checkpoint_contract: dict[str, Any] = {}
    for stride, path in checkpoint_paths.items():
        model, adapter, _normalizer, checkpoint = load_frozen_residual_checkpoint(
            path, device
        )
        frozen = _validate_checkpoint_contract(
            checkpoint,
            stride=stride,
            expected_cases=expected_cases,
            data_sha256=data_sha256,
            saved_time_sha256=saved_time_sha256,
            split=args.split,
        )
        models[stride] = (model, adapter)
        checkpoint_contract[str(stride)] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "best_epoch": checkpoint.get("best_epoch"),
            "frozen_coordinates": frozen,
        }
    if args.case_count < 0 or args.case_count > expected_cases.size:
        raise ValueError("--case-count must be zero or fit within the frozen split")
    if args.case_count:
        expected_cases = expected_cases[: args.case_count]
    config = solver_config_from_source(
        source,
        ng=args.ghost_cells,
        rho_floor=args.rho_floor,
        p_floor=args.p_floor,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    contract = {
        "experiment": "D033_large_step_same_state_solver_consistency",
        "data_path": str(args.data_path),
        "data_sha256": data_sha256,
        "saved_time_sha256": saved_time_sha256,
        "case_ids": expected_cases.tolist(),
        "split": args.split,
        "strides": list(REQUIRED_STRIDES),
        "start_frames": sorted(set(args.start_frames)),
        "rescue_frames": sorted(set(args.rescue_frames)),
        "horizon": args.horizon,
        "summary_horizons": sorted(set(args.summary_horizons)),
        "error_budgets": sorted(set(args.error_budgets)),
        "truth_replay_tolerance": {
            "max": args.truth_replay_tolerance,
            "mean": args.truth_replay_mean_tolerance,
            "p99": args.truth_replay_p99_tolerance,
        },
        "truth_replay_metric": "fixed_scale_conservative_relative_l2",
        "same_state_definition": (
            "identical_raw_primitive_state_and_case_metadata_at_call_start"
        ),
        "raw_inference": "no_clipping_no_projection_no_learned_state_floor",
        "oracle_rescue_role": "diagnostic_one_call_reference_replacement_only",
        "reference_macro": ("documented_solver_replayed_one_saved_interval_at_a_time"),
        "solver_replay": config.__dict__,
        "solver_workers": args.solver_workers,
        "device": str(device),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "checkpoints": checkpoint_contract,
    }
    with ThreadPoolExecutor(max_workers=args.solver_workers) as executor:
        truth_rows, truth_cache = _truth_replay(
            source,
            expected_cases,
            sorted(set(args.start_frames)),
            REQUIRED_STRIDES,
            config,
            executor,
        )
        _write_csv(args.output_dir / "truth_replay.csv", truth_rows)
        replay_gate = truth_replay_gate(
            [float(row["fixed_scale_conservative_relative_l2"]) for row in truth_rows],
            max_tolerance=args.truth_replay_tolerance,
            mean_tolerance=args.truth_replay_mean_tolerance,
            p99_tolerance=args.truth_replay_p99_tolerance,
        )
        contract["truth_replay_gate"] = replay_gate
        contract["truth_replay_max_error"] = replay_gate["max"]
        contract["truth_replay_gate_passed"] = replay_gate["passed"]
        if not contract["truth_replay_gate_passed"]:
            report = {
                "contract": contract,
                "status": "truth_replay_gate_failed",
                "runtime_seconds": time.perf_counter() - started,
            }
            (args.output_dir / "report.json").write_text(
                json.dumps(json_ready(report), indent=2, sort_keys=True),
                encoding="utf-8",
            )
            return report
        same_state = _same_state_rows(
            source,
            expected_cases,
            models,
            sorted(set(args.start_frames)),
            truth_cache,
            config,
            executor,
            device,
            args.shock_radius_cells,
        )
        rescue_rows: list[dict[str, Any]] = []
        for stride, (model, adapter) in sorted(models.items()):
            rescue_rows.extend(
                _rollout_variant(
                    source,
                    expected_cases,
                    stride,
                    model,
                    adapter,
                    args.horizon,
                    None,
                    config,
                    executor,
                    device,
                )
            )
            for rescue_frame in sorted(set(args.rescue_frames)):
                rescue_rows.extend(
                    _rollout_variant(
                        source,
                        expected_cases,
                        stride,
                        model,
                        adapter,
                        args.horizon,
                        rescue_frame,
                        config,
                        executor,
                        device,
                    )
                )

    same_state_summary = _summarize_same_state(same_state)
    rescue_summary = summarize_direct_curves(
        rescue_rows,
        args.summary_horizons,
        num_cases=expected_cases.size,
    )
    rescue_survival = error_budget_survival_rows(
        rescue_rows,
        args.error_budgets,
        max_horizon=args.horizon,
    )
    rescue_comparisons = _paired_rescue_comparisons(
        rescue_rows,
        horizon=args.horizon,
        num_cases=expected_cases.size,
    )
    report = {
        "contract": contract,
        "status": "complete",
        "same_state_summary": same_state_summary,
        "rescue_summary": rescue_summary,
        "rescue_comparisons": rescue_comparisons,
        "runtime_seconds": time.perf_counter() - started,
    }
    _write_csv(args.output_dir / "same_state.csv", same_state)
    _write_csv(args.output_dir / "same_state_summary.csv", same_state_summary)
    _write_jsonl(args.output_dir / "rescue_curves.jsonl", rescue_rows)
    _write_csv(args.output_dir / "rescue_summary.csv", rescue_summary)
    _write_csv(args.output_dir / "rescue_survival.csv", rescue_survival)
    _write_csv(args.output_dir / "rescue_comparisons.csv", rescue_comparisons)
    (args.output_dir / "report.json").write_text(
        json.dumps(json_ready(report), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        json.dumps(
            json_ready(
                {
                    "truth_replay_max_error": contract["truth_replay_max_error"],
                    "rescue_comparisons": rescue_comparisons,
                }
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return report


def main(argv: Sequence[str] | None = None) -> None:
    report = run(parse_args(argv))
    if report.get("status") != "complete":
        raise SystemExit(f"diagnostic failed with status={report.get('status')}")


if __name__ == "__main__":
    main()
