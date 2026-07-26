"""Benchmark learned Euler flow maps against the documented reference solver.

The reference side replays the dataset's WENO5--HLLC--ADER2 configuration on
the same frozen cases at a small, predeclared resolution ladder.  Learned and
reference rows are joined only at identical physical endpoints.  An
accuracy-matched comparison selects the fastest reference row whose error is
no worse than the learned path's error on the same cases.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch

if __package__:
    from scripts.time_dependent_no.euler1d_weno_hllc_ader_dataset import (
        CaseConfig,
        integrate_case,
    )
    from scripts.time_dependent_no.train_euler1d_target_ladder import (
        json_ready,
        pressure_front_top2_metrics_np,
        sha256_file,
    )
    from scripts.time_dependent_no.evaluate_euler1d_flow_map_frontier import (
        _make_model_batch,
        _select_device,
        _synchronize,
        load_frozen_residual_checkpoint,
    )
else:
    from euler1d_weno_hllc_ader_dataset import CaseConfig, integrate_case
    from train_euler1d_target_ladder import (
        json_ready,
        pressure_front_top2_metrics_np,
        sha256_file,
    )
    from evaluate_euler1d_flow_map_frontier import (
        _make_model_batch,
        _select_device,
        _synchronize,
        load_frozen_residual_checkpoint,
    )
from utility.time_dependent_no.euler1d_data import (
    Euler1DNPZ,
    load_euler1d_npz,
    primitive_to_conservative_np,
)

EPS = 1.0e-12
PHYSICAL_SCALES = np.array([1.0, 1.0, 2.5], dtype=np.float64)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--horizons", type=int, nargs="+", default=[32, 64, 96])
    parser.add_argument(
        "--resolutions",
        type=int,
        nargs="+",
        default=[32, 64, 128, 256],
    )
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--split-seed", type=int, default=20260707)
    parser.add_argument("--train-cases", type=int, default=384)
    parser.add_argument("--val-cases", type=int, default=64)
    parser.add_argument("--test-cases", type=int, default=64)
    parser.add_argument("--case-count", type=int, default=16)
    parser.add_argument("--warmup-cases", type=int, default=1)
    parser.add_argument(
        "--initialization-mode",
        choices=("cell_center", "exact_cell_average"),
        default="cell_center",
    )
    parser.add_argument("--learned-paths-jsonl", type=Path)
    parser.add_argument("--learned-timing-csv", type=Path)
    parser.add_argument(
        "--checkpoint",
        nargs=2,
        action="append",
        metavar=("STRIDE", "PATH"),
        default=[],
        help="Frozen stride and checkpoint for direct full-rollout timing.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--learned-warmup", type=int, default=10)
    parser.add_argument("--learned-repeats", type=int, default=50)
    parser.add_argument("--throughput-batch-size", type=int, default=64)
    parser.add_argument("--reference-workers", type=int, default=4)
    parser.add_argument("--reference-throughput-warmup", type=int, default=1)
    parser.add_argument("--reference-throughput-repeats", type=int, default=3)
    parser.add_argument(
        "--allow-uncontrolled-reference-threads",
        action="store_true",
        help="Allow reference timing without OMP_NUM_THREADS=MKL_NUM_THREADS=1.",
    )
    return parser.parse_args(argv)


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


def prolong_piecewise_constant(
    primitive: np.ndarray,
    target_cells: int,
) -> np.ndarray:
    """Prolong uniform coarse cell averages without interpolation smoothing."""

    values = np.asarray(primitive)
    source_cells = int(values.shape[-2])
    if source_cells < 1 or target_cells % source_cells:
        raise ValueError("target resolution must be divisible by source resolution")
    return np.repeat(values, target_cells // source_cells, axis=-2)


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
    exact = primitive_to_conservative_np(truth, gamma) / PHYSICAL_SCALES
    return _relative_l2(pred, exact)


def _checkpoint_paths(entries: Sequence[Sequence[str]]) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for raw_stride, raw_path in entries:
        stride = int(raw_stride)
        if stride < 1:
            raise ValueError("checkpoint strides must be positive")
        if stride in result:
            raise ValueError(f"duplicate checkpoint stride {stride}")
        result[stride] = Path(raw_path)
    return result


def _rollout_callbacks(
    source: Euler1DNPZ,
    case_ids: np.ndarray,
    stride: int,
    horizon: int,
    model: torch.nn.Module,
    adapter: torch.nn.Module,
    device: torch.device,
) -> dict[str, Callable[[], tuple[torch.Tensor, torch.Tensor] | np.ndarray]]:
    """Build device-resident and host-to-host full-rollout callbacks."""

    ids = np.asarray(case_ids, dtype=np.int64)
    if horizon % stride:
        raise ValueError("rollout horizon must be divisible by stride")
    initial_numpy = np.ascontiguousarray(source.data[ids, 0], dtype=np.float32)
    device_base = _make_model_batch(
        source,
        ids,
        initial_numpy,
        0,
        stride,
        device,
    )
    device_initial_primitive = device_base.current_primitive
    device_initial_conservative = device_base.current_conservative
    device_dts = [
        torch.from_numpy(
            np.ascontiguousarray(
                source.t[ids, frame + stride] - source.t[ids, frame],
                dtype=np.float32,
            )
        ).to(device)
        for frame in range(0, horizon, stride)
    ]

    def run_device_resident() -> tuple[torch.Tensor, torch.Tensor]:
        current_primitive = device_initial_primitive
        current_conservative = device_initial_conservative
        with torch.inference_mode():
            for dt in device_dts:
                batch = replace(
                    device_base,
                    current_primitive=current_primitive,
                    current_conservative_state=current_conservative,
                    dt=dt,
                )
                decoded = adapter(model(batch), batch)
                current_primitive = decoded.primitive
                current_conservative = decoded.conservative
        return current_primitive, current_conservative

    def run_host_to_host() -> np.ndarray:
        base = _make_model_batch(
            source,
            ids,
            initial_numpy,
            0,
            stride,
            device,
        )
        current_primitive = base.current_primitive
        current_conservative = base.current_conservative
        dts = [
            torch.from_numpy(
                np.ascontiguousarray(
                    source.t[ids, frame + stride] - source.t[ids, frame],
                    dtype=np.float32,
                )
            ).to(device)
            for frame in range(0, horizon, stride)
        ]
        with torch.inference_mode():
            for dt in dts:
                batch = replace(
                    base,
                    current_primitive=current_primitive,
                    current_conservative_state=current_conservative,
                    dt=dt,
                )
                decoded = adapter(model(batch), batch)
                current_primitive = decoded.primitive
                current_conservative = decoded.conservative
        return current_primitive.detach().cpu().numpy()

    return {
        "device_resident": run_device_resident,
        "host_to_host": run_host_to_host,
    }


def measure_synchronized_rollout(
    callback: Callable[[], Any],
    *,
    warmup: int,
    repeats: int,
    device: torch.device,
) -> dict[str, float]:
    """Time an entire fixed-call rollout with synchronization at its boundary."""

    if warmup < 0 or repeats < 1:
        raise ValueError("rollout timing needs nonnegative warmup and positive repeats")
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
        "full_rollout_median_ms": float(np.median(elapsed_ms)),
        "full_rollout_p95_ms": float(np.quantile(elapsed_ms, 0.95)),
        "full_rollout_min_ms": float(np.min(elapsed_ms)),
    }


def benchmark_learned_full_rollouts(
    source: Euler1DNPZ,
    case_ids: np.ndarray,
    models: dict[int, tuple[torch.nn.Module, torch.nn.Module]],
    horizons: Sequence[int],
    device: torch.device,
    *,
    warmup: int,
    repeats: int,
    throughput_batch_size: int,
) -> list[dict[str, Any]]:
    """Directly measure complete recurrent trajectories at fixed call counts."""

    batch_sizes = sorted(
        {
            1,
            min(int(throughput_batch_size), int(np.asarray(case_ids).size)),
        }
    )
    rows: list[dict[str, Any]] = []
    for stride, (model, adapter) in sorted(models.items()):
        model.eval()
        for horizon in sorted(set(int(value) for value in horizons)):
            if horizon % stride:
                raise ValueError(
                    f"timing horizon {horizon} is not divisible by stride {stride}"
                )
            for batch_size in batch_sizes:
                selected = np.asarray(case_ids[:batch_size], dtype=np.int64)
                callbacks = _rollout_callbacks(
                    source,
                    selected,
                    stride,
                    horizon,
                    model,
                    adapter,
                    device,
                )
                for boundary, callback in callbacks.items():
                    timing = measure_synchronized_rollout(
                        callback,
                        warmup=warmup,
                        repeats=repeats,
                        device=device,
                    )
                    endpoint = callback()
                    _synchronize(device)
                    primitive = (
                        endpoint[0].detach().cpu().numpy()
                        if isinstance(endpoint, tuple)
                        else np.asarray(endpoint)
                    )
                    admissible = np.all(np.isfinite(primitive), axis=(1, 2))
                    admissible &= np.all(primitive[..., 0] > 0.0, axis=1)
                    admissible &= np.all(primitive[..., 2] > 0.0, axis=1)
                    median_seconds = timing["full_rollout_median_ms"] / 1000.0
                    rows.append(
                        {
                            "stride": stride,
                            "horizon": horizon,
                            "num_calls": horizon // stride,
                            "batch_size": batch_size,
                            "measurement_boundary": boundary,
                            "measurement_scope": "direct_full_rollout",
                            "warmup": warmup,
                            "repeats": repeats,
                            "fixed_call_count": True,
                            "raw_conservative_recurrence": True,
                            "endpoint_admissibility_fraction": float(
                                np.mean(admissible)
                            ),
                            "cases_per_second": (
                                batch_size / median_seconds
                                if median_seconds > 0.0
                                else float("nan")
                            ),
                            "amortized_median_ms_per_case": (
                                timing["full_rollout_median_ms"] / batch_size
                            ),
                            **timing,
                        }
                    )
    return rows


def _scalar(metadata: dict[str, Any], name: str) -> Any:
    if name not in metadata:
        raise KeyError(f"dataset metadata is missing {name}")
    return np.asarray(metadata[name]).item()


def _case_config(
    source: Euler1DNPZ,
    case_id: int,
    t_final: float,
) -> CaseConfig:
    domains = np.asarray(source.metadata["domains"], dtype=np.float64)
    discontinuities = np.asarray(source.metadata["x_disc"], dtype=np.float64)
    return CaseConfig(
        x_left=float(domains[case_id, 0]),
        x_right=float(domains[case_id, 1]),
        x_disc=float(discontinuities[case_id]),
        left_state=np.asarray(source.left_states[case_id], dtype=np.float64),
        right_state=np.asarray(source.right_states[case_id], dtype=np.float64),
        t_final=t_final,
    )


def _run_reference_prediction(
    source: Euler1DNPZ,
    case_id: int,
    horizon: int,
    resolution: int,
    initialization_mode: str,
) -> tuple[float, np.ndarray, int]:
    physical_horizon = float(source.t[case_id, horizon] - source.t[case_id, 0])
    case = _case_config(source, case_id, physical_horizon)
    _x, _t, snapshots, fallback_count = integrate_case(
        case,
        nx=resolution,
        n_steps=horizon,
        gamma=source.gamma,
        cfl=float(_scalar(source.metadata, "cfl")),
        ng=3,
        rho_floor=1.0e-12,
        p_floor=1.0e-12,
        use_shock_flattening=bool(_scalar(source.metadata, "use_shock_flattening")),
        use_hlle_on_troubled_faces=bool(
            _scalar(source.metadata, "use_hlle_on_troubled_faces")
        ),
        shock_sensor_threshold=float(
            _scalar(source.metadata, "shock_sensor_threshold")
        ),
        shock_flatten_radius=int(_scalar(source.metadata, "shock_flatten_radius")),
        return_face_flux_integral=False,
        initialization_mode=initialization_mode,
        storage_dtype=str(source.data.dtype),
    )
    prediction = prolong_piecewise_constant(snapshots[-1], source.num_cells)
    return physical_horizon, prediction, int(fallback_count)


def _run_reference_case(
    source: Euler1DNPZ,
    case_id: int,
    horizon: int,
    resolution: int,
    initialization_mode: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    physical_horizon, prediction, fallback_count = _run_reference_prediction(
        source,
        case_id,
        horizon,
        resolution,
        initialization_mode,
    )
    wall_seconds = time.perf_counter() - started
    truth = np.asarray(source.data[case_id, horizon], dtype=np.float64)
    front = pressure_front_top2_metrics_np(
        prediction[None],
        truth[None],
        np.asarray(source.x[case_id], dtype=np.float64),
        min_separation_cells=8,
    )
    return {
        "case_id": case_id,
        "horizon": horizon,
        "physical_horizon": physical_horizon,
        "resolution": resolution,
        "wall_seconds": wall_seconds,
        "primitive_relative_l2": _relative_l2(prediction, truth),
        "fixed_scale_conservative_relative_l2": (
            _fixed_scale_conservative_relative_l2(
                prediction,
                truth,
                source.gamma,
            )
        ),
        "shock_top2_position_mae": float(front["position_assignment_mae"][0]),
        "min_density": float(np.min(prediction[:, 0])),
        "min_pressure": float(np.min(prediction[:, 2])),
        "fallback_count": fallback_count,
        "completed": True,
        "termination_reason": None,
    }


def summarize_reference_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(
            (int(row["resolution"]), int(row["horizon"])),
            [],
        ).append(row)
    summary: list[dict[str, Any]] = []
    for (resolution, horizon), group in sorted(groups.items()):
        complete = [row for row in group if row["completed"]]
        errors = np.asarray(
            [row["fixed_scale_conservative_relative_l2"] for row in complete],
            dtype=np.float64,
        )
        walls = np.asarray(
            [row["wall_seconds"] for row in complete],
            dtype=np.float64,
        )
        summary.append(
            {
                "resolution": resolution,
                "horizon": horizon,
                "physical_horizon": float(group[0]["physical_horizon"]),
                "num_cases": len(group),
                "completion_fraction": len(complete) / len(group),
                "fixed_scale_conservative_relative_l2_mean": (
                    float(np.mean(errors)) if errors.size else float("nan")
                ),
                "fixed_scale_conservative_relative_l2_median": (
                    float(np.median(errors)) if errors.size else float("nan")
                ),
                "fixed_scale_conservative_relative_l2_p95": (
                    float(np.quantile(errors, 0.95)) if errors.size else float("nan")
                ),
                "batch1_wall_seconds_median": (
                    float(np.median(walls)) if walls.size else float("nan")
                ),
                "batch1_wall_seconds_p95": (
                    float(np.quantile(walls, 0.95)) if walls.size else float("nan")
                ),
                "sequential_cases_per_second": (
                    1.0 / float(np.median(walls))
                    if walls.size and np.median(walls) > 0.0
                    else float("nan")
                ),
                "fallback_count_total": int(
                    sum(row["fallback_count"] for row in complete)
                ),
            }
        )
    return summary


def benchmark_reference_throughput(
    source: Euler1DNPZ,
    case_ids: np.ndarray,
    horizons: Sequence[int],
    resolutions: Sequence[int],
    initialization_mode: str,
    *,
    workers: int,
    warmup: int,
    repeats: int,
) -> list[dict[str, Any]]:
    """Measure concurrent independent reference solves on the same CPU."""

    if workers < 1 or warmup < 0 or repeats < 1:
        raise ValueError("invalid reference throughput timing counts")
    ids = np.asarray(case_ids, dtype=np.int64)
    rows: list[dict[str, Any]] = []
    for resolution in sorted(set(int(value) for value in resolutions)):
        for horizon in sorted(set(int(value) for value in horizons)):
            with ThreadPoolExecutor(max_workers=workers) as executor:

                def run_batch() -> None:
                    futures = [
                        executor.submit(
                            _run_reference_prediction,
                            source,
                            int(case_id),
                            horizon,
                            resolution,
                            initialization_mode,
                        )
                        for case_id in ids
                    ]
                    for future in futures:
                        future.result()

                for _ in range(warmup):
                    run_batch()
                elapsed = np.empty(repeats, dtype=np.float64)
                for repeat in range(repeats):
                    started = time.perf_counter()
                    run_batch()
                    elapsed[repeat] = time.perf_counter() - started
            median = float(np.median(elapsed))
            rows.append(
                {
                    "resolution": resolution,
                    "horizon": horizon,
                    "batch_size": int(ids.size),
                    "workers": workers,
                    "warmup": warmup,
                    "repeats": repeats,
                    "batch_wall_seconds_median": median,
                    "batch_wall_seconds_p95": float(np.quantile(elapsed, 0.95)),
                    "cases_per_second": (
                        float(ids.size) / median if median > 0.0 else float("nan")
                    ),
                    "measurement_scope": (
                        "concurrent_reference_preprocessing_integration_and_"
                        "final_prolonged_prediction"
                    ),
                }
            )
    return rows


def select_accuracy_matched_reference(
    learned_error: float,
    candidates: Sequence[dict[str, Any]],
) -> dict[str, Any] | None:
    """Return the fastest complete reference row with no larger mean error."""

    eligible = [
        row
        for row in candidates
        if float(row["completion_fraction"]) == 1.0
        and np.isfinite(row["fixed_scale_conservative_relative_l2_mean"])
        and float(row["fixed_scale_conservative_relative_l2_mean"]) <= learned_error
    ]
    if not eligible:
        return None
    return min(eligible, key=lambda row: float(row["batch1_wall_seconds_median"]))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_accuracy_matched_rows(
    learned_path_rows: list[dict[str, Any]],
    learned_timing_rows: list[dict[str, Any]],
    reference_summary: list[dict[str, Any]],
    case_ids: np.ndarray,
    reference_throughput_rows: Sequence[dict[str, Any]] = (),
) -> list[dict[str, Any]]:
    selected_case_list = [int(value) for value in np.asarray(case_ids).reshape(-1)]
    selected_cases = set(selected_case_list)
    if not selected_cases:
        raise ValueError("accuracy matching requires at least one selected case")
    if len(selected_cases) != len(selected_case_list):
        raise ValueError("accuracy-matching case IDs must be unique")
    legacy_timing = {
        int(row["stride"]): row
        for row in learned_timing_rows
        if int(row["batch_size"]) == 1 and "measurement_boundary" not in row
    }
    full_timing = {
        (
            int(row["stride"]),
            int(row["horizon"]),
            str(row["measurement_boundary"]),
        ): row
        for row in learned_timing_rows
        if int(row["batch_size"]) == 1
        and row.get("measurement_scope") == "direct_full_rollout"
    }
    throughput_timing = {
        (
            int(row["stride"]),
            int(row["horizon"]),
            str(row["measurement_boundary"]),
        ): row
        for row in learned_timing_rows
        if int(row["batch_size"]) > 1
        and row.get("measurement_scope") == "direct_full_rollout"
    }
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in learned_path_rows:
        if int(row["case_id"]) in selected_cases:
            groups.setdefault(
                (str(row["path"]), int(row["horizon"])),
                [],
            ).append(row)
    output: list[dict[str, Any]] = []
    for (path, horizon), group in sorted(groups.items()):
        group_case_ids = [int(row["case_id"]) for row in group]
        if len(set(group_case_ids)) != len(group_case_ids):
            raise ValueError(
                f"duplicate learned path rows for path={path}, horizon={horizon}"
            )
        case_coverage_complete = set(group_case_ids) == selected_cases
        complete = [row for row in group if row["completed_horizon"]]
        learned_error = (
            float(
                np.mean(
                    [row["fixed_scale_conservative_relative_l2"] for row in complete]
                )
            )
            if complete
            else float("nan")
        )
        call_stride = int(group[0]["call_stride"])
        num_calls = int(group[0]["num_calls"])
        device_row = full_timing.get((call_stride, horizon, "device_resident"))
        host_row = full_timing.get((call_stride, horizon, "host_to_host"))
        legacy_row = legacy_timing.get(call_stride)
        direct_timing_available = device_row is not None and host_row is not None
        candidates = [
            row for row in reference_summary if int(row["horizon"]) == horizon
        ]
        completion_fraction = len(complete) / len(selected_cases)
        error_is_claim_comparable = (
            case_coverage_complete and completion_fraction == 1.0
        )
        matched = (
            select_accuracy_matched_reference(learned_error, candidates)
            if error_is_claim_comparable and np.isfinite(learned_error)
            else None
        )
        if direct_timing_available:
            assert device_row is not None and host_row is not None
            device_wall = float(device_row["full_rollout_median_ms"]) / 1000.0
            host_wall = float(host_row["full_rollout_median_ms"]) / 1000.0
            timing_basis = "direct_synchronized_full_rollout"
        else:
            device_wall = (
                num_calls * float(legacy_row["device_resident_median_ms"]) / 1000.0
                if legacy_row is not None
                else float("nan")
            )
            host_wall = (
                num_calls * float(legacy_row["host_to_host_median_ms"]) / 1000.0
                if legacy_row is not None
                else float("nan")
            )
            timing_basis = (
                "legacy_per_call_estimate" if legacy_row is not None else "missing"
            )
        device_throughput = throughput_timing.get(
            (call_stride, horizon, "device_resident")
        )
        host_throughput = throughput_timing.get((call_stride, horizon, "host_to_host"))
        matched_reference_throughput = next(
            (
                row
                for row in reference_throughput_rows
                if matched is not None
                and int(row["resolution"]) == int(matched["resolution"])
                and int(row["horizon"]) == horizon
            ),
            None,
        )
        reference_wall = (
            float(matched["batch1_wall_seconds_median"])
            if matched is not None
            else float("nan")
        )
        output.append(
            {
                "path": path,
                "horizon": horizon,
                "call_stride": call_stride,
                "num_calls": num_calls,
                "num_matched_cases": len(group),
                "num_expected_cases": len(selected_cases),
                "case_coverage_complete": case_coverage_complete,
                "learned_completion_fraction": completion_fraction,
                "learned_fixed_scale_conservative_relative_l2_mean": (learned_error),
                "learned_device_resident_wall_seconds": device_wall,
                "learned_host_to_host_wall_seconds": host_wall,
                "learned_timing_basis": timing_basis,
                "learned_device_resident_amortized_cases_per_second": (
                    float(device_throughput["cases_per_second"])
                    if device_throughput is not None
                    else float("nan")
                ),
                "learned_host_to_host_amortized_cases_per_second": (
                    float(host_throughput["cases_per_second"])
                    if host_throughput is not None
                    else float("nan")
                ),
                "reference_match_available": matched is not None,
                "reference_resolution": (
                    int(matched["resolution"]) if matched is not None else None
                ),
                "reference_fixed_scale_conservative_relative_l2_mean": (
                    float(matched["fixed_scale_conservative_relative_l2_mean"])
                    if matched is not None
                    else float("nan")
                ),
                "reference_batch1_wall_seconds_median": reference_wall,
                "reference_amortized_cases_per_second": (
                    float(matched_reference_throughput["cases_per_second"])
                    if matched_reference_throughput is not None
                    else float("nan")
                ),
                "device_resident_speedup_vs_accuracy_matched_reference": (
                    reference_wall / device_wall
                    if matched is not None and device_wall > 0.0
                    else float("nan")
                ),
                "host_to_host_speedup_vs_accuracy_matched_reference": (
                    reference_wall / host_wall
                    if matched is not None and host_wall > 0.0
                    else float("nan")
                ),
                "claim_eligible": (
                    matched is not None
                    and direct_timing_available
                    and case_coverage_complete
                    and completion_fraction == 1.0
                ),
                "accuracy_match_rule": (
                    "fastest_complete_reference_with_mean_error_learned_error"
                ),
            }
        )
    return output


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


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.torch_threads < 1:
        raise ValueError("--torch-threads must be positive")
    if args.learned_warmup < 0 or args.learned_repeats < 1:
        raise ValueError("learned timing counts are invalid")
    if args.throughput_batch_size < 1:
        raise ValueError("--throughput-batch-size must be positive")
    if (
        args.reference_workers < 1
        or args.reference_throughput_warmup < 0
        or args.reference_throughput_repeats < 1
    ):
        raise ValueError("reference throughput timing counts are invalid")
    if not args.allow_uncontrolled_reference_threads and (
        os.environ.get("OMP_NUM_THREADS") != "1"
        or os.environ.get("MKL_NUM_THREADS") != "1"
    ):
        raise ValueError(
            "set OMP_NUM_THREADS=1 and MKL_NUM_THREADS=1 for reference timing"
        )
    torch.set_num_threads(args.torch_threads)
    device = _select_device(args.device)
    source = load_euler1d_npz(args.data_path)
    source.validate()
    horizons = sorted(set(int(value) for value in args.horizons))
    resolutions = sorted(set(int(value) for value in args.resolutions))
    if any(value < 1 or value >= source.num_frames for value in horizons):
        raise ValueError("all horizons must be valid saved frame indices")
    if any(
        value < 1 or value > source.num_cells or source.num_cells % value
        for value in resolutions
    ):
        raise ValueError(
            "resolutions must be positive divisors of the dataset resolution"
        )
    if args.case_count < 1 or args.warmup_cases < 0:
        raise ValueError("case count must be positive and warmup nonnegative")
    if args.learned_timing_csv is not None and args.learned_paths_jsonl is None:
        raise ValueError("legacy learned timing requires learned paths")
    if (
        args.learned_paths_jsonl is not None
        and not args.checkpoint
        and args.learned_timing_csv is None
    ):
        raise ValueError("learned paths require checkpoints or legacy timing")
    split_cases = _frozen_split(
        source,
        split_seed=args.split_seed,
        train_cases=args.train_cases,
        val_cases=args.val_cases,
        test_cases=args.test_cases,
        split=args.split,
    )
    if args.case_count > split_cases.size:
        raise ValueError("--case-count exceeds the frozen split")
    case_ids = split_cases[: args.case_count]
    models: dict[int, tuple[torch.nn.Module, torch.nn.Module]] = {}
    checkpoint_contract: dict[str, Any] = {}
    data_sha256 = sha256_file(args.data_path)
    for stride, path in _checkpoint_paths(args.checkpoint).items():
        model, adapter, _normalizer, checkpoint = load_frozen_residual_checkpoint(
            path,
            device,
        )
        if int(checkpoint["args"]["step_stride"]) != stride:
            raise ValueError(f"checkpoint stride mismatch for {path}")
        checkpoint_cases = np.asarray(
            checkpoint["val_cases" if args.split == "validation" else "test_cases"],
            dtype=np.int64,
        )
        if not np.array_equal(checkpoint_cases, split_cases):
            raise ValueError(f"checkpoint stride {stride} uses a different split")
        if checkpoint.get("data_sha256") != data_sha256:
            raise ValueError(f"checkpoint stride {stride} uses a different dataset")
        models[stride] = (model, adapter)
        checkpoint_contract[str(stride)] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "best_epoch": checkpoint.get("best_epoch"),
            "step_stride": stride,
            "recurrent_coordinates": checkpoint["args"].get("recurrent_coordinates"),
        }

    for warmup_id in case_ids[: args.warmup_cases]:
        _run_reference_case(
            source,
            int(warmup_id),
            horizons[0],
            resolutions[-1],
            args.initialization_mode,
        )

    rows: list[dict[str, Any]] = []
    for resolution in resolutions:
        for horizon in horizons:
            for case_id in case_ids:
                try:
                    rows.append(
                        _run_reference_case(
                            source,
                            int(case_id),
                            horizon,
                            resolution,
                            args.initialization_mode,
                        )
                    )
                except Exception as error:
                    rows.append(
                        {
                            "case_id": int(case_id),
                            "horizon": horizon,
                            "physical_horizon": float(
                                source.t[int(case_id), horizon]
                                - source.t[int(case_id), 0]
                            ),
                            "resolution": resolution,
                            "wall_seconds": float("nan"),
                            "primitive_relative_l2": float("nan"),
                            "fixed_scale_conservative_relative_l2": float("nan"),
                            "shock_top2_position_mae": float("nan"),
                            "min_density": float("nan"),
                            "min_pressure": float("nan"),
                            "fallback_count": 0,
                            "completed": False,
                            "termination_reason": (f"{type(error).__name__}: {error}"),
                        }
                    )
    summary = summarize_reference_rows(rows)
    reference_throughput = benchmark_reference_throughput(
        source,
        case_ids,
        horizons,
        resolutions,
        args.initialization_mode,
        workers=args.reference_workers,
        warmup=args.reference_throughput_warmup,
        repeats=args.reference_throughput_repeats,
    )
    learned_timing = (
        benchmark_learned_full_rollouts(
            source,
            case_ids,
            models,
            horizons,
            device,
            warmup=args.learned_warmup,
            repeats=args.learned_repeats,
            throughput_batch_size=args.throughput_batch_size,
        )
        if models
        else []
    )
    if args.learned_timing_csv is not None:
        learned_timing.extend(_read_csv(args.learned_timing_csv))
    accuracy_matched: list[dict[str, Any]] = []
    if args.learned_paths_jsonl is not None:
        accuracy_matched = build_accuracy_matched_rows(
            _read_jsonl(args.learned_paths_jsonl),
            learned_timing,
            summary,
            case_ids,
            reference_throughput,
        )

    contract = {
        "data_path": str(args.data_path),
        "data_sha256": data_sha256,
        "benchmark_source_sha256": sha256_file(Path(__file__).resolve()),
        "reference_solver_source_sha256": sha256_file(
            Path(__file__).with_name("euler1d_weno_hllc_ader_dataset.py")
        ),
        "split": args.split,
        "split_seed": args.split_seed,
        "case_ids": case_ids.tolist(),
        "horizons": horizons,
        "resolutions": resolutions,
        "reference_solver": {
            "method": str(_scalar(source.metadata, "method")),
            "cfl": float(_scalar(source.metadata, "cfl")),
            "initialization_mode": args.initialization_mode,
            "gamma": source.gamma,
            "ng": 3,
            "rho_floor": 1.0e-12,
            "p_floor": 1.0e-12,
            "saved_target_boundaries_preserved": True,
            "coarse_comparison": (
                "piecewise_constant_prolongation_to_native_truth_grid"
            ),
        },
        "timing_protocol": {
            "host": platform.node(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
            "reference_batch1": {
                "batch_size": 1,
                "warmup_cases": args.warmup_cases,
                "synchronization": "synchronous_numpy_cpu_call",
                "boundary": (
                    "case_metadata_to_config_through_adaptive_integration_and_"
                    "final_piecewise_constant_prolongation"
                ),
                "metric_computation_excluded": True,
                "data_transfer": "cpu_only_no_accelerator_transfer",
            },
            "reference_amortized": {
                "batch_size": int(case_ids.size),
                "workers": args.reference_workers,
                "warmup": args.reference_throughput_warmup,
                "repeats": args.reference_throughput_repeats,
                "thread_pool_creation_excluded": True,
                "task_scheduling_included": True,
            },
            "learned": {
                "device": str(device),
                "device_name": (
                    torch.cuda.get_device_name(device)
                    if device.type == "cuda"
                    else "cpu"
                ),
                "warmup": args.learned_warmup,
                "repeats": args.learned_repeats,
                "batch_sizes": sorted(
                    {1, min(args.throughput_batch_size, int(case_ids.size))}
                ),
                "synchronization": (
                    "torch_cuda_synchronize_before_and_after_entire_rollout"
                    if device.type == "cuda"
                    else "synchronous_cpu_call"
                ),
                "device_resident_boundary": (
                    "prebuilt_device_state_geometry_metadata_through_all_"
                    "recurrent_calls_to_device_endpoint"
                ),
                "host_to_host_boundary": (
                    "loaded_numpy_initial_state_and_metadata_through_initial_"
                    "h2d_all_device_recurrent_calls_and_final_d2h"
                ),
                "per_call_host_transfer": False,
                "raw_conservative_recurrence": True,
                "fixed_call_count_for_timing": True,
            },
        },
        "checkpoints": checkpoint_contract,
        "learned_paths_jsonl": (
            str(args.learned_paths_jsonl)
            if args.learned_paths_jsonl is not None
            else None
        ),
        "learned_paths_jsonl_sha256": (
            sha256_file(args.learned_paths_jsonl)
            if args.learned_paths_jsonl is not None
            else None
        ),
        "learned_timing_csv": (
            str(args.learned_timing_csv)
            if args.learned_timing_csv is not None
            else None
        ),
        "learned_timing_csv_sha256": (
            sha256_file(args.learned_timing_csv)
            if args.learned_timing_csv is not None
            else None
        ),
        "legacy_timing_claim_eligible": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.output_dir / "reference_cases.jsonl", rows)
    _write_csv(args.output_dir / "reference_summary.csv", summary)
    _write_csv(
        args.output_dir / "reference_throughput.csv",
        reference_throughput,
    )
    _write_csv(
        args.output_dir / "learned_full_rollout_timing.csv",
        learned_timing,
    )
    _write_csv(
        args.output_dir / "accuracy_matched_pareto.csv",
        accuracy_matched,
    )
    (args.output_dir / "contract.json").write_text(
        json.dumps(json_ready(contract), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(
            json_ready(
                {
                    "contract": contract,
                    "reference_summary": summary,
                    "reference_throughput": reference_throughput,
                    "learned_full_rollout_timing": learned_timing,
                    "accuracy_matched_pareto": accuracy_matched,
                }
            ),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(json.dumps(json_ready(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
