"""Attribute the frozen Line-1 stride gap with identical-state model forks.

This XLINE-001 diagnostic performs no training and applies no inference repair.
It selects the registered small-stride failures and descriptor-matched stable
controls from frozen artifacts, then:

1. records both exact vector decompositions of one learned call; and
2. compares G_8, G_4^2, G_2^4, and G_1^8 from identical truth, stride-1, and
   stride-8 source states at identical physical endpoints.

The diagnostic is intentionally narrow. It does not select a method, smooth a
state, project an inadmissible proposal, or access a new split.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.time_dependent_no.diagnose_euler1d_flow_map_solver_consistency import (  # noqa: E402
    REQUIRED_STRIDES,
    _checkpoint_paths,
    _on_policy_state_bank,
    _reference_batch,
    _select_device,
    _validate_checkpoint_contract,
    require_fresh_output_dir,
    truth_replay_gate,
)
from scripts.time_dependent_no.diagnose_euler1d_generated_state_consistency import (  # noqa: E402
    shock_region_mask,
    solver_config_from_source,
)
from scripts.time_dependent_no.evaluate_euler1d_flow_map_frontier import (  # noqa: E402
    _euler_state_is_admissible,
    _frozen_split,
    _predict_model_batch_state,
    _saved_time_sha256,
    _write_csv,
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

SCHEMA = "euler1d_stride_attribution_xline001_v1"
PATH_HORIZON = 8
STATE_SOURCE_STRIDES = (1, 8)
STATE_SOURCES = ("truth", "s1", "s8")
SMALL_FAILURE_STRIDES = (1, 2)
EPS = 1.0e-30
DESCRIPTOR_FIELDS = (
    "absolute_log_density_ratio",
    "absolute_log_pressure_ratio",
    "discontinuity_fraction",
    "initial_effective_cfl_stride8",
    "max_initial_characteristic_speed",
    "normalized_pressure_jump",
    "velocity_jump_over_max_sound_speed",
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
    parser.add_argument("--frontier-contract", type=Path, required=True)
    parser.add_argument("--direct-curves", type=Path, required=True)
    parser.add_argument("--descriptors", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-failures", type=int, default=4)
    parser.add_argument("--max-controls", type=int, default=2)
    parser.add_argument("--lookback-macros", type=int, default=2)
    parser.add_argument("--split-seed", type=int, default=20260707)
    parser.add_argument("--train-cases", type=int, default=384)
    parser.add_argument("--val-cases", type=int, default=64)
    parser.add_argument("--test-cases", type=int, default=64)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--solver-workers", type=int, default=4)
    parser.add_argument("--ghost-cells", type=int, default=3)
    parser.add_argument("--rho-floor", type=float, default=1.0e-10)
    parser.add_argument("--p-floor", type=float, default=1.0e-10)
    parser.add_argument("--shock-radius-cells", type=int, default=4)
    parser.add_argument("--truth-replay-max-tolerance", type=float, default=5.0e-5)
    parser.add_argument("--truth-replay-mean-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--truth-replay-p99-tolerance", type=float, default=1.0e-6)
    args = parser.parse_args(argv)
    if args.max_failures < 1 or args.max_failures > 4:
        parser.error("--max-failures must lie in [1,4]")
    if args.max_controls < 1 or args.max_controls > 4:
        parser.error("--max-controls must lie in [1,4]")
    if args.lookback_macros < 0:
        parser.error("--lookback-macros must be nonnegative")
    if min(args.torch_threads, args.solver_workers) < 1:
        parser.error("thread counts must be positive")
    if args.ghost_cells < 3:
        parser.error("--ghost-cells must be at least three")
    if args.shock_radius_cells < 0:
        parser.error("--shock-radius-cells must be nonnegative")
    return args


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            value = json.loads(stripped)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    if not rows:
        raise ValueError(f"no rows found in {path}")
    return rows


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if not rows:
        raise ValueError(f"no rows found in {path}")
    return rows


def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no", "", "none", "null"}:
        return False
    raise ValueError(f"cannot interpret boolean value {value!r}")


def _descriptor_table(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[int, np.ndarray], np.ndarray, np.ndarray]:
    vectors: dict[int, np.ndarray] = {}
    for row in rows:
        case_id = int(row["case_id"])
        if case_id in vectors:
            raise ValueError(f"duplicate descriptor row for case {case_id}")
        vector = np.asarray(
            [float(row[name]) for name in DESCRIPTOR_FIELDS],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(vector)):
            raise ValueError(f"nonfinite descriptor for case {case_id}")
        vectors[case_id] = vector
    matrix = np.stack([vectors[case_id] for case_id in sorted(vectors)])
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    std = np.where(std > 0.0, std, 1.0)
    return vectors, mean, std


def select_cohort(
    direct_rows: Sequence[Mapping[str, Any]],
    descriptor_rows: Sequence[Mapping[str, Any]],
    *,
    max_failures: int,
    max_controls: int,
) -> dict[str, Any]:
    """Select failures and controls without consulting later rollout metrics."""

    if not 1 <= max_failures <= 4 or not 1 <= max_controls <= 4:
        raise ValueError("cohort caps must lie in [1,4]")
    horizons = [int(row["frame"]) for row in direct_rows]
    horizon = max(horizons)
    failures_by_case: dict[int, dict[str, int]] = {}
    for row in direct_rows:
        stride = int(row["stride"])
        if stride not in SMALL_FAILURE_STRIDES or _bool_value(
            row["raw_state_admissible"]
        ):
            continue
        case_id = int(row["case_id"])
        candidate = {
            "case_id": case_id,
            "failure_stride": stride,
            "failure_frame": int(row["frame"]),
        }
        previous = failures_by_case.get(case_id)
        if previous is None or (
            candidate["failure_frame"],
            candidate["failure_stride"],
        ) < (
            previous["failure_frame"],
            previous["failure_stride"],
        ):
            failures_by_case[case_id] = candidate
    failures = sorted(
        failures_by_case.values(),
        key=lambda row: (row["failure_frame"], row["case_id"]),
    )[:max_failures]
    if not failures:
        raise ValueError("no registered small-stride raw failure was found")

    descriptor_vectors, descriptor_mean, descriptor_std = _descriptor_table(
        descriptor_rows
    )
    direct_case_ids = {int(row["case_id"]) for row in direct_rows}
    missing_descriptors = direct_case_ids.difference(descriptor_vectors)
    if missing_descriptors:
        raise ValueError(f"missing descriptors for cases {sorted(missing_descriptors)}")

    failure_ids = {row["case_id"] for row in failures}
    stable_candidates: list[int] = []
    for case_id in sorted(direct_case_ids.difference(failure_ids)):
        stable = all(
            any(
                int(row["case_id"]) == case_id
                and int(row["stride"]) == stride
                and int(row["frame"]) == horizon
                and _bool_value(row["raw_state_admissible"])
                for row in direct_rows
            )
            for stride in REQUIRED_STRIDES
        )
        if stable:
            stable_candidates.append(case_id)
    if len(stable_candidates) < max_controls:
        raise ValueError(
            f"only {len(stable_candidates)} eligible controls for "
            f"--max-controls={max_controls}"
        )

    controls: list[dict[str, Any]] = []
    available = set(stable_candidates)
    failure_index = 0
    while len(controls) < max_controls:
        failure = failures[failure_index % len(failures)]
        failure_index += 1
        failure_vector = (
            descriptor_vectors[failure["case_id"]] - descriptor_mean
        ) / descriptor_std
        distances = []
        for case_id in available:
            control_vector = (
                descriptor_vectors[case_id] - descriptor_mean
            ) / descriptor_std
            distances.append(
                (
                    float(np.linalg.norm(control_vector - failure_vector)),
                    case_id,
                )
            )
        if not distances:
            raise RuntimeError("control matching exhausted eligible cases")
        distance, case_id = min(distances)
        available.remove(case_id)
        controls.append(
            {
                "case_id": case_id,
                "role": "stable_control",
                "matched_failure_case_id": failure["case_id"],
                "matched_failure_frame": failure["failure_frame"],
                "descriptor_distance": distance,
            }
        )

    return {
        "horizon": horizon,
        "descriptor_fields": list(DESCRIPTOR_FIELDS),
        "descriptor_standardization": {
            "mean": descriptor_mean.tolist(),
            "std": descriptor_std.tolist(),
        },
        "failures": [
            {
                **failure,
                "role": "failure",
                "matched_failure_case_id": failure["case_id"],
                "matched_failure_frame": failure["failure_frame"],
                "descriptor_distance": 0.0,
            }
            for failure in failures
        ],
        "controls": controls,
    }


def event_start_frames(
    failure_frame: int,
    *,
    lookback_macros: int,
    path_horizon: int = PATH_HORIZON,
) -> tuple[int, ...]:
    if failure_frame <= 0 or lookback_macros < 0 or path_horizon <= 0:
        raise ValueError("invalid event-frame contract")
    final_start = ((failure_frame - 1) // path_horizon) * path_horizon
    first_start = max(0, final_start - lookback_macros * path_horizon)
    return tuple(range(first_start, final_start + 1, path_horizon))


def conservative_scale(gamma: float) -> np.ndarray:
    return np.asarray((1.0, 1.0, 1.0 / (gamma - 1.0)), dtype=np.float64)


def scaled_inner(
    left: np.ndarray,
    right: np.ndarray,
    scale: np.ndarray,
    mask: np.ndarray | None = None,
) -> float:
    left_value = np.asarray(left, dtype=np.float64) / scale
    right_value = np.asarray(right, dtype=np.float64) / scale
    if left_value.shape != right_value.shape or left_value.ndim != 2:
        raise ValueError("fields must share shape [cells,components]")
    if mask is not None:
        selected = np.asarray(mask, dtype=bool)
        if selected.shape != (left_value.shape[0],):
            raise ValueError("mask must have shape [cells]")
        left_value = left_value[selected]
        right_value = right_value[selected]
    if left_value.size == 0:
        return float("nan")
    return float(np.mean(np.sum(left_value * right_value, axis=-1)))


def decomposition_metrics(
    total: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    scale: np.ndarray,
    *,
    mask: np.ndarray | None = None,
) -> dict[str, float | int | str]:
    """Measure one exact additive vector identity under equal-cell weighting."""

    total_value = np.asarray(total, dtype=np.float64)
    first_value = np.asarray(first, dtype=np.float64)
    second_value = np.asarray(second, dtype=np.float64)
    if (
        total_value.ndim != 2
        or first_value.shape != total_value.shape
        or second_value.shape != total_value.shape
    ):
        raise ValueError("decomposition fields must share shape [cells,components]")
    selected = (
        np.ones(total_value.shape[0], dtype=bool)
        if mask is None
        else np.asarray(mask, dtype=bool)
    )
    if selected.shape != (total_value.shape[0],):
        raise ValueError("mask must have shape [cells]")
    if not np.any(selected):
        return {"status": "empty", "selected_cell_count": 0}

    total_energy = scaled_inner(total_value, total_value, scale, selected)
    first_energy = scaled_inner(first_value, first_value, scale, selected)
    second_energy = scaled_inner(second_value, second_value, scale, selected)
    cross = scaled_inner(first_value, second_value, scale, selected)
    residual = total_value - first_value - second_value
    residual_energy = scaled_inner(residual, residual, scale, selected)
    total_norm = math.sqrt(max(total_energy, 0.0))
    first_norm = math.sqrt(max(first_energy, 0.0))
    second_norm = math.sqrt(max(second_energy, 0.0))
    residual_norm = math.sqrt(max(residual_energy, 0.0))
    magnitude_sum = first_norm + second_norm
    cosine = float("nan")
    if first_norm > EPS and second_norm > EPS:
        cosine = cross / (first_norm * second_norm)
    energy_scale = max(
        abs(total_energy),
        first_energy + second_energy + 2.0 * abs(cross),
        EPS,
    )
    return {
        "status": "available",
        "selected_cell_count": int(np.count_nonzero(selected)),
        "total_norm": total_norm,
        "first_norm": first_norm,
        "second_norm": second_norm,
        "first_magnitude_share": (
            float("nan") if magnitude_sum <= EPS else first_norm / magnitude_sum
        ),
        "first_second_cosine": cosine,
        "first_energy_fraction_of_total": first_energy / max(total_energy, EPS),
        "second_energy_fraction_of_total": second_energy / max(total_energy, EPS),
        "cross_energy_fraction_of_total": 2.0 * cross / max(total_energy, EPS),
        "relative_reconstruction_residual": residual_norm
        / max(total_norm, magnitude_sum, EPS),
        "relative_energy_identity_residual": abs(
            total_energy - first_energy - second_energy - 2.0 * cross
        )
        / energy_scale,
    }


def scaled_relative_l2(
    prediction: np.ndarray,
    target: np.ndarray,
    scale: np.ndarray,
) -> float:
    difference = (np.asarray(prediction) - np.asarray(target)) / scale
    denominator = np.asarray(target) / scale
    return float(
        np.linalg.norm(difference.reshape(-1))
        / max(np.linalg.norm(denominator.reshape(-1)), EPS)
    )


def _predict_single(
    source: Euler1DNPZ,
    case_id: int,
    primitive: np.ndarray,
    conservative: np.ndarray,
    start_frame: int,
    stride: int,
    model: torch.nn.Module,
    adapter: torch.nn.Module,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    prediction, prediction_conservative = _predict_model_batch_state(
        source,
        np.asarray([case_id], dtype=np.int64),
        np.asarray(primitive, dtype=np.float64)[None],
        start_frame,
        stride,
        model,
        adapter,
        device,
        conservative=np.asarray(conservative, dtype=np.float64)[None],
    )
    return (
        np.asarray(prediction[0], dtype=np.float64),
        np.asarray(prediction_conservative[0], dtype=np.float64),
    )


def apply_learned_path(
    source: Euler1DNPZ,
    case_id: int,
    primitive: np.ndarray,
    conservative: np.ndarray,
    start_frame: int,
    endpoint_horizon: int,
    stride: int,
    model: torch.nn.Module,
    adapter: torch.nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    if endpoint_horizon <= 0 or endpoint_horizon % stride:
        raise ValueError("endpoint horizon must be a positive stride multiple")
    current_primitive = np.asarray(primitive, dtype=np.float64).copy()
    current_conservative = np.asarray(conservative, dtype=np.float64).copy()
    for call_index in range(endpoint_horizon // stride):
        call_start = start_frame + call_index * stride
        proposed, proposed_conservative = _predict_single(
            source,
            case_id,
            current_primitive,
            current_conservative,
            call_start,
            stride,
            model,
            adapter,
            device,
        )
        finite = bool(
            np.all(np.isfinite(proposed)) and np.all(np.isfinite(proposed_conservative))
        )
        admissible = finite and _euler_state_is_admissible(proposed)
        if not admissible:
            return {
                "completed": False,
                "failure_frame": call_start + stride,
                "failure_cause": (
                    "nonfinite_raw_state" if not finite else "nonpositive_raw_state"
                ),
                "primitive": proposed,
                "conservative": proposed_conservative,
                "calls_completed": call_index,
            }
        current_primitive = proposed
        current_conservative = proposed_conservative
    return {
        "completed": True,
        "failure_frame": None,
        "failure_cause": None,
        "primitive": current_primitive,
        "conservative": current_conservative,
        "calls_completed": endpoint_horizon // stride,
    }


def _reference_single(
    source: Euler1DNPZ,
    case_id: int,
    primitive: np.ndarray,
    start_frame: int,
    stride: int,
    config: Any,
    executor: ThreadPoolExecutor,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    return _reference_batch(
        source,
        np.asarray([case_id], dtype=np.int64),
        np.asarray(primitive, dtype=np.float64)[None],
        start_frame,
        stride,
        config,
        executor,
    )[0]


def _flatten_metrics(
    row: dict[str, Any],
    prefix: str,
    metrics: Mapping[str, Any],
) -> None:
    for name, value in metrics.items():
        row[f"{prefix}_{name}"] = value


def _summary_rows(
    rows: Sequence[Mapping[str, Any]],
    keys: Sequence[str],
    metrics: Sequence[str],
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = tuple(row[name] for name in keys)
        groups.setdefault(key, []).append(row)
    output: list[dict[str, Any]] = []
    for key, group in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        summary = {name: value for name, value in zip(keys, key)}
        summary["count"] = len(group)
        for metric in metrics:
            values = np.asarray(
                [
                    float(row[metric])
                    for row in group
                    if metric in row and row[metric] is not None
                ],
                dtype=np.float64,
            )
            values = values[np.isfinite(values)]
            summary[f"{metric}_mean"] = (
                float(np.mean(values)) if values.size else float("nan")
            )
            summary[f"{metric}_median"] = (
                float(np.median(values)) if values.size else float("nan")
            )
        output.append(summary)
    return output


def _load_frontier_contract(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("frontier contract must be a JSON object")
    return value


def _cohort_evaluations(
    cohort: Mapping[str, Any],
    lookback_macros: int,
) -> list[dict[str, Any]]:
    evaluations: list[dict[str, Any]] = []
    for entry in (*cohort["failures"], *cohort["controls"]):
        failure_frame = int(entry["matched_failure_frame"])
        frames = event_start_frames(
            failure_frame,
            lookback_macros=lookback_macros,
        )
        final_start = frames[-1]
        for start_frame in frames:
            evaluations.append(
                {
                    **entry,
                    "start_frame": start_frame,
                    "event_offset_frames": start_frame - final_start,
                    "endpoint_frame": start_frame + PATH_HORIZON,
                }
            )
    return evaluations


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    require_fresh_output_dir(args.output_dir)
    torch.set_num_threads(args.torch_threads)
    device = _select_device(args.device)
    source = load_euler1d_npz(args.data_path)
    source.validate()
    if not np.array_equal(source.t, np.broadcast_to(source.t[:1], source.t.shape)):
        raise ValueError("XLINE-001 requires common saved times")

    direct_rows = read_jsonl(args.direct_curves)
    descriptor_rows = read_csv_rows(args.descriptors)
    cohort = select_cohort(
        direct_rows,
        descriptor_rows,
        max_failures=args.max_failures,
        max_controls=args.max_controls,
    )
    evaluations = _cohort_evaluations(cohort, args.lookback_macros)
    if max(row["endpoint_frame"] for row in evaluations) >= source.num_frames:
        raise ValueError("event-aligned endpoint exceeds the saved trajectory")

    expected_cases = _frozen_split(
        source,
        split_seed=args.split_seed,
        train_cases=args.train_cases,
        val_cases=args.val_cases,
        test_cases=args.test_cases,
        split="test",
    )
    contract_case_ids = {
        int(case_id)
        for case_id in _load_frontier_contract(args.frontier_contract)["case_ids"]
    }
    if contract_case_ids != set(map(int, expected_cases)):
        raise ValueError("frontier contract and reconstructed test split disagree")
    cohort_case_ids = sorted({int(row["case_id"]) for row in evaluations})
    if not set(cohort_case_ids).issubset(contract_case_ids):
        raise ValueError("cohort contains a case outside the frozen test split")

    data_sha256 = sha256_file(args.data_path)
    saved_time_sha256 = _saved_time_sha256(source)
    frontier_contract = _load_frontier_contract(args.frontier_contract)
    if frontier_contract.get("data_sha256") != data_sha256:
        raise ValueError("dataset hash disagrees with the frontier contract")
    if frontier_contract.get("saved_time_sha256") != saved_time_sha256:
        raise ValueError("saved-time hash disagrees with the frontier contract")

    checkpoint_paths = _checkpoint_paths(args.checkpoint)
    models: dict[int, tuple[torch.nn.Module, torch.nn.Module]] = {}
    checkpoint_contract: dict[str, Any] = {}
    for stride, path in sorted(checkpoint_paths.items()):
        model, adapter, _normalizer, checkpoint = load_frozen_residual_checkpoint(
            path,
            device,
        )
        frozen = _validate_checkpoint_contract(
            checkpoint,
            stride=stride,
            expected_cases=expected_cases,
            data_sha256=data_sha256,
            saved_time_sha256=saved_time_sha256,
            split="test",
        )
        actual_hash = sha256_file(path)
        expected_hash = frontier_contract["checkpoints"][str(stride)]["sha256"]
        if actual_hash != expected_hash:
            raise ValueError(f"stride-{stride} checkpoint hash mismatch")
        models[stride] = (model, adapter)
        checkpoint_contract[str(stride)] = {
            "sha256": actual_hash,
            "best_epoch": checkpoint.get("best_epoch"),
            "frozen_coordinates": frozen,
        }

    case_ids_array = np.asarray(expected_cases, dtype=np.int64)
    case_position = {
        int(case_id): position for position, case_id in enumerate(case_ids_array)
    }
    max_start = max(int(row["start_frame"]) for row in evaluations)
    source_banks = {
        stride: _on_policy_state_bank(
            source,
            case_ids_array,
            stride,
            models[stride][0],
            models[stride][1],
            max_start,
            device,
        )
        for stride in STATE_SOURCE_STRIDES
    }
    solver_config = solver_config_from_source(
        source,
        ng=args.ghost_cells,
        rho_floor=args.rho_floor,
        p_floor=args.p_floor,
    )
    scale = conservative_scale(source.gamma)

    truth_cache: dict[
        tuple[int, int, int],
        tuple[np.ndarray, np.ndarray, dict[str, int]],
    ] = {}
    truth_replay_rows: list[dict[str, Any]] = []
    unique_case_frames = sorted(
        {(int(row["case_id"]), int(row["start_frame"])) for row in evaluations}
    )
    with ThreadPoolExecutor(max_workers=args.solver_workers) as executor:
        for case_id, start_frame in unique_case_frames:
            truth_primitive = np.asarray(
                source.data[case_id, start_frame],
                dtype=np.float64,
            )
            for stride in REQUIRED_STRIDES:
                reference = _reference_single(
                    source,
                    case_id,
                    truth_primitive,
                    start_frame,
                    stride,
                    solver_config,
                    executor,
                )
                truth_cache[(case_id, start_frame, stride)] = reference
                target_conservative = primitive_to_conservative_np(
                    np.asarray(
                        source.data[case_id, start_frame + stride],
                        dtype=np.float64,
                    ),
                    source.gamma,
                )
                replay_error = scaled_relative_l2(
                    reference[1],
                    target_conservative,
                    scale,
                )
                truth_replay_rows.append(
                    {
                        "case_id": case_id,
                        "start_frame": start_frame,
                        "stride": stride,
                        "target_frame": start_frame + stride,
                        "fixed_scale_conservative_relative_l2": replay_error,
                        **{
                            f"reference_{name}": value
                            for name, value in reference[2].items()
                        },
                    }
                )
        replay_gate = truth_replay_gate(
            [row["fixed_scale_conservative_relative_l2"] for row in truth_replay_rows],
            max_tolerance=args.truth_replay_max_tolerance,
            mean_tolerance=args.truth_replay_mean_tolerance,
            p99_tolerance=args.truth_replay_p99_tolerance,
        )

        args.output_dir.mkdir(parents=True, exist_ok=True)
        _write_csv(args.output_dir / "truth_replay.csv", truth_replay_rows)
        base_contract = {
            "schema": SCHEMA,
            "status": (
                "truth_replay_gate_passed"
                if replay_gate["passed"]
                else "truth_replay_gate_failed"
            ),
            "data_sha256": data_sha256,
            "saved_time_sha256": saved_time_sha256,
            "frontier_contract_sha256": sha256_file(args.frontier_contract),
            "direct_curves_sha256": sha256_file(args.direct_curves),
            "descriptors_sha256": sha256_file(args.descriptors),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "checkpoints": checkpoint_contract,
            "case_ids": cohort_case_ids,
            "state_sources": list(STATE_SOURCES),
            "direct_composed_paths": [
                "G1^8",
                "G2^4",
                "G4^2",
                "G8",
            ],
            "path_horizon_frames": PATH_HORIZON,
            "lookback_macros": args.lookback_macros,
            "raw_inference": "no_clipping_no_projection_no_state_floor",
            "source_state_rollout_batch_size": int(expected_cases.size),
            "fork_model_batch_size": 1,
            "batch_presentation_contract": (
                "source states replay the full frozen split as in D032; "
                "identical-state forks use batch one"
            ),
            "cell_weight": "equal_cell_physical_scale",
            "component_scale": scale.tolist(),
            "truth_replay_gate": replay_gate,
            "solver_replay": solver_config.__dict__,
            "device": str(device),
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
        }
        (args.output_dir / "cohort.json").write_text(
            json.dumps(json_ready(cohort), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        if not replay_gate["passed"]:
            report = {
                "contract": base_contract,
                "status": "truth_replay_gate_failed",
                "runtime_seconds": time.perf_counter() - started,
            }
            (args.output_dir / "report.json").write_text(
                json.dumps(json_ready(report), indent=2, sort_keys=True),
                encoding="utf-8",
            )
            return report

        decomposition_rows: list[dict[str, Any]] = []
        vector_bank: dict[str, list[np.ndarray]] = {
            "total": [],
            "same_state_model_defect": [],
            "reference_propagation": [],
            "learned_amplification": [],
            "truth_state_consistency": [],
        }
        vector_row_ids: list[str] = []
        path_rows: list[dict[str, Any]] = []
        path_endpoints: list[np.ndarray] = []
        path_row_ids: list[str] = []

        def source_state(
            state_source: str,
            case_id: int,
            start_frame: int,
        ) -> tuple[np.ndarray, np.ndarray]:
            if state_source == "truth":
                primitive = np.asarray(
                    source.data[case_id, start_frame],
                    dtype=np.float64,
                )
                return primitive, primitive_to_conservative_np(
                    primitive,
                    source.gamma,
                )
            source_stride = int(state_source[1:])
            primitive_bank, conservative_bank, active = source_banks[source_stride][
                start_frame
            ]
            position = case_position[case_id]
            if not bool(active[position]):
                raise RuntimeError(
                    f"{state_source} case {case_id} is inactive at frame "
                    f"{start_frame}"
                )
            return (
                np.asarray(primitive_bank[position], dtype=np.float64),
                np.asarray(conservative_bank[position], dtype=np.float64),
            )

        for evaluation_id, evaluation in enumerate(evaluations):
            case_id = int(evaluation["case_id"])
            start_frame = int(evaluation["start_frame"])
            truth_primitive, truth_conservative = source_state(
                "truth",
                case_id,
                start_frame,
            )
            for state_source in STATE_SOURCES:
                current_primitive, current_conservative = source_state(
                    state_source,
                    case_id,
                    start_frame,
                )
                existing_error = current_conservative - truth_conservative
                for stride, (model, adapter) in sorted(models.items()):
                    truth_reference = truth_cache[(case_id, start_frame, stride)]
                    if state_source == "truth":
                        state_reference = truth_reference
                    else:
                        state_reference = _reference_single(
                            source,
                            case_id,
                            current_primitive,
                            start_frame,
                            stride,
                            solver_config,
                            executor,
                        )
                    truth_prediction_primitive, truth_prediction = _predict_single(
                        source,
                        case_id,
                        truth_primitive,
                        truth_conservative,
                        start_frame,
                        stride,
                        model,
                        adapter,
                        device,
                    )
                    if state_source == "truth":
                        state_prediction_primitive = truth_prediction_primitive
                        state_prediction = truth_prediction
                    else:
                        (
                            state_prediction_primitive,
                            state_prediction,
                        ) = _predict_single(
                            source,
                            case_id,
                            current_primitive,
                            current_conservative,
                            start_frame,
                            stride,
                            model,
                            adapter,
                            device,
                        )

                    total = state_prediction - truth_reference[1]
                    same_state_model_defect = state_prediction - state_reference[1]
                    reference_propagation = state_reference[1] - truth_reference[1]
                    learned_amplification = state_prediction - truth_prediction
                    truth_state_consistency = truth_prediction - truth_reference[1]
                    target_primitive = np.asarray(
                        source.data[case_id, start_frame + stride],
                        dtype=np.float64,
                    )
                    target_conservative = primitive_to_conservative_np(
                        target_primitive,
                        source.gamma,
                    )
                    shock_mask = shock_region_mask(
                        target_primitive,
                        radius_cells=args.shock_radius_cells,
                    )
                    row_id = (
                        f"d{evaluation_id:03d}_c{case_id}_f{start_frame}_"
                        f"{state_source}_s{stride}"
                    )
                    physical_dt = float(
                        source.t[case_id, start_frame + stride]
                        - source.t[case_id, start_frame]
                    )
                    row: dict[str, Any] = {
                        "row_id": row_id,
                        **evaluation,
                        "state_source": state_source,
                        "map_stride": stride,
                        "target_frame": start_frame + stride,
                        "physical_dt": physical_dt,
                        "source_state_error_norm": math.sqrt(
                            max(
                                scaled_inner(
                                    existing_error,
                                    existing_error,
                                    scale,
                                ),
                                0.0,
                            )
                        ),
                        "model_raw_state_finite": bool(
                            np.all(np.isfinite(state_prediction_primitive))
                            and np.all(np.isfinite(state_prediction))
                        ),
                        "model_raw_state_admissible": bool(
                            np.all(np.isfinite(state_prediction_primitive))
                            and _euler_state_is_admissible(state_prediction_primitive)
                        ),
                        "model_serialized_truth_relative_l2": scaled_relative_l2(
                            state_prediction,
                            target_conservative,
                            scale,
                        ),
                        "model_same_state_reference_relative_l2": (
                            scaled_relative_l2(
                                state_prediction,
                                state_reference[1],
                                scale,
                            )
                        ),
                        "model_defect_norm_per_physical_time": math.sqrt(
                            max(
                                scaled_inner(
                                    same_state_model_defect,
                                    same_state_model_defect,
                                    scale,
                                ),
                                0.0,
                            )
                        )
                        / physical_dt,
                        "prediction_min_density": float(
                            np.nanmin(state_prediction_primitive[:, 0])
                        ),
                        "prediction_min_pressure": float(
                            np.nanmin(state_prediction_primitive[:, 2])
                        ),
                    }
                    for region_name, mask in (
                        ("full", None),
                        ("shock", shock_mask),
                        ("smooth", ~shock_mask),
                    ):
                        _flatten_metrics(
                            row,
                            f"model_reference_{region_name}",
                            decomposition_metrics(
                                total,
                                same_state_model_defect,
                                reference_propagation,
                                scale,
                                mask=mask,
                            ),
                        )
                        _flatten_metrics(
                            row,
                            f"learned_truth_{region_name}",
                            decomposition_metrics(
                                total,
                                learned_amplification,
                                truth_state_consistency,
                                scale,
                                mask=mask,
                            ),
                        )
                    decomposition_rows.append(row)
                    vector_row_ids.append(row_id)
                    vector_bank["total"].append(total)
                    vector_bank["same_state_model_defect"].append(
                        same_state_model_defect
                    )
                    vector_bank["reference_propagation"].append(reference_propagation)
                    vector_bank["learned_amplification"].append(learned_amplification)
                    vector_bank["truth_state_consistency"].append(
                        truth_state_consistency
                    )

                truth_reference_h8 = truth_cache[(case_id, start_frame, PATH_HORIZON)]
                state_reference_h8 = (
                    truth_reference_h8
                    if state_source == "truth"
                    else _reference_single(
                        source,
                        case_id,
                        current_primitive,
                        start_frame,
                        PATH_HORIZON,
                        solver_config,
                        executor,
                    )
                )
                serialized_target = primitive_to_conservative_np(
                    np.asarray(
                        source.data[case_id, start_frame + PATH_HORIZON],
                        dtype=np.float64,
                    ),
                    source.gamma,
                )
                path_results = {
                    stride: apply_learned_path(
                        source,
                        case_id,
                        current_primitive,
                        current_conservative,
                        start_frame,
                        PATH_HORIZON,
                        stride,
                        models[stride][0],
                        models[stride][1],
                        device,
                    )
                    for stride in REQUIRED_STRIDES
                }
                direct_endpoint = path_results[PATH_HORIZON]["conservative"]
                direct_completed = bool(path_results[PATH_HORIZON]["completed"])
                for stride, result in sorted(path_results.items()):
                    row_id = (
                        f"p{evaluation_id:03d}_c{case_id}_f{start_frame}_"
                        f"{state_source}_s{stride}"
                    )
                    completed = bool(result["completed"])
                    endpoint = np.asarray(result["conservative"], dtype=np.float64)
                    row = {
                        "row_id": row_id,
                        **evaluation,
                        "state_source": state_source,
                        "path_stride": stride,
                        "path_label": (
                            f"G{stride}"
                            if stride == PATH_HORIZON
                            else f"G{stride}^{PATH_HORIZON // stride}"
                        ),
                        "learned_call_count": PATH_HORIZON // stride,
                        "completed": completed,
                        "path_failure_frame": result["failure_frame"],
                        "path_failure_cause": result["failure_cause"],
                        "source_state_error_norm": math.sqrt(
                            max(
                                scaled_inner(
                                    existing_error,
                                    existing_error,
                                    scale,
                                ),
                                0.0,
                            )
                        ),
                    }
                    if completed:
                        row.update(
                            {
                                "serialized_truth_relative_l2": scaled_relative_l2(
                                    endpoint,
                                    serialized_target,
                                    scale,
                                ),
                                "same_state_reference_relative_l2": (
                                    scaled_relative_l2(
                                        endpoint,
                                        state_reference_h8[1],
                                        scale,
                                    )
                                ),
                                "truth_reference_relative_l2": scaled_relative_l2(
                                    endpoint,
                                    truth_reference_h8[1],
                                    scale,
                                ),
                                "semigroup_defect_to_G8_relative_l2": (
                                    scaled_relative_l2(
                                        endpoint,
                                        direct_endpoint,
                                        scale,
                                    )
                                    if direct_completed
                                    else float("nan")
                                ),
                            }
                        )
                    path_rows.append(row)
                    path_row_ids.append(row_id)
                    path_endpoints.append(
                        endpoint
                        if completed
                        else np.full_like(current_conservative, np.nan)
                    )

    decomposition_summary = _summary_rows(
        decomposition_rows,
        ("role", "state_source", "map_stride", "event_offset_frames"),
        (
            "model_serialized_truth_relative_l2",
            "model_same_state_reference_relative_l2",
            "model_defect_norm_per_physical_time",
            "model_reference_full_first_magnitude_share",
            "model_reference_full_first_second_cosine",
            "learned_truth_full_first_magnitude_share",
            "learned_truth_full_first_second_cosine",
            "prediction_min_pressure",
        ),
    )
    path_summary = _summary_rows(
        path_rows,
        ("role", "state_source", "path_stride", "event_offset_frames"),
        (
            "serialized_truth_relative_l2",
            "same_state_reference_relative_l2",
            "semigroup_defect_to_G8_relative_l2",
        ),
    )
    max_identity_residual = max(
        float(row[name])
        for row in decomposition_rows
        for name in (
            "model_reference_full_relative_reconstruction_residual",
            "model_reference_full_relative_energy_identity_residual",
            "learned_truth_full_relative_reconstruction_residual",
            "learned_truth_full_relative_energy_identity_residual",
        )
    )
    base_contract["status"] = "complete"
    base_contract["max_decomposition_identity_residual"] = max_identity_residual
    base_contract["decomposition_identity_gate_passed"] = (
        max_identity_residual <= 1.0e-12
    )
    report = {
        "contract": base_contract,
        "status": "complete",
        "classification": "pending_temporal_precedence_and_same_state_review",
        "cohort": cohort,
        "decomposition_summary": decomposition_summary,
        "path_summary": path_summary,
        "runtime_seconds": time.perf_counter() - started,
    }
    _write_csv(args.output_dir / "decomposition.csv", decomposition_rows)
    _write_csv(
        args.output_dir / "decomposition_summary.csv",
        decomposition_summary,
    )
    np.savez_compressed(
        args.output_dir / "decomposition_vectors.npz",
        row_id=np.asarray(vector_row_ids),
        **{
            name: np.stack(values).astype(np.float64)
            for name, values in vector_bank.items()
        },
    )
    _write_csv(args.output_dir / "direct_composed_paths.csv", path_rows)
    _write_csv(args.output_dir / "direct_composed_summary.csv", path_summary)
    np.savez_compressed(
        args.output_dir / "direct_composed_endpoints.npz",
        row_id=np.asarray(path_row_ids),
        conservative=np.stack(path_endpoints).astype(np.float64),
    )
    (args.output_dir / "report.json").write_text(
        json.dumps(json_ready(report), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        json.dumps(
            json_ready(
                {
                    "status": report["status"],
                    "cohort": cohort,
                    "truth_replay_gate": base_contract["truth_replay_gate"],
                    "max_decomposition_identity_residual": (max_identity_residual),
                }
            ),
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return report


def main(argv: Sequence[str] | None = None) -> None:
    report = run(parse_args(argv))
    if report.get("status") != "complete":
        raise SystemExit(f"diagnostic failed with status={report.get('status')}")


if __name__ == "__main__":
    main()
