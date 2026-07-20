#!/usr/bin/env python3
"""Run the bounded analytic preflight for the 1D latent-forecast pilot.

This entry point deliberately contains no learned model or optimizer.  It uses
the frozen 384/64/64 trajectory split and stride-compatible frames from the
selected stride-8 Line-1 dataset to test whether privileged front coordinates
and matched current-state front speeds improve reconstruction and conditional-
future ambiguity before any learned front encoder is authorized.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utility.time_dependent_no.euler1d_data import (
    Euler1DNPZ,
    conservative_to_primitive_np,
    load_euler1d_npz,
    primitive_to_conservative_np,
)
from utility.time_dependent_no.latent_forecast import (
    Euler1DFrontSet,
    LinearCodePCA,
    OracleFrontKinematicPOD,
    OracleFrontPOD,
    VolumeWeightedPOD,
    cell_edges_from_centers,
    conditional_future_diagnostics,
    decode_registered_state,
    extract_euler1d_front_kinematics,
    extract_euler1d_fronts,
    fixed_scale_relative_l2,
    front_reconstruction_metrics,
    grouped_bootstrap_median_ratio,
    register_conservative_state,
)


SCHEMA_VERSION = "euler1d_latent_forecast_preflight_v2"
EXPECTED_SHAPE = (512, 101, 256, 3)
SPLIT_SEED = 20260707
TRAIN_CASES = 384
VALIDATION_CASES = 64
TEST_CASES = 64
STEP_STRIDE = 8
ENERGY_FRACTION = 0.995
FRONT_SCALARS = 12
KINEMATIC_SCALARS = 3
NEIGHBORS = 8
BOOTSTRAP_REPETITIONS = 1000
BOOTSTRAP_SEED = 20260720


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser(
        "preflight",
        help="Run the zero-training POD/oracle-front/kinematic diagnostic.",
    )
    preflight.add_argument("--data-path", type=Path, required=True)
    preflight.add_argument(
        "--expected-data-sha256",
        required=True,
        help="Required SHA256 of the frozen Euler1D NPZ artifact.",
    )
    preflight.add_argument("--output-path", type=Path, required=True)
    preflight.add_argument(
        "--step-stride",
        type=int,
        choices=(STEP_STRIDE,),
        default=STEP_STRIDE,
        help="Frozen saved-frame stride; only the selected stride 8 is eligible.",
    )
    preflight.add_argument(
        "--bootstrap-repetitions",
        type=int,
        default=BOOTSTRAP_REPETITIONS,
        help="Trajectory-grouped bootstrap draws (minimum 100).",
    )
    preflight.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing JSON artifact at --output-path.",
    )
    return parser.parse_args(argv)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.view(np.uint8)).hexdigest()


def _validate_sha256(value: str) -> str:
    normalized = value.strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(
            "--expected-data-sha256 must be exactly 64 hexadecimal characters"
        )
    return normalized


def _split_cases(source: Euler1DNPZ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    requested = TRAIN_CASES + VALIDATION_CASES + TEST_CASES
    if source.num_cases != requested:
        raise ValueError(
            f"preflight requires exactly {requested} cases, found {source.num_cases}"
        )
    permutation = np.random.default_rng(SPLIT_SEED).permutation(source.num_cases)
    train_end = TRAIN_CASES
    validation_end = train_end + VALIDATION_CASES
    train = np.sort(permutation[:train_end]).astype(np.int64)
    validation = np.sort(permutation[train_end:validation_end]).astype(np.int64)
    test = np.sort(permutation[validation_end:requested]).astype(np.int64)
    return train, validation, test


def _validate_source(
    source: Euler1DNPZ, step_stride: int
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if source.data.shape != EXPECTED_SHAPE:
        raise ValueError(
            f"preflight requires data shape {EXPECTED_SHAPE}, found {source.data.shape}"
        )
    if not math.isclose(source.gamma, 1.4, rel_tol=0.0, abs_tol=5.0e-7):
        raise ValueError(f"preflight requires gamma=1.4, found {source.gamma}")
    if not np.isfinite(source.data).all():
        raise ValueError("dataset contains nonfinite primitive states")
    if np.any(source.data[..., 0] <= 0.0) or np.any(source.data[..., 2] <= 0.0):
        raise ValueError("dataset contains nonpositive density or pressure")
    if not np.isfinite(source.x).all() or not np.isfinite(source.t).all():
        raise ValueError("dataset geometry or saved times contain nonfinite values")
    if not np.all(np.diff(source.x.astype(np.float64), axis=1) > 0.0):
        raise ValueError("every case must have strictly increasing cell centers")
    if not np.all(np.diff(source.t.astype(np.float64), axis=1) > 0.0):
        raise ValueError("every case must have strictly increasing saved times")
    if not np.array_equal(source.t, np.broadcast_to(source.t[0], source.t.shape)):
        raise ValueError("all cases must share the same saved times")

    case_edges = np.stack(
        [cell_edges_from_centers(coordinates) for coordinates in source.x]
    )
    cell_volumes = np.diff(case_edges, axis=1)
    reference_volume = cell_volumes[0]
    volume_absolute_deviation = np.abs(cell_volumes - reference_volume[None])
    volume_relative_deviation = volume_absolute_deviation / np.maximum(
        np.abs(reference_volume[None]), 1.0e-30
    )
    if float(np.max(volume_relative_deviation)) > 2.0e-5:
        raise ValueError("all cases must share the same cell-volume sequence")
    domain_lengths = case_edges[:, -1] - case_edges[:, 0]
    geometry_diagnostics = {
        "coordinates_identical": bool(
            np.array_equal(source.x, np.broadcast_to(source.x[0], source.x.shape))
        ),
        "left_edge_min": float(np.min(case_edges[:, 0])),
        "left_edge_max": float(np.max(case_edges[:, 0])),
        "right_edge_min": float(np.min(case_edges[:, -1])),
        "right_edge_max": float(np.max(case_edges[:, -1])),
        "domain_length_min": float(np.min(domain_lengths)),
        "domain_length_max": float(np.max(domain_lengths)),
        "representative_volume_case": 0,
        "maximum_volume_absolute_deviation": float(np.max(volume_absolute_deviation)),
        "maximum_volume_relative_deviation": float(np.max(volume_relative_deviation)),
        "accepted_volume_relative_tolerance": 2.0e-5,
    }

    frame_ids = np.arange(0, source.num_frames, step_stride, dtype=np.int64)
    if frame_ids.size < 3 or frame_ids[-1] != 96:
        raise ValueError("stride-8 preflight requires compatible endpoints 0:8:96")
    if not np.all(np.diff(frame_ids) == step_stride):
        raise ValueError("selected saved frames are not stride-compatible")
    return frame_ids, reference_volume, geometry_diagnostics


def _case_frames(
    source: Euler1DNPZ,
    case_ids: np.ndarray,
    frame_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    primitive = np.asarray(
        source.data[case_ids[:, None], frame_ids[None, :]], dtype=np.float64
    )
    coordinates = np.broadcast_to(
        source.x[case_ids, None, :].astype(np.float64),
        primitive.shape[:-1],
    ).copy()
    return primitive, coordinates


def _fronts_reshape(
    fronts: Euler1DFrontSet,
    leading_shape: tuple[int, ...],
) -> Euler1DFrontSet:
    return Euler1DFrontSet(
        position_fraction=np.asarray(fronts.position_fraction).reshape(
            *leading_shape, 3
        ),
        thickness_fraction=np.asarray(fronts.thickness_fraction).reshape(
            *leading_shape, 3
        ),
        signed_strength=np.asarray(fronts.signed_strength).reshape(*leading_shape, 3),
        valid=np.asarray(fronts.valid).reshape(*leading_shape, 3),
        score=np.asarray(fronts.score).reshape(*leading_shape, 3),
    )


def _fronts_slice(fronts: Euler1DFrontSet, index: Any) -> Euler1DFrontSet:
    return Euler1DFrontSet(
        position_fraction=np.asarray(fronts.position_fraction)[index],
        thickness_fraction=np.asarray(fronts.thickness_fraction)[index],
        signed_strength=np.asarray(fronts.signed_strength)[index],
        valid=np.asarray(fronts.valid)[index],
        score=np.asarray(fronts.score)[index],
    )


def _fronts_flatten(fronts: Euler1DFrontSet) -> Euler1DFrontSet:
    return Euler1DFrontSet(
        position_fraction=np.asarray(fronts.position_fraction).reshape(-1, 3),
        thickness_fraction=np.asarray(fronts.thickness_fraction).reshape(-1, 3),
        signed_strength=np.asarray(fronts.signed_strength).reshape(-1, 3),
        valid=np.asarray(fronts.valid).reshape(-1, 3),
        score=np.asarray(fronts.score).reshape(-1, 3),
    )


def _front_reconstruction_errors(
    truth: Euler1DFrontSet,
    prediction: Euler1DFrontSet,
    *,
    num_cells: int,
) -> dict[str, np.ndarray]:
    truth_valid = np.asarray(truth.valid, dtype=bool).reshape(-1, 3)
    prediction_valid = np.asarray(prediction.valid, dtype=bool).reshape(-1, 3)
    truth_position = np.asarray(truth.position_fraction).reshape(-1, 3)
    predicted_position = np.asarray(prediction.position_fraction).reshape(-1, 3)
    truth_strength = np.asarray(truth.signed_strength).reshape(-1, 3)
    predicted_strength = np.asarray(prediction.signed_strength).reshape(-1, 3)
    truth_thickness = np.asarray(truth.thickness_fraction).reshape(-1, 3)
    predicted_thickness = np.asarray(prediction.thickness_fraction).reshape(-1, 3)

    output = {
        "position_cells": np.full(truth_valid.shape[0], np.nan),
        "strength_relative": np.full(truth_valid.shape[0], np.nan),
        "thickness_relative": np.full(truth_valid.shape[0], np.nan),
    }
    for sample in range(truth_valid.shape[0]):
        slots = truth_valid[sample]
        if not np.any(slots):
            continue
        common = slots & prediction_valid[sample]
        position_error = np.full(3, float(num_cells))
        strength_error = np.ones(3)
        thickness_error = np.ones(3)
        position_error[common] = (
            np.abs(predicted_position[sample, common] - truth_position[sample, common])
            * num_cells
        )
        strength_error[common] = np.abs(
            predicted_strength[sample, common] - truth_strength[sample, common]
        ) / np.maximum(np.abs(truth_strength[sample, common]), 1.0e-8)
        thickness_error[common] = np.abs(
            predicted_thickness[sample, common] - truth_thickness[sample, common]
        ) / np.maximum(truth_thickness[sample, common], 1.0 / num_cells)
        output["position_cells"][sample] = float(
            np.sqrt(np.mean(position_error[slots] ** 2))
        )
        output["strength_relative"][sample] = float(
            np.sqrt(np.mean(strength_error[slots] ** 2))
        )
        output["thickness_relative"][sample] = float(
            np.sqrt(np.mean(thickness_error[slots] ** 2))
        )
    return output


def _front_thickness_ratios(
    truth: Euler1DFrontSet,
    prediction: Euler1DFrontSet,
    *,
    num_cells: int,
) -> np.ndarray:
    truth_valid = np.asarray(truth.valid, dtype=bool)
    prediction_valid = np.asarray(prediction.valid, dtype=bool)
    common = truth_valid & prediction_valid
    if not np.any(common):
        return np.empty(0, dtype=np.float64)
    return np.asarray(prediction.thickness_fraction)[common] / np.maximum(
        np.asarray(truth.thickness_fraction)[common], 1.0 / num_cells
    )


def _front_neighbor_ambiguity(
    train_future: Euler1DFrontSet,
    query_future: Euler1DFrontSet,
    neighbor_indices: np.ndarray,
    neighbor_weights: np.ndarray,
    *,
    num_cells: int,
) -> dict[str, np.ndarray]:
    train_valid = np.asarray(train_future.valid, dtype=bool).reshape(-1, 3)
    query_valid = np.asarray(query_future.valid, dtype=bool).reshape(-1, 3)
    train_position = np.asarray(train_future.position_fraction).reshape(-1, 3)
    query_position = np.asarray(query_future.position_fraction).reshape(-1, 3)
    train_strength = np.asarray(train_future.signed_strength).reshape(-1, 3)
    query_strength = np.asarray(query_future.signed_strength).reshape(-1, 3)
    train_thickness = np.asarray(train_future.thickness_fraction).reshape(-1, 3)
    query_thickness = np.asarray(query_future.thickness_fraction).reshape(-1, 3)
    output = {
        "position_cells": np.full(query_valid.shape[0], np.nan),
        "strength_relative": np.full(query_valid.shape[0], np.nan),
        "thickness_relative": np.full(query_valid.shape[0], np.nan),
    }
    for query_index in range(query_valid.shape[0]):
        slots = query_valid[query_index]
        if not np.any(slots):
            continue
        per_neighbor = {name: [] for name in output}
        for neighbor in neighbor_indices[query_index]:
            common = slots & train_valid[neighbor]
            position_error = np.full(3, float(num_cells))
            strength_error = np.ones(3)
            thickness_error = np.ones(3)
            position_error[common] = (
                np.abs(
                    train_position[neighbor, common]
                    - query_position[query_index, common]
                )
                * num_cells
            )
            strength_error[common] = np.abs(
                train_strength[neighbor, common] - query_strength[query_index, common]
            ) / np.maximum(np.abs(query_strength[query_index, common]), 1.0e-8)
            thickness_error[common] = np.abs(
                train_thickness[neighbor, common] - query_thickness[query_index, common]
            ) / np.maximum(query_thickness[query_index, common], 1.0 / num_cells)
            per_neighbor["position_cells"].append(
                float(np.mean(position_error[slots] ** 2))
            )
            per_neighbor["strength_relative"].append(
                float(np.mean(strength_error[slots] ** 2))
            )
            per_neighbor["thickness_relative"].append(
                float(np.mean(thickness_error[slots] ** 2))
            )
        for name, squared_error in per_neighbor.items():
            output[name][query_index] = float(
                np.sqrt(np.dot(neighbor_weights[query_index], squared_error))
            )
    return output


def _identity_code(
    conservative: np.ndarray,
    cell_volume: np.ndarray,
    component_scale: np.ndarray,
) -> np.ndarray:
    weight = np.sqrt(cell_volume / cell_volume.sum())[:, None]
    return (conservative / component_scale[None, None] * weight[None]).reshape(
        conservative.shape[0], -1
    )


def _conditioned_transition_code(
    latent_code: np.ndarray,
    source: Euler1DNPZ,
    case_ids: np.ndarray,
    start_frame_ids: np.ndarray,
    end_frame_ids: np.ndarray,
) -> np.ndarray:
    values = np.asarray(latent_code, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError("latent_code must have shape [cases, pairs, dimensions]")
    if values.shape[:2] != (case_ids.size, start_frame_ids.size):
        raise ValueError("latent_code does not align with cases and frame pairs")
    if start_frame_ids.shape != end_frame_ids.shape:
        raise ValueError("start and end frame arrays must share shape")
    case_edges = np.stack(
        [cell_edges_from_centers(source.x[case]) for case in case_ids]
    )
    static_context = np.concatenate(
        (
            source.left_states[case_ids].astype(np.float64),
            source.right_states[case_ids].astype(np.float64),
            case_edges[:, (0, -1)],
        ),
        axis=1,
    )
    repeated_context = np.broadcast_to(
        static_context[:, None, :],
        (case_ids.size, start_frame_ids.size, static_context.shape[1]),
    )
    dt = (
        source.t[case_ids[:, None], end_frame_ids[None, :]]
        - source.t[case_ids[:, None], start_frame_ids[None, :]]
    ).astype(np.float64)
    conditioned = np.concatenate((values, repeated_context, dt[..., None]), axis=-1)
    return conditioned.reshape(-1, conditioned.shape[-1])


def _summary(values: np.ndarray) -> dict[str, float | int | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"count": 0, "median": None, "p90": None, "p95": None}
    return {
        "count": int(finite.size),
        "median": float(np.median(finite)),
        "p90": float(np.quantile(finite, 0.90)),
        "p95": float(np.quantile(finite, 0.95)),
    }


def _bootstrap_ratio(
    numerator: np.ndarray,
    denominator: np.ndarray,
    groups: np.ndarray,
    *,
    repetitions: int,
    seed: int,
) -> dict[str, float | int] | None:
    first = np.asarray(numerator, dtype=np.float64)
    second = np.asarray(denominator, dtype=np.float64)
    group_values = np.asarray(groups)
    mask = np.isfinite(first) & np.isfinite(second)
    if np.count_nonzero(mask) < 2 or np.unique(group_values[mask]).size < 2:
        return None
    return grouped_bootstrap_median_ratio(
        first[mask],
        second[mask],
        group_values[mask],
        repetitions=repetitions,
        seed=seed,
    )


def _chart_diagnostics(fronts: Euler1DFrontSet) -> dict[str, Any]:
    valid = np.asarray(fronts.valid, dtype=bool).reshape(-1, 3)
    position = np.asarray(fronts.position_fraction).reshape(-1, 3)
    minimum = np.empty(valid.shape[0], dtype=np.float64)
    maximum = np.empty(valid.shape[0], dtype=np.float64)
    conditioned = np.empty(valid.shape[0], dtype=bool)
    for sample in range(valid.shape[0]):
        anchors = np.sort(position[sample, valid[sample]])
        reference = np.arange(1, anchors.size + 1, dtype=np.float64) / (
            anchors.size + 1
        )
        physical_knots = np.concatenate(([0.0], anchors, [1.0]))
        reference_knots = np.concatenate(([0.0], reference, [1.0]))
        slopes = np.diff(physical_knots) / np.diff(reference_knots)
        minimum[sample] = float(np.min(slopes))
        maximum[sample] = float(np.max(slopes))
        conditioned[sample] = minimum[sample] >= 0.25 and maximum[sample] <= 4.0
    return {
        "samples": int(valid.shape[0]),
        "minimum_slope": float(np.min(minimum)),
        "maximum_slope": float(np.max(maximum)),
        "conditioned_fraction": float(np.mean(conditioned)),
    }


def _gate(
    value: float | int | None, comparison: str, threshold: float
) -> dict[str, Any]:
    finite = value is not None and np.isfinite(float(value))
    passed = False
    if finite:
        numeric = float(value)
        if comparison == "<=":
            passed = numeric <= threshold
        elif comparison == ">=":
            passed = numeric >= threshold
        elif comparison == ">":
            passed = numeric > threshold
        elif comparison == "==":
            passed = numeric == threshold
        else:
            raise ValueError(f"unsupported comparison {comparison}")
    return {
        "passed": bool(passed),
        "value": value,
        "comparison": comparison,
        "threshold": threshold,
    }


def _history_not_materially_better_gate(
    ratio: dict[str, float | int] | None,
) -> dict[str, Any]:
    lower_bound = None if ratio is None else float(ratio["ci95_lower"])
    return _gate(lower_bound, ">", 0.80)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    return value


def _write_artifact(path: Path, artifact: dict[str, Any], *, overwrite: bool) -> None:
    if path.suffix.lower() != ".json":
        raise ValueError("--output-path must name one .json artifact")
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"output already exists: {path.name}; pass --overwrite to replace it"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        _json_ready(artifact),
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(encoded + "\n", encoding="utf-8")
    temporary.replace(path)


def _full_rank_remap(
    conservative: np.ndarray,
    primitive: np.ndarray,
    coordinates: np.ndarray,
    fronts: Euler1DFrontSet,
) -> tuple[np.ndarray, list[dict[str, float | int]]]:
    decoded = np.empty_like(conservative)
    chart: list[dict[str, float | int]] = []
    for sample in range(conservative.shape[0]):
        sample_front = _fronts_slice(fronts, sample)
        registered, diagnostics = register_conservative_state(
            conservative[sample], coordinates[sample], sample_front
        )
        decoded[sample] = decode_registered_state(
            registered, coordinates[sample], sample_front
        )
        chart.append(diagnostics)
    return decoded, chart


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    if args.bootstrap_repetitions < 100:
        raise ValueError("--bootstrap-repetitions must be at least 100")
    expected_hash = _validate_sha256(args.expected_data_sha256)
    actual_hash = _sha256_file(args.data_path)
    if actual_hash != expected_hash:
        raise ValueError(
            "dataset SHA256 mismatch; refusing to evaluate an unregistered artifact"
        )
    source = load_euler1d_npz(args.data_path)
    frame_ids, cell_volume, geometry_diagnostics = _validate_source(
        source, args.step_stride
    )
    train_cases, validation_cases, test_cases = _split_cases(source)
    component_scale = np.array([1.0, 1.0, 1.0 / (source.gamma - 1.0)], dtype=np.float64)

    train_primitive, train_x = _case_frames(source, train_cases, frame_ids)
    validation_primitive, validation_x = _case_frames(
        source, validation_cases, frame_ids
    )
    train_conservative = primitive_to_conservative_np(train_primitive, source.gamma)
    validation_conservative = primitive_to_conservative_np(
        validation_primitive, source.gamma
    )
    train_shape = train_primitive.shape[:2]
    validation_shape = validation_primitive.shape[:2]
    flat_train_primitive = train_primitive.reshape(-1, source.num_cells, 3)
    flat_validation_primitive = validation_primitive.reshape(-1, source.num_cells, 3)
    flat_train_conservative = train_conservative.reshape(-1, source.num_cells, 3)
    flat_validation_conservative = validation_conservative.reshape(
        -1, source.num_cells, 3
    )
    flat_train_x = train_x.reshape(-1, source.num_cells)
    flat_validation_x = validation_x.reshape(-1, source.num_cells)
    validation_groups_snapshot = np.repeat(validation_cases, frame_ids.size)

    raw_pod = VolumeWeightedPOD.fit(
        flat_train_conservative,
        cell_volume,
        component_scale,
        energy_fraction=ENERGY_FRACTION,
    )
    pod_rank = raw_pod.rank
    compression_limit = 3 * source.num_cells / 4
    compression_eligible = (
        FRONT_SCALARS + KINEMATIC_SCALARS + 4 <= pod_rank <= compression_limit
    )

    train_fronts_flat = extract_euler1d_fronts(
        flat_train_primitive, flat_train_x, gamma=source.gamma
    )
    validation_fronts_flat = extract_euler1d_fronts(
        flat_validation_primitive, flat_validation_x, gamma=source.gamma
    )
    all_valid = np.concatenate(
        (
            np.asarray(train_fronts_flat.valid).reshape(-1, 3),
            np.asarray(validation_fronts_flat.valid).reshape(-1, 3),
        ),
        axis=0,
    )
    extraction = {
        "samples": int(all_valid.shape[0]),
        "any_pressure_front_fraction": float(np.mean(np.any(all_valid[:, :2], axis=1))),
        "any_front_fraction": float(np.mean(np.any(all_valid, axis=1))),
        "slot_valid_fraction": np.mean(all_valid, axis=0).tolist(),
        "slot_names": ["pressure_left", "pressure_right", "contact"],
    }
    chart_fronts = Euler1DFrontSet(
        position_fraction=np.concatenate(
            (
                np.asarray(train_fronts_flat.position_fraction).reshape(-1, 3),
                np.asarray(validation_fronts_flat.position_fraction).reshape(-1, 3),
            )
        ),
        thickness_fraction=np.concatenate(
            (
                np.asarray(train_fronts_flat.thickness_fraction).reshape(-1, 3),
                np.asarray(validation_fronts_flat.thickness_fraction).reshape(-1, 3),
            )
        ),
        signed_strength=np.concatenate(
            (
                np.asarray(train_fronts_flat.signed_strength).reshape(-1, 3),
                np.asarray(validation_fronts_flat.signed_strength).reshape(-1, 3),
            )
        ),
        valid=all_valid,
        score=np.zeros_like(all_valid, dtype=np.float64),
    )
    chart_diagnostics = _chart_diagnostics(chart_fronts)

    gates: dict[str, dict[str, Any]] = {
        "compression_budget_minimum": _gate(
            pod_rank, ">=", FRONT_SCALARS + KINEMATIC_SCALARS + 4
        ),
        "compression_budget_useful": _gate(pod_rank, "<=", compression_limit),
        "front_extraction_completion": _gate(
            extraction["any_pressure_front_fraction"], ">=", 0.95
        ),
        "chart_conditioning": _gate(
            chart_diagnostics["conditioned_fraction"], "==", 1.0
        ),
    }

    artifact: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "stage": "analytic_preflight",
        "data": {
            "basename": args.data_path.name,
            "sha256": actual_hash,
            "saved_time_sha256": _array_sha256(source.t),
            "selected_saved_time_sha256": _array_sha256(source.t[:, frame_ids]),
            "shape": list(source.data.shape),
            "gamma": source.gamma,
            "face_flux_integral_present": source.face_flux_integral is not None,
            "geometry": geometry_diagnostics,
        },
        "split": {
            "seed": SPLIT_SEED,
            "train_count": TRAIN_CASES,
            "validation_count": VALIDATION_CASES,
            "test_count": TEST_CASES,
            "train_case_ids": train_cases.tolist(),
            "validation_case_ids": validation_cases.tolist(),
            "test_case_ids": test_cases.tolist(),
            "train_case_sha256": _array_sha256(train_cases),
            "validation_case_sha256": _array_sha256(validation_cases),
            "test_case_sha256": _array_sha256(test_cases),
        },
        "frame_contract": {
            "step_stride": args.step_stride,
            "selected_frame_ids": frame_ids.tolist(),
            "pair_start_frame_ids": frame_ids[:-1].tolist(),
            "pair_end_frame_ids": frame_ids[1:].tolist(),
            "discarded_tail_frame_ids": list(
                range(int(frame_ids[-1] + 1), source.num_frames)
            ),
            "pairs_per_trajectory": int(frame_ids.size - 1),
        },
        "representation_contract": {
            "physical_scalar_size": int(3 * source.num_cells),
            "pod_energy_fraction": ENERGY_FRACTION,
            "pod_rank": pod_rank,
            "pod_retained_energy": raw_pod.retained_energy,
            "oracle_front_scalars": FRONT_SCALARS,
            "oracle_kinematic_scalars": KINEMATIC_SCALARS,
            "oracle_registered_rank": pod_rank - FRONT_SCALARS
            if compression_eligible
            else None,
            "kinematic_oracle_registered_rank": (
                pod_rank - FRONT_SCALARS - KINEMATIC_SCALARS
                if compression_eligible
                else None
            ),
            "compression_eligible": compression_eligible,
            "component_scale": component_scale.tolist(),
        },
        "metrics": {
            "front_extraction": extraction,
            "chart": chart_diagnostics,
        },
        "gates": gates,
        "interventions": {
            "learned_training": False,
            "gpu_used": False,
            "rollout_launched": False,
            "decode_reencode": False,
            "clipping": False,
            "density_floor": False,
            "pressure_floor": False,
            "limiter": False,
            "projection": False,
            "reset": False,
            "truth_replacement": False,
            "analysis": False,
            "per_cycle_fit": False,
        },
        "code": {
            "entry_point": {
                "basename": Path(__file__).name,
                "sha256": _sha256_file(Path(__file__)),
            },
            "latent_module": {
                "basename": "latent_forecast.py",
                "sha256": _sha256_file(
                    Path(__file__).resolve().parents[2]
                    / "utility"
                    / "time_dependent_no"
                    / "latent_forecast.py"
                ),
            },
        },
    }

    required_gate_names = [
        "compression_budget_minimum",
        "compression_budget_useful",
        "front_extraction_completion",
        "chart_conditioning",
        "remap_front_retention",
        "remap_front_precision",
        "remap_thickness_preservation",
        "oracle_reconstruction_admissibility",
        "oracle_reconstruction_front_recall",
        "oracle_reconstruction_front_precision",
        "oracle_state_reconstruction_noninferiority",
        "oracle_front_position_reconstruction",
        "oracle_front_strength_reconstruction",
        "oracle_front_thickness_reconstruction",
        "oracle_closure_median",
        "oracle_closure_p90",
        "oracle_encoded_closure_median",
        "oracle_encoded_closure_p90",
        "oracle_encoded_ambiguity_ratio",
        "oracle_latent_neighborhood_median",
        "oracle_latent_neighborhood_p90",
        "oracle_state_ambiguity_ratio",
        "oracle_front_position_ambiguity_ratio",
        "oracle_front_strength_ambiguity_ratio",
        "oracle_front_thickness_ambiguity_ratio",
        "fixed_size_history_not_materially_better",
        "fixed_size_history_encoded_not_materially_better",
    ]
    kinematic_gate_names = [
        "kinematic_speed_completion",
        "kinematic_reconstruction_admissibility",
        "kinematic_state_reconstruction_noninferiority",
        "kinematic_front_position_reconstruction_noninferiority",
        "kinematic_front_strength_reconstruction_noninferiority",
        "kinematic_front_thickness_reconstruction_noninferiority",
        "kinematic_state_ambiguity_improvement",
        "kinematic_encoded_ambiguity_improvement",
        "kinematic_front_position_ambiguity_noninferiority",
        "kinematic_front_strength_ambiguity_noninferiority",
        "kinematic_front_thickness_ambiguity_noninferiority",
        "kinematic_history_not_materially_better",
        "kinematic_history_encoded_not_materially_better",
    ]
    required_gate_names.extend(kinematic_gate_names)

    if compression_eligible:
        oracle = OracleFrontPOD.fit(
            flat_train_primitive,
            flat_train_x,
            cell_volume,
            component_scale,
            total_size=pod_rank,
            gamma=source.gamma,
        )
        kinematic_oracle = OracleFrontKinematicPOD.fit(
            flat_train_primitive,
            flat_train_x,
            cell_volume,
            component_scale,
            total_size=pod_rank,
            gamma=source.gamma,
        )

        full_rank_decoded, full_rank_chart = _full_rank_remap(
            flat_validation_conservative,
            flat_validation_primitive,
            flat_validation_x,
            validation_fronts_flat,
        )
        full_rank_primitive = conservative_to_primitive_np(
            full_rank_decoded, source.gamma
        )
        full_rank_fronts = extract_euler1d_fronts(
            full_rank_primitive, flat_validation_x, gamma=source.gamma
        )
        remap_state_error = fixed_scale_relative_l2(
            full_rank_decoded,
            flat_validation_conservative,
            cell_volume,
            component_scale,
        )
        remap_front_metrics = front_reconstruction_metrics(
            validation_fronts_flat,
            full_rank_fronts,
            num_cells=source.num_cells,
        )
        remap_thickness_ratio = _front_thickness_ratios(
            validation_fronts_flat,
            full_rank_fronts,
            num_cells=source.num_cells,
        )
        remap_thickness_distortion = np.maximum(
            remap_thickness_ratio,
            1.0 / np.maximum(remap_thickness_ratio, 1.0e-12),
        )
        truth_front_count = int(np.count_nonzero(validation_fronts_flat.valid))
        predicted_front_count = int(np.count_nonzero(full_rank_fronts.valid))
        common_front_count = int(remap_front_metrics["common_fronts"])
        remap_retention = (
            common_front_count / truth_front_count if truth_front_count > 0 else None
        )
        remap_precision = (
            common_front_count / predicted_front_count
            if predicted_front_count > 0
            else None
        )
        artifact["metrics"]["full_rank_remap"] = {
            "state_relative_l2": _summary(remap_state_error),
            "front": remap_front_metrics,
            "front_retention": remap_retention,
            "front_precision": remap_precision,
            "thickness_ratio": _summary(remap_thickness_ratio),
            "thickness_symmetric_distortion": _summary(remap_thickness_distortion),
            "minimum_chart_slope": min(
                float(item["minimum_chart_slope"]) for item in full_rank_chart
            ),
            "maximum_chart_slope": max(
                float(item["maximum_chart_slope"]) for item in full_rank_chart
            ),
        }
        gates["remap_front_retention"] = _gate(remap_retention, ">=", 0.95)
        gates["remap_front_precision"] = _gate(remap_precision, ">=", 0.95)
        gates["remap_thickness_preservation"] = _gate(
            artifact["metrics"]["full_rank_remap"]["thickness_symmetric_distortion"][
                "p95"
            ],
            "<=",
            1.05,
        )

        validation_pod_code = raw_pod.encode(flat_validation_conservative)
        validation_pod_decoded = raw_pod.decode(validation_pod_code)
        validation_oracle_code = oracle.encode(
            flat_validation_primitive, flat_validation_x
        )
        validation_kinematic_code = kinematic_oracle.encode(
            flat_validation_primitive, flat_validation_x
        )
        validation_oracle_decoded = oracle.decode_conservative(
            validation_oracle_code, flat_validation_x
        )
        validation_kinematic_decoded = kinematic_oracle.decode_conservative(
            validation_kinematic_code, flat_validation_x
        )
        validation_pod_primitive = conservative_to_primitive_np(
            validation_pod_decoded, source.gamma
        )
        validation_oracle_primitive = conservative_to_primitive_np(
            validation_oracle_decoded, source.gamma
        )
        validation_kinematic_primitive = conservative_to_primitive_np(
            validation_kinematic_decoded, source.gamma
        )
        validation_pod_fronts = extract_euler1d_fronts(
            validation_pod_primitive, flat_validation_x, gamma=source.gamma
        )
        validation_oracle_fronts = extract_euler1d_fronts(
            validation_oracle_primitive, flat_validation_x, gamma=source.gamma
        )
        validation_kinematic_fronts = extract_euler1d_fronts(
            validation_kinematic_primitive,
            flat_validation_x,
            gamma=source.gamma,
        )
        pod_state_error = fixed_scale_relative_l2(
            validation_pod_decoded,
            flat_validation_conservative,
            cell_volume,
            component_scale,
        )
        oracle_state_error = fixed_scale_relative_l2(
            validation_oracle_decoded,
            flat_validation_conservative,
            cell_volume,
            component_scale,
        )
        kinematic_state_error = fixed_scale_relative_l2(
            validation_kinematic_decoded,
            flat_validation_conservative,
            cell_volume,
            component_scale,
        )
        pod_front_error = _front_reconstruction_errors(
            validation_fronts_flat,
            validation_pod_fronts,
            num_cells=source.num_cells,
        )
        oracle_front_error = _front_reconstruction_errors(
            validation_fronts_flat,
            validation_oracle_fronts,
            num_cells=source.num_cells,
        )
        kinematic_front_error = _front_reconstruction_errors(
            validation_fronts_flat,
            validation_kinematic_fronts,
            num_cells=source.num_cells,
        )
        reconstruction_ratios: dict[str, Any] = {
            "state": _bootstrap_ratio(
                oracle_state_error,
                pod_state_error,
                validation_groups_snapshot,
                repetitions=args.bootstrap_repetitions,
                seed=BOOTSTRAP_SEED,
            )
        }
        for offset, name in enumerate(
            ("position_cells", "strength_relative", "thickness_relative"), start=1
        ):
            reconstruction_ratios[name] = _bootstrap_ratio(
                oracle_front_error[name],
                pod_front_error[name],
                validation_groups_snapshot,
                repetitions=args.bootstrap_repetitions,
                seed=BOOTSTRAP_SEED + offset,
            )
        kinematic_reconstruction_ratios: dict[str, Any] = {
            "state": _bootstrap_ratio(
                kinematic_state_error,
                oracle_state_error,
                validation_groups_snapshot,
                repetitions=args.bootstrap_repetitions,
                seed=BOOTSTRAP_SEED + 30,
            )
        }
        for offset, name in enumerate(
            ("position_cells", "strength_relative", "thickness_relative"),
            start=31,
        ):
            kinematic_reconstruction_ratios[name] = _bootstrap_ratio(
                kinematic_front_error[name],
                oracle_front_error[name],
                validation_groups_snapshot,
                repetitions=args.bootstrap_repetitions,
                seed=BOOTSTRAP_SEED + offset,
            )
        oracle_case_admissible = (
            np.isfinite(validation_oracle_primitive).all(axis=(1, 2))
            & (validation_oracle_primitive[..., 0] > 0.0).all(axis=1)
            & (validation_oracle_primitive[..., 2] > 0.0).all(axis=1)
        )
        kinematic_case_admissible = (
            np.isfinite(validation_kinematic_primitive).all(axis=(1, 2))
            & (validation_kinematic_primitive[..., 0] > 0.0).all(axis=1)
            & (validation_kinematic_primitive[..., 2] > 0.0).all(axis=1)
        )
        validation_kinematics = extract_euler1d_front_kinematics(
            flat_validation_primitive,
            flat_validation_x,
            fronts=validation_fronts_flat,
            gamma=source.gamma,
        )
        valid_speed = np.asarray(validation_kinematics.valid, dtype=bool)
        valid_speed_count = int(np.count_nonzero(valid_speed))
        speed_completion = (
            float(np.mean(np.isfinite(validation_kinematics.speed[valid_speed])))
            if valid_speed_count > 0
            else None
        )
        oracle_front_metrics = front_reconstruction_metrics(
            validation_fronts_flat,
            validation_oracle_fronts,
            num_cells=source.num_cells,
        )
        oracle_truth_fronts = int(oracle_front_metrics["truth_fronts"])
        oracle_predicted_fronts = int(oracle_front_metrics["predicted_fronts"])
        oracle_common_fronts = int(oracle_front_metrics["common_fronts"])
        oracle_front_recall = (
            oracle_common_fronts / oracle_truth_fronts
            if oracle_truth_fronts > 0
            else None
        )
        oracle_front_precision = (
            oracle_common_fronts / oracle_predicted_fronts
            if oracle_predicted_fronts > 0
            else None
        )
        artifact["metrics"]["reconstruction"] = {
            "pod": {
                "state_relative_l2": _summary(pod_state_error),
                "front": front_reconstruction_metrics(
                    validation_fronts_flat,
                    validation_pod_fronts,
                    num_cells=source.num_cells,
                ),
                "front_error": {
                    name: _summary(values) for name, values in pod_front_error.items()
                },
            },
            "oracle": {
                "state_relative_l2": _summary(oracle_state_error),
                "front": oracle_front_metrics,
                "front_error": {
                    name: _summary(values)
                    for name, values in oracle_front_error.items()
                },
                "reconstruction_admissible_fraction": float(
                    np.mean(oracle_case_admissible)
                ),
                "front_recall": oracle_front_recall,
                "front_precision": oracle_front_precision,
            },
            "kinematic_oracle": {
                "state_relative_l2": _summary(kinematic_state_error),
                "front": front_reconstruction_metrics(
                    validation_fronts_flat,
                    validation_kinematic_fronts,
                    num_cells=source.num_cells,
                ),
                "front_error": {
                    name: _summary(values)
                    for name, values in kinematic_front_error.items()
                },
                "reconstruction_admissible_fraction": float(
                    np.mean(kinematic_case_admissible)
                ),
            },
            "oracle_over_pod_grouped_bootstrap": reconstruction_ratios,
            "kinematic_over_oracle_grouped_bootstrap": (
                kinematic_reconstruction_ratios
            ),
        }
        artifact["metrics"]["front_kinematics"] = {
            "definition": "pressure=least_squares_RH;contact=mean_plateau_velocity",
            "uses_trajectory_history": False,
            "valid_speed_count": valid_speed_count,
            "completion_fraction": speed_completion,
            "slot_names": ["pressure_left", "pressure_right", "contact"],
            "speed": [
                _summary(
                    validation_kinematics.speed[
                        np.asarray(validation_kinematics.valid)[:, slot], slot
                    ]
                )
                for slot in range(3)
            ],
            "consistency_residual": [
                _summary(
                    validation_kinematics.consistency_residual[
                        np.asarray(validation_kinematics.valid)[:, slot], slot
                    ]
                )
                for slot in range(3)
            ],
        }
        gates["oracle_reconstruction_admissibility"] = _gate(
            float(np.mean(oracle_case_admissible)), "==", 1.0
        )
        gates["kinematic_speed_completion"] = _gate(speed_completion, "==", 1.0)
        gates["kinematic_reconstruction_admissibility"] = _gate(
            float(np.mean(kinematic_case_admissible)), "==", 1.0
        )
        gates["kinematic_state_reconstruction_noninferiority"] = _gate(
            kinematic_reconstruction_ratios["state"]["ci95_upper"],
            "<=",
            1.05,
        )
        for metric, gate_name in (
            (
                "position_cells",
                "kinematic_front_position_reconstruction_noninferiority",
            ),
            (
                "strength_relative",
                "kinematic_front_strength_reconstruction_noninferiority",
            ),
            (
                "thickness_relative",
                "kinematic_front_thickness_reconstruction_noninferiority",
            ),
        ):
            gates[gate_name] = _gate(
                kinematic_reconstruction_ratios[metric]["ci95_upper"],
                "<=",
                1.05,
            )
        gates["oracle_reconstruction_front_recall"] = _gate(
            oracle_front_recall, ">=", 0.95
        )
        gates["oracle_reconstruction_front_precision"] = _gate(
            oracle_front_precision, ">=", 0.95
        )
        gates["oracle_state_reconstruction_noninferiority"] = _gate(
            None
            if reconstruction_ratios["state"] is None
            else reconstruction_ratios["state"]["ci95_upper"],
            "<=",
            1.05,
        )
        for metric, gate_name in (
            ("position_cells", "oracle_front_position_reconstruction"),
            ("strength_relative", "oracle_front_strength_reconstruction"),
            ("thickness_relative", "oracle_front_thickness_reconstruction"),
        ):
            ratio = reconstruction_ratios[metric]
            gates[gate_name] = _gate(
                None if ratio is None else ratio["ci95_upper"], "<=", 0.80
            )

        train_pod_codes = raw_pod.encode(flat_train_conservative).reshape(
            *train_shape, pod_rank
        )
        validation_pod_codes = validation_pod_code.reshape(*validation_shape, pod_rank)
        train_oracle_codes = oracle.encode(flat_train_primitive, flat_train_x).reshape(
            *train_shape, pod_rank
        )
        validation_oracle_codes = validation_oracle_code.reshape(
            *validation_shape, pod_rank
        )
        train_kinematic_codes = kinematic_oracle.encode(
            flat_train_primitive, flat_train_x
        ).reshape(*train_shape, pod_rank)
        validation_kinematic_codes = validation_kinematic_code.reshape(
            *validation_shape, pod_rank
        )
        train_fronts = _fronts_reshape(train_fronts_flat, train_shape)
        validation_fronts = _fronts_reshape(validation_fronts_flat, validation_shape)

        train_current = train_conservative[:, :-1].reshape(-1, source.num_cells, 3)
        train_future = train_conservative[:, 1:].reshape(-1, source.num_cells, 3)
        validation_current = validation_conservative[:, :-1].reshape(
            -1, source.num_cells, 3
        )
        validation_future = validation_conservative[:, 1:].reshape(
            -1, source.num_cells, 3
        )
        validation_groups_pair = np.repeat(validation_cases, frame_ids.size - 1)
        train_front_future = _fronts_flatten(
            _fronts_slice(train_fronts, (slice(None), slice(1, None)))
        )
        validation_front_future = _fronts_flatten(
            _fronts_slice(validation_fronts, (slice(None), slice(1, None)))
        )

        pair_start_frame_ids = frame_ids[:-1]
        pair_end_frame_ids = frame_ids[1:]
        train_identity_current_code = _identity_code(
            train_current, cell_volume, component_scale
        ).reshape(train_cases.size, pair_start_frame_ids.size, -1)
        train_identity_future_code = _identity_code(
            train_future, cell_volume, component_scale
        ).reshape(train_cases.size, pair_start_frame_ids.size, -1)
        validation_identity_current_code = _identity_code(
            validation_current, cell_volume, component_scale
        ).reshape(validation_cases.size, pair_start_frame_ids.size, -1)
        validation_identity_future_code = _identity_code(
            validation_future, cell_volume, component_scale
        ).reshape(validation_cases.size, pair_start_frame_ids.size, -1)

        train_conditioned_identity_current = _conditioned_transition_code(
            train_identity_current_code,
            source,
            train_cases,
            pair_start_frame_ids,
            pair_end_frame_ids,
        )
        train_conditioned_identity_future = _conditioned_transition_code(
            train_identity_future_code,
            source,
            train_cases,
            pair_start_frame_ids,
            pair_end_frame_ids,
        )
        validation_conditioned_identity_current = _conditioned_transition_code(
            validation_identity_current_code,
            source,
            validation_cases,
            pair_start_frame_ids,
            pair_end_frame_ids,
        )
        validation_conditioned_identity_future = _conditioned_transition_code(
            validation_identity_future_code,
            source,
            validation_cases,
            pair_start_frame_ids,
            pair_end_frame_ids,
        )
        train_conditioned_pod_current = _conditioned_transition_code(
            train_pod_codes[:, :-1],
            source,
            train_cases,
            pair_start_frame_ids,
            pair_end_frame_ids,
        )
        train_conditioned_pod_future = _conditioned_transition_code(
            train_pod_codes[:, 1:],
            source,
            train_cases,
            pair_start_frame_ids,
            pair_end_frame_ids,
        )
        validation_conditioned_pod_current = _conditioned_transition_code(
            validation_pod_codes[:, :-1],
            source,
            validation_cases,
            pair_start_frame_ids,
            pair_end_frame_ids,
        )
        validation_conditioned_pod_future = _conditioned_transition_code(
            validation_pod_codes[:, 1:],
            source,
            validation_cases,
            pair_start_frame_ids,
            pair_end_frame_ids,
        )
        train_conditioned_oracle_current = _conditioned_transition_code(
            train_oracle_codes[:, :-1],
            source,
            train_cases,
            pair_start_frame_ids,
            pair_end_frame_ids,
        )
        train_conditioned_oracle_future = _conditioned_transition_code(
            train_oracle_codes[:, 1:],
            source,
            train_cases,
            pair_start_frame_ids,
            pair_end_frame_ids,
        )
        validation_conditioned_oracle_current = _conditioned_transition_code(
            validation_oracle_codes[:, :-1],
            source,
            validation_cases,
            pair_start_frame_ids,
            pair_end_frame_ids,
        )
        validation_conditioned_oracle_future = _conditioned_transition_code(
            validation_oracle_codes[:, 1:],
            source,
            validation_cases,
            pair_start_frame_ids,
            pair_end_frame_ids,
        )
        train_conditioned_kinematic_current = _conditioned_transition_code(
            train_kinematic_codes[:, :-1],
            source,
            train_cases,
            pair_start_frame_ids,
            pair_end_frame_ids,
        )
        train_conditioned_kinematic_future = _conditioned_transition_code(
            train_kinematic_codes[:, 1:],
            source,
            train_cases,
            pair_start_frame_ids,
            pair_end_frame_ids,
        )
        validation_conditioned_kinematic_current = _conditioned_transition_code(
            validation_kinematic_codes[:, :-1],
            source,
            validation_cases,
            pair_start_frame_ids,
            pair_end_frame_ids,
        )
        validation_conditioned_kinematic_future = _conditioned_transition_code(
            validation_kinematic_codes[:, 1:],
            source,
            validation_cases,
            pair_start_frame_ids,
            pair_end_frame_ids,
        )

        identity_closure = conditional_future_diagnostics(
            train_conditioned_identity_current,
            train_current,
            train_future,
            validation_conditioned_identity_current,
            validation_current,
            validation_future,
            cell_volume,
            component_scale,
            train_future_code=train_conditioned_identity_future,
            query_future_code=validation_conditioned_identity_future,
            neighbors=NEIGHBORS,
        )
        pod_closure = conditional_future_diagnostics(
            train_conditioned_pod_current,
            train_current,
            train_future,
            validation_conditioned_pod_current,
            validation_current,
            validation_future,
            cell_volume,
            component_scale,
            train_future_code=train_conditioned_pod_future,
            query_future_code=validation_conditioned_pod_future,
            neighbors=NEIGHBORS,
        )
        oracle_closure = conditional_future_diagnostics(
            train_conditioned_oracle_current,
            train_current,
            train_future,
            validation_conditioned_oracle_current,
            validation_current,
            validation_future,
            cell_volume,
            component_scale,
            train_future_code=train_conditioned_oracle_future,
            query_future_code=validation_conditioned_oracle_future,
            neighbors=NEIGHBORS,
        )
        kinematic_closure = conditional_future_diagnostics(
            train_conditioned_kinematic_current,
            train_current,
            train_future,
            validation_conditioned_kinematic_current,
            validation_current,
            validation_future,
            cell_volume,
            component_scale,
            train_future_code=train_conditioned_kinematic_future,
            query_future_code=validation_conditioned_kinematic_future,
            neighbors=NEIGHBORS,
        )
        pod_front_ambiguity = _front_neighbor_ambiguity(
            train_front_future,
            validation_front_future,
            pod_closure.neighbor_indices,
            pod_closure.neighbor_weights,
            num_cells=source.num_cells,
        )
        oracle_front_ambiguity = _front_neighbor_ambiguity(
            train_front_future,
            validation_front_future,
            oracle_closure.neighbor_indices,
            oracle_closure.neighbor_weights,
            num_cells=source.num_cells,
        )
        kinematic_front_ambiguity = _front_neighbor_ambiguity(
            train_front_future,
            validation_front_future,
            kinematic_closure.neighbor_indices,
            kinematic_closure.neighbor_weights,
            num_cells=source.num_cells,
        )
        if (
            pod_closure.encoded_future_ambiguity is None
            or oracle_closure.encoded_future_ambiguity is None
            or kinematic_closure.encoded_future_ambiguity is None
        ):
            raise RuntimeError("encoded-future closure diagnostics are required")
        ambiguity_ratios: dict[str, Any] = {
            "state": _bootstrap_ratio(
                oracle_closure.future_ambiguity,
                pod_closure.future_ambiguity,
                validation_groups_pair,
                repetitions=args.bootstrap_repetitions,
                seed=BOOTSTRAP_SEED + 10,
            ),
            "encoded_state": _bootstrap_ratio(
                oracle_closure.encoded_future_ambiguity,
                pod_closure.encoded_future_ambiguity,
                validation_groups_pair,
                repetitions=args.bootstrap_repetitions,
                seed=BOOTSTRAP_SEED + 15,
            ),
        }
        for offset, name in enumerate(
            ("position_cells", "strength_relative", "thickness_relative"), start=11
        ):
            ambiguity_ratios[name] = _bootstrap_ratio(
                oracle_front_ambiguity[name],
                pod_front_ambiguity[name],
                validation_groups_pair,
                repetitions=args.bootstrap_repetitions,
                seed=BOOTSTRAP_SEED + offset,
            )
        kinematic_ambiguity_ratios: dict[str, Any] = {
            "state": _bootstrap_ratio(
                kinematic_closure.future_ambiguity,
                oracle_closure.future_ambiguity,
                validation_groups_pair,
                repetitions=args.bootstrap_repetitions,
                seed=BOOTSTRAP_SEED + 40,
            ),
            "encoded_state": _bootstrap_ratio(
                kinematic_closure.encoded_future_ambiguity,
                oracle_closure.encoded_future_ambiguity,
                validation_groups_pair,
                repetitions=args.bootstrap_repetitions,
                seed=BOOTSTRAP_SEED + 41,
            ),
        }
        for offset, name in enumerate(
            ("position_cells", "strength_relative", "thickness_relative"),
            start=42,
        ):
            kinematic_ambiguity_ratios[name] = _bootstrap_ratio(
                kinematic_front_ambiguity[name],
                oracle_front_ambiguity[name],
                validation_groups_pair,
                repetitions=args.bootstrap_repetitions,
                seed=BOOTSTRAP_SEED + offset,
            )
        artifact["metrics"]["closure"] = {
            "neighbors": NEIGHBORS,
            "reference": "train_pairs",
            "query": "validation_pairs",
            "conditioning_contract": {
                "latent_state_plus_exogenous_context": True,
                "exogenous_scalars": [
                    "left_rho",
                    "left_velocity",
                    "left_pressure",
                    "right_rho",
                    "right_velocity",
                    "right_pressure",
                    "domain_left_edge",
                    "domain_right_edge",
                    "macro_dt",
                ],
                "exogenous_scalar_count": 9,
                "future_code_reuses_static_context": True,
            },
            "identity": identity_closure.summary(),
            "pod": pod_closure.summary(),
            "oracle": oracle_closure.summary(),
            "kinematic_oracle": kinematic_closure.summary(),
            "front_ambiguity": {
                "pod": {
                    name: _summary(values)
                    for name, values in pod_front_ambiguity.items()
                },
                "oracle": {
                    name: _summary(values)
                    for name, values in oracle_front_ambiguity.items()
                },
                "kinematic_oracle": {
                    name: _summary(values)
                    for name, values in kinematic_front_ambiguity.items()
                },
            },
            "oracle_over_pod_grouped_bootstrap": ambiguity_ratios,
            "kinematic_over_oracle_grouped_bootstrap": (kinematic_ambiguity_ratios),
        }
        identity_summary = identity_closure.summary()
        oracle_summary = oracle_closure.summary()
        gates["oracle_closure_median"] = _gate(
            oracle_summary["future_ambiguity_median"],
            "<=",
            max(0.10, 2.0 * float(identity_summary["future_ambiguity_median"])),
        )
        gates["oracle_closure_p90"] = _gate(
            oracle_summary["future_ambiguity_p90"],
            "<=",
            max(0.25, 2.0 * float(identity_summary["future_ambiguity_p90"])),
        )
        gates["oracle_encoded_closure_median"] = _gate(
            oracle_summary["encoded_future_ambiguity_median"],
            "<=",
            max(
                0.25,
                2.0 * float(identity_summary["encoded_future_ambiguity_median"]),
            ),
        )
        gates["oracle_encoded_closure_p90"] = _gate(
            oracle_summary["encoded_future_ambiguity_p90"],
            "<=",
            max(
                0.50,
                2.0 * float(identity_summary["encoded_future_ambiguity_p90"]),
            ),
        )
        gates["oracle_encoded_ambiguity_ratio"] = _gate(
            ambiguity_ratios["encoded_state"]["ci95_upper"],
            "<=",
            0.80,
        )
        gates["oracle_latent_neighborhood_median"] = _gate(
            oracle_summary["neighbor_radius_over_latent_step_median"],
            "<=",
            2.0,
        )
        gates["oracle_latent_neighborhood_p90"] = _gate(
            oracle_summary["neighbor_radius_over_latent_step_p90"],
            "<=",
            4.0,
        )
        gates["oracle_state_ambiguity_ratio"] = _gate(
            None
            if ambiguity_ratios["state"] is None
            else ambiguity_ratios["state"]["ci95_upper"],
            "<=",
            0.80,
        )
        for metric, gate_name in (
            ("position_cells", "oracle_front_position_ambiguity_ratio"),
            ("strength_relative", "oracle_front_strength_ambiguity_ratio"),
            ("thickness_relative", "oracle_front_thickness_ambiguity_ratio"),
        ):
            ratio = ambiguity_ratios[metric]
            gates[gate_name] = _gate(
                None if ratio is None else ratio["ci95_upper"], "<=", 0.80
            )
        gates["kinematic_state_ambiguity_improvement"] = _gate(
            kinematic_ambiguity_ratios["state"]["ci95_upper"],
            "<=",
            0.80,
        )
        gates["kinematic_encoded_ambiguity_improvement"] = _gate(
            kinematic_ambiguity_ratios["encoded_state"]["ci95_upper"],
            "<=",
            0.80,
        )
        for metric, gate_name in (
            (
                "position_cells",
                "kinematic_front_position_ambiguity_noninferiority",
            ),
            (
                "strength_relative",
                "kinematic_front_strength_ambiguity_noninferiority",
            ),
            (
                "thickness_relative",
                "kinematic_front_thickness_ambiguity_noninferiority",
            ),
        ):
            gates[gate_name] = _gate(
                kinematic_ambiguity_ratios[metric]["ci95_upper"],
                "<=",
                1.05,
            )

        train_history_raw = np.concatenate(
            (
                train_oracle_codes[:, 1:-1],
                train_oracle_codes[:, 1:-1] - train_oracle_codes[:, :-2],
            ),
            axis=-1,
        ).reshape(-1, 2 * pod_rank)
        train_history_future_raw = np.concatenate(
            (
                train_oracle_codes[:, 2:],
                train_oracle_codes[:, 2:] - train_oracle_codes[:, 1:-1],
            ),
            axis=-1,
        ).reshape(-1, 2 * pod_rank)
        validation_history_raw = np.concatenate(
            (
                validation_oracle_codes[:, 1:-1],
                validation_oracle_codes[:, 1:-1] - validation_oracle_codes[:, :-2],
            ),
            axis=-1,
        ).reshape(-1, 2 * pod_rank)
        validation_history_future_raw = np.concatenate(
            (
                validation_oracle_codes[:, 2:],
                validation_oracle_codes[:, 2:] - validation_oracle_codes[:, 1:-1],
            ),
            axis=-1,
        ).reshape(-1, 2 * pod_rank)
        history_pca = LinearCodePCA.fit(train_history_raw, output_size=pod_rank)
        train_history_code = history_pca.transform(train_history_raw)
        validation_history_code = history_pca.transform(validation_history_raw)
        train_history_future_code = history_pca.transform(train_history_future_raw)
        validation_history_future_code = history_pca.transform(
            validation_history_future_raw
        )
        history_start_frame_ids = frame_ids[1:-1]
        history_end_frame_ids = frame_ids[2:]
        train_conditioned_one_state_history_current = _conditioned_transition_code(
            train_oracle_codes[:, 1:-1],
            source,
            train_cases,
            history_start_frame_ids,
            history_end_frame_ids,
        )
        train_conditioned_one_state_history_future = _conditioned_transition_code(
            train_oracle_codes[:, 2:],
            source,
            train_cases,
            history_start_frame_ids,
            history_end_frame_ids,
        )
        validation_conditioned_one_state_history_current = _conditioned_transition_code(
            validation_oracle_codes[:, 1:-1],
            source,
            validation_cases,
            history_start_frame_ids,
            history_end_frame_ids,
        )
        validation_conditioned_one_state_history_future = _conditioned_transition_code(
            validation_oracle_codes[:, 2:],
            source,
            validation_cases,
            history_start_frame_ids,
            history_end_frame_ids,
        )
        train_conditioned_history_current = _conditioned_transition_code(
            train_history_code.reshape(
                train_cases.size, history_start_frame_ids.size, pod_rank
            ),
            source,
            train_cases,
            history_start_frame_ids,
            history_end_frame_ids,
        )
        train_conditioned_history_future = _conditioned_transition_code(
            train_history_future_code.reshape(
                train_cases.size, history_start_frame_ids.size, pod_rank
            ),
            source,
            train_cases,
            history_start_frame_ids,
            history_end_frame_ids,
        )
        validation_conditioned_history_current = _conditioned_transition_code(
            validation_history_code.reshape(
                validation_cases.size, history_start_frame_ids.size, pod_rank
            ),
            source,
            validation_cases,
            history_start_frame_ids,
            history_end_frame_ids,
        )
        validation_conditioned_history_future = _conditioned_transition_code(
            validation_history_future_code.reshape(
                validation_cases.size, history_start_frame_ids.size, pod_rank
            ),
            source,
            validation_cases,
            history_start_frame_ids,
            history_end_frame_ids,
        )
        history_train_current = train_conservative[:, 1:-1].reshape(
            -1, source.num_cells, 3
        )
        history_train_future = train_conservative[:, 2:].reshape(
            -1, source.num_cells, 3
        )
        history_validation_current = validation_conservative[:, 1:-1].reshape(
            -1, source.num_cells, 3
        )
        history_validation_future = validation_conservative[:, 2:].reshape(
            -1, source.num_cells, 3
        )
        one_state_history_subset = conditional_future_diagnostics(
            train_conditioned_one_state_history_current,
            history_train_current,
            history_train_future,
            validation_conditioned_one_state_history_current,
            history_validation_current,
            history_validation_future,
            cell_volume,
            component_scale,
            train_future_code=train_conditioned_one_state_history_future,
            query_future_code=validation_conditioned_one_state_history_future,
            neighbors=NEIGHBORS,
        )
        history_closure = conditional_future_diagnostics(
            train_conditioned_history_current,
            history_train_current,
            history_train_future,
            validation_conditioned_history_current,
            history_validation_current,
            history_validation_future,
            cell_volume,
            component_scale,
            train_future_code=train_conditioned_history_future,
            query_future_code=validation_conditioned_history_future,
            neighbors=NEIGHBORS,
        )
        validation_groups_history = np.repeat(validation_cases, frame_ids.size - 2)
        history_ratio = _bootstrap_ratio(
            history_closure.future_ambiguity,
            one_state_history_subset.future_ambiguity,
            validation_groups_history,
            repetitions=args.bootstrap_repetitions,
            seed=BOOTSTRAP_SEED + 20,
        )
        if (
            history_closure.encoded_future_ambiguity is None
            or one_state_history_subset.encoded_future_ambiguity is None
        ):
            raise RuntimeError("encoded-future history diagnostics are required")
        history_encoded_ratio = _bootstrap_ratio(
            history_closure.encoded_future_ambiguity,
            one_state_history_subset.encoded_future_ambiguity,
            validation_groups_history,
            repetitions=args.bootstrap_repetitions,
            seed=BOOTSTRAP_SEED + 21,
        )
        artifact["metrics"]["fixed_size_history"] = {
            "definition": "PCA_B([a_n, a_n-a_n_minus_1])",
            "retained_scalar_size": pod_rank,
            "one_state": one_state_history_subset.summary(),
            "history": history_closure.summary(),
            "history_over_one_state_grouped_bootstrap": history_ratio,
            "encoded_history_over_one_state_grouped_bootstrap": (history_encoded_ratio),
        }
        gates["fixed_size_history_not_materially_better"] = (
            _history_not_materially_better_gate(history_ratio)
        )
        gates["fixed_size_history_encoded_not_materially_better"] = (
            _history_not_materially_better_gate(history_encoded_ratio)
        )

        train_kinematic_history_raw = np.concatenate(
            (
                train_kinematic_codes[:, 1:-1],
                train_kinematic_codes[:, 1:-1] - train_kinematic_codes[:, :-2],
            ),
            axis=-1,
        ).reshape(-1, 2 * pod_rank)
        train_kinematic_history_future_raw = np.concatenate(
            (
                train_kinematic_codes[:, 2:],
                train_kinematic_codes[:, 2:] - train_kinematic_codes[:, 1:-1],
            ),
            axis=-1,
        ).reshape(-1, 2 * pod_rank)
        validation_kinematic_history_raw = np.concatenate(
            (
                validation_kinematic_codes[:, 1:-1],
                validation_kinematic_codes[:, 1:-1]
                - validation_kinematic_codes[:, :-2],
            ),
            axis=-1,
        ).reshape(-1, 2 * pod_rank)
        validation_kinematic_history_future_raw = np.concatenate(
            (
                validation_kinematic_codes[:, 2:],
                validation_kinematic_codes[:, 2:] - validation_kinematic_codes[:, 1:-1],
            ),
            axis=-1,
        ).reshape(-1, 2 * pod_rank)
        kinematic_history_pca = LinearCodePCA.fit(
            train_kinematic_history_raw, output_size=pod_rank
        )
        train_kinematic_history_code = kinematic_history_pca.transform(
            train_kinematic_history_raw
        )
        train_kinematic_history_future_code = kinematic_history_pca.transform(
            train_kinematic_history_future_raw
        )
        validation_kinematic_history_code = kinematic_history_pca.transform(
            validation_kinematic_history_raw
        )
        validation_kinematic_history_future_code = kinematic_history_pca.transform(
            validation_kinematic_history_future_raw
        )
        train_conditioned_kinematic_history_one_state = _conditioned_transition_code(
            train_kinematic_codes[:, 1:-1],
            source,
            train_cases,
            history_start_frame_ids,
            history_end_frame_ids,
        )
        train_conditioned_kinematic_history_one_state_future = (
            _conditioned_transition_code(
                train_kinematic_codes[:, 2:],
                source,
                train_cases,
                history_start_frame_ids,
                history_end_frame_ids,
            )
        )
        validation_conditioned_kinematic_history_one_state = (
            _conditioned_transition_code(
                validation_kinematic_codes[:, 1:-1],
                source,
                validation_cases,
                history_start_frame_ids,
                history_end_frame_ids,
            )
        )
        validation_conditioned_kinematic_history_one_state_future = (
            _conditioned_transition_code(
                validation_kinematic_codes[:, 2:],
                source,
                validation_cases,
                history_start_frame_ids,
                history_end_frame_ids,
            )
        )
        train_conditioned_kinematic_history = _conditioned_transition_code(
            train_kinematic_history_code.reshape(
                train_cases.size, history_start_frame_ids.size, pod_rank
            ),
            source,
            train_cases,
            history_start_frame_ids,
            history_end_frame_ids,
        )
        train_conditioned_kinematic_history_future = _conditioned_transition_code(
            train_kinematic_history_future_code.reshape(
                train_cases.size, history_start_frame_ids.size, pod_rank
            ),
            source,
            train_cases,
            history_start_frame_ids,
            history_end_frame_ids,
        )
        validation_conditioned_kinematic_history = _conditioned_transition_code(
            validation_kinematic_history_code.reshape(
                validation_cases.size, history_start_frame_ids.size, pod_rank
            ),
            source,
            validation_cases,
            history_start_frame_ids,
            history_end_frame_ids,
        )
        validation_conditioned_kinematic_history_future = _conditioned_transition_code(
            validation_kinematic_history_future_code.reshape(
                validation_cases.size, history_start_frame_ids.size, pod_rank
            ),
            source,
            validation_cases,
            history_start_frame_ids,
            history_end_frame_ids,
        )
        kinematic_one_state_history_subset = conditional_future_diagnostics(
            train_conditioned_kinematic_history_one_state,
            history_train_current,
            history_train_future,
            validation_conditioned_kinematic_history_one_state,
            history_validation_current,
            history_validation_future,
            cell_volume,
            component_scale,
            train_future_code=(train_conditioned_kinematic_history_one_state_future),
            query_future_code=(
                validation_conditioned_kinematic_history_one_state_future
            ),
            neighbors=NEIGHBORS,
        )
        kinematic_history_closure = conditional_future_diagnostics(
            train_conditioned_kinematic_history,
            history_train_current,
            history_train_future,
            validation_conditioned_kinematic_history,
            history_validation_current,
            history_validation_future,
            cell_volume,
            component_scale,
            train_future_code=train_conditioned_kinematic_history_future,
            query_future_code=validation_conditioned_kinematic_history_future,
            neighbors=NEIGHBORS,
        )
        kinematic_history_ratio = _bootstrap_ratio(
            kinematic_history_closure.future_ambiguity,
            kinematic_one_state_history_subset.future_ambiguity,
            validation_groups_history,
            repetitions=args.bootstrap_repetitions,
            seed=BOOTSTRAP_SEED + 50,
        )
        if (
            kinematic_history_closure.encoded_future_ambiguity is None
            or kinematic_one_state_history_subset.encoded_future_ambiguity is None
        ):
            raise RuntimeError(
                "encoded-future kinematic history diagnostics are required"
            )
        kinematic_history_encoded_ratio = _bootstrap_ratio(
            kinematic_history_closure.encoded_future_ambiguity,
            kinematic_one_state_history_subset.encoded_future_ambiguity,
            validation_groups_history,
            repetitions=args.bootstrap_repetitions,
            seed=BOOTSTRAP_SEED + 51,
        )
        artifact["metrics"]["kinematic_fixed_size_history"] = {
            "definition": "PCA_B([a_kin_n,a_kin_n-a_kin_n_minus_1])",
            "retained_scalar_size": pod_rank,
            "one_state": kinematic_one_state_history_subset.summary(),
            "history": kinematic_history_closure.summary(),
            "history_over_one_state_grouped_bootstrap": kinematic_history_ratio,
            "encoded_history_over_one_state_grouped_bootstrap": (
                kinematic_history_encoded_ratio
            ),
        }
        gates["kinematic_history_not_materially_better"] = (
            _history_not_materially_better_gate(kinematic_history_ratio)
        )
        gates["kinematic_history_encoded_not_materially_better"] = (
            _history_not_materially_better_gate(kinematic_history_encoded_ratio)
        )

    for gate_name in required_gate_names:
        if gate_name not in gates:
            gates[gate_name] = {
                "passed": False,
                "value": None,
                "comparison": "required",
                "threshold": None,
                "reason": "unavailable",
            }
    failed_gates = [
        gate_name for gate_name in required_gate_names if not gates[gate_name]["passed"]
    ]
    failed_kinematic_gates = [
        gate_name
        for gate_name in kinematic_gate_names
        if not gates[gate_name]["passed"]
    ]
    if not failed_gates:
        failure_classification = None
    elif any(name.startswith("remap_") for name in failed_gates):
        failure_classification = "oracle_chart_or_remap_implementation"
    elif any(name.startswith(("front_extraction_", "chart_")) for name in failed_gates):
        failure_classification = "front_coordinate_extraction_or_conditioning"
    elif any("reconstruction" in name for name in failed_gates):
        failure_classification = "matched_registered_compression"
    elif any(
        token in name
        for name in failed_gates
        for token in ("closure", "ambiguity", "history", "neighborhood")
    ):
        failure_classification = "one_state_markov_closure_or_forecastability"
    else:
        failure_classification = "precondition_or_admissibility"
    artifact["required_gate_names"] = required_gate_names
    artifact["failed_gates"] = failed_gates
    artifact["kinematic_component"] = {
        "hypothesis": (
            "three current-state RH/contact speeds remove the material history "
            "benefit at fixed total latent size"
        ),
        "matched_control": (f"oracle_front_pod_without_speed_same_{pod_rank}_scalars"),
        "required_gate_names": kinematic_gate_names,
        "failed_gates": failed_kinematic_gates,
        "passed": not failed_kinematic_gates,
        "decision": (
            "eligible_for_separate_chart_redesign"
            if not failed_kinematic_gates
            else "reject_current_state_front_speed_augmentation"
        ),
    }
    artifact["promotion_passed"] = not failed_gates
    artifact["learned_training_authorized"] = False
    artifact["failure_classification"] = failure_classification
    artifact["promotion_decision"] = (
        "eligible_for_separate_review_before_learned_pilot"
        if not failed_gates
        else "stop_before_learned_representation"
    )
    artifact["status"] = "completed"
    return artifact


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command != "preflight":
        raise ValueError(f"unsupported command: {args.command}")
    artifact = run_preflight(args)
    _write_artifact(args.output_path, artifact, overwrite=args.overwrite)
    print(
        json.dumps(
            {
                "status": artifact["status"],
                "promotion_passed": artifact["promotion_passed"],
                "failed_gates": artifact["failed_gates"],
                "kinematic_component_passed": artifact["kinematic_component"]["passed"],
                "failed_kinematic_gates": artifact["kinematic_component"][
                    "failed_gates"
                ],
                "artifact": args.output_path.name,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
