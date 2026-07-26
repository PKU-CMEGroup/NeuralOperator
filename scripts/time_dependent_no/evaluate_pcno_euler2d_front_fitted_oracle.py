#!/usr/bin/env python3
"""Run the D062 front-fitted conservative-remap capacity oracle."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import sys
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.time_dependent_no.diagnose_pcno_euler2d_multirate_headroom import (  # noqa: E402
    _load_trajectory,
    _scalar,
    _validated_summary,
    _variant_metrics,
    _vortex_center,
)
from scripts.time_dependent_no.evaluate_pcno_shock_vortex_baseline import (  # noqa: E402
    atomic_write_json,
    json_safe,
    sha256_file,
    write_csv,
)
from utility.time_dependent_no.pcno_ripple_diagnostics import (  # noqa: E402
    conservative_to_primitive_raw,
)

SCHEMA = "pcno_shock_vortex_front_fitted_oracle_v1"
EXPERIMENT_ID = "D062"
D060_SOURCE = {
    "summary_sha256": "50ad14177f1efd1154124c56a74224102331af5cc9d1da914e179847a2d1f2e2",
    "checkpoint_sha256": "95e6c180a3298c4b38662d8c6cb77301c0e7fd0286e9a53571bddc487b8379c9",
    "checkpoint_config_digest": "0064571b541476e49a6f433cd44c5fd31cee71cfd5a8cc563fa151efe68d8c4f",
    "normalization_digest": "9df7efb2f1aadf315f399dcabf3b366bf1b2f46794b026cbc5b7d80ba22e1a1a",
    "data_manifest_digest": "f8d228ae3c6e08fe6697f37df21476fe621abc354d0b0a6eed2a623ed2b9f96c",
}
DEFAULT_TRAJECTORY_KEYS = (
    "sv_e00_y00",
    "sv_e00_y08",
    "sv_e06_y00",
    "sv_e06_y08",
    "sv_e11_y00",
    "sv_e11_y08",
)
DEFAULT_CALLS = (15, 30)
SHOCK_WINDOW_HALF_WIDTH = 0.10
MAX_TOTAL_DEFECT = 1.0e-10
MIN_MEDIAN_STATE_REDUCTION = 0.15
MIN_MEDIAN_FRONT_REDUCTION = 0.50
MAX_RELATIVE_STRUCTURE_WORSENING = 0.05
MIN_CASE_COUNT = 5
MAX_RETAINED_BYTES = 256 * 1024 * 1024
STRUCTURE_METRICS = (
    "front_centroid_distance",
    "front_symmetric_chamfer",
    "shock_strength_log_error",
    "shock_thickness_log_error",
    "vortex_core_density_relative_error",
)


@dataclass(frozen=True)
class UniformRowMajorGrid:
    """Validated row-major Cartesian cell grid used by the D062 chart."""

    x: np.ndarray
    y: np.ndarray
    x_edges: np.ndarray
    y_edges: np.ndarray
    nx: int
    ny: int
    dx: float
    dy: float


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--trajectory-keys", nargs="+", default=list(DEFAULT_TRAJECTORY_KEYS)
    )
    parser.add_argument("--calls", type=int, nargs="+", default=DEFAULT_CALLS)
    return parser.parse_args(argv)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(json_safe(row), sort_keys=True, allow_nan=False))
            handle.write("\n")
    os.replace(temporary, path)


def validate_uniform_row_major_grid(
    positions: np.ndarray,
    volumes: np.ndarray,
    mesh_cell_to_graph_node: np.ndarray,
    *,
    coordinate_convention: str,
    reference_config: Mapping[str, Any],
) -> UniformRowMajorGrid:
    """Close the tensor-grid, identity-map, and physical-volume contract."""

    pos = np.asarray(positions, dtype=np.float64)
    volume = np.asarray(volumes, dtype=np.float64).reshape(-1)
    mapping = np.asarray(mesh_cell_to_graph_node, dtype=np.int64).reshape(-1)
    if pos.ndim != 2 or pos.shape[1] != 2 or volume.shape != (pos.shape[0],):
        raise ValueError("positions and physical volumes have incompatible shapes")
    if not np.all(np.isfinite(pos)) or np.any(volume <= 0.0):
        raise ValueError("grid coordinates and volumes must be finite and positive")
    if coordinate_convention != "FV_row_major_cell_centers_xy":
        raise ValueError("D062 requires the frozen row-major FV convention")
    if not np.array_equal(mapping, np.arange(pos.shape[0], dtype=np.int64)):
        raise ValueError("D062 requires the audited identity mesh-to-graph map")

    x = np.unique(pos[:, 0])
    y = np.unique(pos[:, 1])
    nx = int(x.size)
    ny = int(y.size)
    if nx * ny != pos.shape[0] or nx < 3 or ny < 1:
        raise ValueError("positions do not form a complete 2D tensor grid")
    xx, yy = np.meshgrid(x, y, indexing="xy")
    expected = np.stack((xx.reshape(-1), yy.reshape(-1)), axis=-1)
    if float(np.max(np.abs(pos - expected))) > 5.0e-7:
        raise ValueError("positions are not in row-major tensor-grid order")
    dx_values = np.diff(x)
    dy_values = np.diff(y)
    dx = float(np.mean(dx_values))
    dy = (
        float(np.mean(dy_values))
        if ny > 1
        else float(reference_config["y_max"] - reference_config["y_min"])
    )
    if (
        dx <= 0.0
        or dy <= 0.0
        or not np.allclose(dx_values, dx, rtol=0.0, atol=5.0e-7)
        or (ny > 1 and not np.allclose(dy_values, dy, rtol=0.0, atol=5.0e-7))
    ):
        raise ValueError("D062 requires a uniform Cartesian cell grid")
    x_min = float(reference_config["x_min"])
    x_max = float(reference_config["x_max"])
    y_min = float(reference_config["y_min"])
    y_max = float(reference_config["y_max"])
    x_edges = np.linspace(x_min, x_max, nx + 1, dtype=np.float64)
    y_edges = np.linspace(y_min, y_max, ny + 1, dtype=np.float64)
    center_error = float(
        max(
            np.max(np.abs(x - 0.5 * (x_edges[:-1] + x_edges[1:]))),
            np.max(np.abs(y - 0.5 * (y_edges[:-1] + y_edges[1:]))),
        )
    )
    if center_error > 5.0e-7:
        raise ValueError("cell centers do not match the physical bounds")
    expected_volume = (x_max - x_min) * (y_max - y_min) / (nx * ny)
    if float(np.max(np.abs(volume - expected_volume))) > 1.0e-10:
        raise ValueError("physical cell volumes disagree with the tensor grid")
    return UniformRowMajorGrid(
        x=x,
        y=y,
        x_edges=x_edges,
        y_edges=y_edges,
        nx=nx,
        ny=ny,
        dx=dx,
        dy=dy,
    )


def validate_oriented_face_geometry(
    artifact: Mapping[str, np.ndarray], grid: UniformRowMajorGrid
) -> dict[str, float]:
    """Validate the faces used to support total and boundary accounting."""

    centers = np.asarray(artifact["positions"], dtype=np.float64)
    face_centers = np.asarray(artifact["face_centers"], dtype=np.float64)
    measures = np.asarray(artifact["face_measures"], dtype=np.float64).reshape(-1)
    normals = np.asarray(artifact["face_normals"], dtype=np.float64)
    owner = np.asarray(artifact["face_owner"], dtype=np.int64).reshape(-1)
    neighbor = np.asarray(artifact["face_neighbor"], dtype=np.int64).reshape(-1)
    tags = np.asarray(artifact["face_boundary_tag"], dtype=np.int64).reshape(-1)
    expected_faces = (grid.nx + 1) * grid.ny + grid.nx * (grid.ny + 1)
    if (
        face_centers.shape != (expected_faces, 2)
        or normals.shape != face_centers.shape
        or measures.shape != (expected_faces,)
        or owner.shape != measures.shape
        or neighbor.shape != measures.shape
        or tags.shape != measures.shape
    ):
        raise ValueError("physical face arrays do not match the tensor grid")
    if np.any(measures <= 0.0) or not np.all(np.isfinite(face_centers)):
        raise ValueError("face geometry must be finite with positive measures")
    if np.any(owner < 0) or np.any(owner >= centers.shape[0]):
        raise ValueError("face owner lies outside the cell axis")
    interior = neighbor >= 0
    if (
        np.any(neighbor[interior] >= centers.shape[0])
        or np.any(tags[interior] != 0)
        or np.any(tags[~interior] <= 0)
    ):
        raise ValueError("face connectivity and boundary tags disagree")
    unit_error = float(np.max(np.abs(np.linalg.norm(normals, axis=1) - 1.0)))
    if unit_error > 1.0e-12:
        raise ValueError("face normals are not unit length")
    interior_orientation = np.einsum(
        "ij,ij->i",
        centers[neighbor[interior]] - centers[owner[interior]],
        normals[interior],
    )
    boundary_orientation = np.einsum(
        "ij,ij->i",
        face_centers[~interior] - centers[owner[~interior]],
        normals[~interior],
    )
    if np.any(interior_orientation <= 0.0) or np.any(boundary_orientation <= 0.0):
        raise ValueError("face normals violate owner-oriented connectivity")
    area_vector = np.zeros((centers.shape[0], 2), dtype=np.float64)
    np.add.at(area_vector, owner, measures[:, None] * normals)
    np.add.at(
        area_vector,
        neighbor[interior],
        -measures[interior, None] * normals[interior],
    )
    closure = float(np.max(np.linalg.norm(area_vector, axis=1)))
    if closure > 1.0e-12:
        raise ValueError("physical cell face-area vectors do not close")
    return {
        "face_count": float(expected_faces),
        "maximum_face_normal_unit_error": unit_error,
        "maximum_cell_area_vector_closure": closure,
        "minimum_owner_oriented_face_distance": float(
            min(np.min(interior_orientation), np.min(boundary_orientation))
        ),
    }


def front_curve_from_state(
    state: np.ndarray,
    grid: UniformRowMajorGrid,
    *,
    gamma: float,
    shock_x: float,
    half_width: float = SHOCK_WINDOW_HALF_WIDTH,
) -> dict[str, np.ndarray]:
    """Extract one unsmoothed pressure-jump shock coordinate per y row."""

    values = np.asarray(state, dtype=np.float64)
    if values.shape != (grid.nx * grid.ny, 4):
        raise ValueError("state does not match the declared tensor grid")
    if half_width <= 0.0:
        raise ValueError("shock window half-width must be positive")
    pressure = conservative_to_primitive_raw(values, gamma=gamma)[:, 3]
    pressure = pressure.reshape(grid.ny, grid.nx)
    jumps = np.abs(np.diff(pressure, axis=1))
    face_x = 0.5 * (grid.x[:-1] + grid.x[1:])
    eligible = np.flatnonzero(
        (face_x >= shock_x - half_width) & (face_x <= shock_x + half_width)
    )
    if eligible.size < 3:
        raise ValueError("declared shock window contains fewer than three x-faces")
    curve = np.empty(grid.ny, dtype=np.float64)
    peak_jump = np.empty(grid.ny, dtype=np.float64)
    for row in range(grid.ny):
        peak = int(eligible[int(np.argmax(jumps[row, eligible]))])
        neighbors = eligible[(eligible >= peak - 1) & (eligible <= peak + 1)]
        weights = jumps[row, neighbors]
        total = float(np.sum(weights))
        if not math.isfinite(total) or total <= 0.0:
            raise ValueError("pressure-jump front extractor found a zero row")
        curve[row] = float(np.sum(weights * face_x[neighbors]) / total)
        peak_jump[row] = float(jumps[row, peak])
    return {"curve": curve, "peak_pressure_jump": peak_jump}


def _piecewise_map(
    coordinate: np.ndarray | float,
    *,
    x_min: float,
    x_pred: float,
    x_target: float,
    x_max: float,
) -> np.ndarray:
    values = np.asarray(coordinate, dtype=np.float64)
    if not x_min < x_pred < x_max or not x_min < x_target < x_max:
        raise ValueError("front coordinates must lie strictly inside the domain")
    left_slope = (x_target - x_min) / (x_pred - x_min)
    right_slope = (x_max - x_target) / (x_max - x_pred)
    return np.where(
        values <= x_pred,
        x_min + left_slope * (values - x_min),
        x_target + right_slope * (values - x_pred),
    )


def _pushforward_row(
    state: np.ndarray,
    edges: np.ndarray,
    *,
    x_pred: float,
    x_target: float,
) -> tuple[np.ndarray, tuple[float, float]]:
    """Push cell integrals through one exact piecewise-linear row map."""

    values = np.asarray(state, dtype=np.float64)
    boundaries = np.asarray(edges, dtype=np.float64)
    nx = boundaries.size - 1
    if values.shape != (nx, 4) or np.any(np.diff(boundaries) <= 0.0):
        raise ValueError("row state and cell edges are incompatible")
    x_min = float(boundaries[0])
    x_max = float(boundaries[-1])
    breaks = np.unique(np.concatenate((boundaries, np.asarray([x_pred]))))
    mapped = _piecewise_map(
        breaks,
        x_min=x_min,
        x_pred=x_pred,
        x_target=x_target,
        x_max=x_max,
    )
    if np.any(np.diff(mapped) <= 0.0):
        raise ValueError("front warp is not strictly monotone")
    output_integral = np.zeros((nx, 4), dtype=np.float64)
    coverage = np.zeros(nx, dtype=np.float64)
    target_width = np.diff(boundaries)
    for segment in range(breaks.size - 1):
        left = float(breaks[segment])
        right = float(breaks[segment + 1])
        mapped_left = float(mapped[segment])
        mapped_right = float(mapped[segment + 1])
        source_index = min(
            nx - 1,
            max(0, int(np.searchsorted(boundaries, 0.5 * (left + right)) - 1)),
        )
        mapped_width = mapped_right - mapped_left
        pushed_density = values[source_index] * (right - left) / mapped_width
        first = max(0, int(np.searchsorted(boundaries, mapped_left, side="right") - 1))
        last = min(
            nx - 1,
            int(np.searchsorted(boundaries, mapped_right, side="left")),
        )
        for target_index in range(first, last + 1):
            overlap = min(mapped_right, boundaries[target_index + 1]) - max(
                mapped_left, boundaries[target_index]
            )
            if overlap > 0.0:
                output_integral[target_index] += pushed_density * overlap
                coverage[target_index] += overlap
    if not np.allclose(coverage, target_width, rtol=0.0, atol=5.0e-14):
        raise RuntimeError("conservative front warp missed a target cell")
    left_slope = (x_target - x_min) / (x_pred - x_min)
    right_slope = (x_max - x_target) / (x_max - x_pred)
    return output_integral / target_width[:, None], (left_slope, right_slope)


def conservative_front_warp(
    state: np.ndarray,
    grid: UniformRowMajorGrid,
    predicted_curve: np.ndarray,
    target_curve: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply the row-wise domain-anchored conservative phase remap."""

    values = np.asarray(state, dtype=np.float64)
    pred = np.asarray(predicted_curve, dtype=np.float64).reshape(-1)
    target = np.asarray(target_curve, dtype=np.float64).reshape(-1)
    if values.shape != (grid.nx * grid.ny, 4):
        raise ValueError("state does not match the tensor grid")
    if pred.shape != (grid.ny,) or target.shape != (grid.ny,):
        raise ValueError("front curves must contain one value per y row")
    output = np.empty((grid.ny, grid.nx, 4), dtype=np.float64)
    slopes = np.empty((grid.ny, 2), dtype=np.float64)
    source = values.reshape(grid.ny, grid.nx, 4)
    for row in range(grid.ny):
        output[row], slopes[row] = _pushforward_row(
            source[row],
            grid.x_edges,
            x_pred=float(pred[row]),
            x_target=float(target[row]),
        )
    displacement = (target - pred) / grid.dx
    return output.reshape(-1, 4), {
        "minimum_warp_jacobian": float(np.min(slopes)),
        "maximum_warp_jacobian": float(np.max(slopes)),
        "mean_absolute_displacement_cells": float(np.mean(np.abs(displacement))),
        "maximum_absolute_displacement_cells": float(np.max(np.abs(displacement))),
        "rms_displacement_cells": float(np.sqrt(np.mean(displacement**2))),
        "displacement_cells": displacement,
    }


def fit_two_sided_strength_contrast(
    phase_state: np.ndarray,
    target: np.ndarray,
    grid: UniformRowMajorGrid,
    target_curve: np.ndarray,
    volumes: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit four target-informed strength values in a row-zero-total basis."""

    phase = np.asarray(phase_state, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    volume = np.asarray(volumes, dtype=np.float64).reshape(grid.ny, grid.nx)
    curve = np.asarray(target_curve, dtype=np.float64).reshape(-1)
    if phase.shape != truth.shape or phase.shape != (grid.nx * grid.ny, 4):
        raise ValueError("phase state and target do not match the grid")
    if curve.shape != (grid.ny,) or np.any(volume <= 0.0):
        raise ValueError("strength chart has invalid curve or volumes")
    left = grid.x[None, :] < curve[:, None]
    left_volume = np.sum(volume * left, axis=1)
    right_volume = np.sum(volume * ~left, axis=1)
    if np.any(left_volume <= 0.0) or np.any(right_volume <= 0.0):
        raise ValueError("target front must leave cells on both sides in every row")
    contrast = np.where(
        left,
        1.0,
        -(left_volume / right_volume)[:, None],
    )
    flat_contrast = contrast.reshape(-1)
    flat_volume = volume.reshape(-1)
    denominator = float(np.sum(flat_volume * flat_contrast**2))
    coefficient = (
        np.sum(
            flat_volume[:, None] * flat_contrast[:, None] * (truth - phase),
            axis=0,
        )
        / denominator
    )
    correction = flat_contrast[:, None] * coefficient[None, :]
    candidate = phase + correction
    right_offsets = -(left_volume / right_volume)[:, None] * coefficient[None, :]
    return candidate, {
        "strength_coefficient": coefficient,
        "minimum_right_offset_by_component": np.min(right_offsets, axis=0),
        "maximum_right_offset_by_component": np.max(right_offsets, axis=0),
        "left_volume_fraction_by_row": left_volume / (left_volume + right_volume),
        "active_cell_fraction": float(
            np.count_nonzero(flat_contrast) / flat_contrast.size
        ),
    }


def row_total_conservation(
    reference: np.ndarray,
    candidate: np.ndarray,
    volumes: np.ndarray,
    component_scale: np.ndarray,
    grid: UniformRowMajorGrid,
) -> dict[str, Any]:
    """Measure row-wise and global conservative-total preservation."""

    first = np.asarray(reference, dtype=np.float64).reshape(grid.ny, grid.nx, 4)
    second = np.asarray(candidate, dtype=np.float64).reshape(grid.ny, grid.nx, 4)
    volume = np.asarray(volumes, dtype=np.float64).reshape(grid.ny, grid.nx)
    scale = np.asarray(component_scale, dtype=np.float64).reshape(4)
    first_total = np.sum(volume[:, :, None] * first, axis=1)
    second_total = np.sum(volume[:, :, None] * second, axis=1)
    defect = second_total - first_total
    row_volume = np.sum(volume, axis=1)
    denominator = np.maximum(
        np.abs(first_total), np.abs(scale)[None, :] * row_volume[:, None]
    )
    relative = np.abs(defect) / np.maximum(denominator, 1.0e-30)
    global_defect = np.sum(defect, axis=0)
    global_scale = np.maximum(
        np.abs(np.sum(first_total, axis=0)),
        np.abs(scale) * float(np.sum(volume)),
    )
    return {
        "maximum_absolute_row_component_defect": float(np.max(np.abs(defect))),
        "maximum_relative_row_component_defect": float(np.max(relative)),
        "global_component_defect": global_defect,
        "maximum_relative_global_component_defect": float(
            np.max(np.abs(global_defect) / np.maximum(global_scale, 1.0e-30))
        ),
    }


def scaled_volume_norm(
    value: np.ndarray, volumes: np.ndarray, component_scale: np.ndarray
) -> float:
    delta = np.asarray(value, dtype=np.float64)
    volume = np.asarray(volumes, dtype=np.float64).reshape(-1)
    scale = np.asarray(component_scale, dtype=np.float64).reshape(1, 4)
    if delta.shape != (volume.size, 4):
        raise ValueError("scaled norm arrays do not align")
    return float(
        np.sqrt(np.sum(volume[:, None] * (delta / scale) ** 2) / np.sum(volume))
    )


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if not math.isfinite(numerator) or not math.isfinite(denominator):
        return None
    if denominator <= 1.0e-30:
        return 0.0 if numerator <= 1.0e-30 else None
    return numerator / denominator


def _relative_reduction(candidate: float, baseline: float) -> float:
    ratio = _safe_ratio(candidate, baseline)
    return 1.0 - ratio if ratio is not None else -math.inf


def front_curve_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    grid: UniformRowMajorGrid,
    *,
    gamma: float,
    shock_x: float,
    target_curve: np.ndarray | None = None,
) -> dict[str, Any]:
    pred = front_curve_from_state(prediction, grid, gamma=gamma, shock_x=shock_x)
    truth = (
        front_curve_from_state(target, grid, gamma=gamma, shock_x=shock_x)
        if target_curve is None
        else {"curve": np.asarray(target_curve, dtype=np.float64)}
    )
    difference = np.asarray(pred["curve"]) - np.asarray(truth["curve"])
    return {
        "mean_absolute_error": float(np.mean(np.abs(difference))),
        "mean_absolute_error_cells": float(np.mean(np.abs(difference)) / grid.dx),
        "rms_error": float(np.sqrt(np.mean(difference**2))),
        "maximum_absolute_error": float(np.max(np.abs(difference))),
        "curve": np.asarray(pred["curve"]),
        "peak_pressure_jump": np.asarray(pred["peak_pressure_jump"]),
    }


def correction_norms(
    baseline: np.ndarray,
    phase: np.ndarray,
    primary: np.ndarray,
    target: np.ndarray,
    current: np.ndarray,
    *,
    volumes: np.ndarray,
    component_scale: np.ndarray,
) -> dict[str, Any]:
    error_norm = scaled_volume_norm(baseline - target, volumes, component_scale)
    model_update_norm = scaled_volume_norm(baseline - current, volumes, component_scale)
    phase_norm = scaled_volume_norm(phase - baseline, volumes, component_scale)
    strength_norm = scaled_volume_norm(primary - phase, volumes, component_scale)
    total_norm = scaled_volume_norm(primary - baseline, volumes, component_scale)
    return {
        "baseline_target_error_norm": error_norm,
        "baseline_model_update_norm": model_update_norm,
        "phase_correction_norm": phase_norm,
        "strength_correction_norm": strength_norm,
        "total_correction_norm": total_norm,
        "total_correction_to_baseline_error": _safe_ratio(total_norm, error_norm),
        "total_correction_to_model_update": _safe_ratio(total_norm, model_update_norm),
        "strength_to_phase_correction": _safe_ratio(strength_norm, phase_norm),
    }


def structure_acceptance(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    max_relative_worsening: float = MAX_RELATIVE_STRUCTURE_WORSENING,
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for name in STRUCTURE_METRICS:
        base = baseline.get(name)
        proposed = candidate.get(name)
        valid = (
            base is not None
            and proposed is not None
            and math.isfinite(float(base))
            and math.isfinite(float(proposed))
        )
        accepted = (
            valid
            and float(proposed)
            <= (1.0 + max_relative_worsening) * float(base) + 1.0e-12
        )
        report[name] = {
            "accepted": bool(accepted),
            "baseline": float(base) if valid else None,
            "candidate": float(proposed) if valid else None,
        }
    return {
        "passed": all(item["accepted"] for item in report.values()),
        "metrics": report,
    }


def promotion_decision(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_trajectories: int,
    expected_calls: Sequence[int] = DEFAULT_CALLS,
) -> dict[str, Any]:
    expected = expected_trajectories * len(expected_calls)
    h60_call = max(map(int, expected_calls))
    h60 = [row for row in rows if int(row["call"]) == h60_call]
    all_rows = len(rows) == expected and len(h60) == expected_trajectories
    all_admissible = all_rows and all(
        bool(row["all_variants_raw_admissible"]) for row in rows
    )
    maximum_conservation_defect = max(
        (
            float(row[variant]["maximum_relative_row_component_defect"])
            for row in rows
            for variant in ("phase_conservation", "primary_conservation")
        ),
        default=math.inf,
    )
    state = [float(row["primary_state_reduction"]) for row in h60]
    front = [float(row["primary_front_curve_reduction"]) for row in h60]
    highpass = [float(row["primary_highpass_reduction"]) for row in h60]
    median_state = float(np.median(state)) if state else None
    median_front = float(np.median(front)) if front else None
    median_highpass = float(np.median(highpass)) if highpass else None
    joint_count = sum(
        bool(row["primary_joint_state_front_highpass_nonworse"]) for row in h60
    )
    metric_counts = {
        name: sum(
            bool(row["primary_structure_acceptance"]["metrics"][name]["accepted"])
            for row in h60
        )
        for name in STRUCTURE_METRICS
    }
    gates = {
        "all_expected_rows_available": all_rows,
        "all_variants_raw_admissible": all_admissible,
        "maximum_row_total_defect_at_most_1e_10": (
            maximum_conservation_defect <= MAX_TOTAL_DEFECT
        ),
        "h60_median_state_reduction_at_least_0p15": (
            median_state is not None and median_state >= MIN_MEDIAN_STATE_REDUCTION
        ),
        "h60_median_front_curve_reduction_at_least_0p50": (
            median_front is not None and median_front >= MIN_MEDIAN_FRONT_REDUCTION
        ),
        "h60_median_smooth_highpass_does_not_increase": (
            median_highpass is not None and median_highpass >= 0.0
        ),
        "h60_joint_state_front_highpass_nonworse_in_at_least_5_of_6": (
            joint_count >= MIN_CASE_COUNT
        ),
        "h60_each_structure_metric_within_5pct_in_at_least_5_of_6": all(
            count >= MIN_CASE_COUNT for count in metric_counts.values()
        ),
    }
    passed = all(gates.values())
    phase_state = [float(row["phase_state_reduction"]) for row in h60]
    phase_front = [float(row["phase_front_curve_reduction"]) for row in h60]
    return {
        "passed": passed,
        "decision": (
            "authorize_front_chart_tiny_fit_contract_only"
            if passed
            else "reject_row_graph_two_sided_strength_chart"
        ),
        "expected_rows": expected,
        "observed_rows": len(rows),
        "h60_rows": len(h60),
        "maximum_relative_row_component_total_defect": maximum_conservation_defect,
        "h60_median_primary_state_reduction": median_state,
        "h60_median_primary_front_curve_reduction": median_front,
        "h60_median_primary_highpass_reduction": median_highpass,
        "h60_joint_nonworse_case_count": joint_count,
        "h60_structure_metric_acceptance_counts": metric_counts,
        "h60_median_phase_only_state_reduction": (
            float(np.median(phase_state)) if phase_state else None
        ),
        "h60_median_phase_only_front_curve_reduction": (
            float(np.median(phase_front)) if phase_front else None
        ),
        "gates": gates,
    }


def main(argv: Sequence[str] | None = None) -> None:
    started = perf_counter()
    implementation_sha256 = sha256_file(Path(__file__).resolve())
    args = parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")
    keys = [str(key) for key in args.trajectory_keys]
    calls = sorted(set(int(call) for call in args.calls))
    if keys != list(DEFAULT_TRAJECTORY_KEYS):
        raise ValueError("D062 requires the frozen six-case D013 trajectory order")
    if tuple(calls) != DEFAULT_CALLS:
        raise ValueError("D062 requires calls 15 and 30")
    baseline_summary, artifact_index, source = _validated_summary(
        args.baseline_dir, expected_stride=2, keys=keys
    )
    source_mismatches = {
        name: {"expected": expected, "observed": source.get(name)}
        for name, expected in D060_SOURCE.items()
        if source.get(name) != expected
    }
    if source_mismatches:
        raise ValueError(f"baseline directory is not frozen D060: {source_mismatches}")
    if source["num_steps"] != 30 or not math.isclose(
        source["physical_horizon"], 0.6, rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise ValueError("D060 horizon contract does not close")
    source_data = baseline_summary.get("data_contract", {})
    source_resolution = source_data.get("resolution_contract", {})
    if source_data.get("mesh_to_graph_map") != (
        "identity_FV_cell_index_to_PCNO_node_index"
    ):
        raise ValueError("D060 mesh-to-graph contract does not close")
    if source_resolution.get("stored_grid") != [250, 100]:
        raise ValueError("D060 stored-grid contract does not close")

    preflight: dict[
        str, tuple[dict[str, np.ndarray], dict[str, Any], UniformRowMajorGrid]
    ] = {}
    contracts: list[dict[str, Any]] = []
    for key in keys:
        artifact = _load_trajectory(
            args.baseline_dir,
            key=key,
            index_row=artifact_index[key],
            source=source,
        )
        if int(_scalar(artifact["valid_length"])) < max(calls):
            raise ValueError(f"D060 trajectory does not reach call 30: {key}")
        if _scalar(artifact["failure_cause"]) != "completed":
            raise ValueError(f"D060 trajectory did not complete: {key}")
        reference_config = json.loads(str(_scalar(artifact["reference_config_json"])))
        grid = validate_uniform_row_major_grid(
            artifact["positions"],
            artifact["physical_cell_volumes"],
            artifact["mesh_cell_to_graph_node"],
            coordinate_convention=str(_scalar(artifact["coordinate_convention"])),
            reference_config=reference_config,
        )
        face_contract = validate_oriented_face_geometry(artifact, grid)
        contracts.append(
            {
                "trajectory": key,
                "grid_shape": [grid.nx, grid.ny],
                "identity_mesh_to_graph_map": True,
                "coordinate_convention": str(
                    _scalar(artifact["coordinate_convention"])
                ),
                **face_contract,
            }
        )
        preflight[key] = (artifact, reference_config, grid)

    # Do not create an experiment directory until every frozen source,
    # trajectory, and geometry contract has passed read-only preflight.
    args.output_dir.mkdir(parents=True)
    trajectory_dir = args.output_dir / "trajectories"
    trajectory_dir.mkdir()
    rows: list[dict[str, Any]] = []
    variant_rows: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []

    for key in keys:
        artifact, reference_config, grid = preflight[key]
        parameters = json.loads(str(_scalar(artifact["parameters_json"])))

        initial = np.asarray(artifact["initial_state"], dtype=np.float64)
        targets = np.asarray(artifact["targets"], dtype=np.float64)
        baseline_rollout = np.asarray(artifact["predictions"], dtype=np.float64)
        positions = np.asarray(artifact["positions"], dtype=np.float64)
        edges = np.asarray(artifact["edges"], dtype=np.int64)
        volumes = np.asarray(artifact["physical_cell_volumes"], dtype=np.float64)
        component_scale = np.asarray(
            artifact["state_component_scale"], dtype=np.float64
        )
        physical_times = np.asarray(artifact["physical_times"], dtype=np.float64)
        interval_exchange = np.asarray(
            artifact["reference_interval_boundary_exchange"], dtype=np.float64
        )
        gamma = float(_scalar(artifact["gamma"]))
        shock_quantile = float(_scalar(artifact["shock_quantile"]))
        shock_x = float(reference_config["shock_x"])

        saved_current = []
        saved_targets = []
        saved_baseline = []
        saved_phase = []
        saved_primary = []
        saved_target_curve = []
        saved_baseline_curve = []
        saved_phase_curve = []
        saved_primary_curve = []
        saved_strength = []
        saved_min_jacobian = []
        saved_max_jacobian = []

        for call in calls:
            target = targets[call - 1]
            baseline = baseline_rollout[call - 1]
            current = initial if call == 1 else baseline_rollout[call - 2]
            target_front = front_curve_from_state(
                target, grid, gamma=gamma, shock_x=shock_x
            )
            baseline_front = front_curve_metrics(
                baseline,
                target,
                grid,
                gamma=gamma,
                shock_x=shock_x,
                target_curve=target_front["curve"],
            )
            phase, warp = conservative_front_warp(
                baseline,
                grid,
                baseline_front["curve"],
                target_front["curve"],
            )
            primary, strength = fit_two_sided_strength_contrast(
                phase, target, grid, target_front["curve"], volumes
            )
            phase_front = front_curve_metrics(
                phase,
                target,
                grid,
                gamma=gamma,
                shock_x=shock_x,
                target_curve=target_front["curve"],
            )
            primary_front = front_curve_metrics(
                primary,
                target,
                grid,
                gamma=gamma,
                shock_x=shock_x,
                target_curve=target_front["curve"],
            )
            phase_conservation = row_total_conservation(
                baseline, phase, volumes, component_scale, grid
            )
            primary_conservation = row_total_conservation(
                baseline, primary, volumes, component_scale, grid
            )
            saved_stop = 2 * call
            cumulative_exchange = np.sum(interval_exchange[:saved_stop], axis=0)
            center = _vortex_center(
                reference_config,
                parameters,
                absolute_time=float(physical_times[saved_stop]),
            )
            variants = {
                "d060_baseline": baseline,
                "phase_only_conservative_warp": phase,
                "phase_strength_conservative_oracle": primary,
            }
            metrics = {
                name: _variant_metrics(
                    prediction,
                    target,
                    initial,
                    positions=positions,
                    edges=edges,
                    volumes=volumes,
                    component_scale=component_scale,
                    gamma=gamma,
                    shock_quantile=shock_quantile,
                    vortex_center=center,
                    cumulative_boundary_exchange=cumulative_exchange,
                )
                for name, prediction in variants.items()
            }
            curve_metrics = {
                "d060_baseline": baseline_front,
                "phase_only_conservative_warp": phase_front,
                "phase_strength_conservative_oracle": primary_front,
            }
            physical_time = float(physical_times[saved_stop] - physical_times[0])
            for variant, values in metrics.items():
                variant_rows.append(
                    {
                        "trajectory": key,
                        "call": call,
                        "physical_time": physical_time,
                        "variant": variant,
                        "front_curve_mean_absolute_error": curve_metrics[variant][
                            "mean_absolute_error"
                        ],
                        "front_curve_mean_absolute_error_cells": curve_metrics[variant][
                            "mean_absolute_error_cells"
                        ],
                        **values,
                    }
                )
            baseline_metrics = metrics["d060_baseline"]
            phase_metrics = metrics["phase_only_conservative_warp"]
            primary_metrics = metrics["phase_strength_conservative_oracle"]
            baseline_state = float(
                baseline_metrics["scaled_relative_l2_physical_volume"]
            )
            phase_state = float(phase_metrics["scaled_relative_l2_physical_volume"])
            primary_state = float(primary_metrics["scaled_relative_l2_physical_volume"])
            baseline_highpass = float(
                baseline_metrics["smooth_region_graph_highpass_rms"]
            )
            primary_highpass = float(
                primary_metrics["smooth_region_graph_highpass_rms"]
            )
            baseline_front_error = float(baseline_front["mean_absolute_error"])
            phase_front_error = float(phase_front["mean_absolute_error"])
            primary_front_error = float(primary_front["mean_absolute_error"])
            admissible = all(
                bool(values["admissibility"]["all_finite"])
                and bool(values["admissibility"]["all_admissible"])
                for values in metrics.values()
            )
            structure = structure_acceptance(baseline_metrics, primary_metrics)
            norm_report = correction_norms(
                baseline,
                phase,
                primary,
                target,
                current,
                volumes=volumes,
                component_scale=component_scale,
            )
            rows.append(
                {
                    "trajectory": key,
                    "call": call,
                    "physical_time": physical_time,
                    "all_variants_raw_admissible": admissible,
                    "phase_conservation": phase_conservation,
                    "primary_conservation": primary_conservation,
                    "warp": warp,
                    "strength": strength,
                    "correction_norms": norm_report,
                    "phase_state_reduction": _relative_reduction(
                        phase_state, baseline_state
                    ),
                    "primary_state_reduction": _relative_reduction(
                        primary_state, baseline_state
                    ),
                    "phase_front_curve_reduction": _relative_reduction(
                        phase_front_error, baseline_front_error
                    ),
                    "primary_front_curve_reduction": _relative_reduction(
                        primary_front_error, baseline_front_error
                    ),
                    "primary_highpass_reduction": _relative_reduction(
                        primary_highpass, baseline_highpass
                    ),
                    "primary_joint_state_front_highpass_nonworse": (
                        primary_state <= baseline_state + 1.0e-12
                        and primary_front_error <= baseline_front_error + 1.0e-12
                        and primary_highpass
                        <= (1.0 + MAX_RELATIVE_STRUCTURE_WORSENING) * baseline_highpass
                        + 1.0e-12
                    ),
                    "primary_structure_acceptance": structure,
                    "front_curve_metrics": curve_metrics,
                    "metrics": metrics,
                }
            )
            saved_current.append(current.astype(np.float32))
            saved_targets.append(target.astype(np.float32))
            saved_baseline.append(baseline.astype(np.float32))
            # Preserve the precision on which the 1e-10 conservation gate is
            # evaluated; float32 storage cannot witness that contract.
            saved_phase.append(np.asarray(phase, dtype=np.float64))
            saved_primary.append(np.asarray(primary, dtype=np.float64))
            saved_target_curve.append(
                np.asarray(target_front["curve"], dtype=np.float64)
            )
            saved_baseline_curve.append(
                np.asarray(baseline_front["curve"], dtype=np.float64)
            )
            saved_phase_curve.append(np.asarray(phase_front["curve"], dtype=np.float64))
            saved_primary_curve.append(
                np.asarray(primary_front["curve"], dtype=np.float64)
            )
            saved_strength.append(
                np.asarray(strength["strength_coefficient"], dtype=np.float64)
            )
            saved_min_jacobian.append(float(warp["minimum_warp_jacobian"]))
            saved_max_jacobian.append(float(warp["maximum_warp_jacobian"]))

        artifact_path = trajectory_dir / f"trajectory_{key}.npz"
        np.savez_compressed(
            artifact_path,
            schema=np.asarray(SCHEMA),
            experiment_id=np.asarray(EXPERIMENT_ID),
            trajectory=np.asarray(key),
            split=np.asarray("validation"),
            parameters_json=artifact["parameters_json"],
            reference_config_json=artifact["reference_config_json"],
            mach=np.asarray(float(reference_config["shock_mach"]), dtype=np.float64),
            gamma=np.asarray(gamma, dtype=np.float64),
            calls=np.asarray(calls, dtype=np.int64),
            physical_times=physical_times[2 * np.asarray(calls, dtype=np.int64)],
            physical_delta_t=np.asarray(
                2.0 * float(baseline_summary["evaluation"]["saved_delta_t"]),
                dtype=np.float64,
            ),
            initial_state=initial.astype(np.float32),
            current_states=np.asarray(saved_current),
            targets=np.asarray(saved_targets),
            d060_predictions=np.asarray(saved_baseline),
            phase_only_predictions=np.asarray(saved_phase),
            phase_strength_predictions=np.asarray(saved_primary),
            target_front_curve=np.asarray(saved_target_curve),
            d060_front_curve=np.asarray(saved_baseline_curve),
            phase_only_front_curve=np.asarray(saved_phase_curve),
            phase_strength_front_curve=np.asarray(saved_primary_curve),
            strength_coefficient=np.asarray(saved_strength),
            minimum_warp_jacobian=np.asarray(saved_min_jacobian),
            maximum_warp_jacobian=np.asarray(saved_max_jacobian),
            front_window_half_width=np.asarray(
                SHOCK_WINDOW_HALF_WIDTH, dtype=np.float64
            ),
            positions=artifact["positions"],
            edges=artifact["edges"],
            node_type=artifact["node_type"],
            physical_cell_volumes=artifact["physical_cell_volumes"],
            mesh_cell_to_graph_node=artifact["mesh_cell_to_graph_node"],
            face_centers=artifact["face_centers"],
            face_measures=artifact["face_measures"],
            face_normals=artifact["face_normals"],
            face_owner=artifact["face_owner"],
            face_neighbor=artifact["face_neighbor"],
            face_axis=artifact["face_axis"],
            face_boundary_tag=artifact["face_boundary_tag"],
            boundary_tag_names_json=artifact["boundary_tag_names_json"],
            coordinate_convention=artifact["coordinate_convention"],
            face_orientation_convention=artifact["face_orientation_convention"],
            reference_interval_boundary_exchange=artifact[
                "reference_interval_boundary_exchange"
            ],
            state_component_scale=artifact["state_component_scale"],
            shock_quantile=artifact["shock_quantile"],
            valid_length=artifact["valid_length"],
            failure_cause=artifact["failure_cause"],
            source_artifact_sha256=np.asarray(artifact_index[key]["sha256"]),
            implementation_sha256=np.asarray(implementation_sha256),
            checkpoint_sha256=np.asarray(source["checkpoint_sha256"]),
            checkpoint_config_digest=np.asarray(source["checkpoint_config_digest"]),
            normalization_digest=np.asarray(source["normalization_digest"]),
            data_manifest_digest=artifact["data_manifest_digest"],
            geometry_digest=artifact["geometry_digest"],
            boundary_mode=np.asarray("model_all_nodes"),
            raw_recurrence=np.asarray(True),
            source_raw_recurrence=np.asarray(True),
            transformed_states_raw_recurrence=np.asarray(False),
            inference_interventions_json=artifact["inference_interventions_json"],
            oracle_target_use=np.asarray(
                "target pressure front and four strength coefficients only"
            ),
        )
        artifacts.append(
            {
                "trajectory": key,
                "artifact": f"trajectories/{artifact_path.name}",
                "sha256": sha256_file(artifact_path),
                "source_artifact_sha256": artifact_index[key]["sha256"],
            }
        )

    decision = promotion_decision(
        rows, expected_trajectories=len(keys), expected_calls=calls
    )
    _write_jsonl(args.output_dir / "oracle_rows.jsonl", rows)
    write_csv(args.output_dir / "variant_metrics.csv", variant_rows)
    retained_without_summary = sum(
        path.stat().st_size for path in args.output_dir.rglob("*") if path.is_file()
    )
    if retained_without_summary > MAX_RETAINED_BYTES:
        raise ValueError("D062 retained output exceeds the frozen 0.25 GiB budget")
    summary = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "status": "complete",
        "implementation": {
            "entry_point": "evaluate_pcno_euler2d_front_fitted_oracle.py",
            "sha256": implementation_sha256,
        },
        "source_contract": {
            "d060": source,
            "trajectory_geometry_checks": contracts,
            "expected_d060_source": D060_SOURCE,
        },
        "evaluation": {
            "split": "validation",
            "trajectory_keys": keys,
            "calls": calls,
            "physical_frames": [2 * call for call in calls],
            "physical_horizon": source["physical_horizon"],
            "shock_window_half_width": SHOCK_WINDOW_HALF_WIDTH,
            "front_extractor": (
                "row-wise pressure-jump argmax inside shock_x +/- 0.10; "
                "jump-weighted centroid of the maximizing x-face and its "
                "immediate in-window neighbors; no smoothing or clipping"
            ),
            "phase_oracle": (
                "target-informed row front displacement; domain-anchored "
                "piecewise-linear map with exact cell-integral pushforward"
            ),
            "strength_oracle": (
                "four target-informed coefficients in one row-zero-total "
                "two-sided contrast; physical-volume least squares"
            ),
            "test_access": False,
            "checkpoint_execution": False,
            "training": False,
            "source_raw_recurrence": True,
            "transformed_states_are_raw_recurrence": False,
        },
        "promotion": decision,
        "cost": {
            "gpu_hours": 0.0,
            "checkpoint_calls": 0,
            "offline_wall_seconds": perf_counter() - started,
            "retained_bytes_excluding_summary": retained_without_summary,
            "maximum_retained_bytes": MAX_RETAINED_BYTES,
        },
        "trajectory_artifacts": artifacts,
        "artifact_schema": {
            "trajectory_npz": (
                "current/target/D060/phase/primary states, all front curves, "
                "strength and warp variables, physical time, graph plus "
                "validated FV geometry, diagnostic weights, boundary exchange, "
                "validity, checkpoint/config/data/geometry provenance, and "
                "float64 transformed states for the conservation gate"
            ),
            "oracle_rows_jsonl": (
                "state/front/high-pass headroom, row-total preservation, warp "
                "and strength variables, correction norms, anti-smearing, and "
                "full variant metrics"
            ),
            "variant_metrics_csv": (
                "state, row-front, graph-front, shock, vortex, smooth-region, "
                "admissibility, physical-total, and boundary metrics"
            ),
        },
        "claim_boundary": {
            "verified": (
                "offline capacity of the exact D062 chart on frozen D060 raw "
                "validation states under the validated finite-volume geometry"
            ),
            "not_verified": [
                "an autonomous front or strength predictor",
                "a trained front-fitted neural operator",
                "test or strength-OOD performance",
                "predicted face flux or reference face-impulse accuracy",
                "physical conservation by the D060 neural recurrence",
            ],
        },
    }
    atomic_write_json(args.output_dir / "summary.json", summary)
    print(
        json.dumps(
            {
                "summary": str(args.output_dir / "summary.json"),
                "decision": decision["decision"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
