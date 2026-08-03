"""Residual-scale sequence diagnostics for frozen PCNO rollouts.

The functions in this module are model-independent.  They operate on saved
conservative states and increments, use physical cell volumes, and never pool
nodes across cases or meshes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

DEFAULT_WAVELENGTH_BANDS = (
    (0.05, 0.125),
    (0.125, 0.25),
    (0.25, 0.5),
    (0.5, np.inf),
)
CONSERVATIVE_COMPONENTS = ("rho", "rho_u", "rho_v", "energy")
CHARACTERISTIC_FAMILIES = (
    "acoustic_minus",
    "entropy",
    "shear",
    "acoustic_plus",
)


def _field(value: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[-1] != 4:
        raise ValueError(f"{name} must have shape [nodes, 4]")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _sequence(value: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 3 or array.shape[-1] != 4:
        raise ValueError(f"{name} must have shape [time, nodes, 4]")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _metric_inputs(
    nodes: int,
    volumes: np.ndarray,
    component_scale: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    weights = np.asarray(volumes, dtype=np.float64).reshape(-1)
    scale = np.asarray(component_scale, dtype=np.float64).reshape(-1)
    if weights.shape != (nodes,) or np.any(weights <= 0.0):
        raise ValueError("volumes must be positive and align with the node axis")
    if scale.shape != (4,) or np.any(scale <= 0.0):
        raise ValueError("component_scale must contain four positive values")
    total_volume = float(weights.sum())
    if not np.isfinite(total_volume) or total_volume <= 0.0:
        raise ValueError("total physical volume must be positive and finite")
    return weights, scale, total_volume


def weighted_inner(
    left: np.ndarray,
    right: np.ndarray,
    *,
    volumes: np.ndarray,
    component_scale: Sequence[float] | np.ndarray,
    mask: np.ndarray | None = None,
) -> float:
    """Physical-volume inner product after fixed component scaling."""

    left_array = _field(left, name="left")
    right_array = _field(right, name="right")
    if left_array.shape != right_array.shape:
        raise ValueError("left and right fields must have identical shapes")
    weights, scale, total_volume = _metric_inputs(
        left_array.shape[0], volumes, component_scale
    )
    selected = np.ones(weights.size, dtype=bool)
    if mask is not None:
        selected = np.asarray(mask)
        if selected.shape != (weights.size,) or selected.dtype != np.bool_:
            raise ValueError("mask must be a boolean vector on the node axis")
        selected_volume = float(weights[selected].sum())
        if selected_volume <= 0.0:
            return 0.0
        total_volume = selected_volume
    scaled_left = left_array[selected] / scale[None, :]
    scaled_right = right_array[selected] / scale[None, :]
    return float(
        np.einsum(
            "n,nc,nc->",
            weights[selected],
            scaled_left,
            scaled_right,
            optimize=True,
        )
        / total_volume
    )


def weighted_rms(
    value: np.ndarray,
    *,
    volumes: np.ndarray,
    component_scale: Sequence[float] | np.ndarray,
    mask: np.ndarray | None = None,
) -> float:
    """Physical-volume RMS in fixed component-scale units."""

    squared = weighted_inner(
        value,
        value,
        volumes=volumes,
        component_scale=component_scale,
        mask=mask,
    )
    return float(np.sqrt(max(squared, 0.0)))


def component_rms(
    value: np.ndarray,
    *,
    volumes: np.ndarray,
    component_scale: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Return one residual-scaled physical RMS per conservative component."""

    array = _field(value, name="value")
    weights, scale, total_volume = _metric_inputs(
        array.shape[0], volumes, component_scale
    )
    energy = np.einsum(
        "n,nc->c", weights, np.square(array / scale[None, :]), optimize=True
    )
    return np.sqrt(np.maximum(energy / total_volume, 0.0))


def safe_ratio(numerator: float, denominator: float) -> float | None:
    """Return an exact ratio without hiding a zero denominator behind epsilon."""

    if not np.isfinite(numerator) or not np.isfinite(denominator):
        raise ValueError("ratio operands must be finite")
    if denominator == 0.0:
        return None
    return float(numerator / denominator)


def cosine(
    left: np.ndarray,
    right: np.ndarray,
    *,
    volumes: np.ndarray,
    component_scale: Sequence[float] | np.ndarray,
) -> float | None:
    left_norm = weighted_rms(left, volumes=volumes, component_scale=component_scale)
    right_norm = weighted_rms(right, volumes=volumes, component_scale=component_scale)
    denominator = left_norm * right_norm
    if denominator == 0.0:
        return None
    return float(
        weighted_inner(
            left,
            right,
            volumes=volumes,
            component_scale=component_scale,
        )
        / denominator
    )


def _sequence_norms(
    sequence: np.ndarray,
    *,
    volumes: np.ndarray,
    component_scale: Sequence[float] | np.ndarray,
) -> np.ndarray:
    array = _sequence(sequence, name="sequence")
    weights, scale, total_volume = _metric_inputs(
        array.shape[1], volumes, component_scale
    )
    energy = np.einsum(
        "n,tnc,tnc->t",
        weights,
        array / scale[None, None, :],
        array / scale[None, None, :],
        optimize=True,
    )
    return np.sqrt(np.maximum(energy / total_volume, 0.0))


def _sequence_dots(
    left: np.ndarray,
    right: np.ndarray,
    *,
    volumes: np.ndarray,
    component_scale: Sequence[float] | np.ndarray,
) -> np.ndarray:
    left_array = _sequence(left, name="left")
    right_array = _sequence(right, name="right")
    if left_array.shape != right_array.shape:
        raise ValueError("sequence shapes must match")
    weights, scale, total_volume = _metric_inputs(
        left_array.shape[1], volumes, component_scale
    )
    return (
        np.einsum(
            "n,tnc,tnc->t",
            weights,
            left_array / scale[None, None, :],
            right_array / scale[None, None, :],
            optimize=True,
        )
        / total_volume
    )


def _pod_summary(matrix: np.ndarray) -> dict[str, Any]:
    if matrix.shape[0] == 0:
        return {
            "first_mode_energy_fraction": None,
            "first_three_energy_fraction": None,
            "modes_for_95_percent": 0,
            "energy_fractions": [],
        }
    gram = matrix @ matrix.T
    eigenvalues = np.linalg.eigvalsh(gram)[::-1]
    eigenvalues = np.maximum(eigenvalues, 0.0)
    total = float(eigenvalues.sum())
    if total == 0.0:
        fractions = np.zeros_like(eigenvalues)
        modes_for_95 = 0
    else:
        fractions = eigenvalues / total
        modes_for_95 = int(np.searchsorted(np.cumsum(fractions), 0.95) + 1)
    return {
        "first_mode_energy_fraction": (float(fractions[0]) if fractions.size else None),
        "first_three_energy_fraction": float(fractions[:3].sum()),
        "modes_for_95_percent": modes_for_95,
        "energy_fractions": fractions.tolist(),
    }


def sequence_diagnostics(
    defects: np.ndarray,
    truth_increments: np.ndarray,
    *,
    volumes: np.ndarray,
    component_scale: Sequence[float] | np.ndarray,
    max_lag: int = 10,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Aggregate a residual-error sequence without discarding vector signs."""

    defect = _sequence(defects, name="defects")
    truth = _sequence(truth_increments, name="truth_increments")
    if defect.shape != truth.shape or defect.shape[0] < 1:
        raise ValueError("defect and truth sequences must align and be non-empty")
    weights, scale, total_volume = _metric_inputs(
        defect.shape[1], volumes, component_scale
    )
    defect_norm = _sequence_norms(defect, volumes=weights, component_scale=scale)
    truth_norm = _sequence_norms(truth, volumes=weights, component_scale=scale)
    cumulative_defect = np.cumsum(defect, axis=0)
    cumulative_truth = np.cumsum(truth, axis=0)
    cumulative_defect_norm = _sequence_norms(
        cumulative_defect, volumes=weights, component_scale=scale
    )
    cumulative_truth_norm = _sequence_norms(
        cumulative_truth, volumes=weights, component_scale=scale
    )
    path = np.cumsum(defect_norm)
    dot_defect_truth = _sequence_dots(
        defect, truth, volumes=weights, component_scale=scale
    )
    truth_energy = np.square(truth_norm)
    cumulative_dot = np.cumsum(dot_defect_truth)
    cumulative_truth_energy = np.cumsum(truth_energy)
    cumulative_beta = np.divide(
        cumulative_dot,
        cumulative_truth_energy,
        out=np.full_like(cumulative_dot, np.nan),
        where=cumulative_truth_energy != 0.0,
    )

    rows: list[dict[str, Any]] = []
    previous_defect: np.ndarray | None = None
    for index in range(defect.shape[0]):
        rows.append(
            {
                "step": index + 1,
                "instant_defect_rms": float(defect_norm[index]),
                "instant_truth_rms": float(truth_norm[index]),
                "instant_relative": safe_ratio(
                    float(defect_norm[index]), float(truth_norm[index])
                ),
                "cumulative_defect_rms": float(cumulative_defect_norm[index]),
                "cumulative_truth_rms": float(cumulative_truth_norm[index]),
                "cumulative_relative": safe_ratio(
                    float(cumulative_defect_norm[index]),
                    float(cumulative_truth_norm[index]),
                ),
                "path_sum_rms": float(path[index]),
                "temporal_coherence": safe_ratio(
                    float(cumulative_defect_norm[index]), float(path[index])
                ),
                "cumulative_parallel_bias": (
                    float(cumulative_beta[index])
                    if np.isfinite(cumulative_beta[index])
                    else None
                ),
                "cumulative_defect_truth_cosine": cosine(
                    cumulative_defect[index],
                    cumulative_truth[index],
                    volumes=weights,
                    component_scale=scale,
                ),
                "successive_defect_cosine": (
                    cosine(
                        previous_defect,
                        defect[index],
                        volumes=weights,
                        component_scale=scale,
                    )
                    if previous_defect is not None
                    else None
                ),
            }
        )
        previous_defect = defect[index]

    total_defect_energy = float(np.square(defect_norm).sum())
    total_truth_energy = float(truth_energy.sum())
    global_beta = safe_ratio(float(dot_defect_truth.sum()), total_truth_energy)
    if global_beta is None:
        perpendicular = defect.copy()
    else:
        perpendicular = defect - global_beta * truth
    perpendicular_energy = float(
        np.square(
            _sequence_norms(perpendicular, volumes=weights, component_scale=scale)
        ).sum()
    )

    weighted = (
        defect / scale[None, None, :] * np.sqrt(weights / total_volume)[None, :, None]
    ).reshape(defect.shape[0], -1)
    centered = weighted - weighted.mean(axis=0, keepdims=True)
    uncentered_pod = _pod_summary(weighted)
    centered_pod = _pod_summary(centered)

    mean_defect = defect.mean(axis=0)
    lag_rows: list[dict[str, Any]] = []
    centered_defect = defect - mean_defect[None, :, :]
    for lag in range(1, min(max_lag, defect.shape[0] - 1) + 1):
        left = centered_defect[:-lag]
        right = centered_defect[lag:]
        numerator = float(
            _sequence_dots(left, right, volumes=weights, component_scale=scale).sum()
        )
        left_energy = float(
            np.square(
                _sequence_norms(left, volumes=weights, component_scale=scale)
            ).sum()
        )
        right_energy = float(
            np.square(
                _sequence_norms(right, volumes=weights, component_scale=scale)
            ).sum()
        )
        lag_rows.append(
            {
                "lag": lag,
                "correlation": safe_ratio(
                    numerator, float(np.sqrt(left_energy * right_energy))
                ),
            }
        )

    summary = {
        "steps": int(defect.shape[0]),
        "residual_error_rms_over_time": float(
            np.sqrt(total_defect_energy / defect.shape[0])
        ),
        "aggregate_relative_residual_energy": (
            float(np.sqrt(total_defect_energy / total_truth_energy))
            if total_truth_energy != 0.0
            else None
        ),
        "path_sum_rms": float(path[-1]),
        "net_defect_rms": float(cumulative_defect_norm[-1]),
        "net_truth_change_rms": float(cumulative_truth_norm[-1]),
        "net_relative_to_truth_change": safe_ratio(
            float(cumulative_defect_norm[-1]),
            float(cumulative_truth_norm[-1]),
        ),
        "temporal_coherence": safe_ratio(
            float(cumulative_defect_norm[-1]), float(path[-1])
        ),
        "global_parallel_bias": global_beta,
        "perpendicular_relative_residual_energy": (
            float(np.sqrt(perpendicular_energy / total_truth_energy))
            if total_truth_energy != 0.0
            else None
        ),
        "net_defect_truth_cosine": cosine(
            cumulative_defect[-1],
            cumulative_truth[-1],
            volumes=weights,
            component_scale=scale,
        ),
        "uncentered_pod": uncentered_pod,
        "centered_pod": centered_pod,
        "lag_correlations": lag_rows,
    }
    return rows, summary


def recurrence_diagnostics(
    errors: np.ndarray,
    defects: np.ndarray,
    *,
    volumes: np.ndarray,
    component_scale: Sequence[float] | np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Verify error recurrence and decompose its signed squared-norm growth."""

    error = _sequence(errors, name="errors")
    defect = _sequence(defects, name="defects")
    if error.shape[0] != defect.shape[0] + 1 or error.shape[1:] != defect.shape[1:]:
        raise ValueError("errors must contain one more state than defects")
    rows: list[dict[str, Any]] = []
    maximum_closure_rms = 0.0
    maximum_growth_closure = 0.0
    for index in range(defect.shape[0]):
        closure = error[index + 1] - error[index] - defect[index]
        closure_rms = weighted_rms(
            closure, volumes=volumes, component_scale=component_scale
        )
        left = weighted_inner(
            error[index + 1],
            error[index + 1],
            volumes=volumes,
            component_scale=component_scale,
        ) - weighted_inner(
            error[index],
            error[index],
            volumes=volumes,
            component_scale=component_scale,
        )
        interaction = 2.0 * weighted_inner(
            error[index],
            defect[index],
            volumes=volumes,
            component_scale=component_scale,
        )
        innovation = weighted_inner(
            defect[index],
            defect[index],
            volumes=volumes,
            component_scale=component_scale,
        )
        growth_closure = left - interaction - innovation
        rows.append(
            {
                "step": index + 1,
                "recurrence_closure_rms": closure_rms,
                "squared_error_growth": left,
                "signed_interaction": interaction,
                "defect_energy": innovation,
                "growth_closure": growth_closure,
            }
        )
        maximum_closure_rms = max(maximum_closure_rms, closure_rms)
        maximum_growth_closure = max(maximum_growth_closure, abs(growth_closure))
    return rows, {
        "maximum_recurrence_closure_rms": maximum_closure_rms,
        "maximum_growth_closure_absolute": maximum_growth_closure,
    }


def algebra_identity_closure(
    coarse_current: np.ndarray,
    coarse_prediction: np.ndarray,
    restricted_fine_current: np.ndarray,
    restricted_fine_prediction: np.ndarray,
) -> np.ndarray:
    """Return D_F - D_N + g for the generalized commutator identity."""

    coarse_u = _field(coarse_current, name="coarse_current")
    coarse_n = _field(coarse_prediction, name="coarse_prediction")
    fine_u = _field(restricted_fine_current, name="restricted_fine_current")
    fine_n = _field(restricted_fine_prediction, name="restricted_fine_prediction")
    if not (coarse_u.shape == coarse_n.shape == fine_u.shape == fine_n.shape):
        raise ValueError("commutator identity fields must align")
    d_f = (coarse_n - coarse_u) - (fine_n - fine_u)
    d_n = coarse_n - fine_n
    gap = coarse_u - fine_u
    return d_f - d_n + gap


def decomposition_diagnostics(
    total: np.ndarray,
    mesh: np.ndarray,
    state: np.ndarray,
    *,
    volumes: np.ndarray,
    component_scale: Sequence[float] | np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Verify delta=delta_mesh+delta_state and attribute the cross term."""

    total_array = _sequence(total, name="total")
    mesh_array = _sequence(mesh, name="mesh")
    state_array = _sequence(state, name="state")
    if not (total_array.shape == mesh_array.shape == state_array.shape):
        raise ValueError("decomposition sequences must align")
    rows: list[dict[str, Any]] = []
    maximum_closure = 0.0
    for index in range(total_array.shape[0]):
        closure = total_array[index] - mesh_array[index] - state_array[index]
        closure_rms = weighted_rms(
            closure, volumes=volumes, component_scale=component_scale
        )
        total_energy = weighted_inner(
            total_array[index],
            total_array[index],
            volumes=volumes,
            component_scale=component_scale,
        )
        mesh_energy = weighted_inner(
            mesh_array[index],
            mesh_array[index],
            volumes=volumes,
            component_scale=component_scale,
        )
        state_energy = weighted_inner(
            state_array[index],
            state_array[index],
            volumes=volumes,
            component_scale=component_scale,
        )
        cross = weighted_inner(
            mesh_array[index],
            state_array[index],
            volumes=volumes,
            component_scale=component_scale,
        )
        rows.append(
            {
                "step": index + 1,
                "decomposition_closure_rms": closure_rms,
                "total_energy": total_energy,
                "mesh_energy": mesh_energy,
                "state_energy": state_energy,
                "twice_cross_term": 2.0 * cross,
                "mesh_symmetric_attribution": safe_ratio(
                    mesh_energy + cross, total_energy
                ),
                "state_symmetric_attribution": safe_ratio(
                    state_energy + cross, total_energy
                ),
            }
        )
        maximum_closure = max(maximum_closure, closure_rms)
    return rows, {"maximum_decomposition_closure_rms": maximum_closure}


def conservative_to_pressure(state: np.ndarray, *, gamma: float) -> np.ndarray:
    array = _field(state, name="state")
    density = array[:, 0]
    if np.any(density <= 0.0):
        raise ValueError("density must be positive")
    kinetic = 0.5 * (np.square(array[:, 1]) + np.square(array[:, 2])) / density
    pressure = (float(gamma) - 1.0) * (array[:, 3] - kinetic)
    if np.any(pressure <= 0.0):
        raise ValueError("pressure must be positive")
    return pressure


def shock_vortex_regions(
    reference_state: np.ndarray,
    nodes: np.ndarray,
    *,
    resolution: tuple[int, int],
    gamma: float,
    boundary_width: float = 0.05,
    shock_core_width: float = 0.02,
    shock_envelope_width: float = 0.05,
    vortex_radius: float = 0.18,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Build fixed-physical masks and an x-dominant shock phase mode."""

    state = _field(reference_state, name="reference_state")
    positions = np.asarray(nodes, dtype=np.float64)
    nx, ny = resolution
    if positions.shape != (nx * ny, 2) or state.shape[0] != nx * ny:
        raise ValueError("state/nodes do not match the declared resolution")
    x = positions[:, 0].reshape(ny, nx)
    y = positions[:, 1].reshape(ny, nx)
    x_values = x[0]
    y_values = y[:, 0]
    dx = float(np.median(np.diff(x_values)))
    dy = float(np.median(np.diff(y_values)))
    if dx <= 0.0 or dy <= 0.0:
        raise ValueError("nodes must be an increasing row-major grid")
    x_min = float(x_values[0] - 0.5 * dx)
    x_max = float(x_values[-1] + 0.5 * dx)
    y_min = float(y_values[0] - 0.5 * dy)
    y_max = float(y_values[-1] + 0.5 * dy)
    distance_to_boundary = np.minimum.reduce(
        (
            positions[:, 0] - x_min,
            x_max - positions[:, 0],
            positions[:, 1] - y_min,
            y_max - positions[:, 1],
        )
    )
    boundary_002 = distance_to_boundary <= 0.02
    boundary_005 = distance_to_boundary <= boundary_width

    pressure = conservative_to_pressure(state, gamma=gamma).reshape(ny, nx)
    pressure_jump = np.abs(np.diff(pressure, axis=1)) / dx
    face_x = 0.5 * (x_values[:-1] + x_values[1:])
    eligible = np.flatnonzero(
        (face_x - x_min >= boundary_width) & (x_max - face_x >= boundary_width)
    )
    if eligible.size == 0:
        raise ValueError("shock search excludes every x face")
    shock_face = np.asarray(
        [eligible[np.argmax(row[eligible])] for row in pressure_jump],
        dtype=np.int64,
    )
    shock_x_by_row = face_x[shock_face]
    shock_distance = np.abs(x - shock_x_by_row[:, None]).reshape(-1)
    shock_core = shock_distance <= shock_core_width
    shock_envelope = shock_distance <= shock_envelope_width

    density = state[:, 0].reshape(ny, nx)
    velocity_x = state[:, 1].reshape(ny, nx) / density
    velocity_y = state[:, 2].reshape(ny, nx) / density
    vorticity = np.gradient(velocity_y, dx, axis=1) - np.gradient(
        velocity_x, dy, axis=0
    )
    vortex_candidates = ~(boundary_005 | shock_envelope)
    if not np.any(vortex_candidates):
        raise ValueError("vortex search mask is empty")
    candidate_strength = np.where(
        vortex_candidates.reshape(ny, nx), np.abs(vorticity), -np.inf
    )
    vortex_flat = int(np.argmax(candidate_strength))
    vortex_y_index, vortex_x_index = np.unravel_index(
        vortex_flat, candidate_strength.shape
    )
    vortex_center = np.asarray(
        [x[vortex_y_index, vortex_x_index], y[vortex_y_index, vortex_x_index]]
    )
    vortex_distance = np.linalg.norm(positions - vortex_center[None, :], axis=1)
    vortex_core = vortex_distance <= vortex_radius

    disjoint_boundary = boundary_005
    disjoint_shock = shock_envelope & ~disjoint_boundary
    disjoint_vortex = vortex_core & ~(disjoint_boundary | disjoint_shock)
    disjoint_smooth = ~(disjoint_boundary | disjoint_shock | disjoint_vortex)
    masks = {
        "boundary_le_0.02": boundary_002,
        "boundary_0.02_to_0.05": boundary_005 & ~boundary_002,
        "boundary_le_0.05": boundary_005,
        "shock_core_le_0.02": shock_core,
        "shock_envelope_le_0.05": shock_envelope,
        "vortex_core_le_0.18": vortex_core,
        "partition_boundary": disjoint_boundary,
        "partition_shock": disjoint_shock,
        "partition_vortex": disjoint_vortex,
        "partition_smooth": disjoint_smooth,
    }
    partition_count = sum(
        masks[name].astype(np.int8)
        for name in (
            "partition_boundary",
            "partition_shock",
            "partition_vortex",
            "partition_smooth",
        )
    )
    if not np.all(partition_count == 1):
        raise AssertionError("physical region partition does not close")

    state_grid = state.reshape(ny, nx, 4)
    phase_mode = np.gradient(state_grid, dx, axis=1).reshape(-1, 4)
    phase_mode[~shock_envelope] = 0.0
    row_slope = np.gradient(shock_x_by_row, dy)
    normals_by_row = np.stack((np.ones_like(row_slope), -row_slope), axis=-1)
    normals_by_row /= np.linalg.norm(normals_by_row, axis=-1, keepdims=True)
    normals = np.repeat(normals_by_row, nx, axis=0)
    return masks, phase_mode, normals


def region_energy_rows(
    defect: np.ndarray,
    *,
    volumes: np.ndarray,
    component_scale: Sequence[float] | np.ndarray,
    masks: Mapping[str, np.ndarray],
) -> list[dict[str, Any]]:
    """Report total-energy share and volume-density enrichment by region."""

    array = _field(defect, name="defect")
    weights, scale, total_volume = _metric_inputs(
        array.shape[0], volumes, component_scale
    )
    node_energy = np.square(array / scale[None, :]).sum(axis=-1)
    total_energy = float(np.dot(weights, node_energy))
    rows = []
    for name, mask_value in masks.items():
        mask = np.asarray(mask_value)
        if mask.shape != (weights.size,) or mask.dtype != np.bool_:
            raise ValueError(f"invalid region mask: {name}")
        region_volume = float(weights[mask].sum())
        region_energy = float(np.dot(weights[mask], node_energy[mask]))
        volume_share = region_volume / total_volume
        energy_share = safe_ratio(region_energy, total_energy)
        rows.append(
            {
                "region": name,
                "node_count": int(mask.sum()),
                "physical_volume": region_volume,
                "volume_share": volume_share,
                "energy_share": energy_share,
                "energy_density_enrichment": (
                    safe_ratio(float(energy_share), volume_share)
                    if energy_share is not None
                    else None
                ),
            }
        )
    return rows


def phase_projection(
    defect: np.ndarray,
    phase_mode: np.ndarray,
    *,
    shock_mask: np.ndarray,
    volumes: np.ndarray,
    component_scale: Sequence[float] | np.ndarray,
) -> dict[str, float | None]:
    """Project a shock-local defect onto the reference translation mode."""

    defect_array = _field(defect, name="defect")
    phase_array = _field(phase_mode, name="phase_mode")
    denominator = weighted_inner(
        phase_array,
        phase_array,
        volumes=volumes,
        component_scale=component_scale,
        mask=shock_mask,
    )
    if denominator == 0.0:
        return {
            "phase_coefficient": None,
            "estimated_displacement": None,
            "phase_energy_fraction": None,
        }
    coefficient = (
        weighted_inner(
            defect_array,
            phase_array,
            volumes=volumes,
            component_scale=component_scale,
            mask=shock_mask,
        )
        / denominator
    )
    projected = coefficient * phase_array
    projected_energy = weighted_inner(
        projected,
        projected,
        volumes=volumes,
        component_scale=component_scale,
        mask=shock_mask,
    )
    defect_energy = weighted_inner(
        defect_array,
        defect_array,
        volumes=volumes,
        component_scale=component_scale,
        mask=shock_mask,
    )
    return {
        "phase_coefficient": float(coefficient),
        "estimated_displacement": float(-coefficient),
        "phase_energy_fraction": safe_ratio(projected_energy, defect_energy),
    }


def _tukey(length: int, alpha: float) -> np.ndarray:
    if length < 2 or not 0.0 <= alpha <= 1.0:
        raise ValueError("invalid Tukey-window arguments")
    if alpha == 0.0:
        return np.ones(length, dtype=np.float64)
    if alpha == 1.0:
        return np.hanning(length)
    x = np.linspace(0.0, 1.0, length)
    window = np.ones(length, dtype=np.float64)
    left = x < alpha / 2.0
    right = x > 1.0 - alpha / 2.0
    window[left] = 0.5 * (1.0 + np.cos(2.0 * np.pi / alpha * (x[left] - alpha / 2.0)))
    window[right] = 0.5 * (
        1.0 + np.cos(2.0 * np.pi / alpha * (x[right] - 1.0 + alpha / 2.0))
    )
    return window


def spectral_energy_rows(
    defect: np.ndarray,
    *,
    resolution: tuple[int, int],
    domain_lengths: tuple[float, float],
    component_scale: Sequence[float] | np.ndarray,
    bands: Sequence[tuple[float, float]] = DEFAULT_WAVELENGTH_BANDS,
    tukey_alpha: float = 0.1,
) -> list[dict[str, Any]]:
    """Windowed common-physical-wavelength energy shares of a defect field."""

    array = _field(defect, name="defect")
    nx, ny = resolution
    lx, ly = (float(value) for value in domain_lengths)
    if array.shape[0] != nx * ny or lx <= 0.0 or ly <= 0.0:
        raise ValueError("defect does not match the declared physical grid")
    scale = np.asarray(component_scale, dtype=np.float64)
    field = array.reshape(ny, nx, 4) / scale[None, None, :]
    field = field - field.mean(axis=(0, 1), keepdims=True)
    window = _tukey(ny, tukey_alpha)[:, None] * _tukey(nx, tukey_alpha)[None, :]
    spectrum = np.fft.rfftn(field * window[:, :, None], axes=(0, 1), norm="ortho")
    mode_energy = np.square(np.abs(spectrum)).sum(axis=-1)
    total_energy = float(mode_energy.sum())
    frequency_x = np.fft.rfftfreq(nx, d=lx / nx)
    frequency_y = np.fft.fftfreq(ny, d=ly / ny)
    radial_frequency = np.hypot(frequency_y[:, None], frequency_x[None, :])
    rows = []
    for wavelength_min, wavelength_max in bands:
        lower_frequency = 0.0 if np.isinf(wavelength_max) else 1.0 / wavelength_max
        upper_frequency = 1.0 / wavelength_min
        mask = (radial_frequency >= lower_frequency) & (
            radial_frequency < upper_frequency
        )
        band_energy = float(mode_energy[mask].sum())
        rows.append(
            {
                "wavelength_min": float(wavelength_min),
                "wavelength_max": (
                    None if np.isinf(wavelength_max) else float(wavelength_max)
                ),
                "mode_count": int(mask.sum()),
                "spectral_energy": band_energy,
                "spectral_energy_share": safe_ratio(band_energy, total_energy),
            }
        )
    return rows


def characteristic_energy(
    defect: np.ndarray,
    reference_state: np.ndarray,
    normals: np.ndarray,
    *,
    shock_mask: np.ndarray,
    gamma: float,
    component_scale: Sequence[float] | np.ndarray,
) -> dict[str, Any]:
    """Decompose shock-local defects in a deterministically normalized Euler basis."""

    delta = _field(defect, name="defect")
    reference = _field(reference_state, name="reference_state")
    normal = np.asarray(normals, dtype=np.float64)
    mask = np.asarray(shock_mask)
    scale = np.asarray(component_scale, dtype=np.float64)
    if normal.shape != (delta.shape[0], 2) or mask.shape != (delta.shape[0],):
        raise ValueError("normal or shock mask does not align with the field")
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return {
            **{
                f"{name}_coefficient_energy_share": None
                for name in CHARACTERISTIC_FAMILIES
            },
            "median_eigenvector_condition_number": None,
            "maximum_eigenvector_condition_number": None,
        }
    coefficient_energy = np.zeros(4, dtype=np.float64)
    conditions = []
    for index in indices:
        rho, momentum_x, momentum_y, energy = reference[index]
        nx_value, ny_value = normal[index]
        tx_value, ty_value = -ny_value, nx_value
        velocity_x = momentum_x / rho
        velocity_y = momentum_y / rho
        velocity_n = velocity_x * nx_value + velocity_y * ny_value
        velocity_t = velocity_x * tx_value + velocity_y * ty_value
        pressure = (gamma - 1.0) * (
            energy - 0.5 * rho * (velocity_x**2 + velocity_y**2)
        )
        sound_speed = float(np.sqrt(gamma * pressure / rho))
        enthalpy = (energy + pressure) / rho
        normal_basis = np.asarray(
            [
                [1.0, 1.0, 0.0, 1.0],
                [velocity_n - sound_speed, velocity_n, 0.0, velocity_n + sound_speed],
                [velocity_t, velocity_t, 1.0, velocity_t],
                [
                    enthalpy - velocity_n * sound_speed,
                    0.5 * (velocity_n**2 + velocity_t**2),
                    velocity_t,
                    enthalpy + velocity_n * sound_speed,
                ],
            ],
            dtype=np.float64,
        )
        basis = normal_basis.copy()
        normal_momentum = basis[1].copy()
        tangent_momentum = basis[2].copy()
        basis[1] = nx_value * normal_momentum + tx_value * tangent_momentum
        basis[2] = ny_value * normal_momentum + ty_value * tangent_momentum
        for column in range(4):
            norm = float(np.linalg.norm(basis[:, column] / scale))
            basis[:, column] /= norm
            first = np.flatnonzero(np.abs(basis[:, column]) > 1.0e-14)
            if first.size and basis[first[0], column] < 0.0:
                basis[:, column] *= -1.0
        conditions.append(float(np.linalg.cond(basis / scale[:, None])))
        coefficients = np.linalg.solve(basis, delta[index])
        coefficient_energy += np.square(coefficients)
    total = float(coefficient_energy.sum())
    result: dict[str, Any] = {
        "median_eigenvector_condition_number": float(np.median(conditions)),
        "maximum_eigenvector_condition_number": float(np.max(conditions)),
    }
    for family, energy in zip(CHARACTERISTIC_FAMILIES, coefficient_energy):
        result[f"{family}_coefficient_energy_share"] = safe_ratio(float(energy), total)
    return result
