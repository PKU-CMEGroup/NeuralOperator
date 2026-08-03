"""Frozen residual corrections for the D071 two-channel diagnostic.

The persistent channel is an offline physical-coordinate cosine bias.  The
local channel is a proposal-only symmetric graph diffusion.  This module is
model independent: callers own checkpoint binding, calibration/evaluation
splits, recurrence, admissibility, and family-specific interpretation of node
weights and node types.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DissipationResult:
    """One controlled-dissipation correction and its audit fields."""

    correction: np.ndarray
    sensor: np.ndarray
    uncapped_relative_norm: float
    applied_relative_norm: float
    applied_scale: float
    weighted_mean_closure: np.ndarray
    eligible_edge_count: int


@dataclass(frozen=True)
class WeightedSubspaceDecomposition:
    """Exact weighted projection with non-type-0 nodes as a third partition."""

    parallel: np.ndarray
    orthogonal: np.ndarray
    excluded_non_type0: np.ndarray
    effective_rank: int
    total_energy: float
    parallel_energy: float
    orthogonal_energy: float
    excluded_non_type0_energy: float
    parallel_orthogonal_inner: float
    component_parallel_orthogonal_inner: np.ndarray
    energy_closure: float
    maximum_reconstruction_error: float
    component_total_energy: np.ndarray
    component_parallel_energy: np.ndarray
    component_orthogonal_energy: np.ndarray
    component_excluded_non_type0_energy: np.ndarray


def _positive_weights(weights: np.ndarray, node_count: int) -> np.ndarray:
    mass = np.asarray(weights, dtype=np.float64).reshape(-1)
    if mass.shape != (node_count,):
        raise ValueError("weights must contain one value per node")
    if not np.isfinite(mass).all() or np.any(mass <= 0.0):
        raise ValueError("weights must be positive and finite")
    return mass


def _component_scale(scale: Sequence[float] | np.ndarray, channels: int) -> np.ndarray:
    value = np.asarray(scale, dtype=np.float64).reshape(-1)
    if value.shape != (channels,):
        raise ValueError("component scale does not match the field")
    if not np.isfinite(value).all() or np.any(value <= 0.0):
        raise ValueError("component scale must be positive and finite")
    return value


def physical_cosine_basis(
    nodes: np.ndarray,
    *,
    rank: int,
    domain_bounds: Sequence[float],
) -> tuple[np.ndarray, tuple[tuple[int, int], ...]]:
    """Evaluate the first separable cosine modes ordered by physical frequency.

    ``domain_bounds`` is ``(x_min, x_max, y_min, y_max)``.  The constant mode
    is first.  Ordering uses ``sqrt((kx/Lx)^2 + (ky/Ly)^2)`` with a stable
    integer tie break, so the selected subspace is resolution independent.
    """

    positions = np.asarray(nodes, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("nodes must have shape [N,2]")
    if not np.isfinite(positions).all() or positions.shape[0] < 1:
        raise ValueError("nodes must be nonempty and finite")
    if int(rank) != rank or rank < 1:
        raise ValueError("rank must be a positive integer")
    bounds = np.asarray(domain_bounds, dtype=np.float64).reshape(-1)
    if bounds.shape != (4,) or not np.isfinite(bounds).all():
        raise ValueError("domain bounds must contain four finite values")
    x_min, x_max, y_min, y_max = bounds
    length_x = x_max - x_min
    length_y = y_max - y_min
    if length_x <= 0.0 or length_y <= 0.0:
        raise ValueError("domain lengths must be positive")
    tolerance = 1.0e-10 * max(length_x, length_y, 1.0)
    if (
        np.any(positions[:, 0] < x_min - tolerance)
        or np.any(positions[:, 0] > x_max + tolerance)
        or np.any(positions[:, 1] < y_min - tolerance)
        or np.any(positions[:, 1] > y_max + tolerance)
    ):
        raise ValueError("nodes lie outside the declared physical domain")

    maximum_index = max(1, int(rank))
    candidates = [
        (kx, ky) for kx in range(maximum_index + 1) for ky in range(maximum_index + 1)
    ]
    candidates.sort(
        key=lambda mode: (
            np.hypot(mode[0] / length_x, mode[1] / length_y),
            mode[0] + mode[1],
            mode[0],
            mode[1],
        )
    )
    modes = tuple(candidates[:rank])
    normalized_x = (positions[:, 0] - x_min) / length_x
    normalized_y = (positions[:, 1] - y_min) / length_y
    basis = np.column_stack(
        [
            np.cos(np.pi * kx * normalized_x) * np.cos(np.pi * ky * normalized_y)
            for kx, ky in modes
        ]
    )
    return np.asarray(basis, dtype=np.float64), modes


def fit_weighted_coefficients(
    field: np.ndarray,
    basis: np.ndarray,
    weights: np.ndarray,
    *,
    ridge: float = 1.0e-10,
) -> np.ndarray:
    """Return proxy/volume-weighted least-squares coefficients ``[R,C]``."""

    values = np.asarray(field, dtype=np.float64)
    design = np.asarray(basis, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or design.ndim != 2 or design.shape[0] != values.shape[0]:
        raise ValueError("field and basis must align as [N,C] and [N,R]")
    if not np.isfinite(values).all() or not np.isfinite(design).all():
        raise ValueError("field and basis must be finite")
    if ridge < 0.0 or not np.isfinite(ridge):
        raise ValueError("ridge must be finite and nonnegative")
    mass = _positive_weights(weights, values.shape[0])
    normalized_mass = mass / mass.sum()
    gram = design.T @ (normalized_mass[:, None] * design)
    scale = float(np.trace(gram) / max(gram.shape[0], 1))
    regularized = gram + float(ridge) * max(scale, 1.0) * np.eye(gram.shape[0])
    right = design.T @ (normalized_mass[:, None] * values)
    return np.linalg.solve(regularized, right)


def fit_error_coefficient_sequence(
    errors: np.ndarray,
    basis: np.ndarray,
    weights: np.ndarray,
    *,
    component_scale: Sequence[float] | np.ndarray,
    ridge: float = 1.0e-10,
) -> np.ndarray:
    """Fit one scaled residual-error coefficient matrix per call."""

    sequence = np.asarray(errors, dtype=np.float64)
    if sequence.ndim != 3:
        raise ValueError("errors must have shape [calls,N,C]")
    scale = _component_scale(component_scale, sequence.shape[-1])
    return np.asarray(
        [
            fit_weighted_coefficients(
                frame / scale[None, :], basis, weights, ridge=ridge
            )
            for frame in sequence
        ],
        dtype=np.float64,
    )


def mean_calibration_coefficients(case_coefficients: np.ndarray) -> np.ndarray:
    """Freeze the call-indexed coefficient mean across calibration cases."""

    values = np.asarray(case_coefficients, dtype=np.float64)
    if values.ndim != 4 or values.shape[0] < 1 or not np.isfinite(values).all():
        raise ValueError("case coefficients must have shape [cases,calls,R,C]")
    return values.mean(axis=0)


def reconstruct_coefficient_sequence(
    coefficients: np.ndarray,
    basis: np.ndarray,
    *,
    component_scale: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Reconstruct physical residual fields ``[calls,N,C]`` from coefficients."""

    values = np.asarray(coefficients, dtype=np.float64)
    design = np.asarray(basis, dtype=np.float64)
    if values.ndim != 3 or design.ndim != 2 or values.shape[1] != design.shape[1]:
        raise ValueError("coefficients and basis do not share one mode dimension")
    scale = _component_scale(component_scale, values.shape[-1])
    return np.einsum("nr,trc->tnc", design, values) * scale[None, None, :]


def weighted_subspace_decomposition(
    field: np.ndarray,
    basis: np.ndarray,
    weights: np.ndarray,
    interior_mask: np.ndarray,
    *,
    component_scale: Sequence[float] | np.ndarray,
) -> WeightedSubspaceDecomposition:
    """Project a field by weighted QR on interior nodes without ridge bias.

    The basis projection and its orthogonal complement live only on the
    selected interior support. Values on all remaining nodes form a separate
    excluded-support partition, so they are never mislabeled as orthogonal
    residuals. Its family-local meaning remains the caller's responsibility.
    Energies use the full-domain weight sum and fixed component scaling.
    """

    values = np.asarray(field, dtype=np.float64)
    design = np.asarray(basis, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or values.shape[0] < 1:
        raise ValueError("field must have finite shape [N,C]")
    if design.ndim != 2 or design.shape[0] != values.shape[0]:
        raise ValueError("basis must have shape [N,R] and align with field")
    if design.shape[1] < 1:
        raise ValueError("basis must contain at least one mode")
    if not np.isfinite(values).all() or not np.isfinite(design).all():
        raise ValueError("field and basis must be finite")
    mass = _positive_weights(weights, values.shape[0])
    scale = _component_scale(component_scale, values.shape[1])
    interior = np.asarray(interior_mask)
    if interior.shape != (values.shape[0],) or interior.dtype != np.bool_:
        raise ValueError("interior_mask must be a boolean vector on the node axis")
    if int(interior.sum()) < design.shape[1]:
        raise ValueError("interior support is smaller than the requested basis")

    square_root_mass = np.sqrt(mass[interior])
    weighted_design = square_root_mass[:, None] * design[interior]
    q_matrix, r_matrix = np.linalg.qr(weighted_design, mode="reduced")
    singular_values = np.linalg.svd(r_matrix, compute_uv=False)
    tolerance = (
        max(weighted_design.shape)
        * np.finfo(np.float64).eps
        * max(float(singular_values[0]), 1.0)
    )
    effective_rank = int(np.count_nonzero(singular_values > tolerance))
    if effective_rank != design.shape[1]:
        raise ValueError("weighted basis is rank deficient on the interior support")

    scaled_interior = values[interior] / scale[None, :]
    weighted_values = square_root_mass[:, None] * scaled_interior
    weighted_projection = q_matrix @ (q_matrix.T @ weighted_values)
    scaled_projection = weighted_projection / square_root_mass[:, None]

    parallel = np.zeros_like(values)
    orthogonal = np.zeros_like(values)
    excluded_non_type0 = np.zeros_like(values)
    parallel[interior] = scaled_projection * scale[None, :]
    orthogonal[interior] = values[interior] - parallel[interior]
    excluded_non_type0[~interior] = values[~interior]

    total_mass = float(mass.sum())

    def component_energy(partition: np.ndarray) -> np.ndarray:
        scaled = partition / scale[None, :]
        return np.einsum("n,nc->c", mass, np.square(scaled), optimize=True) / total_mass

    component_total = component_energy(values)
    component_parallel = component_energy(parallel)
    component_orthogonal = component_energy(orthogonal)
    component_excluded_non_type0 = component_energy(excluded_non_type0)
    component_parallel_orthogonal_inner = (
        np.einsum(
            "n,nc,nc->c",
            mass,
            parallel / scale[None, :],
            orthogonal / scale[None, :],
            optimize=True,
        )
        / total_mass
    )
    parallel_orthogonal_inner = float(component_parallel_orthogonal_inner.sum())
    total_energy = float(component_total.sum())
    parallel_energy = float(component_parallel.sum())
    orthogonal_energy = float(component_orthogonal.sum())
    excluded_non_type0_energy = float(component_excluded_non_type0.sum())
    reconstructed = parallel + orthogonal + excluded_non_type0
    return WeightedSubspaceDecomposition(
        parallel=parallel,
        orthogonal=orthogonal,
        excluded_non_type0=excluded_non_type0,
        effective_rank=effective_rank,
        total_energy=total_energy,
        parallel_energy=parallel_energy,
        orthogonal_energy=orthogonal_energy,
        excluded_non_type0_energy=excluded_non_type0_energy,
        parallel_orthogonal_inner=parallel_orthogonal_inner,
        component_parallel_orthogonal_inner=component_parallel_orthogonal_inner,
        energy_closure=(
            total_energy
            - parallel_energy
            - orthogonal_energy
            - excluded_non_type0_energy
        ),
        maximum_reconstruction_error=float(np.max(np.abs(values - reconstructed))),
        component_total_energy=component_total,
        component_parallel_energy=component_parallel,
        component_orthogonal_energy=component_orthogonal,
        component_excluded_non_type0_energy=component_excluded_non_type0,
    )


def _unique_undirected_edges(edges: np.ndarray, node_count: int) -> np.ndarray:
    edge_index = np.asarray(edges, dtype=np.int64)
    if edge_index.ndim != 2 or edge_index.shape[1] != 2:
        raise ValueError("edges must have shape [E,2]")
    if edge_index.size and (np.any(edge_index < 0) or np.any(edge_index >= node_count)):
        raise ValueError("edge endpoint lies outside the graph")
    ordered = np.sort(edge_index, axis=1)
    ordered = ordered[ordered[:, 0] != ordered[:, 1]]
    if ordered.shape[0] == 0:
        return np.empty((0, 2), dtype=np.int64)
    return np.unique(ordered, axis=0)


def _weighted_rms(field: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(field, dtype=np.float64)
    return float(
        np.sqrt(
            np.sum(weights[:, None] * np.square(values))
            / (weights.sum() * values.shape[1])
        )
    )


def controlled_graph_dissipation(
    update: np.ndarray,
    edges: np.ndarray,
    weights: np.ndarray,
    normal_mask: np.ndarray,
    *,
    component_scale: Sequence[float] | np.ndarray,
    sensor_quantile: float = 0.8,
    norm_cap: float = 0.02,
) -> DissipationResult:
    """Smooth proposal-local graph variation with a symmetric capped update.

    Conductance on an eligible undirected edge ``(i,j)`` is bounded by both
    ``w_i/degree_i`` and ``w_j/degree_j``.  The pair receives equal and opposite
    weighted flux, so the added correction has zero weighted mean for every
    component.  Only the interpretation of ``weights`` determines whether
    that identity is physical (dynamic FV) or a diagnostic proxy (bump).
    """

    values = np.asarray(update, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 1 or not np.isfinite(values).all():
        raise ValueError("update must have finite shape [N,C]")
    mass = _positive_weights(weights, values.shape[0])
    scale = _component_scale(component_scale, values.shape[1])
    normal = np.asarray(normal_mask, dtype=bool).reshape(-1)
    if normal.shape != (values.shape[0],) or not np.any(normal):
        raise ValueError("normal_mask must select at least one node")
    if not 0.0 < sensor_quantile < 1.0:
        raise ValueError("sensor_quantile must lie in (0,1)")
    if not 0.0 < norm_cap <= 1.0:
        raise ValueError("norm_cap must lie in (0,1]")

    unique_edges = _unique_undirected_edges(edges, values.shape[0])
    eligible = (
        normal[unique_edges[:, 0]] & normal[unique_edges[:, 1]]
        if unique_edges.shape[0]
        else np.zeros(0, dtype=bool)
    )
    eligible_edges = unique_edges[eligible]
    scaled = values / scale[None, :]
    all_degree = np.ones(values.shape[0], dtype=np.float64)
    all_average = np.array(scaled, copy=True)
    if eligible_edges.shape[0]:
        left, right = eligible_edges.T
        np.add.at(all_degree, left, 1.0)
        np.add.at(all_degree, right, 1.0)
        np.add.at(all_average, left, scaled[right])
        np.add.at(all_average, right, scaled[left])
    highpass = scaled - all_average / all_degree[:, None]
    amplitude = np.linalg.norm(highpass, axis=1)
    selected_amplitude = amplitude[normal]
    lower = float(np.quantile(selected_amplitude, sensor_quantile))
    upper_quantile = min(0.95, 0.5 * (1.0 + sensor_quantile))
    upper = float(np.quantile(selected_amplitude, upper_quantile))
    denominator = max(upper - lower, np.finfo(np.float64).eps)
    sensor = np.clip((amplitude - lower) / denominator, 0.0, 1.0)
    sensor[~normal] = 0.0

    raw_scaled = np.zeros_like(scaled)
    if eligible_edges.shape[0]:
        left, right = eligible_edges.T
        degree = np.zeros(values.shape[0], dtype=np.float64)
        np.add.at(degree, left, 1.0)
        np.add.at(degree, right, 1.0)
        conductance = (
            0.5
            * (sensor[left] + sensor[right])
            * np.minimum(mass[left] / degree[left], mass[right] / degree[right])
        )
        flux = conductance[:, None] * (scaled[right] - scaled[left])
        np.add.at(raw_scaled, left, flux / mass[left, None])
        np.add.at(raw_scaled, right, -flux / mass[right, None])

    update_norm = _weighted_rms(scaled[normal], mass[normal])
    raw_norm = _weighted_rms(raw_scaled[normal], mass[normal])
    uncapped_relative = 0.0 if update_norm == 0.0 else raw_norm / update_norm
    applied_scale = (
        0.0 if raw_norm == 0.0 else min(1.0, norm_cap * update_norm / raw_norm)
    )
    correction = applied_scale * raw_scaled * scale[None, :]
    applied_relative = (
        0.0
        if update_norm == 0.0
        else _weighted_rms((correction / scale[None, :])[normal], mass[normal])
        / update_norm
    )
    closure = np.sum(mass[:, None] * correction, axis=0) / mass.sum()
    return DissipationResult(
        correction=correction,
        sensor=sensor,
        uncapped_relative_norm=float(uncapped_relative),
        applied_relative_norm=float(applied_relative),
        applied_scale=float(applied_scale),
        weighted_mean_closure=np.asarray(closure, dtype=np.float64),
        eligible_edge_count=int(eligible_edges.shape[0]),
    )
