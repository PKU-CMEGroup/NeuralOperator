"""Graph-native diagnostics for PCNO ripple formation on irregular meshes.

The routines in this module are deliberately diagnostic rather than corrective.
They distinguish finite Fourier-basis truncation from discrete quadrature
leakage, measure graph-spectral error growth, expose PCNO's spectral,
pointwise, and differential branches, and relate local error to mesh geometry.

All node measures used here are explicitly diagnostic proxies.  Nothing in
this module interprets reconstructed vertex weights as physical control-volume
measures or reports a physical conservation balance.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import heapq
import math
from typing import Any

import numpy as np
import torch

from pcno.pcno import compute_Fourier_bases
from utility.time_dependent_no.pcno_euler2d import graph_neighbor_highpass

RIPPLE_DIAGNOSTIC_SCHEMA = "pcno_euler2d_ripple_d013_v1"
DEFAULT_SPECTRAL_BANDS = (0.0, 0.1, 0.3, 0.6, 1.0000001)


def conservative_to_primitive_raw(
    conservative: np.ndarray,
    *,
    gamma: float = 1.4,
) -> np.ndarray:
    """Convert ``[rho, rho*u, rho*v, E]`` without clipping or floors."""

    state = np.asarray(conservative, dtype=np.float64)
    if state.shape[-1] != 4:
        raise ValueError("conservative state must have four components")
    if gamma <= 1.0:
        raise ValueError("gamma must be greater than one")
    rho = state[..., 0]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        velocity_x = state[..., 1] / rho
        velocity_y = state[..., 2] / rho
        internal_energy = (
            state[..., 3] - 0.5 * (state[..., 1] ** 2 + state[..., 2] ** 2) / rho
        )
        pressure = (gamma - 1.0) * internal_energy
    return np.stack((rho, velocity_x, velocity_y, pressure), axis=-1)


def raw_admissibility_summary(
    conservative: np.ndarray,
    *,
    gamma: float = 1.4,
) -> dict[str, Any]:
    """Summarize finiteness and Euler admissibility without modifying state."""

    state = np.asarray(conservative, dtype=np.float64)
    primitive = conservative_to_primitive_raw(state, gamma=gamma)
    rho = state[..., 0]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        internal = state[..., 3] - 0.5 * (state[..., 1] ** 2 + state[..., 2] ** 2) / rho
    pressure = primitive[..., 3]
    finite_components = np.isfinite(state).all(axis=-1)
    admissible = (
        finite_components
        & np.isfinite(internal)
        & np.isfinite(pressure)
        & (rho > 0.0)
        & (internal > 0.0)
        & (pressure > 0.0)
    )

    def finite_minimum(value: np.ndarray) -> float | None:
        finite = np.asarray(value)[np.isfinite(value)]
        return None if finite.size == 0 else float(np.min(finite))

    return {
        "all_finite": bool(np.all(finite_components)),
        "all_admissible": bool(np.all(admissible)),
        "inadmissible_node_count": int(np.count_nonzero(~admissible)),
        "min_density": finite_minimum(rho),
        "min_internal_energy": finite_minimum(internal),
        "min_pressure": finite_minimum(pressure),
    }


def normalized_node_weights(weights: np.ndarray, *, name: str) -> np.ndarray:
    """Return a positive unit-sum diagnostic node measure."""

    value = np.asarray(weights, dtype=np.float64)
    if value.ndim == 2:
        value = value.sum(axis=-1)
    if value.ndim != 1:
        raise ValueError(f"{name} weights must have shape [N] or [N,M]")
    if not np.isfinite(value).all() or np.any(value <= 0.0):
        raise ValueError(f"{name} weights must be finite and strictly positive")
    total = float(value.sum())
    if total <= 0.0:
        raise ValueError(f"{name} weights must have positive sum")
    return value / total


def _validated_edges(edges: np.ndarray, num_nodes: int) -> np.ndarray:
    edge_index = np.asarray(edges, dtype=np.int64)
    if edge_index.ndim != 2 or edge_index.shape[1] != 2:
        raise ValueError("edges must have shape [E,2]")
    if edge_index.size and (
        int(edge_index.min()) < 0 or int(edge_index.max()) >= num_nodes
    ):
        raise ValueError("edge index lies outside the node array")
    if edge_index.shape[0] == 0:
        raise ValueError("graph diagnostics require at least one edge")
    return edge_index


def real_fourier_basis(
    nodes: np.ndarray,
    modes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return PCNO's real ``[1, cos(kx), sin(kx)]`` basis and magnitudes."""

    positions = np.asarray(nodes, dtype=np.float64)
    wave_numbers = np.asarray(modes, dtype=np.float64)
    if positions.ndim != 2:
        raise ValueError("nodes must have shape [N,D]")
    if wave_numbers.ndim == 3:
        if wave_numbers.shape[-1] != 1:
            raise ValueError("the current diagnostic supports one PCNO measure")
        wave_numbers = wave_numbers[..., 0]
    if wave_numbers.ndim != 2 or wave_numbers.shape[1] != positions.shape[1]:
        raise ValueError("modes must have shape [K,D] or [K,D,1]")
    phase = positions @ wave_numbers.T
    basis = np.concatenate(
        (
            np.ones((positions.shape[0], 1), dtype=np.float64),
            np.cos(phase),
            np.sin(phase),
        ),
        axis=1,
    )
    magnitude = np.linalg.norm(wave_numbers, axis=1)
    column_magnitude = np.concatenate(([0.0], magnitude, magnitude))
    return basis, column_magnitude


def fourier_gram_audit(
    nodes: np.ndarray,
    modes: np.ndarray,
    weights: np.ndarray,
    *,
    rcond: float = 1e-10,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Audit discrete PCNO-basis orthogonality under one node measure."""

    if not 0.0 < rcond < 1.0:
        raise ValueError("rcond must lie in (0,1)")
    basis, magnitude = real_fourier_basis(nodes, modes)
    mass = normalized_node_weights(weights, name="Fourier Gram")
    if mass.shape[0] != basis.shape[0]:
        raise ValueError("weights and nodes must have the same length")
    gram = basis.T @ (mass[:, None] * basis)
    diagonal = np.diag(gram)
    if np.any(diagonal <= 0.0):
        raise ValueError("Fourier Gram matrix has a nonpositive diagonal")
    inverse_sqrt = 1.0 / np.sqrt(diagonal)
    correlation = gram * inverse_sqrt[:, None] * inverse_sqrt[None, :]
    correlation = 0.5 * (correlation + correlation.T)
    eigenvalues = np.linalg.eigvalsh(correlation)
    largest = float(np.max(eigenvalues))
    retained = eigenvalues > largest * rcond
    rank = int(np.count_nonzero(retained))
    condition = float(largest / np.min(eigenvalues[retained])) if rank else math.inf
    off_diagonal = correlation - np.diag(np.diag(correlation))
    diagonal_norm = float(np.linalg.norm(np.diag(np.diag(correlation))))

    positive_modes = magnitude[magnitude > 0.0]
    if positive_modes.size:
        q1, q2 = np.quantile(positive_modes, (1.0 / 3.0, 2.0 / 3.0))
        band_id = np.digitize(magnitude, (q1, q2), right=True)
    else:
        q1 = q2 = 0.0
        band_id = np.zeros_like(magnitude, dtype=np.int64)
    cross_band = band_id[:, None] != band_id[None, :]
    cross_band[0, :] = False
    cross_band[:, 0] = False
    cross_band_fraction = float(
        np.linalg.norm(off_diagonal * cross_band)
        / max(np.linalg.norm(correlation), np.finfo(np.float64).eps)
    )
    summary = {
        "basis_columns": int(basis.shape[1]),
        "numerical_rank": rank,
        "rank_fraction": float(rank / basis.shape[1]),
        "correlation_condition_number": condition,
        "correlation_min_eigenvalue": float(np.min(eigenvalues)),
        "correlation_max_eigenvalue": largest,
        "off_diagonal_frobenius_ratio": float(
            np.linalg.norm(off_diagonal) / max(diagonal_norm, np.finfo(float).eps)
        ),
        "max_absolute_off_diagonal": float(np.max(np.abs(off_diagonal))),
        "constant_mode_max_coupling": float(np.max(np.abs(correlation[0, 1:]))),
        "cross_frequency_band_leakage": cross_band_fraction,
        "frequency_band_cutoffs": [float(q1), float(q2)],
        "rcond": float(rcond),
    }
    arrays = {
        "gram": gram,
        "correlation_gram": correlation,
        "correlation_eigenvalues": eigenvalues,
        "basis_column_magnitude": magnitude,
    }
    return summary, arrays


def _weighted_error_summary(
    reconstruction: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    eps: float = 1e-12,
) -> dict[str, float]:
    prediction = np.asarray(reconstruction, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    if prediction.shape != truth.shape or prediction.ndim != 2:
        raise ValueError("reconstruction and target must have shape [N,C]")
    mass = normalized_node_weights(weights, name="reconstruction")
    if mask is not None:
        selected = np.asarray(mask, dtype=bool)
        if selected.shape != (truth.shape[0],) or not np.any(selected):
            raise ValueError("region mask must select at least one node")
        prediction = prediction[selected]
        truth = truth[selected]
        mass = mass[selected]
        mass = mass / mass.sum()
    error = prediction - truth
    error_energy = float(np.sum(mass[:, None] * error**2))
    target_energy = float(np.sum(mass[:, None] * truth**2))
    return {
        "rmse": float(np.sqrt(error_energy)),
        "relative_l2": float(np.sqrt(error_energy / max(target_energy, eps))),
        "max_absolute_error": float(np.max(np.abs(error))),
    }


def fourier_reconstruction_audit(
    nodes: np.ndarray,
    modes: np.ndarray,
    weights: np.ndarray,
    field: np.ndarray,
    *,
    regions: Mapping[str, np.ndarray] | None = None,
    rcond: float = 1e-10,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Compare PCNO's uniform formula with a mass-orthogonal projection.

    The first reconstruction applies exactly the coefficient convention used by
    the PCNO spectral transform for an identity modal multiplier.  The second
    solves the weighted normal equations.  A large improvement in the second
    result is evidence for discrete quadrature leakage, not classic Gibbs
    ringing alone.
    """

    values = np.asarray(field, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("field must be a finite [N,C] array")
    basis, _ = real_fourier_basis(nodes, modes)
    mass = normalized_node_weights(weights, name="Fourier reconstruction")
    if values.shape[0] != basis.shape[0] or mass.shape[0] != basis.shape[0]:
        raise ValueError("nodes, field, and weights must share the node axis")
    mode_count = (basis.shape[1] - 1) // 2
    coefficients = basis.T @ (mass[:, None] * values)
    naive = (
        basis[:, :1] @ coefficients[:1]
        + 2.0 * basis[:, 1 : 1 + mode_count] @ coefficients[1 : 1 + mode_count]
        + 2.0 * basis[:, 1 + mode_count :] @ coefficients[1 + mode_count :]
    )
    gram = basis.T @ (mass[:, None] * basis)
    orthogonal_coefficients = np.linalg.pinv(gram, rcond=rcond) @ coefficients
    orthogonal = basis @ orthogonal_coefficients

    target_min = np.min(values, axis=0)
    target_max = np.max(values, axis=0)
    target_range = np.maximum(target_max - target_min, 1e-12)

    def reconstruction_summary(reconstruction: np.ndarray) -> dict[str, Any]:
        result: dict[str, Any] = _weighted_error_summary(reconstruction, values, mass)
        result["overshoot_fraction_by_component"] = (
            np.maximum(np.max(reconstruction, axis=0) - target_max, 0.0) / target_range
        ).tolist()
        result["undershoot_fraction_by_component"] = (
            np.maximum(target_min - np.min(reconstruction, axis=0), 0.0) / target_range
        ).tolist()
        if regions:
            result["regions"] = {
                name: _weighted_error_summary(
                    reconstruction,
                    values,
                    mass,
                    mask=np.asarray(mask, dtype=bool),
                )
                for name, mask in regions.items()
                if np.any(mask)
            }
        return result

    naive_summary = reconstruction_summary(naive)
    orthogonal_summary = reconstruction_summary(orthogonal)
    improvement = float(naive_summary["rmse"] / max(orthogonal_summary["rmse"], 1e-12))
    summary = {
        "pcno_uniform_formula": naive_summary,
        "mass_orthogonal_projection": orthogonal_summary,
        "rmse_improvement_factor": improvement,
        "interpretation": (
            "improvement isolates discrete Gram/quadrature leakage; residual "
            "ringing in the orthogonal projection is a finite-basis limit"
        ),
    }
    return summary, {
        "pcno_uniform_reconstruction": naive,
        "mass_orthogonal_reconstruction": orthogonal,
        "target": values,
    }


def _normalized_laplacian_matvec(
    vector: np.ndarray,
    edges: np.ndarray,
    mass: np.ndarray,
) -> np.ndarray:
    """Apply ``M^-1/2 B^T B M^-1/2`` without forming a sparse matrix."""

    value = np.asarray(vector, dtype=np.float64)
    root_mass = np.sqrt(mass)
    scaled = value / root_mass
    source = edges[:, 0]
    target = edges[:, 1]
    jump = scaled[source] - scaled[target]
    output = np.bincount(source, weights=jump, minlength=scaled.shape[0])
    output -= np.bincount(target, weights=jump, minlength=scaled.shape[0])
    return output / root_mass


def estimate_generalized_laplacian_radius(
    edges: np.ndarray,
    weights: np.ndarray,
    *,
    iterations: int = 30,
    seed: int = 0,
) -> float:
    """Estimate the largest generalized graph-Laplacian eigenvalue."""

    if iterations < 2:
        raise ValueError("power iterations must be at least two")
    mass = normalized_node_weights(weights, name="graph spectrum")
    edge_index = _validated_edges(edges, mass.shape[0])
    generator = np.random.default_rng(seed)
    vector = generator.standard_normal(mass.shape[0])
    vector -= vector.mean()
    vector /= max(float(np.linalg.norm(vector)), np.finfo(float).eps)
    radius = 0.0
    for _ in range(iterations):
        product = _normalized_laplacian_matvec(vector, edge_index, mass)
        norm = float(np.linalg.norm(product))
        if norm <= np.finfo(float).eps:
            return 0.0
        vector = product / norm
        radius = float(vector @ _normalized_laplacian_matvec(vector, edge_index, mass))
    return max(radius, 0.0)


def _lanczos_measure(
    signal: np.ndarray,
    edges: np.ndarray,
    mass: np.ndarray,
    *,
    steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return Lanczos Ritz values and signal-energy quadrature weights."""

    value = np.asarray(signal, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if norm <= np.finfo(float).eps:
        return np.asarray([0.0]), np.asarray([0.0])
    q = value / norm
    previous = np.zeros_like(q)
    previous_beta = 0.0
    basis_vectors: list[np.ndarray] = []
    diagonal: list[float] = []
    off_diagonal: list[float] = []
    for index in range(steps):
        product = _normalized_laplacian_matvec(q, edges, mass)
        if index:
            product -= previous_beta * previous
        alpha = float(q @ product)
        product -= alpha * q
        # Full reorthogonalization prevents duplicated Ritz values from
        # masquerading as high-band mass at this bounded diagnostic depth.
        for old_q in basis_vectors:
            product -= float(old_q @ product) * old_q
        diagonal.append(alpha)
        basis_vectors.append(q.copy())
        beta = float(np.linalg.norm(product))
        if index == steps - 1 or beta <= 1e-12:
            break
        off_diagonal.append(beta)
        previous, q = q, product / beta
        previous_beta = beta

    tridiagonal = np.diag(np.asarray(diagonal, dtype=np.float64))
    if off_diagonal:
        off = np.asarray(off_diagonal, dtype=np.float64)
        tridiagonal += np.diag(off, 1) + np.diag(off, -1)
    eigenvalues, eigenvectors = np.linalg.eigh(tridiagonal)
    spectral_weights = norm**2 * eigenvectors[0] ** 2
    return eigenvalues, spectral_weights


def graph_spectral_bands(
    field: np.ndarray,
    edges: np.ndarray,
    weights: np.ndarray,
    *,
    bands: Sequence[float] = DEFAULT_SPECTRAL_BANDS,
    lanczos_steps: int = 24,
    radius: float | None = None,
    radius_iterations: int = 30,
    seed: int = 0,
) -> dict[str, Any]:
    """Approximate generalized graph-spectral energy in normalized bands.

    ``B^T B phi = lambda M phi`` uses an unweighted graph incidence matrix and
    a declared diagnostic node measure ``M``.  Both the graph operator and node
    measures are proxies on the bump artifact; the result is not a physical
    modal decomposition.
    """

    values = np.asarray(field, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("graph-spectral field must be a finite [N,C] array")
    if lanczos_steps < 2:
        raise ValueError("lanczos_steps must be at least two")
    boundaries = np.asarray(bands, dtype=np.float64)
    if (
        boundaries.ndim != 1
        or boundaries.size < 2
        or boundaries[0] != 0.0
        or np.any(np.diff(boundaries) <= 0.0)
        or boundaries[-1] < 1.0
    ):
        raise ValueError("bands must increase from zero through at least one")
    mass = normalized_node_weights(weights, name="graph spectrum")
    if mass.shape[0] != values.shape[0]:
        raise ValueError("graph-spectral weights and field must align")
    edge_index = _validated_edges(edges, values.shape[0])
    spectral_radius = (
        estimate_generalized_laplacian_radius(
            edge_index,
            mass,
            iterations=radius_iterations,
            seed=seed,
        )
        if radius is None
        else float(radius)
    )
    if not np.isfinite(spectral_radius) or spectral_radius <= 0.0:
        raise ValueError("generalized Laplacian spectral radius must be positive")

    band_energy = np.zeros(boundaries.size - 1, dtype=np.float64)
    component_energy = []
    for component in range(values.shape[1]):
        mass_scaled = np.sqrt(mass) * values[:, component]
        eigenvalues, spectral_weights = _lanczos_measure(
            mass_scaled,
            edge_index,
            mass,
            steps=lanczos_steps,
        )
        normalized_frequency = np.clip(eigenvalues / spectral_radius, 0.0, 1.0)
        ids = np.searchsorted(boundaries, normalized_frequency, side="right") - 1
        ids = np.clip(ids, 0, band_energy.size - 1)
        for band_index in range(band_energy.size):
            band_energy[band_index] += float(
                np.sum(spectral_weights[ids == band_index])
            )
        component_energy.append(float(np.sum(spectral_weights)))
    total = float(np.sum(band_energy))
    fraction = band_energy / max(total, np.finfo(float).eps)
    return {
        "band_edges_normalized": boundaries.tolist(),
        "band_energy": band_energy.tolist(),
        "band_fraction": fraction.tolist(),
        "total_weighted_energy": total,
        "high_band_energy": float(band_energy[-1]),
        "high_band_fraction": float(fraction[-1]),
        "component_energy": component_energy,
        "estimated_spectral_radius": spectral_radius,
        "lanczos_steps": int(lanczos_steps),
        "operator": "generalized_proxy_graph_laplacian_BtB_phi=lambda_M_phi",
    }


def induced_subgraph(
    field: np.ndarray,
    edges: np.ndarray,
    weights: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Restrict a field and graph to a selected node region."""

    values = np.asarray(field)
    selected = np.asarray(mask, dtype=bool)
    if selected.shape != (values.shape[0],) or np.count_nonzero(selected) < 2:
        raise ValueError("induced subgraph mask must select at least two nodes")
    edge_index = _validated_edges(edges, values.shape[0])
    kept_edges = edge_index[selected[edge_index[:, 0]] & selected[edge_index[:, 1]]]
    if kept_edges.shape[0] == 0:
        raise ValueError("induced subgraph contains no edges")
    remapping = np.full(values.shape[0], -1, dtype=np.int64)
    remapping[selected] = np.arange(np.count_nonzero(selected), dtype=np.int64)
    restricted_weights = np.asarray(weights)
    if restricted_weights.ndim == 2:
        restricted_weights = restricted_weights.sum(axis=-1)
    return values[selected], remapping[kept_edges], restricted_weights[selected]


def node_highpass_field(field: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Return the linear self-plus-neighbor graph high-pass field."""

    values = np.asarray(field, dtype=np.float64)
    squeeze = values.ndim == 1
    if squeeze:
        values = values[:, None]
    if values.ndim != 2:
        raise ValueError("high-pass field must have shape [N,C]")
    edge_index = _validated_edges(edges, values.shape[0])
    source = edge_index[:, 0]
    target = edge_index[:, 1]
    aggregate = values.copy()
    degree = np.ones(values.shape[0], dtype=np.float64)
    np.add.at(aggregate, source, values[target])
    np.add.at(aggregate, target, values[source])
    np.add.at(degree, source, 1.0)
    np.add.at(degree, target, 1.0)
    highpass = values - aggregate / degree[:, None]
    return highpass[:, 0] if squeeze else highpass


def node_highpass_amplitude(field: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Return norm of node value minus self-plus-neighbor graph average."""

    highpass = node_highpass_field(field, edges)
    if highpass.ndim == 1:
        return np.abs(highpass)
    return np.linalg.norm(highpass, axis=-1)


def _multi_source_graph_distance(
    nodes: np.ndarray,
    edges: np.ndarray,
    source_mask: np.ndarray,
) -> np.ndarray:
    positions = np.asarray(nodes, dtype=np.float64)
    edge_index = _validated_edges(edges, positions.shape[0])
    selected = np.asarray(source_mask, dtype=bool)
    if selected.shape != (positions.shape[0],) or not np.any(selected):
        return np.full(positions.shape[0], np.nan, dtype=np.float64)
    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(positions.shape[0])]
    lengths = np.linalg.norm(
        positions[edge_index[:, 0]] - positions[edge_index[:, 1]], axis=-1
    )
    for (left, right), length in zip(edge_index, lengths, strict=True):
        distance = float(length)
        adjacency[int(left)].append((int(right), distance))
        adjacency[int(right)].append((int(left), distance))
    result = np.full(positions.shape[0], np.inf, dtype=np.float64)
    queue: list[tuple[float, int]] = []
    for source in np.flatnonzero(selected):
        result[source] = 0.0
        heapq.heappush(queue, (0.0, int(source)))
    while queue:
        distance, node = heapq.heappop(queue)
        if distance != result[node]:
            continue
        for neighbor, edge_length in adjacency[node]:
            candidate = distance + edge_length
            if candidate < result[neighbor]:
                result[neighbor] = candidate
                heapq.heappush(queue, (candidate, neighbor))
    result[~np.isfinite(result)] = np.nan
    return result


def graph_distance_to_mask(
    nodes: np.ndarray,
    edges: np.ndarray,
    source_mask: np.ndarray,
) -> np.ndarray:
    """Return physical edge-path distance to a declared node set."""

    return _multi_source_graph_distance(nodes, edges, source_mask)


def geometry_conditioning_features(
    nodes: np.ndarray,
    edges: np.ndarray,
    directed_edges: np.ndarray,
    node_type: np.ndarray,
    node_measures: np.ndarray,
    *,
    shock_mask: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Return declared mesh-density, stencil, and distance predictors."""

    positions = np.asarray(nodes, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("geometry features require [N,2] node positions")
    edge_index = _validated_edges(edges, positions.shape[0])
    directed = np.asarray(directed_edges, dtype=np.int64)
    if directed.ndim != 2 or directed.shape[1] != 2:
        raise ValueError("directed_edges must have shape [E,2]")
    if directed.size and (directed.min() < 0 or directed.max() >= positions.shape[0]):
        raise ValueError("directed edge index lies outside node positions")
    types = np.asarray(node_type).reshape(-1)
    if types.shape != (positions.shape[0],):
        raise ValueError("node_type must contain one value per node")
    measures = np.asarray(node_measures, dtype=np.float64)
    if measures.ndim == 2:
        measures = measures.sum(axis=-1)
    if measures.shape != (positions.shape[0],):
        raise ValueError("node_measures must contain one value per node")

    left = edge_index[:, 0]
    right = edge_index[:, 1]
    edge_length = np.linalg.norm(positions[left] - positions[right], axis=-1)
    degree = np.zeros(positions.shape[0], dtype=np.float64)
    length_sum = np.zeros_like(degree)
    np.add.at(degree, left, 1.0)
    np.add.at(degree, right, 1.0)
    np.add.at(length_sum, left, edge_length)
    np.add.at(length_sum, right, edge_length)
    mean_edge_length = np.divide(
        length_sum,
        degree,
        out=np.full_like(length_sum, np.nan),
        where=degree > 0.0,
    )

    target = directed[:, 0]
    source = directed[:, 1]
    displacement = positions[source] - positions[target]
    sxx = np.zeros(positions.shape[0], dtype=np.float64)
    sxy = np.zeros_like(sxx)
    syy = np.zeros_like(sxx)
    np.add.at(sxx, target, displacement[:, 0] ** 2)
    np.add.at(sxy, target, displacement[:, 0] * displacement[:, 1])
    np.add.at(syy, target, displacement[:, 1] ** 2)
    trace = sxx + syy
    discriminant = np.sqrt(np.maximum((sxx - syy) ** 2 + 4.0 * sxy**2, 0.0))
    maximum_eigenvalue = 0.5 * (trace + discriminant)
    minimum_eigenvalue = 0.5 * (trace - discriminant)
    valid_stencil = minimum_eigenvalue > (np.maximum(maximum_eigenvalue, 1.0) * 1e-14)
    stencil_condition = np.full(positions.shape[0], np.nan, dtype=np.float64)
    stencil_condition[valid_stencil] = np.sqrt(
        maximum_eigenvalue[valid_stencil] / minimum_eigenvalue[valid_stencil]
    )

    boundary_mask = types != 0
    shock = (
        np.zeros(positions.shape[0], dtype=bool)
        if shock_mask is None
        else np.asarray(shock_mask, dtype=bool)
    )
    if shock.shape != (positions.shape[0],):
        raise ValueError("shock_mask must contain one boolean per node")
    cell_size = np.full_like(measures, np.nan)
    positive_measure = measures > 0.0
    cell_size[positive_measure] = np.sqrt(measures[positive_measure])
    return {
        "node_degree": degree,
        "mean_edge_length": mean_edge_length,
        "mesh_density_proxy": np.divide(
            1.0,
            mean_edge_length,
            out=np.full_like(mean_edge_length, np.nan),
            where=mean_edge_length > 0.0,
        ),
        "cell_size_proxy": cell_size,
        "stencil_condition": stencil_condition,
        "boundary_graph_distance": _multi_source_graph_distance(
            positions, edge_index, boundary_mask
        ),
        "shock_graph_distance": _multi_source_graph_distance(
            positions, edge_index, shock
        ),
    }


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denominator <= np.finfo(float).eps:
        return None
    return float((x @ y) / denominator)


def spatial_correlation_summary(
    response: np.ndarray,
    features: Mapping[str, np.ndarray],
    *,
    mask: np.ndarray | None = None,
) -> dict[str, dict[str, float | int | None]]:
    """Return Pearson and Spearman correlations for finite selected nodes."""

    value = np.asarray(response, dtype=np.float64).reshape(-1)
    base = np.isfinite(value)
    if mask is not None:
        selected = np.asarray(mask, dtype=bool)
        if selected.shape != value.shape:
            raise ValueError("correlation mask must match response")
        base &= selected
    result: dict[str, dict[str, float | int | None]] = {}
    for name, feature in features.items():
        predictor = np.asarray(feature, dtype=np.float64).reshape(-1)
        if predictor.shape != value.shape:
            raise ValueError(f"feature {name} does not match response")
        finite = base & np.isfinite(predictor)
        count = int(np.count_nonzero(finite))
        if count < 3:
            result[name] = {"count": count, "pearson": None, "spearman": None}
            continue
        x = predictor[finite]
        y = value[finite]
        result[name] = {
            "count": count,
            "pearson": _pearson(x, y),
            "spearman": _pearson(_average_ranks(x), _average_ranks(y)),
        }
    return result


def weighted_relative_l2_numpy(
    prediction: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    component_scale: np.ndarray,
    *,
    eps: float = 1e-12,
) -> float:
    """Return scaled diagnostic relative L2 under a declared node measure."""

    pred = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    scale = np.asarray(component_scale, dtype=np.float64).reshape(1, -1)
    if pred.shape != truth.shape or pred.ndim != 2 or pred.shape[1] != scale.shape[1]:
        raise ValueError("prediction, target, and component scale do not align")
    mass = normalized_node_weights(weights, name="relative L2")
    error_energy = np.sum(mass[:, None] * ((pred - truth) / scale) ** 2)
    target_energy = np.sum(mass[:, None] * (truth / scale) ** 2)
    return float(np.sqrt(error_energy / max(float(target_energy), eps)))


def _torch_branch_stats(
    value: torch.Tensor,
    node_weights: torch.Tensor,
    directed_edges: torch.Tensor,
) -> dict[str, float]:
    mass = node_weights.sum(dim=-1)
    weighted_energy = (mass.unsqueeze(1) * value.square()).sum() / (
        mass.sum() * value.shape[1]
    ).clamp_min(1e-12)
    batch = torch.arange(value.shape[0], device=value.device).unsqueeze(1)
    target = directed_edges[..., 0]
    source = directed_edges[..., 1]
    source_value = value.permute(0, 2, 1)[batch, source]
    target_value = value.permute(0, 2, 1)[batch, target]
    jump_energy = (source_value - target_value).square().mean()
    return {
        "weighted_rms": float(torch.sqrt(weighted_energy).detach().cpu()),
        "mean_directed_edge_jump_rms": float(torch.sqrt(jump_energy).detach().cpu()),
        "edge_to_node_energy_ratio": float(
            (jump_energy / weighted_energy.clamp_min(1e-12)).detach().cpu()
        ),
    }


def _torch_weighted_cosine(
    left: torch.Tensor,
    right: torch.Tensor,
    node_weights: torch.Tensor,
) -> float:
    mass = node_weights.sum(dim=-1).unsqueeze(1)
    numerator = (mass * left * right).sum()
    denominator = torch.sqrt(
        (mass * left.square()).sum() * (mass * right.square()).sum()
    )
    return float((numerator / denominator.clamp_min(1e-12)).detach().cpu())


def _torch_smooth_highpass_energy_decomposition(
    branches: Mapping[str, torch.Tensor],
    *,
    directed_edges: torch.Tensor,
    node_mask: torch.Tensor,
    smooth_region_mask: torch.Tensor,
    weight_maps: Mapping[str, torch.Tensor],
) -> dict[str, dict[str, Any]]:
    """Decompose smooth-region graph-high-pass energy across additive branches."""

    if not branches:
        raise ValueError("at least one branch is required")
    first = next(iter(branches.values()))
    if first.ndim != 3:
        raise ValueError("branch tensors must have shape [B,C,N]")
    batch_size, channels, num_nodes = first.shape
    for name, value in branches.items():
        if value.shape != first.shape:
            raise ValueError(f"branch {name!r} does not match the other branches")
    if smooth_region_mask.shape == (batch_size, num_nodes, 1):
        smooth_region_mask = smooth_region_mask[..., 0]
    if smooth_region_mask.shape != (batch_size, num_nodes):
        raise ValueError("smooth region mask must have shape [B,N] or [B,N,1]")
    if node_mask.shape == (batch_size, num_nodes, 1):
        valid_nodes = node_mask[..., 0].to(dtype=torch.bool)
    elif node_mask.shape == (batch_size, num_nodes):
        valid_nodes = node_mask.to(dtype=torch.bool)
    else:
        raise ValueError("node mask must have shape [B,N] or [B,N,1]")
    smooth = smooth_region_mask.to(device=first.device, dtype=torch.bool) & valid_nodes
    highpasses = {
        name: graph_neighbor_highpass(
            value.permute(0, 2, 1),
            directed_edges,
            node_mask,
        )
        for name, value in branches.items()
    }
    combined_highpass = sum(highpasses.values())
    branch_names = tuple(branches)
    branch_pairs = [
        (branch_names[left], branch_names[right])
        for left in range(len(branch_names))
        for right in range(left + 1, len(branch_names))
    ]
    result: dict[str, dict[str, Any]] = {}
    for weight_name, raw_weights in weight_maps.items():
        weights = raw_weights
        if weights.shape == (batch_size, num_nodes, 1):
            weights = weights[..., 0]
        if weights.shape != (batch_size, num_nodes):
            raise ValueError(
                f"diagnostic weight map {weight_name!r} must have shape "
                "[B,N] or [B,N,1]"
            )
        weights = weights.to(device=first.device, dtype=first.dtype)
        if not bool(torch.isfinite(weights).all()) or bool((weights < 0.0).any()):
            raise ValueError(
                f"diagnostic weight map {weight_name!r} must be finite and nonnegative"
            )
        selected_weights = weights * smooth.to(dtype=weights.dtype)
        selected_mass = selected_weights.sum()
        smooth_node_count = int(smooth.sum().detach().cpu())
        if float(selected_mass.detach().cpu()) <= 0.0:
            result[weight_name] = {
                "status": "unavailable_empty_or_zero_weight_smooth_region",
                "smooth_node_count": smooth_node_count,
            }
            continue
        denominator = selected_mass * channels

        def inner(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
            return (selected_weights.unsqueeze(-1) * left * right).sum() / denominator

        branch_energy_t = {
            name: inner(value, value) for name, value in highpasses.items()
        }
        combined_energy_t = inner(combined_highpass, combined_highpass)
        cross_t = {
            f"{left}_{right}": 2.0 * inner(highpasses[left], highpasses[right])
            for left, right in branch_pairs
        }
        sum_branch_energy_t = sum(branch_energy_t.values())
        reconstructed_energy_t = sum_branch_energy_t + sum(cross_t.values())
        identity_residual_t = combined_energy_t - reconstructed_energy_t
        scale_t = torch.maximum(
            torch.maximum(combined_energy_t.abs(), reconstructed_energy_t.abs()),
            sum_branch_energy_t.abs(),
        ).clamp_min(1e-30)
        sum_branch_energy = float(sum_branch_energy_t.detach().cpu())
        if sum_branch_energy > 0.0:
            cancellation_fraction = float(
                (1.0 - combined_energy_t / sum_branch_energy_t).detach().cpu()
            )
            pair_shares = {
                name: float((-value / sum_branch_energy_t).detach().cpu())
                for name, value in cross_t.items()
            }
        else:
            cancellation_fraction = None
            pair_shares = {name: None for name in cross_t}
        result[weight_name] = {
            "status": "available",
            "smooth_node_count": smooth_node_count,
            "branch_energy": {
                name: float(value.detach().cpu())
                for name, value in branch_energy_t.items()
            },
            "sum_branch_energy": sum_branch_energy,
            "combined_energy": float(combined_energy_t.detach().cpu()),
            "twice_pair_inner_product": {
                name: float(value.detach().cpu()) for name, value in cross_t.items()
            },
            "reconstructed_combined_energy": float(
                reconstructed_energy_t.detach().cpu()
            ),
            "identity_residual": float(identity_residual_t.detach().cpu()),
            "relative_identity_residual": float(
                (identity_residual_t.abs() / scale_t).detach().cpu()
            ),
            "cancellation_fraction": cancellation_fraction,
            "pair_cancellation_share": pair_shares,
            "sum_pair_cancellation_share": (
                None
                if cancellation_fraction is None
                else float(
                    sum(value for value in pair_shares.values() if value is not None)
                )
            ),
            "highpass_operator": "self_plus_directed_neighbor_average",
        }
    return result


def _torch_neighbor_band_proxy(
    value: torch.Tensor,
    *,
    directed_edges: torch.Tensor,
    node_mask: torch.Tensor,
    region_masks: Mapping[str, torch.Tensor],
    weight_maps: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    """Summarize a scalable, nonorthogonal three-band graph decomposition.

    The bands use two applications of the D013 self-plus-neighbor average:
    high=x-Sx, mid=Sx-S^2x, and low=S^2x. They reconstruct the input exactly
    but are not graph-Laplacian eigenspaces, so their energy shares are
    explicitly diagnostic proxies rather than an orthogonal spectrum.
    """

    if value.ndim != 3:
        raise ValueError("band-proxy tensors must have shape [B,C,N]")
    batch_size, channels, num_nodes = value.shape
    if not region_masks:
        raise ValueError("at least one band-proxy region mask is required")
    if not weight_maps:
        raise ValueError("at least one band-proxy weight map is required")

    value_bnc = value.permute(0, 2, 1)
    high = graph_neighbor_highpass(value_bnc, directed_edges, node_mask)
    first_average = value_bnc - high
    mid = graph_neighbor_highpass(first_average, directed_edges, node_mask)
    low = first_average - mid
    reconstructed = low + mid + high

    if node_mask.shape == (batch_size, num_nodes, 1):
        valid_nodes = node_mask[..., 0].to(dtype=torch.bool)
    elif node_mask.shape == (batch_size, num_nodes):
        valid_nodes = node_mask.to(dtype=torch.bool)
    else:
        raise ValueError("node mask must have shape [B,N] or [B,N,1]")

    result: dict[str, Any] = {
        "operator": "two_level_self_plus_directed_neighbor_average",
        "orthogonal": False,
        "bands": {"low": "S2x", "mid": "Sx-S2x", "high": "x-Sx"},
        "regions": {},
    }
    band_values = {"low": low, "mid": mid, "high": high}
    for region_name, raw_region in region_masks.items():
        region = raw_region
        if region.shape == (batch_size, num_nodes, 1):
            region = region[..., 0]
        if region.shape != (batch_size, num_nodes):
            raise ValueError(
                f"band-proxy region {region_name!r} must have shape [B,N] or [B,N,1]"
            )
        selected = region.to(device=value.device, dtype=torch.bool) & valid_nodes
        region_result: dict[str, Any] = {}
        for weight_name, raw_weights in weight_maps.items():
            weights = raw_weights
            if weights.shape == (batch_size, num_nodes, 1):
                weights = weights[..., 0]
            if weights.shape != (batch_size, num_nodes):
                raise ValueError(
                    f"band-proxy weight map {weight_name!r} must have shape "
                    "[B,N] or [B,N,1]"
                )
            weights = weights.to(device=value.device, dtype=value.dtype)
            if not bool(torch.isfinite(weights).all()) or bool((weights < 0.0).any()):
                raise ValueError(
                    f"band-proxy weight map {weight_name!r} must be finite "
                    "and nonnegative"
                )
            selected_weights = weights * selected.to(dtype=weights.dtype)
            selected_mass = selected_weights.sum()
            node_count = int(selected.sum().detach().cpu())
            if float(selected_mass.detach().cpu()) <= 0.0:
                region_result[weight_name] = {
                    "status": "unavailable_empty_or_zero_weight_region",
                    "node_count": node_count,
                }
                continue
            denominator = (selected_mass * channels).clamp_min(1e-30)

            def energy(field: torch.Tensor) -> torch.Tensor:
                return (
                    selected_weights.unsqueeze(-1) * field.square()
                ).sum() / denominator

            original_energy = energy(value_bnc)
            reconstructed_energy = energy(reconstructed)
            band_energy = {name: energy(field) for name, field in band_values.items()}
            sum_band_energy = sum(band_energy.values())
            shares = {
                name: float((item / sum_band_energy.clamp_min(1e-30)).detach().cpu())
                for name, item in band_energy.items()
            }
            scale = torch.maximum(
                original_energy.abs(), reconstructed_energy.abs()
            ).clamp_min(1e-30)
            region_result[weight_name] = {
                "status": "available",
                "node_count": node_count,
                "original_rms": float(torch.sqrt(original_energy).detach().cpu()),
                "band_rms": {
                    name: float(torch.sqrt(item).detach().cpu())
                    for name, item in band_energy.items()
                },
                "normalized_band_energy_share": shares,
                "high_to_original_energy_ratio": float(
                    (band_energy["high"] / original_energy.clamp_min(1e-30))
                    .detach()
                    .cpu()
                ),
                "relative_reconstruction_energy_residual": float(
                    ((reconstructed_energy - original_energy).abs() / scale)
                    .detach()
                    .cpu()
                ),
            }
        result["regions"][region_name] = region_result
    return result


@torch.no_grad()
def trace_pcno_branches(
    backbone: torch.nn.Module,
    model_input: torch.Tensor,
    aux: Sequence[torch.Tensor],
    *,
    disabled_branch: tuple[int, str] | None = None,
    branch_gains: Mapping[tuple[int, str], float] | None = None,
    paired_model_input: torch.Tensor | None = None,
    return_paired_output: bool = False,
    collect_summaries: bool = True,
    reference_smooth_region_mask: torch.Tensor | None = None,
    diagnostic_weight_maps: Mapping[str, torch.Tensor] | None = None,
) -> (
    tuple[torch.Tensor, list[dict[str, Any]]]
    | tuple[torch.Tensor, list[dict[str, Any]], torch.Tensor]
):
    """Replay PCNO exactly while exposing its three additive branches.

    A disabled branch or bounded branch gain is a frozen diagnostic
    counterfactual, never a trained ablation. Supported names are spectral,
    pointwise, and differential. A paired input records finite directional
    responses at every branch and hidden update. ``return_paired_output`` adds
    the exactly decoded paired output as a third return without changing the
    default two-return API.
    """

    if len(aux) != 5:
        raise ValueError("PCNO aux must contain five tensors")
    _, nodes, node_weights, directed_edges, edge_gradient_weights = aux
    valid_names = {"spectral", "pointwise", "differential"}
    if disabled_branch is not None and branch_gains:
        raise ValueError("branch disabling and branch gains are separate diagnostics")
    if disabled_branch is not None:
        layer_index, branch_name = disabled_branch
        if branch_name not in valid_names:
            raise ValueError(f"unsupported disabled branch {branch_name!r}")
        if layer_index < 0 or layer_index >= len(backbone.ws):
            raise ValueError("disabled branch layer lies outside the PCNO")
    gains: dict[tuple[int, str], float] = {}
    for key, value in (branch_gains or {}).items():
        if not isinstance(key, tuple) or len(key) != 2:
            raise ValueError("branch gain keys must be (layer, branch) tuples")
        layer_index, branch_name = key
        if not isinstance(layer_index, int) or not 0 <= layer_index < len(backbone.ws):
            raise ValueError("branch gain layer lies outside the PCNO")
        if branch_name not in valid_names:
            raise ValueError(f"unsupported branch gain name {branch_name!r}")
        gain = float(value)
        if not math.isfinite(gain) or not 0.0 <= gain <= 1.0:
            raise ValueError("diagnostic branch gains must lie in [0,1]")
        gains[(layer_index, branch_name)] = gain
    if paired_model_input is not None:
        if paired_model_input.shape != model_input.shape:
            raise ValueError("paired PCNO inputs must have identical shapes")
        if disabled_branch is not None:
            raise ValueError(
                "paired branch response and branch disabling are separate diagnostics"
            )
    elif return_paired_output:
        raise ValueError("return_paired_output requires paired_model_input")
    if (reference_smooth_region_mask is None) != (diagnostic_weight_maps is None):
        raise ValueError(
            "reference smooth mask and diagnostic weight maps must be supplied together"
        )
    if diagnostic_weight_maps is not None and not diagnostic_weight_maps:
        raise ValueError("at least one diagnostic weight map is required")

    bases_c, bases_s, bases_0 = compute_Fourier_bases(nodes, backbone.modes)
    wbases_c = torch.einsum("bxkw,bxw->bxkw", bases_c, node_weights)
    wbases_s = torch.einsum("bxkw,bxw->bxkw", bases_s, node_weights)
    wbases_0 = torch.einsum("bxkw,bxw->bxkw", bases_0, node_weights)
    hidden = backbone.fc0(model_input).permute(0, 2, 1)
    paired_hidden = (
        None
        if paired_model_input is None
        else backbone.fc0(paired_model_input).permute(0, 2, 1)
    )
    summaries: list[dict[str, Any]] = []
    final_layer = len(backbone.ws) - 1
    layers = zip(
        backbone.sp_convs,
        backbone.ws,
        backbone.gws,
        strict=True,
    )
    for index, (spectral_layer, pointwise_layer, differential_layer) in enumerate(
        layers
    ):
        spectral = spectral_layer(
            hidden,
            bases_c,
            bases_s,
            bases_0,
            wbases_c,
            wbases_s,
            wbases_0,
        )
        pointwise = pointwise_layer(hidden)
        differential = differential_layer(hidden, directed_edges, edge_gradient_weights)
        layer_gains = {name: gains.get((index, name), 1.0) for name in valid_names}

        def apply_gain(name: str, value: torch.Tensor) -> torch.Tensor:
            gain = layer_gains[name]
            return value if gain == 1.0 else value * gain

        spectral = apply_gain("spectral", spectral)
        pointwise = apply_gain("pointwise", pointwise)
        differential = apply_gain("differential", differential)
        original = {
            "spectral": spectral,
            "pointwise": pointwise,
            "differential": differential,
        }
        paired_original = None
        input_delta = None
        if paired_hidden is not None:
            input_delta = paired_hidden - hidden
            paired_original = {
                "spectral": apply_gain(
                    "spectral",
                    spectral_layer(
                        paired_hidden,
                        bases_c,
                        bases_s,
                        bases_0,
                        wbases_c,
                        wbases_s,
                        wbases_0,
                    ),
                ),
                "pointwise": apply_gain("pointwise", pointwise_layer(paired_hidden)),
                "differential": apply_gain(
                    "differential",
                    differential_layer(
                        paired_hidden,
                        directed_edges,
                        edge_gradient_weights,
                    ),
                ),
            }
        layer_summary = None
        if collect_summaries:
            stats = {
                name: _torch_branch_stats(value, node_weights, directed_edges)
                for name, value in original.items()
            }
            combined = spectral + pointwise + differential
            combined_stats = _torch_branch_stats(combined, node_weights, directed_edges)
            norm_sum = sum(value["weighted_rms"] for value in stats.values())
            layer_summary = {
                "layer": index,
                "branch_gains": layer_gains,
                "input_hidden": _torch_branch_stats(
                    hidden, node_weights, directed_edges
                ),
                "branches": stats,
                "combined": combined_stats,
                "combined_to_sum_branch_rms": (
                    combined_stats["weighted_rms"] / max(norm_sum, 1e-12)
                ),
                "pairwise_weighted_cosine": {
                    "spectral_pointwise": _torch_weighted_cosine(
                        spectral, pointwise, node_weights
                    ),
                    "spectral_differential": _torch_weighted_cosine(
                        spectral, differential, node_weights
                    ),
                    "pointwise_differential": _torch_weighted_cosine(
                        pointwise, differential, node_weights
                    ),
                },
                "differential_gradient_scale": float(
                    differential_layer.gw1.detach().cpu()
                ),
            }
            if (
                reference_smooth_region_mask is not None
                and diagnostic_weight_maps is not None
            ):
                layer_summary["smooth_highpass_cancellation"] = (
                    _torch_smooth_highpass_energy_decomposition(
                        original,
                        directed_edges=directed_edges,
                        node_mask=aux[0],
                        smooth_region_mask=reference_smooth_region_mask,
                        weight_maps=diagnostic_weight_maps,
                    )
                )
        if disabled_branch is not None and disabled_branch[0] == index:
            if disabled_branch[1] == "spectral":
                spectral = torch.zeros_like(spectral)
            elif disabled_branch[1] == "pointwise":
                pointwise = torch.zeros_like(pointwise)
            else:
                differential = torch.zeros_like(differential)
        update = spectral + pointwise + differential
        if backbone.act is not None and index != final_layer:
            next_hidden = hidden + backbone.act(update)
        else:
            next_hidden = update
        if collect_summaries and layer_summary is not None:
            layer_summary["post_activation_hidden"] = _torch_branch_stats(
                next_hidden, node_weights, directed_edges
            )
            if (
                reference_smooth_region_mask is not None
                and diagnostic_weight_maps is not None
            ):
                valid_nodes = aux[0].to(dtype=torch.bool)
                smooth_nodes = reference_smooth_region_mask.to(dtype=torch.bool)
                layer_summary["graph_band_proxy"] = {
                    name: _torch_neighbor_band_proxy(
                        tensor,
                        directed_edges=directed_edges,
                        node_mask=aux[0],
                        region_masks={
                            "smooth_reference": smooth_nodes,
                            "non_smooth_reference_complement": (
                                valid_nodes & ~smooth_nodes
                            ),
                        },
                        weight_maps=diagnostic_weight_maps,
                    )
                    for name, tensor in {
                        "input_hidden": hidden,
                        "spectral": spectral,
                        "pointwise": pointwise,
                        "differential": differential,
                        "combined_update": update,
                        "post_activation_hidden": next_hidden,
                    }.items()
                }
        if paired_original is not None and paired_hidden is not None:
            paired_update = sum(paired_original.values())
            if backbone.act is not None and index != final_layer:
                next_paired_hidden = paired_hidden + backbone.act(paired_update)
            else:
                next_paired_hidden = paired_update
            if collect_summaries and layer_summary is not None:
                assert input_delta is not None
                delta_branches = {
                    name: paired_original[name] - original[name] for name in original
                }
                delta_stats = {
                    name: _torch_branch_stats(
                        value,
                        node_weights,
                        directed_edges,
                    )
                    for name, value in delta_branches.items()
                }
                input_stats = _torch_branch_stats(
                    input_delta,
                    node_weights,
                    directed_edges,
                )
                combined_delta_stats = _torch_branch_stats(
                    sum(delta_branches.values()),
                    node_weights,
                    directed_edges,
                )
                next_hidden_stats = _torch_branch_stats(
                    next_paired_hidden - next_hidden,
                    node_weights,
                    directed_edges,
                )
                input_rms = max(input_stats["weighted_rms"], 1e-12)
                layer_summary["paired_response"] = {
                    "input_hidden_delta": input_stats,
                    "branches": delta_stats,
                    "branch_to_input_rms_gain": {
                        name: stats["weighted_rms"] / input_rms
                        for name, stats in delta_stats.items()
                    },
                    "combined_update_delta": combined_delta_stats,
                    "combined_update_to_input_rms_gain": (
                        combined_delta_stats["weighted_rms"] / input_rms
                    ),
                    "next_hidden_delta": next_hidden_stats,
                    "next_hidden_to_input_rms_gain": (
                        next_hidden_stats["weighted_rms"] / input_rms
                    ),
                }
                if (
                    reference_smooth_region_mask is not None
                    and diagnostic_weight_maps is not None
                ):
                    layer_summary["paired_response"]["smooth_highpass_cancellation"] = (
                        _torch_smooth_highpass_energy_decomposition(
                            delta_branches,
                            directed_edges=directed_edges,
                            node_mask=aux[0],
                            smooth_region_mask=reference_smooth_region_mask,
                            weight_maps=diagnostic_weight_maps,
                        )
                    )
            paired_hidden = next_paired_hidden
        if collect_summaries and layer_summary is not None:
            summaries.append(layer_summary)
        hidden = next_hidden

    def decode(value: torch.Tensor) -> torch.Tensor:
        output = value.permute(0, 2, 1)
        if backbone.fc_dim > 0:
            output = backbone.fc1(output)
            if backbone.act is not None:
                output = backbone.act(output)
        return backbone.fc2(output)

    output = decode(hidden)
    if return_paired_output:
        assert paired_hidden is not None
        return output, summaries, decode(paired_hidden)
    return output, summaries
