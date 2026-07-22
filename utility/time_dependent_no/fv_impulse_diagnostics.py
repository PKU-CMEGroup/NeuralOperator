"""Sparse finite-volume face-impulse identifiability diagnostics.

This module analyzes the *algebraic* map from owner-oriented face errors to
cell updates on a validated finite-volume mesh.  It does not establish that a
generated trajectory, face impulse, or flux is physical reference truth.

For the cell-by-face incidence matrix ``B`` (``+1`` at the owner and ``-1`` at
an interior neighbor), the dual-volume face weight is ``W_f = A_f d_f``.  The
two reported decoder gains are

``G_I = V^{-1/2} B W^{1/2}``

for cumulative face impulses and

``G_F(dt) = dt V^{-1/2} B A W^{-1/2}``

for face-flux densities.  Only sparse matrices and linear operators are used,
so the implementation is suitable for the 25k-cell validation mesh.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse import csgraph
from scipy.sparse.linalg import (
    ArpackNoConvergence,
    LinearOperator,
    eigsh,
    lsmr,
    splu,
)

CLAIM_BOUNDARY = (
    "solver_internal_algebraic_diagnostic_only; physical reference claims require "
    "independently validated states, geometry, orientation, boundary accounting, "
    "and cumulative accepted-substep face impulses"
)


@dataclass(frozen=True)
class FVTopologySummary:
    """Sparse incidence topology counts and exact graph-theoretic ranks."""

    num_cells: int
    num_faces: int
    num_interior_faces: int
    num_boundary_faces: int
    num_connected_components: int
    num_components_with_boundary: int
    component_sizes: tuple[int, ...]
    interior_rank: int
    interior_nullity: int
    full_rank: int
    full_nullity: int

    def to_dict(self) -> dict[str, int | list[int]]:
        """Return a JSON-compatible summary."""

        return {
            "num_cells": self.num_cells,
            "num_faces": self.num_faces,
            "num_interior_faces": self.num_interior_faces,
            "num_boundary_faces": self.num_boundary_faces,
            "num_connected_components": self.num_connected_components,
            "num_components_with_boundary": self.num_components_with_boundary,
            "component_sizes": list(self.component_sizes),
            "interior_rank": self.interior_rank,
            "interior_nullity": self.interior_nullity,
            "full_rank": self.full_rank,
            "full_nullity": self.full_nullity,
        }


@dataclass(frozen=True)
class FVImpulseOperators:
    """Validated geometry and sparse operators for face-error diagnostics."""

    cell_centers: np.ndarray
    cell_volume: np.ndarray
    face_centers: np.ndarray
    face_measure: np.ndarray
    face_owner: np.ndarray
    face_neighbor: np.ndarray
    face_boundary_tag: np.ndarray
    dual_width: np.ndarray
    face_weight: np.ndarray
    incidence: sparse.csr_matrix
    interior_incidence: sparse.csr_matrix
    interior_face_indices: np.ndarray
    boundary_face_indices: np.ndarray
    component_labels: np.ndarray
    topology: FVTopologySummary


@dataclass(frozen=True)
class TopSingularGain:
    """Largest singular value and its normalized eigen-residual."""

    value: float
    relative_residual: float

    def to_dict(self) -> dict[str, float]:
        """Return a JSON-compatible summary."""

        return {
            "value": self.value,
            "relative_residual": self.relative_residual,
        }


@dataclass(frozen=True)
class FVDecoderGainSummary:
    """Worst-case impulse and flux decoder gains for one physical timestep."""

    dt: float
    impulse_gain: TopSingularGain
    flux_gain: TopSingularGain

    def to_dict(self) -> dict[str, float | dict[str, float] | str]:
        """Return a JSON-compatible summary with the claim boundary attached."""

        return {
            "dt": self.dt,
            "impulse_gain": self.impulse_gain.to_dict(),
            "flux_gain": self.flux_gain.to_dict(),
            "claim_boundary": CLAIM_BOUNDARY,
        }


@dataclass(frozen=True)
class InteriorDivergenceBandMode:
    """One unit-``W_f^-1`` interior mode in a graph-frequency band.

    The mode is constructed in ``range(W_f B_i.T)`` and therefore contains no
    interior cycle component. ``decoded_gain`` is the physical-volume norm of
    its cell-integral divergence because ``face_winv_norm`` is normalized to
    one. The stored face array is owner-oriented and zero on boundary faces.
    """

    band: str
    seed: int
    lowpass_steps: int
    transition_steps: int
    face_impulse: np.ndarray
    face_winv_norm: float
    decoded_gain: float
    normalized_frequency: float
    normal_operator_upper_bound: float
    compatibility_relative_l2: float

    def summary(self) -> dict[str, float | int | str]:
        """Return scalar metadata without serializing the full face mode."""

        return {
            "band": self.band,
            "seed": self.seed,
            "lowpass_steps": self.lowpass_steps,
            "transition_steps": self.transition_steps,
            "face_winv_norm": self.face_winv_norm,
            "decoded_gain": self.decoded_gain,
            "normalized_frequency": self.normalized_frequency,
            "normal_operator_upper_bound": self.normal_operator_upper_bound,
            "compatibility_relative_l2": self.compatibility_relative_l2,
            "construction": (
                "matrix_free_interior_normal_operator_filter; "
                "mode_in_range(W_f B_i.T); boundary_zero"
            ),
            "claim_boundary": CLAIM_BOUNDARY,
        }


@dataclass(frozen=True)
class InteriorImpulseDecomposition:
    """Weighted divergence/cycle split of a full owner-oriented face error.

    ``divergence_active`` and ``cycle`` are nonzero only on interior faces.
    ``boundary`` contains the input on boundary faces.  Their sum reconstructs
    ``impulse_error`` without folding physical boundary exchange into the
    interior cycle-space analysis.
    """

    impulse_error: np.ndarray
    divergence_active: np.ndarray
    cycle: np.ndarray
    boundary: np.ndarray
    interior_energy: float
    divergence_active_energy: float
    cycle_energy: float
    boundary_energy: float
    interior_energy_by_component: tuple[float, ...]
    divergence_active_energy_by_component: tuple[float, ...]
    cycle_energy_by_component: tuple[float, ...]
    boundary_energy_by_component: tuple[float, ...]
    reconstruction_relative_l2: float
    cycle_divergence_l2: float
    cycle_divergence_relative_l2: float
    winv_orthogonality: float
    winv_orthogonality_relative: float
    observed_interior_decoded_gain: float
    observed_full_decoded_gain: float
    lsmr_stop_codes: tuple[int, ...]
    lsmr_iterations: tuple[int, ...]

    def summary(self) -> dict[str, float | list[int] | str]:
        """Return scalar diagnostics suitable for artifact metadata."""

        return {
            "interior_energy": self.interior_energy,
            "divergence_active_energy": self.divergence_active_energy,
            "cycle_energy": self.cycle_energy,
            "boundary_energy": self.boundary_energy,
            "interior_energy_by_component": list(self.interior_energy_by_component),
            "divergence_active_energy_by_component": list(
                self.divergence_active_energy_by_component
            ),
            "cycle_energy_by_component": list(self.cycle_energy_by_component),
            "boundary_energy_by_component": list(self.boundary_energy_by_component),
            "reconstruction_relative_l2": self.reconstruction_relative_l2,
            "cycle_divergence_l2": self.cycle_divergence_l2,
            "cycle_divergence_relative_l2": self.cycle_divergence_relative_l2,
            "winv_orthogonality": self.winv_orthogonality,
            "winv_orthogonality_relative": self.winv_orthogonality_relative,
            "observed_interior_decoded_gain": self.observed_interior_decoded_gain,
            "observed_full_decoded_gain": self.observed_full_decoded_gain,
            "lsmr_stop_codes": list(self.lsmr_stop_codes),
            "lsmr_iterations": list(self.lsmr_iterations),
            "claim_boundary": CLAIM_BOUNDARY,
        }


@dataclass(frozen=True)
class MinimumNormFaceImpulse:
    """Canonical face field for a cell integral and fixed boundary exchange.

    The interior field minimizes ``sum_f I_f^2 / W_f`` subject to matching the
    requested cell integral after the supplied physical boundary impulses are
    removed.  Compatibility is reported, never silently repaired.
    """

    target_cell_integral: np.ndarray
    decoded_cell_integral: np.ndarray
    face_impulse: np.ndarray
    compatibility_residual: np.ndarray
    compatibility_l2: float
    compatibility_relative_l2: float
    decoded_residual_l2: float
    decoded_residual_relative_l2: float
    interior_winv_energy: float
    interior_winv_energy_by_component: tuple[float, ...]
    lsmr_stop_codes: tuple[int, ...]
    lsmr_iterations: tuple[int, ...]
    solver: str = "lsmr"
    compatibility_projection_relative_l2: float | None = None
    reduced_solve_residual_relative_l2: float | None = None
    factorization_count: int = 0
    anchor_cells: tuple[int, ...] = ()

    def summary(self) -> dict[str, float | list[int] | str]:
        """Return scalar diagnostics suitable for artifact metadata."""

        return {
            "compatibility_l2": self.compatibility_l2,
            "compatibility_relative_l2": self.compatibility_relative_l2,
            "compatibility_max_absolute": float(
                np.max(np.abs(self.compatibility_residual), initial=0.0)
            ),
            "decoded_residual_l2": self.decoded_residual_l2,
            "decoded_residual_relative_l2": self.decoded_residual_relative_l2,
            "interior_winv_energy": self.interior_winv_energy,
            "interior_winv_energy_by_component": list(
                self.interior_winv_energy_by_component
            ),
            "lsmr_stop_codes": list(self.lsmr_stop_codes),
            "lsmr_iterations": list(self.lsmr_iterations),
            "solver": self.solver,
            "compatibility_projection_relative_l2": (
                self.compatibility_projection_relative_l2
            ),
            "reduced_solve_residual_relative_l2": (
                self.reduced_solve_residual_relative_l2
            ),
            "factorization_count": self.factorization_count,
            "anchor_cells": list(self.anchor_cells),
            "claim_boundary": CLAIM_BOUNDARY,
        }


@dataclass(frozen=True)
class DirectMinimumNormProjector:
    """Reusable direct projector for one fixed finite-volume mesh.

    The reduced weighted graph Laplacian is factored once per nontrivial
    connected component.  The lowest global cell index fixes only the potential
    gauge; face impulses are gauge invariant.
    """

    operators: FVImpulseOperators
    interior_weight: np.ndarray
    component_cells: tuple[np.ndarray, ...]
    free_cells: tuple[np.ndarray, ...]
    anchor_cells: tuple[int, ...]
    reduced_laplacians: tuple[sparse.csc_matrix, ...]
    factors: tuple[object | None, ...]
    factor_nnz: tuple[int, ...]

    @property
    def factorization_count(self) -> int:
        """Return the number of nontrivial component factors."""

        return sum(factor is not None for factor in self.factors)

    def summary(self) -> dict[str, int | list[int] | str]:
        """Return immutable factorization provenance."""

        return {
            "solver": "scipy_sparse_superlu",
            "factorization_dtype": "float64",
            "component_count": len(self.component_cells),
            "factorization_count": self.factorization_count,
            "anchor_cells": list(self.anchor_cells),
            "factor_nnz": list(self.factor_nnz),
            "compatibility_projection": "componentwise_arithmetic_mean",
            "regularization": "none",
            "iterative_refinement": "none",
        }

    def solve(
        self,
        target_cell_integral: np.ndarray,
        boundary_face_impulse: np.ndarray,
    ) -> MinimumNormFaceImpulse:
        """Apply the already-factored minimum-norm projector to one target."""

        (
            target,
            boundary_values,
            interior_target,
            compatibility,
            compatibility_l2,
            compatibility_relative,
        ) = _prepare_minimum_norm_inputs(
            self.operators,
            target_cell_integral,
            boundary_face_impulse,
        )
        compatible_target = interior_target.copy()
        for cells in self.component_cells:
            compatible_target[cells] -= np.mean(
                interior_target[cells], axis=0, keepdims=True
            )
        projection_correction = compatible_target - interior_target
        compatibility_projection_relative = _safe_ratio(
            float(np.linalg.norm(projection_correction)),
            float(np.linalg.norm(interior_target)),
        )

        potential = np.zeros_like(compatible_target)
        residual_squared = 0.0
        rhs_squared = 0.0
        for free, reduced, factor in zip(
            self.free_cells,
            self.reduced_laplacians,
            self.factors,
            strict=True,
        ):
            if free.size == 0:
                continue
            if factor is None:
                raise RuntimeError("nontrivial component is missing its direct factor")
            rhs = compatible_target[free]
            solution = np.asarray(factor.solve(rhs), dtype=np.float64)
            if solution.shape != rhs.shape or not np.all(np.isfinite(solution)):
                raise RuntimeError(
                    "direct minimum-norm solve produced invalid potentials"
                )
            potential[free] = solution
            residual = reduced @ solution - rhs
            residual_squared += float(np.sum(residual * residual))
            rhs_squared += float(np.sum(rhs * rhs))

        if not np.all(np.isfinite(potential)):
            raise RuntimeError(
                "direct minimum-norm solve produced nonfinite potentials"
            )
        reduced_residual_relative = _safe_ratio(
            float(np.sqrt(residual_squared)),
            float(np.sqrt(rhs_squared)),
        )
        interior_impulse = self.interior_weight[:, None] * (
            self.operators.interior_incidence.T @ potential
        )
        if not np.all(np.isfinite(interior_impulse)):
            raise RuntimeError(
                "direct minimum-norm solve produced nonfinite face values"
            )
        return _assemble_minimum_norm_result(
            operators=self.operators,
            target=target,
            boundary_values=boundary_values,
            interior_impulse=np.asarray(interior_impulse),
            compatibility=compatibility,
            compatibility_l2=compatibility_l2,
            compatibility_relative=compatibility_relative,
            lsmr_stop_codes=(),
            lsmr_iterations=(),
            solver="direct_weighted_laplacian",
            compatibility_projection_relative_l2=float(
                compatibility_projection_relative
            ),
            reduced_solve_residual_relative_l2=float(reduced_residual_relative),
            factorization_count=self.factorization_count,
            anchor_cells=self.anchor_cells,
        )


def build_fv_impulse_operators(
    *,
    cell_centers: np.ndarray,
    cell_volume: np.ndarray,
    face_centers: np.ndarray,
    face_measure: np.ndarray,
    face_owner: np.ndarray,
    face_neighbor: np.ndarray,
    face_boundary_tag: np.ndarray,
) -> FVImpulseOperators:
    """Validate geometry and build sparse incidence/dual-volume operators.

    Integer boundary tags follow the dynamic benchmark contract: tag zero is
    interior and every boundary face has a nonzero tag.
    """

    centers = np.asarray(cell_centers, dtype=np.float64)
    volume = np.asarray(cell_volume, dtype=np.float64)
    face_centers_array = np.asarray(face_centers, dtype=np.float64)
    measure = np.asarray(face_measure, dtype=np.float64)
    owner = np.asarray(face_owner)
    neighbor = np.asarray(face_neighbor)
    tags = np.asarray(face_boundary_tag)
    _validate_geometry_arrays(
        centers,
        volume,
        face_centers_array,
        measure,
        owner,
        neighbor,
        tags,
    )
    owner = owner.astype(np.int64, copy=False)
    neighbor = neighbor.astype(np.int64, copy=False)
    tags = tags.astype(np.int64, copy=False)

    num_cells = volume.size
    num_faces = owner.size
    face_index = np.arange(num_faces, dtype=np.int64)
    interior = neighbor >= 0
    interior_indices = face_index[interior]
    boundary_indices = face_index[~interior]

    rows = np.concatenate((owner, neighbor[interior]))
    columns = np.concatenate((face_index, interior_indices))
    values = np.concatenate(
        (
            np.ones(num_faces, dtype=np.float64),
            -np.ones(interior_indices.size, dtype=np.float64),
        )
    )
    incidence = sparse.coo_matrix(
        (values, (rows, columns)), shape=(num_cells, num_faces)
    ).tocsr()
    interior_incidence = incidence[:, interior_indices].tocsr()

    dual_width = np.empty(num_faces, dtype=np.float64)
    dual_width[interior] = np.linalg.norm(
        centers[owner[interior]] - centers[neighbor[interior]], axis=1
    )
    dual_width[~interior] = 2.0 * np.linalg.norm(
        centers[owner[~interior]] - face_centers_array[~interior], axis=1
    )
    if not np.all(np.isfinite(dual_width)) or np.any(dual_width <= 0.0):
        raise ValueError("all owner-neighbor/boundary dual widths must be positive")
    face_weight = measure * dual_width
    if not np.all(np.isfinite(face_weight)) or np.any(face_weight <= 0.0):
        raise ValueError("all dual-volume face weights A*d must be positive")

    adjacency_rows = np.concatenate((owner[interior], neighbor[interior]))
    adjacency_columns = np.concatenate((neighbor[interior], owner[interior]))
    adjacency = sparse.coo_matrix(
        (
            np.ones(adjacency_rows.size, dtype=np.int8),
            (adjacency_rows, adjacency_columns),
        ),
        shape=(num_cells, num_cells),
    ).tocsr()
    num_components, component_labels = csgraph.connected_components(
        adjacency, directed=False, return_labels=True
    )
    component_sizes = np.bincount(component_labels, minlength=num_components)
    boundary_components = np.unique(component_labels[owner[~interior]])
    components_with_boundary = int(boundary_components.size)
    interior_rank = int(num_cells - num_components)
    full_rank = int(interior_rank + components_with_boundary)
    topology = FVTopologySummary(
        num_cells=int(num_cells),
        num_faces=int(num_faces),
        num_interior_faces=int(interior_indices.size),
        num_boundary_faces=int(boundary_indices.size),
        num_connected_components=int(num_components),
        num_components_with_boundary=components_with_boundary,
        component_sizes=tuple(int(value) for value in component_sizes),
        interior_rank=interior_rank,
        interior_nullity=int(interior_indices.size - interior_rank),
        full_rank=full_rank,
        full_nullity=int(num_faces - full_rank),
    )
    return FVImpulseOperators(
        cell_centers=centers,
        cell_volume=volume,
        face_centers=face_centers_array,
        face_measure=measure,
        face_owner=owner,
        face_neighbor=neighbor,
        face_boundary_tag=tags,
        dual_width=dual_width,
        face_weight=face_weight,
        incidence=incidence,
        interior_incidence=interior_incidence,
        interior_face_indices=interior_indices,
        boundary_face_indices=boundary_indices,
        component_labels=component_labels,
        topology=topology,
    )


def decoder_gain_summary(
    operators: FVImpulseOperators,
    *,
    dt: float,
    tolerance: float = 1.0e-10,
    max_iterations: int | None = None,
) -> FVDecoderGainSummary:
    """Return top singular gains of the impulse and flux decoder maps."""

    dt_value = float(dt)
    if not np.isfinite(dt_value) or dt_value <= 0.0:
        raise ValueError("dt must be finite and positive")
    if tolerance <= 0.0 or not np.isfinite(tolerance):
        raise ValueError("tolerance must be finite and positive")
    if max_iterations is not None and max_iterations <= 0:
        raise ValueError("max_iterations must be positive when supplied")

    impulse_edge_scale = np.sqrt(operators.face_weight)
    flux_edge_scale = dt_value * operators.face_measure / np.sqrt(operators.face_weight)
    return FVDecoderGainSummary(
        dt=dt_value,
        impulse_gain=_top_singular_gain(
            operators.incidence,
            operators.cell_volume,
            impulse_edge_scale,
            tolerance=tolerance,
            max_iterations=max_iterations,
        ),
        flux_gain=_top_singular_gain(
            operators.incidence,
            operators.cell_volume,
            flux_edge_scale,
            tolerance=tolerance,
            max_iterations=max_iterations,
        ),
    )


def interior_divergence_band_modes(
    operators: FVImpulseOperators,
    *,
    seed: int,
    lowpass_steps: int = 64,
    transition_steps: int = 8,
) -> tuple[InteriorDivergenceBandMode, ...]:
    """Construct deterministic low/mid/high divergence-active face modes.

    Let ``G = V^-1/2 B_i W_f^1/2``. A seeded cell-space vector is split with
    powers of the stable filter ``S = I - G G.T / lambda_upper``:

    ``low = S^lowpass_steps r``
    ``mid = S^transition_steps r - low``
    ``high = r - S^transition_steps r``.

    Each cell vector is mapped through ``G.T`` and returned as an owner-
    oriented face impulse with unit ``W_f^-1`` norm. This is an algebraic
    conditioning probe; it does not identify a physical flux or a learned
    architectural source.
    """

    if lowpass_steps <= 0:
        raise ValueError("lowpass_steps must be positive")
    if transition_steps <= 0 or transition_steps >= lowpass_steps:
        raise ValueError(
            "transition_steps must be positive and smaller than lowpass_steps"
        )
    if operators.topology.num_interior_faces == 0:
        raise ValueError("divergence-band modes require at least one interior face")

    upper_bound = _interior_normal_operator_upper_bound(operators)
    if not np.isfinite(upper_bound) or upper_bound <= 0.0:
        raise RuntimeError("interior normal-operator upper bound is invalid")

    generator = np.random.default_rng(int(seed))
    base = generator.standard_normal(operators.topology.num_cells)
    base = _remove_component_nullspace(operators, base)
    base_norm = float(np.linalg.norm(base))
    if not np.isfinite(base_norm) or base_norm <= np.finfo(np.float64).tiny:
        raise RuntimeError("seeded cell probe collapsed into the component nullspace")
    base /= base_norm

    smoothed = base.copy()
    transition: np.ndarray | None = None
    for step in range(1, lowpass_steps + 1):
        smoothed -= _apply_interior_normal_operator(operators, smoothed) / upper_bound
        smoothed = _remove_component_nullspace(operators, smoothed)
        if step == transition_steps:
            transition = smoothed.copy()
    if transition is None:
        raise RuntimeError("failed to capture the requested transition filter")

    cell_bands = (
        ("low", smoothed),
        ("mid", transition - smoothed),
        ("high", base - transition),
    )
    return tuple(
        _cell_probe_to_face_mode(
            operators,
            band=band,
            seed=int(seed),
            cell_probe=cell_probe,
            lowpass_steps=lowpass_steps,
            transition_steps=transition_steps,
            upper_bound=upper_bound,
        )
        for band, cell_probe in cell_bands
    )


def minimum_winv_boundary_impulse_for_totals(
    operators: FVImpulseOperators,
    component_totals: np.ndarray,
    *,
    allowed_boundary: np.ndarray | None = None,
) -> np.ndarray:
    """Allocate totals over allowed boundary faces with minimum W^-1 norm.

    Every boundary column of the incidence matrix has cell sum one, so the
    requested component totals are exactly the compatibility condition for an
    interior lift. This construction uses only the totals and geometry. It is
    an algebraic closure, not a physical boundary flux model.
    """

    totals = np.asarray(component_totals, dtype=np.float64)
    if totals.ndim != 1 or totals.size == 0 or not np.all(np.isfinite(totals)):
        raise ValueError("component_totals must be a nonempty finite vector")
    boundary_count = operators.boundary_face_indices.size
    if allowed_boundary is None:
        allowed = np.ones((boundary_count, totals.size), dtype=bool)
    else:
        allowed = np.asarray(allowed_boundary, dtype=bool)
        if allowed.ndim == 1:
            allowed = np.repeat(allowed[:, None], totals.size, axis=1)
        if allowed.shape != (boundary_count, totals.size):
            raise ValueError(
                "allowed_boundary must have shape [boundary] or [boundary, variables]"
            )
    weight = operators.face_weight[operators.boundary_face_indices]
    result = np.zeros((boundary_count, totals.size), dtype=np.float64)
    for component, total in enumerate(totals):
        selected = allowed[:, component]
        denominator = float(np.sum(weight[selected]))
        if denominator <= 0.0:
            if total != 0.0:
                raise ValueError(f"component {component} has no allowed boundary face")
            continue
        result[selected, component] = weight[selected] * total / denominator
    return result


def decompose_interior_impulse_error(
    operators: FVImpulseOperators,
    impulse_error: np.ndarray,
    *,
    tolerance: float = 1.0e-11,
    max_iterations: int | None = None,
) -> InteriorImpulseDecomposition:
    """Project full face-impulse errors into interior div/cycle components.

    The projection is orthogonal in the ``W^{-1}`` inner product and uses

    ``e_div = W B_i.T (B_i W B_i.T)^+ B_i e``.

    LSMR acts on ``W^{1/2} B_i.T`` directly, avoiding both a dense pseudoinverse
    and an arbitrary gauge.  Boundary errors are returned separately because
    they represent boundary exchange, not an interior cycle component.
    """

    error = np.asarray(impulse_error, dtype=np.float64)
    if error.ndim == 1:
        error = error[:, None]
    if error.ndim != 2 or error.shape[0] != operators.topology.num_faces:
        raise ValueError("impulse_error must have shape [faces] or [faces, variables]")
    if error.shape[1] == 0 or not np.all(np.isfinite(error)):
        raise ValueError("impulse_error must have at least one finite variable")
    if tolerance <= 0.0 or not np.isfinite(tolerance):
        raise ValueError("tolerance must be finite and positive")
    if max_iterations is not None and max_iterations <= 0:
        raise ValueError("max_iterations must be positive when supplied")

    interior_indices = operators.interior_face_indices
    boundary_indices = operators.boundary_face_indices
    interior_error = error[interior_indices]
    interior_weight = operators.face_weight[interior_indices]
    sqrt_weight = np.sqrt(interior_weight)
    projection_matrix = sparse.diags(sqrt_weight) @ operators.interior_incidence.T
    scaled_error = interior_error / sqrt_weight[:, None]
    if max_iterations is None:
        max_iterations = min(
            max(1_000, 4 * operators.topology.num_cells),
            25_000,
        )

    divergence_active_interior = np.empty_like(interior_error)
    stop_codes: list[int] = []
    iterations: list[int] = []
    for channel in range(error.shape[1]):
        solve = lsmr(
            projection_matrix,
            scaled_error[:, channel],
            atol=tolerance,
            btol=tolerance,
            maxiter=max_iterations,
        )
        potential = solve[0]
        divergence_active_interior[:, channel] = sqrt_weight * (
            projection_matrix @ potential
        )
        stop_codes.append(int(solve[1]))
        iterations.append(int(solve[2]))
    if not np.all(np.isfinite(divergence_active_interior)):
        raise RuntimeError("LSMR produced a nonfinite divergence-active projection")

    cycle_interior = interior_error - divergence_active_interior
    divergence_active = np.zeros_like(error)
    cycle = np.zeros_like(error)
    boundary = np.zeros_like(error)
    divergence_active[interior_indices] = divergence_active_interior
    cycle[interior_indices] = cycle_interior
    boundary[boundary_indices] = error[boundary_indices]

    reconstructed = divergence_active + cycle + boundary
    reconstruction_error = np.linalg.norm(error - reconstructed)
    reconstruction_relative = _safe_ratio(reconstruction_error, np.linalg.norm(error))

    cycle_divergence = operators.interior_incidence @ cycle_interior
    input_divergence = operators.interior_incidence @ interior_error
    cycle_divergence_l2 = float(np.linalg.norm(cycle_divergence))
    cycle_divergence_relative = _safe_ratio(
        cycle_divergence_l2, float(np.linalg.norm(input_divergence))
    )

    inv_weight = 1.0 / interior_weight[:, None]
    interior_energy_by_component = np.sum(
        interior_error * interior_error * inv_weight, axis=0
    )
    divergence_energy_by_component = np.sum(
        divergence_active_interior * divergence_active_interior * inv_weight,
        axis=0,
    )
    cycle_energy_by_component = np.sum(
        cycle_interior * cycle_interior * inv_weight, axis=0
    )
    boundary_weight = operators.face_weight[boundary_indices, None]
    boundary_energy_by_component = np.sum(
        error[boundary_indices] ** 2 / boundary_weight, axis=0
    )
    interior_energy = float(np.sum(interior_energy_by_component))
    divergence_energy = float(np.sum(divergence_energy_by_component))
    cycle_energy = float(np.sum(cycle_energy_by_component))
    boundary_energy = float(np.sum(boundary_energy_by_component))
    orthogonality = float(
        np.sum(divergence_active_interior * cycle_interior * inv_weight)
    )
    # Normalizing by sqrt(E_div * E_cycle) is ill-conditioned when either
    # exact component is zero and turns roundoff into an O(1) diagnostic.
    orthogonality_relative = _safe_ratio(abs(orthogonality), interior_energy)

    observed_interior_gain = _observed_gain(
        operators.interior_incidence,
        operators.cell_volume,
        interior_weight,
        interior_error,
    )
    observed_full_gain = _observed_gain(
        operators.incidence,
        operators.cell_volume,
        operators.face_weight,
        error,
    )

    return InteriorImpulseDecomposition(
        impulse_error=error,
        divergence_active=divergence_active,
        cycle=cycle,
        boundary=boundary,
        interior_energy=interior_energy,
        divergence_active_energy=divergence_energy,
        cycle_energy=cycle_energy,
        boundary_energy=boundary_energy,
        interior_energy_by_component=tuple(
            float(value) for value in interior_energy_by_component
        ),
        divergence_active_energy_by_component=tuple(
            float(value) for value in divergence_energy_by_component
        ),
        cycle_energy_by_component=tuple(
            float(value) for value in cycle_energy_by_component
        ),
        boundary_energy_by_component=tuple(
            float(value) for value in boundary_energy_by_component
        ),
        reconstruction_relative_l2=float(reconstruction_relative),
        cycle_divergence_l2=cycle_divergence_l2,
        cycle_divergence_relative_l2=float(cycle_divergence_relative),
        winv_orthogonality=orthogonality,
        winv_orthogonality_relative=float(orthogonality_relative),
        observed_interior_decoded_gain=float(observed_interior_gain),
        observed_full_decoded_gain=float(observed_full_gain),
        lsmr_stop_codes=tuple(stop_codes),
        lsmr_iterations=tuple(iterations),
    )


def _prepare_minimum_norm_inputs(
    operators: FVImpulseOperators,
    target_cell_integral: np.ndarray,
    boundary_face_impulse: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Validate and split a fixed-boundary minimum-norm problem."""

    target = np.asarray(target_cell_integral, dtype=np.float64)
    if target.ndim == 1:
        target = target[:, None]
    if target.ndim != 2 or target.shape[0] != operators.topology.num_cells:
        raise ValueError(
            "target_cell_integral must have shape [cells] or [cells, variables]"
        )
    if target.shape[1] == 0 or not np.all(np.isfinite(target)):
        raise ValueError("target_cell_integral must contain finite variables")

    boundary_values = np.asarray(boundary_face_impulse, dtype=np.float64)
    if boundary_values.ndim == 1:
        boundary_values = boundary_values[:, None]
    expected_boundary_shape = (
        operators.boundary_face_indices.size,
        target.shape[1],
    )
    if boundary_values.shape != expected_boundary_shape:
        raise ValueError(
            "boundary_face_impulse must have shape "
            f"{expected_boundary_shape}, received {boundary_values.shape}"
        )
    if not np.all(np.isfinite(boundary_values)):
        raise ValueError("boundary_face_impulse must be finite")

    boundary_incidence = operators.incidence[:, operators.boundary_face_indices]
    interior_target = target - boundary_incidence @ boundary_values
    component_residuals = []
    for component in range(operators.topology.num_connected_components):
        selected = operators.component_labels == component
        component_residuals.append(np.sum(interior_target[selected], axis=0))
    compatibility = np.stack(component_residuals, axis=0)
    compatibility_l2 = float(np.linalg.norm(compatibility))
    compatibility_relative = _safe_ratio(
        compatibility_l2,
        float(np.linalg.norm(interior_target)),
    )
    return (
        target,
        boundary_values,
        np.asarray(interior_target),
        compatibility,
        compatibility_l2,
        float(compatibility_relative),
    )


def _assemble_minimum_norm_result(
    *,
    operators: FVImpulseOperators,
    target: np.ndarray,
    boundary_values: np.ndarray,
    interior_impulse: np.ndarray,
    compatibility: np.ndarray,
    compatibility_l2: float,
    compatibility_relative: float,
    lsmr_stop_codes: tuple[int, ...],
    lsmr_iterations: tuple[int, ...],
    solver: str,
    compatibility_projection_relative_l2: float | None = None,
    reduced_solve_residual_relative_l2: float | None = None,
    factorization_count: int = 0,
    anchor_cells: tuple[int, ...] = (),
) -> MinimumNormFaceImpulse:
    """Assemble common diagnostics after either minimum-norm solve."""

    face_impulse = np.zeros(
        (operators.topology.num_faces, target.shape[1]), dtype=np.float64
    )
    face_impulse[operators.interior_face_indices] = interior_impulse
    face_impulse[operators.boundary_face_indices] = boundary_values
    decoded = operators.incidence @ face_impulse
    decoded_residual = decoded - target
    decoded_residual_l2 = float(np.linalg.norm(decoded_residual))
    decoded_residual_relative = _safe_ratio(
        decoded_residual_l2,
        float(np.linalg.norm(target)),
    )
    interior_weight = operators.face_weight[operators.interior_face_indices]
    energy_by_component = np.sum(
        interior_impulse * interior_impulse / interior_weight[:, None], axis=0
    )
    return MinimumNormFaceImpulse(
        target_cell_integral=target,
        decoded_cell_integral=np.asarray(decoded),
        face_impulse=face_impulse,
        compatibility_residual=compatibility,
        compatibility_l2=compatibility_l2,
        compatibility_relative_l2=float(compatibility_relative),
        decoded_residual_l2=decoded_residual_l2,
        decoded_residual_relative_l2=float(decoded_residual_relative),
        interior_winv_energy=float(np.sum(energy_by_component)),
        interior_winv_energy_by_component=tuple(
            float(value) for value in energy_by_component
        ),
        lsmr_stop_codes=lsmr_stop_codes,
        lsmr_iterations=lsmr_iterations,
        solver=solver,
        compatibility_projection_relative_l2=(compatibility_projection_relative_l2),
        reduced_solve_residual_relative_l2=reduced_solve_residual_relative_l2,
        factorization_count=factorization_count,
        anchor_cells=anchor_cells,
    )


def factorize_direct_minimum_winv_norm_projector(
    operators: FVImpulseOperators,
) -> DirectMinimumNormProjector:
    """Factor the anchored weighted graph Laplacian for a fixed mesh once."""

    interior_weight = np.asarray(
        operators.face_weight[operators.interior_face_indices], dtype=np.float64
    )
    weighted_incidence = operators.interior_incidence @ sparse.diags(interior_weight)
    laplacian = (weighted_incidence @ operators.interior_incidence.T).tocsc()
    asymmetry = laplacian - laplacian.T
    if asymmetry.nnz and float(np.max(np.abs(asymmetry.data))) > 1.0e-14:
        raise RuntimeError("weighted graph Laplacian is not symmetric")

    component_cells: list[np.ndarray] = []
    free_cells: list[np.ndarray] = []
    anchor_cells: list[int] = []
    reduced_laplacians: list[sparse.csc_matrix] = []
    factors: list[object | None] = []
    factor_nnz: list[int] = []
    for component in range(operators.topology.num_connected_components):
        cells = np.flatnonzero(operators.component_labels == component).astype(
            np.int64, copy=False
        )
        if cells.size == 0:
            raise RuntimeError("connected-component labels contain an empty component")
        anchor = int(cells[0])
        free = cells[1:]
        reduced = laplacian[free][:, free].tocsc()
        factor = None
        nnz = 0
        if free.size:
            factor = splu(
                reduced,
                permc_spec="MMD_AT_PLUS_A",
                diag_pivot_thresh=1.0,
            )
            if (
                not np.all(np.isfinite(factor.L.data))
                or not np.all(np.isfinite(factor.U.data))
                or np.any(factor.U.diagonal() == 0.0)
            ):
                raise RuntimeError("direct weighted-Laplacian factor is invalid")
            nnz = int(factor.L.nnz + factor.U.nnz)
        component_cells.append(cells)
        free_cells.append(free)
        anchor_cells.append(anchor)
        reduced_laplacians.append(reduced)
        factors.append(factor)
        factor_nnz.append(nnz)

    return DirectMinimumNormProjector(
        operators=operators,
        interior_weight=interior_weight,
        component_cells=tuple(component_cells),
        free_cells=tuple(free_cells),
        anchor_cells=tuple(anchor_cells),
        reduced_laplacians=tuple(reduced_laplacians),
        factors=tuple(factors),
        factor_nnz=tuple(factor_nnz),
    )


def minimum_winv_norm_face_impulse(
    operators: FVImpulseOperators,
    target_cell_integral: np.ndarray,
    boundary_face_impulse: np.ndarray,
    *,
    tolerance: float = 1.0e-11,
    max_iterations: int | None = None,
) -> MinimumNormFaceImpulse:
    """Recover the canonical interior face field for a fixed boundary field.

    ``target_cell_integral`` follows the incidence convention ``B I``.  For the
    owner-oriented decoder used in this project it therefore equals
    ``-V * (U_next - U_current)``.  ``boundary_face_impulse`` is ordered exactly
    like ``operators.boundary_face_indices``.

    With ``A = B_i W_i^(1/2)``, LSMR solves the minimum-Euclidean-norm system
    ``A y = rhs`` and the physical interior impulse is ``W_i^(1/2) y``.  This is
    the unique representative in the ``W_i^-1``-orthogonal complement of the
    interior cycle space.  An incompatible cell integral remains visible in
    the returned residual.
    """

    if tolerance <= 0.0 or not np.isfinite(tolerance):
        raise ValueError("tolerance must be finite and positive")
    if max_iterations is not None and max_iterations <= 0:
        raise ValueError("max_iterations must be positive when supplied")
    (
        target,
        boundary_values,
        interior_target,
        compatibility,
        compatibility_l2,
        compatibility_relative,
    ) = _prepare_minimum_norm_inputs(
        operators,
        target_cell_integral,
        boundary_face_impulse,
    )

    interior_weight = operators.face_weight[operators.interior_face_indices]
    sqrt_weight = np.sqrt(interior_weight)
    scaled_incidence = operators.interior_incidence @ sparse.diags(sqrt_weight)
    if max_iterations is None:
        max_iterations = min(
            max(1_000, 4 * operators.topology.num_cells),
            25_000,
        )

    interior_impulse = np.empty(
        (operators.interior_face_indices.size, target.shape[1]),
        dtype=np.float64,
    )
    stop_codes: list[int] = []
    iterations: list[int] = []
    for channel in range(target.shape[1]):
        solve = lsmr(
            scaled_incidence,
            interior_target[:, channel],
            atol=tolerance,
            btol=tolerance,
            maxiter=max_iterations,
        )
        interior_impulse[:, channel] = sqrt_weight * solve[0]
        stop_codes.append(int(solve[1]))
        iterations.append(int(solve[2]))
    if not np.all(np.isfinite(interior_impulse)):
        raise RuntimeError("LSMR produced a nonfinite minimum-norm face field")

    return _assemble_minimum_norm_result(
        operators=operators,
        target=target,
        boundary_values=boundary_values,
        interior_impulse=interior_impulse,
        compatibility=compatibility,
        compatibility_l2=compatibility_l2,
        compatibility_relative=compatibility_relative,
        lsmr_stop_codes=tuple(stop_codes),
        lsmr_iterations=tuple(iterations),
        solver="lsmr",
    )


def _top_singular_gain(
    incidence: sparse.csr_matrix,
    cell_volume: np.ndarray,
    edge_scale: np.ndarray,
    *,
    tolerance: float,
    max_iterations: int | None,
) -> TopSingularGain:
    num_cells = incidence.shape[0]
    inv_sqrt_volume = 1.0 / np.sqrt(cell_volume)
    edge_scale_squared = np.asarray(edge_scale, dtype=np.float64) ** 2

    def normal_matvec(vector: np.ndarray) -> np.ndarray:
        scaled_cell = inv_sqrt_volume * vector
        face_value = incidence.T @ scaled_cell
        return inv_sqrt_volume * (incidence @ (edge_scale_squared * face_value))

    normal = LinearOperator(
        shape=(num_cells, num_cells),
        matvec=normal_matvec,
        rmatvec=normal_matvec,
        dtype=np.float64,
    )
    if num_cells == 1:
        eigenvector = np.ones(1, dtype=np.float64)
        eigenvalue = float(normal_matvec(eigenvector)[0])
    else:
        seed = np.arange(1, num_cells + 1, dtype=np.float64)
        initial = np.sin(seed) + np.cos(np.sqrt(2.0) * seed)
        initial /= np.linalg.norm(initial)
        try:
            eigenvalues, eigenvectors = eigsh(
                normal,
                k=1,
                which="LA",
                v0=initial,
                tol=tolerance,
                maxiter=max_iterations,
            )
        except ArpackNoConvergence as exc:
            if exc.eigenvalues is None or len(exc.eigenvalues) == 0:
                raise RuntimeError(
                    "top decoder singular-value solve did not converge"
                ) from exc
            eigenvalues = exc.eigenvalues
            eigenvectors = exc.eigenvectors
        largest = int(np.argmax(eigenvalues))
        eigenvalue = float(eigenvalues[largest])
        eigenvector = np.asarray(eigenvectors[:, largest], dtype=np.float64)
    eigenvalue = max(eigenvalue, 0.0)
    residual = normal_matvec(eigenvector) - eigenvalue * eigenvector
    residual_scale = max(
        float(np.linalg.norm(normal_matvec(eigenvector))),
        abs(eigenvalue) * float(np.linalg.norm(eigenvector)),
        np.finfo(np.float64).tiny,
    )
    return TopSingularGain(
        value=float(np.sqrt(eigenvalue)),
        relative_residual=float(np.linalg.norm(residual) / residual_scale),
    )


def _interior_normal_operator_upper_bound(operators: FVImpulseOperators) -> float:
    """Return a Gershgorin upper bound for ``V^-1/2 B_i W B_i.T V^-1/2``."""

    interior = operators.interior_face_indices
    owner = operators.face_owner[interior]
    neighbor = operators.face_neighbor[interior]
    weight = operators.face_weight[interior]
    volume = operators.cell_volume
    diagonal = np.zeros(operators.topology.num_cells, dtype=np.float64)
    off_diagonal = np.zeros_like(diagonal)
    np.add.at(diagonal, owner, weight / volume[owner])
    np.add.at(diagonal, neighbor, weight / volume[neighbor])
    coupling = weight / np.sqrt(volume[owner] * volume[neighbor])
    np.add.at(off_diagonal, owner, coupling)
    np.add.at(off_diagonal, neighbor, coupling)
    return float(np.max(diagonal + off_diagonal, initial=0.0))


def _apply_interior_normal_operator(
    operators: FVImpulseOperators, vector: np.ndarray
) -> np.ndarray:
    """Apply ``V^-1/2 B_i W B_i.T V^-1/2`` without forming it."""

    value = np.asarray(vector, dtype=np.float64)
    if value.shape != (operators.topology.num_cells,) or not np.all(np.isfinite(value)):
        raise ValueError("cell probe must be a finite vector with one value per cell")
    inverse_sqrt_volume = 1.0 / np.sqrt(operators.cell_volume)
    potential = inverse_sqrt_volume * value
    interior_gradient = operators.interior_incidence.T @ potential
    return inverse_sqrt_volume * (
        operators.interior_incidence
        @ (operators.face_weight[operators.interior_face_indices] * interior_gradient)
    )


def _remove_component_nullspace(
    operators: FVImpulseOperators, vector: np.ndarray
) -> np.ndarray:
    """Remove the constant-potential null mode on every interior component."""

    value = np.asarray(vector, dtype=np.float64).copy()
    if value.shape != (operators.topology.num_cells,) or not np.all(np.isfinite(value)):
        raise ValueError("cell probe must be a finite vector with one value per cell")
    sqrt_volume = np.sqrt(operators.cell_volume)
    for component in range(operators.topology.num_connected_components):
        cells = np.flatnonzero(operators.component_labels == component)
        basis = sqrt_volume[cells]
        denominator = float(np.dot(basis, basis))
        if denominator <= 0.0:
            raise RuntimeError("component nullspace basis has zero norm")
        value[cells] -= basis * float(np.dot(value[cells], basis) / denominator)
    return value


def _cell_probe_to_face_mode(
    operators: FVImpulseOperators,
    *,
    band: str,
    seed: int,
    cell_probe: np.ndarray,
    lowpass_steps: int,
    transition_steps: int,
    upper_bound: float,
) -> InteriorDivergenceBandMode:
    """Map one filtered cell probe through ``W B_i.T V^-1/2`` and normalize."""

    probe = _remove_component_nullspace(operators, cell_probe)
    inverse_sqrt_volume = 1.0 / np.sqrt(operators.cell_volume)
    interior = operators.interior_face_indices
    interior_weight = operators.face_weight[interior]
    interior_impulse = interior_weight * (
        operators.interior_incidence.T @ (inverse_sqrt_volume * probe)
    )
    face_norm = float(np.sqrt(np.sum(interior_impulse**2 / interior_weight)))
    if not np.isfinite(face_norm) or face_norm <= 100.0 * np.finfo(np.float64).tiny:
        raise RuntimeError(f"{band} band collapsed under graph filtering")
    interior_impulse /= face_norm

    face_impulse = np.zeros(operators.topology.num_faces, dtype=np.float64)
    face_impulse[interior] = interior_impulse
    decoded = operators.interior_incidence @ interior_impulse
    decoded_norm = float(np.sqrt(np.sum(decoded**2 / operators.cell_volume)))
    normalized_frequency = decoded_norm**2 / upper_bound
    component_sums = np.bincount(
        operators.component_labels,
        weights=decoded,
        minlength=operators.topology.num_connected_components,
    )
    compatibility_relative = _safe_ratio(
        float(np.linalg.norm(component_sums)), float(np.linalg.norm(decoded))
    )
    return InteriorDivergenceBandMode(
        band=band,
        seed=seed,
        lowpass_steps=lowpass_steps,
        transition_steps=transition_steps,
        face_impulse=face_impulse,
        face_winv_norm=float(np.sqrt(np.sum(face_impulse**2 / operators.face_weight))),
        decoded_gain=decoded_norm,
        normalized_frequency=float(normalized_frequency),
        normal_operator_upper_bound=float(upper_bound),
        compatibility_relative_l2=float(compatibility_relative),
    )


def _observed_gain(
    incidence: sparse.csr_matrix,
    cell_volume: np.ndarray,
    face_weight: np.ndarray,
    face_error: np.ndarray,
) -> float:
    input_norm = float(np.sqrt(np.sum(face_error**2 / face_weight[:, None])))
    cell_integral_error = incidence @ face_error
    decoded_norm = float(np.sqrt(np.sum(cell_integral_error**2 / cell_volume[:, None])))
    return _safe_ratio(decoded_norm, input_norm)


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator > 0.0:
        return float(numerator / denominator)
    return 0.0 if numerator == 0.0 else float("inf")


def _validate_geometry_arrays(
    cell_centers: np.ndarray,
    cell_volume: np.ndarray,
    face_centers: np.ndarray,
    face_measure: np.ndarray,
    face_owner: np.ndarray,
    face_neighbor: np.ndarray,
    face_boundary_tag: np.ndarray,
) -> None:
    if cell_centers.ndim != 2 or cell_centers.shape[0] == 0:
        raise ValueError("cell_centers must have shape [cells, dimensions]")
    if cell_centers.shape[1] == 0 or not np.all(np.isfinite(cell_centers)):
        raise ValueError("cell_centers must be finite with at least one dimension")
    num_cells, dimensions = cell_centers.shape
    if cell_volume.shape != (num_cells,):
        raise ValueError("cell_volume must have shape [cells]")
    if not np.all(np.isfinite(cell_volume)) or np.any(cell_volume <= 0.0):
        raise ValueError("cell_volume must be finite and positive")
    if face_centers.ndim != 2 or face_centers.shape[1] != dimensions:
        raise ValueError("face_centers must have shape [faces, dimensions]")
    num_faces = face_centers.shape[0]
    if num_faces == 0 or not np.all(np.isfinite(face_centers)):
        raise ValueError("face_centers must contain at least one finite face")
    for name, values in (
        ("face_measure", face_measure),
        ("face_owner", face_owner),
        ("face_neighbor", face_neighbor),
        ("face_boundary_tag", face_boundary_tag),
    ):
        if values.shape != (num_faces,):
            raise ValueError(f"{name} must have shape [faces]")
    if not np.all(np.isfinite(face_measure)) or np.any(face_measure <= 0.0):
        raise ValueError("face_measure must be finite and positive")
    for name, indices in (("face_owner", face_owner), ("face_neighbor", face_neighbor)):
        if not np.issubdtype(indices.dtype, np.integer):
            raise ValueError(f"{name} must have an integer dtype")
    if np.any(face_owner < 0) or np.any(face_owner >= num_cells):
        raise ValueError("face_owner contains an out-of-range cell index")
    if np.any(face_neighbor < -1) or np.any(face_neighbor >= num_cells):
        raise ValueError("face_neighbor contains an out-of-range cell index")
    interior = face_neighbor >= 0
    if np.any(face_owner[interior] == face_neighbor[interior]):
        raise ValueError("an interior face cannot have the same owner and neighbor")
    if not np.issubdtype(face_boundary_tag.dtype, np.integer):
        raise ValueError("face_boundary_tag must have an integer dtype")
    if np.any(face_boundary_tag[interior] != 0):
        raise ValueError("interior faces must have boundary tag zero")
    if np.any(face_boundary_tag[~interior] == 0):
        raise ValueError("boundary faces must have a nonzero boundary tag")
