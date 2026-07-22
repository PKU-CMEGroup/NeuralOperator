from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from scripts.time_dependent_no.diagnose_fv_divergence_active_target import (
    DIRECT_GATE_ROW_METRICS,
    DIRECT_GATE_THRESHOLDS,
    GATE_ROW_METRICS,
    GATE_THRESHOLDS,
    _aggregate_gate_metrics,
    _row_metrics,
)
from scripts.time_dependent_no.diagnose_fv_divergence_conditioning import (
    _aggregate as _aggregate_d049,
    _parse_resolution,
)
from scripts.time_dependent_no.diagnose_pcno_residual_face_lift import (
    _aggregate as _aggregate_d050,
)
from utility.time_dependent_no.fv_impulse_diagnostics import (
    CLAIM_BOUNDARY,
    build_fv_impulse_operators,
    decoder_gain_summary,
    decompose_interior_impulse_error,
    factorize_direct_minimum_winv_norm_projector,
    interior_divergence_band_modes,
    minimum_winv_boundary_impulse_for_totals,
    minimum_winv_norm_face_impulse,
)


def _rectangular_geometry(nx: int, ny: int) -> dict[str, np.ndarray]:
    dx = 1.5
    dy = 0.75
    x = (np.arange(nx, dtype=np.float64) + 0.5) * dx
    y = (np.arange(ny, dtype=np.float64) + 0.5) * dy
    xx, yy = np.meshgrid(x, y, indexing="xy")
    cell_centers = np.stack((xx.reshape(-1), yy.reshape(-1)), axis=1)
    cell_volume = np.full(nx * ny, dx * dy, dtype=np.float64)

    centers: list[tuple[float, float]] = []
    measures: list[float] = []
    owners: list[int] = []
    neighbors: list[int] = []
    tags: list[int] = []
    for j in range(ny):
        for i in range(nx + 1):
            centers.append((i * dx, (j + 0.5) * dy))
            measures.append(dy)
            if i == 0:
                owners.append(j * nx)
                neighbors.append(-1)
                tags.append(1)
            elif i == nx:
                owners.append(j * nx + nx - 1)
                neighbors.append(-1)
                tags.append(2)
            else:
                owners.append(j * nx + i - 1)
                neighbors.append(j * nx + i)
                tags.append(0)
    for j in range(ny + 1):
        for i in range(nx):
            centers.append(((i + 0.5) * dx, j * dy))
            measures.append(dx)
            if j == 0:
                owners.append(i)
                neighbors.append(-1)
                tags.append(3)
            elif j == ny:
                owners.append((ny - 1) * nx + i)
                neighbors.append(-1)
                tags.append(4)
            else:
                owners.append((j - 1) * nx + i)
                neighbors.append(j * nx + i)
                tags.append(0)
    return {
        "cell_centers": cell_centers,
        "cell_volume": cell_volume,
        "face_centers": np.asarray(centers, dtype=np.float64),
        "face_measure": np.asarray(measures, dtype=np.float64),
        "face_owner": np.asarray(owners, dtype=np.int64),
        "face_neighbor": np.asarray(neighbors, dtype=np.int64),
        "face_boundary_tag": np.asarray(tags, dtype=np.int8),
    }


def _operators(nx: int, ny: int):
    return build_fv_impulse_operators(**_rectangular_geometry(nx, ny))


def _interior_face(operators, owner: int, neighbor: int) -> int:
    selected = np.flatnonzero(
        (operators.face_owner == owner) & (operators.face_neighbor == neighbor)
    )
    assert selected.size == 1
    return int(selected[0])


@pytest.mark.parametrize(
    ("nx", "ny", "faces", "interior_faces", "interior_nullity"),
    [(2, 2, 12, 4, 1), (3, 2, 17, 7, 2)],
)
def test_rectangular_topology_counts_and_exact_ranks(
    nx: int,
    ny: int,
    faces: int,
    interior_faces: int,
    interior_nullity: int,
) -> None:
    operators = _operators(nx, ny)
    topology = operators.topology

    assert topology.num_cells == nx * ny
    assert topology.num_faces == faces
    assert topology.num_interior_faces == interior_faces
    assert topology.num_boundary_faces == faces - interior_faces
    assert topology.num_connected_components == 1
    assert topology.num_components_with_boundary == 1
    assert topology.component_sizes == (nx * ny,)
    assert topology.interior_rank == nx * ny - 1
    assert topology.interior_nullity == interior_nullity
    assert topology.full_rank == nx * ny
    assert topology.full_nullity == faces - nx * ny


def test_exact_cell_loop_is_entirely_cycle_space() -> None:
    operators = _operators(2, 2)
    error = np.zeros((operators.topology.num_faces, 2), dtype=np.float64)
    # Counter-clockwise circulation around the four cells.  The incidence
    # divergence is exactly zero under the stored owner orientations.
    error[_interior_face(operators, 0, 1), 0] = 1.0
    error[_interior_face(operators, 2, 3), 0] = -1.0
    error[_interior_face(operators, 0, 2), 0] = -1.0
    error[_interior_face(operators, 1, 3), 0] = 1.0
    error[:, 1] = -2.5 * error[:, 0]

    result = decompose_interior_impulse_error(operators, error)

    np.testing.assert_allclose(result.divergence_active, 0.0, atol=2.0e-13)
    np.testing.assert_allclose(result.cycle, error, atol=2.0e-13)
    np.testing.assert_allclose(result.boundary, 0.0, atol=0.0)
    assert result.reconstruction_relative_l2 == 0.0
    assert result.cycle_divergence_l2 <= 2.0e-13
    assert result.winv_orthogonality_relative <= 2.0e-13
    assert result.observed_interior_decoded_gain <= 2.0e-13
    assert result.cycle_energy == pytest.approx(result.interior_energy)
    assert sum(result.interior_energy_by_component) == pytest.approx(
        result.interior_energy
    )
    assert sum(result.divergence_active_energy_by_component) == pytest.approx(
        result.divergence_active_energy
    )
    assert sum(result.cycle_energy_by_component) == pytest.approx(result.cycle_energy)
    assert sum(result.boundary_energy_by_component) == pytest.approx(
        result.boundary_energy
    )


def test_weighted_gradient_is_entirely_divergence_active() -> None:
    operators = _operators(3, 2)
    potential = np.asarray([0.2, -0.7, 1.4, 0.9, -1.3, 0.1])
    interior = operators.interior_face_indices
    weighted_gradient = operators.face_weight[interior] * (
        operators.interior_incidence.T @ potential
    )
    error = np.zeros((operators.topology.num_faces, 1), dtype=np.float64)
    error[interior, 0] = weighted_gradient

    result = decompose_interior_impulse_error(operators, error)

    np.testing.assert_allclose(
        result.divergence_active, error, rtol=2.0e-10, atol=2.0e-11
    )
    np.testing.assert_allclose(result.cycle, 0.0, rtol=0.0, atol=2.0e-11)
    assert result.reconstruction_relative_l2 == 0.0
    assert result.cycle_divergence_relative_l2 <= 2.0e-10
    assert result.winv_orthogonality_relative <= 2.0e-10
    assert result.divergence_active_energy == pytest.approx(
        result.interior_energy, rel=2.0e-10
    )
    assert len(result.divergence_active_energy_by_component) == 1


def test_sparse_decoder_gains_match_dense_svd() -> None:
    operators = _operators(3, 2)
    dt = 0.037
    result = decoder_gain_summary(operators, dt=dt)
    incidence = operators.incidence.toarray()
    inverse_sqrt_volume = np.diag(operators.cell_volume**-0.5)
    impulse_decoder = (
        inverse_sqrt_volume @ incidence @ np.diag(np.sqrt(operators.face_weight))
    )
    flux_decoder = (
        dt
        * inverse_sqrt_volume
        @ incidence
        @ np.diag(operators.face_measure / np.sqrt(operators.face_weight))
    )

    expected_impulse = np.linalg.svd(impulse_decoder, compute_uv=False)[0]
    expected_flux = np.linalg.svd(flux_decoder, compute_uv=False)[0]
    assert result.impulse_gain.value == pytest.approx(expected_impulse, rel=2.0e-12)
    assert result.flux_gain.value == pytest.approx(expected_flux, rel=2.0e-12)
    assert result.impulse_gain.relative_residual <= 1.0e-11
    assert result.flux_gain.relative_residual <= 1.0e-11
    assert result.to_dict()["claim_boundary"] == CLAIM_BOUNDARY


def test_divergence_band_modes_are_normalized_active_and_frequency_ordered() -> None:
    operators = _operators(24, 12)

    modes = interior_divergence_band_modes(
        operators,
        seed=20260722,
        lowpass_steps=48,
        transition_steps=6,
    )

    assert tuple(mode.band for mode in modes) == ("low", "mid", "high")
    assert all(
        mode.face_impulse.shape == (operators.topology.num_faces,) for mode in modes
    )
    assert all(mode.face_winv_norm == pytest.approx(1.0, abs=2.0e-13) for mode in modes)
    assert all(
        np.max(np.abs(mode.face_impulse[operators.boundary_face_indices])) == 0.0
        for mode in modes
    )
    assert all(mode.compatibility_relative_l2 <= 2.0e-13 for mode in modes)
    assert all(0.0 < mode.normalized_frequency <= 1.0 + 2.0e-13 for mode in modes)
    assert modes[0].decoded_gain < modes[1].decoded_gain < modes[2].decoded_gain

    for mode in modes:
        decomposition = decompose_interior_impulse_error(
            operators,
            mode.face_impulse,
            tolerance=1.0e-13,
        )
        np.testing.assert_allclose(
            decomposition.divergence_active[:, 0],
            mode.face_impulse,
            rtol=2.0e-10,
            atol=2.0e-11,
        )
        assert decomposition.cycle_energy <= 2.0e-18
        assert mode.summary()["claim_boundary"] == CLAIM_BOUNDARY


@pytest.mark.parametrize(
    ("lowpass_steps", "transition_steps", "message"),
    [(0, 1, "lowpass_steps"), (8, 0, "transition_steps"), (8, 8, "transition_steps")],
)
def test_divergence_band_modes_reject_invalid_filter_contract(
    lowpass_steps: int, transition_steps: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        interior_divergence_band_modes(
            _operators(4, 3),
            seed=1,
            lowpass_steps=lowpass_steps,
            transition_steps=transition_steps,
        )


def test_minimum_boundary_allocation_matches_totals_and_weighted_optimum() -> None:
    operators = _operators(3, 2)
    boundary = operators.boundary_face_indices
    allowed = np.isin(operators.face_boundary_tag[boundary], (1, 2))
    totals = np.asarray([0.7, -0.2, 0.0, 1.1])

    result = minimum_winv_boundary_impulse_for_totals(
        operators,
        totals,
        allowed_boundary=allowed,
    )

    np.testing.assert_allclose(np.sum(result, axis=0), totals, atol=2.0e-15)
    assert np.max(np.abs(result[~allowed])) == 0.0
    weight = operators.face_weight[boundary]
    selected = np.flatnonzero(allowed)
    perturbation = np.zeros_like(result)
    perturbation[selected[0], 0] = 0.3
    perturbation[selected[1], 0] = -0.3
    base_energy = float(np.sum(result**2 / weight[:, None]))
    perturbed_energy = float(np.sum((result + perturbation) ** 2 / weight[:, None]))
    assert perturbed_energy > base_energy

    with pytest.raises(ValueError, match="no allowed boundary"):
        minimum_winv_boundary_impulse_for_totals(
            operators,
            np.asarray([1.0]),
            allowed_boundary=np.zeros(boundary.size, dtype=bool),
        )


def test_direct_projector_separates_target_projection_from_solve_closure() -> None:
    operators = _operators(3, 2)
    projector = factorize_direct_minimum_winv_norm_projector(operators)
    incompatible_target = np.zeros((operators.topology.num_cells, 1))
    incompatible_target[0, 0] = 1.0
    boundary = np.zeros((operators.boundary_face_indices.size, 1))

    result = projector.solve(incompatible_target, boundary)

    assert result.decoded_residual_relative_l2 > 0.1
    assert result.compatibility_relative_l2 > 0.1
    assert result.reduced_solve_residual_relative_l2 is not None
    assert result.reduced_solve_residual_relative_l2 <= 2.0e-13


@pytest.mark.parametrize(
    ("field", "mutation", "message"),
    [
        ("cell_volume", lambda value: value.__setitem__(0, 0.0), "positive"),
        ("face_owner", lambda value: value.__setitem__(0, -1), "out-of-range"),
        (
            "face_boundary_tag",
            lambda value: value.__setitem__(0, 0),
            "nonzero boundary tag",
        ),
        (
            "face_centers",
            lambda value: value.__setitem__(0, np.asarray([0.75, 0.375])),
            "dual widths",
        ),
    ],
)
def test_tampered_geometry_is_rejected(field, mutation, message: str) -> None:
    geometry = _rectangular_geometry(2, 2)
    geometry[field] = geometry[field].copy()
    mutation(geometry[field])

    with pytest.raises(ValueError, match=message):
        build_fv_impulse_operators(**geometry)


def test_boundary_error_is_not_mislabeled_as_an_interior_cycle() -> None:
    operators = _operators(2, 2)
    error = np.zeros(operators.topology.num_faces, dtype=np.float64)
    boundary_face = int(operators.boundary_face_indices[0])
    error[boundary_face] = 3.0

    result = decompose_interior_impulse_error(operators, error)

    assert result.boundary[boundary_face, 0] == 3.0
    np.testing.assert_allclose(result.divergence_active, 0.0)
    np.testing.assert_allclose(result.cycle, 0.0)
    assert result.boundary_energy > 0.0
    assert result.observed_full_decoded_gain > 0.0
    assert result.summary()["claim_boundary"] == CLAIM_BOUNDARY


def test_minimum_norm_target_matches_projected_reference_field() -> None:
    operators = _operators(3, 2)
    generator = np.random.default_rng(17)
    reference = generator.normal(size=(operators.topology.num_faces, 3))
    decomposition = decompose_interior_impulse_error(
        operators,
        reference,
        tolerance=1.0e-13,
    )
    target = operators.incidence @ reference
    result = minimum_winv_norm_face_impulse(
        operators,
        target,
        reference[operators.boundary_face_indices],
        tolerance=1.0e-13,
    )
    expected = decomposition.divergence_active + decomposition.boundary

    np.testing.assert_allclose(
        result.face_impulse, expected, rtol=2.0e-10, atol=2.0e-11
    )
    np.testing.assert_allclose(
        result.decoded_cell_integral, target, rtol=2.0e-10, atol=2.0e-11
    )
    assert result.compatibility_relative_l2 <= 2.0e-14
    assert result.decoded_residual_relative_l2 <= 2.0e-10
    assert set(result.lsmr_stop_codes) <= {1, 2}
    assert result.summary()["claim_boundary"] == CLAIM_BOUNDARY


def test_minimum_norm_target_reports_incompatible_fixed_boundary() -> None:
    operators = _operators(2, 2)
    target = np.ones((operators.topology.num_cells, 1), dtype=np.float64)
    boundary = np.zeros((operators.boundary_face_indices.size, 1), dtype=np.float64)

    result = minimum_winv_norm_face_impulse(operators, target, boundary)

    assert result.compatibility_l2 == pytest.approx(float(target.sum()))
    assert result.compatibility_relative_l2 > 1.0
    assert result.decoded_residual_relative_l2 > 0.9


def test_direct_minimum_norm_projector_matches_projected_reference() -> None:
    operators = _operators(4, 3)
    generator = np.random.default_rng(23)
    reference = generator.normal(size=(operators.topology.num_faces, 4))
    target = operators.incidence @ reference
    decomposition = decompose_interior_impulse_error(
        operators,
        reference,
        tolerance=1.0e-13,
    )
    projector = factorize_direct_minimum_winv_norm_projector(operators)

    result = projector.solve(
        target,
        reference[operators.boundary_face_indices],
    )
    expected = decomposition.divergence_active + decomposition.boundary

    np.testing.assert_allclose(
        result.face_impulse, expected, rtol=2.0e-10, atol=2.0e-11
    )
    np.testing.assert_allclose(
        result.decoded_cell_integral, target, rtol=2.0e-13, atol=2.0e-13
    )
    assert result.solver == "direct_weighted_laplacian"
    assert result.compatibility_projection_relative_l2 is not None
    assert result.compatibility_projection_relative_l2 <= 2.0e-15
    assert result.reduced_solve_residual_relative_l2 is not None
    assert result.reduced_solve_residual_relative_l2 <= 2.0e-13
    assert result.factorization_count == 1
    assert result.anchor_cells == (0,)
    assert projector.summary()["regularization"] == "none"


def test_direct_projector_reuses_factors_and_reports_incompatibility() -> None:
    operators = _operators(3, 2)
    projector = factorize_direct_minimum_winv_norm_projector(operators)
    factor_ids = tuple(id(factor) for factor in projector.factors)
    boundary = np.zeros((operators.boundary_face_indices.size, 2), dtype=np.float64)
    incompatible = np.ones((operators.topology.num_cells, 2), dtype=np.float64)

    first = projector.solve(incompatible, boundary)
    second = projector.solve(2.0 * incompatible, boundary)

    assert tuple(id(factor) for factor in projector.factors) == factor_ids
    assert first.compatibility_relative_l2 > 1.0
    assert first.compatibility_projection_relative_l2 == pytest.approx(1.0)
    assert first.decoded_residual_relative_l2 == pytest.approx(1.0)
    assert second.compatibility_projection_relative_l2 == pytest.approx(1.0)
    assert first.reduced_solve_residual_relative_l2 == 0.0
    assert second.reduced_solve_residual_relative_l2 == 0.0


def test_d046_row_metrics_close_a_synthetic_state_transition() -> None:
    operators = _operators(3, 2)
    generator = np.random.default_rng(29)
    impulse = generator.normal(scale=0.01, size=(operators.topology.num_faces, 4))
    wall = np.isin(operators.face_boundary_tag, [3, 4])
    impulse[wall[:, None] & np.array([[True, True, False, True]])] = 0.0
    current = generator.normal(size=(operators.topology.num_cells, 4))
    delta = -(operators.incidence @ impulse) / operators.cell_volume[:, None]
    reference_states = np.stack((current, current + delta), axis=0)
    geometry = SimpleNamespace(
        cell_volume=operators.cell_volume,
        face_boundary_tag=operators.face_boundary_tag,
        boundary_tag_names=("interior", "x_min", "x_max", "y_min", "y_max"),
    )

    result = _row_metrics(
        operators=operators,
        geometry=geometry,
        reference_states=reference_states,
        shard_states=reference_states.astype(np.float32).astype(np.float64),
        reference_impulse=impulse,
    )

    assert result["reference_closure_relative_l2"] <= 2.0e-14
    assert result["canonical_reference_closure_relative_l2"] <= 2.0e-10
    assert result["independent_canonical_field_relative_l2"] <= 2.0e-9
    assert result["canonical_full_winv_norm_ratio"] <= 1.0 + 2.0e-12
    assert result["wall_forbidden_exchange_absolute"] == 0.0
    assert result["finite"] is True
    aggregates = _aggregate_gate_metrics([result])
    assert (
        aggregates["reference_closure_relative_l2_max"]
        == result["reference_closure_relative_l2"]
    )


def test_d046_row_uses_state_constrained_field_as_canonical_target() -> None:
    operators = _operators(3, 2)
    generator = np.random.default_rng(41)
    reference_impulse = generator.normal(
        scale=0.01, size=(operators.topology.num_faces, 4)
    )
    wall = np.isin(operators.face_boundary_tag, [3, 4])
    reference_impulse[wall[:, None] & np.array([[True, True, False, True]])] = 0.0
    potential = generator.normal(size=operators.topology.num_cells)
    extra_interior = operators.face_weight[operators.interior_face_indices] * (
        operators.interior_incidence.T @ potential
    )
    target_integral = operators.incidence @ reference_impulse
    target_integral += (operators.interior_incidence @ extra_interior)[
        :, None
    ] * np.array([[0.03, -0.02, 0.01, 0.04]])
    current = generator.normal(size=(operators.topology.num_cells, 4))
    delta = -target_integral / operators.cell_volume[:, None]
    states = np.stack((current, current + delta), axis=0)
    geometry = SimpleNamespace(
        cell_volume=operators.cell_volume,
        face_boundary_tag=operators.face_boundary_tag,
        boundary_tag_names=("interior", "x_min", "x_max", "y_min", "y_max"),
    )

    result = _row_metrics(
        operators=operators,
        geometry=geometry,
        reference_states=states,
        shard_states=states,
        reference_impulse=reference_impulse,
    )

    assert result["reference_closure_relative_l2"] > 1.0e-3
    assert result["canonical_reference_closure_relative_l2"] <= 2.0e-10
    assert result["independent_canonical_field_relative_l2"] > 1.0e-3


def test_d047_row_uses_reused_direct_projector_and_extra_gates() -> None:
    operators = _operators(3, 2)
    projector = factorize_direct_minimum_winv_norm_projector(operators)
    generator = np.random.default_rng(53)
    impulse = generator.normal(scale=0.01, size=(operators.topology.num_faces, 4))
    wall = np.isin(operators.face_boundary_tag, [3, 4])
    impulse[wall[:, None] & np.array([[True, True, False, True]])] = 0.0
    current = generator.normal(size=(operators.topology.num_cells, 4))
    delta = -(operators.incidence @ impulse) / operators.cell_volume[:, None]
    states = np.stack((current, current + delta), axis=0)
    geometry = SimpleNamespace(
        cell_volume=operators.cell_volume,
        face_boundary_tag=operators.face_boundary_tag,
        boundary_tag_names=("interior", "x_min", "x_max", "y_min", "y_max"),
    )

    result = _row_metrics(
        operators=operators,
        geometry=geometry,
        reference_states=states,
        shard_states=states.astype(np.float32).astype(np.float64),
        reference_impulse=impulse,
        direct_projector=projector,
    )

    assert result["canonical_solver"] == "direct_weighted_laplacian"
    assert result["canonical_reference_closure_relative_l2"] <= 2.0e-13
    assert result["compatibility_projection_relative_l2"] <= 2.0e-14
    assert result["reduced_solve_residual_relative_l2"] <= 2.0e-13
    assert result["direct_factorization_count"] == 1
    assert result["direct_anchor_cells"] == [0]
    thresholds = {**GATE_THRESHOLDS, **DIRECT_GATE_THRESHOLDS}
    row_metrics = {**GATE_ROW_METRICS, **DIRECT_GATE_ROW_METRICS}
    aggregates = _aggregate_gate_metrics(
        [result],
        gate_thresholds=thresholds,
        gate_row_metrics=row_metrics,
    )
    assert (
        aggregates["direct_reduced_solve_residual_relative_l2_max"]
        == result["reduced_solve_residual_relative_l2"]
    )


def test_d049_aggregate_requires_frequency_order_and_refinement_scaling() -> None:
    canonical_rows = [
        {
            "decoded_residual_relative_l2": 1.0e-10,
            "cycle_decoded_gain": 1.0e-10,
            "target_component_decoded_gain": [0.001, 0.002, 0.003, 0.004],
        }
    ]
    band_rows = [
        {"seed": 7, "band": "low", "decoded_gain": 0.3},
        {"seed": 7, "band": "mid", "decoded_gain": 0.9},
        {"seed": 7, "band": "high", "decoded_gain": 2.2},
    ]
    resolution_rows = [
        {"mesh_nx": 25, "mesh_ny": 10, "flux_top_gain": 1.0},
        {"mesh_nx": 50, "mesh_ny": 20, "flux_top_gain": 2.0},
    ]

    result = _aggregate_d049(canonical_rows, band_rows, resolution_rows)

    assert result["passed"] is True
    assert result["classification"] == (
        "supports_face_norm_discrete_divergence_conditioning"
    )
    broken_bands = [*band_rows[:2], {**band_rows[2], "decoded_gain": 0.5}]
    rejected = _aggregate_d049(canonical_rows, broken_bands, resolution_rows)
    assert rejected["passed"] is False
    assert rejected["checks"]["band_order_fraction_min"] is False


def test_d049_resolution_parser_rejects_invalid_contracts() -> None:
    assert _parse_resolution("250x100") == (250, 100)
    with pytest.raises(ValueError, match="NXxNY"):
        _parse_resolution("250")
    with pytest.raises(ValueError, match="at least two"):
        _parse_resolution("1x100")


def test_d050_aggregate_separates_structural_and_oracle_headroom_gates() -> None:
    row = {
        "call": 60,
        "legal_state_reconstruction_relative_l2": 1.0e-10,
        "legal_lift_closure_relative_l2": 2.0e-10,
        "oracle_lift_closure_relative_l2": 3.0e-10,
        "legal_wall_exchange_absolute": 0.0,
        "raw_all_admissible": True,
        "legal_all_admissible": True,
        "oracle_all_admissible": True,
        "oracle_state_error_reduction": 0.20,
        "oracle_budget_defect_reduction": 0.95,
        "correction_to_raw_update_norm": 0.04,
        "oracle_to_raw_front_chamfer_ratio": 1.01,
        "oracle_to_raw_shock_thickness_error_ratio": 0.99,
        "oracle_to_raw_shock_strength_error_ratio": 1.02,
        "oracle_to_raw_smooth_error_ratio": 1.00,
        "raw_state_error": 0.01,
        "oracle_state_error": 0.008,
        "raw_budget_defect": 0.05,
        "oracle_budget_defect": 0.0025,
    }

    result = _aggregate_d050([row])

    assert result["structural_passed"] is True
    assert result["oracle_headroom_passed"] is True
    assert result["next_route"] == (
        "design_legal_current_state_boundary_budget_preflight_no_training_yet"
    )
    weak = _aggregate_d050([{**row, "oracle_state_error_reduction": 0.10}])
    assert weak["structural_passed"] is True
    assert weak["oracle_headroom_passed"] is False
    broken = _aggregate_d050(
        [{**row, "legal_state_reconstruction_relative_l2": 1.0e-4}]
    )
    assert broken["structural_passed"] is False
    assert broken["next_route"] == "stop_and_reaudit_D050_no_face_method"
