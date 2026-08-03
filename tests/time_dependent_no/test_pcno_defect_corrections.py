from __future__ import annotations

import numpy as np

from utility.time_dependent_no.pcno_defect_corrections import (
    controlled_graph_dissipation,
    fit_error_coefficient_sequence,
    fit_weighted_coefficients,
    mean_calibration_coefficients,
    physical_cosine_basis,
    reconstruct_coefficient_sequence,
    weighted_subspace_decomposition,
)


def test_cosine_coefficients_and_sequence_reconstruct_exact_basis_field() -> None:
    x = (np.arange(8, dtype=np.float64) + 0.5) / 8.0
    y = (np.arange(4, dtype=np.float64) + 0.5) / 4.0
    xx, yy = np.meshgrid(x, y)
    nodes = np.column_stack((xx.reshape(-1), yy.reshape(-1)))
    basis, modes = physical_cosine_basis(
        nodes, rank=5, domain_bounds=(0.0, 1.0, 0.0, 1.0)
    )
    assert modes[0] == (0, 0)
    assert len(set(modes)) == 5
    coefficients = np.arange(10, dtype=np.float64).reshape(5, 2) / 7.0
    field = basis @ coefficients
    fitted = fit_weighted_coefficients(field, basis, np.ones(nodes.shape[0]))
    np.testing.assert_allclose(fitted, coefficients, atol=2.0e-9, rtol=2.0e-9)

    errors = np.stack((field, 2.0 * field), axis=0)
    scale = np.asarray([2.0, 4.0])
    sequence = fit_error_coefficient_sequence(
        errors,
        basis,
        np.ones(nodes.shape[0]),
        component_scale=scale,
    )
    reconstructed = reconstruct_coefficient_sequence(
        sequence, basis, component_scale=scale
    )
    np.testing.assert_allclose(reconstructed, errors, atol=3.0e-9, rtol=3.0e-9)
    frozen = mean_calibration_coefficients(np.stack((sequence, 3.0 * sequence)))
    np.testing.assert_allclose(frozen, 2.0 * sequence)


def test_controlled_graph_dissipation_is_capped_mean_neutral_and_smoothing() -> None:
    edges = np.asarray([[0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2]], dtype=np.int64)
    weights = np.asarray([1.0, 2.0, 3.0, 4.0])
    update = np.asarray([[0.0], [0.0], [8.0], [0.0]])
    result = controlled_graph_dissipation(
        update,
        edges,
        weights,
        np.ones(4, dtype=bool),
        component_scale=(2.0,),
        sensor_quantile=0.5,
        norm_cap=0.05,
    )
    assert result.eligible_edge_count == 3
    assert result.applied_relative_norm <= 0.05 + 1.0e-12
    np.testing.assert_allclose(result.weighted_mean_closure, 0.0, atol=1.0e-14)
    before = np.square(update[edges[::2, 0]] - update[edges[::2, 1]]).sum()
    corrected = update + result.correction
    after = np.square(corrected[edges[::2, 0]] - corrected[edges[::2, 1]]).sum()
    assert after < before


def test_controlled_graph_dissipation_leaves_non_normal_nodes_untouched() -> None:
    edges = np.asarray([[0, 1], [1, 2], [2, 3]], dtype=np.int64)
    update = np.asarray([[0.0], [5.0], [0.0], [0.0]])
    result = controlled_graph_dissipation(
        update,
        edges,
        np.ones(4),
        np.asarray([True, True, False, True]),
        component_scale=(1.0,),
        sensor_quantile=0.5,
        norm_cap=0.1,
    )
    assert result.eligible_edge_count == 1
    assert result.correction[2, 0] == 0.0
    assert result.correction[3, 0] == 0.0
    np.testing.assert_allclose(result.weighted_mean_closure, 0.0, atol=1.0e-14)


def test_weighted_subspace_decomposition_is_orthogonal_and_keeps_contacts() -> None:
    x = (np.arange(6, dtype=np.float64) + 0.5) / 6.0
    y = (np.arange(4, dtype=np.float64) + 0.5) / 4.0
    xx, yy = np.meshgrid(x, y)
    nodes = np.column_stack((xx.reshape(-1), yy.reshape(-1)))
    basis, _ = physical_cosine_basis(nodes, rank=4, domain_bounds=(0.0, 1.0, 0.0, 1.0))
    interior = np.ones(nodes.shape[0], dtype=bool)
    interior[[0, 5, 18, 23]] = False
    weights = np.linspace(0.5, 1.5, nodes.shape[0])
    scale = np.asarray((2.0, 3.0))
    coefficients = np.asarray(((1.0, -0.5), (0.2, 0.3), (-0.4, 0.1), (0.05, -0.2)))
    field = basis @ coefficients * scale[None, :]
    field[:, 0] += 0.15 * nodes[:, 0] * nodes[:, 1]
    field[~interior] += np.asarray((7.0, -4.0))

    result = weighted_subspace_decomposition(
        field,
        basis,
        weights,
        interior,
        component_scale=scale,
    )

    assert result.effective_rank == 4
    np.testing.assert_array_equal(result.parallel[~interior], 0.0)
    np.testing.assert_array_equal(result.orthogonal[~interior], 0.0)
    np.testing.assert_array_equal(result.excluded_non_type0[interior], 0.0)
    np.testing.assert_allclose(
        result.parallel + result.orthogonal + result.excluded_non_type0,
        field,
        atol=1.0e-14,
    )
    assert abs(result.parallel_orthogonal_inner) < 1.0e-13
    assert abs(result.energy_closure) < 1.0e-12
    assert result.maximum_reconstruction_error < 1.0e-14
    np.testing.assert_allclose(
        result.component_total_energy,
        result.component_parallel_energy
        + result.component_orthogonal_energy
        + result.component_excluded_non_type0_energy,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        result.component_parallel_orthogonal_inner,
        0.0,
        atol=1.0e-12,
    )


def test_weighted_subspace_decomposition_rejects_rank_deficiency() -> None:
    field = np.ones((4, 2))
    duplicated_basis = np.ones((4, 2))
    with np.testing.assert_raises_regex(ValueError, "rank deficient"):
        weighted_subspace_decomposition(
            field,
            duplicated_basis,
            np.ones(4),
            np.ones(4, dtype=bool),
            component_scale=(1.0, 1.0),
        )


def test_local_sensor_is_invariant_to_non_normal_node_values() -> None:
    edges = np.asarray([[0, 1], [1, 2]], dtype=np.int64)
    normal = np.asarray([True, True, False])
    baseline = np.asarray([[0.0], [5.0], [0.0]])
    changed_boundary = np.asarray([[0.0], [5.0], [1.0e6]])
    first = controlled_graph_dissipation(
        baseline,
        edges,
        np.ones(3),
        normal,
        component_scale=(1.0,),
        sensor_quantile=0.5,
        norm_cap=0.1,
    )
    second = controlled_graph_dissipation(
        changed_boundary,
        edges,
        np.ones(3),
        normal,
        component_scale=(1.0,),
        sensor_quantile=0.5,
        norm_cap=0.1,
    )
    np.testing.assert_allclose(first.sensor, second.sensor)
    np.testing.assert_allclose(first.correction, second.correction)
