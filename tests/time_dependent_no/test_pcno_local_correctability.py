from __future__ import annotations

import numpy as np

from scripts.time_dependent_no.analyze_pcno_local_correctability import (
    CASE_IDS,
    _cap_scale,
    _positive_temporal_coherence,
    _quadratic_energy,
    _routing_summary,
    _shock_edge_multipliers,
    _SubspaceProjector,
)
from utility.time_dependent_no.pcno_defect_corrections import (
    DissipationResult,
    physical_cosine_basis,
    weighted_subspace_decomposition,
)


def test_positive_temporal_coherence_keeps_only_aligned_nodes() -> None:
    current = np.asarray([[1.0, 0.0], [-1.0, 0.0], [0.0, 0.0]])
    previous = np.asarray([[2.0, 0.0], [2.0, 0.0], [1.0, 0.0]])
    np.testing.assert_allclose(
        _positive_temporal_coherence(current, previous),
        np.asarray([1.0, 0.0, 0.0]),
    )
    np.testing.assert_array_equal(
        _positive_temporal_coherence(current, None), np.zeros(3)
    )


def test_shock_edge_multipliers_split_normal_and_tangential_edges() -> None:
    nodes = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    edges = np.asarray([[0, 1], [0, 2]], dtype=np.int64)
    normals = np.tile(np.asarray([[1.0, 0.0]]), (3, 1))
    normal, tangential = _shock_edge_multipliers(nodes, edges, normals)
    np.testing.assert_allclose(normal, np.asarray([1.0, 0.0]), atol=1.0e-15)
    np.testing.assert_allclose(tangential, 1.0 - normal, atol=1.0e-15)


def test_cap_scale_recovers_registered_relative_cap() -> None:
    result = DissipationResult(
        correction=np.ones((2, 1)),
        sensor=np.ones(2),
        uncapped_relative_norm=0.2,
        applied_relative_norm=0.2,
        applied_scale=1.0,
        weighted_mean_closure=np.zeros(1),
        eligible_edge_count=1,
    )
    factor, applied = _cap_scale(result, 0.05)
    assert factor == 0.25
    assert applied == 0.05


def test_cached_projector_matches_exact_weighted_decomposition() -> None:
    x = (np.arange(6, dtype=np.float64) + 0.5) / 6.0
    y = (np.arange(4, dtype=np.float64) + 0.5) / 4.0
    xx, yy = np.meshgrid(x, y)
    nodes = np.column_stack((xx.reshape(-1), yy.reshape(-1)))
    basis, _ = physical_cosine_basis(nodes, rank=4, domain_bounds=(0.0, 1.0, 0.0, 1.0))
    volumes = np.linspace(0.5, 1.5, nodes.shape[0])
    scale = np.asarray([2.0, 3.0])
    interior = np.ones(nodes.shape[0], dtype=bool)
    interior[[0, 5, 18, 23]] = False
    field = np.column_stack((nodes[:, 0] + nodes[:, 1], nodes[:, 0] ** 2))
    cached = _SubspaceProjector.build(basis, volumes, scale, interior).energies(field)
    exact = weighted_subspace_decomposition(
        field, basis, volumes, interior, component_scale=scale
    )
    np.testing.assert_allclose(cached["total"], exact.total_energy, atol=1.0e-14)
    np.testing.assert_allclose(cached["parallel"], exact.parallel_energy, atol=1.0e-14)
    np.testing.assert_allclose(
        cached["orthogonal"], exact.orthogonal_energy, atol=1.0e-14
    )
    np.testing.assert_allclose(
        cached["excluded_contact"], exact.excluded_non_type0_energy, atol=1.0e-14
    )


def test_quadratic_energy_matches_explicit_scalar_example() -> None:
    assert _quadratic_energy(4.0, -1.0, 1.0, 0.5) == 3.25


def test_routing_summary_accepts_bounded_subset_signal_case_first() -> None:
    rows = []
    for case_index, case_id in enumerate(CASE_IDS):
        rows.append(
            {
                "case_id": case_id,
                "arm": "zero",
                "pathway": "zero",
                "norm_cap": 0.0,
                "aggregate_defect_ratio": 1.0,
                "cumulative_defect_ratio": 1.0,
                "shock_aggregate_defect_ratio": 1.0,
                "vortex_aggregate_defect_ratio": 1.0,
                "correction_to_negative_defect_cosine": None,
            }
        )
        rows.append(
            {
                "case_id": case_id,
                "arm": "shock_tangential__cap0p01",
                "pathway": "shock_tangential",
                "norm_cap": 0.01,
                "aggregate_defect_ratio": 1.01,
                "cumulative_defect_ratio": 1.01,
                "shock_aggregate_defect_ratio": 0.97,
                "vortex_aggregate_defect_ratio": 1.0,
                "correction_to_negative_defect_cosine": (
                    0.1 if case_index < 2 else -0.1
                ),
            }
        )
    summary = _routing_summary(rows)
    assert summary["routing_authorized"]
    assert summary["eligible_candidates_ranked"] == ["shock_tangential__cap0p01"]
