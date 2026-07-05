from __future__ import annotations

import numpy as np

from utility.time_dependent_no.cpg_interface_latents import (
    admissibility_summary,
    classify_wave_edges,
    clip_reconstruct_to_edge_box,
    conservative_update_from_flux,
    edge_pressure_jump_scores,
    edge_shock_mask_from_scores,
    flux_error_summary,
    divergence_cancellation_summary,
    llf_flux_decomposition,
    midpoint_reconstruct_prims,
    owner_neighbor_reconstruct_prims,
    split_directed_reconstruct_prims,
    summarize_trace_likeness,
    trace_likeness_arrays,
    time_midpoint_reconstruct_prims,
    update_error_summary,
)


def test_llf_decomposition_reduces_to_physical_flux_for_equal_states():
    primitive = np.array([[1.0, 2.0, 0.0, 1.0]], dtype=np.float64)
    normals = np.array([[1.0, 0.0]], dtype=np.float64)

    decomp = llf_flux_decomposition(primitive, primitive, normals)

    np.testing.assert_allclose(decomp["dissipation"], 0.0)
    np.testing.assert_allclose(decomp["flux"], np.array([[2.0, 5.0, 0.0, 11.0]]))
    np.testing.assert_allclose(decomp["wave_speed"], np.array([2.0 + np.sqrt(1.4)]))


def test_directed_flux_update_matches_author_antisymmetric_layout():
    flux = np.array([[2.0, 3.0, 4.0, 5.0]], dtype=np.float64)
    edge_factor = np.array([[0.5], [0.25]], dtype=np.float64)
    full_edges = np.array([[0, 1], [1, 0]], dtype=np.int64)

    update = conservative_update_from_flux(
        flux, edge_factor, full_edges, num_nodes=2
    )

    np.testing.assert_allclose(update["messages"][0], 0.5 * flux[0])
    np.testing.assert_allclose(update["messages"][1], -0.25 * flux[0])
    np.testing.assert_allclose(update["conservative_delta"][1], -0.5 * flux[0])
    np.testing.assert_allclose(update["conservative_delta"][0], 0.25 * flux[0])


def test_trace_likeness_uses_target_node_as_left_owner():
    nodes = np.array(
        [[1.0, 0.0, 0.0, 1.0], [2.0, 0.0, 0.0, 3.0]], dtype=np.float64
    )
    unique_edges = np.array([[0, 1]], dtype=np.int64)
    reconstruct = np.array(
        [[2.0, 0.0, 0.0, 3.0], [1.0, 0.0, 0.0, 1.0]], dtype=np.float64
    )

    left, right = split_directed_reconstruct_prims(reconstruct)
    arrays = trace_likeness_arrays(left, right, nodes, unique_edges)
    summary = summarize_trace_likeness(arrays)

    assert arrays["left_owner_l2"][0] == 0.0
    assert arrays["right_owner_l2"][0] == 0.0
    assert summary["left_owner_closer_fraction"] == 1.0
    assert summary["right_owner_closer_fraction"] == 1.0
    assert summary["bounded_fraction"]["pres"] == 1.0


def test_trace_likeness_reports_interface_overshoot():
    nodes = np.array(
        [[1.0, 0.0, 0.0, 1.0], [2.0, 0.0, 0.0, 3.0]], dtype=np.float64
    )
    unique_edges = np.array([[0, 1]], dtype=np.int64)
    left = np.array([[2.0, 0.0, 0.0, 4.0]], dtype=np.float64)
    right = np.array([[1.0, 0.0, 0.0, 1.0]], dtype=np.float64)

    arrays = trace_likeness_arrays(left, right, nodes, unique_edges)
    summary = summarize_trace_likeness(arrays)

    np.testing.assert_allclose(arrays["left_overshoot"][0, 3], 1.0)
    assert summary["bounded_fraction"]["pres"] == 0.5
    assert summary["overshoot_max"]["pres"] == 1.0


def test_admissibility_and_update_error_summaries_are_region_safe():
    left = np.array([[1.0, 0.0, 0.0, 1.0]], dtype=np.float64)
    right = np.array([[2.0, 0.0, 0.0, -0.5]], dtype=np.float64)
    nodes = np.array(
        [[1.0, 0.0, 0.0, 1.0], [2.0, 0.0, 0.0, 3.0]], dtype=np.float64
    )
    induced = np.array([[1.0, 0.0, 0.0, 2.0], [0.0, 1.0, 0.0, 0.0]])
    truth = induced.copy()

    admissibility = admissibility_summary(left, right, nodes)
    update = update_error_summary(induced, truth, node_mask=np.array([True, False]))

    assert not admissibility["right_interface"]["all_positive"]
    assert admissibility["right_interface"]["nonpositive_pressure_count"] == 1
    assert update["node_count"] == 1
    assert update["relative_l2"]["energy"] == 0.0


def test_edge_shock_mask_uses_pressure_jump_quantile():
    nodes = np.array(
        [
            [1.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 2.0],
            [1.0, 0.0, 0.0, 10.0],
        ],
        dtype=np.float64,
    )
    edges = np.array([[0, 1], [1, 2]], dtype=np.int64)

    scores = edge_pressure_jump_scores(nodes, edges)
    mask = edge_shock_mask_from_scores(scores, quantile=0.75)

    np.testing.assert_allclose(scores, np.array([1.0, 8.0]))
    np.testing.assert_array_equal(mask, np.array([False, True]))


def test_wave_type_classification_separates_basic_edge_families():
    nodes = np.array(
        [
            [1.0, -1.0, 0.0, 1.0],
            [3.0, 1.0, 0.0, 8.0],
            [1.0, -1.0, 0.0, 3.0],
            [1.0, 1.0, 0.0, 2.0],
            [1.0, 1.0, 0.0, 3.0],
            [1.0, -1.0, 0.0, 2.0],
            [3.0, 0.0, 2.0, 1.0],
            [1.0, 0.0, -2.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    edges = np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], dtype=np.int64)
    normals = np.tile(np.array([[1.0, 0.0]], dtype=np.float64), (edges.shape[0], 1))
    shock = np.array([True, False, False, False, False])

    result = classify_wave_edges(
        nodes,
        edges,
        normals,
        shock_edge_mask=shock,
        high_quantile=0.5,
        smooth_quantile=0.25,
    )
    masks = result["masks"]

    assert masks["shock_front"][0]
    assert masks["compression"][1]
    assert masks["rarefaction"][2]
    assert masks["contact_like"][3]
    assert masks["smooth"][4]

def test_physical_trace_candidates_use_author_pair_layout():
    current = np.array(
        [[1.0, 0.0, 0.0, 1.0], [3.0, 2.0, 0.0, 5.0]], dtype=np.float64
    )
    target = current + np.array(
        [[0.2, 0.0, 0.0, 0.4], [0.4, 0.0, 0.0, 0.6]], dtype=np.float64
    )
    edges = np.array([[0, 1]], dtype=np.int64)

    left, right = owner_neighbor_reconstruct_prims(current, edges)
    mid_left, mid_right = midpoint_reconstruct_prims(current, edges)
    time_left, time_right = time_midpoint_reconstruct_prims(current, target, edges)

    np.testing.assert_allclose(left, current[[1]])
    np.testing.assert_allclose(right, current[[0]])
    np.testing.assert_allclose(mid_left, np.array([[2.0, 1.0, 0.0, 3.0]]))
    np.testing.assert_allclose(mid_right, mid_left)
    np.testing.assert_allclose(time_left, np.array([[3.2, 2.0, 0.0, 5.3]]))
    np.testing.assert_allclose(time_right, np.array([[1.1, 0.0, 0.0, 1.2]]))


def test_clip_reconstruct_to_edge_box_enforces_bounded_positive_states():
    nodes = np.array(
        [[1.0, -2.0, 0.0, 1.0], [3.0, 2.0, 0.0, 5.0]], dtype=np.float64
    )
    edges = np.array([[0, 1]], dtype=np.int64)
    left = np.array([[10.0, 5.0, 0.0, 7.0]], dtype=np.float64)
    right = np.array([[-1.0, -5.0, 0.0, -2.0]], dtype=np.float64)

    clipped_left, clipped_right = clip_reconstruct_to_edge_box(
        left, right, nodes, edges, positivity_floor=0.25
    )

    np.testing.assert_allclose(clipped_left, np.array([[3.0, 2.0, 0.0, 5.0]]))
    np.testing.assert_allclose(clipped_right, np.array([[1.0, -2.0, 0.0, 1.0]]))


def test_flux_error_summary_and_divergence_cancellation_are_consistent():
    reference = np.array([[2.0, 0.0, 0.0, 0.0]], dtype=np.float64)
    candidate = np.array([[3.0, 0.0, 0.0, 4.0]], dtype=np.float64)
    edge_factor = np.array([[1.0], [1.0]], dtype=np.float64)
    full_edges = np.array([[0, 1], [1, 0]], dtype=np.int64)

    flux = flux_error_summary(candidate, reference)
    cancellation = divergence_cancellation_summary(
        candidate, reference, edge_factor, full_edges, num_nodes=2
    )

    assert flux["edge_count"] == 1
    np.testing.assert_allclose(flux["relative_l2"]["rho"], 0.5)
    np.testing.assert_allclose(flux["relative_l2"]["energy"], 4.0 / 1.0e-12)
    np.testing.assert_allclose(
        cancellation["weighted_message_delta_l2"], np.sqrt(34.0)
    )
    np.testing.assert_allclose(cancellation["node_update_delta_l2"], np.sqrt(34.0))
    np.testing.assert_allclose(cancellation["update_to_message_l2_ratio"], 1.0)


