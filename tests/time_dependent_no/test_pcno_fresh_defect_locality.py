from __future__ import annotations

import numpy as np
import pytest

from scripts.time_dependent_no.diagnose_pcno_euler2d_fresh_defect_locality import (
    FRESH_LOCALITY_SCHEMA,
    build_graph_adjacency,
    fresh_defect_locality_metrics,
    fresh_defect_locality_selector,
    graph_hop_distance,
)
from scripts.time_dependent_no.diagnose_pcno_euler2d_proposal_sensor import (
    PROPOSAL_SENSOR_SCHEMA,
    proposal_sensor_metrics,
    proposal_sensor_selector,
)
from scripts.time_dependent_no.evaluate_pcno_euler2d_proposal_correction_oracle import (
    REALIZABILITY_SCHEMA,
    realizability_selector,
    volume_balanced_support_direction,
    volume_weighted_scaled_norm,
)


def test_locality_metrics_use_bounded_graph_halo_and_sparse_oracle() -> None:
    edges = np.asarray([[index, index + 1] for index in range(9)], dtype=np.int64)
    adjacency = build_graph_adjacency(10, edges)
    seed = np.zeros(10, dtype=bool)
    seed[2] = True
    np.testing.assert_array_equal(
        graph_hop_distance(adjacency, seed),
        np.asarray([2, 1, 0, 1, 2, 3, 4, 5, 6, 7]),
    )

    highpass = np.zeros((10, 4), dtype=np.float64)
    highpass[5, 0] = 3.0
    highpass[6, 0] = -4.0
    interior = np.ones(10, dtype=bool)
    smooth = np.zeros(10, dtype=bool)
    smooth[5:] = True
    metrics = fresh_defect_locality_metrics(
        highpass,
        edges=edges,
        adjacency=adjacency,
        weights=np.ones(10),
        interior_mask=interior,
        smooth_mask=smooth,
        current_shock_seed=seed,
        current_pressure_jump_score=np.arange(10, dtype=np.float64),
    )

    assert metrics["status"] == "available"
    assert metrics["truth_oracle"]["selected_node_count"] == 1
    assert metrics["truth_oracle"]["smooth_highpass_energy_capture"] == pytest.approx(
        16.0 / 25.0
    )
    assert metrics["current_state_sensor"]["halo_fraction_interior"] == pytest.approx(
        0.7
    )
    assert metrics["current_state_sensor"][
        "smooth_highpass_energy_capture"
    ] == pytest.approx(1.0)
    assert metrics["distance_rings"]["hop_3_to_4"][
        "smooth_highpass_energy_fraction"
    ] == pytest.approx(1.0)


def _selector_rows(
    oracle_capture: float, halo_capture: float, halo_support: float
) -> list[dict]:
    rows = []
    for trajectory in range(6):
        for call_index in range(1, 61):
            rows.append(
                {
                    "schema": FRESH_LOCALITY_SCHEMA,
                    "trajectory": str(trajectory),
                    "call_index": call_index,
                    "locality": {
                        "status": "available",
                        "truth_oracle": {
                            "smooth_highpass_energy_capture": oracle_capture,
                        },
                        "current_state_sensor": {
                            "smooth_highpass_energy_capture": halo_capture,
                            "halo_fraction_interior": halo_support,
                        },
                    },
                    "mask_contract": {"smooth_fallback_to_interior": False},
                    "teacher_admissibility": {"all_admissible": True},
                    "d053_replay": {
                        "interior_fresh_norm_relative_error": 0.0,
                        "smooth_highpass_fresh_norm_relative_error": 0.0,
                    },
                }
            )
    return rows


def test_selector_requires_oracle_and_causal_repetition() -> None:
    selected = fresh_defect_locality_selector(_selector_rows(0.8, 0.6, 0.2))
    assert selected["contract_complete"]
    assert selected["classification"] == "shock_conditioned_local_target_candidate"
    assert selected["route"] == "authorize_one_matched_tiny_fit_contract_only"

    unidentifiable = fresh_defect_locality_selector(
        _selector_rows(0.8, 0.2, 0.2)
    )
    assert (
        unidentifiable["classification"]
        == "localized_but_not_causally_shock_localizable"
    )
    assert unidentifiable["route"] == "reject_simple_shock_gate_no_training"

    diffuse = fresh_defect_locality_selector(_selector_rows(0.5, 0.2, 0.2))
    assert diffuse["classification"] == "diffuse_or_unresolved_fresh_defect"
    assert diffuse["route"] == "reject_local_target_no_training"

    incomplete_rows = _selector_rows(0.8, 0.6, 0.2)[:-1]
    incomplete = fresh_defect_locality_selector(incomplete_rows)
    assert not incomplete["contract_complete"]
    assert incomplete["classification"] == "incomplete_contract"


def test_proposal_sensor_selects_only_from_legal_nonshock_nodes() -> None:
    edges = np.asarray([[index, index + 1] for index in range(9)], dtype=np.int64)
    adjacency = build_graph_adjacency(10, edges)
    proposal = np.zeros((10, 4), dtype=np.float64)
    fresh = np.zeros_like(proposal)
    proposal[6, 0] = 5.0
    proposal[5, 0] = 4.0
    fresh[6, 0] = -4.0
    fresh[5, 0] = 3.0
    interior = np.ones(10, dtype=bool)
    smooth = np.zeros(10, dtype=bool)
    smooth[5:] = True
    seed = np.zeros(10, dtype=bool)
    seed[2] = True

    metrics = proposal_sensor_metrics(
        proposal,
        fresh,
        adjacency=adjacency,
        weights=np.ones(10),
        interior_mask=interior,
        d053_smooth_mask=smooth,
        current_shock_seed=seed,
    )

    assert metrics["status"] == "available"
    assert metrics["eligible_node_count"] == 5
    assert metrics["selected_node_count"] == 1
    assert metrics["selected_fraction_interior"] == pytest.approx(0.1)
    assert metrics["smooth_highpass_energy_capture"] == pytest.approx(16.0 / 25.0)
    assert metrics["truth_oracle_smooth_highpass_energy_capture"] == pytest.approx(
        16.0 / 25.0
    )


def _proposal_selector_rows(capture: float) -> list[dict]:
    rows = []
    for trajectory in range(6):
        for call_index in (1, 10, 30, 60):
            rows.append(
                {
                    "schema": PROPOSAL_SENSOR_SCHEMA,
                    "trajectory": str(trajectory),
                    "call_index": call_index,
                    "sensor": {
                        "status": "available",
                        "smooth_highpass_energy_capture": capture,
                        "selected_fraction_interior": 0.15,
                    },
                    "mask_contract": {"smooth_fallback_to_interior": False},
                    "teacher_admissibility": {"all_admissible": True},
                    "source_replay": {
                        "d053_interior_fresh_norm_relative_error": 0.0,
                        "d053_smooth_highpass_fresh_norm_relative_error": 0.0,
                        "d054_truth_oracle_capture_relative_error": 0.0,
                    },
                }
            )
    return rows


def test_proposal_sensor_selector_routes_only_repeated_capture() -> None:
    selected = proposal_sensor_selector(_proposal_selector_rows(0.6))
    assert selected["contract_complete"]
    assert selected["classification"] == "proposal_self_sensor_candidate"
    assert selected["route"] == "authorize_one_matched_tiny_fit_contract_only"

    rejected = proposal_sensor_selector(_proposal_selector_rows(0.4))
    assert rejected["classification"] == "sparse_but_not_legally_self_localizable"
    assert rejected["route"] == "reject_sensor_gated_local_detail_no_training"

    incomplete = proposal_sensor_selector(_proposal_selector_rows(0.6)[:-1])
    assert not incomplete["contract_complete"]
    assert incomplete["classification"] == "incomplete_contract"


def test_volume_balanced_support_direction_is_local_and_exact() -> None:
    prediction = np.zeros((4, 2), dtype=np.float64)
    target = np.asarray(
        [[1.0, 2.0], [3.0, -1.0], [8.0, 4.0], [9.0, 5.0]],
        dtype=np.float64,
    )
    support = np.asarray([True, True, False, False])
    volumes = np.asarray([1.0, 3.0, 2.0, 4.0])
    direction = volume_balanced_support_direction(
        prediction, target, support, volumes
    )

    np.testing.assert_allclose(direction[~support], 0.0)
    np.testing.assert_allclose(np.sum(volumes[:, None] * direction, axis=0), 0.0)
    assert volume_weighted_scaled_norm(
        direction, volumes, np.ones(2)
    ) == pytest.approx(np.sqrt(0.975))


def _realizability_rows(state_reduction: float, smooth_reduction: float) -> list[dict]:
    rows = []
    for trajectory in range(6):
        for call_index in (1, 10, 30, 60):
            rows.append(
                {
                    "schema": REALIZABILITY_SCHEMA,
                    "trajectory": str(trajectory),
                    "call_index": call_index,
                    "support_fraction_interior": 0.17,
                    "oracle": {
                        "raw_admissible": True,
                        "anti_smearing_pass": True,
                        "applied_update_ratio": 0.08,
                        "volume_integral_correction_max_abs": 1.0e-14,
                        "state_error_reduction": state_reduction,
                        "smooth_highpass_error_reduction": smooth_reduction,
                    },
                    "d055_replay": {
                        "support_fraction_relative_error": 0.0,
                        "fresh_energy_capture_relative_error": 0.0,
                    },
                }
            )
    return rows


def test_realizability_selector_requires_state_and_highpass_headroom() -> None:
    selected = realizability_selector(_realizability_rows(0.2, 0.3))
    assert selected["contract_complete"]
    assert selected["classification"] == "bounded_causal_support_correction_realizable"
    assert (
        selected["route"]
        == "authorize_one_frozen_global_tiny_detail_fit_implementation"
    )

    rejected = realizability_selector(_realizability_rows(0.1, 0.3))
    assert rejected["classification"] == "bounded_causal_support_correction_insufficient"
    assert rejected["route"] == "reject_learned_detail_route_no_training"

    incomplete = realizability_selector(_realizability_rows(0.2, 0.3)[:-1])
    assert not incomplete["contract_complete"]
    assert incomplete["classification"] == "incomplete_contract"
