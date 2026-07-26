from __future__ import annotations

import numpy as np
import pytest

from scripts.time_dependent_no.diagnose_pcno_euler2d_multirate_headroom import (
    ANTI_SMEARING_METRICS,
    anti_smearing_acceptance,
    convex_blend,
    promotion_decision,
    proposal_sensor_metrics,
    select_oracle_blend,
    validate_trajectory_pair,
)


def _admissible_state(density: np.ndarray) -> np.ndarray:
    values = np.asarray(density, dtype=np.float64).reshape(-1)
    return np.stack(
        [values, np.zeros_like(values), np.zeros_like(values), 2.5 * values],
        axis=-1,
    )


def test_oracle_blend_selects_admissible_interior_alpha() -> None:
    target = _admissible_state(np.asarray([1.0, 1.0]))
    direct = _admissible_state(np.asarray([0.8, 1.2]))
    composed = _admissible_state(np.asarray([1.2, 0.8]))

    result = select_oracle_blend(
        direct,
        composed,
        target,
        volumes=np.asarray([0.25, 0.75]),
        component_scale=np.ones(4),
        gamma=1.4,
        alpha_grid=np.asarray([0.0, 0.5, 1.0]),
    )

    assert result["alpha"] == 0.5
    np.testing.assert_allclose(result["prediction"], target)
    assert result["state_error"] == pytest.approx(0.0)
    assert all(row["raw_admissible"] for row in result["curve"])


def test_convex_blend_rejects_extrapolation() -> None:
    state = _admissible_state(np.asarray([1.0]))
    with pytest.raises(ValueError, match="alpha"):
        convex_blend(state, state, 1.01)


def test_disagreement_sensor_reports_correlation_and_capture() -> None:
    target = _admissible_state(np.ones(10))
    direct = target.copy()
    composed = target.copy()
    ramp = np.linspace(0.01, 0.10, 10)
    direct[:, 0] += ramp
    composed[:, 0] -= ramp

    result = proposal_sensor_metrics(
        direct,
        composed,
        target,
        component_scale=np.ones(4),
        volumes=np.ones(10),
    )

    correlation = result["correlation_with_better_parent_error"]
    assert correlation["spearman"] == pytest.approx(1.0)
    assert result["top20_disagreement_capture_of_better_parent_error_energy"] > 0.3


def test_anti_smearing_uses_metricwise_better_parent_envelope() -> None:
    direct = {name: 1.0 for name in ANTI_SMEARING_METRICS}
    composed = {name: 2.0 for name in ANTI_SMEARING_METRICS}
    accepted = {name: 1.04 for name in ANTI_SMEARING_METRICS}
    rejected = dict(accepted)
    rejected[ANTI_SMEARING_METRICS[0]] = 1.06

    assert anti_smearing_acceptance(direct, composed, accepted)["passed"]
    report = anti_smearing_acceptance(direct, composed, rejected)
    assert not report["passed"]
    assert not report["metrics"][ANTI_SMEARING_METRICS[0]]["accepted"]


def _promotion_rows() -> list[dict[str, object]]:
    rows = []
    for trajectory in range(6):
        for call in (1, 5, 15, 30):
            rows.append(
                {
                    "trajectory": f"case_{trajectory}",
                    "call": call,
                    "all_variants_raw_admissible": True,
                    "oracle_state_reduction_vs_better_parent": 0.15,
                    "oracle_highpass_reduction_vs_stride2": 0.25,
                    "oracle_state_and_highpass_nonworse_than_both": True,
                    "oracle_anti_smearing": {"passed": True},
                    "proposal_sensor": {
                        "correlation_with_better_parent_error": {"spearman": 0.6},
                        "top20_disagreement_capture_of_better_parent_error_energy": 0.6,
                    },
                }
            )
    return rows


def test_promotion_decision_is_conjunctive() -> None:
    rows = _promotion_rows()
    passed = promotion_decision(rows, expected_trajectories=6)
    assert passed["passed"]
    assert passed["decision"].endswith("tiny_fit_only")
    assert passed["autonomous_indicator"]["passed"]

    for row in rows:
        if row["call"] == 30:
            row["oracle_highpass_reduction_vs_stride2"] = 0.05
    failed = promotion_decision(rows, expected_trajectories=6)
    assert not failed["passed"]
    assert not failed["gates"]["h60_median_highpass_reduction_vs_stride2_at_least_0p20"]


def _paired_artifacts() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    nodes = np.asarray([[0.0, 0.0], [1.0, 0.0]])
    faces = np.asarray([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]])
    common = {
        "initial_state": _admissible_state(np.ones(2)),
        "positions": nodes,
        "physical_cell_volumes": np.asarray([0.5, 0.5]),
        "face_centers": faces,
        "face_measures": np.ones(3),
        "face_normals": np.asarray([[-1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]),
        "physical_times": np.arange(5, dtype=np.float64) * 0.01,
        "reference_interval_boundary_exchange": np.zeros((4, 4)),
        "state_component_scale": np.ones(4),
        "edges": np.asarray([[0, 1]], dtype=np.int64),
        "node_type": np.zeros(2, dtype=np.int64),
        "mesh_cell_to_graph_node": np.arange(2, dtype=np.int64),
        "face_owner": np.asarray([0, 0, 1], dtype=np.int64),
        "face_neighbor": np.asarray([-1, 1, -1], dtype=np.int64),
        "face_boundary_tag": np.asarray([1, 0, 2], dtype=np.int64),
        "coordinate_convention": np.asarray("xy"),
        "face_orientation_convention": np.asarray("owner_outward"),
        "geometry_digest": np.asarray("digest"),
        "failure_cause": np.asarray("completed"),
    }
    state1 = _admissible_state(np.asarray([1.0, 1.0]))
    state2 = _admissible_state(np.asarray([1.1, 0.9]))
    stride1 = {
        **common,
        "valid_length": np.asarray(4),
        "targets": np.stack([state1, state2, state1, state2]),
    }
    stride2 = {
        **common,
        "valid_length": np.asarray(2),
        "targets": np.stack([state2, state2]),
    }
    return stride1, stride2


def test_trajectory_pair_requires_even_frame_target_alignment() -> None:
    stride1, stride2 = _paired_artifacts()
    checks = validate_trajectory_pair(stride1, stride2, required_stride2_calls=2)
    assert checks["maximum_even_frame_target_difference"] == 0.0

    stride2["targets"] = stride2["targets"].copy()
    stride2["targets"][1, 0, 0] += 0.01
    with pytest.raises(ValueError, match="even-frame targets"):
        validate_trajectory_pair(stride1, stride2, required_stride2_calls=2)
