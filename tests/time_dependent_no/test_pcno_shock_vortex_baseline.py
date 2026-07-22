from __future__ import annotations

import numpy as np

from scripts.time_dependent_no.evaluate_pcno_shock_vortex_baseline import (
    aggregate_variant,
    endpoint_metrics,
    owned_state_frame,
    owned_store_array,
    parameter_time_control_weights,
    physical_call_metrics,
)


def test_owned_store_array_does_not_alias_an_evictable_source() -> None:
    source = np.arange(6, dtype=np.int64).reshape(3, 2)

    class FakeStore:
        def array(self, key: str, name: str) -> np.ndarray:
            assert key == "case" and name == "edges"
            return source

    owned = owned_store_array(FakeStore(), "case", "edges", dtype=np.int64)
    source[:] = -1

    np.testing.assert_array_equal(owned, np.arange(6).reshape(3, 2))
    assert owned.flags.owndata


def test_owned_state_frame_does_not_alias_an_evictable_source() -> None:
    source = np.arange(24, dtype=np.float32).reshape(2, 3, 4)

    class FakeStore:
        def states(self, key: str) -> np.ndarray:
            assert key == "case"
            return source

    owned = owned_state_frame(FakeStore(), "case", 1)
    source[:] = -1.0

    np.testing.assert_array_equal(owned, np.arange(12, 24).reshape(3, 4))
    assert owned.flags.owndata


def test_parameter_time_controls_are_train_only_scaled_and_deterministic() -> None:
    candidates = {
        "b": {"vortex_epsilon": 0.0, "vortex_y": 0.0},
        "a": {"vortex_epsilon": 1.0, "vortex_y": 0.0},
        "d": {"vortex_epsilon": 0.0, "vortex_y": 2.0},
        "c": {"vortex_epsilon": 1.0, "vortex_y": 2.0},
        "far": {"vortex_epsilon": 3.0, "vortex_y": 2.0},
    }
    result = parameter_time_control_weights(
        {"vortex_epsilon": 0.9, "vortex_y": 0.1},
        candidates,
        neighbor_count=4,
    )

    assert result["nearest_key"] == "a"
    assert result["selected_keys"][0] == "a"
    assert "far" not in result["selected_keys"]
    np.testing.assert_allclose(sum(result["inverse_distance_weights"]), 1.0)


def test_physical_call_metrics_uses_recorded_outward_exchange() -> None:
    initial = np.asarray([[1.0, 0.4, 0.0, 2.5], [0.8, 0.2, 0.0, 2.0]], dtype=np.float64)
    delta = np.asarray([0.1, -0.04, 0.02, 0.2], dtype=np.float64)
    target = initial + delta
    volumes = np.asarray([0.25, 0.75], dtype=np.float64)
    reference_exchange = -delta

    exact = physical_call_metrics(
        target,
        target,
        initial,
        volumes=volumes,
        reference_cumulative_boundary_exchange=reference_exchange,
        component_scale=np.ones(4),
    )
    np.testing.assert_allclose(exact["physical_total_error"], 0.0)
    np.testing.assert_allclose(
        exact["target_reference_balance_defect"], 0.0, atol=1.0e-14
    )
    np.testing.assert_allclose(
        exact["prediction_reference_balance_defect"], 0.0, atol=1.0e-14
    )

    prediction = target.copy()
    prediction[:, 0] += 0.05
    perturbed = physical_call_metrics(
        prediction,
        target,
        initial,
        volumes=volumes,
        reference_cumulative_boundary_exchange=reference_exchange,
        component_scale=np.ones(4),
    )
    np.testing.assert_allclose(
        perturbed["prediction_reference_balance_defect"],
        [0.05, 0.0, 0.0, 0.0],
        atol=1.0e-14,
    )
    assert perturbed["prediction_reference_balance_component_scaled_rmse"] > 0.0


def test_endpoint_metrics_uses_physical_volume_smooth_region() -> None:
    positions = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float64
    )
    edges = np.asarray([[0, 1], [0, 2], [1, 3], [2, 3]], dtype=np.int64)
    target = np.asarray(
        [
            [1.0, 0.2, 0.0, 2.6],
            [1.0, 0.2, 0.0, 2.6],
            [0.7, 0.05, 0.0, 1.8],
            [0.7, 0.05, 0.0, 1.8],
        ],
        dtype=np.float64,
    )
    prediction = target.copy()
    prediction[0, 0] += 0.01
    metrics = endpoint_metrics(
        prediction,
        target,
        positions=positions,
        edges=edges,
        volumes=np.asarray([0.1, 0.2, 0.3, 0.4]),
        component_scale=np.ones(4),
        gamma=1.4,
        shock_quantile=0.5,
        vortex_center=(0.0, 0.0),
    )
    assert metrics["smooth_region_contract"].startswith("target_pressure_front")
    assert 0.0 <= metrics["smooth_region_volume_fraction"] <= 1.0
    assert metrics["smooth_region_graph_highpass_energy"] >= 0.0
    assert metrics["front_iou"] is not None
    assert metrics["vortex_core_density_relative_error"] is not None
    assert metrics["vortex_core_density_relative_error"] >= 0.0


def test_aggregate_variant_distinguishes_censoring_contracts() -> None:
    trajectories = [
        {
            "trajectory": "a",
            "completed": True,
            "valid_length": 3,
            "survival_fraction": 1.0,
            "failure_cause": "completed",
            "total_forward_seconds": 0.3,
        },
        {
            "trajectory": "b",
            "completed": False,
            "valid_length": 2,
            "survival_fraction": 2.0 / 3.0,
            "failure_cause": "inadmissible_state",
            "total_forward_seconds": 0.2,
        },
    ]
    calls = []
    for trajectory, length, offset in (("a", 3, 0.0), ("b", 2, 0.1)):
        for call in range(1, length + 1):
            calls.append(
                {
                    "trajectory": trajectory,
                    "call": call,
                    "scaled_relative_l2_physical_volume": offset + 0.01 * call,
                    "prediction_reference_balance_component_scaled_rmse": (
                        offset + 0.001 * call
                    ),
                }
            )
    summary = aggregate_variant(trajectories, calls)
    assert summary["completion_rate"] == 0.5
    assert summary["common_endpoint_call"] == 2
    assert summary["failure_cause_counts"] == {
        "completed": 1,
        "inadmissible_state": 1,
    }
    assert summary["completed_case_mean_final_relative_l2"] == 0.03
