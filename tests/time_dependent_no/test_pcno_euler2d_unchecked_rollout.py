from __future__ import annotations

import numpy as np
import pytest
import torch

from scripts.time_dependent_no.diagnose_pcno_euler2d_unchecked_rollout import (
    corner_geometry_summary,
    raw_frame_metrics,
    raw_inadmissible_node_mask,
    unchecked_rollout,
)


def test_raw_frame_metrics_reports_inadmissibility_without_modifying_state() -> None:
    state = np.asarray(
        [
            [1.0, 0.0, 0.0, 2.5],
            [1.0, 3.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    original = state.copy()
    metrics = raw_frame_metrics(state, gamma=1.4)

    np.testing.assert_array_equal(state, original)
    assert metrics["nonpositive_density_node_count"] == 1
    assert metrics["nonpositive_internal_energy_node_count"] == 1
    assert metrics["nonpositive_pressure_node_count"] == 1
    assert metrics["inadmissible_node_count"] == 2
    assert metrics["all_conservative_finite"] is True
    np.testing.assert_array_equal(
        raw_inadmissible_node_mask(state, gamma=1.4),
        np.asarray([False, True, True]),
    )


def test_unchecked_rollout_continues_after_negative_internal_energy() -> None:
    current = torch.tensor([[[1.0, 0.0, 0.0, 0.2]]], dtype=torch.float32)
    sample = {"current": current}

    def step(_model, _sample, value):
        proposal = value.clone()
        proposal[..., 3] -= 0.15
        return proposal

    result = unchecked_rollout(
        None,
        sample,
        num_steps=4,
        device=torch.device("cpu"),
        call_fn=step,
    )

    assert result["predictions"].shape == (4, 1, 4)
    assert result["predictions"][1, 0, 3] < 0.0
    assert result["predictions"][3, 0, 3] == pytest.approx(-0.4)


def test_unchecked_rollout_enforces_replay_prefix() -> None:
    current = torch.zeros((1, 1, 4), dtype=torch.float32)
    sample = {"current": current}

    def step(_model, _sample, value):
        return value + 1.0

    replay = np.stack(
        (
            np.ones((1, 4), dtype=np.float32),
            np.full((1, 4), 3.0, dtype=np.float32),
        )
    )
    with pytest.raises(ValueError, match="replay mismatch at call 2"):
        unchecked_rollout(
            None,
            sample,
            num_steps=3,
            device=torch.device("cpu"),
            replay_prefix=replay,
            call_fn=step,
        )


def test_corner_geometry_summary_uses_exact_directed_stencil() -> None:
    positions = np.asarray(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]],
        dtype=np.float64,
    )
    node_type = np.asarray([3, 3, 1, 1], dtype=np.int64)
    node_weights = np.full((4, 1), 0.25, dtype=np.float64)
    directed = np.asarray(
        [
            [0, 1],
            [0, 3],
            [1, 0],
            [1, 2],
            [2, 1],
            [2, 3],
            [3, 0],
            [3, 2],
        ],
        dtype=np.int64,
    )
    gradients = np.tile(np.asarray([[1.0, 0.0], [0.0, 1.0]]), (4, 1))

    summary = corner_geometry_summary(
        positions, node_type, node_weights, directed, gradients
    )

    assert summary["index"] == 1
    assert summary["node_type_name"] == "inflow"
    assert summary["directed_stencil_degree"] == 2
    assert {row["index"] for row in summary["neighbors"]} == {0, 2}
