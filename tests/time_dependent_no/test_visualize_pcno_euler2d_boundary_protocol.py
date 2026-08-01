from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.time_dependent_no.visualize_pcno_euler2d_boundary_protocol import (
    animation_frame_calls,
    assert_capture_matches_summary,
    render_boundary_protocol_animation,
    select_visual_cases,
)
from utility.time_dependent_no.euler2d import primitive_to_conservative


def _rollout_row(
    trajectory: str,
    *,
    valid_length: int,
    horizon_error: float | None,
    failure_cause: str = "completed",
) -> dict[str, object]:
    endpoints = {"2": 0.02}
    if horizon_error is not None:
        endpoints["5"] = horizon_error
    return {
        "trajectory": trajectory,
        "valid_length": valid_length,
        "completed": valid_length == 5,
        "failure_cause": failure_cause,
        "endpoint_relative_l2": endpoints,
    }


def test_select_visual_cases_uses_predeclared_rules() -> None:
    native = [
        _rollout_row(
            "18",
            valid_length=2,
            horizon_error=None,
            failure_cause="nonpositive_internal_energy",
        ),
        _rollout_row(
            "128",
            valid_length=4,
            horizon_error=None,
            failure_cause="nonpositive_internal_energy",
        ),
        _rollout_row("7", valid_length=5, horizon_error=0.10),
        _rollout_row("16", valid_length=5, horizon_error=0.10),
        _rollout_row("23", valid_length=5, horizon_error=0.10),
        _rollout_row("47", valid_length=5, horizon_error=0.10),
    ]
    causal = [dict(row) for row in native]
    for row in causal:
        row["valid_length"] = 5
        row["completed"] = True
        row["failure_cause"] = "completed"
        row["endpoint_relative_l2"] = {"2": 0.03, "5": 0.12}
    minimum = [dict(row) for row in causal]
    minimum_errors = {"7": 0.11, "16": 0.09, "23": 0.08, "47": 0.07}
    for row in minimum:
        key = str(row["trajectory"])
        row["endpoint_relative_l2"] = {
            "2": 0.02,
            "5": minimum_errors.get(key, 0.08),
        }

    summary = {
        "evaluation": {"rollout_steps": 5},
        "native": {"rollout": {"trajectories": native}},
        "causal": {"rollout": {"trajectories": causal}},
        "minimum_change": {"rollout": {"trajectories": minimum}},
    }
    selected = select_visual_cases(summary)

    assert selected[0]["trajectory"] == "18"
    assert selected[0]["first_excluded_call"] == 3
    # Improvements are -10%, 10%, 20%, 30%; the upper-median rank is 20%.
    assert selected[1]["trajectory"] == "23"
    assert selected[1]["rank_zero_based"] == 2


def test_animation_frame_calls_retains_failure_transition() -> None:
    assert animation_frame_calls(
        rollout_steps=7,
        frame_stride=3,
        first_excluded_calls=(5,),
    ) == [0, 3, 4, 5, 6, 7]


def test_capture_parity_uses_replay_specific_failure_call() -> None:
    captured = {
        "completed": False,
        "valid_length": 3,
        "failure_cause": "nonpositive_internal_energy",
        "all_relative_l2": np.asarray([0.0, 0.01, 0.02, 0.03]),
    }
    expected = {
        "completed": False,
        "valid_length": 2,
        "failure_cause": "nonpositive_internal_energy",
        "endpoint_relative_l2": {"1": 0.0101, "2": 0.0201},
    }

    parity = assert_capture_matches_summary(captured, expected)

    assert parity["completion_match"] is True
    assert parity["valid_length_difference"] == 1
    assert parity["all_endpoint_values_within_diagnostic_tolerance"] is True

    captured["all_relative_l2"] = np.asarray([0.0, 0.02, 0.04, 0.06])
    parity = assert_capture_matches_summary(captured, expected)
    assert parity["all_endpoint_values_within_diagnostic_tolerance"] is False

    captured["completed"] = True
    captured["failure_cause"] = "completed"
    with pytest.raises(ValueError, match="completion"):
        assert_capture_matches_summary(captured, expected)


def _synthetic_states(num_calls: int, nodes: np.ndarray, offset: float) -> np.ndarray:
    states = []
    for call in range(num_calls):
        primitive = np.column_stack(
            (
                1.0 + 0.01 * nodes[:, 0],
                np.full(nodes.shape[0], 1.5),
                np.zeros(nodes.shape[0]),
                np.full(nodes.shape[0], 1.0 + 0.05 * call + offset * call),
            )
        )
        states.append(primitive_to_conservative(primitive).astype(np.float32))
    return np.stack(states)


def test_render_boundary_protocol_animation_marks_excluded_failure(
    tmp_path: Path,
) -> None:
    nodes = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.5, 0.5],
            [0.25, 0.5],
        ],
        dtype=np.float32,
    )
    node_type = np.asarray([1, 2, 1, 3, 0, 0], dtype=np.int64)
    reference = _synthetic_states(4, nodes, 0.0)
    native = _synthetic_states(2, nodes, 0.01)
    causal = _synthetic_states(4, nodes, 0.02)
    minimum = _synthetic_states(4, nodes, 0.005)

    def row(
        states: np.ndarray,
        *,
        valid_length: int,
        failure_cause: str,
        first_excluded_call: int | None,
    ) -> dict[str, object]:
        errors = np.linspace(0.0, 0.03, valid_length + 1)
        return {
            "states": states,
            "all_relative_l2": errors,
            "boundary_relative_l2": 1.2 * errors,
            "valid_length": valid_length,
            "completed": first_excluded_call is None,
            "failure_cause": failure_cause,
            "first_excluded_call": first_excluded_call,
        }

    output_path = tmp_path / "comparison.gif"
    snapshot_path = tmp_path / "comparison.png"
    metadata = render_boundary_protocol_animation(
        nodes=nodes,
        node_type=node_type,
        reference_states=reference,
        rollouts={
            "native": row(
                native,
                valid_length=1,
                failure_cause="nonpositive_internal_energy",
                first_excluded_call=2,
            ),
            "causal": row(
                causal,
                valid_length=3,
                failure_cause="completed",
                first_excluded_call=None,
            ),
            "minimum_change": row(
                minimum,
                valid_length=3,
                failure_cause="completed",
                first_excluded_call=None,
            ),
        },
        trajectory_key="18",
        role="stability_rescue",
        rollout_steps=3,
        step_stride=1,
        start_frame=0,
        amp="bf16",
        frame_stride=2,
        fps=2,
        dpi=60,
        error_quantile=1.0,
        output_path=output_path,
        snapshot_path=snapshot_path,
    )

    assert output_path.stat().st_size > 0
    assert snapshot_path.stat().st_size > 0
    assert metadata["rendered_calls"] == [0, 1, 2, 3]
    assert metadata["visual_node_subsampling"] is False
    assert metadata["failure_marker_semantics"].startswith("first excluded")
