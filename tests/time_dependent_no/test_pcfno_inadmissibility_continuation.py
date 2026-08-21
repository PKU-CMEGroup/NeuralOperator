from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.time_dependent_no.evaluate_pcfno_inadmissibility_continuation import (
    NUM_STEPS,
    SCHEMA,
    conservative_fields,
    error_decomposition,
    failure_location,
    inadmissibility_episodes,
    invalid_node_mask,
    weighted_scaled_relative_l2,
)
from scripts.time_dependent_no.visualize_pcfno_inadmissibility_continuation import (
    animation_fields,
    field_scales,
    load_case,
    render_animation,
)


def _admissible_state(node_count: int) -> np.ndarray:
    state = np.zeros((node_count, 4), dtype=np.float32)
    state[:, 0] = 1.0
    state[:, 3] = 2.5
    return state


def test_admissibility_uses_internal_energy_density() -> None:
    state = _admissible_state(3)
    state[1] = np.asarray([1.0, 4.0, 0.0, 1.0], dtype=np.float32)
    fields = conservative_fields(state)
    invalid = invalid_node_mask(state)
    location = failure_location(state, np.asarray([0, 1, 0]), gamma=1.4)

    assert fields["internal_energy"][1] == -7.0
    assert invalid.tolist() == [False, True, False]
    assert location == {
        "node": 1,
        "node_type": 1,
        "node_type_name": "wall",
        "quantity": "internal_energy",
        "value": -7.0,
    }


def test_inadmissibility_episodes_separate_temporary_and_durable_recovery() -> None:
    temporary = inadmissibility_episodes(
        [3, 4, 5, 6], [True, False, True, True], terminal_call=6
    )
    durable = inadmissibility_episodes(
        [3, 4, 5, 6, NUM_STEPS],
        [True, False, False, False, False],
        terminal_call=NUM_STEPS,
    )

    assert temporary[0]["recovery_call"] == 4
    assert temporary[0]["durable_through_h79"] is False
    assert temporary[1]["recovered"] is False
    assert durable == [
        {
            "start_call": 3,
            "end_call": 3,
            "duration_calls": 1,
            "recovered": True,
            "recovery_call": 4,
            "durable_through_h79": True,
        }
    ]


def test_failure_decomposition_closes_and_distinguishes_sources() -> None:
    target = np.ones((4, 4), dtype=np.float32)
    teacher = target.copy()
    teacher[:, 0] += 0.25
    failed = teacher.copy()
    failed[:, 3] += 1.0
    result = error_decomposition(
        failed,
        teacher,
        target,
        np.ones(4),
        np.ones(4),
        mask=np.asarray([True, True, False, False]),
    )

    assert result["fresh_defect_norm"] == 0.125
    assert result["propagated_response_norm"] == 0.5
    assert result["total_norm"] == np.sqrt(0.125**2 + 0.5**2)
    assert result["reconstruction_residual_norm"] == 0.0
    assert result["fresh_propagated_cosine"] == 0.0


def test_weighted_relative_l2_respects_mask() -> None:
    target = np.ones((2, 4), dtype=np.float32)
    prediction = target.copy()
    prediction[1] += 100.0
    assert (
        weighted_scaled_relative_l2(
            prediction,
            target,
            np.ones(2),
            np.ones(4),
            mask=np.asarray([True, False]),
        )
        == 0.0
    )


def test_visual_bundle_load_and_field_shapes(tmp_path: Path) -> None:
    node_count = 4
    reference = np.repeat(_admissible_state(node_count)[None], NUM_STEPS + 1, axis=0)
    prediction = reference.copy()
    prediction[:, :, 3] += np.linspace(0.0, 0.5, NUM_STEPS + 1)[:, None]
    path = tmp_path / "trajectory_54.npz"
    np.savez_compressed(
        path,
        schema=np.asarray(SCHEMA),
        trajectory_key=np.asarray("54"),
        reference_states_conservative=reference,
        deployed_states_conservative=prediction,
        positions=np.asarray(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            dtype=np.float32,
        ),
        node_type=np.asarray([0, 1, 2, 3]),
        physical_times=np.arange(NUM_STEPS + 1, dtype=np.float64),
        failure_call=np.asarray(3, dtype=np.int64),
        last_retained_call=np.asarray(NUM_STEPS, dtype=np.int64),
    )

    arrays = load_case(path)
    fields = animation_fields(arrays)
    scales = field_scales(fields)

    assert fields["reference_pressure"].shape == (NUM_STEPS + 1, node_count)
    assert fields["true_pressure_residual"].shape == (NUM_STEPS, node_count)
    assert scales["pressure"][0] < scales["pressure"][1]
    assert scales["residual_error"][0] < 0.0 < scales["residual_error"][1]


def test_animation_smoke_writes_two_fixed_scale_frames(tmp_path: Path) -> None:
    node_count = 4
    reference = np.repeat(_admissible_state(node_count)[None], NUM_STEPS + 1, axis=0)
    prediction = reference.copy()
    prediction[:, :, 3] += np.linspace(0.0, 0.5, NUM_STEPS + 1)[:, None]
    prediction[2, 0] = np.asarray([1.0, 4.0, 0.0, 1.0])
    arrays = {
        "trajectory_key": np.asarray("54"),
        "reference_states_conservative": reference,
        "deployed_states_conservative": prediction,
        "positions": np.asarray(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            dtype=np.float32,
        ),
        "failure_call": np.asarray(2, dtype=np.int64),
        "last_retained_call": np.asarray(2, dtype=np.int64),
        "physical_times": np.arange(NUM_STEPS + 1, dtype=np.float64),
    }
    output = tmp_path / "smoke.gif"

    payload = render_animation(arrays, output, fps=2, dpi=50)

    assert output.is_file() and output.stat().st_size > 0
    assert payload["frames"] == 2
    assert payload["rendering"].startswith("all graph nodes")
