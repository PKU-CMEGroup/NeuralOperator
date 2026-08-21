from __future__ import annotations

import numpy as np
from PIL import Image

from scripts.time_dependent_no.animate_pcno_bump_gradient_ablation import (
    render_rollout_animation,
)


def _state(pressure: np.ndarray) -> np.ndarray:
    rho = np.ones_like(pressure)
    zeros = np.zeros_like(pressure)
    return np.stack((rho, zeros, zeros, pressure / 0.4), axis=-1).astype(np.float32)


def _artifact(*, fail_call: int | None) -> dict[str, np.ndarray]:
    nodes = 10
    calls = 4
    x = np.linspace(0.0, 1.0, nodes, dtype=np.float32)
    positions = np.column_stack((x, 0.1 * np.sin(2.0 * np.pi * x))).astype(
        np.float32
    )
    initial = _state(np.ones(nodes, dtype=np.float32))
    targets = np.stack(
        [_state(1.0 + 0.02 * (call + 1) * x) for call in range(calls)]
    )
    valid_length = calls if fail_call is None else fail_call - 1
    artifact = {
        "trajectory_key": np.asarray("54"),
        "positions": positions,
        "node_type": np.zeros(nodes, dtype=np.int64),
        "initial_conservative": initial,
        "reference_targets_conservative": targets,
        "pcno_baseline_predictions_conservative": targets[:valid_length].copy(),
        "physical_target_times": np.arange(1, calls + 1, dtype=np.float64) * 0.025,
        "physical_delta_t": np.asarray(0.025),
        "baseline_valid_length": np.asarray(valid_length),
        "baseline_failure_call": np.asarray(-1 if fail_call is None else fail_call),
        "baseline_failure_cause": np.asarray(
            "completed" if fail_call is None else "inadmissible_state"
        ),
    }
    if fail_call is not None:
        proposal = targets[fail_call - 1].copy()
        proposal[3, 3] = -0.5
        artifact["baseline_failed_proposal"] = proposal
    return artifact


def test_animation_marks_rejected_proposal_without_imputing_later_states(
    tmp_path,
) -> None:
    output = tmp_path / "failure.gif"
    record = render_rollout_animation(
        {"full": _artifact(fail_call=None), "no_gradient": _artifact(fail_call=3)},
        output,
        seed=20260718,
        trajectory="54",
        fps=2,
        dpi=45,
    )
    assert output.is_file() and output.stat().st_size > 0
    with Image.open(output) as image:
        assert "transparency" not in image.info
    assert record["frames"] == 4
    assert record["failure"]["full"] is None
    assert record["failure"]["no_gradient"]["call"] == 3
    assert record["failure"]["no_gradient"]["quantity"] == "internal_energy"
    assert record["claim_boundary"]["rejected_proposal_is_not_recurred"] is True
    assert record["claim_boundary"]["missing_post_failure_states_are_not_imputed"] is True
