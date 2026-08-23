from __future__ import annotations

import math

import numpy as np
import pytest
import torch

import scripts.time_dependent_no.evaluate_pcno_bump_b1_c5_fixed_map as fixed_map
from scripts.time_dependent_no.analyze_pcno_bump_b1_c5_fixed_map import (
    classify_primary_effects,
    paired_floor_record,
)
from scripts.time_dependent_no.evaluate_pcno_bump_b1_c5_fixed_map import (
    CHECKPOINT_ROLES,
    EXPECTED_STEPS,
    INPUT_VIEWS,
    TRAJECTORY_COUNTS,
    aggregate_map_rows,
    paired_map_metrics,
    select_fixed_map_descriptors,
)


def test_paired_map_metrics_closes_signed_quadratic_identity() -> None:
    target = torch.ones((1, 2, 4), dtype=torch.float32)
    selected = 2.0 * target
    candidate = 3.0 * target
    current = torch.zeros_like(target)
    metrics = paired_map_metrics(
        candidate_prediction=candidate,
        selected_prediction=selected,
        target=target,
        input_state=current,
        node_weights=torch.tensor([[[1.0], [3.0]]]),
        node_mask=torch.ones((1, 2, 1)),
        component_scale=torch.ones(4),
    )

    assert metrics["candidate_state_relative_l2"] == pytest.approx(2.0)
    assert metrics["selected_state_relative_l2"] == pytest.approx(1.0)
    assert metrics["candidate_minus_selected_relative_l2"] == pytest.approx(1.0)
    assert metrics["candidate_over_selected_relative_l2"] == pytest.approx(2.0)
    assert metrics["candidate_error_scaled_mse"] == pytest.approx(4.0)
    assert metrics["selected_error_scaled_mse"] == pytest.approx(1.0)
    assert metrics["drift_scaled_mse"] == pytest.approx(1.0)
    assert metrics["selected_error_drift_cross_scaled"] == pytest.approx(1.0)
    assert metrics["drift_over_selected_residual"] == pytest.approx(0.5)
    assert metrics["selected_error_drift_cosine"] == pytest.approx(1.0)
    assert metrics["quadratic_closure_relative"] <= 1.0e-15


def test_paired_map_metrics_rejects_nonfinite_or_drifted_geometry() -> None:
    state = torch.ones((1, 2, 4))
    kwargs = {
        "candidate_prediction": state,
        "selected_prediction": state,
        "target": state,
        "input_state": state,
        "node_weights": torch.ones((1, 2, 1)),
        "node_mask": torch.ones((1, 2, 1)),
        "component_scale": torch.ones(4),
    }
    with pytest.raises(ValueError, match="finite tensors"):
        paired_map_metrics(
            **{**kwargs, "candidate_prediction": state.clone().fill_(math.nan)}
        )
    with pytest.raises(ValueError, match="component scale"):
        paired_map_metrics(**{**kwargs, "component_scale": torch.ones(3)})
    with pytest.raises(ValueError, match="nonnegative"):
        paired_map_metrics(**{**kwargs, "node_weights": -torch.ones((1, 2, 1))})


def test_select_fixed_map_descriptors_requires_exact_steps_and_shared_contracts() -> (
    None
):
    descriptors = []
    for count in TRAJECTORY_COUNTS:
        for role in CHECKPOINT_ROLES:
            descriptors.append(
                {
                    "trajectory_count": count,
                    "checkpoint_role": role,
                    "optimizer_step": EXPECTED_STEPS[role],
                    "normalization_digest": f"normalizer-{count}",
                    "config_digest": f"config-{count}",
                }
            )
    selected = select_fixed_map_descriptors(descriptors)
    assert set(selected) == set(TRAJECTORY_COUNTS)
    assert all(set(by_role) == set(CHECKPOINT_ROLES) for by_role in selected.values())

    drifted = [dict(descriptor) for descriptor in descriptors]
    drifted[0]["optimizer_step"] += 1
    with pytest.raises(ValueError, match="step changed"):
        select_fixed_map_descriptors(drifted)

    drifted = [dict(descriptor) for descriptor in descriptors]
    drifted[1]["normalization_digest"] = "different"
    with pytest.raises(ValueError, match="do not share"):
        select_fixed_map_descriptors(drifted)


def _synthetic_row(*, count: int, role: str, view: str, call: int) -> dict[str, object]:
    effect = (call / 1000.0) if role == "terminal" else 0.0
    return {
        "trajectory_count": count,
        "checkpoint_role": role,
        "input_view": view,
        "call_index": call,
        "candidate_state_relative_l2": 1.0 + effect,
        "selected_state_relative_l2": 1.0,
        "candidate_minus_selected_relative_l2": effect,
        "candidate_over_selected_relative_l2": 1.0 + effect,
        "candidate_error_scaled_mse": 1.0 + effect,
        "selected_error_scaled_mse": 1.0,
        "candidate_minus_selected_scaled_mse": effect,
        "drift_scaled_mse": effect**2,
        "drift_scaled_rms": abs(effect),
        "drift_over_selected_residual": abs(effect),
        "selected_error_drift_cross_scaled": effect,
        "selected_error_drift_cosine": 1.0 if effect else None,
        "quadratic_closure_relative": 0.0,
        "candidate_admissibility": {"all_admissible": True},
        "selected_admissibility": {"all_admissible": True},
    }


def test_aggregate_map_rows_preserves_endpoint_and_window_mean() -> None:
    rows = [
        _synthetic_row(count=count, role=role, view=view, call=call)
        for count in TRAJECTORY_COUNTS
        for role in CHECKPOINT_ROLES
        for view in INPUT_VIEWS
        for call in range(1, 80)
    ]
    aggregates = aggregate_map_rows(rows)
    endpoint = aggregates["128"]["terminal"]["common_selected_path"]["endpoints"]["20"]
    assert endpoint["candidate_minus_selected_relative_l2_mean"] == pytest.approx(0.02)
    assert endpoint["pooled_drift_scaled_rms"] == pytest.approx(0.02)
    early = aggregates["128"]["terminal"]["common_selected_path"]["windows"][
        "early_1_20"
    ]
    assert early["candidate_minus_selected_relative_l2_mean"] == pytest.approx(
        sum(range(1, 21)) / 20_000.0
    )


def test_paired_floor_and_primary_classification_follow_registered_rule() -> None:
    resolved = paired_floor_record((-0.10, -0.102), expected_sign=-1)
    assert resolved["resolved"] is True
    assert resolved["matches_expected_direction"] is True

    noisy = paired_floor_record((-0.10, -0.20), expected_sign=-1)
    assert noisy["floor_pass"] is False
    assert noisy["resolved"] is False

    records = []
    for count in TRAJECTORY_COUNTS:
        for call in (20, 79):
            records.append(
                {
                    "trajectory_count": count,
                    "call_index": call,
                    "resolved": True,
                    "matches_expected_direction": True,
                }
            )
    classification = classify_primary_effects(records)
    assert classification["classification"] == (
        "resolved_common_input_functional_map_crossover"
    )

    records[-1]["matches_expected_direction"] = False
    classification = classify_primary_effects(records)
    assert classification["classification"] == (
        "resolved_common_input_response_does_not_match_full_crossover"
    )

    records[-1]["resolved"] = False
    classification = classify_primary_effects(records)
    assert classification["classification"] == "numerically_unresolved"


class _SyntheticStore:
    def __init__(self) -> None:
        base = np.zeros((80, 2, 4), dtype=np.float32)
        base[..., 0] = 1.0
        base[..., 3] = 2.0
        for frame in range(80):
            base[frame, :, 0] += 0.2 * frame
            base[frame, :, 3] += 0.2 * frame
        self._states = base

    def states(self, key: str) -> np.ndarray:
        assert key == "synthetic"
        return self._states

    def tensor_sample(
        self, key: str, time_index: int, *, step_stride: int, device: torch.device
    ) -> dict[str, torch.Tensor]:
        assert key == "synthetic"
        assert time_index == 0
        assert step_stride == 1
        return {
            "current": torch.as_tensor(
                self._states[0:1], dtype=torch.float32, device=device
            ),
            "target": torch.as_tensor(
                self._states[1:2], dtype=torch.float32, device=device
            ),
            "node_weights": torch.ones((1, 2, 1), device=device),
            "node_mask": torch.ones((1, 2, 1), device=device),
        }


class _AdditiveMap:
    def __init__(self, increment: float) -> None:
        self.increment = increment
        self.state_scale = torch.ones(4)
        self.gamma = 1.4


def test_evaluate_count_uses_one_frozen_selected_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    increments = {
        "matched_20480": 0.05,
        "selected": 0.10,
        "terminal": 0.30,
    }

    def build_model(
        checkpoint: dict[str, object], device: torch.device
    ) -> _AdditiveMap:
        assert device.type == "cpu"
        return _AdditiveMap(float(checkpoint["increment"]))

    def forward(
        model: _AdditiveMap,
        sample: dict[str, torch.Tensor],
        current: torch.Tensor,
        *,
        boundary_policy: dict[str, object],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del sample, boundary_policy
        increment = torch.zeros_like(current)
        increment[..., (0, 3)] = model.increment
        prediction = current + increment
        return prediction, prediction, current

    monkeypatch.setattr(fixed_map, "build_bump_checkpoint_model", build_model)
    monkeypatch.setattr(fixed_map, "contract_forward_sample", forward)
    descriptors = {
        role: {"checkpoint_sha256": f"sha-{role}"} for role in CHECKPOINT_ROLES
    }
    checkpoints = {
        role: {"increment": increment} for role, increment in increments.items()
    }
    rows, record = fixed_map._evaluate_count(
        trajectory_count=128,
        descriptors=descriptors,
        checkpoints=checkpoints,
        store=_SyntheticStore(),
        outside_keys=["synthetic"],
        policies={"synthetic": {}},
        device=torch.device("cpu"),
        execution_id="fp32_1",
    )

    assert len(rows) == len(CHECKPOINT_ROLES) * len(INPUT_VIEWS) * 79
    assert record["trajectories"][0]["selected_path_recurrence_exact"] is True
    selected_common = next(
        row
        for row in rows
        if row["checkpoint_role"] == "selected"
        and row["input_view"] == "common_selected_path"
        and row["call_index"] == 20
    )
    assert selected_common["drift_scaled_rms"] == pytest.approx(0.0)
    terminal_exact = next(
        row
        for row in rows
        if row["checkpoint_role"] == "terminal"
        and row["input_view"] == "exact_reference"
        and row["call_index"] == 20
    )
    terminal_common = next(
        row
        for row in rows
        if row["checkpoint_role"] == "terminal"
        and row["input_view"] == "common_selected_path"
        and row["call_index"] == 20
    )
    assert terminal_exact["drift_scaled_rms"] == pytest.approx(
        0.2 / math.sqrt(2.0), rel=2.0e-6
    )
    assert terminal_common["drift_scaled_rms"] == pytest.approx(
        terminal_exact["drift_scaled_rms"]
    )
