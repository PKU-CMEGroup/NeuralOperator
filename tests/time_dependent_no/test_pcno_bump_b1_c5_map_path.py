from __future__ import annotations

import math

import numpy as np
import pytest
import torch

import scripts.time_dependent_no.evaluate_pcno_bump_b1_c5_map_path as map_path
from scripts.time_dependent_no.analyze_pcno_bump_b1_c5_fixed_map import (
    paired_floor_record,
)
from scripts.time_dependent_no.analyze_pcno_bump_b1_c5_map_path import (
    classify_mechanism,
)
from scripts.time_dependent_no.evaluate_pcno_bump_b1_c5_fixed_map import (
    TRAJECTORY_COUNTS,
)
from scripts.time_dependent_no.evaluate_pcno_bump_b1_c5_map_path import (
    OUTPUT_LABELS,
    aggregate_map_path_rows,
    four_way_map_path_metrics,
)


def _metric_kwargs() -> dict[str, torch.Tensor]:
    return {
        "node_weights": torch.tensor([[[1.0], [3.0]]]),
        "node_mask": torch.ones((1, 2, 1)),
        "component_scale": torch.ones(4),
    }


def test_four_way_metrics_close_scalar_and_output_identities() -> None:
    target = torch.ones((1, 2, 4), dtype=torch.float32)
    metrics = four_way_map_path_metrics(
        predictions={
            "ss": 2.0 * target,
            "ts": 1.5 * target,
            "st": 3.0 * target,
            "tt": 4.0 * target,
        },
        selected_input=torch.zeros_like(target),
        terminal_input=0.25 * target,
        target=target,
        **_metric_kwargs(),
    )

    assert metrics["error_ss"] == pytest.approx(1.0)
    assert metrics["error_ts"] == pytest.approx(0.5)
    assert metrics["error_st"] == pytest.approx(2.0)
    assert metrics["error_tt"] == pytest.approx(3.0)
    assert metrics["map_effect_selected_path"] == pytest.approx(-0.5)
    assert metrics["path_effect_selected_map"] == pytest.approx(1.0)
    assert metrics["map_path_interaction"] == pytest.approx(1.5)
    assert metrics["autonomous_total_effect"] == pytest.approx(2.0)
    assert metrics["zero_interaction_counterfactual_effect"] == pytest.approx(0.5)
    assert metrics["output_map_selected_scaled_rms"] == pytest.approx(0.5)
    assert metrics["output_path_selected_scaled_rms"] == pytest.approx(1.0)
    assert metrics["output_interaction_scaled_rms"] == pytest.approx(1.5)
    assert metrics["output_total_scaled_rms"] == pytest.approx(2.0)
    assert metrics["maximum_scalar_closure_relative"] <= 1.0e-15
    assert metrics["output_closure_relative"] <= 1.0e-15


def test_four_way_metrics_reject_nonfinite_or_incomplete_square() -> None:
    state = torch.ones((1, 2, 4))
    kwargs = {
        "predictions": {label: state for label in OUTPUT_LABELS},
        "selected_input": state,
        "terminal_input": state,
        "target": state,
        **_metric_kwargs(),
    }
    with pytest.raises(ValueError, match="ss, ts, st, and tt"):
        four_way_map_path_metrics(
            **{**kwargs, "predictions": {"ss": state, "tt": state}}
        )
    drifted = dict(kwargs["predictions"])
    drifted["ts"] = state.clone().fill_(math.nan)
    with pytest.raises(ValueError, match="finite tensors"):
        four_way_map_path_metrics(**{**kwargs, "predictions": drifted})
    with pytest.raises(ValueError, match="component scale"):
        four_way_map_path_metrics(**{**kwargs, "component_scale": torch.ones(3)})


def _synthetic_row(*, count: int, call: int) -> dict[str, object]:
    base = call / 1000.0
    row: dict[str, object] = {
        "trajectory_count": count,
        "call_index": call,
        "maximum_scalar_closure_relative": 0.0,
        "output_closure_relative": 0.0,
        "admissibility": {label: {"all_admissible": True} for label in OUTPUT_LABELS},
    }
    for label in OUTPUT_LABELS:
        row[f"error_{label}"] = base
    for field in map_path.AGGREGATE_FIELDS:
        row.setdefault(field, base)
    return row


def test_aggregate_map_path_rows_preserves_endpoint_and_window_means() -> None:
    rows = [
        _synthetic_row(count=count, call=call)
        for count in TRAJECTORY_COUNTS
        for call in range(1, 80)
    ]
    aggregates = aggregate_map_path_rows(rows)
    assert aggregates["128"]["endpoints"]["20"][
        "map_effect_selected_path_mean"
    ] == pytest.approx(0.02)
    assert aggregates["256"]["windows"]["early_1_20"][
        "autonomous_total_effect_mean"
    ] == pytest.approx(sum(range(1, 21)) / 20_000.0)


def _classification_records(
    *, counter_128: tuple[float, float], counter_256: tuple[float, float]
) -> dict[int, dict[str, dict[str, object]]]:
    records = {}
    for count, counter in zip(TRAJECTORY_COUNTS, (counter_128, counter_256)):
        records[count] = {
            "map_effect_selected_path": {
                "floor": paired_floor_record((-0.10, -0.101), expected_sign=-1)
            },
            "autonomous_total_effect": {
                "floor": paired_floor_record((0.20, 0.202), expected_sign=1)
            },
            "zero_interaction_counterfactual_effect": {
                "floor": paired_floor_record(counter)
            },
            "map_path_interaction": {"floor": paired_floor_record((0.25, 0.252))},
        }
    return records


def test_mechanism_classification_follows_registered_sufficiency_tree() -> None:
    path = classify_mechanism(
        _classification_records(counter_128=(0.05, 0.051), counter_256=(0.04, 0.041))
    )
    assert path["classification"] == "resolved_path_displacement_sufficient"

    interaction = classify_mechanism(
        _classification_records(
            counter_128=(-0.05, -0.051), counter_256=(-0.04, -0.041)
        )
    )
    assert interaction["classification"] == "resolved_map_path_interaction_required"

    mixed = classify_mechanism(
        _classification_records(counter_128=(0.05, 0.051), counter_256=(-0.04, -0.041))
    )
    assert mixed["classification"] == "mixed_count_mechanism"

    unresolved_records = _classification_records(
        counter_128=(0.05, 0.10), counter_256=(0.04, 0.041)
    )
    assert classify_mechanism(unresolved_records)["classification"] == (
        "numerically_unresolved"
    )

    parent_mismatch = _classification_records(
        counter_128=(0.05, 0.051), counter_256=(0.04, 0.041)
    )
    parent_mismatch[256]["autonomous_total_effect"] = {
        "floor": paired_floor_record((-0.20, -0.202), expected_sign=1)
    }
    assert classify_mechanism(parent_mismatch)["classification"] == (
        "parent_crossover_not_reproduced"
    )


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
        self.state_scale = torch.ones((1, 1, 4))
        self.gamma = 1.4


def test_evaluate_count_advances_only_owner_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    increments = {"selected": 0.10, "terminal": 0.30}

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

    monkeypatch.setattr(map_path, "build_bump_checkpoint_model", build_model)
    monkeypatch.setattr(map_path, "contract_forward_sample", forward)
    descriptors = {
        role: {"checkpoint_sha256": f"sha-{role}"} for role in map_path.CHECKPOINT_ROLES
    }
    checkpoints = {
        role: {"increment": increment} for role, increment in increments.items()
    }
    rows, record = map_path._evaluate_count(
        trajectory_count=128,
        descriptors=descriptors,
        checkpoints=checkpoints,
        store=_SyntheticStore(),
        outside_keys=["synthetic"],
        policies={"synthetic": {}},
        device=torch.device("cpu"),
        execution_id="fp32_1",
    )

    assert len(rows) == 79
    trajectory = record["trajectories"][0]
    assert trajectory["call1_path_inputs_equal"] is True
    assert trajectory["selected_path_recurrence_exact"] is True
    assert trajectory["terminal_path_recurrence_exact"] is True
    call1 = rows[0]
    assert call1["path_effect_selected_map"] == pytest.approx(0.0)
    assert call1["map_path_interaction"] == pytest.approx(0.0)
    call20 = rows[19]
    assert call20["output_input_path_scaled_rms"] == pytest.approx(
        3.8 / math.sqrt(2.0), rel=2.0e-6
    )
    assert call20["output_map_selected_scaled_rms"] == pytest.approx(
        0.2 / math.sqrt(2.0), rel=2.0e-6
    )
    assert call20["replay"]["ss"]["replay_drift_scaled_mse"] == pytest.approx(0.0)
    assert call20["replay"]["tt"]["replay_drift_scaled_mse"] == pytest.approx(0.0)
