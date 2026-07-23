import numpy as np
import pytest
import torch

from scripts.time_dependent_no import diagnose_euler1d_stride_attribution as diagnostic
from utility.time_dependent_no.euler1d_data import (
    Euler1DNPZ,
    primitive_to_conservative_np,
)


def _curve(case_id, stride, frame, admissible):
    return {
        "case_id": case_id,
        "stride": stride,
        "frame": frame,
        "raw_state_admissible": admissible,
    }


def _descriptor(case_id, value):
    return {
        "case_id": case_id,
        **{name: value for name in diagnostic.DESCRIPTOR_FIELDS},
    }


def test_select_cohort_orders_failures_and_matches_without_outcome_fields():
    rows = []
    for case_id in (10, 20, 30, 40, 50):
        for stride in diagnostic.REQUIRED_STRIDES:
            rows.append(_curve(case_id, stride, 96, True))
    rows.extend(
        (
            _curve(10, 1, 64, False),
            _curve(20, 1, 56, False),
        )
    )
    descriptors = [
        _descriptor(10, 0.0),
        _descriptor(20, 10.0),
        _descriptor(30, 0.1),
        _descriptor(40, 9.9),
        _descriptor(50, 50.0),
    ]

    cohort = diagnostic.select_cohort(
        rows,
        descriptors,
        max_failures=4,
        max_controls=2,
    )

    assert [row["case_id"] for row in cohort["failures"]] == [20, 10]
    assert [row["case_id"] for row in cohort["controls"]] == [40, 30]
    assert [row["matched_failure_case_id"] for row in cohort["controls"]] == [
        20,
        10,
    ]
    assert cohort["descriptor_fields"] == list(diagnostic.DESCRIPTOR_FIELDS)


def test_event_start_frames_are_common_and_strictly_pre_failure():
    assert diagnostic.event_start_frames(64, lookback_macros=2) == (40, 48, 56)
    assert diagnostic.event_start_frames(71, lookback_macros=2) == (48, 56, 64)


def test_decomposition_metrics_preserve_vector_identity_and_alignment():
    first = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    second = np.array([[-0.25, 0.0, 0.0], [-0.5, 0.0, 0.0]])
    total = first + second

    metrics = diagnostic.decomposition_metrics(
        total,
        first,
        second,
        np.ones(3),
    )

    assert metrics["relative_reconstruction_residual"] == pytest.approx(0.0)
    assert metrics["relative_energy_identity_residual"] == pytest.approx(0.0)
    assert metrics["first_second_cosine"] == pytest.approx(-1.0)
    assert (
        metrics["first_energy_fraction_of_total"]
        + metrics["second_energy_fraction_of_total"]
        + metrics["cross_energy_fraction_of_total"]
    ) == pytest.approx(1.0)


def _source(cells=8, frames=9):
    data = np.ones((1, frames, cells, 3), dtype=np.float32)
    data[..., 1] = 0.0
    return Euler1DNPZ(
        data=data,
        x=np.linspace(0.0, 1.0, cells, dtype=np.float32)[None],
        t=np.linspace(0.0, 0.08, frames, dtype=np.float32)[None],
        left_states=np.array([[1.0, 0.0, 1.0]], dtype=np.float32),
        right_states=np.array([[1.0, 0.0, 1.0]], dtype=np.float32),
        gamma=1.4,
        metadata={},
    )


def test_apply_learned_path_counts_calls_and_stops_on_raw_failure(monkeypatch):
    source = _source()
    calls = []

    def fake_predict(
        _source,
        _case_ids,
        primitive,
        start_frame,
        stride,
        _model,
        _adapter,
        _device,
        *,
        conservative=None,
    ):
        calls.append((start_frame, stride))
        proposed = np.asarray(primitive).copy()
        proposed[..., 0] += 0.1
        if start_frame == 4:
            proposed[..., 2] = -0.1
        return proposed, primitive_to_conservative_np(proposed, source.gamma)

    monkeypatch.setattr(diagnostic, "_predict_model_batch_state", fake_predict)
    primitive = source.data[0, 0].astype(np.float64)
    conservative = primitive_to_conservative_np(primitive, source.gamma)
    result = diagnostic.apply_learned_path(
        source,
        0,
        primitive,
        conservative,
        0,
        8,
        2,
        object(),
        object(),
        torch.device("cpu"),
    )

    assert calls == [(0, 2), (2, 2), (4, 2)]
    assert result["completed"] is False
    assert result["failure_frame"] == 6
    assert result["failure_cause"] == "nonpositive_raw_state"
    assert result["calls_completed"] == 2
