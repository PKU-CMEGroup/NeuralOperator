from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import torch

from scripts.time_dependent_no import (
    diagnose_euler1d_flow_map_solver_consistency as diagnostic,
)
from utility.time_dependent_no.euler1d_data import (
    Euler1DNPZ,
    conservative_to_primitive_np,
    primitive_to_conservative_np,
)


def _source(cases=1, cells=32, frames=9):
    data = np.ones((cases, frames, cells, 3), dtype=np.float32)
    data[..., 1] = 0.0
    for frame in range(frames):
        data[:, frame, :, 0] += 0.01 * frame
    return Euler1DNPZ(
        data=data,
        x=np.broadcast_to(
            np.linspace(0.0, 1.0, cells, dtype=np.float32),
            (cases, cells),
        ).copy(),
        t=np.broadcast_to(
            np.linspace(0.0, 0.08, frames, dtype=np.float32),
            (cases, frames),
        ).copy(),
        left_states=np.array([[1.0, 0.0, 1.0]] * cases, dtype=np.float32),
        right_states=np.array([[1.0, 0.0, 1.0]] * cases, dtype=np.float32),
        gamma=1.4,
        metadata={},
    )


def test_diagnostic_requires_fresh_output_directory(tmp_path):
    output_dir = tmp_path / "diagnostic"
    diagnostic.require_fresh_output_dir(output_dir)
    output_dir.mkdir()
    diagnostic.require_fresh_output_dir(output_dir)
    (output_dir / "stale.csv").write_text("stale\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="stale artifacts"):
        diagnostic.require_fresh_output_dir(output_dir)


def test_diagnostic_main_exits_nonzero_when_truth_gate_fails(monkeypatch):
    monkeypatch.setattr(diagnostic, "parse_args", lambda _argv: object())
    monkeypatch.setattr(
        diagnostic,
        "run",
        lambda _args: {"status": "truth_replay_gate_failed"},
    )

    with pytest.raises(SystemExit, match="truth_replay_gate_failed"):
        diagnostic.main([])


def test_reference_macro_replays_each_saved_interval_and_aggregates(monkeypatch):
    calls = []

    def fake_advance(conservative, _left, _dx, interval_dt, config):
        calls.append(interval_dt)
        updated = conservative.copy()
        updated[:, 0] += interval_dt
        primitive = conservative_to_primitive_np(updated, config.gamma)
        return (
            primitive,
            updated,
            {
                "substeps": 2,
                "retry_halvings": 1,
                "fallback_steps": 0,
            },
        )

    monkeypatch.setattr(
        diagnostic,
        "advance_reference_conservative",
        fake_advance,
    )
    primitive = np.ones((8, 3), dtype=np.float64)
    primitive[:, 1] = 0.0
    result, _conservative, stats = diagnostic.advance_reference_macro(
        primitive,
        np.array([1.0, 0.0, 1.0]),
        0.1,
        [0.01, 0.02, 0.03],
        diagnostic.SolverReplayConfig(gamma=1.4, cfl=0.35),
    )

    assert calls == [0.01, 0.02, 0.03]
    assert stats == {
        "substeps": 6,
        "retry_halvings": 3,
        "fallback_steps": 0,
        "saved_intervals": 3,
    }
    assert np.all(result[:, 0] > primitive[:, 0])


def test_frozen_checkpoint_contract_rejects_coordinate_drift():
    args = {
        "model": "fno",
        "target": "residual",
        "fno_width": 64,
        "fno_modes": 24,
        "fno_layers": 4,
        "input_coordinates": "primitive",
        "loss_coordinates": "conservative",
        "recurrent_coordinates": "conservative",
        "input_normalization": "fixed_physical",
        "loss_normalization": "fixed_physical",
        "step_stride": 4,
    }
    checkpoint = {
        "args": args,
        "test_cases": np.array([3, 7]),
        "data_sha256": "data",
        "saved_time_sha256": "times",
    }

    with pytest.raises(ValueError, match="frozen contract"):
        diagnostic._validate_checkpoint_contract(
            checkpoint,
            stride=4,
            expected_cases=np.array([3, 7]),
            data_sha256="data",
            saved_time_sha256="times",
            split="test",
        )


def test_truth_replay_gate_accepts_bounded_outlier_but_rejects_bulk_drift():
    bounded = diagnostic.truth_replay_gate(
        [1.0e-7] * 1000 + [2.1e-5],
        max_tolerance=5.0e-5,
        mean_tolerance=1.0e-6,
        p99_tolerance=1.0e-6,
    )
    diffuse = diagnostic.truth_replay_gate(
        [2.0e-6] * 1001,
        max_tolerance=5.0e-5,
        mean_tolerance=1.0e-6,
        p99_tolerance=1.0e-6,
    )

    assert bounded["passed"] is True
    assert diffuse["passed"] is False
    assert diffuse["max"] < diffuse["max_tolerance"]


def test_oracle_rescue_replaces_exactly_one_call(monkeypatch):
    source = _source()
    model_calls = []
    reference_calls = []

    def exact_model(
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
        model_calls.append((start_frame, stride))
        result = primitive.copy()
        result[..., 0] += 0.01 * stride
        return (
            result,
            primitive_to_conservative_np(result, _source.gamma),
        )

    def exact_reference(
        _source,
        _case_ids,
        primitive,
        start_frame,
        stride,
        _config,
        _executor,
    ):
        reference_calls.append((start_frame, stride))
        result = primitive.copy()
        result[..., 0] += 0.01 * stride
        return [
            (
                result[position],
                primitive_to_conservative_np(result[position], source.gamma),
                {
                    "substeps": stride,
                    "retry_halvings": 0,
                    "fallback_steps": 0,
                    "saved_intervals": stride,
                },
            )
            for position in range(result.shape[0])
        ]

    monkeypatch.setattr(diagnostic, "_predict_model_batch_state", exact_model)
    monkeypatch.setattr(diagnostic, "_reference_batch", exact_reference)
    with ThreadPoolExecutor(max_workers=1) as executor:
        rows = diagnostic._rollout_variant(
            source,
            np.array([0], dtype=np.int64),
            2,
            object(),
            object(),
            8,
            4,
            diagnostic.SolverReplayConfig(gamma=1.4, cfl=0.35),
            executor,
            torch.device("cpu"),
        )

    assert model_calls == [(0, 2), (2, 2), (6, 2)]
    assert reference_calls == [(4, 2)]
    assert [row["call_kind"] for row in rows] == [
        "learned_map",
        "learned_map",
        "oracle_reference_macro",
        "learned_map",
    ]
    assert all(row["completed_step"] for row in rows)


def test_paired_rescue_comparison_separates_completion_and_error():
    def endpoint(path, case_id, error):
        return {
            "stride": 8,
            "path": path,
            "case_id": case_id,
            "frame": 96,
            "completed_step": True,
            "fixed_scale_conservative_relative_l2": error,
        }

    rows = [
        endpoint("s8_raw", 0, 0.08),
        endpoint("s8_oracle_rescue_f32", 0, 0.04),
        endpoint("s8_oracle_rescue_f32", 1, 0.03),
    ]
    comparison = diagnostic._paired_rescue_comparisons(
        rows,
        horizon=96,
        num_cases=2,
    )[0]

    assert comparison["completion_fraction_delta"] == pytest.approx(0.5)
    assert comparison["num_paired_completed"] == 1
    assert comparison["paired_rescue_to_raw_error_ratio_mean"] == pytest.approx(0.5)
