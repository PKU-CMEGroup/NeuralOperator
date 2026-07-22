import json

import numpy as np
import pytest

from scripts.time_dependent_no import (
    evaluate_euler1d_flow_map_frontier as flow_map_frontier,
)
from scripts.time_dependent_no.benchmark_euler1d_flow_map_runtime import (
    _rollout_callbacks,
    _write_csv as write_runtime_csv,
    build_accuracy_matched_rows,
    measure_synchronized_rollout,
    prolong_piecewise_constant,
    select_accuracy_matched_reference,
)
from scripts.time_dependent_no.generate_euler1d_flow_map_ood import (
    OOD_REGIMES,
    TRAINING_SUPPORT,
    merge_regime_files,
)
from scripts.time_dependent_no.evaluate_euler1d_flow_map_frontier import (
    build_composition_paths,
    fixed_norm_conservative_perturbation,
    fixed_scale_perturbation_amplification,
    last_common_frame,
    learned_path_pareto_rows,
    normalized_amplification_rate,
    run_call_path,
    semigroup_error_bounds_hold,
    truth_wave_audit,
)
from utility.time_dependent_no.euler1d_data import Euler1DNPZ


def test_composition_paths_have_exact_common_endpoint_call_counts():
    stride4 = build_composition_paths(target_stride=4, horizon=32)
    stride8 = build_composition_paths(target_stride=8, horizon=32)

    assert stride4 == {
        "s4_direct": (4,) * 8,
        "s4_from_s2": (2,) * 16,
        "s4_from_s1": (1,) * 32,
    }
    assert stride8 == {
        "s8_direct": (8,) * 4,
        "s8_from_s4": (4,) * 8,
        "s8_from_s2": (2,) * 16,
        "s8_from_s1": (1,) * 32,
    }


def test_checkpoint_epoch_prefers_current_trainer_field_with_legacy_fallback():
    assert flow_map_frontier._checkpoint_epoch({"checkpoint_epoch": 58}) == 58
    assert flow_map_frontier._checkpoint_epoch({"epoch": 47}) == 47
    assert (
        flow_map_frontier._checkpoint_epoch({"checkpoint_epoch": 58, "epoch": 999})
        == 58
    )


def test_call_paths_reach_same_truth_endpoint():
    initial = np.array([0.0])

    def advance(state, stride, _frame):
        return state + stride

    for path in build_composition_paths(8, 32).values():
        rollout = run_call_path(initial, path, advance)
        assert rollout.completed_frame == 32
        assert rollout.completed is True
        np.testing.assert_allclose(rollout.states[32], np.array([32.0]))


def test_last_common_frame_uses_only_states_both_paths_reached():
    initial = np.array([0.0])

    direct = run_call_path(
        initial,
        (4, 4),
        lambda state, stride, _frame: state + stride,
    )

    def fail_after_frame_four(state, stride, frame):
        if frame >= 4:
            return np.array([np.nan])
        return state + stride

    composed = run_call_path(initial, (2, 2, 2, 2), fail_after_frame_four)

    assert composed.completed is False
    assert composed.completed_frame == 4
    assert last_common_frame(direct, composed, requested_frame=8) == 4


def test_semigroup_defect_is_interpreted_beside_truth_errors():
    truth = np.array([1.0, 2.0])
    direct = np.array([1.1, 2.0])
    composed = np.array([0.9, 2.2])

    defect = np.linalg.norm(direct - composed)
    direct_error = np.linalg.norm(direct - truth)
    composed_error = np.linalg.norm(composed - truth)

    assert semigroup_error_bounds_hold(defect, direct_error, composed_error)
    assert not semigroup_error_bounds_hold(
        defect + direct_error + composed_error,
        direct_error,
        composed_error,
    )


def test_amplification_rate_is_normalized_per_physical_time():
    assert normalized_amplification_rate(4.0, 0.5) == pytest.approx(2.0 * np.log(4.0))
    with pytest.raises(ValueError, match="positive"):
        normalized_amplification_rate(1.0, 0.0)


def test_fixed_conservative_perturbation_has_requested_norm_and_is_admissible():
    primitive = np.ones((16, 3), dtype=np.float64)
    primitive[:, 1] = np.linspace(-0.2, 0.2, 16)
    perturbed, realized = fixed_norm_conservative_perturbation(
        primitive,
        gamma=1.4,
        relative_norm=1.0e-4,
        rng=np.random.default_rng(7),
    )

    assert realized == pytest.approx(1.0e-4, rel=1.0e-10)
    assert np.all(perturbed[:, 0] > 0.0)
    assert np.all(perturbed[:, 2] > 0.0)


def test_fixed_scale_amplification_recovers_linear_factor():
    primitive = np.ones((8, 3), dtype=np.float64)
    primitive[:, 1] = 0.1
    perturbed, _ = fixed_norm_conservative_perturbation(
        primitive,
        gamma=1.4,
        relative_norm=1.0e-5,
        rng=np.random.default_rng(11),
    )
    factor = 2.5
    output_base = primitive
    output_perturbed = primitive + factor * (perturbed - primitive)

    metrics = fixed_scale_perturbation_amplification(
        primitive,
        perturbed,
        output_base,
        output_perturbed,
        gamma=1.4,
        physical_dt=0.25,
    )

    assert metrics["amplification"] == pytest.approx(factor, rel=1.0e-4)
    assert metrics["log_amplification_per_physical_time"] == pytest.approx(
        np.log(factor) / 0.25,
        rel=1.0e-4,
    )


def test_trajectory_increment_error_is_zero_for_the_stored_update():
    previous = np.ones((8, 3), dtype=np.float64)
    previous[:, 1] = 0.1
    current = previous.copy()
    current[:, 0] += 0.02

    metrics = flow_map_frontier.trajectory_increment_errors(
        previous,
        current,
        previous,
        current,
        gamma=1.4,
    )

    assert metrics["trajectory_increment_relative_l2"] == pytest.approx(0.0)
    assert metrics["trajectory_increment_state_normalized_l2"] == pytest.approx(0.0)


def test_error_budget_survival_distinguishes_error_invalidity_and_censoring():
    def row(case_id, frame, error, *, completed=True, reason=None):
        return {
            "checkpoint_set_label": "candidate_epoch_50",
            "path": "s8_direct",
            "stride": 8,
            "case_id": case_id,
            "frame": frame,
            "physical_elapsed_time": frame * 0.01,
            "completed_step": completed,
            "termination_reason": reason,
            "fixed_scale_conservative_relative_l2": error,
        }

    rows = [
        row(0, 8, 0.02),
        row(0, 16, 0.08),
        row(1, 8, 0.01),
        row(
            1,
            16,
            np.nan,
            completed=False,
            reason="nonpositive_raw_state",
        ),
        row(2, 8, 0.01),
        row(2, 16, 0.02),
    ]

    survival = flow_map_frontier.error_budget_survival_rows(
        rows,
        [0.05],
        max_horizon=16,
    )
    by_case = {entry["case_id"]: entry for entry in survival}

    assert by_case[0]["event_kind"] == "error_budget_exceeded"
    assert by_case[0]["event_frame"] == 16
    assert by_case[1]["event_kind"] == "nonpositive_raw_state"
    assert by_case[1]["raw_admissibility_failure_frame"] == 16
    assert by_case[2]["right_censored"] is True
    assert by_case[2]["completed_max_horizon"] is True


def test_direct_curve_audit_rolls_each_stride_once_and_reuses_endpoints(monkeypatch):
    cases = 2
    cells = 32
    frames = 9
    data = np.ones((cases, frames, cells, 3), dtype=np.float32)
    data[..., 1] = 0.0
    for frame in range(frames):
        data[:, frame, :, 0] += 0.01 * frame
    source = Euler1DNPZ(
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
    calls = []

    def exact_advance(
        _source,
        case_ids,
        primitive,
        start_frame,
        stride,
        _model,
        _adapter,
        _device,
        *,
        conservative=None,
    ):
        calls.append((tuple(case_ids.tolist()), start_frame, stride))
        result = primitive.copy()
        result[..., 0] += 0.01 * stride
        return (
            result,
            flow_map_frontier.primitive_to_conservative_np(result, _source.gamma),
        )

    monkeypatch.setattr(
        flow_map_frontier,
        "_predict_model_batch_state",
        exact_advance,
    )
    rows = flow_map_frontier.evaluate_direct_curves(
        source,
        np.array([0, 1], dtype=np.int64),
        {2: (object(), object())},
        8,
        flow_map_frontier.torch.device("cpu"),
        checkpoint_set_label="primary_selected",
    )
    summary = flow_map_frontier.summarize_direct_curves(
        rows,
        [8],
        num_cases=2,
    )

    assert len(calls) == 4
    assert len(rows) == 8
    assert summary[0]["num_completed"] == 2
    assert summary[0]["fixed_scale_conservative_relative_l2_mean"] < 1.0e-6
    assert summary[0]["trajectory_increment_relative_l2_mean"] < 1.0e-5


def test_learned_pareto_uses_common_endpoint_call_accounting():
    rows = learned_path_pareto_rows(
        [
            {
                "path": "s8_from_s2",
                "horizon": 32,
                "call_stride": 2,
                "num_calls": 16,
            }
        ],
        [
            {
                "stride": 2,
                "batch_size": 1,
                "device_resident_median_ms": 1.5,
                "host_to_host_median_ms": 2.0,
            }
        ],
        np.linspace(0.0, 0.5, 101),
    )

    assert rows[0]["physical_horizon"] == pytest.approx(0.16)
    assert rows[0]["estimated_device_resident_wall_ms"] == pytest.approx(24.0)
    assert rows[0]["estimated_host_to_host_wall_ms"] == pytest.approx(32.0)


def test_reference_prolongation_is_piecewise_constant():
    coarse = np.arange(12, dtype=np.float64).reshape(4, 3)
    prolonged = prolong_piecewise_constant(coarse, target_cells=8)

    assert prolonged.shape == (8, 3)
    np.testing.assert_array_equal(prolonged[0], coarse[0])
    np.testing.assert_array_equal(prolonged[1], coarse[0])
    np.testing.assert_array_equal(prolonged[-1], coarse[-1])


def test_accuracy_match_selects_fastest_no_worse_reference():
    candidates = [
        {
            "completion_fraction": 1.0,
            "fixed_scale_conservative_relative_l2_mean": 0.08,
            "batch1_wall_seconds_median": 0.6,
        },
        {
            "completion_fraction": 1.0,
            "fixed_scale_conservative_relative_l2_mean": 0.04,
            "batch1_wall_seconds_median": 1.2,
        },
        {
            "completion_fraction": 1.0,
            "fixed_scale_conservative_relative_l2_mean": 0.12,
            "batch1_wall_seconds_median": 0.2,
        },
    ]

    matched = select_accuracy_matched_reference(0.10, candidates)

    assert matched is candidates[0]
    assert select_accuracy_matched_reference(0.03, candidates) is None


def test_accuracy_matched_rows_require_completion_for_claim_eligibility():
    learned_paths = [
        {
            "case_id": 3,
            "path": "s4_from_s2",
            "horizon": 32,
            "call_stride": 2,
            "num_calls": 16,
            "completed_horizon": True,
            "fixed_scale_conservative_relative_l2": 0.08,
        },
        {
            "case_id": 5,
            "path": "s4_from_s2",
            "horizon": 32,
            "call_stride": 2,
            "num_calls": 16,
            "completed_horizon": False,
            "fixed_scale_conservative_relative_l2": np.nan,
        },
    ]
    timing = [
        {
            "stride": 2,
            "batch_size": 1,
            "device_resident_median_ms": 1.0,
            "host_to_host_median_ms": 2.0,
        }
    ]
    references = [
        {
            "resolution": 64,
            "horizon": 32,
            "completion_fraction": 1.0,
            "fixed_scale_conservative_relative_l2_mean": 0.07,
            "batch1_wall_seconds_median": 0.5,
        }
    ]

    rows = build_accuracy_matched_rows(
        learned_paths,
        timing,
        references,
        np.array([3, 5], dtype=np.int64),
    )

    assert rows[0]["reference_resolution"] is None
    assert rows[0]["learned_host_to_host_wall_seconds"] == pytest.approx(0.032)
    assert rows[0]["learned_timing_basis"] == "legacy_per_call_estimate"
    assert rows[0]["case_coverage_complete"] is True
    assert rows[0]["learned_completion_fraction"] == pytest.approx(0.5)
    assert rows[0]["claim_eligible"] is False


def test_accuracy_matched_rows_reject_subset_case_coverage_for_matching():
    learned_paths = [
        {
            "case_id": 3,
            "path": "s8_direct",
            "horizon": 32,
            "call_stride": 8,
            "num_calls": 4,
            "completed_horizon": True,
            "fixed_scale_conservative_relative_l2": 0.08,
        }
    ]
    references = [
        {
            "resolution": 64,
            "horizon": 32,
            "completion_fraction": 1.0,
            "fixed_scale_conservative_relative_l2_mean": 0.07,
            "batch1_wall_seconds_median": 0.5,
        }
    ]

    rows = build_accuracy_matched_rows(
        learned_paths,
        [],
        references,
        np.array([3, 5], dtype=np.int64),
    )

    assert rows[0]["case_coverage_complete"] is False
    assert rows[0]["num_matched_cases"] == 1
    assert rows[0]["num_expected_cases"] == 2
    assert rows[0]["reference_match_available"] is False
    assert rows[0]["claim_eligible"] is False


def test_accuracy_matched_rows_reject_duplicate_case_rows():
    row = {
        "case_id": 3,
        "path": "s8_direct",
        "horizon": 32,
        "call_stride": 8,
        "num_calls": 4,
        "completed_horizon": True,
        "fixed_scale_conservative_relative_l2": 0.08,
    }

    with pytest.raises(ValueError, match="duplicate learned path rows"):
        build_accuracy_matched_rows(
            [row, dict(row)],
            [],
            [],
            np.array([3], dtype=np.int64),
        )


def test_accuracy_matched_rows_use_direct_full_rollout_measurements():
    learned_paths = [
        {
            "case_id": 3,
            "path": "s8_direct",
            "horizon": 32,
            "call_stride": 8,
            "num_calls": 4,
            "completed_horizon": True,
            "fixed_scale_conservative_relative_l2": 0.08,
        }
    ]
    timing = [
        {
            "stride": 8,
            "horizon": 32,
            "batch_size": 1,
            "measurement_boundary": "device_resident",
            "measurement_scope": "direct_full_rollout",
            "full_rollout_median_ms": 4.0,
        },
        {
            "stride": 8,
            "horizon": 32,
            "batch_size": 1,
            "measurement_boundary": "host_to_host",
            "measurement_scope": "direct_full_rollout",
            "full_rollout_median_ms": 6.0,
        },
        {
            "stride": 8,
            "horizon": 32,
            "batch_size": 16,
            "measurement_boundary": "device_resident",
            "measurement_scope": "direct_full_rollout",
            "full_rollout_median_ms": 8.0,
            "cases_per_second": 2000.0,
        },
        {
            "stride": 8,
            "horizon": 32,
            "batch_size": 16,
            "measurement_boundary": "host_to_host",
            "measurement_scope": "direct_full_rollout",
            "full_rollout_median_ms": 10.0,
            "cases_per_second": 1600.0,
        },
    ]
    references = [
        {
            "resolution": 64,
            "horizon": 32,
            "completion_fraction": 1.0,
            "fixed_scale_conservative_relative_l2_mean": 0.07,
            "batch1_wall_seconds_median": 0.5,
        }
    ]
    reference_throughput = [
        {
            "resolution": 64,
            "horizon": 32,
            "cases_per_second": 12.0,
        }
    ]

    rows = build_accuracy_matched_rows(
        learned_paths,
        timing,
        references,
        np.array([3], dtype=np.int64),
        reference_throughput,
    )

    assert rows[0]["learned_device_resident_wall_seconds"] == pytest.approx(0.004)
    assert rows[0]["learned_host_to_host_wall_seconds"] == pytest.approx(0.006)
    assert rows[0]["learned_timing_basis"] == "direct_synchronized_full_rollout"
    assert rows[0]["learned_device_resident_amortized_cases_per_second"] == 2000.0
    assert rows[0]["reference_amortized_cases_per_second"] == 12.0
    assert rows[0]["claim_eligible"] is True


def test_runtime_csv_writer_truncates_stale_output_when_rows_are_empty(tmp_path):
    output_path = tmp_path / "accuracy_matched_pareto.csv"
    output_path.write_text("stale\n", encoding="utf-8")

    write_runtime_csv(output_path, [])

    assert output_path.read_text(encoding="utf-8") == ""


def test_full_rollout_timer_counts_warmup_and_repeats():
    calls = []

    timing = measure_synchronized_rollout(
        lambda: calls.append(None),
        warmup=2,
        repeats=3,
        device=flow_map_frontier.torch.device("cpu"),
    )

    assert len(calls) == 5
    assert timing["full_rollout_median_ms"] >= 0.0


def test_rollout_callback_retains_decoded_conservative_state():
    cells = 16
    frames = 5
    data = np.ones((1, frames, cells, 3), dtype=np.float32)
    data[..., 1] = 0.0
    source = Euler1DNPZ(
        data=data,
        x=np.linspace(0.0, 1.0, cells, dtype=np.float32)[None],
        t=np.linspace(0.0, 0.04, frames, dtype=np.float32)[None],
        left_states=np.array([[1.0, 0.0, 1.0]], dtype=np.float32),
        right_states=np.array([[1.0, 0.0, 1.0]], dtype=np.float32),
        gamma=1.4,
        metadata={},
    )

    class Recorder(flow_map_frontier.torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.seen = []

        def forward(self, batch):
            self.seen.append(batch.current_conservative.detach().clone())
            return batch

    class Adapter(flow_map_frontier.torch.nn.Module):
        def forward(self, _raw, batch):
            decoded = type("Decoded", (), {})()
            decoded.primitive = batch.current_primitive
            decoded.conservative = batch.current_conservative + 1.0
            return decoded

    model = Recorder()
    callback = _rollout_callbacks(
        source,
        np.array([0], dtype=np.int64),
        stride=2,
        horizon=4,
        model=model,
        adapter=Adapter(),
        device=flow_map_frontier.torch.device("cpu"),
    )["device_resident"]
    callback()

    assert len(model.seen) == 2
    np.testing.assert_allclose(
        model.seen[1].numpy(),
        (model.seen[0] + 1.0).numpy(),
    )


def test_local_semigroup_uses_one_large_call_and_exact_compositions(monkeypatch):
    cells = 4
    frames = 9
    data = np.ones((1, frames, cells, 3), dtype=np.float32)
    for frame in range(frames):
        data[0, frame, :, 0] += 0.01 * frame
    source = Euler1DNPZ(
        data=data,
        x=np.linspace(0.0, 1.0, cells, dtype=np.float32)[None],
        t=np.linspace(0.0, 0.08, frames, dtype=np.float32)[None],
        left_states=np.array([[1.0, 0.0, 1.0]], dtype=np.float32),
        right_states=np.array([[1.0, 0.0, 1.0]], dtype=np.float32),
        gamma=1.4,
        metadata={},
    )

    def exact_advance(
        _source,
        _case_ids,
        primitive,
        _start_frame,
        stride,
        _model,
        _adapter,
        _device,
        *,
        conservative=None,
    ):
        result = primitive.copy()
        result[..., 0] += 0.01 * stride
        return (
            result,
            flow_map_frontier.primitive_to_conservative_np(result, _source.gamma),
        )

    monkeypatch.setattr(
        flow_map_frontier,
        "_predict_model_batch_state",
        exact_advance,
    )
    rows = flow_map_frontier.evaluate_local_semigroup(
        source,
        np.array([0], dtype=np.int64),
        {1: (object(), object()), 2: (object(), object()), 4: (object(), object())},
        [0],
        flow_map_frontier.torch.device("cpu"),
    )

    stride4 = [row for row in rows if row["target_stride"] == 4]
    assert {row["composed_path"] for row in stride4} == {
        "s4_from_s1",
        "s4_from_s2",
    }
    assert all(row["both_completed_target"] for row in stride4)
    assert all(
        row["semigroup_defect_at_common"] == pytest.approx(0.0) for row in stride4
    )
    assert all(row["direct_truth_error_at_target"] < 1.0e-7 for row in stride4)


def test_ood_compatibility_requires_identical_grid_and_saved_times():
    data = np.ones((1, 3, 4, 3), dtype=np.float32)
    source = Euler1DNPZ(
        data=data,
        x=np.linspace(0.125, 0.875, 4, dtype=np.float32)[None],
        t=np.array([[0.0, 0.01, 0.02]], dtype=np.float32),
        left_states=np.array([[1.0, 0.0, 1.0]], dtype=np.float32),
        right_states=np.array([[1.0, 0.0, 1.0]], dtype=np.float32),
        gamma=1.4,
        metadata={},
    )
    flow_map_frontier._validate_ood_compatibility(source, source)
    shifted_times = Euler1DNPZ(
        data=data,
        x=source.x,
        t=np.array([[0.0, 0.02, 0.04]], dtype=np.float32),
        left_states=source.left_states,
        right_states=source.right_states,
        gamma=source.gamma,
        metadata={},
    )

    with pytest.raises(ValueError, match="saved physical time"):
        flow_map_frontier._validate_ood_compatibility(source, shifted_times)


def test_fixed_ood_groups_cross_only_the_declared_support_boundaries():
    high = OOD_REGIMES["high_inflow"]
    low = OOD_REGIMES["low_ambient"]

    assert high.left_rho_range[0] == TRAINING_SUPPORT.left_rho_range[1]
    assert high.left_u_range[0] == TRAINING_SUPPORT.left_u_range[1]
    assert high.left_p_range[0] == TRAINING_SUPPORT.left_p_range[1]
    assert high.right_rho_range == TRAINING_SUPPORT.right_rho_range
    assert high.right_p_range == TRAINING_SUPPORT.right_p_range
    assert low.right_rho_range[1] == TRAINING_SUPPORT.right_rho_range[0]
    assert low.right_p_range[1] == TRAINING_SUPPORT.right_p_range[0]
    assert low.left_rho_range == TRAINING_SUPPORT.left_rho_range
    assert low.left_u_range == TRAINING_SUPPORT.left_u_range
    assert low.left_p_range == TRAINING_SUPPORT.left_p_range


def test_ood_regime_merge_records_labels_and_contract(tmp_path):
    paths = []
    for regime_id in range(2):
        path = tmp_path / f"regime_{regime_id}.npz"
        np.savez_compressed(
            path,
            data=np.full((2, 1, 1, 3), regime_id, dtype=np.float32),
            n_cases=np.array(2, dtype=np.int32),
            method=np.array("test_solver"),
        )
        paths.append(path)
    output = tmp_path / "merged.npz"

    merge_regime_files(
        paths,
        ["high_inflow", "low_ambient"],
        output,
        cases_per_regime=2,
        seed=7,
    )

    with np.load(output, allow_pickle=False) as arrays:
        assert arrays["data"].shape == (4, 1, 1, 3)
        assert arrays["n_cases"].item() == 4
        assert arrays["ood_regime"].tolist() == [
            "high_inflow",
            "high_inflow",
            "low_ambient",
            "low_ambient",
        ]
        contract = json.loads(arrays["ood_contract_json"].item())
        assert contract["label"] == "mild_support_extrapolation_v1"


def test_ood_metadata_is_validated_and_attached_per_case():
    cases = 4
    source = Euler1DNPZ(
        data=np.ones((cases, 2, 4, 3), dtype=np.float32),
        x=np.broadcast_to(
            np.linspace(0.0, 1.0, 4, dtype=np.float32), (cases, 4)
        ).copy(),
        t=np.broadcast_to(np.array([0.0, 0.01], dtype=np.float32), (cases, 2)).copy(),
        left_states=np.ones((cases, 3), dtype=np.float32),
        right_states=np.ones((cases, 3), dtype=np.float32),
        gamma=1.4,
        metadata={
            "ood_contract_json": json.dumps(
                {
                    "label": "mild_support_extrapolation_v1",
                    "cases_per_regime": 2,
                    "regimes": {"high_inflow": {}, "low_ambient": {}},
                }
            ),
            "ood_regime": np.array(
                ["high_inflow", "high_inflow", "low_ambient", "low_ambient"]
            ),
            "ood_generator_source_sha256": "a" * 64,
        },
    )

    contract, labels, generator_hash, counts = (
        flow_map_frontier._validated_ood_metadata(
            source, "mild_support_extrapolation_v1"
        )
    )
    rows = [{"case_id": 0}, {"case_id": 3}]
    flow_map_frontier._attach_ood_regimes(rows, labels)

    assert contract["cases_per_regime"] == 2
    assert generator_hash == "a" * 64
    assert counts == {"high_inflow": 2, "low_ambient": 2}
    assert [row["ood_regime"] for row in rows] == ["high_inflow", "low_ambient"]
    with pytest.raises(ValueError, match="does not match"):
        flow_map_frontier._validated_ood_metadata(source, "wrong_suite")


def test_state_total_mismatch_metric_is_not_claimed_as_flux_closure():
    truth = np.tile(
        np.array([[1.0, 0.0, 1.0]], dtype=np.float64),
        (32, 1),
    )
    prediction = truth.copy()
    prediction[:, 0] += 0.1
    initial = np.full_like(truth, 7.0)

    metrics = flow_map_frontier._state_metrics(
        prediction,
        truth,
        initial,
        np.linspace(0.0, 1.0, 32, dtype=np.float64),
        1.4,
    )

    assert metrics["global_conserved_total_mismatch_relative_l2"] > 0.0
    assert metrics["conservative_budget_relative_l2"] == pytest.approx(
        metrics["global_conserved_total_mismatch_relative_l2"]
    )


def test_ood_path_summaries_preserve_regime_separation():
    rows = []
    for case_id, (regime, error) in enumerate(
        (("high_inflow", 0.08), ("low_ambient", 0.03))
    ):
        rows.append(
            {
                "case_id": case_id,
                "ood_regime": regime,
                "path": "s8_direct",
                "horizon": 96,
                "target_stride": 8,
                "call_stride": 8,
                "num_calls": 12,
                "completed_horizon": True,
                "primitive_relative_l2": error,
                "fixed_scale_conservative_relative_l2": error,
                "shock_top2_position_mae": error,
            }
        )

    summaries = flow_map_frontier._summarize_paths_by_ood_regime(rows)

    assert {row["ood_regime"] for row in summaries} == {
        "high_inflow",
        "low_ambient",
    }
    means = {
        row["ood_regime"]: row["fixed_scale_conservative_relative_l2_mean"]
        for row in summaries
    }
    assert means == {
        "high_inflow": pytest.approx(0.08),
        "low_ambient": pytest.approx(0.03),
    }


def test_frontier_csv_writer_truncates_stale_output_when_rows_are_empty(tmp_path):
    output_path = tmp_path / "summary_by_ood_regime.csv"
    output_path.write_text("stale\n", encoding="utf-8")

    flow_map_frontier._write_csv(output_path, [])

    assert output_path.read_text(encoding="utf-8") == ""


def test_truth_wave_audit_detects_active_interior_reflected_front():
    cells = 32
    frames = 17
    pressure = np.ones((1, frames, cells), dtype=np.float32)
    for frame in range(frames):
        face = min(24, 8 + frame)
        pressure[0, frame, face:] += 0.2 + 0.05 * frame
    data = np.zeros((1, frames, cells, 3), dtype=np.float32)
    data[..., 0] = 1.0
    data[..., 2] = pressure
    source = Euler1DNPZ(
        data=data,
        x=np.linspace(0.0, 1.0, cells, dtype=np.float32)[None],
        t=np.linspace(0.0, 0.16, frames, dtype=np.float32)[None],
        left_states=np.array([[1.0, 0.0, 1.0]], dtype=np.float32),
        right_states=np.array([[1.0, 0.0, 1.0]], dtype=np.float32),
        gamma=1.4,
        metadata={},
    )

    audit = truth_wave_audit(
        source,
        np.array([0], dtype=np.int64),
        horizons=[8, 16],
        boundary_margin_cells=4,
        min_peak_fraction=0.05,
    )

    assert audit["horizons"]["8"]["interior_active_fraction"] == 1.0
    assert audit["horizons"]["16"]["state_change_from_frame"] == 8
    assert audit["horizons"]["16"]["state_change_to_frame"] == 16
    assert audit["horizons"]["16"]["state_change_relative_l2_median"] > 0.0
