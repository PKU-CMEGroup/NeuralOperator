import numpy as np
import pytest
import torch

from scripts.time_dependent_no import (
    visualize_euler1d_flow_map_frontier as visualization,
)
from utility.time_dependent_no.euler1d_data import (
    Euler1DNPZ,
    primitive_to_conservative_np,
)


def _synthetic_source(cases: int = 2, frames: int = 9, cells: int = 64) -> Euler1DNPZ:
    data = np.ones((cases, frames, cells, 3), dtype=np.float32)
    data[..., 1] = 0.1
    for frame in range(frames):
        data[:, frame, :, 0] += 0.01 * frame
        data[:, frame, :, 2] += 0.005 * frame
    x = np.broadcast_to(
        np.linspace(0.0, 1.0, cells, dtype=np.float32),
        (cases, cells),
    ).copy()
    t = np.broadcast_to(
        np.linspace(0.0, 0.08, frames, dtype=np.float32),
        (cases, frames),
    ).copy()
    return Euler1DNPZ(
        data=data,
        x=x,
        t=t,
        left_states=np.array([[1.0, 0.1, 1.0]] * cases, dtype=np.float32),
        right_states=np.array([[1.0, 0.1, 1.0]] * cases, dtype=np.float32),
        gamma=1.4,
        metadata={},
    )


def test_front_alignment_removes_a_known_two_front_translation():
    cells = 128
    truth = np.ones((cells, 3), dtype=np.float64)
    truth[:, 1] = 0.2
    truth[32:, 0] += 0.5
    truth[32:, 2] += 0.8
    truth[80:, 0] -= 0.2
    truth[80:, 2] -= 0.35
    prediction = np.empty_like(truth)
    prediction[:2] = truth[0]
    prediction[2:] = truth[:-2]

    metrics = visualization.front_aligned_metrics(
        prediction,
        truth,
        gamma=1.4,
        max_shift_cells=8,
    )

    assert metrics["front_alignment_requested_shift_cells"] == 2
    assert metrics["front_alignment_shift_cells"] == 2
    assert metrics["front_alignment_shift_clipped"] is False
    assert metrics["front_aligned_fixed_scale_conservative_relative_l2"] < 1.0e-12
    assert metrics["front_translation_removable_l2_fraction"] > 0.99


def test_front_alignment_oracle_never_increases_error():
    rng = np.random.default_rng(2401)
    truth = np.ones((96, 3), dtype=np.float64)
    truth[:, 1] = 0.2
    prediction = truth + 0.02 * rng.standard_normal(truth.shape)

    metrics = visualization.front_aligned_metrics(
        prediction,
        truth,
        gamma=1.4,
        max_shift_cells=6,
    )

    assert (
        metrics["front_aligned_fixed_scale_conservative_relative_l2"]
        <= metrics["front_alignment_unaligned_fixed_scale_conservative_relative_l2"]
    )
    assert metrics["front_translation_removable_l2_fraction"] >= 0.0


def test_frozen_rollout_bank_reaches_exact_common_endpoints(monkeypatch):
    source = _synthetic_source()

    def exact_advance(
        _source,
        case_ids,
        _primitive,
        start_frame,
        stride,
        _model,
        _adapter,
        _device,
        *,
        conservative=None,
    ):
        del conservative
        result = np.asarray(
            _source.data[case_ids, start_frame + stride],
            dtype=np.float64,
        )
        return result, primitive_to_conservative_np(result, _source.gamma)

    monkeypatch.setattr(visualization, "_predict_model_batch_state", exact_advance)
    common_frames, predictions, rows, terminations = visualization.run_frozen_rollouts(
        source,
        np.array([0, 1], dtype=np.int64),
        {stride: (object(), object()) for stride in (1, 2, 4, 8)},
        horizon=8,
        common_frame_step=8,
        device=torch.device("cpu"),
        shock_radius_cells=2,
        max_alignment_shift_cells=4,
    )

    np.testing.assert_array_equal(common_frames, np.array([0, 8]))
    assert all(values.shape == (2, 2, 64, 3) for values in predictions.values())
    assert all(row["completed_horizon"] for row in terminations)
    assert all(row["completed_frame"] for row in rows)
    assert max(row["fixed_scale_conservative_relative_l2"] for row in rows) < 1.0e-7


def test_curve_statistics_do_not_carry_failed_cases_forward():
    rows = [
        {
            "path": "s1_direct",
            "stride": 1,
            "case_id": 0,
            "frame": 1,
            "physical_elapsed_time": 0.01,
            "completed_step": True,
            "fixed_scale_conservative_relative_l2": 0.1,
            "time_mean_fixed_scale_conservative_relative_l2": 0.05,
            "smooth_region_relative_l2": 0.08,
            "shock_top2_position_mae": 0.01,
            "shock_top2_strength_relative_l1": 0.1,
        },
        {
            "path": "s1_direct",
            "stride": 1,
            "case_id": 1,
            "frame": 1,
            "physical_elapsed_time": 0.01,
            "completed_step": False,
        },
        {
            "path": "s1_direct",
            "stride": 1,
            "case_id": 0,
            "frame": 2,
            "physical_elapsed_time": 0.02,
            "completed_step": True,
            "fixed_scale_conservative_relative_l2": 0.2,
            "time_mean_fixed_scale_conservative_relative_l2": 0.1,
            "smooth_region_relative_l2": 0.15,
            "shock_top2_position_mae": 0.02,
            "shock_top2_strength_relative_l1": 0.2,
        },
    ]

    statistics = visualization.direct_curve_statistics(rows)

    assert statistics[0]["raw_completion_fraction"] == pytest.approx(0.5)
    assert statistics[1]["raw_completion_fraction"] == pytest.approx(0.5)
    assert statistics[1]["error_mean"] == pytest.approx(0.2)


def test_casewise_oracle_uses_only_common_valid_cases():
    rows = []
    errors = {
        0: {2: 0.10, 4: 0.08, 8: 0.09},
        1: {2: 0.12, 4: 0.11, 8: 0.07},
    }
    for case_id, by_stride in errors.items():
        for stride, error in by_stride.items():
            rows.append(
                {
                    "case_id": case_id,
                    "stride": stride,
                    "frame": 8,
                    "completed_step": True,
                    "fixed_scale_conservative_relative_l2": error,
                }
            )

    winners, oracle = visualization.casewise_frontier_rows(rows)

    fractions = {row["stride"]: row["winner_fraction"] for row in winners}
    assert fractions == {2: 0.0, 4: 0.5, 8: 0.5}
    assert oracle[0]["population_best_stride"] == 8
    assert oracle[0]["per_case_oracle_mean_error"] == pytest.approx(0.075)


def test_nearest_centroid_loo_recovers_separated_stride_classes():
    features = np.array(
        [
            [-2.2, -2.0],
            [-2.0, -2.1],
            [-1.8, -1.9],
            [-0.2, 0.0],
            [0.0, 0.1],
            [0.2, -0.1],
            [1.8, 2.0],
            [2.0, 2.1],
            [2.2, 1.9],
        ],
        dtype=np.float64,
    )
    labels = np.array([2, 2, 2, 4, 4, 4, 8, 8, 8], dtype=np.int64)

    predictions, baseline = visualization.nearest_centroid_loo(
        features,
        labels,
        tie_preference=8,
    )

    np.testing.assert_array_equal(predictions, labels)
    assert np.mean(baseline == labels) < 1.0


def test_case_artifact_writer_emits_animation_arrays_and_heatmap(tmp_path):
    source = _synthetic_source(cases=1)
    common_frames = np.array([0, 8], dtype=np.int64)
    exact = source.data[:, common_frames].copy()
    predictions = {stride: exact.copy() for stride in (1, 2, 4, 8)}
    terminations = [
        {
            "case_id": 0,
            "stride": stride,
            "requested_horizon": 8,
            "completed_horizon": True,
            "termination_frame": float("nan"),
            "termination_reason": None,
        }
        for stride in (1, 2, 4, 8)
    ]

    manifest = visualization.save_case_rollout_artifacts(
        source,
        case_id=0,
        case_position=0,
        common_frames=common_frames,
        all_predictions=predictions,
        termination_rows=terminations,
        output_dir=tmp_path,
        fps=1.0,
    )

    for key in (
        "npz",
        "gif",
        "final_frame_pdf",
        "final_frame_png",
        "pressure_error_spacetime_pdf",
        "pressure_error_spacetime_png",
    ):
        assert (tmp_path / manifest[key]).is_file()


def test_ripple_budget_analysis_tracks_first_exceedance_and_censoring():
    rows = []
    values = {
        "s4": {0: (1.2, 1.6), 1: (1.2, 1.3)},
        "s8": {0: (1.6, 1.7), 1: (1.2, 1.7)},
    }
    for model, cases in values.items():
        for case_id, case_values in cases.items():
            for frame, value in zip((8, 16), case_values, strict=True):
                rows.append(
                    {
                        "mode": "autoregressive",
                        "model": model,
                        "stride": 4 if model == "s4" else 8,
                        "case_id": case_id,
                        "target_frame": frame,
                        "proposal_valid": True,
                        "smooth_state_tv_ratio": value,
                    }
                )

    survival, summary = visualization.ripple_budget_analysis(
        rows,
        ("s4", "s8"),
        (1.5,),
        horizon=16,
        common_frame_step=8,
    )

    h16 = {row["model"]: row for row in survival if row["frame"] == 16}
    assert h16["s4"]["num_surviving"] == 1
    assert h16["s8"]["num_surviving"] == 0
    by_model = {row["model"]: row for row in summary}
    assert by_model["s4"]["median_right_censored"] is True
    assert by_model["s8"]["median_first_exceedance_frame"] == 12.0
    assert by_model["s8"]["restricted_mean_last_acceptable_frame"] == 4.0


def _synthetic_modal_rows():
    rows = []
    for stride, model in zip((1, 2, 4, 8), ("s1", "s2", "s4", "s8"), strict=True):
        for frame in (8, 16):
            for mode in range(1, 9):
                target_power = 1.0 / mode
                relative_error = 0.1 * stride + 0.01 * mode
                error_power = target_power * relative_error**2
                rows.append(
                    {
                        "evaluation_mode": "autoregressive_state",
                        "model": model,
                        "stride": stride,
                        "source_frame": frame - stride,
                        "target_frame": frame,
                        "mode_index": mode,
                        "num_samples": 64,
                        "truth_resolved": mode != 8,
                        "modal_prediction_power_mean": target_power,
                        "modal_target_power_mean": target_power,
                        "modal_error_power_mean": error_power,
                        "modal_relative_error_rms": relative_error,
                        "modal_error_energy_fraction": mode / 36.0,
                    }
                )
    return rows


def test_modal_heatmap_masks_truth_unresolved_ratios() -> None:
    modes, frames, matrix = visualization.modal_heatmap_matrix(
        _synthetic_modal_rows(),
        "s4",
        "autoregressive_state",
        (8, 16),
        "modal_relative_error_rms",
    )

    assert modes.tolist() == list(range(1, 9))
    assert frames.tolist() == [8, 16]
    assert np.isfinite(matrix[:-1]).all()
    assert np.isnan(matrix[-1]).all()


def test_modal_band_statistics_aggregate_power_before_ratio() -> None:
    rows = [
        {
            "evaluation_mode": "autoregressive_state",
            "model": "s4",
            "stride": 4,
            "target_frame": 8,
            "mode_index": mode,
            "num_samples": 64,
            "truth_resolved": True,
            "modal_prediction_power_mean": 1.0,
            "modal_target_power_mean": 1.0,
            "modal_error_power_mean": 0.25,
        }
        for mode in range(1, 6)
    ]

    summary = visualization.modal_band_statistics(
        rows,
        ("s4",),
        "autoregressive_state",
        (8,),
        saved_frame_dt=0.005,
    )

    low = next(row for row in summary if row["band"] == "low_1_4")
    np.testing.assert_allclose(low["modal_relative_error_rms"], 0.5)
    np.testing.assert_allclose(low["modal_error_energy_fraction"], 0.8)
    assert low["num_modes"] == 4


def test_modal_shape_statistics_reports_error_frequency_geometry() -> None:
    rows = [
        {
            "evaluation_mode": "autoregressive_state",
            "model": "s4",
            "stride": 4,
            "target_frame": 8,
            "mode_index": mode,
            "num_samples": 64,
            "truth_resolved": True,
            "modal_target_power_mean": 1.0,
            "modal_error_power_mean": 1.0,
        }
        for mode in range(1, 6)
    ]

    summary = visualization.modal_shape_statistics(
        rows,
        ("s4",),
        "autoregressive_state",
        (8,),
        saved_frame_dt=0.005,
    )[0]

    np.testing.assert_allclose(summary["modal_relative_error_rms"], 1.0)
    np.testing.assert_allclose(summary["error_spectral_centroid_mode"], 3.0)
    np.testing.assert_allclose(summary["error_spectral_rms_mode"], np.sqrt(11.0))
    np.testing.assert_allclose(
        summary["error_spectral_d2_shape_factor"],
        np.sqrt(979.0 / 5.0),
    )
    assert summary["error_energy_q90_mode"] == 5
    assert summary["error_energy_q95_mode"] == 5


def test_modal_heatmap_plotter_emits_vector_and_raster_outputs(tmp_path) -> None:
    rows = _synthetic_modal_rows()
    rows[0]["modal_relative_error_rms"] = 13.9
    primary_models = {1: "s1", 2: "s2", 4: "s4", 8: "s8"}

    color_scale = visualization._plot_modal_panels(
        rows,
        primary_models,
        evaluation_mode="autoregressive_state",
        frames=(8, 16),
        metric="modal_relative_error_rms",
        log10_limits=(-3.0, 1.0),
        colorbar_label="Relative error",
        stem="modal_test",
        saved_frame_dt=0.005,
        output_dir=tmp_path,
    )

    assert (tmp_path / "modal_test.pdf").is_file()
    assert (tmp_path / "modal_test.png").is_file()
    assert color_scale["data_range"][1] == 13.9
    assert color_scale["raw_limits"] == [1.0e-3, 20.0]
    assert color_scale["num_values_outside_base_scale"] == 1


def test_operating_envelope_preserves_ties_and_metric_direction() -> None:
    statistics = []
    for stride, error, front in (
        (1, 0.12, 0.01),
        (2, 0.08, 0.02),
        (4, 0.06, 0.03),
        (8, 0.04, 0.04),
    ):
        statistics.append(
            {
                "stride": stride,
                "frame": 8,
                "error_mean": error,
                "time_mean_error_mean": error / 2.0,
                "smooth_error_mean": error,
                "shock_position_error_mean": front,
                "raw_completion_fraction": 1.0,
            }
        )
    reliability = [
        {
            "error_budget": budget,
            "stride": stride,
            "frame": 8,
            "reliability_fraction": 1.0,
        }
        for budget in visualization.CLOSEOUT_ERROR_BUDGETS
        for stride in visualization.REQUIRED_STRIDES
    ]
    ripple = []
    for stride, model, d2, tv_ratio in (
        (1, "s1", 8.0, 1.20),
        (2, "s2", 6.0, 1.10),
        (4, "s4", 4.0, 1.02),
        (8, "s8_seed20260707", 5.0, 0.95),
    ):
        ripple.append(
            {
                "model": model,
                "stride": stride,
                "frame": 8,
                "smooth_error_d2_rms_mean": d2,
                "smooth_state_tv_ratio_mean": tv_ratio,
            }
        )
    pareto = [
        {
            "call_stride": stride,
            "horizon": 8,
            "path": f"s{stride}_direct",
            "claim_eligible": True,
            "learned_host_to_host_wall_seconds": 1.0 / stride,
        }
        for stride in (2, 4, 8)
    ]

    rows = visualization.operating_envelope_rows(
        statistics,
        reliability,
        ripple,
        pareto,
        horizons=(8,),
    )
    indexed = {row["objective"]: row for row in rows}

    assert indexed["endpoint_global_error"]["winner_stride"] == 8
    assert indexed["front_position_error"]["winner_stride"] == 1
    assert indexed["away_front_d2_error"]["winner_stride"] == 4
    assert indexed["away_front_tv_deviation"]["winner_stride"] == 4
    assert indexed["host_to_host_latency"]["winner_stride"] == 8
    assert indexed["raw_completion"]["num_winners"] == 4
    assert indexed["raw_completion"]["winner_stride"] is None
    assert indexed["raw_completion"]["winner_strides"] == "1|2|4|8"


def test_closeout_paired_statistics_use_only_common_completed_cases() -> None:
    direct_rows = []
    for stride, values in ((4, (0.10, 0.20, 0.30)), (8, (0.05, 0.10, 0.15))):
        for case_id, value in enumerate(values):
            direct_rows.append(
                {
                    "stride": stride,
                    "frame": 8,
                    "case_id": case_id,
                    "path": f"s{stride}_direct",
                    "completed_step": not (stride == 4 and case_id == 2),
                    "fixed_scale_conservative_relative_l2": value,
                }
            )
    envelope = [
        {
            "frame": 8,
            "objective": "endpoint_global_error",
            "num_winners": 1,
            "winner_stride": 8,
            "runner_up_stride": 4,
        }
    ]

    rows = visualization.closeout_paired_statistics(
        direct_rows,
        envelope,
        bootstrap_replicates=100,
        bootstrap_seed=17,
    )

    assert len(rows) == 1
    assert rows[0]["num_common_cases"] == 2
    assert rows[0]["paired_mean_difference"] == pytest.approx(-0.075)
    assert rows[0]["winner_better_case_fraction"] == 1.0
    assert rows[0]["aggregate_selection_preserved_on_common_cases"] is True


def test_closeout_audit_detects_cross_artifact_metric_drift() -> None:
    contract = {
        "checkpoints": {
            str(stride): {"sha256": f"hash-{stride}"}
            for stride in visualization.REQUIRED_STRIDES
        },
        "saved_times": [0.005 * frame for frame in range(101)],
        "saved_times_common_across_cases": True,
        "case_ids": list(range(64)),
    }
    wave_occupancy = {
        "horizons": {
            "96": {
                "interior_active_fraction": 1.0,
                "state_change_relative_l2_median": 0.2,
            }
        }
    }
    d033_rows = []
    for stride, truth_error, on_policy_error in (
        (1, 0.1, 0.4),
        (2, 0.2, 0.3),
        (4, 0.3, 0.2),
        (8, 0.4, 0.1),
    ):
        d033_rows.extend(
            [
                {
                    "stride": stride,
                    "start_frame": 32,
                    "state_source": "truth",
                    "model_reference_cons_scaled_rel_l2_mean": truth_error,
                },
                {
                    "stride": stride,
                    "start_frame": 32,
                    "state_source": "on_policy",
                    "current_truth_cons_scaled_rel_l2_mean": on_policy_error,
                },
            ]
        )
    duplicate_row = {
        "model": "s4",
        "frame": 96,
        "state_cons_scaled_rel_l2_mean": 0.02,
        "smooth_error_d2_rms_mean": 10.0,
        "smooth_state_tv_ratio_mean": 1.1,
        "raw_completion_fraction": 1.0,
    }
    teacher_rows = [
        {
            "model": model,
            "band": "low_1_4",
            "modal_relative_error_rms": value,
        }
        for model, value in (
            ("s4", 1.0),
            ("s8_seed20260707", 1.2),
            ("s8_seed20260708", 1.3),
            ("s8_seed20260709", 1.4),
        )
    ]

    passing = visualization.closeout_consistency_audit(
        contract,
        wave_occupancy,
        d033_rows,
        [duplicate_row],
        [dict(duplicate_row)],
        teacher_rows,
    )
    drifted = dict(duplicate_row)
    drifted["state_cons_scaled_rel_l2_mean"] = 0.021
    failing = visualization.closeout_consistency_audit(
        contract,
        wave_occupancy,
        d033_rows,
        [duplicate_row],
        [drifted],
        teacher_rows,
    )

    assert passing["status"] == "pass"
    assert failing["status"] == "fail"
    assert failing["checks"]["d036_d038_duplicate_rows_match"] is False
