from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import scripts.time_dependent_no.evaluate_pcno_defect_corrections as correction_evaluator
from scripts.time_dependent_no.evaluate_pcno_defect_corrections import (
    HOOK_EQUIVALENCE_TOLERANCE_SOURCE,
    _append_metrics,
    _bias_sequence,
    _evaluate_case,
    _hook_equivalence_limits,
    _promotion_summary,
    _proposal_arms,
    _rollout_completion_checks,
    _run_hook_equivalence,
    _safe_admissibility_summary,
    _scale_band_fields,
    _teacher_forced_coverage_checks,
    parse_args,
)


@pytest.mark.parametrize(
    ("family", "absolute_limit", "relative_limit"),
    (
        ("bump", 2.0e-3, 1.0e-5),
        ("dynamic_fv", 2.0e-5, 1.0e-7),
    ),
)
def test_d071_hook_limits_are_bound_to_accepted_d068_contract(
    family: str, absolute_limit: float, relative_limit: float
) -> None:
    limits = _hook_equivalence_limits(family)
    assert limits == {
        "native_repeat_absolute_limit": absolute_limit,
        "shared_relative_l2_limit": relative_limit,
    }


def test_d071_forwards_and_reports_family_hook_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    device = object()

    def fake_hook(model: object, case: object, **kwargs: object) -> dict[str, object]:
        del model, case
        captured.update(kwargs)
        return {"passed": True, "hook_absolute_limit": 0.0025}

    monkeypatch.setattr(correction_evaluator, "_hook_equivalence", fake_hook)
    result = _run_hook_equivalence(
        object(),
        SimpleNamespace(family="bump"),
        device=device,
    )
    assert captured == {
        "device": device,
        "amp": "none",
        "absolute_limit": 2.0e-3,
        "relative_limit": 1.0e-5,
    }
    assert result["tolerance_source"] == HOOK_EQUIVALENCE_TOLERANCE_SOURCE
    assert result["native_repeat_absolute_limit"] == 2.0e-3
    assert result["shared_relative_l2_limit"] == 1.0e-5
    assert result["hook_absolute_limit"] == 0.0025


def test_d071_provenance_failure_precedes_model_construction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    built = False

    def fail_provenance(*args: object, **kwargs: object) -> dict:
        del args, kwargs
        raise RuntimeError("provenance stop")

    def build_model(*args: object, **kwargs: object) -> tuple[object, object]:
        nonlocal built
        del args, kwargs
        built = True
        return object(), object()

    monkeypatch.setattr(
        correction_evaluator,
        "select_device",
        lambda _: correction_evaluator.torch.device("cpu"),
    )
    monkeypatch.setattr(
        correction_evaluator,
        "PCNOEuler2DShardStore",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        correction_evaluator,
        "load_checkpoint_payload",
        lambda _: {},
    )
    monkeypatch.setattr(
        correction_evaluator,
        "_verify_common_provenance",
        fail_provenance,
    )
    monkeypatch.setattr(correction_evaluator, "build_checkpoint_model", build_model)
    args = SimpleNamespace(
        family="bump",
        device="cpu",
        data_dir=tmp_path,
        checkpoint=tmp_path / "checkpoint.pt",
    )
    with pytest.raises(RuntimeError, match="provenance stop"):
        correction_evaluator.run(args)
    assert not built


def test_proposal_arms_keep_persistent_and_local_channels_additive() -> None:
    case = SimpleNamespace(
        edges=np.asarray([[0, 1], [1, 2], [2, 3]], dtype=np.int64),
        weights=np.ones(4),
        physical_node_type=np.zeros(4, dtype=np.int64),
        residual_scale=np.ones(2),
    )
    current = np.zeros((4, 2))
    base = np.asarray([[0.0, 0.0], [2.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    bias = np.full((4, 2), 0.1)
    predictions, corrections, local = _proposal_arms(
        case,
        current=current,
        base_prediction=base,
        bias=bias,
        filter_cap=0.02,
    )
    np.testing.assert_allclose(
        corrections["combined"],
        corrections["persistent_rank8"] + corrections["local_dissipation"],
    )
    np.testing.assert_allclose(predictions["combined"], base + corrections["combined"])
    assert local["local_summary"].applied_relative_norm <= 0.02 + 1.0e-12


def test_persistent_bias_is_exactly_zero_on_non_normal_nodes() -> None:
    case = SimpleNamespace(
        family="bump",
        resolution_name="native_graph",
        nodes=np.asarray(
            [[0.0, 0.0], [1.0, 0.5], [2.0, 1.0], [3.0, 1.5]],
            dtype=np.float64,
        ),
        residual_scale=np.ones(2),
        physical_node_type=np.asarray([1, 0, 0, 2], dtype=np.int64),
    )
    _, bias = _bias_sequence(
        case,
        {"native_graph": np.ones((3, 8, 2), dtype=np.float64)},
    )
    np.testing.assert_allclose(bias[:, [0, 3]], 0.0)
    assert np.any(np.abs(bias[:, [1, 2]]) > 0.0)


def test_family_specific_scale_bands_reconstruct_the_field() -> None:
    field = np.arange(32, dtype=np.float64).reshape(8, 4) / 11.0
    edges = np.asarray([[index, index + 1] for index in range(7)], dtype=np.int64)
    common = {
        "residual_scale": np.asarray([1.0, 2.0, 3.0, 4.0]),
        "edges": edges,
    }
    dynamic = SimpleNamespace(family="dynamic_fv", resolution=(4, 2), **common)
    dynamic_bands, dynamic_closure, dynamic_contract = _scale_band_fields(
        dynamic, field
    )
    bump = SimpleNamespace(family="bump", resolution=None, **common)
    bump_bands, bump_closure, bump_contract = _scale_band_fields(bump, field)
    for bands, closure in (
        (dynamic_bands, dynamic_closure),
        (bump_bands, bump_closure),
    ):
        np.testing.assert_allclose(
            bands["large"] + bands["transition"] + bands["local"],
            field,
            atol=2.0e-12,
        )
        assert closure < 2.0e-12
    assert dynamic_contract == "physical_dct_wavelength"
    assert bump_contract == "nonorthogonal_two_level_graph_proxy"


def test_promotion_summary_enforces_family_controls() -> None:
    cases = ("a", "b")
    metric_rows = []
    for case_id in cases:
        for arm, value in (("baseline", 1.0), ("combined", 0.9)):
            for region in ("all", "shock", "smooth", "boundary_nodes", "vortex"):
                metric_rows.append(
                    {
                        "case_id": case_id,
                        "resolution": "250x100",
                        "mode": "free_rollout",
                        "call": 30,
                        "arm": arm,
                        "region": region,
                        "state_error_rms": value,
                        "state_error_graph_highpass_rms": value,
                    }
                )
    front_rows = []
    for case_id in cases:
        for arm, value in (("baseline", 1.0), ("combined", 0.9)):
            front_rows.append(
                {
                    "case_id": case_id,
                    "resolution": "250x100",
                    "mode": "free_rollout",
                    "call": 30,
                    "arm": arm,
                    "proxy_centroid_absolute_error": value,
                    "proxy_strength_absolute_error": value,
                    "proxy_thickness_absolute_error": value,
                }
            )
    completion = [
        {
            "case_id": case_id,
            "resolution": "250x100",
            "arm": arm,
            "call": "summary",
            "accepted": True,
        }
        for case_id in cases
        for arm in ("baseline", "persistent_rank8", "local_dissipation", "combined")
    ]
    summary = _promotion_summary("dynamic_fv", metric_rows, front_rows, completion, 30)
    assert summary["promoted"]
    assert summary["median_combined_to_baseline_final_state_error_ratio"] == 0.9
    metric_rows[-1]["state_error_rms"] = 2.0
    assert not _promotion_summary(
        "dynamic_fv", metric_rows, front_rows, completion, 30
    )["promoted"]


def test_promotion_rejects_missing_and_zero_baseline_case_controls() -> None:
    cases = ("a", "b")
    rows = []
    for case_id in cases:
        for arm, value in (("baseline", 1.0), ("combined", 0.9)):
            for region in ("all", "shock", "smooth", "boundary_nodes"):
                rows.append(
                    {
                        "case_id": case_id,
                        "resolution": "native_graph",
                        "mode": "free_rollout",
                        "call": 20,
                        "arm": arm,
                        "region": region,
                        "state_error_rms": value,
                        "state_error_graph_highpass_rms": value,
                    }
                )
    completion = [
        {
            "case_id": case_id,
            "resolution": "native_graph",
            "arm": arm,
            "call": "summary",
            "accepted": True,
        }
        for case_id in cases
        for arm in ("baseline", "persistent_rank8", "local_dissipation", "combined")
    ]
    assert _promotion_summary("bump", rows, [], completion, 20)["promoted"]
    missing = [
        row
        for row in rows
        if not (
            row["case_id"] == "b"
            and row["arm"] == "combined"
            and row["region"] == "smooth"
        )
    ]
    assert not _promotion_summary("bump", missing, [], completion, 20)["promoted"]
    zero = [dict(row) for row in rows]
    next(
        row
        for row in zero
        if row["case_id"] == "b"
        and row["arm"] == "baseline"
        and row["region"] == "shock"
    )["state_error_rms"] = 0.0
    assert not _promotion_summary("bump", zero, [], completion, 20)["promoted"]


def test_rollout_completion_gate_rejects_one_inadmissible_arm() -> None:
    rows = []
    for arm in ("baseline", "persistent_rank8", "local_dissipation", "combined"):
        rows.append(
            {
                "case_id": "a",
                "resolution": "native_graph",
                "arm": arm,
                "call": 1,
                "accepted": arm != "combined",
            }
        )
        rows.append(
            {
                "case_id": "a",
                "resolution": "native_graph",
                "arm": arm,
                "call": "summary",
                "accepted": arm != "combined",
            }
        )
    checks = _rollout_completion_checks(rows, [("a", "native_graph")], 1)
    assert checks["step_matrix_complete"]
    assert checks["summary_matrix_complete"]
    assert not checks["complete_and_admissible"]


def test_nonfinite_admissibility_is_recordable_without_nan_reduction() -> None:
    summary = _safe_admissibility_summary(
        np.asarray([[np.nan, 0.0, 0.0, 1.0]]), gamma=1.4
    )
    assert not summary["finite"]
    assert not summary["admissible"]
    assert summary["minimum_pressure"] is None


def test_component_metrics_report_absolute_and_residual_scaled_update_error() -> None:
    nodes = np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    current = np.tile(np.asarray([1.0, 0.0, 0.0, 2.5]), (4, 1))
    truth_update = np.tile(np.asarray([0.1, 0.0, 0.0, 0.1]), (4, 1))
    update_error = np.tile(np.asarray([0.2, 0.0, 0.0, 0.0]), (4, 1))
    target = current + truth_update
    prediction = current + truth_update + update_error
    case = SimpleNamespace(
        family="bump",
        case_id="synthetic",
        resolution_name="native_graph",
        reference_states=np.asarray([current, target]),
        physical_times=np.asarray([0.0, 1.0]),
        nodes=nodes,
        edges=np.asarray([[0, 1], [1, 2], [2, 3]], dtype=np.int64),
        weights=np.ones(4),
        physical_node_type=np.zeros(4, dtype=np.int64),
        boundary_distance=np.ones(4),
        state_scale=np.ones(4),
        residual_scale=np.asarray([2.0, 1.0, 1.0, 1.0]),
        gamma=1.4,
    )
    metrics: list[dict] = []
    components: list[dict] = []
    fronts: list[dict] = []
    _append_metrics(
        metrics,
        components,
        fronts,
        case,
        mode="free_rollout",
        call=1,
        arm="baseline",
        current=current,
        prediction=prediction,
        target=target,
        correction=np.zeros_like(current),
        shock_quantile=0.9,
    )
    rho = next(row for row in components if row["component"] == "rho")
    assert rho["update_error_absolute_rms"] == pytest.approx(0.2)
    assert rho["update_error_rms"] == pytest.approx(0.1)


def test_d071_parser_rejects_unregistered_dynamic_grid(tmp_path: Path) -> None:
    argv = [
        "--family",
        "dynamic_fv",
        "--checkpoint",
        str(tmp_path / "checkpoint.pt"),
        "--normalization-json",
        str(tmp_path / "normalization.json"),
        "--split-json",
        str(tmp_path / "split.json"),
        "--data-dir",
        str(tmp_path / "data"),
        "--output-dir",
        str(tmp_path / "output"),
        "--rollout-calls",
        "30",
        "--expected-checkpoint-sha256",
        "b" * 64,
        "--expected-normalization-sha256",
        "c" * 64,
        "--expected-split-sha256",
        "d" * 64,
        "--expected-data-manifest-digest",
        "e" * 64,
        "--expected-source-base-git-head",
        "1" * 40,
        "--source-manifest",
        str(tmp_path / "source_manifest.json"),
        "--expected-source-manifest-sha256",
        "2" * 64,
        "--family-root",
        str(tmp_path / "family"),
        "--multires-reference-root",
        str(tmp_path / "multires"),
        "--expected-family-manifest-sha256",
        "f" * 64,
        "--resolutions",
        "125x50",
        "250x100",
    ]
    with pytest.raises(ValueError, match="freezes dynamic grids"):
        parse_args(argv)


def test_evaluate_case_deactivates_only_nonfinite_recurrent_arm(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    nodes = np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    initial = np.tile(np.asarray([1.0, 0.0, 0.0, 2.5]), (4, 1))
    states = np.asarray(
        [
            initial,
            initial + np.asarray([0.1, 0.0, 0.0, 0.1]),
            initial + np.asarray([0.2, 0.0, 0.0, 0.2]),
        ]
    )
    case = SimpleNamespace(
        family="bump",
        case_id="synthetic",
        resolution_name="native_graph",
        reference_states=states,
        physical_times=np.asarray([0.0, 1.0, 2.0]),
        nodes=nodes,
        edges=np.asarray([[0, 1], [1, 2], [2, 3]], dtype=np.int64),
        weights=np.ones(4),
        physical_node_type=np.zeros(4, dtype=np.int64),
        boundary_distance=np.ones(4),
        state_scale=np.ones(4),
        residual_scale=np.ones(4),
        gamma=1.4,
    )

    def fake_predict(
        model: object, active_case: object, state: np.ndarray
    ) -> np.ndarray:
        del model, active_case
        if np.isclose(np.mean(state[:, 0]), 1.04):
            return np.full_like(state, np.nan)
        prediction = np.array(state, copy=True)
        prediction[:, 0] += 0.05
        prediction[:, 3] += 0.05
        return prediction

    def fake_proposals(
        active_case: object,
        *,
        current: np.ndarray,
        base_prediction: np.ndarray,
        bias: np.ndarray,
        filter_cap: float,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict]:
        del bias, filter_cap
        amounts = {
            "baseline": 0.0,
            "persistent_rank8": -0.01,
            "local_dissipation": 0.02,
            "combined": 0.01,
        }
        corrections = {}
        predictions = {}
        for arm, amount in amounts.items():
            correction = np.zeros_like(current)
            correction[:, 0] = amount
            corrections[arm] = correction
            predictions[arm] = base_prediction + correction
        local_summary = SimpleNamespace(
            eligible_edge_count=len(active_case.edges),
            applied_relative_norm=0.0,
            weighted_mean_closure=np.zeros(4),
        )
        return (
            predictions,
            corrections,
            {
                "sensor": np.zeros(len(current)),
                "local_correction": corrections["local_dissipation"],
                "local_summary": local_summary,
            },
        )

    monkeypatch.setattr(correction_evaluator, "_predict", fake_predict)
    monkeypatch.setattr(correction_evaluator, "_proposal_arms", fake_proposals)
    metric_rows: list[dict] = []
    component_rows: list[dict] = []
    band_rows: list[dict] = []
    front_rows: list[dict] = []
    correction_rows: list[dict] = []
    completion_rows: list[dict] = []
    teacher_rows: list[dict] = []
    _evaluate_case(
        object(),
        case,
        frozen_coefficients={"native_graph": np.zeros((2, 8, 4))},
        filter_cap=0.02,
        shock_quantile=0.9,
        metric_rows=metric_rows,
        component_rows=component_rows,
        band_rows=band_rows,
        front_rows=front_rows,
        correction_rows=correction_rows,
        completion_rows=completion_rows,
        teacher_completion_rows=teacher_rows,
        save_visual=False,
        visual_dir=tmp_path,
    )
    summaries = {row["arm"]: row for row in completion_rows if row["call"] == "summary"}
    assert summaries["persistent_rank8"]["valid_length"] == 1
    assert all(
        summaries[arm]["valid_length"] == 2
        for arm in ("baseline", "local_dissipation", "combined")
    )
    assert any(
        row.get("failure_stage") == "base_prediction_nonfinite"
        and row["arm"] == "persistent_rank8"
        for row in completion_rows
    )
    assert not any(
        row["mode"] == "free_rollout"
        and row["arm"] == "persistent_rank8"
        and row["call"] == 2
        for row in metric_rows
    )
    completion = _rollout_completion_checks(
        completion_rows, [("synthetic", "native_graph")], 2
    )
    assert completion["step_matrix_complete"]
    assert not completion["complete_and_admissible"]
    assert _teacher_forced_coverage_checks(
        teacher_rows, [("synthetic", "native_graph")], 2
    )["complete_and_finite"]
