from __future__ import annotations

import json

import numpy as np
import torch

import scripts.time_dependent_no.evaluate_pcno_local_correction_pilot as d080
import scripts.time_dependent_no.evaluate_pcno_native_residual_correction as parent
import scripts.time_dependent_no.evaluate_pcno_time_windowed_local_correction as d081
from scripts.time_dependent_no.evaluate_pcno_node_type_interventions import CaseData
from utility.time_dependent_no.pcno_defect_corrections import DissipationResult


def _constant_case(calls: int = 3) -> CaseData:
    state = np.asarray([[1.0, 0.0, 0.0, 2.5], [1.0, 0.0, 0.0, 2.5]])
    return CaseData(
        family="dynamic_fv",
        case_id="synthetic",
        resolution=(2, 1),
        sample={},
        reference_states=np.repeat(state[None, :, :], calls + 1, axis=0),
        physical_times=np.arange(calls + 1, dtype=np.float64) * 0.02,
        nodes=np.asarray([[0.5, 0.5], [1.5, 0.5]]),
        edges=np.asarray([[0, 1], [1, 0]], dtype=np.int64),
        weights=np.ones(2),
        physical_node_type=np.zeros(2, dtype=np.int64),
        boundary_distance=np.ones(2),
        state_scale=np.ones(4),
        residual_scale=np.ones(4),
        gamma=1.4,
        provenance={},
    )


def test_frozen_schedule_inventory_has_equal_nominal_dose_controls() -> None:
    rows = {row["arm"]: row for row in d081._schedule_rows(30)}
    assert rows["combined_shock_normal"]["active_local_calls"] == 30
    assert rows[d081.PRIMARY_ARM]["active_local_calls"] == 20
    assert rows["combined_shock_normal_late20"]["active_local_calls"] == 20
    assert rows["combined_shock_normal_dose_matched"]["active_local_calls"] == 30
    np.testing.assert_allclose(
        rows[d081.PRIMARY_ARM]["nominal_cap_dose"],
        rows["combined_shock_normal_late20"]["nominal_cap_dose"],
    )
    np.testing.assert_allclose(
        rows[d081.PRIMARY_ARM]["nominal_cap_dose"],
        rows["combined_shock_normal_dose_matched"]["nominal_cap_dose"],
    )


def test_call_window_turns_local_correction_off_without_changing_persistent(
    monkeypatch,
) -> None:
    case = _constant_case()
    monkeypatch.setattr(
        d080,
        "_unique_undirected_edges",
        lambda edges: np.asarray([[0, 1]], dtype=np.int64),
    )
    monkeypatch.setattr(
        parent,
        "_predict",
        lambda model, observed_case, current: np.array(current, copy=True),
    )
    monkeypatch.setattr(
        parent,
        "_summarize_rollout",
        lambda case, states, defects, complete, shock_quantile: {
            "final_state_error": float(np.linalg.norm(states[-1] - states[0])),
            "residual_rms": float(np.sqrt(np.mean(np.square(defects)))),
            "controls": {},
        },
    )

    def fake_local(update, current, observed_case, edges, arm):
        correction = np.zeros_like(update)
        if arm.pathway is not None:
            correction[:, 0] = 0.01
        return (
            DissipationResult(
                correction=correction,
                sensor=np.zeros(update.shape[0]),
                uncapped_relative_norm=arm.cap,
                applied_relative_norm=arm.cap,
                applied_scale=1.0 if arm.pathway is not None else 0.0,
                weighted_mean_closure=np.zeros(update.shape[1]),
                eligible_edge_count=1 if arm.pathway is not None else 0,
            ),
            int(np.count_nonzero(arm.pathway is not None)),
        )

    monkeypatch.setattr(d080, "_local_correction", fake_local)
    arm = d080.LocalArmSpec(
        "scheduled",
        "shock_normal",
        0.01,
        True,
        local_start_call=1,
        local_stop_call=2,
    )
    payload = d080._rollout_arm(
        torch.nn.Identity(),
        case,
        parent.Candidate(key="zero", rank=0, gain=0.0),
        bias_sequence=np.zeros_like(case.reference_states[1:]),
        arm=arm,
        shock_quantile=0.9,
        calls=3,
    )
    assert payload.result.complete
    np.testing.assert_allclose(payload.local_corrections[:2, :, 0], 0.01)
    np.testing.assert_allclose(payload.local_corrections[2], 0.0)
    np.testing.assert_allclose(payload.result.states[-1, :, 0], 1.02)


def _promotion_rows(dose_endpoint: float = 0.97):
    endpoints = {
        "combined_shock_normal": 0.99,
        d081.PRIMARY_ARM: 0.95,
        "combined_shock_normal_late20": 0.98,
        "combined_shock_normal_dose_matched": dose_endpoint,
    }
    summaries = []
    comparisons = []
    for index in range(6):
        case_id = f"case{index}"
        summaries.append(
            {
                "case_id": case_id,
                "arm": "persistent",
                "final_state_error": 1.0,
                "residual_rms": 1.0,
                "net_defect_rms": 1.0,
            }
        )
        for arm, endpoint in endpoints.items():
            residual = 0.98
            cumulative = endpoint
            summaries.append(
                {
                    "case_id": case_id,
                    "arm": arm,
                    "final_state_error": endpoint,
                    "residual_rms": residual,
                    "net_defect_rms": cumulative,
                }
            )
            comparisons.append(
                {
                    "case_id": case_id,
                    "comparison": f"{arm}_vs_persistent",
                    "endpoint_state_ratio": endpoint,
                    "residual_rms_ratio": residual,
                    "maximum_control_ratio": 1.0,
                    "control_ratios_json": json.dumps(
                        {"endpoint_state__shock": endpoint}
                    ),
                }
            )
    return comparisons, summaries


def test_promotion_separates_efficacy_from_timing_mechanism() -> None:
    comparisons, summaries = _promotion_rows()
    result = d081._promotion(comparisons, summaries)
    assert result["efficacy_passed"]
    assert result["selected_arm"] == d081.PRIMARY_ARM
    assert result["timing_mechanism_supported"]
    assert all(
        row["median_endpoint_ratio"] < 1.0
        and row["median_cumulative_defect_ratio"] < 1.0
        for row in result["timing_rows"]
    )

    comparisons, summaries = _promotion_rows(dose_endpoint=0.94)
    result = d081._promotion(comparisons, summaries)
    assert result["efficacy_passed"]
    assert not result["timing_mechanism_supported"]


def test_d081_spec_preserves_d080_and_uses_its_own_callback() -> None:
    assert d080.D080_SPEC.evaluation_runner is d080._conditional_evaluation
    assert d081.D081_SPEC.evaluation_runner is d081._conditional_evaluation
    assert d081.D081_SPEC.deterministic_algorithms
    assert d081.DOSE_MATCHED_CAP == 1.0 / 150.0


def test_semantic_selector_digest_ignores_only_wrapper_identity(tmp_path) -> None:
    first = {
        "schema": "d080",
        "experiment_contract": "old",
        "artifact_sha256": {"one": "aaa"},
        "response_policy": {"schema": "d080", "gain": 0.5},
        "probe_calls": 5,
    }
    second = {
        **first,
        "schema": "d081",
        "experiment_contract": "new",
        "artifact_sha256": {"one": "bbb"},
        "response_policy": {"schema": "d081", "gain": 0.5},
    }
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    first_path.write_text(json.dumps(first), encoding="utf-8")
    second_path.write_text(json.dumps(second), encoding="utf-8")
    assert d081._semantic_json_sha256(
        first_path, selector=True
    ) == d081._semantic_json_sha256(second_path, selector=True)

    second["response_policy"]["gain"] = 0.25
    second_path.write_text(json.dumps(second), encoding="utf-8")
    assert d081._semantic_json_sha256(
        first_path, selector=True
    ) != d081._semantic_json_sha256(second_path, selector=True)


def test_semantic_coefficient_digest_ignores_only_wrapper_schema(tmp_path) -> None:
    first_path = tmp_path / "first.npz"
    second_path = tmp_path / "second.npz"
    coefficients = np.arange(12, dtype=np.float64).reshape(3, 4)
    np.savez(
        first_path,
        schema=np.asarray("d080"),
        rank=np.asarray(8),
        coefficients=coefficients,
    )
    np.savez(
        second_path,
        schema=np.asarray("d081"),
        rank=np.asarray(8),
        coefficients=coefficients,
    )
    assert d081._semantic_coefficient_sha256(
        first_path
    ) == d081._semantic_coefficient_sha256(second_path)

    np.savez(
        second_path,
        schema=np.asarray("d081"),
        rank=np.asarray(8),
        coefficients=coefficients + 1.0,
    )
    assert d081._semantic_coefficient_sha256(
        first_path
    ) != d081._semantic_coefficient_sha256(second_path)
