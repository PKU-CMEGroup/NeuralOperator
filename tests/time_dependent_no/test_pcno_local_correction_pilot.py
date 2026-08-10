from __future__ import annotations

import json

import numpy as np
import torch

import scripts.time_dependent_no.evaluate_pcno_local_correction_pilot as pilot
import scripts.time_dependent_no.evaluate_pcno_native_residual_correction as parent
from scripts.time_dependent_no.evaluate_pcno_node_type_interventions import CaseData


def _constant_case() -> CaseData:
    state = np.asarray([[1.0, 0.0, 0.0, 2.5], [1.0, 0.0, 0.0, 2.5]])
    return CaseData(
        family="dynamic_fv",
        case_id="synthetic",
        resolution=(2, 1),
        sample={},
        reference_states=np.stack((state, state, state)),
        physical_times=np.asarray([0.0, 0.02, 0.04]),
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


def test_persistent_rollout_uses_base_proposal_then_adds_frozen_bias(
    monkeypatch,
) -> None:
    case = _constant_case()
    monkeypatch.setattr(
        pilot,
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
    selected = parent.Candidate(key="rank8_gain0p5_raw", rank=8, gain=0.5)
    bias = np.zeros((2, 2, 4), dtype=np.float64)
    bias[:, :, 0] = 0.01
    payload = pilot._rollout_arm(
        torch.nn.Identity(),
        case,
        selected,
        bias_sequence=bias,
        arm=next(arm for arm in pilot.ARM_SPECS if arm.name == "persistent"),
        shock_quantile=0.9,
        calls=2,
    )
    assert payload.result.complete
    np.testing.assert_allclose(payload.local_corrections, 0.0)
    np.testing.assert_allclose(payload.persistent_corrections[:, :, 0], -0.005)
    np.testing.assert_allclose(payload.result.states[1, :, 0], 0.995)
    np.testing.assert_allclose(payload.result.states[2, :, 0], 0.990)
    np.testing.assert_allclose(
        payload.result.states[1:] - case.reference_states[1:],
        np.cumsum(payload.result.defects, axis=0),
        atol=1.0e-15,
    )


def test_prefix_replay_is_exact_for_identical_rollouts(monkeypatch) -> None:
    case = _constant_case()
    monkeypatch.setattr(
        pilot,
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
    selected = parent.Candidate(key="zero", rank=0, gain=0.0)
    bias = np.zeros((2, 2, 4), dtype=np.float64)
    arm = pilot.ARM_SPECS[0]
    first = pilot._rollout_arm(
        torch.nn.Identity(),
        case,
        selected,
        bias_sequence=bias,
        arm=arm,
        shock_quantile=0.9,
        calls=2,
    )
    second = pilot._rollout_arm(
        torch.nn.Identity(),
        case,
        selected,
        bias_sequence=bias,
        arm=arm,
        shock_quantile=0.9,
        calls=2,
    )
    row = pilot._prefix_replay_row(
        case.case_id, arm.name, first.result, second.result, probe_calls=2
    )
    assert row["passed"]
    assert row["exact_equal"]
    assert row["maximum_absolute"] == 0.0
    assert row["relative_l2"] == 0.0


def test_promotion_selects_only_combined_arm_passing_all_case_first_gates() -> None:
    comparisons = []
    summaries = []
    combined = [arm for arm in pilot.ARM_SPECS if arm.name.startswith("combined_")]
    for case_index in range(6):
        case_id = f"case{case_index}"
        summaries.append(
            {"case_id": case_id, "arm": "persistent", "net_defect_rms": 1.0}
        )
        for arm_index, arm in enumerate(combined):
            ratio = 0.99 if arm_index == 0 else 1.01
            summaries.append(
                {
                    "case_id": case_id,
                    "arm": arm.name,
                    "net_defect_rms": ratio,
                }
            )
            region_key = (
                "endpoint_state__vortex"
                if arm.pathway == "vortex_isotropic"
                else "endpoint_state__shock"
            )
            comparisons.append(
                {
                    "case_id": case_id,
                    "comparison": f"{arm.name}_vs_persistent",
                    "endpoint_state_ratio": ratio,
                    "residual_rms_ratio": ratio,
                    "maximum_control_ratio": 1.0,
                    "control_ratios_json": json.dumps({region_key: ratio}),
                }
            )
    result = pilot._promotion(comparisons, summaries)
    assert result["efficacy_passed"]
    assert result["selected_arm"] == "combined_shock_isotropic"
    assert result["candidate_rows"][0]["endpoint_nonworse_count"] == 6


def test_d080_spec_uses_callback_without_changing_d078_default() -> None:
    assert pilot.d078.D078_SPEC.evaluation_runner is None
    assert pilot.D080_SPEC.evaluation_runner is pilot._conditional_evaluation
    assert pilot.D080_SPEC.deterministic_algorithms


def test_materialized_csv_rows_omits_only_empty_tables() -> None:
    rows = [{"case_id": "sv_e00_y00", "arm": "zero"}]
    materialized = pilot._materialized_csv_rows(
        {
            "evaluation_case_summary.csv": rows,
            "projection_metrics.csv": [],
            "projection_component_metrics.csv": [],
        }
    )
    assert materialized == {"evaluation_case_summary.csv": rows}
