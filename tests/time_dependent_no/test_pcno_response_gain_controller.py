from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import scripts.time_dependent_no.evaluate_pcno_response_gain_controller as d076
from scripts.time_dependent_no.evaluate_pcno_native_residual_correction import (
    Candidate,
)


def _case() -> SimpleNamespace:
    return SimpleNamespace(
        family="dynamic_fv",
        case_id="synthetic",
        resolution_name="250x100",
        nodes=np.asarray(
            [[0.0, 0.0], [0.5, 0.25], [1.0, 0.75], [1.5, 1.0]],
            dtype=np.float64,
        ),
        weights=np.asarray([1.0, 2.0, 1.0, 1.0], dtype=np.float64),
        physical_node_type=np.asarray([0, 0, 0, 1], dtype=np.int64),
        state_scale=np.asarray([2.0, 3.0, 4.0, 5.0], dtype=np.float64),
        residual_scale=np.ones(4),
        gamma=1.4,
    )


def _features(amplitude: float, response: float = 0.1) -> dict[str, float]:
    return {
        "initial_vortex_amplitude": amplitude,
        "initial_vortex_centroid_y": 0.5,
        "response_state_rms": response,
        "response_integral__rho": 0.01,
        "response_integral__rho_u": 0.02,
        "response_integral__rho_v": 0.03,
        "response_integral__energy": 0.04,
        "response_amplification": 0.8,
        "response_correction_cosine": 0.9,
    }


def test_candidate_inventory_is_zero_plus_frozen_raw_gain_grid() -> None:
    candidates = d076._candidate_inventory(smoke=False)
    assert candidates[0] == Candidate(key="zero", rank=0, gain=0.0)
    assert [candidate.gain for candidate in candidates[1:]] == [
        0.125,
        0.25,
        0.5,
        1.0,
    ]
    assert {candidate.rank for candidate in candidates[1:]} == {8}
    assert {candidate.correction_policy for candidate in candidates} == {"raw"}


def test_parse_args_populates_inherited_dynamic_loader_contract(tmp_path) -> None:
    required = d076.parent.REQUIRED_ARTIFACTS["dynamic_fv"]
    args = d076.parse_args(
        [
            "--checkpoint",
            str(tmp_path / "best.pt"),
            "--normalization-json",
            str(tmp_path / "normalization.json"),
            "--split-json",
            str(tmp_path / "split.json"),
            "--data-dir",
            str(tmp_path / "shards"),
            "--family-root",
            str(tmp_path / "family"),
            "--multires-reference-root",
            str(tmp_path / "references"),
            "--output-dir",
            str(tmp_path / "output"),
            "--expected-checkpoint-sha256",
            required["checkpoint"],
            "--expected-normalization-sha256",
            required["normalization"],
            "--expected-split-sha256",
            required["split"],
            "--expected-data-manifest-digest",
            required["data_manifest"],
            "--expected-family-manifest-sha256",
            required["family_manifest"],
            "--expected-source-base-git-head",
            "b2193946dd20a350053520e5406b98bfee3c4ae3",
            "--source-manifest",
            str(tmp_path / "manifest.json"),
            "--expected-source-manifest-sha256",
            "a" * 64,
        ]
    )
    assert args.family == "dynamic_fv"
    assert args.resolutions == ("250x100",)
    assert args.training_resolution == "250x100"
    assert args.rollout_calls == 30
    assert args.probe_calls == d076.PROBE_CALLS


def test_initial_vortex_features_use_only_dynamic_type0_physical_volume() -> None:
    case = _case()
    state = np.zeros((4, 4), dtype=np.float64)
    state[:, 2] = [1.0, 3.0, -1.0, 1000.0]
    amplitude, centroid = d076._initial_vortex_features(case, state)
    weights = case.weights[:3]
    values = state[:3, 2]
    mean = np.dot(weights, values) / weights.sum()
    centered_square = np.square(values - mean)
    expected_amplitude = (
        np.sqrt(np.dot(weights, centered_square) / weights.sum()) / case.state_scale[2]
    )
    expected_centroid = np.dot(weights * centered_square, case.nodes[:3, 1]) / np.dot(
        weights, centered_square
    )
    np.testing.assert_allclose(amplitude, expected_amplitude)
    np.testing.assert_allclose(centroid, expected_centroid)


def test_probe_rollout_accepts_only_supplied_initial_state(monkeypatch) -> None:
    case = _case()
    initial = np.zeros((4, 4), dtype=np.float64)
    candidate = Candidate(key="rank8_gain0p5_raw", rank=8, gain=0.5)
    bias = np.ones((2, 4, 4), dtype=np.float64)

    def fake_predict(model, observed_case, current):
        del model
        assert observed_case is case
        assert not hasattr(observed_case, "reference_states")
        return current + 1.0

    monkeypatch.setattr(d076.parent, "_predict", fake_predict)
    monkeypatch.setattr(
        d076.parent,
        "_safe_admissibility_summary",
        lambda prediction, *, gamma: {
            "finite": bool(np.isfinite(prediction).all()),
            "admissible": True,
        },
    )
    result = d076._rollout_probe(
        object(),
        case,
        candidate,
        initial_state=initial,
        bias_sequence=bias,
        probe_calls=2,
    )
    assert result.complete
    np.testing.assert_allclose(result.corrections, -0.5)
    np.testing.assert_allclose(result.states[1], 0.5)
    np.testing.assert_allclose(result.states[2], 1.0)


def test_response_probe_features_match_registered_denominators() -> None:
    case = _case()
    initial = np.zeros((4, 4), dtype=np.float64)
    initial[:, 2] = [1.0, 3.0, -1.0, 0.0]
    response = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 3.0, 0.0],
            [0.0, 0.0, 0.0, 4.0],
        ]
    )
    direct = 2.0 * response
    baseline = d076.ProbeRollout(
        candidate=Candidate(key="zero", rank=0, gain=0.0),
        complete=True,
        states=np.asarray([initial, initial]),
        corrections=np.zeros((1, 4, 4)),
        admissibility_rows=(),
    )
    corrected = d076.ProbeRollout(
        candidate=Candidate(key="rank8_gain0p5_raw", rank=8, gain=0.5),
        complete=True,
        states=np.asarray([initial, initial + response]),
        corrections=np.asarray([direct]),
        admissibility_rows=(),
    )
    observed = d076._response_probe_features(case, baseline, corrected, probe_calls=1)
    expected_rms = d076.parent._weighted_rms(
        response,
        weights=case.weights,
        scale=case.state_scale,
    )
    expected_integrals = np.einsum("n,nc->c", case.weights, response) / case.state_scale
    np.testing.assert_allclose(observed["response_state_rms"], expected_rms)
    for index, name in enumerate(d076.parent.COMPONENTS):
        np.testing.assert_allclose(
            observed[f"response_integral__{name}"], expected_integrals[index]
        )
    np.testing.assert_allclose(observed["response_amplification"], 0.5)
    np.testing.assert_allclose(observed["response_correction_cosine"], 1.0)
    assert tuple(observed) == d076.FEATURE_NAMES


def _policy_fixture():
    candidates = (
        Candidate(key="zero", rank=0, gain=0.0),
        Candidate(key="rank8_gain0p125_raw", rank=8, gain=0.125),
        Candidate(key="rank8_gain0p5_raw", rank=8, gain=0.5),
    )
    rows = []
    for candidate in candidates[1:]:
        for index, amplitude in enumerate((1.0, 1.1, 1.2, 1.3)):
            rows.append(
                {
                    "case_id": f"c{index}",
                    "candidate": candidate.key,
                    "gain": candidate.gain,
                    **_features(amplitude, response=0.1 + 0.01 * index),
                    "safe": candidate.gain == 0.125,
                    "endpoint_state_ratio": (
                        0.90 + 0.01 * index if candidate.gain == 0.125 else 0.80
                    ),
                    "residual_rms_ratio": 0.99,
                    "maximum_control_ratio": (1.01 if candidate.gain == 0.125 else 1.2),
                }
            )
    return candidates, rows


def test_response_policy_selects_only_three_neighbor_safe_gain() -> None:
    candidates, rows = _policy_fixture()
    query = {
        candidate.key: {
            "case_id": "query",
            "candidate": candidate.key,
            "gain": candidate.gain,
            **_features(1.15),
        }
        for candidate in candidates[1:]
    }
    frozen = d076._frozen_policy_record(rows, candidates)
    selected, choices, neighbors = d076._select_response_gain(
        query,
        rows,
        candidates,
        frozen_policy=frozen,
    )
    assert selected.gain == 0.125
    assert len(choices) == 2
    assert len(neighbors) == 2 * d076.PROBE_K
    assert next(row for row in choices if row["gain"] == 0.125)["eligible"]
    assert not next(row for row in choices if row["gain"] == 0.5)["eligible"]


def test_response_policy_fails_closed_above_amplitude_support() -> None:
    candidates, rows = _policy_fixture()
    query = {
        candidate.key: {
            "case_id": "query",
            "candidate": candidate.key,
            "gain": candidate.gain,
            **_features(1.31),
        }
        for candidate in candidates[1:]
    }
    frozen = d076._frozen_policy_record(rows, candidates)
    selected, choices, _ = d076._select_response_gain(
        query,
        rows,
        candidates,
        frozen_policy=frozen,
    )
    assert selected.is_zero
    assert all(not row["upper_amplitude_support_pass"] for row in choices)


def test_response_policy_rejects_changed_frozen_feature_inventory() -> None:
    candidates, rows = _policy_fixture()
    query = {
        candidate.key: {
            "case_id": "query",
            "candidate": candidate.key,
            "gain": candidate.gain,
            **_features(1.15),
        }
        for candidate in candidates[1:]
    }
    frozen = d076._frozen_policy_record(rows, candidates)
    frozen["feature_names"] = frozen["feature_names"][:-1]
    try:
        d076._select_response_gain(
            query,
            rows,
            candidates,
            frozen_policy=frozen,
        )
    except ValueError as error:
        assert "feature inventory" in str(error)
    else:
        raise AssertionError("changed frozen feature inventory was accepted")


def _inventory_fixture():
    cases = ("c0", "c1")
    calls = (1, 2)
    arms = ("baseline", "static", "selected")
    components = d076.parent.COMPONENTS
    candidate = Candidate(key="rank8_gain0p125_raw", rank=8, gain=0.125)
    candidates = (Candidate(key="zero", rank=0, gain=0.0), candidate)
    selections = [
        {"case_id": "c0", "selected_nonzero": True},
        {"case_id": "c1", "selected_nonzero": False},
    ]
    sequence_keys = {
        ("c0", "baseline", "corrected_defect_on_own_recurrent_inputs"),
        ("c0", "static", "corrected_defect_on_own_recurrent_inputs"),
        ("c0", "static", "base_defect_on_same_corrected_inputs"),
        ("c0", "selected", "corrected_defect_on_own_recurrent_inputs"),
        ("c0", "selected", "base_defect_on_same_corrected_inputs"),
        ("c1", "baseline", "corrected_defect_on_own_recurrent_inputs"),
        ("c1", "static", "corrected_defect_on_own_recurrent_inputs"),
        ("c1", "static", "base_defect_on_same_corrected_inputs"),
        ("c1", "selected", "corrected_defect_on_own_recurrent_inputs"),
    }
    base_budget_fields = (
        "base_defect_same_corrected_input",
        "correction",
        "corrected_defect",
        "cumulative_correction",
        "state_error",
    )
    budget_keys = {
        (case, arm, call, field, component)
        for case in cases
        for arm in arms
        for call in calls
        for field in base_budget_fields
        for component in components
    }
    budget_keys.update(
        {
            (case, arm, call, field, component)
            for case, arm in (("c0", "static"), ("c0", "selected"), ("c1", "static"))
            for call in calls
            for field in (
                "correction_constant_mode",
                "correction_nonconstant_modes",
            )
            for component in components
        }
    )
    projection_keys = {
        (case, arm, call)
        for case, arm in (("c0", "static"), ("c0", "selected"), ("c1", "static"))
        for call in calls
    }
    outputs = {
        "evaluation_probe_features.csv": [
            {"case_id": case, "candidate": candidate.key} for case in cases
        ],
        "evaluation_controller_choices.csv": [
            {"case_id": case, "candidate": candidate.key} for case in cases
        ],
        "evaluation_controller_neighbors.csv": [
            {
                "case_id": case,
                "candidate": candidate.key,
                "neighbor_rank": rank,
            }
            for case in cases
            for rank in range(1, d076.PROBE_K + 1)
        ],
        "evaluation_case_summary.csv": [
            {"case_id": case, "arm": arm} for case in cases for arm in arms
        ],
        "evaluation_comparisons.csv": [
            {"case_id": case, "comparison": comparison}
            for case in cases
            for comparison in (
                "selected_vs_zero",
                "selected_vs_static",
                "static_vs_zero",
            )
        ],
        "evaluation_probe_replay.csv": [{"case_id": case} for case in cases],
        "evaluation_call_metrics.csv": [
            {"case_id": case, "arm": arm, "call": call}
            for case in cases
            for arm in arms
            for call in calls
        ],
        "completion.csv": [
            {"case_id": case, "arm": arm, "call": call, "accepted": True}
            for case in cases
            for arm in arms
            for call in calls
        ],
        "correction_integral_audit.csv": [
            {
                "case_id": case,
                "arm": arm,
                "call": call,
                "component": component,
            }
            for case in cases
            for arm in arms
            for call in calls
            for component in components
        ],
        "sequence_summaries.csv": [
            {"case_id": case, "arm": arm, "defect_kind": kind}
            for case, arm, kind in sequence_keys
        ],
        "sequence_time_metrics.csv": [
            {
                "case_id": case,
                "arm": arm,
                "defect_kind": kind,
                "step": call,
            }
            for case, arm, kind in sequence_keys
            for call in calls
        ],
        "lag_correlations.csv": [
            {"case_id": case, "arm": arm, "defect_kind": kind, "lag": 1}
            for case, arm, kind in sequence_keys
        ],
        "pod_summaries.csv": [
            {
                "case_id": case,
                "arm": arm,
                "defect_kind": kind,
                "centering": centering,
            }
            for case, arm, kind in sequence_keys
            for centering in ("uncentered", "centered")
        ],
        "signed_component_budgets.csv": [
            {
                "case_id": case,
                "arm": arm,
                "call": call,
                "field": field,
                "component": component,
            }
            for case, arm, call, field, component in budget_keys
        ],
        "projection_metrics.csv": [
            {"case_id": case, "arm": arm, "call": call}
            for case, arm, call in projection_keys
        ],
        "projection_component_metrics.csv": [
            {
                "case_id": case,
                "arm": arm,
                "call": call,
                "component": component,
            }
            for case, arm, call in projection_keys
            for component in components
        ],
        "projection_status.csv": [{"case_id": case} for case in cases],
        "visual_payload_inventory.csv": [{"case_id": "c0"}],
    }
    return outputs, cases, candidates, selections


def test_evaluation_inventory_is_exact_and_detects_missing_rows() -> None:
    outputs, cases, candidates, selections = _inventory_fixture()
    checks = d076._evaluation_inventory_checks(
        outputs,
        evaluation_ids=cases,
        candidates=candidates,
        selection_rows=selections,
        rollout_calls=2,
        visual_case_ids=("c0",),
    )
    assert checks["row_inventories_pass"]
    outputs["signed_component_budgets.csv"].pop()
    checks = d076._evaluation_inventory_checks(
        outputs,
        evaluation_ids=cases,
        candidates=candidates,
        selection_rows=selections,
        rollout_calls=2,
        visual_case_ids=("c0",),
    )
    assert not checks["budget_inventory_pass"]
    assert not checks["row_inventories_pass"]


def test_promotion_requires_adaptive_and_static_no_harm_gates() -> None:
    cases = [f"c{index}" for index in range(6)]
    selections = [
        {"case_id": case, "selected_gain": 0.125 if index < 4 else 0.0}
        for index, case in enumerate(cases)
    ]
    comparisons = []
    for case in cases:
        comparisons.extend(
            (
                {
                    "case_id": case,
                    "comparison": "selected_vs_zero",
                    "endpoint_state_ratio": 0.94,
                    "residual_rms_ratio": 0.99,
                    "maximum_control_ratio": 1.05,
                },
                {
                    "case_id": case,
                    "comparison": "selected_vs_static",
                    "endpoint_state_ratio": 0.97,
                    "residual_rms_ratio": 1.0,
                    "maximum_control_ratio": 1.0,
                },
            )
        )
    promotion = d076._promotion(comparisons, selections)
    assert promotion["passed"]
    comparisons[0]["residual_rms_ratio"] = 1.0001
    promotion = d076._promotion(comparisons, selections)
    assert not promotion["passed"]
    assert not promotion["all_residual_no_harm"]


def test_probe_replay_checks_absolute_and_relative_prefix_identity() -> None:
    states = np.zeros((3, 2, 4), dtype=np.float64)
    probe = d076.ProbeRollout(
        candidate=Candidate(key="zero", rank=0, gain=0.0),
        complete=True,
        states=states,
        corrections=np.zeros((2, 2, 4)),
        admissibility_rows=(),
    )
    rollout = SimpleNamespace(complete=True, states=states.copy())
    exact = d076._probe_replay(probe, rollout, probe_calls=2)
    assert exact["passed"]
    rollout.states[2, 0, 0] = 1.0e-3
    failed = d076._probe_replay(probe, rollout, probe_calls=2)
    assert not failed["passed"]
