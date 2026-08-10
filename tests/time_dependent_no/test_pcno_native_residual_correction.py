from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import scripts.time_dependent_no.evaluate_pcno_native_residual_correction as d074
from scripts.time_dependent_no.evaluate_pcno_native_residual_correction import (
    Candidate,
    RolloutResult,
    candidate_inventory,
    select_candidate,
)
from utility.time_dependent_no.pcno_resolution_transfer import make_model_sample


def _result(
    candidate: Candidate,
    *,
    state_error: float,
    residual_rms: float,
    control: float,
    complete: bool = True,
    family: str | None = None,
) -> RolloutResult:
    controls = (
        {name: control for name in d074.expected_control_keys(family)}
        if family is not None
        else {"structure": control}
    )
    return RolloutResult(
        candidate=candidate,
        complete=complete,
        valid_length=2 if complete else 1,
        states=np.empty((0, 0, 4)),
        base_defects=np.empty((0, 0, 4)),
        corrections=np.empty((0, 0, 4)),
        defects=np.empty((0, 0, 4)),
        truth_increments=np.empty((0, 0, 4)),
        admissibility_rows=[],
        summary={
            "complete": complete,
            "valid_length": 2 if complete else 1,
            "final_state_error": state_error,
            "residual_rms": residual_rms,
            "controls": controls,
        },
    )


def test_candidate_inventory_has_one_zero_and_sixteen_unique_nonzero_arms() -> None:
    candidates = candidate_inventory()
    assert len(candidates) == 17
    assert candidates[0] == Candidate(key="zero", rank=0, gain=0.0)
    assert len({candidate.key for candidate in candidates}) == 17
    assert all(candidate.gain > 0.0 for candidate in candidates[1:])
    assert candidate_inventory(smoke=True) == (
        Candidate(key="zero", rank=0, gain=0.0),
        Candidate(key="rank8_gain1", rank=8, gain=1.0),
    )


def test_d075_inventory_is_zero_plus_matched_rank8_policy_matrix() -> None:
    candidates = candidate_inventory(experiment_contract="d075_integral_neutral")
    assert len(candidates) == 13
    assert candidates[0] == Candidate(key="zero", rank=0, gain=0.0)
    positive = candidates[1:]
    assert {candidate.rank for candidate in positive} == {8}
    assert {candidate.gain for candidate in positive} == {0.125, 0.25, 0.5, 1.0}
    assert {candidate.correction_policy for candidate in positive} == set(
        d074.CORRECTION_POLICIES
    )
    assert len({candidate.key for candidate in candidates}) == len(candidates)

    smoke = candidate_inventory(
        smoke=True,
        experiment_contract="d075_integral_neutral",
    )
    assert len(smoke) == 4
    assert smoke[0].is_zero
    assert {candidate.gain for candidate in smoke[1:]} == {0.5}
    assert {candidate.correction_policy for candidate in smoke[1:]} == set(
        d074.CORRECTION_POLICIES
    )


def test_integral_policies_close_on_type0_support_and_are_idempotent() -> None:
    case = SimpleNamespace(
        family="dynamic_fv",
        nodes=np.zeros((4, 2)),
        weights=np.asarray([1.0, 2.0, 3.0, 4.0]),
        physical_node_type=np.asarray([0, 0, 0, 1]),
        residual_scale=np.asarray([2.0, 3.0, 4.0, 5.0]),
    )
    sequence = np.arange(2 * 4 * 4, dtype=np.float64).reshape(2, 4, 4)
    sequence[:, 3] = 0.0

    raw = d074._apply_integral_policy(
        sequence,
        case,
        correction_policy="raw",
    )
    assert raw is sequence

    energy = d074._apply_integral_policy(
        sequence,
        case,
        correction_policy="energy_integral_neutral",
    )
    np.testing.assert_array_equal(energy[:, :, :3], sequence[:, :, :3])
    np.testing.assert_array_equal(energy[:, 3], 0.0)
    energy_integral = np.einsum("n,tn->t", case.weights, energy[:, :, 3])
    np.testing.assert_allclose(energy_integral, 0.0, atol=1.0e-13)
    energy_repeat = d074._apply_integral_policy(
        energy,
        case,
        correction_policy="energy_integral_neutral",
    )
    np.testing.assert_allclose(energy_repeat, energy, atol=1.0e-13)

    all_neutral = d074._apply_integral_policy(
        sequence,
        case,
        correction_policy="all_integrals_neutral",
    )
    np.testing.assert_array_equal(all_neutral[:, 3], 0.0)
    all_integrals = np.einsum("n,tnc->tc", case.weights, all_neutral)
    np.testing.assert_allclose(all_integrals, 0.0, atol=1.0e-13)
    all_repeat = d074._apply_integral_policy(
        all_neutral,
        case,
        correction_policy="all_integrals_neutral",
    )
    np.testing.assert_allclose(all_repeat, all_neutral, atol=1.0e-13)


def test_candidate_bias_adjusts_constant_coefficients_and_reconstructs() -> None:
    case = SimpleNamespace(
        family="dynamic_fv",
        nodes=np.asarray([[0.25, 0.25], [0.75, 0.25], [1.25, 0.75], [1.75, 0.75]]),
        weights=np.asarray([1.0, 2.0, 3.0, 4.0]),
        physical_node_type=np.asarray([0, 0, 0, 1]),
        residual_scale=np.asarray([2.0, 3.0, 4.0, 5.0]),
    )
    coefficients = np.arange(2 * 2 * 4, dtype=np.float64).reshape(2, 2, 4) / 10.0
    candidate = Candidate(
        key="rank2_gain0p5_all_integrals_neutral",
        rank=2,
        gain=0.5,
        correction_policy="all_integrals_neutral",
    )
    basis, bias, adjusted = d074._candidate_bias_sequence(
        case,
        coefficients,
        candidate,
    )
    reconstructed = d074.reconstruct_coefficient_sequence(
        adjusted,
        basis,
        component_scale=case.residual_scale,
    )
    reconstructed[:, case.physical_node_type != 0] = 0.0
    np.testing.assert_allclose(reconstructed, bias, atol=1.0e-13)
    np.testing.assert_allclose(
        np.einsum("n,tnc->tc", case.weights, bias),
        0.0,
        atol=1.0e-13,
    )


def test_selector_fail_closes_projected_candidate_without_integral_audit() -> None:
    zero = Candidate(key="zero", rank=0, gain=0.0)
    projected = Candidate(
        key="rank8_gain0p5_all_integrals_neutral",
        rank=8,
        gain=0.5,
        correction_policy="all_integrals_neutral",
    )
    baseline = _result(zero, state_error=10.0, residual_rms=5.0, control=2.0)
    corrected = _result(projected, state_error=9.0, residual_rms=4.0, control=2.0)
    selected, rows, _, _ = select_candidate(
        (zero, projected),
        ("case",),
        {"zero": {"case": baseline}, projected.key: {"case": corrected}},
    )
    assert selected is zero
    projected_row = next(row for row in rows if row["candidate"] == projected.key)
    assert not projected_row["correction_integral_contract_pass"]
    assert not projected_row["eligible"]

    corrected.summary["correction_integral_audit"] = {
        "maximum_required_integral_closure": 1.0e-15,
        "passed": True,
    }
    selected, _, _, _ = select_candidate(
        (zero, projected),
        ("case",),
        {"zero": {"case": baseline}, projected.key: {"case": corrected}},
    )
    assert selected is projected


def test_selector_uses_complete_case_matrix_and_deterministic_complexity_tie() -> None:
    zero = Candidate(key="zero", rank=0, gain=0.0)
    rank2 = Candidate(key="rank2_gain1", rank=2, gain=1.0)
    rank4 = Candidate(key="rank4_gain0p25", rank=4, gain=0.25)
    cases = ("a", "b")
    baseline = {
        case_id: _result(zero, state_error=10.0, residual_rms=5.0, control=2.0)
        for case_id in cases
    }
    rollouts = {
        "zero": baseline,
        "rank2_gain1": {
            "a": _result(rank2, state_error=9.0, residual_rms=4.8, control=2.0),
            "b": _result(rank2, state_error=9.2, residual_rms=4.9, control=2.0),
        },
        "rank4_gain0p25": {
            "a": _result(rank4, state_error=9.0, residual_rms=4.8, control=2.0),
            "b": _result(rank4, state_error=9.2, residual_rms=4.9, control=2.0),
        },
    }
    selected, rows, summaries, metric_rows = select_candidate(
        (zero, rank2, rank4), cases, rollouts
    )
    assert selected is rank2
    assert len(rows) == 6
    assert len(summaries) == 3
    assert len(metric_rows) == 3 * 2 * 3
    assert all(row["eligible"] for row in rows)


def test_selector_falls_back_to_zero_and_rejects_unresolved_denominator() -> None:
    zero = Candidate(key="zero", rank=0, gain=0.0)
    harmful = Candidate(key="rank2_gain1", rank=2, gain=1.0)
    baseline_result = _result(zero, state_error=1.0, residual_rms=2.0, control=3.0)
    selected, _, _, _ = select_candidate(
        (zero, harmful),
        ("case",),
        {
            "zero": {"case": baseline_result},
            "rank2_gain1": {
                "case": _result(harmful, state_error=1.1, residual_rms=2.0, control=3.0)
            },
        },
    )
    assert selected is zero

    unresolved = _result(
        zero,
        state_error=d074.RATIO_DENOMINATOR_FLOOR,
        residual_rms=2.0,
        control=3.0,
    )
    with pytest.raises(ValueError, match="numerically unresolved"):
        select_candidate(
            (zero, harmful),
            ("case",),
            {
                "zero": {"case": unresolved},
                "rank2_gain1": {
                    "case": _result(
                        harmful, state_error=0.5, residual_rms=1.0, control=2.0
                    )
                },
            },
        )


def test_selector_lower_gain_key_ties_and_input_order_are_deterministic() -> None:
    zero = Candidate(key="zero", rank=0, gain=0.0)
    high_gain = Candidate(key="rank2_gain0p5", rank=2, gain=0.5)
    key_b = Candidate(key="rank2_gain0p25_b", rank=2, gain=0.25)
    key_a = Candidate(key="rank2_gain0p25_a", rank=2, gain=0.25)
    candidates = (zero, high_gain, key_b, key_a)
    rollouts = {
        "zero": {"case": _result(zero, state_error=10.0, residual_rms=5.0, control=2.0)}
    }
    for candidate in candidates[1:]:
        rollouts[candidate.key] = {
            "case": _result(candidate, state_error=9.0, residual_rms=4.0, control=2.0)
        }
    selected, _, _, _ = select_candidate(candidates, ("case",), rollouts)
    reversed_selected, _, _, _ = select_candidate(
        tuple(reversed(candidates)), ("case",), rollouts
    )
    assert selected is key_a
    assert reversed_selected is key_a


def test_compact_selector_summaries_are_exactly_selection_equivalent() -> None:
    zero = Candidate(key="zero", rank=0, gain=0.0)
    corrected = Candidate(key="rank2_gain1", rank=2, gain=1.0)
    incomplete = Candidate(key="rank4_gain1", rank=4, gain=1.0)
    candidates = (zero, corrected, incomplete)
    cases = ("a", "b")
    full = {
        "zero": {
            case_id: _result(
                zero,
                state_error=10.0,
                residual_rms=5.0,
                control=2.0,
            )
            for case_id in cases
        },
        "rank2_gain1": {
            "a": _result(
                corrected,
                state_error=9.0,
                residual_rms=4.5,
                control=2.0,
            ),
            "b": _result(
                corrected,
                state_error=9.2,
                residual_rms=4.6,
                control=2.0,
            ),
        },
        "rank4_gain1": {
            "a": _result(
                incomplete,
                state_error=8.0,
                residual_rms=4.0,
                control=2.0,
            ),
            "b": _result(
                incomplete,
                state_error=8.0,
                residual_rms=4.0,
                control=2.0,
                complete=False,
            ),
        },
    }
    compact = {
        candidate: {
            case_id: d074._selector_rollout_summary(result)
            for case_id, result in by_case.items()
        }
        for candidate, by_case in full.items()
    }
    assert select_candidate(candidates, cases, compact) == select_candidate(
        candidates, cases, full
    )


def test_crossfit_excludes_each_held_out_case_from_its_bias(monkeypatch) -> None:
    zero = Candidate(key="zero", rank=0, gain=0.0)
    corrected = Candidate(key="rank8_gain1", rank=8, gain=1.0)
    cases = [
        SimpleNamespace(case_id=value, family="dynamic_fv") for value in ("a", "b", "c")
    ]
    case_values = {"a": 1.0, "b": 3.0, "c": 8.0}
    observed_bias = {}

    def fake_fit(model, case, ranks):
        assert tuple(ranks) == (8,)
        return {8: np.full((1, 8, 4), case_values[case.case_id])}

    def fake_bias(case, coefficients, *, rank):
        assert rank == 8
        observed_bias[case.case_id] = float(coefficients[0, 0, 0])
        return np.empty((0, 0)), coefficients

    def fake_rollout(
        model,
        case,
        candidate,
        *,
        bias_sequence,
        shock_quantile,
    ):
        del model, shock_quantile
        if candidate.is_zero:
            assert bias_sequence is None
            return _result(
                candidate,
                state_error=10.0,
                residual_rms=5.0,
                control=2.0,
                family=case.family,
            )
        assert bias_sequence is not None
        return _result(
            candidate,
            state_error=9.0,
            residual_rms=4.0,
            control=2.0,
            family=case.family,
        )

    monkeypatch.setattr(d074, "_fit_case_coefficients", fake_fit)
    monkeypatch.setattr(d074, "_bias_sequence", fake_bias)
    monkeypatch.setattr(d074, "_rollout_candidate", fake_rollout)
    selected, coefficients, rows, _, metric_rows = d074._crossfit_calibration(
        object(),
        cases,
        (zero, corrected),
        shock_quantile=0.9,
        smoke=False,
    )
    assert selected is corrected
    assert set(coefficients) == {"a", "b", "c"}
    assert len(rows) == 6
    assert len(metric_rows) == 2 * 3 * (
        len(d074.expected_control_keys("dynamic_fv")) + 2
    )
    assert observed_bias == {
        "a": (3.0 + 8.0) / 2.0,
        "b": (1.0 + 8.0) / 2.0,
        "c": (1.0 + 3.0) / 2.0,
    }


def test_crossfit_retains_only_compact_selector_summaries(monkeypatch) -> None:
    zero = Candidate(key="zero", rank=0, gain=0.0)
    corrected = Candidate(key="rank8_gain1", rank=8, gain=1.0)
    cases = [
        SimpleNamespace(case_id=value, family="dynamic_fv") for value in ("a", "b")
    ]

    def fake_fit(model, case, ranks):
        del model, case
        assert tuple(ranks) == (8,)
        return {8: np.ones((1, 8, 4))}

    def fake_bias(case, coefficients, *, rank):
        del case
        assert rank == 8
        return np.empty((0, 0)), coefficients

    def fake_rollout(
        model,
        case,
        candidate,
        *,
        bias_sequence,
        shock_quantile,
    ):
        del model, bias_sequence, shock_quantile
        return _result(
            candidate,
            state_error=10.0 if candidate.is_zero else 9.0,
            residual_rms=5.0 if candidate.is_zero else 4.0,
            control=2.0,
            family=case.family,
        )

    def fake_select(candidates, case_ids, rollouts, *, family):
        assert tuple(case_ids) == ("a", "b")
        assert family == "dynamic_fv"
        assert set(rollouts) == {"zero", "rank8_gain1"}
        assert all(
            isinstance(result, d074.SelectorRolloutSummary)
            and set(result.summary)
            == {
                "final_state_error",
                "residual_rms",
                "controls",
                "correction_integral_audit",
            }
            and not hasattr(result, "states")
            for by_case in rollouts.values()
            for result in by_case.values()
        )
        return candidates[0], [], [], []

    monkeypatch.setattr(d074, "_fit_case_coefficients", fake_fit)
    monkeypatch.setattr(d074, "_bias_sequence", fake_bias)
    monkeypatch.setattr(d074, "_rollout_candidate", fake_rollout)
    monkeypatch.setattr(d074, "select_candidate", fake_select)
    selected, coefficients, rows, summaries, metric_rows = d074._crossfit_calibration(
        object(),
        cases,
        (zero, corrected),
        shock_quantile=0.9,
        smoke=False,
    )
    assert selected is zero
    assert set(coefficients) == {"a", "b"}
    assert rows == []
    assert summaries == []
    assert metric_rows == []


def test_rollout_uses_reference_increments_and_closes_recurrence(monkeypatch) -> None:
    reference_states = np.asarray(
        [
            np.zeros((1, 4)),
            np.ones((1, 4)),
            np.full((1, 4), 3.0),
        ]
    )
    case = SimpleNamespace(
        reference_states=reference_states,
        gamma=1.4,
        weights=np.ones(1),
        residual_scale=np.ones(4),
    )
    increments = (1.2, 2.4)
    call = 0

    def fake_predict(model, case, current):
        nonlocal call
        del model, case
        prediction = current + increments[call] + 0.1 * np.square(current)
        call += 1
        return prediction

    monkeypatch.setattr(d074, "_predict", fake_predict)
    monkeypatch.setattr(
        d074,
        "_safe_admissibility_summary",
        lambda prediction, *, gamma: {
            "finite": bool(np.all(np.isfinite(prediction))),
            "admissible": True,
        },
    )
    monkeypatch.setattr(
        d074,
        "_summarize_rollout",
        lambda case, *, states, defects, complete, shock_quantile: {
            "complete": complete,
            "valid_length": defects.shape[0],
            "final_state_error": 0.0,
            "residual_rms": 0.0,
            "controls": {},
        },
    )

    zero = Candidate(key="zero", rank=0, gain=0.0)
    zero_result = d074._rollout_candidate(
        object(), case, zero, bias_sequence=None, shock_quantile=0.9
    )
    expected_truth = np.asarray([np.ones((1, 4)), np.full((1, 4), 2.0)])
    expected_zero_base_defects = np.asarray(
        [np.full((1, 4), 0.2), np.full((1, 4), 0.544)]
    )
    np.testing.assert_allclose(zero_result.truth_increments, expected_truth)
    np.testing.assert_allclose(zero_result.base_defects, expected_zero_base_defects)
    np.testing.assert_array_equal(zero_result.corrections, 0.0)
    np.testing.assert_allclose(zero_result.defects, expected_zero_base_defects)
    state_error = zero_result.states - reference_states
    np.testing.assert_allclose(np.diff(state_error, axis=0), zero_result.defects)

    call = 0
    corrected = Candidate(key="rank2_gain0p5", rank=2, gain=0.5)
    bias = np.asarray([np.full((1, 4), 0.4), np.full((1, 4), 0.2)])
    corrected_result = d074._rollout_candidate(
        object(), case, corrected, bias_sequence=bias, shock_quantile=0.9
    )
    expected_corrected_base_defects = np.asarray(
        [np.full((1, 4), 0.2), np.full((1, 4), 0.5)]
    )
    expected_correction = -0.5 * bias
    np.testing.assert_allclose(corrected_result.corrections, expected_correction)
    np.testing.assert_allclose(
        corrected_result.defects,
        expected_corrected_base_defects + expected_correction,
        atol=1.0e-15,
    )
    assert not np.allclose(
        corrected_result.base_defects[1], zero_result.base_defects[1]
    )
    corrected_state_error = corrected_result.states - reference_states
    np.testing.assert_allclose(
        np.diff(corrected_state_error, axis=0),
        corrected_result.defects,
        atol=1.0e-15,
    )


def _diagnostic_fixture():
    nodes = np.asarray([[0.0, 0.0], [0.5, 0.25], [1.0, 0.5], [1.5, 0.75]])
    weights = np.asarray([1.0, 2.0, 1.5, 0.5])
    node_type = np.asarray([0, 0, 0, 1])
    reference_states = np.asarray(
        [
            np.zeros((4, 4)),
            np.ones((4, 4)),
            np.full((4, 4), 3.0),
        ]
    )
    truth_increments = np.diff(reference_states, axis=0)
    basis = np.column_stack((np.ones(4), nodes[:, 0]))
    applied_coefficients = np.asarray(
        [
            np.vstack((np.full(4, 0.20), np.full(4, 0.05))),
            np.vstack((np.full(4, 0.10), np.full(4, -0.04))),
        ]
    )
    correction = np.einsum("nr,trc->tnc", basis, applied_coefficients)
    correction[:, node_type != 0] = 0.0
    base_defects = np.asarray([np.full((4, 4), 0.30), np.full((4, 4), -0.10)])
    defects = base_defects + correction
    errors = np.concatenate((np.zeros((1, 4, 4)), np.cumsum(defects, axis=0)))
    states = reference_states + errors
    selected = RolloutResult(
        candidate=Candidate(key="rank2_gain0p5", rank=2, gain=0.5),
        complete=True,
        valid_length=2,
        states=states,
        base_defects=base_defects,
        corrections=correction,
        defects=defects,
        truth_increments=truth_increments,
        admissibility_rows=[],
        summary={"complete": True},
    )
    baseline_errors = np.concatenate(
        (np.zeros((1, 4, 4)), np.cumsum(base_defects, axis=0))
    )
    baseline = RolloutResult(
        candidate=Candidate(key="zero", rank=0, gain=0.0),
        complete=True,
        valid_length=2,
        states=reference_states + baseline_errors,
        base_defects=base_defects,
        corrections=np.zeros_like(base_defects),
        defects=base_defects,
        truth_increments=truth_increments,
        admissibility_rows=[],
        summary={"complete": True},
    )
    case = SimpleNamespace(
        family="dynamic_fv",
        case_id="synthetic",
        resolution_name="250x100",
        reference_states=reference_states,
        physical_times=np.asarray([0.0, 0.02, 0.04]),
        nodes=nodes,
        edges=np.asarray([[0, 1], [1, 2], [2, 3]]),
        weights=weights,
        physical_node_type=node_type,
        state_scale=np.ones(4),
        residual_scale=np.ones(4),
        gamma=1.4,
    )
    return case, baseline, selected, basis, applied_coefficients


def test_same_input_diagnostics_close_and_preserve_family_support() -> None:
    case, _, selected, basis, applied_coefficients = _diagnostic_fixture()
    artifacts = d074._same_input_rows(
        case,
        selected,
        arm="selected",
        basis=basis,
        applied_coefficients=applied_coefficients,
    )
    assert len(artifacts["call_rows"]) == 2
    assert len(artifacts["sequence_summary_rows"]) == 2
    assert len(artifacts["sequence_time_rows"]) == 4
    assert len(artifacts["lag_rows"]) == 2
    assert len(artifacts["pod_rows"]) == 4
    assert len(artifacts["projection_rows"]) == 2
    assert len(artifacts["projection_component_rows"]) == 8
    assert len(artifacts["budget_rows"]) == 56
    for row in artifacts["call_rows"]:
        assert row["same_input_pointwise_closure_max"] <= 1.0e-15
        assert row["recurrence_closure_rms"] <= 1.0e-15
        assert "instant_defect_physical_rms__rho" in row
        assert row["correction_base_cosine_valid"]
    for row in artifacts["projection_rows"]:
        assert row["effective_rank"] == 2
        assert row["maximum_non_type0_correction"] == 0.0
        assert row["maximum_coefficient_reconstruction_error"] <= 1.0e-15
        assert row["correction_orthogonal_energy"] <= 1.0e-28
        assert row["correction_excluded_non_type0_energy"] == 0.0
        assert row["maximum_orthogonal_partition_change"] <= 1.0e-15
        assert row["maximum_excluded_partition_change"] == 0.0
    for row in artifacts["projection_component_rows"]:
        assert abs(row["constant_nonconstant_energy_closure"]) <= 1.0e-15
        np.testing.assert_allclose(
            row["correction_total_energy"],
            row["correction_constant_mode_energy"]
            + row["correction_nonconstant_modes_energy"]
            + row["twice_constant_nonconstant_inner"],
            atol=1.0e-15,
        )
    assert {row["quantity_semantics"] for row in artifacts["budget_rows"]} == {
        "signed physical-volume integral"
    }


def test_visual_payload_replays_cumulative_and_signed_growth(tmp_path) -> None:
    case, baseline, selected, _, _ = _diagnostic_fixture()
    path = tmp_path / "visual.npz"
    record = d074._save_visual_payload(path, case, baseline, selected)
    assert record["calls"] == 2
    with np.load(path, allow_pickle=False) as payload:
        assert payload["selected_candidate"].item() == "rank2_gain0p5"
        assert payload["selected_correction_policy"].item() == "raw"
        assert payload["expected_calls"].item() == 2
        assert payload["weight_semantics"].item() == "physical_cell_volume"
        cumulative = payload["cumulative_error"].astype(np.float64)
        defect = payload["corrected_defect"].astype(np.float64)
        np.testing.assert_allclose(cumulative, np.cumsum(defect, axis=0), atol=2.0e-7)
        contribution = payload["signed_growth_contribution"].astype(np.float64)
        growth = contribution.sum(axis=(1, 2))
        previous = np.concatenate((np.zeros_like(cumulative[:1]), cumulative[:-1]))
        direct = np.asarray(
            [
                np.sum(
                    case.weights[:, None]
                    * (np.square(cumulative[index]) - np.square(previous[index]))
                )
                / case.weights.sum()
                for index in range(2)
            ]
        )
        np.testing.assert_allclose(growth, direct, atol=2.0e-7)


def test_incomplete_candidate_has_no_horizon_ratio_and_cannot_evaluate() -> None:
    zero = Candidate(key="zero", rank=0, gain=0.0)
    failed = Candidate(key="rank2_gain1", rank=2, gain=1.0)
    baseline = _result(
        zero,
        state_error=1.0,
        residual_rms=2.0,
        control=3.0,
        family="dynamic_fv",
    )
    incomplete = _result(
        failed,
        state_error=0.1,
        residual_rms=0.2,
        control=0.3,
        complete=False,
        family="dynamic_fv",
    )
    selected, rows, _, metric_rows = select_candidate(
        (zero, failed),
        ("case",),
        {"zero": {"case": baseline}, "rank2_gain1": {"case": incomplete}},
        family="dynamic_fv",
    )
    assert selected is zero
    failed_row = next(row for row in rows if row["candidate"] == failed.key)
    assert failed_row["endpoint_state_ratio"] is None
    assert failed_row["residual_rms_ratio"] is None
    assert all(
        row["ratio"] is None for row in metric_rows if row["candidate"] == failed.key
    )
    with pytest.raises(ValueError, match="unequal-horizon"):
        d074._evaluation_gate(
            "dynamic_fv", {"case": baseline}, {"case": incomplete}, failed
        )


def test_family_global_quantities_and_control_inventories_are_distinct() -> None:
    errors = np.asarray(
        [
            [[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]],
            [[2.0, 4.0, 6.0, 8.0], [4.0, 6.0, 8.0, 10.0]],
        ]
    )
    dynamic = SimpleNamespace(
        family="dynamic_fv",
        weights=np.asarray([1.0, 3.0]),
        state_scale=np.ones(4),
    )
    bump = SimpleNamespace(
        family="bump",
        weights=dynamic.weights,
        state_scale=np.ones(4),
    )
    dynamic_controls, dynamic_budget = d074._global_quantity_metrics(errors, dynamic)
    bump_controls, bump_budget = d074._global_quantity_metrics(errors, bump)
    assert set(dynamic_controls) != set(bump_controls)
    assert set(dynamic_controls) <= set(d074.expected_control_keys("dynamic_fv"))
    assert set(bump_controls) <= set(d074.expected_control_keys("bump"))
    assert dynamic_budget["raw_final_signed__rho"] == 14.0
    assert bump_budget["raw_final_signed__rho"] == 3.5
    assert "nonphysical" in bump_budget["semantics"]


def _native_dynamic_case() -> SimpleNamespace:
    config, geometry, node_type, _ = d074._dynamic_native_geometry_contract()
    node_count = geometry.nodes.shape[0]
    return SimpleNamespace(
        family="dynamic_fv",
        case_id="native_contract_fixture",
        resolution_name="250x100",
        sample=make_model_sample(
            geometry,
            node_type,
            mach=config.shock_mach,
            device=torch.device("cpu"),
        ),
        reference_states=np.zeros((3, node_count, 4), dtype=np.float64),
        physical_times=np.asarray([0.0, 0.02, 0.04]),
        nodes=np.array(geometry.nodes, copy=True),
        edges=np.array(geometry.directed_edges, copy=True),
        weights=np.array(geometry.node_measures, copy=True).reshape(-1),
        physical_node_type=np.array(node_type, copy=True),
        state_scale=np.ones(4),
        residual_scale=np.ones(4),
        gamma=1.4,
    )


def test_dynamic_native_geometry_matches_frozen_250x100_contract() -> None:
    case = _native_dynamic_case()
    contract = d074._validate_case_contract(case)
    assert contract["node_type_meanings"] == {
        "0": "interior",
        "1": "y_symmetry_contact",
        "2": "x_extrapolation_contact",
        "3": "x_and_y_contact",
    }
    assert case.nodes.shape == (25000, 2)
    assert case.edges.shape == (99300, 2)
    assert contract["node_type_counts"] == {
        "0": 24304,
        "1": 496,
        "2": 196,
        "3": 4,
    }
    assert contract["total_weight"] == 2.0
    assert contract["geometry_contract"] == "regenerated_dynamic_fv_250x100_v1"
    assert contract["model_sample_geometry_exact"] is True
    for key, expected in d074.DYNAMIC_NATIVE_ARRAY_SHA256.items():
        assert contract[f"{key}_sha256"] == expected


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("node", "dynamic nodes differ"),
        ("node_swap", "dynamic nodes differ"),
        ("weight", "physical cell measures differ"),
        ("node_type", "dynamic node types differ"),
        ("edge", "dynamic directed edges differ"),
        ("sample_node", "sample tensor nodes differs"),
        ("sample_measure", "sample tensor node_measures differs"),
        ("sample_type", "sample tensor node_type differs"),
        ("sample_edge", "sample tensor directed_edges differs"),
    ),
)
def test_dynamic_native_geometry_contract_rejects_perturbations(
    mutation: str, message: str
) -> None:
    case = _native_dynamic_case()
    if mutation == "node":
        case.nodes[0, 0] += 1.0e-12
    elif mutation == "node_swap":
        case.nodes[[0, 1]] = case.nodes[[1, 0]]
    elif mutation == "weight":
        case.weights[0] += 1.0e-12
    elif mutation == "node_type":
        case.physical_node_type[0] = 0
    elif mutation == "edge":
        case.edges[0, 1] += 1
    elif mutation == "sample_node":
        case.sample["nodes"][0, 0, 0] += 1.0e-6
    elif mutation == "sample_measure":
        case.sample["node_measures"][0, 0, 0] += 1.0e-6
    elif mutation == "sample_type":
        case.sample["node_type"][0, 0] = 0
    elif mutation == "sample_edge":
        case.sample["directed_edges"][0, 0, 1] += 1
    else:  # pragma: no cover - parameter inventory is fixed above
        raise AssertionError(mutation)
    with pytest.raises(ValueError, match=message):
        d074._validate_case_contract(case)


def test_family_local_node_meaning_maps_remain_distinct() -> None:
    assert d074.NODE_TYPE_MEANINGS["dynamic_fv"] == {
        0: "interior",
        1: "y_symmetry_contact",
        2: "x_extrapolation_contact",
        3: "x_and_y_contact",
    }
    assert d074.NODE_TYPE_MEANINGS["bump"] == {
        0: "normal",
        1: "wall",
        2: "outflow",
        3: "inflow",
    }


def test_evaluation_requires_immutable_selector_freeze(tmp_path) -> None:
    selector = tmp_path / "selector.json"
    coefficients = tmp_path / "frozen_selected_coefficients.npz"
    selector.write_text("{}", encoding="utf-8")
    coefficients.write_bytes(b"frozen")
    digest = d074.sha256_file(selector)
    artifacts = {coefficients.name: d074.sha256_file(coefficients)}
    rows = [{"phase": "selector_frozen", "evaluation_targets_loaded": False}]
    d074._assert_selector_frozen_before_evaluation(selector, digest, artifacts, rows)
    coefficients.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="bundle artifact changed"):
        d074._assert_selector_frozen_before_evaluation(
            selector, digest, artifacts, rows
        )
    coefficients.write_bytes(b"frozen")
    selector.write_text('{"changed": true}', encoding="utf-8")
    with pytest.raises(ValueError, match="missing or changed"):
        d074._assert_selector_frozen_before_evaluation(
            selector, digest, artifacts, rows
        )
