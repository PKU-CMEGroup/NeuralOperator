from __future__ import annotations

import numpy as np
import pytest

from utility.time_dependent_no.pcno_cross_resolution_teacher_forced import (
    build_fixed_cosine_projector,
)
from utility.time_dependent_no.pcno_propagated_sensitivity import (
    PolicyRecord,
    grouped_directional_crossfit,
    grouped_gain_threshold_crossfit,
    modal_partition_fields,
    propagated_lookahead,
    weighted_quadratic_statistics,
)


def _projector():
    nx, ny = 8, 4
    x = (np.arange(nx, dtype=np.float64) + 0.5) * (2.0 / nx)
    y = (np.arange(ny, dtype=np.float64) + 0.5) * (1.0 / ny)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    nodes = np.column_stack((xx.reshape(-1), yy.reshape(-1)))
    volumes = np.full(nx * ny, 2.0 / (nx * ny), dtype=np.float64)
    node_type = np.zeros(nx * ny, dtype=np.int64)
    node_type[:nx] = 1
    projector = build_fixed_cosine_projector(
        nodes,
        volumes,
        node_type,
        rank=8,
        domain_bounds=(0.0, 2.0, 0.0, 1.0),
    )
    return volumes, projector


def test_propagated_lookahead_is_target_free_and_uses_two_calls():
    raw = np.arange(24, dtype=np.float64).reshape(6, 4)
    correction = np.full_like(raw, 0.25)
    calls = []

    def predictor(state):
        calls.append(np.array(state, copy=True))
        return 1.5 * state - 2.0

    result = propagated_lookahead(raw, correction, predictor=predictor)

    assert len(calls) == 2
    np.testing.assert_array_equal(calls[0], raw)
    np.testing.assert_array_equal(calls[1], raw + correction)
    np.testing.assert_allclose(result.response, 1.5 * correction)


def test_zero_correction_has_zero_deterministic_response():
    raw = np.arange(12, dtype=np.float64).reshape(3, 4)
    result = propagated_lookahead(
        raw,
        np.zeros_like(raw),
        predictor=lambda state: np.square(state),
    )
    np.testing.assert_array_equal(result.response, np.zeros_like(raw))


def test_weighted_quadratic_identity_and_directional_coefficient():
    error = np.asarray([[2.0, -1.0], [1.0, 3.0]], dtype=np.float64)
    response = -0.25 * error
    result = weighted_quadratic_statistics(
        error,
        response,
        volumes=np.asarray([[1.0], [3.0]]),
        component_scale=np.asarray([2.0, 0.5]),
    )

    assert result["status"] == "ok"
    assert result["quadratic_closure_abs"] <= 1.0e-14
    assert result["directional_gamma"] == pytest.approx(-4.0)
    assert result["cosine"] == pytest.approx(-1.0)


def test_weighted_quadratic_reports_empty_region():
    result = weighted_quadratic_statistics(
        np.ones((3, 2)),
        np.ones((3, 2)),
        volumes=np.ones(3),
        component_scale=np.ones(2),
        mask=np.zeros(3, dtype=bool),
    )
    assert result["status"] == "small_measure"
    assert result["corrected_energy"] is None


def test_modal_partition_reconstructs_full_field():
    rng = np.random.default_rng(3)
    _, projector = _projector()
    field = rng.normal(size=(32, 4))
    parts = modal_partition_fields(
        field,
        projector,
        component_scale=np.asarray([0.5, 1.0, 1.5, 2.0]),
    )
    reconstructed = sum(
        parts[name]
        for name in (
            "constant",
            "low_active_modes_1_3",
            "upper_active_modes_4_7",
            "rank8_inactive",
            "high_rank_remainder",
        )
    )
    np.testing.assert_allclose(reconstructed, field, atol=1.0e-12, rtol=0.0)
    np.testing.assert_allclose(
        parts["sp19_active"],
        parts["low_active_modes_1_3"] + parts["upper_active_modes_4_7"],
        atol=1.0e-12,
        rtol=0.0,
    )


def _policy_fixture(*, harmful_controls: bool = False):
    groups = {f"g{group}": (f"case{group}a", f"case{group}b") for group in range(3)}
    records = []
    for group_id, cases in groups.items():
        for case_index, case_id in enumerate(cases):
            for input_call in (0, 1):
                low_gain = (case_index + input_call) % 2 == 0
                gain = 0.1 if low_gain else 1.0
                corrected_state = 0.64 if low_gain else 1.44
                corrected_control = (
                    1.21
                    if harmful_controls and low_gain
                    else (0.81 if low_gain else 1.21)
                )
                records.append(
                    PolicyRecord(
                        case_id=case_id,
                        group_id=group_id,
                        input_call=input_call,
                        response_gain=gain,
                        raw_state_energy=1.0,
                        corrected_state_energy=corrected_state,
                        raw_control_energy={"integral::density": 1.0},
                        corrected_control_energy={
                            "integral::density": corrected_control
                        },
                        corrected_valid=True,
                    )
                )
    return groups, records


def test_grouped_gain_selector_uses_training_groups_only():
    groups, records = _policy_fixture()
    result = grouped_gain_threshold_crossfit(
        records,
        expected_groups=groups,
        expected_input_calls=(0, 1),
    )

    assert len(result["folds"]) == 3
    assert all(
        fold["selected_name"] in {"q25", "q50", "q75"} for fold in result["folds"]
    )
    assert result["oof"]["state_rms_ratio"] < 1.0
    assert result["oof"]["controls"]["integral::density"]["rms_ratio"] <= 1.0


def test_grouped_gain_selector_fails_closed_when_control_harms():
    groups, records = _policy_fixture(harmful_controls=True)
    result = grouped_gain_threshold_crossfit(
        records,
        expected_groups=groups,
        expected_input_calls=(0, 1),
    )
    assert all(fold["selected_name"] == "never" for fold in result["folds"])
    assert result["oof"]["coverage"] == 0.0


def test_grouped_gain_selector_rejects_incomplete_inventory():
    groups, records = _policy_fixture()
    with pytest.raises(ValueError, match="inventory"):
        grouped_gain_threshold_crossfit(
            records[:-1],
            expected_groups=groups,
            expected_input_calls=(0, 1),
        )


def test_directional_crossfit_recovers_exact_scalar_relation():
    groups = {f"g{group}": (f"case{group}",) for group in range(3)}
    rows = [
        {
            "case_id": case_id,
            "group_id": group_id,
            "input_call": input_call,
            "view": "full",
            "status": "ok",
            "error_energy": 4.0,
            "response_energy": 1.0,
            "cross": -2.0,
            "corrected_energy": 1.0,
        }
        for group_id, cases in groups.items()
        for case_id in cases
        for input_call in (0, 1)
    ]
    result = grouped_directional_crossfit(
        rows,
        expected_groups=groups,
        expected_input_calls=(0, 1),
    )

    summary = result["views"]["full"]
    assert summary["fold_gamma_min"] == pytest.approx(-2.0)
    assert summary["fold_gamma_max"] == pytest.approx(-2.0)
    assert summary["oof_prediction_r2"] == pytest.approx(1.0)
    assert summary["oof_clipped_dose_rms_ratio"] == pytest.approx(0.5)
    assert summary["median_case_cosine"] == pytest.approx(-1.0)
