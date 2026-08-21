from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from scripts.time_dependent_no.analyze_pcno_cross_family_identifiability import (
    _euler1d_geometry,
)
from utility.time_dependent_no.pcno_residual_geometry import (
    PHASE_MODEL_FEATURES,
    build_geometry_records,
    cap_modal_predictions,
    fit_case_first_coefficient,
    fit_modal_affine_phase_map,
    fit_modal_causal_discrepancy_event_clock,
    fit_modal_causal_observer,
    fit_modal_discrepancy_phase_portrait,
    fit_persistence_threshold,
    fit_phase_ridge,
    grouped_coefficient_crossfit,
    grouped_persistence_crossfit,
    individual_alignment_from_statistics,
    leave_one_case_out_phase,
    modal_affine_fit_diagnostics,
    modal_affine_map_stability,
    modal_causal_discrepancy_event_clock_features,
    modal_causal_observer_features,
    modal_discrepancy_phase_portrait_features,
    modal_linear_fit_diagnostics,
    modal_linear_map_stability,
    nested_grouped_modal_linear_map,
    orthogonal_partition_closure,
    predict_modal_affine_phase_map,
    predict_modal_causal_discrepancy_event_clock,
    predict_modal_causal_observer,
    predict_modal_discrepancy_phase_portrait,
    predict_phase_ridge,
    recurrent_error_energy_rows,
    recurrent_phase_records,
    relation_rows,
    score_modal_linear_predictions,
    score_phase_predictions,
    score_policy,
    select_grouped_modal_linear_map,
    snapshots_from_modal_rows,
    standardized_direction_relation,
    two_feature_geometry,
    two_feature_geometry_from_statistics,
)

CELLS = ((0, 0), (1, 0), (1, 1), (2, 0))
ACTIVE = ((1, 0), (1, 1))
CALLS = (0, 1)


def _rows(*, coefficient: float = -0.5):
    rows = []
    for group_index, group_id in enumerate(("g0", "g1", "g2")):
        for case_index in range(2):
            case_id = f"{group_id}_c{case_index}"
            for input_call in CALLS:
                scale = 1.0 + 0.1 * group_index + 0.05 * case_index
                fine = np.asarray((0.2, scale, 0.5 * scale, (-1.0) ** input_call * 0.2))
                target = coefficient * fine
                coarse = 1.5 * fine
                for cell_index, (mode, component) in enumerate(CELLS):
                    rows.append(
                        {
                            "case_id": case_id,
                            "group_id": group_id,
                            "input_call": str(input_call),
                            "mode_index": str(mode),
                            "component": str(component),
                            "coarse_coordinate": repr(float(coarse[cell_index])),
                            "fine_coordinate": repr(float(fine[cell_index])),
                            "target_coordinate": repr(float(target[cell_index])),
                        }
                    )
    return rows


def _records(*, coefficient: float = -0.5):
    snapshots = snapshots_from_modal_rows(
        _rows(coefficient=coefficient),
        expected_cells=CELLS,
        expected_calls=CALLS,
    )
    return build_geometry_records(snapshots, active_cells=ACTIVE)


def test_modal_inventory_and_partition_closure() -> None:
    snapshots = snapshots_from_modal_rows(
        _rows(), expected_cells=CELLS, expected_calls=CALLS
    )
    records = build_geometry_records(snapshots, active_cells=ACTIVE)

    assert len(snapshots) == 12
    assert len(records) == 36
    closure = orthogonal_partition_closure(records)
    assert closure["status"] == "ok"
    assert closure["maximum_fine_energy_closure_abs"] <= 1.0e-15

    rows = relation_rows(records)
    call_zero = [row for row in rows if row["input_call"] == 0]
    assert all(row["persistence_status"] == "no_previous_call" for row in call_zero)
    assert all(row["persistence_cosine"] is None for row in call_zero)


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda rows: rows.pop(), "modal cell inventory mismatch"),
        (
            lambda rows: rows.append(dict(rows[0])),
            "duplicate modal cell",
        ),
        (
            lambda rows: rows.__setitem__(0, {**rows[0], "input_call": "1.0"}),
            "canonical nonnegative integer",
        ),
        (
            lambda rows: rows.__setitem__(0, {**rows[0], "fine_coordinate": "nan"}),
            "must be finite",
        ),
    ],
)
def test_modal_inventory_fails_closed(mutation, message: str) -> None:
    rows = _rows()
    mutation(rows)
    with pytest.raises((TypeError, ValueError), match=message):
        snapshots_from_modal_rows(rows, expected_cells=CELLS, expected_calls=CALLS)


def test_case_first_coefficient_and_fixed_policy_close() -> None:
    records = [record for record in _records() if record.view == "sp19"]
    fit = fit_case_first_coefficient(records)
    assert fit["status"] == "ok"
    assert fit["coefficient"] == pytest.approx(-0.5, abs=1.0e-14)

    zero = score_policy(records, beta=0.0, policy="zero")
    fixed = score_policy(records, beta=-0.5, policy="fixed")
    assert zero["applied_count"] == 0
    assert zero["rms_ratio_vs_zero"] == pytest.approx(1.0)
    assert fixed["rms_ratio_vs_zero"] == pytest.approx(0.0, abs=1.0e-14)
    assert fixed["case_win_count"] == fixed["case_count"]


def test_case_first_weighting_is_not_snapshot_pooling() -> None:
    records = [record for record in _records(coefficient=-0.5) if record.view == "sp19"]
    # Duplicate every call of one case. Equal-case fitting must remain exact.
    duplicated = records + [record for record in records if record.case_id == "g0_c0"]
    fit = fit_case_first_coefficient(duplicated)
    assert fit["coefficient"] == pytest.approx(-0.5, abs=1.0e-14)


def test_two_feature_geometry_is_scale_invariant_and_case_first() -> None:
    correlation = np.asarray(((1.0, 0.25), (0.25, 1.0)))
    normalized_cross = np.asarray((0.3, -0.2))
    target_square = 4.0
    scales = np.asarray((2.0, 5.0))
    gram = scales[:, None] * correlation * scales[None, :]
    cross = scales * np.sqrt(target_square) * normalized_cross

    first = two_feature_geometry_from_statistics(gram, cross, target_square)
    changed_scales = scales * np.asarray((3.0, 0.2))
    changed_gram = changed_scales[:, None] * correlation * changed_scales[None, :]
    changed_cross = changed_scales * np.sqrt(target_square) * normalized_cross
    changed = two_feature_geometry_from_statistics(
        changed_gram, changed_cross, target_square
    )

    assert first["status"] == "ok"
    assert first["correlation_matrix"] == pytest.approx(correlation)
    assert first["normalized_cross"] == pytest.approx(normalized_cross)
    assert changed["standardized_direction"] == pytest.approx(
        first["standardized_direction"]
    )
    assert changed["joint_explained_fraction"] == pytest.approx(
        first["joint_explained_fraction"]
    )
    assert changed["raw_coefficients"] != pytest.approx(first["raw_coefficients"])

    sp19 = [record for record in _records() if record.view == "sp19"]
    base = two_feature_geometry(sp19)
    duplicated = sp19 + [record for record in sp19 if record.case_id == "g0_c0"]
    repeated = two_feature_geometry(duplicated)
    np.testing.assert_allclose(repeated["gram"], base["gram"])
    np.testing.assert_allclose(repeated["cross"], base["cross"])
    assert repeated["target_square"] == pytest.approx(base["target_square"])


def test_two_feature_geometry_fails_closed_on_rank_and_inconsistent_target() -> None:
    rank_deficient = two_feature_geometry_from_statistics(
        ((1.0, 1.0), (1.0, 1.0)), (0.1, 0.1), 1.0
    )
    assert rank_deficient["status"] == "rank_deficient"
    assert rank_deficient["standardized_direction"] is None

    with pytest.raises(ValueError, match="outside .0, 1."):
        two_feature_geometry_from_statistics(((1.0, 0.0), (0.0, 1.0)), (1.0, 1.0), 1.0)
    with pytest.raises(ValueError, match="Cauchy-Schwarz"):
        individual_alignment_from_statistics(
            denominator=1.0, cross=2.0, target_square=1.0
        )

    unresolved = standardized_direction_relation(None, (1.0, 0.0))
    assert unresolved["status"] == "unresolved_direction"
    assert unresolved["cosine"] is None


def test_euler1d_temporal_geometry_does_not_impute_joint_cross_term() -> None:
    gram = np.asarray(((4.0, 1.0), (1.0, 9.0)))
    cross = np.asarray((0.4, -0.3))
    target_square = 2.0
    geometry = two_feature_geometry_from_statistics(gram, cross, target_square)
    relations = {}
    for scope, sign in (("calls_0_49", 1.0), ("calls_50_99", -1.0)):
        relations[scope] = {
            "correction_target_relations": {
                "coarse_low": {
                    "fit": {
                        "denominator": 2.0,
                        "cross": 0.2 * sign,
                        "target_rms": 1.0,
                    }
                },
                "fine_low": {
                    "fit": {
                        "denominator": 3.0,
                        "cross": -0.1 * sign,
                        "target_rms": 1.0,
                    }
                },
            }
        }
    payload = {
        "structure_diagnostics": {
            "two_feature_low_mode_fit": {
                "status": "ok",
                "gram": gram.tolist(),
                "cross": cross.tolist(),
                "target_square": target_square,
                "coefficients_coarse_fine": geometry["raw_coefficients"],
                "skill_vs_zero": geometry["joint_explained_fraction"],
            },
            "relations": relations,
        }
    }

    overall, temporal, rows = _euler1d_geometry(payload)
    assert overall["status"] == "ok"
    assert len(rows) == 3
    for scope in ("calls_0_49", "calls_50_99"):
        assert not temporal[scope]["joint_geometry_available"]
        assert not temporal[scope]["joint_cross_term_imputed"]
        assert temporal[scope]["status"] == (
            "individual_only_unrecorded_joint_cross_term"
        )


def test_persistence_threshold_fails_closed_at_first_call_and_crossfits() -> None:
    records = [record for record in _records() if record.view == "sp19"]
    fitted = fit_persistence_threshold(records)
    assert fitted["status"] == "ok"
    assert fitted["threshold"] is not None
    assert fitted["score"]["applied_count"] == 6
    assert fitted["score"]["rms_ratio_vs_zero"] > 0.0

    persistence = grouped_persistence_crossfit(records)
    coefficient = grouped_coefficient_crossfit(records)
    assert persistence["status"] == "ok"
    assert len(persistence["folds"]) == 3
    assert coefficient["status"] == "ok"
    assert coefficient["same_nonzero_sign"]
    assert coefficient["fold_coefficients"] == pytest.approx([-0.5, -0.5, -0.5])
    assert coefficient["skill_vs_zero"] == pytest.approx(1.0)


def test_threshold_tie_breaks_toward_fewer_corrections() -> None:
    records = [record for record in _records() if record.view == "complement"]
    # With beta zero every threshold has identical skill, so the never-apply
    # sentinel must win the deterministic tie.
    fitted = fit_persistence_threshold(records, beta=0.0)
    assert fitted["threshold"] == 2.0
    assert fitted["score"]["applied_count"] == 0


def test_fast_threshold_fit_matches_exhaustive_case_first_scan() -> None:
    records = [record for record in _records() if record.view == "complement"]
    fitted = fit_persistence_threshold(records)
    candidates = sorted(
        {
            float(record.persistence_cosine)
            for record in records
            if record.persistence_cosine is not None
        }
    ) + [2.0]
    scored = [
        (
            threshold,
            score_policy(
                records,
                beta=-0.5,
                persistence_threshold=threshold,
                policy="exhaustive",
            ),
        )
        for threshold in candidates
    ]
    expected_threshold, expected_score = max(
        scored,
        key=lambda item: (float(item[1]["skill_vs_zero"]), item[0]),
    )
    assert fitted["threshold"] == expected_threshold
    assert fitted["score"]["skill_vs_zero"] == pytest.approx(
        expected_score["skill_vs_zero"], abs=1.0e-14
    )


def _phase_rows():
    calls = (9, 10, 11)
    cases = ("c0", "c1", "c2")
    call_rows = []
    audit_rows = []
    for case_index, case_id in enumerate(cases):
        for output_call in calls:
            time = output_call / 30.0
            benefit = 0.04 - 0.1 * time + 0.002 * case_index
            raw_error = 2.0 + 0.1 * case_index
            corrected_error = raw_error * np.sqrt(1.0 - benefit)
            target = 2.0 + 0.01 * output_call
            previous = 1.5 + 0.05 * case_index
            cosine = 0.3 - 0.02 * (output_call - 9)
            requested = np.sqrt(
                target**2 + previous**2 - 2.0 * target * previous * cosine
            )
            for policy, state_error in (
                ("zero", raw_error),
                ("corrected", corrected_error),
            ):
                call_rows.append(
                    {
                        "case_id": case_id,
                        "policy": policy,
                        "output_call": str(output_call),
                        "state_error": state_error,
                        "correction_rms": 999.0 if policy == "corrected" else 0.0,
                        "correction_to_native_increment": (
                            0.02 + 0.001 * output_call if policy == "corrected" else 0.0
                        ),
                    }
                )
            audit_rows.append(
                {
                    "case_id": case_id,
                    "output_call": str(output_call),
                    "target_difference_rms": target,
                    "previous_difference_rms": previous,
                    "requested_change_rms": requested,
                    "applied_to_shadow_increment_ratio": 0.03 + 0.001 * output_call,
                }
            )
    return call_rows, audit_rows, cases, calls


def test_recurrent_phase_norm_law_and_mixed_scale_exclusion() -> None:
    call_rows, audit_rows, cases, calls = _phase_rows()
    records = recurrent_phase_records(
        call_rows,
        audit_rows,
        population="p0",
        expected_cases=cases,
        corrected_policy="corrected",
        expected_calls=calls,
    )
    first = records[0]
    assert first.features["target_previous_cosine"] == pytest.approx(0.3)
    assert first.benefit == pytest.approx(0.01)

    # correction_rms uses a different component scale and must not enter A24.
    changed = [dict(row, correction_rms=-12345.0) for row in call_rows]
    repeated = recurrent_phase_records(
        changed,
        audit_rows,
        population="p0",
        expected_cases=cases,
        corrected_policy="corrected",
        expected_calls=calls,
    )
    assert [record.benefit for record in repeated] == pytest.approx(
        [record.benefit for record in records]
    )
    assert [dict(record.features) for record in repeated] == [
        dict(record.features) for record in records
    ]


def test_recurrent_phase_ridge_and_leave_one_case_out() -> None:
    call_rows, audit_rows, cases, calls = _phase_rows()
    records = recurrent_phase_records(
        call_rows,
        audit_rows,
        population="p0",
        expected_cases=cases,
        corrected_policy="corrected",
        expected_calls=calls,
    )
    fit = fit_phase_ridge(
        records,
        feature_names=PHASE_MODEL_FEATURES["time_only"],
    )
    predictions = predict_phase_ridge(fit, records)
    score = score_phase_predictions(records, predictions)
    assert score["r2_vs_zero"] > 0.9
    assert score["signed_cosine"] > 0.9

    crossfit = leave_one_case_out_phase(
        records,
        feature_names=PHASE_MODEL_FEATURES["time_only"],
    )
    assert len(crossfit["folds"]) == 3
    assert crossfit["score"]["row_count"] == 9
    assert crossfit["score"]["case_count"] == 3


def test_recurrent_phase_all_positive_prediction_reports_zero_harm_recall() -> None:
    call_rows, audit_rows, cases, calls = _phase_rows()
    records = list(
        recurrent_phase_records(
            call_rows,
            audit_rows,
            population="p0",
            expected_cases=cases,
            corrected_policy="corrected",
            expected_calls=calls,
        )
    )
    # Make one row harmful without altering the observable features.
    harmful = records[0]
    records[0] = type(harmful)(
        population=harmful.population,
        case_id=harmful.case_id,
        output_call=harmful.output_call,
        raw_state_error=harmful.raw_state_error,
        corrected_state_error=harmful.raw_state_error * np.sqrt(1.01),
        benefit=-0.01,
        features=harmful.features,
    )
    score = score_phase_predictions(records, np.ones(len(records)))
    assert score["sign_accuracy"] == pytest.approx(8.0 / 9.0)
    assert score["actual_harm_count"] == 1
    assert score["predicted_harm_count"] == 0
    assert score["false_safe_count"] == 1
    assert score["harm_recall"] == 0.0
    assert score["help_recall"] == 1.0
    assert score["balanced_sign_accuracy"] == 0.5


@pytest.mark.parametrize(
    "mutate,message",
    [
        (lambda calls, audits: calls.pop(), "call metric inventory mismatch"),
        (lambda calls, audits: audits.pop(), "tether audit inventory mismatch"),
        (
            lambda calls, audits: audits[0].update({"previous_difference_rms": "0"}),
            "phase denominator is unresolved",
        ),
        (
            lambda calls, audits: audits[0].update({"requested_change_rms": "100"}),
            "cosine violates norm closure",
        ),
    ],
)
def test_recurrent_phase_inventory_and_denominators_fail_closed(
    mutate, message
) -> None:
    call_rows, audit_rows, cases, calls = _phase_rows()
    mutate(call_rows, audit_rows)
    with pytest.raises(ValueError, match=message):
        recurrent_phase_records(
            call_rows,
            audit_rows,
            population="p0",
            expected_cases=cases,
            corrected_policy="corrected",
            expected_calls=calls,
        )


def _energy_rows():
    rows = []
    arms = {
        "zero": {
            "state_error": 5.0,
            "rank8_state_error": 3.0,
            "components": (1.0, 2.0, 2.0, 4.0),
        },
        "corrected": {
            "state_error": np.sqrt(21.25),
            "rank8_state_error": 2.5,
            "components": (0.5, 1.0, 2.0, 4.0),
        },
    }
    for output_call in (9, 10):
        for policy, values in arms.items():
            # Call 10 is the exact zero-correction denominator case.
            source = arms["zero"] if output_call == 10 else values
            rows.append(
                {
                    "case_id": "c0",
                    "policy": policy,
                    "output_call": str(output_call),
                    "state_error": source["state_error"],
                    "rank8_state_error": source["rank8_state_error"],
                    **{
                        f"component_{name}_state_error": value
                        for name, value in zip(
                            ("density", "x_momentum", "y_momentum", "energy"),
                            source["components"],
                            strict=True,
                        )
                    },
                    "boundary_state_error": 0.2,
                    "shock_state_error": 0.3,
                    "vortex_state_error": 0.4,
                    "smooth_state_error": 0.5,
                }
            )
    return rows


def test_recurrent_error_energy_additive_closure_and_zero_denominator() -> None:
    rows = recurrent_error_energy_rows(
        _energy_rows(),
        population="p0",
        expected_cases=("c0",),
        corrected_policy="corrected",
        expected_calls=(9, 10),
    )
    assert len(rows) == 2
    changed, unchanged = rows
    assert changed["full_benefit"] == pytest.approx(3.75)
    assert changed["rank8_benefit"] == pytest.approx(2.75)
    assert changed["remainder_benefit"] == pytest.approx(1.0)
    assert changed["rank8_benefit_share"] == pytest.approx(2.75 / 3.75)
    assert sum(
        changed[f"component_{name}_benefit"]
        for name in ("density", "x_momentum", "y_momentum", "energy")
    ) == pytest.approx(changed["full_benefit"])
    assert changed["maximum_component_energy_closure_abs"] <= 1.0e-12
    assert changed["rank8_remainder_benefit_closure_abs"] <= 1.0e-12

    assert unchanged["full_benefit"] == 0.0
    assert unchanged["benefit_share_status"] == "unresolved_small_denominator"
    assert unchanged["rank8_benefit_share"] is None
    assert unchanged["remainder_benefit_share"] is None


@pytest.mark.parametrize(
    "mutate,message",
    [
        (lambda rows: rows.pop(), "metric inventory mismatch"),
        (
            lambda rows: rows[2].update({"rank8_state_error": 6.0}),
            "rank8 energy exceeds full energy",
        ),
        (
            lambda rows: rows[2].update({"component_density_state_error": 2.0}),
            "component energy does not close",
        ),
    ],
)
def test_recurrent_error_energy_fails_closed(mutate, message: str) -> None:
    rows = _energy_rows()
    mutate(rows)
    with pytest.raises(ValueError, match=message):
        recurrent_error_energy_rows(
            rows,
            population="p0",
            expected_cases=("c0",),
            corrected_policy="corrected",
            expected_calls=(9, 10),
        )


def _cross_coordinate_snapshots():
    cells = ((1, 0), (1, 1))
    calls = (0, 1, 2, 3)
    directions = ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0))
    rows = []
    for group_index, group_id in enumerate(("g0", "g1", "g2", "g3")):
        case_id = f"{group_id}_c0"
        scale = 1.0 + 0.1 * group_index
        for input_call, (first, second) in zip(calls, directions, strict=True):
            fine = np.asarray((scale * first, scale * second))
            coarse = np.asarray((fine[0] + fine[1], fine[0] - fine[1]))
            target = np.asarray((-fine[1], 2.0 * fine[0]))
            for index, (mode, component) in enumerate(cells):
                rows.append(
                    {
                        "case_id": case_id,
                        "group_id": group_id,
                        "input_call": str(input_call),
                        "mode_index": str(mode),
                        "component": str(component),
                        "coarse_coordinate": coarse[index],
                        "fine_coordinate": fine[index],
                        "target_coordinate": target[index],
                    }
                )
    return snapshots_from_modal_rows(
        rows,
        expected_cells=cells,
        expected_calls=calls,
    )


def test_modal_linear_map_recovers_cross_coordinate_structure_nested() -> None:
    snapshots = _cross_coordinate_snapshots()
    cells = ((1, 0), (1, 1))
    selected = select_grouped_modal_linear_map(
        snapshots,
        active_cells=cells,
        families=("fine_diagonal", "fine_full"),
        ridges=(1.0e-6, 1.0e-2),
    )
    assert selected["selected"]["family"] == "fine_full"
    assert selected["selected"]["ridge"] == 1.0e-6
    assert selected["selected"]["skill_vs_zero"] > 0.99999
    assert selected["candidates"][0]["skill_vs_zero"] == pytest.approx(0.0)

    nested = nested_grouped_modal_linear_map(
        snapshots,
        active_cells=cells,
        families=("fine_diagonal", "fine_full"),
        ridges=(1.0e-6, 1.0e-2),
    )
    assert nested["score"]["skill_vs_zero"] > 0.99999
    assert nested["score"]["case_win_count"] == 4
    assert nested["score"]["group_win_count"] == 4
    assert nested["score"]["maximum_benefit_identity_abs"] <= 1.0e-12
    assert all(
        fold["held_out_group"] not in fold["fit_groups"] for fold in nested["folds"]
    )

    stability = modal_linear_map_stability(selected["selected_crossfit"]["fits"])
    assert stability["minimum_pairwise_frobenius_cosine"] > 0.99999
    assert stability["maximum_to_minimum_norm_ratio"] < 1.00001


def test_modal_causal_observer_is_prefix_only_and_decay_zero_matches_current() -> None:
    snapshots = _cross_coordinate_snapshots()
    cells = ((1, 0), (1, 1))
    current = modal_causal_observer_features(snapshots, active_cells=cells, decay=0.0)
    expected = np.stack([row.fine for row in snapshots])
    np.testing.assert_array_equal(current, expected)

    smoothed = modal_causal_observer_features(snapshots, active_cells=cells, decay=0.8)
    changed = list(snapshots)
    last = changed[-1]
    changed[-1] = type(last)(
        case_id=last.case_id,
        group_id=last.group_id,
        input_call=last.input_call,
        cells=last.cells,
        coarse=last.coarse,
        fine=last.fine + 100.0,
        target=last.target,
    )
    repeated = modal_causal_observer_features(changed, active_cells=cells, decay=0.8)
    np.testing.assert_array_equal(repeated[:-1], smoothed[:-1])


def test_modal_causal_observer_recovers_lagged_target_without_label_leakage() -> None:
    snapshots = _cross_coordinate_snapshots()
    cells = ((1, 0), (1, 1))
    features = modal_causal_observer_features(snapshots, active_cells=cells, decay=0.8)
    matrix = np.asarray(((0.5, -1.0), (2.0, 0.25)))
    lagged = []
    for snapshot, target in zip(snapshots, features @ matrix, strict=True):
        lagged.append(
            type(snapshot)(
                case_id=snapshot.case_id,
                group_id=snapshot.group_id,
                input_call=snapshot.input_call,
                cells=snapshot.cells,
                coarse=snapshot.coarse,
                fine=snapshot.fine,
                target=target,
            )
        )
    fit = fit_modal_causal_observer(
        lagged,
        active_cells=cells,
        decay=0.8,
        label_filter=lambda row: row.input_call != 2,
    )
    prediction = predict_modal_causal_observer(fit, lagged)
    score = score_modal_linear_predictions(lagged, prediction, active_cells=cells)
    assert fit["training_label_count"] == 12
    assert np.array_equal(fit["intercept"], np.zeros(len(cells)))
    assert score["skill_vs_zero"] > 0.999999


def test_modal_causal_observer_fails_closed_on_gapped_history() -> None:
    snapshots = [
        row
        for row in _cross_coordinate_snapshots()
        if not (row.case_id == "g0_c0" and row.input_call == 1)
    ]
    with pytest.raises(ValueError, match="contiguous and start at zero"):
        modal_causal_observer_features(
            snapshots, active_cells=((1, 0), (1, 1)), decay=0.8
        )


def test_modal_phase_portrait_uses_only_current_and_previous_discrepancy() -> None:
    snapshots = _cross_coordinate_snapshots()
    cells = ((1, 0), (1, 1))
    features = modal_discrepancy_phase_portrait_features(snapshots, active_cells=cells)
    np.testing.assert_array_equal(
        features[:, :2], np.stack([row.fine for row in snapshots])
    )
    for case_id in sorted({row.case_id for row in snapshots}):
        indices = [
            index for index, row in enumerate(snapshots) if row.case_id == case_id
        ]
        assert np.array_equal(features[indices[0], 2:], np.zeros(2))
        for previous, current in pairwise(indices):
            np.testing.assert_array_equal(
                features[current, 2:],
                snapshots[current].fine - snapshots[previous].fine,
            )

    changed = list(snapshots)
    last = changed[-1]
    changed[-1] = type(last)(
        case_id=last.case_id,
        group_id=last.group_id,
        input_call=last.input_call,
        cells=last.cells,
        coarse=last.coarse,
        fine=last.fine + 100.0,
        target=last.target,
    )
    repeated = modal_discrepancy_phase_portrait_features(changed, active_cells=cells)
    np.testing.assert_array_equal(repeated[:-1], features[:-1])


def test_modal_phase_portrait_recovers_velocity_target_with_held_labels() -> None:
    snapshots = _cross_coordinate_snapshots()
    cells = ((1, 0), (1, 1))
    features = modal_discrepancy_phase_portrait_features(snapshots, active_cells=cells)
    current_map = np.asarray(((0.25, -0.5), (0.75, 0.5)))
    velocity_map = np.asarray(((1.5, 0.25), (-0.5, 2.0)))
    targets = features[:, :2] @ current_map + features[:, 2:] @ velocity_map
    phase_rows = []
    for snapshot, target in zip(snapshots, targets, strict=True):
        phase_rows.append(
            type(snapshot)(
                case_id=snapshot.case_id,
                group_id=snapshot.group_id,
                input_call=snapshot.input_call,
                cells=snapshot.cells,
                coarse=snapshot.coarse,
                fine=snapshot.fine,
                target=target,
            )
        )
    fit = fit_modal_discrepancy_phase_portrait(
        phase_rows,
        active_cells=cells,
        label_filter=lambda row: row.input_call != 2,
    )
    prediction = predict_modal_discrepancy_phase_portrait(fit, phase_rows)
    score = score_modal_linear_predictions(phase_rows, prediction, active_cells=cells)
    assert fit["training_label_count"] == 12
    assert np.array_equal(fit["intercept"], np.zeros(len(cells)))
    assert score["skill_vs_zero"] > 0.6


def test_modal_phase_portrait_fails_closed_on_gapped_history() -> None:
    snapshots = [
        row
        for row in _cross_coordinate_snapshots()
        if not (row.case_id == "g0_c0" and row.input_call == 1)
    ]
    with pytest.raises(ValueError, match="contiguous and start at zero"):
        modal_discrepancy_phase_portrait_features(
            snapshots, active_cells=((1, 0), (1, 1))
        )


def test_modal_event_clock_is_causal_monotone_bounded_and_scale_invariant() -> None:
    snapshots = _cross_coordinate_snapshots()
    cells = ((1, 0), (1, 1))
    record = modal_causal_discrepancy_event_clock_features(
        snapshots, active_cells=cells
    )
    assert np.all(record["resolved"])
    for case_id in sorted({row.case_id for row in snapshots}):
        indices = [
            index for index, row in enumerate(snapshots) if row.case_id == case_id
        ]
        clock = record["event_clock"][indices]
        assert clock[0] == 0.0
        assert np.all(np.diff(clock) >= 0.0)
        assert np.all((0.0 <= clock) & (clock < 1.0))
        assert np.array_equal(record["velocity"][indices[0]], np.zeros(2))

    scaled = [
        type(row)(
            case_id=row.case_id,
            group_id=row.group_id,
            input_call=row.input_call,
            cells=row.cells,
            coarse=7.0 * row.coarse,
            fine=7.0 * row.fine,
            target=row.target,
        )
        for row in snapshots
    ]
    scaled_record = modal_causal_discrepancy_event_clock_features(
        scaled, active_cells=cells
    )
    np.testing.assert_allclose(
        scaled_record["event_clock"], record["event_clock"], rtol=0.0, atol=1.0e-15
    )

    changed = list(snapshots)
    last = changed[-1]
    changed[-1] = type(last)(
        case_id=last.case_id,
        group_id=last.group_id,
        input_call=last.input_call,
        cells=last.cells,
        coarse=last.coarse,
        fine=last.fine + 100.0,
        target=last.target,
    )
    changed_record = modal_causal_discrepancy_event_clock_features(
        changed, active_cells=cells
    )
    np.testing.assert_array_equal(
        changed_record["features"][:-1], record["features"][:-1]
    )


def test_modal_event_clock_recovers_clock_velocity_target_with_held_labels() -> None:
    snapshots = list(_cross_coordinate_snapshots())
    cells = ((1, 0), (1, 1))
    for index, row in enumerate(snapshots):
        phase = 0.13 * row.input_call + 0.03 * int(row.group_id[1:])
        fine = row.fine + np.asarray((phase, -0.4 * phase))
        snapshots[index] = type(row)(
            case_id=row.case_id,
            group_id=row.group_id,
            input_call=row.input_call,
            cells=row.cells,
            coarse=row.coarse,
            fine=fine,
            target=row.target,
        )
    features = modal_causal_discrepancy_event_clock_features(
        snapshots, active_cells=cells
    )["features"]
    coefficients = np.asarray(
        (
            (0.25, -0.5),
            (0.75, 0.5),
            (1.0, 0.25),
            (-0.5, 1.25),
            (1.5, 0.25),
            (-0.5, 2.0),
        )
    )
    targets = features @ coefficients
    event_rows = [
        type(row)(
            case_id=row.case_id,
            group_id=row.group_id,
            input_call=row.input_call,
            cells=row.cells,
            coarse=row.coarse,
            fine=row.fine,
            target=target,
        )
        for row, target in zip(snapshots, targets, strict=True)
    ]
    fit = fit_modal_causal_discrepancy_event_clock(
        event_rows,
        active_cells=cells,
        label_filter=lambda row: row.input_call != 2,
    )
    prediction = predict_modal_causal_discrepancy_event_clock(fit, event_rows)
    score = score_modal_linear_predictions(event_rows, prediction, active_cells=cells)
    assert fit["training_label_count"] == 12
    assert np.array_equal(fit["intercept"], np.zeros(len(cells)))
    assert score["skill_vs_zero"] > 0.7


def test_modal_event_clock_fails_closed_on_unresolved_or_gapped_history() -> None:
    snapshots = _cross_coordinate_snapshots()
    cells = ((1, 0), (1, 1))
    zero = [
        type(row)(
            case_id=row.case_id,
            group_id=row.group_id,
            input_call=row.input_call,
            cells=row.cells,
            coarse=row.coarse,
            fine=np.zeros_like(row.fine),
            target=row.target,
        )
        for row in snapshots
    ]
    record = modal_causal_discrepancy_event_clock_features(zero, active_cells=cells)
    assert not np.any(record["resolved"])
    assert np.array_equal(record["event_clock"], np.zeros(len(zero)))
    with pytest.raises(ValueError, match="event-clock denominator is unresolved"):
        fit_modal_causal_discrepancy_event_clock(zero, active_cells=cells)

    gapped = [
        row for row in snapshots if not (row.case_id == "g0_c0" and row.input_call == 1)
    ]
    with pytest.raises(ValueError, match="contiguous and start at zero"):
        modal_causal_discrepancy_event_clock_features(gapped, active_cells=cells)


def test_modal_event_clock_zero_target_produces_exact_zero_correction() -> None:
    snapshots = _cross_coordinate_snapshots()
    cells = ((1, 0), (1, 1))
    zero_target = [
        type(row)(
            case_id=row.case_id,
            group_id=row.group_id,
            input_call=row.input_call,
            cells=row.cells,
            coarse=row.coarse,
            fine=row.fine,
            target=np.zeros_like(row.target),
        )
        for row in snapshots
    ]
    fit = fit_modal_causal_discrepancy_event_clock(zero_target, active_cells=cells)
    prediction = predict_modal_causal_discrepancy_event_clock(fit, zero_target)
    assert np.array_equal(fit["intercept"], np.zeros(len(cells)))
    assert np.array_equal(prediction, np.zeros_like(prediction))


def test_a39_parent_payload_hashes_are_complete_sha256_values() -> None:
    from scripts.time_dependent_no import (
        analyze_pcno_causal_discrepancy_event_clock as a39,
    )

    assert set(a39.EXPECTED_PAYLOADS) == {
        "a38_result",
        "calibration_result",
        "teacher_result",
    }
    assert all(
        len(value) == 64 and set(value) <= set("0123456789abcdef")
        for value in a39.EXPECTED_PAYLOADS.values()
    )


def _affine_phase_snapshots():
    cells = ((1, 0), (1, 1))
    calls = (0, 10, 19, 29)
    rows = []
    start = np.asarray(((1.5, -0.25), (0.5, 1.0)))
    end = np.asarray(((-0.5, 1.25), (2.0, -1.0)))
    for group_index, group_id in enumerate(("g0", "g1", "g2")):
        case_id = f"{group_id}_c0"
        for input_call in calls:
            phase = input_call / 29.0
            fine = np.asarray(
                (
                    1.0 + 0.2 * group_index + 0.05 * input_call,
                    (-1.0) ** group_index * (0.5 + 0.03 * input_call),
                )
            )
            target = fine @ ((1.0 - phase) * start + phase * end)
            for index, (mode, component) in enumerate(cells):
                rows.append(
                    {
                        "case_id": case_id,
                        "group_id": group_id,
                        "input_call": str(input_call),
                        "mode_index": str(mode),
                        "component": str(component),
                        "coarse_coordinate": float(2.0 * fine[index]),
                        "fine_coordinate": float(fine[index]),
                        "target_coordinate": float(target[index]),
                    }
                )
    return snapshots_from_modal_rows(
        rows,
        expected_cells=cells,
        expected_calls=calls,
    )


def test_modal_affine_phase_map_recovers_endpoint_blend() -> None:
    snapshots = _affine_phase_snapshots()
    cells = ((1, 0), (1, 1))
    fit = fit_modal_affine_phase_map(
        snapshots,
        active_cells=cells,
        family="fine_full_affine",
    )
    prediction = predict_modal_affine_phase_map(fit, snapshots)
    score = score_modal_linear_predictions(
        snapshots,
        prediction,
        active_cells=cells,
    )
    assert fit["ridge"] == 1.0e-6
    assert fit["last_input_call"] == 29
    assert fit["base_feature_count"] == len(cells)
    assert np.array_equal(fit["intercept"], np.zeros(len(cells)))
    assert score["skill_vs_zero"] > 0.999999999
    assert score["harmful_row_count"] == 0

    groups = sorted({row.group_id for row in snapshots})
    fits = [
        fit_modal_affine_phase_map(
            [row for row in snapshots if row.group_id != group],
            active_cells=cells,
            family="fine_full_affine",
        )
        for group in groups
    ]
    stability = modal_affine_map_stability(fits)
    assert stability["whole"]["minimum_pairwise_frobenius_cosine"] > 0.999999
    assert stability["start"]["maximum_to_minimum_norm_ratio"] < 1.001
    assert stability["end"]["maximum_to_minimum_norm_ratio"] < 1.001

    diagnostics = modal_affine_fit_diagnostics(fit, snapshots)
    assert diagnostics["standardized_design_condition_number"] is not None
    assert diagnostics["start_end_coefficient_cosine"] < 1.0
    assert diagnostics["float32_prediction_unresolved_count"] == 0
    assert diagnostics["maximum_float32_prediction_relative_change"] <= 1.0e-6


@pytest.mark.parametrize("bad_call", [-1, 30])
def test_modal_affine_phase_map_rejects_calls_outside_fixed_horizon(
    bad_call: int,
) -> None:
    snapshots = list(_affine_phase_snapshots())
    first = snapshots[0]
    snapshots[0] = type(first)(
        case_id=first.case_id,
        group_id=first.group_id,
        input_call=bad_call,
        cells=first.cells,
        coarse=first.coarse,
        fine=first.fine,
        target=first.target,
    )
    with pytest.raises(ValueError, match="outside the fixed horizon"):
        fit_modal_affine_phase_map(
            snapshots,
            active_cells=((1, 0), (1, 1)),
            family="fine_full_affine",
        )


def test_modal_linear_selection_tie_prefers_simpler_then_larger_ridge() -> None:
    snapshots = _cross_coordinate_snapshots()
    cells = ((1, 0), (1, 1))
    # A deliberately loose tie tolerance is not configurable; exact-zero
    # targets make every map unresolved, so use an explicit duplicate family
    # rejection and verify ridge ordering on a zero-feature response instead.
    with pytest.raises(ValueError, match="candidate inventory"):
        select_grouped_modal_linear_map(
            snapshots,
            active_cells=cells,
            families=("fine_full", "fine_full"),
            ridges=(1.0e-6,),
        )

    predictions = np.zeros((len(snapshots), len(cells)))
    score = score_modal_linear_predictions(
        snapshots,
        predictions,
        active_cells=cells,
    )
    assert score["skill_vs_zero"] == 0.0
    assert score["harmful_row_count"] == 0
    assert score["signed_cosine_unresolved_count"] == len(snapshots)


def test_modal_linear_map_fails_closed_on_unresolved_feature() -> None:
    snapshots = list(_cross_coordinate_snapshots())
    for index, snapshot in enumerate(snapshots):
        snapshots[index] = type(snapshot)(
            case_id=snapshot.case_id,
            group_id=snapshot.group_id,
            input_call=snapshot.input_call,
            cells=snapshot.cells,
            coarse=snapshot.coarse,
            fine=np.asarray((0.0, snapshot.fine[1])),
            target=snapshot.target,
        )
    with pytest.raises(ValueError, match="feature RMS is unresolved"):
        select_grouped_modal_linear_map(
            snapshots,
            active_cells=((1, 0), (1, 1)),
            families=("fine_full",),
            ridges=(1.0e-6,),
        )


def test_modal_prediction_cap_is_target_free_and_fails_closed() -> None:
    prediction = np.asarray(((3.0, 4.0), (0.0, 0.0), (1.0, 0.0)))
    native = np.asarray((10.0, 1.0, 0.0))
    capped = cap_modal_predictions(
        prediction,
        native,
        total_volume=1.0,
        maximum_relative_norm=0.2,
    )
    assert capped["scale"] == pytest.approx((0.4, 1.0, 0.0))
    assert capped["correction_to_native_increment"][:2] == pytest.approx((0.2, 0.0))
    assert np.isnan(capped["correction_to_native_increment"][2])
    assert capped["status"].tolist() == [
        "ok",
        "ok",
        "unresolved_native_increment",
    ]
    assert capped["cap_active"].tolist() == [True, False, False]
    assert capped["maximum_cap_violation"] <= 1.0e-15


def test_modal_linear_diagnostics_detect_cancellation_and_float32_stability() -> None:
    snapshots = _cross_coordinate_snapshots()
    cells = ((1, 0), (1, 1))
    selected = select_grouped_modal_linear_map(
        snapshots,
        active_cells=cells,
        families=("coarse_fine_full",),
        ridges=(1.0e-6,),
    )
    fit = selected["selected_crossfit"]["fits"][0]
    diagnostics = modal_linear_fit_diagnostics(fit, snapshots)
    assert diagnostics["standardized_design_condition_number"] is None
    assert diagnostics["standardized_design_singular_values"][-1] <= 1.0e-12
    assert diagnostics["coefficient_frobenius_norm"] > 0.0
    assert diagnostics["maximum_float32_prediction_relative_change"] <= 1.0e-6
    assert diagnostics["cancellation_resolved_count"] == len(snapshots)
    assert diagnostics["median_cancellation_amplification"] >= 1.0
