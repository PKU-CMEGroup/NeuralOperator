from __future__ import annotations

import json

import numpy as np
import pytest

from scripts.time_dependent_no.analyze_euler1d_phase_conditioned_crossfit import (
    PHASE_SCOPES,
    crossfit_phase_schedule,
    readiness_checks,
    reconstruct_scope_statistics,
    verify_equal_phase_partition,
)
from scripts.time_dependent_no.analyze_euler1d_state_conditioned_crossfit import (
    _fit_state_map,
    _predict_coefficients,
    _support_audit,
    _with_json_payload_sha256,
    nested_state_crossfit,
    riemann_descriptors,
)
from scripts.time_dependent_no.evaluate_euler1d_cross_resolution_correction import (
    _case_first_control,
    synthetic_summary,
)
from utility.time_dependent_no.euler1d_cross_resolution_correction import (
    PHYSICAL_COMPONENT_SCALES,
    Euler1DResolutionContract,
    ScalarFitRecord,
    apply_capped_correction,
    build_cosine_projector,
    common_source_states,
    component_integrals,
    corrections_for_coefficient,
    crossfit_case_groups,
    fit_case_first_scalar,
    mapped_increment_basis,
    prolong_piecewise_constant,
    restrict_cell_averages,
    score_records,
    validate_record_inventory,
    weighted_scaled_rms,
)


def test_nested_restriction_and_piecewise_prolongation_conserve_integrals() -> None:
    rng = np.random.default_rng(4)
    fine = rng.normal(size=(16, 3))
    native = restrict_cell_averages(fine, 8)
    coarse = restrict_cell_averages(native, 4)

    np.testing.assert_allclose(
        restrict_cell_averages(fine, 4), coarse, rtol=0.0, atol=1.0e-16
    )
    np.testing.assert_allclose(
        fine.mean(axis=0), native.mean(axis=0), rtol=0.0, atol=1.0e-15
    )
    np.testing.assert_allclose(
        native.mean(axis=0), coarse.mean(axis=0), rtol=0.0, atol=1.0e-15
    )
    prolonged = prolong_piecewise_constant(coarse, 8)
    np.testing.assert_allclose(
        prolonged.mean(axis=0), coarse.mean(axis=0), rtol=0.0, atol=1.0e-15
    )
    assert np.array_equal(restrict_cell_averages(prolonged, 4), coarse)


def test_mapped_increment_basis_uses_common_source_and_registered_signs() -> None:
    contract = Euler1DResolutionContract(4, 8, 16)
    x = (np.arange(16, dtype=np.float64) + 0.5) / 16.0
    fine = np.column_stack((1.0 + x, x - 0.2, 2.0 + x * x))
    states = common_source_states(fine, contract)
    coarse_increment = np.full((4, 3), (1.0, 2.0, 3.0))
    native_increment = np.full((8, 3), (2.0, 4.0, 6.0))
    fine_increment = np.full((16, 3), (4.0, 8.0, 12.0))
    predicted = type(states)(
        coarse=states.coarse + coarse_increment,
        native=states.native + native_increment,
        fine=states.fine + fine_increment,
    )

    basis = mapped_increment_basis(states, predicted, contract)

    np.testing.assert_allclose(basis.native_minus_coarse, native_increment / 2.0)
    np.testing.assert_allclose(basis.fine_minus_native, native_increment)
    np.testing.assert_allclose(
        basis.fine_on_native.mean(axis=0), fine_increment.mean(axis=0)
    )
    np.testing.assert_allclose(
        basis.coarse_on_native.mean(axis=0), coarse_increment.mean(axis=0)
    )


def test_rank7_cosine_projection_removes_only_constant_and_closes_bands() -> None:
    cells = 32
    volumes = np.full(cells, 1.0 / cells)
    projector = build_cosine_projector(cells, rank=cells, volumes=volumes)
    rng = np.random.default_rng(9)
    field = rng.normal(size=(cells, 3))
    constant = projector.project(
        field,
        active_modes=(0,),
        component_scale=PHYSICAL_COMPONENT_SCALES,
    )
    low = projector.project(
        field,
        active_modes=tuple(range(1, 8)),
        component_scale=PHYSICAL_COMPONENT_SCALES,
    )
    transition = projector.project(
        field,
        active_modes=tuple(range(8, 16)),
        component_scale=PHYSICAL_COMPONENT_SCALES,
    )
    local = projector.project(
        field,
        active_modes=tuple(range(16, cells)),
        component_scale=PHYSICAL_COMPONENT_SCALES,
    )

    np.testing.assert_allclose(
        constant + low + transition + local, field, rtol=0.0, atol=2.0e-14
    )
    np.testing.assert_allclose(
        component_integrals(
            low,
            volumes=volumes,
            component_scale=PHYSICAL_COMPONENT_SCALES,
        ),
        0.0,
        rtol=0.0,
        atol=2.0e-15,
    )
    assert projector.maximum_orthogonality_error <= 2.0e-15


def test_zero_and_cap_paths_are_explicit_and_mean_neutral() -> None:
    cells = 16
    volumes = np.full(cells, 1.0 / cells)
    projector = build_cosine_projector(cells, rank=8, volumes=volumes)
    x = (np.arange(cells, dtype=np.float64) + 0.5) / cells
    feature = np.column_stack(
        (
            np.cos(np.pi * x),
            np.cos(2.0 * np.pi * x),
            np.cos(3.0 * np.pi * x),
        )
    )
    projected = projector.project(
        feature,
        active_modes=tuple(range(1, 8)),
        component_scale=PHYSICAL_COMPONENT_SCALES,
    )
    native = 0.01 * np.ones_like(projected)

    correction, audit = apply_capped_correction(
        native,
        projected,
        coefficient=-10.0,
        volumes=volumes,
        component_scale=PHYSICAL_COMPONENT_SCALES,
    )

    assert audit.status == "ok"
    assert audit.cap_active
    assert audit.correction_to_native_increment == pytest.approx(0.1)
    assert audit.maximum_component_integral_abs <= 2.0e-15
    assert weighted_scaled_rms(
        correction,
        volumes=volumes,
        component_scale=PHYSICAL_COMPONENT_SCALES,
    ) == pytest.approx(
        0.1
        * weighted_scaled_rms(
            native,
            volumes=volumes,
            component_scale=PHYSICAL_COMPONENT_SCALES,
        )
    )

    zero, zero_audit = apply_capped_correction(
        np.zeros_like(native),
        projected,
        coefficient=-0.5,
        volumes=volumes,
        component_scale=PHYSICAL_COMPONENT_SCALES,
    )
    assert zero_audit.status == "small_native_increment"
    assert np.array_equal(zero, np.zeros_like(zero))


def _stored_scalar_relation(
    statistics: list[dict[str, float | str]],
    *,
    calls: tuple[int, ...],
) -> dict[str, object]:
    rows = []
    for statistic in statistics:
        target_square = float(statistic["target_square"])
        feature_square = float(statistic["feature_square"])
        cross = float(statistic["cross"])
        rows.append(
            {
                "case_id": statistic["case_id"],
                "input_calls": list(calls),
                "zero_sse": target_square,
                "correction_rms": np.sqrt(feature_square),
                "corrected_sse": target_square + feature_square - 2.0 * cross,
            }
        )
    target_square = float(np.mean([row["zero_sse"] for row in rows]))
    feature_square = float(np.mean([float(row["correction_rms"]) ** 2 for row in rows]))
    cross = float(
        np.mean(
            [
                0.5
                * (
                    float(row["zero_sse"])
                    + float(row["correction_rms"]) ** 2
                    - float(row["corrected_sse"])
                )
                for row in rows
            ]
        )
    )
    return {
        "fit": {
            "coefficient": cross / feature_square,
            "cross": cross,
            "denominator": feature_square,
            "feature_rms": np.sqrt(feature_square),
            "target_rms": np.sqrt(target_square),
            "status": "ok",
        },
        "unit_feature_score": {
            "case_scores": rows,
            "zero_sse_case_mean": target_square,
            "corrected_sse_case_mean": float(
                np.mean([row["corrected_sse"] for row in rows])
            ),
            "correction_rms": np.sqrt(feature_square),
        },
    }


def test_phase_statistic_reconstruction_recovers_case_first_fit() -> None:
    statistics = [
        {
            "case_id": "case_a",
            "target_square": 4.0,
            "feature_square": 1.0,
            "cross": -1.0,
        },
        {
            "case_id": "case_b",
            "target_square": 9.0,
            "feature_square": 4.0,
            "cross": 2.0,
        },
    ]
    relation = _stored_scalar_relation(statistics, calls=(0, 1))
    reconstructed = reconstruct_scope_statistics(
        relation,
        expected_case_ids=("case_a", "case_b"),
        expected_calls=(0, 1),
    )
    aggregate = reconstructed["aggregate"]
    assert aggregate["target_square"] == pytest.approx(6.5)
    assert aggregate["feature_square"] == pytest.approx(2.5)
    assert aggregate["cross"] == pytest.approx(0.5)
    assert aggregate["coefficient"] == pytest.approx(0.2)
    assert reconstructed["maximum_reconstruction_abs"] <= 1.0e-15


def _phase_crossfit_fixture(scale: float = 1.0) -> dict[str, dict[str, object]]:
    early = []
    late = []
    full = []
    for index in range(16):
        case_id = f"case_{index:02d}"
        feature_square = scale * (1.0 + 0.05 * index)
        target_square = scale * (2.0 + 0.03 * index)
        early_row = {
            "case_id": case_id,
            "input_calls": list(range(50)),
            "target_square": target_square,
            "feature_square": feature_square,
            "cross": -0.5 * feature_square,
            "unit_corrected_square": target_square + 2.0 * feature_square,
        }
        late_row = {
            **early_row,
            "input_calls": list(range(50, 100)),
            "cross": 0.5 * feature_square,
            "unit_corrected_square": target_square,
        }
        full_row = {
            **early_row,
            "input_calls": list(range(100)),
            "cross": 0.0,
            "unit_corrected_square": target_square + feature_square,
        }
        early.append(early_row)
        late.append(late_row)
        full.append(full_row)
    return {
        "all": {"case_statistics": full},
        "calls_0_49": {"case_statistics": early},
        "calls_50_99": {"case_statistics": late},
    }


def test_phase_crossfit_uses_held_out_cases_and_passes_registered_gate() -> None:
    reconstructed = _phase_crossfit_fixture()
    phase_closure = verify_equal_phase_partition(reconstructed)
    assert phase_closure["maximum_case_phase_closure_abs"] <= 1.0e-15

    crossfit = crossfit_phase_schedule(reconstructed)
    assert len(crossfit["folds"]) == 16
    assert all(
        fold["held_out_case_id"] not in fold["fit_case_ids"]
        for fold in crossfit["folds"]
    )
    assert crossfit["full_fit_coefficients"] == pytest.approx(
        {"all": 0.0, "calls_0_49": -0.5, "calls_50_99": 0.5}
    )
    assert crossfit["scores"]["global"]["combined"]["skill_vs_zero"] == pytest.approx(
        0.0
    )
    assert crossfit["scores"]["two_phase"]["combined"]["skill_vs_zero"] > 0.1
    assert crossfit["two_phase_case_win_count"] == 16
    assert all(readiness_checks(crossfit).values())


def test_phase_crossfit_is_invariant_to_common_energy_scaling() -> None:
    first = crossfit_phase_schedule(_phase_crossfit_fixture(scale=1.0))
    second = crossfit_phase_schedule(_phase_crossfit_fixture(scale=1.0e-8))
    for scope in ("all", *PHASE_SCOPES):
        assert second["full_fit_coefficients"][scope] == pytest.approx(
            first["full_fit_coefficients"][scope]
        )
    assert second["scores"]["two_phase"]["combined"]["skill_vs_zero"] == pytest.approx(
        first["scores"]["two_phase"]["combined"]["skill_vs_zero"]
    )


@pytest.mark.parametrize("failure", ["calls", "duplicate", "cauchy"])
def test_phase_statistic_reconstruction_fails_closed(failure: str) -> None:
    statistics = [
        {
            "case_id": "case_a",
            "target_square": 4.0,
            "feature_square": 1.0,
            "cross": 1.0,
        },
        {
            "case_id": "case_b",
            "target_square": 4.0,
            "feature_square": 1.0,
            "cross": -1.0,
        },
    ]
    relation = _stored_scalar_relation(statistics, calls=(0, 1))
    rows = relation["unit_feature_score"]["case_scores"]
    if failure == "calls":
        rows[0]["input_calls"] = [0, 2]
    elif failure == "duplicate":
        rows[1]["case_id"] = "case_a"
    else:
        rows[0]["corrected_sse"] = -10.0
    with pytest.raises((TypeError, ValueError)):
        reconstruct_scope_statistics(
            relation,
            expected_case_ids=("case_a", "case_b"),
            expected_calls=(0, 1),
        )


def test_riemann_descriptors_follow_frozen_dimensionless_contract() -> None:
    left = np.asarray(
        [
            [1.0, 2.0, 4.0],
            [2.0, -1.0, 8.0],
            [4.0, 0.5, 2.0],
        ]
    )
    right = np.asarray(
        [
            [0.5, -1.0, 1.0],
            [1.0, 1.0, 2.0],
            [1.0, -0.5, 8.0],
        ]
    )
    descriptors = riemann_descriptors(
        left,
        right,
        gamma=1.4,
        case_numbers=(0, 1, 2),
    )
    first = descriptors["case_000"]
    assert first[0] == pytest.approx(np.log(2.0))
    assert first[1] == pytest.approx(np.log(4.0))
    sound_left = np.sqrt(1.4 * 4.0)
    sound_right = np.sqrt(1.4 * 1.0 / 0.5)
    assert first[2] == pytest.approx(3.0 / (sound_left + sound_right))


def _state_crossfit_fixture() -> tuple[
    dict[str, np.ndarray], dict[str, dict[str, dict[str, float]]]
]:
    descriptors = {}
    statistics = {scope: {} for scope in PHASE_SCOPES}
    for index in range(16):
        case_id = f"case_{index:02d}"
        descriptor = np.asarray(
            (
                (index - 7.5) / 5.0,
                np.sin(0.4 * index),
                np.cos(0.7 * index),
            ),
            dtype=np.float64,
        )
        descriptors[case_id] = descriptor
        feature_square = 1.0 + 0.02 * index
        target_square = 20.0 + 0.1 * index
        early_beta = -0.4 + np.asarray((0.3, -0.2, 0.1)) @ descriptor
        late_beta = 0.5 + np.asarray((-0.15, 0.25, 0.2)) @ descriptor
        for scope, beta in zip(PHASE_SCOPES, (early_beta, late_beta), strict=True):
            statistics[scope][case_id] = {
                "feature_square": feature_square,
                "target_square": target_square,
                "cross": beta * feature_square,
            }
    return descriptors, statistics


def test_state_map_recovers_phase_by_descriptor_coefficients() -> None:
    descriptors, statistics = _state_crossfit_fixture()
    case_ids = tuple(sorted(descriptors))
    fit = _fit_state_map(
        descriptors,
        statistics,
        train_case_ids=case_ids[:-1],
        ridge=1.0e-4,
    )
    prediction = _predict_coefficients(fit, descriptors, query_case_id=case_ids[-1])
    expected_early = (
        statistics["calls_0_49"][case_ids[-1]]["cross"]
        / statistics["calls_0_49"][case_ids[-1]]["feature_square"]
    )
    expected_late = (
        statistics["calls_50_99"][case_ids[-1]]["cross"]
        / statistics["calls_50_99"][case_ids[-1]]["feature_square"]
    )
    assert prediction["calls_0_49"] == pytest.approx(expected_early, abs=2.0e-5)
    assert prediction["calls_50_99"] == pytest.approx(expected_late, abs=2.0e-5)
    support = _support_audit(fit, descriptors, query_case_id=case_ids[-1])
    assert support["nearest_training_distance"] >= 0.0
    assert support["training_support_radius"] > 0.0


def test_nested_state_crossfit_is_case_held_out_and_improves_fixture() -> None:
    descriptors, statistics = _state_crossfit_fixture()
    result = nested_state_crossfit(descriptors, statistics)
    assert len(result["folds"]) == 16
    assert all(
        fold["held_out_case_id"] not in fold["fit_case_ids"] for fold in result["folds"]
    )
    assert result["guarded_score"]["combined"]["skill_vs_zero"] > 0.01
    assert result["guarded_score"]["case_win_count"] >= 12
    assert result["full_population_selection"]["selected_ridge"] in (
        1.0e-4,
        1.0e-2,
        1.0,
        100.0,
    )
    payload = _with_json_payload_sha256({"nested_crossfit": result})
    encoded = json.dumps(payload, sort_keys=True, allow_nan=False)
    assert payload["payload_sha256"] in encoded


def test_state_support_guard_abstains_on_extrapolated_descriptor() -> None:
    descriptors, statistics = _state_crossfit_fixture()
    train = tuple(sorted(descriptors))
    fit = _fit_state_map(
        descriptors,
        statistics,
        train_case_ids=train,
        ridge=1.0,
    )
    extrapolated = dict(descriptors)
    extrapolated["outside"] = np.asarray((100.0, -100.0, 50.0))
    support = _support_audit(fit, extrapolated, query_case_id="outside")
    assert not support["supported"]
    assert support["support_margin"] < 0.0


@pytest.mark.parametrize("failure", ["duplicate", "thermodynamic", "index"])
def test_riemann_descriptors_fail_closed(failure: str) -> None:
    left = np.asarray([[1.0, 0.0, 1.0], [2.0, 0.2, 3.0]])
    right = np.asarray([[0.5, 0.0, 2.0], [1.5, -0.2, 1.0]])
    indices = (0, 1)
    if failure == "duplicate":
        left[1] = left[0]
        right[1] = right[0]
    elif failure == "thermodynamic":
        left[0, 0] = 0.0
    else:
        indices = (0, 2)
    with pytest.raises(ValueError):
        riemann_descriptors(left, right, gamma=1.4, case_numbers=indices)


def _fit_records(coefficient: float = -0.4) -> list[ScalarFitRecord]:
    cells = 16
    volumes = np.full(cells, 1.0 / cells)
    projector = build_cosine_projector(cells, rank=8, volumes=volumes)
    x = (np.arange(cells, dtype=np.float64) + 0.5) / cells
    records = []
    for case in range(8):
        for call in range(2):
            amplitude = 0.1 + 0.01 * case + 0.02 * call
            feature = amplitude * np.column_stack(
                (
                    np.cos(np.pi * x),
                    np.cos(2.0 * np.pi * x),
                    np.cos(3.0 * np.pi * x),
                )
            )
            feature = projector.project(
                feature,
                active_modes=tuple(range(1, 8)),
                component_scale=PHYSICAL_COMPONENT_SCALES,
            )
            records.append(
                ScalarFitRecord(
                    case_id=f"case_{case}",
                    input_call=call,
                    feature=feature,
                    target_correction=coefficient * feature,
                    native_increment=np.ones_like(feature),
                    volumes=volumes,
                    component_scale=PHYSICAL_COMPONENT_SCALES,
                )
            )
    return records


def test_case_first_scalar_fit_crossfit_and_oracle_score() -> None:
    records = _fit_records()
    fit = fit_case_first_scalar(records)
    assert fit.status == "ok"
    assert fit.coefficient == pytest.approx(-0.4, abs=1.0e-14)

    corrections, audits = corrections_for_coefficient(records, fit.coefficient)
    score = score_records(corrections)
    assert score["skill_vs_zero"] == pytest.approx(1.0)
    assert score["rms_ratio_vs_zero"] == pytest.approx(0.0, abs=1.0e-14)
    assert all(row["status"] == "ok" for row in audits)

    groups = {
        "fold_0": tuple(f"case_{case}" for case in range(4)),
        "fold_1": tuple(f"case_{case}" for case in range(4, 8)),
    }
    crossfit = crossfit_case_groups(
        records,
        expected_groups=groups,
        expected_input_calls=(0, 1),
    )
    assert crossfit["status"] == "ok"
    assert crossfit["relative_iqr"] == pytest.approx(0.0, abs=1.0e-14)
    assert crossfit["oof_score"]["skill_vs_zero"] == pytest.approx(1.0)
    assert crossfit["fold_coefficients"] == pytest.approx([-0.4, -0.4])


def test_inventory_is_canonical_and_fails_closed() -> None:
    records = _fit_records()
    cases = tuple(f"case_{case}" for case in range(8))
    assert validate_record_inventory(
        list(reversed(records)),
        expected_case_ids=cases,
        expected_input_calls=(0, 1),
    ) == (tuple(sorted(cases)), (0, 1))

    with pytest.raises(ValueError, match="duplicate"):
        validate_record_inventory(
            [*records, records[0]],
            expected_case_ids=cases,
            expected_input_calls=(0, 1),
        )
    with pytest.raises(ValueError, match="inventory mismatch"):
        validate_record_inventory(
            records[:-1],
            expected_case_ids=cases,
            expected_input_calls=(0, 1),
        )
    with pytest.raises(ValueError, match="nonnegative integers"):
        validate_record_inventory(
            records,
            expected_case_ids=cases,
            expected_input_calls=(False, 1),
        )

    mixed = list(records)
    mixed[-1] = ScalarFitRecord(
        case_id=mixed[-1].case_id,
        input_call=mixed[-1].input_call,
        feature=mixed[-1].feature,
        target_correction=mixed[-1].target_correction,
        native_increment=mixed[-1].native_increment,
        volumes=2.0 * mixed[-1].volumes,
        component_scale=mixed[-1].component_scale,
    )
    with pytest.raises(ValueError, match="share native volumes"):
        crossfit_case_groups(
            mixed,
            expected_groups={
                "fold_0": tuple(f"case_{case}" for case in range(4)),
                "fold_1": tuple(f"case_{case}" for case in range(4, 8)),
            },
            expected_input_calls=(0, 1),
        )


def test_offline_masked_relation_fit_accepts_different_region_sizes() -> None:
    records = _fit_records()[:2]
    first = records[0]
    second = records[1]
    masked = [
        first,
        ScalarFitRecord(
            case_id=second.case_id,
            input_call=second.input_call,
            feature=second.feature[:-2],
            target_correction=second.target_correction[:-2],
            native_increment=second.native_increment[:-2],
            volumes=second.volumes[:-2],
            component_scale=second.component_scale,
        ),
    ]

    fit = fit_case_first_scalar(masked)

    assert fit.status == "ok"
    assert fit.coefficient == pytest.approx(-0.4)


def test_fitted_coefficient_is_not_identifiable_from_predictions_alone() -> None:
    records = _fit_records(coefficient=-0.4)
    shifted = [
        ScalarFitRecord(
            case_id=row.case_id,
            input_call=row.input_call,
            feature=row.feature,
            target_correction=row.target_correction + 0.7 * row.feature,
            native_increment=row.native_increment,
            volumes=row.volumes,
            component_scale=row.component_scale,
        )
        for row in records
    ]

    original = fit_case_first_scalar(records)
    alternative = fit_case_first_scalar(shifted)

    assert alternative.coefficient == pytest.approx(original.coefficient + 0.7)
    assert all(
        np.array_equal(first.feature, second.feature)
        and np.array_equal(first.native_increment, second.native_increment)
        for first, second in zip(records, shifted, strict=True)
    )


def test_thin_evaluator_dry_run_is_target_closed_and_repeatable() -> None:
    first = synthetic_summary(seed=7)
    second = synthetic_summary(seed=7)

    assert first == second
    assert first["status"] == "passed"
    assert all(first["checks"].values())
    assert first["payload_sha256"]


def test_exact_zero_no_harm_control_is_resolved_without_claiming_efficacy() -> None:
    rows = [
        {
            "case_id": f"case_{case}",
            "raw_integral_density": 0.0,
            "corrected_integral_density": 0.0,
        }
        for case in range(2)
    ]

    control = _case_first_control(rows, "integral_density")

    assert control["status"] == "exact_zero_no_change"
    assert control["ratio"] == 1.0
    assert all(row["status"] == "exact_zero_no_change" for row in control["case_rows"])
