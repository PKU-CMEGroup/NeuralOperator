from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import numpy as np
import pytest

from scripts.time_dependent_no import (
    evaluate_pcno_cross_resolution_correction as evaluator,
)
from utility.time_dependent_no import (
    pcno_cross_resolution_correction as cross_resolution,
)
from utility.time_dependent_no.pcno_resolution_transfer import (
    restrict_nested_state,
)


def _contract() -> cross_resolution.ResolutionContract:
    return cross_resolution.ResolutionContract(
        coarse=(2, 1),
        native=(4, 2),
        fine=(8, 4),
    )


def _basis(
    coarse_feature: np.ndarray,
    fine_feature: np.ndarray,
) -> cross_resolution.NativeIncrementBasis:
    zero = np.zeros_like(coarse_feature, dtype=np.float64)
    return cross_resolution.NativeIncrementBasis(
        native_increment=zero,
        coarse_on_native=zero,
        fine_on_native=zero,
        native_minus_coarse=np.asarray(coarse_feature, dtype=np.float64),
        fine_minus_native=np.asarray(fine_feature, dtype=np.float64),
    )


def _snapshot(
    *,
    case_id: str,
    group_id: str,
    input_call: int,
    coarse_feature: np.ndarray,
    fine_feature: np.ndarray,
    target: np.ndarray,
    masks: dict[str, np.ndarray] | None = None,
) -> cross_resolution.DiagnosticSnapshot:
    nodes = target.shape[0]
    return cross_resolution.DiagnosticSnapshot(
        case_id=case_id,
        group_id=group_id,
        input_call=input_call,
        basis=_basis(coarse_feature, fine_feature),
        target_correction=np.asarray(target, dtype=np.float64),
        volumes=np.full(nodes, 2.0 / nodes, dtype=np.float64),
        component_scale=np.asarray((0.5, 0.75, 1.0, 1.25)),
        masks={} if masks is None else masks,
    )


def test_nested_maps_preserve_integrals_leading_dims_and_projectors() -> None:
    rng = np.random.default_rng(1)
    contract = _contract()
    coarse = rng.normal(size=(3, 2, 4)).astype(np.float32)

    native = cross_resolution.prolong_nested_state(
        coarse,
        coarse_resolution=contract.coarse,
        fine_resolution=contract.native,
    )
    recovered = restrict_nested_state(
        native,
        fine_resolution=contract.native,
        coarse_resolution=contract.coarse,
    )
    assert native.shape == (3, 8, 4)
    assert native.dtype == np.float64
    assert np.array_equal(recovered, coarse.astype(np.float64))

    coarse_volume = np.full(2, 2.0 / 2)
    native_volume = np.full(8, 2.0 / 8)
    np.testing.assert_allclose(
        np.einsum("n,tnc->tc", coarse_volume, coarse),
        np.einsum("n,tnc->tc", native_volume, native),
        rtol=0.0,
        atol=1.0e-14,
    )

    fine = rng.normal(size=(32, 4))
    restricted = restrict_nested_state(
        fine,
        fine_resolution=contract.fine,
        coarse_resolution=contract.native,
    )
    projected = cross_resolution.prolong_nested_state(
        restricted,
        coarse_resolution=contract.native,
        fine_resolution=contract.fine,
    )
    projected_twice = cross_resolution.prolong_nested_state(
        restrict_nested_state(
            projected,
            fine_resolution=contract.fine,
            coarse_resolution=contract.native,
        ),
        coarse_resolution=contract.native,
        fine_resolution=contract.fine,
    )
    np.testing.assert_allclose(projected_twice, projected, rtol=0.0, atol=0.0)


def test_direction_specific_transfer_floors_close_for_both_directions() -> None:
    rng = np.random.default_rng(2)
    contract = _contract()
    native = rng.normal(size=(8, 4))

    coarse_query = restrict_nested_state(
        native,
        fine_resolution=contract.native,
        coarse_resolution=contract.coarse,
    )
    coarse_floors = cross_resolution.transfer_floor_fields(
        coarse_query,
        native,
        query_resolution=contract.coarse,
        native_resolution=contract.native,
    )
    assert np.max(np.abs(coarse_floors.query_round_trip)) == 0.0
    assert np.max(np.abs(coarse_floors.pipeline_truth_floor)) == 0.0
    assert np.max(np.abs(coarse_floors.native_information_mismatch)) > 0.0
    assert np.max(np.abs(coarse_floors.closure_residual)) < 1.0e-15

    fine_base = cross_resolution.prolong_nested_state(
        native,
        coarse_resolution=contract.native,
        fine_resolution=contract.fine,
    )
    fine_variation = rng.normal(size=fine_base.shape)
    fine_variation -= cross_resolution.prolong_nested_state(
        restrict_nested_state(
            fine_variation,
            fine_resolution=contract.fine,
            coarse_resolution=contract.native,
        ),
        coarse_resolution=contract.native,
        fine_resolution=contract.fine,
    )
    fine_query = fine_base + fine_variation
    fine_floors = cross_resolution.transfer_floor_fields(
        fine_query,
        native,
        query_resolution=contract.fine,
        native_resolution=contract.native,
    )
    assert np.max(np.abs(fine_floors.native_information_mismatch)) < 1.0e-15
    assert np.max(np.abs(fine_floors.query_round_trip)) > 0.0
    assert np.max(np.abs(fine_floors.pipeline_truth_floor)) > 0.0
    np.testing.assert_allclose(
        fine_floors.query_round_trip,
        fine_floors.pipeline_truth_floor,
        rtol=0.0,
        atol=1.0e-14,
    )
    assert np.max(np.abs(fine_floors.closure_residual)) < 1.0e-14


@pytest.mark.parametrize(
    "coarse,native,fine",
    [
        ((4, 2), (4, 4), (8, 8)),
        ((4, 2), (8, 4), (8, 8)),
        ((4, 2), (8, 3), (16, 6)),
    ],
)
def test_resolution_roles_are_strict_and_nested(
    coarse: tuple[int, int],
    native: tuple[int, int],
    fine: tuple[int, int],
) -> None:
    with pytest.raises(ValueError):
        cross_resolution.ResolutionContract(
            coarse=coarse,
            native=native,
            fine=fine,
        )


def test_native_basis_maps_increments_not_full_states() -> None:
    rng = np.random.default_rng(3)
    contract = _contract()
    native_current = rng.normal(size=(8, 4))
    prepared = cross_resolution.prepare_common_native_inputs(
        native_current,
        contract=contract,
    )
    coarse_increment = rng.normal(size=(2, 4))
    native_increment = rng.normal(size=(8, 4))
    fine_increment = rng.normal(size=(32, 4))
    predictions = {
        contract.coarse: prepared.model_inputs[contract.coarse] + coarse_increment,
        contract.native: prepared.model_inputs[contract.native] + native_increment,
        contract.fine: prepared.model_inputs[contract.fine] + fine_increment,
    }

    basis = cross_resolution.common_native_increment_basis(
        prepared,
        predictions,
        contract=contract,
    )
    coarse_increment = (
        predictions[contract.coarse] - prepared.model_inputs[contract.coarse]
    )
    native_increment = (
        predictions[contract.native] - prepared.model_inputs[contract.native]
    )
    fine_increment = predictions[contract.fine] - prepared.model_inputs[contract.fine]
    expected_coarse = cross_resolution.prolong_nested_state(
        coarse_increment,
        coarse_resolution=contract.coarse,
        fine_resolution=contract.native,
    )
    expected_fine = restrict_nested_state(
        fine_increment,
        fine_resolution=contract.fine,
        coarse_resolution=contract.native,
    )
    np.testing.assert_allclose(basis.coarse_on_native, expected_coarse)
    np.testing.assert_allclose(basis.fine_on_native, expected_fine)
    np.testing.assert_allclose(
        basis.native_minus_coarse,
        native_increment - expected_coarse,
    )
    np.testing.assert_allclose(
        basis.fine_minus_native,
        expected_fine - native_increment,
    )


def test_common_native_preparation_records_fp32_floors_and_rejects_tampering() -> None:
    rng = np.random.default_rng(31)
    contract = _contract()
    native = rng.normal(size=(8, 4))
    prepared = cross_resolution.prepare_common_native_inputs(
        native,
        contract=contract,
    )
    assert prepared.nesting_floors["pre_model_coarse_from_native_max_abs"] == 0.0
    assert prepared.nesting_floors["pre_model_fine_to_native_max_abs"] == 0.0
    assert np.isfinite(prepared.nesting_floors["post_fp32_coarse_from_native_max_abs"])
    assert np.isfinite(prepared.nesting_floors["post_fp32_fine_to_native_max_abs"])

    direct_post_coarse = prepared.model_inputs[contract.coarse] - restrict_nested_state(
        prepared.model_inputs[contract.native],
        fine_resolution=contract.native,
        coarse_resolution=contract.coarse,
    )
    assert prepared.nesting_floors[
        "post_fp32_coarse_from_native_max_abs"
    ] == pytest.approx(float(np.max(np.abs(direct_post_coarse))))

    predictions = {
        resolution: np.array(value, copy=True)
        for resolution, value in prepared.model_inputs.items()
    }
    tampered_inputs = dict(prepared.model_inputs)
    tampered_inputs[contract.coarse] = np.array(
        tampered_inputs[contract.coarse], copy=True
    )
    tampered_inputs[contract.coarse][0, 0] += 1.0
    with pytest.raises(ValueError, match="common-native FP32 contract"):
        cross_resolution.common_native_increment_basis(
            replace(prepared, model_inputs=tampered_inputs),
            predictions,
            contract=contract,
        )
    tampered_floors = dict(prepared.nesting_floors)
    tampered_floors["post_fp32_coarse_from_native_max_abs"] += 1.0
    with pytest.raises(ValueError, match="nesting floors"):
        cross_resolution.common_native_increment_basis(
            replace(prepared, nesting_floors=tampered_floors),
            predictions,
            contract=contract,
        )


def test_zero_correction_is_bitwise_native_prediction() -> None:
    rng = np.random.default_rng(4)
    native_prediction = rng.normal(size=(8, 4)).astype(np.float32)
    basis = _basis(
        rng.normal(size=(8, 4)),
        rng.normal(size=(8, 4)),
    )
    corrected = cross_resolution.corrected_native_prediction(
        native_prediction,
        basis,
        alpha=0.0,
        beta=0.0,
    )
    assert corrected.dtype == native_prediction.dtype
    assert np.array_equal(corrected, native_prediction)
    assert corrected is not native_prediction


def test_offline_mapped_error_relationships_are_signed_and_labeled() -> None:
    contract = _contract()
    native_reference = np.ones((8, 4))
    shared_error = np.linspace(0.1, 0.8, 8)[:, None] * np.ones((1, 4))
    basis = cross_resolution.NativeIncrementBasis(
        native_increment=native_reference + shared_error,
        coarse_on_native=native_reference + 2.0 * shared_error,
        fine_on_native=native_reference + 3.0 * shared_error,
        native_minus_coarse=-shared_error,
        fine_minus_native=2.0 * shared_error,
    )
    errors = cross_resolution.mapped_increment_errors(
        basis,
        native_reference_increment=native_reference,
        contract=contract,
    )
    np.testing.assert_allclose(
        basis.native_minus_coarse,
        errors["native"] - errors["coarse_on_native"],
    )
    np.testing.assert_allclose(
        basis.fine_minus_native,
        errors["fine_on_native"] - errors["native"],
    )
    np.testing.assert_allclose(
        -errors["native"], native_reference - basis.native_increment
    )
    rows = cross_resolution.predicted_error_relationships(
        errors,
        volumes=np.full(8, 0.25),
        component_scale=np.ones(4),
    )
    assert len(rows) == 3
    assert all(row["offline_only"] is True for row in rows)
    assert all(row["signed_cosine"] == pytest.approx(1.0) for row in rows)
    assert all(row["pearson"] == pytest.approx(1.0) for row in rows)
    assert all(row["signed_cosine_status"] == "ok" for row in rows)
    assert all(row["pearson_status"] == "ok" for row in rows)
    assert all(row["signed_cosine_denominator"] > 0.0 for row in rows)
    assert all(row["pearson_denominator"] > 0.0 for row in rows)

    unresolved = cross_resolution.predicted_error_relationships(
        {name: np.zeros((8, 4)) for name in errors},
        volumes=np.full(8, 0.25),
        component_scale=np.ones(4),
    )
    assert all(row["signed_cosine"] is None for row in unresolved)
    assert all(row["pearson"] is None for row in unresolved)
    assert all(row["signed_cosine_denominator"] == 0.0 for row in unresolved)
    assert all(row["signed_cosine_status"] != "ok" for row in unresolved)
    assert all(row["pearson_status"] != "ok" for row in unresolved)


def test_statistics_are_additive_and_case_first() -> None:
    nodes = 8
    feature = np.ones((nodes, 4))
    zero = np.zeros_like(feature)
    case_rows = []
    raw_statistics = []
    for call in range(9):
        statistics = cross_resolution.scalar_correction_statistics(
            _basis(feature, zero),
            feature,
            volumes=np.ones(nodes),
            component_scale=np.ones(4),
        )
        raw_statistics.append(statistics)
        case_rows.append(("many_calls", statistics))
    other = cross_resolution.scalar_correction_statistics(
        _basis(feature, zero),
        3.0 * feature,
        volumes=np.ones(nodes),
        component_scale=np.ones(4),
    )
    raw_statistics.append(other)
    case_rows.append(("one_call", other))

    summed = cross_resolution.sum_scalar_correction_statistics(raw_statistics)
    assert summed.snapshot_count == 10
    np.testing.assert_allclose(
        summed.gram,
        np.sum([value.gram for value in raw_statistics], axis=0),
    )
    fit = cross_resolution.fit_scalar_correction(
        cross_resolution.case_first_statistics(case_rows),
        model="coarse_only",
    )
    assert fit.status == "ok"
    assert fit.coefficients == pytest.approx((2.0, 0.0))
    score = cross_resolution.score_scalar_correction(
        cross_resolution.case_first_statistics(case_rows),
        (2.0, 0.0),
    )
    assert score.zero_sse == pytest.approx(5.0)
    assert score.corrected_sse == pytest.approx(1.0)
    assert score.skill_vs_zero == pytest.approx(0.8)
    assert score.rms_ratio_vs_zero == pytest.approx(np.sqrt(0.2))


def test_scalar_scores_match_direct_weighted_centered_formulas() -> None:
    feature_values = np.asarray((-2.0, -0.5, 1.0, 3.0, 4.0))
    target_values = np.asarray((3.0, -1.0, 2.0, 7.0, 0.0))
    volumes = np.asarray((1.0, 2.0, 3.0, 4.0, 5.0))
    weights = volumes / volumes.sum()
    feature = np.zeros((feature_values.size, 4))
    target = np.zeros_like(feature)
    feature[:, 0] = feature_values
    target[:, 0] = target_values
    statistics = cross_resolution.scalar_correction_statistics(
        _basis(feature, np.zeros_like(feature)),
        target,
        volumes=volumes,
        component_scale=np.ones(4),
        components=(0,),
    )
    score = cross_resolution.score_scalar_correction(statistics, (1.0, 0.0))

    target_mean = float(np.dot(weights, target_values))
    feature_mean = float(np.dot(weights, feature_values))
    target_centered = target_values - target_mean
    feature_centered = feature_values - feature_mean
    zero_sse = float(np.dot(weights, np.square(target_values)))
    corrected_sse = float(np.dot(weights, np.square(target_values - feature_values)))
    centered_target_ss = float(np.dot(weights, np.square(target_centered)))
    centered_feature_ss = float(np.dot(weights, np.square(feature_centered)))
    expected_cosine = float(
        np.dot(weights, feature_values * target_values)
        / np.sqrt(
            np.dot(weights, np.square(feature_values))
            * np.dot(weights, np.square(target_values))
        )
    )
    expected_correlation = float(
        np.dot(weights, feature_centered * target_centered)
        / np.sqrt(centered_feature_ss * centered_target_ss)
    )

    assert score.status == "ok"
    assert score.zero_sse == pytest.approx(zero_sse)
    assert score.corrected_sse == pytest.approx(corrected_sse)
    assert score.target_rms == pytest.approx(np.sqrt(zero_sse))
    assert score.correction_rms == pytest.approx(
        np.sqrt(np.dot(weights, np.square(feature_values)))
    )
    assert score.target_centered_rms == pytest.approx(np.sqrt(centered_target_ss))
    assert score.correction_centered_rms == pytest.approx(np.sqrt(centered_feature_ss))
    assert score.skill_vs_zero == pytest.approx(1.0 - corrected_sse / zero_sse)
    assert score.centered_r2 == pytest.approx(1.0 - corrected_sse / centered_target_ss)
    assert score.cosine == pytest.approx(expected_cosine)
    assert score.correlation == pytest.approx(expected_correlation)
    assert score.cosine_denominator == pytest.approx(
        np.sqrt(
            np.dot(weights, np.square(feature_values))
            * np.dot(weights, np.square(target_values))
        )
    )
    assert score.correlation_denominator == pytest.approx(
        np.sqrt(centered_feature_ss * centered_target_ss)
    )
    assert score.skill_status == "ok"
    assert score.centered_r2_status == "ok"
    assert score.cosine_status == "ok"
    assert score.correlation_status == "ok"


def test_population_signed_metric_is_median_of_case_concatenations() -> None:
    case_metrics = []
    pooled_features = []
    pooled_targets = []
    for sign, scale in ((1.0, 1.0), (1.0, 1.0), (-1.0, 100.0)):
        feature = np.zeros((2, 4))
        target = np.zeros_like(feature)
        feature[:, 0] = scale * np.asarray((-1.0, 1.0))
        target[:, 0] = sign * feature[:, 0]
        case_metrics.append(
            cross_resolution.field_relation_metrics(
                feature,
                target,
                volumes=np.ones(2),
                component_scale=np.ones(4),
                components=(0,),
            )["signed_cosine"]
        )
        pooled_features.append(feature)
        pooled_targets.append(target)
    pooled = cross_resolution.field_relation_metrics(
        np.concatenate(pooled_features),
        np.concatenate(pooled_targets),
        volumes=np.ones(6),
        component_scale=np.ones(4),
        components=(0,),
    )["signed_cosine"]
    assert np.median(case_metrics) == pytest.approx(1.0)
    assert pooled == pytest.approx(-9998.0 / 10002.0)


@pytest.mark.parametrize("variation", [0.0, 1.0e-10])
def test_scalar_scores_report_unresolved_centered_denominators(
    variation: float,
) -> None:
    nodes = 8
    feature = np.zeros((nodes, 4))
    target = np.zeros_like(feature)
    feature[:, 0] = np.linspace(-1.0, 1.0, nodes)
    target[:, 0] = 2.0 + variation * np.linspace(-1.0, 1.0, nodes)
    statistics = cross_resolution.scalar_correction_statistics(
        _basis(feature, np.zeros_like(feature)),
        target,
        volumes=np.ones(nodes),
        component_scale=np.ones(4),
        components=(0,),
    )
    score = cross_resolution.score_scalar_correction(statistics, (1.0, 0.0))
    assert score.skill_status == "ok"
    assert score.centered_r2 is None
    assert score.correlation is None
    assert score.centered_r2_status == "zero_centered_target_denominator"
    assert score.correlation_status == "zero_centered_target_or_correction_denominator"
    assert score.status == "unresolved_metric_denominator"

    zero_score = cross_resolution.score_scalar_correction(
        statistics,
        (0.0, 0.0),
    )
    assert zero_score.cosine is None
    assert zero_score.cosine_status == "zero_target_or_correction_denominator"
    assert zero_score.correction_rms == 0.0
    assert zero_score.status == "unresolved_metric_denominator"


def test_denominator_floor_is_closed_at_equality() -> None:
    nodes = 4
    feature = np.zeros((nodes, 4))
    target = np.zeros_like(feature)
    feature[:, 0] = 1.0
    target[:, 0] = 1.0e-8
    statistics = cross_resolution.scalar_correction_statistics(
        _basis(feature, np.zeros_like(feature)),
        target,
        volumes=np.ones(nodes),
        component_scale=np.ones(4),
        components=(0,),
    )
    score = cross_resolution.score_scalar_correction(statistics, (1.0, 0.0))
    assert score.target_rms == pytest.approx(1.0e-8, rel=0.0, abs=0.0)
    assert score.skill_vs_zero is None
    assert score.skill_status == "zero_target_denominator"

    target[:, 0] = 2.0e-8
    resolved_statistics = cross_resolution.scalar_correction_statistics(
        _basis(feature, np.zeros_like(feature)),
        target,
        volumes=np.ones(nodes),
        component_scale=np.ones(4),
        components=(0,),
    )
    resolved = cross_resolution.score_scalar_correction(
        resolved_statistics,
        (1.0, 0.0),
    )
    assert resolved.skill_status == "ok"


def test_grouped_crossfit_recovers_fixed_scalars_without_case_leakage() -> None:
    rng = np.random.default_rng(5)
    resolution = (4, 2)
    snapshots = []
    for group in range(3):
        for member in range(2):
            for call in range(2):
                coarse = rng.normal(size=(8, 4))
                fine = rng.normal(size=(8, 4))
                snapshots.append(
                    _snapshot(
                        case_id=f"g{group}_m{member}",
                        group_id=f"g{group}",
                        input_call=call,
                        coarse_feature=coarse,
                        fine_feature=fine,
                        target=0.3 * coarse - 0.2 * fine,
                    )
                )

    result = cross_resolution.grouped_crossfit(
        snapshots,
        resolution=resolution,
        expected_groups={
            f"g{group}": tuple(f"g{group}_m{member}" for member in range(2))
            for group in range(3)
        },
        expected_input_calls=(0, 1),
    )
    assert result["selected_model"] == "two_term"
    assert result["selected_coefficients"] == pytest.approx((0.3, -0.2))
    selected_rows = result["models"]["two_term"]
    assert selected_rows["scored_case_count"] == 6
    assert selected_rows["corrected_sse_case_sum"] == pytest.approx(0.0, abs=1.0e-14)
    assert selected_rows["median_oof_case_cosine"] == pytest.approx(1.0)
    for fold in result["models"]["two_term"]["folds"]:
        assert set(fold["held_out_case_ids"]).isdisjoint(fold["fit_case_ids"])
        assert all(
            not case_id.startswith(fold["held_out_group"])
            for case_id in fold["fit_case_ids"]
        )
        assert [row["case_id"] for row in fold["case_scores"]] == fold[
            "held_out_case_ids"
        ]
        assert fold["all_case_signed_metrics_resolved"] is True
        assert fold["median_case_cosine"] == pytest.approx(1.0)


def test_grouped_crossfit_rejects_duplicate_and_incomplete_case_call_design() -> None:
    rng = np.random.default_rng(51)
    expected_groups = {"g0": ("c0",), "g1": ("c1",)}
    expected_calls = (0, 1)
    snapshots = []
    for group_id, case_ids in expected_groups.items():
        for case_id in case_ids:
            for input_call in expected_calls:
                coarse = rng.normal(size=(8, 4))
                fine = rng.normal(size=(8, 4))
                snapshots.append(
                    _snapshot(
                        case_id=case_id,
                        group_id=group_id,
                        input_call=input_call,
                        coarse_feature=coarse,
                        fine_feature=fine,
                        target=0.3 * coarse - 0.2 * fine,
                    )
                )

    forward = cross_resolution.grouped_crossfit(
        snapshots,
        resolution=(4, 2),
        expected_groups=expected_groups,
        expected_input_calls=expected_calls,
    )
    backward = cross_resolution.grouped_crossfit(
        list(reversed(snapshots)),
        resolution=(4, 2),
        expected_groups=expected_groups,
        expected_input_calls=expected_calls,
    )
    assert forward == backward

    with pytest.raises(ValueError, match="duplicate case/call"):
        cross_resolution.grouped_crossfit(
            [*snapshots, snapshots[0]],
            resolution=(4, 2),
            expected_groups=expected_groups,
            expected_input_calls=expected_calls,
        )
    with pytest.raises(ValueError, match="nonnegative integers"):
        cross_resolution.grouped_crossfit(
            snapshots,
            resolution=(4, 2),
            expected_groups=expected_groups,
            expected_input_calls=(0.0, 1),
        )
    with pytest.raises(ValueError, match="wrong group"):
        cross_resolution.grouped_crossfit(
            [replace(snapshots[0], group_id="g1"), *snapshots[1:]],
            resolution=(4, 2),
            expected_groups=expected_groups,
            expected_input_calls=expected_calls,
        )
    with pytest.raises(ValueError, match="identical native volumes"):
        cross_resolution.grouped_crossfit(
            [
                *snapshots[:-1],
                replace(snapshots[-1], volumes=2.0 * snapshots[-1].volumes),
            ],
            resolution=(4, 2),
            expected_groups=expected_groups,
            expected_input_calls=expected_calls,
        )
    with pytest.raises(ValueError, match="inventory does not match"):
        cross_resolution.grouped_crossfit(
            snapshots[:-1],
            resolution=(4, 2),
            expected_groups=expected_groups,
            expected_input_calls=expected_calls,
        )


def test_rank_deficiency_and_small_denominators_fail_closed() -> None:
    nodes = 8
    zero = np.zeros((nodes, 4))
    ones = np.ones((nodes, 4))
    zero_statistics = cross_resolution.scalar_correction_statistics(
        _basis(zero, ones),
        ones,
        volumes=np.ones(nodes),
        component_scale=np.ones(4),
    )
    zero_fit = cross_resolution.fit_scalar_correction(
        zero_statistics,
        model="coarse_only",
    )
    assert zero_fit.status == "zero_denominator"
    assert zero_fit.coefficients is None

    collinear_statistics = cross_resolution.scalar_correction_statistics(
        _basis(ones, 2.0 * ones),
        3.0 * ones,
        volumes=np.ones(nodes),
        component_scale=np.ones(4),
    )
    collinear_fit = cross_resolution.fit_scalar_correction(
        collinear_statistics,
        model="two_term",
    )
    assert collinear_fit.status == "rank_deficient"
    assert collinear_fit.coefficients is None


def test_band_region_and_component_statistics_keep_exact_closure() -> None:
    rng = np.random.default_rng(6)
    resolution = (8, 4)
    coarse = rng.normal(size=(32, 4))
    fine = rng.normal(size=(32, 4))
    mask = np.zeros(32, dtype=bool)
    mask[:16] = True
    snapshot = _snapshot(
        case_id="case",
        group_id="group",
        input_call=0,
        coarse_feature=coarse,
        fine_feature=fine,
        target=0.5 * coarse - 0.1 * fine,
        masks={"left_half": mask},
    )
    statistics, closure = cross_resolution.snapshot_statistics(
        snapshot,
        resolution=resolution,
        band="large",
        region="left_half",
        components=(0, 3),
    )
    assert statistics.value_count == 32
    assert closure["maximum_reconstruction_abs_residual_scaled"] < 1.0e-12
    assert closure["maximum_instantaneous_energy_relative_closure"] < 1.0e-12
    with pytest.raises(ValueError, match="lacks required region"):
        cross_resolution.snapshot_statistics(
            snapshot,
            resolution=resolution,
            region="missing",
        )


def test_synchronized_step_calls_three_grids_from_one_native_state() -> None:
    rng = np.random.default_rng(7)
    contract = _contract()
    native = rng.normal(size=(8, 4))
    calls: list[tuple[tuple[int, int], np.ndarray]] = []

    def predictor(resolution: tuple[int, int], state: np.ndarray) -> np.ndarray:
        calls.append((resolution, np.array(state, copy=True)))
        return state + 0.01 * (resolution[0] / contract.native[0])

    step = cross_resolution.synchronized_fusion_step(
        native,
        contract=contract,
        predictor=predictor,
        alpha=0.0,
        beta=0.0,
    )
    assert [resolution for resolution, _ in calls] == [
        contract.coarse,
        contract.native,
        contract.fine,
    ]
    expected = cross_resolution.prepare_common_native_inputs(
        native,
        contract=contract,
    ).model_inputs
    for resolution, state in calls:
        assert np.array_equal(state, expected[resolution])
    assert np.array_equal(
        step.next_native_state,
        step.predictions[contract.native],
    )

    calls.clear()
    cross_resolution.synchronized_fusion_step(
        step.next_native_state,
        contract=contract,
        predictor=predictor,
        alpha=0.1,
        beta=-0.2,
    )
    assert len(calls) == 3
    next_expected = cross_resolution.prepare_common_native_inputs(
        step.next_native_state,
        contract=contract,
    ).model_inputs
    for resolution, state in calls:
        assert np.array_equal(state, next_expected[resolution])


def test_dry_run_and_synthetic_smoke_open_no_external_inputs(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        evaluator,
        "SOURCE_PATHS",
        ("tests/time_dependent_no/test_pcno_cross_resolution_correction.py",),
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("external array loading is forbidden in A1")

    monkeypatch.setattr(np, "load", forbidden)
    assert evaluator.main(["dry-run"]) == 0
    dry_run = json.loads(capsys.readouterr().out)
    assert dry_run["executes_checkpoint"] is False
    assert dry_run["executes_dataset_arrays"] is False
    assert dry_run["true_error_is_offline_only"] is True

    first = evaluator.synthetic_smoke_summary(seed=11)
    second = evaluator.synthetic_smoke_summary(seed=11)
    assert first == second
    assert first["status"] == "passed"
    assert first["selected_model"] == "two_term"
    assert first["selected_coefficients"] == pytest.approx((0.4, -0.25))
    assert first["checks"]["exactly_three_predictions"] is True
    assert first["checks"]["zero_prediction_exact"] is True
    assert all(first["checks"].values())
    assert set(first["source_identity"]["file_sha256"]) == {
        "tests/time_dependent_no/test_pcno_cross_resolution_correction.py"
    }
    assert first["source_identity"]["git"]["commit"] != "unknown"
    hashed = dict(first)
    expected_digest = hashed.pop("payload_sha256")
    actual_digest = hashlib.sha256(
        json.dumps(hashed, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert actual_digest == expected_digest


def test_synthetic_cli_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        evaluator,
        "synthetic_smoke_summary",
        lambda _seed: {"status": "failed"},
    )
    assert evaluator.main(["synthetic", "--seed", "3"]) == 1
