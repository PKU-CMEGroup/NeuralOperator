from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from scripts.time_dependent_no import (
    evaluate_pcno_fine_discrepancy_correction as evaluator,
)
from scripts.time_dependent_no import (
    evaluate_pcno_modal_affine_transfer as affine_transfer,
)
from utility.time_dependent_no.pcno_cross_resolution_correction import (
    DiagnosticSnapshot,
    NativeIncrementBasis,
    ResolutionContract,
    prepare_common_native_inputs,
    prolong_nested_state,
)
from utility.time_dependent_no.pcno_cross_resolution_teacher_forced import (
    VIEW_SPECS,
    build_fixed_cosine_projector,
)
from utility.time_dependent_no.pcno_fine_discrepancy_correction import (
    MAX_CORRECTION_TO_NATIVE_INCREMENT,
    fine_discrepancy_correction,
    modal_coordinates,
    reconstruct_modal_field,
    recurrent_gate,
    score_precomputed_correction_inventory,
    select_calibration_policy,
    synchronized_fine_discrepancy_step,
    teacher_forced_gate,
)

NATIVE = (8, 4)
CONTRACT = ResolutionContract(coarse=(4, 2), native=NATIVE, fine=(16, 8))
SCALE = np.asarray((0.5, 0.75, 1.0, 1.25), dtype=np.float64)


def _grid():
    nx, ny = NATIVE
    x = (np.arange(nx, dtype=np.float64) + 0.5) * (2.0 / nx)
    y = (np.arange(ny, dtype=np.float64) + 0.5) * (1.0 / ny)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    nodes = np.column_stack((xx.reshape(-1), yy.reshape(-1)))
    volumes = np.linspace(0.8, 1.2, nx * ny, dtype=np.float64)
    volumes *= 2.0 / volumes.sum()
    node_type = np.zeros(nx * ny, dtype=np.int64)
    node_type[:nx] = 1
    projector = build_fixed_cosine_projector(
        nodes,
        volumes,
        node_type,
        rank=8,
    )
    return nodes, volumes, node_type, projector


def _basis(*, native_scale: float = 1.0, seed: int = 0) -> NativeIncrementBasis:
    rng = np.random.default_rng(seed)
    native = native_scale * rng.normal(size=(NATIVE[0] * NATIVE[1], 4))
    fine_feature = rng.normal(size=native.shape)
    return NativeIncrementBasis(
        native_increment=native,
        coarse_on_native=np.array(native, copy=True),
        fine_on_native=native + fine_feature,
        native_minus_coarse=np.zeros_like(native),
        fine_minus_native=fine_feature,
    )


def test_modal_coordinates_round_trip_to_weighted_parallel_field() -> None:
    _, _, _, projector = _grid()
    field = np.random.default_rng(1).normal(size=(NATIVE[0] * NATIVE[1], 4))
    coordinates = modal_coordinates(field, projector, component_scale=SCALE)
    reconstructed = reconstruct_modal_field(
        coordinates,
        projector,
        component_scale=SCALE,
    )
    pieces, _ = projector.split(field)

    assert coordinates.shape == (8, 4)
    assert np.allclose(reconstructed, pieces["parallel"], atol=2.0e-15, rtol=0.0)
    assert np.count_nonzero(reconstructed[~projector.interior_mask]) == 0


def test_rank7_policy_is_weighted_mean_neutral_and_zero_on_contacts() -> None:
    _, volumes, _, projector = _grid()
    correction, audit = fine_discrepancy_correction(
        _basis(native_scale=100.0, seed=2),
        projector,
        policy="rank7_fine_away_half",
        volumes=volumes,
        component_scale=SCALE,
    )
    scaled_mean = (
        np.einsum(
            "n,nc->c",
            volumes,
            correction / SCALE[None, :],
        )
        / volumes.sum()
    )
    coordinates = modal_coordinates(correction, projector, component_scale=SCALE)

    assert audit.status == "ok"
    assert audit.cap_active is False
    assert np.max(np.abs(scaled_mean)) <= 2.0e-16
    assert np.max(np.abs(coordinates[0])) <= 2.0e-16
    assert np.count_nonzero(correction[~projector.interior_mask]) == 0
    assert audit.maximum_modal_reconstruction_abs <= 2.0e-15


def test_rank8_policy_is_exact_fine_away_half_before_cap() -> None:
    _, volumes, _, projector = _grid()
    basis = _basis(native_scale=100.0, seed=3)
    correction, audit = fine_discrepancy_correction(
        basis,
        projector,
        policy="rank8_fine_away_half",
        volumes=volumes,
        component_scale=SCALE,
    )
    pieces, _ = projector.split(basis.fine_minus_native)

    assert audit.status == "ok"
    assert audit.applied_scale == 1.0
    assert np.allclose(
        correction,
        -0.5 * pieces["parallel"],
        atol=2.0e-15,
        rtol=0.0,
    )


def test_trust_region_caps_relative_norm_and_zero_is_exact() -> None:
    _, volumes, _, projector = _grid()
    basis = _basis(native_scale=0.01, seed=4)
    correction, audit = fine_discrepancy_correction(
        basis,
        projector,
        policy="rank8_fine_away_half",
        volumes=volumes,
        component_scale=SCALE,
    )
    zero, zero_audit = fine_discrepancy_correction(
        basis,
        projector,
        policy="zero",
        volumes=volumes,
        component_scale=SCALE,
    )

    assert audit.status == "ok"
    assert audit.cap_active is True
    assert audit.correction_to_native_increment == pytest.approx(
        MAX_CORRECTION_TO_NATIVE_INCREMENT
    )
    assert np.count_nonzero(correction) > 0
    assert np.array_equal(zero, np.zeros_like(basis.native_increment))
    assert zero_audit.status == "zero"

    unresolved_basis = replace(
        basis,
        native_increment=np.zeros_like(basis.native_increment),
    )
    abstained, unresolved = fine_discrepancy_correction(
        unresolved_basis,
        projector,
        policy="rank8_fine_away_half",
        volumes=volumes,
        component_scale=SCALE,
    )
    assert unresolved.status == "unresolved_zero_native_increment"
    assert np.count_nonzero(abstained) == 0


def test_synchronized_step_makes_native_then_fine_calls_from_one_state() -> None:
    _, volumes, _, projector = _grid()
    state = np.random.default_rng(5).normal(size=(NATIVE[0] * NATIVE[1], 4))
    expected = prepare_common_native_inputs(state, contract=CONTRACT)
    calls = []

    def predictor(resolution, value):
        calls.append((resolution, np.array(value, copy=True)))
        return value + 0.1

    step = synchronized_fine_discrepancy_step(
        state,
        contract=CONTRACT,
        projector=projector,
        predictor=predictor,
        policy="rank7_fine_away_half",
        volumes=volumes,
        component_scale=SCALE,
    )

    assert [resolution for resolution, _ in calls] == [CONTRACT.native, CONTRACT.fine]
    assert all(
        np.array_equal(value, expected.model_inputs[resolution])
        for resolution, value in calls
    )
    assert CONTRACT.coarse not in step.predictions
    assert np.allclose(
        step.next_native_state,
        step.predictions[CONTRACT.native] + step.correction,
        atol=0.0,
        rtol=0.0,
    )


def _calibration_evidence(*, rank8_ratio: float, rank7_ratio: float):
    common = {
        "inventory_exact": True,
        "full_skill_nonnegative": True,
        "rank8_skill_at_least_0p05": True,
        "minimum_eight_group_wins": True,
        "all_controls_no_harm": True,
        "all_proposals_finite_admissible": True,
        "all_audits_resolved": True,
        "closure_passed": True,
    }
    return {
        "rank8_fine_away_half": {**common, "rank8_rms_ratio": rank8_ratio},
        "rank7_fine_away_half": {**common, "rank8_rms_ratio": rank7_ratio},
    }


def test_selector_is_zero_inclusive_and_ties_prefer_rank7() -> None:
    selected = select_calibration_policy(
        _calibration_evidence(rank8_ratio=0.9, rank7_ratio=0.9)
    )
    assert selected["status"] == "qualified"
    assert selected["selected_policy"] == "rank7_fine_away_half"

    evidence = _calibration_evidence(rank8_ratio=0.8, rank7_ratio=0.9)
    evidence["rank8_fine_away_half"]["all_controls_no_harm"] = False
    selected = select_calibration_policy(evidence)
    assert selected["selected_policy"] == "rank7_fine_away_half"

    evidence["rank7_fine_away_half"]["minimum_eight_group_wins"] = False
    stopped = select_calibration_policy(evidence)
    assert stopped["status"] == "not_qualified"
    assert stopped["selected_policy"] == "zero"


def test_teacher_and_recurrent_gates_fail_closed_on_inventory_or_one_check() -> None:
    teacher_checks = {
        "calibration_qualified": True,
        "inventory_exact": True,
        "full_skill_nonnegative": True,
        "rank8_skill_at_least_0p05": True,
        "minimum_four_case_wins": True,
        "both_half_horizons_positive": True,
        "all_controls_no_harm": True,
        "all_proposals_finite_admissible": True,
        "all_audits_resolved": True,
        "closure_passed": True,
        "source_and_artifacts_exact": True,
    }
    assert teacher_forced_gate(teacher_checks)["recurrent_pilot_authorized"] is True
    failed = dict(teacher_checks)
    failed["all_controls_no_harm"] = False
    assert teacher_forced_gate(failed)["status"] == "stopped"
    with pytest.raises(ValueError, match="inventory"):
        teacher_forced_gate({**teacher_checks, "extra": True})

    recurrent_checks = {
        "teacher_gate_passed": True,
        "inventory_exact": True,
        "all_rollouts_complete_finite_admissible": True,
        "median_endpoint_ratio_at_most_0p98": True,
        "minimum_four_endpoint_wins": True,
        "maximum_endpoint_ratio_at_most_1p02": True,
        "aggregate_state_rms_ratio_at_most_0p99": True,
        "increment_and_cumulative_ratios_at_most_one": True,
        "all_controls_no_harm": True,
        "two_call_common_source_closure": True,
        "correction_audits_pass": True,
        "deterministic_prefix_exact": True,
        "source_and_artifacts_exact": True,
    }
    assert recurrent_gate(recurrent_checks)["adaptive_recurrent_pass"] is True
    recurrent_checks["minimum_four_endpoint_wins"] = False
    assert recurrent_gate(recurrent_checks)["status"] == "adaptive_recurrent_failed"


def test_synthetic_recurrence_cancels_persistent_rank7_error_without_truth_input() -> (
    None
):
    _, volumes, _, projector = _grid()
    rng = np.random.default_rng(6)
    modal = np.zeros((8, 4), dtype=np.float64)
    modal[1:, :] = rng.normal(scale=2.0e-3, size=(7, 4))
    bias = reconstruct_modal_field(modal, projector, component_scale=SCALE)
    exact_increment = 30.0 * bias
    fine_bias = prolong_nested_state(
        3.0 * bias,
        coarse_resolution=CONTRACT.native,
        fine_resolution=CONTRACT.fine,
    )
    fine_increment = prolong_nested_state(
        exact_increment,
        coarse_resolution=CONTRACT.native,
        fine_resolution=CONTRACT.fine,
    )

    def predictor(resolution, state):
        if resolution == CONTRACT.native:
            return state + exact_increment + bias
        if resolution == CONTRACT.fine:
            return state + fine_increment + fine_bias
        raise AssertionError("coarse prediction must not be requested")

    raw = np.zeros((NATIVE[0] * NATIVE[1], 4), dtype=np.float64)
    corrected = np.array(raw, copy=True)
    reference = np.array(raw, copy=True)
    for _ in range(5):
        raw = predictor(CONTRACT.native, raw)
        step = synchronized_fine_discrepancy_step(
            corrected,
            contract=CONTRACT,
            projector=projector,
            predictor=predictor,
            policy="rank7_fine_away_half",
            volumes=volumes,
            component_scale=SCALE,
        )
        corrected = step.next_native_state
        reference = reference + exact_increment
        assert step.audit.status == "ok"
        assert step.audit.cap_active is False

    raw_error = np.linalg.norm(raw - reference)
    corrected_error = np.linalg.norm(corrected - reference)
    assert raw_error > 0.0
    assert corrected_error <= 1.0e-5 * raw_error
    assert np.max(np.abs(corrected - reference)) <= 1.0e-7


def test_evaluator_synthetic_cli_and_exact_zero_control_are_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        evaluator,
        "SOURCE_PATHS",
        ("tests/time_dependent_no/test_pcno_fine_discrepancy_correction.py",),
    )
    payload = evaluator.synthetic_summary(seed=7)
    args = evaluator.parse_args(["synthetic", "--seed", "7"])
    exact_zero = evaluator._rms_control(
        key="front_thickness",
        scope="case",
        rows=({"zero_error": 0.0, "corrected_error": 0.0},),
    )
    small_denominator_harm = evaluator._rms_control(
        key="front_thickness",
        scope="case",
        rows=({"zero_error": 0.0, "corrected_error": 1.0e-4},),
    )

    assert args.command == "synthetic"
    assert args.seed == 7
    assert payload["status"] == "passed"
    assert all(payload["checks"].values())
    assert payload["checkpoint_loaded"] is False
    assert payload["dataset_loaded"] is False
    assert payload["reference_loaded"] is False
    assert exact_zero["status"] == "exact_zero_no_change"
    assert exact_zero["ratio"] == 1.0
    assert evaluator._control_passed(exact_zero)
    assert small_denominator_harm["status"] == "small_denominator_harm"
    assert not evaluator._control_passed(small_denominator_harm)


def test_evaluator_gates_every_registered_field_band_region_and_component() -> None:
    assert evaluator.FIELD_CONTROL_VIEWS == (
        "full",
        "band_large",
        "band_transition",
        "band_local",
        "region_boundary",
        "region_shock",
        "region_vortex",
        "region_smooth",
        "smooth_local",
        "component_density",
        "component_x_momentum",
        "component_y_momentum",
        "component_energy",
    )


def test_precomputed_correction_inventory_reuses_registered_views() -> None:
    _, volumes, _, projector = _grid()
    target = reconstruct_modal_field(
        np.arange(32, dtype=np.float64).reshape(8, 4),
        projector,
        component_scale=SCALE,
    )
    masks = {
        "boundary_le_0.05": np.ones(volumes.size, dtype=bool),
        "partition_shock": np.ones(volumes.size, dtype=bool),
        "partition_vortex": np.ones(volumes.size, dtype=bool),
        "partition_smooth": np.ones(volumes.size, dtype=bool),
    }
    snapshots = []
    corrections = []
    for case_id in ("case0", "case1"):
        for input_call in (0, 1):
            scale = 1.0 + 0.1 * input_call
            value = scale * target
            zero = np.zeros_like(value)
            snapshots.append(
                DiagnosticSnapshot(
                    case_id=case_id,
                    group_id="group",
                    input_call=input_call,
                    basis=NativeIncrementBasis(
                        native_increment=zero,
                        coarse_on_native=zero,
                        fine_on_native=zero,
                        native_minus_coarse=zero,
                        fine_minus_native=zero,
                    ),
                    target_correction=value,
                    volumes=volumes,
                    component_scale=SCALE,
                    masks=masks,
                )
            )
            corrections.append(0.5 * value)
    score = score_precomputed_correction_inventory(
        snapshots,
        corrections,
        label="synthetic",
        expected_case_ids=("case0", "case1"),
        expected_input_calls=(0, 1),
        resolution=NATIVE,
        projector=projector,
    )
    population = next(
        row
        for row in score["rows"]
        if row["scope"] == "population" and row["view"] == "full"
    )
    assert score["views"] == [view.key for view in VIEW_SPECS]
    assert population["skill_vs_zero"] == pytest.approx(0.75)
    assert population["rms_ratio_vs_zero"] == pytest.approx(0.5)
    assert score["maximum_closure"]["band"] <= 1.0e-12


def test_affine_transfer_gate_fails_closed_on_control_or_admissibility() -> None:
    rows = []
    for case_id in affine_transfer.CASE_IDS:
        for view in ("full", "rank8_parallel"):
            rows.append(
                {
                    "view": view,
                    "scope": "case",
                    "case_id": case_id,
                    "skill_status": "ok",
                    "skill_vs_zero": 0.1,
                }
            )
    rows.extend(
        {
            "view": view,
            "scope": "population",
            "case_id": None,
            "skill_status": "ok",
            "skill_vs_zero": 0.1,
        }
        for view in ("full", "rank8_parallel")
    )
    score_payloads = {
        label: {"rows": list(rows)} for label, *_ in affine_transfer._population_specs()
    }
    audit = {
        "status": "ok",
        "correction_to_native_increment": 0.01,
        "maximum_scaled_component_mean_abs": 0.0,
        "maximum_excluded_abs": 0.0,
        "maximum_inactive_coordinate_abs": 0.0,
        "maximum_modal_reconstruction_abs": 0.0,
    }
    nesting = {
        "pre_model_coarse_from_native_max_abs": 0.0,
        "pre_model_fine_to_native_max_abs": 0.0,
        "post_fp32_coarse_from_native_max_abs": 0.0,
        "post_fp32_fine_to_native_max_abs": 0.0,
    }
    execution = {"maximum_repeat_abs_difference": 0.0}
    good = affine_transfer._prospective_gate(
        score_payloads=score_payloads,
        controls=[{"status": "ok", "ratio": 1.0}],
        proposal_rows=[{"finite": True, "admissible": True}],
        audit_rows=[audit],
        nesting_rows=[nesting],
        execution=execution,
        structural_checks={"inventory": True},
    )
    bad = affine_transfer._prospective_gate(
        score_payloads=score_payloads,
        controls=[{"status": "ok", "ratio": 1.05 + 1.0e-12}],
        proposal_rows=[{"finite": True, "admissible": False}],
        audit_rows=[audit],
        nesting_rows=[nesting],
        execution=execution,
        structural_checks={"inventory": True},
    )

    assert good["status"] == "passed"
    assert bad["status"] == "failed"
    assert "controls_no_harm" in bad["failed_checks"]
    assert "proposals_finite_admissible" in bad["failed_checks"]


def test_modal_diagnostic_reconstructs_collinear_grid_error_triplet() -> None:
    _, volumes, _, projector = _grid()
    modal = np.zeros((8, 4), dtype=np.float64)
    modal[1, 0] = 2.0
    target = reconstruct_modal_field(modal, projector, component_scale=SCALE)
    snapshots = []
    for input_call, multiplier in enumerate((1.0, 2.0)):
        current_target = multiplier * target
        coarse = 0.2 * current_target
        fine = -0.5 * current_target
        zero = np.zeros_like(current_target)
        snapshots.append(
            DiagnosticSnapshot(
                case_id="case",
                group_id="group",
                input_call=input_call,
                basis=NativeIncrementBasis(
                    native_increment=zero,
                    coarse_on_native=zero,
                    fine_on_native=zero,
                    native_minus_coarse=coarse,
                    fine_minus_native=fine,
                ),
                target_correction=current_target,
                volumes=volumes,
                component_scale=SCALE,
                masks={},
            )
        )

    _, summary = evaluator._modal_records(snapshots, projector)
    relation = next(row for row in summary["relation_rows"] if row["scope"] == "all")

    assert relation["feature_relation_status"] == "ok"
    assert relation["fine_fit_status"] == "ok"
    assert relation["coarse_innovation_status"] == "small_denominator"
    assert relation["coarse_native_error_cosine"] == pytest.approx(1.0)
    assert relation["native_fine_error_cosine"] == pytest.approx(1.0)
    assert relation["coarse_fine_error_cosine"] == pytest.approx(1.0)
    assert relation["coarse_native_error_left_to_right_norm_ratio"] == pytest.approx(
        1.2
    )
    assert relation["native_fine_error_left_to_right_norm_ratio"] == pytest.approx(
        2.0 / 3.0
    )
    assert relation["common_grid_error_rank1_energy_fraction"] == pytest.approx(1.0)
    assert summary["target_modal_matrix_rank"] == 1
    assert summary["target_modal_rank_status"] == "ok"
    assert summary["median_fine_consecutive_cosine"] == pytest.approx(1.0)
    assert summary["median_target_consecutive_cosine"] == pytest.approx(1.0)


def test_execution_accounting_separates_native_and_fine_costs() -> None:
    execution = {
        "logical_model_calls": 0,
        "actual_forward_passes": 0,
        "total_forward_seconds": 0.0,
        "maximum_peak_gpu_memory_bytes": 0,
        "native_logical_calls": 0,
        "fine_logical_calls": 0,
        "native_actual_forward_passes": 0,
        "fine_actual_forward_passes": 0,
        "native_forward_seconds": 0.0,
        "fine_forward_seconds": 0.0,
    }
    evaluator._account_execution(
        execution,
        {"forward_seconds": (0.1, 0.2), "peak_gpu_memory_bytes": 10},
        resolution=evaluator.NATIVE_RESOLUTION,
    )
    evaluator._account_execution(
        execution,
        {"forward_seconds": (0.4,), "peak_gpu_memory_bytes": 20},
        resolution=evaluator.RESOLUTION_CONTRACT.fine,
    )

    assert execution["logical_model_calls"] == 2
    assert execution["actual_forward_passes"] == 3
    assert execution["native_logical_calls"] == 1
    assert execution["fine_logical_calls"] == 1
    assert execution["native_actual_forward_passes"] == 2
    assert execution["fine_actual_forward_passes"] == 1
    assert execution["native_forward_seconds"] == pytest.approx(0.3)
    assert execution["fine_forward_seconds"] == pytest.approx(0.4)
    assert execution["maximum_peak_gpu_memory_bytes"] == 20


def test_rollout_controls_cover_trajectory_and_endpoint_inventories() -> None:
    endpoint_keys = (
        "rank8_state_error",
        "boundary_state_error",
        "shock_state_error",
        "vortex_state_error",
        "smooth_state_error",
        "front_position",
        "front_strength_log_ratio",
        "front_thickness_log_ratio",
        "component_density_state_error",
        "component_x_momentum_state_error",
        "component_y_momentum_state_error",
        "component_energy_state_error",
    )
    rows = []
    for case_id in evaluator.EVALUATION_CASE_IDS:
        for input_call in evaluator.ALL_INPUT_CALLS:
            for policy, value in (("zero", 1.0), ("rank7_fine_away_half", 0.9)):
                row = {
                    "case_id": case_id,
                    "input_call": input_call,
                    "policy": policy,
                    "state_error": value,
                    "increment_defect": value,
                    "cumulative_defect": value,
                }
                row.update({key: value for key in endpoint_keys})
                row.update(
                    {
                        f"integral_{name}_error": value
                        for name in evaluator.INTEGRAL_COMPONENT_NAMES
                    }
                )
                rows.append(row)

    _, controls, population = evaluator._paired_rollout_controls(rows)
    control_keys = {row["key"] for row in controls}

    assert len(controls) == 224
    assert "trajectory::rank8_state_error" in control_keys
    assert "endpoint::rank8_state_error" in control_keys
    assert "trajectory::front_position" in control_keys
    assert "endpoint::component_energy_state_error" in control_keys
    assert "integral_rms::density" in control_keys
    assert "integral_endpoint::energy" in control_keys
    assert all(evaluator._control_passed(row) for row in controls)
    assert population["median_endpoint_state_ratio"] == pytest.approx(0.9)
