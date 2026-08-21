from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from scripts.time_dependent_no import (
    analyze_pcno_binary_position_phase_rescore as binary_rescore,
)
from scripts.time_dependent_no import (
    evaluate_pcno_affine_shadow_tether_rollout as shadow_tether_rollout,
)
from scripts.time_dependent_no import (
    evaluate_pcno_binary_position_phase_rollout as binary_rollout,
)
from scripts.time_dependent_no import (
    evaluate_pcno_modal_affine_transfer as affine_transfer,
)
from scripts.time_dependent_no import (
    evaluate_pcno_sparse_modal_correction as evaluator,
)
from utility.time_dependent_no.pcno_cross_resolution_correction import (
    DiagnosticSnapshot,
    NativeIncrementBasis,
    ResolutionContract,
    prepare_common_native_inputs,
    prolong_nested_state,
)
from utility.time_dependent_no.pcno_cross_resolution_teacher_forced import (
    build_fixed_cosine_projector,
)
from utility.time_dependent_no.pcno_fine_discrepancy_correction import (
    MAX_CORRECTION_TO_NATIVE_INCREMENT,
    modal_coordinates,
    reconstruct_modal_field,
)
from utility.time_dependent_no.pcno_sparse_modal_correction import (
    AFFINE_MODAL_POLICY,
    BINARY_POSITION_PHASE_CALLS,
    FROZEN_ACTIVE_CELLS,
    SPARSE_POLICY,
    affine_modal_correction,
    binary_position_phase_correction_active,
    masked_increment_basis,
    masked_modal_field,
    masked_snapshot,
    positive_half_skill_cells,
    sparse_modal_correction,
    synchronized_affine_modal_step,
    synchronized_binary_affine_modal_step,
    synchronized_shadow_tethered_binary_affine_step,
    synchronized_sparse_modal_step,
    validate_active_cells,
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
    projector = build_fixed_cosine_projector(nodes, volumes, node_type, rank=8)
    return volumes, projector


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


def _modal_rows() -> list[dict[str, object]]:
    rows = []
    active = set(FROZEN_ACTIVE_CELLS)
    for mode in range(8):
        for component in range(4):
            rows.append(
                {
                    "mode_index": mode,
                    "component": component,
                    "fixed_half_skill": (
                        0.25 if (mode, component) in active else -0.25
                    ),
                    "fixed_half_skill_status": "ok",
                }
            )
    return rows


def test_positive_half_skill_mask_is_exact_and_fails_closed() -> None:
    rows = _modal_rows()
    assert positive_half_skill_cells(rows) == FROZEN_ACTIVE_CELLS

    with pytest.raises(ValueError, match="missing"):
        positive_half_skill_cells(rows[:-1])
    with pytest.raises(ValueError, match="duplicate"):
        positive_half_skill_cells([*rows, dict(rows[0])])
    with pytest.raises(ValueError, match="unresolved"):
        positive_half_skill_cells(
            [
                replace_row
                if index
                else {**replace_row, "fixed_half_skill_status": "x"}
                for index, replace_row in enumerate(rows)
            ]
        )
    with pytest.raises(TypeError, match="numeric"):
        positive_half_skill_cells(
            [
                replace_row if index else {**replace_row, "fixed_half_skill": True}
                for index, replace_row in enumerate(rows)
            ]
        )
    with pytest.raises(ValueError, match="nonconstant"):
        validate_active_cells(((0, 0),), rank=8, components=4)


def test_mask_retains_exact_modal_cells_and_zeroes_contacts() -> None:
    _, projector = _grid()
    coordinates = np.arange(1, 33, dtype=np.float64).reshape(8, 4)
    field = reconstruct_modal_field(
        coordinates,
        projector,
        component_scale=SCALE,
    )
    masked = masked_modal_field(field, projector, component_scale=SCALE)
    actual = modal_coordinates(masked, projector, component_scale=SCALE)
    expected = np.zeros_like(coordinates)
    for mode, component in FROZEN_ACTIVE_CELLS:
        expected[mode, component] = coordinates[mode, component]

    assert np.allclose(actual, expected, atol=5.0e-14, rtol=0.0)
    assert np.count_nonzero(masked[~projector.interior_mask]) == 0


def test_masked_basis_and_snapshot_preserve_algebra_and_labels() -> None:
    volumes, projector = _grid()
    basis = _basis(seed=2)
    transformed = masked_increment_basis(
        basis,
        projector,
        component_scale=SCALE,
    )
    snapshot = DiagnosticSnapshot(
        case_id="case",
        group_id="group",
        input_call=3,
        basis=basis,
        target_correction=np.ones_like(basis.native_increment),
        volumes=volumes,
        component_scale=SCALE,
        masks={"all": np.ones(volumes.size, dtype=bool)},
    )
    transformed_snapshot = masked_snapshot(snapshot, projector)

    assert np.array_equal(
        transformed.fine_on_native,
        transformed.native_increment + transformed.fine_minus_native,
    )
    assert transformed_snapshot.case_id == snapshot.case_id
    assert transformed_snapshot.input_call == snapshot.input_call
    assert np.array_equal(
        transformed_snapshot.target_correction, snapshot.target_correction
    )
    assert np.array_equal(transformed_snapshot.masks["all"], snapshot.masks["all"])


def test_sparse_correction_has_mean_support_cap_and_exact_zero() -> None:
    volumes, projector = _grid()
    basis = _basis(native_scale=100.0, seed=3)
    correction, audit = sparse_modal_correction(
        basis,
        projector,
        policy=SPARSE_POLICY,
        volumes=volumes,
        component_scale=SCALE,
    )
    coordinates = modal_coordinates(correction, projector, component_scale=SCALE)
    inactive = {(mode, component) for mode in range(8) for component in range(4)} - set(
        FROZEN_ACTIVE_CELLS
    )

    assert audit.policy == SPARSE_POLICY
    assert audit.status == "ok"
    assert audit.cap_active is False
    assert max(abs(coordinates[cell]) for cell in inactive) <= 2.0e-15
    assert audit.maximum_scaled_component_mean_abs <= 2.0e-16
    assert np.count_nonzero(correction[~projector.interior_mask]) == 0

    small_basis = replace(
        basis,
        native_increment=1.0e-3 * basis.native_increment,
    )
    capped, capped_audit = sparse_modal_correction(
        small_basis,
        projector,
        policy=SPARSE_POLICY,
        volumes=volumes,
        component_scale=SCALE,
    )
    zero, zero_audit = sparse_modal_correction(
        basis,
        projector,
        policy="zero",
        volumes=volumes,
        component_scale=SCALE,
    )
    assert capped_audit.cap_active is True
    assert capped_audit.correction_to_native_increment == pytest.approx(
        MAX_CORRECTION_TO_NATIVE_INCREMENT
    )
    assert np.count_nonzero(capped) > 0
    assert np.array_equal(zero, np.zeros_like(zero))
    assert zero_audit.status == "zero"


def test_synchronized_step_uses_two_views_of_one_state_and_one_update() -> None:
    volumes, projector = _grid()
    state = np.random.default_rng(4).normal(size=(NATIVE[0] * NATIVE[1], 4))
    prepared = prepare_common_native_inputs(state, contract=CONTRACT)
    native_increment = np.full_like(state, 0.2)
    fine_native_increment = native_increment + masked_modal_field(
        np.random.default_rng(5).normal(size=state.shape),
        projector,
        component_scale=SCALE,
    )
    fine_increment = prolong_nested_state(
        fine_native_increment,
        coarse_resolution=NATIVE,
        fine_resolution=CONTRACT.fine,
    )
    calls: list[tuple[tuple[int, int], np.ndarray]] = []

    def predictor(resolution, value):
        calls.append((resolution, np.array(value, copy=True)))
        increment = native_increment if resolution == NATIVE else fine_increment
        return value + increment

    step = synchronized_sparse_modal_step(
        state,
        contract=CONTRACT,
        projector=projector,
        predictor=predictor,
        policy=SPARSE_POLICY,
        volumes=volumes,
        component_scale=SCALE,
    )

    assert [resolution for resolution, _ in calls] == [NATIVE, CONTRACT.fine]
    assert np.array_equal(calls[0][1], prepared.model_inputs[NATIVE])
    assert np.array_equal(calls[1][1], prepared.model_inputs[CONTRACT.fine])
    assert np.allclose(
        step.next_native_state,
        step.predictions[NATIVE] + step.correction,
        atol=1.0e-14,
        rtol=0.0,
    )
    coordinates = modal_coordinates(
        step.correction,
        projector,
        component_scale=SCALE,
    )
    assert np.max(np.abs(coordinates[0])) <= 2.0e-15


def test_affine_modal_correction_decodes_phase_cells_and_caps() -> None:
    volumes, projector = _grid()
    basis = _basis(native_scale=100.0, seed=7)
    cells = FROZEN_ACTIVE_CELLS
    start = -0.25 * np.eye(len(cells))
    end = -0.75 * np.eye(len(cells))
    correction, audit = affine_modal_correction(
        basis,
        projector,
        input_call=29,
        start_coefficients=start,
        end_coefficients=end,
        volumes=volumes,
        component_scale=SCALE,
    )
    feature = modal_coordinates(
        basis.fine_minus_native,
        projector,
        component_scale=SCALE,
    )
    actual = modal_coordinates(correction, projector, component_scale=SCALE)
    expected = np.zeros_like(actual)
    for cell in cells:
        expected[cell] = -0.75 * feature[cell]

    assert audit.policy == AFFINE_MODAL_POLICY
    assert audit.status == "ok"
    assert audit.normalized_phase == 1.0
    assert audit.cap_active is False
    assert np.allclose(actual, expected, atol=3.0e-14, rtol=0.0)
    assert audit.maximum_inactive_coordinate_abs == 0.0
    assert audit.maximum_excluded_abs == 0.0
    assert audit.maximum_scaled_component_mean_abs <= 3.0e-16

    small = replace(basis, native_increment=1.0e-4 * basis.native_increment)
    _, capped = affine_modal_correction(
        small,
        projector,
        input_call=0,
        start_coefficients=start,
        end_coefficients=end,
        volumes=volumes,
        component_scale=SCALE,
    )
    assert capped.cap_active is True
    assert capped.correction_to_native_increment == pytest.approx(
        MAX_CORRECTION_TO_NATIVE_INCREMENT
    )


def test_synchronized_affine_step_uses_two_common_state_queries() -> None:
    volumes, projector = _grid()
    state = np.random.default_rng(8).normal(size=(NATIVE[0] * NATIVE[1], 4))
    prepared = prepare_common_native_inputs(state, contract=CONTRACT)
    cells = FROZEN_ACTIVE_CELLS
    calls: list[tuple[tuple[int, int], np.ndarray]] = []

    def predictor(resolution, value):
        calls.append((resolution, np.array(value, copy=True)))
        return value + 0.1

    step = synchronized_affine_modal_step(
        state,
        input_call=14,
        contract=CONTRACT,
        projector=projector,
        predictor=predictor,
        start_coefficients=-0.25 * np.eye(len(cells)),
        end_coefficients=-0.75 * np.eye(len(cells)),
        volumes=volumes,
        component_scale=SCALE,
    )
    assert [resolution for resolution, _ in calls] == [NATIVE, CONTRACT.fine]
    assert np.array_equal(calls[0][1], prepared.model_inputs[NATIVE])
    assert np.array_equal(calls[1][1], prepared.model_inputs[CONTRACT.fine])
    assert np.array_equal(
        step.next_native_state,
        step.predictions[NATIVE] + step.correction,
    )


def test_binary_affine_step_skips_fine_and_is_exact_raw_when_inactive() -> None:
    volumes, projector = _grid()
    state = np.random.default_rng(18).normal(size=(NATIVE[0] * NATIVE[1], 4))
    cells = FROZEN_ACTIVE_CELLS
    calls: list[tuple[tuple[int, int], np.ndarray]] = []

    def predictor(resolution, value):
        calls.append((resolution, np.array(value, copy=True)))
        return value + 0.125

    inactive = synchronized_binary_affine_modal_step(
        state,
        position_selected=True,
        input_call=8,
        contract=CONTRACT,
        projector=projector,
        predictor=predictor,
        start_coefficients=-0.25 * np.eye(len(cells)),
        end_coefficients=-0.75 * np.eye(len(cells)),
        volumes=volumes,
        component_scale=SCALE,
    )
    assert [resolution for resolution, _ in calls] == [NATIVE]
    assert inactive.logical_call_count == 1
    assert inactive.correction_active is False
    assert inactive.audit.status == "inactive_exact_raw_native"
    assert np.count_nonzero(inactive.correction) == 0
    assert np.array_equal(
        inactive.next_native_state, inactive.predictions[NATIVE]
    )

    calls.clear()
    active = synchronized_binary_affine_modal_step(
        state,
        position_selected=True,
        input_call=22,
        contract=CONTRACT,
        projector=projector,
        predictor=predictor,
        start_coefficients=-0.25 * np.eye(len(cells)),
        end_coefficients=-0.75 * np.eye(len(cells)),
        volumes=volumes,
        component_scale=SCALE,
    )
    assert [resolution for resolution, _ in calls] == [NATIVE, CONTRACT.fine]
    prepared = prepare_common_native_inputs(state, contract=CONTRACT)
    assert np.array_equal(calls[0][1], prepared.model_inputs[NATIVE])
    assert np.array_equal(calls[1][1], prepared.model_inputs[CONTRACT.fine])
    assert active.logical_call_count == 2
    assert active.correction_active is True
    assert BINARY_POSITION_PHASE_CALLS == (*range(8), *range(22, 30))
    assert binary_position_phase_correction_active(
        position_selected=True, input_call=29
    )
    assert not binary_position_phase_correction_active(
        position_selected=False, input_call=29
    )
    with pytest.raises(TypeError, match="Boolean"):
        binary_position_phase_correction_active(
            position_selected=1, input_call=0
        )


def test_shadow_tethered_affine_step_preserves_raw_integrals_and_call_order() -> None:
    volumes, projector = _grid()
    rng = np.random.default_rng(31)
    accepted = rng.normal(size=(NATIVE[0] * NATIVE[1], 4))
    shadow = rng.normal(size=accepted.shape)
    cells = FROZEN_ACTIVE_CELLS
    calls: list[tuple[tuple[int, int], np.ndarray]] = []

    def predictor(resolution, value):
        calls.append((resolution, np.array(value, copy=True)))
        if resolution == CONTRACT.fine:
            return value + 0.2
        return 1.01 * value + 0.1

    step = synchronized_shadow_tethered_binary_affine_step(
        accepted,
        shadow,
        position_selected=True,
        input_call=22,
        contract=CONTRACT,
        projector=projector,
        predictor=predictor,
        start_coefficients=-0.25 * np.eye(len(cells)),
        end_coefficients=-0.75 * np.eye(len(cells)),
        volumes=volumes,
        residual_scale=SCALE,
        state_scale=SCALE,
    )
    assert [resolution for resolution, _ in calls] == [
        NATIVE,
        NATIVE,
        CONTRACT.fine,
    ]
    assert step.logical_call_count == 3
    np.testing.assert_allclose(
        step.next_native_state - step.shadow_prediction,
        step.retained_displacement,
        atol=1.0e-14,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        np.sum(volumes[:, None] * step.retained_displacement, axis=0),
        0.0,
        atol=1.0e-14,
        rtol=0.0,
    )
    np.testing.assert_array_equal(
        step.retained_displacement[~projector.interior_mask],
        np.zeros_like(step.retained_displacement[~projector.interior_mask]),
    )
    assert step.tether_audit.maximum_projection_idempotence_abs <= 1.0e-14
    assert step.maximum_update_identity_abs <= 1.0e-14


def test_shadow_tethered_inactive_selected_false_is_exact_raw_shadow() -> None:
    volumes, projector = _grid()
    state = np.random.default_rng(32).normal(
        size=(NATIVE[0] * NATIVE[1], 4)
    )
    calls: list[tuple[int, int]] = []

    def predictor(resolution, value):
        calls.append(resolution)
        return value + 0.125

    step = synchronized_shadow_tethered_binary_affine_step(
        state,
        state,
        position_selected=False,
        input_call=0,
        contract=CONTRACT,
        projector=projector,
        predictor=predictor,
        start_coefficients=np.zeros((19, 19)),
        end_coefficients=np.zeros((19, 19)),
        volumes=volumes,
        residual_scale=SCALE,
        state_scale=SCALE,
    )
    assert calls == [NATIVE, NATIVE]
    assert step.logical_call_count == 2
    assert step.proposal.correction_active is False
    assert np.array_equal(step.proposal.next_native_state, step.shadow_prediction)
    assert np.array_equal(step.next_native_state, step.shadow_prediction)
    assert np.count_nonzero(step.retained_displacement) == 0
    assert step.maximum_update_identity_abs == 0.0


def test_shadow_tether_rollout_inventory_and_tether_gate_fail_closed(
    monkeypatch,
) -> None:
    summary = shadow_tether_rollout.synthetic_summary()
    assert summary["status"] == "passed"
    assert all(summary["checks"].values())

    monkeypatch.setattr(
        shadow_tether_rollout.a31,
        "_gate",
        lambda **kwargs: {
            "status": "qualified",
            "checks": {"inherited_gate": True},
            "failed_checks": [],
        },
    )
    tether_rows = [
        {
            "correction_active": index < 64,
            "call_order_exact": True,
            "total_logical_call_count": 3 if index < 64 else 2,
            "status": "ok",
            "maximum_integral_difference_abs": 0.0,
            "maximum_boundary_difference_abs": 0.0,
            "maximum_projection_idempotence_abs": 0.0,
            "maximum_update_identity_abs": 0.0,
        }
        for index in range(420)
    ]
    arguments = {
        "score_rows": [],
        "case_rows": [],
        "controls": [],
        "rollouts": [],
        "audit_rows": [],
        "tether_rows": tether_rows,
        "inactive_max_abs": {},
        "shadow_raw_max_abs": {
            case_id: 0.0 for case_id in shadow_tether_rollout.CASE_IDS
        },
        "prefix_repeat_abs": 0.0,
        "structural_checks": {},
    }
    qualified = shadow_tether_rollout._gate(**arguments)
    assert qualified["status"] == "qualified"
    assert all(qualified["checks"].values())

    tether_rows[0] = {
        **tether_rows[0],
        "maximum_integral_difference_abs": np.nextafter(
            shadow_tether_rollout.STRICT_TOLERANCE, np.inf
        ),
    }
    failed = shadow_tether_rollout._gate(**arguments)
    assert failed["status"] == "failed"
    assert failed["failed_checks"] == [
        "tether_integral_boundary_projection_closure"
    ]


def test_persistent_active_modal_error_is_reduced_without_truth_input() -> None:
    volumes, projector = _grid()
    coordinates = np.zeros((8, 4), dtype=np.float64)
    coordinates[4, 0] = 0.1
    discrepancy = reconstruct_modal_field(
        coordinates,
        projector,
        component_scale=SCALE,
    )
    native_increment = np.ones_like(discrepancy)
    basis = NativeIncrementBasis(
        native_increment=native_increment,
        coarse_on_native=native_increment,
        fine_on_native=discrepancy,
        native_minus_coarse=native_increment,
        fine_minus_native=discrepancy,
    )
    correction, _ = sparse_modal_correction(
        basis,
        projector,
        policy=SPARSE_POLICY,
        volumes=volumes,
        component_scale=SCALE,
    )
    unknown_target = -0.5 * discrepancy

    assert np.linalg.norm(unknown_target - correction) < np.linalg.norm(unknown_target)


def test_driver_adapter_is_scoped_and_synthetic_contract_passes() -> None:
    original = evaluator.parent.synchronized_fine_discrepancy_step
    with evaluator.sparse_rollout_driver():
        assert evaluator.parent.synchronized_fine_discrepancy_step is not original
    assert evaluator.parent.synchronized_fine_discrepancy_step is original
    assert all(evaluator.synthetic_summary()["checks"].values())


def test_affine_transfer_inventory_and_strict_skill_gate_are_frozen() -> None:
    summary = affine_transfer.synthetic_summary()

    assert summary["status"] == "passed"
    assert all(summary["checks"].values())
    assert len(affine_transfer.CASE_IDS) == 14
    assert affine_transfer.INPUT_CALLS == tuple(range(30))
    assert len(affine_transfer._population_specs()) == 15
    assert affine_transfer._positive_skill(
        {"skill_status": "ok", "skill_vs_zero": np.nextafter(0.0, 1.0)}
    )
    assert not affine_transfer._positive_skill(
        {"skill_status": "ok", "skill_vs_zero": 0.0}
    )
    assert not affine_transfer._positive_skill(
        {"skill_status": "unresolved", "skill_vs_zero": 1.0}
    )


def test_binary_position_phase_rescore_is_exact_and_fail_closed() -> None:
    assert binary_rescore.selector_active(0.8125, 0)
    assert binary_rescore.selector_active(0.8125, 29)
    assert not binary_rescore.selector_active(np.nextafter(0.8125, 1.0), 0)
    assert not binary_rescore.selector_active(0.7, 8)
    assert not binary_rescore.selector_active(0.7, 21)
    with pytest.raises(TypeError, match="integer"):
        binary_rescore.selector_active(0.7, True)
    with pytest.raises(ValueError, match="0..29"):
        binary_rescore.selector_active(0.7, 30)

    rollout_summary = binary_rollout.synthetic_summary()
    assert rollout_summary["status"] == "passed"
    assert all(rollout_summary["checks"].values())
    assert binary_rollout.EXPECTED_ACTIVE_CASES == (
        "sv_e12_y01",
        "sv_e12_y07",
        "sv_e14_y01",
        "sv_e14_y07",
    )

    source = {
        "zero_sse": 4.0,
        "corrected_sse": 2.0,
        "correction_rms": 1.0,
        "cosine": 0.75,
        "cosine_status": "ok",
    }
    statistics = binary_rescore.sufficient_statistics(source)
    assert statistics == {
        "target_square": 4.0,
        "correction_square": 1.0,
        "cross": 1.5,
        "corrected_square": 2.0,
    }
    score = binary_rescore.score_statistics(statistics)
    assert score["skill_vs_zero"] == pytest.approx(0.5)
    assert score["rms_ratio_vs_zero"] == pytest.approx(np.sqrt(0.5))
    assert score["cosine"] == pytest.approx(0.75)
    assert score["correlation"] is None

    combined = binary_rescore.aggregate_statistics(
        [
            statistics,
            {"target_square": 1.0, "correction_square": 0.0, "cross": 0.0},
        ],
        [0.5, 0.5],
    )
    assert combined == {
        "target_square": 2.5,
        "correction_square": 0.5,
        "cross": 0.75,
    }
    with pytest.raises(ValueError, match="weights must sum"):
        binary_rescore.aggregate_statistics([statistics], [0.5])
    with pytest.raises(ValueError, match="cosine does not close"):
        binary_rescore.sufficient_statistics({**source, "cosine": 0.5})

    active = binary_rescore._selected_error(
        {"zero_error": 2.0, "corrected_error": 1.0}, active=True
    )
    inactive = binary_rescore._selected_error(
        {"zero_error": 2.0, "corrected_error": 1.0}, active=False
    )
    assert active == {"zero_error": 2.0, "corrected_error": 1.0}
    assert inactive == {"zero_error": 2.0, "corrected_error": 2.0}
    assert binary_rescore.rms_control(
        key="x", scope="case", rows=[inactive]
    )["ratio"] == 1.0
    unresolved = binary_rescore.rms_control(
        key="x",
        scope="case",
        rows=[{"zero_error": 0.0, "corrected_error": 1.0}],
    )
    assert unresolved["ratio"] is None
    assert unresolved["status"] == "small_denominator_harm"
