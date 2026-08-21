from __future__ import annotations

import numpy as np
import pytest

from scripts.time_dependent_no import (
    evaluate_pcno_scheduled_sparse_modal_correction as evaluator,
)
from utility.time_dependent_no.pcno_cross_resolution_correction import (
    ResolutionContract,
    prepare_common_native_inputs,
    prolong_nested_state,
)
from utility.time_dependent_no.pcno_cross_resolution_teacher_forced import (
    build_fixed_cosine_projector,
)
from utility.time_dependent_no.pcno_fine_discrepancy_correction import (
    modal_coordinates,
    reconstruct_modal_field,
)
from utility.time_dependent_no.pcno_scheduled_sparse_modal_correction import (
    ACTIVE_START_INPUT_CALL,
    HORIZON,
    correction_active,
    synchronized_scheduled_sparse_modal_step,
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


def test_fixed_schedule_boundary_and_invalid_calls_fail_closed() -> None:
    assert correction_active(ACTIVE_START_INPUT_CALL - 1) is False
    assert correction_active(ACTIVE_START_INPUT_CALL) is True
    assert correction_active(HORIZON - 1) is True

    for value in (True, 1.0, "15"):
        with pytest.raises(TypeError, match="integer"):
            correction_active(value)  # type: ignore[arg-type]
    for value in (-1, HORIZON):
        with pytest.raises(ValueError, match="input_call"):
            correction_active(value)


def test_early_step_makes_one_call_and_applies_exact_zero() -> None:
    volumes, projector = _grid()
    state = np.random.default_rng(1).normal(size=(NATIVE[0] * NATIVE[1], 4))
    prepared = prepare_common_native_inputs(state, contract=CONTRACT)
    calls: list[tuple[tuple[int, int], np.ndarray]] = []

    def predictor(resolution, value):
        calls.append((resolution, np.array(value, copy=True)))
        return value + 0.2

    step = synchronized_scheduled_sparse_modal_step(
        state,
        input_call=ACTIVE_START_INPUT_CALL - 1,
        contract=CONTRACT,
        projector=projector,
        predictor=predictor,
        volumes=volumes,
        component_scale=SCALE,
    )

    assert [resolution for resolution, _ in calls] == [NATIVE]
    assert np.array_equal(calls[0][1], prepared.model_inputs[NATIVE])
    assert set(step.predictions) == {NATIVE}
    assert np.array_equal(step.correction, np.zeros_like(step.correction))
    assert np.array_equal(step.next_native_state, step.predictions[NATIVE])
    assert step.audit.status == "zero"


def test_late_step_uses_two_same_state_views_and_sparse_correction() -> None:
    volumes, projector = _grid()
    state = np.random.default_rng(2).normal(size=(NATIVE[0] * NATIVE[1], 4))
    prepared = prepare_common_native_inputs(state, contract=CONTRACT)
    native_increment = np.full_like(state, 0.2)
    coordinates = np.zeros((8, 4), dtype=np.float64)
    coordinates[4, 0] = 0.1
    discrepancy = reconstruct_modal_field(
        coordinates,
        projector,
        component_scale=SCALE,
    )
    fine_increment = prolong_nested_state(
        native_increment + discrepancy,
        coarse_resolution=NATIVE,
        fine_resolution=CONTRACT.fine,
    )
    calls: list[tuple[tuple[int, int], np.ndarray]] = []

    def predictor(resolution, value):
        calls.append((resolution, np.array(value, copy=True)))
        increment = native_increment if resolution == NATIVE else fine_increment
        return value + increment

    step = synchronized_scheduled_sparse_modal_step(
        state,
        input_call=ACTIVE_START_INPUT_CALL,
        contract=CONTRACT,
        projector=projector,
        predictor=predictor,
        volumes=volumes,
        component_scale=SCALE,
    )

    assert [resolution for resolution, _ in calls] == [NATIVE, CONTRACT.fine]
    assert np.array_equal(calls[0][1], prepared.model_inputs[NATIVE])
    assert np.array_equal(calls[1][1], prepared.model_inputs[CONTRACT.fine])
    assert np.count_nonzero(step.correction) > 0
    assert np.allclose(
        step.next_native_state,
        step.predictions[NATIVE] + step.correction,
        atol=1.0e-14,
        rtol=0.0,
    )
    actual = modal_coordinates(step.correction, projector, component_scale=SCALE)
    assert np.max(np.abs(actual[0])) <= 2.0e-15
    assert step.audit.maximum_scaled_component_mean_abs <= 2.0e-16


def test_driver_resets_schedule_counts_calls_and_restores_parent() -> None:
    volumes, projector = _grid()
    original = evaluator.parent.synchronized_fine_discrepancy_step
    calls = []
    closure_rows: list[dict[str, object]] = []
    state = np.zeros((NATIVE[0] * NATIVE[1], 4), dtype=np.float64)

    def predictor(resolution, value):
        calls.append(resolution)
        return value + 0.1

    with evaluator.late_window_rollout_driver(closure_rows):
        injected = evaluator.parent.synchronized_fine_discrepancy_step
        for _ in range(HORIZON):
            step = injected(
                state,
                contract=CONTRACT,
                projector=projector,
                predictor=predictor,
                policy=evaluator.INTERNAL_POLICY,
                volumes=volumes,
                component_scale=SCALE,
            )
            state = step.next_native_state

    assert evaluator.parent.synchronized_fine_discrepancy_step is original
    assert len(calls) == HORIZON + (HORIZON - ACTIVE_START_INPUT_CALL)
    assert calls.count(NATIVE) == HORIZON
    assert calls.count(CONTRACT.fine) == HORIZON - ACTIVE_START_INPUT_CALL
    assert [int(row["input_call"]) for row in closure_rows] == list(range(HORIZON))
    assert all(
        bool(row["active"]) == (int(row["input_call"]) >= ACTIVE_START_INPUT_CALL)
        for row in closure_rows
    )
    assert all(
        float(row["correction_max_abs"]) == 0.0
        for row in closure_rows[:ACTIVE_START_INPUT_CALL]
    )

    reset_rows: list[dict[str, object]] = []
    with evaluator.late_window_rollout_driver(reset_rows):
        evaluator.parent.synchronized_fine_discrepancy_step(
            np.zeros_like(state),
            contract=CONTRACT,
            projector=projector,
            predictor=predictor,
            policy=evaluator.INTERNAL_POLICY,
            volumes=volumes,
            component_scale=SCALE,
        )
    assert reset_rows[0]["input_call"] == 0
    assert reset_rows[0]["active"] is False


def test_gate_inventory_and_synthetic_contract_fail_closed() -> None:
    names = {
        "teacher_gate_passed",
        "predecessor_identity_exact",
        "inventory_exact",
        "all_rollouts_complete_finite_admissible",
        "median_endpoint_ratio_at_most_0p98",
        "minimum_four_endpoint_wins",
        "maximum_endpoint_ratio_at_most_1p02",
        "aggregate_state_rms_ratio_at_most_0p99",
        "increment_and_cumulative_ratios_at_most_one",
        "all_controls_no_harm",
        "scheduled_common_source_closure",
        "correction_audits_pass",
        "schedule_and_mask_closure_passed",
        "deterministic_prefix_exact",
        "source_and_artifacts_exact",
    }
    gate = evaluator._recurrent_gate({name: True for name in names})
    assert gate["adaptive_recurrent_pass"] is True
    with pytest.raises(ValueError, match="inventory"):
        evaluator._recurrent_gate({name: True for name in names - {"inventory_exact"}})
    assert all(evaluator.synthetic_summary()["checks"].values())
