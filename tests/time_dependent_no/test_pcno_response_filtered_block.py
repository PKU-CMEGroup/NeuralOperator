from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.time_dependent_no import evaluate_pcno_response_filtered_block as runner
from scripts.time_dependent_no import (
    visualize_pcno_response_filtered_block as visualizer,
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
    FineDiscrepancyAudit,
)
from utility.time_dependent_no.pcno_response_filtered_block import (
    FROZEN_POSITION_BUFFER_THRESHOLD,
    FROZEN_POSITION_TRUST_THRESHOLD,
    IntegralAnchorAudit,
    ResponseProjectionAudit,
    ShadowAnchoredBlock,
    TargetFreeFrontBranchAudit,
    TransverseVelocityPositionDescriptor,
    buffered_frozen_offset_position_route,
    fixed_late_terminal_ramp_gain,
    monotone_persistence_gain,
    phase_triggered_terminal_ramp_gain,
    physical_integral_anchor,
    position_trusts_cross_resolution,
    projected_shadow_tether,
    recurrent_response_filtered_blocks,
    relaxed_projected_shadow_tether,
    slew_limited_projected_shadow_tether,
    synchronized_persistence_probe,
    synchronized_response_filtered_block,
    synchronized_shadow_anchored_block,
    target_free_front_branch_audit,
    transverse_velocity_position_descriptor,
)


def _fixture():
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
    state = np.ones((nx * ny, 4), dtype=np.float64)
    pattern = np.zeros_like(state)
    pattern[:, 0] = projector.basis[:, 1]
    fine_pattern = prolong_nested_state(
        pattern,
        coarse_resolution=(nx, ny),
        fine_resolution=(2 * nx, 2 * ny),
    )
    contract = ResolutionContract(
        coarse=(nx // 2, ny // 2),
        native=(nx, ny),
        fine=(2 * nx, 2 * ny),
    )
    return state, volumes, projector, fine_pattern, contract


def test_transverse_velocity_position_descriptor_is_target_free_and_signed():
    nodes = np.array(
        [
            [0.25, 0.35],
            [0.75, 0.35],
            [0.25, 0.50],
            [0.75, 0.50],
            [0.25, 0.65],
            [0.75, 0.65],
        ],
        dtype=np.float64,
    )
    volumes = np.full(6, 1.0 / 6.0)
    state = np.zeros((6, 4), dtype=np.float64)
    state[:, 0] = 1.0
    state[:2, 2] = (-0.2, 0.2)

    descriptor = transverse_velocity_position_descriptor(
        state,
        nodes=nodes,
        volumes=volumes,
        y_min=0.0,
        y_max=1.0,
    )

    assert descriptor.status == "ok"
    assert descriptor.vertical_centroid == pytest.approx(0.35)
    assert descriptor.normalized_wall_distance == pytest.approx(0.70)
    assert position_trusts_cross_resolution(descriptor)
    assert FROZEN_POSITION_TRUST_THRESHOLD == 0.7375

    state[:, 2] = 0.0
    unresolved = transverse_velocity_position_descriptor(
        state,
        nodes=nodes,
        volumes=volumes,
        y_min=0.0,
        y_max=1.0,
    )
    assert unresolved.status == "unresolved_transverse_velocity"
    assert unresolved.vertical_centroid is None
    assert not position_trusts_cross_resolution(unresolved)


def test_buffered_position_route_is_target_free_symmetric_and_fail_closed():
    def descriptor(distance):
        return TransverseVelocityPositionDescriptor(
            status="ok",
            vertical_centroid=0.5 * distance,
            normalized_wall_distance=distance,
            transverse_velocity_l1=1.0,
        )

    assert FROZEN_POSITION_BUFFER_THRESHOLD == 0.8125
    assert buffered_frozen_offset_position_route(descriptor(0.7375)) == (
        "edge_candidate"
    )
    assert buffered_frozen_offset_position_route(descriptor(0.775)) == (
        "raw_uncertainty_buffer"
    )
    assert buffered_frozen_offset_position_route(descriptor(0.85)) == (
        "interior_frozen_offset"
    )
    unresolved = TransverseVelocityPositionDescriptor(
        status="unresolved_transverse_velocity",
        vertical_centroid=None,
        normalized_wall_distance=None,
        transverse_velocity_l1=0.0,
    )
    assert buffered_frozen_offset_position_route(unresolved) == "raw_unresolved"
    with pytest.raises(ValueError, match="ordered"):
        buffered_frozen_offset_position_route(
            descriptor(0.8), trust_threshold=0.9, buffer_threshold=0.8
        )


def test_target_free_front_branch_audit_detects_discrete_thickness_change():
    nx, ny = 16, 2
    gamma = 1.4

    def state_from_pressure(profile):
        pressure = np.tile(np.asarray(profile, dtype=np.float64), (ny, 1))
        state = np.zeros((ny, nx, 4), dtype=np.float64)
        state[..., 0] = 1.0
        state[..., 3] = pressure / (gamma - 1.0)
        return state.reshape(-1, 4)

    raw_profile = np.r_[np.ones(4), 2.0 * np.ones(nx - 4)]
    wide_profile = raw_profile.copy()
    wide_profile[3] = 1.3
    raw = state_from_pressure(raw_profile)
    wide = state_from_pressure(wide_profile)

    audit = target_free_front_branch_audit(
        raw,
        wide,
        raw,
        raw,
        resolution=(nx, ny),
        x_min=0.0,
        x_max=2.0,
        gamma=gamma,
        shock_center_x=0.5,
    )

    assert audit.status == "branch_changed"
    assert audit.branch_changed
    assert audit.first_candidate_thickness_cells == 1
    assert audit.first_shadow_thickness_cells == 1
    assert audit.second_candidate_thickness_cells == 2
    assert audit.second_shadow_thickness_cells == 1


def test_response_filtered_block_uses_exact_four_call_order():
    state, volumes, projector, fine_pattern, contract = _fixture()
    calls = []

    def predictor(resolution, value):
        calls.append((resolution, np.array(value, copy=True)))
        if resolution == contract.fine:
            return value + 0.1 + 0.04 * fine_pattern
        return 1.1 * value + 0.1

    block = synchronized_response_filtered_block(
        state,
        contract=contract,
        projector=projector,
        predictor=predictor,
        volumes=volumes[:, None],
        residual_scale=np.ones(4),
        state_scale=np.ones(4),
    )

    assert [resolution for resolution, _ in calls] == [
        contract.native,
        contract.fine,
        contract.native,
        contract.native,
    ]
    np.testing.assert_array_equal(calls[2][1], block.raw_first_state)
    np.testing.assert_array_equal(calls[3][1], block.corrected_first_state)
    assert block.first_audit.correction_rms > 0.0


def test_filtered_response_is_mean_neutral_boundary_zero_and_idempotent():
    state, volumes, projector, fine_pattern, contract = _fixture()

    def predictor(resolution, value):
        if resolution == contract.fine:
            return value + 0.1 + 0.04 * fine_pattern
        return 1.1 * value + 0.1

    block = synchronized_response_filtered_block(
        state,
        contract=contract,
        projector=projector,
        predictor=predictor,
        volumes=volumes,
        residual_scale=np.ones(4),
        state_scale=np.ones(4),
    )
    audit = block.projection_audit

    assert audit.maximum_first_correction_integral_abs <= 1.0e-12
    assert audit.maximum_filtered_response_integral_abs <= 1.0e-12
    assert audit.maximum_first_boundary_abs <= 1.0e-12
    assert audit.maximum_filtered_boundary_abs <= 1.0e-12
    assert audit.maximum_projection_idempotence_abs <= 1.0e-12
    np.testing.assert_allclose(
        block.filtered_second_state,
        block.raw_second_state + block.filtered_response,
    )


def test_zero_cross_resolution_discrepancy_fails_closed_to_zero_correction():
    state, volumes, projector, _, contract = _fixture()

    def predictor(_resolution, value):
        return value + 0.1

    block = synchronized_response_filtered_block(
        state,
        contract=contract,
        projector=projector,
        predictor=predictor,
        volumes=volumes,
        residual_scale=np.ones(4),
        state_scale=np.ones(4),
    )
    np.testing.assert_array_equal(block.first_correction, np.zeros_like(state))
    np.testing.assert_array_equal(block.full_response, np.zeros_like(state))
    np.testing.assert_array_equal(block.filtered_response, np.zeros_like(state))
    assert block.first_audit.status == "unresolved_zero_discrepancy"


def test_persistence_probe_reuses_native_prediction_and_common_source():
    state, volumes, projector, fine_pattern, contract = _fixture()
    native_prediction = state + 0.1
    calls = []

    def fine_predictor(resolution, value):
        calls.append((resolution, np.array(value, copy=True)))
        return value + 0.1 + 0.04 * fine_pattern

    probe = synchronized_persistence_probe(
        state,
        native_prediction,
        contract=contract,
        projector=projector,
        fine_predictor=fine_predictor,
        volumes=volumes,
        component_scale=np.ones(4),
    )

    assert len(calls) == 1
    assert calls[0][0] == contract.fine
    np.testing.assert_array_equal(calls[0][1], probe.fine_model_input)
    np.testing.assert_array_equal(probe.native_model_input, state.astype(np.float32))
    assert probe.audit.status == "ok"
    assert probe.audit.correction_rms > 0.0


def test_monotone_persistence_gain_clips_decays_and_fails_closed():
    state, volumes, _, _, _ = _fixture()
    offset = np.array(state, copy=True)
    offset[:, 1:] = 0.0
    baseline = float(np.sum(volumes[:, None] * offset**2))
    initialized = monotone_persistence_gain(
        offset,
        0.4 * offset,
        baseline_alignment=None,
        previous_gain=1.0,
        volumes=volumes,
        component_scale=np.ones(4),
    )
    assert initialized.baseline_alignment == pytest.approx(0.4 * baseline)
    assert initialized.applied_gain == 1.0
    first = monotone_persistence_gain(
        offset,
        2.0 * offset,
        baseline_alignment=baseline,
        previous_gain=1.0,
        volumes=volumes,
        component_scale=np.ones(4),
    )
    assert first.status == "ok"
    assert first.clipped_gain == 1.0
    assert first.applied_gain == 1.0
    assert first.alignment_cosine == pytest.approx(1.0)

    decayed = monotone_persistence_gain(
        offset,
        0.25 * offset,
        baseline_alignment=baseline,
        previous_gain=0.8,
        volumes=volumes,
        component_scale=np.ones(4),
    )
    assert decayed.applied_gain == pytest.approx(0.25)
    rebound = monotone_persistence_gain(
        offset,
        0.7 * offset,
        baseline_alignment=baseline,
        previous_gain=decayed.applied_gain,
        volumes=volumes,
        component_scale=np.ones(4),
    )
    assert rebound.clipped_gain == pytest.approx(0.7)
    assert rebound.applied_gain == pytest.approx(0.25)

    anti_aligned = monotone_persistence_gain(
        offset,
        -offset,
        baseline_alignment=baseline,
        previous_gain=1.0,
        volumes=volumes,
        component_scale=np.ones(4),
    )
    assert anti_aligned.applied_gain == 0.0
    unresolved = monotone_persistence_gain(
        offset,
        offset,
        baseline_alignment=0.0,
        previous_gain=1.0,
        volumes=volumes,
        component_scale=np.ones(4),
    )
    assert unresolved.status == "unresolved_nonpositive_baseline_alignment"
    assert unresolved.alignment_ratio is None
    assert unresolved.applied_gain == 0.0


def test_weighted_projection_coefficient_recovers_signed_scale_and_denominator():
    direction = np.arange(1.0, 9.0, dtype=np.float64).reshape(2, 4)
    volumes = np.array([0.25, 0.75], dtype=np.float64)
    scale = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    relation = runner._weighted_scaled_projection_coefficient(
        direction,
        -0.3 * direction,
        volumes=volumes,
        component_scale=scale,
    )
    assert relation["status"] == "ok"
    assert relation["value"] == pytest.approx(-0.3)
    assert relation["denominator"] > 0.0

    unresolved = runner._weighted_scaled_projection_coefficient(
        np.zeros_like(direction),
        direction,
        volumes=volumes,
        component_scale=scale,
    )
    assert unresolved["status"] == "unresolved_small_denominator"
    assert unresolved["value"] is None


def test_phase_triggered_terminal_ramp_is_unit_then_linear_to_zero():
    before = phase_triggered_terminal_ramp_gain(
        block_index=8,
        terminal_block=15,
        alignment_cosine=0.1,
        trigger_block=None,
    )
    assert before.status == "pretrigger_unit_gain"
    assert before.applied_gain == 1.0
    triggered = phase_triggered_terminal_ramp_gain(
        block_index=9,
        terminal_block=15,
        alignment_cosine=-0.01,
        trigger_block=None,
    )
    assert triggered.trigger_block == 9
    assert triggered.applied_gain == 1.0
    middle = phase_triggered_terminal_ramp_gain(
        block_index=12,
        terminal_block=15,
        alignment_cosine=0.2,
        trigger_block=triggered.trigger_block,
    )
    assert middle.applied_gain == pytest.approx(0.5)
    terminal = phase_triggered_terminal_ramp_gain(
        block_index=15,
        terminal_block=15,
        alignment_cosine=0.2,
        trigger_block=9,
    )
    assert terminal.applied_gain == 0.0
    forced_terminal = phase_triggered_terminal_ramp_gain(
        block_index=15,
        terminal_block=15,
        alignment_cosine=0.2,
        trigger_block=None,
    )
    assert forced_terminal.status == "forced_terminal_raw"
    assert forced_terminal.trigger_block == 15
    assert forced_terminal.applied_gain == 0.0
    unresolved = phase_triggered_terminal_ramp_gain(
        block_index=7,
        terminal_block=15,
        alignment_cosine=None,
        trigger_block=None,
    )
    assert unresolved.trigger_block == 7

    with pytest.raises(ValueError, match="ordered"):
        phase_triggered_terminal_ramp_gain(
            block_index=15,
            terminal_block=14,
            alignment_cosine=0.0,
            trigger_block=None,
        )


def test_fixed_late_terminal_ramp_is_exact_and_validated():
    gains = [
        fixed_late_terminal_ramp_gain(
            block_index=block,
            ramp_start_block=9,
            terminal_block=14,
        )
        for block in range(4, 15)
    ]
    assert gains == [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
    with pytest.raises(ValueError, match="ordered"):
        fixed_late_terminal_ramp_gain(
            block_index=4,
            ramp_start_block=14,
            terminal_block=14,
        )


def test_invalid_volume_shape_is_rejected():
    state, volumes, projector, fine_pattern, contract = _fixture()

    def predictor(resolution, value):
        if resolution == contract.fine:
            return value + 0.1 + 0.04 * fine_pattern
        return 1.1 * value + 0.1

    with pytest.raises(ValueError, match="volumes"):
        synchronized_response_filtered_block(
            state,
            contract=contract,
            projector=projector,
            predictor=predictor,
            volumes=volumes[:-1],
            residual_scale=np.ones(4),
            state_scale=np.ones(4),
        )


def test_synthetic_entry_point_passes_and_sealed_followups_are_absent():
    payload = runner.synthetic_summary()
    assert payload["status"] == "passed"
    assert all(payload["checks"].values())
    with pytest.raises(SystemExit):
        runner.parse_args(["strength-ood"])


def test_state_summary_uses_case_first_weighting():
    records = []
    for case_index, case_id in enumerate(runner.CALIBRATION_CASE_IDS):
        for input_call in runner.INPUT_CALLS:
            raw = 1.0 + 0.1 * case_index
            filtered = 0.81 * raw
            records.append(
                {
                    "case_id": case_id,
                    "raw_first_state_energy": raw,
                    "corrected_first_state_energy": 0.9025 * raw,
                    "raw_second_state_energy": raw,
                    "full_second_state_energy": 0.9025 * raw,
                    "filtered_second_state_energy": filtered,
                }
            )
    summary = runner._state_summary(
        records,
        case_ids=runner.CALIBRATION_CASE_IDS,
        groups=runner.CALIBRATION_GROUPS,
    )
    assert summary["one_step_sp19_rms_ratio"] == pytest.approx(0.95)
    assert summary["response_filtered_second_rms_ratio"] == pytest.approx(0.9)
    assert summary["case_win_count"] == len(runner.CALIBRATION_CASE_IDS)
    assert summary["group_win_count"] == len(runner.CALIBRATION_GROUPS)


def test_state_summary_supports_disjoint_evaluation_population():
    records = []
    for case_id in runner.EVALUATION_CASE_IDS:
        for input_call in runner.INPUT_CALLS:
            records.append(
                {
                    "case_id": case_id,
                    "raw_first_state_energy": 1.0,
                    "corrected_first_state_energy": 0.9801,
                    "raw_second_state_energy": 1.0,
                    "full_second_state_energy": 0.9801,
                    "filtered_second_state_energy": 0.990025,
                }
            )
    summary = runner._state_summary(
        records,
        case_ids=runner.EVALUATION_CASE_IDS,
        groups=runner.EVALUATION_GROUPS,
    )
    assert summary["response_filtered_second_rms_ratio"] == pytest.approx(0.995)
    assert summary["case_win_count"] == len(runner.EVALUATION_CASE_IDS)
    assert summary["group_win_count"] == len(runner.EVALUATION_GROUPS)


def test_control_ratio_treats_only_zero_to_zero_as_exact_no_change():
    assert runner._control_ratio(0.0, 0.0) == {
        "ratio": 1.0,
        "status": "exact_zero_no_change",
    }
    assert runner._control_ratio(1.0e-15, 1.0e-15) == {
        "ratio": 1.0,
        "status": "exact_zero_no_change",
    }
    assert runner._control_ratio(0.0, 1.0e-6) == {
        "ratio": None,
        "status": "nonzero_from_zero",
    }
    assert runner._control_ratio(2.0, 1.0) == {
        "ratio": 0.5,
        "status": "ok",
    }


def test_control_gate_accepts_exact_no_change_but_rejects_nonzero_from_zero():
    rows = [
        {
            "kind": "front",
            "ratio": 1.0,
            "status": "exact_zero_no_change",
        },
        {"kind": "front", "ratio": 1.01, "status": "ok"},
    ]
    assert runner._controls_pass(rows, kind="front", limit=1.05)
    rows[0] = {
        "kind": "front",
        "ratio": None,
        "status": "nonzero_from_zero",
    }
    assert not runner._controls_pass(rows, kind="front", limit=1.05)


def test_typed_rollout_rows_converts_numeric_boolean_and_blank_fields():
    rows = runner._typed_rollout_rows(
        [
            {
                "case_id": "sv_e00_y00",
                "policy": "zero",
                "input_call": "2",
                "output_call": "3",
                "finite": "True",
                "admissible": "False",
                "cap_active": "False",
                "correction_status": "raw_shadow",
                "state_error": "0.125",
                "constant_mode_energy_fraction": "",
            }
        ]
    )
    assert rows == [
        {
            "case_id": "sv_e00_y00",
            "policy": "zero",
            "input_call": 2,
            "output_call": 3,
            "finite": True,
            "admissible": False,
            "cap_active": False,
            "correction_status": "raw_shadow",
            "state_error": 0.125,
            "constant_mode_energy_fraction": None,
        }
    ]


def test_recurrent_blocks_emit_two_states_and_recur_only_from_selected_second():
    state, volumes, projector, fine_pattern, contract = _fixture()
    predictor_calls = []
    builder_inputs = []

    def predictor(resolution, value):
        predictor_calls.append(resolution)
        if resolution == contract.fine:
            return value + 0.1 + 0.04 * fine_pattern
        return 1.1 * value + 0.1

    def builder(value):
        builder_inputs.append(np.array(value, copy=True))
        return synchronized_response_filtered_block(
            value,
            contract=contract,
            projector=projector,
            predictor=predictor,
            volumes=volumes,
            residual_scale=np.ones(4),
            state_scale=np.ones(4),
        )

    trajectory = recurrent_response_filtered_blocks(
        state,
        block_count=3,
        block_builder=builder,
        response_mode="filtered",
    )
    assert len(trajectory.states) == 7
    assert len(trajectory.blocks) == 3
    assert len(predictor_calls) == 12
    np.testing.assert_array_equal(builder_inputs[0], state)
    for index in range(1, 3):
        np.testing.assert_array_equal(
            builder_inputs[index], trajectory.states[2 * index]
        )


def test_recurrent_block_mode_selects_full_or_filtered_second_state():
    state, volumes, projector, fine_pattern, contract = _fixture()

    def build(value):
        def predictor(resolution, current):
            if resolution == contract.fine:
                return current + 0.1 + 0.04 * fine_pattern
            return 1.1 * current + 0.1

        return synchronized_response_filtered_block(
            value,
            contract=contract,
            projector=projector,
            predictor=predictor,
            volumes=volumes,
            residual_scale=np.ones(4),
            state_scale=np.ones(4),
        )

    full = recurrent_response_filtered_blocks(
        state, block_count=1, block_builder=build, response_mode="full"
    )
    filtered = recurrent_response_filtered_blocks(
        state, block_count=1, block_builder=build, response_mode="filtered"
    )
    np.testing.assert_array_equal(
        full.states[-1], full.blocks[0].fully_corrected_second_state
    )
    np.testing.assert_array_equal(
        filtered.states[-1], filtered.blocks[0].filtered_second_state
    )
    with pytest.raises(ValueError, match="response_mode"):
        recurrent_response_filtered_blocks(
            state, block_count=1, block_builder=build, response_mode="bad"
        )


def test_physical_integral_anchor_is_exact_boundary_zero_and_idempotent():
    state, volumes, projector, _, _ = _fixture()
    candidate = state.copy()
    candidate[:, 0] += np.linspace(-0.2, 0.3, state.shape[0])
    shadow = state.copy()
    shadow[:, 0] += 0.05
    anchored, correction, audit = physical_integral_anchor(
        candidate,
        shadow,
        volumes=volumes[:, None],
        interior_mask=projector.interior_mask,
        component_scale=np.ones(4),
    )
    np.testing.assert_allclose(
        np.sum(volumes[:, None] * anchored, axis=0),
        np.sum(volumes[:, None] * shadow, axis=0),
        atol=1.0e-14,
        rtol=0.0,
    )
    np.testing.assert_array_equal(
        correction[~projector.interior_mask],
        np.zeros_like(correction[~projector.interior_mask]),
    )
    anchored_twice, second_correction, second_audit = physical_integral_anchor(
        anchored,
        shadow,
        volumes=volumes,
        interior_mask=projector.interior_mask,
        component_scale=np.ones(4),
    )
    np.testing.assert_allclose(anchored_twice, anchored, atol=1.0e-14, rtol=0.0)
    np.testing.assert_allclose(second_correction, 0.0, atol=1.0e-14, rtol=0.0)
    assert audit.maximum_boundary_correction_abs == 0.0
    assert second_audit.maximum_integral_mismatch_after_abs <= 1.0e-14


def test_projected_shadow_tether_is_nonconstant_boundary_zero_and_idempotent():
    state, volumes, projector, _, _ = _fixture()
    shadow = state.copy()
    candidate = state.copy()
    candidate[:, 0] += 0.1 + 0.03 * projector.basis[:, 1]
    candidate[:, 2] += 0.02 * projector.basis[:, 7]

    tethered, retained, audit = projected_shadow_tether(
        candidate,
        shadow,
        projector=projector,
        volumes=volumes[:, None],
        component_scale=np.ones(4),
    )

    np.testing.assert_allclose(tethered - shadow, retained, atol=1.0e-14, rtol=0.0)
    np.testing.assert_allclose(
        np.sum(volumes[:, None] * retained, axis=0),
        0.0,
        atol=1.0e-14,
        rtol=0.0,
    )
    np.testing.assert_array_equal(
        retained[~projector.interior_mask],
        np.zeros_like(retained[~projector.interior_mask]),
    )
    tethered_twice, retained_twice, second_audit = projected_shadow_tether(
        tethered,
        shadow,
        projector=projector,
        volumes=volumes,
        component_scale=np.ones(4),
    )
    np.testing.assert_allclose(tethered_twice, tethered, atol=1.0e-14, rtol=0.0)
    np.testing.assert_allclose(retained_twice, retained, atol=1.0e-14, rtol=0.0)
    assert audit.maximum_boundary_difference_abs == 0.0
    assert audit.maximum_projection_idempotence_abs <= 1.0e-14
    assert second_audit.maximum_integral_difference_abs <= 1.0e-14


def test_slew_limited_tether_bounds_increment_change_and_preserves_closure():
    state, volumes, projector, _, _ = _fixture()
    previous_shadow = state.copy()
    previous_candidate = state.copy()
    previous_candidate[:, 0] += 0.02 * projector.basis[:, 1]
    previous_accepted, _, _ = projected_shadow_tether(
        previous_candidate,
        previous_shadow,
        projector=projector,
        volumes=volumes,
        component_scale=np.ones(4),
    )
    shadow = previous_shadow.copy()
    shadow[:, 0] += 0.01
    candidate = shadow.copy()
    candidate[:, 0] += 0.20 * projector.basis[:, 1]

    accepted, retained, audit = slew_limited_projected_shadow_tether(
        candidate,
        shadow,
        previous_accepted,
        previous_shadow,
        projector=projector,
        volumes=volumes,
        component_scale=np.ones(4),
        relative_change_limit=0.10,
    )

    previous_difference = previous_accepted - previous_shadow
    increment_change = (accepted - previous_accepted) - (
        shadow - previous_shadow
    )
    np.testing.assert_allclose(
        increment_change,
        retained - previous_difference,
        atol=1.0e-14,
        rtol=0.0,
    )
    assert audit.cap_active
    assert audit.applied_to_shadow_increment_ratio == pytest.approx(0.10)
    assert audit.applied_change_rms <= audit.change_limit_rms + 1.0e-14
    np.testing.assert_allclose(
        np.sum(volumes[:, None] * retained, axis=0),
        0.0,
        atol=1.0e-14,
        rtol=0.0,
    )
    np.testing.assert_array_equal(
        retained[~projector.interior_mask],
        np.zeros_like(retained[~projector.interior_mask]),
    )
    assert audit.maximum_projection_idempotence_abs <= 1.0e-14
    assert audit.maximum_update_identity_abs == 0.0


def test_slew_limited_tether_zero_change_is_exact_and_uncapped():
    state, volumes, projector, _, _ = _fixture()
    previous_shadow = state.copy()
    previous_candidate = state.copy()
    previous_candidate[:, 2] += 0.03 * projector.basis[:, 7]
    previous_accepted, _, _ = projected_shadow_tether(
        previous_candidate,
        previous_shadow,
        projector=projector,
        volumes=volumes,
        component_scale=np.ones(4),
    )
    shadow = state.copy()
    shadow[:, 0] += 0.02
    candidate = shadow + (previous_accepted - previous_shadow)

    accepted, retained, audit = slew_limited_projected_shadow_tether(
        candidate,
        shadow,
        previous_accepted,
        previous_shadow,
        projector=projector,
        volumes=volumes,
        component_scale=np.ones(4),
        relative_change_limit=0.10,
    )

    np.testing.assert_allclose(accepted, candidate, atol=1.0e-14, rtol=0.0)
    np.testing.assert_allclose(
        retained,
        previous_accepted - previous_shadow,
        atol=1.0e-14,
        rtol=0.0,
    )
    assert not audit.cap_active
    assert audit.requested_change_rms <= 1.0e-14
    assert audit.applied_change_rms <= 1.0e-14


def test_slew_limited_tether_rejects_invalid_limit():
    state, volumes, projector, _, _ = _fixture()
    with pytest.raises(ValueError, match="relative_change_limit"):
        slew_limited_projected_shadow_tether(
            state,
            state,
            state,
            state,
            projector=projector,
            volumes=volumes,
            component_scale=np.ones(4),
            relative_change_limit=1.01,
        )


def test_slew_limited_tether_safely_transitions_unprojected_handoff():
    state, volumes, projector, _, _ = _fixture()
    unprojected = state.copy()
    unprojected[:, 0] += 0.02 + 0.01 * projector.basis[:, 1]
    shadow = state.copy()
    shadow[:, 0] += 0.01
    candidate = shadow.copy()
    candidate[:, 0] += 0.04 * projector.basis[:, 1]
    accepted, retained, audit = slew_limited_projected_shadow_tether(
        candidate,
        shadow,
        unprojected,
        state,
        projector=projector,
        volumes=volumes,
        component_scale=np.ones(4),
        relative_change_limit=0.10,
    )
    increment_change = (accepted - unprojected) - (shadow - state)
    np.testing.assert_allclose(
        increment_change,
        retained - (unprojected - state),
        atol=1.0e-14,
        rtol=0.0,
    )
    assert audit.previous_projection_residual_rms > 0.0
    assert audit.retained_projection_residual_rms < (
        audit.previous_projection_residual_rms
    )
    assert audit.maximum_boundary_contraction_violation_abs <= 1.0e-14
    assert audit.applied_to_shadow_increment_ratio == pytest.approx(0.10)


def test_relaxed_tether_applies_fixed_target_gap_fraction_and_closure():
    state, volumes, projector, _, _ = _fixture()
    previous_shadow = state.copy()
    previous_accepted = state.copy()
    previous_accepted[:, 0] += 0.02 + 0.01 * projector.basis[:, 1]
    shadow = state.copy()
    shadow[:, 0] += 0.01
    candidate = shadow.copy()
    candidate[:, 0] += 0.05 * projector.basis[:, 1]

    accepted, retained, audit = relaxed_projected_shadow_tether(
        candidate,
        shadow,
        previous_accepted,
        previous_shadow,
        projector=projector,
        volumes=volumes,
        component_scale=np.ones(4),
        relaxation=0.10,
    )

    previous_difference = previous_accepted - previous_shadow
    _, target, _ = projected_shadow_tether(
        candidate,
        shadow,
        projector=projector,
        volumes=volumes,
        component_scale=np.ones(4),
    )
    np.testing.assert_allclose(
        retained,
        previous_difference + 0.10 * (target - previous_difference),
        atol=1.0e-14,
        rtol=0.0,
    )
    np.testing.assert_allclose(accepted, shadow + retained, atol=1.0e-14, rtol=0.0)
    assert audit.applied_scale == 0.10
    assert audit.cap_active
    assert audit.applied_change_rms == pytest.approx(
        0.10 * audit.requested_change_rms
    )
    assert audit.retained_projection_residual_rms < (
        audit.previous_projection_residual_rms
    )
    assert audit.maximum_boundary_contraction_violation_abs <= 1.0e-14
    assert audit.maximum_update_identity_abs == 0.0


def test_relaxed_tether_zero_relaxation_freezes_displacement():
    state, volumes, projector, _, _ = _fixture()
    previous_accepted = state.copy()
    previous_accepted[:, 0] += 0.02 * projector.basis[:, 1]
    shadow = state.copy()
    shadow[:, 0] += 0.01
    candidate = shadow.copy()
    candidate[:, 0] += 0.05 * projector.basis[:, 1]
    accepted, retained, audit = relaxed_projected_shadow_tether(
        candidate,
        shadow,
        previous_accepted,
        state,
        projector=projector,
        volumes=volumes,
        component_scale=np.ones(4),
        relaxation=0.0,
    )
    np.testing.assert_allclose(
        retained, previous_accepted - state, atol=1.0e-14, rtol=0.0
    )
    np.testing.assert_allclose(accepted, shadow + retained, atol=1.0e-14, rtol=0.0)
    assert audit.applied_change_rms == 0.0


def test_relaxed_tether_rejects_invalid_relaxation():
    state, volumes, projector, _, _ = _fixture()
    with pytest.raises(ValueError, match="relaxation"):
        relaxed_projected_shadow_tether(
            state,
            state,
            state,
            state,
            projector=projector,
            volumes=volumes,
            component_scale=np.ones(4),
            relaxation=-0.01,
        )


def test_shadow_anchored_block_uses_six_calls_and_matches_shadow_integrals():
    state, volumes, projector, fine_pattern, contract = _fixture()
    accepted = state.copy()
    accepted[:, 0] += 0.02 * projector.basis[:, 2]
    shadow = state.copy()
    calls = []

    def predictor(resolution, value):
        calls.append((resolution, np.array(value, copy=True)))
        if resolution == contract.fine:
            return value + 0.1 + 0.04 * fine_pattern
        return 1.1 * value + 0.1

    block = synchronized_shadow_anchored_block(
        accepted,
        shadow,
        contract=contract,
        projector=projector,
        predictor=predictor,
        volumes=volumes,
        residual_scale=np.ones(4),
        state_scale=np.ones(4),
    )
    assert [resolution for resolution, _ in calls] == [
        contract.native,
        contract.native,
        contract.native,
        contract.fine,
        contract.native,
        contract.native,
    ]
    np.testing.assert_array_equal(calls[0][1], shadow)
    np.testing.assert_array_equal(calls[1][1], block.shadow_first_state)
    prepared = prepare_common_native_inputs(accepted, contract=contract)
    np.testing.assert_array_equal(calls[2][1], prepared.model_inputs[contract.native])
    np.testing.assert_array_equal(calls[3][1], prepared.model_inputs[contract.fine])
    np.testing.assert_array_equal(calls[4][1], block.candidate_raw_first_state)
    np.testing.assert_array_equal(calls[5][1], block.anchored_first_state)
    for candidate, target in (
        (block.anchored_first_state, block.shadow_first_state),
        (block.anchored_second_state, block.shadow_second_state),
    ):
        np.testing.assert_allclose(
            np.sum(volumes[:, None] * candidate, axis=0),
            np.sum(volumes[:, None] * target, axis=0),
            atol=1.0e-14,
            rtol=0.0,
        )
    assert block.first_anchor_audit.maximum_boundary_correction_abs == 0.0
    assert block.second_anchor_audit.maximum_boundary_correction_abs == 0.0


def test_explicit_shadow_call_checks_accept_registered_fp32_floor_and_fail_above():
    execution_rows = []
    closure_rows = []
    for case_id in runner.EVALUATION_CASE_IDS:
        execution_rows.append(
            {
                "case_id": case_id,
                "logical_model_calls": 90,
                "native_logical_calls": 75,
                "fine_logical_calls": 15,
                "shadow_native_logical_calls": 30,
                "candidate_native_logical_calls": 45,
                "candidate_fine_logical_calls": 15,
                "completed_calls": 30,
            }
        )
        closure_rows.append(
            {
                "case_id": case_id,
                "call_order_errors": 0,
                "shadow_recurrence_abs": 0.0,
                "accepted_recurrence_abs": 2.4e-7,
                "candidate_common_native_input_abs": 0.0,
                "candidate_lookahead_input_abs": 1.0e-19,
            }
        )

    checks = runner._explicit_shadow_call_checks(execution_rows, closure_rows)
    assert all(checks.values())

    closure_rows[0]["accepted_recurrence_abs"] = 1.1e-6
    checks = runner._explicit_shadow_call_checks(execution_rows, closure_rows)
    assert not checks["accepted_post_fp32_floor_at_most_1e_6"]
    assert all(
        value
        for key, value in checks.items()
        if key != "accepted_post_fp32_floor_at_most_1e_6"
    )


def test_strength_ood_inventory_is_one_complete_group_and_call_checks_are_generic():
    assert runner.STRENGTH_OOD_GROUP_ID == "strength_ood_e13"
    assert runner.STRENGTH_OOD_CASE_IDS == tuple(
        f"sv_e13_y{index:02d}" for index in range(9)
    )
    assert set(runner.STRENGTH_OOD_ANIMATION_CASE_IDS) <= set(
        runner.STRENGTH_OOD_CASE_IDS
    )
    assert all(
        "e12" not in case_id and "e14" not in case_id
        for case_id in runner.STRENGTH_OOD_CASE_IDS
    )

    case_ids = ("case_a", "case_b")
    execution_rows = [
        {
            "case_id": case_id,
            "logical_model_calls": 90,
            "native_logical_calls": 75,
            "fine_logical_calls": 15,
            "shadow_native_logical_calls": 30,
            "candidate_native_logical_calls": 45,
            "candidate_fine_logical_calls": 15,
            "completed_calls": 30,
        }
        for case_id in case_ids
    ]
    closure_rows = [
        {
            "case_id": case_id,
            "call_order_errors": 0,
            "shadow_recurrence_abs": 0.0,
            "accepted_recurrence_abs": 2.4e-7,
            "candidate_common_native_input_abs": 0.0,
            "candidate_lookahead_input_abs": 0.0,
        }
        for case_id in case_ids
    ]
    assert all(
        runner._explicit_shadow_call_checks(
            execution_rows, closure_rows, case_ids=case_ids
        ).values()
    )
    metric_keys = (
        "state_error",
        "increment_defect",
        "cumulative_defect",
        "rank8_state_error",
        "boundary_state_error",
        "shock_state_error",
        "vortex_state_error",
        "smooth_state_error",
        "front_position",
        "front_strength_log_ratio",
        "front_thickness_log_ratio",
        *(f"component_{name}_state_error" for name in runner.INTEGRAL_COMPONENT_NAMES),
        *(f"integral_{name}_error" for name in runner.INTEGRAL_COMPONENT_NAMES),
    )
    paired_rows = []
    for case_id in case_ids:
        for input_call in range(30):
            for policy, value in (("zero", 1.0), ("candidate", 0.9)):
                paired_rows.append(
                    {
                        "case_id": case_id,
                        "input_call": input_call,
                        "policy": policy,
                        **{key: value for key in metric_keys},
                    }
                )
    case_rows, controls, population = runner._paired_rollout_controls_for_cases(
        paired_rows, case_ids=case_ids
    )
    assert len(case_rows) == len(case_ids)
    assert controls
    assert population["endpoint_win_count"] == len(case_ids)
    with pytest.raises(ValueError, match="nonempty and unique"):
        runner._explicit_shadow_call_checks([], [], case_ids=("case_a", "case_a"))


def test_prospective_structural_population_is_exactly_e12_and_keeps_e14_sealed():
    assert runner.PROSPECTIVE_STRUCTURAL_GROUP_ID == "strength_ood_e12"
    assert runner.PROSPECTIVE_STRUCTURAL_CASE_IDS == tuple(
        f"sv_e12_y{index:02d}" for index in range(9)
    )
    assert runner.PROSPECTIVE_STRUCTURAL_ANIMATION_CASE_IDS == (
        "sv_e12_y00",
        "sv_e12_y04",
        "sv_e12_y08",
    )
    assert set(runner.PROSPECTIVE_STRUCTURAL_CASE_IDS).isdisjoint(
        runner.STRENGTH_OOD_CASE_IDS
    )
    assert all(
        "e14" not in case_id for case_id in runner.PROSPECTIVE_STRUCTURAL_CASE_IDS
    )
    assert runner.PROSPECTIVE_STRUCTURAL_ANIMATION_CONTRACT["case_ids"] == list(
        runner.PROSPECTIVE_STRUCTURAL_ANIMATION_CASE_IDS
    )
    assert runner.EXPECTED_P6_A7_ROLLOUT_SHA256 == (
        "c6dade89b5966281f0d771f543c331f7aec76840223a87ebf6efd5f6432e072b"
    )


def test_prospective_buffered_offset_population_is_exactly_e14():
    assert runner.PROSPECTIVE_BUFFERED_OFFSET_GROUP_ID == "strength_ood_e14"
    assert runner.PROSPECTIVE_BUFFERED_OFFSET_CASE_IDS == tuple(
        f"sv_e14_y{index:02d}" for index in range(9)
    )
    assert runner.PROSPECTIVE_BUFFERED_OFFSET_ANIMATION_CASE_IDS == (
        "sv_e14_y00",
        "sv_e14_y01",
        "sv_e14_y04",
        "sv_e14_y08",
    )
    assert set(runner.PROSPECTIVE_BUFFERED_OFFSET_CASE_IDS).isdisjoint(
        runner.PROSPECTIVE_STRUCTURAL_CASE_IDS
    )
    assert runner.PROSPECTIVE_BUFFERED_OFFSET_ANIMATION_CONTRACT["case_ids"] == list(
        runner.PROSPECTIVE_BUFFERED_OFFSET_ANIMATION_CASE_IDS
    )
    assert runner.EXPECTED_P6_A13_ROLLOUT_SHA256 == (
        "ee2e7f7c7497cacf281064bfcf79a92bebda817932125d6ef52a96d1d9580926"
    )


def test_shard_native_reference_check_requires_bound_audit():
    truth = {
        "mode": "checkpoint_bound_shard_states_float32",
        "recovery_audit": {"path_sha256": "audit", "payload_sha256": "payload"},
    }
    check = {
        "reference_schema": runner.SHARD_NATIVE_REFERENCE_SCHEMA,
        "reference_source": "checkpoint_bound_shard_states_conservative",
        "retained_resolution": "250x100",
        "state_dtype": "float32",
        "state_shape": [61, 25000, 4],
        "restriction_crosscheck_max_abs": 0.0,
        "serialization_floor": "exact_original_reference_after_float32_cast",
        "shard_native_reference_audit_sha256": "audit",
        "shard_native_reference_audit_payload_sha256": "payload",
    }
    assert runner._native_reference_check_exact(check, truth_contract=truth)
    check["shard_native_reference_audit_payload_sha256"] = "tampered"
    assert not runner._native_reference_check_exact(check, truth_contract=truth)


def test_shard_native_reference_loader_fails_on_array_tamper(tmp_path):
    folder = tmp_path / "traj_case"
    folder.mkdir()
    arrays = {
        "states_conservative": np.ones((61, 25000, 4), dtype=np.float32),
        "physical_times": np.arange(61, dtype=np.float64) * 0.01,
        "nodes": np.zeros((25000, 2), dtype=np.float32),
        "node_measures": np.ones(25000, dtype=np.float32),
    }
    arrays["physical_times"][35] = 0.35
    hashes = {}
    for name, value in arrays.items():
        path = folder / f"{name}.npy"
        np.save(path, value)
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    entry = {
        "folder": folder.name,
        "array_sha256": hashes,
        "state_digest": runner._sha256_array_payload(arrays["states_conservative"]),
        "source_reference_sha256": "reference",
    }

    class Store:
        root = tmp_path
        manifest_digest = "manifest"

        @staticmethod
        def entry(_case_id):
            return entry

        @staticmethod
        def array(_case_id, name):
            return arrays[name]

    geometry = SimpleNamespace(
        nodes=arrays["nodes"], node_measures=arrays["node_measures"]
    )
    runtime = SimpleNamespace(
        store=Store(), geometry_by_resolution={runner.NATIVE_RESOLUTION: geometry}
    )
    reference, check = runner._load_shard_native_reference(runtime, "case")
    assert reference["conservative_states"].dtype == np.float64
    assert check["state_dtype"] == "float32"

    (folder / "states_conservative.npy").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="array digest mismatch"):
        runner._load_shard_native_reference(runtime, "case")


def test_shard_native_audit_verifier_fails_closed_on_source_drift(
    tmp_path, monkeypatch
):
    family = tmp_path / "family"
    data = tmp_path / "data"
    family.mkdir()
    data.mkdir()
    (family / "family_manifest.json").write_text("family", encoding="utf-8")
    (data / "manifest.json").write_text("data", encoding="utf-8")
    payload = runner.with_payload_sha256(
        {
            "schema": runner.SHARD_NATIVE_REFERENCE_AUDIT_SCHEMA,
            "working_id": runner.SHARD_NATIVE_REFERENCE_AUDIT_WORKING_ID,
            "status": "passed",
            "case_ids": list(runner.PROSPECTIVE_STRUCTURAL_CASE_IDS),
            "source_sha256": {"source": "hash"},
            "family_manifest_sha256": hashlib.sha256(b"family").hexdigest(),
            "shard_manifest_sha256": hashlib.sha256(b"data").hexdigest(),
            "checks": {"all": True},
        }
    )
    path = tmp_path / "audit.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(runner, "_source_hashes", lambda: {"source": "hash"})
    args = SimpleNamespace(family_root=family, data_dir=data)
    assert runner._verify_shard_native_reference_audit(path, args) == payload
    monkeypatch.setattr(runner, "_source_hashes", lambda: {"source": "drift"})
    with pytest.raises(ValueError, match="differs from contract"):
        runner._verify_shard_native_reference_audit(path, args)


def test_prospective_buffered_preflight_fails_closed_on_refit(monkeypatch):
    source_hashes = {"source.py": "source-hash"}
    payload = runner.with_payload_sha256(
        {
            "schema": runner.PROSPECTIVE_BUFFERED_OFFSET_PREFLIGHT_SCHEMA,
            "working_id": runner.PROSPECTIVE_BUFFERED_OFFSET_WORKING_ID,
            "status": "passed",
            "source_manifest": {
                "source_sha256": source_hashes,
                "qualified_a13_rollout_sha256": "a13-hash",
                "population": {
                    "case_ids": list(runner.PROSPECTIVE_BUFFERED_OFFSET_CASE_IDS),
                },
                "protocol": {"frozen_from": runner.BUFFERED_OFFSET_WORKING_ID},
            },
            "reference_arrays_loaded": False,
            "recurrence_executed": False,
            "new_sealed_population_opened": (
                runner.PROSPECTIVE_BUFFERED_OFFSET_GROUP_ID
            ),
            "still_sealed_groups": [],
            "all_cases_in_named_e14_group": True,
            "warm_start_blocks": runner.FROZEN_OFFSET_WARM_BLOCKS,
            "frozen_offset": True,
            "buffered_offset": True,
            "coast_offset_feedback": False,
            "position_trust_threshold": FROZEN_POSITION_TRUST_THRESHOLD,
            "position_buffer_threshold": FROZEN_POSITION_BUFFER_THRESHOLD,
            "coast_projection_active_cells": [
                list(cell) for cell in runner.FROZEN_ACTIVE_CELLS
            ],
            "coefficient_refit": False,
            "additional_e12_selection": False,
        }
    )
    monkeypatch.setattr(runner, "_read_json", lambda _path: payload)
    monkeypatch.setattr(runner, "_source_hashes", lambda: source_hashes)
    monkeypatch.setattr(runner, "sha256_file", lambda _path: "a13-hash")
    args = SimpleNamespace(a13_rollout="a13.json")

    assert runner._verify_prospective_buffered_offset_preflight("preflight.json", args)
    payload["additional_e12_selection"] = True
    payload["payload_sha256"] = runner.with_payload_sha256(
        {key: value for key, value in payload.items() if key != "payload_sha256"}
    )["payload_sha256"]
    with pytest.raises(ValueError, match="differs"):
        runner._verify_prospective_buffered_offset_preflight("preflight.json", args)


def test_prospective_buffered_preflight_accepts_established_provenance_schema(
    tmp_path, monkeypatch
):
    source = {
        "population": {"case_ids": list(runner.PROSPECTIVE_BUFFERED_OFFSET_CASE_IDS)}
    }
    provenance = {
        case_id: {
            "case_id": case_id,
            "split": "test",
            "split_group_id": runner.PROSPECTIVE_BUFFERED_OFFSET_GROUP_ID,
            "parameters": {
                "vortex_epsilon": 0.3875,
                "vortex_y": 0.35 + 0.0375 * index,
            },
        }
        for index, case_id in enumerate(runner.PROSPECTIVE_BUFFERED_OFFSET_CASE_IDS)
    }
    store = SimpleNamespace(
        keys=set(runner.PROSPECTIVE_BUFFERED_OFFSET_CASE_IDS), close=lambda: None
    )
    monkeypatch.setattr(
        runner,
        "_open_prospective_buffered_offset_contract",
        lambda _args: (
            None,
            object(),
            store,
            source,
            {"payload_sha256": "a13-payload"},
        ),
    )
    monkeypatch.setattr(
        runner.collection,
        "family_case_provenance",
        lambda _manifest, case_id: provenance[case_id],
    )
    output = tmp_path / "preflight.json"
    payload = runner.run_prospective_buffered_offset_preflight(
        SimpleNamespace(output=output)
    )
    assert payload["all_cases_in_named_e14_group"]
    assert payload["reference_arrays_loaded"] is False
    assert payload["recurrence_executed"] is False


def test_a23_preflight_binds_open_e14_without_model_or_recurrence(
    tmp_path, monkeypatch
):
    source = {
        "population": {"case_ids": list(runner.PROSPECTIVE_BUFFERED_OFFSET_CASE_IDS)}
    }
    provenance = {
        case_id: {
            "case_id": case_id,
            "split": "test",
            "split_group_id": runner.PROSPECTIVE_BUFFERED_OFFSET_GROUP_ID,
            "parameters": {
                "vortex_epsilon": 0.3875,
                "vortex_y": 0.35 + 0.0375 * index,
            },
        }
        for index, case_id in enumerate(runner.PROSPECTIVE_BUFFERED_OFFSET_CASE_IDS)
    }
    store = SimpleNamespace(
        keys=set(runner.PROSPECTIVE_BUFFERED_OFFSET_CASE_IDS), close=lambda: None
    )
    monkeypatch.setattr(
        runner,
        "_open_buffered_relaxed_tether_e14_contract",
        lambda _args: (
            None,
            object(),
            store,
            source,
            {"payload_sha256": "a22-r1"},
            {"payload_sha256": "a14"},
        ),
    )
    monkeypatch.setattr(
        runner.collection,
        "family_case_provenance",
        lambda _manifest, case_id: provenance[case_id],
    )
    output = tmp_path / "preflight.json"
    payload = runner.run_buffered_relaxed_tether_e14_preflight(
        SimpleNamespace(output=output)
    )
    assert payload["all_cases_in_opened_e14_group"]
    assert payload["new_population_opened"] is False
    assert payload["reference_arrays_loaded"] is False
    assert payload["recurrence_executed"] is False
    assert payload["registered_scored_calls"] == 580


def test_a23_preflight_verifier_fails_closed_on_e14_selection(monkeypatch):
    source_hashes = {"source.py": "source-hash"}
    payload = runner.with_payload_sha256(
        {
            "schema": runner.BUFFERED_RELAXED_TETHER_E14_PREFLIGHT_SCHEMA,
            "working_id": runner.BUFFERED_RELAXED_TETHER_E14_WORKING_ID,
            "status": "passed",
            "source_manifest": {
                "source_sha256": source_hashes,
                "qualified_a22_r1_rollout_sha256": "a22-r1-hash",
                "stopped_a14_comparator_sha256": "a14-hash",
                "population": {
                    "case_ids": list(runner.PROSPECTIVE_BUFFERED_OFFSET_CASE_IDS)
                },
            },
            "reference_arrays_loaded": False,
            "recurrence_executed": False,
            "new_population_opened": False,
            "still_sealed_groups": [],
            "all_cases_in_opened_e14_group": True,
            "warm_start_blocks": runner.FROZEN_OFFSET_WARM_BLOCKS,
            "buffered_relaxed_tether": True,
            "position_trust_threshold": runner.FROZEN_POSITION_TRUST_THRESHOLD,
            "position_buffer_threshold": runner.FROZEN_POSITION_BUFFER_THRESHOLD,
            "relaxation_rate": runner.RELAXED_TETHER_RATE,
            "truth_or_reference_used_at_inference": False,
            "e14_outcome_used_for_protocol_selection": False,
            "registered_scored_calls": 580,
        }
    )
    monkeypatch.setattr(runner, "_read_json", lambda _path: payload)
    monkeypatch.setattr(runner, "_source_hashes", lambda: source_hashes)
    monkeypatch.setattr(
        runner,
        "sha256_file",
        lambda path: "a14-hash" if "a14" in str(path) else "a22-r1-hash",
    )
    args = SimpleNamespace(a22_r1_rollout="a22-r1.json", a14_rollout="a14.json")
    assert runner._verify_buffered_relaxed_tether_e14_preflight(
        "preflight.json", args
    )
    payload["e14_outcome_used_for_protocol_selection"] = True
    payload["payload_sha256"] = runner.with_payload_sha256(
        {key: value for key, value in payload.items() if key != "payload_sha256"}
    )["payload_sha256"]
    with pytest.raises(ValueError, match="frozen contract"):
        runner._verify_buffered_relaxed_tether_e14_preflight("preflight.json", args)


def test_prospective_preflight_verifier_fails_closed_on_refit(monkeypatch):
    source_hashes = {"source.py": "source-hash"}
    payload = runner.with_payload_sha256(
        {
            "schema": runner.PROSPECTIVE_STRUCTURAL_PREFLIGHT_SCHEMA,
            "working_id": runner.PROSPECTIVE_STRUCTURAL_WORKING_ID,
            "status": "passed",
            "source_manifest": {
                "source_sha256": source_hashes,
                "qualified_a7_rollout_sha256": "a7-hash",
                "population": {
                    "case_ids": list(runner.PROSPECTIVE_STRUCTURAL_CASE_IDS),
                    "split_group_id": runner.PROSPECTIVE_STRUCTURAL_GROUP_ID,
                },
            },
            "reference_arrays_loaded": False,
            "recurrence_executed": False,
            "new_sealed_population_opened": runner.PROSPECTIVE_STRUCTURAL_GROUP_ID,
            "still_sealed_groups": ["strength_ood_e14"],
            "all_cases_in_named_e12_group": True,
            "coefficient_or_threshold_refit": False,
            "position_trust_threshold": FROZEN_POSITION_TRUST_THRESHOLD,
        }
    )
    monkeypatch.setattr(runner, "_read_json", lambda _path: payload)
    monkeypatch.setattr(runner, "_source_hashes", lambda: source_hashes)
    monkeypatch.setattr(runner, "sha256_file", lambda _path: "a7-hash")
    args = SimpleNamespace(a7_rollout="a7.json")

    assert runner._verify_prospective_structural_preflight("preflight.json", args)
    payload["coefficient_or_threshold_refit"] = True
    payload["payload_sha256"] = runner.with_payload_sha256(
        {key: value for key, value in payload.items() if key != "payload_sha256"}
    )["payload_sha256"]
    with pytest.raises(ValueError, match="differs from contract"):
        runner._verify_prospective_structural_preflight("preflight.json", args)


def test_shadow_candidate_preflight_binds_a8_and_forbids_selection(monkeypatch):
    source_hashes = {"source.py": "source-hash"}
    payload = runner.with_payload_sha256(
        {
            "schema": runner.SHADOW_CANDIDATE_PREFLIGHT_SCHEMA,
            "working_id": runner.SHADOW_CANDIDATE_WORKING_ID,
            "status": "passed",
            "source_manifest": {
                "source_sha256": source_hashes,
                "qualified_a8_rollout_sha256": "a8-hash",
                "population": {
                    "case_ids": list(runner.PROSPECTIVE_STRUCTURAL_CASE_IDS),
                },
                "protocol": {
                    "features": list(runner.A9_FEATURE_NAMES),
                    "targets": list(runner.A9_TARGET_NAMES),
                },
            },
            "reference_arrays_loaded": False,
            "recurrence_executed": False,
            "new_sealed_population_opened": False,
            "still_sealed_groups": ["strength_ood_e14"],
            "all_cases_in_opened_e12_group": True,
            "feature_or_threshold_selection": False,
            "coefficient_or_threshold_refit": False,
        }
    )
    monkeypatch.setattr(runner, "_read_json", lambda _path: payload)
    monkeypatch.setattr(runner, "_source_hashes", lambda: source_hashes)
    monkeypatch.setattr(runner, "sha256_file", lambda _path: "a8-hash")
    args = SimpleNamespace(a8_rollout="a8.json")

    assert runner._verify_shadow_candidate_preflight("preflight.json", args)
    payload["feature_or_threshold_selection"] = True
    payload["payload_sha256"] = runner.with_payload_sha256(
        {key: value for key, value in payload.items() if key != "payload_sha256"}
    )["payload_sha256"]
    with pytest.raises(ValueError, match="differs from contract"):
        runner._verify_shadow_candidate_preflight("preflight.json", args)


def test_shadow_candidate_preflight_cli_returns_zero(monkeypatch, capsys):
    args = SimpleNamespace(command="shadow-candidate-preflight")
    monkeypatch.setattr(runner, "parse_args", lambda _argv: args)
    monkeypatch.setattr(
        runner,
        "run_shadow_candidate_preflight",
        lambda _args: {"status": "passed"},
    )

    assert runner.main([]) == 0
    assert '"status": "passed"' in capsys.readouterr().out


def test_warm_start_preflight_binds_a8_a9_and_fails_closed_on_window(monkeypatch):
    source_hashes = {"source.py": "source-hash"}
    payload = runner.with_payload_sha256(
        {
            "schema": runner.WARM_START_PREFLIGHT_SCHEMA,
            "working_id": runner.WARM_START_WORKING_ID,
            "status": "passed",
            "source_manifest": {
                "source_sha256": source_hashes,
                "qualified_a8_rollout_sha256": "a8-hash",
                "completed_a9_rescore_sha256": "a9-hash",
                "population": {
                    "case_ids": list(runner.PROSPECTIVE_STRUCTURAL_CASE_IDS),
                },
            },
            "reference_arrays_loaded": False,
            "recurrence_executed": False,
            "new_sealed_population_opened": False,
            "still_sealed_groups": ["strength_ood_e14"],
            "all_cases_in_opened_e12_group": True,
            "warm_start_blocks": runner.WARM_START_BLOCKS,
            "coast_reset_to_shadow": False,
            "coefficient_or_threshold_refit": False,
        }
    )
    hashes = {"a8.json": "a8-hash", "a9.json": "a9-hash"}
    monkeypatch.setattr(runner, "_read_json", lambda _path: payload)
    monkeypatch.setattr(runner, "_source_hashes", lambda: source_hashes)
    monkeypatch.setattr(runner, "sha256_file", lambda path: hashes[str(path)])
    args = SimpleNamespace(a8_rollout="a8.json", a9_rescore="a9.json")

    assert runner._verify_warm_start_preflight("preflight.json", args)
    payload["warm_start_blocks"] = runner.WARM_START_BLOCKS + 1
    payload["payload_sha256"] = runner.with_payload_sha256(
        {key: value for key, value in payload.items() if key != "payload_sha256"}
    )["payload_sha256"]
    with pytest.raises(ValueError, match="differs from contract"):
        runner._verify_warm_start_preflight("preflight.json", args)


def test_warm_start_preflight_cli_returns_zero(monkeypatch, capsys):
    args = SimpleNamespace(command="warm-start-preflight")
    monkeypatch.setattr(runner, "parse_args", lambda _argv: args)
    monkeypatch.setattr(
        runner,
        "run_warm_start_preflight",
        lambda _args: {"status": "passed"},
    )

    assert runner.main([]) == 0
    assert '"status": "passed"' in capsys.readouterr().out


def test_projected_coast_preflight_binds_a10_and_active_cells(monkeypatch):
    source_hashes = {"source.py": "source-hash"}
    payload = runner.with_payload_sha256(
        {
            "schema": runner.PROJECTED_COAST_PREFLIGHT_SCHEMA,
            "working_id": runner.PROJECTED_COAST_WORKING_ID,
            "status": "passed",
            "source_manifest": {
                "source_sha256": source_hashes,
                "stopped_a10_rollout_sha256": "a10-hash",
                "population": {
                    "case_ids": list(runner.PROSPECTIVE_STRUCTURAL_CASE_IDS),
                },
            },
            "reference_arrays_loaded": False,
            "recurrence_executed": False,
            "new_sealed_population_opened": False,
            "still_sealed_groups": ["strength_ood_e14"],
            "all_cases_in_opened_e12_group": True,
            "warm_start_blocks": runner.WARM_START_BLOCKS,
            "projected_coast": True,
            "coast_projection_active_cells": [
                list(cell) for cell in runner.FROZEN_ACTIVE_CELLS
            ],
            "coefficient_or_threshold_refit": False,
        }
    )
    monkeypatch.setattr(runner, "_read_json", lambda _path: payload)
    monkeypatch.setattr(runner, "_source_hashes", lambda: source_hashes)
    monkeypatch.setattr(runner, "sha256_file", lambda _path: "a10-hash")
    args = SimpleNamespace(a10_rollout="a10.json")

    assert runner._verify_projected_coast_preflight("preflight.json", args)
    payload["coast_projection_active_cells"] = payload["coast_projection_active_cells"][
        :-1
    ]
    payload["payload_sha256"] = runner.with_payload_sha256(
        {key: value for key, value in payload.items() if key != "payload_sha256"}
    )["payload_sha256"]
    with pytest.raises(ValueError, match="differs from contract"):
        runner._verify_projected_coast_preflight("preflight.json", args)


def test_projected_coast_preflight_cli_returns_zero(monkeypatch, capsys):
    args = SimpleNamespace(command="projected-coast-preflight")
    monkeypatch.setattr(runner, "parse_args", lambda _argv: args)
    monkeypatch.setattr(
        runner,
        "run_projected_coast_preflight",
        lambda _args: {"status": "passed"},
    )

    assert runner.main([]) == 0
    assert '"status": "passed"' in capsys.readouterr().out


def test_frozen_offset_preflight_binds_a11_window_and_no_feedback(monkeypatch):
    source_hashes = {"source.py": "source-hash"}
    payload = runner.with_payload_sha256(
        {
            "schema": runner.FROZEN_OFFSET_PREFLIGHT_SCHEMA,
            "working_id": runner.FROZEN_OFFSET_WORKING_ID,
            "status": "passed",
            "source_manifest": {
                "source_sha256": source_hashes,
                "stopped_a11_rollout_sha256": "a11-hash",
                "population": {
                    "case_ids": list(runner.PROSPECTIVE_STRUCTURAL_CASE_IDS),
                },
            },
            "reference_arrays_loaded": False,
            "recurrence_executed": False,
            "new_sealed_population_opened": False,
            "still_sealed_groups": ["strength_ood_e14"],
            "all_cases_in_opened_e12_group": True,
            "warm_start_blocks": runner.FROZEN_OFFSET_WARM_BLOCKS,
            "frozen_offset": True,
            "coast_offset_feedback": False,
            "coast_projection_active_cells": [
                list(cell) for cell in runner.FROZEN_ACTIVE_CELLS
            ],
            "coefficient_or_threshold_refit": False,
        }
    )
    monkeypatch.setattr(runner, "_read_json", lambda _path: payload)
    monkeypatch.setattr(runner, "_source_hashes", lambda: source_hashes)
    monkeypatch.setattr(runner, "sha256_file", lambda _path: "a11-hash")
    args = SimpleNamespace(a11_rollout="a11.json")

    assert runner._verify_frozen_offset_preflight("preflight.json", args)
    payload["coast_offset_feedback"] = True
    payload["payload_sha256"] = runner.with_payload_sha256(
        {key: value for key, value in payload.items() if key != "payload_sha256"}
    )["payload_sha256"]
    with pytest.raises(ValueError, match="differs from contract"):
        runner._verify_frozen_offset_preflight("preflight.json", args)


def test_frozen_offset_preflight_cli_returns_zero(monkeypatch, capsys):
    args = SimpleNamespace(command="frozen-offset-preflight")
    monkeypatch.setattr(runner, "parse_args", lambda _argv: args)
    monkeypatch.setattr(
        runner,
        "run_frozen_offset_preflight",
        lambda _args: {"status": "passed"},
    )

    assert runner.main([]) == 0
    assert '"status": "passed"' in capsys.readouterr().out


def test_buffered_offset_preflight_cli_returns_zero(monkeypatch, capsys):
    args = SimpleNamespace(command="buffered-offset-preflight")
    monkeypatch.setattr(runner, "parse_args", lambda _argv: args)
    monkeypatch.setattr(
        runner,
        "run_buffered_offset_preflight",
        lambda _args: {"status": "passed"},
    )

    assert runner.main([]) == 0
    assert '"status": "passed"' in capsys.readouterr().out


def test_prospective_buffered_offset_cli_dispatches_preflight_and_rollout(
    monkeypatch, capsys
):
    preflight_args = SimpleNamespace(command="prospective-buffered-offset-preflight")
    monkeypatch.setattr(runner, "parse_args", lambda _argv: preflight_args)
    monkeypatch.setattr(
        runner,
        "run_prospective_buffered_offset_preflight",
        lambda _args: {"status": "passed"},
    )
    assert runner.main([]) == 0
    assert '"status": "passed"' in capsys.readouterr().out

    rollout_args = SimpleNamespace(command="prospective-buffered-offset-rollout")
    monkeypatch.setattr(runner, "parse_args", lambda _argv: rollout_args)
    monkeypatch.setattr(
        runner,
        "run_prospective_buffered_offset_rollout",
        lambda _args: ({"status": "qualified_prospective"}, 0),
    )
    assert runner.main([]) == 0
    assert '"status": "qualified_prospective"' in capsys.readouterr().out


def test_persistence_gain_cli_dispatches_preflight_and_rollout(monkeypatch, capsys):
    preflight_args = SimpleNamespace(command="persistence-gain-preflight")
    monkeypatch.setattr(runner, "parse_args", lambda _argv: preflight_args)
    monkeypatch.setattr(
        runner,
        "run_persistence_gain_preflight",
        lambda _args: {"status": "passed"},
    )
    assert runner.main([]) == 0
    assert '"status": "passed"' in capsys.readouterr().out

    rollout_args = SimpleNamespace(command="persistence-gain-rollout")
    monkeypatch.setattr(runner, "parse_args", lambda _argv: rollout_args)
    monkeypatch.setattr(
        runner,
        "run_persistence_gain_rollout",
        lambda _args: ({"status": "qualified_calibration"}, 0),
    )
    assert runner.main([]) == 0
    assert '"status": "qualified_calibration"' in capsys.readouterr().out


def test_terminal_ramp_cli_dispatches_preflight_and_rollout(monkeypatch, capsys):
    preflight_args = SimpleNamespace(command="terminal-ramp-preflight")
    monkeypatch.setattr(runner, "parse_args", lambda _argv: preflight_args)
    monkeypatch.setattr(
        runner,
        "run_terminal_ramp_preflight",
        lambda _args: {"status": "passed"},
    )
    assert runner.main([]) == 0
    assert '"status": "passed"' in capsys.readouterr().out

    rollout_args = SimpleNamespace(command="terminal-ramp-rollout")
    monkeypatch.setattr(runner, "parse_args", lambda _argv: rollout_args)
    monkeypatch.setattr(
        runner,
        "run_terminal_ramp_rollout",
        lambda _args: ({"status": "qualified_calibration"}, 0),
    )
    assert runner.main([]) == 0
    assert '"status": "qualified_calibration"' in capsys.readouterr().out


def test_fixed_late_ramp_cli_dispatches_preflight_and_rollout(monkeypatch, capsys):
    preflight_args = SimpleNamespace(command="fixed-late-ramp-preflight")
    monkeypatch.setattr(runner, "parse_args", lambda _argv: preflight_args)
    monkeypatch.setattr(
        runner,
        "run_fixed_late_ramp_preflight",
        lambda _args: {"status": "passed"},
    )
    assert runner.main([]) == 0
    assert '"status": "passed"' in capsys.readouterr().out

    rollout_args = SimpleNamespace(command="fixed-late-ramp-rollout")
    monkeypatch.setattr(runner, "parse_args", lambda _argv: rollout_args)
    monkeypatch.setattr(
        runner,
        "run_fixed_late_ramp_rollout",
        lambda _args: ({"status": "qualified_calibration"}, 0),
    )
    assert runner.main([]) == 0
    assert '"status": "qualified_calibration"' in capsys.readouterr().out


def test_fixed_late_ramp_rescore_cli_returns_zero(monkeypatch, capsys):
    args = SimpleNamespace(command="fixed-late-ramp-rescore")
    monkeypatch.setattr(runner, "parse_args", lambda _argv: args)
    monkeypatch.setattr(
        runner,
        "run_fixed_late_ramp_rescore",
        lambda _args: ({"status": "qualified_calibration_rescore"}, 0),
    )

    assert runner.main([]) == 0
    assert '"status": "qualified_calibration_rescore"' in capsys.readouterr().out


def test_fixed_late_ramp_population_preserves_a16_runtime_lineage(monkeypatch, tmp_path):
    marker13 = {"status": "qualified_calibration"}
    marker16 = {"status": "stopped_calibration"}
    built = (
        object(),
        {"payload_sha256": "source"},
        object(),
        object(),
        object(),
        object(),
        object(),
        object(),
        object(),
        object(),
        object(),
        marker13,
        marker16,
    )
    monkeypatch.setattr(
        runner,
        "_verify_fixed_late_ramp_preflight",
        lambda *_: {"source_manifest": built[1]},
    )
    monkeypatch.setattr(runner, "_build_fixed_late_ramp_runtime", lambda _args: built)
    monkeypatch.setattr(runner.torch, "use_deterministic_algorithms", lambda _value: None)
    monkeypatch.setattr(runner.collection, "_close_runtime", lambda _runtime: None)

    def stop_after_binding(_runtime, **_kwargs):
        frame = __import__("inspect").currentframe().f_back
        assert frame.f_locals["a13_rollout"] is marker13
        assert frame.f_locals["a16_rollout"] is marker16
        raise RuntimeError("bound")

    monkeypatch.setattr(runner, "_shadow_rollout_arm", stop_after_binding)
    args = SimpleNamespace(
        output_dir=tmp_path / "out",
        preflight=tmp_path / "preflight.json",
        shard_native_reference_audit=None,
    )
    with pytest.raises(RuntimeError, match="bound"):
        runner._run_structural_gate_population(
            args,
            prospective=False,
            warm_start=True,
            frozen_offset=True,
            buffered_offset=True,
            fixed_late_ramp=True,
        )


def test_buffered_offset_rescore_cli_returns_zero(monkeypatch, capsys):
    args = SimpleNamespace(command="buffered-offset-rescore")
    monkeypatch.setattr(runner, "parse_args", lambda _argv: args)
    monkeypatch.setattr(
        runner,
        "run_buffered_offset_rescore",
        lambda _args: ({"status": "qualified_calibration_rescore"}, 0),
    )

    assert runner.main([]) == 0
    assert '"status": "qualified_calibration_rescore"' in capsys.readouterr().out


def test_buffered_offset_call_checks_bind_three_way_route():
    zero = np.zeros((2, 4), dtype=np.float64)
    shadow = [zero + call for call in range(31)]

    def base(case_id, distance, route):
        return {
            "case_id": case_id,
            "position_descriptor": {
                "status": "ok",
                "vertical_centroid": 0.5 * distance,
                "normalized_wall_distance": distance,
                "transverse_velocity_l1": 1.0,
            },
            "position_trusted": route == "edge_candidate",
            "position_route": route,
            "position_buffered": route == "raw_uncertainty_buffer",
            "buffered_offset": True,
            "warm_start_blocks": runner.FROZEN_OFFSET_WARM_BLOCKS,
            "frozen_offset": True,
            "front_branch_veto_call": None,
            "projected_coast_audits": [],
            "maxima": {
                "call_order_errors": 0,
                "shadow_recurrence_abs": 0.0,
                "accepted_recurrence_abs": 0.0,
                "candidate_common_native_input_abs": 0.0,
                "candidate_lookahead_input_abs": 0.0,
                "accepted_coast_shadow_integral_abs": 0.0,
                "frozen_offset_integral_abs": 0.0,
                "frozen_offset_boundary_abs": 0.0,
                "frozen_offset_idempotence_abs": 0.0,
                "frozen_offset_constancy_abs": 0.0,
                "frozen_offset_increment_identity_abs": 0.0,
            },
        }

    buffered = base("buffer", 0.775, "raw_uncertainty_buffer")
    buffered.update(
        {
            "front_audits": [],
            "frozen_offset_audits": [],
            "execution": {
                "logical_model_calls": 30,
                "native_logical_calls": 30,
                "fine_logical_calls": 0,
                "shadow_native_logical_calls": 30,
                "candidate_native_logical_calls": 0,
                "candidate_fine_logical_calls": 0,
                "accepted_coast_native_logical_calls": 0,
                "completed_calls": 30,
            },
            "accepted_states": [value.copy() for value in shadow],
            "shadow_states": [value.copy() for value in shadow],
            "accepted_rows": [
                {"correction_status": "position_uncertainty_buffer_raw"}
                for _ in range(30)
            ],
        }
    )

    interior = base("interior", 0.85, "interior_frozen_offset")
    handoff = 2 * runner.FROZEN_OFFSET_WARM_BLOCKS
    offset = np.full_like(zero, 0.1)
    accepted = [value.copy() for value in shadow]
    accepted[1:handoff] = [value + 0.05 for value in shadow[1:handoff]]
    accepted[handoff:] = [value + offset for value in shadow[handoff:]]
    front = [
        {
            "phase": "warm_candidate",
            "branch_changed": False,
            "first_output_call": 2 * block + 1,
        }
        for block in range(runner.FROZEN_OFFSET_WARM_BLOCKS)
    ]
    front.extend(
        {
            "phase": "frozen_offset_coast",
            "branch_changed": False,
            "first_output_call": 2 * block + 1,
        }
        for block in range(runner.FROZEN_OFFSET_WARM_BLOCKS, 15)
    )
    interior.update(
        {
            "front_audits": front,
            "frozen_offset_audits": [{"output_call": handoff}],
            "execution": {
                "logical_model_calls": 46,
                "native_logical_calls": 42,
                "fine_logical_calls": 4,
                "shadow_native_logical_calls": 30,
                "candidate_native_logical_calls": 12,
                "candidate_fine_logical_calls": 4,
                "accepted_coast_native_logical_calls": 0,
                "completed_calls": 30,
            },
            "accepted_states": accepted,
            "shadow_states": [value.copy() for value in shadow],
            "accepted_rows": [
                {
                    "correction_status": (
                        "frozen_sp19_output_offset"
                        if call >= handoff
                        else "warm_sp19_plus_integral_anchor"
                    )
                }
                for call in range(30)
            ],
        }
    )

    checks = runner._buffered_offset_call_checks(
        [buffered, interior], case_ids=("buffer", "interior")
    )
    assert all(checks.values())
    buffered["accepted_states"][-1] = zero + 99.0
    checks = runner._buffered_offset_call_checks(
        [buffered, interior], case_ids=("buffer", "interior")
    )
    assert not checks["buffered_offset_raw_buffer_recurrence_exact"]


def test_persistence_gain_call_checks_bind_probe_inventory_and_monotone_gain():
    zero = np.zeros((2, 4), dtype=np.float64)
    shadow = [zero + call for call in range(31)]
    case_id = "interior"
    audits = [
        {
            "block_index": block,
            "output_call": 2 * block + 1,
            "applied_gain": 1.0 - 0.05 * (block - 4),
        }
        for block in range(4, 15)
    ]
    rollout = {
        "case_id": case_id,
        "position_descriptor": {
            "status": "ok",
            "vertical_centroid": 0.45,
            "normalized_wall_distance": 0.9,
            "transverse_velocity_l1": 1.0,
        },
        "position_route": "interior_frozen_offset",
        "position_buffered": False,
        "buffered_offset": True,
        "persistence_probe": True,
        "front_audits": [
            {
                "phase": (
                    "warm_candidate"
                    if block < runner.FROZEN_OFFSET_WARM_BLOCKS
                    else "persistence_offset_coast"
                )
            }
            for block in range(15)
        ],
        "frozen_offset_audits": [{"output_call": 8}],
        "persistence_probe_audits": audits,
        "execution": {
            "logical_model_calls": 57,
            "native_logical_calls": 42,
            "fine_logical_calls": 15,
            "candidate_native_logical_calls": 12,
            "candidate_fine_logical_calls": 4,
            "persistence_probe_fine_logical_calls": 11,
            "accepted_coast_native_logical_calls": 0,
        },
        "accepted_states": [value.copy() for value in shadow],
        "shadow_states": [value.copy() for value in shadow],
        "maxima": {
            "persistence_common_native_input_abs": 2.0e-7,
            "persistence_fine_input_abs": 0.0,
            "persistence_offset_boundary_abs": 0.0,
            "persistence_offset_integral_abs": 1.0e-14,
            "persistence_gain_increase_abs": 0.0,
            "persistence_offset_bookkeeping_abs": 1.0e-14,
            "persistence_within_block_identity_abs": 1.0e-14,
            "call_order_errors": 0,
        },
    }
    checks = runner._persistence_gain_call_checks([rollout], case_ids=(case_id,))
    assert all(checks.values())
    audits[-1]["applied_gain"] = 0.9
    checks = runner._persistence_gain_call_checks([rollout], case_ids=(case_id,))
    assert not checks["persistence_gain_and_offset_bookkeeping_exact"]


def test_terminal_ramp_call_checks_bind_phase_trigger_and_linear_retirement():
    zero = np.zeros((2, 4), dtype=np.float64)
    shadow = [zero + call for call in range(31)]
    audits = []
    for block in range(4, 15):
        trigger = 8
        gain = 1.0 if block <= trigger else (14 - block) / (14 - trigger)
        audits.append(
            {
                "block_index": block,
                "output_call": 2 * block + 1,
                "alignment_cosine": 0.1 if block < trigger else -0.1,
                "applied_gain": gain,
                "gain_status": (
                    "pretrigger_unit_gain"
                    if block < trigger
                    else "phase_triggered_terminal_ramp"
                ),
            }
        )
    rollout = {
        "case_id": "interior",
        "position_descriptor": {
            "status": "ok",
            "vertical_centroid": 0.45,
            "normalized_wall_distance": 0.9,
            "transverse_velocity_l1": 1.0,
        },
        "position_route": "interior_frozen_offset",
        "position_buffered": False,
        "buffered_offset": True,
        "persistence_probe": True,
        "terminal_ramp": True,
        "terminal_ramp_trigger_block": 8,
        "front_audits": [
            {
                "phase": (
                    "warm_candidate"
                    if block < runner.FROZEN_OFFSET_WARM_BLOCKS
                    else "terminal_ramp_offset_coast"
                )
            }
            for block in range(15)
        ],
        "frozen_offset_audits": [{"output_call": 8}],
        "persistence_probe_audits": audits,
        "execution": {
            "logical_model_calls": 57,
            "native_logical_calls": 42,
            "fine_logical_calls": 15,
            "candidate_native_logical_calls": 12,
            "candidate_fine_logical_calls": 4,
            "persistence_probe_fine_logical_calls": 11,
            "accepted_coast_native_logical_calls": 0,
        },
        "accepted_states": [value.copy() for value in shadow],
        "shadow_states": [value.copy() for value in shadow],
        "maxima": {
            "persistence_common_native_input_abs": 2.0e-7,
            "persistence_fine_input_abs": 0.0,
            "persistence_offset_boundary_abs": 0.0,
            "persistence_offset_integral_abs": 1.0e-14,
            "persistence_gain_increase_abs": 0.0,
            "persistence_offset_bookkeeping_abs": 1.0e-14,
            "persistence_within_block_identity_abs": 1.0e-14,
            "call_order_errors": 0,
        },
    }
    checks = runner._terminal_ramp_call_checks(
        [rollout], case_ids=("interior",)
    )
    assert all(checks.values())
    audits[6]["applied_gain"] += 0.1
    checks = runner._terminal_ramp_call_checks(
        [rollout], case_ids=("interior",)
    )
    assert not checks["terminal_ramp_trigger_and_linear_path_exact"]


def test_fixed_late_ramp_call_checks_allow_only_registered_interior_ramp():
    zero = np.zeros((2, 4), dtype=np.float64)
    shadow = [zero + call for call in range(31)]
    handoff = 2 * runner.FROZEN_OFFSET_WARM_BLOCKS
    offset = np.full_like(zero, 0.1)
    gains = [
        fixed_late_terminal_ramp_gain(
            block_index=block,
            ramp_start_block=9,
            terminal_block=14,
        )
        for block in range(4, 15)
    ]
    accepted = [value.copy() for value in shadow]
    accepted[1:handoff] = [value + 0.05 for value in shadow[1:handoff]]
    for call in range(handoff, 31):
        accepted[call] = shadow[call] + fixed_late_terminal_ramp_gain(
            block_index=(call - 1) // 2,
            ramp_start_block=9,
            terminal_block=14,
        ) * offset
    rollout = {
        "case_id": "interior",
        "position_descriptor": {
            "status": "ok",
            "vertical_centroid": 0.45,
            "normalized_wall_distance": 0.9,
            "transverse_velocity_l1": 1.0,
        },
        "position_route": "interior_frozen_offset",
        "position_buffered": False,
        "buffered_offset": True,
        "fixed_late_ramp": True,
        "persistence_probe": False,
        "front_audits": [
            {
                "phase": (
                    "warm_candidate"
                    if block < runner.FROZEN_OFFSET_WARM_BLOCKS
                    else "fixed_late_ramp_offset_coast"
                )
            }
            for block in range(15)
        ],
        "frozen_offset_audits": [{"output_call": handoff}],
        "persistence_probe_audits": [
            {
                "block_index": block,
                "output_call": 2 * block + 1,
                "applied_gain": gain,
                "gain_status": "fixed_late_terminal_ramp",
            }
            for block, gain in zip(range(4, 15), gains, strict=True)
        ],
        "execution": {
            "logical_model_calls": 46,
            "native_logical_calls": 42,
            "fine_logical_calls": 4,
            "candidate_native_logical_calls": 12,
            "candidate_fine_logical_calls": 4,
            "persistence_probe_fine_logical_calls": 0,
            "accepted_coast_native_logical_calls": 0,
            "completed_calls": 30,
        },
        "accepted_states": accepted,
        "shadow_states": [value.copy() for value in shadow],
        "accepted_rows": [
            {
                "correction_status": (
                    "fixed_late_terminal_ramp_sp19_output_offset"
                    if call >= handoff
                    else "warm_sp19_plus_integral_anchor"
                )
            }
            for call in range(31)
        ],
        "maxima": {
            "persistence_gain_increase_abs": 0.0,
            "persistence_offset_bookkeeping_abs": 0.0,
            "persistence_within_block_identity_abs": 0.0,
            "call_order_errors": 0,
        },
    }
    checks = runner._fixed_late_ramp_call_checks(
        [rollout], case_ids=("interior",)
    )
    assert all(checks.values())

    rollout["persistence_probe_audits"][-1]["applied_gain"] = 0.1
    checks = runner._fixed_late_ramp_call_checks(
        [rollout], case_ids=("interior",)
    )
    assert not checks["fixed_late_ramp_schedule_cost_and_bookkeeping_exact"]


def test_structural_gate_call_checks_bind_position_reject_and_latched_fallback():
    state = np.zeros((2, 4), dtype=np.float64)
    raw_states = [state + call for call in range(31)]
    trusted_states = [value.copy() for value in raw_states]
    trusted_states[1:29] = [value + 0.1 for value in raw_states[1:29]]
    rollouts = [
        {
            "case_id": "interior",
            "position_descriptor": {
                "status": "ok",
                "normalized_wall_distance": 1.0,
            },
            "position_trusted": False,
            "front_audits": [],
            "front_branch_veto_call": None,
            "execution": {
                "logical_model_calls": 30,
                "native_logical_calls": 30,
                "fine_logical_calls": 0,
                "shadow_native_logical_calls": 30,
                "candidate_native_logical_calls": 0,
                "candidate_fine_logical_calls": 0,
                "completed_calls": 30,
            },
            "maxima": {
                "call_order_errors": 0,
                "shadow_recurrence_abs": 0.0,
                "accepted_recurrence_abs": 0.0,
                "candidate_common_native_input_abs": 0.0,
                "candidate_lookahead_input_abs": 0.0,
            },
            "accepted_states": raw_states,
            "shadow_states": raw_states,
        },
        {
            "case_id": "trusted_then_vetoed",
            "position_descriptor": {
                "status": "ok",
                "normalized_wall_distance": 0.7,
            },
            "position_trusted": True,
            "front_audits": [
                {
                    "first_output_call": 2 * block + 1,
                    "branch_changed": block == 14,
                }
                for block in range(15)
            ],
            "front_branch_veto_call": 29,
            "execution": {
                "logical_model_calls": 90,
                "native_logical_calls": 75,
                "fine_logical_calls": 15,
                "shadow_native_logical_calls": 30,
                "candidate_native_logical_calls": 45,
                "candidate_fine_logical_calls": 15,
                "completed_calls": 30,
            },
            "maxima": {
                "call_order_errors": 0,
                "shadow_recurrence_abs": 0.0,
                "accepted_recurrence_abs": 2.4e-7,
                "candidate_common_native_input_abs": 0.0,
                "candidate_lookahead_input_abs": 0.0,
            },
            "accepted_states": trusted_states,
            "shadow_states": raw_states,
        },
    ]
    checks = runner._structural_gate_call_checks(
        rollouts, case_ids=("interior", "trusted_then_vetoed")
    )
    assert all(checks.values())

    rollouts[1]["accepted_states"][-1] = state + 99.0
    checks = runner._structural_gate_call_checks(
        rollouts, case_ids=("interior", "trusted_then_vetoed")
    )
    assert not checks["structural_gate_fallback_recurrence_exact"]


def test_warm_start_call_checks_bind_window_coast_and_accepted_recurrence():
    case_ids = ("trusted", "interior")
    zero = np.zeros((2, 4), dtype=np.float64)
    shadow_states = [zero + call for call in range(31)]

    def rollout(case_id: str, *, trusted: bool):
        candidate_blocks = 15 if trusted else runner.WARM_START_BLOCKS
        coast_blocks = 0 if trusted else 15 - runner.WARM_START_BLOCKS
        accepted_states = [value.copy() for value in shadow_states]
        if not trusted:
            accepted_states[1:] = [value + 0.1 for value in shadow_states[1:]]
        return {
            "case_id": case_id,
            "position_descriptor": {
                "status": "ok",
                "normalized_wall_distance": 0.5 if trusted else 1.0,
            },
            "position_trusted": trusted,
            "front_audits": [
                {
                    "branch_changed": False,
                    "first_output_call": 2 * block + 1,
                }
                for block in range(candidate_blocks)
            ],
            "front_branch_veto_call": None,
            "warm_start_blocks": runner.WARM_START_BLOCKS,
            "execution": {
                "logical_model_calls": 30 + 4 * candidate_blocks + 2 * coast_blocks,
                "native_logical_calls": 30 + 3 * candidate_blocks + 2 * coast_blocks,
                "fine_logical_calls": candidate_blocks,
                "shadow_native_logical_calls": 30,
                "candidate_native_logical_calls": 3 * candidate_blocks,
                "candidate_fine_logical_calls": candidate_blocks,
                "accepted_coast_native_logical_calls": 2 * coast_blocks,
                "completed_calls": 30,
            },
            "accepted_rows": [
                {
                    "correction_status": (
                        "accepted_native_coast"
                        if not trusted and call >= 2 * runner.WARM_START_BLOCKS
                        else "accepted_candidate"
                    )
                }
                for call in range(30)
            ],
            "accepted_states": accepted_states,
            "shadow_states": shadow_states,
            "maxima": {
                "call_order_errors": 0,
                "shadow_recurrence_abs": 0.0,
                "accepted_recurrence_abs": 2.0e-7,
                "accepted_coast_recurrence_abs": 2.0e-7,
                "candidate_common_native_input_abs": 0.0,
                "candidate_lookahead_input_abs": 0.0,
            },
        }

    rollouts = [
        rollout("trusted", trusted=True),
        rollout("interior", trusted=False),
    ]
    checks = runner._warm_start_call_checks(rollouts, case_ids=case_ids)
    assert all(checks.values())

    rollouts[1]["execution"]["accepted_coast_native_logical_calls"] -= 1
    checks = runner._warm_start_call_checks(rollouts, case_ids=case_ids)
    assert not checks["warm_start_call_counts_exact"]


def test_projected_coast_call_checks_bind_projection_and_required_shadow():
    case_ids = ("trusted", "interior")
    zero = np.zeros((2, 4), dtype=np.float64)
    shadow_states = [zero + call for call in range(31)]

    def rollout(case_id: str, *, trusted: bool):
        warm_blocks = 15 if trusted else runner.WARM_START_BLOCKS
        coast_blocks = 0 if trusted else 15 - runner.WARM_START_BLOCKS
        accepted_states = [value.copy() for value in shadow_states]
        if not trusted:
            accepted_states[1:] = [value + 0.1 for value in shadow_states[1:]]
        front_audits = [
            {
                "phase": "trusted_candidate" if trusted else "warm_candidate",
                "branch_changed": False,
                "first_output_call": 2 * block + 1,
            }
            for block in range(warm_blocks)
        ]
        front_audits.extend(
            {
                "phase": "projected_coast",
                "branch_changed": False,
                "first_output_call": 2 * block + 1,
            }
            for block in range(runner.WARM_START_BLOCKS, 15)
            if not trusted
        )
        return {
            "case_id": case_id,
            "position_descriptor": {
                "status": "ok",
                "normalized_wall_distance": 0.5 if trusted else 1.0,
            },
            "position_trusted": trusted,
            "front_audits": front_audits,
            "front_branch_veto_call": None,
            "warm_start_blocks": runner.WARM_START_BLOCKS,
            "projected_coast": True,
            "projected_coast_audits": [
                {"output_call": call}
                for call in range(2 * runner.WARM_START_BLOCKS + 1, 31)
            ]
            if not trusted
            else [],
            "execution": {
                "logical_model_calls": 30 + 4 * warm_blocks + 2 * coast_blocks,
                "native_logical_calls": 30 + 3 * warm_blocks + 2 * coast_blocks,
                "fine_logical_calls": warm_blocks,
                "shadow_native_logical_calls": 30,
                "candidate_native_logical_calls": 3 * warm_blocks,
                "candidate_fine_logical_calls": warm_blocks,
                "accepted_coast_native_logical_calls": 2 * coast_blocks,
                "completed_calls": 30,
            },
            "accepted_rows": [
                {
                    "correction_status": (
                        "projected_coast_sp19_tether"
                        if not trusted and call >= 2 * runner.WARM_START_BLOCKS
                        else "accepted_candidate"
                    )
                }
                for call in range(30)
            ],
            "accepted_states": accepted_states,
            "shadow_states": shadow_states,
            "maxima": {
                "call_order_errors": 0,
                "shadow_recurrence_abs": 0.0,
                "accepted_recurrence_abs": 2.0e-7,
                "accepted_coast_recurrence_abs": 2.0e-7,
                "candidate_common_native_input_abs": 0.0,
                "candidate_lookahead_input_abs": 0.0,
                "accepted_coast_shadow_integral_abs": 1.0e-14,
                "projected_coast_boundary_abs": 0.0,
                "projected_coast_idempotence_abs": 1.0e-14,
                "projected_coast_tether_abs": 0.0,
            },
        }

    rollouts = [
        rollout("trusted", trusted=True),
        rollout("interior", trusted=False),
    ]
    checks = runner._projected_coast_call_checks(rollouts, case_ids=case_ids)
    assert all(checks.values())

    rollouts[1]["maxima"]["accepted_coast_shadow_integral_abs"] = 1.1e-10
    checks = runner._projected_coast_call_checks(rollouts, case_ids=case_ids)
    assert not checks["projected_coast_common_source_lookahead_and_projection_exact"]


def test_frozen_offset_call_checks_bind_raw_coast_and_increment_identity():
    case_ids = ("trusted", "interior")
    zero = np.zeros((2, 4), dtype=np.float64)
    shadow_states = [zero + call for call in range(31)]

    def rollout(case_id: str, *, trusted: bool):
        warm_blocks = 15 if trusted else runner.FROZEN_OFFSET_WARM_BLOCKS
        accepted_states = [value.copy() for value in shadow_states]
        if not trusted:
            offset = np.full_like(zero, 0.1)
            accepted_states[1 : 2 * warm_blocks] = [
                value + 0.05 for value in shadow_states[1 : 2 * warm_blocks]
            ]
            accepted_states[2 * warm_blocks :] = [
                value + offset for value in shadow_states[2 * warm_blocks :]
            ]
        front_audits = [
            {
                "phase": "trusted_candidate" if trusted else "warm_candidate",
                "branch_changed": False,
                "first_output_call": 2 * block + 1,
            }
            for block in range(warm_blocks)
        ]
        front_audits.extend(
            {
                "phase": "frozen_offset_coast",
                "branch_changed": False,
                "first_output_call": 2 * block + 1,
            }
            for block in range(runner.FROZEN_OFFSET_WARM_BLOCKS, 15)
            if not trusted
        )
        return {
            "case_id": case_id,
            "position_descriptor": {
                "status": "ok",
                "normalized_wall_distance": 0.5 if trusted else 1.0,
            },
            "position_trusted": trusted,
            "front_audits": front_audits,
            "front_branch_veto_call": None,
            "warm_start_blocks": runner.FROZEN_OFFSET_WARM_BLOCKS,
            "frozen_offset": True,
            "frozen_offset_audits": ([] if trusted else [{"output_call": 8}]),
            "execution": {
                "logical_model_calls": 30 + 4 * warm_blocks,
                "native_logical_calls": 30 + 3 * warm_blocks,
                "fine_logical_calls": warm_blocks,
                "shadow_native_logical_calls": 30,
                "candidate_native_logical_calls": 3 * warm_blocks,
                "candidate_fine_logical_calls": warm_blocks,
                "accepted_coast_native_logical_calls": 0,
                "completed_calls": 30,
            },
            "accepted_rows": [
                {
                    "correction_status": (
                        "frozen_sp19_output_offset"
                        if not trusted and call >= 2 * warm_blocks
                        else "accepted_candidate"
                    )
                }
                for call in range(30)
            ],
            "accepted_states": accepted_states,
            "shadow_states": shadow_states,
            "maxima": {
                "call_order_errors": 0,
                "shadow_recurrence_abs": 0.0,
                "accepted_recurrence_abs": 2.0e-7,
                "candidate_common_native_input_abs": 0.0,
                "candidate_lookahead_input_abs": 0.0,
                "accepted_coast_shadow_integral_abs": 1.0e-14,
                "frozen_offset_integral_abs": 1.0e-14,
                "frozen_offset_boundary_abs": 0.0,
                "frozen_offset_idempotence_abs": 1.0e-14,
                "frozen_offset_constancy_abs": 0.0,
                "frozen_offset_increment_identity_abs": 0.0,
            },
        }

    rollouts = [
        rollout("trusted", trusted=True),
        rollout("interior", trusted=False),
    ]
    checks = runner._frozen_offset_call_checks(rollouts, case_ids=case_ids)
    assert all(checks.values())

    rollouts[1]["accepted_states"][-1] += 1.0e-5
    checks = runner._frozen_offset_call_checks(rollouts, case_ids=case_ids)
    assert not checks["frozen_offset_raw_coast_and_fallback_exact"]


def test_structural_gate_rollout_vetoes_whole_block_and_latches_raw(monkeypatch):
    state, volumes, projector, _, contract = _fixture()
    nx, ny = contract.native
    xx, yy = np.meshgrid(
        (np.arange(nx, dtype=np.float64) + 0.5) * (2.0 / nx),
        (np.arange(ny, dtype=np.float64) + 0.5) * (1.0 / ny),
        indexing="xy",
    )
    nodes = np.column_stack((xx.reshape(-1), yy.reshape(-1)))
    x = nodes[:, 0]
    y = nodes[:, 1]
    state[:, 0] = 1.0
    state[:, 1] = 0.2
    state[:, 2] = np.exp(-(((y - 0.125) / 0.08) ** 2)) * np.sin(2.0 * np.pi * x)
    pressure = np.where(x < 0.5, 1.2, 1.0)
    state[:, 3] = (
        pressure / 0.4 + 0.5 * (state[:, 1] ** 2 + state[:, 2] ** 2) / state[:, 0]
    )
    reference_states = np.stack([state + 0.005 * call for call in range(61)], axis=0)

    monkeypatch.setattr(runner, "NATIVE_RESOLUTION", contract.native)
    monkeypatch.setattr(runner.parent, "NATIVE_RESOLUTION", contract.native)
    monkeypatch.setattr(runner.parent, "RESOLUTION_CONTRACT", contract)
    monkeypatch.setattr(runner.collection, "RESOLUTION_CONTRACT", contract)

    def load_reference(*_args, **_kwargs):
        return (
            {
                "conservative_states": reference_states,
                "retained_resolution": contract.native,
            },
            {"retained_resolution": f"{nx}x{ny}"},
        )

    def predict(_model, _sample, value, **_kwargs):
        return np.asarray(value, dtype=np.float64) + 0.01, {
            "forward_seconds": [0.0],
            "peak_gpu_memory_bytes": 0,
        }

    zero_anchor = IntegralAnchorAudit(
        status="ok",
        maximum_integral_mismatch_before_abs=0.0,
        maximum_integral_mismatch_after_abs=0.0,
        maximum_boundary_correction_abs=0.0,
        maximum_idempotence_abs=0.0,
        correction_rms=0.0,
    )
    zero_response = ResponseProjectionAudit(
        status="ok",
        maximum_first_correction_integral_abs=0.0,
        maximum_filtered_response_integral_abs=0.0,
        maximum_first_boundary_abs=0.0,
        maximum_filtered_boundary_abs=0.0,
        maximum_projection_idempotence_abs=0.0,
        full_response_rms=0.0,
        filtered_response_rms=0.0,
        retained_response_fraction=None,
    )

    def fake_block(accepted, shadow, *, contract, predictor, **_kwargs):
        shadow_first = predictor(contract.native, shadow)
        shadow_second = predictor(contract.native, shadow_first)
        prepared = prepare_common_native_inputs(accepted, contract=contract)
        raw_first = predictor(contract.native, prepared.model_inputs[contract.native])
        predictor(contract.fine, prepared.model_inputs[contract.fine])
        correction = np.zeros_like(raw_first)
        correction[:, 2] = 0.002 * np.sin(2.0 * np.pi * nodes[:, 1])
        anchored_first = raw_first + correction
        raw_second = predictor(contract.native, raw_first)
        corrected_second = predictor(contract.native, anchored_first)
        anchored_second = corrected_second + correction
        return ShadowAnchoredBlock(
            shadow_first_state=shadow_first,
            shadow_second_state=shadow_second,
            candidate_raw_first_state=raw_first,
            candidate_first_before_anchor=anchored_first,
            anchored_first_state=anchored_first,
            candidate_raw_second_state=raw_second,
            full_response=correction,
            filtered_response=correction,
            candidate_second_before_anchor=anchored_second,
            anchored_second_state=anchored_second,
            first_correction=correction,
            first_anchor_correction=np.zeros_like(correction),
            second_anchor_correction=np.zeros_like(correction),
            first_audit=SimpleNamespace(cap_active=False),
            response_audit=zero_response,
            first_anchor_audit=zero_anchor,
            second_anchor_audit=zero_anchor,
        )

    audit_calls = 0

    def fake_front_audit(*_args, **_kwargs):
        nonlocal audit_calls
        audit_calls += 1
        changed = audit_calls == 2
        return TargetFreeFrontBranchAudit(
            status="branch_changed" if changed else "matched",
            branch_changed=changed,
            first_shadow_thickness_cells=1,
            first_candidate_thickness_cells=1,
            second_shadow_thickness_cells=1,
            second_candidate_thickness_cells=2 if changed else 1,
            maximum_position_shift=0.0,
        )

    monkeypatch.setattr(runner, "load_resolution_reference", load_reference)
    monkeypatch.setattr(runner, "predict_resolution_sample", predict)
    monkeypatch.setattr(runner, "synchronized_shadow_anchored_block", fake_block)
    monkeypatch.setattr(runner, "target_free_front_branch_audit", fake_front_audit)
    monkeypatch.setattr(
        runner.parent,
        "shock_vortex_regions",
        lambda *_args, **_kwargs: (
            {
                key: np.ones(nx * ny, dtype=bool)
                for key in (
                    "boundary_le_0.05",
                    "partition_shock",
                    "partition_vortex",
                    "partition_smooth",
                )
            },
            {},
            {},
        ),
    )
    runtime = SimpleNamespace(
        args=SimpleNamespace(family_root="family", multires_reference_root="multi"),
        store=object(),
        manifest={},
        model=object(),
        device=SimpleNamespace(type="cpu"),
        sample_by_resolution={contract.native: object(), contract.fine: object()},
        geometry_by_resolution={
            contract.native: SimpleNamespace(
                nodes=nodes, node_measures=volumes[:, None]
            )
        },
        normalization=SimpleNamespace(
            state_scale=np.ones(4), residual_scale=np.ones(4), gamma=1.4
        ),
        native_projector=projector,
        first_config=SimpleNamespace(
            x_min=0.0,
            x_max=2.0,
            y_min=0.0,
            y_max=1.0,
            shock_x=0.5,
        ),
    )

    rollout = runner._shadow_rollout_arm(
        runtime,
        case_id="synthetic",
        horizon=6,
        native_truth_only=True,
        structural_gate=True,
    )

    assert rollout["position_trusted"]
    assert rollout["front_branch_veto_call"] == 3
    assert len(rollout["front_audits"]) == 2
    assert rollout["execution"]["logical_model_calls"] == 14
    assert rollout["execution"]["candidate_fine_logical_calls"] == 2
    for accepted, shadow in zip(
        rollout["accepted_states"][3:], rollout["shadow_states"][3:], strict=True
    ):
        np.testing.assert_array_equal(accepted, shadow)
    assert all(
        row["correction_status"] == "latched_raw_fallback"
        for row in rollout["accepted_rows"][4:]
    )

    audit_calls = 0
    diagnostic = runner._shadow_rollout_arm(
        runtime,
        case_id="synthetic",
        horizon=6,
        native_truth_only=True,
        structural_gate=True,
        diagnostic_shadow_candidates=True,
    )
    assert diagnostic["policy"] == "shadow_candidate_diagnostic"
    assert diagnostic["front_branch_veto_call"] is None
    assert len(diagnostic["front_audits"]) == 3
    assert len(diagnostic["diagnostic_blocks"]) == 3
    assert diagnostic["execution"]["logical_model_calls"] == 18
    assert diagnostic["execution"]["candidate_fine_logical_calls"] == 3
    assert diagnostic["maxima"]["discard_to_raw_recurrence_abs"] == 0.0
    assert all(
        row["correction_status"].startswith("diagnostic_")
        for row in diagnostic["accepted_rows"]
    )
    assert all(
        set(runner.A9_FEATURE_NAMES).issubset(row)
        and all(f"{name}_status" in row for name in runner.A9_FEATURE_NAMES)
        for row in diagnostic["diagnostic_blocks"]
    )

    def matched_front_audit(*_args, **_kwargs):
        return TargetFreeFrontBranchAudit(
            status="matched",
            branch_changed=False,
            first_shadow_thickness_cells=1,
            first_candidate_thickness_cells=1,
            second_shadow_thickness_cells=1,
            second_candidate_thickness_cells=1,
            maximum_position_shift=0.0,
        )

    monkeypatch.setattr(runner, "position_trusts_cross_resolution", lambda _row: False)
    monkeypatch.setattr(runner, "target_free_front_branch_audit", matched_front_audit)
    warm = runner._shadow_rollout_arm(
        runtime,
        case_id="synthetic",
        horizon=6,
        native_truth_only=True,
        structural_gate=True,
        warm_start_blocks=1,
    )
    assert warm["policy"] == "warm_start_coast"
    assert not warm["position_trusted"]
    assert warm["front_branch_veto_call"] is None
    assert len(warm["front_audits"]) == 1
    assert warm["execution"]["logical_model_calls"] == 14
    assert warm["execution"]["candidate_fine_logical_calls"] == 1
    assert warm["execution"]["accepted_coast_native_logical_calls"] == 4
    assert all(
        row["correction_status"] == "accepted_native_coast"
        for row in warm["accepted_rows"][2:]
    )
    assert any(
        not np.array_equal(accepted, shadow)
        for accepted, shadow in zip(
            warm["accepted_states"][2:], warm["shadow_states"][2:], strict=True
        )
    )

    projected = runner._shadow_rollout_arm(
        runtime,
        case_id="synthetic",
        horizon=6,
        native_truth_only=True,
        structural_gate=True,
        warm_start_blocks=1,
        projected_coast=True,
    )
    assert projected["policy"] == "projected_coast"
    assert projected["projected_coast"]
    assert len(projected["front_audits"]) == 3
    assert len(projected["projected_coast_audits"]) == 4
    assert projected["execution"]["logical_model_calls"] == 14
    assert projected["execution"]["candidate_fine_logical_calls"] == 1
    assert projected["execution"]["accepted_coast_native_logical_calls"] == 4
    assert all(
        row["correction_status"] == "projected_coast_sp19_tether"
        for row in projected["accepted_rows"][2:]
    )
    assert projected["maxima"]["accepted_coast_shadow_integral_abs"] <= 1.0e-14
    assert projected["maxima"]["projected_coast_boundary_abs"] == 0.0
    assert projected["maxima"]["projected_coast_idempotence_abs"] <= 1.0e-14
    assert projected["maxima"]["projected_coast_tether_abs"] <= 1.0e-14

    slew = runner._shadow_rollout_arm(
        runtime,
        case_id="synthetic",
        horizon=6,
        native_truth_only=True,
        structural_gate=True,
        warm_start_blocks=1,
        slew_limited_tether=True,
    )
    assert slew["policy"] == "slew_limited_tether"
    assert slew["slew_limited_tether"]
    assert len(slew["front_audits"]) == 3
    assert len(slew["slew_limited_tether_audits"]) == 4
    assert slew["execution"]["logical_model_calls"] == 14
    assert slew["execution"]["candidate_fine_logical_calls"] == 1
    assert slew["execution"]["accepted_coast_native_logical_calls"] == 4
    assert all(
        row["correction_status"] == "slew_limited_sp19_tether"
        for row in slew["accepted_rows"][2:]
    )
    assert (
        slew["maxima"]["slew_applied_to_shadow_increment_ratio"]
        <= runner.SLEW_LIMITED_TETHER_RELATIVE_CHANGE_LIMIT + 1.0e-12
    )
    assert slew["maxima"]["slew_change_limit_violation_rms"] <= 1.0e-14
    assert slew["maxima"]["slew_boundary_contraction_violation_abs"] <= 1.0e-14
    assert slew["maxima"]["slew_projection_residual_increase_rms"] <= 1.0e-14
    assert slew["maxima"]["slew_update_identity_abs"] <= 1.0e-14
    assert slew["maxima"]["slew_cap_active_count"] >= 0

    relaxed = runner._shadow_rollout_arm(
        runtime,
        case_id="synthetic",
        horizon=6,
        native_truth_only=True,
        structural_gate=True,
        warm_start_blocks=1,
        relaxed_tether=True,
    )
    assert relaxed["policy"] == "relaxed_tether"
    assert relaxed["relaxed_tether"]
    assert len(relaxed["front_audits"]) == 3
    assert len(relaxed["relaxed_tether_audits"]) == 4
    assert relaxed["execution"]["logical_model_calls"] == 14
    assert relaxed["execution"]["candidate_fine_logical_calls"] == 1
    assert relaxed["execution"]["accepted_coast_native_logical_calls"] == 4
    assert all(
        row["correction_status"] == "relaxed_sp19_tether"
        for row in relaxed["accepted_rows"][2:]
    )
    assert all(
        row["status"] == "ok"
        and row["cap_active"]
        and row["applied_scale"] == runner.RELAXED_TETHER_RATE
        and abs(
            row["applied_change_rms"]
            - runner.RELAXED_TETHER_RATE * row["requested_change_rms"]
        )
        <= 1.0e-14
        for row in relaxed["relaxed_tether_audits"]
    )
    assert relaxed["maxima"]["slew_change_limit_violation_rms"] <= 1.0e-14
    assert relaxed["maxima"]["slew_boundary_contraction_violation_abs"] <= 1.0e-14
    assert relaxed["maxima"]["slew_projection_residual_increase_rms"] <= 1.0e-14
    assert relaxed["maxima"]["slew_update_identity_abs"] <= 1.0e-14

    relaxed_full = runner._shadow_rollout_arm(
        runtime,
        case_id="synthetic",
        horizon=30,
        native_truth_only=True,
        structural_gate=True,
        warm_start_blocks=runner.FROZEN_OFFSET_WARM_BLOCKS,
        relaxed_tether=True,
    )
    relaxed_full["position_descriptor"]["normalized_wall_distance"] = (
        runner.FROZEN_POSITION_TRUST_THRESHOLD + 0.1
    )
    # The fake warm block declares a zero anchor but does not actually apply
    # the production integral anchor; isolate the call-check test from that
    # deliberately simplified fixture.
    relaxed_full["maxima"]["slew_integral_abs"] = 0.0
    relaxed_full["maxima"]["accepted_coast_shadow_integral_abs"] = 0.0
    exact_checks = runner._slew_limited_tether_call_checks(
        [relaxed_full], case_ids=("synthetic",), relaxed=True
    )
    assert all(exact_checks.values()), (
        exact_checks,
        relaxed_full["maxima"],
    )
    relaxed_full["relaxed_tether_audits"][0]["applied_scale"] = 0.2
    tampered_checks = runner._slew_limited_tether_call_checks(
        [relaxed_full], case_ids=("synthetic",), relaxed=True
    )
    assert not tampered_checks["relaxed_tether_bound_and_closure_exact"]

    original_position_route = runner.buffered_frozen_offset_position_route
    monkeypatch.setattr(
        runner,
        "buffered_frozen_offset_position_route",
        lambda _descriptor: "raw_uncertainty_buffer",
    )
    buffered_relaxed = runner._shadow_rollout_arm(
        runtime,
        case_id="synthetic",
        horizon=6,
        native_truth_only=True,
        structural_gate=True,
        warm_start_blocks=1,
        relaxed_tether=True,
        buffered_relaxed_tether=True,
    )
    assert buffered_relaxed["policy"] == "buffered_relaxed_tether"
    assert buffered_relaxed["position_route"] == "raw_uncertainty_buffer"
    assert buffered_relaxed["position_buffered"]
    assert buffered_relaxed["execution"]["logical_model_calls"] == 6
    assert buffered_relaxed["execution"]["native_logical_calls"] == 6
    assert buffered_relaxed["execution"]["fine_logical_calls"] == 0
    assert buffered_relaxed["execution"]["candidate_native_logical_calls"] == 0
    assert not buffered_relaxed["front_audits"]
    assert not buffered_relaxed["anchor_rows"]
    assert not buffered_relaxed["relaxed_tether_audits"]
    assert all(
        np.array_equal(accepted, shadow)
        for accepted, shadow in zip(
            buffered_relaxed["accepted_states"],
            buffered_relaxed["shadow_states"],
            strict=True,
        )
    )
    assert all(
        row["correction_status"] == "position_uncertainty_buffer_raw"
        for row in buffered_relaxed["accepted_rows"]
    )
    monkeypatch.setattr(
        runner,
        "buffered_frozen_offset_position_route",
        original_position_route,
    )

    frozen = runner._shadow_rollout_arm(
        runtime,
        case_id="synthetic",
        horizon=6,
        native_truth_only=True,
        structural_gate=True,
        warm_start_blocks=1,
        frozen_offset=True,
    )
    assert frozen["policy"] == "frozen_offset"
    assert frozen["frozen_offset"]
    assert len(frozen["front_audits"]) == 3
    assert len(frozen["frozen_offset_audits"]) == 1
    assert frozen["execution"]["logical_model_calls"] == 10
    assert frozen["execution"]["candidate_fine_logical_calls"] == 1
    assert frozen["execution"]["accepted_coast_native_logical_calls"] == 0
    assert all(
        row["correction_status"] == "frozen_sp19_output_offset"
        for row in frozen["accepted_rows"][2:]
    )
    offset = frozen["accepted_states"][2] - frozen["shadow_states"][2]
    for call in range(2, 7):
        np.testing.assert_allclose(
            frozen["accepted_states"][call] - frozen["shadow_states"][call],
            offset,
            atol=1.0e-14,
            rtol=0.0,
        )
    for call in range(3, 7):
        np.testing.assert_allclose(
            frozen["accepted_states"][call] - frozen["accepted_states"][call - 1],
            frozen["shadow_states"][call] - frozen["shadow_states"][call - 1],
            atol=1.0e-14,
            rtol=0.0,
        )
    assert frozen["maxima"]["frozen_offset_integral_abs"] <= 1.0e-14
    assert frozen["maxima"]["frozen_offset_boundary_abs"] == 0.0
    assert frozen["maxima"]["frozen_offset_idempotence_abs"] <= 1.0e-14
    assert frozen["maxima"]["frozen_offset_constancy_abs"] <= 1.0e-14
    assert frozen["maxima"]["frozen_offset_increment_identity_abs"] <= 1.0e-14

    def fake_probe(native_state, native_prediction, *, fine_predictor, contract, **_kwargs):
        fine_input = prolong_nested_state(
            native_state,
            coarse_resolution=contract.native,
            fine_resolution=contract.fine,
        )
        fine_prediction = fine_predictor(contract.fine, fine_input)
        return SimpleNamespace(
            correction=0.5 * offset,
            native_model_input=np.asarray(native_state, dtype=np.float32),
            fine_model_input=fine_input,
            fine_prediction=fine_prediction,
            audit=FineDiscrepancyAudit(
                policy="rank8_fine_away_half",
                status="ok",
                raw_gain=-0.5,
                applied_scale=1.0,
                native_increment_rms=1.0,
                raw_correction_rms=1.0,
                correction_rms=1.0,
                correction_to_native_increment=1.0,
                cap_active=False,
                constant_mode_energy_fraction=0.0,
                maximum_scaled_component_mean_abs=0.0,
                maximum_excluded_abs=0.0,
                maximum_modal_reconstruction_abs=0.0,
            ),
        )

    monkeypatch.setattr(runner, "synchronized_persistence_probe", fake_probe)
    persistence = runner._shadow_rollout_arm(
        runtime,
        case_id="synthetic",
        horizon=6,
        native_truth_only=True,
        structural_gate=True,
        warm_start_blocks=1,
        frozen_offset=True,
        buffered_offset=True,
        persistence_probe=True,
    )
    assert persistence["policy"] == "persistence_gain"
    assert len(persistence["persistence_probe_audits"]) == 2
    assert len(persistence["persistence_error_structure"]) == 2
    assert persistence["execution"]["logical_model_calls"] == 12
    assert persistence["execution"]["persistence_probe_fine_logical_calls"] == 2
    assert [row["applied_gain"] for row in persistence["persistence_probe_audits"]] == [
        1.0,
        1.0,
    ]
    assert persistence["maxima"]["persistence_offset_bookkeeping_abs"] <= 1.0e-14
    assert persistence["maxima"]["persistence_within_block_identity_abs"] <= 1.0e-14
    assert all(
        row["correction_status"] == "persistence_gated_sp19_output_offset"
        for row in persistence["accepted_rows"][2:]
    )
    assert all(
        row["probe_needed_cosine_status"]
        in {"ok", "unresolved_small_denominator"}
        and row["handoff_offset_oracle_coefficient_status"] == "ok"
        for row in persistence["persistence_error_structure"]
    )

    probe_calls = 0

    def phase_probe(
        native_state, native_prediction, *, fine_predictor, contract, **_kwargs
    ):
        nonlocal probe_calls
        probe_calls += 1
        fine_input = prolong_nested_state(
            native_state,
            coarse_resolution=contract.native,
            fine_resolution=contract.fine,
        )
        fine_prediction = fine_predictor(contract.fine, fine_input)
        correction = (0.5 if probe_calls == 1 else -0.5) * offset
        return SimpleNamespace(
            correction=correction,
            native_model_input=np.asarray(native_state, dtype=np.float32),
            fine_model_input=fine_input,
            fine_prediction=fine_prediction,
            audit=FineDiscrepancyAudit(
                policy="rank8_fine_away_half",
                status="ok",
                raw_gain=-0.5,
                applied_scale=1.0,
                native_increment_rms=1.0,
                raw_correction_rms=1.0,
                correction_rms=1.0,
                correction_to_native_increment=1.0,
                cap_active=False,
                constant_mode_energy_fraction=0.0,
                maximum_scaled_component_mean_abs=0.0,
                maximum_excluded_abs=0.0,
                maximum_modal_reconstruction_abs=0.0,
            ),
        )

    monkeypatch.setattr(runner, "synchronized_persistence_probe", phase_probe)
    terminal_ramp = runner._shadow_rollout_arm(
        runtime,
        case_id="synthetic",
        horizon=6,
        native_truth_only=True,
        structural_gate=True,
        warm_start_blocks=1,
        frozen_offset=True,
        buffered_offset=True,
        persistence_probe=True,
        terminal_ramp=True,
    )
    assert terminal_ramp["policy"] == "terminal_ramp"
    assert terminal_ramp["terminal_ramp_trigger_block"] == 2
    assert terminal_ramp["execution"]["logical_model_calls"] == 12
    assert [
        row["applied_gain"] for row in terminal_ramp["persistence_probe_audits"]
    ] == [1.0, 0.0]
    assert terminal_ramp["persistence_probe_audits"][-1]["gain_status"] == (
        "phase_triggered_terminal_ramp"
    )
    np.testing.assert_array_equal(
        terminal_ramp["accepted_states"][-1], terminal_ramp["shadow_states"][-1]
    )
    assert terminal_ramp["maxima"]["persistence_offset_bookkeeping_abs"] <= 1.0e-14

    fixed_late_ramp = runner._shadow_rollout_arm(
        runtime,
        case_id="synthetic",
        horizon=6,
        native_truth_only=True,
        structural_gate=True,
        warm_start_blocks=1,
        frozen_offset=True,
        buffered_offset=True,
        fixed_late_ramp=True,
        fixed_late_ramp_start_block=1,
    )
    assert fixed_late_ramp["policy"] == "fixed_late_ramp"
    assert fixed_late_ramp["execution"]["logical_model_calls"] == 10
    assert fixed_late_ramp["execution"]["persistence_probe_fine_logical_calls"] == 0
    assert [
        row["applied_gain"] for row in fixed_late_ramp["persistence_probe_audits"]
    ] == [1.0, 0.0]
    np.testing.assert_array_equal(
        fixed_late_ramp["accepted_states"][-1],
        fixed_late_ramp["shadow_states"][-1],
    )
    assert fixed_late_ramp["maxima"]["persistence_offset_bookkeeping_abs"] <= 1.0e-14
    assert fixed_late_ramp["maxima"]["persistence_within_block_identity_abs"] <= (
        1.0e-14
    )

    monkeypatch.setattr(
        runner,
        "buffered_frozen_offset_position_route",
        lambda _descriptor: "raw_uncertainty_buffer",
    )
    buffered = runner._shadow_rollout_arm(
        runtime,
        case_id="synthetic",
        horizon=6,
        native_truth_only=True,
        structural_gate=True,
        warm_start_blocks=1,
        frozen_offset=True,
        buffered_offset=True,
    )
    assert buffered["policy"] == "buffered_offset"
    assert buffered["position_buffered"]
    assert buffered["position_route"] == "raw_uncertainty_buffer"
    assert buffered["execution"]["logical_model_calls"] == 6
    assert buffered["execution"]["fine_logical_calls"] == 0
    assert buffered["front_audits"] == []
    assert buffered["frozen_offset_audits"] == []
    for accepted, shadow in zip(
        buffered["accepted_states"], buffered["shadow_states"], strict=True
    ):
        np.testing.assert_array_equal(accepted, shadow)
    assert all(
        row["correction_status"] == "position_uncertainty_buffer_raw"
        for row in buffered["accepted_rows"]
    )


def test_shadow_candidate_associations_are_grouped_and_exact_for_linear_signal():
    rows = []
    for case_index, case_id in enumerate(("case_a", "case_b", "case_c")):
        for block_index in range(4):
            feature = 1.0 + case_index + 0.25 * block_index
            rows.append(
                {
                    "case_id": case_id,
                    "block_index": block_index,
                    "first_correction_rms": feature,
                    "block_state_sse_skill": 2.0 * feature,
                }
            )

    associations, case_rows, oof_rows = runner._shadow_candidate_associations(rows)
    selected = next(
        row
        for row in associations
        if row["feature"] == "first_correction_rms"
        and row["target"] == "block_state_sse_skill"
    )
    assert selected["pearson"] == pytest.approx(1.0)
    assert selected["spearman"] == pytest.approx(1.0)
    assert selected["valid_case_spearman_count"] == 3
    assert selected["case_spearman_same_population_sign_count"] == 3
    assert selected["cv_r2_vs_zero"] == pytest.approx(1.0)
    assert selected["cv_prediction_count"] == 12
    assert (
        len(
            [
                row
                for row in case_rows
                if row["feature"] == "first_correction_rms"
                and row["target"] == "block_state_sse_skill"
            ]
        )
        == 3
    )
    assert (
        len(
            [
                row
                for row in oof_rows
                if row["feature"] == "first_correction_rms"
                and row["target"] == "block_state_sse_skill"
            ]
        )
        == 12
    )


def test_shadow_candidate_denominators_fail_closed():
    relation = runner._diagnostic_ratio(0.0, runner.A9_DENOMINATOR_FLOOR)
    assert relation == {
        "value": None,
        "status": "unresolved_small_denominator",
    }
    cosine = runner._weighted_scaled_cosine(
        np.zeros((2, 1)),
        np.ones((2, 1)),
        volumes=np.ones(2),
        component_scale=np.ones(1),
    )
    assert cosine["value"] is None
    assert cosine["status"] == "unresolved_small_denominator"


def test_shadow_candidate_rescore_repairs_only_the_e12_inventory_default():
    execution_rows = []
    closure_rows = []
    for case_id in runner.PROSPECTIVE_STRUCTURAL_CASE_IDS:
        execution_rows.append(
            {
                "case_id": case_id,
                "logical_model_calls": 90,
                "native_logical_calls": 75,
                "fine_logical_calls": 15,
                "shadow_native_logical_calls": 30,
                "candidate_native_logical_calls": 45,
                "candidate_fine_logical_calls": 15,
                "completed_calls": 30,
            }
        )
        closure_rows.append(
            {
                "case_id": case_id,
                "call_order_errors": 0,
                "shadow_recurrence_abs": 0.0,
                "accepted_recurrence_abs": 2.0e-7,
                "candidate_common_native_input_abs": 0.0,
                "candidate_lookahead_input_abs": 1.0e-20,
            }
        )
    prior = {"unrelated_scientific_check": True}
    prior.update({key: False for key in runner.A9_MISWIRED_CALL_CHECKS})

    corrected = runner._rescore_shadow_candidate_call_checks(
        prior, execution_rows, closure_rows
    )

    assert all(corrected.values())
    assert corrected["unrelated_scientific_check"]
    bad = dict(prior)
    bad["unrelated_scientific_check"] = False
    with pytest.raises(ValueError, match="exact call-inventory failure"):
        runner._rescore_shadow_candidate_call_checks(bad, execution_rows, closure_rows)


def _a22_position_decisions():
    rows = []
    distances = (0.70, 0.775, 0.85, 0.925, 1.0, 0.925, 0.85, 0.775, 0.70)
    for case_id, distance in zip(
        runner.PROSPECTIVE_STRUCTURAL_CASE_IDS, distances, strict=True
    ):
        rows.append(
            {
                "case_id": case_id,
                "status": "ok",
                "vertical_centroid": 0.5 * distance,
                "normalized_wall_distance": distance,
                "transverse_velocity_l1": 0.02,
            }
        )
    return rows


def _a22_metric_rows():
    rows = []
    for case_index, case_id in enumerate(runner.PROSPECTIVE_STRUCTURAL_CASE_IDS):
        for input_call in runner.parent.ALL_INPUT_CALLS:
            for policy in ("zero", "relaxed_tether"):
                value = 1.0 if policy == "zero" else 0.99 - 0.001 * case_index
                rows.append(
                    {
                        "case_id": case_id,
                        "input_call": input_call,
                        "policy": policy,
                        "state_error": value,
                    }
                )
    return rows


def test_a22_routes_are_frozen_two_edge_two_buffer_five_deep():
    routes = runner._buffered_relaxed_tether_routes(_a22_position_decisions())
    assert [routes[case_id] for case_id in runner.PROSPECTIVE_STRUCTURAL_CASE_IDS] == [
        "edge_candidate",
        "raw_uncertainty_buffer",
        "interior_frozen_offset",
        "interior_frozen_offset",
        "interior_frozen_offset",
        "interior_frozen_offset",
        "interior_frozen_offset",
        "raw_uncertainty_buffer",
        "edge_candidate",
    ]
    assert runner._buffered_relaxed_tether_call_budget(routes) == {
        "logical_model_calls": 580,
        "native_logical_calls": 530,
        "fine_logical_calls": 50,
        "raw_native_logical_call_comparator": 270,
        "a19_relaxed_tether_logical_call_comparator": 656,
        "deterministic_prefix_logical_model_calls": 12,
        "logical_model_calls_including_deterministic_prefix": 592,
    }


def test_a22_composition_selects_raw_only_in_the_frozen_buffer():
    routes = runner._buffered_relaxed_tether_routes(_a22_position_decisions())
    selected = runner._compose_buffered_relaxed_tether_rows(
        _a22_metric_rows(), routes=routes
    )
    candidates = [row for row in selected if row["policy"] != "zero"]
    assert len(candidates) == 9 * len(runner.parent.ALL_INPUT_CALLS)
    for row in candidates:
        if row["case_id"] in {"sv_e12_y01", "sv_e12_y07"}:
            assert row["source_policy"] == "zero"
            assert row["state_error"] == 1.0
        else:
            assert row["source_policy"] == "relaxed_tether"
            assert row["state_error"] < 1.0


@pytest.mark.parametrize(
    "mutator,match",
    [
        (
            lambda rows: rows + [dict(rows[0])],
            "row count mismatch",
        ),
        (
            lambda rows: [
                {**row, "case_id": "sv_e12_y00"} if index == 1 else row
                for index, row in enumerate(rows)
            ],
            "unique case identifiers",
        ),
        (
            lambda rows: [
                {**row, "normalized_wall_distance": 0.70}
                if row["case_id"] == "sv_e12_y01"
                else row
                for row in rows
            ],
            "route inventory mismatch",
        ),
    ],
)
def test_a22_routes_fail_closed_on_inventory_changes(mutator, match):
    with pytest.raises(ValueError, match=match):
        runner._buffered_relaxed_tether_routes(mutator(_a22_position_decisions()))


def test_a22_composition_fails_closed_on_duplicate_or_missing_rows():
    routes = runner._buffered_relaxed_tether_routes(_a22_position_decisions())
    rows = _a22_metric_rows()
    with pytest.raises(ValueError, match="duplicate"):
        runner._compose_buffered_relaxed_tether_rows(
            [*rows, dict(rows[0])], routes=routes
        )
    with pytest.raises(ValueError, match="inventory mismatch"):
        runner._compose_buffered_relaxed_tether_rows(rows[:-1], routes=routes)


def test_a22_replay_scoring_checks_close_at_1e_12_and_fail_above(tmp_path):
    case_rows = [
        {
            "case_id": "case",
            "trajectory_state_rms_ratio": 0.9,
            "endpoint_state_ratio": 0.95,
        }
    ]
    controls = [
        {
            "key": "trajectory::rank8_state_error",
            "scope": "population",
            "zero_rms": 2.0,
            "corrected_rms": 1.8,
            "ratio": 0.9,
            "status": "ok",
        }
    ]
    rows = [
        {
            "case_id": "case",
            "policy": "zero",
            "input_call": 0,
            "output_call": 1,
            "state_error": 1.0,
            "finite": True,
            "admissible": True,
            "cap_active": False,
            "correction_status": "raw_shadow",
            "source_policy": None,
            "position_route": None,
        }
    ]
    runner.write_csv(tmp_path / "selected_call_metrics.csv", rows)
    runner.write_csv(tmp_path / "rollout_case_metrics.csv", case_rows)
    runner.write_csv(tmp_path / "rollout_controls.csv", controls)
    result = {
        "population": {
            "aggregate_state_rms_ratio": 0.9,
            "strict_deep_interior_trajectory_win_count": 1,
        },
        "position_routes": [
            {"case_id": "case", "route": "interior_frozen_offset"}
        ],
    }
    descriptor_rows = [
        {"case_id": "case", "position_route": "interior_frozen_offset"}
    ]
    checks = runner._a22_replay_scoring_checks(
        a22_rescore_path=tmp_path / "buffered_relaxed_tether_rescore.json",
        a22_rescore=result,
        rows=rows,
        case_rows=case_rows,
        controls=controls,
        population={"aggregate_state_rms_ratio": 0.9},
        descriptor_rows=descriptor_rows,
    )
    assert all(checks.values())
    tampered = [dict(rows[0], state_error=1.0 + 1.1e-12)]
    checks = runner._a22_replay_scoring_checks(
        a22_rescore_path=tmp_path / "buffered_relaxed_tether_rescore.json",
        a22_rescore=result,
        rows=tampered,
        case_rows=case_rows,
        controls=controls,
        population={"aggregate_state_rms_ratio": 0.9},
        descriptor_rows=descriptor_rows,
    )
    assert not checks["a22_selected_call_metrics_reproduced_1e_12"]


def test_animation_bundles_are_visualization_only_and_use_frozen_frames(
    tmp_path, monkeypatch
):
    nx, ny = 4, 2
    calls = (0, 2, 4, 30)
    contract = {
        **runner.STRENGTH_OOD_ANIMATION_CONTRACT,
        "native_resolution": [nx, ny],
        "output_calls": list(calls),
        "physical_times": [0.02 * call for call in calls],
    }
    monkeypatch.setattr(runner, "NATIVE_RESOLUTION", (nx, ny))
    monkeypatch.setattr(runner, "STRENGTH_OOD_ANIMATION_CALLS", calls)
    monkeypatch.setattr(runner, "STRENGTH_OOD_ANIMATION_CONTRACT", contract)

    nodes = np.column_stack(
        np.meshgrid(
            (np.arange(nx) + 0.5) * (2.0 / nx),
            (np.arange(ny) + 0.5) * (1.0 / ny),
            indexing="xy",
        )
    ).reshape(-1, 2)
    runtime = SimpleNamespace(
        geometry_by_resolution={
            (nx, ny): SimpleNamespace(
                nodes=nodes,
                node_measures=np.full((nx * ny, 1), 2.0 / (nx * ny)),
            )
        },
        normalization=SimpleNamespace(
            state_scale=np.ones(4), residual_scale=np.ones(4)
        ),
    )
    base = np.zeros((nx * ny, 4), dtype=np.float64)
    base[:, 0] = 1.0
    base[:, 1] = 1.0
    base[:, 3] = 3.0
    rollouts = []
    for case_index, case_id in enumerate(runner.STRENGTH_OOD_ANIMATION_CASE_IDS):
        truth = [base + 0.001 * frame for frame in range(31)]
        raw = [value + 0.02 for value in truth]
        corrected = [value + 0.01 for value in truth]
        rollouts.append(
            {
                "case_id": case_id,
                "reference_states": truth,
                "shadow_states": raw,
                "accepted_states": corrected,
            }
        )
    manifest = runner._write_strength_ood_animation_bundles(
        tmp_path / "bundles", rollouts=rollouts, runtime=runtime
    )
    assert manifest["inference_or_gate_input"] is False
    assert len(manifest["bundles"]) == 3
    first = manifest["bundles"][0]
    with np.load(tmp_path / "bundles" / first["path"], allow_pickle=False) as data:
        assert data["truth_conservative"].dtype == np.float32
        assert data["truth_conservative"].shape == (len(calls), nx * ny, 4)
        np.testing.assert_array_equal(data["output_calls"], calls)


def test_animation_loader_accepts_registered_prospective_e12_contract(tmp_path):
    manifest = runner.with_payload_sha256(
        {
            "schema": "pcno_response_filtered_block_animation_bundles_v1",
            "working_id": runner.PROSPECTIVE_STRUCTURAL_WORKING_ID,
            "contract": runner.PROSPECTIVE_STRUCTURAL_ANIMATION_CONTRACT,
            "bundles": [],
            "inference_or_gate_input": False,
        }
    )
    manifest_path = tmp_path / "bundle_manifest.json"
    runner.atomic_write_json(manifest_path, manifest)
    rollout = runner.with_payload_sha256(
        {
            "schema": runner.PROSPECTIVE_STRUCTURAL_SCHEMA,
            "working_id": runner.PROSPECTIVE_STRUCTURAL_WORKING_ID,
            "recurrence_executed": True,
            "animation_bundle_manifest": manifest,
            "animation_bundle_manifest_sha256": runner.sha256_file(manifest_path),
        }
    )
    rollout_path = tmp_path / "prospective_structural_rollout.json"
    runner.atomic_write_json(rollout_path, rollout)

    loaded_rollout, loaded_manifest = visualizer._load_contract(
        rollout_path, manifest_path
    )
    assert loaded_rollout["working_id"] == runner.PROSPECTIVE_STRUCTURAL_WORKING_ID
    assert loaded_manifest["contract"]["case_ids"] == list(
        runner.PROSPECTIVE_STRUCTURAL_ANIMATION_CASE_IDS
    )

    nx, ny = runner.PROSPECTIVE_STRUCTURAL_ANIMATION_CONTRACT["native_resolution"]
    frame_count = len(runner.PROSPECTIVE_STRUCTURAL_ANIMATION_CONTRACT["output_calls"])
    bundle_path = tmp_path / "sv_e12_y00_shadow_anchored_h30.npz"
    np.savez_compressed(
        bundle_path,
        case_id=np.asarray("sv_e12_y00"),
        split_group_id=np.asarray("strength_ood_e12"),
        output_calls=np.asarray(
            runner.PROSPECTIVE_STRUCTURAL_ANIMATION_CONTRACT["output_calls"]
        ),
        physical_times=np.asarray(
            runner.PROSPECTIVE_STRUCTURAL_ANIMATION_CONTRACT["physical_times"]
        ),
        native_resolution=np.asarray([nx, ny]),
        nodes=np.zeros((nx * ny, 2), dtype=np.float32),
        volumes=np.ones(nx * ny, dtype=np.float32),
        truth_conservative=np.zeros((frame_count, nx * ny, 4), dtype=np.float32),
        raw_shadow_conservative=np.zeros((frame_count, nx * ny, 4), dtype=np.float32),
        corrected_conservative=np.zeros((frame_count, nx * ny, 4), dtype=np.float32),
    )
    loaded_bundle = visualizer._load_bundle(
        bundle_path,
        {"case_id": "sv_e12_y00", "sha256": runner.sha256_file(bundle_path)},
        runner.PROSPECTIVE_STRUCTURAL_ANIMATION_CONTRACT,
    )
    assert loaded_bundle["split_group_id"].item() == "strength_ood_e12"


def test_animation_relative_improvement_is_signed_and_floor_safe():
    raw = np.array([2.0e-2, 1.0e-8, 1.0e-2])
    corrected = np.array([1.0e-2, 2.0e-8, 2.0e-2])
    improvement, resolved = visualizer._relative_improvement(
        raw, corrected, floor=1.0e-5
    )
    np.testing.assert_allclose(improvement, [0.5, 0.0, -0.5])
    np.testing.assert_array_equal(resolved, [True, False, True])


def test_animation_loader_accepts_registered_warm_start_contract(tmp_path):
    manifest = runner.with_payload_sha256(
        {
            "schema": "pcno_response_filtered_block_animation_bundles_v1",
            "working_id": runner.WARM_START_WORKING_ID,
            "contract": runner.WARM_START_ANIMATION_CONTRACT,
            "inference_or_gate_input": False,
            "bundles": [],
        }
    )
    manifest_path = tmp_path / "bundle_manifest.json"
    runner.atomic_write_json(manifest_path, manifest)
    rollout = runner.with_payload_sha256(
        {
            "schema": runner.WARM_START_SCHEMA,
            "working_id": runner.WARM_START_WORKING_ID,
            "recurrence_executed": True,
            "animation_bundle_manifest": manifest,
            "animation_bundle_manifest_sha256": runner.sha256_file(manifest_path),
        }
    )
    rollout_path = tmp_path / "warm_start_rollout.json"
    runner.atomic_write_json(rollout_path, rollout)

    loaded_rollout, loaded_manifest = visualizer._load_contract(
        rollout_path, manifest_path
    )
    assert loaded_rollout["working_id"] == runner.WARM_START_WORKING_ID
    assert loaded_manifest["contract"] == runner.WARM_START_ANIMATION_CONTRACT


def test_animation_loader_accepts_buffered_relaxed_replay_contract(tmp_path):
    manifest = runner.with_payload_sha256(
        {
            "schema": "pcno_response_filtered_block_animation_bundles_v1",
            "working_id": runner.BUFFERED_RELAXED_TETHER_REPLAY_WORKING_ID,
            "contract": runner.BUFFERED_RELAXED_TETHER_REPLAY_ANIMATION_CONTRACT,
            "inference_or_gate_input": False,
            "bundles": [],
        }
    )
    manifest_path = tmp_path / "bundle_manifest.json"
    runner.atomic_write_json(manifest_path, manifest)
    rollout = runner.with_payload_sha256(
        {
            "schema": runner.BUFFERED_RELAXED_TETHER_REPLAY_SCHEMA,
            "working_id": runner.BUFFERED_RELAXED_TETHER_REPLAY_WORKING_ID,
            "recurrence_executed": True,
            "animation_bundle_manifest": manifest,
            "animation_bundle_manifest_sha256": runner.sha256_file(manifest_path),
        }
    )
    rollout_path = tmp_path / "buffered_relaxed_tether_replay.json"
    runner.atomic_write_json(rollout_path, rollout)

    loaded_rollout, loaded_manifest = visualizer._load_contract(
        rollout_path, manifest_path
    )
    assert (
        loaded_rollout["working_id"]
        == runner.BUFFERED_RELAXED_TETHER_REPLAY_WORKING_ID
    )
    assert (
        loaded_manifest["contract"]
        == runner.BUFFERED_RELAXED_TETHER_REPLAY_ANIMATION_CONTRACT
    )


def test_animation_loader_accepts_buffered_relaxed_e14_contract(tmp_path):
    manifest = runner.with_payload_sha256(
        {
            "schema": "pcno_response_filtered_block_animation_bundles_v1",
            "working_id": runner.BUFFERED_RELAXED_TETHER_E14_WORKING_ID,
            "contract": runner.BUFFERED_RELAXED_TETHER_E14_ANIMATION_CONTRACT,
            "inference_or_gate_input": False,
            "bundles": [],
        }
    )
    manifest_path = tmp_path / "bundle_manifest.json"
    runner.atomic_write_json(manifest_path, manifest)
    rollout = runner.with_payload_sha256(
        {
            "schema": runner.BUFFERED_RELAXED_TETHER_E14_SCHEMA,
            "working_id": runner.BUFFERED_RELAXED_TETHER_E14_WORKING_ID,
            "recurrence_executed": True,
            "animation_bundle_manifest": manifest,
            "animation_bundle_manifest_sha256": runner.sha256_file(manifest_path),
        }
    )
    rollout_path = tmp_path / "buffered_relaxed_tether_e14_rollout.json"
    runner.atomic_write_json(rollout_path, rollout)

    loaded_rollout, loaded_manifest = visualizer._load_contract(
        rollout_path, manifest_path
    )
    assert (
        loaded_rollout["working_id"]
        == runner.BUFFERED_RELAXED_TETHER_E14_WORKING_ID
    )
    assert (
        loaded_manifest["contract"]
        == runner.BUFFERED_RELAXED_TETHER_E14_ANIMATION_CONTRACT
    )


def test_animation_loader_accepts_registered_projected_coast_contract(tmp_path):
    manifest = runner.with_payload_sha256(
        {
            "schema": "pcno_response_filtered_block_animation_bundles_v1",
            "working_id": runner.PROJECTED_COAST_WORKING_ID,
            "contract": runner.PROJECTED_COAST_ANIMATION_CONTRACT,
            "inference_or_gate_input": False,
            "bundles": [],
        }
    )
    manifest_path = tmp_path / "bundle_manifest.json"
    runner.atomic_write_json(manifest_path, manifest)
    rollout = runner.with_payload_sha256(
        {
            "schema": runner.PROJECTED_COAST_SCHEMA,
            "working_id": runner.PROJECTED_COAST_WORKING_ID,
            "recurrence_executed": True,
            "animation_bundle_manifest": manifest,
            "animation_bundle_manifest_sha256": runner.sha256_file(manifest_path),
        }
    )
    rollout_path = tmp_path / "projected_coast_rollout.json"
    runner.atomic_write_json(rollout_path, rollout)

    loaded_rollout, loaded_manifest = visualizer._load_contract(
        rollout_path, manifest_path
    )
    assert loaded_rollout["working_id"] == runner.PROJECTED_COAST_WORKING_ID
    assert loaded_manifest["contract"] == runner.PROJECTED_COAST_ANIMATION_CONTRACT


def test_animation_loader_accepts_registered_frozen_offset_contract(tmp_path):
    manifest = runner.with_payload_sha256(
        {
            "schema": "pcno_response_filtered_block_animation_bundles_v1",
            "working_id": runner.FROZEN_OFFSET_WORKING_ID,
            "contract": runner.FROZEN_OFFSET_ANIMATION_CONTRACT,
            "inference_or_gate_input": False,
            "bundles": [],
        }
    )
    manifest_path = tmp_path / "bundle_manifest.json"
    runner.atomic_write_json(manifest_path, manifest)
    rollout = runner.with_payload_sha256(
        {
            "schema": runner.FROZEN_OFFSET_SCHEMA,
            "working_id": runner.FROZEN_OFFSET_WORKING_ID,
            "recurrence_executed": True,
            "animation_bundle_manifest": manifest,
            "animation_bundle_manifest_sha256": runner.sha256_file(manifest_path),
        }
    )
    rollout_path = tmp_path / "frozen_offset_rollout.json"
    runner.atomic_write_json(rollout_path, rollout)

    loaded_rollout, loaded_manifest = visualizer._load_contract(
        rollout_path, manifest_path
    )
    assert loaded_rollout["working_id"] == runner.FROZEN_OFFSET_WORKING_ID
    assert loaded_manifest["contract"] == runner.FROZEN_OFFSET_ANIMATION_CONTRACT


def test_animation_loader_accepts_prospective_buffered_offset_contract(tmp_path):
    manifest = runner.with_payload_sha256(
        {
            "schema": "pcno_response_filtered_block_animation_bundles_v1",
            "working_id": runner.PROSPECTIVE_BUFFERED_OFFSET_WORKING_ID,
            "contract": runner.PROSPECTIVE_BUFFERED_OFFSET_ANIMATION_CONTRACT,
            "inference_or_gate_input": False,
            "bundles": [],
        }
    )
    manifest_path = tmp_path / "bundle_manifest.json"
    runner.atomic_write_json(manifest_path, manifest)
    rollout = runner.with_payload_sha256(
        {
            "schema": runner.PROSPECTIVE_BUFFERED_OFFSET_SCHEMA,
            "working_id": runner.PROSPECTIVE_BUFFERED_OFFSET_WORKING_ID,
            "recurrence_executed": True,
            "animation_bundle_manifest": manifest,
            "animation_bundle_manifest_sha256": runner.sha256_file(manifest_path),
        }
    )
    rollout_path = tmp_path / "prospective_buffered_offset_rollout.json"
    runner.atomic_write_json(rollout_path, rollout)

    loaded_rollout, loaded_manifest = visualizer._load_contract(
        rollout_path, manifest_path
    )
    assert loaded_rollout["working_id"] == runner.PROSPECTIVE_BUFFERED_OFFSET_WORKING_ID
    assert (
        loaded_manifest["contract"]
        == runner.PROSPECTIVE_BUFFERED_OFFSET_ANIMATION_CONTRACT
    )


def test_animation_renderer_writes_fixed_scale_gif_and_final_frame(tmp_path):
    nx, ny = 4, 2
    truth = np.zeros((2, nx * ny, 4), dtype=np.float32)
    truth[..., 0] = 1.0
    truth[..., 1] = 1.0
    truth[..., 3] = 3.0
    raw = truth.copy()
    corrected = truth.copy()
    raw[..., 0] += 0.02
    corrected[..., 0] += 0.01
    bundle = {
        "case_id": np.asarray("sv_e13_y04"),
        "physical_times": np.asarray([0.0, 0.6]),
        "native_resolution": np.asarray([nx, ny]),
        "gamma": np.asarray(1.4),
        "truth_conservative": truth,
        "raw_shadow_conservative": raw,
        "corrected_conservative": corrected,
    }
    paths, saturation = visualizer._render_one(
        bundle,
        field="density",
        contract=runner.STRENGTH_OOD_ANIMATION_CONTRACT,
        output_dir=tmp_path,
    )
    assert [path.suffix for path in paths] == [".gif", ".png"]
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths)
    assert saturation["field_below_fraction"] == 0.0
    assert saturation["field_above_fraction"] == 0.0


def test_strength_ood_rollout_requests_only_checkpoint_bound_native_truth(monkeypatch):
    observed = []

    def stop_after_reference_contract(
        family_root,
        multires_reference_root,
        store,
        manifest,
        case_id,
        *,
        training_resolution,
    ):
        observed.append(multires_reference_root)
        raise RuntimeError("reference-contract-observed")

    monkeypatch.setattr(
        runner, "load_resolution_reference", stop_after_reference_contract
    )
    runtime = SimpleNamespace(
        args=SimpleNamespace(
            family_root="family", multires_reference_root="unused-multires"
        ),
        store=object(),
        manifest={},
    )
    with pytest.raises(RuntimeError, match="reference-contract-observed"):
        runner._shadow_rollout_arm(
            runtime,
            case_id=runner.STRENGTH_OOD_CASE_IDS[0],
            horizon=2,
            native_truth_only=True,
        )
    assert observed == [None]
