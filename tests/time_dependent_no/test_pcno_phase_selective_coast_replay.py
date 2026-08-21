from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.time_dependent_no import (
    evaluate_pcno_phase_selective_coast_replay as evaluator,
)
from scripts.time_dependent_no import (
    visualize_pcno_phase_selective_coast_replay as visualizer,
)
from utility.time_dependent_no.pcno_cross_resolution_teacher_forced import (
    build_fixed_cosine_projector,
)


def _grid() -> tuple[np.ndarray, object]:
    nx, ny = (8, 4)
    x = (np.arange(nx, dtype=np.float64) + 0.5) * (2.0 / nx)
    y = (np.arange(ny, dtype=np.float64) + 0.5) * (1.0 / ny)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    nodes = np.column_stack((xx.reshape(-1), yy.reshape(-1)))
    volumes = np.linspace(0.8, 1.2, nx * ny, dtype=np.float64)
    volumes *= 2.0 / volumes.sum()
    node_type = np.zeros(nx * ny, dtype=np.int64)
    node_type[:nx] = 1
    return volumes, build_fixed_cosine_projector(nodes, volumes, node_type, rank=8)


def test_synthetic_contract_freezes_route_inventory_cost_and_source_scope() -> None:
    summary = evaluator.synthetic_summary()

    assert summary["status"] == "passed"
    assert all(summary["checks"].values())
    assert evaluator.COAST_CALLS == tuple(range(8, 22))
    assert evaluator.CANDIDATE_CALLS == 548
    assert evaluator.OPTIMIZED_A32_CALLS == 604
    assert evaluator.OWNED_SOURCE_PATHS == (
        "docs/time_dependent_no/W26_L5_RESPONSE_FILTERED_BLOCK_PREREGISTRATION.md",
        "scripts/time_dependent_no/evaluate_pcno_phase_selective_coast_replay.py",
        "scripts/time_dependent_no/visualize_pcno_phase_selective_coast_replay.py",
        "tests/time_dependent_no/test_pcno_phase_selective_coast_replay.py",
    )


def test_coast_step_uses_one_shadow_call_and_frozen_response_map() -> None:
    volumes, projector = _grid()
    state_scale = np.asarray((0.5, 0.75, 1.0, 1.25), dtype=np.float64)
    runtime_view = SimpleNamespace(
        native_projector=projector,
        normalization=SimpleNamespace(state_scale=state_scale),
    )
    coordinates = np.linspace(
        -0.02, 0.03, len(evaluator.FROZEN_DIAGONAL), dtype=np.float64
    )
    shadow = np.zeros((volumes.size, 4), dtype=np.float64)
    accepted = shadow + evaluator.a42.a41._field_from_active(coordinates, runtime_view)
    calls: list[tuple[int, int]] = []

    def predictor(resolution, value):
        calls.append(resolution)
        return np.asarray(value, dtype=np.float64) + 0.25

    step = evaluator.phase_selective_active_step(
        accepted,
        shadow,
        input_call=8,
        projector=projector,
        predictor=predictor,
        start_coefficients=np.zeros((19, 19)),
        end_coefficients=np.zeros((19, 19)),
        volumes=volumes,
        residual_scale=state_scale,
        state_scale=state_scale,
    )
    expected = evaluator.a42.a41._field_from_active(
        np.asarray(evaluator.FROZEN_DIAGONAL) * coordinates,
        runtime_view,
    )

    assert calls == [evaluator.NATIVE_RESOLUTION]
    assert step.route == "surrogate_coast"
    assert step.logical_call_count == 1
    assert step.correction_active is False
    assert np.count_nonzero(step.correction) == 0
    np.testing.assert_allclose(
        step.next_native_state - step.shadow_prediction, expected, atol=1.0e-12
    )
    np.testing.assert_allclose(
        step.metric_intervention, step.retained_displacement, atol=1.0e-12
    )
    assert step.maximum_integral_difference_abs <= 1.0e-12
    assert step.maximum_boundary_difference_abs <= 1.0e-12
    assert step.maximum_projection_idempotence_abs <= 1.0e-12
    assert step.maximum_update_identity_abs <= 1.0e-12


def test_exact_window_preserves_three_call_a32_metric_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = np.zeros((3, 4), dtype=np.float64)
    correction = np.full_like(state, 0.125)
    call_order: list[tuple[int, int]] = []

    def predictor(resolution, value):
        call_order.append(resolution)
        return np.asarray(value, dtype=np.float64) + 1.0

    def exact_step(accepted, shadow, *, predictor, **kwargs):
        del kwargs
        shadow_prediction = predictor(evaluator.NATIVE_RESOLUTION, shadow)
        accepted_prediction = predictor(evaluator.NATIVE_RESOLUTION, accepted)
        predictor(evaluator.FINE_RESOLUTION, accepted)
        proposal_state = accepted_prediction + correction
        audit = SimpleNamespace(
            cap_active=False,
            correction_to_native_increment=0.01,
            maximum_scaled_component_mean_abs=0.0,
            maximum_excluded_abs=0.0,
            maximum_inactive_coordinate_abs=0.0,
            maximum_modal_reconstruction_abs=0.0,
        )
        proposal = SimpleNamespace(
            correction_active=True,
            prepared_inputs=SimpleNamespace(
                nesting_floors={
                    "pre_model_fine_to_native_max_abs": 0.0,
                    "post_fp32_fine_to_native_max_abs": 0.0,
                }
            ),
            audit=audit,
            predictions={evaluator.NATIVE_RESOLUTION: accepted_prediction},
            next_native_state=proposal_state,
            correction=correction,
        )
        return SimpleNamespace(
            shadow_prediction=shadow_prediction,
            proposal=proposal,
            next_native_state=proposal_state,
            retained_displacement=proposal_state - shadow_prediction,
            tether_audit=SimpleNamespace(
                maximum_integral_difference_abs=0.0,
                maximum_boundary_difference_abs=0.0,
                maximum_projection_idempotence_abs=0.0,
            ),
            maximum_update_identity_abs=0.0,
            logical_call_count=3,
        )

    monkeypatch.setattr(
        evaluator, "synchronized_shadow_tethered_binary_affine_step", exact_step
    )
    step = evaluator.phase_selective_active_step(
        state,
        state,
        input_call=0,
        projector=object(),
        predictor=predictor,
        start_coefficients=np.zeros((19, 19)),
        end_coefficients=np.zeros((19, 19)),
        volumes=np.ones(state.shape[0]),
        residual_scale=np.ones(state.shape[1]),
        state_scale=np.ones(state.shape[1]),
    )

    assert call_order == [
        evaluator.NATIVE_RESOLUTION,
        evaluator.NATIVE_RESOLUTION,
        evaluator.FINE_RESOLUTION,
    ]
    assert step.route == "exact_a32_window"
    assert step.logical_call_count == 3
    assert step.correction_active is True
    np.testing.assert_array_equal(step.metric_intervention, correction)
    assert step.maximum_proposal_update_abs == 0.0


def _score_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for scope in (
        "population",
        "active_population",
        "active_group_e12",
        "active_group_e14",
    ):
        for view in ("full", "rank8_parallel"):
            ratio = 0.9
            if scope == "active_population" and view == "full":
                ratio = evaluator.A32_ACTIVE_FULL - 0.01
            if scope == "active_population" and view == "rank8_parallel":
                ratio = evaluator.A32_ACTIVE_RANK8 - 0.01
            rows.append(
                {
                    "cell": "overall",
                    "view": view,
                    "scope": scope,
                    "case_id": None,
                    "skill_status": "ok",
                    "rms_ratio_vs_zero": ratio,
                }
            )
    rows.append(
        {
            "cell": "endpoint_29",
            "view": "full",
            "scope": "active_population",
            "case_id": None,
            "skill_status": "ok",
            "rms_ratio_vs_zero": evaluator.A32_ACTIVE_ENDPOINT - 0.01,
        }
    )
    for case_id in evaluator.EXPECTED_ACTIVE_CASES:
        for cell in ("overall", "endpoint_29"):
            rows.append(
                {
                    "cell": cell,
                    "view": "full",
                    "scope": "case",
                    "case_id": case_id,
                    "skill_status": "ok",
                    "rms_ratio_vs_zero": 0.9,
                }
            )
    return rows


def _audit_rows() -> list[dict[str, object]]:
    exact = {
        "route": "exact_a32_window",
        "call_order_exact": True,
        "logical_call_count": 3,
        "correction_active": True,
        "correction_to_native_increment": 0.01,
        "pre_model_fine_to_native_max_abs": 0.0,
        "post_fp32_fine_to_native_max_abs": 0.0,
        "maximum_scaled_component_mean_abs": 0.0,
        "maximum_excluded_abs": 0.0,
        "maximum_inactive_coordinate_abs": 0.0,
        "maximum_modal_reconstruction_abs": 0.0,
        "maximum_proposal_update_abs": 0.0,
        "maximum_integral_difference_abs": 0.0,
        "maximum_boundary_difference_abs": 0.0,
        "maximum_projection_idempotence_abs": 0.0,
        "maximum_update_identity_abs": 0.0,
    }
    coast = {
        "route": "surrogate_coast",
        "call_order_exact": True,
        "logical_call_count": 1,
        "correction_active": False,
        "pre_model_fine_to_native_max_abs": 0.0,
        "post_fp32_fine_to_native_max_abs": 0.0,
        "maximum_proposal_update_abs": 0.0,
        "maximum_integral_difference_abs": 0.0,
        "maximum_boundary_difference_abs": 0.0,
        "maximum_projection_idempotence_abs": 0.0,
        "maximum_update_identity_abs": 0.0,
    }
    inactive = {
        "route": "inactive_raw",
        "call_order_exact": True,
        "logical_call_count": 1,
        "retained_displacement_rms": 0.0,
    }
    return [
        *(deepcopy(exact) for _ in range(64)),
        *(deepcopy(coast) for _ in range(56)),
        *(deepcopy(inactive) for _ in range(300)),
    ]


def _gate_arguments() -> dict[str, object]:
    return {
        "score_rows": _score_rows(),
        "case_rows": [
            {
                "case_id": case_id,
                "increment_defect_rms_ratio": 0.9,
                "endpoint_cumulative_defect_ratio": 0.9,
            }
            for case_id in evaluator.EXPECTED_ACTIVE_CASES
        ],
        "controls": [{"status": "ok", "ratio": 1.0}],
        "rollouts": [
            {
                "execution": {"completed_calls": 30},
                "rows": [{"finite": True, "admissible": True}],
            }
            for _ in range(2 * len(evaluator.CASE_IDS))
        ],
        "audit_rows": _audit_rows(),
        "inactive_max_abs": {case_id: 0.0 for case_id in evaluator.INACTIVE_CASES},
        "shadow_raw_max_abs": {
            case_id: 0.0 for case_id in evaluator.EXPECTED_ACTIVE_CASES
        },
        "prefix_repeat_abs": 0.0,
        "main_execution": {
            "candidate": {"logical_model_calls": evaluator.CANDIDATE_CALLS}
        },
        "structural_checks": {"structural": True},
    }


def test_gate_requires_routes_cost_and_strict_science_thresholds() -> None:
    arguments = _gate_arguments()
    qualified = evaluator._gate(**arguments)

    assert qualified["status"] == "qualified"
    assert all(qualified["checks"].values())

    equality = deepcopy(arguments)
    row = evaluator.a31._score_row(
        equality["score_rows"],
        cell="overall",
        view="full",
        scope="population",
    )
    row["rms_ratio_vs_zero"] = 1.0
    failed = evaluator._gate(**equality)

    assert failed["status"] == "failed"
    assert "population_full_rank8_trajectory_strict" in failed["failed_checks"]


def test_visualizer_accepts_only_registered_unrendered_a43_contract(
    tmp_path: Path,
) -> None:
    manifest = evaluator.with_payload_sha256(
        {
            "schema": evaluator.ANIMATION_SCHEMA,
            "working_id": evaluator.WORKING_ID,
            "contract": evaluator.ANIMATION_CONTRACT,
            "bundles": [],
            "inference_or_gate_input": False,
        }
    )
    manifest_path = tmp_path / "bundle_manifest.json"
    evaluator.atomic_write_json(manifest_path, manifest)
    rollout = evaluator.with_payload_sha256(
        {
            "schema": evaluator.RESULT_SCHEMA,
            "working_id": evaluator.WORKING_ID,
            "recurrence_executed": True,
            "animations_rendered": False,
            "animation_bundle_manifest": manifest,
            "animation_bundle_manifest_sha256": evaluator.sha256_file(manifest_path),
        }
    )
    rollout_path = tmp_path / "phase_selective_coast_replay.json"
    evaluator.atomic_write_json(rollout_path, rollout)

    loaded_rollout, loaded_manifest = visualizer._load_contract(
        rollout_path, manifest_path
    )
    assert loaded_rollout["working_id"] == evaluator.WORKING_ID
    assert loaded_manifest["contract"] == evaluator.ANIMATION_CONTRACT

    bad_manifest = evaluator.with_payload_sha256(
        {**manifest, "contract": {**manifest["contract"], "fps": 99}}
    )
    evaluator.atomic_write_json(manifest_path, bad_manifest)
    with pytest.raises(ValueError, match="registered A43 contract"):
        visualizer._load_contract(rollout_path, manifest_path)
