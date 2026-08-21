from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from scripts.time_dependent_no import (
    evaluate_pcno_affine_shadow_tether_query_grid as evaluator,
)
from scripts.time_dependent_no import (
    rescore_pcno_affine_shadow_tether_query_grid as rescorer,
)
from scripts.time_dependent_no import (
    visualize_pcno_affine_shadow_tether_query_grid as visualizer,
)
from utility.time_dependent_no.pcno_cross_resolution_correction import (
    prolong_nested_state,
    transfer_floor_fields,
)
from utility.time_dependent_no.pcno_resolution_transfer import restrict_nested_state


def test_a33_inventory_and_cost_contract_is_frozen() -> None:
    summary = evaluator.synthetic_summary()

    assert summary["status"] == "passed"
    assert all(summary["checks"].values())
    assert evaluator.CASE_IDS == (
        "sv_e00_y00",
        "sv_e00_y08",
        "sv_e06_y00",
        "sv_e06_y08",
        "sv_e11_y00",
        "sv_e11_y08",
    )


def test_error_cost_table_is_case_first_and_counts_every_arm() -> None:
    rows = []
    for policy_index, policy in enumerate(evaluator.POLICIES, start=1):
        for case_id in evaluator.CASE_IDS:
            rows.extend(
                {
                    "policy": policy,
                    "case_id": case_id,
                    "state_error": float(policy_index),
                }
                for _ in evaluator.INPUT_CALLS
            )
    execution = {
        policy: {
            "logical_model_calls": 180,
            "actual_forward_passes": 180,
            "total_forward_seconds": 10.0,
            "wall_seconds": 12.0,
        }
        for policy in evaluator.POLICIES
    }
    execution["mapping"] = {
        "matched_initialization_seconds": 1.0,
        "raw_transfer_output_seconds": 2.0,
        "corrected_transfer_output_seconds": 3.0,
    }

    table = evaluator._arm_error_cost(rows, execution)

    assert set(table) == set(evaluator.POLICIES)
    assert table[evaluator.DIRECT_QUERY_POLICY]["case_first_h30_state_rms"] == 1.0
    assert (
        table[evaluator.RAW_TRANSFER_POLICY]["case_first_trajectory_state_rms"] == 3.0
    )
    assert table[evaluator.CORRECTED_TRANSFER_POLICY][
        "trajectory_error_ratio_vs_raw_transfer"
    ] == pytest.approx(4.0 / 3.0)
    assert (
        table[evaluator.RAW_TRANSFER_POLICY]["model_plus_required_mapping_seconds"]
        == 12.0
    )

    with pytest.raises(ValueError, match="inventory"):
        evaluator._arm_error_cost(rows[:-1], execution)
    assert evaluator.INPUT_CALLS == tuple(range(30))
    assert evaluator.ANIMATION_CALLS == tuple(range(0, 31, 2))
    assert evaluator.EXPECTED_SELECTED_CASES == evaluator.CASE_IDS
    assert evaluator.OWNED_SOURCE_PATHS == (
        "docs/time_dependent_no/W26_L5_RESPONSE_FILTERED_BLOCK_PREREGISTRATION.md",
        "scripts/time_dependent_no/evaluate_pcno_affine_shadow_tether_query_grid.py",
        "scripts/time_dependent_no/visualize_pcno_affine_shadow_tether_query_grid.py",
        "tests/time_dependent_no/test_pcno_affine_shadow_tether_query_grid.py",
    )


def test_query_native_maps_conserve_and_transfer_fields_close() -> None:
    rng = np.random.default_rng(33)
    native_resolution = (4, 2)
    query_resolution = (8, 4)
    query = rng.normal(size=(32, 4))
    evolved_native = restrict_nested_state(
        query,
        fine_resolution=query_resolution,
        coarse_resolution=native_resolution,
    ) + rng.normal(scale=1.0e-3, size=(8, 4))
    floors = transfer_floor_fields(
        query,
        evolved_native,
        query_resolution=query_resolution,
        native_resolution=native_resolution,
    )

    np.testing.assert_allclose(
        floors.query_round_trip,
        floors.mapped_information_mismatch + floors.pipeline_truth_floor,
        rtol=0.0,
        atol=5.0e-16,
    )
    np.testing.assert_allclose(floors.closure_residual, 0.0, rtol=0.0, atol=5.0e-16)
    assert np.max(np.abs(floors.native_information_mismatch)) > 0.0

    native = rng.normal(size=(8, 4))
    prolonged = prolong_nested_state(
        native,
        coarse_resolution=native_resolution,
        fine_resolution=query_resolution,
    )
    restricted = restrict_nested_state(
        prolonged,
        fine_resolution=query_resolution,
        coarse_resolution=native_resolution,
    )
    np.testing.assert_array_equal(restricted, native)
    np.testing.assert_allclose(
        np.mean(prolonged, axis=0), np.mean(native, axis=0), rtol=0.0, atol=1.0e-15
    )


def test_transfer_floor_rows_use_separate_native_reference() -> None:
    rng = np.random.default_rng(34)
    query_truth = rng.normal(size=(31, 32, 4))
    native_truth = [
        restrict_nested_state(
            state,
            fine_resolution=(8, 4),
            coarse_resolution=(4, 2),
        )
        + 1.0e-4
        for state in query_truth
    ]
    runtime = SimpleNamespace(
        geometry_by_resolution={
            (8, 4): SimpleNamespace(node_measures=np.full((32, 1), 1.0 / 32.0)),
            (4, 2): SimpleNamespace(node_measures=np.full((8, 1), 1.0 / 8.0)),
        },
        normalization=SimpleNamespace(
            state_scale=np.ones(4), residual_scale=np.ones(4)
        ),
    )
    old_query = evaluator.QUERY_RESOLUTION
    old_native = evaluator.NATIVE_RESOLUTION
    try:
        evaluator.QUERY_RESOLUTION = (8, 4)
        evaluator.NATIVE_RESOLUTION = (4, 2)
        rows, closure = evaluator._transfer_floor_rows(
            runtime,
            case_id=evaluator.CASE_IDS[0],
            query_truth=query_truth,
            native_truth=native_truth,
        )
    finally:
        evaluator.QUERY_RESOLUTION = old_query
        evaluator.NATIVE_RESOLUTION = old_native

    assert len(rows) == 61
    assert closure <= 5.0e-16
    state_rows = [row for row in rows if row["object_kind"] == "state"]
    assert all(
        float(row["native_information_mismatch_scaled_rms"]) > 0.0 for row in state_rows
    )


def test_trajectory_error_recurrence_closes_and_fails_on_inventory() -> None:
    rng = np.random.default_rng(35)
    truth = rng.normal(size=(31, 7, 4))
    predicted = [truth[index] + 0.01 * index for index in range(31)]

    rows, maximum = evaluator._trajectory_identity_rows(
        case_id=evaluator.CASE_IDS[0],
        policy=evaluator.RAW_TRANSFER_POLICY,
        states=predicted,
        truth_states=truth,
    )
    assert len(rows) == 31
    assert maximum <= 2.0e-15
    with pytest.raises(ValueError, match="aligned"):
        evaluator._trajectory_identity_rows(
            case_id=evaluator.CASE_IDS[0],
            policy=evaluator.RAW_TRANSFER_POLICY,
            states=predicted[:-1],
            truth_states=truth,
        )


def _direct_case_rows(ratios: list[float | None]) -> list[dict[str, object]]:
    return [
        {"case_id": case_id, "endpoint_full_ratio": ratio}
        for case_id, ratio in zip(evaluator.CASE_IDS, ratios, strict=True)
    ]


def test_direct_gate_thresholds_and_denominators_are_fail_closed() -> None:
    control = {"status": "ok", "ratio": evaluator.CONTROL_LIMIT}
    qualified = evaluator._direct_gate(
        case_rows=_direct_case_rows([0.90, 0.90, 0.95, 0.95, 1.0, 1.0]),
        controls=[control],
        structural_checks={"synthetic": True},
    )
    assert qualified["status"] == "qualified"
    assert qualified["median_endpoint_state_ratio"] == 0.95
    assert qualified["case_win_count"] == 4

    equality_loses = evaluator._direct_gate(
        case_rows=_direct_case_rows([0.90, 0.90, 1.0, 1.0, 1.0, 1.0]),
        controls=[control],
        structural_checks={"synthetic": True},
    )
    assert equality_loses["status"] == "failed"
    assert "at_least_four_case_wins" in equality_loses["failed_checks"]

    unresolved = evaluator._direct_gate(
        case_rows=_direct_case_rows([0.90, 0.90, 0.90, 0.90, 0.90, None]),
        controls=[control],
        structural_checks={"synthetic": True},
    )
    assert unresolved["status"] == "failed"
    assert unresolved["median_endpoint_state_ratio"] is None
    assert "endpoint_denominators_resolved" in unresolved["failed_checks"]

    exact_zero = evaluator._direct_gate(
        case_rows=_direct_case_rows([0.90] * 6),
        controls=[{"status": "exact_zero_no_change", "ratio": 1.0}],
        structural_checks={"synthetic": True},
    )
    assert exact_zero["status"] == "failed"
    assert "all_controls_no_harm" in exact_zero["failed_checks"]


def test_visualizer_uses_all_four_arms_and_signed_a32_improvement() -> None:
    conservative = np.zeros((2, 8, 4), dtype=np.float64)
    conservative[..., 0] = 1.0
    conservative[..., 3] = 2.5
    bundle = {
        "gamma": np.asarray(1.4),
        "query_resolution": np.asarray((4, 2)),
        "truth_conservative": conservative,
        "direct_query_conservative": conservative + np.asarray((0.0, 0.0, 0.0, 0.1)),
        "matched_direct_conservative": conservative + np.asarray((0.0, 0.0, 0.0, 0.2)),
        "raw_transfer_conservative": conservative + np.asarray((0.0, 0.0, 0.0, 0.3)),
        "corrected_transfer_conservative": conservative
        + np.asarray((0.0, 0.0, 0.0, 0.15)),
    }
    contract = {"relative_improvement_denominator_floor": 1.0e-5}

    arrays = visualizer._animation_arrays(bundle, field="pressure", contract=contract)

    assert {
        "truth",
        "direct_query",
        "matched_direct",
        "raw_transfer",
        "corrected_transfer",
        "improvement",
    } <= set(arrays)
    assert arrays["truth"].shape == (2, 2, 4)
    assert np.all(arrays["improvement"] > 0.0)


def test_rescore_replaces_only_miswired_routing_check() -> None:
    original_checks = {
        "population_strict": True,
        rescorer.MISWIRED_CHECK: False,
        "all_controls": True,
    }
    prior = {
        "correction_survival_gate": {
            "status": "failed",
            "checks": original_checks,
            "failed_checks": [rescorer.MISWIRED_CHECK],
            "case_joint_win_count": 6,
        }
    }
    amended = rescorer._amended_correction_gate(
        prior,
        {
            rescorer.REPLACEMENT_CHECK: True,
            "routing_inventory_exact": True,
        },
    )

    assert amended["status"] == "qualified"
    assert amended["failed_checks"] == []
    assert rescorer.MISWIRED_CHECK not in amended["checks"]
    assert amended["checks"][rescorer.REPLACEMENT_CHECK] is True
    assert original_checks[rescorer.MISWIRED_CHECK] is False
    assert amended["model_calls"] == 0

    failed = rescorer._amended_correction_gate(
        prior,
        {rescorer.REPLACEMENT_CHECK: False},
    )
    assert failed["status"] == "failed"
    assert failed["failed_checks"] == [rescorer.REPLACEMENT_CHECK]


def test_rescore_routing_csv_contract_is_fail_closed(tmp_path) -> None:
    positions = []
    references = []
    for case_id in evaluator.CASE_IDS:
        positions.append(
            {
                "case_id": case_id,
                "position_selected": "True",
                "expected_selected": "True",
                "normalized_wall_distance": "0.7",
                "routing_native_vs_shard_max_abs": "1.2e-7",
                "source": (
                    "authenticated_query_call_zero_conservative_restriction_"
                    "before_model_calls"
                ),
            }
        )
        references.append(
            {
                "case_id": case_id,
                "loaded_native_reference_crosscheck_max_abs": "1.8e-15",
            }
        )

    def write(name, rows):
        import csv

        with (tmp_path / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    write("position_decisions.csv", positions)
    write("reference_checks.csv", references)
    checks, observed = rescorer._routing_check(tmp_path)
    assert all(checks.values())
    assert observed["maximum_float64_routing_reference_crosscheck_abs"] == 1.8e-15
    assert observed["maximum_float32_shard_audit_difference_abs"] == 1.2e-7

    positions[0]["source"] = "checkpoint_shard"
    write("position_decisions.csv", positions)
    checks, _ = rescorer._routing_check(tmp_path)
    assert checks["routing_source_is_authenticated_query_restriction"] is False
