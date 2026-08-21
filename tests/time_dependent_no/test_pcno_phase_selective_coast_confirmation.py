from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from scripts.time_dependent_no import (
    evaluate_pcno_phase_selective_coast_confirmation as evaluator,
)
from scripts.time_dependent_no import (
    visualize_pcno_phase_selective_coast_confirmation as visualizer,
)


def test_synthetic_contract_freezes_disjoint_population_cost_and_source_scope() -> None:
    summary = evaluator.synthetic_summary()

    assert summary["status"] == "passed"
    assert all(summary["checks"].values())
    assert evaluator.CASE_IDS == tuple(
        [*(f"sv_e00_y{i:02d}" for i in range(1, 8))]
        + [*(f"sv_e11_y{i:02d}" for i in range(1, 8))]
    )
    assert evaluator.EXPECTED_ACTIVE_CASES == (
        "sv_e00_y01",
        "sv_e00_y07",
        "sv_e11_y01",
        "sv_e11_y07",
    )
    assert evaluator.CANDIDATE_CALLS == 548
    assert evaluator.RAW_CALLS == 420
    assert evaluator.OWNED_SOURCE_PATHS == (
        "docs/time_dependent_no/W26_L5_A43_INDEPENDENT_CONFIRMATION_PREREGISTRATION.md",
        "scripts/time_dependent_no/evaluate_pcno_phase_selective_coast_confirmation.py",
        "scripts/time_dependent_no/visualize_pcno_phase_selective_coast_confirmation.py",
        "tests/time_dependent_no/test_pcno_phase_selective_coast_confirmation.py",
    )


class _Store:
    def __init__(self, rows: dict[str, dict[str, Any]]) -> None:
        self.rows = rows

    def entry(self, case_id: str) -> dict[str, Any]:
        return self.rows[case_id]


def test_native_truth_inventory_binds_shards_without_reference_archives_and_tamper(
    tmp_path: Path,
) -> None:
    rows: dict[str, dict[str, Any]] = {}
    for case_id in evaluator.CASE_IDS:
        folder = f"traj_{case_id}"
        case_root = tmp_path / folder
        case_root.mkdir(parents=True)
        states_path = case_root / "states_conservative.npy"
        times_path = case_root / "physical_times.npy"
        states_path.write_bytes(f"states:{case_id}".encode())
        times_path.write_bytes(f"times:{case_id}".encode())
        source_sha = f"source-reference-{case_id}"
        (case_root / "metadata.json").write_text(
            json.dumps(
                {
                    "source_key": case_id,
                    "source_reference_sha256": source_sha,
                    "float32_serialization": {
                        "state_global_relative_l2": 2.0e-8,
                        "state_max_frame_relative_l2": 3.0e-8,
                    },
                    "float32_tolerances": {"state_relative_l2": 1.0e-6},
                }
            ),
            encoding="utf-8",
        )
        rows[case_id] = {
            "folder": folder,
            "source_reference_sha256": source_sha,
            "state_digest": f"state-digest-{case_id}",
            "array_sha256": {
                "states_conservative": evaluator.sha256_file(states_path),
                "physical_times": evaluator.sha256_file(times_path),
            },
        }
    args = SimpleNamespace(data_dir=tmp_path)

    inventory = evaluator._native_truth_inventory(args, _Store(rows))

    assert set(inventory) == set(evaluator.CASE_IDS)
    assert all(
        row["truth_kind"] == "checkpoint_bound_native_shard"
        and row["arrays"]["states_conservative"]["bytes"] > 0
        for row in inventory.values()
    )
    assert not list(tmp_path.rglob("reference.npz"))
    tampered = (
        tmp_path / rows[evaluator.CASE_IDS[0]]["folder"] / "states_conservative.npy"
    )
    tampered.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="native truth digest differs"):
        evaluator._native_truth_inventory(args, _Store(rows))

    first_case = evaluator.CASE_IDS[0]
    tampered.write_bytes(f"states:{first_case}".encode())
    metadata_path = tmp_path / rows[first_case]["folder"] / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["float32_serialization"]["state_max_frame_relative_l2"] = 2.0e-6
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="serialization is unverified"):
        evaluator._native_truth_inventory(args, _Store(rows))


def _score_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for scope in (
        "population",
        "active_population",
        "active_group_e00",
        "active_group_e11",
    ):
        for view in ("full", "rank8_parallel"):
            rows.append(
                {
                    "cell": "overall",
                    "view": view,
                    "scope": scope,
                    "case_id": None,
                    "skill_status": "ok",
                    "rms_ratio_vs_zero": 0.9,
                }
            )
    for scope in ("active_population", "active_group_e00", "active_group_e11"):
        rows.append(
            {
                "cell": "endpoint_29",
                "view": "full",
                "scope": scope,
                "case_id": None,
                "skill_status": "ok",
                "rms_ratio_vs_zero": 0.9,
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
            "raw": {"logical_model_calls": evaluator.RAW_CALLS},
            "candidate": {"logical_model_calls": evaluator.CANDIDATE_CALLS},
        },
        "structural_checks": {"structural": True},
    }


def test_gate_passes_only_strict_group_and_case_confirmation() -> None:
    arguments = _gate_arguments()
    qualified = evaluator._gate(**arguments)

    assert qualified["status"] == "qualified"
    assert all(qualified["checks"].values())
    assert qualified["active_case_joint_win_count"] == 4

    equality = deepcopy(arguments)
    row = evaluator.a31._score_row(
        equality["score_rows"],
        cell="endpoint_29",
        view="full",
        scope="active_group_e11",
    )
    row["rms_ratio_vs_zero"] = 1.0
    failed = evaluator._gate(**equality)

    assert failed["status"] == "failed"
    assert "active_group_e11_full_endpoint_strict" in failed["failed_checks"]


def test_gate_fails_closed_on_route_or_cost_mismatch() -> None:
    route = _gate_arguments()
    route["audit_rows"] = route["audit_rows"][:-1]
    assert evaluator._gate(**route)["status"] == "failed"

    cost = _gate_arguments()
    cost["main_execution"]["candidate"]["logical_model_calls"] += 1
    failed = evaluator._gate(**cost)
    assert failed["status"] == "failed"
    assert "logical_cost_exact" in failed["failed_checks"]


def test_visualizer_accepts_only_registered_unrendered_a44_contract(
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
    rollout_path = tmp_path / "phase_selective_coast_confirmation.json"
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
    with pytest.raises(ValueError, match="registered A44 contract"):
        visualizer._load_contract(rollout_path, manifest_path)


def test_visualizer_bundle_loader_uses_training_support_group_binding(
    tmp_path: Path,
) -> None:
    contract = {
        "native_resolution": [2, 1],
        "output_calls": [0, 2],
        "physical_times": [0.0, 0.04],
    }
    path = tmp_path / "sv_e00_y01_phase_selective_coast_confirmation_h30.npz"
    state = np.zeros((2, 2, 4), dtype=np.float32)

    def write(group_id: str) -> None:
        np.savez_compressed(
            path,
            case_id=np.asarray("sv_e00_y01"),
            split_group_id=np.asarray(group_id),
            output_calls=np.asarray(contract["output_calls"], dtype=np.int64),
            physical_times=np.asarray(contract["physical_times"], dtype=np.float64),
            native_resolution=np.asarray(contract["native_resolution"], dtype=np.int64),
            nodes=np.zeros((2, 2), dtype=np.float32),
            volumes=np.ones(2, dtype=np.float32),
            truth_conservative=state,
            raw_shadow_conservative=state,
            corrected_conservative=state,
        )

    write("train_e00")
    expected = {
        "case_id": "sv_e00_y01",
        "split_group_id": "train_e00",
        "frames": 2,
        "sha256": evaluator.sha256_file(path),
    }
    loaded = visualizer._load_confirmation_bundle(path, expected, contract)
    assert loaded["split_group_id"].item() == "train_e00"

    write("strength_ood_e00")
    expected["sha256"] = evaluator.sha256_file(path)
    with pytest.raises(ValueError, match="bundle contract mismatch"):
        visualizer._load_confirmation_bundle(path, expected, contract)
