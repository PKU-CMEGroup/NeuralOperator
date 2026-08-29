from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.time_dependent_no import evaluate_p0_restart_sufficiency_a2 as evaluator
from scripts.time_dependent_no import run_p0_restart_sufficiency_a2_solver as runner
from utility.time_dependent_no import p0_restart_sufficiency_a2 as a2
from utility.time_dependent_no import pcno_resolution_transfer, pcno_runtime
from utility.time_dependent_no.shock_vortex_coarse_cfd import CoarseCFDRollout


def _solver_summary(role: str) -> dict:
    rows = []
    for case in a2.FIXED_CASES:
        rows.append(
            {
                "trajectory_id": case["trajectory_id"],
                "frame": case["frame"],
                "stored_input_sha256": case["input_frame_sha256"],
                "same_process_state_bytes_identical": True,
                "same_process_exchange_bytes_identical": True,
                "accepted_steps": 7,
                "rejected_attempts": 0,
                "face_fallbacks": 0,
                "boundary_balance_residual": 1.0e-14,
                "finite": True,
                "admissible": True,
            }
        )
    return {
        "schema": a2.A2_SOLVER_SUMMARY_SCHEMA,
        "stage": "P0-A2",
        "status": "pass",
        "process_role": role,
        "cases_contract": a2.case_contract(),
        "bindings": {
            "a0_artifact_manifest_sha256": "a" * 64,
            "a1_closeout_manifest_sha256": "b" * 64,
            "source_manifest_sha256": "c" * 64,
            "source_mapping_sha256": "d" * 64,
            "solver_config_sha256": "e" * 64,
        },
        "cases": rows,
    }


def _states(value: float = 1.0) -> dict[str, np.ndarray]:
    return {
        str(case["trajectory_id"]): np.full((4, 4), value, dtype=np.float64)
        for case in a2.FIXED_CASES
    }


def test_a2_evaluator_uses_shared_pcno_runtime_infrastructure() -> None:
    assert (
        evaluator.checkpoint_model_node_type_input
        is pcno_runtime.checkpoint_model_node_type_input
    )
    assert evaluator.load_checkpoint is pcno_runtime.load_checkpoint
    assert evaluator.select_device is pcno_runtime.select_device
    assert evaluator.timed_model_call is pcno_runtime.timed_model_call
    assert (
        evaluator.build_resolution_checkpoint_model
        is pcno_resolution_transfer.build_resolution_checkpoint_model
    )

    checkpoint = {
        "step_stride": 1,
        "data_manifest_digest": evaluator.DATASET_MANIFEST_SHA256,
        "data_contract": {
            "data_manifest_digest": evaluator.DATASET_MANIFEST_SHA256,
        },
        "model_node_type_input": "physical",
        "model_config": {
            "k_max": 8,
            "domain_lengths": [2.0, 1.0],
            "layers": [128, 128, 128, 128, 128],
            "fc_dim": 128,
            "nmeasures": 1,
        },
        "normalization": {"state_scale": list(evaluator.CHANNEL_SCALES)},
    }
    evaluator._validate_checkpoint(checkpoint, label="synthetic", stride=1)


def test_scaled_rms_implements_registered_volume_metric() -> None:
    left = np.asarray([[1.0, 2.0], [3.0, 5.0]])
    right = np.asarray([[0.0, 0.0], [1.0, 1.0]])
    volumes = np.asarray([1.0, 3.0])
    scale = np.asarray([1.0, 2.0])
    observed = a2.scaled_rms_distance(
        left, right, volumes=volumes, component_scale=scale
    )
    expected = np.sqrt((1.0 + 1.0 + 3.0 * (4.0 + 4.0)) / 4.0)
    assert observed == pytest.approx(expected)


def test_scaled_rms_rejects_nonpositive_weights_and_nonfinite_states() -> None:
    state = np.ones((2, 4))
    with pytest.raises(ValueError, match="positive"):
        a2.scaled_rms_distance(state, state, volumes=np.asarray([1.0, 0.0]))
    state[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        a2.scaled_rms_distance(state, np.ones((2, 4)), volumes=np.ones(2))


def test_bias_gate_is_per_case_and_fails_zero_denominators() -> None:
    passing = a2.bias_gate_row(baseline=0.1, d044=0.5, d060=0.4, separation=0.5)
    assert passing["pass"] is True
    assert passing["baseline_over_min_model_defect"] == pytest.approx(0.25)
    unresolved = a2.bias_gate_row(baseline=0.0, d044=0.0, d060=1.0, separation=0.0)
    assert unresolved["pass"] is False
    assert unresolved["model_defect_denominator_resolved"] is False
    assert unresolved["model_separation_denominator_resolved"] is False


def test_exact_array_bytes_checks_dtype_shape_and_content() -> None:
    first = np.arange(12, dtype=np.float64).reshape(3, 4)
    a2.require_exact_array_bytes(first, first.copy(), label="state")
    with pytest.raises(ValueError, match="bytes differ"):
        a2.require_exact_array_bytes(first, first.astype(np.float32), label="state")
    changed = first.copy()
    changed[0, 0] = -0.0
    with pytest.raises(ValueError, match="bytes differ"):
        a2.require_exact_array_bytes(first, changed, label="state")


def test_process_comparison_checks_bindings_counts_and_primary_distance() -> None:
    primary = _solver_summary("primary")
    repeat = _solver_summary("fresh_repeat")
    states = _states()
    rows = a2.compare_solver_process_summaries(
        primary,
        repeat,
        primary_states=states,
        repeat_states={key: value.copy() for key, value in states.items()},
        volumes=np.ones(4),
    )
    assert len(rows) == 3
    assert all(row["primary_distance_scaled_rms"] == 0.0 for row in rows)

    repeat["cases"][0]["accepted_steps"] += 1
    with pytest.raises(ValueError, match="count differs"):
        a2.compare_solver_process_summaries(
            primary,
            repeat,
            primary_states=states,
            repeat_states=states,
            volumes=np.ones(4),
        )


def test_process_comparison_enforces_fixed_tolerance() -> None:
    primary = _solver_summary("primary")
    repeat = _solver_summary("fresh_repeat")
    left = _states()
    right = {key: value.copy() for key, value in left.items()}
    right[str(a2.FIXED_CASES[1]["trajectory_id"])][0, 0] += 1.0e-10
    with pytest.raises(ValueError, match="exceeds tolerance"):
        a2.compare_solver_process_summaries(
            primary,
            repeat,
            primary_states=left,
            repeat_states=right,
            volumes=np.ones(4),
        )


def test_primitive_metrics_and_admissibility() -> None:
    state = np.asarray([[1.0, 2.0, 0.0, 4.5], [2.0, 2.0, 2.0, 6.0]])
    primitive = a2.primitive_fields(state)
    np.testing.assert_allclose(primitive[:, :3], [[1.0, 2.0, 0.0], [2.0, 1.0, 1.0]])
    summary = a2.admissibility_summary(state)
    assert summary["finite"] is True
    assert summary["admissible"] is True
    invalid = state.copy()
    invalid[0, 0] = -1.0
    assert a2.admissibility_summary(invalid)["admissible"] is False


def test_relative_l2_and_conservative_change_use_physical_volumes() -> None:
    reference = np.ones((2, 4))
    prediction = reference.copy()
    prediction[1] += 1.0
    volumes = np.asarray([1.0, 3.0])
    metrics = a2.weighted_relative_l2_metrics(prediction, reference, volumes=volumes)
    assert metrics["unscaled_physical_volume_relative_l2"] == pytest.approx(
        np.sqrt(3.0 / 4.0)
    )
    np.testing.assert_allclose(
        a2.integrated_conservative_change(prediction, reference, volumes=volumes),
        np.full(4, 3.0),
    )


def test_solver_checks_are_json_serializable() -> None:
    config = runner._config()
    initial = np.tile(np.asarray([1.0, 0.0, 0.0, 2.5]), (a2.NATIVE_NODES, 1))
    states = np.stack((initial, initial))
    rollout = CoarseCFDRollout(
        config=config,
        states=states,
        interval_boundary_exchange=np.zeros((1, 4), dtype=np.float64),
        core_seconds=0.0,
        accepted_steps=1,
        rejected_attempts=0,
        face_reconstruction_fallbacks=0,
        minimum_density=1.0,
        minimum_pressure=1.0,
    )

    checks = runner._validate_rollout(rollout, initial=initial)

    assert json.loads(json.dumps(checks)) == checks
    assert isinstance(checks["integrated_conservative_change"], list)
    assert isinstance(checks["recorded_outward_boundary_exchange"], list)


def test_a2_physical_volumes_validate_serialized_geometry_and_restore_fp64() -> None:
    serialized = np.full(
        (a2.NATIVE_NODES, 1),
        np.float32(a2.NATIVE_DOMAIN_MEASURE / a2.NATIVE_NODES),
        dtype=np.float32,
    )

    volumes = a2.validated_a2_physical_volumes(serialized)

    assert volumes.dtype == np.float64
    assert volumes.shape == (a2.NATIVE_NODES,)
    assert volumes.sum() == pytest.approx(
        a2.NATIVE_DOMAIN_MEASURE, rel=0.0, abs=1.0e-15
    )
    with pytest.raises(ValueError, match="cell-volume contract"):
        a2.validated_a2_physical_volumes(serialized.astype(np.float64))
    serialized[0, 0] = np.nextafter(serialized[0, 0], np.float32(np.inf))
    with pytest.raises(ValueError, match="cell-volume contract"):
        a2.validated_a2_physical_volumes(serialized)


def test_a2_requires_the_bound_deterministic_cublas_workspace() -> None:
    a2.require_a2_cublas_workspace_config(
        a2.DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG
    )
    with pytest.raises(ValueError, match="CUBLAS_WORKSPACE_CONFIG"):
        a2.require_a2_cublas_workspace_config(None)
    with pytest.raises(ValueError, match="CUBLAS_WORKSPACE_CONFIG"):
        a2.require_a2_cublas_workspace_config(":16:8")


def test_payload_binding_converts_numpy_before_hashing() -> None:
    bound = a2.bind_payload_sha256(
        {
            "vector": np.asarray([1.0, 2.0], dtype=np.float64),
            "count": np.int64(3),
        }
    )
    payload_sha256 = bound.pop("payload_sha256")

    assert bound == {"vector": [1.0, 2.0], "count": 3}
    assert payload_sha256 == a2.canonical_json_sha256(bound)
    json.dumps(bound)
    with pytest.raises(ValueError, match="already SHA-256 bound"):
        a2.bind_payload_sha256({"payload_sha256": "0" * 64})


def test_artifact_manifest_round_trip_and_tamper(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    (root / "result.txt").write_text("fixed\n", encoding="utf-8")
    manifest = a2.build_artifact_manifest(root, schema="test_schema")
    a2.atomic_write_json(root / "artifact_manifest.json", manifest)
    validated = a2.validate_artifact_tree(root, expected_schema="test_schema")
    assert validated["member_count"] == 1
    (root / "result.txt").write_text("drift\n", encoding="utf-8")
    with pytest.raises(ValueError, match="drifted"):
        a2.validate_artifact_tree(root, expected_schema="test_schema")


def test_artifact_manifest_rejects_path_traversal(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    payload = {
        "schema": "test_schema",
        "root": "output_dir",
        "self_exclusion": "artifact_manifest.json",
        "member_count": 1,
        "total_member_bytes": 0,
        "members": {"../escape": {"bytes": 0, "sha256": "0" * 64}},
        "canonical_member_mapping_sha256": "0" * 64,
    }
    (root / "artifact_manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unsafe"):
        a2.validate_artifact_tree(root, expected_schema="test_schema")


def test_source_manifest_detects_hash_drift_and_unmanifested_python(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "source"
    root.mkdir()
    (root / "base.py").write_text("BASE = 1\n", encoding="utf-8")
    (root / "runner.py").write_text("RUN = 1\n", encoding="utf-8")
    base = {"base.py": a2.sha256_file(root / "base.py")}
    members = {
        "base.py": base["base.py"],
        "runner.py": a2.sha256_file(root / "runner.py"),
    }
    monkeypatch.setitem(
        a2.EXPECTED_BASE_MAPPING,
        "native_solver",
        a2.canonical_json_sha256(base),
    )
    monkeypatch.setitem(
        a2.REQUIRED_A2_SOURCE_MEMBERS,
        "native_solver",
        frozenset(members),
    )
    payload = {
        "schema": a2.A2_SOURCE_MANIFEST_SCHEMA,
        "source_role": "native_solver",
        "base_members": base,
        "base_mapping_sha256": a2.canonical_json_sha256(base),
        "members": members,
        "member_count": len(members),
        "canonical_member_mapping_sha256": a2.canonical_json_sha256(members),
    }
    assert (
        a2.validate_a2_source_manifest(
            payload, source_root=root, source_role="native_solver"
        )
        == members
    )

    (root / "extra.py").write_text("EXTRA = 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="inventory differ"):
        a2.validate_a2_source_manifest(
            payload, source_root=root, source_role="native_solver"
        )
    (root / "extra.py").unlink()
    (root / "runner.py").write_text("RUN = 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="drifted"):
        a2.validate_a2_source_manifest(
            payload, source_root=root, source_role="native_solver"
        )
