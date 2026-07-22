from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

import utility.time_dependent_no.cpg_release as cpg_release  # noqa: E402
from utility.time_dependent_no.cpg_release import (  # noqa: E402
    cpg_graph_frame_metadata,
    fingerprint_cpg_trajectory,
    graph_distance_from_sources,
    hash_cpg_group_targets,
    hash_rollout_targets,
    release_rollout_metrics,
    rollout_rmse_by_graph_distance,
    sha256_file,
    validate_cpg_reference_source,
)


def _fake_group(num_steps: int = 4, num_nodes: int = 5) -> dict[str, np.ndarray]:
    time = np.arange(num_steps, dtype=np.float64)[:, None, None]
    nodes = np.arange(num_nodes, dtype=np.float64)[None, :, None]
    primitives = {
        "rho": 1.0 + 0.01 * time + 0.1 * nodes,
        "v1": 0.2 + 0.02 * time + np.zeros_like(nodes),
        "v2": -0.1 + 0.03 * nodes + np.zeros_like(time),
        "pres": 1.0 + 0.04 * time + 0.2 * (nodes >= 2),
    }
    pos = np.zeros((num_steps, num_nodes, 2), dtype=np.float64)
    pos[..., 0] = np.linspace(0.0, 1.0, num_nodes)[None, :]
    edges = np.tile(
        np.stack((np.arange(num_nodes - 1), np.arange(1, num_nodes)), axis=-1)[
            None, ...
        ],
        (num_steps, 1, 1),
    )
    node_type = np.zeros((num_steps, num_nodes, 1), dtype=np.int32)
    node_type[:, -1] = 1
    mach = np.full((num_steps, num_nodes, 1), 2.25, dtype=np.float64)
    return {
        **primitives,
        "pos": pos,
        "edges": edges,
        "node_type": node_type,
        "Mach": mach,
    }


def _primitive_targets(group: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(group[key][1:], dtype=np.float32)
            for key in ("rho", "v1", "v2", "pres")
        ],
        axis=-1,
    )


def _load_audit_module():
    root = Path(__file__).resolve().parents[2]
    path = root / "scripts" / "time_dependent_no" / "audit_cpg_release_provenance.py"
    spec = importlib.util.spec_from_file_location("audit_cpg_release_provenance", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_evaluate_module():
    root = Path(__file__).resolve().parents[2]
    path = root / "scripts" / "time_dependent_no" / "evaluate_cpg_release.py"
    spec = importlib.util.spec_from_file_location("evaluate_cpg_release", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_evaluator_selection_uses_explicit_hdf5_keys():
    module = _load_evaluate_module()

    assert module.select_trajectory_keys(["b", "a"], [], 1) == ["b"]
    assert module.select_trajectory_keys(["b", "a"], ["a"], None) == ["a"]
    with pytest.raises(KeyError, match="absent"):
        module.select_trajectory_keys(["b", "a"], ["missing"], None)
    with pytest.raises(ValueError, match="unique"):
        module.select_trajectory_keys(["b", "a"], ["a", "a"], None)


def test_evaluator_binds_completed_legal_training_manifest(tmp_path: Path):
    module = _load_evaluate_module()
    checkpoint_sha256 = "a" * 64
    path = tmp_path / "training_manifest.json"
    payload = {
        "schema": module.LEGAL_TRAINING_SCHEMA,
        "status": "complete",
        "claim_scope": "release-bundle legal training fixture",
        "boundary_mode": module.LEGAL_BOUNDARY_MODE,
        "uses_future_reference_boundary": False,
        "checkpoint": {"sha256": checkpoint_sha256},
        "dataset": {
            "file_name": "train.h5",
            "split": "train",
            "sha256": "b" * 64,
            "selected_keys": ["0", "1"],
        },
        "model_configuration": {
            "message_passing_num": 12,
            "node_input_size": 6,
            "edge_input_size": 5,
            "dt": 0.025,
        },
        "training_configuration": {
            "seed": 7,
            "microbatch_size": 1,
            "gradient_accumulation_steps": 3,
            "effective_batch_size": 3,
            "num_steps": 3,
            "stage1_epochs": 15,
            "stage2_epochs": 5,
            "checkpoint_selection": "final completed epoch",
        },
        "source_sha256": {"trainer": "c" * 64},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    result = module.load_checkpoint_training_manifest(
        path,
        checkpoint_sha256=checkpoint_sha256,
    )
    assert result["uses_future_reference_boundary"] is False
    assert result["training_dataset"]["selected_key_count"] == 2
    assert result["training_configuration"]["effective_batch_size"] == 3
    assert result["sha256"] == sha256_file(path)

    with pytest.raises(ValueError, match="checkpoint SHA256 differs"):
        module.load_checkpoint_training_manifest(
            path,
            checkpoint_sha256="d" * 64,
        )
    payload["uses_future_reference_boundary"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="future boundaries"):
        module.load_checkpoint_training_manifest(
            path,
            checkpoint_sha256=checkpoint_sha256,
        )


def test_reference_source_hash_fallback_is_explicit_and_strict(monkeypatch):
    root = Path("artifacts") / "time_dependent_no" / "cpg_reference_fixture"
    root.mkdir(parents=True, exist_ok=True)
    source = root / "runtime.py"
    source.write_text("PINNED = True\n", encoding="utf-8")
    monkeypatch.setattr(
        cpg_release,
        "CPG_REFERENCE_RUNTIME_SHA256",
        {"runtime.py": sha256_file(source)},
    )

    manifest = validate_cpg_reference_source(root)

    assert manifest["git_commit"] is None
    assert manifest["git_commit_verified"] is False
    assert manifest["runtime_files_verified"] is True
    assert manifest["verification"] == "pinned_runtime_file_hashes"

    source.write_text("PINNED = False\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="runtime.py"):
        validate_cpg_reference_source(root)


def test_target_hash_matches_release_float32_order():
    group = _fake_group()
    targets = _primitive_targets(group)

    assert hash_cpg_group_targets(group) == hash_rollout_targets(targets)

    changed = targets.copy()
    changed[-1, -1, -1] += np.float32(1e-3)
    assert hash_cpg_group_targets(group) != hash_rollout_targets(changed)


def test_trajectory_fingerprint_records_graph_and_parameter_contract():
    group = _fake_group()

    fingerprint = fingerprint_cpg_trajectory(group, trajectory_key="case-a")
    pos, edges, node_type, mach = cpg_graph_frame_metadata(group)

    assert fingerprint.trajectory_key == "case-a"
    assert fingerprint.num_time_steps == 4
    assert fingerprint.num_nodes == 5
    assert fingerprint.num_edges == 4
    assert fingerprint.target_steps == 3
    assert fingerprint.mach_min == pytest.approx(2.25)
    assert fingerprint.mach_max == pytest.approx(2.25)
    assert pos.shape == (5, 2)
    assert edges.shape == (4, 2)
    assert node_type.shape == (5,)
    assert mach.shape == (5,)


def test_release_metrics_separate_truth_clamped_boundary_nodes():
    targets = np.ones((2, 4, 4), dtype=np.float32)
    predictions = targets.copy()
    predictions[:, -1, 0] += 2.0
    node_type = np.array([0, 0, 0, 1], dtype=np.int64)

    metrics = release_rollout_metrics(predictions, targets, node_type)

    assert metrics["all"]["rollout_rmse"][0] == pytest.approx(1.0)
    assert metrics["normal"]["rollout_rmse"] == pytest.approx([0.0] * 4)
    assert metrics["boundary"]["rollout_rmse"][0] == pytest.approx(2.0)
    assert metrics["wall"]["num_nodes"] == 1
    assert metrics["outflow"]["rollout_rmse"] is None


def test_graph_distance_is_strict_and_reports_distance_rmse():
    edges = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int64)
    distance = graph_distance_from_sources(edges, [False, False, False, True])
    assert distance.tolist() == [3, 2, 1, 0]

    targets = np.zeros((1, 4, 4), dtype=np.float32)
    predictions = targets.copy()
    predictions[0, :, 0] = np.arange(4, dtype=np.float32)
    rows = rollout_rmse_by_graph_distance(predictions, targets, distance)
    assert [row["distance"] for row in rows] == [0, 1, 2, 3]
    assert rows[0]["rollout_rmse"][0] == pytest.approx(3.0)

    with pytest.raises(ValueError, match="outside the graph"):
        graph_distance_from_sources([[0, 4]], [False, False, False, True])


def test_result_audit_matches_targets_not_numeric_filename():
    module = _load_audit_module()
    root = Path("artifacts") / "time_dependent_no" / "cpg_release_fixture"
    root.mkdir(parents=True, exist_ok=True)
    test_file = root / "test.h5"
    result_dir = root / "result"
    result_dir.mkdir(parents=True, exist_ok=True)
    for stale in result_dir.glob("*.h5"):
        stale.unlink()
    checkpoint = root / "simulator.pth"
    checkpoint.write_bytes(b"checkpoint fixture")
    checkpoint_sha256 = sha256_file(checkpoint)

    group_data = _fake_group()
    with h5py.File(test_file, "w") as handle:
        group = handle.create_group("non_numeric_case")
        for key, value in group_data.items():
            group.create_dataset(key, data=value)

    targets = _primitive_targets(group_data)
    with h5py.File(result_dir / "9.h5", "w") as handle:
        handle.create_dataset("predicteds", data=targets)
        handle.create_dataset("targets", data=targets)
        handle.attrs["trajectory_key"] = "non_numeric_case"
        handle.attrs["checkpoint_sha256"] = checkpoint_sha256
        handle.attrs["reference_verification"] = "pinned_runtime_file_hashes"

    with h5py.File(test_file, "r") as handle:
        fingerprints = [
            fingerprint_cpg_trajectory(
                handle["non_numeric_case"], trajectory_key="non_numeric_case"
            )
        ]
    rows, violations = module.audit_results(
        result_dir,
        test_file,
        fingerprints,
        checkpoint_sha256,
        expected_results=1,
    )

    assert violations == []
    assert len(rows) == 1
    assert rows[0]["trajectory_key"] == "non_numeric_case"
    assert rows[0]["checkpoint_bound"] is True
    assert rows[0]["metrics"]["all"]["rollout_rmse"] == pytest.approx([0.0] * 4)

    evaluator_sources = {"evaluate.py": "source-digest"}
    reference = {"runtime_file_sha256": {"runtime.py": "runtime-digest"}}
    dataset = {
        "file_sha256": "dataset-digest",
        "hdf5_iteration_keys": ["non_numeric_case"],
    }
    expected_attributes = {
        "boundary_mode": module.RELEASE_BOUNDARY_MODE,
        "seed": 0,
        "split": "test",
        "dt": 0.025,
    }
    run_payload = {
        "schema": module.EVALUATOR_SCHEMA,
        "boundary_mode": module.RELEASE_BOUNDARY_MODE,
        "seed": 0,
        "reference": {
            "expected_commit": module.CPG_REFERENCE_COMMIT,
            "runtime_file_sha256": reference["runtime_file_sha256"],
        },
        "evaluator": {"source_sha256": evaluator_sources},
        "checkpoint": {
            "sha256": checkpoint_sha256,
            "binding": "explicit Simulator.load_checkpoint argument",
        },
        "dataset": {
            "sha256": dataset["file_sha256"],
            "split": "test",
            "hdf5_iteration_keys": dataset["hdf5_iteration_keys"],
            "selected_keys": ["non_numeric_case"],
        },
        "model_configuration": {
            "dt": 0.025,
            "message_passing_num": 12,
            "node_input_size": 6,
            "edge_input_size": 5,
        },
        "aggregate": {
            "post_all_rollout_rmse": [0.0] * 4,
            "post_normal_rollout_rmse": [0.0] * 4,
        },
        "trajectories": [
            {
                "result_file": "9.h5",
                "result_sha256": rows[0]["result_sha256"],
                "trajectory_key": "non_numeric_case",
            }
        ],
    }
    run_manifest = root / "run_manifest.json"
    run_manifest.write_text(json.dumps(run_payload), encoding="utf-8")

    summary, run_violations = module.audit_run_manifest(
        run_manifest,
        reference=reference,
        evaluator_sources=evaluator_sources,
        checkpoint_sha256=checkpoint_sha256,
        dataset=dataset,
        expected_attributes=expected_attributes,
        results=rows,
    )

    assert run_violations == []
    assert summary["cross_checked"] is True

    run_payload["trajectories"][0]["result_sha256"] = "wrong"
    run_manifest.write_text(json.dumps(run_payload), encoding="utf-8")
    _, run_violations = module.audit_run_manifest(
        run_manifest,
        reference=reference,
        evaluator_sources=evaluator_sources,
        checkpoint_sha256=checkpoint_sha256,
        dataset=dataset,
        expected_attributes=expected_attributes,
        results=rows,
    )
    assert any("result_sha256" in violation for violation in run_violations)
