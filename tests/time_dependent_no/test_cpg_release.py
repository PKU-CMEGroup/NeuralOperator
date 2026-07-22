from __future__ import annotations

import copy
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

import utility.time_dependent_no.cpg_release as cpg_release  # noqa: E402
from utility.time_dependent_no.cpg_reach import normalize_scale_positions  # noqa: E402
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


def _legal_training_manifest(module, checkpoint_sha256: str) -> dict:
    evaluation_dataset_sha256 = "e" * 64
    local_source_sha256 = dict(
        cpg_release.CPG_ARCHIVAL_LOCAL_TRAINING_SOURCE_SHA256
    )
    return {
        "schema": module.LEGAL_TRAINING_SCHEMA,
        "status": "complete",
        "claim_scope": "release-bundle legal training fixture",
        "boundary_mode": module.LEGAL_BOUNDARY_MODE,
        "uses_future_reference_boundary": False,
        "uses_future_reference_boundary_in_rollout_inputs_or_recurrent_state": False,
        "future_reference_boundary_training_use": {
            "supervised_target": True,
            "output_normalizer_statistics": True,
            "model_input": False,
            "recurrent_state": False,
        },
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
            "dt": cpg_release.CPG_MODEL_DT,
        },
        "training_configuration": {
            "seed": 7,
            "microbatch_size": 1,
            "gradient_accumulation_steps": 3,
            "effective_batch_size": 3,
            "num_steps": 3,
            "stage1_epochs": 15,
            "stage2_epochs": 5,
            "max_trajectories": None,
            "max_batches_per_epoch": None,
            "checkpoint_selection": "final completed epoch",
        },
        "promotion_eligible": True,
        "reference": {
            "expected_commit": cpg_release.CPG_REFERENCE_COMMIT,
            "runtime_pin_schema": cpg_release.CPG_REFERENCE_RUNTIME_PIN_SCHEMA,
            "runtime_files_verified": True,
            "runtime_file_sha256": dict(
                cpg_release.CPG_REFERENCE_RUNTIME_SHA256
            ),
            "git_commit_verified": False,
        },
        "source_sha256": local_source_sha256,
        "source_provenance": {
            "schema": cpg_release.CPG_LOCAL_TRAINING_SOURCE_SCHEMA,
            "git_commit": cpg_release.CPG_LEGACY_LOCAL_TRAINING_SOURCE_COMMIT,
            "tracked_files_clean": True,
            "file_hash_semantics": "git_blob_sha256",
            "file_sha256": local_source_sha256,
            "relative_paths": dict(cpg_release.CPG_LOCAL_TRAINING_SOURCE_FILES),
        },
        "graph_policy_validation": {
            "schema": cpg_release.CPG_MESH_AUDIT_SCHEMA,
            "status": "valid",
            "graph_mesh_identity": "verified_all_selected_cases",
            "boundary_geometry": "verified_all_selected_cases",
            "contract_completeness": (
                "hardened_sampled_alignment_and_all_frame_static_graph"
            ),
            "case_count": 20,
            "sha256": "1" * 64,
            "dataset_sha256": evaluation_dataset_sha256,
            "max_boundary_source_hops": (
                cpg_release.CPG_LEGAL_BOUNDARY_MAX_SOURCE_HOPS
            ),
        },
    }


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
    payload = _legal_training_manifest(module, checkpoint_sha256)
    path.write_text(json.dumps(payload), encoding="utf-8")

    result = module.load_checkpoint_training_manifest(
        path,
        checkpoint_sha256=checkpoint_sha256,
        expected_evaluation_dataset_sha256="e" * 64,
    )
    assert result["uses_future_reference_boundary"] is False
    assert result[
        "uses_future_reference_boundary_in_rollout_inputs_or_recurrent_state"
    ] is False
    assert result["future_reference_boundary_training_use"][
        "output_normalizer_statistics"
    ] is True
    assert result["training_dataset"]["selected_key_count"] == 2
    assert result["training_configuration"]["effective_batch_size"] == 3
    module.validate_trained_mesh_contract(
        result,
        mesh_audit_sha256="1" * 64,
        max_boundary_source_hops=cpg_release.CPG_LEGAL_BOUNDARY_MAX_SOURCE_HOPS,
    )
    with pytest.raises(ValueError, match="mesh-audit SHA256 values differ"):
        module.validate_trained_mesh_contract(
            result,
            mesh_audit_sha256="2" * 64,
            max_boundary_source_hops=(
                cpg_release.CPG_LEGAL_BOUNDARY_MAX_SOURCE_HOPS
            ),
        )
    altered = copy.deepcopy(result)
    altered["graph_policy_validation"]["max_boundary_source_hops"] = 2
    with pytest.raises(ValueError, match="source-hop contracts differ"):
        module.validate_trained_mesh_contract(
            altered,
            mesh_audit_sha256="1" * 64,
            max_boundary_source_hops=(
                cpg_release.CPG_LEGAL_BOUNDARY_MAX_SOURCE_HOPS
            ),
        )
    assert result["reference_provenance"]["completeness"] == "complete"
    assert result["local_source_provenance"]["completeness"] == (
        "git_commit_and_full_local_source_closure"
    )
    assert result["sha256"] == sha256_file(path)

    with pytest.raises(ValueError, match="checkpoint SHA256 differs"):
        module.load_checkpoint_training_manifest(
            path,
            checkpoint_sha256="d" * 64,
            expected_evaluation_dataset_sha256="e" * 64,
        )
    payload["uses_future_reference_boundary"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="future boundaries"):
        module.load_checkpoint_training_manifest(
            path,
            checkpoint_sha256=checkpoint_sha256,
            expected_evaluation_dataset_sha256="e" * 64,
        )


def test_evaluator_accepts_only_the_exact_legacy_runtime_pin_gap(tmp_path: Path):
    module = _load_evaluate_module()
    checkpoint_sha256 = "a" * 64
    path = tmp_path / "training_manifest.json"
    payload = _legal_training_manifest(module, checkpoint_sha256)
    payload["reference"].pop("runtime_pin_schema")
    for name in cpg_release.CPG_LEGACY_TRAINING_PIN_OMISSIONS:
        payload["reference"]["runtime_file_sha256"].pop(name)
    payload["source_sha256"] = dict(
        cpg_release.CPG_LEGACY_LOCAL_TRAINING_SOURCE_SHA256
    )
    payload.pop("source_provenance")
    for name in (
        "schema",
        "status",
        "graph_mesh_identity",
        "boundary_geometry",
    ):
        payload["graph_policy_validation"].pop(name)
    path.write_text(json.dumps(payload), encoding="utf-8")

    result = module.load_checkpoint_training_manifest(
        path,
        checkpoint_sha256=checkpoint_sha256,
        expected_evaluation_dataset_sha256="e" * 64,
    )
    assert result["reference_provenance"]["completeness"] == (
        "legacy_missing_training_runtime_pins"
    )
    assert result["local_source_provenance"]["completeness"] == (
        "legacy_two_recorded_hashes_match_archival_commit"
    )

    altered_legacy = copy.deepcopy(payload)
    altered_legacy["source_sha256"]["trainer"] = "f" * 64
    path.write_text(json.dumps(altered_legacy), encoding="utf-8")
    with pytest.raises(ValueError, match="archived run"):
        module.load_checkpoint_training_manifest(
            path,
            checkpoint_sha256=checkpoint_sha256,
            expected_evaluation_dataset_sha256="e" * 64,
        )

    third = next(
        name
        for name in cpg_release.CPG_REFERENCE_RUNTIME_SHA256
        if name not in cpg_release.CPG_LEGACY_TRAINING_PIN_OMISSIONS
    )
    payload["reference"]["runtime_file_sha256"].pop(third)
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete"):
        module.load_checkpoint_training_manifest(
            path,
            checkpoint_sha256=checkpoint_sha256,
            expected_evaluation_dataset_sha256="e" * 64,
        )


def test_git_commit_closes_exact_legacy_reference_pin_gap(tmp_path: Path):
    module = _load_evaluate_module()
    checkpoint_sha256 = "a" * 64
    path = tmp_path / "training_manifest.json"
    payload = _legal_training_manifest(module, checkpoint_sha256)
    payload["reference"].update(
        {
            "git_commit_verified": True,
            "git_commit": cpg_release.CPG_REFERENCE_COMMIT,
            "tracked_clean": True,
        }
    )
    payload["reference"].pop("runtime_pin_schema")
    for name in cpg_release.CPG_LEGACY_TRAINING_PIN_OMISSIONS:
        payload["reference"]["runtime_file_sha256"].pop(name)
    path.write_text(json.dumps(payload), encoding="utf-8")

    result = module.load_checkpoint_training_manifest(
        path,
        checkpoint_sha256=checkpoint_sha256,
        expected_evaluation_dataset_sha256="e" * 64,
    )

    assert result["reference_provenance"]["completeness"] == (
        "git_commit_closes_legacy_runtime_pin_gap"
    )


def test_evaluator_rejects_incomplete_or_inconsistent_training_provenance(
    tmp_path: Path,
):
    module = _load_evaluate_module()
    checkpoint_sha256 = "a" * 64
    path = tmp_path / "training_manifest.json"
    base = _legal_training_manifest(module, checkpoint_sha256)

    def rejected(mutator, match: str) -> None:
        payload = copy.deepcopy(base)
        mutator(payload)
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match=match):
            module.load_checkpoint_training_manifest(
                path,
                checkpoint_sha256=checkpoint_sha256,
                expected_evaluation_dataset_sha256="e" * 64,
            )

    rejected(
        lambda payload: payload["reference"].update(expected_commit="0" * 40),
        "commit",
    )
    rejected(
        lambda payload: payload["reference"].update(
            runtime_pin_schema="unsupported"
        ),
        "pin schema",
    )
    runtime_file = next(iter(cpg_release.CPG_REFERENCE_RUNTIME_SHA256))
    rejected(
        lambda payload: payload["reference"]["runtime_file_sha256"].update(
            {runtime_file: "0" * 64}
        ),
        "hashes differ",
    )
    rejected(
        lambda payload: payload["model_configuration"].update(dt=0.01),
        "hardcodes dt",
    )
    rejected(
        lambda payload: payload["dataset"].update(sha256="B" * 64),
        "dataset SHA256",
    )
    rejected(
        lambda payload: payload["dataset"].update(selected_keys=["0", "0"]),
        "nonempty and unique",
    )
    rejected(
        lambda payload: payload["dataset"].update(selected_keys=[""]),
        "nonempty and unique",
    )
    rejected(
        lambda payload: payload["training_configuration"].update(num_steps=True),
        "invalid values",
    )
    rejected(
        lambda payload: payload["source_sha256"].update(trainer="invalid"),
        "source hashes",
    )
    rejected(
        lambda payload: payload["source_provenance"].update(git_commit="0" * 40),
        "does not contain",
    )
    rejected(
        lambda payload: payload["source_provenance"]["relative_paths"].update(
            trainer="wrong.py"
        ),
        "required closure",
    )
    rejected(
        lambda payload: payload["graph_policy_validation"].pop("sha256"),
        "graph-policy provenance",
    )
    rejected(
        lambda payload: payload["graph_policy_validation"].pop(
            "boundary_geometry"
        ),
        "identity contract",
    )
    rejected(
        lambda payload: payload["graph_policy_validation"].update(
            dataset_sha256="2" * 64
        ),
        "and evaluation dataset differ",
    )
    rejected(
        lambda payload: payload["training_configuration"].update(
            max_batches_per_epoch=1
        ),
        "identity contract",
    )

    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="root must be an object"):
        module.load_checkpoint_training_manifest(
            path,
            checkpoint_sha256=checkpoint_sha256,
            expected_evaluation_dataset_sha256="e" * 64,
        )


@pytest.mark.parametrize("bad_dt", [0.01, float("nan"), float("inf")])
def test_cpg_cli_parsers_reject_unimplemented_timesteps(
    tmp_path: Path, bad_dt: float
):
    evaluator = _load_evaluate_module()
    evaluator_args = [
        "--reference-repo",
        str(tmp_path),
        "--dataset-root",
        str(tmp_path),
        "--checkpoint",
        str(tmp_path / "checkpoint.pth"),
        "--output-dir",
        str(tmp_path / "output"),
        "--dt",
        str(bad_dt),
    ]
    with pytest.raises(ValueError, match="hardcodes dt"):
        evaluator.parse_args(evaluator_args)

    audit = _load_audit_module()
    audit_args = [
        "--train-file",
        str(tmp_path / "train.h5"),
        "--test-file",
        str(tmp_path / "test.h5"),
        "--result-dir",
        str(tmp_path / "result"),
        "--run-manifest",
        str(tmp_path / "run_manifest.json"),
        "--checkpoint",
        str(tmp_path / "checkpoint.pth"),
        "--reference-repo",
        str(tmp_path),
        "--output-dir",
        str(tmp_path / "audit"),
        "--expected-dt",
        str(bad_dt),
    ]
    with pytest.raises(ValueError, match="hardcodes dt"):
        audit.parse_args(audit_args)


def test_training_runtime_pin_set_includes_direct_dependencies():
    assert cpg_release.CPG_REFERENCE_RUNTIME_SHA256["utils/lossCompute.py"] == (
        "935a1025787d723d9980375275b3c4167ea931ad95b6c4c70b7694fe702ecc42"
    )
    assert cpg_release.CPG_REFERENCE_RUNTIME_SHA256["utils/noise.py"] == (
        "67269a162bf812e3d43da988bc9ce41feba3b083ef204863144b22f59f6ffcb2"
    )


def test_local_source_manifest_uses_git_blob_hashes_for_clean_eol_worktree(
    tmp_path: Path, monkeypatch
):
    source = tmp_path / "source.py"
    source.write_bytes(b"value = 1\r\n")
    commit = "a" * 40
    blob_digest = sha256(b"value = 1\n").hexdigest()
    raw_digest = sha256_file(source)

    monkeypatch.setattr(
        cpg_release, "CPG_LOCAL_TRAINING_SOURCE_FILES", {"trainer": "source.py"}
    )

    def fake_git_output(_root, *args):
        if args == ("rev-parse", "HEAD"):
            return commit
        if args[:3] == ("status", "--short", "--"):
            return ""
        raise AssertionError(args)

    monkeypatch.setattr(cpg_release, "_git_output", fake_git_output)
    monkeypatch.setattr(
        cpg_release,
        "_git_blob_sha256",
        lambda _root, _commit, _path: blob_digest,
    )

    manifest = cpg_release.cpg_local_training_source_manifest(tmp_path)
    validated = cpg_release.validate_cpg_local_training_source_manifest(
        manifest, repo_root=tmp_path
    )

    assert manifest["file_hash_semantics"] == "git_blob_sha256"
    assert manifest["file_sha256"]["trainer"] == blob_digest
    assert manifest["working_tree_sha256"]["trainer"] == raw_digest
    assert raw_digest != blob_digest
    assert validated["file_sha256"]["trainer"] == blob_digest


def test_cpg_mach_range_rejects_mixed_finite_and_nonfinite_values():
    group = _fake_group()
    group["Mach"][1, 0, 0] = np.nan

    with pytest.raises(ValueError, match="nonfinite"):
        cpg_release.cpg_mach_range(group)


def test_reference_source_hash_fallback_is_explicit_and_strict(
    tmp_path: Path, monkeypatch
):
    root = tmp_path / "cpg_reference_fixture"
    root.mkdir()
    source = root / "runtime.py"
    source.write_text("PINNED = True\n", encoding="utf-8")
    monkeypatch.setattr(
        cpg_release,
        "CPG_REFERENCE_EVALUATION_RUNTIME_SHA256",
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


def test_reference_validation_separates_evaluation_and_training_pins(
    tmp_path: Path, monkeypatch
):
    runtime = tmp_path / "runtime.py"
    runtime.write_text("RUNTIME = True\n", encoding="utf-8")
    monkeypatch.setattr(
        cpg_release,
        "CPG_REFERENCE_EVALUATION_RUNTIME_SHA256",
        {"runtime.py": sha256_file(runtime)},
    )
    monkeypatch.setattr(
        cpg_release,
        "CPG_REFERENCE_RUNTIME_SHA256",
        {
            "runtime.py": sha256_file(runtime),
            "training.py": "0" * 64,
        },
    )

    result = validate_cpg_reference_source(tmp_path)
    assert result["runtime_scope"] == "evaluation_only"
    with pytest.raises(RuntimeError, match="missing training.py"):
        validate_cpg_reference_source(
            tmp_path,
            include_training_dependencies=True,
        )


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

    nonfinite = predictions.copy()
    nonfinite[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        release_rollout_metrics(nonfinite, targets, node_type)
    with pytest.raises(ValueError, match="finite"):
        rollout_rmse_by_graph_distance(
            nonfinite,
            targets,
            np.arange(targets.shape[1]),
        )


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


def test_legal_outflow_characteristic_diagnostic_detects_subsonic_state():
    torch = pytest.importorskip("torch")
    module = _load_evaluate_module()
    policy = {
        "target_nodes": torch.tensor([1]),
        "outflow_rows": torch.tensor([0]),
        "target_normals": torch.tensor([[1.0, 0.0]]),
        "gamma": 1.4,
    }
    supersonic = torch.tensor(
        [[1.0, 0.0, 0.0, 1.0], [1.4, 2.5, 0.0, 1.0]]
    )
    subsonic = supersonic.clone()
    subsonic[1, 1] = 0.5

    assert module.legal_outflow_normal_mach_min(
        torch, supersonic, policy
    ) > 1.0
    assert module.legal_outflow_normal_mach_min(
        torch, subsonic, policy
    ) < 1.0


def test_failed_rollout_excludes_terminal_nonphysical_state(monkeypatch):
    torch = pytest.importorskip("torch")
    module = _load_evaluate_module()

    class Graph:
        def __init__(
            self,
            current: float,
            target: float,
            node_types: tuple[int, int] = (0, 0),
        ):
            self.x = torch.tensor(
                [
                    [node_types[0], current, 0.0, 0.0, 1.0, 2.5],
                    [node_types[1], current, 0.0, 0.0, 1.0, 2.5],
                ]
            )
            self.y = torch.tensor(
                [[target, 0.0, 0.0, 1.0], [target, 0.0, 0.0, 1.0]]
            )
            self.pos = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
            self.edge_index = torch.tensor([[0, 1], [1, 0]])
            self.edge_attr = torch.tensor([[1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]])

        def to(self, _device):
            return self

    outputs = [
        torch.tensor([[1.1, 0.0, 0.0, 1.0], [1.1, 0.0, 0.0, 1.0]]),
        torch.tensor([[-0.1, 0.0, 0.0, 1.0], [-0.1, 0.0, 0.0, 1.0]]),
    ]

    class Model:
        def __init__(self):
            self.index = 0

        def __call__(self, _graph, sequence_noise=None):
            assert sequence_noise is None
            value = outputs[self.index]
            self.index += 1
            return value

    dataset = SimpleNamespace(cur_targecity_length=4)
    dataset.change_file = lambda _index: None
    monkeypatch.setattr(
        module,
        "apply_torch_boundary_policy",
        lambda _torch, state, _policy: state,
    )
    rollout = module.rollout_trajectory(
        api={
            "torch": torch,
            "make_edges_undirected": lambda graph: graph,
        },
        model=Model(),
        dataset=dataset,
        dataloader=[Graph(1.0, 1.1), Graph(1.1, 1.2)],
        transformer=lambda graph: graph,
        trajectory_index=0,
        device=torch.device("cpu"),
        boundary_mode=module.LEGAL_BOUNDARY_MODE,
        boundary_policy={},
    )

    termination = rollout["termination"]
    assert termination == {
        "completed": False,
        "attempted_steps": 2,
        "admissible_steps": 1,
        "valid_steps": 1,
        "expected_steps": 3,
        "failure_step": 1,
        "failure_step_one_based": 2,
        "failure_reason": "nonpositive_recurrent_density",
        "terminal_failure_included_in_arrays": True,
    }
    assert rollout["arrays"]["predicteds"].shape[0] == 2

    class FixedModel:
        def __call__(self, _graph, sequence_noise=None):
            assert sequence_noise is None
            return torch.tensor(
                [[1.1, 0.0, 0.0, 1.0], [float("nan"), 0.0, 0.0, 1.0]]
            )

    dataset.cur_targecity_length = 2
    oracle = module.rollout_trajectory(
        api={
            "torch": torch,
            "make_edges_undirected": lambda graph: graph,
        },
        model=FixedModel(),
        dataset=dataset,
        dataloader=[Graph(1.0, 1.1, node_types=(0, 1))],
        transformer=lambda graph: graph,
        trajectory_index=0,
        device=torch.device("cpu"),
        boundary_mode=module.ORACLE_BOUNDARY_MODE,
    )
    assert oracle["termination"]["failure_reason"] == "nonfinite_raw_prediction"
    assert oracle["termination"]["admissible_steps"] == 0
    assert np.isfinite(oracle["arrays"]["predicteds"]).all()

    class NegativeOracleModel:
        def __call__(self, _graph, sequence_noise=None):
            assert sequence_noise is None
            return torch.tensor(
                [[-0.1, 0.0, 0.0, 1.0], [1.1, 0.0, 0.0, 1.0]]
            )

    negative_oracle = module.rollout_trajectory(
        api={
            "torch": torch,
            "make_edges_undirected": lambda graph: graph,
        },
        model=NegativeOracleModel(),
        dataset=dataset,
        dataloader=[Graph(1.0, 1.1, node_types=(0, 1))],
        transformer=lambda graph: graph,
        trajectory_index=0,
        device=torch.device("cpu"),
        boundary_mode=module.ORACLE_BOUNDARY_MODE,
    )
    assert negative_oracle["termination"]["failure_reason"] == (
        "nonpositive_recurrent_density"
    )
    assert negative_oracle["termination"]["admissible_steps"] == 0


def test_primary_aggregate_requires_complete_equal_horizon():
    module = _load_evaluate_module()

    def row(*, completed: bool, admissible: int, expected: int, value: float):
        result = {
            "completed": completed,
            "admissible_steps": admissible,
            "expected_steps": expected,
        }
        for prefix in ("post_all", "post_normal", "raw_boundary"):
            for name in module.PRIMITIVE_NAMES:
                result[f"{prefix}_{name}_rollout_rmse"] = value
        return result

    complete = module._primary_aggregate(
        [
            row(completed=True, admissible=3, expected=3, value=1.0),
            row(completed=True, admissible=3, expected=3, value=2.0),
        ]
    )
    assert complete["primary_metric_status"] == "complete_equal_horizon"
    assert complete["post_all_rollout_rmse"] == pytest.approx([1.5] * 4)

    incomplete = module._primary_aggregate(
        [row(completed=False, admissible=2, expected=3, value=1.0)]
    )
    assert incomplete["post_all_rollout_rmse"] is None

    unequal = module._primary_aggregate(
        [
            row(completed=True, admissible=2, expected=2, value=1.0),
            row(completed=True, admissible=3, expected=3, value=1.0),
        ]
    )
    assert unequal["primary_metric_status"] == (
        "unavailable_incomplete_or_unequal_horizon"
    )


def test_result_audit_matches_targets_not_numeric_filename(tmp_path: Path):
    module = _load_audit_module()
    root = tmp_path / "cpg_release_fixture"
    root.mkdir()
    test_file = root / "test.h5"
    result_dir = root / "result"
    result_dir.mkdir()
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
        handle.create_dataset("raw_predicteds", data=targets)
        handle.attrs["trajectory_key"] = "non_numeric_case"
        handle.attrs["checkpoint_sha256"] = checkpoint_sha256
        handle.attrs["reference_verification"] = "pinned_runtime_file_hashes"
        handle.attrs["completed"] = True
        handle.attrs["attempted_steps"] = targets.shape[0]
        handle.attrs["admissible_steps"] = targets.shape[0]
        handle.attrs["valid_steps"] = targets.shape[0]
        handle.attrs["expected_steps"] = targets.shape[0]
        handle.attrs["failure_step"] = -1
        handle.attrs["failure_step_one_based"] = -1
        handle.attrs["failure_reason"] = ""
        handle.attrs["terminal_failure_included_in_arrays"] = False
        handle.attrs["metric_scope"] = (
            "admissible_prefix_excluding_terminal_failure"
        )
        handle.attrs["termination_accounting_schema"] = (
            cpg_release.CPG_TERMINATION_ACCOUNTING_SCHEMA
        )

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
        expected_termination_accounting_schema=(
            cpg_release.CPG_TERMINATION_ACCOUNTING_SCHEMA
        ),
    )

    assert violations == []
    assert len(rows) == 1
    assert rows[0]["trajectory_key"] == "non_numeric_case"
    assert rows[0]["checkpoint_bound"] is True
    assert rows[0]["metrics"]["all"]["rollout_rmse"] == pytest.approx([0.0] * 4)

    evaluator_sources = {"evaluate.py": "source-digest"}
    reference = {
        "runtime_file_sha256": {"runtime.py": "runtime-digest"},
        "runtime_pin_schema": "fixture-evaluation-pin-v1",
        "runtime_scope": "evaluation_only",
        "git_commit": None,
        "git_commit_verified": False,
        "tracked_clean": None,
        "verification": "pinned_runtime_file_hashes",
    }
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
            **reference,
        },
        "evaluator": {
            "source_sha256": evaluator_sources,
            "termination_accounting_schema": (
                cpg_release.CPG_TERMINATION_ACCOUNTING_SCHEMA
            ),
        },
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
            "raw_boundary_rollout_rmse": [0.0] * 4,
            "primary_metric_status": "complete_equal_horizon",
            "completed_trajectories": 1,
            "minimum_valid_steps": targets.shape[0],
            "maximum_attempted_steps": targets.shape[0],
            "expected_steps": [targets.shape[0]],
        },
        "trajectories": [
            {
                "result_file": "9.h5",
                "result_sha256": rows[0]["result_sha256"],
                "trajectory_key": "non_numeric_case",
                "termination": rows[0]["termination"],
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

    run_payload["aggregate"]["completed_trajectories"] = 0
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
    assert any("completed_trajectories" in item for item in run_violations)
    run_payload["aggregate"]["completed_trajectories"] = 1

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

    altered_targets = targets.astype(np.float64)
    altered_targets[0, 0, 0] += 1.0e-12
    assert hash_rollout_targets(altered_targets) == hash_rollout_targets(targets)
    with h5py.File(result_dir / "9.h5", "w") as handle:
        handle.create_dataset("predicteds", data=targets)
        handle.create_dataset("targets", data=altered_targets)
        handle.attrs["trajectory_key"] = "non_numeric_case"
        handle.attrs["checkpoint_sha256"] = checkpoint_sha256
        handle.attrs["reference_verification"] = "pinned_runtime_file_hashes"
    rows, violations = module.audit_results(
        result_dir,
        test_file,
        fingerprints,
        checkpoint_sha256,
        expected_results=1,
    )
    assert rows == []
    assert any("targets are not canonical float32" in item for item in violations)


def test_result_audit_links_saved_oracle_instrumentation_to_dataset(
    tmp_path: Path,
):
    module = _load_audit_module()
    test_file = tmp_path / "test.h5"
    result_dir = tmp_path / "result"
    result_dir.mkdir()
    checkpoint = tmp_path / "simulator.pth"
    checkpoint.write_bytes(b"checkpoint fixture")
    checkpoint_sha256 = sha256_file(checkpoint)

    group_data = _fake_group()
    with h5py.File(test_file, "w") as handle:
        group = handle.create_group("case")
        for key, value in group_data.items():
            group.create_dataset(key, data=value)
    targets = _primitive_targets(group_data)
    current = np.concatenate(
        [
            np.asarray(group_data[key][:-1], dtype=np.float32)
            for key in ("rho", "v1", "v2", "pres")
        ],
        axis=-1,
    )
    pos = np.asarray(group_data["pos"][0], dtype=np.float32)
    edges = np.asarray(group_data["edges"][0], dtype=np.int64)
    node_type = np.asarray(group_data["node_type"][0]).reshape(-1).astype(np.int64)
    mach = np.asarray(group_data["Mach"][0], dtype=np.float32).reshape(-1)
    boundary = node_type != 0
    model_inputs = current.copy()
    model_inputs[:, boundary] = targets[:, boundary]
    model_pos = normalize_scale_positions(pos)
    directed = np.concatenate((edges, edges[:, ::-1]), axis=0)
    displacement = model_pos[directed[:, 0]] - model_pos[directed[:, 1]]
    edge_attr = np.column_stack(
        (displacement, np.linalg.norm(displacement, axis=1))
    )

    result_path = result_dir / "1.h5"
    with h5py.File(result_path, "w") as handle:
        arrays = {
            "predicteds": targets,
            "targets": targets,
            "raw_predicteds": targets,
            "reference_current": current,
            "inputs_before_boundary": current,
            "model_inputs": model_inputs,
            "model_mach": np.broadcast_to(mach, (targets.shape[0], mach.size)),
            "injection_mask": boundary,
            "boundary_node_mask": boundary,
            "model_pos": model_pos,
            "directed_edges": directed,
            "edge_attr_before_model": edge_attr,
            "pos": pos,
            "edges": edges,
            "node_type": node_type,
            "Mach": mach,
            "boundary_graph_distance": graph_distance_from_sources(edges, boundary),
        }
        for name, value in arrays.items():
            handle.create_dataset(name, data=value)
        handle.attrs["trajectory_key"] = "case"
        handle.attrs["checkpoint_sha256"] = checkpoint_sha256
        handle.attrs["reference_verification"] = "pinned_runtime_file_hashes"

    with h5py.File(test_file, "r") as handle:
        fingerprints = [
            fingerprint_cpg_trajectory(handle["case"], trajectory_key="case")
        ]
    rows, violations = module.audit_results(
        result_dir,
        test_file,
        fingerprints,
        checkpoint_sha256,
        expected_results=1,
        required_datasets=module.REQUIRED_INSTRUMENTATION,
    )
    assert violations == []
    assert rows[0]["artifact_linkage"]["valid"] is True
    assert rows[0]["artifact_linkage"]["model_graph_mapping"][
        "dataset_node_to_model_node"
    ] == "verified_index_identity"

    with h5py.File(result_path, "r+") as handle:
        handle["model_inputs"][0, -1, 0] += 1.0
    _, violations = module.audit_results(
        result_dir,
        test_file,
        fingerprints,
        checkpoint_sha256,
        expected_results=1,
        required_datasets=module.REQUIRED_INSTRUMENTATION,
    )
    assert any("next-reference injection" in item for item in violations)

    with h5py.File(result_path, "r+") as handle:
        handle["model_inputs"][0, -1, 0] -= 1.0
        handle["raw_predicteds"][0, 0, 0] += 1.0
    _, violations = module.audit_results(
        result_dir,
        test_file,
        fingerprints,
        checkpoint_sha256,
        expected_results=1,
        required_datasets=module.REQUIRED_INSTRUMENTATION,
    )
    assert any("changes normal-node predictions" in item for item in violations)
