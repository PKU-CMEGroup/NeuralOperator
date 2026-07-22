"""Run an explicitly bound frozen CPGNet boundary evaluator.

The public ``rollout.py`` ignores its ``--model_dir`` argument and saves only
predictions and targets.  This entry point keeps the public model code external,
loads the requested checkpoint explicitly, reproduces the released oracle
boundary contract, and writes enough state and provenance to audit the result.

The default ``oracle_next_reference`` mode exactly reproduces the released
future-boundary injection.  ``causal_nodal_physical`` is available only when a
completed mesh audit binds each HDF graph to its extracted Abaqus/VTU case.  It
uses fixed freestream inflow, current-interior slip-wall projection, and
current-interior zero-gradient outflow.  Without a bound legal-training
manifest, that mode is a frozen sensitivity counterfactual.  With one, it is a
release-bundle train/evaluate comparison under the causal nodal contract.  It
is never an exact Trixi DG surface-flux replay or evidence of paper identity.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import random
import sys
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utility.time_dependent_no.cpg_release import (  # noqa: E402
    CPG_ARCHIVAL_LOCAL_TRAINING_SOURCE_SHA256,
    CPG_EVALUATOR_SOURCE_FILES,
    CPG_LEGACY_MESH_AUDIT_SCHEMA,
    CPG_LEGACY_LOCAL_TRAINING_SOURCE_COMMIT,
    CPG_LEGACY_LOCAL_TRAINING_SOURCE_SHA256,
    CPG_LEGACY_TRAINING_PIN_OMISSIONS,
    CPG_LEGAL_BOUNDARY_MAX_SOURCE_HOPS,
    CPG_LOCAL_TRAINING_SOURCE_FILES,
    CPG_LOCAL_TRAINING_SOURCE_SCHEMA,
    CPG_MESH_AUDIT_SCHEMA,
    CPG_MODEL_DT,
    CPG_REFERENCE_COMMIT,
    CPG_REFERENCE_RUNTIME_PIN_SCHEMA,
    CPG_REFERENCE_RUNTIME_SHA256,
    CPG_TERMINATION_ACCOUNTING_SCHEMA,
    cpg_evaluator_source_manifest,
    graph_distance_from_sources,
    release_rollout_metrics,
    rollout_rmse_by_graph_distance,
    sha256_file,
    validate_cpg_local_training_source_manifest,
    validate_cpg_reference_source,
    validate_cpg_model_dt,
    validate_cpg_static_graph,
)
from utility.time_dependent_no.cpg_mesh_contract import (  # noqa: E402
    WALL_NODE,
    apply_torch_boundary_policy,
    audit_bump_julia_config,
    boundary_stencil_sha256,
    build_boundary_stencil,
    build_torch_boundary_policy,
    legal_outflow_normal_mach_min,
    parse_abaqus_mesh,
    validate_hdf_mesh_identity,
)
from utility.time_dependent_no.euler2d import (  # noqa: E402
    EulerNodeType,
    PRIMITIVE_NAMES,
)


ORACLE_BOUNDARY_MODE = "oracle_next_reference"
LEGAL_BOUNDARY_MODE = "causal_nodal_physical"
BOUNDARY_MODE = ORACLE_BOUNDARY_MODE
RUN_SCHEMA = "cpg_frozen_release_evaluation_v1"
LEGAL_RUN_SCHEMA = "cpg_frozen_legal_boundary_evaluation_v1"
LEGAL_TRAINED_RUN_SCHEMA = "cpg_legal_trained_boundary_evaluation_v1"
MESH_AUDIT_SCHEMA = CPG_MESH_AUDIT_SCHEMA
LEGAL_TRAINING_SCHEMA = "cpg_legal_boundary_training_v1"
TERMINATION_ACCOUNTING_SCHEMA = CPG_TERMINATION_ACCOUNTING_SCHEMA


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-repo", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--trajectory-key",
        action="append",
        default=[],
        help="Explicit HDF5 key; repeat to evaluate multiple trajectories.",
    )
    parser.add_argument(
        "--max-trajectories",
        type=int,
        help="Evaluate the first N release-order trajectories for a sanity run.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dt", type=float, default=CPG_MODEL_DT)
    parser.add_argument(
        "--boundary-mode",
        choices=(ORACLE_BOUNDARY_MODE, LEGAL_BOUNDARY_MODE),
        default=ORACLE_BOUNDARY_MODE,
    )
    parser.add_argument(
        "--mesh-audit-summary",
        type=Path,
        help="Required all-case mesh audit summary for causal nodal boundaries",
    )
    parser.add_argument(
        "--raw-case-root",
        type=Path,
        help="Required extracted one-based bump case root for causal boundaries",
    )
    parser.add_argument(
        "--checkpoint-training-manifest",
        type=Path,
        help=(
            "Completed legal-boundary training manifest to bind checkpoint "
            "provenance; valid only with causal_nodal_physical"
        ),
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if args.trajectory_key and args.max_trajectories is not None:
        raise ValueError("use either --trajectory-key or --max-trajectories")
    if args.max_trajectories is not None and args.max_trajectories < 1:
        raise ValueError("--max-trajectories must be positive")
    args.dt = validate_cpg_model_dt(args.dt)
    if args.boundary_mode == LEGAL_BOUNDARY_MODE:
        if args.mesh_audit_summary is None or args.raw_case_root is None:
            raise ValueError(
                "causal_nodal_physical requires --mesh-audit-summary and "
                "--raw-case-root"
            )
        if args.split != "test":
            raise ValueError("the recovered raw-case contract is currently test-only")
    elif args.checkpoint_training_manifest is not None:
        raise ValueError(
            "--checkpoint-training-manifest requires causal_nodal_physical"
        )
    return args


def _import_reference_api(reference_repo: Path) -> dict[str, Any]:
    try:
        import h5py  # type: ignore[import-not-found]
        import torch
        from torch_geometric.loader import DataLoader
        import torch_geometric.transforms as T
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "the reference evaluator requires h5py, torch, and torch-geometric"
        ) from exc
    reference = str(reference_repo.resolve())
    if reference not in sys.path:
        sys.path.insert(0, reference)
    try:
        from dataset.fpcMulti import FPC_ROLLOUT
        from modelEdgeUpd.simulator import Simulator
        from utils.to_undirected import make_edges_undirected
    except ModuleNotFoundError as exc:
        raise RuntimeError("the pinned external CPGNet checkout is incomplete") from exc

    return {
        "h5py": h5py,
        "torch": torch,
        "DataLoader": DataLoader,
        "T": T,
        "FPC_ROLLOUT": FPC_ROLLOUT,
        "Simulator": Simulator,
        "make_edges_undirected": make_edges_undirected,
    }


def _select_device(torch: Any, name: str, gpu: int) -> Any:
    use_cuda = name == "cuda" or (name == "auto" and torch.cuda.is_available())
    if use_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        torch.cuda.set_device(gpu)
        return torch.device("cuda", gpu)
    return torch.device("cpu")


def _set_seed(torch: Any, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_trajectory_keys(
    available: Sequence[str],
    requested: Sequence[str],
    max_trajectories: int | None,
) -> list[str]:
    keys = [str(key) for key in available]
    if requested:
        missing = [key for key in requested if key not in keys]
        if missing:
            raise KeyError(f"requested trajectories are absent: {missing}")
        if len(set(requested)) != len(requested):
            raise ValueError("trajectory keys must be unique")
        return [str(key) for key in requested]
    if max_trajectories is None:
        return keys
    return keys[:max_trajectories]


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _runtime_manifest() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "h5py": _package_version("h5py"),
        "torch": _package_version("torch"),
        "torch_geometric": _package_version("torch-geometric"),
    }


def _stack(items: list[np.ndarray], name: str) -> np.ndarray:
    if not items:
        raise RuntimeError(f"rollout produced no {name}")
    return np.stack(items, axis=0)


def load_mesh_audit(
    path: Path,
    *,
    dataset_sha256: str,
    selected_keys: Sequence[str],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("mesh audit root must be an object")
    mesh_schema = payload.get("schema")
    if mesh_schema not in {MESH_AUDIT_SCHEMA, CPG_LEGACY_MESH_AUDIT_SCHEMA}:
        raise ValueError("mesh audit has an unsupported schema")
    if payload.get("status") != "valid":
        raise ValueError("mesh audit did not pass its shared-contract gate")
    if payload.get("dataset_sha256") != dataset_sha256:
        raise ValueError("mesh audit and evaluator dataset SHA256 differ")
    if payload.get("graph_mesh_identity") != "verified_all_selected_cases":
        raise ValueError("mesh audit does not verify graph-to-mesh identity")
    if payload.get("boundary_geometry") != "verified_all_selected_cases":
        raise ValueError("mesh audit does not verify boundary geometry")
    if payload.get("case_offset") != 1:
        raise ValueError("mesh audit does not use the one-based raw-case mapping")
    max_source_hops = payload.get("max_boundary_source_hops")
    if (
        not isinstance(max_source_hops, int)
        or isinstance(max_source_hops, bool)
        or max_source_hops != CPG_LEGAL_BOUNDARY_MAX_SOURCE_HOPS
    ):
        raise ValueError("mesh audit uses a different boundary source-hop contract")
    is_legacy = mesh_schema == CPG_LEGACY_MESH_AUDIT_SCHEMA
    if not is_legacy and (
        payload.get("temporal_alignment_evidence")
        != "sampled_first_midpoint_last_not_full_temporal_identity"
        or payload.get("static_graph_evidence")
        != "all_frames_exact_release_facing_metadata"
    ):
        raise ValueError("mesh audit lacks hardened temporal/static evidence")
    case_records = payload.get("cases")
    if not isinstance(case_records, list):
        raise ValueError("mesh audit cases must be a list")
    records: dict[str, dict[str, Any]] = {}
    for record in case_records:
        if not isinstance(record, dict):
            raise ValueError("mesh audit case records must be objects")
        key = str(record.get("trajectory_key", ""))
        raw_case_id = record.get("raw_case_id")
        if (
            not key
            or not key.isdigit()
            or key in records
            or not isinstance(raw_case_id, int)
            or isinstance(raw_case_id, bool)
            or raw_case_id != int(key) + 1
        ):
            raise ValueError("mesh audit trajectory records are missing or duplicated")
        config = record.get("config")
        if not isinstance(config, Mapping):
            raise ValueError(f"mesh audit trajectory {key!r} lacks configuration")
        try:
            validate_cpg_model_dt(config.get("save_dt"))
        except ValueError as exc:
            raise ValueError(
                f"mesh audit trajectory {key!r} does not use the pinned model dt"
            ) from exc
        if not is_legacy:
            sampled = record.get("temporal_alignment_sampled_hdf_frames")
            if (
                not isinstance(sampled, list)
                or len(sampled) < 2
                or sampled != sorted(set(sampled))
                or sampled[0] != 0
            ):
                raise ValueError(
                    f"mesh audit trajectory {key!r} lacks sampled-frame evidence"
                )
            boundary = record.get("boundary")
            if not isinstance(boundary, Mapping):
                raise ValueError(
                    f"mesh audit trajectory {key!r} lacks causal boundary evidence"
                )
            try:
                causal_outflow_mach_min = float(
                    boundary["causal_outflow_normal_mach_min"]
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"mesh audit trajectory {key!r} lacks causal outflow evidence"
                ) from exc
            if (
                not np.isfinite(causal_outflow_mach_min)
                or causal_outflow_mach_min <= 1.0
                or boundary.get("causal_outflow_evidence")
                != "all_frames_extrapolated_from_interior_stencil"
            ):
                raise ValueError(
                    f"mesh audit trajectory {key!r} causal outflow is not "
                    "verified outward-supersonic"
                )
            if not _is_sha256(boundary.get("boundary_stencil_sha256")):
                raise ValueError(
                    f"mesh audit trajectory {key!r} lacks a stencil digest"
                )
        records[key] = record
    missing = [key for key in selected_keys if key not in records]
    if missing:
        raise ValueError(f"mesh audit does not cover selected trajectories {missing}")
    normalized_payload = dict(payload)
    normalized_payload["contract_completeness"] = (
        "legacy_endpoint_temporal_alignment"
        if is_legacy
        else "hardened_sampled_alignment_and_all_frame_static_graph"
    )
    return normalized_payload, records


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def validate_trained_mesh_contract(
    checkpoint_training: Mapping[str, Any] | None,
    *,
    mesh_audit_sha256: str,
    max_boundary_source_hops: int,
) -> None:
    """Require evaluation to use the exact policy audit bound during training."""

    if checkpoint_training is None:
        return
    policy = checkpoint_training["graph_policy_validation"]
    if policy["sha256"] != mesh_audit_sha256:
        raise ValueError(
            "checkpoint training and evaluation mesh-audit SHA256 values differ"
        )
    if policy["max_boundary_source_hops"] not in (
        None,
        max_boundary_source_hops,
    ):
        raise ValueError(
            "checkpoint training and evaluation boundary source-hop contracts differ"
        )


def _training_reference_provenance(payload: Mapping[str, Any]) -> dict[str, Any]:
    reference = payload.get("reference")
    if not isinstance(reference, Mapping):
        raise ValueError("checkpoint training manifest lacks reference provenance")
    if reference.get("expected_commit") != CPG_REFERENCE_COMMIT:
        raise ValueError("checkpoint training reference commit is not the pinned commit")
    if reference.get("runtime_files_verified") is not True:
        raise ValueError("checkpoint training runtime files were not verified")
    raw_runtime_hashes = reference.get("runtime_file_sha256")
    if not isinstance(raw_runtime_hashes, Mapping):
        raise ValueError("checkpoint training manifest lacks runtime file hashes")
    runtime_hashes = {
        str(path): digest for path, digest in raw_runtime_hashes.items()
    }
    if len(runtime_hashes) != len(raw_runtime_hashes):
        raise ValueError("checkpoint training runtime hash keys are ambiguous")

    expected_paths = set(CPG_REFERENCE_RUNTIME_SHA256)
    actual_paths = {str(path) for path in runtime_hashes}
    unexpected = sorted(actual_paths - expected_paths)
    if unexpected:
        raise ValueError(
            f"checkpoint training runtime hash set has unexpected files {unexpected}"
        )
    mismatched = sorted(
        path
        for path in actual_paths
        if runtime_hashes[path] != CPG_REFERENCE_RUNTIME_SHA256[path]
    )
    if mismatched:
        raise ValueError(
            f"checkpoint training runtime hashes differ for {mismatched}"
        )
    missing = sorted(expected_paths - actual_paths)

    git_complete = reference.get("git_commit_verified") is True
    pin_schema = reference.get("runtime_pin_schema")
    if pin_schema not in {None, CPG_REFERENCE_RUNTIME_PIN_SCHEMA}:
        raise ValueError("checkpoint training runtime pin schema is unsupported")
    if git_complete:
        if reference.get("git_commit") != CPG_REFERENCE_COMMIT:
            raise ValueError("checkpoint training Git commit differs from the pin")
        if reference.get("tracked_clean") is not True:
            raise ValueError("checkpoint training reference checkout was not clean")
    if missing:
        if set(missing) == set(CPG_LEGACY_TRAINING_PIN_OMISSIONS):
            completeness = (
                "git_commit_closes_legacy_runtime_pin_gap"
                if git_complete
                else "legacy_missing_training_runtime_pins"
            )
        else:
            raise ValueError(
                f"checkpoint training runtime hash set is incomplete: {missing}"
            )
    else:
        if pin_schema is None and not git_complete:
            raise ValueError("complete runtime hashes lack a pin schema")
        completeness = "complete"
    return {
        "completeness": completeness,
        "missing_runtime_files": missing,
        "git_commit_verified": git_complete,
    }


def _training_local_source_provenance(payload: Mapping[str, Any]) -> dict[str, Any]:
    source_sha256 = payload.get("source_sha256")
    if not isinstance(source_sha256, Mapping):
        raise ValueError("checkpoint training manifest lacks local source hashes")
    source_sha256 = dict(source_sha256)
    source_provenance = payload.get("source_provenance")
    if source_provenance is None:
        if source_sha256 != CPG_LEGACY_LOCAL_TRAINING_SOURCE_SHA256:
            raise ValueError(
                "legacy checkpoint training source hashes do not match the "
                "archived run"
            )
        validate_cpg_local_training_source_manifest(
            {
                "schema": CPG_LOCAL_TRAINING_SOURCE_SCHEMA,
                "git_commit": CPG_LEGACY_LOCAL_TRAINING_SOURCE_COMMIT,
                "tracked_files_clean": True,
                "file_hash_semantics": "git_blob_sha256",
                "file_sha256": dict(CPG_ARCHIVAL_LOCAL_TRAINING_SOURCE_SHA256),
                "relative_paths": dict(CPG_LOCAL_TRAINING_SOURCE_FILES),
            },
            repo_root=REPO_ROOT,
        )
        return {
            "schema": "cpg_legacy_local_training_source_v1",
            "completeness": "legacy_two_recorded_hashes_match_archival_commit",
            "archival_git_commit": CPG_LEGACY_LOCAL_TRAINING_SOURCE_COMMIT,
            "recorded_file_sha256": source_sha256,
            "missing_manifest_files": sorted(
                set(CPG_LOCAL_TRAINING_SOURCE_FILES) - set(source_sha256)
            ),
            "training_worktree_clean_verified": False,
        }
    if not isinstance(source_provenance, Mapping):
        raise ValueError("checkpoint local source provenance must be an object")
    if (
        set(source_sha256) != set(CPG_LOCAL_TRAINING_SOURCE_FILES)
        or any(not _is_sha256(value) for value in source_sha256.values())
    ):
        raise ValueError("checkpoint training source hashes are incomplete")
    normalized = validate_cpg_local_training_source_manifest(
        source_provenance,
        repo_root=REPO_ROOT,
    )
    if source_sha256 != normalized["file_sha256"]:
        raise ValueError(
            "checkpoint training source hashes differ from their Git provenance"
        )
    return normalized


def load_checkpoint_training_manifest(
    path: Path,
    *,
    checkpoint_sha256: str,
    expected_evaluation_dataset_sha256: str | None = None,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint training manifest root must be an object")
    if payload.get("schema") != LEGAL_TRAINING_SCHEMA:
        raise ValueError("checkpoint training manifest has an unsupported schema")
    if payload.get("status") != "complete":
        raise ValueError("checkpoint training manifest is not complete")
    if payload.get("boundary_mode") != LEGAL_BOUNDARY_MODE:
        raise ValueError("checkpoint was not trained under the legal boundary mode")
    if payload.get("uses_future_reference_boundary") is not False:
        raise ValueError(
            "checkpoint training used or did not exclude future boundaries"
        )

    checkpoint = payload.get("checkpoint")
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint training manifest lacks checkpoint provenance")
    if checkpoint.get("sha256") != checkpoint_sha256:
        raise ValueError("checkpoint SHA256 differs from its training manifest")
    dataset = payload.get("dataset")
    if not isinstance(dataset, dict) or dataset.get("split") != "train":
        raise ValueError("checkpoint training manifest lacks the train split contract")
    if not _is_sha256(dataset.get("sha256")):
        raise ValueError("checkpoint training manifest lacks the dataset SHA256")
    selected_keys = dataset.get("selected_keys")
    if (
        not isinstance(selected_keys, list)
        or not selected_keys
        or any(not isinstance(key, str) or not key for key in selected_keys)
        or len(set(selected_keys)) != len(selected_keys)
    ):
        raise ValueError(
            "checkpoint training selected keys must be nonempty and unique"
        )
    model = payload.get("model_configuration")
    expected_model = {
        "message_passing_num": 12,
        "node_input_size": 6,
        "edge_input_size": 5,
    }
    if not isinstance(model, dict) or any(
        model.get(name) != value for name, value in expected_model.items()
    ):
        raise ValueError("checkpoint training architecture differs from the evaluator")
    model_dt = validate_cpg_model_dt(model.get("dt"))
    training = payload.get("training_configuration")
    required_training = (
        "seed",
        "microbatch_size",
        "gradient_accumulation_steps",
        "effective_batch_size",
        "num_steps",
        "stage1_epochs",
        "stage2_epochs",
        "max_trajectories",
        "max_batches_per_epoch",
        "checkpoint_selection",
    )
    if not isinstance(training, dict) or any(
        name not in training for name in required_training
    ):
        raise ValueError("checkpoint training configuration is incomplete")
    positive_integer_fields = (
        "microbatch_size",
        "gradient_accumulation_steps",
        "effective_batch_size",
        "num_steps",
    )
    nonnegative_integer_fields = ("stage1_epochs", "stage2_epochs")
    if (
        not isinstance(training["seed"], int)
        or isinstance(training["seed"], bool)
        or any(
            not isinstance(training[name], int)
            or isinstance(training[name], bool)
            or training[name] <= 0
            for name in positive_integer_fields
        )
        or any(
            not isinstance(training[name], int)
            or isinstance(training[name], bool)
            or training[name] < 0
            for name in nonnegative_integer_fields
        )
        or not isinstance(training["checkpoint_selection"], str)
        or not training["checkpoint_selection"].strip()
    ):
        raise ValueError("checkpoint training configuration has invalid values")
    if training["effective_batch_size"] != (
        training["microbatch_size"] * training["gradient_accumulation_steps"]
    ):
        raise ValueError("checkpoint training effective batch is inconsistent")
    reference_provenance = _training_reference_provenance(payload)
    local_source_provenance = _training_local_source_provenance(payload)

    policy = payload.get("graph_policy_validation")
    if (
        not isinstance(policy, Mapping)
        or policy.get("case_count") != 20
        or not _is_sha256(policy.get("sha256"))
        or not _is_sha256(policy.get("dataset_sha256"))
    ):
        raise ValueError("checkpoint training graph-policy provenance is incomplete")
    if local_source_provenance["completeness"] == (
        "git_commit_and_full_local_source_closure"
    ) and (
        payload.get("promotion_eligible") is not True
        or training["max_trajectories"] is not None
        or training["max_batches_per_epoch"] is not None
        or policy.get("schema") != CPG_MESH_AUDIT_SCHEMA
        or policy.get("status") != "valid"
        or policy.get("graph_mesh_identity") != "verified_all_selected_cases"
        or policy.get("boundary_geometry") != "verified_all_selected_cases"
        or policy.get("contract_completeness")
        != "hardened_sampled_alignment_and_all_frame_static_graph"
        or policy.get("max_boundary_source_hops")
        != CPG_LEGAL_BOUNDARY_MAX_SOURCE_HOPS
        or payload.get(
            "uses_future_reference_boundary_in_rollout_inputs_or_recurrent_state"
        )
        is not False
        or payload.get("future_reference_boundary_training_use")
        != {
            "supervised_target": True,
            "output_normalizer_statistics": True,
            "model_input": False,
            "recurrent_state": False,
        }
    ):
        raise ValueError(
            "checkpoint training graph-policy identity contract is incomplete"
        )
    if (
        expected_evaluation_dataset_sha256 is not None
        and policy["dataset_sha256"] != expected_evaluation_dataset_sha256
    ):
        raise ValueError(
            "checkpoint training graph-policy validation and evaluation dataset differ"
        )

    return {
        "file_name": path.name,
        "sha256": sha256_file(path),
        "schema": payload["schema"],
        "status": payload["status"],
        "claim_scope": payload.get("claim_scope"),
        "boundary_mode": payload["boundary_mode"],
        "uses_future_reference_boundary": False,
        "uses_future_reference_boundary_in_rollout_inputs_or_recurrent_state": (
            payload.get(
                "uses_future_reference_boundary_in_rollout_inputs_or_recurrent_state",
                False,
            )
        ),
        "future_reference_boundary_training_use": payload.get(
            "future_reference_boundary_training_use"
        ),
        "checkpoint_sha256": checkpoint_sha256,
        "training_dataset": {
            "file_name": dataset.get("file_name"),
            "split": dataset["split"],
            "sha256": dataset["sha256"],
            "selected_key_count": len(selected_keys),
        },
        "model_configuration": {
            **expected_model,
            "dt": model_dt,
        },
        "training_configuration": {name: training[name] for name in required_training},
        "source_sha256": dict(payload["source_sha256"]),
        "local_source_provenance": local_source_provenance,
        "reference_provenance": reference_provenance,
        "graph_policy_validation": {
            "sha256": policy["sha256"],
            "dataset_sha256": policy["dataset_sha256"],
            "case_count": policy["case_count"],
            "max_boundary_source_hops": policy.get(
                "max_boundary_source_hops"
            ),
        },
    }


def _prepare_legal_boundary_policy(
    *,
    torch: Any,
    device: Any,
    raw_case_root: Path,
    audit_record: Mapping[str, Any],
    raw_pos: np.ndarray,
    raw_edges: np.ndarray,
    node_type: np.ndarray,
    mach_field: np.ndarray,
    max_source_hops: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    case_id = int(audit_record["raw_case_id"])
    case_dir = raw_case_root / str(case_id)
    mesh_path = case_dir / "Bump.inp"
    julia_path = case_dir / "Bump.jl"
    mach_path = case_dir / "Mach.txt"
    for path in (mesh_path, julia_path, mach_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    expected_hashes = audit_record.get("source_sha256", {})
    for name, path in (
        ("Bump.inp", mesh_path),
        ("Bump.jl", julia_path),
        ("Mach.txt", mach_path),
    ):
        if sha256_file(path) != expected_hashes.get(name):
            raise ValueError(f"raw case {case_id} {name} differs from the mesh audit")

    mesh = parse_abaqus_mesh(mesh_path)
    mesh_summary, geometry = validate_hdf_mesh_identity(
        mesh,
        hdf_pos=raw_pos,
        hdf_edges=raw_edges,
        hdf_node_type=node_type,
    )
    config = audit_bump_julia_config(julia_path)
    if config != audit_record.get("config"):
        raise ValueError(f"raw case {case_id} configuration differs from the audit")
    mach = float(mach_path.read_text(encoding="utf-8").strip())
    if not np.isfinite(mach) or not np.all(np.isfinite(mach_field)):
        raise ValueError(f"raw case {case_id} Mach contains nonfinite values")
    if float(np.max(np.abs(np.asarray(mach_field) - mach))) > 2.0e-10:
        raise ValueError(f"raw case {case_id} Mach differs from the HDF field")
    stencil = build_boundary_stencil(
        pos=raw_pos,
        edges=raw_edges,
        node_type=node_type,
        node_normal=geometry["node_normal"],
        max_source_hops=max_source_hops,
    )
    expected_fallbacks = int(
        audit_record["boundary"]["boundary_stencil_fallback_target_count"]
    )
    if stencil.fallback_target_count != expected_fallbacks:
        raise ValueError(f"raw case {case_id} boundary stencil differs from the audit")
    expected_stencil_sha256 = audit_record["boundary"].get(
        "boundary_stencil_sha256"
    )
    if (
        expected_stencil_sha256 is not None
        and boundary_stencil_sha256(stencil) != expected_stencil_sha256
    ):
        raise ValueError(
            f"raw case {case_id} boundary stencil digest differs from the audit"
        )
    policy = build_torch_boundary_policy(
        torch=torch,
        device=device,
        node_type=node_type,
        node_normal=geometry["node_normal"],
        wall_normal_coherence=geometry["node_boundary_normal_coherence"],
        stencil=stencil,
        mach=mach,
        config=config,
    )
    metadata = {
        "raw_case_id": case_id,
        "raw_source_sha256": {
            name: expected_hashes[name] for name in ("Bump.inp", "Bump.jl", "Mach.txt")
        },
        "mesh": mesh_summary,
        "configuration": config,
        "mach": mach,
        "boundary_stencil": {
            "target_count": int(stencil.target_nodes.size),
            "entry_count": int(stencil.source_nodes.size),
            "fallback_target_count": stencil.fallback_target_count,
            "max_source_hops": max_source_hops,
            "sharp_wall_corner_count": int(
                np.count_nonzero(
                    (node_type == WALL_NODE)
                    & (geometry["node_boundary_normal_coherence"] < 0.95)
                )
            ),
        },
    }
    return policy, metadata


def primitive_termination_reason(torch: Any, state: Any) -> str | None:
    if not bool(torch.all(torch.isfinite(state))):
        return "nonfinite_recurrent_state"
    if bool(torch.any(state[:, 0] <= 0.0)):
        return "nonpositive_recurrent_density"
    if bool(torch.any(state[:, 3] <= 0.0)):
        return "nonpositive_recurrent_pressure"
    return None


def rollout_trajectory(
    *,
    api: dict[str, Any],
    model: Any,
    dataset: Any,
    dataloader: Any,
    transformer: Any,
    trajectory_index: int,
    device: Any,
    boundary_mode: str = ORACLE_BOUNDARY_MODE,
    boundary_policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    torch = api["torch"]
    make_edges_undirected = api["make_edges_undirected"]
    if boundary_mode == LEGAL_BOUNDARY_MODE and boundary_policy is None:
        raise ValueError("causal nodal rollout requires a boundary policy")
    if boundary_mode not in {ORACLE_BOUNDARY_MODE, LEGAL_BOUNDARY_MODE}:
        raise ValueError(f"unsupported boundary mode {boundary_mode}")
    dataset.change_file(trajectory_index)
    expected_steps = int(dataset.cur_targecity_length) - 1

    recurrent_prediction = None
    boundary_mask = None
    reference_current: list[np.ndarray] = []
    inputs_before_boundary: list[np.ndarray] = []
    model_inputs: list[np.ndarray] = []
    model_mach: list[np.ndarray] = []
    raw_predictions: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    reference_node_type = None
    model_geometry = None
    failure_reason = None
    failure_step = None
    minimum_attempted_legal_outflow_normal_mach = None

    with torch.no_grad():
        for graph in dataloader:
            graph = make_edges_undirected(graph)
            graph = transformer(graph)
            graph = graph.to(device)

            node_type = graph.x[:, 0]
            step_boundary_mask = node_type != int(EulerNodeType.NORMAL)
            if boundary_mask is None:
                boundary_mask = step_boundary_mask
                reference_node_type = node_type.clone()
            elif not torch.equal(reference_node_type, node_type):
                raise RuntimeError("node types changed during one trajectory")

            next_reference = graph.y
            current_reference = graph.x[:, 1:5].clone()
            reference_current.append(current_reference.detach().cpu().numpy())
            if recurrent_prediction is not None:
                graph.x[:, 1:5] = recurrent_prediction.detach()
            inputs_before_boundary.append(graph.x[:, 1:5].detach().cpu().numpy().copy())

            if boundary_mode == ORACLE_BOUNDARY_MODE:
                graph.x[boundary_mask, 1:5] = next_reference[boundary_mask]
            else:
                graph.x[:, 1:5] = apply_torch_boundary_policy(
                    torch, graph.x[:, 1:5], boundary_policy
                )
            model_inputs.append(graph.x[:, 1:5].detach().cpu().numpy().copy())
            model_mach.append(graph.x[:, 5].detach().cpu().numpy().copy())

            if model_geometry is None:
                model_geometry = {
                    "model_pos": graph.pos.detach().clone(),
                    "directed_edges": graph.edge_index.detach().clone(),
                    "edge_attr_before_model": graph.edge_attr.detach().clone(),
                }
            elif (
                not torch.equal(model_geometry["model_pos"], graph.pos)
                or not torch.equal(model_geometry["directed_edges"], graph.edge_index)
                or not torch.equal(
                    model_geometry["edge_attr_before_model"], graph.edge_attr
                )
            ):
                raise RuntimeError(
                    "model graph geometry changed; the artifact schema requires "
                    "a static mesh and stencil"
                )

            raw_prediction = model(graph, sequence_noise=None)
            if boundary_mode == ORACLE_BOUNDARY_MODE:
                recurrent_prediction = raw_prediction.clone()
                recurrent_prediction[boundary_mask] = next_reference[boundary_mask]
            else:
                recurrent_prediction = apply_torch_boundary_policy(
                    torch, raw_prediction, boundary_policy
                )

            raw_predictions.append(raw_prediction.detach().cpu().numpy())
            predictions.append(recurrent_prediction.detach().cpu().numpy())
            targets.append(next_reference.detach().cpu().numpy())
            reason = None
            if not bool(torch.all(torch.isfinite(raw_prediction))):
                reason = "nonfinite_raw_prediction"
            elif boundary_mode == LEGAL_BOUNDARY_MODE:
                reason = primitive_termination_reason(torch, recurrent_prediction)
                if reason is None:
                    outflow_mach_min = legal_outflow_normal_mach_min(
                        torch, recurrent_prediction, boundary_policy
                    )
                    if outflow_mach_min is not None:
                        minimum_attempted_legal_outflow_normal_mach = (
                            outflow_mach_min
                            if minimum_attempted_legal_outflow_normal_mach is None
                            else min(
                                minimum_attempted_legal_outflow_normal_mach,
                                outflow_mach_min,
                            )
                        )
                        if outflow_mach_min <= 1.0:
                            reason = "nonsupersonic_recurrent_outflow"
            else:
                reason = primitive_termination_reason(torch, recurrent_prediction)
            if reason is not None:
                failure_reason = reason
                failure_step = len(predictions) - 1
                break

    if boundary_mask is None or model_geometry is None:
        raise RuntimeError("rollout produced no graph frames")
    attempted_steps = len(predictions)
    terminal_failure_included = failure_reason is not None
    admissible_steps = attempted_steps - int(terminal_failure_included)
    if failure_reason is None and attempted_steps != expected_steps:
        failure_reason = "unexpected_rollout_length"
    completed = failure_reason is None and attempted_steps == expected_steps
    injection_mask = (
        boundary_mask
        if boundary_mode == ORACLE_BOUNDARY_MODE
        else torch.zeros_like(boundary_mask)
    )
    arrays = {
        "reference_current": _stack(reference_current, "reference states"),
        "inputs_before_boundary": _stack(inputs_before_boundary, "pre-boundary inputs"),
        "model_inputs": _stack(model_inputs, "model inputs"),
        "model_mach": _stack(model_mach, "Mach inputs"),
        "raw_predicteds": _stack(raw_predictions, "raw predictions"),
        "predicteds": _stack(predictions, "predictions"),
        "targets": _stack(targets, "targets"),
        "injection_mask": injection_mask.detach().cpu().numpy(),
        "boundary_node_mask": boundary_mask.detach().cpu().numpy(),
        "model_pos": model_geometry["model_pos"].cpu().numpy().copy(),
        "directed_edges": model_geometry["directed_edges"].cpu().numpy().T.copy(),
        "edge_attr_before_model": model_geometry["edge_attr_before_model"]
        .cpu()
        .numpy()
        .copy(),
    }
    return {
        "arrays": arrays,
        "minimum_attempted_legal_outflow_normal_mach": (
            minimum_attempted_legal_outflow_normal_mach
        ),
        "termination": {
            "completed": completed,
            "attempted_steps": attempted_steps,
            "admissible_steps": admissible_steps,
            "valid_steps": admissible_steps,
            "expected_steps": expected_steps,
            "failure_step": failure_step,
            "failure_step_one_based": (
                None if failure_step is None else failure_step + 1
            ),
            "failure_reason": failure_reason,
            "terminal_failure_included_in_arrays": terminal_failure_included,
        },
    }


def _create_dataset(handle: Any, name: str, value: np.ndarray) -> None:
    handle.create_dataset(
        name,
        data=value,
        compression="gzip",
        compression_opts=4,
        shuffle=True,
    )


def _empty_rollout_metrics(node_type: np.ndarray) -> dict[str, Any]:
    types = np.asarray(node_type, dtype=np.int64).reshape(-1)
    masks = {
        "all": np.ones_like(types, dtype=bool),
        "normal": types == int(EulerNodeType.NORMAL),
        "wall": types == int(EulerNodeType.WALL),
        "outflow": types == int(EulerNodeType.OUTFLOW),
        "inflow": types == int(EulerNodeType.INFLOW),
    }
    masks["boundary"] = ~masks["normal"]
    result: dict[str, Any] = {"num_steps": 0, "num_nodes": int(types.size)}
    for name, mask in masks.items():
        result[name] = {
            "num_nodes": int(np.count_nonzero(mask)),
            "rollout_rmse": None,
            "final_step_rmse": None,
            "cumulative_rmse": None,
            "max_abs_error": None,
        }
    return result


def save_rollout_artifact(
    *,
    h5py: Any,
    path: Path,
    arrays: dict[str, np.ndarray],
    raw_pos: np.ndarray,
    raw_edges: np.ndarray,
    node_type: np.ndarray,
    mach: np.ndarray,
    attributes: dict[str, Any],
) -> None:
    with h5py.File(path, "w") as handle:
        for name, value in arrays.items():
            _create_dataset(handle, name, np.asarray(value))
        _create_dataset(handle, "pos", raw_pos)
        _create_dataset(handle, "edges", raw_edges)
        _create_dataset(handle, "node_type", node_type)
        _create_dataset(handle, "Mach", mach)
        for name, value in attributes.items():
            handle.attrs[name] = value


def _metric_row(
    *,
    result_file: str,
    trajectory_key: str,
    post_metrics: dict[str, Any],
    raw_metrics: dict[str, Any],
    max_distance: int,
    boundary_mode: str,
    termination: Mapping[str, Any],
    minimum_attempted_legal_outflow_normal_mach: float | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "result_file": result_file,
        "trajectory_key": trajectory_key,
        "num_steps": post_metrics["num_steps"],
        "num_nodes": post_metrics["num_nodes"],
        "boundary_mode": boundary_mode,
        "completed": termination["completed"],
        "attempted_steps": termination["attempted_steps"],
        "admissible_steps": termination["admissible_steps"],
        "valid_steps": termination["valid_steps"],
        "expected_steps": termination["expected_steps"],
        "failure_step": termination["failure_step"],
        "failure_step_one_based": termination["failure_step_one_based"],
        "failure_reason": termination["failure_reason"],
        "metrics_include_terminal_failure": False,
        "minimum_attempted_legal_outflow_normal_mach": (
            minimum_attempted_legal_outflow_normal_mach
        ),
        "max_boundary_graph_distance": max_distance,
        "post_boundary_max_abs_error": post_metrics["boundary"]["max_abs_error"],
        "raw_boundary_max_abs_error": raw_metrics["boundary"]["max_abs_error"],
    }
    for prefix, metrics in (("post", post_metrics), ("raw", raw_metrics)):
        for mask in ("all", "normal", "boundary"):
            values = metrics[mask]["rollout_rmse"]
            for index, name in enumerate(PRIMITIVE_NAMES):
                row[f"{prefix}_{mask}_{name}_rollout_rmse"] = (
                    None if values is None else values[index]
                )
    return row


def write_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty metric table")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _aggregate(rows: list[dict[str, Any]], key_prefix: str) -> list[float]:
    return [
        float(np.mean([row[f"{key_prefix}_{name}_rollout_rmse"] for row in rows]))
        for name in PRIMITIVE_NAMES
    ]


def _primary_aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("primary aggregation requires at least one trajectory")
    completed_horizons = {int(row["admissible_steps"]) for row in rows}
    expected_horizons = {int(row["expected_steps"]) for row in rows}
    available = (
        all(bool(row["completed"]) for row in rows)
        and len(completed_horizons) == 1
        and len(expected_horizons) == 1
        and completed_horizons == expected_horizons
    )
    if not available:
        return {
            "post_all_rollout_rmse": None,
            "post_normal_rollout_rmse": None,
            "raw_boundary_rollout_rmse": None,
            "primary_metric_status": "unavailable_incomplete_or_unequal_horizon",
        }
    return {
        "post_all_rollout_rmse": _aggregate(rows, "post_all"),
        "post_normal_rollout_rmse": _aggregate(rows, "post_normal"),
        "raw_boundary_rollout_rmse": _aggregate(rows, "raw_boundary"),
        "primary_metric_status": "complete_equal_horizon",
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    reference = validate_cpg_reference_source(args.reference_repo)
    evaluator_sources = cpg_evaluator_source_manifest(REPO_ROOT)
    if args.boundary_mode == LEGAL_BOUNDARY_MODE:
        mesh_source = "utility/time_dependent_no/cpg_mesh_contract.py"
        evaluator_sources[mesh_source] = sha256_file(REPO_ROOT / mesh_source)
    evaluator_sha256 = evaluator_sources[CPG_EVALUATOR_SOURCE_FILES[0]]
    dataset_file = args.dataset_root / f"{args.split}.h5"
    required_files = [dataset_file, args.checkpoint]
    if args.checkpoint_training_manifest is not None:
        required_files.append(args.checkpoint_training_manifest)
    for path in required_files:
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.raw_case_root is not None and not args.raw_case_root.is_dir():
        raise FileNotFoundError(args.raw_case_root)
    if args.output_dir.exists():
        raise FileExistsError(
            f"output directory already exists; choose a new run directory: {args.output_dir}"
        )

    api = _import_reference_api(args.reference_repo)
    torch = api["torch"]
    _set_seed(torch, args.seed)
    device = _select_device(torch, args.device, args.gpu)
    checkpoint_sha256 = sha256_file(args.checkpoint)
    dataset_sha256 = sha256_file(dataset_file)
    checkpoint_training = None
    if args.checkpoint_training_manifest is not None:
        checkpoint_training = load_checkpoint_training_manifest(
            args.checkpoint_training_manifest,
            checkpoint_sha256=checkpoint_sha256,
            expected_evaluation_dataset_sha256=dataset_sha256,
        )

    model = api["Simulator"](
        message_passing_num=12,
        node_input_size=6,
        edge_input_size=5,
        device=device,
    )
    model.load_checkpoint(str(args.checkpoint))
    model.eval()

    dataset = api["FPC_ROLLOUT"](
        str(args.dataset_root),
        split=args.split,
        time_iterval=args.dt,
    )
    dataloader = api["DataLoader"](dataset=dataset, batch_size=1)
    transformer = api["T"].Compose(
        [
            api["T"].NormalizeScale(),
            api["T"].Cartesian(norm=False),
            api["T"].Distance(norm=False),
        ]
    )
    available_keys = [str(key) for key in dataset.datasets]
    selected_keys = select_trajectory_keys(
        available_keys,
        args.trajectory_key,
        args.max_trajectories,
    )
    key_to_index = {key: index for index, key in enumerate(available_keys)}
    mesh_audit = None
    mesh_records: dict[str, dict[str, Any]] = {}
    mesh_audit_sha256 = None
    max_boundary_source_hops = None
    if args.boundary_mode == LEGAL_BOUNDARY_MODE:
        mesh_audit, mesh_records = load_mesh_audit(
            args.mesh_audit_summary,
            dataset_sha256=dataset_sha256,
            selected_keys=available_keys,
        )
        mesh_audit_sha256 = sha256_file(args.mesh_audit_summary)
        max_boundary_source_hops = int(mesh_audit["max_boundary_source_hops"])
        validate_trained_mesh_contract(
            checkpoint_training,
            mesh_audit_sha256=mesh_audit_sha256,
            max_boundary_source_hops=max_boundary_source_hops,
        )

    args.output_dir.mkdir(parents=True)
    result_dir = args.output_dir / "result"
    result_dir.mkdir()
    if args.boundary_mode == ORACLE_BOUNDARY_MODE:
        run_schema = RUN_SCHEMA
    elif checkpoint_training is None:
        run_schema = LEGAL_RUN_SCHEMA
    else:
        run_schema = LEGAL_TRAINED_RUN_SCHEMA

    rows: list[dict[str, Any]] = []
    trajectory_records: list[dict[str, Any]] = []
    with api["h5py"].File(dataset_file, "r") as dataset_handle:
        for trajectory_key in selected_keys:
            trajectory_index = key_to_index[trajectory_key]
            raw_pos, raw_edges, node_type, mach = validate_cpg_static_graph(
                dataset_handle[trajectory_key]
            )
            boundary_policy = None
            legal_metadata = None
            if args.boundary_mode == LEGAL_BOUNDARY_MODE:
                boundary_policy, legal_metadata = _prepare_legal_boundary_policy(
                    torch=torch,
                    device=device,
                    raw_case_root=args.raw_case_root,
                    audit_record=mesh_records[trajectory_key],
                    raw_pos=raw_pos,
                    raw_edges=raw_edges,
                    node_type=node_type,
                    mach_field=mach,
                    max_source_hops=max_boundary_source_hops,
                )
            rollout = rollout_trajectory(
                api=api,
                model=model,
                dataset=dataset,
                dataloader=dataloader,
                transformer=transformer,
                trajectory_index=trajectory_index,
                device=device,
                boundary_mode=args.boundary_mode,
                boundary_policy=boundary_policy,
            )
            arrays = rollout["arrays"]
            termination = rollout["termination"]
            minimum_attempted_legal_outflow_normal_mach = rollout[
                "minimum_attempted_legal_outflow_normal_mach"
            ]
            metric_steps = int(termination["admissible_steps"])
            if metric_steps:
                post_metrics = release_rollout_metrics(
                    arrays["predicteds"][:metric_steps],
                    arrays["targets"][:metric_steps],
                    node_type,
                )
                raw_metrics = release_rollout_metrics(
                    arrays["raw_predicteds"][:metric_steps],
                    arrays["targets"][:metric_steps],
                    node_type,
                )
            else:
                post_metrics = _empty_rollout_metrics(node_type)
                raw_metrics = _empty_rollout_metrics(node_type)
            boundary_distance = graph_distance_from_sources(
                raw_edges, node_type != int(EulerNodeType.NORMAL)
            )
            distance_metrics = (
                rollout_rmse_by_graph_distance(
                    arrays["predicteds"][:metric_steps],
                    arrays["targets"][:metric_steps],
                    boundary_distance,
                )
                if metric_steps
                else []
            )
            arrays["boundary_graph_distance"] = boundary_distance

            result_name = f"{trajectory_index + 1}.h5"
            result_path = result_dir / result_name
            attributes = {
                "schema": run_schema,
                "trajectory_key": trajectory_key,
                "trajectory_index": trajectory_index,
                "split": args.split,
                "boundary_mode": args.boundary_mode,
                "checkpoint_sha256": checkpoint_sha256,
                "dataset_sha256": dataset_sha256,
                "evaluator_sha256": evaluator_sha256,
                "reference_commit": (
                    reference["git_commit"] or reference["expected_commit"]
                ),
                "reference_verification": reference["verification"],
                "dt": args.dt,
                "seed": args.seed,
                "primitive_order": json.dumps(list(PRIMITIVE_NAMES)),
                "checkpoint_binding": "explicit Simulator.load_checkpoint argument",
                "completed": termination["completed"],
                "attempted_steps": termination["attempted_steps"],
                "admissible_steps": termination["admissible_steps"],
                "valid_steps": termination["valid_steps"],
                "expected_steps": termination["expected_steps"],
                "failure_step": (
                    -1
                    if termination["failure_step"] is None
                    else termination["failure_step"]
                ),
                "failure_step_one_based": (
                    -1
                    if termination["failure_step_one_based"] is None
                    else termination["failure_step_one_based"]
                ),
                "failure_reason": termination["failure_reason"] or "",
                "terminal_failure_included_in_arrays": termination[
                    "terminal_failure_included_in_arrays"
                ],
                "metric_scope": "admissible_prefix_excluding_terminal_failure",
                "termination_accounting_schema": TERMINATION_ACCOUNTING_SCHEMA,
            }
            if args.boundary_mode == LEGAL_BOUNDARY_MODE:
                attributes.update(
                    {
                        "mesh_audit_sha256": mesh_audit_sha256,
                        "uses_future_reference_boundary": False,
                        "legal_boundary_metadata": json.dumps(
                            legal_metadata, sort_keys=True
                        ),
                    }
                )
                if minimum_attempted_legal_outflow_normal_mach is not None:
                    attributes["minimum_attempted_legal_outflow_normal_mach"] = (
                        minimum_attempted_legal_outflow_normal_mach
                    )
                if checkpoint_training is not None:
                    attributes["checkpoint_training_manifest_sha256"] = (
                        checkpoint_training["sha256"]
                    )
                    attributes["checkpoint_training_provenance_completeness"] = (
                        checkpoint_training["reference_provenance"]["completeness"]
                    )
                    attributes[
                        "checkpoint_training_local_source_completeness"
                    ] = checkpoint_training["local_source_provenance"][
                        "completeness"
                    ]
            save_rollout_artifact(
                h5py=api["h5py"],
                path=result_path,
                arrays=arrays,
                raw_pos=raw_pos,
                raw_edges=raw_edges,
                node_type=node_type,
                mach=mach,
                attributes=attributes,
            )
            row = _metric_row(
                result_file=result_name,
                trajectory_key=trajectory_key,
                post_metrics=post_metrics,
                raw_metrics=raw_metrics,
                max_distance=int(np.max(boundary_distance)),
                boundary_mode=args.boundary_mode,
                termination=termination,
                minimum_attempted_legal_outflow_normal_mach=(
                    minimum_attempted_legal_outflow_normal_mach
                ),
            )
            rows.append(row)
            trajectory_records.append(
                {
                    "result_file": result_name,
                    "result_sha256": sha256_file(result_path),
                    "trajectory_key": trajectory_key,
                    "trajectory_index": trajectory_index,
                    "termination": termination,
                    "minimum_attempted_legal_outflow_normal_mach": (
                        minimum_attempted_legal_outflow_normal_mach
                    ),
                    "post_policy_metrics": post_metrics,
                    "raw_prediction_metrics": raw_metrics,
                    "boundary_distance_metrics": distance_metrics,
                    "legal_boundary": legal_metadata,
                }
            )

    write_metrics_csv(args.output_dir / "trajectory_metrics.csv", rows)
    is_oracle = args.boundary_mode == ORACLE_BOUNDARY_MODE
    is_legally_trained = checkpoint_training is not None
    aggregate_metrics = _primary_aggregate(rows)
    training_reference_provenance = (
        None
        if checkpoint_training is None
        else checkpoint_training["reference_provenance"]["completeness"]
    )
    training_local_provenance = (
        None
        if checkpoint_training is None
        else checkpoint_training["local_source_provenance"]["completeness"]
    )
    training_provenance_complete = (
        training_reference_provenance
        in {"complete", "git_commit_closes_legacy_runtime_pin_gap"}
        and training_local_provenance
        == "git_commit_and_full_local_source_closure"
    )
    manifest = {
        "schema": run_schema,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "claim_scope": (
            "pinned-release oracle-boundary replay; not paper identity"
            if is_oracle
            else (
                "release-bundle checkpoint trained and evaluated with causal nodal "
                "boundaries; not exact DG replay, paper identity, or evidence of a "
                "general learned solver"
                + (
                    ""
                    if training_provenance_complete
                    else "; historical public-runtime and/or local-source "
                    "provenance is incomplete"
                )
                if is_legally_trained
                else "frozen checkpoint causal nodal boundary sensitivity; not exact "
                "DG replay, fair autonomous training, or paper identity"
            )
        ),
        "reference": reference,
        "runtime": _runtime_manifest(),
        "device": str(device),
        "seed": args.seed,
        "boundary_mode": args.boundary_mode,
        "model_configuration": {
            "message_passing_num": 12,
            "node_input_size": 6,
            "edge_input_size": 5,
            "dt": args.dt,
        },
        "checkpoint": {
            "file_name": args.checkpoint.name,
            "sha256": checkpoint_sha256,
            "binding": "explicit Simulator.load_checkpoint argument",
            "training_manifest": checkpoint_training,
        },
        "dataset": {
            "split": args.split,
            "file_name": dataset_file.name,
            "sha256": dataset_sha256,
            "hdf5_iteration_keys": available_keys,
            "selected_keys": selected_keys,
            "mesh_audit": (
                None
                if mesh_audit is None
                else {
                    "file_name": args.mesh_audit_summary.name,
                    "sha256": mesh_audit_sha256,
                    "schema": mesh_audit["schema"],
                    "status": mesh_audit["status"],
                    "contract_completeness": mesh_audit[
                        "contract_completeness"
                    ],
                    "dataset_sha256": mesh_audit["dataset_sha256"],
                    "temporal_alignment_offsets": mesh_audit[
                        "temporal_alignment_offsets"
                    ],
                    "temporal_alignment_evidence": mesh_audit.get(
                        "temporal_alignment_evidence",
                        "legacy_first_last_only",
                    ),
                    "solver_dof_identity": mesh_audit["solver_dof_identity"],
                    "control_volume_identity": mesh_audit["control_volume_identity"],
                    "physical_face_identity": mesh_audit["physical_face_identity"],
                }
            ),
        },
        "evaluator": {
            "source_sha256": evaluator_sources,
            "target_source": "dataset ground truth",
            "uses_future_reference_boundary": is_oracle,
            "input_boundary_policy": (
                "next-reference state on every non-normal node"
                if is_oracle
                else "current-state freestream inflow, interior-extrapolated "
                "slip wall, and interior-extrapolated supersonic outflow"
            ),
            "output_boundary_policy": (
                "next-reference clamp on every non-normal node"
                if is_oracle
                else "same causal nodal policy applied to the raw model output"
            ),
            "inference_floor_or_limiter": "none",
            "exact_dg_boundary_replay": False if not is_oracle else None,
            "training_contract_caveat": (
                None
                if is_oracle
                else (
                    "checkpoint provenance binds legal-nodal training without "
                    "future-reference boundary injection into model inputs or "
                    "recurrent state; full truth targets still feed supervision and "
                    "output-normalizer statistics; no validation split or exact DG replay"
                    + (
                        ""
                        if training_provenance_complete
                        else "; the historical manifest omitted loss/noise public "
                        "runtime pins and local provenance/euler utility hashes"
                    )
                    if is_legally_trained
                    else "checkpoint was trained with next-reference boundary injection"
                )
            ),
            "release_metric": "all-node autoregressive rollout RMSE",
            "termination_accounting_schema": TERMINATION_ACCOUNTING_SCHEMA,
            "additional_metrics": [
                "normal-only RMSE",
                "raw pre-clamp boundary RMSE",
                "RMSE by graph distance from every non-normal node",
            ],
        },
        "aggregate": {
            **aggregate_metrics,
            "completed_trajectories": int(sum(bool(row["completed"]) for row in rows)),
            "minimum_valid_steps": int(min(row["valid_steps"] for row in rows)),
            "maximum_attempted_steps": int(
                max(row["attempted_steps"] for row in rows)
            ),
            "expected_steps": sorted({int(row["expected_steps"]) for row in rows}),
        },
        "trajectories": trajectory_records,
    }
    manifest_path = args.output_dir / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "trajectories": len(rows),
                "checkpoint_sha256": checkpoint_sha256,
                "dataset_sha256": dataset_sha256,
                "evaluator_sha256": evaluator_sha256,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
