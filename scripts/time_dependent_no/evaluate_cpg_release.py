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
    CPG_EVALUATOR_SOURCE_FILES,
    cpg_evaluator_source_manifest,
    cpg_graph_frame_metadata,
    graph_distance_from_sources,
    release_rollout_metrics,
    rollout_rmse_by_graph_distance,
    sha256_file,
    validate_cpg_reference_source,
)
from utility.time_dependent_no.cpg_mesh_contract import (  # noqa: E402
    WALL_NODE,
    apply_torch_boundary_policy,
    audit_bump_julia_config,
    build_boundary_stencil,
    build_torch_boundary_policy,
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
MESH_AUDIT_SCHEMA = "cpg_bump_mesh_provenance_audit_v1"
LEGAL_TRAINING_SCHEMA = "cpg_legal_boundary_training_v1"


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
    parser.add_argument("--dt", type=float, default=0.025)
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
    if args.dt <= 0.0:
        raise ValueError("--dt must be positive")
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
    reference = str(reference_repo.resolve())
    if reference not in sys.path:
        sys.path.insert(0, reference)

    try:
        import h5py  # type: ignore[import-not-found]
        import torch
        from torch_geometric.loader import DataLoader
        import torch_geometric.transforms as T

        from dataset.fpcMulti import FPC_ROLLOUT
        from modelEdgeUpd.simulator import Simulator
        from utils.to_undirected import make_edges_undirected
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "the reference evaluator requires h5py, torch, torch-geometric, and "
            "the pinned external CPGNet checkout"
        ) from exc

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
    if payload.get("schema") != MESH_AUDIT_SCHEMA:
        raise ValueError("mesh audit has an unsupported schema")
    if payload.get("status") != "valid":
        raise ValueError("mesh audit did not pass its shared-contract gate")
    if payload.get("dataset_sha256") != dataset_sha256:
        raise ValueError("mesh audit and evaluator dataset SHA256 differ")
    if payload.get("graph_mesh_identity") != "verified_all_selected_cases":
        raise ValueError("mesh audit does not verify graph-to-mesh identity")
    if payload.get("boundary_geometry") != "verified_all_selected_cases":
        raise ValueError("mesh audit does not verify boundary geometry")
    records: dict[str, dict[str, Any]] = {}
    for record in payload.get("cases", []):
        key = str(record.get("trajectory_key", ""))
        if not key or key in records:
            raise ValueError("mesh audit trajectory records are missing or duplicated")
        records[key] = record
    missing = [key for key in selected_keys if key not in records]
    if missing:
        raise ValueError(f"mesh audit does not cover selected trajectories {missing}")
    return payload, records


def load_checkpoint_training_manifest(
    path: Path,
    *,
    checkpoint_sha256: str,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
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
    if not isinstance(dataset.get("sha256"), str):
        raise ValueError("checkpoint training manifest lacks the dataset SHA256")
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
    training = payload.get("training_configuration")
    required_training = (
        "seed",
        "microbatch_size",
        "gradient_accumulation_steps",
        "effective_batch_size",
        "num_steps",
        "stage1_epochs",
        "stage2_epochs",
        "checkpoint_selection",
    )
    if not isinstance(training, dict) or any(
        name not in training for name in required_training
    ):
        raise ValueError("checkpoint training configuration is incomplete")
    if training["effective_batch_size"] != (
        training["microbatch_size"] * training["gradient_accumulation_steps"]
    ):
        raise ValueError("checkpoint training effective batch is inconsistent")

    return {
        "file_name": path.name,
        "sha256": sha256_file(path),
        "schema": payload["schema"],
        "status": payload["status"],
        "claim_scope": payload.get("claim_scope"),
        "boundary_mode": payload["boundary_mode"],
        "uses_future_reference_boundary": False,
        "checkpoint_sha256": checkpoint_sha256,
        "training_dataset": {
            "file_name": dataset.get("file_name"),
            "split": dataset["split"],
            "sha256": dataset["sha256"],
            "selected_key_count": len(dataset.get("selected_keys", [])),
        },
        "model_configuration": {
            **expected_model,
            "dt": model.get("dt"),
        },
        "training_configuration": {name: training[name] for name in required_training},
        "source_sha256": payload.get("source_sha256"),
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
            if boundary_mode == LEGAL_BOUNDARY_MODE:
                reason = primitive_termination_reason(torch, recurrent_prediction)
                if reason is not None:
                    failure_reason = reason
                    failure_step = len(predictions) - 1
                    break

    if boundary_mask is None or model_geometry is None:
        raise RuntimeError("rollout produced no graph frames")
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
        "termination": {
            "completed": failure_reason is None and len(predictions) == expected_steps,
            "valid_steps": len(predictions),
            "expected_steps": expected_steps,
            "failure_step": failure_step,
            "failure_reason": failure_reason,
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
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "result_file": result_file,
        "trajectory_key": trajectory_key,
        "num_steps": post_metrics["num_steps"],
        "num_nodes": post_metrics["num_nodes"],
        "boundary_mode": boundary_mode,
        "completed": termination["completed"],
        "valid_steps": termination["valid_steps"],
        "expected_steps": termination["expected_steps"],
        "failure_step": termination["failure_step"],
        "failure_reason": termination["failure_reason"],
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
            raw_pos, raw_edges, node_type, mach = cpg_graph_frame_metadata(
                dataset_handle[trajectory_key], frame=0
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
            post_metrics = release_rollout_metrics(
                arrays["predicteds"], arrays["targets"], node_type
            )
            raw_metrics = release_rollout_metrics(
                arrays["raw_predicteds"], arrays["targets"], node_type
            )
            boundary_distance = graph_distance_from_sources(
                raw_edges, node_type != int(EulerNodeType.NORMAL)
            )
            distance_metrics = rollout_rmse_by_graph_distance(
                arrays["predicteds"], arrays["targets"], boundary_distance
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
                "valid_steps": termination["valid_steps"],
                "expected_steps": termination["expected_steps"],
                "failure_step": (
                    -1
                    if termination["failure_step"] is None
                    else termination["failure_step"]
                ),
                "failure_reason": termination["failure_reason"] or "",
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
                if checkpoint_training is not None:
                    attributes["checkpoint_training_manifest_sha256"] = (
                        checkpoint_training["sha256"]
                    )
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
            )
            rows.append(row)
            trajectory_records.append(
                {
                    "result_file": result_name,
                    "result_sha256": sha256_file(result_path),
                    "trajectory_key": trajectory_key,
                    "trajectory_index": trajectory_index,
                    "termination": termination,
                    "post_policy_metrics": post_metrics,
                    "raw_prediction_metrics": raw_metrics,
                    "boundary_distance_metrics": distance_metrics,
                    "legal_boundary": legal_metadata,
                }
            )

    write_metrics_csv(args.output_dir / "trajectory_metrics.csv", rows)
    is_oracle = args.boundary_mode == ORACLE_BOUNDARY_MODE
    is_legally_trained = checkpoint_training is not None
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
                    "dataset_sha256": mesh_audit["dataset_sha256"],
                    "temporal_alignment_offsets": mesh_audit[
                        "temporal_alignment_offsets"
                    ],
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
                    "checkpoint provenance binds legal-nodal training without future-"
                    "reference boundaries; no validation split or exact DG replay"
                    if is_legally_trained
                    else "checkpoint was trained with next-reference boundary injection"
                )
            ),
            "release_metric": "all-node autoregressive rollout RMSE",
            "additional_metrics": [
                "normal-only RMSE",
                "raw pre-clamp boundary RMSE",
                "RMSE by graph distance from every non-normal node",
            ],
        },
        "aggregate": {
            "post_all_rollout_rmse": _aggregate(rows, "post_all"),
            "post_normal_rollout_rmse": _aggregate(rows, "post_normal"),
            "raw_boundary_rollout_rmse": _aggregate(rows, "raw_boundary"),
            "completed_trajectories": int(sum(bool(row["completed"]) for row in rows)),
            "minimum_valid_steps": int(min(row["valid_steps"] for row in rows)),
            "expected_steps": sorted({int(row["expected_steps"]) for row in rows}),
        },
        "trajectories": trajectory_records,
    }
    manifest_path = args.output_dir / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
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
