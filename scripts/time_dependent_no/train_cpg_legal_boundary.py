"""Train released CPGNet with causal nodal, rather than oracle, boundaries.

This release-bundle counterfactual keeps the public architecture, state loss,
noise scales, and two-stage schedule. It is not a paper reproduction or an
exact replay of the source DG boundary flux.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import importlib.metadata
from itertools import islice
import json
from pathlib import Path
import platform
import random
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utility.time_dependent_no.cpg_mesh_contract import (  # noqa: E402
    GRAPH_BOUNDARY_GEOMETRY_ATOL,
    INFLOW_NODE,
    NORMAL_NODE,
    OUTFLOW_NODE,
    WALL_NODE,
    BoundaryStencil,
    apply_causal_nodal_boundaries,
    apply_torch_boundary_policy,
    build_boundary_stencil,
    build_torch_boundary_policy,
    freestream_primitive,
    legal_outflow_normal_mach_min,
    recover_graph_boundary_geometry,
)
from utility.time_dependent_no.cpg_release import (  # noqa: E402
    CPG_LEGAL_BOUNDARY_MAX_SOURCE_HOPS,
    CPG_MESH_AUDIT_SCHEMA,
    CPG_MODEL_DT,
    cpg_local_training_source_manifest,
    sha256_file,
    validate_cpg_reference_source,
    validate_cpg_model_dt,
    validate_cpg_static_graph,
)

RUN_SCHEMA = "cpg_legal_boundary_training_v1"
POLICY_VALIDATION_SCHEMA = CPG_MESH_AUDIT_SCHEMA
BOUNDARY_MODE = "causal_nodal_physical"
PHYSICAL_CONFIG = {"gamma": 1.4, "rho_inf": 1.4, "p_inf": 1.0}
STATE_KEYS = ("rho", "v1", "v2", "pres")
DATA_KEYS = ("pos", "edges", "node_type", "pres", "rho", "v1", "v2", "Mach")


@dataclass(frozen=True)
class PolicySpec:
    trajectory_key: str
    num_nodes: int
    node_type: np.ndarray
    node_normal: np.ndarray
    wall_normal_coherence: np.ndarray
    stencil: BoundaryStencil
    mach: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-repo", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--graph-policy-validation-summary", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--dt", type=float, default=CPG_MODEL_DT)
    parser.add_argument("--teacher-forcing-batch-size", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=3)
    parser.add_argument("--num-steps", type=int, default=3)
    parser.add_argument("--stage1-epochs", type=int, default=15)
    parser.add_argument("--stage2-epochs", type=int, default=5)
    parser.add_argument("--stage1-lr", type=float, default=1.0e-4)
    parser.add_argument("--stage2-lr", type=float, default=1.0e-5)
    parser.add_argument("--max-trajectories", type=int)
    parser.add_argument("--max-batches-per-epoch", type=int)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--expected-dataset-sha256")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if args.split != "train":
        raise ValueError("legal-boundary training is pinned to --split train")
    args.dt = validate_cpg_model_dt(args.dt)
    for name in (
        "teacher_forcing_batch_size",
        "batch_size",
        "gradient_accumulation_steps",
        "num_steps",
        "stage1_lr",
        "stage2_lr",
    ):
        value = getattr(args, name)
        if (isinstance(value, float) and not np.isfinite(value)) or value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.stage1_epochs < 0 or args.stage2_epochs < 0:
        raise ValueError("stage epoch counts cannot be negative")
    if not args.audit_only and args.stage1_epochs + args.stage2_epochs == 0:
        raise ValueError("training requires at least one epoch")
    if args.batch_size * args.gradient_accumulation_steps != 3:
        raise ValueError(
            "microbatch size times gradient accumulation must equal the released "
            "effective batch size three"
        )
    if args.teacher_forcing_batch_size != 3:
        raise ValueError(
            "teacher-forcing batch size must equal released batch size three"
        )
    for name in ("max_trajectories", "max_batches_per_epoch"):
        value = getattr(args, name)
        if value is not None and value < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    return args


def _import_reference_api(reference_repo: Path) -> dict[str, Any]:
    try:
        import h5py  # type: ignore[import-not-found]
        import torch
        from torch_geometric.loader import DataLoader
        import torch_geometric.transforms as T
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "training requires h5py, torch, and torch-geometric"
        ) from exc
    reference = str(reference_repo.resolve())
    if reference not in sys.path:
        sys.path.insert(0, reference)
    try:
        from dataset.fpcMulti import FPCBase
        from modelEdgeUpd.simulator import Simulator
        from utils.lossCompute import dataLoss
        from utils.noise import get_noise
        from utils.to_undirected import make_edges_undirected
    except ModuleNotFoundError as exc:
        raise RuntimeError("the pinned external CPGNet checkout is incomplete") from exc
    return {
        "h5py": h5py,
        "torch": torch,
        "DataLoader": DataLoader,
        "T": T,
        "FPCBase": FPCBase,
        "Simulator": Simulator,
        "dataLoss": dataLoss,
        "get_noise": get_noise,
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


def _set_seed(torch: Any, seed: int, *, seed_cuda: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if seed_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_policy_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("graph policy validation root must be an object")
    schema = payload.get("schema")
    if schema != POLICY_VALIDATION_SCHEMA:
        raise ValueError(
            "new legal-boundary training requires the hardened v2 graph policy"
        )
    if payload.get("status") != "valid":
        raise ValueError("graph policy validation did not pass")
    cases = payload.get("cases")
    if not isinstance(cases, list):
        raise ValueError("graph policy validation cases must be a list")
    if payload.get("case_count") != len(cases) or len(cases) != 20:
        raise ValueError("graph policy validation must cover all 20 test cases")
    if payload.get("graph_mesh_identity") != "verified_all_selected_cases":
        raise ValueError("graph policy validation lacks graph-mesh identity")
    if payload.get("boundary_geometry") != "verified_all_selected_cases":
        raise ValueError("graph policy validation lacks boundary geometry")
    if payload.get("case_offset") != 1:
        raise ValueError("graph policy validation does not use one-based raw cases")
    if (
        payload.get("temporal_alignment_evidence")
        != "sampled_first_midpoint_last_not_full_temporal_identity"
        or payload.get("static_graph_evidence")
        != "all_frames_exact_release_facing_metadata"
    ):
        raise ValueError("graph policy validation lacks hardened evidence")
    max_source_hops = payload.get("max_boundary_source_hops")
    if (
        not isinstance(max_source_hops, int)
        or isinstance(max_source_hops, bool)
        or max_source_hops != CPG_LEGAL_BOUNDARY_MAX_SOURCE_HOPS
    ):
        raise ValueError(
            "graph policy validation uses a different boundary source-hop contract"
        )
    normal_errors = []
    coherence_errors = []
    trajectory_keys: set[str] = set()
    raw_case_ids: set[int] = set()
    causal_outflow_mach_minima = []
    for record in cases:
        if not isinstance(record, Mapping):
            raise ValueError("graph policy case records must be objects")
        trajectory_key = record.get("trajectory_key")
        raw_case_id = record.get("raw_case_id")
        if (
            not isinstance(trajectory_key, str)
            or not trajectory_key
            or not trajectory_key.isdigit()
            or trajectory_key in trajectory_keys
            or not isinstance(raw_case_id, int)
            or isinstance(raw_case_id, bool)
            or raw_case_id != int(trajectory_key) + 1
            or raw_case_id in raw_case_ids
        ):
            raise ValueError(
                "graph policy cases need the unique zero-to-one-based case mapping"
            )
        trajectory_keys.add(trajectory_key)
        raw_case_ids.add(raw_case_id)
        mesh = record.get("mesh")
        if not isinstance(mesh, Mapping):
            raise ValueError("graph policy case lacks mesh evidence")
        if mesh.get("graph_native_boundary_mapping") != (
            "verified_against_abaqus_boundary"
        ):
            raise ValueError("graph-native policy was not validated on every case")
        try:
            normal_error = float(mesh["graph_native_normal_max_abs_error"])
            coherence_error = float(
                mesh["graph_native_normal_coherence_max_abs_error"]
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("graph policy geometry errors are invalid") from exc
        if not np.isfinite(normal_error) or not np.isfinite(coherence_error):
            raise ValueError("graph policy geometry errors must be finite")
        normal_errors.append(normal_error)
        coherence_errors.append(coherence_error)
        config = record.get("config")
        if not isinstance(config, Mapping):
            raise ValueError("graph policy case lacks physical configuration")
        try:
            validate_cpg_model_dt(config.get("save_dt"))
            physical = {
                name: float(config[name])
                for name in ("gamma", "rho_inf", "p_inf")
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "graph policy validation has an invalid physical configuration"
            ) from exc
        if (
            not all(np.isfinite(value) for value in physical.values())
            or physical != PHYSICAL_CONFIG
        ):
            raise ValueError(
                "graph policy validation uses a different physical configuration"
            )
        boundary = record.get("boundary")
        if not isinstance(boundary, Mapping):
            raise ValueError("graph policy case lacks causal boundary evidence")
        try:
            causal_min = float(boundary["causal_outflow_normal_mach_min"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("graph policy case lacks causal outflow evidence") from exc
        if not np.isfinite(causal_min) or causal_min <= 1.0:
            raise ValueError("graph policy causal outflow is not outward-supersonic")
        if boundary.get("causal_outflow_evidence") != (
            "all_frames_extrapolated_from_interior_stencil"
        ):
            raise ValueError("graph policy causal outflow evidence is incomplete")
        if not _is_sha256(boundary.get("boundary_stencil_sha256")):
            raise ValueError("graph policy case lacks a valid boundary-stencil digest")
        causal_outflow_mach_minima.append(causal_min)
    if max(normal_errors + coherence_errors) > GRAPH_BOUNDARY_GEOMETRY_ATOL:
        raise ValueError("graph-native policy validation exceeds its tolerance")
    dataset_sha256 = payload.get("dataset_sha256")
    if not _is_sha256(dataset_sha256):
        raise ValueError("graph policy validation lacks a dataset SHA256")
    return {
        "file_name": path.name,
        "sha256": sha256_file(path),
        "case_count": len(cases),
        "dataset_sha256": dataset_sha256,
        "schema": payload["schema"],
        "status": payload["status"],
        "contract_completeness": (
            "hardened_sampled_alignment_and_all_frame_static_graph"
        ),
        "max_boundary_source_hops": max_source_hops,
        "graph_mesh_identity": payload["graph_mesh_identity"],
        "boundary_geometry": payload["boundary_geometry"],
        "max_normal_abs_error": max(normal_errors),
        "max_normal_coherence_abs_error": max(coherence_errors),
        "causal_outflow_normal_mach_min": min(causal_outflow_mach_minima),
        "raw_mesh_contract": "validated_on_available_test_cases",
    }


def _primitive(group: Any, frame: int) -> np.ndarray:
    primitive = np.concatenate(
        [np.asarray(group[name][frame], dtype=np.float64) for name in STATE_KEYS],
        axis=1,
    )
    if not np.all(np.isfinite(primitive)):
        raise ValueError(f"HDF primitive frame {frame} contains nonfinite values")
    return primitive


def _validate_all_primitive_frames(
    group: Any,
    key: str,
    *,
    chunk_frames: int = 8,
) -> tuple[float, float]:
    if chunk_frames <= 0:
        raise ValueError("chunk_frames must be positive")
    minima: dict[str, float] = {}
    for name in STATE_KEYS:
        dataset = group[name]
        field_minimum = float("inf")
        for start in range(0, int(dataset.shape[0]), chunk_frames):
            values = np.asarray(
                dataset[start : start + chunk_frames],
                dtype=np.float64,
            )
            if not np.all(np.isfinite(values)):
                raise ValueError(
                    f"trajectory {key} field {name} contains nonfinite values"
                )
            field_minimum = min(field_minimum, float(np.min(values)))
        minima[name] = field_minimum
    if minima["rho"] <= 0.0 or minima["pres"] <= 0.0:
        raise ValueError(f"trajectory {key} contains inadmissible primitive states")
    return minima["rho"], minima["pres"]


def _audit_case(
    group: Any, key: str, *, max_source_hops: int
) -> tuple[PolicySpec, dict[str, Any]]:
    frames = int(group["pres"].shape[0])
    if frames <= 3:
        raise ValueError(f"trajectory {key} has too few frames")
    pos, edges, node_type, mach_field = validate_cpg_static_graph(group)
    state_density_min, state_pressure_min = _validate_all_primitive_frames(group, key)
    pos = np.asarray(pos, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.int64)
    node_type = np.asarray(node_type, dtype=np.int64)
    mach_field = np.asarray(mach_field, dtype=np.float64)
    mach = float(mach_field[0])
    if not np.isfinite(mach) or not np.all(np.isfinite(mach_field)):
        raise ValueError(f"trajectory {key} Mach contains nonfinite values")
    if float(np.max(np.abs(mach_field - mach))) > 2.0e-10:
        raise ValueError(f"trajectory {key} Mach is not spatially constant")
    geometry = recover_graph_boundary_geometry(
        pos=pos, edges=edges, node_type=node_type
    )
    stencil = build_boundary_stencil(
        pos=pos,
        edges=edges,
        node_type=node_type,
        node_normal=geometry["node_normal"],
        max_source_hops=max_source_hops,
    )
    if stencil.fallback_target_count:
        raise ValueError(f"trajectory {key} boundary stencil used a fallback")
    unique_sources, source_inverse = np.unique(
        stencil.source_nodes, return_inverse=True
    )
    source_fields = []
    for name in STATE_KEYS:
        values = np.asarray(
            group[name][:, unique_sources, :],
            dtype=np.float64,
        )
        source_fields.append(values[:, source_inverse, :])
    source_state = np.concatenate(source_fields, axis=2)
    extrapolated = np.zeros(
        (frames, stencil.target_nodes.size, 4),
        dtype=np.float64,
    )
    for entry, (target_row, weight) in enumerate(
        zip(stencil.target_rows, stencil.weights, strict=True)
    ):
        extrapolated[:, int(target_row), :] += (
            source_state[:, entry, :] * float(weight)
        )
    outflow_rows = np.flatnonzero(
        node_type[stencil.target_nodes] == OUTFLOW_NODE
    )
    if outflow_rows.size == 0:
        raise ValueError(f"trajectory {key} boundary stencil has no outflow targets")
    causal_outflow = extrapolated[:, outflow_rows, :]
    if (
        not np.all(np.isfinite(causal_outflow))
        or np.any(causal_outflow[:, :, 0] <= 0.0)
        or np.any(causal_outflow[:, :, 3] <= 0.0)
    ):
        raise ValueError(
            f"trajectory {key} causal outflow is not physically admissible"
        )
    outflow_nodes = stencil.target_nodes[outflow_rows]
    outflow_normals = geometry["node_normal"][outflow_nodes]
    causal_normal_speed = np.sum(
        causal_outflow[:, :, 1:3] * outflow_normals[None, :, :],
        axis=2,
    )
    causal_sound_speed = np.sqrt(
        PHYSICAL_CONFIG["gamma"]
        * causal_outflow[:, :, 3]
        / causal_outflow[:, :, 0]
    )
    causal_outflow_normal_mach_min = float(
        np.min(causal_normal_speed / causal_sound_speed)
    )
    if causal_outflow_normal_mach_min <= 1.0:
        raise ValueError(
            f"trajectory {key} causal outflow is not outward-supersonic"
        )

    inflow_error = 0.0
    wall_normal_velocity = 0.0
    outflow_normal_mach = float("inf")
    boundary_remap_rmse = 0.0
    stream = freestream_primitive(mach, **PHYSICAL_CONFIG)
    audit_frames = sorted({0, min(20, frames - 1), min(40, frames - 1), frames - 1})
    for frame in audit_frames:
        state = _primitive(group, frame)
        if np.any(state[:, 0] <= 0.0) or np.any(state[:, 3] <= 0.0):
            raise ValueError(
                f"trajectory {key} frame {frame} is not physically admissible"
            )
        inflow = node_type == INFLOW_NODE
        wall = node_type == WALL_NODE
        outflow = node_type == OUTFLOW_NODE
        inflow_error = max(inflow_error, float(np.max(np.abs(state[inflow] - stream))))
        wall_velocity = np.sum(state[wall, 1:3] * geometry["node_normal"][wall], axis=1)
        wall_normal_velocity = max(
            wall_normal_velocity, float(np.max(np.abs(wall_velocity)))
        )
        sound = np.sqrt(
            PHYSICAL_CONFIG["gamma"] * state[outflow, 3] / state[outflow, 0]
        )
        normal_speed = np.sum(
            state[outflow, 1:3] * geometry["node_normal"][outflow], axis=1
        )
        outflow_normal_mach = min(
            outflow_normal_mach, float(np.min(normal_speed / sound))
        )
        legal = apply_causal_nodal_boundaries(
            state,
            node_type=node_type,
            node_normal=geometry["node_normal"],
            wall_normal_coherence=geometry["node_boundary_normal_coherence"],
            stencil=stencil,
            mach=mach,
            **PHYSICAL_CONFIG,
        )
        boundary = node_type != NORMAL_NODE
        boundary_remap_rmse = max(
            boundary_remap_rmse,
            float(np.sqrt(np.mean((legal[boundary] - state[boundary]) ** 2))),
        )
    if outflow_normal_mach <= 1.0:
        raise ValueError(f"trajectory {key} outflow is not supersonic")
    spec = PolicySpec(
        trajectory_key=key,
        num_nodes=int(node_type.size),
        node_type=node_type,
        node_normal=geometry["node_normal"],
        wall_normal_coherence=geometry["node_boundary_normal_coherence"],
        stencil=stencil,
        mach=mach,
    )
    record = {
        "trajectory_key": key,
        "frames": frames,
        "nodes": int(node_type.size),
        "edges": int(edges.shape[0]),
        "mach": mach,
        "sharp_wall_nodes": int(
            np.count_nonzero(
                (node_type == WALL_NODE)
                & (geometry["node_boundary_normal_coherence"] < 0.95)
            )
        ),
        "inflow_reference_max_abs_error": inflow_error,
        "wall_reference_normal_velocity_max_abs": wall_normal_velocity,
        "outflow_reference_normal_mach_min": outflow_normal_mach,
        "causal_outflow_normal_mach_min": causal_outflow_normal_mach_min,
        "boundary_reference_remap_rmse_max": boundary_remap_rmse,
        "all_frame_density_min": state_density_min,
        "all_frame_pressure_min": state_pressure_min,
        "normalize_scale_center": np.mean(pos, axis=0).tolist(),
        "normalize_scale_max_abs_after_center": float(
            np.max(np.abs(pos - np.mean(pos, axis=0)))
        ),
    }
    return spec, record


def _audit_dataset(
    *,
    h5py: Any,
    dataset_path: Path,
    keys: Sequence[str],
    max_source_hops: int = 3,
) -> tuple[list[PolicySpec], dict[str, Any]]:
    specs = []
    records = []
    with h5py.File(dataset_path, "r") as handle:
        for key in keys:
            spec, record = _audit_case(
                handle[key], key, max_source_hops=max_source_hops
            )
            specs.append(spec)
            records.append(record)
    normalize_scale_signatures = {
        (
            tuple(record["normalize_scale_center"]),
            record["normalize_scale_max_abs_after_center"],
        )
        for record in records
    }
    summary = {
        "case_count": len(records),
        "selected_keys": list(keys),
        "frames": sorted({record["frames"] for record in records}),
        "total_frames": sum(record["frames"] for record in records),
        "node_count_range": [
            min(record["nodes"] for record in records),
            max(record["nodes"] for record in records),
        ],
        "mach_range": [
            min(record["mach"] for record in records),
            max(record["mach"] for record in records),
        ],
        "sharp_wall_node_counts": sorted(
            {record["sharp_wall_nodes"] for record in records}
        ),
        "max_inflow_reference_abs_error": max(
            record["inflow_reference_max_abs_error"] for record in records
        ),
        "max_wall_reference_normal_velocity": max(
            record["wall_reference_normal_velocity_max_abs"] for record in records
        ),
        "min_outflow_reference_normal_mach": min(
            record["outflow_reference_normal_mach_min"] for record in records
        ),
        "min_causal_outflow_normal_mach": min(
            record["causal_outflow_normal_mach_min"] for record in records
        ),
        "max_boundary_reference_remap_rmse": max(
            record["boundary_reference_remap_rmse_max"] for record in records
        ),
        "boundary_stencil_fallback_targets": 0,
        "graph_geometry_static": True,
        "normalize_scale_affine_shared_across_trajectories": (
            len(normalize_scale_signatures) == 1
        ),
    }
    return specs, summary


def _dataset_class(torch: Any, h5py: Any, fpc_base: Any) -> Any:
    class TrajectoryWindows(torch.utils.data.IterableDataset):
        def __init__(
            self,
            dataset_path: Path,
            keys: Sequence[str],
            *,
            num_steps: int,
            dt: float,
            seed: int,
        ) -> None:
            super().__init__()
            self.dataset_path = dataset_path
            self.keys = list(keys)
            self.num_steps = num_steps
            self.dt = dt
            self.seed = seed
            self.epoch = 0

        def set_epoch(self, epoch: int) -> None:
            self.epoch = epoch

        def __iter__(self):
            if torch.utils.data.get_worker_info() is not None:
                raise RuntimeError("legal CPG training requires num_workers=0")
            rng = np.random.default_rng(self.seed + self.epoch)
            with h5py.File(self.dataset_path, "r") as handle:
                for policy_index in rng.permutation(len(self.keys)):
                    group = handle[self.keys[int(policy_index)]]
                    frame_count = int(group["pres"].shape[0])
                    frames = rng.permutation(frame_count - self.num_steps)
                    for frame in frames:
                        data = []
                        for name in DATA_KEYS:
                            if name in STATE_KEYS:
                                value = np.asarray(
                                    group[name][
                                        int(frame) : int(frame) + self.num_steps + 1
                                    ],
                                    dtype=np.float32,
                                )
                            else:
                                value = np.asarray(group[name][0])
                                if name in ("edges", "node_type"):
                                    value = value.astype(np.int32)
                            data.append(value)
                        data.append(
                            np.asarray([self.dt * int(frame)], dtype=np.float32)
                        )
                        graph = fpc_base.datas_to_graph(data)
                        graph.policy_id = torch.tensor(
                            [int(policy_index)], dtype=torch.long
                        )
                        graph.source_frame = torch.tensor(
                            [int(frame)], dtype=torch.long
                        )
                        yield graph

    return TrajectoryWindows


def _concatenate_policies(
    torch: Any,
    *,
    policies: Sequence[Mapping[str, Any]],
    policy_indices: Any,
    ptr: Any,
    device: Any,
) -> dict[str, Any]:
    fields: dict[str, list[Any]] = {
        name: []
        for name in (
            "target_nodes",
            "target_rows",
            "source_nodes",
            "weights",
            "wall_rows",
            "outflow_rows",
            "sharp_wall_rows",
            "target_normals",
            "inflow_nodes",
            "freestream",
        )
    }
    gammas = {float(policy["gamma"]) for policy in policies}
    if len(gammas) != 1:
        raise ValueError("batched boundary policies must use one shared gamma")
    target_offset = 0
    for batch_index, policy_index in enumerate(policy_indices.reshape(-1).tolist()):
        policy = policies[int(policy_index)]
        node_offset = int(ptr[batch_index])
        expected_nodes = int(ptr[batch_index + 1] - ptr[batch_index])
        if policy["num_nodes"] != expected_nodes:
            raise ValueError("batched graph and boundary policy node counts differ")
        fields["target_nodes"].append(policy["target_nodes"] + node_offset)
        fields["target_rows"].append(policy["target_rows"] + target_offset)
        fields["source_nodes"].append(policy["source_nodes"] + node_offset)
        fields["weights"].append(policy["weights"])
        fields["wall_rows"].append(policy["wall_rows"] + target_offset)
        fields["outflow_rows"].append(policy["outflow_rows"] + target_offset)
        fields["sharp_wall_rows"].append(policy["sharp_wall_rows"] + target_offset)
        fields["target_normals"].append(policy["target_normals"])
        fields["inflow_nodes"].append(policy["inflow_nodes"] + node_offset)
        fields["freestream"].append(
            policy["freestream"][None, :].expand(policy["inflow_nodes"].numel(), -1)
        )
        target_offset += int(policy["target_nodes"].numel())
    return {
        "num_nodes": int(ptr[-1]),
        "gamma": gammas.pop(),
        **{
            name: torch.cat(values, dim=0).to(device) for name, values in fields.items()
        },
    }


def _legal_recurrent_outflow_mach_min(
    torch: Any,
    physical_prediction: Any,
    policy: Mapping[str, Any],
    *,
    step: int,
) -> float:
    closed = apply_torch_boundary_policy(
        torch, physical_prediction.detach(), policy
    )
    density_min = float(closed[:, 0].min().cpu())
    pressure_min = float(closed[:, 3].min().cpu())
    if density_min <= 0.0 or pressure_min <= 0.0:
        raise FloatingPointError(
            f"inadmissible recurrent prediction at rollout step {step}"
        )
    outflow_mach = legal_outflow_normal_mach_min(torch, closed, policy)
    if outflow_mach is None or not np.isfinite(outflow_mach) or outflow_mach <= 1.0:
        raise FloatingPointError(
            f"non-supersonic recurrent outflow at rollout step {step}"
        )
    return outflow_mach


def _record_step_loss(
    total_loss: Any, step_loss: Any, backward_weight: float | None
) -> Any:
    if backward_weight is None:
        return total_loss + step_loss
    (step_loss * backward_weight).backward()
    return total_loss + step_loss.detach()


def _microbatch_loss(
    *,
    api: Mapping[str, Any],
    model: Any,
    graph: Any,
    transformer: Any,
    policies: Sequence[Mapping[str, Any]],
    device: Any,
    noise_std: Any,
    teacher_forcing: bool,
    backward_weight: float | None,
) -> tuple[Any, Any, Any, float | None, int, int]:
    torch = api["torch"]
    policy = _concatenate_policies(
        torch,
        policies=policies,
        policy_indices=graph.policy_id,
        ptr=graph.ptr,
        device=device,
    )
    graph = api["make_edges_undirected"](graph)
    graph = transformer(graph)
    graph = graph.to(device)
    if graph.x.ndim != 2 or graph.x.shape[1] != 6:
        raise ValueError(
            f"transformed graph.x must have shape (nodes, 6), got {tuple(graph.x.shape)}"
        )
    if (
        graph.future_prims.ndim != 2
        or graph.future_prims.shape[0] != graph.x.shape[0]
        or graph.future_prims.shape[1] == 0
        or graph.future_prims.shape[1] % 4 != 0
    ):
        raise ValueError(
            "future_prims must have shape (nodes, positive multiple of four)"
        )
    if not bool(torch.isfinite(graph.x).all()) or not bool(
        torch.isfinite(graph.future_prims).all()
    ):
        raise FloatingPointError("training graph contains nonfinite state features")
    node_type = graph.x[:, 0]
    normal_mask = node_type == NORMAL_NODE
    boundary_mask = ~normal_mask
    current = graph.x[:, 1:5]
    total_loss = current.new_zeros(())
    min_density = None
    min_pressure = None
    min_recurrent_outflow_mach = None
    num_steps = int(graph.future_prims.shape[1] // 4)
    for step in range(num_steps):
        target = graph.future_prims[:, step * 4 : (step + 1) * 4]
        current = apply_torch_boundary_policy(torch, current, policy)
        graph.x[:, 1:5] = current
        graph.y = target
        if teacher_forcing or step == 0:
            base_noise = api["get_noise"](
                graph,
                boundary_mask,
                noise_std=noise_std,
                device=device,
            )
            legal_noised = apply_torch_boundary_policy(
                torch, current + base_noise, policy
            )
            sequence_noise = legal_noised - current
        else:
            sequence_noise = torch.zeros_like(current)
        predicted, target_normalized = model(graph.clone(), sequence_noise)
        expected_shape = (graph.x.shape[0], 4)
        if tuple(predicted.shape) != expected_shape or tuple(
            target_normalized.shape
        ) != expected_shape:
            raise ValueError(
                "model prediction and normalized target must both have shape "
                f"{expected_shape}"
            )
        step_loss = api["dataLoss"](graph, predicted, target_normalized, normal_mask)
        physical_prediction = model._output_normalizer.inverse(predicted)
        if not bool(
            torch.isfinite(step_loss) & torch.isfinite(physical_prediction).all()
        ):
            raise FloatingPointError(
                f"nonfinite loss or physical prediction at rollout step {step}"
            )
        density = physical_prediction[:, 0].detach().min()
        pressure = physical_prediction[:, 3].detach().min()
        min_density = (
            density if min_density is None else torch.minimum(min_density, density)
        )
        min_pressure = (
            pressure if min_pressure is None else torch.minimum(min_pressure, pressure)
        )
        if not teacher_forcing:
            outflow_mach = _legal_recurrent_outflow_mach_min(
                torch, physical_prediction, policy, step=step
            )
            min_recurrent_outflow_mach = (
                outflow_mach
                if min_recurrent_outflow_mach is None
                else min(min_recurrent_outflow_mach, outflow_mach)
            )
        current = target if teacher_forcing else physical_prediction
        total_loss = _record_step_loss(total_loss, step_loss, backward_weight)
    if not bool(torch.isfinite(total_loss)):
        raise FloatingPointError("nonfinite microbatch loss")
    assert min_density is not None and min_pressure is not None
    return (
        total_loss,
        min_density,
        min_pressure,
        min_recurrent_outflow_mach,
        int(normal_mask.sum()),
        int(graph.num_graphs),
    )


def _finite_gradient_norm(torch: Any, optimizer: Any) -> float:
    parameters = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
        if parameter.grad is not None
    ]
    if not parameters:
        raise FloatingPointError("optimizer step has no gradients")
    try:
        norm = torch.nn.utils.clip_grad_norm_(
            parameters,
            max_norm=float("inf"),
            error_if_nonfinite=True,
        )
    except RuntimeError as exc:
        raise FloatingPointError("optimizer step has nonfinite gradients") from exc
    value = float(norm.detach().cpu())
    if not np.isfinite(value):
        raise FloatingPointError("optimizer step has a nonfinite gradient norm")
    return value


def _assert_finite_training_state(torch: Any, model: Any, optimizer: Any) -> None:
    with torch.no_grad():
        for group_index, group in enumerate(optimizer.param_groups):
            for parameter_index, parameter in enumerate(group["params"]):
                if not bool(torch.isfinite(parameter).all()):
                    raise FloatingPointError(
                        "nonfinite model parameter after epoch "
                        f"(group={group_index}, index={parameter_index})"
                    )
        if hasattr(model, "named_buffers"):
            for name, buffer in model.named_buffers():
                if buffer is not None and not bool(torch.isfinite(buffer).all()):
                    raise FloatingPointError(f"nonfinite model buffer {name!r}")
        for normalizer_name in ("_output_normalizer", "_node_normalizer"):
            normalizer = getattr(model, normalizer_name, None)
            if normalizer is None:
                continue
            for field in (
                "_std_epsilon",
                "_acc_count",
                "_num_accumulations",
                "_acc_sum",
                "_acc_sum_squared",
            ):
                value = getattr(normalizer, field, None)
                if torch.is_tensor(value) and not bool(torch.isfinite(value).all()):
                    raise FloatingPointError(
                        f"nonfinite {normalizer_name}.{field}"
                    )
        for state in optimizer.state.values():
            for name, value in state.items():
                if torch.is_tensor(value) and not bool(torch.isfinite(value).all()):
                    raise FloatingPointError(
                        f"nonfinite optimizer state tensor {name!r}"
                    )


def _require_frozen_normalizers(model: Any) -> None:
    for name in ("_output_normalizer", "_node_normalizer"):
        normalizer = getattr(model, name, None)
        if normalizer is None:
            raise RuntimeError(f"model lacks required normalizer {name}")
        accumulated = float(normalizer._num_accumulations.detach().cpu())
        maximum = float(normalizer._max_accumulations)
        if accumulated < maximum:
            raise RuntimeError(
                "sequential multistep microbatches require frozen normalizers"
            )


def _normalizer_diagnostics(normalizer: Any) -> dict[str, Any]:
    return {
        "accumulation_calls": float(
            normalizer._num_accumulations.detach().cpu()
        ),
        "accumulated_node_count_float32": float(
            normalizer._acc_count.detach().cpu()
        ),
        "mean": normalizer._mean().detach().cpu().reshape(-1).tolist(),
        "std_with_epsilon": normalizer._std_with_epsilon()
        .detach()
        .cpu()
        .reshape(-1)
        .tolist(),
    }


def _train_epoch(
    *,
    api: Mapping[str, Any],
    model: Any,
    loader: Any,
    optimizer: Any,
    transformer: Any,
    policies: Sequence[Mapping[str, Any]],
    device: Any,
    noise_std: Any,
    teacher_forcing: bool,
    max_batches: int | None,
    accumulation_steps: int,
) -> dict[str, Any]:
    torch = api["torch"]
    model.train()
    loss_sum = 0.0
    batches = 0
    microbatches = 0
    samples = 0
    min_density = None
    min_pressure = None
    min_recurrent_outflow_mach = None
    gradient_norm_sum = 0.0
    max_gradient_norm = 0.0
    started = time.time()
    iterator = iter(loader)
    while True:
        if max_batches is not None and batches >= max_batches:
            break
        group = list(islice(iterator, accumulation_steps))
        if not group:
            break
        if len(group) != accumulation_steps:
            raise RuntimeError(
                "training epoch ended with an incomplete gradient-accumulation group"
            )
        normal_counts = [int((graph.x[:, 0] == NORMAL_NODE).sum()) for graph in group]
        total_normal_nodes = sum(normal_counts)
        if total_normal_nodes <= 0:
            raise RuntimeError("gradient-accumulation group has no normal nodes")
        optimizer.zero_grad(set_to_none=True)
        batch_loss = 0.0
        optimizer_samples = 0
        for graph, expected_normal_nodes in zip(group, normal_counts, strict=True):
            weight = expected_normal_nodes / total_normal_nodes
            (
                microbatch_loss,
                microbatch_min_density,
                microbatch_min_pressure,
                microbatch_min_outflow_mach,
                actual_normal_nodes,
                microbatch_samples,
            ) = _microbatch_loss(
                api=api,
                model=model,
                graph=graph,
                transformer=transformer,
                policies=policies,
                device=device,
                noise_std=noise_std,
                teacher_forcing=teacher_forcing,
                backward_weight=weight if teacher_forcing else None,
            )
            if actual_normal_nodes != expected_normal_nodes:
                raise RuntimeError("normal-node count changed during preprocessing")
            if not teacher_forcing:
                (microbatch_loss * weight).backward()
            batch_loss += float(microbatch_loss.detach().cpu()) * weight
            microbatches += 1
            samples += microbatch_samples
            optimizer_samples += microbatch_samples
            density = float(microbatch_min_density.cpu())
            pressure = float(microbatch_min_pressure.cpu())
            min_density = density if min_density is None else min(min_density, density)
            min_pressure = (
                pressure if min_pressure is None else min(min_pressure, pressure)
            )
            if microbatch_min_outflow_mach is not None:
                min_recurrent_outflow_mach = (
                    microbatch_min_outflow_mach
                    if min_recurrent_outflow_mach is None
                    else min(
                        min_recurrent_outflow_mach,
                        microbatch_min_outflow_mach,
                    )
                )
        if optimizer_samples != 3:
            raise RuntimeError(
                "each optimizer update must contain exactly three trajectory windows"
            )
        gradient_norm = _finite_gradient_norm(torch, optimizer)
        optimizer.step()
        gradient_norm_sum += gradient_norm
        max_gradient_norm = max(max_gradient_norm, gradient_norm)
        batches += 1
        loss_sum += batch_loss
        if (batches - 1) % 20 == 0:
            print(
                json.dumps(
                    {
                        "batch": batches - 1,
                        "loss": batch_loss,
                        "microbatches": len(group),
                        "teacher_forcing": teacher_forcing,
                        "gradient_norm": gradient_norm,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    if batches == 0:
        raise RuntimeError("training epoch produced no batches")
    _assert_finite_training_state(torch, model, optimizer)
    assert min_density is not None and min_pressure is not None
    return {
        "mean_total_loss": loss_sum / batches,
        "batches": batches,
        "microbatches": microbatches,
        "gradient_accumulation_steps": accumulation_steps,
        "mean_gradient_norm": gradient_norm_sum / batches,
        "max_gradient_norm": max_gradient_norm,
        "samples": samples,
        "wall_seconds": time.time() - started,
        "min_raw_predicted_density": min_density,
        "min_raw_predicted_pressure": min_pressure,
        "min_recurrent_outflow_normal_mach": min_recurrent_outflow_mach,
        "teacher_forcing": teacher_forcing,
    }


def _write_manifest(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    local_source_provenance = cpg_local_training_source_manifest(REPO_ROOT)
    reference = validate_cpg_reference_source(
        args.reference_repo,
        include_training_dependencies=True,
    )
    policy_validation = _validate_policy_summary(args.graph_policy_validation_summary)
    dataset_path = args.dataset_root / f"{args.split}.h5"
    if not dataset_path.is_file():
        raise FileNotFoundError(dataset_path)
    if args.output_dir.exists():
        raise FileExistsError(
            "output directory already exists; choose a new run directory: "
            f"{args.output_dir}"
        )
    dataset_sha256 = sha256_file(dataset_path)
    if (
        args.expected_dataset_sha256 is not None
        and dataset_sha256 != args.expected_dataset_sha256
    ):
        raise ValueError("training dataset SHA256 differs from the expected digest")
    api = _import_reference_api(args.reference_repo)
    torch = api["torch"]
    _set_seed(torch, args.seed, seed_cuda=not args.audit_only and args.device != "cpu")
    with api["h5py"].File(dataset_path, "r") as handle:
        available_keys = [str(key) for key in handle.keys()]
    keys = (
        available_keys
        if args.max_trajectories is None
        else available_keys[: args.max_trajectories]
    )
    if not keys:
        raise ValueError("training split contains no selected trajectories")
    specs, dataset_audit = _audit_dataset(
        h5py=api["h5py"],
        dataset_path=dataset_path,
        keys=keys,
        max_source_hops=policy_validation["max_boundary_source_hops"],
    )
    if args.num_steps >= min(dataset_audit["frames"]):
        raise ValueError("--num-steps must be smaller than every trajectory")
    trajectory_window_count = (
        dataset_audit["total_frames"] - len(keys) * args.num_steps
    )
    dataset_audit["trajectory_window_count"] = trajectory_window_count
    if (
        not args.audit_only
        and trajectory_window_count % args.teacher_forcing_batch_size
    ):
        raise ValueError(
            "selected trajectory windows do not form complete effective batches"
        )
    if (
        not args.audit_only
        and args.stage2_epochs
        and args.gradient_accumulation_steps > 1
        and not dataset_audit[
            "normalize_scale_affine_shared_across_trajectories"
        ]
    ):
        raise ValueError(
            "sequential multistep microbatches require one shared NormalizeScale "
            "affine transform"
        )

    promotion_eligible = (
        not args.audit_only
        and args.max_trajectories is None
        and args.max_batches_per_epoch is None
    )
    args.output_dir.mkdir(parents=True)
    manifest_path = args.output_dir / "run_manifest.json"
    manifest: dict[str, Any] = {
        "schema": RUN_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "audited" if args.audit_only else "running",
        "claim_scope": (
            (
                "release-bundle legal-nodal-boundary retraining; not paper identity "
                "or exact DG boundary replay"
            )
            if promotion_eligible
            else "bounded audit/debug run; not eligible for trained-baseline claims"
        ),
        "promotion_eligible": promotion_eligible,
        "reference": reference,
        "runtime": _runtime_manifest(),
        "boundary_mode": BOUNDARY_MODE,
        "uses_future_reference_boundary": False,
        "uses_future_reference_boundary_in_rollout_inputs_or_recurrent_state": False,
        "future_reference_boundary_training_use": {
            "supervised_target": True,
            "output_normalizer_statistics": True,
            "model_input": False,
            "recurrent_state": False,
        },
        "model_configuration": {
            "message_passing_num": 12,
            "node_input_size": 6,
            "edge_input_size": 5,
            "dt": args.dt,
            "dt_source": "pinned modelEdgeUpd/modelEU.py DELTA_T",
        },
        "training_configuration": {
            "seed": args.seed,
            "teacher_forcing_batch_size": args.teacher_forcing_batch_size,
            "teacher_forcing_gradient_accumulation_steps": 1,
            "multistep_microbatch_size": args.batch_size,
            "multistep_gradient_accumulation_steps": (args.gradient_accumulation_steps),
            "microbatch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "effective_batch_size": 3,
            "num_workers": 0,
            "num_steps": args.num_steps,
            "stage1_epochs": args.stage1_epochs,
            "stage2_epochs": args.stage2_epochs,
            "stage1_lr": args.stage1_lr,
            "stage2_lr": args.stage2_lr,
            "stage1_noise_std": [0.02, 0.02, 0.01, 0.02],
            "stage2_noise_std": [0.002, 0.002, 0.001, 0.002],
            "stage1_teacher_forcing": True,
            "stage2_teacher_forcing": False,
            "max_trajectories": args.max_trajectories,
            "max_batches_per_epoch": args.max_batches_per_epoch,
            "checkpoint_selection": "final completed epoch; no validation split",
            "state_loss_mask": "normal nodes only",
            "inference_floor_or_limiter": "none",
            "release_schedule_deviations": [
                "single-process deterministic data iteration",
                "noise is closed under the same causal nodal boundary policy",
                (
                    "teacher-forcing step losses are backpropagated separately before "
                    "one optimizer update; they have no recurrent gradient dependency"
                ),
                (
                    "multistep effective batch three is evaluated as sequential "
                    "microbatches with the same normal-node-weighted aggregation"
                ),
            ],
        },
        "dataset": {
            "split": args.split,
            "file_name": dataset_path.name,
            "sha256": dataset_sha256,
            "available_keys": available_keys,
            "selected_keys": keys,
            "audit": dataset_audit,
        },
        "graph_policy_validation": policy_validation,
        "boundary_contract": {
            "input": (
                "same-time fixed freestream inflow, current-interior slip wall, "
                "and current-interior supersonic outflow"
            ),
            "recurrent_output": "same causal nodal policy",
            "noise": "normal-node release noise followed by causal boundary closure",
            "exact_dg_boundary_replay": False,
            "graph_to_control_volume_identity": False,
        },
        "source_sha256": local_source_provenance["file_sha256"],
        "source_provenance": local_source_provenance,
        "history": [],
    }
    _write_manifest(manifest_path, manifest)
    if args.audit_only:
        print(json.dumps({"manifest": str(manifest_path), "status": "audited"}))
        return 0

    device = _select_device(torch, args.device, args.gpu)
    policies = [
        build_torch_boundary_policy(
            torch=torch,
            device=device,
            node_type=spec.node_type,
            node_normal=spec.node_normal,
            wall_normal_coherence=spec.wall_normal_coherence,
            stencil=spec.stencil,
            mach=spec.mach,
            config=PHYSICAL_CONFIG,
        )
        for spec in specs
    ]
    dataset_type = _dataset_class(torch, api["h5py"], api["FPCBase"])
    dataset = dataset_type(
        dataset_path,
        keys,
        num_steps=args.num_steps,
        dt=args.dt,
        seed=args.seed,
    )
    teacher_forcing_loader = api["DataLoader"](
        dataset=dataset,
        batch_size=args.teacher_forcing_batch_size,
        num_workers=0,
    )
    multistep_loader = api["DataLoader"](
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=0,
    )
    transformer = api["T"].Compose(
        [
            api["T"].NormalizeScale(),
            api["T"].Cartesian(norm=False),
            api["T"].Distance(norm=False),
        ]
    )
    model = api["Simulator"](
        message_passing_num=12,
        node_input_size=6,
        edge_input_size=5,
        device=device,
        model_dir=str(args.output_dir / "checkpoint_latest.pth"),
    )
    normalizer_max_accumulations = 300000
    model._output_normalizer._max_accumulations = normalizer_max_accumulations
    model._node_normalizer._max_accumulations = normalizer_max_accumulations
    manifest["training_configuration"]["normalizer_max_accumulations"] = (
        normalizer_max_accumulations
    )
    manifest["training_configuration"]["normalizer_accumulation_rationale"] = (
        "released cap retained because teacher forcing uses the released batch of three"
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.stage1_lr)
    _assert_finite_training_state(torch, model, optimizer)
    manifest["device"] = str(device)
    manifest["model_parameters"] = int(
        sum(parameter.numel() for parameter in model.parameters())
    )
    started = time.time()
    global_epoch = 0
    stages = (
        (
            "teacher_forcing",
            args.stage1_epochs,
            args.stage1_lr,
            [0.02, 0.02, 0.01, 0.02],
            True,
            teacher_forcing_loader,
            1,
        ),
        (
            "multistep",
            args.stage2_epochs,
            args.stage2_lr,
            [0.002, 0.002, 0.001, 0.002],
            False,
            multistep_loader,
            args.gradient_accumulation_steps,
        ),
    )
    for (
        stage_name,
        epochs,
        learning_rate,
        noise,
        teacher_forcing,
        loader,
        accumulation_steps,
    ) in stages:
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = learning_rate
        if not teacher_forcing and epochs and accumulation_steps > 1:
            _require_frozen_normalizers(model)
        for stage_epoch in range(epochs):
            dataset.set_epoch(global_epoch)
            row = _train_epoch(
                api=api,
                model=model,
                loader=loader,
                optimizer=optimizer,
                transformer=transformer,
                policies=policies,
                device=device,
                noise_std=torch.tensor(noise, dtype=torch.float32),
                teacher_forcing=teacher_forcing,
                max_batches=args.max_batches_per_epoch,
                accumulation_steps=accumulation_steps,
            )
            row.update(
                {
                    "stage": stage_name,
                    "stage_epoch": stage_epoch + 1,
                    "global_epoch": global_epoch + 1,
                    "learning_rate": learning_rate,
                    "output_normalizer_accumulations": float(
                        model._output_normalizer._num_accumulations.detach().cpu()
                    ),
                    "node_normalizer_accumulations": float(
                        model._node_normalizer._num_accumulations.detach().cpu()
                    ),
                    "output_normalizer": _normalizer_diagnostics(
                        model._output_normalizer
                    ),
                    "node_normalizer": _normalizer_diagnostics(
                        model._node_normalizer
                    ),
                }
            )
            manifest["history"].append(row)
            model.save_checkpoint(str(args.output_dir / "checkpoint_latest.pth"))
            manifest["checkpoint"] = {
                "file_name": "checkpoint_latest.pth",
                "sha256": sha256_file(args.output_dir / "checkpoint_latest.pth"),
                "selection": "final completed epoch",
            }
            _write_manifest(manifest_path, manifest)
            print(json.dumps(row, sort_keys=True), flush=True)
            global_epoch += 1
    completion_status = "complete" if promotion_eligible else "complete_diagnostic"
    manifest["status"] = completion_status
    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["wall_seconds"] = time.time() - started
    _write_manifest(manifest_path, manifest)
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "checkpoint": str(args.output_dir / "checkpoint_latest.pth"),
                "status": completion_status,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
