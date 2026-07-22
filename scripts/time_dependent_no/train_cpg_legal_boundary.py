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
    recover_graph_boundary_geometry,
)
from utility.time_dependent_no.cpg_release import (  # noqa: E402
    sha256_file,
    validate_cpg_reference_source,
)

RUN_SCHEMA = "cpg_legal_boundary_training_v1"
POLICY_VALIDATION_SCHEMA = "cpg_bump_mesh_provenance_audit_v1"
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
    parser.add_argument("--dt", type=float, default=0.025)
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
    for name in (
        "dt",
        "teacher_forcing_batch_size",
        "batch_size",
        "gradient_accumulation_steps",
        "num_steps",
        "stage1_lr",
        "stage2_lr",
    ):
        if getattr(args, name) <= 0:
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
    reference = str(reference_repo.resolve())
    if reference not in sys.path:
        sys.path.insert(0, reference)
    try:
        import h5py  # type: ignore[import-not-found]
        import torch
        from torch_geometric.loader import DataLoader
        import torch_geometric.transforms as T

        from dataset.fpcMulti import FPCBase
        from modelEdgeUpd.simulator import Simulator
        from utils.lossCompute import dataLoss
        from utils.noise import get_noise
        from utils.to_undirected import make_edges_undirected
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "training requires h5py, torch, torch-geometric, and the pinned "
            "external CPGNet checkout"
        ) from exc
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


def _validate_policy_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != POLICY_VALIDATION_SCHEMA:
        raise ValueError("graph policy validation has an unsupported schema")
    if payload.get("status") != "valid":
        raise ValueError("graph policy validation did not pass")
    cases = payload.get("cases", [])
    if payload.get("case_count") != len(cases) or len(cases) != 20:
        raise ValueError("graph policy validation must cover all 20 test cases")
    if payload.get("graph_mesh_identity") != "verified_all_selected_cases":
        raise ValueError("graph policy validation lacks graph-mesh identity")
    if payload.get("boundary_geometry") != "verified_all_selected_cases":
        raise ValueError("graph policy validation lacks boundary geometry")
    normal_errors = []
    coherence_errors = []
    for record in cases:
        mesh = record.get("mesh", {})
        if mesh.get("graph_native_boundary_mapping") != (
            "verified_against_abaqus_boundary"
        ):
            raise ValueError("graph-native policy was not validated on every case")
        normal_errors.append(float(mesh["graph_native_normal_max_abs_error"]))
        coherence_errors.append(
            float(mesh["graph_native_normal_coherence_max_abs_error"])
        )
    if max(normal_errors + coherence_errors) > GRAPH_BOUNDARY_GEOMETRY_ATOL:
        raise ValueError("graph-native policy validation exceeds its tolerance")
    return {
        "file_name": path.name,
        "sha256": sha256_file(path),
        "case_count": len(cases),
        "dataset_sha256": payload.get("dataset_sha256"),
        "max_normal_abs_error": max(normal_errors),
        "max_normal_coherence_abs_error": max(coherence_errors),
        "raw_mesh_contract": "validated_on_available_test_cases",
    }


def _primitive(group: Any, frame: int) -> np.ndarray:
    return np.concatenate(
        [np.asarray(group[name][frame], dtype=np.float64) for name in STATE_KEYS],
        axis=1,
    )


def _audit_case(
    group: Any, key: str, *, max_source_hops: int
) -> tuple[PolicySpec, dict[str, Any]]:
    frames = int(group["pres"].shape[0])
    if frames <= 3:
        raise ValueError(f"trajectory {key} has too few frames")
    pos = np.asarray(group["pos"][0], dtype=np.float64)
    edges = np.asarray(group["edges"][0], dtype=np.int64)
    node_type = np.asarray(group["node_type"][0]).reshape(-1).astype(np.int64)
    mach_field = np.asarray(group["Mach"][0], dtype=np.float64).reshape(-1)
    mach = float(mach_field[0])
    if float(np.max(np.abs(mach_field - mach))) > 2.0e-10:
        raise ValueError(f"trajectory {key} Mach is not spatially constant")
    if not all(
        np.array_equal(group[name][0], group[name][-1])
        for name in ("pos", "edges", "node_type", "Mach")
    ):
        raise ValueError(f"trajectory {key} graph geometry is not static")
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

    inflow_error = 0.0
    wall_normal_velocity = 0.0
    outflow_normal_mach = float("inf")
    boundary_remap_rmse = 0.0
    stream = freestream_primitive(mach, **PHYSICAL_CONFIG)
    audit_frames = sorted({0, min(20, frames - 1), min(40, frames - 1), frames - 1})
    for frame in audit_frames:
        state = _primitive(group, frame)
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
        "boundary_reference_remap_rmse_max": boundary_remap_rmse,
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
    summary = {
        "case_count": len(records),
        "selected_keys": list(keys),
        "frames": sorted({record["frames"] for record in records}),
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
        "max_boundary_reference_remap_rmse": max(
            record["boundary_reference_remap_rmse_max"] for record in records
        ),
        "boundary_stencil_fallback_targets": 0,
        "graph_geometry_static": True,
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
            "sharp_wall_rows",
            "target_normals",
            "inflow_nodes",
            "freestream",
        )
    }
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
        fields["sharp_wall_rows"].append(policy["sharp_wall_rows"] + target_offset)
        fields["target_normals"].append(policy["target_normals"])
        fields["inflow_nodes"].append(policy["inflow_nodes"] + node_offset)
        fields["freestream"].append(
            policy["freestream"][None, :].expand(policy["inflow_nodes"].numel(), -1)
        )
        target_offset += int(policy["target_nodes"].numel())
    return {
        "num_nodes": int(ptr[-1]),
        **{
            name: torch.cat(values, dim=0).to(device) for name, values in fields.items()
        },
    }


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
) -> tuple[Any, Any, Any, int, int]:
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
    node_type = graph.x[:, 0]
    normal_mask = node_type == NORMAL_NODE
    boundary_mask = ~normal_mask
    current = graph.x[:, 1:5]
    total_loss = current.new_zeros(())
    min_density = None
    min_pressure = None
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
        step_loss = api["dataLoss"](graph, predicted, target_normalized, normal_mask)
        if not bool(torch.isfinite(step_loss)):
            raise FloatingPointError(f"nonfinite loss at rollout step {step}")
        physical_prediction = model._output_normalizer.inverse(predicted)
        density = physical_prediction[:, 0].detach().min()
        pressure = physical_prediction[:, 3].detach().min()
        min_density = (
            density if min_density is None else torch.minimum(min_density, density)
        )
        min_pressure = (
            pressure if min_pressure is None else torch.minimum(min_pressure, pressure)
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
        int(normal_mask.sum()),
        int(graph.num_graphs),
    )


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
    model.train()
    loss_sum = 0.0
    batches = 0
    microbatches = 0
    samples = 0
    min_density = None
    min_pressure = None
    started = time.time()
    iterator = iter(loader)
    while True:
        if max_batches is not None and batches >= max_batches:
            break
        group = list(islice(iterator, accumulation_steps))
        if not group:
            break
        normal_counts = [int((graph.x[:, 0] == NORMAL_NODE).sum()) for graph in group]
        total_normal_nodes = sum(normal_counts)
        if total_normal_nodes <= 0:
            raise RuntimeError("gradient-accumulation group has no normal nodes")
        optimizer.zero_grad(set_to_none=True)
        batch_loss = 0.0
        for graph, expected_normal_nodes in zip(group, normal_counts, strict=True):
            weight = expected_normal_nodes / total_normal_nodes
            (
                microbatch_loss,
                microbatch_min_density,
                microbatch_min_pressure,
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
            density = float(microbatch_min_density.cpu())
            pressure = float(microbatch_min_pressure.cpu())
            min_density = density if min_density is None else min(min_density, density)
            min_pressure = (
                pressure if min_pressure is None else min(min_pressure, pressure)
            )
        optimizer.step()
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
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    if batches == 0:
        raise RuntimeError("training epoch produced no batches")
    assert min_density is not None and min_pressure is not None
    return {
        "mean_total_loss": loss_sum / batches,
        "batches": batches,
        "microbatches": microbatches,
        "gradient_accumulation_steps": accumulation_steps,
        "samples": samples,
        "wall_seconds": time.time() - started,
        "min_raw_predicted_density": min_density,
        "min_raw_predicted_pressure": min_pressure,
        "teacher_forcing": teacher_forcing,
    }


def _write_manifest(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    reference = validate_cpg_reference_source(args.reference_repo)
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
        h5py=api["h5py"], dataset_path=dataset_path, keys=keys
    )
    if args.num_steps >= min(dataset_audit["frames"]):
        raise ValueError("--num-steps must be smaller than every trajectory")

    args.output_dir.mkdir(parents=True)
    manifest_path = args.output_dir / "run_manifest.json"
    manifest: dict[str, Any] = {
        "schema": RUN_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "audited" if args.audit_only else "running",
        "claim_scope": (
            "release-bundle legal-nodal-boundary retraining; not paper identity "
            "or exact DG boundary replay"
        ),
        "reference": reference,
        "runtime": _runtime_manifest(),
        "boundary_mode": BOUNDARY_MODE,
        "uses_future_reference_boundary": False,
        "model_configuration": {
            "message_passing_num": 12,
            "node_input_size": 6,
            "edge_input_size": 5,
            "dt": args.dt,
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
        "source_sha256": {
            "trainer": sha256_file(Path(__file__)),
            "boundary_utility": sha256_file(
                REPO_ROOT / "utility/time_dependent_no/cpg_mesh_contract.py"
            ),
        },
        "history": [],
    }
    _write_manifest(manifest_path, manifest)
    if args.audit_only:
        print(json.dumps({"manifest": str(manifest_path), "status": "audited"}))
        return 0

    device = _select_device(torch, args.device, args.gpu)
    cpu_policies = [
        build_torch_boundary_policy(
            torch=torch,
            device=torch.device("cpu"),
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
        for stage_epoch in range(epochs):
            dataset.set_epoch(global_epoch)
            row = _train_epoch(
                api=api,
                model=model,
                loader=loader,
                optimizer=optimizer,
                transformer=transformer,
                policies=cpu_policies,
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
    manifest["status"] = "complete"
    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["wall_seconds"] = time.time() - started
    _write_manifest(manifest_path, manifest)
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "checkpoint": str(args.output_dir / "checkpoint_latest.pth"),
                "status": "complete",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
