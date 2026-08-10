#!/usr/bin/env python3
"""Evaluate frozen D041 PCNO on a fixed-Fourier 90-degree bump rotation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pcno.geo_utility import compute_node_measures
from utility.time_dependent_no.euler2d_metrics import (
    shock_front_scores,
    shock_indicator,
)
from utility.time_dependent_no.pcno_artifacts import (
    atomic_write_json_with_paths as atomic_write_json,
)
from utility.time_dependent_no.pcno_artifacts import (
    git_state,
    jsonable_args,
    runtime_environment,
    sha256_file,
)
from utility.time_dependent_no.pcno_artifacts import (
    write_csv_with_paths as write_csv,
)
from utility.time_dependent_no.pcno_euler2d import (
    BOUNDARY_FIELD_NONE,
    BOUNDARY_RESIDUAL_NONE,
    NODE_TYPE_FEATURE_ONE_HOT,
    PCNOEuler2DResidual,
    PCNOEuler2DShardStore,
)
from utility.time_dependent_no.pcno_geometric_consistency import (
    BUMP_NODE_TYPE_NAMES,
    ROTATION_90_CCW,
    BumpQueryGraph,
    build_bump_query_graph,
    regenerate_differential_weights,
    regenerate_element_differential_geometry,
    relative_l2,
    rotate_euler_field,
    rotate_points,
    rotation_center,
    transported_fourier_tensors,
)
from utility.time_dependent_no.pcno_resolution_transfer import (
    conservative_admissibility_summary,
)
from utility.time_dependent_no.pcno_ripple_diagnostics import (
    conservative_to_primitive_raw,
    graph_distance_to_mask,
)
from utility.time_dependent_no.pcno_runtime import (
    build_checkpoint_model,
    load_checkpoint,
    select_device,
    synchronize,
)

SCHEMA = "pcno_d085_bump_fixed_fourier_rotation_v1"
PHASES = ("g1b",)
TYPE_ARMS = ("correct", "all_normal")
CHECKPOINT_SHA256 = "2bb5ee3ca831a6ffc498f01e309ae7ea957b2df0f411023fb3812919a6732964"
NORMALIZATION_SHA256 = (
    "717a948a1f219af9eabd6aafeacbdb8e2d00a362133e6a2b6b4408e60b71b5af"
)
SPLIT_SHA256 = "ba648aa0bf404f61f5f8ceabf2be963efda9935ca325908435c3c52bad88519f"
DATA_MANIFEST_SHA256 = (
    "5d5373fdcc682544bf330fba6d54fe65509dabe936baee7443c71a8c0c8d9fa7"
)
REPLAY_SHA256 = "550c8357ae4f3d53624a9a6ed0a4ee880d6d3d582e4f07768495e2f2d029a204"
D041_CONFIG_DIGEST = "5cd903ca0b0af43651d1b72280e55a89d63c93fff9b8b17e90fc85e5b0494418"
REPLAY_ABSOLUTE_LIMIT = 2.0e-3
REPLAY_RELATIVE_LIMIT = 1.0e-5
BATCH_INCREMENT_RELATIVE_LIMIT = 2.0e-5
BATCH_POINTWISE_SCALED_RELATIVE_LIMIT = 1.0e-4
DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG = ":4096:8"
DIFFERENTIAL_IDENTITY_ABSOLUTE_LIMIT = 1.0e-5
DIFFERENTIAL_IDENTITY_RELATIVE_LIMIT = 1.0e-7
COVARIANCE_ABSOLUTE_LIMIT = 2.0e-5
COVARIANCE_RELATIVE_LIMIT = 2.0e-6
VISUALIZATION_CASES = ("172", "58", "187")
FOURIER_POLICY = "checkpoint_native_modes_fixed_world_coordinates"
FOURIER_PHASE_ORIGIN = (0.0, 0.0)
METRIC_IDENTITY_FIELDS = (
    "case_id",
    "phase",
    "arm",
    "mode",
    "call",
    "metric",
    "frame",
    "region",
    "component",
)


@dataclass(frozen=True)
class BumpCase:
    case_id: str
    nodes: np.ndarray
    elements: np.ndarray
    node_measures: np.ndarray
    node_weights: np.ndarray
    node_rhos: np.ndarray
    edges: np.ndarray
    directed_edges: np.ndarray
    edge_gradient_weights: np.ndarray
    node_type: np.ndarray
    states: np.ndarray
    mach: float
    physical_times: np.ndarray
    gamma: float
    state_scale: np.ndarray
    residual_scale: np.ndarray


@dataclass
class GeometryArm:
    name: str
    sample: dict[str, torch.Tensor]
    fourier_tensors: tuple[torch.Tensor, ...]
    truth: np.ndarray
    weights: np.ndarray


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--normalization-json", type=Path, required=True)
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=PHASES, default="g1b")
    parser.add_argument("--case-ids", nargs="+")
    parser.add_argument("--visualization-cases", nargs="*", default=VISUALIZATION_CASES)
    parser.add_argument("--rollout-calls", type=int, default=79)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--query-radius", type=int, default=2)
    parser.add_argument("--gradient-rcond", type=float, default=1.0e-3)
    parser.add_argument("--shock-quantile", type=float, default=0.9)
    parser.add_argument("--replay-case", default="128")
    parser.add_argument("--replay-calls", type=int, default=20)
    parser.add_argument("--expected-checkpoint-sha256", default=CHECKPOINT_SHA256)
    parser.add_argument("--expected-normalization-sha256", default=NORMALIZATION_SHA256)
    parser.add_argument("--expected-split-sha256", default=SPLIT_SHA256)
    parser.add_argument("--expected-data-manifest-sha256", default=DATA_MANIFEST_SHA256)
    parser.add_argument(
        "--expected-source",
        action="append",
        default=[],
        metavar="PATH=SHA256",
    )
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(
            f"output directory already exists; choose a new path: {args.output_dir}"
        )
    expected_calls = 2 if args.smoke else 79
    if args.rollout_calls != expected_calls:
        raise ValueError(
            f"{'smoke' if args.smoke else 'scientific'} execution requires "
            f"--rollout-calls {expected_calls}"
        )
    if args.query_radius != 2:
        raise ValueError(
            "D085 retains graph-radius-2 only as an inactive legacy argument"
        )
    if args.gradient_rcond != 1.0e-3:
        raise ValueError("D085 freezes the D041 least-squares rcond at 1e-3")
    if not 0.0 < args.shock_quantile < 1.0:
        raise ValueError("--shock-quantile must lie in (0,1)")
    if args.shock_quantile != 0.9:
        raise ValueError("D085 freezes the reference shock quantile at 0.9")
    if args.replay_calls != 20 or args.replay_case != "128":
        raise ValueError("D085 freezes the D068 case-128, 20-call replay gate")
    if tuple(args.visualization_cases) != VISUALIZATION_CASES:
        raise ValueError(
            f"D085 freezes visualization cases in order as {VISUALIZATION_CASES}"
        )
    frozen_digests = {
        "checkpoint": (args.expected_checkpoint_sha256, CHECKPOINT_SHA256),
        "normalization": (args.expected_normalization_sha256, NORMALIZATION_SHA256),
        "split": (args.expected_split_sha256, SPLIT_SHA256),
        "data_manifest": (
            args.expected_data_manifest_sha256,
            DATA_MANIFEST_SHA256,
        ),
    }
    mismatched = {
        name: {"supplied": str(supplied), "frozen": frozen}
        for name, (supplied, frozen) in frozen_digests.items()
        if str(supplied).lower() != frozen
    }
    if mismatched:
        raise ValueError(f"D085 frozen digest override rejected: {mismatched}")
    if not args.expected_source:
        raise ValueError("at least one --expected-source PATH=SHA256 is required")
    return args


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _canonical_mapping_digest(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _parse_source_bindings(values: Sequence[str]) -> dict[str, str]:
    bindings: dict[str, str] = {}
    for value in values:
        relative, separator, digest = str(value).partition("=")
        if not separator or not relative or len(digest) != 64:
            raise ValueError(f"invalid source binding: {value!r}")
        if relative in bindings:
            raise ValueError(f"duplicate source binding: {relative}")
        bindings[relative] = digest.lower()
    return bindings


def _verify_source_bindings(values: Sequence[str]) -> dict[str, str]:
    bindings = _parse_source_bindings(values)
    loaded = _loaded_project_source_hashes()
    if set(bindings) != set(loaded):
        missing = sorted(set(loaded) - set(bindings))
        extra = sorted(set(bindings) - set(loaded))
        raise ValueError(
            f"complete loaded-source binding required: missing={missing}, extra={extra}"
        )
    actual = {}
    for relative, expected in bindings.items():
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        actual[relative] = sha256_file(path)
        if actual[relative] != expected:
            raise ValueError(f"source hash mismatch: {relative}")
        if actual[relative] != loaded[relative]:
            raise ValueError(
                f"loaded source changed during provenance gate: {relative}"
            )
    return actual


def _loaded_project_source_hashes() -> dict[str, str]:
    paths = set()
    for module in tuple(sys.modules.values()):
        raw = getattr(module, "__file__", None)
        if raw is None:
            continue
        path = Path(raw)
        if not path.is_absolute() or path.suffix != ".py":
            continue
        try:
            relative = path.resolve().relative_to(ROOT.resolve())
        except (OSError, ValueError):
            continue
        if path.is_file():
            paths.add(relative.as_posix())
    return {relative: sha256_file(ROOT / relative) for relative in sorted(paths)}


def _verify_provenance(
    args: argparse.Namespace,
    checkpoint: Mapping[str, Any],
    store: PCNOEuler2DShardStore,
) -> dict[str, Any]:
    expected = {
        "checkpoint": args.expected_checkpoint_sha256.lower(),
        "normalization": args.expected_normalization_sha256.lower(),
        "split": args.expected_split_sha256.lower(),
    }
    actual = {
        "checkpoint": sha256_file(args.checkpoint),
        "normalization": sha256_file(args.normalization_json),
        "split": sha256_file(args.split_json),
    }
    if actual != expected:
        raise ValueError(f"frozen artifact digest mismatch: {actual} != {expected}")
    if store.manifest_digest != args.expected_data_manifest_sha256:
        raise ValueError("active shard manifest differs from D041")
    if checkpoint.get("data_manifest_digest") != args.expected_data_manifest_sha256:
        raise ValueError("checkpoint data-manifest digest differs from D041")
    if checkpoint.get("boundary_mode") != "model_all_nodes":
        raise ValueError("D085 freezes the model_all_nodes physical policy")
    if checkpoint.get("raw_recurrence") is not True:
        raise ValueError("D085 requires raw recurrence")
    if int(checkpoint.get("step_stride", -1)) != 1:
        raise ValueError("D085 requires the D041 stride-1 recurrence")
    normalization = json.loads(args.normalization_json.read_text(encoding="utf-8"))
    if normalization != checkpoint["normalization"]:
        raise ValueError("normalization file and checkpoint mapping differ")
    split = json.loads(args.split_json.read_text(encoding="utf-8"))
    if split.get("data_manifest_digest") != args.expected_data_manifest_sha256:
        raise ValueError("split and shard manifest bindings differ")
    return {
        "artifact_sha256": actual,
        "data_manifest_sha256": store.manifest_digest,
        "normalization_mapping_digest": _canonical_mapping_digest(normalization),
        "source_sha256": _verify_source_bindings(args.expected_source),
        "split": split,
    }


def _load_case(
    store: PCNOEuler2DShardStore,
    checkpoint: Mapping[str, Any],
    case_id: str,
    *,
    rollout_calls: int,
) -> BumpCase:
    states = np.asarray(store.states(case_id), dtype=np.float64)
    if states.shape[0] != 80 or rollout_calls >= states.shape[0]:
        raise ValueError(f"case {case_id} does not support the declared horizon")
    node_type = np.asarray(store.array(case_id, "node_type"), dtype=np.int64).reshape(
        -1
    )
    if {int(value) for value in np.unique(node_type)} - set(BUMP_NODE_TYPE_NAMES):
        raise ValueError(f"case {case_id} contains unsupported bump node types")
    dt = float(store.manifest["dt"])
    normalization = checkpoint["normalization"]
    return BumpCase(
        case_id=str(case_id),
        nodes=np.array(store.array(case_id, "nodes"), dtype=np.float64, copy=True),
        elements=np.array(store.array(case_id, "elements"), dtype=np.int64, copy=True),
        node_measures=np.array(
            store.array(case_id, "node_measures"), dtype=np.float64, copy=True
        ),
        node_weights=np.array(
            store.array(case_id, "node_weights"), dtype=np.float64, copy=True
        ),
        node_rhos=np.array(
            store.array(case_id, "node_rhos"), dtype=np.float64, copy=True
        ),
        edges=np.array(store.array(case_id, "edges"), dtype=np.int64, copy=True),
        directed_edges=np.array(
            store.array(case_id, "directed_edges"), dtype=np.int64, copy=True
        ),
        edge_gradient_weights=np.array(
            store.array(case_id, "edge_gradient_weights"),
            dtype=np.float64,
            copy=True,
        ),
        node_type=node_type,
        states=np.array(states[: rollout_calls + 1], copy=True),
        mach=float(store.entry(case_id)["mach"]),
        physical_times=dt * np.arange(rollout_calls + 1, dtype=np.float64),
        gamma=float(normalization["gamma"]),
        state_scale=np.asarray(normalization["state_scale"], dtype=np.float64),
        residual_scale=np.asarray(normalization["residual_scale"], dtype=np.float64),
    )


def _sample(
    *,
    nodes: np.ndarray,
    node_measures: np.ndarray,
    node_weights: np.ndarray,
    node_rhos: np.ndarray,
    directed_edges: np.ndarray,
    edge_gradient_weights: np.ndarray,
    node_type: np.ndarray,
    mach: float,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    count = int(np.asarray(nodes).shape[0])

    def batched(value: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
        return torch.as_tensor(
            np.array(value, copy=True), dtype=dtype, device=device
        ).unsqueeze(0)

    return {
        "node_mask": torch.ones((1, count, 1), dtype=torch.float32, device=device),
        "nodes": batched(nodes, torch.float32),
        "node_measures": batched(node_measures, torch.float32),
        "node_weights": batched(node_weights, torch.float32),
        "node_rhos": batched(node_rhos, torch.float32),
        "directed_edges": batched(directed_edges, torch.int64),
        "edge_gradient_weights": batched(edge_gradient_weights, torch.float32),
        "node_type": batched(np.asarray(node_type).reshape(-1), torch.int64),
        "mach": torch.tensor([float(mach)], dtype=torch.float32, device=device),
    }


def _expanded(tensor: torch.Tensor, batch_size: int) -> torch.Tensor:
    if tensor.shape[0] != 1:
        raise ValueError("D085 geometry tensors must have singleton batch axes")
    return tensor.expand(batch_size, *tensor.shape[1:])


@torch.inference_mode()
def _predict_batched(
    model: PCNOEuler2DResidual,
    geometry: GeometryArm,
    currents: Sequence[np.ndarray],
    type_arms: Sequence[str],
    *,
    device: torch.device,
) -> dict[str, np.ndarray]:
    if len(currents) != len(type_arms) or not currents:
        raise ValueError("currents and type arms must be nonempty and aligned")
    if any(name not in TYPE_ARMS for name in type_arms):
        raise ValueError("unsupported type arm")
    batch_size = len(currents)
    current = torch.as_tensor(
        np.asarray(currents, dtype=np.float32), dtype=torch.float32, device=device
    )
    node_type = _expanded(geometry.sample["node_type"], batch_size).clone()
    for index, arm in enumerate(type_arms):
        if arm == "all_normal":
            node_type[index].zero_()
    model_input = model.normalized_input(
        current,
        nodes=_expanded(geometry.sample["nodes"], batch_size),
        node_rhos=_expanded(geometry.sample["node_rhos"], batch_size),
        node_type=node_type,
        mach=geometry.sample["mach"].expand(batch_size),
    )
    fourier = tuple(_expanded(value, batch_size) for value in geometry.fourier_tensors)
    normalized_residual = model.backbone(
        model_input,
        (
            _expanded(geometry.sample["node_mask"], batch_size),
            _expanded(geometry.sample["nodes"], batch_size),
            _expanded(geometry.sample["node_weights"], batch_size),
            _expanded(geometry.sample["directed_edges"], batch_size),
            _expanded(geometry.sample["edge_gradient_weights"], batch_size),
        ),
        fourier_tensors=fourier,
    )
    prediction = (current + normalized_residual * model.residual_scale) * _expanded(
        geometry.sample["node_mask"], batch_size
    )
    synchronize(device)
    values = prediction.detach().float().cpu().numpy().astype(np.float64)
    return {name: values[index] for index, name in enumerate(type_arms)}


@torch.inference_mode()
def _predict(
    model: PCNOEuler2DResidual,
    geometry: GeometryArm,
    currents: Sequence[np.ndarray],
    type_arms: Sequence[str],
    *,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Evaluate every scientific arm with the frozen batch-one replay path."""

    if len(currents) != len(type_arms) or not currents:
        raise ValueError("currents and type arms must be nonempty and aligned")
    if len(set(type_arms)) != len(type_arms):
        raise ValueError("scientific inference requires unique type-arm labels")
    predictions = {}
    for current, type_arm in zip(currents, type_arms):
        predictions.update(
            _predict_batched(
                model,
                geometry,
                (current,),
                (type_arm,),
                device=device,
            )
        )
    return predictions


def _difference_metrics(value: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    difference = np.asarray(value, dtype=np.float64) - np.asarray(
        reference, dtype=np.float64
    )
    return {
        "max_abs": float(np.max(np.abs(difference))),
        "relative_l2": relative_l2(difference, reference),
    }


def _tensor_tuple_difference(
    value: Sequence[torch.Tensor], reference: Sequence[torch.Tensor]
) -> dict[str, float]:
    if len(value) != len(reference) or not value:
        raise ValueError("tensor tuples must be nonempty and aligned")
    numerator = sum(
        torch.square(left - right).sum() for left, right in zip(value, reference)
    )
    denominator = sum(torch.square(right).sum() for right in reference)
    return {
        "max_abs": max(
            float((left - right).abs().max().detach().cpu())
            for left, right in zip(value, reference)
        ),
        "relative_l2": float(
            torch.sqrt(numerator / torch.clamp_min(denominator, 1.0e-30)).detach().cpu()
        ),
    }


def _require_close(
    name: str,
    metrics: Mapping[str, float],
    *,
    absolute_limit: float,
    relative_limit: float,
) -> None:
    finite = math.isfinite(float(metrics["max_abs"])) and math.isfinite(
        float(metrics["relative_l2"])
    )
    if (
        not finite
        or float(metrics["max_abs"]) > absolute_limit
        or float(metrics["relative_l2"]) > relative_limit
    ):
        raise RuntimeError(f"{name} gate failed: {dict(metrics)}")


def _require_batch_consistency(name: str, metrics: Mapping[str, float]) -> None:
    required = (
        "max_abs",
        "relative_l2",
        "pointwise_component_scaled_relative_max",
    )
    if (
        not all(math.isfinite(float(metrics[key])) for key in required)
        or float(metrics["relative_l2"]) > BATCH_INCREMENT_RELATIVE_LIMIT
        or float(metrics["pointwise_component_scaled_relative_max"])
        > BATCH_POINTWISE_SCALED_RELATIVE_LIMIT
    ):
        raise RuntimeError(f"{name} gate failed: {dict(metrics)}")


@contextmanager
def _deterministic_identity_gate(device: torch.device):
    if (
        device.type == "cuda"
        and os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        != DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG
    ):
        raise RuntimeError(
            "deterministic identity gates require "
            f"CUBLAS_WORKSPACE_CONFIG={DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG}"
        )
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        if not torch.are_deterministic_algorithms_enabled():
            raise RuntimeError("deterministic identity gate did not activate")
        yield
    finally:
        torch.use_deterministic_algorithms(previous)


def _pressure(state: np.ndarray, gamma: float) -> np.ndarray:
    values = np.asarray(state, dtype=np.float64)
    rho = values[..., 0]
    kinetic = 0.5 * np.square(values[..., 1:3]).sum(axis=-1) / rho
    return (gamma - 1.0) * (values[..., 3] - kinetic)


def _build_geometry_arms(
    case: BumpCase,
    model: PCNOEuler2DResidual,
    args: argparse.Namespace,
    *,
    device: torch.device,
) -> tuple[dict[str, GeometryArm], BumpQueryGraph | None, dict[str, Any]]:
    native_sample = _sample(
        nodes=case.nodes,
        node_measures=case.node_measures,
        node_weights=case.node_weights,
        node_rhos=case.node_rhos,
        directed_edges=case.directed_edges,
        edge_gradient_weights=case.edge_gradient_weights,
        node_type=case.node_type,
        mach=case.mach,
        device=device,
    )
    native_fourier = model.backbone.prepare_fourier_tensors(
        native_sample["nodes"], native_sample["node_weights"]
    )
    checkpoint_modes_sha256 = _array_sha256(model.backbone.modes.detach().cpu().numpy())
    arms = {
        "native": GeometryArm(
            name="native",
            sample=native_sample,
            fourier_tensors=native_fourier,
            truth=case.states,
            weights=case.node_weights.sum(axis=-1),
        )
    }
    audit: dict[str, Any] = {
        "case_id": case.case_id,
        "bump_node_type_semantics": {
            str(key): value for key, value in BUMP_NODE_TYPE_NAMES.items()
        },
        "node_type_counts": {
            str(code): int(np.count_nonzero(case.node_type == code))
            for code in range(4)
        },
        "native_node_count": int(case.nodes.shape[0]),
        "native_directed_edge_count": int(case.directed_edges.shape[0]),
        "native_proxy_mass": float(case.node_measures.sum()),
    }

    native_regenerated, native_rank = regenerate_differential_weights(
        case.nodes,
        case.directed_edges,
        rcond=args.gradient_rcond,
    )
    identity_metrics = _difference_metrics(
        native_regenerated, case.edge_gradient_weights
    )
    _require_close(
        "native differential reconstruction",
        identity_metrics,
        absolute_limit=DIFFERENTIAL_IDENTITY_ABSOLUTE_LIMIT,
        relative_limit=DIFFERENTIAL_IDENTITY_RELATIVE_LIMIT,
    )
    audit["native_differential_reconstruction"] = {
        **identity_metrics,
        **native_rank,
    }
    native_element_edges, native_element_weights = (
        regenerate_element_differential_geometry(
            case.nodes, case.elements, rcond=args.gradient_rcond
        )
    )
    if not np.array_equal(native_element_edges, case.directed_edges):
        raise RuntimeError(
            "retained elements do not exactly regenerate native connectivity/order"
        )
    element_identity = _difference_metrics(
        native_element_weights, case.edge_gradient_weights
    )
    _require_close(
        "native element differential reconstruction",
        element_identity,
        absolute_limit=DIFFERENTIAL_IDENTITY_ABSOLUTE_LIMIT,
        relative_limit=DIFFERENTIAL_IDENTITY_RELATIVE_LIMIT,
    )
    audit["native_element_graph_reconstruction"] = {
        "connectivity_and_order_exact": True,
        "directed_edges_sha256": _array_sha256(native_element_edges),
        **element_identity,
    }

    query: BumpQueryGraph | None = None
    if args.phase in {"g1a", "both"}:
        query = build_bump_query_graph(
            case.nodes,
            case.directed_edges,
            case.node_measures,
            case.node_type,
            radius=args.query_radius,
            gradient_rcond=args.gradient_rcond,
        )
        query_truth = query.restrict(case.states)
        constant_floor = float(
            np.max(np.abs(query.restrict(np.ones_like(case.states[:1])) - 1.0))
        )
        tag_mismatch = int(
            np.count_nonzero(query.node_type[query.fine_to_coarse] != case.node_type)
        )
        if constant_floor > 1.0e-12 or tag_mismatch:
            raise RuntimeError("query restriction or type-stratification gate failed")
        native_integrals = np.einsum("n,tnc->tc", case.node_measures[:, 0], case.states)
        query_integrals = np.einsum("n,tnc->tc", query.node_measures[:, 0], query_truth)
        integral_preservation = _difference_metrics(query_integrals, native_integrals)
        _require_close(
            "all-frame query proxy integral",
            integral_preservation,
            absolute_limit=1.0e-10,
            relative_limit=1.0e-12,
        )
        _, query_rank = regenerate_differential_weights(
            query.nodes,
            query.directed_edges,
            rcond=args.gradient_rcond,
        )
        cluster_centroids = query.restrict(case.nodes)
        anchor_offsets = np.linalg.norm(query.nodes - cluster_centroids, axis=-1)
        cluster_diameters = []
        for cluster in range(query.coarse_node_count):
            members = case.nodes[query.fine_to_coarse == cluster]
            pairwise = members[:, None, :] - members[None, :, :]
            cluster_diameters.append(float(np.linalg.norm(pairwise, axis=-1).max()))
        cluster_diameters_array = np.asarray(cluster_diameters)
        query_sample = _sample(
            nodes=query.nodes,
            node_measures=query.node_measures,
            node_weights=query.node_weights,
            node_rhos=query.node_rhos,
            directed_edges=query.directed_edges,
            edge_gradient_weights=query.edge_gradient_weights,
            node_type=query.node_type,
            mach=case.mach,
            device=device,
        )
        arms["query"] = GeometryArm(
            name="query",
            sample=query_sample,
            fourier_tensors=model.backbone.prepare_fourier_tensors(
                query_sample["nodes"], query_sample["node_weights"]
            ),
            truth=query_truth,
            weights=query.node_weights.sum(axis=-1),
        )
        audit["query_graph"] = {
            "interpretation": "proxy_mass_query_graph_not_pde_resolution_transfer",
            "radius": query.radius,
            "coarse_node_count": query.coarse_node_count,
            "coarse_to_native_node_ratio": (
                query.coarse_node_count / query.fine_node_count
            ),
            "coarse_directed_edge_count": int(query.directed_edges.shape[0]),
            "proxy_mass_difference": float(
                query.node_measures.sum() - case.node_measures.sum()
            ),
            "constant_restriction_max_abs": constant_floor,
            "type_mismatch_count": tag_mismatch,
            "restriction_sha256": _array_sha256(query.fine_to_coarse),
            "anchor_sha256": _array_sha256(query.anchor_indices),
            "all_frame_proxy_integral_preservation": integral_preservation,
            "differential_rank": query_rank,
            "anchor_to_proxy_centroid_rms": float(
                np.sqrt(np.mean(np.square(anchor_offsets)))
            ),
            "anchor_to_proxy_centroid_max": float(anchor_offsets.max()),
            "cluster_physical_diameter_median": float(
                np.median(cluster_diameters_array)
            ),
            "cluster_physical_diameter_p95": float(
                np.quantile(cluster_diameters_array, 0.95)
            ),
            "cluster_physical_diameter_max": float(cluster_diameters_array.max()),
        }

    if args.phase in {"g1b", "both"}:
        center = rotation_center(case.nodes)
        rotated_nodes = rotate_points(case.nodes, center)
        rotated_edges, rotated_weights = regenerate_element_differential_geometry(
            rotated_nodes, case.elements, rcond=args.gradient_rcond
        )
        if not np.array_equal(rotated_edges, case.directed_edges):
            raise RuntimeError("rigid rotation changed regenerated graph/order")
        _, rotated_rank = regenerate_differential_weights(
            rotated_nodes, rotated_edges, rcond=args.gradient_rcond
        )
        differential_covariance = _difference_metrics(
            rotated_weights, native_element_weights @ ROTATION_90_CCW.T
        )
        _require_close(
            "rotated differential covariance",
            differential_covariance,
            absolute_limit=1.0e-10,
            relative_limit=1.0e-10,
        )
        rotated_truth = rotate_euler_field(case.states)
        manually_rotated_truth = np.array(case.states, copy=True)
        manually_rotated_truth[..., 1] = -case.states[..., 2]
        manually_rotated_truth[..., 2] = case.states[..., 1]
        raw_transform_metrics = _difference_metrics(
            rotated_truth, manually_rotated_truth
        )
        _require_close(
            "explicit one-pass raw transform",
            raw_transform_metrics,
            absolute_limit=0.0,
            relative_limit=0.0,
        )
        twice_rotated = rotate_euler_field(rotated_truth)
        double_transform_negative_control = _difference_metrics(
            twice_rotated, rotated_truth
        )
        if double_transform_negative_control["relative_l2"] <= 1.0e-3:
            raise RuntimeError(
                "double-transform negative control is not discriminating"
            )
        point_roundtrip = _difference_metrics(
            rotate_points(rotated_nodes, center, inverse=True), case.nodes
        )
        state_roundtrip = _difference_metrics(
            rotate_euler_field(rotated_truth, inverse=True), case.states
        )
        residual = np.diff(case.states, axis=0)
        directional_residual_covariance = _difference_metrics(
            np.diff(rotated_truth, axis=0), rotate_euler_field(residual)
        )
        _require_close(
            "directional residual covariance",
            directional_residual_covariance,
            absolute_limit=1.0e-12,
            relative_limit=1.0e-12,
        )
        residual_roundtrip = _difference_metrics(
            rotate_euler_field(rotate_euler_field(residual), inverse=True), residual
        )
        for name, metrics in (
            ("point round trip", point_roundtrip),
            ("state round trip", state_roundtrip),
            ("residual round trip", residual_roundtrip),
        ):
            _require_close(
                name,
                metrics,
                absolute_limit=1.0e-12,
                relative_limit=1.0e-12,
            )

        original_measure = compute_node_measures(case.nodes, case.elements)
        rotated_measure = compute_node_measures(rotated_nodes, case.elements)
        measure_covariance = _difference_metrics(rotated_measure, original_measure)
        _require_close(
            "rotated reconstructed measure",
            measure_covariance,
            absolute_limit=1.0e-12,
            relative_limit=1.0e-10,
        )
        stored_measure_identity = _difference_metrics(
            original_measure, case.node_measures
        )
        _require_close(
            "stored reconstructed measure",
            stored_measure_identity,
            absolute_limit=1.0e-7,
            relative_limit=1.0e-6,
        )
        edge = rotated_edges
        native_lengths = np.linalg.norm(
            case.nodes[edge[:, 1]] - case.nodes[edge[:, 0]], axis=-1
        )
        rotated_lengths = np.linalg.norm(
            rotated_nodes[edge[:, 1]] - rotated_nodes[edge[:, 0]], axis=-1
        )
        metric_covariance = _difference_metrics(rotated_lengths, native_lengths)
        _require_close(
            "rotated edge metric",
            metric_covariance,
            absolute_limit=1.0e-12,
            relative_limit=1.0e-12,
        )
        pressure_covariance = _difference_metrics(
            _pressure(rotated_truth, case.gamma), _pressure(case.states, case.gamma)
        )
        _require_close(
            "rotated pressure",
            pressure_covariance,
            absolute_limit=1.0e-12,
            relative_limit=1.0e-12,
        )

        rotated_sample = _sample(
            nodes=rotated_nodes,
            node_measures=case.node_measures,
            node_weights=case.node_weights,
            node_rhos=case.node_rhos,
            directed_edges=rotated_edges,
            edge_gradient_weights=rotated_weights,
            node_type=case.node_type,
            mach=case.mach,
            device=device,
        )
        # D085 intentionally keeps the checkpoint's native wavevectors fixed in
        # world coordinates.  The rotated coordinates enter the ordinary PCNO
        # Fourier preparation directly; no Qk or transformed phase origin enters
        # scientific inference.
        rotated_fourier = model.backbone.prepare_fourier_tensors(
            rotated_sample["nodes"], rotated_sample["node_weights"]
        )
        modes_after_preparation = _array_sha256(
            model.backbone.modes.detach().cpu().numpy()
        )
        if modes_after_preparation != checkpoint_modes_sha256:
            raise RuntimeError(
                "fixed checkpoint Fourier modes changed during preparation"
            )

        # This transported basis is a diagnostic-only negative control for the
        # superseded D083 contract.  It is never attached to a GeometryArm.
        transported_control = transported_fourier_tensors(
            rotated_sample["nodes"],
            rotated_sample["node_weights"],
            model.backbone.modes,
            center=center,
        )
        transported_control_vs_native = _tensor_tuple_difference(
            transported_control, native_fourier
        )
        _require_close(
            "diagnostic-only transported Fourier control",
            transported_control_vs_native,
            absolute_limit=COVARIANCE_ABSOLUTE_LIMIT,
            relative_limit=COVARIANCE_RELATIVE_LIMIT,
        )
        fixed_vs_native = _tensor_tuple_difference(rotated_fourier, native_fourier)
        fixed_vs_transported = _tensor_tuple_difference(
            rotated_fourier, transported_control
        )
        if not all(
            math.isfinite(float(value)) for value in fixed_vs_transported.values()
        ) or (
            fixed_vs_transported["max_abs"] <= COVARIANCE_ABSOLUTE_LIMIT
            and fixed_vs_transported["relative_l2"] <= COVARIANCE_RELATIVE_LIMIT
        ):
            raise RuntimeError(
                "fixed-coordinate Fourier basis is not distinct from the "
                f"transported-mode negative control: {fixed_vs_transported}"
            )
        rotated_current = torch.as_tensor(
            rotated_truth[0].astype(np.float32), dtype=torch.float32, device=device
        ).unsqueeze(0)
        normalized = model.normalized_input(
            rotated_current,
            nodes=rotated_sample["nodes"],
            node_rhos=rotated_sample["node_rhos"],
            node_type=rotated_sample["node_type"],
            mach=rotated_sample["mach"],
        )
        expected_normalized_state = (
            rotated_current - model.state_mean
        ) / model.state_scale
        normalization_max_abs = float(
            (normalized[..., 3:7] - expected_normalized_state)
            .abs()
            .max()
            .detach()
            .cpu()
        )
        if normalization_max_abs != 0.0:
            raise RuntimeError("raw-state-before-normalization gate failed")
        arms["rotated"] = GeometryArm(
            name="rotated",
            sample=rotated_sample,
            fourier_tensors=rotated_fourier,
            truth=rotated_truth,
            weights=case.node_weights.sum(axis=-1),
        )
        audit["rotation"] = {
            "matrix": ROTATION_90_CCW.tolist(),
            "center": center.tolist(),
            "inference_fourier_policy": FOURIER_POLICY,
            "inference_phase_origin": list(FOURIER_PHASE_ORIGIN),
            "domain_lengths": [float(value) for value in model.domain_lengths],
            "checkpoint_modes_sha256_before": checkpoint_modes_sha256,
            "checkpoint_modes_sha256_after": modes_after_preparation,
            "point_roundtrip": point_roundtrip,
            "state_roundtrip": state_roundtrip,
            "residual_roundtrip": residual_roundtrip,
            "directional_residual_covariance": directional_residual_covariance,
            "one_pass_raw_transform": raw_transform_metrics,
            "double_transform_negative_control": double_transform_negative_control,
            "pressure_covariance": pressure_covariance,
            "proxy_measure_covariance": measure_covariance,
            "stored_proxy_measure_reconstruction": stored_measure_identity,
            "edge_metric_covariance": metric_covariance,
            "differential_covariance": {
                **differential_covariance,
                **rotated_rank,
            },
            "fixed_basis_vs_native": fixed_vs_native,
            "fixed_basis_vs_transported_negative_control": fixed_vs_transported,
            "transported_negative_control_vs_native": (transported_control_vs_native),
            "transported_negative_control_used_for_inference": False,
            "raw_transform_before_normalization_max_abs": normalization_max_abs,
            "node_type_counts_preserved": bool(
                np.array_equal(
                    np.bincount(case.node_type, minlength=4),
                    np.bincount(np.array(case.node_type, copy=True), minlength=4),
                )
            ),
            "node_type_sha256_before": _array_sha256(case.node_type),
            "node_type_sha256_after": _array_sha256(
                np.array(case.node_type, copy=True)
            ),
            "connectivity_preserved": True,
            "regenerated_directed_edges_sha256": _array_sha256(rotated_edges),
            "proxy_mass_difference": float(
                rotated_measure.sum() - original_measure.sum()
            ),
            "normal_or_vector_boundary_model_inputs": [],
            "freestream_direction_transport": "raw_momentum_rotation",
            "raw_state_transform_count": 1,
        }
    return arms, query, audit


def _weighted_norm(
    value: np.ndarray,
    *,
    weights: np.ndarray,
    scale: np.ndarray,
    mask: np.ndarray | None = None,
) -> float:
    array = np.asarray(value, dtype=np.float64)
    mass = np.asarray(weights, dtype=np.float64).reshape(-1)
    component_scale = np.asarray(scale, dtype=np.float64).reshape(-1)
    if array.ndim != 2 or array.shape[0] != mass.size:
        raise ValueError("weighted norm expects [node,component]")
    if component_scale.shape != (array.shape[1],) or bool((component_scale <= 0).any()):
        raise ValueError("component scale does not match the field")
    selected = np.ones(mass.size, dtype=bool) if mask is None else np.asarray(mask)
    if selected.shape != (mass.size,) or selected.dtype != np.bool_:
        raise ValueError("metric mask must be a boolean node vector")
    selected_mass = float(mass[selected].sum())
    if selected_mass <= 0.0:
        return math.nan
    squared = np.square(array[selected] / component_scale[None, :]).sum(axis=-1)
    return float(np.sqrt(np.dot(mass[selected], squared) / selected_mass))


def _increment_identity_metrics(
    value: np.ndarray,
    reference: np.ndarray,
    current: np.ndarray,
    *,
    weights: np.ndarray,
    residual_scale: np.ndarray,
) -> dict[str, float]:
    difference = np.asarray(value, dtype=np.float64) - np.asarray(
        reference, dtype=np.float64
    )
    reference_increment = np.asarray(reference, dtype=np.float64) - np.asarray(
        current, dtype=np.float64
    )
    scaled_numerator = _weighted_norm(
        difference,
        weights=weights,
        scale=residual_scale,
    )
    scaled_denominator = _weighted_norm(
        reference_increment,
        weights=weights,
        scale=residual_scale,
    )
    component_scale = np.asarray(residual_scale, dtype=np.float64).reshape(-1)
    pointwise_scaled_max = float(np.max(np.abs(difference / component_scale[None, :])))
    reference_pointwise_scaled_max = float(
        np.max(np.abs(reference_increment / component_scale[None, :]))
    )
    return {
        "max_abs": float(np.max(np.abs(difference))),
        "relative_l2": float(scaled_numerator / max(scaled_denominator, 1.0e-30)),
        "component_scaled_proxy_weighted_rms": scaled_numerator,
        "reference_increment_component_scaled_proxy_weighted_rms": (scaled_denominator),
        "pointwise_component_scaled_max": pointwise_scaled_max,
        "reference_increment_pointwise_component_scaled_max": (
            reference_pointwise_scaled_max
        ),
        "pointwise_component_scaled_relative_max": float(
            pointwise_scaled_max / max(reference_pointwise_scaled_max, 1.0e-30)
        ),
        "raw_physical_proxy_weighted_rms": _weighted_norm(
            difference,
            weights=weights,
            scale=np.ones(difference.shape[1]),
        ),
        "state_relative_l2": relative_l2(difference, reference),
    }


def _weighted_inner(
    left: np.ndarray,
    right: np.ndarray,
    *,
    weights: np.ndarray,
    scale: np.ndarray,
) -> float:
    lhs = np.asarray(left, dtype=np.float64) / np.asarray(scale, dtype=np.float64)
    rhs = np.asarray(right, dtype=np.float64) / np.asarray(scale, dtype=np.float64)
    mass = np.asarray(weights, dtype=np.float64).reshape(-1)
    return float(np.einsum("n,nc,nc->", mass, lhs, rhs) / mass.sum())


def _metric_row(
    *,
    case_id: str,
    phase: str,
    arm: str,
    mode: str,
    call: int,
    physical_time: float,
    metric: str,
    numerator: float,
    denominator: float | None = None,
    frame: str,
    region: str = "all",
    component: str = "all",
    raw_physical_numerator: float | None = None,
    raw_physical_denominator: float | None = None,
    numerator_contract: str = "component_scaled_proxy_weighted_rms",
) -> dict[str, Any]:
    value = numerator
    if denominator is not None:
        value = None if denominator <= 1.0e-30 else numerator / denominator
    return {
        "case_id": case_id,
        "phase": phase,
        "arm": arm,
        "mode": mode,
        "call": int(call),
        "physical_time": float(physical_time),
        "metric": metric,
        "frame": frame,
        "region": region,
        "component": component,
        "numerator": float(numerator),
        "denominator": None if denominator is None else float(denominator),
        "numerator_contract": numerator_contract,
        "raw_physical_numerator": (
            None if raw_physical_numerator is None else float(raw_physical_numerator)
        ),
        "raw_physical_denominator": (
            None
            if raw_physical_denominator is None
            else float(raw_physical_denominator)
        ),
        "value": value,
    }


def _assert_unique_metric_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    seen: dict[tuple[Any, ...], int] = {}
    for index, row in enumerate(rows):
        key = tuple(row[field] for field in METRIC_IDENTITY_FIELDS)
        previous = seen.get(key)
        if previous is not None:
            raise ValueError(
                "duplicate D085 metric identity at rows "
                f"{previous} and {index}: {key}"
            )
        seen[key] = index


def _region_masks(
    case: BumpCase,
    reference: np.ndarray,
    boundary_distance: np.ndarray,
    *,
    shock_quantile: float,
) -> dict[str, np.ndarray]:
    primitive = conservative_to_primitive_raw(reference, gamma=case.gamma)
    seed = shock_indicator(primitive, case.directed_edges, quantile=shock_quantile)
    shock_distance = graph_distance_to_mask(case.nodes, case.directed_edges, seed)
    boundary_005 = boundary_distance <= 0.05
    boundary_010 = boundary_distance <= 0.10
    shock = (shock_distance <= 0.05) & ~boundary_010
    return {
        "all": np.ones(case.nodes.shape[0], dtype=bool),
        "boundary_nodes": case.node_type != 0,
        "boundary_distance_le_0.05": boundary_005,
        "boundary_distance_le_0.10": boundary_010,
        "shock": shock,
        "smooth": ~(boundary_010 | shock),
    }


def _record_native_state_metrics(
    rows: list[dict[str, Any]],
    case: BumpCase,
    prediction: np.ndarray,
    reference: np.ndarray,
    regions: Mapping[str, np.ndarray],
    *,
    phase: str,
    arm: str,
    call: int,
) -> None:
    weights = case.node_weights.sum(axis=-1)
    error = np.asarray(prediction) - np.asarray(reference)
    for region, mask in regions.items():
        numerator = _weighted_norm(
            error, weights=weights, scale=case.state_scale, mask=mask
        )
        denominator = _weighted_norm(
            reference, weights=weights, scale=case.state_scale, mask=mask
        )
        rows.append(
            _metric_row(
                case_id=case.case_id,
                phase=phase,
                arm=arm,
                mode="free_rollout",
                call=call,
                physical_time=case.physical_times[call],
                metric="state_relative_l2",
                numerator=numerator,
                denominator=denominator,
                frame="native",
                region=region,
                raw_physical_numerator=_weighted_norm(
                    error,
                    weights=weights,
                    scale=np.ones(4),
                    mask=mask,
                ),
                raw_physical_denominator=_weighted_norm(
                    reference,
                    weights=weights,
                    scale=np.ones(4),
                    mask=mask,
                ),
            )
        )
    for component in range(4):
        numerator = _weighted_norm(
            error[:, component : component + 1],
            weights=weights,
            scale=case.state_scale[component : component + 1],
        )
        denominator = _weighted_norm(
            reference[:, component : component + 1],
            weights=weights,
            scale=case.state_scale[component : component + 1],
        )
        rows.append(
            _metric_row(
                case_id=case.case_id,
                phase=phase,
                arm=arm,
                mode="free_rollout",
                call=call,
                physical_time=case.physical_times[call],
                metric="state_relative_l2",
                numerator=numerator,
                denominator=denominator,
                frame="native",
                component=str(component),
                raw_physical_numerator=_weighted_norm(
                    error[:, component : component + 1],
                    weights=weights,
                    scale=np.ones(1),
                ),
                raw_physical_denominator=_weighted_norm(
                    reference[:, component : component + 1],
                    weights=weights,
                    scale=np.ones(1),
                ),
            )
        )


def _record_residual_metric(
    rows: list[dict[str, Any]],
    case: BumpCase,
    prediction_increment: np.ndarray,
    truth_increment: np.ndarray,
    *,
    weights: np.ndarray,
    phase: str,
    arm: str,
    mode: str,
    call: int,
    frame: str,
) -> None:
    numerator = _weighted_norm(
        prediction_increment - truth_increment,
        weights=weights,
        scale=case.residual_scale,
    )
    denominator = _weighted_norm(
        truth_increment,
        weights=weights,
        scale=case.residual_scale,
    )
    rows.append(
        _metric_row(
            case_id=case.case_id,
            phase=phase,
            arm=arm,
            mode=mode,
            call=call,
            physical_time=case.physical_times[call],
            metric="predicted_increment_relative_error",
            numerator=numerator,
            denominator=denominator,
            frame=frame,
            raw_physical_numerator=_weighted_norm(
                prediction_increment - truth_increment,
                weights=weights,
                scale=np.ones(4),
            ),
            raw_physical_denominator=_weighted_norm(
                truth_increment,
                weights=weights,
                scale=np.ones(4),
            ),
        )
    )
    for component in range(4):
        component_error = (
            prediction_increment[:, component : component + 1]
            - truth_increment[:, component : component + 1]
        )
        component_truth = truth_increment[:, component : component + 1]
        component_numerator = _weighted_norm(
            component_error,
            weights=weights,
            scale=case.residual_scale[component : component + 1],
        )
        component_denominator = _weighted_norm(
            component_truth,
            weights=weights,
            scale=case.residual_scale[component : component + 1],
        )
        rows.append(
            _metric_row(
                case_id=case.case_id,
                phase=phase,
                arm=arm,
                mode=mode,
                call=call,
                physical_time=case.physical_times[call],
                metric="predicted_increment_relative_error",
                numerator=component_numerator,
                denominator=component_denominator,
                frame=frame,
                component=str(component),
                raw_physical_numerator=_weighted_norm(
                    component_error,
                    weights=weights,
                    scale=np.ones(1),
                ),
                raw_physical_denominator=_weighted_norm(
                    component_truth,
                    weights=weights,
                    scale=np.ones(1),
                ),
            )
        )


def _record_native_residual_regions(
    rows: list[dict[str, Any]],
    case: BumpCase,
    prediction_increment: np.ndarray,
    truth_increment: np.ndarray,
    regions: Mapping[str, np.ndarray],
    *,
    phase: str,
    arm: str,
    mode: str,
    call: int,
    metric: str = "predicted_increment_relative_error",
    include_all_region: bool = True,
) -> None:
    weights = case.node_weights.sum(axis=-1)
    error = np.asarray(prediction_increment) - np.asarray(truth_increment)
    for region, mask in regions.items():
        if region == "all" and not include_all_region:
            # The all-component/all-region row is emitted by
            # _record_residual_metric.  Region rows are intentionally disjoint.
            continue
        numerator = _weighted_norm(
            error,
            weights=weights,
            scale=case.residual_scale,
            mask=mask,
        )
        denominator = _weighted_norm(
            truth_increment,
            weights=weights,
            scale=case.residual_scale,
            mask=mask,
        )
        rows.append(
            _metric_row(
                case_id=case.case_id,
                phase=phase,
                arm=arm,
                mode=mode,
                call=call,
                physical_time=case.physical_times[call],
                metric=metric,
                numerator=numerator,
                denominator=denominator,
                frame="native",
                region=region,
                raw_physical_numerator=_weighted_norm(
                    error, weights=weights, scale=np.ones(4), mask=mask
                ),
                raw_physical_denominator=_weighted_norm(
                    truth_increment,
                    weights=weights,
                    scale=np.ones(4),
                    mask=mask,
                ),
            )
        )


def _record_defect_metrics(
    rows: list[dict[str, Any]],
    case: BumpCase,
    defect: np.ndarray,
    reference_increment: np.ndarray,
    *,
    weights: np.ndarray,
    phase: str,
    arm: str,
    mode: str,
    call: int,
    metric: str,
    frame: str,
) -> None:
    for component in [None, *range(4)]:
        if component is None:
            selected_defect = defect
            selected_reference = reference_increment
            selected_scale = case.residual_scale
            label = "all"
        else:
            selected_defect = defect[:, component : component + 1]
            selected_reference = reference_increment[:, component : component + 1]
            selected_scale = case.residual_scale[component : component + 1]
            label = str(component)
        numerator = _weighted_norm(
            selected_defect, weights=weights, scale=selected_scale
        )
        denominator = _weighted_norm(
            selected_reference, weights=weights, scale=selected_scale
        )
        rows.append(
            _metric_row(
                case_id=case.case_id,
                phase=phase,
                arm=arm,
                mode=mode,
                call=call,
                physical_time=case.physical_times[call],
                metric=metric,
                numerator=numerator,
                denominator=denominator,
                frame=frame,
                component=label,
                raw_physical_numerator=_weighted_norm(
                    selected_defect,
                    weights=weights,
                    scale=np.ones(selected_defect.shape[1]),
                ),
                raw_physical_denominator=_weighted_norm(
                    selected_reference,
                    weights=weights,
                    scale=np.ones(selected_reference.shape[1]),
                ),
            )
        )


def _record_native_defect_regions(
    rows: list[dict[str, Any]],
    case: BumpCase,
    defect: np.ndarray,
    reference_increment: np.ndarray,
    regions: Mapping[str, np.ndarray],
    *,
    phase: str,
    arm: str,
    mode: str,
    call: int,
    metric: str,
    include_all_region: bool = True,
) -> None:
    weights = case.node_weights.sum(axis=-1)
    for region, mask in regions.items():
        if region == "all" and not include_all_region:
            # The all-component/all-region row is emitted by
            # _record_defect_metrics.  Region rows are intentionally disjoint.
            continue
        numerator = _weighted_norm(
            defect,
            weights=weights,
            scale=case.residual_scale,
            mask=mask,
        )
        denominator = _weighted_norm(
            reference_increment,
            weights=weights,
            scale=case.residual_scale,
            mask=mask,
        )
        rows.append(
            _metric_row(
                case_id=case.case_id,
                phase=phase,
                arm=arm,
                mode=mode,
                call=call,
                physical_time=case.physical_times[call],
                metric=metric,
                numerator=numerator,
                denominator=denominator,
                frame="native",
                region=region,
                raw_physical_numerator=_weighted_norm(
                    defect, weights=weights, scale=np.ones(4), mask=mask
                ),
                raw_physical_denominator=_weighted_norm(
                    reference_increment,
                    weights=weights,
                    scale=np.ones(4),
                    mask=mask,
                ),
            )
        )


def _record_shock_structure(
    rows: list[dict[str, Any]],
    case: BumpCase,
    prediction: np.ndarray,
    reference: np.ndarray,
    *,
    phase: str,
    arm: str,
    call: int,
    shock_quantile: float,
) -> None:
    pred_primitive = conservative_to_primitive_raw(prediction, gamma=case.gamma)
    ref_primitive = conservative_to_primitive_raw(reference, gamma=case.gamma)
    pred_scores = shock_front_scores(pred_primitive, case.directed_edges)
    ref_scores = shock_front_scores(ref_primitive, case.directed_edges)
    weights = case.node_weights.sum(axis=-1)
    numerator = _weighted_norm(
        (pred_scores - ref_scores)[:, None],
        weights=weights,
        scale=np.ones(1),
    )
    denominator = _weighted_norm(ref_scores[:, None], weights=weights, scale=np.ones(1))
    rows.append(
        _metric_row(
            case_id=case.case_id,
            phase=phase,
            arm=arm,
            mode="free_rollout",
            call=call,
            physical_time=case.physical_times[call],
            metric="shock_score_relative_error",
            numerator=numerator,
            denominator=denominator,
            frame="native",
            region="shock_proxy",
            numerator_contract="dimensionless_shock_score_difference",
        )
    )
    pred_mask = shock_indicator(
        pred_primitive, case.directed_edges, quantile=shock_quantile
    )
    ref_mask = shock_indicator(
        ref_primitive, case.directed_edges, quantile=shock_quantile
    )
    union = int(np.count_nonzero(pred_mask | ref_mask))
    intersection = int(np.count_nonzero(pred_mask & ref_mask))
    rows.append(
        _metric_row(
            case_id=case.case_id,
            phase=phase,
            arm=arm,
            mode="free_rollout",
            call=call,
            physical_time=case.physical_times[call],
            metric="shock_mask_jaccard",
            numerator=float(intersection),
            denominator=float(union),
            frame="native",
            region="shock_proxy",
            numerator_contract="node_count",
        )
    )


def _temporal_structure_rows(
    case: BumpCase,
    sequence: Sequence[np.ndarray],
    *,
    weights: np.ndarray,
    phase: str,
    arm: str,
    frame: str,
) -> list[dict[str, Any]]:
    if not sequence:
        return []
    defects = np.asarray(sequence, dtype=np.float64)
    mass = np.asarray(weights, dtype=np.float64).reshape(-1)
    scaled = defects / case.residual_scale[None, None, :]
    scaled *= np.sqrt(mass / mass.sum())[None, :, None]
    matrix = scaled.reshape((scaled.shape[0], -1))
    norms = np.linalg.norm(matrix, axis=1)
    cumulative = matrix.sum(axis=0)
    coherence = float(np.linalg.norm(cumulative) / max(float(norms.sum()), 1.0e-30))
    lag_cosines = []
    for index in range(1, matrix.shape[0]):
        denominator = float(norms[index - 1] * norms[index])
        if denominator > 1.0e-30:
            lag_cosines.append(
                float(np.dot(matrix[index - 1], matrix[index]) / denominator)
            )
    gram = matrix @ matrix.T
    eigenvalues = np.linalg.eigvalsh(gram)[::-1]
    eigenvalues = np.maximum(eigenvalues, 0.0)
    total = float(eigenvalues.sum())
    fractions = np.cumsum(eigenvalues) / max(total, 1.0e-30)
    modes_95 = int(np.searchsorted(fractions, 0.95) + 1)
    values = {
        "temporal_coherence": coherence,
        "lag1_cosine_mean": (None if not lag_cosines else float(np.mean(lag_cosines))),
        "lag1_cosine_median": (
            None if not lag_cosines else float(np.median(lag_cosines))
        ),
        "pod_energy_fraction_rank1": float(fractions[0]),
        "pod_energy_fraction_rank3": float(fractions[min(2, len(fractions) - 1)]),
        "pod_energy_fraction_rank8": float(fractions[min(7, len(fractions) - 1)]),
        "pod_modes_for_95_percent": float(modes_95),
    }
    rows = []
    for metric, value in values.items():
        if value is None:
            continue
        rows.append(
            _metric_row(
                case_id=case.case_id,
                phase=phase,
                arm=arm,
                mode="free_rollout",
                call=len(sequence),
                physical_time=case.physical_times[len(sequence)],
                metric=metric,
                numerator=float(value),
                frame=frame,
                numerator_contract=(
                    "mode_count"
                    if metric == "pod_modes_for_95_percent"
                    else "dimensionless_temporal_structure_statistic"
                ),
            )
        )
    return rows


def _phases_for_geometry(geometry: str, requested: str) -> tuple[str, ...]:
    enabled = ("g1a", "g1b") if requested == "both" else (requested,)
    if geometry == "native":
        return enabled
    if geometry == "query" and "g1a" in enabled:
        return ("g1a",)
    if geometry == "rotated" and "g1b" in enabled:
        return ("g1b",)
    return ()


def _evaluate_case(
    case: BumpCase,
    model: PCNOEuler2DResidual,
    args: argparse.Namespace,
    output_dir: Path,
    *,
    device: torch.device,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
    Path | None,
]:
    arms, query, geometry_audit = _build_geometry_arms(case, model, args, device=device)
    boundary_distance = graph_distance_to_mask(
        case.nodes, case.directed_edges, case.node_type != 0
    )
    metric_rows: list[dict[str, Any]] = []
    completion_rows: list[dict[str, Any]] = []
    closure_rows: list[dict[str, Any]] = []
    free_states = {
        geometry: {arm: np.array(payload.truth[0], copy=True) for arm in TYPE_ARMS}
        for geometry, payload in arms.items()
    }
    active = {geometry: {arm: True for arm in TYPE_ARMS} for geometry in arms}
    first_inadmissible = {
        geometry: {arm: None for arm in TYPE_ARMS} for geometry in arms
    }
    trajectories = {
        geometry: {
            arm: np.full(payload.truth.shape, np.nan, dtype=np.float32)
            for arm in TYPE_ARMS
        }
        for geometry, payload in arms.items()
    }
    for geometry, payload in arms.items():
        for arm in TYPE_ARMS:
            trajectories[geometry][arm][0] = payload.truth[0].astype(np.float32)

    query_defects = {arm: [] for arm in TYPE_ARMS}
    rotation_defects = {arm: [] for arm in TYPE_ARMS}
    retain_visual = case.case_id in set(args.visualization_cases)
    visual_decomposition: dict[str, np.ndarray] = {}
    if retain_visual and query is not None:
        for type_arm in TYPE_ARMS:
            for name in ("mesh", "state"):
                visual_decomposition[f"query_{name}_{type_arm}"] = np.full(
                    (args.rollout_calls, case.nodes.shape[0], 4),
                    np.nan,
                    dtype=np.float32,
                )
    if retain_visual and "rotated" in arms:
        for type_arm in TYPE_ARMS:
            for name in ("mesh", "state"):
                visual_decomposition[f"rotation_{name}_{type_arm}"] = np.full(
                    (args.rollout_calls, case.nodes.shape[0], 4),
                    np.nan,
                    dtype=np.float32,
                )

    for call in range(1, args.rollout_calls + 1):
        native_regions = _region_masks(
            case,
            case.states[call],
            boundary_distance,
            shock_quantile=args.shock_quantile,
        )
        native_truth_increment = case.states[call] - case.states[call - 1]
        teacher_predictions: dict[str, dict[str, np.ndarray]] = {}
        free_predictions: dict[str, dict[str, np.ndarray]] = {}
        free_currents: dict[str, dict[str, np.ndarray]] = {}
        admissibility: dict[str, dict[str, dict[str, Any]]] = {}
        for geometry, payload in arms.items():
            teacher_current = payload.truth[call - 1]
            teacher_predictions[geometry] = _predict(
                model,
                payload,
                (teacher_current, teacher_current),
                TYPE_ARMS,
                device=device,
            )
            active_arms = [arm for arm in TYPE_ARMS if active[geometry][arm]]
            free_currents[geometry] = {
                arm: np.array(free_states[geometry][arm], copy=True)
                for arm in active_arms
            }
            if call == 1:
                free_predictions[geometry] = {
                    arm: teacher_predictions[geometry][arm] for arm in active_arms
                }
            elif active_arms:
                free_predictions[geometry] = _predict(
                    model,
                    payload,
                    [free_currents[geometry][arm] for arm in active_arms],
                    active_arms,
                    device=device,
                )
            else:
                free_predictions[geometry] = {}
            admissibility[geometry] = {}

            truth_increment = payload.truth[call] - payload.truth[call - 1]
            for type_arm in TYPE_ARMS:
                arm_name = f"{geometry}_{type_arm}"
                teacher_increment = (
                    teacher_predictions[geometry][type_arm] - teacher_current
                )
                metric_teacher_increment = teacher_increment
                metric_truth_increment = truth_increment
                metric_frame = geometry
                if geometry == "rotated":
                    metric_teacher_increment = rotate_euler_field(
                        teacher_increment, inverse=True
                    )
                    metric_truth_increment = rotate_euler_field(
                        truth_increment, inverse=True
                    )
                    metric_frame = "native"
                native_teacher_increment = metric_teacher_increment
                if geometry == "query":
                    if query is None:
                        raise RuntimeError("query geometry lacks its restriction")
                    native_teacher_increment = query.prolong(teacher_increment)
                for phase in _phases_for_geometry(geometry, args.phase):
                    _record_residual_metric(
                        metric_rows,
                        case,
                        metric_teacher_increment,
                        metric_truth_increment,
                        weights=payload.weights,
                        phase=phase,
                        arm=arm_name,
                        mode="teacher_forced",
                        call=call,
                        frame=metric_frame,
                    )
                    _record_native_residual_regions(
                        metric_rows,
                        case,
                        native_teacher_increment,
                        native_truth_increment,
                        native_regions,
                        phase=phase,
                        arm=arm_name,
                        mode="teacher_forced",
                        call=call,
                        include_all_region=metric_frame != "native",
                    )
                if type_arm not in free_predictions[geometry]:
                    continue
                prediction = free_predictions[geometry][type_arm]
                current = free_currents[geometry][type_arm]
                status = conservative_admissibility_summary(
                    prediction, gamma=case.gamma
                )
                admissibility[geometry][type_arm] = status
                completion_rows.append(
                    {
                        "case_id": case.case_id,
                        "geometry": geometry,
                        "type_arm": type_arm,
                        "call": call,
                        "physical_time": float(case.physical_times[call]),
                        **status,
                    }
                )
                trajectories[geometry][type_arm][call] = prediction.astype(np.float32)
                free_increment = prediction - current
                metric_free_increment = free_increment
                if geometry == "rotated":
                    metric_free_increment = rotate_euler_field(
                        free_increment, inverse=True
                    )
                native_free_increment = metric_free_increment
                if geometry == "query":
                    if query is None:
                        raise RuntimeError("query geometry lacks its restriction")
                    native_free_increment = query.prolong(free_increment)
                for phase in _phases_for_geometry(geometry, args.phase):
                    _record_residual_metric(
                        metric_rows,
                        case,
                        metric_free_increment,
                        metric_truth_increment,
                        weights=payload.weights,
                        phase=phase,
                        arm=arm_name,
                        mode="free_rollout",
                        call=call,
                        frame=metric_frame,
                    )
                    _record_native_residual_regions(
                        metric_rows,
                        case,
                        native_free_increment,
                        native_truth_increment,
                        native_regions,
                        phase=phase,
                        arm=arm_name,
                        mode="free_rollout",
                        call=call,
                        include_all_region=metric_frame != "native",
                    )

                if geometry == "native":
                    native_prediction = prediction
                elif geometry == "query":
                    if query is None:
                        raise RuntimeError("query geometry lacks its restriction")
                    native_prediction = query.prolong(prediction)
                elif geometry == "rotated":
                    native_prediction = rotate_euler_field(prediction, inverse=True)
                else:
                    raise RuntimeError(f"unexpected geometry: {geometry}")
                for phase in _phases_for_geometry(geometry, args.phase):
                    _record_native_state_metrics(
                        metric_rows,
                        case,
                        native_prediction,
                        case.states[call],
                        native_regions,
                        phase=phase,
                        arm=arm_name,
                        call=call,
                    )
                    if status["admissible"]:
                        _record_shock_structure(
                            metric_rows,
                            case,
                            native_prediction,
                            case.states[call],
                            phase=phase,
                            arm=arm_name,
                            call=call,
                            shock_quantile=args.shock_quantile,
                        )
                if status["admissible"]:
                    free_states[geometry][type_arm] = prediction
                else:
                    active[geometry][type_arm] = False
                    first_inadmissible[geometry][type_arm] = call

        query_same_input_predictions: dict[str, np.ndarray] = {}
        if "query" in arms:
            if query is None:
                raise RuntimeError("query arm is missing its mapping")
            common_arms = [
                arm
                for arm in TYPE_ARMS
                if arm in free_predictions["native"]
                and arm in free_predictions["query"]
            ]
            if call == 1:
                query_same_input_predictions = {
                    arm: free_predictions["query"][arm] for arm in common_arms
                }
            elif common_arms:
                query_same_input_predictions = _predict(
                    model,
                    arms["query"],
                    [
                        query.restrict(free_currents["native"][arm])
                        for arm in common_arms
                    ],
                    common_arms,
                    device=device,
                )

        rotation_same_input_predictions: dict[str, np.ndarray] = {}
        if "rotated" in arms:
            common_arms = [
                arm
                for arm in TYPE_ARMS
                if arm in free_predictions["native"]
                and arm in free_predictions["rotated"]
            ]
            if call == 1:
                rotation_same_input_predictions = {
                    arm: free_predictions["rotated"][arm] for arm in common_arms
                }
            elif common_arms:
                rotation_same_input_predictions = _predict(
                    model,
                    arms["rotated"],
                    [
                        rotate_euler_field(free_currents["native"][arm])
                        for arm in common_arms
                    ],
                    common_arms,
                    device=device,
                )

        if "query" in arms:
            if query is None:
                raise RuntimeError("query arm is missing its mapping")
            query_truth = arms["query"].truth
            true_increment = query_truth[call] - query_truth[call - 1]
            cumulative_truth = query_truth[call] - query_truth[0]
            restriction_floor = cumulative_truth - query.restrict(
                case.states[call] - case.states[0]
            )
            metric_rows.append(
                _metric_row(
                    case_id=case.case_id,
                    phase="g1a",
                    arm="reference_query_vs_native",
                    mode="reference",
                    call=call,
                    physical_time=case.physical_times[call],
                    metric="accumulated_change_restriction_floor",
                    numerator=_weighted_norm(
                        restriction_floor,
                        weights=arms["query"].weights,
                        scale=case.state_scale,
                    ),
                    denominator=_weighted_norm(
                        cumulative_truth,
                        weights=arms["query"].weights,
                        scale=case.state_scale,
                    ),
                    frame="query",
                    raw_physical_numerator=_weighted_norm(
                        restriction_floor,
                        weights=arms["query"].weights,
                        scale=np.ones(4),
                    ),
                    raw_physical_denominator=_weighted_norm(
                        cumulative_truth,
                        weights=arms["query"].weights,
                        scale=np.ones(4),
                    ),
                )
            )
            information_loss = query.prolong(query_truth[call]) - case.states[call]
            metric_rows.append(
                _metric_row(
                    case_id=case.case_id,
                    phase="g1a",
                    arm="reference_query_vs_native",
                    mode="reference",
                    call=call,
                    physical_time=case.physical_times[call],
                    metric="native_information_loss",
                    numerator=_weighted_norm(
                        information_loss,
                        weights=arms["native"].weights,
                        scale=case.state_scale,
                    ),
                    denominator=_weighted_norm(
                        case.states[call],
                        weights=arms["native"].weights,
                        scale=case.state_scale,
                    ),
                    frame="native",
                    raw_physical_numerator=_weighted_norm(
                        information_loss,
                        weights=arms["native"].weights,
                        scale=np.ones(4),
                    ),
                    raw_physical_denominator=_weighted_norm(
                        case.states[call],
                        weights=arms["native"].weights,
                        scale=np.ones(4),
                    ),
                )
            )
            accumulated_loss = query.prolong(cumulative_truth) - (
                case.states[call] - case.states[0]
            )
            metric_rows.append(
                _metric_row(
                    case_id=case.case_id,
                    phase="g1a",
                    arm="reference_query_vs_native",
                    mode="reference",
                    call=call,
                    physical_time=case.physical_times[call],
                    metric="accumulated_change_information_loss",
                    numerator=_weighted_norm(
                        accumulated_loss,
                        weights=arms["native"].weights,
                        scale=case.state_scale,
                    ),
                    denominator=_weighted_norm(
                        case.states[call] - case.states[0],
                        weights=arms["native"].weights,
                        scale=case.state_scale,
                    ),
                    frame="native",
                    raw_physical_numerator=_weighted_norm(
                        accumulated_loss,
                        weights=arms["native"].weights,
                        scale=np.ones(4),
                    ),
                    raw_physical_denominator=_weighted_norm(
                        case.states[call] - case.states[0],
                        weights=arms["native"].weights,
                        scale=np.ones(4),
                    ),
                )
            )
            for type_arm in TYPE_ARMS:
                native_tf_current = arms["native"].truth[call - 1]
                query_tf_current = query_truth[call - 1]
                native_tf_output = teacher_predictions["native"][type_arm]
                query_tf_output = teacher_predictions["query"][type_arm]
                native_tf_increment = native_tf_output - native_tf_current
                query_tf_increment = query_tf_output - query_tf_current
                teacher_defect = query_tf_increment - query.restrict(
                    native_tf_increment
                )
                output_commutator = query_tf_output - query.restrict(native_tf_output)
                identity = _difference_metrics(teacher_defect, output_commutator)
                _require_close(
                    "teacher-forced query commutator identity",
                    identity,
                    absolute_limit=1.0e-10,
                    relative_limit=math.inf,
                )
                cross_arm = f"query_vs_native_{type_arm}"
                metric_rows.append(
                    _metric_row(
                        case_id=case.case_id,
                        phase="g1a",
                        arm=cross_arm,
                        mode="teacher_forced",
                        call=call,
                        physical_time=case.physical_times[call],
                        metric="increment_commutator",
                        numerator=_weighted_norm(
                            teacher_defect,
                            weights=arms["query"].weights,
                            scale=case.residual_scale,
                        ),
                        denominator=_weighted_norm(
                            true_increment,
                            weights=arms["query"].weights,
                            scale=case.residual_scale,
                        ),
                        frame="query",
                        raw_physical_numerator=_weighted_norm(
                            teacher_defect,
                            weights=arms["query"].weights,
                            scale=np.ones(4),
                        ),
                        raw_physical_denominator=_weighted_norm(
                            true_increment,
                            weights=arms["query"].weights,
                            scale=np.ones(4),
                        ),
                    )
                )
                _record_defect_metrics(
                    metric_rows,
                    case,
                    teacher_defect,
                    true_increment,
                    weights=arms["query"].weights,
                    phase="g1a",
                    arm=cross_arm,
                    mode="teacher_forced",
                    call=call,
                    metric="teacher_forced_mesh_defect",
                    frame="query",
                )
                _record_native_defect_regions(
                    metric_rows,
                    case,
                    query.prolong(teacher_defect),
                    native_truth_increment,
                    native_regions,
                    phase="g1a",
                    arm=cross_arm,
                    mode="teacher_forced",
                    call=call,
                    metric="teacher_forced_mesh_defect",
                )
                closure_rows.append(
                    {
                        "case_id": case.case_id,
                        "phase": "g1a",
                        "arm": cross_arm,
                        "call": call,
                        "closure": "paired_increment_equals_output_commutator",
                        **identity,
                    }
                )
                if (
                    type_arm not in free_predictions["native"]
                    or type_arm not in free_predictions["query"]
                ):
                    continue
                native_current = free_currents["native"][type_arm]
                query_current = free_currents["query"][type_arm]
                native_output = free_predictions["native"][type_arm]
                query_output = free_predictions["query"][type_arm]
                error_before = query_current - query.restrict(native_current)
                defect = (query_output - query_current) - query.restrict(
                    native_output - native_current
                )
                error_after = query_output - query.restrict(native_output)
                matched_current = query.restrict(native_current)
                matched_output = query_same_input_predictions[type_arm]
                mesh_defect = (matched_output - matched_current) - query.restrict(
                    native_output - native_current
                )
                state_defect = (query_output - query_current) - (
                    matched_output - matched_current
                )
                if retain_visual:
                    visual_decomposition[f"query_mesh_{type_arm}"][call - 1] = (
                        query.prolong(mesh_defect).astype(np.float32)
                    )
                    visual_decomposition[f"query_state_{type_arm}"][call - 1] = (
                        query.prolong(state_defect).astype(np.float32)
                    )
                decomposition = _difference_metrics(defect, mesh_defect + state_defect)
                _require_close(
                    "free query mesh/state decomposition",
                    decomposition,
                    absolute_limit=1.0e-10,
                    relative_limit=math.inf,
                )
                recurrence = _difference_metrics(error_after, error_before + defect)
                _require_close(
                    "free query recurrence",
                    recurrence,
                    absolute_limit=1.0e-10,
                    relative_limit=math.inf,
                )
                lhs = _weighted_inner(
                    error_after,
                    error_after,
                    weights=arms["query"].weights,
                    scale=case.state_scale,
                ) - _weighted_inner(
                    error_before,
                    error_before,
                    weights=arms["query"].weights,
                    scale=case.state_scale,
                )
                cross_term = 2.0 * _weighted_inner(
                    error_before,
                    defect,
                    weights=arms["query"].weights,
                    scale=case.state_scale,
                )
                defect_energy = _weighted_inner(
                    defect,
                    defect,
                    weights=arms["query"].weights,
                    scale=case.state_scale,
                )
                rhs = cross_term + defect_energy
                growth_error = abs(lhs - rhs)
                if not all(math.isfinite(value) for value in (lhs, rhs, growth_error)):
                    raise RuntimeError("free query signed-growth identity is nonfinite")
                if growth_error > 1.0e-10:
                    raise RuntimeError("free query signed-growth identity failed")
                defect_norm = _weighted_norm(
                    defect,
                    weights=arms["query"].weights,
                    scale=case.residual_scale,
                )
                true_norm = _weighted_norm(
                    true_increment,
                    weights=arms["query"].weights,
                    scale=case.residual_scale,
                )
                metric_rows.extend(
                    (
                        _metric_row(
                            case_id=case.case_id,
                            phase="g1a",
                            arm=cross_arm,
                            mode="free_rollout",
                            call=call,
                            physical_time=case.physical_times[call],
                            metric="increment_commutator",
                            numerator=defect_norm,
                            denominator=true_norm,
                            frame="query",
                            raw_physical_numerator=_weighted_norm(
                                defect,
                                weights=arms["query"].weights,
                                scale=np.ones(4),
                            ),
                            raw_physical_denominator=_weighted_norm(
                                true_increment,
                                weights=arms["query"].weights,
                                scale=np.ones(4),
                            ),
                        ),
                        _metric_row(
                            case_id=case.case_id,
                            phase="g1a",
                            arm=cross_arm,
                            mode="free_rollout",
                            call=call,
                            physical_time=case.physical_times[call],
                            metric="accumulated_predicted_gap",
                            numerator=_weighted_norm(
                                error_after,
                                weights=arms["query"].weights,
                                scale=case.state_scale,
                            ),
                            denominator=_weighted_norm(
                                cumulative_truth,
                                weights=arms["query"].weights,
                                scale=case.state_scale,
                            ),
                            frame="query",
                            raw_physical_numerator=_weighted_norm(
                                error_after,
                                weights=arms["query"].weights,
                                scale=np.ones(4),
                            ),
                            raw_physical_denominator=_weighted_norm(
                                cumulative_truth,
                                weights=arms["query"].weights,
                                scale=np.ones(4),
                            ),
                        ),
                        _metric_row(
                            case_id=case.case_id,
                            phase="g1a",
                            arm=cross_arm,
                            mode="free_rollout",
                            call=call,
                            physical_time=case.physical_times[call],
                            metric="signed_error_growth",
                            numerator=lhs,
                            frame="query",
                            numerator_contract=(
                                "component_scaled_proxy_weighted_mean_squared_"
                                "error_change"
                            ),
                        ),
                        _metric_row(
                            case_id=case.case_id,
                            phase="g1a",
                            arm=cross_arm,
                            mode="free_rollout",
                            call=call,
                            physical_time=case.physical_times[call],
                            metric="signed_growth_cross_term_2e_delta",
                            numerator=cross_term,
                            frame="query",
                            numerator_contract=(
                                "component_scaled_proxy_weighted_mean_squared_"
                                "cross_term"
                            ),
                        ),
                        _metric_row(
                            case_id=case.case_id,
                            phase="g1a",
                            arm=cross_arm,
                            mode="free_rollout",
                            call=call,
                            physical_time=case.physical_times[call],
                            metric="signed_growth_defect_energy",
                            numerator=defect_energy,
                            frame="query",
                            numerator_contract=(
                                "component_scaled_proxy_weighted_mean_squared_"
                                "defect_energy"
                            ),
                        ),
                    )
                )
                for name, value in (
                    ("free_total_defect", defect),
                    ("free_mesh_defect", mesh_defect),
                    ("free_state_defect", state_defect),
                ):
                    _record_defect_metrics(
                        metric_rows,
                        case,
                        value,
                        true_increment,
                        weights=arms["query"].weights,
                        phase="g1a",
                        arm=cross_arm,
                        mode="free_rollout",
                        call=call,
                        metric=name,
                        frame="query",
                    )
                    _record_native_defect_regions(
                        metric_rows,
                        case,
                        query.prolong(value),
                        native_truth_increment,
                        native_regions,
                        phase="g1a",
                        arm=cross_arm,
                        mode="free_rollout",
                        call=call,
                        metric=name,
                    )
                closure_rows.append(
                    {
                        "case_id": case.case_id,
                        "phase": "g1a",
                        "arm": cross_arm,
                        "call": call,
                        "closure": "free_recurrence_and_signed_growth",
                        **recurrence,
                        "signed_growth_absolute_error": growth_error,
                        "signed_growth_lhs": lhs,
                        "signed_growth_rhs": rhs,
                        "mesh_state_decomposition_max_abs": decomposition["max_abs"],
                        "mesh_state_decomposition_relative_l2": decomposition[
                            "relative_l2"
                        ],
                    }
                )
                if (
                    admissibility["native"][type_arm]["admissible"]
                    and admissibility["query"][type_arm]["admissible"]
                ):
                    query_defects[type_arm].append(defect)

        if "rotated" in arms:
            cumulative_truth = case.states[call] - case.states[0]
            inverse_reference = rotate_euler_field(
                arms["rotated"].truth[call], inverse=True
            )
            reference_floor = inverse_reference - case.states[call]
            metric_rows.append(
                _metric_row(
                    case_id=case.case_id,
                    phase="g1b",
                    arm="reference_rotated_vs_native",
                    mode="reference",
                    call=call,
                    physical_time=case.physical_times[call],
                    metric="rotation_transformation_floor",
                    numerator=_weighted_norm(
                        reference_floor,
                        weights=arms["native"].weights,
                        scale=case.state_scale,
                    ),
                    denominator=_weighted_norm(
                        case.states[call],
                        weights=arms["native"].weights,
                        scale=case.state_scale,
                    ),
                    frame="native",
                    raw_physical_numerator=_weighted_norm(
                        reference_floor,
                        weights=arms["native"].weights,
                        scale=np.ones(4),
                    ),
                    raw_physical_denominator=_weighted_norm(
                        case.states[call],
                        weights=arms["native"].weights,
                        scale=np.ones(4),
                    ),
                )
            )
            true_increment = case.states[call] - case.states[call - 1]
            for type_arm in TYPE_ARMS:
                native_tf_current = case.states[call - 1]
                rotated_tf_current = arms["rotated"].truth[call - 1]
                native_tf_output = teacher_predictions["native"][type_arm]
                rotated_tf_output = teacher_predictions["rotated"][type_arm]
                native_tf_increment = native_tf_output - native_tf_current
                inverse_rotated_increment = rotate_euler_field(
                    rotated_tf_output - rotated_tf_current, inverse=True
                )
                teacher_defect = inverse_rotated_increment - native_tf_increment
                output_commutator = (
                    rotate_euler_field(rotated_tf_output, inverse=True)
                    - native_tf_output
                )
                identity = _difference_metrics(teacher_defect, output_commutator)
                _require_close(
                    "teacher-forced rotation commutator identity",
                    identity,
                    absolute_limit=1.0e-10,
                    relative_limit=math.inf,
                )
                cross_arm = f"rotated_vs_native_{type_arm}"
                metric_rows.append(
                    _metric_row(
                        case_id=case.case_id,
                        phase="g1b",
                        arm=cross_arm,
                        mode="teacher_forced",
                        call=call,
                        physical_time=case.physical_times[call],
                        metric="increment_covariance_defect",
                        numerator=_weighted_norm(
                            teacher_defect,
                            weights=arms["native"].weights,
                            scale=case.residual_scale,
                        ),
                        denominator=_weighted_norm(
                            true_increment,
                            weights=arms["native"].weights,
                            scale=case.residual_scale,
                        ),
                        frame="native",
                        raw_physical_numerator=_weighted_norm(
                            teacher_defect,
                            weights=arms["native"].weights,
                            scale=np.ones(4),
                        ),
                        raw_physical_denominator=_weighted_norm(
                            true_increment,
                            weights=arms["native"].weights,
                            scale=np.ones(4),
                        ),
                    )
                )
                _record_defect_metrics(
                    metric_rows,
                    case,
                    teacher_defect,
                    true_increment,
                    weights=arms["native"].weights,
                    phase="g1b",
                    arm=cross_arm,
                    mode="teacher_forced",
                    call=call,
                    metric="teacher_forced_rotation_defect",
                    frame="native",
                )
                _record_native_defect_regions(
                    metric_rows,
                    case,
                    teacher_defect,
                    native_truth_increment,
                    native_regions,
                    phase="g1b",
                    arm=cross_arm,
                    mode="teacher_forced",
                    call=call,
                    metric="teacher_forced_rotation_defect",
                    include_all_region=False,
                )
                closure_rows.append(
                    {
                        "case_id": case.case_id,
                        "phase": "g1b",
                        "arm": cross_arm,
                        "call": call,
                        "closure": "paired_increment_equals_output_commutator",
                        **identity,
                    }
                )
                if (
                    type_arm not in free_predictions["native"]
                    or type_arm not in free_predictions["rotated"]
                ):
                    continue
                native_current = free_currents["native"][type_arm]
                rotated_current = free_currents["rotated"][type_arm]
                native_output = free_predictions["native"][type_arm]
                rotated_output = free_predictions["rotated"][type_arm]
                error_before = (
                    rotate_euler_field(rotated_current, inverse=True) - native_current
                )
                defect = rotate_euler_field(
                    rotated_output - rotated_current, inverse=True
                ) - (native_output - native_current)
                error_after = (
                    rotate_euler_field(rotated_output, inverse=True) - native_output
                )
                matched_current = rotate_euler_field(native_current)
                matched_output = rotation_same_input_predictions[type_arm]
                mesh_defect = rotate_euler_field(
                    matched_output - matched_current, inverse=True
                ) - (native_output - native_current)
                state_defect = rotate_euler_field(
                    rotated_output - rotated_current, inverse=True
                ) - rotate_euler_field(matched_output - matched_current, inverse=True)
                if retain_visual:
                    visual_decomposition[f"rotation_mesh_{type_arm}"][call - 1] = (
                        mesh_defect.astype(np.float32)
                    )
                    visual_decomposition[f"rotation_state_{type_arm}"][call - 1] = (
                        state_defect.astype(np.float32)
                    )
                decomposition = _difference_metrics(defect, mesh_defect + state_defect)
                _require_close(
                    "free rotation mesh/state decomposition",
                    decomposition,
                    absolute_limit=1.0e-10,
                    relative_limit=math.inf,
                )
                recurrence = _difference_metrics(error_after, error_before + defect)
                _require_close(
                    "free rotation recurrence",
                    recurrence,
                    absolute_limit=1.0e-10,
                    relative_limit=math.inf,
                )
                lhs = _weighted_inner(
                    error_after,
                    error_after,
                    weights=arms["native"].weights,
                    scale=case.state_scale,
                ) - _weighted_inner(
                    error_before,
                    error_before,
                    weights=arms["native"].weights,
                    scale=case.state_scale,
                )
                cross_term = 2.0 * _weighted_inner(
                    error_before,
                    defect,
                    weights=arms["native"].weights,
                    scale=case.state_scale,
                )
                defect_energy = _weighted_inner(
                    defect,
                    defect,
                    weights=arms["native"].weights,
                    scale=case.state_scale,
                )
                rhs = cross_term + defect_energy
                growth_error = abs(lhs - rhs)
                if not all(math.isfinite(value) for value in (lhs, rhs, growth_error)):
                    raise RuntimeError(
                        "free rotation signed-growth identity is nonfinite"
                    )
                if growth_error > 1.0e-10:
                    raise RuntimeError("free rotation signed-growth identity failed")
                metric_rows.extend(
                    (
                        _metric_row(
                            case_id=case.case_id,
                            phase="g1b",
                            arm=cross_arm,
                            mode="free_rollout",
                            call=call,
                            physical_time=case.physical_times[call],
                            metric="increment_covariance_defect",
                            numerator=_weighted_norm(
                                defect,
                                weights=arms["native"].weights,
                                scale=case.residual_scale,
                            ),
                            denominator=_weighted_norm(
                                true_increment,
                                weights=arms["native"].weights,
                                scale=case.residual_scale,
                            ),
                            frame="native",
                            raw_physical_numerator=_weighted_norm(
                                defect,
                                weights=arms["native"].weights,
                                scale=np.ones(4),
                            ),
                            raw_physical_denominator=_weighted_norm(
                                true_increment,
                                weights=arms["native"].weights,
                                scale=np.ones(4),
                            ),
                        ),
                        _metric_row(
                            case_id=case.case_id,
                            phase="g1b",
                            arm=cross_arm,
                            mode="free_rollout",
                            call=call,
                            physical_time=case.physical_times[call],
                            metric="rotation_consistency_CQ",
                            numerator=_weighted_norm(
                                error_after,
                                weights=arms["native"].weights,
                                scale=case.state_scale,
                            ),
                            denominator=_weighted_norm(
                                case.states[call],
                                weights=arms["native"].weights,
                                scale=case.state_scale,
                            ),
                            frame="native",
                            raw_physical_numerator=_weighted_norm(
                                error_after,
                                weights=arms["native"].weights,
                                scale=np.ones(4),
                            ),
                            raw_physical_denominator=_weighted_norm(
                                case.states[call],
                                weights=arms["native"].weights,
                                scale=np.ones(4),
                            ),
                        ),
                        _metric_row(
                            case_id=case.case_id,
                            phase="g1b",
                            arm=cross_arm,
                            mode="free_rollout",
                            call=call,
                            physical_time=case.physical_times[call],
                            metric="signed_error_growth",
                            numerator=lhs,
                            frame="native",
                            numerator_contract=(
                                "component_scaled_proxy_weighted_mean_squared_"
                                "error_change"
                            ),
                        ),
                        _metric_row(
                            case_id=case.case_id,
                            phase="g1b",
                            arm=cross_arm,
                            mode="free_rollout",
                            call=call,
                            physical_time=case.physical_times[call],
                            metric="signed_growth_cross_term_2e_delta",
                            numerator=cross_term,
                            frame="native",
                            numerator_contract=(
                                "component_scaled_proxy_weighted_mean_squared_"
                                "cross_term"
                            ),
                        ),
                        _metric_row(
                            case_id=case.case_id,
                            phase="g1b",
                            arm=cross_arm,
                            mode="free_rollout",
                            call=call,
                            physical_time=case.physical_times[call],
                            metric="signed_growth_defect_energy",
                            numerator=defect_energy,
                            frame="native",
                            numerator_contract=(
                                "component_scaled_proxy_weighted_mean_squared_"
                                "defect_energy"
                            ),
                        ),
                    )
                )
                for name, value in (
                    ("free_total_defect", defect),
                    ("free_mesh_defect", mesh_defect),
                    ("free_state_defect", state_defect),
                ):
                    _record_defect_metrics(
                        metric_rows,
                        case,
                        value,
                        native_truth_increment,
                        weights=arms["native"].weights,
                        phase="g1b",
                        arm=cross_arm,
                        mode="free_rollout",
                        call=call,
                        metric=name,
                        frame="native",
                    )
                    _record_native_defect_regions(
                        metric_rows,
                        case,
                        value,
                        native_truth_increment,
                        native_regions,
                        phase="g1b",
                        arm=cross_arm,
                        mode="free_rollout",
                        call=call,
                        metric=name,
                        include_all_region=False,
                    )
                closure_rows.append(
                    {
                        "case_id": case.case_id,
                        "phase": "g1b",
                        "arm": cross_arm,
                        "call": call,
                        "closure": "free_recurrence_and_signed_growth",
                        **recurrence,
                        "signed_growth_absolute_error": growth_error,
                        "signed_growth_lhs": lhs,
                        "signed_growth_rhs": rhs,
                        "mesh_state_decomposition_max_abs": decomposition["max_abs"],
                        "mesh_state_decomposition_relative_l2": decomposition[
                            "relative_l2"
                        ],
                    }
                )
                if (
                    admissibility["native"][type_arm]["admissible"]
                    and admissibility["rotated"][type_arm]["admissible"]
                ):
                    rotation_defects[type_arm].append(defect)

    if query is not None:
        for type_arm in TYPE_ARMS:
            metric_rows.extend(
                _temporal_structure_rows(
                    case,
                    query_defects[type_arm],
                    weights=arms["query"].weights,
                    phase="g1a",
                    arm=f"query_vs_native_{type_arm}",
                    frame="query",
                )
            )
    if "rotated" in arms:
        for type_arm in TYPE_ARMS:
            metric_rows.extend(
                _temporal_structure_rows(
                    case,
                    rotation_defects[type_arm],
                    weights=arms["native"].weights,
                    phase="g1b",
                    arm=f"rotated_vs_native_{type_arm}",
                    frame="native",
                )
            )

    completion_summary = []
    for geometry in arms:
        for type_arm in TYPE_ARMS:
            first = first_inadmissible[geometry][type_arm]
            completion_summary.append(
                {
                    "case_id": case.case_id,
                    "geometry": geometry,
                    "type_arm": type_arm,
                    "fully_admissible": first is None,
                    "first_inadmissible_call": first,
                    "last_admissible_call": (
                        args.rollout_calls if first is None else int(first) - 1
                    ),
                    "horizon_output_produced": bool(
                        np.isfinite(trajectories[geometry][type_arm][-1]).all()
                    ),
                }
            )

    bundle_path = None
    if case.case_id in set(args.visualization_cases):
        arrays: dict[str, Any] = {
            "schema": np.asarray(SCHEMA),
            "case_id": np.asarray(case.case_id),
            "phase": np.asarray(args.phase),
            "physical_times": case.physical_times,
            "nodes": case.nodes.astype(np.float32),
            "physical_node_type": case.node_type.astype(np.int64),
            "node_weights": case.node_weights.astype(np.float64),
            "boundary_distance": boundary_distance.astype(np.float64),
            "truth": case.states.astype(np.float32),
            "state_scale": case.state_scale.astype(np.float64),
            "residual_scale": case.residual_scale.astype(np.float64),
            "gamma": np.asarray(case.gamma),
            "color_scale_contract": np.asarray(
                "reference_only_rollout_wide_physical_no_per_frame_normalization"
            ),
        }
        for type_arm in TYPE_ARMS:
            arrays[f"native_{type_arm}"] = trajectories["native"][type_arm]
        if query is not None:
            arrays["query_nodes"] = query.nodes.astype(np.float32)
            arrays["query_node_type"] = query.node_type.astype(np.int64)
            arrays["query_fine_to_coarse"] = query.fine_to_coarse.astype(np.int64)
            arrays["query_anchor_indices"] = query.anchor_indices.astype(np.int64)
            arrays["query_truth_prolonged"] = query.prolong(arms["query"].truth).astype(
                np.float32
            )
            for type_arm in TYPE_ARMS:
                arrays[f"query_{type_arm}_prolonged"] = query.prolong(
                    trajectories["query"][type_arm]
                ).astype(np.float32)
        if "rotated" in arms:
            arrays["rotation_matrix"] = ROTATION_90_CCW
            arrays["rotation_center"] = rotation_center(case.nodes)
            arrays["rotated_nodes"] = (
                arms["rotated"].sample["nodes"][0].detach().cpu().numpy()
            )
            arrays["rotated_truth_inverse"] = rotate_euler_field(
                arms["rotated"].truth, inverse=True
            ).astype(np.float32)
            for type_arm in TYPE_ARMS:
                arrays[f"rotated_{type_arm}_inverse"] = rotate_euler_field(
                    trajectories["rotated"][type_arm], inverse=True
                ).astype(np.float32)
        arrays.update(visual_decomposition)
        bundle_path = output_dir / "arrays" / f"bump_{case.case_id}.npz"
        np.savez_compressed(bundle_path, **arrays)
    return (
        metric_rows,
        completion_rows,
        closure_rows,
        {
            "geometry_audit": geometry_audit,
            "completion_summary": completion_summary,
        },
        bundle_path,
    )


@torch.inference_mode()
def _replay_gate(
    args: argparse.Namespace,
    model: PCNOEuler2DResidual,
    case: BumpCase,
    *,
    device: torch.device,
) -> dict[str, Any]:
    path = args.replay_root / f"trajectory_{args.replay_case}.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    if sha256_file(path) != REPLAY_SHA256:
        raise ValueError("D085 replay payload digest differs from frozen D068")
    with np.load(path, allow_pickle=False) as artifact:
        scalar_contract = {
            "schema": "pcno_euler2d_ripple_d013_v1",
            "trajectory_key": "128",
            "valid_length": 20,
            "failure_cause": "completed",
            "failure_call": -1,
            "start_frame": 0,
            "step_stride": 1,
            "delta_t": 0.025,
            "boundary_mode": "model_all_nodes",
            "coordinate_convention": "source_HDF5_node_order_xy",
            "state_convention": "conservative_[rho,rho_v1,rho_v2,E]",
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "config_digest": D041_CONFIG_DIGEST,
            "weight_provenance": (
                "reconstructed_vertex_lumped_proxy_and_equal_node_proxy"
            ),
        }
        mismatched_scalars = {
            name: {"actual": artifact[name].item(), "expected": expected_value}
            for name, expected_value in scalar_contract.items()
            if artifact[name].item() != expected_value
        }
        if mismatched_scalars:
            raise ValueError(
                f"D041 replay scalar contract differs: {mismatched_scalars}"
            )
        if artifact["checkpoint_sha256"].item() != args.expected_checkpoint_sha256:
            raise ValueError("D041 replay checkpoint digest differs")
        if artifact["boundary_mode"].item() != "model_all_nodes":
            raise ValueError("D041 replay physical policy differs")
        currents = np.asarray(artifact["rollout_currents"], dtype=np.float64)
        expected = np.asarray(artifact["predictions"], dtype=np.float64)
        reference_currents = np.asarray(artifact["reference_currents"])
        targets = np.asarray(artifact["targets"])
        positions = np.asarray(artifact["positions"])
        edges = np.asarray(artifact["edges"])
        node_type = np.asarray(artifact["node_type"])
        node_measures = np.asarray(artifact["node_measures_proxy"])
        node_weights = np.asarray(artifact["reconstructed_node_weights_proxy"])
        physical_times = np.asarray(artifact["physical_target_times"])
        mach = float(artifact["mach"].item())
    array_contract = {
        "positions": np.array_equal(positions, case.nodes.astype(np.float32)),
        "edges": np.array_equal(edges, case.edges),
        "node_type": np.array_equal(node_type, case.node_type),
        "node_measures": np.array_equal(
            node_measures, case.node_measures.astype(np.float32)
        ),
        "node_weights": np.array_equal(
            node_weights, case.node_weights.sum(axis=-1).astype(np.float32)
        ),
        "reference_currents": np.array_equal(
            reference_currents, case.states[:20].astype(np.float32)
        ),
        "targets": np.array_equal(targets, case.states[1:21].astype(np.float32)),
        "initial_current": np.array_equal(
            currents[0].astype(np.float32), case.states[0].astype(np.float32)
        ),
        "free_recurrence": np.array_equal(
            currents[1:].astype(np.float32), expected[:-1].astype(np.float32)
        ),
        "physical_times": np.array_equal(
            physical_times, 0.025 * np.arange(1, 21, dtype=np.float64)
        ),
        "mach": mach == case.mach,
    }
    failed_arrays = [name for name, passed in array_contract.items() if not passed]
    if failed_arrays:
        raise ValueError(f"D041 replay array contract differs: {failed_arrays}")
    replay_geometry, _, replay_geometry_audit = _build_geometry_arms(
        case, model, args, device=device
    )
    if set(replay_geometry) != {"native", "rotated"}:
        raise RuntimeError(
            "D085 replay must bind native and fixed-Fourier rotated arms"
        )
    geometry = replay_geometry["native"]
    rotated_geometry = replay_geometry["rotated"]
    predictions = []
    for index in range(args.replay_calls):
        predictions.append(
            _predict(
                model,
                geometry,
                (currents[index],),
                ("correct",),
                device=device,
            )["correct"]
        )
    actual = np.asarray(predictions)
    replay_metrics = _difference_metrics(actual, expected[: args.replay_calls])
    _require_close(
        "D041 open-validation replay",
        replay_metrics,
        absolute_limit=REPLAY_ABSOLUTE_LIMIT,
        relative_limit=REPLAY_RELATIVE_LIMIT,
    )

    deterministic_before = torch.are_deterministic_algorithms_enabled()
    with _deterministic_identity_gate(device):
        current = torch.as_tensor(
            currents[0].astype(np.float32), dtype=torch.float32, device=device
        ).unsqueeze(0)
        wrapper = (
            model(
                current,
                node_mask=geometry.sample["node_mask"],
                nodes=geometry.sample["nodes"],
                node_weights=geometry.sample["node_weights"],
                node_rhos=geometry.sample["node_rhos"],
                directed_edges=geometry.sample["directed_edges"],
                edge_gradient_weights=geometry.sample["edge_gradient_weights"],
                node_type=geometry.sample["node_type"],
                mach=geometry.sample["mach"],
            )[0]
            .detach()
            .float()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        custom = _predict(
            model,
            geometry,
            (currents[0],),
            ("correct",),
            device=device,
        )["correct"]
        custom_metrics = _increment_identity_metrics(
            custom,
            wrapper,
            currents[0],
            weights=geometry.weights,
            residual_scale=case.residual_scale,
        )
        _require_close(
            "deterministic custom Fourier inference identity",
            custom_metrics,
            absolute_limit=1.0e-6,
            relative_limit=1.0e-7,
        )
        rotated_current_raw = rotate_euler_field(currents[0])
        rotated_current = torch.as_tensor(
            rotated_current_raw.astype(np.float32),
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)
        rotated_wrapper = (
            model(
                rotated_current,
                node_mask=rotated_geometry.sample["node_mask"],
                nodes=rotated_geometry.sample["nodes"],
                node_weights=rotated_geometry.sample["node_weights"],
                node_rhos=rotated_geometry.sample["node_rhos"],
                directed_edges=rotated_geometry.sample["directed_edges"],
                edge_gradient_weights=rotated_geometry.sample["edge_gradient_weights"],
                node_type=rotated_geometry.sample["node_type"],
                mach=rotated_geometry.sample["mach"],
            )[0]
            .detach()
            .float()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        rotated_custom = _predict(
            model,
            rotated_geometry,
            (rotated_current_raw,),
            ("correct",),
            device=device,
        )["correct"]
        rotated_custom_metrics = _increment_identity_metrics(
            rotate_euler_field(rotated_custom, inverse=True),
            rotate_euler_field(rotated_wrapper, inverse=True),
            currents[0],
            weights=rotated_geometry.weights,
            residual_scale=case.residual_scale,
        )
        _require_close(
            "deterministic fixed-Fourier rotated custom-wrapper identity",
            rotated_custom_metrics,
            absolute_limit=1.0e-6,
            relative_limit=1.0e-7,
        )
        batched = _predict_batched(
            model,
            geometry,
            (currents[0], currents[0]),
            TYPE_ARMS,
            device=device,
        )
        batch_identity = {}
        for type_arm in TYPE_ARMS:
            sequential = _predict(
                model,
                geometry,
                (currents[0],),
                (type_arm,),
                device=device,
            )[type_arm]
            metrics = _increment_identity_metrics(
                batched[type_arm],
                sequential,
                currents[0],
                weights=geometry.weights,
                residual_scale=case.residual_scale,
            )
            _require_batch_consistency(
                f"deterministic batch-two versus sequential increment identity ({type_arm})",
                metrics,
            )
            batch_identity[type_arm] = metrics
    if torch.are_deterministic_algorithms_enabled() != deterministic_before:
        raise RuntimeError("deterministic identity gate did not restore runtime mode")
    return {
        "case_id": args.replay_case,
        "calls": args.replay_calls,
        "artifact_sha256": REPLAY_SHA256,
        "absolute_limit": REPLAY_ABSOLUTE_LIMIT,
        "relative_l2_limit": REPLAY_RELATIVE_LIMIT,
        "batch_increment_relative_limit": BATCH_INCREMENT_RELATIVE_LIMIT,
        "batch_pointwise_scaled_relative_limit": (
            BATCH_POINTWISE_SCALED_RELATIVE_LIMIT
        ),
        "scientific_inference_batch_size": 1,
        "deterministic_identity_gate": {
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "algorithms_enabled_during_gate": True,
            "algorithms_restored_after_gate": True,
            "ordinary_rollout_algorithms_deterministic": deterministic_before,
        },
        **replay_metrics,
        "custom_fourier_wrapper_identity": custom_metrics,
        "fixed_fourier_rotation_gate": {
            "policy": FOURIER_POLICY,
            "phase_origin": list(FOURIER_PHASE_ORIGIN),
            "custom_wrapper_identity_native_frame": rotated_custom_metrics,
            "geometry_audit": replay_geometry_audit["rotation"],
        },
        "batch_two_vs_sequential_identity": batch_identity,
        "array_contract": array_contract,
        "rollout_currents_sha256": _array_sha256(currents.astype(np.float32)),
        "passed": True,
    }


def _phase_geometry_pairs(requested_phase: str) -> tuple[tuple[str, str], ...]:
    pairs = {
        "g1a": (("g1a", "native"), ("g1a", "query")),
        "g1b": (("g1b", "native"), ("g1b", "rotated")),
        "both": (
            ("g1a", "native"),
            ("g1a", "query"),
            ("g1b", "native"),
            ("g1b", "rotated"),
        ),
    }
    return pairs[requested_phase]


def _completion_lookup(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str, str], Mapping[str, Any]]:
    lookup: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for row in rows:
        key = (str(row["case_id"]), str(row["geometry"]), str(row["type_arm"]))
        if key in lookup:
            raise ValueError(f"duplicate D085 completion-summary row: {key}")
        lookup[key] = row
    return lookup


def _matrix_coverage(
    completion: Sequence[Mapping[str, Any]],
    *,
    requested_phase: str,
    cases: Sequence[str],
) -> dict[str, Any]:
    lookup = _completion_lookup(completion)
    expected = [
        (phase, case_id, geometry, type_arm)
        for phase, geometry in _phase_geometry_pairs(requested_phase)
        for case_id in cases
        for type_arm in TYPE_ARMS
    ]
    missing = []
    inadmissible = []
    horizon_output_missing = []
    horizon_complete = []
    for phase, case_id, geometry, type_arm in expected:
        label = {
            "phase": phase,
            "case_id": case_id,
            "geometry": geometry,
            "type_arm": type_arm,
        }
        row = lookup.get((case_id, geometry, type_arm))
        if row is None:
            missing.append(label)
        elif not bool(row["fully_admissible"]):
            inadmissible.append(
                {**label, "first_inadmissible_call": row["first_inadmissible_call"]}
            )
        elif not bool(row["horizon_output_produced"]):
            horizon_output_missing.append(label)
        else:
            horizon_complete.append(label)
    expected_count = len(expected)
    attempted_count = expected_count - len(missing)
    complete_count = len(horizon_complete)
    return {
        "expected_phase_arm_case_count": expected_count,
        "attempted_phase_arm_case_count": attempted_count,
        "horizon_complete_phase_arm_case_count": complete_count,
        "missing_completion_count": len(missing),
        "inadmissible_phase_arm_case_count": len(inadmissible),
        "missing_horizon_output_count": len(horizon_output_missing),
        "declared_matrix_attempt_complete": (
            expected_count > 0 and attempted_count == expected_count
        ),
        "declared_matrix_horizon_complete": (
            expected_count > 0 and complete_count == expected_count
        ),
        "missing_completion": missing,
        "inadmissible": inadmissible,
        "missing_horizon_output": horizon_output_missing,
    }


def _final_summary(
    rows: Sequence[Mapping[str, Any]],
    completion: Sequence[Mapping[str, Any]],
    *,
    requested_phase: str,
    cases: Sequence[str],
    horizon: int,
) -> list[dict[str, Any]]:
    completion_by_arm = _completion_lookup(completion)
    endpoint_rows: dict[tuple[str, str, str], float] = {}
    for row in rows:
        if (
            row["mode"] == "free_rollout"
            and int(row["call"]) == horizon
            and row["metric"] == "state_relative_l2"
            and row["frame"] == "native"
            and row["region"] == "all"
            and row["component"] == "all"
            and row["value"] is not None
        ):
            key = (str(row["case_id"]), str(row["phase"]), str(row["arm"]))
            if key in endpoint_rows:
                raise ValueError(f"duplicate D085 endpoint metric row: {key}")
            endpoint_rows[key] = float(row["value"])

    result = []
    for phase, geometry in _phase_geometry_pairs(requested_phase):
        for type_arm in TYPE_ARMS:
            arm = f"{geometry}_{type_arm}"
            eligible_cases = []
            values = []
            for case_id in cases:
                completion_row = completion_by_arm.get((case_id, geometry, type_arm))
                eligible = bool(
                    completion_row is not None
                    and completion_row["fully_admissible"]
                    and completion_row["horizon_output_produced"]
                )
                if not eligible:
                    continue
                eligible_cases.append(case_id)
                value = endpoint_rows.get((case_id, phase, arm))
                if value is not None:
                    values.append(value)
            result.append(
                {
                    "phase": phase,
                    "arm": arm,
                    "expected_case_count": len(cases),
                    "horizon_admissible_case_count": len(eligible_cases),
                    "endpoint_metric_case_count": len(values),
                    "missing_or_inadmissible_case_count": (
                        len(cases) - len(eligible_cases)
                    ),
                    "missing_endpoint_metric_count": (
                        len(eligible_cases) - len(values)
                    ),
                    "endpoint_claim_allowed": len(values) == len(cases),
                    "mean_horizon_state_relative_l2": (
                        None if not values else float(np.mean(values))
                    ),
                    "median_horizon_state_relative_l2": (
                        None if not values else float(np.median(values))
                    ),
                    "maximum_horizon_state_relative_l2": (
                        None if not values else float(np.max(values))
                    ),
                }
            )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    device = select_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint)
    config = checkpoint["model_config"]
    if (
        int(config["k_max"]) != 8
        or tuple(float(value) for value in config["domain_lengths"]) != (6.0, 2.0)
        or str(config.get("node_type_feature_mode", NODE_TYPE_FEATURE_ONE_HOT))
        != NODE_TYPE_FEATURE_ONE_HOT
        or str(config.get("boundary_field_mode", BOUNDARY_FIELD_NONE))
        != BOUNDARY_FIELD_NONE
        or str(config.get("boundary_residual_mode", BOUNDARY_RESIDUAL_NONE))
        != BOUNDARY_RESIDUAL_NONE
    ):
        raise ValueError(
            "checkpoint does not match the D041 categorical input contract"
        )
    model, _ = build_checkpoint_model(
        checkpoint, device, model_node_type_input="physical"
    )
    model.eval()

    with PCNOEuler2DShardStore(args.data_dir) as store:
        provenance = _verify_provenance(args, checkpoint, store)
        split = provenance["split"]
        validation = [str(value) for value in split.get("val_keys", [])]
        requested = validation if args.case_ids is None else list(args.case_ids)
        if not requested or any(case_id not in validation for case_id in requested):
            raise ValueError(
                "D085 cases must come only from the frozen validation split"
            )
        if args.smoke and requested != ["128"]:
            raise ValueError("D085 H2 smoke freezes the single replay case 128")
        if not args.smoke and requested != validation:
            raise ValueError(
                "scientific D085 requires all 30 validation cases in frozen split order"
            )
        replay_case = _load_case(
            store,
            checkpoint,
            args.replay_case,
            rollout_calls=max(args.rollout_calls, args.replay_calls),
        )
        replay = _replay_gate(args, model, replay_case, device=device)
        del replay_case
        if device.type == "cuda":
            torch.cuda.empty_cache()

        args.output_dir.mkdir(parents=True)
        (args.output_dir / "arrays").mkdir()
        metric_rows: list[dict[str, Any]] = []
        completion_rows: list[dict[str, Any]] = []
        closure_rows: list[dict[str, Any]] = []
        geometry_audits: list[dict[str, Any]] = []
        completion_summary: list[dict[str, Any]] = []
        bundle_paths: list[Path] = []
        for index, case_id in enumerate(requested, start=1):
            case = _load_case(
                store,
                checkpoint,
                case_id,
                rollout_calls=args.rollout_calls,
            )
            (
                case_metrics,
                case_completion,
                case_closures,
                case_summary,
                bundle_path,
            ) = _evaluate_case(
                case,
                model,
                args,
                args.output_dir,
                device=device,
            )
            metric_rows.extend(case_metrics)
            completion_rows.extend(case_completion)
            closure_rows.extend(case_closures)
            geometry_audits.append(case_summary["geometry_audit"])
            completion_summary.extend(case_summary["completion_summary"])
            if bundle_path is not None:
                bundle_paths.append(bundle_path)
            atomic_write_json(
                args.output_dir / "progress.json",
                {
                    "schema": SCHEMA,
                    "status": "running",
                    "completed_cases": index,
                    "total_cases": len(requested),
                    "last_case_id": case_id,
                },
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()

    terminal_project_sources = _loaded_project_source_hashes()
    if terminal_project_sources != provenance["source_sha256"]:
        raise RuntimeError(
            "loaded project-source inventory changed after the provenance gate"
        )
    _assert_unique_metric_rows(metric_rows)
    paths = {
        "metrics": args.output_dir / "metrics.csv",
        "completion": args.output_dir / "completion.csv",
        "closures": args.output_dir / "closures.csv",
        "geometry_audit": args.output_dir / "geometry_audit.json",
    }
    write_csv(paths["metrics"], metric_rows)
    write_csv(paths["completion"], completion_rows)
    write_csv(paths["closures"], closure_rows)
    atomic_write_json(
        paths["geometry_audit"],
        {"schema": SCHEMA, "cases": geometry_audits},
    )
    artifact_paths = [*paths.values(), *bundle_paths]
    artifact_hashes = {
        path.relative_to(args.output_dir).as_posix(): sha256_file(path)
        for path in artifact_paths
    }
    manifest_path = args.output_dir / "artifact_manifest.json"
    atomic_write_json(
        manifest_path,
        {
            "schema": "pcno_declared_artifact_manifest_v1",
            "files": artifact_hashes,
        },
    )
    matrix_coverage = _matrix_coverage(
        completion_summary,
        requested_phase=args.phase,
        cases=requested,
    )
    final_summary = _final_summary(
        metric_rows,
        completion_summary,
        requested_phase=args.phase,
        cases=requested,
        horizon=args.rollout_calls,
    )
    matrix_attempt_complete = bool(matrix_coverage["declared_matrix_attempt_complete"])
    matrix_horizon_complete = bool(matrix_coverage["declared_matrix_horizon_complete"])
    if args.smoke:
        status = "smoke_complete"
    elif matrix_attempt_complete:
        status = "complete"
    else:
        status = "complete_with_missing_attempts"
    summary = {
        "schema": SCHEMA,
        "status": status,
        "execution_complete": True,
        "scientific_interpretation_allowed": (
            not args.smoke and matrix_attempt_complete
        ),
        "declared_matrix_attempt_complete": matrix_attempt_complete,
        "declared_matrix_horizon_complete": matrix_horizon_complete,
        "args": jsonable_args(args),
        "phase": args.phase,
        "case_count": len(requested),
        "cases": requested,
        "rollout_calls": args.rollout_calls,
        "physical_policy": "model_all_nodes",
        "node_type_semantics": {
            str(key): value for key, value in BUMP_NODE_TYPE_NAMES.items()
        },
        "checkpoint_contract": {
            "k_max": 8,
            "physical_domain_lengths": [6.0, 2.0],
            "raw_recurrence": True,
            "step_stride": 1,
            "scientific_inference_batch_size": 1,
            "boundary_field_inputs": [],
            "normal_or_vector_boundary_inputs": [],
            "fourier_policy": FOURIER_POLICY,
            "fourier_phase_origin": list(FOURIER_PHASE_ORIGIN),
        },
        "replay_gate": replay,
        "provenance": provenance,
        "loaded_project_source_sha256": terminal_project_sources,
        "geometry_audits": geometry_audits,
        "completion_summary": completion_summary,
        "matrix_coverage": matrix_coverage,
        "final_summary": final_summary,
        "row_counts": {
            "metrics": len(metric_rows),
            "completion": len(completion_rows),
            "closures": len(closure_rows),
        },
        "artifact_manifest_sha256": sha256_file(manifest_path),
        "artifact_files": artifact_hashes,
        "git": git_state(),
        "runtime": runtime_environment(device),
        "claim_boundary": {
            "g1b": (
                "fixed-checkpoint-Fourier transformed-input test on analytically "
                "rotated retained cases; not transported-mode covariance, an "
                "independently solved rotated PDE, unseen-case generalization, broad "
                "geometry generalization, or architecture-level rotation equivariance"
            ),
            "boundary": (
                "operator-consistent representation under frozen model_all_nodes policy; "
                "not boundary-condition improvement"
            ),
            "population": "30-case open validation only; sealed and test populations untouched",
            "vortex": "no bump vortex-core mask is declared or inferred",
        },
    }
    summary_path = args.output_dir / "summary.json"
    atomic_write_json(summary_path, summary)
    receipt_path = args.output_dir / "terminal_receipt.json"
    atomic_write_json(
        receipt_path,
        {
            "schema": "pcno_d085_terminal_receipt_v1",
            "summary_sha256": sha256_file(summary_path),
            "artifact_manifest_sha256": sha256_file(manifest_path),
            "execution_status": status,
            "declared_matrix_attempt_complete": matrix_attempt_complete,
            "declared_matrix_horizon_complete": matrix_horizon_complete,
        },
    )
    atomic_write_json(
        args.output_dir / "progress.json",
        {
            "schema": SCHEMA,
            "status": summary["status"],
            "completed_cases": len(requested),
            "total_cases": len(requested),
        },
    )
    print(json.dumps(summary["final_summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
