"""Reusable pieces for conservative-residual PCNO Euler rollouts.

The legacy 2D Euler PCNO script predicts a positive primitive next state and
materializes every current/next pair.  This module provides the smaller set of
components needed by the time-dependent branch:

* sharded, full-resolution trajectory access;
* fixed training-set conservative and residual scales;
* a centered conservative-residual wrapper around :class:`pcno.pcno.PCNO`;
* admissible training-only input noise; and
* proxy-mass-weighted losses and deterministic presentation sampling.

The reconstructed vertex weights consumed here are PCNO quadrature proxies.
They are not asserted to be physical finite-volume cell measures.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from pcno.pcno import PCNO, compute_Fourier_modes, compute_gradient
from utility.time_dependent_no.cpg_mesh_contract import (
    INFLOW_NODE,
    NORMAL_NODE,
    OUTFLOW_NODE,
    WALL_NODE,
    apply_torch_boundary_policy,
    boundary_stencil_sha256,
    build_boundary_stencil,
    build_torch_boundary_policy,
    freestream_primitive,
    recover_graph_boundary_geometry,
)

SCHEMA_VERSION = 1
NUM_EULER_COMPONENTS = 4
NUM_NODE_TYPES = 4
DEFAULT_MAX_CACHED_GEOMETRY_BYTES = 256 * 1024 * 1024
NODE_TYPE_FEATURE_ONE_HOT = "one_hot"
NODE_TYPE_FEATURE_CONSTANT_ZERO = "constant_zero"
NODE_TYPE_FEATURE_OMITTED = "omitted"
NODE_TYPE_FEATURE_MODES = (
    NODE_TYPE_FEATURE_ONE_HOT,
    NODE_TYPE_FEATURE_CONSTANT_ZERO,
    NODE_TYPE_FEATURE_OMITTED,
)
BOUNDARY_FIELD_NONE = "none"
BOUNDARY_FIELD_GEOMETRY_COLLAR = "geometry_collar"
BOUNDARY_FIELD_SEMANTIC_COLLAR = "semantic_collar"
BOUNDARY_FIELD_MODES = (
    BOUNDARY_FIELD_NONE,
    BOUNDARY_FIELD_GEOMETRY_COLLAR,
    BOUNDARY_FIELD_SEMANTIC_COLLAR,
)
BOUNDARY_RESIDUAL_NONE = "none"
BOUNDARY_RESIDUAL_SEMANTIC_COLLAR_POINTWISE = "semantic_collar_gated_pointwise"
BOUNDARY_RESIDUAL_MODES = (
    BOUNDARY_RESIDUAL_NONE,
    BOUNDARY_RESIDUAL_SEMANTIC_COLLAR_POINTWISE,
)
TYPE_FEATURE_START = 2 + 1 + NUM_EULER_COMPONENTS
TYPE_FEATURE_STOP = TYPE_FEATURE_START + NUM_NODE_TYPES


def _component_array(value: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (NUM_EULER_COMPONENTS,):
        raise ValueError(f"{name} must have shape (4,), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


@dataclass(frozen=True)
class Euler2DNormalization:
    """Fixed training-set scales used by the residual model."""

    state_mean: np.ndarray
    state_scale: np.ndarray
    residual_scale: np.ndarray
    mach_mean: float
    mach_scale: float
    mach_scale_floor: float = 0.1
    gamma: float = 1.4
    weight_provenance: str = "reconstructed_vertex_lumped_proxy"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "state_mean", _component_array(self.state_mean, "state_mean")
        )
        object.__setattr__(
            self, "state_scale", _component_array(self.state_scale, "state_scale")
        )
        object.__setattr__(
            self,
            "residual_scale",
            _component_array(self.residual_scale, "residual_scale"),
        )
        if np.any(self.state_scale <= 0.0) or np.any(self.residual_scale <= 0.0):
            raise ValueError("state and residual scales must be positive")
        if (
            not np.isfinite(self.mach_mean)
            or not np.isfinite(self.mach_scale)
            or not np.isfinite(self.mach_scale_floor)
        ):
            raise ValueError("Mach statistics must be finite")
        if self.mach_scale_floor <= 0.0 or self.mach_scale < self.mach_scale_floor:
            raise ValueError("mach_scale must respect the positive declared floor")
        if self.gamma <= 1.0:
            raise ValueError("gamma must be greater than one")

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_mean": self.state_mean.tolist(),
            "state_scale": self.state_scale.tolist(),
            "residual_scale": self.residual_scale.tolist(),
            "mach_mean": float(self.mach_mean),
            "mach_scale": float(self.mach_scale),
            "mach_scale_floor": float(self.mach_scale_floor),
            "gamma": float(self.gamma),
            "weight_provenance": self.weight_provenance,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "Euler2DNormalization":
        return cls(
            state_mean=np.asarray(value["state_mean"], dtype=np.float64),
            state_scale=np.asarray(value["state_scale"], dtype=np.float64),
            residual_scale=np.asarray(value["residual_scale"], dtype=np.float64),
            mach_mean=float(value["mach_mean"]),
            mach_scale=float(value["mach_scale"]),
            mach_scale_floor=float(value.get("mach_scale_floor", value["mach_scale"])),
            gamma=float(value.get("gamma", 1.4)),
            weight_provenance=str(
                value.get("weight_provenance", "reconstructed_vertex_lumped_proxy")
            ),
        )


def primitive_to_conservative_torch(
    primitive: torch.Tensor,
    *,
    gamma: float = 1.4,
) -> torch.Tensor:
    """Convert ``[rho, v1, v2, pressure]`` to conservative variables."""

    if primitive.shape[-1] != NUM_EULER_COMPONENTS:
        raise ValueError("primitive tensor must have four components")
    if gamma <= 1.0:
        raise ValueError("gamma must be greater than one")
    rho, v1, v2, pressure = primitive.unbind(dim=-1)
    energy = pressure / (gamma - 1.0) + 0.5 * rho * (v1.square() + v2.square())
    return torch.stack((rho, rho * v1, rho * v2, energy), dim=-1)


def conservative_to_primitive_torch(
    conservative: torch.Tensor,
    *,
    gamma: float = 1.4,
) -> torch.Tensor:
    """Convert conservative variables without hiding inadmissible states."""

    if conservative.shape[-1] != NUM_EULER_COMPONENTS:
        raise ValueError("conservative tensor must have four components")
    if gamma <= 1.0:
        raise ValueError("gamma must be greater than one")
    rho, momentum_x, momentum_y, energy = conservative.unbind(dim=-1)
    v1 = momentum_x / rho
    v2 = momentum_y / rho
    pressure = (gamma - 1.0) * (
        energy - 0.5 * (momentum_x.square() + momentum_y.square()) / rho
    )
    return torch.stack((rho, v1, v2, pressure), dim=-1)


def conservative_admissibility(
    conservative: torch.Tensor,
    *,
    gamma: float = 1.4,
) -> dict[str, torch.Tensor]:
    """Return raw finiteness and Euler admissibility diagnostics."""

    rho = conservative[..., 0]
    momentum_sq = conservative[..., 1].square() + conservative[..., 2].square()
    internal_energy = conservative[..., 3] - 0.5 * momentum_sq / rho
    pressure = (gamma - 1.0) * internal_energy
    finite_components = torch.isfinite(conservative).all(dim=-1)
    return {
        "finite_components": finite_components,
        "density": rho,
        "internal_energy": internal_energy,
        "pressure": pressure,
        "admissible": (
            finite_components
            & torch.isfinite(internal_energy)
            & torch.isfinite(pressure)
            & (rho > 0.0)
            & (internal_energy > 0.0)
            & (pressure > 0.0)
        ),
    }


def build_graph_causal_boundary_policy(
    store: "PCNOEuler2DShardStore",
    key: str,
    *,
    device: torch.device,
    max_source_hops: int = 3,
    rho_inf: float = 1.4,
    p_inf: float = 1.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the legal graph-native nodal closure used by serious bump training.

    The closure is causal and deterministic but is not an exact replay of the
    source DG boundary flux. A fallback stencil is rejected because it lacks an
    inward current-interior source under the registered graph geometry.
    """

    if max_source_hops < 1:
        raise ValueError("max_source_hops must be positive")
    if not math.isfinite(rho_inf) or not math.isfinite(p_inf):
        raise ValueError("freestream density and pressure must be finite")
    if rho_inf <= 0.0 or p_inf <= 0.0:
        raise ValueError("freestream density and pressure must be positive")
    positions = np.array(store.array(key, "nodes"), copy=True)
    edges = np.array(store.array(key, "edges"), copy=True)
    node_type = np.array(store.array(key, "node_type"), copy=True).reshape(-1)
    geometry = recover_graph_boundary_geometry(
        pos=positions,
        edges=edges,
        node_type=node_type,
    )
    stencil = build_boundary_stencil(
        pos=positions,
        edges=edges,
        node_type=node_type,
        node_normal=geometry["node_normal"],
        max_source_hops=max_source_hops,
    )
    if stencil.fallback_target_count:
        raise ValueError(f"trajectory {key} boundary stencil used a fallback")
    gamma = float(store.manifest.get("gamma", 1.4))
    mach = float(store.entry(key)["mach"])
    config = {"gamma": gamma, "rho_inf": float(rho_inf), "p_inf": float(p_inf)}
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
    policy["closure_kind"] = "causal_interior_reconstruction"
    counts = {
        "normal": int(np.count_nonzero(node_type == NORMAL_NODE)),
        "wall": int(np.count_nonzero(node_type == WALL_NODE)),
        "outflow": int(np.count_nonzero(node_type == OUTFLOW_NODE)),
        "inflow": int(np.count_nonzero(node_type == INFLOW_NODE)),
    }
    if any(counts[name] < 1 for name in ("normal", "wall", "outflow", "inflow")):
        raise ValueError(
            f"trajectory {key} lacks a required interior or boundary node type"
        )
    if int(stencil.target_nodes.size) != counts["wall"] + counts["outflow"]:
        raise RuntimeError(f"trajectory {key} has an incomplete boundary stencil")
    metadata = {
        "schema": "pcno_graph_causal_nodal_boundary_v1",
        "trajectory_key": str(key),
        "geometry_digest": store.entry(key).get("geometry_digest"),
        "num_nodes": int(node_type.size),
        "node_type_counts": counts,
        "max_source_hops": int(max_source_hops),
        "target_count": int(stencil.target_nodes.size),
        "entry_count": int(stencil.source_nodes.size),
        "fallback_target_count": int(stencil.fallback_target_count),
        "sharp_wall_corner_count": int(policy["sharp_wall_rows"].numel()),
        "boundary_stencil_sha256": boundary_stencil_sha256(stencil),
        "config": config,
        "mach": mach,
        "policy": (
            "fixed freestream inflow, current-interior slip wall, and "
            "current-interior supersonic outflow"
        ),
        "exact_dg_boundary_replay": False,
        "future_reference_boundary_values": False,
    }
    metadata["policy_digest"] = digest_mapping(metadata)
    return policy, metadata


def apply_causal_boundary_conservative_batch(
    state: torch.Tensor,
    policy: Mapping[str, Any],
    *,
    gamma: float,
) -> torch.Tensor:
    """Apply one graph policy independently to a homogeneous state batch."""

    if state.ndim != 3 or state.shape[1:] != (int(policy["num_nodes"]), 4):
        raise ValueError("state must have shape [B,num_nodes,4]")
    primitive = conservative_to_primitive_torch(state, gamma=gamma)
    closed = torch.stack(
        [apply_torch_boundary_policy(torch, sample, policy) for sample in primitive],
        dim=0,
    )
    encoded = primitive_to_conservative_torch(closed, gamma=gamma)
    boundary_nodes = torch.cat((policy["target_nodes"], policy["inflow_nodes"]), dim=0)
    output = state.clone()
    output[:, boundary_nodes] = encoded[:, boundary_nodes]
    return output


def _minimum_change_wall_projectors(
    geometry: Mapping[str, np.ndarray],
    *,
    sharp_corner_coherence: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return nodal velocity projectors for incident slip-wall constraints."""

    if not 0.0 < sharp_corner_coherence < 1.0:
        raise ValueError("sharp_corner_coherence must lie in (0,1)")
    node_type = np.asarray(geometry["node_type"]).reshape(-1)
    boundary_edges = np.asarray(geometry["boundary_edges"], dtype=np.int64)
    edge_normals = np.asarray(geometry["boundary_edge_normal"], dtype=np.float64)
    edge_lengths = np.asarray(geometry["boundary_edge_length"], dtype=np.float64)
    edge_types = np.asarray(
        geometry["boundary_edge_node_type"], dtype=np.int64
    ).reshape(-1)
    if (
        boundary_edges.shape != (edge_types.size, 2)
        or edge_normals.shape != (edge_types.size, 2)
        or edge_lengths.shape != (edge_types.size,)
    ):
        raise ValueError("recovered boundary edge geometry is inconsistent")

    incident: list[list[tuple[np.ndarray, float]]] = [
        [] for _ in range(node_type.size)
    ]
    for edge, normal, length, edge_type in zip(
        boundary_edges, edge_normals, edge_lengths, edge_types, strict=True
    ):
        if int(edge_type) != WALL_NODE:
            continue
        for node in edge:
            incident[int(node)].append((normal, float(length)))

    projectors = np.broadcast_to(
        np.eye(2, dtype=np.float64), (node_type.size, 2, 2)
    ).copy()
    constraint_rank = np.zeros(node_type.size, dtype=np.int64)
    coherence = np.ones(node_type.size, dtype=np.float64)
    for node, faces in enumerate(incident):
        if not faces:
            continue
        normals = np.stack([normal for normal, _ in faces], axis=0)
        lengths = np.asarray([length for _, length in faces], dtype=np.float64)
        normal_sum = np.sum(normals * lengths[:, None], axis=0)
        normal_sum_norm = float(np.linalg.norm(normal_sum))
        total_length = float(np.sum(lengths))
        if (
            not np.isfinite(normal_sum_norm)
            or not np.isfinite(total_length)
            or normal_sum_norm <= 0.0
            or total_length <= 0.0
        ):
            raise ValueError(f"wall constraints at node {node} have no mean normal")
        coherence[node] = normal_sum_norm / total_length
        mean_normal = normal_sum / normal_sum_norm

        constraints = mean_normal.reshape(1, 2)
        if (
            len(faces) > 1
            and coherence[node] < sharp_corner_coherence
            and np.linalg.matrix_rank(normals, tol=1.0e-8) > 1
        ):
            constraints = normals
        gram = constraints @ constraints.T
        projector = np.eye(2) - constraints.T @ np.linalg.pinv(
            gram, rcond=1.0e-10
        ) @ constraints
        projector = 0.5 * (projector + projector.T)
        projector[np.abs(projector) < 1.0e-14] = 0.0
        projectors[node] = projector
        constraint_rank[node] = int(
            np.linalg.matrix_rank(constraints, tol=1.0e-8)
        )
    return projectors, constraint_rank, coherence


def build_graph_minimum_change_boundary_policy(
    store: PCNOEuler2DShardStore,
    key: str,
    *,
    device: torch.device,
    rho_inf: float = 1.4,
    p_inf: float = 1.0,
    sharp_corner_coherence: float = 0.95,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a graph-native minimum-change primitive boundary projection.

    Inflow is fixed to the freestream. Slip-wall nodes retain density, pressure,
    and legal tangential velocity. Outflow is left unchanged when it remains
    outward-supersonic; callers must treat loss of that condition as a rollout
    failure because the released nodal data contain no exterior characteristic
    state. Smooth curved walls use their length-weighted mean normal, while
    genuinely sharp wall corners enforce every independent incident normal.

    The projection uses the Euclidean primitive-variable metric. It is a legal
    nodal counterfactual, not a DG face/flux replay or a conservation claim.
    """

    if not math.isfinite(rho_inf) or not math.isfinite(p_inf):
        raise ValueError("freestream density and pressure must be finite")
    if rho_inf <= 0.0 or p_inf <= 0.0:
        raise ValueError("freestream density and pressure must be positive")
    positions = np.array(store.array(key, "nodes"), copy=True)
    edges = np.array(store.array(key, "edges"), copy=True)
    node_type = np.array(store.array(key, "node_type"), copy=True).reshape(-1)
    geometry = recover_graph_boundary_geometry(
        pos=positions,
        edges=edges,
        node_type=node_type,
    )
    projectors, wall_rank, wall_coherence = _minimum_change_wall_projectors(
        geometry,
        sharp_corner_coherence=sharp_corner_coherence,
    )
    gamma = float(store.manifest.get("gamma", 1.4))
    mach = float(store.entry(key)["mach"])
    stream = freestream_primitive(
        mach,
        gamma=gamma,
        rho_inf=float(rho_inf),
        p_inf=float(p_inf),
    )
    inflow_nodes = np.flatnonzero(node_type == INFLOW_NODE)
    wall_constrained_nodes = np.flatnonzero(wall_rank > 0)
    incompatible_inflow = [
        int(node)
        for node in inflow_nodes
        if wall_rank[node] > 0
        and np.linalg.norm(
            (np.eye(2) - projectors[node]) @ np.asarray(stream[1:3])
        )
        > 1.0e-8
    ]
    if incompatible_inflow:
        raise ValueError(
            "full freestream inflow conflicts with incident slip-wall constraints "
            f"at nodes {incompatible_inflow[:8]}"
        )

    boundary_nodes = np.flatnonzero(node_type != NORMAL_NODE)
    target_nodes = np.flatnonzero(
        (node_type == WALL_NODE) | (node_type == OUTFLOW_NODE)
    )
    target_type = node_type[target_nodes]
    wall_rows = np.flatnonzero(target_type == WALL_NODE)
    outflow_rows = np.flatnonzero(target_type == OUTFLOW_NODE)
    sharp_wall_rows = np.flatnonzero(wall_rank[target_nodes] > 1)
    counts = {
        "normal": int(np.count_nonzero(node_type == NORMAL_NODE)),
        "wall": int(np.count_nonzero(node_type == WALL_NODE)),
        "outflow": int(np.count_nonzero(node_type == OUTFLOW_NODE)),
        "inflow": int(np.count_nonzero(node_type == INFLOW_NODE)),
    }
    if any(counts[name] < 1 for name in ("normal", "wall", "outflow", "inflow")):
        raise ValueError(
            f"trajectory {key} lacks a required interior or boundary node type"
        )

    policy = {
        "closure_kind": "minimum_change_primitive_projection",
        "num_nodes": int(node_type.size),
        "node_type": torch.as_tensor(node_type, dtype=torch.long, device=device),
        "boundary_nodes": torch.as_tensor(
            boundary_nodes, dtype=torch.long, device=device
        ),
        "target_nodes": torch.as_tensor(
            target_nodes, dtype=torch.long, device=device
        ),
        "target_normals": torch.as_tensor(
            geometry["node_normal"][target_nodes],
            dtype=torch.float32,
            device=device,
        ),
        "wall_rows": torch.as_tensor(wall_rows, dtype=torch.long, device=device),
        "outflow_rows": torch.as_tensor(
            outflow_rows, dtype=torch.long, device=device
        ),
        "sharp_wall_rows": torch.as_tensor(
            sharp_wall_rows, dtype=torch.long, device=device
        ),
        "wall_constrained_nodes": torch.as_tensor(
            wall_constrained_nodes, dtype=torch.long, device=device
        ),
        "wall_velocity_projectors": torch.as_tensor(
            projectors[wall_constrained_nodes],
            dtype=torch.float32,
            device=device,
        ),
        "wall_constraint_rank": torch.as_tensor(
            wall_rank[wall_constrained_nodes],
            dtype=torch.long,
            device=device,
        ),
        "inflow_nodes": torch.as_tensor(
            inflow_nodes, dtype=torch.long, device=device
        ),
        "freestream": torch.as_tensor(stream, dtype=torch.float32, device=device),
        "gamma": gamma,
    }
    metadata = {
        "schema": "pcno_graph_minimum_change_boundary_v1",
        "trajectory_key": str(key),
        "geometry_digest": store.entry(key).get("geometry_digest"),
        "num_nodes": int(node_type.size),
        "node_type_counts": counts,
        "wall_constrained_node_count": int(wall_constrained_nodes.size),
        "fallback_target_count": 0,
        "uses_interior_stencil": False,
        "rank_two_wall_corner_count": int(np.count_nonzero(wall_rank > 1)),
        "incident_wall_junction_count": int(
            np.count_nonzero(
                (wall_rank > 0)
                & ((node_type == INFLOW_NODE) | (node_type == OUTFLOW_NODE))
            )
        ),
        "minimum_wall_normal_coherence": float(
            wall_coherence[wall_constrained_nodes].min()
        ),
        "sharp_corner_coherence": float(sharp_corner_coherence),
        "config": {
            "gamma": gamma,
            "rho_inf": float(rho_inf),
            "p_inf": float(p_inf),
        },
        "mach": mach,
        "projection_metric": "euclidean_primitive_variables",
        "policy": (
            "fixed freestream inflow; minimum-change slip-wall velocity; "
            "unchanged outward-supersonic outflow"
        ),
        "outflow_characteristic_treatment": (
            "preserve all outgoing modes while outward normal Mach exceeds one; "
            "otherwise stop because no exterior incoming characteristic is retained"
        ),
        "exact_dg_boundary_replay": False,
        "physical_conservation_claim": False,
        "future_reference_boundary_values": False,
    }
    metadata["policy_digest"] = digest_mapping(metadata)
    return policy, metadata


def apply_minimum_change_boundary_conservative_batch(
    state: torch.Tensor,
    policy: Mapping[str, Any],
    *,
    gamma: float,
) -> torch.Tensor:
    """Apply the minimum-change boundary projection to a homogeneous batch."""

    if state.ndim != 3 or state.shape[1:] != (int(policy["num_nodes"]), 4):
        raise ValueError("state must have shape [B,num_nodes,4]")
    primitive = conservative_to_primitive_torch(state, gamma=gamma)
    closed = primitive.clone()
    wall_nodes = policy["wall_constrained_nodes"]
    if wall_nodes.numel():
        projectors = policy["wall_velocity_projectors"].to(
            dtype=closed.dtype, device=closed.device
        )
        with torch.autocast(device_type=closed.device.type, enabled=False):
            projected_velocity = torch.einsum(
                "nij,bnj->bni",
                projectors,
                closed[:, wall_nodes, 1:3],
            )
        closed[:, wall_nodes, 1:3] = projected_velocity
    inflow_nodes = policy["inflow_nodes"]
    stream = policy["freestream"].to(dtype=closed.dtype, device=closed.device)
    closed[:, inflow_nodes] = stream
    encoded = primitive_to_conservative_torch(closed, gamma=gamma)
    boundary_nodes = policy["boundary_nodes"]
    output = state.clone()
    output[:, boundary_nodes] = encoded[:, boundary_nodes]
    return output


def normal_node_mask(
    node_type: torch.Tensor,
    node_mask: torch.Tensor,
) -> torch.Tensor:
    """Return the padding-aware mask for CPG normal/interior graph nodes."""

    if node_type.ndim == 3 and node_type.shape[-1] == 1:
        node_type = node_type[..., 0]
    if node_type.shape != node_mask.shape[:2]:
        raise ValueError("node_type and node_mask must share batch/node axes")
    return node_mask * (node_type == NORMAL_NODE).to(dtype=node_mask.dtype).unsqueeze(
        -1
    )


def boundary_band_normal_node_mask(
    node_type: torch.Tensor,
    node_mask: torch.Tensor,
    directed_edges: torch.Tensor,
    *,
    max_hops: int,
) -> torch.Tensor:
    """Select normal nodes within ``max_hops`` graph edges of a boundary."""

    if max_hops < 1:
        raise ValueError("max_hops must be positive")
    if node_mask.ndim != 3 or node_mask.shape[-1] != 1:
        raise ValueError("node_mask must have shape [B,N,1]")
    batch_size, num_nodes, _ = node_mask.shape
    if node_type.ndim == 3 and node_type.shape[-1] == 1:
        node_type = node_type[..., 0]
    if node_type.shape != (batch_size, num_nodes):
        raise ValueError("node_type and node_mask must share batch/node axes")
    if (
        directed_edges.ndim != 3
        or directed_edges.shape[0] != batch_size
        or directed_edges.shape[-1] != 2
    ):
        raise ValueError("directed_edges must have shape [B,E,2]")

    valid = node_mask[..., 0].to(dtype=torch.bool)
    normal = valid & (node_type == NORMAL_NODE)
    boundary = valid & ~normal
    if not bool(boundary.any(dim=1).all()):
        raise ValueError("every sample must contain at least one boundary node")

    edges = directed_edges.to(dtype=torch.long)
    if edges.numel() and (bool((edges < 0).any()) or bool((edges >= num_nodes).any())):
        raise ValueError("directed edge index lies outside the node axis")
    target = edges[..., 0]
    source = edges[..., 1]
    batch = torch.arange(batch_size, device=node_mask.device).unsqueeze(1)
    valid_edges = valid[batch, target] & valid[batch, source]

    reached = boundary.clone()
    frontier = boundary
    for _ in range(max_hops):
        neighbor_count = torch.zeros(
            (batch_size, num_nodes), dtype=torch.int64, device=node_mask.device
        )
        neighbor_count.scatter_add_(
            1,
            target,
            (frontier[batch, source] & valid_edges).to(dtype=torch.int64),
        )
        frontier = (neighbor_count > 0) & ~reached
        reached |= frontier

    band = normal & reached
    if not bool(band.any(dim=1).all()):
        raise ValueError("boundary band does not contain a normal node in every sample")
    return band.to(dtype=node_mask.dtype).unsqueeze(-1)


def apply_admissible_primitive_noise(
    current_conservative: torch.Tensor,
    noise_std: float,
    *,
    gamma: float = 1.4,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Perturb the current state while preserving positive density and pressure.

    Density and pressure receive log-normal noise; both velocity components
    receive additive Gaussian noise.  The clean next state remains the target.
    """

    if noise_std < 0.0:
        raise ValueError("noise_std must be nonnegative")
    if noise_std == 0.0:
        return current_conservative
    primitive = conservative_to_primitive_torch(current_conservative, gamma=gamma)
    diagnostics = conservative_admissibility(current_conservative, gamma=gamma)
    if not bool(diagnostics["admissible"].all()):
        raise ValueError("training noise requires an admissible current state")
    noise = torch.randn(
        primitive.shape,
        dtype=primitive.dtype,
        device=primitive.device,
        generator=generator,
    ) * float(noise_std)
    noisy = torch.empty_like(primitive)
    noisy[..., 0] = primitive[..., 0] * torch.exp(noise[..., 0])
    noisy[..., 1] = primitive[..., 1] + noise[..., 1]
    noisy[..., 2] = primitive[..., 2] + noise[..., 2]
    noisy[..., 3] = primitive[..., 3] * torch.exp(noise[..., 3])
    return primitive_to_conservative_torch(noisy, gamma=gamma)


def proxy_mass_weights(
    node_weights: torch.Tensor, node_mask: torch.Tensor
) -> torch.Tensor:
    """Collapse PCNO measures to one nonnegative per-node proxy weight."""

    if node_weights.ndim != 3:
        raise ValueError("node_weights must have shape [B, N, M]")
    if node_mask.shape != node_weights.shape[:2] + (1,):
        raise ValueError("node_mask must have shape [B, N, 1]")
    weights = node_weights.sum(dim=-1, keepdim=True) * node_mask
    if bool((weights < 0.0).any()):
        raise ValueError("node weights must be nonnegative")
    if bool((weights.sum(dim=1) <= 0.0).any()):
        raise ValueError("every sample must have positive total node weight")
    return weights


def weighted_scaled_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    node_weights: torch.Tensor,
    node_mask: torch.Tensor,
    component_scale: torch.Tensor,
) -> torch.Tensor:
    """Proxy-mass-weighted MSE in fixed component coordinates."""

    if prediction.shape != target.shape or prediction.shape[-1] != NUM_EULER_COMPONENTS:
        raise ValueError("prediction and target must share shape [B, N, 4]")
    scale = component_scale.reshape(1, 1, NUM_EULER_COMPONENTS).to(
        dtype=prediction.dtype,
        device=prediction.device,
    )
    weights = proxy_mass_weights(node_weights, node_mask)
    squared = ((prediction - target) / scale).square()
    denominator = weights.sum() * prediction.shape[-1]
    return (weights * squared).sum() / denominator


def weighted_scaled_relative_l2(
    prediction: torch.Tensor,
    target: torch.Tensor,
    node_weights: torch.Tensor,
    node_mask: torch.Tensor,
    component_scale: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Relative L2 after fixed component scaling and proxy mass weighting."""

    scale = component_scale.reshape(1, 1, NUM_EULER_COMPONENTS).to(
        dtype=prediction.dtype,
        device=prediction.device,
    )
    weights = proxy_mass_weights(node_weights, node_mask)
    error_energy = (weights * ((prediction - target) / scale).square()).sum()
    target_energy = (weights * (target / scale).square()).sum()
    return torch.sqrt(error_energy / target_energy.clamp_min(eps))


def _boolean_node_mask(
    node_mask: torch.Tensor,
    *,
    batch_size: int,
    num_nodes: int,
) -> torch.Tensor:
    if node_mask.shape == (batch_size, num_nodes, 1):
        node_mask = node_mask[..., 0]
    if node_mask.shape != (batch_size, num_nodes):
        raise ValueError("node mask must have shape [B,N] or [B,N,1]")
    return node_mask.to(dtype=torch.bool)


def graph_neighbor_highpass(
    field: torch.Tensor,
    directed_edges: torch.Tensor,
    node_mask: torch.Tensor,
) -> torch.Tensor:
    """Subtract a self-plus-neighbor graph average from a node field.

    This differentiable D013 counterpart constructs a residual for a
    diagnostic or loss. It never smooths the model state itself.
    """

    if field.ndim != 3:
        raise ValueError("field must have shape [B,N,C]")
    batch_size, num_nodes, channels = field.shape
    if (
        directed_edges.ndim != 3
        or directed_edges.shape[0] != batch_size
        or directed_edges.shape[-1] != 2
    ):
        raise ValueError("directed_edges must have shape [B,E,2]")
    valid_nodes = _boolean_node_mask(
        node_mask, batch_size=batch_size, num_nodes=num_nodes
    )
    edges = directed_edges.to(dtype=torch.long)
    if edges.numel() and (bool((edges < 0).any()) or bool((edges >= num_nodes).any())):
        raise ValueError("directed edge index lies outside the node axis")

    target = edges[..., 0]
    source = edges[..., 1]
    batch = torch.arange(batch_size, device=field.device).unsqueeze(1)
    valid_edges = valid_nodes[batch, target] & valid_nodes[batch, source]
    source_values = field[batch, source] * valid_edges.unsqueeze(-1)
    aggregate = (field * valid_nodes.unsqueeze(-1)).scatter_add(
        1,
        target.unsqueeze(-1).expand(-1, -1, channels),
        source_values,
    )
    degree = valid_nodes.to(dtype=field.dtype).scatter_add(
        1, target, valid_edges.to(dtype=field.dtype)
    )
    average = aggregate / degree.clamp_min(1.0).unsqueeze(-1)
    return torch.where(
        valid_nodes.unsqueeze(-1), field - average, torch.zeros_like(field)
    )


@torch.no_grad()
def reference_smooth_region_mask(
    reference_conservative: torch.Tensor,
    directed_edges: torch.Tensor,
    node_mask: torch.Tensor,
    *,
    interior_mask: torch.Tensor | None = None,
    shock_quantile: float = 0.9,
    dilation_hops: int = 2,
    gamma: float = 1.4,
) -> torch.Tensor:
    """Build a reference-only smooth interior mask from pressure jumps.

    This is a training diagnostic contract, not an inference-time shock
    detector. Graph dilation excludes a fixed neighborhood around the largest
    pressure jumps in each reference sample.
    """

    if (
        reference_conservative.ndim != 3
        or reference_conservative.shape[-1] != NUM_EULER_COMPONENTS
    ):
        raise ValueError("reference state must have shape [B,N,4]")
    if not 0.0 < shock_quantile < 1.0:
        raise ValueError("shock_quantile must lie in (0,1)")
    if dilation_hops < 0:
        raise ValueError("dilation_hops must be nonnegative")
    batch_size, num_nodes, _ = reference_conservative.shape
    valid_nodes = _boolean_node_mask(
        node_mask, batch_size=batch_size, num_nodes=num_nodes
    )
    if interior_mask is None:
        interior = valid_nodes
    else:
        interior = _boolean_node_mask(
            interior_mask, batch_size=batch_size, num_nodes=num_nodes
        )
        interior &= valid_nodes
        if not bool(interior.any(dim=1).all()):
            raise ValueError("interior_mask must select a node in every sample")

    edges = directed_edges.to(dtype=torch.long)
    if edges.ndim != 3 or edges.shape[0] != batch_size or edges.shape[-1] != 2:
        raise ValueError("directed_edges must have shape [B,E,2]")
    if edges.numel() and (bool((edges < 0).any()) or bool((edges >= num_nodes).any())):
        raise ValueError("directed edge index lies outside the node axis")
    target = edges[..., 0]
    source = edges[..., 1]
    batch = torch.arange(batch_size, device=reference_conservative.device).unsqueeze(1)
    valid_edges = valid_nodes[batch, target] & valid_nodes[batch, source]

    pressure = conservative_to_primitive_torch(reference_conservative, gamma=gamma)[
        ..., 3
    ]
    jumps = (pressure[batch, target] - pressure[batch, source]).abs()
    jumps = torch.where(valid_edges, jumps, torch.zeros_like(jumps))
    scores = torch.zeros(
        (batch_size, num_nodes), dtype=pressure.dtype, device=pressure.device
    )
    scores.scatter_reduce_(1, target, jumps, reduce="amax", include_self=True)

    shock = torch.zeros_like(valid_nodes)
    for batch_index in range(batch_size):
        selected_scores = scores[batch_index, interior[batch_index]]
        if selected_scores.numel() == 0:
            raise ValueError("interior_mask must select a node in every sample")
        threshold = torch.quantile(selected_scores, shock_quantile)
        shock[batch_index] = (
            (scores[batch_index] >= threshold)
            & (scores[batch_index] > 0.0)
            & interior[batch_index]
        )

    for _ in range(dilation_hops):
        neighbor_shock = torch.zeros(
            (batch_size, num_nodes), dtype=torch.int64, device=shock.device
        )
        neighbor_shock.scatter_add_(
            1,
            target,
            (shock[batch, source] & valid_edges).to(dtype=torch.int64),
        )
        shock |= neighbor_shock > 0
    return (interior & ~shock).unsqueeze(-1)


def directional_highpass_stability(
    reference_prediction: torch.Tensor,
    perturbed_prediction: torch.Tensor,
    input_perturbation: torch.Tensor,
    *,
    directed_edges: torch.Tensor,
    node_weights: torch.Tensor,
    node_mask: torch.Tensor,
    smooth_mask: torch.Tensor,
    component_scale: torch.Tensor,
    gain_cap: float = 1.0,
    eps: float = 1e-12,
) -> dict[str, torch.Tensor]:
    """Measure and penalize smooth-region graph-high-pass directional gain.

    The response is the difference of two full learned macro-map calls. This
    function neither edits predictions nor applies an inference-time filter.
    Bump weights retain their declared diagnostic-proxy interpretation.
    """

    expected_shape = reference_prediction.shape
    if (
        reference_prediction.ndim != 3
        or reference_prediction.shape[-1] != NUM_EULER_COMPONENTS
        or perturbed_prediction.shape != expected_shape
        or input_perturbation.shape != expected_shape
    ):
        raise ValueError("paired predictions and perturbation must share [B,N,4]")
    if not np.isfinite(gain_cap) or gain_cap <= 0.0:
        raise ValueError("gain_cap must be positive and finite")
    if not np.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be positive and finite")

    scale = component_scale.reshape(1, 1, NUM_EULER_COMPONENTS).to(
        dtype=reference_prediction.dtype, device=reference_prediction.device
    )
    response = (perturbed_prediction - reference_prediction) / scale
    normalized_input = input_perturbation / scale
    response_highpass = graph_neighbor_highpass(response, directed_edges, node_mask)
    input_highpass = graph_neighbor_highpass(
        normalized_input, directed_edges, node_mask
    )

    weights = proxy_mass_weights(node_weights, node_mask)
    smooth = _boolean_node_mask(
        smooth_mask,
        batch_size=expected_shape[0],
        num_nodes=expected_shape[1],
    ).unsqueeze(-1)
    smooth_weights = weights * smooth
    if not bool((smooth_weights.sum(dim=1) > 0.0).all()):
        raise ValueError("smooth_mask must retain positive proxy mass per sample")

    response_energy = (smooth_weights * response_highpass.square()).sum()
    input_energy = (smooth_weights * input_highpass.square()).sum()
    gain = torch.sqrt(response_energy / input_energy.clamp_min(float(eps)))
    loss = torch.relu(gain - float(gain_cap)).square()
    return {
        "loss": loss,
        "gain": gain,
        "response_energy": response_energy,
        "input_energy": input_energy,
        "smooth_proxy_mass_fraction": smooth_weights.sum() / weights.sum(),
    }


class PCNOEuler2DResidual(nn.Module):
    """Centered conservative-residual PCNO with explicit geometry semantics."""

    def __init__(
        self,
        *,
        normalization: Euler2DNormalization,
        k_max: int = 8,
        domain_lengths: Sequence[float] = (6.0, 2.0),
        layers: Sequence[int] = (128, 128, 128, 128, 128),
        fc_dim: int = 128,
        nmeasures: int = 1,
        act: str = "gelu",
        zero_initialize: bool = True,
        node_type_feature_mode: str = NODE_TYPE_FEATURE_ONE_HOT,
        boundary_field_mode: str = BOUNDARY_FIELD_NONE,
        boundary_field_names: Sequence[str] = (),
        boundary_residual_mode: str = BOUNDARY_RESIDUAL_NONE,
        boundary_residual_names: Sequence[str] = (),
        boundary_residual_width: int = 64,
    ) -> None:
        super().__init__()
        if k_max < 1:
            raise ValueError("k_max must be positive")
        if len(domain_lengths) != 2 or any(
            float(length) <= 0.0 for length in domain_lengths
        ):
            raise ValueError("domain_lengths must contain two positive values")
        if len(layers) < 2 or any(int(width) < 1 for width in layers):
            raise ValueError("layers must contain at least two positive widths")
        if nmeasures != 1:
            raise ValueError(
                "the current Euler artifact contract uses exactly one measure"
            )
        if node_type_feature_mode not in NODE_TYPE_FEATURE_MODES:
            raise ValueError(
                "node_type_feature_mode must be one of "
                f"{NODE_TYPE_FEATURE_MODES}, got {node_type_feature_mode!r}"
            )
        if boundary_field_mode not in BOUNDARY_FIELD_MODES:
            raise ValueError(
                f"boundary_field_mode must be one of {BOUNDARY_FIELD_MODES}, "
                f"got {boundary_field_mode!r}"
            )
        if boundary_residual_mode not in BOUNDARY_RESIDUAL_MODES:
            raise ValueError(
                f"boundary_residual_mode must be one of {BOUNDARY_RESIDUAL_MODES}, "
                f"got {boundary_residual_mode!r}"
            )
        if int(boundary_residual_width) < 1:
            raise ValueError("boundary_residual_width must be positive")
        field_names = tuple(str(name) for name in boundary_field_names)
        if any(not name for name in field_names) or len(set(field_names)) != len(
            field_names
        ):
            raise ValueError("boundary_field_names must be nonempty and unique")
        if boundary_field_mode == BOUNDARY_FIELD_NONE and field_names:
            raise ValueError("the none boundary-field mode cannot declare field names")
        if boundary_field_mode != BOUNDARY_FIELD_NONE and not field_names:
            raise ValueError("an active boundary-field mode requires field names")
        if (
            boundary_field_mode != BOUNDARY_FIELD_NONE
            and node_type_feature_mode != NODE_TYPE_FEATURE_OMITTED
        ):
            raise ValueError(
                "continuous boundary fields and categorical node-type features "
                "are separate model inputs"
            )
        residual_names = tuple(str(name) for name in boundary_residual_names)
        if any(not name for name in residual_names) or len(set(residual_names)) != len(
            residual_names
        ):
            raise ValueError("boundary_residual_names must be nonempty and unique")
        if boundary_residual_mode == BOUNDARY_RESIDUAL_NONE and residual_names:
            raise ValueError("the none boundary-residual mode cannot declare names")
        if boundary_residual_mode != BOUNDARY_RESIDUAL_NONE and not residual_names:
            raise ValueError("an active boundary-residual mode requires semantic names")
        if boundary_residual_mode != BOUNDARY_RESIDUAL_NONE:
            if boundary_field_mode != BOUNDARY_FIELD_NONE:
                raise ValueError(
                    "lifted boundary fields and a boundary-residual side path are "
                    "separate representation studies"
                )
            if node_type_feature_mode != NODE_TYPE_FEATURE_OMITTED:
                raise ValueError(
                    "a boundary-residual side path requires the no-type PCNO backbone"
                )

        modes = compute_Fourier_modes(
            2,
            [int(k_max), int(k_max)],
            [float(domain_lengths[0]), float(domain_lengths[1])],
        )
        modes_tensor = torch.as_tensor(modes, dtype=torch.float32)
        # coordinates (2), quadrature density (1), normalized conservative
        # state (4), one selected boundary representation, normalized Mach (1)
        type_feature_count = (
            0
            if node_type_feature_mode == NODE_TYPE_FEATURE_OMITTED
            else NUM_NODE_TYPES
        )
        boundary_feature_count = 0
        if boundary_field_mode == BOUNDARY_FIELD_GEOMETRY_COLLAR:
            boundary_feature_count = 1
        elif boundary_field_mode == BOUNDARY_FIELD_SEMANTIC_COLLAR:
            boundary_feature_count = len(field_names)
        in_dim = (
            2
            + nmeasures
            + NUM_EULER_COMPONENTS
            + type_feature_count
            + boundary_feature_count
            + 1
        )
        self.backbone = PCNO(
            2,
            modes_tensor,
            nmeasures=nmeasures,
            layers=[int(width) for width in layers],
            fc_dim=int(fc_dim),
            in_dim=in_dim,
            out_dim=NUM_EULER_COMPONENTS,
            act=act,
        )
        self.register_buffer(
            "state_mean",
            torch.as_tensor(normalization.state_mean, dtype=torch.float32).reshape(
                1, 1, -1
            ),
        )
        self.register_buffer(
            "state_scale",
            torch.as_tensor(normalization.state_scale, dtype=torch.float32).reshape(
                1, 1, -1
            ),
        )
        self.register_buffer(
            "residual_scale",
            torch.as_tensor(normalization.residual_scale, dtype=torch.float32).reshape(
                1, 1, -1
            ),
        )
        self.register_buffer(
            "mach_mean",
            torch.tensor(float(normalization.mach_mean), dtype=torch.float32),
        )
        self.register_buffer(
            "mach_scale",
            torch.tensor(float(normalization.mach_scale), dtype=torch.float32),
        )
        self.gamma = float(normalization.gamma)
        self.k_max = int(k_max)
        self.domain_lengths = tuple(float(value) for value in domain_lengths)
        self.layer_widths = tuple(int(value) for value in layers)
        self.projection_width = int(fc_dim)
        self.node_type_feature_mode = str(node_type_feature_mode)
        self.boundary_field_mode = str(boundary_field_mode)
        self.boundary_field_names = field_names
        self.boundary_field_input_count = int(boundary_feature_count)
        self.boundary_residual_mode = str(boundary_residual_mode)
        self.boundary_residual_names = residual_names
        self.boundary_residual_width = int(boundary_residual_width)
        if self.boundary_residual_mode == BOUNDARY_RESIDUAL_NONE:
            self.boundary_residual_input = None
            self.boundary_residual_hidden = None
            self.boundary_residual_output = None
        else:
            residual_input_dim = NUM_EULER_COMPONENTS + 1 + 3 * len(
                self.boundary_residual_names
            )
            self.boundary_residual_input = nn.Linear(
                residual_input_dim, self.boundary_residual_width
            )
            self.boundary_residual_hidden = nn.Linear(
                self.boundary_residual_width, self.boundary_residual_width
            )
            self.boundary_residual_output = nn.Linear(
                self.boundary_residual_width, NUM_EULER_COMPONENTS
            )
            nn.init.zeros_(self.boundary_residual_output.weight)
            nn.init.zeros_(self.boundary_residual_output.bias)
        if zero_initialize:
            self.zero_initialize_update_head()

    def zero_initialize_update_head(self) -> None:
        """Make the initial operator the identity map exactly."""

        nn.init.zeros_(self.backbone.fc2.weight)
        nn.init.zeros_(self.backbone.fc2.bias)

    def model_config(self) -> dict[str, Any]:
        return {
            "model": "PCNOEuler2DResidual",
            "k_max": self.k_max,
            "domain_lengths": list(self.domain_lengths),
            "layers": list(self.layer_widths),
            "fc_dim": self.projection_width,
            "in_dim": self.backbone.in_dim,
            "out_dim": self.backbone.out_dim,
            "nmeasures": self.backbone.nmeasures,
            "node_type_feature_mode": self.node_type_feature_mode,
            "boundary_field_mode": self.boundary_field_mode,
            "boundary_field_names": list(self.boundary_field_names),
            "boundary_field_input_count": self.boundary_field_input_count,
            "boundary_residual_mode": self.boundary_residual_mode,
            "boundary_residual_names": list(self.boundary_residual_names),
            "boundary_residual_width": self.boundary_residual_width,
            "boundary_residual_input_feature_names": (
                self.boundary_residual_input_feature_names()
            ),
            "input_feature_names": self.input_feature_names(),
        }

    def input_feature_names(self) -> list[str]:
        """Return the exact lifting-column layout in model-input order."""

        names = [
            "coordinate_x",
            "coordinate_y",
            "quadrature_density",
            "normalized_conservative_rho",
            "normalized_conservative_rho_u",
            "normalized_conservative_rho_v",
            "normalized_conservative_energy",
        ]
        if self.node_type_feature_mode == NODE_TYPE_FEATURE_ONE_HOT:
            names.extend(f"node_type_one_hot_{index}" for index in range(4))
        elif self.node_type_feature_mode == NODE_TYPE_FEATURE_CONSTANT_ZERO:
            names.extend(f"constant_zero_type_slot_{index}" for index in range(4))
        if self.boundary_field_mode == BOUNDARY_FIELD_GEOMETRY_COLLAR:
            names.append("boundary_geometry_collar")
        elif self.boundary_field_mode == BOUNDARY_FIELD_SEMANTIC_COLLAR:
            names.extend(
                f"boundary_semantic_collar_{name}"
                for name in self.boundary_field_names
            )
        names.append("normalized_mach")
        if len(names) != self.backbone.in_dim:
            raise AssertionError("input feature names do not match the lifting width")
        return names

    def boundary_residual_input_feature_names(self) -> list[str]:
        """Return the side-path layout without adding PCNO lifting columns."""

        if self.boundary_residual_mode == BOUNDARY_RESIDUAL_NONE:
            return []
        names = [
            "normalized_conservative_rho",
            "normalized_conservative_rho_u",
            "normalized_conservative_rho_v",
            "normalized_conservative_energy",
            "normalized_mach",
        ]
        names.extend(
            f"boundary_semantic_collar_{name}"
            for name in self.boundary_residual_names
        )
        for name in self.boundary_residual_names:
            names.extend(
                (
                    f"boundary_semantic_collar_{name}_unit_gradient_x",
                    f"boundary_semantic_collar_{name}_unit_gradient_y",
                )
            )
        if self.boundary_residual_input is None:
            raise RuntimeError("active boundary-residual mode lacks its input layer")
        if len(names) != self.boundary_residual_input.in_features:
            raise AssertionError("side-path feature names do not match its input width")
        return names

    def _selected_boundary_fields(
        self, boundary_features: torch.Tensor
    ) -> torch.Tensor:
        fields = boundary_features.to(dtype=self.backbone.fc0.weight.dtype)
        if self.boundary_field_mode == BOUNDARY_FIELD_GEOMETRY_COLLAR:
            return fields.amax(dim=-1, keepdim=True)
        if self.boundary_field_mode == BOUNDARY_FIELD_SEMANTIC_COLLAR:
            return fields
        raise ValueError("the model has no active continuous boundary field")

    def boundary_lift_contribution(
        self, boundary_features: torch.Tensor
    ) -> torch.Tensor:
        """Return the exact additive pre-activation lift W_B B(x).

        This is an algebraic decomposition of the linear lifting layer. It is
        useful for diagnostics but is not by itself a causal claim about later
        nonlinear PCNO blocks or rollout behavior.
        """

        if boundary_features.ndim != 3 or boundary_features.shape[-1] != len(
            self.boundary_field_names
        ):
            raise ValueError(
                "boundary_features must have shape [B, N, raw_boundary_fields]"
            )
        if (
            not bool(torch.isfinite(boundary_features).all())
            or bool((boundary_features < 0.0).any())
            or bool((boundary_features > 1.0).any())
        ):
            raise ValueError("boundary_features must be finite and lie in [0,1]")
        selected = self._selected_boundary_fields(boundary_features)
        first = TYPE_FEATURE_START
        last = first + self.boundary_field_input_count
        return F.linear(
            selected,
            self.backbone.fc0.weight[:, first:last],
            bias=None,
        )

    @staticmethod
    def _expanded_mach(mach: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, _ = current.shape
        if mach.ndim == 0:
            return mach.reshape(1, 1, 1).expand(batch_size, num_nodes, 1)
        if mach.ndim == 1:
            if mach.shape[0] != batch_size:
                raise ValueError(
                    "one-dimensional Mach tensor must have one value per batch"
                )
            return mach.reshape(batch_size, 1, 1).expand(batch_size, num_nodes, 1)
        if mach.ndim == 2 and mach.shape == (batch_size, 1):
            return mach.reshape(batch_size, 1, 1).expand(batch_size, num_nodes, 1)
        if mach.ndim == 3 and mach.shape[:2] == (batch_size, num_nodes):
            return mach
        raise ValueError("Mach must be scalar, [B], [B,1], or [B,N,1]")

    def normalized_input(
        self,
        current_conservative: torch.Tensor,
        *,
        nodes: torch.Tensor,
        node_rhos: torch.Tensor,
        node_type: torch.Tensor,
        mach: torch.Tensor,
        boundary_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if current_conservative.ndim != 3 or current_conservative.shape[-1] != 4:
            raise ValueError("current state must have shape [B, N, 4]")
        if nodes.shape != current_conservative.shape[:2] + (2,):
            raise ValueError("nodes must have shape [B, N, 2]")
        if node_rhos.shape[:2] != current_conservative.shape[:2]:
            raise ValueError("node_rhos must align with current state")
        if node_type.ndim == 3 and node_type.shape[-1] == 1:
            node_type = node_type[..., 0]
        if node_type.shape != current_conservative.shape[:2]:
            raise ValueError("node_type must have shape [B, N] or [B, N, 1]")
        if bool(((node_type < 0) | (node_type >= NUM_NODE_TYPES)).any()):
            raise ValueError("node_type contains an unsupported code")
        mach_nodes = self._expanded_mach(mach, current_conservative)
        normalized_mach = (mach_nodes - self.mach_mean) / self.mach_scale
        normalized_state = (current_conservative - self.state_mean) / self.state_scale
        features = [nodes, node_rhos, normalized_state]
        if self.node_type_feature_mode == NODE_TYPE_FEATURE_ONE_HOT:
            features.append(
                F.one_hot(
                    node_type.to(torch.int64),
                    num_classes=NUM_NODE_TYPES,
                ).to(dtype=current_conservative.dtype)
            )
        elif self.node_type_feature_mode == NODE_TYPE_FEATURE_CONSTANT_ZERO:
            features.append(
                current_conservative.new_zeros(
                    current_conservative.shape[:2] + (NUM_NODE_TYPES,)
                )
            )
        if self.boundary_field_mode != BOUNDARY_FIELD_NONE:
            if boundary_features is None:
                raise ValueError("the model requires boundary_features")
            expected = current_conservative.shape[:2] + (
                len(self.boundary_field_names),
            )
            if boundary_features.shape != expected:
                raise ValueError(
                    f"boundary_features must have shape {expected}, got "
                    f"{tuple(boundary_features.shape)}"
                )
            if (
                not bool(torch.isfinite(boundary_features).all())
                or bool((boundary_features < 0.0).any())
                or bool((boundary_features > 1.0).any())
            ):
                raise ValueError("boundary_features must be finite and lie in [0,1]")
            selected_fields = self._selected_boundary_fields(boundary_features)
            features.append(selected_fields.to(dtype=current_conservative.dtype))
        features.append(normalized_mach)
        return torch.cat(features, dim=-1)

    def normalized_boundary_residual(
        self,
        current_conservative: torch.Tensor,
        *,
        mach: torch.Tensor,
        boundary_features: torch.Tensor | None,
        directed_edges: torch.Tensor,
        edge_gradient_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Return the gated side-path correction in normalized-residual units.

        Semantic collars never enter the shared PCNO lift or any PCNO block. The
        pointwise side path is multiplied by the union collar, so it is exactly
        zero outside the fixed physical support. This is a bounded volume-
        residual operator, not a surface quadrature or a BNO claim.
        """

        if self.boundary_residual_mode == BOUNDARY_RESIDUAL_NONE:
            return torch.zeros_like(current_conservative)
        if boundary_features is None:
            raise ValueError("the boundary-residual side path requires boundary_features")
        expected = current_conservative.shape[:2] + (
            len(self.boundary_residual_names),
        )
        if boundary_features.shape != expected:
            raise ValueError(
                f"boundary_features must have shape {expected}, got "
                f"{tuple(boundary_features.shape)}"
            )
        if (
            not bool(torch.isfinite(boundary_features).all())
            or bool((boundary_features < 0.0).any())
            or bool((boundary_features > 1.0).any())
        ):
            raise ValueError("boundary_features must be finite and lie in [0,1]")
        if self.boundary_residual_input is None:
            raise RuntimeError("active boundary-residual mode lacks its input layer")
        if self.boundary_residual_hidden is None:
            raise RuntimeError("active boundary-residual mode lacks its hidden layer")
        if self.boundary_residual_output is None:
            raise RuntimeError("active boundary-residual mode lacks its output layer")

        normalized_state = (current_conservative - self.state_mean) / self.state_scale
        mach_nodes = self._expanded_mach(mach, current_conservative)
        normalized_mach = (mach_nodes - self.mach_mean) / self.mach_scale
        fields = boundary_features.to(dtype=current_conservative.dtype)
        batch_size, node_count, field_count = fields.shape
        field_gradients = compute_gradient(
            fields.permute(0, 2, 1),
            directed_edges,
            edge_gradient_weights.to(dtype=fields.dtype),
        ).reshape(batch_size, field_count, 2, node_count)
        gradient_norm = torch.linalg.vector_norm(
            field_gradients, dim=2, keepdim=True
        )
        unit_gradients = field_gradients / gradient_norm.clamp_min(1.0e-12)
        unit_gradients = torch.where(
            gradient_norm > 1.0e-12,
            unit_gradients,
            torch.zeros_like(unit_gradients),
        )
        unit_gradients = unit_gradients.permute(0, 3, 1, 2).reshape(
            batch_size, node_count, 2 * field_count
        )
        branch_input = torch.cat(
            (normalized_state, normalized_mach, fields, unit_gradients), dim=-1
        ).to(dtype=self.boundary_residual_input.weight.dtype)
        hidden = F.gelu(self.boundary_residual_input(branch_input))
        hidden = F.gelu(self.boundary_residual_hidden(hidden))
        correction = self.boundary_residual_output(hidden)
        union_gate = fields.amax(dim=-1, keepdim=True).to(dtype=correction.dtype)
        return correction * union_gate

    def forward(
        self,
        current_conservative: torch.Tensor,
        *,
        node_mask: torch.Tensor,
        nodes: torch.Tensor,
        node_weights: torch.Tensor,
        node_rhos: torch.Tensor,
        directed_edges: torch.Tensor,
        edge_gradient_weights: torch.Tensor,
        node_type: torch.Tensor,
        mach: torch.Tensor,
        boundary_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        model_input = self.normalized_input(
            current_conservative,
            nodes=nodes,
            node_rhos=node_rhos,
            node_type=node_type,
            mach=mach,
            boundary_features=boundary_features,
        )
        normalized_residual = self.backbone(
            model_input,
            (node_mask, nodes, node_weights, directed_edges, edge_gradient_weights),
        )
        normalized_residual = normalized_residual + self.normalized_boundary_residual(
            current_conservative,
            mach=mach,
            boundary_features=boundary_features,
            directed_edges=directed_edges,
            edge_gradient_weights=edge_gradient_weights,
        )
        prediction = current_conservative + normalized_residual * self.residual_scale
        return prediction * node_mask


def copy_no_boundary_initialization_to_boundary_field_model(
    no_boundary_model: PCNOEuler2DResidual,
    boundary_field_model: PCNOEuler2DResidual,
) -> dict[str, Any]:
    """Make a boundary-field model exactly match an eight-input model initially.

    The seven coordinate, quadrature-density, and state columns are copied in
    place; normalized Mach is moved to the final lifting column.  All inserted
    boundary-field columns start at zero, while every non-lifting parameter is
    copied exactly.  The new fields can still learn immediately through their
    lifting-weight gradients.
    """

    if no_boundary_model.node_type_feature_mode != NODE_TYPE_FEATURE_OMITTED:
        raise ValueError("source model must omit categorical node-type features")
    if no_boundary_model.boundary_field_mode != BOUNDARY_FIELD_NONE:
        raise ValueError("source model must omit continuous boundary fields")
    if boundary_field_model.node_type_feature_mode != NODE_TYPE_FEATURE_OMITTED:
        raise ValueError("target model must omit categorical node-type features")
    if boundary_field_model.boundary_field_mode == BOUNDARY_FIELD_NONE:
        raise ValueError("target model must use a continuous boundary field")

    source = no_boundary_model.state_dict()
    target = boundary_field_model.state_dict()
    if set(source) != set(target):
        raise ValueError("source and target state dictionaries have different keys")
    if no_boundary_model.backbone.in_dim != TYPE_FEATURE_START + 1:
        raise ValueError("source lifting layer does not have eight inputs")
    target_field_count = boundary_field_model.backbone.in_dim - (
        TYPE_FEATURE_START + 1
    )
    if target_field_count < 1:
        raise ValueError("target lifting layer has no inserted boundary columns")

    copied: dict[str, torch.Tensor] = {}
    for name, target_value in target.items():
        source_value = source[name]
        if name == "backbone.fc0.weight":
            mapped = torch.zeros_like(target_value)
            mapped[:, :TYPE_FEATURE_START] = source_value[:, :TYPE_FEATURE_START]
            mapped[:, -1] = source_value[:, -1]
            copied[name] = mapped
        else:
            if source_value.shape != target_value.shape:
                raise ValueError(f"non-lifting parameter shape differs: {name}")
            copied[name] = source_value.detach().clone()
    boundary_field_model.load_state_dict(copied, strict=True)

    target_lift = boundary_field_model.backbone.fc0.weight.detach()
    source_lift = no_boundary_model.backbone.fc0.weight.detach()
    exact = bool(
        torch.equal(
            target_lift[:, :TYPE_FEATURE_START],
            source_lift[:, :TYPE_FEATURE_START],
        )
        and torch.equal(target_lift[:, -1], source_lift[:, -1])
        and torch.count_nonzero(
            target_lift[:, TYPE_FEATURE_START:-1]
        ).item()
        == 0
    )
    if not exact:
        raise RuntimeError("matched boundary-field lifting copy is not exact")
    return {
        "schema": "pcno_boundary_field_initialization_v1",
        "kind": "copied_from_no_boundary_model",
        "source_input_dim": int(no_boundary_model.backbone.in_dim),
        "target_input_dim": int(boundary_field_model.backbone.in_dim),
        "inserted_boundary_columns": list(
            range(TYPE_FEATURE_START, boundary_field_model.backbone.in_dim - 1)
        ),
        "lifting_copy_exact": True,
        "all_non_lifting_state_copied": True,
        "active_weight_rescale_factor": 1.0,
        "mathematical_initial_function_match": True,
    }


def copy_no_boundary_initialization_to_boundary_residual_model(
    no_boundary_model: PCNOEuler2DResidual,
    boundary_residual_model: PCNOEuler2DResidual,
) -> dict[str, Any]:
    """Copy an eight-input PCNO into an exactly dormant residual-side-path model."""

    if no_boundary_model.node_type_feature_mode != NODE_TYPE_FEATURE_OMITTED:
        raise ValueError("source model must omit categorical node-type features")
    if no_boundary_model.boundary_field_mode != BOUNDARY_FIELD_NONE:
        raise ValueError("source model must omit lifted continuous boundary fields")
    if no_boundary_model.boundary_residual_mode != BOUNDARY_RESIDUAL_NONE:
        raise ValueError("source model must omit a boundary-residual side path")
    if boundary_residual_model.node_type_feature_mode != NODE_TYPE_FEATURE_OMITTED:
        raise ValueError("target model must omit categorical node-type features")
    if boundary_residual_model.boundary_field_mode != BOUNDARY_FIELD_NONE:
        raise ValueError("target model must omit lifted continuous boundary fields")
    if boundary_residual_model.boundary_residual_mode == BOUNDARY_RESIDUAL_NONE:
        raise ValueError("target model must use a boundary-residual side path")

    source = no_boundary_model.state_dict()
    target = boundary_residual_model.state_dict()
    copied = {
        name: (source[name].detach().clone() if name in source else value)
        for name, value in target.items()
    }
    boundary_residual_model.load_state_dict(copied, strict=True)
    common_exact = all(torch.equal(source[name], copied[name]) for name in source)
    output = boundary_residual_model.boundary_residual_output
    if output is None:
        raise RuntimeError("target side path lacks its output layer")
    output_zero = bool(
        torch.count_nonzero(output.weight.detach()).item() == 0
        and torch.count_nonzero(output.bias.detach()).item() == 0
    )
    if not common_exact or not output_zero:
        raise RuntimeError("matched boundary-residual initialization is not exact")
    return {
        "schema": "pcno_boundary_residual_initialization_v1",
        "kind": "copied_from_no_boundary_model",
        "source_input_dim": int(no_boundary_model.backbone.in_dim),
        "target_input_dim": int(boundary_residual_model.backbone.in_dim),
        "pcno_backbone_state_copied_exactly": True,
        "normalization_state_copied_exactly": True,
        "boundary_residual_output_zero": True,
        "mathematical_initial_function_match": True,
        "shared_pcno_lift_receives_boundary_features": False,
    }


def copy_no_type_initialization_to_zero_channels(
    no_type_model: PCNOEuler2DResidual,
    zero_channel_model: PCNOEuler2DResidual,
) -> dict[str, Any]:
    """Copy an 8-input initialization into the corresponding 12-input model.

    The first seven active features keep their columns, while normalized Mach
    moves from column 7 to column 11.  Columns 7:11 of the target are set to
    zero and always receive literal-zero inputs.  The resulting mathematical
    function exactly matches ``no_type_model`` at initialization.
    """

    if no_type_model.node_type_feature_mode != NODE_TYPE_FEATURE_OMITTED:
        raise ValueError("source model must omit node-type features")
    if zero_channel_model.node_type_feature_mode != NODE_TYPE_FEATURE_CONSTANT_ZERO:
        raise ValueError("target model must use four constant-zero features")

    source = no_type_model.state_dict()
    target = zero_channel_model.state_dict()
    if set(source) != set(target):
        raise ValueError("source and target state dictionaries have different keys")

    copied: dict[str, torch.Tensor] = {}
    for name, target_value in target.items():
        source_value = source[name]
        if name == "backbone.fc0.weight":
            if source_value.shape[1] != TYPE_FEATURE_START + 1:
                raise ValueError("no-type lifting layer does not have eight inputs")
            if target_value.shape[1] != TYPE_FEATURE_STOP + 1:
                raise ValueError("zero-channel lifting layer does not have twelve inputs")
            mapped = torch.zeros_like(target_value)
            mapped[:, :TYPE_FEATURE_START] = source_value[:, :TYPE_FEATURE_START]
            mapped[:, TYPE_FEATURE_STOP] = source_value[:, TYPE_FEATURE_START]
            copied[name] = mapped
        else:
            if source_value.shape != target_value.shape:
                raise ValueError(f"non-lifting parameter shape differs: {name}")
            copied[name] = source_value.detach().clone()
    zero_channel_model.load_state_dict(copied, strict=True)

    target_lift = zero_channel_model.backbone.fc0.weight.detach()
    source_lift = no_type_model.backbone.fc0.weight.detach()
    exact = bool(
        torch.equal(
            target_lift[:, :TYPE_FEATURE_START],
            source_lift[:, :TYPE_FEATURE_START],
        )
        and torch.equal(
            target_lift[:, TYPE_FEATURE_STOP],
            source_lift[:, TYPE_FEATURE_START],
        )
        and torch.count_nonzero(
            target_lift[:, TYPE_FEATURE_START:TYPE_FEATURE_STOP]
        ).item()
        == 0
    )
    if not exact:
        raise RuntimeError("matched zero-channel lifting copy is not exact")
    return {
        "schema": "pcno_zero_channel_initialization_v1",
        "kind": "copied_from_no_type_model",
        "source_input_dim": int(no_type_model.backbone.in_dim),
        "target_input_dim": int(zero_channel_model.backbone.in_dim),
        "active_column_mapping": [
            *[[index, index] for index in range(TYPE_FEATURE_START)],
            [TYPE_FEATURE_START, TYPE_FEATURE_STOP],
        ],
        "constant_zero_target_columns": list(
            range(TYPE_FEATURE_START, TYPE_FEATURE_STOP)
        ),
        "lifting_copy_exact": True,
        "all_non_lifting_state_copied": True,
        "active_weight_rescale_factor": 1.0,
        "mathematical_initial_function_match": True,
    }


class PCNOEuler2DShardStore:
    """Lazy access to full-resolution trajectory shards produced by the prep CLI."""

    def __init__(
        self,
        root: str | Path,
        *,
        max_cached_trajectories: int = 8,
        max_cached_geometry_bytes: int = DEFAULT_MAX_CACHED_GEOMETRY_BYTES,
    ) -> None:
        if max_cached_trajectories < 1:
            raise ValueError("max_cached_trajectories must be positive")
        if max_cached_geometry_bytes < 0:
            raise ValueError("max_cached_geometry_bytes must be nonnegative")
        self.root = Path(root)
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"missing PCNO shard manifest: {manifest_path}")
        self.manifest_path = manifest_path
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(self.manifest.get("schema_version", -1)) != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported shard schema {self.manifest.get('schema_version')}; "
                f"expected {SCHEMA_VERSION}"
            )
        entries = self.manifest.get("trajectories", [])
        self._entries = {str(entry["key"]): dict(entry) for entry in entries}
        if len(self._entries) != len(entries):
            raise ValueError("trajectory keys must be unique")
        if not self._entries:
            raise ValueError("shard manifest contains no trajectories")
        self.max_cached_trajectories = int(max_cached_trajectories)
        self.max_cached_geometry_bytes = int(max_cached_geometry_bytes)
        self._arrays: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()
        self._geometry_tensors: dict[
            torch.device, OrderedDict[str, dict[str, torch.Tensor]]
        ] = {}
        self._geometry_tensor_bytes: dict[torch.device, int] = {}

    def __enter__(self) -> "PCNOEuler2DShardStore":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    @property
    def keys(self) -> list[str]:
        return sorted(self._entries, key=_natural_key)

    @property
    def manifest_digest(self) -> str:
        return hashlib.sha256(self.manifest_path.read_bytes()).hexdigest()

    @property
    def boundary_field_contract(self) -> dict[str, Any] | None:
        """Return the validated optional semantic-field artifact contract."""

        raw = self.manifest.get("boundary_field_contract")
        if raw is None:
            return None
        if not isinstance(raw, Mapping):
            raise ValueError("boundary_field_contract must be a mapping")
        contract = dict(raw)
        if contract.get("schema") != "pcno_bounded_semantic_collar_v1":
            raise ValueError("unsupported boundary-field contract schema")
        names = contract.get("channel_names")
        if (
            not isinstance(names, list)
            or not names
            or not all(isinstance(name, str) and name for name in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError("boundary-field channel names are invalid")
        width = contract.get("physical_width")
        if (
            not isinstance(width, (int, float))
            or not math.isfinite(float(width))
            or float(width) <= 0.0
        ):
            raise ValueError("boundary-field physical width is invalid")
        if contract.get("continuum_object") != "bounded_volume_descriptor":
            raise ValueError("boundary fields must be bounded volume descriptors")
        if contract.get("surface_measure_scaling") is not False:
            raise ValueError("surface-measure-scaled fields are not model channels")
        return contract

    @property
    def boundary_field_names(self) -> tuple[str, ...]:
        contract = self.boundary_field_contract
        if contract is None:
            return ()
        return tuple(str(name) for name in contract["channel_names"])

    def entry(self, key: str) -> dict[str, Any]:
        try:
            return self._entries[str(key)]
        except KeyError as exc:
            raise KeyError(f"unknown trajectory key {key!r}") from exc

    def _path(self, key: str, name: str) -> Path:
        entry = self.entry(key)
        return self.root / str(entry["folder"]) / f"{name}.npy"

    def array(self, key: str, name: str) -> np.ndarray:
        trajectory_key = str(key)
        arrays = self._arrays.get(trajectory_key)
        if arrays is None:
            while len(self._arrays) >= self.max_cached_trajectories:
                _, evicted = self._arrays.popitem(last=False)
                self._close_arrays(evicted.values())
            arrays = {}
            self._arrays[trajectory_key] = arrays
        else:
            self._arrays.move_to_end(trajectory_key)
        if name not in arrays:
            path = self._path(trajectory_key, name)
            if not path.is_file():
                raise FileNotFoundError(f"missing trajectory array: {path}")
            arrays[name] = np.load(path, mmap_mode="r")
        return arrays[name]

    def close(self) -> None:
        """Release cached device tensors and close memory maps deterministically."""

        self.clear_geometry_cache()
        for arrays in self._arrays.values():
            self._close_arrays(arrays.values())
        self._arrays.clear()

    @property
    def cached_geometry_bytes(self) -> int:
        """Total bytes currently retained by immutable geometry caches."""

        return sum(self._geometry_tensor_bytes.values())

    @property
    def cached_geometry_entries(self) -> int:
        """Number of trajectory/device geometry entries currently retained."""

        return sum(len(entries) for entries in self._geometry_tensors.values())

    def clear_geometry_cache(self, device: torch.device | None = None) -> None:
        """Drop immutable geometry tensors for one device or for all devices."""

        if device is None:
            self._geometry_tensors.clear()
            self._geometry_tensor_bytes.clear()
            return
        canonical_device = _canonical_device(device)
        self._geometry_tensors.pop(canonical_device, None)
        self._geometry_tensor_bytes.pop(canonical_device, None)

    @staticmethod
    def _close_arrays(arrays: Iterable[np.ndarray]) -> None:
        for array in arrays:
            mmap_handle = getattr(array, "_mmap", None)
            if mmap_handle is not None:
                mmap_handle.close()

    def states(self, key: str) -> np.ndarray:
        states = self.array(key, "states_conservative")
        if states.ndim != 3 or states.shape[-1] != NUM_EULER_COMPONENTS:
            raise ValueError(
                f"invalid state shape for trajectory {key}: {states.shape}"
            )
        return states

    def geometry_numpy(self, key: str) -> dict[str, np.ndarray | float]:
        geometry: dict[str, np.ndarray | float] = {
            "nodes": self.array(key, "nodes"),
            "node_measures": self.array(key, "node_measures"),
            "node_weights": self.array(key, "node_weights"),
            "node_rhos": self.array(key, "node_rhos"),
            "directed_edges": self.array(key, "directed_edges"),
            "edge_gradient_weights": self.array(key, "edge_gradient_weights"),
            "node_type": self.array(key, "node_type"),
            "mach": float(self.entry(key)["mach"]),
        }
        if self.boundary_field_contract is not None:
            boundary_features = self.array(key, "boundary_features")
            expected = (
                int(np.asarray(geometry["nodes"]).shape[0]),
                len(self.boundary_field_names),
            )
            if boundary_features.shape != expected:
                raise ValueError(
                    f"invalid boundary feature shape for trajectory {key}: "
                    f"{boundary_features.shape}, expected {expected}"
                )
            if (
                not np.all(np.isfinite(boundary_features))
                or np.any(boundary_features < 0.0)
                or np.any(boundary_features > 1.0)
            ):
                raise ValueError("boundary features must be finite and lie in [0,1]")
            geometry["boundary_features"] = boundary_features
        return geometry

    def tensor_sample(
        self,
        key: str,
        time_index: int,
        *,
        step_stride: int,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        return self.tensor_batch(
            key,
            [time_index],
            step_stride=step_stride,
            device=device,
        )

    def tensor_batch(
        self,
        key: str,
        time_indices: Sequence[int],
        *,
        step_stride: int,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        """Materialize a homogeneous batch from one trajectory geometry.

        Static geometry tensors are read-only expanded views and may share
        storage across calls. Current and target states always own fresh storage.
        """

        indices = np.asarray([int(index) for index in time_indices], dtype=np.int64)
        if indices.ndim != 1 or indices.size == 0:
            raise ValueError("time_indices must contain at least one index")
        if step_stride < 1:
            raise ValueError("step_stride must be positive")
        states = self.states(key)
        target_indices = indices + int(step_stride)
        if bool((indices < 0).any()) or bool((target_indices >= states.shape[0]).any()):
            raise IndexError(
                f"invalid time batch for trajectory {key} with {states.shape[0]} frames"
            )
        canonical_device = _canonical_device(device)
        current = _copy_tensor(states[indices], torch.float32, canonical_device)
        target = _copy_tensor(states[target_indices], torch.float32, canonical_device)
        batch_size = int(indices.size)

        geometry_tensors = self._tensor_geometry(str(key), device=canonical_device)

        def expanded_geometry(name: str) -> torch.Tensor:
            tensor = geometry_tensors[name]
            return tensor.unsqueeze(0).expand(batch_size, *tensor.shape)

        nodes = expanded_geometry("nodes")
        sample = {
            "current": current,
            "target": target,
            "node_mask": expanded_geometry("node_mask"),
            "nodes": nodes,
            "node_measures": expanded_geometry("node_measures"),
            "node_weights": expanded_geometry("node_weights"),
            "node_rhos": expanded_geometry("node_rhos"),
            "directed_edges": expanded_geometry("directed_edges"),
            "edge_gradient_weights": expanded_geometry("edge_gradient_weights"),
            "node_type": expanded_geometry("node_type"),
            "mach": torch.full(
                (batch_size,),
                float(self.entry(key)["mach"]),
                dtype=torch.float32,
                device=canonical_device,
            ),
        }
        if "boundary_features" in geometry_tensors:
            sample["boundary_features"] = expanded_geometry("boundary_features")
        return sample

    def _tensor_geometry(
        self,
        key: str,
        *,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        """Return read-only trajectory-static tensors, caching within a byte limit."""

        device_cache = self._geometry_tensors.setdefault(device, OrderedDict())
        cached = device_cache.get(key)
        if cached is not None:
            device_cache.move_to_end(key)
            return cached

        geometry = self.geometry_numpy(key)
        tensors = {
            "nodes": _copy_tensor(geometry["nodes"], torch.float32, device),
            "node_measures": _copy_tensor(
                geometry["node_measures"], torch.float32, device
            ),
            "node_weights": _copy_tensor(
                geometry["node_weights"], torch.float32, device
            ),
            "node_rhos": _copy_tensor(geometry["node_rhos"], torch.float32, device),
            "directed_edges": _copy_tensor(
                geometry["directed_edges"], torch.int64, device
            ),
            "edge_gradient_weights": _copy_tensor(
                geometry["edge_gradient_weights"], torch.float32, device
            ),
            "node_type": _copy_tensor(geometry["node_type"], torch.int64, device),
            "node_mask": torch.ones(
                (int(np.asarray(geometry["nodes"]).shape[0]), 1),
                dtype=torch.float32,
                device=device,
            ),
        }
        if "boundary_features" in geometry:
            tensors["boundary_features"] = _copy_tensor(
                geometry["boundary_features"], torch.float32, device
            )
        entry_bytes = sum(
            tensor.numel() * tensor.element_size() for tensor in tensors.values()
        )
        budget = self.max_cached_geometry_bytes
        if budget == 0 or entry_bytes > budget:
            if not device_cache:
                self._geometry_tensors.pop(device, None)
                self._geometry_tensor_bytes.pop(device, None)
            return tensors

        retained_bytes = self._geometry_tensor_bytes.get(device, 0)
        while device_cache and retained_bytes + entry_bytes > budget:
            _, evicted = device_cache.popitem(last=False)
            retained_bytes -= sum(
                tensor.numel() * tensor.element_size()
                for tensor in evicted.values()
            )
        device_cache[key] = tensors
        self._geometry_tensor_bytes[device] = retained_bytes + entry_bytes
        return tensors


def _copy_tensor(value: Any, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    array = np.array(value, copy=True)
    return torch.as_tensor(array, dtype=dtype, device=device)


def _canonical_device(device: torch.device) -> torch.device:
    requested = torch.device(device)
    if requested.type == "cuda" and requested.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return requested


def _natural_key(value: str) -> tuple[int, int | str]:
    text = str(value)
    try:
        return (0, int(text))
    except ValueError:
        return (1, text)


def fit_normalization(
    store: PCNOEuler2DShardStore,
    keys: Sequence[str],
    *,
    step_stride: int = 1,
    time_stride: int = 1,
    gamma: float = 1.4,
    mach_scale_floor: float = 0.1,
) -> Euler2DNormalization:
    """Fit proxy-mass-weighted fixed scales without materializing all pairs."""

    if not keys:
        raise ValueError("normalization requires at least one training trajectory")
    if step_stride < 1 or time_stride < 1:
        raise ValueError("step_stride and time_stride must be positive")
    if not np.isfinite(mach_scale_floor) or mach_scale_floor <= 0.0:
        raise ValueError("mach_scale_floor must be positive and finite")
    state_sum = np.zeros(NUM_EULER_COMPONENTS, dtype=np.float64)
    state_square_sum = np.zeros(NUM_EULER_COMPONENTS, dtype=np.float64)
    residual_square_sum = np.zeros(NUM_EULER_COMPONENTS, dtype=np.float64)
    state_weight = 0.0
    residual_weight = 0.0
    mach_values = []

    for key in keys:
        states = store.states(key)
        weights = np.asarray(store.array(key, "node_weights"), dtype=np.float64).sum(
            axis=-1
        )
        weight_sum = float(weights.sum())
        if not np.isfinite(weights).all() or weight_sum <= 0.0:
            raise ValueError(f"trajectory {key} has invalid proxy weights")
        for time_index in range(0, states.shape[0], time_stride):
            state = np.asarray(states[time_index], dtype=np.float64)
            state_sum += np.einsum("n,nc->c", weights, state)
            state_square_sum += np.einsum("n,nc->c", weights, np.square(state))
            state_weight += weight_sum
        for time_index in range(0, states.shape[0] - step_stride, time_stride):
            residual = np.asarray(
                states[time_index + step_stride], dtype=np.float64
            ) - np.asarray(states[time_index], dtype=np.float64)
            residual_square_sum += np.einsum("n,nc->c", weights, np.square(residual))
            residual_weight += weight_sum
        mach_values.append(float(store.entry(key)["mach"]))

    if state_weight <= 0.0 or residual_weight <= 0.0:
        raise ValueError("normalization received no weighted states or residuals")
    state_mean = state_sum / state_weight
    state_variance = np.maximum(
        state_square_sum / state_weight - np.square(state_mean),
        0.0,
    )
    state_scale = np.maximum(np.sqrt(state_variance), 1e-6)
    residual_rms = np.sqrt(residual_square_sum / residual_weight)
    residual_scale = np.maximum(residual_rms, 1e-6 * state_scale)
    mach_array = np.asarray(mach_values, dtype=np.float64)
    mach_scale = max(float(mach_array.std()), float(mach_scale_floor))
    return Euler2DNormalization(
        state_mean=state_mean,
        state_scale=state_scale,
        residual_scale=residual_scale,
        mach_mean=float(mach_array.mean()),
        mach_scale=mach_scale,
        mach_scale_floor=float(mach_scale_floor),
        gamma=gamma,
        weight_provenance=str(
            store.manifest.get("weight_provenance", "reconstructed_vertex_lumped_proxy")
        ),
    )


def stratified_train_val_split(
    store: PCNOEuler2DShardStore,
    *,
    val_count: int,
    seed: int,
    keys: Sequence[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Select validation cases across the Mach/node-count range deterministically."""

    candidates = list(store.keys if keys is None else (str(key) for key in keys))
    if val_count < 1 or val_count >= len(candidates):
        raise ValueError("val_count must be between one and len(keys)-1")
    ordered = sorted(
        candidates,
        key=lambda key: (
            float(store.entry(key)["mach"]),
            int(store.entry(key)["num_nodes"]),
            _natural_key(key),
        ),
    )
    rng = np.random.default_rng(int(seed))
    chunks = np.array_split(np.asarray(ordered, dtype=object), val_count)
    validation = [str(chunk[int(rng.integers(0, len(chunk)))]) for chunk in chunks]
    validation_set = set(validation)
    training = [key for key in ordered if key not in validation_set]
    return training, sorted(validation, key=_natural_key)


def balanced_presentations(
    store: PCNOEuler2DShardStore,
    keys: Sequence[str],
    *,
    step_stride: int,
    count: int,
    rng: np.random.Generator,
    minimum_time_index: int = 0,
) -> list[tuple[str, int]]:
    """Sample time pairs while balancing trajectory exposure."""

    trajectory_keys = [str(key) for key in keys]
    if not trajectory_keys:
        raise ValueError("presentation sampling requires at least one trajectory")
    if count < 1 or step_stride < 1 or minimum_time_index < 0:
        raise ValueError(
            "count and step_stride must be positive and minimum time nonnegative"
        )
    presentations: list[tuple[str, int]] = []
    while len(presentations) < count:
        for key in rng.permutation(trajectory_keys).tolist():
            num_steps = int(store.entry(str(key))["num_steps"])
            upper = num_steps - step_stride
            if upper <= minimum_time_index:
                raise ValueError(
                    f"trajectory {key} is too short for stride {step_stride} "
                    f"and minimum time {minimum_time_index}"
                )
            time_index = int(rng.integers(minimum_time_index, upper))
            presentations.append((str(key), time_index))
            if len(presentations) == count:
                break
    return presentations


def full_coverage_presentations(
    store: PCNOEuler2DShardStore,
    keys: Sequence[str],
    *,
    step_stride: int,
    rng: np.random.Generator,
    minimum_time_index: int = 0,
) -> list[tuple[str, int]]:
    """Visit every eligible transition exactly once in shuffled order."""

    trajectory_keys = [str(key) for key in keys]
    if not trajectory_keys:
        raise ValueError("full coverage requires at least one trajectory")
    if step_stride < 1 or minimum_time_index < 0:
        raise ValueError("step_stride must be positive and minimum time nonnegative")
    presentations: list[tuple[str, int]] = []
    for key in rng.permutation(trajectory_keys).tolist():
        upper = int(store.entry(str(key))["num_steps"]) - int(step_stride)
        if upper <= minimum_time_index:
            raise ValueError(
                f"trajectory {key} is too short for stride {step_stride} "
                f"and minimum time {minimum_time_index}"
            )
        indices = np.arange(minimum_time_index, upper, dtype=np.int64)
        rng.shuffle(indices)
        presentations.extend((str(key), int(index)) for index in indices.tolist())
    return presentations


def fixed_tiny_presentations(
    store: PCNOEuler2DShardStore,
    keys: Sequence[str],
    *,
    step_stride: int,
    count: int,
    seed: int,
) -> list[tuple[str, int]]:
    """Create one immutable tiny-fit pair bank."""

    return balanced_presentations(
        store,
        keys,
        step_stride=step_stride,
        count=count,
        rng=np.random.default_rng(int(seed)),
    )


def homogeneous_presentation_batches(
    pairs: Sequence[tuple[str, int]],
    *,
    batch_size: int,
    rng: np.random.Generator | None = None,
) -> list[tuple[str, list[int]]]:
    """Group presentations by trajectory without padding across different meshes."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    grouped: dict[str, list[int]] = {}
    for key, time_index in pairs:
        grouped.setdefault(str(key), []).append(int(time_index))
    batches = [
        (key, indices[start : start + batch_size])
        for key, indices in grouped.items()
        for start in range(0, len(indices), batch_size)
    ]
    if rng is not None and len(batches) > 1:
        order = rng.permutation(len(batches))
        batches = [batches[int(index)] for index in order]
    return batches


def homogeneous_optimizer_step_count(
    pairs: Sequence[tuple[str, int]],
    *,
    batch_size: int,
) -> int:
    """Count optimizer steps without materializing homogeneous batches."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    counts: dict[str, int] = {}
    for key, _ in pairs:
        counts[str(key)] = counts.get(str(key), 0) + 1
    if not counts:
        raise ValueError("optimizer-step accounting requires at least one pair")
    return int(sum(math.ceil(count / batch_size) for count in counts.values()))


def parameter_count(model: nn.Module) -> int:
    return int(
        sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
    )


def digest_mapping(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
