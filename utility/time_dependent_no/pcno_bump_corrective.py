"""Solver-free Bump corrections on native meshes; no dataset or solver IO."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import nn
from torch.nn import functional as F

from utility.time_dependent_no.pcno_euler2d import (
    PCNOEuler2DResidual,
    normal_node_mask,
    weighted_scaled_mse,
)
from utility.time_dependent_no.pcno_naca0012_corrective_extension import (
    FourStepVPredictionScheduler,
)
from utility.time_dependent_no.pcno_rollout import (
    close_boundary,
    contract_forward_sample,
)

TRAIN_ARMS = (
    "CLEAN",
    "IID_RECOVERY",
    "CURRICULUM_EMA_PREFIX_K13",
    "PREFIX_ERROR_CORRECTOR_K13",
    "PCNO_PDEREFINER_K3_VPRED",
)
EVALUATION_ARMS = (
    "CLEAN",
    "CLEAN_EMA",
    *TRAIN_ARMS[1:],
    "MAPPED_TRAIN_PCA_R7",
    "MAPPED_TRAIN_PCA_R32",
)


def keyed_seed(seed: int, *parts: object) -> int:
    payload = "|".join(map(str, ("b5_comparison_v1", seed, *parts))).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**63 - 1)


def generator_for(device: torch.device, seed: int, *parts: object) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(keyed_seed(seed, *parts))


def exposure_probability(step: int, total_steps: int) -> float:
    """Zero for the first 10%; linear ramp to probability 0.5 at 40%."""
    return 0.5 * min(1.0, max(0.0, (step / total_steps - 0.1) / 0.3))


def prefix_start(target_index: int, requested_depth: int, *, training_map: bool):
    """Keep the common clean target fixed, truncating only unavailable history."""
    if target_index < 1 or requested_depth not in (1, 2, 3):
        raise ValueError("invalid target or requested prefix depth")
    end = target_index - 1 if training_map else target_index
    depth = min(requested_depth, end)
    return end - depth, depth


@torch.no_grad()
def update_ema(ema: nn.Module, online: nn.Module, decay: float = 0.995) -> None:
    """Average parameters; copy fixed normalization/geometry buffers exactly."""
    for target, source in zip(ema.parameters(), online.parameters(), strict=True):
        target.lerp_(source, 1.0 - decay)
    for target, source in zip(ema.buffers(), online.buffers(), strict=True):
        target.copy_(source)


def normal_loss(prediction, target, sample, model):
    return weighted_scaled_mse(
        prediction,
        target,
        sample["node_weights"],
        normal_node_mask(sample["node_type"], sample["node_mask"]),
        model.state_scale,
    )


def map_step(model, sample, current, policy):
    return contract_forward_sample(model, sample, current, boundary_policy=policy)[0]


@torch.no_grad()
def detached_prefix(model, sample, initial, depth, policy):
    state = initial.detach()
    for _ in range(depth):
        state = map_step(model, sample, state, policy).detach()
        if not torch.isfinite(state).all():
            raise FloatingPointError("nonfinite detached prefix")
    return state


class BumpRefiner(PCNOEuler2DResidual):
    """Twenty-input shared PCNO predicting velocity in residual coordinates."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        old = self.backbone.fc0
        extended = nn.Linear(old.in_features + 8, old.out_features)
        with torch.no_grad():
            extended.weight.zero_()
            extended.weight[:, : old.in_features].copy_(old.weight)
            extended.bias.copy_(old.bias)
        self.backbone.fc0 = extended
        self.backbone.in_dim = old.in_features + 8

    def velocity(self, current, candidate, timesteps, sample):
        features = self.normalized_input(
            current,
            nodes=sample["nodes"],
            node_rhos=sample["node_rhos"],
            node_type=sample["node_type"],
            mach=sample["mach"],
        )
        levels = F.one_hot(timesteps, num_classes=4).to(current.dtype)
        levels = levels[:, None].expand(-1, current.shape[1], -1)
        inputs = torch.cat((features, candidate, levels), dim=-1)
        aux = tuple(
            sample[k]
            for k in (
                "node_mask",
                "nodes",
                "node_weights",
                "directed_edges",
                "edge_gradient_weights",
            )
        )
        return self.backbone(inputs, aux).float()


class BumpRefinerScheduler(FourStepVPredictionScheduler):
    """Reuse the audited DDPM equations with a Bump training-calibrated floor."""

    def __init__(self, minimum_noise_std: float):
        if not 0.0 < minimum_noise_std < 1.0:
            raise ValueError("train-calibrated residual noise RMS must lie in (0,1)")
        minimum_beta = minimum_noise_std**2
        self.betas = tuple(minimum_beta ** (1.0 - i / 3.0) for i in range(4))
        self.alphas_cumprod = tuple(np.cumprod(1.0 - np.array(self.betas)).tolist())


def refiner_loss(model, sample, policy, scheduler, generator):
    current = close_boundary(sample["current"], policy, gamma=model.gamma)
    residual = (sample["target"] - current) / model.residual_scale
    noise = torch.randn(residual.shape, device=residual.device, generator=generator)
    levels = torch.randint(
        4, (current.shape[0],), device=current.device, generator=generator
    )
    candidate = scheduler.add_noise(residual, noise, levels)
    target = scheduler.velocity_target(residual, noise, levels)
    velocity = model.velocity(current, candidate, levels, sample)
    loss = weighted_scaled_mse(
        velocity,
        target,
        sample["node_weights"],
        normal_node_mask(sample["node_type"], sample["node_mask"]),
        torch.ones(4, device=current.device),
    )
    return loss, int(levels.item())


@torch.no_grad()
def refiner_step(model, sample, current, policy, scheduler, generator):
    current = close_boundary(current, policy, gamma=model.gamma)
    candidate = torch.randn(current.shape, device=current.device, generator=generator)
    for level in (3, 2, 1, 0):
        levels = torch.full(
            (current.shape[0],), level, device=current.device, dtype=torch.long
        )
        velocity = model.velocity(current, candidate, levels, sample)
        noise = (
            torch.randn(current.shape, device=current.device, generator=generator)
            if level
            else None
        )
        candidate = scheduler.step(velocity, level, candidate, noise=noise)
    return close_boundary(
        current + candidate * model.residual_scale, policy, gamma=model.gamma
    )


def geometry_descriptor(nodes: np.ndarray, node_type: np.ndarray, mach: float):
    """Mach plus lower/upper wall profiles, independent of all state values."""
    nodes = np.asarray(nodes, dtype=np.float64)
    wall = nodes[np.asarray(node_type).reshape(-1) == 1]
    if len(wall) < 4:
        raise ValueError("wall geometry is required for mapped PCA")
    x = np.linspace(nodes[:, 0].min(), nodes[:, 0].max(), 65)
    # Top and bottom walls are separated by the channel mid-height.
    middle = (nodes[:, 1].min() + nodes[:, 1].max()) / 2
    profiles = []
    for side in (wall[wall[:, 1] < middle], wall[wall[:, 1] >= middle]):
        if len(side) < 2:
            raise ValueError("both channel walls must be represented")
        order = np.argsort(side[:, 0], kind="stable")
        profiles.append(np.interp(x, side[order, 0], side[order, 1]))
    return np.r_[mach, profiles[0], profiles[1]]


def nearest_training_geometry(target, train_descriptors: Mapping[str, np.ndarray]):
    """Frozen equally weighted condition and wall-profile distances."""
    keys = sorted(train_descriptors, key=int)
    bank = np.stack([train_descriptors[k] for k in keys])
    scale = np.maximum(bank.std(axis=0), 1e-6)
    squared = ((bank - np.asarray(target)) / scale) ** 2
    distances = squared[:, 0] + squared[:, 1:].mean(axis=1)
    index = int(np.argmin(distances))
    return keys[index], float(distances[index])


def channel_coordinates(nodes, node_type):
    nodes = np.asarray(nodes, dtype=np.float64)
    desc = geometry_descriptor(nodes, node_type, 0.0)
    x0, x1 = nodes[:, 0].min(), nodes[:, 0].max()
    xgrid = np.linspace(x0, x1, 65)
    lower = np.interp(nodes[:, 0], xgrid, desc[1:66])
    upper = np.interp(nodes[:, 0], xgrid, desc[66:])
    if np.any(upper <= lower):
        raise ValueError("invalid channel height")
    return np.c_[
        (nodes[:, 0] - x0) / (x1 - x0), (nodes[:, 1] - lower) / (upper - lower)
    ]


def geometry_remap(source_nodes, source_types, target_nodes, target_types):
    """Four-neighbor inverse-distance map within each physical node type."""
    source = channel_coordinates(source_nodes, source_types)
    target = channel_coordinates(target_nodes, target_types)
    source_types, target_types = (
        np.asarray(source_types).ravel(),
        np.asarray(target_types).ravel(),
    )
    indices = np.empty((len(target), 4), dtype=np.int64)
    weights = np.zeros((len(target), 4), dtype=np.float64)
    for kind in np.unique(target_types):
        src = np.flatnonzero(source_types == kind)
        dst = np.flatnonzero(target_types == kind)
        if len(src) == 0:
            raise ValueError("source training mesh lacks a target node type")
        count = min(4, len(src))
        distance, near = cKDTree(source[src]).query(target[dst], k=count)
        distance, near = (
            distance.reshape(len(dst), count),
            near.reshape(len(dst), count),
        )
        local = 1.0 / np.maximum(distance, 1e-12) ** 2
        exact = distance[:, 0] < 1e-12
        local[exact] = 0
        local[exact, 0] = 1
        local /= local.sum(axis=1, keepdims=True)
        indices[dst] = src[near[:, :1]]
        indices[dst, :count] = src[near]
        weights[dst, :count] = local
    return indices, weights


class MappedTrainingPCA:
    """PCA of one training neighbor remapped to the known evaluation geometry."""

    def __init__(self, snapshots, normalization, node_weights, rank, device):
        z = (
            np.asarray(snapshots, dtype=np.float64) - normalization.state_mean
        ) / normalization.state_scale
        self.rank = int(rank)
        mean = z.mean(axis=0)
        centered = (z - mean).reshape(len(z), -1)
        weights = (
            np.asarray(node_weights, dtype=np.float64)
            .reshape(len(mean), -1)
            .sum(axis=1)
        )
        weights /= weights.sum()
        sqrt_weight = np.sqrt(np.repeat(weights / 4, 4))
        weighted = centered * sqrt_weight
        eigenvalues, vectors = np.linalg.eigh(weighted @ weighted.T)
        order = np.argsort(eigenvalues)[::-1]
        tolerance = max(float(eigenvalues.max()) * 1e-12, np.finfo(float).eps)
        retained = order[eigenvalues[order] > tolerance][:rank]
        basis = vectors[:, retained].T @ weighted / np.sqrt(eigenvalues[retained, None])
        self.effective_rank = len(retained)
        self.mean = torch.tensor(mean, device=device, dtype=torch.float32)
        self.scale = torch.tensor(
            normalization.state_scale, device=device, dtype=torch.float32
        )
        self.offset = torch.tensor(
            normalization.state_mean, device=device, dtype=torch.float32
        )
        self.sqrt_weight = torch.tensor(sqrt_weight, device=device, dtype=torch.float32)
        self.basis = torch.tensor(basis, device=device, dtype=torch.float32)

    def __call__(self, state):
        normalized = (state - self.offset) / self.scale
        centered = (normalized - self.mean).flatten(start_dim=1)
        coordinates = (centered * self.sqrt_weight) @ self.basis.T
        projected_weighted = coordinates @ self.basis
        # Proxy weights in the audited bundle are strictly positive.
        if torch.any(self.sqrt_weight <= 0):
            raise ValueError("mapped PCA requires positive proxy weights")
        reconstructed = (projected_weighted / self.sqrt_weight).reshape_as(
            normalized
        ) + self.mean
        return reconstructed * self.scale + self.offset


def learning_rate(step: int, total: int) -> float:
    warmup = max(2, round(0.02 * total))
    if step < warmup:
        factor = 0.1 + 0.9 * step / (warmup - 1)
    else:
        progress = min(1.0, (step - warmup) / max(1, total - warmup - 1))
        factor = 0.02 + 0.98 * 0.5 * (1.0 + math.cos(math.pi * progress))
    return 1e-3 * factor
