#!/usr/bin/env python3
"""Audit train-only temporal linear reducibility for the Supersonic Bump.

This is a descriptive, per-trajectory POD audit on native meshes.  It does not
construct a deployable cross-trajectory projector and never reads development
or historical-test state arrays.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import shutil
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.time_dependent_no.train_pcno_bump_scaling import load_split_manifest
from utility.time_dependent_no.euler2d import conservative_to_primitive
from utility.time_dependent_no.euler2d_metrics import (
    boundary_leakage_metrics,
    positivity_metrics,
    shock_centroid_distance,
    shock_indicator,
    shock_smearing_metrics,
)
from utility.time_dependent_no.pcno_euler2d import (
    Euler2DNormalization,
    PCNOEuler2DShardStore,
    fit_normalization,
)

CONTRACT_SCHEMA = "time_dependent_no.b5_bump_solver_free_transfer_contract.v1"
SUMMARY_SCHEMA = "time_dependent_no.b5_bump_reducibility_summary.v1"
SOURCE_SCHEMA = "time_dependent_no.b5_bump_reducibility_source_manifest.v1"
FINAL_HASH_SCHEMA = "time_dependent_no.b5_bump_reducibility_final_hash_manifest.v1"
DEFAULT_CONTRACT = (
    REPO_ROOT
    / "docs"
    / "time_dependent_no"
    / "B5_BUMP_SOLVER_FREE_TRANSFER_CONTRACT.json"
)
NUM_FIELDS = 4
BLUE = "#0072B2"
ORANGE = "#E69F00"
GREY = "#999999"
OUTPUT_FILES = {
    "bump_temporal_reducibility.pdf",
    "bump_temporal_reducibility.png",
    "reconstruction_metrics.csv",
    "reducibility_summary.json",
    "source_manifest.json",
    "trajectory_spectra.csv",
}
SOURCE_FILES = (
    "scripts/time_dependent_no/analyze_bump_solver_free_reducibility.py",
    "scripts/time_dependent_no/train_pcno_bump_scaling.py",
    "utility/time_dependent_no/pcno_euler2d.py",
    "utility/time_dependent_no/euler2d.py",
    "utility/time_dependent_no/euler2d_metrics.py",
)


@dataclass(frozen=True)
class PODSpectrum:
    """Eigendecomposition and effective ranks for one snapshot Gram matrix."""

    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    explained_variance_ratio: np.ndarray
    cumulative_explained_variance: np.ndarray
    numerical_rank: int
    zero_temporal_variance: bool
    participation_rank: float
    entropy_rank: float

    def rank_at(self, threshold: float) -> int:
        if not 0.0 < threshold <= 1.0:
            raise ValueError("variance threshold must lie in (0, 1]")
        if self.zero_temporal_variance:
            return 0
        return int(np.searchsorted(self.cumulative_explained_variance, threshold) + 1)


def canonical_sha256(value: Any) -> str:
    rendered = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return sha256(rendered).hexdigest()


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _self_hashed(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    if "canonical_payload_sha256" in result:
        raise ValueError("canonical payload hash was supplied twice")
    result["canonical_payload_sha256"] = canonical_sha256(result)
    return result


def _file_record(path: Path, *, root: Path = REPO_ROOT) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"source file escapes repository: {path}") from error
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"source is absent or aliased: {relative}")
    return {
        "relative_path": relative,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


SOURCE_AT_IMPORT = _file_record(Path(__file__))


def load_contract(path: Path = DEFAULT_CONTRACT) -> dict[str, Any]:
    """Load and close the immutable Stage 0 B5 contract."""

    if path.resolve() != DEFAULT_CONTRACT.resolve():
        raise ValueError("B5 Stage 0 requires the repository contract path")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != CONTRACT_SCHEMA:
        raise ValueError("unsupported B5 transfer contract")
    claimed = payload.get("canonical_payload_sha256")
    canonical = dict(payload)
    canonical.pop("canonical_payload_sha256", None)
    if claimed != canonical_sha256(canonical):
        raise ValueError("B5 contract canonical payload digest does not close")
    if payload.get("remote_execution_authorized") is not False:
        raise ValueError("Stage 0 contract unexpectedly authorizes remote execution")
    if payload.get("historical_test_population_opened") is not False:
        raise ValueError("historical-test population must remain sealed")
    stage0 = payload.get("stage0_reducibility")
    if not isinstance(stage0, Mapping):
        raise TypeError("B5 contract lacks Stage 0 reducibility settings")
    if stage0.get("execution_status") != "authorized_after_identity_checks":
        raise ValueError("B5 Stage 0 reducibility is not authorized")

    prereg = payload.get("preregistration")
    if not isinstance(prereg, Mapping):
        raise TypeError("B5 contract lacks its preregistration binding")
    prereg_path = REPO_ROOT / str(prereg.get("relative_path"))
    if sha256_file(prereg_path) != prereg.get("file_sha256"):
        raise ValueError("B5 preregistration hash does not close")
    return payload


def normalized_node_weights(
    num_nodes: int,
    *,
    metric: str,
    proxy_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Return normalized native-node weights for a declared diagnostic metric."""

    if num_nodes < 1:
        raise ValueError("num_nodes must be positive")
    if metric == "uniform_normalized_state":
        return np.full(num_nodes, 1.0 / num_nodes, dtype=np.float64)
    if metric != "proxy_mass_normalized_state":
        raise ValueError(f"unknown POD metric: {metric}")
    if proxy_weights is None:
        raise ValueError("proxy weights are required for the proxy-mass metric")
    weights = np.asarray(proxy_weights, dtype=np.float64)
    if weights.ndim == 2:
        weights = np.sum(weights, axis=-1)
    if weights.shape != (num_nodes,):
        raise ValueError("proxy weights must have shape [N] or [N,K]")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("proxy weights must be finite and nonnegative")
    total = float(np.sum(weights))
    if total <= 0.0:
        raise ValueError("proxy weights must have positive total")
    return weights / total


def temporal_snapshot_grams(
    normalized_states: np.ndarray,
    *,
    start: int,
    stop: int,
    proxy_weights: np.ndarray,
    block_nodes: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Compute uniform and proxy-weighted centered time-time Gram matrices."""

    values = np.asarray(normalized_states, dtype=np.float64)
    if values.ndim != 3 or values.shape[-1] != NUM_FIELDS:
        raise ValueError("normalized states must have shape [T,N,4]")
    if not np.all(np.isfinite(values)):
        raise ValueError("normalized states contain nonfinite values")
    if start < 0 or stop > values.shape[0] or stop - start < 2:
        raise ValueError("snapshot frame interval must contain at least two frames")
    if block_nodes < 1:
        raise ValueError("block_nodes must be positive")

    snapshots = values[start:stop]
    center = np.mean(snapshots, axis=0, dtype=np.float64)
    metrics = (
        "uniform_normalized_state",
        "proxy_mass_normalized_state",
    )
    weights = {
        metric: normalized_node_weights(
            values.shape[1], metric=metric, proxy_weights=proxy_weights
        )
        for metric in metrics
    }
    grams = {
        metric: np.zeros((snapshots.shape[0], snapshots.shape[0]), dtype=np.float64)
        for metric in metrics
    }
    for first in range(0, values.shape[1], block_nodes):
        last = min(first + block_nodes, values.shape[1])
        centered = snapshots[:, first:last] - center[None, first:last]
        for metric in metrics:
            scale = np.sqrt(weights[metric][first:last] / NUM_FIELDS)
            block = centered * scale[None, :, None]
            flat = block.reshape(snapshots.shape[0], -1)
            grams[metric] += flat @ flat.T
    for metric in metrics:
        grams[metric] = 0.5 * (grams[metric] + grams[metric].T)
    return center, grams


def pod_from_gram(gram: np.ndarray) -> PODSpectrum:
    """Return a stable POD eigendecomposition without truncating threshold ranks."""

    values = np.asarray(gram, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1] or values.shape[0] < 2:
        raise ValueError("snapshot Gram matrix must be square with size at least two")
    if not np.all(np.isfinite(values)):
        raise ValueError("snapshot Gram matrix contains nonfinite values")
    values = 0.5 * (values + values.T)
    eigenvalues, eigenvectors = np.linalg.eigh(values)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    largest = max(float(eigenvalues[0]), 0.0)
    negative_tolerance = max(largest, 1.0) * values.shape[0] * np.finfo(float).eps
    if float(np.min(eigenvalues)) < -100.0 * negative_tolerance:
        raise ValueError("snapshot Gram matrix is materially non-positive-semidefinite")
    eigenvalues = np.maximum(eigenvalues, 0.0)
    total = float(np.sum(eigenvalues))
    if total <= negative_tolerance:
        zeros = np.zeros_like(eigenvalues)
        return PODSpectrum(
            eigenvalues=zeros,
            eigenvectors=eigenvectors,
            explained_variance_ratio=zeros,
            cumulative_explained_variance=zeros,
            numerical_rank=0,
            zero_temporal_variance=True,
            participation_rank=0.0,
            entropy_rank=0.0,
        )

    numerical_tolerance = max(largest * 1.0e-12, np.finfo(float).eps * total)
    numerical_rank = int(np.count_nonzero(eigenvalues > numerical_tolerance))
    eigenvalues[numerical_rank:] = 0.0
    total = float(np.sum(eigenvalues))
    explained = eigenvalues / total
    cumulative = np.cumsum(explained)
    cumulative[-1] = 1.0
    positive = explained > 0.0
    participation = float(1.0 / np.sum(np.square(explained[positive])))
    entropy = float(np.exp(-np.sum(explained[positive] * np.log(explained[positive]))))

    for component in range(eigenvectors.shape[1]):
        pivot = int(np.argmax(np.abs(eigenvectors[:, component])))
        if eigenvectors[pivot, component] < 0.0:
            eigenvectors[:, component] *= -1.0
    return PODSpectrum(
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        explained_variance_ratio=explained,
        cumulative_explained_variance=cumulative,
        numerical_rank=numerical_rank,
        zero_temporal_variance=False,
        participation_rank=participation,
        entropy_rank=entropy,
    )


def prefix_reconstructions(
    normalized_states: np.ndarray,
    *,
    prefix_start: int,
    prefix_stop: int,
    future_stop: int,
    center: np.ndarray,
    spectrum: PODSpectrum,
    node_weights: np.ndarray,
    ranks: Sequence[int],
    block_nodes: int,
) -> dict[int, tuple[int, np.ndarray]]:
    """Project future states into a weighted POD basis fitted on the prefix."""

    values = np.asarray(normalized_states, dtype=np.float64)
    weights = np.asarray(node_weights, dtype=np.float64)
    requested = tuple(int(rank) for rank in ranks)
    if not requested or min(requested) < 1 or len(set(requested)) != len(requested):
        raise ValueError("reconstruction ranks must be unique positive integers")
    if (
        prefix_start != 0
        or not prefix_start < prefix_stop < future_stop <= values.shape[0]
    ):
        raise ValueError("invalid prefix/future frame intervals")
    if center.shape != values.shape[1:] or weights.shape != (values.shape[1],):
        raise ValueError("POD center or weights do not match the states")

    prefix = values[prefix_start:prefix_stop]
    future = values[prefix_stop:future_stop]
    maximum = min(max(requested), spectrum.numerical_rank)
    if maximum == 0:
        baseline = np.broadcast_to(center, future.shape).copy()
        return {rank: (0, baseline.copy()) for rank in requested}

    eigenvalues = spectrum.eigenvalues[:maximum]
    temporal_modes = spectrum.eigenvectors[:, :maximum]
    if temporal_modes.shape[0] != prefix.shape[0]:
        raise ValueError("prefix spectrum has the wrong number of snapshots")
    basis_weighted = np.zeros((maximum, values.shape[1], NUM_FIELDS), dtype=np.float64)
    coefficients = np.zeros((future.shape[0], maximum), dtype=np.float64)
    for first in range(0, values.shape[1], block_nodes):
        last = min(first + block_nodes, values.shape[1])
        scale = np.sqrt(weights[first:last] / NUM_FIELDS)
        prefix_block = prefix[:, first:last] - center[None, first:last]
        prefix_weighted = prefix_block * scale[None, :, None]
        right = temporal_modes.T @ prefix_weighted.reshape(prefix.shape[0], -1)
        right /= np.sqrt(eigenvalues)[:, None]
        basis_weighted[:, first:last] = right.reshape(maximum, last - first, NUM_FIELDS)
        future_block = future[:, first:last] - center[None, first:last]
        future_weighted = future_block * scale[None, :, None]
        coefficients += future_weighted.reshape(future.shape[0], -1) @ right.T

    outputs: dict[int, tuple[int, np.ndarray]] = {}
    for rank in requested:
        effective = min(rank, maximum)
        reconstruction = np.broadcast_to(center, future.shape).copy()
        if effective:
            for first in range(0, values.shape[1], block_nodes):
                last = min(first + block_nodes, values.shape[1])
                scale = np.sqrt(weights[first:last] / NUM_FIELDS)
                weighted = coefficients[:, :effective] @ basis_weighted[
                    :effective, first:last
                ].reshape(effective, -1)
                positive = scale > 0.0
                centered = np.zeros(
                    (future.shape[0], last - first, NUM_FIELDS), dtype=np.float64
                )
                if np.any(positive):
                    centered[:, positive, :] = (
                        weighted.reshape(future.shape[0], last - first, NUM_FIELDS)[
                            :, positive, :
                        ]
                        / scale[None, positive, None]
                    )
                reconstruction[:, first:last] += centered
        outputs[rank] = (effective, reconstruction)
    return outputs


def _weighted_reconstruction_errors(
    reconstruction: np.ndarray,
    target: np.ndarray,
    center: np.ndarray,
    node_weights: np.ndarray,
) -> dict[str, float]:
    difference = reconstruction - target
    centered_target = target - center[None]
    error_energy = (
        np.einsum("tnc,n->t", np.square(difference), node_weights, optimize=True)
        / NUM_FIELDS
    )
    target_energy = (
        np.einsum("tnc,n->t", np.square(centered_target), node_weights, optimize=True)
        / NUM_FIELDS
    )
    denominator = float(np.sum(target_energy))
    relative = math.sqrt(float(np.sum(error_energy)) / max(denominator, 1.0e-30))
    frame_relative = np.sqrt(error_energy / np.maximum(target_energy, 1.0e-30))
    return {
        "future_relative_error": relative,
        "future_normalized_rmse_mean": float(np.mean(np.sqrt(error_energy))),
        "future_frame_relative_error_median": float(np.median(frame_relative)),
        "future_frame_relative_error_max": float(np.max(frame_relative)),
    }


def _masked_centered_relative_error(
    reconstruction: np.ndarray,
    target: np.ndarray,
    center: np.ndarray,
    mask: np.ndarray,
) -> float | None:
    selected = np.asarray(mask, dtype=bool)[..., None]
    numerator = float(
        np.sum(np.where(selected, np.square(reconstruction - target), 0.0))
    )
    denominator = float(
        np.sum(np.where(selected, np.square(target - center[None]), 0.0))
    )
    if denominator <= 0.0:
        return None
    return math.sqrt(numerator / denominator)


def _finite_or_none(value: Any) -> float | None:
    number = float(np.asarray(value))
    return number if math.isfinite(number) else None


def reconstruction_diagnostics(
    reconstruction_normalized: np.ndarray,
    target_normalized: np.ndarray,
    *,
    center: np.ndarray,
    node_weights: np.ndarray,
    normalization: Euler2DNormalization,
    geometry: Mapping[str, np.ndarray | float],
) -> dict[str, Any]:
    """Measure state error and transported-front damage after POD truncation."""

    metrics: dict[str, Any] = _weighted_reconstruction_errors(
        reconstruction_normalized, target_normalized, center, node_weights
    )
    reconstruction = (
        reconstruction_normalized * normalization.state_scale[None, None, :]
        + normalization.state_mean[None, None, :]
    )
    target = (
        target_normalized * normalization.state_scale[None, None, :]
        + normalization.state_mean[None, None, :]
    )
    reconstruction_primitive = conservative_to_primitive(
        reconstruction, gamma=normalization.gamma
    )
    target_primitive = conservative_to_primitive(target, gamma=normalization.gamma)
    edges = np.asarray(geometry["directed_edges"])
    nodes = np.asarray(geometry["nodes"], dtype=np.float64)
    node_type = np.asarray(geometry["node_type"])

    positivity = positivity_metrics(reconstruction_primitive)
    smearing = shock_smearing_metrics(
        reconstruction_primitive, target_primitive, edges, scalar_index=3
    )
    centroid = shock_centroid_distance(
        reconstruction_primitive,
        target_primitive,
        nodes,
        edges,
        scalar_index=3,
    )
    boundary = boundary_leakage_metrics(
        reconstruction_primitive, target_primitive, node_type
    )
    target_front = shock_indicator(
        target_primitive, edges, scalar_index=3, quantile=0.9
    )
    metrics.update(
        {
            "min_density": _finite_or_none(positivity["min_density"]),
            "min_pressure": _finite_or_none(positivity["min_pressure"]),
            "fraction_nonpositive_density": _finite_or_none(
                positivity["fraction_nonpositive_density"]
            ),
            "fraction_nonpositive_pressure": _finite_or_none(
                positivity["fraction_nonpositive_pressure"]
            ),
            "shock_centroid_distance_mean": _finite_or_none(np.mean(centroid)),
            "shock_centroid_distance_max": _finite_or_none(np.max(centroid)),
            "shock_thickness_ratio_mean": _finite_or_none(
                np.mean(smearing["thickness_ratio"])
            ),
            "shock_strength_ratio_mean": _finite_or_none(
                np.mean(smearing["strength_ratio"])
            ),
            "boundary_relative_l2": _finite_or_none(boundary["boundary_relative_l2"]),
            "boundary_max_abs": _finite_or_none(boundary["boundary_max_abs"]),
            "high_gradient_centered_relative_error": _masked_centered_relative_error(
                reconstruction_normalized,
                target_normalized,
                center,
                target_front,
            ),
            "smooth_centered_relative_error": _masked_centered_relative_error(
                reconstruction_normalized,
                target_normalized,
                center,
                ~target_front,
            ),
        }
    )
    return metrics


def _spectrum_rows(
    key: str,
    metric: str,
    scope: str,
    spectrum: PODSpectrum,
    *,
    mach: float,
    num_nodes: int,
    thresholds: Sequence[float],
) -> list[dict[str, Any]]:
    ranks = {
        f"rank_{1000 * threshold:g}_permille": spectrum.rank_at(threshold)
        for threshold in thresholds
    }
    rows = []
    for index, (eigenvalue, explained, cumulative) in enumerate(
        zip(
            spectrum.eigenvalues,
            spectrum.explained_variance_ratio,
            spectrum.cumulative_explained_variance,
            strict=True,
        ),
        start=1,
    ):
        rows.append(
            {
                "trajectory_key": key,
                "mach": mach,
                "num_nodes": num_nodes,
                "metric": metric,
                "basis_scope": scope,
                "component": index,
                "eigenvalue": float(eigenvalue),
                "explained_variance_ratio": float(explained),
                "cumulative_explained_variance": float(cumulative),
                "numerical_rank": spectrum.numerical_rank,
                "zero_temporal_variance": spectrum.zero_temporal_variance,
                "participation_rank": spectrum.participation_rank,
                "entropy_rank": spectrum.entropy_rank,
                **ranks,
            }
        )
    return rows


def analyze_trajectory(
    store: PCNOEuler2DShardStore,
    key: str,
    *,
    normalization: Euler2DNormalization,
    stage0: Mapping[str, Any],
    block_nodes: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """Analyze one training trajectory without sharing state vectors across meshes."""

    states = np.asarray(store.states(key), dtype=np.float64)
    expected_frames = int(stage0["model_input_frame_count"]) + 1
    if states.shape[0] != expected_frames or states.shape[-1] != NUM_FIELDS:
        raise ValueError(f"trajectory {key} has unexpected state shape {states.shape}")
    normalized = (
        states - normalization.state_mean[None, None, :]
    ) / normalization.state_scale[None, None, :]
    geometry = store.geometry_numpy(key)
    proxy_raw = np.asarray(geometry["node_weights"], dtype=np.float64)
    input_start, input_last = (int(value) for value in stage0["model_input_frames"])
    prefix_start, prefix_last = (int(value) for value in stage0["prefix_fit_frames"])
    future_start, future_last = (int(value) for value in stage0["prefix_future_frames"])
    if future_start != prefix_last + 1:
        raise ValueError("prefix and future frame intervals must be contiguous")
    oracle_center, oracle_grams = temporal_snapshot_grams(
        normalized,
        start=input_start,
        stop=input_last + 1,
        proxy_weights=proxy_raw,
        block_nodes=block_nodes,
    )
    del oracle_center
    prefix_center, prefix_grams = temporal_snapshot_grams(
        normalized,
        start=prefix_start,
        stop=prefix_last + 1,
        proxy_weights=proxy_raw,
        block_nodes=block_nodes,
    )

    metrics = tuple(str(value) for value in stage0["metrics"])
    thresholds = tuple(float(value) for value in stage0["variance_thresholds"])
    ranks = tuple(int(value) for value in stage0["reconstruction_ranks"])
    spectra_rows: list[dict[str, Any]] = []
    reconstruction_rows: list[dict[str, Any]] = []
    oracle_rank_999: dict[str, int] = {}
    mach = float(store.entry(key)["mach"])
    for metric in metrics:
        oracle = pod_from_gram(oracle_grams[metric])
        prefix = pod_from_gram(prefix_grams[metric])
        spectra_rows.extend(
            _spectrum_rows(
                key,
                metric,
                "oracle_model_inputs_0_78",
                oracle,
                mach=mach,
                num_nodes=states.shape[1],
                thresholds=thresholds,
            )
        )
        spectra_rows.extend(
            _spectrum_rows(
                key,
                metric,
                "prefix_fit_0_39",
                prefix,
                mach=mach,
                num_nodes=states.shape[1],
                thresholds=thresholds,
            )
        )
        oracle_rank_999[metric] = oracle.rank_at(0.999)
        weights = normalized_node_weights(
            states.shape[1], metric=metric, proxy_weights=proxy_raw
        )
        reconstructions = prefix_reconstructions(
            normalized,
            prefix_start=prefix_start,
            prefix_stop=prefix_last + 1,
            future_stop=future_last + 1,
            center=prefix_center,
            spectrum=prefix,
            node_weights=weights,
            ranks=ranks,
            block_nodes=block_nodes,
        )
        target = normalized[future_start : future_last + 1]
        for requested_rank in ranks:
            effective_rank, reconstruction = reconstructions[requested_rank]
            diagnostics = reconstruction_diagnostics(
                reconstruction,
                target,
                center=prefix_center,
                node_weights=weights,
                normalization=normalization,
                geometry=geometry,
            )
            reconstruction_rows.append(
                {
                    "trajectory_key": key,
                    "mach": mach,
                    "num_nodes": states.shape[1],
                    "metric": metric,
                    "requested_rank": requested_rank,
                    "effective_rank": effective_rank,
                    "zero_proxy_weight_nodes": int(np.count_nonzero(weights == 0.0)),
                    **diagnostics,
                }
            )
    return spectra_rows, reconstruction_rows, oracle_rank_999


def stage0_gate(
    uniform_oracle_ranks_999: Sequence[int],
    *,
    reference_rank: int,
    minimum_count: int,
    total_count: int,
) -> dict[str, Any]:
    ranks = tuple(int(value) for value in uniform_oracle_ranks_999)
    if len(ranks) != total_count:
        raise ValueError("Stage 0 gate received the wrong number of trajectories")
    count = sum(value > reference_rank for value in ranks)
    return {
        "reference_rank": reference_rank,
        "trajectories_above_reference_rank": count,
        "minimum_required": minimum_count,
        "total_trajectories": total_count,
        "passed": count >= minimum_count,
    }


def _numeric_summary(values: Sequence[Any]) -> dict[str, float | int | None]:
    finite = np.asarray(
        [
            float(value)
            for value in values
            if value is not None and math.isfinite(float(value))
        ],
        dtype=np.float64,
    )
    if finite.size == 0:
        return {"count": 0, "median": None, "q25": None, "q75": None}
    return {
        "count": int(finite.size),
        "median": float(np.median(finite)),
        "q25": float(np.quantile(finite, 0.25)),
        "q75": float(np.quantile(finite, 0.75)),
    }


def _require_exact_trajectory_coverage(
    rows: Sequence[Mapping[str, Any]],
    audit_keys: Sequence[str],
    *,
    context: str,
) -> None:
    expected = tuple(str(key) for key in audit_keys)
    observed = tuple(str(row["trajectory_key"]) for row in rows)
    if len(observed) != len(expected) or set(observed) != set(expected):
        raise ValueError(
            f"{context} does not cover each fixed audit trajectory exactly once: "
            f"expected {len(expected)}, observed {len(observed)}"
        )


def aggregate_results(
    spectrum_rows: Sequence[Mapping[str, Any]],
    reconstruction_rows: Sequence[Mapping[str, Any]],
    *,
    audit_keys: Sequence[str],
    stage0: Mapping[str, Any],
) -> dict[str, Any]:
    """Aggregate fixed metrics while retaining paired trajectory-level CSVs."""

    metrics = tuple(str(value) for value in stage0["metrics"])
    aggregate: dict[str, Any] = {}
    for metric in metrics:
        oracle_first = [
            row
            for row in spectrum_rows
            if row["metric"] == metric
            and row["basis_scope"] == "oracle_model_inputs_0_78"
            and row["component"] == 1
        ]
        _require_exact_trajectory_coverage(
            oracle_first,
            audit_keys,
            context=f"oracle spectrum rows for metric {metric}",
        )
        metric_summary: dict[str, Any] = {
            "oracle_rank_99_9": _numeric_summary(
                [row["rank_999_permille"] for row in oracle_first]
            ),
            "oracle_participation_rank": _numeric_summary(
                [row["participation_rank"] for row in oracle_first]
            ),
            "oracle_entropy_rank": _numeric_summary(
                [row["entropy_rank"] for row in oracle_first]
            ),
            "prefix_future_by_rank": {},
        }
        for rank in stage0["reconstruction_ranks"]:
            selected = [
                row
                for row in reconstruction_rows
                if row["metric"] == metric and row["requested_rank"] == rank
            ]
            _require_exact_trajectory_coverage(
                selected,
                audit_keys,
                context=f"reconstruction rows for metric {metric}, rank {rank}",
            )
            metric_summary["prefix_future_by_rank"][str(rank)] = {
                "relative_error": _numeric_summary(
                    [row["future_relative_error"] for row in selected]
                ),
                "shock_centroid_distance": _numeric_summary(
                    [row["shock_centroid_distance_mean"] for row in selected]
                ),
                "shock_thickness_ratio": _numeric_summary(
                    [row["shock_thickness_ratio_mean"] for row in selected]
                ),
                "shock_strength_ratio": _numeric_summary(
                    [row["shock_strength_ratio_mean"] for row in selected]
                ),
                "fraction_nonpositive_pressure": _numeric_summary(
                    [row["fraction_nonpositive_pressure"] for row in selected]
                ),
            }
        aggregate[metric] = metric_summary

    uniform = [
        row
        for row in spectrum_rows
        if row["metric"] == "uniform_normalized_state"
        and row["basis_scope"] == "oracle_model_inputs_0_78"
        and row["component"] == 1
    ]
    _require_exact_trajectory_coverage(
        uniform,
        audit_keys,
        context="uniform Stage 0 gate rows",
    )
    gate_rule = stage0["pass_rule"]
    gate = stage0_gate(
        [row["rank_999_permille"] for row in uniform],
        reference_rank=7,
        minimum_count=int(
            gate_rule["minimum_trajectories_with_uniform_oracle_rank_99_9_above_7"]
        ),
        total_count=int(gate_rule["total_trajectories"]),
    )
    return {
        "audit_trajectory_count": len(audit_keys),
        "aggregate": aggregate,
        "linear_reducibility_contrast_gate": gate,
    }


def render_reducibility_figure(
    pdf_path: Path,
    png_path: Path,
    spectrum_rows: Sequence[Mapping[str, Any]],
    reconstruction_rows: Sequence[Mapping[str, Any]],
    *,
    gate: Mapping[str, Any],
    max_components: int = 32,
) -> None:
    """Render the compact, paper-compatible B5 temporal-POD diagnostic."""

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.0,
            "legend.fontsize": 7.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(7.0, 2.65), constrained_layout=True)
    styles = {
        "uniform_normalized_state": (BLUE, "Equal-node metric"),
        "proxy_mass_normalized_state": (ORANGE, "Proxy-mass metric"),
    }
    for metric, (color, label) in styles.items():
        oracle = [
            row
            for row in spectrum_rows
            if row["metric"] == metric
            and row["basis_scope"] == "oracle_model_inputs_0_78"
            and int(row["component"]) <= max_components
        ]
        components = sorted({int(row["component"]) for row in oracle})
        medians, lowers, uppers = [], [], []
        for component in components:
            values = [
                float(row["cumulative_explained_variance"])
                for row in oracle
                if int(row["component"]) == component
            ]
            medians.append(np.median(values))
            lowers.append(np.quantile(values, 0.25))
            uppers.append(np.quantile(values, 0.75))
        axes[0].plot(components, medians, color=color, linewidth=1.6, label=label)
        axes[0].fill_between(components, lowers, uppers, color=color, alpha=0.16)

        selected = [row for row in reconstruction_rows if row["metric"] == metric]
        ranks = sorted({int(row["requested_rank"]) for row in selected})
        errors = [
            np.median(
                [
                    float(row["future_relative_error"])
                    for row in selected
                    if int(row["requested_rank"]) == rank
                ]
            )
            for rank in ranks
        ]
        axes[1].plot(
            ranks,
            errors,
            color=color,
            marker="o",
            markersize=3.5,
            linewidth=1.4,
            label=label,
        )

    axes[0].axhline(0.999, color=GREY, linewidth=0.8, linestyle=":")
    axes[0].axvline(7, color=GREY, linewidth=0.8, linestyle="--")
    axes[0].set_xlim(1, max_components)
    axes[0].set_ylim(0.0, 1.01)
    axes[0].set_xlabel("Per-trajectory POD rank")
    axes[0].set_ylabel("Cumulative explained variance")
    axes[0].set_title("Oracle temporal spectrum (frames 0--78)")
    axes[0].grid(color="#DDDDDD", linewidth=0.5, alpha=0.7)
    axes[0].text(
        0.98,
        0.05,
        f"rank > 7: {gate['trajectories_above_reference_rank']}/{gate['total_trajectories']}",
        transform=axes[0].transAxes,
        ha="right",
        va="bottom",
        fontsize=7.3,
    )
    axes[0].text(
        0.015,
        0.98,
        "(a)",
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        fontweight="bold",
    )

    axes[1].set_xscale("log", base=2)
    axes[1].set_yscale("log")
    axes[1].set_xticks([2, 4, 7, 16, 32], labels=["2", "4", "7", "16", "32"])
    axes[1].set_xlabel("Prefix POD rank")
    axes[1].set_ylabel("Median future relative error")
    axes[1].set_title("Prefix fit 0--39, evaluate 40--78")
    axes[1].grid(color="#DDDDDD", linewidth=0.5, alpha=0.7)
    axes[1].legend(loc="best", frameon=False)
    axes[1].text(
        0.015,
        0.98,
        "(b)",
        transform=axes[1].transAxes,
        ha="left",
        va="top",
        fontweight="bold",
    )
    figure.savefig(pdf_path, bbox_inches="tight")
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path.name}")
    fieldnames = list(rows[0])
    if any(list(row) != fieldnames for row in rows):
        raise ValueError(f"CSV rows do not share one schema: {path.name}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _source_manifest(
    *,
    contract_path: Path,
    split_path: Path,
    data_manifest_sha256: str,
) -> dict[str, Any]:
    records = [_file_record(REPO_ROOT / relative) for relative in SOURCE_FILES]
    contract_record = _file_record(contract_path)
    split_record = _file_record(split_path)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    prereg_record = _file_record(
        REPO_ROOT / str(contract["preregistration"]["relative_path"])
    )
    return _self_hashed(
        {
            "schema": SOURCE_SCHEMA,
            "experiment_id": contract["experiment_id"],
            "sources": records,
            "contract": contract_record,
            "preregistration": prereg_record,
            "split_manifest": split_record,
            "data_manifest": {
                "relative_to_data_root": "manifest.json",
                "sha256": data_manifest_sha256,
            },
        }
    )


def _reverify_sources(source_manifest: Mapping[str, Any]) -> None:
    for record in source_manifest["sources"]:
        path = REPO_ROOT / str(record["relative_path"])
        if _file_record(path) != record:
            raise ValueError(
                f"source changed during analysis: {record['relative_path']}"
            )
    for name in ("contract", "preregistration", "split_manifest"):
        record = source_manifest[name]
        if _file_record(REPO_ROOT / str(record["relative_path"])) != record:
            raise ValueError(f"{name} changed during analysis")
    if _file_record(Path(__file__)) != SOURCE_AT_IMPORT:
        raise ValueError("analysis script changed after import")


def _packet_file_record(path: Path, packet_root: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"packet output is absent or aliased: {path.name}")
    relative = path.resolve().relative_to(packet_root.resolve()).as_posix()
    return {
        "relative_path": relative,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def write_reducibility_packet(
    output_dir: Path,
    *,
    data_root: Path,
    contract_path: Path = DEFAULT_CONTRACT,
    block_nodes: int = 2048,
) -> dict[str, Any]:
    """Run Stage 0B and atomically publish a provenance-bound result packet."""

    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    if output_dir.parent.is_symlink():
        raise ValueError("output parent must not be a symlink")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    contract = load_contract(contract_path)
    split_path = REPO_ROOT / str(contract["split"]["relative_path"])
    if sha256_file(split_path) != contract["split"]["file_sha256"]:
        raise ValueError("registered split file hash changed")
    split = load_split_manifest(split_path, require_registered=True)
    if (
        split["canonical_payload_sha256"]
        != contract["split"]["canonical_payload_sha256"]
    ):
        raise ValueError("split canonical digest disagrees with the B5 contract")
    if split["partition_digest"] != contract["split"]["partition_digest"]:
        raise ValueError("split partition digest disagrees with the B5 contract")

    stage0 = contract["stage0_reducibility"]
    audit_keys = [str(value) for value in stage0["audit_keys"]]
    registered_audit = [
        str(value)
        for value in split["nested_exposure"]["subsets"][str(len(audit_keys))]
    ]
    if audit_keys != registered_audit:
        raise ValueError("B5 audit keys do not match the registered nested subset")
    train_keys = [str(value) for value in split["split"]["train_pool_keys"]]
    development_keys = [str(value) for value in split["split"]["open_validation_keys"]]
    if not set(audit_keys).issubset(train_keys):
        raise ValueError("B5 reducibility audit attempts to leave the train population")

    with PCNOEuler2DShardStore(data_root, max_cached_trajectories=1) as store:
        expected_data_digest = contract["split"]["prepared_shard_manifest_sha256"]
        if store.manifest_digest != expected_data_digest:
            raise ValueError("prepared shard manifest digest does not close")
        if set(store.keys) != set(train_keys) | set(development_keys):
            raise ValueError(
                "prepared shard keys do not close the frozen development split"
            )
        source_manifest = _source_manifest(
            contract_path=contract_path,
            split_path=split_path,
            data_manifest_sha256=store.manifest_digest,
        )
        normalization = fit_normalization(
            store,
            train_keys,
            step_stride=1,
            time_stride=1,
            gamma=float(store.manifest.get("gamma", 1.4)),
        )
        spectrum_rows: list[dict[str, Any]] = []
        reconstruction_rows: list[dict[str, Any]] = []
        for key in audit_keys:
            spectra, reconstructions, _ = analyze_trajectory(
                store,
                key,
                normalization=normalization,
                stage0=stage0,
                block_nodes=block_nodes,
            )
            spectrum_rows.extend(spectra)
            reconstruction_rows.extend(reconstructions)
        aggregate = aggregate_results(
            spectrum_rows,
            reconstruction_rows,
            audit_keys=audit_keys,
            stage0=stage0,
        )
        if store.manifest_digest != expected_data_digest:
            raise ValueError("prepared shard manifest changed during analysis")

    summary = _self_hashed(
        {
            "schema": SUMMARY_SCHEMA,
            "experiment_id": contract["experiment_id"],
            "scientific_role": contract["scientific_role"],
            "contract_canonical_payload_sha256": contract["canonical_payload_sha256"],
            "source_manifest_canonical_payload_sha256": source_manifest[
                "canonical_payload_sha256"
            ],
            "fit_scope": {
                "normalization_population": "all_256_registered_training_trajectories",
                "pod_population": "registered_nested_training_subset_n32",
                "model_input_frames": stage0["model_input_frames"],
                "prefix_fit_frames": stage0["prefix_fit_frames"],
                "prefix_future_frames": stage0["prefix_future_frames"],
                "state_coordinates": stage0["formulas"]["state_coordinates"],
                "metrics": stage0["metrics"],
                "formulas": stage0["formulas"],
                "normalization": normalization.to_dict(),
                "normalization_canonical_payload_sha256": canonical_sha256(
                    normalization.to_dict()
                ),
            },
            "access": {
                "state_arrays_opened_in_analysis": True,
                "state_array_populations_opened": ["train"],
                "normalization_state_keys_opened": train_keys,
                "pod_state_keys_opened": audit_keys,
                "development_state_arrays_opened": False,
                "prospective_population_opened": False,
                "historical_test_population_opened": False,
            },
            "results": aggregate,
            "interpretation": {
                "supported_scope": (
                    "per-trajectory temporal linear compressibility on fixed native "
                    "training meshes"
                ),
                "not_supported": [
                    "a deployable cross-trajectory PCA projector",
                    "global linear or nonlinear manifold dimension",
                    "physical conservation under proxy weights",
                    "solver-relative displaced-state dynamics",
                    "prospective or historical-test generalization",
                    "a universal ranking of corrective mechanisms",
                ],
                "projection_note": (
                    "NACA0012 PCA projection is treated as a problem-specific special "
                    "case; B5 tests a distinct geometry-dependent transported-structure "
                    "setting."
                ),
            },
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "platform": platform.platform(),
                "block_nodes": block_nodes,
            },
        }
    )

    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    try:
        _write_csv(staging / "trajectory_spectra.csv", spectrum_rows)
        _write_csv(staging / "reconstruction_metrics.csv", reconstruction_rows)
        _write_json(staging / "source_manifest.json", source_manifest)
        _write_json(staging / "reducibility_summary.json", summary)
        render_reducibility_figure(
            staging / "bump_temporal_reducibility.pdf",
            staging / "bump_temporal_reducibility.png",
            spectrum_rows,
            reconstruction_rows,
            gate=aggregate["linear_reducibility_contrast_gate"],
        )
        _reverify_sources(source_manifest)
        if (
            sha256_file(data_root / "manifest.json")
            != contract["split"]["prepared_shard_manifest_sha256"]
        ):
            raise ValueError("prepared shard manifest changed before publication")
        observed = {path.name for path in staging.iterdir()}
        if observed != OUTPUT_FILES:
            raise ValueError(f"incomplete B5 packet before sealing: {sorted(observed)}")
        final_manifest = {
            "schema": FINAL_HASH_SCHEMA,
            "experiment_id": contract["experiment_id"],
            "self_hash_excluded": True,
            "development_state_arrays_opened": False,
            "prospective_population_opened": False,
            "historical_test_population_opened": False,
            "files": {
                name: _packet_file_record(staging / name, staging)
                for name in sorted(OUTPUT_FILES)
            },
        }
        _write_json(staging / "final_hash_manifest.json", final_manifest)
        os.replace(staging, output_dir)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--block-nodes", type=int, default=2048)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = write_reducibility_packet(
        args.output_dir,
        data_root=args.data_root,
        block_nodes=args.block_nodes,
    )
    gate = summary["results"]["linear_reducibility_contrast_gate"]
    print(
        json.dumps(
            {
                "experiment_id": summary["experiment_id"],
                "output_dir": str(args.output_dir),
                "linear_reducibility_contrast_gate": gate,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
