#!/usr/bin/env python3
"""Visualize train-only variance concentration in the NACA0012 state path.

The primary diagnostic uses the exact 238 train-current states seen by the
one-step PCNO objective, train-fitted per-field normalization, and the model's
uniform-coordinate metric.  Area-weighted state PCA and complete-BDF2-input
PCA are reported only as robustness checks.  No protected population is read.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
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

from scripts.time_dependent_no.train_pcno_naca0012 import _load_dataset
from utility.time_dependent_no.pcno_naca0012 import (
    BASELINE_CONTRACT_SHA256,
    NACANormalization,
    load_naca_baseline_contract,
)
from utility.time_dependent_no.su2_restart_contract import sha256_file

SUMMARY_SCHEMA = "time_dependent_no.naca0012_train_pca_summary.v1"
ARRAYS_SCHEMA = "time_dependent_no.naca0012_train_pca_arrays.v1"
FINAL_HASH_SCHEMA = "time_dependent_no.naca0012_train_pca_final_hash_manifest.v1"
OUTPUT_FILES = {
    "naca0012_train_pca.pdf",
    "naca0012_train_pca.png",
    "pca_arrays.npz",
    "pca_summary.json",
}
TRAIN_FRAME_COUNT = 240
TRAIN_TRANSITION_COUNT = 238
NUM_FIELDS = 5
DEFAULT_COMPONENTS = 16
DEFAULT_BLOCK_NODES = 1024
VARIANCE_THRESHOLDS = (0.90, 0.99, 0.999)

# Okabe--Ito colors; the chronological scatter uses perceptually uniform viridis.
BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
VERMILLION = "#D55E00"
GREY = "#999999"


@dataclass(frozen=True)
class SnapshotPCA:
    """PCA spectrum and chronological sample scores from a snapshot Gram matrix."""

    eigenvalues: np.ndarray
    explained_variance_ratio: np.ndarray
    cumulative_explained_variance: np.ndarray
    scores: np.ndarray
    total_sum_squared_deviation: float
    participation_ratio: float
    entropy_effective_rank: float


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    rendered = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return sha256(rendered).hexdigest()


def _self_hashed(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    if "canonical_payload_sha256" in result:
        raise ValueError("canonical payload hash was supplied twice")
    result["canonical_payload_sha256"] = _canonical_sha256(result)
    return result


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def _file_record(path: Path, root: Path) -> dict[str, Any]:
    resolved_root = root.resolve()
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(resolved_root).as_posix()
    except ValueError as error:
        raise ValueError("output file escapes its packet") from error
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"output is absent or aliased: {relative}")
    return {
        "relative_path": relative,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _source_record() -> dict[str, Any]:
    path = Path(__file__).resolve()
    return {
        "relative_path": path.relative_to(REPO_ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


PCA_SOURCE_AT_IMPORT = _source_record()


def _reverify_source() -> None:
    if _source_record() != PCA_SOURCE_AT_IMPORT:
        raise ValueError("PCA visualization source changed during execution")


def select_train_bdf2_views(
    states: np.ndarray,
    frame_indices: np.ndarray,
    *,
    expected_frame_count: int = TRAIN_FRAME_COUNT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return previous/current states and current-frame identities for BDF2 centers."""

    values = np.asarray(states)
    frames = np.asarray(frame_indices)
    if (
        values.dtype != np.float64
        or values.ndim != 3
        or values.shape[-1] != NUM_FIELDS
        or values.shape[0] != expected_frame_count
    ):
        raise ValueError("train states must retain float64 shape [expected_frames,N,5]")
    if (
        frames.dtype != np.int64
        or frames.ndim != 1
        or frames.shape[0] != expected_frame_count
    ):
        raise ValueError("train frame indices must be aligned int64 values")
    if not np.all(np.isfinite(values)):
        raise ValueError("train states contain nonfinite values")
    if np.any(np.diff(frames) != 1):
        raise ValueError("train frame indices must be contiguous and ordered")
    previous = values[:-2]
    current = values[1:-1]
    centers = frames[1:-1]
    expected_transitions = expected_frame_count - 2
    if previous.shape[0] != expected_transitions or current.shape != previous.shape:
        raise AssertionError("BDF2 train slicing did not close")
    return previous, current, centers


def _validate_state_sequences(
    state_sequences: Sequence[np.ndarray], normalization: NACANormalization
) -> tuple[tuple[np.ndarray, ...], int, int]:
    if not isinstance(normalization, NACANormalization):
        raise TypeError("a NACANormalization is required")
    sequences = tuple(np.asarray(sequence) for sequence in state_sequences)
    if not sequences:
        raise ValueError("at least one state sequence is required")
    first = sequences[0]
    if first.dtype != np.float64 or first.ndim != 3 or first.shape[-1] != NUM_FIELDS:
        raise ValueError("state sequences must retain float64 shape [T,N,5]")
    if first.shape[0] < 2 or first.shape[1] < 1 or not np.all(np.isfinite(first)):
        raise ValueError("state sequences must contain finite samples and nodes")
    for sequence in sequences[1:]:
        if (
            sequence.dtype != np.float64
            or sequence.shape != first.shape
            or not np.all(np.isfinite(sequence))
        ):
            raise ValueError("all state sequences must share finite float64 shape")
    return sequences, int(first.shape[0]), int(first.shape[1])


def snapshot_gram(
    state_sequences: Sequence[np.ndarray],
    normalization: NACANormalization,
    *,
    node_weights: np.ndarray | None = None,
    block_nodes: int = DEFAULT_BLOCK_NODES,
) -> np.ndarray:
    """Build the temporally centered snapshot Gram matrix in bounded node blocks."""

    sequences, num_samples, num_nodes = _validate_state_sequences(
        state_sequences, normalization
    )
    if isinstance(block_nodes, bool) or not isinstance(block_nodes, int):
        raise TypeError("block_nodes must be an integer")
    if block_nodes < 1:
        raise ValueError("block_nodes must be positive")
    weights: np.ndarray | None = None
    if node_weights is not None:
        raw_weights = np.asarray(node_weights)
        if raw_weights.shape == (num_nodes, 1):
            raw_weights = raw_weights[:, 0]
        if (
            raw_weights.dtype != np.float64
            or raw_weights.shape != (num_nodes,)
            or not np.all(np.isfinite(raw_weights))
            or np.any(raw_weights <= 0.0)
        ):
            raise ValueError("node weights must be positive float64 shape [N] or [N,1]")
        weights = raw_weights

    gram = np.zeros((num_samples, num_samples), dtype=np.float64)
    mean = normalization.state_mean.reshape(1, 1, NUM_FIELDS)
    scale = normalization.state_scale.reshape(1, 1, NUM_FIELDS)
    for sequence in sequences:
        for start in range(0, num_nodes, block_nodes):
            stop = min(start + block_nodes, num_nodes)
            block = (sequence[:, start:stop, :] - mean) / scale
            block = block - np.mean(block, axis=0, keepdims=True, dtype=np.float64)
            if weights is not None:
                block = block * np.sqrt(weights[start:stop])[None, :, None]
            flattened = block.reshape(num_samples, -1)
            gram += flattened @ flattened.T
    gram = 0.5 * (gram + gram.T)
    if not np.all(np.isfinite(gram)):
        raise FloatingPointError("snapshot Gram matrix became nonfinite")
    return gram


def pca_from_gram(gram: np.ndarray, *, max_components: int) -> SnapshotPCA:
    """Diagonalize a centered snapshot Gram matrix with deterministic score signs."""

    matrix = np.asarray(gram)
    if (
        matrix.dtype != np.float64
        or matrix.ndim != 2
        or matrix.shape[0] != matrix.shape[1]
        or matrix.shape[0] < 2
        or not np.all(np.isfinite(matrix))
    ):
        raise ValueError("Gram matrix must be finite square float64")
    if not np.allclose(matrix, matrix.T, rtol=1.0e-12, atol=1.0e-12):
        raise ValueError("Gram matrix must be symmetric")
    if isinstance(max_components, bool) or not isinstance(max_components, int):
        raise TypeError("max_components must be an integer")
    if max_components < 2:
        raise ValueError("at least two PCA components are required")

    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    tolerance = (
        max(float(eigenvalues[0]), 1.0) * matrix.shape[0] * np.finfo(np.float64).eps
    )
    if float(np.min(eigenvalues)) < -100.0 * tolerance:
        raise ValueError("Gram matrix is not positive semidefinite")
    eigenvalues = np.maximum(eigenvalues, 0.0)
    total = float(np.sum(eigenvalues, dtype=np.float64))
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("PCA input has no positive finite variance")
    explained = eigenvalues / total
    cumulative = np.cumsum(explained, dtype=np.float64)
    cumulative[-1] = 1.0
    rank = min(max_components, matrix.shape[0])
    scores = eigenvectors[:, :rank] * np.sqrt(eigenvalues[:rank])[None, :]
    for component in range(rank):
        pivot = int(np.argmax(np.abs(scores[:, component])))
        if scores[pivot, component] < 0.0:
            scores[:, component] *= -1.0
    positive = explained[explained > 0.0]
    participation = float(1.0 / np.sum(np.square(positive), dtype=np.float64))
    entropy_rank = float(np.exp(-np.sum(positive * np.log(positive))))
    return SnapshotPCA(
        eigenvalues=eigenvalues,
        explained_variance_ratio=explained,
        cumulative_explained_variance=cumulative,
        scores=scores,
        total_sum_squared_deviation=total,
        participation_ratio=participation,
        entropy_effective_rank=entropy_rank,
    )


def compute_snapshot_pca(
    state_sequences: Sequence[np.ndarray],
    normalization: NACANormalization,
    *,
    node_weights: np.ndarray | None = None,
    block_nodes: int = DEFAULT_BLOCK_NODES,
    max_components: int = DEFAULT_COMPONENTS,
) -> SnapshotPCA:
    return pca_from_gram(
        snapshot_gram(
            state_sequences,
            normalization,
            node_weights=node_weights,
            block_nodes=block_nodes,
        ),
        max_components=max_components,
    )


def _rank_for_variance(result: SnapshotPCA, target: float) -> int:
    indices = np.flatnonzero(result.cumulative_explained_variance >= target)
    return int(indices[0]) + 1 if indices.size else len(result.eigenvalues)


def _result_summary(result: SnapshotPCA) -> dict[str, Any]:
    return {
        "explained_variance_ratio_first_16": result.explained_variance_ratio[
            :DEFAULT_COMPONENTS
        ].tolist(),
        "cumulative_explained_variance_first_16": (
            result.cumulative_explained_variance[:DEFAULT_COMPONENTS].tolist()
        ),
        "cumulative_explained_variance_at_2": float(
            result.cumulative_explained_variance[1]
        ),
        "cumulative_explained_variance_at_4": float(
            result.cumulative_explained_variance[3]
        ),
        "cumulative_explained_variance_at_6": float(
            result.cumulative_explained_variance[5]
        ),
        "rank_at_90_percent": _rank_for_variance(result, 0.90),
        "rank_at_99_percent": _rank_for_variance(result, 0.99),
        "rank_at_99_9_percent": _rank_for_variance(result, 0.999),
        "participation_ratio": result.participation_ratio,
        "entropy_effective_rank": result.entropy_effective_rank,
        "total_sum_squared_deviation": result.total_sum_squared_deviation,
    }


def render_pca_figure(
    pdf_path: Path,
    png_path: Path,
    frame_indices: np.ndarray,
    model: SnapshotPCA,
    area_weighted: SnapshotPCA,
    *,
    plot_components: int = 12,
) -> None:
    """Render a two-panel chronological embedding and variance spectrum."""

    frames = np.asarray(frame_indices)
    if frames.dtype != np.int64 or frames.shape != (model.scores.shape[0],):
        raise ValueError("frame identities do not align with PCA scores")
    if model.scores.shape[1] < 2:
        raise ValueError("the PCA embedding needs at least two components")
    count = min(
        plot_components,
        model.explained_variance_ratio.size,
        area_weighted.explained_variance_ratio.size,
    )
    if count < 6:
        raise ValueError("the spectrum panel requires at least six components")

    style = {
        "font.family": "DejaVu Sans",
        "font.size": 9.0,
        "axes.labelsize": 9.0,
        "axes.linewidth": 0.8,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 7.5,
        "lines.linewidth": 1.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    with plt.rc_context(style):
        figure, (embedding_axis, spectrum_axis) = plt.subplots(
            1,
            2,
            figsize=(7.2, 3.0),
            gridspec_kw={"width_ratios": [1.02, 1.0]},
            constrained_layout=True,
        )
        scores = model.scores[:, :2]
        embedding_axis.plot(
            scores[:, 0], scores[:, 1], color=GREY, linewidth=0.8, alpha=0.75
        )
        scatter = embedding_axis.scatter(
            scores[:, 0],
            scores[:, 1],
            c=frames,
            cmap="viridis",
            s=13.0,
            edgecolors="none",
            zorder=2,
        )
        embedding_axis.scatter(
            scores[0, 0],
            scores[0, 1],
            s=28.0,
            marker="o",
            facecolors="none",
            edgecolors=GREEN,
            linewidths=1.2,
            label="start",
            zorder=3,
        )
        embedding_axis.scatter(
            scores[-1, 0],
            scores[-1, 1],
            s=31.0,
            marker="s",
            facecolors="none",
            edgecolors=VERMILLION,
            linewidths=1.2,
            label="end",
            zorder=3,
        )
        embedding_axis.set_xlabel(
            f"PC1 ({100.0 * model.explained_variance_ratio[0]:.1f}%)"
        )
        embedding_axis.set_ylabel(
            f"PC2 ({100.0 * model.explained_variance_ratio[1]:.1f}%)"
        )
        embedding_axis.legend(loc="best", frameon=False, handletextpad=0.3)
        embedding_axis.grid(color="#DDDDDD", linewidth=0.5, alpha=0.7)
        embedding_axis.text(
            0.01,
            0.99,
            "(a)",
            transform=embedding_axis.transAxes,
            ha="left",
            va="top",
            fontweight="bold",
        )
        colorbar = figure.colorbar(scatter, ax=embedding_axis, pad=0.02)
        colorbar.set_label("Training frame")
        colorbar.ax.tick_params(labelsize=7.5)

        components = np.arange(1, count + 1)
        model_explained = 100.0 * model.explained_variance_ratio[:count]
        model_cumulative = 100.0 * model.cumulative_explained_variance[:count]
        area_cumulative = 100.0 * area_weighted.cumulative_explained_variance[:count]
        spectrum_axis.bar(
            components,
            model_explained,
            width=0.72,
            color=BLUE,
            alpha=0.55,
            linewidth=0.0,
            label="per PC, model metric",
        )
        spectrum_axis.plot(
            components,
            model_cumulative,
            color=BLUE,
            marker="o",
            markersize=3.2,
            label="cumulative, model metric",
        )
        spectrum_axis.plot(
            components,
            area_cumulative,
            color=ORANGE,
            linestyle="--",
            marker="^",
            markersize=3.3,
            label="cumulative, area metric",
        )
        spectrum_axis.axhline(99.9, color="#BBBBBB", linewidth=0.7, zorder=0)
        spectrum_axis.annotate(
            f"2 PCs: {model_cumulative[1]:.2f}%",
            xy=(2, model_cumulative[1]),
            xytext=(3.0, 76.0),
            arrowprops={"arrowstyle": "-", "color": BLUE, "linewidth": 0.7},
            color=BLUE,
            fontsize=7.5,
        )
        spectrum_axis.annotate(
            f"6 PCs: {model_cumulative[5]:.3f}%",
            xy=(6, model_cumulative[5]),
            xytext=(6.6, 88.0),
            arrowprops={"arrowstyle": "-", "color": BLUE, "linewidth": 0.7},
            color=BLUE,
            fontsize=7.5,
        )
        spectrum_axis.set_xlim(0.35, count + 0.65)
        spectrum_axis.set_ylim(0.0, 104.0)
        spectrum_axis.set_xticks(components)
        spectrum_axis.set_xlabel("Principal component")
        spectrum_axis.set_ylabel("Explained variance (%)")
        spectrum_axis.grid(axis="y", color="#DDDDDD", linewidth=0.5, alpha=0.7)
        spectrum_axis.legend(loc="lower right", frameon=False)
        spectrum_axis.text(
            0.01,
            0.99,
            "(b)",
            transform=spectrum_axis.transAxes,
            ha="left",
            va="top",
            fontweight="bold",
        )
        figure.savefig(pdf_path, bbox_inches="tight")
        figure.savefig(png_path, dpi=300, bbox_inches="tight")
        plt.close(figure)


def _write_arrays(
    path: Path,
    frame_indices: np.ndarray,
    model: SnapshotPCA,
    area_weighted: SnapshotPCA,
    bdf2: SnapshotPCA,
) -> None:
    np.savez_compressed(
        path,
        schema=np.asarray(ARRAYS_SCHEMA),
        train_center_frame_indices=np.asarray(frame_indices, dtype=np.int64),
        model_metric_explained_variance_ratio=model.explained_variance_ratio,
        model_metric_cumulative_explained_variance=(
            model.cumulative_explained_variance
        ),
        model_metric_scores=model.scores,
        area_metric_explained_variance_ratio=(area_weighted.explained_variance_ratio),
        area_metric_cumulative_explained_variance=(
            area_weighted.cumulative_explained_variance
        ),
        bdf2_model_metric_explained_variance_ratio=bdf2.explained_variance_ratio,
        bdf2_model_metric_cumulative_explained_variance=(
            bdf2.cumulative_explained_variance
        ),
    )


def write_pca_packet(
    output_dir: Path,
    *,
    dataset_manifest: Mapping[str, Any],
    dataset_final_hash_manifest_sha256: str,
    num_nodes: int,
    node_weights: np.ndarray,
    train_states: np.ndarray,
    train_frame_indices: np.ndarray,
    normalization: NACANormalization,
    block_nodes: int = DEFAULT_BLOCK_NODES,
    max_components: int = DEFAULT_COMPONENTS,
) -> dict[str, Any]:
    """Compute and write one self-contained, train-only PCA evidence packet."""

    _reverify_source()
    if output_dir.is_symlink() or output_dir.parent.is_symlink():
        raise ValueError("PCA output or its parent is aliased")
    output = output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"PCA output already exists: {output}")
    if num_nodes != train_states.shape[1]:
        raise ValueError("declared node count differs from train states")
    previous, current, centers = select_train_bdf2_views(
        train_states, train_frame_indices
    )
    if current.shape[0] != TRAIN_TRANSITION_COUNT:
        raise AssertionError("primary state PCA must use exactly 238 samples")

    model = compute_snapshot_pca(
        (current,),
        normalization,
        block_nodes=block_nodes,
        max_components=max_components,
    )
    area_weighted = compute_snapshot_pca(
        (current,),
        normalization,
        node_weights=node_weights,
        block_nodes=block_nodes,
        max_components=max_components,
    )
    bdf2 = compute_snapshot_pca(
        (previous, current),
        normalization,
        block_nodes=block_nodes,
        max_components=max_components,
    )

    output.mkdir(parents=True)
    pdf_path = output / "naca0012_train_pca.pdf"
    png_path = output / "naca0012_train_pca.png"
    arrays_path = output / "pca_arrays.npz"
    summary_path = output / "pca_summary.json"
    render_pca_figure(pdf_path, png_path, centers, model, area_weighted)
    _write_arrays(arrays_path, centers, model, area_weighted, bdf2)
    _reverify_source()

    summary = _self_hashed(
        {
            "schema": SUMMARY_SCHEMA,
            "status": "complete",
            "contract_sha256": BASELINE_CONTRACT_SHA256,
            "dataset_manifest_payload_sha256": dataset_manifest[
                "canonical_payload_sha256"
            ],
            "dataset_final_hash_manifest_sha256": (dataset_final_hash_manifest_sha256),
            "dataset_source_set_sha256": dataset_manifest["source_manifest"][
                "source_set_sha256"
            ],
            "producer": dict(PCA_SOURCE_AT_IMPORT),
            "fit_scope": {
                "population_role": "train",
                "primary_samples": TRAIN_TRANSITION_COUNT,
                "primary_frame_indices_inclusive": [
                    int(centers[0]),
                    int(centers[-1]),
                ],
                "primary_state_definition": "train_states[1:-1]",
                "primary_coordinate_count": int(num_nodes * NUM_FIELDS),
                "normalization": (
                    "train-fitted per-field state normalization from the "
                    "baseline packet"
                ),
                "primary_metric": "uniform model-coordinate Euclidean metric",
                "robustness_metrics": [
                    "vertex-area-weighted state coordinates",
                    "complete normalized BDF2 pair in model-coordinate metric",
                ],
            },
            "method": {
                "temporal_centering": "separate mean for every flattened coordinate",
                "algorithm": "symmetric eigendecomposition of block-accumulated snapshot Gram matrix",
                "node_block_size": block_nodes,
                "stored_score_components": min(max_components, current.shape[0]),
                "score_sign_rule": (
                    "largest-magnitude chronological score is nonnegative"
                ),
                "variance_thresholds": list(VARIANCE_THRESHOLDS),
            },
            "model_metric_state_pca": _result_summary(model),
            "area_weighted_state_pca": _result_summary(area_weighted),
            "complete_bdf2_model_metric_pca": _result_summary(bdf2),
            "interpretation": {
                "supported": (
                    "The normalized training trajectory has strong low-rank, "
                    "path-like variance concentration under the stated metrics."
                ),
                "not_supported": [
                    "an exact intrinsic or topological dimension",
                    "existence or smoothness of a data manifold",
                    "rollout stability or accuracy of any learned model",
                ],
            },
            "access": {
                "dataset_packet_roles_verified_by_authoritative_loader": [
                    "train",
                    "development",
                ],
                "population_values_used_in_pca": ["train"],
                "development_values_used_in_pca": False,
                "native_replay_margin_values_used_in_pca": False,
                "prospective_opened": False,
                "sealed_opened": False,
            },
            "files": {
                "figure_pdf": _file_record(pdf_path, output),
                "figure_png": _file_record(png_path, output),
                "arrays": _file_record(arrays_path, output),
            },
        }
    )
    _write_json(summary_path, summary)
    if {path.name for path in output.iterdir()} != OUTPUT_FILES:
        raise ValueError("PCA packet has missing or unexpected files before sealing")

    files = {name: _file_record(output / name, output) for name in sorted(OUTPUT_FILES)}
    final = {
        "schema": FINAL_HASH_SCHEMA,
        "contract_sha256": BASELINE_CONTRACT_SHA256,
        "dataset_final_hash_manifest_sha256": dataset_final_hash_manifest_sha256,
        "files": files,
        "self_hash_excluded": True,
        "prospective_opened": False,
        "sealed_opened": False,
    }
    _write_json(output / "final_hash_manifest.json", final)
    _reverify_source()
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--block-nodes", type=int, default=DEFAULT_BLOCK_NODES)
    parser.add_argument("--max-components", type=int, default=DEFAULT_COMPONENTS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        contract = load_naca_baseline_contract(arguments.contract)
        (
            dataset_manifest,
            geometry,
            normalization,
            roles,
            dataset_packet_sha256,
        ) = _load_dataset(arguments.dataset_dir, contract)
        train_states, train_frame_indices, _ = roles["train"]
        summary = write_pca_packet(
            arguments.output_dir,
            dataset_manifest=dataset_manifest,
            dataset_final_hash_manifest_sha256=dataset_packet_sha256,
            num_nodes=geometry.num_nodes,
            node_weights=geometry.node_weights,
            train_states=train_states,
            train_frame_indices=train_frame_indices,
            normalization=normalization,
            block_nodes=arguments.block_nodes,
            max_components=arguments.max_components,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"NACA0012 train-only PCA visualization failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
