"""Diagnostics for 2D Euler primitive-variable rollouts.

The functions here are intentionally representation-light. They operate on
primitive arrays with last dimension ``[rho, v1, v2, pres]`` and optional graph
edges, so they can be used before deciding whether a baseline is grid-based,
point-cloud based, or graph based.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from utility.time_dependent_no.euler2d import (
    CONSERVATIVE_NAMES,
    EulerNodeType,
    primitive_to_conservative,
)
from utility.time_dependent_no.errors import relative_l2_error, root_mean_squared_error


ArrayLike = Any


def rollout_rmse_by_time(
    prediction: ArrayLike,
    target: ArrayLike,
    *,
    node_mask: ArrayLike | None = None,
) -> np.ndarray:
    """Return RMSE over nodes and variables for each rollout time."""

    pred, truth = _matching_arrays(prediction, target)
    pred, truth = _select_nodes(pred, truth, node_mask)
    return root_mean_squared_error(pred, truth, axis=(-2, -1))


def rollout_relative_l2_by_time(
    prediction: ArrayLike,
    target: ArrayLike,
    *,
    node_mask: ArrayLike | None = None,
    eps: float = 1e-12,
) -> np.ndarray:
    """Return relative L2 error over nodes and variables for each time."""

    pred, truth = _matching_arrays(prediction, target)
    pred, truth = _select_nodes(pred, truth, node_mask)
    return relative_l2_error(pred, truth, axis=(-2, -1), eps=eps)


def valid_prediction_time(
    error_by_time: ArrayLike,
    threshold: float,
    *,
    dt: float = 1.0,
) -> np.ndarray:
    """Return first time where error exceeds ``threshold``, or the last time."""

    if threshold <= 0.0:
        raise ValueError("threshold must be positive")
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    errors = np.asarray(error_by_time, dtype=np.float64)
    if errors.ndim == 0:
        raise ValueError("error_by_time must have a time axis")
    failed = errors > threshold
    first_failure = np.argmax(failed, axis=-1)
    has_failure = np.any(failed, axis=-1)
    last_index = errors.shape[-1] - 1
    indices = np.where(has_failure, first_failure, last_index)
    return indices.astype(np.float64) * dt


def positivity_metrics(
    primitive: ArrayLike,
    *,
    node_mask: ArrayLike | None = None,
    eps: float = 0.0,
) -> dict[str, float | int | bool]:
    """Count nonpositive density and pressure values."""

    prim = _validate_primitive(primitive)
    if node_mask is not None:
        prim = prim[..., _as_node_mask(node_mask), :]
    rho = prim[..., 0]
    pressure = prim[..., 3]
    total = int(rho.size)
    bad_rho = rho <= eps
    bad_pressure = pressure <= eps
    return {
        "total_values": total,
        "min_density": float(np.min(rho)) if total else float("nan"),
        "min_pressure": float(np.min(pressure)) if total else float("nan"),
        "num_nonpositive_density": int(np.sum(bad_rho)),
        "num_nonpositive_pressure": int(np.sum(bad_pressure)),
        "fraction_nonpositive_density": float(np.mean(bad_rho)) if total else 0.0,
        "fraction_nonpositive_pressure": float(np.mean(bad_pressure)) if total else 0.0,
        "all_positive": bool(not np.any(bad_rho) and not np.any(bad_pressure)),
    }


def conservation_totals(
    primitive: ArrayLike,
    *,
    node_weights: ArrayLike | None = None,
    node_mask: ArrayLike | None = None,
    gamma: float = 1.4,
) -> np.ndarray:
    """Return weighted totals of conservative variables over nodes."""

    prim = _validate_primitive(primitive)
    if node_mask is not None:
        mask = _as_node_mask(node_mask)
        prim = prim[..., mask, :]
        if node_weights is not None:
            node_weights = np.asarray(node_weights, dtype=np.float64)[mask]
    cons = primitive_to_conservative(prim, gamma=gamma)
    if node_weights is None:
        return np.sum(cons, axis=-2)

    weights = np.asarray(node_weights, dtype=np.float64)
    if weights.ndim != 1 or weights.shape[0] != cons.shape[-2]:
        raise ValueError("node_weights must have shape (num_selected_nodes,)")
    weight_shape = (1,) * (cons.ndim - 2) + (weights.shape[0], 1)
    return np.sum(cons * weights.reshape(weight_shape), axis=-2)


def conservation_drift(
    primitive_sequence: ArrayLike,
    *,
    node_weights: ArrayLike | None = None,
    node_mask: ArrayLike | None = None,
    gamma: float = 1.4,
    eps: float = 1e-12,
) -> dict[str, np.ndarray | dict[str, float]]:
    """Measure drift of conservative totals relative to the first time step."""

    sequence = _validate_primitive(primitive_sequence)
    if sequence.ndim < 3:
        raise ValueError("primitive_sequence must include time and node axes")
    totals = conservation_totals(
        sequence,
        node_weights=node_weights,
        node_mask=node_mask,
        gamma=gamma,
    )
    initial = np.take(totals, indices=[0], axis=-2)
    absolute = totals - initial
    relative = absolute / np.maximum(np.abs(initial), eps)
    max_relative = np.max(np.abs(relative), axis=tuple(range(relative.ndim - 1)))
    return {
        "totals": totals,
        "absolute_drift": absolute,
        "relative_drift": relative,
        "max_relative_by_variable": {
            name: float(value)
            for name, value in zip(CONSERVATIVE_NAMES, max_relative, strict=True)
        },
    }


def node_variation_score(values: ArrayLike, edges: ArrayLike) -> np.ndarray:
    """Return per-node max edge jump for scalar or vector node fields."""

    field = np.asarray(values, dtype=np.float64)
    if field.ndim == 1:
        edge_num_nodes = field.shape[-1]
    elif field.shape[-1] == 4:
        edge_num_nodes = field.shape[-2]
    else:
        edge_num_nodes = field.shape[-1]
    edge_index = _validate_edges(edges, num_nodes=edge_num_nodes)
    if field.ndim == 1:
        field = field[None, :, None]
        prefix: tuple[int, ...] = ()
    elif field.ndim >= 2 and field.shape[-1] == 4:
        prefix = field.shape[:-2]
        field = field.reshape((-1, field.shape[-2], field.shape[-1]))
    else:
        prefix = field.shape[:-1]
        field = field.reshape((-1, field.shape[-1], 1))

    num_nodes = field.shape[1]
    scores = np.zeros((field.shape[0], num_nodes), dtype=np.float64)
    src = edge_index[:, 0]
    dst = edge_index[:, 1]
    for batch_index in range(field.shape[0]):
        jumps = np.linalg.norm(
            field[batch_index, src] - field[batch_index, dst], axis=-1
        )
        np.maximum.at(scores[batch_index], src, jumps)
        np.maximum.at(scores[batch_index], dst, jumps)
    return scores.reshape(prefix + (num_nodes,))


def shock_indicator(
    primitive: ArrayLike,
    edges: ArrayLike,
    *,
    scalar_index: int = 3,
    quantile: float = 0.9,
) -> np.ndarray:
    """Return a high-gradient node mask used as a shock-region proxy."""

    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    prim = _validate_primitive(primitive)
    scalar = prim[..., scalar_index]
    scores = node_variation_score(scalar, edges)
    threshold = np.quantile(scores, quantile, axis=-1, keepdims=True)
    return (scores >= threshold) & (scores > 0.0)


def shock_region_metrics(
    primitive: ArrayLike,
    edges: ArrayLike,
    *,
    scalar_index: int = 3,
    quantile: float = 0.9,
) -> dict[str, np.ndarray]:
    """Summarize the size and strength of a graph-gradient shock proxy."""

    prim = _validate_primitive(primitive)
    scores = node_variation_score(prim[..., scalar_index], edges)
    mask = shock_indicator(prim, edges, scalar_index=scalar_index, quantile=quantile)
    return {
        "shock_fraction": np.mean(mask, axis=-1),
        "mean_variation_score": np.mean(scores, axis=-1),
        "max_variation_score": np.max(scores, axis=-1),
    }


def shock_smearing_metrics(
    prediction: ArrayLike,
    target: ArrayLike,
    edges: ArrayLike,
    *,
    scalar_index: int = 3,
    relative_threshold: float = 0.25,
    eps: float = 1e-12,
) -> dict[str, np.ndarray]:
    """Compare graph-gradient shock thickness and strength proxies.

    The active-node fraction is a mesh/graph proxy for shock thickness: it
    counts nodes whose edge-jump score is at least ``relative_threshold`` times
    the maximum score for that field. A smeared prediction should usually have
    a larger thickness ratio and a smaller strength ratio than the target.
    """

    if not 0.0 < relative_threshold <= 1.0:
        raise ValueError("relative_threshold must be in (0, 1]")
    pred, truth = _matching_arrays(prediction, target)
    pred_scores = node_variation_score(pred[..., scalar_index], edges)
    truth_scores = node_variation_score(truth[..., scalar_index], edges)
    pred_max = np.max(pred_scores, axis=-1)
    truth_max = np.max(truth_scores, axis=-1)
    pred_fraction = _relative_active_fraction(
        pred_scores, pred_max, relative_threshold=relative_threshold
    )
    truth_fraction = _relative_active_fraction(
        truth_scores, truth_max, relative_threshold=relative_threshold
    )
    return {
        "prediction_active_fraction": pred_fraction,
        "target_active_fraction": truth_fraction,
        "thickness_ratio": pred_fraction / np.maximum(truth_fraction, eps),
        "prediction_max_variation": pred_max,
        "target_max_variation": truth_max,
        "strength_ratio": pred_max / np.maximum(truth_max, eps),
    }


def near_shock_error(
    prediction: ArrayLike,
    target: ArrayLike,
    shock_mask: ArrayLike,
    *,
    eps: float = 1e-12,
) -> dict[str, np.ndarray]:
    """Compare relative error inside and outside a shock-region mask."""

    mask = _as_node_mask(shock_mask)
    if not np.any(mask):
        raise ValueError("shock_mask must contain at least one True entry")
    if np.all(mask):
        raise ValueError("shock_mask must leave at least one smooth node")
    pred, truth = _matching_arrays(prediction, target)
    shock_pred, shock_truth = _select_nodes(pred, truth, mask)
    smooth_pred, smooth_truth = _select_nodes(pred, truth, ~mask)
    return {
        "shock_relative_l2": relative_l2_error(
            shock_pred, shock_truth, axis=(-2, -1), eps=eps
        ),
        "smooth_relative_l2": relative_l2_error(
            smooth_pred, smooth_truth, axis=(-2, -1), eps=eps
        ),
        "shock_rmse": root_mean_squared_error(shock_pred, shock_truth, axis=(-2, -1)),
        "smooth_rmse": root_mean_squared_error(
            smooth_pred, smooth_truth, axis=(-2, -1)
        ),
    }


def shock_centroid(
    primitive: ArrayLike,
    positions: ArrayLike,
    edges: ArrayLike,
    *,
    scalar_index: int = 3,
    eps: float = 1e-12,
) -> np.ndarray:
    """Return variation-weighted centroid of a shock proxy."""

    prim = _validate_primitive(primitive)
    pos = np.asarray(positions, dtype=np.float64)
    if pos.ndim != 2 or pos.shape[0] != prim.shape[-2]:
        raise ValueError("positions must have shape (num_nodes, dim)")
    scores = node_variation_score(prim[..., scalar_index], edges)
    total = np.sum(scores, axis=-1, keepdims=True)
    weighted = np.einsum("...n,nd->...d", scores, pos)
    return weighted / np.maximum(total, eps)


def shock_centroid_distance(
    prediction: ArrayLike,
    target: ArrayLike,
    positions: ArrayLike,
    edges: ArrayLike,
    *,
    scalar_index: int = 3,
) -> np.ndarray:
    """Return distance between predicted and target shock centroids."""

    pred_centroid = shock_centroid(
        prediction, positions, edges, scalar_index=scalar_index
    )
    truth_centroid = shock_centroid(target, positions, edges, scalar_index=scalar_index)
    return np.linalg.norm(pred_centroid - truth_centroid, axis=-1)


def boundary_leakage_metrics(
    prediction: ArrayLike,
    target: ArrayLike,
    node_type: ArrayLike,
    *,
    eps: float = 1e-12,
) -> dict[str, float | int]:
    """Measure prediction error on non-normal boundary nodes."""

    node_types = np.asarray(node_type)
    if node_types.ndim > 1 and node_types.shape[-1] == 1:
        node_types = np.squeeze(node_types, axis=-1)
    boundary = node_types != int(EulerNodeType.NORMAL)
    if not np.any(boundary):
        return {
            "num_boundary_nodes": 0,
            "boundary_rmse": float("nan"),
            "boundary_relative_l2": float("nan"),
            "boundary_max_abs": float("nan"),
        }
    pred, truth = _matching_arrays(prediction, target)
    pred, truth = _select_nodes(pred, truth, boundary)
    return {
        "num_boundary_nodes": int(np.sum(boundary)),
        "boundary_rmse": float(
            np.mean(root_mean_squared_error(pred, truth, axis=(-2, -1)))
        ),
        "boundary_relative_l2": float(
            np.mean(relative_l2_error(pred, truth, axis=(-2, -1), eps=eps))
        ),
        "boundary_max_abs": float(np.max(np.abs(pred - truth))),
    }


def summarize_euler2d_rollout(
    prediction: ArrayLike,
    target: ArrayLike,
    *,
    node_type: ArrayLike | None = None,
    edges: ArrayLike | None = None,
    positions: ArrayLike | None = None,
    node_weights: ArrayLike | None = None,
    gamma: float = 1.4,
) -> dict[str, float | int | bool]:
    """Compact scalar summary for first-pass rollout diagnostics."""

    pred, truth = _matching_arrays(prediction, target)
    rmse = rollout_rmse_by_time(pred, truth)
    rel = rollout_relative_l2_by_time(pred, truth)
    positivity = positivity_metrics(pred)
    drift = conservation_drift(pred, node_weights=node_weights, gamma=gamma)
    summary: dict[str, float | int | bool] = {
        "mean_rmse": float(np.mean(rmse)),
        "final_rmse": float(np.mean(np.take(rmse, -1, axis=-1))),
        "mean_relative_l2": float(np.mean(rel)),
        "final_relative_l2": float(np.mean(np.take(rel, -1, axis=-1))),
        "all_density_pressure_positive": bool(positivity["all_positive"]),
        "num_nonpositive_density": int(positivity["num_nonpositive_density"]),
        "num_nonpositive_pressure": int(positivity["num_nonpositive_pressure"]),
    }
    for name, value in drift["max_relative_by_variable"].items():
        summary[f"max_relative_{name}_drift"] = float(value)

    if node_type is not None:
        boundary = boundary_leakage_metrics(pred, truth, node_type)
        summary.update(boundary)
    if edges is not None:
        shock = shock_region_metrics(
            truth[..., -1, :, :] if truth.ndim >= 3 else truth, edges
        )
        smearing = shock_smearing_metrics(
            pred[..., -1, :, :] if pred.ndim >= 3 else pred,
            truth[..., -1, :, :] if truth.ndim >= 3 else truth,
            edges,
        )
        summary["target_final_shock_fraction"] = float(np.mean(shock["shock_fraction"]))
        summary["target_final_max_shock_score"] = float(
            np.mean(shock["max_variation_score"])
        )
        summary["final_shock_thickness_ratio"] = float(
            np.mean(smearing["thickness_ratio"])
        )
        summary["final_shock_strength_ratio"] = float(
            np.mean(smearing["strength_ratio"])
        )
    if edges is not None and positions is not None:
        distance = shock_centroid_distance(
            pred[..., -1, :, :] if pred.ndim >= 3 else pred,
            truth[..., -1, :, :] if truth.ndim >= 3 else truth,
            positions,
            edges,
        )
        summary["final_shock_centroid_distance"] = float(np.mean(distance))
    return summary


def _matching_arrays(
    prediction: ArrayLike, target: ArrayLike
) -> tuple[np.ndarray, np.ndarray]:
    pred = _validate_primitive(prediction)
    truth = _validate_primitive(target)
    if pred.shape != truth.shape:
        raise ValueError(
            f"prediction and target must match, got {pred.shape} and {truth.shape}"
        )
    return pred, truth


def _validate_primitive(value: ArrayLike) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim < 2 or array.shape[-1] != 4:
        raise ValueError("Euler primitive arrays must have last dimension 4")
    return array


def _select_nodes(
    prediction: np.ndarray,
    target: np.ndarray,
    node_mask: ArrayLike | None,
) -> tuple[np.ndarray, np.ndarray]:
    if node_mask is None:
        return prediction, target
    mask = _as_node_mask(node_mask)
    if prediction.shape[-2] != mask.shape[0]:
        raise ValueError("node_mask must match the node axis")
    return prediction[..., mask, :], target[..., mask, :]


def _as_node_mask(mask: ArrayLike) -> np.ndarray:
    result = np.asarray(mask, dtype=bool)
    if result.ndim != 1:
        raise ValueError("node_mask must be one-dimensional")
    return result


def _validate_edges(edges: ArrayLike, *, num_nodes: int) -> np.ndarray:
    edge_index = np.asarray(edges, dtype=np.int64)
    if edge_index.ndim != 2 or edge_index.shape[1] != 2:
        raise ValueError("edges must have shape (num_edges, 2)")
    if edge_index.size and (np.min(edge_index) < 0 or np.max(edge_index) >= num_nodes):
        raise ValueError("edges contain node indices outside the field")
    return edge_index


def _relative_active_fraction(
    scores: np.ndarray,
    max_scores: np.ndarray,
    *,
    relative_threshold: float,
) -> np.ndarray:
    threshold = relative_threshold * max_scores[..., None]
    active = (scores >= threshold) & (scores > 0.0)
    return np.mean(active, axis=-1)

