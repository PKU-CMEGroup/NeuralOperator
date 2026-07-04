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


def shock_front_scores(
    primitive: ArrayLike,
    edges: ArrayLike,
    *,
    scalar_index: int = 3,
) -> np.ndarray:
    """Return graph-gradient scores for a scalar primitive field."""

    prim = _validate_primitive(primitive)
    return node_variation_score(prim[..., scalar_index], edges)


def shock_front_mask_from_scores(
    scores: ArrayLike,
    *,
    quantile: float = 0.9,
    node_mask: ArrayLike | None = None,
) -> np.ndarray:
    """Return high-gradient front masks from per-node shock scores.

    The quantile is computed on ``node_mask`` only when a mask is provided, but
    the returned array keeps the original node axis and clears nodes outside
    that mask.
    """

    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim < 1:
        raise ValueError("scores must include a node axis")
    if node_mask is None:
        mask = np.ones(values.shape[-1], dtype=bool)
    else:
        mask = _as_node_mask(node_mask)
        if mask.shape[0] != values.shape[-1]:
            raise ValueError("node_mask must match the score node axis")
    if not np.any(mask):
        raise ValueError("node_mask must select at least one node")

    selected = values[..., mask]
    threshold = np.quantile(selected, quantile, axis=-1)
    front = (values >= np.expand_dims(threshold, axis=-1)) & (values > 0.0)
    return front & mask


def shock_front_masks(
    prediction: ArrayLike,
    target: ArrayLike,
    edges: ArrayLike,
    *,
    scalar_index: int = 3,
    quantile: float = 0.9,
    node_mask: ArrayLike | None = None,
) -> dict[str, np.ndarray]:
    """Return predicted and target front masks from graph pressure gradients."""

    pred, truth = _matching_arrays(prediction, target)
    num_nodes = pred.shape[-2]
    edge_index = _validate_edges(edges, num_nodes=num_nodes)
    pred_scores = shock_front_scores(pred, edge_index, scalar_index=scalar_index)
    target_scores = shock_front_scores(truth, edge_index, scalar_index=scalar_index)
    return {
        "prediction_scores": pred_scores,
        "target_scores": target_scores,
        "prediction_mask": shock_front_mask_from_scores(
            pred_scores, quantile=quantile, node_mask=node_mask
        ),
        "target_mask": shock_front_mask_from_scores(
            target_scores, quantile=quantile, node_mask=node_mask
        ),
    }


def front_overlap_metrics(
    prediction_mask: ArrayLike,
    target_mask: ArrayLike,
) -> dict[str, np.ndarray]:
    """Return IoU/F1-style overlap metrics for front masks over the node axis."""

    pred, truth = _matching_masks(prediction_mask, target_mask)
    intersection = np.count_nonzero(pred & truth, axis=-1)
    union = np.count_nonzero(pred | truth, axis=-1)
    pred_count = np.count_nonzero(pred, axis=-1)
    target_count = np.count_nonzero(truth, axis=-1)
    precision = _safe_divide(intersection, pred_count)
    recall = _safe_divide(intersection, target_count)
    f1 = _safe_divide(2.0 * precision * recall, precision + recall)
    iou = _safe_divide(intersection, union)
    return {
        "intersection_count": intersection,
        "union_count": union,
        "prediction_count": pred_count,
        "target_count": target_count,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
    }


def front_region_masks(
    prediction_mask: ArrayLike,
    target_mask: ArrayLike,
    *,
    node_mask: ArrayLike | None = None,
) -> dict[str, np.ndarray]:
    """Split nodes into overlap, displaced-front, and smooth regions."""

    pred, truth = _matching_masks(prediction_mask, target_mask)
    if node_mask is None:
        base = np.ones(pred.shape[-1], dtype=bool)
    else:
        base = _as_node_mask(node_mask)
        if base.shape[0] != pred.shape[-1]:
            raise ValueError("node_mask must match the front mask node axis")
    base = _broadcast_node_mask(base, pred.ndim)
    pred = pred & base
    truth = truth & base
    overlap = pred & truth
    pred_only = pred & ~truth
    target_only = truth & ~pred
    union = pred | truth
    return {
        "target_front": truth,
        "predicted_front": pred,
        "front_overlap": overlap,
        "front_union": union,
        "predicted_front_only": pred_only,
        "target_front_only": target_only,
        "smooth": base & ~union,
    }


def front_centroid(
    front_mask: ArrayLike,
    positions: ArrayLike,
    *,
    weights: ArrayLike | None = None,
) -> np.ndarray:
    """Return front centroids in physical coordinates."""

    mask = np.asarray(front_mask, dtype=bool)
    if mask.ndim < 1:
        raise ValueError("front_mask must include a node axis")
    pos = _validate_positions(positions, num_nodes=mask.shape[-1])
    if weights is None:
        weighted_mask = mask.astype(np.float64)
    else:
        weight_values = np.asarray(weights, dtype=np.float64)
        if weight_values.shape != mask.shape:
            raise ValueError("weights must have the same shape as front_mask")
        weighted_mask = np.where(mask, weight_values, 0.0)
    total = np.sum(weighted_mask, axis=-1)
    centroid = np.einsum("...n,nd->...d", weighted_mask, pos)
    out = np.full(centroid.shape, np.nan, dtype=np.float64)
    valid = total > 0.0
    out[valid] = centroid[valid] / total[valid, None]
    return out


def front_centroid_distance(
    prediction_mask: ArrayLike,
    target_mask: ArrayLike,
    positions: ArrayLike,
    *,
    prediction_weights: ArrayLike | None = None,
    target_weights: ArrayLike | None = None,
) -> np.ndarray:
    """Return physical distance between predicted and target front centroids."""

    pred, truth = _matching_masks(prediction_mask, target_mask)
    pred_centroid = front_centroid(pred, positions, weights=prediction_weights)
    target_centroid = front_centroid(truth, positions, weights=target_weights)
    return np.linalg.norm(pred_centroid - target_centroid, axis=-1)


def front_distance_metrics(
    prediction_mask: ArrayLike,
    target_mask: ArrayLike,
    positions: ArrayLike,
) -> dict[str, np.ndarray]:
    """Return nearest-front and symmetric Chamfer distances in physical units."""

    pred, truth = _matching_masks(prediction_mask, target_mask)
    pos = _validate_positions(positions, num_nodes=pred.shape[-1])
    prefix = pred.shape[:-1]
    pred_flat = pred.reshape((-1, pred.shape[-1]))
    truth_flat = truth.reshape((-1, truth.shape[-1]))

    pred_to_target = np.full(pred_flat.shape[0], np.nan, dtype=np.float64)
    target_to_pred = np.full(pred_flat.shape[0], np.nan, dtype=np.float64)
    pred_to_target_max = np.full(pred_flat.shape[0], np.nan, dtype=np.float64)
    target_to_pred_max = np.full(pred_flat.shape[0], np.nan, dtype=np.float64)
    pred_counts = np.count_nonzero(pred_flat, axis=1)
    target_counts = np.count_nonzero(truth_flat, axis=1)

    for index, (pred_row, truth_row) in enumerate(
        zip(pred_flat, truth_flat, strict=True)
    ):
        if not np.any(pred_row) or not np.any(truth_row):
            continue
        p2t = _nearest_distances(pos[pred_row], pos[truth_row])
        t2p = _nearest_distances(pos[truth_row], pos[pred_row])
        pred_to_target[index] = np.mean(p2t)
        target_to_pred[index] = np.mean(t2p)
        pred_to_target_max[index] = np.max(p2t)
        target_to_pred_max[index] = np.max(t2p)

    chamfer = 0.5 * (pred_to_target + target_to_pred)
    hausdorff = np.maximum(pred_to_target_max, target_to_pred_max)
    return {
        "prediction_count": pred_counts.reshape(prefix),
        "target_count": target_counts.reshape(prefix),
        "prediction_to_target_mean": pred_to_target.reshape(prefix),
        "target_to_prediction_mean": target_to_pred.reshape(prefix),
        "symmetric_chamfer_mean": chamfer.reshape(prefix),
        "hausdorff": hausdorff.reshape(prefix),
    }


def median_edge_length(positions: ArrayLike, edges: ArrayLike) -> float:
    """Return the median physical edge length for a graph."""

    pos = np.asarray(positions, dtype=np.float64)
    if pos.ndim != 2:
        raise ValueError("positions must have shape (num_nodes, dim)")
    edge_index = _validate_edges(edges, num_nodes=pos.shape[0])
    if edge_index.size == 0:
        return 0.0
    lengths = np.linalg.norm(pos[edge_index[:, 0]] - pos[edge_index[:, 1]], axis=-1)
    return float(np.median(lengths))


def shift_grid(
    max_shift: float,
    *,
    grid_size: int = 5,
    dim: int = 2,
) -> np.ndarray:
    """Return a Cartesian translation grid centered on zero."""

    if max_shift < 0.0:
        raise ValueError("max_shift must be nonnegative")
    if grid_size < 1:
        raise ValueError("grid_size must be positive")
    if dim < 1:
        raise ValueError("dim must be positive")
    if max_shift == 0.0 or grid_size == 1:
        return np.zeros((1, dim), dtype=np.float64)
    axes = [np.linspace(-max_shift, max_shift, grid_size, dtype=np.float64)] * dim
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.stack([item.reshape(-1) for item in mesh], axis=-1)


def local_shift_alignment_metrics(
    prediction: ArrayLike,
    target: ArrayLike,
    positions: ArrayLike,
    region_mask: ArrayLike,
    shifts: ArrayLike,
    *,
    scalar_index: int = 3,
) -> dict[str, np.ndarray]:
    """Measure how much scalar error drops after small spatial translations.

    For each candidate shift, predicted node values are sampled at target node
    positions by nearest neighbor from ``positions + shift``. A large RMSE
    reduction after a small shift is evidence for front displacement rather
    than pure amplitude error.
    """

    pred, truth = _matching_arrays(prediction, target)
    pos = _validate_positions(positions, num_nodes=pred.shape[-2])
    mask = np.asarray(region_mask, dtype=bool)
    if mask.shape != pred.shape[:-1]:
        raise ValueError("region_mask must match prediction without variable axis")
    shift_values = np.asarray(shifts, dtype=np.float64)
    if shift_values.ndim != 2 or shift_values.shape[1] != pos.shape[1]:
        raise ValueError("shifts must have shape (num_shifts, coordinate_dim)")
    if shift_values.shape[0] == 0:
        raise ValueError("shifts must contain at least one candidate")

    mappings = [_nearest_indices(pos, pos + shift) for shift in shift_values]
    pred_scalar = pred[..., scalar_index]
    target_scalar = truth[..., scalar_index]
    prefix = pred_scalar.shape[:-1]
    pred_flat = pred_scalar.reshape((-1, pred_scalar.shape[-1]))
    target_flat = target_scalar.reshape((-1, target_scalar.shape[-1]))
    mask_flat = mask.reshape((-1, mask.shape[-1]))

    baseline = np.full(pred_flat.shape[0], np.nan, dtype=np.float64)
    best = np.full(pred_flat.shape[0], np.nan, dtype=np.float64)
    best_index = np.full(pred_flat.shape[0], -1, dtype=np.int64)
    selected_count = np.count_nonzero(mask_flat, axis=1)

    for row_index, (pred_row, target_row, mask_row) in enumerate(
        zip(pred_flat, target_flat, mask_flat, strict=True)
    ):
        if not np.any(mask_row):
            continue
        baseline[row_index] = _rmse_1d(pred_row[mask_row], target_row[mask_row])
        candidate_errors = np.array(
            [
                _rmse_1d(pred_row[mapping][mask_row], target_row[mask_row])
                for mapping in mappings
            ],
            dtype=np.float64,
        )
        best_index[row_index] = int(np.nanargmin(candidate_errors))
        best[row_index] = candidate_errors[best_index[row_index]]

    best_shift = np.full((pred_flat.shape[0], pos.shape[1]), np.nan, dtype=np.float64)
    valid = best_index >= 0
    best_shift[valid] = shift_values[best_index[valid]]
    reduction = _safe_divide(baseline - best, baseline)
    return {
        "selected_count": selected_count.reshape(prefix),
        "baseline_rmse": baseline.reshape(prefix),
        "best_shift_rmse": best.reshape(prefix),
        "relative_rmse_reduction": reduction.reshape(prefix),
        "best_shift": best_shift.reshape(prefix + (pos.shape[1],)),
        "best_shift_norm": np.linalg.norm(best_shift, axis=-1).reshape(prefix),
    }


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


def _matching_masks(
    prediction_mask: ArrayLike,
    target_mask: ArrayLike,
) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(prediction_mask, dtype=bool)
    truth = np.asarray(target_mask, dtype=bool)
    if pred.shape != truth.shape:
        raise ValueError(
            f"prediction and target masks must match, got {pred.shape} and {truth.shape}"
        )
    if pred.ndim < 1:
        raise ValueError("front masks must include a node axis")
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


def _validate_positions(positions: ArrayLike, *, num_nodes: int) -> np.ndarray:
    pos = np.asarray(positions, dtype=np.float64)
    if pos.ndim != 2 or pos.shape[0] != num_nodes:
        raise ValueError("positions must have shape (num_nodes, coordinate_dim)")
    if pos.shape[1] < 1:
        raise ValueError("positions must include at least one coordinate")
    return pos


def _broadcast_node_mask(mask: np.ndarray, ndim: int) -> np.ndarray:
    return mask.reshape((1,) * (ndim - 1) + (mask.shape[0],))


def _safe_divide(numerator: ArrayLike, denominator: ArrayLike) -> np.ndarray:
    num = np.asarray(numerator, dtype=np.float64)
    den = np.asarray(denominator, dtype=np.float64)
    out = np.full(np.broadcast_shapes(num.shape, den.shape), np.nan, dtype=np.float64)
    num_b = np.broadcast_to(num, out.shape)
    den_b = np.broadcast_to(den, out.shape)
    valid = den_b != 0.0
    out[valid] = num_b[valid] / den_b[valid]
    return out


def _nearest_distances(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    if query.size == 0 or reference.size == 0:
        return np.empty((0,), dtype=np.float64)
    try:
        from scipy.spatial import cKDTree  # type: ignore[import-not-found]

        distances, _ = cKDTree(reference).query(query, k=1)
        return np.asarray(distances, dtype=np.float64)
    except ModuleNotFoundError:
        return _nearest_distances_chunked(query, reference)


def _nearest_indices(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    if query.ndim != 2 or reference.ndim != 2 or query.shape[1] != reference.shape[1]:
        raise ValueError("query and reference must have shape (num_points, dim)")
    try:
        from scipy.spatial import cKDTree  # type: ignore[import-not-found]

        _, indices = cKDTree(reference).query(query, k=1)
        return np.asarray(indices, dtype=np.int64)
    except ModuleNotFoundError:
        distances = _pairwise_distances_chunked(query, reference)
        return np.argmin(distances, axis=1).astype(np.int64)


def _nearest_distances_chunked(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return np.min(_pairwise_distances_chunked(query, reference), axis=1)


def _pairwise_distances_chunked(
    query: np.ndarray,
    reference: np.ndarray,
    chunk_size: int = 4096,
) -> np.ndarray:
    chunks = []
    for start in range(0, query.shape[0], chunk_size):
        diff = query[start : start + chunk_size, None, :] - reference[None, :, :]
        chunks.append(np.linalg.norm(diff, axis=-1))
    return np.concatenate(chunks, axis=0)


def _rmse_1d(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((prediction - target) ** 2)))


def _relative_active_fraction(
    scores: np.ndarray,
    max_scores: np.ndarray,
    *,
    relative_threshold: float,
) -> np.ndarray:
    threshold = relative_threshold * max_scores[..., None]
    active = (scores >= threshold) & (scores > 0.0)
    return np.mean(active, axis=-1)
