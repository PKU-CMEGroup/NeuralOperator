"""Time-alignment diagnostics for autoregressive rollout arrays."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TimeAlignment:
    """Best ground-truth time for every predicted rollout time."""

    mse_matrix: np.ndarray
    best_target_indices: np.ndarray
    best_mse: np.ndarray
    diagonal_mse: np.ndarray


def _select_rollout_values(
    values: np.ndarray,
    *,
    variables: np.ndarray | list[int] | tuple[int, ...] | None,
    node_mask: np.ndarray | None,
) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 3:
        raise ValueError(f"rollout arrays must have shape [time, nodes, channels], got {arr.shape}")
    if variables is None:
        channel_indices = np.arange(arr.shape[-1])
    else:
        channel_indices = np.asarray(variables, dtype=np.int64)
        if channel_indices.ndim != 1:
            raise ValueError("variables must be a one-dimensional index list")
        if np.any(channel_indices < 0) or np.any(channel_indices >= arr.shape[-1]):
            raise IndexError(f"variables {channel_indices.tolist()} outside channel range [0, {arr.shape[-1]})")
    if node_mask is None:
        mask = slice(None)
    else:
        mask = np.asarray(node_mask, dtype=bool)
        if mask.ndim != 1 or mask.shape[0] != arr.shape[1]:
            raise ValueError(f"node_mask must have shape [{arr.shape[1]}], got {mask.shape}")
        if not np.any(mask):
            raise ValueError("node_mask selects no nodes")
    return arr[:, mask, :][..., channel_indices].astype(np.float64, copy=False)


def compute_time_alignment(
    prediction: np.ndarray,
    truth: np.ndarray,
    *,
    variables: np.ndarray | list[int] | tuple[int, ...] | None = None,
    node_mask: np.ndarray | None = None,
) -> TimeAlignment:
    """Compute best target time for each predicted time by MSE.

    Args:
        prediction: Predicted rollout array with shape ``[T_pred, N, C]``.
        truth: Ground-truth rollout array with shape ``[T_truth, N, C]``.
        variables: Optional channel indices to compare.
        node_mask: Optional boolean node mask.

    Returns:
        ``TimeAlignment`` containing an MSE matrix with shape
        ``[T_pred, T_truth]`` and the argmin target index per predicted time.
    """

    pred = _select_rollout_values(prediction, variables=variables, node_mask=node_mask)
    target = _select_rollout_values(truth, variables=variables, node_mask=node_mask)
    if pred.shape[1:] != target.shape[1:]:
        raise ValueError(f"prediction and truth spatial/channel shapes disagree: {pred.shape} vs {target.shape}")

    pred_flat = pred.reshape(pred.shape[0], -1)
    target_flat = target.reshape(target.shape[0], -1)
    if pred_flat.shape[1] == 0:
        raise ValueError("selected comparison field is empty")

    pred_norm = np.mean(pred_flat * pred_flat, axis=1)[:, None]
    target_norm = np.mean(target_flat * target_flat, axis=1)[None, :]
    cross = (pred_flat @ target_flat.T) * (2.0 / pred_flat.shape[1])
    mse = np.maximum(pred_norm + target_norm - cross, 0.0)

    best_target_indices = np.argmin(mse, axis=1)
    best_mse = mse[np.arange(mse.shape[0]), best_target_indices]
    diagonal_mse = np.full(mse.shape[0], np.nan, dtype=np.float64)
    diagonal_count = min(mse.shape)
    diagonal_mse[:diagonal_count] = mse[np.arange(diagonal_count), np.arange(diagonal_count)]
    return TimeAlignment(
        mse_matrix=mse,
        best_target_indices=best_target_indices,
        best_mse=best_mse,
        diagonal_mse=diagonal_mse,
    )


def fit_best_time_curve(pred_steps: np.ndarray, best_target_steps: np.ndarray) -> dict[str, float]:
    """Fit ``best_target_step = slope * pred_step + intercept``."""

    x = np.asarray(pred_steps, dtype=np.float64)
    y = np.asarray(best_target_steps, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("pred_steps and best_target_steps must be one-dimensional arrays with the same shape")
    if x.size < 2:
        raise ValueError("at least two points are required for a line fit")
    slope, intercept = np.polyfit(x, y, deg=1)
    fitted = slope * x + intercept
    residual = y - fitted
    ss_res = float(np.sum(residual * residual))
    centered = y - float(np.mean(y))
    ss_tot = float(np.sum(centered * centered))
    r2 = 1.0 if ss_tot <= 0.0 and ss_res <= 1.0e-12 else 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0
    lag = y - x
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": float(r2),
        "mean_lag": float(np.mean(lag)),
        "median_lag": float(np.median(lag)),
        "mean_abs_lag": float(np.mean(np.abs(lag))),
    }
