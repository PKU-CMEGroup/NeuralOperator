"""Error metrics for PDE states.

The default convention is that the last axis is the spatial grid. For a
trajectory array with shape ``(num_samples, num_times, num_grid)``, calling
``relative_l2_error(prediction, truth)`` returns one error per sample and
time, with shape ``(num_samples, num_times)``.
"""

from __future__ import annotations

from typing import Any

import numpy as np


ArrayLike = Any


def _as_arrays(prediction: ArrayLike, target: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    if pred.shape != truth.shape:
        raise ValueError(
            f"prediction and target must have the same shape, got "
            f"{pred.shape} and {truth.shape}"
        )
    return pred, truth


def absolute_l2_error(
    prediction: ArrayLike,
    target: ArrayLike,
    *,
    axis: int | tuple[int, ...] = -1,
) -> np.ndarray:
    """Return ``||prediction - target||_2`` along the chosen axis."""

    pred, truth = _as_arrays(prediction, target)
    return np.sqrt(np.sum((pred - truth) ** 2, axis=axis))


def relative_l2_error(
    prediction: ArrayLike,
    target: ArrayLike,
    *,
    axis: int | tuple[int, ...] = -1,
    eps: float = 1e-12,
) -> np.ndarray:
    """Return ``||prediction - target||_2 / max(||target||_2, eps)``."""

    pred, truth = _as_arrays(prediction, target)
    numerator = np.sqrt(np.sum((pred - truth) ** 2, axis=axis))
    denominator = np.sqrt(np.sum(truth**2, axis=axis))
    return numerator / np.maximum(denominator, eps)


def root_mean_squared_error(
    prediction: ArrayLike,
    target: ArrayLike,
    *,
    axis: int | tuple[int, ...] = -1,
) -> np.ndarray:
    """Return root mean squared error along the chosen axis."""

    pred, truth = _as_arrays(prediction, target)
    return np.sqrt(np.mean((pred - truth) ** 2, axis=axis))


def analysis_improvement(
    forecast: ArrayLike,
    analysis: ArrayLike,
    target: ArrayLike,
    *,
    axis: int | tuple[int, ...] = -1,
    relative: bool = True,
    eps: float = 1e-12,
) -> dict[str, np.ndarray]:
    """Compare forecast-before-correction and analysis-after-correction errors.

    Positive ``improvement`` means the analysis is closer to the target than
    the forecast. This small helper protects an important project habit:
    never report corrected error without also reporting forecast error.
    """

    metric = relative_l2_error if relative else absolute_l2_error
    if relative:
        forecast_error = metric(forecast, target, axis=axis, eps=eps)
        analysis_error = metric(analysis, target, axis=axis, eps=eps)
    else:
        forecast_error = metric(forecast, target, axis=axis)
        analysis_error = metric(analysis, target, axis=axis)
    return {
        "forecast_error": forecast_error,
        "analysis_error": analysis_error,
        "improvement": forecast_error - analysis_error,
    }

