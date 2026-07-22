"""Shared physical and graph-native metrics for the shock--vortex benchmark."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from utility.time_dependent_no.euler2d_metrics import (
    front_centroid_distance,
    front_distance_metrics,
    front_overlap_metrics,
    shock_front_masks,
    shock_smearing_metrics,
)
from utility.time_dependent_no.pcno_ripple_diagnostics import (
    conservative_to_primitive_raw,
    node_highpass_amplitude,
)


def _dilate_mask(mask: np.ndarray, edges: np.ndarray) -> np.ndarray:
    result = np.asarray(mask, dtype=bool).copy()
    edge_index = np.asarray(edges, dtype=np.int64)
    expanded = result.copy()
    np.logical_or.at(expanded, edge_index[:, 0], result[edge_index[:, 1]])
    np.logical_or.at(expanded, edge_index[:, 1], result[edge_index[:, 0]])
    return expanded


def _scalar(value: Any) -> float | None:
    numeric = float(np.asarray(value).reshape(-1)[0])
    return numeric if math.isfinite(numeric) else None


def _log_ratio_error(value: float | None) -> float | None:
    if value is None or value <= 0.0:
        return None
    return abs(math.log(value))


def endpoint_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    positions: np.ndarray,
    edges: np.ndarray,
    volumes: np.ndarray,
    component_scale: np.ndarray,
    gamma: float,
    shock_quantile: float,
    vortex_center: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Evaluate front, shock, vortex, and smooth-region endpoint structure."""

    pred_primitive = conservative_to_primitive_raw(prediction, gamma=gamma)
    target_primitive = conservative_to_primitive_raw(target, gamma=gamma)
    fronts = shock_front_masks(
        pred_primitive, target_primitive, edges, quantile=shock_quantile
    )
    overlap = front_overlap_metrics(fronts["prediction_mask"], fronts["target_mask"])
    distances = front_distance_metrics(
        fronts["prediction_mask"], fronts["target_mask"], positions
    )
    smearing = shock_smearing_metrics(
        pred_primitive, target_primitive, edges, scalar_index=3
    )
    smooth = ~_dilate_mask(fronts["target_mask"], edges)
    if not np.any(smooth):
        smooth = ~fronts["target_mask"]
    scale = np.asarray(component_scale, dtype=np.float64).reshape(1, 4)
    scaled_error = (np.asarray(prediction) - np.asarray(target)) / scale
    scaled_target = np.asarray(target) / scale
    normalized_volume = np.asarray(volumes, dtype=np.float64)
    normalized_volume = normalized_volume / normalized_volume.sum()
    smooth_weight = normalized_volume * smooth
    smooth_numerator = float(np.sum(smooth_weight[:, None] * scaled_error**2))
    smooth_denominator = float(np.sum(smooth_weight[:, None] * scaled_target**2))
    highpass = node_highpass_amplitude(scaled_error, edges)
    thickness_ratio = _scalar(smearing["thickness_ratio"])
    strength_ratio = _scalar(smearing["strength_ratio"])
    vortex_core_error = None
    if vortex_center is not None:
        center = np.asarray(vortex_center, dtype=np.float64)
        vortex_window = (
            np.sum((np.asarray(positions) - center[None, :]) ** 2, axis=1) <= 0.18**2
        )
        if not np.any(vortex_window):
            raise ValueError("declared moving-vortex window contains no cells")
        predicted_core_density = float(np.min(pred_primitive[vortex_window, 0]))
        target_core_density = float(np.min(target_primitive[vortex_window, 0]))
        vortex_core_error = abs(predicted_core_density - target_core_density) / max(
            abs(target_core_density), 1.0e-30
        )
    return {
        "front_iou": _scalar(overlap["iou"]),
        "front_precision": _scalar(overlap["precision"]),
        "front_recall": _scalar(overlap["recall"]),
        "front_centroid_distance": _scalar(
            front_centroid_distance(
                fronts["prediction_mask"], fronts["target_mask"], positions
            )
        ),
        "front_symmetric_chamfer": _scalar(distances["symmetric_chamfer_mean"]),
        "shock_thickness_ratio": thickness_ratio,
        "shock_strength_ratio": strength_ratio,
        "shock_thickness_log_error": _log_ratio_error(thickness_ratio),
        "shock_strength_log_error": _log_ratio_error(strength_ratio),
        "vortex_core_density_relative_error": vortex_core_error,
        "smooth_region_contract": "target_pressure_front_dilated_one_graph_hop",
        "smooth_region_volume_fraction": float(np.sum(smooth_weight)),
        "smooth_region_scaled_relative_l2": math.sqrt(
            smooth_numerator / max(smooth_denominator, 1.0e-30)
        ),
        "smooth_region_graph_highpass_energy": float(
            np.sum(smooth_weight * highpass**2) / max(np.sum(smooth_weight), 1.0e-30)
        ),
    }


def physical_call_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    initial: np.ndarray,
    *,
    volumes: np.ndarray,
    reference_cumulative_boundary_exchange: np.ndarray,
    component_scale: np.ndarray,
) -> dict[str, Any]:
    """Compare physical cell totals with one declared boundary exchange."""

    pred = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    first = np.asarray(initial, dtype=np.float64)
    volume = np.asarray(volumes, dtype=np.float64).reshape(-1)
    exchange = np.asarray(reference_cumulative_boundary_exchange, dtype=np.float64)
    if (
        pred.shape != truth.shape
        or pred.shape != first.shape
        or pred.shape != (volume.size, 4)
    ):
        raise ValueError("states and physical volumes have incompatible shapes")
    if exchange.shape != (4,):
        raise ValueError(
            "reference cumulative boundary exchange must have four components"
        )
    pred_total = np.sum(volume[:, None] * pred, axis=0)
    target_total = np.sum(volume[:, None] * truth, axis=0)
    initial_total = np.sum(volume[:, None] * first, axis=0)
    pred_implied_exchange = -(pred_total - initial_total)
    scale = np.asarray(component_scale, dtype=np.float64) * float(np.sum(volume))

    def scaled_rmse(value: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.square(value / scale))))

    total_error = pred_total - target_total
    exchange_error = pred_implied_exchange - exchange
    target_balance = target_total - initial_total + exchange
    pred_balance = pred_total - initial_total + exchange
    return {
        "physical_total_error": total_error,
        "physical_total_component_scaled_rmse": scaled_rmse(total_error),
        "state_implied_outward_boundary_exchange": pred_implied_exchange,
        "reference_outward_boundary_exchange": exchange,
        "state_implied_boundary_exchange_error": exchange_error,
        "state_implied_boundary_exchange_component_scaled_rmse": scaled_rmse(
            exchange_error
        ),
        "prediction_reference_balance_defect": pred_balance,
        "prediction_reference_balance_component_scaled_rmse": scaled_rmse(pred_balance),
        "target_reference_balance_defect": target_balance,
        "target_reference_balance_component_scaled_rmse": scaled_rmse(target_balance),
    }
