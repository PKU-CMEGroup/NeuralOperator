"""Interface-latent diagnostics for CPGNet-style Euler updates.

The author CPGNet path learns one primitive state per directed edge and then
pairs opposite directions into a Local Lax-Friedrichs flux.  This module keeps
the reusable analysis NumPy-only so it can be tested without importing the
author PyTorch Geometric implementation.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from utility.time_dependent_no.euler2d import (
    CONSERVATIVE_NAMES,
    PRIMITIVE_NAMES,
    primitive_to_conservative,
)


ArrayLike = Any
EPS = 1.0e-12
WAVE_TYPE_ORDER = (
    "smooth",
    "compression",
    "rarefaction",
    "contact_like",
    "shock_front",
    "other",
)


def normalize_edge_index(
    edge_index: ArrayLike,
    *,
    num_nodes: int | None = None,
) -> np.ndarray:
    """Return edges as an ``(E, 2)`` int64 array."""

    edges = np.asarray(edge_index, dtype=np.int64)
    if edges.ndim != 2:
        raise ValueError("edge_index must have two dimensions")
    if edges.shape[1] == 2:
        out = edges
    elif edges.shape[0] == 2:
        out = edges.T
    else:
        raise ValueError(f"edge_index must have shape (E, 2) or (2, E), got {edges.shape}")
    if num_nodes is not None and out.size:
        if np.min(out) < 0 or np.max(out) >= num_nodes:
            raise ValueError("edge_index contains node ids outside [0, num_nodes)")
    return np.asarray(out, dtype=np.int64)


def split_directed_reconstruct_prims(
    reconstruct_prims: ArrayLike,
    *,
    num_unique_edges: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Split ``[left states, right states]`` from the author directed layout."""

    prim = _validate_primitive(reconstruct_prims, name="reconstruct_prims")
    if num_unique_edges is None:
        if prim.shape[0] % 2:
            raise ValueError("reconstruct_prims must contain 2 * num_unique_edges rows")
        num_unique_edges = prim.shape[0] // 2
    if prim.shape[0] != 2 * num_unique_edges:
        raise ValueError(
            "reconstruct_prims length must equal 2 * num_unique_edges, got "
            f"{prim.shape[0]} and {num_unique_edges}"
        )
    return prim[:num_unique_edges], prim[num_unique_edges:]


def primitive_flux(
    primitive: ArrayLike,
    normals: ArrayLike,
    *,
    gamma: float = 1.4,
) -> np.ndarray:
    """Return the 2D Euler physical flux projected onto edge normals."""

    prim = _validate_primitive(primitive, name="primitive")
    normal = _validate_normals(normals, prim.shape[:-1])
    rho = prim[..., 0]
    v1 = prim[..., 1]
    v2 = prim[..., 2]
    pressure = prim[..., 3]
    nx = normal[..., 0]
    ny = normal[..., 1]
    normal_velocity = v1 * nx + v2 * ny
    rho_vn = rho * normal_velocity
    energy = pressure / (gamma - 1.0) + 0.5 * rho * (v1 * v1 + v2 * v2)
    return np.stack(
        (
            rho_vn,
            rho_vn * v1 + pressure * nx,
            rho_vn * v2 + pressure * ny,
            (energy + pressure) * normal_velocity,
        ),
        axis=-1,
    )


def llf_flux_decomposition(
    left_primitives: ArrayLike,
    right_primitives: ArrayLike,
    normals: ArrayLike,
    *,
    gamma: float = 1.4,
) -> dict[str, np.ndarray]:
    """Decompose Local Lax-Friedrichs flux into central and dissipative terms."""

    left = _validate_primitive(left_primitives, name="left_primitives")
    right = _validate_primitive(right_primitives, name="right_primitives")
    if left.shape != right.shape:
        raise ValueError(f"left/right primitive shapes differ: {left.shape} vs {right.shape}")
    normal = _validate_normals(normals, left.shape[:-1])
    cons_left = primitive_to_conservative(left, gamma=gamma)
    cons_right = primitive_to_conservative(right, gamma=gamma)
    central = 0.5 * (
        primitive_flux(left, normal, gamma=gamma)
        + primitive_flux(right, normal, gamma=gamma)
    )
    vn_left = left[..., 1] * normal[..., 0] + left[..., 2] * normal[..., 1]
    vn_right = right[..., 1] * normal[..., 0] + right[..., 2] * normal[..., 1]
    c_left = _sound_speed(left[..., 0], left[..., 3], gamma=gamma)
    c_right = _sound_speed(right[..., 0], right[..., 3], gamma=gamma)
    normal_norm = np.linalg.norm(normal, axis=-1)
    wave_speed = np.maximum(np.abs(vn_left), np.abs(vn_right)) + np.maximum(
        c_left, c_right
    ) * normal_norm
    dissipation = -0.5 * wave_speed[..., None] * (cons_right - cons_left)
    return {
        "flux": central + dissipation,
        "central": central,
        "dissipation": dissipation,
        "wave_speed": wave_speed,
        "normal_velocity_left": vn_left,
        "normal_velocity_right": vn_right,
        "sound_speed_left": c_left,
        "sound_speed_right": c_right,
        "conservative_left": cons_left,
        "conservative_right": cons_right,
    }


def conservative_update_from_flux(
    directed_flux: ArrayLike,
    edge_factor: ArrayLike,
    full_edge_index: ArrayLike,
    *,
    num_nodes: int,
) -> dict[str, np.ndarray]:
    """Recreate the author finite-volume update from unique directed fluxes."""

    flux = np.asarray(directed_flux, dtype=np.float64)
    if flux.ndim != 2 or flux.shape[1] != len(CONSERVATIVE_NAMES):
        raise ValueError("directed_flux must have shape (E, 4)")
    edges = normalize_edge_index(full_edge_index, num_nodes=num_nodes)
    if edges.shape[0] != 2 * flux.shape[0]:
        raise ValueError(
            "full_edge_index must contain original plus reversed edges; got "
            f"{edges.shape[0]} rows for {flux.shape[0]} unique fluxes"
        )
    factors = np.asarray(edge_factor, dtype=np.float64)
    if factors.ndim == 1:
        factors = factors[:, None]
    if factors.shape != (edges.shape[0], 1):
        raise ValueError(
            f"edge_factor must have shape ({edges.shape[0]}, 1), got {factors.shape}"
        )
    messages = factors * np.concatenate((flux, -flux), axis=0)
    divergence = np.zeros((num_nodes, flux.shape[1]), dtype=np.float64)
    np.add.at(divergence, edges[:, 1], messages)
    return {
        "messages": messages,
        "flux_divergence": divergence,
        "conservative_delta": -divergence,
    }


def admissibility_summary(
    left_primitives: ArrayLike,
    right_primitives: ArrayLike,
    node_primitives: ArrayLike,
    *,
    node_mask: ArrayLike | None = None,
) -> dict[str, dict[str, float | int | bool]]:
    """Summarize density/pressure positivity for interface and node states."""

    left = _validate_primitive(left_primitives, name="left_primitives")
    right = _validate_primitive(right_primitives, name="right_primitives")
    node = _validate_primitive(node_primitives, name="node_primitives")
    if node_mask is not None:
        mask = np.asarray(node_mask, dtype=bool)
        if mask.ndim != 1 or mask.shape[0] != node.shape[0]:
            raise ValueError("node_mask must match the node axis")
        node = node[mask]
    return {
        "left_interface": _positivity_summary(left),
        "right_interface": _positivity_summary(right),
        "all_interface": _positivity_summary(np.concatenate((left, right), axis=0)),
        "nodes": _positivity_summary(node),
    }


def trace_likeness_arrays(
    left_primitives: ArrayLike,
    right_primitives: ArrayLike,
    node_primitives: ArrayLike,
    unique_edge_index: ArrayLike,
) -> dict[str, np.ndarray]:
    """Return per-edge arrays for one-sided trace diagnostics.

    In the author layout, ``edge_index[1]`` is the central/owner node for the
    first half of ``reconstruct_prims`` and ``edge_index[0]`` owns the reversed
    state in the second half.
    """

    left = _validate_primitive(left_primitives, name="left_primitives")
    right = _validate_primitive(right_primitives, name="right_primitives")
    node = _validate_primitive(node_primitives, name="node_primitives")
    if left.shape != right.shape:
        raise ValueError("left and right primitives must have the same shape")
    edges = normalize_edge_index(unique_edge_index, num_nodes=node.shape[0])
    if edges.shape[0] != left.shape[0]:
        raise ValueError("unique_edge_index length must match left/right states")
    src = edges[:, 0]
    dst = edges[:, 1]
    left_owner = node[dst]
    left_neighbor = node[src]
    right_owner = node[src]
    right_neighbor = node[dst]
    lo = np.minimum(left_owner, left_neighbor)
    hi = np.maximum(left_owner, left_neighbor)
    left_overshoot = np.maximum(lo - left, 0.0) + np.maximum(left - hi, 0.0)
    right_overshoot = np.maximum(lo - right, 0.0) + np.maximum(right - hi, 0.0)
    return {
        "left_owner_abs": np.abs(left - left_owner),
        "left_neighbor_abs": np.abs(left - left_neighbor),
        "right_owner_abs": np.abs(right - right_owner),
        "right_neighbor_abs": np.abs(right - right_neighbor),
        "left_owner_l2": np.linalg.norm(left - left_owner, axis=-1),
        "left_neighbor_l2": np.linalg.norm(left - left_neighbor, axis=-1),
        "right_owner_l2": np.linalg.norm(right - right_owner, axis=-1),
        "right_neighbor_l2": np.linalg.norm(right - right_neighbor, axis=-1),
        "left_bounded": left_overshoot <= 0.0,
        "right_bounded": right_overshoot <= 0.0,
        "left_overshoot": left_overshoot,
        "right_overshoot": right_overshoot,
    }


def owner_neighbor_reconstruct_prims(
    node_primitives: ArrayLike,
    unique_edge_index: ArrayLike,
) -> tuple[np.ndarray, np.ndarray]:
    """Return piecewise-constant physical traces in the author pair layout."""

    node = _validate_primitive(node_primitives, name="node_primitives")
    edges = normalize_edge_index(unique_edge_index, num_nodes=node.shape[0])
    src = edges[:, 0]
    dst = edges[:, 1]
    return node[dst].copy(), node[src].copy()


def midpoint_reconstruct_prims(
    node_primitives: ArrayLike,
    unique_edge_index: ArrayLike,
) -> tuple[np.ndarray, np.ndarray]:
    """Return equal left/right edge midpoint primitive states."""

    left, right = owner_neighbor_reconstruct_prims(node_primitives, unique_edge_index)
    midpoint = 0.5 * (left + right)
    return midpoint.copy(), midpoint.copy()


def time_midpoint_reconstruct_prims(
    current_primitives: ArrayLike,
    target_primitives: ArrayLike,
    unique_edge_index: ArrayLike,
) -> tuple[np.ndarray, np.ndarray]:
    """Return owner/neighbor traces averaged between current and target frames."""

    current = _validate_primitive(current_primitives, name="current_primitives")
    target = _validate_primitive(target_primitives, name="target_primitives")
    if current.shape != target.shape:
        raise ValueError("current_primitives and target_primitives shapes must match")
    return owner_neighbor_reconstruct_prims(0.5 * (current + target), unique_edge_index)


def clip_reconstruct_to_edge_box(
    left_primitives: ArrayLike,
    right_primitives: ArrayLike,
    node_primitives: ArrayLike,
    unique_edge_index: ArrayLike,
    *,
    margin_factor: float = 0.0,
    margin_abs: float = 0.0,
    positivity_floor: float = 1.0e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Clip learned states to the local owner-neighbor primitive box."""

    if margin_factor < 0.0 or margin_abs < 0.0:
        raise ValueError("margins must be nonnegative")
    left = _validate_primitive(left_primitives, name="left_primitives")
    right = _validate_primitive(right_primitives, name="right_primitives")
    owner_left, owner_right = owner_neighbor_reconstruct_prims(
        node_primitives, unique_edge_index
    )
    if left.shape != owner_left.shape or right.shape != owner_right.shape:
        raise ValueError("left/right primitive shapes must match edge count")
    lo = np.minimum(owner_left, owner_right)
    hi = np.maximum(owner_left, owner_right)
    width = hi - lo
    margin = margin_abs + margin_factor * width
    lo = lo - margin
    hi = hi + margin
    lo[:, 0] = np.maximum(lo[:, 0], positivity_floor)
    lo[:, 3] = np.maximum(lo[:, 3], positivity_floor)
    return np.clip(left, lo, hi), np.clip(right, lo, hi)


def flux_error_summary(
    candidate_flux: ArrayLike,
    reference_flux: ArrayLike,
    *,
    edge_mask: ArrayLike | None = None,
) -> dict[str, Any]:
    """Summarize candidate edge-flux error against a reference flux."""

    candidate = _validate_conservative(candidate_flux, name="candidate_flux")
    reference = _validate_conservative(reference_flux, name="reference_flux")
    if candidate.shape != reference.shape:
        raise ValueError("candidate_flux and reference_flux shapes must match")
    mask = _edge_mask(edge_mask, candidate.shape[0])
    if not np.any(mask):
        return _empty_flux_summary()
    err = candidate[mask] - reference[mask]
    ref = reference[mask]
    err_l2 = np.sqrt(np.sum(err * err, axis=0))
    ref_l2 = np.sqrt(np.sum(ref * ref, axis=0))
    edge_err_norm = np.linalg.norm(err, axis=-1)
    edge_ref_norm = np.linalg.norm(ref, axis=-1)
    return {
        "edge_count": int(np.count_nonzero(mask)),
        "rmse": _by_conservative(np.sqrt(np.mean(err * err, axis=0))),
        "relative_l2": _by_conservative(err_l2 / np.maximum(ref_l2, EPS)),
        "edge_relative_l2": float(
            np.sqrt(np.sum(edge_err_norm * edge_err_norm))
            / max(np.sqrt(np.sum(edge_ref_norm * edge_ref_norm)), EPS)
        ),
        "edge_error_norm_mean": float(np.mean(edge_err_norm)),
        "edge_reference_norm_mean": float(np.mean(edge_ref_norm)),
        "edge_cosine_mean": _safe_edge_cosine(candidate[mask], reference[mask]),
    }


def divergence_cancellation_summary(
    candidate_flux: ArrayLike,
    reference_flux: ArrayLike,
    edge_factor: ArrayLike,
    full_edge_index: ArrayLike,
    *,
    num_nodes: int,
    node_mask: ArrayLike | None = None,
) -> dict[str, float]:
    """Compare edge-flux perturbation size with its induced update difference."""

    candidate = _validate_conservative(candidate_flux, name="candidate_flux")
    reference = _validate_conservative(reference_flux, name="reference_flux")
    if candidate.shape != reference.shape:
        raise ValueError("candidate_flux and reference_flux shapes must match")
    cand_update = conservative_update_from_flux(
        candidate, edge_factor, full_edge_index, num_nodes=num_nodes
    )
    ref_update = conservative_update_from_flux(
        reference, edge_factor, full_edge_index, num_nodes=num_nodes
    )
    if node_mask is None:
        mask = np.ones(num_nodes, dtype=bool)
    else:
        mask = np.asarray(node_mask, dtype=bool)
        if mask.ndim != 1 or mask.shape[0] != num_nodes:
            raise ValueError("node_mask must match the node axis")
    edge_delta = candidate - reference
    factors = np.asarray(edge_factor, dtype=np.float64)
    if factors.ndim == 1:
        factors = factors[:, None]
    unique_count = candidate.shape[0]
    if factors.shape[0] != 2 * unique_count:
        raise ValueError("edge_factor must contain unique and reversed factors")
    weighted_messages = np.concatenate(
        (factors[:unique_count] * edge_delta, -factors[unique_count:] * edge_delta),
        axis=0,
    )
    update_delta = cand_update["conservative_delta"] - ref_update["conservative_delta"]
    message_norm = float(np.linalg.norm(weighted_messages))
    update_norm = float(np.linalg.norm(update_delta[mask]))
    return {
        "weighted_message_delta_l2": message_norm,
        "node_update_delta_l2": update_norm,
        "update_to_message_l2_ratio": update_norm / max(message_norm, EPS),
    }

def summarize_trace_likeness(
    trace_arrays: dict[str, np.ndarray],
    *,
    edge_mask: ArrayLike | None = None,
) -> dict[str, Any]:
    """Summarize per-edge trace-likeness arrays."""

    mask = _edge_mask(edge_mask, next(iter(trace_arrays.values())).shape[0])
    left_owner_l2 = trace_arrays["left_owner_l2"][mask]
    left_neighbor_l2 = trace_arrays["left_neighbor_l2"][mask]
    right_owner_l2 = trace_arrays["right_owner_l2"][mask]
    right_neighbor_l2 = trace_arrays["right_neighbor_l2"][mask]
    left_overshoot = trace_arrays["left_overshoot"][mask]
    right_overshoot = trace_arrays["right_overshoot"][mask]
    left_bounded = trace_arrays["left_bounded"][mask]
    right_bounded = trace_arrays["right_bounded"][mask]
    return {
        "edge_count": int(np.count_nonzero(mask)),
        "left_owner_l2_mean": _safe_mean(left_owner_l2),
        "left_neighbor_l2_mean": _safe_mean(left_neighbor_l2),
        "right_owner_l2_mean": _safe_mean(right_owner_l2),
        "right_neighbor_l2_mean": _safe_mean(right_neighbor_l2),
        "left_owner_closer_fraction": _safe_mean(left_owner_l2 <= left_neighbor_l2),
        "right_owner_closer_fraction": _safe_mean(right_owner_l2 <= right_neighbor_l2),
        "left_abs_owner_mean": _by_primitive_mean(trace_arrays["left_owner_abs"][mask]),
        "right_abs_owner_mean": _by_primitive_mean(trace_arrays["right_owner_abs"][mask]),
        "bounded_fraction": _by_primitive_mean(
            np.concatenate((left_bounded, right_bounded), axis=0).astype(np.float64)
        ),
        "overshoot_mean": _by_primitive_mean(
            np.concatenate((left_overshoot, right_overshoot), axis=0)
        ),
        "overshoot_max": _by_primitive_max(
            np.concatenate((left_overshoot, right_overshoot), axis=0)
        ),
    }


def edge_pressure_jump_scores(
    primitives: ArrayLike,
    unique_edge_index: ArrayLike,
) -> np.ndarray:
    """Return absolute pressure jumps on unique edges."""

    prim = _validate_primitive(primitives, name="primitives")
    edges = normalize_edge_index(unique_edge_index, num_nodes=prim.shape[0])
    return np.abs(prim[edges[:, 1], 3] - prim[edges[:, 0], 3])


def edge_shock_mask_from_scores(
    scores: ArrayLike,
    *,
    quantile: float = 0.9,
    edge_mask: ArrayLike | None = None,
) -> np.ndarray:
    """Select high-pressure-jump edges as a shock-front proxy."""

    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("scores must be one-dimensional")
    mask = _edge_mask(edge_mask, values.shape[0])
    out = np.zeros(values.shape[0], dtype=bool)
    if not np.any(mask):
        return out
    selected = values[mask]
    threshold = np.quantile(selected, quantile)
    out[mask] = (selected >= threshold) & (selected > 0.0)
    return out


def node_mask_from_edge_mask(
    unique_edge_index: ArrayLike,
    edge_mask: ArrayLike,
    *,
    num_nodes: int,
) -> np.ndarray:
    """Return nodes incident to selected unique edges."""

    edges = normalize_edge_index(unique_edge_index, num_nodes=num_nodes)
    mask = _edge_mask(edge_mask, edges.shape[0])
    nodes = np.zeros(num_nodes, dtype=bool)
    if np.any(mask):
        nodes[edges[mask, 0]] = True
        nodes[edges[mask, 1]] = True
    return nodes


def update_error_summary(
    induced_delta: ArrayLike,
    true_delta: ArrayLike,
    *,
    node_mask: ArrayLike | None = None,
) -> dict[str, Any]:
    """Summarize conservative-update error on selected nodes."""

    induced = _validate_conservative(induced_delta, name="induced_delta")
    truth = _validate_conservative(true_delta, name="true_delta")
    if induced.shape != truth.shape:
        raise ValueError("induced_delta and true_delta shapes must match")
    if node_mask is None:
        mask = np.ones(induced.shape[0], dtype=bool)
    else:
        mask = np.asarray(node_mask, dtype=bool)
        if mask.ndim != 1 or mask.shape[0] != induced.shape[0]:
            raise ValueError("node_mask must match the node axis")
    if not np.any(mask):
        return _empty_update_summary()
    err = induced[mask] - truth[mask]
    return {
        "node_count": int(np.count_nonzero(mask)),
        "rmse": _by_conservative(np.sqrt(np.mean(err * err, axis=0))),
        "relative_l2": _by_conservative(
            np.sqrt(np.sum(err * err, axis=0))
            / np.maximum(np.sqrt(np.sum(truth[mask] * truth[mask], axis=0)), EPS)
        ),
        "mean_signed_error": _by_conservative(np.mean(err, axis=0)),
        "induced_delta_l2_mean": float(np.mean(np.linalg.norm(induced[mask], axis=-1))),
        "true_delta_l2_mean": float(np.mean(np.linalg.norm(truth[mask], axis=-1))),
    }


def classify_wave_edges(
    node_primitives: ArrayLike,
    unique_edge_index: ArrayLike,
    normals: ArrayLike,
    *,
    shock_edge_mask: ArrayLike | None = None,
    edge_mask: ArrayLike | None = None,
    high_quantile: float = 0.75,
    smooth_quantile: float = 0.50,
) -> dict[str, Any]:
    """Approximate wave-type edge strata from local primitive jumps."""

    if not 0.0 <= high_quantile <= 1.0 or not 0.0 <= smooth_quantile <= 1.0:
        raise ValueError("quantiles must be in [0, 1]")
    prim = _validate_primitive(node_primitives, name="node_primitives")
    edges = normalize_edge_index(unique_edge_index, num_nodes=prim.shape[0])
    normal = _validate_normals(normals, (edges.shape[0],))
    valid = _edge_mask(edge_mask, edges.shape[0])
    shock = (
        np.asarray(shock_edge_mask, dtype=bool).copy()
        if shock_edge_mask is not None
        else np.zeros(edges.shape[0], dtype=bool)
    )
    if shock.shape != (edges.shape[0],):
        raise ValueError("shock_edge_mask must match the edge axis")
    shock &= valid

    src = edges[:, 0]
    dst = edges[:, 1]
    left = prim[dst]
    right = prim[src]
    tangent = np.stack((-normal[:, 1], normal[:, 0]), axis=-1)
    vn_left = left[:, 1] * normal[:, 0] + left[:, 2] * normal[:, 1]
    vn_right = right[:, 1] * normal[:, 0] + right[:, 2] * normal[:, 1]
    vt_left = left[:, 1] * tangent[:, 0] + left[:, 2] * tangent[:, 1]
    vt_right = right[:, 1] * tangent[:, 0] + right[:, 2] * tangent[:, 1]
    features = {
        "pressure_jump": right[:, 3] - left[:, 3],
        "density_jump": right[:, 0] - left[:, 0],
        "normal_velocity_jump": vn_right - vn_left,
        "tangential_velocity_jump": vt_right - vt_left,
    }
    abs_p = np.abs(features["pressure_jump"])
    abs_rho = np.abs(features["density_jump"])
    abs_vn = np.abs(features["normal_velocity_jump"])
    abs_vt = np.abs(features["tangential_velocity_jump"])
    nonshock = valid & ~shock
    p_hi = _masked_quantile(abs_p, nonshock, high_quantile)
    rho_hi = _masked_quantile(abs_rho, nonshock, high_quantile)
    vn_hi = _masked_quantile(abs_vn, nonshock, high_quantile)
    vt_hi = _masked_quantile(abs_vt, nonshock, high_quantile)
    p_lo = _masked_quantile(abs_p, nonshock, smooth_quantile)
    rho_lo = _masked_quantile(abs_rho, nonshock, smooth_quantile)
    vn_lo = _masked_quantile(abs_vn, nonshock, smooth_quantile)

    smooth = nonshock & (abs_p <= p_lo) & (abs_rho <= rho_lo) & (abs_vn <= vn_lo)
    contact = (
        nonshock
        & ~smooth
        & (abs_p <= p_lo)
        & ((abs_rho >= rho_hi) | (abs_vt >= vt_hi))
    )
    compression = (
        nonshock
        & ~smooth
        & ~contact
        & (features["normal_velocity_jump"] <= -vn_hi)
    )
    rarefaction = (
        nonshock
        & ~smooth
        & ~contact
        & (features["normal_velocity_jump"] >= vn_hi)
    )
    assigned = smooth | contact | compression | rarefaction | shock
    masks = {
        "smooth": smooth,
        "compression": compression,
        "rarefaction": rarefaction,
        "contact_like": contact,
        "shock_front": shock,
        "other": valid & ~assigned,
    }
    thresholds = {
        "pressure_high": float(p_hi),
        "density_high": float(rho_hi),
        "normal_velocity_high": float(vn_hi),
        "tangential_velocity_high": float(vt_hi),
        "pressure_smooth": float(p_lo),
        "density_smooth": float(rho_lo),
        "normal_velocity_smooth": float(vn_lo),
    }
    return {"masks": masks, "features": features, "thresholds": thresholds}


def _validate_primitive(value: ArrayLike, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != len(PRIMITIVE_NAMES):
        raise ValueError(f"{name} must have shape (N, 4)")
    return array


def _validate_conservative(value: ArrayLike, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != len(CONSERVATIVE_NAMES):
        raise ValueError(f"{name} must have shape (N, 4)")
    return array


def _validate_normals(value: ArrayLike, prefix: tuple[int, ...]) -> np.ndarray:
    normal = np.asarray(value, dtype=np.float64)
    if normal.shape != prefix + (2,):
        raise ValueError(f"normals must have shape {prefix + (2,)}, got {normal.shape}")
    return normal


def _sound_speed(rho: np.ndarray, pressure: np.ndarray, *, gamma: float) -> np.ndarray:
    value = np.full(np.broadcast_shapes(rho.shape, pressure.shape), np.nan, dtype=np.float64)
    rho_b = np.broadcast_to(rho, value.shape)
    pressure_b = np.broadcast_to(pressure, value.shape)
    valid = (rho_b > 0.0) & (pressure_b >= 0.0)
    value[valid] = gamma * pressure_b[valid] / rho_b[valid]
    value[valid] = np.sqrt(value[valid])
    return value


def _positivity_summary(primitive: np.ndarray) -> dict[str, float | int | bool]:
    rho = primitive[:, 0]
    pressure = primitive[:, 3]
    bad_rho = rho <= 0.0
    bad_pressure = pressure <= 0.0
    return {
        "count": int(primitive.shape[0]),
        "min_density": float(np.min(rho)) if rho.size else float("nan"),
        "min_pressure": float(np.min(pressure)) if pressure.size else float("nan"),
        "nonpositive_density_count": int(np.count_nonzero(bad_rho)),
        "nonpositive_pressure_count": int(np.count_nonzero(bad_pressure)),
        "nonpositive_density_fraction": _safe_mean(bad_rho),
        "nonpositive_pressure_fraction": _safe_mean(bad_pressure),
        "all_positive": bool(not np.any(bad_rho) and not np.any(bad_pressure)),
    }


def _edge_mask(mask: ArrayLike | None, edge_count: int) -> np.ndarray:
    if mask is None:
        return np.ones(edge_count, dtype=bool)
    out = np.asarray(mask, dtype=bool)
    if out.ndim != 1 or out.shape[0] != edge_count:
        raise ValueError("edge_mask must match the edge axis")
    return out


def _safe_mean(values: ArrayLike) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    finite = arr[np.isfinite(arr)]
    return float(np.mean(finite)) if finite.size else float("nan")


def _by_primitive(values: ArrayLike) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {name: float(arr[i]) for i, name in enumerate(PRIMITIVE_NAMES)}


def _by_conservative(values: ArrayLike) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {name: float(arr[i]) for i, name in enumerate(CONSERVATIVE_NAMES)}


def _by_primitive_mean(values: ArrayLike) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return _by_primitive(np.full(len(PRIMITIVE_NAMES), np.nan))
    return _by_primitive(np.mean(arr, axis=0))


def _by_primitive_max(values: ArrayLike) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return _by_primitive(np.full(len(PRIMITIVE_NAMES), np.nan))
    return _by_primitive(np.max(arr, axis=0))


def _empty_update_summary() -> dict[str, Any]:
    nan = np.full(len(CONSERVATIVE_NAMES), np.nan)
    return {
        "node_count": 0,
        "rmse": _by_conservative(nan),
        "relative_l2": _by_conservative(nan),
        "mean_signed_error": _by_conservative(nan),
        "induced_delta_l2_mean": float("nan"),
        "true_delta_l2_mean": float("nan"),
    }


def _empty_flux_summary() -> dict[str, Any]:
    nan = np.full(len(CONSERVATIVE_NAMES), np.nan)
    return {
        "edge_count": 0,
        "rmse": _by_conservative(nan),
        "relative_l2": _by_conservative(nan),
        "edge_relative_l2": float("nan"),
        "edge_error_norm_mean": float("nan"),
        "edge_reference_norm_mean": float("nan"),
        "edge_cosine_mean": float("nan"),
    }


def _safe_edge_cosine(candidate: np.ndarray, reference: np.ndarray) -> float:
    denom = np.linalg.norm(candidate, axis=-1) * np.linalg.norm(reference, axis=-1)
    valid = denom > EPS
    if not np.any(valid):
        return float("nan")
    cosine = np.sum(candidate[valid] * reference[valid], axis=-1) / denom[valid]
    return float(np.mean(cosine))

def _masked_quantile(values: np.ndarray, mask: np.ndarray, quantile: float) -> float:
    if np.any(mask):
        return float(np.quantile(values[mask], quantile))
    return float("inf")


