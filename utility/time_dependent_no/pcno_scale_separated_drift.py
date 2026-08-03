"""Orthogonal spatial-scale diagnostics for saved PCNO increment defects.

This module is deliberately model-independent.  It operates on conservative
increment fields already compared on one structured finite-volume mesh.  A
cell-centred, orthonormal DCT-II supplies a nonperiodic reflective basis, so the
registered physical-wavelength bands reconstruct the field and obey Parseval.
Metrics use one frozen four-component residual scale and uniform cell volumes;
nodes are never pooled across cases or meshes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy.fft import dctn, idctn

from utility.time_dependent_no.pcno_residual_structure import (
    conservative_to_pressure,
    shock_vortex_regions,
)

SPATIAL_BANDS = (
    ("large", 0.125, np.inf),
    ("transition", 0.05, 0.125),
    ("local", 0.0, 0.05),
)
CONSERVATIVE_COMPONENTS = ("rho", "rho_u", "rho_v", "energy")
PHYSICAL_PARTITIONS = (
    "partition_boundary",
    "partition_shock",
    "partition_vortex",
    "partition_smooth",
)


def _sequence(value: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 3 or array.shape[-1] != 4:
        raise ValueError(f"{name} must have shape [time, nodes, 4]")
    if array.shape[0] < 1 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be nonempty and finite")
    return array


def _component_scale(value: Sequence[float] | np.ndarray) -> np.ndarray:
    scale = np.asarray(value, dtype=np.float64).reshape(-1)
    if scale.shape != (4,) or np.any(scale <= 0.0) or not np.isfinite(scale).all():
        raise ValueError("component_scale must contain four positive finite values")
    return scale


def dct_band_masks(
    resolution: tuple[int, int],
    *,
    domain_lengths: tuple[float, float] = (2.0, 1.0),
) -> dict[str, np.ndarray]:
    """Return a complete partition of cell-centred DCT modes by wavelength."""

    nx, ny = (int(value) for value in resolution)
    lx, ly = (float(value) for value in domain_lengths)
    if nx < 2 or ny < 2 or lx <= 0.0 or ly <= 0.0:
        raise ValueError("resolution and domain lengths must be positive")
    frequency_x = np.arange(nx, dtype=np.float64) / (2.0 * lx)
    frequency_y = np.arange(ny, dtype=np.float64) / (2.0 * ly)
    radial_frequency = np.hypot(frequency_y[:, None], frequency_x[None, :])
    large_cutoff = 1.0 / 0.125
    local_cutoff = 1.0 / 0.05
    masks = {
        "large": radial_frequency <= large_cutoff,
        "transition": (radial_frequency > large_cutoff)
        & (radial_frequency <= local_cutoff),
        "local": radial_frequency > local_cutoff,
    }
    coverage = sum(mask.astype(np.int8) for mask in masks.values())
    if not np.all(coverage == 1):
        raise AssertionError("DCT wavelength bands do not partition the modes")
    return masks


def project_dct_bands(
    defects: np.ndarray,
    *,
    resolution: tuple[int, int],
    component_scale: Sequence[float] | np.ndarray,
    domain_lengths: tuple[float, float] = (2.0, 1.0),
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Project a physical defect sequence into orthogonal wavelength bands."""

    array = _sequence(defects, name="defects")
    scale = _component_scale(component_scale)
    nx, ny = (int(value) for value in resolution)
    if array.shape[1] != nx * ny:
        raise ValueError("defects do not match the declared structured grid")
    scaled = array.reshape(array.shape[0], ny, nx, 4) / scale[None, None, None, :]
    coefficients = dctn(scaled, type=2, axes=(1, 2), norm="ortho")
    masks = dct_band_masks(resolution, domain_lengths=domain_lengths)
    projected: dict[str, np.ndarray] = {}
    reconstructed = np.zeros_like(scaled)
    coefficient_energy = np.square(coefficients).sum(axis=(1, 2, 3))
    band_energy_sum = np.zeros(array.shape[0], dtype=np.float64)
    for name, mask in masks.items():
        band_coefficients = coefficients * mask[None, :, :, None]
        band_scaled = idctn(
            band_coefficients,
            type=2,
            axes=(1, 2),
            norm="ortho",
        )
        reconstructed += band_scaled
        projected[name] = (band_scaled * scale[None, None, None, :]).reshape(
            array.shape
        )
        band_energy_sum += np.square(band_coefficients).sum(axis=(1, 2, 3))
    energy_denominator = np.maximum(coefficient_energy, np.finfo(np.float64).tiny)
    closure = {
        "maximum_reconstruction_abs_residual_scaled": float(
            np.max(np.abs(reconstructed - scaled))
        ),
        "maximum_instantaneous_energy_relative_closure": float(
            np.max(np.abs(band_energy_sum - coefficient_energy) / energy_denominator)
        ),
    }
    return projected, closure


def _energy(sequence: np.ndarray) -> np.ndarray:
    return np.mean(np.sum(np.square(sequence), axis=2), axis=1)


def _inner(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.mean(np.sum(left * right, axis=2), axis=1)


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= 0.0 or not np.isfinite(denominator):
        return None
    return float(numerator / denominator)


def _pod_summary(sequence: np.ndarray) -> dict[str, float | int | None]:
    matrix = sequence.reshape(sequence.shape[0], -1)
    gram = matrix @ matrix.T / sequence.shape[1]
    eigenvalues = np.linalg.eigvalsh(gram)[::-1]
    eigenvalues = np.maximum(eigenvalues, 0.0)
    total = float(eigenvalues.sum())
    if total <= 0.0:
        return {
            "first_mode_energy_fraction": None,
            "first_three_energy_fraction": None,
            "modes_for_95_percent": 0,
        }
    cumulative = np.cumsum(eigenvalues) / total
    return {
        "first_mode_energy_fraction": float(eigenvalues[0] / total),
        "first_three_energy_fraction": float(eigenvalues[:3].sum() / total),
        "modes_for_95_percent": int(np.searchsorted(cumulative, 0.95) + 1),
    }


def _lag_correlations(
    sequence: np.ndarray, max_lag: int, *, centered: bool
) -> list[dict[str, Any]]:
    values = sequence - sequence.mean(axis=0, keepdims=True) if centered else sequence
    maximum_lag = min(max_lag, values.shape[0] - 1)
    if centered:
        centered_energy = float(np.sum(np.square(values)))
        uncentered_energy = float(np.sum(np.square(sequence)))
        if centered_energy <= np.finfo(np.float64).eps * max(
            uncentered_energy, np.finfo(np.float64).tiny
        ):
            return [
                {"lag": lag, "correlation": None} for lag in range(1, maximum_lag + 1)
            ]
    rows: list[dict[str, Any]] = []
    for lag in range(1, maximum_lag + 1):
        left = values[:-lag]
        right = values[lag:]
        numerator = float(np.sum(left * right))
        denominator = float(np.sqrt(np.sum(np.square(left)) * np.sum(np.square(right))))
        rows.append(
            {
                "lag": lag,
                "correlation": _safe_ratio(numerator, denominator),
            }
        )
    return rows


def _scaled_grid(
    sequence: np.ndarray,
    *,
    resolution: tuple[int, int],
    component_scale: np.ndarray,
) -> np.ndarray:
    nx, ny = resolution
    return (
        sequence.reshape(sequence.shape[0], ny * nx, 4) / component_scale[None, None, :]
    )


def scale_separated_diagnostics(
    defects: np.ndarray,
    truth_increments: np.ndarray,
    *,
    resolution: tuple[int, int],
    component_scale: Sequence[float] | np.ndarray,
    domain_lengths: tuple[float, float] = (2.0, 1.0),
    max_lag: int = 10,
    return_projections: bool = False,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, float],
    dict[str, np.ndarray] | None,
]:
    """Measure how spatial bands accumulate or cancel through time.

    The coherent-mean fraction is
    ``||sum_t delta_t||^2 / (T * sum_t ||delta_t||^2)``.  It equals one for a
    time-independent drift and zero for an exactly cancelling sequence.
    """

    defect_array = _sequence(defects, name="defects")
    truth_array = _sequence(truth_increments, name="truth_increments")
    if defect_array.shape != truth_array.shape:
        raise ValueError("defects and truth_increments must have identical shapes")
    if max_lag < 1:
        raise ValueError("max_lag must be positive")
    scale = _component_scale(component_scale)
    projections, closure = project_dct_bands(
        defect_array,
        resolution=resolution,
        component_scale=scale,
        domain_lengths=domain_lengths,
    )
    scaled_total = _scaled_grid(
        defect_array,
        resolution=resolution,
        component_scale=scale,
    )
    scaled_truth = _scaled_grid(
        truth_array,
        resolution=resolution,
        component_scale=scale,
    )
    scaled_sequences = {
        "total": scaled_total,
        **{
            name: _scaled_grid(
                value,
                resolution=resolution,
                component_scale=scale,
            )
            for name, value in projections.items()
        },
    }

    total_energy = _energy(scaled_total)
    truth_energy = _energy(scaled_truth)
    cumulative_total = np.cumsum(scaled_total, axis=0)
    cumulative_truth = np.cumsum(scaled_truth, axis=0)
    cumulative_total_energy = _energy(cumulative_total)
    cumulative_truth_energy = _energy(cumulative_truth)
    total_path_energy = float(total_energy.sum())
    steps = np.arange(1, defect_array.shape[0] + 1, dtype=np.float64)
    time_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    band_cumulative_energy_sum = np.zeros(defect_array.shape[0], dtype=np.float64)

    for band, sequence in scaled_sequences.items():
        energy = _energy(sequence)
        rms = np.sqrt(energy)
        cumulative = np.cumsum(sequence, axis=0)
        cumulative_energy = _energy(cumulative)
        cumulative_rms = np.sqrt(cumulative_energy)
        path_sum_rms = np.cumsum(rms)
        path_energy = np.cumsum(energy)
        cumulative_before = np.concatenate(
            (np.zeros_like(cumulative[:1]), cumulative[:-1]), axis=0
        )
        signed_interaction = 2.0 * _inner(cumulative_before, sequence)
        signed_growth = signed_interaction + energy
        direct_growth = cumulative_energy - np.concatenate(
            (np.zeros(1, dtype=np.float64), cumulative_energy[:-1])
        )
        if band != "total":
            band_cumulative_energy_sum += cumulative_energy
        for index in range(defect_array.shape[0]):
            time_rows.append(
                {
                    "step": index + 1,
                    "band": band,
                    "instantaneous_rms": float(rms[index]),
                    "instantaneous_energy_share": _safe_ratio(
                        float(energy[index]), float(total_energy[index])
                    ),
                    "instantaneous_relative_to_true": _safe_ratio(
                        float(rms[index]), float(np.sqrt(truth_energy[index]))
                    ),
                    "cumulative_rms": float(cumulative_rms[index]),
                    "cumulative_energy_share": _safe_ratio(
                        float(cumulative_energy[index]),
                        float(cumulative_total_energy[index]),
                    ),
                    "cumulative_relative_to_truth_change": _safe_ratio(
                        float(cumulative_rms[index]),
                        float(np.sqrt(cumulative_truth_energy[index])),
                    ),
                    "temporal_coherence": _safe_ratio(
                        float(cumulative_rms[index]), float(path_sum_rms[index])
                    ),
                    "coherent_mean_fraction": _safe_ratio(
                        float(cumulative_energy[index]),
                        float(steps[index] * path_energy[index]),
                    ),
                    "signed_interaction": float(signed_interaction[index]),
                    "defect_energy": float(energy[index]),
                    "signed_growth": float(signed_growth[index]),
                    "direct_growth": float(direct_growth[index]),
                    "signed_interaction_over_defect_energy": _safe_ratio(
                        float(signed_interaction[index]), float(energy[index])
                    ),
                    "signed_interaction_over_total_path_energy": _safe_ratio(
                        float(signed_interaction[index]), total_path_energy
                    ),
                    "defect_energy_over_total_path_energy": _safe_ratio(
                        float(energy[index]), total_path_energy
                    ),
                    "signed_growth_over_total_path_energy": _safe_ratio(
                        float(signed_growth[index]), total_path_energy
                    ),
                }
            )

        centered = sequence - sequence.mean(axis=0, keepdims=True)
        pod = _pod_summary(sequence)
        centered_pod = _pod_summary(centered)
        path_energy_total = float(energy.sum())
        final_energy = float(cumulative_energy[-1])
        total_path_energy = float(total_energy.sum())
        total_final_energy = float(cumulative_total_energy[-1])
        path_energy_share = _safe_ratio(path_energy_total, total_path_energy)
        endpoint_energy_share = _safe_ratio(final_energy, total_final_energy)
        truth_path_energy = float(truth_energy.sum())
        aggregate_rows.append(
            {
                "band": band,
                "steps": defect_array.shape[0],
                "path_energy": path_energy_total,
                "path_energy_share": path_energy_share,
                "endpoint_energy": final_energy,
                "endpoint_energy_share": endpoint_energy_share,
                "accumulation_enrichment": (
                    None
                    if endpoint_energy_share is None or path_energy_share is None
                    else _safe_ratio(endpoint_energy_share, path_energy_share)
                ),
                "residual_error_rms_over_time": float(
                    np.sqrt(path_energy_total / defect_array.shape[0])
                ),
                "aggregate_relative_residual_energy": (
                    None
                    if truth_path_energy <= 0.0
                    else float(np.sqrt(path_energy_total / truth_path_energy))
                ),
                "endpoint_relative_to_truth_change": _safe_ratio(
                    float(np.sqrt(final_energy)),
                    float(np.sqrt(cumulative_truth_energy[-1])),
                ),
                "temporal_coherence": _safe_ratio(
                    float(np.sqrt(final_energy)), float(rms.sum())
                ),
                "coherent_mean_fraction": _safe_ratio(
                    final_energy,
                    float(defect_array.shape[0] * path_energy_total),
                ),
                "temporal_fluctuation_energy_fraction": (
                    None
                    if path_energy_total <= 0.0
                    else float(
                        np.clip(
                            1.0
                            - final_energy
                            / (defect_array.shape[0] * path_energy_total),
                            0.0,
                            1.0,
                        )
                    )
                ),
                "negative_interaction_fraction": float(
                    np.mean(signed_interaction < 0.0)
                ),
                "negative_growth_fraction": float(np.mean(signed_growth < 0.0)),
                "aggregate_signed_interaction_over_defect_energy": _safe_ratio(
                    float(signed_interaction.sum()), path_energy_total
                ),
                "uncentered_first_mode_energy_fraction": pod[
                    "first_mode_energy_fraction"
                ],
                "uncentered_first_three_energy_fraction": pod[
                    "first_three_energy_fraction"
                ],
                "uncentered_modes_for_95_percent": pod["modes_for_95_percent"],
                "centered_first_mode_energy_fraction": centered_pod[
                    "first_mode_energy_fraction"
                ],
                "centered_modes_for_95_percent": centered_pod["modes_for_95_percent"],
                "lag_correlations": _lag_correlations(sequence, max_lag, centered=True),
                "uncentered_lag_correlations": _lag_correlations(
                    sequence, max_lag, centered=False
                ),
            }
        )
        closure[f"{band}_maximum_signed_growth_absolute_closure"] = float(
            np.max(np.abs(signed_growth - direct_growth))
        )

    denominator = np.maximum(cumulative_total_energy, np.finfo(np.float64).tiny)
    closure["maximum_cumulative_energy_relative_closure"] = float(
        np.max(
            np.abs(band_cumulative_energy_sum - cumulative_total_energy) / denominator
        )
    )
    output_projections = projections if return_projections else None
    return time_rows, aggregate_rows, closure, output_projections


def scale_component_diagnostics(
    defects: np.ndarray,
    truth_increments: np.ndarray,
    projections: Mapping[str, np.ndarray],
    *,
    component_scale: Sequence[float] | np.ndarray,
    max_lag: int = 10,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, float]]:
    """Resolve each orthogonal wavelength band by conservative component.

    For scaled defect ``d[n, b, c, i]``, component-band energy is the uniform
    finite-volume mean ``mean_i d**2``.  Path and endpoint shares use the full
    band-by-component partition as denominator.  Signed growth uses the exact
    prefix sum of the same projected sequence, so it is a true recurrence
    contribution for free-rollout defects and a formal accumulated contribution
    for exact-input defects.
    """

    defect_array = _sequence(defects, name="defects")
    truth_array = _sequence(truth_increments, name="truth_increments")
    if defect_array.shape != truth_array.shape:
        raise ValueError("defects and truth_increments must have identical shapes")
    if max_lag < 1:
        raise ValueError("max_lag must be positive")
    expected_bands = {name for name, _, _ in SPATIAL_BANDS}
    if set(projections) != expected_bands:
        raise ValueError("projections must contain the complete registered band set")
    projection_arrays = {
        name: _sequence(value, name=f"projections[{name}]")
        for name, value in projections.items()
    }
    if any(value.shape != defect_array.shape for value in projection_arrays.values()):
        raise ValueError("projected sequences must match the defect shape")

    scale = _component_scale(component_scale)
    scaled_total = defect_array / scale[None, None, :]
    scaled_truth = truth_array / scale[None, None, :]
    scaled_sequences = {
        "total": scaled_total,
        **{
            name: value / scale[None, None, :]
            for name, value in projection_arrays.items()
        },
    }
    reconstructed = sum(scaled_sequences[name] for name, _, _ in SPATIAL_BANDS)
    component_total_energy = np.mean(np.square(scaled_total), axis=1)
    cumulative_total = np.cumsum(scaled_total, axis=0)
    component_cumulative_total_energy = np.mean(np.square(cumulative_total), axis=1)
    total_energy = component_total_energy.sum(axis=1)
    cumulative_total_energy = component_cumulative_total_energy.sum(axis=1)
    total_path_energy = float(total_energy.sum())
    total_endpoint_energy = float(cumulative_total_energy[-1])
    component_path_energy = component_total_energy.sum(axis=0)
    component_endpoint_energy = component_cumulative_total_energy[-1]
    steps = np.arange(1, defect_array.shape[0] + 1, dtype=np.float64)

    time_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    band_component_energy_sum = np.zeros(defect_array.shape[0], dtype=np.float64)
    band_component_cumulative_energy_sum = np.zeros(
        defect_array.shape[0], dtype=np.float64
    )
    maximum_growth_closure = 0.0

    for band, sequence in scaled_sequences.items():
        energy_by_component = np.mean(np.square(sequence), axis=1)
        cumulative = np.cumsum(sequence, axis=0)
        cumulative_energy_by_component = np.mean(np.square(cumulative), axis=1)
        if band != "total":
            band_component_energy_sum += energy_by_component.sum(axis=1)
            band_component_cumulative_energy_sum += cumulative_energy_by_component.sum(
                axis=1
            )

        for component_index, component in enumerate(CONSERVATIVE_COMPONENTS):
            component_sequence = sequence[:, :, component_index : component_index + 1]
            component_cumulative = cumulative[
                :, :, component_index : component_index + 1
            ]
            energy = energy_by_component[:, component_index]
            rms = np.sqrt(energy)
            cumulative_energy = cumulative_energy_by_component[:, component_index]
            cumulative_rms = np.sqrt(cumulative_energy)
            path_sum_rms = np.cumsum(rms)
            path_energy_prefix = np.cumsum(energy)
            cumulative_before = np.concatenate(
                (
                    np.zeros_like(component_cumulative[:1]),
                    component_cumulative[:-1],
                ),
                axis=0,
            )
            signed_interaction = 2.0 * _inner(cumulative_before, component_sequence)
            signed_growth = signed_interaction + energy
            direct_growth = cumulative_energy - np.concatenate(
                (np.zeros(1, dtype=np.float64), cumulative_energy[:-1])
            )
            maximum_growth_closure = max(
                maximum_growth_closure,
                float(np.max(np.abs(signed_growth - direct_growth))),
            )
            truth_component = scaled_truth[:, :, component_index : component_index + 1]
            truth_energy = _energy(truth_component)
            cumulative_truth_energy = _energy(np.cumsum(truth_component, axis=0))

            for index in range(defect_array.shape[0]):
                time_rows.append(
                    {
                        "step": index + 1,
                        "band": band,
                        "component": component,
                        "instantaneous_rms": float(rms[index]),
                        "instantaneous_energy_share": _safe_ratio(
                            float(energy[index]), float(total_energy[index])
                        ),
                        "instantaneous_band_share_within_component": _safe_ratio(
                            float(energy[index]),
                            float(component_total_energy[index, component_index]),
                        ),
                        "instantaneous_relative_to_true_component": _safe_ratio(
                            float(rms[index]), float(np.sqrt(truth_energy[index]))
                        ),
                        "cumulative_rms": float(cumulative_rms[index]),
                        "cumulative_energy_share": _safe_ratio(
                            float(cumulative_energy[index]),
                            float(cumulative_total_energy[index]),
                        ),
                        "cumulative_band_share_within_component": _safe_ratio(
                            float(cumulative_energy[index]),
                            float(
                                component_cumulative_total_energy[
                                    index, component_index
                                ]
                            ),
                        ),
                        "cumulative_relative_to_truth_component_change": _safe_ratio(
                            float(cumulative_rms[index]),
                            float(np.sqrt(cumulative_truth_energy[index])),
                        ),
                        "temporal_coherence": _safe_ratio(
                            float(cumulative_rms[index]),
                            float(path_sum_rms[index]),
                        ),
                        "coherent_mean_fraction": _safe_ratio(
                            float(cumulative_energy[index]),
                            float(steps[index] * path_energy_prefix[index]),
                        ),
                        "signed_interaction": float(signed_interaction[index]),
                        "defect_energy": float(energy[index]),
                        "signed_growth": float(signed_growth[index]),
                        "direct_growth": float(direct_growth[index]),
                        "signed_interaction_over_defect_energy": _safe_ratio(
                            float(signed_interaction[index]), float(energy[index])
                        ),
                        "signed_interaction_over_total_path_energy": _safe_ratio(
                            float(signed_interaction[index]), total_path_energy
                        ),
                        "defect_energy_over_total_path_energy": _safe_ratio(
                            float(energy[index]), total_path_energy
                        ),
                        "signed_growth_over_total_path_energy": _safe_ratio(
                            float(signed_growth[index]), total_path_energy
                        ),
                    }
                )

            path_energy = float(energy.sum())
            endpoint_energy = float(cumulative_energy[-1])
            path_energy_share = _safe_ratio(path_energy, total_path_energy)
            endpoint_energy_share = _safe_ratio(endpoint_energy, total_endpoint_energy)
            truth_path_energy = float(truth_energy.sum())
            aggregate_rows.append(
                {
                    "band": band,
                    "component": component,
                    "steps": defect_array.shape[0],
                    "path_energy": path_energy,
                    "path_energy_share": path_energy_share,
                    "path_band_share_within_component": _safe_ratio(
                        path_energy, float(component_path_energy[component_index])
                    ),
                    "endpoint_energy": endpoint_energy,
                    "endpoint_energy_share": endpoint_energy_share,
                    "endpoint_band_share_within_component": _safe_ratio(
                        endpoint_energy,
                        float(component_endpoint_energy[component_index]),
                    ),
                    "accumulation_enrichment": (
                        None
                        if endpoint_energy_share is None or path_energy_share is None
                        else _safe_ratio(endpoint_energy_share, path_energy_share)
                    ),
                    "residual_error_rms_over_time": float(
                        np.sqrt(path_energy / defect_array.shape[0])
                    ),
                    "aggregate_relative_residual_energy": (
                        None
                        if truth_path_energy <= 0.0
                        else float(np.sqrt(path_energy / truth_path_energy))
                    ),
                    "endpoint_relative_to_truth_component_change": _safe_ratio(
                        float(np.sqrt(endpoint_energy)),
                        float(np.sqrt(cumulative_truth_energy[-1])),
                    ),
                    "temporal_coherence": _safe_ratio(
                        float(np.sqrt(endpoint_energy)), float(rms.sum())
                    ),
                    "coherent_mean_fraction": _safe_ratio(
                        endpoint_energy,
                        float(defect_array.shape[0] * path_energy),
                    ),
                    "temporal_fluctuation_energy_fraction": (
                        None
                        if path_energy <= 0.0
                        else float(
                            np.clip(
                                1.0
                                - endpoint_energy
                                / (defect_array.shape[0] * path_energy),
                                0.0,
                                1.0,
                            )
                        )
                    ),
                    "negative_interaction_fraction": float(
                        np.mean(signed_interaction < 0.0)
                    ),
                    "negative_growth_fraction": float(np.mean(signed_growth < 0.0)),
                    "aggregate_signed_interaction_over_defect_energy": _safe_ratio(
                        float(signed_interaction.sum()), path_energy
                    ),
                    "lag_correlations": _lag_correlations(
                        component_sequence, max_lag, centered=True
                    ),
                    "uncentered_lag_correlations": _lag_correlations(
                        component_sequence, max_lag, centered=False
                    ),
                }
            )

    energy_denominator = np.maximum(total_energy, np.finfo(np.float64).tiny)
    cumulative_denominator = np.maximum(
        cumulative_total_energy, np.finfo(np.float64).tiny
    )
    closure = {
        "maximum_component_band_reconstruction_abs_residual_scaled": float(
            np.max(np.abs(reconstructed - scaled_total))
        ),
        "maximum_component_band_instantaneous_energy_relative_closure": float(
            np.max(
                np.abs(band_component_energy_sum - total_energy) / energy_denominator
            )
        ),
        "maximum_component_band_cumulative_energy_relative_closure": float(
            np.max(
                np.abs(band_component_cumulative_energy_sum - cumulative_total_energy)
                / cumulative_denominator
            )
        ),
        "maximum_component_band_signed_growth_absolute_closure": (
            maximum_growth_closure
        ),
    }
    return time_rows, aggregate_rows, closure


def _weighted_inner(left: np.ndarray, right: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(weights[:, None] * left * right))


def pathway_scale_region_diagnostics(
    total_defects: np.ndarray,
    mesh_defects: np.ndarray,
    state_defects: np.ndarray,
    reference_states: np.ndarray,
    nodes: np.ndarray,
    volumes: np.ndarray,
    *,
    resolution: tuple[int, int],
    component_scale: Sequence[float] | np.ndarray,
    gamma: float,
    domain_lengths: tuple[float, float] = (2.0, 1.0),
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, float],
    dict[str, dict[str, np.ndarray]],
]:
    """Resolve free defects by wavelength, physical region, and causal pathway.

    Mesh and state contributions use symmetric attribution of their cross term.
    Region masks at the input frame localize instantaneous energy and signed
    growth.  Masks at the output frame localize cumulative energy.  Because the
    shock and vortex masks move, regional signed growth is a density
    contribution, not a finite difference of moving-region cumulative energy.
    """

    total = _sequence(total_defects, name="total_defects")
    mesh = _sequence(mesh_defects, name="mesh_defects")
    state = _sequence(state_defects, name="state_defects")
    if mesh.shape != total.shape or state.shape != total.shape:
        raise ValueError("total, mesh, and state defects must have identical shapes")
    reference = np.asarray(reference_states, dtype=np.float64)
    if reference.shape != (total.shape[0] + 1, total.shape[1], 4):
        raise ValueError("reference_states must have shape [steps + 1, nodes, 4]")
    positions = np.asarray(nodes, dtype=np.float64)
    nx, ny = resolution
    if positions.shape != (nx * ny, 2) or total.shape[1] != nx * ny:
        raise ValueError("states and nodes do not match the declared resolution")
    weights = np.asarray(volumes, dtype=np.float64).reshape(-1)
    if weights.shape != (nx * ny,) or np.any(weights <= 0.0):
        raise ValueError("volumes must contain one positive value per node")
    weights = weights / weights.sum()
    scale = _component_scale(component_scale)

    scaled_closure = (total - mesh - state) / scale[None, None, :]
    mesh_projection, mesh_closure = project_dct_bands(
        mesh,
        resolution=resolution,
        component_scale=scale,
        domain_lengths=domain_lengths,
    )
    state_projection, state_closure = project_dct_bands(
        state,
        resolution=resolution,
        component_scale=scale,
        domain_lengths=domain_lengths,
    )
    projections: dict[str, dict[str, np.ndarray]] = {}
    for band in ("total", *(name for name, _, _ in SPATIAL_BANDS)):
        mesh_value = mesh if band == "total" else mesh_projection[band]
        state_value = state if band == "total" else state_projection[band]
        projections[band] = {
            "mesh": mesh_value,
            "state": state_value,
            "total": mesh_value + state_value,
        }

    scaled = {
        band: {
            pathway: value / scale[None, None, :] for pathway, value in values.items()
        }
        for band, values in projections.items()
    }
    cumulative = {
        band: {pathway: np.cumsum(value, axis=0) for pathway, value in values.items()}
        for band, values in scaled.items()
    }

    time_rows: list[dict[str, Any]] = []
    maximum_energy_closure = 0.0
    maximum_cumulative_closure = 0.0
    maximum_growth_closure = 0.0
    for step in range(total.shape[0]):
        input_masks, _, _ = shock_vortex_regions(
            reference[step], positions, resolution=resolution, gamma=gamma
        )
        output_masks, _, _ = shock_vortex_regions(
            reference[step + 1], positions, resolution=resolution, gamma=gamma
        )
        input_regions = {
            "all": np.ones(total.shape[1], dtype=bool),
            **{name: input_masks[name] for name in PHYSICAL_PARTITIONS},
        }
        output_regions = {
            "all": np.ones(total.shape[1], dtype=bool),
            **{name: output_masks[name] for name in PHYSICAL_PARTITIONS},
        }
        for band, values in scaled.items():
            mesh_now = values["mesh"][step]
            state_now = values["state"][step]
            total_now = values["total"][step]
            mesh_cumulative = cumulative[band]["mesh"][step]
            state_cumulative = cumulative[band]["state"][step]
            total_cumulative = cumulative[band]["total"][step]
            total_before = (
                np.zeros_like(total_now)
                if step == 0
                else cumulative[band]["total"][step - 1]
            )
            for region, input_mask in input_regions.items():
                input_weight = weights * input_mask
                output_weight = weights * output_regions[region]
                mesh_energy = _weighted_inner(mesh_now, mesh_now, input_weight)
                state_energy = _weighted_inner(state_now, state_now, input_weight)
                cross = _weighted_inner(mesh_now, state_now, input_weight)
                total_energy = _weighted_inner(total_now, total_now, input_weight)
                cumulative_mesh_energy = _weighted_inner(
                    mesh_cumulative, mesh_cumulative, output_weight
                )
                cumulative_state_energy = _weighted_inner(
                    state_cumulative, state_cumulative, output_weight
                )
                cumulative_cross = _weighted_inner(
                    mesh_cumulative, state_cumulative, output_weight
                )
                cumulative_total_energy = _weighted_inner(
                    total_cumulative, total_cumulative, output_weight
                )
                mesh_prior_interaction = 2.0 * _weighted_inner(
                    total_before, mesh_now, input_weight
                )
                state_prior_interaction = 2.0 * _weighted_inner(
                    total_before, state_now, input_weight
                )
                mesh_growth = mesh_prior_interaction + mesh_energy + cross
                state_growth = state_prior_interaction + state_energy + cross
                total_growth = (
                    mesh_prior_interaction + state_prior_interaction + total_energy
                )
                maximum_energy_closure = max(
                    maximum_energy_closure,
                    abs(total_energy - mesh_energy - state_energy - 2.0 * cross),
                )
                maximum_cumulative_closure = max(
                    maximum_cumulative_closure,
                    abs(
                        cumulative_total_energy
                        - cumulative_mesh_energy
                        - cumulative_state_energy
                        - 2.0 * cumulative_cross
                    ),
                )
                maximum_growth_closure = max(
                    maximum_growth_closure,
                    abs(total_growth - mesh_growth - state_growth),
                )
                time_rows.append(
                    {
                        "step": step + 1,
                        "band": band,
                        "region": region,
                        "input_region_volume_fraction": float(input_weight.sum()),
                        "output_region_volume_fraction": float(output_weight.sum()),
                        "total_energy": total_energy,
                        "mesh_energy": mesh_energy,
                        "state_energy": state_energy,
                        "mesh_state_cross": cross,
                        "mesh_symmetric_energy_attribution": mesh_energy + cross,
                        "state_symmetric_energy_attribution": state_energy + cross,
                        "cumulative_total_energy": cumulative_total_energy,
                        "cumulative_mesh_energy": cumulative_mesh_energy,
                        "cumulative_state_energy": cumulative_state_energy,
                        "cumulative_mesh_state_cross": cumulative_cross,
                        "cumulative_mesh_symmetric_attribution": (
                            cumulative_mesh_energy + cumulative_cross
                        ),
                        "cumulative_state_symmetric_attribution": (
                            cumulative_state_energy + cumulative_cross
                        ),
                        "mesh_prior_interaction": mesh_prior_interaction,
                        "state_prior_interaction": state_prior_interaction,
                        "signed_growth_total": total_growth,
                        "signed_growth_mesh_attribution": mesh_growth,
                        "signed_growth_state_attribution": state_growth,
                    }
                )

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in time_rows:
        grouped.setdefault((str(row["band"]), str(row["region"])), []).append(row)
    total_path_energy = float(
        sum(row["total_energy"] for row in grouped["total", "all"])
    )
    total_endpoint_energy = float(
        grouped["total", "all"][-1]["cumulative_total_energy"]
    )
    aggregate_rows: list[dict[str, Any]] = []
    for (band, region), rows in grouped.items():
        endpoint = rows[-1]
        path_total = float(sum(row["total_energy"] for row in rows))
        path_mesh = float(sum(row["mesh_energy"] for row in rows))
        path_state = float(sum(row["state_energy"] for row in rows))
        path_cross = float(sum(row["mesh_state_cross"] for row in rows))
        mesh_growth = np.asarray(
            [row["signed_growth_mesh_attribution"] for row in rows]
        )
        state_growth = np.asarray(
            [row["signed_growth_state_attribution"] for row in rows]
        )
        total_growth = np.asarray([row["signed_growth_total"] for row in rows])
        aggregate_rows.append(
            {
                "band": band,
                "region": region,
                "steps": len(rows),
                "median_input_region_volume_fraction": float(
                    np.median([row["input_region_volume_fraction"] for row in rows])
                ),
                "path_total_energy": path_total,
                "path_total_energy_share_global": _safe_ratio(
                    path_total, total_path_energy
                ),
                "path_mesh_energy": path_mesh,
                "path_state_energy": path_state,
                "path_mesh_state_cross": path_cross,
                "path_mesh_symmetric_attribution": path_mesh + path_cross,
                "path_state_symmetric_attribution": path_state + path_cross,
                "path_mesh_symmetric_attribution_share": _safe_ratio(
                    path_mesh + path_cross, path_total
                ),
                "path_state_symmetric_attribution_share": _safe_ratio(
                    path_state + path_cross, path_total
                ),
                "endpoint_total_energy": float(endpoint["cumulative_total_energy"]),
                "endpoint_total_energy_share_global": _safe_ratio(
                    float(endpoint["cumulative_total_energy"]), total_endpoint_energy
                ),
                "endpoint_mesh_energy": float(endpoint["cumulative_mesh_energy"]),
                "endpoint_state_energy": float(endpoint["cumulative_state_energy"]),
                "endpoint_mesh_state_cross": float(
                    endpoint["cumulative_mesh_state_cross"]
                ),
                "endpoint_mesh_symmetric_attribution": float(
                    endpoint["cumulative_mesh_symmetric_attribution"]
                ),
                "endpoint_state_symmetric_attribution": float(
                    endpoint["cumulative_state_symmetric_attribution"]
                ),
                "endpoint_mesh_symmetric_attribution_share": _safe_ratio(
                    float(endpoint["cumulative_mesh_symmetric_attribution"]),
                    float(endpoint["cumulative_total_energy"]),
                ),
                "endpoint_state_symmetric_attribution_share": _safe_ratio(
                    float(endpoint["cumulative_state_symmetric_attribution"]),
                    float(endpoint["cumulative_total_energy"]),
                ),
                "aggregate_signed_growth_total": float(total_growth.sum()),
                "aggregate_signed_growth_mesh_attribution": float(mesh_growth.sum()),
                "aggregate_signed_growth_state_attribution": float(state_growth.sum()),
                "negative_growth_fraction_total": float(np.mean(total_growth < 0.0)),
                "negative_growth_fraction_mesh": float(np.mean(mesh_growth < 0.0)),
                "negative_growth_fraction_state": float(np.mean(state_growth < 0.0)),
            }
        )

    by_key = {(row["band"], row["region"]): row for row in aggregate_rows}
    for row in aggregate_rows:
        band_all = by_key[row["band"], "all"]
        region_total = by_key["total", row["region"]]
        row["path_region_share_within_band"] = _safe_ratio(
            float(row["path_total_energy"]), float(band_all["path_total_energy"])
        )
        row["endpoint_region_share_within_band"] = _safe_ratio(
            float(row["endpoint_total_energy"]),
            float(band_all["endpoint_total_energy"]),
        )
        row["path_band_share_within_region"] = _safe_ratio(
            float(row["path_total_energy"]), float(region_total["path_total_energy"])
        )
        row["endpoint_band_share_within_region"] = _safe_ratio(
            float(row["endpoint_total_energy"]),
            float(region_total["endpoint_total_energy"]),
        )
        volume_fraction = float(row["median_input_region_volume_fraction"])
        row["path_energy_density_enrichment_within_band"] = (
            _safe_ratio(float(row["path_region_share_within_band"]), volume_fraction)
            if row["path_region_share_within_band"] is not None
            else None
        )

    maximum_partition_closure = 0.0
    maximum_band_energy_relative_closure = 0.0
    maximum_band_cumulative_relative_closure = 0.0
    maximum_band_growth_absolute_closure = 0.0
    for step in range(1, total.shape[0] + 1):
        step_rows = [row for row in time_rows if row["step"] == step]
        indexed = {(row["band"], row["region"]): row for row in step_rows}
        for band in scaled:
            for field in (
                "total_energy",
                "mesh_energy",
                "state_energy",
                "mesh_state_cross",
                "cumulative_total_energy",
                "cumulative_mesh_energy",
                "cumulative_state_energy",
                "cumulative_mesh_state_cross",
                "signed_growth_total",
                "signed_growth_mesh_attribution",
                "signed_growth_state_attribution",
            ):
                partition_sum = sum(
                    float(indexed[band, region][field])
                    for region in PHYSICAL_PARTITIONS
                )
                maximum_partition_closure = max(
                    maximum_partition_closure,
                    abs(partition_sum - float(indexed[band, "all"][field])),
                )
        spatial_bands = [name for name, _, _ in SPATIAL_BANDS]
        total_row = indexed["total", "all"]
        band_energy = sum(
            float(indexed[name, "all"]["total_energy"]) for name in spatial_bands
        )
        band_cumulative = sum(
            float(indexed[name, "all"]["cumulative_total_energy"])
            for name in spatial_bands
        )
        band_growth = sum(
            float(indexed[name, "all"]["signed_growth_total"]) for name in spatial_bands
        )
        maximum_band_energy_relative_closure = max(
            maximum_band_energy_relative_closure,
            abs(band_energy - float(total_row["total_energy"]))
            / max(float(total_row["total_energy"]), np.finfo(np.float64).tiny),
        )
        maximum_band_cumulative_relative_closure = max(
            maximum_band_cumulative_relative_closure,
            abs(band_cumulative - float(total_row["cumulative_total_energy"]))
            / max(
                float(total_row["cumulative_total_energy"]),
                np.finfo(np.float64).tiny,
            ),
        )
        maximum_band_growth_absolute_closure = max(
            maximum_band_growth_absolute_closure,
            abs(band_growth - float(total_row["signed_growth_total"])),
        )

    closure = {
        "maximum_total_mesh_state_reconstruction_abs_residual_scaled": float(
            np.max(np.abs(scaled_closure))
        ),
        "maximum_mesh_band_reconstruction_abs_residual_scaled": mesh_closure[
            "maximum_reconstruction_abs_residual_scaled"
        ],
        "maximum_state_band_reconstruction_abs_residual_scaled": state_closure[
            "maximum_reconstruction_abs_residual_scaled"
        ],
        "maximum_pathway_instantaneous_energy_absolute_closure": (
            maximum_energy_closure
        ),
        "maximum_pathway_cumulative_energy_absolute_closure": (
            maximum_cumulative_closure
        ),
        "maximum_pathway_signed_growth_absolute_closure": maximum_growth_closure,
        "maximum_physical_partition_absolute_closure": maximum_partition_closure,
        "maximum_band_instantaneous_energy_relative_closure": (
            maximum_band_energy_relative_closure
        ),
        "maximum_band_cumulative_energy_relative_closure": (
            maximum_band_cumulative_relative_closure
        ),
        "maximum_band_signed_growth_absolute_closure": (
            maximum_band_growth_absolute_closure
        ),
    }
    return time_rows, aggregate_rows, closure, projections


def _shock_profile_features(
    state: np.ndarray,
    reference_state: np.ndarray,
    nodes: np.ndarray,
    *,
    resolution: tuple[int, int],
    gamma: float,
    search_half_width: float = 0.08,
    boundary_width: float = 0.05,
) -> dict[str, np.ndarray]:
    nx, ny = resolution
    positions = np.asarray(nodes, dtype=np.float64)
    x = positions[:, 0].reshape(ny, nx)
    x_values = x[0]
    dx = float(np.median(np.diff(x_values)))
    face_x = 0.5 * (x_values[:-1] + x_values[1:])
    x_min = float(x_values[0] - 0.5 * dx)
    x_max = float(x_values[-1] + 0.5 * dx)
    eligible = (face_x - x_min >= boundary_width) & (x_max - face_x >= boundary_width)
    if not np.any(eligible):
        raise ValueError("shock profile search excludes every x face")
    reference_pressure = conservative_to_pressure(reference_state, gamma=gamma).reshape(
        ny, nx
    )
    pressure = conservative_to_pressure(state, gamma=gamma).reshape(ny, nx)
    reference_gradient = np.diff(reference_pressure, axis=1)
    candidate_gradient = np.diff(pressure, axis=1)
    eligible_indices = np.flatnonzero(eligible)
    reference_faces = np.asarray(
        [
            eligible_indices[np.argmax(np.abs(row[eligible_indices]))]
            for row in reference_gradient
        ],
        dtype=np.int64,
    )
    reference_position = face_x[reference_faces]
    position = np.empty(ny, dtype=np.float64)
    net_strength = np.empty(ny, dtype=np.float64)
    total_variation = np.empty(ny, dtype=np.float64)
    thickness = np.empty(ny, dtype=np.float64)
    for row in range(ny):
        window = np.abs(face_x - reference_position[row]) <= search_half_width
        gradient = candidate_gradient[row, window]
        locations = face_x[window]
        absolute_gradient = np.abs(gradient)
        mass = float(absolute_gradient.sum())
        if mass <= np.finfo(np.float64).tiny:
            position[row] = reference_position[row]
            thickness[row] = 0.0
        else:
            position[row] = float(np.sum(locations * absolute_gradient) / mass)
            thickness[row] = float(
                np.sqrt(
                    np.sum(absolute_gradient * np.square(locations - position[row]))
                    / mass
                )
            )
        net_strength[row] = abs(float(gradient.sum()))
        total_variation[row] = mass
    return {
        "position": position,
        "reference_position": reference_position,
        "net_strength": net_strength,
        "total_variation": total_variation,
        "thickness": thickness,
    }


def _shock_mode_projection(
    error: np.ndarray,
    reference_state: np.ndarray,
    reference_position: np.ndarray,
    nodes: np.ndarray,
    volumes: np.ndarray,
    *,
    resolution: tuple[int, int],
    gamma: float,
    component_scale: np.ndarray,
) -> dict[str, float | None]:
    nx, ny = resolution
    masks, phase_mode, _ = shock_vortex_regions(
        reference_state, nodes, resolution=resolution, gamma=gamma
    )
    shock_mask = masks["shock_envelope_le_0.05"]
    x = np.asarray(nodes, dtype=np.float64)[:, 0].reshape(ny, nx)
    offset = (x - reference_position[:, None]).reshape(-1)
    translation_mode = -phase_mode
    dilation_mode = -offset[:, None] * phase_mode
    reference_grid = np.asarray(reference_state, dtype=np.float64).reshape(ny, nx, 4)
    amplitude_mode = np.zeros_like(reference_grid)
    x_values = x[0]
    for row in range(ny):
        left = (x_values >= reference_position[row] - 0.08) & (
            x_values <= reference_position[row] - 0.04
        )
        right = (x_values >= reference_position[row] + 0.04) & (
            x_values <= reference_position[row] + 0.08
        )
        if not np.any(left) or not np.any(right):
            continue
        midpoint = 0.5 * (
            reference_grid[row, left].mean(axis=0)
            + reference_grid[row, right].mean(axis=0)
        )
        amplitude_mode[row] = reference_grid[row] - midpoint
    amplitude_mode = amplitude_mode.reshape(-1, 4)
    amplitude_mode[~shock_mask] = 0.0

    normalized_volume = np.asarray(volumes, dtype=np.float64).reshape(-1)
    normalized_volume = normalized_volume / normalized_volume.sum()
    root_weight = np.sqrt(normalized_volume[shock_mask])[:, None]
    scaled_error = (
        np.asarray(error, dtype=np.float64)[shock_mask]
        / component_scale[None, :]
        * root_weight
    ).reshape(-1)
    modes = [translation_mode, dilation_mode, amplitude_mode]
    columns = [
        (mode[shock_mask] / component_scale[None, :] * root_weight).reshape(-1)
        for mode in modes
    ]
    design = np.stack(columns, axis=1)
    error_energy = float(np.dot(scaled_error, scaled_error))
    translation_energy = float(np.dot(columns[0], columns[0]))
    translation_coefficient = (
        float(np.dot(scaled_error, columns[0]) / translation_energy)
        if translation_energy > np.finfo(np.float64).tiny
        else None
    )
    translation_fraction = (
        float(
            np.square(np.dot(scaled_error, columns[0]))
            / (error_energy * translation_energy)
        )
        if error_energy > np.finfo(np.float64).tiny
        and translation_energy > np.finfo(np.float64).tiny
        else None
    )
    coefficients, _, _, singular_values = np.linalg.lstsq(
        design, scaled_error, rcond=None
    )
    fitted = design @ coefficients
    fitted_energy = float(np.dot(fitted, fitted))
    joint_fraction = (
        float(np.clip(fitted_energy / error_energy, 0.0, 1.0))
        if error_energy > np.finfo(np.float64).tiny
        else None
    )
    condition = (
        float(singular_values[0] / singular_values[-1])
        if singular_values.size and singular_values[-1] > 0.0
        else None
    )
    return {
        "translation_coefficient": translation_coefficient,
        "translation_only_energy_fraction": translation_fraction,
        "joint_translation_dilation_amplitude_energy_fraction": joint_fraction,
        "joint_unexplained_energy_fraction": (
            None if joint_fraction is None else float(1.0 - joint_fraction)
        ),
        "joint_translation_coefficient": float(coefficients[0]),
        "joint_dilation_coefficient": float(coefficients[1]),
        "joint_amplitude_coefficient": float(coefficients[2]),
        "joint_mode_condition_number": condition,
    }


def shock_profile_diagnostics(
    reference_states: np.ndarray,
    coarse_states: np.ndarray,
    restricted_fine_states: np.ndarray,
    nodes: np.ndarray,
    volumes: np.ndarray,
    *,
    resolution: tuple[int, int],
    gamma: float,
    component_scale: Sequence[float] | np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Track robust shock position, jump, total variation, and profile modes."""

    reference = np.asarray(reference_states, dtype=np.float64)
    coarse = np.asarray(coarse_states, dtype=np.float64)
    fine = np.asarray(restricted_fine_states, dtype=np.float64)
    if reference.shape != coarse.shape or reference.shape != fine.shape:
        raise ValueError("reference, coarse, and restricted-fine states must match")
    if reference.ndim != 3 or reference.shape[-1] != 4:
        raise ValueError("shock trajectories must have shape [time, nodes, 4]")
    scale = _component_scale(component_scale)
    comparisons = (
        ("coarse_vs_reference", coarse, reference),
        ("restricted_fine_vs_reference", fine, reference),
        ("coarse_vs_restricted_fine", coarse, fine),
    )
    time_rows: list[dict[str, Any]] = []
    for frame in range(reference.shape[0]):
        reference_features = _shock_profile_features(
            reference[frame],
            reference[frame],
            nodes,
            resolution=resolution,
            gamma=gamma,
        )
        feature_cache = {
            "reference": reference_features,
            "coarse": _shock_profile_features(
                coarse[frame],
                reference[frame],
                nodes,
                resolution=resolution,
                gamma=gamma,
            ),
            "fine": _shock_profile_features(
                fine[frame],
                reference[frame],
                nodes,
                resolution=resolution,
                gamma=gamma,
            ),
        }
        labels = {
            "coarse_vs_reference": ("coarse", "reference"),
            "restricted_fine_vs_reference": ("fine", "reference"),
            "coarse_vs_restricted_fine": ("coarse", "fine"),
        }
        for comparison, prediction, target in comparisons:
            prediction_label, target_label = labels[comparison]
            predicted_features = feature_cache[prediction_label]
            target_features = feature_cache[target_label]
            position_error = (
                predicted_features["position"] - target_features["position"]
            )
            strength_ratio = predicted_features["net_strength"] / np.maximum(
                target_features["net_strength"], np.finfo(np.float64).tiny
            )
            variation_ratio = predicted_features["total_variation"] / np.maximum(
                target_features["total_variation"], np.finfo(np.float64).tiny
            )
            thickness_ratio = predicted_features["thickness"] / np.maximum(
                target_features["thickness"], np.finfo(np.float64).tiny
            )
            mode_projection = _shock_mode_projection(
                prediction[frame] - target[frame],
                reference[frame],
                reference_features["reference_position"],
                nodes,
                volumes,
                resolution=resolution,
                gamma=gamma,
                component_scale=scale,
            )
            time_rows.append(
                {
                    "frame": frame,
                    "comparison": comparison,
                    "mean_shock_position_bias": float(np.mean(position_error)),
                    "rms_shock_position_error": float(
                        np.sqrt(np.mean(np.square(position_error)))
                    ),
                    "median_absolute_shock_position_error": float(
                        np.median(np.abs(position_error))
                    ),
                    "median_net_strength_ratio": float(np.median(strength_ratio)),
                    "median_absolute_net_strength_log_error": float(
                        np.median(np.abs(np.log(strength_ratio)))
                    ),
                    "median_total_variation_ratio": float(np.median(variation_ratio)),
                    "median_absolute_total_variation_log_error": float(
                        np.median(np.abs(np.log(variation_ratio)))
                    ),
                    "median_thickness_ratio": float(np.median(thickness_ratio)),
                    "median_absolute_thickness_log_error": float(
                        np.median(np.abs(np.log(thickness_ratio)))
                    ),
                    **mode_projection,
                }
            )

    aggregate_rows: list[dict[str, Any]] = []
    for comparison, _, _ in comparisons:
        rows = [
            row
            for row in time_rows
            if row["comparison"] == comparison and row["frame"] > 0
        ]
        endpoint = rows[-1]
        aggregate: dict[str, Any] = {
            "comparison": comparison,
            "frames": len(rows),
        }
        fields = (
            "rms_shock_position_error",
            "median_absolute_shock_position_error",
            "median_absolute_net_strength_log_error",
            "median_absolute_total_variation_log_error",
            "median_absolute_thickness_log_error",
            "translation_only_energy_fraction",
            "joint_translation_dilation_amplitude_energy_fraction",
            "joint_unexplained_energy_fraction",
        )
        for field in fields:
            values = [float(row[field]) for row in rows if row[field] is not None]
            aggregate[f"median_{field}"] = float(np.median(values)) if values else None
            aggregate[f"endpoint_{field}"] = endpoint[field]
        aggregate_rows.append(aggregate)
    return time_rows, aggregate_rows


def band_contract() -> list[Mapping[str, Any]]:
    """Return the JSON-ready registered wavelength contract."""

    return [
        {
            "name": name,
            "wavelength_min": float(wavelength_min),
            "wavelength_max": (
                None if np.isinf(wavelength_max) else float(wavelength_max)
            ),
        }
        for name, wavelength_min, wavelength_max in SPATIAL_BANDS
    ]
