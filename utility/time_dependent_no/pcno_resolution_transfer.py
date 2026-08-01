"""Resolution-translation utilities for the dynamic shock--vortex PCNO.

The functions in this module keep the physical problem fixed while rebuilding
its sampled state, finite-volume quadrature, node types, graph connectivity, and
least-squares differential weights on each requested Cartesian model grid.
They deliberately do not evolve a native-grid CFD solver.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any, Literal

import numpy as np
import torch

from utility.time_dependent_no.pcno_fv_geometry import (
    PCNOFiniteVolumeGeometry,
    build_pcno_finite_volume_geometry,
)
from utility.time_dependent_no.shock_vortex_coarse_cfd import (
    restrict_uniform_cell_averages,
)
from utility.time_dependent_no.shock_vortex_fv import (
    BOUNDARY_TAG_NAMES,
    ShockVortexFVConfig,
    make_structured_fv_geometry,
    shock_vortex_initial_cell_averages,
)

Resolution = tuple[int, int]
MULTIRES_REFERENCE_SCHEMA = "shock_vortex_multires_restriction_reference_v1"
NodeTypeProtocol = Literal[
    "physical",
    "all_normal",
    "swapped_boundary_kinds",
    "training_band",
]
NODE_TYPE_PROTOCOLS: tuple[NodeTypeProtocol, ...] = (
    "physical",
    "all_normal",
    "swapped_boundary_kinds",
    "training_band",
)


def parse_resolution(value: str) -> Resolution:
    """Parse an ``NXxNY`` model-grid declaration."""

    pieces = str(value).lower().split("x")
    if len(pieces) != 2:
        raise ValueError(f"resolution must have form NXxNY, got {value!r}")
    try:
        nx, ny = (int(piece) for piece in pieces)
    except ValueError as exc:
        raise ValueError(f"resolution must have form NXxNY, got {value!r}") from exc
    if nx < 2 or ny < 2:
        raise ValueError("both resolution dimensions must be at least two")
    return nx, ny


def resolution_label(resolution: Resolution) -> str:
    nx, ny = resolution
    return f"{int(nx)}x{int(ny)}"


def config_for_model_grid(
    base: ShockVortexFVConfig,
    resolution: Resolution,
) -> ShockVortexFVConfig:
    """Return the same physical case sampled directly on one model grid."""

    nx, ny = resolution
    return replace(
        base,
        nx=int(nx),
        ny=int(ny),
        coarse_nx=int(nx),
        coarse_ny=int(ny),
    ).validated()


def build_resolution_geometry(
    base: ShockVortexFVConfig,
    resolution: Resolution,
) -> tuple[ShockVortexFVConfig, PCNOFiniteVolumeGeometry]:
    """Regenerate FV quadrature, graph edges, types, and gradient weights."""

    config = config_for_model_grid(base, resolution)
    finite_volume = make_structured_fv_geometry(config)
    geometry = build_pcno_finite_volume_geometry(
        cell_centers=finite_volume.cell_centers,
        cell_volume=finite_volume.cell_volume,
        face_owner=finite_volume.face_owner,
        face_neighbor=finite_volume.face_neighbor,
        face_boundary_tag=finite_volume.face_boundary_tag,
        boundary_tag_names=BOUNDARY_TAG_NAMES,
    )
    return config, geometry


def initial_state_for_model_grid(
    base: ShockVortexFVConfig,
    resolution: Resolution,
    *,
    dtype: torch.dtype = torch.float64,
    quadrature_order: int | None = None,
) -> np.ndarray:
    """Sample the physical initial condition as conservative cell averages."""

    config = config_for_model_grid(base, resolution)
    state = shock_vortex_initial_cell_averages(
        config,
        device="cpu",
        dtype=dtype,
        quadrature_order=quadrature_order,
    )
    return state.detach().cpu().numpy().reshape(config.nx * config.ny, 4)


def initial_states_from_common_source(
    base: ShockVortexFVConfig,
    resolutions: Sequence[Resolution],
    *,
    source_resolution: Resolution,
    dtype: torch.dtype = torch.float64,
    quadrature_order: int | None = None,
) -> tuple[np.ndarray, dict[Resolution, np.ndarray]]:
    """Sample once on a common fine grid, then conservatively restrict.

    This is the finite-volume sampling contract for a restriction-consistent
    physical map.  Every returned component is a cell average of a conserved
    variable; primitive variables are never interpolated or restricted.
    """

    source_nx, source_ny = source_resolution
    requested = list(resolutions)
    if not requested:
        raise ValueError("at least one target resolution is required")
    if len(set(requested)) != len(requested):
        raise ValueError("target resolutions must be unique")
    for target_nx, target_ny in requested:
        if source_nx % target_nx or source_ny % target_ny:
            raise ValueError(
                "every target grid must divide the common source grid exactly"
            )

    source = initial_state_for_model_grid(
        base,
        source_resolution,
        dtype=dtype,
        quadrature_order=quadrature_order,
    )
    states = {}
    for resolution in requested:
        if resolution == source_resolution:
            states[resolution] = np.array(source, copy=True)
        else:
            states[resolution] = restrict_nested_state(
                source,
                fine_resolution=source_resolution,
                coarse_resolution=resolution,
            )
    return source, states


def node_types_for_protocol(
    geometry: PCNOFiniteVolumeGeometry,
    config: ShockVortexFVConfig,
    protocol: NodeTypeProtocol,
    *,
    training_resolution: Resolution,
) -> np.ndarray:
    """Return dynamic-family node types under one declared intervention.

    ``physical`` regenerates the one-cell boundary-touch descriptor on the
    target mesh. ``all_normal`` removes the descriptor.
    ``swapped_boundary_kinds`` preserves the tagged set but exchanges the
    y-symmetry-only and x-extrapolation-only categories (codes 1 and 2);
    corner code 3 is unchanged. ``training_band`` is a binary
    cell-intersection diagnostic that approximates a boundary strip one
    training cell wide. It is exact only when the target mesh resolves that
    width and is not a replacement boundary policy.
    """

    if protocol not in NODE_TYPE_PROTOCOLS:
        raise ValueError(f"unsupported node-type protocol: {protocol}")
    if geometry.nodes.shape != (config.nx * config.ny, 2):
        raise ValueError("geometry and model-grid configuration disagree")
    if protocol == "physical":
        return np.array(geometry.node_type, dtype=np.int64, copy=True)
    if protocol == "all_normal":
        return np.zeros(config.nx * config.ny, dtype=np.int64)
    if protocol == "swapped_boundary_kinds":
        physical = np.asarray(geometry.node_type, dtype=np.int64)
        swapped = np.array(physical, copy=True)
        swapped[physical == 1] = 2
        swapped[physical == 2] = 1
        return swapped

    training_nx, training_ny = training_resolution
    if training_nx < 1 or training_ny < 1:
        raise ValueError("training resolution must be positive")
    fixed_x_width = (config.x_max - config.x_min) / training_nx
    fixed_y_width = (config.y_max - config.y_min) / training_ny
    target_dx = (config.x_max - config.x_min) / config.nx
    target_dy = (config.y_max - config.y_min) / config.ny
    x = geometry.nodes[:, 0]
    y = geometry.nodes[:, 1]
    left_edge = x - 0.5 * target_dx
    right_edge = x + 0.5 * target_dx
    bottom_edge = y - 0.5 * target_dy
    top_edge = y + 0.5 * target_dy
    tolerance = (
        32.0
        * np.finfo(np.float64).eps
        * max(
            abs(config.x_min),
            abs(config.x_max),
            abs(config.y_min),
            abs(config.y_max),
            1.0,
        )
    )
    touches_x_band = (left_edge < config.x_min + fixed_x_width - tolerance) | (
        right_edge > config.x_max - fixed_x_width + tolerance
    )
    touches_y_band = (bottom_edge < config.y_min + fixed_y_width - tolerance) | (
        top_edge > config.y_max - fixed_y_width + tolerance
    )
    return 2 * touches_x_band.astype(np.int64) + touches_y_band.astype(np.int64)


def node_type_scaling_summary(
    geometry: PCNOFiniteVolumeGeometry,
    config: ShockVortexFVConfig,
    node_type: np.ndarray,
) -> dict[str, Any]:
    """Summarize raw counts, physical mass, and tagged physical widths."""

    codes = np.asarray(node_type, dtype=np.int64)
    if codes.shape != (config.nx * config.ny,):
        raise ValueError("node_type must contain one code per model-grid cell")
    if np.any((codes < 0) | (codes > 3)):
        raise ValueError("dynamic-family node types must lie in [0, 3]")
    volumes = np.asarray(geometry.node_measures[:, 0], dtype=np.float64)
    normalized_weights = np.asarray(geometry.node_weights[:, 0], dtype=np.float64)
    tagged = codes != 0
    touches_x = (codes == 2) | (codes == 3)
    touches_y = (codes == 1) | (codes == 3)
    dx = (config.x_max - config.x_min) / config.nx
    dy = (config.y_max - config.y_min) / config.ny
    edge_delta = (
        geometry.nodes[geometry.edges[:, 1]] - geometry.nodes[geometry.edges[:, 0]]
    )
    edge_length = np.linalg.norm(edge_delta, axis=-1)

    def tagged_width(mask: np.ndarray, coordinate: np.ndarray, *, axis: str) -> float:
        if not bool(mask.any()):
            return 0.0
        if axis == "x":
            distance = np.minimum(
                coordinate - config.x_min,
                config.x_max - coordinate,
            )
            return float(np.max(distance[mask]) + 0.5 * dx)
        distance = np.minimum(
            coordinate - config.y_min,
            config.y_max - coordinate,
        )
        return float(np.max(distance[mask]) + 0.5 * dy)

    counts = {str(code): int(np.count_nonzero(codes == code)) for code in range(4)}
    physical_mass = {
        str(code): float(volumes[codes == code].sum()) for code in range(4)
    }
    normalized_mass = {
        str(code): float(normalized_weights[codes == code].sum()) for code in range(4)
    }
    return {
        "nx": int(config.nx),
        "ny": int(config.ny),
        "num_nodes": int(codes.size),
        "dx": float(dx),
        "dy": float(dy),
        "counts": counts,
        "fractions": {
            code: count / float(codes.size) for code, count in counts.items()
        },
        "physical_mass_by_type": physical_mass,
        "normalized_mass_by_type": normalized_mass,
        "tagged_count": int(tagged.sum()),
        "tagged_fraction": float(tagged.mean()),
        "tagged_physical_volume": float(volumes[tagged].sum()),
        "tagged_normalized_volume_mass": float(normalized_weights[tagged].sum()),
        "x_tagged_width_per_side": tagged_width(
            touches_x, geometry.nodes[:, 0], axis="x"
        ),
        "y_tagged_width_per_side": tagged_width(
            touches_y, geometry.nodes[:, 1], axis="y"
        ),
        "minimum_one_hop_edge_length": float(edge_length.min()),
        "maximum_one_hop_edge_length": float(edge_length.max()),
        "differential_gradient_stencil_hops": 1,
        "differential_postgradient_smoothing_hops": 2,
        "differential_input_support_hops_upper_bound": 3,
        "differential_input_support_physical_upper_bound": float(
            3.0 * edge_length.max()
        ),
    }


def make_model_sample(
    geometry: PCNOFiniteVolumeGeometry,
    node_type: np.ndarray,
    *,
    mach: float,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Materialize one unpadded, homogeneous-resolution PCNO geometry batch."""

    codes = np.asarray(node_type, dtype=np.int64)
    if codes.shape != (geometry.nodes.shape[0],):
        raise ValueError("node_type must contain one code per geometry node")

    def batched(value: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
        copied = np.array(value, copy=True)
        return torch.as_tensor(copied, dtype=dtype, device=device).unsqueeze(0)

    num_nodes = geometry.nodes.shape[0]
    return {
        "node_mask": torch.ones((1, num_nodes, 1), dtype=torch.float32, device=device),
        "nodes": batched(geometry.nodes, torch.float32),
        "node_measures": batched(geometry.node_measures, torch.float32),
        "node_weights": batched(geometry.node_weights, torch.float32),
        "node_rhos": batched(geometry.node_rhos, torch.float32),
        "directed_edges": batched(geometry.directed_edges, torch.int64),
        "edge_gradient_weights": batched(geometry.edge_gradient_weights, torch.float32),
        "node_type": batched(codes, torch.int64),
        "mach": torch.tensor([float(mach)], dtype=torch.float32, device=device),
    }


def restrict_nested_state(
    state: np.ndarray,
    *,
    fine_resolution: Resolution,
    coarse_resolution: Resolution,
) -> np.ndarray:
    """Conservatively restrict flattened fine-grid cell averages."""

    fine_nx, fine_ny = fine_resolution
    coarse_nx, coarse_ny = coarse_resolution
    return restrict_uniform_cell_averages(
        np.asarray(state),
        target_nx=fine_nx,
        target_ny=fine_ny,
        coarse_nx=coarse_nx,
        coarse_ny=coarse_ny,
    )


def fixed_boundary_distance_mask(
    nodes: np.ndarray,
    config: ShockVortexFVConfig,
    width: float,
) -> np.ndarray:
    """Select cell centers within a fixed physical distance of the boundary."""

    if not np.isfinite(width) or width <= 0.0:
        raise ValueError("boundary-band width must be positive and finite")
    positions = np.asarray(nodes, dtype=np.float64)
    distance = np.minimum.reduce(
        (
            positions[:, 0] - config.x_min,
            config.x_max - positions[:, 0],
            positions[:, 1] - config.y_min,
            config.y_max - positions[:, 1],
        )
    )
    return distance <= float(width)


def weighted_scaled_rms(
    value: np.ndarray,
    *,
    volumes: np.ndarray,
    component_scale: Sequence[float] | np.ndarray,
    mask: np.ndarray | None = None,
) -> float:
    """Physical-volume RMS after fixed per-component scaling."""

    array = np.asarray(value, dtype=np.float64)
    weights = np.asarray(volumes, dtype=np.float64).reshape(-1)
    scale = np.asarray(component_scale, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] != weights.size:
        raise ValueError("value and volumes must align on the node axis")
    if scale.shape != (array.shape[1],) or np.any(scale <= 0.0):
        raise ValueError("component_scale must be positive and match components")
    selected = np.ones(weights.size, dtype=bool) if mask is None else np.asarray(mask)
    if selected.shape != (weights.size,) or selected.dtype != np.bool_:
        raise ValueError("mask must be a boolean vector on the node axis")
    selected_mass = float(weights[selected].sum())
    if selected_mass <= 0.0:
        raise ValueError("metric mask has zero physical volume")
    squared = np.square(array[selected] / scale[None, :]).sum(axis=-1)
    return float(np.sqrt(np.dot(weights[selected], squared) / selected_mass))


def weighted_scaled_relative_l2(
    prediction: np.ndarray,
    reference: np.ndarray,
    *,
    volumes: np.ndarray,
    component_scale: Sequence[float] | np.ndarray,
    mask: np.ndarray | None = None,
    denominator_epsilon: float = 1.0e-30,
) -> float | None:
    """Physical-volume component-scaled relative L2 on an optional region."""

    prediction_array = np.asarray(prediction, dtype=np.float64)
    reference_array = np.asarray(reference, dtype=np.float64)
    if prediction_array.shape != reference_array.shape:
        raise ValueError("prediction and reference shapes must match")
    weights = np.asarray(volumes, dtype=np.float64).reshape(-1)
    scale = np.asarray(component_scale, dtype=np.float64)
    selected = np.ones(weights.size, dtype=bool) if mask is None else np.asarray(mask)
    if selected.shape != (weights.size,) or selected.dtype != np.bool_:
        raise ValueError("mask must be a boolean vector on the node axis")
    error = (prediction_array[selected] - reference_array[selected]) / scale
    baseline = reference_array[selected] / scale
    numerator = float(np.einsum("n,nc,nc->", weights[selected], error, error))
    denominator = float(np.einsum("n,nc,nc->", weights[selected], baseline, baseline))
    if denominator <= denominator_epsilon:
        return None
    return float(np.sqrt(numerator / denominator))


def physical_wavelength_band_metrics(
    prediction: np.ndarray,
    reference: np.ndarray,
    *,
    resolution: Resolution,
    domain_lengths: tuple[float, float],
    component_scale: Sequence[float] | np.ndarray,
    wavelength_min: float,
    wavelength_max: float,
    denominator_epsilon: float = 1.0e-30,
) -> dict[str, float | int | None]:
    """Measure regular-grid spectral error in one fixed physical wavelength band.

    The band is declared in physical units and therefore does not move with a
    grid's Nyquist frequency. Inputs are row-major cell-centered fields with
    components on the final axis.
    """

    nx, ny = resolution
    lx, ly = (float(value) for value in domain_lengths)
    if (
        nx < 2
        or ny < 2
        or lx <= 0.0
        or ly <= 0.0
        or not np.isfinite(wavelength_min)
        or not np.isfinite(wavelength_max)
        or wavelength_min <= 0.0
        or wavelength_max <= wavelength_min
    ):
        raise ValueError("invalid grid, domain, or physical wavelength band")
    prediction_array = np.asarray(prediction, dtype=np.float64)
    reference_array = np.asarray(reference, dtype=np.float64)
    if (
        prediction_array.ndim != 2
        or prediction_array.shape[0] != nx * ny
        or reference_array.shape != prediction_array.shape
    ):
        raise ValueError("prediction/reference must be flattened row-major grid fields")
    scale = np.asarray(component_scale, dtype=np.float64)
    if scale.shape != (prediction_array.shape[-1],) or np.any(scale <= 0.0):
        raise ValueError("component_scale must be positive and match components")

    scaled_prediction = prediction_array.reshape(ny, nx, -1) / scale.reshape(1, 1, -1)
    scaled_reference = reference_array.reshape(ny, nx, -1) / scale.reshape(1, 1, -1)
    error_spectrum = np.fft.rfftn(
        scaled_prediction - scaled_reference,
        axes=(0, 1),
        norm="ortho",
    )
    reference_spectrum = np.fft.rfftn(
        scaled_reference,
        axes=(0, 1),
        norm="ortho",
    )
    frequency_x = np.fft.rfftfreq(nx, d=lx / nx)
    frequency_y = np.fft.fftfreq(ny, d=ly / ny)
    radial_frequency = np.hypot(
        frequency_y[:, None],
        frequency_x[None, :],
    )
    lower_frequency = 1.0 / wavelength_max
    upper_frequency = 1.0 / wavelength_min
    mask = (radial_frequency >= lower_frequency) & (radial_frequency < upper_frequency)
    mode_count = int(np.count_nonzero(mask))
    if mode_count == 0:
        raise ValueError("the physical wavelength band contains no Fourier modes")
    error_energy = float(np.sum(np.abs(error_spectrum[mask]) ** 2))
    reference_energy = float(np.sum(np.abs(reference_spectrum[mask]) ** 2))
    return {
        "wavelength_min": float(wavelength_min),
        "wavelength_max": float(wavelength_max),
        "mode_count": mode_count,
        "error_spectral_l2": float(np.sqrt(error_energy)),
        "reference_spectral_l2": float(np.sqrt(reference_energy)),
        "relative_spectral_l2": (
            float(np.sqrt(error_energy / reference_energy))
            if reference_energy > denominator_epsilon
            else None
        ),
    }


def commutator_metrics(
    *,
    coarse_current: np.ndarray,
    coarse_prediction: np.ndarray,
    fine_current: np.ndarray,
    fine_prediction: np.ndarray,
    coarse_resolution: Resolution,
    fine_resolution: Resolution,
    coarse_volumes: np.ndarray,
    state_scale: Sequence[float] | np.ndarray,
    residual_scale: Sequence[float] | np.ndarray,
) -> dict[str, float | None]:
    """Measure state and learned-update commutation after exact restriction."""

    restricted_current = restrict_nested_state(
        fine_current,
        fine_resolution=fine_resolution,
        coarse_resolution=coarse_resolution,
    )
    restricted_prediction = restrict_nested_state(
        fine_prediction,
        fine_resolution=fine_resolution,
        coarse_resolution=coarse_resolution,
    )
    coarse_update = np.asarray(coarse_prediction) - np.asarray(coarse_current)
    restricted_fine_update = restrict_nested_state(
        np.asarray(fine_prediction) - np.asarray(fine_current),
        fine_resolution=fine_resolution,
        coarse_resolution=coarse_resolution,
    )
    input_difference = np.asarray(coarse_current) - restricted_current
    prediction_difference = np.asarray(coarse_prediction) - restricted_prediction
    update_difference = coarse_update - restricted_fine_update
    input_rms = weighted_scaled_rms(
        input_difference,
        volumes=coarse_volumes,
        component_scale=state_scale,
    )
    prediction_rms = weighted_scaled_rms(
        prediction_difference,
        volumes=coarse_volumes,
        component_scale=state_scale,
    )
    update_rms = weighted_scaled_rms(
        update_difference,
        volumes=coarse_volumes,
        component_scale=residual_scale,
    )
    restricted_update_rms = weighted_scaled_rms(
        restricted_fine_update,
        volumes=coarse_volumes,
        component_scale=residual_scale,
    )
    return {
        "input_restriction_gap_scaled_rms": input_rms,
        "input_restriction_gap_relative_l2": weighted_scaled_relative_l2(
            coarse_current,
            restricted_current,
            volumes=coarse_volumes,
            component_scale=state_scale,
        ),
        "prediction_commutator_scaled_rms": prediction_rms,
        "prediction_commutator_relative_l2": weighted_scaled_relative_l2(
            coarse_prediction,
            restricted_prediction,
            volumes=coarse_volumes,
            component_scale=state_scale,
        ),
        "prediction_commutator_minus_input_gap_scaled_rms": max(
            prediction_rms - input_rms, 0.0
        ),
        "update_commutator_scaled_rms": update_rms,
        "restricted_fine_update_scaled_rms": restricted_update_rms,
        "update_commutator_relative_to_fine_update": (
            update_rms / restricted_update_rms
            if restricted_update_rms > 1.0e-30
            else None
        ),
    }
