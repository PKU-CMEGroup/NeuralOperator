"""Synthetic shock-representation and PCNO differential-path diagnostics.

This module owns the reusable, CPU-sized surface for W26-L2-P0/P1.  It builds
exact cell averages on nested Cartesian finite-volume grids, computes
front-preserving diagnostics, and replays every learned differential stage of
an existing :class:`pcno.pcno.PCNO` without changing model parameters.

"Gibbs-like" is used by the project only as an informal analogy.  The metrics
below describe shock-local oscillation; they do not classify classical Gibbs
without a separate spectral-order study.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from scipy.fft import dctn, idctn

from pcno.pcno import compute_gradient, graph_neighbor_average
from utility.time_dependent_no.pcno_fv_geometry import (
    PCNOFiniteVolumeGeometry,
    build_pcno_finite_volume_geometry,
)
from utility.time_dependent_no.pcno_resolution_pathways import (
    build_graph_ball_operator,
    graph_ball_average,
    prepare_graph_ball_operator,
)
from utility.time_dependent_no.shock_vortex_coarse_cfd import (
    restrict_uniform_cell_averages,
)

DOMAIN_LENGTHS = (1.0, 0.5)
REGISTERED_RESOLUTIONS = ((32, 16), (64, 32), (128, 64))
NATIVE_RESOLUTION = (64, 32)
DISPLACEMENT = 1.0 / 8.0
PULSE_WIDTH = 1.0 / 4.0
SMOOTH_WIDTH = 1.0 / 32.0
FRONT_BAND_WIDTH = 1.0 / 32.0
FIXED_PHYSICAL_RADIUS = 1.0 / 32.0
SMOOTH_FILTER_PASS_WAVENUMBER = 8.0
SMOOTH_FILTER_STOP_WAVENUMBER = 16.0
ANCHORS = (1.0 / 4.0, 3.0 / 8.0, 1.0 / 2.0)
TRAIN_PHASES = (0.0, 0.25, 0.5, 0.75)
HELD_PHASES = (0.125, 0.375, 0.625, 0.875)
CASE_FAMILIES = ("step", "pulse", "smooth_tanh", "smooth_sine")


@dataclass(frozen=True)
class FrontSpec:
    """One registered discontinuity with its adjacent continuum states."""

    position: float
    left_state: float
    right_state: float

    @property
    def jump(self) -> float:
        return float(self.right_state - self.left_state)


@dataclass(frozen=True)
class StructuredCellGrid:
    """A nested Cartesian cell grid and its maintained PCNO FV geometry."""

    resolution: tuple[int, int]
    lengths: tuple[float, float]
    x_edges: np.ndarray
    y_edges: np.ndarray
    x_centers: np.ndarray
    y_centers: np.ndarray
    nodes: np.ndarray
    cell_volumes: np.ndarray
    geometry: PCNOFiniteVolumeGeometry

    @property
    def nx(self) -> int:
        return self.resolution[0]

    @property
    def ny(self) -> int:
        return self.resolution[1]

    @property
    def array_shape(self) -> tuple[int, int]:
        """Structured field shape ``[ny,nx]`` in maintained flattened order."""

        return self.ny, self.nx

    @property
    def hx(self) -> float:
        return self.lengths[0] / self.nx

    @property
    def hy(self) -> float:
        return self.lengths[1] / self.ny


@dataclass(frozen=True)
class SyntheticFrontCase:
    """Exact current, next, and increment cell averages for one case."""

    family: str
    position: float
    displacement: float
    width: float | None
    current: np.ndarray
    target: np.ndarray
    increment: np.ndarray
    increment_fronts: tuple[FrontSpec, ...]
    target_fronts: tuple[FrontSpec, ...]
    feature_positions: tuple[float, ...]
    is_discontinuous: bool


@dataclass(frozen=True)
class DifferentialLayerTrace:
    """Frozen tensors at every stage of one PCNO differential branch."""

    layer: int
    raw_gradient: torch.Tensor
    post_aggregation: torch.Tensor
    pre_softsign: torch.Tensor
    post_softsign: torch.Tensor
    branch_output: torch.Tensor
    decoded_differential_ablation_response: torch.Tensor
    saturation: dict[str, float]


@dataclass(frozen=True)
class DifferentialPathTrace:
    """Exact full replay plus intervention-defined decoded marginals."""

    model_output: torch.Tensor
    replay_output: torch.Tensor
    layers: tuple[DifferentialLayerTrace, ...]


def _validated_resolution(value: Sequence[int]) -> tuple[int, int]:
    if len(value) != 2:
        raise ValueError("resolution must contain nx and ny")
    nx, ny = (int(item) for item in value)
    if nx < 2 or ny < 2:
        raise ValueError("resolution axes must each contain at least two cells")
    return nx, ny


def _structured_faces(nx: int, ny: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    owner: list[int] = []
    neighbor: list[int] = []
    boundary_tag: list[int] = []

    def node(i: int, j: int) -> int:
        return j * nx + i

    for i in range(nx - 1):
        for j in range(ny):
            owner.append(node(i, j))
            neighbor.append(node(i + 1, j))
            boundary_tag.append(0)
    for i in range(nx):
        for j in range(ny - 1):
            owner.append(node(i, j))
            neighbor.append(node(i, j + 1))
            boundary_tag.append(0)
    for j in range(ny):
        owner.extend((node(0, j), node(nx - 1, j)))
        neighbor.extend((-1, -1))
        boundary_tag.extend((1, 2))
    for i in range(nx):
        owner.extend((node(i, 0), node(i, ny - 1)))
        neighbor.extend((-1, -1))
        boundary_tag.extend((3, 4))
    return (
        np.asarray(owner, dtype=np.int64),
        np.asarray(neighbor, dtype=np.int64),
        np.asarray(boundary_tag, dtype=np.int64),
    )


def build_structured_cell_grid(
    resolution: Sequence[int],
    *,
    lengths: Sequence[float] = DOMAIN_LENGTHS,
) -> StructuredCellGrid:
    """Build one exact nested cell grid through the maintained FV geometry."""

    nx, ny = _validated_resolution(resolution)
    if len(lengths) != 2:
        raise ValueError("lengths must contain Lx and Ly")
    lx, ly = (float(item) for item in lengths)
    if not np.isfinite((lx, ly)).all() or lx <= 0.0 or ly <= 0.0:
        raise ValueError("domain lengths must be finite and positive")
    x_edges = np.linspace(0.0, lx, nx + 1, dtype=np.float64)
    y_edges = np.linspace(0.0, ly, ny + 1, dtype=np.float64)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    xx, yy = np.meshgrid(x_centers, y_centers, indexing="xy")
    nodes = np.stack((xx, yy), axis=-1).reshape(-1, 2)
    volumes = np.full(nx * ny, (lx / nx) * (ly / ny), dtype=np.float64)
    owner, neighbor, boundary_tag = _structured_faces(nx, ny)
    geometry = build_pcno_finite_volume_geometry(
        cell_centers=nodes,
        cell_volume=volumes,
        face_owner=owner,
        face_neighbor=neighbor,
        face_boundary_tag=boundary_tag,
        boundary_tag_names=("interior", "x_min", "x_max", "y_min", "y_max"),
        gradient_rcond=1.0e-12,
    )
    return StructuredCellGrid(
        resolution=(nx, ny),
        lengths=(lx, ly),
        x_edges=x_edges,
        y_edges=y_edges,
        x_centers=x_centers,
        y_centers=y_centers,
        nodes=nodes,
        cell_volumes=volumes,
        geometry=geometry,
    )


def interval_cell_average(
    cell_edges: np.ndarray, left: float, right: float
) -> np.ndarray:
    """Return exact cell averages of ``1_[left,right)``."""

    edges = np.asarray(cell_edges, dtype=np.float64)
    if edges.ndim != 1 or edges.size < 2 or np.any(np.diff(edges) <= 0.0):
        raise ValueError("cell_edges must be a strictly increasing vector")
    left_value, right_value = float(left), float(right)
    if not np.isfinite((left_value, right_value)).all() or right_value <= left_value:
        raise ValueError("interval endpoints must be finite with left < right")
    overlap = np.maximum(
        0.0,
        np.minimum(edges[1:], right_value) - np.maximum(edges[:-1], left_value),
    )
    return overlap / np.diff(edges)


def left_step_cell_average(cell_edges: np.ndarray, position: float) -> np.ndarray:
    """Return exact cell averages of ``1_{x < position}``."""

    edges = np.asarray(cell_edges, dtype=np.float64)
    return interval_cell_average(edges, float(edges[0]), float(position))


def _log_cosh(value: np.ndarray) -> np.ndarray:
    return np.logaddexp(value, -value) - np.log(2.0)


def smooth_left_step_cell_average(
    cell_edges: np.ndarray, position: float, width: float
) -> np.ndarray:
    """Cell-average ``0.5 * (1 - tanh((x-position)/width))`` exactly."""

    edges = np.asarray(cell_edges, dtype=np.float64)
    width_value = float(width)
    if not np.isfinite(width_value) or width_value <= 0.0:
        raise ValueError("smooth width must be finite and positive")
    argument = (edges - float(position)) / width_value
    antiderivative = 0.5 * edges - 0.5 * width_value * _log_cosh(argument)
    return np.diff(antiderivative) / np.diff(edges)


def sine_cell_average(
    cell_edges: np.ndarray, *, phase: float, wavenumber: float
) -> np.ndarray:
    """Return exact averages of ``sin(2*pi*m*(x-phase))``."""

    edges = np.asarray(cell_edges, dtype=np.float64)
    frequency = 2.0 * np.pi * float(wavenumber)
    if not np.isfinite(frequency) or frequency == 0.0:
        raise ValueError("wavenumber must be finite and nonzero")
    primitive = -np.cos(frequency * (edges - float(phase))) / frequency
    return np.diff(primitive) / np.diff(edges)


def _extrude_x(profile: np.ndarray, ny: int) -> np.ndarray:
    return np.repeat(np.asarray(profile, dtype=np.float64)[None, :], ny, axis=0)


def make_translated_front_case(
    grid: StructuredCellGrid,
    family: str,
    *,
    position: float,
    displacement: float = DISPLACEMENT,
    pulse_width: float = PULSE_WIDTH,
    smooth_width: float = SMOOTH_WIDTH,
    smooth_wavenumber: float = 4.0,
) -> SyntheticFrontCase:
    """Build one exact translated discontinuous or smooth control case."""

    family_value = str(family)
    if family_value not in CASE_FAMILIES:
        raise ValueError(f"family must lie in {CASE_FAMILIES}")
    position_value = float(position)
    displacement_value = float(displacement)
    width_value = float(pulse_width)
    lx = grid.lengths[0]
    if (
        not np.isfinite((position_value, displacement_value)).all()
        or position_value <= 0.0
        or displacement_value <= 0.0
    ):
        raise ValueError("position and displacement must be finite and positive")

    if family_value == "step":
        if position_value + displacement_value >= lx:
            raise ValueError("translated step must remain inside the domain")
        current_x = left_step_cell_average(grid.x_edges, position_value)
        target_x = left_step_cell_average(
            grid.x_edges, position_value + displacement_value
        )
        increment_fronts = (
            FrontSpec(position_value, 0.0, 1.0),
            FrontSpec(position_value + displacement_value, 1.0, 0.0),
        )
        target_fronts = (FrontSpec(position_value + displacement_value, 1.0, 0.0),)
        feature_positions = tuple(front.position for front in increment_fronts)
        registered_width: float | None = None
        discontinuous = True
    elif family_value == "pulse":
        if (
            not np.isfinite(width_value)
            or width_value <= displacement_value
            or position_value + displacement_value + width_value >= lx
        ):
            raise ValueError(
                "pulse requires displacement < width and all fronts inside the domain"
            )
        current_x = interval_cell_average(
            grid.x_edges, position_value, position_value + width_value
        )
        target_x = interval_cell_average(
            grid.x_edges,
            position_value + displacement_value,
            position_value + displacement_value + width_value,
        )
        increment_fronts = (
            FrontSpec(position_value, 0.0, -1.0),
            FrontSpec(position_value + displacement_value, -1.0, 0.0),
            FrontSpec(position_value + width_value, 0.0, 1.0),
            FrontSpec(position_value + displacement_value + width_value, 1.0, 0.0),
        )
        target_fronts = (
            FrontSpec(position_value + displacement_value, 0.0, 1.0),
            FrontSpec(position_value + displacement_value + width_value, 1.0, 0.0),
        )
        feature_positions = tuple(front.position for front in increment_fronts)
        registered_width = width_value
        discontinuous = True
    elif family_value == "smooth_tanh":
        if position_value + displacement_value >= lx:
            raise ValueError("translated tanh control must remain inside the domain")
        current_x = smooth_left_step_cell_average(
            grid.x_edges, position_value, smooth_width
        )
        target_x = smooth_left_step_cell_average(
            grid.x_edges, position_value + displacement_value, smooth_width
        )
        increment_fronts = ()
        target_fronts = ()
        feature_positions = (position_value, position_value + displacement_value)
        registered_width = float(smooth_width)
        discontinuous = False
    else:
        current_x = sine_cell_average(
            grid.x_edges, phase=position_value, wavenumber=smooth_wavenumber
        )
        target_x = sine_cell_average(
            grid.x_edges,
            phase=position_value + displacement_value,
            wavenumber=smooth_wavenumber,
        )
        increment_fronts = ()
        target_fronts = ()
        feature_positions = ()
        registered_width = None
        discontinuous = False

    current = _extrude_x(current_x, grid.ny)
    target = _extrude_x(target_x, grid.ny)
    return SyntheticFrontCase(
        family=family_value,
        position=position_value,
        displacement=displacement_value,
        width=registered_width,
        current=current,
        target=target,
        increment=target - current,
        increment_fronts=increment_fronts,
        target_fronts=target_fronts,
        feature_positions=feature_positions,
        is_discontinuous=discontinuous,
    )


def registered_phase_positions(
    *,
    native_resolution: Sequence[int] = NATIVE_RESOLUTION,
    held_out: bool,
) -> tuple[float, ...]:
    """Return the preregistered physical positions for train or held phases."""

    nx, _ = _validated_resolution(native_resolution)
    native_h = DOMAIN_LENGTHS[0] / nx
    phases = HELD_PHASES if held_out else TRAIN_PHASES
    return tuple(anchor + phase * native_h for anchor in ANCHORS for phase in phases)


def grid_local_phase_positions(
    grid: StructuredCellGrid, *, held_out: bool
) -> tuple[float, ...]:
    """Return phase-matched positions using the spacing of each tested grid.

    These positions diagnose distributional scaling at a common subcell phase.
    They are deliberately separate from :func:`registered_phase_positions`,
    whose fixed physical positions define the resolution-transfer contract.
    """

    phases = HELD_PHASES if held_out else TRAIN_PHASES
    return tuple(anchor + phase * grid.hx for anchor in ANCHORS for phase in phases)


def physical_front_band_mask(
    grid: StructuredCellGrid,
    positions: Sequence[float],
    *,
    band_width: float = FRONT_BAND_WIDTH,
) -> np.ndarray:
    """Return a flattened maintained-order mask for fixed physical front bands."""

    front_positions = np.asarray(tuple(positions), dtype=np.float64)
    band = float(band_width)
    if (
        front_positions.ndim != 1
        or front_positions.size == 0
        or not np.isfinite(front_positions).all()
        or not np.isfinite(band)
        or band <= 0.0
    ):
        raise ValueError("positions and band_width must be finite and nonempty")
    mask_x = np.any(
        np.abs(grid.x_centers[:, None] - front_positions[None, :]) <= band,
        axis=1,
    )
    return np.repeat(mask_x[None, :], grid.ny, axis=0).reshape(-1)


def restrict_nested_cell_averages(
    field: np.ndarray,
    *,
    fine_resolution: Sequence[int],
    coarse_resolution: Sequence[int],
) -> np.ndarray:
    """Conservatively average a uniform nested cell field onto a coarse grid."""

    fine_nx, fine_ny = _validated_resolution(fine_resolution)
    coarse_nx, coarse_ny = _validated_resolution(coarse_resolution)
    if fine_nx % coarse_nx or fine_ny % coarse_ny:
        raise ValueError("fine resolution must be an integer refinement")
    values = np.asarray(field)
    if values.shape[:2] != (fine_ny, fine_nx):
        raise ValueError("field leading axes must match fine_resolution")
    tail = values.shape[2:]
    flattened = values.reshape(fine_nx * fine_ny, -1)
    restricted = restrict_uniform_cell_averages(
        flattened,
        target_nx=fine_nx,
        target_ny=fine_ny,
        coarse_nx=coarse_nx,
        coarse_ny=coarse_ny,
    )
    return restricted.reshape(coarse_ny, coarse_nx, *tail)


def physical_cosine_frequencies(
    grid: StructuredCellGrid,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return DCT-II frequencies in physical cycles per unit length."""

    qx = np.arange(grid.nx, dtype=np.float64) / (2.0 * grid.lengths[0])
    qy = np.arange(grid.ny, dtype=np.float64) / (2.0 * grid.lengths[1])
    qx_grid, qy_grid = np.meshgrid(qx, qy, indexing="xy")
    return qx_grid, qy_grid, np.hypot(qx_grid, qy_grid)


def physical_cosine_filter(
    field: np.ndarray,
    grid: StructuredCellGrid,
    *,
    kind: str,
    pass_wavenumber: float = SMOOTH_FILTER_PASS_WAVENUMBER,
    stop_wavenumber: float = SMOOTH_FILTER_STOP_WAVENUMBER,
) -> np.ndarray:
    """Apply one zero, smooth, or hard physical-wavenumber frozen filter.

    ``kind='zero'`` is an exact bypass and deliberately avoids a transform.
    The transform acts only on the two structured spatial axes.  It is a frozen
    diagnostic, not a differentiable training layer.
    """

    values = np.asarray(field)
    if values.shape[:2] != grid.array_shape or not np.isfinite(values).all():
        raise ValueError("field must be finite with leading grid-resolution axes")
    kind_value = str(kind)
    if kind_value == "zero":
        return values
    if not np.issubdtype(values.dtype, np.floating):
        raise ValueError("nonzero physical filters require floating-point input")
    pass_value, stop_value = float(pass_wavenumber), float(stop_wavenumber)
    if not np.isfinite((pass_value, stop_value)).all() or pass_value < 0.0:
        raise ValueError("filter wavenumbers must be finite and nonnegative")
    _, _, physical_q = physical_cosine_frequencies(grid)
    if kind_value == "smooth":
        if stop_value <= pass_value:
            raise ValueError("smooth filter requires stop_wavenumber > pass_wavenumber")
        transfer = np.ones_like(physical_q)
        transition = (physical_q > pass_value) & (physical_q < stop_value)
        transfer[physical_q >= stop_value] = 0.0
        transfer[transition] = 0.5 * (
            1.0
            + np.cos(
                np.pi
                * (physical_q[transition] - pass_value)
                / (stop_value - pass_value)
            )
        )
    elif kind_value == "hard":
        transfer = (physical_q <= pass_value).astype(np.float64)
    else:
        raise ValueError("kind must be 'zero', 'smooth', or 'hard'")
    coefficients = dctn(values, axes=(0, 1), norm="ortho")
    transfer_shape = transfer.shape + (1,) * (values.ndim - 2)
    filtered = idctn(
        coefficients * transfer.reshape(transfer_shape),
        axes=(0, 1),
        norm="ortho",
    )
    return np.asarray(filtered, dtype=values.dtype)


def physical_cosine_spectrum(
    field: np.ndarray,
    grid: StructuredCellGrid,
    *,
    common_resolution: Sequence[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return physical radial wavenumbers and physical DCT coefficient energy.

    ``common_resolution`` truncates both rectangular mode axes to the modes
    shared with a declared comparison grid.  Coefficient energy is multiplied
    by cell volume so its sum obeys the physical discrete Parseval identity.
    """

    values = np.asarray(field, dtype=np.float64)
    if values.shape[:2] != grid.array_shape or not np.isfinite(values).all():
        raise ValueError("field must be finite with leading grid-resolution axes")
    coefficients = dctn(values, axes=(0, 1), norm="ortho")
    energy = np.square(np.abs(coefficients)) * grid.hx * grid.hy
    if values.ndim > 2:
        energy = energy.sum(axis=tuple(range(2, values.ndim)))
    _, _, physical_q = physical_cosine_frequencies(grid)
    if common_resolution is not None:
        common_nx, common_ny = _validated_resolution(common_resolution)
        if common_nx > grid.nx or common_ny > grid.ny:
            raise ValueError("common_resolution cannot exceed the field grid")
        physical_q = physical_q[:common_ny, :common_nx]
        energy = energy[:common_ny, :common_nx]
    return physical_q.reshape(-1), energy.reshape(-1)


def physical_spectrum_summary(
    field: np.ndarray,
    grid: StructuredCellGrid,
    *,
    pass_wavenumber: float = SMOOTH_FILTER_PASS_WAVENUMBER,
    stop_wavenumber: float = SMOOTH_FILTER_STOP_WAVENUMBER,
    common_resolution: Sequence[int] | None = None,
) -> dict[str, float | int]:
    """Summarize energy in common physical-wavenumber bands."""

    physical_q, energy = physical_cosine_spectrum(
        field, grid, common_resolution=common_resolution
    )
    common_nx, common_ny = (
        grid.resolution
        if common_resolution is None
        else _validated_resolution(common_resolution)
    )
    total = float(np.sum(energy))
    denominator = max(total, np.finfo(np.float64).tiny)
    pass_value, stop_value = float(pass_wavenumber), float(stop_wavenumber)
    return {
        "total_energy": total,
        "low_fraction": float(np.sum(energy[physical_q <= pass_value]) / denominator),
        "transition_fraction": float(
            np.sum(energy[(physical_q > pass_value) & (physical_q < stop_value)])
            / denominator
        ),
        "high_fraction": float(np.sum(energy[physical_q >= stop_value]) / denominator),
        "maximum_physical_wavenumber": float(np.max(physical_q)),
        "common_mode_nx": common_nx,
        "common_mode_ny": common_ny,
        "included_mode_count": common_nx * common_ny,
    }


def _total_variation(field: np.ndarray, grid: StructuredCellGrid) -> float:
    values = np.asarray(field, dtype=np.float64)
    return float(
        grid.hy * np.sum(np.abs(np.diff(values, axis=1)))
        + grid.hx * np.sum(np.abs(np.diff(values, axis=0)))
    )


def _crossing_position(
    x: np.ndarray,
    profile: np.ndarray,
    *,
    front: FrontSpec,
    fraction: float,
    half_window: float,
) -> tuple[float | None, int]:
    jump = front.jump
    if jump == 0.0:
        return None, 0
    normalized = (profile - front.left_state) / jump
    candidates: list[float] = []
    for index in range(x.size - 1):
        if max(x[index], x[index + 1]) < front.position - half_window:
            continue
        if min(x[index], x[index + 1]) > front.position + half_window:
            continue
        left = float(normalized[index] - fraction)
        right = float(normalized[index + 1] - fraction)
        if left == 0.0:
            candidates.append(float(x[index]))
        if left * right < 0.0 or right == 0.0:
            denominator = right - left
            if denominator != 0.0:
                weight = -left / denominator
                candidates.append(float(x[index] + weight * (x[index + 1] - x[index])))
    if not candidates:
        return None, 0
    unique = sorted(set(candidates))
    return min(unique, key=lambda value: abs(value - front.position)), len(unique)


def _window_mean(
    x: np.ndarray, profile: np.ndarray, left: float, right: float
) -> float:
    mask = (x >= left) & (x <= right)
    if not np.any(mask):
        midpoint = 0.5 * (left + right)
        return float(profile[int(np.argmin(np.abs(x - midpoint)))])
    return float(np.mean(profile[mask]))


def _oscillatory_lobe_mass(
    error_field: np.ndarray,
    outside_front_band: np.ndarray,
    *,
    cell_area: float,
    normalization: float,
    tolerance: float,
) -> tuple[float, int, float]:
    if error_field.ndim != 2 or error_field.shape[1] != outside_front_band.size:
        raise ValueError("error_field must have shape [ny,nx]")
    error_profile = np.mean(error_field, axis=0)
    oscillatory_mass = 0.0
    signed_bias = 0.0
    lobe_count = 0
    index = 0
    while index < error_profile.size:
        if not outside_front_band[index]:
            index += 1
            continue
        stop = index
        while stop < error_profile.size and outside_front_band[stop]:
            stop += 1
        field_segment = error_field[:, index:stop]
        positive = float(np.sum(np.maximum(field_segment, 0.0)))
        negative = float(np.sum(np.maximum(-field_segment, 0.0)))
        oscillatory_mass += 2.0 * min(positive, negative)
        signed_bias += float(np.sum(field_segment))
        segment = error_profile[index:stop]
        active = np.flatnonzero(np.abs(segment) > tolerance) + index
        if active.size:
            runs: list[list[int]] = [[int(active[0])]]
            last_sign = int(np.sign(error_profile[active[0]]))
            for active_index in active[1:]:
                sign = int(np.sign(error_profile[active_index]))
                if sign == last_sign:
                    runs[-1].append(int(active_index))
                else:
                    runs.append([int(active_index)])
                    last_sign = sign
            if len(runs) >= 2:
                lobe_count += len(runs)
        index = stop
    physical_scale = cell_area
    denominator = max(normalization, np.finfo(np.float64).tiny)
    return (
        oscillatory_mass * physical_scale / denominator,
        lobe_count,
        signed_bias * physical_scale / denominator,
    )


def shock_representation_metrics(
    predicted_increment: np.ndarray,
    case: SyntheticFrontCase,
    grid: StructuredCellGrid,
    *,
    front_band_width: float = FRONT_BAND_WIDTH,
) -> dict[str, Any]:
    """Compute the preregistered cell-average structure diagnostics."""

    prediction = np.asarray(predicted_increment, dtype=np.float64)
    target_increment = np.asarray(case.increment, dtype=np.float64)
    target_next = np.asarray(case.target, dtype=np.float64)
    if (
        prediction.shape != grid.array_shape
        or target_increment.shape != grid.array_shape
        or target_next.shape != grid.array_shape
    ):
        raise ValueError("prediction and target must match the grid resolution")
    if not np.isfinite(prediction).all():
        raise ValueError("prediction must be finite")
    volume = grid.hx * grid.hy
    predicted_next = case.current + prediction
    error = prediction - target_increment
    target_l2 = float(np.sqrt(np.sum(np.square(target_increment)) * volume))
    error_l2 = float(np.sqrt(np.sum(np.square(error)) * volume))
    target_l1 = float(np.sum(np.abs(target_increment)) * volume)
    # Every registered family has unit jump or unit sinusoidal amplitude.
    amplitude = 1.0
    band = float(front_band_width)
    if not np.isfinite(band) or band <= 0.0:
        raise ValueError("front_band_width must be finite and positive")
    feature_positions = case.feature_positions
    outside_x = np.ones(grid.nx, dtype=bool)
    for position in feature_positions:
        outside_x &= np.abs(grid.x_centers - position) > band
    outside = np.repeat(outside_x[None, :], grid.ny, axis=0)
    smooth_error_l2 = float(np.sqrt(np.sum(np.square(error[outside])) * volume))
    smooth_volume = max(float(np.sum(outside) * volume), volume)
    profile = np.mean(predicted_next, axis=0)
    target_profile = np.mean(target_next, axis=0)
    oscillatory_mass, alternating_lobes, signed_bias = _oscillatory_lobe_mass(
        error,
        outside_x,
        cell_area=volume,
        normalization=target_l1,
        tolerance=1.0e-10 * amplitude,
    )
    target_tv = _total_variation(target_next, grid)
    prediction_tv = _total_variation(predicted_next, grid)
    tv_change = (prediction_tv - target_tv) / max(target_tv, np.finfo(np.float64).tiny)
    increment_integral_error = abs(float(np.sum(error) * volume)) / max(
        target_l1, np.finfo(np.float64).tiny
    )
    state_integral_error = abs(float(np.sum(predicted_next - target_next) * volume))
    state_integral_error /= max(
        float(np.sum(np.abs(target_next)) * volume), np.finfo(np.float64).tiny
    )
    front_rows: list[dict[str, Any]] = []
    for front in case.target_fronts:
        predicted_crossings = {
            fraction: _crossing_position(
                grid.x_centers,
                profile,
                front=front,
                fraction=fraction,
                half_window=2.0 * band,
            )
            for fraction in (0.1, 0.5, 0.9)
        }
        target_crossings = {
            fraction: _crossing_position(
                grid.x_centers,
                target_profile,
                front=front,
                fraction=fraction,
                half_window=2.0 * band,
            )
            for fraction in (0.1, 0.5, 0.9)
        }
        predicted_left = _window_mean(
            grid.x_centers,
            profile,
            front.position - 2.0 * band,
            front.position - band,
        )
        predicted_right = _window_mean(
            grid.x_centers,
            profile,
            front.position + band,
            front.position + 2.0 * band,
        )
        target_left = _window_mean(
            grid.x_centers,
            target_profile,
            front.position - 2.0 * band,
            front.position - band,
        )
        target_right = _window_mean(
            grid.x_centers,
            target_profile,
            front.position + band,
            front.position + 2.0 * band,
        )
        predicted_midpoint = predicted_crossings[0.5][0]
        target_midpoint = target_crossings[0.5][0]
        predicted_edge_crossings = (
            predicted_crossings[0.1][0],
            predicted_crossings[0.9][0],
        )
        target_edge_crossings = (
            target_crossings[0.1][0],
            target_crossings[0.9][0],
        )
        predicted_thickness = (
            None
            if any(value is None for value in predicted_edge_crossings)
            else abs(predicted_edge_crossings[1] - predicted_edge_crossings[0])
        )
        target_thickness = (
            None
            if any(value is None for value in target_edge_crossings)
            else abs(target_edge_crossings[1] - target_edge_crossings[0])
        )
        position_error = (
            None
            if predicted_midpoint is None or target_midpoint is None
            else abs(predicted_midpoint - target_midpoint)
        )
        thickness_excess = (
            None
            if predicted_thickness is None or target_thickness is None
            else predicted_thickness - target_thickness
        )
        predicted_crossings_valid = all(
            count == 1 for _, count in predicted_crossings.values()
        )
        target_crossings_valid = all(
            count == 1 for _, count in target_crossings.values()
        )
        target_strength = target_right - target_left
        front_rows.append(
            {
                "position": float(front.position),
                "position_error": position_error,
                "predicted_midpoint_crossing_count": int(predicted_crossings[0.5][1]),
                "target_midpoint_crossing_count": int(target_crossings[0.5][1]),
                "ambiguous_predicted_midpoint": bool(predicted_crossings[0.5][1] != 1),
                "predicted_crossings_valid": bool(predicted_crossings_valid),
                "target_crossings_valid": bool(target_crossings_valid),
                "front_gate_valid": bool(
                    predicted_crossings_valid and target_crossings_valid
                ),
                "strength_error": float(
                    abs((predicted_right - predicted_left) - target_strength)
                    / max(abs(target_strength), np.finfo(np.float64).tiny)
                ),
                "predicted_thickness": predicted_thickness,
                "target_thickness": target_thickness,
                "thickness_excess": thickness_excess,
            }
        )
    return {
        "relative_increment_l2": error_l2 / max(target_l2, np.finfo(np.float64).tiny),
        "normalized_overshoot": max(
            0.0, float(np.max(predicted_next) - np.max(target_next)) / amplitude
        ),
        "normalized_undershoot": max(
            0.0, float(np.min(target_next) - np.min(predicted_next)) / amplitude
        ),
        "oscillatory_mass_outside_front_band": float(oscillatory_mass),
        "alternating_lobe_count_outside_front_band": int(alternating_lobes),
        "signed_bias_outside_front_band": float(signed_bias),
        "smooth_region_error": smooth_error_l2 / (amplitude * np.sqrt(smooth_volume)),
        "total_variation_change": float(tv_change),
        "positive_total_variation_excess": max(0.0, float(tv_change)),
        "total_variation_deficit": max(0.0, float(-tv_change)),
        "pulse_integral_error": (
            float(increment_integral_error) if case.family == "pulse" else None
        ),
        "increment_integral_error": float(increment_integral_error),
        "next_state_integral_error": float(state_integral_error),
        "fronts": front_rows,
        "physical_spectrum": physical_spectrum_summary(error, grid),
    }


def pcno_aux_tensors(
    grid: StructuredCellGrid,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, ...]:
    """Convert one unpadded structured grid to the current PCNO aux contract."""

    geometry = grid.geometry
    return (
        torch.ones((1, grid.nx * grid.ny, 1), dtype=dtype, device=device),
        torch.as_tensor(geometry.nodes, dtype=dtype, device=device).unsqueeze(0),
        torch.as_tensor(geometry.node_weights, dtype=dtype, device=device).unsqueeze(0),
        torch.as_tensor(
            geometry.directed_edges, dtype=torch.long, device=device
        ).unsqueeze(0),
        torch.as_tensor(
            geometry.edge_gradient_weights, dtype=dtype, device=device
        ).unsqueeze(0),
    )


def pcno_case_input(
    case: SyntheticFrontCase,
    grid: StructuredCellGrid,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Return ``[x,y,quadrature-density,current]`` for a scalar PCNO case."""

    features = np.concatenate(
        (
            grid.nodes,
            grid.geometry.node_rhos,
            case.current.reshape(-1, 1),
        ),
        axis=1,
    )
    return torch.as_tensor(features, dtype=dtype, device=device).unsqueeze(0)


@torch.no_grad()
def scalar_gradient_variants(
    field: np.ndarray,
    grid: StructuredCellGrid,
    *,
    fixed_radius: float = FIXED_PHYSICAL_RADIUS,
    pass_wavenumber: float = SMOOTH_FILTER_PASS_WAVENUMBER,
    stop_wavenumber: float = SMOOTH_FILTER_STOP_WAVENUMBER,
) -> dict[str, np.ndarray]:
    """Return raw, native-hop, fixed-radius, and mollified scalar gradients."""

    values = np.asarray(field, dtype=np.float64)
    if values.shape != grid.array_shape or not np.isfinite(values).all():
        raise ValueError("field must be finite and match the grid resolution")
    aux = pcno_aux_tensors(grid, dtype=torch.float64)
    tensor = torch.as_tensor(values.reshape(1, 1, -1), dtype=torch.float64)
    raw = compute_gradient(tensor, aux[3], aux[4])
    native = graph_neighbor_average(raw, aux[3], iterations=2)
    local_radius = 2.0 * grid.hx
    local_operator = build_graph_ball_operator(
        grid.nodes,
        grid.geometry.edges,
        grid.cell_volumes,
        radius=local_radius,
    )
    local_prepared = prepare_graph_ball_operator(
        local_operator, device=raw.device, dtype=raw.dtype
    )
    local_ball = graph_ball_average(raw, local_prepared)
    if np.isclose(local_radius, float(fixed_radius), rtol=0.0, atol=1.0e-15):
        fixed = local_ball
    else:
        fixed_operator = build_graph_ball_operator(
            grid.nodes,
            grid.geometry.edges,
            grid.cell_volumes,
            radius=float(fixed_radius),
        )
        fixed_prepared = prepare_graph_ball_operator(
            fixed_operator, device=raw.device, dtype=raw.dtype
        )
        fixed = graph_ball_average(raw, fixed_prepared)
    raw_grid = raw.squeeze(0).T.cpu().numpy().reshape(grid.ny, grid.nx, 2)
    mollified_grid = physical_cosine_filter(
        raw_grid,
        grid,
        kind="smooth",
        pass_wavenumber=pass_wavenumber,
        stop_wavenumber=stop_wavenumber,
    )
    mollified = mollified_grid.reshape(-1, 2).T
    return {
        "raw_gradient": raw.squeeze(0).cpu().numpy(),
        "native_two_hop": native.squeeze(0).cpu().numpy(),
        "local_two_hop_radius_ball": local_ball.squeeze(0).cpu().numpy(),
        "fixed_physical_radius": fixed.squeeze(0).cpu().numpy(),
        "h_scaled_raw": (
            np.asarray((grid.hx, grid.hy), dtype=np.float64)[:, None]
            * raw.squeeze(0).cpu().numpy()
        ),
        "fixed_physical_mollified": mollified,
    }


def gradient_distribution_statistics(
    gradient: np.ndarray,
    grid: StructuredCellGrid,
    fronts: Sequence[FrontSpec],
) -> dict[str, Any]:
    """Separate distributional peak scaling from physical support and strength."""

    values = np.asarray(gradient, dtype=np.float64)
    node_count = grid.nx * grid.ny
    if values.shape != (2, node_count) or not np.isfinite(values).all():
        raise ValueError("scalar 2-D gradient must have shape [2,N] and be finite")
    if not fronts:
        raise ValueError("at least one front is required")
    magnitude = np.linalg.norm(values, axis=0)
    mass = magnitude * grid.cell_volumes
    total_mass = float(np.sum(mass))
    front_positions = np.asarray([front.position for front in fronts])
    distance = np.min(
        np.abs(grid.nodes[:, :1] - front_positions.reshape(1, -1)), axis=1
    )
    order = np.argsort(distance, kind="stable")
    cumulative = np.cumsum(mass[order])

    def support_radius(fraction: float) -> float:
        if total_mass <= np.finfo(np.float64).tiny:
            return 0.0
        index = int(np.searchsorted(cumulative, fraction * total_mass, side="left"))
        return float(distance[order[min(index, order.size - 1)]])

    x_gradient = values[0].reshape(grid.array_shape)
    signed_strength: list[dict[str, float]] = []
    front_distance = np.abs(grid.x_centers[:, None] - front_positions.reshape(1, -1))
    nearest_front = np.argmin(front_distance, axis=1)
    for front_index, front in enumerate(fronts):
        mask_x = nearest_front == front_index
        integral = float(np.sum(x_gradient[:, mask_x]) * grid.hx * grid.hy)
        signed_strength.append(
            {
                "position": float(front.position),
                "normalized_signed_strength": integral / (front.jump * grid.lengths[1]),
            }
        )
    radius_50 = support_radius(0.5)
    radius_90 = support_radius(0.9)
    return {
        "peak": float(np.max(magnitude)),
        "q99": float(np.quantile(magnitude, 0.99)),
        "h_scaled_peak": float(grid.hx * np.max(magnitude)),
        "l1_strength": total_mass,
        "support_50_half_width": radius_50,
        "support_90_half_width": radius_90,
        "support_50_width_in_cells": 2.0 * radius_50 / grid.hx,
        "support_90_width_in_cells": 2.0 * radius_90 / grid.hx,
        "transverse_to_normal_l1_ratio": float(
            np.sum(np.abs(values[1]) * grid.cell_volumes)
            / max(
                np.sum(np.abs(values[0]) * grid.cell_volumes),
                np.finfo(np.float64).tiny,
            )
        ),
        "front_signed_strength": signed_strength,
    }


def softsign_saturation_summary(
    pre_softsign: torch.Tensor,
    node_weights: torch.Tensor,
    *,
    output_threshold: float = 0.9,
    front_band_mask: torch.Tensor | None = None,
) -> dict[str, float]:
    """Return global and optional front-band weighted Softsign diagnostics."""

    if pre_softsign.ndim != 3:
        raise ValueError("pre_softsign must have shape [B,C,N]")
    weights = node_weights
    if weights.ndim == 3 and weights.shape[-1] == 1:
        weights = weights[..., 0]
    if weights.shape != (pre_softsign.shape[0], pre_softsign.shape[2]):
        raise ValueError("node_weights must align with batch and node axes")
    threshold = float(output_threshold)
    if not 0.0 < threshold < 1.0:
        raise ValueError("output_threshold must lie in (0,1)")
    base_weights = weights.to(device=pre_softsign.device, dtype=pre_softsign.dtype)
    absolute = torch.abs(pre_softsign)
    output_magnitude = absolute / (1.0 + absolute)
    saturated = output_magnitude >= threshold
    energy = torch.square(pre_softsign)
    derivative = torch.pow(1.0 + absolute, -2.0)

    def weighted_quantile(
        value: torch.Tensor, weight: torch.Tensor, quantile: float
    ) -> float:
        flattened_value = value.reshape(-1)
        flattened_weight = weight.expand_as(value).reshape(-1)
        order = torch.argsort(flattened_value)
        sorted_value = flattened_value[order]
        cumulative = torch.cumsum(flattened_weight[order], dim=0)
        total = cumulative[-1]
        if float(total) <= 0.0:
            raise ValueError("saturation region has zero physical weight")
        index = torch.searchsorted(cumulative, quantile * total, right=False)
        return float(sorted_value[torch.clamp(index, max=sorted_value.numel() - 1)])

    def region_summary(region_weights: torch.Tensor, prefix: str) -> dict[str, float]:
        broadcast = region_weights[:, None, :]
        denominator = pre_softsign.shape[1] * torch.sum(region_weights)
        if float(denominator) <= 0.0:
            raise ValueError("saturation region has zero physical weight")
        energy_denominator = torch.sum(energy * broadcast)
        return {
            f"{prefix}volume_channel_fraction": float(
                torch.sum(saturated * broadcast) / denominator
            ),
            f"{prefix}feature_energy_fraction": float(
                torch.sum(energy * saturated * broadcast)
                / torch.clamp_min(
                    energy_denominator,
                    torch.finfo(pre_softsign.dtype).tiny,
                )
            ),
            f"{prefix}median_derivative": weighted_quantile(derivative, broadcast, 0.5),
            f"{prefix}q10_derivative": weighted_quantile(derivative, broadcast, 0.1),
            f"{prefix}q01_derivative": weighted_quantile(derivative, broadcast, 0.01),
        }

    result = {
        "softsign_output_threshold": threshold,
        "equivalent_input_threshold": threshold / (1.0 - threshold),
    }
    result.update(region_summary(base_weights, ""))
    if front_band_mask is not None:
        mask = front_band_mask
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        if mask.ndim == 3 and mask.shape[-1] == 1:
            mask = mask[..., 0]
        if mask.shape != base_weights.shape:
            raise ValueError("front_band_mask must align with batch and node axes")
        result.update(
            region_summary(
                base_weights
                * mask.to(device=base_weights.device, dtype=base_weights.dtype),
                "front_band_",
            )
        )
    return result


def _pcno_decode(backbone: torch.nn.Module, hidden: torch.Tensor) -> torch.Tensor:
    decoded = hidden.permute(0, 2, 1)
    if backbone.fc_dim > 0:
        decoded = backbone.fc1(decoded)
        if backbone.act is not None:
            decoded = backbone.act(decoded)
    return backbone.fc2(decoded)


def _pcno_advance(
    backbone: torch.nn.Module,
    hidden: torch.Tensor,
    combined: torch.Tensor,
    layer: int,
) -> torch.Tensor:
    if backbone.act is not None and layer != len(backbone.ws) - 1:
        return hidden + backbone.act(combined)
    return combined


def _pcno_branches(
    backbone: torch.nn.Module,
    hidden: torch.Tensor,
    layer: int,
    fourier_tensors: Sequence[torch.Tensor],
    aux: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    spectral = backbone.sp_convs[layer](hidden, *fourier_tensors)
    pointwise = backbone.ws[layer](hidden)
    differential = backbone.gws[layer](hidden, aux[3], aux[4])
    return spectral, pointwise, differential


@torch.no_grad()
def trace_pcno_differential_stages(
    backbone: torch.nn.Module,
    model_input: torch.Tensor,
    aux: Sequence[torch.Tensor],
    *,
    front_band_mask: torch.Tensor | None = None,
) -> DifferentialPathTrace:
    """Replay current PCNO order and expose every differential-path stage.

    ``decoded_differential_ablation_response`` is the full output minus a
    same-hidden intervention that disables the selected layer's differential
    branch and then recomputes every downstream layer.  It is not interpreted
    as an additive share.
    """

    required = ("fc0", "sp_convs", "ws", "gws", "fc2", "act", "fc_dim")
    if any(not hasattr(backbone, name) for name in required):
        raise TypeError("backbone does not expose the maintained PCNO modules")
    if len(aux) != 5 or model_input.ndim != 3:
        raise ValueError("model_input and aux do not satisfy the PCNO contract")
    module_lengths = {
        len(backbone.sp_convs),
        len(backbone.ws),
        len(backbone.gws),
    }
    if len(module_lengths) != 1:
        raise ValueError("PCNO branch module lists must have equal lengths")
    fourier_tensors = backbone.prepare_fourier_tensors(aux[1], aux[2])
    model_output = backbone(model_input, aux, fourier_tensors=fourier_tensors)
    hidden = backbone.fc0(model_input).permute(0, 2, 1)
    hidden_inputs: list[torch.Tensor] = []
    stage_values: list[tuple[torch.Tensor, ...]] = []
    for layer, differential in enumerate(backbone.gws):
        hidden_inputs.append(hidden)
        raw = compute_gradient(hidden, aux[3], aux[4])
        post_aggregation = graph_neighbor_average(raw, aux[3], iterations=2)
        pre_softsign = differential.gw1 * post_aggregation
        post_softsign = differential.geo_act(pre_softsign)
        branch_output = differential.gw2(post_softsign)
        spectral = backbone.sp_convs[layer](hidden, *fourier_tensors)
        pointwise = backbone.ws[layer](hidden)
        combined = spectral + pointwise + branch_output
        stage_values.append(
            (raw, post_aggregation, pre_softsign, post_softsign, branch_output)
        )
        hidden = _pcno_advance(backbone, hidden, combined, layer)
    replay_output = _pcno_decode(backbone, hidden)

    traces: list[DifferentialLayerTrace] = []
    for selected_layer, values in enumerate(stage_values):
        counterfactual_hidden = hidden_inputs[selected_layer]
        spectral = backbone.sp_convs[selected_layer](
            counterfactual_hidden, *fourier_tensors
        )
        pointwise = backbone.ws[selected_layer](counterfactual_hidden)
        counterfactual_hidden = _pcno_advance(
            backbone,
            counterfactual_hidden,
            spectral + pointwise,
            selected_layer,
        )
        for downstream_layer in range(selected_layer + 1, len(backbone.ws)):
            branches = _pcno_branches(
                backbone,
                counterfactual_hidden,
                downstream_layer,
                fourier_tensors,
                aux,
            )
            counterfactual_hidden = _pcno_advance(
                backbone,
                counterfactual_hidden,
                sum(branches),
                downstream_layer,
            )
        counterfactual_output = _pcno_decode(backbone, counterfactual_hidden)
        raw, post_aggregation, pre_softsign, post_softsign, branch_output = values
        traces.append(
            DifferentialLayerTrace(
                layer=selected_layer,
                raw_gradient=raw,
                post_aggregation=post_aggregation,
                pre_softsign=pre_softsign,
                post_softsign=post_softsign,
                branch_output=branch_output,
                decoded_differential_ablation_response=(
                    model_output - counterfactual_output
                ),
                saturation=softsign_saturation_summary(
                    pre_softsign,
                    aux[2],
                    front_band_mask=front_band_mask,
                ),
            )
        )
    return DifferentialPathTrace(
        model_output=model_output,
        replay_output=replay_output,
        layers=tuple(traces),
    )
