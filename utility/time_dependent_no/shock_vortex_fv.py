"""Canonical 2D Euler shock--vortex finite-volume reference solver.

This module owns the dynamic finite-volume data contract used by D037.  It is
deliberately independent of the neural models: a dimension-by-dimension
WENO5-JS reconstruction, HLLC interface flux, and SSPRK3 time integrator evolve
the published Mach-1.1 shock--isentropic-vortex case.  Every accepted SSPRK
substep contributes its correctly weighted, owner-oriented face impulse to the
saved coarse control-volume artifact.

The solver never clips or floors an accepted cell state.  Invalid reconstructed
face states fall back to their adjacent admissible cell states; an invalid RK
stage rejects the attempted step and retries with half the step size.  Both
events are recorded as numerical provenance.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

BOUNDARY_TAG_NAMES = ("interior", "x_min", "x_max", "y_min", "y_max")
REFERENCE_CONTRACT_CHECK_KEYS = frozenset(
    {
        "initial_cell_average_quadrature_certified",
        "restricted_initial_cell_average_quadrature_certified",
        "cell_volumes_positive",
        "face_measures_positive",
        "face_normals_unit",
        "oriented_connectivity_valid",
        "face_normal_orientation_consistent",
        "boundary_tags_and_locations_consistent",
        "structured_face_geometry_consistent",
        "cell_incidence_complete",
        "cell_area_vector_closure",
        "artifact_array_shapes_consistent",
        "accepted_times_strictly_increasing",
        "accepted_dt_matches_time_increments",
        "accepted_intervals_and_dt_sums_valid",
        "all_saved_times_reached_exactly",
        "raw_states_finite",
        "raw_density_pressure_admissible",
        "interval_impulse_closure",
        "boundary_exchange_closure",
    }
)


@dataclass(frozen=True)
class ShockVortexFVConfig:
    """Frozen canonical case and reference-solver configuration."""

    nx: int = 1000
    ny: int = 400
    coarse_nx: int = 250
    coarse_ny: int = 100
    x_min: float = 0.0
    x_max: float = 2.0
    y_min: float = 0.0
    y_max: float = 1.0
    gamma: float = 1.4
    shock_mach: float = 1.1
    shock_x: float = 0.5
    right_rho: float = 1.1691
    right_u: float = 1.1133
    right_pressure: float = 1.245
    vortex_x: float = 0.25
    vortex_y: float = 0.5
    vortex_epsilon: float = 0.3
    vortex_alpha: float = 0.204
    vortex_radius: float = 0.05
    t_final: float = 0.6
    output_times: tuple[float, ...] = (
        0.0,
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
    )
    cfl: float = 0.35
    ghost_cells: int = 3
    weno_epsilon: float = 1.0e-6
    density_admissibility_threshold: float = 1.0e-10
    pressure_admissibility_threshold: float = 1.0e-10
    max_step_retries: int = 12
    initial_quadrature_order: int = 8

    def validated(self) -> "ShockVortexFVConfig":
        """Validate the geometry, canonical constants, and save-time contract."""

        if self.nx < 6 or self.ny < 6:
            raise ValueError("nx and ny must both be at least six for WENO5")
        if self.coarse_nx < 1 or self.coarse_ny < 1:
            raise ValueError("coarse_nx and coarse_ny must be positive")
        if self.nx % self.coarse_nx or self.ny % self.coarse_ny:
            raise ValueError("fine dimensions must be divisible by coarse dimensions")
        if not self.x_min < self.x_max or not self.y_min < self.y_max:
            raise ValueError("domain bounds must be strictly increasing")
        if self.gamma <= 1.0:
            raise ValueError("gamma must exceed one")
        if self.vortex_alpha <= 0.0 or self.vortex_radius <= 0.0:
            raise ValueError("vortex alpha and radius must be positive")
        if not 0.0 < self.cfl < 1.0:
            raise ValueError("cfl must lie in (0, 1)")
        if self.ghost_cells < 3:
            raise ValueError("at least three ghost cells are required")
        if self.weno_epsilon <= 0.0:
            raise ValueError("weno_epsilon must be positive")
        if self.density_admissibility_threshold < 0.0:
            raise ValueError("density admissibility threshold must be nonnegative")
        if self.pressure_admissibility_threshold < 0.0:
            raise ValueError("pressure admissibility threshold must be nonnegative")
        if self.max_step_retries < 1:
            raise ValueError("max_step_retries must be positive")
        if self.initial_quadrature_order < 2:
            raise ValueError("initial_quadrature_order must be at least two")
        times = np.asarray(self.output_times, dtype=np.float64)
        if times.ndim != 1 or times.size < 2 or not np.all(np.isfinite(times)):
            raise ValueError("output_times must contain at least two finite values")
        if not np.isclose(times[0], 0.0, rtol=0.0, atol=1.0e-14):
            raise ValueError("output_times must start at zero")
        if not np.isclose(times[-1], self.t_final, rtol=0.0, atol=1.0e-14):
            raise ValueError("output_times must end at t_final")
        if np.any(np.diff(times) <= 0.0):
            raise ValueError("output_times must be strictly increasing")
        return self

    @property
    def dx(self) -> float:
        return (self.x_max - self.x_min) / self.nx

    @property
    def dy(self) -> float:
        return (self.y_max - self.y_min) / self.ny

    @property
    def restriction_x(self) -> int:
        return self.nx // self.coarse_nx

    @property
    def restriction_y(self) -> int:
        return self.ny // self.coarse_ny

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["output_times"] = list(self.output_times)
        payload["dx"] = self.dx
        payload["dy"] = self.dy
        payload["restriction_x"] = self.restriction_x
        payload["restriction_y"] = self.restriction_y
        return payload


@dataclass(frozen=True)
class StructuredFVGeometry:
    """Owner-oriented Cartesian control-volume geometry on the coarse mesh."""

    cell_centers: np.ndarray
    cell_volume: np.ndarray
    face_centers: np.ndarray
    face_measure: np.ndarray
    face_normal: np.ndarray
    face_owner: np.ndarray
    face_neighbor: np.ndarray
    face_axis: np.ndarray
    face_boundary_tag: np.ndarray

    @property
    def interior_face_mask(self) -> np.ndarray:
        return self.face_neighbor >= 0


@dataclass(frozen=True)
class ShockVortexReferenceResult:
    """In-memory reference trajectory and accepted-step provenance."""

    config: ShockVortexFVConfig
    geometry: StructuredFVGeometry
    times: np.ndarray
    states: np.ndarray
    face_impulses: np.ndarray
    accepted_step_times: np.ndarray
    accepted_step_dt: np.ndarray
    accepted_step_interval: np.ndarray
    accepted_step_rejections: np.ndarray
    accepted_step_face_fallbacks: np.ndarray
    interval_rejection_count: np.ndarray
    interval_face_fallback_count: np.ndarray
    interval_closure_max_abs: np.ndarray
    interval_closure_relative_l2: np.ndarray
    interval_boundary_exchange: np.ndarray
    initial_quadrature_max_abs: float
    initial_quadrature_relative_l2: float
    restricted_initial_quadrature_max_abs: float
    restricted_initial_quadrature_relative_l2: float
    minimum_density: float
    minimum_pressure: float


def make_structured_fv_geometry(config: ShockVortexFVConfig) -> StructuredFVGeometry:
    """Construct the validated coarse Cartesian geometry and face orientation."""

    config.validated()
    nx = config.coarse_nx
    ny = config.coarse_ny
    dx = (config.x_max - config.x_min) / nx
    dy = (config.y_max - config.y_min) / ny
    x = config.x_min + (np.arange(nx, dtype=np.float64) + 0.5) * dx
    y = config.y_min + (np.arange(ny, dtype=np.float64) + 0.5) * dy
    xx, yy = np.meshgrid(x, y, indexing="xy")
    cell_centers = np.stack((xx.reshape(-1), yy.reshape(-1)), axis=-1)
    cell_volume = np.full(nx * ny, dx * dy, dtype=np.float64)

    centers: list[tuple[float, float]] = []
    measures: list[float] = []
    normals: list[tuple[float, float]] = []
    owners: list[int] = []
    neighbors: list[int] = []
    axes: list[int] = []
    tags: list[int] = []

    for j in range(ny):
        yc = config.y_min + (j + 0.5) * dy
        for i in range(nx + 1):
            centers.append((config.x_min + i * dx, yc))
            measures.append(dy)
            axes.append(0)
            if i == 0:
                owners.append(j * nx)
                neighbors.append(-1)
                normals.append((-1.0, 0.0))
                tags.append(1)
            elif i == nx:
                owners.append(j * nx + nx - 1)
                neighbors.append(-1)
                normals.append((1.0, 0.0))
                tags.append(2)
            else:
                owners.append(j * nx + i - 1)
                neighbors.append(j * nx + i)
                normals.append((1.0, 0.0))
                tags.append(0)

    for j in range(ny + 1):
        yc = config.y_min + j * dy
        for i in range(nx):
            centers.append((config.x_min + (i + 0.5) * dx, yc))
            measures.append(dx)
            axes.append(1)
            if j == 0:
                owners.append(i)
                neighbors.append(-1)
                normals.append((0.0, -1.0))
                tags.append(3)
            elif j == ny:
                owners.append((ny - 1) * nx + i)
                neighbors.append(-1)
                normals.append((0.0, 1.0))
                tags.append(4)
            else:
                owners.append((j - 1) * nx + i)
                neighbors.append(j * nx + i)
                normals.append((0.0, 1.0))
                tags.append(0)

    return StructuredFVGeometry(
        cell_centers=cell_centers,
        cell_volume=cell_volume,
        face_centers=np.asarray(centers, dtype=np.float64),
        face_measure=np.asarray(measures, dtype=np.float64),
        face_normal=np.asarray(normals, dtype=np.float64),
        face_owner=np.asarray(owners, dtype=np.int64),
        face_neighbor=np.asarray(neighbors, dtype=np.int64),
        face_axis=np.asarray(axes, dtype=np.int8),
        face_boundary_tag=np.asarray(tags, dtype=np.int8),
    )


def fine_cell_centers(
    config: ShockVortexFVConfig,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return fine-grid cell centers with shape ``[ny, nx, 2]``."""

    config.validated()
    x = (
        config.x_min
        + (torch.arange(config.nx, device=device, dtype=dtype) + 0.5) * config.dx
    )
    y = (
        config.y_min
        + (torch.arange(config.ny, device=device, dtype=dtype) + 0.5) * config.dy
    )
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((xx, yy), dim=-1)


def shock_vortex_initial_primitive(
    positions: torch.Tensor,
    config: ShockVortexFVConfig,
) -> torch.Tensor:
    """Evaluate the published primitive initial condition at cell centers."""

    config.validated()
    if positions.ndim < 2 or positions.shape[-1] != 2:
        raise ValueError("positions must have last dimension two")
    x = positions[..., 0]
    y = positions[..., 1]
    left = x < config.shock_x
    dx = x - config.vortex_x
    dy = y - config.vortex_y
    radius = torch.sqrt(dx * dx + dy * dy)
    tau = radius / config.vortex_radius
    envelope = torch.exp(config.vortex_alpha * (1.0 - tau * tau))
    thermal = 1.0 - (
        (config.gamma - 1.0)
        * config.vortex_epsilon**2
        * envelope**2
        / (4.0 * config.vortex_alpha * config.gamma)
    )
    vortex_rho = thermal ** (1.0 / (config.gamma - 1.0))
    vortex_pressure = vortex_rho**config.gamma
    upstream_u = config.shock_mach * np.sqrt(config.gamma)
    vortex_u = upstream_u + (
        config.vortex_epsilon * envelope * dy / config.vortex_radius
    )
    vortex_v = -config.vortex_epsilon * envelope * dx / config.vortex_radius
    rho = torch.where(left, vortex_rho, torch.full_like(x, config.right_rho))
    u = torch.where(left, vortex_u, torch.full_like(x, config.right_u))
    v = torch.where(left, vortex_v, torch.zeros_like(x))
    pressure = torch.where(
        left,
        vortex_pressure,
        torch.full_like(x, config.right_pressure),
    )
    primitive = torch.stack((rho, u, v, pressure), dim=-1)
    if not torch.isfinite(primitive).all():
        raise ValueError("initial primitive state is non-finite")
    if torch.any(rho <= 0.0) or torch.any(pressure <= 0.0):
        raise ValueError("initial primitive state is not admissible")
    return primitive


def shock_vortex_initial_cell_averages(
    config: ShockVortexFVConfig,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
    quadrature_order: int | None = None,
) -> torch.Tensor:
    """Return conservative fine-cell averages of the discontinuous initial state.

    Tensor-product Gauss--Legendre quadrature is applied separately to the
    smooth portions on either side of ``shock_x``.  Splitting a crossing cell
    at the shock is essential: center sampling makes the initial finite-volume
    state depend on whether the discontinuity happens to hit a center or face.
    """

    config.validated()
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("initial-state dtype must be torch.float32 or torch.float64")
    order = (
        config.initial_quadrature_order
        if quadrature_order is None
        else int(quadrature_order)
    )
    if order < 2:
        raise ValueError("quadrature_order must be at least two")
    resolved_device = torch.device(device)
    rule_nodes_np, rule_weights_np = np.polynomial.legendre.leggauss(order)
    rule_nodes = torch.as_tensor(rule_nodes_np, device=resolved_device, dtype=dtype)
    rule_weights = torch.as_tensor(rule_weights_np, device=resolved_device, dtype=dtype)

    x_left = (
        config.x_min
        + torch.arange(config.nx, device=resolved_device, dtype=dtype) * config.dx
    )
    x_right = x_left + config.dx
    y_bottom = (
        config.y_min
        + torch.arange(config.ny, device=resolved_device, dtype=dtype) * config.dy
    )
    y_mid = y_bottom + 0.5 * config.dy
    y_points = y_mid[:, None] + 0.5 * config.dy * rule_nodes[None, :]
    y_weights = 0.5 * rule_weights
    average = torch.zeros(
        (config.ny, config.nx, 4), device=resolved_device, dtype=dtype
    )

    shock = torch.as_tensor(config.shock_x, device=resolved_device, dtype=dtype)
    segments = (
        (x_left, torch.minimum(x_right, shock)),
        (torch.maximum(x_left, shock), x_right),
    )
    for segment_left, segment_right in segments:
        width = torch.clamp(segment_right - segment_left, min=0.0)
        midpoint = 0.5 * (segment_left + segment_right)
        x_points = midpoint[:, None] + 0.5 * width[:, None] * rule_nodes[None, :]
        x_weights = 0.5 * width[:, None] * rule_weights[None, :] / config.dx
        for x_index in range(order):
            x = x_points[:, x_index].expand(config.ny, -1)
            x_weight = x_weights[:, x_index][None, :, None]
            for y_index in range(order):
                y = y_points[:, y_index][:, None].expand(-1, config.nx)
                positions = torch.stack((x, y), dim=-1)
                conservative = primitive_to_conservative(
                    shock_vortex_initial_primitive(positions, config),
                    config.gamma,
                )
                average += y_weights[y_index] * x_weight * conservative
    if not bool(torch.isfinite(average).all()):
        raise ValueError("quadrature produced a non-finite initial cell average")
    return average


def primitive_to_conservative(primitive: torch.Tensor, gamma: float) -> torch.Tensor:
    """Convert ``[rho,u,v,p]`` to ``[rho,rho*u,rho*v,E]`` without clipping."""

    rho, u, v, pressure = primitive.unbind(dim=-1)
    energy = pressure / (gamma - 1.0) + 0.5 * rho * (u * u + v * v)
    return torch.stack((rho, rho * u, rho * v, energy), dim=-1)


def conservative_to_primitive(conservative: torch.Tensor, gamma: float) -> torch.Tensor:
    """Convert conservative variables to primitives without applying floors."""

    rho = conservative[..., 0]
    u = conservative[..., 1] / rho
    v = conservative[..., 2] / rho
    kinetic = 0.5 * rho * (u * u + v * v)
    pressure = (gamma - 1.0) * (conservative[..., 3] - kinetic)
    return torch.stack((rho, u, v, pressure), dim=-1)


def state_is_admissible(
    conservative: torch.Tensor,
    config: ShockVortexFVConfig,
) -> bool:
    """Return whether a raw cell state is finite with positive density/pressure."""

    if not bool(torch.isfinite(conservative).all()):
        return False
    primitive = conservative_to_primitive(conservative, config.gamma)
    return bool(
        torch.isfinite(primitive).all()
        and torch.all(primitive[..., 0] > config.density_admissibility_threshold)
        and torch.all(primitive[..., 3] > config.pressure_admissibility_threshold)
    )


def _fill_primitive_ghosts(
    primitive: torch.Tensor,
    config: ShockVortexFVConfig,
) -> torch.Tensor:
    """Apply linear x extrapolation and y symmetry in primitive variables."""

    ng = config.ghost_cells
    ny, nx, variables = primitive.shape
    ghosted = torch.empty(
        (ny + 2 * ng, nx + 2 * ng, variables),
        dtype=primitive.dtype,
        device=primitive.device,
    )
    ghosted[ng : ng + ny, ng : ng + nx] = primitive
    for offset in range(1, ng + 1):
        ghosted[ng : ng + ny, ng - offset] = (offset + 1) * primitive[
            :, 0
        ] - offset * primitive[:, 1]
        ghosted[ng : ng + ny, ng + nx - 1 + offset] = (offset + 1) * primitive[
            :, -1
        ] - offset * primitive[:, -2]
    for offset in range(1, ng + 1):
        bottom = ghosted[ng + offset - 1].clone()
        top = ghosted[ng + ny - offset].clone()
        bottom[..., 2] = -bottom[..., 2]
        top[..., 2] = -top[..., 2]
        ghosted[ng - offset] = bottom
        ghosted[ng + ny - 1 + offset] = top
    return ghosted


def _weno5_face_states(
    jm2: torch.Tensor,
    jm1: torch.Tensor,
    j0: torch.Tensor,
    jp1: torch.Tensor,
    jp2: torch.Tensor,
    jp3: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fifth-order WENO-JS primitive reconstruction from six aligned slices."""

    q0 = (2.0 * jm2 - 7.0 * jm1 + 11.0 * j0) / 6.0
    q1 = (-jm1 + 5.0 * j0 + 2.0 * jp1) / 6.0
    q2 = (2.0 * j0 + 5.0 * jp1 - jp2) / 6.0
    beta0 = (13.0 / 12.0) * (jm2 - 2.0 * jm1 + j0) ** 2 + 0.25 * (
        jm2 - 4.0 * jm1 + 3.0 * j0
    ) ** 2
    beta1 = (13.0 / 12.0) * (jm1 - 2.0 * j0 + jp1) ** 2 + 0.25 * (jm1 - jp1) ** 2
    beta2 = (13.0 / 12.0) * (j0 - 2.0 * jp1 + jp2) ** 2 + 0.25 * (
        3.0 * j0 - 4.0 * jp1 + jp2
    ) ** 2
    alpha0 = 0.1 / (epsilon + beta0) ** 2
    alpha1 = 0.6 / (epsilon + beta1) ** 2
    alpha2 = 0.3 / (epsilon + beta2) ** 2
    alpha_sum = alpha0 + alpha1 + alpha2
    left = (alpha0 * q0 + alpha1 * q1 + alpha2 * q2) / alpha_sum

    q0r = (2.0 * jp3 - 7.0 * jp2 + 11.0 * jp1) / 6.0
    q1r = (-jp2 + 5.0 * jp1 + 2.0 * j0) / 6.0
    q2r = (2.0 * jp1 + 5.0 * j0 - jm1) / 6.0
    beta0r = (13.0 / 12.0) * (jp3 - 2.0 * jp2 + jp1) ** 2 + 0.25 * (
        jp3 - 4.0 * jp2 + 3.0 * jp1
    ) ** 2
    beta1r = (13.0 / 12.0) * (jp2 - 2.0 * jp1 + j0) ** 2 + 0.25 * (jp2 - j0) ** 2
    beta2r = (13.0 / 12.0) * (jp1 - 2.0 * j0 + jm1) ** 2 + 0.25 * (
        3.0 * jp1 - 4.0 * j0 + jm1
    ) ** 2
    alpha0r = 0.1 / (epsilon + beta0r) ** 2
    alpha1r = 0.6 / (epsilon + beta1r) ** 2
    alpha2r = 0.3 / (epsilon + beta2r) ** 2
    alpha_sum_r = alpha0r + alpha1r + alpha2r
    right = (alpha0r * q0r + alpha1r * q1r + alpha2r * q2r) / alpha_sum_r
    return left, right


def _primitive_face_is_admissible(
    primitive: torch.Tensor,
    config: ShockVortexFVConfig,
) -> torch.Tensor:
    return (
        torch.isfinite(primitive).all(dim=-1)
        & (primitive[..., 0] > config.density_admissibility_threshold)
        & (primitive[..., 3] > config.pressure_admissibility_threshold)
    )


def _fallback_invalid_face_states(
    left: torch.Tensor,
    right: torch.Tensor,
    left_cell: torch.Tensor,
    right_cell: torch.Tensor,
    config: ShockVortexFVConfig,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    valid_left = _primitive_face_is_admissible(left, config)
    valid_right = _primitive_face_is_admissible(right, config)
    left = torch.where(valid_left.unsqueeze(-1), left, left_cell)
    right = torch.where(valid_right.unsqueeze(-1), right, right_cell)
    fallback_count = int(torch.count_nonzero(~(valid_left & valid_right)).item())
    return left, right, fallback_count


def _safe_signed(value: torch.Tensor, epsilon: float = 1.0e-14) -> torch.Tensor:
    replacement = torch.where(
        value >= 0.0,
        torch.full_like(value, epsilon),
        torch.full_like(value, -epsilon),
    )
    return torch.where(torch.abs(value) < epsilon, replacement, value)


def hllc_flux(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    gamma: float,
    axis: int,
) -> torch.Tensor:
    """Return the 2D HLLC flux in the positive coordinate-axis direction."""

    if axis not in (0, 1):
        raise ValueError("axis must be zero (x) or one (y)")
    rho_l, u_l, v_l, p_l = left.unbind(dim=-1)
    rho_r, u_r, v_r, p_r = right.unbind(dim=-1)
    if axis == 0:
        normal_l, tangent_l = u_l, v_l
        normal_r, tangent_r = u_r, v_r
    else:
        normal_l, tangent_l = v_l, u_l
        normal_r, tangent_r = v_r, u_r

    energy_l = p_l / (gamma - 1.0) + 0.5 * rho_l * (normal_l**2 + tangent_l**2)
    energy_r = p_r / (gamma - 1.0) + 0.5 * rho_r * (normal_r**2 + tangent_r**2)
    conservative_l = torch.stack(
        (rho_l, rho_l * normal_l, rho_l * tangent_l, energy_l), dim=-1
    )
    conservative_r = torch.stack(
        (rho_r, rho_r * normal_r, rho_r * tangent_r, energy_r), dim=-1
    )
    flux_l = torch.stack(
        (
            rho_l * normal_l,
            rho_l * normal_l**2 + p_l,
            rho_l * normal_l * tangent_l,
            normal_l * (energy_l + p_l),
        ),
        dim=-1,
    )
    flux_r = torch.stack(
        (
            rho_r * normal_r,
            rho_r * normal_r**2 + p_r,
            rho_r * normal_r * tangent_r,
            normal_r * (energy_r + p_r),
        ),
        dim=-1,
    )
    sound_l = torch.sqrt(gamma * p_l / rho_l)
    sound_r = torch.sqrt(gamma * p_r / rho_r)
    speed_l = torch.minimum(normal_l - sound_l, normal_r - sound_r)
    speed_r = torch.maximum(normal_l + sound_l, normal_r + sound_r)
    denominator = _safe_signed(
        rho_l * (speed_l - normal_l) - rho_r * (speed_r - normal_r)
    )
    speed_m = (
        p_r
        - p_l
        + rho_l * normal_l * (speed_l - normal_l)
        - rho_r * normal_r * (speed_r - normal_r)
    ) / denominator
    rho_star_l = rho_l * (speed_l - normal_l) / _safe_signed(speed_l - speed_m)
    rho_star_r = rho_r * (speed_r - normal_r) / _safe_signed(speed_r - speed_m)
    energy_star_l = rho_star_l * (
        energy_l / rho_l
        + (speed_m - normal_l)
        * (speed_m + p_l / _safe_signed(rho_l * (speed_l - normal_l)))
    )
    energy_star_r = rho_star_r * (
        energy_r / rho_r
        + (speed_m - normal_r)
        * (speed_m + p_r / _safe_signed(rho_r * (speed_r - normal_r)))
    )
    star_l = torch.stack(
        (rho_star_l, rho_star_l * speed_m, rho_star_l * tangent_l, energy_star_l),
        dim=-1,
    )
    star_r = torch.stack(
        (rho_star_r, rho_star_r * speed_m, rho_star_r * tangent_r, energy_star_r),
        dim=-1,
    )
    flux_star_l = flux_l + speed_l.unsqueeze(-1) * (star_l - conservative_l)
    flux_star_r = flux_r + speed_r.unsqueeze(-1) * (star_r - conservative_r)
    local_flux = torch.where(
        (speed_l >= 0.0).unsqueeze(-1),
        flux_l,
        torch.where(
            (speed_m >= 0.0).unsqueeze(-1),
            flux_star_l,
            torch.where((speed_r > 0.0).unsqueeze(-1), flux_star_r, flux_r),
        ),
    )
    if axis == 0:
        flux = local_flux
    else:
        flux = torch.stack(
            (
                local_flux[..., 0],
                local_flux[..., 2],
                local_flux[..., 1],
                local_flux[..., 3],
            ),
            dim=-1,
        )
    if not bool(torch.isfinite(flux).all()):
        raise FloatingPointError("HLLC produced a non-finite face flux")
    return flux


def reference_rhs_and_fluxes(
    conservative: torch.Tensor,
    config: ShockVortexFVConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Evaluate the semi-discrete FV operator and all physical face fluxes."""

    if conservative.shape != (config.ny, config.nx, 4):
        raise ValueError("conservative state shape does not match config")
    primitive = conservative_to_primitive(conservative, config.gamma)
    ghosted = _fill_primitive_ghosts(primitive, config)
    ng = config.ghost_cells
    ny = config.ny
    nx = config.nx

    x_rows = slice(ng, ng + ny)
    x_start = ng - 1
    x_stop = x_start + nx + 1
    left_x, right_x = _weno5_face_states(
        ghosted[x_rows, x_start - 2 : x_stop - 2],
        ghosted[x_rows, x_start - 1 : x_stop - 1],
        ghosted[x_rows, x_start:x_stop],
        ghosted[x_rows, x_start + 1 : x_stop + 1],
        ghosted[x_rows, x_start + 2 : x_stop + 2],
        ghosted[x_rows, x_start + 3 : x_stop + 3],
        config.weno_epsilon,
    )
    left_x, right_x, fallback_x = _fallback_invalid_face_states(
        left_x,
        right_x,
        ghosted[x_rows, x_start:x_stop],
        ghosted[x_rows, x_start + 1 : x_stop + 1],
        config,
    )
    flux_x = hllc_flux(left_x, right_x, gamma=config.gamma, axis=0)

    y_columns = slice(ng, ng + nx)
    y_start = ng - 1
    y_stop = y_start + ny + 1
    left_y, right_y = _weno5_face_states(
        ghosted[y_start - 2 : y_stop - 2, y_columns],
        ghosted[y_start - 1 : y_stop - 1, y_columns],
        ghosted[y_start:y_stop, y_columns],
        ghosted[y_start + 1 : y_stop + 1, y_columns],
        ghosted[y_start + 2 : y_stop + 2, y_columns],
        ghosted[y_start + 3 : y_stop + 3, y_columns],
        config.weno_epsilon,
    )
    left_y, right_y, fallback_y = _fallback_invalid_face_states(
        left_y,
        right_y,
        ghosted[y_start:y_stop, y_columns],
        ghosted[y_start + 1 : y_stop + 1, y_columns],
        config,
    )
    bottom_inside = right_y[0].clone()
    bottom_mirror = bottom_inside.clone()
    bottom_mirror[..., 2] = -bottom_mirror[..., 2]
    left_y[0] = bottom_mirror
    right_y[0] = bottom_inside
    top_inside = left_y[-1].clone()
    top_mirror = top_inside.clone()
    top_mirror[..., 2] = -top_mirror[..., 2]
    left_y[-1] = top_inside
    right_y[-1] = top_mirror
    flux_y = hllc_flux(left_y, right_y, gamma=config.gamma, axis=1)

    rhs = -(
        (flux_x[:, 1:] - flux_x[:, :-1]) / config.dx
        + (flux_y[1:] - flux_y[:-1]) / config.dy
    )
    return rhs, flux_x, flux_y, fallback_x + fallback_y


def stable_time_step(
    conservative: torch.Tensor,
    config: ShockVortexFVConfig,
) -> float:
    """Return the unsplit multidimensional CFL step for an admissible state."""

    primitive = conservative_to_primitive(conservative, config.gamma)
    rho, u, v, pressure = primitive.unbind(dim=-1)
    sound = torch.sqrt(config.gamma * pressure / rho)
    inverse_step = (torch.abs(u) + sound) / config.dx + (
        torch.abs(v) + sound
    ) / config.dy
    maximum = float(torch.max(inverse_step).item())
    if not np.isfinite(maximum) or maximum <= 0.0:
        raise FloatingPointError("invalid characteristic speed for CFL step")
    return config.cfl / maximum


def ssprk3_step(
    conservative: torch.Tensor,
    dt: float,
    config: ShockVortexFVConfig,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, int, bool]:
    """Attempt one WENO5--HLLC SSPRK3 step and its equivalent face flux."""

    rhs0, flux_x0, flux_y0, fallback0 = reference_rhs_and_fluxes(conservative, config)
    stage1 = conservative + dt * rhs0
    if not state_is_admissible(stage1, config):
        return stage1, None, None, fallback0, False
    rhs1, flux_x1, flux_y1, fallback1 = reference_rhs_and_fluxes(stage1, config)
    stage2 = 0.75 * conservative + 0.25 * (stage1 + dt * rhs1)
    if not state_is_admissible(stage2, config):
        return stage2, None, None, fallback0 + fallback1, False
    rhs2, flux_x2, flux_y2, fallback2 = reference_rhs_and_fluxes(stage2, config)
    updated = (1.0 / 3.0) * conservative + (2.0 / 3.0) * (stage2 + dt * rhs2)
    if not state_is_admissible(updated, config):
        return updated, None, None, fallback0 + fallback1 + fallback2, False
    combined_x = (1.0 / 6.0) * flux_x0 + (1.0 / 6.0) * flux_x1 + (2.0 / 3.0) * flux_x2
    combined_y = (1.0 / 6.0) * flux_y0 + (1.0 / 6.0) * flux_y1 + (2.0 / 3.0) * flux_y2
    return (
        updated,
        combined_x,
        combined_y,
        fallback0 + fallback1 + fallback2,
        True,
    )


def restrict_cell_averages(
    conservative: torch.Tensor,
    config: ShockVortexFVConfig,
) -> torch.Tensor:
    """Conservatively restrict fine cell averages to the declared coarse mesh."""

    if conservative.shape != (config.ny, config.nx, 4):
        raise ValueError("conservative state shape does not match config")
    return conservative.reshape(
        config.coarse_ny,
        config.restriction_y,
        config.coarse_nx,
        config.restriction_x,
        4,
    ).mean(dim=(1, 3))


def restrict_owner_oriented_face_impulse(
    flux_x: torch.Tensor,
    flux_y: torch.Tensor,
    dt: float,
    config: ShockVortexFVConfig,
) -> torch.Tensor:
    """Aggregate one accepted fine-grid RK flux into coarse physical impulses.

    ``flux_x`` and ``flux_y`` point in the positive coordinate directions.  The
    returned flattened faces follow :func:`make_structured_fv_geometry` and are
    oriented outward at boundaries and from owner to neighbor in the interior.
    """

    if flux_x.shape != (config.ny, config.nx + 1, 4):
        raise ValueError("flux_x shape does not match config")
    if flux_y.shape != (config.ny + 1, config.nx, 4):
        raise ValueError("flux_y shape does not match config")
    impulse_x = float(dt) * config.dy * flux_x
    impulse_y = float(dt) * config.dx * flux_y
    impulse_x = impulse_x.clone()
    impulse_y = impulse_y.clone()
    impulse_x[:, 0] = -impulse_x[:, 0]
    impulse_y[0] = -impulse_y[0]

    x_indices = torch.arange(
        0,
        config.nx + 1,
        config.restriction_x,
        device=flux_x.device,
    )
    y_indices = torch.arange(
        0,
        config.ny + 1,
        config.restriction_y,
        device=flux_y.device,
    )
    vertical = impulse_x.reshape(
        config.coarse_ny,
        config.restriction_y,
        config.nx + 1,
        4,
    ).sum(dim=1)[:, x_indices]
    horizontal = (
        impulse_y[y_indices]
        .reshape(
            config.coarse_ny + 1,
            config.coarse_nx,
            config.restriction_x,
            4,
        )
        .sum(dim=2)
    )
    return torch.cat((vertical.reshape(-1, 4), horizontal.reshape(-1, 4)), dim=0)


def decode_owner_oriented_face_impulse(
    face_impulse: np.ndarray,
    geometry: StructuredFVGeometry,
) -> np.ndarray:
    """Decode owner-oriented physical impulses into coarse cell-state changes."""

    impulse = np.asarray(face_impulse, dtype=np.float64)
    if impulse.ndim != 2 or impulse.shape != (geometry.face_owner.size, 4):
        raise ValueError("face_impulse must have shape [coarse_faces, 4]")
    change = np.zeros((geometry.cell_volume.size, 4), dtype=np.float64)
    np.add.at(
        change,
        geometry.face_owner,
        -impulse / geometry.cell_volume[geometry.face_owner, None],
    )
    interior = geometry.face_neighbor >= 0
    np.add.at(
        change,
        geometry.face_neighbor[interior],
        impulse[interior]
        / geometry.cell_volume[geometry.face_neighbor[interior], None],
    )
    return change


def _closure_metrics(
    before: np.ndarray,
    after: np.ndarray,
    face_impulse: np.ndarray,
    geometry: StructuredFVGeometry,
) -> tuple[float, float, np.ndarray]:
    decoded = decode_owner_oriented_face_impulse(face_impulse, geometry)
    residual = after - before - decoded
    integrated_residual = geometry.cell_volume[:, None] * residual
    integrated_change = geometry.cell_volume[:, None] * (after - before)
    maximum = float(np.max(np.abs(integrated_residual), initial=0.0))
    denominator = max(float(np.linalg.norm(integrated_change)), 1.0e-30)
    relative = float(np.linalg.norm(integrated_residual) / denominator)
    boundary = geometry.face_neighbor < 0
    boundary_exchange = np.sum(face_impulse[boundary], axis=0)
    return maximum, relative, boundary_exchange


def run_shock_vortex_reference(
    config: ShockVortexFVConfig,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> ShockVortexReferenceResult:
    """Integrate the canonical case and return a conservative coarse artifact."""

    config.validated()
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("reference dtype must be torch.float32 or torch.float64")
    resolved_device = torch.device(device)
    geometry = make_structured_fv_geometry(config)
    num_faces = geometry.face_owner.size
    states: list[np.ndarray] = []
    interval_impulses: list[np.ndarray] = []
    accepted_times: list[float] = []
    accepted_dt: list[float] = []
    accepted_intervals: list[int] = []
    accepted_rejections: list[int] = []
    accepted_fallbacks: list[int] = []
    interval_rejections: list[int] = []
    interval_fallbacks: list[int] = []
    closure_max_abs: list[float] = []
    closure_relative: list[float] = []
    boundary_exchange: list[np.ndarray] = []

    with torch.inference_mode():
        conservative = shock_vortex_initial_cell_averages(
            config,
            device=resolved_device,
            dtype=dtype,
        )
        certified_initial = shock_vortex_initial_cell_averages(
            config,
            device=resolved_device,
            dtype=dtype,
            quadrature_order=2 * config.initial_quadrature_order,
        )
        if not state_is_admissible(conservative, config):
            raise ValueError("canonical initial state is not admissible")
        quadrature_delta = conservative - certified_initial
        initial_quadrature_max_abs = float(
            torch.max(torch.abs(quadrature_delta)).item()
        )
        initial_quadrature_relative_l2 = float(
            torch.linalg.vector_norm(quadrature_delta).item()
            / max(
                float(torch.linalg.vector_norm(certified_initial).item()),
                1.0e-30,
            )
        )
        initial_coarse = restrict_cell_averages(conservative, config)
        certified_initial_coarse = restrict_cell_averages(certified_initial, config)
        restricted_quadrature_delta = initial_coarse - certified_initial_coarse
        restricted_initial_quadrature_max_abs = float(
            torch.max(torch.abs(restricted_quadrature_delta)).item()
        )
        restricted_initial_quadrature_relative_l2 = float(
            torch.linalg.vector_norm(restricted_quadrature_delta).item()
            / max(
                float(torch.linalg.vector_norm(certified_initial_coarse).item()),
                1.0e-30,
            )
        )
        states.append(initial_coarse.reshape(-1, 4).double().cpu().numpy())
        primitive = conservative_to_primitive(conservative, config.gamma)
        minimum_density = float(torch.min(primitive[..., 0]).item())
        minimum_pressure = float(torch.min(primitive[..., 3]).item())
        current_time = 0.0

        for interval_index, target_time in enumerate(config.output_times[1:]):
            interval_impulse = torch.zeros(
                (num_faces, 4),
                dtype=torch.float64,
                device=resolved_device,
            )
            rejected_in_interval = 0
            fallbacks_in_interval = 0
            while current_time < target_time - 1.0e-14:
                trial_dt = min(
                    stable_time_step(conservative, config),
                    float(target_time - current_time),
                )
                step_rejections = 0
                attempt_fallbacks = 0
                for _ in range(config.max_step_retries):
                    (
                        candidate,
                        flux_x,
                        flux_y,
                        fallback_count,
                        accepted,
                    ) = ssprk3_step(conservative, trial_dt, config)
                    attempt_fallbacks += fallback_count
                    if accepted:
                        break
                    step_rejections += 1
                    rejected_in_interval += 1
                    trial_dt *= 0.5
                else:
                    raise RuntimeError(
                        "reference step remained inadmissible after "
                        f"{config.max_step_retries} retries at t={current_time:.16g}"
                    )
                if flux_x is None or flux_y is None:
                    raise AssertionError("accepted reference step has no face flux")
                step_impulse = restrict_owner_oriented_face_impulse(
                    flux_x, flux_y, trial_dt, config
                )
                interval_impulse += step_impulse.to(dtype=torch.float64)
                conservative = candidate
                current_time += trial_dt
                if abs(current_time - target_time) <= 5.0e-14:
                    current_time = float(target_time)
                fallbacks_in_interval += attempt_fallbacks
                accepted_times.append(current_time)
                accepted_dt.append(trial_dt)
                accepted_intervals.append(interval_index)
                accepted_rejections.append(step_rejections)
                accepted_fallbacks.append(attempt_fallbacks)

                primitive = conservative_to_primitive(conservative, config.gamma)
                minimum_density = min(
                    minimum_density, float(torch.min(primitive[..., 0]).item())
                )
                minimum_pressure = min(
                    minimum_pressure, float(torch.min(primitive[..., 3]).item())
                )

            coarse = restrict_cell_averages(conservative, config)
            coarse_numpy = coarse.reshape(-1, 4).double().cpu().numpy()
            impulse_numpy = interval_impulse.cpu().numpy()
            maximum, relative, exchange = _closure_metrics(
                states[-1], coarse_numpy, impulse_numpy, geometry
            )
            states.append(coarse_numpy)
            interval_impulses.append(impulse_numpy)
            interval_rejections.append(rejected_in_interval)
            interval_fallbacks.append(fallbacks_in_interval)
            closure_max_abs.append(maximum)
            closure_relative.append(relative)
            boundary_exchange.append(exchange)

    return ShockVortexReferenceResult(
        config=config,
        geometry=geometry,
        times=np.asarray(config.output_times, dtype=np.float64),
        states=np.stack(states, axis=0),
        face_impulses=np.stack(interval_impulses, axis=0),
        accepted_step_times=np.asarray(accepted_times, dtype=np.float64),
        accepted_step_dt=np.asarray(accepted_dt, dtype=np.float64),
        accepted_step_interval=np.asarray(accepted_intervals, dtype=np.int64),
        accepted_step_rejections=np.asarray(accepted_rejections, dtype=np.int64),
        accepted_step_face_fallbacks=np.asarray(accepted_fallbacks, dtype=np.int64),
        interval_rejection_count=np.asarray(interval_rejections, dtype=np.int64),
        interval_face_fallback_count=np.asarray(interval_fallbacks, dtype=np.int64),
        interval_closure_max_abs=np.asarray(closure_max_abs, dtype=np.float64),
        interval_closure_relative_l2=np.asarray(closure_relative, dtype=np.float64),
        interval_boundary_exchange=np.stack(boundary_exchange, axis=0),
        initial_quadrature_max_abs=initial_quadrature_max_abs,
        initial_quadrature_relative_l2=initial_quadrature_relative_l2,
        restricted_initial_quadrature_max_abs=(restricted_initial_quadrature_max_abs),
        restricted_initial_quadrature_relative_l2=(
            restricted_initial_quadrature_relative_l2
        ),
        minimum_density=minimum_density,
        minimum_pressure=minimum_pressure,
    )


def reference_contract_checks(
    result: ShockVortexReferenceResult,
    *,
    closure_relative_tolerance: float,
    initial_quadrature_tolerance: float = 1.0e-8,
) -> dict[str, bool]:
    """Return the non-negotiable geometry, time, admissibility, and closure gates."""

    geometry = result.geometry
    config = result.config
    accepted = result.accepted_step_times
    num_cells = geometry.cell_volume.size
    num_faces = geometry.face_owner.size
    num_intervals = len(config.output_times) - 1
    interval_end_times = np.asarray(config.output_times[1:], dtype=np.float64)
    reached = np.array(
        [
            np.any(np.isclose(accepted, time, rtol=0.0, atol=5.0e-14))
            for time in interval_end_times
        ]
    )
    integrated_change = np.einsum(
        "c,tck->tk",
        geometry.cell_volume,
        result.states[1:] - result.states[:-1],
    )
    boundary_balance = integrated_change + result.interval_boundary_exchange
    interior = geometry.face_neighbor >= 0
    incident_count = np.bincount(geometry.face_owner, minlength=num_cells)
    incident_count += np.bincount(geometry.face_neighbor[interior], minlength=num_cells)
    area_vector_closure = np.zeros((num_cells, 2), dtype=np.float64)
    oriented_area = geometry.face_measure[:, None] * geometry.face_normal
    np.add.at(area_vector_closure, geometry.face_owner, oriented_area)
    np.add.at(
        area_vector_closure,
        geometry.face_neighbor[interior],
        -oriented_area[interior],
    )
    expected_face_center = geometry.cell_centers[geometry.face_owner].copy()
    expected_face_center[interior] = 0.5 * (
        geometry.cell_centers[geometry.face_owner[interior]]
        + geometry.cell_centers[geometry.face_neighbor[interior]]
    )
    boundary = ~interior
    half_width = np.where(
        geometry.face_axis == 0,
        0.5 * config.dx * config.restriction_x,
        0.5 * config.dy * config.restriction_y,
    )
    expected_face_center[boundary] += (
        half_width[boundary, None] * geometry.face_normal[boundary]
    )
    expected_measure = np.where(
        geometry.face_axis == 0,
        config.dy * config.restriction_y,
        config.dx * config.restriction_x,
    )
    expected_normal = np.zeros_like(geometry.face_normal)
    interior_displacement = (
        geometry.cell_centers[geometry.face_neighbor[interior]]
        - geometry.cell_centers[geometry.face_owner[interior]]
    )
    interior_distance = np.linalg.norm(interior_displacement, axis=1)
    if np.all(interior_distance > 0.0):
        expected_normal[interior] = interior_displacement / interior_distance[:, None]
    expected_axis = np.full(num_faces, -1, dtype=np.int8)
    if np.any(interior):
        expected_axis[interior] = np.argmax(
            np.abs(interior_displacement), axis=1
        ).astype(np.int8)
    expected_boundary_tag = np.zeros(num_faces, dtype=np.int8)
    boundary_contracts = (
        (1, 0, config.x_min, (-1.0, 0.0)),
        (2, 0, config.x_max, (1.0, 0.0)),
        (3, 1, config.y_min, (0.0, -1.0)),
        (4, 1, config.y_max, (0.0, 1.0)),
    )
    boundary_location_covered = np.zeros(num_faces, dtype=bool)
    for tag, axis, coordinate, normal in boundary_contracts:
        selected = (
            boundary
            & (geometry.face_axis == axis)
            & np.isclose(
                geometry.face_centers[:, axis],
                coordinate,
                rtol=0.0,
                atol=1.0e-14,
            )
        )
        expected_boundary_tag[selected] = tag
        expected_axis[selected] = axis
        expected_normal[selected] = normal
        boundary_location_covered |= selected
    normal_orientation_consistent = bool(
        np.all(interior_distance > 0.0)
        and np.array_equal(geometry.face_axis, expected_axis)
        and np.allclose(
            geometry.face_normal,
            expected_normal,
            rtol=0.0,
            atol=1.0e-14,
        )
    )
    boundary_tags_consistent = bool(
        np.all(boundary_location_covered[boundary])
        and not np.any(boundary_location_covered[interior])
        and np.array_equal(geometry.face_boundary_tag, expected_boundary_tag)
        and np.count_nonzero(geometry.face_boundary_tag == 1) == config.coarse_ny
        and np.count_nonzero(geometry.face_boundary_tag == 2) == config.coarse_ny
        and np.count_nonzero(geometry.face_boundary_tag == 3) == config.coarse_nx
        and np.count_nonzero(geometry.face_boundary_tag == 4) == config.coarse_nx
    )
    accepted_shapes_match = all(
        array.shape == accepted.shape
        for array in (
            result.accepted_step_dt,
            result.accepted_step_interval,
            result.accepted_step_rejections,
            result.accepted_step_face_fallbacks,
        )
    )
    interval_shapes_match = all(
        array.shape[0] == num_intervals
        for array in (
            result.interval_rejection_count,
            result.interval_face_fallback_count,
            result.interval_closure_max_abs,
            result.interval_closure_relative_l2,
            result.interval_boundary_exchange,
        )
    )
    accepted_time_increments = (
        np.diff(np.concatenate(([0.0], accepted)))
        if accepted.size
        else np.empty(0, dtype=np.float64)
    )
    interval_dt_sums = np.asarray(
        [
            np.sum(result.accepted_step_dt[result.accepted_step_interval == index])
            for index in range(num_intervals)
        ],
        dtype=np.float64,
    )
    checks = {
        "initial_cell_average_quadrature_certified": bool(
            result.initial_quadrature_relative_l2 <= initial_quadrature_tolerance
        ),
        "restricted_initial_cell_average_quadrature_certified": bool(
            result.restricted_initial_quadrature_relative_l2
            <= initial_quadrature_tolerance
        ),
        "cell_volumes_positive": bool(np.all(geometry.cell_volume > 0.0)),
        "face_measures_positive": bool(np.all(geometry.face_measure > 0.0)),
        "face_normals_unit": bool(
            np.allclose(
                np.linalg.norm(geometry.face_normal, axis=1),
                1.0,
                rtol=0.0,
                atol=1.0e-14,
            )
        ),
        "oriented_connectivity_valid": bool(
            np.all(geometry.face_owner >= 0)
            and np.all(geometry.face_owner < geometry.cell_volume.size)
            and np.all(geometry.face_neighbor < geometry.cell_volume.size)
            and np.all(geometry.face_neighbor >= -1)
            and np.all(geometry.face_owner != geometry.face_neighbor)
        ),
        "face_normal_orientation_consistent": normal_orientation_consistent,
        "boundary_tags_and_locations_consistent": boundary_tags_consistent,
        "structured_face_geometry_consistent": bool(
            np.all(np.isin(geometry.face_axis, (0, 1)))
            and np.allclose(
                geometry.face_measure,
                expected_measure,
                rtol=0.0,
                atol=1.0e-14,
            )
            and np.allclose(
                geometry.face_centers,
                expected_face_center,
                rtol=0.0,
                atol=1.0e-14,
            )
        ),
        "cell_incidence_complete": bool(np.all(incident_count == 4)),
        "cell_area_vector_closure": bool(
            np.allclose(area_vector_closure, 0.0, rtol=0.0, atol=1.0e-14)
        ),
        "artifact_array_shapes_consistent": bool(
            result.states.shape == (num_intervals + 1, num_cells, 4)
            and result.face_impulses.shape == (num_intervals, num_faces, 4)
            and result.times.shape == (num_intervals + 1,)
            and accepted_shapes_match
            and interval_shapes_match
        ),
        "accepted_times_strictly_increasing": bool(
            accepted.size > 0 and np.all(np.diff(accepted) > 0.0)
        ),
        "accepted_dt_matches_time_increments": bool(
            accepted_shapes_match
            and np.all(result.accepted_step_dt > 0.0)
            and np.allclose(
                accepted_time_increments,
                result.accepted_step_dt,
                rtol=0.0,
                atol=5.0e-14,
            )
        ),
        "accepted_intervals_and_dt_sums_valid": bool(
            accepted_shapes_match
            and np.all(result.accepted_step_interval >= 0)
            and np.all(result.accepted_step_interval < num_intervals)
            and np.all(np.diff(result.accepted_step_interval) >= 0)
            and np.allclose(
                interval_dt_sums,
                np.diff(np.asarray(config.output_times, dtype=np.float64)),
                rtol=0.0,
                atol=5.0e-14,
            )
        ),
        "all_saved_times_reached_exactly": bool(np.all(reached)),
        "raw_states_finite": bool(np.all(np.isfinite(result.states))),
        "raw_density_pressure_admissible": bool(
            result.minimum_density > config.density_admissibility_threshold
            and result.minimum_pressure > config.pressure_admissibility_threshold
        ),
        "interval_impulse_closure": bool(
            np.max(result.interval_closure_relative_l2, initial=0.0)
            <= closure_relative_tolerance
        ),
        "boundary_exchange_closure": bool(
            np.max(np.abs(boundary_balance), initial=0.0)
            <= max(closure_relative_tolerance, 1.0e-12)
        ),
    }
    if set(checks) != REFERENCE_CONTRACT_CHECK_KEYS:
        raise AssertionError("reference contract-check key set drifted")
    return checks
