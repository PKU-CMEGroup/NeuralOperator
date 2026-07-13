"""1D Euler helpers for solver-facing target diagnostics.

The collaborator dataset stores primitive variables ``[rho, u, p]`` on cell
centers. Training adapters should update conservative variables
``[rho, rho*u, E]`` and convert back to primitives for metrics/losses.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from utility.time_dependent_no.fv import FiniteVolumeGeometry, finite_volume_update


EULER1D_PRIMITIVE_NAMES: tuple[str, ...] = ("rho", "u", "p")
EULER1D_CONSERVATIVE_NAMES: tuple[str, ...] = ("rho", "rho_u", "energy")


@dataclass(frozen=True)
class Euler1DBatch:
    """Batched 1D Euler state plus finite-volume geometry."""

    current_primitive: torch.Tensor
    geometry: FiniteVolumeGeometry
    dt: torch.Tensor
    target_primitive: torch.Tensor | None = None
    gamma: float = 1.4
    left_boundary_primitive: torch.Tensor | None = None
    right_initial_primitive: torch.Tensor | None = None
    right_boundary_primitive: torch.Tensor | None = None

    def to(self, device: torch.device | str) -> "Euler1DBatch":
        return Euler1DBatch(
            current_primitive=self.current_primitive.to(device),
            geometry=self.geometry.to(device),
            dt=self.dt.to(device),
            target_primitive=None
            if self.target_primitive is None
            else self.target_primitive.to(device),
            gamma=self.gamma,
            left_boundary_primitive=None
            if self.left_boundary_primitive is None
            else self.left_boundary_primitive.to(device),
            right_initial_primitive=None
            if self.right_initial_primitive is None
            else self.right_initial_primitive.to(device),
            right_boundary_primitive=None
            if self.right_boundary_primitive is None
            else self.right_boundary_primitive.to(device),
        )

    @property
    def current_conservative(self) -> torch.Tensor:
        return primitive_to_conservative(self.current_primitive, gamma=self.gamma)

    @property
    def target_conservative(self) -> torch.Tensor | None:
        if self.target_primitive is None:
            return None
        return primitive_to_conservative(self.target_primitive, gamma=self.gamma)


def primitive_to_conservative(
    primitive: torch.Tensor, gamma: float = 1.4
) -> torch.Tensor:
    """Convert primitive ``[rho, u, p]`` to conservative variables."""

    if primitive.shape[-1] != 3:
        raise ValueError("1D Euler primitive state must have last dimension 3")
    rho, velocity, pressure = primitive.unbind(dim=-1)
    momentum = rho * velocity
    energy = pressure / (gamma - 1.0) + 0.5 * rho * velocity.square()
    return torch.stack((rho, momentum, energy), dim=-1)


def conservative_to_primitive(
    conservative: torch.Tensor,
    gamma: float = 1.4,
    rho_floor: float | None = None,
    pressure_floor: float | None = None,
) -> torch.Tensor:
    """Convert conservative ``[rho, rho*u, E]`` to primitive variables."""

    if conservative.shape[-1] != 3:
        raise ValueError("1D Euler conservative state must have last dimension 3")
    rho, momentum, energy = conservative.unbind(dim=-1)
    if rho_floor is not None:
        rho = rho.clamp_min(rho_floor)
    velocity = momentum / rho
    pressure = (gamma - 1.0) * (energy - 0.5 * momentum * velocity)
    if pressure_floor is not None:
        pressure = pressure.clamp_min(pressure_floor)
    return torch.stack((rho, velocity, pressure), dim=-1)


def decode_primitive(
    raw: torch.Tensor,
    *,
    positive_transform: str = "none",
    rho_floor: float = 1.0e-6,
    pressure_floor: float = 1.0e-6,
) -> torch.Tensor:
    """Decode network output as primitive state with optional positivity."""

    if raw.shape[-1] != 3:
        raise ValueError("raw primitive state must have last dimension 3")
    rho_raw, velocity, pressure_raw = raw.unbind(dim=-1)
    if positive_transform == "none":
        rho = rho_raw
        pressure = pressure_raw
    elif positive_transform == "softplus":
        rho = F.softplus(rho_raw) + rho_floor
        pressure = F.softplus(pressure_raw) + pressure_floor
    elif positive_transform == "exp":
        rho = torch.exp(rho_raw) + rho_floor
        pressure = torch.exp(pressure_raw) + pressure_floor
    else:
        raise ValueError(f"unsupported positive_transform: {positive_transform}")
    return torch.stack((rho, velocity, pressure), dim=-1)


def euler_flux_from_primitive(
    primitive: torch.Tensor, gamma: float = 1.4
) -> torch.Tensor:
    """Physical 1D Euler flux in the positive x direction."""

    rho, velocity, pressure = primitive.unbind(dim=-1)
    conservative = primitive_to_conservative(primitive, gamma=gamma)
    energy = conservative[..., 2]
    return torch.stack(
        (
            rho * velocity,
            rho * velocity.square() + pressure,
            (energy + pressure) * velocity,
        ),
        dim=-1,
    )


def normal_flux_from_primitive(
    primitive: torch.Tensor,
    normal: torch.Tensor,
    gamma: float = 1.4,
) -> torch.Tensor:
    """Physical flux through a 1D face with outward normal ``+1`` or ``-1``."""

    normal_scalar = normal.squeeze(-1)
    return euler_flux_from_primitive(primitive, gamma=gamma) * normal_scalar.unsqueeze(
        -1
    )


def rusanov_flux_from_primitive(
    owner_primitive: torch.Tensor,
    neighbor_primitive: torch.Tensor,
    normal: torch.Tensor,
    gamma: float = 1.4,
) -> torch.Tensor:
    """Rusanov/LLF flux oriented from owner to neighbor/boundary."""

    owner_cons = primitive_to_conservative(owner_primitive, gamma=gamma)
    neighbor_cons = primitive_to_conservative(neighbor_primitive, gamma=gamma)
    owner_flux = normal_flux_from_primitive(owner_primitive, normal, gamma=gamma)
    neighbor_flux = normal_flux_from_primitive(neighbor_primitive, normal, gamma=gamma)

    n = normal.squeeze(-1)
    rho_l, u_l, p_l = owner_primitive.unbind(dim=-1)
    rho_r, u_r, p_r = neighbor_primitive.unbind(dim=-1)
    c_l = torch.sqrt((gamma * p_l / rho_l).clamp_min(0.0))
    c_r = torch.sqrt((gamma * p_r / rho_r).clamp_min(0.0))
    speed = torch.maximum(
        (u_l * n).abs() + c_l * n.abs(), (u_r * n).abs() + c_r * n.abs()
    )

    return 0.5 * (owner_flux + neighbor_flux) - 0.5 * speed.unsqueeze(-1) * (
        neighbor_cons - owner_cons
    )


def reflect_primitive(primitive: torch.Tensor) -> torch.Tensor:
    """Reflect a primitive state at a stationary 1D wall."""

    rho, velocity, pressure = primitive.unbind(dim=-1)
    return torch.stack((rho, -velocity, pressure), dim=-1)


def make_uniform_1d_geometry(x: torch.Tensor) -> FiniteVolumeGeometry:
    """Create cell/face geometry for a batched uniform 1D grid."""

    if x.ndim == 1:
        x = x.unsqueeze(0)
    if x.ndim != 2:
        raise ValueError("x must have shape [cells] or [batch, cells]")
    batch_size, num_cells = x.shape
    if num_cells < 2:
        raise ValueError("at least two cells are required")

    dx_cells = x[:, 1:] - x[:, :-1]
    if not torch.all(dx_cells > 0):
        raise ValueError("x coordinates must be strictly increasing")

    dtype = x.dtype
    device = x.device
    left_dx = dx_cells[:, :1]
    right_dx = dx_cells[:, -1:]
    interior_faces = 0.5 * (x[:, :-1] + x[:, 1:])
    left_face = x[:, :1] - 0.5 * left_dx
    right_face = x[:, -1:] + 0.5 * right_dx
    face_centers = torch.cat((left_face, interior_faces, right_face), dim=1).unsqueeze(
        -1
    )

    left_width = interior_faces[:, :1] - left_face
    middle_width = interior_faces[:, 1:] - interior_faces[:, :-1]
    right_width = right_face - interior_faces[:, -1:]
    cell_volume = torch.cat((left_width, middle_width, right_width), dim=1)

    face_area = torch.ones(batch_size, num_cells + 1, dtype=dtype, device=device)
    face_normal = torch.ones(batch_size, num_cells + 1, 1, dtype=dtype, device=device)
    face_normal[:, 0, 0] = -1.0

    owner_single = torch.empty(num_cells + 1, dtype=torch.long, device=device)
    owner_single[0] = 0
    owner_single[1:-1] = torch.arange(num_cells - 1, dtype=torch.long, device=device)
    owner_single[-1] = num_cells - 1
    neighbor_single = torch.empty(num_cells + 1, dtype=torch.long, device=device)
    neighbor_single[0] = -1
    neighbor_single[1:-1] = torch.arange(1, num_cells, dtype=torch.long, device=device)
    neighbor_single[-1] = -1

    return FiniteVolumeGeometry(
        cell_centers=x.unsqueeze(-1),
        cell_volume=cell_volume,
        face_centers=face_centers,
        face_area=face_area,
        face_normal=face_normal,
        face_owner=owner_single.unsqueeze(0).expand(batch_size, -1),
        face_neighbor=neighbor_single.unsqueeze(0).expand(batch_size, -1),
    )


def make_euler1d_batch(
    current_primitive: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor | float,
    *,
    target_primitive: torch.Tensor | None = None,
    gamma: float = 1.4,
    left_boundary_primitive: torch.Tensor | None = None,
    right_initial_primitive: torch.Tensor | None = None,
    right_boundary_primitive: torch.Tensor | None = None,
) -> Euler1DBatch:
    """Build an ``Euler1DBatch`` from primitive snapshots and cell centers."""

    if current_primitive.ndim != 3 or current_primitive.shape[-1] != 3:
        raise ValueError("current_primitive must have shape [batch, cells, 3]")
    batch_size = current_primitive.shape[0]
    if x.ndim == 1:
        x = x.unsqueeze(0).expand(batch_size, -1)
    if x.shape != current_primitive.shape[:2]:
        raise ValueError("x must have shape [cells] or [batch, cells]")

    dt_tensor = torch.as_tensor(
        dt,
        dtype=current_primitive.dtype,
        device=current_primitive.device,
    )
    if dt_tensor.ndim == 0:
        dt_tensor = dt_tensor.expand(batch_size)
    dt_tensor = dt_tensor.reshape(batch_size)

    return Euler1DBatch(
        current_primitive=current_primitive,
        geometry=make_uniform_1d_geometry(x.to(current_primitive.device)),
        dt=dt_tensor,
        target_primitive=target_primitive,
        gamma=gamma,
        left_boundary_primitive=left_boundary_primitive,
        right_initial_primitive=right_initial_primitive,
        right_boundary_primitive=right_boundary_primitive,
    )


def update_from_face_flux(
    batch: Euler1DBatch,
    face_flux: torch.Tensor,
) -> torch.Tensor:
    """Update current state with a face-normal conservative flux."""

    return finite_volume_update(
        batch.current_conservative, face_flux, batch.geometry, batch.dt
    )
