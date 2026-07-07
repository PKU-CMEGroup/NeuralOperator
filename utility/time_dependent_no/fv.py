"""Finite-volume geometry and differentiable conservative updates.

The classes here are PDE-agnostic. Euler-specific flux functions live in
``euler1d.py`` and later 2D modules can reuse this same update contract.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FiniteVolumeGeometry:
    """Batched cell/face geometry for a finite-volume update.

    Face fluxes are oriented from ``face_owner`` toward ``face_neighbor``. For a
    boundary face, ``face_neighbor`` is ``-1`` and the flux is outward from the
    owner cell.
    """

    cell_centers: torch.Tensor
    cell_volume: torch.Tensor
    face_centers: torch.Tensor
    face_area: torch.Tensor
    face_normal: torch.Tensor
    face_owner: torch.Tensor
    face_neighbor: torch.Tensor

    def to(self, device: torch.device | str) -> "FiniteVolumeGeometry":
        return FiniteVolumeGeometry(
            cell_centers=self.cell_centers.to(device),
            cell_volume=self.cell_volume.to(device),
            face_centers=self.face_centers.to(device),
            face_area=self.face_area.to(device),
            face_normal=self.face_normal.to(device),
            face_owner=self.face_owner.to(device),
            face_neighbor=self.face_neighbor.to(device),
        )


def gather_cells(
    cell_values: torch.Tensor,
    indices: torch.Tensor,
    *,
    fill_value: float | None = None,
) -> torch.Tensor:
    """Gather cell values with optional filling for negative indices."""

    if cell_values.ndim != 3:
        raise ValueError("cell_values must have shape [batch, cells, channels]")
    if indices.ndim != 2:
        raise ValueError("indices must have shape [batch, items]")
    if cell_values.shape[0] != indices.shape[0]:
        raise ValueError("batch dimensions of cell_values and indices disagree")

    safe_indices = indices.clamp_min(0)
    gather_index = safe_indices.unsqueeze(-1).expand(-1, -1, cell_values.shape[-1])
    gathered = torch.gather(cell_values, dim=1, index=gather_index)

    if fill_value is None:
        return gathered

    invalid = indices.lt(0).unsqueeze(-1)
    fill = torch.full_like(gathered, fill_value)
    return torch.where(invalid, fill, gathered)


def scatter_faces_to_cells(
    face_values: torch.Tensor,
    cell_indices: torch.Tensor,
    num_cells: int,
) -> torch.Tensor:
    """Scatter-add face values to cells, ignoring negative cell indices."""

    if face_values.ndim != 3:
        raise ValueError("face_values must have shape [batch, faces, channels]")
    if cell_indices.ndim != 2:
        raise ValueError("cell_indices must have shape [batch, faces]")
    if face_values.shape[:2] != cell_indices.shape:
        raise ValueError("face_values and cell_indices shape mismatch")

    valid = cell_indices.ge(0)
    safe_indices = cell_indices.clamp_min(0)
    src = torch.where(valid.unsqueeze(-1), face_values, torch.zeros_like(face_values))
    index = safe_indices.unsqueeze(-1).expand(-1, -1, face_values.shape[-1])

    out = torch.zeros(
        face_values.shape[0],
        num_cells,
        face_values.shape[-1],
        dtype=face_values.dtype,
        device=face_values.device,
    )
    out.scatter_add_(dim=1, index=index, src=src)
    return out


def finite_volume_update(
    conservative: torch.Tensor,
    face_flux: torch.Tensor,
    geometry: FiniteVolumeGeometry,
    dt: torch.Tensor | float,
) -> torch.Tensor:
    """Apply one explicit conservative FV update.

    ``face_flux`` is the normal conservative flux through each face, oriented
    from owner to neighbor. Interior faces subtract from owner and add to
    neighbor. Boundary faces only update their owner cell.
    """

    if conservative.ndim != 3:
        raise ValueError("conservative must have shape [batch, cells, variables]")
    if face_flux.ndim != 3:
        raise ValueError("face_flux must have shape [batch, faces, variables]")
    if conservative.shape[0] != face_flux.shape[0]:
        raise ValueError("batch dimensions of conservative and face_flux disagree")
    if conservative.shape[-1] != face_flux.shape[-1]:
        raise ValueError("conservative and face_flux variable dimensions disagree")

    batch_size, num_cells, _ = conservative.shape
    if geometry.face_owner.shape != face_flux.shape[:2]:
        raise ValueError("geometry.face_owner must match [batch, faces]")
    if geometry.face_neighbor.shape != face_flux.shape[:2]:
        raise ValueError("geometry.face_neighbor must match [batch, faces]")

    dt_tensor = torch.as_tensor(dt, dtype=conservative.dtype, device=conservative.device)
    if dt_tensor.ndim == 0:
        dt_tensor = dt_tensor.expand(batch_size)
    dt_tensor = dt_tensor.reshape(batch_size, 1)

    face_area = geometry.face_area.to(dtype=conservative.dtype, device=conservative.device)
    cell_volume = geometry.cell_volume.to(dtype=conservative.dtype, device=conservative.device)
    owner = geometry.face_owner.to(device=conservative.device)
    neighbor = geometry.face_neighbor.to(device=conservative.device)

    owner_volume = torch.gather(cell_volume, dim=1, index=owner)
    owner_scale = (dt_tensor * face_area / owner_volume).unsqueeze(-1)
    owner_delta = -owner_scale * face_flux

    neighbor_valid = neighbor.ge(0)
    neighbor_safe = neighbor.clamp_min(0)
    neighbor_volume = torch.gather(cell_volume, dim=1, index=neighbor_safe)
    neighbor_scale = (dt_tensor * face_area / neighbor_volume).unsqueeze(-1)
    neighbor_delta = torch.where(
        neighbor_valid.unsqueeze(-1),
        neighbor_scale * face_flux,
        torch.zeros_like(face_flux),
    )

    return (
        conservative
        + scatter_faces_to_cells(owner_delta, owner, num_cells)
        + scatter_faces_to_cells(neighbor_delta, neighbor, num_cells)
    )
