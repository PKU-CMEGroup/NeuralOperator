"""Lightweight 1D Euler model heads for solver-facing target diagnostics."""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn

from baselines.fno import FNO1d
from utility.time_dependent_no.euler1d import Euler1DBatch
from utility.time_dependent_no.euler1d_targets import owner_neighbor_primitives
from utility.time_dependent_no.fv import gather_cells, scatter_faces_to_cells


Euler1DTarget = Literal["state", "residual", "flux", "interface"]


def target_output_dim(target: Euler1DTarget) -> int:
    if target in ("state", "residual"):
        return 3
    if target == "flux":
        return 3
    if target == "interface":
        return 6
    raise ValueError(f"unsupported target: {target}")


def reshape_target_output(raw: torch.Tensor, target: Euler1DTarget) -> torch.Tensor:
    if target == "interface":
        if raw.shape[-1] != 6:
            raise ValueError("interface head must output 6 channels")
        return raw.reshape(*raw.shape[:-1], 2, 3)
    return raw


def cell_features(batch: Euler1DBatch) -> torch.Tensor:
    x = batch.geometry.cell_centers.to(
        dtype=batch.current_primitive.dtype,
        device=batch.current_primitive.device,
    )
    return torch.cat((batch.current_primitive, x), dim=-1)


def face_features(batch: Euler1DBatch) -> torch.Tensor:
    owner, neighbor = owner_neighbor_primitives(batch)
    face_centers = batch.geometry.face_centers.to(
        dtype=batch.current_primitive.dtype,
        device=batch.current_primitive.device,
    )
    face_normal = batch.geometry.face_normal.to(
        dtype=batch.current_primitive.dtype,
        device=batch.current_primitive.device,
    )
    boundary = batch.geometry.face_neighbor.lt(0).to(
        dtype=batch.current_primitive.dtype,
        device=batch.current_primitive.device,
    )
    return torch.cat(
        (
            owner,
            neighbor,
            face_centers,
            face_normal,
            boundary.unsqueeze(-1),
        ),
        dim=-1,
    )


class FNOEuler1DHead(nn.Module):
    """Resolution-flexible FNO head for cell or face quantities."""

    def __init__(
        self,
        target: Euler1DTarget,
        *,
        modes: list[int],
        width: int = 32,
        layers: list[int] | None = None,
        fc_dim: int = 128,
        pad_ratio: float = 0.0,
    ) -> None:
        super().__init__()
        self.target = target
        in_dim = 4 if target in ("state", "residual") else 9
        self.model = FNO1d(
            modes=modes,
            width=width,
            layers=layers,
            fc_dim=fc_dim,
            in_dim=in_dim,
            out_dim=target_output_dim(target),
            pad_ratio=pad_ratio,
        )

    def forward(self, batch: Euler1DBatch) -> torch.Tensor:
        features = cell_features(batch) if self.target in ("state", "residual") else face_features(batch)
        return reshape_target_output(self.model(features), self.target)


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        *,
        layers: int = 2,
        activation: type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError("layers must be >= 1")
        modules: list[nn.Module] = []
        current = in_dim
        for _ in range(layers - 1):
            modules.append(nn.Linear(current, hidden_dim))
            modules.append(activation())
            current = hidden_dim
        modules.append(nn.Linear(current, out_dim))
        self.net = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CPGStyleEuler1DHead(nn.Module):
    """Small CPG-style edge/message-passing head for 1D experiments.

    This is a branch-local experimental module inspired by the CPGNet update
    decomposition. It intentionally avoids importing or vendoring the external
    author implementation.
    """

    def __init__(
        self,
        target: Euler1DTarget,
        *,
        hidden_dim: int = 64,
        message_passing_steps: int = 3,
        mlp_layers: int = 3,
    ) -> None:
        super().__init__()
        if message_passing_steps < 0:
            raise ValueError("message_passing_steps must be >= 0")
        self.target = target
        self.message_passing_steps = int(message_passing_steps)
        self.node_encoder = MLP(4, hidden_dim, hidden_dim, layers=mlp_layers)
        self.edge_encoder = MLP(2 * hidden_dim + 9, hidden_dim, hidden_dim, layers=mlp_layers)
        self.node_update = MLP(2 * hidden_dim, hidden_dim, hidden_dim, layers=mlp_layers)
        self.node_decoder = MLP(hidden_dim, hidden_dim, 3, layers=mlp_layers)
        self.face_decoder = MLP(
            2 * hidden_dim + 9,
            hidden_dim,
            target_output_dim(target),
            layers=mlp_layers,
        )

    def forward(self, batch: Euler1DBatch) -> torch.Tensor:
        h = self.node_encoder(cell_features(batch))
        face_base = face_features(batch)
        owner = batch.geometry.face_owner.to(device=h.device)
        neighbor = batch.geometry.face_neighbor.to(device=h.device)

        for _ in range(self.message_passing_steps):
            h_owner = gather_cells(h, owner)
            h_neighbor = gather_cells(h, neighbor, fill_value=0.0)
            boundary = neighbor.lt(0).unsqueeze(-1)
            h_neighbor = torch.where(boundary, h_owner, h_neighbor)
            edge_msg = self.edge_encoder(torch.cat((h_owner, h_neighbor, face_base), dim=-1))
            agg = scatter_faces_to_cells(edge_msg, owner, h.shape[1])
            agg = agg + scatter_faces_to_cells(edge_msg, neighbor, h.shape[1])
            h = h + self.node_update(torch.cat((h, agg), dim=-1))

        if self.target in ("state", "residual"):
            return self.node_decoder(h)

        h_owner = gather_cells(h, owner)
        h_neighbor = gather_cells(h, neighbor, fill_value=0.0)
        boundary = neighbor.lt(0).unsqueeze(-1)
        h_neighbor = torch.where(boundary, h_owner, h_neighbor)
        raw = self.face_decoder(torch.cat((h_owner, h_neighbor, face_base), dim=-1))
        return reshape_target_output(raw, self.target)
