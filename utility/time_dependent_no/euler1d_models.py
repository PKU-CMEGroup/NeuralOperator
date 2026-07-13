"""1D Euler model heads for solver-facing target diagnostics."""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn

from baselines.fno import FNO1d
from utility.time_dependent_no.euler1d import Euler1DBatch
from utility.time_dependent_no.euler1d_targets import owner_neighbor_primitives
from utility.time_dependent_no.fv import gather_cells, scatter_faces_to_cells


Euler1DTarget = Literal[
    "state",
    "residual",
    "primitive_residual",
    "limited_residual",
    "flux",
    "limited_flux",
    "physical_flux_correction",
    "interface",
    "positive_limited_interface",
    "cpg_interface",
]

CELL_TARGETS: tuple[str, ...] = (
    "state",
    "residual",
    "primitive_residual",
    "limited_residual",
)


def target_output_dim(target: Euler1DTarget) -> int:
    if target in CELL_TARGETS:
        return 3
    if target in ("flux", "limited_flux", "physical_flux_correction"):
        return 3
    if target in ("interface", "positive_limited_interface", "cpg_interface"):
        return 6
    raise ValueError(f"unsupported target: {target}")


def reshape_target_output(raw: torch.Tensor, target: Euler1DTarget) -> torch.Tensor:
    if target in ("interface", "positive_limited_interface", "cpg_interface"):
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
        width: int = 64,
        layers: list[int] | None = None,
        fc_dim: int = 128,
        pad_ratio: float = 0.0,
    ) -> None:
        super().__init__()
        self.target = target
        in_dim = 4 if target in CELL_TARGETS else 9
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
        features = (
            cell_features(batch)
            if self.target in CELL_TARGETS
            else face_features(batch)
        )
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


class ReferenceMLP(nn.Module):
    """Reference-style MLP block: Linear/ReLU stack with optional LayerNorm."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        *,
        layers: int = 3,
        layer_norm: bool = True,
    ) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError("layers must be >= 1")
        modules: list[nn.Module] = []
        current = in_dim
        for _ in range(layers - 1):
            modules.append(nn.Linear(current, hidden_dim))
            modules.append(nn.ReLU())
            current = hidden_dim
        modules.append(nn.Linear(current, out_dim))
        if layer_norm:
            modules.append(nn.LayerNorm(out_dim))
        self.net = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DirectedMessageLayer(nn.Module):
    """One source/target directed edge-update layer in the CPGNet style."""

    def __init__(self, hidden_dim: int, *, mlp_layers: int = 3) -> None:
        super().__init__()
        self.message_mlp = ReferenceMLP(
            4 * hidden_dim,
            hidden_dim,
            hidden_dim,
            layers=mlp_layers,
            layer_norm=True,
        )
        self.root = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.node_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        h: torch.Tensor,
        edge_attr: torch.Tensor,
        source: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h_source = gather_cells(h, source)
        h_target = gather_cells(h, target)
        message_input = torch.cat(
            (h_target, h_source, h_source - h_target, edge_attr),
            dim=-1,
        )
        message = self.message_mlp(message_input)
        aggregate = scatter_faces_to_cells(message, target, h.shape[1])
        h_next = torch.relu(self.node_norm(self.root(h) + aggregate))
        return h_next, edge_attr + message, message


class GeometricEdgeLayer(nn.Module):
    """Reference edge-encoder layer operating only on geometry features."""

    def __init__(
        self,
        node_in_dim: int,
        edge_in_dim: int,
        out_dim: int,
        *,
        mlp_layers: int = 3,
    ) -> None:
        super().__init__()
        self.message_mlp = ReferenceMLP(
            2 * node_in_dim + edge_in_dim,
            out_dim,
            out_dim,
            layers=mlp_layers,
            layer_norm=True,
        )
        self.root = nn.Linear(node_in_dim, out_dim, bias=False)
        self.node_norm = nn.LayerNorm(out_dim)

    def forward(
        self,
        node_attr: torch.Tensor,
        edge_attr: torch.Tensor,
        source: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source_attr = gather_cells(node_attr, source)
        target_attr = gather_cells(node_attr, target)
        message = self.message_mlp(
            torch.cat((target_attr, source_attr, edge_attr), dim=-1)
        )
        aggregate = scatter_faces_to_cells(message, target, node_attr.shape[1])
        node_next = torch.relu(self.node_norm(self.root(node_attr) + aggregate))
        return node_next, message


class CPGNetEuler1D(nn.Module):
    """Paper-faithful CPG reconstruction solver adapted to exact 1D geometry."""

    def __init__(
        self,
        *,
        hidden_dim: int = 128,
        message_passing_steps: int = 12,
        mlp_layers: int = 3,
        edge_hidden_dim: int = 32,
        edge_encoder_steps: int = 3,
        primitive_mean: torch.Tensor | None = None,
        primitive_std: torch.Tensor | None = None,
        positivity_eps: float = 1.0e-6,
    ) -> None:
        super().__init__()
        if message_passing_steps < 1:
            raise ValueError("message_passing_steps must be >= 1")
        if edge_encoder_steps < 0:
            raise ValueError("edge_encoder_steps must be >= 0")
        self.hidden_dim = int(hidden_dim)
        self.message_passing_steps = int(message_passing_steps)
        self.edge_encoder_steps = int(edge_encoder_steps)
        self.positivity_eps = float(positivity_eps)

        mean = torch.zeros(3) if primitive_mean is None else primitive_mean.reshape(3)
        std = torch.ones(3) if primitive_std is None else primitive_std.reshape(3)
        self.register_buffer("primitive_mean", mean.detach().clone())
        self.register_buffer("primitive_std", std.detach().clone().clamp_min(1.0e-6))

        node_solution_dim = 9
        node_geometry_dim = 4
        edge_geometry_dim = 3
        self.node_encoder = ReferenceMLP(
            node_solution_dim + node_geometry_dim,
            hidden_dim,
            hidden_dim,
            layers=mlp_layers,
        )
        self.first_edge_encoder = GeometricEdgeLayer(
            node_geometry_dim,
            edge_geometry_dim,
            edge_hidden_dim,
            mlp_layers=mlp_layers,
        )
        self.edge_encoder_layers = nn.ModuleList(
            GeometricEdgeLayer(
                edge_hidden_dim,
                edge_hidden_dim,
                edge_hidden_dim,
                mlp_layers=mlp_layers,
            )
            for _ in range(edge_encoder_steps)
        )
        self.edge_expand = nn.Linear(edge_hidden_dim, hidden_dim, bias=False)
        self.message_layers = nn.ModuleList(
            DirectedMessageLayer(hidden_dim, mlp_layers=mlp_layers)
            for _ in range(message_passing_steps)
        )
        reconstruction_dim = hidden_dim + hidden_dim + edge_hidden_dim
        self.reconstruction = nn.Sequential(
            nn.Linear(reconstruction_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.rho_decoder = ReferenceMLP(hidden_dim, 32, 1, layers=3, layer_norm=False)
        self.velocity_decoder = ReferenceMLP(
            hidden_dim, 32, 1, layers=3, layer_norm=False
        )
        self.pressure_decoder = ReferenceMLP(
            hidden_dim, 32, 1, layers=3, layer_norm=False
        )

    def _normalize_primitive(self, primitive: torch.Tensor) -> torch.Tensor:
        mean = self.primitive_mean.to(dtype=primitive.dtype).view(1, 1, 3)
        std = self.primitive_std.to(dtype=primitive.dtype).view(1, 1, 3)
        return (primitive - mean) / std

    def _graph_features(
        self, batch: Euler1DBatch
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        current = batch.current_primitive
        batch_size, num_cells, _ = current.shape
        _, exterior = owner_neighbor_primitives(batch)
        node_primitive = torch.cat((exterior[:, :1], current, exterior[:, -1:]), dim=1)

        x = batch.geometry.cell_centers.to(dtype=current.dtype, device=current.device)
        left_dx = x[:, 1:2] - x[:, :1]
        right_dx = x[:, -1:] - x[:, -2:-1]
        node_x = torch.cat((x[:, :1] - left_dx, x, x[:, -1:] + right_dx), dim=1)
        midpoint = 0.5 * (node_x[:, :1] + node_x[:, -1:])
        half_span = (0.5 * (node_x[:, -1:] - node_x[:, :1])).clamp_min(1.0e-8)
        node_x = (node_x - midpoint) / half_span

        node_type = current.new_zeros(batch_size, num_cells + 2, 3)
        node_type[:, 1:-1, 0] = 1.0
        node_type[:, 0, 1] = 1.0
        node_type[:, -1, 2] = 1.0
        node_geometry = torch.cat((node_x, node_type), dim=-1)

        left_case = (
            exterior[:, 0]
            if batch.left_boundary_primitive is None
            else batch.left_boundary_primitive.to(
                dtype=current.dtype, device=current.device
            )
        )
        right_case = (
            current[:, -1]
            if batch.right_initial_primitive is None
            else batch.right_initial_primitive.to(
                dtype=current.dtype, device=current.device
            )
        )
        node_solution = torch.cat(
            (
                self._normalize_primitive(node_primitive),
                self._normalize_primitive(left_case.unsqueeze(1)).expand(
                    -1, num_cells + 2, -1
                ),
                self._normalize_primitive(right_case.unsqueeze(1)).expand(
                    -1, num_cells + 2, -1
                ),
            ),
            dim=-1,
        )

        owner = batch.geometry.face_owner.to(device=current.device) + 1
        neighbor = batch.geometry.face_neighbor.to(device=current.device)
        normal = batch.geometry.face_normal.to(device=current.device).squeeze(-1)
        boundary = neighbor.lt(0)
        boundary_node = torch.where(
            normal.lt(0),
            torch.zeros_like(neighbor),
            torch.full_like(neighbor, num_cells + 1),
        )
        neighbor = torch.where(boundary, boundary_node, neighbor + 1)
        source = torch.cat((neighbor, owner), dim=1)
        target = torch.cat((owner, neighbor), dim=1)

        source_x = gather_cells(node_x, source)
        target_x = gather_cells(node_x, target)
        displacement = source_x - target_x
        distance = displacement.abs().clamp_min(1.0e-8)
        direction = displacement / distance
        edge_geometry = torch.cat((displacement, distance, direction), dim=-1)
        return node_solution, node_geometry, edge_geometry, source, target

    def _encode(
        self, batch: Euler1DBatch
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        node_solution, node_geometry, edge_geometry, source, target = (
            self._graph_features(batch)
        )
        geometric_nodes, encoded_edge = self.first_edge_encoder(
            node_geometry, edge_geometry, source, target
        )
        for layer in self.edge_encoder_layers:
            old_edge = encoded_edge
            geometric_nodes, new_edge = layer(
                geometric_nodes, encoded_edge, source, target
            )
            encoded_edge = old_edge + new_edge

        node_hidden = self.node_encoder(
            torch.cat((node_solution, node_geometry), dim=-1)
        )
        original_edge = encoded_edge
        flow_edge = self.edge_expand(encoded_edge)
        for layer in self.message_layers:
            node_hidden, flow_edge, _ = layer(node_hidden, flow_edge, source, target)
        return node_hidden, flow_edge, original_edge, target

    def directed_message_diagnostics(
        self, batch: Euler1DBatch
    ) -> dict[str, torch.Tensor]:
        node_solution, node_geometry, edge_geometry, source, target = (
            self._graph_features(batch)
        )
        geometric_nodes, encoded_edge = self.first_edge_encoder(
            node_geometry, edge_geometry, source, target
        )
        for layer in self.edge_encoder_layers:
            old_edge = encoded_edge
            geometric_nodes, new_edge = layer(
                geometric_nodes, encoded_edge, source, target
            )
            encoded_edge = old_edge + new_edge
        node_hidden = self.node_encoder(
            torch.cat((node_solution, node_geometry), dim=-1)
        )
        flow_edge = self.edge_expand(encoded_edge)
        _, _, message = self.message_layers[0](node_hidden, flow_edge, source, target)
        num_faces = batch.geometry.face_owner.shape[1]
        return {
            "owner_side": message[:, :num_faces],
            "neighbor_side": message[:, num_faces:],
        }

    def forward(self, batch: Euler1DBatch) -> torch.Tensor:
        node_hidden, flow_edge, original_edge, target = self._encode(batch)
        target_hidden = gather_cells(node_hidden, target)
        reconstruction = self.reconstruction(
            torch.cat((target_hidden, flow_edge, original_edge), dim=-1)
        )
        rho = torch.exp(self.rho_decoder(reconstruction)) + self.positivity_eps
        velocity = self.velocity_decoder(reconstruction)
        pressure = (
            torch.exp(self.pressure_decoder(reconstruction)) + self.positivity_eps
        )
        directed_state = torch.cat((rho, velocity, pressure), dim=-1)
        num_faces = batch.geometry.face_owner.shape[1]
        owner_state = directed_state[:, :num_faces]
        neighbor_state = directed_state[:, num_faces:]

        _, physical_exterior = owner_neighbor_primitives(batch)
        normal = batch.geometry.face_normal.to(device=owner_state.device)
        boundary = batch.geometry.face_neighbor.lt(0).unsqueeze(-1)
        left_boundary = boundary & normal.lt(0)
        right_boundary = boundary & normal.gt(0)
        neighbor_state = torch.where(left_boundary, physical_exterior, neighbor_state)
        reflected_owner = owner_state.clone()
        reflected_owner[..., 1] = -reflected_owner[..., 1]
        neighbor_state = torch.where(right_boundary, reflected_owner, neighbor_state)
        return torch.stack((owner_state, neighbor_state), dim=-2)


class CPGStyleTargetEuler1DHead(nn.Module):
    """Deprecated directed target head; this is not the CPG solver baseline.

    The branch-local 1D data do not use the reference PyG graph object, so this
    The module keeps selected reference architecture decisions in tensor form:
    source/target directed edge messages, unshared message layers, ReLU plus
    LayerNorm MLP blocks, 128-wide hidden states, 12 flow layers by default, and
    face outputs built from both directed embeddings of each finite-volume face.
    """

    def __init__(
        self,
        target: Euler1DTarget,
        *,
        hidden_dim: int = 128,
        message_passing_steps: int = 12,
        mlp_layers: int = 3,
    ) -> None:
        super().__init__()
        if message_passing_steps < 1:
            raise ValueError("message_passing_steps must be >= 1")
        self.target = target
        self.hidden_dim = int(hidden_dim)
        self.message_passing_steps = int(message_passing_steps)
        self.node_encoder = ReferenceMLP(4, hidden_dim, hidden_dim, layers=mlp_layers)
        self.edge_encoder = ReferenceMLP(9, hidden_dim, hidden_dim, layers=mlp_layers)
        self.message_layers = nn.ModuleList(
            DirectedMessageLayer(hidden_dim, mlp_layers=mlp_layers)
            for _ in range(message_passing_steps)
        )
        self.node_decoder = ReferenceMLP(
            hidden_dim,
            hidden_dim,
            3,
            layers=mlp_layers,
            layer_norm=False,
        )
        self.face_decoder = ReferenceMLP(
            2 * hidden_dim + 9,
            hidden_dim,
            target_output_dim(target),
            layers=mlp_layers,
            layer_norm=False,
        )

    @staticmethod
    def directed_edge_indices(batch: Euler1DBatch) -> tuple[torch.Tensor, torch.Tensor]:
        owner = batch.geometry.face_owner.to(device=batch.current_primitive.device)
        neighbor = batch.geometry.face_neighbor.to(
            device=batch.current_primitive.device
        )
        valid_neighbor = neighbor.ge(0)

        owner_side_source = torch.where(valid_neighbor, neighbor, owner)
        owner_side_target = owner
        neighbor_side_source = owner
        neighbor_side_target = torch.where(valid_neighbor, neighbor, owner)
        source = torch.cat((owner_side_source, neighbor_side_source), dim=1)
        target = torch.cat((owner_side_target, neighbor_side_target), dim=1)
        return source, target

    @staticmethod
    def directed_edge_features(batch: Euler1DBatch) -> torch.Tensor:
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
        boundary = boundary.unsqueeze(-1)

        owner_side = torch.cat(
            (neighbor, owner, face_centers, face_normal, boundary),
            dim=-1,
        )
        neighbor_side = torch.cat(
            (owner, neighbor, face_centers, -face_normal, boundary),
            dim=-1,
        )
        return torch.cat((owner_side, neighbor_side), dim=1)

    def _encode(
        self, batch: Euler1DBatch
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        source, target = self.directed_edge_indices(batch)
        h = self.node_encoder(cell_features(batch))
        edge_attr = self.edge_encoder(self.directed_edge_features(batch))
        return h, edge_attr, source, target

    def directed_message_diagnostics(
        self, batch: Euler1DBatch
    ) -> dict[str, torch.Tensor]:
        """Return first-layer directed messages split by face side for tests."""

        h, edge_attr, source, target = self._encode(batch)
        _, _, message = self.message_layers[0](h, edge_attr, source, target)
        num_faces = batch.geometry.face_owner.shape[1]
        return {
            "owner_side": message[:, :num_faces],
            "neighbor_side": message[:, num_faces:],
        }

    def forward(self, batch: Euler1DBatch) -> torch.Tensor:
        h, edge_attr, source, target = self._encode(batch)
        for layer in self.message_layers:
            h, edge_attr, _ = layer(h, edge_attr, source, target)

        if self.target in CELL_TARGETS:
            return self.node_decoder(h)

        num_faces = batch.geometry.face_owner.shape[1]
        owner_side = edge_attr[:, :num_faces]
        neighbor_side = edge_attr[:, num_faces:]
        face_hidden = torch.cat(
            (owner_side, neighbor_side, face_features(batch)), dim=-1
        )
        raw = self.face_decoder(face_hidden)
        return reshape_target_output(raw, self.target)


class CPGStylePilotEuler1DHead(nn.Module):
    """Deprecated small CPG-style pilot head kept only for explicit ablations."""

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
        self.edge_encoder = MLP(
            2 * hidden_dim + 9, hidden_dim, hidden_dim, layers=mlp_layers
        )
        self.node_update = MLP(
            2 * hidden_dim, hidden_dim, hidden_dim, layers=mlp_layers
        )
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
            edge_msg = self.edge_encoder(
                torch.cat((h_owner, h_neighbor, face_base), dim=-1)
            )
            agg = scatter_faces_to_cells(edge_msg, owner, h.shape[1])
            agg = agg + scatter_faces_to_cells(edge_msg, neighbor, h.shape[1])
            h = h + self.node_update(torch.cat((h, agg), dim=-1))

        if self.target in CELL_TARGETS:
            return self.node_decoder(h)

        h_owner = gather_cells(h, owner)
        h_neighbor = gather_cells(h, neighbor, fill_value=0.0)
        boundary = neighbor.lt(0).unsqueeze(-1)
        h_neighbor = torch.where(boundary, h_owner, h_neighbor)
        raw = self.face_decoder(torch.cat((h_owner, h_neighbor, face_base), dim=-1))
        return reshape_target_output(raw, self.target)


# Backward-compatible import name. Training scripts no longer map this to cpgnet.
CPGStyleEuler1DHead = CPGStylePilotEuler1DHead
