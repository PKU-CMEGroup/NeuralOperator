"""Regular-grid PCNO building blocks for REALM benchmark cases.

The wrapper consumes already transformed and normalized states and emits a raw
normalized tensor.  Direct versus residual reconstruction remains external via
``realm_benchmark.apply_parameterization``.  This keeps the architecture and
checkpoint surface independent of that scientific choice.

The released coordinate and domain contracts must be audited case by case.
Fourier periods are explicit inputs; they are never guessed from point-center
extents.  Default quadrature weights are an operational uniform measure and do
not support a physical-conservation claim.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from pcno.pcno import PCNO, compute_Fourier_modes

REALM_REGULAR_GRID_GEOMETRY_SCHEMA = "realm_regular_grid_pcno_geometry_v1"
REALM_REGULAR_GRID_MODEL_SCHEMA = "realm_regular_grid_pcno_v1"


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _positive_float_pair(values: Sequence[float], name: str) -> tuple[float, float]:
    if len(values) != 2:
        raise ValueError(f"{name} must contain exactly two values")
    result = (float(values[0]), float(values[1]))
    if not np.all(np.isfinite(result)) or any(value <= 0.0 for value in result):
        raise ValueError(f"{name} must contain positive finite values")
    return result


def _positive_integer_pair(values: Sequence[int], name: str) -> tuple[int, int]:
    if len(values) != 2:
        raise ValueError(f"{name} must contain exactly two values")
    return (
        _positive_integer(values[0], f"{name}[0]"),
        _positive_integer(values[1], f"{name}[1]"),
    )


@dataclass(frozen=True)
class RealmPCNOConfig:
    """Architecture contract for a fixed-grid REALM PCNO."""

    channels: int
    mode_counts_xy: tuple[int, int]
    layers: tuple[int, ...]
    fc_dim: int = 128
    activation: str = "gelu"
    zero_initialize_head: bool = False
    use_gradient: bool = True

    def __post_init__(self) -> None:
        _positive_integer(self.channels, "channels")
        object.__setattr__(
            self,
            "mode_counts_xy",
            _positive_integer_pair(self.mode_counts_xy, "mode_counts_xy"),
        )
        if len(self.layers) < 2:
            raise ValueError("layers must contain at least two widths")
        object.__setattr__(
            self,
            "layers",
            tuple(_positive_integer(value, "layer width") for value in self.layers),
        )
        _positive_integer(self.fc_dim, "fc_dim")
        if not isinstance(self.activation, str) or not self.activation:
            raise ValueError("activation must be a nonempty string")
        if not isinstance(self.zero_initialize_head, bool):
            raise TypeError("zero_initialize_head must be a bool")
        if not isinstance(self.use_gradient, bool):
            raise TypeError("use_gradient must be a bool")


class _ZeroGradientLayer(nn.Module):
    """Parameter-free shape-preserving replacement for a PCNO gradient branch."""

    def __init__(self, out_channels: int) -> None:
        super().__init__()
        self.out_channels = _positive_integer(out_channels, "out_channels")

    def forward(
        self,
        values: torch.Tensor,
        directed_edges: torch.Tensor,
        edge_gradient_weights: torch.Tensor,
        neighbor_degree: torch.Tensor | None = None,
        flat_edge_indices: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        del directed_edges, edge_gradient_weights, neighbor_degree, flat_edge_indices
        return values.new_zeros(values.shape[0], self.out_channels, values.shape[-1])


@dataclass(frozen=True)
class RealmRegularGridGeometry:
    """Static PCNO tensors for one released rectilinear REALM grid."""

    height: int
    width: int
    domain_lengths_xy: tuple[float, float]
    released_coordinate_order: tuple[str, str]
    nodes: np.ndarray
    node_mask: np.ndarray
    node_weights: np.ndarray
    node_rhos: np.ndarray
    directed_edges: np.ndarray
    edge_gradient_weights: np.ndarray
    contract: dict[str, Any]


def _strict_monotone_axis(values: np.ndarray, name: str) -> np.ndarray:
    axis = np.asarray(values, dtype=np.float64).reshape(-1)
    if axis.size < 2 or not np.all(np.isfinite(axis)):
        raise ValueError(f"{name} must contain at least two finite coordinates")
    differences = np.diff(axis)
    if not (np.all(differences > 0.0) or np.all(differences < 0.0)):
        raise ValueError(f"{name} must be strictly monotone")
    return axis


def _cartesian_gradient_graph(
    x_axis: np.ndarray,
    y_axis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct axial least-squares weights without per-node pseudoinverses."""

    height = y_axis.size
    width = x_axis.size
    indices = np.arange(height * width, dtype=np.int64).reshape(height, width)

    left_offset = np.zeros(width, dtype=np.float64)
    right_offset = np.zeros(width, dtype=np.float64)
    left_offset[1:] = x_axis[:-1] - x_axis[1:]
    right_offset[:-1] = x_axis[1:] - x_axis[:-1]
    x_denominator = left_offset**2 + right_offset**2

    upper_offset = np.zeros(height, dtype=np.float64)
    lower_offset = np.zeros(height, dtype=np.float64)
    upper_offset[1:] = y_axis[:-1] - y_axis[1:]
    lower_offset[:-1] = y_axis[1:] - y_axis[:-1]
    y_denominator = upper_offset**2 + lower_offset**2

    edge_blocks: list[np.ndarray] = []
    weight_blocks: list[np.ndarray] = []

    def append(
        targets: np.ndarray,
        sources: np.ndarray,
        weights_x: np.ndarray,
        weights_y: np.ndarray,
    ) -> None:
        edge_blocks.append(np.column_stack((targets.reshape(-1), sources.reshape(-1))))
        weight_blocks.append(
            np.column_stack((weights_x.reshape(-1), weights_y.reshape(-1)))
        )

    horizontal_zeros = np.zeros((height, width - 1), dtype=np.float64)
    append(
        indices[:, 1:],
        indices[:, :-1],
        np.broadcast_to(left_offset[1:] / x_denominator[1:], horizontal_zeros.shape),
        horizontal_zeros,
    )
    append(
        indices[:, :-1],
        indices[:, 1:],
        np.broadcast_to(right_offset[:-1] / x_denominator[:-1], horizontal_zeros.shape),
        horizontal_zeros,
    )

    vertical_zeros = np.zeros((height - 1, width), dtype=np.float64)
    append(
        indices[1:, :],
        indices[:-1, :],
        vertical_zeros,
        np.broadcast_to(
            (upper_offset[1:] / y_denominator[1:])[:, None], vertical_zeros.shape
        ),
    )
    append(
        indices[:-1, :],
        indices[1:, :],
        vertical_zeros,
        np.broadcast_to(
            (lower_offset[:-1] / y_denominator[:-1])[:, None],
            vertical_zeros.shape,
        ),
    )

    return (
        np.concatenate(edge_blocks, axis=0).astype(np.int64, copy=False),
        np.concatenate(weight_blocks, axis=0).astype(np.float32, copy=False),
    )


def build_realm_regular_grid_geometry(
    released_coordinates: np.ndarray,
    *,
    domain_lengths_xy: Sequence[float],
    released_coordinate_order: Sequence[str],
    node_weights_yx: np.ndarray | None = None,
) -> RealmRegularGridGeometry:
    """Build a non-wrapping PCNO graph from released Cartesian coordinates.

    Source channels may be ``[y, x]`` or ``[x, y]``.  PCNO geometry, domain
    lengths, and gradient weights always use ``[x, y]`` order.
    """

    source = np.asarray(released_coordinates)
    if source.ndim != 3 or source.shape[0] != 2:
        raise ValueError("released_coordinates must have shape [2, height, width]")
    if not np.issubdtype(source.dtype, np.floating) or not np.all(np.isfinite(source)):
        raise ValueError("released_coordinates must be finite floating values")
    order = tuple(str(value) for value in released_coordinate_order)
    if len(order) != 2 or set(order) != {"x", "y"}:
        raise ValueError(
            "released_coordinate_order must be a permutation of ('x', 'y')"
        )
    lengths = _positive_float_pair(domain_lengths_xy, "domain_lengths_xy")

    x_grid = np.asarray(source[order.index("x")], dtype=np.float64)
    y_grid = np.asarray(source[order.index("y")], dtype=np.float64)
    height, width = x_grid.shape
    if y_grid.shape != (height, width) or min(height, width) < 2:
        raise ValueError("coordinate channels must share a grid of at least 2 by 2")
    if not np.allclose(x_grid, x_grid[:1, :], rtol=0.0, atol=1.0e-10):
        raise ValueError("x coordinates must vary only along the width axis")
    if not np.allclose(y_grid, y_grid[:, :1], rtol=0.0, atol=1.0e-10):
        raise ValueError("y coordinates must vary only along the height axis")

    x_axis = _strict_monotone_axis(x_grid[0], "x axis")
    y_axis = _strict_monotone_axis(y_grid[:, 0], "y axis")
    point_extents = (
        float(np.max(x_axis) - np.min(x_axis)),
        float(np.max(y_axis) - np.min(y_axis)),
    )
    tolerance = 1.0e-10 * max(1.0, *point_extents)
    if any(
        length + tolerance < extent for length, extent in zip(lengths, point_extents)
    ):
        raise ValueError(
            "domain_lengths_xy cannot be smaller than coordinate point extents"
        )

    if node_weights_yx is None:
        weights = np.ones((height, width), dtype=np.float64)
        weight_source = "uniform_operational_measure"
    else:
        weights = np.asarray(node_weights_yx, dtype=np.float64)
        if weights.shape != (height, width):
            raise ValueError("node_weights_yx must have shape [height, width]")
        if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
            raise ValueError("node_weights_yx must be positive and finite")
        weight_source = "explicit_caller_measure"
    weights = weights.reshape(-1, 1)
    weights /= weights.sum()
    node_rhos = weights / np.mean(weights)

    physical_points = np.stack((x_grid.reshape(-1), y_grid.reshape(-1)), axis=-1)
    origin = np.asarray([np.min(x_axis), np.min(y_axis)], dtype=np.float64)
    nodes = physical_points - origin
    directed_edges, edge_gradient_weights = _cartesian_gradient_graph(x_axis, y_axis)
    contract: dict[str, Any] = {
        "schema": REALM_REGULAR_GRID_GEOMETRY_SCHEMA,
        "shape": [height, width],
        "released_coordinate_order": list(order),
        "model_node_order": ["x", "y"],
        "axis_direction_xy": [
            "ascending" if np.diff(x_axis)[0] > 0.0 else "descending",
            "ascending" if np.diff(y_axis)[0] > 0.0 else "descending",
        ],
        "coordinate_origin_xy": origin.tolist(),
        "coordinate_point_extents_xy": list(point_extents),
        "domain_lengths_xy": list(lengths),
        "domain_length_policy": "explicit_audited_external_contract",
        "graph_policy": "nonwrapping_four_neighbor_rectilinear_least_squares",
        "quadrature_policy": weight_source,
        "physical_volume_claim": False,
        "boundary_metadata_used": False,
    }
    return RealmRegularGridGeometry(
        height=height,
        width=width,
        domain_lengths_xy=lengths,
        released_coordinate_order=order,
        nodes=nodes.astype(np.float32),
        node_mask=np.ones((height * width, 1), dtype=np.float32),
        node_weights=weights.astype(np.float32),
        node_rhos=node_rhos.astype(np.float32),
        directed_edges=directed_edges,
        edge_gradient_weights=edge_gradient_weights,
        contract=contract,
    )


def estimate_realm_pcno_static_bytes(
    height: int,
    width: int,
    mode_counts_xy: Sequence[int],
    *,
    floating_bytes: int = 4,
    index_bytes: int = 8,
) -> dict[str, int]:
    """Estimate fixed geometry and six cached Fourier tensors for batch one."""

    height = _positive_integer(height, "height")
    width = _positive_integer(width, "width")
    modes_x, modes_y = _positive_integer_pair(mode_counts_xy, "mode_counts_xy")
    floating_bytes = _positive_integer(floating_bytes, "floating_bytes")
    index_bytes = _positive_integer(index_bytes, "index_bytes")
    node_count = height * width
    edge_count = 2 * (height * (width - 1) + (height - 1) * width)
    mode_count = 2 * modes_x * modes_y + modes_x + modes_y
    pieces = {
        "nodes": node_count * 2 * floating_bytes,
        "node_mask": node_count * floating_bytes,
        "node_weights": node_count * floating_bytes,
        "node_rhos": node_count * floating_bytes,
        "directed_edges": edge_count * 2 * index_bytes,
        "edge_gradient_weights": edge_count * 2 * floating_bytes,
        "modes": mode_count * 2 * floating_bytes,
        "fourier_tensors": (4 * node_count * mode_count + 2 * node_count)
        * floating_bytes,
    }
    pieces["node_count"] = node_count
    pieces["directed_edge_count"] = edge_count
    pieces["fourier_mode_count"] = mode_count
    pieces["total_bytes"] = sum(
        value
        for key, value in pieces.items()
        if key not in {"node_count", "directed_edge_count", "fourier_mode_count"}
    )
    return pieces


class RealmRegularGridPCNO(nn.Module):
    """Raw normalized PCNO map with the same call shape as ``RealmFFNO2d``."""

    def __init__(
        self,
        *,
        config: RealmPCNOConfig,
        geometry: RealmRegularGridGeometry,
    ) -> None:
        super().__init__()
        modes = torch.as_tensor(
            compute_Fourier_modes(
                2,
                list(config.mode_counts_xy),
                list(geometry.domain_lengths_xy),
            ),
            dtype=torch.float32,
        )
        self.backbone = PCNO(
            2,
            modes,
            nmeasures=1,
            layers=list(config.layers),
            fc_dim=config.fc_dim,
            in_dim=config.channels + 3,
            out_dim=config.channels,
            act=config.activation,
        )
        if not config.use_gradient:
            self.backbone.gws = nn.ModuleList(
                _ZeroGradientLayer(out_size) for out_size in config.layers[1:]
            )
            self.backbone.normal_params = list(self.backbone.parameters())
        self.config = config
        self.height = int(geometry.height)
        self.width = int(geometry.width)
        self.released_coordinate_order = tuple(geometry.released_coordinate_order)
        self.geometry_contract = dict(geometry.contract)

        for name, value in (
            ("nodes", geometry.nodes),
            ("node_mask", geometry.node_mask),
            ("node_weights", geometry.node_weights),
            ("node_rhos", geometry.node_rhos),
            ("directed_edges", geometry.directed_edges),
            ("edge_gradient_weights", geometry.edge_gradient_weights),
        ):
            self.register_buffer(
                name,
                torch.as_tensor(value).unsqueeze(0),
                persistent=False,
            )
        with torch.no_grad():
            cached = self.backbone.prepare_fourier_tensors(
                self.nodes, self.node_weights
            )
        for name, value in zip(
            (
                "basis_cos",
                "basis_sin",
                "basis_zero",
                "weighted_basis_cos",
                "weighted_basis_sin",
                "weighted_basis_zero",
            ),
            cached,
        ):
            self.register_buffer(name, value, persistent=False)
        if config.zero_initialize_head:
            self.zero_initialize_output_head()

    def zero_initialize_output_head(self) -> None:
        nn.init.zeros_(self.backbone.fc2.weight)
        nn.init.zeros_(self.backbone.fc2.bias)

    def model_contract(self) -> dict[str, Any]:
        return {
            "schema": REALM_REGULAR_GRID_MODEL_SCHEMA,
            "channels": self.config.channels,
            "mode_counts_xy": list(self.config.mode_counts_xy),
            "layers": list(self.config.layers),
            "fc_dim": self.config.fc_dim,
            "activation": self.config.activation,
            "zero_initialized_output_head": self.config.zero_initialize_head,
            "gradient_branch": self.config.use_gradient,
            "input_representation": "transformed_normalized_released_state",
            "output_representation": "raw_normalized_tensor",
            "parameterization": "external_required_direct_or_residual",
            "pde_residual_claim": False,
            "conservative_update_claim": False,
            "geometry": self.geometry_contract,
        }

    def _expanded_geometry(self, batch_size: int) -> list[torch.Tensor]:
        return [
            self.node_mask.expand(batch_size, -1, -1),
            self.nodes.expand(batch_size, -1, -1),
            self.node_weights.expand(batch_size, -1, -1),
            self.directed_edges.expand(batch_size, -1, -1),
            self.edge_gradient_weights.expand(batch_size, -1, -1),
        ]

    def _expanded_fourier(self, batch_size: int) -> tuple[torch.Tensor, ...]:
        return tuple(
            value.expand(batch_size, *value.shape[1:])
            for value in (
                self.basis_cos,
                self.basis_sin,
                self.basis_zero,
                self.weighted_basis_cos,
                self.weighted_basis_sin,
                self.weighted_basis_zero,
            )
        )

    def _validate_inputs(
        self,
        state: torch.Tensor,
        static_coordinates: torch.Tensor,
    ) -> None:
        if state.ndim != 4 or tuple(state.shape[1:]) != (
            self.config.channels,
            self.height,
            self.width,
        ):
            raise ValueError(
                "state must have shape "
                f"[batch, {self.config.channels}, {self.height}, {self.width}]"
            )
        if static_coordinates.ndim != 4 or tuple(static_coordinates.shape[1:]) != (
            2,
            self.height,
            self.width,
        ):
            raise ValueError(
                "static_coordinates must have shape "
                f"[case, 2, {self.height}, {self.width}]"
            )
        if static_coordinates.shape[0] not in (1, state.shape[0]):
            raise ValueError(
                "coordinate case count must be one or match the state batch"
            )
        for value, name in ((state, "state"), (static_coordinates, "coordinates")):
            if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} must contain finite floating values")
        if (
            state.dtype != static_coordinates.dtype
            or state.device != static_coordinates.device
        ):
            raise ValueError("state and coordinates must share dtype and device")
        if state.device != self.nodes.device or state.dtype != self.nodes.dtype:
            raise ValueError("inputs and model geometry must share dtype and device")

    def forward(
        self,
        state: torch.Tensor,
        static_coordinates: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_inputs(state, static_coordinates)
        batch_size = state.shape[0]
        flat_state = state.permute(0, 2, 3, 1).reshape(
            batch_size, self.height * self.width, self.config.channels
        )
        coordinates = static_coordinates.expand(batch_size, -1, -1, -1)
        x_index = self.released_coordinate_order.index("x")
        y_index = self.released_coordinate_order.index("y")
        flat_coordinates_xy = (
            coordinates[:, [x_index, y_index]]
            .permute(0, 2, 3, 1)
            .reshape(batch_size, self.height * self.width, 2)
        )
        features = torch.cat(
            (
                flat_state,
                flat_coordinates_xy,
                self.node_rhos.expand(batch_size, -1, -1),
            ),
            dim=-1,
        )
        output = self.backbone(
            features,
            self._expanded_geometry(batch_size),
            fourier_tensors=self._expanded_fourier(batch_size),
        )
        return output.reshape(
            batch_size,
            self.height,
            self.width,
            self.config.channels,
        ).permute(0, 3, 1, 2)


__all__ = [
    "REALM_REGULAR_GRID_GEOMETRY_SCHEMA",
    "REALM_REGULAR_GRID_MODEL_SCHEMA",
    "RealmPCNOConfig",
    "RealmRegularGridGeometry",
    "RealmRegularGridPCNO",
    "build_realm_regular_grid_geometry",
    "estimate_realm_pcno_static_bytes",
]
