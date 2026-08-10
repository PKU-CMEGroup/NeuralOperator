"""PCNO building blocks for the RealPDE Track 2 airfoil forecast.

The competition exposes normalized (B, 20, 32, 64, 3) windows. Only velocity
channels u and v are measured; pressure is a zero-filled interface channel.
This module keeps the forecast model independent of the streaming adaptation
policy so the Stage 1 checkpoint can be evaluated frozen.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from pcno.geo_utility import compute_edge_gradient_weights_helper
from pcno.pcno import PCNO, compute_Fourier_modes
from utility.time_dependent_no.pcno_boundary_fields import (
    compact_cubic_collar,
    semantic_collar_fields,
)

TRACK2_INPUT_STEPS = 20
TRACK2_OUTPUT_STEPS = 20
TRACK2_CHANNELS = 2
TRACK2_SUBSAMPLING = 2
TRACK2_GEOMETRY_SCHEMA = "realpde_track2_dynamic_solid_geometry_v2"
TRACK2_MODEL_SCHEMA = "realpde_track2_pcno_v2"


def _finite_vector(value: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.shape != (TRACK2_CHANNELS,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain two finite velocity values")
    return array


@dataclass(frozen=True)
class Track2Normalization:
    """Separate affine statistics for observed and forecast velocity fields."""

    input_mean: np.ndarray
    target_mean: np.ndarray
    input_std: np.ndarray
    target_std: np.ndarray

    def __post_init__(self) -> None:
        for name in ("input_mean", "target_mean", "input_std", "target_std"):
            object.__setattr__(self, name, _finite_vector(getattr(self, name), name))
        if np.any(self.input_std <= 0.0) or np.any(self.target_std <= 0.0):
            raise ValueError(
                "velocity normalization standard deviations must be positive"
            )

    @classmethod
    def from_stats_tuple(cls, values: Sequence[Any]) -> Track2Normalization:
        if len(values) != 4:
            raise ValueError("normalization stats must contain four tensors")
        arrays = [
            value.detach().cpu().numpy()
            if torch.is_tensor(value)
            else np.asarray(value)
            for value in values
        ]
        return cls(
            input_mean=arrays[0][:TRACK2_CHANNELS],
            target_mean=arrays[1][:TRACK2_CHANNELS],
            input_std=arrays[2][:TRACK2_CHANNELS],
            target_std=arrays[3][:TRACK2_CHANNELS],
        )

    @classmethod
    def from_file(cls, path: Path) -> Track2Normalization:
        try:
            values = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            values = torch.load(path, map_location="cpu")
        return cls.from_stats_tuple(values)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> Track2Normalization:
        return cls(
            input_mean=value["input_mean"],
            target_mean=value["target_mean"],
            input_std=value["input_std"],
            target_std=value["target_std"],
        )

    def to_mapping(self) -> dict[str, list[float]]:
        return {
            "input_mean": self.input_mean.tolist(),
            "target_mean": self.target_mean.tolist(),
            "input_std": self.input_std.tolist(),
            "target_std": self.target_std.tolist(),
        }


@dataclass(frozen=True)
class Track2Geometry:
    """Static structured-grid tensors and their representation contract."""

    height: int
    width: int
    nodes: np.ndarray
    node_mask: np.ndarray
    node_weights: np.ndarray
    directed_edges: np.ndarray
    edge_gradient_weights: np.ndarray
    static_features: np.ndarray
    solid_mask: np.ndarray
    feature_names: tuple[str, ...]
    contract: dict[str, Any]

    def to_mapping(self) -> dict[str, Any]:
        return {
            "height": self.height,
            "width": self.width,
            "nodes": self.nodes,
            "node_mask": self.node_mask,
            "node_weights": self.node_weights,
            "directed_edges": self.directed_edges,
            "edge_gradient_weights": self.edge_gradient_weights,
            "static_features": self.static_features,
            "solid_mask": self.solid_mask,
            "feature_names": list(self.feature_names),
            "contract": self.contract,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> Track2Geometry:
        return cls(
            height=int(value["height"]),
            width=int(value["width"]),
            nodes=np.asarray(value["nodes"], dtype=np.float32),
            node_mask=np.asarray(value["node_mask"], dtype=np.float32),
            node_weights=np.asarray(value["node_weights"], dtype=np.float32),
            directed_edges=np.asarray(value["directed_edges"], dtype=np.int64),
            edge_gradient_weights=np.asarray(
                value["edge_gradient_weights"], dtype=np.float32
            ),
            static_features=np.asarray(value["static_features"], dtype=np.float32),
            solid_mask=np.asarray(value["solid_mask"], dtype=bool),
            feature_names=tuple(str(name) for name in value["feature_names"]),
            contract=dict(value["contract"]),
        )


def _minimum_point_distance(points: np.ndarray, references: np.ndarray) -> np.ndarray:
    if references.shape[0] == 0:
        raise ValueError("the simulation-derived solid mask is empty")
    result = np.empty(points.shape[0], dtype=np.float64)
    for first in range(0, points.shape[0], 4096):
        chunk = points[first : first + 4096]
        squared = np.sum(np.square(chunk[:, None, :] - references[None, :, :]), axis=-1)
        result[first : first + chunk.shape[0]] = np.sqrt(np.min(squared, axis=1))
    return result


def _structured_gradient_graph(
    nodes: np.ndarray, height: int, width: int
) -> tuple[np.ndarray, np.ndarray]:
    adjacency = [set() for _ in range(height * width)]
    for row in range(height):
        for col in range(width):
            index = row * width + col
            for other_row, other_col in (
                (row - 1, col),
                (row + 1, col),
                (row, col - 1),
                (row, col + 1),
            ):
                if 0 <= other_row < height and 0 <= other_col < width:
                    adjacency[index].add(other_row * width + other_col)
    directed_edges, weights, _ = compute_edge_gradient_weights_helper(
        nodes.astype(np.float64),
        np.full(height * width, 2, dtype=np.int64),
        adjacency,
    )
    return directed_edges.astype(np.int64), weights.astype(np.float32)


def build_track2_geometry(
    x: np.ndarray,
    y: np.ndarray,
    observed_solid_union: np.ndarray,
    *,
    collar_width: float = 0.01,
    subsampling: int = TRACK2_SUBSAMPLING,
) -> Track2Geometry:
    """Build fixed PCNO geometry from one validated simulation grid and mask."""

    x_array = np.asarray(x, dtype=np.float64)
    y_array = np.asarray(y, dtype=np.float64)
    observed_solid = np.asarray(observed_solid_union, dtype=bool)
    if (
        x_array.ndim != 2
        or y_array.shape != x_array.shape
        or observed_solid.shape != x_array.shape
    ):
        raise ValueError("x, y, and solid_mask must share shape [H, W]")
    if min(x_array.shape) < 2 or not np.all(np.isfinite(x_array + y_array)):
        raise ValueError("geometry must be finite with at least two nodes per axis")
    if not np.allclose(x_array, x_array[:1], rtol=0.0, atol=1.0e-10):
        raise ValueError("Track 2 x coordinates must be rectilinear")
    if not np.allclose(y_array, y_array[:, :1], rtol=0.0, atol=1.0e-10):
        raise ValueError("Track 2 y coordinates must be rectilinear")
    if not np.isfinite(collar_width) or collar_width <= 0.0:
        raise ValueError("collar_width must be positive and finite")

    height, width = x_array.shape
    x_min, x_max = float(x_array.min()), float(x_array.max())
    y_min, y_max = float(y_array.min()), float(y_array.max())
    if not x_min < x_max or not y_min < y_max:
        raise ValueError("coordinate bounds must be strictly ordered")

    physical_points = np.stack((x_array.reshape(-1), y_array.reshape(-1)), axis=-1)
    nodes = physical_points - np.asarray([x_min, y_min])
    rectangle = semantic_collar_fields(
        physical_points,
        {
            "inflow": np.asarray([[[x_min, y_min], [x_min, y_max]]]),
            "outflow": np.asarray([[[x_max, y_min], [x_max, y_max]]]),
            "bottom_farfield": np.asarray([[[x_min, y_min], [x_max, y_min]]]),
            "top_farfield": np.asarray([[[x_min, y_max], [x_max, y_max]]]),
        },
        physical_width=collar_width,
        family="realpde_track2_naca4418",
        boundary_geometry_provenance="released_fixed_rectilinear_coordinates",
        corner_policy="overlap_of_independent_semantic_collars",
        semantic_factorization="inflow_outflow_bottom_top_independent_channels",
    )
    observed_solid_flat = observed_solid.reshape(-1)
    airfoil_distance = _minimum_point_distance(
        physical_points, physical_points[observed_solid_flat]
    )
    airfoil_proximity = compact_cubic_collar(airfoil_distance, collar_width).astype(
        np.float32
    )
    solid_support = (airfoil_distance < collar_width).reshape(height, width)

    x_feature = 2.0 * (x_array.reshape(-1) - x_min) / (x_max - x_min) - 1.0
    y_feature = 2.0 * (y_array.reshape(-1) - y_min) / (y_max - y_min) - 1.0
    x_axis = x_array[0]
    y_axis = y_array[:, 0]
    area = np.abs(np.outer(np.gradient(y_axis), np.gradient(x_axis)))
    if not float(area.sum()) > 0.0:
        raise ValueError("quadrature mass must be positive")
    node_weights = area.reshape(-1, 1) / area.sum()
    density = node_weights[:, 0] / np.mean(node_weights[:, 0])
    static_features = np.column_stack(
        (
            x_feature,
            y_feature,
            density,
            observed_solid_flat.astype(np.float32),
            rectangle.values,
            airfoil_proximity,
        )
    ).astype(np.float32)
    feature_names = (
        "coordinate_x",
        "coordinate_y",
        "quadrature_density",
        "solid_mask",
        "inflow_collar",
        "outflow_collar",
        "bottom_farfield_collar",
        "top_farfield_collar",
        "airfoil_proximity",
    )
    directed_edges, edge_weights = _structured_gradient_graph(nodes, height, width)
    contract = {
        "schema": TRACK2_GEOMETRY_SCHEMA,
        "source": "released_simulation_x_y_and_time_invariant_zero_velocity_mask",
        "subsampling": int(subsampling),
        "shape": [height, width],
        "domain_origin": [x_min, y_min],
        "domain_lengths": [x_max - x_min, y_max - y_min],
        "observed_solid_union_node_count": int(observed_solid.sum()),
        "solid_support_node_count": int(solid_support.sum()),
        "solid_support_policy": (
            "observed_training_union_dilated_by_boundary_collar_width"
        ),
        "solid_policy": (
            "hard_zero_velocity_on_input_persistent_zero_nodes_within_"
            "simulation_derived_support"
        ),
        "quadrature_policy": (
            "per_sample_zero_inferred_solid_weights_then_renormalize"
        ),
        "piv_missing_policy": "causal_input_encoding_without_hard_fill",
        "boundary_field_names": list(feature_names[4:]),
        "boundary_collar_width": float(collar_width),
        "boundary_collar_units": "released_coordinate_units",
        "physical_boundary_policy_changed": False,
    }
    return Track2Geometry(
        height=height,
        width=width,
        nodes=nodes.astype(np.float32),
        node_mask=np.ones((height * width, 1), dtype=np.float32),
        node_weights=node_weights.astype(np.float32),
        directed_edges=directed_edges,
        edge_gradient_weights=edge_weights,
        static_features=static_features,
        solid_mask=solid_support,
        feature_names=feature_names,
        contract=contract,
    )


class RealPDETrack2PCNO(nn.Module):
    """Direct 20-to-20 velocity forecaster with a persistence residual."""

    def __init__(
        self,
        *,
        normalization: Track2Normalization,
        geometry: Track2Geometry,
        residual_scale: np.ndarray,
        n_modes: Sequence[int] = (20, 10),
        layers: Sequence[int] = (96, 96, 96, 96, 96),
        fc_dim: int = 128,
        zero_initialize: bool = True,
    ) -> None:
        super().__init__()
        if len(n_modes) != 2 or any(int(value) < 1 for value in n_modes):
            raise ValueError("n_modes must contain two positive integers")
        if len(layers) < 2 or any(int(value) < 1 for value in layers):
            raise ValueError("layers must contain at least two positive widths")
        scale = np.asarray(residual_scale, dtype=np.float64)
        if scale.shape != (TRACK2_OUTPUT_STEPS, TRACK2_CHANNELS):
            raise ValueError("residual_scale must have shape [20, 2]")
        if not np.all(np.isfinite(scale)) or np.any(scale <= 0.0):
            raise ValueError("residual_scale must be positive and finite")

        lengths = geometry.contract["domain_lengths"]
        modes = torch.as_tensor(
            compute_Fourier_modes(2, [int(v) for v in n_modes], lengths),
            dtype=torch.float32,
        )
        dynamic_features = TRACK2_INPUT_STEPS * TRACK2_CHANNELS + 2
        self.backbone = PCNO(
            2,
            modes,
            nmeasures=1,
            layers=[int(value) for value in layers],
            fc_dim=int(fc_dim),
            in_dim=dynamic_features + len(geometry.feature_names),
            out_dim=TRACK2_OUTPUT_STEPS * TRACK2_CHANNELS,
            act="gelu",
        )
        self.height = int(geometry.height)
        self.width = int(geometry.width)
        self.n_modes = tuple(int(value) for value in n_modes)
        self.layer_widths = tuple(int(value) for value in layers)
        self.projection_width = int(fc_dim)
        self.geometry_contract = dict(geometry.contract)
        self.static_feature_names = tuple(geometry.feature_names)
        self.quadrature_density_index = self.static_feature_names.index(
            "quadrature_density"
        )
        self.solid_feature_index = self.static_feature_names.index("solid_mask")
        self.airfoil_proximity_index = self.static_feature_names.index(
            "airfoil_proximity"
        )

        self.register_buffer(
            "input_mean", torch.as_tensor(normalization.input_mean, dtype=torch.float32)
        )
        self.register_buffer(
            "target_mean",
            torch.as_tensor(normalization.target_mean, dtype=torch.float32),
        )
        self.register_buffer(
            "input_std", torch.as_tensor(normalization.input_std, dtype=torch.float32)
        )
        self.register_buffer(
            "target_std", torch.as_tensor(normalization.target_std, dtype=torch.float32)
        )
        self.register_buffer(
            "residual_scale",
            torch.as_tensor(scale, dtype=torch.float32).reshape(1, 20, 1, 1, 2),
        )
        self.register_buffer("nodes", torch.as_tensor(geometry.nodes).unsqueeze(0))
        self.register_buffer(
            "node_mask", torch.as_tensor(geometry.node_mask).unsqueeze(0)
        )
        self.register_buffer(
            "node_weights", torch.as_tensor(geometry.node_weights).unsqueeze(0)
        )
        self.register_buffer(
            "directed_edges", torch.as_tensor(geometry.directed_edges).unsqueeze(0)
        )
        self.register_buffer(
            "edge_gradient_weights",
            torch.as_tensor(geometry.edge_gradient_weights).unsqueeze(0),
        )
        self.register_buffer(
            "static_features", torch.as_tensor(geometry.static_features).unsqueeze(0)
        )
        self.register_buffer(
            "solid_support_mask",
            torch.as_tensor(geometry.solid_mask, dtype=torch.bool),
        )
        support_indices = np.flatnonzero(geometry.solid_mask.reshape(-1))
        if support_indices.size == 0:
            raise ValueError("geometry solid support must be nonempty")
        support_distances = np.sqrt(
            np.sum(
                np.square(
                    geometry.nodes[:, None, :]
                    - geometry.nodes[support_indices][None, :, :]
                ),
                axis=-1,
            )
        )
        collar_kernel = compact_cubic_collar(
            support_distances,
            float(geometry.contract["boundary_collar_width"]),
        )
        self.register_buffer(
            "solid_support_indices",
            torch.as_tensor(support_indices, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "airfoil_collar_kernel",
            torch.as_tensor(collar_kernel, dtype=torch.float32),
            persistent=False,
        )
        with torch.no_grad():
            cached = self.backbone.prepare_fourier_tensors(
                self.nodes, self.node_weights
            )
        for name, value in zip(
            ("basis_cos", "basis_sin", "basis_zero"),
            cached[:3],
        ):
            self.register_buffer(name, value, persistent=False)
        if zero_initialize:
            self.zero_initialize_residual_head()

    def zero_initialize_residual_head(self) -> None:
        nn.init.zeros_(self.backbone.fc2.weight)
        nn.init.zeros_(self.backbone.fc2.bias)

    def normalization(self) -> Track2Normalization:
        return Track2Normalization(
            input_mean=self.input_mean.detach().cpu().numpy(),
            target_mean=self.target_mean.detach().cpu().numpy(),
            input_std=self.input_std.detach().cpu().numpy(),
            target_std=self.target_std.detach().cpu().numpy(),
        )

    def set_domain_scaling(
        self, normalization: Track2Normalization, residual_scale: np.ndarray
    ) -> None:
        scale = np.asarray(residual_scale, dtype=np.float64)
        if scale.shape != (20, 2) or np.any(scale <= 0.0):
            raise ValueError("residual_scale must be positive with shape [20, 2]")
        with torch.no_grad():
            self.input_mean.copy_(torch.as_tensor(normalization.input_mean))
            self.target_mean.copy_(torch.as_tensor(normalization.target_mean))
            self.input_std.copy_(torch.as_tensor(normalization.input_std))
            self.target_std.copy_(torch.as_tensor(normalization.target_std))
            self.residual_scale.copy_(
                torch.as_tensor(scale, dtype=self.residual_scale.dtype).reshape(
                    1, 20, 1, 1, 2
                )
            )

    def model_config(self) -> dict[str, Any]:
        return {
            "schema": TRACK2_MODEL_SCHEMA,
            "n_modes": list(self.n_modes),
            "layers": list(self.layer_widths),
            "fc_dim": self.projection_width,
            "input_steps": TRACK2_INPUT_STEPS,
            "output_steps": TRACK2_OUTPUT_STEPS,
            "learned_channels": ["u", "v"],
            "baseline": "raw_space_persistence",
            "residual_scaling": "per_lead_per_channel_rms",
            "static_feature_names": list(self.static_feature_names),
            "dynamic_feature_names": [
                *[
                    f"input_t{step:02d}_{channel}"
                    for step in range(20)
                    for channel in ("u", "v")
                ],
                "observed_zero_fraction",
                "last_frame_zero",
            ],
            "solid_mask_inference": (
                "persistent_raw_zero_intersect_simulation_derived_support"
            ),
            "fourier_basis_cache": (
                "nonpersistent_fixed_bases_with_per_sample_quadrature"
            ),
        }

    def input_to_raw(self, value: torch.Tensor) -> torch.Tensor:
        return value * self.input_std + self.input_mean

    def target_to_raw(self, value: torch.Tensor) -> torch.Tensor:
        return value * self.target_std + self.target_mean

    def raw_to_input(self, value: torch.Tensor) -> torch.Tensor:
        return (value - self.input_mean) / self.input_std

    def raw_to_target(self, value: torch.Tensor) -> torch.Tensor:
        return (value - self.target_mean) / self.target_std

    def _normalized_zero_mask(
        self,
        value: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> torch.Tensor:
        zero_norm = -mean / std
        return torch.logical_and(
            torch.isclose(value[..., 0], zero_norm[0], rtol=0.0, atol=1.0e-5),
            torch.isclose(value[..., 1], zero_norm[1], rtol=0.0, atol=1.0e-5),
        )

    def input_solid_mask(self, input_norm: torch.Tensor) -> torch.Tensor:
        missing = self._normalized_zero_mask(
            input_norm[..., :TRACK2_CHANNELS],
            self.input_mean,
            self.input_std,
        )
        return torch.logical_and(
            missing.all(dim=1),
            self.solid_support_mask[None],
        )

    def target_solid_mask(self, target_norm: torch.Tensor) -> torch.Tensor:
        missing = self._normalized_zero_mask(
            target_norm[..., :TRACK2_CHANNELS],
            self.target_mean,
            self.target_std,
        )
        return torch.logical_and(
            missing.all(dim=1),
            self.solid_support_mask[None],
        )

    def airfoil_proximity(self, solid_mask: torch.Tensor) -> torch.Tensor:
        solid_flat = solid_mask.reshape(solid_mask.shape[0], -1)
        support_values = solid_flat.index_select(1, self.solid_support_indices).to(
            self.airfoil_collar_kernel.dtype
        )
        return torch.amax(
            self.airfoil_collar_kernel[None] * support_values[:, None, :],
            dim=-1,
        )

    def _dynamic_node_weights(
        self, solid_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = solid_mask.shape[0]
        fluid = (~solid_mask).reshape(batch_size, -1, 1).to(self.node_weights.dtype)
        node_weights = self.node_weights.expand(batch_size, -1, -1) * fluid
        mass = node_weights.sum(dim=1, keepdim=True)
        if bool((mass <= 0.0).any()):
            raise ValueError("inferred solid mask leaves no fluid quadrature mass")
        node_weights = node_weights / mass
        density = node_weights * fluid.sum(dim=1, keepdim=True)
        return node_weights, density

    def _dynamic_static_features(
        self,
        solid_mask: torch.Tensor,
        density: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = solid_mask.shape[0]
        features = self.static_features.expand(batch_size, -1, -1).clone()
        features[..., self.quadrature_density_index] = density[..., 0]
        features[..., self.solid_feature_index] = solid_mask.reshape(batch_size, -1).to(
            features.dtype
        )
        features[..., self.airfoil_proximity_index] = self.airfoil_proximity(solid_mask)
        return features

    def _expanded_geometry(
        self,
        batch_size: int,
        node_weights: torch.Tensor | None = None,
    ):
        if node_weights is None:
            node_weights = self.node_weights.expand(batch_size, -1, -1)
        return [
            self.node_mask.expand(batch_size, -1, -1),
            self.nodes.expand(batch_size, -1, -1),
            node_weights,
            self.directed_edges.expand(batch_size, -1, -1),
            self.edge_gradient_weights.expand(batch_size, -1, -1),
        ]

    def _expanded_fourier(
        self,
        batch_size: int,
        node_weights: torch.Tensor | None = None,
    ):
        if node_weights is None:
            node_weights = self.node_weights.expand(batch_size, -1, -1)
        bases = tuple(
            value.expand(batch_size, *value.shape[1:])
            for value in (
                self.basis_cos,
                self.basis_sin,
                self.basis_zero,
            )
        )
        weighted = tuple(basis * node_weights.unsqueeze(2) for basis in bases)
        return (*bases, *weighted)

    def forward(self, input_norm: torch.Tensor) -> torch.Tensor:
        if input_norm.ndim != 5 or input_norm.shape[1] != TRACK2_INPUT_STEPS:
            raise ValueError("input_norm must have shape [B, 20, H, W, C]")
        if input_norm.shape[2:4] != (self.height, self.width):
            raise ValueError("input spatial shape does not match fixed geometry")
        if input_norm.shape[-1] < TRACK2_CHANNELS:
            raise ValueError("input_norm must contain u and v")
        flow = input_norm[..., :TRACK2_CHANNELS]
        if not bool(torch.isfinite(flow).all()):
            raise ValueError("input_norm velocity channels must be finite")
        batch_size = flow.shape[0]
        history = flow.permute(0, 2, 3, 1, 4).reshape(
            batch_size, self.height * self.width, -1
        )
        missing = self._normalized_zero_mask(
            flow,
            self.input_mean,
            self.input_std,
        )
        solid_mask = torch.logical_and(
            missing.all(dim=1),
            self.solid_support_mask[None],
        )
        missing_fraction = missing.float().mean(dim=1).reshape(batch_size, -1, 1)
        last_missing = missing[:, -1].float().reshape(batch_size, -1, 1)
        node_weights, density = self._dynamic_node_weights(solid_mask)
        static_features = self._dynamic_static_features(solid_mask, density)
        features = torch.cat(
            (
                history,
                missing_fraction,
                last_missing,
                static_features,
            ),
            dim=-1,
        )
        residual = self.backbone(
            features,
            self._expanded_geometry(batch_size, node_weights),
            fourier_tensors=self._expanded_fourier(batch_size, node_weights),
        )
        residual = residual.reshape(
            batch_size, self.height, self.width, TRACK2_OUTPUT_STEPS, 2
        ).permute(0, 3, 1, 2, 4)
        persistence = self.raw_to_target(self.input_to_raw(flow[:, -1]))
        prediction = persistence[:, None] + self.residual_scale * residual
        zero_target = -self.target_mean / self.target_std
        return torch.where(
            solid_mask[:, None, :, :, None],
            zero_target.reshape(1, 1, 1, 1, 2),
            prediction,
        )

    def forward_with_pressure(self, input_norm: torch.Tensor) -> torch.Tensor:
        velocity = self(input_norm)
        return torch.cat((velocity, torch.zeros_like(velocity[..., :1])), dim=-1)


def track2_error_tensors(
    prediction_norm: torch.Tensor,
    target_norm: torch.Tensor,
    normalization: Track2Normalization | RealPDETrack2PCNO,
) -> dict[str, torch.Tensor]:
    """Differentiable per-sample versions of the three forecast errors."""

    if prediction_norm.shape != target_norm.shape or prediction_norm.ndim != 5:
        raise ValueError("prediction and target must share shape [B, T, H, W, 2]")
    if isinstance(normalization, RealPDETrack2PCNO):
        target_std = normalization.target_std
        target_mean = normalization.target_mean
    else:
        target_std = torch.as_tensor(
            normalization.target_std,
            dtype=prediction_norm.dtype,
            device=prediction_norm.device,
        )
        target_mean = torch.as_tensor(
            normalization.target_mean,
            dtype=prediction_norm.dtype,
            device=prediction_norm.device,
        )
    prediction = prediction_norm * target_std + target_mean
    target = target_norm * target_std + target_mean
    batch_size = prediction.shape[0]
    relative_l2 = torch.linalg.vector_norm(
        (prediction - target).reshape(batch_size, -1), dim=1
    ) / torch.linalg.vector_norm(target.reshape(batch_size, -1), dim=1).clamp_min(
        1.0e-8
    )
    pred_tke = 0.5 * (
        prediction[..., 0].var(dim=1, unbiased=False)
        + prediction[..., 1].var(dim=1, unbiased=False)
    )
    target_tke = 0.5 * (
        target[..., 0].var(dim=1, unbiased=False)
        + target[..., 1].var(dim=1, unbiased=False)
    )
    tke = torch.linalg.vector_norm(
        (pred_tke - target_tke).reshape(batch_size, -1), dim=1
    ) / torch.linalg.vector_norm(target_tke.reshape(batch_size, -1), dim=1).clamp_min(
        1.0e-8
    )

    height, width = prediction.shape[2:4]
    probe_y = [
        16 + 2 * offset for offset in range(-4, 5) if 0 <= 16 + 2 * offset < height
    ]
    probe_errors = []
    for probe_x in (13, 21, 29, 37):
        if probe_x >= width or not probe_y:
            continue
        pred_probe = prediction[:, :, probe_y, probe_x, :].mean(dim=1)
        target_probe = target[:, :, probe_y, probe_x, :].mean(dim=1)
        probe_errors.append(
            torch.linalg.vector_norm(
                (pred_probe - target_probe).reshape(batch_size, -1), dim=1
            )
            / torch.linalg.vector_norm(
                target_probe.reshape(batch_size, -1), dim=1
            ).clamp_min(1.0e-8)
        )
    mvpe = (
        torch.stack(probe_errors, dim=0).mean(dim=0)
        if probe_errors
        else torch.zeros_like(relative_l2)
    )
    return {"rel_l2": relative_l2, "tke": tke, "mvpe": mvpe}


def track2_training_loss(
    model: RealPDETrack2PCNO,
    prediction_norm: torch.Tensor,
    target_norm: torch.Tensor,
    *,
    relative_l2_weight: float = 0.05,
    tke_weight: float = 0.05,
    mvpe_weight: float = 0.05,
    boundary_weight: float = 0.10,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Metric-aware loss anchored by normalized per-element MSE."""

    squared = torch.square(prediction_norm - target_norm)
    target_solid = model.target_solid_mask(target_norm)
    airfoil = model.airfoil_proximity(target_solid).reshape(
        target_norm.shape[0], model.height, model.width
    )
    spatial_weight = 1.0 + float(boundary_weight) * airfoil
    normalized_mse = torch.mean(squared * spatial_weight[:, None, :, :, None])
    errors = track2_error_tensors(prediction_norm, target_norm, model)
    loss = (
        normalized_mse
        + float(relative_l2_weight) * errors["rel_l2"].mean()
        + float(tke_weight) * errors["tke"].mean()
        + float(mvpe_weight) * errors["mvpe"].mean()
    )
    metrics = {"normalized_mse": normalized_mse, **errors}
    return loss, metrics


def track2_forecast_score(errors: Mapping[str, float]) -> float:
    values = [
        100.0 / (1.0 + 0.5 * max(float(errors[name]), 0.0))
        for name in ("rel_l2", "tke", "mvpe")
    ]
    return float(np.mean(values))


def parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def build_track2_model_from_payload(
    payload: Mapping[str, Any],
    *,
    device: torch.device | str = "cpu",
) -> RealPDETrack2PCNO:
    """Reconstruct a frozen Track 2 PCNO from a self-contained model payload."""

    if payload.get("schema") != "realpde_track2_pcno_model_v2":
        raise ValueError("unsupported Track 2 PCNO model payload schema")
    config = payload["model_config"]
    geometry = Track2Geometry.from_mapping(payload["geometry"])
    normalization = Track2Normalization.from_mapping(payload["normalization"])
    model = RealPDETrack2PCNO(
        normalization=normalization,
        geometry=geometry,
        residual_scale=np.asarray(payload["residual_scale"], dtype=np.float32),
        n_modes=tuple(config["n_modes"]),
        layers=tuple(config["layers"]),
        fc_dim=int(config["fc_dim"]),
        zero_initialize=False,
    ).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    return model


@dataclass(frozen=True)
class Track2FileMetadata:
    path: Path
    reynolds: int
    angle_of_attack: int
    frames: int

    def to_mapping(self) -> dict[str, Any]:
        return {
            "name": self.path.name,
            "reynolds": self.reynolds,
            "angle_of_attack": self.angle_of_attack,
            "frames": self.frames,
        }


def inspect_track2_file(path: Path) -> Track2FileMetadata:
    with h5py.File(path, "r") as handle:
        required = {"u", "v", "x", "y", "re", "aoa"}
        missing = sorted(required - set(handle.keys()))
        if missing:
            raise ValueError(f"{path} is missing datasets: {missing}")
        if handle["u"].shape != handle["v"].shape or handle["u"].ndim != 3:
            raise ValueError(f"{path} u and v must share shape [T, H, W]")
        if (
            handle["x"].shape != handle["u"].shape[1:]
            or handle["y"].shape != handle["x"].shape
        ):
            raise ValueError(f"{path} coordinate arrays do not match velocity grids")
        return Track2FileMetadata(
            path=path,
            reynolds=int(handle["re"][()]),
            angle_of_attack=int(handle["aoa"][()]),
            frames=int(handle["u"].shape[0]),
        )


def discover_track2_files(data_dir: Path) -> list[Track2FileMetadata]:
    paths = sorted(data_dir.glob("*.h5"))
    if not paths:
        raise FileNotFoundError(f"no Track 2 HDF5 files under {data_dir}")
    metadata = [inspect_track2_file(path) for path in paths]
    return sorted(
        metadata,
        key=lambda item: (item.angle_of_attack, item.reynolds, item.path.name),
    )


def split_track2_files(
    metadata: Sequence[Track2FileMetadata],
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[list[Track2FileMetadata], list[Track2FileMetadata]]:
    """Hold out whole operating conditions, stratified by angle of attack."""

    if not 0.0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must lie strictly between 0 and 0.5")
    groups: dict[int, list[Track2FileMetadata]] = {}
    for item in metadata:
        groups.setdefault(item.angle_of_attack, []).append(item)
    generator = np.random.default_rng(seed)
    train: list[Track2FileMetadata] = []
    validation: list[Track2FileMetadata] = []
    for angle in sorted(groups):
        group = sorted(groups[angle], key=lambda item: (item.reynolds, item.path.name))
        if len(group) < 2:
            train.extend(group)
            continue
        count = max(1, round(validation_fraction * len(group)))
        selected = {int(index) for index in generator.permutation(len(group))[:count]}
        for index, item in enumerate(group):
            (validation if index in selected else train).append(item)
    if not train or not validation:
        raise ValueError(
            "trajectory split must produce nonempty train and validation sets"
        )
    return train, validation


def _load_track2_geometry_source(
    path: Path,
    *,
    subsampling: int,
    time_chunk: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        u = handle["u"]
        v = handle["v"]
        solid = np.ones(u.shape[1:], dtype=bool)
        for first in range(0, u.shape[0], time_chunk):
            last = min(first + time_chunk, u.shape[0])
            solid &= np.all(np.asarray(u[first:last]) == 0.0, axis=0)
            solid &= np.all(np.asarray(v[first:last]) == 0.0, axis=0)
        x = np.asarray(handle["x"])[::subsampling, ::subsampling]
        y = np.asarray(handle["y"])[::subsampling, ::subsampling]
    return x, y, solid[::subsampling, ::subsampling]


def load_track2_geometry_from_simulations(
    paths: Sequence[Path],
    *,
    subsampling: int = TRACK2_SUBSAMPLING,
    collar_width: float = 0.01,
    time_chunk: int = 64,
) -> Track2Geometry:
    """Build one grid and an AoA-varying solid support from simulations."""

    if subsampling < 1 or time_chunk < 1:
        raise ValueError("subsampling and time_chunk must be positive")
    sources = [Path(path) for path in paths]
    if not sources:
        raise ValueError("paths must contain at least one simulation")

    reference_x = reference_y = observed_union = None
    for path in sources:
        x, y, solid = _load_track2_geometry_source(
            path,
            subsampling=subsampling,
            time_chunk=time_chunk,
        )
        if reference_x is None:
            reference_x = x
            reference_y = y
            observed_union = solid.copy()
            continue
        if not np.allclose(x, reference_x, rtol=0.0, atol=3.0e-5):
            raise ValueError(f"{path} x coordinates drift from geometry")
        if not np.allclose(y, reference_y, rtol=0.0, atol=3.0e-5):
            raise ValueError(f"{path} y coordinates drift from geometry")
        observed_union |= solid

    geometry = build_track2_geometry(
        reference_x,
        reference_y,
        observed_union,
        collar_width=collar_width,
        subsampling=subsampling,
    )
    geometry.contract["solid_support_source_trajectory_count"] = len(sources)
    return geometry


def load_track2_geometry_from_simulation(
    path: Path,
    *,
    subsampling: int = TRACK2_SUBSAMPLING,
    collar_width: float = 0.01,
    time_chunk: int = 64,
) -> Track2Geometry:
    """Build a dynamic-solid geometry support from one simulation."""

    return load_track2_geometry_from_simulations(
        [path],
        subsampling=subsampling,
        collar_width=collar_width,
        time_chunk=time_chunk,
    )


class Track2WindowDataset(Dataset):
    """Eager host-memory cache of normalized 20-to-20 trajectory windows."""

    def __init__(
        self,
        metadata: Sequence[Track2FileMetadata],
        normalization: Track2Normalization,
        *,
        stride: int,
        subsampling: int = TRACK2_SUBSAMPLING,
        expected_geometry: Track2Geometry | None = None,
        validate_solid: bool = False,
    ) -> None:
        if stride < 1 or subsampling < 1:
            raise ValueError("stride and subsampling must be positive")
        if not metadata:
            raise ValueError("metadata must contain at least one trajectory")
        self.normalization = normalization
        self.trajectories: list[np.ndarray] = []
        self.metadata = list(metadata)
        self.windows: list[tuple[int, int]] = []
        expected_x = expected_y = expected_solid = None
        if expected_geometry is not None:
            origin = np.asarray(expected_geometry.contract["domain_origin"])
            nodes = expected_geometry.nodes.reshape(
                expected_geometry.height, expected_geometry.width, 2
            )
            expected_x = nodes[..., 0] + origin[0]
            expected_y = nodes[..., 1] + origin[1]
            expected_solid = expected_geometry.solid_mask

        for trajectory_index, item in enumerate(self.metadata):
            with h5py.File(item.path, "r") as handle:
                u = np.asarray(
                    handle["u"][:, ::subsampling, ::subsampling], dtype=np.float32
                )
                v = np.asarray(
                    handle["v"][:, ::subsampling, ::subsampling], dtype=np.float32
                )
                if expected_geometry is not None:
                    x = np.asarray(handle["x"])[::subsampling, ::subsampling]
                    y = np.asarray(handle["y"])[::subsampling, ::subsampling]
                    if not np.allclose(x, expected_x, rtol=0.0, atol=3.0e-5):
                        raise ValueError(
                            f"{item.path} x coordinates drift from geometry"
                        )
                    if not np.allclose(y, expected_y, rtol=0.0, atol=3.0e-5):
                        raise ValueError(
                            f"{item.path} y coordinates drift from geometry"
                        )
            values = np.stack((u, v), axis=-1)
            if values.shape[0] < TRACK2_INPUT_STEPS + TRACK2_OUTPUT_STEPS:
                raise ValueError(f"{item.path} has fewer than 40 frames")
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{item.path} contains nonfinite velocity values")
            if validate_solid:
                solid = np.all(values == 0.0, axis=(0, 3))
                if not np.any(solid):
                    raise ValueError(f"{item.path} simulation solid mask is empty")
                if np.any(solid & ~expected_solid):
                    raise ValueError(
                        f"{item.path} simulation solid mask leaves its support"
                    )
            self.trajectories.append(values)
            maximum = values.shape[0] - TRACK2_INPUT_STEPS - TRACK2_OUTPUT_STEPS
            self.windows.extend(
                (trajectory_index, start) for start in range(0, maximum + 1, stride)
            )
        if not self.windows:
            raise ValueError("dataset contains no complete 20-to-20 windows")

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        trajectory_index, start = self.windows[index]
        values = torch.from_numpy(
            self.trajectories[trajectory_index][
                start : start + TRACK2_INPUT_STEPS + TRACK2_OUTPUT_STEPS
            ]
        )
        input_mean = torch.as_tensor(self.normalization.input_mean, dtype=values.dtype)
        target_mean = torch.as_tensor(
            self.normalization.target_mean, dtype=values.dtype
        )
        input_std = torch.as_tensor(self.normalization.input_std, dtype=values.dtype)
        target_std = torch.as_tensor(self.normalization.target_std, dtype=values.dtype)
        input_norm = (values[:TRACK2_INPUT_STEPS] - input_mean) / input_std
        target_norm = (values[TRACK2_INPUT_STEPS:] - target_mean) / target_std
        return input_norm, target_norm


def fit_track2_frame_normalization(
    metadata: Sequence[Track2FileMetadata],
    *,
    subsampling: int = TRACK2_SUBSAMPLING,
) -> Track2Normalization:
    """Fit sim-only velocity statistics without pooling simulation and PIV."""

    total = np.zeros(2, dtype=np.float64)
    squared = np.zeros(2, dtype=np.float64)
    count = 0
    for item in metadata:
        with h5py.File(item.path, "r") as handle:
            for channel in ("u", "v"):
                values = np.asarray(
                    handle[channel][:, ::subsampling, ::subsampling],
                    dtype=np.float64,
                )
                index = 0 if channel == "u" else 1
                total[index] += float(values.sum(dtype=np.float64))
                squared[index] += float(np.square(values).sum(dtype=np.float64))
                if index == 0:
                    count += values.size
    mean = total / count
    variance = np.maximum(squared / count - np.square(mean), 1.0e-12)
    std = np.sqrt(variance)
    return Track2Normalization(
        input_mean=mean,
        target_mean=mean,
        input_std=std,
        target_std=std,
    )


def estimate_track2_residual_scale(
    dataset: Track2WindowDataset,
    *,
    maximum_windows: int = 4096,
    floor: float = 1.0e-3,
) -> np.ndarray:
    """Estimate per-lead normalized persistence-residual RMS."""

    if maximum_windows < 1 or floor <= 0.0:
        raise ValueError("maximum_windows and floor must be positive")
    indices = np.linspace(
        0,
        len(dataset) - 1,
        num=min(maximum_windows, len(dataset)),
        dtype=np.int64,
    )
    squared = np.zeros((TRACK2_OUTPUT_STEPS, 2), dtype=np.float64)
    count = 0
    normalization = dataset.normalization
    for index in indices:
        trajectory_index, start = dataset.windows[int(index)]
        values = dataset.trajectories[trajectory_index]
        last_raw = values[start + TRACK2_INPUT_STEPS - 1]
        target_raw = values[
            start + TRACK2_INPUT_STEPS : start
            + TRACK2_INPUT_STEPS
            + TRACK2_OUTPUT_STEPS
        ]
        baseline = (last_raw - normalization.target_mean) / normalization.target_std
        target = (target_raw - normalization.target_mean) / normalization.target_std
        residual = target - baseline[None]
        squared += np.square(residual, dtype=np.float64).sum(axis=(1, 2))
        count += residual.shape[1] * residual.shape[2]
    return np.maximum(np.sqrt(squared / count), floor).astype(np.float32)
