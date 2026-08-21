"""Frozen, single-location interventions for the W26-L2 PCNO filter screen.

The helpers in this module deliberately replay the maintained core-PCNO order.
They are inference diagnostics: the DCT filter crosses through NumPy/SciPy and
is not a trainable layer.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from pcno.pcno import compute_gradient, compute_neighbor_degree
from utility.time_dependent_no.pcno_resolution_pathways import (
    PreparedGraphBallOperator,
    build_graph_ball_operator,
    graph_ball_average,
    prepare_graph_ball_operator,
)
from utility.time_dependent_no.pcno_shock_representation import (
    FIXED_PHYSICAL_RADIUS,
    StructuredCellGrid,
    physical_cosine_filter,
)

NATIVE_REPLAY = "native_replay"
REGISTERED_INTERVENTIONS = (
    "R_grad_fixed_physical",
    "S_grad",
    "H_grad",
    "S_differential_output",
    "S_pointwise_output",
    "S_preactivation",
    "S_postactivation",
)


def filter_channel_first(
    values: torch.Tensor,
    grid: StructuredCellGrid,
    *,
    kind: str,
) -> torch.Tensor:
    """Filter a ``[batch, channels, nodes]`` tensor on structured spatial axes.

    ``kind='zero'`` is a true tensor-object bypass. Nonzero filters are frozen
    inference operations and intentionally have no autograd contract.
    """

    if values.ndim != 3 or values.shape[-1] != grid.nx * grid.ny:
        raise ValueError("values must have shape [batch, channels, grid nodes]")
    if kind == "zero":
        return values
    if torch.is_grad_enabled() and values.requires_grad:
        raise RuntimeError("physical filter screen is inference-only")
    spatial = (
        values.detach()
        .reshape(values.shape[0], values.shape[1], grid.ny, grid.nx)
        .permute(2, 3, 0, 1)
        .cpu()
        .numpy()
    )
    filtered = physical_cosine_filter(spatial, grid, kind=kind)
    channel_first = np.ascontiguousarray(filtered.transpose(2, 3, 0, 1))
    return torch.as_tensor(
        channel_first,
        dtype=values.dtype,
        device=values.device,
    ).reshape_as(values)


def prepare_fixed_physical_operator(
    grid: StructuredCellGrid,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
    radius: float = FIXED_PHYSICAL_RADIUS,
) -> PreparedGraphBallOperator:
    """Build the registered volume-weighted geodesic-ball replacement."""

    operator = build_graph_ball_operator(
        grid.nodes,
        grid.geometry.edges,
        grid.cell_volumes,
        radius=radius,
    )
    return prepare_graph_ball_operator(operator, device=device, dtype=dtype)


def _validate_pcno(
    backbone: torch.nn.Module,
    model_input: torch.Tensor,
    aux: Sequence[torch.Tensor],
    selected_layer: int,
) -> int:
    required = ("fc0", "sp_convs", "ws", "gws", "fc2", "act", "fc_dim")
    if any(not hasattr(backbone, name) for name in required):
        raise TypeError("backbone does not expose the maintained PCNO modules")
    if model_input.ndim != 3 or len(aux) != 5:
        raise ValueError("model_input and aux do not satisfy the PCNO contract")
    lengths = (len(backbone.sp_convs), len(backbone.ws), len(backbone.gws))
    if len(set(lengths)) != 1:
        raise ValueError("PCNO branch module lists must have equal lengths")
    if not 0 <= selected_layer < lengths[0]:
        raise ValueError("selected_layer is outside the PCNO block range")
    return lengths[0]


def _decode(backbone: torch.nn.Module, hidden: torch.Tensor) -> torch.Tensor:
    decoded = hidden.permute(0, 2, 1)
    if backbone.fc_dim > 0:
        decoded = backbone.fc1(decoded)
        if backbone.act is not None:
            decoded = backbone.act(decoded)
    return backbone.fc2(decoded)


@torch.inference_mode()
def replay_pcno_intervention(
    backbone: torch.nn.Module,
    model_input: torch.Tensor,
    aux: Sequence[torch.Tensor],
    grid: StructuredCellGrid,
    *,
    intervention: str,
    selected_layer: int = 0,
    fixed_operator: PreparedGraphBallOperator | None = None,
    fourier_tensors: Sequence[torch.Tensor] | None = None,
) -> torch.Tensor:
    """Replay PCNO with exactly one registered intermediate intervention.

    ``NATIVE_REPLAY`` exists only for closure against ``backbone(...)``. The
    registered zero arm must call the native model directly and must not use
    this function.
    """

    allowed = (NATIVE_REPLAY,) + REGISTERED_INTERVENTIONS
    if intervention not in allowed:
        raise ValueError(f"unknown intervention {intervention!r}")
    block_count = _validate_pcno(backbone, model_input, aux, selected_layer)
    if intervention == "S_postactivation" and (
        selected_layer == block_count - 1 or backbone.act is None
    ):
        raise ValueError("postactivation intervention requires a nonfinal active block")
    if intervention == "R_grad_fixed_physical" and fixed_operator is None:
        raise ValueError("fixed physical support requires a prepared operator")

    _, nodes, node_weights, directed_edges, edge_gradient_weights = aux
    if fourier_tensors is None:
        fourier_tensors = backbone.prepare_fourier_tensors(nodes, node_weights)
    if len(fourier_tensors) != 6:
        raise ValueError("fourier_tensors must contain six basis tensors")
    hidden = backbone.fc0(model_input).permute(0, 2, 1)
    neighbor_degree = compute_neighbor_degree(
        directed_edges,
        hidden.shape[-1],
        dtype=hidden.dtype,
    )

    for layer, (spectral_layer, pointwise_layer, differential_layer) in enumerate(
        zip(backbone.sp_convs, backbone.ws, backbone.gws, strict=True)
    ):
        spectral = spectral_layer(hidden, *fourier_tensors)
        pointwise = pointwise_layer(hidden)
        selected = layer == selected_layer and intervention != NATIVE_REPLAY

        if selected and intervention in {
            "R_grad_fixed_physical",
            "S_grad",
            "H_grad",
        }:
            raw_gradient = compute_gradient(
                hidden,
                directed_edges,
                edge_gradient_weights,
            )
            if intervention == "R_grad_fixed_physical":
                aggregated = graph_ball_average(raw_gradient, fixed_operator)
            else:
                aggregated = filter_channel_first(
                    raw_gradient,
                    grid,
                    kind="smooth" if intervention == "S_grad" else "hard",
                )
            differential = differential_layer.gw2(
                differential_layer.geo_act(differential_layer.gw1 * aggregated)
            )
        else:
            differential = differential_layer(
                hidden,
                directed_edges,
                edge_gradient_weights,
                neighbor_degree=neighbor_degree,
                flat_edge_indices=None,
            )

        if selected and intervention == "S_differential_output":
            differential = filter_channel_first(differential, grid, kind="smooth")
        if selected and intervention == "S_pointwise_output":
            pointwise = filter_channel_first(pointwise, grid, kind="smooth")

        combined = spectral + pointwise + differential
        if selected and intervention == "S_preactivation":
            combined = filter_channel_first(combined, grid, kind="smooth")

        if backbone.act is not None and layer != block_count - 1:
            activated = backbone.act(combined)
            if selected and intervention == "S_postactivation":
                activated = filter_channel_first(activated, grid, kind="smooth")
            hidden = hidden + activated
        else:
            hidden = combined

    return _decode(backbone, hidden)
