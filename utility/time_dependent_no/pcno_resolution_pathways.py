"""Frozen-PCNO pathway diagnostics for nested-resolution inputs.

This module replays the additive PCNO blocks without changing parameters.  It
keeps latent quantities explicitly dimensionless and leaves physical output
metrics to the experiment runner, where checkpoint residual scales and finite-
volume cell volumes are available.
"""

from __future__ import annotations

import heapq
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from pcno.pcno import (
    compute_Fourier_bases,
    compute_gradient,
    graph_neighbor_average,
)

BRANCH_NAMES = ("spectral", "pointwise", "differential")
DEFAULT_INPUT_GROUPS = {
    "coordinates": (0, 2),
    "quadrature_density": (2, 3),
    "normalized_state": (3, 7),
    "node_type": (7, 11),
    "normalized_mach": (11, 12),
}
DEFAULT_WAVELENGTH_BANDS = (
    (0.05, 0.125),
    (0.125, 0.25),
    (0.25, 0.5),
    (0.5, np.inf),
)

_GRAPH_BALL_MESSAGE_CHUNK_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class GraphBallOperator:
    """Deterministic row-stochastic graph-geodesic ball average."""

    node_count: int
    radius: float
    geometry_tolerance: float
    undirected_edge_count: int
    targets: np.ndarray
    sources: np.ndarray
    coefficients: np.ndarray
    distances: np.ndarray
    maximum_row_sum_error: float
    minimum_coefficient: float
    maximum_neighbors: int


@dataclass(frozen=True)
class PreparedGraphBallOperator:
    """One explicitly materialized graph-ball operator for a device and dtype."""

    node_count: int
    radius: float
    geometry_tolerance: float
    targets: torch.Tensor
    sources: torch.Tensor
    coefficients: torch.Tensor


def graph_geometry_tolerance(nodes: np.ndarray) -> float:
    """Return a translation-invariant tolerance from source precision and span."""

    source = np.asarray(nodes)
    if source.ndim != 2 or source.shape[0] < 1:
        raise ValueError("nodes must have finite shape [N,D]")
    if not np.issubdtype(source.dtype, np.floating):
        precision = np.finfo(np.float64).eps
    else:
        precision = np.finfo(source.dtype).eps
    positions = np.asarray(source, dtype=np.float64)
    if not np.isfinite(positions).all():
        raise ValueError("nodes must be finite")
    span = float(np.max(np.ptp(positions, axis=0)))
    scale = max(span, np.finfo(np.float64).tiny)
    return float(64.0 * precision * scale)


def _graph_adjacency(
    nodes: np.ndarray, edges: np.ndarray
) -> tuple[list[list[tuple[int, float]]], np.ndarray]:
    positions = np.asarray(nodes, dtype=np.float64)
    connectivity = np.asarray(edges, dtype=np.int64)
    if positions.ndim != 2 or positions.shape[0] < 1:
        raise ValueError("nodes must have finite shape [N,D]")
    if not np.isfinite(positions).all():
        raise ValueError("nodes must be finite")
    if connectivity.ndim != 2 or connectivity.shape[1] != 2:
        raise ValueError("edges must have shape [E,2]")
    if connectivity.size and (
        connectivity.min() < 0 or connectivity.max() >= positions.shape[0]
    ):
        raise ValueError("edge index lies outside nodes")
    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(positions.shape[0])]
    undirected: set[tuple[int, int]] = set()
    for left_value, right_value in connectivity:
        left, right = int(left_value), int(right_value)
        if left == right:
            continue
        edge = (left, right) if left < right else (right, left)
        if edge in undirected:
            continue
        length = float(np.linalg.norm(positions[left] - positions[right]))
        if not np.isfinite(length) or length <= 0.0:
            raise ValueError("graph edges must have positive physical length")
        undirected.add(edge)
        adjacency[left].append((right, length))
        adjacency[right].append((left, length))
    for neighbors in adjacency:
        neighbors.sort(key=lambda item: item[0])
    canonical_edges = np.asarray(sorted(undirected), dtype=np.int64).reshape(-1, 2)
    return adjacency, canonical_edges


def graph_two_hop_physical_extents(
    nodes: np.ndarray,
    edges: np.ndarray,
) -> np.ndarray:
    """Return maximum shortest path length constrained to at most two hops."""

    adjacency, _ = _graph_adjacency(nodes, edges)
    extents = np.zeros(len(adjacency), dtype=np.float64)
    for source in range(len(adjacency)):
        best: dict[tuple[int, int], float] = {(source, 0): 0.0}
        reached = {source: 0.0}
        frontier = [(source, 0.0)]
        for hop in (1, 2):
            next_frontier: list[tuple[int, float]] = []
            for node, distance in frontier:
                for neighbor, edge_length in adjacency[node]:
                    candidate = distance + edge_length
                    key = (neighbor, hop)
                    if candidate >= best.get(key, float("inf")):
                        continue
                    best[key] = candidate
                    reached[neighbor] = min(
                        reached.get(neighbor, float("inf")), candidate
                    )
                    next_frontier.append((neighbor, candidate))
            frontier = next_frontier
        extents[source] = max(reached.values())
    return extents


def graph_two_hop_physical_radius(
    nodes: np.ndarray,
    edges: np.ndarray,
    *,
    node_mask: np.ndarray,
) -> float:
    """Return the median two-hop physical extent on a declared node subset."""

    selected = np.asarray(node_mask, dtype=bool).reshape(-1)
    if selected.shape != (np.asarray(nodes).shape[0],) or not np.any(selected):
        raise ValueError("node_mask must select at least one aligned node")
    extents = graph_two_hop_physical_extents(nodes, edges)
    radius = float(np.median(extents[selected]))
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("two-hop physical radius must be positive")
    return radius


def build_graph_ball_operator(
    nodes: np.ndarray,
    edges: np.ndarray,
    node_weights: np.ndarray,
    *,
    radius: float,
) -> GraphBallOperator:
    """Build a volume/proxy-weighted average over bounded geodesic balls."""

    positions = np.asarray(nodes, dtype=np.float64)
    mass = np.asarray(node_weights, dtype=np.float64).reshape(-1)
    if mass.shape != (positions.shape[0],):
        raise ValueError("node_weights must align with nodes")
    if not np.isfinite(mass).all() or np.any(mass <= 0.0):
        raise ValueError("node_weights must be finite and positive")
    radius_value = float(radius)
    if not np.isfinite(radius_value) or radius_value <= 0.0:
        raise ValueError("radius must be finite and positive")
    adjacency, canonical_edges = _graph_adjacency(positions, edges)
    tolerance = graph_geometry_tolerance(nodes)
    targets: list[int] = []
    sources: list[int] = []
    coefficients: list[float] = []
    realized_distances: list[float] = []
    row_sums = np.zeros(positions.shape[0], dtype=np.float64)
    maximum_neighbors = 0
    for source in range(positions.shape[0]):
        distances = {source: 0.0}
        queue = [(0.0, source)]
        while queue:
            distance, node = heapq.heappop(queue)
            if distance > distances[node] + tolerance:
                continue
            for neighbor, edge_length in adjacency[node]:
                candidate = distance + edge_length
                if candidate > radius_value + tolerance:
                    continue
                if candidate + tolerance >= distances.get(neighbor, float("inf")):
                    continue
                distances[neighbor] = candidate
                heapq.heappush(queue, (candidate, neighbor))
        members = np.asarray(sorted(distances), dtype=np.int64)
        denominator = float(np.sum(mass[members]))
        row_coefficients = mass[members] / denominator
        targets.extend([source] * members.size)
        sources.extend(int(value) for value in members)
        coefficients.extend(float(value) for value in row_coefficients)
        realized_distances.extend(float(distances[int(value)]) for value in members)
        row_sums[source] = float(np.sum(row_coefficients))
        maximum_neighbors = max(maximum_neighbors, int(members.size))
    target_array = np.asarray(targets, dtype=np.int64)
    source_array = np.asarray(sources, dtype=np.int64)
    coefficient_array = np.asarray(coefficients, dtype=np.float64)
    distance_array = np.asarray(realized_distances, dtype=np.float64)
    for value in (target_array, source_array, coefficient_array, distance_array):
        value.setflags(write=False)
    return GraphBallOperator(
        node_count=int(positions.shape[0]),
        radius=radius_value,
        geometry_tolerance=tolerance,
        undirected_edge_count=int(canonical_edges.shape[0]),
        targets=target_array,
        sources=source_array,
        coefficients=coefficient_array,
        distances=distance_array,
        maximum_row_sum_error=float(np.max(np.abs(row_sums - 1.0))),
        minimum_coefficient=float(np.min(coefficient_array)),
        maximum_neighbors=maximum_neighbors,
    )


def prepare_graph_ball_operator(
    operator: GraphBallOperator,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> PreparedGraphBallOperator:
    """Copy one graph-ball operator to its explicit execution device once."""

    if not dtype.is_floating_point:
        raise ValueError("graph-ball coefficients require a floating dtype")
    return PreparedGraphBallOperator(
        node_count=operator.node_count,
        radius=operator.radius,
        geometry_tolerance=operator.geometry_tolerance,
        targets=torch.tensor(operator.targets, dtype=torch.long, device=device),
        sources=torch.tensor(operator.sources, dtype=torch.long, device=device),
        coefficients=torch.tensor(operator.coefficients, dtype=dtype, device=device),
    )


def graph_ball_average(
    values: torch.Tensor,
    operator: PreparedGraphBallOperator,
    *,
    message_chunk_bytes: int = _GRAPH_BALL_MESSAGE_CHUNK_BYTES,
) -> torch.Tensor:
    """Apply one fixed graph-ball operator to channel-first batched fields."""

    if values.ndim != 3 or values.shape[-1] != operator.node_count:
        raise ValueError("values must have shape [B,C,N] aligned with the operator")
    if (
        operator.targets.device != values.device
        or operator.sources.device != values.device
        or operator.coefficients.device != values.device
        or operator.coefficients.dtype != values.dtype
    ):
        raise ValueError("prepared operator must match the value device and dtype")
    chunk_bytes = int(message_chunk_bytes)
    if chunk_bytes < 1:
        raise ValueError("message_chunk_bytes must be positive")
    targets = operator.targets
    sources = operator.sources
    coefficients = operator.coefficients
    output = torch.zeros_like(values)
    message_bytes = max(1, values.shape[0] * values.shape[1] * values.element_size())
    chunk_size = max(
        1,
        min(
            sources.numel(),
            chunk_bytes // message_bytes,
        ),
    )
    for start in range(0, sources.numel(), chunk_size):
        end = min(start + chunk_size, sources.numel())
        source_values = torch.index_select(values, 2, sources[start:end])
        source_values = source_values * coefficients[start:end].view(1, 1, -1)
        output.scatter_add_(
            2,
            targets[start:end]
            .view(1, 1, -1)
            .expand(values.shape[0], values.shape[1], -1),
            source_values,
        )
    return output


def graph_ball_device_invariants(
    operator: PreparedGraphBallOperator,
) -> dict[str, float]:
    """Measure executed row-sum and constant-field errors in model arithmetic."""

    row_sums = torch.zeros(
        operator.node_count,
        dtype=operator.coefficients.dtype,
        device=operator.coefficients.device,
    )
    row_sums.scatter_add_(0, operator.targets, operator.coefficients)
    constant = torch.ones(
        (1, 1, operator.node_count),
        dtype=operator.coefficients.dtype,
        device=operator.coefficients.device,
    )
    reproduced = graph_ball_average(constant, operator)
    return {
        "maximum_row_sum_error": float(
            torch.max(torch.abs(row_sums - 1.0)).detach().cpu()
        ),
        "maximum_constant_error": float(
            torch.max(torch.abs(reproduced - constant)).detach().cpu()
        ),
    }


def _resolution(value: Sequence[int], *, name: str) -> tuple[int, int]:
    if len(value) != 2:
        raise ValueError(f"{name} must contain nx and ny")
    nx, ny = (int(item) for item in value)
    if nx < 1 or ny < 1:
        raise ValueError(f"{name} must be positive")
    return nx, ny


def restrict_channel_first(
    value: torch.Tensor,
    *,
    fine_resolution: Sequence[int],
    coarse_resolution: Sequence[int],
) -> torch.Tensor:
    """Block-average a ``[batch, channels, nodes]`` nested-grid field."""

    fine_nx, fine_ny = _resolution(fine_resolution, name="fine_resolution")
    coarse_nx, coarse_ny = _resolution(coarse_resolution, name="coarse_resolution")
    if fine_nx % coarse_nx or fine_ny % coarse_ny:
        raise ValueError("fine resolution must be an integer refinement of coarse")
    if value.ndim != 3 or value.shape[-1] != fine_nx * fine_ny:
        raise ValueError("value must have shape [batch, channels, fine_nodes]")
    ratio_x = fine_nx // coarse_nx
    ratio_y = fine_ny // coarse_ny
    reshaped = value.reshape(
        value.shape[0],
        value.shape[1],
        coarse_ny,
        ratio_y,
        coarse_nx,
        ratio_x,
    )
    return reshaped.mean(dim=(3, 5)).reshape(
        value.shape[0], value.shape[1], coarse_nx * coarse_ny
    )


def prolong_channel_first(
    value: torch.Tensor,
    *,
    coarse_resolution: Sequence[int],
    fine_resolution: Sequence[int],
) -> torch.Tensor:
    """Piecewise-constantly inject a ``[batch, channels, nodes]`` grid field."""

    coarse_nx, coarse_ny = _resolution(coarse_resolution, name="coarse_resolution")
    fine_nx, fine_ny = _resolution(fine_resolution, name="fine_resolution")
    if fine_nx % coarse_nx or fine_ny % coarse_ny:
        raise ValueError("fine resolution must be an integer refinement of coarse")
    if value.ndim != 3 or value.shape[-1] != coarse_nx * coarse_ny:
        raise ValueError("value must have shape [batch, channels, coarse_nodes]")
    ratio_x = fine_nx // coarse_nx
    ratio_y = fine_ny // coarse_ny
    grid = value.reshape(value.shape[0], value.shape[1], coarse_ny, coarse_nx)
    return (
        grid.repeat_interleave(ratio_y, dim=2)
        .repeat_interleave(ratio_x, dim=3)
        .reshape(value.shape[0], value.shape[1], fine_nx * fine_ny)
    )


def restrict_node_last(
    value: torch.Tensor,
    *,
    fine_resolution: Sequence[int],
    coarse_resolution: Sequence[int],
) -> torch.Tensor:
    """Block-average a ``[batch, nodes, channels]`` nested-grid field."""

    if value.ndim != 3:
        raise ValueError("value must have shape [batch, nodes, channels]")
    return restrict_channel_first(
        value.permute(0, 2, 1),
        fine_resolution=fine_resolution,
        coarse_resolution=coarse_resolution,
    ).permute(0, 2, 1)


def _validate_aux(aux: Sequence[torch.Tensor]) -> None:
    if len(aux) != 5:
        raise ValueError("PCNO aux must contain five tensors")
    if aux[0].shape[0] != 1:
        raise ValueError("pathway diagnostics require an unpadded batch of one")


def _bases(
    backbone: torch.nn.Module, aux: Sequence[torch.Tensor]
) -> tuple[torch.Tensor, ...]:
    _, nodes, node_weights, _, _ = aux
    bases_c, bases_s, bases_0 = compute_Fourier_bases(nodes, backbone.modes)
    wbases_c = torch.einsum("bxkw,bxw->bxkw", bases_c, node_weights)
    wbases_s = torch.einsum("bxkw,bxw->bxkw", bases_s, node_weights)
    wbases_0 = torch.einsum("bxkw,bxw->bxkw", bases_0, node_weights)
    return bases_c, bases_s, bases_0, wbases_c, wbases_s, wbases_0


def _layer_branches(
    backbone: torch.nn.Module,
    layer: int,
    hidden: torch.Tensor,
    aux: Sequence[torch.Tensor],
    bases: Sequence[torch.Tensor],
) -> dict[str, torch.Tensor]:
    _, _, _, directed_edges, edge_gradient_weights = aux
    return {
        "spectral": backbone.sp_convs[layer](hidden, *bases),
        "pointwise": backbone.ws[layer](hidden),
        "differential": backbone.gws[layer](
            hidden, directed_edges, edge_gradient_weights
        ),
    }


def _spectral_modal_output(
    spectral: torch.nn.Module,
    hidden: torch.Tensor,
    weighted_bases: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply the learned spectral multiplier and retain modal coefficients."""

    wbases_c, wbases_s, wbases_0 = weighted_bases
    x_c_hat = torch.einsum("bix,bxkw->bikw", hidden, wbases_c)
    x_s_hat = -torch.einsum("bix,bxkw->bikw", hidden, wbases_s)
    x_0_hat = torch.einsum("bix,bxkw->bikw", hidden, wbases_0)
    f_c_hat = torch.einsum(
        "bikw,iokw->bokw", x_c_hat, spectral.weights_c
    ) - torch.einsum("bikw,iokw->bokw", x_s_hat, spectral.weights_s)
    f_s_hat = torch.einsum(
        "bikw,iokw->bokw", x_s_hat, spectral.weights_c
    ) + torch.einsum("bikw,iokw->bokw", x_c_hat, spectral.weights_s)
    f_0_hat = torch.einsum("bikw,iokw->bokw", x_0_hat, spectral.weights_0)
    return f_c_hat, f_s_hat, f_0_hat


def _spectral_synthesis(
    modal: Sequence[torch.Tensor], bases: Sequence[torch.Tensor]
) -> torch.Tensor:
    f_c_hat, f_s_hat, f_0_hat = modal
    bases_c, bases_s, bases_0 = bases
    return (
        torch.einsum("bokw,bxkw->box", f_0_hat, bases_0)
        + 2.0 * torch.einsum("bokw,bxkw->box", f_c_hat, bases_c)
        - 2.0 * torch.einsum("bokw,bxkw->box", f_s_hat, bases_s)
    )


@torch.no_grad()
def decompose_spectral_same_hidden_mesh_gap(
    spectral: torch.nn.Module,
    fine_hidden: torch.Tensor,
    coarse_aux: Sequence[torch.Tensor],
    fine_aux: Sequence[torch.Tensor],
    *,
    coarse_resolution: Sequence[int],
    fine_resolution: Sequence[int],
) -> dict[str, torch.Tensor]:
    """Close a spectral gap into quadrature, subcell, and synthesis terms.

    Let ``h_c = R h_f`` and let ``P`` be piecewise-constant injection.  The
    intermediate ``S_c M A_f(P h_c)`` evaluates the same coarse hidden values
    with fine-grid bases and weights.  Its difference from the coarse-native
    branch is therefore a controlled quadrature/analysis-grid response, not an
    exact continuum quadrature error.  The next term measures fine subcell
    hidden content, and the last measures synthesis/restriction.
    """

    _validate_aux(coarse_aux)
    _validate_aux(fine_aux)
    coarse_resolution = _resolution(coarse_resolution, name="coarse_resolution")
    fine_resolution = _resolution(fine_resolution, name="fine_resolution")
    coarse_hidden = restrict_channel_first(
        fine_hidden,
        fine_resolution=fine_resolution,
        coarse_resolution=coarse_resolution,
    )
    coarse_bases = _bases_for_modes(spectral, coarse_aux)
    fine_bases = _bases_for_modes(spectral, fine_aux)
    coarse_native = spectral(coarse_hidden, *coarse_bases)
    prolonged_coarse_hidden = prolong_channel_first(
        coarse_hidden,
        coarse_resolution=coarse_resolution,
        fine_resolution=fine_resolution,
    )
    prolonged_modal = _spectral_modal_output(
        spectral, prolonged_coarse_hidden, fine_bases[3:]
    )
    cross_prolonged = _spectral_synthesis(prolonged_modal, coarse_bases[:3])
    fine_modal = _spectral_modal_output(spectral, fine_hidden, fine_bases[3:])
    cross = _spectral_synthesis(fine_modal, coarse_bases[:3])
    fine_native = _spectral_synthesis(fine_modal, fine_bases[:3])
    restricted_fine = restrict_channel_first(
        fine_native,
        fine_resolution=fine_resolution,
        coarse_resolution=coarse_resolution,
    )
    quadrature = coarse_native - cross_prolonged
    subcell = cross_prolonged - cross
    analysis = quadrature + subcell
    synthesis = cross - restricted_fine
    mesh = coarse_native - restricted_fine
    return {
        "coarse_native": coarse_native,
        "cross_prolonged_analysis_coarse_synthesis": cross_prolonged,
        "cross_fine_analysis_coarse_synthesis": cross,
        "restricted_fine_native": restricted_fine,
        "quadrature_response": quadrature,
        "subcell_hidden_response": subcell,
        "analysis_response": analysis,
        "synthesis_restriction_response": synthesis,
        "mesh_gap": mesh,
        "analysis_closure": analysis - quadrature - subcell,
        "closure": mesh - quadrature - subcell - synthesis,
    }


def _bases_for_modes(
    spectral: torch.nn.Module, aux: Sequence[torch.Tensor]
) -> tuple[torch.Tensor, ...]:
    """Build PCNO bases for a spectral layer without requiring its backbone."""

    _, nodes, node_weights, _, _ = aux
    modes = spectral.modes.to(device=nodes.device, dtype=nodes.dtype)
    bases_c, bases_s, bases_0 = compute_Fourier_bases(nodes, modes)
    wbases_c = torch.einsum("bxkw,bxw->bxkw", bases_c, node_weights)
    wbases_s = torch.einsum("bxkw,bxw->bxkw", bases_s, node_weights)
    wbases_0 = torch.einsum("bxkw,bxw->bxkw", bases_0, node_weights)
    return bases_c, bases_s, bases_0, wbases_c, wbases_s, wbases_0


def _two_hop_average(value: torch.Tensor, aux: Sequence[torch.Tensor]) -> torch.Tensor:
    directed_edges = aux[3]
    return graph_neighbor_average(value, directed_edges, iterations=2)


def _gradient_output_from_smoothed(
    differential: torch.nn.Module, smoothed_gradient: torch.Tensor
) -> torch.Tensor:
    return differential.gw2(differential.geo_act(differential.gw1 * smoothed_gradient))


@torch.no_grad()
def trace_same_hidden_physical_radius_outputs(
    backbone: torch.nn.Module,
    fine_model_input: torch.Tensor,
    coarse_aux: Sequence[torch.Tensor],
    fine_aux: Sequence[torch.Tensor],
    *,
    layer: int,
    coarse_resolution: Sequence[int],
    fine_resolution: Sequence[int],
    coarse_local_operator: PreparedGraphBallOperator,
    fine_local_operator: PreparedGraphBallOperator,
    coarse_fixed_operator: PreparedGraphBallOperator,
    fine_fixed_operator: PreparedGraphBallOperator,
) -> dict[str, dict[str, torch.Tensor]]:
    """Evaluate final-layer A0/A1/A2 arms from one native fine hidden field.

    A0 retains the checkpoint's repeated two-hop average. A1 and A2 replace
    only that average by local- and fixed-radius graph-ball operators. The
    native fine prefix, least-squares gradient, learned maps, other branches,
    and decoder are shared exactly.
    """

    _validate_aux(coarse_aux)
    _validate_aux(fine_aux)
    selected_layer = int(layer)
    if selected_layer != len(backbone.ws) - 1:
        raise ValueError("physical-radius stage currently requires the final layer")
    coarse_resolution = _resolution(coarse_resolution, name="coarse_resolution")
    fine_resolution = _resolution(fine_resolution, name="fine_resolution")
    coarse_bases = _bases(backbone, coarse_aux)
    fine_bases = _bases(backbone, fine_aux)
    fine_hidden = backbone.fc0(fine_model_input).permute(0, 2, 1)
    for prefix_layer in range(selected_layer):
        branches = _layer_branches(
            backbone, prefix_layer, fine_hidden, fine_aux, fine_bases
        )
        fine_hidden = _advance_hidden(backbone, fine_hidden, branches, prefix_layer)
    coarse_hidden = restrict_channel_first(
        fine_hidden,
        fine_resolution=fine_resolution,
        coarse_resolution=coarse_resolution,
    )
    coarse_gradient = compute_gradient(coarse_hidden, coarse_aux[3], coarse_aux[4])
    fine_gradient = compute_gradient(fine_hidden, fine_aux[3], fine_aux[4])
    differential = backbone.gws[selected_layer]
    coarse_native_smoothed = _two_hop_average(coarse_gradient, coarse_aux)
    fine_native_smoothed = _two_hop_average(fine_gradient, fine_aux)
    coarse_native_post_softsign = differential.geo_act(
        differential.gw1 * coarse_native_smoothed
    )
    fine_native_post_softsign = differential.geo_act(
        differential.gw1 * fine_native_smoothed
    )
    coarse_native_differential = differential.gw2(coarse_native_post_softsign)
    fine_native_differential = differential.gw2(fine_native_post_softsign)
    coarse_native = {
        "spectral": backbone.sp_convs[selected_layer](coarse_hidden, *coarse_bases),
        "pointwise": backbone.ws[selected_layer](coarse_hidden),
        "differential": coarse_native_differential,
    }
    fine_native = {
        "spectral": backbone.sp_convs[selected_layer](fine_hidden, *fine_bases),
        "pointwise": backbone.ws[selected_layer](fine_hidden),
        "differential": fine_native_differential,
    }
    coarse_local_smoothed = graph_ball_average(coarse_gradient, coarse_local_operator)
    fine_local_smoothed = graph_ball_average(fine_gradient, fine_local_operator)
    coarse_smoothed = {
        "A0": coarse_native_smoothed,
        "A1": coarse_local_smoothed,
        "A2": (
            coarse_local_smoothed
            if coarse_fixed_operator is coarse_local_operator
            else graph_ball_average(coarse_gradient, coarse_fixed_operator)
        ),
    }
    fine_smoothed = {
        "A0": fine_native_smoothed,
        "A1": fine_local_smoothed,
        "A2": (
            fine_local_smoothed
            if fine_fixed_operator is fine_local_operator
            else graph_ball_average(fine_gradient, fine_fixed_operator)
        ),
    }
    coarse_post_softsign = {
        "A0": coarse_native_post_softsign,
        **{
            arm: differential.geo_act(differential.gw1 * coarse_smoothed[arm])
            for arm in ("A1", "A2")
        },
    }
    fine_post_softsign = {
        "A0": fine_native_post_softsign,
        **{
            arm: differential.geo_act(differential.gw1 * fine_smoothed[arm])
            for arm in ("A1", "A2")
        },
    }
    coarse_differentials = {
        "A0": coarse_native_differential,
        **{arm: differential.gw2(coarse_post_softsign[arm]) for arm in ("A1", "A2")},
    }
    fine_differentials = {
        "A0": fine_native_differential,
        **{arm: differential.gw2(fine_post_softsign[arm]) for arm in ("A1", "A2")},
    }
    restricted_fine_spectral = restrict_channel_first(
        fine_native["spectral"],
        fine_resolution=fine_resolution,
        coarse_resolution=coarse_resolution,
    )
    restricted_fine_pointwise = restrict_channel_first(
        fine_native["pointwise"],
        fine_resolution=fine_resolution,
        coarse_resolution=coarse_resolution,
    )
    native_coarse_output = _decode(
        backbone,
        _advance_hidden(backbone, coarse_hidden, coarse_native, selected_layer),
    )
    native_fine_output = _decode(
        backbone,
        _advance_hidden(backbone, fine_hidden, fine_native, selected_layer),
    )
    native_restricted_fine_output = restrict_node_last(
        native_fine_output,
        fine_resolution=fine_resolution,
        coarse_resolution=coarse_resolution,
    )
    outputs: dict[str, dict[str, torch.Tensor]] = {}
    for arm in ("A0", "A1", "A2"):
        coarse_branches = dict(coarse_native)
        coarse_branches["differential"] = coarse_differentials[arm]
        fine_branches = dict(fine_native)
        fine_branches["differential"] = fine_differentials[arm]
        if arm == "A0":
            coarse_output = native_coarse_output
            restricted_fine_output = native_restricted_fine_output
        else:
            coarse_output = _decode(
                backbone,
                _advance_hidden(
                    backbone, coarse_hidden, coarse_branches, selected_layer
                ),
            )
            fine_output = _decode(
                backbone,
                _advance_hidden(backbone, fine_hidden, fine_branches, selected_layer),
            )
            restricted_fine_output = restrict_node_last(
                fine_output,
                fine_resolution=fine_resolution,
                coarse_resolution=coarse_resolution,
            )
        outputs[arm] = {
            "coarse_spectral": coarse_native["spectral"],
            "restricted_fine_spectral": restricted_fine_spectral,
            "coarse_pointwise": coarse_native["pointwise"],
            "restricted_fine_pointwise": restricted_fine_pointwise,
            "coarse_gradient": coarse_gradient,
            "restricted_fine_gradient": restrict_channel_first(
                fine_gradient,
                fine_resolution=fine_resolution,
                coarse_resolution=coarse_resolution,
            ),
            "coarse_smoothed_gradient": coarse_smoothed[arm],
            "restricted_fine_smoothed_gradient": restrict_channel_first(
                fine_smoothed[arm],
                fine_resolution=fine_resolution,
                coarse_resolution=coarse_resolution,
            ),
            "coarse_post_softsign": coarse_post_softsign[arm],
            "restricted_fine_post_softsign": restrict_channel_first(
                fine_post_softsign[arm],
                fine_resolution=fine_resolution,
                coarse_resolution=coarse_resolution,
            ),
            "coarse_differential": coarse_differentials[arm],
            "restricted_fine_differential": restrict_channel_first(
                fine_differentials[arm],
                fine_resolution=fine_resolution,
                coarse_resolution=coarse_resolution,
            ),
            "coarse_output": coarse_output,
            "restricted_fine_output": restricted_fine_output,
            "native_coarse_output": native_coarse_output,
            "native_restricted_fine_output": native_restricted_fine_output,
        }
    return outputs


@torch.no_grad()
def decompose_differential_same_hidden_mesh_gap(
    differential: torch.nn.Module,
    fine_hidden: torch.Tensor,
    coarse_aux: Sequence[torch.Tensor],
    fine_aux: Sequence[torch.Tensor],
    *,
    coarse_resolution: Sequence[int],
    fine_resolution: Sequence[int],
) -> dict[str, torch.Tensor]:
    """Close one differential mesh gap into gradient and fixed-hop terms.

    The pre-nonlinear split is exact and isolates the least-squares gradient
    response from the two-hop neighbor-average commutator.  The decoded branch
    split is also exact, but its second term includes restriction through
    ``gw1``, Softsign, and ``gw2`` and is therefore labeled composite.
    """

    _validate_aux(coarse_aux)
    _validate_aux(fine_aux)
    coarse_resolution = _resolution(coarse_resolution, name="coarse_resolution")
    fine_resolution = _resolution(fine_resolution, name="fine_resolution")
    coarse_hidden = restrict_channel_first(
        fine_hidden,
        fine_resolution=fine_resolution,
        coarse_resolution=coarse_resolution,
    )
    coarse_gradient = compute_gradient(coarse_hidden, coarse_aux[3], coarse_aux[4])
    fine_gradient = compute_gradient(fine_hidden, fine_aux[3], fine_aux[4])
    restricted_fine_gradient = restrict_channel_first(
        fine_gradient,
        fine_resolution=fine_resolution,
        coarse_resolution=coarse_resolution,
    )
    coarse_smoothed = _two_hop_average(coarse_gradient, coarse_aux)
    coarse_from_restricted_gradient = _two_hop_average(
        restricted_fine_gradient, coarse_aux
    )
    fine_smoothed = _two_hop_average(fine_gradient, fine_aux)
    restricted_fine_smoothed = restrict_channel_first(
        fine_smoothed,
        fine_resolution=fine_resolution,
        coarse_resolution=coarse_resolution,
    )
    gradient_pre = coarse_smoothed - coarse_from_restricted_gradient
    fixed_hop_pre = coarse_from_restricted_gradient - restricted_fine_smoothed
    pre_mesh = coarse_smoothed - restricted_fine_smoothed

    coarse_output = _gradient_output_from_smoothed(differential, coarse_smoothed)
    coarse_from_fine_gradient_output = _gradient_output_from_smoothed(
        differential, coarse_from_restricted_gradient
    )
    fine_output = _gradient_output_from_smoothed(differential, fine_smoothed)
    restricted_fine_output = restrict_channel_first(
        fine_output,
        fine_resolution=fine_resolution,
        coarse_resolution=coarse_resolution,
    )
    gradient_output = coarse_output - coarse_from_fine_gradient_output
    fixed_hop_composite_output = (
        coarse_from_fine_gradient_output - restricted_fine_output
    )
    output_mesh = coarse_output - restricted_fine_output
    return {
        "coarse_gradient": coarse_gradient,
        "restricted_fine_gradient": restricted_fine_gradient,
        "coarse_smoothed_gradient": coarse_smoothed,
        "coarse_from_restricted_fine_gradient": coarse_from_restricted_gradient,
        "restricted_fine_smoothed_gradient": restricted_fine_smoothed,
        "gradient_pre_response": gradient_pre,
        "fixed_hop_pre_response": fixed_hop_pre,
        "pre_mesh_gap": pre_mesh,
        "pre_closure": pre_mesh - gradient_pre - fixed_hop_pre,
        "coarse_native": coarse_output,
        "coarse_from_restricted_fine_gradient_output": (
            coarse_from_fine_gradient_output
        ),
        "restricted_fine_native": restricted_fine_output,
        "gradient_output_response": gradient_output,
        "fixed_hop_composite_output_response": fixed_hop_composite_output,
        "mesh_gap": output_mesh,
        "output_closure": (output_mesh - gradient_output - fixed_hop_composite_output),
    }


def _pointwise_float64_mesh_gap(
    pointwise: torch.nn.Module,
    fine_hidden: torch.Tensor,
    *,
    fine_resolution: Sequence[int],
    coarse_resolution: Sequence[int],
) -> torch.Tensor:
    """Evaluate the affine pointwise/restriction identity in float64.

    The scientific trace remains in checkpoint-native float32. This separate
    reference removes float32 reduction-order effects from the exact identity
    ``W(Rh) = R W(h)`` without changing any model output.
    """

    weight = pointwise.weight[..., 0].to(dtype=torch.float64)
    bias = None if pointwise.bias is None else pointwise.bias.to(dtype=torch.float64)
    fine64 = fine_hidden.to(dtype=torch.float64)
    restricted_hidden64 = restrict_channel_first(
        fine64,
        fine_resolution=fine_resolution,
        coarse_resolution=coarse_resolution,
    )
    coarse_output64 = F.linear(
        restricted_hidden64.permute(0, 2, 1), weight, bias
    ).permute(0, 2, 1)
    fine_output64 = F.linear(fine64.permute(0, 2, 1), weight, bias).permute(0, 2, 1)
    return coarse_output64 - restrict_channel_first(
        fine_output64,
        fine_resolution=fine_resolution,
        coarse_resolution=coarse_resolution,
    )


def _validate_gains(
    backbone: torch.nn.Module,
    gains: Mapping[tuple[int, str], float] | None,
) -> dict[tuple[int, str], float]:
    validated: dict[tuple[int, str], float] = {}
    for key, value in (gains or {}).items():
        if not isinstance(key, tuple) or len(key) != 2:
            raise ValueError("branch gain keys must be (layer, branch) tuples")
        layer, branch = key
        if not isinstance(layer, int) or not 0 <= layer < len(backbone.ws):
            raise ValueError("branch gain layer lies outside the PCNO")
        if branch not in BRANCH_NAMES:
            raise ValueError(f"unsupported branch name {branch!r}")
        gain = float(value)
        if not np.isfinite(gain) or not 0.0 <= gain <= 1.0:
            raise ValueError("diagnostic branch gains must lie in [0, 1]")
        validated[(layer, branch)] = gain
    return validated


def _decode(backbone: torch.nn.Module, hidden: torch.Tensor) -> torch.Tensor:
    output = hidden.permute(0, 2, 1)
    if backbone.fc_dim > 0:
        output = backbone.fc1(output)
        if backbone.act is not None:
            output = backbone.act(output)
    return backbone.fc2(output)


def _advance_hidden(
    backbone: torch.nn.Module,
    hidden: torch.Tensor,
    branches: Mapping[str, torch.Tensor],
    layer: int,
) -> torch.Tensor:
    update = sum(branches.values())
    if backbone.act is not None and layer != len(backbone.ws) - 1:
        return hidden + backbone.act(update)
    return update


def _continue_coarse_from_layer(
    backbone: torch.nn.Module,
    hidden: torch.Tensor,
    aux: Sequence[torch.Tensor],
    bases: Sequence[torch.Tensor],
    *,
    start_layer: int,
    first_branches: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    value = _advance_hidden(backbone, hidden, first_branches, start_layer)
    for layer in range(start_layer + 1, len(backbone.ws)):
        branches = _layer_branches(backbone, layer, value, aux, bases)
        value = _advance_hidden(backbone, value, branches, layer)
    return _decode(backbone, value)


@torch.no_grad()
def trace_same_hidden_subpath_replacements(
    backbone: torch.nn.Module,
    fine_model_input: torch.Tensor,
    coarse_aux: Sequence[torch.Tensor],
    fine_aux: Sequence[torch.Tensor],
    *,
    layer: int,
    coarse_resolution: Sequence[int],
    fine_resolution: Sequence[int],
) -> dict[str, torch.Tensor]:
    """Decode single-layer pathway replacements from one paired hidden field.

    The fine prefix is native.  At ``layer`` its hidden field is restricted to
    the coarse grid and becomes the common input for every coarse continuation.
    Each arm replaces exactly one coarse branch value at that layer, then uses
    the unchanged coarse checkpoint for all later layers.  These are causal
    representation interventions, not standalone deployable models.
    """

    _validate_aux(coarse_aux)
    _validate_aux(fine_aux)
    if not 0 <= int(layer) < len(backbone.ws):
        raise ValueError("layer lies outside the PCNO")
    coarse_resolution = _resolution(coarse_resolution, name="coarse_resolution")
    fine_resolution = _resolution(fine_resolution, name="fine_resolution")
    coarse_bases = _bases(backbone, coarse_aux)
    fine_bases = _bases(backbone, fine_aux)
    fine_hidden = backbone.fc0(fine_model_input).permute(0, 2, 1)
    for prefix_layer in range(layer):
        fine_branches = _layer_branches(
            backbone, prefix_layer, fine_hidden, fine_aux, fine_bases
        )
        fine_hidden = _advance_hidden(
            backbone, fine_hidden, fine_branches, prefix_layer
        )
    coarse_hidden = restrict_channel_first(
        fine_hidden,
        fine_resolution=fine_resolution,
        coarse_resolution=coarse_resolution,
    )
    coarse_branches = _layer_branches(
        backbone, layer, coarse_hidden, coarse_aux, coarse_bases
    )
    fine_branches = _layer_branches(backbone, layer, fine_hidden, fine_aux, fine_bases)
    restricted_fine_branches = {
        name: restrict_channel_first(
            value,
            fine_resolution=fine_resolution,
            coarse_resolution=coarse_resolution,
        )
        for name, value in fine_branches.items()
    }
    spectral = decompose_spectral_same_hidden_mesh_gap(
        backbone.sp_convs[layer],
        fine_hidden,
        coarse_aux,
        fine_aux,
        coarse_resolution=coarse_resolution,
        fine_resolution=fine_resolution,
    )
    differential = decompose_differential_same_hidden_mesh_gap(
        backbone.gws[layer],
        fine_hidden,
        coarse_aux,
        fine_aux,
        coarse_resolution=coarse_resolution,
        fine_resolution=fine_resolution,
    )
    replacements = {
        "baseline": None,
        "fourier_quadrature": (
            spectral["coarse_native"] - spectral["quadrature_response"]
        ),
        "fourier_subcell": (
            spectral["coarse_native"] - spectral["subcell_hidden_response"]
        ),
        "fourier_analysis": spectral["cross_fine_analysis_coarse_synthesis"],
        "fourier_synthesis": (
            spectral["coarse_native"] - spectral["synthesis_restriction_response"]
        ),
        "fourier_full": restricted_fine_branches["spectral"],
        "differential_gradient": differential[
            "coarse_from_restricted_fine_gradient_output"
        ],
        "differential_fixed_hop": (
            differential["coarse_native"]
            - differential["fixed_hop_composite_output_response"]
        ),
        "differential_full": restricted_fine_branches["differential"],
        "pointwise_full": restricted_fine_branches["pointwise"],
    }
    branch_for_arm = {
        "fourier_quadrature": "spectral",
        "fourier_subcell": "spectral",
        "fourier_analysis": "spectral",
        "fourier_synthesis": "spectral",
        "fourier_full": "spectral",
        "differential_gradient": "differential",
        "differential_fixed_hop": "differential",
        "differential_full": "differential",
        "pointwise_full": "pointwise",
    }
    outputs: dict[str, torch.Tensor] = {}
    for name, replacement in replacements.items():
        branches = dict(coarse_branches)
        if replacement is not None:
            branches[branch_for_arm[name]] = replacement
        outputs[name] = _continue_coarse_from_layer(
            backbone,
            coarse_hidden,
            coarse_aux,
            coarse_bases,
            start_layer=layer,
            first_branches=branches,
        )

    fine_value = _advance_hidden(backbone, fine_hidden, fine_branches, layer)
    for suffix_layer in range(layer + 1, len(backbone.ws)):
        suffix_branches = _layer_branches(
            backbone, suffix_layer, fine_value, fine_aux, fine_bases
        )
        fine_value = _advance_hidden(
            backbone, fine_value, suffix_branches, suffix_layer
        )
    outputs["restricted_fine"] = restrict_node_last(
        _decode(backbone, fine_value),
        fine_resolution=fine_resolution,
        coarse_resolution=coarse_resolution,
    )
    return outputs


@torch.no_grad()
def trace_native_hidden_to_layer(
    backbone: torch.nn.Module,
    model_input: torch.Tensor,
    aux: Sequence[torch.Tensor],
    *,
    layer: int,
) -> torch.Tensor:
    """Return the unchanged native hidden field entering one PCNO block."""

    _validate_aux(aux)
    if not 0 <= int(layer) < len(backbone.ws):
        raise ValueError("layer lies outside the PCNO")
    bases = _bases(backbone, aux)
    hidden = backbone.fc0(model_input).permute(0, 2, 1)
    for prefix_layer in range(layer):
        branches = _layer_branches(backbone, prefix_layer, hidden, aux, bases)
        hidden = _advance_hidden(backbone, hidden, branches, prefix_layer)
    return hidden


@torch.no_grad()
def trace_native_branch_replacement_output(
    backbone: torch.nn.Module,
    coarse_model_input: torch.Tensor,
    coarse_aux: Sequence[torch.Tensor],
    fine_model_input: torch.Tensor,
    fine_aux: Sequence[torch.Tensor],
    *,
    branch: str,
    coarse_resolution: Sequence[int],
    fine_resolution: Sequence[int],
    replacement_layers: Sequence[int] | None = None,
) -> dict[str, torch.Tensor]:
    """Replace one coarse branch by restricted native-fine values while decoding."""

    _validate_aux(coarse_aux)
    _validate_aux(fine_aux)
    if branch not in BRANCH_NAMES:
        raise ValueError(f"unsupported branch name {branch!r}")
    coarse_resolution = _resolution(coarse_resolution, name="coarse_resolution")
    fine_resolution = _resolution(fine_resolution, name="fine_resolution")
    selected = (
        set(range(len(backbone.ws)))
        if replacement_layers is None
        else {int(value) for value in replacement_layers}
    )
    if any(value < 0 or value >= len(backbone.ws) for value in selected):
        raise ValueError("replacement layer lies outside the PCNO")
    coarse_bases = _bases(backbone, coarse_aux)
    fine_bases = _bases(backbone, fine_aux)
    coarse_hidden = backbone.fc0(coarse_model_input).permute(0, 2, 1)
    fine_hidden = backbone.fc0(fine_model_input).permute(0, 2, 1)
    for layer in range(len(backbone.ws)):
        coarse_branches = _layer_branches(
            backbone, layer, coarse_hidden, coarse_aux, coarse_bases
        )
        fine_branches = _layer_branches(
            backbone, layer, fine_hidden, fine_aux, fine_bases
        )
        if layer in selected:
            coarse_branches[branch] = restrict_channel_first(
                fine_branches[branch],
                fine_resolution=fine_resolution,
                coarse_resolution=coarse_resolution,
            )
        coarse_hidden = _advance_hidden(backbone, coarse_hidden, coarse_branches, layer)
        fine_hidden = _advance_hidden(backbone, fine_hidden, fine_branches, layer)
    return {
        "hybrid": _decode(backbone, coarse_hidden),
        "restricted_fine": restrict_node_last(
            _decode(backbone, fine_hidden),
            fine_resolution=fine_resolution,
            coarse_resolution=coarse_resolution,
        ),
    }


@torch.no_grad()
def trace_backbone_output(
    backbone: torch.nn.Module,
    model_input: torch.Tensor,
    aux: Sequence[torch.Tensor],
    *,
    branch_gains: Mapping[tuple[int, str], float] | None = None,
) -> torch.Tensor:
    """Replay a PCNO backbone, optionally with frozen diagnostic gains."""

    _validate_aux(aux)
    gains = _validate_gains(backbone, branch_gains)
    bases = _bases(backbone, aux)
    hidden = backbone.fc0(model_input).permute(0, 2, 1)
    final_layer = len(backbone.ws) - 1
    for layer in range(len(backbone.ws)):
        branches = _layer_branches(backbone, layer, hidden, aux, bases)
        update = sum(
            value * gains.get((layer, name), 1.0) for name, value in branches.items()
        )
        if backbone.act is not None and layer != final_layer:
            hidden = hidden + backbone.act(update)
        else:
            hidden = update
    return _decode(backbone, hidden)


def _weights(aux: Sequence[torch.Tensor]) -> torch.Tensor:
    node_weights = aux[2]
    if node_weights.ndim != 3 or node_weights.shape[0] != 1:
        raise ValueError("node weights must have shape [1, nodes, measures]")
    if node_weights.shape[-1] != 1:
        raise ValueError("pathway diagnostics require exactly one measure")
    weights = node_weights[..., 0]
    if bool((weights <= 0.0).any()):
        raise ValueError("node weights must be positive")
    return weights


def _inner(left: torch.Tensor, right: torch.Tensor, weights: torch.Tensor) -> float:
    if left.shape != right.shape or left.ndim != 3 or left.shape[0] != 1:
        raise ValueError("latent fields must align as [1, channels, nodes]")
    if weights.shape != (1, left.shape[-1]):
        raise ValueError("weights do not align with latent nodes")
    numerator = torch.einsum("bn,bcn,bcn->", weights, left, right)
    return float((numerator / weights.sum()).detach().cpu())


def _norm(value: torch.Tensor, weights: torch.Tensor) -> float:
    return float(np.sqrt(max(_inner(value, value, weights), 0.0)))


def _ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0.0:
        return None
    return float(numerator / denominator)


def _symmetric_scale(
    left: torch.Tensor, right: torch.Tensor, weights: torch.Tensor
) -> float:
    return float(
        np.sqrt(
            max(
                0.5 * (_inner(left, left, weights) + _inner(right, right, weights)),
                0.0,
            )
        )
    )


def _gap_row(
    *,
    stage: str,
    layer: int,
    coarse: torch.Tensor,
    restricted_fine: torch.Tensor,
    weights: torch.Tensor,
) -> dict[str, Any]:
    gap = coarse - restricted_fine
    scale = _symmetric_scale(coarse, restricted_fine, weights)
    return {
        "record_kind": "hidden",
        "stage": stage,
        "layer": layer,
        "gap_rms": _norm(gap, weights),
        "symmetric_scale_rms": scale,
        "symmetric_relative_gap": _ratio(_norm(gap, weights), scale),
    }


@torch.no_grad()
def trace_resolution_pair(
    backbone: torch.nn.Module,
    coarse_model_input: torch.Tensor,
    coarse_aux: Sequence[torch.Tensor],
    fine_model_input: torch.Tensor,
    fine_aux: Sequence[torch.Tensor],
    *,
    coarse_resolution: Sequence[int],
    fine_resolution: Sequence[int],
    collect_fields: bool = False,
    check_pointwise_float64: bool = True,
    input_groups: Mapping[str, tuple[int, int]] = DEFAULT_INPUT_GROUPS,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]], dict[str, np.ndarray]]:
    """Trace native and same-hidden mesh commutators through a resolution pair."""

    _validate_aux(coarse_aux)
    _validate_aux(fine_aux)
    coarse_resolution = _resolution(coarse_resolution, name="coarse_resolution")
    fine_resolution = _resolution(fine_resolution, name="fine_resolution")
    if coarse_model_input.shape[-1] != fine_model_input.shape[-1]:
        raise ValueError("coarse and fine model inputs must have the same channels")
    if coarse_model_input.shape[-1] != backbone.in_dim:
        raise ValueError("model input channels do not match the backbone")
    coarse_bases = _bases(backbone, coarse_aux)
    fine_bases = _bases(backbone, fine_aux)
    weights = _weights(coarse_aux)
    restricted_fine_input = restrict_node_last(
        fine_model_input,
        fine_resolution=fine_resolution,
        coarse_resolution=coarse_resolution,
    )
    input_gap = coarse_model_input - restricted_fine_input
    coarse_hidden = backbone.fc0(coarse_model_input).permute(0, 2, 1)
    fine_hidden = backbone.fc0(fine_model_input).permute(0, 2, 1)
    restricted_fine_hidden = restrict_channel_first(
        fine_hidden,
        fine_resolution=fine_resolution,
        coarse_resolution=coarse_resolution,
    )
    rows: list[dict[str, Any]] = [
        _gap_row(
            stage="lift",
            layer=-1,
            coarse=coarse_hidden,
            restricted_fine=restricted_fine_hidden,
            weights=weights,
        )
    ]
    fields: dict[str, np.ndarray] = {}

    lift_gap = coarse_hidden - restricted_fine_hidden
    lift_energy = _inner(lift_gap, lift_gap, weights)
    reconstructed_lift = torch.zeros_like(lift_gap)
    for name, (start, stop) in input_groups.items():
        if not 0 <= start < stop <= backbone.in_dim:
            raise ValueError(f"invalid input group {name!r}")
        contribution = F.linear(
            input_gap[..., start:stop],
            backbone.fc0.weight[:, start:stop],
            bias=None,
        ).permute(0, 2, 1)
        reconstructed_lift = reconstructed_lift + contribution
        rows.append(
            {
                "record_kind": "input_group",
                "stage": "lift",
                "layer": -1,
                "input_group": name,
                "gap_rms": _norm(contribution, weights),
                "symmetric_attribution": _ratio(
                    _inner(contribution, lift_gap, weights), lift_energy
                ),
            }
        )
    rows.append(
        {
            "record_kind": "closure",
            "stage": "lift",
            "layer": -1,
            "closure_rms": _norm(lift_gap - reconstructed_lift, weights),
            "relative_closure": _ratio(
                _norm(lift_gap - reconstructed_lift, weights),
                _norm(lift_gap, weights),
            ),
        }
    )

    final_layer = len(backbone.ws) - 1
    for layer in range(len(backbone.ws)):
        restricted_fine_hidden = restrict_channel_first(
            fine_hidden,
            fine_resolution=fine_resolution,
            coarse_resolution=coarse_resolution,
        )
        coarse_branches = _layer_branches(
            backbone, layer, coarse_hidden, coarse_aux, coarse_bases
        )
        fine_branches = _layer_branches(
            backbone, layer, fine_hidden, fine_aux, fine_bases
        )
        same_hidden_coarse = _layer_branches(
            backbone, layer, restricted_fine_hidden, coarse_aux, coarse_bases
        )
        native_gaps: dict[str, torch.Tensor] = {}
        mesh_gaps: dict[str, torch.Tensor] = {}
        state_gaps: dict[str, torch.Tensor] = {}
        restricted_fine_branches: dict[str, torch.Tensor] = {}
        for branch in BRANCH_NAMES:
            restricted = restrict_channel_first(
                fine_branches[branch],
                fine_resolution=fine_resolution,
                coarse_resolution=coarse_resolution,
            )
            restricted_fine_branches[branch] = restricted
            native_gaps[branch] = coarse_branches[branch] - restricted
            mesh_gaps[branch] = same_hidden_coarse[branch] - restricted
            state_gaps[branch] = coarse_branches[branch] - same_hidden_coarse[branch]

        combined_native = sum(native_gaps.values())
        combined_mesh = sum(mesh_gaps.values())
        combined_state = sum(state_gaps.values())
        combined_energy = _inner(combined_native, combined_native, weights)
        combined_native_rms = _norm(combined_native, weights)
        rows.append(
            {
                "record_kind": "preactivation",
                "stage": "preactivation",
                "layer": layer,
                "native_gap_rms": combined_native_rms,
                "mesh_gap_rms": _norm(combined_mesh, weights),
                "state_gap_rms": _norm(combined_state, weights),
                "mesh_symmetric_attribution": _ratio(
                    _inner(combined_mesh, combined_native, weights), combined_energy
                ),
                "state_symmetric_attribution": _ratio(
                    _inner(combined_state, combined_native, weights), combined_energy
                ),
                "relative_closure": _ratio(
                    _norm(combined_native - combined_mesh - combined_state, weights),
                    _norm(combined_native, weights),
                ),
            }
        )
        for branch in BRANCH_NAMES:
            native = native_gaps[branch]
            mesh = mesh_gaps[branch]
            state = state_gaps[branch]
            scale = _symmetric_scale(
                coarse_branches[branch], restricted_fine_branches[branch], weights
            )
            native_energy = _inner(native, native, weights)
            mesh_rms = _norm(mesh, weights)
            pointwise_float64_mesh_rms: float | None = None
            pointwise_float64_mesh_symmetric_relative: float | None = None
            if branch == "pointwise" and check_pointwise_float64:
                mesh64 = _pointwise_float64_mesh_gap(
                    backbone.ws[layer],
                    fine_hidden,
                    fine_resolution=fine_resolution,
                    coarse_resolution=coarse_resolution,
                )
                weights64 = weights.to(dtype=torch.float64)
                fine_pointwise64 = F.linear(
                    fine_hidden.to(dtype=torch.float64).permute(0, 2, 1),
                    backbone.ws[layer].weight[..., 0].to(dtype=torch.float64),
                    (
                        None
                        if backbone.ws[layer].bias is None
                        else backbone.ws[layer].bias.to(dtype=torch.float64)
                    ),
                ).permute(0, 2, 1)
                restricted_fine_pointwise64 = restrict_channel_first(
                    fine_pointwise64,
                    fine_resolution=fine_resolution,
                    coarse_resolution=coarse_resolution,
                )
                coarse_pointwise64 = restricted_fine_pointwise64 + mesh64
                scale64 = _symmetric_scale(
                    coarse_pointwise64,
                    restricted_fine_pointwise64,
                    weights64,
                )
                pointwise_float64_mesh_rms = _norm(mesh64, weights64)
                pointwise_float64_mesh_symmetric_relative = _ratio(
                    pointwise_float64_mesh_rms, scale64
                )
            rows.append(
                {
                    "record_kind": "branch",
                    "stage": "preactivation",
                    "layer": layer,
                    "branch": branch,
                    "native_gap_rms": _norm(native, weights),
                    "mesh_gap_rms": mesh_rms,
                    "state_gap_rms": _norm(state, weights),
                    "symmetric_branch_scale_rms": scale,
                    "native_symmetric_relative": _ratio(_norm(native, weights), scale),
                    "mesh_symmetric_relative": _ratio(mesh_rms, scale),
                    "mesh_to_combined_native_relative": _ratio(
                        mesh_rms, combined_native_rms
                    ),
                    "pointwise_float64_mesh_rms": pointwise_float64_mesh_rms,
                    "pointwise_float64_mesh_symmetric_relative": (
                        pointwise_float64_mesh_symmetric_relative
                    ),
                    "state_symmetric_relative": _ratio(_norm(state, weights), scale),
                    "mesh_symmetric_attribution": _ratio(
                        _inner(mesh, native, weights), native_energy
                    ),
                    "state_symmetric_attribution": _ratio(
                        _inner(state, native, weights), native_energy
                    ),
                    "combined_branch_attribution": _ratio(
                        _inner(native, combined_native, weights), combined_energy
                    ),
                    "relative_closure": _ratio(
                        _norm(native - mesh - state, weights), _norm(native, weights)
                    ),
                }
            )
            if collect_fields:
                prefix = f"layer{layer}_{branch}"
                fields[f"{prefix}_native"] = (
                    native.permute(0, 2, 1).detach().cpu().numpy()[0]
                )
                fields[f"{prefix}_mesh"] = (
                    mesh.permute(0, 2, 1).detach().cpu().numpy()[0]
                )
                fields[f"{prefix}_state"] = (
                    state.permute(0, 2, 1).detach().cpu().numpy()[0]
                )

        coarse_update = sum(coarse_branches.values())
        fine_update = sum(fine_branches.values())
        if backbone.act is not None and layer != final_layer:
            coarse_hidden = coarse_hidden + backbone.act(coarse_update)
            fine_hidden = fine_hidden + backbone.act(fine_update)
        else:
            coarse_hidden = coarse_update
            fine_hidden = fine_update
        restricted_post = restrict_channel_first(
            fine_hidden,
            fine_resolution=fine_resolution,
            coarse_resolution=coarse_resolution,
        )
        rows.append(
            _gap_row(
                stage="post_hidden",
                layer=layer,
                coarse=coarse_hidden,
                restricted_fine=restricted_post,
                weights=weights,
            )
        )
    return (
        _decode(backbone, coarse_hidden),
        _decode(backbone, fine_hidden),
        rows,
        fields,
    )


def _tukey(length: int, alpha: float) -> np.ndarray:
    if length < 2 or not 0.0 <= alpha <= 1.0:
        raise ValueError("invalid Tukey-window arguments")
    if alpha == 0.0:
        return np.ones(length, dtype=np.float64)
    if alpha == 1.0:
        return np.hanning(length)
    x = np.linspace(0.0, 1.0, length)
    window = np.ones(length, dtype=np.float64)
    left = x < alpha / 2.0
    right = x > 1.0 - alpha / 2.0
    window[left] = 0.5 * (1.0 + np.cos(2.0 * np.pi / alpha * (x[left] - alpha / 2.0)))
    window[right] = 0.5 * (
        1.0 + np.cos(2.0 * np.pi / alpha * (x[right] - 1.0 + alpha / 2.0))
    )
    return window


def generic_spectral_energy_rows(
    field: np.ndarray,
    *,
    resolution: Sequence[int],
    domain_lengths: Sequence[float],
    bands: Sequence[tuple[float, float]] = DEFAULT_WAVELENGTH_BANDS,
    tukey_alpha: float = 0.1,
) -> list[dict[str, Any]]:
    """Windowed wavelength-energy shares for any latent channel count."""

    nx, ny = _resolution(resolution, name="resolution")
    if len(domain_lengths) != 2:
        raise ValueError("domain_lengths must contain lx and ly")
    lx, ly = (float(value) for value in domain_lengths)
    array = np.asarray(field, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] != nx * ny or array.shape[1] < 1:
        raise ValueError("field must have shape [nodes, channels]")
    if not np.isfinite(array).all() or lx <= 0.0 or ly <= 0.0:
        raise ValueError("field and domain must be finite and valid")
    grid = array.reshape(ny, nx, -1)
    grid = grid - grid.mean(axis=(0, 1), keepdims=True)
    window = _tukey(ny, tukey_alpha)[:, None] * _tukey(nx, tukey_alpha)[None, :]
    spectrum = np.fft.rfftn(grid * window[:, :, None], axes=(0, 1), norm="ortho")
    mode_energy = np.square(np.abs(spectrum)).sum(axis=-1)
    total = float(mode_energy.sum())
    frequency_x = np.fft.rfftfreq(nx, d=lx / nx)
    frequency_y = np.fft.fftfreq(ny, d=ly / ny)
    radial = np.hypot(frequency_y[:, None], frequency_x[None, :])
    rows: list[dict[str, Any]] = []
    for wavelength_min, wavelength_max in bands:
        lower = 0.0 if np.isinf(wavelength_max) else 1.0 / wavelength_max
        upper = 1.0 / wavelength_min
        mask = (radial >= lower) & (radial < upper)
        energy = float(mode_energy[mask].sum())
        rows.append(
            {
                "wavelength_min": float(wavelength_min),
                "wavelength_max": (
                    None if np.isinf(wavelength_max) else float(wavelength_max)
                ),
                "mode_count": int(mask.sum()),
                "spectral_energy": energy,
                "spectral_energy_share": _ratio(energy, total),
            }
        )
    return rows


def select_dominant_pathway(
    rows: Sequence[Mapping[str, Any]],
    *,
    case_ids: Sequence[str],
    pairs: Sequence[str],
    calls: Sequence[int],
    minimum_absolute_elasticity: float = 0.5,
) -> dict[str, Any]:
    """Apply the preregistered repeated-evidence pathway selector."""

    expected_units = {
        (str(case_id), str(pair), int(call))
        for case_id in case_ids
        for pair in pairs
        for call in calls
    }
    by_unit: dict[tuple[str, str, int], dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        unit = (str(row["case_id"]), str(row["pair"]), int(row["call"]))
        family = str(row["branch"])
        if unit not in expected_units or family not in BRANCH_NAMES:
            continue
        by_unit.setdefault(unit, {})[family] = row
    if set(by_unit) != expected_units or any(
        set(values) != set(BRANCH_NAMES) for values in by_unit.values()
    ):
        raise ValueError("family-gain rows do not complete the registered matrix")

    winners: dict[tuple[str, str, int], str] = {}
    for unit, values in by_unit.items():
        winners[unit] = max(
            BRANCH_NAMES,
            key=lambda name: abs(float(values[name]["defect_norm_elasticity"])),
        )
    result: dict[str, Any] = {"families": {}}
    dominant: list[str] = []
    for family in BRANCH_NAMES:
        elasticities = np.asarray(
            [
                abs(float(by_unit[unit][family]["defect_norm_elasticity"]))
                for unit in sorted(expected_units)
            ],
            dtype=np.float64,
        )
        repeated_by_pair: dict[str, int] = {}
        for pair in pairs:
            repeated_by_pair[str(pair)] = sum(
                sum(
                    winners[(str(case_id), str(pair), int(call))] == family
                    for call in calls
                )
                >= 3
                for case_id in case_ids
            )
        winner_fraction = float(
            np.mean([winner == family for winner in winners.values()])
        )
        median_abs = float(np.median(elasticities))
        qualifies = median_abs >= minimum_absolute_elasticity and all(
            count >= 5 for count in repeated_by_pair.values()
        )
        if qualifies:
            dominant.append(family)
        result["families"][family] = {
            "median_absolute_defect_norm_elasticity": median_abs,
            "winner_fraction": winner_fraction,
            "repeated_cases_by_pair": repeated_by_pair,
            "qualifies_as_dominant": qualifies,
        }
    result["decision"] = (
        dominant[0] if len(dominant) == 1 else "composite_or_unresolved"
    )
    result["dominant_candidates"] = dominant
    result["minimum_absolute_elasticity"] = float(minimum_absolute_elasticity)
    return result
