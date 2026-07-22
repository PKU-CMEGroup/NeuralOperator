from __future__ import annotations

import numpy as np
import pytest
import torch

from pcno.pcno import compute_gradient
from scripts.time_dependent_no.prepare_pcno_shock_vortex_shards import (
    RESIDUAL_FLOAT32_RELATIVE_TOLERANCE,
    STATE_FLOAT32_RELATIVE_TOLERANCE,
    _serialization_metrics,
    _serialization_passes,
)
from utility.time_dependent_no.pcno_fv_geometry import (
    build_pcno_finite_volume_geometry,
)
from utility.time_dependent_no.shock_vortex_fv import (
    BOUNDARY_TAG_NAMES,
    ShockVortexFVConfig,
    make_structured_fv_geometry,
)


def _structured_geometry(nx: int = 2, ny: int = 2):
    config = ShockVortexFVConfig(
        nx=6,
        ny=6,
        coarse_nx=nx,
        coarse_ny=ny,
        t_final=0.001,
        output_times=(0.0, 0.001),
    )
    return make_structured_fv_geometry(config)


def test_fv_cells_map_to_physical_pcno_graph_and_exact_linear_gradient() -> None:
    source = _structured_geometry()
    geometry = build_pcno_finite_volume_geometry(
        cell_centers=source.cell_centers,
        cell_volume=source.cell_volume,
        face_owner=source.face_owner,
        face_neighbor=source.face_neighbor,
        face_boundary_tag=source.face_boundary_tag,
        boundary_tag_names=BOUNDARY_TAG_NAMES,
    )

    assert geometry.nodes.shape == (4, 2)
    assert geometry.edges.shape == (4, 2)
    assert geometry.directed_edges.shape == (8, 2)
    assert np.array_equal(geometry.mesh_cell_to_graph_node, np.arange(4))
    assert np.all(geometry.node_type == 3)
    np.testing.assert_allclose(geometry.node_measures[:, 0], 0.5)
    np.testing.assert_allclose(geometry.node_weights[:, 0], 0.25)
    np.testing.assert_allclose(geometry.node_rhos[:, 0], 0.5)
    assert geometry.maximum_coordinate_gradient_error <= 1.0e-14
    assert geometry.maximum_stencil_condition_number == pytest.approx(2.0)

    interior = source.face_neighbor >= 0
    assert np.all(geometry.face_to_directed_edge[interior] >= 0)
    assert np.all(geometry.face_to_directed_edge[~interior] == -1)
    mapped_edges = geometry.directed_edges[geometry.face_to_directed_edge[interior, 0]]
    np.testing.assert_array_equal(
        mapped_edges,
        np.stack(
            (source.face_owner[interior], source.face_neighbor[interior]), axis=-1
        ),
    )

    values = geometry.nodes[:, 0] + 2.0 * geometry.nodes[:, 1]
    gradient = compute_gradient(
        torch.as_tensor(values, dtype=torch.float64).reshape(1, 1, -1),
        torch.as_tensor(geometry.directed_edges).unsqueeze(0),
        torch.as_tensor(geometry.edge_gradient_weights).unsqueeze(0),
    )
    expected = torch.tensor([1.0, 2.0], dtype=torch.float64).reshape(1, 2, 1)
    torch.testing.assert_close(gradient, expected.expand_as(gradient))


def test_boundary_codes_distinguish_symmetry_extrapolation_and_corners() -> None:
    source = _structured_geometry(nx=3, ny=3)
    geometry = build_pcno_finite_volume_geometry(
        cell_centers=source.cell_centers,
        cell_volume=source.cell_volume,
        face_owner=source.face_owner,
        face_neighbor=source.face_neighbor,
        face_boundary_tag=source.face_boundary_tag,
        boundary_tag_names=BOUNDARY_TAG_NAMES,
    )
    codes = geometry.node_type.reshape(3, 3)
    np.testing.assert_array_equal(
        codes,
        np.asarray([[3, 1, 3], [2, 0, 2], [3, 1, 3]], dtype=np.int64),
    )


def test_fv_graph_rejects_duplicate_or_rank_deficient_connectivity() -> None:
    source = _structured_geometry()
    interior_index = int(np.flatnonzero(source.face_neighbor >= 0)[0])
    owner = np.concatenate((source.face_owner, source.face_owner[[interior_index]]))
    neighbor = np.concatenate(
        (source.face_neighbor, source.face_neighbor[[interior_index]])
    )
    tags = np.concatenate(
        (source.face_boundary_tag, source.face_boundary_tag[[interior_index]])
    )
    with pytest.raises(ValueError, match="multiple physical faces"):
        build_pcno_finite_volume_geometry(
            cell_centers=source.cell_centers,
            cell_volume=source.cell_volume,
            face_owner=owner,
            face_neighbor=neighbor,
            face_boundary_tag=tags,
            boundary_tag_names=BOUNDARY_TAG_NAMES,
        )

    collinear_centers = source.cell_centers.copy()
    collinear_centers[:, 1] = 0.0
    with pytest.raises(ValueError, match="rank-deficient"):
        build_pcno_finite_volume_geometry(
            cell_centers=collinear_centers,
            cell_volume=source.cell_volume,
            face_owner=source.face_owner,
            face_neighbor=source.face_neighbor,
            face_boundary_tag=source.face_boundary_tag,
            boundary_tag_names=BOUNDARY_TAG_NAMES,
        )


def test_float32_serialization_gate_covers_state_and_residual() -> None:
    rng = np.random.default_rng(17)
    states = 1.0 + 0.01 * rng.standard_normal((5, 12, 4))
    metrics = _serialization_metrics(states)
    assert metrics["state_global_relative_l2"] < STATE_FLOAT32_RELATIVE_TOLERANCE
    assert metrics["residual_global_relative_l2"] < RESIDUAL_FLOAT32_RELATIVE_TOLERANCE
    assert _serialization_passes(metrics)

    failed = dict(metrics)
    failed["residual_max_frame_relative_l2"] = 2.0 * RESIDUAL_FLOAT32_RELATIVE_TOLERANCE
    assert not _serialization_passes(failed)
