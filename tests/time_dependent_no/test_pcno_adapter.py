import numpy as np

from utility.time_dependent_no.euler2d import make_cpg_graph_frame
from utility.time_dependent_no.euler2d_synthetic import (
    SyntheticEuler2DConfig,
    make_structured_grid_edges,
    make_structured_grid_positions,
    make_synthetic_cpg_trajectory,
)
from utility.time_dependent_no.pcno_adapter import (
    compute_graph_edge_gradient_weights,
    make_pcno_euler7_frame_batch,
    make_pcno_frame_batch,
)


def test_pcno_frame_batch_shapes_match_existing_training_convention():
    config = SyntheticEuler2DConfig(nx=7, ny=5, num_steps=4)
    group = make_synthetic_cpg_trajectory(config)
    frame = make_cpg_graph_frame(group, 1, num_steps=2)

    batch = make_pcno_frame_batch(frame)

    num_nodes = config.nx * config.ny
    assert batch.x.shape == (1, num_nodes, 8)
    assert batch.y.shape == (1, num_nodes, 4)
    assert batch.node_mask.shape == (1, num_nodes, 1)
    assert batch.nodes.shape == (1, num_nodes, 2)
    assert batch.node_weights.shape == (1, num_nodes, 1)
    assert batch.directed_edges.shape[-1] == 2
    assert batch.edge_gradient_weights.shape[-1] == 2
    np.testing.assert_allclose(np.sum(batch.node_weights), 1.0)
    assert batch.metadata["node_weight_policy"] == "equal_normalized"


def test_graph_gradient_weights_recover_linear_field_on_structured_fixture():
    config = SyntheticEuler2DConfig(nx=6, ny=5, num_steps=3)
    positions = make_structured_grid_positions(config)
    edges = make_structured_grid_edges(config)

    directed_edges, weights = compute_graph_edge_gradient_weights(positions, edges)
    field = 2.0 * positions[:, 0] - 3.0 * positions[:, 1] + 0.7
    gradients = np.zeros((positions.shape[0], 2), dtype=np.float64)
    for (target, source), weight in zip(directed_edges, weights, strict=True):
        gradients[target] += weight * (field[source] - field[target])

    expected = np.tile(np.array([2.0, -3.0]), (positions.shape[0], 1))
    np.testing.assert_allclose(gradients, expected, atol=1.0e-10)

def test_pcno_euler7_frame_batch_matches_checkpoint_feature_layout():
    config = SyntheticEuler2DConfig(nx=6, ny=4, num_steps=4)
    group = make_synthetic_cpg_trajectory(config)
    frame = make_cpg_graph_frame(group, 1, num_steps=1)

    batch = make_pcno_euler7_frame_batch(frame)

    num_nodes = config.nx * config.ny
    assert batch.x.shape == (1, num_nodes, 7)
    np.testing.assert_allclose(batch.x[0, :, :2], frame["pos"].astype(np.float32))
    np.testing.assert_allclose(batch.x[0, :, 2], 1.0)
    np.testing.assert_allclose(batch.x[0, :, 3:], frame["current_primitives"].astype(np.float32))
    assert batch.y.shape == (1, num_nodes, 4)
    assert batch.metadata["x_layout"] == "pos2_node_rho1_primitives4"


def test_pcno_euler7_frame_batch_allows_autoregressive_current_override():
    config = SyntheticEuler2DConfig(nx=5, ny=4, num_steps=4)
    group = make_synthetic_cpg_trajectory(config)
    frame = make_cpg_graph_frame(group, 1, num_steps=1)
    override = frame["current_primitives"] + np.array([0.1, 0.2, 0.3, 0.4])

    batch = make_pcno_euler7_frame_batch(
        frame,
        current_primitives=override,
        node_rho_policy="node_weights",
    )

    np.testing.assert_allclose(batch.x[0, :, 2], batch.node_weights[0, :, 0])
    np.testing.assert_allclose(batch.x[0, :, 3:], override.astype(np.float32))
