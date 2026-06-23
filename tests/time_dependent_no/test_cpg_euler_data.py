import numpy as np

from utility.time_dependent_no.euler2d import (
    EulerNodeType,
    conservative_to_primitive,
    inspect_cpg_hdf5_mapping,
    inspect_cpg_trajectory,
    load_cpg_primitive_sequence,
    make_cpg_graph_frame,
    node_type_masks,
    primitive_to_conservative,
    stack_primitive,
)


def _fake_cpg_group(num_steps=4, num_nodes=5):
    time = np.arange(num_steps, dtype=np.float64)[:, None, None]
    nodes = np.arange(num_nodes, dtype=np.float64)[None, :, None]
    rho = 1.0 + 0.1 * nodes + 0.01 * time
    v1 = 0.2 + 0.01 * time + np.zeros_like(nodes)
    v2 = -0.1 + 0.02 * nodes + np.zeros_like(time)
    pres = 1.0 + 0.2 * (nodes >= 2.0) + 0.01 * time
    pos = np.zeros((num_steps, num_nodes, 2), dtype=np.float64)
    pos[..., 0] = np.linspace(0.0, 1.0, num_nodes)[None, :]
    edges = np.tile(
        np.array([[0, 1], [1, 2], [2, 3], [3, 4]], dtype=np.int64)[None, :, :],
        (num_steps, 1, 1),
    )
    node_type = np.tile(
        np.array(
            [
                EulerNodeType.NORMAL,
                EulerNodeType.WALL,
                EulerNodeType.OUTFLOW,
                EulerNodeType.INFLOW,
                EulerNodeType.NORMAL,
            ],
            dtype=np.int64,
        )[None, :, None],
        (num_steps, 1, 1),
    )
    mach = np.full((num_steps, num_nodes, 1), 2.0, dtype=np.float64)
    return {
        "pos": pos,
        "edges": edges,
        "node_type": node_type,
        "rho": rho,
        "v1": v1,
        "v2": v2,
        "pres": pres,
        "Mach": mach,
    }


def test_euler_primitive_conservative_roundtrip():
    primitive = stack_primitive(
        np.array([1.0, 2.0]),
        np.array([0.2, -0.1]),
        np.array([0.0, 0.3]),
        np.array([1.0, 2.5]),
    )

    conservative = primitive_to_conservative(primitive)
    recovered = conservative_to_primitive(conservative)

    np.testing.assert_allclose(recovered, primitive)


def test_node_type_masks_follow_cpg_codes():
    node_type = np.array([[0], [1], [2], [3], [0]])
    masks = node_type_masks(node_type)

    assert masks["normal"].tolist() == [True, False, False, False, True]
    assert masks["boundary"].tolist() == [False, True, True, True, False]


def test_inspect_cpg_trajectory_reports_schema_and_ranges():
    summary = inspect_cpg_trajectory(_fake_cpg_group(), name="traj0")

    assert summary.name == "traj0"
    assert summary.missing_keys == []
    assert summary.num_time_steps == 4
    assert summary.num_nodes == 5
    assert summary.num_edges == 4
    assert summary.node_type_counts == {0: 2, 1: 1, 2: 1, 3: 1}
    assert summary.arrays["rho"].min is not None
    assert not summary.warnings


def test_inspect_cpg_hdf5_mapping_limits_trajectories():
    mapping = {"b": _fake_cpg_group(), "a": _fake_cpg_group()}

    summary = inspect_cpg_hdf5_mapping(mapping, path="fake.h5", max_trajectories=1)

    assert summary.path == "fake.h5"
    assert summary.num_trajectories == 2
    assert summary.inspected_trajectories == 1
    assert summary.trajectory_names == ["a", "b"]
    assert "inspected first 1 of 2 trajectories" in summary.warnings
    assert summary.to_dict()["trajectories"][0]["name"] == "a"


def test_load_cpg_primitive_sequence_and_graph_frame_match_reference_order():
    group = _fake_cpg_group()

    primitive = load_cpg_primitive_sequence(group)
    frame = make_cpg_graph_frame(group, 1, num_steps=2)

    assert primitive.shape == (4, 5, 4)
    assert frame["x"].shape == (5, 6)
    assert frame["y"].shape == (5, 4)
    assert frame["future_primitives"].shape == (2, 5, 4)
    np.testing.assert_allclose(frame["x"][:, 1:5], primitive[1])
    np.testing.assert_allclose(frame["y"], primitive[2])
    np.testing.assert_allclose(frame["future_primitives"], primitive[2:4])
    assert frame["edges"].shape == (4, 2)


def test_make_cpg_graph_frame_accepts_time_node_matrices():
    group = _fake_cpg_group()
    group["node_type"] = np.squeeze(group["node_type"], axis=-1)
    group["Mach"] = np.squeeze(group["Mach"], axis=-1)

    frame = make_cpg_graph_frame(group, 1)

    assert frame["x"].shape == (5, 6)
    assert frame["node_type"].shape == (5,)
    np.testing.assert_allclose(frame["mach"], np.full(5, 2.0))

