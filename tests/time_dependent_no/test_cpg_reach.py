from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from scripts.time_dependent_no.diagnose_cpg_characteristic_reach import main
from utility.time_dependent_no.cpg_reach import (
    build_incoming_travel_adjacency,
    compare_characteristic_and_model_reach,
    construct_pointwise_admissible_causal_pair,
    cpg_dependency_mask,
    cpg_dependency_trace,
    euler_characteristic_travel_times,
    minimum_predecessor_times,
    normalize_scale_positions,
    validate_release_graph_mapping,
)
from utility.time_dependent_no.cpg_release import CPG_REFERENCE_COMMIT


def _chain_graph(nodes: int, spacing: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    pos = np.column_stack(
        (np.arange(nodes, dtype=np.float64) * spacing, np.zeros(nodes))
    )
    edges = np.column_stack(
        (np.arange(nodes - 1, dtype=np.int64), np.arange(1, nodes, dtype=np.int64))
    )
    return pos, edges


def _uniform_state(nodes: int, velocity: float = 2.0) -> np.ndarray:
    state = np.ones((nodes, 4), dtype=np.float64)
    state[:, 1] = velocity
    state[:, 2] = 0.0
    return state


def _model_geometry(
    pos: np.ndarray, edges: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model_pos = normalize_scale_positions(pos)
    directed = np.concatenate((edges, edges[:, ::-1]), axis=0)
    displacement = model_pos[directed[:, 0]] - model_pos[directed[:, 1]]
    distance = np.linalg.norm(displacement, axis=1, keepdims=True)
    edge_attr = np.concatenate((displacement, distance), axis=1)
    return model_pos, directed, edge_attr


def test_pinned_dependency_trace_adds_flux_aggregation_hop():
    trace = cpg_dependency_trace(12)

    assert trace.final_output_radius == 13
    assert trace.reference_commit == CPG_REFERENCE_COMMIT
    assert trace.to_dict()["support_kind"] == (
        "exact_architectural_current_state_support"
    )


def test_dependency_set_is_exact_radius_ball_on_chain():
    _, edges = _chain_graph(20)

    mask, distance = cpg_dependency_mask(
        edges, 20, target_node=19, message_passing_layers=12
    )

    np.testing.assert_array_equal(np.flatnonzero(mask), np.arange(6, 20))
    assert distance[6] == 13
    assert distance[5] == -1


def test_release_graph_mapping_checks_exact_edge_layout_and_features():
    pos, edges = _chain_graph(4)
    model_pos, directed, edge_attr = _model_geometry(pos, edges)

    result = validate_release_graph_mapping(
        raw_pos=pos,
        raw_edges=edges,
        model_pos=model_pos,
        directed_edges=directed,
        edge_attr_before_model=edge_attr,
    )

    assert result["dataset_node_to_model_node"] == "verified_index_identity"
    assert result["physical_face_mapping"] == "missing"

    bad = directed.copy()
    bad[[0, 1]] = bad[[1, 0]]
    try:
        validate_release_graph_mapping(
            raw_pos=pos,
            raw_edges=edges,
            model_pos=model_pos,
            directed_edges=bad,
            edge_attr_before_model=edge_attr,
        )
    except ValueError as error:
        assert "raw unique edges followed by reversals" in str(error)
    else:
        raise AssertionError("bad directed edge order was accepted")


def test_directed_characteristic_disallows_nonpositive_reverse_traversal():
    pos, edges = _chain_graph(2)
    state = _uniform_state(2, velocity=2.0)

    result = euler_characteristic_travel_times(pos, edges, state)

    assert np.isfinite(result["plus_travel_time"][0])
    assert np.isinf(result["plus_travel_time"][1])
    assert np.all(np.isfinite(result["symmetric_travel_time"]))
    np.testing.assert_allclose(result["plus_speed_max"][0], 2.0 + np.sqrt(1.4))


def test_reach_comparison_and_causal_pair_respect_dependency_support():
    pos, edges = _chain_graph(3, spacing=0.1)
    state = _uniform_state(3, velocity=2.0)
    travel = euler_characteristic_travel_times(pos, edges, state)
    incoming = build_incoming_travel_adjacency(
        travel["directed_edges"], travel["plus_travel_time"], 3
    )
    symmetric_incoming = build_incoming_travel_adjacency(
        travel["directed_edges"], travel["symmetric_travel_time"], 3
    )
    predecessor_time, next_hop = minimum_predecessor_times(incoming, 2, cutoff=0.1)
    symmetric_time, _ = minimum_predecessor_times(symmetric_incoming, 2, cutoff=0.1)
    summary, arrays = compare_characteristic_and_model_reach(
        edges=edges,
        num_nodes=3,
        target_node=2,
        directed_predecessor_time=predecessor_time,
        symmetric_predecessor_time=symmetric_time,
        macro_dt=0.1,
        message_passing_layers=0,
    )

    assert summary["model_dependency_radius"] == 1
    assert summary["directed_uncovered_count"] == 1
    assert arrays["directed_uncovered_mask"][0]

    pair = construct_pointwise_admissible_causal_pair(
        base_primitive=state,
        pos=pos,
        edges=edges,
        target_node=2,
        dependency_mask=arrays["dependency_mask"],
        predecessor_time=predecessor_time,
        next_hop=next_hop,
        macro_dt=0.1,
        normal_mask=np.ones(3, dtype=bool),
    )

    assert pair is not None
    assert pair["source_node"] == 0
    assert pair["perturbed_node_count"] == 1
    np.testing.assert_array_equal(
        pair["state_a"][arrays["dependency_mask"]],
        pair["state_b"][arrays["dependency_mask"]],
    )
    assert np.all(pair["state_b"][:, 0] > 0.0)
    assert np.all(pair["state_b"][:, 3] > 0.0)


def _write_synthetic_release_artifact(path: Path) -> None:
    nodes = 15
    pos, edges = _chain_graph(nodes, spacing=0.01)
    model_pos, directed, edge_attr = _model_geometry(pos, edges)
    state = _uniform_state(nodes, velocity=2.0).astype(np.float32)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("reference_current", data=state[None])
        handle.create_dataset("targets", data=state[None])
        handle.create_dataset("pos", data=pos.astype(np.float32))
        handle.create_dataset("edges", data=edges)
        handle.create_dataset("node_type", data=np.zeros(nodes, dtype=np.int64))
        handle.create_dataset("Mach", data=np.full(nodes, 3.0, dtype=np.float32))
        handle.create_dataset("injection_mask", data=np.zeros(nodes, dtype=bool))
        handle.create_dataset("model_pos", data=model_pos.astype(np.float32))
        handle.create_dataset("directed_edges", data=directed)
        handle.create_dataset(
            "edge_attr_before_model", data=edge_attr.astype(np.float32)
        )
        handle.attrs["reference_commit"] = CPG_REFERENCE_COMMIT
        handle.attrs["boundary_mode"] = "oracle_next_reference"
        handle.attrs["trajectory_key"] = "synthetic"
        handle.attrs["dt"] = 0.1


def test_characteristic_reach_cli_writes_bounded_causal_candidate(tmp_path: Path):
    artifact = tmp_path / "1.h5"
    output = tmp_path / "reach"
    _write_synthetic_release_artifact(artifact)

    assert (
        main(
            [
                "--artifact",
                str(artifact),
                "--output-dir",
                str(output),
                "--frames",
                "0",
                "--target-node",
                "14",
                "--max-causal-pairs",
                "1",
            ]
        )
        == 0
    )

    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["row_count"] == 1
    assert summary["rows_with_directed_cone_outside_support"] == 1
    assert len(summary["causal_pairs"]) == 1
    pair_path = output / summary["causal_pairs"][0]["file"]
    assert pair_path.is_file()
    with np.load(pair_path) as pair:
        dependency = pair["dependency_mask"].astype(bool)
        np.testing.assert_array_equal(
            pair["state_a"][dependency], pair["state_b"][dependency]
        )
