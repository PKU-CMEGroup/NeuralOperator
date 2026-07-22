from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from scripts.time_dependent_no.audit_cpg_mesh_contract import main
from scripts.time_dependent_no.evaluate_cpg_release import (
    LEGAL_BOUNDARY_MODE,
    load_mesh_audit,
    parse_args as parse_evaluator_args,
)
from utility.time_dependent_no.cpg_mesh_contract import (
    INFLOW_NODE,
    NORMAL_NODE,
    OUTFLOW_NODE,
    WALL_NODE,
    apply_causal_nodal_boundaries,
    apply_torch_boundary_policy,
    audit_bump_julia_config,
    build_boundary_stencil,
    build_torch_boundary_policy,
    expected_cpg_node_types,
    freestream_primitive,
    mesh_primal_edges,
    parse_abaqus_mesh,
    read_ascii_vtu,
    recover_boundary_geometry,
    recover_graph_boundary_geometry,
    validate_hdf_mesh_identity,
    validate_vtu_mesh_identity,
    vtu_primitive,
)
from utility.time_dependent_no.cpg_release import sha256_file


def _write_mesh(path: Path) -> None:
    path.write_text(
        """*Heading
synthetic bump mesh
*NODE
1, 0.0, 0.0, 0.0
2, 0.5, 0.0, 0.0
3, 1.0, 0.0, 0.0
4, 0.0, 0.5, 0.0
5, 0.5, 0.5, 0.0
6, 1.0, 0.5, 0.0
7, 0.0, 1.0, 0.0
8, 0.5, 1.0, 0.0
9, 1.0, 1.0, 0.0
******* E L E M E N T S *************
*ELEMENT, type=T3D2, ELSET=BoundaryLines
1, 1, 2
2, 2, 3
3, 3, 6
4, 6, 9
5, 9, 8
6, 8, 7
7, 7, 4
8, 4, 1
*ELEMENT, type=CPS4, ELSET=Surface1
9, 1, 2, 5, 4
10, 2, 3, 6, 5
11, 4, 5, 8, 7
12, 5, 6, 9, 8
*ELSET,ELSET=Bottom
1,
*ELSET,ELSET=Wall
2,
*ELSET,ELSET=Right
3, 4,
*ELSET,ELSET=Top
5, 6,
*ELSET,ELSET=Left
7, 8,
*ELSET,ELSET=PhysicalSurface6,GENERATE
9, 12, 1
*NSET,NSET=Bottom
1, 2,
*NSET,NSET=Wall
2, 3,
*NSET,NSET=Right
3, 6, 9,
*NSET,NSET=Top
7, 8, 9,
*NSET,NSET=Left
1, 4, 7,
*NSET,NSET=PhysicalSurface6,GENERATE
1, 9, 1
""",
        encoding="utf-8",
    )


def _synthetic_mesh(tmp_path: Path):
    path = tmp_path / "Bump.inp"
    _write_mesh(path)
    return parse_abaqus_mesh(path)


def _write_vtu(path: Path, mesh, primitive: np.ndarray) -> None:
    node_index = {int(node_id): index for index, node_id in enumerate(mesh.node_ids)}
    cells = [
        [node_index[node_id] for node_id in element.node_ids]
        for element in mesh.elements
    ]
    connectivity = [node for cell in cells for node in cell]
    offsets = np.cumsum([len(cell) for cell in cells])
    types = [3 if len(cell) == 2 else 9 for cell in cells]
    point_values = np.column_stack((mesh.points, np.zeros(mesh.num_nodes))).reshape(-1)

    def values(array) -> str:
        return " ".join(str(value) for value in np.asarray(array).reshape(-1))

    path.write_text(
        f"""<?xml version="1.0"?>
<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">
  <UnstructuredGrid>
    <Piece NumberOfPoints="{mesh.num_nodes}" NumberOfCells="{len(cells)}">
      <PointData>
        <DataArray type="Float64" Name="p" format="ascii">{values(primitive[:, 3])}</DataArray>
        <DataArray type="Float64" Name="rho" format="ascii">{values(primitive[:, 0])}</DataArray>
        <DataArray type="Float64" Name="v1" format="ascii">{values(primitive[:, 1])}</DataArray>
        <DataArray type="Float64" Name="v2" format="ascii">{values(primitive[:, 2])}</DataArray>
      </PointData>
      <Points>
        <DataArray type="Float64" Name="Points" NumberOfComponents="3" format="ascii">{values(point_values)}</DataArray>
      </Points>
      <Cells>
        <DataArray type="Int64" Name="connectivity" format="ascii">{values(connectivity)}</DataArray>
        <DataArray type="Int64" Name="offsets" format="ascii">{values(offsets)}</DataArray>
        <DataArray type="UInt8" Name="types" format="ascii">{values(types)}</DataArray>
      </Cells>
    </Piece>
  </UnstructuredGrid>
</VTKFile>
""",
        encoding="utf-8",
    )


def test_mesh_contract_recovers_exact_vertex_graph_and_outward_normals(tmp_path: Path):
    mesh = _synthetic_mesh(tmp_path)
    edges = mesh_primal_edges(mesh)
    node_type = expected_cpg_node_types(mesh)

    summary, geometry = validate_hdf_mesh_identity(
        mesh,
        hdf_pos=mesh.points.astype(np.float32),
        hdf_edges=edges[:, ::-1],
        hdf_node_type=node_type[:, None],
    )

    assert mesh.num_nodes == 9
    assert edges.shape == (12, 2)
    assert summary["num_volume_quads"] == 4
    assert summary["planar_euler_characteristic"] == 1
    assert summary["hdf_graph_mapping"] == "verified_quadrilateral_primal_edge_set"
    assert summary["graph_native_boundary_mapping"] == (
        "verified_against_abaqus_boundary"
    )
    assert summary["graph_native_normal_max_abs_error"] == pytest.approx(0.0)
    assert summary["graph_node_control_volume_status"] == (
        "not_established_point_sample"
    )
    np.testing.assert_array_equal(
        node_type,
        [
            INFLOW_NODE,
            WALL_NODE,
            OUTFLOW_NODE,
            INFLOW_NODE,
            NORMAL_NODE,
            OUTFLOW_NODE,
            INFLOW_NODE,
            WALL_NODE,
            OUTFLOW_NODE,
        ],
    )
    np.testing.assert_allclose(
        geometry["node_normal"][[0, 3, 6]], np.tile([-1.0, 0.0], (3, 1))
    )
    np.testing.assert_allclose(
        geometry["node_normal"][[2, 5, 8]], np.tile([1.0, 0.0], (3, 1))
    )
    np.testing.assert_allclose(geometry["node_normal"][1], [0.0, -1.0])
    np.testing.assert_allclose(geometry["node_normal"][7], [0.0, 1.0])
    assert np.sum(geometry["lumped_vertex_area"]) == pytest.approx(1.0)

    graph_geometry = recover_graph_boundary_geometry(
        pos=mesh.points,
        edges=edges,
        node_type=node_type,
    )
    np.testing.assert_allclose(
        graph_geometry["node_normal"], geometry["node_normal"], atol=1.0e-12
    )
    np.testing.assert_allclose(
        graph_geometry["node_boundary_normal_coherence"],
        geometry["node_boundary_normal_coherence"],
        atol=1.0e-12,
    )


def test_graph_boundary_geometry_rejects_boundary_chord(tmp_path: Path):
    mesh = _synthetic_mesh(tmp_path)
    edges = np.vstack((mesh_primal_edges(mesh), [0, 8]))

    with pytest.raises(ValueError, match="simple boundary cycle"):
        recover_graph_boundary_geometry(
            pos=mesh.points,
            edges=edges,
            node_type=expected_cpg_node_types(mesh),
        )


def test_mesh_contract_rejects_non_mesh_graph_edge(tmp_path: Path):
    mesh = _synthetic_mesh(tmp_path)
    edges = mesh_primal_edges(mesh).copy()
    edges[0] = [0, 8]

    with pytest.raises(ValueError, match="not the quadrilateral primal-edge set"):
        validate_hdf_mesh_identity(
            mesh,
            hdf_pos=mesh.points,
            hdf_edges=edges,
            hdf_node_type=expected_cpg_node_types(mesh),
        )


def test_ascii_vtu_matches_abaqus_points_cells_and_primitives(tmp_path: Path):
    mesh = _synthetic_mesh(tmp_path)
    primitive = np.arange(mesh.num_nodes * 4, dtype=np.float64).reshape(
        mesh.num_nodes, 4
    )
    path = tmp_path / "sol_1.vtu"
    _write_vtu(path, mesh, primitive)

    frame = read_ascii_vtu(path)
    summary = validate_vtu_mesh_identity(mesh, frame)

    assert summary["vtu_to_abaqus_node_mapping"] == "verified_index_identity"
    assert summary["vtu_to_abaqus_cell_mapping"] == (
        "verified_element_id_order_identity"
    )
    np.testing.assert_allclose(vtu_primitive(frame), primitive)


def test_causal_nodal_boundary_operator_uses_only_current_interior_state(
    tmp_path: Path,
):
    mesh = _synthetic_mesh(tmp_path)
    edges = mesh_primal_edges(mesh)
    geometry = recover_boundary_geometry(mesh)
    node_type = geometry["node_type"]
    state = np.full((mesh.num_nodes, 4), 99.0)
    state[4] = [2.0, 3.0, 4.0, 5.0]
    stencil = build_boundary_stencil(
        pos=mesh.points,
        edges=edges,
        node_type=node_type,
        node_normal=geometry["node_normal"],
    )

    result = apply_causal_nodal_boundaries(
        state,
        node_type=node_type,
        node_normal=geometry["node_normal"],
        stencil=stencil,
        mach=2.5,
    )

    np.testing.assert_allclose(
        result[node_type == INFLOW_NODE], np.tile([1.4, 2.5, 0.0, 1.0], (3, 1))
    )
    np.testing.assert_allclose(
        result[node_type == OUTFLOW_NODE], np.tile([2.0, 3.0, 4.0, 5.0], (3, 1))
    )
    np.testing.assert_allclose(result[[1, 7], 0], 2.0)
    np.testing.assert_allclose(result[[1, 7], 1:3], np.tile([3.0, 0.0], (2, 1)))
    np.testing.assert_allclose(result[[1, 7], 3], 5.0)
    np.testing.assert_allclose(result[4], state[4])
    assert stencil.fallback_target_count == 0

    nonphysical = state.copy()
    nonphysical[4, 3] = -2.0
    raw = apply_causal_nodal_boundaries(
        nonphysical,
        node_type=node_type,
        node_normal=geometry["node_normal"],
        stencil=stencil,
        mach=2.5,
    )
    assert np.all(raw[node_type == OUTFLOW_NODE, 3] == -2.0)

    sharp_coherence = geometry["node_boundary_normal_coherence"].copy()
    sharp_coherence[1] = 0.7
    sharp_result = apply_causal_nodal_boundaries(
        state,
        node_type=node_type,
        node_normal=geometry["node_normal"],
        wall_normal_coherence=sharp_coherence,
        stencil=stencil,
        mach=2.5,
    )
    np.testing.assert_allclose(sharp_result[1, 1:3], 0.0)
    np.testing.assert_allclose(sharp_result[7, 1:3], [3.0, 0.0])

    torch = pytest.importorskip("torch")
    torch_policy = build_torch_boundary_policy(
        torch=torch,
        device=torch.device("cpu"),
        node_type=node_type,
        node_normal=geometry["node_normal"],
        wall_normal_coherence=sharp_coherence,
        stencil=stencil,
        mach=2.5,
        config={"gamma": 1.4, "rho_inf": 1.4, "p_inf": 1.0},
    )
    torch_result = apply_torch_boundary_policy(
        torch, torch.as_tensor(state, dtype=torch.float32), torch_policy
    )
    np.testing.assert_allclose(torch_result.numpy(), sharp_result, rtol=1.0e-6)


def test_bump_config_and_freestream_contract(tmp_path: Path):
    path = tmp_path / "Bump.jl"
    path.write_text(
        """const gamma = 1.4
const rho_inf = 1.4
const p_inf = 1.0
boundary_condition_inflow = BoundaryConditionDirichlet(initial_condition_mach3_flow)
flux = Trixi.flux(u_inner, normal_direction, equations)
boundary_condition_slip_wall
polydeg = 3
mesh = P4estMesh{2}(mesh_file)
save_solution = SaveSolutionCallback(dt = 0.025)
""",
        encoding="utf-8",
    )

    config = audit_bump_julia_config(path)

    assert config["polydeg"] == 3
    assert config["save_dt"] == pytest.approx(0.025)
    assert config["boundary_contract"]["wall"] == "slip_wall"
    np.testing.assert_allclose(freestream_primitive(2.5), [1.4, 2.5, 0.0, 1.0])


def test_legal_evaluator_requires_completed_mesh_contract_inputs(tmp_path: Path):
    required = [
        "--reference-repo",
        str(tmp_path / "reference"),
        "--dataset-root",
        str(tmp_path / "data"),
        "--checkpoint",
        str(tmp_path / "model.pth"),
        "--output-dir",
        str(tmp_path / "output"),
        "--boundary-mode",
        LEGAL_BOUNDARY_MODE,
    ]

    with pytest.raises(ValueError, match="requires --mesh-audit-summary"):
        parse_evaluator_args(required)


def test_mesh_audit_cli_binds_hdf_to_one_based_raw_case_and_vtu_offset(
    tmp_path: Path,
):
    raw_root = tmp_path / "raw"
    case_dir = raw_root / "1"
    output = tmp_path / "audit"
    (case_dir / "outFO").mkdir(parents=True)
    _write_mesh(case_dir / "Bump.inp")
    mesh = parse_abaqus_mesh(case_dir / "Bump.inp")
    (case_dir / "Mach.txt").write_text("2.5\n", encoding="utf-8")
    (case_dir / "Bump.jl").write_text(
        """const gamma = 1.4
const rho_inf = 1.4
const p_inf = 1.0
boundary_condition_inflow = BoundaryConditionDirichlet(initial_condition_mach3_flow)
flux = Trixi.flux(u_inner, normal_direction, equations)
boundary_condition_slip_wall
polydeg = 3
mesh = P4estMesh{2}(mesh_file)
save_solution = SaveSolutionCallback(dt = 0.025)
""",
        encoding="utf-8",
    )
    base = np.tile([1.4, 2.5, 0.0, 1.0], (mesh.num_nodes, 1))
    states = [base.copy() for _ in range(3)]
    states[1][4] = [1.6, 2.4, 0.1, 1.2]
    states[2][4] = [1.8, 2.3, 0.2, 1.4]
    for index, state in enumerate(states):
        _write_vtu(case_dir / "outFO" / f"sol_{index}.vtu", mesh, state)

    dataset_path = tmp_path / "test.h5"
    edges = mesh_primal_edges(mesh)
    node_type = expected_cpg_node_types(mesh)
    hdf_state = np.stack(states[1:], axis=0)
    with h5py.File(dataset_path, "w") as handle:
        group = handle.create_group("00")
        group.create_dataset("pos", data=np.tile(mesh.points[None], (2, 1, 1)))
        group.create_dataset("edges", data=np.tile(edges[None], (2, 1, 1)))
        group.create_dataset(
            "node_type", data=np.tile(node_type[None, :, None], (2, 1, 1))
        )
        group.create_dataset(
            "Mach", data=np.full((2, mesh.num_nodes, 1), 2.5, dtype=np.float64)
        )
        for channel, key in enumerate(("rho", "v1", "v2", "pres")):
            group.create_dataset(key, data=hdf_state[:, :, channel, None])

    assert (
        main(
            [
                "--dataset-h5",
                str(dataset_path),
                "--raw-case-root",
                str(raw_root),
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )

    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "valid"
    assert summary["case_count"] == 1
    assert summary["temporal_alignment_offsets"] == [1]
    assert summary["cases"][0]["raw_case_id"] == 1
    assert summary["cases"][0]["graph_to_dg_status"] == (
        "incompatible_with_one_to_one_nominal_dg_dof_count"
    )
    assert (output / summary["cases"][0]["geometry_artifact"]).is_file()

    audit, records = load_mesh_audit(
        output / "summary.json",
        dataset_sha256=sha256_file(dataset_path),
        selected_keys=["00"],
    )
    assert audit["status"] == "valid"
    assert records["00"]["raw_case_id"] == 1
    with pytest.raises(ValueError, match="dataset SHA256 differ"):
        load_mesh_audit(
            output / "summary.json",
            dataset_sha256="0" * 64,
            selected_keys=["00"],
        )
