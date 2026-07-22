"""Audit CPG HDF5 graphs against extracted Abaqus and VTU bump cases.

This is a provenance and geometry gate, not a rollout.  For each selected test
trajectory it verifies the expected zero-based HDF group to one-based raw-case
mapping, exact point order, quadrilateral primal-edge topology, boundary node
labels, outward normals, VTU cell identity, and a sampled first/midpoint/last
HDF-to-VTU offset check.  The sampled offset is not full temporal identity.

The output includes compact geometry NPZ files for a later frozen legal-
boundary counterfactual.  It explicitly does not claim graph vertices are
control volumes, graph edges are control-volume faces, or VTU point samples are
one-to-one DG degrees of freedom.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import h5py  # noqa: E402
import numpy as np  # noqa: E402

from utility.time_dependent_no.cpg_mesh_contract import (  # noqa: E402
    INFLOW_NODE,
    MESH_CONTRACT_SCHEMA,
    OUTFLOW_NODE,
    WALL_NODE,
    apply_causal_nodal_boundaries,
    audit_bump_julia_config,
    boundary_stencil_sha256,
    build_boundary_stencil,
    freestream_primitive,
    parse_abaqus_mesh,
    primitive_error,
    read_ascii_vtu,
    validate_hdf_mesh_identity,
    validate_vtu_mesh_identity,
    vtu_primitive,
)
from utility.time_dependent_no.cpg_release import (  # noqa: E402
    CPG_LEGAL_BOUNDARY_MAX_SOURCE_HOPS,
    CPG_MESH_AUDIT_SCHEMA,
    sha256_file,
    validate_cpg_model_dt,
    validate_cpg_static_graph,
)


AUDIT_SCHEMA = CPG_MESH_AUDIT_SCHEMA
DEFAULT_POSITION_ATOL = 2.0e-6
DEFAULT_PRIMITIVE_ATOL = 2.0e-10
DEFAULT_SOURCE_HOPS = CPG_LEGAL_BOUNDARY_MAX_SOURCE_HOPS


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-h5", type=Path, required=True)
    parser.add_argument(
        "--raw-case-root",
        type=Path,
        required=True,
        help="Directory containing one-based extracted case folders",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--trajectory-key",
        action="append",
        default=[],
        help="Explicit HDF group; defaults to every group",
    )
    parser.add_argument("--case-offset", type=int, default=1)
    parser.add_argument("--position-atol", type=float, default=DEFAULT_POSITION_ATOL)
    parser.add_argument("--primitive-atol", type=float, default=DEFAULT_PRIMITIVE_ATOL)
    parser.add_argument(
        "--max-boundary-source-hops", type=int, default=DEFAULT_SOURCE_HOPS
    )
    parser.add_argument("--expected-dataset-sha256")
    return parser.parse_args(argv)


def select_trajectory_keys(
    available: Sequence[str], requested: Sequence[str]
) -> list[str]:
    keys = list(available)
    if requested:
        if len(set(requested)) != len(requested):
            raise ValueError("trajectory keys must be unique")
        missing = [key for key in requested if key not in keys]
        if missing:
            raise KeyError(f"requested trajectory keys are absent: {missing}")
        return list(requested)
    return sorted(keys, key=lambda key: (0, int(key)) if key.isdigit() else (1, key))


def raw_case_id(trajectory_key: str, case_offset: int) -> int:
    if not trajectory_key.isdigit():
        raise ValueError(
            f"trajectory key {trajectory_key!r} cannot use integer case mapping"
        )
    case_id = int(trajectory_key) + case_offset
    if case_id < 1:
        raise ValueError("raw case id must be one-based and positive")
    return case_id


def primitive_frame(group: Any, frame: int) -> np.ndarray:
    arrays = []
    for key in ("rho", "v1", "v2", "pres"):
        values = np.asarray(group[key][frame], dtype=np.float64)
        if values.ndim == 1:
            values = values[:, None]
        if values.ndim != 2 or values.shape[1] != 1:
            raise ValueError(f"{key} frame has unsupported shape {values.shape}")
        arrays.append(values)
    primitive = np.concatenate(arrays, axis=1)
    if not np.all(np.isfinite(primitive)):
        raise ValueError(f"HDF primitive frame {frame} contains nonfinite values")
    return primitive


def choose_vtu_alignment(
    *,
    hdf_frames: dict[int, np.ndarray],
    vtu_frames: dict[int, np.ndarray],
    primitive_atol: float,
) -> tuple[int, dict[str, Any]]:
    if not hdf_frames:
        raise ValueError("HDF-to-VTU temporal alignment needs sampled HDF frames")
    if len(hdf_frames) < 2:
        raise ValueError(
            "HDF-to-VTU temporal alignment needs at least two sampled HDF frames"
        )
    if not np.isfinite(primitive_atol) or primitive_atol <= 0.0:
        raise ValueError("primitive_atol must be finite and positive")
    candidates: dict[str, Any] = {}
    for offset in (0, 1):
        if any(index + offset not in vtu_frames for index in hdf_frames):
            continue
        frame_errors = {
            str(index): {
                "vtu_index": index + offset,
                **primitive_error(state, vtu_frames[index + offset]),
            }
            for index, state in sorted(hdf_frames.items())
        }
        candidates[str(offset)] = {
            "sampled_hdf_frames": sorted(hdf_frames),
            "frames": frame_errors,
            "score": max(error["max_abs"] for error in frame_errors.values()),
        }
    if not candidates:
        raise ValueError("no complete HDF-to-VTU temporal alignment candidate exists")
    acceptable = [
        key
        for key, candidate in candidates.items()
        if candidate["score"] <= primitive_atol
    ]
    if not acceptable:
        best = min(candidate["score"] for candidate in candidates.values())
        raise ValueError(
            "no HDF-to-VTU temporal alignment candidate is within tolerance "
            f"(best max error {best})"
        )
    if len(acceptable) != 1:
        raise ValueError(
            "HDF-to-VTU temporal alignment is ambiguous within tolerance"
        )
    return int(acceptable[0]), candidates


def _boundary_reference_summary(
    group: Any,
    *,
    node_type: np.ndarray,
    node_normal: np.ndarray,
    wall_normal_coherence: np.ndarray,
    edges: np.ndarray,
    pos: np.ndarray,
    mach: float,
    config: dict[str, Any],
    max_source_hops: int,
) -> tuple[dict[str, Any], Any]:
    boundary_ids = np.flatnonzero(node_type != 0)
    field_arrays = []
    for key in ("rho", "v1", "v2", "pres"):
        values = np.asarray(group[key][:, boundary_ids, :], dtype=np.float64)
        field_arrays.append(values.reshape(values.shape[0], boundary_ids.size, 1))
    boundary_state = np.concatenate(field_arrays, axis=2)
    if not np.all(np.isfinite(boundary_state)):
        raise ValueError("HDF boundary reference contains nonfinite primitives")
    if np.any(boundary_state[:, :, 0] <= 0.0) or np.any(
        boundary_state[:, :, 3] <= 0.0
    ):
        raise ValueError("HDF boundary reference is not physically admissible")
    boundary_types = node_type[boundary_ids]
    boundary_normals = node_normal[boundary_ids]
    inflow = boundary_types == INFLOW_NODE
    wall = boundary_types == WALL_NODE
    outflow = boundary_types == OUTFLOW_NODE

    inflow_truth = boundary_state[:, inflow, :]
    inflow_reference = freestream_primitive(
        mach,
        gamma=float(config["gamma"]),
        rho_inf=float(config["rho_inf"]),
        p_inf=float(config["p_inf"]),
    )
    inflow_max_abs = float(np.max(np.abs(inflow_truth - inflow_reference)))

    wall_velocity = boundary_state[:, wall, 1:3]
    wall_normal = boundary_normals[wall]
    wall_normal_velocity = np.sum(wall_velocity * wall_normal[None, :, :], axis=2)

    outflow_state = boundary_state[:, outflow, :]
    outflow_normal = boundary_normals[outflow]
    outflow_normal_velocity = np.sum(
        outflow_state[:, :, 1:3] * outflow_normal[None, :, :], axis=2
    )
    outflow_sound_speed = np.sqrt(
        float(config["gamma"]) * outflow_state[:, :, 3] / outflow_state[:, :, 0]
    )
    reference_outflow_mach = outflow_normal_velocity / outflow_sound_speed
    if float(np.min(reference_outflow_mach)) <= 1.0:
        raise ValueError("reference outflow is not outward-supersonic at every frame")

    stencil = build_boundary_stencil(
        pos=pos,
        edges=edges,
        node_type=node_type,
        node_normal=node_normal,
        max_source_hops=max_source_hops,
    )
    unique_sources, source_inverse = np.unique(
        stencil.source_nodes, return_inverse=True
    )
    source_fields = []
    for key in ("rho", "v1", "v2", "pres"):
        values = np.asarray(
            group[key][:, unique_sources, :],
            dtype=np.float64,
        )
        source_fields.append(values[:, source_inverse, :])
    source_state = np.concatenate(source_fields, axis=2)
    if not np.all(np.isfinite(source_state)):
        raise ValueError("causal boundary source states contain nonfinite values")
    extrapolated = np.zeros(
        (source_state.shape[0], stencil.target_nodes.size, 4),
        dtype=np.float64,
    )
    for entry, (target_row, weight) in enumerate(
        zip(stencil.target_rows, stencil.weights, strict=True)
    ):
        extrapolated[:, int(target_row), :] += (
            source_state[:, entry, :] * float(weight)
        )
    outflow_rows = np.flatnonzero(
        node_type[stencil.target_nodes] == OUTFLOW_NODE
    )
    if outflow_rows.size == 0:
        raise ValueError("causal boundary stencil has no outflow targets")
    causal_outflow = extrapolated[:, outflow_rows, :]
    if (
        not np.all(np.isfinite(causal_outflow))
        or np.any(causal_outflow[:, :, 0] <= 0.0)
        or np.any(causal_outflow[:, :, 3] <= 0.0)
    ):
        raise ValueError("causal outflow extrapolation is not physically admissible")
    causal_outflow_nodes = stencil.target_nodes[outflow_rows]
    causal_outflow_normal = node_normal[causal_outflow_nodes]
    causal_outflow_normal_velocity = np.sum(
        causal_outflow[:, :, 1:3] * causal_outflow_normal[None, :, :],
        axis=2,
    )
    causal_outflow_sound_speed = np.sqrt(
        float(config["gamma"])
        * causal_outflow[:, :, 3]
        / causal_outflow[:, :, 0]
    )
    causal_outflow_mach = (
        causal_outflow_normal_velocity / causal_outflow_sound_speed
    )
    if float(np.min(causal_outflow_mach)) <= 1.0:
        raise ValueError(
            "causal outflow extrapolation is not outward-supersonic at every frame"
        )
    remapping: dict[str, list[float]] = {"wall": [], "outflow": [], "boundary": []}
    total_frames = int(group["rho"].shape[0])
    selected_frames = sorted(
        {0, total_frames // 4, total_frames // 2, total_frames - 1}
    )
    for frame in selected_frames:
        current = primitive_frame(group, frame)
        legal = apply_causal_nodal_boundaries(
            current,
            node_type=node_type,
            node_normal=node_normal,
            wall_normal_coherence=wall_normal_coherence,
            stencil=stencil,
            mach=mach,
            gamma=float(config["gamma"]),
            rho_inf=float(config["rho_inf"]),
            p_inf=float(config["p_inf"]),
        )
        for name, mask in (
            ("wall", node_type == WALL_NODE),
            ("outflow", node_type == OUTFLOW_NODE),
            ("boundary", node_type != 0),
        ):
            remapping[name].append(
                float(np.sqrt(np.mean((legal[mask] - current[mask]) ** 2)))
            )

    summary = {
        "inflow_freestream_max_abs_error": inflow_max_abs,
        "wall_normal_velocity_max_abs": float(np.max(np.abs(wall_normal_velocity))),
        "wall_normal_velocity_rms": float(np.sqrt(np.mean(wall_normal_velocity**2))),
        "outflow_normal_mach_min": float(
            np.min(reference_outflow_mach)
        ),
        "outflow_normal_mach_max": float(
            np.max(reference_outflow_mach)
        ),
        "causal_outflow_normal_mach_min": float(np.min(causal_outflow_mach)),
        "causal_outflow_normal_mach_max": float(np.max(causal_outflow_mach)),
        "causal_outflow_evidence": "all_frames_extrapolated_from_interior_stencil",
        "legal_nodal_remapping_frames": selected_frames,
        "legal_nodal_remapping_rmse": {
            name: {
                "mean": float(np.mean(values)),
                "max": float(np.max(values)),
                "per_frame": values,
            }
            for name, values in remapping.items()
        },
        "boundary_stencil_target_count": int(stencil.target_nodes.size),
        "boundary_stencil_entry_count": int(stencil.source_nodes.size),
        "boundary_stencil_fallback_target_count": stencil.fallback_target_count,
        "boundary_stencil_sha256": boundary_stencil_sha256(stencil),
        "sharp_wall_corner_count": int(
            np.count_nonzero((node_type == WALL_NODE) & (wall_normal_coherence < 0.95))
        ),
        "boundary_operator_status": (
            "causal_nodal_counterfactual_available_not_exact_dg_surface_flux"
        ),
    }
    return summary, stencil


def _write_geometry_npz(
    path: Path,
    *,
    trajectory_key: str,
    case_id: int,
    mesh: Any,
    geometry: dict[str, np.ndarray],
    stencil: Any,
    mach: float,
) -> None:
    np.savez_compressed(
        path,
        schema=np.asarray(MESH_CONTRACT_SCHEMA),
        trajectory_key=np.asarray(trajectory_key),
        raw_case_id=np.asarray(case_id, dtype=np.int64),
        mach=np.asarray(mach, dtype=np.float64),
        node_ids=mesh.node_ids,
        pos=mesh.points,
        primal_edges=geometry["primal_edges"],
        node_type=geometry["node_type"],
        node_normal=geometry["node_normal"],
        node_boundary_normal_coherence=geometry["node_boundary_normal_coherence"],
        node_boundary_measure=geometry["node_boundary_measure"],
        lumped_vertex_area=geometry["lumped_vertex_area"],
        boundary_edges=geometry["boundary_edges"],
        boundary_edge_normal=geometry["boundary_edge_normal"],
        boundary_edge_length=geometry["boundary_edge_length"],
        boundary_edge_node_type=geometry["boundary_edge_node_type"],
        stencil_target_nodes=stencil.target_nodes,
        stencil_target_rows=stencil.target_rows,
        stencil_source_nodes=stencil.source_nodes,
        stencil_weights=stencil.weights,
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _report(summary: dict[str, Any]) -> str:
    lines = [
        "# CPG bump mesh-contract audit",
        "",
        f"- Status: `{summary['status']}`",
        f"- Cases: `{summary['case_count']}`",
        f"- HDF-to-VTU offsets: `{summary['temporal_alignment_offsets']}`",
        "- Temporal alignment evidence: sampled first/midpoint/last frames only; "
        "full temporal identity was not established.",
        "- Graph nodes: verified Abaqus/VTU point samples.",
        "- Graph edges: verified quadrilateral primal mesh edges.",
        "- Boundary labels and outward normals: verified from named mesh sets.",
        "- Legal boundary status: causal nodal sensitivity counterfactual available.",
        "- Control-volume and physical-face status: not established.",
        "- DG mapping: not one-to-one at the source-derived nominal DOF count.",
        "",
        "The geometry artifact must not be used for exact finite-volume conservation or "
        "reference-flux claims. The nodal legal-boundary operator is not an exact replay "
        "of Trixi DG surface stages and a checkpoint trained with oracle boundaries is "
        "not thereby converted into a fairly trained autonomous baseline.",
        "",
    ]
    return "\n".join(lines)


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if (
        not np.isfinite(args.position_atol)
        or not np.isfinite(args.primitive_atol)
        or args.position_atol <= 0.0
        or args.primitive_atol <= 0.0
    ):
        raise ValueError("audit tolerances must be finite and positive")
    if args.case_offset != 1:
        raise ValueError("the CPG release contract requires --case-offset 1")
    if args.max_boundary_source_hops != CPG_LEGAL_BOUNDARY_MAX_SOURCE_HOPS:
        raise ValueError(
            "the CPG legal-boundary contract requires "
            f"--max-boundary-source-hops {CPG_LEGAL_BOUNDARY_MAX_SOURCE_HOPS}"
        )
    if not args.dataset_h5.is_file():
        raise FileNotFoundError(args.dataset_h5)
    if not args.raw_case_root.is_dir():
        raise FileNotFoundError(args.raw_case_root)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    geometry_dir = args.output_dir / "geometry"
    geometry_dir.mkdir()

    dataset_digest = sha256_file(args.dataset_h5)
    if (
        args.expected_dataset_sha256
        and dataset_digest.lower() != args.expected_dataset_sha256.lower()
    ):
        raise RuntimeError(
            f"dataset SHA256 {dataset_digest} does not match expected "
            f"{args.expected_dataset_sha256}"
        )

    case_records: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []
    with h5py.File(args.dataset_h5, "r") as dataset:
        keys = select_trajectory_keys(list(dataset.keys()), args.trajectory_key)
        for trajectory_key in keys:
            group = dataset[trajectory_key]
            case_id = raw_case_id(trajectory_key, args.case_offset)
            case_dir = args.raw_case_root / str(case_id)
            mesh_path = case_dir / "Bump.inp"
            julia_path = case_dir / "Bump.jl"
            mach_path = case_dir / "Mach.txt"
            if not all(path.is_file() for path in (mesh_path, julia_path, mach_path)):
                raise FileNotFoundError(f"raw case {case_id} is incomplete")

            mesh = parse_abaqus_mesh(mesh_path)
            pos, edges, node_type, mach_field = validate_cpg_static_graph(group)
            mesh_summary, geometry = validate_hdf_mesh_identity(
                mesh,
                hdf_pos=pos,
                hdf_edges=edges,
                hdf_node_type=node_type,
                position_atol=args.position_atol,
            )
            mach = float(mach_path.read_text(encoding="utf-8").strip())
            if not np.isfinite(mach) or not np.all(np.isfinite(mach_field)):
                raise ValueError(
                    f"trajectory {trajectory_key} Mach contains nonfinite values"
                )
            mach_error = float(np.max(np.abs(mach_field - mach)))
            if mach_error > args.primitive_atol:
                raise ValueError(
                    f"trajectory {trajectory_key} Mach field disagrees with raw case"
                )
            config = audit_bump_julia_config(julia_path)
            try:
                validate_cpg_model_dt(config["save_dt"])
            except ValueError as exc:
                raise ValueError(
                    f"raw case {case_id} does not use the pinned CPG model timestep"
                ) from exc

            hdf_frames = int(group["rho"].shape[0])
            sampled_hdf_indices = sorted({0, hdf_frames // 2, hdf_frames - 1})
            sampled_hdf_frames = {
                index: primitive_frame(group, index)
                for index in sampled_hdf_indices
            }
            vtu_indices = sorted(
                {
                    index + offset
                    for index in sampled_hdf_indices
                    for offset in (0, 1)
                }
            )
            vtu_states: dict[int, np.ndarray] = {}
            vtu_mesh_summaries: dict[str, Any] = {}
            vtu_hashes: dict[str, str] = {}
            for index in vtu_indices:
                vtu_path = case_dir / "outFO" / f"sol_{index}.vtu"
                if not vtu_path.is_file():
                    continue
                vtu = read_ascii_vtu(vtu_path)
                vtu_mesh_summaries[str(index)] = validate_vtu_mesh_identity(mesh, vtu)
                vtu_states[index] = vtu_primitive(vtu)
                vtu_hashes[str(index)] = sha256_file(vtu_path)
            selected_offset, alignment_candidates = choose_vtu_alignment(
                hdf_frames=sampled_hdf_frames,
                vtu_frames=vtu_states,
                primitive_atol=args.primitive_atol,
            )
            alignment = alignment_candidates[str(selected_offset)]

            boundary_summary, stencil = _boundary_reference_summary(
                group,
                node_type=node_type,
                node_normal=geometry["node_normal"],
                wall_normal_coherence=geometry["node_boundary_normal_coherence"],
                edges=edges,
                pos=pos,
                mach=mach,
                config=config,
                max_source_hops=args.max_boundary_source_hops,
            )
            nominal_dg_dofs = (
                mesh_summary["num_volume_quads"] * (int(config["polydeg"]) + 1) ** 2
            )
            dg_status = (
                "incompatible_with_one_to_one_nominal_dg_dof_count"
                if nominal_dg_dofs != mesh.num_nodes
                else "count_compatible_but_identity_unverified"
            )
            geometry_name = f"trajectory_{trajectory_key}_case_{case_id}.npz"
            _write_geometry_npz(
                geometry_dir / geometry_name,
                trajectory_key=trajectory_key,
                case_id=case_id,
                mesh=mesh,
                geometry=geometry,
                stencil=stencil,
                mach=mach,
            )

            record = {
                "trajectory_key": trajectory_key,
                "raw_case_id": case_id,
                "mach": mach,
                "mach_max_abs_error": mach_error,
                "mesh": mesh_summary,
                "vtu_mesh": vtu_mesh_summaries,
                "temporal_alignment_offset": selected_offset,
                "temporal_alignment_sampled_hdf_frames": sampled_hdf_indices,
                "temporal_alignment_candidates": alignment_candidates,
                "config": config,
                "nominal_discontinuous_dg_dofs_per_scalar": nominal_dg_dofs,
                "graph_point_count": mesh.num_nodes,
                "graph_to_dg_status": dg_status,
                "boundary": boundary_summary,
                "geometry_artifact": str(Path("geometry") / geometry_name),
                "source_sha256": {
                    "Bump.inp": sha256_file(mesh_path),
                    "Bump.jl": sha256_file(julia_path),
                    "Mach.txt": sha256_file(mach_path),
                    "VTU": vtu_hashes,
                },
            }
            case_records.append(record)
            csv_rows.append(
                {
                    "trajectory_key": trajectory_key,
                    "raw_case_id": case_id,
                    "mach": mach,
                    "num_nodes": mesh.num_nodes,
                    "num_volume_quads": mesh_summary["num_volume_quads"],
                    "num_primal_edges": mesh_summary["num_primal_edges"],
                    "num_boundary_edges": mesh_summary["num_boundary_edges"],
                    "position_max_abs_error": mesh_summary["position_max_abs_error"],
                    "temporal_alignment_offset": selected_offset,
                    "temporal_alignment_max_abs_error": alignment["score"],
                    "nominal_dg_dofs_per_scalar": nominal_dg_dofs,
                    "graph_to_dg_status": dg_status,
                    "inflow_freestream_max_abs_error": boundary_summary[
                        "inflow_freestream_max_abs_error"
                    ],
                    "wall_normal_velocity_max_abs": boundary_summary[
                        "wall_normal_velocity_max_abs"
                    ],
                    "outflow_normal_mach_min": boundary_summary[
                        "outflow_normal_mach_min"
                    ],
                    "boundary_stencil_fallback_targets": boundary_summary[
                        "boundary_stencil_fallback_target_count"
                    ],
                }
            )

    offsets = sorted({record["temporal_alignment_offset"] for record in case_records})
    config_signatures = {
        json.dumps(record["config"], sort_keys=True) for record in case_records
    }
    summary = {
        "schema": AUDIT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "valid"
        if len(offsets) == 1 and len(config_signatures) == 1
        else "invalid",
        "case_count": len(case_records),
        "dataset_sha256": dataset_digest,
        "expected_dataset_sha256": args.expected_dataset_sha256,
        "case_offset": args.case_offset,
        "position_atol": args.position_atol,
        "primitive_atol": args.primitive_atol,
        "max_boundary_source_hops": args.max_boundary_source_hops,
        "temporal_alignment_offsets": offsets,
        "temporal_alignment_evidence": (
            "sampled_first_midpoint_last_not_full_temporal_identity"
        ),
        "shared_case_configuration": len(config_signatures) == 1,
        "graph_mesh_identity": "verified_all_selected_cases",
        "static_graph_evidence": "all_frames_exact_release_facing_metadata",
        "boundary_geometry": "verified_all_selected_cases",
        "legal_boundary_counterfactual": (
            "enabled_causal_nodal_projection_not_exact_dg_boundary_replay"
        ),
        "solver_dof_identity": "not_recovered_one_to_one_count_incompatible",
        "control_volume_identity": "missing",
        "physical_face_identity": "missing",
        "physical_conservation_claim": "blocked",
        "paper_dataset_identity": "unresolved",
        "cases": case_records,
        "source_manifest": {
            "script": sha256_file(Path(__file__)),
            "mesh_utility": sha256_file(
                ROOT / "utility" / "time_dependent_no" / "cpg_mesh_contract.py"
            ),
            "provenance_utility": sha256_file(
                ROOT / "utility" / "time_dependent_no" / "cpg_release.py"
            ),
        },
    }
    if summary["status"] != "valid":
        raise RuntimeError(
            "selected cases do not share one mesh/configuration contract"
        )
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
            default=_json_scalar,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_csv(args.output_dir / "cases.csv", csv_rows)
    (args.output_dir / "report.md").write_text(_report(summary), encoding="utf-8")
    print(
        json.dumps(
            {
                key: summary[key]
                for key in ("status", "case_count", "temporal_alignment_offsets")
            },
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
