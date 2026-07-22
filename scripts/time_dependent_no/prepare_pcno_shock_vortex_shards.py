#!/usr/bin/env python3
"""Prepare validated shock--vortex FV trajectories for residual PCNO training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utility.time_dependent_no.pcno_euler2d import SCHEMA_VERSION  # noqa: E402
from utility.time_dependent_no.pcno_fv_geometry import (  # noqa: E402
    PCNOFiniteVolumeGeometry,
    build_pcno_finite_volume_geometry,
)
from utility.time_dependent_no.shock_vortex_family import (  # noqa: E402
    REFERENCE_ARTIFACT_SCHEMA,
    family_case_by_id,
    family_case_provenance,
    load_shock_vortex_family_manifest,
)

AUDIT_SCHEMA = "shock_vortex_family_artifact_audit_v1"
STATE_FLOAT32_RELATIVE_TOLERANCE = 1.0e-6
RESIDUAL_FLOAT32_RELATIVE_TOLERANCE = 1.0e-4
GEOMETRY_ARRAY_NAMES = (
    "cell_centers",
    "cell_volume",
    "face_owner",
    "face_neighbor",
    "face_boundary_tag",
)
SHARD_ARRAY_NAMES = (
    "states_conservative",
    "physical_times",
    "parameters",
    "nodes",
    "edges",
    "node_type",
    "node_measures",
    "node_weights",
    "node_rhos",
    "directed_edges",
    "edge_gradient_weights",
    "mesh_cell_to_graph_node",
    "face_to_directed_edge",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scope", choices=("smoke", "complete"), required=True)
    parser.add_argument("--gradient-rcond", type=float, default=1.0e-12)
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _folder_name(case_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", case_id).strip("._") or "trajectory"
    suffix = hashlib.sha256(case_id.encode("utf-8")).hexdigest()[:8]
    return f"traj_{safe}_{suffix}"


def _save_array(
    folder: Path,
    name: str,
    value: np.ndarray,
    dtype: np.dtype[Any],
) -> None:
    np.save(folder / f"{name}.npy", np.asarray(value, dtype=dtype))


def _load_source_audit(
    family_root: Path,
    *,
    scope: str,
    manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    audit_name = "smoke_audit.json" if scope == "smoke" else "complete_audit.json"
    audit_path = family_root / audit_name
    if not audit_path.is_file():
        raise FileNotFoundError(
            f"{scope} shard preparation requires the passed source audit: {audit_path}"
        )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("schema") != AUDIT_SCHEMA or audit.get("status") != "passed":
        raise ValueError(f"source family audit is not a passed {AUDIT_SCHEMA}")
    if audit.get("manifest_digest_sha256") != manifest["manifest_digest_sha256"]:
        raise ValueError("source audit and family manifest digests differ")
    expected = (
        list(manifest["smoke_case_ids"])
        if scope == "smoke"
        else [case["case_id"] for case in manifest["cases"]]
    )
    requested = [str(value) for value in audit.get("requested_case_ids", [])]
    if requested != expected:
        raise ValueError("source audit case order differs from the declared scope")
    if int(audit.get("passed_case_count", -1)) != len(expected):
        raise ValueError("source audit did not pass every requested case")
    return audit, expected


def _audit_rows_by_case(audit: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows = {str(row["case_id"]): dict(row) for row in audit.get("rows", [])}
    if len(rows) != len(audit.get("rows", [])):
        raise ValueError("source audit contains duplicate case rows")
    return rows


def _serialization_metrics(states: np.ndarray) -> dict[str, float]:
    states64 = np.asarray(states, dtype=np.float64)
    states32 = states64.astype(np.float32).astype(np.float64)
    residual64 = np.diff(states64, axis=0)
    residual32 = np.diff(states32, axis=0)
    state_delta = states32 - states64
    residual_delta = residual32 - residual64
    state_frame_denominator = np.maximum(
        np.linalg.norm(states64.reshape(states64.shape[0], -1), axis=1),
        1.0e-30,
    )
    residual_frame_denominator = np.maximum(
        np.linalg.norm(residual64.reshape(residual64.shape[0], -1), axis=1),
        1.0e-30,
    )
    return {
        "state_global_relative_l2": float(
            np.linalg.norm(state_delta) / max(np.linalg.norm(states64), 1.0e-30)
        ),
        "state_max_frame_relative_l2": float(
            np.max(
                np.linalg.norm(state_delta.reshape(state_delta.shape[0], -1), axis=1)
                / state_frame_denominator
            )
        ),
        "residual_global_relative_l2": float(
            np.linalg.norm(residual_delta) / max(np.linalg.norm(residual64), 1.0e-30)
        ),
        "residual_max_frame_relative_l2": float(
            np.max(
                np.linalg.norm(
                    residual_delta.reshape(residual_delta.shape[0], -1), axis=1
                )
                / residual_frame_denominator
            )
        ),
    }


def _serialization_passes(metrics: Mapping[str, float]) -> bool:
    return bool(
        metrics["state_global_relative_l2"] <= STATE_FLOAT32_RELATIVE_TOLERANCE
        and metrics["state_max_frame_relative_l2"] <= STATE_FLOAT32_RELATIVE_TOLERANCE
        and metrics["residual_global_relative_l2"]
        <= RESIDUAL_FLOAT32_RELATIVE_TOLERANCE
        and metrics["residual_max_frame_relative_l2"]
        <= RESIDUAL_FLOAT32_RELATIVE_TOLERANCE
    )


def _build_geometry(
    artifact: Mapping[str, np.ndarray],
    *,
    gradient_rcond: float,
) -> PCNOFiniteVolumeGeometry:
    names = tuple(json.loads(artifact["boundary_tag_names_json"].item()))
    return build_pcno_finite_volume_geometry(
        cell_centers=artifact["cell_centers"],
        cell_volume=artifact["cell_volume"],
        face_owner=artifact["face_owner"],
        face_neighbor=artifact["face_neighbor"],
        face_boundary_tag=artifact["face_boundary_tag"],
        boundary_tag_names=names,
        gradient_rcond=gradient_rcond,
    )


def _geometry_digest(geometry: PCNOFiniteVolumeGeometry) -> str:
    return _sha256_arrays(
        geometry.nodes,
        geometry.edges,
        geometry.node_type,
        geometry.node_measures,
        geometry.node_weights,
        geometry.node_rhos,
        geometry.directed_edges,
        geometry.edge_gradient_weights,
        geometry.mesh_cell_to_graph_node,
        geometry.face_to_directed_edge,
    )


def _publish_case(
    *,
    family_root: Path,
    output_root: Path,
    manifest: Mapping[str, Any],
    audit_row: Mapping[str, Any],
    case_id: str,
    gradient_rcond: float,
    expected_geometry_digest: str | None,
) -> tuple[dict[str, Any], str]:
    case = family_case_by_id(manifest, case_id)
    expected_provenance = family_case_provenance(manifest, case_id)
    source_dir = family_root / case_id
    source_path = source_dir / "reference.npz"
    summary_path = source_dir / "summary.json"
    if not source_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError(f"source case is incomplete: {case_id}")
    source_sha256 = _sha256(source_path)
    if audit_row.get("status") != "passed":
        raise ValueError(f"source audit row did not pass: {case_id}")
    if audit_row.get("reference_artifact_sha256") != source_sha256:
        raise ValueError(f"source artifact digest differs from audit: {case_id}")

    with np.load(source_path, allow_pickle=False) as artifact:
        if artifact["schema"].item() != REFERENCE_ARTIFACT_SCHEMA:
            raise ValueError(f"unexpected source artifact schema: {case_id}")
        provenance = json.loads(artifact["family_contract_json"].item())
        if provenance != expected_provenance:
            raise ValueError(f"source artifact provenance mismatch: {case_id}")
        states = np.asarray(artifact["conservative_states"], dtype=np.float64)
        times = np.asarray(artifact["physical_times"], dtype=np.float64)
        parameters = np.asarray(
            [
                provenance["parameters"]["vortex_epsilon"],
                provenance["parameters"]["vortex_y"],
            ],
            dtype=np.float64,
        )
        geometry = _build_geometry(artifact, gradient_rcond=gradient_rcond)
        geometry_digest = _geometry_digest(geometry)
        if (
            expected_geometry_digest is not None
            and geometry_digest != expected_geometry_digest
        ):
            raise ValueError(f"PCNO geometry differs between family cases: {case_id}")
        serialization = _serialization_metrics(states)
        if not _serialization_passes(serialization):
            raise ValueError(f"float32 serialization gate failed: {case_id}")
        config = json.loads(artifact["config_json"].item())

    folder_name = _folder_name(case_id)
    final_folder = output_root / folder_name
    state_digest = _sha256_arrays(states.astype(np.float32))
    if final_folder.exists():
        metadata_path = final_folder / "metadata.json"
        if not metadata_path.is_file():
            raise FileExistsError(f"existing shard lacks metadata: {final_folder}")
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        entry = existing.get("manifest_entry")
        if (
            int(existing.get("schema_version", -1)) != SCHEMA_VERSION
            or existing.get("source_reference_sha256") != source_sha256
            or existing.get("family_provenance") != expected_provenance
            or existing.get("source_family_manifest_digest")
            != manifest["manifest_digest_sha256"]
            or float(existing.get("gradient_rcond", -1.0)) != float(gradient_rcond)
            or not isinstance(entry, Mapping)
            or entry.get("geometry_digest") != geometry_digest
            or entry.get("state_digest") != state_digest
        ):
            raise FileExistsError(f"existing shard contract differs: {case_id}")
        array_sha256 = entry.get("array_sha256")
        if not isinstance(array_sha256, Mapping) or set(array_sha256) != set(
            SHARD_ARRAY_NAMES
        ):
            raise FileExistsError(
                f"existing shard lacks exact array digests: {case_id}"
            )
        for name in SHARD_ARRAY_NAMES:
            array_path = final_folder / f"{name}.npy"
            if not array_path.is_file() or _sha256(array_path) != array_sha256[name]:
                raise FileExistsError(f"existing shard array differs: {case_id}/{name}")
        return dict(entry), geometry_digest

    temporary = Path(tempfile.mkdtemp(prefix=f".{folder_name}.tmp-", dir=output_root))
    try:
        _save_array(temporary, "states_conservative", states, np.float32)
        _save_array(temporary, "physical_times", times, np.float64)
        _save_array(temporary, "parameters", parameters, np.float32)
        _save_array(temporary, "nodes", geometry.nodes, np.float32)
        _save_array(temporary, "edges", geometry.edges, np.int64)
        _save_array(temporary, "node_type", geometry.node_type, np.int64)
        _save_array(temporary, "node_measures", geometry.node_measures, np.float32)
        _save_array(temporary, "node_weights", geometry.node_weights, np.float32)
        _save_array(temporary, "node_rhos", geometry.node_rhos, np.float32)
        _save_array(temporary, "directed_edges", geometry.directed_edges, np.int64)
        _save_array(
            temporary,
            "edge_gradient_weights",
            geometry.edge_gradient_weights,
            np.float32,
        )
        _save_array(
            temporary,
            "mesh_cell_to_graph_node",
            geometry.mesh_cell_to_graph_node,
            np.int64,
        )
        _save_array(
            temporary,
            "face_to_directed_edge",
            geometry.face_to_directed_edge,
            np.int64,
        )
        array_sha256 = {
            name: _sha256(temporary / f"{name}.npy") for name in SHARD_ARRAY_NAMES
        }
        entry = {
            "key": case_id,
            "folder": folder_name,
            "num_steps": int(states.shape[0]),
            "num_nodes": int(states.shape[1]),
            "num_edges": int(geometry.edges.shape[0]),
            "num_directed_edges": int(geometry.directed_edges.shape[0]),
            "num_elements": 0,
            "mach": float(config["shock_mach"]),
            "split": case["split"],
            "split_group_id": case["split_group_id"],
            "parameters": case["parameters"],
            "geometry_digest": geometry_digest,
            "state_digest": state_digest,
            "array_sha256": array_sha256,
            "source_reference_sha256": source_sha256,
            "weight_provenance": "validated_physical_cell_volume_normalized",
            "zero_weight_nodes": int(
                np.count_nonzero(geometry.node_weights[:, 0] == 0.0)
            ),
        }
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "source_key": case_id,
            "source_reference_relative_to_family_root": f"{case_id}/reference.npz",
            "source_reference_sha256": source_sha256,
            "source_family_manifest_digest": manifest["manifest_digest_sha256"],
            "family_provenance": expected_provenance,
            "gamma": float(config["gamma"]),
            "dt": float(times[1] - times[0]),
            "state_convention": "conservative_[rho,rho_u,rho_v,E]",
            "coordinate_convention": "FV_row_major_cell_centers_xy",
            "boundary_codes": {
                "0": "interior",
                "1": "touches_y_symmetry",
                "2": "touches_x_extrapolation",
                "3": "touches_x_extrapolation_and_y_symmetry",
            },
            "mesh_to_graph_map": "identity_FV_cell_index_to_PCNO_node_index",
            "face_to_graph_map": (
                "two directed PCNO edges per physical interior face; boundary faces map to -1"
            ),
            "gradient_rcond": float(gradient_rcond),
            "minimum_stencil_singular_value": (geometry.minimum_stencil_singular_value),
            "maximum_stencil_condition_number": (
                geometry.maximum_stencil_condition_number
            ),
            "maximum_coordinate_gradient_error": (
                geometry.maximum_coordinate_gradient_error
            ),
            "float32_serialization": serialization,
            "float32_tolerances": {
                "state_relative_l2": STATE_FLOAT32_RELATIVE_TOLERANCE,
                "residual_relative_l2": RESIDUAL_FLOAT32_RELATIVE_TOLERANCE,
            },
            "manifest_entry": entry,
        }
        (temporary / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(final_folder)
    except Exception:
        for child in temporary.glob("*"):
            child.unlink(missing_ok=True)
        temporary.rmdir()
        raise
    return entry, geometry_digest


def _write_manifest(
    *,
    output_dir: Path,
    family_manifest: Mapping[str, Any],
    gradient_rcond: float,
    entries: Sequence[Mapping[str, Any]],
) -> Path:
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            int(payload.get("schema_version", -1)) != SCHEMA_VERSION
            or payload.get("dataset") != "shock_vortex_fv_family"
            or payload.get("source_family_id") != family_manifest["family_id"]
            or payload.get("source_family_manifest_digest")
            != family_manifest["manifest_digest_sha256"]
            or float(payload.get("gradient_rcond", -1.0)) != float(gradient_rcond)
        ):
            raise ValueError("cannot merge shards with a different adapter contract")
    else:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "dataset": "shock_vortex_fv_family",
            "source_family_manifest_relative_to_family_root": "family_manifest.json",
            "source_family_id": family_manifest["family_id"],
            "source_family_manifest_digest": family_manifest["manifest_digest_sha256"],
            "gamma": float(family_manifest["fixed_physics"]["gamma"]),
            "dt": float(family_manifest["time_contract"]["saved_delta_t"]),
            "state_convention": "conservative_[rho,rho_u,rho_v,E]",
            "coordinate_convention": "FV_row_major_cell_centers_xy",
            "weight_provenance": "validated_physical_cell_volume_normalized",
            "mesh_to_graph_map": "identity_FV_cell_index_to_PCNO_node_index",
            "reference_face_impulses": "retained_in_source_reference_artifacts",
            "array_digest_contract": "sha256_of_each_published_npy_file",
            "gradient_rcond": float(gradient_rcond),
            "declared_split_counts": family_manifest["split_contract"]["split_counts"],
            "trajectories": [],
        }
    merged = {
        str(entry["key"]): dict(entry) for entry in payload.get("trajectories", [])
    }
    for entry in entries:
        merged[str(entry["key"])] = dict(entry)
    case_order = {
        str(case["case_id"]): index
        for index, case in enumerate(family_manifest["cases"])
    }
    unknown_keys = sorted(set(merged) - set(case_order))
    if unknown_keys:
        raise ValueError(
            f"shard manifest contains unknown family cases: {unknown_keys}"
        )
    ordered = sorted(merged.values(), key=lambda entry: case_order[str(entry["key"])])
    payload["trajectories"] = ordered
    payload["prepared_split_counts"] = {
        split: sum(entry["split"] == split for entry in ordered)
        for split in ("train", "validation", "test")
    }
    payload["splits"] = {
        split: [entry["key"] for entry in ordered if entry["split"] == split]
        for split in ("train", "validation", "test")
    }
    temporary = output_dir / ".manifest.json.tmp"
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, manifest_path)
    return manifest_path


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if not np.isfinite(args.gradient_rcond) or args.gradient_rcond <= 0.0:
        raise ValueError("--gradient-rcond must be positive and finite")
    family_manifest_path = args.family_root / "family_manifest.json"
    family_manifest = load_shock_vortex_family_manifest(family_manifest_path)
    source_audit, case_ids = _load_source_audit(
        args.family_root,
        scope=args.scope,
        manifest=family_manifest,
    )
    rows = _audit_rows_by_case(source_audit)
    if set(rows) != set(case_ids):
        raise ValueError("source audit rows do not exactly match the requested cases")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    geometry_digest: str | None = None
    for case_id in case_ids:
        entry, candidate_digest = _publish_case(
            family_root=args.family_root,
            output_root=args.output_dir,
            manifest=family_manifest,
            audit_row=rows[case_id],
            case_id=case_id,
            gradient_rcond=args.gradient_rcond,
            expected_geometry_digest=geometry_digest,
        )
        geometry_digest = candidate_digest
        entries.append(entry)
        print(
            f"prepared {case_id}: split={entry['split']} "
            f"nodes={entry['num_nodes']} directed_edges={entry['num_directed_edges']}"
        )
    manifest_path = _write_manifest(
        output_dir=args.output_dir,
        family_manifest=family_manifest,
        gradient_rcond=args.gradient_rcond,
        entries=entries,
    )
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "scope": args.scope,
                "prepared_trajectories": len(entries),
                "geometry_digest": geometry_digest,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
