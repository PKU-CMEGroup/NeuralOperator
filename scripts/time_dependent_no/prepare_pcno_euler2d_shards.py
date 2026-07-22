#!/usr/bin/env python3
"""Prepare full-resolution supersonic-bump shards for residual PCNO training.

The source HDF5 graph does not contain the PCNO quadrature and differential
operators.  This entry point reuses the collaborator-compatible planar-cell
reconstruction and PCNO geometry preprocessing, preserves HDF5 node order, and
writes one memory-mappable shard per trajectory.  The reconstructed vertex
areas are explicitly recorded as proxy quadrature, not physical control-volume
measures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time
from typing import Any, Sequence

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LEGACY_EULER_DIR = ROOT / "scripts" / "2d_Euler_eq"
for path in (ROOT, LEGACY_EULER_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pcno.geo_utility import (  # noqa: E402
    compute_edge_gradient_weights,
    compute_node_measures,
    compute_node_weights,
)
from reconstruct_elems_from_edges import (  # noqa: E402
    cells_to_elems,
    reconstruct_cells_from_edges,
    undirected_unique_edges,
)
from utility.time_dependent_no.euler2d import (  # noqa: E402
    load_cpg_primitive_sequence,
    primitive_to_conservative,
)
from utility.time_dependent_no.pcno_euler2d import SCHEMA_VERSION  # noqa: E402

REQUIRED_KEYS = ("pos", "edges", "node_type", "rho", "v1", "v2", "pres", "Mach")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-h5", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--trajectory-keys",
        nargs="*",
        default=None,
        help="Exact HDF5 group keys. By default, process all groups.",
    )
    parser.add_argument(
        "--max-trajectories",
        type=int,
        default=None,
        help="Process at most this many selected groups; intended for the sanity gate.",
    )
    parser.add_argument("--gamma", type=float, default=1.4)
    parser.add_argument("--dt", type=float, default=0.025)
    parser.add_argument(
        "--reconstruction-method", choices=("auto", "face", "cycles"), default="auto"
    )
    parser.add_argument("--min-face-nodes", type=int, default=3)
    parser.add_argument("--max-face-nodes", type=int, default=4)
    parser.add_argument("--gradient-rcond", type=float, default=1e-3)
    parser.add_argument(
        "--static-check",
        choices=("all", "endpoints"),
        default="all",
        help="Validate that repeated geometry and boundary fields are time independent.",
    )
    return parser.parse_args(argv)


def natural_key(value: str) -> tuple[int, int | str]:
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


def selected_keys(handle: h5py.File, args: argparse.Namespace) -> list[str]:
    available = sorted((str(key) for key in handle.keys()), key=natural_key)
    if args.trajectory_keys:
        requested = [str(key) for key in args.trajectory_keys]
        missing = [key for key in requested if key not in handle]
        if missing:
            raise KeyError(f"source HDF5 is missing trajectory keys: {missing}")
        keys = requested
    else:
        keys = available
    if args.max_trajectories is not None:
        if args.max_trajectories < 1:
            raise ValueError("--max-trajectories must be positive")
        keys = keys[: args.max_trajectories]
    if not keys:
        raise ValueError("no trajectories selected")
    return keys


def temporal_static_frame(
    dataset: h5py.Dataset,
    *,
    num_steps: int,
    name: str,
    static_check: str,
) -> np.ndarray:
    """Return a static frame and verify repeated time-major storage."""

    if dataset.ndim >= 2 and dataset.shape[0] == num_steps:
        first = np.asarray(dataset[0])
        indices = range(1, num_steps) if static_check == "all" else (num_steps - 1,)
        for index in indices:
            candidate = np.asarray(dataset[index])
            if not np.array_equal(candidate, first):
                raise ValueError(f"{name} changes between frames 0 and {index}")
        return first
    return np.asarray(dataset[...])


def flatten_node_scalar(value: np.ndarray, *, name: str, num_nodes: int) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 2 and array.shape[-1] == 1:
        array = array[:, 0]
    if array.shape != (num_nodes,):
        raise ValueError(
            f"{name} must reduce to shape ({num_nodes},), got {array.shape}"
        )
    return array


def sha256_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(contiguous.dtype.str.encode("ascii"))
        digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def shard_folder_name(key: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", key).strip("._") or "trajectory"
    suffix = hashlib.sha256(key.encode("utf-8")).hexdigest()[:8]
    return f"traj_{safe}_{suffix}"


def save_array(
    folder: Path, name: str, value: np.ndarray, dtype: np.dtype[Any]
) -> None:
    np.save(folder / f"{name}.npy", np.asarray(value, dtype=dtype))


def prepare_trajectory(
    group: h5py.Group,
    *,
    key: str,
    output_root: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    missing = [name for name in REQUIRED_KEYS if name not in group]
    if missing:
        raise KeyError(f"trajectory {key} is missing required datasets: {missing}")
    primitive = load_cpg_primitive_sequence(group)
    if primitive.ndim != 3 or primitive.shape[-1] != 4:
        raise ValueError(
            f"trajectory {key} has invalid primitive shape {primitive.shape}"
        )
    num_steps, num_nodes, _ = primitive.shape
    nodes = temporal_static_frame(
        group["pos"],
        num_steps=num_steps,
        name=f"{key}/pos",
        static_check=args.static_check,
    )
    raw_edges = temporal_static_frame(
        group["edges"],
        num_steps=num_steps,
        name=f"{key}/edges",
        static_check=args.static_check,
    )
    node_type = temporal_static_frame(
        group["node_type"],
        num_steps=num_steps,
        name=f"{key}/node_type",
        static_check=args.static_check,
    )
    mach_nodes = temporal_static_frame(
        group["Mach"],
        num_steps=num_steps,
        name=f"{key}/Mach",
        static_check=args.static_check,
    )
    nodes = np.asarray(nodes, dtype=np.float64)
    if nodes.shape != (num_nodes, 2):
        raise ValueError(f"trajectory {key} positions must have shape ({num_nodes}, 2)")
    raw_edges = np.asarray(raw_edges, dtype=np.int64)
    if raw_edges.ndim != 2 or raw_edges.shape[-1] != 2:
        raise ValueError(f"trajectory {key} edges must have shape [E,2]")
    node_type = flatten_node_scalar(
        node_type, name="node_type", num_nodes=num_nodes
    ).astype(
        np.int64,
        copy=False,
    )
    if not np.isin(node_type, np.arange(4)).all():
        raise ValueError(f"trajectory {key} contains unsupported node-type codes")
    mach_nodes = flatten_node_scalar(
        mach_nodes, name="Mach", num_nodes=num_nodes
    ).astype(
        np.float64,
        copy=False,
    )
    if not np.isfinite(mach_nodes).all() or not np.allclose(mach_nodes, mach_nodes[0]):
        raise ValueError(f"trajectory {key} Mach must be one finite broadcast scalar")
    mach = float(mach_nodes[0])

    edges, edge_stats = undirected_unique_edges(raw_edges, num_nodes)
    cells, reconstruction_method = reconstruct_cells_from_edges(
        edges,
        nodes,
        method=args.reconstruction_method,
        min_face_nodes=args.min_face_nodes,
        max_face_nodes=args.max_face_nodes,
    )
    elements = cells_to_elems(
        cells,
        max_nodes_per_elem=args.max_face_nodes,
        padding_value=-1,
    )
    node_measures_raw = compute_node_measures(nodes, elements)
    if node_measures_raw.shape[1] != 1:
        raise ValueError(
            f"trajectory {key} produced {node_measures_raw.shape[1]} measures; expected one"
        )
    node_measures, node_weights = compute_node_weights(
        np.asarray([num_nodes], dtype=np.int64),
        node_measures_raw[None, ...],
        equal_measure=False,
    )
    node_measures = node_measures[0]
    node_weights = node_weights[0]
    node_rhos = np.zeros_like(node_weights)
    positive_measure = node_measures > 0.0
    node_rhos[positive_measure] = (
        node_weights[positive_measure] / node_measures[positive_measure]
    )
    directed_edges, edge_gradient_weights, _ = compute_edge_gradient_weights(
        nodes,
        elements,
        mesh_type="vertex_centered",
        adjacent_type="element",
        rcond=float(args.gradient_rcond),
    )
    conservative = primitive_to_conservative(primitive, gamma=float(args.gamma)).astype(
        np.float32,
        copy=False,
    )
    if not np.isfinite(conservative).all():
        raise ValueError(f"trajectory {key} contains nonfinite conservative states")

    folder_name = shard_folder_name(key)
    final_folder = output_root / folder_name
    if final_folder.exists():
        metadata_path = final_folder / "metadata.json"
        if not metadata_path.is_file():
            raise FileExistsError(f"existing shard has no metadata: {final_folder}")
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        if existing.get("source_key") != key:
            raise FileExistsError(f"shard collision at {final_folder}")
        print(f"skip existing trajectory {key}: {final_folder}")
        return existing["manifest_entry"]

    temporary_folder = Path(
        tempfile.mkdtemp(prefix=f".{folder_name}.tmp-", dir=output_root)
    )
    save_array(temporary_folder, "states_conservative", conservative, np.float32)
    save_array(temporary_folder, "nodes", nodes, np.float32)
    save_array(temporary_folder, "edges", edges, np.int64)
    save_array(temporary_folder, "elements", elements, np.int32)
    save_array(temporary_folder, "node_type", node_type, np.int64)
    save_array(temporary_folder, "node_measures", node_measures, np.float32)
    save_array(temporary_folder, "node_weights", node_weights, np.float32)
    save_array(temporary_folder, "node_rhos", node_rhos, np.float32)
    save_array(temporary_folder, "directed_edges", directed_edges, np.int64)
    save_array(
        temporary_folder,
        "edge_gradient_weights",
        edge_gradient_weights,
        np.float32,
    )
    geometry_digest = sha256_arrays(nodes, edges, node_type, np.asarray([mach]))
    state_digest = sha256_arrays(conservative)
    entry = {
        "key": key,
        "folder": folder_name,
        "num_steps": int(num_steps),
        "num_nodes": int(num_nodes),
        "num_raw_edges": int(raw_edges.shape[0]),
        "num_edges": int(edges.shape[0]),
        "num_directed_edges": int(directed_edges.shape[0]),
        "num_elements": int(elements.shape[0]),
        "mach": mach,
        "geometry_digest": geometry_digest,
        "state_digest": state_digest,
        "reconstruction_method": reconstruction_method,
        "weight_provenance": "reconstructed_vertex_lumped_proxy",
        "zero_weight_nodes": int(np.count_nonzero(node_weights[:, 0] == 0.0)),
    }
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "source_key": key,
        "gamma": float(args.gamma),
        "dt": float(args.dt),
        "state_convention": "conservative_[rho,rho_v1,rho_v2,E]",
        "coordinate_convention": "source_HDF5_node_order_xy",
        "boundary_codes": {"0": "interior", "1": "wall", "2": "outflow", "3": "inflow"},
        "manifest_entry": entry,
        "edge_stats": edge_stats,
        "gradient_rcond": float(args.gradient_rcond),
    }
    (temporary_folder / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for attempt in range(6):
        try:
            temporary_folder.replace(final_folder)
            break
        except PermissionError:
            if attempt == 5:
                raise
            # Windows scanners can briefly retain a handle to newly written
            # shard files. Keep the atomic publish, but tolerate that transient
            # sharing violation.
            time.sleep(0.025 * (2**attempt))
    print(
        f"prepared {key}: nodes={num_nodes}, edges={edges.shape[0]}, "
        f"directed={directed_edges.shape[0]}, cells={elements.shape[0]}"
    )
    return entry


def write_manifest(
    output_dir: Path,
    *,
    source_h5: Path,
    args: argparse.Namespace,
    new_entries: Sequence[dict[str, Any]],
) -> Path:
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        existing_source = Path(str(manifest["source_h5"])).resolve()
        if existing_source != source_h5.resolve():
            raise ValueError("cannot merge shards from a different source HDF5")
    else:
        stat = source_h5.stat()
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "source_h5": str(source_h5.resolve()),
            "source_size_bytes": int(stat.st_size),
            "source_mtime_ns": int(stat.st_mtime_ns),
            "gamma": float(args.gamma),
            "dt": float(args.dt),
            "state_convention": "conservative_[rho,rho_v1,rho_v2,E]",
            "coordinate_convention": "source_HDF5_node_order_xy",
            "weight_provenance": "reconstructed_vertex_lumped_proxy",
            "trajectories": [],
        }
    entries = {str(entry["key"]): dict(entry) for entry in manifest["trajectories"]}
    for entry in new_entries:
        entries[str(entry["key"])] = dict(entry)
    manifest["trajectories"] = [
        entries[key] for key in sorted(entries, key=natural_key)
    ]
    temporary_path = output_dir / ".manifest.json.tmp"
    temporary_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, manifest_path)
    return manifest_path


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.source_h5.is_file():
        raise FileNotFoundError(args.source_h5)
    if args.gamma <= 1.0 or args.dt <= 0.0:
        raise ValueError("gamma must exceed one and dt must be positive")
    if args.min_face_nodes < 3 or args.max_face_nodes < args.min_face_nodes:
        raise ValueError("invalid face-node bounds")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    with h5py.File(args.source_h5, "r") as handle:
        keys = selected_keys(handle, args)
        print(f"selected {len(keys)} trajectories from {args.source_h5}")
        for key in keys:
            entries.append(
                prepare_trajectory(
                    handle[key],
                    key=key,
                    output_root=args.output_dir,
                    args=args,
                )
            )
    manifest_path = write_manifest(
        args.output_dir,
        source_h5=args.source_h5,
        args=args,
        new_entries=entries,
    )
    print(
        json.dumps(
            {"manifest": str(manifest_path), "trajectories": len(entries)}, indent=2
        )
    )


if __name__ == "__main__":
    main()
