#!/usr/bin/env python3
"""Audit bump training geometry and freeze its physical collar width.

Only geometry datasets for the split-declared training keys are decoded from
the source HDF5.  Conservative states, targets, and validation geometry are
not opened.  The audit separately binds the source preprocessing
representation and the serialized PCNO shard representation because their
legacy dtypes differ even though their coordinate and connectivity values
agree exactly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utility.time_dependent_no.cpg_mesh_contract import (  # noqa: E402
    INFLOW_NODE,
    OUTFLOW_NODE,
    WALL_NODE,
)
from utility.time_dependent_no.pcno_boundary_fields import (  # noqa: E402
    recover_tagged_boundary_cycle,
)

AUDIT_SCHEMA = "pcno_bump_training_geometry_audit_v1"
WIDTH_FRACTION = 0.05
SEMANTIC_CODES = {
    "wall": WALL_NODE,
    "outflow": OUTFLOW_NODE,
    "inflow": INFLOW_NODE,
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-h5", type=Path, required=True)
    parser.add_argument("--source-shard-dir", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument("--split-sha256", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
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


def _static_geometry_frame(
    dataset: h5py.Dataset,
    *,
    num_steps: int,
    name: str,
) -> np.ndarray:
    if dataset.ndim >= 2 and dataset.shape[0] == num_steps:
        first = np.asarray(dataset[0])
        for index in range(1, num_steps):
            if not np.array_equal(np.asarray(dataset[index]), first):
                raise ValueError(f"training geometry {name} changes at frame {index}")
        return first
    return np.asarray(dataset[...])


def _flatten_node_scalar(
    value: np.ndarray,
    *,
    name: str,
    num_nodes: int,
) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 2 and array.shape[-1] == 1:
        array = array[:, 0]
    if array.shape != (num_nodes,):
        raise ValueError(
            f"{name} must reduce to shape {(num_nodes,)}, got {array.shape}"
        )
    return array


def _undirected_unique_edges(
    raw_edges: np.ndarray,
    *,
    num_nodes: int,
) -> np.ndarray:
    seen: set[tuple[int, int]] = set()
    unique: list[list[int]] = []
    for raw_left, raw_right in np.asarray(raw_edges, dtype=np.int32):
        left = int(raw_left)
        right = int(raw_right)
        if left < 0 or right < 0 or left >= num_nodes or right >= num_nodes:
            raise ValueError("training connectivity references an invalid node")
        if left == right:
            continue
        edge = (left, right) if left < right else (right, left)
        if edge not in seen:
            seen.add(edge)
            unique.append([edge[0], edge[1]])
    return np.asarray(unique, dtype=np.int32)


def _semantic_segment_statistics(
    nodes: np.ndarray,
    boundary_edges: np.ndarray,
    node_type: np.ndarray,
) -> tuple[dict[str, dict[str, float | int]], int]:
    endpoint_types = node_type[boundary_edges]
    lengths = np.linalg.norm(
        nodes[boundary_edges[:, 1]] - nodes[boundary_edges[:, 0]], axis=1
    )
    statistics: dict[str, dict[str, float | int]] = {}
    for name, code in SEMANTIC_CODES.items():
        selected = np.any(endpoint_types == int(code), axis=1)
        if not np.any(selected):
            raise ValueError(f"training geometry has no {name} boundary segment")
        statistics[name] = {
            "segment_count": int(np.count_nonzero(selected)),
            "polyline_length": float(np.sum(lengths[selected], dtype=np.float64)),
            "mixed_endpoint_segment_count": int(
                np.count_nonzero(
                    selected & (endpoint_types[:, 0] != endpoint_types[:, 1])
                )
            ),
        }
    mixed_count = int(np.count_nonzero(endpoint_types[:, 0] != endpoint_types[:, 1]))
    return statistics, mixed_count


def _summary(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(np.min(array)),
        "median": float(np.median(array)),
        "maximum": float(np.max(array)),
    }


def audit_bump_training_geometry(
    *,
    source_h5: Path,
    source_shard_dir: Path,
    source_manifest_sha256: str,
    split_json: Path,
    split_sha256: str,
    output_json: Path,
) -> Path:
    """Write one atomic manifest for the split-declared training geometry."""

    source_h5 = Path(source_h5).resolve(strict=True)
    shard_root = Path(source_shard_dir).resolve(strict=True)
    source_manifest_path = shard_root / "manifest.json"
    split_path = Path(split_json).resolve(strict=True)
    output_path = Path(output_json).resolve(strict=False)
    if output_path.exists():
        raise FileExistsError(f"output audit already exists: {output_path}")
    observed_manifest_digest = _sha256(source_manifest_path)
    if observed_manifest_digest != str(source_manifest_sha256).lower():
        raise ValueError("source shard manifest digest mismatch")
    observed_split_digest = _sha256(split_path)
    if observed_split_digest != str(split_sha256).lower():
        raise ValueError("split manifest digest mismatch")

    manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if split.get("data_manifest_digest") != observed_manifest_digest:
        raise ValueError("split is not bound to the source shard manifest")
    train_keys = [str(value) for value in split.get("train_keys", [])]
    validation_keys = [str(value) for value in split.get("val_keys", [])]
    if (
        not train_keys
        or len(train_keys) != len(set(train_keys))
        or set(train_keys) & set(validation_keys)
        or split.get("test_keys") not in (None, [])
    ):
        raise ValueError("split does not define disjoint train/open-validation keys")
    entries = {
        str(entry["key"]): dict(entry) for entry in manifest.get("trajectories", [])
    }
    if len(entries) != len(manifest.get("trajectories", [])):
        raise ValueError("source manifest has duplicate trajectory keys")
    missing_entries = sorted(set(train_keys) - set(entries))
    if missing_entries:
        raise ValueError(
            f"training keys are absent from source manifest: {missing_entries}"
        )

    h5_stat = source_h5.stat()
    if int(manifest.get("source_size_bytes", -1)) != int(h5_stat.st_size):
        raise ValueError("source HDF5 size differs from the shard manifest")
    if int(manifest.get("source_mtime_ns", -1)) != int(h5_stat.st_mtime_ns):
        raise ValueError(
            "source HDF5 modification time differs from the shard manifest"
        )

    rows: list[dict[str, Any]] = []
    with h5py.File(source_h5, "r") as handle:
        for key in train_keys:
            if key not in handle:
                raise KeyError(f"source HDF5 is missing training geometry {key}")
            entry = entries[key]
            group = handle[key]
            required = ("pos", "edges", "node_type", "Mach")
            missing = [name for name in required if name not in group]
            if missing:
                raise KeyError(f"training geometry {key} lacks datasets: {missing}")
            num_steps = int(entry["num_steps"])
            num_nodes = int(entry["num_nodes"])
            nodes = np.asarray(
                _static_geometry_frame(
                    group["pos"], num_steps=num_steps, name=f"{key}/pos"
                ),
                dtype=np.float64,
            )
            if nodes.shape != (num_nodes, 2) or not np.all(np.isfinite(nodes)):
                raise ValueError(f"invalid training coordinates for {key}")
            raw_edges = np.asarray(
                _static_geometry_frame(
                    group["edges"], num_steps=num_steps, name=f"{key}/edges"
                ),
                dtype=np.int64,
            )
            node_type = _flatten_node_scalar(
                _static_geometry_frame(
                    group["node_type"], num_steps=num_steps, name=f"{key}/node_type"
                ),
                name="node_type",
                num_nodes=num_nodes,
            ).astype(np.int64, copy=False)
            if not np.isin(node_type, np.arange(4)).all():
                raise ValueError(f"training geometry {key} has invalid node codes")
            mach_nodes = _flatten_node_scalar(
                _static_geometry_frame(
                    group["Mach"], num_steps=num_steps, name=f"{key}/Mach"
                ),
                name="Mach",
                num_nodes=num_nodes,
            ).astype(np.float64, copy=False)
            if not np.all(np.isfinite(mach_nodes)) or not np.allclose(
                mach_nodes, mach_nodes[0]
            ):
                raise ValueError(f"training geometry {key} has nonconstant Mach")
            edges = _undirected_unique_edges(raw_edges, num_nodes=num_nodes)
            source_geometry_digest = _sha256_arrays(
                nodes,
                edges,
                node_type,
                np.asarray([float(mach_nodes[0])]),
            )
            if source_geometry_digest != entry.get("geometry_digest"):
                raise ValueError(
                    f"source geometry digest mismatch for training key {key}"
                )

            folder = shard_root / str(entry["folder"])
            stored_nodes = np.load(
                folder / "nodes.npy", mmap_mode="r", allow_pickle=False
            )
            stored_edges = np.load(
                folder / "edges.npy", mmap_mode="r", allow_pickle=False
            )
            stored_type = np.load(
                folder / "node_type.npy", mmap_mode="r", allow_pickle=False
            )
            try:
                if not (
                    np.array_equal(stored_nodes, nodes.astype(np.float32))
                    and np.array_equal(stored_edges, edges.astype(np.int64))
                    and np.array_equal(stored_type, node_type)
                ):
                    raise ValueError(f"serialized shard geometry differs for {key}")
                stored_geometry_digest = _sha256_arrays(
                    stored_nodes, stored_edges, stored_type
                )
            finally:
                for array in (stored_nodes, stored_edges, stored_type):
                    mmap_handle = getattr(array, "_mmap", None)
                    if mmap_handle is not None:
                        mmap_handle.close()

            boundary_edges = recover_tagged_boundary_cycle(
                edges.astype(np.int64), node_type
            )
            semantic_statistics, mixed_count = _semantic_segment_statistics(
                nodes, boundary_edges, node_type
            )
            minima = np.min(nodes, axis=0)
            maxima = np.max(nodes, axis=0)
            spans = maxima - minima
            rows.append(
                {
                    "key": key,
                    "num_nodes": num_nodes,
                    "x_min": float(minima[0]),
                    "x_max": float(maxima[0]),
                    "y_min": float(minima[1]),
                    "y_max": float(maxima[1]),
                    "x_span": float(spans[0]),
                    "y_span": float(spans[1]),
                    "minimum_span": float(np.min(spans)),
                    "source_coordinate_digest": _sha256_arrays(nodes),
                    "source_connectivity_digest": _sha256_arrays(edges),
                    "source_node_type_digest": _sha256_arrays(node_type),
                    "source_geometry_digest": source_geometry_digest,
                    "stored_coordinate_file_sha256": _sha256(folder / "nodes.npy"),
                    "stored_connectivity_file_sha256": _sha256(folder / "edges.npy"),
                    "stored_node_type_file_sha256": _sha256(folder / "node_type.npy"),
                    "stored_geometry_digest": stored_geometry_digest,
                    "boundary_edge_set_digest": _sha256_arrays(boundary_edges),
                    "boundary_segment_count": int(boundary_edges.shape[0]),
                    "mixed_endpoint_segment_count": mixed_count,
                    "semantic_segments": semantic_statistics,
                }
            )

    minimum_spans = [float(row["minimum_span"]) for row in rows]
    median_minimum_span = float(np.median(np.asarray(minimum_spans)))
    primary_width = WIDTH_FRACTION * median_minimum_span
    payload = {
        "schema": AUDIT_SCHEMA,
        "generator": "scripts/time_dependent_no/audit_pcno_bump_training_geometry.py",
        "generator_sha256": _sha256(Path(__file__)),
        "source_h5_identity": {
            "name": source_h5.name,
            "size_bytes": int(h5_stat.st_size),
            "mtime_ns": int(h5_stat.st_mtime_ns),
            "content_digest_available": False,
        },
        "source_shard_manifest_sha256": observed_manifest_digest,
        "split_manifest_sha256": observed_split_digest,
        "split_seed": split.get("split_seed"),
        "training_keys": train_keys,
        "training_key_count": len(train_keys),
        "open_validation_key_count": len(validation_keys),
        "validation_geometry_opened": False,
        "state_or_target_arrays_opened": False,
        "geometry_static_check": "all_saved_frames_for_training_keys",
        "node_code_map": {
            "0": "normal",
            "1": "wall",
            "2": "outflow",
            "3": "inflow",
        },
        "source_digest_representation": {
            "coordinates": "HDF5 pos cast to float64",
            "connectivity": "first-occurrence canonical undirected edges as int32",
            "node_type": "flattened int64",
            "mach": "broadcast scalar as float64",
        },
        "stored_shard_representation": {
            "coordinates": "float32",
            "connectivity": "int64",
            "node_type": "int64",
            "value_match_to_source_preprocessing": "exact_after_declared_casts",
            "legacy_manifest_geometry_digest_expected_to_differ_by_dtype": True,
        },
        "boundary_cycle_source": "released_graph_non_normal_induced_cycle",
        "corner_policy": "mixed_endpoint_edges_belong_to_both_semantic_subsets",
        "width_rule": {
            "fraction": WIDTH_FRACTION,
            "population": "training_geometry_only",
            "scale_per_geometry": "min(x_span,y_span)",
            "aggregation": "median",
            "median_minimum_span": median_minimum_span,
            "primary_physical_width": primary_width,
            "units": "source_coordinate_units",
        },
        "span_summary": {
            "x_span": _summary([float(row["x_span"]) for row in rows]),
            "y_span": _summary([float(row["y_span"]) for row in rows]),
            "minimum_span": _summary(minimum_spans),
        },
        "training_geometry": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}")
    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, output_path)
    return output_path


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output = audit_bump_training_geometry(
        source_h5=args.source_h5,
        source_shard_dir=args.source_shard_dir,
        source_manifest_sha256=args.source_manifest_sha256,
        split_json=args.split_json,
        split_sha256=args.split_sha256,
        output_json=args.output_json,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "output_json": str(output),
                "output_sha256": _sha256(output),
                "training_key_count": payload["training_key_count"],
                "primary_physical_width": payload["width_rule"][
                    "primary_physical_width"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
