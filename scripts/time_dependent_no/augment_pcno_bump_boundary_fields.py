#!/usr/bin/env python3
"""Add frozen-width semantic collars to the open bump shard population.

The collar width must come from the separately generated training-only geometry
audit.  Once that width is frozen, this adapter decodes only geometry arrays
for the 270 train and 30 open-validation graphs.  Existing arrays, including
states, are hard-linked and SHA-256 bound without decoding; only
``boundary_features.npy`` is newly generated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Sequence

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
    BoundaryFieldData,
    recover_tagged_boundary_cycle,
    tagged_polyline_boundary_fields,
)

DERIVATION_SCHEMA = "pcno_bump_open_boundary_field_derivation_v1"
GEOMETRY_AUDIT_SCHEMA = "pcno_bump_training_geometry_audit_v1"
OPEN_SPLITS = ("train", "validation")
ARRAY_NAMES = (
    "states_conservative",
    "nodes",
    "edges",
    "elements",
    "node_type",
    "node_measures",
    "node_weights",
    "node_rhos",
    "directed_edges",
    "edge_gradient_weights",
)
SAFE_COMPONENT_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-shard-dir", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument("--split-sha256", required=True)
    parser.add_argument("--geometry-audit-json", type=Path, required=True)
    parser.add_argument("--geometry-audit-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
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


def _safe_component(value: object, *, label: str) -> str:
    component = str(value)
    if (
        not component
        or component in {".", ".."}
        or not SAFE_COMPONENT_PATTERN.fullmatch(component)
    ):
        raise ValueError(f"unsafe {label}: {component!r}")
    return component


def _load_geometry(folder: Path, *, num_nodes: int) -> tuple[np.ndarray, ...]:
    arrays = [
        np.load(folder / f"{name}.npy", mmap_mode="r", allow_pickle=False)
        for name in ("nodes", "edges", "node_type")
    ]
    try:
        nodes = np.array(arrays[0], dtype=np.float64, copy=True)
        edges = np.array(arrays[1], dtype=np.int64, copy=True)
        node_type = np.array(arrays[2], dtype=np.int64, copy=True).reshape(-1)
    finally:
        for array in arrays:
            mmap_handle = getattr(array, "_mmap", None)
            if mmap_handle is not None:
                mmap_handle.close()
    if nodes.shape != (num_nodes, 2):
        raise ValueError(f"invalid bump node shape {nodes.shape}")
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError(f"invalid bump edge shape {edges.shape}")
    if node_type.shape != (num_nodes,) or not np.isin(node_type, np.arange(4)).all():
        raise ValueError("invalid bump node-type array")
    return nodes, edges, node_type


def augment_open_bump_shards(
    *,
    source_shard_dir: Path,
    source_manifest_sha256: str,
    split_json: Path,
    split_sha256: str,
    geometry_audit_json: Path,
    geometry_audit_sha256: str,
    output_dir: Path,
) -> Path:
    """Publish one atomic, manifest-split bump population with collar fields."""

    source_root = Path(source_shard_dir).resolve(strict=True)
    source_manifest_path = source_root / "manifest.json"
    split_path = Path(split_json).resolve(strict=True)
    audit_path = Path(geometry_audit_json).resolve(strict=True)
    target_root = Path(output_dir).resolve(strict=False)
    if target_root == source_root or source_root in target_root.parents:
        raise ValueError("output_dir must not equal or be nested in source_shard_dir")
    if target_root.exists():
        raise FileExistsError(f"output directory already exists: {target_root}")

    observed_source_digest = _sha256(source_manifest_path)
    observed_split_digest = _sha256(split_path)
    observed_audit_digest = _sha256(audit_path)
    if observed_source_digest != str(source_manifest_sha256).lower():
        raise ValueError("source shard manifest digest mismatch")
    if observed_split_digest != str(split_sha256).lower():
        raise ValueError("split manifest digest mismatch")
    if observed_audit_digest != str(geometry_audit_sha256).lower():
        raise ValueError("geometry audit digest mismatch")

    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    split = json.loads(split_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if source_manifest.get("boundary_field_contract") is not None:
        raise ValueError("source manifest must be the boundary-field-free control")
    if split.get("data_manifest_digest") != observed_source_digest:
        raise ValueError("split is not bound to the source shard manifest")
    if (
        audit.get("schema") != GEOMETRY_AUDIT_SCHEMA
        or audit.get("source_shard_manifest_sha256") != observed_source_digest
        or audit.get("split_manifest_sha256") != observed_split_digest
        or audit.get("state_or_target_arrays_opened") is not False
        or audit.get("validation_geometry_opened") is not False
    ):
        raise ValueError("geometry audit does not satisfy the frozen provenance gate")

    train_keys = [str(value) for value in split.get("train_keys", [])]
    validation_keys = [str(value) for value in split.get("val_keys", [])]
    if (
        not train_keys
        or not validation_keys
        or len(train_keys) != len(set(train_keys))
        or len(validation_keys) != len(set(validation_keys))
        or set(train_keys) & set(validation_keys)
        or split.get("test_keys") not in (None, [])
        or audit.get("training_keys") != train_keys
    ):
        raise ValueError("split and training-only geometry audit disagree")
    split_by_key = {
        **{key: "train" for key in train_keys},
        **{key: "validation" for key in validation_keys},
    }
    raw_entries = source_manifest.get("trajectories", [])
    entries_by_key = {str(entry["key"]): dict(entry) for entry in raw_entries}
    if len(entries_by_key) != len(raw_entries) or set(entries_by_key) != set(
        split_by_key
    ):
        raise ValueError("source trajectory population differs from the frozen split")
    width = float(audit["width_rule"]["primary_physical_width"])
    if not np.isfinite(width) or width <= 0.0:
        raise ValueError("geometry audit has an invalid primary physical width")

    target_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(prefix=f".{target_root.name}.tmp-", dir=target_root.parent)
    )
    derived_entries: list[dict[str, Any]] = []
    common_contract: dict[str, Any] | None = None
    try:
        for source_entry in raw_entries:
            key = str(source_entry["key"])
            folder_name = _safe_component(source_entry.get("folder"), label="folder")
            source_folder = source_root / folder_name
            target_folder = temporary_root / folder_name
            metadata_path = source_folder / "metadata.json"
            if not metadata_path.is_file():
                raise FileNotFoundError(f"missing source metadata: {metadata_path}")
            source_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if source_metadata.get("manifest_entry") != source_entry:
                raise ValueError(f"source metadata entry differs for {key}")
            target_folder.mkdir()
            array_sha256: dict[str, str] = {}
            for name in ARRAY_NAMES:
                source_path = source_folder / f"{name}.npy"
                target_path = target_folder / f"{name}.npy"
                if not source_path.is_file():
                    raise FileNotFoundError(f"missing source array: {source_path}")
                array_sha256[name] = _sha256(source_path)
                try:
                    os.link(source_path, target_path)
                except OSError as exc:
                    raise OSError(
                        "bump shard arrays must be hard-linked on one filesystem; "
                        f"failed for {source_path}"
                    ) from exc

            nodes, edges, node_type = _load_geometry(
                source_folder, num_nodes=int(source_entry["num_nodes"])
            )
            boundary_edges = recover_tagged_boundary_cycle(edges, node_type)
            fields: BoundaryFieldData = tagged_polyline_boundary_fields(
                nodes,
                boundary_edges,
                node_type,
                semantic_codes={
                    "wall": WALL_NODE,
                    "outflow": OUTFLOW_NODE,
                    "inflow": INFLOW_NODE,
                },
                physical_width=width,
            )
            if common_contract is None:
                common_contract = fields.contract
            elif fields.contract != common_contract:
                raise ValueError("boundary-field contracts differ across bump graphs")
            boundary_path = target_folder / "boundary_features.npy"
            np.save(boundary_path, np.asarray(fields.values, dtype=np.float32))
            array_sha256["boundary_features"] = _sha256(boundary_path)

            entry = deepcopy(source_entry)
            entry["split"] = split_by_key[key]
            entry["array_sha256"] = array_sha256
            entry["boundary_features_digest"] = _sha256_arrays(fields.values)
            derived_entries.append(entry)
            metadata = deepcopy(source_metadata)
            metadata["boundary_field_contract"] = fields.contract
            metadata["derived_from_shard_manifest_sha256"] = observed_source_digest
            metadata["split_manifest_sha256"] = observed_split_digest
            metadata["geometry_audit_sha256"] = observed_audit_digest
            metadata["source_array_reuse"] = "hardlink_preserving_npy_bytes"
            metadata["manifest_entry"] = entry
            (target_folder / "metadata.json").write_text(
                json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )

        if common_contract is None:
            raise RuntimeError("no bump boundary-field contract was generated")
        target_manifest = deepcopy(source_manifest)
        target_manifest["boundary_field_contract"] = common_contract
        target_manifest["array_digest_contract"] = "sha256_of_each_published_npy_file"
        target_manifest["requested_splits"] = list(OPEN_SPLITS)
        target_manifest["declared_split_counts"] = {
            "train": len(train_keys),
            "validation": len(validation_keys),
            "test": 0,
        }
        target_manifest["prepared_split_counts"] = deepcopy(
            target_manifest["declared_split_counts"]
        )
        target_manifest["splits"] = {
            "train": train_keys,
            "validation": validation_keys,
            "test": [],
        }
        target_manifest["trajectories"] = derived_entries
        target_manifest["derived_shard_provenance"] = {
            "schema": DERIVATION_SCHEMA,
            "generator_sha256": _sha256(Path(__file__)),
            "source_manifest_sha256": observed_source_digest,
            "split_manifest_sha256": observed_split_digest,
            "geometry_audit_sha256": observed_audit_digest,
            "primary_physical_width": width,
            "selected_splits": list(OPEN_SPLITS),
            "source_array_reuse": "hardlink_preserving_npy_bytes",
            "state_arrays_decoded_during_derivation": False,
            "state_files_sha256_bound": True,
            "decoded_source_arrays": ["nodes", "edges", "node_type"],
            "generated_arrays": ["boundary_features"],
        }
        (temporary_root / "manifest.json").write_text(
            json.dumps(
                target_manifest,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_root, target_root)
    except Exception:
        if temporary_root.exists():
            shutil.rmtree(temporary_root)
        raise
    return target_root / "manifest.json"


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    manifest_path = augment_open_bump_shards(
        source_shard_dir=args.source_shard_dir,
        source_manifest_sha256=args.source_manifest_sha256,
        split_json=args.split_json,
        split_sha256=args.split_sha256,
        geometry_audit_json=args.geometry_audit_json,
        geometry_audit_sha256=args.geometry_audit_sha256,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "manifest_sha256": _sha256(manifest_path),
                "requested_splits": list(OPEN_SPLITS),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
