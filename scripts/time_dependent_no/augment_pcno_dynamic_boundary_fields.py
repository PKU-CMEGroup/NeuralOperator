#!/usr/bin/env python3
"""Add bounded dynamic-FV boundary fields to verified open PCNO shards.

This adapter exists for retained shard populations whose raw family artifacts
are unavailable on the training machine.  It selects the open train and
validation entries from manifest metadata before constructing any trajectory
path.  Existing arrays are hard-linked byte-for-byte; only ``nodes.npy`` from
the selected open population is decoded to construct the two physical collar
fields.  Test-population folders and arrays are never visited.
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
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utility.time_dependent_no.pcno_boundary_fields import (  # noqa: E402
    BoundaryFieldData,
    factorized_rectangle_boundary_fields,
)

OPEN_SPLITS = ("train", "validation")
DERIVATION_SCHEMA = "pcno_open_shard_boundary_field_derivation_v1"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
SAFE_COMPONENT_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-shard-dir", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--boundary-collar-width", type=float, required=True)
    parser.add_argument("--x-min", type=float, required=True)
    parser.add_argument("--x-max", type=float, required=True)
    parser.add_argument("--y-min", type=float, required=True)
    parser.add_argument("--y-max", type=float, required=True)
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


def _selected_open_entries(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_entries = manifest.get("trajectories")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ValueError("source manifest has no trajectory entries")
    entries = [dict(entry) for entry in raw_entries]
    keys = [str(entry.get("key")) for entry in entries]
    if len(keys) != len(set(keys)):
        raise ValueError("source manifest has duplicate trajectory keys")

    declared = manifest.get("declared_split_counts")
    prepared = manifest.get("prepared_split_counts")
    splits = manifest.get("splits")
    if not all(isinstance(value, Mapping) for value in (declared, prepared, splits)):
        raise ValueError("source manifest lacks complete split metadata")
    for split in ("train", "validation", "test"):
        expected_keys = [
            str(entry["key"]) for entry in entries if entry.get("split") == split
        ]
        if list(splits.get(split, [])) != expected_keys:
            raise ValueError(f"source manifest {split} key list is inconsistent")
        if int(prepared.get(split, -1)) != len(expected_keys):
            raise ValueError(f"source manifest {split} prepared count is inconsistent")

    selected = [entry for entry in entries if entry.get("split") in OPEN_SPLITS]
    for split in OPEN_SPLITS:
        selected_count = sum(entry.get("split") == split for entry in selected)
        if selected_count != int(declared.get(split, -1)):
            raise ValueError(f"source manifest does not contain the full {split} split")
    if not selected:
        raise ValueError("source manifest contains no open trajectories")
    return selected


def _load_open_nodes(path: Path, *, expected_nodes: int) -> np.ndarray:
    nodes = np.load(path, mmap_mode="r", allow_pickle=False)
    try:
        if nodes.shape != (expected_nodes, 2):
            raise ValueError(
                f"invalid open-population node shape {nodes.shape}; "
                f"expected {(expected_nodes, 2)}"
            )
        if not np.all(np.isfinite(nodes)):
            raise ValueError("open-population nodes contain nonfinite values")
        return np.array(nodes, dtype=np.float64, copy=True)
    finally:
        mmap_handle = getattr(nodes, "_mmap", None)
        if mmap_handle is not None:
            mmap_handle.close()


def _hardlink_open_arrays(
    source_folder: Path,
    target_folder: Path,
    array_sha256: Mapping[str, Any],
) -> None:
    if "boundary_features" in array_sha256:
        raise ValueError("source shards already contain boundary features")
    if "nodes" not in array_sha256:
        raise ValueError("source shard entry does not declare nodes.npy")
    target_folder.mkdir()
    for raw_name in array_sha256:
        name = _safe_component(raw_name, label="array name")
        source_path = source_folder / f"{name}.npy"
        target_path = target_folder / f"{name}.npy"
        if not source_path.is_file():
            raise FileNotFoundError(f"missing declared open array: {source_path}")
        try:
            os.link(source_path, target_path)
        except OSError as exc:
            raise OSError(
                "open shard arrays must be hard-linked on one filesystem; "
                f"failed for {source_path}"
            ) from exc


def augment_open_dynamic_shards(
    *,
    source_shard_dir: Path,
    source_manifest_sha256: str,
    output_dir: Path,
    boundary_collar_width: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> Path:
    """Publish an atomic, open-only derived shard population."""

    expected_digest = str(source_manifest_sha256).lower()
    if not SHA256_PATTERN.fullmatch(expected_digest):
        raise ValueError("source_manifest_sha256 must be a lowercase SHA-256 digest")
    bounds = np.asarray([x_min, x_max, y_min, y_max], dtype=np.float64)
    if (
        not np.all(np.isfinite(bounds))
        or not float(x_min) < float(x_max)
        or not float(y_min) < float(y_max)
    ):
        raise ValueError("domain bounds must be finite and strictly ordered")
    width = float(boundary_collar_width)
    if not np.isfinite(width) or width <= 0.0:
        raise ValueError("boundary_collar_width must be positive and finite")

    source_root = Path(source_shard_dir).resolve(strict=True)
    target_root = Path(output_dir).resolve(strict=False)
    if target_root == source_root or source_root in target_root.parents:
        raise ValueError("output_dir must not equal or be nested in source_shard_dir")
    if target_root.exists():
        raise FileExistsError(f"output directory already exists: {target_root}")
    source_manifest_path = source_root / "manifest.json"
    if not source_manifest_path.is_file():
        raise FileNotFoundError(f"missing source manifest: {source_manifest_path}")
    actual_digest = _sha256(source_manifest_path)
    if actual_digest != expected_digest:
        raise ValueError(
            "source manifest digest mismatch: "
            f"expected {expected_digest}, observed {actual_digest}"
        )
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("boundary_field_contract") is not None:
        raise ValueError("source manifest must be the boundary-field-free control")
    selected = _selected_open_entries(source_manifest)

    target_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(prefix=f".{target_root.name}.tmp-", dir=target_root.parent)
    )
    entries: list[dict[str, Any]] = []
    common_contract: dict[str, Any] | None = None
    try:
        for source_entry in selected:
            folder_name = _safe_component(source_entry.get("folder"), label="folder")
            source_folder = source_root / folder_name
            target_folder = temporary_root / folder_name
            raw_digests = source_entry.get("array_sha256")
            if not isinstance(raw_digests, Mapping) or not raw_digests:
                raise ValueError(
                    f"trajectory {source_entry.get('key')} lacks array digests"
                )
            metadata_path = source_folder / "metadata.json"
            if not metadata_path.is_file():
                raise FileNotFoundError(f"missing open metadata: {metadata_path}")
            source_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if source_metadata.get("manifest_entry") != source_entry:
                raise ValueError(
                    f"source metadata entry differs for {source_entry.get('key')}"
                )

            _hardlink_open_arrays(source_folder, target_folder, raw_digests)
            nodes = _load_open_nodes(
                source_folder / "nodes.npy",
                expected_nodes=int(source_entry["num_nodes"]),
            )
            fields: BoundaryFieldData = factorized_rectangle_boundary_fields(
                nodes,
                x_min=float(x_min),
                x_max=float(x_max),
                y_min=float(y_min),
                y_max=float(y_max),
                physical_width=width,
            )
            if common_contract is None:
                common_contract = fields.contract
            elif fields.contract != common_contract:
                raise ValueError(
                    "boundary-field contracts differ across open trajectories"
                )
            boundary_path = target_folder / "boundary_features.npy"
            np.save(boundary_path, np.asarray(fields.values, dtype=np.float32))

            entry = deepcopy(source_entry)
            entry["array_sha256"] = dict(raw_digests)
            entry["array_sha256"]["boundary_features"] = _sha256(boundary_path)
            entry["boundary_features_digest"] = _sha256_arrays(fields.values)
            entries.append(entry)

            metadata = deepcopy(source_metadata)
            metadata["boundary_field_contract"] = fields.contract
            metadata["derived_from_shard_manifest_sha256"] = actual_digest
            metadata["source_array_reuse"] = "hardlink_preserving_npy_bytes"
            metadata["manifest_entry"] = entry
            (target_folder / "metadata.json").write_text(
                json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )

        if common_contract is None:
            raise RuntimeError("no boundary-field contract was generated")
        target_manifest = deepcopy(source_manifest)
        target_manifest["boundary_field_contract"] = common_contract
        target_manifest["requested_splits"] = list(OPEN_SPLITS)
        target_manifest["trajectories"] = entries
        target_manifest["prepared_split_counts"] = {
            split: sum(entry["split"] == split for entry in entries)
            for split in ("train", "validation", "test")
        }
        target_manifest["splits"] = {
            split: [entry["key"] for entry in entries if entry["split"] == split]
            for split in ("train", "validation", "test")
        }
        target_manifest["derived_shard_provenance"] = {
            "schema": DERIVATION_SCHEMA,
            "generator_sha256": _sha256(Path(__file__)),
            "source_manifest_sha256": actual_digest,
            "selected_splits": list(OPEN_SPLITS),
            "sealed_test_arrays_opened": False,
            "source_array_reuse": "hardlink_preserving_npy_bytes",
            "decoded_source_arrays": ["nodes"],
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
    manifest_path = augment_open_dynamic_shards(
        source_shard_dir=args.source_shard_dir,
        source_manifest_sha256=args.source_manifest_sha256,
        output_dir=args.output_dir,
        boundary_collar_width=args.boundary_collar_width,
        x_min=args.x_min,
        x_max=args.x_max,
        y_min=args.y_min,
        y_max=args.y_max,
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
