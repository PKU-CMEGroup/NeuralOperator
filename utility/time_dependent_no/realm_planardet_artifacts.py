"""Source and checkpoint provenance helpers for REALM PlanarDet PCNO runs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import torch

from utility.time_dependent_no.realm_benchmark import canonical_json_sha256
from utility.time_dependent_no.realm_planardet import sha256_file

SOURCE_MANIFEST_SCHEMA = "realm_planardet_pcno_source_snapshot_v1"

EXECUTABLE_ENTRYPOINTS = (
    "scripts/time_dependent_no/smoke_realm_planardet_pcno.py",
    "scripts/time_dependent_no/train_realm_planardet_pcno.py",
    "scripts/time_dependent_no/evaluate_realm_planardet_pcno.py",
)

BASE_EXECUTABLE_SOURCES = (
    "pcno/__init__.py",
    "pcno/geo_utility.py",
    "pcno/pcno.py",
    "utility/__init__.py",
    "utility/adam.py",
    "utility/losses.py",
    "utility/normalizer.py",
    "utility/time_dependent_no/__init__.py",
    "utility/time_dependent_no/realm_benchmark.py",
    "utility/time_dependent_no/realm_pcno.py",
    "utility/time_dependent_no/realm_planardet.py",
    "utility/time_dependent_no/realm_planardet_artifacts.py",
    "utility/time_dependent_no/realm_planardet_metrics.py",
    "utility/time_dependent_no/realm_planardet_runtime.py",
)


def _canonical_relative_path(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("source path must be a nonempty string")
    parsed = PurePosixPath(value)
    if (
        "\\" in value
        or parsed.is_absolute()
        or parsed.as_posix() != value
        or any(part in {"", ".", ".."} for part in parsed.parts)
    ):
        raise ValueError("source path must be canonical relative POSIX")
    return value


def build_source_manifest(
    repo_root: Path,
    *,
    entrypoints: Sequence[str],
) -> dict[str, Any]:
    """Hash the fixed transitive core plus the exact scientific entry points."""

    paths = tuple(
        sorted(
            {
                *BASE_EXECUTABLE_SOURCES,
                *(_canonical_relative_path(path) for path in entrypoints),
            }
        )
    )
    rows: list[dict[str, Any]] = []
    for relative in paths:
        path = repo_root.joinpath(*PurePosixPath(relative).parts)
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"missing or non-regular executable source: {relative}")
        rows.append(
            {
                "path": relative,
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    payload: dict[str, Any] = {
        "schema": SOURCE_MANIFEST_SCHEMA,
        "files": rows,
    }
    payload["canonical_payload_sha256"] = canonical_json_sha256(payload)
    return payload


def validate_source_manifest(
    payload: Mapping[str, Any],
    repo_root: Path,
    *,
    entrypoints: Sequence[str],
) -> dict[str, Any]:
    expected = build_source_manifest(repo_root, entrypoints=entrypoints)
    if payload != expected:
        raise ValueError("current executable sources differ from the frozen manifest")
    return expected


def recursive_to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: recursive_to_cpu(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(recursive_to_cpu(item) for item in value)
    if isinstance(value, list):
        return [recursive_to_cpu(item) for item in value]
    return value


def structured_state_sha256(value: Any) -> str:
    """Hash a nested state without depending on pickle serialization bytes."""

    digest = hashlib.sha256()

    def update(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            descriptor = json.dumps(
                {"dtype": str(tensor.dtype), "shape": list(tensor.shape)},
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
            digest.update(b"tensor")
            digest.update(len(descriptor).to_bytes(8, byteorder="big"))
            digest.update(descriptor)
            digest.update(len(raw).to_bytes(8, byteorder="big"))
            digest.update(raw)
            return
        if isinstance(item, Mapping):
            if not all(isinstance(key, str) for key in item):
                raise TypeError("structured state mapping keys must be strings")
            digest.update(b"mapping")
            digest.update(len(item).to_bytes(8, byteorder="big"))
            for key in sorted(item):
                encoded_key = key.encode("utf-8")
                digest.update(len(encoded_key).to_bytes(8, byteorder="big"))
                digest.update(encoded_key)
                update(item[key])
            return
        if isinstance(item, (tuple, list)):
            digest.update(b"tuple" if isinstance(item, tuple) else b"list")
            digest.update(len(item).to_bytes(8, byteorder="big"))
            for child in item:
                update(child)
            return
        if item is None or isinstance(item, (bool, int, float, str)):
            encoded = json.dumps(
                {"type": type(item).__name__, "value": item},
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            digest.update(b"scalar")
            digest.update(len(encoded).to_bytes(8, byteorder="big"))
            digest.update(encoded)
            return
        raise TypeError(f"unsupported structured state value: {type(item).__name__}")

    update(value)
    return digest.hexdigest()


__all__ = [
    "BASE_EXECUTABLE_SOURCES",
    "EXECUTABLE_ENTRYPOINTS",
    "SOURCE_MANIFEST_SCHEMA",
    "build_source_manifest",
    "recursive_to_cpu",
    "structured_state_sha256",
    "validate_source_manifest",
]
