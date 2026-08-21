"""Artifact I/O and provenance helpers shared by maintained PCNO entry points."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PCNO_SOURCE_SNAPSHOT_SCHEMA = "pcno_euler2d_source_snapshot_v6"
PCNO_SOURCE_SNAPSHOT_V5_SCHEMA = "pcno_euler2d_source_snapshot_v5"
PCNO_SOURCE_SNAPSHOT_V4_SCHEMA = "pcno_euler2d_source_snapshot_v4"
PCNO_SOURCE_SNAPSHOT_V3_SCHEMA = "pcno_euler2d_source_snapshot_v3"
PCNO_SOURCE_SNAPSHOT_V2_SCHEMA = "pcno_euler2d_source_snapshot_v2"
PCNO_SOURCE_PROVENANCE_FILES = (
    "docs/time_dependent_no/RESEARCH_DIRECTION_DECISION.md",
    "docs/time_dependent_no/MECHANISTIC_DIAGNOSTIC_TRACKER.md",
)
PCNO_SOURCE_SNAPSHOT_V3_FILES = (
    "scripts/time_dependent_no/train_pcno_euler2d_residual.py",
    "scripts/time_dependent_no/evaluate_pcno_euler2d_residual.py",
    "utility/time_dependent_no/pcno_artifacts.py",
    "utility/time_dependent_no/euler2d_metrics.py",
    "utility/time_dependent_no/pcno_euler2d.py",
    "utility/time_dependent_no/pcno_ripple_diagnostics.py",
    "utility/time_dependent_no/pcno_runtime.py",
    "utility/time_dependent_no/pcno_rollout.py",
    "utility/time_dependent_no/cpg_mesh_contract.py",
    "pcno/pcno.py",
)
PCNO_SOURCE_SNAPSHOT_V4_FILES = (
    *PCNO_SOURCE_SNAPSHOT_V3_FILES,
    "pcno/__init__.py",
    "pcno/geo_utility.py",
    "utility/__init__.py",
    "utility/adam.py",
    "utility/losses.py",
    "utility/normalizer.py",
    "utility/time_dependent_no/__init__.py",
    "utility/time_dependent_no/euler2d.py",
    "utility/time_dependent_no/errors.py",
)
PCNO_SOURCE_SNAPSHOT_V5_FILES = (
    *PCNO_SOURCE_SNAPSHOT_V4_FILES,
    "utility/time_dependent_no/pcno_boundary_fields.py",
)
PCNO_SOURCE_SNAPSHOT_FILES = PCNO_SOURCE_SNAPSHOT_V5_FILES
PCNO_SOURCE_SNAPSHOT_V2_FILES = (
    *PCNO_SOURCE_PROVENANCE_FILES,
    *PCNO_SOURCE_SNAPSHOT_V3_FILES,
)


def jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    result = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            result[key] = str(value)
        elif isinstance(value, tuple):
            result[key] = list(value)
        else:
            result[key] = value
    return result


def git_state(root: Path = REPOSITORY_ROOT) -> dict[str, Any]:
    def command(*arguments: str) -> str:
        try:
            return subprocess.check_output(
                ["git", *arguments],
                cwd=root,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return "unknown"

    status = command("status", "--porcelain")
    return {
        "commit": command("rev-parse", "HEAD"),
        "branch": command("branch", "--show-current"),
        "dirty": bool(status and status != "unknown"),
    }


def git_head(root: Path = REPOSITORY_ROOT) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_files(
    paths: Sequence[str | Path],
    *,
    root: Path = REPOSITORY_ROOT,
) -> dict[str, str]:
    return {str(path): sha256_file(root / path) for path in paths}


def digest_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def runtime_environment(device: torch.device) -> dict[str, Any]:
    """Record the runtime facts needed to interpret throughput and numerics."""

    cuda_device = None
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        cuda_device = {
            "name": properties.name,
            "total_memory_bytes": int(properties.total_memory),
            "capability": list(torch.cuda.get_device_capability(device)),
        }
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_device": cuda_device,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }


def _digest_mapping(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalize_extra_source_files(
    relative_names: Sequence[str | Path],
) -> tuple[str, ...]:
    if isinstance(relative_names, (str, bytes, Path)):
        raise TypeError("extra source files must be a sequence of repository paths")

    root = REPOSITORY_ROOT.resolve()
    registered = set(PCNO_SOURCE_SNAPSHOT_FILES) | set(PCNO_SOURCE_PROVENANCE_FILES)
    normalized: list[str] = []
    seen: set[str] = set()
    for value in relative_names:
        candidate = Path(value)
        if candidate.is_absolute():
            raise ValueError("extra source files must be relative to the repository")
        try:
            relative = (root / candidate).resolve().relative_to(root)
        except ValueError as error:
            raise ValueError(
                "extra source files must remain inside the repository"
            ) from error
        name = relative.as_posix()
        if name in registered:
            raise ValueError(f"extra source file is already registered: {name}")
        if name in seen:
            raise ValueError(f"duplicate extra source file: {name}")
        seen.add(name)
        normalized.append(name)
    return tuple(sorted(normalized))


def write_source_snapshot(
    output_dir: Path,
    *,
    extra_source_files: Sequence[str | Path] = (),
) -> dict[str, Any]:
    """Retain the exact registered source surface used by a new training run."""

    normalized_extras = _normalize_extra_source_files(extra_source_files)
    snapshot_dir = output_dir / "source_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=False)

    def copy_files(relative_names: Sequence[str]) -> dict[str, dict[str, Any]]:
        records = {}
        for relative_name in relative_names:
            source = REPOSITORY_ROOT / relative_name
            if not source.is_file():
                raise FileNotFoundError(source)
            destination = snapshot_dir / relative_name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            records[relative_name] = {
                "sha256": sha256_file(destination),
                "bytes": int(destination.stat().st_size),
            }
        return records

    files = copy_files((*PCNO_SOURCE_SNAPSHOT_FILES, *normalized_extras))
    provenance_files = copy_files(PCNO_SOURCE_PROVENANCE_FILES)
    payload = {
        "schema": PCNO_SOURCE_SNAPSHOT_SCHEMA,
        "git": git_state(),
        "files": files,
        "extra_source_files": list(normalized_extras),
        "provenance_files": provenance_files,
        "source_set_digest": _digest_mapping(files),
        "provenance_set_digest": _digest_mapping(provenance_files),
    }
    write_json(snapshot_dir / "manifest.json", payload)
    return payload


def verify_source_snapshot(snapshot: Mapping[str, Any]) -> None:
    """Reject continuation unless the schema-specific bound source still matches."""

    schema = snapshot.get("schema")
    supported_schemas = {
        PCNO_SOURCE_SNAPSHOT_V2_SCHEMA,
        PCNO_SOURCE_SNAPSHOT_V3_SCHEMA,
        PCNO_SOURCE_SNAPSHOT_V4_SCHEMA,
        PCNO_SOURCE_SNAPSHOT_V5_SCHEMA,
        PCNO_SOURCE_SNAPSHOT_SCHEMA,
    }
    if schema not in supported_schemas:
        expected = ", ".join(repr(value) for value in sorted(supported_schemas))
        raise ValueError(
            f"unsupported PCNO source snapshot schema: {schema!r}; expected {expected}"
        )

    if schema == PCNO_SOURCE_SNAPSHOT_V2_SCHEMA:
        expected_files = PCNO_SOURCE_SNAPSHOT_V2_FILES
    elif schema == PCNO_SOURCE_SNAPSHOT_V3_SCHEMA:
        expected_files = PCNO_SOURCE_SNAPSHOT_V3_FILES
    elif schema == PCNO_SOURCE_SNAPSHOT_V4_SCHEMA:
        expected_files = PCNO_SOURCE_SNAPSHOT_V4_FILES
    elif schema == PCNO_SOURCE_SNAPSHOT_V5_SCHEMA:
        expected_files = PCNO_SOURCE_SNAPSHOT_V5_FILES
    else:
        extra_source_files = snapshot.get("extra_source_files")
        if not isinstance(extra_source_files, Sequence) or isinstance(
            extra_source_files, (str, bytes)
        ):
            raise ValueError("v6 source snapshot lacks its extra-source registry")
        try:
            normalized_extras = _normalize_extra_source_files(extra_source_files)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "v6 source snapshot has an invalid extra-source registry"
            ) from error
        if list(extra_source_files) != list(normalized_extras):
            raise ValueError("v6 extra-source registry is not sorted and unique")
        expected_files = (*PCNO_SOURCE_SNAPSHOT_FILES, *normalized_extras)
    files = snapshot.get("files")
    if not isinstance(files, Mapping) or set(files) != set(expected_files):
        raise ValueError("source snapshot does not cover the registered source set")
    mismatches = []
    for relative_name in expected_files:
        record = files[relative_name]
        if not isinstance(record, Mapping):
            mismatches.append(relative_name)
            continue
        source = REPOSITORY_ROOT / relative_name
        if not source.is_file() or sha256_file(source) != record.get("sha256"):
            mismatches.append(relative_name)
    if mismatches:
        raise ValueError(f"current source differs from run snapshot: {mismatches}")

    if schema == PCNO_SOURCE_SNAPSHOT_V2_SCHEMA:
        if snapshot.get("source_set_digest") != _digest_mapping(files):
            raise ValueError("v2 source snapshot source-set digest mismatch")
        return

    provenance_files = snapshot.get("provenance_files")
    if not isinstance(provenance_files, Mapping) or set(provenance_files) != set(
        PCNO_SOURCE_PROVENANCE_FILES
    ):
        raise ValueError("source snapshot does not cover the provenance-only set")
    if snapshot.get("source_set_digest") != _digest_mapping(files):
        raise ValueError("source snapshot source-set digest mismatch")
    if snapshot.get("provenance_set_digest") != _digest_mapping(provenance_files):
        raise ValueError("source snapshot provenance-set digest mismatch")


def json_safe(value: Any) -> Any:
    """Match the residual evaluator's JSON conversion contract."""

    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def json_safe_with_paths(value: Any) -> Any:
    """Match the resolution evaluators' JSON conversion contract."""

    if isinstance(value, Mapping):
        return {str(key): json_safe_with_paths(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe_with_paths(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return json_safe_with_paths(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Write residual-evaluator JSON, replacing nonfinite numerics with null."""

    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_write_json_with_paths(path: Path, value: Mapping[str, Any]) -> None:
    """Write resolution-evaluator JSON, also serializing paths as strings."""

    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            json_safe_with_paths(value),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write CSV with the residual evaluator's established behavior."""

    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(str(key))
    serialized_rows = []
    for row in rows:
        serialized = {}
        for key, value in row.items():
            safe_value = json_safe(value)
            if isinstance(safe_value, (dict, list)):
                safe_value = json.dumps(
                    safe_value, sort_keys=True, separators=(",", ":")
                )
            serialized[str(key)] = safe_value
        serialized_rows.append(serialized)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(serialized_rows)


def write_csv_with_paths(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write CSV with the resolution evaluators' established behavior."""

    if not rows:
        raise ValueError(f"cannot write an empty CSV: {path}")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            key_string = str(key)
            if key_string not in fieldnames:
                fieldnames.append(key_string)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            serialized: dict[str, Any] = {}
            for key, value in row.items():
                safe = json_safe_with_paths(value)
                if isinstance(safe, (dict, list)):
                    safe = json.dumps(safe, sort_keys=True, separators=(",", ":"))
                serialized[str(key)] = safe
            writer.writerow(serialized)


def atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Write strict training JSON; nonfinite values remain an error."""

    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
