#!/usr/bin/env python3
"""Materialize the frozen train/development-only NACA0012 PCNO dataset."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from hashlib import sha256
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
BOUND_RESOURCE_MANIFEST = (
    REPO_ROOT / "docs/time_dependent_no/R0_SU2_NACA_RESOURCE_MANIFEST.json"
)
SOURCE_FILES = (
    "pcno/__init__.py",
    "pcno/geo_utility.py",
    "pcno/pcno.py",
    "utility/__init__.py",
    "utility/adam.py",
    "utility/losses.py",
    "utility/normalizer.py",
    "utility/time_dependent_no/__init__.py",
    "utility/time_dependent_no/pcno_naca0012.py",
    "utility/time_dependent_no/su2_naca_phase_pilot.py",
    "utility/time_dependent_no/su2_naca_trajectory.py",
    "utility/time_dependent_no/su2_restart_contract.py",
    "utility/time_dependent_no/su2_native_replay.py",
    "scripts/time_dependent_no/prepare_pcno_naca0012_dataset.py",
    "scripts/time_dependent_no/train_pcno_naca0012.py",
    "scripts/time_dependent_no/evaluate_pcno_naca0012.py",
)


def _read_source_records() -> dict[str, dict[str, Any]]:
    """Snapshot every executing repository source from one read per file."""

    records: dict[str, dict[str, Any]] = {}
    for relative in SOURCE_FILES:
        path = REPO_ROOT / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"required source file is absent or aliased: {relative}")
        payload = path.read_bytes()
        records[relative] = {
            "bytes": len(payload),
            "sha256": sha256(payload).hexdigest(),
        }
    return records


# Capture the exact local dependency bytes before importing any repository code.
_SOURCE_RECORDS_AT_IMPORT = _read_source_records()


def captured_source_records() -> dict[str, dict[str, Any]]:
    """Return a caller-owned copy of the import-time source inventory."""

    return {
        relative: dict(record) for relative, record in _SOURCE_RECORDS_AT_IMPORT.items()
    }


if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from utility.time_dependent_no.pcno_naca0012 import (
    BASELINE_CONTRACT_SCHEMA,
    BASELINE_CONTRACT_SHA256,
    build_naca_geometry,
    extract_verified_diagnostic_state,
    extract_verified_trajectory_state,
    fit_naca_normalization,
    load_naca_baseline_contract,
    validate_parent_evidence,
)
from utility.time_dependent_no.su2_restart_contract import (
    audit_unsteady_naca0012_bundle,
    sha256_file,
)

DATASET_SCHEMA = "time_dependent_no.su2_naca0012_pcno_dataset.v1"
NORMALIZATION_SCHEMA = "time_dependent_no.naca_normalization.v1"
SOURCE_MANIFEST_SCHEMA = "time_dependent_no.naca_pcno_source_manifest.v1"
FINAL_HASH_MANIFEST_SCHEMA = "time_dependent_no.naca_pcno_final_hash_manifest.v1"
OPEN_ROLES = ("train", "development")
DIAGNOSTIC_INDICES = (497, 498, 499)


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    rendered = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return sha256(rendered).hexdigest()


def _self_hashed(value: dict[str, Any]) -> dict[str, Any]:
    if "canonical_payload_sha256" in value:
        raise ValueError("canonical hash field must not be supplied by the caller")
    result = dict(value)
    result["canonical_payload_sha256"] = _canonical_sha256(result)
    return result


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _file_record(path: Path, root: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise ValueError(f"artifact is aliased: {path}")
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"artifact escapes its packet root: {path}") from error
    if not resolved.is_file():
        raise ValueError(f"artifact is absent or aliased: {path}")
    return {
        "relative_path": relative,
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _git_value(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip()


def _resource_manifest(payload: bytes) -> dict[str, Any]:
    value = json.loads(payload)
    if (
        not isinstance(value, dict)
        or value.get("schema") != "time_dependent_no.su2_naca0012_resources.v1"
    ):
        raise ValueError("unsupported NACA resource manifest")
    return value


def _verified_resource_manifest_bytes(path: Path, expected_sha256: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError("resource manifest is absent or aliased")
    payload = path.read_bytes()
    if sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("resource manifest differs from verified parent evidence")
    _resource_manifest(payload)
    return payload


def _exact_role_indices(
    contract: Mapping[str, Any], role: str
) -> tuple[np.ndarray, np.ndarray]:
    record = contract["phase_population"]["roles"][role]
    first, last = record["owned_frame_indices_inclusive"]
    center_first, center_last = record["dense_transition_center_indices_inclusive"]
    frames = np.arange(first, last + 1, dtype=np.int64)
    centers = np.arange(center_first, center_last + 1, dtype=np.int64)
    if frames.size != 240 or centers.size != 238:
        raise ValueError(f"{role} population differs from the frozen cardinality")
    return frames, centers


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--trajectory-dir", type=Path, required=True)
    parser.add_argument("--resource-manifest", type=Path, required=True)
    parser.add_argument("--resource-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def materialize(arguments: argparse.Namespace) -> dict[str, Any]:
    source_before = captured_source_records()
    if _read_source_records() != source_before:
        raise RuntimeError("source files changed after import-time binding")
    if BASELINE_CONTRACT_SHA256 != (
        "94069e65ef520d31860735b6c17f2b7515da1c4acd34d68518a2a16b3181fa87"
    ):
        raise ValueError("utility is not bound to the final frozen contract")
    if arguments.contract.is_symlink():
        raise ValueError("baseline contract is aliased")
    contract_path = arguments.contract.resolve()
    contract = load_naca_baseline_contract(contract_path)
    if contract.get("schema") != BASELINE_CONTRACT_SCHEMA:
        raise ValueError("baseline contract schema differs")
    if arguments.resource_dir.is_symlink():
        raise ValueError("Stage-0 resource directory is aliased")
    if arguments.trajectory_dir.is_symlink():
        raise ValueError("trajectory directory is aliased")
    context = validate_parent_evidence(
        contract,
        repository_root=REPO_ROOT,
        resource_root=arguments.resource_dir,
    )
    trajectory_root = context.trajectory_root.resolve()
    if arguments.trajectory_dir.resolve() != trajectory_root:
        raise ValueError("trajectory directory differs from frozen parent evidence")

    if arguments.resource_manifest.is_symlink():
        raise ValueError("resource manifest is aliased")
    resource_manifest_path = arguments.resource_manifest.resolve()
    context_resource_manifest = (
        context.repository_root
        / "docs/time_dependent_no/R0_SU2_NACA_RESOURCE_MANIFEST.json"
    )
    if (
        resource_manifest_path != BOUND_RESOURCE_MANIFEST.resolve()
        or resource_manifest_path != context_resource_manifest.resolve()
    ):
        raise ValueError(
            "resource manifest is not the repository-bound Stage-0 manifest"
        )
    resource_manifest_sha256 = context.resource_manifest_sha256
    resource_manifest_bytes = _verified_resource_manifest_bytes(
        resource_manifest_path, resource_manifest_sha256
    )
    resource_root = arguments.resource_dir.resolve()
    resource_audit = audit_unsteady_naca0012_bundle(
        resource_dir=resource_root,
        manifest_path=resource_manifest_path,
    )
    if resource_audit.get("status") != "stage0_valid":
        raise ValueError("canonical resource bundle did not pass Stage-0 audit")
    if (
        _verified_resource_manifest_bytes(
            resource_manifest_path, resource_manifest_sha256
        )
        != resource_manifest_bytes
    ):
        raise RuntimeError("resource manifest changed during Stage-0 validation")

    if arguments.output_dir.is_symlink() or arguments.output_dir.parent.is_symlink():
        raise ValueError("dataset output or its parent is aliased")
    output = arguments.output_dir.resolve()
    staging = output.with_name(f".{output.name}.staging")
    if output.exists():
        raise FileExistsError(f"dataset output already exists: {output}")
    if staging.exists():
        raise FileExistsError(f"stale dataset staging directory exists: {staging}")
    output.parent.mkdir(parents=True, exist_ok=True)

    staging.mkdir()
    try:
        geometry = build_naca_geometry(context)
        role_arrays: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for role in OPEN_ROLES:
            frame_indices, centers = _exact_role_indices(contract, role)
            states = np.stack(
                [
                    extract_verified_trajectory_state(
                        context,
                        role,
                        int(index),
                        expected_coordinates=geometry.native_coordinates,
                    )
                    for index in frame_indices
                ],
                axis=0,
            )
            if states.dtype != np.float64 or states.shape != (
                240,
                geometry.num_nodes,
                5,
            ):
                raise ValueError(f"{role} state tensor differs from frozen storage")
            role_arrays[role] = (states, frame_indices, centers)

        train_states, train_indices, _ = role_arrays["train"]
        train_positions = {
            int(index): position for position, index in enumerate(train_indices)
        }
        normalization = fit_naca_normalization(
            lambda index: train_states[train_positions[index]], contract
        )

        diagnostic_states = []
        for index in DIAGNOSTIC_INDICES:
            diagnostic_states.append(
                extract_verified_diagnostic_state(
                    context,
                    index,
                    expected_coordinates=geometry.native_coordinates,
                )
            )
        diagnostic_array = np.stack(diagnostic_states, axis=0)

        np.savez_compressed(staging / "geometry.npz", **geometry.to_mapping())
        for role in OPEN_ROLES:
            states, frame_indices, _ = role_arrays[role]
            np.save(staging / f"{role}_states.npy", states, allow_pickle=False)
            np.save(
                staging / f"{role}_frame_indices.npy",
                frame_indices,
                allow_pickle=False,
            )
        np.save(
            staging / "replay_margin_states.npy",
            diagnostic_array,
            allow_pickle=False,
        )
        np.save(
            staging / "replay_margin_frame_indices.npy",
            np.asarray(DIAGNOSTIC_INDICES, dtype=np.int64),
            allow_pickle=False,
        )

        normalization_payload = normalization.to_mapping()
        if normalization_payload.get("schema") != NORMALIZATION_SCHEMA:
            raise ValueError("normalization serializer returned an unknown schema")
        normalization_payload.update(
            {
                "contract_sha256": BASELINE_CONTRACT_SHA256,
                "fit_frames": contract["normalization"]["fit_frames"],
                "fit_transitions": contract["normalization"]["fit_transitions"],
            }
        )
        normalization_payload = _self_hashed(normalization_payload)
        _write_json(staging / "normalization.json", normalization_payload)

        source_after = _read_source_records()
        if source_after != source_before:
            raise RuntimeError("source files changed during dataset materialization")
        status = _git_value("status", "--short")
        source_payload = _self_hashed(
            {
                "schema": SOURCE_MANIFEST_SCHEMA,
                "contract_sha256": BASELINE_CONTRACT_SHA256,
                "git": {
                    "commit": _git_value("rev-parse", "HEAD"),
                    "branch": _git_value("branch", "--show-current"),
                    "dirty": bool(status),
                },
                "files": source_before,
                "source_set_sha256": _canonical_sha256(source_before),
            }
        )
        _write_json(staging / "source_manifest.json", source_payload)

        role_manifest: dict[str, Any] = {}
        for role in OPEN_ROLES:
            _, frame_indices, centers = role_arrays[role]
            role_manifest[role] = {
                "states": _file_record(staging / f"{role}_states.npy", staging),
                "frame_indices": _file_record(
                    staging / f"{role}_frame_indices.npy", staging
                ),
                "owned_frame_indices_inclusive": [
                    int(frame_indices[0]),
                    int(frame_indices[-1]),
                ],
                "frame_count": int(frame_indices.size),
                "dense_transition_center_indices_inclusive": [
                    int(centers[0]),
                    int(centers[-1]),
                ],
                "dense_transition_count": int(centers.size),
            }

        geometry_record = _file_record(staging / "geometry.npz", staging)
        normalization_record = _file_record(staging / "normalization.json", staging)
        source_record = _file_record(staging / "source_manifest.json", staging)
        coordinates = geometry.native_coordinates
        parent_evidence = contract["parent_evidence"]
        dataset_payload = _self_hashed(
            {
                "schema": DATASET_SCHEMA,
                "status": "complete",
                "contract": {
                    "file": contract_path.name,
                    "schema": BASELINE_CONTRACT_SCHEMA,
                    "sha256": BASELINE_CONTRACT_SHA256,
                },
                "parent_evidence": {
                    "trajectory_receipt_payload_sha256": parent_evidence["trajectory"][
                        "trajectory_receipt_payload_sha256"
                    ],
                    "storage_manifest_payload_sha256": parent_evidence["trajectory"][
                        "storage_manifest_payload_sha256"
                    ],
                    "ordered_restart_records_sha256": parent_evidence["trajectory"][
                        "ordered_restart_records_sha256"
                    ],
                    "resource_manifest_sha256": resource_manifest_sha256,
                    "resource_audit_status": resource_audit["status"],
                },
                "access": {
                    "materialized_population_roles": list(OPEN_ROLES),
                    "diagnostic_sets": ["native_replay_margin"],
                    "prospective_opened": False,
                    "sealed_opened": False,
                },
                "geometry": {
                    "file": geometry_record,
                    "num_nodes": geometry.num_nodes,
                    "num_elements": int(geometry.elements.shape[0]),
                    "num_directed_edges": int(geometry.directed_edges.shape[0]),
                    "coordinate_extrema": {
                        "minimum": np.min(coordinates, axis=0).tolist(),
                        "maximum": np.max(coordinates, axis=0).tolist(),
                    },
                    "coordinate_extents": geometry.fourier_lengths.tolist(),
                    "fourier_lengths": geometry.fourier_lengths.tolist(),
                },
                "normalization": {
                    "file": normalization_record,
                    "canonical_payload_sha256": normalization_payload[
                        "canonical_payload_sha256"
                    ],
                    "fit_frames": contract["normalization"]["fit_frames"],
                    "fit_transitions": contract["normalization"]["fit_transitions"],
                },
                "roles": role_manifest,
                "diagnostic_sets": {
                    "native_replay_margin": {
                        "states": _file_record(
                            staging / "replay_margin_states.npy", staging
                        ),
                        "frame_indices": _file_record(
                            staging / "replay_margin_frame_indices.npy", staging
                        ),
                        "indices": list(DIAGNOSTIC_INDICES),
                        "excluded_from_id_population": True,
                        "excluded_from_normalization": True,
                        "excluded_from_training": True,
                        "excluded_from_checkpoint_selection": True,
                        "excluded_from_support_estimates": True,
                    }
                },
                "source_manifest": {
                    "file": source_record,
                    "canonical_payload_sha256": source_payload[
                        "canonical_payload_sha256"
                    ],
                    "source_set_sha256": source_payload["source_set_sha256"],
                },
                "source_identity_unchanged": True,
                "claim_boundary": {
                    "train_and_development_population_materialized": True,
                    "pcno_code_audited": False,
                    "pcno_training_authorized": False,
                    "pcno_outcome_claimed": False,
                    "prospective_opened": False,
                    "sealed_opened": False,
                },
            }
        )
        _write_json(staging / "dataset_manifest.json", dataset_payload)

        final_files: dict[str, Any] = {}
        for path in sorted(staging.iterdir(), key=lambda item: item.name):
            if path.name == "final_hash_manifest.json":
                continue
            final_files[path.name] = _file_record(path, staging)
        final_payload = {
            "schema": FINAL_HASH_MANIFEST_SCHEMA,
            "contract_sha256": BASELINE_CONTRACT_SHA256,
            "files": final_files,
            "self_hash_excluded": True,
            "prospective_opened": False,
            "sealed_opened": False,
        }
        _write_json(staging / "final_hash_manifest.json", final_payload)
        if _read_source_records() != source_before:
            raise RuntimeError("source files changed before dataset finalization")
        if (
            _verified_resource_manifest_bytes(
                resource_manifest_path, resource_manifest_sha256
            )
            != resource_manifest_bytes
        ):
            raise RuntimeError("resource manifest changed before dataset finalization")
        os.replace(staging, output)
        return dataset_payload
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        result = materialize(arguments)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"NACA0012 PCNO dataset preparation failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
