#!/usr/bin/env python3
"""Evaluate the hash-bound B3/B4 NACA corrective successor.

Development evaluation and a data-free prospective freeze are open execution
modes.  Prospective evaluation is available only behind an exact freeze and a
separate owner authorization, both verified before any scientific input path
is resolved or read.  Sealed/test access is never implemented here.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import platform
import secrets
import shutil
import sys
import tempfile
import time
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import torch

MODULE_PATH = Path(__file__)
if MODULE_PATH.is_symlink():
    raise RuntimeError("successor evaluator source is aliased")
REPO_ROOT = MODULE_PATH.resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _bootstrap_source_record(relative: str) -> dict[str, Any]:
    """Capture source bytes before importing any repository-owned module."""
    candidate = REPO_ROOT / relative
    if candidate.is_symlink():
        raise RuntimeError(f"bootstrap source is aliased: {relative}")
    resolved = candidate.resolve()
    try:
        resolved.relative_to(REPO_ROOT)
    except ValueError as error:
        raise RuntimeError(
            f"bootstrap source escapes repository: {relative}"
        ) from error
    if not resolved.is_file():
        raise RuntimeError(f"bootstrap source is absent: {relative}")
    payload = resolved.read_bytes()
    if resolved.read_bytes() != payload:
        raise RuntimeError(f"bootstrap source changed while reading: {relative}")
    return {"bytes": len(payload), "sha256": sha256(payload).hexdigest()}


SUCCESSOR_SOURCE_RECORDS_AT_IMPORT = {
    relative: _bootstrap_source_record(relative)
    for relative in (
        "scripts/time_dependent_no/evaluate_pcno_naca0012_successor.py",
        "utility/time_dependent_no/pcno_naca0012_successor.py",
    )
}

# These are immutable R0 readers and metric definitions.  Importing them does
# not reuse the R0 execution authorization: this evaluator emits and verifies a
# new successor source closure and a new dynamic input manifest.
from scripts.time_dependent_no.evaluate_pcno_naca0012 import (
    _airfoil_geometry,
    _field_metrics,
    _force_scales,
    _metric_columns,
    _structure_metrics,
    _view_weights,
)
from scripts.time_dependent_no.train_pcno_naca0012 import (
    SOURCE_RECORDS_AT_IMPORT as R0_SOURCE_RECORDS_AT_IMPORT,
)
from scripts.time_dependent_no.train_pcno_naca0012 import (
    _load_dataset,
)
from utility.time_dependent_no.pcno_naca0012 import (
    NACANormalization,
    VerifiedNACAGeometry,
    _extract_native_state,
    _load_verified_storage_manifest,
    build_naca_pcno,
    load_naca_baseline_contract,
    recurrent_step,
    validate_naca_model_config,
)
from utility.time_dependent_no.pcno_naca0012 import (
    _safe_child as _naca_safe_child,
)
from utility.time_dependent_no.pcno_naca0012_successor import (
    RecoveryCalibration,
    TrainPathProjector,
    corrected_recurrent_step,
    identity_corrector,
    validate_successor_math_contract,
)

# Bound the source read above to the files present immediately after all
# repository-owned imports.  This narrows the bootstrap/import race without
# requiring a mutable external launcher.
if {
    relative: _bootstrap_source_record(relative)
    for relative in SUCCESSOR_SOURCE_RECORDS_AT_IMPORT
} != SUCCESSOR_SOURCE_RECORDS_AT_IMPORT:
    raise RuntimeError("successor source changed during module bootstrap")

SUCCESSOR_CONTRACT_SCHEMA = "time_dependent_no.naca_corrective_successor_contract.v1"
EXPERIMENT_ID = "B3B4_NACA_CM_20260901A"
SOURCE_MANIFEST_SCHEMA = "time_dependent_no.naca_corrective_successor_source.v1"
RUNTIME_MANIFEST_SCHEMA = "time_dependent_no.naca_corrective_successor_runtime.v1"
FINAL_HASH_MANIFEST_SCHEMA = (
    "time_dependent_no.naca_corrective_successor_final_hash_manifest.v1"
)
PROSPECTIVE_FREEZE_SCHEMA = (
    "time_dependent_no.naca_corrective_successor_prospective_freeze.v1"
)
PROSPECTIVE_AUTHORIZATION_SCHEMA = (
    "time_dependent_no.naca_corrective_successor_prospective_authorization.v1"
)
PROSPECTIVE_ACTION = "evaluate_pcno_naca0012_successor_prospective"
PROSPECTIVE_AUTHORIZATION_FIELDS = frozenset(
    {
        "schema",
        "status",
        "experiment_id",
        "successor_contract_sha256",
        "prospective_freeze_final_hash_manifest_sha256",
        "prospective_freeze_payload_sha256",
        "prospective_freeze_source_set_sha256",
        "prospective_freeze_root_sha256",
        "allowed_actions",
        "authorized_population_role",
        "prospective_reveal_authorized",
        "sealed_reveal_authorized",
        "prospective_opened",
        "sealed_opened",
        "authorized_by",
        "canonical_payload_sha256",
    }
)
PROSPECTIVE_FRAMES = (1511, 1750)
PROSPECTIVE_ANCHORS = (1512, 1516, 1521, 1525, 1529, 1534, 1538, 1542)
SEEDS = (17, 29, 43)
LEARNED_ARMS = (
    "CLEAN",
    "IID_RECOVERY",
    "ERROR_SUBSPACE_RECOVERY",
    "DETACHED_PUSHFORWARD",
)
ALL_ARMS = LEARNED_ARMS + ("PATH_PROJECTION",)
FIELDS = ("Density", "Momentum_x", "Momentum_y", "Energy", "Nu_Tilde")
HORIZONS = (1, 8, 35, 104, 208)
FULL_TRACE = (1, 208)
LATE_WINDOW = (174, 208)
TANGENT_STABILITY_RMS = 1.0e-12
EXPECTED_TRAINING_FILES = {
    "best.pt",
    "config.json",
    "history.json",
    "input_manifest.json",
    "last.pt",
    "runtime_manifest.json",
    "status.json",
    "summary.json",
}
EXPECTED_CALIBRATION_FILES = {"calibration.json", "calibration_arrays.npz"}
OUTPUT_FILES = {
    "cost_metrics.csv",
    "input_manifest.json",
    "method_summary.csv",
    "offline_diagnostics.csv",
    "pairwise_summary.csv",
    "result.json",
    "rollout_metrics.csv",
    "rollout_snapshots.npz",
    "rollout_structure.csv",
    "runtime_manifest.json",
    "source_manifest.json",
}
SCIENTIFIC_OUTPUT_FILES = OUTPUT_FILES - {"runtime_manifest.json"}
FREEZE_OUTPUT_FILES = {"prospective_freeze.json", "source_manifest.json"}
PRIMARY_METRICS = (
    "normalized_state_error_auc",
    "late_window_normalized_state_error_median",
)
PRIMARY_DEPLOYMENTS = (
    ("CLEAN", "raw", "raw"),
    ("IID_RECOVERY", "raw", "raw"),
    ("ERROR_SUBSPACE_RECOVERY", "raw", "raw"),
    ("DETACHED_PUSHFORWARD", "raw", "raw"),
    ("PATH_PROJECTION", "path_projection", "corrected"),
)
MEDIATOR_METRICS = (
    "clean_forcing_normalized_rms",
    "one_prefix_next_normalized_rms",
    "one_prefix_response_gain",
    "one_prefix_response_normalized_rms",
    "one_prefix_response_transverse_rms",
    "one_prefix_response_graph_dirichlet_energy",
)
EXPECTED_RESULT_CARDINALITIES = {
    "offline_diagnostics.csv": 96,
    "rollout_metrics.csv": 34944,
    "rollout_structure.csv": 34944,
    "method_summary.csv": 168,
    "pairwise_summary.csv": 240,
    "cost_metrics.csv": 18,
    "rollout_snapshots.npz": 760,
}
MEDIATOR_ARMS = (
    "IID_RECOVERY",
    "ERROR_SUBSPACE_RECOVERY",
    "DETACHED_PUSHFORWARD",
)
FREEZE_CLAIM_BOUNDARY = {
    "prediction_matrix_is_development_informed": True,
    "automatic_C2_claim_decision_made": False,
    "path_projection_zero_path_distance_is_construction_only": True,
    "trusted_displaced_state_SU2_response_claimed": False,
    "online_solver_calls": 0,
    "prospective_opened": False,
    "sealed_opened": False,
}
FREEZE_FIELDS = {
    "schema",
    "status",
    "experiment_id",
    "successor_contract_sha256",
    "population_role",
    "prospective_frames_inclusive",
    "anchors",
    "seeds",
    "primary_deployments",
    "identity_parity_deployment",
    "intervention_magnitude_deployment",
    "primary_metrics",
    "mediator_metrics",
    "horizons",
    "full_trace_inclusive",
    "late_window_inclusive",
    "tie_ratio_inclusive",
    "aggregation",
    "sampling",
    "expected_result_cardinalities",
    "parent_bindings",
    "evaluator_source_set_sha256",
    "predictions",
    "claim_boundary",
    "prospective_opened",
    "sealed_opened",
    "canonical_payload_sha256",
}
PARENT_DIGEST_FIELDS = (
    "inherited_r0_contract_sha256",
    "preregistration_sha256",
    "dataset_final_hash_manifest_sha256",
    "dataset_manifest_payload_sha256",
    "calibration_final_hash_manifest_sha256",
    "calibration_payload_sha256",
    "development_evaluation_final_hash_manifest_sha256",
    "development_scientific_files_sha256",
    "development_input_manifest_payload_sha256",
    "development_result_payload_sha256",
    "development_evaluator_source_set_sha256",
    "production_authority_set_sha256",
    "training_source_set_sha256",
    "trajectory_receipt_file_sha256",
    "trajectory_receipt_payload_sha256",
    "trajectory_storage_manifest_file_sha256",
    "trajectory_storage_manifest_payload_sha256",
    "ordered_restart_records_sha256",
)


class ProtectedPopulationError(ValueError):
    """A protected role was requested without its complete reveal authority."""


class EvaluationOutputError(RuntimeError):
    """A closed output packet could not be written atomically."""


@dataclass(frozen=True)
class VerifiedProspectiveControl:
    freeze_packet: ClosedPacket
    freeze: dict[str, Any]
    source: dict[str, Any]
    authorization_path: Path
    authorization: dict[str, Any]
    authorization_file_sha256: str


@dataclass(frozen=True)
class ProspectivePopulation:
    states: np.ndarray
    indices: np.ndarray
    provenance: dict[str, Any]


@dataclass(frozen=True)
class ProspectiveAccessReceipt:
    path: Path
    value: dict[str, Any]
    file_sha256: str


@dataclass(frozen=True)
class ClosedPacket:
    root: Path
    final: dict[str, Any]
    files: dict[str, dict[str, Any]]
    final_file_sha256: str


@dataclass(frozen=True)
class TrainingPacket:
    packet: ClosedPacket
    arm: str
    seed: int
    config: dict[str, Any]
    inputs: dict[str, Any]
    runtime: dict[str, Any]
    summary: dict[str, Any]
    checkpoint_record: dict[str, Any]


@dataclass(frozen=True)
class VerifiedAuthorityArtifact:
    path: Path
    payload: dict[str, Any]
    file_sha256: str


@dataclass(frozen=True)
class VerifiedAuthoritySet:
    records: tuple[dict[str, Any], ...]
    artifacts: tuple[VerifiedAuthorityArtifact, ...]
    set_sha256: str


@dataclass(frozen=True)
class Projection:
    state: np.ndarray
    distance: float
    segment: int
    alpha: float
    tangent: np.ndarray | None


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def _self_hashed(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    if "canonical_payload_sha256" in result:
        raise ValueError("canonical payload hash was supplied twice")
    result["canonical_payload_sha256"] = _canonical_sha256(result)
    return result


def _json_number(value: float) -> float | str:
    value = float(value)
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "Infinity" if value > 0.0 else "-Infinity"
    return value


def _json_safe(value: Any) -> Any:
    if isinstance(value, (float, np.floating)):
        return _json_number(float(value))
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _numeric(value: Any) -> float:
    if isinstance(value, bool):
        raise TypeError("boolean is not a numeric metric")
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    if value == "NaN":
        return math.nan
    if value == "Infinity":
        return math.inf
    if value == "-Infinity":
        return -math.inf
    raise ValueError("unsupported JSON numeric representation")


def _require_digest(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _canonical_path_sha256(path: Path) -> str:
    canonical = os.path.normcase(os.path.normpath(os.fspath(path.resolve(strict=True))))
    return sha256(canonical.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_regular_file(path: Path, *, label: str) -> Path:
    if path.is_symlink():
        raise ValueError(f"{label} is aliased")
    resolved = path.resolve()
    if not resolved.is_file():
        raise ValueError(f"{label} is absent or aliased")
    return resolved


def _safe_directory(path: Path, *, label: str) -> Path:
    if path.is_symlink():
        raise ValueError(f"{label} is aliased")
    resolved = path.resolve()
    if not resolved.is_dir():
        raise ValueError(f"{label} is absent or aliased")
    return resolved


def _repo_source_file(relative: Any, *, label: str) -> Path:
    if (
        not isinstance(relative, str)
        or Path(relative).is_absolute()
        or Path(relative).as_posix() != relative
        or ".." in Path(relative).parts
    ):
        raise ValueError(f"{label} path is not repository relative")
    path = _safe_regular_file(REPO_ROOT / relative, label=label)
    try:
        path.relative_to(REPO_ROOT)
    except ValueError as error:
        raise ValueError(f"{label} escapes the repository") from error
    return path


def _read_json_bytes(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    resolved = _safe_regular_file(path, label=label)
    payload = resolved.read_bytes()
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise TypeError(f"{label} is not a JSON object")
    if resolved.read_bytes() != payload:
        raise ValueError(f"{label} changed while reading")
    return value, payload


def _read_self_hashed_json(
    path: Path, *, schema: str, label: str
) -> tuple[dict[str, Any], str]:
    value, payload = _read_json_bytes(path, label=label)
    if value.get("schema") != schema:
        raise ValueError(f"{label} uses an unsupported schema")
    observed = value.pop("canonical_payload_sha256", None)
    if observed != _canonical_sha256(value):
        raise ValueError(f"{label} canonical payload hash differs")
    value["canonical_payload_sha256"] = observed
    return value, sha256(payload).hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def _file_record(path: Path, root: Path) -> dict[str, Any]:
    resolved = _safe_regular_file(path, label=f"artifact {path.name}")
    try:
        relative = resolved.relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"artifact escapes packet root: {path}") from error
    return {
        "relative_path": relative,
        "bytes": resolved.stat().st_size,
        "sha256": _file_sha256(resolved),
    }


def _verify_record(
    root: Path, record: Mapping[str, Any], expected: str
) -> tuple[Path, bytes]:
    if record.get("relative_path") != expected:
        raise ValueError(f"artifact record path differs for {expected}")
    if Path(expected).as_posix() != expected or Path(expected).is_absolute():
        raise ValueError(
            f"artifact record is not a canonical relative path: {expected}"
        )
    candidate = root / expected
    resolved = _safe_regular_file(candidate, label=f"packet artifact {expected}")
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"packet artifact escapes its root: {expected}") from error
    payload = resolved.read_bytes()
    if len(payload) != record.get("bytes") or sha256(payload).hexdigest() != record.get(
        "sha256"
    ):
        raise ValueError(f"packet artifact differs from final hashes: {expected}")
    if resolved.read_bytes() != payload:
        raise ValueError(f"packet artifact changed while reading: {expected}")
    return resolved, payload


def _verify_closed_packet(
    root: Path,
    *,
    final_schema: str,
    expected_files: set[str] | None,
    label: str,
) -> ClosedPacket:
    packet_root = _safe_directory(root, label=label)
    final_path = packet_root / "final_hash_manifest.json"
    final, final_bytes = _read_json_bytes(final_path, label=f"{label} final manifest")
    files = final.get("files")
    if (
        final.get("schema") != final_schema
        or final.get("self_hash_excluded") is not True
        or not isinstance(files, dict)
        or not files
    ):
        raise ValueError(f"{label} final manifest differs")
    if final.get("sealed_opened") is not False:
        raise ValueError(f"{label} claims sealed access")
    names = set(files)
    if expected_files is not None and names != expected_files:
        raise ValueError(f"{label} has an unexpected exact file set")
    observed = {path.name for path in packet_root.iterdir()}
    if observed != names | {"final_hash_manifest.json"}:
        raise ValueError(f"{label} has missing or unmanifested artifacts")
    records: dict[str, dict[str, Any]] = {}
    for name in sorted(names):
        record = files[name]
        if not isinstance(record, dict):
            raise TypeError(f"{label} record is invalid: {name}")
        _verify_record(packet_root, record, name)
        records[name] = dict(record)
    if final_path.read_bytes() != final_bytes:
        raise ValueError(f"{label} final manifest changed while reading")
    return ClosedPacket(
        root=packet_root,
        final=final,
        files=records,
        final_file_sha256=sha256(final_bytes).hexdigest(),
    )


def _packet_json(
    packet: ClosedPacket, name: str, *, schema: str, label: str
) -> dict[str, Any]:
    path, payload = _verify_record(packet.root, packet.files[name], name)
    value = json.loads(payload)
    if not isinstance(value, dict) or value.get("schema") != schema:
        raise ValueError(f"{label} uses an unsupported schema")
    observed = value.pop("canonical_payload_sha256", None)
    if observed != _canonical_sha256(value):
        raise ValueError(f"{label} canonical payload hash differs")
    value["canonical_payload_sha256"] = observed
    if path.read_bytes() != payload:
        raise ValueError(f"{label} changed while loading")
    return value


def _binding(value: Mapping[str, Any], key: str, expected: Any, label: str) -> None:
    if value.get(key) != expected:
        raise ValueError(f"{label} {key} binding differs")


def _load_successor_contract(
    path: Path, baseline_path: Path, preregistration_path: Path
) -> tuple[dict[str, Any], str, Any]:
    value, payload = _read_json_bytes(path, label="successor contract")
    file_sha256 = sha256(payload).hexdigest()
    if value.get("schema") != SUCCESSOR_CONTRACT_SCHEMA:
        raise ValueError("unsupported successor contract schema")
    if value.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("successor experiment identity differs")
    packet_schemas = value.get("packet_schemas")
    required_schemas = {
        "calibration",
        "execution_authorization",
        "training",
        "training_config",
        "training_inputs",
        "evaluation",
        "evaluation_inputs",
        "final_hash_manifest",
        "prospective_freeze",
        "prospective_authorization",
        "resource_smoke",
    }
    if not isinstance(packet_schemas, dict) or set(packet_schemas) != required_schemas:
        raise ValueError("successor packet schemas are incomplete")
    if any(not isinstance(packet_schemas[name], str) for name in required_schemas):
        raise ValueError("successor packet schema names must be strings")
    if (
        packet_schemas["final_hash_manifest"] != FINAL_HASH_MANIFEST_SCHEMA
        or packet_schemas["prospective_freeze"] != PROSPECTIVE_FREEZE_SCHEMA
        or packet_schemas["prospective_authorization"]
        != PROSPECTIVE_AUTHORIZATION_SCHEMA
    ):
        raise ValueError("successor prospective packet schemas differ")

    baseline_resolved = _safe_regular_file(baseline_path, label="R0 contract")
    baseline_sha256 = _file_sha256(baseline_resolved)
    _binding(
        value,
        "inherited_r0_contract_sha256",
        baseline_sha256,
        "successor contract",
    )
    baseline = load_naca_baseline_contract(baseline_resolved)
    validate_successor_math_contract(value, baseline)
    preregistration = _safe_regular_file(
        preregistration_path, label="successor preregistration"
    )
    _binding(
        value,
        "preregistration_sha256",
        _file_sha256(preregistration),
        "successor contract",
    )

    dataset = value.get("dataset", {})
    expected_ranges = {
        "train_frames_inclusive": [955, 1194],
        "train_centers_inclusive": [956, 1193],
        "development_frames_inclusive": [1233, 1472],
        "development_centers_inclusive": [1234, 1471],
        "prospective_frames_inclusive": [1511, 1750],
        "sealed_frames_inclusive": [1755, 1994],
    }
    if not isinstance(dataset, dict):
        raise TypeError("successor dataset contract is invalid")
    _require_digest(dataset.get("final_hash_manifest_sha256"), "dataset digest")
    for key, expected in expected_ranges.items():
        _binding(dataset, key, expected, "successor dataset")

    optimization = value.get("optimization", {})
    if (
        optimization.get("seeds") != list(SEEDS)
        or optimization.get("epochs") != 100
        or optimization.get("effective_batch_size") != 4
        or optimization.get("rollout_used_for_selection") is not False
    ):
        raise ValueError("successor optimization contract differs")
    evaluation = value.get("evaluation", {})
    expected_anchors = [1234, 1238, 1243, 1247, 1251, 1256, 1260, 1264]
    if (
        evaluation.get("development_anchors") != expected_anchors
        or evaluation.get("horizons") != list(HORIZONS)
        or evaluation.get("full_trace_inclusive") != list(FULL_TRACE)
        or evaluation.get("late_window_inclusive") != list(LATE_WINDOW)
        or evaluation.get("tie_ratio_inclusive") != [0.95, 1.05]
        or evaluation.get("sampling_statement")
        != "deterministic_finite_population_common_trajectory"
    ):
        raise ValueError("successor evaluation contract differs")
    protection = value.get("protection", {})
    if (
        protection.get("prospective_opened") is not False
        or protection.get("sealed_opened") is not False
        or protection.get("online_solver_calls") is not False
        or protection.get("online_defect_trigger") is not False
    ):
        raise ValueError("successor protection boundary differs")
    if _safe_regular_file(path, label="successor contract").read_bytes() != payload:
        raise ValueError("successor contract changed while loading")
    return value, file_sha256, baseline


def _source_snapshot() -> dict[str, Any]:
    records = {
        relative: dict(record)
        for relative, record in sorted(R0_SOURCE_RECORDS_AT_IMPORT.items())
    }
    records.update(
        {
            relative: dict(record)
            for relative, record in sorted(SUCCESSOR_SOURCE_RECORDS_AT_IMPORT.items())
        }
    )
    return _self_hashed(
        {
            "schema": SOURCE_MANIFEST_SCHEMA,
            "experiment_id": EXPERIMENT_ID,
            "files": records,
            "source_set_sha256": _canonical_sha256(records),
        }
    )


def _reverify_source(source: Mapping[str, Any]) -> None:
    observed: dict[str, Any] = {}
    for relative, record in source["files"].items():
        path = _repo_source_file(relative, label=f"source {relative}")
        observed[relative] = {
            "bytes": path.stat().st_size,
            "sha256": _file_sha256(path),
        }
        if observed[relative] != record:
            raise ValueError(f"executing source changed: {relative}")
    if _canonical_sha256(observed) != source.get("source_set_sha256"):
        raise ValueError("executing source-set digest changed")


def _reverify_packet(packet: ClosedPacket) -> None:
    observed = {path.name for path in packet.root.iterdir()}
    if observed != set(packet.files) | {"final_hash_manifest.json"}:
        raise ValueError("verified packet gained or lost an artifact")
    for name, record in packet.files.items():
        _verify_record(packet.root, record, name)
    if (
        _file_sha256(packet.root / "final_hash_manifest.json")
        != packet.final_file_sha256
    ):
        raise ValueError("verified packet final manifest changed")


def _validate_freeze_parent_bindings(value: Any) -> None:
    if not isinstance(value, dict) or set(value) != set(PARENT_DIGEST_FIELDS) | {
        "training_packets"
    }:
        raise ValueError("prospective parent-binding field set differs")
    for key in PARENT_DIGEST_FIELDS:
        _require_digest(value.get(key), f"freeze {key}")
    training_packets = value.get("training_packets")
    expected_keys = [(arm, seed) for arm in LEARNED_ARMS for seed in SEEDS]
    if not isinstance(training_packets, list) or len(training_packets) != len(
        expected_keys
    ):
        raise ValueError("freeze training packet inventory differs")
    for record, expected_key in zip(training_packets, expected_keys, strict=True):
        if not isinstance(record, dict) or set(record) != {
            "arm",
            "seed",
            "final_hash_manifest_sha256",
            "checkpoint_sha256",
            "source_set_sha256",
        }:
            raise ValueError("freeze training packet binding fields differ")
        if (record.get("arm"), record.get("seed")) != expected_key:
            raise ValueError("freeze training packet ordering or key differs")
        for key in (
            "final_hash_manifest_sha256",
            "checkpoint_sha256",
            "source_set_sha256",
        ):
            _require_digest(record.get(key), f"freeze training packet {key}")
        if record["source_set_sha256"] != value["training_source_set_sha256"]:
            raise ValueError("freeze training source-set binding differs")


def _validate_prospective_freeze_semantics(
    freeze: Mapping[str, Any],
    source: Mapping[str, Any],
) -> None:
    if set(freeze) != FREEZE_FIELDS:
        raise ValueError("prospective freeze field set differs")
    _require_digest(
        freeze.get("successor_contract_sha256"),
        "freeze successor contract digest",
    )
    _require_digest(
        freeze.get("canonical_payload_sha256"),
        "freeze canonical payload digest",
    )
    _require_digest(
        source.get("source_set_sha256"),
        "freeze source-set digest",
    )
    if (
        freeze.get("schema") != PROSPECTIVE_FREEZE_SCHEMA
        or freeze.get("status") != "complete"
        or freeze.get("experiment_id") != EXPERIMENT_ID
        or freeze.get("population_role") != "prospective"
        or freeze.get("prospective_frames_inclusive") != list(PROSPECTIVE_FRAMES)
        or freeze.get("anchors") != list(PROSPECTIVE_ANCHORS)
        or freeze.get("seeds") != list(SEEDS)
        or freeze.get("primary_deployments")
        != [list(item) for item in PRIMARY_DEPLOYMENTS]
        or freeze.get("identity_parity_deployment")
        != ["CLEAN", "identity", "corrected"]
        or freeze.get("intervention_magnitude_deployment")
        != ["PATH_PROJECTION", "path_projection", "raw_pre_correction"]
        or freeze.get("primary_metrics") != list(PRIMARY_METRICS)
        or freeze.get("mediator_metrics") != list(MEDIATOR_METRICS)
        or freeze.get("horizons") != list(HORIZONS)
        or freeze.get("full_trace_inclusive") != list(FULL_TRACE)
        or freeze.get("late_window_inclusive") != list(LATE_WINDOW)
        or freeze.get("tie_ratio_inclusive") != [0.95, 1.05]
        or freeze.get("aggregation") != "paired_per_anchor_ratio_then_seedwise_median"
        or freeze.get("sampling") != "deterministic_finite_population_common_trajectory"
        or freeze.get("expected_result_cardinalities") != EXPECTED_RESULT_CARDINALITIES
        or freeze.get("evaluator_source_set_sha256") != source.get("source_set_sha256")
        or freeze.get("claim_boundary") != FREEZE_CLAIM_BOUNDARY
        or freeze.get("prospective_opened") is not False
        or freeze.get("sealed_opened") is not False
    ):
        raise ValueError("prospective freeze semantic content differs")
    _validate_freeze_parent_bindings(freeze.get("parent_bindings"))
    _validate_prediction_matrix(freeze.get("predictions"))


def _load_prospective_control(
    freeze_root: Path, authorization_path: Path
) -> VerifiedProspectiveControl:
    """Verify reveal authority without resolving any scientific input path."""

    packet = _verify_closed_packet(
        freeze_root,
        final_schema=FINAL_HASH_MANIFEST_SCHEMA,
        expected_files=FREEZE_OUTPUT_FILES,
        label="prospective freeze packet",
    )
    if (
        set(packet.final)
        != {
            "schema",
            "experiment_id",
            "packet_role",
            "successor_contract_sha256",
            "population_role",
            "files",
            "self_hash_excluded",
            "prospective_opened",
            "sealed_opened",
        }
        or packet.final.get("experiment_id") != EXPERIMENT_ID
        or packet.final.get("packet_role") != "prospective_freeze"
        or packet.final.get("population_role") != "prospective"
        or packet.final.get("prospective_opened") is not False
        or packet.final.get("sealed_opened") is not False
    ):
        raise ProtectedPopulationError("prospective freeze final manifest differs")
    freeze = _packet_json(
        packet,
        "prospective_freeze.json",
        schema=PROSPECTIVE_FREEZE_SCHEMA,
        label="prospective freeze payload",
    )
    source = _packet_json(
        packet,
        "source_manifest.json",
        schema=SOURCE_MANIFEST_SCHEMA,
        label="prospective freeze source manifest",
    )
    try:
        _validate_prospective_freeze_semantics(freeze, source)
    except (TypeError, ValueError) as error:
        raise ProtectedPopulationError(str(error)) from error
    if packet.final.get("successor_contract_sha256") != freeze.get(
        "successor_contract_sha256"
    ):
        raise ProtectedPopulationError("prospective freeze contract binding differs")
    current_source = _source_snapshot()
    if source != current_source:
        raise ProtectedPopulationError(
            "executing source differs from prospective freeze"
        )
    _reverify_source(source)

    try:
        authorization, authorization_file_sha256 = _read_self_hashed_json(
            authorization_path,
            schema=PROSPECTIVE_AUTHORIZATION_SCHEMA,
            label="prospective authorization",
        )
    except (OSError, TypeError, ValueError) as error:
        raise ProtectedPopulationError(str(error)) from error
    if (
        set(authorization) != PROSPECTIVE_AUTHORIZATION_FIELDS
        or authorization.get("status") != "authorized"
        or authorization.get("experiment_id") != EXPERIMENT_ID
        or authorization.get("successor_contract_sha256")
        != freeze.get("successor_contract_sha256")
        or authorization.get("prospective_freeze_final_hash_manifest_sha256")
        != packet.final_file_sha256
        or authorization.get("prospective_freeze_payload_sha256")
        != freeze.get("canonical_payload_sha256")
        or authorization.get("prospective_freeze_source_set_sha256")
        != source.get("source_set_sha256")
        or authorization.get("prospective_freeze_root_sha256")
        != _canonical_path_sha256(packet.root)
        or authorization.get("allowed_actions") != [PROSPECTIVE_ACTION]
        or authorization.get("authorized_population_role") != "prospective"
        or authorization.get("prospective_reveal_authorized") is not True
        or authorization.get("sealed_reveal_authorized") is not False
        or authorization.get("prospective_opened") is not False
        or authorization.get("sealed_opened") is not False
        or not isinstance(authorization.get("authorized_by"), str)
        or not authorization["authorized_by"].strip()
    ):
        raise ProtectedPopulationError("prospective authorization differs")
    return VerifiedProspectiveControl(
        freeze_packet=packet,
        freeze=freeze,
        source=source,
        authorization_path=_safe_regular_file(
            authorization_path, label="prospective authorization"
        ),
        authorization=authorization,
        authorization_file_sha256=authorization_file_sha256,
    )


def _reverify_prospective_control(control: VerifiedProspectiveControl) -> None:
    _reverify_packet(control.freeze_packet)
    _reverify_source(control.source)
    _validate_prospective_freeze_semantics(control.freeze, control.source)
    authorization, file_sha256 = _read_self_hashed_json(
        control.authorization_path,
        schema=PROSPECTIVE_AUTHORIZATION_SCHEMA,
        label="prospective authorization",
    )
    if (
        authorization != control.authorization
        or file_sha256 != control.authorization_file_sha256
    ):
        raise ProtectedPopulationError("prospective authorization changed")


def _prospective_access_receipt_path(
    control: VerifiedProspectiveControl,
) -> Path:
    """Return the sole receipt path authorized by this freeze/authority pair."""

    binding_sha256 = _canonical_sha256(
        {
            "prospective_freeze_final_hash_manifest_sha256": (
                control.freeze_packet.final_file_sha256
            ),
            "prospective_authorization_payload_sha256": (
                control.authorization["canonical_payload_sha256"]
            ),
        }
    )
    return control.freeze_packet.root.parent / (
        f"prospective_access_receipt_{binding_sha256}.json"
    )


def _absolute_path(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _paths_overlap(left: Path, right: Path) -> bool:
    left_forms = {_absolute_path(left), left.resolve(strict=False)}
    right_forms = {_absolute_path(right), right.resolve(strict=False)}
    return any(
        left_form == right_form
        or left_form in right_form.parents
        or right_form in left_form.parents
        for left_form in left_forms
        for right_form in right_forms
    )


def _validate_output_disjoint(
    output: Path,
    inputs: Sequence[tuple[str, Path]],
    *,
    label: str,
) -> None:
    for input_label, input_path in inputs:
        if _paths_overlap(output, input_path):
            raise ValueError(f"{label} overlaps {input_label}")


def _validate_prospective_reveal_paths(
    arguments: argparse.Namespace,
    control: VerifiedProspectiveControl,
) -> None:
    """Keep prospective outputs disjoint from every immutable scientific input."""

    supplied_receipt = getattr(arguments, "prospective_access_receipt", None)
    output = getattr(arguments, "output_dir", None)
    if supplied_receipt is None or output is None:
        raise ProtectedPopulationError(
            "prospective access receipt and output paths are required"
        )
    receipt = _absolute_path(supplied_receipt)
    expected_receipt = _absolute_path(_prospective_access_receipt_path(control))
    if receipt != expected_receipt:
        raise ProtectedPopulationError(
            "prospective access receipt path differs from its frozen deterministic path"
        )
    output = _absolute_path(output)
    if _paths_overlap(receipt, output):
        raise ProtectedPopulationError(
            "prospective access receipt and evaluation output overlap"
        )

    inputs: list[tuple[str, Path]] = [
        ("prospective freeze", control.freeze_packet.root),
        ("prospective authorization", control.authorization_path),
    ]
    for name in (
        "contract",
        "successor_contract",
        "preregistration",
        "dataset_dir",
        "calibration_dir",
        "development_evaluation_dir",
        "trajectory_dir",
        "trajectory_storage_manifest",
    ):
        value = getattr(arguments, name, None)
        if value is not None:
            inputs.append((name.replace("_", " "), value))
    for name in (
        "training_dir",
        "resource_smoke_receipt",
        "execution_authorization",
    ):
        for value in getattr(arguments, name, None) or ():
            inputs.append((name.replace("_", " "), value))
    for label, input_path in inputs:
        if _paths_overlap(receipt, input_path):
            raise ProtectedPopulationError(
                f"prospective access receipt overlaps {label}"
            )
        if _paths_overlap(output, input_path):
            raise ProtectedPopulationError(
                f"prospective evaluation output overlaps {label}"
            )


def _projection_from_core(
    state: np.ndarray,
    corrected: np.ndarray,
    segment: int,
    alpha: float,
    projector: TrainPathProjector,
) -> Projection:
    normalized = np.asarray(
        projector.normalization.normalize_state(state), dtype=np.float64
    )
    corrected_normalized = np.asarray(
        projector.normalization.normalize_state(corrected), dtype=np.float64
    )
    distance = float(np.sqrt(np.mean(np.square(normalized - corrected_normalized))))
    path = projector.normalized_path.detach().cpu().numpy()
    tangent_delta = np.asarray(path[segment + 1] - path[segment], dtype=np.float64)
    tangent_rms = float(np.sqrt(np.mean(np.square(tangent_delta))))
    tangent = (
        tangent_delta / tangent_rms
        if math.isfinite(tangent_rms) and tangent_rms > TANGENT_STABILITY_RMS
        else None
    )
    return Projection(
        state=np.asarray(corrected, dtype=np.float64),
        distance=distance,
        segment=int(segment),
        alpha=float(alpha),
        tangent=tangent,
    )


def _project_state(projector: TrainPathProjector, state: np.ndarray) -> Projection:
    query = torch.from_numpy(np.asarray(state, dtype=np.float64)[None, ...].copy())
    result = projector.project(query)
    return _projection_from_core(
        state,
        result.corrected_state[0].detach().cpu().numpy(),
        int(result.segment_indices[0].item()),
        float(result.segment_fractions[0].item()),
        projector,
    )


def _reference_bank_bytes(projector: TrainPathProjector) -> int:
    return sum(
        int(tensor.numel() * tensor.element_size())
        for tensor in (
            projector.normalized_path,
            projector.pca_mean,
            projector.pca_basis,
            projector.path_embedding,
        )
    )


def _load_calibration(
    root: Path,
    *,
    successor: Mapping[str, Any],
    successor_sha256: str,
    dataset_sha256: str,
    normalization: NACANormalization,
    train_states: np.ndarray,
    train_indices: np.ndarray,
) -> tuple[ClosedPacket, dict[str, Any], TrainPathProjector]:
    schemas = successor["packet_schemas"]
    packet = _verify_closed_packet(
        root,
        final_schema=schemas["final_hash_manifest"],
        expected_files=EXPECTED_CALIBRATION_FILES,
        label="successor calibration packet",
    )
    for key, expected in {
        "experiment_id": EXPERIMENT_ID,
        "packet_role": "calibration",
        "successor_contract_sha256": successor_sha256,
        "prospective_opened": False,
        "sealed_opened": False,
    }.items():
        _binding(packet.final, key, expected, "calibration final manifest")
    calibration = _packet_json(
        packet,
        "calibration.json",
        schema=schemas["calibration"],
        label="calibration payload",
    )
    for key, expected in {
        "status": "complete",
        "experiment_id": EXPERIMENT_ID,
        "successor_contract_sha256": successor_sha256,
        "inherited_r0_contract_sha256": successor["inherited_r0_contract_sha256"],
        "dataset_final_hash_manifest_sha256": dataset_sha256,
        "parent_r0_evaluation_final_hash_manifest_sha256": successor["parent_r0"][
            "evaluation_final_hash_manifest_sha256"
        ],
        "fit_role": "train_only",
    }.items():
        _binding(calibration, key, expected, "calibration payload")
    if calibration.get("access") != {
        "train_opened": True,
        "development_opened": True,
        "prospective_opened": False,
        "sealed_opened": False,
    }:
        raise ValueError("calibration access record differs")
    if calibration.get("claim_boundary") != {
        "trusted_displaced_state_response_measured": False,
        "manifold_drift_established": False,
        "successor_rollout_opened": False,
    }:
        raise ValueError("calibration claim boundary differs")
    if not isinstance(calibration.get("runtime"), dict):
        raise TypeError("calibration runtime record is absent")
    arrays_record = calibration.get("arrays")
    if not isinstance(arrays_record, dict) or set(arrays_record) != {"file", "members"}:
        raise ValueError("calibration array inventory is invalid")
    if arrays_record["file"] != packet.files["calibration_arrays.npz"]:
        raise ValueError("calibration array record differs from final hashes")
    _, arrays_bytes = _verify_record(
        packet.root, packet.files["calibration_arrays.npz"], "calibration_arrays.npz"
    )
    with np.load(io.BytesIO(arrays_bytes), allow_pickle=False) as archive:
        arrays = {key: np.array(archive[key], copy=True) for key in archive.files}
    recovery_mapping = {
        key.removeprefix("recovery__"): value
        for key, value in arrays.items()
        if key.startswith("recovery__")
    }
    path_mapping = {
        key.removeprefix("path__"): value
        for key, value in arrays.items()
        if key.startswith("path__")
    }
    if (
        not recovery_mapping
        or not path_mapping
        or len(arrays) != len(recovery_mapping) + len(path_mapping)
    ):
        raise ValueError(
            "calibration NPZ must contain only recovery__ and path__ mappings"
        )
    recovery = RecoveryCalibration.from_mapping(recovery_mapping)
    projector = TrainPathProjector.from_mapping(path_mapping, normalization)
    member_records = arrays_record["members"]
    if not isinstance(member_records, dict) or set(member_records) != set(arrays):
        raise ValueError("calibration array-member inventory differs")
    for key, array in arrays.items():
        contiguous = np.ascontiguousarray(array)
        digest = sha256()
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(contiguous.dtype.str.encode("ascii"))
        digest.update(contiguous.tobytes(order="C"))
        expected_record = {
            "shape": list(array.shape),
            "dtype": array.dtype.str,
            "array_sha256": digest.hexdigest(),
        }
        if member_records[key] != expected_record:
            raise ValueError(f"calibration array member record differs: {key}")
    node_count = int(train_states.shape[1])
    pair_dimension = 2 * node_count * len(FIELDS)
    state_dimension = node_count * len(FIELDS)
    if recovery.num_nodes != node_count or projector.num_nodes != node_count:
        raise ValueError("calibration mappings differ from the fixed mesh")
    expected_centers = np.arange(956, 1194, dtype=np.int64)
    positions = {int(index): offset for offset, index in enumerate(train_indices)}
    if train_indices.dtype != np.int64 or any(
        int(index) not in positions for index in expected_centers
    ):
        raise ValueError("train role lacks the exact path-fit current-state centers")
    expected_train_path = np.asarray(
        normalization.normalize_state(
            np.stack(
                [train_states[positions[int(index)]] for index in expected_centers]
            )
        ),
        dtype=np.float64,
    )
    if not np.array_equal(
        projector.normalized_path.detach().cpu().numpy(),
        expected_train_path,
    ):
        raise ValueError("calibration path states differ from normalized train states")
    summary = calibration.get("calibration")
    if not isinstance(summary, dict):
        raise TypeError("calibration numerical summary is absent")
    expected_summary = {
        "iid_field_std": recovery.iid_field_rms.tolist(),
        "iid_expected_history_pair_squared_norm": (
            recovery.iid_expected_history_pair_energy
        ),
        "error_pair_count": recovery.pair_sample_count,
        "error_pair_dimension": pair_dimension,
        "error_pair_rank16_capture_fraction": recovery.structured_captured_variance,
        "error_pair_energy_rescale": recovery.structured_energy_rescale,
        "error_pair_matched_expected_squared_norm": float(
            torch.sum(torch.square(recovery.structured_coefficient_std)).item()
        ),
        "structured_sampling_mean": "zero",
        "path_state_count": 238,
        "path_state_dimension": state_dimension,
        "path_rank": projector.rank,
        "path_rank_without_cap": projector.variance_rank_without_cap,
        "path_rank_cap_active": projector.rank_cap_active,
        "path_retained_variance_fraction": projector.captured_variance,
    }
    if set(summary) != set(expected_summary):
        raise ValueError("calibration numerical summary differs")
    for key, expected in expected_summary.items():
        observed = summary[key]
        if isinstance(expected, bool):
            matches = observed is expected
        elif isinstance(expected, float):
            matches = _same_number(observed, expected)
        else:
            matches = observed == expected
        if not matches:
            raise ValueError(f"calibration numerical summary differs: {key}")
    parent_records = calibration.get("parent_checkpoints")
    if not isinstance(parent_records, dict) or set(parent_records) != {
        str(seed) for seed in SEEDS
    }:
        raise ValueError("calibration parent checkpoint inventory differs")
    for seed in SEEDS:
        record = parent_records[str(seed)]
        expected = successor["parent_r0"]["checkpoints"][str(seed)]
        if (
            not isinstance(record, dict)
            or record.get("final_hash_manifest_sha256")
            != expected["final_hash_manifest_sha256"]
            or record.get("checkpoint_sha256") != expected["checkpoint_sha256"]
        ):
            raise ValueError(f"calibration parent checkpoint differs: {seed}")
    source_record = calibration.get("source")
    if not isinstance(source_record, dict) or set(source_record) != {
        "files",
        "source_set_sha256",
    }:
        raise ValueError("calibration source closure is invalid")
    source_files = source_record["files"]
    if not isinstance(source_files, dict) or not source_files:
        raise ValueError("calibration source closure is empty")
    if source_record["source_set_sha256"] != _canonical_sha256(source_files):
        raise ValueError("calibration source-set digest differs")
    for relative, record in source_files.items():
        path = _repo_source_file(relative, label=f"calibration source {relative}")
        if record != {"bytes": path.stat().st_size, "sha256": _file_sha256(path)}:
            raise ValueError(f"calibration source differs: {relative}")
    _reverify_packet(packet)
    return packet, calibration, projector


def _same_number(left: Any, right: Any) -> bool:
    left_value = _numeric(left)
    right_value = _numeric(right)
    if math.isnan(left_value) or math.isnan(right_value):
        return math.isnan(left_value) and math.isnan(right_value)
    return left_value == right_value


def _selection_score(value: Any, *, label: str, json_artifact: bool) -> float:
    score = _numeric(value)
    if math.isnan(score) or score == -math.inf:
        raise ValueError(f"{label} must be finite or canonical +Infinity")
    if json_artifact and score == math.inf and value != "Infinity":
        raise ValueError(f"{label} uses a noncanonical +Infinity representation")
    return score


def _validate_training_bindings(
    value: Mapping[str, Any],
    *,
    label: str,
    arm: str,
    seed: int,
    successor: Mapping[str, Any],
    successor_sha256: str,
    dataset_sha256: str,
    dataset_payload_sha256: str,
    calibration_sha256: str,
    calibration_payload_sha256: str,
) -> None:
    _binding(value, "experiment_id", EXPERIMENT_ID, label)
    _binding(value, "arm", arm, label)
    _binding(value, "seed", seed, label)
    _binding(
        value,
        "successor_contract_sha256",
        successor_sha256,
        label,
    )
    _binding(
        value,
        "preregistration_sha256",
        successor["preregistration_sha256"],
        label,
    )
    _binding(
        value,
        "inherited_r0_contract_sha256",
        successor["inherited_r0_contract_sha256"],
        label,
    )
    _binding(
        value,
        "dataset_manifest_payload_sha256",
        dataset_payload_sha256,
        label,
    )
    _binding(
        value,
        "dataset_final_hash_manifest_sha256",
        dataset_sha256,
        label,
    )
    _binding(
        value,
        "calibration_payload_sha256",
        calibration_payload_sha256,
        label,
    )
    _binding(
        value,
        "calibration_final_hash_manifest_sha256",
        calibration_sha256,
        label,
    )
    _require_digest(value.get("source_set_sha256"), f"{label} source-set digest")
    if (
        value.get("prospective_opened") is not False
        or value.get("sealed_opened") is not False
        or value.get("online_solver_calls") is not False
        or value.get("online_defect_trigger") is not False
    ):
        raise ValueError(f"{label} crosses a protected population boundary")


def _load_checkpoint_mapping(
    packet: ClosedPacket, device: torch.device
) -> tuple[dict[str, Any], bytes]:
    _, payload = _verify_record(packet.root, packet.files["best.pt"], "best.pt")
    checkpoint = torch.load(io.BytesIO(payload), map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict):
        raise TypeError("selected checkpoint is not a mapping")
    return checkpoint, payload


def _verify_training_packet(
    root: Path,
    *,
    successor: Mapping[str, Any],
    successor_sha256: str,
    dataset_sha256: str,
    dataset_payload_sha256: str,
    calibration_sha256: str,
    calibration_payload_sha256: str,
    baseline: Mapping[str, Any],
    geometry: VerifiedNACAGeometry,
) -> TrainingPacket:
    schemas = successor["packet_schemas"]
    packet = _verify_closed_packet(
        root,
        final_schema=schemas["final_hash_manifest"],
        expected_files=EXPECTED_TRAINING_FILES,
        label="successor training packet",
    )
    summary = _packet_json(
        packet,
        "summary.json",
        schema=schemas["training"],
        label="training summary",
    )
    arm = summary.get("arm")
    seed = summary.get("seed")
    if arm not in LEARNED_ARMS:
        raise ValueError("training packet has an unsupported learned arm")
    if isinstance(seed, bool) or seed not in SEEDS:
        raise ValueError("training packet has an unsupported seed")
    seed = int(seed)
    config = _packet_json(
        packet,
        "config.json",
        schema=schemas["training_config"],
        label="training config",
    )
    inputs = _packet_json(
        packet,
        "input_manifest.json",
        schema=schemas["training_inputs"],
        label="training input manifest",
    )
    status = _packet_json(
        packet,
        "status.json",
        schema=schemas["training"],
        label="training status",
    )
    runtime = _packet_json(
        packet,
        "runtime_manifest.json",
        schema=schemas["training"],
        label="training runtime manifest",
    )
    for key, expected in {
        "experiment_id": EXPERIMENT_ID,
        "arm": arm,
        "seed": seed,
        "successor_contract_sha256": successor_sha256,
        "inherited_r0_contract_sha256": successor["inherited_r0_contract_sha256"],
        "dataset_final_hash_manifest_sha256": dataset_sha256,
        "calibration_final_hash_manifest_sha256": calibration_sha256,
        "prospective_opened": False,
        "sealed_opened": False,
        "online_solver_calls": False,
        "online_defect_trigger": False,
    }.items():
        _binding(packet.final, key, expected, "training final manifest")
    for value, label in (
        (config, "training config"),
        (inputs, "training input manifest"),
        (summary, "training summary"),
        (status, "training status"),
        (runtime, "training runtime manifest"),
    ):
        _validate_training_bindings(
            value,
            label=label,
            arm=arm,
            seed=seed,
            successor=successor,
            successor_sha256=successor_sha256,
            dataset_sha256=dataset_sha256,
            dataset_payload_sha256=dataset_payload_sha256,
            calibration_sha256=calibration_sha256,
            calibration_payload_sha256=calibration_payload_sha256,
        )
    source_digests = {
        value["source_set_sha256"]
        for value in (config, inputs, summary, status, runtime)
    }
    if len(source_digests) != 1:
        raise ValueError("training packet source bindings disagree")
    source_files = inputs.get("source_files")
    if (
        not isinstance(source_files, dict)
        or not source_files
        or _canonical_sha256(source_files) != next(iter(source_digests))
    ):
        raise ValueError("training source closure is invalid")
    for relative, record in source_files.items():
        path = _repo_source_file(relative, label=f"training source {relative}")
        if record != {"bytes": path.stat().st_size, "sha256": _file_sha256(path)}:
            raise ValueError(f"training source differs: {relative}")
    optimization = successor["optimization"]
    if (
        config.get("epochs") != optimization["epochs"]
        or config.get("effective_batch_size") != optimization["effective_batch_size"]
        or config.get("checkpoint_epochs") != optimization["checkpoint_epochs"]
        or config.get("rollout_used_for_selection") is not False
        or summary.get("record_kind") != "summary"
        or summary.get("status") != "complete"
        or summary.get("completed_epochs") != optimization["epochs"]
        or summary.get("checkpoint_selection_used_rollout") is not False
        or status.get("record_kind") != "status"
        or status.get("status") != "complete"
        or status.get("last_completed_epoch") != optimization["epochs"]
        or runtime.get("record_kind") != "runtime_manifest"
        or not isinstance(runtime.get("runtime"), dict)
        or runtime.get("production_device_binding")
        != inputs.get("production_device_binding")
    ):
        raise ValueError("training completion or selection contract differs")
    best_epoch = summary.get("best_epoch")
    if (
        isinstance(best_epoch, bool)
        or best_epoch not in optimization["checkpoint_epochs"]
        or status.get("best_epoch") != best_epoch
    ):
        raise ValueError("training selected an unsupported checkpoint epoch")
    best_score_key = "best_clean_development_score"
    if best_score_key not in summary or best_score_key not in status:
        raise ValueError("training packet lacks its clean selection score")
    _selection_score(
        summary[best_score_key],
        label="training summary selection score",
        json_artifact=True,
    )
    _selection_score(
        status[best_score_key],
        label="training status selection score",
        json_artifact=True,
    )
    if not _same_number(status[best_score_key], summary[best_score_key]):
        raise ValueError("training status selection score differs from summary")

    initial_model_sha256 = _require_digest(
        summary.get("initial_model_state_sha256"),
        "training initial-model state digest",
    )
    presentation_sha256 = _require_digest(
        summary.get("presentation_schedule_sha256"),
        "training presentation-schedule digest",
    )
    presentation = config.get("presentation")
    if (
        not isinstance(presentation, dict)
        or presentation.get("presentation_schedule_sha256") != presentation_sha256
    ):
        raise ValueError(
            "training paired initialization or presentation binding differs"
        )
    for value, label in (
        (packet.final, "training final manifest"),
        (config, "training config"),
        (inputs, "training input manifest"),
        (runtime, "training runtime manifest"),
        (summary, "training summary"),
        (status, "training status"),
    ):
        if (
            value.get("paired_initialization_and_clean_order") is not True
            or value.get("initial_model_state_sha256") != initial_model_sha256
            or value.get("presentation_schedule_sha256") != presentation_sha256
        ):
            raise ValueError(f"{label} paired initialization binding differs")
    for key in (
        "resource_smoke_payload_sha256",
        "resource_smoke_file_sha256",
        "authorization_payload_sha256",
        "authorization_file_sha256",
    ):
        _require_digest(inputs.get(key), f"training input {key}")
    if not isinstance(inputs.get("production_device_binding"), dict):
        raise TypeError("training input production-device binding is invalid")

    checkpoint, checkpoint_bytes = _load_checkpoint_mapping(packet, torch.device("cpu"))
    if (
        checkpoint.get("schema") != schemas["training"]
        or checkpoint.get("record_kind") != "selected_checkpoint"
    ):
        raise ValueError("selected checkpoint schema or kind differs")
    _validate_training_bindings(
        checkpoint,
        label="selected checkpoint",
        arm=arm,
        seed=seed,
        successor=successor,
        successor_sha256=successor_sha256,
        dataset_sha256=dataset_sha256,
        dataset_payload_sha256=dataset_payload_sha256,
        calibration_sha256=calibration_sha256,
        calibration_payload_sha256=calibration_payload_sha256,
    )
    if (
        checkpoint.get("model_only") is not True
        or checkpoint.get("epoch") != best_epoch
        or checkpoint.get("best_epoch") != best_epoch
        or not _same_number(checkpoint.get(best_score_key), summary[best_score_key])
        or checkpoint.get("initial_model_state_sha256") != initial_model_sha256
        or checkpoint.get("presentation_schedule_sha256") != presentation_sha256
        or checkpoint.get("paired_initialization_and_clean_order") is not True
    ):
        raise ValueError("selected checkpoint does not match the training summary")
    _selection_score(
        checkpoint.get(best_score_key),
        label="selected checkpoint selection score",
        json_artifact=False,
    )
    validate_naca_model_config(checkpoint.get("model_config", {}), baseline, geometry)
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict) or not state:
        raise ValueError("selected checkpoint lacks a model-only state dictionary")
    del checkpoint
    if (
        len(checkpoint_bytes) != packet.files["best.pt"]["bytes"]
        or sha256(checkpoint_bytes).hexdigest() != packet.files["best.pt"]["sha256"]
    ):
        raise ValueError("selected checkpoint bytes changed while validating")
    _reverify_packet(packet)
    return TrainingPacket(
        packet=packet,
        arm=arm,
        seed=seed,
        config=config,
        inputs=inputs,
        runtime=runtime,
        summary=summary,
        checkpoint_record=dict(packet.files["best.pt"]),
    )


def _load_training_packets(
    roots: Sequence[Path],
    *,
    successor: Mapping[str, Any],
    successor_sha256: str,
    dataset_sha256: str,
    dataset_payload_sha256: str,
    calibration_sha256: str,
    calibration_payload_sha256: str,
    baseline: Mapping[str, Any],
    geometry: VerifiedNACAGeometry,
) -> list[TrainingPacket]:
    if len(roots) != len(LEARNED_ARMS) * len(SEEDS):
        raise ValueError("evaluation requires exactly twelve learned training packets")
    packets = [
        _verify_training_packet(
            root,
            successor=successor,
            successor_sha256=successor_sha256,
            dataset_sha256=dataset_sha256,
            dataset_payload_sha256=dataset_payload_sha256,
            calibration_sha256=calibration_sha256,
            calibration_payload_sha256=calibration_payload_sha256,
            baseline=baseline,
            geometry=geometry,
        )
        for root in roots
    ]
    keys = [(packet.arm, packet.seed) for packet in packets]
    expected = {(arm, seed) for arm in LEARNED_ARMS for seed in SEEDS}
    if set(keys) != expected or len(keys) != len(set(keys)):
        raise ValueError("training packets do not form the four-arm three-seed grid")
    source_sets = {packet.summary["source_set_sha256"] for packet in packets}
    if len(source_sets) != 1:
        raise ValueError(
            "training packets were not produced from one frozen source set"
        )
    _verify_paired_training_hashes(packets)
    packets.sort(key=lambda packet: (LEARNED_ARMS.index(packet.arm), packet.seed))
    return packets


def _verify_paired_training_hashes(packets: Sequence[TrainingPacket]) -> None:
    for seed in SEEDS:
        paired = [packet for packet in packets if packet.seed == seed]
        initial_models = {
            packet.summary["initial_model_state_sha256"] for packet in paired
        }
        presentations = {
            packet.summary["presentation_schedule_sha256"] for packet in paired
        }
        if (
            len(paired) != len(LEARNED_ARMS)
            or {packet.arm for packet in paired} != set(LEARNED_ARMS)
            or len(initial_models) != 1
            or len(presentations) != 1
        ):
            raise ValueError(
                f"learned arms are not paired by initialization and presentation for seed {seed}"
            )


def _instantiate_model(
    packet: TrainingPacket,
    *,
    baseline: Mapping[str, Any],
    geometry: VerifiedNACAGeometry,
    device: torch.device,
) -> torch.nn.Module:
    checkpoint, payload = _load_checkpoint_mapping(packet.packet, device)
    if sha256(payload).hexdigest() != packet.checkpoint_record["sha256"]:
        raise ValueError("selected checkpoint changed before model construction")
    validate_naca_model_config(checkpoint["model_config"], baseline, geometry)
    model = build_naca_pcno(baseline, geometry).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model


def _load_authority_artifacts(
    paths: Sequence[Path], *, schema: str, label: str
) -> dict[tuple[str, str], VerifiedAuthorityArtifact]:
    artifacts: dict[tuple[str, str], VerifiedAuthorityArtifact] = {}
    for position, path in enumerate(paths):
        payload, file_sha256 = _read_self_hashed_json(
            path, schema=schema, label=f"{label} {position}"
        )
        payload_sha256 = _require_digest(
            payload.get("canonical_payload_sha256"), f"{label} payload digest"
        )
        key = (file_sha256, payload_sha256)
        if key in artifacts:
            raise ValueError(f"{label} was supplied more than once")
        artifacts[key] = VerifiedAuthorityArtifact(
            path=_safe_regular_file(path, label=label),
            payload=payload,
            file_sha256=file_sha256,
        )
    return artifacts


def _verify_production_authorities(
    *,
    smoke_paths: Sequence[Path],
    authorization_paths: Sequence[Path],
    training_packets: Sequence[TrainingPacket],
    successor: Mapping[str, Any],
    successor_sha256: str,
    dataset_sha256: str,
    calibration_sha256: str,
    geometry: VerifiedNACAGeometry,
) -> VerifiedAuthoritySet:
    schemas = successor["packet_schemas"]
    smokes = _load_authority_artifacts(
        smoke_paths, schema=schemas["resource_smoke"], label="resource-smoke receipt"
    )
    authorizations = _load_authority_artifacts(
        authorization_paths,
        schema=schemas["execution_authorization"],
        label="execution authorization",
    )
    expected_smokes = {
        (
            packet.inputs["resource_smoke_file_sha256"],
            packet.inputs["resource_smoke_payload_sha256"],
        )
        for packet in training_packets
    }
    expected_authorizations = {
        (
            packet.inputs["authorization_file_sha256"],
            packet.inputs["authorization_payload_sha256"],
        )
        for packet in training_packets
    }
    if set(smokes) != expected_smokes:
        raise ValueError("resource-smoke receipts do not exactly cover training inputs")
    if set(authorizations) != expected_authorizations:
        raise ValueError(
            "execution authorizations do not exactly cover training inputs"
        )

    records: list[dict[str, Any]] = []
    shared_smoke_keys = (
        "experiment_id",
        "arm",
        "seed",
        "successor_contract_sha256",
        "preregistration_sha256",
        "inherited_r0_contract_sha256",
        "dataset_manifest_payload_sha256",
        "dataset_final_hash_manifest_sha256",
        "calibration_payload_sha256",
        "calibration_final_hash_manifest_sha256",
        "source_set_sha256",
        "prospective_opened",
        "sealed_opened",
        "online_solver_calls",
        "online_defect_trigger",
    )
    for packet in training_packets:
        inputs = packet.inputs
        smoke_key = (
            inputs["resource_smoke_file_sha256"],
            inputs["resource_smoke_payload_sha256"],
        )
        authorization_key = (
            inputs["authorization_file_sha256"],
            inputs["authorization_payload_sha256"],
        )
        smoke = smokes[smoke_key].payload
        authorization = authorizations[authorization_key].payload
        loss = smoke.get("loss")
        gradient_norm = smoke.get("gradient_norm_before_clip")
        presentation = packet.config["presentation"]
        smoke_intervention_rng = smoke.get("intervention_rng")
        config_intervention_rng = packet.config.get("intervention_rng")
        if (
            any(smoke.get(key) != inputs.get(key) for key in shared_smoke_keys)
            or smoke.get("status") != "succeeded"
            or smoke.get("effective_batch_size") != 4
            or smoke.get("microbatch_size") != packet.config.get("microbatch_size")
            or smoke.get("gradient_accumulation_steps")
            != packet.config.get("gradient_accumulation_steps")
            or smoke.get("full_resolution_num_nodes") != geometry.num_nodes
            or smoke.get("initial_model_state_sha256")
            != packet.summary["initial_model_state_sha256"]
            or smoke.get("presentation_schedule_sha256")
            != presentation["presentation_schedule_sha256"]
            or smoke.get("paired_initialization_and_clean_order") is not True
            or smoke.get("runtime") != packet.runtime["runtime"]
            or not isinstance(smoke_intervention_rng, dict)
            or not isinstance(config_intervention_rng, dict)
            or smoke_intervention_rng.get("initial_generator_state_sha256")
            != config_intervention_rng.get("initial_generator_state_sha256")
            or smoke.get("production_device_binding")
            != inputs["production_device_binding"]
            or smoke.get("forward_backward_and_adamw_step_completed") is not True
            or smoke.get("scientific_hyperparameters_changed") is not False
            or smoke.get("checkpoint_written") is not False
            or isinstance(loss, bool)
            or not isinstance(loss, (int, float))
            or not math.isfinite(float(loss))
            or isinstance(gradient_norm, bool)
            or not isinstance(gradient_norm, (int, float))
            or not math.isfinite(float(gradient_norm))
            or isinstance(smoke.get("training_model_calls"), bool)
            or not isinstance(smoke.get("training_model_calls"), int)
            or smoke["training_model_calls"] < 1
        ):
            raise ValueError(
                f"resource-smoke receipt differs for {packet.arm} seed {packet.seed}"
            )

        authorized_arms = authorization.get("authorized_arms")
        authorized_seeds = authorization.get("authorized_seeds")
        expected_authorization = {
            "status": "authorized",
            "experiment_id": EXPERIMENT_ID,
            "successor_contract_sha256": successor_sha256,
            "preregistration_sha256": successor["preregistration_sha256"],
            "inherited_r0_contract_sha256": successor["inherited_r0_contract_sha256"],
            "dataset_final_hash_manifest_sha256": dataset_sha256,
            "calibration_final_hash_manifest_sha256": calibration_sha256,
            "source_set_sha256": packet.summary["source_set_sha256"],
            "resource_smoke_payload_sha256": smoke_key[1],
            "resource_smoke_file_sha256": smoke_key[0],
            "production_device_binding": inputs["production_device_binding"],
            "allowed_actions": ["train_pcno_naca0012_successor"],
            "pcno_code_audited": True,
            "successor_training_authorized": True,
            "prospective_opened": False,
            "sealed_opened": False,
            "online_solver_calls": False,
            "online_defect_trigger": False,
        }
        if (
            any(
                authorization.get(key) != expected
                for key, expected in expected_authorization.items()
            )
            or not isinstance(authorized_arms, list)
            or len(authorized_arms) != len(set(authorized_arms))
            or any(arm not in LEARNED_ARMS for arm in authorized_arms)
            or packet.arm not in authorized_arms
            or not isinstance(authorized_seeds, list)
            or len(authorized_seeds) != len(set(authorized_seeds))
            or any(seed not in SEEDS for seed in authorized_seeds)
            or packet.seed not in authorized_seeds
            or not isinstance(authorization.get("authorized_by"), str)
            or not authorization["authorized_by"].strip()
        ):
            raise ValueError(
                f"execution authorization differs for {packet.arm} seed {packet.seed}"
            )
        records.append(
            {
                "arm": packet.arm,
                "seed": packet.seed,
                "resource_smoke_file_sha256": smoke_key[0],
                "resource_smoke_payload_sha256": smoke_key[1],
                "authorization_file_sha256": authorization_key[0],
                "authorization_payload_sha256": authorization_key[1],
            }
        )

    records.sort(key=lambda item: (LEARNED_ARMS.index(item["arm"]), item["seed"]))
    record_tuple = tuple(records)
    artifacts = tuple(
        sorted(
            [*smokes.values(), *authorizations.values()],
            key=lambda artifact: artifact.file_sha256,
        )
    )
    result = VerifiedAuthoritySet(
        records=record_tuple,
        artifacts=artifacts,
        set_sha256=_canonical_sha256({"training_authorities": list(record_tuple)}),
    )
    _reverify_authorities(result)
    return result


def _reverify_authorities(authorities: VerifiedAuthoritySet) -> None:
    for artifact in authorities.artifacts:
        resolved = _safe_regular_file(
            artifact.path, label="production authority artifact"
        )
        if resolved != artifact.path:
            raise ValueError("production authority artifact path changed")
        payload = resolved.read_bytes()
        if sha256(payload).hexdigest() != artifact.file_sha256:
            raise ValueError("production authority artifact changed during evaluation")
        value = json.loads(payload)
        if not isinstance(value, dict):
            raise TypeError("production authority artifact ceased to be a JSON object")
        observed = value.pop("canonical_payload_sha256", None)
        if observed != _canonical_sha256(value):
            raise ValueError("production authority artifact self-hash changed")
        if resolved.read_bytes() != payload:
            raise ValueError("production authority artifact changed while reverifying")


def _graph_edges(geometry: VerifiedNACAGeometry) -> np.ndarray:
    edges: set[tuple[int, int]] = set()
    for element in np.asarray(geometry.elements[:, 1:], dtype=np.int64):
        for first, second in zip(element, np.roll(element, -1), strict=True):
            edge = tuple(sorted((int(first), int(second))))
            if edge[0] == edge[1]:
                raise ValueError("geometry contains a self edge")
            edges.add(edge)
    result = np.asarray(sorted(edges), dtype=np.int64)
    if result.ndim != 2 or result.shape[1] != 2 or result.size == 0:
        raise ValueError("geometry has no usable undirected graph edges")
    return result


def _structure_context(
    geometry: VerifiedNACAGeometry, train_states: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    airfoil, farfield, edges, edge_geometry = _airfoil_geometry(geometry)
    normals = np.zeros((geometry.num_nodes, 2), dtype=np.float64)
    for (first, second), record in zip(edges, edge_geometry, strict=True):
        normal = np.asarray(record[:2], dtype=np.float64)
        length = float(record[2])
        normals[first] += length * normal
        normals[second] += length * normal
    norms = np.linalg.norm(normals[airfoil], axis=1)
    if np.any(norms <= 0.0):
        raise ValueError("airfoil vertex normal is undefined")
    normals[airfoil] /= norms[:, None]
    _, force_scale = _force_scales(train_states, edges, edge_geometry)
    return airfoil, farfield, normals, edges, edge_geometry, force_scale


def _vector_diagnostics(
    difference: np.ndarray,
    *,
    normalization: NACANormalization,
    tangent: np.ndarray | None,
    graph_edges: np.ndarray,
) -> dict[str, Any]:
    vector = np.asarray(difference, dtype=np.float64)
    finite = bool(np.all(np.isfinite(vector)))
    if not finite:
        return {
            "finite": False,
            "normalized_rms": math.inf,
            **{f"normalized_rms_{field}": math.inf for field in FIELDS},
            "tangent_signed": math.nan,
            "tangent_magnitude": math.nan,
            "transverse_rms": math.nan,
            "tangent_direction_valid": tangent is not None,
            "graph_dirichlet_energy": math.inf,
            "graph_dirichlet_to_nodal_energy": math.inf,
        }
    normalized = vector / normalization.state_scale
    per_field = np.sqrt(np.mean(np.square(normalized), axis=0))
    total_sq = float(np.mean(np.square(normalized)))
    if tangent is None:
        signed = math.nan
        transverse = math.nan
    else:
        signed = float(np.mean(normalized * tangent))
        transverse = math.sqrt(max(total_sq - signed * signed, 0.0))
    edge_difference = normalized[graph_edges[:, 0]] - normalized[graph_edges[:, 1]]
    dirichlet = float(np.mean(np.square(edge_difference)))
    return {
        "finite": True,
        "normalized_rms": math.sqrt(total_sq),
        **_metric_columns("normalized_rms", per_field),
        "tangent_signed": signed,
        "tangent_magnitude": abs(signed),
        "transverse_rms": transverse,
        "tangent_direction_valid": tangent is not None,
        "graph_dirichlet_energy": dirichlet,
        "graph_dirichlet_to_nodal_energy": dirichlet / max(total_sq, 1.0e-30),
    }


def _prefix_columns(prefix: str, value: Mapping[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{key}": item for key, item in value.items()}


def _state_metrics(
    prediction: np.ndarray | None,
    target: np.ndarray,
    *,
    target_projection: Projection,
    prediction_projection: Projection | None,
    projector: TrainPathProjector,
    normalization: NACANormalization,
    weights: Mapping[str, np.ndarray],
    graph_edges: np.ndarray,
    geometry: VerifiedNACAGeometry,
    structure_context: tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ],
) -> dict[str, Any]:
    if prediction is None:
        return {
            "finite": False,
            "finite_value_fraction": 0.0,
            "normalized_state_error": math.inf,
            "relative_l2": math.inf,
            **{f"normalized_state_error_{field}": math.inf for field in FIELDS},
            **{f"relative_l2_{field}": math.inf for field in FIELDS},
            "near_body_wake_normalized_state_error": math.inf,
            "physical_area_normalized_state_error": math.inf,
            "path_distance": math.inf,
            "projected_path_discrepancy": math.inf,
            "path_segment": None,
            "path_alpha": math.nan,
            "tangent_signed_error": math.nan,
            "tangent_error_magnitude": math.nan,
            "transverse_error": math.nan,
            "tangent_direction_valid": target_projection.tangent is not None,
            "graph_dirichlet_error_energy": math.inf,
            "graph_dirichlet_to_nodal_error_energy": math.inf,
        }
    state = np.asarray(prediction, dtype=np.float64)
    finite_fraction = float(np.mean(np.isfinite(state)))
    if finite_fraction != 1.0:
        row = _state_metrics(
            None,
            target,
            target_projection=target_projection,
            prediction_projection=None,
            projector=projector,
            normalization=normalization,
            weights=weights,
            graph_edges=graph_edges,
            geometry=geometry,
            structure_context=structure_context,
        )
        row["finite_value_fraction"] = finite_fraction
        return row
    uniform = _field_metrics(state, target, weights["uniform_node"], normalization)
    near = _field_metrics(
        state, target, weights["near_body_wake_uniform_node"], normalization
    )
    physical = _field_metrics(
        state, target, weights["physical_vertex_area"], normalization
    )
    projection = prediction_projection or _project_state(projector, state)
    projected_error = _field_metrics(
        projection.state, target, weights["uniform_node"], normalization
    )["train_state_scale"]
    vector = _vector_diagnostics(
        state - target,
        normalization=normalization,
        tangent=target_projection.tangent,
        graph_edges=graph_edges,
    )
    airfoil, farfield, normals, edges, edge_geometry, force_scale = structure_context
    structure = _structure_metrics(
        state,
        target,
        geometry,
        normalization,
        airfoil,
        farfield,
        normals,
        edges,
        edge_geometry,
        force_scale,
    )
    return {
        "finite": True,
        "finite_value_fraction": 1.0,
        "normalized_state_error": uniform["train_state_scale"],
        "relative_l2": uniform["relative_l2"],
        **_metric_columns(
            "normalized_state_error", uniform["train_state_scale_per_field"]
        ),
        **_metric_columns("relative_l2", uniform["relative_l2_per_field"]),
        "near_body_wake_normalized_state_error": near["train_state_scale"],
        "physical_area_normalized_state_error": physical["train_state_scale"],
        "path_distance": projection.distance,
        "projected_path_discrepancy": projected_error,
        "path_segment": projection.segment,
        "path_alpha": projection.alpha,
        "tangent_signed_error": vector["tangent_signed"],
        "tangent_error_magnitude": vector["tangent_magnitude"],
        "transverse_error": vector["transverse_rms"],
        "tangent_direction_valid": vector["tangent_direction_valid"],
        "graph_dirichlet_error_energy": vector["graph_dirichlet_energy"],
        "graph_dirichlet_to_nodal_error_energy": vector[
            "graph_dirichlet_to_nodal_energy"
        ],
        **structure,
    }


def _reference_cache(
    states: np.ndarray, indices: np.ndarray, projector: TrainPathProjector
) -> dict[int, Projection]:
    if states.shape[0] != indices.shape[0]:
        raise ValueError("reference state/index cardinality differs")
    cache: dict[int, Projection] = {}
    for offset, index in enumerate(indices):
        cache[int(index)] = _project_state(projector, states[offset])
    return cache


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _offline_diagnostics(
    *,
    packet: TrainingPacket,
    model: torch.nn.Module,
    states: np.ndarray,
    indices: np.ndarray,
    anchors: Sequence[int],
    geometry: VerifiedNACAGeometry,
    normalization: NACANormalization,
    projector: TrainPathProjector,
    reference_cache: Mapping[int, Projection],
    graph_edges: np.ndarray,
    weights: Mapping[str, np.ndarray],
    structure_context: tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ],
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, int | float]]:
    positions = {int(index): offset for offset, index in enumerate(indices)}
    for anchor in anchors:
        for required in (anchor - 1, anchor, anchor + 1, anchor + 2):
            if required not in positions:
                raise ValueError(
                    "offline diagnostic anchor escapes its population role"
                )
    previous = torch.as_tensor(
        np.stack([states[positions[anchor - 1]] for anchor in anchors]).astype(
            np.float32
        ),
        device=device,
    )
    current = torch.as_tensor(
        np.stack([states[positions[anchor]] for anchor in anchors]).astype(np.float32),
        device=device,
    )
    clean_next = torch.as_tensor(
        np.stack([states[positions[anchor + 1]] for anchor in anchors]).astype(
            np.float32
        ),
        device=device,
    )
    geometry_batch = geometry.expand(len(anchors), device)
    fourier = model.prepare_fourier_tensors(geometry_batch)  # type: ignore[attr-defined]
    _synchronize(device)
    started = time.perf_counter()
    with torch.no_grad():
        _, generated_current = recurrent_step(
            model,
            previous,
            current,
            geometry_batch,
            normalization,
            fourier_tensors=fourier,
        )
        _, clean_forced_next = recurrent_step(
            model,
            current,
            clean_next,
            geometry_batch,
            normalization,
            fourier_tensors=fourier,
        )
        _, prefix_next = recurrent_step(
            model,
            current,
            generated_current,
            geometry_batch,
            normalization,
            fourier_tensors=fourier,
        )
    _synchronize(device)
    elapsed = time.perf_counter() - started
    generated_np = generated_current.detach().cpu().numpy().astype(np.float64)
    clean_forced_np = clean_forced_next.detach().cpu().numpy().astype(np.float64)
    prefix_np = prefix_next.detach().cpu().numpy().astype(np.float64)
    rows: list[dict[str, Any]] = []
    for offset, anchor in enumerate(anchors):
        target_current = np.asarray(states[positions[anchor + 1]], dtype=np.float64)
        target_next = np.asarray(states[positions[anchor + 2]], dtype=np.float64)
        forcing = _vector_diagnostics(
            generated_np[offset] - target_current,
            normalization=normalization,
            tangent=reference_cache[anchor + 1].tangent,
            graph_edges=graph_edges,
        )
        clean_error = _vector_diagnostics(
            clean_forced_np[offset] - target_next,
            normalization=normalization,
            tangent=reference_cache[anchor + 2].tangent,
            graph_edges=graph_edges,
        )
        prefix_error = _vector_diagnostics(
            prefix_np[offset] - target_next,
            normalization=normalization,
            tangent=reference_cache[anchor + 2].tangent,
            graph_edges=graph_edges,
        )
        response = _vector_diagnostics(
            prefix_np[offset] - clean_forced_np[offset],
            normalization=normalization,
            tangent=reference_cache[anchor + 2].tangent,
            graph_edges=graph_edges,
        )
        generated_projection = (
            _project_state(projector, generated_np[offset])
            if forcing["finite"]
            else None
        )
        clean_forced_projection = (
            _project_state(projector, clean_forced_np[offset])
            if clean_error["finite"]
            else None
        )
        prefix_projection = (
            _project_state(projector, prefix_np[offset])
            if prefix_error["finite"]
            else None
        )
        clean_generated_state = _state_metrics(
            generated_np[offset],
            target_current,
            target_projection=reference_cache[anchor + 1],
            prediction_projection=generated_projection,
            projector=projector,
            normalization=normalization,
            weights=weights,
            graph_edges=graph_edges,
            geometry=geometry,
            structure_context=structure_context,
        )
        clean_forced_state = _state_metrics(
            clean_forced_np[offset],
            target_next,
            target_projection=reference_cache[anchor + 2],
            prediction_projection=clean_forced_projection,
            projector=projector,
            normalization=normalization,
            weights=weights,
            graph_edges=graph_edges,
            geometry=geometry,
            structure_context=structure_context,
        )
        prefix_state = _state_metrics(
            prefix_np[offset],
            target_next,
            target_projection=reference_cache[anchor + 2],
            prediction_projection=prefix_projection,
            projector=projector,
            normalization=normalization,
            weights=weights,
            graph_edges=graph_edges,
            geometry=geometry,
            structure_context=structure_context,
        )
        forcing_norm = float(forcing["normalized_rms"])
        response_norm = float(response["normalized_rms"])
        rows.append(
            {
                "arm": packet.arm,
                "seed": packet.seed,
                "anchor": anchor,
                "input_role": "common_clean_and_one_prefix",
                **_prefix_columns("clean_forcing", forcing),
                **_prefix_columns("clean_next", clean_error),
                **_prefix_columns("one_prefix_next", prefix_error),
                **_prefix_columns("one_prefix_response", response),
                "one_prefix_response_gain": response_norm / max(forcing_norm, 1.0e-30),
                **_prefix_columns("clean_generated_state", clean_generated_state),
                **_prefix_columns("clean_forced_state", clean_forced_state),
                **_prefix_columns("one_prefix_state", prefix_state),
            }
        )
    return rows, {
        "model_invocations": 3,
        "state_transition_evaluations": 3 * len(anchors),
        "wall_time_seconds": elapsed,
    }


STRUCTURE_KEYS = {
    "finite",
    "finite_value_fraction",
    "airfoil_boundary_train_state_scale_error",
    "farfield_boundary_train_state_scale_error",
    "wall_normal_momentum_leakage",
    "predicted_wall_normal_momentum_rms",
    "reference_wall_normal_momentum_rms",
    "density_nonpositive_fraction",
    "internal_energy_diagnostic_invalid_fraction",
    "internal_energy_nonpositive_fraction_among_valid_density",
    "nu_tilde_minimum",
    "nu_tilde_negative_fraction",
    "volume_integral_error",
    "pressure_force_valid",
    "predicted_pressure_drag_proxy",
    "predicted_pressure_lift_proxy",
    "reference_pressure_drag_proxy",
    "reference_pressure_lift_proxy",
    "pressure_drag_proxy_normalized_error",
    "pressure_lift_proxy_normalized_error",
} | {f"volume_integral_error_{field}" for field in FIELDS}


def _append_rollout_record(
    *,
    metric_rows: list[dict[str, Any]],
    structure_rows: list[dict[str, Any]],
    identifiers: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> None:
    metric_rows.append({**identifiers, **metrics})
    structure_rows.append(
        {
            **identifiers,
            **{key: metrics.get(key) for key in sorted(STRUCTURE_KEYS)},
        }
    )


def _raw_rollout(
    *,
    packet: TrainingPacket,
    model: torch.nn.Module,
    states: np.ndarray,
    indices: np.ndarray,
    anchors: Sequence[int],
    geometry: VerifiedNACAGeometry,
    normalization: NACANormalization,
    projector: TrainPathProjector,
    reference_cache: Mapping[int, Projection],
    graph_edges: np.ndarray,
    weights: Mapping[str, np.ndarray],
    structure_context: tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ],
    device: torch.device,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, np.ndarray],
    dict[str, Any],
]:
    positions = {int(index): offset for offset, index in enumerate(indices)}
    for anchor in anchors:
        if any(
            required not in positions
            for required in (anchor - 1, anchor, anchor + FULL_TRACE[1])
        ):
            raise ValueError("rollout anchor escapes its population role")
    previous = torch.as_tensor(
        np.stack([states[positions[anchor - 1]] for anchor in anchors]).astype(
            np.float32
        ),
        device=device,
    )
    current = torch.as_tensor(
        np.stack([states[positions[anchor]] for anchor in anchors]).astype(np.float32),
        device=device,
    )
    geometry_batch = geometry.expand(len(anchors), device)
    fourier = model.prepare_fourier_tensors(geometry_batch)  # type: ignore[attr-defined]
    active = torch.ones(len(anchors), dtype=torch.bool, device=device)
    first_nonfinite: list[int | None] = [None] * len(anchors)
    completed = np.zeros(len(anchors), dtype=np.int64)
    metric_rows: list[dict[str, Any]] = []
    structure_rows: list[dict[str, Any]] = []
    snapshots: dict[str, np.ndarray] = {}
    identity_checks = 0
    _synchronize(device)
    started = time.perf_counter()
    with torch.no_grad():
        for horizon in range(FULL_TRACE[0], FULL_TRACE[1] + 1):
            if packet.arm == "CLEAN":
                step = corrected_recurrent_step(
                    model,
                    previous,
                    current,
                    geometry_batch,
                    normalization,
                    identity_corrector,
                    fourier_tensors=fourier,
                )
                candidate = step.raw_next_state
                if step.corrected_next_state is not candidate:
                    raise AssertionError(
                        "identity corrector changed the raw prediction"
                    )
                recurrent_previous = step.recurrent_previous
                identity_checks += len(anchors)
            else:
                recurrent_previous, candidate = recurrent_step(
                    model,
                    previous,
                    current,
                    geometry_batch,
                    normalization,
                    fourier_tensors=fourier,
                )
            finite = torch.all(torch.isfinite(candidate), dim=(1, 2))
            newly_failed = active & ~finite
            for offset in (
                torch.nonzero(newly_failed, as_tuple=False).flatten().tolist()
            ):
                first_nonfinite[offset] = horizon
            retained = active & finite
            candidate_np = candidate.detach().cpu().numpy().astype(np.float64)
            for offset, anchor in enumerate(anchors):
                prediction = candidate_np[offset] if bool(retained[offset]) else None
                target = np.asarray(
                    states[positions[anchor + horizon]], dtype=np.float64
                )
                metrics = _state_metrics(
                    prediction,
                    target,
                    target_projection=reference_cache[anchor + horizon],
                    prediction_projection=None,
                    projector=projector,
                    normalization=normalization,
                    weights=weights,
                    graph_edges=graph_edges,
                    geometry=geometry,
                    structure_context=structure_context,
                )
                identifiers = {
                    "arm": packet.arm,
                    "seed": packet.seed,
                    "deployment": "raw",
                    "state_stage": "raw",
                    "anchor": anchor,
                    "horizon": horizon,
                }
                _append_rollout_record(
                    metric_rows=metric_rows,
                    structure_rows=structure_rows,
                    identifiers=identifiers,
                    metrics=metrics,
                )
                if packet.arm == "CLEAN":
                    identity_identifiers = {
                        **identifiers,
                        "deployment": "identity",
                        "state_stage": "corrected",
                    }
                    _append_rollout_record(
                        metric_rows=metric_rows,
                        structure_rows=structure_rows,
                        identifiers=identity_identifiers,
                        metrics=metrics,
                    )
                if prediction is not None:
                    completed[offset] += 1
                    if horizon in HORIZONS:
                        key = (
                            f"{packet.arm.lower()}_seed{packet.seed}_raw_"
                            f"a{anchor}_h{horizon}"
                        )
                        snapshots[key] = np.asarray(prediction, dtype=np.float32)
                        reference_key = f"reference_a{anchor}_h{horizon}"
                        snapshots.setdefault(
                            reference_key, np.asarray(target, dtype=np.float64)
                        )
            active = retained
            safe_candidate = torch.where(active[:, None, None], candidate, current)
            previous, current = recurrent_previous, safe_candidate
    _synchronize(device)
    elapsed = time.perf_counter() - started
    summary = {
        "first_nonfinite_step_by_anchor": {
            str(anchor): first_nonfinite[offset]
            for offset, anchor in enumerate(anchors)
        },
        "completed_steps_by_anchor": {
            str(anchor): int(completed[offset]) for offset, anchor in enumerate(anchors)
        },
        "identity_parity_state_checks": identity_checks,
        "identity_parity_exact": packet.arm == "CLEAN",
        "model_invocations": FULL_TRACE[1],
        "state_transition_evaluations": FULL_TRACE[1] * len(anchors),
        "corrector_calls": identity_checks if packet.arm == "CLEAN" else 0,
        "wall_time_seconds": elapsed,
    }
    return metric_rows, structure_rows, snapshots, summary


def _path_projection_rollout(
    *,
    packet: TrainingPacket,
    model: torch.nn.Module,
    states: np.ndarray,
    indices: np.ndarray,
    anchors: Sequence[int],
    geometry: VerifiedNACAGeometry,
    normalization: NACANormalization,
    projector: TrainPathProjector,
    reference_cache: Mapping[int, Projection],
    graph_edges: np.ndarray,
    weights: Mapping[str, np.ndarray],
    structure_context: tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ],
    device: torch.device,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, np.ndarray],
    dict[str, Any],
]:
    if packet.arm != "CLEAN":
        raise ValueError("PATH_PROJECTION can only attach to CLEAN")
    positions = {int(index): offset for offset, index in enumerate(indices)}
    previous = torch.as_tensor(
        np.stack([states[positions[anchor - 1]] for anchor in anchors]).astype(
            np.float32
        ),
        device=device,
    )
    current = torch.as_tensor(
        np.stack([states[positions[anchor]] for anchor in anchors]).astype(np.float32),
        device=device,
    )
    geometry_batch = geometry.expand(len(anchors), device)
    fourier = model.prepare_fourier_tensors(geometry_batch)  # type: ignore[attr-defined]
    device_projector = projector.to(device, dtype=torch.float32)
    active = torch.ones(len(anchors), dtype=torch.bool, device=device)
    first_nonfinite: list[int | None] = [None] * len(anchors)
    completed = np.zeros(len(anchors), dtype=np.int64)
    metric_rows: list[dict[str, Any]] = []
    structure_rows: list[dict[str, Any]] = []
    snapshots: dict[str, np.ndarray] = {}
    corrector_calls = 0
    _synchronize(device)
    started = time.perf_counter()
    with torch.no_grad():
        for horizon in range(FULL_TRACE[0], FULL_TRACE[1] + 1):
            projection_results: list[Any] = []
            retained_masks: list[torch.Tensor] = []

            def apply_projection(
                raw_state: torch.Tensor,
                *,
                active_now: torch.Tensor = active,
                current_now: torch.Tensor = current,
                results: list[Any] = projection_results,
                masks: list[torch.Tensor] = retained_masks,
            ) -> torch.Tensor:
                raw_finite = torch.all(torch.isfinite(raw_state), dim=(1, 2))
                retained_now = active_now & raw_finite
                safe_query = torch.where(
                    retained_now[:, None, None], raw_state, current_now
                )
                result = device_projector.project(safe_query)
                results.append(result)
                masks.append(retained_now)
                return torch.where(
                    retained_now[:, None, None], result.corrected_state, current_now
                )

            step = corrected_recurrent_step(
                model,
                previous,
                current,
                geometry_batch,
                normalization,
                apply_projection,
                fourier_tensors=fourier,
            )
            if len(projection_results) != 1 or len(retained_masks) != 1:
                raise AssertionError("PATH_PROJECTION corrector call count differs")
            raw_candidate = step.raw_next_state
            corrected = step.corrected_next_state
            projection_result = projection_results[0]
            retained = retained_masks[0]
            corrector_calls += len(anchors)
            raw_finite = torch.all(torch.isfinite(raw_candidate), dim=(1, 2))
            if not torch.equal(retained, active & raw_finite):
                raise AssertionError("PATH_PROJECTION finite mask changed")
            newly_failed = active & ~raw_finite
            for offset in (
                torch.nonzero(newly_failed, as_tuple=False).flatten().tolist()
            ):
                first_nonfinite[offset] = horizon
            raw_np = raw_candidate.detach().cpu().numpy().astype(np.float64)
            corrected_np = corrected.detach().cpu().numpy().astype(np.float64)
            projections: list[Projection | None] = [None] * len(anchors)
            for offset in range(len(anchors)):
                if bool(retained[offset]):
                    projection = _projection_from_core(
                        raw_np[offset],
                        corrected_np[offset],
                        int(projection_result.segment_indices[offset].item()),
                        float(projection_result.segment_fractions[offset].item()),
                        projector,
                    )
                    projections[offset] = projection
            corrected_finite = torch.all(torch.isfinite(corrected), dim=(1, 2))
            if torch.any(retained & ~corrected_finite):
                raise FloatingPointError("PATH_PROJECTION produced a nonfinite state")
            for offset, anchor in enumerate(anchors):
                target = np.asarray(
                    states[positions[anchor + horizon]], dtype=np.float64
                )
                raw_prediction = raw_np[offset] if bool(retained[offset]) else None
                projection = projections[offset]
                raw_metrics = _state_metrics(
                    raw_prediction,
                    target,
                    target_projection=reference_cache[anchor + horizon],
                    prediction_projection=projection,
                    projector=projector,
                    normalization=normalization,
                    weights=weights,
                    graph_edges=graph_edges,
                    geometry=geometry,
                    structure_context=structure_context,
                )
                common = {
                    "arm": "PATH_PROJECTION",
                    "parent_arm": "CLEAN",
                    "seed": packet.seed,
                    "deployment": "path_projection",
                    "anchor": anchor,
                    "horizon": horizon,
                }
                _append_rollout_record(
                    metric_rows=metric_rows,
                    structure_rows=structure_rows,
                    identifiers={**common, "state_stage": "raw_pre_correction"},
                    metrics=raw_metrics,
                )
                if projection is None:
                    corrected_prediction = None
                    corrected_projection = None
                else:
                    corrected_prediction = projection.state
                    corrected_projection = Projection(
                        state=projection.state,
                        distance=0.0,
                        segment=projection.segment,
                        alpha=projection.alpha,
                        tangent=projection.tangent,
                    )
                corrected_metrics = _state_metrics(
                    corrected_prediction,
                    target,
                    target_projection=reference_cache[anchor + horizon],
                    prediction_projection=corrected_projection,
                    projector=projector,
                    normalization=normalization,
                    weights=weights,
                    graph_edges=graph_edges,
                    geometry=geometry,
                    structure_context=structure_context,
                )
                _append_rollout_record(
                    metric_rows=metric_rows,
                    structure_rows=structure_rows,
                    identifiers={**common, "state_stage": "corrected"},
                    metrics=corrected_metrics,
                )
                if corrected_prediction is not None:
                    completed[offset] += 1
                    if horizon in HORIZONS:
                        snapshots[
                            f"path_projection_seed{packet.seed}_raw_"
                            f"a{anchor}_h{horizon}"
                        ] = np.asarray(raw_prediction, dtype=np.float32)
                        snapshots[
                            f"path_projection_seed{packet.seed}_corrected_"
                            f"a{anchor}_h{horizon}"
                        ] = np.asarray(corrected_prediction, dtype=np.float32)
                        snapshots.setdefault(
                            f"reference_a{anchor}_h{horizon}",
                            np.asarray(target, dtype=np.float64),
                        )
            active = retained
            previous, current = step.recurrent_previous, corrected
    _synchronize(device)
    elapsed = time.perf_counter() - started
    return (
        metric_rows,
        structure_rows,
        snapshots,
        {
            "first_nonfinite_step_by_anchor": {
                str(anchor): first_nonfinite[offset]
                for offset, anchor in enumerate(anchors)
            },
            "completed_steps_by_anchor": {
                str(anchor): int(completed[offset])
                for offset, anchor in enumerate(anchors)
            },
            "model_invocations": FULL_TRACE[1],
            "state_transition_evaluations": FULL_TRACE[1] * len(anchors),
            "corrector_calls": corrector_calls,
            "corrected_state_recurrence": True,
            "wall_time_seconds": elapsed,
        },
    )


def _nearest_rank(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("nearest-rank aggregation received no values")
    ordered = sorted(float(value) for value in values)
    rank = max(1, math.ceil(probability * len(ordered)))
    return ordered[rank - 1]


def _aggregate(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        raise ValueError("deterministic aggregation received no values")
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "median": _json_number(float(np.median(array))),
        "q90_nearest_rank": _json_number(_nearest_rank(values, 0.9)),
        "maximum": _json_number(float(np.max(array))),
    }


def _aggregate_valid_directions(values: Sequence[float]) -> dict[str, Any]:
    valid = [float(value) for value in values if math.isfinite(float(value))]
    if not valid:
        return {
            "count": 0,
            "median": "NaN",
            "q90_nearest_rank": "NaN",
            "maximum": "NaN",
        }
    return _aggregate(valid)


def _valid_direction_median(values: Sequence[float]) -> float:
    valid = np.asarray(
        [float(value) for value in values if math.isfinite(float(value))],
        dtype=np.float64,
    )
    return float(np.median(valid)) if valid.size else math.nan


def _normalized_auc(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (FULL_TRACE[1] - FULL_TRACE[0] + 1,):
        raise ValueError("AUC requires the complete 1--208 trace")
    if not np.all(np.isfinite(array)):
        return math.inf
    # Unit-spaced trapezoidal area divided by the 207-step interval length.
    return float(
        (0.5 * array[0] + np.sum(array[1:-1]) + 0.5 * array[-1]) / (array.size - 1)
    )


def _ratio(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        return 1.0 if numerator == 0.0 else math.inf
    if math.isinf(denominator):
        return 1.0 if math.isinf(numerator) else 0.0
    return numerator / denominator


def _ratio_classification(value: float) -> str:
    if 0.95 <= value <= 1.05:
        return "tie"
    return "improved" if value < 0.95 else "worse"


def _summaries(
    rows: Sequence[Mapping[str, Any]], anchors: Sequence[int]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    variants = sorted(
        {
            (
                str(row["arm"]),
                int(row["seed"]),
                str(row["deployment"]),
                str(row["state_stage"]),
            )
            for row in rows
        },
        key=lambda item: (ALL_ARMS.index(item[0]), item[1], item[2], item[3]),
    )
    lookup: dict[tuple[str, int, str, str, int, int], Mapping[str, Any]] = {}
    for row in rows:
        key = (
            str(row["arm"]),
            int(row["seed"]),
            str(row["deployment"]),
            str(row["state_stage"]),
            int(row["anchor"]),
            int(row["horizon"]),
        )
        if key in lookup:
            raise ValueError("rollout metric trace contains a duplicate key")
        lookup[key] = row
    expected_per_variant = len(anchors) * (FULL_TRACE[1] - FULL_TRACE[0] + 1)
    if len(lookup) != len(variants) * expected_per_variant:
        raise ValueError("rollout metric trace is incomplete")

    method_rows: list[dict[str, Any]] = []
    method_result: dict[str, Any] = {}
    for arm, seed, deployment, state_stage in variants:
        per_anchor: list[dict[str, Any]] = []
        for anchor in anchors:
            trace = [
                lookup[(arm, seed, deployment, state_stage, anchor, horizon)]
                for horizon in range(FULL_TRACE[0], FULL_TRACE[1] + 1)
            ]
            errors = [float(row["normalized_state_error"]) for row in trace]
            late = [
                float(row["normalized_state_error"])
                for row in trace
                if LATE_WINDOW[0] <= int(row["horizon"]) <= LATE_WINDOW[1]
            ]
            record = {
                "arm": arm,
                "seed": seed,
                "deployment": deployment,
                "state_stage": state_stage,
                "anchor": anchor,
                "normalized_state_error_auc": _normalized_auc(errors),
                "late_window_normalized_state_error_median": float(
                    np.median(np.asarray(late, dtype=np.float64))
                ),
                "late_window_path_distance_median": float(
                    np.median(
                        np.asarray(
                            [
                                float(row["path_distance"])
                                for row in trace
                                if LATE_WINDOW[0]
                                <= int(row["horizon"])
                                <= LATE_WINDOW[1]
                            ],
                            dtype=np.float64,
                        )
                    )
                ),
                "late_window_projected_path_discrepancy_median": float(
                    np.median(
                        np.asarray(
                            [
                                float(row["projected_path_discrepancy"])
                                for row in trace
                                if LATE_WINDOW[0]
                                <= int(row["horizon"])
                                <= LATE_WINDOW[1]
                            ],
                            dtype=np.float64,
                        )
                    )
                ),
                "late_window_transverse_error_median": _valid_direction_median(
                    [
                        float(row["transverse_error"])
                        for row in trace
                        if LATE_WINDOW[0] <= int(row["horizon"]) <= LATE_WINDOW[1]
                        and bool(row["tangent_direction_valid"])
                    ]
                ),
                "late_window_graph_dirichlet_error_energy_median": float(
                    np.median(
                        np.asarray(
                            [
                                float(row["graph_dirichlet_error_energy"])
                                for row in trace
                                if LATE_WINDOW[0]
                                <= int(row["horizon"])
                                <= LATE_WINDOW[1]
                            ],
                            dtype=np.float64,
                        )
                    )
                ),
                "complete_finite_rollout": all(bool(row["finite"]) for row in trace),
            }
            method_rows.append(record)
            per_anchor.append(record)
        variant_key = f"{arm}:{seed}:{deployment}:{state_stage}"
        method_result[variant_key] = {
            "arm": arm,
            "seed": seed,
            "deployment": deployment,
            "state_stage": state_stage,
            "per_anchor": {
                str(record["anchor"]): _json_safe(record) for record in per_anchor
            },
            "normalized_state_error_auc": _aggregate(
                [float(record["normalized_state_error_auc"]) for record in per_anchor]
            ),
            "late_window_normalized_state_error_median": _aggregate(
                [
                    float(record["late_window_normalized_state_error_median"])
                    for record in per_anchor
                ]
            ),
            "late_window_path_distance_median": _aggregate(
                [
                    float(record["late_window_path_distance_median"])
                    for record in per_anchor
                ]
            ),
            "late_window_projected_path_discrepancy_median": _aggregate(
                [
                    float(record["late_window_projected_path_discrepancy_median"])
                    for record in per_anchor
                ]
            ),
            "late_window_transverse_error_median": _aggregate_valid_directions(
                [
                    float(record["late_window_transverse_error_median"])
                    for record in per_anchor
                ]
            ),
            "late_window_graph_dirichlet_error_energy_median": _aggregate(
                [
                    float(record["late_window_graph_dirichlet_error_energy_median"])
                    for record in per_anchor
                ]
            ),
            "complete_finite_anchor_count": sum(
                bool(record["complete_finite_rollout"]) for record in per_anchor
            ),
        }

    method_lookup = {
        (
            str(row["arm"]),
            int(row["seed"]),
            str(row["deployment"]),
            str(row["state_stage"]),
            int(row["anchor"]),
        ): row
        for row in method_rows
    }
    pairwise_rows: list[dict[str, Any]] = []
    primary_variants = [
        variant
        for variant in variants
        if not (variant[0] == "PATH_PROJECTION" and variant[3] != "corrected")
    ]
    for arm, seed, deployment, state_stage in primary_variants:
        if arm == "CLEAN" and deployment == "raw":
            continue
        for anchor in anchors:
            candidate = method_lookup[(arm, seed, deployment, state_stage, anchor)]
            baseline = method_lookup[("CLEAN", seed, "raw", "raw", anchor)]
            for metric in (
                "normalized_state_error_auc",
                "late_window_normalized_state_error_median",
            ):
                ratio = _ratio(float(candidate[metric]), float(baseline[metric]))
                pairwise_rows.append(
                    {
                        "arm": arm,
                        "seed": seed,
                        "deployment": deployment,
                        "state_stage": state_stage,
                        "anchor": anchor,
                        "metric": metric,
                        "candidate": candidate[metric],
                        "clean_raw": baseline[metric],
                        "ratio_to_clean_raw": ratio,
                        "classification": _ratio_classification(ratio),
                        "tie_ratio_lower_inclusive": 0.95,
                        "tie_ratio_upper_inclusive": 1.05,
                    }
                )
    return method_rows, pairwise_rows, method_result


def _training_cost(summary: Mapping[str, Any]) -> tuple[int, float]:
    calls = summary.get("training_model_calls")
    wall_time = summary.get("training_wall_time_seconds")
    if isinstance(calls, bool) or not isinstance(calls, int) or calls < 0:
        raise ValueError("training summary lacks nonnegative training_model_calls")
    if (
        isinstance(wall_time, bool)
        or not isinstance(wall_time, (int, float))
        or not math.isfinite(float(wall_time))
        or float(wall_time) < 0.0
    ):
        raise ValueError("training summary lacks finite training_wall_time_seconds")
    return calls, float(wall_time)


def _scientific_cost(cost: Mapping[str, Any]) -> dict[str, Any]:
    if "wall_time_seconds" not in cost:
        raise ValueError("evaluation cost lacks separate timing bookkeeping")
    return {key: value for key, value in cost.items() if key != "wall_time_seconds"}


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write an empty table: {path.name}")
    columns: list[str] = []
    observed: set[str] = set()
    for row in rows:
        for key in row:
            if key not in observed:
                columns.append(key)
                observed.add(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(_json_safe(row) for row in rows)


def _write_deterministic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    if not arrays:
        raise ValueError("cannot write an empty rollout snapshot archive")
    with zipfile.ZipFile(
        path, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for name in sorted(arrays):
            if not name or "/" in name or "\\" in name:
                raise ValueError("snapshot key is not a flat logical name")
            buffer = io.BytesIO()
            np.lib.format.write_array(
                buffer, np.ascontiguousarray(arrays[name]), allow_pickle=False
            )
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            info.create_system = 3
            archive.writestr(
                info, buffer.getvalue(), compress_type=zipfile.ZIP_DEFLATED
            )


def _merge_snapshots(
    destination: dict[str, np.ndarray], incoming: Mapping[str, np.ndarray]
) -> None:
    for key, value in incoming.items():
        if key in destination:
            if not np.array_equal(destination[key], value):
                raise ValueError(f"common snapshot differs across methods: {key}")
        else:
            destination[key] = np.array(value, copy=True)


def _runtime(device: torch.device) -> dict[str, Any]:
    cuda = device.type == "cuda"
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device_name": torch.cuda.get_device_name(device) if cuda else None,
        "device_capability": (
            list(torch.cuda.get_device_capability(device)) if cuda else None
        ),
    }


def _rename_directory_no_replace(source: Path, destination: Path) -> None:
    """Atomically publish a directory while refusing every existing destination."""

    if os.name == "nt":
        os.rename(source, destination)
        return
    if sys.platform.startswith("linux"):
        import ctypes

        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            raise RuntimeError("renameat2 is unavailable for no-clobber publication")
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        result = renameat2(
            -100,
            os.fsencode(source),
            -100,
            os.fsencode(destination),
            1,
        )
        if result != 0:
            error_number = ctypes.get_errno()
            raise OSError(
                error_number, os.strerror(error_number), os.fspath(destination)
            )
        return
    if sys.platform == "darwin":
        import ctypes

        libc = ctypes.CDLL(None, use_errno=True)
        renamex_np = getattr(libc, "renamex_np", None)
        if renamex_np is None:
            raise RuntimeError("renamex_np is unavailable for no-clobber publication")
        renamex_np.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        renamex_np.restype = ctypes.c_int
        result = renamex_np(os.fsencode(source), os.fsencode(destination), 0x00000004)
        if result != 0:
            error_number = ctypes.get_errno()
            raise OSError(
                error_number, os.strerror(error_number), os.fspath(destination)
            )
        return
    raise RuntimeError("atomic no-clobber directory publication is unsupported")


def _make_staging_directory(parent: Path, *, prefix: str) -> Path:
    """Create private POSIX staging or parent-inheriting Windows staging."""

    if os.name != "nt":
        return Path(tempfile.mkdtemp(prefix=prefix, dir=parent))
    for _ in range(128):
        candidate = parent / f"{prefix}{secrets.token_hex(12)}"
        try:
            candidate.mkdir()
        except FileExistsError:
            continue
        return candidate
    raise FileExistsError("could not allocate a unique staging directory")


def _write_staged_packet(output: Path, writer: Callable[[Path], None]) -> None:
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"evaluation output already exists: {output}")
    if output.parent.is_symlink():
        raise ValueError("evaluation output parent is aliased")
    staging: Path | None = None
    owned = False
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.parent.is_symlink() or not output.parent.is_dir():
            raise ValueError("evaluation output parent is absent or aliased")
        staging = _make_staging_directory(
            output.parent, prefix=f".{output.name}.staging-"
        )
        owned = True
        writer(staging)
        _rename_directory_no_replace(staging, output)
        owned = False
    except Exception as error:
        if owned and staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
        if isinstance(error, (FileExistsError, ValueError)):
            raise
        raise EvaluationOutputError("evaluation packet write failed") from error


def _write_final_manifest(
    output: Path,
    *,
    successor: Mapping[str, Any],
    successor_sha256: str,
    authorities: VerifiedAuthoritySet,
    population_role: str,
    prospective_control: VerifiedProspectiveControl | None,
    access_receipt: ProspectiveAccessReceipt | None,
) -> None:
    files = {
        path.name: _file_record(path, output)
        for path in sorted(output.iterdir(), key=lambda item: item.name)
        if path.name != "final_hash_manifest.json"
    }
    if set(files) != OUTPUT_FILES:
        raise ValueError("evaluation output does not have its exact closed file set")
    scientific_files = {name: files[name] for name in sorted(SCIENTIFIC_OUTPUT_FILES)}
    prospective_opened = population_role == "prospective"
    if prospective_opened != (prospective_control is not None) or (
        prospective_opened != (access_receipt is not None)
    ):
        raise ValueError("evaluation final prospective authority differs")
    final: dict[str, Any] = {
        "schema": successor["packet_schemas"]["final_hash_manifest"],
        "experiment_id": EXPERIMENT_ID,
        "successor_contract_sha256": successor_sha256,
        "population_role": population_role,
        "production_authority_set_sha256": authorities.set_sha256,
        "production_authorities": list(authorities.records),
        "scientific_files_sha256": _canonical_sha256(scientific_files),
        "runtime_manifest_excluded_from_scientific_files_sha256": True,
        "files": files,
        "self_hash_excluded": True,
        "prospective_opened": prospective_opened,
        "sealed_opened": False,
    }
    if prospective_control is not None and access_receipt is not None:
        final["prospective_authority"] = {
            "freeze_final_hash_manifest_sha256": prospective_control.freeze_packet.final_file_sha256,
            "freeze_payload_sha256": prospective_control.freeze[
                "canonical_payload_sha256"
            ],
            "authorization_file_sha256": prospective_control.authorization_file_sha256,
            "authorization_payload_sha256": prospective_control.authorization[
                "canonical_payload_sha256"
            ],
            "access_receipt_file_sha256": access_receipt.file_sha256,
            "access_receipt_payload_sha256": access_receipt.value[
                "canonical_payload_sha256"
            ],
        }
    _write_json(output / "final_hash_manifest.json", final)


def _finite_prediction_ratio(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{label} must be a JSON numeric value")
    try:
        ratio = float(value)
    except OverflowError as error:
        raise ValueError(f"{label} exceeds the supported numeric range") from error
    if not math.isfinite(ratio) or ratio < 0.0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return ratio


def _primary_prediction_ratio(value: Any, *, label: str) -> float:
    if value == "Infinity":
        return math.inf
    if isinstance(value, str):
        raise TypeError(f'{label} must be finite or canonical positive "Infinity"')
    ratio = _finite_prediction_ratio(value, label=label)
    if math.isinf(ratio):
        raise ValueError(f"{label} uses noncanonical positive infinity")
    return ratio


def _validate_prediction_matrix(
    value: Any,
    *,
    observed_mediator_availability: bool = False,
) -> None:
    if not isinstance(value, dict) or set(value) != {"pairwise", "mediators"}:
        raise ValueError("prospective prediction matrix fields differ")
    pairwise = value.get("pairwise")
    mediators = value.get("mediators")
    if not isinstance(pairwise, list) or not isinstance(mediators, list):
        raise TypeError("prospective prediction records must be lists")

    expected_pair_keys = [
        (seed, metric, left, right)
        for seed in SEEDS
        for metric in PRIMARY_METRICS
        for left, right in combinations(PRIMARY_DEPLOYMENTS, 2)
    ]
    if len(pairwise) != len(expected_pair_keys):
        raise ValueError("prospective pairwise prediction cardinality differs")
    actual_pair_keys: list[
        tuple[int, str, tuple[str, str, str], tuple[str, str, str]]
    ] = []
    for record in pairwise:
        if not isinstance(record, dict) or set(record) != {
            "seed",
            "metric",
            "left",
            "right",
            "paired_ratio_median",
            "classification",
        }:
            raise ValueError("prospective pairwise prediction fields differ")
        left = record.get("left")
        right = record.get("right")
        if (
            not isinstance(left, list)
            or len(left) != 3
            or not all(isinstance(item, str) for item in left)
            or not isinstance(right, list)
            or len(right) != 3
            or not all(isinstance(item, str) for item in right)
        ):
            raise TypeError("prospective pairwise deployments differ")
        seed = record.get("seed")
        metric = record.get("metric")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("prospective pairwise seed differs")
        if not isinstance(metric, str):
            raise TypeError("prospective pairwise metric differs")
        left_key = (left[0], left[1], left[2])
        right_key = (right[0], right[1], right[2])
        actual_pair_keys.append((seed, metric, left_key, right_key))
        ratio = _primary_prediction_ratio(
            record.get("paired_ratio_median"),
            label="prospective paired ratio",
        )
        expected_classification = {
            "improved": "left_better",
            "tie": "tie",
            "worse": "right_better",
        }[_ratio_classification(ratio)]
        if record.get("classification") != expected_classification:
            raise ValueError("prospective pairwise classification differs")
    if actual_pair_keys != expected_pair_keys or len(set(actual_pair_keys)) != len(
        expected_pair_keys
    ):
        raise ValueError("prospective pairwise key set or ordering differs")

    expected_mediator_keys = [
        (arm, seed, metric)
        for seed in SEEDS
        for arm in MEDIATOR_ARMS
        for metric in MEDIATOR_METRICS
    ]
    if len(mediators) != len(expected_mediator_keys):
        raise ValueError("prospective mediator prediction cardinality differs")
    actual_mediator_keys: list[tuple[str, int, str]] = []
    for record in mediators:
        expected_fields = {
            "arm",
            "seed",
            "metric",
            "ratio_to_clean",
            "relation_to_clean",
        }
        if observed_mediator_availability:
            expected_fields.add("available")
        if not isinstance(record, dict) or set(record) != expected_fields:
            raise ValueError("prospective mediator prediction fields differ")
        arm = record.get("arm")
        seed = record.get("seed")
        metric = record.get("metric")
        if not isinstance(arm, str) or not isinstance(metric, str):
            raise TypeError("prospective mediator arm or metric differs")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("prospective mediator seed differs")
        actual_mediator_keys.append((arm, seed, metric))
        available = record.get("available", True)
        if not isinstance(available, bool):
            raise TypeError("prospective mediator availability differs")
        if not available:
            if (
                not observed_mediator_availability
                or record.get("ratio_to_clean") is not None
                or record.get("relation_to_clean") != "unavailable"
            ):
                raise ValueError("unavailable prospective mediator differs")
            continue
        ratio = _finite_prediction_ratio(
            record.get("ratio_to_clean"), label="prospective mediator ratio"
        )
        expected_relation = "equal"
        if ratio < 1.0:
            expected_relation = "lower"
        elif ratio > 1.0:
            expected_relation = "higher"
        if record.get("relation_to_clean") != expected_relation:
            raise ValueError("prospective mediator relation differs")
    if actual_mediator_keys != expected_mediator_keys or len(
        set(actual_mediator_keys)
    ) != len(expected_mediator_keys):
        raise ValueError("prospective mediator key set or ordering differs")


def _prediction_matrix(
    method_result: Mapping[str, Any],
    offline_result: Mapping[str, Any],
    anchors: Sequence[int],
    *,
    observed_mediator_availability: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    if len(anchors) != 8 or len(set(anchors)) != 8:
        raise ValueError("prediction matrix requires eight distinct anchors")
    pairwise: list[dict[str, Any]] = []
    for seed in SEEDS:
        for metric in PRIMARY_METRICS:
            for left, right in combinations(PRIMARY_DEPLOYMENTS, 2):
                left_key = f"{left[0]}:{seed}:{left[1]}:{left[2]}"
                right_key = f"{right[0]}:{seed}:{right[1]}:{right[2]}"
                left_summary = method_result.get(left_key)
                right_summary = method_result.get(right_key)
                if not isinstance(left_summary, Mapping) or not isinstance(
                    right_summary, Mapping
                ):
                    raise TypeError("prediction matrix lacks a primary deployment")
                ratios = []
                for anchor in anchors:
                    left_anchor = left_summary.get("per_anchor", {}).get(str(anchor))
                    right_anchor = right_summary.get("per_anchor", {}).get(str(anchor))
                    if not isinstance(left_anchor, Mapping) or not isinstance(
                        right_anchor, Mapping
                    ):
                        raise TypeError("prediction matrix lacks a common anchor")
                    ratios.append(
                        _ratio(
                            _numeric(left_anchor.get(metric)),
                            _numeric(right_anchor.get(metric)),
                        )
                    )
                paired_ratio_median = float(
                    np.median(np.asarray(ratios, dtype=np.float64))
                )
                relative = _ratio_classification(paired_ratio_median)
                pairwise.append(
                    {
                        "seed": seed,
                        "metric": metric,
                        "left": list(left),
                        "right": list(right),
                        "paired_ratio_median": _json_number(paired_ratio_median),
                        "classification": {
                            "improved": "left_better",
                            "tie": "tie",
                            "worse": "right_better",
                        }[relative],
                    }
                )

    mediators: list[dict[str, Any]] = []
    for seed in SEEDS:
        clean = offline_result.get(f"CLEAN:{seed}")
        if not isinstance(clean, Mapping):
            raise TypeError("prediction matrix lacks CLEAN offline diagnostics")
        for arm in MEDIATOR_ARMS:
            candidate = offline_result.get(f"{arm}:{seed}")
            if not isinstance(candidate, Mapping):
                raise TypeError("prediction matrix lacks intervention diagnostics")
            for metric in MEDIATOR_METRICS:
                candidate_summary = candidate.get(metric)
                clean_summary = clean.get(metric)
                if not isinstance(candidate_summary, Mapping) or not isinstance(
                    clean_summary, Mapping
                ):
                    raise TypeError("prediction matrix lacks a mediator summary")
                ratio = _ratio(
                    _numeric(candidate_summary.get("median")),
                    _numeric(clean_summary.get("median")),
                )
                if not math.isfinite(ratio):
                    if not observed_mediator_availability:
                        raise ValueError(
                            "mediator prediction ratio must be finite; unavailable "
                            "development diagnostics cannot be frozen"
                        )
                    mediators.append(
                        {
                            "arm": arm,
                            "seed": seed,
                            "metric": metric,
                            "ratio_to_clean": None,
                            "relation_to_clean": "unavailable",
                            "available": False,
                        }
                    )
                    continue
                relation = "equal"
                if ratio < 1.0:
                    relation = "lower"
                elif ratio > 1.0:
                    relation = "higher"
                record = {
                    "arm": arm,
                    "seed": seed,
                    "metric": metric,
                    "ratio_to_clean": _json_number(ratio),
                    "relation_to_clean": relation,
                }
                if observed_mediator_availability:
                    record["available"] = True
                mediators.append(record)
    if len(pairwise) != 60 or len(mediators) != 54:
        raise AssertionError("prospective prediction matrix cardinality differs")
    result = {"pairwise": pairwise, "mediators": mediators}
    _validate_prediction_matrix(
        result,
        observed_mediator_availability=observed_mediator_availability,
    )
    return result


def _prediction_assessment(
    frozen: Mapping[str, Any],
    method_result: Mapping[str, Any],
    offline_result: Mapping[str, Any],
    anchors: Sequence[int],
) -> dict[str, Any]:
    _validate_prediction_matrix(frozen)
    observed = _prediction_matrix(
        method_result,
        offline_result,
        anchors,
        observed_mediator_availability=True,
    )
    frozen_pairwise = frozen.get("pairwise")
    frozen_mediators = frozen.get("mediators")
    if not isinstance(frozen_pairwise, list) or not isinstance(frozen_mediators, list):
        raise TypeError("frozen prospective prediction matrix differs")

    def pair_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
        return (
            int(record["seed"]),
            str(record["metric"]),
            tuple(record["left"]),
            tuple(record["right"]),
        )

    def mediator_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
        return (str(record["arm"]), int(record["seed"]), str(record["metric"]))

    expected_pairs = {pair_key(record): record for record in frozen_pairwise}
    expected_mediators = {mediator_key(record): record for record in frozen_mediators}
    observed_pairs = {pair_key(record): record for record in observed["pairwise"]}
    observed_mediators = {
        mediator_key(record): record for record in observed["mediators"]
    }
    if (
        len(expected_pairs) != 60
        or len(expected_mediators) != 54
        or set(expected_pairs) != set(observed_pairs)
        or set(expected_mediators) != set(observed_mediators)
    ):
        raise ValueError("prospective prediction assessment key set differs")
    pair_records = []
    for key in sorted(expected_pairs, key=str):
        expected = expected_pairs[key]
        actual = observed_pairs[key]
        pair_records.append(
            {
                **actual,
                "predicted_classification": expected["classification"],
                "classification_match": (
                    actual["classification"] == expected["classification"]
                ),
            }
        )
    mediator_records = []
    for key in sorted(expected_mediators, key=str):
        expected = expected_mediators[key]
        actual = observed_mediators[key]
        mediator_records.append(
            {
                **actual,
                "predicted_relation_to_clean": expected["relation_to_clean"],
                "relation_match": (
                    actual["relation_to_clean"] == expected["relation_to_clean"]
                ),
            }
        )
    return {
        "pairwise": pair_records,
        "pairwise_exact_match_count": sum(
            bool(record["classification_match"]) for record in pair_records
        ),
        "pairwise_total": 60,
        "mediators": mediator_records,
        "mediator_exact_match_count": sum(
            bool(record["relation_match"]) for record in mediator_records
        ),
        "mediator_total": 54,
        "automatic_C2_claim_decision_made": False,
    }


def _csv_row_count(packet: ClosedPacket, name: str) -> int:
    _, payload = _verify_record(packet.root, packet.files[name], name)
    rows = list(csv.DictReader(io.StringIO(payload.decode("utf-8"))))
    return len(rows)


def _snapshot_member_count(packet: ClosedPacket) -> int:
    path, _ = _verify_record(
        packet.root,
        packet.files["rollout_snapshots.npz"],
        "rollout_snapshots.npz",
    )
    with zipfile.ZipFile(path, "r") as archive:
        names = archive.namelist()
    if any(not name.endswith(".npy") for name in names):
        raise ValueError("development snapshot archive has a non-array member")
    return len(names)


def freeze_prospective(arguments: argparse.Namespace) -> dict[str, Any]:
    """Close source, predictions, and lineage without protected data access."""

    _validate_output_disjoint(
        arguments.output_dir,
        [
            ("R0 contract", arguments.contract),
            ("successor contract", arguments.successor_contract),
            ("successor preregistration", arguments.preregistration),
            ("development evaluation", arguments.development_evaluation_dir),
        ],
        label="prospective freeze output",
    )
    source = _source_snapshot()
    _reverify_source(source)
    successor, successor_sha256, baseline = _load_successor_contract(
        arguments.successor_contract,
        arguments.contract,
        arguments.preregistration,
    )
    development = _verify_closed_packet(
        arguments.development_evaluation_dir,
        final_schema=successor["packet_schemas"]["final_hash_manifest"],
        expected_files=OUTPUT_FILES,
        label="closed development evaluation",
    )
    if (
        development.final.get("experiment_id") != EXPERIMENT_ID
        or development.final.get("successor_contract_sha256") != successor_sha256
        or development.final.get("population_role") != "development"
        or development.final.get("prospective_opened") is not False
        or development.final.get("sealed_opened") is not False
    ):
        raise ValueError("closed development evaluation identity differs")
    scientific_files = {
        name: development.files[name] for name in sorted(SCIENTIFIC_OUTPUT_FILES)
    }
    if development.final.get("scientific_files_sha256") != _canonical_sha256(
        scientific_files
    ):
        raise ValueError("closed development scientific digest differs")
    input_manifest = _packet_json(
        development,
        "input_manifest.json",
        schema=successor["packet_schemas"]["evaluation_inputs"],
        label="development input manifest",
    )
    result = _packet_json(
        development,
        "result.json",
        schema=successor["packet_schemas"]["evaluation"],
        label="development result",
    )
    development_source = _packet_json(
        development,
        "source_manifest.json",
        schema=SOURCE_MANIFEST_SCHEMA,
        label="development source manifest",
    )
    if (
        input_manifest.get("population_role") != "development"
        or input_manifest.get("anchors")
        != list(successor["evaluation"]["development_anchors"])
        or input_manifest.get("prospective_opened") is not False
        or input_manifest.get("sealed_opened") is not False
        or input_manifest.get("evaluator_source_set_sha256")
        != development_source.get("source_set_sha256")
        or result.get("status") != "complete"
        or result.get("population_role") != "development"
        or result.get("successor_contract_sha256") != successor_sha256
        or result.get("input_manifest_payload_sha256")
        != input_manifest.get("canonical_payload_sha256")
        or result.get("claim_boundary", {}).get("prospective_opened") is not False
        or result.get("claim_boundary", {}).get("sealed_opened") is not False
    ):
        raise ValueError("development result lineage differs")
    observed_cardinalities = {
        name: _csv_row_count(development, name)
        for name in EXPECTED_RESULT_CARDINALITIES
        if name.endswith(".csv")
    }
    observed_cardinalities["rollout_snapshots.npz"] = _snapshot_member_count(
        development
    )
    if observed_cardinalities != EXPECTED_RESULT_CARDINALITIES:
        raise ValueError("development result cardinalities differ")

    prospective_role = baseline["phase_population"]["roles"]["prospective"]
    trajectory = baseline["parent_evidence"]["trajectory"]
    if (
        prospective_role.get("owned_frame_indices_inclusive")
        != list(PROSPECTIVE_FRAMES)
        or prospective_role.get("anchor_input_indices") != list(PROSPECTIVE_ANCHORS)
        or any(
            anchor - 1 < PROSPECTIVE_FRAMES[0]
            or anchor + FULL_TRACE[1] > PROSPECTIVE_FRAMES[1]
            for anchor in PROSPECTIVE_ANCHORS
        )
    ):
        raise ValueError("baseline prospective role differs")
    training_packets = input_manifest.get("training_packets")
    if not isinstance(training_packets, list) or len(training_packets) != 12:
        raise ValueError("development training packet inventory differs")
    training_source_sets = {
        packet.get("source_set_sha256")
        for packet in training_packets
        if isinstance(packet, Mapping)
    }
    if len(training_source_sets) != 1:
        raise ValueError("training source-set inventory differs")
    method_result = result.get("method_summaries")
    offline_result = result.get("offline_diagnostics")
    if not isinstance(method_result, Mapping) or not isinstance(
        offline_result, Mapping
    ):
        raise TypeError("development result lacks prediction inputs")
    predictions = _prediction_matrix(
        method_result,
        offline_result,
        successor["evaluation"]["development_anchors"],
    )
    parent_bindings = {
        "inherited_r0_contract_sha256": successor["inherited_r0_contract_sha256"],
        "preregistration_sha256": successor["preregistration_sha256"],
        "dataset_final_hash_manifest_sha256": input_manifest[
            "dataset_final_hash_manifest_sha256"
        ],
        "dataset_manifest_payload_sha256": input_manifest[
            "dataset_manifest_payload_sha256"
        ],
        "calibration_final_hash_manifest_sha256": input_manifest[
            "calibration_final_hash_manifest_sha256"
        ],
        "calibration_payload_sha256": input_manifest["calibration_payload_sha256"],
        "development_evaluation_final_hash_manifest_sha256": development.final_file_sha256,
        "development_scientific_files_sha256": development.final[
            "scientific_files_sha256"
        ],
        "development_input_manifest_payload_sha256": input_manifest[
            "canonical_payload_sha256"
        ],
        "development_result_payload_sha256": result["canonical_payload_sha256"],
        "development_evaluator_source_set_sha256": development_source[
            "source_set_sha256"
        ],
        "production_authority_set_sha256": input_manifest[
            "production_authority_set_sha256"
        ],
        "training_source_set_sha256": next(iter(training_source_sets)),
        "training_packets": training_packets,
        "trajectory_receipt_file_sha256": trajectory["trajectory_receipt_file_sha256"],
        "trajectory_receipt_payload_sha256": trajectory[
            "trajectory_receipt_payload_sha256"
        ],
        "trajectory_storage_manifest_file_sha256": trajectory[
            "storage_manifest_file_sha256"
        ],
        "trajectory_storage_manifest_payload_sha256": trajectory[
            "storage_manifest_payload_sha256"
        ],
        "ordered_restart_records_sha256": trajectory["ordered_restart_records_sha256"],
    }
    freeze = _self_hashed(
        {
            "schema": successor["packet_schemas"]["prospective_freeze"],
            "status": "complete",
            "experiment_id": EXPERIMENT_ID,
            "successor_contract_sha256": successor_sha256,
            "population_role": "prospective",
            "prospective_frames_inclusive": list(PROSPECTIVE_FRAMES),
            "anchors": list(PROSPECTIVE_ANCHORS),
            "seeds": list(SEEDS),
            "primary_deployments": [list(item) for item in PRIMARY_DEPLOYMENTS],
            "identity_parity_deployment": ["CLEAN", "identity", "corrected"],
            "intervention_magnitude_deployment": [
                "PATH_PROJECTION",
                "path_projection",
                "raw_pre_correction",
            ],
            "primary_metrics": list(PRIMARY_METRICS),
            "mediator_metrics": list(MEDIATOR_METRICS),
            "horizons": list(HORIZONS),
            "full_trace_inclusive": list(FULL_TRACE),
            "late_window_inclusive": list(LATE_WINDOW),
            "tie_ratio_inclusive": [0.95, 1.05],
            "aggregation": "paired_per_anchor_ratio_then_seedwise_median",
            "sampling": "deterministic_finite_population_common_trajectory",
            "expected_result_cardinalities": EXPECTED_RESULT_CARDINALITIES,
            "parent_bindings": parent_bindings,
            "evaluator_source_set_sha256": source["source_set_sha256"],
            "predictions": predictions,
            "claim_boundary": FREEZE_CLAIM_BOUNDARY,
            "prospective_opened": False,
            "sealed_opened": False,
        }
    )
    _validate_prospective_freeze_semantics(freeze, source)

    def verify_packet(root: Path) -> ClosedPacket:
        closed = _verify_closed_packet(
            root,
            final_schema=successor["packet_schemas"]["final_hash_manifest"],
            expected_files=FREEZE_OUTPUT_FILES,
            label="prospective freeze packet",
        )
        if (
            set(closed.final)
            != {
                "schema",
                "experiment_id",
                "packet_role",
                "successor_contract_sha256",
                "population_role",
                "files",
                "self_hash_excluded",
                "prospective_opened",
                "sealed_opened",
            }
            or closed.final.get("experiment_id") != EXPERIMENT_ID
            or closed.final.get("packet_role") != "prospective_freeze"
            or closed.final.get("successor_contract_sha256") != successor_sha256
            or closed.final.get("population_role") != "prospective"
            or closed.final.get("prospective_opened") is not False
            or closed.final.get("sealed_opened") is not False
        ):
            raise EvaluationOutputError("closed prospective freeze differs")
        closed_freeze = _packet_json(
            closed,
            "prospective_freeze.json",
            schema=successor["packet_schemas"]["prospective_freeze"],
            label="prospective freeze payload",
        )
        closed_source = _packet_json(
            closed,
            "source_manifest.json",
            schema=SOURCE_MANIFEST_SCHEMA,
            label="prospective freeze source manifest",
        )
        if closed_freeze != freeze or closed_source != source:
            raise EvaluationOutputError("closed prospective freeze payload differs")
        _validate_prospective_freeze_semantics(closed_freeze, closed_source)
        _reverify_packet(closed)
        return closed

    def write_packet(staging: Path) -> None:
        _write_json(staging / "prospective_freeze.json", freeze)
        _write_json(staging / "source_manifest.json", source)
        files = {
            path.name: _file_record(path, staging)
            for path in sorted(staging.iterdir(), key=lambda item: item.name)
        }
        if set(files) != FREEZE_OUTPUT_FILES:
            raise ValueError("prospective freeze output file set differs")
        _write_json(
            staging / "final_hash_manifest.json",
            {
                "schema": successor["packet_schemas"]["final_hash_manifest"],
                "experiment_id": EXPERIMENT_ID,
                "packet_role": "prospective_freeze",
                "successor_contract_sha256": successor_sha256,
                "population_role": "prospective",
                "files": files,
                "self_hash_excluded": True,
                "prospective_opened": False,
                "sealed_opened": False,
            },
        )
        _reverify_source(source)
        verify_packet(staging)

    _write_staged_packet(arguments.output_dir.resolve(), write_packet)
    verify_packet(arguments.output_dir)
    _reverify_source(source)
    return freeze


def _write_prospective_access_receipt(
    path: Path, control: VerifiedProspectiveControl
) -> ProspectiveAccessReceipt:
    """Persist a conservative opening record before the first protected read."""

    _reverify_prospective_control(control)
    expected = _absolute_path(_prospective_access_receipt_path(control))
    if _absolute_path(path) != expected:
        raise ProtectedPopulationError(
            "prospective access receipt path differs from its frozen deterministic path"
        )
    parent = _safe_directory(expected.parent, label="prospective access receipt parent")
    resolved = parent / expected.name
    value = _self_hashed(
        {
            "schema": "time_dependent_no.naca_corrective_successor_access_receipt.v1",
            "status": "prospective_access_authority_consumed",
            "experiment_id": EXPERIMENT_ID,
            "successor_contract_sha256": control.freeze["successor_contract_sha256"],
            "prospective_freeze_final_hash_manifest_sha256": control.freeze_packet.final_file_sha256,
            "prospective_freeze_payload_sha256": control.freeze[
                "canonical_payload_sha256"
            ],
            "prospective_freeze_root_sha256": _canonical_path_sha256(
                control.freeze_packet.root
            ),
            "prospective_authorization_file_sha256": control.authorization_file_sha256,
            "prospective_authorization_payload_sha256": control.authorization[
                "canonical_payload_sha256"
            ],
            "receipt_binding_sha256": _canonical_sha256(
                {
                    "prospective_freeze_final_hash_manifest_sha256": (
                        control.freeze_packet.final_file_sha256
                    ),
                    "prospective_authorization_payload_sha256": (
                        control.authorization["canonical_payload_sha256"]
                    ),
                }
            ),
            "evaluator_source_set_sha256": control.source["source_set_sha256"],
            "population_role": "prospective",
            "prospective_frames_inclusive": list(PROSPECTIVE_FRAMES),
            "access_recorded_before_first_scientific_read": True,
            "prospective_opened": True,
            "sealed_opened": False,
        }
    )
    payload = (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    descriptor: int | None = None
    claimed_identity: tuple[int, int] | None = None
    try:
        descriptor = os.open(resolved, flags, 0o600)
    except FileExistsError as error:
        raise ProtectedPopulationError(
            "prospective access receipt already exists; authorization was consumed"
        ) from error
    try:
        claimed = os.fstat(descriptor)
        claimed_identity = (claimed.st_dev, claimed.st_ino)
        stream = os.fdopen(descriptor, "wb")
        descriptor = None
        with stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        loaded, file_sha256 = _read_self_hashed_json(
            resolved,
            schema="time_dependent_no.naca_corrective_successor_access_receipt.v1",
            label="prospective access receipt",
        )
        if loaded != value:
            raise ValueError("prospective access receipt changed after writing")
    except Exception:
        if descriptor is not None:
            os.close(descriptor)
        try:
            current = os.stat(resolved, follow_symlinks=False)
            if claimed_identity == (current.st_dev, current.st_ino):
                resolved.unlink()
        except FileNotFoundError:
            pass
        raise
    return ProspectiveAccessReceipt(
        path=resolved,
        value=value,
        file_sha256=file_sha256,
    )


def _reverify_prospective_access_receipt(
    receipt: ProspectiveAccessReceipt,
) -> None:
    value, file_sha256 = _read_self_hashed_json(
        receipt.path,
        schema="time_dependent_no.naca_corrective_successor_access_receipt.v1",
        label="prospective access receipt",
    )
    if value != receipt.value or file_sha256 != receipt.file_sha256:
        raise ProtectedPopulationError("prospective access receipt changed")


def _load_authorized_prospective_population(
    *,
    trajectory_dir: Path,
    storage_manifest_path: Path,
    baseline: Mapping[str, Any],
    geometry: VerifiedNACAGeometry,
    control: VerifiedProspectiveControl,
    receipt: ProspectiveAccessReceipt,
) -> ProspectivePopulation:
    """Read the exact prospective frame block after reveal authority is consumed."""

    _reverify_prospective_control(control)
    _reverify_prospective_access_receipt(receipt)
    root = _safe_directory(trajectory_dir, label="authorized trajectory directory")
    manifest_path = _safe_regular_file(
        storage_manifest_path, label="authorized trajectory storage manifest"
    )
    if manifest_path.parent != root:
        raise ValueError("trajectory storage manifest is outside its trajectory root")
    parent = baseline["parent_evidence"]["trajectory"]
    if _file_sha256(manifest_path) != parent["storage_manifest_file_sha256"]:
        raise ValueError("trajectory storage manifest file hash differs")
    storage = _load_verified_storage_manifest(manifest_path)
    if (
        storage.get("status") != "validated"
        or storage.get("canonical_payload_sha256")
        != parent["storage_manifest_payload_sha256"]
        or storage.get("output_contract", {}).get("first_index") != 499
        or storage.get("output_contract", {}).get("last_index") != 1999
        or storage.get("output_contract", {}).get("count") != 1501
        or storage.get("aggregate", {}).get("ordered_file_records_sha256")
        != parent["ordered_restart_records_sha256"]
    ):
        raise ValueError("trajectory storage manifest lineage differs")
    files = storage.get("files")
    if not isinstance(files, list) or len(files) != 1501:
        raise ValueError("trajectory storage manifest file inventory differs")
    by_index: dict[int, Mapping[str, Any]] = {}
    for record in files:
        if not isinstance(record, Mapping):
            raise TypeError("trajectory storage record is invalid")
        index = record.get("index")
        if isinstance(index, bool) or not isinstance(index, int) or index in by_index:
            raise ValueError("trajectory storage record index differs")
        by_index[index] = record
    if set(by_index) != set(range(499, 2000)):
        raise ValueError("trajectory storage record index set differs")

    frame_indices = np.arange(
        PROSPECTIVE_FRAMES[0], PROSPECTIVE_FRAMES[1] + 1, dtype=np.int64
    )
    states: list[np.ndarray] = []
    subset_records: list[dict[str, Any]] = []
    for frame_index in frame_indices:
        index = int(frame_index)
        record = by_index[index]
        expected_name = f"trajectory_flow_{index:05d}.dat"
        if record.get("file") != expected_name:
            raise ValueError("prospective trajectory filename differs")
        digest = _require_digest(record.get("sha256"), "prospective restart")
        byte_count = record.get("bytes")
        if isinstance(byte_count, bool) or not isinstance(byte_count, int):
            raise TypeError("prospective restart byte count differs")
        path = _naca_safe_child(root, expected_name, "prospective restart")
        path = _safe_regular_file(path, label="prospective restart")
        if path.stat().st_size != byte_count or _file_sha256(path) != digest:
            raise ValueError("prospective restart differs from storage manifest")
        state = _extract_native_state(
            path,
            expected_sha256=digest,
            expected_num_points=geometry.num_nodes,
            expected_coordinates=geometry.native_coordinates,
        )
        states.append(state)
        subset_records.append(
            {
                "index": index,
                "file": expected_name,
                "bytes": byte_count,
                "sha256": digest,
            }
        )
    state_array = np.stack(states, axis=0)
    if (
        state_array.dtype != np.float64
        or state_array.shape != (240, geometry.num_nodes, len(FIELDS))
        or not np.all(np.isfinite(state_array))
        or frame_indices.tolist()
        != list(range(PROSPECTIVE_FRAMES[0], PROSPECTIVE_FRAMES[1] + 1))
    ):
        raise ValueError("authorized prospective population differs")
    provenance = {
        "trajectory_storage_manifest_file_sha256": parent[
            "storage_manifest_file_sha256"
        ],
        "trajectory_storage_manifest_payload_sha256": parent[
            "storage_manifest_payload_sha256"
        ],
        "parent_ordered_restart_records_sha256": parent[
            "ordered_restart_records_sha256"
        ],
        "prospective_subset_records_sha256": _canonical_sha256(
            {"files": subset_records}
        ),
        "prospective_state_bytes_sha256": sha256(
            np.ascontiguousarray(state_array).tobytes()
        ).hexdigest(),
        "prospective_frame_indices_bytes_sha256": sha256(
            np.ascontiguousarray(frame_indices).tobytes()
        ).hexdigest(),
        "state_dtype": str(state_array.dtype),
        "state_shape": list(state_array.shape),
        "frame_indices_dtype": str(frame_indices.dtype),
        "prospective_frames_inclusive": list(PROSPECTIVE_FRAMES),
        "prospective_opened": True,
        "sealed_opened": False,
    }
    if _file_sha256(manifest_path) != parent["storage_manifest_file_sha256"]:
        raise ValueError("trajectory storage manifest changed during access")
    _reverify_prospective_control(control)
    _reverify_prospective_access_receipt(receipt)
    return ProspectivePopulation(state_array, frame_indices, provenance)


def _offline_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    keys = sorted({(str(row["arm"]), int(row["seed"])) for row in rows})
    for arm, seed in keys:
        selected = [
            row for row in rows if row["arm"] == arm and int(row["seed"]) == seed
        ]
        if len(selected) != 8:
            raise ValueError("offline diagnostics do not contain eight common anchors")
        result[f"{arm}:{seed}"] = {
            "clean_forcing_normalized_rms": _aggregate(
                [float(row["clean_forcing_normalized_rms"]) for row in selected]
            ),
            "one_prefix_response_normalized_rms": _aggregate(
                [float(row["one_prefix_response_normalized_rms"]) for row in selected]
            ),
            "one_prefix_response_gain": _aggregate(
                [float(row["one_prefix_response_gain"]) for row in selected]
            ),
            "one_prefix_next_normalized_rms": _aggregate(
                [float(row["one_prefix_next_normalized_rms"]) for row in selected]
            ),
            "one_prefix_response_transverse_rms": _aggregate_valid_directions(
                [float(row["one_prefix_response_transverse_rms"]) for row in selected]
            ),
            "one_prefix_response_graph_dirichlet_energy": _aggregate(
                [
                    float(row["one_prefix_response_graph_dirichlet_energy"])
                    for row in selected
                ]
            ),
            "stable_tangent_anchor_count": sum(
                bool(row["one_prefix_response_tangent_direction_valid"])
                for row in selected
            ),
        }
    return result


def _validate_population(
    states: np.ndarray,
    indices: np.ndarray,
    *,
    role: str,
    expected_frames: Sequence[int],
    node_count: int,
) -> None:
    if (
        states.dtype != np.float64
        or states.shape != (len(expected_frames), node_count, 5)
        or indices.dtype != np.int64
        or indices.tolist() != list(expected_frames)
        or not np.all(np.isfinite(states))
    ):
        raise ValueError(f"{role} population shape, indices, dtype, or values differ")


def _verify_prospective_parent_bindings(
    *,
    control: VerifiedProspectiveControl,
    successor: Mapping[str, Any],
    successor_sha256: str,
    baseline: Mapping[str, Any],
    dataset_manifest: Mapping[str, Any],
    dataset_sha256: str,
    calibration_packet: ClosedPacket,
    calibration: Mapping[str, Any],
    training_packets: Sequence[TrainingPacket],
    authorities: VerifiedAuthoritySet,
    development_evaluation_dir: Path,
) -> ClosedPacket:
    bindings = control.freeze["parent_bindings"]
    expected_training = [
        {
            "arm": packet.arm,
            "seed": packet.seed,
            "final_hash_manifest_sha256": packet.packet.final_file_sha256,
            "checkpoint_sha256": packet.checkpoint_record["sha256"],
            "source_set_sha256": packet.summary["source_set_sha256"],
        }
        for packet in training_packets
    ]
    training_source_sets = {
        packet.summary["source_set_sha256"] for packet in training_packets
    }
    trajectory = baseline["parent_evidence"]["trajectory"]
    checks = {
        "inherited_r0_contract_sha256": successor["inherited_r0_contract_sha256"],
        "preregistration_sha256": successor["preregistration_sha256"],
        "dataset_final_hash_manifest_sha256": dataset_sha256,
        "dataset_manifest_payload_sha256": dataset_manifest["canonical_payload_sha256"],
        "calibration_final_hash_manifest_sha256": calibration_packet.final_file_sha256,
        "calibration_payload_sha256": calibration["canonical_payload_sha256"],
        "production_authority_set_sha256": authorities.set_sha256,
        "training_packets": expected_training,
        "training_source_set_sha256": (
            next(iter(training_source_sets)) if len(training_source_sets) == 1 else None
        ),
        "trajectory_receipt_file_sha256": trajectory["trajectory_receipt_file_sha256"],
        "trajectory_receipt_payload_sha256": trajectory[
            "trajectory_receipt_payload_sha256"
        ],
        "trajectory_storage_manifest_file_sha256": trajectory[
            "storage_manifest_file_sha256"
        ],
        "trajectory_storage_manifest_payload_sha256": trajectory[
            "storage_manifest_payload_sha256"
        ],
        "ordered_restart_records_sha256": trajectory["ordered_restart_records_sha256"],
    }
    if control.freeze.get("successor_contract_sha256") != successor_sha256 or any(
        bindings.get(key) != value for key, value in checks.items()
    ):
        raise ProtectedPopulationError("prospective frozen parent binding differs")
    development = _verify_closed_packet(
        development_evaluation_dir,
        final_schema=successor["packet_schemas"]["final_hash_manifest"],
        expected_files=OUTPUT_FILES,
        label="frozen development evaluation parent",
    )
    if (
        development.final_file_sha256
        != bindings["development_evaluation_final_hash_manifest_sha256"]
        or development.final.get("scientific_files_sha256")
        != bindings["development_scientific_files_sha256"]
        or development.final.get("population_role") != "development"
        or development.final.get("prospective_opened") is not False
        or development.final.get("sealed_opened") is not False
    ):
        raise ProtectedPopulationError("development parent differs from freeze")
    development_input = _packet_json(
        development,
        "input_manifest.json",
        schema=successor["packet_schemas"]["evaluation_inputs"],
        label="frozen development input manifest",
    )
    development_result = _packet_json(
        development,
        "result.json",
        schema=successor["packet_schemas"]["evaluation"],
        label="frozen development result",
    )
    development_source = _packet_json(
        development,
        "source_manifest.json",
        schema=SOURCE_MANIFEST_SCHEMA,
        label="frozen development source manifest",
    )
    if (
        development_input.get("canonical_payload_sha256")
        != bindings["development_input_manifest_payload_sha256"]
        or development_result.get("canonical_payload_sha256")
        != bindings["development_result_payload_sha256"]
        or development_source.get("source_set_sha256")
        != bindings["development_evaluator_source_set_sha256"]
    ):
        raise ProtectedPopulationError("development payload differs from freeze")
    _reverify_prospective_control(control)
    return development


def evaluate(arguments: argparse.Namespace) -> dict[str, Any]:
    mode = arguments.mode
    prospective_control: VerifiedProspectiveControl | None = None
    if mode == "prospective":
        freeze_root = getattr(arguments, "prospective_freeze_dir", None)
        authorization_path = getattr(arguments, "prospective_authorization", None)
        if freeze_root is None or authorization_path is None:
            raise ProtectedPopulationError(
                "prospective freeze and authorization are required; "
                "no scientific input path was inspected"
            )
        prospective_control = _load_prospective_control(freeze_root, authorization_path)
        _validate_prospective_reveal_paths(arguments, prospective_control)
    elif mode == "development":
        development_inputs = [
            ("R0 contract", arguments.contract),
            ("successor contract", arguments.successor_contract),
            ("successor preregistration", arguments.preregistration),
            ("dataset packet", arguments.dataset_dir),
            ("calibration packet", arguments.calibration_dir),
        ]
        development_inputs.extend(
            ("training packet", path) for path in arguments.training_dir
        )
        development_inputs.extend(
            ("resource smoke receipt", path)
            for path in arguments.resource_smoke_receipt
        )
        development_inputs.extend(
            ("execution authorization", path)
            for path in arguments.execution_authorization
        )
        _validate_output_disjoint(
            arguments.output_dir,
            development_inputs,
            label="development evaluation output",
        )
    else:
        raise ValueError("evaluation mode must be development or prospective")
    source = _source_snapshot()
    _reverify_source(source)
    output = arguments.output_dir.resolve()
    if arguments.output_dir.is_symlink() or arguments.output_dir.parent.is_symlink():
        raise ValueError("evaluation output or its parent is aliased")
    if output.exists():
        raise FileExistsError(f"evaluation output already exists: {output}")
    successor, successor_sha256, baseline = _load_successor_contract(
        arguments.successor_contract,
        arguments.contract,
        arguments.preregistration,
    )
    dataset_root = _safe_directory(arguments.dataset_dir, label="open dataset packet")
    (
        dataset_manifest,
        geometry,
        normalization,
        roles,
        dataset_sha256,
    ) = _load_dataset(dataset_root, baseline)
    expected_dataset_sha256 = successor["dataset"]["final_hash_manifest_sha256"]
    if dataset_sha256 != expected_dataset_sha256:
        raise ValueError("open dataset packet differs from the successor contract")
    train_states, train_indices, _ = roles["train"]
    development_states, development_indices, _ = roles["development"]
    _validate_population(
        train_states,
        train_indices,
        role="train",
        expected_frames=range(955, 1195),
        node_count=geometry.num_nodes,
    )
    _validate_population(
        development_states,
        development_indices,
        role="development",
        expected_frames=range(1233, 1473),
        node_count=geometry.num_nodes,
    )
    calibration_packet, calibration, projector = _load_calibration(
        arguments.calibration_dir,
        successor=successor,
        successor_sha256=successor_sha256,
        dataset_sha256=dataset_sha256,
        normalization=normalization,
        train_states=train_states,
        train_indices=train_indices,
    )
    training_packets = _load_training_packets(
        arguments.training_dir,
        successor=successor,
        successor_sha256=successor_sha256,
        dataset_sha256=dataset_sha256,
        dataset_payload_sha256=dataset_manifest["canonical_payload_sha256"],
        calibration_sha256=calibration_packet.final_file_sha256,
        calibration_payload_sha256=calibration["canonical_payload_sha256"],
        baseline=baseline,
        geometry=geometry,
    )
    authorities = _verify_production_authorities(
        smoke_paths=arguments.resource_smoke_receipt,
        authorization_paths=arguments.execution_authorization,
        training_packets=training_packets,
        successor=successor,
        successor_sha256=successor_sha256,
        dataset_sha256=dataset_sha256,
        calibration_sha256=calibration_packet.final_file_sha256,
        geometry=geometry,
    )
    device = torch.device(arguments.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    access_receipt: ProspectiveAccessReceipt | None = None
    prospective_provenance: dict[str, Any] | None = None
    development_parent: ClosedPacket | None = None
    if prospective_control is None:
        population_role = "development"
        population_states = development_states
        population_indices = development_indices
        anchors = tuple(successor["evaluation"]["development_anchors"])
    else:
        for name in (
            "development_evaluation_dir",
            "trajectory_dir",
            "trajectory_storage_manifest",
            "prospective_access_receipt",
        ):
            if getattr(arguments, name, None) is None:
                raise ValueError(
                    f"prospective evaluation requires --{name.replace('_', '-')}"
                )
        development_parent = _verify_prospective_parent_bindings(
            control=prospective_control,
            successor=successor,
            successor_sha256=successor_sha256,
            baseline=baseline,
            dataset_manifest=dataset_manifest,
            dataset_sha256=dataset_sha256,
            calibration_packet=calibration_packet,
            calibration=calibration,
            training_packets=training_packets,
            authorities=authorities,
            development_evaluation_dir=arguments.development_evaluation_dir,
        )
        access_receipt = _write_prospective_access_receipt(
            arguments.prospective_access_receipt, prospective_control
        )
        prospective_population = _load_authorized_prospective_population(
            trajectory_dir=arguments.trajectory_dir,
            storage_manifest_path=arguments.trajectory_storage_manifest,
            baseline=baseline,
            geometry=geometry,
            control=prospective_control,
            receipt=access_receipt,
        )
        population_role = "prospective"
        population_states = prospective_population.states
        population_indices = prospective_population.indices
        prospective_provenance = prospective_population.provenance
        anchors = PROSPECTIVE_ANCHORS
        _validate_population(
            population_states,
            population_indices,
            role="prospective",
            expected_frames=range(PROSPECTIVE_FRAMES[0], PROSPECTIVE_FRAMES[1] + 1),
            node_count=geometry.num_nodes,
        )
    if len(anchors) != 8:
        raise ValueError("evaluated role does not contain exactly eight common anchors")
    positions = {int(index): offset for offset, index in enumerate(population_indices)}
    if any(
        required not in positions
        for anchor in anchors
        for required in (anchor - 1, anchor, anchor + FULL_TRACE[1])
    ):
        raise ValueError("an evaluation anchor violates role-contained recurrence")

    weights = _view_weights(geometry)
    graph_edges = _graph_edges(geometry)
    structure_context = _structure_context(geometry, train_states)
    reference_cache = _reference_cache(population_states, population_indices, projector)
    offline_rows: list[dict[str, Any]] = []
    rollout_rows: list[dict[str, Any]] = []
    structure_rows: list[dict[str, Any]] = []
    snapshots: dict[str, np.ndarray] = {}
    cost_rows: list[dict[str, Any]] = []
    rollout_bookkeeping: dict[str, Any] = {}
    timing_bookkeeping: dict[str, Any] = {}
    total_started = time.perf_counter()
    for packet in training_packets:
        model = _instantiate_model(
            packet, baseline=baseline, geometry=geometry, device=device
        )
        packet_offline, offline_cost = _offline_diagnostics(
            packet=packet,
            model=model,
            states=population_states,
            indices=population_indices,
            anchors=anchors,
            geometry=geometry,
            normalization=normalization,
            projector=projector,
            reference_cache=reference_cache,
            graph_edges=graph_edges,
            weights=weights,
            structure_context=structure_context,
            device=device,
        )
        packet_rollout, packet_structure, packet_snapshots, raw_cost = _raw_rollout(
            packet=packet,
            model=model,
            states=population_states,
            indices=population_indices,
            anchors=anchors,
            geometry=geometry,
            normalization=normalization,
            projector=projector,
            reference_cache=reference_cache,
            graph_edges=graph_edges,
            weights=weights,
            structure_context=structure_context,
            device=device,
        )
        offline_rows.extend(packet_offline)
        rollout_rows.extend(packet_rollout)
        structure_rows.extend(packet_structure)
        _merge_snapshots(snapshots, packet_snapshots)
        training_calls, training_wall = _training_cost(packet.summary)
        raw_key = f"{packet.arm}:{packet.seed}:raw"
        rollout_bookkeeping[raw_key] = _scientific_cost(raw_cost)
        timing_bookkeeping[raw_key] = {
            "predictor_training_wall_time_seconds": training_wall,
            "offline_diagnostic_wall_time_seconds": offline_cost["wall_time_seconds"],
            "rollout_wall_time_seconds": raw_cost["wall_time_seconds"],
        }
        cost_rows.append(
            {
                "arm": packet.arm,
                "seed": packet.seed,
                "deployment": "raw",
                "predictor_training_model_calls": training_calls,
                "incremental_corrector_training_model_calls": 0,
                "diagnostic_model_invocations": offline_cost["model_invocations"],
                "diagnostic_state_transition_evaluations": offline_cost[
                    "state_transition_evaluations"
                ],
                "rollout_model_invocations": raw_cost["model_invocations"],
                "rollout_state_transition_evaluations": raw_cost[
                    "state_transition_evaluations"
                ],
                "corrector_calls": 0,
                "stored_reference_bank_bytes": 0,
                "online_solver_calls": 0,
            }
        )
        if packet.arm == "CLEAN":
            cost_rows.append(
                {
                    **cost_rows[-1],
                    "deployment": "identity",
                    "corrector_calls": raw_cost["corrector_calls"],
                }
            )
            timing_bookkeeping[f"CLEAN:{packet.seed}:identity"] = dict(
                timing_bookkeeping[raw_key]
            )
            path_rows, path_structure, path_snapshots, path_cost = (
                _path_projection_rollout(
                    packet=packet,
                    model=model,
                    states=population_states,
                    indices=population_indices,
                    anchors=anchors,
                    geometry=geometry,
                    normalization=normalization,
                    projector=projector,
                    reference_cache=reference_cache,
                    graph_edges=graph_edges,
                    weights=weights,
                    structure_context=structure_context,
                    device=device,
                )
            )
            rollout_rows.extend(path_rows)
            structure_rows.extend(path_structure)
            _merge_snapshots(snapshots, path_snapshots)
            path_key = f"PATH_PROJECTION:{packet.seed}:path_projection"
            rollout_bookkeeping[path_key] = _scientific_cost(path_cost)
            timing_bookkeeping[path_key] = {
                "predictor_training_wall_time_seconds": training_wall,
                "offline_diagnostic_wall_time_seconds": offline_cost[
                    "wall_time_seconds"
                ],
                "rollout_wall_time_seconds": path_cost["wall_time_seconds"],
            }
            cost_rows.append(
                {
                    "arm": "PATH_PROJECTION",
                    "parent_arm": "CLEAN",
                    "seed": packet.seed,
                    "deployment": "path_projection",
                    "predictor_training_model_calls": training_calls,
                    "incremental_corrector_training_model_calls": 0,
                    "diagnostic_model_invocations": offline_cost["model_invocations"],
                    "diagnostic_state_transition_evaluations": offline_cost[
                        "state_transition_evaluations"
                    ],
                    "rollout_model_invocations": path_cost["model_invocations"],
                    "rollout_state_transition_evaluations": path_cost[
                        "state_transition_evaluations"
                    ],
                    "corrector_calls": path_cost["corrector_calls"],
                    "stored_reference_bank_bytes": _reference_bank_bytes(projector),
                    "online_solver_calls": 0,
                }
            )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        _reverify_packet(packet.packet)
    total_wall_time = time.perf_counter() - total_started

    method_rows, pairwise_rows, method_result = _summaries(rollout_rows, anchors)
    offline_result = _offline_summary(offline_rows)
    runtime_manifest = _self_hashed(
        {
            "schema": RUNTIME_MANIFEST_SCHEMA,
            "experiment_id": EXPERIMENT_ID,
            "population_role": population_role,
            "runtime": _runtime(device),
            "evaluation_total_wall_time_seconds": total_wall_time,
            "timing_bookkeeping": timing_bookkeeping,
            "excluded_from_scientific_payload_digest": True,
        }
    )
    input_manifest = _self_hashed(
        {
            "schema": successor["packet_schemas"]["evaluation_inputs"],
            "experiment_id": EXPERIMENT_ID,
            "successor_contract_sha256": successor_sha256,
            "preregistration_sha256": successor["preregistration_sha256"],
            "inherited_r0_contract_sha256": successor["inherited_r0_contract_sha256"],
            "dataset_final_hash_manifest_sha256": dataset_sha256,
            "dataset_manifest_payload_sha256": dataset_manifest[
                "canonical_payload_sha256"
            ],
            "calibration_final_hash_manifest_sha256": calibration_packet.final_file_sha256,
            "calibration_payload_sha256": calibration["canonical_payload_sha256"],
            "training_packets": [
                {
                    "arm": packet.arm,
                    "seed": packet.seed,
                    "final_hash_manifest_sha256": packet.packet.final_file_sha256,
                    "checkpoint_sha256": packet.checkpoint_record["sha256"],
                    "source_set_sha256": packet.summary["source_set_sha256"],
                }
                for packet in training_packets
            ],
            "evaluator_source_set_sha256": source["source_set_sha256"],
            "production_authority_set_sha256": authorities.set_sha256,
            "production_authorities": list(authorities.records),
            "population_role": population_role,
            "anchors": list(anchors),
            "horizons": list(HORIZONS),
            "full_trace_inclusive": list(FULL_TRACE),
            "late_window_inclusive": list(LATE_WINDOW),
            **(
                {
                    "prospective_authority": {
                        "freeze_final_hash_manifest_sha256": prospective_control.freeze_packet.final_file_sha256,
                        "freeze_payload_sha256": prospective_control.freeze[
                            "canonical_payload_sha256"
                        ],
                        "authorization_file_sha256": prospective_control.authorization_file_sha256,
                        "authorization_payload_sha256": prospective_control.authorization[
                            "canonical_payload_sha256"
                        ],
                        "access_receipt_file_sha256": access_receipt.file_sha256,
                        "access_receipt_payload_sha256": access_receipt.value[
                            "canonical_payload_sha256"
                        ],
                        "development_evaluation_final_hash_manifest_sha256": development_parent.final_file_sha256,
                    },
                    "prospective_population": prospective_provenance,
                }
                if prospective_control is not None
                and access_receipt is not None
                and development_parent is not None
                and prospective_provenance is not None
                else {}
            ),
            "prospective_opened": population_role == "prospective",
            "sealed_opened": False,
        }
    )
    prospective_prediction_assessment = (
        _prediction_assessment(
            prospective_control.freeze["predictions"],
            method_result,
            offline_result,
            anchors,
        )
        if prospective_control is not None
        else None
    )
    result = _self_hashed(
        _json_safe(
            {
                "schema": successor["packet_schemas"]["evaluation"],
                "experiment_id": EXPERIMENT_ID,
                "status": "complete",
                "classification": "SCIENTIFIC_RESULT",
                "population_role": population_role,
                "successor_contract_sha256": successor_sha256,
                "input_manifest_payload_sha256": input_manifest[
                    "canonical_payload_sha256"
                ],
                "production_authority_set_sha256": authorities.set_sha256,
                "production_authorities": list(authorities.records),
                "seeds": list(SEEDS),
                "learned_arms": list(LEARNED_ARMS),
                "operational_correctors": ["identity", "path_projection"],
                "offline_diagnostics": offline_result,
                "method_summaries": method_result,
                "rollout_bookkeeping": rollout_bookkeeping,
                **(
                    {
                        "prospective_prediction_assessment": prospective_prediction_assessment
                    }
                    if prospective_prediction_assessment is not None
                    else {}
                ),
                "aggregation": {
                    "sampling": "deterministic_finite_population_common_trajectory",
                    "anchors_are_not_independent_trajectories": True,
                    "iid_confidence_intervals_reported": False,
                    "q90_definition": "deterministic_nearest_rank",
                    "pairwise_tie_ratio_inclusive": [0.95, 1.05],
                    "failed_trace_primary_metric": "Infinity",
                },
                "metric_definitions": {
                    "normalized_state_error": (
                        "uniform-node equal-field RMS after division by train-only "
                        "state scales"
                    ),
                    "normalized_state_error_auc": (
                        "unit-spaced trapezoidal AUC over h=1..208 divided by 207"
                    ),
                    "late_window": "per-anchor median over h=174..208",
                    "path_distance": (
                        "full-state normalized RMS to the interpolant on the segment "
                        "selected in the train-PCA embedding"
                    ),
                    "projected_path_discrepancy": (
                        "uniform-node normalized state error between that full-state "
                        "interpolant and the time-matched reference"
                    ),
                    "tangent_transverse": (
                        "full normalized error decomposed against the selected "
                        "train-path finite-difference segment; unstable directions "
                        "are marked invalid"
                    ),
                    "tangent_stability_rms_threshold": TANGENT_STABILITY_RMS,
                    "graph_dirichlet": (
                        "mean squared difference of normalized error across unique "
                        "undirected mesh edges and five fields"
                    ),
                    "stored_reference_bank_bytes": (
                        "sum of serialized floating-point train-path, PCA-mean, "
                        "PCA-basis, and path-embedding tensor bytes"
                    ),
                },
                "claim_boundary": {
                    "offline_one_prefix_is_model_side_not_trusted_solver_response": True,
                    "arbitrary_displaced_state_SU2_response_claimed": False,
                    "online_solver_calls": 0,
                    "raw_and_corrected_path_recurrences_separated": True,
                    "identity_parity_required_exact": True,
                    "prospective_opened": population_role == "prospective",
                    "sealed_opened": False,
                },
            }
        )
    )

    _reverify_source(source)
    if _file_sha256(arguments.successor_contract) != successor_sha256:
        raise ValueError("successor contract changed during evaluation")
    if _file_sha256(arguments.contract) != successor["inherited_r0_contract_sha256"]:
        raise ValueError("R0 contract changed during evaluation")
    if _file_sha256(arguments.preregistration) != successor["preregistration_sha256"]:
        raise ValueError("successor preregistration changed during evaluation")
    if _file_sha256(dataset_root / "final_hash_manifest.json") != dataset_sha256:
        raise ValueError("open dataset packet changed during evaluation")
    _reverify_packet(calibration_packet)
    for packet in training_packets:
        _reverify_packet(packet.packet)
    _reverify_authorities(authorities)
    if prospective_control is not None and access_receipt is not None:
        _reverify_prospective_control(prospective_control)
        _reverify_prospective_access_receipt(access_receipt)
        if development_parent is None:
            raise AssertionError("prospective development parent is absent")
        _reverify_packet(development_parent)

    def verify_packet(root: Path) -> ClosedPacket:
        closed = _verify_closed_packet(
            root,
            final_schema=successor["packet_schemas"]["final_hash_manifest"],
            expected_files=OUTPUT_FILES,
            label="successor evaluation output",
        )
        scientific_files = {
            name: closed.files[name] for name in sorted(SCIENTIFIC_OUTPUT_FILES)
        }
        expected_fields = {
            "schema",
            "experiment_id",
            "successor_contract_sha256",
            "population_role",
            "production_authority_set_sha256",
            "production_authorities",
            "scientific_files_sha256",
            "runtime_manifest_excluded_from_scientific_files_sha256",
            "files",
            "self_hash_excluded",
            "prospective_opened",
            "sealed_opened",
        }
        expected_prospective_authority = None
        if prospective_control is not None and access_receipt is not None:
            expected_fields.add("prospective_authority")
            expected_prospective_authority = {
                "freeze_final_hash_manifest_sha256": (
                    prospective_control.freeze_packet.final_file_sha256
                ),
                "freeze_payload_sha256": prospective_control.freeze[
                    "canonical_payload_sha256"
                ],
                "authorization_file_sha256": (
                    prospective_control.authorization_file_sha256
                ),
                "authorization_payload_sha256": prospective_control.authorization[
                    "canonical_payload_sha256"
                ],
                "access_receipt_file_sha256": access_receipt.file_sha256,
                "access_receipt_payload_sha256": access_receipt.value[
                    "canonical_payload_sha256"
                ],
            }
        if (
            set(closed.final) != expected_fields
            or closed.final.get("experiment_id") != EXPERIMENT_ID
            or closed.final.get("successor_contract_sha256") != successor_sha256
            or closed.final.get("population_role") != population_role
            or closed.final.get("production_authority_set_sha256")
            != authorities.set_sha256
            or closed.final.get("production_authorities") != list(authorities.records)
            or closed.final.get("scientific_files_sha256")
            != _canonical_sha256(scientific_files)
            or closed.final.get(
                "runtime_manifest_excluded_from_scientific_files_sha256"
            )
            is not True
            or closed.final.get("prospective_opened")
            != (population_role == "prospective")
            or closed.final.get("sealed_opened") is not False
            or closed.final.get("prospective_authority")
            != expected_prospective_authority
        ):
            raise EvaluationOutputError(
                "closed output role, authority, or scientific digest differs"
            )
        expected_json = {
            "source_manifest.json": source,
            "runtime_manifest.json": runtime_manifest,
            "input_manifest.json": input_manifest,
            "result.json": result,
        }
        for name, expected in expected_json.items():
            schema = expected.get("schema")
            if not isinstance(schema, str):
                raise EvaluationOutputError(f"staged {name} schema is invalid")
            observed = _packet_json(
                closed,
                name,
                schema=schema,
                label=f"successor evaluation {name}",
            )
            if observed != expected:
                raise EvaluationOutputError(f"staged {name} content differs")
        observed_cardinalities = {
            name: _csv_row_count(closed, name)
            for name in EXPECTED_RESULT_CARDINALITIES
            if name.endswith(".csv")
        }
        observed_cardinalities["rollout_snapshots.npz"] = _snapshot_member_count(closed)
        if observed_cardinalities != EXPECTED_RESULT_CARDINALITIES:
            raise EvaluationOutputError("evaluation result cardinalities differ")
        _reverify_packet(closed)
        return closed

    def write_packet(staging: Path) -> None:
        _write_csv(staging / "offline_diagnostics.csv", offline_rows)
        _write_csv(staging / "rollout_metrics.csv", rollout_rows)
        _write_csv(staging / "rollout_structure.csv", structure_rows)
        _write_csv(staging / "method_summary.csv", method_rows)
        _write_csv(staging / "pairwise_summary.csv", pairwise_rows)
        _write_csv(staging / "cost_metrics.csv", cost_rows)
        _write_deterministic_npz(staging / "rollout_snapshots.npz", snapshots)
        _write_json(staging / "source_manifest.json", source)
        _write_json(staging / "runtime_manifest.json", runtime_manifest)
        _write_json(staging / "input_manifest.json", input_manifest)
        _write_json(staging / "result.json", result)
        _reverify_source(source)
        _reverify_authorities(authorities)
        if prospective_control is not None and access_receipt is not None:
            _reverify_prospective_control(prospective_control)
            _reverify_prospective_access_receipt(access_receipt)
        _write_final_manifest(
            staging,
            successor=successor,
            successor_sha256=successor_sha256,
            authorities=authorities,
            population_role=population_role,
            prospective_control=prospective_control,
            access_receipt=access_receipt,
        )
        verify_packet(staging)

    _write_staged_packet(output, write_packet)
    verify_packet(output)
    _reverify_source(source)
    _reverify_authorities(authorities)
    if prospective_control is not None and access_receipt is not None:
        _reverify_prospective_control(prospective_control)
        _reverify_prospective_access_receipt(access_receipt)
    return result


def _mode_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--mode",
        choices=("development", "prospective-freeze", "prospective"),
        default="development",
    )
    parser.add_argument("--prospective-freeze-dir", type=Path)
    parser.add_argument("--prospective-authorization", type=Path)
    return parser


def _parser(mode: str = "development") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("development", "prospective-freeze", "prospective"),
        default="development",
    )
    parser.add_argument(
        "--contract", type=Path, required=True, help="inherited R0 contract"
    )
    parser.add_argument("--successor-contract", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    if mode == "prospective-freeze":
        parser.add_argument("--development-evaluation-dir", type=Path, required=True)
    else:
        parser.add_argument("--dataset-dir", type=Path, required=True)
        parser.add_argument("--calibration-dir", type=Path, required=True)
        parser.add_argument("--training-dir", type=Path, action="append", required=True)
        parser.add_argument(
            "--resource-smoke-receipt", type=Path, action="append", required=True
        )
        parser.add_argument(
            "--execution-authorization", type=Path, action="append", required=True
        )
    if mode == "prospective":
        parser.add_argument("--prospective-freeze-dir", type=Path, required=True)
        parser.add_argument("--prospective-authorization", type=Path, required=True)
        parser.add_argument("--development-evaluation-dir", type=Path, required=True)
        parser.add_argument("--trajectory-dir", type=Path, required=True)
        parser.add_argument("--trajectory-storage-manifest", type=Path, required=True)
        parser.add_argument("--prospective-access-receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    if mode != "prospective-freeze":
        parser.add_argument("--device", default="cuda")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw_arguments = tuple(sys.argv[1:] if argv is None else argv)
    mode_arguments, _ = _mode_parser().parse_known_args(raw_arguments)
    if (
        mode_arguments.mode == "prospective"
        and "--help" not in raw_arguments
        and "-h" not in raw_arguments
    ):
        if (
            mode_arguments.prospective_freeze_dir is None
            or mode_arguments.prospective_authorization is None
        ):
            print(
                "NACA successor protected access refused: prospective freeze and "
                "separate owner authorization are required before any scientific "
                "input argument is inspected",
                file=sys.stderr,
            )
            return 4
        try:
            _load_prospective_control(
                mode_arguments.prospective_freeze_dir,
                mode_arguments.prospective_authorization,
            )
        except (OSError, TypeError, ValueError) as error:
            print(
                f"NACA successor protected access refused: {error}",
                file=sys.stderr,
            )
            return 4
    arguments = _parser(mode_arguments.mode).parse_args(raw_arguments)
    try:
        result = (
            freeze_prospective(arguments)
            if arguments.mode == "prospective-freeze"
            else evaluate(arguments)
        )
    except ProtectedPopulationError as error:
        print(f"NACA successor protected access refused: {error}", file=sys.stderr)
        return 4
    except torch.cuda.OutOfMemoryError as error:
        print(
            f"NACA successor evaluation infrastructure failure: {error}",
            file=sys.stderr,
        )
        return 3
    except (OSError, TypeError, ValueError) as error:
        print(f"NACA successor evaluation invalid artifact: {error}", file=sys.stderr)
        return 2
    except RuntimeError as error:
        print(
            f"NACA successor evaluation infrastructure failure: {error}",
            file=sys.stderr,
        )
        return 3
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
