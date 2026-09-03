#!/usr/bin/env python3
"""Evaluate the six-arm NACA corrective extension on the open development role.

This is deliberately a thin reuse layer over the closed successor evaluator.
It replaces only the extension contract/training-packet/deployment seams and the
four-call PDE-Refiner transition.  Prospective and sealed roles, online solver
calls, and online defect-triggered behavior are not implemented.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

MODULE_PATH = Path(__file__)
if MODULE_PATH.is_symlink():
    raise RuntimeError("extension evaluator source is aliased")
REPO_ROOT = MODULE_PATH.resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.time_dependent_no import (
    evaluate_pcno_naca0012_successor as parent,
)
from scripts.time_dependent_no import (
    train_pcno_naca0012_corrective_extension as trainer,
)
from utility.time_dependent_no.pcno_naca0012 import (
    NACANormalization,
    VerifiedNACAGeometry,
    build_naca_pcno,
    validate_naca_model_config,
)
from utility.time_dependent_no.pcno_naca0012_corrective_extension import (
    EXTENSION_CONTRACT_SHA256,
    EXTENSION_EXPERIMENT_ID,
    EXTENSION_LEARNED_ARMS,
    EXTENSION_PREREGISTRATION_SHA256,
    ExponentialMovingAverage,
    FourStepVPredictionScheduler,
    RefinerNoiseTape,
    VerifiedExtensionContract,
    build_naca_pcno_refiner,
    load_extension_contract,
    refined_recurrent_step,
    sample_refiner_noise_tape,
)

EXPERIMENT_ID = EXTENSION_EXPERIMENT_ID
LEARNED_ARMS = EXTENSION_LEARNED_ARMS
INHERITED_REFERENCE_ARM = "CLEAN"
INHERITED_COMPARISON_ARMS = ("DETACHED_PUSHFORWARD", "PATH_PROJECTION")
ALL_ARMS = LEARNED_ARMS + (INHERITED_REFERENCE_ARM,) + INHERITED_COMPARISON_ARMS
SEEDS = (17, 29, 43)
EMA_ARMS = trainer.EMA_ARMS
REFINER_ARM = "PCNO_PDEREFINER_K3_VPRED"
REFINER_SAMPLER_SEEDS = (101, 211, 307)
DEVELOPMENT_ANCHORS = (1234, 1238, 1243, 1247, 1251, 1256, 1260, 1264)
HORIZONS = parent.HORIZONS
FULL_TRACE = parent.FULL_TRACE
LATE_WINDOW = parent.LATE_WINDOW

SOURCE_MANIFEST_SCHEMA = (
    "time_dependent_no.naca_corrective_extension_evaluation_source.v1"
)
RUNTIME_MANIFEST_SCHEMA = (
    "time_dependent_no.naca_corrective_extension_evaluation_runtime.v1"
)
INPUT_MANIFEST_SCHEMA = (
    "time_dependent_no.naca_corrective_extension_evaluation_inputs.v1"
)
RESULT_SCHEMA = "time_dependent_no.naca_corrective_extension_evaluation.v1"
FINAL_MANIFEST_SCHEMA = (
    "time_dependent_no.naca_corrective_extension_evaluation_final_manifest.v1"
)
OUTPUT_FILES = set(parent.OUTPUT_FILES) | {"refiner_intermediate_diagnostics.csv"}
SCIENTIFIC_OUTPUT_FILES = OUTPUT_FILES - {"runtime_manifest.json"}
TRAINING_FILES = set(trainer.TRAINING_FILES)
EVALUATED_STATE_VARIANT_COUNT = 54
UNIQUE_SNAPSHOT_VARIANT_COUNT = 51
EXPECTED_CARDINALITIES = {
    "offline_diagnostics.csv": 45 * 8,
    "rollout_metrics.csv": EVALUATED_STATE_VARIANT_COUNT * 8 * 208,
    "rollout_structure.csv": EVALUATED_STATE_VARIANT_COUNT * 8 * 208,
    "refiner_intermediate_diagnostics.csv": 18 * 8 * len(HORIZONS) * 5,
    "method_summary.csv": EVALUATED_STATE_VARIANT_COUNT * 8,
    "pairwise_summary.csv": 42 * 8 * len(parent.PRIMARY_METRICS),
    "cost_metrics.csv": 48,
    "rollout_snapshots.npz": (
        UNIQUE_SNAPSHOT_VARIANT_COUNT * 8 * len(HORIZONS) + 8 * len(HORIZONS)
    ),
}
VALIDITY_KEYS = (
    "finite",
    "finite_value_fraction",
    "density_nonpositive_fraction",
    "internal_energy_diagnostic_invalid_fraction",
    "internal_energy_nonpositive_fraction_among_valid_density",
    "nu_tilde_minimum",
    "nu_tilde_negative_fraction",
)

_canonical_sha256 = parent._canonical_sha256
_file_sha256 = parent._file_sha256
_json_safe = parent._json_safe
_self_hashed = parent._self_hashed
_write_json = parent._write_json


@dataclass(frozen=True)
class ExtensionTrainingPacket:
    packet: parent.ClosedPacket
    arm: str
    seed: int
    config: dict[str, Any]
    inputs: dict[str, Any]
    runtime: dict[str, Any]
    summary: dict[str, Any]
    checkpoint_record: dict[str, Any]


@dataclass(frozen=True)
class Deployment:
    packet: ExtensionTrainingPacket
    deployment: str
    sampler_seed: int | None

    @property
    def arm(self) -> str:
        return self.packet.arm

    @property
    def seed(self) -> int:
        return self.packet.seed

    @property
    def key(self) -> str:
        sampler = "none" if self.sampler_seed is None else str(self.sampler_seed)
        return f"{self.arm}:{self.seed}:{self.deployment}:sampler_{sampler}"


@dataclass(frozen=True)
class InheritedBaselinePacket:
    """A parent packet admitted only through the closed parent validator."""

    training: parent.TrainingPacket

    @property
    def seed(self) -> int:
        return self.training.seed

    @property
    def arm(self) -> str:
        return self.training.arm


@dataclass(frozen=True)
class RecursiveInputPacket:
    root: Path
    final_path: Path
    final: dict[str, Any]
    final_sha256: str


def _require_digest(value: Any, label: str) -> str:
    return parent._require_digest(value, label)


def _stable_source_record(path: Path, label: str) -> dict[str, Any]:
    resolved = parent._safe_regular_file(path, label=label)
    before = resolved.stat()
    payload = resolved.read_bytes()
    after = resolved.stat()
    if (
        before.st_size != len(payload)
        or after.st_size != len(payload)
        or before.st_mtime_ns != after.st_mtime_ns
    ):
        raise RuntimeError(f"{label} changed while being captured")
    return {"bytes": len(payload), "sha256": sha256(payload).hexdigest()}


def _capture_sources() -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    groups = (
        parent.R0_SOURCE_RECORDS_AT_IMPORT,
        parent.SUCCESSOR_SOURCE_RECORDS_AT_IMPORT,
        trainer.SOURCE_RECORDS_AT_IMPORT,
    )
    for group in groups:
        for relative, record in group.items():
            candidate = dict(record)
            if relative in records and records[relative] != candidate:
                raise RuntimeError(f"source closures disagree for {relative}")
            records[relative] = candidate
    relative = MODULE_PATH.resolve().relative_to(REPO_ROOT).as_posix()
    records[relative] = _stable_source_record(MODULE_PATH, "extension evaluator")
    return dict(sorted(records.items()))


SOURCE_RECORDS_AT_IMPORT = _capture_sources()
SOURCE_SET_SHA256 = _canonical_sha256(SOURCE_RECORDS_AT_IMPORT)


def _source_snapshot() -> dict[str, Any]:
    return _self_hashed(
        {
            "schema": SOURCE_MANIFEST_SCHEMA,
            "experiment_id": EXPERIMENT_ID,
            "files": SOURCE_RECORDS_AT_IMPORT,
            "source_set_sha256": SOURCE_SET_SHA256,
        }
    )


def _reverify_source(source: Mapping[str, Any]) -> None:
    observed = {
        relative: _stable_source_record(REPO_ROOT / relative, f"source {relative}")
        for relative in source["files"]
    }
    if observed != source["files"] or _canonical_sha256(observed) != source.get(
        "source_set_sha256"
    ):
        raise ValueError("extension evaluator source closure changed")


def _validate_population_role(role: str) -> str:
    if role != "development":
        raise parent.ProtectedPopulationError(
            "the extension evaluator implements only the open development role"
        )
    return role


def _safe_recursive_relative(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} path is not a string")
    relative = Path(value)
    if (
        relative.is_absolute()
        or relative.as_posix() != value
        or not relative.parts
        or ".." in relative.parts
        or "." in relative.parts
    ):
        raise ValueError(f"{label} path is not canonical and relative")
    return value


def _verify_recursive_input_packet(
    final_path: Path, *, schema: str, label: str
) -> RecursiveInputPacket:
    resolved_final = parent._safe_regular_file(final_path, label=f"{label} final")
    root = parent._safe_directory(resolved_final.parent, label=label)
    if (
        resolved_final.parent != root
        or resolved_final.name != "final_hash_manifest.json"
    ):
        raise ValueError(f"{label} final manifest path differs")
    final_bytes = resolved_final.read_bytes()
    final = json.loads(final_bytes)
    files = final.get("files") if isinstance(final, dict) else None
    if (
        not isinstance(final, dict)
        or final.get("schema") != schema
        or final.get("self_hash_excluded") is not True
        or not isinstance(files, dict)
        or not files
        or final.get("prospective_opened") is not False
        or final.get("sealed_opened") is not False
    ):
        raise ValueError(f"{label} final manifest differs")
    expected = set(files) | {"final_hash_manifest.json"}
    all_entries = list(root.rglob("*"))
    if any(path.is_symlink() for path in all_entries):
        raise ValueError(f"{label} contains an aliased entry")
    observed = {
        path.resolve().relative_to(root).as_posix()
        for path in all_entries
        if path.is_file()
    }
    if observed != expected:
        raise ValueError(f"{label} contains missing or unmanifested files")
    for relative, record in files.items():
        relative = _safe_recursive_relative(relative, f"{label} artifact")
        if not isinstance(record, Mapping) or record.get("relative_path") != relative:
            raise ValueError(f"{label} artifact record differs: {relative}")
        candidate = root / relative
        resolved = parent._safe_regular_file(
            candidate, label=f"{label} artifact {relative}"
        )
        try:
            resolved.relative_to(root)
        except ValueError as error:
            raise ValueError(f"{label} artifact escapes its root") from error
        if (
            isinstance(record.get("bytes"), bool)
            or record.get("bytes") != resolved.stat().st_size
            or _require_digest(
                record.get("sha256"), f"{label} artifact {relative} digest"
            )
            != _file_sha256(resolved)
        ):
            raise ValueError(f"{label} artifact hash differs: {relative}")
    if resolved_final.read_bytes() != final_bytes:
        raise ValueError(f"{label} final manifest changed while reading")
    return RecursiveInputPacket(
        root=root,
        final_path=resolved_final,
        final=final,
        final_sha256=sha256(final_bytes).hexdigest(),
    )


def _load_solver_label_provenance(
    paired_bank_dir: Path,
    pilot_final_manifest: Path,
    *,
    expected_coordinates: np.ndarray,
    train_states: np.ndarray,
    train_frame_indices: np.ndarray,
    normalization: NACANormalization,
) -> tuple[RecursiveInputPacket, RecursiveInputPacket]:
    pilot = _verify_recursive_input_packet(
        pilot_final_manifest,
        schema=trainer.RELABEL_PILOT_FINAL_SCHEMA,
        label="SU2 relabel pilot packet",
    )
    if pilot.final.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("SU2 relabel pilot experiment differs")
    receipt_path = pilot.root / trainer.RELABEL_PILOT_RECEIPT_FILE
    receipt, _ = parent._read_self_hashed_json(
        receipt_path,
        schema=trainer.RELABEL_PILOT_RECEIPT_SCHEMA,
        label="SU2 relabel pilot receipt",
    )
    if (
        receipt.get("status") != "pilot_succeeded"
        or receipt.get("scientifically_usable") is not True
        or receipt.get("paired_query_bank_unlocked") is not True
        or receipt.get("prospective_opened") is not False
        or receipt.get("sealed_opened") is not False
        or receipt.get("online_solver_calls") is not False
    ):
        raise ValueError("SU2 relabel pilot did not unlock the paired bank")
    bank = _verify_recursive_input_packet(
        paired_bank_dir / "final_hash_manifest.json",
        schema=trainer.PAIRED_BANK_FINAL_SCHEMA,
        label="paired displaced-state bank",
    )
    if (
        bank.final.get("experiment_id") != EXPERIMENT_ID
        or bank.final.get("packet_role") != "paired_query_bank"
        or bank.final.get("status") != "complete"
        or bank.final.get("extension_contract_sha256") != EXTENSION_CONTRACT_SHA256
        or bank.final.get("preregistration_sha256") != EXTENSION_PREREGISTRATION_SHA256
        or bank.final.get("dataset_final_hash_manifest_sha256")
        != trainer.DATASET_FINAL_SHA256
        or bank.final.get("pilot_final_hash_manifest_sha256") != pilot.final_sha256
        or bank.final.get("online_solver_calls") is not False
    ):
        raise ValueError("paired displaced-state bank provenance differs")
    verified_bank = trainer._load_paired_bank(
        paired_bank_dir,
        expected_final_sha256=bank.final_sha256,
        pilot_final_manifest_path=pilot.final_path,
        expected_pilot_final_sha256=pilot.final_sha256,
        expected_coordinates=expected_coordinates,
        train_states=train_states,
        train_frame_indices=train_frame_indices,
        normalization=normalization,
    )
    trainer._reverify_paired_bank(verified_bank)
    del verified_bank
    return pilot, bank


def _validate_packet_binding(
    value: Mapping[str, Any],
    *,
    label: str,
    arm: str,
    seed: int,
    dataset_sha256: str,
    dataset_payload_sha256: str,
    calibration_sha256: str,
    calibration_payload_sha256: str,
    paired_bank_sha256: str,
    pilot_sha256: str,
) -> None:
    paired = arm in trainer.PAIRED_ARMS
    expected = {
        "experiment_id": EXPERIMENT_ID,
        "arm": arm,
        "seed": seed,
        "extension_contract_sha256": EXTENSION_CONTRACT_SHA256,
        "extension_preregistration_sha256": EXTENSION_PREREGISTRATION_SHA256,
        "inherited_r0_contract_sha256": trainer.R0_CONTRACT_SHA256,
        "inherited_successor_contract_sha256": trainer.SUCCESSOR_CONTRACT_SHA256,
        "dataset_manifest_payload_sha256": dataset_payload_sha256,
        "dataset_final_hash_manifest_sha256": dataset_sha256,
        "successor_calibration_payload_sha256": calibration_payload_sha256,
        "successor_calibration_final_hash_manifest_sha256": calibration_sha256,
        "paired_bank_final_hash_manifest_sha256": (
            paired_bank_sha256 if paired else None
        ),
        "relabel_pilot_final_hash_manifest_sha256": pilot_sha256 if paired else None,
        "train_opened": True,
        "development_opened": True,
        "prospective_opened": False,
        "sealed_opened": False,
        "offline_training_label_solver_calls": False,
        "online_solver_calls": False,
        "online_defect_trigger": False,
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise ValueError(f"{label} {key} binding differs")
    _require_digest(value.get("source_set_sha256"), f"{label} source-set digest")


def _checkpoint_mapping(
    packet: parent.ClosedPacket, device: torch.device
) -> tuple[dict[str, Any], bytes]:
    _, payload = parent._verify_record(packet.root, packet.files["best.pt"], "best.pt")
    checkpoint = torch.load(io.BytesIO(payload), map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict):
        raise TypeError("extension selected checkpoint is not a mapping")
    return checkpoint, payload


def _verify_training_packet(
    root: Path,
    *,
    dataset_sha256: str,
    dataset_payload_sha256: str,
    calibration_sha256: str,
    calibration_payload_sha256: str,
    paired_bank_sha256: str,
    pilot_sha256: str,
) -> ExtensionTrainingPacket:
    packet = parent._verify_closed_packet(
        root,
        final_schema=trainer.FINAL_MANIFEST_SCHEMA,
        expected_files=TRAINING_FILES,
        label="extension training packet",
    )
    artifacts = {
        name: parent._packet_json(
            packet,
            name,
            schema=trainer.TRAINING_SCHEMA,
            label=f"extension training {name}",
        )
        for name in (
            "config.json",
            "history.json",
            "input_manifest.json",
            "runtime_manifest.json",
            "status.json",
            "summary.json",
        )
    }
    summary = artifacts["summary.json"]
    arm = summary.get("arm")
    seed = summary.get("seed")
    if arm not in LEARNED_ARMS:
        raise ValueError("extension training packet arm differs")
    if isinstance(seed, bool) or seed not in SEEDS:
        raise ValueError("extension training packet seed differs")
    seed = int(seed)
    for value, label in [(packet.final, "final manifest")] + [
        (value, name) for name, value in artifacts.items()
    ]:
        _validate_packet_binding(
            value,
            label=f"extension training {label}",
            arm=arm,
            seed=seed,
            dataset_sha256=dataset_sha256,
            dataset_payload_sha256=dataset_payload_sha256,
            calibration_sha256=calibration_sha256,
            calibration_payload_sha256=calibration_payload_sha256,
            paired_bank_sha256=paired_bank_sha256,
            pilot_sha256=pilot_sha256,
        )
    config = artifacts["config.json"]
    history = artifacts["history.json"]
    inputs = artifacts["input_manifest.json"]
    runtime = artifacts["runtime_manifest.json"]
    status = artifacts["status.json"]
    if (
        summary.get("record_kind") != "summary"
        or summary.get("status") != "complete"
        or summary.get("classification") != "SCIENTIFIC_TRAINING_ARTIFACT"
        or summary.get("completed_epochs") != 100
        or summary.get("checkpoint_selection_used_rollout") is not False
        or summary.get("scientific_claim_made") is not False
        or status.get("record_kind") != "status"
        or status.get("status") != "complete"
        or status.get("last_completed_epoch") != 100
        or config.get("record_kind") != "config"
        or config.get("epochs") != 100
        or config.get("effective_batch_size") != 4
        or config.get("checkpoint_epochs") != list(range(5, 101, 5))
        or config.get("checkpoint_selection_used_rollout") is not False
        or history.get("record_kind") != "history"
        or not isinstance(history.get("epochs"), list)
        or len(history["epochs"]) != 100
        or inputs.get("record_kind") != "input_manifest"
        or runtime.get("record_kind") != "runtime_manifest"
        or not isinstance(runtime.get("runtime"), Mapping)
    ):
        raise ValueError("extension training packet is incomplete")
    best_epoch = summary.get("best_epoch")
    if (
        isinstance(best_epoch, bool)
        or best_epoch not in range(5, 101, 5)
        or status.get("best_epoch") != best_epoch
        or config.get("checkpoint_selection_deployment")
        != ("ema" if arm in EMA_ARMS else "online")
        or summary.get("checkpoint_selection_deployment")
        != ("ema" if arm in EMA_ARMS else "online")
    ):
        raise ValueError("extension checkpoint selection differs")
    source_files = inputs.get("source_files")
    source_set = summary["source_set_sha256"]
    if (
        not isinstance(source_files, Mapping)
        or not source_files
        or _canonical_sha256(source_files) != source_set
    ):
        raise ValueError("extension training source closure differs")
    for relative, record in source_files.items():
        path = parent._repo_source_file(relative, label=f"training source {relative}")
        if record != {"bytes": path.stat().st_size, "sha256": _file_sha256(path)}:
            raise ValueError(f"extension training source differs: {relative}")
    model_cost = summary.get("model_call_cost")
    if (
        not isinstance(model_cost, Mapping)
        or summary.get("model_call_cost_sha256") != _canonical_sha256(model_cost)
        or _require_digest(
            summary.get("model_call_schedule_sha256"),
            "extension training model-call schedule",
        )
        != summary["model_call_schedule_sha256"]
        or not isinstance(summary.get("training_wall_time_seconds"), (int, float))
    ):
        raise ValueError("extension training cost record differs")
    checkpoint, checkpoint_bytes = _checkpoint_mapping(packet, torch.device("cpu"))
    _validate_packet_binding(
        checkpoint,
        label="extension selected checkpoint",
        arm=arm,
        seed=seed,
        dataset_sha256=dataset_sha256,
        dataset_payload_sha256=dataset_payload_sha256,
        calibration_sha256=calibration_sha256,
        calibration_payload_sha256=calibration_payload_sha256,
        paired_bank_sha256=paired_bank_sha256,
        pilot_sha256=pilot_sha256,
    )
    ema_expected = arm in EMA_ARMS
    if (
        checkpoint.get("schema") != trainer.TRAINING_SCHEMA
        or checkpoint.get("record_kind") != "selected_checkpoint"
        or checkpoint.get("model_only") is not True
        or checkpoint.get("epoch") != best_epoch
        or checkpoint.get("best_epoch") != best_epoch
        or checkpoint.get("checkpoint_selection_used_rollout") is not False
        or not isinstance(checkpoint.get("online_model_state_dict"), Mapping)
        or (checkpoint.get("ema_state_dict") is not None) is not ema_expected
        or (checkpoint.get("ema_num_updates") is not None) is not ema_expected
        or (
            ema_expected
            and (
                isinstance(checkpoint.get("ema_num_updates"), bool)
                or checkpoint["ema_num_updates"] <= 0
            )
        )
        or (
            arm == REFINER_ARM
            and checkpoint.get("refiner_scheduler")
            != {"betas": list(FourStepVPredictionScheduler().betas)}
        )
        or (arm != REFINER_ARM and checkpoint.get("refiner_scheduler") is not None)
    ):
        raise ValueError("extension selected checkpoint inventory differs")
    if (
        len(checkpoint_bytes) != packet.files["best.pt"]["bytes"]
        or sha256(checkpoint_bytes).hexdigest() != packet.files["best.pt"]["sha256"]
    ):
        raise ValueError("extension selected checkpoint changed")
    parent._reverify_packet(packet)
    return ExtensionTrainingPacket(
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
    dataset_sha256: str,
    dataset_payload_sha256: str,
    calibration_sha256: str,
    calibration_payload_sha256: str,
    paired_bank_sha256: str,
    pilot_sha256: str,
) -> list[ExtensionTrainingPacket]:
    if len(roots) != len(LEARNED_ARMS) * len(SEEDS):
        raise ValueError("evaluation requires the exact six-arm three-seed grid")
    packets = [
        _verify_training_packet(
            root,
            dataset_sha256=dataset_sha256,
            dataset_payload_sha256=dataset_payload_sha256,
            calibration_sha256=calibration_sha256,
            calibration_payload_sha256=calibration_payload_sha256,
            paired_bank_sha256=paired_bank_sha256,
            pilot_sha256=pilot_sha256,
        )
        for root in roots
    ]
    keys = [(packet.arm, packet.seed) for packet in packets]
    expected = {(arm, seed) for arm in LEARNED_ARMS for seed in SEEDS}
    if set(keys) != expected or len(keys) != len(set(keys)):
        raise ValueError("training packets do not form the six-arm three-seed grid")
    if len({packet.summary["source_set_sha256"] for packet in packets}) != 1:
        raise ValueError("training packets do not share one frozen source closure")
    for seed in SEEDS:
        same_architecture = [
            packet
            for packet in packets
            if packet.seed == seed and packet.arm != REFINER_ARM
        ]
        if (
            len(
                {
                    packet.packet.final["initial_model_state_sha256"]
                    for packet in same_architecture
                }
            )
            != 1
        ):
            raise ValueError(f"baseline-family initialization differs for seed {seed}")
        if (
            len(
                {
                    packet.packet.final["presentation_schedule_sha256"]
                    for packet in same_architecture
                }
            )
            != 1
        ):
            raise ValueError(f"presentation schedule differs for seed {seed}")
    packets.sort(key=lambda item: (LEARNED_ARMS.index(item.arm), item.seed))
    return packets


def _load_inherited_baseline_packets(
    clean_roots: Sequence[Path],
    detached_roots: Sequence[Path],
    *,
    successor: Mapping[str, Any],
    successor_sha256: str,
    dataset_sha256: str,
    dataset_payload_sha256: str,
    calibration_sha256: str,
    calibration_payload_sha256: str,
    baseline: Mapping[str, Any],
    geometry: VerifiedNACAGeometry,
) -> tuple[InheritedBaselinePacket, ...]:
    if len(clean_roots) != len(SEEDS) or len(detached_roots) != len(SEEDS):
        raise ValueError(
            "exactly three inherited CLEAN and three inherited "
            "DETACHED_PUSHFORWARD packets are required"
        )
    packets = tuple(
        InheritedBaselinePacket(
            parent._verify_training_packet(
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
        )
        for root in (*clean_roots, *detached_roots)
    )
    expected = {
        (arm, seed)
        for arm in (INHERITED_REFERENCE_ARM, "DETACHED_PUSHFORWARD")
        for seed in SEEDS
    }
    keys = {(item.arm, item.seed) for item in packets}
    if (
        any(item.arm != INHERITED_REFERENCE_ARM for item in packets[: len(SEEDS)])
        or any(item.arm != "DETACHED_PUSHFORWARD" for item in packets[len(SEEDS) :])
        or keys != expected
        or len(packets) != len(expected)
        or len({item.training.summary["source_set_sha256"] for item in packets}) != 1
    ):
        raise ValueError("inherited parent baseline packet inventory differs")
    for seed in SEEDS:
        paired = [item.training for item in packets if item.seed == seed]
        if (
            len(paired) != 2
            or {item.arm for item in paired}
            != {INHERITED_REFERENCE_ARM, "DETACHED_PUSHFORWARD"}
            or len({item.summary["initial_model_state_sha256"] for item in paired}) != 1
            or len({item.summary["presentation_schedule_sha256"] for item in paired})
            != 1
        ):
            raise ValueError(
                "inherited CLEAN and DETACHED_PUSHFORWARD are not paired "
                f"for seed {seed}"
            )
    return tuple(
        sorted(packets, key=lambda item: (ALL_ARMS.index(item.arm), item.seed))
    )


def _deployment_inventory(
    packets: Sequence[ExtensionTrainingPacket],
) -> tuple[Deployment, ...]:
    deployments: list[Deployment] = []
    for packet in packets:
        kinds = ("online", "ema") if packet.arm in EMA_ARMS else ("online",)
        sampler_seeds: tuple[int | None, ...] = (
            tuple(REFINER_SAMPLER_SEEDS) if packet.arm == REFINER_ARM else (None,)
        )
        for kind in kinds:
            for sampler_seed in sampler_seeds:
                deployments.append(Deployment(packet, kind, sampler_seed))
    expected = 39
    if (
        len(deployments) != expected
        or len({item.key for item in deployments}) != expected
    ):
        raise ValueError("extension deployment inventory differs")
    return tuple(deployments)


def _instantiate_deployment_model(
    deployment: Deployment,
    *,
    extension: VerifiedExtensionContract,
    baseline: Mapping[str, Any],
    geometry: VerifiedNACAGeometry,
    device: torch.device,
) -> nn.Module:
    checkpoint, payload = _checkpoint_mapping(deployment.packet.packet, device)
    if sha256(payload).hexdigest() != deployment.packet.checkpoint_record["sha256"]:
        raise ValueError("selected checkpoint changed before model construction")
    if deployment.arm == REFINER_ARM:
        model = build_naca_pcno_refiner(extension, geometry).to(device)
        if checkpoint.get("model_config") != model.model_config():
            raise ValueError("refiner checkpoint model configuration differs")
    else:
        validate_naca_model_config(
            checkpoint.get("model_config", {}), baseline, geometry
        )
        model = build_naca_pcno(baseline, geometry).to(device)
    model.load_state_dict(checkpoint["online_model_state_dict"], strict=True)
    if deployment.deployment == "online":
        model.eval()
        return model
    if deployment.deployment != "ema" or deployment.arm not in EMA_ARMS:
        raise ValueError("unsupported extension deployment")
    ema = ExponentialMovingAverage(model, decay=0.995).to(device)
    ema.load_state_dict(checkpoint["ema_state_dict"], strict=True)
    if (
        int(ema.num_updates.item()) != checkpoint["ema_num_updates"]
        or ema.decay != 0.995
    ):
        raise ValueError("EMA checkpoint state differs")
    ema.eval()
    if ema.model.training:
        raise AssertionError("EMA deployment entered training mode")
    return ema.model


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


class RefinerTransitionAdapter(nn.Module):
    """Expose the four-call refiner through the parent one-transition API."""

    def __init__(
        self,
        model: nn.Module,
        normalization: NACANormalization,
        *,
        sampler_seed: int,
        device: torch.device,
    ) -> None:
        super().__init__()
        if sampler_seed not in REFINER_SAMPLER_SEEDS:
            raise ValueError("unregistered PDE-Refiner sampler seed")
        self.model = model
        self.normalization = normalization
        self.scheduler = FourStepVPredictionScheduler()
        self.generator = torch.Generator(device=device).manual_seed(sampler_seed)
        self.sampler_seed = sampler_seed
        self._initial_generator_state_sha256 = _tensor_sha256(
            self.generator.get_state()
        )
        self.physical_steps = 0
        self.inner_model_calls = 0
        self.draw_tensors = 0
        self.noise_tapes_drawn = 0
        self.paired_response_tape_reuses = 0
        self._state_shape: tuple[int, ...] | None = None
        self._state_dtype: str | None = None
        self._paired_response_position: int | None = None
        self._paired_response_tape: RefinerNoiseTape | None = None

    def prepare_fourier_tensors(
        self, geometry_batch: Mapping[str, torch.Tensor]
    ) -> tuple[torch.Tensor, ...]:
        return self.model.prepare_fourier_tensors(geometry_batch)  # type: ignore[attr-defined]

    def _sample_tape(self, reference: torch.Tensor) -> RefinerNoiseTape:
        shape = tuple(reference.shape)
        dtype = str(reference.dtype)
        if self._state_shape is None:
            self._state_shape = shape
            self._state_dtype = dtype
        elif (shape, dtype) != (self._state_shape, self._state_dtype):
            raise ValueError("PDE-Refiner noise-tape draw shape changed")
        tape = sample_refiner_noise_tape(reference, generator=self.generator)
        self.noise_tapes_drawn += 1
        self.draw_tensors += 4
        return tape

    def _record_step(self, model_calls: int) -> None:
        if model_calls != 4:
            raise AssertionError("PDE-Refiner did not execute four calls")
        self.physical_steps += 1
        self.inner_model_calls += model_calls

    @contextmanager
    def paired_response_diagnostic(self):
        """Replay one tape for clean-forced and one-prefix response branches."""

        if self._paired_response_position is not None:
            raise RuntimeError("paired refiner diagnostic is already active")
        self._paired_response_position = 0
        self._paired_response_tape = None
        try:
            yield
        except Exception:
            self._paired_response_position = None
            self._paired_response_tape = None
            raise
        if self._paired_response_position != 3:
            self._paired_response_position = None
            self._paired_response_tape = None
            raise RuntimeError(
                "paired refiner diagnostic did not make three transitions"
            )
        self._paired_response_position = None
        self._paired_response_tape = None

    def _forward_tape(self, reference: torch.Tensor) -> RefinerNoiseTape:
        position = self._paired_response_position
        if position is None:
            return self._sample_tape(reference)
        if position == 0:
            tape = self._sample_tape(reference)
        elif position == 1:
            tape = self._sample_tape(reference)
            self._paired_response_tape = tape
        elif position == 2:
            tape = self._paired_response_tape
            if tape is None:
                raise AssertionError("paired response tape is absent")
            self.paired_response_tape_reuses += 1
        else:
            raise RuntimeError("paired refiner diagnostic made too many transitions")
        self._paired_response_position = position + 1
        return tape

    def refine(
        self,
        previous: torch.Tensor,
        current: torch.Tensor,
        geometry_batch: Mapping[str, torch.Tensor],
        *,
        fourier_tensors: tuple[torch.Tensor, ...] | None,
    ):
        tape = self._sample_tape(current)
        step = refined_recurrent_step(
            self.model,
            previous,
            current,
            geometry_batch,
            self.normalization,
            self.scheduler,
            tape,
            fourier_tensors=fourier_tensors,
        )
        if step.model_calls != 4 or len(step.intermediate_candidates) != 5:
            raise AssertionError("PDE-Refiner did not execute four calls")
        self._record_step(step.model_calls)
        return step

    def forward(
        self,
        features: torch.Tensor,
        geometry_batch: Mapping[str, torch.Tensor],
        *,
        fourier_tensors: tuple[torch.Tensor, ...] | None = None,
    ) -> torch.Tensor:
        if features.ndim != 3 or features.shape[-1] != 16:
            raise ValueError(
                "refiner transition adapter expects parent 16-channel input"
            )
        tape = self._forward_tape(features[..., 11:16])
        candidate = tape.initial
        # Parent input assembly already produced the exact normalized histories.
        # Use those bytes directly instead of inverse-normalizing and rounding.
        for timestep in (3, 2, 1, 0):
            one_hot = torch.nn.functional.one_hot(
                torch.full(
                    (features.shape[0],),
                    timestep,
                    dtype=torch.int64,
                    device=features.device,
                ),
                num_classes=4,
            ).to(dtype=features.dtype)
            refiner_features = torch.cat(
                (
                    features,
                    candidate,
                    one_hot[:, None, :].expand(-1, features.shape[1], -1),
                ),
                dim=-1,
            )
            velocity = self.model(
                refiner_features,
                geometry_batch,
                fourier_tensors=fourier_tensors,
            )
            candidate = self.scheduler.step(
                velocity,
                timestep,
                candidate,
                noise=tape.reverse_noise(timestep),
            )
        self._record_step(4)
        return candidate

    def noise_schedule_record(self) -> dict[str, Any]:
        record = {
            "sampler_seed": self.sampler_seed,
            "generator_device": str(self.generator.device),
            "initial_generator_state_sha256": self._initial_generator_state_sha256,
            "final_generator_state_sha256": _tensor_sha256(self.generator.get_state()),
            "physical_steps": self.physical_steps,
            "draw_tensors": self.draw_tensors,
            "noise_tapes_drawn": self.noise_tapes_drawn,
            "paired_response_tape_reuses": self.paired_response_tape_reuses,
            "state_shape": list(self._state_shape or ()),
            "state_dtype": self._state_dtype,
            "inner_model_calls": self.inner_model_calls,
        }
        return {**record, "schedule_sha256": _canonical_sha256(record)}


def _decorate_rows(
    rows: Sequence[Mapping[str, Any]], deployment: Deployment
) -> list[dict[str, Any]]:
    return [
        {
            **dict(row),
            "deployment": deployment.deployment,
            "sampler_seed": deployment.sampler_seed,
        }
        for row in rows
    ]


def _rename_snapshots(
    snapshots: Mapping[str, np.ndarray], deployment: Deployment
) -> dict[str, np.ndarray]:
    suffix = f"_{deployment.deployment}" + (
        "" if deployment.sampler_seed is None else f"_sampler{deployment.sampler_seed}"
    )
    renamed: dict[str, np.ndarray] = {}
    for name, value in snapshots.items():
        if name.startswith("reference_"):
            renamed[name] = value
        else:
            renamed[f"{name}{suffix}"] = value
    return renamed


def _prediction_snapshot_key(row: Mapping[str, Any]) -> str:
    arm = str(row["arm"])
    seed = int(row["seed"])
    deployment = str(row["deployment"])
    state_stage = str(row["state_stage"])
    anchor = int(row["anchor"])
    horizon = int(row["horizon"])
    sampler_value = row.get("sampler_seed")
    sampler_seed = None if sampler_value in (None, "") else int(sampler_value)
    if arm == INHERITED_REFERENCE_ARM:
        if (deployment, state_stage, sampler_seed) not in {
            ("raw", "raw", None),
            ("identity", "corrected", None),
        }:
            raise ValueError("inherited CLEAN snapshot variant differs")
        # The identity deployment is exactly the raw CLEAN prediction and shares it.
        return f"clean_seed{seed}_raw_a{anchor}_h{horizon}"
    if arm == "DETACHED_PUSHFORWARD":
        if (deployment, state_stage, sampler_seed) != ("raw", "raw", None):
            raise ValueError("inherited detached snapshot variant differs")
        return f"detached_pushforward_seed{seed}_raw_a{anchor}_h{horizon}"
    if arm == "PATH_PROJECTION":
        if deployment != "path_projection" or sampler_seed is not None:
            raise ValueError("path-projection snapshot variant differs")
        stage = {
            "raw_pre_correction": "raw",
            "corrected": "corrected",
        }.get(state_stage)
        if stage is None:
            raise ValueError("path-projection snapshot stage differs")
        return f"path_projection_seed{seed}_{stage}_a{anchor}_h{horizon}"
    if arm not in LEARNED_ARMS or state_stage != "raw":
        raise ValueError("extension snapshot variant differs")
    if deployment not in {"online", "ema"}:
        raise ValueError("extension snapshot deployment differs")
    if (arm == REFINER_ARM) != (sampler_seed in REFINER_SAMPLER_SEEDS):
        raise ValueError("extension snapshot sampler binding differs")
    suffix = f"_{deployment}" + (
        "" if sampler_seed is None else f"_sampler{sampler_seed}"
    )
    return f"{arm.lower()}_seed{seed}_raw_a{anchor}_h{horizon}{suffix}"


def _close_registered_snapshots(
    snapshots: Mapping[str, np.ndarray],
    rollout_rows: Sequence[Mapping[str, Any]],
    *,
    states: np.ndarray,
    indices: np.ndarray,
    anchors: Sequence[int],
) -> dict[str, np.ndarray]:
    positions = {int(index): offset for offset, index in enumerate(indices)}
    if len(positions) != len(indices):
        raise ValueError("snapshot reference indices are not unique")
    registered: dict[str, list[bool]] = {}
    sampled_variants: set[tuple[str, int, str, str, int | None]] = set()
    anchor_set = {int(anchor) for anchor in anchors}
    for row in rollout_rows:
        horizon = int(row["horizon"])
        if horizon not in HORIZONS:
            continue
        anchor = int(row["anchor"])
        if anchor not in anchor_set:
            raise ValueError("snapshot rollout row has an unregistered anchor")
        finite = row.get("finite")
        if not isinstance(finite, (bool, np.bool_)):
            raise TypeError("snapshot rollout row has a non-boolean finite flag")
        sampler_value = row.get("sampler_seed")
        sampler_seed = None if sampler_value in (None, "") else int(sampler_value)
        sampled_variants.add(
            (
                str(row["arm"]),
                int(row["seed"]),
                str(row["deployment"]),
                str(row["state_stage"]),
                sampler_seed,
            )
        )
        registered.setdefault(_prediction_snapshot_key(row), []).append(bool(finite))
    if len(sampled_variants) != EVALUATED_STATE_VARIANT_COUNT:
        raise ValueError("snapshot state-variant inventory differs")
    expected_prediction_members = (
        UNIQUE_SNAPSHOT_VARIANT_COUNT * len(anchors) * len(HORIZONS)
    )
    if len(registered) != expected_prediction_members:
        raise ValueError("snapshot prediction-key inventory differs")

    reference_keys: set[str] = set()
    expected_shape = tuple(states.shape[1:])
    closed = {name: np.asarray(value) for name, value in snapshots.items()}
    for anchor in anchors:
        for horizon in HORIZONS:
            position = positions.get(int(anchor) + horizon)
            if position is None:
                raise ValueError("snapshot reference escapes development")
            target = np.asarray(states[position], dtype=np.float64)
            if (
                target.shape != expected_shape
                or not np.isrealobj(target)
                or not np.all(np.isfinite(target))
            ):
                raise ValueError("snapshot reference state is not finite and real")
            reference_key = f"reference_a{anchor}_h{horizon}"
            reference_keys.add(reference_key)
            existing_reference = closed.get(reference_key)
            if existing_reference is None:
                closed[reference_key] = target
            elif (
                existing_reference.shape != expected_shape
                or not np.isrealobj(existing_reference)
                or not np.all(np.isfinite(existing_reference))
                or not np.array_equal(existing_reference, target)
            ):
                raise ValueError("snapshot reference member differs")

    allowed = set(registered) | reference_keys
    if not set(closed).issubset(allowed):
        raise ValueError("snapshot packet contains an unregistered member")
    for name, finite_flags in registered.items():
        if len(set(finite_flags)) != 1:
            raise ValueError("shared snapshot variants disagree on finite status")
        finite = finite_flags[0]
        value = closed.get(name)
        if finite:
            if value is None:
                raise ValueError("finite registered prediction snapshot is missing")
            if (
                value.shape != expected_shape
                or value.dtype != np.dtype(np.float32)
                or not np.all(np.isfinite(value))
            ):
                raise ValueError("finite registered prediction snapshot differs")
            continue
        if value is not None and not (
            value.shape == expected_shape
            and value.dtype == np.dtype(np.float32)
            and np.all(np.isnan(value))
        ):
            raise ValueError("unavailable prediction snapshot is not a NaN sentinel")
        closed[name] = np.full(expected_shape, np.nan, dtype=np.float32)
    if (
        set(closed) != allowed
        or len(closed) != EXPECTED_CARDINALITIES["rollout_snapshots.npz"]
    ):
        raise ValueError("closed snapshot member inventory differs")
    return closed


def _refiner_intermediate_row(
    *,
    deployment: Deployment,
    anchor: int,
    horizon: int,
    stage: str,
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "arm": deployment.arm,
        "seed": deployment.seed,
        "deployment": deployment.deployment,
        "sampler_seed": deployment.sampler_seed,
        "anchor": anchor,
        "horizon": horizon,
        "stage": stage,
        "normalized_state_error": metrics["normalized_state_error"],
        "path_distance": metrics["path_distance"],
        "projected_path_discrepancy": metrics["projected_path_discrepancy"],
        **{key: metrics[key] for key in VALIDITY_KEYS},
    }


def _refiner_rollout(
    *,
    deployment: Deployment,
    adapter: RefinerTransitionAdapter,
    states: np.ndarray,
    indices: np.ndarray,
    anchors: Sequence[int],
    geometry: VerifiedNACAGeometry,
    normalization: NACANormalization,
    projector: Any,
    reference_cache: Mapping[int, parent.Projection],
    graph_edges: np.ndarray,
    weights: Mapping[str, np.ndarray],
    structure_context: tuple[np.ndarray, ...],
    device: torch.device,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, np.ndarray],
    dict[str, Any],
]:
    positions = {int(index): offset for offset, index in enumerate(indices)}
    if any(
        required not in positions
        for anchor in anchors
        for required in (anchor - 1, anchor, anchor + FULL_TRACE[1])
    ):
        raise ValueError("refiner rollout anchor escapes development")
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
    fourier = adapter.prepare_fourier_tensors(geometry_batch)
    active = torch.ones(len(anchors), dtype=torch.bool, device=device)
    first_nonfinite: list[int | None] = [None] * len(anchors)
    completed = np.zeros(len(anchors), dtype=np.int64)
    metric_rows: list[dict[str, Any]] = []
    structure_rows: list[dict[str, Any]] = []
    intermediate_rows: list[dict[str, Any]] = []
    snapshots: dict[str, np.ndarray] = {}
    stage_names = ("initial", "post_t3", "post_t2", "post_t1", "post_t0_final")
    parent._synchronize(device)
    started = time.perf_counter()
    with torch.no_grad():
        for horizon in range(FULL_TRACE[0], FULL_TRACE[1] + 1):
            step = adapter.refine(
                previous,
                current,
                geometry_batch,
                fourier_tensors=fourier,
            )
            candidate = step.next_state
            finite = torch.all(torch.isfinite(candidate), dim=(1, 2))
            newly_failed = active & ~finite
            for offset in (
                torch.nonzero(newly_failed, as_tuple=False).flatten().tolist()
            ):
                first_nonfinite[offset] = horizon
            retained = active & finite
            candidate_np = candidate.detach().cpu().numpy().astype(np.float64)
            intermediate_np = (
                [
                    (current + normalization.decode_residual(value))
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float64)
                    for value in step.intermediate_candidates
                ]
                if horizon in HORIZONS
                else None
            )
            for offset, anchor in enumerate(anchors):
                prediction = candidate_np[offset] if bool(retained[offset]) else None
                target = np.asarray(
                    states[positions[anchor + horizon]], dtype=np.float64
                )
                metrics = parent._state_metrics(
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
                    "arm": deployment.arm,
                    "seed": deployment.seed,
                    "deployment": deployment.deployment,
                    "sampler_seed": deployment.sampler_seed,
                    "state_stage": "raw",
                    "anchor": anchor,
                    "horizon": horizon,
                }
                parent._append_rollout_record(
                    metric_rows=metric_rows,
                    structure_rows=structure_rows,
                    identifiers=identifiers,
                    metrics=metrics,
                )
                if intermediate_np is not None:
                    for stage, values in zip(stage_names, intermediate_np, strict=True):
                        stage_prediction = values[offset]
                        stage_metrics = parent._state_metrics(
                            stage_prediction
                            if bool(active[offset])
                            and np.all(np.isfinite(stage_prediction))
                            else None,
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
                        intermediate_rows.append(
                            _refiner_intermediate_row(
                                deployment=deployment,
                                anchor=anchor,
                                horizon=horizon,
                                stage=stage,
                                metrics=stage_metrics,
                            )
                        )
                if prediction is not None:
                    completed[offset] += 1
                    if horizon in HORIZONS:
                        base = (
                            f"{deployment.arm.lower()}_seed{deployment.seed}_raw_"
                            f"a{anchor}_h{horizon}"
                        )
                        snapshots[base] = prediction.astype(np.float32)
                        snapshots.setdefault(
                            f"reference_a{anchor}_h{horizon}", target.astype(np.float64)
                        )
            active = retained
            safe_candidate = torch.where(active[:, None, None], candidate, current)
            previous, current = step.recurrent_previous, safe_candidate
    parent._synchronize(device)
    elapsed = time.perf_counter() - started
    return (
        metric_rows,
        structure_rows,
        intermediate_rows,
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
            "physical_transition_evaluations": FULL_TRACE[1] * len(anchors),
            "transition_batches": FULL_TRACE[1],
            "model_calls": 4 * FULL_TRACE[1],
            "model_sample_calls": 4 * FULL_TRACE[1] * len(anchors),
            "calls_per_physical_step": 4,
            "wall_time_seconds": elapsed,
        },
    )


def _variant_key(
    arm: str,
    seed: int,
    deployment: str,
    sampler_seed: int | None,
    state_stage: str = "raw",
) -> str:
    sampler = "none" if sampler_seed is None else str(sampler_seed)
    return f"{arm}:{seed}:{deployment}:{state_stage}:sampler_{sampler}"


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
                row.get("sampler_seed"),
            )
            for row in rows
        },
        key=lambda item: (
            ALL_ARMS.index(item[0]),
            item[1],
            {"online": 0, "ema": 1, "raw": 2, "path_projection": 3}.get(item[2], 4),
            {"raw": 0, "raw_pre_correction": 1, "corrected": 2}.get(item[3], 3),
            -1 if item[4] is None else int(item[4]),
        ),
    )
    lookup: dict[
        tuple[str, int, str, str, int | None, int, int], Mapping[str, Any]
    ] = {}
    for row in rows:
        sampler = row.get("sampler_seed")
        key = (
            str(row["arm"]),
            int(row["seed"]),
            str(row["deployment"]),
            str(row["state_stage"]),
            None if sampler in (None, "") else int(sampler),
            int(row["anchor"]),
            int(row["horizon"]),
        )
        if key in lookup:
            raise ValueError("extension rollout trace contains a duplicate key")
        lookup[key] = row
    expected = len(variants) * len(anchors) * FULL_TRACE[1]
    if len(lookup) != expected:
        raise ValueError("extension rollout trace is incomplete")
    method_rows: list[dict[str, Any]] = []
    method_result: dict[str, Any] = {}
    for arm, seed, deployment, state_stage, sampler_seed in variants:
        per_anchor: list[dict[str, Any]] = []
        for anchor in anchors:
            trace = [
                lookup[
                    (
                        arm,
                        seed,
                        deployment,
                        state_stage,
                        sampler_seed,
                        anchor,
                        horizon,
                    )
                ]
                for horizon in range(FULL_TRACE[0], FULL_TRACE[1] + 1)
            ]
            late = [
                row
                for row in trace
                if LATE_WINDOW[0] <= int(row["horizon"]) <= LATE_WINDOW[1]
            ]
            record = {
                "arm": arm,
                "seed": seed,
                "deployment": deployment,
                "sampler_seed": sampler_seed,
                "state_stage": state_stage,
                "anchor": anchor,
                "normalized_state_error_auc": parent._normalized_auc(
                    [float(row["normalized_state_error"]) for row in trace]
                ),
                "late_window_normalized_state_error_median": float(
                    np.median(
                        np.asarray(
                            [float(row["normalized_state_error"]) for row in late]
                        )
                    )
                ),
                "late_window_path_distance_median": float(
                    np.median(np.asarray([float(row["path_distance"]) for row in late]))
                ),
                "late_window_projected_path_discrepancy_median": float(
                    np.median(
                        np.asarray(
                            [float(row["projected_path_discrepancy"]) for row in late]
                        )
                    )
                ),
                "late_window_transverse_error_median": parent._valid_direction_median(
                    [
                        float(row["transverse_error"])
                        for row in late
                        if bool(row["tangent_direction_valid"])
                    ]
                ),
                "late_window_graph_dirichlet_error_energy_median": float(
                    np.median(
                        np.asarray(
                            [float(row["graph_dirichlet_error_energy"]) for row in late]
                        )
                    )
                ),
                "complete_finite_rollout": all(bool(row["finite"]) for row in trace),
            }
            method_rows.append(record)
            per_anchor.append(record)
        key = _variant_key(arm, seed, deployment, sampler_seed, state_stage)
        method_result[key] = {
            "arm": arm,
            "seed": seed,
            "deployment": deployment,
            "sampler_seed": sampler_seed,
            "state_stage": state_stage,
            "per_anchor": {
                str(record["anchor"]): _json_safe(record) for record in per_anchor
            },
            "normalized_state_error_auc": parent._aggregate(
                [float(record["normalized_state_error_auc"]) for record in per_anchor]
            ),
            "late_window_normalized_state_error_median": parent._aggregate(
                [
                    float(record["late_window_normalized_state_error_median"])
                    for record in per_anchor
                ]
            ),
            "late_window_path_distance_median": parent._aggregate(
                [
                    float(record["late_window_path_distance_median"])
                    for record in per_anchor
                ]
            ),
            "complete_finite_anchor_count": sum(
                bool(record["complete_finite_rollout"]) for record in per_anchor
            ),
        }
    method_lookup = {
        (
            row["arm"],
            row["seed"],
            row["deployment"],
            row["state_stage"],
            row["sampler_seed"],
            row["anchor"],
        ): row
        for row in method_rows
    }
    pairwise: list[dict[str, Any]] = []
    for row in method_rows:
        arm = str(row["arm"])
        deployment = str(row["deployment"])
        state_stage = str(row["state_stage"])
        if (arm == "CLEAN_EMA" and deployment == "online") or arm == "CLEAN":
            continue
        if arm == "PATH_PROJECTION" and state_stage != "corrected":
            continue
        if arm in INHERITED_COMPARISON_ARMS:
            baseline_arm = INHERITED_REFERENCE_ARM
            baseline_deployment = "raw"
        else:
            baseline_arm = "CLEAN_EMA"
            baseline_deployment = (
                "online"
                if arm == "CLEAN_EMA"
                else ("ema" if deployment == "ema" else "online")
            )
        baseline = method_lookup[
            (
                baseline_arm,
                row["seed"],
                baseline_deployment,
                "raw",
                None,
                row["anchor"],
            )
        ]
        for metric in parent.PRIMARY_METRICS:
            ratio = parent._ratio(float(row[metric]), float(baseline[metric]))
            pairwise.append(
                {
                    "arm": arm,
                    "seed": row["seed"],
                    "deployment": deployment,
                    "sampler_seed": row["sampler_seed"],
                    "state_stage": state_stage,
                    "anchor": row["anchor"],
                    "metric": metric,
                    "candidate": row[metric],
                    "matched_clean": baseline[metric],
                    "matched_clean_arm": baseline_arm,
                    "matched_clean_deployment": baseline_deployment,
                    "matched_clean_state_stage": "raw",
                    "ratio_to_matched_clean": ratio,
                    "classification": parent._ratio_classification(ratio),
                    "tie_ratio_lower_inclusive": 0.95,
                    "tie_ratio_upper_inclusive": 1.05,
                }
            )
    return method_rows, pairwise, method_result


def _offline_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    keys = sorted(
        {
            (
                str(row["arm"]),
                int(row["seed"]),
                str(row["deployment"]),
                row.get("sampler_seed"),
            )
            for row in rows
        },
        key=lambda item: (
            ALL_ARMS.index(item[0]),
            item[1],
            item[2],
            -1 if item[3] is None else int(item[3]),
        ),
    )
    for arm, seed, deployment, sampler_seed in keys:
        selected = [
            row
            for row in rows
            if row["arm"] == arm
            and int(row["seed"]) == seed
            and row["deployment"] == deployment
            and row.get("sampler_seed") == sampler_seed
        ]
        if len(selected) != len(DEVELOPMENT_ANCHORS):
            raise ValueError("offline diagnostics lack eight common anchors")
        key = _variant_key(arm, seed, deployment, sampler_seed)
        result[key] = {
            metric: parent._aggregate([float(row[metric]) for row in selected])
            for metric in parent.MEDIATOR_METRICS
            if metric != "one_prefix_response_transverse_rms"
        }
        result[key]["one_prefix_response_transverse_rms"] = (
            parent._aggregate_valid_directions(
                [float(row["one_prefix_response_transverse_rms"]) for row in selected]
            )
        )
        result[key]["stable_tangent_anchor_count"] = sum(
            bool(row["one_prefix_response_tangent_direction_valid"]) for row in selected
        )
    return result


def _refiner_sampler_variation(
    method_result: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for seed in SEEDS:
        for deployment in ("online", "ema"):
            records = [
                method_result[_variant_key(REFINER_ARM, seed, deployment, sampler)]
                for sampler in REFINER_SAMPLER_SEEDS
            ]
            metrics: dict[str, Any] = {}
            for metric in parent.PRIMARY_METRICS:
                values = np.asarray(
                    [float(record[metric]["median"]) for record in records],
                    dtype=np.float64,
                )
                metrics[metric] = {
                    "by_sampler_seed": {
                        str(sampler): _json_safe(value)
                        for sampler, value in zip(
                            REFINER_SAMPLER_SEEDS, values.tolist(), strict=True
                        )
                    },
                    "mean": float(np.mean(values)),
                    "standard_deviation_population": float(np.std(values, ddof=0)),
                    "minimum": float(np.min(values)),
                    "maximum": float(np.max(values)),
                }
            result[f"{seed}:{deployment}"] = metrics
    return result


def _training_cost(summary: Mapping[str, Any]) -> dict[str, Any]:
    cost = summary.get("model_call_cost")
    if not isinstance(cost, Mapping):
        raise TypeError("extension training summary lacks model-call cost")
    keys = (
        "training_model_calls",
        "training_model_sample_calls",
        "prefix_model_calls",
        "prefix_model_sample_calls",
        "development_online_model_calls",
        "development_online_model_sample_calls",
        "development_ema_model_calls",
        "development_ema_model_sample_calls",
    )
    if any(
        isinstance(cost.get(key), bool)
        or not isinstance(cost.get(key), int)
        or cost[key] < 0
        for key in keys
    ):
        raise ValueError("extension training model-call cost differs")
    return {key: int(cost[key]) for key in keys}


def _runtime_peak(device: torch.device) -> dict[str, int | None]:
    if device.type != "cuda":
        return {
            "cuda_peak_memory_allocated_bytes": None,
            "cuda_peak_memory_reserved_bytes": None,
        }
    return {
        "cuda_peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "cuda_peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }


def _write_final_manifest(staging: Path, *, input_manifest: Mapping[str, Any]) -> None:
    if {path.name for path in staging.iterdir()} != OUTPUT_FILES:
        raise ValueError("extension evaluation output file inventory differs")
    files = {
        name: parent._file_record(staging / name, staging)
        for name in sorted(OUTPUT_FILES)
    }
    scientific = {name: files[name] for name in sorted(SCIENTIFIC_OUTPUT_FILES)}
    _write_json(
        staging / "final_hash_manifest.json",
        {
            "schema": FINAL_MANIFEST_SCHEMA,
            "experiment_id": EXPERIMENT_ID,
            "extension_contract_sha256": EXTENSION_CONTRACT_SHA256,
            "population_role": "development",
            "input_manifest_payload_sha256": input_manifest["canonical_payload_sha256"],
            "scientific_files_sha256": _canonical_sha256(scientific),
            "runtime_manifest_excluded_from_scientific_files_sha256": True,
            "files": files,
            "self_hash_excluded": True,
            "online_solver_calls": False,
            "online_defect_trigger": False,
            "prospective_opened": False,
            "sealed_opened": False,
        },
    )


def _verify_output_packet(output: Path) -> parent.ClosedPacket:
    packet = parent._verify_closed_packet(
        output,
        final_schema=FINAL_MANIFEST_SCHEMA,
        expected_files=OUTPUT_FILES,
        label="extension evaluation output",
    )
    scientific = {name: packet.files[name] for name in sorted(SCIENTIFIC_OUTPUT_FILES)}
    if (
        packet.final.get("experiment_id") != EXPERIMENT_ID
        or packet.final.get("extension_contract_sha256") != EXTENSION_CONTRACT_SHA256
        or packet.final.get("population_role") != "development"
        or packet.final.get("scientific_files_sha256") != _canonical_sha256(scientific)
        or packet.final.get("runtime_manifest_excluded_from_scientific_files_sha256")
        is not True
        or packet.final.get("online_solver_calls") is not False
        or packet.final.get("online_defect_trigger") is not False
        or packet.final.get("prospective_opened") is not False
        or packet.final.get("sealed_opened") is not False
    ):
        raise parent.EvaluationOutputError("extension evaluation closure differs")
    input_manifest = parent._packet_json(
        packet,
        "input_manifest.json",
        schema=INPUT_MANIFEST_SCHEMA,
        label="extension evaluation input manifest",
    )
    for name, schema in (
        ("source_manifest.json", SOURCE_MANIFEST_SCHEMA),
        ("runtime_manifest.json", RUNTIME_MANIFEST_SCHEMA),
        ("result.json", RESULT_SCHEMA),
    ):
        parent._packet_json(
            packet,
            name,
            schema=schema,
            label=f"extension evaluation {name}",
        )
    if (
        packet.final.get("input_manifest_payload_sha256")
        != input_manifest["canonical_payload_sha256"]
    ):
        raise parent.EvaluationOutputError(
            "extension evaluation input-manifest binding differs"
        )
    cardinalities = {
        name: parent._csv_row_count(packet, name)
        for name in EXPECTED_CARDINALITIES
        if name.endswith(".csv")
    }
    cardinalities["rollout_snapshots.npz"] = parent._snapshot_member_count(packet)
    if cardinalities != EXPECTED_CARDINALITIES:
        raise parent.EvaluationOutputError(
            "extension evaluation result cardinalities differ"
        )
    return packet


def evaluate(arguments: argparse.Namespace) -> dict[str, Any]:
    _validate_population_role(arguments.population_role)
    source = _source_snapshot()
    _reverify_source(source)
    extension = load_extension_contract(
        arguments.extension_contract, arguments.extension_preregistration
    )
    extension_payload = extension.payload()
    trainer._validate_training_contract(extension_payload)
    evaluation_contract = extension_payload["evaluation"]
    if (
        tuple(evaluation_contract["development_anchors"]) != DEVELOPMENT_ANCHORS
        or tuple(evaluation_contract["horizons"]) != HORIZONS
        or tuple(evaluation_contract["late_window_inclusive"]) != LATE_WINDOW
        or tuple(evaluation_contract["pderefiner_inference_seeds"])
        != REFINER_SAMPLER_SEEDS
    ):
        raise ValueError("extension evaluation contract differs")
    successor, successor_sha256, baseline = parent._load_successor_contract(
        arguments.successor_contract,
        arguments.contract,
        arguments.successor_preregistration,
    )
    dataset_root = parent._safe_directory(
        arguments.dataset_dir, label="open dataset packet"
    )
    dataset_manifest, geometry, normalization, roles, dataset_sha256 = (
        parent._load_dataset(dataset_root, baseline)
    )
    if dataset_sha256 != trainer.DATASET_FINAL_SHA256:
        raise ValueError("development dataset differs from extension contract")
    if set(roles) != {"train", "development"}:
        raise ValueError("extension dataset exposes a protected role")
    train_states, train_indices, _ = roles["train"]
    development_states, development_indices, _ = roles["development"]
    parent._validate_population(
        train_states,
        train_indices,
        role="train",
        expected_frames=range(955, 1195),
        node_count=geometry.num_nodes,
    )
    parent._validate_population(
        development_states,
        development_indices,
        role="development",
        expected_frames=range(1233, 1473),
        node_count=geometry.num_nodes,
    )
    calibration_packet, calibration, projector = parent._load_calibration(
        arguments.calibration_dir,
        successor=successor,
        successor_sha256=successor_sha256,
        dataset_sha256=dataset_sha256,
        normalization=normalization,
        train_states=train_states,
        train_indices=train_indices,
    )
    if calibration_packet.final_file_sha256 != trainer.SUCCESSOR_CALIBRATION_SHA256:
        raise ValueError("successor calibration differs from extension contract")
    pilot, paired_bank = _load_solver_label_provenance(
        arguments.paired_bank_dir,
        arguments.relabel_pilot_final_hash_manifest,
        expected_coordinates=geometry.native_coordinates,
        train_states=train_states,
        train_frame_indices=train_indices,
        normalization=normalization,
    )
    packets = _load_training_packets(
        arguments.training_dir,
        dataset_sha256=dataset_sha256,
        dataset_payload_sha256=dataset_manifest["canonical_payload_sha256"],
        calibration_sha256=calibration_packet.final_file_sha256,
        calibration_payload_sha256=calibration["canonical_payload_sha256"],
        paired_bank_sha256=paired_bank.final_sha256,
        pilot_sha256=pilot.final_sha256,
    )
    inherited_baselines = _load_inherited_baseline_packets(
        arguments.inherited_clean_training_dir,
        arguments.inherited_detached_training_dir,
        successor=successor,
        successor_sha256=successor_sha256,
        dataset_sha256=dataset_sha256,
        dataset_payload_sha256=dataset_manifest["canonical_payload_sha256"],
        calibration_sha256=calibration_packet.final_file_sha256,
        calibration_payload_sha256=calibration["canonical_payload_sha256"],
        baseline=baseline,
        geometry=geometry,
    )
    deployments = _deployment_inventory(packets)
    output = arguments.output_dir.resolve()
    protected_inputs = [
        arguments.contract,
        arguments.successor_contract,
        arguments.successor_preregistration,
        arguments.extension_contract,
        arguments.extension_preregistration,
        arguments.dataset_dir,
        arguments.calibration_dir,
        arguments.paired_bank_dir,
        arguments.relabel_pilot_final_hash_manifest,
        *arguments.training_dir,
        *arguments.inherited_clean_training_dir,
        *arguments.inherited_detached_training_dir,
    ]
    parent._validate_output_disjoint(
        arguments.output_dir,
        [("extension evaluation input", path) for path in protected_inputs],
        label="extension development evaluation output",
    )
    if output.exists():
        raise FileExistsError(f"evaluation output already exists: {output}")
    device = torch.device(arguments.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    weights = parent._view_weights(geometry)
    graph_edges = parent._graph_edges(geometry)
    structure_context = parent._structure_context(geometry, train_states)
    reference_cache = parent._reference_cache(
        development_states, development_indices, projector
    )
    offline_rows: list[dict[str, Any]] = []
    rollout_rows: list[dict[str, Any]] = []
    structure_rows: list[dict[str, Any]] = []
    intermediate_rows: list[dict[str, Any]] = []
    snapshots: dict[str, np.ndarray] = {}
    cost_rows: list[dict[str, Any]] = []
    rollout_bookkeeping: dict[str, Any] = {}
    timing_bookkeeping: dict[str, Any] = {}
    refiner_noise_records: dict[tuple[int, int], dict[str, Any]] = {}
    total_started = time.perf_counter()
    for deployment in deployments:
        model = _instantiate_deployment_model(
            deployment,
            extension=extension,
            baseline=baseline,
            geometry=geometry,
            device=device,
        )
        adapter: RefinerTransitionAdapter | None = None
        transition_model: nn.Module = model
        if deployment.arm == REFINER_ARM:
            if deployment.sampler_seed is None:
                raise AssertionError("PDE-Refiner deployment lacks a sampler seed")
            adapter = RefinerTransitionAdapter(
                model,
                normalization,
                sampler_seed=deployment.sampler_seed,
                device=device,
            ).to(device)
            adapter.eval()
            transition_model = adapter
        packet_proxy = parent.TrainingPacket(
            packet=deployment.packet.packet,
            arm=deployment.arm,
            seed=deployment.seed,
            config=deployment.packet.config,
            inputs=deployment.packet.inputs,
            runtime=deployment.packet.runtime,
            summary=deployment.packet.summary,
            checkpoint_record=deployment.packet.checkpoint_record,
        )
        before_offline_calls = adapter.inner_model_calls if adapter is not None else 0
        offline_arguments = {
            "packet": packet_proxy,
            "model": transition_model,
            "states": development_states,
            "indices": development_indices,
            "anchors": DEVELOPMENT_ANCHORS,
            "geometry": geometry,
            "normalization": normalization,
            "projector": projector,
            "reference_cache": reference_cache,
            "graph_edges": graph_edges,
            "weights": weights,
            "structure_context": structure_context,
            "device": device,
        }
        if adapter is None:
            packet_offline, offline_cost = parent._offline_diagnostics(
                **offline_arguments
            )
        else:
            with adapter.paired_response_diagnostic():
                packet_offline, offline_cost = parent._offline_diagnostics(
                    **offline_arguments
                )
        packet_offline = _decorate_rows(packet_offline, deployment)
        offline_model_calls = (
            adapter.inner_model_calls - before_offline_calls
            if adapter is not None
            else offline_cost["model_invocations"]
        )
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        if adapter is None:
            raw_rows, raw_structure, raw_snapshots, raw_cost = parent._raw_rollout(
                packet=packet_proxy,
                model=transition_model,
                states=development_states,
                indices=development_indices,
                anchors=DEVELOPMENT_ANCHORS,
                geometry=geometry,
                normalization=normalization,
                projector=projector,
                reference_cache=reference_cache,
                graph_edges=graph_edges,
                weights=weights,
                structure_context=structure_context,
                device=device,
            )
            raw_rows = _decorate_rows(raw_rows, deployment)
            raw_structure = _decorate_rows(raw_structure, deployment)
            raw_cost = {
                **raw_cost,
                "physical_transition_evaluations": raw_cost[
                    "state_transition_evaluations"
                ],
                "transition_batches": raw_cost["model_invocations"],
                "model_calls": raw_cost["model_invocations"],
                "model_sample_calls": raw_cost["state_transition_evaluations"],
                "calls_per_physical_step": 1,
            }
        else:
            (
                raw_rows,
                raw_structure,
                refiner_intermediates,
                raw_snapshots,
                raw_cost,
            ) = _refiner_rollout(
                deployment=deployment,
                adapter=adapter,
                states=development_states,
                indices=development_indices,
                anchors=DEVELOPMENT_ANCHORS,
                geometry=geometry,
                normalization=normalization,
                projector=projector,
                reference_cache=reference_cache,
                graph_edges=graph_edges,
                weights=weights,
                structure_context=structure_context,
                device=device,
            )
            intermediate_rows.extend(refiner_intermediates)
        peak = _runtime_peak(device)
        offline_rows.extend(packet_offline)
        rollout_rows.extend(raw_rows)
        structure_rows.extend(raw_structure)
        parent._merge_snapshots(snapshots, _rename_snapshots(raw_snapshots, deployment))
        scientific_cost = {
            key: value for key, value in raw_cost.items() if key != "wall_time_seconds"
        }
        rollout_bookkeeping[deployment.key] = scientific_cost
        timing_bookkeeping[deployment.key] = {
            "offline_diagnostic_wall_time_seconds": offline_cost["wall_time_seconds"],
            "rollout_wall_time_seconds": raw_cost["wall_time_seconds"],
            "rollout_latency_seconds_per_transition_batch": raw_cost[
                "wall_time_seconds"
            ]
            / FULL_TRACE[1],
            "rollout_latency_seconds_per_predicted_state": raw_cost["wall_time_seconds"]
            / (FULL_TRACE[1] * len(DEVELOPMENT_ANCHORS)),
            **peak,
        }
        noise_record = adapter.noise_schedule_record() if adapter is not None else None
        if noise_record is not None:
            timing_bookkeeping[deployment.key]["noise_tape_schedule"] = noise_record
            pair_key = (deployment.seed, int(deployment.sampler_seed))
            comparable = {
                key: value
                for key, value in noise_record.items()
                if key != "schedule_sha256"
            }
            if pair_key in refiner_noise_records:
                if refiner_noise_records[pair_key] != comparable:
                    raise ValueError("online and EMA did not replay common noise tapes")
            else:
                refiner_noise_records[pair_key] = comparable
        training_cost = _training_cost(deployment.packet.summary)
        cost_rows.append(
            {
                "arm": deployment.arm,
                "seed": deployment.seed,
                "deployment": deployment.deployment,
                "sampler_seed": deployment.sampler_seed,
                **training_cost,
                "diagnostic_transition_batches": offline_cost["model_invocations"],
                "diagnostic_model_calls": offline_model_calls,
                "diagnostic_physical_transition_evaluations": offline_cost[
                    "state_transition_evaluations"
                ],
                "rollout_transition_batches": raw_cost["transition_batches"],
                "rollout_model_calls": raw_cost["model_calls"],
                "rollout_model_sample_calls": raw_cost["model_sample_calls"],
                "rollout_physical_transition_evaluations": raw_cost[
                    "physical_transition_evaluations"
                ],
                "calls_per_physical_step": raw_cost["calls_per_physical_step"],
                "noise_tape_schedule_sha256": (
                    noise_record["schedule_sha256"]
                    if noise_record is not None
                    else None
                ),
                "common_online_ema_noise_tape_replay": (
                    True if noise_record is not None else None
                ),
                "corrector_calls": 0,
                "stored_reference_bank_bytes": 0,
                "online_solver_calls": 0,
                "online_defect_trigger": False,
            }
        )
        del transition_model, model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        parent._reverify_packet(deployment.packet.packet)
    for inherited in inherited_baselines:
        packet = inherited.training
        model = parent._instantiate_model(
            packet,
            baseline=baseline,
            geometry=geometry,
            device=device,
        )
        packet_offline, offline_cost = parent._offline_diagnostics(
            packet=packet,
            model=model,
            states=development_states,
            indices=development_indices,
            anchors=DEVELOPMENT_ANCHORS,
            geometry=geometry,
            normalization=normalization,
            projector=projector,
            reference_cache=reference_cache,
            graph_edges=graph_edges,
            weights=weights,
            structure_context=structure_context,
            device=device,
        )
        packet_offline = [
            {**dict(row), "deployment": "raw", "sampler_seed": None}
            for row in packet_offline
        ]
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        packet_rollout, packet_structure, packet_snapshots, raw_cost = (
            parent._raw_rollout(
                packet=packet,
                model=model,
                states=development_states,
                indices=development_indices,
                anchors=DEVELOPMENT_ANCHORS,
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
        packet_rollout = [{**dict(row), "sampler_seed": None} for row in packet_rollout]
        packet_structure = [
            {**dict(row), "sampler_seed": None} for row in packet_structure
        ]
        offline_rows.extend(packet_offline)
        rollout_rows.extend(packet_rollout)
        structure_rows.extend(packet_structure)
        parent._merge_snapshots(snapshots, packet_snapshots)
        raw_scientific_cost = {
            **{
                key: value
                for key, value in raw_cost.items()
                if key != "wall_time_seconds"
            },
            "physical_transition_evaluations": raw_cost["state_transition_evaluations"],
            "transition_batches": raw_cost["model_invocations"],
            "model_calls": raw_cost["model_invocations"],
            "model_sample_calls": raw_cost["state_transition_evaluations"],
            "calls_per_physical_step": 1,
        }
        raw_key = f"{packet.arm}:{packet.seed}:raw:sampler_none"
        rollout_bookkeeping[raw_key] = raw_scientific_cost
        training_calls, training_wall = parent._training_cost(packet.summary)
        timing_bookkeeping[raw_key] = {
            "predictor_training_wall_time_seconds": training_wall,
            "offline_diagnostic_wall_time_seconds": offline_cost["wall_time_seconds"],
            "rollout_wall_time_seconds": raw_cost["wall_time_seconds"],
            "rollout_latency_seconds_per_transition_batch": raw_cost[
                "wall_time_seconds"
            ]
            / FULL_TRACE[1],
            "rollout_latency_seconds_per_predicted_state": raw_cost["wall_time_seconds"]
            / (FULL_TRACE[1] * len(DEVELOPMENT_ANCHORS)),
            **_runtime_peak(device),
        }
        cost_rows.append(
            {
                "arm": packet.arm,
                "seed": packet.seed,
                "deployment": "raw",
                "sampler_seed": None,
                "training_model_calls": training_calls,
                "training_model_sample_calls": None,
                "prefix_model_calls": None,
                "prefix_model_sample_calls": None,
                "development_online_model_calls": None,
                "development_online_model_sample_calls": None,
                "development_ema_model_calls": None,
                "development_ema_model_sample_calls": None,
                "diagnostic_transition_batches": offline_cost["model_invocations"],
                "diagnostic_model_calls": offline_cost["model_invocations"],
                "diagnostic_physical_transition_evaluations": offline_cost[
                    "state_transition_evaluations"
                ],
                "diagnostic_reused_from_clean_reference": False,
                "rollout_transition_batches": raw_cost["model_invocations"],
                "rollout_model_calls": raw_cost["model_invocations"],
                "rollout_model_sample_calls": raw_cost["state_transition_evaluations"],
                "rollout_physical_transition_evaluations": raw_cost[
                    "state_transition_evaluations"
                ],
                "calls_per_physical_step": 1,
                "noise_tape_schedule_sha256": None,
                "common_online_ema_noise_tape_replay": None,
                "corrector_calls": 0,
                "stored_reference_bank_bytes": 0,
                "online_solver_calls": 0,
                "online_defect_trigger": False,
            }
        )
        if packet.arm == INHERITED_REFERENCE_ARM:
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            path_rows, path_structure, path_snapshots, path_cost = (
                parent._path_projection_rollout(
                    packet=packet,
                    model=model,
                    states=development_states,
                    indices=development_indices,
                    anchors=DEVELOPMENT_ANCHORS,
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
            path_rows = [{**dict(row), "sampler_seed": None} for row in path_rows]
            path_structure = [
                {**dict(row), "sampler_seed": None} for row in path_structure
            ]
            rollout_rows.extend(path_rows)
            structure_rows.extend(path_structure)
            parent._merge_snapshots(snapshots, path_snapshots)
            path_scientific_cost = {
                **{
                    key: value
                    for key, value in path_cost.items()
                    if key != "wall_time_seconds"
                },
                "physical_transition_evaluations": path_cost[
                    "state_transition_evaluations"
                ],
                "transition_batches": path_cost["model_invocations"],
                "model_calls": path_cost["model_invocations"],
                "model_sample_calls": path_cost["state_transition_evaluations"],
                "calls_per_physical_step": 1,
            }
            path_key = f"PATH_PROJECTION:{packet.seed}:path_projection:sampler_none"
            rollout_bookkeeping[path_key] = path_scientific_cost
            timing_bookkeeping[path_key] = {
                "predictor_training_wall_time_seconds": training_wall,
                "offline_diagnostics_reused_from_clean_reference": True,
                "rollout_wall_time_seconds": path_cost["wall_time_seconds"],
                "rollout_latency_seconds_per_transition_batch": path_cost[
                    "wall_time_seconds"
                ]
                / FULL_TRACE[1],
                "rollout_latency_seconds_per_predicted_state": path_cost[
                    "wall_time_seconds"
                ]
                / (FULL_TRACE[1] * len(DEVELOPMENT_ANCHORS)),
                **_runtime_peak(device),
            }
            cost_rows.append(
                {
                    "arm": "PATH_PROJECTION",
                    "parent_arm": INHERITED_REFERENCE_ARM,
                    "seed": packet.seed,
                    "deployment": "path_projection",
                    "sampler_seed": None,
                    "training_model_calls": training_calls,
                    "training_model_sample_calls": None,
                    "prefix_model_calls": None,
                    "prefix_model_sample_calls": None,
                    "development_online_model_calls": None,
                    "development_online_model_sample_calls": None,
                    "development_ema_model_calls": None,
                    "development_ema_model_sample_calls": None,
                    "diagnostic_transition_batches": offline_cost["model_invocations"],
                    "diagnostic_model_calls": offline_cost["model_invocations"],
                    "diagnostic_physical_transition_evaluations": offline_cost[
                        "state_transition_evaluations"
                    ],
                    "diagnostic_reused_from_clean_reference": True,
                    "rollout_transition_batches": path_cost["model_invocations"],
                    "rollout_model_calls": path_cost["model_invocations"],
                    "rollout_model_sample_calls": path_cost[
                        "state_transition_evaluations"
                    ],
                    "rollout_physical_transition_evaluations": path_cost[
                        "state_transition_evaluations"
                    ],
                    "calls_per_physical_step": 1,
                    "noise_tape_schedule_sha256": None,
                    "common_online_ema_noise_tape_replay": None,
                    "corrector_calls": path_cost["corrector_calls"],
                    "stored_reference_bank_bytes": parent._reference_bank_bytes(
                        projector
                    ),
                    "online_solver_calls": 0,
                    "online_defect_trigger": False,
                }
            )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        parent._reverify_packet(packet.packet)
    if len(refiner_noise_records) != len(SEEDS) * len(REFINER_SAMPLER_SEEDS):
        raise ValueError("PDE-Refiner common-noise inventory differs")
    snapshots = _close_registered_snapshots(
        snapshots,
        rollout_rows,
        states=development_states,
        indices=development_indices,
        anchors=DEVELOPMENT_ANCHORS,
    )
    total_wall_time = time.perf_counter() - total_started
    method_rows, pairwise_rows, method_result = _summaries(
        rollout_rows, DEVELOPMENT_ANCHORS
    )
    offline_result = _offline_summary(offline_rows)
    sampler_variation = _refiner_sampler_variation(method_result)
    runtime_manifest = _self_hashed(
        {
            "schema": RUNTIME_MANIFEST_SCHEMA,
            "experiment_id": EXPERIMENT_ID,
            "population_role": "development",
            "runtime": parent._runtime(device),
            "evaluation_total_wall_time_seconds": total_wall_time,
            "timing_and_peak_memory_by_deployment": timing_bookkeeping,
            "excluded_from_scientific_payload_digest": True,
        }
    )
    input_manifest = _self_hashed(
        {
            "schema": INPUT_MANIFEST_SCHEMA,
            "experiment_id": EXPERIMENT_ID,
            "extension_contract_sha256": EXTENSION_CONTRACT_SHA256,
            "extension_preregistration_sha256": EXTENSION_PREREGISTRATION_SHA256,
            "inherited_r0_contract_sha256": trainer.R0_CONTRACT_SHA256,
            "inherited_successor_contract_sha256": successor_sha256,
            "dataset_final_hash_manifest_sha256": dataset_sha256,
            "dataset_manifest_payload_sha256": dataset_manifest[
                "canonical_payload_sha256"
            ],
            "successor_calibration_final_hash_manifest_sha256": calibration_packet.final_file_sha256,
            "successor_calibration_payload_sha256": calibration[
                "canonical_payload_sha256"
            ],
            "relabel_pilot_final_hash_manifest_sha256": pilot.final_sha256,
            "paired_bank_final_hash_manifest_sha256": paired_bank.final_sha256,
            "training_packets": [
                {
                    "arm": packet.arm,
                    "seed": packet.seed,
                    "final_hash_manifest_sha256": packet.packet.final_file_sha256,
                    "checkpoint_sha256": packet.checkpoint_record["sha256"],
                    "source_set_sha256": packet.summary["source_set_sha256"],
                }
                for packet in packets
            ],
            "inherited_baselines": {
                "required_inventory": {
                    "complete_parent_run_packets": 6,
                    "CLEAN_seeds": list(SEEDS),
                    "DETACHED_PUSHFORWARD_seeds": list(SEEDS),
                    "selected_checkpoint_slices_accepted": False,
                },
                "clean_training_packets": [
                    {
                        "arm": item.arm,
                        "seed": item.seed,
                        "parent_experiment_id": parent.EXPERIMENT_ID,
                        "parent_successor_contract_sha256": successor_sha256,
                        "final_hash_manifest_sha256": (
                            item.training.packet.final_file_sha256
                        ),
                        "checkpoint_sha256": item.training.checkpoint_record["sha256"],
                        "source_set_sha256": item.training.summary["source_set_sha256"],
                        "validation_surface": "closed_parent_evaluator",
                    }
                    for item in inherited_baselines
                    if item.arm == INHERITED_REFERENCE_ARM
                ],
                "detached_pushforward_training_packets": [
                    {
                        "arm": item.arm,
                        "seed": item.seed,
                        "parent_experiment_id": parent.EXPERIMENT_ID,
                        "parent_successor_contract_sha256": successor_sha256,
                        "final_hash_manifest_sha256": (
                            item.training.packet.final_file_sha256
                        ),
                        "checkpoint_sha256": item.training.checkpoint_record["sha256"],
                        "source_set_sha256": item.training.summary["source_set_sha256"],
                        "validation_surface": "closed_parent_evaluator",
                    }
                    for item in inherited_baselines
                    if item.arm == "DETACHED_PUSHFORWARD"
                ],
                "path_projection": {
                    "arm": "PATH_PROJECTION",
                    "predictor_arm": INHERITED_REFERENCE_ARM,
                    "predictor_deployment": "raw",
                    "predictor_source": "inherited_clean_training_packets",
                    "same_re_evaluated_clean_reference_reused": True,
                    "seeds": list(SEEDS),
                },
            },
            "extension_deployments": [
                {
                    "arm": item.arm,
                    "seed": item.seed,
                    "deployment": item.deployment,
                    "sampler_seed": item.sampler_seed,
                }
                for item in deployments
            ],
            "inherited_baseline_deployments": [
                {
                    "arm": item.arm,
                    "seed": item.seed,
                    "deployment": "raw",
                    "sampler_seed": None,
                    "source": "complete_parent_run_packet",
                }
                for item in inherited_baselines
            ]
            + [
                {
                    "arm": "PATH_PROJECTION",
                    "parent_arm": INHERITED_REFERENCE_ARM,
                    "seed": seed,
                    "deployment": "path_projection",
                    "sampler_seed": None,
                    "source": "reused_inherited_clean_model",
                }
                for seed in SEEDS
            ],
            "evaluator_source_set_sha256": source["source_set_sha256"],
            "population_role": "development",
            "anchors": list(DEVELOPMENT_ANCHORS),
            "horizons": list(HORIZONS),
            "full_trace_inclusive": list(FULL_TRACE),
            "late_window_inclusive": list(LATE_WINDOW),
            "online_solver_calls": False,
            "online_defect_trigger": False,
            "prospective_opened": False,
            "sealed_opened": False,
        }
    )
    result = _self_hashed(
        _json_safe(
            {
                "schema": RESULT_SCHEMA,
                "experiment_id": EXPERIMENT_ID,
                "status": "complete",
                "classification": "SCIENTIFIC_DEVELOPMENT_RESULT",
                "population_role": "development",
                "extension_contract_sha256": EXTENSION_CONTRACT_SHA256,
                "input_manifest_payload_sha256": input_manifest[
                    "canonical_payload_sha256"
                ],
                "seeds": list(SEEDS),
                "learned_arms": list(LEARNED_ARMS),
                "inherited_reference_arm": INHERITED_REFERENCE_ARM,
                "inherited_comparison_arms": list(INHERITED_COMPARISON_ARMS),
                "extension_deployment_count": len(deployments),
                "deployment_count": (
                    len(deployments) + len(inherited_baselines) + len(SEEDS)
                ),
                "evaluated_state_variant_count": EVALUATED_STATE_VARIANT_COUNT,
                "pderefiner_inference_seeds": list(REFINER_SAMPLER_SEEDS),
                "pderefiner_common_online_ema_noise_pair_count": len(
                    refiner_noise_records
                ),
                "offline_diagnostics": offline_result,
                "method_summaries": method_result,
                "pderefiner_sampler_variation": sampler_variation,
                "rollout_bookkeeping": rollout_bookkeeping,
                "aggregation": {
                    "sampling": "deterministic_finite_population_common_trajectory",
                    "anchors_are_not_independent_trajectories": True,
                    "iid_confidence_intervals_reported": False,
                    "q90_definition": "deterministic_nearest_rank",
                    "pairwise_tie_ratio_inclusive": [0.95, 1.05],
                    "failed_trace_primary_metric": "Infinity",
                    "refiner_sampler_variation_is_descriptive": True,
                },
                "metric_definitions": {
                    "inherited_successor_metrics": True,
                    "pderefiner_intermediate_stage": (
                        "physical candidate current plus decoded normalized residual; "
                        "reported at registered horizons; only post_t0_final enters "
                        "recurrence"
                    ),
                    "four_call_latency": (
                        "runtime-only wall clock per eight-anchor transition batch and "
                        "per predicted state"
                    ),
                    "common_clean_reference": (
                        "the re-evaluated inherited CLEAN raw deployment; "
                        "PATH_PROJECTION reuses that exact predictor checkpoint"
                    ),
                    "inherited_comparisons": (
                        "DETACHED_PUSHFORWARD and PATH_PROJECTION are re-evaluated on "
                        "the same development anchors and metrics; no scalar result "
                        "splicing is used"
                    ),
                },
                "claim_boundary": {
                    "development_only": True,
                    "solver_used_only_for_frozen_offline_training_labels": True,
                    "online_solver_calls": 0,
                    "online_defect_trigger": False,
                    "remaining_near_train_path_not_sufficient_for_accuracy": True,
                    "prospective_opened": False,
                    "sealed_opened": False,
                },
            }
        )
    )

    def reverify_inputs() -> None:
        _reverify_source(source)
        if _file_sha256(arguments.extension_contract) != EXTENSION_CONTRACT_SHA256:
            raise ValueError("extension contract changed during evaluation")
        if (
            _file_sha256(arguments.extension_preregistration)
            != EXTENSION_PREREGISTRATION_SHA256
        ):
            raise ValueError("extension preregistration changed during evaluation")
        if _file_sha256(arguments.successor_contract) != successor_sha256:
            raise ValueError("successor contract changed during evaluation")
        if _file_sha256(arguments.contract) != trainer.R0_CONTRACT_SHA256:
            raise ValueError("R0 contract changed during evaluation")
        if (
            _file_sha256(arguments.successor_preregistration)
            != trainer.SUCCESSOR_PREREGISTRATION_SHA256
        ):
            raise ValueError("successor preregistration changed during evaluation")
        if _file_sha256(dataset_root / "final_hash_manifest.json") != dataset_sha256:
            raise ValueError("development dataset changed during evaluation")
        parent._reverify_packet(calibration_packet)
        for packet in packets:
            parent._reverify_packet(packet.packet)
        for item in inherited_baselines:
            parent._reverify_packet(item.training.packet)
        observed_pilot = _verify_recursive_input_packet(
            pilot.final_path,
            schema=trainer.RELABEL_PILOT_FINAL_SCHEMA,
            label="SU2 relabel pilot packet",
        )
        observed_bank = _verify_recursive_input_packet(
            paired_bank.final_path,
            schema=trainer.PAIRED_BANK_FINAL_SCHEMA,
            label="paired displaced-state bank",
        )
        if (
            observed_pilot.final_sha256 != pilot.final_sha256
            or observed_bank.final_sha256 != paired_bank.final_sha256
        ):
            raise ValueError("solver-label provenance changed during evaluation")

    def write_packet(staging: Path) -> None:
        parent._write_csv(staging / "offline_diagnostics.csv", offline_rows)
        parent._write_csv(staging / "rollout_metrics.csv", rollout_rows)
        parent._write_csv(staging / "rollout_structure.csv", structure_rows)
        parent._write_csv(
            staging / "refiner_intermediate_diagnostics.csv", intermediate_rows
        )
        parent._write_csv(staging / "method_summary.csv", method_rows)
        parent._write_csv(staging / "pairwise_summary.csv", pairwise_rows)
        parent._write_csv(staging / "cost_metrics.csv", cost_rows)
        parent._write_deterministic_npz(staging / "rollout_snapshots.npz", snapshots)
        _write_json(staging / "source_manifest.json", source)
        _write_json(staging / "runtime_manifest.json", runtime_manifest)
        _write_json(staging / "input_manifest.json", input_manifest)
        _write_json(staging / "result.json", result)
        reverify_inputs()
        _write_final_manifest(staging, input_manifest=input_manifest)
        _verify_output_packet(staging)

    reverify_inputs()
    parent._write_staged_packet(output, write_packet)
    _verify_output_packet(output)
    reverify_inputs()
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population-role", default="development")
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--successor-contract", type=Path, required=True)
    parser.add_argument("--successor-preregistration", type=Path, required=True)
    parser.add_argument("--extension-contract", type=Path, required=True)
    parser.add_argument("--extension-preregistration", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--paired-bank-dir", type=Path, required=True)
    parser.add_argument("--relabel-pilot-final-hash-manifest", type=Path, required=True)
    parser.add_argument("--training-dir", type=Path, action="append", required=True)
    parser.add_argument(
        "--inherited-clean-training-dir",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument(
        "--inherited-detached-training-dir",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser


def _exception_chain(error: BaseException) -> list[dict[str, str]]:
    chain: list[dict[str, str]] = []
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append({"type": type(current).__name__, "message": str(current)})
        if current.__cause__ is not None:
            current = current.__cause__
        elif not current.__suppress_context__:
            current = current.__context__
        else:
            current = None
    return chain


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        result = evaluate(arguments)
    except Exception as error:  # noqa: BLE001
        chain = _exception_chain(error)
        payload: dict[str, Any] = {
            "status": "error",
            "error": str(error),
            "error_type": type(error).__name__,
            "exception_chain": chain,
        }
        if len(chain) > 1:
            payload["underlying_error"] = chain[-1]
        print(json.dumps(payload, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
