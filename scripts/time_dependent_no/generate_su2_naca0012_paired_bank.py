"""Generate the frozen train-only SU2 NACA0012 paired query bank.

The successful eleven-call pilot is an immutable prerequisite.  This entry
point then generates all primary directions before any solver call, performs
only deterministic thermodynamic-admissibility redraws, and executes the
registered 476 center/sign continuations in ascending center order.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from scripts.time_dependent_no import run_su2_naca0012_relabel_pilot as pilot
from scripts.time_dependent_no import train_pcno_naca0012 as r0_trainer
from utility.time_dependent_no.pcno_naca0012 import (
    load_naca_baseline_contract,
)
from utility.time_dependent_no.su2_naca0012_relabel import (
    EXPERIMENT_ID,
    load_verified_relabel_receipt,
    naca_state_admissibility,
)
from utility.time_dependent_no.su2_restart_contract import (
    NACA_DYNAMIC_FIELDS,
    sha256_file,
)

EXTENSION_CONTRACT_FILE_SHA256 = pilot.EXTENSION_CONTRACT_FILE_SHA256
EXTENSION_PREREGISTRATION_SHA256 = pilot.EXTENSION_PREREGISTRATION_SHA256
DATASET_FINAL_SHA256 = pilot.DATASET_FINAL_SHA256
PILOT_FINAL_MANIFEST_SHA256 = (
    "f55dbab47943d34f0a34ccc40f54bab2f3676faa9bee28f29b5d7a10868b5603"
)

TRAIN_CENTERS = tuple(range(956, 1194))
SIGNS = (-1, 1)
BASE_SEED = 20260902
MAX_REDRAWS_PER_CENTER = 10_000

BANK_SCHEMA = "time_dependent_no.naca_corrective_extension_paired_bank.v1"
BANK_CASE_SCHEMA = "time_dependent_no.naca_corrective_extension_paired_bank_case.v1"
BANK_FINAL_SCHEMA = (
    "time_dependent_no.naca_corrective_extension_paired_bank_final_manifest.v1"
)
BANK_ARRAY_FILE = "paired_bank.npz"
BANK_METADATA_FILE = "bank.json"
BANK_FINAL_FILE = "final_hash_manifest.json"
BANK_ARRAY_KEYS = (
    "centers",
    "signs",
    "coordinates",
    "normalized_directions",
    "direction_draw_indices",
    "redraw_counts",
    "displaced_previous",
    "displaced_current",
    "solver_future",
)
EXPECTED_PILOT_GATES = {
    "all_registered_cases",
    "all_case_receipts",
    "realized_displacement_scale",
    "zero_control",
    "repeatability",
    "auxiliary_invariance",
    "response_separation",
    "authority_immutable",
}
SOURCE_FILES = (
    "scripts/time_dependent_no/generate_su2_naca0012_paired_bank.py",
    "scripts/time_dependent_no/run_su2_naca0012_relabel_pilot.py",
    "utility/time_dependent_no/su2_naca0012_relabel.py",
    "utility/time_dependent_no/pcno_naca0012_corrective_extension.py",
)


@dataclass(frozen=True)
class VerifiedPilot:
    root: Path
    final_manifest_path: Path
    final_manifest_sha256: str
    files: Mapping[str, Mapping[str, Any]]
    receipt: Mapping[str, Any]
    centers: np.ndarray
    directions: np.ndarray


@dataclass(frozen=True)
class DirectionBank:
    values: np.ndarray
    draw_indices: np.ndarray
    redraw_counts: np.ndarray
    redraw_log: tuple[Mapping[str, Any], ...]
    initial_generator_state_sha256: str
    after_primary_generator_state_sha256: str
    final_generator_state_sha256: str
    total_draw_count: int
    realized_field_rms: np.ndarray
    realized_field_rms_ratio: np.ndarray


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return sha256(_canonical_bytes(value)).hexdigest()


def _self_hashed(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    if "canonical_payload_sha256" in result:
        raise ValueError("canonical hash must not be supplied")
    result["canonical_payload_sha256"] = _canonical_sha256(result)
    return result


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    if tuple(arrays) != BANK_ARRAY_KEYS:
        raise ValueError("paired-bank array order or membership differs")
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _array_record(value: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(value)
    return {
        "shape": list(array.shape),
        "dtype": array.dtype.str,
        "array_sha256": _array_sha256(array),
    }


def _generator_state_sha256(generator: torch.Generator) -> str:
    state = generator.get_state().detach().cpu().contiguous().numpy()
    return sha256(state.tobytes(order="C")).hexdigest()


def _file_record(path: Path, root: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"packet file is absent or aliased: {path}")
    relative = path.resolve().relative_to(root.resolve()).as_posix()
    return {
        "file": path.name,
        "relative_path": relative,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _verify_file_record(root: Path, relative: str, record: Mapping[str, Any]) -> None:
    candidate_relative = Path(relative)
    if candidate_relative.is_absolute() or ".." in candidate_relative.parts:
        raise ValueError("unsafe packet-relative path")
    candidate = root / candidate_relative
    if _file_record(candidate, root) != dict(record):
        raise ValueError(f"packet file differs: {relative}")


def _load_self_hashed(path: Path, schema: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"self-hashed record is absent or aliased: {path}")
    value = json.loads(path.read_text(encoding="utf-8", errors="strict"))
    if not isinstance(value, dict) or value.get("schema") != schema:
        raise ValueError(f"unsupported record schema: {path.name}")
    observed = value.pop("canonical_payload_sha256", None)
    if observed != _canonical_sha256(value):
        raise ValueError(f"canonical payload SHA256 differs: {path.name}")
    value["canonical_payload_sha256"] = observed
    return value


def verify_successful_pilot(final_manifest_path: str | Path) -> VerifiedPilot:
    """Rehash the complete immutable pilot and rederive its unlock decision."""

    final_path = pilot._require_absolute(
        final_manifest_path, "pilot final-hash manifest"
    )
    if final_path.is_symlink() or not final_path.is_file():
        raise ValueError("pilot final-hash manifest is absent or aliased")
    if final_path.name != pilot.PILOT_FINAL_HASH_FILENAME:
        raise ValueError("pilot final-hash manifest is misnamed")
    if sha256_file(final_path) != PILOT_FINAL_MANIFEST_SHA256:
        raise ValueError("pilot final-hash manifest differs from the unlocked R1 pilot")
    root = final_path.resolve().parent
    final = json.loads(final_path.read_text(encoding="utf-8", errors="strict"))
    files = final.get("files") if isinstance(final, Mapping) else None
    observed_files = {
        candidate.resolve().relative_to(root).as_posix()
        for candidate in root.rglob("*")
        if candidate.is_file() and candidate.name != pilot.PILOT_FINAL_HASH_FILENAME
    }
    if (
        not isinstance(final, Mapping)
        or final.get("schema") != pilot.PILOT_FINAL_HASH_SCHEMA
        or final.get("experiment_id") != EXPERIMENT_ID
        or final.get("self_hash_excluded") is not True
        or final.get("prospective_opened") is not False
        or final.get("sealed_opened") is not False
        or not isinstance(files, Mapping)
        or observed_files != set(files)
    ):
        raise ValueError("pilot final-hash manifest content differs")
    for relative, record in files.items():
        if not isinstance(relative, str) or not isinstance(record, Mapping):
            raise TypeError("pilot file record is malformed")
        _verify_file_record(root, relative, record)

    receipt = _load_self_hashed(
        root / pilot.PILOT_RECEIPT_FILENAME, pilot.PILOT_RECEIPT_SCHEMA
    )
    gates = receipt.get("gates")
    case_records = receipt.get("case_receipts")
    if (
        receipt.get("experiment_id") != EXPERIMENT_ID
        or receipt.get("status") != "pilot_succeeded"
        or receipt.get("scientifically_usable") is not True
        or receipt.get("paired_query_bank_unlocked") is not True
        or receipt.get("error") is not None
        or receipt.get("registered_case_count") != 11
        or receipt.get("attempted_solver_call_count") != 11
        or receipt.get("completed_case_receipt_count") != 11
        or receipt.get("schedule")
        != [case.to_mapping() for case in pilot.pilot_cases()]
        or not isinstance(gates, Mapping)
        or set(gates) != EXPECTED_PILOT_GATES
        or any(gates[name] is not True for name in EXPECTED_PILOT_GATES)
        or not isinstance(case_records, list)
        or len(case_records) != 11
        or receipt.get("online_solver_calls") is not False
        or receipt.get("prospective_opened") is not False
        or receipt.get("sealed_opened") is not False
    ):
        raise ValueError("pilot receipt does not unlock the paired bank")
    for expected_case, record in zip(pilot.pilot_cases(), case_records, strict=True):
        if not isinstance(record, Mapping):
            raise TypeError("pilot case receipt index is malformed")
        relative = record.get("relative_path")
        if not isinstance(relative, str):
            raise TypeError("pilot case receipt path is malformed")
        case_receipt = load_verified_relabel_receipt((root / relative).resolve())
        if (
            record.get("case") != expected_case.to_mapping()
            or record.get("status") != "validation_succeeded"
            or record.get("scientifically_usable") is not True
            or record.get("canonical_payload_sha256")
            != case_receipt["canonical_payload_sha256"]
            or case_receipt.get("status") != "validation_succeeded"
            or case_receipt.get("scientifically_usable") is not True
        ):
            raise ValueError("pilot case receipt does not validate")

    direction_index = receipt.get("directions")
    if not isinstance(direction_index, Mapping):
        raise TypeError("pilot direction index is absent")
    centers_record = direction_index.get("centers")
    directions_record = direction_index.get("normalized_directions")
    if not isinstance(centers_record, Mapping) or not isinstance(
        directions_record, Mapping
    ):
        raise TypeError("pilot direction records are malformed")
    centers_path = root / str(centers_record.get("relative_path"))
    directions_path = root / str(directions_record.get("relative_path"))
    centers = np.load(centers_path, allow_pickle=False)
    directions = np.load(directions_path, allow_pickle=False)
    if (
        centers.dtype != np.int64
        or centers.tolist() != list(pilot.PILOT_CENTERS)
        or directions.dtype != np.float32
        or directions.shape[:2] != (len(pilot.PILOT_CENTERS), 2)
        or directions.shape[-1] != len(NACA_DYNAMIC_FIELDS)
        or directions_record.get("base_seed") != BASE_SEED
        or directions_record.get("stream")
        != "one_ascending_center_stream_956_through_1193"
        or directions_record.get("little_endian_c_order_sha256")
        != pilot._array_record(directions)["little_endian_c_order_sha256"]
    ):
        raise ValueError("pilot direction arrays differ")
    centers.setflags(write=False)
    directions.setflags(write=False)
    for relative, record in files.items():
        _verify_file_record(root, relative, record)
    if sha256_file(final_path) != PILOT_FINAL_MANIFEST_SHA256:
        raise RuntimeError("pilot changed during verification")
    return VerifiedPilot(
        root=root,
        final_manifest_path=final_path.resolve(),
        final_manifest_sha256=PILOT_FINAL_MANIFEST_SHA256,
        files={str(name): dict(record) for name, record in files.items()},
        receipt=receipt,
        centers=centers,
        directions=directions,
    )


def _reverify_pilot(value: VerifiedPilot) -> bool:
    try:
        if sha256_file(value.final_manifest_path) != value.final_manifest_sha256:
            return False
        observed = {
            candidate.resolve().relative_to(value.root).as_posix()
            for candidate in value.root.rglob("*")
            if candidate.is_file() and candidate.name != pilot.PILOT_FINAL_HASH_FILENAME
        }
        if observed != set(value.files):
            return False
        for relative, record in value.files.items():
            _verify_file_record(value.root, relative, record)
    except (OSError, TypeError, ValueError):
        return False
    return True


def _validate_bank_contract(extension: Mapping[str, Any]) -> None:
    expected = {
        "gate": "all_relabeling_pilot_requirements_pass",
        "centers_inclusive": [956, 1193],
        "center_count": 238,
        "base_seed": BASE_SEED,
        "directions_per_center": 1,
        "antithetic_signs": [-1, 1],
        "input_count": 476,
        "target": "one_step_fixed_su2_bdf2_continuation",
        "solver_failure_policy": "stop_without_drop_or_replacement",
        "input_admissibility_redraw": "deterministic_and_logged_before_solver_execution",
    }
    if extension.get("paired_query_bank") != expected:
        raise ValueError("paired-query-bank contract differs")


def _load_bank_authority(**paths: Path) -> pilot.PilotAuthority:
    """Extend the audited pilot authority to every train-only native frame."""

    base = pilot._load_pilot_authority(**paths)
    _validate_bank_contract(base.extension)
    baseline_path = paths["baseline_contract_path"]
    dataset_dir = paths["dataset_dir"]
    trajectory_dir = paths["trajectory_dir"]
    baseline = load_naca_baseline_contract(baseline_path)
    dataset_manifest, _geometry, _, roles, dataset_sha256 = r0_trainer._load_dataset(
        dataset_dir, baseline
    )
    if dataset_sha256 != DATASET_FINAL_SHA256:
        raise ValueError("dataset final hash differs from the bank contract")
    train_states, train_indices, train_dataset = roles["train"]
    if (
        train_indices.tolist() != list(range(955, 1195))
        or list(train_dataset.center_indices) != list(TRAIN_CENTERS)
        or train_states.dtype != np.float64
    ):
        raise ValueError("train population differs from the bank contract")
    storage_path = trajectory_dir / "trajectory_storage_manifest.json"
    storage = pilot._load_self_hashed_json(
        storage_path, "time_dependent_no.su2_naca0012_trajectory_storage.v1"
    )
    parent_trajectory = baseline["parent_evidence"]["trajectory"]
    if (
        sha256_file(storage_path) != parent_trajectory["storage_manifest_file_sha256"]
        or storage.get("canonical_payload_sha256")
        != parent_trajectory["storage_manifest_payload_sha256"]
    ):
        raise ValueError("train trajectory storage differs from parent evidence")
    positions = {int(index): offset for offset, index in enumerate(train_indices)}
    clean_states: dict[int, np.ndarray] = {}
    native_frames: dict[int, pilot.NativeFrame] = {}
    for index in range(955, 1195):
        state = np.array(train_states[positions[index]], dtype=np.float64, copy=True)
        frame = pilot._native_frame(
            trajectory_root=trajectory_dir,
            storage=storage,
            index=index,
            expected_coordinates=base.coordinates,
            expected_state=state,
        )
        state.setflags(write=False)
        clean_states[index] = state
        native_frames[index] = frame
    generator_source = pilot._snapshot(
        REPO_ROOT / SOURCE_FILES[0], "paired-bank generator source"
    )
    snapshots = dict(base.snapshots)
    snapshots["paired_bank_generator_source"] = generator_source
    authority = replace(
        base,
        clean_states=clean_states,
        native_frames=native_frames,
        snapshots=snapshots,
        authority_summary={
            **dict(base.authority_summary),
            "train_frame_count": 240,
            "train_centers_inclusive": [956, 1193],
            "paired_bank_generator_source_sha256": generator_source.sha256,
            "dataset_manifest_payload_sha256": dataset_manifest[
                "canonical_payload_sha256"
            ],
        },
    )
    if not _reverify_bank_authority(authority):
        raise RuntimeError("bank authority changed during loading")
    return authority


def _reverify_bank_authority(authority: pilot.PilotAuthority) -> bool:
    if not pilot._reverify_authority(authority):
        return False
    try:
        for frame in authority.native_frames.values():
            if (
                frame.path.is_symlink()
                or not frame.path.is_file()
                or frame.path.stat().st_size != frame.bytes
                or sha256_file(frame.path) != frame.sha256
            ):
                return False
    except OSError:
        return False
    return True


def _draw_direction(generator: torch.Generator, num_nodes: int) -> np.ndarray:
    scale = torch.tensor(pilot.PER_FIELD_NORMALIZED_STD, dtype=torch.float32)
    value = torch.randn(
        (2, num_nodes, len(NACA_DYNAMIC_FIELDS)),
        dtype=torch.float32,
        device="cpu",
        generator=generator,
    )
    value.mul_(scale)
    return value.numpy().copy()


def generate_primary_directions(
    num_nodes: int,
) -> tuple[np.ndarray, torch.Generator, Mapping[str, str]]:
    if isinstance(num_nodes, bool) or not isinstance(num_nodes, int) or num_nodes < 1:
        raise ValueError("num_nodes must be a positive integer")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(BASE_SEED)
    initial = _generator_state_sha256(generator)
    directions = np.empty(
        (len(TRAIN_CENTERS), 2, num_nodes, len(NACA_DYNAMIC_FIELDS)),
        dtype=np.float32,
    )
    for offset, _ in enumerate(TRAIN_CENTERS):
        directions[offset] = _draw_direction(generator, num_nodes)
    return (
        directions,
        generator,
        {
            "initial": initial,
            "after_primary": _generator_state_sha256(generator),
        },
    )


def _candidate_inputs(
    authority: pilot.PilotAuthority, center: int, direction: np.ndarray, sign: int
) -> tuple[np.ndarray, np.ndarray, Mapping[str, Any]]:
    return pilot._input_states(
        authority, pilot.PilotCase(0, center, sign, "bank"), direction
    )


def _admissibility_summary(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "finite": all(report["finite"]["passed"] for report in reports),
        "positive_density": all(report["density"]["passed"] for report in reports),
        "positive_ideal_gas_pressure": all(
            report["ideal_gas_pressure"]["passed"] for report in reports
        ),
        "minimum_density": min(
            float(report["density"]["minimum_finite"]) for report in reports
        ),
        "minimum_ideal_gas_pressure": min(
            float(report["ideal_gas_pressure"]["minimum_finite"]) for report in reports
        ),
        "nu_tilde_negative_count": sum(
            int(report["nu_tilde"]["negative_count"]) for report in reports
        ),
        "nu_tilde_acceptance_gate": False,
    }


def _direction_reports(
    authority: pilot.PilotAuthority, center: int, direction: np.ndarray
) -> tuple[bool, dict[str, Any], np.ndarray, np.ndarray]:
    summaries: dict[str, Any] = {}
    rms = np.empty((2, len(NACA_DYNAMIC_FIELDS)), dtype=np.float64)
    ratios = np.empty_like(rms)
    thermodynamic = True
    for sign_offset, sign in enumerate(SIGNS):
        previous, current, displacement = _candidate_inputs(
            authority, center, direction, sign
        )
        reports = (
            naca_state_admissibility(previous),
            naca_state_admissibility(current),
        )
        summary = _admissibility_summary(reports)
        if not summary["finite"]:
            raise FloatingPointError("Gaussian direction produced a nonfinite input")
        thermodynamic = bool(
            thermodynamic
            and summary["positive_density"]
            and summary["positive_ideal_gas_pressure"]
        )
        summaries[str(sign)] = summary
        rms[sign_offset] = np.asarray(
            displacement["realized_field_rms"], dtype=np.float64
        )
        ratios[sign_offset] = np.asarray(
            displacement["realized_field_rms_ratio"], dtype=np.float64
        )
    return thermodynamic, summaries, rms, ratios


def prepare_admissible_directions(authority: pilot.PilotAuthority) -> DirectionBank:
    """Select final directions before SU2, redrawing only thermodynamic failures."""

    values, generator, states = generate_primary_directions(
        authority.coordinates.shape[0]
    )
    draw_indices = np.arange(len(TRAIN_CENTERS), dtype=np.int64)
    redraw_counts = np.zeros(len(TRAIN_CENTERS), dtype=np.int32)
    realized_rms = np.empty((2, len(TRAIN_CENTERS), 5), dtype=np.float64)
    realized_ratio = np.empty_like(realized_rms)
    redraw_log: list[Mapping[str, Any]] = []
    next_draw_index = len(TRAIN_CENTERS)
    for center_offset, center in enumerate(TRAIN_CENTERS):
        attempts: list[Mapping[str, Any]] = []
        direction = values[center_offset]
        while True:
            accepted, summaries, rms, ratios = _direction_reports(
                authority, center, direction
            )
            if accepted:
                realized_rms[:, center_offset] = rms
                realized_ratio[:, center_offset] = ratios
                break
            attempts.append(
                {
                    "draw_index": int(draw_indices[center_offset]),
                    "reason": "nonpositive_density_or_ideal_gas_pressure",
                    "signs": summaries,
                }
            )
            if redraw_counts[center_offset] >= MAX_REDRAWS_PER_CENTER:
                raise RuntimeError(
                    f"center {center} exceeded the deterministic redraw limit"
                )
            direction = _draw_direction(generator, authority.coordinates.shape[0])
            values[center_offset] = direction
            draw_indices[center_offset] = next_draw_index
            next_draw_index += 1
            redraw_counts[center_offset] += 1
        if attempts:
            redraw_log.append(
                {
                    "center": center,
                    "rejected_attempts": attempts,
                    "accepted_draw_index": int(draw_indices[center_offset]),
                    "redraw_count": int(redraw_counts[center_offset]),
                }
            )
    if not np.all(np.isfinite(realized_ratio)):
        raise FloatingPointError("realized displacement ratio is nonfinite")
    return DirectionBank(
        values=values,
        draw_indices=draw_indices,
        redraw_counts=redraw_counts,
        redraw_log=tuple(redraw_log),
        initial_generator_state_sha256=states["initial"],
        after_primary_generator_state_sha256=states["after_primary"],
        final_generator_state_sha256=_generator_state_sha256(generator),
        total_draw_count=next_draw_index,
        realized_field_rms=realized_rms,
        realized_field_rms_ratio=realized_ratio,
    )


def _pilot_direction_proof(
    directions: DirectionBank, verified_pilot: VerifiedPilot
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for pilot_offset, center in enumerate(pilot.PILOT_CENTERS):
        bank_offset = center - TRAIN_CENTERS[0]
        bank_value = np.ascontiguousarray(directions.values[bank_offset])
        pilot_value = np.ascontiguousarray(verified_pilot.directions[pilot_offset])
        equal = bool(
            np.array_equal(bank_value.view(np.uint32), pilot_value.view(np.uint32))
        )
        no_redraw = bool(directions.redraw_counts[bank_offset] == 0)
        records.append(
            {
                "center": center,
                "bank_center_offset": bank_offset,
                "bank_draw_index": int(directions.draw_indices[bank_offset]),
                "redraw_count": int(directions.redraw_counts[bank_offset]),
                "bank_direction_array_sha256": _array_sha256(bank_value),
                "pilot_direction_array_sha256": _array_sha256(pilot_value),
                "bitwise_equal": equal,
                "used_primary_draw_without_redraw": no_redraw,
            }
        )
    passed = all(
        record["bitwise_equal"] and record["used_primary_draw_without_redraw"]
        for record in records
    )
    if not passed:
        raise ValueError("paired-bank directions do not reproduce the pilot selections")
    return {"passed": True, "records": records}


def _case_index(ordinal: int, center: int, sign: int) -> dict[str, Any]:
    sign_offset = SIGNS.index(sign)
    center_offset = center - TRAIN_CENTERS[0]
    return {
        "ordinal": ordinal,
        "center": center,
        "sign": sign,
        "center_offset": center_offset,
        "sign_offset": sign_offset,
        "history_indices": [center - 1, center],
        "expected_output_index": center + 1,
    }


def _write_case_receipt(
    *,
    path: Path,
    execution: pilot.CaseExecution,
    direction_bank: DirectionBank,
    center_offset: int,
    sign_offset: int,
    previous: np.ndarray,
    current: np.ndarray,
    future: np.ndarray | None,
) -> dict[str, Any]:
    case = execution.case
    source_receipt = dict(execution.receipt)
    arrays: dict[str, Any] = {
        "normalized_direction": _array_record(direction_bank.values[center_offset]),
        "displaced_previous": _array_record(previous),
        "displaced_current": _array_record(current),
    }
    if future is not None:
        arrays["solver_future"] = _array_record(future)
    record = _self_hashed(
        {
            "schema": BANK_CASE_SCHEMA,
            "experiment_id": EXPERIMENT_ID,
            "status": source_receipt["status"],
            "scientifically_usable": source_receipt["scientifically_usable"],
            "case": _case_index(case.ordinal, case.center, case.sign),
            "direction": {
                "source_draw_index": int(direction_bank.draw_indices[center_offset]),
                "redraw_count": int(direction_bank.redraw_counts[center_offset]),
                "redraw_occurred_before_any_solver_execution": bool(
                    direction_bank.redraw_counts[center_offset] > 0
                ),
            },
            "array_slices": arrays,
            "source_solver_receipt_schema": source_receipt["schema"],
            "source_solver_receipt_canonical_payload_sha256": source_receipt[
                "canonical_payload_sha256"
            ],
            "source_solver_receipt": source_receipt,
            "offline_training_label_solver_call": True,
            "online_solver_calls": False,
            "nu_tilde_acceptance_gate": False,
            "prospective_opened": False,
            "sealed_opened": False,
        }
    )
    _atomic_json(path, record)
    return record


def _source_records() -> dict[str, Mapping[str, Any]]:
    return {
        relative: pilot._snapshot(REPO_ROOT / relative, relative).to_mapping()
        for relative in SOURCE_FILES
    }


def _write_final_manifest(
    root: Path, *, status: str, pilot_sha256: str
) -> tuple[dict[str, Any], str]:
    files = {
        path.resolve().relative_to(root.resolve()).as_posix(): _file_record(path, root)
        for path in sorted(root.rglob("*"), key=lambda item: item.as_posix())
        if path.is_file() and path.name != BANK_FINAL_FILE
    }
    manifest = {
        "schema": BANK_FINAL_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "packet_role": "paired_query_bank",
        "status": status,
        "extension_contract_sha256": EXTENSION_CONTRACT_FILE_SHA256,
        "preregistration_sha256": EXTENSION_PREREGISTRATION_SHA256,
        "dataset_final_hash_manifest_sha256": DATASET_FINAL_SHA256,
        "pilot_final_hash_manifest_sha256": pilot_sha256,
        "files": files,
        "self_hash_excluded": True,
        "online_solver_calls": False,
        "prospective_opened": False,
        "sealed_opened": False,
    }
    path = root / BANK_FINAL_FILE
    _atomic_json(path, manifest)
    for relative, record in files.items():
        _verify_file_record(root, relative, record)
    return manifest, sha256_file(path)


def _publish(staging: Path, output: Path) -> None:
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    os.replace(staging, output)


def generate_su2_naca0012_paired_bank(
    *,
    extension_contract_path: str | Path,
    extension_preregistration_path: str | Path,
    successor_contract_path: str | Path,
    baseline_contract_path: str | Path,
    dataset_dir: str | Path,
    calibration_dir: str | Path,
    trajectory_dir: str | Path,
    resource_manifest_path: str | Path,
    resource_dir: str | Path,
    executable_path: str | Path,
    pilot_final_manifest_path: str | Path,
    output_dir: str | Path,
    timeout_seconds: float = 3600.0,
    process_runner: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Execute the exact 476-call train-only bank and close a fail-closed packet."""

    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0.0:
        raise ValueError("timeout_seconds must be finite and positive")
    paths = {
        "extension_contract_path": pilot._require_absolute(
            extension_contract_path, "extension_contract_path"
        ),
        "extension_preregistration_path": pilot._require_absolute(
            extension_preregistration_path, "extension_preregistration_path"
        ),
        "successor_contract_path": pilot._require_absolute(
            successor_contract_path, "successor_contract_path"
        ),
        "baseline_contract_path": pilot._require_absolute(
            baseline_contract_path, "baseline_contract_path"
        ),
        "dataset_dir": pilot._require_absolute(dataset_dir, "dataset_dir"),
        "calibration_dir": pilot._require_absolute(calibration_dir, "calibration_dir"),
        "trajectory_dir": pilot._require_absolute(trajectory_dir, "trajectory_dir"),
        "resource_manifest_path": pilot._require_absolute(
            resource_manifest_path, "resource_manifest_path"
        ),
        "resource_dir": pilot._require_absolute(resource_dir, "resource_dir"),
        "executable_path": pilot._require_absolute(executable_path, "executable_path"),
    }
    output = pilot._require_absolute(output_dir, "output_dir").resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    if not output.parent.is_dir() or output.parent.is_symlink():
        raise ValueError("output parent is absent or aliased")

    verified_pilot = verify_successful_pilot(pilot_final_manifest_path)
    authority = _load_bank_authority(**paths)
    if verified_pilot.directions.shape[2] != authority.coordinates.shape[0]:
        raise ValueError("pilot and bank node counts differ")
    direction_bank = prepare_admissible_directions(authority)
    pilot_proof = _pilot_direction_proof(direction_bank, verified_pilot)
    if not _reverify_pilot(verified_pilot):
        raise RuntimeError("verified pilot changed before solver execution")
    if not _reverify_bank_authority(authority):
        raise RuntimeError("verified bank authority changed before solver execution")

    staging = output.with_name(f".{output.name}.{uuid4().hex}.staging")
    staging.mkdir()
    case_receipt_root = staging / "case_receipts"
    case_receipt_root.mkdir()
    work_root = staging / "_solver_work"
    work_root.mkdir()
    runner = process_runner or subprocess.run
    attempted_solver_call_count = 0

    def counted_runner(*args: Any, **kwargs: Any) -> Any:
        nonlocal attempted_solver_call_count
        attempted_solver_call_count += 1
        return runner(*args, **kwargs)

    num_nodes = authority.coordinates.shape[0]
    state_shape = (len(SIGNS), len(TRAIN_CENTERS), num_nodes, 5)
    displaced_previous = np.empty(state_shape, dtype=np.float32)
    displaced_current = np.empty(state_shape, dtype=np.float32)
    solver_future = np.empty(state_shape, dtype=np.float32)
    case_records: list[dict[str, Any]] = []
    error_message: str | None = None
    completed = 0
    for center_offset, center in enumerate(TRAIN_CENTERS):
        for sign_offset, sign in enumerate(SIGNS):
            ordinal = center_offset * len(SIGNS) + sign_offset
            case = pilot.PilotCase(ordinal, center, sign, "bank")
            previous, current, _ = _candidate_inputs(
                authority, center, direction_bank.values[center_offset], sign
            )
            displaced_previous[sign_offset, center_offset] = previous
            displaced_current[sign_offset, center_offset] = current
            try:
                execution = pilot._run_case(
                    staging_root=work_root,
                    authority=authority,
                    case=case,
                    direction=direction_bank.values[center_offset],
                    timeout_seconds=timeout_seconds,
                    process_runner=counted_runner,
                )
                future = (
                    np.asarray(execution.output_state, dtype=np.float32)
                    if execution.output_state is not None
                    else None
                )
                if future is not None:
                    solver_future[sign_offset, center_offset] = future
                receipt_path = case_receipt_root / f"case_{ordinal:05d}.json"
                wrapper = _write_case_receipt(
                    path=receipt_path,
                    execution=execution,
                    direction_bank=direction_bank,
                    center_offset=center_offset,
                    sign_offset=sign_offset,
                    previous=previous,
                    current=current,
                    future=future,
                )
                completed += 1
                case_records.append(
                    {
                        "relative_path": receipt_path.relative_to(staging).as_posix(),
                        "ordinal": ordinal,
                        "center": center,
                        "sign": sign,
                        "status": wrapper["status"],
                        "scientifically_usable": wrapper["scientifically_usable"],
                        "canonical_payload_sha256": wrapper["canonical_payload_sha256"],
                        "source_solver_receipt_canonical_payload_sha256": wrapper[
                            "source_solver_receipt_canonical_payload_sha256"
                        ],
                    }
                )
                if wrapper["scientifically_usable"] is not True:
                    error_message = f"registered case {case.name} failed; remaining cases were not run"
                    break
                shutil.rmtree(execution.directory)
            except Exception as error:  # noqa: BLE001 - close the incomplete packet
                error_message = f"{type(error).__name__}: {error}"
                break
        if error_message is not None:
            break

    authority_immutable = _reverify_bank_authority(authority)
    pilot_immutable = _reverify_pilot(verified_pilot)
    complete = bool(
        error_message is None
        and completed == 476
        and attempted_solver_call_count == 476
        and authority_immutable
        and pilot_immutable
        and all(record["scientifically_usable"] is True for record in case_records)
    )
    arrays_record: dict[str, Any] | None = None
    realized_ratio = direction_bank.realized_field_rms_ratio
    scale_passed = bool(
        np.all(realized_ratio >= pilot.REALIZED_RMS_RATIO_RANGE[0])
        and np.all(realized_ratio <= pilot.REALIZED_RMS_RATIO_RANGE[1])
    )
    if complete and not scale_passed:
        complete = False
        error_message = (
            "realized full-bank displacement scale escaped the pilot tolerance"
        )
    if complete:
        if work_root.exists():
            work_root.rmdir()
        arrays = {
            "centers": np.arange(956, 1194, dtype=np.int64),
            "signs": np.asarray(SIGNS, dtype=np.int8),
            "coordinates": np.asarray(authority.coordinates, dtype=np.float32),
            "normalized_directions": np.asarray(
                direction_bank.values, dtype=np.float32
            ),
            "direction_draw_indices": np.asarray(
                direction_bank.draw_indices, dtype=np.int64
            ),
            "redraw_counts": np.asarray(direction_bank.redraw_counts, dtype=np.int32),
            "displaced_previous": displaced_previous,
            "displaced_current": displaced_current,
            "solver_future": solver_future,
        }
        arrays_path = staging / BANK_ARRAY_FILE
        _atomic_npz(arrays_path, arrays)
        arrays_record = {
            "file": _file_record(arrays_path, staging),
            "members": {name: _array_record(value) for name, value in arrays.items()},
        }

    status = "complete" if complete else "failed_incomplete_not_scientific_evidence"
    metadata = _self_hashed(
        {
            "schema": BANK_SCHEMA,
            "status": status,
            "scientifically_usable": complete,
            "error": error_message,
            "experiment_id": EXPERIMENT_ID,
            "packet_role": "paired_query_bank",
            "extension_contract_sha256": EXTENSION_CONTRACT_FILE_SHA256,
            "preregistration_sha256": EXTENSION_PREREGISTRATION_SHA256,
            "dataset_final_hash_manifest_sha256": DATASET_FINAL_SHA256,
            "pilot_final_hash_manifest_sha256": verified_pilot.final_manifest_sha256,
            "population_role": "train_only",
            "centers_inclusive": [956, 1193],
            "center_count": 238,
            "base_seed": BASE_SEED,
            "directions_per_center": 1,
            "antithetic_signs": [-1, 1],
            "input_count": 476,
            "registered_solver_call_count": 476,
            "attempted_solver_call_count": attempted_solver_call_count,
            "completed_case_receipt_count": completed,
            "all_solver_cases_succeeded": complete,
            "pilot_gate_passed": True,
            "pilot_validation": {
                "status": verified_pilot.receipt["status"],
                "scientifically_usable": verified_pilot.receipt[
                    "scientifically_usable"
                ],
                "paired_query_bank_unlocked": verified_pilot.receipt[
                    "paired_query_bank_unlocked"
                ],
                "gates": dict(verified_pilot.receipt["gates"]),
                "registered_case_count": verified_pilot.receipt[
                    "registered_case_count"
                ],
                "case_statuses": [
                    record["status"]
                    for record in verified_pilot.receipt["case_receipts"]
                ],
                "final_hash_manifest_sha256": verified_pilot.final_manifest_sha256,
                "immutable_after_bank": pilot_immutable,
            },
            "state_schema": {
                "dynamic_fields": list(NACA_DYNAMIC_FIELDS),
                "history_slot_order": ["previous", "current"],
                "history_indices": "center_minus_1_and_center",
                "expected_output_index": "center_plus_1",
                "state_coordinate_system": (
                    "raw_physical_conservative_variables_model_facing_float32"
                ),
                "direction_coordinate_system": "state_normalized",
                "coordinate_row_order": "native_su2_restart_row_order",
                "coordinate_dtype": "float32",
                "state_axes": ["sign", "center", "node", "field"],
                "direction_axes": ["center", "history_slot", "node", "field"],
            },
            "direction_generation": {
                "library": "torch",
                "device": "cpu",
                "generator": "torch.Generator.manual_seed",
                "distribution_call": "torch.randn",
                "dtype": "float32",
                "draw_shape": [2, num_nodes, 5],
                "stream_order": "all_238_primary_draws_then_ascending_center_redraws",
                "primary_draw_count": 238,
                "total_draw_count": direction_bank.total_draw_count,
                "initial_generator_state_sha256": (
                    direction_bank.initial_generator_state_sha256
                ),
                "after_primary_generator_state_sha256": (
                    direction_bank.after_primary_generator_state_sha256
                ),
                "final_generator_state_sha256": (
                    direction_bank.final_generator_state_sha256
                ),
                "redraw_policy": (
                    "only_nonpositive_density_or_ideal_gas_pressure_before_any_solver_call"
                ),
                "redraw_limit_per_center": MAX_REDRAWS_PER_CENTER,
                "redrawn_center_count": len(direction_bank.redraw_log),
                "redraw_log": list(direction_bank.redraw_log),
                "nu_tilde_acceptance_gate": False,
                "pilot_direction_proof": pilot_proof,
            },
            "realized_displacement": {
                "registered_per_field_state_normalized_std": (
                    pilot.PER_FIELD_NORMALIZED_STD.tolist()
                ),
                "required_ratio_inclusive": list(pilot.REALIZED_RMS_RATIO_RANGE),
                "all_476_inputs_passed": scale_passed,
                "per_field_ratio_min": np.min(realized_ratio, axis=(0, 1)).tolist(),
                "per_field_ratio_max": np.max(realized_ratio, axis=(0, 1)).tolist(),
            },
            "call_schedule": {
                "order": "ascending_center_then_sign_minus_plus",
                "canonical_sha256": sha256(
                    json.dumps(
                        [[center, sign] for center in TRAIN_CENTERS for sign in SIGNS],
                        separators=(",", ":"),
                    ).encode("ascii")
                ).hexdigest(),
                "failure_policy": "stop_without_drop_or_replacement",
            },
            "case_receipts": case_records,
            "arrays": arrays_record,
            "source_files": _source_records(),
            "authority": dict(authority.authority_summary),
            "authority_immutable": authority_immutable,
            "offline_training_label_solver_calls": True,
            "online_solver_calls": False,
            "prospective_opened": False,
            "sealed_opened": False,
            "runtime": {
                "python": sys.version,
                "numpy": np.__version__,
                "torch": torch.__version__,
            },
        }
    )
    _atomic_json(staging / BANK_METADATA_FILE, metadata)
    final, final_sha256 = _write_final_manifest(
        staging, status=status, pilot_sha256=verified_pilot.final_manifest_sha256
    )
    _publish(staging, output)
    for relative, record in final["files"].items():
        _verify_file_record(output, relative, record)
    if sha256_file(output / BANK_FINAL_FILE) != final_sha256:
        raise RuntimeError("paired-bank final manifest changed during publication")
    return {
        "receipt": _load_self_hashed(output / BANK_METADATA_FILE, BANK_SCHEMA),
        "packet": {
            "directory": output.name,
            "final_hash_manifest_sha256": final_sha256,
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    docs = REPO_ROOT / "docs/time_dependent_no"
    parser.add_argument(
        "--extension-contract",
        type=Path,
        default=docs / "B3B4_NACA_CORRECTIVE_EXTENSION_CONTRACT.json",
    )
    parser.add_argument(
        "--extension-preregistration",
        type=Path,
        default=docs / "B3B4_NACA_CORRECTIVE_EXTENSION_PREREGISTRATION.md",
    )
    parser.add_argument(
        "--successor-contract",
        type=Path,
        default=docs / "B3B4_NACA_CORRECTIVE_SUCCESSOR_CONTRACT.json",
    )
    parser.add_argument(
        "--baseline-contract",
        type=Path,
        default=docs / "R0_NACA_PCNO_BASELINE_CONTRACT.json",
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--trajectory-dir", type=Path, required=True)
    parser.add_argument(
        "--resource-manifest",
        type=Path,
        default=docs / "R0_SU2_NACA_RESOURCE_MANIFEST.json",
    )
    parser.add_argument("--resource-dir", type=Path, required=True)
    parser.add_argument("--su2-executable", type=Path, required=True)
    parser.add_argument("--pilot-final-hash-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=3600.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    try:
        result = generate_su2_naca0012_paired_bank(
            extension_contract_path=arguments.extension_contract,
            extension_preregistration_path=arguments.extension_preregistration,
            successor_contract_path=arguments.successor_contract,
            baseline_contract_path=arguments.baseline_contract,
            dataset_dir=arguments.dataset_dir,
            calibration_dir=arguments.calibration_dir,
            trajectory_dir=arguments.trajectory_dir,
            resource_manifest_path=arguments.resource_manifest,
            resource_dir=arguments.resource_dir,
            executable_path=arguments.su2_executable,
            pilot_final_manifest_path=arguments.pilot_final_hash_manifest,
            output_dir=arguments.output_dir,
            timeout_seconds=arguments.timeout_seconds,
        )
    except (ArithmeticError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"NACA0012 paired-bank generation failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result["receipt"]["scientifically_usable"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
