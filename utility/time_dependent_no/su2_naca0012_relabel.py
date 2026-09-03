"""Train-only SU2 restart and validation primitives for NACA relabeling.

This module is intentionally separate from the evidence-bound Stage-0 replay
utilities.  It does not execute SU2 or access a population.  Callers must bind
template restarts to the verified train trajectory before using the writer.
"""

from __future__ import annotations

import csv
import json
import math
import os
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from utility.time_dependent_no.su2_naca_trajectory import NACA_HISTORY_FIELDS
from utility.time_dependent_no.su2_native_replay import (
    NACA_NATIVE_RESTART_FIELDS,
)
from utility.time_dependent_no.su2_restart_contract import (
    NACA_DYNAMIC_FIELDS,
    SU2_FIELD_NAME_BYTES,
    SU2_HEADER_INTS,
    SU2Restart,
    read_su2_binary_restart,
    sha256_file,
)

EXPERIMENT_ID = "B3B4_NACA_CM_EXT_20260902A"
TRAIN_CENTER_RANGE = (956, 1193)
RELABEL_INPUT_STEM = "relabel_input"
RELABEL_OUTPUT_STEM = "relabel_output"
RELABEL_HISTORY_STEM = "history"
RELABEL_RECEIPT_SCHEMA = "time_dependent_no.naca_su2_relabel_receipt.v1"
RESTART_WRITE_SCHEMA = "time_dependent_no.naca_su2_restart_write.v1"
RELABEL_HISTORY_SCHEMA = "time_dependent_no.naca_su2_relabel_history.v1"
RELABEL_OUTPUT_SCHEMA = "time_dependent_no.naca_su2_relabel_output.v1"
IDEAL_GAS_GAMMA = 1.4
INNER_ITER_LIMIT = 10
MAX_FINAL_REL_RMS_DENSITY = -3.0

REQUIRED_RECEIPT_CHECKS = (
    "source_immutable",
    "restart_write_verified",
    "input_finite",
    "input_positive_density",
    "input_positive_pressure",
    "process_exit_zero",
    "exactly_one_expected_output",
    "output_finite",
    "output_positive_density",
    "output_positive_pressure",
    "convergence_relrms_density",
)

_DYNAMIC_INDICES = tuple(
    NACA_NATIVE_RESTART_FIELDS.index(field) for field in NACA_DYNAMIC_FIELDS
)
_COORDINATE_INDICES = tuple(
    NACA_NATIVE_RESTART_FIELDS.index(field) for field in ("x", "y")
)
_NON_DYNAMIC_INDICES = tuple(
    index
    for index in range(len(NACA_NATIVE_RESTART_FIELDS))
    if index not in _DYNAMIC_INDICES
)
_RMS_FIELDS = ("rms[Rho]", "rms[RhoU]", "rms[RhoV]", "rms[RhoE]", "rms[nu]")
_REL_RMS_FIELDS = (
    "relrms[Rho]",
    "relrms[RhoU]",
    "relrms[RhoV]",
    "relrms[RhoE]",
    "relrms[nu]",
)


@dataclass(frozen=True)
class RelabelStepIndex:
    """Absolute BDF2 filename and iteration alignment for one train center."""

    center: int
    previous_index: int
    current_index: int
    restart_iter: int
    time_iter: int
    expected_output_index: int

    @property
    def input_filenames(self) -> tuple[str, str]:
        return tuple(
            f"{RELABEL_INPUT_STEM}_{index:05d}.dat"
            for index in (self.previous_index, self.current_index)
        )

    @property
    def output_filename(self) -> str:
        return f"{RELABEL_OUTPUT_STEM}_{self.expected_output_index:05d}.dat"

    @property
    def history_filename(self) -> str:
        return f"{RELABEL_HISTORY_STEM}_{self.restart_iter:05d}.csv"

    def to_mapping(self) -> dict[str, Any]:
        return {
            "center": self.center,
            "history_indices": [self.previous_index, self.current_index],
            "restart_iter": self.restart_iter,
            "time_iter": self.time_iter,
            "expected_output_index": self.expected_output_index,
            "input_filenames": list(self.input_filenames),
            "output_filename": self.output_filename,
            "history_filename": self.history_filename,
        }


def relabel_step_index(center: int) -> RelabelStepIndex:
    """Map one authorized train center to the absolute SU2 BDF2 indices."""

    if isinstance(center, bool) or not isinstance(center, int):
        raise TypeError("relabel center must be an integer")
    first, last = TRAIN_CENTER_RANGE
    if center < first or center > last:
        raise PermissionError("relabel center is outside the open train population")
    return RelabelStepIndex(
        center=center,
        previous_index=center - 1,
        current_index=center,
        restart_iter=center + 1,
        time_iter=center + 2,
        expected_output_index=center + 1,
    )


def _require_canonical_step(step: RelabelStepIndex) -> None:
    if not isinstance(step, RelabelStepIndex):
        raise TypeError("step must be a RelabelStepIndex")
    if step != relabel_step_index(step.center):
        raise ValueError("step differs from the frozen train-only index mapping")


def relabel_config_overrides(center: int) -> dict[str, str]:
    """Return the frozen one-step SU2 overrides for a train center."""

    step = relabel_step_index(center)
    return {
        "RESTART_SOL": "YES",
        "RESTART_ITER": str(step.restart_iter),
        "TIME_ITER": str(step.time_iter),
        "TIME_DOMAIN": "YES",
        "TIME_MARCHING": "DUAL_TIME_STEPPING-2ND_ORDER",
        "INNER_ITER": str(INNER_ITER_LIMIT),
        "CONV_FIELD": "REL_RMS_DENSITY",
        "CONV_RESIDUAL_MINVAL": str(MAX_FINAL_REL_RMS_DENSITY),
        "SOLUTION_FILENAME": RELABEL_INPUT_STEM,
        "RESTART_FILENAME": RELABEL_OUTPUT_STEM,
        "WINDOW_CAUCHY_CRIT": "NO",
        # SU2 validates this index even when windowed Cauchy monitoring is off.
        # Preserve the upstream one-step offset from RESTART_ITER.
        "WINDOW_START_ITER": str(step.restart_iter + 1),
        "HISTORY_WRT_FREQ_INNER": "0",
        "OUTPUT_FILES": "( RESTART )",
        "OUTPUT_WRT_FREQ": "( 1 )",
        "WRT_RESTART_COMPACT": "NO",
    }


def _file_record(path: Path) -> dict[str, Any]:
    return {
        "file": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _require_absolute_regular_file(path: Path, label: str) -> Path:
    if not path.is_absolute():
        raise ValueError(f"{label} must be absolute")
    resolved = path.resolve()
    if not resolved.is_file() or path.is_symlink():
        raise ValueError(f"{label} is absent or aliased: {resolved}")
    return resolved


def _require_new_absolute_file(path: Path, label: str) -> Path:
    if not path.is_absolute():
        raise ValueError(f"{label} must be absolute")
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    parent = path.parent.resolve()
    if not parent.is_dir() or path.parent.is_symlink():
        raise ValueError(f"{label} parent is absent or aliased")
    destination = parent / path.name
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    return destination


def _float64_bits(array: np.ndarray) -> np.ndarray:
    value = np.ascontiguousarray(array, dtype="<f8")
    return value.view("<u8")


def _write_binary_payload(
    path: Path,
    *,
    header: Sequence[int],
    fields: Sequence[str],
    values: np.ndarray,
) -> None:
    if len(header) != SU2_HEADER_INTS:
        raise ValueError("SU2 restart header must contain five integers")
    if values.shape != (int(header[2]), int(header[1])):
        raise ValueError("SU2 restart values do not match the header")
    with path.open("xb") as handle:
        handle.write(struct.pack("<5i", *(int(value) for value in header)))
        for field in fields:
            encoded = field.encode("ascii")
            if not encoded or len(encoded) >= SU2_FIELD_NAME_BYTES:
                raise ValueError(f"invalid SU2 restart field name: {field!r}")
            handle.write(encoded + b"\0" * (SU2_FIELD_NAME_BYTES - len(encoded)))
        handle.write(np.ascontiguousarray(values, dtype="<f8").tobytes(order="C"))
        handle.flush()
        os.fsync(handle.fileno())


def _verify_written_restart(
    *,
    candidate_path: Path,
    template: SU2Restart,
    authoritative_state: np.ndarray,
) -> tuple[SU2Restart, list[str]]:
    candidate = read_su2_binary_restart(candidate_path)
    if candidate.header != template.header:
        raise ValueError("written restart header differs from its template")
    if candidate.fields != NACA_NATIVE_RESTART_FIELDS:
        raise ValueError("written restart field schema differs from the native schema")
    replacement64 = authoritative_state.astype(np.float64, copy=False)
    if not np.array_equal(candidate.values[:, _DYNAMIC_INDICES], replacement64):
        raise ValueError("written evolved fields differ from the authoritative state")
    if not np.array_equal(
        _float64_bits(candidate.values[:, _NON_DYNAMIC_INDICES]),
        _float64_bits(template.values[:, _NON_DYNAMIC_INDICES]),
    ):
        raise ValueError("written restart changed a non-evolved field")
    changed = [
        field
        for index, field in enumerate(candidate.fields)
        if not np.array_equal(
            _float64_bits(candidate.values[:, index]),
            _float64_bits(template.values[:, index]),
        )
    ]
    if any(field not in NACA_DYNAMIC_FIELDS for field in changed):
        raise ValueError("written restart changed a field outside the evolved state")
    return candidate, changed


def write_restart_from_verified_template(
    *,
    template_path: str | Path,
    expected_template_sha256: str,
    destination_path: str | Path,
    authoritative_state: np.ndarray,
) -> dict[str, Any]:
    """Atomically clone a native restart while replacing five evolved fields.

    ``authoritative_state`` must be the exact float32 array that the paired
    learning arms will consume.  Its float64 representation is written to SU2.
    Population authorization remains a caller duty.  The required expected
    digest must come from the caller's already-verified train manifest.
    """

    source = _require_absolute_regular_file(Path(template_path), "template_path")
    destination = _require_new_absolute_file(Path(destination_path), "destination_path")
    if source == destination:
        raise ValueError("template and destination must differ")
    if not isinstance(authoritative_state, np.ndarray):
        raise TypeError("authoritative_state must be a numpy array")
    if authoritative_state.dtype != np.float32:
        raise TypeError("authoritative_state must retain float32 dtype")
    if (
        not isinstance(expected_template_sha256, str)
        or len(expected_template_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in expected_template_sha256
        )
    ):
        raise ValueError("expected_template_sha256 must be lowercase hexadecimal")

    source_before = _file_record(source)
    if source_before["sha256"] != expected_template_sha256:
        raise ValueError("template restart SHA256 differs from verified train metadata")
    template = read_su2_binary_restart(source)
    if template.fields != NACA_NATIVE_RESTART_FIELDS or template.num_fields != 19:
        raise ValueError("template does not have the exact native 19-field schema")
    if authoritative_state.shape != (template.num_points, len(NACA_DYNAMIC_FIELDS)):
        raise ValueError("authoritative_state must have shape [num_points,5]")
    if not np.all(np.isfinite(authoritative_state)):
        raise ValueError("authoritative_state contains nonfinite values")

    values = np.array(template.values, dtype=np.float64, order="C", copy=True)
    values[:, _DYNAMIC_INDICES] = authoritative_state.astype(np.float64, copy=False)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    published = False
    try:
        _write_binary_payload(
            temporary,
            header=template.header,
            fields=template.fields,
            values=values,
        )
        candidate, changed_fields = _verify_written_restart(
            candidate_path=temporary,
            template=template,
            authoritative_state=authoritative_state,
        )
        if _file_record(source) != source_before:
            raise RuntimeError("template restart changed during the write")
        os.replace(temporary, destination)
        published = True
        final, final_changed_fields = _verify_written_restart(
            candidate_path=destination,
            template=template,
            authoritative_state=authoritative_state,
        )
        if final.sha256 != candidate.sha256 or final_changed_fields != changed_fields:
            raise RuntimeError("published restart differs from its verified temporary")
        if _file_record(source) != source_before:
            raise RuntimeError("template restart changed during publication")
    except Exception:
        if temporary.exists():
            temporary.unlink()
        if published and destination.exists():
            destination.unlink()
        raise

    state_bytes = np.ascontiguousarray(authoritative_state, dtype="<f4").tobytes(
        order="C"
    )
    return {
        "schema": RESTART_WRITE_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "template": source_before,
        "template_sha256_verified": True,
        "destination": _file_record(destination),
        "header": list(final.header),
        "fields": list(final.fields),
        "replaced_fields": list(NACA_DYNAMIC_FIELDS),
        "bitwise_changed_fields": final_changed_fields,
        "non_evolved_fields_bitwise_preserved": True,
        "authoritative_state": {
            "dtype": "float32",
            "shape": list(authoritative_state.shape),
            "little_endian_c_order_sha256": sha256(state_bytes).hexdigest(),
        },
        "source_immutable": _file_record(source) == source_before,
        "atomic_destination": True,
    }


def naca_state_admissibility(state: np.ndarray) -> dict[str, Any]:
    """Report thermodynamic admissibility and the non-gating SA sign view."""

    array = np.asarray(state)
    if array.ndim != 2 or array.shape[1] != len(NACA_DYNAMIC_FIELDS):
        raise ValueError("NACA state must have shape [num_points,5]")
    values = np.asarray(array, dtype=np.float64)
    point_count = int(values.shape[0])
    if point_count < 1:
        raise ValueError("NACA state must contain at least one point")

    finite_entries = np.isfinite(values)
    finite_rows = np.all(finite_entries, axis=1)
    density = values[:, 0]
    momentum_x = values[:, 1]
    momentum_y = values[:, 2]
    energy = values[:, 3]
    nu_tilde = values[:, 4]
    density_defined = np.isfinite(density)
    positive_density = density_defined & (density > 0.0)
    thermo_defined = (
        positive_density
        & np.isfinite(momentum_x)
        & np.isfinite(momentum_y)
        & np.isfinite(energy)
    )
    internal_energy = np.full(point_count, np.nan, dtype=np.float64)
    internal_energy[thermo_defined] = energy[thermo_defined] - (
        np.square(momentum_x[thermo_defined]) + np.square(momentum_y[thermo_defined])
    ) / (2.0 * density[thermo_defined])
    pressure = (IDEAL_GAS_GAMMA - 1.0) * internal_energy
    positive_internal_energy = np.isfinite(internal_energy) & (internal_energy > 0.0)
    positive_pressure = np.isfinite(pressure) & (pressure > 0.0)
    finite_nu = np.isfinite(nu_tilde)
    negative_nu = finite_nu & (nu_tilde < 0.0)

    def finite_minimum(value: np.ndarray) -> float | None:
        finite = value[np.isfinite(value)]
        return float(np.min(finite)) if finite.size else None

    all_finite = bool(np.all(finite_entries))
    density_passed = bool(np.all(positive_density))
    internal_energy_passed = bool(np.all(positive_internal_energy))
    pressure_passed = bool(np.all(positive_pressure))
    return {
        "point_count": point_count,
        "entry_count": int(values.size),
        "finite": {
            "passed": all_finite,
            "nonfinite_entry_count": int(
                values.size - np.count_nonzero(finite_entries)
            ),
            "nonfinite_point_count": int(point_count - np.count_nonzero(finite_rows)),
        },
        "density": {
            "passed": density_passed,
            "minimum_finite": finite_minimum(density),
            "nonpositive_count": int(
                np.count_nonzero(density_defined & (density <= 0.0))
            ),
            "undefined_count": int(point_count - np.count_nonzero(density_defined)),
        },
        "internal_energy_density": {
            "passed": internal_energy_passed,
            "minimum_finite": finite_minimum(internal_energy),
            "nonpositive_count": int(
                np.count_nonzero(
                    np.isfinite(internal_energy) & (internal_energy <= 0.0)
                )
            ),
            "undefined_count": int(
                point_count - np.count_nonzero(np.isfinite(internal_energy))
            ),
        },
        "ideal_gas_pressure": {
            "gamma": IDEAL_GAS_GAMMA,
            "passed": pressure_passed,
            "minimum_finite": finite_minimum(pressure),
            "nonpositive_count": int(
                np.count_nonzero(np.isfinite(pressure) & (pressure <= 0.0))
            ),
            "undefined_count": int(
                point_count - np.count_nonzero(np.isfinite(pressure))
            ),
        },
        "nu_tilde": {
            "acceptance_gate": False,
            "policy": "report_without_clip_mask_or_rejection",
            "minimum_finite": finite_minimum(nu_tilde),
            "negative_count": int(np.count_nonzero(negative_nu)),
            "negative_fraction": float(np.count_nonzero(negative_nu) / point_count),
            "undefined_count": int(point_count - np.count_nonzero(finite_nu)),
        },
        "thermodynamic_gate_passed": bool(
            all_finite and density_passed and internal_energy_passed and pressure_passed
        ),
    }


def _parse_history_row(path: Path) -> dict[str, float | int]:
    with path.open("r", encoding="utf-8", errors="strict", newline="") as handle:
        rows = csv.reader(handle, skipinitialspace=True)
        try:
            header = tuple(value.strip() for value in next(rows))
        except StopIteration as error:
            raise ValueError("relabel history is empty") from error
        if header != NACA_HISTORY_FIELDS:
            raise ValueError("relabel history has an unsupported field schema")
        data = list(rows)
    if len(data) != 1:
        raise ValueError("relabel history must contain exactly one final row")
    raw = [value.strip() for value in data[0]]
    if len(raw) != len(header):
        raise ValueError("relabel history final row has the wrong width")
    numeric: list[float] = []
    for value in raw:
        try:
            parsed = float(value)
        except ValueError as error:
            raise ValueError("relabel history final row is nonnumeric") from error
        if not math.isfinite(parsed):
            raise ValueError("relabel history final row is nonfinite")
        numeric.append(parsed)
    if not numeric[0].is_integer() or not numeric[1].is_integer():
        raise ValueError("relabel history iteration columns must be integers")
    return {
        field: int(value) if offset < 2 else value
        for offset, (field, value) in enumerate(zip(header, numeric, strict=True))
    }


def validate_relabel_history(
    path: str | Path, *, step: RelabelStepIndex
) -> dict[str, Any]:
    """Validate the sole final SU2 row and the frozen density convergence gate."""

    _require_canonical_step(step)
    history = _require_absolute_regular_file(Path(path), "history path")
    if history.name != step.history_filename:
        raise ValueError("relabel history filename differs from the absolute index")
    row = _parse_history_row(history)
    if row["Time_Iter"] != step.expected_output_index:
        raise ValueError("relabel history time index differs from the expected output")
    inner_iter = int(row["Inner_Iter"])
    if inner_iter < 0 or inner_iter >= INNER_ITER_LIMIT:
        raise ValueError("relabel history inner iteration is outside the frozen limit")
    final_rel_rms_density = float(row["relrms[Rho]"])
    if final_rel_rms_density > MAX_FINAL_REL_RMS_DENSITY:
        raise ValueError("relabel density residual did not meet the frozen threshold")
    return {
        "schema": RELABEL_HISTORY_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        **_file_record(history),
        "time_iter": int(row["Time_Iter"]),
        "inner_iter": inner_iter,
        "inner_iter_limit": INNER_ITER_LIMIT,
        "rms": {field: float(row[field]) for field in _RMS_FIELDS},
        "relative_rms": {field: float(row[field]) for field in _REL_RMS_FIELDS},
        "convergence": {
            "field": "relrms[Rho]",
            "threshold_lte": MAX_FINAL_REL_RMS_DENSITY,
            "observed": final_rel_rms_density,
            "passed": True,
        },
    }


def validate_relabel_output(
    path: str | Path,
    *,
    step: RelabelStepIndex,
    expected_coordinates: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Validate one expected native successor and its thermodynamic gate."""

    _require_canonical_step(step)
    output = _require_absolute_regular_file(Path(path), "output path")
    if output.name != step.output_filename:
        raise ValueError("relabel output filename differs from the absolute index")
    restart = read_su2_binary_restart(output)
    if restart.fields != NACA_NATIVE_RESTART_FIELDS or restart.num_fields != 19:
        raise ValueError("relabel output does not have the native 19-field schema")
    coordinates = np.asarray(expected_coordinates)
    if coordinates.dtype != np.float64 or coordinates.shape != (restart.num_points, 2):
        raise ValueError("expected_coordinates must retain float64 shape [N,2]")
    observed_coordinates = restart.values[:, _COORDINATE_INDICES]
    if not np.array_equal(
        _float64_bits(observed_coordinates), _float64_bits(coordinates)
    ):
        raise ValueError("relabel output coordinates differ from the fixed mesh")
    dynamic = np.array(
        restart.values[:, _DYNAMIC_INDICES], dtype=np.float64, order="C", copy=True
    )
    admissibility = naca_state_admissibility(dynamic)
    if not admissibility["thermodynamic_gate_passed"]:
        raise ValueError("relabel output failed thermodynamic admissibility")
    dynamic_bytes = np.ascontiguousarray(dynamic, dtype="<f8").tobytes(order="C")
    return dynamic, {
        "schema": RELABEL_OUTPUT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        **_file_record(output),
        "index": step.expected_output_index,
        "header": list(restart.header),
        "fields": list(restart.fields),
        "dynamic_state_little_endian_float64_sha256": sha256(dynamic_bytes).hexdigest(),
        "admissibility": admissibility,
        "passed": True,
    }


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_fail_closed_receipt(
    path: str | Path,
    *,
    center: int,
    checks: Mapping[str, bool],
    details: Mapping[str, Any] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    """Atomically write a self-hashed receipt whose success is derived only."""

    destination = _require_new_absolute_file(Path(path), "receipt path")
    step = relabel_step_index(center)
    if not isinstance(checks, Mapping):
        raise TypeError("checks must be a mapping")
    expected = set(REQUIRED_RECEIPT_CHECKS)
    observed = set(checks)
    if observed != expected:
        raise ValueError("receipt checks differ from the frozen required set")
    normalized_checks: dict[str, bool] = {}
    for name in REQUIRED_RECEIPT_CHECKS:
        value = checks[name]
        if not isinstance(value, bool):
            raise TypeError(f"receipt check {name} must be boolean")
        normalized_checks[name] = value
    if error is not None and (not isinstance(error, str) or not error.strip()):
        raise ValueError("receipt error must be absent or a nonempty string")
    if details is not None and not isinstance(details, Mapping):
        raise TypeError("receipt details must be a mapping")
    normalized_details = {} if details is None else dict(details)
    all_checks_pass = all(normalized_checks.values())
    succeeded = bool(all_checks_pass and error is None)
    failure = error
    if not succeeded and failure is None:
        failure = "one or more frozen validation checks failed"
    payload: dict[str, Any] = {
        "schema": RELABEL_RECEIPT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "status": "validation_succeeded" if succeeded else "validation_failed",
        "scientifically_usable": succeeded,
        "error": failure,
        "index_contract": step.to_mapping(),
        "checks": normalized_checks,
        "details": normalized_details,
        "nu_tilde_acceptance_gate": False,
        "prospective_opened": False,
        "sealed_opened": False,
    }
    payload["canonical_payload_sha256"] = sha256(
        _canonical_json_bytes(payload)
    ).hexdigest()
    _atomic_json(destination, payload)
    return payload


def load_verified_relabel_receipt(path: str | Path) -> dict[str, Any]:
    """Load a receipt and rederive its hash and fail-closed status."""

    receipt = _require_absolute_regular_file(Path(path), "receipt path")
    payload = json.loads(receipt.read_text(encoding="utf-8", errors="strict"))
    if not isinstance(payload, dict) or payload.get("schema") != RELABEL_RECEIPT_SCHEMA:
        raise ValueError("unsupported relabel receipt schema")
    observed_hash = payload.pop("canonical_payload_sha256", None)
    expected_hash = sha256(_canonical_json_bytes(payload)).hexdigest()
    if observed_hash != expected_hash:
        raise ValueError("relabel receipt canonical payload SHA256 differs")
    if payload.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("relabel receipt experiment identity differs")
    index_contract = payload.get("index_contract")
    if not isinstance(index_contract, dict):
        raise TypeError("relabel receipt index contract must be a mapping")
    center = index_contract.get("center")
    if isinstance(center, bool) or not isinstance(center, int):
        raise TypeError("relabel receipt center must be an integer")
    if index_contract != relabel_step_index(center).to_mapping():
        raise ValueError(
            "relabel receipt index contract differs from the frozen mapping"
        )
    checks = payload.get("checks")
    if not isinstance(checks, dict) or set(checks) != set(REQUIRED_RECEIPT_CHECKS):
        raise ValueError("relabel receipt checks differ from the frozen set")
    if any(not isinstance(checks[name], bool) for name in REQUIRED_RECEIPT_CHECKS):
        raise TypeError("relabel receipt contains a nonboolean check")
    expected_success = all(checks.values()) and payload.get("error") is None
    if payload.get("scientifically_usable") is not expected_success:
        raise ValueError("relabel receipt usability is not fail-closed")
    expected_status = (
        "validation_succeeded" if expected_success else "validation_failed"
    )
    if payload.get("status") != expected_status:
        raise ValueError("relabel receipt status differs from its checks")
    if payload.get("nu_tilde_acceptance_gate") is not False:
        raise ValueError("relabel receipt incorrectly gates on Nu_Tilde sign")
    if payload.get("prospective_opened") is not False:
        raise PermissionError("relabel receipt claims prospective access")
    if payload.get("sealed_opened") is not False:
        raise PermissionError("relabel receipt claims sealed access")
    if not isinstance(payload.get("details"), dict):
        raise TypeError("relabel receipt details must be a mapping")
    payload["canonical_payload_sha256"] = observed_hash
    return payload
