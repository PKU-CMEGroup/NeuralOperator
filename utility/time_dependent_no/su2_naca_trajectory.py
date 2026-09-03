"""Fail-closed solver-only trajectory generation for the pinned NACA0012 case."""

from __future__ import annotations

import csv
import json
import math
import os
import re
import shutil
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from utility.time_dependent_no.su2_native_replay import (
    NACA_NATIVE_RESTART_FIELDS,
)
from utility.time_dependent_no.su2_restart_contract import (
    NACA_BDF2_HISTORY_INDICES,
    NACA_DYNAMIC_FIELDS,
    NACA_REPLAY_TARGET_INDEX,
    audit_unsteady_naca0012_bundle,
    parse_su2_config,
    parse_su2_mesh,
    read_su2_binary_restart,
    sha256_file,
)

NACA_TRAJECTORY_CONFIG_FILENAME = "trajectory.cfg"
NACA_TRAJECTORY_CASE_FILENAME = "trajectory_case.json"
NACA_TRAJECTORY_RECEIPT_FILENAME = "trajectory_receipt.json"
NACA_TRAJECTORY_STORAGE_FILENAME = "trajectory_storage_manifest.json"
NACA_TRAJECTORY_STDOUT_FILENAME = "trajectory_stdout.log"
NACA_TRAJECTORY_STDERR_FILENAME = "trajectory_stderr.log"
NACA_TRAJECTORY_OUTPUT_STEM = "trajectory_flow"
NACA_TRAJECTORY_HISTORY_FILENAME = "history_00499.csv"
NACA_TRAJECTORY_DEFAULT_FINAL_TIME_ITER = 2000
NACA_TRAJECTORY_CASE_SCHEMA = "time_dependent_no.su2_naca0012_trajectory_case.v1"
NACA_TRAJECTORY_RECEIPT_SCHEMA = "time_dependent_no.su2_naca0012_trajectory_receipt.v1"
NACA_TRAJECTORY_STORAGE_SCHEMA = "time_dependent_no.su2_naca0012_trajectory_storage.v1"

NACA_HISTORY_FIELDS = (
    "Time_Iter",
    "Inner_Iter",
    "rms[Rho]",
    "rms[RhoU]",
    "rms[RhoV]",
    "rms[RhoE]",
    "rms[nu]",
    "RefForce",
    "CD",
    "CL",
    "CSF",
    "CMx",
    "CMy",
    "CMz",
    "CFx",
    "CFy",
    "CFz",
    "CEff",
    "Buffet",
    "relrms[Rho]",
    "relrms[RhoU]",
    "relrms[RhoV]",
    "relrms[RhoE]",
    "relrms[nu]",
    "tavg[RefForce]",
    "tavg[CD]",
    "tavg[CL]",
    "tavg[CSF]",
    "tavg[CMx]",
    "tavg[CMy]",
    "tavg[CMz]",
    "tavg[CFx]",
    "tavg[CFy]",
    "tavg[CFz]",
    "tavg[CEff]",
    "tavg[Buffet]",
    "Cauchy[tavg[CD]]",
    "Cauchy[tavg[CL]]",
)

_OUTPUT_PATTERN = re.compile(r"trajectory_flow_(?P<index>\d{5})\.dat\Z")
_VERSION_PATTERN = re.compile(r"SU2 v(?P<version>\d+\.\d+\.\d+)")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


class TrajectoryRunError(RuntimeError):
    """A solver-only trajectory run failed a preparation or validation gate."""

    def __init__(self, message: str, *, receipt_path: Path | None = None):
        super().__init__(message)
        self.receipt_path = receipt_path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _write_self_hashed_json(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    if path.exists():
        raise FileExistsError(path)
    rendered = dict(payload)
    rendered["canonical_payload_sha256"] = sha256(
        _canonical_bytes(rendered)
    ).hexdigest()
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(rendered, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return rendered


def load_verified_trajectory_receipt(path: str | Path) -> dict[str, Any]:
    """Load a trajectory receipt and verify its schema and self hash."""

    receipt_path = Path(path)
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != NACA_TRAJECTORY_RECEIPT_SCHEMA
    ):
        raise ValueError(f"{receipt_path}: unsupported trajectory receipt schema")
    observed = payload.pop("canonical_payload_sha256", None)
    expected = sha256(_canonical_bytes(payload)).hexdigest()
    if observed != expected:
        raise ValueError(f"{receipt_path}: canonical payload SHA256 differs")
    payload["canonical_payload_sha256"] = observed
    return payload


def _file_record(path: Path) -> dict[str, Any]:
    return {
        "file": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _file_record_or_error(path: Path) -> dict[str, Any] | None:
    try:
        if not path.is_file():
            return None
        return _file_record(path)
    except OSError as error:
        return {"file": path.name, "error": f"{type(error).__name__}: {error}"}


def _source_inventory() -> dict[str, dict[str, Any]]:
    repository = Path(__file__).resolve().parents[2]
    sources = (
        Path(__file__).resolve(),
        repository / "scripts/time_dependent_no/run_su2_naca0012_trajectory.py",
        repository / "utility/time_dependent_no/su2_restart_contract.py",
        repository / "utility/time_dependent_no/su2_native_replay.py",
    )
    records: dict[str, dict[str, Any]] = {}
    for source in sources:
        if not source.is_file() or source.is_symlink():
            raise FileNotFoundError(f"trajectory source is absent or aliased: {source}")
        relative = source.relative_to(repository).as_posix()
        records[relative] = {
            "bytes": source.stat().st_size,
            "sha256": sha256_file(source),
        }
    return records


def _trajectory_overrides(final_time_iter: int) -> dict[str, str]:
    if final_time_iter <= NACA_REPLAY_TARGET_INDEX:
        raise ValueError("final_time_iter must be greater than 499")
    if final_time_iter > 100000:
        raise ValueError("final_time_iter exceeds the five-digit restart index bound")
    return {
        "RESTART_SOL": "YES",
        "RESTART_ITER": "499",
        "TIME_ITER": str(final_time_iter),
        "TIME_DOMAIN": "YES",
        "TIME_MARCHING": "DUAL_TIME_STEPPING-2ND_ORDER",
        "MESH_FILENAME": "unsteady_naca0012_mesh.su2",
        "SOLUTION_FILENAME": "restart_flow",
        "RESTART_FILENAME": NACA_TRAJECTORY_OUTPUT_STEM,
        "WINDOW_CAUCHY_CRIT": "NO",
        "HISTORY_WRT_FREQ_INNER": "0",
        "OUTPUT_FILES": "( RESTART )",
        "OUTPUT_WRT_FREQ": "( 1 )",
        "WRT_RESTART_COMPACT": "NO",
    }


def _render_config(source_text: str, overrides: Mapping[str, str]) -> str:
    rendered: list[str] = []
    observed: set[str] = set()
    for raw_line in source_text.splitlines():
        active = raw_line.split("%", maxsplit=1)[0].strip()
        if "=" in active:
            key = active.split("=", maxsplit=1)[0].strip().upper()
            if key in overrides:
                if key in observed:
                    raise ValueError(f"duplicate trajectory override target {key}")
                rendered.append(f"{key}= {overrides[key]}")
                observed.add(key)
                continue
        rendered.append(raw_line)
    missing = [key for key in overrides if key not in observed]
    if missing:
        rendered.extend(("", "% Frozen solver-only trajectory overrides"))
        rendered.extend(f"{key}= {overrides[key]}" for key in missing)
    return "\n".join(rendered) + "\n"


def _authority_inventory(
    *, resource_root: Path, manifest_file: Path, manifest: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    records = {"manifest": _file_record(manifest_file)}
    resources = manifest.get("resources")
    if not isinstance(resources, dict):
        raise TypeError("resource manifest lacks resources")
    for role, record in resources.items():
        if not isinstance(record, dict):
            raise TypeError(f"resource record is not an object: {role}")
        relative = Path(str(record.get("file")))
        candidate = (resource_root / relative).resolve()
        if candidate.parent != resource_root or relative.name != str(relative):
            raise ValueError(f"unsafe resource filename: {role}")
        observed = _file_record(candidate)
        records[role] = observed
    return records


def _case_input_inventory(case_root: Path, names: Sequence[str]) -> dict[str, Any]:
    inventory: dict[str, Any] = {}
    for name in names:
        relative = Path(name)
        candidate = (case_root / relative).resolve()
        if relative.name != name or candidate.parent != case_root:
            raise ValueError(f"unsafe staged input filename: {name}")
        if not candidate.is_file() or candidate.is_symlink():
            raise FileNotFoundError(candidate)
        inventory[name] = {
            "bytes": candidate.stat().st_size,
            "sha256": sha256_file(candidate),
        }
    return inventory


def _release_contract(manifest: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    upstream = manifest.get("upstream")
    if not isinstance(upstream, dict):
        raise TypeError("resource manifest lacks upstream identity")
    release = upstream.get("target_replay_release")
    if not isinstance(release, dict) or release.get("tag") != "v8.5.0":
        raise ValueError("resource manifest does not pin SU2 v8.5.0")
    asset = release.get("windows_mpi_asset")
    if not isinstance(asset, dict):
        raise TypeError("resource manifest lacks the Windows MPI asset")
    executable = asset.get("executable")
    if not isinstance(executable, dict):
        raise TypeError("resource manifest lacks executable identity")
    digest = executable.get("sha256")
    if not isinstance(digest, str) or _SHA256_PATTERN.fullmatch(digest) is None:
        raise ValueError("resource manifest has an invalid executable SHA256")
    return release, digest


def _validate_history(path: Path, expected_indices: Sequence[int]) -> dict[str, Any]:
    with path.open("r", encoding="utf-8", errors="strict", newline="") as handle:
        rows = csv.reader(handle, skipinitialspace=True)
        try:
            header = tuple(value.strip() for value in next(rows))
        except StopIteration as error:
            raise ValueError("trajectory history is empty") from error
        if header != NACA_HISTORY_FIELDS:
            raise ValueError("trajectory history has an unsupported field schema")
        parsed: list[dict[str, float | int]] = []
        for line_number, row in enumerate(rows, start=2):
            values = [value.strip() for value in row]
            if len(values) != len(header):
                raise ValueError(
                    f"trajectory history row {line_number} has the wrong width"
                )
            numeric: list[float] = []
            for value in values:
                try:
                    number = float(value)
                except ValueError as error:
                    raise ValueError(
                        f"trajectory history row {line_number} is nonnumeric"
                    ) from error
                if not math.isfinite(number):
                    raise ValueError(
                        f"trajectory history row {line_number} is nonfinite"
                    )
                numeric.append(number)
            time_iter = numeric[0]
            inner_iter = numeric[1]
            if not time_iter.is_integer() or not inner_iter.is_integer():
                raise ValueError(
                    "trajectory history iteration columns must be integers"
                )
            parsed.append(
                {
                    name: int(value) if offset < 2 else value
                    for offset, (name, value) in enumerate(
                        zip(header, numeric, strict=True)
                    )
                }
            )
    observed_indices = [int(row["Time_Iter"]) for row in parsed]
    if observed_indices != list(expected_indices):
        raise ValueError("trajectory history indices differ from expected outputs")
    if len(set(observed_indices)) != len(observed_indices):
        raise ValueError("trajectory history contains duplicate indices")
    if any(int(row["Inner_Iter"]) < 0 for row in parsed):
        raise ValueError("trajectory history contains a negative inner iteration")
    final = parsed[-1]
    return {
        **_file_record(path),
        "schema": list(header),
        "row_count": len(parsed),
        "first_time_iter": observed_indices[0],
        "last_time_iter": observed_indices[-1],
        "final_row": {
            key: final[key]
            for key in ("Time_Iter", "Inner_Iter", "CD", "CL", "CFx", "CFy")
        },
    }


def _validate_outputs(
    *, case_root: Path, mesh_path: Path, expected_indices: Sequence[int]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    input_restart_names = sorted(
        path.name for path in case_root.glob("restart_flow_*.dat")
    )
    if "restart_flow_00499.dat" in input_restart_names:
        raise ValueError("canonical target name appeared inside the trajectory case")
    if input_restart_names != [
        "restart_flow_00497.dat",
        "restart_flow_00498.dat",
    ]:
        raise ValueError("trajectory case contains an unexpected restart_flow output")
    outputs = sorted(case_root.glob(f"{NACA_TRAJECTORY_OUTPUT_STEM}_*.dat"))
    indexed_outputs: list[tuple[int, Path]] = []
    for output in outputs:
        match = _OUTPUT_PATTERN.fullmatch(output.name)
        if match is None or output.is_symlink():
            raise ValueError(f"unsupported trajectory output filename: {output.name}")
        indexed_outputs.append((int(match.group("index")), output))
    observed_indices = [index for index, _ in indexed_outputs]
    if observed_indices != list(expected_indices):
        raise ValueError("trajectory restart indices differ from the frozen sequence")

    mesh = parse_su2_mesh(mesh_path)
    coordinate_indices = [
        NACA_NATIVE_RESTART_FIELDS.index("x"),
        NACA_NATIVE_RESTART_FIELDS.index("y"),
    ]
    dynamic_indices = [
        NACA_NATIVE_RESTART_FIELDS.index(field) for field in NACA_DYNAMIC_FIELDS
    ]
    records: list[dict[str, Any]] = []
    for index, output in indexed_outputs:
        restart = read_su2_binary_restart(output)
        if restart.fields != NACA_NATIVE_RESTART_FIELDS:
            raise ValueError(f"{output.name}: unsupported native restart schema")
        if restart.num_points != mesh.num_points:
            raise ValueError(f"{output.name}: point count differs from the mesh")
        coordinate_error = float(
            np.max(np.abs(restart.values[:, coordinate_indices] - mesh.points))
        )
        if coordinate_error > 1.0e-12:
            raise ValueError(f"{output.name}: coordinates differ from the mesh")
        dynamic = restart.values[:, dynamic_indices]
        if not np.all(np.isfinite(dynamic)):
            raise ValueError(f"{output.name}: evolved fields contain nonfinite values")
        records.append(
            {
                "index": index,
                "file": output.name,
                "bytes": output.stat().st_size,
                "sha256": restart.sha256,
                "header": list(restart.header),
                "fields": list(restart.fields),
                "coordinate_max_abs_error": coordinate_error,
                "dynamic_minimum": [float(value) for value in np.min(dynamic, axis=0)],
                "dynamic_maximum": [float(value) for value in np.max(dynamic, axis=0)],
            }
        )

    aggregate_core = [
        {
            "index": record["index"],
            "file": record["file"],
            "bytes": record["bytes"],
            "sha256": record["sha256"],
        }
        for record in records
    ]
    total_bytes = sum(int(record["bytes"]) for record in records)
    aggregate = {
        "file_count": len(records),
        "first_index": expected_indices[0],
        "last_index": expected_indices[-1],
        "total_bytes": total_bytes,
        "total_gibibytes": total_bytes / (1024**3),
        "ordered_file_records_sha256": sha256(
            _canonical_bytes({"files": aggregate_core})
        ).hexdigest(),
        "five_evolved_float64_bytes": (
            len(records) * mesh.num_points * len(NACA_DYNAMIC_FIELDS) * 8
        ),
        "bdf2_distinct_frame_semantics": {
            "staged_history_indices": list(NACA_BDF2_HISTORY_INDICES),
            "generated_indices": list(expected_indices),
            "distinct_frames": len(records) + len(NACA_BDF2_HISTORY_INDICES),
            "transition_level_frame_tripling": False,
        },
    }
    return records, aggregate


def _probe_release(
    runner: Callable[..., Any],
    executable: Path,
    case_root: Path,
    environment: Mapping[str, str],
) -> dict[str, Any]:
    completed = runner(
        [str(executable), "--help"],
        cwd=case_root,
        env=dict(environment),
        shell=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
        check=False,
    )
    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    match = _VERSION_PATTERN.search(f"{stdout}\n{stderr}")
    return {
        "argv": ["<bound-su2-executable>", "--help"],
        "return_code": completed.returncode,
        "version": match.group("version") if match else None,
        "stdout_sha256": sha256(stdout.encode("utf-8")).hexdigest(),
        "stderr_sha256": sha256(stderr.encode("utf-8")).hexdigest(),
    }


def _failure_claims() -> dict[str, bool]:
    return {
        "restart_sufficiency_claimed": False,
        "trusted_transition_claimed": False,
        "trajectory_population_claimed": False,
        "id_population_claimed": False,
        "boundary_and_force_gate_complete": False,
    }


def _finalize_preparation_failure(
    *, staging: Path, destination: Path, run_id: str, error: Exception
) -> Path:
    staging.mkdir(exist_ok=True)
    receipt_path = staging / NACA_TRAJECTORY_RECEIPT_FILENAME
    _write_self_hashed_json(
        receipt_path,
        {
            "schema": NACA_TRAJECTORY_RECEIPT_SCHEMA,
            "run_id": run_id,
            "status": "preparation_failed",
            "error": f"{type(error).__name__}: {error}",
            "process_attempted": False,
            "process_started": False,
            "execution_succeeded": False,
            "trajectory_validated": False,
            **_failure_claims(),
        },
    )
    staging.replace(destination)
    return destination / NACA_TRAJECTORY_RECEIPT_FILENAME


def run_su2_naca0012_trajectory(
    *,
    resource_dir: str | Path,
    manifest_path: str | Path,
    case_dir: str | Path,
    target_path: str | Path,
    executable_path: str | Path,
    expected_executable_sha256: str,
    final_time_iter: int = NACA_TRAJECTORY_DEFAULT_FINAL_TIME_ITER,
    timeout_seconds: float = 86400.0,
    process_runner: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Prepare, execute, and validate one isolated native trajectory case."""

    if timeout_seconds <= 0.0:
        raise ValueError("timeout_seconds must be positive")
    overrides = _trajectory_overrides(final_time_iter)
    expected_indices = tuple(range(NACA_REPLAY_TARGET_INDEX, final_time_iter))
    if _SHA256_PATTERN.fullmatch(expected_executable_sha256) is None:
        raise ValueError("expected_executable_sha256 must be lowercase hexadecimal")

    raw_paths = {
        "resource_dir": Path(resource_dir),
        "manifest_path": Path(manifest_path),
        "case_dir": Path(case_dir),
        "target_path": Path(target_path),
        "executable_path": Path(executable_path),
    }
    for name, path in raw_paths.items():
        if not path.is_absolute():
            raise ValueError(f"{name} must be absolute")
    resource_root = raw_paths["resource_dir"].resolve()
    manifest_file = raw_paths["manifest_path"].resolve()
    destination = raw_paths["case_dir"].resolve()
    target = raw_paths["target_path"].resolve()
    executable = raw_paths["executable_path"].resolve()
    if destination.exists():
        raise FileExistsError(destination)
    if not destination.parent.is_dir() or destination.parent.is_symlink():
        raise ValueError("trajectory case parent is absent or aliased")
    if resource_root == destination or resource_root in destination.parents:
        raise ValueError("trajectory case must remain outside the resource bundle")
    if destination in resource_root.parents:
        raise ValueError("trajectory case cannot contain the resource bundle")
    for label, path in (("target", target), ("executable", executable)):
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"{label} is absent or aliased: {path}")
        if path == destination or destination in path.parents:
            raise ValueError(f"{label} must remain outside the trajectory case")

    run_id = str(uuid4())
    staging = destination.with_name(f".{destination.name}.{run_id}.staging")
    staging.mkdir()
    runner = process_runner or subprocess.run
    source_at_start: dict[str, Any] | None = None
    try:
        source_at_start = _source_inventory()
        stage0 = audit_unsteady_naca0012_bundle(
            resource_dir=resource_root, manifest_path=manifest_file
        )
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        resources = manifest["resources"]
        authority_before = _authority_inventory(
            resource_root=resource_root,
            manifest_file=manifest_file,
            manifest=manifest,
        )
        authority = stage0["authority_binding"]
        if authority_before["manifest"]["sha256"] != authority["manifest_sha256"]:
            raise ValueError("live manifest differs from the Stage-0 authority")
        release, manifest_executable_sha256 = _release_contract(manifest)
        if expected_executable_sha256 != manifest_executable_sha256:
            raise ValueError("requested executable SHA256 differs from the manifest")
        executable_before = _file_record(executable)
        executable_contract = release["windows_mpi_asset"]["executable"]
        if (
            executable_before["sha256"] != expected_executable_sha256
            or executable_before["bytes"] != executable_contract["bytes"]
        ):
            raise ValueError(
                "SU2 executable differs from the official release identity"
            )

        target_record = resources["restart_00499"]
        canonical_target = (resource_root / target_record["file"]).resolve()
        if target != canonical_target:
            raise ValueError("target path is not the frozen Stage-0 target")
        target_before = _file_record(target)
        if target_before["sha256"] != authority["resource_sha256"]["restart_00499"]:
            raise ValueError("canonical target differs from the Stage-0 authority")

        staged_roles = {
            "license": ("license", "LICENSE"),
            "mesh": ("mesh", resources["mesh"]["file"]),
            "restart_00497": ("restart_00497", resources["restart_00497"]["file"]),
            "restart_00498": ("restart_00498", resources["restart_00498"]["file"]),
            "upstream_config": ("config", "upstream_unsteady_naca0012.cfg"),
        }
        staged_inputs: dict[str, dict[str, Any]] = {}
        for role, (resource_key, staged_name) in staged_roles.items():
            source = resource_root / resources[resource_key]["file"]
            staged = staging / staged_name
            shutil.copyfile(source, staged)
            observed = _file_record(staged)
            if observed["sha256"] != authority["resource_sha256"][resource_key]:
                raise ValueError(f"staged input differs from Stage-0 authority: {role}")
            staged_inputs[role] = observed
        staged_manifest = staging / "stage0_resource_manifest.json"
        shutil.copyfile(manifest_file, staged_manifest)
        if sha256_file(staged_manifest) != authority["manifest_sha256"]:
            raise ValueError("staged manifest differs from Stage-0 authority")

        upstream_config = staging / staged_inputs["upstream_config"]["file"]
        trajectory_config = staging / NACA_TRAJECTORY_CONFIG_FILENAME
        trajectory_config.write_text(
            _render_config(
                upstream_config.read_text(encoding="utf-8", errors="strict"),
                overrides,
            ),
            encoding="utf-8",
            newline="\n",
        )
        parsed_config = parse_su2_config(trajectory_config)
        for key, expected in overrides.items():
            if parsed_config.get(key) != expected:
                raise ValueError(f"trajectory config {key} differs from its contract")
        staged_restart_names = sorted(
            path.name for path in staging.glob("restart_flow_*.dat")
        )
        expected_history_names = [
            resources[f"restart_{index:05d}"]["file"]
            for index in NACA_BDF2_HISTORY_INDICES
        ]
        if staged_restart_names != expected_history_names:
            raise ValueError(
                "trajectory case does not contain exactly histories 497/498"
            )
        if (staging / target.name).exists():
            raise ValueError("canonical target was copied into the trajectory case")

        case_contract = _write_self_hashed_json(
            staging / NACA_TRAJECTORY_CASE_FILENAME,
            {
                "schema": NACA_TRAJECTORY_CASE_SCHEMA,
                "status": "prepared_not_executed",
                "stage0_authority_binding": authority,
                "config": {**_file_record(trajectory_config), "overrides": overrides},
                "staged_inputs": staged_inputs,
                "staged_manifest": _file_record(staged_manifest),
                "external_target": {**target_before, "copied_into_case": False},
                "expected_outputs": {
                    "stem": NACA_TRAJECTORY_OUTPUT_STEM,
                    "first_index": expected_indices[0],
                    "last_index": expected_indices[-1],
                    "count": len(expected_indices),
                },
                "expected_history": NACA_TRAJECTORY_HISTORY_FILENAME,
                "native_solver_executed": False,
                **_failure_claims(),
            },
        )
        input_names = [record["file"] for record in staged_inputs.values()] + [
            staged_manifest.name,
            trajectory_config.name,
            NACA_TRAJECTORY_CASE_FILENAME,
        ]
        case_inputs_before = _case_input_inventory(staging, input_names)
        source_after_preparation = _source_inventory()
        authority_after_preparation = _authority_inventory(
            resource_root=resource_root,
            manifest_file=manifest_file,
            manifest=manifest,
        )
        if source_after_preparation != source_at_start:
            raise ValueError("trajectory source changed during preparation")
        if _file_record(target) != target_before:
            raise ValueError("canonical target changed during preparation")
        if _file_record(executable) != executable_before:
            raise ValueError("SU2 executable changed during preparation")
        if authority_after_preparation != authority_before:
            raise ValueError("Stage-0 inputs changed during preparation")
        staging.replace(destination)
    except Exception as error:
        try:
            receipt_path = _finalize_preparation_failure(
                staging=staging,
                destination=destination,
                run_id=run_id,
                error=error,
            )
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        raise TrajectoryRunError(str(error), receipt_path=receipt_path) from error

    case_root = destination
    receipt_path = case_root / NACA_TRAJECTORY_RECEIPT_FILENAME
    stdout_path = case_root / NACA_TRAJECTORY_STDOUT_FILENAME
    stderr_path = case_root / NACA_TRAJECTORY_STDERR_FILENAME
    selected_environment = {"OMP_NUM_THREADS": "1", "OMP_DYNAMIC": "FALSE"}
    environment = os.environ.copy()
    environment.update(selected_environment)
    command = [
        str(executable),
        "--threads",
        "1",
        NACA_TRAJECTORY_CONFIG_FILENAME,
    ]
    probe: dict[str, Any] | None = None
    process_attempted = False
    process_started = False
    return_code: int | None = None
    timed_out = False
    process_error: str | None = None
    started_at: str | None = None
    ended_at: str | None = None
    duration_seconds: float | None = None
    status = "process_failed"
    validation_error: str | None = None
    source_after_process: dict[str, Any] | None = None
    source_after_validation: dict[str, Any] | None = None
    authority_after_process: dict[str, Any] | None = None
    authority_after_validation: dict[str, Any] | None = None
    case_inputs_after_process: dict[str, Any] | None = None
    case_inputs_after_validation: dict[str, Any] | None = None
    target_after: dict[str, Any] | None = None
    executable_after: dict[str, Any] | None = None
    history_record: dict[str, Any] | None = None
    storage_manifest: dict[str, Any] | None = None
    storage_manifest_record: dict[str, Any] | None = None
    try:
        with (
            stdout_path.open("x", encoding="utf-8", newline="\n") as stdout_handle,
            stderr_path.open("x", encoding="utf-8", newline="\n") as stderr_handle,
        ):
            probe = _probe_release(runner, executable, case_root, environment)
            if probe["return_code"] != 0 or probe["version"] != "8.5.0":
                raise RuntimeError("SU2 banner does not match the pinned release")
            process_attempted = True
            started_at = _utc_now()
            start_clock = time.perf_counter()
            try:
                completed = runner(
                    command,
                    cwd=case_root,
                    env=environment,
                    shell=False,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=timeout_seconds,
                    check=False,
                )
                return_code = completed.returncode
                process_started = True
            except subprocess.TimeoutExpired:
                timed_out = True
                process_started = True
                process_error = f"timeout after {timeout_seconds:g} seconds"
            except OSError as error:
                process_error = f"{type(error).__name__}: {error}"
            duration_seconds = time.perf_counter() - start_clock
            ended_at = _utc_now()

        source_after_process = _source_inventory()
        authority_after_process = _authority_inventory(
            resource_root=resource_root,
            manifest_file=manifest_file,
            manifest=manifest,
        )
        case_inputs_after_process = _case_input_inventory(case_root, input_names)
        target_after = _file_record(target)
        executable_after = _file_record(executable)
        if source_after_process != source_at_start:
            raise ValueError("trajectory source changed during execution")
        if target_after != target_before:
            raise ValueError("canonical target changed during execution")
        if executable_after != executable_before:
            raise ValueError("SU2 executable changed during execution")
        if authority_after_process != authority_before:
            raise ValueError("Stage-0 inputs changed during execution")
        if case_inputs_after_process != case_inputs_before:
            raise ValueError("staged trajectory input changed during execution")
        if timed_out:
            raise RuntimeError(process_error)
        if process_error is not None:
            raise RuntimeError(process_error)
        if return_code != 0:
            raise RuntimeError(f"SU2 exited with code {return_code}")

        history_paths = sorted(case_root.glob("history_*.csv"))
        if history_paths != [case_root / NACA_TRAJECTORY_HISTORY_FILENAME]:
            raise ValueError("solver did not produce exactly history_00499.csv")
        output_records, aggregate = _validate_outputs(
            case_root=case_root,
            mesh_path=case_root / staged_inputs["mesh"]["file"],
            expected_indices=expected_indices,
        )
        history_record = _validate_history(history_paths[0], expected_indices)
        storage_manifest = _write_self_hashed_json(
            case_root / NACA_TRAJECTORY_STORAGE_FILENAME,
            {
                "schema": NACA_TRAJECTORY_STORAGE_SCHEMA,
                "status": "validated",
                "output_contract": {
                    "native_fields": list(NACA_NATIVE_RESTART_FIELDS),
                    "dynamic_fields": list(NACA_DYNAMIC_FIELDS),
                    "first_index": expected_indices[0],
                    "last_index": expected_indices[-1],
                    "count": len(expected_indices),
                },
                "files": output_records,
                "aggregate": aggregate,
                "history": history_record,
            },
        )
        storage_manifest_record = _file_record(
            case_root / NACA_TRAJECTORY_STORAGE_FILENAME
        )
        source_after_validation = _source_inventory()
        authority_after_validation = _authority_inventory(
            resource_root=resource_root,
            manifest_file=manifest_file,
            manifest=manifest,
        )
        case_inputs_after_validation = _case_input_inventory(case_root, input_names)
        target_after = _file_record(target)
        executable_after = _file_record(executable)
        if source_after_validation != source_at_start:
            raise ValueError("trajectory source changed during validation")
        if target_after != target_before:
            raise ValueError("canonical target changed during validation")
        if executable_after != executable_before:
            raise ValueError("SU2 executable changed during validation")
        if authority_after_validation != authority_before:
            raise ValueError("Stage-0 inputs changed during validation")
        if case_inputs_after_validation != case_inputs_before:
            raise ValueError("staged trajectory input changed during validation")
        status = "execution_and_validation_succeeded"
    except Exception as error:  # noqa: BLE001 - every failure must preserve evidence
        validation_error = f"{type(error).__name__}: {error}"
        if probe is not None and (
            probe.get("return_code") != 0 or probe.get("version") != "8.5.0"
        ):
            status = "release_probe_failed"
        elif return_code == 0 and not timed_out and process_error is None:
            status = "validation_failed"
        else:
            status = "process_failed"

    generated_artifacts = {
        path.name: _file_record_or_error(path)
        for path in sorted(case_root.iterdir(), key=lambda item: item.name)
        if path.is_file() and path.name not in input_names
    }
    receipt = _write_self_hashed_json(
        receipt_path,
        {
            "schema": NACA_TRAJECTORY_RECEIPT_SCHEMA,
            "run_id": run_id,
            "status": status,
            "error": validation_error or process_error,
            "stage0_authority_binding": authority,
            "prepared_case": {
                "case_contract": case_contract,
                "input_inventory_before": case_inputs_before,
                "input_inventory_after_process": case_inputs_after_process,
                "input_inventory_after_validation": case_inputs_after_validation,
            },
            "trajectory_contract": {
                "restart_iter": NACA_REPLAY_TARGET_INDEX,
                "final_time_iter": final_time_iter,
                "generated_first_index": expected_indices[0],
                "generated_last_index": expected_indices[-1],
                "generated_count": len(expected_indices),
                "bdf2_history_indices": list(NACA_BDF2_HISTORY_INDICES),
                "output_every_time_step": True,
                "output_stem": NACA_TRAJECTORY_OUTPUT_STEM,
                "history_wrt_freq_inner": 0,
                "one_final_inner_history_row_per_time_iter": True,
            },
            "runtime_identity": {
                "executable_before": {
                    **executable_before,
                    "absolute_path": str(executable),
                },
                "executable_after": executable_after,
                "release_asset": release["windows_mpi_asset"],
                "release_probe": probe,
                "selected_environment": selected_environment,
                "argv": [
                    "<bound-su2-executable>",
                    "--threads",
                    "1",
                    NACA_TRAJECTORY_CONFIG_FILENAME,
                ],
                "execution_mode": "direct_single_mpi_rank_one_openmp_thread",
            },
            "process": {
                "attempted": process_attempted,
                "started": process_started,
                "started_at_utc": started_at,
                "ended_at_utc": ended_at,
                "duration_seconds": duration_seconds,
                "timeout_seconds": timeout_seconds,
                "timed_out": timed_out,
                "return_code": return_code,
                "stdout": _file_record_or_error(stdout_path),
                "stderr": _file_record_or_error(stderr_path),
            },
            "provenance_bracket": {
                "source_at_start": source_at_start,
                "source_after_preparation": source_after_preparation,
                "source_after_process": source_after_process,
                "source_after_validation": source_after_validation,
                "authority_inputs_before": authority_before,
                "authority_inputs_after_preparation": authority_after_preparation,
                "authority_inputs_after_process": authority_after_process,
                "authority_inputs_after_validation": authority_after_validation,
                "target_before": target_before,
                "target_after": target_after,
                "executable_before": executable_before,
                "executable_after": executable_after,
            },
            "external_target": {**target_before, "copied_into_case": False},
            "history": history_record,
            "storage_manifest": storage_manifest_record,
            "storage_aggregate": (
                storage_manifest.get("aggregate")
                if storage_manifest is not None
                else None
            ),
            "generated_artifacts": generated_artifacts,
            "process_attempted": process_attempted,
            "process_started": process_started,
            "execution_succeeded": (
                return_code == 0 and not timed_out and process_error is None
            ),
            "trajectory_validated": status == "execution_and_validation_succeeded",
            **_failure_claims(),
        },
    )
    if status != "execution_and_validation_succeeded":
        raise TrajectoryRunError(
            validation_error or process_error or "trajectory run failed",
            receipt_path=receipt_path,
        )
    return receipt
