from __future__ import annotations

import csv
import json
import struct
import subprocess
import sys
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.time_dependent_no.run_su2_naca0012_trajectory import parse_args
from utility.time_dependent_no import su2_naca_trajectory as trajectory
from utility.time_dependent_no.su2_naca_trajectory import (
    NACA_HISTORY_FIELDS,
    NACA_NATIVE_RESTART_FIELDS,
    NACA_TRAJECTORY_CASE_FILENAME,
    NACA_TRAJECTORY_CONFIG_FILENAME,
    NACA_TRAJECTORY_DEFAULT_FINAL_TIME_ITER,
    NACA_TRAJECTORY_HISTORY_FILENAME,
    NACA_TRAJECTORY_RECEIPT_FILENAME,
    NACA_TRAJECTORY_STORAGE_FILENAME,
    TrajectoryRunError,
    load_verified_trajectory_receipt,
    run_su2_naca0012_trajectory,
)
from utility.time_dependent_no.su2_restart_contract import (
    NACA_CONFIG_CONTRACT,
    NACA_DYNAMIC_FIELDS,
    NACA_RESTART_FIELDS,
    NACA_STAGE0_CLAIM_BOUNDARY,
    RESOURCE_MANIFEST_SCHEMA,
    SU2_BINARY_MAGIC,
    parse_su2_config,
)


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _write_mesh(path: Path) -> np.ndarray:
    points = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        dtype=np.float64,
    )
    path.write_text(
        """NDIME=2
NELEM=1
9 0 1 2 3 0
NPOIN=4
0.0 0.0 0
1.0 0.0 1
1.0 1.0 2
0.0 1.0 3
NMARK=2
MARKER_TAG= airfoil
MARKER_ELEMS=4
3 0 1
3 1 2
3 2 3
3 3 0
MARKER_TAG= farfield
MARKER_ELEMS=4
3 0 1
3 1 2
3 2 3
3 3 0
""",
        encoding="utf-8",
    )
    return points


def _write_restart(path: Path, values: np.ndarray, fields: tuple[str, ...]) -> None:
    with path.open("wb") as handle:
        handle.write(
            struct.pack(
                "<5i",
                SU2_BINARY_MAGIC,
                len(fields),
                values.shape[0],
                0,
                0,
            )
        )
        for field in fields:
            encoded = field.encode("ascii")
            handle.write(encoded + b"\0" * (33 - len(encoded)))
        handle.write(np.asarray(values, dtype="<f8").tobytes(order="C"))


def _canonical_values(points: np.ndarray, offset: float) -> np.ndarray:
    values = np.zeros((len(points), len(NACA_RESTART_FIELDS)), dtype=np.float64)
    values[:, :2] = points
    for channel, field in enumerate(NACA_DYNAMIC_FIELDS, start=1):
        values[:, NACA_RESTART_FIELDS.index(field)] = channel + offset
    return values


def _native_values(points: np.ndarray, offset: float) -> np.ndarray:
    canonical = _canonical_values(points, offset)
    values = np.zeros((len(points), len(NACA_NATIVE_RESTART_FIELDS)))
    for field in NACA_RESTART_FIELDS:
        values[:, NACA_NATIVE_RESTART_FIELDS.index(field)] = canonical[
            :, NACA_RESTART_FIELDS.index(field)
        ]
    values[:, NACA_NATIVE_RESTART_FIELDS.index("Velocity_x")] = 10.0 + offset
    values[:, NACA_NATIVE_RESTART_FIELDS.index("Velocity_y")] = -2.0 - offset
    return values


def _write_history(path: Path, indices: list[int], *, mode: str = "success") -> None:
    fields = list(NACA_HISTORY_FIELDS)
    if mode == "history_schema":
        fields[2] = "renamed_rms"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        row_indices = list(indices)
        if mode == "history_missing_index":
            row_indices.pop()
        elif mode == "history_extra_index":
            row_indices.append(indices[-1] + 1)
        elif mode == "history_wrong_index":
            row_indices[0] -= 1
        for index in row_indices:
            row = [0.0] * len(fields)
            row[0] = index
            row[1] = 8
            row[fields.index("CD")] = 0.30
            row[fields.index("CL")] = 0.62
            row[fields.index("CFx")] = 0.10
            row[fields.index("CFy")] = 0.68
            writer.writerow(row)


def _write_fixture(tmp_path: Path) -> dict[str, Path | str | np.ndarray]:
    executable = tmp_path / "SU2_CFD.exe"
    executable.write_bytes(b"synthetic pinned SU2 v8.5.0 executable")
    executable_sha256 = _digest(executable)
    resources = tmp_path / "resources"
    resources.mkdir()
    license_path = resources / "LICENSE"
    license_path.write_text("synthetic test license\n", encoding="utf-8")
    config = resources / "unsteady_naca0012.cfg"
    config.write_text(
        """SOLVER= RANS
KIND_TURB_MODEL= SA
RESTART_SOL= YES
RESTART_ITER= 499
TIME_DOMAIN= YES
TIME_MARCHING= DUAL_TIME_STEPPING-2ND_ORDER
TIME_STEP= 5e-4
INNER_ITER= 10
AOA= 17.0
MESH_FILENAME= unsteady_naca0012_mesh.su2
SOLUTION_FILENAME= restart_flow
TIME_ITER= 500
RESTART_FILENAME= restart_flow
OUTPUT_FILES= ( RESTART, PARAVIEW )
OUTPUT_WRT_FREQ= ( 1, 1 )
""",
        encoding="utf-8",
    )
    mesh = resources / "unsteady_naca0012_mesh.su2"
    points = _write_mesh(mesh)
    restart_paths: dict[str, Path] = {}
    for offset, index in enumerate((497, 498, 499)):
        restart = resources / f"restart_flow_{index:05d}.dat"
        _write_restart(
            restart,
            _canonical_values(points, float(offset)),
            NACA_RESTART_FIELDS,
        )
        restart_paths[f"restart_{index:05d}"] = restart
    named_paths = {
        "license": license_path,
        "config": config,
        "mesh": mesh,
        **restart_paths,
    }
    manifest = {
        "schema": RESOURCE_MANIFEST_SCHEMA,
        "upstream": {
            "tutorial_commit": "synthetic",
            "target_replay_release": {
                "tag": "v8.5.0",
                "windows_mpi_asset": {
                    "executable": {
                        "file": "bin/SU2_CFD.exe",
                        "bytes": executable.stat().st_size,
                        "sha256": executable_sha256,
                    }
                },
            },
        },
        "resources": {
            role: {
                "file": path.name,
                "bytes": path.stat().st_size,
                "sha256": _digest(path),
            }
            for role, path in named_paths.items()
        },
        "config_contract": dict(NACA_CONFIG_CONTRACT),
        "mesh_contract": {
            "dimension": 2,
            "num_elements": 1,
            "num_points": 4,
            "element_type_counts": {"9": 1},
            "marker_element_counts": {"airfoil": 4, "farfield": 4},
        },
        "restart_contract": {
            "indices": [497, 498, 499],
            "bdf2_history_indices": [497, 498],
            "one_step_replay_target_index": 499,
            "fields": list(NACA_RESTART_FIELDS),
            "dynamic_fields": list(NACA_DYNAMIC_FIELDS),
        },
        "claim_boundary": dict(NACA_STAGE0_CLAIM_BOUNDARY),
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return {
        "resource_dir": resources,
        "manifest": manifest_path,
        "target": restart_paths["restart_00499"],
        "executable": executable,
        "executable_sha256": executable_sha256,
        "points": points,
    }


def _fake_runner(
    *,
    executable: Path,
    case_dir: Path,
    target: Path,
    authority_input: Path,
    points: np.ndarray,
    expected_indices: list[int],
    mode: str = "success",
):
    calls: list[list[str]] = []

    def run(argv, **kwargs):
        calls.append(list(argv))
        assert argv[0] == str(executable)
        assert kwargs["cwd"] == case_dir
        assert kwargs["shell"] is False
        assert kwargs["env"]["OMP_NUM_THREADS"] == "1"
        assert kwargs["env"]["OMP_DYNAMIC"] == "FALSE"
        if argv[1:] == ["--help"]:
            assert kwargs["capture_output"] is True
            version = "8.4.0" if mode == "wrong_probe_version" else "8.5.0"
            return SimpleNamespace(
                returncode=0,
                stdout=f"SU2 v{version} synthetic banner\n",
                stderr="",
            )

        assert argv == [
            str(executable),
            "--threads",
            "1",
            NACA_TRAJECTORY_CONFIG_FILENAME,
        ]
        assert kwargs["timeout"] == 10.0
        produced = list(expected_indices)
        if mode == "missing_output":
            produced.pop()
        elif mode == "extra_output":
            produced.append(expected_indices[-1] + 1)
        elif mode == "wrong_output_indices":
            produced = [index + 1 for index in produced]
        fields = NACA_NATIVE_RESTART_FIELDS
        if mode == "wrong_schema":
            fields = tuple(
                "Renamed_Density" if field == "Density" else field for field in fields
            )
        for offset, index in enumerate(produced):
            values = _native_values(points, float(offset))
            if mode == "wrong_coordinates" and index == produced[0]:
                values[0, 0] += 1.0e-3
            if mode == "nonfinite_dynamic" and index == produced[0]:
                values[0, fields.index("Density")] = np.nan
            _write_restart(
                case_dir / f"trajectory_flow_{index:05d}.dat",
                values,
                fields,
            )
        if mode != "missing_history":
            history_mode = mode if mode.startswith("history_") else "success"
            _write_history(
                case_dir / NACA_TRAJECTORY_HISTORY_FILENAME,
                expected_indices,
                mode=history_mode,
            )
            if mode == "extra_history_file":
                _write_history(case_dir / "history_00500.csv", expected_indices)
        if mode == "target_mutation":
            target.write_bytes(target.read_bytes() + b"mutated")
        if mode == "input_mutation":
            trajectory_config = case_dir / NACA_TRAJECTORY_CONFIG_FILENAME
            trajectory_config.write_text(
                trajectory_config.read_text(encoding="utf-8") + "% mutated\n",
                encoding="utf-8",
            )
        if mode == "unexpected_input_stem":
            _write_restart(
                case_dir / "restart_flow_00500.dat",
                _native_values(points, 0.0),
                NACA_NATIVE_RESTART_FIELDS,
            )
        if mode == "authority_input_mutation":
            authority_input.write_text(
                authority_input.read_text(encoding="utf-8") + "% mutated\n",
                encoding="utf-8",
            )
        if mode == "executable_mutation":
            executable.write_bytes(executable.read_bytes() + b"mutated")
        return SimpleNamespace(returncode=7 if mode == "nonzero" else 0)

    run.calls = calls
    return run


def _run(tmp_path: Path, *, mode: str = "success"):
    fixture = _write_fixture(tmp_path)
    case_dir = tmp_path / "trajectory_case"
    indices = [499, 500, 501]
    runner = _fake_runner(
        executable=fixture["executable"],
        case_dir=case_dir,
        target=fixture["target"],
        authority_input=fixture["resource_dir"] / "unsteady_naca0012.cfg",
        points=fixture["points"],
        expected_indices=indices,
        mode=mode,
    )
    arguments = {
        "resource_dir": fixture["resource_dir"],
        "manifest_path": fixture["manifest"],
        "case_dir": case_dir,
        "target_path": fixture["target"],
        "executable_path": fixture["executable"],
        "expected_executable_sha256": fixture["executable_sha256"],
        "final_time_iter": 502,
        "timeout_seconds": 10.0,
        "process_runner": runner,
    }
    return fixture, case_dir, runner, arguments


def _load_self_hashed(path: Path, expected_schema: str) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema"] == expected_schema
    observed = payload.pop("canonical_payload_sha256")
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    assert observed == sha256(canonical).hexdigest()
    return payload


def test_trajectory_success_freezes_bdf2_config_storage_and_claim_boundary(tmp_path):
    fixture, case_dir, runner, arguments = _run(tmp_path)
    target_before = _digest(fixture["target"])

    receipt = run_su2_naca0012_trajectory(**arguments)

    assert receipt["status"] == "execution_and_validation_succeeded"
    assert receipt["trajectory_validated"] is True
    assert receipt["process"]["return_code"] == 0
    assert receipt["runtime_identity"]["execution_mode"] == (
        "direct_single_mpi_rank_one_openmp_thread"
    )
    assert runner.calls == [
        [str(fixture["executable"]), "--help"],
        [
            str(fixture["executable"]),
            "--threads",
            "1",
            NACA_TRAJECTORY_CONFIG_FILENAME,
        ],
    ]
    config = parse_su2_config(case_dir / NACA_TRAJECTORY_CONFIG_FILENAME)
    assert config["RESTART_ITER"] == "499"
    assert config["TIME_ITER"] == "502"
    assert config["TIME_MARCHING"] == "DUAL_TIME_STEPPING-2ND_ORDER"
    assert config["RESTART_FILENAME"] == "trajectory_flow"
    assert config["HISTORY_WRT_FREQ_INNER"] == "0"
    assert config["OUTPUT_FILES"] == "( RESTART )"
    assert config["OUTPUT_WRT_FREQ"] == "( 1 )"
    assert config["WRT_RESTART_COMPACT"] == "NO"
    assert sorted(path.name for path in case_dir.glob("restart_flow_*.dat")) == [
        "restart_flow_00497.dat",
        "restart_flow_00498.dat",
    ]
    assert not (case_dir / "restart_flow_00499.dat").exists()
    assert _digest(fixture["target"]) == target_before

    storage = _load_self_hashed(
        case_dir / NACA_TRAJECTORY_STORAGE_FILENAME,
        trajectory.NACA_TRAJECTORY_STORAGE_SCHEMA,
    )
    assert [record["index"] for record in storage["files"]] == [499, 500, 501]
    assert storage["aggregate"]["file_count"] == 3
    assert storage["aggregate"]["total_bytes"] == sum(
        record["bytes"] for record in storage["files"]
    )
    assert storage["aggregate"]["five_evolved_float64_bytes"] == 3 * 4 * 5 * 8
    assert storage["aggregate"]["bdf2_distinct_frame_semantics"] == {
        "staged_history_indices": [497, 498],
        "generated_indices": [499, 500, 501],
        "distinct_frames": 5,
        "transition_level_frame_tripling": False,
    }
    assert receipt["history"]["row_count"] == 3
    assert (
        receipt["trajectory_contract"]["one_final_inner_history_row_per_time_iter"]
        is True
    )
    assert (
        receipt["provenance_bracket"]["source_at_start"]
        == receipt["provenance_bracket"]["source_after_validation"]
    )
    assert (
        receipt["provenance_bracket"]["authority_inputs_before"]
        == receipt["provenance_bracket"]["authority_inputs_after_validation"]
    )
    for flag in (
        "restart_sufficiency_claimed",
        "trusted_transition_claimed",
        "trajectory_population_claimed",
        "id_population_claimed",
        "boundary_and_force_gate_complete",
    ):
        assert receipt[flag] is False
    verified = load_verified_trajectory_receipt(
        case_dir / NACA_TRAJECTORY_RECEIPT_FILENAME
    )
    assert verified["canonical_payload_sha256"] == receipt["canonical_payload_sha256"]


@pytest.mark.parametrize(
    ("mode", "message"),
    [
        ("missing_output", "restart indices"),
        ("extra_output", "restart indices"),
        ("wrong_output_indices", "restart indices"),
        ("wrong_schema", "native restart schema"),
        ("wrong_coordinates", "coordinates differ"),
        ("nonfinite_dynamic", "nonfinite"),
        ("unexpected_input_stem", "unexpected restart_flow output"),
    ],
)
def test_trajectory_rejects_output_contract_mutations(tmp_path, mode, message):
    _, case_dir, _, arguments = _run(tmp_path, mode=mode)
    with pytest.raises(TrajectoryRunError, match=message) as captured:
        run_su2_naca0012_trajectory(**arguments)
    receipt = load_verified_trajectory_receipt(captured.value.receipt_path)
    assert receipt["status"] == "validation_failed"
    assert receipt["trajectory_validated"] is False
    assert receipt["trusted_transition_claimed"] is False
    assert not (case_dir / NACA_TRAJECTORY_STORAGE_FILENAME).exists()


@pytest.mark.parametrize(
    "mode",
    [
        "missing_history",
        "extra_history_file",
        "history_schema",
        "history_missing_index",
        "history_extra_index",
        "history_wrong_index",
    ],
)
def test_trajectory_rejects_missing_extra_or_malformed_history(tmp_path, mode):
    _, _, _, arguments = _run(tmp_path, mode=mode)
    with pytest.raises(TrajectoryRunError) as captured:
        run_su2_naca0012_trajectory(**arguments)
    receipt = load_verified_trajectory_receipt(captured.value.receipt_path)
    assert receipt["status"] == "validation_failed"
    assert "history" in receipt["error"]
    assert receipt["trajectory_population_claimed"] is False


@pytest.mark.parametrize(
    ("mode", "message"),
    [
        ("target_mutation", "canonical target changed"),
        ("input_mutation", "staged trajectory input changed"),
        ("authority_input_mutation", "Stage-0 inputs changed"),
        ("executable_mutation", "SU2 executable changed"),
    ],
)
def test_trajectory_rehashes_target_and_staged_inputs_after_process(
    tmp_path, mode, message
):
    _, _, _, arguments = _run(tmp_path, mode=mode)
    with pytest.raises(TrajectoryRunError, match=message) as captured:
        run_su2_naca0012_trajectory(**arguments)
    receipt = load_verified_trajectory_receipt(captured.value.receipt_path)
    assert receipt["status"] == "validation_failed"
    assert receipt["execution_succeeded"] is True
    assert receipt["trajectory_validated"] is False


@pytest.mark.parametrize(
    ("mode", "status"),
    [("wrong_probe_version", "release_probe_failed"), ("nonzero", "process_failed")],
)
def test_trajectory_preserves_probe_and_process_failure_receipts(
    tmp_path, mode, status
):
    _, _, runner, arguments = _run(tmp_path, mode=mode)
    with pytest.raises(TrajectoryRunError) as captured:
        run_su2_naca0012_trajectory(**arguments)
    receipt = load_verified_trajectory_receipt(captured.value.receipt_path)
    assert receipt["status"] == status
    assert receipt["execution_succeeded"] is False
    assert receipt["trajectory_validated"] is False
    assert receipt["trusted_transition_claimed"] is False
    if mode == "wrong_probe_version":
        assert len(runner.calls) == 1
        assert receipt["process_attempted"] is False
    else:
        assert len(runner.calls) == 2
        assert receipt["process_started"] is True


def test_trajectory_rejects_source_drift_after_process(tmp_path, monkeypatch):
    _, _, _, arguments = _run(tmp_path)
    real_inventory = trajectory._source_inventory
    calls = 0

    def drifting_inventory():
        nonlocal calls
        calls += 1
        inventory = json.loads(json.dumps(real_inventory()))
        if calls >= 3:
            first = next(iter(inventory))
            inventory[first]["sha256"] = "f" * 64
        return inventory

    monkeypatch.setattr(trajectory, "_source_inventory", drifting_inventory)
    with pytest.raises(TrajectoryRunError, match="source changed") as captured:
        run_su2_naca0012_trajectory(**arguments)
    receipt = load_verified_trajectory_receipt(captured.value.receipt_path)
    assert receipt["status"] == "validation_failed"
    assert receipt["trajectory_validated"] is False


def test_trajectory_stage0_failure_is_atomic_and_never_launches(tmp_path):
    fixture, case_dir, _, arguments = _run(tmp_path)
    config = fixture["resource_dir"] / "unsteady_naca0012.cfg"
    config.write_text(config.read_text(encoding="utf-8") + "% drift\n")
    launched = False

    def fail_if_called(*args, **kwargs):
        nonlocal launched
        launched = True
        raise AssertionError("solver must not be called")

    arguments["process_runner"] = fail_if_called
    with pytest.raises(TrajectoryRunError) as captured:
        run_su2_naca0012_trajectory(**arguments)
    assert launched is False
    assert captured.value.receipt_path == case_dir / NACA_TRAJECTORY_RECEIPT_FILENAME
    receipt = load_verified_trajectory_receipt(captured.value.receipt_path)
    assert receipt["status"] == "preparation_failed"
    assert receipt["process_attempted"] is False
    assert case_dir.is_dir()
    assert not list(tmp_path.glob(f".{case_dir.name}.*.staging"))


def test_trajectory_rejects_wrong_executable_identity_before_launch(tmp_path):
    _, _, _, arguments = _run(tmp_path)
    arguments["expected_executable_sha256"] = "f" * 64
    launched = False

    def fail_if_called(*args, **kwargs):
        nonlocal launched
        launched = True
        raise AssertionError("solver must not be called")

    arguments["process_runner"] = fail_if_called
    with pytest.raises(TrajectoryRunError, match="differs from the manifest"):
        run_su2_naca0012_trajectory(**arguments)
    assert launched is False


def test_cli_requires_explicit_resources_and_defaults_only_final_horizon(tmp_path):
    with pytest.raises(SystemExit):
        parse_args([])
    arguments = parse_args(
        [
            "--resource-dir",
            str(tmp_path / "resources"),
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--case-dir",
            str(tmp_path / "case"),
            "--target",
            str(tmp_path / "target.dat"),
            "--su2-executable",
            str(tmp_path / "SU2_CFD.exe"),
            "--expected-executable-sha256",
            "a" * 64,
        ]
    )
    assert arguments.final_time_iter == NACA_TRAJECTORY_DEFAULT_FINAL_TIME_ITER
    assert arguments.final_time_iter == 2000


def test_cli_is_directly_executable_outside_repository_cwd(tmp_path):
    script = (
        Path(__file__).resolve().parents[2]
        / "scripts/time_dependent_no/run_su2_naca0012_trajectory.py"
    )
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
        check=False,
    )
    assert completed.returncode == 0
    assert "--resource-dir" in completed.stdout


def test_case_contract_is_self_hashed_and_contains_no_generated_target(tmp_path):
    _, case_dir, _, arguments = _run(tmp_path)
    run_su2_naca0012_trajectory(**arguments)
    case_contract = _load_self_hashed(
        case_dir / NACA_TRAJECTORY_CASE_FILENAME,
        trajectory.NACA_TRAJECTORY_CASE_SCHEMA,
    )
    assert case_contract["expected_outputs"] == {
        "stem": "trajectory_flow",
        "first_index": 499,
        "last_index": 501,
        "count": 3,
    }
    assert case_contract["external_target"]["copied_into_case"] is False
