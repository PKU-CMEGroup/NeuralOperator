from __future__ import annotations

import json
import shutil
import struct
import subprocess
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import utility.time_dependent_no.su2_native_replay as native_replay
from utility.time_dependent_no.su2_native_replay import (
    NACA_NATIVE_RECEIPT_FILENAME,
    NativeReplayError,
    compare_native_replay_receipts,
    evaluate_restart_repeat,
    evaluate_restart_replay,
    load_verified_json_receipt,
    lumped_marker_lengths,
    lumped_vertex_areas,
    run_native_naca0012_replay,
)
from utility.time_dependent_no.su2_restart_contract import (
    NACA_DYNAMIC_FIELDS,
    NACA_REPLAY_OUTPUT_FILENAME,
    NACA_RESTART_FIELDS,
    RESOURCE_MANIFEST_SCHEMA,
    SU2_BINARY_MAGIC,
    audit_unsteady_naca0012_bundle,
    parse_su2_config,
    parse_su2_mesh,
    prepare_unsteady_naca0012_replay_case,
    read_su2_binary_restart,
)


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _write_mesh(path: Path, *, reverse_airfoil: bool = False) -> np.ndarray:
    points = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        dtype=np.float64,
    )
    airfoil_edges = (
        "3 0 3\n3 3 2\n3 2 1\n3 1 0"
        if reverse_airfoil
        else "3 0 1\n3 1 2\n3 2 3\n3 3 0"
    )
    path.write_text(
        f"""NDIME=2
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
{airfoil_edges}
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


def _restart_values(points: np.ndarray, offset: float) -> np.ndarray:
    values = np.zeros((points.shape[0], len(NACA_RESTART_FIELDS)), dtype=np.float64)
    values[:, :2] = points
    for channel, name in enumerate(NACA_DYNAMIC_FIELDS, start=1):
        values[:, NACA_RESTART_FIELDS.index(name)] = channel + offset
    return values


def _native_restart_values(canonical_values: np.ndarray) -> np.ndarray:
    native_fields = native_replay.NACA_NATIVE_RESTART_FIELDS
    native_values = np.zeros((canonical_values.shape[0], len(native_fields)))
    for field in NACA_RESTART_FIELDS:
        native_values[:, native_fields.index(field)] = canonical_values[
            :, NACA_RESTART_FIELDS.index(field)
        ]
    native_values[:, native_fields.index("Velocity_x")] = 123.0
    native_values[:, native_fields.index("Velocity_y")] = -456.0
    return native_values


def _write_restart(
    path: Path,
    values: np.ndarray,
    *,
    fields: tuple[str, ...] = NACA_RESTART_FIELDS,
) -> None:
    if values.shape[1] != len(fields):
        raise ValueError("restart fixture columns do not match its fields")
    with path.open("wb") as handle:
        handle.write(
            struct.pack(
                "<5i",
                SU2_BINARY_MAGIC,
                values.shape[1],
                values.shape[0],
                0,
                0,
            )
        )
        for name in fields:
            encoded = name.encode("ascii")
            handle.write(encoded + b"\0" * (33 - len(encoded)))
        handle.write(values.astype("<f8", copy=False).tobytes(order="C"))


def _write_bundle(
    tmp_path: Path,
    *,
    executable_sha256: str = "0" * 64,
    executable_bytes: int = 0,
) -> tuple[Path, Path]:
    resource_dir = tmp_path / "resources"
    resource_dir.mkdir()
    config = resource_dir / "unsteady_naca0012.cfg"
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
""",
        encoding="utf-8",
    )
    mesh = resource_dir / "unsteady_naca0012_mesh.su2"
    points = _write_mesh(mesh)
    restart_paths = []
    for offset, index in enumerate((497, 498, 499)):
        path = resource_dir / f"restart_flow_{index:05d}.dat"
        _write_restart(path, _restart_values(points, float(offset)))
        restart_paths.append(path)
    license_path = resource_dir / "LICENSE"
    license_path.write_text("synthetic test fixture\n", encoding="utf-8")

    named_paths = {
        "license": license_path,
        "config": config,
        "mesh": mesh,
        **{
            f"restart_{index:05d}": path
            for index, path in zip((497, 498, 499), restart_paths, strict=True)
        },
    }
    manifest = {
        "schema": RESOURCE_MANIFEST_SCHEMA,
        "upstream": {
            "target_replay_release": {
                "tag": "v8.5.0",
                "windows_mpi_asset": {
                    "executable": {
                        "bytes": executable_bytes,
                        "sha256": executable_sha256,
                    }
                },
            }
        },
        "resources": {
            name: {
                "file": path.name,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for name, path in named_paths.items()
        },
        "config_contract": {
            "SOLVER": "RANS",
            "KIND_TURB_MODEL": "SA",
            "RESTART_SOL": "YES",
            "RESTART_ITER": "499",
            "TIME_DOMAIN": "YES",
            "TIME_MARCHING": "DUAL_TIME_STEPPING-2ND_ORDER",
            "TIME_STEP": "5e-4",
            "INNER_ITER": "10",
            "MESH_FILENAME": "unsteady_naca0012_mesh.su2",
            "SOLUTION_FILENAME": "restart_flow",
        },
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
        "claim_boundary": {
            "native_solver_executed": False,
            "restart_sufficiency_claimed": False,
            "trusted_transition_claimed": False,
        },
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return resource_dir, manifest_path


def _prepare_runtime_fixture(
    tmp_path: Path, *, case_name: str = "native_case"
) -> tuple[Path, Path, Path, str]:
    executable = tmp_path / "SU2_CFD.exe"
    if not executable.exists():
        executable.write_bytes(b"synthetic SU2 v8.5.0 executable")
    executable_sha256 = _sha256(executable)
    resource_dir = tmp_path / "resources"
    manifest_path = tmp_path / "manifest.json"
    if not resource_dir.exists():
        resource_dir, manifest_path = _write_bundle(
            tmp_path,
            executable_sha256=executable_sha256,
            executable_bytes=executable.stat().st_size,
        )
    case_dir = tmp_path / case_name
    prepare_unsteady_naca0012_replay_case(
        resource_dir=resource_dir,
        manifest_path=manifest_path,
        output_dir=case_dir,
    )
    return (
        case_dir.resolve(),
        (resource_dir / "restart_flow_00499.dat").resolve(),
        executable.resolve(),
        executable_sha256,
    )


def _fake_su2_run(
    target_path: Path,
    expected_case_dir: Path,
    expected_executable: Path,
    *,
    return_code: int = 0,
    dynamic_offset: float = 0.0,
    extra_output: bool = False,
    mutate_target: bool = False,
    delete_mesh: bool = False,
    time_out: bool = False,
    launch_error: bool = False,
    probe_version: str = "8.5.0",
):
    def fake_run(args, **kwargs):
        assert Path(args[0]).resolve() == expected_executable.resolve()
        assert Path(kwargs["cwd"]).resolve() == expected_case_dir.resolve()
        if kwargs.get("capture_output"):
            assert args == [str(expected_executable), "--help"]
            assert kwargs["shell"] is False
            assert kwargs["timeout"] == 30
            assert kwargs["check"] is False
            assert kwargs["env"]["OMP_NUM_THREADS"] == "1"
            assert kwargs["env"]["OMP_DYNAMIC"] == "FALSE"
            return SimpleNamespace(
                returncode=0,
                stdout=(
                    f'SU2 v{probe_version} "Harrier", The Open-Source CFD Code\n'
                ),
                stderr="",
            )
        assert args == [str(expected_executable), "--threads", "1", "replay.cfg"]
        assert kwargs["shell"] is False
        assert kwargs["check"] is False
        assert kwargs["text"] is True
        assert kwargs["encoding"] == "utf-8"
        assert kwargs["errors"] == "replace"
        assert kwargs["timeout"] == 10.0
        assert kwargs["env"]["OMP_NUM_THREADS"] == "1"
        assert kwargs["env"]["OMP_DYNAMIC"] == "FALSE"
        kwargs["stdout"].write("synthetic native replay\n")
        kwargs["stderr"].write("")
        if launch_error:
            raise OSError("injected launch failure")
        if time_out:
            raise subprocess.TimeoutExpired(args, kwargs["timeout"])
        case_dir = Path(kwargs["cwd"])
        if delete_mesh:
            (case_dir / "unsteady_naca0012_mesh.su2").unlink()
        if return_code == 0:
            output = case_dir / NACA_REPLAY_OUTPUT_FILENAME
            restart = read_su2_binary_restart(target_path)
            values = restart.values.copy()
            density = NACA_RESTART_FIELDS.index("Density")
            values[:, density] += dynamic_offset
            native_fields = native_replay.NACA_NATIVE_RESTART_FIELDS
            native_values = _native_restart_values(values)
            _write_restart(output, native_values, fields=native_fields)
            if extra_output:
                shutil.copyfile(output, case_dir / "replay_flow_00500.dat")
        if mutate_target:
            target_path.write_bytes(target_path.read_bytes() + b"x")
        return SimpleNamespace(returncode=return_code)

    return fake_run


def _rewrite_receipt(path: Path, mutate) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.pop("canonical_payload_sha256")
    mutate(payload)
    rendered = native_replay._with_payload_sha256(payload)
    path.write_text(
        json.dumps(rendered, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _run_replay_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    second_dynamic_offset: float = 0.0,
) -> tuple[Path, Path]:
    case_a, target, executable, executable_sha256 = _prepare_runtime_fixture(
        tmp_path, case_name="native_a"
    )
    case_b, _, _, _ = _prepare_runtime_fixture(tmp_path, case_name="native_b")
    first_fake = _fake_su2_run(target, case_a, executable)
    second_fake = _fake_su2_run(
        target,
        case_b,
        executable,
        dynamic_offset=second_dynamic_offset,
    )

    def dispatch(args, **kwargs):
        selected = second_fake if Path(kwargs["cwd"]) == case_b else first_fake
        return selected(args, **kwargs)

    monkeypatch.setattr(
        native_replay,
        "subprocess",
        SimpleNamespace(run=dispatch, TimeoutExpired=subprocess.TimeoutExpired),
    )
    for case_dir in (case_a, case_b):
        run_native_naca0012_replay(
            case_dir=case_dir,
            target_path=target,
            executable_path=executable,
            expected_executable_sha256=executable_sha256,
            timeout_seconds=10.0,
        )
    return case_a, case_b


def test_reads_native_binary_restart_and_mesh(tmp_path: Path):
    resource_dir, _ = _write_bundle(tmp_path)

    mesh = parse_su2_mesh(resource_dir / "unsteady_naca0012_mesh.su2")
    restart = read_su2_binary_restart(resource_dir / "restart_flow_00497.dat")

    assert mesh.dimension == 2
    assert mesh.element_type_counts == {9: 1}
    assert restart.byte_order == "little"
    assert restart.header == (SU2_BINARY_MAGIC, 17, 4, 0, 0)
    assert restart.fields == NACA_RESTART_FIELDS
    np.testing.assert_allclose(restart.values[:, :2], mesh.points)


def test_bundle_audit_separates_stage0_from_solver_replay(tmp_path: Path):
    resource_dir, manifest_path = _write_bundle(tmp_path)

    summary = audit_unsteady_naca0012_bundle(
        resource_dir=resource_dir,
        manifest_path=manifest_path,
    )

    assert summary["status"] == "stage0_valid"
    assert summary["native_solver_executed"] is False
    assert summary["restart_sufficiency_claimed"] is False
    assert summary["trusted_transition_claimed"] is False
    assert summary["authority_binding"]["manifest_sha256"] == _sha256(manifest_path)
    assert summary["authority_binding"]["resource_sha256"] == {
        name: record["sha256"]
        for name, record in json.loads(manifest_path.read_text(encoding="utf-8"))[
            "resources"
        ].items()
    }
    assert summary["restart"]["bdf2_history_indices"] == [497, 498]
    assert summary["restart"]["one_step_replay_target_index"] == 499
    assert summary["restart"]["dynamic_fields"] == list(NACA_DYNAMIC_FIELDS)


def test_binary_reader_rejects_magic_and_trailing_bytes(tmp_path: Path):
    resource_dir, _ = _write_bundle(tmp_path)
    source = resource_dir / "restart_flow_00497.dat"
    malformed = tmp_path / "bad.dat"
    payload = bytearray(source.read_bytes())
    payload[:4] = struct.pack("<i", 0)
    malformed.write_bytes(payload)
    with pytest.raises(ValueError, match="invalid SU2 binary magic"):
        read_su2_binary_restart(malformed)

    trailing = tmp_path / "trailing.dat"
    trailing.write_bytes(source.read_bytes() + b"x")
    with pytest.raises(ValueError, match="file size is inconsistent"):
        read_su2_binary_restart(trailing)


def test_bundle_audit_rejects_coordinate_mismatch(tmp_path: Path):
    resource_dir, manifest_path = _write_bundle(tmp_path)
    restart_path = resource_dir / "restart_flow_00499.dat"
    restart = read_su2_binary_restart(restart_path)
    values = restart.values.copy()
    values[0, 0] += 1.0e-3
    _write_restart(restart_path, values)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = manifest["resources"]["restart_00499"]
    record["bytes"] = restart_path.stat().st_size
    record["sha256"] = _sha256(restart_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="coordinates do not match"):
        audit_unsteady_naca0012_bundle(
            resource_dir=resource_dir,
            manifest_path=manifest_path,
        )


def test_manifest_verification_fails_on_changed_resource(tmp_path: Path):
    resource_dir, manifest_path = _write_bundle(tmp_path)
    config = resource_dir / "unsteady_naca0012.cfg"
    config.write_text(config.read_text(encoding="utf-8") + "% changed\n")

    with pytest.raises(ValueError, match="byte size differs from manifest"):
        audit_unsteady_naca0012_bundle(
            resource_dir=resource_dir,
            manifest_path=manifest_path,
        )


def test_tracked_manifest_freezes_public_naca_contract():
    manifest_path = (
        Path(__file__).resolve().parents[2]
        / "docs/time_dependent_no/R0_SU2_NACA_RESOURCE_MANIFEST.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["upstream"]["tutorial_commit"] == (
        "74095b3464b16a1c0ba2db0aeea4132edb66c175"
    )
    assert manifest["upstream"]["target_replay_release"] == {
        "tag": "v8.5.0",
        "commit": "12eb826f049ef7f67df974dfcb44cf36ee07c0f8",
        "release_url": "https://github.com/su2code/SU2/releases/tag/v8.5.0",
        "windows_mpi_asset": {
            "name": "SU2-v8.5.0-win64-mpi.zip",
            "bytes": 28925770,
            "sha256": (
                "7da985f956de617c41f8a22aaab27fd42f2dfe096342b396a49583b9b3bd5d25"
            ),
            "nested_archive_name": "win64-mpi.zip",
            "nested_archive_sha256": (
                "94a10b2e910109d78e9b5b845ddcaa8efea2001a27624a77e2a06bce5c9a2200"
            ),
            "executable": {
                "file": "bin/SU2_CFD.exe",
                "bytes": 24759883,
                "sha256": (
                    "61a803e0baf49382210888cc32f14f4ea837681a0e6cc2df6b1d5a99ce5baf8c"
                ),
            },
        },
        "identity_status": (
            "local executable verified byte-for-byte against the official release "
            "asset; execution status is tracked separately"
        ),
    }
    assert manifest["config_contract"] == {
        "SOLVER": "RANS",
        "KIND_TURB_MODEL": "SA",
        "RESTART_SOL": "YES",
        "RESTART_ITER": "499",
        "TIME_DOMAIN": "YES",
        "TIME_MARCHING": "DUAL_TIME_STEPPING-2ND_ORDER",
        "TIME_STEP": "5e-4",
        "INNER_ITER": "10",
        "MESH_FILENAME": "unsteady_naca0012_mesh.su2",
        "SOLUTION_FILENAME": "restart_flow",
    }
    assert manifest["mesh_contract"] == {
        "dimension": 2,
        "num_elements": 14336,
        "num_points": 14576,
        "element_type_counts": {"9": 14336},
        "marker_element_counts": {"airfoil": 128, "farfield": 352},
    }
    assert manifest["restart_contract"]["indices"] == [497, 498, 499]
    assert manifest["restart_contract"]["bdf2_history_indices"] == [497, 498]
    assert manifest["restart_contract"]["one_step_replay_target_index"] == 499
    assert manifest["restart_contract"]["fields"] == [
        "x",
        "y",
        "Density",
        "Momentum_x",
        "Momentum_y",
        "Energy",
        "Nu_Tilde",
        "Pressure",
        "Temperature",
        "Mach",
        "Pressure_Coefficient",
        "Laminar_Viscosity",
        "Skin_Friction_Coefficient_x",
        "Skin_Friction_Coefficient_y",
        "Heat_Flux",
        "Y_Plus",
        "Eddy_Viscosity",
    ]
    assert {
        name: record["sha256"] for name, record in manifest["resources"].items()
    } == {
        "license": "20c17d8b8c48a600800dfd14f95d5cb9ff47066a9641ddeab48dc54aec96e331",
        "config": "261ff97815812757f117a942b24e7bac3eb9eb197d1a7893228e25e00288e112",
        "mesh": "20dfc96875b1d1a8377c2e64248b6ee42a1ecf525c50d822fb5f539d06d68d15",
        "restart_00497": "dd3351369677bc8b7bcde6ddd49e8577260d9ed85cee435aba5efa03c5cc9366",
        "restart_00498": "0da7309382a7c10e3e3efca1ae0fa9d54d6632a4b1282652fe2393ea0518c696",
        "restart_00499": "09434c8be8970deefb35e67c573b207d4ba49f2938ce3218d1f2288aba632e21",
    }


def test_manifest_verification_fails_on_same_size_sha_change(tmp_path: Path):
    resource_dir, manifest_path = _write_bundle(tmp_path)
    config = resource_dir / "unsteady_naca0012.cfg"
    payload = bytearray(config.read_bytes())
    payload[-1] = ord(" ")
    config.write_bytes(payload)

    with pytest.raises(ValueError, match="SHA256 differs"):
        audit_unsteady_naca0012_bundle(
            resource_dir=resource_dir,
            manifest_path=manifest_path,
        )


def test_audit_rejects_manifest_that_weakens_bdf2_contract(tmp_path: Path):
    resource_dir, manifest_path = _write_bundle(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["config_contract"]["TIME_MARCHING"] = "DUAL_TIME_STEPPING-1ST_ORDER"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="frozen NACA contract"):
        audit_unsteady_naca0012_bundle(
            resource_dir=resource_dir,
            manifest_path=manifest_path,
        )


def test_audit_rejects_wrong_restart_field_order(tmp_path: Path):
    resource_dir, manifest_path = _write_bundle(tmp_path)
    restart_path = resource_dir / "restart_flow_00499.dat"
    payload = bytearray(restart_path.read_bytes())
    name_offset = 5 * 4
    first = name_offset + 2 * 33
    second = name_offset + 3 * 33
    payload[first : first + 33], payload[second : second + 33] = (
        payload[second : second + 33],
        payload[first : first + 33],
    )
    restart_path.write_bytes(payload)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["resources"]["restart_00499"]["sha256"] = _sha256(restart_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="field order differs"):
        audit_unsteady_naca0012_bundle(
            resource_dir=resource_dir,
            manifest_path=manifest_path,
        )


def test_binary_reader_checks_size_before_header_directed_allocation(tmp_path: Path):
    malformed = tmp_path / "oversized_header.dat"
    malformed.write_bytes(struct.pack("<5i", SU2_BINARY_MAGIC, 2**31 - 1, 4, 0, 0))

    with pytest.raises(ValueError, match="file size is inconsistent"):
        read_su2_binary_restart(malformed)


def test_binary_reader_rejects_truncation_and_nonfinite_values(tmp_path: Path):
    resource_dir, _ = _write_bundle(tmp_path)
    source = resource_dir / "restart_flow_00497.dat"
    truncated = tmp_path / "truncated.dat"
    truncated.write_bytes(source.read_bytes()[:-1])
    with pytest.raises(ValueError, match="file size is inconsistent"):
        read_su2_binary_restart(truncated)

    nonfinite = tmp_path / "nonfinite.dat"
    restart = read_su2_binary_restart(source)
    values = restart.values.copy()
    values[0, 2] = np.nan
    _write_restart(nonfinite, values)
    with pytest.raises(ValueError, match="nonfinite"):
        read_su2_binary_restart(nonfinite)


def test_replay_preparation_isolates_target_and_freezes_config(tmp_path: Path):
    resource_dir, manifest_path = _write_bundle(tmp_path)
    target = resource_dir / "restart_flow_00499.dat"
    target_sha256 = _sha256(target)
    output_dir = tmp_path / "replay_case"

    contract = prepare_unsteady_naca0012_replay_case(
        resource_dir=resource_dir,
        manifest_path=manifest_path,
        output_dir=output_dir,
    )

    assert contract["status"] == "prepared_not_executed"
    assert contract["native_solver_executed"] is False
    assert contract["external_reference_target"] == {
        "file": "restart_flow_00499.dat",
        "sha256": target_sha256,
        "copied_into_case": False,
    }
    assert contract["expected_output"] == "replay_flow_00499.dat"
    assert {path.name for path in output_dir.iterdir()} == {
        "LICENSE",
        "manifest.json",
        "replay.cfg",
        "replay_case.json",
        "restart_flow_00497.dat",
        "restart_flow_00498.dat",
        "unsteady_naca0012_mesh.su2",
        "upstream_unsteady_naca0012.cfg",
    }
    assert not (output_dir / "restart_flow_00499.dat").exists()
    assert not (output_dir / "replay_flow_00499.dat").exists()
    assert _sha256(target) == target_sha256

    replay_config = parse_su2_config(output_dir / "replay.cfg")
    assert replay_config["RESTART_ITER"] == "499"
    assert replay_config["TIME_ITER"] == "500"
    assert replay_config["SOLUTION_FILENAME"] == "restart_flow"
    assert replay_config["RESTART_FILENAME"] == "replay_flow"
    assert replay_config["WINDOW_CAUCHY_CRIT"] == "NO"
    assert replay_config["OUTPUT_FILES"] == "( RESTART )"
    assert replay_config["OUTPUT_WRT_FREQ"] == "( 1 )"
    assert replay_config["WRT_RESTART_COMPACT"] == "NO"


def test_replay_preparation_refuses_existing_or_resource_nested_output(tmp_path: Path):
    resource_dir, manifest_path = _write_bundle(tmp_path)
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        prepare_unsteady_naca0012_replay_case(
            resource_dir=resource_dir,
            manifest_path=manifest_path,
            output_dir=existing,
        )

    with pytest.raises(ValueError, match="outside the resource bundle"):
        prepare_unsteady_naca0012_replay_case(
            resource_dir=resource_dir,
            manifest_path=manifest_path,
            output_dir=resource_dir / "nested_replay",
        )


def test_replay_preparation_cleans_staging_after_mid_copy_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    resource_dir, manifest_path = _write_bundle(tmp_path)
    output_dir = tmp_path / "failed_replay"
    staging_dir = tmp_path / ".failed_replay.staging"
    real_copyfile = shutil.copyfile
    copy_count = 0

    def fail_second_copy(source, destination):
        nonlocal copy_count
        copy_count += 1
        if copy_count == 2:
            raise OSError("injected copy failure")
        return real_copyfile(source, destination)

    monkeypatch.setattr(
        "utility.time_dependent_no.su2_restart_contract.shutil.copyfile",
        fail_second_copy,
    )
    with pytest.raises(OSError, match="injected copy failure"):
        prepare_unsteady_naca0012_replay_case(
            resource_dir=resource_dir,
            manifest_path=manifest_path,
            output_dir=output_dir,
        )

    assert not output_dir.exists()
    assert not staging_dir.exists()


def test_lumped_mesh_measures_match_unit_square(tmp_path: Path):
    resource_dir, _ = _write_bundle(tmp_path)
    mesh = parse_su2_mesh(resource_dir / "unsteady_naca0012_mesh.su2")

    np.testing.assert_allclose(lumped_vertex_areas(mesh), np.full(4, 0.25))
    np.testing.assert_allclose(
        lumped_marker_lengths(mesh, "airfoil"),
        np.ones(4),
    )
    np.testing.assert_allclose(
        lumped_marker_lengths(mesh, "farfield"),
        np.ones(4),
    )


def test_restart_replay_metrics_are_hand_checkable(tmp_path: Path):
    resource_dir, _ = _write_bundle(tmp_path)
    reference = resource_dir / "restart_flow_00499.dat"
    candidate = tmp_path / "candidate.dat"
    restart = read_su2_binary_restart(reference)
    values = restart.values.copy()
    values[:, NACA_RESTART_FIELDS.index("Density")] += 1.0
    _write_restart(
        candidate,
        _native_restart_values(values),
        fields=native_replay.NACA_NATIVE_RESTART_FIELDS,
    )

    metrics = evaluate_restart_replay(
        mesh_path=resource_dir / "unsteady_naca0012_mesh.su2",
        reference_path=reference,
        candidate_path=candidate,
    )

    density = metrics["fields"]["Density"]
    assert density["error_weighted_rms"] == pytest.approx(1.0)
    assert density["weighted_relative_l2"] == pytest.approx(1.0 / 3.0)
    assert density["linf"] == pytest.approx(1.0)
    assert metrics["fields"]["Pressure"]["weighted_relative_l2"] is None
    assert metrics["dynamic_aggregate"][
        "component_balanced_relative_l2"
    ] == pytest.approx(1.0 / (3.0 * np.sqrt(5.0)))
    assert metrics["lift_drag_gate_complete"] is False


def test_weighted_metrics_use_nonuniform_weights():
    reference = np.zeros((3, len(NACA_RESTART_FIELDS)), dtype=np.float64)
    density_index = NACA_RESTART_FIELDS.index("Density")
    reference[:, density_index] = [1.0, 2.0, 4.0]
    candidate = reference.copy()
    candidate[:, density_index] += [1.0, 0.0, 2.0]

    metrics = native_replay._field_error_metrics(
        fields=("Density",),
        reference_values=reference,
        candidate_values=candidate,
        weights=np.asarray([1.0, 3.0, 6.0]),
    )["Density"]

    assert metrics["error_weighted_rms"] == pytest.approx(np.sqrt(25.0 / 10.0))
    assert metrics["weighted_relative_l2"] == pytest.approx(np.sqrt(25.0 / 109.0))
    assert metrics["linf"] == pytest.approx(2.0)


def test_surface_force_proxy_is_invariant_to_airfoil_orientation(tmp_path: Path):
    counterclockwise_path = tmp_path / "counterclockwise.su2"
    clockwise_path = tmp_path / "clockwise.su2"
    points = _write_mesh(counterclockwise_path)
    _write_mesh(clockwise_path, reverse_airfoil=True)
    values = np.zeros((4, len(NACA_RESTART_FIELDS)), dtype=np.float64)
    values[:, :2] = points
    values[:, NACA_RESTART_FIELDS.index("Pressure_Coefficient")] = [
        0.0,
        1.0,
        3.0,
        2.0,
    ]
    values[:, NACA_RESTART_FIELDS.index("Skin_Friction_Coefficient_x")] = [
        0.1,
        0.2,
        0.3,
        0.4,
    ]

    counterclockwise = native_replay._surface_force_proxy(
        parse_su2_mesh(counterclockwise_path),
        values,
        angle_of_attack_degrees=17.0,
    )
    clockwise = native_replay._surface_force_proxy(
        parse_su2_mesh(clockwise_path),
        values,
        angle_of_attack_degrees=17.0,
    )

    assert counterclockwise["airfoil_orientation"] == "counterclockwise"
    assert clockwise["airfoil_orientation"] == "clockwise"
    for key in (
        "body_force_x_proxy",
        "body_force_y_proxy",
        "streamwise_body_force_proxy",
        "cross_stream_body_force_proxy",
    ):
        assert counterclockwise[key] == pytest.approx(clockwise[key])


@pytest.mark.parametrize("failure_mode", ["header", "fields", "coordinates"])
def test_restart_replay_evaluator_rejects_interface_mismatch(
    tmp_path: Path, failure_mode: str
):
    resource_dir, _ = _write_bundle(tmp_path)
    reference = resource_dir / "restart_flow_00499.dat"
    candidate = tmp_path / "candidate.dat"
    restart = read_su2_binary_restart(reference)
    native_values = _native_restart_values(restart.values)
    if failure_mode == "coordinates":
        native_values[0, 0] += 1.0e-3
        _write_restart(
            candidate,
            native_values,
            fields=native_replay.NACA_NATIVE_RESTART_FIELDS,
        )
        expected = "candidate coordinates differ"
    else:
        _write_restart(
            candidate,
            native_values,
            fields=native_replay.NACA_NATIVE_RESTART_FIELDS,
        )
        payload = bytearray(candidate.read_bytes())
        if failure_mode == "header":
            struct.pack_into("<i", payload, 12, 1)
            expected = "unsupported nonzero reserved header"
        else:
            name_offset = 5 * 4
            first = name_offset + 2 * 33
            second = name_offset + 3 * 33
            payload[first : first + 33], payload[second : second + 33] = (
                payload[second : second + 33],
                payload[first : first + 33],
            )
            expected = "candidate restart has an unsupported field schema"
        candidate.write_bytes(payload)

    with pytest.raises(ValueError, match=expected):
        evaluate_restart_replay(
            mesh_path=resource_dir / "unsteady_naca0012_mesh.su2",
            reference_path=reference,
            candidate_path=candidate,
        )


def test_restart_replay_evaluator_name_aligns_exact_native_superset(tmp_path: Path):
    resource_dir, _ = _write_bundle(tmp_path)
    reference_path = resource_dir / "restart_flow_00499.dat"
    reference = read_su2_binary_restart(reference_path)
    native_fields = (
        *NACA_RESTART_FIELDS[:11],
        "Velocity_x",
        "Velocity_y",
        *NACA_RESTART_FIELDS[11:],
    )
    native_values = np.zeros((reference.num_points, len(native_fields)))
    for field in NACA_RESTART_FIELDS:
        native_values[:, native_fields.index(field)] = reference.values[
            :, NACA_RESTART_FIELDS.index(field)
        ]
    native_values[:, native_fields.index("Velocity_x")] = 1.0e9
    native_values[:, native_fields.index("Velocity_y")] = -1.0e9
    candidate = tmp_path / "native_candidate.dat"
    _write_restart(candidate, native_values, fields=native_fields)

    metrics = evaluate_restart_replay(
        mesh_path=resource_dir / "unsteady_naca0012_mesh.su2",
        reference_path=reference_path,
        candidate_path=candidate,
    )

    assert metrics["interface"]["candidate_extra_fields"] == [
        "Velocity_x",
        "Velocity_y",
    ]
    assert metrics["interface"]["canonical_fields_name_aligned"] is True
    assert metrics["dynamic_aggregate"] == {
        "component_balanced_relative_l2": 0.0,
        "maximum_dynamic_field_relative_l2": 0.0,
    }

    perturbed = tmp_path / "perturbed_native_candidate.dat"
    perturbed_values = native_values.copy()
    perturbed_values[:, native_fields.index("Density")] += 0.25
    _write_restart(perturbed, perturbed_values, fields=native_fields)
    perturbed_metrics = evaluate_restart_replay(
        mesh_path=resource_dir / "unsteady_naca0012_mesh.su2",
        reference_path=reference_path,
        candidate_path=perturbed,
    )
    assert perturbed_metrics["fields"]["Density"]["weighted_relative_l2"] > 0.0

    unsupported = tmp_path / "unsupported_candidate.dat"
    unsupported_fields = tuple(
        "Unsupported_Derived_Field" if field == "Velocity_y" else field
        for field in native_fields
    )
    _write_restart(unsupported, native_values, fields=unsupported_fields)
    with pytest.raises(ValueError, match="unsupported field schema"):
        evaluate_restart_replay(
            mesh_path=resource_dir / "unsteady_naca0012_mesh.su2",
            reference_path=reference_path,
            candidate_path=unsupported,
        )


@pytest.mark.parametrize(
    "invalid_fields",
    [
        (
            "Velocity_x",
            *NACA_RESTART_FIELDS,
            "Velocity_y",
        ),
        (
            *NACA_RESTART_FIELDS[:11],
            "Velocity_y",
            "Velocity_x",
            *NACA_RESTART_FIELDS[11:],
        ),
        (
            *NACA_RESTART_FIELDS[:11],
            "Velocity_x",
            *NACA_RESTART_FIELDS[11:],
        ),
        tuple(
            "Renamed_Density" if field == "Density" else field
            for field in NACA_RESTART_FIELDS
        ),
    ],
)
def test_restart_replay_rejects_near_miss_native_schemas(
    tmp_path: Path, invalid_fields: tuple[str, ...]
):
    resource_dir, _ = _write_bundle(tmp_path)
    reference = resource_dir / "restart_flow_00499.dat"
    candidate = tmp_path / "invalid_schema.dat"
    values = np.zeros((4, len(invalid_fields)), dtype=np.float64)
    _write_restart(candidate, values, fields=invalid_fields)

    with pytest.raises(ValueError, match="unsupported field schema"):
        evaluate_restart_replay(
            mesh_path=resource_dir / "unsteady_naca0012_mesh.su2",
            reference_path=reference,
            candidate_path=candidate,
        )


def test_restart_replay_keeps_reference_contract_strictly_canonical(tmp_path: Path):
    resource_dir, _ = _write_bundle(tmp_path)
    canonical = read_su2_binary_restart(resource_dir / "restart_flow_00499.dat")
    native_fields = native_replay.NACA_NATIVE_RESTART_FIELDS
    native_values = np.zeros((canonical.num_points, len(native_fields)))
    for field in NACA_RESTART_FIELDS:
        native_values[:, native_fields.index(field)] = canonical.values[
            :, NACA_RESTART_FIELDS.index(field)
        ]
    native_reference = tmp_path / "native_reference.dat"
    _write_restart(native_reference, native_values, fields=native_fields)

    with pytest.raises(ValueError, match="reference restart fields differ"):
        evaluate_restart_replay(
            mesh_path=resource_dir / "unsteady_naca0012_mesh.su2",
            reference_path=native_reference,
            candidate_path=resource_dir / "restart_flow_00499.dat",
        )


def test_restart_replay_rejects_canonical_schema_as_native_candidate(tmp_path: Path):
    resource_dir, _ = _write_bundle(tmp_path)
    canonical = resource_dir / "restart_flow_00499.dat"

    with pytest.raises(ValueError, match="candidate restart has an unsupported"):
        evaluate_restart_replay(
            mesh_path=resource_dir / "unsteady_naca0012_mesh.su2",
            reference_path=canonical,
            candidate_path=canonical,
        )


def test_restart_repeat_rejects_two_canonical_schema_outputs(tmp_path: Path):
    resource_dir, _ = _write_bundle(tmp_path)
    first = resource_dir / "restart_flow_00498.dat"
    second = resource_dir / "restart_flow_00499.dat"

    with pytest.raises(ValueError, match="first native output restart has an unsupported"):
        evaluate_restart_repeat(
            mesh_path=resource_dir / "unsteady_naca0012_mesh.su2",
            first_path=first,
            second_path=second,
        )


def test_native_replay_success_and_receipt_integrity(tmp_path: Path, monkeypatch):
    case_dir, target, executable, executable_sha256 = _prepare_runtime_fixture(tmp_path)
    monkeypatch.setattr(
        native_replay,
        "subprocess",
        SimpleNamespace(
            run=_fake_su2_run(target, case_dir, executable),
            TimeoutExpired=subprocess.TimeoutExpired,
        ),
    )

    receipt = run_native_naca0012_replay(
        case_dir=case_dir,
        target_path=target,
        executable_path=executable,
        expected_executable_sha256=executable_sha256,
        timeout_seconds=10.0,
    )

    assert receipt["status"] == "execution_and_evaluation_succeeded"
    assert receipt["execution_succeeded"] is True
    assert receipt["replay_evaluated"] is True
    assert receipt["trusted_transition_claimed"] is False
    assert receipt["evaluator_identity"]["unchanged_through_evaluation"] is True
    assert receipt["evaluator_identity"]["source_at_start"] == receipt[
        "evaluator_identity"
    ]["source_after_evaluation"]
    assert receipt["evaluation"]["dynamic_aggregate"] == {
        "component_balanced_relative_l2": 0.0,
        "maximum_dynamic_field_relative_l2": 0.0,
    }
    verified = load_verified_json_receipt(
        case_dir / NACA_NATIVE_RECEIPT_FILENAME,
        expected_schema=native_replay.NACA_NATIVE_REPLAY_SCHEMA,
    )
    assert verified["canonical_payload_sha256"] == receipt["canonical_payload_sha256"]


def test_native_replay_rejects_wrong_executable_before_launch(
    tmp_path: Path, monkeypatch
):
    case_dir, target, executable, _ = _prepare_runtime_fixture(tmp_path)
    launched = False

    def fail_if_called(*args, **kwargs):
        nonlocal launched
        launched = True
        raise AssertionError("subprocess must not be called")

    monkeypatch.setattr(native_replay.subprocess, "run", fail_if_called)
    with pytest.raises(ValueError, match="differs from the manifest"):
        run_native_naca0012_replay(
            case_dir=case_dir,
            target_path=target,
            executable_path=executable,
            expected_executable_sha256="f" * 64,
            timeout_seconds=10.0,
        )
    assert launched is False
    assert not (case_dir / NACA_NATIVE_RECEIPT_FILENAME).exists()


def test_native_replay_cross_binds_staged_manifest_before_launch(
    tmp_path: Path, monkeypatch
):
    case_dir, target, executable, executable_sha256 = _prepare_runtime_fixture(tmp_path)
    contract_path = case_dir / "replay_case.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["stage0_authority_binding"]["tutorial_commit"] = "rewritten"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    launched = False

    def fail_if_called(*args, **kwargs):
        nonlocal launched
        launched = True
        raise AssertionError("subprocess must not be called")

    monkeypatch.setattr(native_replay.subprocess, "run", fail_if_called)
    with pytest.raises(ValueError, match="differs from the staged resource manifest"):
        run_native_naca0012_replay(
            case_dir=case_dir,
            target_path=target,
            executable_path=executable,
            expected_executable_sha256=executable_sha256,
            timeout_seconds=10.0,
        )
    assert launched is False


def test_native_replay_preserves_release_probe_failure(tmp_path: Path, monkeypatch):
    case_dir, target, executable, executable_sha256 = _prepare_runtime_fixture(tmp_path)
    monkeypatch.setattr(
        native_replay,
        "subprocess",
        SimpleNamespace(
            run=_fake_su2_run(
                target,
                case_dir,
                executable,
                probe_version="8.4.0",
            ),
            TimeoutExpired=subprocess.TimeoutExpired,
        ),
    )

    with pytest.raises(NativeReplayError) as captured:
        run_native_naca0012_replay(
            case_dir=case_dir,
            target_path=target,
            executable_path=executable,
            expected_executable_sha256=executable_sha256,
            timeout_seconds=10.0,
        )

    receipt = load_verified_json_receipt(
        captured.value.receipt_path,
        expected_schema=native_replay.NACA_NATIVE_REPLAY_SCHEMA,
    )
    assert receipt["status"] == "release_probe_failed"
    assert receipt["process_attempted"] is False
    assert receipt["process_started"] is False
    assert receipt["runtime_identity"]["release_identity_verified"] is False
    assert receipt["runtime_identity"]["release_probe"]["version"] == "8.4.0"
    assert receipt["process"]["stdout"] is not None
    assert receipt["process"]["stderr"] is not None


def test_native_replay_distinguishes_launch_attempt_from_start(
    tmp_path: Path, monkeypatch
):
    case_dir, target, executable, executable_sha256 = _prepare_runtime_fixture(tmp_path)
    monkeypatch.setattr(
        native_replay,
        "subprocess",
        SimpleNamespace(
            run=_fake_su2_run(
                target,
                case_dir,
                executable,
                launch_error=True,
            ),
            TimeoutExpired=subprocess.TimeoutExpired,
        ),
    )

    with pytest.raises(NativeReplayError) as captured:
        run_native_naca0012_replay(
            case_dir=case_dir,
            target_path=target,
            executable_path=executable,
            expected_executable_sha256=executable_sha256,
            timeout_seconds=10.0,
        )

    receipt = load_verified_json_receipt(
        captured.value.receipt_path,
        expected_schema=native_replay.NACA_NATIVE_REPLAY_SCHEMA,
    )
    assert receipt["status"] == "process_failed"
    assert receipt["process_attempted"] is True
    assert receipt["process_started"] is False
    assert "injected launch failure" in receipt["process"]["error"]


@pytest.mark.parametrize("failure_mode", ["nonzero", "timeout", "ambiguous"])
def test_native_replay_preserves_failed_evidence(
    tmp_path: Path, monkeypatch, failure_mode: str
):
    case_dir, target, executable, executable_sha256 = _prepare_runtime_fixture(tmp_path)
    fake = _fake_su2_run(
        target,
        case_dir,
        executable,
        return_code=7 if failure_mode == "nonzero" else 0,
        time_out=failure_mode == "timeout",
        extra_output=failure_mode == "ambiguous",
    )
    monkeypatch.setattr(
        native_replay,
        "subprocess",
        SimpleNamespace(run=fake, TimeoutExpired=subprocess.TimeoutExpired),
    )

    with pytest.raises(NativeReplayError) as captured:
        run_native_naca0012_replay(
            case_dir=case_dir,
            target_path=target,
            executable_path=executable,
            expected_executable_sha256=executable_sha256,
            timeout_seconds=10.0,
        )

    assert captured.value.receipt_path == case_dir / NACA_NATIVE_RECEIPT_FILENAME
    receipt = load_verified_json_receipt(
        captured.value.receipt_path,
        expected_schema=native_replay.NACA_NATIVE_REPLAY_SCHEMA,
    )
    assert receipt["replay_evaluated"] is False
    assert receipt["trusted_transition_claimed"] is False
    assert receipt["process_started"] is True
    assert (case_dir / native_replay.NACA_STDOUT_FILENAME).is_file()
    assert (case_dir / native_replay.NACA_STDERR_FILENAME).is_file()


def test_native_replay_detects_target_mutation(tmp_path: Path, monkeypatch):
    case_dir, target, executable, executable_sha256 = _prepare_runtime_fixture(tmp_path)
    monkeypatch.setattr(
        native_replay,
        "subprocess",
        SimpleNamespace(
            run=_fake_su2_run(
                target,
                case_dir,
                executable,
                mutate_target=True,
            ),
            TimeoutExpired=subprocess.TimeoutExpired,
        ),
    )

    with pytest.raises(NativeReplayError):
        run_native_naca0012_replay(
            case_dir=case_dir,
            target_path=target,
            executable_path=executable,
            expected_executable_sha256=executable_sha256,
            timeout_seconds=10.0,
        )
    receipt = load_verified_json_receipt(
        case_dir / NACA_NATIVE_RECEIPT_FILENAME,
        expected_schema=native_replay.NACA_NATIVE_REPLAY_SCHEMA,
    )
    assert "canonical target changed" in receipt["evaluation_error"]
    assert receipt["trusted_transition_claimed"] is False


def test_native_replay_preserves_receipt_after_mesh_deletion(tmp_path: Path, monkeypatch):
    case_dir, target, executable, executable_sha256 = _prepare_runtime_fixture(tmp_path)
    monkeypatch.setattr(
        native_replay,
        "subprocess",
        SimpleNamespace(
            run=_fake_su2_run(
                target,
                case_dir,
                executable,
                delete_mesh=True,
            ),
            TimeoutExpired=subprocess.TimeoutExpired,
        ),
    )

    with pytest.raises(NativeReplayError) as captured:
        run_native_naca0012_replay(
            case_dir=case_dir,
            target_path=target,
            executable_path=executable,
            expected_executable_sha256=executable_sha256,
            timeout_seconds=10.0,
        )

    receipt = load_verified_json_receipt(
        captured.value.receipt_path,
        expected_schema=native_replay.NACA_NATIVE_REPLAY_SCHEMA,
    )
    assert receipt["status"] == "evaluation_failed"
    assert "prepared case input changed" in receipt["evaluation_error"]
    assert receipt["process_started"] is True


def test_native_replay_detects_source_change_during_evaluation(
    tmp_path: Path, monkeypatch
):
    case_dir, target, executable, executable_sha256 = _prepare_runtime_fixture(tmp_path)
    monkeypatch.setattr(
        native_replay,
        "subprocess",
        SimpleNamespace(
            run=_fake_su2_run(target, case_dir, executable),
            TimeoutExpired=subprocess.TimeoutExpired,
        ),
    )
    real_inventory = native_replay._evaluator_source_inventory
    inventory_calls = 0

    def changing_inventory():
        nonlocal inventory_calls
        inventory_calls += 1
        inventory = real_inventory()
        if inventory_calls >= 4:
            inventory = json.loads(json.dumps(inventory))
            inventory["su2_native_replay.py"]["sha256"] = "f" * 64
        return inventory

    monkeypatch.setattr(
        native_replay,
        "_evaluator_source_inventory",
        changing_inventory,
    )
    with pytest.raises(NativeReplayError) as captured:
        run_native_naca0012_replay(
            case_dir=case_dir,
            target_path=target,
            executable_path=executable,
            expected_executable_sha256=executable_sha256,
            timeout_seconds=10.0,
        )

    receipt = load_verified_json_receipt(
        captured.value.receipt_path,
        expected_schema=native_replay.NACA_NATIVE_REPLAY_SCHEMA,
    )
    assert receipt["status"] == "evaluation_failed"
    assert receipt["evaluation"] is not None
    assert receipt["output_contract_valid"] is False
    assert receipt["evaluator_identity"]["unchanged_through_evaluation"] is False
    assert "source changed during evaluation" in receipt["evaluation_error"]


def test_two_native_replays_compare_bitwise(tmp_path: Path, monkeypatch):
    case_a, case_b = _run_replay_pair(
        tmp_path,
        monkeypatch,
    )

    comparison = compare_native_replay_receipts(
        first_receipt_path=case_a / NACA_NATIVE_RECEIPT_FILENAME,
        second_receipt_path=case_b / NACA_NATIVE_RECEIPT_FILENAME,
        output_path=tmp_path / "comparison.json",
    )

    assert comparison["identity_match"] is True
    assert comparison["output_sha256_equal"] is True
    assert comparison["bitwise_deterministic"] is True
    assert comparison["numerical_repeat"]["dynamic_aggregate"] == {
        "component_balanced_relative_l2": 0.0,
        "maximum_dynamic_field_relative_l2": 0.0,
    }
    assert comparison["numerical_repeat"]["interface"]["reference_schema"] == (
        "su2_v8.5_native_19"
    )
    assert comparison["trusted_transition_claimed"] is False

    with pytest.raises(ValueError, match="two distinct receipts"):
        compare_native_replay_receipts(
            first_receipt_path=case_a / NACA_NATIVE_RECEIPT_FILENAME,
            second_receipt_path=case_a / NACA_NATIVE_RECEIPT_FILENAME,
            output_path=tmp_path / "same_receipt_comparison.json",
        )


def test_replay_comparison_revalidates_live_output(tmp_path: Path, monkeypatch):
    case_a, case_b = _run_replay_pair(tmp_path, monkeypatch)
    output_b = case_b / NACA_REPLAY_OUTPUT_FILENAME
    output_b.write_bytes(output_b.read_bytes() + b"post-receipt mutation")

    with pytest.raises(ValueError, match="output differs from its receipt"):
        compare_native_replay_receipts(
            first_receipt_path=case_a / NACA_NATIVE_RECEIPT_FILENAME,
            second_receipt_path=case_b / NACA_NATIVE_RECEIPT_FILENAME,
            output_path=tmp_path / "mutated_output_comparison.json",
        )


def test_replay_comparison_rejects_copied_case_evidence(tmp_path: Path, monkeypatch):
    case_a, _ = _run_replay_pair(tmp_path, monkeypatch)
    copied_case = tmp_path / "copied_case"
    shutil.copytree(case_a, copied_case)

    with pytest.raises(ValueError, match="copied or repeated run evidence"):
        compare_native_replay_receipts(
            first_receipt_path=case_a / NACA_NATIVE_RECEIPT_FILENAME,
            second_receipt_path=copied_case / NACA_NATIVE_RECEIPT_FILENAME,
            output_path=tmp_path / "copied_case_comparison.json",
        )


def test_replay_comparison_rejects_receipt_controlled_output_path(
    tmp_path: Path, monkeypatch
):
    case_a, case_b = _run_replay_pair(tmp_path, monkeypatch)
    receipt_b = case_b / NACA_NATIVE_RECEIPT_FILENAME
    _rewrite_receipt(
        receipt_b,
        lambda payload: payload["expected_output"].update(
            {"file": "../replay_flow_00499.dat"}
        ),
    )

    with pytest.raises(ValueError, match="unsafe output filename"):
        compare_native_replay_receipts(
            first_receipt_path=case_a / NACA_NATIVE_RECEIPT_FILENAME,
            second_receipt_path=receipt_b,
            output_path=tmp_path / "unsafe_path_comparison.json",
        )


def test_replay_comparison_reports_valid_nonbitwise_repeat(
    tmp_path: Path, monkeypatch
):
    case_a, case_b = _run_replay_pair(
        tmp_path,
        monkeypatch,
        second_dynamic_offset=0.25,
    )

    comparison = compare_native_replay_receipts(
        first_receipt_path=case_a / NACA_NATIVE_RECEIPT_FILENAME,
        second_receipt_path=case_b / NACA_NATIVE_RECEIPT_FILENAME,
        output_path=tmp_path / "nonbitwise_comparison.json",
    )

    assert comparison["output_sha256_equal"] is False
    assert comparison["bitwise_deterministic"] is False
    assert comparison["numerical_repeat"]["fields"]["Density"][
        "weighted_relative_l2"
    ] > 0.0


def test_restart_repeat_rejects_mismatched_output_schemas(tmp_path: Path):
    resource_dir, _ = _write_bundle(tmp_path)
    canonical_path = resource_dir / "restart_flow_00499.dat"
    canonical = read_su2_binary_restart(canonical_path)
    native_path = tmp_path / "native.dat"
    _write_restart(
        native_path,
        _native_restart_values(canonical.values),
        fields=native_replay.NACA_NATIVE_RESTART_FIELDS,
    )

    with pytest.raises(ValueError, match="field schemas differ"):
        evaluate_restart_repeat(
            mesh_path=resource_dir / "unsteady_naca0012_mesh.su2",
            first_path=native_path,
            second_path=canonical_path,
        )


def test_replay_comparison_rejects_identity_mismatch(tmp_path: Path, monkeypatch):
    case_a, case_b = _run_replay_pair(tmp_path, monkeypatch)
    receipt_b = case_b / NACA_NATIVE_RECEIPT_FILENAME
    _rewrite_receipt(
        receipt_b,
        lambda payload: payload["runtime_identity"]["selected_environment"].update(
            {"OMP_NUM_THREADS": "2"}
        ),
    )

    with pytest.raises(ValueError, match="selected_environment"):
        compare_native_replay_receipts(
            first_receipt_path=case_a / NACA_NATIVE_RECEIPT_FILENAME,
            second_receipt_path=receipt_b,
            output_path=tmp_path / "identity_mismatch_comparison.json",
        )
