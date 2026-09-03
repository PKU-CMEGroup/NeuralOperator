from __future__ import annotations

import csv
import struct
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.time_dependent_no import run_su2_naca0012_relabel_pilot as pilot
from utility.time_dependent_no.pcno_naca0012 import NACANormalization
from utility.time_dependent_no.su2_naca_trajectory import NACA_HISTORY_FIELDS
from utility.time_dependent_no.su2_native_replay import (
    NACA_NATIVE_RESTART_FIELDS,
)
from utility.time_dependent_no.su2_restart_contract import (
    NACA_DYNAMIC_FIELDS,
    SU2_BINARY_MAGIC,
    SU2_FIELD_NAME_BYTES,
    parse_su2_config,
    read_su2_binary_restart,
    sha256_file,
)

_DYNAMIC_INDICES = tuple(
    NACA_NATIVE_RESTART_FIELDS.index(field) for field in NACA_DYNAMIC_FIELDS
)


def _write_restart(path: Path, values: np.ndarray) -> None:
    header = (
        SU2_BINARY_MAGIC,
        len(NACA_NATIVE_RESTART_FIELDS),
        values.shape[0],
        0,
        0,
    )
    with path.open("wb") as handle:
        handle.write(struct.pack("<5i", *header))
        for field in NACA_NATIVE_RESTART_FIELDS:
            encoded = field.encode("ascii")
            handle.write(encoded + b"\0" * (SU2_FIELD_NAME_BYTES - len(encoded)))
        handle.write(np.ascontiguousarray(values, dtype="<f8").tobytes(order="C"))


def _native_values(
    coordinates: np.ndarray, state: np.ndarray, *, auxiliary_offset: float = 0.0
) -> np.ndarray:
    values = np.empty(
        (coordinates.shape[0], len(NACA_NATIVE_RESTART_FIELDS)), dtype=np.float64
    )
    for column in range(values.shape[1]):
        values[:, column] = auxiliary_offset + 10.0 * column
    values[:, NACA_NATIVE_RESTART_FIELDS.index("x")] = coordinates[:, 0]
    values[:, NACA_NATIVE_RESTART_FIELDS.index("y")] = coordinates[:, 1]
    for field_index, field in enumerate(NACA_DYNAMIC_FIELDS):
        values[:, NACA_NATIVE_RESTART_FIELDS.index(field)] = state[:, field_index]
    return values


def _write_history(path: Path, time_iter: int) -> None:
    row: list[float | int] = []
    for field in NACA_HISTORY_FIELDS:
        if field == "Time_Iter":
            row.append(time_iter)
        elif field == "Inner_Iter":
            row.append(9)
        else:
            row.append(-4.0)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(NACA_HISTORY_FIELDS)
        writer.writerow(row)


def _synthetic_authority(tmp_path: Path, num_nodes: int = 1024) -> pilot.PilotAuthority:
    source = tmp_path / "source"
    source.mkdir()
    coordinates = np.column_stack(
        (
            np.linspace(0.0, 1.0, num_nodes, dtype=np.float64),
            np.linspace(-0.5, 0.5, num_nodes, dtype=np.float64),
        )
    )
    required_indices = sorted(
        {
            index
            for center in pilot.PILOT_CENTERS
            for index in (center - 1, center, center + 1)
        }
    )
    clean_states: dict[int, np.ndarray] = {}
    frames: dict[int, pilot.NativeFrame] = {}
    for index in required_indices:
        offset = float(index - 955) * 1.0e-5
        state = np.empty((num_nodes, 5), dtype=np.float64)
        state[:, 0] = 1.0 + offset
        state[:, 1] = 0.05 + offset
        state[:, 2] = 0.02 - offset
        state[:, 3] = 3.0 + offset
        state[:, 4] = 0.10 + offset
        state.setflags(write=False)
        clean_states[index] = state
        path = source / f"trajectory_flow_{index:05d}.dat"
        _write_restart(path, _native_values(coordinates, state))
        frames[index] = pilot.NativeFrame(
            index=index,
            path=path.resolve(),
            bytes=path.stat().st_size,
            sha256=sha256_file(path),
        )
    config = source / "unsteady_naca0012.cfg"
    config.write_text(
        """SOLVER= RANS
KIND_TURB_MODEL= SA
RESTART_SOL= YES
RESTART_ITER= 499
TIME_DOMAIN= YES
TIME_MARCHING= DUAL_TIME_STEPPING-2ND_ORDER
TIME_STEP= 5e-4
INNER_ITER= 10
CONV_FIELD= RMS_DENSITY
CONV_RESIDUAL_MINVAL= -8
MESH_FILENAME= unsteady_naca0012_mesh.su2
SOLUTION_FILENAME= restart_flow
TIME_ITER= 500
RESTART_FILENAME= restart_flow
OUTPUT_FILES= ( RESTART, PARAVIEW )
OUTPUT_WRT_FREQ= ( 1, 1 )
""",
        encoding="utf-8",
    )
    mesh = source / "unsteady_naca0012_mesh.su2"
    mesh.write_text("synthetic fixed mesh\n", encoding="utf-8")
    executable = source / "SU2_CFD.exe"
    executable.write_bytes(b"synthetic executable")
    snapshots = {
        "config": pilot._snapshot(config, "config"),
        "mesh": pilot._snapshot(mesh, "mesh"),
        "executable": pilot._snapshot(executable, "executable"),
        **{
            f"frame_{index}": pilot._snapshot(frame.path, f"frame {index}")
            for index, frame in frames.items()
        },
    }
    normalization = NACANormalization(
        state_mean=np.zeros(5, dtype=np.float64),
        state_scale=np.asarray([0.1, 0.1, 0.1, 0.2, 0.05], dtype=np.float64),
        residual_scale=np.ones(5, dtype=np.float64),
        state_rms=np.ones(5, dtype=np.float64),
    )
    coordinates.setflags(write=False)
    return pilot.PilotAuthority(
        extension={"experiment_id": pilot.EXPERIMENT_ID},
        normalization=normalization,
        clean_states=clean_states,
        native_frames=frames,
        coordinates=coordinates,
        config_path=config.resolve(),
        mesh_path=mesh.resolve(),
        executable_path=executable.resolve(),
        snapshots=snapshots,
        authority_summary={"synthetic": True},
    )


def _fake_runner(
    authority: pilot.PilotAuthority,
    *,
    fail_call: int | None = None,
    extra_output_call: int | None = None,
    repeat_auxiliary_offset: float = 0.0,
    auxiliary_dynamic_offset: float = 0.0,
):
    calls: list[list[str]] = []
    directions = pilot.generate_pilot_directions(authority.coordinates.shape[0])

    def run(argv, **kwargs):
        calls.append(list(argv))
        case_root = kwargs["cwd"]
        assert argv == [
            str(authority.executable_path),
            "--threads",
            "1",
            pilot.PILOT_CONFIG_FILENAME,
        ]
        assert kwargs["shell"] is False
        assert kwargs["check"] is False
        assert kwargs["env"]["OMP_NUM_THREADS"] == "1"
        assert kwargs["env"]["OMP_DYNAMIC"] == "FALSE"
        if fail_call is not None and len(calls) == fail_call:
            return SimpleNamespace(returncode=7)

        config = parse_su2_config(case_root / pilot.PILOT_CONFIG_FILENAME)
        center = int(config["RESTART_ITER"]) - 1
        assert config["TIME_ITER"] == str(center + 2)
        assert config["INNER_ITER"] == "10"
        assert config["CONV_FIELD"] == "REL_RMS_DENSITY"
        assert config["CONV_RESIDUAL_MINVAL"] == "-3.0"
        assert config["SOLUTION_FILENAME"] == "relabel_input"
        assert config["RESTART_FILENAME"] == "relabel_output"
        if "_sign_m1_" in case_root.name:
            sign = -1
        elif "_sign_p1_" in case_root.name:
            sign = 1
        else:
            sign = 0
        if case_root.name.endswith("auxiliary_swap"):
            variant = "auxiliary_swap"
        elif case_root.name.endswith("repeat"):
            variant = "repeat"
        else:
            variant = "base"
        expected_previous, expected_current, _ = pilot._input_states(
            authority,
            pilot.PilotCase(0, center, sign, variant),
            directions[center],
        )
        for index, expected in zip(
            (center - 1, center),
            (expected_previous, expected_current),
            strict=True,
        ):
            staged = read_su2_binary_restart(
                case_root / f"relabel_input_{index:05d}.dat"
            )
            observed = np.ascontiguousarray(staged.values[:, _DYNAMIC_INDICES])
            expected64 = np.ascontiguousarray(expected, dtype=np.float64)
            assert np.array_equal(observed.view(np.uint64), expected64.view(np.uint64))
        state = np.array(authority.clean_states[center + 1], copy=True)
        if sign:
            state += sign * 0.001 * authority.normalization.state_scale
        if variant == "auxiliary_swap":
            state += auxiliary_dynamic_offset * authority.normalization.state_scale
            auxiliary_offset = 1.0
        elif variant == "repeat":
            auxiliary_offset = repeat_auxiliary_offset
        else:
            auxiliary_offset = 0.0
        output = case_root / f"relabel_output_{center + 1:05d}.dat"
        _write_restart(
            output,
            _native_values(
                authority.coordinates,
                state,
                auxiliary_offset=auxiliary_offset,
            ),
        )
        if extra_output_call is not None and len(calls) == extra_output_call:
            _write_restart(
                case_root / "unexpected_restart.dat",
                _native_values(authority.coordinates, state),
            )
        _write_history(case_root / f"history_{center + 1:05d}.csv", center + 1)
        return SimpleNamespace(returncode=0)

    run.calls = calls
    return run


def _run_arguments(tmp_path: Path, output: Path) -> dict[str, Path]:
    return {
        "extension_contract_path": (tmp_path / "extension.json").resolve(),
        "extension_preregistration_path": (tmp_path / "prereg.md").resolve(),
        "successor_contract_path": (tmp_path / "successor.json").resolve(),
        "baseline_contract_path": (tmp_path / "baseline.json").resolve(),
        "dataset_dir": (tmp_path / "dataset").resolve(),
        "calibration_dir": (tmp_path / "calibration").resolve(),
        "trajectory_dir": (tmp_path / "trajectory").resolve(),
        "resource_manifest_path": (tmp_path / "resource.json").resolve(),
        "resource_dir": (tmp_path / "resources").resolve(),
        "executable_path": (tmp_path / "unused.exe").resolve(),
        "output_dir": output.resolve(),
    }


def test_live_extension_contract_matches_frozen_hash_and_semantics() -> None:
    docs = pilot.REPO_ROOT / "docs/time_dependent_no"

    value = pilot._load_extension_contract(
        docs / "B3B4_NACA_CORRECTIVE_EXTENSION_CONTRACT.json",
        docs / "B3B4_NACA_CORRECTIVE_EXTENSION_PREREGISTRATION.md",
    )

    assert value["experiment_id"] == pilot.EXPERIMENT_ID
    assert value["preregistration_sha256"] == pilot.EXTENSION_PREREGISTRATION_SHA256


def test_parent_successor_loader_uses_complete_authority(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    successor_path = tmp_path / "B3B4_NACA_CORRECTIVE_SUCCESSOR_CONTRACT.json"
    baseline_path = tmp_path / "R0_NACA_PCNO_BASELINE_CONTRACT.json"
    observed: list[tuple[Path, Path, Path]] = []

    def fake_load(path: Path, preregistration: Path, baseline: Path) -> dict[str, str]:
        observed.append((path, preregistration, baseline))
        return {"status": "verified"}

    monkeypatch.setattr(pilot.successor_trainer, "_load_successor_contract", fake_load)
    monkeypatch.setattr(
        pilot,
        "_snapshot",
        lambda path, label: SimpleNamespace(sha256="bound-sha256"),
    )

    value, digest = pilot._load_bound_successor_contract(successor_path, baseline_path)

    assert value == {"status": "verified"}
    assert digest == "bound-sha256"
    assert observed == [
        (
            successor_path,
            tmp_path / "B3B4_NACA_CORRECTIVE_SUCCESSOR_PREREGISTRATION.md",
            baseline_path,
        )
    ]


def test_schedule_is_exactly_nine_base_repeat_and_auxiliary() -> None:
    cases = pilot.pilot_cases()

    assert len(cases) == 11
    assert [(case.center, case.sign, case.variant) for case in cases[:9]] == [
        (center, sign, "base")
        for center in pilot.PILOT_CENTERS
        for sign in pilot.PILOT_SIGNS
    ]
    assert (cases[9].center, cases[9].sign, cases[9].variant) == (1075, 1, "repeat")
    assert (cases[10].center, cases[10].sign, cases[10].variant) == (
        1075,
        1,
        "auxiliary_swap",
    )


def test_direction_uses_one_full_ascending_center_torch_stream() -> None:
    num_nodes = 3
    selected = pilot.generate_pilot_directions(num_nodes)
    generator = torch.Generator(device="cpu").manual_seed(pilot.BASE_SEED)
    scale = torch.tensor(pilot.PER_FIELD_NORMALIZED_STD, dtype=torch.float32)
    expected: dict[int, np.ndarray] = {}
    for center in pilot.TRAIN_CENTERS:
        draw = torch.randn(
            (2, num_nodes, 5),
            dtype=torch.float32,
            generator=generator,
        )
        draw.mul_(scale)
        if center in pilot.PILOT_CENTERS:
            expected[center] = draw.numpy().copy()

    assert tuple(selected) == pilot.PILOT_CENTERS
    for center in pilot.PILOT_CENTERS:
        assert np.array_equal(
            selected[center].view(np.uint32), expected[center].view(np.uint32)
        )


def test_input_states_use_previous_current_float32_pair_and_fixed_sign(
    tmp_path: Path,
) -> None:
    authority = _synthetic_authority(tmp_path, 3)
    direction = np.linspace(-0.002, 0.002, 30, dtype=np.float32).reshape(2, 3, 5)
    clean = np.stack(
        (authority.clean_states[1074], authority.clean_states[1075]), axis=0
    ).astype(np.float32)
    scale = np.asarray(authority.normalization.state_scale, dtype=np.float32).reshape(
        1, 1, -1
    )

    previous, current, _ = pilot._input_states(
        authority, pilot.PilotCase(0, 1075, 1, "base"), direction
    )
    expected = clean + scale * direction
    assert previous.dtype == np.float32
    assert current.dtype == np.float32
    assert np.array_equal(previous.view(np.uint32), expected[0].view(np.uint32))
    assert np.array_equal(current.view(np.uint32), expected[1].view(np.uint32))

    zero_previous, zero_current, _ = pilot._input_states(
        authority, pilot.PilotCase(0, 1075, 0, "base"), direction
    )
    assert np.array_equal(zero_previous.view(np.uint32), clean[0].view(np.uint32))
    assert np.array_equal(zero_current.view(np.uint32), clean[1].view(np.uint32))


def test_synthetic_pilot_runs_exactly_eleven_cases_and_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority = _synthetic_authority(tmp_path)
    runner = _fake_runner(authority)
    monkeypatch.setattr(pilot, "_load_pilot_authority", lambda **_: authority)
    output = tmp_path / "pilot_success"

    result = pilot.run_su2_naca0012_relabel_pilot(
        **_run_arguments(tmp_path, output),
        timeout_seconds=10.0,
        process_runner=runner,
    )

    receipt = result["receipt"]
    assert len(runner.calls) == 11
    assert receipt["status"] == "pilot_succeeded"
    assert receipt["scientifically_usable"] is True
    assert receipt["paired_query_bank_unlocked"] is True
    assert receipt["attempted_solver_call_count"] == 11
    assert receipt["completed_case_receipt_count"] == 11
    assert all(receipt["gates"].values())
    assert receipt["metrics"]["trusted_response"]["median"] == pytest.approx(0.001)
    assert receipt["metrics"]["repeatability"] == {
        "full_native_output_file_bytes_equal": True,
        "five_evolved_output_arrays_bitwise_equal": True,
        "history_policy": "convergence_contract_only_not_bitwise",
        "passed": True,
    }
    auxiliary = receipt["metrics"]["auxiliary_probe"]
    assert auxiliary["five_evolved_output_arrays_bitwise_equal"] is True
    assert auxiliary["full_native_output_file_bytes_equal_record_only"] is False
    assert (output / pilot.PILOT_RECEIPT_FILENAME).is_file()
    assert (output / pilot.PILOT_FINAL_HASH_FILENAME).is_file()
    assert (output / "pilot_normalized_directions.npy").is_file()

    aux_receipt = receipt["case_receipts"][10]
    aux_payload = pilot._load_self_hashed_json(
        output / aux_receipt["relative_path"],
        "time_dependent_no.naca_su2_relabel_receipt.v1",
    )
    templates = [
        record["template"]["file"]
        for record in aux_payload["details"]["restart_writes"]
    ]
    assert templates == ["trajectory_flow_01075.dat", "trajectory_flow_01074.dat"]


def test_first_failed_case_stops_without_drop_or_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority = _synthetic_authority(tmp_path)
    runner = _fake_runner(authority, fail_call=2)
    monkeypatch.setattr(pilot, "_load_pilot_authority", lambda **_: authority)
    output = tmp_path / "pilot_failure"

    result = pilot.run_su2_naca0012_relabel_pilot(
        **_run_arguments(tmp_path, output),
        timeout_seconds=10.0,
        process_runner=runner,
    )

    receipt = result["receipt"]
    assert len(runner.calls) == 2
    assert receipt["status"] == "pilot_failed"
    assert receipt["scientifically_usable"] is False
    assert receipt["paired_query_bank_unlocked"] is False
    assert receipt["attempted_solver_call_count"] == 2
    assert receipt["completed_case_receipt_count"] == 2
    assert "remaining cases were not run" in receipt["error"]
    assert len(list(output.glob("case_*"))) == 2
    assert receipt["case_receipts"][0]["scientifically_usable"] is True
    assert receipt["case_receipts"][1]["scientifically_usable"] is False


def test_attempted_call_is_retained_when_case_receipt_write_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority = _synthetic_authority(tmp_path)
    runner = _fake_runner(authority)
    monkeypatch.setattr(pilot, "_load_pilot_authority", lambda **_: authority)

    def fail_receipt(*_: object, **__: object) -> None:
        raise RuntimeError("receipt write failed")

    monkeypatch.setattr(pilot, "write_fail_closed_receipt", fail_receipt)
    output = tmp_path / "pilot_receipt_failure"

    result = pilot.run_su2_naca0012_relabel_pilot(
        **_run_arguments(tmp_path, output),
        timeout_seconds=10.0,
        process_runner=runner,
    )

    receipt = result["receipt"]
    assert len(runner.calls) == 1
    assert receipt["attempted_solver_call_count"] == 1
    assert receipt["completed_case_receipt_count"] == 0
    assert receipt["scientifically_usable"] is False
    assert "receipt write failed" in receipt["error"]


def test_unexpected_restart_file_fails_exact_output_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority = _synthetic_authority(tmp_path)
    runner = _fake_runner(authority, extra_output_call=1)
    monkeypatch.setattr(pilot, "_load_pilot_authority", lambda **_: authority)
    output = tmp_path / "pilot_extra_output_failure"

    result = pilot.run_su2_naca0012_relabel_pilot(
        **_run_arguments(tmp_path, output),
        timeout_seconds=10.0,
        process_runner=runner,
    )

    receipt = result["receipt"]
    assert len(runner.calls) == 1
    assert receipt["completed_case_receipt_count"] == 1
    assert receipt["scientifically_usable"] is False
    case = pilot._load_self_hashed_json(
        output / receipt["case_receipts"][0]["relative_path"],
        "time_dependent_no.naca_su2_relabel_receipt.v1",
    )
    assert case["checks"]["exactly_one_expected_output"] is False


@pytest.mark.parametrize(
    ("runner_options", "failed_gate"),
    [
        ({"repeat_auxiliary_offset": 2.0}, "repeatability"),
        ({"auxiliary_dynamic_offset": 1.0e-4}, "auxiliary_invariance"),
    ],
)
def test_comparison_mismatch_fails_closed_after_all_cases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    runner_options: dict[str, float],
    failed_gate: str,
) -> None:
    authority = _synthetic_authority(tmp_path)
    runner = _fake_runner(authority, **runner_options)
    monkeypatch.setattr(pilot, "_load_pilot_authority", lambda **_: authority)
    output = tmp_path / f"pilot_{failed_gate}_failure"

    result = pilot.run_su2_naca0012_relabel_pilot(
        **_run_arguments(tmp_path, output),
        timeout_seconds=10.0,
        process_runner=runner,
    )

    receipt = result["receipt"]
    assert len(runner.calls) == 11
    assert receipt["status"] == "pilot_failed"
    assert receipt["scientifically_usable"] is False
    assert receipt["paired_query_bank_unlocked"] is False
    assert receipt["gates"][failed_gate] is False
    assert all(item["scientifically_usable"] for item in receipt["case_receipts"])
