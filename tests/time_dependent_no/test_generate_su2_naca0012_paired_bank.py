from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest

from scripts.time_dependent_no import generate_su2_naca0012_paired_bank as bank
from scripts.time_dependent_no import (
    train_pcno_naca0012_corrective_extension as trainer,
)
from utility.time_dependent_no.pcno_naca0012 import NACANormalization
from utility.time_dependent_no.su2_naca0012_relabel import (
    RELABEL_RECEIPT_SCHEMA,
    REQUIRED_RECEIPT_CHECKS,
    write_fail_closed_receipt,
)


@pytest.fixture
def workspace_tmp_path():
    """Avoid the managed Windows runner's inaccessible system temp root."""

    path = Path(__file__).resolve().parent / f"paired_bank_runtime_{uuid4().hex}"
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path)


def _synthetic_authority(root: Path, num_nodes: int = 3) -> bank.pilot.PilotAuthority:
    root.mkdir(exist_ok=True)
    coordinates = np.column_stack(
        (
            np.linspace(0.0, 1.0, num_nodes, dtype=np.float64),
            np.linspace(-0.5, 0.5, num_nodes, dtype=np.float64),
        )
    )
    clean_states: dict[int, np.ndarray] = {}
    for index in range(955, 1195):
        state = np.empty((num_nodes, 5), dtype=np.float64)
        state[:, 0] = 1.0 + (index - 955) * 1.0e-5
        state[:, 1] = 0.05
        state[:, 2] = 0.02
        state[:, 3] = 3.0
        state[:, 4] = 0.1
        state.setflags(write=False)
        clean_states[index] = state
    normalization = NACANormalization(
        state_mean=np.zeros(5, dtype=np.float64),
        state_scale=np.asarray([0.1, 0.1, 0.1, 0.2, 0.05], dtype=np.float64),
        residual_scale=np.ones(5, dtype=np.float64),
        state_rms=np.ones(5, dtype=np.float64),
    )
    coordinates.setflags(write=False)
    placeholder = root / "placeholder"
    placeholder.write_bytes(b"x")
    return bank.pilot.PilotAuthority(
        extension={
            "experiment_id": bank.EXPERIMENT_ID,
            "paired_query_bank": {
                "gate": "all_relabeling_pilot_requirements_pass",
                "centers_inclusive": [956, 1193],
                "center_count": 238,
                "base_seed": 20260902,
                "directions_per_center": 1,
                "antithetic_signs": [-1, 1],
                "input_count": 476,
                "target": "one_step_fixed_su2_bdf2_continuation",
                "solver_failure_policy": "stop_without_drop_or_replacement",
                "input_admissibility_redraw": (
                    "deterministic_and_logged_before_solver_execution"
                ),
            },
        },
        normalization=normalization,
        clean_states=clean_states,
        native_frames={},
        coordinates=coordinates,
        config_path=placeholder.resolve(),
        mesh_path=placeholder.resolve(),
        executable_path=placeholder.resolve(),
        snapshots={},
        authority_summary={"synthetic": True},
    )


def _synthetic_verified_pilot(root: Path, num_nodes: int) -> bank.VerifiedPilot:
    all_directions, _, _ = bank.generate_primary_directions(num_nodes)
    selected = np.stack(
        [all_directions[center - 956] for center in bank.pilot.PILOT_CENTERS]
    )
    centers = np.asarray(bank.pilot.PILOT_CENTERS, dtype=np.int64)
    centers.setflags(write=False)
    selected.setflags(write=False)
    case_records = [
        {
            "case": case.to_mapping(),
            "relative_path": f"case_{case.ordinal:02d}/receipt.json",
            "status": "validation_succeeded",
            "scientifically_usable": True,
            "canonical_payload_sha256": "a" * 64,
        }
        for case in bank.pilot.pilot_cases()
    ]
    receipt = {
        "status": "pilot_succeeded",
        "scientifically_usable": True,
        "paired_query_bank_unlocked": True,
        "registered_case_count": 11,
        "case_receipts": case_records,
        "gates": {name: True for name in bank.EXPECTED_PILOT_GATES},
    }
    placeholder = root / "synthetic_pilot_final.json"
    placeholder.write_text("{}\n", encoding="utf-8")
    return bank.VerifiedPilot(
        root=root,
        final_manifest_path=placeholder.resolve(),
        final_manifest_sha256=bank.PILOT_FINAL_MANIFEST_SHA256,
        files={},
        receipt=receipt,
        centers=centers,
        directions=selected,
    )


def _build_pilot_packet(root: Path, num_nodes: int = 3) -> tuple[Path, str]:
    root.mkdir()
    case_index: list[dict[str, object]] = []
    for case in bank.pilot.pilot_cases():
        case_root = root / f"case_{case.ordinal:02d}"
        case_root.mkdir()
        receipt = write_fail_closed_receipt(
            (case_root / "receipt.json").resolve(),
            center=case.center,
            checks={name: True for name in REQUIRED_RECEIPT_CHECKS},
            details={"case": case.to_mapping()},
            error=None,
        )
        case_index.append(
            {
                "case": case.to_mapping(),
                "relative_path": f"case_{case.ordinal:02d}/receipt.json",
                "status": "validation_succeeded",
                "scientifically_usable": True,
                "canonical_payload_sha256": receipt["canonical_payload_sha256"],
            }
        )
    selected = bank.pilot.generate_pilot_directions(num_nodes)
    centers = np.asarray(bank.pilot.PILOT_CENTERS, dtype=np.int64)
    directions = np.stack([selected[center] for center in bank.pilot.PILOT_CENTERS])
    np.save(root / "pilot_centers.npy", centers, allow_pickle=False)
    np.save(root / "pilot_normalized_directions.npy", directions, allow_pickle=False)
    centers_file = bank.pilot._file_record(root / "pilot_centers.npy", root)
    directions_file = bank.pilot._file_record(
        root / "pilot_normalized_directions.npy", root
    )
    pilot_receipt = bank.pilot._self_hashed(
        {
            "schema": bank.pilot.PILOT_RECEIPT_SCHEMA,
            "experiment_id": bank.EXPERIMENT_ID,
            "status": "pilot_succeeded",
            "scientifically_usable": True,
            "paired_query_bank_unlocked": True,
            "error": None,
            "registered_case_count": 11,
            "attempted_solver_call_count": 11,
            "completed_case_receipt_count": 11,
            "schedule": [case.to_mapping() for case in bank.pilot.pilot_cases()],
            "case_receipts": case_index,
            "directions": {
                "centers": centers_file,
                "normalized_directions": {
                    **directions_file,
                    "base_seed": bank.BASE_SEED,
                    "stream": "one_ascending_center_stream_956_through_1193",
                    **bank.pilot._array_record(directions),
                },
            },
            "gates": {name: True for name in bank.EXPECTED_PILOT_GATES},
            "online_solver_calls": False,
            "prospective_opened": False,
            "sealed_opened": False,
        }
    )
    bank.pilot._atomic_json(root / bank.pilot.PILOT_RECEIPT_FILENAME, pilot_receipt)
    _, final_sha = bank.pilot._write_final_hash_manifest(root)
    return root / bank.pilot.PILOT_FINAL_HASH_FILENAME, final_sha


def test_primary_direction_stream_exactly_matches_pilot_selection() -> None:
    values, _, states = bank.generate_primary_directions(4)
    pilot_values = bank.pilot.generate_pilot_directions(4)

    assert values.shape == (238, 2, 4, 5)
    assert states["initial"] != states["after_primary"]
    for center in bank.pilot.PILOT_CENTERS:
        observed = values[center - 956]
        expected = pilot_values[center]
        assert np.array_equal(observed.view(np.uint32), expected.view(np.uint32))


def test_redraw_uses_post_primary_stream_and_only_thermodynamic_gate(
    workspace_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority = _synthetic_authority(workspace_tmp_path / "authority")
    original = bank.naca_state_admissibility
    calls = 0

    def force_one_nonpilot_density_failure(value: np.ndarray):
        nonlocal calls
        result = original(value)
        if calls == 4:  # first history slot of center 957, after center 956's 4 checks
            result = json.loads(json.dumps(result))
            result["density"]["passed"] = False
            result["density"]["minimum_finite"] = -1.0
        calls += 1
        return result

    monkeypatch.setattr(
        bank, "naca_state_admissibility", force_one_nonpilot_density_failure
    )
    selected = bank.prepare_admissible_directions(authority)

    assert selected.redraw_counts[1] == 1
    assert selected.draw_indices[1] == 238
    assert selected.total_draw_count == 239
    assert selected.redraw_log[0]["center"] == 957
    assert selected.redraw_log[0]["rejected_attempts"][0]["reason"] == (
        "nonpositive_density_or_ideal_gas_pressure"
    )
    for center in bank.pilot.PILOT_CENTERS:
        assert selected.redraw_counts[center - 956] == 0


def test_successful_pilot_validator_rehashes_all_eight_gates_and_cases(
    workspace_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    final_path, final_sha = _build_pilot_packet(workspace_tmp_path / "pilot")
    monkeypatch.setattr(bank, "PILOT_FINAL_MANIFEST_SHA256", final_sha)

    verified = bank.verify_successful_pilot(final_path.resolve())

    assert verified.final_manifest_sha256 == final_sha
    assert verified.directions.shape == (3, 2, 3, 5)
    assert set(verified.receipt["gates"]) == bank.EXPECTED_PILOT_GATES
    assert all(verified.receipt["gates"].values())
    assert len(verified.receipt["case_receipts"]) == 11


def _patch_synthetic_run(
    monkeypatch: pytest.MonkeyPatch,
    authority: bank.pilot.PilotAuthority,
    verified_pilot: bank.VerifiedPilot,
    *,
    fail_ordinal: int | None = None,
) -> list[tuple[int, int, np.ndarray]]:
    calls: list[tuple[int, int, np.ndarray]] = []
    monkeypatch.setattr(bank, "_load_bank_authority", lambda **_: authority)
    monkeypatch.setattr(bank, "verify_successful_pilot", lambda _: verified_pilot)
    monkeypatch.setattr(bank, "_reverify_pilot", lambda _: True)
    monkeypatch.setattr(bank, "_reverify_bank_authority", lambda _: True)

    def fake_run_case(
        *, staging_root, authority, case, direction, timeout_seconds, process_runner
    ):
        del timeout_seconds
        case_root = staging_root / case.name
        case_root.mkdir()
        process_runner(["synthetic-su2"], cwd=case_root)
        calls.append((case.center, case.sign, np.array(direction, copy=True)))
        succeeded = case.ordinal != fail_ordinal
        source = bank._self_hashed(
            {
                "schema": RELABEL_RECEIPT_SCHEMA,
                "experiment_id": bank.EXPERIMENT_ID,
                "status": "validation_succeeded" if succeeded else "validation_failed",
                "scientifically_usable": succeeded,
                "error": None if succeeded else "synthetic failure",
                "index_contract": bank.pilot.relabel_step_index(
                    case.center
                ).to_mapping(),
                "checks": {name: succeeded for name in REQUIRED_RECEIPT_CHECKS},
                "details": {"case": case.to_mapping()},
                "nu_tilde_acceptance_gate": False,
                "prospective_opened": False,
                "sealed_opened": False,
            }
        )
        output = (
            np.asarray(authority.clean_states[case.center + 1], dtype=np.float64)
            + case.sign * 1.0e-4
            if succeeded
            else None
        )
        return bank.pilot.CaseExecution(
            case=case,
            directory=case_root,
            receipt=source,
            output_state=output,
            output_path=None,
            realized_field_rms=None,
        )

    monkeypatch.setattr(bank.pilot, "_run_case", fake_run_case)
    return calls


def _run_args(root: Path, output: Path) -> dict[str, object]:
    return {
        "extension_contract_path": (root / "extension.json").resolve(),
        "extension_preregistration_path": (root / "prereg.md").resolve(),
        "successor_contract_path": (root / "successor.json").resolve(),
        "baseline_contract_path": (root / "baseline.json").resolve(),
        "dataset_dir": (root / "dataset").resolve(),
        "calibration_dir": (root / "calibration").resolve(),
        "trajectory_dir": (root / "trajectory").resolve(),
        "resource_manifest_path": (root / "resource.json").resolve(),
        "resource_dir": (root / "resources").resolve(),
        "executable_path": (root / "su2.exe").resolve(),
        "pilot_final_manifest_path": (root / "pilot.json").resolve(),
        "output_dir": output.resolve(),
        "timeout_seconds": 1.0,
        "process_runner": lambda *_args, **_kwargs: SimpleNamespace(returncode=0),
    }


def test_full_packet_has_exact_476_calls_and_raw_float32_schema(
    workspace_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority = _synthetic_authority(workspace_tmp_path / "authority")
    verified_pilot = _synthetic_verified_pilot(workspace_tmp_path, 3)
    calls = _patch_synthetic_run(monkeypatch, authority, verified_pilot)
    prepare = bank.prepare_admissible_directions

    def stable_small_fixture(value):
        result = prepare(value)
        return bank.replace(
            result,
            realized_field_rms_ratio=np.ones_like(result.realized_field_rms_ratio),
        )

    monkeypatch.setattr(bank, "prepare_admissible_directions", stable_small_fixture)
    output = workspace_tmp_path / "bank"

    result = bank.generate_su2_naca0012_paired_bank(
        **_run_args(workspace_tmp_path, output)
    )

    receipt = result["receipt"]
    assert receipt["status"] == "complete"
    assert receipt["attempted_solver_call_count"] == 476
    assert receipt["completed_case_receipt_count"] == 476
    assert [(center, sign) for center, sign, _ in calls] == [
        (center, sign) for center in bank.TRAIN_CENTERS for sign in bank.SIGNS
    ]
    assert len(list((output / "case_receipts").glob("*.json"))) == 476
    assert not (output / "_solver_work").exists()
    with np.load(output / bank.BANK_ARRAY_FILE, allow_pickle=False) as archive:
        assert tuple(archive.files) == bank.BANK_ARRAY_KEYS
        assert archive["coordinates"].dtype == np.float32
        assert archive["normalized_directions"].shape == (238, 2, 3, 5)
        assert archive["displaced_previous"].shape == (2, 238, 3, 5)
        assert archive["displaced_current"].dtype == np.float32
        assert archive["solver_future"].dtype == np.float32
        clean = np.asarray(authority.clean_states[955], dtype=np.float32)
        scale = np.asarray(authority.normalization.state_scale, dtype=np.float32)
        expected_minus = clean - scale * archive["normalized_directions"][0, 0]
        expected_plus = clean + scale * archive["normalized_directions"][0, 0]
        assert np.array_equal(
            archive["displaced_previous"][0, 0].view(np.uint32),
            expected_minus.view(np.uint32),
        )
        assert np.array_equal(
            archive["displaced_previous"][1, 0].view(np.uint32),
            expected_plus.view(np.uint32),
        )
    final = json.loads((output / bank.BANK_FINAL_FILE).read_text(encoding="utf-8"))
    assert len(final["files"]) == 478
    assert result["packet"]["final_hash_manifest_sha256"] == bank.sha256_file(
        output / bank.BANK_FINAL_FILE
    )
    monkeypatch.setattr(
        trainer,
        "_validate_relabel_pilot_final_manifest",
        lambda *_args, **_kwargs: None,
    )

    def reject_host_rng_replay(*_args, **_kwargs):
        raise AssertionError("paired-bank loading must not replay host RNG")

    monkeypatch.setattr(trainer.torch, "Generator", reject_host_rng_replay)
    monkeypatch.setattr(trainer.torch, "randn", reject_host_rng_replay)
    verified = trainer._load_paired_bank(
        output,
        expected_final_sha256=result["packet"]["final_hash_manifest_sha256"],
        pilot_final_manifest_path=verified_pilot.final_manifest_path,
        expected_pilot_final_sha256=verified_pilot.final_manifest_sha256,
        expected_coordinates=authority.coordinates,
        train_states=np.stack(
            [authority.clean_states[index] for index in range(955, 1195)]
        ),
        train_frame_indices=np.arange(955, 1195, dtype=np.int64),
        normalization=authority.normalization,
    )
    assert verified.normalized_directions.shape == (238, 2, 3, 5)
    assert verified.direction_draw_indices.tolist() == list(range(238))

    arrays_path = output / bank.BANK_ARRAY_FILE
    corrupted = bytearray(arrays_path.read_bytes())
    corrupted[-1] ^= 1
    arrays_path.write_bytes(corrupted)
    with pytest.raises(ValueError, match="artifact hash differs: paired_bank.npz"):
        trainer._load_paired_bank(
            output,
            expected_final_sha256=result["packet"]["final_hash_manifest_sha256"],
            pilot_final_manifest_path=verified_pilot.final_manifest_path,
            expected_pilot_final_sha256=verified_pilot.final_manifest_sha256,
            expected_coordinates=authority.coordinates,
            train_states=np.stack(
                [authority.clean_states[index] for index in range(955, 1195)]
            ),
            train_frame_indices=np.arange(955, 1195, dtype=np.int64),
            normalization=authority.normalization,
        )


def test_solver_failure_stops_without_drop_replacement_or_array_publication(
    workspace_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority = _synthetic_authority(workspace_tmp_path / "authority_failure")
    verified_pilot = _synthetic_verified_pilot(workspace_tmp_path, 3)
    calls = _patch_synthetic_run(monkeypatch, authority, verified_pilot, fail_ordinal=2)
    output = workspace_tmp_path / "bank_failure"

    result = bank.generate_su2_naca0012_paired_bank(
        **_run_args(workspace_tmp_path, output)
    )

    receipt = result["receipt"]
    assert len(calls) == 3
    assert receipt["attempted_solver_call_count"] == 3
    assert receipt["completed_case_receipt_count"] == 3
    assert receipt["status"] == "failed_incomplete_not_scientific_evidence"
    assert receipt["scientifically_usable"] is False
    assert "remaining cases were not run" in receipt["error"]
    assert not (output / bank.BANK_ARRAY_FILE).exists()
    assert (output / "_solver_work").is_dir()
