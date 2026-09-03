from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from scripts.time_dependent_no import calibrate_pcno_naca0012_successor as calibration
from scripts.time_dependent_no.calibrate_pcno_naca0012_successor import (
    _build_arrays,
    _load_successor_contract,
    _preflight_open_dataset,
    _reverify_bound_inputs,
    _unique_undirected_edges,
    _verify_staged_packet,
)
from scripts.time_dependent_no.train_pcno_naca0012 import EXPECTED_DATASET_FILES
from utility.time_dependent_no.pcno_naca0012 import (
    BASELINE_CONTRACT_SHA256,
    NACANormalization,
    load_naca_baseline_contract,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_preflight_packet(
    root: Path, *, train_frame_count: int = 240
) -> tuple[dict, str]:
    inherited_sha256 = "a" * 64
    roles = {
        "train": {
            "owned_frame_indices_inclusive": [955, 1194],
            "frame_count": train_frame_count,
            "dense_transition_center_indices_inclusive": [956, 1193],
            "dense_transition_count": 238,
        },
        "development": {
            "owned_frame_indices_inclusive": [1233, 1472],
            "frame_count": 240,
            "dense_transition_center_indices_inclusive": [1234, 1471],
            "dense_transition_count": 238,
        },
    }
    manifest = calibration._self_hashed(
        {
            "schema": calibration.DATASET_SCHEMA,
            "status": "complete",
            "contract": {"sha256": inherited_sha256},
            "access": {
                "materialized_population_roles": ["train", "development"],
                "diagnostic_sets": ["native_replay_margin"],
                "prospective_opened": False,
                "sealed_opened": False,
            },
            "roles": roles,
        }
    )
    manifest_bytes = (
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    (root / "dataset_manifest.json").write_bytes(manifest_bytes)
    final = {
        "schema": calibration.FINAL_HASH_MANIFEST_SCHEMA,
        "contract_sha256": inherited_sha256,
        "files": {
            "dataset_manifest.json": {
                "relative_path": "dataset_manifest.json",
                "bytes": len(manifest_bytes),
                "sha256": sha256(manifest_bytes).hexdigest(),
            }
        },
        "self_hash_excluded": True,
        "prospective_opened": False,
        "sealed_opened": False,
    }
    final_bytes = (
        json.dumps(final, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    (root / "final_hash_manifest.json").write_bytes(final_bytes)
    final_sha256 = sha256(final_bytes).hexdigest()
    successor = {
        "inherited_r0_contract_sha256": inherited_sha256,
        "dataset": {
            "final_hash_manifest_sha256": final_sha256,
            "train_frames_inclusive": [955, 1194],
            "train_centers_inclusive": [956, 1193],
            "development_frames_inclusive": [1233, 1472],
            "development_centers_inclusive": [1234, 1471],
        },
    }
    return successor, final_sha256


def _file_record(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "relative_path": path.name,
        "bytes": len(payload),
        "sha256": sha256(payload).hexdigest(),
    }


def _write_bound_input_packets(
    root: Path,
) -> tuple[Path, str, list[tuple[int, Path, dict, str]]]:
    dataset = root / "dataset"
    dataset.mkdir()
    for name in EXPECTED_DATASET_FILES:
        (dataset / name).write_bytes(f"dataset:{name}\n".encode())
    dataset_final = {
        "schema": calibration.FINAL_HASH_MANIFEST_SCHEMA,
        "contract_sha256": BASELINE_CONTRACT_SHA256,
        "files": {
            name: _file_record(dataset / name) for name in EXPECTED_DATASET_FILES
        },
        "self_hash_excluded": True,
        "prospective_opened": False,
        "sealed_opened": False,
    }
    calibration._write_json(dataset / "final_hash_manifest.json", dataset_final)
    dataset_sha256 = sha256(
        (dataset / "final_hash_manifest.json").read_bytes()
    ).hexdigest()

    parents: list[tuple[int, Path, dict, str]] = []
    for seed in calibration.EXPECTED_PARENT_SEEDS:
        packet = root / f"parent_{seed}"
        packet.mkdir()
        (packet / "best.pt").write_bytes(f"checkpoint:{seed}\n".encode())
        (packet / "summary.json").write_bytes(f"summary:{seed}\n".encode())
        files = {
            name: _file_record(packet / name) for name in ("best.pt", "summary.json")
        }
        (packet / "final_hash_manifest.json").write_bytes(
            f"parent-final:{seed}\n".encode()
        )
        final_sha256 = sha256(
            (packet / "final_hash_manifest.json").read_bytes()
        ).hexdigest()
        parents.append((seed, packet, files, final_sha256))
    return dataset, dataset_sha256, parents


def _write_staged_packet(root: Path) -> tuple[dict, dict]:
    root.mkdir()
    arrays = {
        "recovery__iid_field_rms": np.arange(5, dtype=np.float32),
        "path__normalized_path": np.zeros((2, 3, 5), dtype=np.float64),
    }
    arrays_path = root / "calibration_arrays.npz"
    np.savez(arrays_path, **arrays)
    metadata = calibration._self_hashed(
        {
            "schema": "synthetic.calibration.v1",
            "status": "complete",
            "experiment_id": calibration.EXPERIMENT_ID,
            "successor_contract_sha256": (calibration.SUCCESSOR_CONTRACT_FILE_SHA256),
            "arrays": {
                "file": _file_record(arrays_path),
                "members": {
                    name: calibration._array_record(value)
                    for name, value in arrays.items()
                },
            },
        }
    )
    metadata_path = root / "calibration.json"
    calibration._write_json(metadata_path, metadata)
    final = {
        "schema": "synthetic.final.v1",
        "experiment_id": calibration.EXPERIMENT_ID,
        "packet_role": "calibration",
        "successor_contract_sha256": calibration.SUCCESSOR_CONTRACT_FILE_SHA256,
        "files": {
            "calibration.json": _file_record(metadata_path),
            "calibration_arrays.npz": _file_record(arrays_path),
        },
        "self_hash_excluded": True,
        "prospective_opened": False,
        "sealed_opened": False,
    }
    calibration._write_json(root / "final_hash_manifest.json", final)
    return metadata, final


def _normalization() -> NACANormalization:
    return NACANormalization(
        state_mean=np.zeros(5),
        state_scale=np.ones(5),
        residual_scale=np.ones(5),
        state_rms=np.ones(5),
    )


def test_successor_contract_and_preregistration_are_bound() -> None:
    value, digest = _load_successor_contract(
        REPO_ROOT
        / "docs/time_dependent_no/B3B4_NACA_CORRECTIVE_SUCCESSOR_CONTRACT.json"
    )
    assert value["experiment_id"] == "B3B4_NACA_CM_20260901A"
    assert len(digest) == 64
    assert value["protection"]["prospective_opened"] is False
    assert value["protection"]["sealed_opened"] is False


def test_wrong_successor_contract_fails_before_json_parsing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = tmp_path / "successor.json"
    contract.write_text("{}", encoding="utf-8")
    parser = Mock(side_effect=AssertionError("wrong contract must not be parsed"))
    monkeypatch.setattr(calibration.json, "loads", parser)

    with pytest.raises(ValueError, match="bytes differ from the frozen contract"):
        _load_successor_contract(contract)
    parser.assert_not_called()


def test_source_mutation_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "probe.py"
    source.write_bytes(b"before\n")
    records = calibration._read_source_records(("probe.py",), tmp_path)
    source.write_bytes(b"after\n")
    monkeypatch.setattr(calibration, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(calibration, "SOURCE_FILES", ("probe.py",))
    monkeypatch.setattr(calibration, "SOURCE_RECORDS_AT_IMPORT", records)

    with pytest.raises(RuntimeError, match="source files changed"):
        calibration._verify_live_sources()


def test_open_dataset_preflight_rejects_wrong_role_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    successor, final_sha256 = _write_preflight_packet(tmp_path, train_frame_count=239)
    monkeypatch.setattr(calibration, "DATASET_FINAL_HASH_MANIFEST_SHA256", final_sha256)

    with pytest.raises(ValueError, match="train role or frame inventory differs"):
        _preflight_open_dataset(tmp_path, successor)


def test_wrong_dataset_identity_fails_before_state_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "final_hash_manifest.json").write_text("{}", encoding="utf-8")
    inherited_sha256 = "a" * 64
    successor = {
        "inherited_r0_contract_sha256": inherited_sha256,
        "dataset": {
            "final_hash_manifest_sha256": (
                calibration.DATASET_FINAL_HASH_MANIFEST_SHA256
            )
        },
    }
    state_loader = Mock(side_effect=AssertionError("state loader must not run"))
    monkeypatch.setattr(calibration, "_verify_live_sources", Mock())
    monkeypatch.setattr(
        calibration,
        "_load_successor_contract",
        Mock(return_value=(successor, calibration.SUCCESSOR_CONTRACT_FILE_SHA256)),
    )
    monkeypatch.setattr(calibration, "sha256_file", Mock(return_value=inherited_sha256))
    monkeypatch.setattr(calibration, "load_naca_baseline_contract", Mock())
    monkeypatch.setattr(calibration, "validate_successor_math_contract", Mock())
    monkeypatch.setattr(calibration, "_load_dataset", state_loader)
    arguments = argparse.Namespace(
        successor_contract=tmp_path / "unused-successor.json",
        baseline_contract=tmp_path / "unused-baseline.json",
        dataset_dir=tmp_path,
    )

    with pytest.raises(ValueError, match="final-hash manifest differs"):
        calibration._run(arguments)
    state_loader.assert_not_called()


def test_bound_input_packets_reject_dataset_and_parent_mutation(tmp_path: Path) -> None:
    dataset, dataset_sha256, parents = _write_bound_input_packets(tmp_path)
    _reverify_bound_inputs(dataset, dataset_sha256, parents)

    dataset_member = dataset / "train_states.npy"
    dataset_member.write_bytes(b"mutated dataset\n")
    with pytest.raises(ValueError, match="verified packet artifact changed"):
        _reverify_bound_inputs(dataset, dataset_sha256, parents)
    dataset_member.write_bytes(b"dataset:train_states.npy\n")

    parent_checkpoint = parents[0][1] / "best.pt"
    parent_checkpoint.write_bytes(b"mutated checkpoint\n")
    with pytest.raises(ValueError, match="verified packet artifact changed"):
        _reverify_bound_inputs(dataset, dataset_sha256, parents)


def test_staged_packet_rejects_tampering_and_extra_files(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    metadata, final = _write_staged_packet(staging)
    _verify_staged_packet(staging, metadata, final)

    arrays_path = staging / "calibration_arrays.npz"
    original_arrays = arrays_path.read_bytes()
    arrays_path.write_bytes(b"tampered arrays\n")
    with pytest.raises(ValueError, match="file hash differs"):
        _verify_staged_packet(staging, metadata, final)
    arrays_path.write_bytes(original_arrays)

    (staging / "unexpected.bin").write_bytes(b"unexpected\n")
    with pytest.raises(ValueError, match="inventory differs"):
        _verify_staged_packet(staging, metadata, final)


def test_unique_edges_remove_only_symmetric_direction_duplicates() -> None:
    directed = np.array([[0, 1], [1, 0], [1, 2], [2, 1]], dtype=np.int64)
    result = _unique_undirected_edges(directed, 3)
    np.testing.assert_array_equal(result, np.array([[0, 1], [1, 2]]))


def test_structured_noise_matches_iid_energy_on_synthetic_bank() -> None:
    rng = np.random.default_rng(23)
    error_banks = {
        seed: rng.normal(size=(238, 3, 5)).astype(np.float32) for seed in (17, 29, 43)
    }
    train_states = rng.normal(size=(238, 3, 5)).astype(np.float64)
    arrays, summary = _build_arrays(
        error_banks=error_banks,
        train_states=train_states,
        train_frames=np.arange(956, 1194, dtype=np.int64),
        normalization=_normalization(),
        parent_contract=load_naca_baseline_contract(
            REPO_ROOT / "docs/time_dependent_no/R0_NACA_PCNO_BASELINE_CONTRACT.json"
        ),
    )
    assert arrays["recovery__iid_field_rms"].shape == (5,)
    assert arrays["recovery__structured_basis"].shape == (16, 2, 3, 5)
    assert arrays["recovery__structured_eigenvalues"].shape == (16,)
    assert arrays["path__normalized_path"].shape == (238, 3, 5)
    np.testing.assert_allclose(
        np.sum(np.square(arrays["recovery__structured_coefficient_std"])),
        summary["iid_expected_history_pair_squared_norm"],
        rtol=1e-6,
    )
    assert summary["structured_sampling_mean"].startswith("zero")
