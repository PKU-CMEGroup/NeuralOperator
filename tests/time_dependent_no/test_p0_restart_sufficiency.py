from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from utility.time_dependent_no.p0_restart_sufficiency import (
    A1_REQUIRED_SOURCE_MEMBERS,
    A1_SOURCE_MANIFEST_SCHEMA,
    BOUNDARY_BALANCE_TOLERANCE,
    FIXED_CASES,
    NATIVE_NODES,
    boundary_balance_residual,
    canonical_json_sha256,
    prepare_solver_input,
    replay_native_layout,
    require_boundary_balance,
    resolve_manifest_member,
    sha256_file,
    validate_a1_source_manifest,
    validate_case_manifest,
    validate_execution_contract,
    validate_synthetic_admissibility_rejection,
    verify_artifact_member,
)


def _case_manifest() -> dict[str, object]:
    return {
        "schema": "p0_restart_sufficiency_case_manifest_v1",
        "dataset_manifest_sha256": (
            "f8d228ae3c6e08fe6697f37df21476fe621abc354d0b0a6eed2a623ed2b9f96c"
        ),
        "open_grouped_split_digest": (
            "ae4be7f0e5fcfb305106173625c62c9a96634a716061207afb35fc9bda724d5a"
        ),
        "eligible_validation_count": 24,
        "access_receipt": {
            "historical_test_members_referenced": False,
            "only_selected_state_paths_referenced": True,
            "selected_splits": ["validation"],
            "strength_ood_members_referenced": False,
        },
        "selection_contract": {
            "assigned_frames": [10, 30, 50],
            "population": "open validation only",
            "selection_performed_before_frame_bytes_were_read": True,
            "sort_key": 'sha256("P0-RS-20260825|" + trajectory_id)',
            "take": 3,
        },
        "cases": copy.deepcopy(list(FIXED_CASES)),
    }


def _execution_contract() -> dict[str, object]:
    return {
        "schema": "p0_restart_sufficiency_execution_contract_v1",
        "state": {
            "channel_order": ["rho", "rho*u", "rho*v", "E"],
            "channel_scales": [
                0.0752340287,
                0.0546168404,
                0.0435805767,
                0.2345080528,
            ],
            "domain": [[0.0, 2.0], [0.0, 1.0]],
            "flattening": "index = y * 250 + x",
            "gamma": 1.4,
            "input_transform": "none",
            "shape": [25000, 4],
        },
        "common_horizon": {
            "d044": "two raw stride-1 residual calls",
            "d060": "one raw stride-2 residual call",
            "delta_t": 0.02,
            "native_solver": {"output_times": [0.0, 0.02], "t_final": 0.02},
            "reference": "stored frame f+2",
            "stored_frame_spacing": 0.01,
        },
        "source_import_roots_must_remain_separate": {
            "historical_model_evaluator": "historical",
            "native_solver": "native",
            "reason": "different registered revisions",
        },
    }


def test_shape_order_channel_replay_preserves_exact_bytes() -> None:
    state = np.empty((NATIVE_NODES, 4), dtype="<f4")
    flat_index = np.arange(NATIVE_NODES, dtype=np.float32)
    for channel in range(4):
        state[:, channel] = 10_000_000.0 * channel + flat_index
    prepared = prepare_solver_input(state)
    replayed = replay_native_layout(prepared)
    assert prepared is not state
    assert prepared.dtype.str == "<f4"
    assert prepared.flags.c_contiguous
    assert prepared.tobytes(order="C") == state.tobytes(order="C")
    np.testing.assert_array_equal(replayed, state)
    assert replayed.reshape(100, 250, 4)[17, 23, 2] == state[17 * 250 + 23, 2]


@pytest.mark.parametrize(
    ("shape", "dtype", "message"),
    [
        ((NATIVE_NODES, 3), np.float32, "shape"),
        ((NATIVE_NODES, 4), np.float64, "float32"),
    ],
)
def test_solver_input_rejects_shape_or_dtype_drift(
    shape: tuple[int, int], dtype: np.dtype, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        prepare_solver_input(np.ones(shape, dtype=dtype))


def test_synthetic_admissibility_rejection_is_fail_closed() -> None:
    checks = validate_synthetic_admissibility_rejection()
    assert all(checks.values())


def test_one_stride_timing_contract_rejects_schedule_drift() -> None:
    contract = _execution_contract()
    validate_execution_contract(contract)
    drifted = copy.deepcopy(contract)
    drifted["common_horizon"]["d044"] = "one raw stride-2 residual call"
    with pytest.raises(ValueError, match="timing"):
        validate_execution_contract(drifted)


def test_boundary_accounting_uses_registered_relative_residual() -> None:
    delta = np.asarray([2.0, -1.0, 0.5, 4.0])
    assert require_boundary_balance(delta, -delta) == 0.0
    bad_exchange = -delta + np.asarray([1.0e-8, 0.0, 0.0, 0.0])
    observed = boundary_balance_residual(delta, bad_exchange)
    assert observed > BOUNDARY_BALANCE_TOLERANCE
    with pytest.raises(ValueError, match="exceeds"):
        require_boundary_balance(delta, bad_exchange)


def test_case_manifest_rejects_reselection_or_population_escape() -> None:
    manifest = _case_manifest()
    assert validate_case_manifest(manifest) == FIXED_CASES
    reselected = copy.deepcopy(manifest)
    reselected["cases"][0]["frame"] = 11
    with pytest.raises(ValueError, match="drifted"):
        validate_case_manifest(reselected)
    escaped = copy.deepcopy(manifest)
    escaped["access_receipt"]["historical_test_members_referenced"] = True
    with pytest.raises(ValueError, match="access boundary"):
        validate_case_manifest(escaped)


def test_physically_opened_artifact_member_is_rehashed() -> None:
    source_root = Path(__file__).resolve().parents[2]
    relative = "tests/time_dependent_no/test_p0_restart_sufficiency.py"
    path = source_root / relative
    members = {
        relative: {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    }
    assert (
        verify_artifact_member(
            inputs_root=source_root,
            artifact_members=members,
            relative_path=relative,
        )
        == path
    )
    members[relative]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="bytes drifted"):
        verify_artifact_member(
            inputs_root=source_root,
            artifact_members=members,
            relative_path=relative,
        )


def test_source_manifest_rejects_hash_drift_and_unmanifested_python(
    tmp_path: Path,
) -> None:
    for relative in A1_REQUIRED_SOURCE_MEMBERS:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {relative}\n", encoding="utf-8")
    members = {
        relative: sha256_file(tmp_path / relative)
        for relative in sorted(A1_REQUIRED_SOURCE_MEMBERS)
    }
    manifest = {
        "schema": A1_SOURCE_MANIFEST_SCHEMA,
        "member_count": len(members),
        "members": members,
        "canonical_member_mapping_sha256": canonical_json_sha256(members),
    }
    assert validate_a1_source_manifest(manifest, source_root=tmp_path) == members

    drifted = copy.deepcopy(manifest)
    drifted["members"][next(iter(members))] = "0" * 64
    drifted["canonical_member_mapping_sha256"] = canonical_json_sha256(
        drifted["members"]
    )
    with pytest.raises(ValueError, match="drifted"):
        validate_a1_source_manifest(drifted, source_root=tmp_path)

    extra = tmp_path / "unregistered.py"
    extra.write_text("# not in manifest\n", encoding="utf-8")
    with pytest.raises(ValueError, match="inventory differ"):
        validate_a1_source_manifest(manifest, source_root=tmp_path)


def test_manifest_path_resolution_rejects_escape(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "inside.txt").write_text("inside", encoding="utf-8")
    assert (
        resolve_manifest_member(root, "inside.txt") == (root / "inside.txt").resolve()
    )
    with pytest.raises(ValueError, match="unsafe"):
        resolve_manifest_member(root, "../outside.txt")
    with pytest.raises(ValueError, match="unsafe"):
        resolve_manifest_member(root, str((tmp_path / "outside.txt").resolve()))


def test_case_manifest_fixture_is_json_roundtrip_stable() -> None:
    manifest = _case_manifest()
    assert json.loads(json.dumps(manifest, sort_keys=True)) == manifest
