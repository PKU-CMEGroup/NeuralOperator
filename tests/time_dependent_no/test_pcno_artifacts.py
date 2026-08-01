from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from utility.time_dependent_no.pcno_artifacts import (
    PCNO_SOURCE_SNAPSHOT_FILES,
    PCNO_SOURCE_SNAPSHOT_SCHEMA,
    atomic_torch_save,
    atomic_write_json,
    atomic_write_json_with_paths,
    digest_array,
    sha256_file,
    sha256_files,
    verify_source_snapshot,
    write_csv,
    write_csv_with_paths,
    write_json,
    write_source_snapshot,
)


def test_strict_and_safe_json_contracts_remain_distinct(tmp_path: Path) -> None:
    strict_path = tmp_path / "strict.json"
    with pytest.raises(ValueError, match="Out of range float"):
        write_json(strict_path, {"value": math.nan})
    assert not strict_path.exists()

    safe_path = tmp_path / "safe.json"
    atomic_write_json(
        safe_path,
        {
            "array": np.array([1.0, np.nan]),
            "value": np.float64(np.inf),
        },
    )
    assert json.loads(safe_path.read_text(encoding="utf-8")) == {
        "array": [1.0, None],
        "value": None,
    }

    residual_path = tmp_path / "residual_path.json"
    with pytest.raises(TypeError, match="not JSON serializable"):
        atomic_write_json(residual_path, {"path": tmp_path})

    resolution_path = tmp_path / "resolution_path.json"
    atomic_write_json_with_paths(resolution_path, {"path": tmp_path})
    assert json.loads(resolution_path.read_text(encoding="utf-8")) == {
        "path": str(tmp_path)
    }


def test_csv_writers_preserve_source_specific_behavior(tmp_path: Path) -> None:
    residual_path = tmp_path / "residual.csv"
    resolution_path = tmp_path / "resolution.csv"
    with pytest.raises(ValueError, match="cannot write empty CSV"):
        write_csv(residual_path, [])
    with pytest.raises(ValueError, match="cannot write an empty CSV"):
        write_csv_with_paths(resolution_path, [])

    common_rows = [
        {"first": 1, "payload": {"b": 2, "a": 1}},
        {"payload": [3, 4], "value": np.float64(np.nan)},
    ]
    write_csv(residual_path, common_rows)
    write_csv_with_paths(resolution_path, common_rows)
    assert residual_path.read_bytes() == resolution_path.read_bytes()
    with residual_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0]) == ["first", "payload", "value"]
    assert rows[0]["payload"] == '{"a":1,"b":2}'
    assert rows[1]["value"] == ""


def test_hash_and_atomic_torch_helpers_match_previous_contract(tmp_path: Path) -> None:
    payload_path = tmp_path / "payload.bin"
    payload_path.write_bytes(b"pcno-artifact")
    expected = hashlib.sha256(b"pcno-artifact").hexdigest()
    assert sha256_file(payload_path) == expected
    assert sha256_files([payload_path.name], root=tmp_path) == {
        payload_path.name: expected
    }

    base = np.arange(6, dtype=np.float32).reshape(2, 3)
    assert digest_array(base) == digest_array(base.copy())
    assert digest_array(base) != digest_array(base.astype(np.float64))

    checkpoint_path = tmp_path / "checkpoint.pt"
    atomic_torch_save({"tensor": torch.arange(3)}, checkpoint_path)
    loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert torch.equal(loaded["tensor"], torch.arange(3))
    assert not checkpoint_path.with_suffix(".pt.tmp").exists()


def test_source_snapshot_v2_is_explicit_and_v1_fails_closed(
    tmp_path: Path,
) -> None:
    snapshot = write_source_snapshot(tmp_path / "run")
    assert snapshot["schema"] == PCNO_SOURCE_SNAPSHOT_SCHEMA
    assert set(snapshot["files"]) == set(PCNO_SOURCE_SNAPSHOT_FILES)
    assert "utility/time_dependent_no/pcno_artifacts.py" in snapshot["files"]
    assert "utility/time_dependent_no/pcno_runtime.py" in snapshot["files"]
    assert "utility/time_dependent_no/pcno_rollout.py" in snapshot["files"]
    verify_source_snapshot(snapshot)

    legacy = json.loads(json.dumps(snapshot))
    legacy["schema"] = "pcno_euler2d_source_snapshot_v1"
    with pytest.raises(ValueError, match="unsupported PCNO source snapshot schema"):
        verify_source_snapshot(legacy)

    corrupted = json.loads(json.dumps(snapshot))
    first_source = next(iter(corrupted["files"]))
    corrupted["files"][first_source]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="current source differs"):
        verify_source_snapshot(corrupted)
