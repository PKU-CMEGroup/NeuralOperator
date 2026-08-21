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
    PCNO_SOURCE_PROVENANCE_FILES,
    PCNO_SOURCE_SNAPSHOT_FILES,
    PCNO_SOURCE_SNAPSHOT_SCHEMA,
    PCNO_SOURCE_SNAPSHOT_V2_FILES,
    PCNO_SOURCE_SNAPSHOT_V2_SCHEMA,
    PCNO_SOURCE_SNAPSHOT_V3_FILES,
    PCNO_SOURCE_SNAPSHOT_V3_SCHEMA,
    PCNO_SOURCE_SNAPSHOT_V4_FILES,
    PCNO_SOURCE_SNAPSHOT_V4_SCHEMA,
    PCNO_SOURCE_SNAPSHOT_V5_FILES,
    PCNO_SOURCE_SNAPSHOT_V5_SCHEMA,
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


def test_source_snapshot_v6_covers_boundary_fields_and_separates_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = write_source_snapshot(tmp_path / "run")
    assert snapshot["schema"] == PCNO_SOURCE_SNAPSHOT_SCHEMA
    assert set(snapshot["files"]) == set(PCNO_SOURCE_SNAPSHOT_FILES)
    assert snapshot["extra_source_files"] == []
    assert set(snapshot["provenance_files"]) == set(PCNO_SOURCE_PROVENANCE_FILES)
    assert not set(snapshot["files"]) & set(snapshot["provenance_files"])
    assert "utility/time_dependent_no/pcno_artifacts.py" in snapshot["files"]
    assert "utility/time_dependent_no/pcno_runtime.py" in snapshot["files"]
    assert "utility/time_dependent_no/pcno_rollout.py" in snapshot["files"]
    assert "pcno/geo_utility.py" in snapshot["files"]
    assert "utility/time_dependent_no/euler2d.py" in snapshot["files"]
    assert "utility/time_dependent_no/errors.py" in snapshot["files"]
    assert "utility/time_dependent_no/pcno_boundary_fields.py" in snapshot["files"]
    assert (
        "docs/time_dependent_no/MECHANISTIC_DIAGNOSTIC_TRACKER.md"
        in snapshot["provenance_files"]
    )
    verify_source_snapshot(snapshot)

    original_sha256_file = sha256_file

    def reject_provenance_read(path: str | Path) -> str:
        relative = Path(path).resolve().relative_to(Path(__file__).resolve().parents[2])
        assert relative.as_posix() not in PCNO_SOURCE_PROVENANCE_FILES
        return original_sha256_file(path)

    monkeypatch.setattr(
        "utility.time_dependent_no.pcno_artifacts.sha256_file",
        reject_provenance_read,
    )
    verify_source_snapshot(snapshot)

    corrupted_source_digest = json.loads(json.dumps(snapshot))
    corrupted_source_digest["source_set_digest"] = "0" * 64
    with pytest.raises(ValueError, match="source-set digest mismatch"):
        verify_source_snapshot(corrupted_source_digest)

    corrupted_provenance = json.loads(json.dumps(snapshot))
    provenance_path = PCNO_SOURCE_PROVENANCE_FILES[0]
    corrupted_provenance["provenance_files"][provenance_path]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="provenance-set digest mismatch"):
        verify_source_snapshot(corrupted_provenance)

    legacy_v1 = json.loads(json.dumps(snapshot))
    legacy_v1["schema"] = "pcno_euler2d_source_snapshot_v1"
    with pytest.raises(ValueError, match="unsupported PCNO source snapshot schema"):
        verify_source_snapshot(legacy_v1)

    corrupted = json.loads(json.dumps(snapshot))
    first_source = next(iter(corrupted["files"]))
    corrupted["files"][first_source]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="current source differs"):
        verify_source_snapshot(corrupted)


def test_source_snapshot_v2_keeps_historical_document_equality(
    tmp_path: Path,
) -> None:
    snapshot = write_source_snapshot(tmp_path / "run")
    current_files = {
        **snapshot["provenance_files"],
        **snapshot["files"],
    }
    historical_files = {
        name: current_files[name] for name in PCNO_SOURCE_SNAPSHOT_V2_FILES
    }
    encoded_files = json.dumps(
        historical_files, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    historical = {
        "schema": PCNO_SOURCE_SNAPSHOT_V2_SCHEMA,
        "files": historical_files,
        "source_set_digest": hashlib.sha256(encoded_files).hexdigest(),
    }
    assert set(historical["files"]) == set(PCNO_SOURCE_SNAPSHOT_V2_FILES)
    verify_source_snapshot(historical)

    corrupted = json.loads(json.dumps(historical))
    history_path = PCNO_SOURCE_PROVENANCE_FILES[0]
    corrupted["files"][history_path]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="current source differs"):
        verify_source_snapshot(corrupted)

    corrupted_digest = json.loads(json.dumps(historical))
    corrupted_digest["source_set_digest"] = "0" * 64
    with pytest.raises(ValueError, match="v2 source snapshot source-set digest"):
        verify_source_snapshot(corrupted_digest)


def test_source_snapshot_v3_remains_compatible(tmp_path: Path) -> None:
    snapshot = write_source_snapshot(tmp_path / "run")
    historical_files = {
        name: snapshot["files"][name] for name in PCNO_SOURCE_SNAPSHOT_V3_FILES
    }
    encoded_files = json.dumps(
        historical_files, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    historical = {
        "schema": PCNO_SOURCE_SNAPSHOT_V3_SCHEMA,
        "files": historical_files,
        "provenance_files": snapshot["provenance_files"],
        "source_set_digest": hashlib.sha256(encoded_files).hexdigest(),
        "provenance_set_digest": snapshot["provenance_set_digest"],
    }

    verify_source_snapshot(historical)


def test_source_snapshot_v4_remains_compatible(tmp_path: Path) -> None:
    snapshot = write_source_snapshot(tmp_path / "run")
    historical_files = {
        name: snapshot["files"][name] for name in PCNO_SOURCE_SNAPSHOT_V4_FILES
    }
    encoded_files = json.dumps(
        historical_files, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    historical = {
        "schema": PCNO_SOURCE_SNAPSHOT_V4_SCHEMA,
        "files": historical_files,
        "provenance_files": snapshot["provenance_files"],
        "source_set_digest": hashlib.sha256(encoded_files).hexdigest(),
        "provenance_set_digest": snapshot["provenance_set_digest"],
    }

    verify_source_snapshot(historical)


def test_source_snapshot_v5_remains_compatible(tmp_path: Path) -> None:
    snapshot = write_source_snapshot(tmp_path / "run")
    historical_files = {
        name: snapshot["files"][name] for name in PCNO_SOURCE_SNAPSHOT_V5_FILES
    }
    encoded_files = json.dumps(
        historical_files, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    historical = {
        "schema": PCNO_SOURCE_SNAPSHOT_V5_SCHEMA,
        "files": historical_files,
        "provenance_files": snapshot["provenance_files"],
        "source_set_digest": hashlib.sha256(encoded_files).hexdigest(),
        "provenance_set_digest": snapshot["provenance_set_digest"],
    }

    verify_source_snapshot(historical)


def test_source_snapshot_v6_binds_registered_extension_sources(
    tmp_path: Path,
) -> None:
    extra = "tests/time_dependent_no/test_pcno_artifacts.py"
    snapshot = write_source_snapshot(
        tmp_path / "run",
        extra_source_files=(extra,),
    )

    assert snapshot["extra_source_files"] == [extra]
    assert set(snapshot["files"]) == {*PCNO_SOURCE_SNAPSHOT_FILES, extra}
    verify_source_snapshot(snapshot)

    missing_registry = json.loads(json.dumps(snapshot))
    missing_registry["extra_source_files"] = []
    with pytest.raises(ValueError, match="registered source set"):
        verify_source_snapshot(missing_registry)

    with pytest.raises(ValueError, match="already registered"):
        write_source_snapshot(
            tmp_path / "duplicate",
            extra_source_files=(PCNO_SOURCE_SNAPSHOT_FILES[0],),
        )

    with pytest.raises(ValueError, match="relative to the repository"):
        write_source_snapshot(
            tmp_path / "absolute",
            extra_source_files=(Path(__file__).resolve(),),
        )
