from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.time_dependent_no import evaluate_pcno_bump_scaling_ladder as ladder


def _metric_row(
    epoch: int,
    *,
    train: float,
    seen: float,
    validation: float,
    rollout: float,
    h79: float,
) -> dict[str, object]:
    return {
        "epoch": epoch,
        "train": {
            "relative_l2": train,
            "completed_optimizer_steps": (epoch + 1) * 256,
            "comparable_seen": {"relative_l2": seen},
        },
        "validation": {"relative_l2": validation},
        "rollout": {
            "mean_full_horizon_relative_l2": rollout,
            "mean_endpoint_relative_l2": {"79": h79},
            "completion_rate": 1.0,
            "hard_failure_count": 0,
            "physical_admissibility_rate": 0.75,
        },
    }


def test_training_metric_scopes_keep_selected_and_terminal_separate(
    tmp_path: Path,
) -> None:
    rows = [
        _metric_row(
            4,
            train=0.2,
            seen=0.21,
            validation=0.22,
            rollout=0.3,
            h79=0.4,
        ),
        _metric_row(
            9,
            train=0.1,
            seen=0.11,
            validation=0.12,
            rollout=0.5,
            h79=0.6,
        ),
    ]
    (tmp_path / "metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    selected = ladder.checkpoint_metric_snapshot(rows[0])
    terminal = ladder.checkpoint_metric_snapshot(rows[1])
    receipt = {
        "schema": ladder.METRIC_RECEIPT_SCHEMA,
        "best_epoch": 4,
        "selected_checkpoint": selected,
        "terminal_checkpoint": terminal,
        "historical_test_population_accessed": False,
    }
    (tmp_path / "d094_metric_receipt.json").write_text(
        json.dumps(receipt), encoding="utf-8"
    )

    scopes = ladder.training_metric_scopes(
        tmp_path, {"best_epoch": 4}, require_receipt=True
    )

    assert scopes["selected_checkpoint"]["rollout_all_call_mean_relative_l2"] == 0.3
    assert scopes["terminal_checkpoint"]["rollout_all_call_mean_relative_l2"] == 0.5
    assert (
        scopes["selected_checkpoint"]["fixed_validation_one_step_relative_l2"] == 0.22
    )
    assert (
        scopes["terminal_checkpoint"]["fixed_validation_one_step_relative_l2"] == 0.12
    )


def test_training_metric_scopes_reject_receipt_drift(tmp_path: Path) -> None:
    row = _metric_row(
        4,
        train=0.2,
        seen=0.21,
        validation=0.22,
        rollout=0.3,
        h79=0.4,
    )
    (tmp_path / "metrics.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    snapshot = ladder.checkpoint_metric_snapshot(row)
    receipt = {
        "schema": ladder.METRIC_RECEIPT_SCHEMA,
        "best_epoch": 4,
        "selected_checkpoint": dict(snapshot),
        "terminal_checkpoint": dict(snapshot),
        "historical_test_population_accessed": False,
    }
    receipt["selected_checkpoint"]["rollout_h79_relative_l2"] = 9.0
    (tmp_path / "d094_metric_receipt.json").write_text(
        json.dumps(receipt), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="differs from metric history"):
        ladder.training_metric_scopes(tmp_path, {"best_epoch": 4}, require_receipt=True)


def test_verify_retained_source_snapshot_rehashes_every_file(tmp_path: Path) -> None:
    snapshot_root = tmp_path / "source_snapshot"
    source = snapshot_root / "source.py"
    provenance = snapshot_root / "decision.md"
    source.parent.mkdir(parents=True)
    source.write_text("source\n", encoding="utf-8")
    provenance.write_text("decision\n", encoding="utf-8")
    files = {
        "source.py": {
            "bytes": source.stat().st_size,
            "sha256": ladder.sha256_file(source),
        }
    }
    provenance_files = {
        "decision.md": {
            "bytes": provenance.stat().st_size,
            "sha256": ladder.sha256_file(provenance),
        }
    }
    manifest = {
        "schema": ladder.PCNO_SOURCE_SNAPSHOT_SCHEMA,
        "files": files,
        "provenance_files": provenance_files,
        "source_set_digest": ladder._mapping_digest(files),
        "provenance_set_digest": ladder._mapping_digest(provenance_files),
    }
    (snapshot_root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    observed = ladder.verify_retained_source_snapshot(
        tmp_path,
        {"source_snapshot": manifest},
        expected_source_set_digest=manifest["source_set_digest"],
    )

    assert observed == ladder.sha256_file(snapshot_root / "manifest.json")
    source.write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="retained source file changed"):
        ladder.verify_retained_source_snapshot(
            tmp_path,
            {"source_snapshot": manifest},
            expected_source_set_digest=manifest["source_set_digest"],
        )


def _fake_descriptor(run_dir: Path, **_: object) -> dict[str, object]:
    tokens = run_dir.name.split("_")
    if run_dir.name.startswith("fresh_"):
        trajectory_count = int(tokens[1][1:])
        architecture = tokens[2]
    else:
        trajectory_count = 256
        architecture = tokens[1]
    return {
        "run_dir": run_dir,
        "run": run_dir.name,
        "trajectory_count": trajectory_count,
        "architecture": architecture,
        "schedule": ladder.SCHEDULE,
    }


def _fake_gate_cells(gate_root: Path) -> dict[tuple[str, str], dict[str, object]]:
    return {
        (architecture, schedule): {
            "run_dir": gate_root / f"gate_{architecture}_{schedule}"
        }
        for architecture in ladder.ARCHITECTURES
        for schedule in ("prefix_tail", ladder.SCHEDULE)
    }


def test_discover_scaling_cells_requires_exact_matrix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ladder_root = tmp_path / "ladder"
    outputs = ladder_root / "outputs"
    outputs.mkdir(parents=True)
    (ladder_root / "ladder.completed").write_text("done\n", encoding="utf-8")
    (ladder_root / "ladder.exit").write_text("0\n", encoding="utf-8")
    for trajectory_count in ladder.FRESH_COUNTS:
        for architecture in ladder.ARCHITECTURES:
            (outputs / f"fresh_n{trajectory_count}_{architecture}").mkdir()
    gate_root = tmp_path / "gate"
    gate_root.mkdir()
    monkeypatch.setattr(ladder, "_descriptor", _fake_descriptor)
    monkeypatch.setattr(ladder, "_discover_cells", _fake_gate_cells)

    cells = ladder.discover_scaling_cells(ladder_root, gate_root)

    assert set(cells) == {
        (trajectory_count, architecture)
        for trajectory_count in ladder.ALL_COUNTS
        for architecture in ladder.ARCHITECTURES
    }

    (outputs / "fresh_n128_pcfno").rmdir()
    with pytest.raises(ValueError, match="exact ten-cell matrix"):
        ladder.discover_scaling_cells(ladder_root, gate_root)


def test_validate_nested_training_subsets_rejects_drift() -> None:
    subsets = {
        str(trajectory_count): [str(index) for index in range(trajectory_count)]
        for trajectory_count in ladder.ALL_COUNTS
    }
    cells = {
        (trajectory_count, architecture): {
            "contract": {"train_keys": list(subsets[str(trajectory_count)])}
        }
        for trajectory_count in ladder.ALL_COUNTS
        for architecture in ladder.ARCHITECTURES
    }
    ladder.validate_nested_training_subsets(
        cells, {"nested_exposure": {"subsets": subsets}}
    )
    cells[(32, "pcfno")]["contract"]["train_keys"] = list(reversed(subsets["32"]))

    with pytest.raises(ValueError, match="train subset changed"):
        ladder.validate_nested_training_subsets(
            cells, {"nested_exposure": {"subsets": subsets}}
        )


def test_validate_registered_cell_contract_is_test_closed() -> None:
    train_keys = ["train-0", "train-1"]
    manifest = {
        "schema": "split-schema",
        "partition_digest": "partition",
        "canonical_payload_sha256": "canonical",
        "source_manifest_sha256": "data",
        "nested_exposure": {"subsets": {"2": train_keys}},
    }
    contract = {
        "schema": ladder.TRAINING_CONTRACT_SCHEMA,
        "status": "completed_unreplicated_pilot",
        "engineering_smoke": False,
        "science_result_eligible": False,
        "requested_epochs": 80,
        "completed_epochs": 80,
        "optimizer_steps_per_epoch": 256,
        "requested_optimizer_steps": 20_480,
        "actual_optimizer_steps": 20_480,
        "microbatch_size": 1,
        "gradient_accumulation_steps": 1,
        "checkpoint_selection_mode": "full_horizon_error_first",
        "rollout_failure_policy": ladder.FINITE_ONLY_ROLLOUT_POLICY,
        "rollout_selection_trajectory_count": 16,
        "rollout_steps": 79,
        "fixed_evaluation_windows_per_trajectory": 4,
        "sentinel_every_epochs": 0,
        "sentinel_payload": "model_only",
        "sentinel_steps": list(ladder.SENTINEL_STEPS),
        "within_update_duplicate_pairs": 0,
        "historical_test_population_accessed": False,
        "split_schema": manifest["schema"],
        "split_partition_digest": manifest["partition_digest"],
        "split_canonical_payload_sha256": manifest["canonical_payload_sha256"],
        "source_manifest_sha256": manifest["source_manifest_sha256"],
        "trajectory_count": 2,
        "train_keys": train_keys,
    }
    descriptor = {
        "contract": contract,
        "split": {
            "train_keys": train_keys,
            "test_keys": [],
            "data_manifest_digest": "data",
        },
        "summary": {"data_manifest_digest": "data"},
    }

    ladder.validate_registered_cell_contract(descriptor, manifest, trajectory_count=2)
    descriptor["split"]["test_keys"] = ["sealed"]
    with pytest.raises(ValueError, match="sealed/test"):
        ladder.validate_registered_cell_contract(
            descriptor, manifest, trajectory_count=2
        )


def test_parser_exposes_no_historical_test_input() -> None:
    destinations = {action.dest for action in ladder.build_parser()._actions}
    assert "test" not in destinations
    assert "test_root" not in destinations
    assert {
        "ladder_root",
        "n256_gate_root",
        "data_dir",
        "split_manifest",
        "output_dir",
    } <= destinations
