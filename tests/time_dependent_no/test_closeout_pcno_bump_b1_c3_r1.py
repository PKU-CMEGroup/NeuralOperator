from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.time_dependent_no import closeout_pcno_bump_b1_c3_r1 as closeout


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _make_attempt(tmp_path: Path) -> Path:
    attempt = tmp_path / "attempt"
    results = attempt / "results"
    launch = attempt / "launch"
    results.mkdir(parents=True)
    launch.mkdir()
    (attempt / "source_commit.tar").write_bytes(b"source")

    for index, count in enumerate(closeout.TRAJECTORY_COUNTS, start=1):
        for architecture, mode in closeout.ARCHITECTURE_MODES:
            cell = results / closeout._cell_name(count, architecture)
            (cell / "sentinels").mkdir(parents=True)
            branch = {
                "mode": mode,
                "state_dict_keys_unchanged": True,
                "initial_full_state_sha256": f"full-{count}",
                "initial_nondifferential_state_sha256": f"shared-{count}",
            }
            summary = {
                "actual_optimizer_steps": closeout.OPTIMIZER_STEPS,
                "completed_epochs": 80,
                "test_trajectories": 0,
                "config_digest": f"config-{architecture}-{count}",
                "normalization_digest": f"normalization-{count}",
                "initialization_control": {"differential_branch": branch},
                "model_config": {"marker": count},
            }
            contract = {
                "status": "completed_seed0_exact_exposure_replay_cell",
                "science_result_eligible": False,
                "historical_test_population_accessed": False,
                "trajectory_count": count,
                "registered_stage": closeout.REGISTERED_STAGE,
                "initialization_seed": closeout.SEED,
                "differential_branch_mode": mode,
                "actual_optimizer_steps": closeout.OPTIMIZER_STEPS,
                "completed_epochs": 80,
                "schedule_arm": closeout.SCHEDULE_ARM,
                "sentinel_steps": [closeout.PRESENTATIONS_PER_TRAJECTORY * count],
                "sentinel_payload": "model_only",
                "selected_terminal_checkpoint_payload": "model_only_evaluation",
                "automatic_continuation_authorized": False,
                "presentation_stream_sha256_by_epoch": [f"stream-{count}"],
            }
            _write_json(cell / "summary.json", summary)
            _write_json(cell / "bump_scaling_contract.json", contract)
            _write_json(
                cell / "run_contract.json",
                {"args": {"init_checkpoint": None, "resume_checkpoint": None}},
            )
            _write_json(cell / "normalization.json", {"marker": count})
            (cell / "metrics.jsonl").write_text(
                json.dumps({"loss": 0.1}) + "\n", encoding="utf-8"
            )
            sentinel = (
                f"sentinels/step_"
                f"{closeout.PRESENTATIONS_PER_TRAJECTORY * count:09d}.pt"
            )
            for relative in (*closeout.CELL_FILES, sentinel):
                path = cell / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                if not path.exists():
                    path.write_bytes(relative.encode("utf-8"))
        (launch / f"cell_{2 * index - 1}.completed").write_text("ok")
        (launch / f"cell_{2 * index}.completed").write_text("ok")
    return attempt


def _mock_reconstruction(summary: dict, _normalization: dict) -> dict[str, str]:
    count = int(summary["model_config"]["marker"])
    return {
        "full_state_sha256": f"full-{count}",
        "parameter_state_sha256": "same-parameters",
        "nondifferential_parameter_state_sha256": "same-shared-parameters",
    }


def test_closeout_accepts_count_specific_buffers_but_requires_shared_parameters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempt = _make_attempt(tmp_path)
    monkeypatch.setattr(closeout, "sha256_file", lambda path: "source-hash" if path.name == "source_commit.tar" else "file-hash")
    monkeypatch.setattr(closeout, "SOURCE_ARCHIVE_SHA256", "source-hash")
    monkeypatch.setattr(closeout, "_reconstruct_initialization", _mock_reconstruction)

    receipt = closeout.build_matrix_receipt(attempt)

    assert receipt["status"] == "complete_after_closeout_correction"
    assert len(receipt["cells"]) == 12
    correction = receipt["closeout_correction"]
    assert correction["whole_state_hash_cardinality_across_counts"] == 6
    assert correction["parameter_only_hash_cardinality_across_counts"] == 1
    assert correction["original_attempt_outputs_modified"] is False


def test_closeout_rejects_pairwise_initialization_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempt = _make_attempt(tmp_path)
    cell = (
        attempt
        / "results"
        / closeout._cell_name(closeout.TRAJECTORY_COUNTS[0], "pcfno")
    )
    summary = json.loads((cell / "summary.json").read_text(encoding="utf-8"))
    summary["initialization_control"]["differential_branch"][
        "initial_full_state_sha256"
    ] = "unpaired"
    _write_json(cell / "summary.json", summary)
    monkeypatch.setattr(closeout, "sha256_file", lambda path: "source-hash" if path.name == "source_commit.tar" else "file-hash")
    monkeypatch.setattr(closeout, "SOURCE_ARCHIVE_SHA256", "source-hash")
    monkeypatch.setattr(closeout, "_reconstruct_initialization", _mock_reconstruction)

    with pytest.raises(ValueError, match="cannot be reconstructed|paired PCNO/PCFNO"):
        closeout.build_matrix_receipt(attempt)


def test_closeout_rejects_nonfinite_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempt = _make_attempt(tmp_path)
    cell = (
        attempt
        / "results"
        / closeout._cell_name(closeout.TRAJECTORY_COUNTS[0], "pcno")
    )
    (cell / "metrics.jsonl").write_text('{"loss": NaN}\n', encoding="utf-8")
    monkeypatch.setattr(closeout, "sha256_file", lambda path: "source-hash" if path.name == "source_commit.tar" else "file-hash")
    monkeypatch.setattr(closeout, "SOURCE_ARCHIVE_SHA256", "source-hash")
    monkeypatch.setattr(closeout, "_reconstruct_initialization", _mock_reconstruction)

    with pytest.raises(ValueError, match="nonfinite"):
        closeout.build_matrix_receipt(attempt)
