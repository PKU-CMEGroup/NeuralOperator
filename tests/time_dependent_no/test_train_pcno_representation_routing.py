from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

import scripts.time_dependent_no.train_pcno_representation_routing as routing
from scripts.time_dependent_no.fit_pcno_shock_representation import sha256_file
from scripts.time_dependent_no.train_pcno_representation_routing import (
    CONFIRM_SEEDS,
    SCHEMA,
    registered_config,
    run_experiment,
)
from utility.time_dependent_no.pcno_structured_representation import (
    REPRESENTATIONS,
)


def test_registered_stages_freeze_full_pcno_budget_and_seeds() -> None:
    smoke = registered_config("native", 9, "smoke", device_type="cpu")
    screen = registered_config("native", 1701, "screen", device_type="cuda")
    confirmations = [
        registered_config(
            "dct_coupled_half_smooth",
            seed,
            "confirm",
            device_type="cuda",
        )
        for seed in CONFIRM_SEEDS
    ]

    assert smoke["steps"] == 2
    assert smoke["resolution"] == [16, 8]
    assert smoke["science_result_eligible"] is False
    assert smoke["claim_eligible"] is False
    assert screen["steps"] == 5_000
    assert screen["resolution"] == [64, 32]
    assert screen["width"] == 128
    assert screen["block_count"] == 4
    assert screen["k_max"] == 8
    assert screen["selection_rule"] == "final_registered_update"
    assert screen["science_result_eligible"] is True
    assert screen["claim_eligible"] is False
    assert {config["steps"] for config in confirmations} == {20_000}
    assert {config["seed"] for config in confirmations} == set(CONFIRM_SEEDS)
    assert all(config["full_pcno_only"] for config in confirmations)
    assert all(config["claim_eligible"] for config in confirmations)

    with pytest.raises(ValueError, match="frozen to seed 1701"):
        registered_config("native", 1702, "screen", device_type="cuda")
    with pytest.raises(ValueError, match="confirmation seed"):
        registered_config("native", 1704, "confirm", device_type="cuda")


def test_three_arm_cpu_smoke_is_paired_and_hash_complete(tmp_path: Path) -> None:
    contracts = {}
    summaries = {}
    expected_files = {
        "checkpoint.pt",
        "history.json",
        "one_step_metrics.json",
        "one_step_predictions.npz",
        "recurrent_arrays.npz",
        "recurrent_metrics.json",
        "run_contract.json",
        "run_state.json",
        "summary.json",
    }
    for representation in REPRESENTATIONS:
        output = tmp_path / representation
        summary = run_experiment(
            output,
            representation=representation,
            seed=1701,
            stage="smoke",
            device_name="cpu",
        )
        contract = json.loads(
            (output / "run_contract.json").read_text(encoding="utf-8")
        )
        manifest = json.loads(
            (output / "manifest.json").read_text(encoding="utf-8")
        )
        recurrent_rows = json.loads(
            (output / "recurrent_metrics.json").read_text(encoding="utf-8")
        )
        contracts[representation] = contract
        summaries[representation] = summary

        assert summary["schema"] == SCHEMA
        assert summary["science_result"] is False
        assert summary["claim_eligible"] is False
        assert summary["completed_steps"] == 2
        assert summary["recurrent_row_count"] == 48
        assert np.isfinite(summary["primary_value"])
        assert np.isfinite(summary["smooth_control_value"])
        assert all(
            value
            for key, value in summary["closures"].items()
            if key.endswith("_pass")
        )
        assert len(recurrent_rows) == 4 * 2 * 3 * 2
        assert {
            (row["family"], row["phase"], row["call"], row["path"])
            for row in recurrent_rows
        } == {
            (family, phase, call, path)
            for family in ("step", "pulse", "smooth_tanh", "smooth_sine")
            for phase in (0.0, 0.875)
            for call in range(3)
            for path in ("teacher", "recurrent")
        }
        selected = [
            row["metrics"]["relative_increment_l2"]
            for row in recurrent_rows
            if row["family"] in {"step", "pulse"}
            and row["phase"] == 0.875
            and row["call"] == 2
            and row["path"] == "recurrent"
        ]
        assert summary["primary_value"] == pytest.approx(float(np.mean(selected)))
        assert set(manifest["output_hashes"]) == expected_files
        assert manifest["output_count"] == len(expected_files)
        for relative, digest in manifest["output_hashes"].items():
            assert sha256_file(output / relative) == digest
        with np.load(output / "recurrent_arrays.npz") as arrays:
            assert "case__step__p0p875__call2__recurrent_next" in arrays.files
            assert "case__pulse__p0p875__call2__propagated_contribution" in arrays.files
        checkpoint = torch.load(
            output / "checkpoint.pt", map_location="cpu", weights_only=False
        )
        assert checkpoint["completed_step"] == 2
        assert checkpoint["config_digest"] == contract["config_digest"]

        if representation == "native":
            state_before = (output / "run_state.json").read_bytes()
            manifest_before = (output / "manifest.json").read_bytes()
            with pytest.raises(FileExistsError, match="refusing nonempty"):
                run_experiment(
                    output,
                    representation=representation,
                    seed=1701,
                    stage="smoke",
                    device_name="cpu",
                )
            with pytest.raises(ValueError, match="completed output"):
                run_experiment(
                    output,
                    representation=representation,
                    seed=1701,
                    stage="smoke",
                    device_name="cpu",
                    resume=True,
                )
            assert (output / "run_state.json").read_bytes() == state_before
            assert (output / "manifest.json").read_bytes() == manifest_before

    assert len(
        {
            contract["backbone_initialization_sha256"]
            for contract in contracts.values()
        }
    ) == 1
    assert len(
        {contract["population"]["sha256"] for contract in contracts.values()}
    ) == 1
    assert set(summaries) == set(REPRESENTATIONS)


def test_interrupted_smoke_resumes_to_the_uninterrupted_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_config = routing.registered_config
    original_write_checkpoint = routing.write_checkpoint

    def four_step_config(
        representation: str,
        seed: int,
        stage: str,
        *,
        device_type: str,
    ) -> dict[str, object]:
        config = original_config(
            representation,
            seed,
            stage,
            device_type=device_type,
        )
        config["steps"] = 4
        config["evaluation_interval"] = 1
        config["checkpoint_interval"] = 1
        return config

    monkeypatch.setattr(routing, "registered_config", four_step_config)
    uninterrupted = routing.run_experiment(
        tmp_path / "uninterrupted",
        representation="dct_coupled_half_smooth",
        seed=1701,
        stage="smoke",
        device_name="cpu",
    )

    def interrupt_after_step_two(path: Path, payload: dict[str, object]) -> None:
        original_write_checkpoint(path, payload)
        if path.name == "checkpoint_latest.pt" and payload["completed_step"] == 2:
            raise RuntimeError("simulated interruption")

    interrupted_output = tmp_path / "interrupted"
    monkeypatch.setattr(routing, "write_checkpoint", interrupt_after_step_two)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        routing.run_experiment(
            interrupted_output,
            representation="dct_coupled_half_smooth",
            seed=1701,
            stage="smoke",
            device_name="cpu",
        )
    assert (interrupted_output / "checkpoint_latest.pt").is_file()
    assert not (interrupted_output / "manifest.json").exists()

    monkeypatch.setattr(routing, "write_checkpoint", original_write_checkpoint)
    resumed = routing.run_experiment(
        interrupted_output,
        representation="dct_coupled_half_smooth",
        seed=1701,
        stage="smoke",
        device_name="cpu",
        resume=True,
    )

    assert resumed["final_model_state_sha256"] == uninterrupted[
        "final_model_state_sha256"
    ]
    assert resumed["primary_value"] == pytest.approx(uninterrupted["primary_value"])
    assert json.loads(
        (interrupted_output / "history.json").read_text(encoding="utf-8")
    ) == json.loads(
        (tmp_path / "uninterrupted" / "history.json").read_text(encoding="utf-8")
    )
