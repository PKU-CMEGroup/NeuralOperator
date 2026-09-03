from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import torch
from torch import nn

from scripts.time_dependent_no import (
    evaluate_pcno_naca0012_corrective_extension as evaluator,
)
from utility.time_dependent_no.pcno_naca0012 import NACANormalization


def _normalization() -> NACANormalization:
    return NACANormalization(
        state_mean=np.zeros(5),
        state_scale=np.ones(5),
        residual_scale=np.ones(5),
        state_rms=np.ones(5),
    )


def _packet(arm: str, seed: int) -> evaluator.ExtensionTrainingPacket:
    closed = cast(
        evaluator.parent.ClosedPacket,
        SimpleNamespace(
            final={
                "initial_model_state_sha256": f"{seed:064x}",
                "presentation_schedule_sha256": "a" * 64,
            }
        ),
    )
    return evaluator.ExtensionTrainingPacket(
        packet=closed,
        arm=arm,
        seed=seed,
        config={},
        inputs={},
        runtime={},
        summary={},
        checkpoint_record={},
    )


def test_deployment_inventory_is_exact_six_arm_online_ema_and_refiner_grid() -> None:
    packets = [
        _packet(arm, seed) for arm in evaluator.LEARNED_ARMS for seed in evaluator.SEEDS
    ]
    inventory = evaluator._deployment_inventory(packets)

    assert len(inventory) == 39
    assert len({item.key for item in inventory}) == 39
    for arm in evaluator.LEARNED_ARMS:
        observed = [item for item in inventory if item.arm == arm and item.seed == 17]
        if arm == evaluator.REFINER_ARM:
            assert {(item.deployment, item.sampler_seed) for item in observed} == {
                (deployment, sampler)
                for deployment in ("online", "ema")
                for sampler in evaluator.REFINER_SAMPLER_SEEDS
            }
        elif arm in evaluator.EMA_ARMS:
            assert [(item.deployment, item.sampler_seed) for item in observed] == [
                ("online", None),
                ("ema", None),
            ]
        else:
            assert [(item.deployment, item.sampler_seed) for item in observed] == [
                ("online", None)
            ]


@pytest.mark.parametrize("role", ["train", "prospective", "sealed", "test"])
def test_only_development_population_is_implemented(role: str) -> None:
    with pytest.raises(evaluator.parent.ProtectedPopulationError):
        evaluator._validate_population_role(role)
    assert evaluator._validate_population_role("development") == "development"


class _CountingVelocity(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, features, geometry_batch, *, fourier_tensors=None):
        del geometry_batch, fourier_tensors
        self.calls += 1
        return torch.zeros(features.shape[:2] + (5,), dtype=features.dtype)


def test_refiner_adapter_executes_four_calls_and_replays_common_noise() -> None:
    previous = torch.zeros(2, 3, 5, dtype=torch.float32)
    current = torch.ones(2, 3, 5, dtype=torch.float32)
    geometry = {"static_features": torch.zeros(2, 3, 6)}
    first_model = _CountingVelocity()
    second_model = _CountingVelocity()
    first = evaluator.RefinerTransitionAdapter(
        first_model,
        _normalization(),
        sampler_seed=101,
        device=torch.device("cpu"),
    )
    second = evaluator.RefinerTransitionAdapter(
        second_model,
        _normalization(),
        sampler_seed=101,
        device=torch.device("cpu"),
    )

    first_step = first.refine(previous, current, geometry, fourier_tensors=None)
    second_step = second.refine(previous, current, geometry, fourier_tensors=None)

    assert first_model.calls == second_model.calls == 4
    assert first_step.model_calls == second_step.model_calls == 4
    assert len(first_step.intermediate_candidates) == 5
    torch.testing.assert_close(
        first_step.next_state, second_step.next_state, rtol=0, atol=0
    )
    assert first.noise_schedule_record() == second.noise_schedule_record()
    assert first.noise_schedule_record()["physical_steps"] == 1
    assert first.noise_schedule_record()["draw_tensors"] == 4


def test_refiner_offline_response_branches_reuse_one_exact_tape() -> None:
    model = _CountingVelocity()
    adapter = evaluator.RefinerTransitionAdapter(
        model,
        _normalization(),
        sampler_seed=211,
        device=torch.device("cpu"),
    )
    features = torch.zeros(2, 3, 16, dtype=torch.float32)
    geometry = {"static_features": torch.zeros(2, 3, 6)}

    with adapter.paired_response_diagnostic():
        generated = adapter(features, geometry)
        clean_forced = adapter(features, geometry)
        one_prefix = adapter(features, geometry)

    assert not torch.equal(generated, clean_forced)
    torch.testing.assert_close(clean_forced, one_prefix, rtol=0, atol=0)
    record = adapter.noise_schedule_record()
    assert model.calls == record["inner_model_calls"] == 12
    assert record["physical_steps"] == 3
    assert record["noise_tapes_drawn"] == 2
    assert record["draw_tensors"] == 8
    assert record["paired_response_tape_reuses"] == 1


def test_packet_bindings_require_bank_only_for_paired_arms() -> None:
    common = {
        "experiment_id": evaluator.EXPERIMENT_ID,
        "arm": "PAIRED_RECOVERY",
        "seed": 17,
        "extension_contract_sha256": evaluator.EXTENSION_CONTRACT_SHA256,
        "extension_preregistration_sha256": evaluator.EXTENSION_PREREGISTRATION_SHA256,
        "inherited_r0_contract_sha256": evaluator.trainer.R0_CONTRACT_SHA256,
        "inherited_successor_contract_sha256": evaluator.trainer.SUCCESSOR_CONTRACT_SHA256,
        "dataset_manifest_payload_sha256": "1" * 64,
        "dataset_final_hash_manifest_sha256": "2" * 64,
        "successor_calibration_payload_sha256": "3" * 64,
        "successor_calibration_final_hash_manifest_sha256": "4" * 64,
        "paired_bank_final_hash_manifest_sha256": "5" * 64,
        "relabel_pilot_final_hash_manifest_sha256": "6" * 64,
        "source_set_sha256": "7" * 64,
        "train_opened": True,
        "development_opened": True,
        "prospective_opened": False,
        "sealed_opened": False,
        "offline_training_label_solver_calls": False,
        "online_solver_calls": False,
        "online_defect_trigger": False,
    }
    evaluator._validate_packet_binding(
        common,
        label="test",
        arm="PAIRED_RECOVERY",
        seed=17,
        dataset_sha256="2" * 64,
        dataset_payload_sha256="1" * 64,
        calibration_sha256="4" * 64,
        calibration_payload_sha256="3" * 64,
        paired_bank_sha256="5" * 64,
        pilot_sha256="6" * 64,
    )
    contaminated = dict(common)
    contaminated["online_defect_trigger"] = True
    with pytest.raises(ValueError, match="online_defect_trigger"):
        evaluator._validate_packet_binding(
            contaminated,
            label="test",
            arm="PAIRED_RECOVERY",
            seed=17,
            dataset_sha256="2" * 64,
            dataset_payload_sha256="1" * 64,
            calibration_sha256="4" * 64,
            calibration_payload_sha256="3" * 64,
            paired_bank_sha256="5" * 64,
            pilot_sha256="6" * 64,
        )


def _write_recursive_packet(root: Path) -> Path:
    root.mkdir()
    artifact = root / "artifact.bin"
    artifact.write_bytes(b"bound")
    final = {
        "schema": "test.recursive.v1",
        "files": {
            "artifact.bin": {
                "relative_path": "artifact.bin",
                "bytes": artifact.stat().st_size,
                "sha256": sha256(artifact.read_bytes()).hexdigest(),
            }
        },
        "self_hash_excluded": True,
        "prospective_opened": False,
        "sealed_opened": False,
    }
    final_path = root / "final_hash_manifest.json"
    final_path.write_text(json.dumps(final), encoding="utf-8")
    return final_path


def test_recursive_input_packet_rejects_unmanifested_artifact(tmp_path: Path) -> None:
    final_path = _write_recursive_packet(tmp_path / "packet")
    packet = evaluator._verify_recursive_input_packet(
        final_path, schema="test.recursive.v1", label="test packet"
    )
    assert packet.final_sha256 == sha256(final_path.read_bytes()).hexdigest()

    (final_path.parent / "extra.bin").write_bytes(b"not bound")
    with pytest.raises(ValueError, match="unmanifested"):
        evaluator._verify_recursive_input_packet(
            final_path, schema="test.recursive.v1", label="test packet"
        )


def _rollout_row(
    arm: str,
    deployment: str,
    sampler_seed: int | None,
    horizon: int,
    error: float,
    state_stage: str = "raw",
) -> dict[str, object]:
    return {
        "arm": arm,
        "seed": 17,
        "deployment": deployment,
        "sampler_seed": sampler_seed,
        "state_stage": state_stage,
        "anchor": 1234,
        "horizon": horizon,
        "normalized_state_error": error,
        "path_distance": error,
        "projected_path_discrepancy": error,
        "transverse_error": error,
        "tangent_direction_valid": True,
        "graph_dirichlet_error_energy": error,
        "finite": True,
    }


def test_summaries_keep_sampler_replicates_and_use_matched_clean(monkeypatch) -> None:
    monkeypatch.setattr(evaluator, "FULL_TRACE", (1, 2))
    monkeypatch.setattr(evaluator, "LATE_WINDOW", (1, 2))
    monkeypatch.setattr(evaluator.parent, "FULL_TRACE", (1, 2))
    rows = []
    for deployment, error in (("online", 2.0), ("ema", 1.0)):
        rows.extend(
            _rollout_row("CLEAN_EMA", deployment, None, horizon, error)
            for horizon in (1, 2)
        )
    for deployment, error in (("online", 1.0), ("ema", 0.5)):
        for sampler in evaluator.REFINER_SAMPLER_SEEDS:
            rows.extend(
                _rollout_row(
                    evaluator.REFINER_ARM,
                    deployment,
                    sampler,
                    horizon,
                    error + sampler * 1.0e-6,
                )
                for horizon in (1, 2)
            )

    method_rows, pairwise, result = evaluator._summaries(rows, [1234])

    assert len(method_rows) == 8
    assert len(result) == 8
    assert len(pairwise) == 14
    refiner = [row for row in pairwise if row["arm"] == evaluator.REFINER_ARM]
    assert {row["sampler_seed"] for row in refiner} == set(
        evaluator.REFINER_SAMPLER_SEEDS
    )
    assert {
        (row["deployment"], row["matched_clean_deployment"]) for row in refiner
    } == {("online", "online"), ("ema", "ema")}
    clean_ema = [row for row in pairwise if row["arm"] == "CLEAN_EMA"]
    assert {row["matched_clean_deployment"] for row in clean_ema} == {"online"}
    assert {row["ratio_to_matched_clean"] for row in clean_ema} == {0.5}


def test_inherited_comparisons_use_parent_clean_and_exclude_pre_correction(
    monkeypatch,
) -> None:
    monkeypatch.setattr(evaluator, "FULL_TRACE", (1, 2))
    monkeypatch.setattr(evaluator, "LATE_WINDOW", (1, 2))
    monkeypatch.setattr(evaluator.parent, "FULL_TRACE", (1, 2))
    rows = []
    rows.extend(_rollout_row("CLEAN", "raw", None, horizon, 2.0) for horizon in (1, 2))
    rows.extend(
        _rollout_row("DETACHED_PUSHFORWARD", "raw", None, horizon, 1.0)
        for horizon in (1, 2)
    )
    for stage, error in (("raw_pre_correction", 3.0), ("corrected", 1.0)):
        rows.extend(
            _rollout_row(
                "PATH_PROJECTION",
                "path_projection",
                None,
                horizon,
                error,
                stage,
            )
            for horizon in (1, 2)
        )

    method_rows, pairwise, _ = evaluator._summaries(rows, [1234])

    assert len(method_rows) == 4
    assert len(pairwise) == 4
    assert {row["arm"] for row in pairwise} == {
        "DETACHED_PUSHFORWARD",
        "PATH_PROJECTION",
    }
    assert {row["matched_clean_arm"] for row in pairwise} == {"CLEAN"}
    assert {row["matched_clean_deployment"] for row in pairwise} == {"raw"}
    assert not any(row["state_stage"] == "raw_pre_correction" for row in pairwise)


def test_inherited_loader_uses_complete_parent_packet_validator(monkeypatch) -> None:
    calls = []

    def verify(root, **kwargs):
        calls.append((root, kwargs))
        arm, seed_text = root.name.rsplit("_", 1)
        packet = cast(
            evaluator.parent.ClosedPacket,
            SimpleNamespace(final_file_sha256=f"{int(seed_text):064x}"),
        )
        return evaluator.parent.TrainingPacket(
            packet=packet,
            arm=arm,
            seed=int(seed_text),
            config={},
            inputs={},
            runtime={},
            summary={
                "source_set_sha256": "f" * 64,
                "initial_model_state_sha256": f"{int(seed_text):064x}",
                "presentation_schedule_sha256": f"{int(seed_text) + 1:064x}",
            },
            checkpoint_record={"sha256": "e" * 64},
        )

    monkeypatch.setattr(evaluator.parent, "_verify_training_packet", verify)
    clean = [Path(f"CLEAN_{seed}") for seed in evaluator.SEEDS]
    detached = [Path(f"DETACHED_PUSHFORWARD_{seed}") for seed in evaluator.SEEDS]

    packets = evaluator._load_inherited_baseline_packets(
        clean,
        detached,
        successor={},
        successor_sha256="1" * 64,
        dataset_sha256="2" * 64,
        dataset_payload_sha256="3" * 64,
        calibration_sha256="4" * 64,
        calibration_payload_sha256="5" * 64,
        baseline={},
        geometry=cast(evaluator.VerifiedNACAGeometry, object()),
    )

    assert len(calls) == 6
    assert {(item.arm, item.seed) for item in packets} == {
        (arm, seed)
        for arm in ("CLEAN", "DETACHED_PUSHFORWARD")
        for seed in evaluator.SEEDS
    }
    assert all("successor" in kwargs and "geometry" in kwargs for _, kwargs in calls)


def test_common_evaluator_cardinalities_include_inherited_reference_and_arms() -> None:
    assert evaluator.EVALUATED_STATE_VARIANT_COUNT == 54
    assert evaluator.UNIQUE_SNAPSHOT_VARIANT_COUNT == 51
    assert evaluator.EXPECTED_CARDINALITIES == {
        "offline_diagnostics.csv": 45 * 8,
        "rollout_metrics.csv": 54 * 8 * 208,
        "rollout_structure.csv": 54 * 8 * 208,
        "refiner_intermediate_diagnostics.csv": 18 * 8 * len(evaluator.HORIZONS) * 5,
        "method_summary.csv": 54 * 8,
        "pairwise_summary.csv": 42 * 8 * len(evaluator.parent.PRIMARY_METRICS),
        "cost_metrics.csv": 48,
        "rollout_snapshots.npz": 51 * 8 * len(evaluator.HORIZONS)
        + 8 * len(evaluator.HORIZONS),
    }


def test_nonfinite_rollouts_close_exact_snapshot_inventory_with_nan_sentinels() -> None:
    packets = [
        _packet(arm, seed) for arm in evaluator.LEARNED_ARMS for seed in evaluator.SEEDS
    ]
    variants = [
        (item.arm, item.seed, item.deployment, "raw", item.sampler_seed)
        for item in evaluator._deployment_inventory(packets)
    ]
    for seed in evaluator.SEEDS:
        variants.extend(
            (
                ("CLEAN", seed, "raw", "raw", None),
                ("CLEAN", seed, "identity", "corrected", None),
                ("DETACHED_PUSHFORWARD", seed, "raw", "raw", None),
                (
                    "PATH_PROJECTION",
                    seed,
                    "path_projection",
                    "raw_pre_correction",
                    None,
                ),
                (
                    "PATH_PROJECTION",
                    seed,
                    "path_projection",
                    "corrected",
                    None,
                ),
            )
        )
    assert len(variants) == evaluator.EVALUATED_STATE_VARIANT_COUNT
    rows = [
        {
            "arm": arm,
            "seed": seed,
            "deployment": deployment,
            "state_stage": state_stage,
            "sampler_seed": sampler_seed,
            "anchor": anchor,
            "horizon": horizon,
            "finite": False,
        }
        for arm, seed, deployment, state_stage, sampler_seed in variants
        for anchor in evaluator.DEVELOPMENT_ANCHORS
        for horizon in evaluator.HORIZONS
    ]
    indices = np.arange(
        max(evaluator.DEVELOPMENT_ANCHORS) + max(evaluator.HORIZONS) + 1,
        dtype=np.int64,
    )
    states = np.arange(indices.size * 2 * 5, dtype=np.float64).reshape(
        indices.size, 2, 5
    )

    closed = evaluator._close_registered_snapshots(
        {},
        rows,
        states=states,
        indices=indices,
        anchors=evaluator.DEVELOPMENT_ANCHORS,
    )

    assert len(closed) == evaluator.EXPECTED_CARDINALITIES["rollout_snapshots.npz"]
    assert len(closed) == 2080
    prediction_members = {
        name: value
        for name, value in closed.items()
        if not name.startswith("reference_")
    }
    assert len(prediction_members) == 51 * 8 * len(evaluator.HORIZONS)
    assert all(value.shape == (2, 5) for value in prediction_members.values())
    assert all(value.dtype == np.float32 for value in prediction_members.values())
    assert all(np.all(np.isnan(value)) for value in prediction_members.values())
    reference_members = {
        name: value for name, value in closed.items() if name.startswith("reference_")
    }
    assert len(reference_members) == 8 * len(evaluator.HORIZONS)
    assert all(np.isrealobj(value) for value in reference_members.values())
    assert all(np.all(np.isfinite(value)) for value in reference_members.values())

    rows[0] = {**rows[0], "finite": True}
    with pytest.raises(
        ValueError, match="finite registered prediction snapshot is missing"
    ):
        evaluator._close_registered_snapshots(
            {},
            rows,
            states=states,
            indices=indices,
            anchors=evaluator.DEVELOPMENT_ANCHORS,
        )


def test_cli_reports_chained_underlying_error(monkeypatch, capsys) -> None:
    class _Parser:
        @staticmethod
        def parse_args(argv):
            del argv
            return SimpleNamespace()

    def fail(_arguments):
        try:
            raise ValueError("inner cardinality detail")
        except ValueError as error:
            raise evaluator.parent.EvaluationOutputError(
                "outer packet failure"
            ) from error

    monkeypatch.setattr(evaluator, "_parser", lambda: _Parser())
    monkeypatch.setattr(evaluator, "evaluate", fail)

    assert evaluator.main([]) == 1
    payload = json.loads(capsys.readouterr().err)
    assert payload == {
        "status": "error",
        "error": "outer packet failure",
        "error_type": "EvaluationOutputError",
        "exception_chain": [
            {"type": "EvaluationOutputError", "message": "outer packet failure"},
            {"type": "ValueError", "message": "inner cardinality detail"},
        ],
        "underlying_error": {
            "type": "ValueError",
            "message": "inner cardinality detail",
        },
    }


def test_live_extension_contract_and_source_closure_are_bound() -> None:
    docs = evaluator.REPO_ROOT / "docs" / "time_dependent_no"
    contract = evaluator.load_extension_contract(
        docs / "B3B4_NACA_CORRECTIVE_EXTENSION_CONTRACT.json",
        docs / "B3B4_NACA_CORRECTIVE_EXTENSION_PREREGISTRATION.md",
    )
    evaluator.trainer._validate_training_contract(contract.payload())
    snapshot = evaluator._source_snapshot()
    evaluator._reverify_source(snapshot)
    assert snapshot["source_set_sha256"] == evaluator.SOURCE_SET_SHA256
