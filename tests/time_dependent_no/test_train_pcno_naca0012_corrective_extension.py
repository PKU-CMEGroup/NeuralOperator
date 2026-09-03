from __future__ import annotations

import io
import json
import shutil
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest
import torch
from torch import nn

from scripts.time_dependent_no import (
    train_pcno_naca0012_corrective_extension as trainer,
)
from utility.time_dependent_no.pcno_naca0012 import (
    NACANormalization,
    _build_synthetic_bdf2_dataset,
)
from utility.time_dependent_no.pcno_naca0012_corrective_extension import (
    ExponentialMovingAverage,
    FourStepVPredictionScheduler,
    load_extension_contract,
)


def _normalization() -> NACANormalization:
    return NACANormalization(
        state_mean=np.zeros(5, dtype=np.float64),
        state_scale=np.ones(5, dtype=np.float64),
        residual_scale=np.ones(5, dtype=np.float64),
        state_rms=np.ones(5, dtype=np.float64),
    )


def _dataset():
    frames = np.arange(955, 961, dtype=np.int64)
    states = np.arange(6, dtype=np.float64)[:, None, None] * np.ones(
        (1, 1, 5), dtype=np.float64
    )
    return _build_synthetic_bdf2_dataset(
        states,
        frames.tolist(),
        [956, 957, 958, 959],
        _normalization(),
    )


def _batch(dataset):
    items = [dataset[index] for index in range(len(dataset))]
    return (
        torch.stack([item[0] for item in items]),
        torch.stack([item[1] for item in items]),
        torch.stack([item[2] for item in items]),
        torch.tensor([item[3] for item in items], dtype=torch.int64),
    )


def _geometry(batch_size: int = 4) -> dict[str, torch.Tensor]:
    return {"static_features": torch.zeros(batch_size, 1, 6)}


class _RecordingBaseline(nn.Module):
    def __init__(self, value: float = 0.1) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(value))
        self.features: list[torch.Tensor] = []

    def forward(self, features, geometry_batch, *, fourier_tensors=None):
        del geometry_batch, fourier_tensors
        self.features.append(features.detach().clone())
        return self.weight.expand(features.shape[0], features.shape[1], 5)

    def model_config(self):
        return {"model": "test_baseline"}


class _ZeroRefiner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.0))

    def forward(self, features, geometry_batch, *, fourier_tensors=None):
        del geometry_batch, fourier_tensors
        return self.weight.expand(features.shape[0], features.shape[1], 5)

    def model_config(self):
        return {"model": "test_refiner"}


def _ledger() -> trainer._CallLedger:
    return trainer._CallLedger()


def test_source_closure_binds_relabel_validator() -> None:
    relative = trainer.RELABEL_SOURCE
    payload = (trainer.REPO_ROOT / relative).read_bytes()
    assert trainer.SOURCE_RECORDS_AT_IMPORT[relative] == {
        "bytes": len(payload),
        "sha256": sha256(payload).hexdigest(),
    }
    trainer._verify_live_sources()


@pytest.fixture
def workspace_tmp_path():
    """Avoid the managed Windows runner's inaccessible system temp root."""

    path = Path(__file__).resolve().parent / f"extension_test_runtime_{uuid4().hex}"
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path)


def test_trainer_accepts_only_the_live_frozen_extension_contract() -> None:
    root = trainer.REPO_ROOT / "docs" / "time_dependent_no"
    contract = load_extension_contract(
        root / "B3B4_NACA_CORRECTIVE_EXTENSION_CONTRACT.json",
        root / "B3B4_NACA_CORRECTIVE_EXTENSION_PREREGISTRATION.md",
    )
    trainer._validate_training_contract(contract.payload())
    assert tuple(contract.payload()["arms"]) == trainer.ARMS


def test_mixed_boundary_depths_use_one_request_and_sample_weighted_terminal_loss() -> (
    None
):
    dataset = _dataset()
    previous, current, target, centers = _batch(dataset)
    model = _RecordingBaseline(0.1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    ledger = _ledger()

    diagnostics = trainer._step_extension_batch(
        arm="CURRICULUM_EMA_PUSHFORWARD_K13",
        model=model,
        optimizer=optimizer,
        ema=ExponentialMovingAverage(model, decay=0.995),
        train_dataset=dataset,
        previous=previous,
        current=current,
        clean_target=target,
        centers=centers,
        geometry_batch=_geometry(),
        fourier_tensors=(),
        normalization=_normalization(),
        microbatch_size=2,
        epoch=10,
        optimizer_step=1,
        depth_generator=torch.Generator().manual_seed(4),
        intervention_generator=torch.Generator().manual_seed(5),
        scheduler=None,
        paired_bank=None,
        ledger=ledger,
        requested_depth_override=3,
    )

    assert diagnostics.requested_depth == 3
    assert diagnostics.realized_depths == (0, 1, 2, 3)
    # Mean of (d+1)^2 * (0.1-1)^2 for d=0,1,2,3.
    assert diagnostics.loss == pytest.approx(6.075, abs=1.0e-6)
    # Prefixes are stopped-gradient: only the terminal prediction differentiates.
    assert diagnostics.gradient_norm_before_clip == pytest.approx(4.5, abs=1.0e-5)
    assert float(model.weight.grad) == pytest.approx(-1.0, abs=1.0e-5)
    assert ledger.counts["training_model_calls"] == 4
    assert ledger.counts["training_model_sample_calls"] == 4
    assert ledger.counts["prefix_model_calls"] == 6
    assert ledger.counts["prefix_model_sample_calls"] == 6


def test_ema_update_occurs_after_optimizer_step() -> None:
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    ema = ExponentialMovingAverage(model, decay=0.995)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    model.weight.grad = torch.tensor([[-1.0]])

    norm = trainer._optimizer_step_with_ema(model=model, optimizer=optimizer, ema=ema)

    assert norm == pytest.approx(1.0)
    torch.testing.assert_close(model.weight, torch.tensor([[1.0]]))
    torch.testing.assert_close(ema.model.weight, torch.tensor([[0.005]]))
    assert int(ema.num_updates.item()) == 1


def _in_memory_bank(num_nodes: int = 1) -> trainer.VerifiedPairedBank:
    shape = (2, 238, num_nodes, 5)
    previous = np.empty(shape, dtype=np.float32)
    current = np.empty(shape, dtype=np.float32)
    future = np.empty(shape, dtype=np.float32)
    previous[0].fill(9.0)
    current[0].fill(10.0)
    future[0].fill(13.0)
    previous[1].fill(19.0)
    current[1].fill(20.0)
    future[1].fill(24.0)
    return trainer.VerifiedPairedBank(
        centers=np.arange(956, 1194, dtype=np.int64),
        signs=np.array([-1, 1], dtype=np.int8),
        coordinates=np.zeros((num_nodes, 2), dtype=np.float32),
        normalized_directions=np.zeros((238, 2, num_nodes, 5), dtype=np.float32),
        direction_draw_indices=np.arange(238, dtype=np.int64),
        redraw_counts=np.zeros(238, dtype=np.int32),
        displaced_previous=previous,
        displaced_current=current,
        solver_future=future,
        final_manifest_sha256="a" * 64,
        pilot_final_manifest_sha256="b" * 64,
        file_records={},
        root=trainer.REPO_ROOT,
        pilot_final_manifest_path=trainer.REPO_ROOT / "not_used_by_in_memory_test",
    )


def _paired_step(arm: str, epoch: int):
    dataset = _dataset()
    previous, current, target, centers = _batch(dataset)
    model = _RecordingBaseline(0.0)
    result = trainer._step_extension_batch(
        arm=arm,
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.0),
        ema=None,
        train_dataset=dataset,
        previous=previous,
        current=current,
        clean_target=target,
        centers=centers,
        geometry_batch=_geometry(),
        fourier_tensors=(),
        normalization=_normalization(),
        microbatch_size=2,
        epoch=epoch,
        optimizer_step=1,
        depth_generator=torch.Generator().manual_seed(2),
        intervention_generator=torch.Generator().manual_seed(3),
        scheduler=None,
        paired_bank=_in_memory_bank(),
        ledger=_ledger(),
    )
    return result, model


def test_paired_arms_share_inputs_sign_schedule_and_change_only_target() -> None:
    recovery, recovery_model = _paired_step("PAIRED_RECOVERY", epoch=1)
    relabel, relabel_model = _paired_step("DYNAMICS_RELABEL", epoch=1)
    assert recovery.paired_sign == relabel.paired_sign == -1
    assert recovery.loss == pytest.approx(22.25)
    assert relabel.loss == pytest.approx(5.0)
    recovery_displaced = torch.cat(recovery_model.features[1::2])
    relabel_displaced = torch.cat(relabel_model.features[1::2])
    torch.testing.assert_close(recovery_displaced, relabel_displaced)
    torch.testing.assert_close(
        recovery_displaced[..., 11:16], torch.full((4, 1, 5), 10.0)
    )

    plus, plus_model = _paired_step("DYNAMICS_RELABEL", epoch=2)
    assert plus.paired_sign == 1
    plus_displaced = torch.cat(plus_model.features[1::2])
    torch.testing.assert_close(plus_displaced[..., 11:16], torch.full((4, 1, 5), 20.0))


def test_refiner_fixed_seed_development_replays_four_calls() -> None:
    dataset = _dataset()
    model = _ZeroRefiner()
    scheduler = FourStepVPredictionScheduler()
    first_ledger = _ledger()
    second_ledger = _ledger()
    arguments = {
        "model": model,
        "dataset": dataset,
        "geometry_batch": _geometry(),
        "fourier_tensors": (),
        "normalization": _normalization(),
        "scheduler": scheduler,
        "device": torch.device("cpu"),
        "microbatch_size": 2,
        "counter": "development_ema_model_calls",
        "epoch": 5,
    }

    first_score, first_tape = trainer._refiner_development_score(
        ledger=first_ledger, **arguments
    )
    second_score, second_tape = trainer._refiner_development_score(
        ledger=second_ledger, **arguments
    )

    assert first_score == second_score
    assert first_tape == second_tape
    assert first_ledger.hexdigest() == second_ledger.hexdigest()
    assert first_ledger.counts["development_ema_model_calls"] == 8
    assert first_ledger.counts["development_ema_model_sample_calls"] == 16


def test_refiner_checkpoint_roundtrip_restores_online_ema_optimizer_and_rng() -> None:
    model = _ZeroRefiner()
    ema = ExponentialMovingAverage(model, decay=0.995)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=100, eta_min=1.0e-5
    )
    model.weight.grad = torch.tensor(-1.0)
    trainer._optimizer_step_with_ema(model=model, optimizer=optimizer, ema=ema)
    scheduler.step()
    intervention = torch.Generator().manual_seed(8)
    depth = torch.Generator().manual_seed(9)
    torch.rand(3, generator=intervention)
    torch.rand(2, generator=depth)
    payload = trainer._checkpoint_payload(
        record_kind="last_checkpoint",
        arm="PCNO_PDEREFINER_K3_VPRED",
        seed=17,
        epoch=1,
        model=model,
        ema=ema,
        optimizer=optimizer,
        learning_rate_scheduler=scheduler,
        intervention_generator=intervention,
        depth_generator=depth,
        bindings={"experiment_id": trainer.EXPERIMENT_ID},
        best_epoch=0,
        best_score=float("inf"),
        model_only=False,
    )
    expected_model = trainer._model_state_sha256(model)
    expected_ema = trainer._model_state_sha256(ema.model)
    expected_intervention = torch.rand(4, generator=intervention)
    expected_depth = torch.rand(4, generator=depth)

    with torch.no_grad():
        model.weight.fill_(99.0)
        ema.model.weight.fill_(88.0)
    intervention.manual_seed(1)
    depth.manual_seed(1)
    trainer._restore_training_checkpoint(
        payload,
        arm="PCNO_PDEREFINER_K3_VPRED",
        seed=17,
        model=model,
        ema=ema,
        optimizer=optimizer,
        learning_rate_scheduler=scheduler,
        intervention_generator=intervention,
        depth_generator=depth,
    )

    assert trainer._model_state_sha256(model) == expected_model
    assert trainer._model_state_sha256(ema.model) == expected_ema
    torch.testing.assert_close(
        torch.rand(4, generator=intervention), expected_intervention
    )
    torch.testing.assert_close(torch.rand(4, generator=depth), expected_depth)
    assert int(ema.num_updates.item()) == 1


def _write_json(path, value) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def test_legacy_minimal_paired_bank_schema_is_rejected(workspace_tmp_path) -> None:
    bank_root = workspace_tmp_path / "bank"
    bank_root.mkdir()
    pilot_root = workspace_tmp_path / "pilot"
    pilot_root.mkdir()
    pilot_receipt = trainer._self_hashed(
        {
            "schema": trainer.RELABEL_PILOT_RECEIPT_SCHEMA,
            "experiment_id": trainer.EXPERIMENT_ID,
            "status": "pilot_succeeded",
            "scientifically_usable": True,
            "paired_query_bank_unlocked": True,
            "error": None,
            "registered_case_count": 11,
            "attempted_solver_call_count": 11,
            "completed_case_receipt_count": 11,
            "case_receipts": [
                {"status": "validation_succeeded", "scientifically_usable": True}
                for _ in range(11)
            ],
            "gates": {
                "all_registered_cases": True,
                "all_case_receipts": True,
                "realized_displacement_scale": True,
                "zero_control": True,
                "repeatability": True,
                "auxiliary_invariance": True,
                "response_separation": True,
                "authority_immutable": True,
            },
            "online_solver_calls": False,
            "prospective_opened": False,
            "sealed_opened": False,
        }
    )
    pilot_receipt_path = pilot_root / trainer.RELABEL_PILOT_RECEIPT_FILE
    _write_json(pilot_receipt_path, pilot_receipt)
    pilot = pilot_root / "final_hash_manifest.json"
    _write_json(
        pilot,
        {
            "schema": trainer.RELABEL_PILOT_FINAL_SCHEMA,
            "experiment_id": trainer.EXPERIMENT_ID,
            "files": {
                trainer.RELABEL_PILOT_RECEIPT_FILE: {
                    "relative_path": trainer.RELABEL_PILOT_RECEIPT_FILE,
                    "bytes": pilot_receipt_path.stat().st_size,
                    "sha256": sha256(pilot_receipt_path.read_bytes()).hexdigest(),
                }
            },
            "self_hash_excluded": True,
            "prospective_opened": False,
            "sealed_opened": False,
        },
    )
    pilot_sha = sha256(pilot.read_bytes()).hexdigest()
    arrays = {
        "centers": np.arange(956, 1194, dtype=np.int64),
        "signs": np.array([-1, 1], dtype=np.int8),
        "displaced_previous": np.zeros((2, 238, 1, 5), dtype=np.float32),
        "displaced_current": np.ones((2, 238, 1, 5), dtype=np.float32),
        "solver_future": np.full((2, 238, 1, 5), 2.0, dtype=np.float32),
    }
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    arrays_path = bank_root / trainer.PAIRED_BANK_ARRAY_FILE
    arrays_path.write_bytes(buffer.getvalue())
    arrays_file_record = {
        "relative_path": trainer.PAIRED_BANK_ARRAY_FILE,
        "bytes": arrays_path.stat().st_size,
        "sha256": sha256(arrays_path.read_bytes()).hexdigest(),
    }
    members = {
        name: {
            "shape": list(value.shape),
            "dtype": value.dtype.str,
            "array_sha256": trainer._array_sha256(value),
        }
        for name, value in arrays.items()
    }
    metadata = trainer._self_hashed(
        {
            "schema": trainer.PAIRED_BANK_SCHEMA,
            "status": "complete",
            "experiment_id": trainer.EXPERIMENT_ID,
            "extension_contract_sha256": trainer.EXTENSION_CONTRACT_SHA256,
            "preregistration_sha256": trainer.EXTENSION_PREREGISTRATION_SHA256,
            "dataset_final_hash_manifest_sha256": trainer.DATASET_FINAL_SHA256,
            "pilot_final_hash_manifest_sha256": pilot_sha,
            "population_role": "train_only",
            "centers_inclusive": [956, 1193],
            "center_count": 238,
            "base_seed": 20260902,
            "antithetic_signs": [-1, 1],
            "input_count": 476,
            "all_solver_cases_succeeded": True,
            "pilot_gate_passed": True,
            "arrays": {"file": arrays_file_record, "members": members},
            "prospective_opened": False,
            "sealed_opened": False,
            "online_solver_calls": False,
        }
    )
    metadata_path = bank_root / trainer.PAIRED_BANK_METADATA_FILE
    _write_json(metadata_path, metadata)
    final = {
        "schema": trainer.PAIRED_BANK_FINAL_SCHEMA,
        "experiment_id": trainer.EXPERIMENT_ID,
        "packet_role": "paired_query_bank",
        "extension_contract_sha256": trainer.EXTENSION_CONTRACT_SHA256,
        "preregistration_sha256": trainer.EXTENSION_PREREGISTRATION_SHA256,
        "dataset_final_hash_manifest_sha256": trainer.DATASET_FINAL_SHA256,
        "pilot_final_hash_manifest_sha256": pilot_sha,
        "files": {
            trainer.PAIRED_BANK_ARRAY_FILE: arrays_file_record,
            trainer.PAIRED_BANK_METADATA_FILE: {
                "relative_path": trainer.PAIRED_BANK_METADATA_FILE,
                "bytes": metadata_path.stat().st_size,
                "sha256": sha256(metadata_path.read_bytes()).hexdigest(),
            },
        },
        "self_hash_excluded": True,
        "prospective_opened": False,
        "sealed_opened": False,
        "online_solver_calls": False,
    }
    final_path = bank_root / "final_hash_manifest.json"
    _write_json(final_path, final)
    final_sha = sha256(final_path.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="final manifest differs"):
        trainer._load_paired_bank(
            bank_root,
            expected_final_sha256=final_sha,
            pilot_final_manifest_path=pilot,
            expected_pilot_final_sha256=pilot_sha,
            expected_coordinates=np.zeros((1, 2), dtype=np.float64),
            train_states=np.zeros((240, 1, 5), dtype=np.float64),
            train_frame_indices=np.arange(955, 1195, dtype=np.int64),
            normalization=_normalization(),
        )


def test_atomic_directory_is_not_published_before_commit(workspace_tmp_path) -> None:
    final = workspace_tmp_path / "packet"
    staging, resolved = trainer._new_staging_directory(final)
    assert staging.is_dir()
    assert not resolved.exists()
    (staging / "marker").write_text("complete", encoding="utf-8")
    trainer._commit_staging_directory(staging, resolved)
    assert not staging.exists()
    assert (resolved / "marker").read_text(encoding="utf-8") == "complete"
