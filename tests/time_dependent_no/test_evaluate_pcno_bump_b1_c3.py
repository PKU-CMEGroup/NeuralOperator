from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.time_dependent_no import evaluate_pcno_bump_b1_c3 as b1_c3


def _metric_row(*, epoch: int, optimizer_step: int) -> dict:
    return {
        "epoch": epoch,
        "train": {
            "completed_optimizer_steps": optimizer_step,
            "relative_l2": 0.1,
            "comparable_seen": {"relative_l2": 0.2},
        },
        "validation": {"relative_l2": 0.3},
        "rollout": {
            "mean_full_horizon_relative_l2": 0.4,
            "mean_endpoint_relative_l2": {"79": 0.5},
            "completion_rate": 1.0,
            "hard_failure_count": 0,
            "physical_admissibility_rate": 0.75,
        },
    }


def _sentinel_base(replication_root: Path, *, count: int = 8) -> dict:
    run_dir = replication_root / "outputs" / "run"
    run_dir.mkdir(parents=True)
    step = b1_c3.matched_exposure_step(count)
    (run_dir / "metrics.jsonl").write_text(
        json.dumps(_metric_row(epoch=step // 256 - 1, optimizer_step=step)) + "\n",
        encoding="utf-8",
    )
    sentinel = run_dir / "sentinels" / f"step_{step:09d}.pt"
    sentinel.parent.mkdir()
    sentinel.write_bytes(b"sentinel")
    return {
        "seed": 20_260_812,
        "trajectory_count": count,
        "architecture": "pcno",
        "run": run_dir.name,
        "run_dir": run_dir,
        "schedule": "b1_c2_replication_stretched",
        "split": {"val_keys": ["v"]},
        "summary": {
            "config_digest": "config",
            "normalization_digest": "normalization",
        },
        "expected_checkpoint_source_set_digest": (
            b1_c3.EXPECTED_REPLICATION_SOURCE_SET_DIGEST
        ),
    }


def _sentinel_payload(*, count: int = 8) -> dict:
    step = b1_c3.matched_exposure_step(count)
    return {
        "epoch": step // 256 - 1,
        "checkpoint_role": "model_only_sentinel",
        "resume_supported": False,
        "sentinel_contract": {
            "schema": b1_c3.SENTINEL_SCHEMA,
            "evaluation_initialization_supported": True,
            "exact_training_resume_supported": False,
        },
        "data_manifest_digest": b1_c3.EXPECTED_DATA_MANIFEST_DIGEST,
        "source_snapshot": {
            "source_set_digest": b1_c3.EXPECTED_REPLICATION_SOURCE_SET_DIGEST
        },
        "config_digest": "config",
        "normalization_digest": "normalization",
    }


def test_matched_exposure_step_is_exact_registered_inventory() -> None:
    assert [b1_c3.matched_exposure_step(count) for count in (8, 16, 256)] == [
        512,
        1_024,
        16_384,
    ]
    with pytest.raises(ValueError, match="unregistered"):
        b1_c3.matched_exposure_step(7)


def test_parser_requires_explicit_precision() -> None:
    parser = b1_c3.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--replication-root",
                "replication",
                "--b1-c2-audit-root",
                "audit",
                "--data-dir",
                "data",
                "--output-dir",
                "output",
            ]
        )
    args = parser.parse_args(
        [
            "--replication-root",
            "replication",
            "--b1-c2-audit-root",
            "audit",
            "--data-dir",
            "data",
            "--output-dir",
            "output",
            "--amp",
            "none",
        ]
    )
    assert args.amp == "none"


def test_metric_at_step_requires_one_exact_history_row(tmp_path: Path) -> None:
    row = _metric_row(epoch=1, optimizer_step=512)
    (tmp_path / "metrics.jsonl").write_text(
        json.dumps(row) + "\n", encoding="utf-8"
    )
    snapshot = b1_c3._metric_at_step(tmp_path, 512)
    assert snapshot["optimizer_step"] == 512
    assert snapshot["stored_fixed_validation_one_step_relative_l2"] == pytest.approx(
        0.3
    )
    with pytest.raises(ValueError, match="does not have one exact"):
        b1_c3._metric_at_step(tmp_path, 1_024)


def test_sentinel_descriptor_binds_receipt_payload_and_metric_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = _sentinel_base(tmp_path)
    checkpoint = Path(base["run_dir"]) / "sentinels" / "step_000000512.pt"
    relative = checkpoint.relative_to(tmp_path).as_posix()
    monkeypatch.setattr(b1_c3, "load_bump_checkpoint", lambda _: _sentinel_payload())
    monkeypatch.setattr(b1_c3, "sha256_file", lambda _: "sentinel-sha")
    descriptor = b1_c3._sentinel_descriptor(
        base,
        replication_root=tmp_path,
        file_records={
            relative: {"bytes": checkpoint.stat().st_size, "sha256": "sentinel-sha"}
        },
    )
    assert descriptor["checkpoint_role"] == "matched_exposure"
    assert descriptor["optimizer_step"] == 512
    assert descriptor["presentations_per_training_trajectory"] == 64
    assert descriptor["selected_training_metrics"]["epoch"] == 1


def test_sentinel_descriptor_rejects_resume_state_and_receipt_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = _sentinel_base(tmp_path)
    checkpoint = Path(base["run_dir"]) / "sentinels" / "step_000000512.pt"
    relative = checkpoint.relative_to(tmp_path).as_posix()
    monkeypatch.setattr(b1_c3, "sha256_file", lambda _: "sentinel-sha")
    monkeypatch.setattr(b1_c3, "load_bump_checkpoint", lambda _: _sentinel_payload())
    with pytest.raises(ValueError, match="differs from matrix receipt"):
        b1_c3._sentinel_descriptor(
            base,
            replication_root=tmp_path,
            file_records={relative: {"bytes": 1, "sha256": "sentinel-sha"}},
        )

    payload = _sentinel_payload()
    payload["optimizer_state"] = {}
    monkeypatch.setattr(b1_c3, "load_bump_checkpoint", lambda _: payload)
    with pytest.raises(ValueError, match="sentinel payload changed"):
        b1_c3._sentinel_descriptor(
            base,
            replication_root=tmp_path,
            file_records={
                relative: {
                    "bytes": checkpoint.stat().st_size,
                    "sha256": "sentinel-sha",
                }
            },
        )


def test_audit_cohorts_reuses_two_seed_holdouts_and_common_nine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validation = [str(index) for index in range(44)]
    common = validation[9:18]
    outside = {
        20_260_812: validation[:28],
        20_260_813: validation[9:37],
    }
    selection = {
        seed: [key for key in validation if key not in keys]
        for seed, keys in outside.items()
    }
    prior = {
        "selection_keys_by_seed": {
            str(seed): keys for seed, keys in selection.items()
        },
        "outside_selection_keys_by_seed": {
            str(seed): keys for seed, keys in outside.items()
        },
        "common_outside_selection_keys": common,
    }
    descriptors = [
        {"seed": seed, "split": {"seed": seed}}
        for seed in b1_c3.REPLICATION_SEEDS
    ]

    def fake_outside(_manifest: dict, splits: list[dict]) -> tuple:
        seed = int(splits[0]["seed"])
        return validation, selection[seed], outside[seed]

    monkeypatch.setattr(b1_c3, "outside_selection_keys", fake_outside)
    monkeypatch.setattr(
        b1_c3,
        "_ordered_key_digest",
        lambda keys: (
            b1_c3.EXPECTED_VALIDATION_KEY_DIGEST
            if len(keys) == 44
            else b1_c3.EXPECTED_COMMON_OUTSIDE_KEY_DIGEST
        ),
    )
    result = b1_c3._audit_cohorts(
        {"split": {"open_validation_keys": validation}}, descriptors, prior
    )
    assert result["common_outside_selection_keys"] == common
    assert result["outside_selection_keys_by_seed"]["20260812"] == outside[
        20_260_812
    ]


def test_prior_audit_must_be_a_regular_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="regular directory"):
        b1_c3._validate_prior_audit(tmp_path / "missing")


def test_recomputed_scopes_fill_fixed_seen_validation_and_selection_rollout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor = {
        "checkpoint": Path("sentinel.pt"),
        "contract": {
            "fixed_seen_pair_bank_sha256": "seen-bank",
            "fixed_validation_pair_bank_sha256": "validation-bank",
        },
        "selected_training_metrics": {
            "epoch": 1,
            "optimizer_step": 512,
            "online_train_one_step_relative_l2": 0.4,
            "stored_fixed_validation_one_step_relative_l2": 0.31,
        },
    }
    seen_pairs = [("seen", 0)]
    validation_pairs = [("validation", 0)]
    monkeypatch.setattr(
        b1_c3,
        "presentation_stream_sha256",
        lambda pairs: f"{pairs[0][0]}-bank",
    )
    monkeypatch.setattr(
        b1_c3, "load_bump_checkpoint", lambda _: {"step_stride": 1}
    )
    monkeypatch.setattr(b1_c3, "build_bump_checkpoint_model", lambda *_: object())
    one_step = iter(
        [
            {"relative_l2": 0.2},
            {"relative_l2": 0.3},
        ]
    )
    monkeypatch.setattr(b1_c3, "evaluate_pairs", lambda *_, **__: next(one_step))
    monkeypatch.setattr(
        b1_c3,
        "evaluate_rollouts",
        lambda *_, **__: {
            "mean_full_horizon_relative_l2": 0.5,
            "mean_endpoint_relative_l2": {"79": 0.6},
            "completion_rate": 1.0,
            "hard_failure_count": 0,
            "physical_admissibility_rate": 0.75,
        },
    )
    result = b1_c3._evaluate_recomputed_scopes(
        descriptor,
        store=object(),
        fixed_seen_pairs=seen_pairs,
        fixed_validation_pairs=validation_pairs,
        selection_keys=["selection"],
        policies={},
        device=b1_c3.torch.device("cpu"),
        amp="none",
    )
    metrics = result["checkpoint_training_metrics"]
    assert metrics["fixed_seen_train_one_step_relative_l2"] == pytest.approx(0.2)
    assert metrics["fixed_validation_one_step_relative_l2"] == pytest.approx(0.3)
    assert metrics["rollout_h79_relative_l2"] == pytest.approx(0.6)
    assert result["fixed_seen_one_step"]["pair_bank_sha256"] == "seen-bank"
    assert result["fixed_validation_one_step"]["pair_bank_sha256"] == (
        "validation-bank"
    )
