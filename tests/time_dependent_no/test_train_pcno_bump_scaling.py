from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

from scripts.time_dependent_no.train_pcno_bump_scaling import (
    D094_METRIC_SEMANTICS,
    D094_REGISTERED_SOURCE_FILES,
    D094_ROLLOUT_SELECTION_COUNT,
    D094_ROLLOUT_STEPS,
    D094_SELECTION_MODE,
    D094_SENTINEL_STEPS,
    MODEL_ONLY_SENTINEL_PAYLOAD,
    REGISTERED_COUNTS,
    ROOT,
    SCHEMA,
    SPLIT_SCHEMA,
    assert_balanced_epoch_stream,
    balanced_queue_presentations,
    canonical_json_sha256,
    configure_parent_args,
    fixed_evaluation_pairs,
    installed_scaling_adapter,
    load_split_manifest,
    model_only_sentinel_payload,
    presentation_stream_sha256,
)
from utility.time_dependent_no.pcno_artifacts import (
    PCNO_SOURCE_SNAPSHOT_SCHEMA,
    verify_source_snapshot,
    write_source_snapshot,
)


class FakeStore:
    def __init__(self, steps: dict[str, int]) -> None:
        self.steps = steps

    def entry(self, key: str) -> dict[str, int]:
        return {"num_steps": self.steps[str(key)]}


def _split_payload() -> dict[str, object]:
    train = [str(index) for index in range(256)]
    validation = [f"v{index}" for index in range(44)]
    subsets = {str(count): train[:count] for count in REGISTERED_COUNTS}
    payload: dict[str, object] = {
        "schema": SPLIT_SCHEMA,
        "source_manifest_sha256": "a" * 64,
        "state_arrays_opened": False,
        "historical_test_population_opened": False,
        "split": {
            "seed": 17,
            "train_pool_count": 256,
            "open_validation_count": 44,
            "train_pool_keys": train,
            "open_validation_keys": validation,
        },
        "nested_exposure": {
            "counts": list(REGISTERED_COUNTS),
            "subsets": subsets,
        },
    }
    payload["partition_digest"] = canonical_json_sha256(
        {"split": payload["split"], "nested_exposure": payload["nested_exposure"]}
    )
    payload["canonical_payload_sha256"] = canonical_json_sha256(payload)
    return payload


def test_split_manifest_closes_and_rejects_mutation(tmp_path) -> None:
    path = tmp_path / "split.json"
    payload = _split_payload()
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_split_manifest(path, require_registered=False) == payload

    mutated = copy.deepcopy(payload)
    mutated["split"]["open_validation_keys"][0] = "0"
    path.write_text(json.dumps(mutated), encoding="utf-8")
    with pytest.raises(ValueError, match="canonical payload digest"):
        load_split_manifest(path, require_registered=False)


def test_balanced_queue_exhausts_every_window_before_reuse() -> None:
    store = FakeStore({"a": 6, "b": 6})
    first = balanced_queue_presentations(
        store, ["a", "b"], step_stride=1, count=6, seed=11, epoch=0
    )
    second = balanced_queue_presentations(
        store, ["a", "b"], step_stride=1, count=6, seed=11, epoch=1
    )
    repeated = balanced_queue_presentations(
        store, ["a", "b"], step_stride=1, count=6, seed=11, epoch=0
    )
    assert first == repeated
    assert presentation_stream_sha256(first) == presentation_stream_sha256(repeated)
    assert_balanced_epoch_stream(first, ["a", "b"], expected_count=6)

    for key in ("a", "b"):
        windows = [index for pair_key, index in [*first, *second] if pair_key == key]
        assert len(windows) == 6
        assert set(windows[:5]) == set(range(5))
        assert windows[5] in range(5)


def test_balanced_queue_rejects_unbalanced_epoch() -> None:
    store = FakeStore({"a": 80, "b": 80, "c": 80})
    with pytest.raises(ValueError, match="divide evenly"):
        balanced_queue_presentations(
            store, ["a", "b", "c"], step_stride=1, count=8, seed=1, epoch=0
        )


def test_fixed_evaluation_bank_is_field_blind_and_evenly_spaced() -> None:
    pairs = fixed_evaluation_pairs(
        FakeStore({"a": 80, "b": 80}),
        ["a", "b"],
        step_stride=1,
        windows_per_trajectory=4,
    )
    assert pairs == [
        ("a", 0),
        ("a", 26),
        ("a", 52),
        ("a", 78),
        ("b", 0),
        ("b", 26),
        ("b", 52),
        ("b", 78),
    ]


def test_parent_args_are_forced_to_one_pair_per_update() -> None:
    parent = SimpleNamespace(
        split_mode="manifest",
        split_seed=0,
        val_count=1,
        presentation_mode="full_coverage",
        presentations_per_epoch=999,
        batch_size=4,
        gradient_accumulation_steps=7,
        tiny_pairs=8,
        val_presentations=1,
        checkpoint_every=99,
        init_checkpoint=None,
        resume_checkpoint=None,
        step_stride=1,
        multistep_loss_steps=1,
        multistep_loss_weight=0.0,
        generated_state_exposure_weight=0.0,
        input_noise_std=0.0,
        epochs=80,
        rollout_checkpoints=(),
        rollout_val_count=5,
        rollout_steps=20,
        selection_mode="historical_all_node",
        rollout_failure_policy="strict_physical",
        scheduler="warmup_cosine",
        warmup_cosine_decay_steps=5_120,
    )
    wrapper = SimpleNamespace(
        split_manifest=(
            ROOT / "docs/time_dependent_no/D094_BUMP_SCALING_SPLIT_MANIFEST.json"
        ),
        optimizer_steps_per_epoch=256,
        evaluation_windows_per_trajectory=4,
        differential_branch_mode="full",
        trajectory_count=64,
        comparable_seen_every_epochs=5,
        sentinel_every_epochs=0,
        sentinel_steps=D094_SENTINEL_STEPS,
        sentinel_payload=MODEL_ONLY_SENTINEL_PAYLOAD,
        engineering_smoke=False,
    )
    configured = configure_parent_args(parent, wrapper, _split_payload())
    assert configured.batch_size == 1
    assert configured.gradient_accumulation_steps == 1
    assert configured.presentations_per_epoch == 256
    assert configured.val_presentations == 176
    assert configured.rollout_val_count == D094_ROLLOUT_SELECTION_COUNT
    assert configured.rollout_steps == D094_ROLLOUT_STEPS
    assert configured.rollout_checkpoints == [D094_ROLLOUT_STEPS]
    assert configured.selection_mode == D094_SELECTION_MODE
    assert configured.rollout_failure_policy == "finite_only"
    assert configured.d094_schedule_arm == "prefix_tail"
    assert configured.scaling_partition_digest
    assert set(configured.source_snapshot_extra_files) == {
        *D094_REGISTERED_SOURCE_FILES,
        "docs/time_dependent_no/D094_BUMP_SCALING_SPLIT_MANIFEST.json",
    }
    assert SCHEMA.startswith("d094_")

    stretched = copy.deepcopy(parent)
    stretched.warmup_cosine_decay_steps = 20_480
    assert (
        configure_parent_args(stretched, copy.deepcopy(wrapper), _split_payload())
        .d094_schedule_arm
        == "stretched"
    )
    unregistered = copy.deepcopy(parent)
    unregistered.warmup_cosine_decay_steps = 0
    with pytest.raises(ValueError, match="explicit 5,120-step"):
        configure_parent_args(unregistered, copy.deepcopy(wrapper), _split_payload())
    bad_sentinels = copy.deepcopy(wrapper)
    bad_sentinels.sentinel_steps = D094_SENTINEL_STEPS[:-1]
    with pytest.raises(ValueError, match="sentinel-step inventory"):
        configure_parent_args(copy.deepcopy(parent), bad_sentinels, _split_payload())


def test_d094_extension_sources_are_copied_into_v6_snapshot(tmp_path) -> None:
    split = ROOT / "docs/time_dependent_no/D094_BUMP_SCALING_SPLIT_MANIFEST.json"
    parent = SimpleNamespace(
        split_mode="manifest",
        split_seed=0,
        val_count=1,
        presentation_mode="full_coverage",
        presentations_per_epoch=999,
        batch_size=4,
        gradient_accumulation_steps=7,
        tiny_pairs=8,
        val_presentations=1,
        checkpoint_every=99,
        init_checkpoint=None,
        resume_checkpoint=None,
        step_stride=1,
        multistep_loss_steps=1,
        multistep_loss_weight=0.0,
        generated_state_exposure_weight=0.0,
        input_noise_std=0.0,
        epochs=80,
        rollout_checkpoints=(),
        rollout_val_count=5,
        rollout_steps=20,
        selection_mode="historical_all_node",
        rollout_failure_policy="strict_physical",
        scheduler="warmup_cosine",
        warmup_cosine_decay_steps=5_120,
    )
    wrapper = SimpleNamespace(
        split_manifest=split,
        optimizer_steps_per_epoch=256,
        evaluation_windows_per_trajectory=4,
        differential_branch_mode="full",
        trajectory_count=64,
        comparable_seen_every_epochs=5,
        sentinel_every_epochs=0,
        sentinel_steps=D094_SENTINEL_STEPS,
        sentinel_payload=MODEL_ONLY_SENTINEL_PAYLOAD,
        engineering_smoke=False,
    )
    configured = configure_parent_args(parent, wrapper, _split_payload())
    snapshot = write_source_snapshot(
        tmp_path / "run",
        extra_source_files=configured.source_snapshot_extra_files,
    )

    assert snapshot["schema"] == PCNO_SOURCE_SNAPSHOT_SCHEMA
    assert snapshot["extra_source_files"] == list(
        configured.source_snapshot_extra_files
    )
    verify_source_snapshot(snapshot)


def test_installed_adapter_uses_frozen_split_metrics_and_sentinels(tmp_path) -> None:
    store = FakeStore({"a": 80, "b": 80, "v": 80})
    saved_paths = []
    saved_payloads = []

    def original_atomic_save(payload, path) -> None:
        saved_paths.append(path)
        saved_payloads.append(payload)

    def original_train_epoch(*values, **keywords):
        del values, keywords
        return {"optimizer_steps": 2}

    trainer = SimpleNamespace(
        parse_args=lambda _: None,
        stratified_train_val_split=lambda *args, **kwargs: (args, kwargs),
        epoch_presentations=lambda *args, **kwargs: (args, kwargs),
        balanced_presentations=lambda *args, **kwargs: (args, kwargs),
        build_model=lambda *args, **kwargs: (args, kwargs),
        train_epoch=original_train_epoch,
        checkpoint_payload=lambda *args, **kwargs: {
            "model_config": {},
            "training_args": {},
        },
        atomic_torch_save=original_atomic_save,
        evaluate_pairs=lambda *args, **kwargs: {
            "loss": 1.0,
            "relative_l2": 2.0,
            "presentations": len(args[2]),
        },
    )
    args = SimpleNamespace(
        split_seed=7,
        step_stride=1,
        presentations_per_epoch=2,
        seed=11,
        val_presentations=1,
        epochs=1,
    )
    wrapper = SimpleNamespace(
        evaluation_windows_per_trajectory=1,
        differential_branch_mode="full",
        comparable_seen_every_epochs=1,
        sentinel_every_epochs=0,
        sentinel_steps=(2,),
        sentinel_payload=MODEL_ONLY_SENTINEL_PAYLOAD,
    )
    hashes = []
    with installed_scaling_adapter(
        trainer,
        args=args,
        wrapper=wrapper,
        train_keys=["a", "b"],
        validation_keys=["v"],
        scaling_contract={},
        presentation_hashes=hashes,
    ):
        assert trainer.stratified_train_val_split(
            store, val_count=1, seed=7
        ) == (["a", "b"], ["v"])
        assert trainer.balanced_presentations(
            store,
            ["v"],
            step_stride=1,
            count=1,
            rng=None,
        ) == [("v", 0)]
        pairs = trainer.epoch_presentations(
            store, ["a", "b"], args=args, epoch=0, tiny_bank=None
        )
        metrics = trainer.train_epoch(
            object(),
            store,
            pairs,
            device="cpu",
            amp="none",
            primary_objective="normal_closed",
            boundary_policies={},
        )
        assert metrics["completed_optimizer_steps"] == 2
        assert metrics["window_equivalent_exposure_per_trajectory"] == 2 / 158
        assert metrics["comparable_seen"]["presentations"] == 2
        assert len(hashes) == 1

        annotated = trainer.checkpoint_payload(
            model=SimpleNamespace(differential_branch_contract={"mode": "full"})
        )
        assert annotated["resume_supported"] is False
        assert "exact-resume exposure accounting" in annotated["resume_blocker"]

        last = tmp_path / "last.pt"
        trainer.atomic_torch_save(
            {
                "epoch": 0,
                "model_state": {"weight": "model"},
                "optimizer_state": {"state": "optimizer"},
                "scheduler_state": {"state": "scheduler"},
                "checkpoint_role": "training",
                "resume_supported": True,
            },
            last,
        )
        assert saved_paths == [
            last,
            tmp_path / "sentinels" / "step_000000002.pt",
        ]
        assert "optimizer_state" in saved_payloads[0]
        assert "optimizer_state" not in saved_payloads[1]
        assert "scheduler_state" not in saved_payloads[1]
        assert saved_payloads[1]["checkpoint_role"] == "model_only_sentinel"
        assert saved_payloads[1]["resume_supported"] is False
        assert (tmp_path / "sentinels").is_dir()

    assert trainer.train_epoch is original_train_epoch
    assert trainer.atomic_torch_save is original_atomic_save


def test_model_only_sentinel_retains_evaluation_state_but_not_resume_state() -> None:
    payload = {
        "epoch": 7,
        "model_state": {"weight": "model"},
        "optimizer_state": {"state": "optimizer"},
        "scheduler_state": {"state": "scheduler"},
        "checkpoint_role": "training",
        "resume_supported": True,
    }
    sentinel = model_only_sentinel_payload(payload)

    assert sentinel["model_state"] == payload["model_state"]
    assert "optimizer_state" not in sentinel
    assert "scheduler_state" not in sentinel
    assert sentinel["checkpoint_role"] == "model_only_sentinel"
    assert sentinel["resume_supported"] is False
    assert payload["resume_supported"] is True


def test_metric_semantics_keep_one_step_and_rollout_objects_distinct() -> None:
    assert set(D094_METRIC_SEMANTICS) == {
        "online_train_one_step",
        "fixed_seen_train_one_step",
        "fixed_open_validation_one_step",
        "autonomous_rollout",
    }
    assert "parameters change" in D094_METRIC_SEMANTICS["online_train_one_step"]
    assert "all-call mean and H79" in D094_METRIC_SEMANTICS["autonomous_rollout"]
