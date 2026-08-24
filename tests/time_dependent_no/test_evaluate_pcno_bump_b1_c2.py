from __future__ import annotations

import pytest

from scripts.time_dependent_no import evaluate_pcno_bump_b1_c2 as audit


def _matrix_receipt() -> dict:
    return {
        "schema": audit.MATRIX_RECEIPT_SCHEMA,
        "status": "complete",
        "attempt": "d094_b1_c2_seed_replication_20260824b",
        "cell_count": 24,
        "seeds": list(audit.REPLICATION_SEEDS),
        "trajectory_counts": list(audit.TRAJECTORY_COUNTS),
        "architectures": list(audit.ARCHITECTURES),
        "source_commit": audit.EXPECTED_REPLICATION_SOURCE_COMMIT,
        "source_archive_sha256": audit.EXPECTED_REPLICATION_ARCHIVE_SHA256,
        "source_set_digest": audit.EXPECTED_REPLICATION_SOURCE_SET_DIGEST,
        "split_partition_digest": audit.EXPECTED_SPLIT_PARTITION_DIGEST,
        "historical_test_population_accessed": False,
        "files": [{"path": "outputs/cell/best.pt", "sha256": "0" * 64}],
    }


def _contract(*, architecture: str = "pcno") -> dict:
    return {
        "schema": audit.TRAINING_CONTRACT_SCHEMA,
        "status": "completed_replication_cell",
        "registered_stage": "b1_c2_seed_replication_20480",
        "initialization_seed": audit.REPLICATION_SEEDS[0],
        "trajectory_count": 128,
        "differential_branch_mode": (
            "full" if architecture == "pcno" else "no_gradient"
        ),
        "schedule_arm": "b1_c2_replication_stretched",
        "requested_optimizer_steps": audit.EXPECTED_OPTIMIZER_STEPS,
        "actual_optimizer_steps": audit.EXPECTED_OPTIMIZER_STEPS,
        "requested_epochs": 80,
        "completed_epochs": 80,
        "optimizer_steps_per_epoch": audit.STEPS_PER_EPOCH,
        "checkpoint_selection_mode": "full_horizon_error_first",
        "rollout_failure_policy": "finite_only",
        "rollout_selection_trajectory_count": 16,
        "rollout_steps": 79,
        "historical_test_population_accessed": False,
        "automatic_continuation_authorized": False,
        "source_manifest_sha256": audit.EXPECTED_DATA_MANIFEST_DIGEST,
        "split_partition_digest": audit.EXPECTED_SPLIT_PARTITION_DIGEST,
        "selected_terminal_checkpoint_payload": "model_only_evaluation",
        "sentinel_steps": [8_192],
    }


def test_matrix_receipt_requires_exact_closed_attempt() -> None:
    receipt = _matrix_receipt()
    audit._validate_matrix_receipt(receipt)

    receipt["cell_count"] = 23
    with pytest.raises(ValueError, match="cell_count"):
        audit._validate_matrix_receipt(receipt)


@pytest.mark.parametrize("architecture", audit.ARCHITECTURES)
def test_replication_contract_requires_exact_registered_cell(
    architecture: str,
) -> None:
    contract = _contract(architecture=architecture)
    audit._validate_replication_contract(
        contract,
        seed=audit.REPLICATION_SEEDS[0],
        trajectory_count=128,
        architecture=architecture,
    )

    contract["actual_optimizer_steps"] = 20_479
    with pytest.raises(ValueError, match="actual_optimizer_steps"):
        audit._validate_replication_contract(
            contract,
            seed=audit.REPLICATION_SEEDS[0],
            trajectory_count=128,
            architecture=architecture,
        )


def test_cell_receipt_preserves_selected_and_terminal_scopes() -> None:
    selected = {"epoch": 54, "optimizer_step": 14_080}
    terminal = {"epoch": 79, "optimizer_step": 20_480}
    receipt = {
        "schema": audit.CELL_RECEIPT_SCHEMA,
        "status": "pass",
        "cell": "cell",
        "seed": audit.REPLICATION_SEEDS[0],
        "trajectory_count": 128,
        "architecture": "pcfno",
        "branch": "no_gradient",
        "source_set_digest": audit.EXPECTED_REPLICATION_SOURCE_SET_DIGEST,
        "split_partition_digest": audit.EXPECTED_SPLIT_PARTITION_DIGEST,
        "historical_test_population_accessed": False,
        "selected_checkpoint": selected,
        "terminal_checkpoint": terminal,
        "sentinel_step": 8_192,
    }
    audit._validate_cell_receipt(
        receipt,
        run_name="cell",
        seed=audit.REPLICATION_SEEDS[0],
        trajectory_count=128,
        architecture="pcfno",
        scopes={"selected_checkpoint": selected, "terminal_checkpoint": terminal},
    )

    receipt["terminal_checkpoint"] = selected
    with pytest.raises(ValueError, match="terminal_checkpoint"):
        audit._validate_cell_receipt(
            receipt,
            run_name="cell",
            seed=audit.REPLICATION_SEEDS[0],
            trajectory_count=128,
            architecture="pcfno",
            scopes={
                "selected_checkpoint": selected,
                "terminal_checkpoint": terminal,
            },
        )


def test_parser_requires_all_three_checkpoint_roots() -> None:
    destinations = {action.dest for action in audit.build_parser()._actions}
    assert {
        "seed0_ladder_root",
        "seed0_n256_root",
        "replication_root",
        "data_dir",
        "split_manifest",
        "output_dir",
    } <= destinations


def test_three_seed_discovery_uses_registered_audit_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def rows(seeds: tuple[int, ...]) -> list[dict]:
        return [
            {
                "seed": seed,
                "trajectory_count": count,
                "architecture": architecture,
                "checkpoint_role": role,
            }
            for seed in seeds
            for count in reversed(audit.TRAJECTORY_COUNTS)
            for architecture in reversed(audit.ARCHITECTURES)
            for role in reversed(audit.CHECKPOINT_ROLES)
        ]

    monkeypatch.setattr(audit, "_seed0_descriptors", lambda *_: rows((audit.SEEDS[0],)))
    monkeypatch.setattr(
        audit,
        "_replication_descriptors",
        lambda *_: rows(audit.REPLICATION_SEEDS),
    )

    discovered = audit.discover_checkpoints(  # type: ignore[arg-type]
        None, None, None, {}
    )

    assert len(discovered) == 72
    assert [
        (
            row["seed"],
            row["trajectory_count"],
            row["architecture"],
            row["checkpoint_role"],
        )
        for row in discovered[:4]
    ] == [
        (audit.SEEDS[0], 8, "pcno", "selected"),
        (audit.SEEDS[0], 8, "pcno", "terminal"),
        (audit.SEEDS[0], 8, "pcfno", "selected"),
        (audit.SEEDS[0], 8, "pcfno", "terminal"),
    ]


def test_audit_cohorts_preserve_seed_specific_holdouts_and_common_nine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validation = [f"v{index:02d}" for index in range(44)]
    selections = {
        audit.SEEDS[0]: validation[0:16],
        audit.SEEDS[1]: validation[10:26],
        audit.SEEDS[2]: validation[19:35],
    }
    monkeypatch.setattr(
        audit, "EXPECTED_VALIDATION_KEY_DIGEST", audit._ordered_key_digest(validation)
    )
    monkeypatch.setattr(
        audit,
        "EXPECTED_SELECTION_KEY_DIGESTS",
        {seed: audit._ordered_key_digest(keys) for seed, keys in selections.items()},
    )
    common = validation[35:44]
    monkeypatch.setattr(
        audit,
        "EXPECTED_COMMON_OUTSIDE_KEY_DIGEST",
        audit._ordered_key_digest(common),
    )
    manifest = {
        "schema": audit.SPLIT_SCHEMA,
        "state_arrays_opened": False,
        "historical_test_population_opened": False,
        "partition_digest": audit.EXPECTED_SPLIT_PARTITION_DIGEST,
        "split": {"open_validation_keys": validation},
    }
    descriptors = [
        {
            "seed": seed,
            "split": {
                "val_keys": validation,
                "rollout_keys": selection,
            },
        }
        for seed, selection in selections.items()
        for _ in range(2)
    ]

    cohorts = audit.audit_cohorts(manifest, descriptors)

    assert cohorts["common_outside_selection_keys"] == common
    assert all(
        len(keys) == 28 for keys in cohorts["outside_selection_keys_by_seed"].values()
    )
    assert (
        cohorts["outside_selection_keys_by_seed"][str(audit.SEEDS[0])]
        == validation[16:44]
    )


def test_audit_cohorts_reject_split_drift_within_a_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validation = [f"v{index:02d}" for index in range(44)]
    selection = validation[:16]
    monkeypatch.setattr(
        audit, "EXPECTED_VALIDATION_KEY_DIGEST", audit._ordered_key_digest(validation)
    )
    monkeypatch.setattr(
        audit,
        "EXPECTED_SELECTION_KEY_DIGESTS",
        {seed: audit._ordered_key_digest(selection) for seed in audit.SEEDS},
    )
    monkeypatch.setattr(
        audit,
        "EXPECTED_COMMON_OUTSIDE_KEY_DIGEST",
        audit._ordered_key_digest(validation[16:]),
    )
    manifest = {
        "schema": audit.SPLIT_SCHEMA,
        "state_arrays_opened": False,
        "historical_test_population_opened": False,
        "partition_digest": audit.EXPECTED_SPLIT_PARTITION_DIGEST,
        "split": {"open_validation_keys": validation},
    }
    descriptors = [
        {
            "seed": seed,
            "split": {"val_keys": validation, "rollout_keys": selection},
        }
        for seed in audit.SEEDS
    ]
    descriptors.append(
        {
            "seed": audit.SEEDS[0],
            "split": {
                "val_keys": validation,
                "rollout_keys": validation[1:17],
            },
        }
    )

    with pytest.raises(ValueError, match="do not share one validation cohort"):
        audit.audit_cohorts(manifest, descriptors)


def test_rollout_subset_summary_uses_only_requested_rows() -> None:
    rows = []
    for index, key in enumerate(("a", "b", "c"), start=1):
        rows.append(
            {
                "trajectory": key,
                "completed": True,
                "mean_prefix_relative_l2": float(index),
                "endpoint_relative_l2": {
                    str(checkpoint): float(index * checkpoint)
                    for checkpoint in audit.ROLLOUT_CHECKPOINTS
                },
                "hard_failure_cause": None,
                "physically_admissible": index != 2,
            }
        )
    rollout = {"trajectories": rows, "rollout_failure_policy": "finite_only"}

    summary = audit._rollout_subset_summary(rollout, ["a", "c"])

    assert summary["num_trajectories"] == 2
    assert summary["mean_full_horizon_relative_l2"] == 2.0
    assert summary["mean_endpoint_relative_l2"]["79"] == 158.0
    assert summary["physical_admissibility_rate"] == 1.0
