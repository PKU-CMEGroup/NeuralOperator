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

    monkeypatch.setattr(
        audit, "_seed0_descriptors", lambda *_: rows((audit.SEEDS[0],))
    )
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
