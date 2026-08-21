from __future__ import annotations

from scripts.time_dependent_no.analyze_pcno_bump_scaling_ladder import (
    ARCHITECTURES,
    COUNTS,
    build_parser,
    matched_exposure_rows,
    paired_ratio_bootstrap,
    shared_bootstrap_indices,
)


def _snapshot(step: int, *, scale: float, rollout: bool = True) -> dict[str, object]:
    value = scale / step
    return {
        "epoch": step // 256 - 1,
        "optimizer_step": step,
        "learning_rate": 1.0e-3,
        "online_train_one_step_relative_l2": value,
        "fixed_seen_train_one_step_relative_l2": value,
        "fixed_validation_one_step_relative_l2": 1.1 * value,
        "rollout_all_call_mean_relative_l2": 2.0 * value if rollout else None,
        "rollout_h79_relative_l2": 3.0 * value if rollout else None,
        "rollout_completion_rate": 1.0 if rollout else None,
        "rollout_hard_failure_count": 0 if rollout else None,
        "physical_admissibility_rate": 1.0 if rollout else None,
    }


def test_paired_bootstrap_uses_shared_trajectory_indices() -> None:
    draws = shared_bootstrap_indices(3, 100, 17)
    result = paired_ratio_bootstrap([1.0, 2.0, 3.0], [2.0, 4.0, 6.0], draws)

    assert result["ratio"] == 0.5
    assert result["ci95_low"] == 0.5
    assert result["ci95_high"] == 0.5


def test_matched_exposure_rows_require_exact_recorded_rollouts() -> None:
    histories = {}
    for count in COUNTS:
        for architecture in ARCHITECTURES:
            scale = 1.0 if architecture == "pcno" else 2.0
            rows = []
            for step in range(256, 20_481, 256):
                recorded_rollout = step % 1_280 == 0 or step == 256
                rows.append(_snapshot(step, scale=scale, rollout=recorded_rollout))
            histories[(count, architecture)] = {"snapshots": rows}

    rows = matched_exposure_rows(histories)
    identities = {
        (
            int(row["step_multiplier_per_trajectory"]),
            int(row["trajectory_count"]),
            str(row["architecture"]),
            int(row["optimizer_step"]),
        )
        for row in rows
    }

    assert (80, 256, "pcno", 20_480) in identities
    assert (160, 128, "pcfno", 20_480) in identities
    assert (320, 64, "pcno", 20_480) in identities
    assert (640, 32, "pcfno", 20_480) in identities
    assert (1_280, 16, "pcno", 20_480) in identities
    assert not any(multiplier == 80 and count == 8 for multiplier, count, _, _ in identities)


def test_parser_has_no_historical_test_surface() -> None:
    destinations = {action.dest for action in build_parser()._actions}

    assert "test" not in destinations
    assert "test_root" not in destinations
    assert {
        "fresh_ladder_root",
        "n256_gate_root",
        "audit_root",
        "output_dir",
    } <= destinations
