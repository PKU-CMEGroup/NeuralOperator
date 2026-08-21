from __future__ import annotations

import json
from pathlib import Path

import torch

from scripts.time_dependent_no.analyze_pcno_zero_gradient_continuation import (
    SCHEMA as ANALYSIS_SCHEMA,
)
from scripts.time_dependent_no.analyze_pcno_zero_gradient_continuation import (
    run_analysis,
)
from scripts.time_dependent_no.continue_pcno_zero_gradient import (
    ARMS,
    SCHEMA,
    SEEDS,
    registered_config,
    run_experiment,
    unwrap_no_gradient_state,
    zero_gradient_outputs,
)
from scripts.time_dependent_no.fit_pcno_shock_representation import build_model
from scripts.time_dependent_no.train_pcno_gradient_ablation import (
    apply_variant,
    shared_state_sha256,
)
from scripts.time_dependent_no.train_pcno_gradient_ablation import (
    registered_config as p2_registered_config,
)
from scripts.time_dependent_no.train_pcno_gradient_ablation import (
    run_experiment as run_p2_experiment,
)


def test_registered_continuation_is_paired_and_zeroes_only_gw2() -> None:
    parent_config = p2_registered_config(
        "no_gradient", 1701, smoke=False, device_type="cuda"
    )
    configs = [
        registered_config(parent_config, arm, seed, smoke=False, device_type="cuda")
        for arm in ARMS
        for seed in SEEDS
    ]

    assert {config["steps"] for config in configs} == {10_000}
    assert {config["optimizer_restart"] for config in configs} == {
        "fresh_matched_adamw_without_parent_moments"
    }
    assert {config["selection_rule"] for config in configs} == {
        "final_continuation_step_only"
    }
    assert len({config["working_run_id"] for config in configs}) == 6

    wrapped = apply_variant(
        build_model(
            p2_registered_config("no_gradient", 1701, smoke=True, device_type="cpu"),
            torch.device("cpu"),
        ),
        "no_gradient",
    )
    full = build_model(
        p2_registered_config("no_gradient", 1701, smoke=True, device_type="cpu"),
        torch.device("cpu"),
    )
    full.load_state_dict(unwrap_no_gradient_state(wrapped.state_dict()), strict=True)
    shared_before = shared_state_sha256(full)
    rows = zero_gradient_outputs(full)

    assert shared_state_sha256(full) == shared_before
    assert all(row["gw2_nonzero"] == 0.0 for row in rows)
    assert all(abs(row["gw1"] - 0.01) < 1.0e-8 for row in rows)
    assert all(torch.count_nonzero(module.gw2.weight) == 0 for module in full.gws)


def test_two_arm_smoke_closes_parent_and_generates_analysis(tmp_path: Path) -> None:
    parent_dir = tmp_path / "parent_no_gradient_s1701"
    run_p2_experiment(
        parent_dir,
        variant="no_gradient",
        seed=1701,
        smoke=True,
        device_name="cpu",
    )
    for arm in ARMS:
        output_dir = tmp_path / f"{arm}_s1701"
        summary = run_experiment(
            output_dir,
            parent_dir,
            arm=arm,
            seed=1701,
            smoke=True,
            device_name="cpu",
        )
        manifest = json.loads(
            (output_dir / "manifest.json").read_text(encoding="utf-8")
        )
        assert summary["schema"] == SCHEMA
        assert summary["science_result"] is False
        assert summary["completed_steps"] == 2
        assert summary["initialization"]["parent_output_max_abs"] == 0.0
        assert summary["branch_replay_max_abs"] <= 1.0e-5
        assert manifest["status"] == "completed"
        assert (output_dir / "branch_cube_metrics.json").is_file()
        assert (output_dir / "branch_cube_predictions.npz").is_file()
        json.dumps(summary, allow_nan=False)

    control = json.loads(
        (tmp_path / "continue_no_gradient_s1701" / "run_contract.json").read_text(
            encoding="utf-8"
        )
    )
    active = json.loads(
        (tmp_path / "activate_zero_gradient_s1701" / "run_contract.json").read_text(
            encoding="utf-8"
        )
    )
    assert control["parent"] == active["parent"]
    assert (
        control["initialization"]["first_backward"]["shared_gradient_sha256"]
        == active["initialization"]["first_backward"]["shared_gradient_sha256"]
    )
    active_branches = active["initialization"]["first_backward"]["branches"]
    assert all(row["gw1_gradient_l2"] == 0.0 for row in active_branches)
    assert any(row["gw2_gradient_l2"] > 0.0 for row in active_branches)

    analysis = run_analysis(tmp_path, tmp_path / "analysis", smoke=True)
    visual = json.loads(
        (tmp_path / "analysis" / "visual_manifest.json").read_text(encoding="utf-8")
    )
    assert analysis["schema"] == ANALYSIS_SCHEMA
    assert analysis["science_result"] is False
    assert analysis["run_count"] == 2
    assert analysis["pairing_closures"]["all_passed"] is True
    assert len(visual["files"]) == 16
    assert set(visual["files"]) == set(visual["sha256"])
    json.dumps(analysis, allow_nan=False)
