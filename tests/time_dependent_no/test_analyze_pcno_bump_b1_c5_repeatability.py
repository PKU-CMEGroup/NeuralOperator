from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from scripts.time_dependent_no.analyze_pcno_bump_b1_c5_repeatability import (
    analyze_summaries,
    run,
)
from scripts.time_dependent_no.evaluate_pcno_bump_b1_c4 import SCHEMA


def _trajectory(
    key: str,
    error: float,
    *,
    event_call: int | None = None,
) -> dict:
    return {
        "trajectory": key,
        "final_relative_l2": error,
        "first_physical_violation": (
            None
            if event_call is None
            else {"cause": "nonpositive_internal_energy", "call": event_call}
        ),
    }


def _cell(
    count: int,
    role: str,
    h79: float,
    errors: dict[str, float],
    *,
    event_call: int | None = None,
) -> dict:
    return {
        "trajectory_count": count,
        "checkpoint_role": role,
        "optimizer_step": 38_400 if role == "selected" else 40_960,
        "checkpoint_sha256": f"checkpoint-{count}-{role}",
        "config_digest": f"config-{count}",
        "normalization_digest": f"normalization-{count}",
        "outside_selection_rollout": {
            "completion_rate": 1.0,
            "hard_failure_count": 0,
            "mean_endpoint_relative_l2": {"79": h79},
            "mean_full_horizon_relative_l2": 0.75 * h79,
            "physically_admissible_count": 3 if event_call is None else 2,
            "trajectories": [
                _trajectory(
                    key,
                    value,
                    event_call=(event_call if key == "c" else None),
                )
                for key, value in errors.items()
            ],
        },
    }


def _summary(
    amp: str,
    offset: float = 0.0,
    *,
    case_b_n256_offset: float = 0.0,
    event_call: int | None = None,
) -> dict:
    keys = ["a", "b", "c"]
    selected_128 = {"a": 0.03 + offset, "b": 0.05 + offset, "c": 0.04 + offset}
    selected_256 = {
        "a": 0.04 + offset,
        "b": 0.045 + offset + case_b_n256_offset,
        "c": 0.06 + offset,
    }
    return {
        "schema": SCHEMA,
        "status": "complete",
        "historical_test_population_accessed": False,
        "checkpoint_reselection_on_outside_cases": False,
        "split_manifest_sha256": "split-file",
        "split_partition_digest": "partition",
        "data_manifest_sha256": "data",
        "checkpoint_source_set_digest": "checkpoint-source",
        "trajectory_counts": [128, 256],
        "checkpoint_steps": {
            "128": [38_400, 40_960],
            "256": [38_400, 40_960],
        },
        "selection_keys": ["selection"],
        "outside_selection_keys": keys,
        "rollout_horizon": 79,
        "rollout_checkpoints": [20, 40, 60, 79],
        "rollout_failure_policy": "finite_only",
        "shock_quantile": 0.9,
        "boundary_policy_digests": {key: f"policy-{key}" for key in keys},
        "runtime_environment": {"torch": "test", "cuda_device": "test"},
        "evaluator_source_snapshot": {"source_set_digest": "evaluator-source"},
        "evaluation_numerics": {
            "amp": amp,
            "deterministic_algorithms_enabled": False,
            "cudnn_deterministic": False,
            "cudnn_benchmark": False,
            "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": True,
        },
        "cells": [
            _cell(128, "selected", 0.040 + offset, selected_128),
            _cell(
                128,
                "terminal",
                0.042 + offset,
                {key: value + 0.002 for key, value in selected_128.items()},
                event_call=event_call,
            ),
            _cell(256, "selected", 0.050 + offset, selected_256),
            _cell(
                256,
                "terminal",
                0.054 + offset,
                {key: value + 0.004 for key, value in selected_256.items()},
                event_call=event_call,
            ),
        ],
    }


def test_repeatability_analysis_applies_effect_relative_gates() -> None:
    result = analyze_summaries(
        [
            _summary("bf16", 0.0, event_call=20),
            _summary("bf16", 1.0e-5, event_call=20),
            _summary("bf16", -1.0e-5, event_call=21),
        ],
        _summary("none", 2.0e-5, event_call=20),
    )

    assert result["selected_count_ordering"]["robust"] is True
    assert result["selected_to_terminal_direction"]["128"]["direction"] == "worse"
    assert result["selected_to_terminal_direction"]["256"]["robust"] is True
    assert result["per_case_selected_ordering"] == {
        "case_count": 3,
        "stable_winner_count": 3,
        "stable_n128_win_count": 2,
        "stable_n256_win_count": 1,
        "unstable_winner_count": 0,
    }
    assert result["maximum_disagreement_case"]["stable"] is True
    assert result["physical_event_repeatability"]["unstable_event_count"] == 0


def test_repeatability_analysis_rejects_binding_drift() -> None:
    bf16 = [_summary("bf16") for _ in range(3)]
    fp32 = _summary("none")
    fp32["data_manifest_sha256"] = "different"

    with pytest.raises(ValueError, match="inputs or environment differ"):
        analyze_summaries(bf16, fp32)


def test_case_ordering_and_physical_events_fail_locally() -> None:
    first = _summary("bf16", event_call=20)
    second = _summary("bf16", event_call=20)
    third = _summary("bf16", event_call=20)
    fp32 = _summary(
        "none",
        case_b_n256_offset=0.02,
        event_call=23,
    )
    result = analyze_summaries([first, second, third], fp32)

    assert result["selected_count_ordering"]["robust"] is True
    assert result["per_case_selected_ordering"]["unstable_winner_count"] == 1
    assert result["physical_event_repeatability"]["unstable_event_count"] == 2


def test_repeatability_analysis_requires_three_bf16_runs_and_fp32_amp() -> None:
    with pytest.raises(ValueError, match="exactly three"):
        analyze_summaries([_summary("bf16")], _summary("none"))

    fp32 = deepcopy(_summary("none"))
    fp32["evaluation_numerics"]["amp"] = "bf16"
    with pytest.raises(ValueError, match="expected amp='none'"):
        analyze_summaries([_summary("bf16") for _ in range(3)], fp32)


def test_run_writes_hash_bound_analysis_packet(tmp_path: Path) -> None:
    paths = []
    for index, summary in enumerate(
        [
            _summary("bf16", 0.0),
            _summary("bf16", 1.0e-5),
            _summary("bf16", -1.0e-5),
            _summary("none", 2.0e-5),
        ]
    ):
        path = tmp_path / f"summary_{index}.json"
        path.write_text(json.dumps(summary), encoding="utf-8")
        paths.append(path)
    output_dir = tmp_path / "analysis"

    result = run(
        [
            "--bf16-summary",
            str(paths[0]),
            "--bf16-summary",
            str(paths[1]),
            "--bf16-summary",
            str(paths[2]),
            "--fp32-summary",
            str(paths[3]),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert result["status"] == "complete"
    assert (output_dir / "summary.json").is_file()
    assert (output_dir / "cell_repeatability.csv").is_file()
    assert (output_dir / "per_case_selected_ordering.csv").is_file()
    assert (output_dir / "physical_event_repeatability.csv").is_file()
    manifest = json.loads(
        (output_dir / "artifact_manifest.json").read_text(encoding="utf-8")
    )
    assert set(manifest["files"]) == {
        "summary.json",
        "cell_repeatability.csv",
        "per_case_selected_ordering.csv",
        "physical_event_repeatability.csv",
    }
