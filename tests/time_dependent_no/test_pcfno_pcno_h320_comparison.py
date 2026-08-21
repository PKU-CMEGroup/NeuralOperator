from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from scripts.time_dependent_no.evaluate_pcfno_h320_rollout import (
    ANIMATION_SCHEMA as PCFNO_ANIMATION_SCHEMA,
)
from scripts.time_dependent_no.evaluate_pcfno_h320_rollout import (
    CHECKPOINT_SHA256 as PCFNO_CHECKPOINT_SHA256,
)
from scripts.time_dependent_no.evaluate_pcfno_h320_rollout import (
    EVENT_NAMES,
    PRODUCTION_STEPS,
    TRAJECTORY_KEYS,
    TRUTH_STEPS,
    array_sha256,
    event_summary,
    retained_prefix_observations,
)
from scripts.time_dependent_no.evaluate_pcfno_h320_rollout import (
    WORKING_ID as PCFNO_WORKING_ID,
)
from scripts.time_dependent_no.evaluate_pcno_h320_rollout import (
    ANIMATION_SCHEMA as PCNO_ANIMATION_SCHEMA,
)
from scripts.time_dependent_no.evaluate_pcno_h320_rollout import (
    B1_DATA_MANIFEST_SHA256,
    SOURCE_PATHS,
    canonical_prefix_inventory,
    expected_prefix,
    parse_args,
)
from scripts.time_dependent_no.evaluate_pcno_h320_rollout import (
    CHECKPOINT_SHA256 as PCNO_CHECKPOINT_SHA256,
)
from scripts.time_dependent_no.evaluate_pcno_h320_rollout import (
    WORKING_ID as PCNO_WORKING_ID,
)
from scripts.time_dependent_no.visualize_pcfno_pcno_h320_comparison import (
    frame_schedule,
    paired_horizons,
    render_comparison_gif,
    summarize_horizons,
    verify_bundle_pair,
)


def test_parse_args_rejects_output_collision(tmp_path: Path) -> None:
    output = tmp_path / "exists"
    output.mkdir()
    with pytest.raises(FileExistsError):
        parse_args(
            [
                "--mode",
                "smoke",
                "--data-dir",
                str(tmp_path),
                "--training-data-dir",
                str(tmp_path),
                "--checkpoint",
                str(tmp_path / "checkpoint.pt"),
                "--prefix-rollout-dir",
                str(tmp_path),
                "--output-dir",
                str(output),
            ]
        )


def test_source_scope_is_exact_and_does_not_touch_core() -> None:
    assert SOURCE_PATHS[:4] == (
        "docs/time_dependent_no/W26_L1_PCFNO_PCNO_H320_COMPARISON_PREREGISTRATION.md",
        "scripts/time_dependent_no/evaluate_pcno_h320_rollout.py",
        "scripts/time_dependent_no/visualize_pcfno_pcno_h320_comparison.py",
        "tests/time_dependent_no/test_pcfno_pcno_h320_comparison.py",
    )
    assert "scripts/time_dependent_no/evaluate_pcfno_h320_rollout.py" in SOURCE_PATHS
    assert "pcno/pcno.py" not in SOURCE_PATHS


def _prefix_archive(path: Path, key: str) -> None:
    initial = np.arange(8, dtype=np.float32).reshape(2, 4)
    predictions = np.repeat(initial[None], 79, axis=0)
    np.savez(
        path,
        schema=np.asarray("pcno_euler2d_official_rollout_v1"),
        trajectory_key=np.asarray(key),
        initial_conservative=initial,
        pcno_baseline_predictions_conservative=predictions,
        checkpoint_sha256=np.asarray(PCNO_CHECKPOINT_SHA256),
        test_manifest_digest=np.asarray(B1_DATA_MANIFEST_SHA256),
        training_manifest_digest=np.asarray(B1_DATA_MANIFEST_SHA256),
        baseline_valid_length=np.asarray(79, dtype=np.int64),
        baseline_failure_cause=np.asarray("completed"),
        baseline_failure_call=np.asarray(-1, dtype=np.int64),
    )


def test_prefix_inventory_and_h0_h79_loader(tmp_path: Path) -> None:
    trajectory_root = tmp_path / "trajectories"
    trajectory_root.mkdir()
    for key in TRAJECTORY_KEYS:
        _prefix_archive(trajectory_root / f"trajectory_{key}.npz", key)
    inventory = canonical_prefix_inventory(tmp_path)
    assert len(inventory["files"]) == 30
    assert len(inventory["sha256"]) == 64
    records = {row["path"]: row for row in inventory["files"]}
    prefix, contract = expected_prefix(
        "7", prefix_root=tmp_path, prefix_contract={"files": records}
    )
    assert prefix.shape == (80, 2, 4)
    assert prefix.dtype == np.float32
    assert np.array_equal(prefix[0], prefix[1])
    assert contract["source_sha256"] == records["trajectory_7.npz"]["sha256"]


def test_retained_prefix_seeds_events_without_model_replay() -> None:
    initial = np.zeros((4, 4), dtype=np.float32)
    initial[:, 0] = 1.0
    initial[:, 3] = 2.5
    prefix = np.repeat(initial[None], TRUTH_STEPS + 1, axis=0)
    prefix[2, :, 3] = -1.0
    observations = retained_prefix_observations(
        prefix,
        key="synthetic",
        weights=np.ones(4, dtype=np.float64),
        component_scale=np.ones(4, dtype=np.float64),
        envelope={
            "maximum_abs_scaled_primitive": 1.0,
            "maximum_proxy_scaled_rms": 1.0,
        },
        gamma=1.4,
        node_mask=torch.ones((1, 4), dtype=torch.float32),
        state_mean=np.zeros(4, dtype=np.float64),
        state_scale=np.ones(4, dtype=np.float64),
        device=torch.device("cpu"),
    )
    assert len(observations["state_hashes"]) == TRUTH_STEPS
    assert all(observations["events"]["finite"])
    summary = event_summary(observations["events"], horizon=TRUTH_STEPS)
    assert summary["admissible"]["first_failure_call"] == 2
    assert summary["admissible"]["recovered_after_failure"] is True


def _records() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model in ("pcno", "pcfno"):
        for trajectory in range(30):
            for event in EVENT_NAMES:
                if model == "pcno":
                    accepted = PRODUCTION_STEPS
                    censored = True
                    failure = None
                else:
                    accepted = 10 + trajectory
                    censored = False
                    failure = accepted + 1
                rows.append(
                    {
                        "model": model,
                        "trajectory": str(trajectory),
                        "event": event,
                        "accepted_prefix_calls": accepted,
                        "physical_time": accepted * 0.025,
                        "first_failure_call": failure,
                        "right_censored": censored,
                        "recovered_after_failure": False,
                    }
                )
    return rows


def test_horizon_statistics_preserve_censoring_and_pairing() -> None:
    records = _records()
    summaries = summarize_horizons(records)
    pcno = next(
        row for row in summaries if row["model"] == "pcno" and row["event"] == "finite"
    )
    pcfno = next(
        row for row in summaries if row["model"] == "pcfno" and row["event"] == "finite"
    )
    assert pcno["failed_count"] == 0
    assert pcno["censored_count"] == 30
    assert pcno["median_calls"] == 320
    assert pcfno["failed_count"] == 30
    assert pcfno["median_calls"] == 24.5
    paired, counts = paired_horizons(records)
    assert len(paired) == 90
    assert counts["finite"] == {"pcno_later": 30, "tie": 0, "pcfno_later": 0}


def test_registered_frame_schedule_resolves_h100_and_event_calls() -> None:
    calls = frame_schedule(PRODUCTION_STEPS, [53, 109, 213, None])
    assert calls == sorted(set(calls))
    assert calls[0] == 0 and calls[-1] == 320
    for required in (53, 79, 100, 109, 120, 160, 200, 213, 240, 280, 320):
        assert required in calls
    assert all(call % 2 == 0 for call in calls if 80 <= call <= 140 and call != 109)


def _bundle(model: str) -> dict[str, np.ndarray]:
    x, y = np.meshgrid(np.linspace(0.0, 1.0, 4), np.linspace(0.0, 0.5, 3))
    positions = np.stack((x.reshape(-1), y.reshape(-1)), axis=-1)
    state0 = np.column_stack(
        (
            np.ones(positions.shape[0]),
            np.zeros(positions.shape[0]),
            np.zeros(positions.shape[0]),
            np.full(positions.shape[0], 2.5),
        )
    ).astype(np.float32)
    state1 = state0.copy()
    state1[:, 3] += 0.05 if model == "pcno" else 0.08
    state2 = state1.copy()
    state2[0, 3] = np.nan
    states = np.stack((state0, state1, state2))
    scales = {
        "density_min": 0.8,
        "density_max": 1.2,
        "pressure_min": 0.8,
        "pressure_max": 1.2,
        "internal_energy_min": 2.0,
        "internal_energy_max": 3.0,
        "pressure_increment_abs_max": 0.2,
        "scaled_increment_max": 0.5,
        "multiplier_for_increment_limits": 5.0,
        "quantiles": [0.005, 0.995],
        "source": "synthetic test",
    }
    return {
        "schema": np.asarray(
            PCNO_ANIMATION_SCHEMA if model == "pcno" else PCFNO_ANIMATION_SCHEMA
        ),
        "working_id": np.asarray(
            PCNO_WORKING_ID if model == "pcno" else PCFNO_WORKING_ID
        ),
        "trajectory": np.asarray("187"),
        "checkpoint_sha256": np.asarray(
            PCNO_CHECKPOINT_SHA256 if model == "pcno" else PCFNO_CHECKPOINT_SHA256
        ),
        "requested_horizon": np.asarray(PRODUCTION_STEPS, dtype=np.int64),
        "recorded_call_count": np.asarray(2, dtype=np.int64),
        "termination_call": np.asarray(2, dtype=np.int64),
        "physical_times": np.asarray([0.0, 0.025, 0.05]),
        "positions": positions.astype(np.float32),
        "node_type": np.zeros(positions.shape[0], dtype=np.int64),
        "deployed_states_conservative": states,
        "state_scale": np.ones(4, dtype=np.float64),
        "presentation_scales_json": np.asarray(json.dumps(scales, sort_keys=True)),
        "event_admissible": np.asarray([True, True, False]),
        "event_bounded": np.asarray([True, True, False]),
        "event_finite": np.asarray([True, True, False]),
        "common_amplitude_ratio": np.asarray([1.0, 1.1, np.nan]),
        "common_scaled_rms_ratio": np.asarray([1.0, 1.1, np.nan]),
        "minimum_internal_energy": np.asarray([2.5, 2.55, np.nan]),
    }


def _case() -> dict[str, object]:
    return {
        "events": {
            name: {
                "accepted_prefix_calls": 1,
                "first_failure_call": 2,
                "right_censored": False,
            }
            for name in EVENT_NAMES
        }
    }


def test_tiny_paired_gif_marks_post_termination_frames(tmp_path: Path) -> None:
    bundles = {model: _bundle(model) for model in ("pcno", "pcfno")}
    verify_bundle_pair(bundles)
    output = tmp_path / "paired.gif"
    result = render_comparison_gif(
        bundles,
        {"pcno": _case(), "pcfno": _case()},
        output,
        fps=2,
        dpi=50,
        raster_width=32,
        frame_calls=[0, 1, 2, 3],
    )
    assert output.is_file() and output.stat().st_size > 0
    assert result["rendered_calls"] == [0, 1, 2, 3]
    assert result["pcno_recorded_calls"] == 2
    with Image.open(output) as image:
        assert image.n_frames >= 3
    assert array_sha256(
        bundles["pcno"]["deployed_states_conservative"][0]
    ) == array_sha256(bundles["pcfno"]["deployed_states_conservative"][0])
