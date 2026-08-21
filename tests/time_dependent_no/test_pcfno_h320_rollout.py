from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.time_dependent_no.evaluate_pcfno_h320_rollout import (
    ANIMATION_SCHEMA,
    CHECKPOINT_SHA256,
    SOURCE_PATHS,
    TRUTH_STEPS,
    WORKING_ID,
    StabilityThresholds,
    array_sha256,
    build_final_hash_manifest,
    common_envelope_metrics,
    conservative_to_primitive_numpy,
    diagnose_euler_state,
    event_summary,
    expected_prefix,
    finite_max,
    internal_energy_numpy,
    invalid_mask_numpy,
    nan_if_none,
    parse_args,
    sha256_file,
    stage_transition_label,
    survival_rows,
    write_npz_atomic,
)
from scripts.time_dependent_no.visualize_pcfno_h320_rollout import (
    animation_fields,
    invalid_categories,
    load_bundle,
    render_animation,
)


def conservative_state(rho: float = 1.0, pressure: float = 1.0) -> np.ndarray:
    state = np.zeros((4, 4), dtype=np.float32)
    state[:, 0] = rho
    state[:, 3] = pressure / 0.4
    return state


def animation_bundle(frame_count: int = 3) -> dict[str, np.ndarray]:
    states = np.stack(
        [
            conservative_state(1.0 + 0.05 * frame, 1.0 + 0.1 * frame)
            for frame in range(frame_count)
        ]
    )
    positions = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32
    )
    scales = {
        "density_min": 0.9,
        "density_max": 1.2,
        "pressure_min": 0.8,
        "pressure_max": 1.3,
        "internal_energy_min": 2.0,
        "internal_energy_max": 3.2,
        "pressure_increment_abs_max": 0.2,
        "scaled_increment_max": 0.3,
    }
    return {
        "schema": np.asarray(ANIMATION_SCHEMA),
        "working_id": np.asarray(WORKING_ID),
        "trajectory": np.asarray("187"),
        "checkpoint_sha256": np.asarray(CHECKPOINT_SHA256),
        "requested_horizon": np.asarray(frame_count - 1, dtype=np.int64),
        "recorded_call_count": np.asarray(frame_count - 1, dtype=np.int64),
        "termination_call": np.asarray(-1, dtype=np.int64),
        "physical_times": np.arange(frame_count, dtype=np.float64) * 0.01,
        "positions": positions,
        "node_type": np.zeros(4, dtype=np.int64),
        "deployed_states_conservative": states,
        "state_scale": np.ones(4, dtype=np.float64),
        "presentation_scales_json": np.asarray(json.dumps(scales, sort_keys=True)),
        "event_admissible": np.ones(frame_count, dtype=np.bool_),
        "event_bounded": np.ones(frame_count, dtype=np.bool_),
        "event_finite": np.ones(frame_count, dtype=np.bool_),
        "common_amplitude_ratio": np.ones(frame_count, dtype=np.float64),
        "common_scaled_rms_ratio": np.ones(frame_count, dtype=np.float64),
        "minimum_internal_energy": np.ones(frame_count, dtype=np.float64),
    }


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
                "--strict-rollout-dir",
                str(tmp_path),
                "--continuation-dir",
                str(tmp_path),
                "--output-dir",
                str(output),
            ]
        )


def test_source_scope_is_exact_and_isolated() -> None:
    assert SOURCE_PATHS[:4] == (
        "docs/time_dependent_no/W26_L1_PCFNO_H320_PREREGISTRATION.md",
        "scripts/time_dependent_no/evaluate_pcfno_h320_rollout.py",
        "scripts/time_dependent_no/visualize_pcfno_h320_rollout.py",
        "tests/time_dependent_no/test_pcfno_h320_rollout.py",
    )
    assert "pcno/pcno.py" not in SOURCE_PATHS


def test_compatibility_copies_match_maintained_d087_locally(tmp_path: Path) -> None:
    try:
        from scripts.time_dependent_no.evaluate_pcno_long_horizon_stability import (
            StabilityThresholds as D087Thresholds,
        )
        from scripts.time_dependent_no.evaluate_pcno_long_horizon_stability import (
            accepted_prefix_event as d087_accepted_prefix_event,
        )
        from scripts.time_dependent_no.evaluate_pcno_long_horizon_stability import (
            build_final_hash_manifest as d087_build_manifest,
        )
        from scripts.time_dependent_no.evaluate_pcno_long_horizon_stability import (
            diagnose_euler_state as d087_diagnose_euler_state,
        )
        from utility.time_dependent_no.pcno_inadmissibility import (
            stage_transition_label as maintained_stage_transition_label,
        )
    except ImportError as exc:
        pytest.skip(f"maintained semantic references are unavailable: {exc}")

    assert StabilityThresholds().as_dict() == D087Thresholds().as_dict()
    for name in ("admissible", "bounded", "finite"):
        for values in ([True, True], [True, False], [False, True], [True]):
            assert event_summary(
                {event: values for event in ("admissible", "bounded", "finite")}, 2
            )[name] == d087_accepted_prefix_event(
                values, requested_horizon=2, event_name=name
            )
    for before in ("admissible", "nonpositive_internal_energy"):
        for after in ("admissible", "nonpositive_internal_energy"):
            assert stage_transition_label(
                before, after, stage="model"
            ) == maintained_stage_transition_label(before, after, stage="model")

    state = torch.tensor(conservative_state()[None])
    ours = diagnose_euler_state(
        state, representation="conservative_rho_m1_m2_E", gamma=1.4
    )
    maintained = d087_diagnose_euler_state(
        state, representation="conservative_rho_m1_m2_E", gamma=1.4
    )
    assert ours == maintained

    ours_root = tmp_path / "ours"
    maintained_root = tmp_path / "maintained"
    ours_root.mkdir()
    maintained_root.mkdir()
    (ours_root / "result.json").write_text("{}\n", encoding="utf-8")
    (maintained_root / "result.json").write_text("{}\n", encoding="utf-8")
    assert build_final_hash_manifest(ours_root, ["result.json"]) == d087_build_manifest(
        maintained_root, ["result.json"]
    )


def test_array_hash_binds_dtype_shape_and_values() -> None:
    value = np.arange(12, dtype=np.float32).reshape(3, 4)
    assert array_sha256(value) == array_sha256(np.asfortranarray(value))
    assert array_sha256(value) != array_sha256(value.astype(np.float64))
    assert array_sha256(value) != array_sha256(value.reshape(4, 3))


def test_primitive_internal_energy_and_invalid_categories() -> None:
    states = np.stack(
        [
            conservative_state(),
            conservative_state(rho=-1.0),
            conservative_state(pressure=-1.0),
            conservative_state(),
        ]
    )
    states[3, 0, 0] = np.nan
    primitive = conservative_to_primitive_numpy(states[0], gamma=1.4)
    assert np.allclose(primitive[:, (0, 3)], 1.0)
    assert np.allclose(internal_energy_numpy(states[0]), 2.5)
    categories = invalid_categories(states)
    assert np.all(categories[0] == 0)
    assert np.all(categories[1] == 1)
    assert np.all(categories[2] == 2)
    assert categories[3, 0] == 3
    assert np.array_equal(invalid_mask_numpy(states[2], gamma=1.4), categories[2] > 0)


def test_common_envelope_ratios_and_nonfinite_path() -> None:
    state = conservative_state()
    weights = np.ones(4)
    scale = np.ones(4)
    primitive = conservative_to_primitive_numpy(state, gamma=1.4)
    amplitude = float(np.max(np.abs(primitive)))
    rms = float(np.sqrt(np.sum(np.square(primitive)) / 4.0))
    envelope = {
        "maximum_abs_scaled_primitive": amplitude,
        "maximum_proxy_scaled_rms": rms,
    }
    assert common_envelope_metrics(
        state, weights=weights, component_scale=scale, envelope=envelope, gamma=1.4
    ) == pytest.approx((1.0, 1.0))
    nonfinite = state.copy()
    nonfinite[0, 0] = np.nan
    assert common_envelope_metrics(
        nonfinite,
        weights=weights,
        component_scale=scale,
        envelope=envelope,
        gamma=1.4,
    ) == (None, None)


def test_event_summary_retains_first_failure_and_recovery() -> None:
    events = event_summary(
        {
            "admissible": [True, False, True],
            "bounded": [True, True, True],
            "finite": [True, True, False],
        },
        horizon=3,
    )
    assert events["admissible"]["first_failure_call"] == 2
    assert events["admissible"]["recovered_after_failure"] is True
    assert events["bounded"]["censor_reason"] == "requested_horizon"
    rows = survival_rows([{"events": events}], horizon=3)
    admissible = [row for row in rows if row["event"] == "admissible"]
    assert admissible[1]["failure_count"] == 1
    assert admissible[1]["kaplan_meier_survival"] == 0.0


def test_nonfinite_safe_summary_helpers() -> None:
    assert finite_max([None, float("nan")]) is None
    assert finite_max([None, 2.0, 1.0]) == 2.0
    assert math.isnan(nan_if_none(None))
    assert nan_if_none(3.0) == 3.0


def test_expected_strict_prefix_is_h0_through_h79(tmp_path: Path) -> None:
    strict = tmp_path / "strict"
    continuation = tmp_path / "continuation"
    (strict / "trajectories").mkdir(parents=True)
    continuation.mkdir()
    initial = conservative_state()
    predictions = np.repeat(initial[None], TRUTH_STEPS, axis=0)
    np.savez_compressed(
        strict / "trajectories" / "trajectory_187.npz",
        initial_conservative=initial,
        pcno_baseline_predictions_conservative=predictions,
        baseline_valid_length=np.asarray(TRUTH_STEPS),
        checkpoint_sha256=np.asarray(CHECKPOINT_SHA256),
    )
    prefix, record = expected_prefix(
        "187",
        strict_root=strict,
        continuation_root=continuation,
        continuation_output_sha256={},
    )
    assert prefix.shape == (TRUTH_STEPS + 1, 4, 4)
    assert np.array_equal(prefix[0], initial)
    assert record["kind"] == "strict_h79_survivor"


def test_animation_bundle_contains_no_reference_array(tmp_path: Path) -> None:
    path = tmp_path / "trajectory_187.npz"
    np.savez_compressed(path, **animation_bundle())
    arrays = load_bundle(path)
    assert not any("reference" in name for name in arrays)
    fields = animation_fields(arrays)
    assert set(fields) == {
        "density",
        "pressure",
        "internal_energy",
        "pressure_increment",
        "scaled_increment",
        "invalid_category",
    }
    assert fields["pressure"].shape == (3, 4)
    assert np.all(fields["pressure_increment"][0] == 0.0)


def test_atomic_uncompressed_bundle_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "trajectory_187.npz"
    expected = animation_bundle()
    write_npz_atomic(path, expected)
    observed = load_bundle(path)
    assert path.is_file()
    assert not path.with_suffix(".npz.tmp").exists()
    assert set(observed) == set(expected)
    for name in expected:
        assert np.array_equal(observed[name], expected[name])


def test_tiny_all_frame_animation_smoke(tmp_path: Path) -> None:
    arrays = animation_bundle()
    output = tmp_path / "tiny.gif"
    record = render_animation(arrays, output, fps=2, dpi=50, raster_width=256)
    assert output.is_file() and output.stat().st_size > 0
    assert record["rendered_frame_count"] == 3
    assert record["all_retained_frames_rendered"] is True
    assert record["sha256"] == sha256_file(output)
