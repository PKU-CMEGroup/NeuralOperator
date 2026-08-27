from __future__ import annotations

import copy
from dataclasses import asdict

import pytest

from scripts.time_dependent_no.run_m1_kolmogorov_q1_r2_diagnostic import (
    POPULATION_RETAINED_ARRAY_KEYS,
    POPULATION_SEEDS,
    POPULATION_WALL_TIME_CAP_SECONDS,
    _full_contract,
    _population_contract,
    compare_shared_spatial_rows,
    summarize_population_qualification,
    summarize_temporal_rows,
    verify_current_spatial_parent_replay,
    verify_parent_input_replay,
    verify_spatial_parent_qualification,
    verify_temporal_parent_qualification,
)


def _spatial_row(error: float = 0.02) -> dict[str, object]:
    return {
        "case": "clean_0",
        "family": "clean",
        "pair": "128_to_256",
        "low_resolution": 128,
        "high_resolution": 256,
        "horizon": 1,
        "state_relative_l2": error,
        "energy_relative_difference": 0.001,
        "enstrophy_relative_difference": 0.002,
        "palinstrophy_relative_difference": 0.003,
        "spectrum_total_variation": 0.004,
        "finite": True,
    }


def test_full_contract_targets_n256_under_both_reference_checks() -> None:
    base, spatial, temporal = _full_contract()

    assert base.resolution == 64
    assert spatial.spatial_resolutions == (128, 256, 512)
    assert spatial.spatial_horizon == 16
    assert spatial.spatial_dt_max == 0.0005
    assert spatial.workers == 3
    assert temporal.spatial_resolution == 256
    assert temporal.path_horizon == 16
    assert temporal.structure_horizon == 64
    assert temporal.time_steps == (0.002, 0.001, 0.0005)


def test_population_contract_matches_frozen_preregistration() -> None:
    config, settings = _population_contract()

    assert config.resolution == 256
    assert config.dt_max == 0.002
    assert POPULATION_SEEDS == (
        2026083101,
        2026083102,
        2026083103,
        2026083104,
    )
    assert settings.total_calls == 5120
    assert settings.capture_calls == (0, 1024, 5120)
    assert settings.burnin_candidates == (1024,)
    assert settings.observation_calls == 4096
    assert settings.block_calls == 512
    assert settings.half_change_maximum == 0.10
    assert settings.drift_spearman_minimum == 0.5
    assert settings.split_rhat_maximum == 1.05
    assert settings.pooled_ess_minimum == 100.0
    assert settings.initial_rms == 4.0
    assert settings.workers == 4
    assert POPULATION_WALL_TIME_CAP_SECONDS == 8 * 60 * 60
    assert "state" not in POPULATION_RETAINED_ARRAY_KEYS


def test_shared_spatial_replay_checks_every_key_and_numeric_value() -> None:
    retained = [_spatial_row()]
    current = copy.deepcopy(retained)

    passed = compare_shared_spatial_rows(current, retained)
    assert passed["pass"]
    assert passed["maximum_absolute_numeric_difference"] == 0.0

    current[0]["state_relative_l2"] = 0.020000000001
    failed = compare_shared_spatial_rows(current, retained)
    assert not failed["pass"]
    assert failed["maximum_absolute_numeric_difference"] > 1.0e-13

    missing = compare_shared_spatial_rows([], retained)
    assert not missing["pass"]
    assert not missing["keys_match"]


def _temporal_rows(path_error: float = 2.0e-4) -> list[dict[str, object]]:
    rows = []
    for family in ("clean", "displaced"):
        for index in range(3):
            rows.append(
                {
                    "case": f"{family}_{index}",
                    "family": family,
                    "dt_max": 0.002,
                    "reference_dt_max": 0.0005,
                    "one_step_relative_l2": 2.0e-6,
                    "path_relative_l2": path_error,
                    "energy_wasserstein": 0.001,
                    "enstrophy_wasserstein": 0.002,
                    "spectrum_total_variation": 0.003,
                    "maximum_projection_relative_l2": 1.0e-15,
                    "finite": True,
                }
            )
    return rows


def test_temporal_summary_keeps_state_structure_and_closure_gates_separate() -> None:
    passed = summarize_temporal_rows(
        _temporal_rows(),
        candidate_dt_max=0.002,
        reference_dt_max=0.0005,
        maximum_projection_relative_l2=1.0e-15,
        all_finite=True,
        canonical_tolerance=1.0e-11,
    )
    assert passed["one_step_pass"]
    assert passed["path_pass"]
    assert passed["structure_pass"]
    assert passed["closure_pass"]
    assert passed["pass"]

    path_failed = summarize_temporal_rows(
        _temporal_rows(path_error=6.0e-3),
        candidate_dt_max=0.002,
        reference_dt_max=0.0005,
        maximum_projection_relative_l2=1.0e-15,
        all_finite=True,
        canonical_tolerance=1.0e-11,
    )
    assert path_failed["one_step_pass"]
    assert not path_failed["path_pass"]
    assert not path_failed["pass"]


def test_parent_input_replay_is_exact_and_fails_closed() -> None:
    parent = {
        "stationarity": {
            "initial_states": [{"seed": 1, "sha256": "initial"}],
            "chosen_post_burnin_states": [{"seed": 1, "sha256": "burnin"}],
        },
        "population": {
            "calibration_inputs": [
                {"case": "clean_0", "sha256": "clean"},
                {"case": "displaced_0", "sha256": "displaced"},
            ]
        },
    }
    outputs = [
        {
            "seed": 1,
            "initial": {"sha256": "initial"},
            "post_burnin": {"sha256": "burnin"},
            "inputs": (
                {"case": "clean_0", "sha256": "clean"},
                {"case": "displaced_0", "sha256": "displaced"},
            ),
        }
    ]

    assert verify_parent_input_replay(parent, outputs)["pass"]
    outputs[0]["inputs"][1]["sha256"] = "wrong"
    with pytest.raises(RuntimeError, match="do not match"):
        verify_parent_input_replay(parent, outputs)


def test_current_spatial_parent_replay_uses_this_attempts_hashes() -> None:
    seed = 2026082601
    parent = {
        "stationarity": {
            "initial_states": [{"seed": seed, "sha256": "initial"}],
            "chosen_post_burnin_states": [{"seed": seed, "sha256": "burnin"}],
            "chosen_burnin_calls": 512,
        },
        "population": {
            "calibration_inputs": [
                {"case": "clean_0", "sha256": "clean"},
                {"case": "displaced_0", "sha256": "displaced"},
            ]
        },
    }
    diagnostic = {
        "initial_rows": [{"seed": seed, "sha256": "initial"}],
        "post_burnin_rows": [{"seed": seed, "sha256": "burnin"}],
        "input_rows": [
            {"case": "clean_0", "sha256": "clean"},
            {"case": "displaced_0", "sha256": "displaced"},
        ],
    }

    assert verify_current_spatial_parent_replay(parent, diagnostic)["pass"]
    diagnostic["input_rows"][0]["sha256"] = "wrong"
    with pytest.raises(RuntimeError, match="do not match"):
        verify_current_spatial_parent_replay(parent, diagnostic)


def test_spatial_parent_qualification_fails_closed() -> None:
    result = {
        "classification": "spatial_candidate_qualified",
        "spatial_diagnostic": {"screen_pass": True},
        "parent_state_replay": {"pass": True},
        "shared_r1_spatial_replay": {"pass": True},
        "source_binding": {"hashes_stable_during_execution": True},
        "data_access": False,
        "checkpoint_access": False,
        "model_access": False,
        "training": False,
        "remote_execution": False,
        "test_access": False,
        "full_state_trajectory_retained": False,
    }

    assert verify_spatial_parent_qualification(result)["pass"]

    wrong_classification = copy.deepcopy(result)
    wrong_classification["classification"] = "spatial_candidate_failed"
    with pytest.raises(RuntimeError, match="classification"):
        verify_spatial_parent_qualification(wrong_classification)

    leaked_access = copy.deepcopy(result)
    leaked_access["data_access"] = True
    with pytest.raises(RuntimeError, match="access_closed"):
        verify_spatial_parent_qualification(leaked_access)


def _qualified_temporal_parent() -> dict[str, object]:
    config, _ = _population_contract()
    return {
        "classification": "reference_candidate_qualified",
        "reference_config": asdict(config),
        "temporal_diagnostic": {"pass_with_repeatability": True},
        "parent_state_replay": {"pass": True},
        "parent_bindings": {
            "r2_spatial": {"qualification": {"pass": True}},
        },
        "source_binding": {"hashes_stable_during_execution": True},
        "data_access": False,
        "checkpoint_access": False,
        "model_access": False,
        "training": False,
        "remote_execution": False,
        "test_access": False,
        "full_state_trajectory_retained": False,
    }


def test_temporal_parent_qualification_fails_closed() -> None:
    result = _qualified_temporal_parent()
    assert verify_temporal_parent_qualification(result)["pass"]

    wrong_grid = copy.deepcopy(result)
    wrong_grid["reference_config"]["resolution"] = 128
    with pytest.raises(RuntimeError, match="reference_contract"):
        verify_temporal_parent_qualification(wrong_grid)

    failed_temporal_gate = copy.deepcopy(result)
    failed_temporal_gate["temporal_diagnostic"][
        "pass_with_repeatability"
    ] = False
    with pytest.raises(RuntimeError, match="temporal_gates"):
        verify_temporal_parent_qualification(failed_temporal_gate)

    leaked_access = copy.deepcopy(result)
    leaked_access["model_access"] = True
    with pytest.raises(RuntimeError, match="access_closed"):
        verify_temporal_parent_qualification(leaked_access)


def _qualified_population_diagnostic() -> dict[str, object]:
    return {
        "candidates": [
            {
                "burnin_calls": 1024,
                "observation_start_call": 1025,
                "observation_end_call": 5120,
                "energy": {"pass": True},
                "enstrophy": {"pass": True},
                "pass": True,
            }
        ],
        "capture_rows": [
            {
                "seed": seed,
                "states": [{"call": call} for call in (0, 1024, 5120)],
            }
            for seed in POPULATION_SEEDS
        ],
        "all_finite": True,
    }


def test_population_qualification_requires_exact_window_and_both_metrics() -> None:
    _, settings = _population_contract()
    diagnostic = _qualified_population_diagnostic()

    passed = summarize_population_qualification(diagnostic, settings)
    assert passed["plumbing_pass"]
    assert passed["pass"]

    failed_enstrophy = copy.deepcopy(diagnostic)
    failed_enstrophy["candidates"][0]["enstrophy"]["pass"] = False
    failed_enstrophy["candidates"][0]["pass"] = False
    failed = summarize_population_qualification(failed_enstrophy, settings)
    assert failed["plumbing_pass"]
    assert not failed["pass"]

    wrong_capture = copy.deepcopy(diagnostic)
    wrong_capture["capture_rows"][0]["states"][-1]["call"] = 5119
    malformed = summarize_population_qualification(wrong_capture, settings)
    assert not malformed["plumbing_pass"]
    assert not malformed["pass"]
