from __future__ import annotations

import copy

import pytest

from scripts.time_dependent_no.run_m1_kolmogorov_q1_r2_diagnostic import (
    _full_contract,
    compare_shared_spatial_rows,
    summarize_temporal_rows,
    verify_current_spatial_parent_replay,
    verify_parent_input_replay,
    verify_spatial_parent_qualification,
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
