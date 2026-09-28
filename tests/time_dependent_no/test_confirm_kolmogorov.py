import json

import numpy as np
import pytest
import torch

from scripts.time_dependent_no import confirm_kolmogorov as experiment
from scripts.time_dependent_no.evaluate_kolmogorov_recovery import sha256
from utility.time_dependent_no.kolmogorov_reference import (
    KolmogorovReferenceConfig, KolmogorovReferenceStepper,
)
from scripts.time_dependent_no.screen_kolmogorov_trajectory_readiness import _initial


def population(tmp_path):
    seeds = [101, 102]
    manifest = dict(status="completed", role="confirmation", scope_sha256="scope",
                    seeds=seeds, shape=[3, 16, 16], dtype="float64", artifacts={})
    for seed in seeds:
        name = f"reference_{seed}.npy"
        np.save(tmp_path / name, np.full((3, 16, 16), seed, dtype=np.float64))
        manifest["artifacts"][name] = sha256(tmp_path / name)
    (tmp_path / "result.json").write_text(json.dumps(manifest))
    return manifest


def test_population_binding_rejects_role_seed_and_content_changes(tmp_path):
    manifest = population(tmp_path)
    loaded = experiment.load_population(tmp_path, "scope", [101, 102], (3, 16, 16))
    np.testing.assert_array_equal(experiment.reference_case(tmp_path, loaded, 102),
                                  np.full((3, 16, 16), 102, dtype=np.float32))
    for change in ({"role": "development"}, {"seeds": [102, 101]}, {"scope_sha256": "other"}):
        (tmp_path / "result.json").write_text(json.dumps({**manifest, **change}))
        with pytest.raises(ValueError):
            experiment.load_population(tmp_path, "scope", [101, 102], (3, 16, 16))
    (tmp_path / "result.json").write_text(json.dumps(manifest))
    np.save(tmp_path / "reference_101.npy", np.zeros((3, 16, 16)))
    with pytest.raises(ValueError, match="hash"):
        experiment.load_population(tmp_path, "scope", [101, 102], (3, 16, 16))


def test_reference_keeps_exact_initial_law_and_macro_steps(tmp_path):
    config = KolmogorovReferenceConfig(resolution=16, viscosity=.01, macro_dt=.002)
    record = experiment.generate_case(tmp_path, 101, config, 2)
    saved = np.load(tmp_path / record["file"])
    stepper = KolmogorovReferenceStepper(config)
    start = _initial(stepper, 101)
    first = stepper.advance_canonical(start).state
    np.testing.assert_array_equal(saved[0], start)
    np.testing.assert_array_equal(saved[1], first)
    np.testing.assert_array_equal(saved[2], stepper.advance_canonical(first).state)
    assert saved.dtype == np.float64
    assert record["steps"] == 2
    assert record["sha256"] == sha256(tmp_path / record["file"])
    assert len(record["diagnostics"]) == 3


def case(seed, error, complete=True):
    return dict(seed=seed, horizons=[dict(horizon=32, complete=complete,
                sse=error**2 if complete else None, target_sse=1. if complete else None,
                relative_l2=error if complete else None)])


def test_paired_effect_aligns_identities_and_resamples_trajectories():
    clean = [[case(101, 2.), case(102, 4.)] for _ in range(3)]
    preserve = [[case(102, 2.), case(101, 1.)] for _ in range(3)]
    result = experiment.paired_effect(clean, preserve, [101, 102])
    assert result["independent_trajectories"] == 2
    assert result["paired_fits"] == 3
    assert result["geometric_mean_ratio"] == pytest.approx(.5)
    assert result["bootstrap_log_interval"] == pytest.approx([np.log(.5)]*2)
    assert result["per_fit_win_counts"] == [2, 2, 2]
    assert result["per_fit_pooled_ratios"] == pytest.approx([.5]*3)


def test_guarded_path_is_not_dropped_from_unconditional_contrast():
    clean = [[case(101, 2.), case(102, 4.)] for _ in range(3)]
    preserve = [[case(101, 1.), case(102, 2.)] for _ in range(3)]
    preserve[1][1] = case(102, 2., complete=False)
    result = experiment.paired_effect(clean, preserve, [101, 102])
    assert result["geometric_mean_ratio"] is None
    assert result["bootstrap_log_interval"] is None
    assert result["complete_pairs"] == [2, 1, 2]


def test_saved_forecast_reduction_excludes_initial_state_and_keeps_controls():
    baseline = np.ones((3, 1), dtype=np.float32)
    arrays = dict(baseline=baseline, reference=baseline,
                  displacement=np.array([[0.], [3.], [4.]], dtype=np.float32),
                  identity_displacement=np.array([[0.], [1.], [2.]], dtype=np.float32),
                  map_forcing=np.ones((2, 1), dtype=np.float32),
                  curvature_estimate=np.zeros_like(baseline),
                  linear_response=np.ones((2, 1), dtype=np.float32),
                  plus_remainder=np.zeros((2, 1), dtype=np.float32),
                  minus_remainder=np.zeros((2, 1), dtype=np.float32),
                  clean_forcing=np.array([[.1], [.2]], dtype=np.float32),
                  ad_no_grad_offset=np.zeros((2, 1), dtype=np.float32))
    row = experiment.saved_forecast_metrics(arrays, 1.)
    assert row["predicted_sse"] == 25.
    assert row["identity_sse"] == 5.
    assert row["target_sse"] == 2.
    assert row["clean_sse"] == pytest.approx(.05)
    assert row["max_remainder_over_linear"] == 0.
    arrays["minus_remainder"][1] = .2
    assert experiment.saved_forecast_metrics(arrays, 1.)["max_remainder_over_linear"] == pytest.approx(.2)


def test_outcomes_require_completed_scope_bound_forecast_freeze(tmp_path):
    with pytest.raises(FileNotFoundError):
        experiment.require_freeze(tmp_path, "scope")
    path = tmp_path / "prediction_freeze.json"
    record = dict(status="completed", scope_sha256="scope", candidate_outcomes_started=False,
                  frozen_unix=1., files={"forecast.json": "wrong"})
    (tmp_path / "forecast.json").write_text("{}")
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="hash"):
        experiment.require_freeze(tmp_path, "scope")
    record["files"]["forecast.json"] = sha256(tmp_path / "forecast.json")
    path.write_text(json.dumps(record))
    assert experiment.require_freeze(tmp_path, "scope")["frozen_unix"] == 1.
    with pytest.raises(ValueError, match="scope"):
        experiment.require_freeze(tmp_path, "other")


def test_freeze_reduces_saved_forecasts_before_outcomes(tmp_path):
    baseline = torch.ones((9, 2), dtype=torch.float32)
    arrays = experiment.forecast.forecast_on_baseline(lambda n, x: 1.01*x + .02,
                                                     baseline, baseline)
    metrics = experiment.forecast.case_metrics(baseline, baseline, arrays, 1.)
    scope = dict(seeds=[101], clean_calibration_gain=3.)
    for key in experiment.CANDIDATES:
        directory = tmp_path/"forecast"/key
        directory.mkdir(parents=True)
        digest = experiment.save_arrays(directory/"forecast_101.npz",
                                        dict(baseline=baseline, reference=baseline, **arrays))
        experiment.write_json(directory/"result.json", dict(status="completed", train_scale=1.,
            cases=[dict(seed=101, status="completed", **metrics)],
            artifacts={"forecast_101.npz": digest}))
    for key in (*experiment.CANDIDATES, "dynamics"):
        directory = tmp_path/"clean"/key
        directory.mkdir(parents=True)
        experiment.write_json(directory/"result.json", dict(bands=[dict(relative_l2=.03)]))
    for directory in (tmp_path/"reference", tmp_path/"baseline/dynamics"):
        directory.mkdir(parents=True)
        experiment.write_json(directory/"result.json", {})
    frozen = experiment.freeze_predictions(scope, "scope", tmp_path)
    assert experiment.require_freeze(tmp_path, "scope") == frozen
    prediction = frozen["predictions"][experiment.CANDIDATES[0]]
    assert prediction["predicted_relative_l2"] == pytest.approx(
        np.sqrt(metrics["predicted_sse"]/metrics["target_sse"]))
    assert prediction["calibrated_clean_relative_l2"] == pytest.approx(.09)
    assert prediction["clean_only_recommendation"] is False
    (tmp_path/"outcomes").mkdir()
    with pytest.raises(ValueError, match="outcomes"):
        experiment.freeze_predictions(scope, "scope", tmp_path)


def test_historical_loader_preserves_weights_and_rejects_changed_dynamics(tmp_path, monkeypatch):
    current, historical, fitted = [tmp_path/n for n in ("current", "historical", "fit")]
    fitter = "scripts/time_dependent_no/fit_kolmogorov_response_preserving.py"
    dynamics = "utility/time_dependent_no/pcno_kolmogorov.py"
    for directory in (current, historical):
        (directory/fitter).parent.mkdir(parents=True)
        (directory/dynamics).parent.mkdir(parents=True)
        (directory/dynamics).write_text("unchanged inference")
    (current/fitter).write_text("new seed-capable fitting interface")
    (historical/fitter).write_text(
        "def load_fitted(parent, manifest, digest, path, device):\n"
        "    return (path/'terminal.pt').read_bytes(), {}, {}\n")
    fitted.mkdir()
    (fitted/"terminal.pt").write_bytes(b"frozen model parameters")
    experiment.write_json(fitted/"result.json", dict(sources={
        p: sha256(historical/p) for p in (fitter, dynamics)}))
    monkeypatch.setattr(experiment, "ROOT", current)
    original_root = experiment.evaluation.training.ROOT
    loaded, _, _ = experiment.load_seed17({}, {}, "scope", fitted, historical, "cpu")
    assert loaded == b"frozen model parameters"
    assert experiment.evaluation.training.ROOT == original_root
    (current/dynamics).write_text("changed inference")
    with pytest.raises(ValueError, match="inference"):
        experiment.load_seed17({}, {}, "scope", fitted, historical, "cpu")
