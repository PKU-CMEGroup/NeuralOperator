from __future__ import annotations

import numpy as np
import pytest
import torch

from utility.time_dependent_no import corrective_objectives as objectives


class ScalarMap(torch.nn.Module):
    def __init__(self, slope=.7, scale=2.):
        super().__init__()
        self.slope = torch.nn.Parameter(torch.tensor(slope, dtype=torch.float64))
        self.register_buffer("train_scale", torch.tensor(scale, dtype=torch.float64))
        self.inputs = []

    def forward(self, x):
        self.inputs.append(x.detach().clone())
        return {"next_state": self.slope * x}


def test_sequence_modes_share_forward_values_but_have_distinct_gradients():
    sequence = torch.tensor([1., .9, .8, .7, .6], dtype=torch.float64).reshape(1, 5, 1, 1)
    fixed = torch.tensor([1., .7, .49, .343], dtype=torch.float64).reshape(1, 4, 1, 1)
    # Hand differentiation of mean_j ((w*x_j-y_j)/2)^2, w=0.7.
    expected = {"fixed": -.0894219625, "detached": -.0894219625,
                "attached": -.2065716, "clean": -.06775}
    results = {}
    for mode, gradient in expected.items():
        model = ScalarMap()
        row = objectives.sequence_update(model, torch.optim.SGD(model.parameters(), lr=0),
                                         sequence, mode, fixed_inputs=fixed)
        assert float(model.slope.grad) == pytest.approx(gradient, abs=1e-12)
        results[mode] = row["step_mse_scaled"]
    np.testing.assert_allclose(results["fixed"], results["detached"], atol=1e-16)
    np.testing.assert_allclose(results["attached"], results["detached"], atol=1e-16)


def test_fixed_generator_is_not_refreshed_as_the_trained_model_changes():
    sequence = torch.tensor([1., .9, .8, .7, .6], dtype=torch.float64).reshape(1, 5, 1, 1)
    fixed = torch.tensor([1., .7, .49, .343], dtype=torch.float64).reshape(1, 4, 1, 1)
    model = ScalarMap(slope=.5)
    objectives.sequence_update(model, torch.optim.SGD(model.parameters(), lr=0), sequence,
                               "fixed", fixed_inputs=fixed.requires_grad_())
    assert [float(x) for x in model.inputs] == [1., .7, .49, .343]
    assert fixed.grad is None
    model.inputs.clear()
    objectives.sequence_update(model, torch.optim.SGD(model.parameters(), lr=0), sequence, "detached")
    assert [float(x) for x in model.inputs] == [1., .5, .25, .125]


def test_mixed_target_has_the_weighted_objectives_gradient():
    rec = torch.tensor([1., 3.], dtype=torch.float64)
    dyn = torch.tensor([5., -1.], dtype=torch.float64)
    target = objectives.mixed_target(rec, dyn, .25)
    torch.testing.assert_close(target, torch.tensor([2., 2.], dtype=torch.float64))
    prediction = torch.tensor([.5, -.5], dtype=torch.float64, requires_grad=True)
    weighted = (.75*(prediction-rec).square()+.25*(prediction-dyn).square()).mean()
    centered = (prediction-target).square().mean()
    assert float((weighted-centered).detach()) == pytest.approx(3.)
    g1, = torch.autograd.grad(weighted, prediction)
    g2, = torch.autograd.grad(centered, prediction)
    torch.testing.assert_close(g1, g2, rtol=0, atol=0)
    torch.testing.assert_close(objectives.mixed_target(rec, dyn, 0), rec)
    torch.testing.assert_close(objectives.mixed_target(rec, dyn, 1), dyn)
    with pytest.raises(ValueError): objectives.mixed_target(rec, dyn, 1.1)


def test_windows_include_onset_and_last_target_without_crossing_trajectories():
    rows = objectives.sample_windows(2, 9, 4, per_path=4, seed=31)
    assert rows == objectives.sample_windows(2, 9, 4, per_path=4, seed=31)
    for p in range(2):
        starts = [r["start"] for r in rows if r["path_index"] == p]
        assert len(set(starts)) == 4 and min(starts) == 0 and max(starts) == 4
    train = np.arange(18, dtype=np.float32).reshape(2, 9, 1, 1)
    selection = [i for i, r in enumerate(rows) if r["start"] == 4]
    batch = objectives.sequence_batch(train, rows, selection, 4)
    np.testing.assert_array_equal(batch[:, :, 0, 0], [[4, 5, 6, 7, 8], [13, 14, 15, 16, 17]])
    with pytest.raises(ValueError): objectives.sample_windows(2, 5, 4, per_path=4)


def test_state_gradients_measure_conflict_and_leave_parameter_gradients_untouched():
    model = ScalarMap(slope=.8, scale=1.)
    model.slope.grad = torch.tensor(19., dtype=torch.float64)
    sequence = torch.tensor([1., .7, 1.], dtype=torch.float64).reshape(1, 3, 1, 1)
    values = objectives.gradient_diagnostics(model, sequence)
    np.testing.assert_allclose(values["state_direct_rms"], [[.1, .36]], atol=1e-14)
    np.testing.assert_allclose(values["state_future_rms"], [[.288, 0.]], atol=1e-14)
    assert values["state_cosine"][0, 0] == pytest.approx(-1.)
    assert np.isnan(values["state_cosine"][0, 1])
    assert values["parameter_attached_l2"] == pytest.approx(.476)
    assert values["parameter_detached_l2"] == pytest.approx(.188)
    assert float(model.slope.grad) == 19.


def test_nonfinite_sequence_cannot_change_weights():
    model = ScalarMap()
    before = model.slope.detach().clone()
    sequence = torch.ones(1, 5, 1, 1, dtype=torch.float64)
    sequence[:, 3] = float("nan")
    with pytest.raises((ValueError, RuntimeError)):
        objectives.sequence_update(model, torch.optim.SGD(model.parameters(), lr=.1), sequence, "attached")
    torch.testing.assert_close(model.slope, before)


def test_empirical_geometry_distinguishes_subspace_residual_from_nearest_sample():
    bank = np.array([[0., 0.], [1., 0.], [2., 0.]])
    queries = np.array([[1., 3.], [4., 0.]])
    values = objectives.empirical_geometry(bank, queries, scale=2., ranks=(1,))
    np.testing.assert_allclose(values["nearest_mse"], [1.125, .5])
    np.testing.assert_allclose(values["pca_mse_r1"], [1.125, 0.], atol=1e-14)
    mean, basis = objectives.pca_basis(bank, 2)
    assert basis.shape == (1, 2)  # Rank-deficient data do not create fictitious directions.
    np.testing.assert_allclose(mean, [1., 0.])
    np.testing.assert_allclose(basis@basis.T, [[1.]], atol=1e-14)


def test_response_statistics_compare_vectors_not_only_gain():
    displacement = np.array([1., -1.])
    values = objectives.response_statistics(-.5*displacement, 2*displacement, displacement, 2.)
    assert values["learned_gain"] == pytest.approx(.5)
    assert values["true_gain"] == pytest.approx(2.)
    assert values["response_error_gain"] == pytest.approx(2.5)
    assert values["cosine"] == pytest.approx(-1.)
    with pytest.raises(ValueError): objectives.response_statistics(displacement, displacement, displacement*0, 1.)


@pytest.fixture
def small_study(tmp_path, monkeypatch):
    import types
    from scripts.time_dependent_no import run_kolmogorov_multistep_study as runner
    monkeypatch.setattr(runner, "WINDOWS_PER_PATH", 4)
    monkeypatch.setattr(runner, "UPDATES", 2)
    monkeypatch.setattr(runner, "RESOURCE_UPDATES", 2)
    study = object.__new__(runner.Study)
    study.out = tmp_path; study.device = "cpu"; study.scale = 2.
    study.train = np.arange(2*9*2*2, dtype=np.float32).reshape(2, 9, 2, 2)/100
    study.train_hash = "fixture"; study.sources = {}; study.registry = {}
    study.context = types.SimpleNamespace(old_model=lambda _: (ScalarMap(), {"fixture": True}))
    study.prepared_hash = "prepared-fixture"
    center = study.train.reshape(-1, 2, 2)[:8].copy()
    study.arrays = {"centers": center, "clean_targets": center*1.1,
                    "native_mixed_inputs": center+.1, "native_mixed_dynamics": (center+.1)*1.1}
    study.prepare_windows()
    return study


def test_training_controls_use_same_windows_and_update_once_per_batch(small_study):
    import json
    study = small_study
    frozen_before = study.fixed_inputs.copy()
    tapes, starts = [], []
    for mode in objectives.SEQUENCE_MODES:
        result = study.train_model(mode+"4")
        assert result["status"] == "completed" and result["updates_completed"] == 2
        path = study.out/"training"/(mode+"4")
        rows = [json.loads(s) for s in (path/"updates.jsonl").read_text().splitlines()]
        tapes.append([r["sequence_indices"] for r in rows])
        starts.append(rows[0]["step_mse_scaled"])
        saved = torch.load(path/"checkpoint_000002.pt", weights_only=False)
        assert all(float(s["step"]) == 2 for s in saved["optimizer"]["state"].values())
        assert result["identity"]["window_manifest_sha256"] == study.window_hash
    assert all(t == tapes[0] for t in tapes[1:])
    np.testing.assert_allclose(starts[1:], [starts[1]]*3)
    np.testing.assert_array_equal(study.fixed_inputs, frozen_before)


def test_midpoint_records_the_weighted_loss_constant(small_study):
    import json
    result = small_study.train_model("mixed_midpoint")
    assert result["identity"]["dynamics_weight"] == .5
    path = small_study.out/"training/mixed_midpoint"
    rows = [json.loads(s) for s in (path/"updates.jsonl").read_text().splitlines()]
    for row in rows:
        ids = row["displaced_indices"]
        gap = small_study.arrays["native_mixed_dynamics"][ids]-small_study.arrays["clean_targets"][ids]
        expected = .125*np.mean((gap/small_study.scale)**2)
        assert row["weighted_target_loss"]-row["loss"] == pytest.approx(expected, rel=1e-5)


def test_resource_check_does_not_create_a_training_checkpoint(small_study, monkeypatch):
    measure = small_study.measure_gradients
    existing_gradients = []
    def inspect(model, out, position):
        existing_gradients.append(any(p.grad is not None for p in model.parameters()))
        return measure(model, out, position)
    monkeypatch.setattr(small_study, "measure_gradients", inspect)
    result = small_study.train_model("attached4", resource=True)
    assert result["status"] == "completed" and result["within_budget"]
    assert existing_gradients == [False, True]
    assert not small_study.registry
    assert not list((small_study.out/"resource").rglob("*.pt"))


def test_failed_training_is_recorded_and_cannot_overwrite_outputs(small_study, monkeypatch):
    import json
    original = objectives.sequence_update
    calls = 0
    def failure(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2: raise RuntimeError("fixture failure")
        return original(*args, **kwargs)
    monkeypatch.setattr(objectives, "sequence_update", failure)
    with pytest.raises(RuntimeError, match="fixture failure"):
        small_study.train_model("detached4")
    path = small_study.out/"training/detached4"
    result = json.loads((path/"result.json").read_text())
    assert result["status"] == "failed" and result["updates_completed"] == 1
    assert not small_study.registry
    with pytest.raises(FileExistsError): small_study.train_model("detached4")


def test_geometry_excludes_the_evaluated_training_trajectory():
    from scripts.time_dependent_no import diagnose_kolmogorov_multistep as diagnostic
    train = np.arange(4*3*2).reshape(4, 3, 1, 2)
    np.testing.assert_array_equal(diagnostic.training_bank(train, 1, 2), train[[0, 2, 3], 2].reshape(3, 2))
    np.testing.assert_array_equal(diagnostic.training_bank(train, None, 2), train[:, 2].reshape(4, 2))


def test_response_queries_keep_direction_amplitude_and_signed_vector_comparisons():
    from scripts.time_dependent_no import diagnose_kolmogorov_multistep as diagnostic
    center, error = np.array([[1., 2.]]), np.array([[3., 4.]])
    inputs, requests, deltas, unresolved = diagnostic.response_queries(center, error, np.array([[1., 0.]]), 2.)
    assert not unresolved and len(requests) == 9
    np.testing.assert_allclose(deltas[0], error)
    np.testing.assert_allclose(deltas[3], [[3., 0.]])
    np.testing.assert_allclose(deltas[6], [[0., 4.]])
    values = diagnostic.score_responses(-inputs, 2*inputs, 2*inputs, requests, deltas, 2.)
    np.testing.assert_allclose(values["learned_gain"], 1., atol=3e-5)
    np.testing.assert_allclose(values["true_gain"], 2., atol=3e-5)
    np.testing.assert_allclose(values["cosine"], -1., atol=1e-10)
    np.testing.assert_allclose(values["response_error_gain"], 3., atol=5e-5)
    np.testing.assert_allclose(values["solver_refinement_relative"], 0.)
    for row, displacement in zip(requests, deltas):
        if row["kind"] == "central":
            assert np.sqrt(np.mean(displacement**2))/2 == pytest.approx(row["amplitude"])
    _, remaining, _, unresolved = diagnostic.response_queries(center, np.array([[3., 0.]]), np.array([[1., 0.]]), 2.)
    assert unresolved == ["pca_orthogonal"] and len(remaining) == 6
    inputs, remaining, deltas, unresolved = diagnostic.response_queries(center, error*0, np.array([[1., 0.]]), 2.)
    assert len(unresolved) == 3 and deltas.shape == (0, 1, 2)
    values = diagnostic.score_responses(inputs, inputs, inputs, remaining, deltas, 2.)
    assert values["cosine"].shape == (0,)


def test_reused_evaluations_must_have_the_current_probe(small_study):
    import json
    from scripts.time_dependent_no import run_kolmogorov_multistep_study as runner
    study = small_study
    old = study.out/"old"
    study.context.out = old
    study.context.model = study.context.old_model
    source = old/"evaluation"
    (source/"pretrained").mkdir(parents=True)
    np.savez_compressed(source/"ground_truth.npz", ground_truth=np.zeros((12, 33, 2, 2)))
    meta = dict(status="completed", training_manifest_sha256=study.train_hash,
        horizon=32, train_scale=study.scale, populations=["train"]*8+["validation"]*4,
        case_ids=[f"train_{s}" for s in runner.base.TRAIN_SEEDS[:8]]+
                 [f"validation_{s}" for s in runner.followup.diagnostic.DEVELOPMENT_SEEDS],
        ground_truth_sha256=runner.base.sha256(source/"ground_truth.npz"))
    (source/"result.json").write_text(json.dumps(meta))
    (source/"pretrained/result.json").write_text(json.dumps(dict(status="completed", identity={"fixture": True},
        probe_name="seed29_pretrained", probe_identity={"different": True}, artifacts={})))
    with pytest.raises(ValueError, match="probe"):
        study.evaluation_data()


def test_plots_pool_squared_error_and_preserve_separate_populations(tmp_path, monkeypatch):
    import csv
    import hashlib
    import json
    from scripts.time_dependent_no import plot_kolmogorov_multistep_study as figures
    current_curve = figures.curve
    # The frozen workstation helper predates the optional omit_first keyword.
    def deployed_curve(ax, x, values, mask, label, color, style="-", paths=False):
        return current_curve(ax, x, values, mask, label, color, style, paths=paths)
    monkeypatch.setattr(figures, "curve", deployed_curve)
    root = tmp_path/"evaluation"
    path = root/"mixed_midpoint"
    path.mkdir(parents=True)
    values = np.tile(np.array([1., 9., 16.])[:, None], (1, 32))
    physical = np.ones((3, 33, 4))
    np.savez_compressed(path/"metrics.npz", clean_mse=values, own_defect_mse=values,
        probe_defect_mse=values, rollout_mse=np.pad(values, ((0, 0), (1, 0))),
        propagated_mse=values, cross_term=values*0, physical_values=physical)
    sha = hashlib.sha256((path/"metrics.npz").read_bytes()).hexdigest()
    (path/"result.json").write_text(json.dumps(dict(artifacts={"metrics.npz": sha},
                                                  physical_names=["a", "b", "c", "d"])))
    np.savez_compressed(root/"ground_truth.npz", physical_truth=physical)
    (root/"result.json").write_text(json.dumps(dict(completed_methods=["mixed_midpoint"],
        populations=["train", "train", "validation"], horizon=32, macro_dt=.05)))
    figures.plot(tmp_path, tmp_path/"figures")
    with (tmp_path/"figures/outcomes.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert float(rows[0]["rollout_H32"]) == pytest.approx(np.sqrt(5.))
    assert float(rows[1]["rollout_H32"]) == pytest.approx(4.)
    assert (tmp_path/"figures/midpoint_errors_validation.pdf").read_bytes().startswith(b"%PDF")
    assert (tmp_path/"figures/index.html").exists()


def test_midpoint_summary_separates_pooled_ratios_from_case_wins(tmp_path):
    import csv
    import json
    from scripts.time_dependent_no import plot_kolmogorov_multistep_study as figures
    path = tmp_path/"midpoint_analysis"
    path.mkdir()
    output = tmp_path/"figures"
    output.mkdir()
    names = ("native_mixed_recovery", "mixed_midpoint", "native_mixed_dynamics")
    meta = dict(case_ids=["t0", "t1", "v0"], populations=["train", "train", "validation"], horizon=32, macro_dt=.05)
    data, common = {}, {}
    for name, errors in zip(names, ([1., 9., 16.], [4., 4., 4.], [9., 25., 9.])):
        values = np.tile(np.array(errors)[:, None], (1, 32))
        data[name] = dict(rollout_mse=np.pad(values, ((0, 0), (1, 0))))
        np.savez(path/(name+"_training_targets.npz"), rec_mse=np.array(errors), dyn_mse=np.array(errors))
        for metric in ("rec_mse", "dyn_mse"):
            common[name+"__"+metric] = values
            common["pretrained__"+metric] = values
    np.savez(path/"common_targets.npz", **common)
    record = dict(status="completed", **meta,
        artifacts={p.name: figures.digest(p) for p in path.glob("*.npz")})
    (path/"result.json").write_text(json.dumps(record))
    figures.midpoint_figures(tmp_path, output, meta, data, {})
    with (output/"midpoint_paired_comparisons.csv").open() as file:
        rows = list(csv.DictReader(file))
    row = next(r for r in rows if r["population"] == "train" and r["horizon"] == "8" and r["endpoint"] == names[0])
    assert float(row["midpoint_to_endpoint"]) == pytest.approx(2/np.sqrt(5))
    assert int(row["midpoint_wins"]) == 1 and int(row["cases"]) == 2
    assert (output/"midpoint_targets_and_outcomes.pdf").read_bytes().startswith(b"%PDF")
    running = tmp_path/"training/clean4"
    running.mkdir(parents=True)
    (running/"result.json").write_text(json.dumps(dict(status="running")))
    captions = {}
    figures.gradient_figures(tmp_path, output, captions)
    assert not captions  # Do not present incomplete training as a final comparison.
