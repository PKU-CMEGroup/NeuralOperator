import numpy as np
import pytest
import torch


def module():
    from scripts.time_dependent_no import evaluate_kolmogorov_acdm
    return evaluate_kolmogorov_acdm


class AdditiveSampler:
    def __init__(self, multiplier=1.):
        self.multiplier = multiplier
        self.inputs = []

    def transition(self, value, tape):
        self.inputs.append(value.clone())
        nxt = self.multiplier*value + .01*tape[0]
        return dict(raw_next=nxt+10, next_state=nxt)


def test_common_noise_preserves_additive_response_in_signed_probe():
    run = module()
    center = np.ones((16, 16), dtype=np.float32)
    direction = np.full_like(center, .2)
    model = AdditiveSampler()
    _, generated = run.shared_predictions(model, np.stack((center, center+direction, center-direction)), 101, "cpu")
    np.testing.assert_allclose(generated[1]-generated[0], direction, atol=2e-7, rtol=1e-6)
    np.testing.assert_allclose(generated[2]-generated[0], -direction, atol=2e-7, rtol=1e-6)
    repeat = run.shared_predictions(model, np.stack((center, center+direction, center-direction)), 101, "cpu")[1]
    np.testing.assert_array_equal(repeat, generated)


def test_sample_error_decomposes_into_mean_error_and_sampling_spread():
    run = module()
    target = np.ones((16, 16))
    samples = np.stack((target-1, target, target+1))
    result = run.sample_moments(samples, target)
    assert result["mean_prediction_sse"] == 0
    assert result["sampling_sse"] == pytest.approx(2*256/3)
    assert result["mean_sample_sse"] == pytest.approx(2*256/3)
    assert abs(result["closure_sse"]) < 1e-12
    biased = run.sample_moments(samples+2, target)
    assert biased["mean_prediction_sse"] == 1024
    assert biased["sampling_sse"] == pytest.approx(result["sampling_sse"])


def test_adapter_uses_fresh_tapes_and_feeds_restricted_prediction_back(monkeypatch):
    run = module()
    monkeypatch.setattr(run.base, "HORIZONS", (2, 4))
    monkeypatch.setattr(run.base, "SNAPSHOTS", set(range(5)))
    reference = np.ones((5, 16, 16), np.float32)
    model = AdditiveSampler(multiplier=1.1)
    adapter = run.SampledTransition(model, 211)
    case, snapshots = run.base.rollout_case(adapter, reference, 1., "cpu")
    assert case["status"] == "completed" and adapter.calls == 4
    for i, current in enumerate(model.inputs):
        np.testing.assert_array_equal(current.numpy()[0], snapshots["state"][i])
    replay_model = AdditiveSampler(multiplier=1.1)
    _, replay = run.base.rollout_case(run.SampledTransition(replay_model, 211), reference, 1., "cpu")
    np.testing.assert_array_equal(replay["state"], snapshots["state"])
    noise0 = snapshots["state"][1]-1.1*snapshots["state"][0]
    noise1 = snapshots["state"][2]-1.1*snapshots["state"][1]
    assert not np.array_equal(noise0, noise1)


def test_spectrum_matches_energy_enstrophy_and_phase_shift():
    run = module()
    x = np.arange(16)*2*np.pi/16
    field = np.broadcast_to(np.sin(2*x)[:, None], (16, 16))
    spectrum = run.spectrum(field)
    assert spectrum.sum() == pytest.approx(.25)
    assert spectrum[2] == pytest.approx(.25)
    moved = np.roll(field, 3, axis=0)
    np.testing.assert_allclose(run.spectrum(moved), spectrum, atol=1e-16)
    assert run.phase_error_sse(moved, field) < 1e-25
    # The forcing cos(4y) only allows y shifts by quarter periods.
    other = np.broadcast_to(np.sin(x)[None, :], (16, 16))
    assert run.phase_error_sse(np.roll(other, 1, axis=1), other) > 1


def test_teacher_moments_are_per_query_not_ensemble_mean_rollout(monkeypatch):
    run = module()
    monkeypatch.setattr(run, "TEACHER_STEPS", (0, 2))
    truth = np.ones((2, 4, 16, 16), np.float32)
    arrays, rows = run.teacher_queries(AdditiveSampler(), {"train": truth}, "cpu")
    assert len(rows) == 4
    assert arrays["train_samples"].shape == (3, 2, 2, 16, 16)
    for row in rows:
        sample = arrays["train_samples"][:, row["path_index"], row["query_index"]].astype(np.float64)
        target = truth[row["path_index"], row["input_step"]+1]
        assert row["moments"]["mean_sample_sse"] == pytest.approx(np.sum((sample-target)**2)/3)
        assert row["moments"]["mean_prediction_sse"] == pytest.approx(np.sum((sample.mean(0)-target)**2))
        assert abs(row["moments"]["closure_sse"]) < 1e-12


def test_internal_nonfinite_sampler_counts_as_failed_path(monkeypatch):
    run = module()
    monkeypatch.setattr(run.base, "HORIZONS", (2, 4))
    class NonfiniteSampler:
        def transition(self, value, tape):
            raise ValueError("joint field must be finite and match model grid/device/dtype")
    case, _ = run.base.rollout_case(run.SampledTransition(NonfiniteSampler(), 101),
                                   np.ones((5, 16, 16), np.float32), 1., "cpu")
    assert case["status"] == "nonfinite_prediction" and case["failed_at_step"] == 1
    assert all(not h["complete"] for h in case["horizons"])


def test_noisy_roundtrip_accounts_for_float32_cancellation_at_high_noise():
    from utility.time_dependent_no.pcno_kolmogorov_acdm import ACDMSchedule
    schedule = ACDMSchedule()
    rng = torch.Generator().manual_seed(920)
    clean = torch.randn(20, 2, 16, 16, generator=rng)
    noise = torch.randn(clean.shape, generator=rng)
    levels = torch.arange(20)
    signal, sigma = schedule.factors(levels, clean)
    noised = schedule.add_noise(clean, noise, levels)
    reconstructed = schedule.reconstruct(noised, noise, levels)
    # Standard floating-point absolute error scales with the intermediate
    # operands and inverse signal, not a single fixed tolerance for all levels.
    bound = 4*torch.finfo(clean.dtype).eps*(signal*clean.abs()+sigma*noise.abs())/signal
    assert torch.all((reconstructed-clean).abs() <= bound)
    exact_noised = schedule.add_noise(clean.double(), noise.double(), levels)
    exact_reconstruction = schedule.reconstruct(exact_noised, noise.double(), levels)
    torch.testing.assert_close(exact_reconstruction, clean.double(), rtol=0, atol=2e-14)
