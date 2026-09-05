"""Synthetic code-to-intent checks for the solver-free Bump comparison."""

import copy
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

import scripts.time_dependent_no.run_pcno_bump_corrective as runner
import utility.time_dependent_no.pcno_bump_corrective as correction
from utility.time_dependent_no.pcno_euler2d import (
    Euler2DNormalization,
    PCNOEuler2DResidual,
)


def test_source_capture_ignores_virtual_torch_module_filenames(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "b5_virtual_module", SimpleNamespace(__file__="_classes.py")
    )
    source = runner.source_records()
    assert "scripts/time_dependent_no/run_pcno_bump_corrective.py" in source
    assert "_classes.py" not in source


@pytest.fixture
def norm():
    return Euler2DNormalization(np.zeros(4), np.ones(4), np.ones(4) * 0.1, 3.0, 0.2)


@pytest.fixture
def sample():
    nodes = torch.cartesian_prod(torch.arange(3), torch.arange(3)).float()
    edges = [
        (i, j)
        for i in range(9)
        for j in range(9)
        if torch.linalg.vector_norm(nodes[i] - nodes[j]) == 1
    ]
    edges = torch.tensor(edges)
    weights = (nodes[edges[:, 1]] - nodes[edges[:, 0]]) * 0.5
    state = torch.tensor([1.4, 4.2, 0.0, 8.8]).expand(1, 9, 4).clone()
    return {
        "current": state,
        "target": state + torch.tensor([0.01, 0.02, 0.005, 0.05]),
        "nodes": nodes[None],
        "node_weights": torch.ones(1, 9, 1) / 9,
        "node_mask": torch.ones(1, 9, 1),
        "node_rhos": torch.ones(1, 9, 1),
        "node_type": torch.zeros(1, 9, dtype=torch.long),
        "mach": torch.tensor([3.0]),
        "directed_edges": edges[None],
        "edge_gradient_weights": weights[None],
    }


@pytest.mark.parametrize("training_map", (False, True))
@pytest.mark.parametrize("target", (1, 2, 4, 79))
def test_prefix_target_timing_and_available_history(training_map, target):
    for requested in (1, 2, 3):
        start, depth = correction.prefix_start(
            target, requested, training_map=training_map
        )
        assert start >= 0
        assert depth <= requested
        assert start + depth + int(training_map) == target
        assert depth == min(requested, target - int(training_map))


def test_prefixes_have_no_gradient_but_corrector_does(monkeypatch):
    base = nn.Linear(1, 1, bias=False)
    base.weight.data.fill_(2)
    monkeypatch.setattr(
        correction, "map_step", lambda model, sample, x, policy: model(x)
    )
    x = torch.tensor([[3.0]], requires_grad=True)
    prefix = correction.detached_prefix(base, {}, x, 3, None)
    assert prefix.item() == 24
    assert not prefix.requires_grad
    corrector = nn.Linear(1, 1, bias=False)
    corrector(prefix).square().mean().backward()
    assert corrector.weight.grad is not None
    assert base.weight.grad is None and x.grad is None


def test_curriculum_and_keyed_rng():
    assert [
        correction.exposure_probability(t, 100) for t in (0, 10, 25, 40, 100)
    ] == pytest.approx([0, 0, 0.25, 0.5, 0.5])
    device = torch.device("cpu")
    a = torch.randn(10, generator=correction.generator_for(device, 17, "iid", 30))
    b = torch.randn(10, generator=correction.generator_for(device, 17, "iid", 30))
    c = torch.randn(10, generator=correction.generator_for(device, 17, "refiner", 30))
    assert torch.equal(a, b) and not torch.equal(a, c)


def test_ema_preserves_fixed_normalization_exactly(norm):
    model = PCNOEuler2DResidual(normalization=norm, k_max=2, layers=(8, 8), fc_dim=8)
    ema = copy.deepcopy(model).requires_grad_(False)
    model.backbone.fc2.bias.data.fill_(1)
    for _ in range(50):
        correction.update_ema(ema, model)
    assert 0 < ema.backbone.fc2.bias[0] < 1
    for a, b in zip(ema.buffers(), model.buffers(), strict=True):
        assert torch.equal(a, b)


def test_refiner_common_initialization_and_residual_identity(norm, sample):
    torch.manual_seed(8)
    base = PCNOEuler2DResidual(normalization=norm, k_max=2, layers=(8, 8), fc_dim=8)
    torch.manual_seed(8)
    refiner = correction.BumpRefiner(
        normalization=norm, k_max=2, layers=(8, 8), fc_dim=8
    )
    for name, value in base.state_dict().items():
        other = refiner.state_dict()[name]
        if name == "backbone.fc0.weight":
            assert torch.equal(value, other[:, :12])
            assert torch.count_nonzero(other[:, 12:]) == 0
        else:
            assert torch.equal(value, other)
    assert torch.equal(
        correction.map_step(base, sample, sample["current"], None), sample["current"]
    )
    scheduler = correction.BumpRefinerScheduler(0.05)
    loss, _ = correction.refiner_loss(
        refiner,
        sample,
        None,
        scheduler,
        correction.generator_for(torch.device("cpu"), 1, "loss"),
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert refiner.backbone.fc2.weight.grad.abs().sum() > 0


def test_refiner_four_reverse_calls_recover_known_clean_residual():
    scheduler = correction.BumpRefinerScheduler(0.03)
    clean = torch.tensor([[[0.1, 0.2, 0.3, 0.4]]])
    sample = torch.randn_like(clean)
    count = 0
    for t in (3, 2, 1, 0):
        levels = torch.tensor([t])
        signal, noise = scheduler._factors(levels, sample)
        velocity = (signal * sample - clean) / noise
        sample = scheduler.step(
            velocity, t, sample, noise=torch.randn_like(sample) if t else None
        )
        count += 1
    torch.testing.assert_close(sample, clean, rtol=1e-5, atol=1e-6)
    assert count == 4
    for invalid in (0, 1, 2, float("nan")):
        with pytest.raises(ValueError):
            correction.BumpRefinerScheduler(invalid)


def test_mapped_pca_is_weighted_projection(norm):
    rng = np.random.default_rng(2)
    factors = rng.normal(size=(12, 2))
    fields = rng.normal(size=(2, 20))
    states = (factors @ fields).reshape(12, 5, 4)
    pca = correction.MappedTrainingPCA(states, norm, np.arange(1, 6)[:, None], 2, "cpu")
    originals = torch.tensor(states, dtype=torch.float32)
    torch.testing.assert_close(pca(originals), originals, rtol=2e-5, atol=2e-5)
    x = torch.randn(3, 5, 4)
    projected = pca(x)
    torch.testing.assert_close(pca(projected), projected, rtol=2e-5, atol=2e-5)
    normalized_error = ((x - projected) / pca.scale).flatten(1) * pca.sqrt_weight
    torch.testing.assert_close(
        normalized_error @ pca.basis.T, torch.zeros(3, 2), atol=2e-6, rtol=0
    )


def channel_mesh():
    x, y = np.meshgrid(np.linspace(0, 6, 7), np.linspace(0, 2, 5))
    nodes = np.c_[x.ravel(), y.ravel()]
    types = np.zeros(len(nodes), dtype=int)
    types[(nodes[:, 1] == 0) | (nodes[:, 1] == 2)] = 1
    types[nodes[:, 0] == 0] = 3
    types[nodes[:, 0] == 6] = 2
    return nodes, types


def test_geometry_remap_copies_coincident_nodes_and_preserves_types():
    nodes, types = channel_mesh()
    order = np.random.default_rng(4).permutation(len(nodes))
    indices, weights = correction.geometry_remap(
        nodes, types, nodes[order], types[order]
    )
    values = np.arange(len(nodes), dtype=float)
    observed = (values[indices] * weights).sum(axis=1)
    np.testing.assert_array_equal(observed, values[order])
    np.testing.assert_allclose(weights.sum(axis=1), 1)
    assert np.all(types[indices] == types[order, None])


def test_projection_fits_train_donor_without_development_states(norm):
    nodes, types = channel_mesh()
    rng = np.random.default_rng(8)
    states = rng.normal(size=(80, len(nodes), 4))

    class Store:
        def array(self, key, name):
            return {
                "nodes": nodes,
                "node_type": types,
                "node_weights": np.ones((len(nodes), 1)),
            }[name]

        def entry(self, key):
            return {"mach": 3.0}

        def states(self, key):
            assert key == "1", "development trajectory was read to fit projection"
            return states

    descriptors = {"1": correction.geometry_descriptor(nodes, types, 3.0)}
    projector, receipt = runner.mapped_projector(
        Store(), "99", ["1"], descriptors, norm, "cpu", 7
    )
    assert receipt["donor"] == "1" and not receipt["development_values_used_to_fit"]
    assert projector.effective_rank == 7


def test_scoped_store_rejects_unopened_keys_before_read(monkeypatch):
    # Exercise the access guard before any base store path or array is consulted.
    store = object.__new__(runner.ScopedStore)
    store.train_keys, store.development_keys = {"1"}, {"2"}
    store.allow_development = False
    for key in ("2", "999"):
        with pytest.raises(PermissionError):
            store.array(key, "states_conservative")


def test_explicit_corrector_evaluation_feeds_back_corrected_state(
    tmp_path, norm, sample, monkeypatch
):
    class Model(nn.Module):
        def __init__(self, shift):
            super().__init__()
            self.shift = shift
            self.gamma = 1.4
            self.state_scale = torch.ones(1, 1, 4)

    class Store:
        def tensor_sample(self, *args, **kwargs):
            return sample

        def states(self, key):
            initial = sample["current"][0].numpy()
            return np.stack([initial + 0.01 * t for t in range(80)])

        def array(self, key, name):
            if name == "edges":
                return sample["directed_edges"][0].numpy()
            return sample[name][0].numpy()

    sample["node_type"][0, 0] = 1
    monkeypatch.setattr(
        runner, "map_step", lambda model, sample, x, policy: x + model.shift
    )
    monkeypatch.setattr(runner, "endpoint_diagnostics", lambda *args, **kwargs: {})
    rows = runner.evaluate_arm(
        "PREFIX_ERROR_CORRECTOR_K13",
        Model(-0.01),
        Model(0.02),
        Store(),
        ["2"],
        ["1"],
        {},
        norm,
        {"2": None},
        tmp_path,
        torch.device("cpu"),
        "none",
        {"refiner_sigma": 0.05},
        17,
    )
    assert rows[0]["completed_calls"] == 79
    assert rows[0]["h79"] < 2e-5
    assert rows[0]["deployed_model_calls_per_step"] == 2
    assert rows[0]["mean_correction_normalized_rms"] == pytest.approx(0.01, rel=1e-4)


@pytest.mark.parametrize("arm", correction.TRAIN_ARMS)
def test_training_objectives_run_on_synthetic_cpu(
    arm, tmp_path, norm, sample, monkeypatch
):
    torch.set_num_threads(1)

    class Store:
        def entry(self, key):
            return {"num_steps": 80, "num_nodes": 9}

        def tensor_sample(self, key, t, **kwargs):
            return {
                **sample,
                "current": sample["current"] + 0.001 * t,
                "target": sample["current"] + 0.001 * (t + 1),
            }

    original = runner.model_for
    monkeypatch.setattr(
        runner,
        "model_for",
        lambda arm, norm, seed, device: original(arm, norm, seed, device, tiny=True),
    )
    base = original("CLEAN", norm, 17, torch.device("cpu"), tiny=True).requires_grad_(
        False
    )
    model, _ = runner.train_arm(
        arm,
        Store(),
        ["1"],
        norm,
        {"1": None},
        tmp_path,
        {},
        17,
        torch.device("cpu"),
        "none",
        {"primitive_noise_std": 0.001, "refiner_sigma": 0.05},
        base,
        smoke=True,
    )
    receipt = json.loads((tmp_path / arm / "training.json").read_text())
    assert receipt["status"] == "smoke_pass" and receipt["updates"] == 4
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert all(p.grad is None for p in base.parameters())
