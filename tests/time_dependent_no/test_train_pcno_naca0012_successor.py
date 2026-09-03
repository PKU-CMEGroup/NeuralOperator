from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from torch import nn

from scripts.time_dependent_no import train_pcno_naca0012_successor as trainer
from utility.time_dependent_no.pcno_naca0012 import NACANormalization


def _normalization() -> NACANormalization:
    return NACANormalization(
        state_mean=np.zeros(5, dtype=np.float64),
        state_scale=np.ones(5, dtype=np.float64),
        residual_scale=np.ones(5, dtype=np.float64),
        state_rms=np.ones(5, dtype=np.float64),
    )


class _RecordingResidual(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.1))
        self.calls: list[tuple[bool, torch.Tensor]] = []

    def forward(
        self,
        features: torch.Tensor,
        geometry_batch: dict[str, torch.Tensor],
        *,
        fourier_tensors=None,
    ) -> torch.Tensor:
        del geometry_batch, fourier_tensors
        self.calls.append((torch.is_grad_enabled(), features.detach().clone()))
        return self.weight.expand(features.shape[0], features.shape[1], 5)


def test_smoke_replay_bindings_are_exact() -> None:
    runtime = {"torch": "test", "device": "cpu"}
    smoke = {
        "runtime": runtime,
        "initial_model_state_sha256": "a" * 64,
        "presentation_schedule_sha256": "b" * 64,
        "intervention_rng": {"initial_generator_state_sha256": "c" * 64},
    }
    expected = {
        "runtime": runtime,
        "initial_model_state_sha256": "a" * 64,
        "presentation_schedule_sha256": "b" * 64,
        "intervention_initial_generator_state_sha256": "c" * 64,
    }
    trainer._validate_smoke_production_bindings(smoke, **expected)

    for key, replacement in (
        ("runtime", {"torch": "different", "device": "cpu"}),
        ("initial_model_state_sha256", "d" * 64),
        ("presentation_schedule_sha256", "e" * 64),
        ("intervention_rng", {"initial_generator_state_sha256": "f" * 64}),
    ):
        tampered = dict(smoke)
        tampered[key] = replacement
        with pytest.raises(PermissionError, match="does not replay"):
            trainer._validate_smoke_production_bindings(tampered, **expected)


def test_nonfinite_development_selection_keeps_epoch_five_fallback() -> None:
    assert trainer._select_development_checkpoint(
        epoch=5,
        score=math.inf,
        best_epoch=None,
        best_score=math.inf,
    )
    assert not trainer._select_development_checkpoint(
        epoch=10,
        score=math.inf,
        best_epoch=5,
        best_score=math.inf,
    )
    assert trainer._select_development_checkpoint(
        epoch=10,
        score=0.75,
        best_epoch=5,
        best_score=math.inf,
    )
    assert not trainer._select_development_checkpoint(
        epoch=15,
        score=0.75,
        best_epoch=10,
        best_score=0.75,
    )
    assert trainer._canonical_development_score(math.nan) == math.inf
    assert trainer._canonical_development_score(math.inf) == math.inf
    assert trainer._canonical_development_score(-math.inf) == math.inf
    assert trainer._json_number(math.inf) == "Infinity"


def test_production_detached_pushforward_keeps_956_clean_and_detaches_prefix() -> None:
    normalization = _normalization()
    model = _RecordingResidual()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    preceding = torch.full((2, 1, 5), 0.5)
    previous = torch.full((2, 1, 5), 1.0)
    current = torch.full((2, 1, 5), 2.0)
    clean_target = torch.stack((torch.zeros((1, 5)), torch.full((1, 5), 0.2)))
    centers = torch.tensor([956, 957], dtype=torch.int64)
    geometry = {"static_features": torch.zeros(2, 1, 6)}
    event_hasher = trainer._EventHasher()
    cost = {
        "training_model_calls": 0,
        "training_model_sample_calls": 0,
        "inference_model_calls": 0,
        "inference_model_sample_calls": 0,
        "development_selection_model_calls": 0,
        "development_selection_model_sample_calls": 0,
    }

    loss, gradient_norm = trainer._step_arm_batch(
        arm="DETACHED_PUSHFORWARD",
        model=model,
        optimizer=optimizer,
        previous=previous,
        current=current,
        clean_target=clean_target,
        centers=centers,
        preceding=preceding,
        noise=None,
        geometry_batch=geometry,
        fourier_tensors=(),
        normalization=normalization,
        microbatch_size=2,
        epoch=1,
        optimizer_step=1,
        event_hasher=event_hasher,
        cost=cost,
    )

    assert [grad_enabled for grad_enabled, _ in model.calls] == [True, False, True]
    assert model.calls[0][1].shape[0] == 2
    assert model.calls[1][1].shape[0] == 1
    assert model.calls[2][1].shape[0] == 1
    torch.testing.assert_close(model.calls[1][1][..., 6:11], preceding[1:])
    torch.testing.assert_close(model.calls[1][1][..., 11:16], previous[1:])
    torch.testing.assert_close(model.calls[2][1][..., 6:11], previous[1:])
    torch.testing.assert_close(
        model.calls[2][1][..., 11:16], torch.full((1, 1, 5), 1.1)
    )
    assert loss == pytest.approx(0.2575, abs=1.0e-6)
    assert gradient_norm == pytest.approx(0.45, abs=1.0e-6)
    assert float(model.weight.grad) == pytest.approx(-0.45, abs=1.0e-6)
    assert event_hasher.count == 3
    assert cost["training_model_calls"] == 2
    assert cost["training_model_sample_calls"] == 3
    assert cost["inference_model_calls"] == 1
    assert cost["inference_model_sample_calls"] == 1

    differentiable_prefix = torch.full((1, 1, 5), 0.1, requires_grad=True)
    generated, target = trainer._build_detached_pushforward_exposure(
        previous[1:],
        current[1:] + clean_target[1:],
        differentiable_prefix,
        normalization,
    )
    assert generated.requires_grad is False
    assert target.requires_grad is False
    torch.testing.assert_close(generated, torch.full((1, 1, 5), 1.1))
    torch.testing.assert_close(target, torch.full((1, 1, 5), 1.1))
