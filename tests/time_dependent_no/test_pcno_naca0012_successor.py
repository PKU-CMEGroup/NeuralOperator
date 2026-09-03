from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from utility.time_dependent_no.pcno_naca0012 import (
    NACANormalization,
    load_naca_baseline_contract,
    predict_normalized_residual,
    recurrent_step,
)
from utility.time_dependent_no.pcno_naca0012_successor import (
    DEVELOPMENT_CENTER_INDICES,
    ERROR_SUBSPACE_RANK,
    PATH_PCA_RANK_CAP,
    STRUCTURED_PAIR_CENTER_INDICES,
    SUCCESSOR_ARMS,
    SUCCESSOR_EXPERIMENT_ID,
    SUCCESSOR_SEEDS,
    TRAIN_CENTER_INDICES,
    CorrectedRecurrentStep,
    HistoryNoise,
    corrected_recurrent_step,
    fit_recovery_calibration,
    fit_train_path_projector,
    identity_corrector,
    make_detached_pushforward_presentation,
    make_recovery_presentation,
    sample_iid_history_noise,
    sample_structured_history_noise,
    validate_pushforward_training_centers,
    validate_successor_development_centers,
    validate_successor_math_contract,
    validate_successor_training_centers,
)

REPOSITORY = Path(__file__).resolve().parents[2]
PARENT_CONTRACT_PATH = (
    REPOSITORY / "docs" / "time_dependent_no" / "R0_NACA_PCNO_BASELINE_CONTRACT.json"
)
SUCCESSOR_CONTRACT_PATH = (
    REPOSITORY
    / "docs"
    / "time_dependent_no"
    / "B3B4_NACA_CORRECTIVE_SUCCESSOR_CONTRACT.json"
)


def _parent_contract():
    return load_naca_baseline_contract(PARENT_CONTRACT_PATH)


def _successor_payload() -> dict:
    return json.loads(SUCCESSOR_CONTRACT_PATH.read_text(encoding="utf-8"))


def _normalization(
    *,
    state_scale: np.ndarray | None = None,
    residual_scale: np.ndarray | None = None,
) -> NACANormalization:
    return NACANormalization(
        state_mean=np.zeros(5, dtype=np.float64),
        state_scale=(
            np.ones(5, dtype=np.float64)
            if state_scale is None
            else np.asarray(state_scale, dtype=np.float64)
        ),
        residual_scale=(
            np.ones(5, dtype=np.float64)
            if residual_scale is None
            else np.asarray(residual_scale, dtype=np.float64)
        ),
        state_rms=np.ones(5, dtype=np.float64),
    )


def _geometry_batch(batch_size: int, num_nodes: int) -> dict[str, torch.Tensor]:
    return {
        "static_features": torch.zeros(batch_size, num_nodes, 6),
    }


class _ToyResidual(nn.Module):
    def __init__(self, previous_weight: float = 0.2, current_weight: float = -0.1):
        super().__init__()
        self.previous_weight = nn.Parameter(torch.tensor(previous_weight))
        self.current_weight = nn.Parameter(torch.tensor(current_weight))

    def forward(
        self,
        features: torch.Tensor,
        geometry_batch: dict[str, torch.Tensor],
        *,
        fourier_tensors=None,
    ) -> torch.Tensor:
        del geometry_batch, fourier_tensors
        previous = features[..., 6:11]
        current = features[..., 11:16]
        return self.previous_weight * previous + self.current_weight * current


def test_successor_math_contract_and_open_roles_are_exact_and_fail_closed() -> None:
    parent = _parent_contract()
    payload = _successor_payload()
    validate_successor_math_contract(payload, parent)
    assert payload["experiment_id"] == SUCCESSOR_EXPERIMENT_ID
    assert tuple(payload["arms"]) == SUCCESSOR_ARMS
    assert tuple(payload["optimization"]["seeds"]) == SUCCESSOR_SEEDS
    assert (
        validate_successor_training_centers(parent, TRAIN_CENTER_INDICES)
        == TRAIN_CENTER_INDICES
    )
    assert (
        validate_successor_development_centers(parent, DEVELOPMENT_CENTER_INDICES)
        == DEVELOPMENT_CENTER_INDICES
    )
    assert (
        validate_pushforward_training_centers(parent, STRUCTURED_PAIR_CENTER_INDICES)
        == STRUCTURED_PAIR_CENTER_INDICES
    )
    assert len(TRAIN_CENTER_INDICES) == 238
    assert len(STRUCTURED_PAIR_CENTER_INDICES) == 237
    assert len(TRAIN_CENTER_INDICES) == 238
    assert len(STRUCTURED_PAIR_CENTER_INDICES) == 237
    with pytest.raises(ValueError, match="complete train-only prefix"):
        validate_pushforward_training_centers(parent, [956])
    with pytest.raises(ValueError, match="leaks"):
        validate_successor_training_centers(parent, [1234])
    tampered = json.loads(json.dumps(payload))
    tampered["calibration"]["structured_rank"] = 15
    with pytest.raises(ValueError, match="calibration differs"):
        validate_successor_math_contract(tampered, parent)


def test_recovery_displaces_both_slots_and_targets_clean_future() -> None:
    normalization = _normalization(
        state_scale=np.asarray([2.0, 3.0, 4.0, 5.0, 6.0]),
        residual_scale=np.asarray([0.5, 1.0, 2.0, 4.0, 8.0]),
    )
    previous = torch.full((2, 3, 5), 1.0)
    current = torch.full((2, 3, 5), 2.0)
    clean_next = torch.full((2, 3, 5), 5.0)
    previous_noise = torch.full_like(previous, 0.25)
    current_noise = torch.full_like(current, -0.125)
    presentation = make_recovery_presentation(
        previous,
        current,
        clean_next,
        HistoryNoise(previous_noise, current_noise),
        normalization,
    )
    state_scale = torch.tensor(normalization.state_scale, dtype=torch.float32)
    residual_scale = torch.tensor(normalization.residual_scale, dtype=torch.float32)
    expected_previous = previous + state_scale * previous_noise
    expected_current = current + state_scale * current_noise
    expected_target = (clean_next - expected_current) / residual_scale
    torch.testing.assert_close(presentation.previous, expected_previous)
    torch.testing.assert_close(presentation.current, expected_current)
    torch.testing.assert_close(presentation.target_normalized_residual, expected_target)

    changed_previous = make_recovery_presentation(
        previous,
        current,
        clean_next,
        HistoryNoise(previous_noise + 7.0, current_noise),
        normalization,
    )
    assert not torch.equal(changed_previous.previous, presentation.previous)
    torch.testing.assert_close(
        changed_previous.target_normalized_residual,
        presentation.target_normalized_residual,
    )


def test_zero_noise_is_exact_clean_parity_and_iid_stream_is_deterministic() -> None:
    normalization = _normalization(
        state_scale=np.asarray([0.5, 1.0, 2.0, 3.0, 4.0]),
        residual_scale=np.asarray([4.0, 3.0, 2.0, 1.0, 0.5]),
    )
    previous = torch.randn(2, 4, 5, generator=torch.Generator().manual_seed(1))
    current = torch.randn(2, 4, 5, generator=torch.Generator().manual_seed(2))
    clean_next = torch.randn(2, 4, 5, generator=torch.Generator().manual_seed(3))
    zero = torch.zeros_like(current)
    clean = make_recovery_presentation(
        previous,
        current,
        clean_next,
        HistoryNoise(zero, zero),
        normalization,
    )
    assert torch.equal(clean.previous, previous)
    assert torch.equal(clean.current, current)
    torch.testing.assert_close(
        clean.target_normalized_residual,
        normalization.normalize_residual(clean_next - current),
    )

    field_rms = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
    generator_a = torch.Generator().manual_seed(914)
    generator_b = torch.Generator().manual_seed(914)
    noise_a = sample_iid_history_noise(current, field_rms, generator=generator_a)
    noise_b = sample_iid_history_noise(current, field_rms, generator=generator_b)
    assert torch.equal(noise_a.previous_normalized, noise_b.previous_normalized)
    assert torch.equal(noise_a.current_normalized, noise_b.current_normalized)
    assert not torch.equal(noise_a.previous_normalized, noise_a.current_normalized)
    noise_next = sample_iid_history_noise(current, field_rms, generator=generator_a)
    assert not torch.equal(noise_a.previous_normalized, noise_next.previous_normalized)

    zero_iid_noise = sample_iid_history_noise(
        current,
        torch.zeros(5),
        generator=torch.Generator().manual_seed(4),
    )
    zero_iid = make_recovery_presentation(
        previous,
        current,
        clean_next,
        zero_iid_noise,
        normalization,
    )
    assert torch.equal(zero_iid.previous, previous)
    assert torch.equal(zero_iid.current, current)
    torch.testing.assert_close(
        zero_iid.target_normalized_residual,
        clean.target_normalized_residual,
    )


def test_train_only_rank16_calibration_matches_iid_energy_and_sampling() -> None:
    parent = _parent_contract()
    errors = torch.randn(
        3,
        len(TRAIN_CENTER_INDICES),
        2,
        5,
        dtype=torch.float64,
        generator=torch.Generator().manual_seed(2718),
    )
    calibration = fit_recovery_calibration(
        errors,
        TRAIN_CENTER_INDICES,
        SUCCESSOR_SEEDS,
        parent,
    )
    repeated_calibration = fit_recovery_calibration(
        errors,
        TRAIN_CENTER_INDICES,
        SUCCESSOR_SEEDS,
        parent,
    )
    assert torch.equal(
        calibration.structured_basis, repeated_calibration.structured_basis
    )
    assert torch.equal(
        calibration.structured_coefficient_std,
        repeated_calibration.structured_coefficient_std,
    )
    restored_calibration = type(calibration).from_mapping(calibration.to_mapping())
    assert torch.equal(calibration.iid_field_rms, restored_calibration.iid_field_rms)
    assert torch.equal(
        calibration.structured_basis, restored_calibration.structured_basis
    )
    tampered_calibration = calibration.to_mapping()
    tampered_calibration["parent_seeds"][0] = 99
    with pytest.raises(ValueError, match="parent_seeds differs"):
        type(calibration).from_mapping(tampered_calibration)
    expected_rms = torch.sqrt(torch.mean(torch.square(errors), dim=(0, 1, 2)))
    torch.testing.assert_close(calibration.iid_field_rms, expected_rms)
    assert calibration.structured_basis.shape == (ERROR_SUBSPACE_RANK, 2, 2, 5)
    basis = calibration.structured_basis.reshape(ERROR_SUBSPACE_RANK, -1)
    torch.testing.assert_close(
        basis @ basis.T,
        torch.eye(ERROR_SUBSPACE_RANK, dtype=torch.float64),
        rtol=1.0e-8,
        atol=1.0e-10,
    )
    expected_energy = 2.0 * 2 * torch.sum(torch.square(expected_rms))
    assert calibration.iid_expected_history_pair_energy == pytest.approx(
        float(expected_energy), rel=1.0e-12
    )
    assert torch.sum(
        torch.square(calibration.structured_coefficient_std)
    ).item() == pytest.approx(float(expected_energy), rel=1.0e-12)
    assert calibration.pair_sample_count == 3 * 237
    assert int(calibration.to_mapping()["structured_rank"]) == 16

    generator_a = torch.Generator().manual_seed(1618)
    generator_b = torch.Generator().manual_seed(1618)
    sample_a = sample_structured_history_noise(
        calibration, 7, generator=generator_a, device="cpu"
    )
    sample_b = sample_structured_history_noise(
        calibration, 7, generator=generator_b, device="cpu"
    )
    assert sample_a.previous_normalized.shape == (7, 2, 5)
    assert torch.equal(sample_a.previous_normalized, sample_b.previous_normalized)
    assert torch.equal(sample_a.current_normalized, sample_b.current_normalized)
    sample_next = sample_structured_history_noise(
        calibration, 7, generator=generator_a, device="cpu"
    )
    assert not torch.equal(
        sample_a.previous_normalized, sample_next.previous_normalized
    )

    with pytest.raises(ValueError, match="all ordered train centers"):
        fit_recovery_calibration(
            None,  # type: ignore[arg-type]
            DEVELOPMENT_CENTER_INDICES,
            SUCCESSOR_SEEDS,
            parent,
        )


def test_detached_pushforward_uses_complete_frames_and_only_final_call_gradients() -> (
    None
):
    parent = _parent_contract()
    normalization = _normalization()
    model = _ToyResidual()
    batch_size, num_nodes = 2, 3
    state_n_minus_2 = torch.full((batch_size, num_nodes, 5), 1.5)
    state_n_minus_1 = torch.full((batch_size, num_nodes, 5), 2.0)
    clean_state_n_plus_1 = torch.full((batch_size, num_nodes, 5), 4.0)
    geometry = _geometry_batch(batch_size, num_nodes)
    presentation = make_detached_pushforward_presentation(
        model,  # type: ignore[arg-type]
        state_n_minus_2,
        state_n_minus_1,
        clean_state_n_plus_1,
        [957, 958],
        geometry,
        normalization,
        parent,
    )
    expected_generated = state_n_minus_1 + (
        model.previous_weight.detach() * state_n_minus_2
        + model.current_weight.detach() * state_n_minus_1
    )
    assert torch.equal(presentation.previous, state_n_minus_1)
    torch.testing.assert_close(presentation.generated_current, expected_generated)
    assert presentation.generated_current.requires_grad is False
    torch.testing.assert_close(
        presentation.target_normalized_residual,
        clean_state_n_plus_1 - expected_generated,
    )

    prediction = predict_normalized_residual(
        model,  # type: ignore[arg-type]
        presentation.previous,
        presentation.generated_current,
        geometry,
        normalization,
    )
    loss = torch.sum(torch.square(prediction - presentation.target_normalized_residual))
    loss.backward()
    observed_gradients = (
        model.previous_weight.grad.detach().clone(),
        model.current_weight.grad.detach().clone(),
    )

    reference_model = _ToyResidual(
        float(model.previous_weight.detach()), float(model.current_weight.detach())
    )
    reference_prediction = predict_normalized_residual(
        reference_model,  # type: ignore[arg-type]
        state_n_minus_1,
        expected_generated.detach(),
        geometry,
        normalization,
    )
    reference_loss = torch.sum(
        torch.square(
            reference_prediction - (clean_state_n_plus_1 - expected_generated.detach())
        )
    )
    reference_loss.backward()
    torch.testing.assert_close(
        observed_gradients[0], reference_model.previous_weight.grad
    )
    torch.testing.assert_close(
        observed_gradients[1], reference_model.current_weight.grad
    )

    with pytest.raises(ValueError, match="complete train-only prefix"):
        make_detached_pushforward_presentation(
            model,  # type: ignore[arg-type]
            state_n_minus_2[:1],
            state_n_minus_1[:1],
            clean_state_n_plus_1[:1],
            [956],
            _geometry_batch(1, num_nodes),
            normalization,
            parent,
        )


def test_train_path_projector_uses_full_state_interpolant_and_rejects_leakage() -> None:
    parent = _parent_contract()
    normalization = _normalization()
    time = np.linspace(-1.0, 1.0, 238, dtype=np.float64)
    states = np.stack((time, time**2, time**3, np.sin(time), np.cos(time)), axis=1)[
        :, None, :
    ]
    projector = fit_train_path_projector(
        states,
        TRAIN_CENTER_INDICES,
        normalization,
        parent,
    )
    assert 1 <= projector.rank <= PATH_PCA_RANK_CAP
    assert projector.captured_variance >= 0.999
    midpoint = 0.5 * (states[100] + states[101])
    query = torch.from_numpy(midpoint.astype(np.float32))[None]
    runtime_projector = projector.to("cpu", dtype=torch.float32)
    result = runtime_projector.project(query)
    repeated_result = runtime_projector.project(query)
    assert torch.equal(result.corrected_state, repeated_result.corrected_state)
    assert torch.equal(result.segment_indices, repeated_result.segment_indices)
    restored_projector = (
        type(projector)
        .from_mapping(projector.to_mapping(), normalization)
        .to("cpu", dtype=torch.float32)
    )
    restored_result = restored_projector.project(query)
    assert torch.equal(result.corrected_state, restored_result.corrected_state)
    assert torch.equal(result.segment_indices, restored_result.segment_indices)
    tampered_projector = projector.to_mapping()
    tampered_projector["tie_break"] = np.asarray("highest_segment_index")
    with pytest.raises(ValueError, match="tie_break differs"):
        type(projector).from_mapping(tampered_projector, normalization)
    torch.testing.assert_close(result.corrected_state, query, rtol=1.0e-5, atol=1.0e-6)
    assert int(result.segment_indices.item()) == 100
    assert float(result.segment_fractions.item()) == pytest.approx(0.5, abs=2.0e-5)
    assert float(result.embedding_distance.item()) == pytest.approx(0.0, abs=1.0e-6)

    class _ForbiddenStates:
        def __array__(self):
            raise AssertionError("nontrain values must not be inspected")

    with pytest.raises(ValueError, match="exact ordered train-current frames"):
        fit_train_path_projector(
            _ForbiddenStates(),  # type: ignore[arg-type]
            DEVELOPMENT_CENTER_INDICES,
            normalization,
            parent,
        )


def test_train_path_pca_records_when_the_rank32_cap_is_active() -> None:
    parent = _parent_contract()
    normalization = _normalization()
    random_columns = np.random.default_rng(99).standard_normal((238, 40))
    matrix = np.concatenate((np.ones((238, 1)), random_columns), axis=1)
    orthogonal, _ = np.linalg.qr(matrix)
    states = np.asarray(orthogonal[:, 1:41], dtype=np.float64).reshape(238, 8, 5)
    projector = fit_train_path_projector(
        states,
        TRAIN_CENTER_INDICES,
        normalization,
        parent,
    )
    assert projector.rank == PATH_PCA_RANK_CAP
    assert projector.variance_rank_without_cap == 40
    assert projector.rank_cap_active is True
    assert projector.captured_variance == pytest.approx(0.8, rel=1.0e-12)


def test_corrected_recurrence_preserves_raw_output_and_identity_parity() -> None:
    normalization = _normalization()
    model = _ToyResidual()
    previous = torch.full((2, 3, 5), 1.0)
    current = torch.full((2, 3, 5), 2.0)
    geometry = _geometry_batch(2, 3)
    expected_previous, expected_raw = recurrent_step(
        model,  # type: ignore[arg-type]
        previous,
        current,
        geometry,
        normalization,
    )
    identity = corrected_recurrent_step(
        model,  # type: ignore[arg-type]
        previous,
        current,
        geometry,
        normalization,
        identity_corrector,
    )
    assert isinstance(identity, CorrectedRecurrentStep)
    assert torch.equal(identity.recurrent_previous, expected_previous)
    assert torch.equal(identity.raw_next_state, expected_raw)
    assert torch.equal(identity.corrected_next_state, expected_raw)
    assert (
        identity.corrected_next_state.data_ptr() == identity.raw_next_state.data_ptr()
    )

    shifted = corrected_recurrent_step(
        model,  # type: ignore[arg-type]
        previous,
        current,
        geometry,
        normalization,
        lambda state: state + 0.75,
    )
    assert torch.equal(shifted.raw_next_state, expected_raw)
    torch.testing.assert_close(shifted.corrected_next_state, expected_raw + 0.75)
    next_step = corrected_recurrent_step(
        model,  # type: ignore[arg-type]
        shifted.recurrent_previous,
        shifted.corrected_next_state,
        geometry,
        normalization,
        identity_corrector,
    )
    _, expected_feedback = recurrent_step(
        model,  # type: ignore[arg-type]
        current,
        expected_raw + 0.75,
        geometry,
        normalization,
    )
    torch.testing.assert_close(next_step.raw_next_state, expected_feedback)
