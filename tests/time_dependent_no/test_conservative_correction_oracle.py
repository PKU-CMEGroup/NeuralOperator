from __future__ import annotations

import numpy as np
import pytest

from utility.time_dependent_no.conservative_correction_oracle import (
    conservative_face_correction_oracle,
    decode_face_impulse_correction,
    euler2d_state_is_admissible,
    select_endpoint_error_faces,
)


def _chain_geometry() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    volume = np.array([1.0, 2.0, 1.5], dtype=np.float64)
    owner = np.array([0, 0, 1, 2], dtype=np.int64)
    neighbor = np.array([-1, 1, 2, -1], dtype=np.int64)
    return volume, owner, neighbor


def _admissible_states() -> tuple[np.ndarray, np.ndarray]:
    current = np.tile(np.array([1.0, 0.2, -0.1, 2.6]), (3, 1))
    predicted = current + np.array([0.01, 0.005, -0.002, 0.02])
    return current, predicted


def _oracle_target(
    predicted: np.ndarray,
    volume: np.ndarray,
    owner: np.ndarray,
    neighbor: np.ndarray,
) -> np.ndarray:
    impulse = np.zeros((owner.size, 4), dtype=np.float64)
    impulse[1] = np.array([0.035, -0.012, 0.008, 0.055])
    impulse[2] = np.array([-0.020, 0.009, -0.004, -0.030])
    return predicted + decode_face_impulse_correction(
        impulse,
        cell_volume=volume,
        face_owner=owner,
        face_neighbor=neighbor,
    )


def test_endpoint_error_support_is_interior_bounded_and_deterministic() -> None:
    prediction = np.zeros((6, 4), dtype=np.float64)
    target = prediction.copy()
    target[:, 0] = np.array([1.0, 4.0, 2.0, 5.0, 3.0, 6.0])
    owner = np.array([0, 0, 1, 2, 3, 4], dtype=np.int64)
    neighbor = np.array([-1, 1, 2, 3, 4, 5], dtype=np.int64)

    selected = select_endpoint_error_faces(
        prediction,
        target,
        face_owner=owner,
        face_neighbor=neighbor,
        state_scale=np.ones(4),
        max_face_fraction=0.4,
    )

    # Two of five interior faces are retained. Face 5 scores 6 and face 3
    # wins the score-5 tie by its lower stored face index.
    np.testing.assert_array_equal(
        selected, np.array([False, False, False, True, False, True])
    )


def test_oracle_recovers_an_interior_face_correction_and_conserves() -> None:
    volume, owner, neighbor = _chain_geometry()
    current, predicted = _admissible_states()
    target = _oracle_target(predicted, volume, owner, neighbor)

    result = conservative_face_correction_oracle(
        predicted,
        target,
        current,
        cell_volume=volume,
        face_owner=owner,
        face_neighbor=neighbor,
        allowed_face_mask=np.array([False, True, True, False]),
        state_scale=np.ones(4),
        max_face_fraction=1.0,
        max_update_ratio=10.0,
        ridge=1e-14,
    )

    assert result.selected_alpha == pytest.approx(1.0)
    assert result.relative_error_reduction > 0.999999
    np.testing.assert_allclose(result.corrected_state, target, atol=1e-9)
    np.testing.assert_allclose(
        np.sum(volume[:, None] * result.cell_correction, axis=0),
        0.0,
        atol=1e-13,
    )
    assert result.conservation_balance_max_abs < 1e-13
    assert result.allowed_face_fraction == pytest.approx(1.0)
    assert result.active_face_fraction == pytest.approx(1.0)
    assert euler2d_state_is_admissible(result.corrected_state)


def test_oracle_enforces_the_correction_norm_budget() -> None:
    volume, owner, neighbor = _chain_geometry()
    current, predicted = _admissible_states()
    target = _oracle_target(predicted, volume, owner, neighbor)

    result = conservative_face_correction_oracle(
        predicted,
        target,
        current,
        cell_volume=volume,
        face_owner=owner,
        face_neighbor=neighbor,
        allowed_face_mask=np.array([False, True, True, False]),
        state_scale=np.ones(4),
        max_face_fraction=1.0,
        max_update_ratio=0.05,
        ridge=1e-14,
    )

    assert 0.0 < result.selected_alpha < 1.0
    assert result.applied_update_ratio <= 0.05 + 1e-12
    assert result.corrected_error < result.baseline_error


def test_oracle_respects_an_anti_smearing_callback() -> None:
    volume, owner, neighbor = _chain_geometry()
    current, predicted = _admissible_states()
    target = _oracle_target(predicted, volume, owner, neighbor)
    minimum_owner_density = predicted[0, 0] - 0.01

    result = conservative_face_correction_oracle(
        predicted,
        target,
        current,
        cell_volume=volume,
        face_owner=owner,
        face_neighbor=neighbor,
        allowed_face_mask=np.array([False, True, True, False]),
        state_scale=np.ones(4),
        max_face_fraction=1.0,
        max_update_ratio=10.0,
        ridge=1e-14,
        line_search_steps=101,
        anti_smearing_check=lambda state: bool(
            state[0, 0] >= minimum_owner_density - 1e-14
        ),
    )

    assert result.anti_smearing_checked
    assert result.corrected_state[0, 0] >= minimum_owner_density - 1e-12
    assert result.selected_alpha < 1.0
    assert result.corrected_error < result.baseline_error


def test_oracle_rejects_boundary_corrections() -> None:
    volume, owner, neighbor = _chain_geometry()
    current, predicted = _admissible_states()

    with pytest.raises(ValueError, match="must all be interior"):
        conservative_face_correction_oracle(
            predicted,
            predicted.copy(),
            current,
            cell_volume=volume,
            face_owner=owner,
            face_neighbor=neighbor,
            allowed_face_mask=np.array([True, False, False, False]),
            state_scale=np.ones(4),
            max_face_fraction=1.0,
            max_update_ratio=0.1,
        )


def test_oracle_rejects_a_support_larger_than_the_locality_budget() -> None:
    volume, owner, neighbor = _chain_geometry()
    current, predicted = _admissible_states()

    with pytest.raises(ValueError, match="exceeds max_face_fraction"):
        conservative_face_correction_oracle(
            predicted,
            predicted.copy(),
            current,
            cell_volume=volume,
            face_owner=owner,
            face_neighbor=neighbor,
            allowed_face_mask=np.array([False, True, True, False]),
            state_scale=np.ones(4),
            max_face_fraction=0.5,
            max_update_ratio=0.1,
        )
