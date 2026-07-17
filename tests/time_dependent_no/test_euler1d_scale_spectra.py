from __future__ import annotations

import numpy as np

from scripts.time_dependent_no.diagnose_euler1d_scale_spectra import (
    checkpoint_metadata,
    face_flux_metrics,
    prediction_validity,
    primitive_to_conservative_np,
    summarize_rollout_population,
    spectral_metrics,
    spectral_pair_metrics,
)


def test_checkpoint_metadata_accepts_legacy_cpg_schema() -> None:
    class LegacyArgs:
        pass

    assert checkpoint_metadata({}, LegacyArgs(), "input_coordinates", "primitive") == (
        "primitive"
    )


def test_model_conversion_preserves_float32_arithmetic() -> None:
    primitive = np.array([[[1.1, 0.7, 1.3]]], dtype=np.float32)

    conservative = primitive_to_conservative_np(primitive, gamma=1.4)

    assert conservative.dtype == np.float32
    expected = np.array(
        [[[1.1, np.float32(1.1) * np.float32(0.7), 3.5195]]],
        dtype=np.float32,
    )
    np.testing.assert_allclose(conservative, expected, rtol=1.0e-6, atol=1.0e-7)


def sinusoid(mode: int, cells: int = 256) -> np.ndarray:
    coordinate = np.arange(cells, dtype=np.float64) / cells
    values = np.zeros((1, cells, 3), dtype=np.float64)
    values[0, :, 0] = np.sin(2.0 * np.pi * mode * coordinate)
    return values


def test_spectral_bands_separate_low_and_high_index_modes() -> None:
    low = spectral_metrics(sinusoid(3), prefix="error")
    high = spectral_metrics(sinusoid(40), prefix="error")

    assert low["error_low_1_4_fraction"][0] > 0.95
    assert low["error_high_25_64_fraction"][0] < 1.0e-6
    assert high["error_high_25_64_fraction"][0] > 0.99
    assert high["error_low_1_4_fraction"][0] < 1.0e-6


def test_spectral_pair_reports_projection_gain() -> None:
    target = sinusoid(40)
    prediction = 0.5 * target

    metrics = spectral_pair_metrics(prediction, target, prefix="update")

    np.testing.assert_allclose(
        metrics["update_high_25_64_projection_gain"],
        0.5,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        metrics["update_high_25_64_amplitude_ratio"],
        0.5,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        metrics["update_high_25_64_relative_error"],
        0.5,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        metrics["update_high_25_64_cosine"],
        1.0,
        atol=1.0e-12,
    )


def test_face_flux_null_mode_has_zero_active_error() -> None:
    faces = 33
    normal = np.ones((1, faces, 1), dtype=np.float64)
    normal[:, 0] = -1.0
    physical_constant = np.array([0.7, -0.4, 1.2]).reshape(1, 1, 3)
    prediction = normal * physical_constant
    target = np.zeros_like(prediction)

    metrics = face_flux_metrics(prediction, target, gamma=1.4)

    assert metrics["flux_raw_mse"][0] > 0.0
    assert metrics["flux_gauge_mse"][0] > 0.0
    assert metrics["flux_active_mse"][0] < 1.0e-28


def test_prediction_validity_distinguishes_density_and_pressure() -> None:
    primitive = np.ones((4, 8, 3), dtype=np.float64)
    primitive[1, 0, 0] = -0.1
    primitive[2, 0, 2] = 0.0
    primitive[3, 0, 1] = np.nan

    valid, reasons = prediction_validity(primitive)

    assert valid.tolist() == [True, False, False, False]
    assert reasons == [
        "",
        "nonpositive_density",
        "nonpositive_pressure",
        "nonfinite_state",
    ]


def test_rollout_population_summary_counts_invalid_proposals() -> None:
    rows = [
        {
            "case_id": 2,
            "call": 4,
            "proposal_valid": False,
            "termination_reason": "nonpositive_pressure",
        }
    ]

    summary = summarize_rollout_population(
        rows,
        "model",
        np.array([1, 2, 3]),
        max_calls=5,
    )

    assert summary["num_completed"] == 2
    assert summary["first_failure_call"] == 4
    assert np.isclose(summary["valid_calls_mean"], 13.0 / 3.0)
