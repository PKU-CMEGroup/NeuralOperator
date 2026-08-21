from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from scripts.time_dependent_no import analyze_pcno_shadow_response_geometry as analysis


def _features(seed: int = 0, rows: int = 8) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(rows, len(analysis.FROZEN_ACTIVE_CELLS)))


def test_scalar_and_diagonal_response_fits_recover_zero_intercept_maps() -> None:
    x = _features()
    diagonal = np.linspace(0.8, 1.2, x.shape[1])
    diagonal_fit = analysis.fit_response_maps(x, x * diagonal)
    scalar_fit = analysis.fit_response_maps(x, 1.125 * x)

    np.testing.assert_allclose(diagonal_fit["diagonal"], diagonal, atol=1.0e-14)
    assert scalar_fit["scalar"] == pytest.approx(1.125)
    np.testing.assert_allclose(
        analysis.predict_response(x[0], "diagonal", diagonal_fit),
        x[0] * diagonal,
    )
    np.testing.assert_allclose(
        analysis.predict_response(x[0], "scalar", scalar_fit),
        1.125 * x[0],
    )
    np.testing.assert_array_equal(
        analysis.predict_response(x[0], "identity", scalar_fit), x[0]
    )
    np.testing.assert_array_equal(
        analysis.predict_response(x[0], "zero", scalar_fit), np.zeros_like(x[0])
    )


def test_response_fit_and_prediction_fail_closed() -> None:
    x = _features(rows=3)
    with pytest.raises(ValueError, match="small denominator"):
        analysis.fit_response_maps(np.zeros_like(x), x)
    with pytest.raises(ValueError, match="shape aligned"):
        analysis.fit_response_maps(x[:, :-1], x)
    with pytest.raises(ValueError, match="unknown response map"):
        analysis.predict_response(x[0], "full", {"scalar": 1.0})
    with pytest.raises(ValueError, match="diagonal"):
        analysis.predict_response(x[0], "diagonal", {"diagonal": np.ones(2)})


def test_residual_map_identity_is_exact_for_constant_residual() -> None:
    rng = np.random.default_rng(2)
    shadow = rng.normal(size=(12, 4))
    displacement = 1.0e-3 * rng.normal(size=shadow.shape)
    constant_residual = 0.1 * rng.normal(size=shadow.shape)

    def flow(state: np.ndarray) -> np.ndarray:
        return state + constant_residual

    output_displacement = flow(shadow + displacement) - flow(shadow)
    np.testing.assert_allclose(output_displacement, displacement, atol=3.0e-16)


def _score_row(
    *,
    case_id: str,
    error: float,
    zero: float,
    dot: float,
    prediction_norm2: float,
    target_norm2: float,
) -> dict[str, object]:
    return {
        "case_id": case_id,
        "response_error_sse": error,
        "response_zero_sse": zero,
        "update_error_sse": error,
        "update_zero_sse": zero,
        "response_dot": dot,
        "response_prediction_norm2": prediction_norm2,
        "response_target_norm2": target_norm2,
    }


def test_scope_metrics_are_case_first_and_use_zero_baseline() -> None:
    rows = [
        _score_row(
            case_id="a",
            error=0.0,
            zero=1.0,
            dot=1.0,
            prediction_norm2=1.0,
            target_norm2=1.0,
        ),
        _score_row(
            case_id="a",
            error=0.0,
            zero=1.0,
            dot=1.0,
            prediction_norm2=1.0,
            target_norm2=1.0,
        ),
        _score_row(
            case_id="b",
            error=1.0,
            zero=1.0,
            dot=0.0,
            prediction_norm2=0.0,
            target_norm2=1.0,
        ),
    ]
    metrics = analysis.scope_metrics(rows, case_ids=("a", "b"))

    assert metrics["response_skill"] == pytest.approx(0.5)
    assert metrics["update_skill"] == pytest.approx(0.5)
    assert metrics["median_response_cosine"] is None
    assert metrics["response_skill_status"] == "ok"
    assert metrics["update_skill_status"] == "ok"
    assert metrics["response_relative_rms"] == pytest.approx(np.sqrt(0.5))
    assert metrics["update_relative_rms"] == pytest.approx(np.sqrt(0.5))
    assert metrics["median_response_cosine_status"] == "small_denominator"
    assert metrics["median_response_norm_ratio_status"] == "ok"
    assert metrics["case_metrics"][0]["response_skill"] == pytest.approx(1.0)
    assert metrics["case_metrics"][1]["response_skill"] == pytest.approx(0.0)
    assert metrics["case_metrics"][1]["cosine_status"] == "small_denominator"


def test_update_baseline_is_shared_across_crossfit_sources() -> None:
    rows = [
        {
            "map": "zero",
            "coefficient_source": "e12_final",
            "case_id": "case",
            "input_call": 0,
            "update_error_sse": 2.0,
        },
        {
            "map": "scalar",
            "coefficient_source": "fit_other_hold_case",
            "case_id": "case",
            "input_call": 0,
            "update_error_sse": 1.0,
        },
    ]
    analysis._attach_update_baseline(rows)
    assert rows[1]["update_zero_sse"] == pytest.approx(2.0)


def test_coefficient_stability_reports_unresolved_zero_norm() -> None:
    folds = {
        analysis.CALIBRATION_CASES[0]: {
            "scalar": 0.0,
            "diagonal": np.zeros(len(analysis.FROZEN_ACTIVE_CELLS)),
        },
        analysis.CALIBRATION_CASES[1]: {
            "scalar": 0.0,
            "diagonal": np.zeros(len(analysis.FROZEN_ACTIVE_CELLS)),
        },
    }
    stability = analysis._coefficient_stability(folds)
    assert stability["scalar_stability_status"] == "small_denominator"
    assert stability["scalar_magnitude_ratio"] is None
    assert stability["diagonal_stability_status"] == "small_denominator"
    assert stability["diagonal_cosine"] is None


def test_candidate_gate_fails_at_strict_threshold_equality() -> None:
    map_name = "identity"
    score_rows = []
    for scope in ("e12", "e14"):
        case_ids = (
            analysis.CALIBRATION_CASES if scope == "e12" else analysis.TRANSFER_CASES
        )
        for band in ("overall", *analysis.FIXED_BANDS):
            threshold = 0.99 if band == "overall" else 0.95
            score_rows.append(
                {
                    "map": map_name,
                    "coefficient_source": "e12_final",
                    "scope": scope,
                    "band": band,
                    "response_skill": threshold,
                    "response_skill_status": "ok",
                    "update_skill": threshold,
                    "update_skill_status": "ok",
                    "median_response_cosine": 1.0,
                    "median_response_cosine_status": "ok",
                    "median_response_norm_ratio": 1.0,
                    "median_response_norm_ratio_status": "ok",
                    "case_metrics": [
                        {
                            "case_id": case_id,
                            "response_skill": 1.0,
                            "response_skill_status": "ok",
                            "update_skill": 1.0,
                            "update_skill_status": "ok",
                            "cosine": 1.0,
                            "cosine_status": "ok",
                        }
                        for case_id in case_ids
                    ],
                }
            )
    row_scores = [
        {
            "map": map_name,
            "coefficient_source": "e12_final",
            "active_case": True,
            "update_error_to_exact_offset": 0.0,
            "update_error_to_exact_offset_status": "ok",
            "exact_offset_rms": 1.0,
            "update_error_rms": 0.0,
            "cap_agreement": True,
            "correction_to_native_increment": 0.0,
            "exact_correction_to_native_increment": 0.0,
        }
    ]
    gate = analysis._candidate_gate(
        map_name,
        score_rows,
        row_scores,
        stability={},
        structural_checks={"structural": True},
    )
    assert gate["status"] == "failed"
    assert gate["failed_checks"] == [
        "overall_response_and_update_skill",
        "band_response_and_update_skill",
    ]


def test_active_modal_field_round_trip_uses_state_scale() -> None:
    rng = np.random.default_rng(4)
    coordinates = rng.normal(size=len(analysis.FROZEN_ACTIVE_CELLS))
    modes = max(cell[0] for cell in analysis.FROZEN_ACTIVE_CELLS) + 1
    nodes = 20
    q, _ = np.linalg.qr(rng.normal(size=(nodes, modes)))
    runtime = SimpleNamespace(
        native_projector=SimpleNamespace(
            q_matrix=q,
            basis=np.zeros((nodes, modes)),
            interior_mask=np.ones(nodes, dtype=bool),
            square_root_mass=np.ones(nodes),
        ),
        normalization=SimpleNamespace(state_scale=np.asarray((0.5, 0.75, 1.0, 1.25))),
    )
    field = analysis._field_from_active(coordinates, runtime)
    np.testing.assert_allclose(
        analysis._active_coordinates(field, runtime), coordinates, atol=1.0e-14
    )


def test_synthetic_contract_is_exact() -> None:
    summary = analysis.synthetic_summary()
    assert summary["status"] == "passed"
    assert all(summary["checks"].values())
