from __future__ import annotations

import numpy as np
import pytest

from utility.time_dependent_no.path_conditioned_tube import (
    aggregate_tube_escape_tail_score,
    paired_solver_response_metrics,
    path_conditioned_tube_metrics,
)


def _field(value: tuple[float, float]) -> np.ndarray:
    return np.asarray(value, dtype=np.float64).reshape(1, 2)


def _metrics(
    *,
    displaced_input: tuple[float, float],
    reference_prediction: tuple[float, float],
    displaced_prediction: tuple[float, float],
) -> dict[str, float | str | None]:
    return path_conditioned_tube_metrics(
        reference_input=_field((0.0, 0.0)),
        displaced_input=_field(displaced_input),
        reference_prediction=_field(reference_prediction),
        displaced_prediction=_field(displaced_prediction),
        reference_next=_field((0.0, 0.0)),
        node_weights=np.ones(1),
        component_scale=np.ones(2),
    )


def test_tube_escape_slope_recovers_aligned_growth_and_exact_identities() -> None:
    metrics = _metrics(
        displaced_input=(2.0, 0.0),
        reference_prediction=(2.0, 0.0),
        displaced_prediction=(4.0, 0.0),
    )

    assert metrics["input_displacement_scaled_rms"] == pytest.approx(np.sqrt(2.0))
    assert metrics["clean_defect_scaled_rms"] == pytest.approx(np.sqrt(2.0))
    assert metrics["learned_secant_gain"] == pytest.approx(1.0)
    assert metrics["tube_escape_slope"] == pytest.approx(1.0)
    assert metrics["defect_response_cosine"] == pytest.approx(1.0)
    assert metrics["squared_excess_per_input_energy"] == pytest.approx(3.0)
    assert metrics["alignment_per_input_energy"] == pytest.approx(2.0)
    assert metrics["output_closure_scaled_rms"] == pytest.approx(0.0)
    assert metrics["energy_identity_relative"] <= 1.0e-15
    assert metrics["slope_identity_absolute"] <= 1.0e-15
    assert metrics["secant_bound_relative_violation"] == 0.0


def test_tube_escape_slope_keeps_helpful_cancellation_signed() -> None:
    metrics = _metrics(
        displaced_input=(1.0, 0.0),
        reference_prediction=(2.0, 0.0),
        displaced_prediction=(1.0, 0.0),
    )

    assert metrics["learned_secant_gain"] == pytest.approx(1.0)
    assert metrics["tube_escape_slope"] == pytest.approx(-1.0)
    assert metrics["defect_response_cosine"] == pytest.approx(-1.0)
    assert abs(float(metrics["tube_escape_slope"])) <= float(
        metrics["learned_secant_gain"]
    )


def test_metrics_are_invariant_to_common_channel_rescaling() -> None:
    baseline = path_conditioned_tube_metrics(
        reference_input=np.zeros((2, 2)),
        displaced_input=np.asarray([[1.0, 4.0], [3.0, 8.0]]),
        reference_prediction=np.asarray([[2.0, 2.0], [4.0, 6.0]]),
        displaced_prediction=np.asarray([[3.0, 6.0], [7.0, 10.0]]),
        reference_next=np.zeros((2, 2)),
        node_weights=np.asarray([[1.0, 1.0], [3.0, 1.0]]),
        node_mask=np.asarray([1, 1]),
        component_scale=np.asarray([1.0, 2.0]),
    )
    factors = np.asarray([7.0, 0.25])
    scaled = path_conditioned_tube_metrics(
        reference_input=np.zeros((2, 2)),
        displaced_input=np.asarray([[1.0, 4.0], [3.0, 8.0]]) * factors,
        reference_prediction=np.asarray([[2.0, 2.0], [4.0, 6.0]]) * factors,
        displaced_prediction=np.asarray([[3.0, 6.0], [7.0, 10.0]]) * factors,
        reference_next=np.zeros((2, 2)),
        node_weights=np.asarray([[1.0, 1.0], [3.0, 1.0]]),
        node_mask=np.asarray([1, 1]),
        component_scale=np.asarray([1.0, 2.0]) * factors,
    )

    for field in (
        "input_displacement_scaled_rms",
        "clean_defect_scaled_rms",
        "learned_response_scaled_rms",
        "displaced_defect_scaled_rms",
        "learned_secant_gain",
        "tube_escape_slope",
        "defect_response_cosine",
    ):
        assert scaled[field] == pytest.approx(baseline[field])


def test_zero_displacement_is_unresolved_without_an_epsilon() -> None:
    metrics = _metrics(
        displaced_input=(0.0, 0.0),
        reference_prediction=(1.0, 0.0),
        displaced_prediction=(1.0, 0.0),
    )
    assert metrics["input_displacement_scaled_rms"] == 0.0
    assert metrics["learned_secant_gain"] is None
    assert metrics["tube_escape_slope"] is None
    assert metrics["squared_excess_per_input_energy"] is None


@pytest.mark.parametrize(
    ("field", "value", "match"),
    (
        ("displaced_input", np.ones((3, 2)), "identical shapes"),
        ("reference_prediction", np.asarray([[np.nan, 0.0]]), "non-finite"),
        ("node_weights", np.asarray([-1.0]), "nonnegative"),
        ("component_scale", np.asarray([1.0, 0.0]), "strictly positive"),
        ("node_mask", np.asarray([0.5]), "zeros and ones"),
    ),
)
def test_metrics_fail_closed_on_invalid_geometry(
    field: str, value: np.ndarray, match: str
) -> None:
    kwargs = {
        "reference_input": np.zeros((1, 2)),
        "displaced_input": np.ones((1, 2)),
        "reference_prediction": np.ones((1, 2)),
        "displaced_prediction": 2.0 * np.ones((1, 2)),
        "reference_next": np.zeros((1, 2)),
        "node_weights": np.ones(1),
        "component_scale": np.ones(2),
        "node_mask": np.ones(1),
    }
    kwargs[field] = value
    with pytest.raises(ValueError, match=match):
        path_conditioned_tube_metrics(**kwargs)


def _probe_rows(recipient: str, values: tuple[float, ...]) -> list[dict[str, object]]:
    rows = []
    index = 0
    for case in ("case-a", "case-b"):
        for amplitude in (0.5, 1.0):
            for call, donor in ((4, "donor-a"), (12, "donor-b")):
                rows.append(
                    {
                        "recipient_model_id": recipient,
                        "donor_model_id": donor,
                        "case_id": case,
                        "call_index": call,
                        "amplitude_multiplier": amplitude,
                        "input_admissible": True,
                        "input_displacement_scaled_rms": 0.1,
                        "tube_escape_slope": values[index],
                    }
                )
                index += 1
    return rows


def test_tail_score_is_cross_fitted_rectangular_and_equally_weighted() -> None:
    rows = _probe_rows("model-a", tuple(float(value) for value in range(8)))
    rows += _probe_rows("model-b", tuple(float(value) / 2.0 for value in range(8)))
    result = aggregate_tube_escape_tail_score(
        rows, prefix_horizon=20, tail_quantile=0.5
    )

    assert result["donor_model_ids"] == ["donor-a", "donor-b"]
    assert result["probe_count_per_recipient"] == 8
    assert result["quantile_method"] == "linear"
    assert result["models"]["model-a"]["cell_probe_counts"] == {
        "case-a|0.5": 2,
        "case-a|1": 2,
        "case-b|0.5": 2,
        "case-b|1": 2,
    }
    assert result["models"]["model-a"]["tube_escape_tail_score"] == pytest.approx(
        3.5
    )
    assert result["models"]["model-b"]["tube_escape_tail_score"] == pytest.approx(
        1.75
    )


def test_tail_score_rejects_leakage_and_nonrectangular_banks() -> None:
    rows = _probe_rows("model-a", tuple(float(value) for value in range(8)))
    rows += _probe_rows("model-b", tuple(float(value) for value in range(8)))

    leaked = [dict(row) for row in rows]
    leaked[0]["donor_model_id"] = "model-a"
    with pytest.raises(ValueError, match="overlap"):
        aggregate_tube_escape_tail_score(leaked, prefix_horizon=20)

    with pytest.raises(ValueError, match="identical rectangular"):
        aggregate_tube_escape_tail_score(rows[:-1], prefix_horizon=20)

    late = [dict(row) for row in rows]
    late[0]["call_index"] = 21
    with pytest.raises(ValueError, match="prefix horizon"):
        aggregate_tube_escape_tail_score(late, prefix_horizon=20)

    fractional_call = [dict(row) for row in rows]
    fractional_call[0]["call_index"] = 4.5
    with pytest.raises(TypeError, match="integer"):
        aggregate_tube_escape_tail_score(fractional_call, prefix_horizon=20)

    nonstring_identity = [dict(row) for row in rows]
    nonstring_identity[0]["case_id"] = 7
    with pytest.raises(TypeError, match="strings"):
        aggregate_tube_escape_tail_score(nonstring_identity, prefix_horizon=20)


def _paired_inputs() -> dict[str, np.ndarray]:
    # S(x, y) = (x + 1, 3*y + 4), u = (2, -1), eta = (0, 2).
    return {
        "reference_input": _field((2.0, -1.0)),
        "displaced_input": _field((2.0, 1.0)),
        "reference_prediction": _field((3.0, 1.0)),
        "displaced_prediction": _field((3.0, 7.0)),
        "reference_next": _field((3.0, 1.0)),
        "trusted_displaced_next": _field((3.0, 7.0)),
        "node_weights": np.ones(1),
        "component_scale": np.ones(2),
    }


def test_paired_response_distinguishes_exact_dynamics_from_affine_recovery() -> None:
    inputs = _paired_inputs()
    dynamics = paired_solver_response_metrics(**inputs)
    # Recovery map Psi(x, y) = (x + 1, 1) agrees with S on y = -1.
    inputs["displaced_prediction"] = _field((3.0, 1.0))
    recovery = paired_solver_response_metrics(**inputs)

    for result in (dynamics, recovery):
        assert result["schema"] == "paired_solver_response_probe_v1"
        assert result["clean_defect_scaled_rms"] == 0.0
        assert result["input_displacement_scaled_rms"] == pytest.approx(np.sqrt(2))
        assert result["trusted_response_scaled_rms"] == pytest.approx(3 * np.sqrt(2))
        assert result["trusted_secant_gain"] == pytest.approx(3.0)
        for name in (
            "recovery_closure_scaled_rms",
            "dynamics_closure_scaled_rms",
            "response_defect_closure_scaled_rms",
        ):
            assert result[name] == 0.0
    assert dynamics["displaced_dynamics_error_scaled_rms"] == 0.0
    assert dynamics["recovery_target_error_scaled_rms"] == pytest.approx(3 * np.sqrt(2))
    assert dynamics["learned_secant_gain"] == pytest.approx(3.0)
    assert dynamics["response_defect_gain"] == 0.0
    assert dynamics["learned_trusted_response_cosine"] == pytest.approx(1.0)
    assert recovery["recovery_target_error_scaled_rms"] == 0.0
    assert recovery["displaced_dynamics_error_scaled_rms"] == pytest.approx(
        3 * np.sqrt(2)
    )
    assert recovery["learned_secant_gain"] == 0.0
    assert recovery["response_defect_gain"] == pytest.approx(3.0)
    assert recovery["learned_trusted_response_cosine"] is None


def test_paired_response_cancels_common_additive_bias_not_clean_error() -> None:
    inputs = _paired_inputs()
    bias = _field((2.0, -4.0))
    inputs["reference_prediction"] = inputs["reference_prediction"] + bias
    inputs["displaced_prediction"] = inputs["displaced_prediction"] + bias
    result = paired_solver_response_metrics(**inputs)

    assert result["clean_defect_scaled_rms"] == pytest.approx(np.sqrt(10))
    assert result["displaced_dynamics_error_scaled_rms"] == pytest.approx(np.sqrt(10))
    assert result["learned_secant_gain"] == pytest.approx(3.0)
    assert result["response_defect_scaled_rms"] == 0.0
    assert result["clean_response_defect_cosine"] is None


def test_paired_response_retains_signed_alignments() -> None:
    inputs = _paired_inputs()
    inputs.update(
        reference_input=_field((0.0, 0.0)),
        displaced_input=_field((1.0, 0.0)),
        reference_next=_field((0.0, 0.0)),
        trusted_displaced_next=_field((2.0, 0.0)),
        reference_prediction=_field((1.0, 0.0)),
        displaced_prediction=_field((-1.0, 0.0)),
    )
    result = paired_solver_response_metrics(**inputs)
    assert result["learned_trusted_response_scaled_inner"] == pytest.approx(-2.0)
    assert result["learned_trusted_response_cosine"] == pytest.approx(-1.0)
    assert result["clean_response_defect_scaled_inner"] == pytest.approx(-2.0)
    assert result["clean_response_defect_cosine"] == pytest.approx(-1.0)
    assert result["response_defect_gain"] == pytest.approx(4.0)


def test_paired_response_uses_finite_secants_and_does_not_sample() -> None:
    inputs = _paired_inputs()
    # Psi(x) = x**2 and S(x) = x: secant 4 differs from derivative 2 at x=1.
    inputs.update(
        reference_input=_field((1.0, 0.0)),
        displaced_input=_field((3.0, 0.0)),
        reference_next=_field((1.0, 0.0)),
        trusted_displaced_next=_field((3.0, 0.0)),
        reference_prediction=_field((1.0, 0.0)),
        displaced_prediction=_field((9.0, 0.0)),
    )
    before = {name: value.copy() for name, value in inputs.items()}
    result = paired_solver_response_metrics(**inputs)
    assert result == paired_solver_response_metrics(**inputs)
    assert result["learned_secant_gain"] == pytest.approx(4.0)
    assert result["trusted_secant_gain"] == pytest.approx(1.0)
    assert result["response_defect_gain"] == pytest.approx(3.0)
    for name, value in inputs.items():
        np.testing.assert_array_equal(value, before[name])


def test_paired_response_weighting_mask_and_channel_rescaling() -> None:
    inputs = {
        name: np.repeat(value, 3, axis=0)
        for name, value in _paired_inputs().items()
        if name not in ("node_weights", "component_scale")
    }
    inputs["displaced_prediction"] += np.asarray([[2, -2], [4, -4], [100, 100]])
    inputs["node_weights"] = np.asarray([[1.0, 1.0], [3.0, 3.0], [2.0, 2.0]])
    inputs["node_mask"] = np.asarray([1, 1, 0])
    inputs["component_scale"] = np.asarray([1.0, 2.0])
    before = {name: value.copy() for name, value in inputs.items()}
    baseline = paired_solver_response_metrics(**inputs)
    # Weighted scaled dynamics-error energy = (2*(4+1) + 6*(16+4))/(8*2).
    assert baseline["displaced_dynamics_error_scaled_rms"] == pytest.approx(
        np.sqrt(8.125)
    )
    factors = np.asarray([7.0, 0.25])
    scaled_inputs = {
        name: value if name in ("node_weights", "node_mask") else value * factors
        for name, value in inputs.items()
    }
    scaled = paired_solver_response_metrics(**scaled_inputs)
    for name, value in baseline.items():
        if isinstance(value, float):
            assert scaled[name] == pytest.approx(value, abs=1e-14)
        else:
            assert scaled[name] == value
    for name, value in inputs.items():
        np.testing.assert_array_equal(value, before[name])


def test_paired_response_zero_displacement_has_no_resolved_gain() -> None:
    inputs = _paired_inputs()
    inputs["displaced_input"] = inputs["reference_input"].copy()
    inputs["trusted_displaced_next"] = inputs["reference_next"].copy()
    inputs["displaced_prediction"] = inputs["reference_prediction"].copy()
    result = paired_solver_response_metrics(**inputs)
    for name in ("learned_secant_gain", "trusted_secant_gain", "response_defect_gain"):
        assert result[name] is None
    assert result["learned_trusted_response_cosine"] is None
    assert result["input_displacement_scaled_rms"] == 0.0
    assert result["response_defect_scaled_rms"] == 0.0


@pytest.mark.parametrize(
    ("field", "value", "match"),
    (
        ("trusted_displaced_next", np.ones((2, 2)), "identical shapes"),
        ("trusted_displaced_next", np.asarray([1, 2]), "shape"),
        ("trusted_displaced_next", _field((np.nan, 0)), "non-finite"),
        ("reference_input", _field((0, np.inf)), "non-finite"),
        ("node_weights", np.asarray([[1.0, -0.5]]), "nonnegative"),
        ("node_weights", np.asarray([np.inf]), "finite"),
        ("node_weights", np.ones((1, 1, 1)), "shape"),
        ("node_weights", np.ones(2), "shape"),
        ("node_weights", np.zeros(1), "positive mass"),
        ("component_scale", np.asarray([1.0, 0.0]), "strictly positive"),
        ("component_scale", np.asarray([1.0, np.inf]), "finite"),
        ("component_scale", np.ones(1), "one value per channel"),
        ("node_mask", np.asarray([0.5]), "zeros and ones"),
        ("node_mask", np.zeros(1), "positive mass"),
    ),
)
def test_paired_response_rejects_invalid_inputs(
    field: str, value: np.ndarray, match: str
) -> None:
    inputs = _paired_inputs()
    inputs[field] = value
    with pytest.raises(ValueError, match=match):
        paired_solver_response_metrics(**inputs)


def test_paired_response_rejects_nonfinite_arithmetic_from_finite_inputs() -> None:
    inputs = _paired_inputs()
    inputs["displaced_prediction"] = _field((1.0e200, 0.0))
    with (
        np.errstate(over="ignore", invalid="ignore"),
        pytest.raises(ArithmeticError, match="non-finite"),
    ):
        paired_solver_response_metrics(**inputs)


@pytest.mark.parametrize(
    "field",
    (
        "reference_input",
        "displaced_input",
        "reference_prediction",
        "displaced_prediction",
        "reference_next",
        "trusted_displaced_next",
        "node_weights",
        "component_scale",
        "node_mask",
    ),
)
@pytest.mark.parametrize("imaginary_part", (0.0, 1.0, np.nan))
def test_paired_response_rejects_complex_inputs_before_real_conversion(
    field: str, imaginary_part: float
) -> None:
    inputs = _paired_inputs()
    inputs["node_mask"] = np.ones(1)
    value = inputs[field].astype(np.complex128)
    value.imag[:] = imaginary_part
    inputs[field] = value
    with pytest.raises(ValueError, match=f"{field} must be real-valued"):
        paired_solver_response_metrics(**inputs)
