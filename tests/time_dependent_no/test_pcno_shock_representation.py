from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from pcno.pcno import PCNO, compute_Fourier_modes
from scripts.time_dependent_no.analyze_pcno_shock_representation import (
    SCHEMA,
    parse_args,
    run_dry_run,
)
from utility.time_dependent_no.pcno_ripple_diagnostics import trace_pcno_branches
from utility.time_dependent_no.pcno_shock_representation import (
    ANCHORS,
    CASE_FAMILIES,
    DISPLACEMENT,
    DOMAIN_LENGTHS,
    FrontSpec,
    build_structured_cell_grid,
    gradient_distribution_statistics,
    make_translated_front_case,
    pcno_aux_tensors,
    pcno_case_input,
    physical_cosine_filter,
    physical_front_band_mask,
    physical_spectrum_summary,
    restrict_nested_cell_averages,
    scalar_gradient_variants,
    shock_representation_metrics,
    trace_pcno_differential_stages,
)
from utility.time_dependent_no.shock_vortex_coarse_cfd import (
    restrict_uniform_cell_averages,
)


@pytest.fixture(scope="module")
def grid32():
    return build_structured_cell_grid((32, 16))


@pytest.fixture(scope="module")
def grid64():
    return build_structured_cell_grid((64, 32))


@pytest.fixture(scope="module")
def grid128():
    return build_structured_cell_grid((128, 64))


def test_exact_cell_averages_and_known_integrals(grid64) -> None:
    position = ANCHORS[1] + 0.375 * grid64.hx
    step = make_translated_front_case(grid64, "step", position=position)
    pulse = make_translated_front_case(grid64, "pulse", position=position)
    sine = make_translated_front_case(grid64, "smooth_sine", position=position)
    tanh = make_translated_front_case(grid64, "smooth_tanh", position=position)
    volume = grid64.hx * grid64.hy

    assert np.min(step.current) >= 0.0
    assert np.max(step.current) <= 1.0
    assert np.sum(step.increment) * volume == pytest.approx(
        DISPLACEMENT * DOMAIN_LENGTHS[1], abs=2.0e-15
    )
    assert np.sum(pulse.increment) * volume == pytest.approx(0.0, abs=2.0e-15)
    assert np.sum(np.abs(pulse.increment)) * volume == pytest.approx(
        2.0 * DISPLACEMENT * DOMAIN_LENGTHS[1], abs=2.0e-15
    )
    assert np.sum(sine.current) * volume == pytest.approx(0.0, abs=2.0e-15)
    assert np.sum(sine.target) * volume == pytest.approx(0.0, abs=2.0e-15)
    assert np.isfinite(tanh.current).all()
    np.testing.assert_allclose(step.increment, step.target - step.current)
    np.testing.assert_allclose(pulse.increment, pulse.target - pulse.current)


@pytest.mark.parametrize("family", CASE_FAMILIES)
def test_common_physical_cases_close_under_nested_restriction(
    family: str, grid32, grid64, grid128
) -> None:
    position = ANCHORS[1] + 0.375 * (DOMAIN_LENGTHS[0] / 64.0)
    for coarse_grid, fine_grid in ((grid32, grid64), (grid64, grid128)):
        coarse = make_translated_front_case(coarse_grid, family, position=position)
        fine = make_translated_front_case(fine_grid, family, position=position)
        for name in ("current", "target", "increment"):
            restricted = restrict_nested_cell_averages(
                getattr(fine, name),
                fine_resolution=fine_grid.resolution,
                coarse_resolution=coarse_grid.resolution,
            )
            np.testing.assert_allclose(
                restricted, getattr(coarse, name), rtol=0.0, atol=3.0e-14
            )


def test_layout_interoperates_with_maintained_flattened_restriction(
    grid32, grid64
) -> None:
    position = ANCHORS[1] + 0.375 * grid64.hx
    fine = make_translated_front_case(grid64, "pulse", position=position)
    local = restrict_nested_cell_averages(
        fine.increment,
        fine_resolution=grid64.resolution,
        coarse_resolution=grid32.resolution,
    )
    maintained = restrict_uniform_cell_averages(
        fine.increment.reshape(-1, 1),
        target_nx=grid64.nx,
        target_ny=grid64.ny,
        coarse_nx=grid32.nx,
        coarse_ny=grid32.ny,
    ).reshape(grid32.array_shape)

    assert fine.increment.shape == grid64.array_shape
    np.testing.assert_array_equal(
        grid64.nodes[:, 0], np.tile(grid64.x_centers, grid64.ny)
    )
    np.testing.assert_array_equal(local, maintained)


def test_grid_local_phase_cases_are_not_common_source_cases(grid32, grid64) -> None:
    coarse = make_translated_front_case(
        grid32, "step", position=ANCHORS[1] + 0.375 * grid32.hx
    )
    fine = make_translated_front_case(
        grid64, "step", position=ANCHORS[1] + 0.375 * grid64.hx
    )
    restricted = restrict_nested_cell_averages(
        fine.current,
        fine_resolution=grid64.resolution,
        coarse_resolution=grid32.resolution,
    )
    assert np.max(np.abs(restricted - coarse.current)) > 0.1


def test_structured_geometry_and_distributional_gradient_scaling(
    grid32, grid64
) -> None:
    rows = []
    for grid in (grid32, grid64):
        assert np.sum(grid.cell_volumes) == pytest.approx(0.5)
        assert grid.geometry.maximum_coordinate_gradient_error < 1.0e-12
        position = ANCHORS[1]
        case = make_translated_front_case(grid, "step", position=position)
        variants = scalar_gradient_variants(case.current, grid)
        raw = gradient_distribution_statistics(
            variants["raw_gradient"],
            grid,
            (FrontSpec(position, 1.0, 0.0),),
        )
        assert raw["transverse_to_normal_l1_ratio"] < 1.0e-13
        assert raw["front_signed_strength"][0][
            "normalized_signed_strength"
        ] == pytest.approx(1.0, abs=2.0e-14)
        rows.append(raw)
        if grid.resolution == (64, 32):
            np.testing.assert_array_equal(
                variants["local_two_hop_radius_ball"],
                variants["fixed_physical_radius"],
            )

    assert rows[1]["peak"] / rows[0]["peak"] == pytest.approx(2.0)
    assert rows[1]["h_scaled_peak"] == pytest.approx(
        rows[0]["h_scaled_peak"], abs=2.0e-14
    )
    assert rows[1]["support_90_width_in_cells"] == pytest.approx(
        rows[0]["support_90_width_in_cells"]
    )


def test_physical_filter_has_true_bypass_and_preserves_dc(grid64) -> None:
    xx, yy = np.meshgrid(grid64.x_centers, grid64.y_centers, indexing="xy")
    field = (
        1.25
        + np.cos(2.0 * np.pi * 4.0 * xx)
        + 0.15 * np.cos(2.0 * np.pi * 12.0 * xx)
        + 0.25
        * np.cos(2.0 * np.pi * 20.0 * xx)
        * np.cos(2.0 * np.pi * 4.0 * yy / grid64.lengths[1])
    )
    bypass = physical_cosine_filter(field, grid64, kind="zero")
    smooth = physical_cosine_filter(field, grid64, kind="smooth")
    hard = physical_cosine_filter(field, grid64, kind="hard")
    constant = np.full(grid64.array_shape, 7.5)

    assert bypass is field
    np.testing.assert_allclose(
        physical_cosine_filter(constant, grid64, kind="smooth"),
        constant,
        rtol=0.0,
        atol=2.0e-14,
    )
    assert (
        physical_spectrum_summary(smooth, grid64)["high_fraction"]
        < (physical_spectrum_summary(field, grid64)["high_fraction"])
    )
    constant_spectrum = physical_spectrum_summary(constant, grid64)
    assert constant_spectrum["total_energy"] == pytest.approx(
        7.5**2 * np.prod(DOMAIN_LENGTHS), rel=2.0e-15
    )
    common_spectrum = physical_spectrum_summary(
        field, grid64, common_resolution=(32, 16)
    )
    assert common_spectrum["included_mode_count"] == 32 * 16
    assert common_spectrum["common_mode_nx"] == 32
    assert common_spectrum["common_mode_ny"] == 16
    assert not np.allclose(smooth, hard)
    with pytest.raises(ValueError, match="floating-point"):
        physical_cosine_filter(
            np.ones(grid64.array_shape, dtype=np.int64), grid64, kind="smooth"
        )


def test_metrics_separate_bias_ringing_shift_and_blur(grid64) -> None:
    position = ANCHORS[1] + 0.5 * grid64.hx
    case = make_translated_front_case(grid64, "step", position=position)
    exact = shock_representation_metrics(case.increment, case, grid64)

    for key in (
        "relative_increment_l2",
        "normalized_overshoot",
        "normalized_undershoot",
        "oscillatory_mass_outside_front_band",
        "smooth_region_error",
        "positive_total_variation_excess",
        "total_variation_deficit",
        "increment_integral_error",
        "next_state_integral_error",
    ):
        assert exact[key] == pytest.approx(0.0, abs=1.0e-14)
    assert all(row["position_error"] == pytest.approx(0.0) for row in exact["fronts"])
    assert exact["pulse_integral_error"] is None
    pulse = make_translated_front_case(grid64, "pulse", position=position)
    assert shock_representation_metrics(pulse.increment, pulse, grid64)[
        "pulse_integral_error"
    ] == pytest.approx(0.0)

    far = int(np.argmin(np.abs(grid64.x_centers - 0.8)))
    positive_bias = case.increment.copy()
    positive_bias[:, far : far + 2] += 0.1
    bias_metrics = shock_representation_metrics(positive_bias, case, grid64)
    assert bias_metrics["oscillatory_mass_outside_front_band"] == pytest.approx(0.0)
    assert bias_metrics["signed_bias_outside_front_band"] > 0.0
    assert bias_metrics["smooth_region_error"] > 0.0

    ringing = case.increment.copy()
    ringing[:, far] += 0.1
    ringing[:, far + 1] -= 0.1
    ringing_metrics = shock_representation_metrics(ringing, case, grid64)
    assert ringing_metrics["oscillatory_mass_outside_front_band"] > 0.0
    assert ringing_metrics["alternating_lobe_count_outside_front_band"] >= 2
    assert ringing_metrics["signed_bias_outside_front_band"] == pytest.approx(
        0.0, abs=1.0e-14
    )

    transverse_cancellation = case.increment.copy()
    transverse_cancellation[0, far] += 0.1
    transverse_cancellation[1, far] -= 0.1
    transverse_metrics = shock_representation_metrics(
        transverse_cancellation, case, grid64
    )
    assert transverse_metrics["oscillatory_mass_outside_front_band"] > 0.0
    assert transverse_metrics["signed_bias_outside_front_band"] == pytest.approx(
        0.0, abs=1.0e-14
    )

    overshot = case.increment.copy()
    plateau = np.argwhere(case.target == 1.0)[0]
    overshot[tuple(plateau)] += 0.1
    assert shock_representation_metrics(overshot, case, grid64)[
        "normalized_overshoot"
    ] == pytest.approx(0.1)

    shifted_next = make_translated_front_case(
        grid64, "step", position=position + 0.5 * grid64.hx
    ).target
    shifted_increment = shifted_next - case.current
    shifted_metrics = shock_representation_metrics(shifted_increment, case, grid64)
    assert shifted_metrics["fronts"][0]["position_error"] > 0.0

    target_next = case.target
    padded = np.pad(target_next, ((0, 0), (1, 1)), mode="edge")
    blurred_next = 0.25 * padded[:, :-2] + 0.5 * padded[:, 1:-1] + 0.25 * padded[:, 2:]
    blurred_metrics = shock_representation_metrics(
        blurred_next - case.current, case, grid64
    )
    assert blurred_metrics["fronts"][0]["thickness_excess"] > 0.0

    lost_front_metrics = shock_representation_metrics(
        np.zeros_like(case.increment), case, grid64
    )
    lost_front = lost_front_metrics["fronts"][0]
    assert lost_front["front_gate_valid"] is False
    assert lost_front["position_error"] is None
    assert lost_front["predicted_thickness"] is None
    assert lost_front["thickness_excess"] is None
    json.dumps(lost_front_metrics, allow_nan=False)


def test_frozen_trace_closes_every_stage_and_preserves_model() -> None:
    torch.manual_seed(17)
    grid = build_structured_cell_grid((16, 8))
    case = make_translated_front_case(grid, "step", position=ANCHORS[1])
    modes = torch.as_tensor(
        compute_Fourier_modes(2, [1, 1], list(DOMAIN_LENGTHS)),
        dtype=torch.float32,
    )
    model = PCNO(
        ndims=2,
        modes=modes,
        nmeasures=1,
        layers=[4, 4, 4],
        fc_dim=5,
        in_dim=4,
        out_dim=1,
    ).eval()
    aux = pcno_aux_tensors(grid)
    model_input = pcno_case_input(case, grid)
    original = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }

    front_mask = torch.as_tensor(
        physical_front_band_mask(grid, (case.position,)), dtype=torch.bool
    ).unsqueeze(0)
    trace = trace_pcno_differential_stages(
        model, model_input, aux, front_band_mask=front_mask
    )

    torch.testing.assert_close(trace.replay_output, trace.model_output)
    for row in trace.layers:
        differential = model.gws[row.layer]
        torch.testing.assert_close(
            row.pre_softsign, differential.gw1 * row.post_aggregation
        )
        torch.testing.assert_close(row.post_softsign, F.softsign(row.pre_softsign))
        torch.testing.assert_close(
            row.branch_output, differential.gw2(row.post_softsign)
        )
        disabled, _ = trace_pcno_branches(
            model,
            model_input,
            aux,
            disabled_branch=(row.layer, "differential"),
            collect_summaries=False,
        )
        torch.testing.assert_close(
            row.decoded_differential_ablation_response,
            trace.model_output - disabled,
        )
        assert 0.0 <= row.saturation["volume_channel_fraction"] <= 1.0
        assert 0.0 <= row.saturation["feature_energy_fraction"] <= 1.0
        assert 0.0 <= row.saturation["front_band_volume_channel_fraction"] <= 1.0
        assert 0.0 <= row.saturation["front_band_feature_energy_fraction"] <= 1.0
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, original[name], rtol=0.0, atol=0.0)


def test_a1_cli_dry_run_is_no_write_and_non_scientific() -> None:
    with pytest.raises(SystemExit):
        parse_args([])
    result = run_dry_run(((16, 8),))

    assert result["schema"] == SCHEMA
    assert result["scientific_interpretation_allowed"] is False
    assert result["stable_d_series_id_allocated"] is False
    assert result["filter_invariants"]["zero_bypass_bitwise"] is True
    assert result["toy_pcno_trace"]["model_state_unchanged"] is True
    assert result["toy_pcno_trace"]["maximum_replay_error"] < 1.0e-6
    json.dumps(result, allow_nan=False)
