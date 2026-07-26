from __future__ import annotations

import numpy as np
import pytest

from scripts.time_dependent_no.evaluate_pcno_euler2d_front_fitted_oracle import (
    STRUCTURE_METRICS,
    _pushforward_row,
    conservative_front_warp,
    fit_two_sided_strength_contrast,
    front_curve_from_state,
    front_curve_metrics,
    promotion_decision,
    row_total_conservation,
    validate_oriented_face_geometry,
    validate_uniform_row_major_grid,
)
from utility.time_dependent_no.shock_vortex_fv import (
    ShockVortexFVConfig,
    make_structured_fv_geometry,
)

GAMMA = 1.4


def _config(nx: int = 40, ny: int = 6) -> ShockVortexFVConfig:
    return ShockVortexFVConfig(
        nx=nx,
        ny=ny,
        coarse_nx=nx,
        coarse_ny=ny,
        output_times=(0.0, 0.6),
        t_final=0.6,
    )


def _grid(nx: int = 40, ny: int = 6):
    config = _config(nx, ny)
    geometry = make_structured_fv_geometry(config)
    grid = validate_uniform_row_major_grid(
        geometry.cell_centers,
        geometry.cell_volume,
        np.arange(nx * ny),
        coordinate_convention="FV_row_major_cell_centers_xy",
        reference_config=config.to_dict(),
    )
    return config, geometry, grid


def _pressure_step(grid, front: np.ndarray) -> np.ndarray:
    curve = np.asarray(front, dtype=np.float64).reshape(grid.ny)
    pressure = np.where(grid.x[None, :] < curve[:, None], 1.0, 1.4)
    density = np.where(grid.x[None, :] < curve[:, None], 1.0, 1.15)
    flat_pressure = pressure.reshape(-1)
    flat_density = density.reshape(-1)
    return np.stack(
        (
            flat_density,
            np.zeros_like(flat_density),
            np.zeros_like(flat_density),
            flat_pressure / (GAMMA - 1.0),
        ),
        axis=-1,
    )


def test_grid_and_face_contract_close_on_canonical_geometry() -> None:
    config, geometry, grid = _grid(nx=8, ny=6)
    result = validate_oriented_face_geometry(
        {
            "positions": geometry.cell_centers,
            "face_centers": geometry.face_centers,
            "face_measures": geometry.face_measure,
            "face_normals": geometry.face_normal,
            "face_owner": geometry.face_owner,
            "face_neighbor": geometry.face_neighbor,
            "face_boundary_tag": geometry.face_boundary_tag,
        },
        grid,
    )

    assert result["maximum_face_normal_unit_error"] < 1.0e-14
    assert result["maximum_cell_area_vector_closure"] < 1.0e-14
    assert result["face_count"] == pytest.approx(
        (config.coarse_nx + 1) * config.coarse_ny
        + config.coarse_nx * (config.coarse_ny + 1)
    )


def test_grid_contract_rejects_nonidentity_order() -> None:
    config, geometry, _ = _grid(nx=8, ny=6)
    positions = geometry.cell_centers.copy()
    positions[[0, 1]] = positions[[1, 0]]
    with pytest.raises(ValueError, match="row-major"):
        validate_uniform_row_major_grid(
            positions,
            geometry.cell_volume,
            np.arange(positions.shape[0]),
            coordinate_convention="FV_row_major_cell_centers_xy",
            reference_config=config.to_dict(),
        )


def test_conservative_front_warp_moves_phase_and_preserves_every_row() -> None:
    _, geometry, grid = _grid()
    predicted_state = _pressure_step(grid, np.full(grid.ny, 0.45))
    target_state = _pressure_step(grid, np.full(grid.ny, 0.55))
    predicted_front = front_curve_from_state(
        predicted_state, grid, gamma=GAMMA, shock_x=0.5
    )["curve"]
    target_front = front_curve_from_state(target_state, grid, gamma=GAMMA, shock_x=0.5)[
        "curve"
    ]

    warped, diagnostics = conservative_front_warp(
        predicted_state, grid, predicted_front, target_front
    )
    conservation = row_total_conservation(
        predicted_state,
        warped,
        geometry.cell_volume,
        np.ones(4),
        grid,
    )
    baseline_error = front_curve_metrics(
        predicted_state,
        target_state,
        grid,
        gamma=GAMMA,
        shock_x=0.5,
        target_curve=target_front,
    )["mean_absolute_error"]
    warped_error = front_curve_metrics(
        warped,
        target_state,
        grid,
        gamma=GAMMA,
        shock_x=0.5,
        target_curve=target_front,
    )["mean_absolute_error"]

    assert conservation["maximum_relative_row_component_defect"] < 1.0e-12
    assert diagnostics["mean_absolute_displacement_cells"] > 1.0
    assert diagnostics["minimum_warp_jacobian"] > 0.0
    assert warped_error < baseline_error


def test_pushforward_splits_a_source_cell_at_the_piecewise_map_kink() -> None:
    edges = np.linspace(0.0, 1.0, 5)
    state = np.zeros((4, 4), dtype=np.float64)
    state[1] = np.asarray([1.0, 2.0, 3.0, 4.0])

    pushed, _ = _pushforward_row(
        state,
        edges,
        x_pred=0.4,
        x_target=0.6,
    )

    expected_density = np.asarray([0.0, 1.0 / 3.0, 2.0 / 3.0, 0.0])
    expected = expected_density[:, None] * state[1][None, :]
    np.testing.assert_allclose(pushed, expected, atol=1.0e-14, rtol=0.0)
    np.testing.assert_allclose(
        np.sum(0.25 * pushed, axis=0),
        0.25 * state[1],
        atol=1.0e-14,
        rtol=0.0,
    )


def test_strength_contrast_recovers_four_dofs_with_rowwise_zero_total() -> None:
    _, geometry, grid = _grid(nx=20, ny=6)
    curve = np.asarray([0.46, 0.48, 0.49, 0.51, 0.52, 0.54])
    phase = _pressure_step(grid, curve)
    volume = geometry.cell_volume.reshape(grid.ny, grid.nx)
    left = grid.x[None, :] < curve[:, None]
    left_volume = np.sum(volume * left, axis=1)
    right_volume = np.sum(volume * ~left, axis=1)
    contrast = np.where(left, 1.0, -(left_volume / right_volume)[:, None]).reshape(-1)
    coefficient = np.asarray([0.01, -0.02, 0.005, 0.03])
    target = phase + contrast[:, None] * coefficient[None, :]

    candidate, diagnostics = fit_two_sided_strength_contrast(
        phase, target, grid, curve, geometry.cell_volume
    )
    conservation = row_total_conservation(
        phase, candidate, geometry.cell_volume, np.ones(4), grid
    )

    np.testing.assert_allclose(
        diagnostics["strength_coefficient"], coefficient, atol=1.0e-14
    )
    np.testing.assert_allclose(candidate, target, atol=1.0e-14)
    assert conservation["maximum_relative_row_component_defect"] < 1.0e-13
    assert diagnostics["active_cell_fraction"] == pytest.approx(1.0)


def _promotion_rows() -> list[dict[str, object]]:
    rows = []
    for trajectory in range(6):
        for call in (15, 30):
            acceptance = {
                name: {"accepted": True, "baseline": 1.0, "candidate": 1.0}
                for name in STRUCTURE_METRICS
            }
            conservation = {"maximum_relative_row_component_defect": 1.0e-14}
            rows.append(
                {
                    "trajectory": f"case_{trajectory}",
                    "call": call,
                    "all_variants_raw_admissible": True,
                    "phase_conservation": conservation,
                    "primary_conservation": conservation,
                    "primary_state_reduction": 0.20,
                    "primary_front_curve_reduction": 0.60,
                    "primary_highpass_reduction": 0.05,
                    "primary_joint_state_front_highpass_nonworse": True,
                    "primary_structure_acceptance": {
                        "passed": True,
                        "metrics": acceptance,
                    },
                    "phase_state_reduction": 0.10,
                    "phase_front_curve_reduction": 0.55,
                }
            )
    return rows


def test_promotion_decision_is_conjunctive() -> None:
    rows = _promotion_rows()
    passed = promotion_decision(rows, expected_trajectories=6)
    assert passed["passed"]
    assert passed["decision"].endswith("contract_only")

    for row in rows:
        if row["call"] == 30:
            row["primary_highpass_reduction"] = -0.01
    failed = promotion_decision(rows, expected_trajectories=6)
    assert not failed["passed"]
    assert not failed["gates"]["h60_median_smooth_highpass_does_not_increase"]
