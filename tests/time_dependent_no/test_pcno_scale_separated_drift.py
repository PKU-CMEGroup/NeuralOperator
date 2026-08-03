from __future__ import annotations

import json

import numpy as np
from scipy.fft import idctn

from scripts.time_dependent_no.analyze_pcno_scale_separated_drift import (
    _bundle_contract,
    _replay_row,
)
from scripts.time_dependent_no.visualize_pcno_scale_separated_drift import (
    _region_overlay,
    plot_pod_summary,
)
from utility.time_dependent_no.pcno_scale_separated_drift import (
    dct_band_masks,
    pathway_scale_region_diagnostics,
    project_dct_bands,
    scale_component_diagnostics,
    scale_separated_diagnostics,
    shock_profile_diagnostics,
)


def _mode(
    resolution: tuple[int, int],
    *,
    mode_x: int,
    mode_y: int = 0,
    component: int = 0,
) -> np.ndarray:
    nx, ny = resolution
    coefficients = np.zeros((ny, nx, 4), dtype=np.float64)
    coefficients[mode_y, mode_x, component] = np.sqrt(nx * ny)
    return idctn(coefficients, type=2, axes=(0, 1), norm="ortho").reshape(nx * ny, 4)


def _aggregate_by_band(rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    return {str(row["band"]): row for row in rows}


def _grid(resolution: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    nx, ny = resolution
    x = (np.arange(nx, dtype=np.float64) + 0.5) * 2.0 / nx
    y = (np.arange(ny, dtype=np.float64) + 0.5) / ny
    xx, yy = np.meshgrid(x, y)
    nodes = np.stack((xx.reshape(-1), yy.reshape(-1)), axis=-1)
    volumes = np.full(nx * ny, 2.0 / (nx * ny), dtype=np.float64)
    return nodes, volumes


def _smooth_shock_state(
    nodes: np.ndarray,
    *,
    center: float = 1.0,
    width: float = 0.025,
    jump: float = 0.7,
    gamma: float = 1.4,
) -> np.ndarray:
    profile = 0.5 * (1.0 + np.tanh((nodes[:, 0] - center) / width))
    density = 1.0 + 0.4 * profile
    pressure = 1.0 + jump * profile
    velocity_x = np.zeros_like(density)
    velocity_y = np.zeros_like(density)
    energy = pressure / (gamma - 1.0)
    return np.stack(
        (density, density * velocity_x, density * velocity_y, energy), axis=-1
    )


def test_dct_masks_partition_modes_and_projection_reconstructs() -> None:
    resolution = (128, 64)
    masks = dct_band_masks(resolution)
    coverage = sum(mask.astype(np.int8) for mask in masks.values())
    assert np.all(coverage == 1)
    assert all(np.any(mask) for mask in masks.values())

    rng = np.random.default_rng(7)
    defects = rng.standard_normal((3, resolution[0] * resolution[1], 4))
    projections, closure = project_dct_bands(
        defects,
        resolution=resolution,
        component_scale=np.ones(4),
    )
    reconstructed = sum(projections.values())
    assert np.max(np.abs(reconstructed - defects)) < 1.0e-12
    assert closure["maximum_reconstruction_abs_residual_scaled"] < 1.0e-12
    assert closure["maximum_instantaneous_energy_relative_closure"] < 1.0e-12


def test_pure_physical_modes_land_in_registered_bands() -> None:
    resolution = (128, 64)
    field = (
        _mode(resolution, mode_x=4)
        + _mode(resolution, mode_x=40, component=1)
        + _mode(resolution, mode_x=100, component=2)
    )
    projections, _ = project_dct_bands(
        field[None, ...],
        resolution=resolution,
        component_scale=np.ones(4),
    )
    assert np.linalg.norm(projections["large"][0, :, 0]) > 0.99
    assert np.linalg.norm(projections["transition"][0, :, 1]) > 0.99
    assert np.linalg.norm(projections["local"][0, :, 2]) > 0.99
    assert np.linalg.norm(projections["local"][0, :, 0]) < 1.0e-12


def test_large_drift_accumulates_while_local_noise_cancels() -> None:
    resolution = (128, 64)
    large = _mode(resolution, mode_x=4)
    local = _mode(resolution, mode_x=100, component=1)
    signs = np.asarray((1.0, -1.0, 1.0, -1.0))
    defects = np.stack([large + sign * local for sign in signs])
    truth = np.repeat((2.0 * large)[None, ...], defects.shape[0], axis=0)
    time_rows, aggregate_rows, closure, _ = scale_separated_diagnostics(
        defects,
        truth,
        resolution=resolution,
        component_scale=np.ones(4),
        max_lag=3,
    )
    aggregate = _aggregate_by_band(aggregate_rows)

    assert np.isclose(aggregate["large"]["path_energy_share"], 0.5)
    assert np.isclose(aggregate["local"]["path_energy_share"], 0.5)
    assert np.isclose(aggregate["large"]["endpoint_energy_share"], 1.0)
    assert np.isclose(aggregate["local"]["endpoint_energy_share"], 0.0)
    assert np.isclose(aggregate["large"]["temporal_coherence"], 1.0)
    assert np.isclose(aggregate["large"]["coherent_mean_fraction"], 1.0)
    assert np.isclose(aggregate["local"]["temporal_coherence"], 0.0, atol=1e-14)
    assert np.isclose(aggregate["local"]["coherent_mean_fraction"], 0.0, atol=1e-14)
    assert np.isclose(
        aggregate["local"]["aggregate_signed_interaction_over_defect_energy"],
        -1.0,
    )
    assert np.isclose(aggregate["total"]["residual_error_rms_over_time"], np.sqrt(2.0))
    assert np.isclose(
        aggregate["total"]["aggregate_relative_residual_energy"], np.sqrt(0.5)
    )
    local_lags = aggregate["local"]["lag_correlations"]
    local_uncentered_lags = aggregate["local"]["uncentered_lag_correlations"]
    assert isinstance(local_lags, list)
    assert np.isclose(local_lags[0]["correlation"], -1.0)
    assert np.isclose(local_uncentered_lags[0]["correlation"], -1.0)
    assert aggregate["large"]["lag_correlations"][0]["correlation"] is None
    assert closure["maximum_cumulative_energy_relative_closure"] < 1.0e-12
    assert (
        max(
            value
            for key, value in closure.items()
            if key.endswith("maximum_signed_growth_absolute_closure")
        )
        < 1.0e-12
    )

    endpoint_rows = {
        str(row["band"]): row for row in time_rows if row["step"] == defects.shape[0]
    }
    assert np.isclose(endpoint_rows["large"]["cumulative_energy_share"], 1.0)
    assert np.isclose(endpoint_rows["local"]["cumulative_energy_share"], 0.0)
    assert np.isclose(
        sum(
            float(row["signed_growth_over_total_path_energy"])
            for row in time_rows
            if row["band"] == "total"
        ),
        float(aggregate["total"]["endpoint_energy"])
        / float(aggregate["total"]["path_energy"]),
    )


def test_component_band_partition_and_signed_growth() -> None:
    resolution = (128, 64)
    large_rho = _mode(resolution, mode_x=4, component=0)
    local_rho_u = _mode(resolution, mode_x=100, component=1)
    signs = np.asarray((1.0, -1.0, 1.0, -1.0))
    defects = np.stack([large_rho + sign * local_rho_u for sign in signs])
    truth = np.repeat((2.0 * large_rho)[None, ...], defects.shape[0], axis=0)
    _, _, _, projections = scale_separated_diagnostics(
        defects,
        truth,
        resolution=resolution,
        component_scale=np.ones(4),
        max_lag=3,
        return_projections=True,
    )
    assert projections is not None
    time_rows, aggregate_rows, closure = scale_component_diagnostics(
        defects,
        truth,
        projections,
        component_scale=np.ones(4),
        max_lag=3,
    )
    aggregate = {
        (str(row["band"]), str(row["component"])): row for row in aggregate_rows
    }

    large = aggregate["large", "rho"]
    local = aggregate["local", "rho_u"]
    assert np.isclose(large["path_energy_share"], 0.5)
    assert np.isclose(local["path_energy_share"], 0.5)
    assert np.isclose(large["path_band_share_within_component"], 1.0)
    assert np.isclose(local["path_band_share_within_component"], 1.0)
    assert np.isclose(large["endpoint_energy_share"], 1.0)
    assert np.isclose(local["endpoint_energy_share"], 0.0)
    assert np.isclose(large["temporal_coherence"], 1.0)
    assert np.isclose(local["temporal_coherence"], 0.0, atol=1.0e-14)
    assert np.isclose(large["aggregate_signed_interaction_over_defect_energy"], 3.0)
    assert np.isclose(local["aggregate_signed_interaction_over_defect_energy"], -1.0)
    assert len(time_rows) == 4 * 4 * defects.shape[0]
    assert (
        closure["maximum_component_band_reconstruction_abs_residual_scaled"] < 1.0e-12
    )
    assert (
        closure["maximum_component_band_instantaneous_energy_relative_closure"]
        < 1.0e-12
    )
    assert (
        closure["maximum_component_band_cumulative_energy_relative_closure"] < 1.0e-12
    )
    assert closure["maximum_component_band_signed_growth_absolute_closure"] < 1.0e-12


def test_pathway_band_region_attribution_closes() -> None:
    resolution = (128, 64)
    nodes, volumes = _grid(resolution)
    large_mesh = _mode(resolution, mode_x=4)
    local_state = _mode(resolution, mode_x=100, component=1)
    signs = np.asarray((1.0, -1.0, 1.0, -1.0))
    mesh = np.repeat(large_mesh[None, ...], len(signs), axis=0)
    state = np.stack([sign * local_state for sign in signs])
    total = mesh + state
    reference_state = _smooth_shock_state(nodes)
    reference = np.repeat(reference_state[None, ...], len(signs) + 1, axis=0)
    _, aggregate_rows, closure, _ = pathway_scale_region_diagnostics(
        total,
        mesh,
        state,
        reference,
        nodes,
        volumes,
        resolution=resolution,
        component_scale=np.ones(4),
        gamma=1.4,
    )
    aggregate = {(str(row["band"]), str(row["region"])): row for row in aggregate_rows}
    large = aggregate["large", "all"]
    local = aggregate["local", "all"]
    assert np.isclose(large["path_mesh_symmetric_attribution_share"], 1.0)
    assert np.isclose(local["path_state_symmetric_attribution_share"], 1.0)
    assert np.isclose(local["endpoint_total_energy"], 0.0, atol=1.0e-13)
    assert local["negative_growth_fraction_state"] > 0.0
    assert max(closure.values()) < 1.0e-10


def test_shock_profile_separates_translation_from_strength() -> None:
    resolution = (128, 64)
    nodes, volumes = _grid(resolution)
    reference_state = _smooth_shock_state(nodes)
    shifted_state = _smooth_shock_state(nodes, center=1.01)
    reference = np.repeat(reference_state[None, ...], 3, axis=0)
    coarse = np.repeat(shifted_state[None, ...], 3, axis=0)
    fine = reference.copy()
    time_rows, aggregate_rows = shock_profile_diagnostics(
        reference,
        coarse,
        fine,
        nodes,
        volumes,
        resolution=resolution,
        gamma=1.4,
        component_scale=np.ones(4),
    )
    coarse_rows = [
        row for row in time_rows if row["comparison"] == "coarse_vs_reference"
    ]
    fine_rows = [
        row for row in time_rows if row["comparison"] == "restricted_fine_vs_reference"
    ]
    assert 0.005 < coarse_rows[-1]["rms_shock_position_error"] < 0.02
    assert coarse_rows[-1]["translation_only_energy_fraction"] > 0.5
    assert np.isclose(fine_rows[-1]["rms_shock_position_error"], 0.0)
    aggregate = {row["comparison"]: row for row in aggregate_rows}
    assert aggregate["coarse_vs_reference"]["median_rms_shock_position_error"] > 0.005


def test_replay_metric_uses_the_frozen_residual_scale() -> None:
    parent = np.ones((2, 8, 4), dtype=np.float64)
    replay = parent.copy()
    replay[:, :, 0] += 0.01
    row = _replay_row(
        parent,
        replay,
        field="delta_free",
        residual_scale=np.asarray((0.1, 1.0, 1.0, 1.0)),
    )
    assert np.isclose(row["maximum_absolute_difference"], 0.01)
    assert np.isclose(row["maximum_step_residual_scaled_rms"], 0.1)


def test_unified_bundle_exposes_only_the_self_consistent_free_sequence() -> None:
    metadata = np.asarray(
        json.dumps(
            {
                "schema": "pcno_residual_structure_diagnostic_v1",
                "case_id": "sv_fixture",
                "checkpoint_sha256": "checkpoint",
                "bundle_payload": "unified",
            }
        )
    )
    case_id, checkpoint, times, scale, prefixes = _bundle_contract(
        {
            "metadata_json": metadata,
            "physical_times": np.asarray((0.02, 0.04)),
            "residual_scale": np.ones(4),
        },
        input_schema="pcno_residual_structure_diagnostic_v1",
    )
    assert case_id == "sv_fixture"
    assert checkpoint == "checkpoint"
    assert np.array_equal(times, np.asarray((0.02, 0.04)))
    assert np.array_equal(scale, np.ones(4))
    assert prefixes == {"free_cross_grid_defect": "delta_free__"}


def test_region_overlay_uses_reference_physical_masks() -> None:
    resolution = (128, 64)
    nodes, _ = _grid(resolution)
    overlay = _region_overlay(
        _smooth_shock_state(nodes),
        nodes,
        resolution=resolution,
        gamma=1.4,
    )
    assert overlay.shape == (resolution[1], resolution[0], 4)
    assert np.any(np.isclose(overlay[..., 3], 0.12))
    assert np.any(np.isclose(overlay[..., 3], 0.16))


def test_pod_summary_plot_emits_png_and_pdf(tmp_path) -> None:
    rows = []
    for target_index, target in enumerate(("125x50->250x100", "250x100->500x200")):
        for band_index, band in enumerate(("total", "large", "transition", "local")):
            for case_index in range(2):
                rows.append(
                    {
                        "target": target,
                        "band": band,
                        "uncentered_first_mode_energy_fraction": str(
                            0.2 + 0.05 * band_index + 0.01 * case_index
                        ),
                        "uncentered_first_three_energy_fraction": str(
                            0.5 + 0.05 * band_index + 0.01 * case_index
                        ),
                        "uncentered_modes_for_95_percent": str(
                            8 + target_index + band_index + case_index
                        ),
                        "centered_modes_for_95_percent": str(
                            9 + target_index + band_index + case_index
                        ),
                    }
                )
    plot_pod_summary(rows, tmp_path, sequence_label="fixture")
    assert (tmp_path / "pod_temporal_rank_summary.png").is_file()
    assert (tmp_path / "pod_temporal_rank_summary.pdf").is_file()
