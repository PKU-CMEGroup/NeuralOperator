from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from pcno.pcno import compute_gradient
from scripts.time_dependent_no.evaluate_pcno_resolution_rollout import (
    _aggregate,
    _commutator_row,
    _pressure_profile_shock_metrics,
    _reference_at_resolution,
    _step_stride,
)
from scripts.time_dependent_no.evaluate_pcno_resolution_transfer import main
from utility.time_dependent_no.pcno_euler2d import (
    Euler2DNormalization,
    PCNOEuler2DResidual,
)
from utility.time_dependent_no.pcno_resolution_transfer import (
    build_resolution_geometry,
    commutator_metrics,
    initial_state_for_model_grid,
    initial_states_from_common_source,
    node_type_scaling_summary,
    node_types_for_protocol,
    parse_resolution,
    physical_wavelength_band_metrics,
    restrict_nested_state,
)
from utility.time_dependent_no.shock_vortex_fv import ShockVortexFVConfig

ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def test_resolution_parsing_is_explicit() -> None:
    assert parse_resolution("125x50") == (125, 50)
    assert parse_resolution("12X8") == (12, 8)
    with pytest.raises(ValueError, match="NXxNY"):
        parse_resolution("125")
    with pytest.raises(ValueError, match="at least two"):
        parse_resolution("1x8")


def test_dynamic_node_types_have_expected_counts_and_volume_scaling() -> None:
    base = ShockVortexFVConfig()
    config, geometry = build_resolution_geometry(base, (10, 8))
    physical = node_types_for_protocol(
        geometry,
        config,
        "physical",
        training_resolution=(20, 16),
    )
    summary = node_type_scaling_summary(geometry, config, physical)

    assert summary["counts"] == {"0": 48, "1": 16, "2": 12, "3": 4}
    assert summary["tagged_count"] == 32
    assert summary["tagged_fraction"] == pytest.approx(0.4)
    assert summary["tagged_physical_volume"] == pytest.approx(0.8)
    assert summary["tagged_normalized_volume_mass"] == pytest.approx(0.4)
    assert summary["x_tagged_width_per_side"] == pytest.approx(0.2)
    assert summary["y_tagged_width_per_side"] == pytest.approx(0.125)
    assert summary["differential_input_support_hops_upper_bound"] == 3
    assert summary["differential_input_support_physical_upper_bound"] == (
        pytest.approx(0.6)
    )

    all_normal = node_types_for_protocol(
        geometry,
        config,
        "all_normal",
        training_resolution=(20, 16),
    )
    assert not bool(all_normal.any())
    swapped = node_types_for_protocol(
        geometry,
        config,
        "swapped_boundary_kinds",
        training_resolution=(20, 16),
    )
    assert np.array_equal(swapped == 0, physical == 0)
    assert np.array_equal(swapped == 3, physical == 3)
    assert np.array_equal(swapped == 1, physical == 2)
    assert np.array_equal(swapped == 2, physical == 1)


def test_training_band_keeps_width_when_target_grid_can_resolve_it() -> None:
    base = ShockVortexFVConfig()
    training_resolution = (20, 16)
    training_config, training_geometry = build_resolution_geometry(
        base, training_resolution
    )
    fine_config, fine_geometry = build_resolution_geometry(base, (40, 32))
    training_types = node_types_for_protocol(
        training_geometry,
        training_config,
        "training_band",
        training_resolution=training_resolution,
    )
    fine_types = node_types_for_protocol(
        fine_geometry,
        fine_config,
        "training_band",
        training_resolution=training_resolution,
    )
    training_summary = node_type_scaling_summary(
        training_geometry, training_config, training_types
    )
    fine_summary = node_type_scaling_summary(fine_geometry, fine_config, fine_types)

    assert training_summary["x_tagged_width_per_side"] == pytest.approx(0.1)
    assert fine_summary["x_tagged_width_per_side"] == pytest.approx(0.1)
    assert training_summary["y_tagged_width_per_side"] == pytest.approx(0.0625)
    assert fine_summary["y_tagged_width_per_side"] == pytest.approx(0.0625)
    assert fine_summary["tagged_physical_volume"] == pytest.approx(
        training_summary["tagged_physical_volume"]
    )
    assert fine_summary["counts"]["2"] > training_summary["counts"]["2"]


def test_conservative_restriction_and_update_commutator() -> None:
    fine_resolution = (4, 4)
    coarse_resolution = (2, 2)
    fine_current = np.arange(4 * 4 * 2, dtype=np.float64).reshape(16, 2)
    coarse_current = restrict_nested_state(
        fine_current,
        fine_resolution=fine_resolution,
        coarse_resolution=coarse_resolution,
    )
    update = np.asarray([0.25, -0.5], dtype=np.float64)
    fine_prediction = fine_current + update
    coarse_prediction = coarse_current + update
    metrics = commutator_metrics(
        coarse_current=coarse_current,
        coarse_prediction=coarse_prediction,
        fine_current=fine_current,
        fine_prediction=fine_prediction,
        coarse_resolution=coarse_resolution,
        fine_resolution=fine_resolution,
        coarse_volumes=np.full((4, 1), 0.5),
        state_scale=np.ones(2),
        residual_scale=np.ones(2),
    )

    assert metrics["input_restriction_gap_scaled_rms"] == pytest.approx(0.0)
    assert metrics["prediction_commutator_scaled_rms"] == pytest.approx(0.0)
    assert metrics["update_commutator_scaled_rms"] == pytest.approx(0.0)
    assert metrics["update_commutator_relative_to_fine_update"] == pytest.approx(0.0)


def test_common_source_initialization_restricts_conservative_cell_averages() -> None:
    base = ShockVortexFVConfig(initial_quadrature_order=4)
    source_resolution = (12, 12)
    targets = ((6, 6), source_resolution)
    source, states = initial_states_from_common_source(
        base,
        targets,
        source_resolution=source_resolution,
        quadrature_order=4,
    )

    assert np.array_equal(states[source_resolution], source)
    expected_coarse = restrict_nested_state(
        source,
        fine_resolution=source_resolution,
        coarse_resolution=(6, 6),
    )
    assert np.array_equal(states[(6, 6)], expected_coarse)
    direct_coarse = initial_state_for_model_grid(base, (6, 6), quadrature_order=4)
    # Direct coarse quadrature is a different representation protocol and need
    # not equal restriction of a common fine quadrature at low resolution.
    assert np.max(np.abs(direct_coarse - expected_coarse)) > 1.0e-5
    assert np.mean(source, axis=0) == pytest.approx(
        np.mean(expected_coarse, axis=0), abs=1.0e-14
    )

    with pytest.raises(ValueError, match="divide the common source"):
        initial_states_from_common_source(
            base,
            ((5, 6),),
            source_resolution=source_resolution,
            quadrature_order=4,
        )


def test_rebuilt_least_squares_gradient_is_affine_exact_across_grids() -> None:
    base = ShockVortexFVConfig()
    for resolution in ((6, 6), (12, 12)):
        _, geometry = build_resolution_geometry(base, resolution)
        nodes = torch.as_tensor(geometry.nodes, dtype=torch.float64)
        affine = 2.0 * nodes[:, 0] - 3.0 * nodes[:, 1] + 0.7
        gradient = compute_gradient(
            affine.reshape(1, 1, -1),
            torch.as_tensor(geometry.directed_edges).unsqueeze(0),
            torch.as_tensor(
                geometry.edge_gradient_weights, dtype=torch.float64
            ).unsqueeze(0),
        )
        expected = torch.tensor([2.0, -3.0], dtype=torch.float64).reshape(1, 2, 1)
        assert torch.max(torch.abs(gradient - expected)).item() < 1.0e-11


def test_reference_restriction_is_available_only_at_or_below_native_grid() -> None:
    native = np.arange(4 * 4 * 2, dtype=np.float64).reshape(16, 2)
    coarse = _reference_at_resolution(
        native,
        reference_resolution=(4, 4),
        target_resolution=(2, 2),
    )
    assert np.array_equal(
        coarse,
        restrict_nested_state(
            native,
            fine_resolution=(4, 4),
            coarse_resolution=(2, 2),
        ),
    )
    assert (
        _reference_at_resolution(
            native,
            reference_resolution=(4, 4),
            target_resolution=(8, 8),
        )
        is None
    )


def test_pressure_profile_shock_metrics_report_physical_and_cell_width() -> None:
    nx, ny = 8, 4
    gamma = 1.4
    pressure = np.ones((ny, nx), dtype=np.float64)
    pressure[:, nx // 2 :] = 2.0
    state = np.zeros((ny, nx, 4), dtype=np.float64)
    state[..., 0] = 1.0
    state[..., 3] = pressure / (gamma - 1.0)
    flattened = state.reshape(nx * ny, 4)
    metrics = _pressure_profile_shock_metrics(
        flattened,
        flattened,
        resolution=(nx, ny),
        x_min=0.0,
        x_max=2.0,
        gamma=gamma,
    )

    assert metrics["shock_position_absolute_error"] == pytest.approx(0.0)
    assert metrics["prediction_shock_position_x"] == pytest.approx(1.0)
    assert metrics["prediction_shock_thickness_cells"] == 1
    assert metrics["prediction_shock_thickness_physical"] == pytest.approx(0.25)
    assert metrics["pressure_profile_shock_thickness_ratio"] == pytest.approx(1.0)


def test_physical_wavelength_band_is_resolution_independent() -> None:
    def field(resolution: tuple[int, int]) -> np.ndarray:
        nx, ny = resolution
        x = (np.arange(nx, dtype=np.float64) + 0.5) * (2.0 / nx)
        wave = np.sin(2.0 * np.pi * 10.0 * x / 2.0)
        return np.broadcast_to(wave[None, :, None], (ny, nx, 1)).reshape(-1, 1)

    rows = []
    for resolution in ((32, 16), (64, 32)):
        reference = field(resolution)
        prediction = 1.25 * reference
        rows.append(
            physical_wavelength_band_metrics(
                prediction,
                reference,
                resolution=resolution,
                domain_lengths=(2.0, 1.0),
                component_scale=np.ones(1),
                wavelength_min=0.125,
                wavelength_max=0.25,
            )
        )
    assert rows[0]["relative_spectral_l2"] == pytest.approx(0.25)
    assert rows[1]["relative_spectral_l2"] == pytest.approx(0.25)


def test_checkpoint_step_stride_declarations_must_agree() -> None:
    checkpoint = {
        "step_stride": 2,
        "training_args": {"step_stride": 2},
        "data_contract": {},
    }
    assert _step_stride(checkpoint) == 2
    checkpoint["training_args"]["step_stride"] = 1
    with pytest.raises(ValueError, match="declarations disagree"):
        _step_stride(checkpoint)


def test_rollout_aggregate_counts_resolution_specific_noncompletion() -> None:
    state_rows = [
        {
            "case_id": "a",
            "case_split": "validation",
            "node_type_protocol": "physical",
            "resolution": "125x50",
            "call": 2,
            "scaled_relative_l2_physical_volume": 0.1,
        },
        {
            "case_id": "a",
            "case_split": "validation",
            "node_type_protocol": "physical",
            "resolution": "250x100",
            "call": 2,
            "scaled_relative_l2_physical_volume": 0.2,
        },
        {
            "case_id": "b",
            "case_split": "validation",
            "node_type_protocol": "physical",
            "resolution": "250x100",
            "call": 2,
            "scaled_relative_l2_physical_volume": 0.3,
        },
    ]
    completion_rows = [
        {
            "case_id": "a",
            "case_split": "validation",
            "node_type_protocol": "physical",
            "resolution": "125x50",
            "full_completion": True,
            "all_completed_calls_admissible": True,
        },
        {
            "case_id": "b",
            "case_split": "validation",
            "node_type_protocol": "physical",
            "resolution": "125x50",
            "full_completion": False,
            "all_completed_calls_admissible": False,
        },
        {
            "case_id": "a",
            "case_split": "validation",
            "node_type_protocol": "physical",
            "resolution": "250x100",
            "full_completion": True,
            "all_completed_calls_admissible": True,
        },
        {
            "case_id": "b",
            "case_split": "validation",
            "node_type_protocol": "physical",
            "resolution": "250x100",
            "full_completion": True,
            "all_completed_calls_admissible": False,
        },
    ]
    teacher_state_rows = [
        {
            "case_id": "a",
            "case_split": "validation",
            "node_type_protocol": "physical",
            "resolution": "125x50",
            "call": 2,
            "finite": True,
            "admissible": True,
            "scaled_relative_l2_physical_volume": 0.04,
        },
        {
            "case_id": "b",
            "case_split": "validation",
            "node_type_protocol": "physical",
            "resolution": "125x50",
            "call": 2,
            "finite": True,
            "admissible": False,
            "scaled_relative_l2_physical_volume": 0.06,
        },
    ]

    aggregate = _aggregate(
        state_rows,
        [],
        completion_rows,
        teacher_state_rows=teacher_state_rows,
        final_call=2,
    )
    by_resolution = {row["resolution"]: row for row in aggregate["final_state"]}
    coarse = by_resolution["125x50"]
    assert coarse["case_count"] == 2
    assert coarse["final_state_case_count"] == 1
    assert coarse["completion_fraction"] == pytest.approx(0.5)
    assert coarse["admissible_fraction_of_cases"] == pytest.approx(0.5)
    assert coarse["mean_scaled_relative_l2_physical_volume"] == pytest.approx(0.1)
    native = by_resolution["250x100"]
    assert native["case_count"] == 2
    assert native["final_state_case_count"] == 2
    assert native["completion_fraction"] == pytest.approx(1.0)
    assert native["admissible_fraction_of_cases"] == pytest.approx(0.5)
    assert native["mean_scaled_relative_l2_physical_volume"] == pytest.approx(0.25)
    teacher = aggregate["final_teacher_forced_state"][0]
    assert teacher["case_count"] == 2
    assert teacher["finite_fraction"] == pytest.approx(1.0)
    assert teacher["admissible_fraction"] == pytest.approx(0.5)
    assert teacher["mean_scaled_relative_l2_physical_volume"] == pytest.approx(0.05)


def test_commutator_row_binds_input_and_output_times() -> None:
    row = _commutator_row(
        case_id="sv_e00_y00",
        case_split="validation",
        protocol="physical",
        kind="teacher_forced_exact_reference_input",
        coarse=(125, 50),
        fine=(250, 100),
        call=5,
        input_frame=8,
        output_frame=10,
        input_physical_time=0.08,
        output_physical_time=0.1,
        reference_gap=0.01,
        metrics={
            "prediction_commutator_scaled_rms": 0.03,
            "prediction_commutator_relative_l2": 0.02,
            "update_commutator_relative_to_fine_update": 0.25,
        },
    )

    assert row["input_call"] == 4
    assert row["input_frame"] == 8
    assert row["output_frame"] == 10
    assert row["input_physical_time"] == pytest.approx(0.08)
    assert row["physical_time"] == pytest.approx(0.1)
    assert row["model_inconsistency_excess_scaled_rms"] == pytest.approx(0.02)


def test_frozen_checkpoint_runs_on_two_node_and_edge_counts(tmp_path: Path) -> None:
    torch.manual_seed(7)
    normalization = Euler2DNormalization(
        state_mean=np.asarray([1.0, 1.0, 0.0, 3.0]),
        state_scale=np.asarray([0.1, 0.1, 0.1, 0.2]),
        residual_scale=np.asarray([1.0e-3] * 4),
        mach_mean=1.1,
        mach_scale=0.1,
        weight_provenance="validated_physical_cell_volume_normalized",
    )
    model = PCNOEuler2DResidual(
        normalization=normalization,
        k_max=1,
        domain_lengths=(2.0, 1.0),
        layers=(8, 8, 8),
        fc_dim=8,
        nmeasures=1,
        zero_initialize=False,
    )
    checkpoint = {
        "checkpoint_schema_version": 4,
        "model_state": model.state_dict(),
        "model_config": model.model_config(),
        "normalization": normalization.to_dict(),
        "normalization_digest": "synthetic-normalization",
        "data_manifest_digest": "synthetic-data",
        "data_contract": {
            "dataset": "synthetic_resolution_fixture",
            "step_stride": 1,
        },
        "config_digest": "synthetic-config",
        "boundary_mode": "model_all_nodes",
        "raw_recurrence": True,
        "inference_interventions": {"boundary_rebinding": False},
        "training_args": {"amp": "none"},
    }
    checkpoint_path = tmp_path / "synthetic.pt"
    torch.save(checkpoint, checkpoint_path)
    output_dir = tmp_path / "resolution"
    manifest = (
        ROOT
        / "artifacts"
        / "time_dependent_no"
        / "shock_vortex_family_frozen_20260720a"
        / "family_manifest.json"
    )

    assert (
        main(
            [
                "--family-manifest",
                str(manifest),
                "--checkpoint",
                str(checkpoint_path),
                "--expected-checkpoint-sha256",
                _sha256(checkpoint_path),
                "--output-dir",
                str(output_dir),
                "--case-ids",
                "sv_e00_y04",
                "--resolutions",
                "6x6",
                "12x12",
                "--training-resolution",
                "6x6",
                "--protocols",
                "physical",
                "all_normal",
                "training_band",
                "--expected-k-max",
                "1",
                "--repeat-forward",
                "2",
                "--device",
                "cpu",
                "--amp",
                "none",
            ]
        )
        == 0
    )
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "complete"
    assert summary["scientific_scope"]["supports_pde_accuracy_claim"] is False
    assert summary["translation_contract"]["resolutions"] == ["6x6", "12x12"]
    assert len(summary["commutators"]) == 3
    assert len(summary["predictions"]) == 6
    assert summary["execution"]["maximum_repeat_abs_difference"] == pytest.approx(0.0)
    assert len(summary["execution"]["gradient_layer_softsign_prefactors"]) == 2
    assert {row["num_nodes"] for row in summary["predictions"]} == {36, 144}
    assert all(row["finite"] for row in summary["predictions"])
