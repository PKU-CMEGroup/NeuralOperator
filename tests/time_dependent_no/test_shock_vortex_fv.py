from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.time_dependent_no.audit_euler2d_shock_vortex_convergence import (
    AUDIT_SCHEMA,
    BOUNDARY_EXCHANGE_ACTIVE_CONTRACTION_MASK,
    IMPULSE_DECOMPOSITION_TOLERANCE,
    WALL_ZERO_FLUX_ABSOLUTE_TOLERANCE,
    _boundary_exchange_gate,
    _canonical_primary_contract_checks,
    independent_agreement_metrics,
    load_reference_run,
    main as audit_convergence_main,
    pair_metrics,
)
from scripts.time_dependent_no.generate_euler2d_shock_vortex_pyro_reference import (
    _output_times as pyro_output_times,
    shock_vortex_primitive_numpy,
)
from scripts.time_dependent_no.generate_euler2d_shock_vortex_reference import (
    _output_times as primary_output_times,
    main as generate_reference_main,
)
from utility.time_dependent_no.shock_vortex_fv import (
    REFERENCE_CONTRACT_CHECK_KEYS,
    ShockVortexFVConfig,
    conservative_to_primitive,
    decode_owner_oriented_face_impulse,
    fine_cell_centers,
    make_structured_fv_geometry,
    primitive_to_conservative,
    reference_contract_checks,
    reference_rhs_and_fluxes,
    restrict_cell_averages,
    restrict_owner_oriented_face_impulse,
    run_shock_vortex_reference,
    shock_vortex_initial_cell_averages,
    shock_vortex_initial_primitive,
    ssprk3_step,
    stable_time_step,
)


def _tiny_config(*, t_final: float = 0.001) -> ShockVortexFVConfig:
    return ShockVortexFVConfig(
        nx=12,
        ny=6,
        coarse_nx=6,
        coarse_ny=3,
        t_final=t_final,
        output_times=(0.0, t_final),
        cfl=0.2,
    )


def test_reference_generators_emit_exact_decimal_time_ticks() -> None:
    expected = ShockVortexFVConfig().output_times
    assert primary_output_times(0.6, 0.05) == expected
    assert pyro_output_times(0.6, 0.05) == expected
    assert primary_output_times(0.11, 0.05) == (0.0, 0.05, 0.1, 0.11)
    assert pyro_output_times(0.11, 0.05) == (0.0, 0.05, 0.1, 0.11)


def test_canonical_initial_condition_matches_published_regions() -> None:
    config = _tiny_config()
    points = torch.tensor([[1.5, 0.5], [0.25, 0.5], [0.25, 0.55]], dtype=torch.float64)
    primitive = shock_vortex_initial_primitive(points, config)

    torch.testing.assert_close(
        primitive[0],
        torch.tensor(
            [config.right_rho, config.right_u, 0.0, config.right_pressure],
            dtype=torch.float64,
        ),
    )
    upstream_u = config.shock_mach * np.sqrt(config.gamma)
    assert primitive[1, 0] < 1.0
    assert primitive[1, 3] < 1.0
    assert float(primitive[1, 1]) == pytest.approx(upstream_u)
    assert float(primitive[1, 2]) == pytest.approx(0.0)
    assert float(primitive[2, 1]) > upstream_u

    numpy_primitive = shock_vortex_primitive_numpy(
        points[:, 0].numpy(), points[:, 1].numpy(), gamma=config.gamma
    )
    np.testing.assert_allclose(numpy_primitive, primitive.numpy(), rtol=1.0e-14)


def test_conservative_initial_cell_averages_commute_with_shock_restriction() -> None:
    coarse = ShockVortexFVConfig(
        nx=10,
        ny=6,
        coarse_nx=10,
        coarse_ny=6,
        vortex_epsilon=0.0,
        t_final=0.001,
        output_times=(0.0, 0.001),
    )
    fine = ShockVortexFVConfig(
        nx=20,
        ny=12,
        coarse_nx=10,
        coarse_ny=6,
        vortex_epsilon=0.0,
        t_final=0.001,
        output_times=(0.0, 0.001),
    )
    coarse_average = shock_vortex_initial_cell_averages(
        coarse, device="cpu", dtype=torch.float64, quadrature_order=4
    )
    fine_average = shock_vortex_initial_cell_averages(
        fine, device="cpu", dtype=torch.float64, quadrature_order=4
    )
    restricted = restrict_cell_averages(fine_average, fine)

    torch.testing.assert_close(coarse_average, restricted, rtol=0.0, atol=2.0e-14)
    crossing_index = 2
    left = primitive_to_conservative(
        shock_vortex_initial_primitive(
            torch.tensor([[0.49, 0.5]], dtype=torch.float64), coarse
        ),
        coarse.gamma,
    )[0]
    right = primitive_to_conservative(
        shock_vortex_initial_primitive(
            torch.tensor([[0.51, 0.5]], dtype=torch.float64), coarse
        ),
        coarse.gamma,
    )[0]
    torch.testing.assert_close(
        coarse_average[:, crossing_index],
        0.5 * (left + right).expand(coarse.ny, -1),
        rtol=0.0,
        atol=2.0e-14,
    )


def test_initial_cell_average_quadrature_is_doubled_order_certified() -> None:
    config = _tiny_config()
    order_eight = shock_vortex_initial_cell_averages(
        config, device="cpu", dtype=torch.float64, quadrature_order=8
    )
    order_sixteen = shock_vortex_initial_cell_averages(
        config, device="cpu", dtype=torch.float64, quadrature_order=16
    )
    relative = torch.linalg.vector_norm(
        order_eight - order_sixteen
    ) / torch.linalg.vector_norm(order_sixteen)
    assert float(relative) <= 1.0e-10

    result = run_shock_vortex_reference(config, device="cpu", dtype=torch.float64)
    full_delta = order_eight - order_sixteen
    expected_full_relative = torch.linalg.vector_norm(
        full_delta
    ) / torch.linalg.vector_norm(order_sixteen)
    restricted_delta = restrict_cell_averages(
        order_eight, config
    ) - restrict_cell_averages(order_sixteen, config)
    expected_restricted_relative = torch.linalg.vector_norm(
        restricted_delta
    ) / torch.linalg.vector_norm(restrict_cell_averages(order_sixteen, config))
    assert result.initial_quadrature_relative_l2 == pytest.approx(
        float(expected_full_relative), rel=1.0e-14, abs=1.0e-30
    )
    assert result.restricted_initial_quadrature_relative_l2 == pytest.approx(
        float(expected_restricted_relative), rel=1.0e-14, abs=1.0e-30
    )


def test_structured_geometry_is_complete_and_owner_oriented() -> None:
    config = _tiny_config()
    geometry = make_structured_fv_geometry(config)

    assert geometry.cell_centers.shape == (18, 2)
    assert geometry.face_centers.shape == (45, 2)
    assert int(np.count_nonzero(geometry.interior_face_mask)) == 27
    assert int(np.count_nonzero(~geometry.interior_face_mask)) == 18
    assert float(np.sum(geometry.cell_volume)) == pytest.approx(2.0)
    np.testing.assert_allclose(np.linalg.norm(geometry.face_normal, axis=1), 1.0)
    assert np.all(geometry.face_owner >= 0)
    assert np.all(geometry.face_owner < geometry.cell_volume.size)
    assert np.all(geometry.face_neighbor >= -1)
    assert np.all(geometry.face_neighbor < geometry.cell_volume.size)
    assert np.all(geometry.face_owner != geometry.face_neighbor)


def test_reference_contract_rejects_normal_and_boundary_tag_drift() -> None:
    result = run_shock_vortex_reference(
        _tiny_config(t_final=0.0001),
        device="cpu",
        dtype=torch.float64,
    )
    normal = result.geometry.face_normal.copy()
    normal[1] *= -1.0
    normal_drift = replace(
        result,
        geometry=replace(result.geometry, face_normal=normal),
    )
    normal_checks = reference_contract_checks(
        normal_drift, closure_relative_tolerance=1.0e-10
    )
    assert not normal_checks["face_normal_orientation_consistent"]

    tags = result.geometry.face_boundary_tag.copy()
    tags[0] = 2
    tag_drift = replace(
        result,
        geometry=replace(result.geometry, face_boundary_tag=tags),
    )
    tag_checks = reference_contract_checks(
        tag_drift, closure_relative_tolerance=1.0e-10
    )
    assert not tag_checks["boundary_tags_and_locations_consistent"]


def test_constant_state_rhs_and_symmetry_boundary_flux_are_exact() -> None:
    config = _tiny_config()
    primitive = torch.empty((config.ny, config.nx, 4), dtype=torch.float64)
    primitive[..., 0] = 1.0
    primitive[..., 1] = 0.8
    primitive[..., 2] = 0.0
    primitive[..., 3] = 1.0
    conservative = primitive_to_conservative(primitive, config.gamma)

    rhs, flux_x, flux_y, fallback_count = reference_rhs_and_fluxes(conservative, config)

    torch.testing.assert_close(rhs, torch.zeros_like(rhs), atol=2.0e-13, rtol=0.0)
    assert fallback_count == 0
    torch.testing.assert_close(
        flux_y[[0, -1], :, 0],
        torch.zeros_like(flux_y[[0, -1], :, 0]),
        atol=2.0e-13,
        rtol=0.0,
    )
    torch.testing.assert_close(
        flux_y[[0, -1], :, 3],
        torch.zeros_like(flux_y[[0, -1], :, 3]),
        atol=2.0e-13,
        rtol=0.0,
    )
    assert flux_x.shape == (config.ny, config.nx + 1, 4)


def test_one_ssprk_step_closes_against_restricted_face_impulses() -> None:
    config = _tiny_config()
    centers = fine_cell_centers(config, device="cpu", dtype=torch.float64)
    primitive = shock_vortex_initial_primitive(centers, config)
    conservative = primitive_to_conservative(primitive, config.gamma)
    dt = 0.25 * stable_time_step(conservative, config)

    updated, flux_x, flux_y, _, accepted = ssprk3_step(conservative, dt, config)

    assert accepted
    assert flux_x is not None and flux_y is not None
    impulse = restrict_owner_oriented_face_impulse(flux_x, flux_y, dt, config)
    before = restrict_cell_averages(conservative, config).reshape(-1, 4).numpy()
    after = restrict_cell_averages(updated, config).reshape(-1, 4).numpy()
    decoded = decode_owner_oriented_face_impulse(
        impulse.numpy(), make_structured_fv_geometry(config)
    )
    np.testing.assert_allclose(after - before, decoded, rtol=2.0e-11, atol=2.0e-13)
    assert torch.all(conservative_to_primitive(updated, config.gamma)[..., 3] > 0.0)


def test_tiny_reference_and_entrypoint_emit_closed_artifact(tmp_path: Path) -> None:
    config = _tiny_config(t_final=0.002)
    result = run_shock_vortex_reference(config, device="cpu", dtype=torch.float64)
    checks = reference_contract_checks(result, closure_relative_tolerance=1.0e-10)

    assert all(checks.values())
    assert set(checks) == REFERENCE_CONTRACT_CHECK_KEYS
    assert result.states.shape == (2, 18, 4)
    assert result.face_impulses.shape == (1, 45, 4)
    assert result.accepted_step_times[-1] == pytest.approx(0.002)
    assert result.minimum_density > 0.0
    assert result.minimum_pressure > 0.0

    output_dir = tmp_path / "reference"
    generate_reference_main(
        [
            "--output-dir",
            str(output_dir),
            "--nx",
            "12",
            "--ny",
            "6",
            "--coarse-nx",
            "6",
            "--coarse-ny",
            "3",
            "--t-final",
            "0.002",
            "--output-dt",
            "0.001",
            "--cfl",
            "0.2",
            "--device",
            "cpu",
            "--dtype",
            "float64",
        ]
    )
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "passed"
    assert all(summary["contract_checks"].values())
    assert summary["canonical_case"] is False
    assert summary["canonical_reference_resolution"] is False
    assert summary["claim_boundary"]["still_required"].startswith("grid convergence")
    assert summary["initial_quadrature_scope"].startswith("full fine-grid")
    assert summary["restricted_initial_quadrature_relative_l2"] >= 0.0

    with np.load(output_dir / "reference.npz", allow_pickle=False) as artifact:
        assert artifact["schema"].item() == "shock_vortex_fv_reference_v2"
        assert artifact["conservative_states"].shape == (3, 18, 4)
        assert artifact["cumulative_accepted_substep_face_impulses"].shape == (2, 45, 4)
        assert artifact["face_normal"].shape == (45, 2)
        assert artifact["face_owner"].shape == artifact["face_neighbor"].shape
        metadata = json.loads(artifact["metadata_json"].item())
        assert metadata["future_reference_boundary_values"] is False
        assert metadata["clipping_or_accepted_state_floors"] is False
        assert metadata["initial_quadrature_scope"].startswith("full fine-grid")

    loaded = load_reference_run(output_dir)
    assert loaded.boundary_tag_names == (
        "interior",
        "x_min",
        "x_max",
        "y_min",
        "y_max",
    )
    summary_path = output_dir / "summary.json"
    missing_check = dict(summary)
    missing_check["contract_checks"] = dict(summary["contract_checks"])
    missing_check["contract_checks"].pop("face_normal_orientation_consistent")
    summary_path.write_text(json.dumps(missing_check), encoding="utf-8")
    with pytest.raises(ValueError, match="contract-check key set mismatch"):
        load_reference_run(output_dir)
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    mismatched_config = dict(summary)
    mismatched_config["config"] = dict(summary["config"])
    mismatched_config["config"]["t_final"] = 0.123
    summary_path.write_text(json.dumps(mismatched_config), encoding="utf-8")
    with pytest.raises(ValueError, match="summary/artifact config mismatch"):
        load_reference_run(output_dir)
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        generate_reference_main(["--output-dir", str(output_dir)])


def test_convergence_metrics_and_noncanonical_smoke_audit(tmp_path: Path) -> None:
    geometry = make_structured_fv_geometry(_tiny_config())
    reference = np.ones((2, 18, 4), dtype=np.float64)
    reference[..., 1] = 0.8
    reference[..., 2] = 0.1
    reference[..., 3] = 3.0
    identical = pair_metrics(
        reference,
        reference,
        physical_times=np.asarray([0.0, 0.1]),
        cell_volume=geometry.cell_volume,
        cell_centers=geometry.cell_centers,
        coarse_nx=6,
        coarse_ny=3,
        gamma=1.4,
    )
    assert all(row["normalized_state_relative_l1"] == 0.0 for row in identical)
    perturbed = reference.copy()
    perturbed[-1, 0, 0] += 0.1
    changed = pair_metrics(
        perturbed,
        reference,
        physical_times=np.asarray([0.0, 0.1]),
        cell_volume=geometry.cell_volume,
        cell_centers=geometry.cell_centers,
        coarse_nx=6,
        coarse_ny=3,
        gamma=1.4,
    )
    assert changed[-1]["normalized_state_relative_l1"] > 0.0

    run_dirs: list[Path] = []
    for nx, ny in ((12, 6), (24, 12), (48, 24)):
        run_dir = tmp_path / f"run_{nx}x{ny}"
        run_dirs.append(run_dir)
        generate_reference_main(
            [
                "--output-dir",
                str(run_dir),
                "--nx",
                str(nx),
                "--ny",
                str(ny),
                "--coarse-nx",
                "6",
                "--coarse-ny",
                "3",
                "--t-final",
                "0.0002",
                "--output-dt",
                "0.0002",
                "--cfl",
                "0.2",
                "--device",
                "cpu",
                "--dtype",
                "float64",
            ]
        )
    audit_dir = tmp_path / "audit"
    arguments = ["--output-dir", str(audit_dir), "--allow-noncanonical-resolutions"]
    for run_dir in run_dirs:
        arguments.extend(("--run-dir", str(run_dir)))
    audit_convergence_main(arguments)

    audit = json.loads((audit_dir / "summary.json").read_text(encoding="utf-8"))
    assert audit["schema"] == AUDIT_SCHEMA
    assert audit["status"] == "smoke_only"
    assert audit["resolutions"] == [[12, 6], [24, 12], [48, 24]]
    assert audit["benchmark_contract_closed"] is False
    assert audit["checks"]["independent_public_solver_agreement"] is False
    assert not audit["checks"]["canonical_comparison_mesh"]
    assert "boundary_exchange_error_contracts_by_tag_and_component" in audit["checks"]
    assert (
        "wall_zero_flux_primary_exchanges_below_absolute_tolerance" in audit["checks"]
    )
    assert audit["impulse_decomposition_lsmr_tolerance"] == (
        IMPULSE_DECOMPOSITION_TOLERANCE
    )
    assert audit["wall_zero_flux_absolute_tolerance"] == (
        WALL_ZERO_FLUX_ABSOLUTE_TOLERANCE
    )
    assert audit["boundary_exchange_active_contraction_mask"] == [
        list(row) for row in BOUNDARY_EXCHANGE_ACTIVE_CONTRACTION_MASK
    ]
    assert np.asarray(
        audit["metrics"]["primary_boundary_exchange_l2_by_run_tag_and_component"]
    ).shape == (3, 4, 4)
    assert (
        len(
            audit["metrics"]["impulse_component_contraction"][
                "divergence_active_winv_error_ratio"
            ]
        )
        == 4
    )
    assert (audit_dir / "pair_metrics.csv").is_file()


@pytest.mark.parametrize("excess_matrix", ("coarse", "fine"))
def test_boundary_exchange_gate_uses_absolute_wall_zero_flux_limit(
    excess_matrix: str,
) -> None:
    active_mask = np.asarray(
        BOUNDARY_EXCHANGE_ACTIVE_CONTRACTION_MASK,
        dtype=bool,
    )
    np.testing.assert_array_equal(
        active_mask,
        np.asarray(
            (
                (True, True, True, True),
                (True, True, True, True),
                (False, False, True, False),
                (False, False, True, False),
            )
        ),
    )
    wall_zero_mask = ~active_mask
    coarse = np.ones((4, 4), dtype=np.float64)
    fine = np.full((4, 4), 0.5, dtype=np.float64)
    primary = np.ones((3, 4, 4), dtype=np.float64)
    coarse[wall_zero_mask] = 0.5 * WALL_ZERO_FLUX_ABSOLUTE_TOLERANCE
    fine[wall_zero_mask] = 0.75 * WALL_ZERO_FLUX_ABSOLUTE_TOLERANCE
    primary[:, wall_zero_mask] = 0.25 * WALL_ZERO_FLUX_ABSOLUTE_TOLERANCE

    passing = _boundary_exchange_gate(
        coarse,
        fine,
        primary,
        contraction_limit=0.9,
    )
    assert passing["active_passed"]
    assert passing["wall_zero_flux_error_passed"]
    assert passing["wall_zero_flux_primary_exchange_passed"]
    assert passing["wall_zero_flux_passed"]
    assert passing["combined_passed"]

    selected = coarse if excess_matrix == "coarse" else fine
    selected[2, 3] = np.nextafter(
        WALL_ZERO_FLUX_ABSOLUTE_TOLERANCE,
        np.inf,
    )
    failing = _boundary_exchange_gate(
        coarse,
        fine,
        primary,
        contraction_limit=0.9,
    )
    assert failing["active_passed"]
    assert not failing["wall_zero_flux_error_passed"]
    assert failing["wall_zero_flux_primary_exchange_passed"]
    assert not failing["wall_zero_flux_passed"]
    assert not failing["combined_passed"]


@pytest.mark.parametrize("active_index", ((0, 0), (2, 2)))
def test_boundary_exchange_gate_retains_active_component_contraction(
    active_index: tuple[int, int],
) -> None:
    active_mask = np.asarray(
        BOUNDARY_EXCHANGE_ACTIVE_CONTRACTION_MASK,
        dtype=bool,
    )
    coarse = np.ones((4, 4), dtype=np.float64)
    fine = np.full((4, 4), 0.5, dtype=np.float64)
    primary = np.ones((3, 4, 4), dtype=np.float64)
    coarse[~active_mask] = 0.0
    fine[~active_mask] = 0.0
    primary[:, ~active_mask] = 0.0
    fine[active_index] = 0.91

    result = _boundary_exchange_gate(
        coarse,
        fine,
        primary,
        contraction_limit=0.9,
    )
    assert not result["active_passed"]
    assert result["wall_zero_flux_passed"]
    assert not result["combined_passed"]


def test_boundary_exchange_gate_rejects_shared_primary_wall_leakage() -> None:
    active_mask = np.asarray(
        BOUNDARY_EXCHANGE_ACTIVE_CONTRACTION_MASK,
        dtype=bool,
    )
    coarse = np.ones((4, 4), dtype=np.float64)
    fine = np.full((4, 4), 0.5, dtype=np.float64)
    primary = np.ones((3, 4, 4), dtype=np.float64)
    coarse[~active_mask] = 0.0
    fine[~active_mask] = 0.0
    primary[:, ~active_mask] = 0.0
    primary[1, 3, 1] = np.nextafter(
        WALL_ZERO_FLUX_ABSOLUTE_TOLERANCE,
        np.inf,
    )

    result = _boundary_exchange_gate(
        coarse,
        fine,
        primary,
        contraction_limit=0.9,
    )
    assert result["active_passed"]
    assert result["wall_zero_flux_error_passed"]
    assert not result["wall_zero_flux_primary_exchange_passed"]
    assert not result["wall_zero_flux_passed"]
    assert not result["combined_passed"]


def test_canonical_primary_contract_freezes_more_than_fine_resolution() -> None:
    runs = []
    for nx, ny in ((250, 100), (500, 200), (1000, 400)):
        runs.append(
            SimpleNamespace(
                resolution=(nx, ny),
                config=ShockVortexFVConfig(nx=nx, ny=ny).to_dict(),
                artifact_schema="shock_vortex_fv_reference_v2",
                summary={
                    "dtype": "float64",
                    "entrypoint_sha256": "a" * 64,
                    "solver_utility_sha256": "b" * 64,
                },
                states=np.empty(0, dtype=np.float64),
                face_impulses=np.empty(0, dtype=np.float64),
                cell_centers=np.empty(0, dtype=np.float64),
                cell_volume=np.empty(0, dtype=np.float64),
                face_centers=np.empty(0, dtype=np.float64),
                face_measure=np.empty(0, dtype=np.float64),
                face_normal=np.empty(0, dtype=np.float64),
                coordinate_convention=(
                    "row-major cell averages; x increases right, y increases up"
                ),
                face_orientation_convention=(
                    "owner outward on boundary; owner-to-neighbor on interior faces"
                ),
                state_convention="[rho,rho*u,rho*v,total_energy]",
                boundary_mode="linear x extrapolation; y symmetry",
                solver_method=(
                    "dimension-by-dimension primitive WENO5-JS + HLLC + SSPRK3"
                ),
                metadata={
                    "future_reference_boundary_values": False,
                    "clipping_or_accepted_state_floors": False,
                    "canonical_case": True,
                    "initial_quadrature_scope": (
                        "full fine-grid conservative cell averages compared "
                        "before restriction"
                    ),
                    "initial_state_contract": (
                        "conservative cell averages; shock-crossing cells split "
                        "exactly; full fine-grid tensor Gauss-Legendre quadrature "
                        "certified at doubled order"
                    ),
                    "reference_truth_source": (
                        "numerical finite-volume evolution from the published "
                        "initial condition"
                    ),
                },
                times=np.asarray(ShockVortexFVConfig().output_times),
            )
        )
    assert all(_canonical_primary_contract_checks(runs).values())

    runs[0].states = np.empty(0, dtype=np.float32)
    checks = _canonical_primary_contract_checks(runs)
    assert not checks["canonical_primary_solver_dtype_and_metadata"]
    runs[0].states = np.empty(0, dtype=np.float64)

    runs[0].config = dict(runs[0].config)
    runs[0].config["coarse_nx"] = 125
    checks = _canonical_primary_contract_checks(runs)
    assert not checks["canonical_comparison_mesh"]
    assert not checks["canonical_physical_configuration_and_save_times"]


def test_independent_agreement_uses_frozen_discretization_envelope() -> None:
    def row(
        state_error: float, shock_error: float, vortex_error: float
    ) -> dict[str, float]:
        return {
            "normalized_state_relative_l1": state_error,
            "shock_position_difference": shock_error,
            "vortex_core_density_relative_difference": vortex_error,
        }

    coarse = [row(0.0, 0.0, 0.0), row(0.02, 0.0, 0.0)]
    fine = [row(0.0, 0.0, 0.0), row(0.01, 0.0, 0.0)]
    independent = [row(0.0, 0.0, 0.0), row(0.04, 0.008, 0.04)]
    checks, metrics = independent_agreement_metrics(
        independent,
        coarse,
        fine,
        coarse_dx=0.008,
        state_envelope_factor=1.5,
        vortex_core_relative_tolerance=0.05,
    )

    assert all(checks.values())
    assert metrics["independent_final_state_envelope"] == pytest.approx(0.045)
    independent[-1] = row(0.046, 0.008, 0.04)
    failed_checks, _ = independent_agreement_metrics(
        independent,
        coarse,
        fine,
        coarse_dx=0.008,
        state_envelope_factor=1.5,
        vortex_core_relative_tolerance=0.05,
    )
    assert not failed_checks["independent_final_state_within_discretization_envelope"]
    independent[0] = row(1.0e-8, 0.0, 0.0)
    initial_checks, _ = independent_agreement_metrics(
        independent,
        coarse,
        fine,
        coarse_dx=0.008,
        state_envelope_factor=1.5,
        vortex_core_relative_tolerance=0.05,
        initial_state_relative_tolerance=1.0e-10,
    )
    assert not initial_checks["independent_initial_state_matches_primary"]
