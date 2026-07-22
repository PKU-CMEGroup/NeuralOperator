from __future__ import annotations

import copy
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from scripts.time_dependent_no.diagnose_pcno_euler2d_local_basis import (
    PRESSURE_FIELD,
    RESIDUAL_FIELD,
    WEIGHT_NAMES,
    _decoded_residual_state_metrics,
    _pressure_structure_metrics,
    _shock_separated_smooth_mask,
    _validate_residual_pairing,
    evaluate_frozen_gates,
)
from utility.time_dependent_no.euler2d_metrics import shock_front_masks
from utility.time_dependent_no.pcno_ripple_diagnostics import (
    LOCAL_BASIS_CENTER_COUNT,
    deterministic_farthest_point_centers,
    weighted_basis_projection_audit,
    wendland_c2_partition_of_unity_basis,
)


def test_local_basis_entry_point_resolves_repo_imports_outside_worktree(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            str(
                root / "scripts/time_dependent_no/diagnose_pcno_euler2d_local_basis.py"
            ),
            "--help",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "saved full-resolution D013 projections" in result.stdout


def test_fps_wendland_basis_is_deterministic_local_and_partitioned() -> None:
    axis = np.linspace(0.0, 1.0, 19)
    x, y = np.meshgrid(axis, axis, indexing="ij")
    nodes = np.stack((3.0 * x.ravel(), y.ravel()), axis=-1)

    summary, arrays = deterministic_farthest_point_centers(
        nodes,
        center_count=LOCAL_BASIS_CENTER_COUNT,
    )
    repeat_summary, repeat_arrays = deterministic_farthest_point_centers(
        nodes,
        center_count=LOCAL_BASIS_CENTER_COUNT,
    )
    assert summary == repeat_summary
    np.testing.assert_array_equal(
        arrays["center_indices"],
        repeat_arrays["center_indices"],
    )
    first = int(arrays["center_indices"][0])
    np.testing.assert_allclose(arrays["normalized_positions"][first], [0.5, 0.5])
    assert np.unique(arrays["center_indices"]).size == LOCAL_BASIS_CENTER_COUNT

    basis_summary, basis = wendland_c2_partition_of_unity_basis(
        arrays["normalized_positions"],
        arrays["normalized_centers"],
        fill_distance=summary["fill_distance_h"],
    )
    assert basis.shape == (nodes.shape[0], LOCAL_BASIS_CENTER_COUNT)
    assert basis_summary["radius"] == pytest.approx(2.0 * summary["fill_distance_h"])
    assert basis_summary["finite"]
    assert basis_summary["covered_node_count"] == nodes.shape[0]
    assert basis_summary["partition_of_unity_max_row_error"] <= 1e-12
    np.testing.assert_allclose(basis.sum(axis=1), 1.0, atol=1e-12)
    assert np.all(basis >= 0.0)


def test_weighted_basis_projection_uses_weighted_pseudoinverse() -> None:
    x = np.linspace(-1.0, 1.0, 21)
    basis = np.stack((np.ones_like(x), x, x**2), axis=-1)
    coefficients = np.asarray([[0.2, -0.5], [1.3, 0.4], [-0.7, 0.8]])
    field = basis @ coefficients
    weights = np.linspace(0.5, 2.0, x.size)

    summary, arrays = weighted_basis_projection_audit(
        basis,
        weights,
        field,
        regions={"left": x < 0.0, "right": x >= 0.0},
    )
    assert summary["numerical_rank"] == 3
    assert summary["rmse"] < 1e-12
    assert summary["regions"]["left"]["rmse"] < 1e-12
    np.testing.assert_allclose(arrays["reconstruction"], field, atol=1e-12)


def _grid_edges(nx: int, ny: int) -> np.ndarray:
    index = np.arange(nx * ny).reshape(nx, ny)
    horizontal = np.stack((index[:-1].ravel(), index[1:].ravel()), axis=-1)
    vertical = np.stack((index[:, :-1].ravel(), index[:, 1:].ravel()), axis=-1)
    return np.concatenate((horizontal, vertical), axis=0)


def test_pressure_structure_metrics_reproduce_exact_front() -> None:
    nx, ny = 20, 10
    x, y = np.meshgrid(
        np.linspace(0.0, 2.0, nx),
        np.linspace(0.0, 1.0, ny),
        indexing="ij",
    )
    positions = np.stack((x.ravel(), y.ravel()), axis=-1)
    edges = _grid_edges(nx, ny)
    pressure = (1.0 + (positions[:, 0] >= 1.0)).reshape(-1, 1)
    interior = np.ones(pressure.shape[0], dtype=bool)
    primitive = np.zeros((pressure.shape[0], 4))
    primitive[:, 3] = pressure[:, 0]
    fronts = shock_front_masks(
        primitive,
        primitive,
        edges,
        scalar_index=3,
        quantile=0.9,
        node_mask=interior,
    )

    metrics = _pressure_structure_metrics(
        pressure,
        pressure,
        positions=positions,
        edges=edges,
        interior=interior,
        saved_target_front=fronts["target_mask"],
    )
    assert metrics["status"] == "available"
    assert metrics["front_centroid_distance"] == pytest.approx(0.0)
    assert metrics["thickness_ratio"] == pytest.approx(1.0)
    assert metrics["strength_ratio"] == pytest.approx(1.0)


def test_decoded_residual_metrics_use_identity_bypass_and_raw_admissibility() -> None:
    nx, ny = 20, 10
    x, y = np.meshgrid(
        np.linspace(0.0, 2.0, nx),
        np.linspace(0.0, 1.0, ny),
        indexing="ij",
    )
    positions = np.stack((x.ravel(), y.ravel()), axis=-1)
    edges = _grid_edges(nx, ny)
    interior = np.ones(positions.shape[0], dtype=bool)
    rho = np.ones(positions.shape[0])
    velocity = np.full(positions.shape[0], 0.2)
    pressure = 1.0 + (positions[:, 0] >= 1.0)
    energy = pressure / 0.4 + 0.5 * rho * velocity**2
    target = np.stack((rho, rho * velocity, np.zeros_like(rho), energy), axis=-1)
    current = target.copy()
    normalized_residual = np.zeros_like(target)
    primitive = np.zeros((positions.shape[0], 4))
    primitive[:, 3] = pressure
    fronts = shock_front_masks(
        primitive,
        primitive,
        edges,
        scalar_index=3,
        quantile=0.9,
        node_mask=interior,
    )

    metrics, decoded = _decoded_residual_state_metrics(
        normalized_residual,
        current,
        target,
        np.ones(4),
        positions=positions,
        edges=edges,
        interior=interior,
        saved_target_front=fronts["target_mask"],
    )
    np.testing.assert_allclose(decoded, target)
    assert metrics["raw_admissibility"]["all_admissible"]
    assert metrics["pressure_structure"]["thickness_ratio"] == pytest.approx(1.0)

    normalized_residual[0, 0] = -2.0
    broken, _ = _decoded_residual_state_metrics(
        normalized_residual,
        current,
        target,
        np.ones(4),
        positions=positions,
        edges=edges,
        interior=interior,
        saved_target_front=fronts["target_mask"],
    )
    assert not broken["raw_admissibility"]["all_admissible"]


def test_shock_separated_smooth_mask_dilates_two_graph_hops() -> None:
    edges = _grid_edges(9, 1)
    front = np.zeros(9, dtype=bool)
    front[4] = True
    smooth = _shock_separated_smooth_mask(
        front,
        np.ones(9, dtype=bool),
        edges,
        dilation_hops=2,
    )
    np.testing.assert_array_equal(
        smooth,
        np.asarray([True, True, False, False, False, False, False, True, True]),
    )


def _gate_row() -> dict:
    pressure = {
        "rmse": 0.5,
        "smooth_error_highpass_rms": 0.5,
        "overshoot_fraction_by_component": [0.01],
        "undershoot_fraction_by_component": [0.01],
        "pressure_structure": {
            "status": "available",
            "front_centroid_distance_in_median_edges": 0.5,
            "thickness_ratio": 1.0,
            "strength_ratio": 1.0,
        },
    }
    fields = {
        RESIDUAL_FIELD: {
            "local_wendland_pou_mass_projection": {
                "rmse": 0.5,
                "smooth_error_highpass_rms": 0.5,
                "decoded_next_state": {
                    "raw_admissibility": {
                        "all_finite": True,
                        "all_admissible": True,
                    },
                    "pressure_overshoot_fraction": 0.01,
                    "pressure_undershoot_fraction": 0.01,
                    "pressure_structure": {
                        "status": "available",
                        "front_centroid_distance_in_median_edges": 0.5,
                        "thickness_ratio": 1.0,
                        "strength_ratio": 1.0,
                    },
                },
            },
            "current_fourier_mass_orthogonal": {
                "rmse": 1.0,
                "smooth_error_highpass_rms": 1.0,
            },
            "coordinate_span_fourier_mass_orthogonal": {
                "rmse": 0.7,
                "smooth_error_highpass_rms": 0.8,
            },
        },
        PRESSURE_FIELD: {
            "local_wendland_pou_mass_projection": pressure,
            "current_fourier_mass_orthogonal": pressure,
            "coordinate_span_fourier_mass_orthogonal": pressure,
        },
    }
    return {
        "local_basis": {
            "finite": True,
            "covered_node_count": 20_000,
            "node_count": 20_000,
            "partition_of_unity_max_row_error": 1e-15,
            "column_support_fraction_median": 0.05,
            "column_support_fraction_p90": 0.1,
        },
        "weights": {
            name: {
                "numerical_rank": LOCAL_BASIS_CENTER_COUNT,
                "fields": copy.deepcopy(fields),
            }
            for name in WEIGHT_NAMES
        },
    }


def test_frozen_local_basis_gates_use_decoded_residual_and_fail_closed() -> None:
    rows = [_gate_row() for _ in range(5)]
    result = evaluate_frozen_gates(rows)
    assert result["passed"]
    assert all(result["gates"].values())

    rows[0]["weights"]["equal_node_proxy"]["fields"][PRESSURE_FIELD][
        "local_wendland_pou_mass_projection"
    ]["pressure_structure"] = {"status": "unavailable"}
    assert evaluate_frozen_gates(rows)["passed"]

    rows[0]["weights"]["equal_node_proxy"]["fields"][RESIDUAL_FIELD][
        "local_wendland_pou_mass_projection"
    ]["decoded_next_state"]["raw_admissibility"]["all_admissible"] = False
    failed = evaluate_frozen_gates(rows)
    assert not failed["passed"]
    assert not failed["gates"]["decoded_residual_admissibility_and_anti_smearing"]
    assert failed["evidence"]["decoded_residual_case_pass"][0] is False


def test_residual_pairing_recovers_scale_and_rejects_mismatch() -> None:
    generator = np.random.default_rng(7)
    current = generator.normal(size=(32, 4))
    normalized = generator.normal(size=(32, 4))
    scale = np.asarray([0.2, 0.3, 0.4, 0.5])
    target = current + normalized * scale
    recovered = _validate_residual_pairing(normalized, current, target)
    np.testing.assert_allclose(recovered, scale)

    target[0, 0] += 0.1
    with pytest.raises(ValueError, match="does not match trajectory"):
        _validate_residual_pairing(normalized, current, target)
