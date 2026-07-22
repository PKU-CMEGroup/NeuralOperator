from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.time_dependent_no.generate_euler2d_shock_vortex_sharpclaw_reference import (
    CFL_DESIRED,
    CFL_MAX,
    CLAWPACK_BUILD,
    CLAWPACK_VERSION,
    GAMMA,
    SCHEMA,
    SHARPCLAW_NUM_GHOST,
    _artifact_arrays,
    _assert_boundary_registration,
    _assert_pinned_clawpack,
    _assert_solver_contract,
    _assign_pyclaw_initial_state,
    _extract_row_major_state,
    _output_times,
    _primitive_from_conservative,
    _primitive_to_conservative,
    _quadrature_certificate,
    _validate_arguments,
    linear_primitive_x_lower_boundary,
    linear_primitive_x_upper_boundary,
    shock_vortex_initial_cell_averages_numpy,
)
from utility.time_dependent_no.shock_vortex_fv import (
    ShockVortexFVConfig,
    shock_vortex_initial_cell_averages,
)


def test_numpy_cell_average_initializer_matches_repaired_primary() -> None:
    config = ShockVortexFVConfig(
        nx=50,
        ny=20,
        coarse_nx=50,
        coarse_ny=20,
        t_final=0.001,
        output_times=(0.0, 0.001),
        initial_quadrature_order=8,
    )
    actual = shock_vortex_initial_cell_averages_numpy(50, 20, quadrature_order=8)
    expected = shock_vortex_initial_cell_averages(
        config,
        device="cpu",
        dtype=torch.float64,
        quadrature_order=8,
    ).numpy()
    np.testing.assert_allclose(actual, expected, rtol=2.0e-14, atol=2.0e-14)

    doubled = shock_vortex_initial_cell_averages_numpy(50, 20, quadrature_order=16)
    certificate = _quadrature_certificate(actual, doubled)
    assert certificate["relative_l2"] <= 1.0e-10
    assert certificate["max_abs"] >= 0.0


def test_pyclaw_layout_round_trip_is_row_major() -> None:
    average = np.arange(3 * 5 * 4, dtype=np.float64).reshape(3, 5, 4)
    state = SimpleNamespace(q=np.empty((4, 5, 3), dtype=np.float64))
    _assign_pyclaw_initial_state(state, average)
    extracted = _extract_row_major_state(state, nx=5, ny=3)
    np.testing.assert_array_equal(extracted, average.reshape(-1, 4))

    invalid = SimpleNamespace(q=np.empty((4, 3, 5), dtype=np.float64))
    with pytest.raises(ValueError, match="expected PyClaw q shape"):
        _assign_pyclaw_initial_state(invalid, average)


def test_custom_x_callbacks_match_primary_linear_primitive_ghosts() -> None:
    nx, ny = 6, 4
    num_ghost = SHARPCLAW_NUM_GHOST
    x_dimension = SimpleNamespace(name="x")
    y_dimension = SimpleNamespace(name="y")
    grid = SimpleNamespace(dimensions=[x_dimension, y_dimension], num_cells=(nx, ny))
    state = SimpleNamespace(
        grid=grid,
        problem_data={"gamma": GAMMA},
        q=np.arange(4 * nx * ny, dtype=np.float64).reshape(4, nx, ny),
    )
    state_before = state.q.copy()
    x = np.arange(nx, dtype=np.float64)[:, None]
    y = np.arange(ny, dtype=np.float64)[None, :]
    primitive = np.stack(
        (
            np.broadcast_to(1.0 + 0.01 * x, (nx, ny)),
            np.broadcast_to(0.7 + 0.02 * x, (nx, ny)),
            np.broadcast_to(-0.03 + 0.01 * y, (nx, ny)),
            np.broadcast_to(1.2 + 0.04 * x + 0.01 * y, (nx, ny)),
        ),
        axis=-1,
    )
    conservative = _primitive_to_conservative(primitive, GAMMA)
    qbc = np.full(
        (4, nx + 2 * num_ghost, ny + 2 * num_ghost),
        np.nan,
        dtype=np.float64,
    )
    qbc[
        :,
        num_ghost : num_ghost + nx,
        num_ghost : num_ghost + ny,
    ] = np.moveaxis(conservative, -1, 0)

    linear_primitive_x_lower_boundary(state, x_dimension, 0.125, qbc, None, num_ghost)
    linear_primitive_x_upper_boundary(state, x_dimension, 0.125, qbc, None, num_ghost)
    np.testing.assert_array_equal(state.q, state_before)
    interior_y = slice(num_ghost, num_ghost + ny)
    for offset in range(1, num_ghost + 1):
        left = _primitive_from_conservative(
            np.moveaxis(qbc[:, num_ghost - offset, interior_y], 0, -1), GAMMA
        )
        right = _primitive_from_conservative(
            np.moveaxis(qbc[:, num_ghost + nx - 1 + offset, interior_y], 0, -1),
            GAMMA,
        )
        np.testing.assert_allclose(
            left, (offset + 1) * primitive[0] - offset * primitive[1]
        )
        np.testing.assert_allclose(
            right, (offset + 1) * primitive[-1] - offset * primitive[-2]
        )

        left_bottom = qbc[:, num_ghost - offset, num_ghost - 1]
        left_inside = qbc[:, num_ghost - offset, num_ghost]
        np.testing.assert_allclose(left_bottom[[0, 1, 3]], left_inside[[0, 1, 3]])
        assert left_bottom[2] == pytest.approx(-left_inside[2])
        right_top = qbc[:, num_ghost + nx - 1 + offset, num_ghost + ny]
        right_inside = qbc[:, num_ghost + nx - 1 + offset, num_ghost + ny - 1]
        np.testing.assert_allclose(right_top[[0, 1, 3]], right_inside[[0, 1, 3]])
        assert right_top[2] == pytest.approx(-right_inside[2])


def test_custom_x_callback_fails_closed_on_pyclaw_contract_drift() -> None:
    x_dimension = SimpleNamespace(name="x")
    y_dimension = SimpleNamespace(name="y")
    state = SimpleNamespace(
        grid=SimpleNamespace(dimensions=[x_dimension, y_dimension], num_cells=(6, 4)),
        problem_data={"gamma": GAMMA},
    )
    qbc = np.ones((4, 12, 10), dtype=np.float64)
    with pytest.raises(RuntimeError, match="valid only for x"):
        linear_primitive_x_lower_boundary(
            state, y_dimension, 0.0, qbc, None, SHARPCLAW_NUM_GHOST
        )
    with pytest.raises(RuntimeError, match="expected 3 SharpClaw ghost cells"):
        linear_primitive_x_lower_boundary(state, x_dimension, 0.0, qbc, None, 2)
    with pytest.raises(RuntimeError, match="qbc component or ghost-axis layout"):
        linear_primitive_x_lower_boundary(
            state,
            x_dimension,
            0.0,
            np.ones((4, 11, 10)),
            None,
            SHARPCLAW_NUM_GHOST,
        )


def test_exact_conda_package_record_is_required(tmp_path: Path) -> None:
    conda_meta = tmp_path / "conda-meta"
    conda_meta.mkdir()
    record = conda_meta / f"clawpack-{CLAWPACK_VERSION}-{CLAWPACK_BUILD}.json"
    record.write_text(
        json.dumps(
            {
                "name": "clawpack",
                "version": CLAWPACK_VERSION,
                "build": CLAWPACK_BUILD,
            }
        ),
        encoding="utf-8",
    )
    package = _assert_pinned_clawpack(tmp_path)
    assert package["version"] == CLAWPACK_VERSION
    assert package["build"] == CLAWPACK_BUILD
    assert len(package["record_sha256"]) == 64

    payload = json.loads(record.read_text(encoding="utf-8"))
    payload["build"] = "wrong_build"
    record.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="expected clawpack"):
        _assert_pinned_clawpack(tmp_path)


def test_solver_and_argument_contract_rejects_drift() -> None:
    solver = SimpleNamespace(
        kernel_language="Fortran",
        lim_type=2,
        weno_order=5,
        char_decomp=0,
        time_integrator="SSP33",
        cfl_desired=CFL_DESIRED,
        cfl_max=CFL_MAX,
        num_ghost=SHARPCLAW_NUM_GHOST,
    )
    _assert_solver_contract(solver)
    solver.char_decomp = 2
    with pytest.raises(RuntimeError, match="char_decomp"):
        _assert_solver_contract(solver)

    custom_value = object()
    wall_value = object()
    boundary_solver = SimpleNamespace(
        bc_lower=[custom_value, wall_value],
        bc_upper=[custom_value, wall_value],
        user_bc_lower=linear_primitive_x_lower_boundary,
        user_bc_upper=linear_primitive_x_upper_boundary,
    )
    _assert_boundary_registration(
        boundary_solver,
        custom_boundary_value=custom_value,
        wall_boundary_value=wall_value,
    )
    boundary_solver.user_bc_upper = linear_primitive_x_lower_boundary
    with pytest.raises(RuntimeError, match="upper custom callback"):
        _assert_boundary_registration(
            boundary_solver,
            custom_boundary_value=custom_value,
            wall_boundary_value=wall_value,
        )

    args = SimpleNamespace(nx=12, ny=6, t_final=0.11, output_dt=0.05)
    assert _validate_arguments(args) == (0.0, 0.05, 0.1, 0.11)
    assert _output_times(0.6, 0.05) == ShockVortexFVConfig().output_times
    args.ny = 5
    with pytest.raises(ValueError, match="at least six"):
        _validate_arguments(args)
    with pytest.raises(ValueError, match="positive"):
        _output_times(0.1, 0.0)


def test_state_only_artifact_schema_excludes_face_impulses() -> None:
    assert SCHEMA == "shock_vortex_sharpclaw_reference_v2"
    states = np.ones((2, 6, 4), dtype=np.float64)
    times = np.asarray([0.0, 0.1])
    centers = np.zeros((6, 2), dtype=np.float64)
    volume = np.full(6, 1.0 / 6.0)
    metadata = {
        "provides_reference_face_impulses": False,
        "state_only_independent_comparison": True,
    }
    arrays = _artifact_arrays(
        states,
        times,
        centers,
        volume,
        {"solver": "pyclaw.SharpClawSolver2D"},
        metadata,
    )
    assert arrays["schema"].item() == SCHEMA
    assert arrays["conservative_states"].shape == (2, 6, 4)
    assert arrays["physical_delta_t"].tolist() == [0.1]
    assert not any("impulse" in name for name in arrays)
    assert json.loads(arrays["metadata_json"].item()) == metadata
    assert "linear primitive-variable extrapolation" in arrays["boundary_mode"].item()


def test_initializer_argument_validation() -> None:
    with pytest.raises(ValueError, match="positive"):
        shock_vortex_initial_cell_averages_numpy(0, 6)
    with pytest.raises(ValueError, match="at least two"):
        shock_vortex_initial_cell_averages_numpy(6, 6, quadrature_order=1)
