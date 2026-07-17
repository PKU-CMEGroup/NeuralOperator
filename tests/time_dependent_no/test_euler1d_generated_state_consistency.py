from __future__ import annotations

import numpy as np

from scripts.time_dependent_no.diagnose_euler1d_generated_state_consistency import (
    SolverReplayConfig,
    advance_reference_conservative,
    cancellation_metrics,
)
from scripts.time_dependent_no.euler1d_weno_hllc_ader_dataset import (
    CaseConfig,
    integrate_case,
    primitive_to_conservative,
)


def test_reference_advance_preserves_uniform_rest_state() -> None:
    gamma = 1.4
    primitive = np.tile(np.array([1.1, 0.0, 1.3]), (32, 1))
    conservative = primitive_to_conservative(primitive, gamma)
    config = SolverReplayConfig(gamma=gamma, cfl=0.25)

    replay_primitive, replay_conservative, diagnostics = (
        advance_reference_conservative(
            conservative,
            primitive[0],
            dx=1.0 / 32.0,
            interval_dt=0.01,
            config=config,
        )
    )

    np.testing.assert_allclose(replay_primitive, primitive, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(
        replay_conservative,
        conservative,
        rtol=0.0,
        atol=1.0e-12,
    )
    assert diagnostics["substeps"] >= 1
    assert diagnostics["fallback_steps"] == 0


def test_reference_advance_replays_one_serialized_solver_interval() -> None:
    gamma = 1.4
    case = CaseConfig(
        x_left=0.0,
        x_right=1.0,
        x_disc=0.3,
        left_state=np.array([1.0, 0.8, 1.0]),
        right_state=np.array([0.15, 0.0, 0.12]),
        t_final=0.02,
    )
    x, t, data, _ = integrate_case(
        case,
        nx=48,
        n_steps=2,
        gamma=gamma,
        cfl=0.25,
        ng=3,
        rho_floor=1.0e-10,
        p_floor=1.0e-10,
        use_shock_flattening=True,
        use_hlle_on_troubled_faces=True,
        shock_sensor_threshold=0.05,
        shock_flatten_radius=4,
    )
    initial_conservative = primitive_to_conservative(data[0].astype(np.float64), gamma)
    replay_primitive, _, diagnostics = advance_reference_conservative(
        initial_conservative,
        case.left_state,
        dx=float(np.mean(np.diff(x))),
        interval_dt=float(t[1] - t[0]),
        config=SolverReplayConfig(gamma=gamma, cfl=0.25),
    )

    np.testing.assert_allclose(replay_primitive, data[1], rtol=2.0e-5, atol=2.0e-6)
    assert diagnostics["fallback_steps"] == 0


def test_cancellation_metric_identifies_exact_trajectory_correction() -> None:
    gamma = 1.4
    truth = np.array([[1.0, 0.2, 2.6], [0.8, 0.1, 2.0]])
    defect = np.array([[0.1, -0.05, 0.2], [-0.05, 0.02, -0.1]])
    reference = truth + defect

    metrics = cancellation_metrics(truth, reference, truth, gamma)

    assert np.isclose(metrics["correction_toward_truth_cosine"], 1.0)
    assert np.isclose(metrics["correction_norm_over_reference_defect"], 1.0)
