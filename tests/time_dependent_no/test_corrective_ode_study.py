from __future__ import annotations

import json
import math
import shutil
import uuid
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from scripts.time_dependent_no.run_corrective_ode_study import (
    LearnedStudyConfig,
    LinearFlow,
    _geometric_sums,
    _test_phases,
    build_training_contract,
    continuous_flow_coefficients,
    estimate_learned_response,
    proxy_geometry_calibration,
    run_exact_study,
    run_study,
    train_matched_models,
    verify_study_packet,
)


@contextmanager
def _runtime_directory(name: str):
    root = Path("artifacts/time_dependent_no").resolve()
    path = root / f"pytest_corrective_ode_{name}_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        resolved = path.resolve()
        if resolved.parent != root:
            raise AssertionError(
                "refusing to clean a test path outside the artifact root"
            )
        shutil.rmtree(resolved)


def test_exact_flow_coefficients_cover_zero_limit_and_semigroup() -> None:
    zero_a, zero_b = continuous_flow_coefficients(
        kappa=0.0, coupling=1.7, step_size=0.2
    )
    near_a, near_b = continuous_flow_coefficients(
        kappa=1.0e-12, coupling=1.7, step_size=0.2
    )
    assert zero_a == 1.0
    assert zero_b == pytest.approx(0.34)
    assert near_a == pytest.approx(zero_a, abs=3.0e-13)
    assert near_b == pytest.approx(zero_b, abs=1.0e-13)

    one_step = LinearFlow.from_continuous(
        omega=0.7, kappa=-0.4, coupling=1.3, step_size=0.15
    )
    two_step = LinearFlow.from_continuous(
        omega=0.7, kappa=-0.4, coupling=1.3, step_size=0.30
    )
    state = np.asarray([[0.2, -0.3], [1.1, 0.4]], dtype=np.float64)
    assert one_step.advance(one_step.advance(state)) == pytest.approx(
        two_step.advance(state), abs=2.0e-15
    )


@pytest.mark.parametrize("normal_gain", (0.0, 0.7, 1.0, 1.0 + 1.0e-9, 1.2))
def test_closed_geometric_sums_match_direct_summation(normal_gain: float) -> None:
    first, second = _geometric_sums(normal_gain, 30)
    direct_first = []
    direct_second = []
    running_first = 0.0
    running_second = 0.0
    power = 1.0
    for index in range(31):
        if index:
            running_first += power
            running_second += direct_first[-1]
            power *= normal_gain
        direct_first.append(running_first)
        direct_second.append(running_second)
    assert first == pytest.approx(direct_first, rel=2.0e-12, abs=2.0e-12)
    assert second == pytest.approx(direct_second, rel=3.0e-11, abs=3.0e-11)


def test_registered_exact_scenarios_close_without_posthoc_changes() -> None:
    summary, scenarios, trajectories, crossover = run_exact_study()
    assert all(summary["checks"].values()), summary["checks"]
    assert summary["status"] == "pass"
    assert all(summary["checks"].values())
    assert len(scenarios) == 4
    assert trajectories
    assert crossover
    assert summary["crossover"]["relabel_win_gain_range"]
    assert summary["crossover"]["recovery_win_gain_range"]
    assert all(row["recurrence_pass"] for row in crossover)
    assert all(row["corrected_recurrence_pass"] for row in summary["composition"])
    false_attractor = next(
        row for row in scenarios if row["scenario"] == "bounded_false_attractor"
    )
    assert false_attractor["normal_fixed_point_type"] == "unique"
    assert false_attractor["normal_fixed_point"] == pytest.approx(0.1)
    assert false_attractor["normal_fixed_point_stable"] is True
    assert false_attractor["final_distance_to_normal_fixed_point"] < 3.0e-4


def test_matched_target_contract_uses_identical_displaced_inputs() -> None:
    config = LearnedStudyConfig(train_phases=16, evaluation_phases=8)
    datasets, metadata = build_training_contract(config)
    clean_inputs, clean_targets = datasets["CLEAN"]
    recovery_inputs, recovery_targets = datasets["RECOVERY"]
    relabel_inputs, relabel_targets = datasets["DYN_RELABEL"]
    count = config.train_phases

    assert clean_inputs.shape == recovery_inputs.shape == relabel_inputs.shape
    assert clean_inputs[count:] == pytest.approx(clean_inputs[:count])
    assert recovery_inputs[count:] == pytest.approx(relabel_inputs[count:])
    assert recovery_targets[count:] == pytest.approx(clean_targets[:count])
    assert relabel_targets[count:] == pytest.approx(
        config.flow.advance(relabel_inputs[count:])
    )
    assert (
        metadata["recovery_displaced_input_digest"]
        == metadata["dyn_relabel_displaced_input_digest"]
    )
    displacements = recovery_inputs[count:, 1]
    assert displacements.min() == pytest.approx(-config.max_displacement)
    assert displacements.max() == pytest.approx(config.max_displacement)
    assert displacements.mean() == pytest.approx(0.0, abs=1.0e-15)
    assert np.min(np.abs(displacements)) <= (
        config.max_displacement * 2.0 / (config.train_phases - 1)
    )
    test_phases = _test_phases(config)
    train_phases = clean_inputs[:count, 0]
    pairwise = np.abs(
        np.angle(np.exp(1j * (test_phases[:, None] - train_phases[None, :])))
    )
    assert np.min(pairwise) > 0.0


class _ExactResidual(nn.Module):
    def __init__(
        self, *, omega_h: float, normal_gain: float, phase_gain: float
    ) -> None:
        super().__init__()
        self.omega_h = omega_h
        self.normal_gain = normal_gain
        self.phase_gain = phase_gain

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        result = state.clone()
        result[:, 0] = state[:, 0] + self.omega_h + self.phase_gain * state[:, 1]
        result[:, 1] = self.normal_gain * state[:, 1]
        return result


def test_operational_projection_zeroes_normal_gain_but_preserves_phase_coupling() -> (
    None
):
    config = LearnedStudyConfig(evaluation_phases=16)
    model = _ExactResidual(
        omega_h=config.omega_h,
        normal_gain=config.trusted_normal_gain,
        phase_gain=config.trusted_normal_to_phase,
    )
    raw = estimate_learned_response(model, config, rho=1.0)
    projected = estimate_learned_response(model, config, rho=0.0)
    assert raw["normal_gain"] == pytest.approx(config.trusted_normal_gain)
    assert raw["normal_to_phase"] == pytest.approx(config.trusted_normal_to_phase)
    assert projected["normal_gain"] == pytest.approx(0.0, abs=1.0e-14)
    assert projected["normal_to_phase"] == pytest.approx(config.trusted_normal_to_phase)


def test_proxy_geometry_exposes_convex_hull_and_sampling_failures() -> None:
    rows = {row["query"]: row for row in proxy_geometry_calibration()}
    assert rows["on_manifold_midpoint"]["exact_embedded_distance"] == 0.0
    assert rows["on_manifold_midpoint"]["knn_distance"] > 0.0
    assert rows["normal_displacement"]["exact_embedded_distance"] == pytest.approx(0.1)
    assert rows["ambient_center"]["sampled_convex_hull_inside"]
    assert rows["ambient_center"]["exact_embedded_distance"] == pytest.approx(1.0)


def test_tiny_training_keeps_paired_initialization_and_checkpoint_roundtrip() -> None:
    config = LearnedStudyConfig(
        train_phases=16,
        hidden_width=8,
        training_steps=3,
        batch_size=8,
        seeds=(5,),
        evaluation_phases=8,
        query_radii=(-0.1, 0.1),
        rollout_steps=3,
        impulse_steps=2,
    )
    with _runtime_directory("tiny_training") as runtime:
        models, rows, metadata = train_matched_models(
            config, checkpoint_dir=runtime / "checkpoints"
        )
        assert set(models[5]) == {"CLEAN", "RECOVERY", "DYN_RELABEL"}
        assert len(rows) == 3
        assert all(math.isfinite(row["final_loss"]) for row in rows)
        assert len({row["initial_parameter_digest"] for row in rows}) == 1
        assert len({row["batch_plan_digest"] for row in rows}) == 1
        assert metadata["seeds"]["5"]["arms"].keys() == models[5].keys()


def test_exact_packet_verification_rejects_result_tampering() -> None:
    with _runtime_directory("packet") as runtime:
        output = runtime / "exact_packet"
        result = run_study(output_dir=output, mode="exact")
        json.dumps(result, allow_nan=False)
        assert result["verification"]["status"] == "verified"
        packet_only = verify_study_packet(output, verify_current_sources=False)
        assert packet_only["status"] == "verified"
        assert packet_only["current_sources_checked"] is False
        summary_path = output / "summary.json"
        summary_path.write_text("{}\n", encoding="utf-8")
        with pytest.raises(ValueError, match="output hash mismatch"):
            verify_study_packet(output)


def test_tiny_all_mode_exercises_learned_evaluation_and_packet() -> None:
    config = LearnedStudyConfig(
        train_phases=16,
        training_steps=2,
        hidden_width=4,
        batch_size=8,
        seeds=(5,),
        evaluation_phases=8,
        query_radii=(-0.1, 0.1),
        rollout_steps=3,
        impulse_steps=2,
    )
    with _runtime_directory("all_mode") as runtime:
        output = runtime / "all_packet"
        result = run_study(output_dir=output, mode="all", config=config)
        learned = result["summary"]["learned"]
        assert learned is not None
        assert "mechanism_realization_pass" in learned["checks"]
        assert (output / "learned_summary.json").is_file()
        assert (output / "figure3_ode_calibration.pdf").is_file()
        assert result["verification"]["status"] == "verified"
