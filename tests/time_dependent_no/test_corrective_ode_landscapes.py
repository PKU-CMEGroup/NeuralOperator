from __future__ import annotations

import json
import shutil
import uuid
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from scripts.time_dependent_no.plot_corrective_ode_landscapes import (
    LandscapeConfig,
    compute_landscape_data,
    generate_landscape_packet,
    periodic_path_for_plot,
    verify_landscape_packet,
)
from scripts.time_dependent_no.run_corrective_ode_study import (
    LearnedStudyConfig,
    run_study,
)


@contextmanager
def _runtime_directory(name: str):
    root = Path("artifacts/time_dependent_no").resolve()
    path = root / f"pytest_ode_landscape_{name}_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        resolved = path.resolve()
        if resolved.parent != root:
            raise AssertionError("refusing to clean outside the artifact root")
        shutil.rmtree(resolved)


class _ExactFlowModel(nn.Module):
    def __init__(self, config: LearnedStudyConfig) -> None:
        super().__init__()
        self.omega_h = config.omega_h
        self.normal_gain = config.trusted_normal_gain
        self.phase_gain = config.trusted_normal_to_phase

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        result = state.clone()
        result[:, 0] = state[:, 0] + self.omega_h + self.phase_gain * state[:, 1]
        result[:, 1] = self.normal_gain * state[:, 1]
        return result


def test_periodic_path_breaks_display_seam_without_changing_values() -> None:
    theta, radius = periodic_path_for_plot(
        np.asarray([3.0, 3.2, 3.4]), np.asarray([0.0, 0.1, 0.2])
    )
    assert np.count_nonzero(np.isnan(theta)) == 1
    assert np.count_nonzero(np.isnan(radius)) == 1
    assert theta[~np.isnan(theta)] == pytest.approx(
        [3.0, 3.2 - 2.0 * np.pi, 3.4 - 2.0 * np.pi]
    )
    assert radius[~np.isnan(radius)] == pytest.approx([0.0, 0.1, 0.2])


def test_common_grid_separates_flow_fidelity_from_operational_projection() -> None:
    config = LearnedStudyConfig(
        train_phases=16,
        evaluation_phases=8,
        seeds=(5,),
        query_radii=(-0.1, 0.1),
        rollout_steps=3,
        impulse_steps=2,
    )
    model = _ExactFlowModel(config)
    models = {5: {"CLEAN": model, "RECOVERY": model, "DYN_RELABEL": model}}
    arrays = compute_landscape_data(
        models,
        config,
        LandscapeConfig(
            theta_points=17,
            radius_points=17,
            clean_rollout_steps=3,
            impulse_rollout_steps=2,
        ),
    )

    assert np.max(arrays["CLEAN_flow_defect_lifted_mean"]) < 3.0e-7
    assert np.max(arrays["DYN_RELABEL_flow_defect_lifted_mean"]) < 3.0e-7
    assert np.array_equal(
        arrays["CLEAN_C0_output_tube_distance_mean"],
        np.zeros_like(arrays["CLEAN_C0_output_tube_distance_mean"]),
    )
    assert np.max(arrays["CLEAN_C0_flow_defect_lifted_mean"]) > 0.05


def test_derived_packet_binds_parent_sources_and_rejects_tampering() -> None:
    config = LearnedStudyConfig(
        train_phases=16,
        hidden_width=4,
        training_steps=2,
        batch_size=8,
        seeds=(5,),
        evaluation_phases=8,
        query_radii=(-0.1, 0.1),
        rollout_steps=3,
        impulse_steps=2,
    )
    with _runtime_directory("packet") as runtime:
        parent = runtime / "parent"
        output = runtime / "derived"
        run_study(output_dir=parent, mode="all", config=config)
        result = generate_landscape_packet(
            packet_dir=parent,
            output_dir=output,
            landscape=LandscapeConfig(
                theta_points=17,
                radius_points=17,
                clean_rollout_steps=3,
                impulse_rollout_steps=2,
            ),
            require_scientific_pass=False,
        )
        assert result["status"] == "verified"
        assert result["parent_manifest_sha256"]
        assert (output / "ode_defect_landscapes.pdf").is_file()
        assert (output / "landscapes.npz").is_file()

        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["parent"]["manifest_sha256"] == result[
            "parent_manifest_sha256"
        ]
        assert len(manifest["parent"]["checkpoint_sha256"]) == 3

        summary_path = output / "summary.json"
        summary_path.write_text("{}\n", encoding="utf-8")
        with pytest.raises(ValueError, match="landscape output hash mismatch"):
            verify_landscape_packet(output, packet_dir=parent)
