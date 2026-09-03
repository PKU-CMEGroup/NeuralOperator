from __future__ import annotations

import copy
from itertools import pairwise

import numpy as np
import pytest
import torch
from torch import nn

from scripts.time_dependent_no.plot_corrective_ode_nonlinear_diagnostics import (
    ARM_KEYS,
    CHECKPOINT_ARMS,
    PAPER_ARM_COLORS,
    PAPER_HEATMAP_CMAP,
    PAPER_REFERENCE_COLOR,
    PAPER_TRUSTED_COLOR,
    DiagnosticConfig,
    compute_landscape_arrays,
    compute_trajectory_rows,
    periodic_path_for_plot,
    select_fixed_forcing,
    validate_parent_manifest_binding,
)
from scripts.time_dependent_no.run_corrective_ode_nonlinear_stress import (
    NonlinearStressConfig,
    flows,
    forcing_bank,
    lifted_coordinate_error,
)


class _IdentityModel(nn.Module):
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return state


def test_palette_matches_the_active_paper_conventions() -> None:
    assert PAPER_HEATMAP_CMAP == "magma"
    assert PAPER_REFERENCE_COLOR == "#00E5FF"
    assert PAPER_TRUSTED_COLOR == "#009E73"
    assert PAPER_ARM_COLORS == {
        "CLEAN": "#4C78A8",
        "RECOVERY": "#F58518",
        "DYN": "#E45756",
    }


def _tiny_config(**overrides: object) -> NonlinearStressConfig:
    values: dict[str, object] = {
        "train_phases": 16,
        "hidden_width": 8,
        "hidden_layers": 1,
        "training_steps": 1,
        "batch_size": 8,
        "seeds": (5,),
        "noise_steps": 3,
        "noise_phases": 8,
        "noise_sequences": 2,
        "min_common_prefix_steps": 1,
    }
    values.update(overrides)
    return NonlinearStressConfig(**values).validated()


def _identity_models(config: NonlinearStressConfig) -> dict[int, dict[str, nn.Module]]:
    return {
        seed: {arm: _IdentityModel() for arm in CHECKPOINT_ARMS}
        for seed in config.seeds
    }


def test_grid_uses_distinct_trusted_flow_and_clean_return_targets() -> None:
    config = _tiny_config()
    diagnostic = DiagnosticConfig(
        theta_points=16, radius_points=17, radius_limit=0.30
    )
    arrays = compute_landscape_arrays(
        _identity_models(config), config, diagnostic
    )

    assert arrays["trusted_flow_defect_by_seed"].shape == (1, 3, 3, 17, 16)
    assert arrays["clean_return_error_by_seed"].shape == (1, 3, 3, 17, 16)
    assert arrays["output_clean_set_distance_by_seed"].shape == (1, 3, 3, 17, 16)
    assert arrays["theta"][0] == pytest.approx(-np.pi)
    assert arrays["theta"][-1] < np.pi
    assert not np.any(np.isclose(arrays["theta"], np.pi))
    assert arrays["radius"][8] == pytest.approx(0.0)

    radius_index = 12
    theta_index = 3
    state = np.asarray(
        [[arrays["theta"][theta_index], arrays["radius"][radius_index]]]
    )
    clean = state.copy()
    clean[:, 1] = 0.0
    identity_output = state
    trusted = flows(config)["C"].advance(state)
    clean_target = flows(config)["C"].advance(clean)
    expected_flow = lifted_coordinate_error(identity_output, trusted)[0]
    expected_return = lifted_coordinate_error(identity_output, clean_target)[0]
    actual_flow = arrays["trusted_flow_defect_by_seed"][
        0, 0, ARM_KEYS.index("CLEAN"), radius_index, theta_index
    ]
    actual_return = arrays["clean_return_error_by_seed"][
        0, 0, ARM_KEYS.index("CLEAN"), radius_index, theta_index
    ]
    assert actual_flow == pytest.approx(expected_flow)
    assert actual_return == pytest.approx(expected_return)
    assert actual_flow != pytest.approx(actual_return)
    assert arrays["output_clean_set_distance_by_seed"][
        0, 0, ARM_KEYS.index("CLEAN"), radius_index, theta_index
    ] == pytest.approx(abs(state[0, 1]))


def test_fixed_forcing_selects_registered_phase_and_sequence_without_truncation() -> None:
    config = _tiny_config(seeds=(17,))
    initial, selected, metadata = select_fixed_forcing(config)
    full_initial, full_forcing, full_digest = forcing_bank(
        config, config.primary_noise_sigma
    )

    assert metadata["seed"] == 17
    assert metadata["phase_index"] == 0
    assert metadata["sequence_index"] == 0
    assert metadata["flat_index"] == 0
    assert metadata["steps"] == config.noise_steps
    assert metadata["full_forcing_bank_digest"] == full_digest
    assert np.array_equal(initial, full_initial[0:1])
    assert np.array_equal(selected, full_forcing[:, 0:1])
    assert selected.shape == (config.noise_steps, 1, 2)

    second_initial, second_forcing, second_metadata = select_fixed_forcing(
        config, phase_index=1, sequence_index=1
    )
    expected_index = config.noise_sequences + 1
    assert second_metadata["flat_index"] == expected_index
    assert np.array_equal(second_initial, full_initial[expected_index : expected_index + 1])
    assert np.array_equal(
        second_forcing, full_forcing[:, expected_index : expected_index + 1]
    )


def test_fixed_trajectory_rows_cover_every_transition_and_arm() -> None:
    config = _tiny_config(seeds=(17,))
    rows, selection = compute_trajectory_rows(_identity_models(config), config)

    assert selection["steps"] == config.noise_steps
    assert len(rows) == 3 * len(ARM_KEYS) * config.noise_steps
    for regime in ("C", "M", "E"):
        for arm in ARM_KEYS:
            selected = [
                row
                for row in rows
                if row["regime"] == regime and row["arm"] == arm
            ]
            assert [row["step"] for row in selected] == list(
                range(config.noise_steps)
            )
            assert all(row["seed"] == 17 for row in selected)
            assert all(
                row["forcing_sigma"] == config.primary_noise_sigma
                for row in selected
            )
            for current, following in pairwise(selected):
                assert current["next_theta_lifted"] == pytest.approx(
                    following["theta_lifted"]
                )
                assert current["next_normal"] == pytest.approx(following["normal"])
            assert "trusted_forced_next_theta_lifted" in selected[-1]
            assert "trusted_forced_next_normal" in selected[-1]


def test_periodic_plot_path_breaks_the_seam_without_changing_samples() -> None:
    theta, radius = periodic_path_for_plot(
        np.asarray([3.0, 3.2, 3.4]), np.asarray([0.0, 0.1, 0.2])
    )
    assert np.count_nonzero(np.isnan(theta)) == 1
    assert np.count_nonzero(np.isnan(radius)) == 1
    assert theta[~np.isnan(theta)] == pytest.approx(
        [3.0, 3.2 - 2.0 * np.pi, 3.4 - 2.0 * np.pi]
    )
    assert radius[~np.isnan(radius)] == pytest.approx([0.0, 0.1, 0.2])


def test_exact_parent_binding_refuses_manifest_or_checkpoint_drift() -> None:
    expected = {
        "checkpoints/seed_1_clean.pt": "a" * 64,
        "checkpoints/seed_1_recovery.pt": "b" * 64,
    }
    manifest = {
        "outputs": {
            name: {"sha256": digest, "bytes": 1}
            for name, digest in expected.items()
        }
    }
    assert validate_parent_manifest_binding(
        manifest,
        actual_manifest_sha256="c" * 64,
        expected_manifest_sha256="c" * 64,
        expected_checkpoint_sha256=expected,
    ) == expected

    with pytest.raises(ValueError, match="manifest hash mismatch"):
        validate_parent_manifest_binding(
            manifest,
            actual_manifest_sha256="d" * 64,
            expected_manifest_sha256="c" * 64,
            expected_checkpoint_sha256=expected,
        )

    changed = copy.deepcopy(manifest)
    changed["outputs"]["checkpoints/seed_1_clean.pt"]["sha256"] = "e" * 64
    with pytest.raises(ValueError, match="checkpoint binding mismatch"):
        validate_parent_manifest_binding(
            changed,
            actual_manifest_sha256="c" * 64,
            expected_manifest_sha256="c" * 64,
            expected_checkpoint_sha256=expected,
        )
