from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from scripts.time_dependent_no.analyze_pcno_moving_front_wake import (
    ARMS,
    PHASES,
    error_metrics,
    evaluate_sequence,
    moving_cases,
    scaled_auxiliary_tensors,
    spatial_masks,
)
from scripts.time_dependent_no.visualize_pcno_moving_front_wake import (
    render_figures,
)
from utility.time_dependent_no.pcno_shock_representation import (
    DISPLACEMENT,
    PULSE_WIDTH,
    build_structured_cell_grid,
)


class _DummyResidualModel(torch.nn.Module):
    def prepare_fourier_tensors(self, nodes, node_weights):
        scalar = torch.zeros((), dtype=nodes.dtype, device=nodes.device)
        return (scalar,) * 6

    def forward(self, values, aux, *, fourier_tensors=None):
        assert fourier_tensors is not None
        scale = torch.mean(torch.abs(aux[-1]))
        return 0.05 * scale * values[..., 3:4]


def test_exact_moving_front_geometry_and_zero_true_wake() -> None:
    grid = build_structured_cell_grid((64, 32))
    start = 0.25 + 0.875 * grid.hx
    for family, expected_front_count in (("step", 2), ("pulse", 4)):
        cases = moving_cases(grid, family, 0.875)
        assert len(cases) == 3
        for call, case in enumerate(cases):
            assert case.position == start + call * DISPLACEMENT
            assert len(case.increment_fronts) == expected_front_count
            masks = spatial_masks(case, grid, initial_position=start)
            assert not np.any(masks["active"] & masks["wake"])
            assert not np.any(masks["ahead"] & (masks["active"] | masks["wake"]))
            assert np.all(masks["active"] | masks["wake"] | masks["ahead"])
            if call == 0:
                assert not np.any(masks["wake"])
            else:
                assert np.any(masks["wake"])
                assert np.max(np.abs(case.increment[:, masks["wake"]])) == 0.0
        if family == "pulse":
            assert cases[0].increment_fronts[2].position == start + PULSE_WIDTH


def test_gradient_weight_scale_only_changes_last_auxiliary_tensor() -> None:
    grid = build_structured_cell_grid((16, 8))
    native = scaled_auxiliary_tensors(
        grid, torch.device("cpu"), gradient_weight_scale=1.0
    )
    suppressed = scaled_auxiliary_tensors(
        grid, torch.device("cpu"), gradient_weight_scale=0.1
    )
    assert len(native) == len(suppressed) == 5
    for left, right in zip(native[:-1], suppressed[:-1], strict=True):
        assert torch.equal(left, right)
    assert torch.allclose(suppressed[-1], 0.1 * native[-1], rtol=0.0, atol=0.0)


def test_teacher_recurrent_decomposition_and_region_metrics() -> None:
    grid = build_structured_cell_grid((16, 8))
    cases = moving_cases(grid, "step", 0.0)
    records, arrays = evaluate_sequence(
        _DummyResidualModel(),
        cases,
        grid,
        torch.device("cpu"),
        gradient_weight_scale=1.0,
    )
    assert records[0]["teacher_recurrent_call0_max_abs"] == 0.0
    assert max(record["decomposition_max_abs"] for record in records) <= 1.0e-15
    masks = spatial_masks(cases[2], grid, initial_position=cases[0].position)
    metrics = error_metrics(
        arrays["call2__recurrent_residual"],
        arrays["call2__recurrent_next"],
        cases[2],
        grid,
        masks,
    )
    assert metrics["regions"]["wake"]["valid"]
    assert metrics["regions"]["wake"]["true_residual_max_abs"] == 0.0
    np.testing.assert_allclose(
        arrays["call2__recurrent_error"],
        arrays["call2__teacher_error"]
        + arrays["call2__propagated_contribution"],
        rtol=0.0,
        atol=1.0e-15,
    )


def _figure_fixture(output_dir: Path) -> None:
    nx, ny = 16, 4
    x = (np.arange(nx) + 0.5) / nx
    y = (np.arange(ny) + 0.5) * 0.5 / ny
    arrays: dict[str, np.ndarray] = {"x_centers": x, "y_centers": y}
    rows = []
    for family in ("step", "pulse", "smooth_tanh", "smooth_sine"):
        for phase in PHASES:
            phase_key = str(float(phase)).replace(".", "p")
            for call in range(3):
                position = 0.25 + phase / nx + call * DISPLACEMENT
                true_profile = np.zeros(nx)
                true_profile[(x >= position) & (x < position + DISPLACEMENT)] = 1.0
                if family == "pulse":
                    true_profile[(x >= position) & (x < position + DISPLACEMENT)] = -1.0
                    true_profile[
                        (x >= position + PULSE_WIDTH)
                        & (x < position + PULSE_WIDTH + DISPLACEMENT)
                    ] = 1.0
                true = np.repeat(true_profile[None, :], ny, axis=0)
                active = np.abs(true_profile) > 0
                wake = (x >= 0.25) & (x < position) & ~active
                ahead = ~(active | wake)
                case_prefix = f"case__{family}__p{phase_key}__call{call}"
                arrays[f"{case_prefix}__current"] = np.zeros_like(true)
                arrays[f"{case_prefix}__target"] = true
                arrays[f"{case_prefix}__true_residual"] = true
                arrays[f"{case_prefix}__mask_active"] = active
                arrays[f"{case_prefix}__mask_wake"] = wake
                arrays[f"{case_prefix}__mask_ahead"] = ahead
                fronts = [position, position + DISPLACEMENT]
                if family == "pulse":
                    fronts += [position + PULSE_WIDTH, position + PULSE_WIDTH + DISPLACEMENT]
                for arm_index, arm in enumerate(ARMS):
                    for seed in (1701, 1702, 1703):
                        amplitude = 0.01 * (arm_index + 1) * (seed - 1699)
                        error_profile = amplitude * np.sin(8 * np.pi * x)
                        error = np.repeat(error_profile[None, :], ny, axis=0)
                        prefix = f"{arm}__s{seed}__{family}__p{phase_key}__call{call}"
                        arrays[f"{prefix}__teacher_residual"] = true + 0.5 * error
                        arrays[f"{prefix}__recurrent_residual"] = true + error
                        arrays[f"{prefix}__teacher_error"] = 0.5 * error
                        arrays[f"{prefix}__recurrent_error"] = error
                        arrays[f"{prefix}__propagated_contribution"] = 0.5 * error
                        arrays[f"{prefix}__recurrent_state"] = np.zeros_like(true)
                        arrays[f"{prefix}__teacher_next"] = true + 0.5 * error
                        arrays[f"{prefix}__recurrent_next"] = true + error
                        arrays[f"{prefix}__recurrent_next_error"] = error
                        region = {
                            "valid": call > 0,
                            "normalized_l1": amplitude if call > 0 else None,
                            "oscillatory_mass": amplitude / 2 if call > 0 else None,
                            "coverage_above_1e_3": 0.75 if call > 0 else None,
                            "true_residual_max_abs": 0.0 if call > 0 else None,
                        }
                        decomposition = {
                            "valid": call > 0,
                            "fresh_energy": amplitude if call > 0 else None,
                            "propagated_energy": amplitude if call > 0 else None,
                            "cross_inner_product": 0.0 if call > 0 else None,
                            "propagated_component_energy_share": 0.5 if call > 0 else None,
                        }
                        for path in ("teacher", "recurrent"):
                            rows.append(
                                {
                                    "arm": arm,
                                    "seed": seed,
                                    "family": family,
                                    "phase": phase,
                                    "call": call,
                                    "path": path,
                                    "increment_front_positions": fronts,
                                    "metrics": {
                                        "relative_increment_l2": amplitude,
                                        "regions": {
                                            "wake": region,
                                            "active": region,
                                            "ahead": region,
                                        },
                                    },
                                    "decomposition": {
                                        "wake": decomposition,
                                        "active": decomposition,
                                        "ahead": decomposition,
                                    },
                                }
                            )
    output_dir.mkdir(parents=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(rows, allow_nan=False), encoding="utf-8"
    )
    np.savez_compressed(output_dir / "arrays.npz", **arrays)


def test_checkpoint_free_figure_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "wake"
    _figure_fixture(output)
    paths = render_figures(output)
    assert len(paths) == 12
    assert {path.suffix for path in paths} == {".pdf", ".png"}
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths)
