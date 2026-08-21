#!/usr/bin/env python3
"""Run the W26-L2-P0/P1 synthetic CPU contract dry-run."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pcno.pcno import PCNO, compute_Fourier_modes
from utility.time_dependent_no.pcno_shock_representation import (
    ANCHORS,
    CASE_FAMILIES,
    DOMAIN_LENGTHS,
    HELD_PHASES,
    REGISTERED_RESOLUTIONS,
    FrontSpec,
    build_structured_cell_grid,
    gradient_distribution_statistics,
    grid_local_phase_positions,
    make_translated_front_case,
    pcno_aux_tensors,
    pcno_case_input,
    physical_cosine_filter,
    physical_front_band_mask,
    physical_spectrum_summary,
    registered_phase_positions,
    restrict_nested_cell_averages,
    scalar_gradient_variants,
    shock_representation_metrics,
    trace_pcno_differential_stages,
)

SCHEMA = "pcno_shock_representation_w26_l2_p0_p1_a1_dry_run_v1"
SEED = 20260811


def _resolution(value: str) -> tuple[int, int]:
    parts = value.lower().split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("resolution must use NXxNY")
    try:
        resolution = (int(parts[0]), int(parts[1]))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("resolution must use integer axes") from exc
    if resolution[0] < 2 or resolution[1] < 2:
        raise argparse.ArgumentTypeError("resolution axes must be at least two")
    return resolution


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="required A1 mode; performs no fitting and writes no artifact",
    )
    parser.add_argument(
        "--resolutions",
        type=_resolution,
        nargs="+",
        default=list(REGISTERED_RESOLUTIONS),
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args(argv)
    if not args.dry_run:
        parser.error("--dry-run is required under the current A1 authorization")
    return args


def _nested_restriction_rows(
    grids: dict[tuple[int, int], Any],
) -> list[dict[str, Any]]:
    ordered = sorted(grids, key=lambda item: item[0] * item[1])
    rows: list[dict[str, Any]] = []
    fixed_positions = registered_phase_positions(held_out=True)
    for coarse_resolution, fine_resolution in pairwise(ordered):
        coarse = grids[coarse_resolution]
        fine = grids[fine_resolution]
        if (
            fine_resolution[0] % coarse_resolution[0]
            or fine_resolution[1] % coarse_resolution[1]
        ):
            continue
        for family in CASE_FAMILIES:
            maximum_error = 0.0
            for position in fixed_positions:
                coarse_case = make_translated_front_case(
                    coarse, family, position=position
                )
                fine_case = make_translated_front_case(fine, family, position=position)
                for name in ("current", "target", "increment"):
                    restricted = restrict_nested_cell_averages(
                        getattr(fine_case, name),
                        fine_resolution=fine_resolution,
                        coarse_resolution=coarse_resolution,
                    )
                    maximum_error = max(
                        maximum_error,
                        float(np.max(np.abs(restricted - getattr(coarse_case, name)))),
                    )
            rows.append(
                {
                    "coarse_resolution": f"{coarse_resolution[0]}x{coarse_resolution[1]}",
                    "fine_resolution": f"{fine_resolution[0]}x{fine_resolution[1]}",
                    "family": family,
                    "case_count": len(fixed_positions),
                    "maximum_cell_average_restriction_error": maximum_error,
                }
            )
    return rows


def _gradient_rows(
    grids: dict[tuple[int, int], Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    phase_matched: list[dict[str, Any]] = []
    common_physical: list[dict[str, Any]] = []
    native_h = DOMAIN_LENGTHS[0] / 64.0
    fixed_position = ANCHORS[1] + HELD_PHASES[1] * native_h
    for resolution in sorted(grids, key=lambda item: item[0] * item[1]):
        grid = grids[resolution]
        local_position = grid_local_phase_positions(grid, held_out=True)[5]
        for contract, position, destination in (
            ("phase_matched", local_position, phase_matched),
            ("fixed_physical", fixed_position, common_physical),
        ):
            case = make_translated_front_case(grid, "step", position=position)
            variants = scalar_gradient_variants(case.current, grid)
            current_front = (FrontSpec(position, 1.0, 0.0),)
            for name, gradient in variants.items():
                if name == "h_scaled_raw":
                    continue
                statistics = gradient_distribution_statistics(
                    gradient,
                    grid,
                    current_front,
                )
                destination.append(
                    {
                        "contract": contract,
                        "resolution": f"{resolution[0]}x{resolution[1]}",
                        "position": position,
                        "stage": name,
                        **statistics,
                    }
                )
    return phase_matched, common_physical


def _metric_oracles(grid: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    position = ANCHORS[1] + HELD_PHASES[0] * (DOMAIN_LENGTHS[0] / 64.0)
    for family in CASE_FAMILIES:
        case = make_translated_front_case(grid, family, position=position)
        metrics = shock_representation_metrics(case.increment, case, grid)
        rows.append({"family": family, **metrics})
    return rows


def _filter_invariants(grid: Any) -> dict[str, Any]:
    xx, yy = np.meshgrid(grid.x_centers, grid.y_centers, indexing="xy")
    field = (
        0.75
        + np.cos(2.0 * np.pi * 4.0 * xx)
        + 0.2
        * np.cos(2.0 * np.pi * 20.0 * xx)
        * np.cos(2.0 * np.pi * 4.0 * yy / grid.lengths[1])
    )
    zero = physical_cosine_filter(field, grid, kind="zero")
    smooth = physical_cosine_filter(field, grid, kind="smooth")
    hard = physical_cosine_filter(field, grid, kind="hard")
    constant = np.full(grid.array_shape, 3.25)
    filtered_constant = physical_cosine_filter(constant, grid, kind="smooth")
    return {
        "zero_bypass_bitwise": bool(np.array_equal(field, zero)),
        "smooth_constant_maximum_error": float(
            np.max(np.abs(filtered_constant - constant))
        ),
        "input_spectrum": physical_spectrum_summary(field, grid),
        "smooth_spectrum": physical_spectrum_summary(smooth, grid),
        "hard_spectrum": physical_spectrum_summary(hard, grid),
    }


def _toy_pcno_trace(grid: Any) -> dict[str, Any]:
    torch.manual_seed(SEED)
    case = make_translated_front_case(grid, "step", position=ANCHORS[1])
    modes = torch.as_tensor(
        compute_Fourier_modes(2, [1, 1], list(DOMAIN_LENGTHS)),
        dtype=torch.float32,
    )
    model = PCNO(
        ndims=2,
        modes=modes,
        nmeasures=1,
        layers=[4, 4, 4],
        fc_dim=5,
        in_dim=4,
        out_dim=1,
    ).eval()
    aux = pcno_aux_tensors(grid)
    model_input = pcno_case_input(case, grid)
    original_state = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    front_mask = torch.as_tensor(
        physical_front_band_mask(grid, (case.position,)), dtype=torch.bool
    ).unsqueeze(0)
    trace = trace_pcno_differential_stages(
        model, model_input, aux, front_band_mask=front_mask
    )
    unchanged = all(
        torch.equal(value, original_state[name])
        for name, value in model.state_dict().items()
    )
    return {
        "resolution": f"{grid.nx}x{grid.ny}",
        "layer_count": len(trace.layers),
        "maximum_replay_error": float(
            torch.max(torch.abs(trace.model_output - trace.replay_output))
        ),
        "model_state_unchanged": unchanged,
        "layers": [
            {
                "layer": row.layer,
                "gw1": float(model.gws[row.layer].gw1.detach()),
                "saturation": row.saturation,
                "decoded_differential_ablation_response_rms": float(
                    torch.sqrt(
                        torch.mean(
                            torch.square(row.decoded_differential_ablation_response)
                        )
                    )
                ),
            }
            for row in trace.layers
        ],
    }


def run_dry_run(resolutions: Sequence[tuple[int, int]]) -> dict[str, Any]:
    unique_resolutions = tuple(dict.fromkeys(tuple(item) for item in resolutions))
    grids = {
        resolution: build_structured_cell_grid(resolution)
        for resolution in unique_resolutions
    }
    smallest = min(grids.values(), key=lambda grid: grid.nx * grid.ny)
    native = grids.get((64, 32), smallest)
    phase_rows, physical_rows = _gradient_rows(grids)
    return {
        "schema": SCHEMA,
        "working_ids": ["W26-L2-P0", "W26-L2-P1"],
        "stable_d_series_id_allocated": False,
        "scientific_interpretation_allowed": False,
        "terminology": {
            "gibbs_usage": "informal analogy only",
            "registered_effect_name": "shock-local oscillation or ripple",
        },
        "seed": SEED,
        "resolutions": [f"{grid.nx}x{grid.ny}" for grid in grids.values()],
        "geometry": [
            {
                "resolution": f"{grid.nx}x{grid.ny}",
                "total_volume": float(np.sum(grid.cell_volumes)),
                "maximum_coordinate_gradient_error": (
                    grid.geometry.maximum_coordinate_gradient_error
                ),
                "maximum_stencil_condition_number": (
                    grid.geometry.maximum_stencil_condition_number
                ),
            }
            for grid in grids.values()
        ],
        "nested_restriction": _nested_restriction_rows(grids),
        "gradient_phase_matched": phase_rows,
        "gradient_fixed_physical": physical_rows,
        "exact_prediction_metric_oracles": _metric_oracles(native),
        "filter_invariants": _filter_invariants(native),
        "toy_pcno_trace": _toy_pcno_trace(smallest),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_dry_run(args.resolutions)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
