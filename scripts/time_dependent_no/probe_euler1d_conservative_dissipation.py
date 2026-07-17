"""Probe local conservative dissipation on a frozen 1D Euler flux FNO.

The intervention adds the zero-boundary interior flux
-kappa * (dx / dt) * (U_right - U_left). On a uniform mesh its contribution
is kappa times the discrete Laplacian. This is a frozen causal diagnostic, not
a trained method or a positivity limiter.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.time_dependent_no.diagnose_euler1d_scale_spectra import (  # noqa: E402
    EPS,
    FrozenModel,
    load_frozen_model,
    prediction_validity,
    primitive_to_conservative_np,
    select_device,
    state_metrics,
)
from scripts.time_dependent_no.train_euler1d_target_ladder import (  # noqa: E402
    conservative_total_np,
    json_ready,
    separated_pressure_fronts_np,
)
from utility.time_dependent_no.euler1d import (  # noqa: E402
    conservative_to_primitive,
    make_euler1d_batch,
    primitive_to_conservative,
    update_from_face_flux,
)
from utility.time_dependent_no.euler1d_data import (  # noqa: E402
    Euler1DNPZ,
    load_euler1d_npz,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--diffusion-strength",
        action="append",
        type=float,
        required=True,
        help="Dimensionless kappa; repeat for a small coefficient ladder.",
    )
    parser.add_argument("--max-calls", type=int, default=50)
    parser.add_argument("--report-call", action="append", type=int, default=[])
    parser.add_argument("--shock-radius-cells", type=int, default=4)
    parser.add_argument("--width-radius-cells", type=int, default=8)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260716)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args(argv)
    raw_strengths = [float(value) for value in args.diffusion_strength]
    strengths = sorted(set(raw_strengths))
    if len(strengths) != len(raw_strengths):
        parser.error("--diffusion-strength values must be unique")
    if not strengths or strengths[0] != 0.0:
        parser.error("the ladder must include the exact kappa=0 control")
    if any(value < 0.0 or value > 0.25 for value in strengths):
        parser.error("diffusion strengths must lie in [0, 0.25]")
    if args.max_calls < 1:
        parser.error("--max-calls must be positive")
    report_calls = sorted(set(args.report_call or [20, args.max_calls]))
    if report_calls[0] < 1 or report_calls[-1] > args.max_calls:
        parser.error("report calls must lie in [1, max-calls]")
    if args.shock_radius_cells < 0 or args.width_radius_cells < 0:
        parser.error("shock and width radii must be nonnegative")
    if args.bootstrap_replicates < 1:
        parser.error("--bootstrap-replicates must be positive")
    args.diffusion_strength = strengths
    args.report_call = report_calls
    return args


def diffusive_face_flux(
    conservative: torch.Tensor,
    cell_volume: torch.Tensor,
    dt: torch.Tensor,
    kappa: float,
) -> torch.Tensor:
    """Return a zero-boundary interior flux giving Laplacian damping."""

    if conservative.ndim != 3:
        raise ValueError("conservative must have shape [batch, cells, channels]")
    if cell_volume.shape != conservative.shape[:2]:
        raise ValueError("cell_volume must match conservative cells")
    if dt.reshape(-1).shape[0] != conservative.shape[0]:
        raise ValueError("dt must contain one value per sample")
    if kappa < 0.0:
        raise ValueError("kappa must be nonnegative")
    correction = conservative.new_zeros(
        conservative.shape[0],
        conservative.shape[1] + 1,
        conservative.shape[2],
    )
    if kappa == 0.0:
        return correction
    face_length = 0.5 * (cell_volume[:, :-1] + cell_volume[:, 1:])
    scale = kappa * face_length / dt.reshape(-1, 1)
    jump = conservative[:, 1:] - conservative[:, :-1]
    correction[:, 1:-1] = -scale.unsqueeze(-1) * jump
    return correction


def pressure_front_effective_width_np(
    primitive: np.ndarray,
    x: np.ndarray,
    *,
    num_fronts: int = 2,
    radius_cells: int = 8,
    min_separation_cells: int = 4,
) -> np.ndarray:
    """Estimate front width by local gradient inverse participation ratio."""

    values = np.asarray(primitive, dtype=np.float64)
    if values.ndim != 3 or values.shape[-1] != 3:
        raise ValueError("primitive must have shape [batch, cells, 3]")
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 or x.shape[0] != values.shape[1]:
        raise ValueError("x must match the primitive cell dimension")
    positions, _ = separated_pressure_fronts_np(
        values,
        x,
        num_fronts=num_fronts,
        min_separation_cells=min_separation_cells,
    )
    face_x = 0.5 * (x[:-1] + x[1:])
    face_dx = np.diff(x)
    pressure_jump = np.abs(np.diff(values[..., 2], axis=1))
    widths = np.full((values.shape[0], num_fronts), np.nan, dtype=np.float64)
    for sample_id in range(values.shape[0]):
        for front_id in range(num_fronts):
            position = positions[sample_id, front_id]
            if not np.isfinite(position):
                continue
            center = int(np.argmin(np.abs(face_x - position)))
            lo = max(0, center - radius_cells)
            hi = min(face_x.size, center + radius_cells + 1)
            local_jump = pressure_jump[sample_id, lo:hi]
            total_jump = float(np.sum(local_jump))
            denominator = float(np.sum(local_jump**2 / face_dx[lo:hi]))
            if total_jump > EPS and denominator > EPS:
                widths[sample_id, front_id] = total_jump**2 / denominator
    return widths


def endpoint_metrics(
    prediction_primitive: np.ndarray,
    prediction_conservative: np.ndarray,
    truth_primitive: np.ndarray,
    truth_conservative: np.ndarray,
    x: np.ndarray,
    gamma: float,
    *,
    shock_radius_cells: int,
    width_radius_cells: int,
) -> dict[str, float]:
    metrics = state_metrics(
        prediction_primitive,
        prediction_conservative,
        truth_primitive,
        truth_conservative,
        x[None, :],
        gamma,
        shock_radius_cells=shock_radius_cells,
    )
    prediction_width = pressure_front_effective_width_np(
        prediction_primitive,
        x,
        radius_cells=width_radius_cells,
    )
    truth_width = pressure_front_effective_width_np(
        truth_primitive,
        x,
        radius_cells=width_radius_cells,
    )
    prediction_mean = float(np.nanmean(prediction_width))
    truth_mean = float(np.nanmean(truth_width))
    width_ratio = prediction_mean / max(truth_mean, EPS)
    result = {key: float(value[0]) for key, value in metrics.items()}
    result["shock_width_ratio"] = width_ratio
    result["shock_width_relative_abs_error"] = abs(width_ratio - 1.0)
    predicted_total = conservative_total_np(
        prediction_primitive[0],
        x,
        gamma,
    )
    truth_total = conservative_total_np(truth_primitive[0], x, gamma)
    result["conservative_total_error"] = float(
        np.linalg.norm(predicted_total - truth_total)
        / max(np.linalg.norm(truth_total), EPS)
    )
    return result


def rollout_case(
    frozen: FrozenModel,
    source: Euler1DNPZ,
    case_id: int,
    kappa: float,
    *,
    max_calls: int,
    report_calls: set[int],
    shock_radius_cells: int,
    width_radius_cells: int,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    stride = frozen.stride
    current_primitive = (
        torch.from_numpy(source.data[case_id, 0]).unsqueeze(0).to(device)
    )
    current_conservative = primitive_to_conservative(
        current_primitive,
        gamma=source.gamma,
    )
    x = torch.from_numpy(source.x[case_id]).unsqueeze(0).to(device)
    left = torch.from_numpy(source.left_states[case_id]).unsqueeze(0).to(device)
    right = torch.from_numpy(source.right_states[case_id]).unsqueeze(0).to(device)
    recurrent_coordinates = str(
        getattr(frozen.args, "recurrent_coordinates", "primitive")
    )
    valid_calls = 0
    failure_call: int | None = None
    termination_reason = ""
    min_density = float("inf")
    min_pressure = float("inf")
    boundary_closure_max = 0.0
    correction_ratios: list[float] = []
    endpoint_rows: list[dict[str, Any]] = []

    with torch.no_grad():
        for call in range(1, max_calls + 1):
            source_frame = (call - 1) * stride
            target_frame = call * stride
            dt = torch.tensor(
                [
                    source.t[case_id, target_frame]
                    - source.t[case_id, source_frame]
                ],
                dtype=current_primitive.dtype,
                device=device,
            )
            model_primitive = (
                conservative_to_primitive(
                    current_conservative,
                    gamma=source.gamma,
                )
                if recurrent_coordinates == "conservative"
                else current_primitive
            )
            batch = make_euler1d_batch(
                model_primitive,
                x,
                dt,
                current_conservative_state=(
                    current_conservative
                    if recurrent_coordinates == "conservative"
                    else None
                ),
                gamma=source.gamma,
                left_boundary_primitive=left,
                right_initial_primitive=right,
            )
            raw = frozen.model(batch)
            decoded = frozen.adapter(raw, batch)
            base_flux = decoded.aux.get("face_flux")
            if not isinstance(base_flux, torch.Tensor):
                raise RuntimeError("D024 requires a decoded face flux")
            if kappa == 0.0:
                proposed_conservative = decoded.conservative.detach()
                proposed_primitive = decoded.primitive.detach()
                corrected_flux = base_flux
            else:
                correction_flux = diffusive_face_flux(
                    current_conservative,
                    batch.geometry.cell_volume,
                    dt,
                    kappa,
                )
                corrected_flux = base_flux + correction_flux
                proposed_conservative = update_from_face_flux(
                    batch,
                    corrected_flux,
                ).detach()
                proposed_primitive = conservative_to_primitive(
                    proposed_conservative,
                    gamma=source.gamma,
                ).detach()

            proposal_np = proposed_primitive.cpu().numpy().astype(np.float64)
            conservative_np = (
                proposed_conservative.cpu().numpy().astype(np.float64)
            )
            current_conservative_np = (
                current_conservative.cpu().numpy().astype(np.float64)
            )
            base_conservative_np = (
                decoded.conservative.detach().cpu().numpy().astype(np.float64)
            )
            base_update_np = base_conservative_np - current_conservative_np
            correction_update_np = conservative_np - base_conservative_np
            correction_ratios.append(
                float(
                    np.linalg.norm(correction_update_np)
                    / max(np.linalg.norm(base_update_np), EPS)
                )
            )
            volume_np = (
                batch.geometry.cell_volume.cpu().numpy().astype(np.float64)
            )
            area_np = batch.geometry.face_area.cpu().numpy().astype(np.float64)
            corrected_flux_np = (
                corrected_flux.detach().cpu().numpy().astype(np.float64)
            )
            weighted_delta = np.sum(
                (conservative_np - current_conservative_np)
                * volume_np[..., None],
                axis=1,
            )
            boundary_flux = (
                corrected_flux_np[:, 0] * area_np[:, 0, None]
                + corrected_flux_np[:, -1] * area_np[:, -1, None]
            )
            closure = weighted_delta + float(dt.item()) * boundary_flux
            boundary_closure_max = max(
                boundary_closure_max,
                float(np.max(np.abs(closure))),
            )
            min_density = min(min_density, float(np.nanmin(proposal_np[..., 0])))
            min_pressure = min(min_pressure, float(np.nanmin(proposal_np[..., 2])))
            valid, reasons = prediction_validity(proposal_np)
            if not bool(valid[0]):
                failure_call = call
                termination_reason = reasons[0]
                break

            valid_calls = call
            if call in report_calls:
                truth_primitive = source.data[
                    case_id : case_id + 1,
                    target_frame,
                ].astype(np.float64)
                truth_conservative = primitive_to_conservative_np(
                    truth_primitive,
                    source.gamma,
                )
                row = endpoint_metrics(
                    proposal_np,
                    conservative_np,
                    truth_primitive,
                    truth_conservative,
                    source.x[case_id],
                    source.gamma,
                    shock_radius_cells=shock_radius_cells,
                    width_radius_cells=width_radius_cells,
                )
                row.update(
                    {
                        "case_id": case_id,
                        "call": call,
                        "diffusion_strength": kappa,
                    }
                )
                endpoint_rows.append(row)
            current_primitive = proposed_primitive
            current_conservative = proposed_conservative

    trajectory = {
        "case_id": case_id,
        "diffusion_strength": kappa,
        "valid_calls": valid_calls,
        "failure_call": failure_call,
        "termination_reason": termination_reason,
        "completed_max_calls": valid_calls == max_calls,
        "min_density": min_density,
        "min_pressure": min_pressure,
        "boundary_closure_max_abs": boundary_closure_max,
        "correction_update_over_base_mean": float(np.mean(correction_ratios)),
        "correction_update_over_base_max": float(np.max(correction_ratios)),
    }
    for call in report_calls:
        trajectory[f"completed_h{call}"] = valid_calls >= call
    return trajectory, endpoint_rows


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(replicates, values.size))
    means = np.mean(values[indices], axis=1)
    lo, hi = np.quantile(means, (0.025, 0.975))
    return float(lo), float(hi)


def mean_ratio(
    candidate: list[dict[str, Any]],
    baseline: list[dict[str, Any]],
    key: str,
) -> dict[str, Any]:
    base_by_case = {int(row["case_id"]): row for row in baseline}
    pairs = [
        (
            float(row[key]),
            float(base_by_case[int(row["case_id"])][key]),
        )
        for row in candidate
        if int(row["case_id"]) in base_by_case
        and math.isfinite(float(row[key]))
        and math.isfinite(float(base_by_case[int(row["case_id"])][key]))
    ]
    ratios = [candidate_value / max(base_value, EPS) for candidate_value, base_value in pairs]
    return {
        "common_cases": len(ratios),
        "mean_ratio": float(np.mean(ratios)) if ratios else float("nan"),
        "ratio_of_means": (
            float(np.mean([pair[0] for pair in pairs]))
            / max(float(np.mean([pair[1] for pair in pairs])), EPS)
            if pairs
            else float("nan")
        ),
        "wins": int(np.sum(np.asarray(ratios) < 1.0)),
    }


def summarize(
    trajectories: list[dict[str, Any]],
    endpoints: list[dict[str, Any]],
    strengths: list[float],
    report_calls: list[int],
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    paired: dict[str, Any] = {}
    baseline = [
        row for row in trajectories if float(row["diffusion_strength"]) == 0.0
    ]
    baseline_endpoints = {
        call: [
            row
            for row in endpoints
            if float(row["diffusion_strength"]) == 0.0
            and int(row["call"]) == call
        ]
        for call in report_calls
    }
    endpoint_keys = (
        "state_cons_scaled_rmse",
        "shock_cons_scaled_rmse",
        "smooth_cons_scaled_rmse",
        "error_high_25_64_rms",
        "error_tail_65_nyquist_rms",
        "error_d2_rms",
        "front_position_mae",
        "front_strength_relative_l1",
        "shock_width_relative_abs_error",
        "state_tv_ratio",
        "conservative_total_error",
    )
    for strength_id, strength in enumerate(strengths):
        selected = [
            row
            for row in trajectories
            if float(row["diffusion_strength"]) == strength
        ]
        row: dict[str, Any] = {
            "diffusion_strength": strength,
            "num_cases": len(selected),
            "valid_calls_mean": float(
                np.mean([float(item["valid_calls"]) for item in selected])
            ),
            "valid_calls_median": float(
                np.median([float(item["valid_calls"]) for item in selected])
            ),
            "boundary_closure_max_abs": float(
                np.max(
                    [float(item["boundary_closure_max_abs"]) for item in selected]
                )
            ),
            "correction_update_over_base_mean": float(
                np.mean(
                    [
                        float(item["correction_update_over_base_mean"])
                        for item in selected
                    ]
                )
            ),
        }
        for call in report_calls:
            endpoint_selected = [
                item
                for item in endpoints
                if float(item["diffusion_strength"]) == strength
                and int(item["call"]) == call
            ]
            row[f"h{call}_completed"] = len(endpoint_selected)
            for key in endpoint_keys:
                values = [
                    float(item[key])
                    for item in endpoint_selected
                    if math.isfinite(float(item[key]))
                ]
                row[f"h{call}_{key}_mean"] = (
                    float(np.mean(values)) if values else float("nan")
                )
        summary_rows.append(row)
        if strength == 0.0:
            continue
        valid_diff = np.asarray(
            [
                float(candidate["valid_calls"]) - float(control["valid_calls"])
                for candidate, control in zip(selected, baseline, strict=True)
            ],
            dtype=np.float64,
        )
        ci = bootstrap_mean_ci(
            valid_diff,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed + strength_id,
        )
        entry: dict[str, Any] = {
            "valid_calls_difference_mean": float(np.mean(valid_diff)),
            "valid_calls_difference_ci95": list(ci),
            "survival_wins_ties_losses": [
                int(np.sum(valid_diff > 0.0)),
                int(np.sum(valid_diff == 0.0)),
                int(np.sum(valid_diff < 0.0)),
            ],
            "endpoints": {},
        }
        for call in report_calls:
            candidate_endpoint = [
                item
                for item in endpoints
                if float(item["diffusion_strength"]) == strength
                and int(item["call"]) == call
            ]
            entry["endpoints"][str(call)] = {
                key: mean_ratio(
                    candidate_endpoint,
                    baseline_endpoints[call],
                    key,
                )
                for key in endpoint_keys
            }
        paired[str(strength)] = entry
    return summary_rows, paired


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    device = select_device(args.device, args.gpu)
    source = load_euler1d_npz(args.data_path)
    frozen = load_frozen_model("frozen_flux", args.checkpoint, device)
    if frozen.adapter.__class__.__name__ != "FluxTargetAdapter":
        raise RuntimeError("D024 requires a flux-target checkpoint")
    if frozen.stride != 1:
        raise RuntimeError("D024 is preregistered for the stride-1 flux baseline")
    max_available = (source.num_frames - 1) // frozen.stride
    if args.max_calls > max_available:
        raise ValueError(f"max-calls exceeds available horizon {max_available}")

    trajectories: list[dict[str, Any]] = []
    endpoints: list[dict[str, Any]] = []
    report_calls = set(args.report_call)
    for strength_id, strength in enumerate(args.diffusion_strength, start=1):
        print(
            f"D024 kappa={strength:g} "
            f"({strength_id}/{len(args.diffusion_strength)})",
            flush=True,
        )
        for case_id in frozen.test_cases.tolist():
            trajectory, case_endpoints = rollout_case(
                frozen,
                source,
                int(case_id),
                strength,
                max_calls=args.max_calls,
                report_calls=report_calls,
                shock_radius_cells=args.shock_radius_cells,
                width_radius_cells=args.width_radius_cells,
                device=device,
            )
            trajectories.append(trajectory)
            endpoints.extend(case_endpoints)

    summary_rows, paired = summarize(
        trajectories,
        endpoints,
        args.diffusion_strength,
        args.report_call,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    baseline = next(
        row for row in summary_rows if float(row["diffusion_strength"]) == 0.0
    )
    final_call = args.report_call[-1]
    first_call = args.report_call[0]
    nonzero = [
        row for row in summary_rows if float(row["diffusion_strength"]) > 0.0
    ]
    candidate = max(
        nonzero,
        key=lambda row: (
            int(row[f"h{final_call}_completed"]),
            -float(row[f"h{first_call}_state_cons_scaled_rmse_mean"]),
        ),
    )
    candidate_strength = float(candidate["diffusion_strength"])
    candidate_paired = paired[str(candidate_strength)]["endpoints"][
        str(first_call)
    ]
    gates = {
        "horizon50_stability_signal": int(
            candidate[f"h{final_call}_completed"]
        )
        >= max(32, int(baseline[f"h{final_call}_completed"]) + 1),
        "horizon20_completion_noninferior": int(
            candidate[f"h{first_call}_completed"]
        )
        >= int(baseline[f"h{first_call}_completed"]),
        "horizon20_state_error_ratio_le_1p20": float(
            candidate_paired["state_cons_scaled_rmse"]["ratio_of_means"]
        )
        <= 1.20,
        "horizon20_front_position_ratio_le_1p15": float(
            candidate_paired["front_position_mae"]["ratio_of_means"]
        )
        <= 1.15,
        "horizon20_width_error_ratio_le_1p25": float(
            candidate_paired["shock_width_relative_abs_error"]["ratio_of_means"]
        )
        <= 1.25,
        "horizon20_high_band_reduced": float(
            candidate_paired["error_high_25_64_rms"]["ratio_of_means"]
        )
        < 1.0,
        "boundary_closure_le_5e_5": float(
            candidate["boundary_closure_max_abs"]
        )
        <= 5.0e-5,
    }
    if all(gates.values()):
        classification = "stabilizes_with_bounded_short_horizon_cost"
    elif int(candidate[f"h{final_call}_completed"]) > int(
        baseline[f"h{final_call}_completed"]
    ):
        classification = "stability_gain_with_accuracy_or_smearing_tradeoff"
    else:
        classification = "no_material_stability_gain"
    report = {
        "experiment": "D024_frozen_conservative_dissipation_probe",
        "claim_boundary": (
            "Single frozen checkpoint causal probe; coefficient selection on "
            "this test split is exploratory, not a trained-method result."
        ),
        "dataset": {
            "cases": source.num_cases,
            "frames": source.num_frames,
            "cells": source.num_cells,
        },
        "model": {
            "parameter_count": frozen.parameter_count,
            "test_cases": int(frozen.test_cases.size),
            "checkpoint_sha256": frozen.checkpoint_sha256,
        },
        "intervention": {
            "strengths": args.diffusion_strength,
            "boundary_correction": "exactly_zero",
            "uniform_grid_update": (
                "kappa_times_discrete_laplacian_of_current_conservative_state"
            ),
            "nyquist_gain_per_call": {
                str(value): 1.0 - 4.0 * value
                for value in args.diffusion_strength
            },
        },
        "summary": summary_rows,
        "paired_vs_zero": paired,
        "selected_exploratory_candidate": candidate_strength,
        "promotion_gates": gates,
        "classification": classification,
        "runtime_seconds": time.perf_counter() - start,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "trajectory_metrics.csv", trajectories)
    write_csv(args.output_dir / "endpoint_metrics.csv", endpoints)
    write_csv(args.output_dir / "summary.csv", summary_rows)
    (args.output_dir / "report.json").write_text(
        json.dumps(json_ready(report), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "candidate": candidate_strength,
                "classification": classification,
                "gates": gates,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return report


def main(argv: Sequence[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
