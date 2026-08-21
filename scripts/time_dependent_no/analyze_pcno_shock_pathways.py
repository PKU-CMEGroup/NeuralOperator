#!/usr/bin/env python3
"""Frozen W26-L2-P1 branch attribution and differential-path analysis."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
from matplotlib import pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pcno.pcno import compute_gradient, graph_neighbor_average
from scripts.time_dependent_no.fit_pcno_shock_representation import (
    SCHEMA as P0_SCHEMA,
    WORKING_RUN_ID as P0_WORKING_RUN_ID,
    RegisteredCase,
    _case_guard,
    build_batch,
    build_model,
    build_registered_cases,
    mapping_sha256,
    model_state_sha256,
    population_sha256,
    sha256_file,
    write_json,
    write_npz,
)
from utility.time_dependent_no.pcno_ripple_diagnostics import trace_pcno_branches
from utility.time_dependent_no.pcno_shock_representation import (
    ANCHORS,
    CASE_FAMILIES,
    DOMAIN_LENGTHS,
    FRONT_BAND_WIDTH,
    HELD_PHASES,
    REGISTERED_RESOLUTIONS,
    FrontSpec,
    StructuredCellGrid,
    build_structured_cell_grid,
    gradient_distribution_statistics,
    make_translated_front_case,
    physical_front_band_mask,
    restrict_nested_cell_averages,
    trace_pcno_differential_stages,
)

SCHEMA = "w26_l2_p1_frozen_branch_pathways_v1"
WORKING_RUN_ID = "W26-L2-P1-FROZEN-S1701"
DEFAULT_P0_RUN = (
    ROOT / "artifacts" / "time_dependent_no" / "w26_l2_p0_full_s1701"
)
DEFAULT_OUTPUT = (
    ROOT / "artifacts" / "time_dependent_no" / "w26_l2_p1_frozen_s1701"
)
BRANCHES = ("spectral", "pointwise", "differential")
MASKS = tuple(f"{value:03b}" for value in range(8))
SOURCE_PATHS = (
    "pcno/pcno.py",
    "utility/time_dependent_no/cpg_mesh_contract.py",
    "utility/time_dependent_no/pcno_euler2d.py",
    "utility/time_dependent_no/pcno_ripple_diagnostics.py",
    "utility/time_dependent_no/pcno_shock_representation.py",
    "scripts/time_dependent_no/fit_pcno_shock_representation.py",
    "scripts/time_dependent_no/analyze_pcno_shock_pathways.py",
)
PROVENANCE_PATHS = (
    "docs/time_dependent_no/W26_L2_SHOCK_PATHWAY_PREREGISTRATION.md",
)
SCALAR_METRICS = (
    "relative_increment_l2",
    "normalized_overshoot",
    "normalized_undershoot",
    "oscillatory_mass_outside_front_band",
    "smooth_region_error",
    "positive_total_variation_excess",
    "total_variation_deficit",
    "increment_integral_error",
    "next_state_integral_error",
    "maximum_front_position_error",
    "maximum_front_strength_error",
    "maximum_front_thickness_excess",
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _source_hashes(paths: Sequence[str]) -> dict[str, str]:
    return {relative: sha256_file(ROOT / relative) for relative in paths}


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _environment(device: torch.device) -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": str(device),
        "cuda_version": torch.version.cuda,
        "cuda_device": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
    }


def branch_gains(mask: str, block_count: int) -> dict[tuple[int, str], float]:
    """Return the global S/P/D gain map for one registered mask."""

    if mask not in MASKS:
        raise ValueError(f"mask must lie in {MASKS}")
    if block_count <= 0:
        raise ValueError("block_count must be positive")
    bits = {name: float(int(bit)) for name, bit in zip(BRANCHES, mask, strict=True)}
    return {
        (layer, name): bits[name]
        for layer in range(block_count)
        for name in BRANCHES
    }


def _verify_p0_bundle(p0_run_dir: Path, *, smoke: bool) -> dict[str, Any]:
    required = (
        "manifest.json",
        "run_contract.json",
        "summary.json",
        "checkpoint.pt",
        "predictions.npz",
    )
    missing = [name for name in required if not (p0_run_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"P0 bundle is missing {missing}")
    manifest = _read_json(p0_run_dir / "manifest.json")
    contract = _read_json(p0_run_dir / "run_contract.json")
    summary = _read_json(p0_run_dir / "summary.json")
    if manifest.get("schema") != P0_SCHEMA or contract.get("schema") != P0_SCHEMA:
        raise ValueError("P0 schema does not match the registered fit")
    expected_run_id = (
        f"{P0_WORKING_RUN_ID}-SMOKE" if smoke else P0_WORKING_RUN_ID
    )
    if contract.get("working_run_id") != expected_run_id:
        raise ValueError("P0 working run ID does not match the requested analysis")
    if bool(summary.get("science_result")) == smoke:
        raise ValueError("P0 scientific/smoke status does not match --smoke")
    if summary.get("status") != "completed" or manifest.get("status") != "completed":
        raise ValueError("P0 input is not completed")
    for relative, expected in manifest["output_hashes"].items():
        path = p0_run_dir / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"P0 output hash mismatch: {relative}")
    config = contract["config"]
    if mapping_sha256(config) != contract["config_digest"]:
        raise ValueError("P0 config digest does not close")
    for relative, expected in manifest["source_sha256"].items():
        path = ROOT / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"current executable source differs from P0: {relative}")
    grid = build_structured_cell_grid(config["resolution"])
    records = build_registered_cases(grid, config, split="train")
    records += build_registered_cases(grid, config, split="held")
    if population_sha256(records) != manifest["population_sha256"]:
        raise ValueError("rebuilt P0 population does not close")
    return {
        "manifest": manifest,
        "contract": contract,
        "summary": summary,
        "config": config,
        "records": records,
        "grid": grid,
    }


def _phase_token(value: float) -> str:
    return f"{float(value):.3f}".rstrip("0").rstrip(".").replace(".", "p")


def build_common_physical_cases(
    grid: StructuredCellGrid,
    config: dict[str, Any],
    *,
    split: str,
) -> list[RegisteredCase]:
    """Build cases whose physical positions are fixed by the P0 native grid."""

    if split not in {"train", "held"}:
        raise ValueError("split must be train or held")
    native_h = DOMAIN_LENGTHS[0] / int(config["resolution"][0])
    records: list[RegisteredCase] = []
    for family in config["families"]:
        for anchor_index, anchor in enumerate(config["anchors"]):
            for phase in config[f"{split}_phases"]:
                position = float(anchor) + float(phase) * native_h
                records.append(
                    RegisteredCase(
                        case_id=(
                            f"{split}_{family}_a{anchor_index}_phase_"
                            f"{_phase_token(float(phase))}"
                        ),
                        split=split,
                        family=str(family),
                        anchor_index=anchor_index,
                        phase=float(phase),
                        position=position,
                        case=make_translated_front_case(
                            grid, str(family), position=position
                        ),
                    )
                )
    return records


def build_grid_local_phase_cases(
    grid: StructuredCellGrid,
    config: dict[str, Any],
) -> list[RegisteredCase]:
    """Build held discontinuities at the same subcell phase on every grid."""

    records: list[RegisteredCase] = []
    for family in ("step", "pulse"):
        for anchor_index, anchor in enumerate(config["anchors"]):
            for phase in config["held_phases"]:
                position = float(anchor) + float(phase) * grid.hx
                records.append(
                    RegisteredCase(
                        case_id=(
                            f"grid_local_{family}_a{anchor_index}_phase_"
                            f"{_phase_token(float(phase))}"
                        ),
                        split="held",
                        family=family,
                        anchor_index=anchor_index,
                        phase=float(phase),
                        position=position,
                        case=make_translated_front_case(
                            grid, family, position=position
                        ),
                    )
                )
    return records


def _replay_predict_records(
    model: torch.nn.Module,
    records: list[RegisteredCase],
    grid: StructuredCellGrid,
    device: torch.device,
    *,
    batch_size: int,
    mask: str | None = None,
    disabled_branch: tuple[int, str] | None = None,
) -> np.ndarray:
    if (mask is None) == (disabled_branch is None):
        raise ValueError("select exactly one global mask or local branch deletion")
    outputs: list[np.ndarray] = []
    gains = None if mask is None else branch_gains(mask, len(model.ws))
    for start in range(0, len(records), batch_size):
        selected = records[start : start + batch_size]
        model_input, _, aux = build_batch(selected, grid, device)
        output, _ = trace_pcno_branches(
            model,
            model_input,
            aux,
            branch_gains=gains,
            disabled_branch=disabled_branch,
            collect_summaries=False,
        )
        outputs.append(
            output[..., 0]
            .detach()
            .cpu()
            .numpy()
            .reshape(len(selected), grid.ny, grid.nx)
        )
    return np.concatenate(outputs, axis=0)


@torch.inference_mode()
def _direct_predict_records(
    model: torch.nn.Module,
    records: list[RegisteredCase],
    grid: StructuredCellGrid,
    device: torch.device,
    *,
    batch_size: int,
) -> np.ndarray:
    """Evaluate the maintained forward on the same batches used by replay."""

    outputs: list[np.ndarray] = []
    for start in range(0, len(records), batch_size):
        selected = records[start : start + batch_size]
        model_input, _, aux = build_batch(selected, grid, device)
        output = model(model_input, aux)
        outputs.append(
            output[..., 0]
            .detach()
            .cpu()
            .numpy()
            .reshape(len(selected), grid.ny, grid.nx)
        )
    return np.concatenate(outputs, axis=0)


def _predict_records(
    model: torch.nn.Module,
    records: list[RegisteredCase],
    grid: StructuredCellGrid,
    device: torch.device,
    *,
    batch_size: int,
    mask: str | None = None,
    disabled_branch: tuple[int, str] | None = None,
) -> np.ndarray:
    """Use the native output for mask 111 and replay actual interventions."""

    if mask == "111" and disabled_branch is None:
        return _direct_predict_records(
            model, records, grid, device, batch_size=batch_size
        )
    return _replay_predict_records(
        model,
        records,
        grid,
        device,
        batch_size=batch_size,
        mask=mask,
        disabled_branch=disabled_branch,
    )


def _metric_scalars(metrics: dict[str, Any]) -> dict[str, float]:
    fronts = metrics["fronts"]

    def maximum(name: str, *, positive_only: bool = False) -> float:
        values = [front[name] for front in fronts if front[name] is not None]
        if positive_only:
            values = [max(0.0, float(value)) for value in values]
        return max((float(value) for value in values), default=0.0)

    return {
        "relative_increment_l2": float(metrics["relative_increment_l2"]),
        "normalized_overshoot": float(metrics["normalized_overshoot"]),
        "normalized_undershoot": float(metrics["normalized_undershoot"]),
        "oscillatory_mass_outside_front_band": float(
            metrics["oscillatory_mass_outside_front_band"]
        ),
        "smooth_region_error": float(metrics["smooth_region_error"]),
        "positive_total_variation_excess": float(
            metrics["positive_total_variation_excess"]
        ),
        "total_variation_deficit": float(metrics["total_variation_deficit"]),
        "increment_integral_error": float(metrics["increment_integral_error"]),
        "next_state_integral_error": float(metrics["next_state_integral_error"]),
        "maximum_front_position_error": maximum("position_error"),
        "maximum_front_strength_error": maximum("strength_error"),
        "maximum_front_thickness_excess": maximum(
            "thickness_excess", positive_only=True
        ),
    }


def _case_rows(
    records: list[RegisteredCase],
    predictions: np.ndarray,
    grid: StructuredCellGrid,
    *,
    mask: str | None = None,
    local_deletion: tuple[int, str] | None = None,
) -> list[dict[str, Any]]:
    from scripts.time_dependent_no.fit_pcno_shock_representation import (
        case_metric_rows,
    )

    rows = case_metric_rows(records, predictions, grid)
    for row in rows:
        row["resolution"] = list(grid.resolution)
        row["mask"] = mask
        row["local_deletion"] = (
            None
            if local_deletion is None
            else {"layer": local_deletion[0], "branch": local_deletion[1]}
        )
        row["scalars"] = _metric_scalars(row["metrics"])
    return rows


def _aggregate(
    rows: Iterable[dict[str, Any]],
    predicate: Callable[[dict[str, Any]], bool],
) -> dict[str, Any]:
    selected = [row for row in rows if predicate(row)]
    return {
        "case_count": len(selected),
        "mean": {
            name: float(np.mean([row["scalars"][name] for row in selected]))
            if selected
            else None
            for name in SCALAR_METRICS
        },
        "median": {
            name: float(np.median([row["scalars"][name] for row in selected]))
            if selected
            else None
            for name in SCALAR_METRICS
        },
    }


def _mask_aggregates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, Callable[[dict[str, Any]], bool]] = {
        "all": lambda row: True,
        "train_discontinuous": lambda row: row["split"] == "train"
        and row["family"] in {"step", "pulse"},
        "held_discontinuous": lambda row: row["split"] == "held"
        and row["family"] in {"step", "pulse"},
        "held_discontinuous_phase_0p875": lambda row: row["split"] == "held"
        and row["family"] in {"step", "pulse"}
        and math.isclose(float(row["phase"]), 0.875),
        "held_smooth": lambda row: row["split"] == "held"
        and row["family"] in {"smooth_tanh", "smooth_sine"},
    }
    return {name: _aggregate(rows, predicate) for name, predicate in groups.items()}


def _factorial_effects(mask_aggregates: dict[str, Any]) -> dict[str, Any]:
    effects: dict[str, Any] = {}
    subset_names = {
        (0,): "S",
        (1,): "P",
        (2,): "D",
        (0, 1): "SxP",
        (0, 2): "SxD",
        (1, 2): "PxD",
        (0, 1, 2): "SxPxD",
    }
    for group in next(iter(mask_aggregates.values())).keys():
        effects[group] = {}
        for metric in SCALAR_METRICS:
            values = {
                mask: mask_aggregates[mask][group]["mean"][metric]
                for mask in MASKS
            }
            if any(value is None for value in values.values()):
                effects[group][metric] = {
                    name: None for name in subset_names.values()
                }
                continue
            metric_effects: dict[str, float] = {}
            for subset, name in subset_names.items():
                contrast = 0.0
                for mask, value in values.items():
                    sign = math.prod(1.0 if mask[index] == "1" else -1.0 for index in subset)
                    contrast += sign * float(value)
                metric_effects[name] = 2.0 * contrast / len(MASKS)
            effects[group][metric] = metric_effects
    return effects


def _frozen_deletion_gates(mask_aggregates: dict[str, Any]) -> dict[str, Any]:
    full = mask_aggregates["111"]
    candidates = {"remove_S": "011", "remove_P": "101", "remove_D": "110"}
    result: dict[str, Any] = {}
    primary_group = "held_discontinuous_phase_0p875"
    if full[primary_group]["case_count"] == 0:
        primary_group = "held_discontinuous"
    primary_metric = "relative_increment_l2"
    control_groups = ("train_discontinuous", "held_smooth")
    for name, mask in candidates.items():
        baseline = float(full[primary_group]["mean"][primary_metric])
        candidate = float(mask_aggregates[mask][primary_group]["mean"][primary_metric])
        reduction = (baseline - candidate) / max(baseline, np.finfo(float).tiny)
        control_ratios = {
            group: float(mask_aggregates[mask][group]["mean"][primary_metric])
            / max(
                float(full[group]["mean"][primary_metric]),
                np.finfo(float).tiny,
            )
            for group in control_groups
        }
        result[name] = {
            "mask": mask,
            "primary_metric": primary_metric,
            "primary_group": primary_group,
            "relative_reduction": float(reduction),
            "control_relative_l2_ratios": control_ratios,
            "frozen_screen_pass": bool(
                reduction >= 0.20 and all(value <= 1.05 for value in control_ratios.values())
            ),
            "interpretation": "acute_checkpoint_dependence_not_trained_ablation",
        }
    return result


def _restrict_channels(
    values: np.ndarray,
    *,
    fine_resolution: Sequence[int],
    coarse_resolution: Sequence[int],
) -> np.ndarray:
    fine_nx, fine_ny = (int(value) for value in fine_resolution)
    channels = values.shape[0]
    structured = values.reshape(channels, fine_ny, fine_nx).transpose(1, 2, 0)
    restricted = restrict_nested_cell_averages(
        structured,
        fine_resolution=fine_resolution,
        coarse_resolution=coarse_resolution,
    )
    return restricted.transpose(2, 0, 1).reshape(channels, -1)


def _normalized_commutator(coarse: np.ndarray, restricted_fine: np.ndarray) -> float:
    numerator = float(np.sqrt(np.mean(np.square(coarse - restricted_fine))))
    denominator = float(np.sqrt(np.mean(np.square(coarse))))
    return numerator / max(denominator, np.finfo(float).tiny)


def _prediction_commutators(
    predictions: dict[str, np.ndarray],
    records_by_resolution: dict[str, list[RegisteredCase]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for coarse_resolution, fine_resolution in zip(
        REGISTERED_RESOLUTIONS[:-1], REGISTERED_RESOLUTIONS[1:], strict=True
    ):
        coarse_key = f"{coarse_resolution[0]}x{coarse_resolution[1]}"
        fine_key = f"{fine_resolution[0]}x{fine_resolution[1]}"
        coarse_records = records_by_resolution[coarse_key]
        fine_records = records_by_resolution[fine_key]
        fine_index = {record.case_id: index for index, record in enumerate(fine_records)}
        for mask in MASKS:
            coarse_values = predictions[f"mask_{mask}_{coarse_key}"]
            fine_values = predictions[f"mask_{mask}_{fine_key}"]
            for coarse_index, record in enumerate(coarse_records):
                if record.case_id not in fine_index:
                    continue
                selected = fine_values[fine_index[record.case_id]]
                restricted = restrict_nested_cell_averages(
                    selected,
                    fine_resolution=fine_resolution,
                    coarse_resolution=coarse_resolution,
                )
                target_fine = fine_records[fine_index[record.case_id]].case.increment
                restricted_target = restrict_nested_cell_averages(
                    target_fine,
                    fine_resolution=fine_resolution,
                    coarse_resolution=coarse_resolution,
                )
                target_coarse = record.case.increment
                denominator = float(np.sqrt(np.mean(np.square(target_coarse))))
                rows.append(
                    {
                        "mask": mask,
                        "case_id": record.case_id,
                        "family": record.family,
                        "phase": record.phase,
                        "coarse_resolution": list(coarse_resolution),
                        "fine_resolution": list(fine_resolution),
                        "prediction_commutator": float(
                            np.sqrt(np.mean(np.square(coarse_values[coarse_index] - restricted)))
                            / max(denominator, np.finfo(float).tiny)
                        ),
                        "analytic_restriction_floor": float(
                            np.sqrt(np.mean(np.square(target_coarse - restricted_target)))
                            / max(denominator, np.finfo(float).tiny)
                        ),
                    }
                )
    return rows


def _current_fronts(record: RegisteredCase) -> tuple[FrontSpec, ...]:
    position = record.position
    if record.family == "step":
        return (FrontSpec(position, 1.0, 0.0),)
    if record.family == "pulse":
        assert record.case.width is not None
        return (
            FrontSpec(position, 0.0, 1.0),
            FrontSpec(position + record.case.width, 1.0, 0.0),
        )
    return ()


def _scalar_gradient_rows(
    config: dict[str, Any], resolutions: Sequence[tuple[int, int]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for resolution in resolutions:
        grid = build_structured_cell_grid(resolution)
        aux = None
        for contract, records in (
            ("common_physical", build_common_physical_cases(grid, config, split="held")),
            ("grid_local_phase", build_grid_local_phase_cases(grid, config)),
        ):
            for record in records:
                if record.family not in {"step", "pulse"}:
                    continue
                if aux is None:
                    from utility.time_dependent_no.pcno_shock_representation import (
                        pcno_aux_tensors,
                    )

                    aux = pcno_aux_tensors(grid, dtype=torch.float64)
                field = torch.as_tensor(
                    record.case.current.reshape(1, 1, -1), dtype=torch.float64
                )
                raw = compute_gradient(field, aux[3], aux[4])
                aggregate = graph_neighbor_average(raw, aux[3], iterations=2)
                fronts = _current_fronts(record)
                for stage, tensor in (
                    ("raw_gradient", raw),
                    ("native_two_hop", aggregate),
                ):
                    rows.append(
                        {
                            "contract": contract,
                            "case_id": record.case_id,
                            "family": record.family,
                            "anchor_index": record.anchor_index,
                            "phase": record.phase,
                            "resolution": list(resolution),
                            "stage": stage,
                            "statistics": gradient_distribution_statistics(
                                tensor.squeeze(0).numpy(), grid, fronts
                            ),
                        }
                    )
    return rows


def _tensor_summary(
    values: np.ndarray,
    grid: StructuredCellGrid,
    feature_positions: Sequence[float],
    *,
    directional: bool,
) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != grid.nx * grid.ny:
        raise ValueError("stage tensor must have shape [C,N]")
    absolute = np.abs(array)
    summary: dict[str, Any] = {
        "channel_count": int(array.shape[0]),
        "maximum_abs": float(np.max(absolute)),
        "q99_abs": float(np.quantile(absolute, 0.99)),
        "weighted_rms": float(np.sqrt(np.mean(np.square(array)))),
        "physical_l1_per_channel": float(
            np.sum(absolute) * grid.hx * grid.hy / array.shape[0]
        ),
    }
    if feature_positions:
        mask = physical_front_band_mask(grid, feature_positions).astype(bool)
        energy = np.square(array)
        summary["front_band_energy_fraction"] = float(
            np.sum(energy[:, mask]) / max(np.sum(energy), np.finfo(float).tiny)
        )
        distance = np.min(
            np.abs(
                grid.nodes[:, :1]
                - np.asarray(feature_positions, dtype=np.float64).reshape(1, -1)
            ),
            axis=1,
        )
        mass = np.sum(absolute, axis=0) * grid.hx * grid.hy
        order = np.argsort(distance, kind="stable")
        cumulative = np.cumsum(mass[order])
        total = float(cumulative[-1])
        for fraction in (0.5, 0.9):
            index = int(np.searchsorted(cumulative, fraction * total, side="left"))
            radius = float(distance[order[min(index, len(order) - 1)]])
            token = int(100 * fraction)
            summary[f"support_{token}_half_width"] = radius
            summary[f"support_{token}_width_in_cells"] = 2.0 * radius / grid.hx
    else:
        summary["front_band_energy_fraction"] = None
        summary["support_50_half_width"] = None
        summary["support_50_width_in_cells"] = None
        summary["support_90_half_width"] = None
        summary["support_90_width_in_cells"] = None
    if directional:
        normal = array[0::2]
        transverse = array[1::2]
        signed = np.sum(normal, axis=1) * grid.hx * grid.hy
        summary["normal_signed_integral_median_abs"] = float(
            np.median(np.abs(signed))
        )
        summary["normal_signed_integral_q90_abs"] = float(
            np.quantile(np.abs(signed), 0.9)
        )
        summary["transverse_to_normal_l1_ratio"] = float(
            np.sum(np.abs(transverse))
            / max(np.sum(np.abs(normal)), np.finfo(float).tiny)
        )
    return summary


def _trace_arrays(trace: Any) -> dict[tuple[int, str], np.ndarray]:
    arrays: dict[tuple[int, str], np.ndarray] = {}
    for layer in trace.layers:
        for stage, tensor in (
            ("raw_gradient", layer.raw_gradient),
            ("post_aggregation", layer.post_aggregation),
            ("pre_softsign", layer.pre_softsign),
            ("post_softsign", layer.post_softsign),
            ("branch_output", layer.branch_output),
            (
                "decoded_response",
                layer.decoded_differential_ablation_response.permute(0, 2, 1),
            ),
        ):
            arrays[(layer.layer, stage)] = tensor.squeeze(0).detach().cpu().numpy()
    return arrays


def _stage_records(config: dict[str, Any], grid: StructuredCellGrid) -> list[RegisteredCase]:
    held = build_common_physical_cases(grid, config, split="held")
    return [
        record
        for record in held
        if record.family in {"step", "pulse"}
        or (
            record.family in {"smooth_tanh", "smooth_sine"}
            and record.anchor_index == min(1, len(config["anchors"]) - 1)
        )
    ]


def _differential_stage_rows(
    model: torch.nn.Module,
    config: dict[str, Any],
    resolutions: Sequence[tuple[int, int]],
    device: torch.device,
    *,
    smoke: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float]:
    grids = {resolution: build_structured_cell_grid(resolution) for resolution in resolutions}
    template_grid = grids[resolutions[0]]
    templates = _stage_records(config, template_grid)
    if smoke:
        templates = [record for record in templates if record.family == "step"][:1]
    rows: list[dict[str, Any]] = []
    commutators: list[dict[str, Any]] = []
    maximum_closure = 0.0
    for template in templates:
        previous: tuple[tuple[int, int], dict[tuple[int, str], np.ndarray]] | None = None
        for resolution in resolutions:
            grid = grids[resolution]
            record = RegisteredCase(
                case_id=template.case_id,
                split=template.split,
                family=template.family,
                anchor_index=template.anchor_index,
                phase=template.phase,
                position=template.position,
                case=make_translated_front_case(
                    grid, template.family, position=template.position
                ),
            )
            model_input, _, aux = build_batch([record], grid, device)
            mask = None
            if record.case.feature_positions:
                mask = torch.as_tensor(
                    physical_front_band_mask(grid, record.case.feature_positions),
                    dtype=torch.bool,
                    device=device,
                ).unsqueeze(0)
            trace = trace_pcno_differential_stages(
                model, model_input, aux, front_band_mask=mask
            )
            maximum_closure = max(
                maximum_closure,
                float(torch.max(torch.abs(trace.model_output - trace.replay_output))),
            )
            arrays = _trace_arrays(trace)
            for layer_trace in trace.layers:
                for stage in (
                    "raw_gradient",
                    "post_aggregation",
                    "pre_softsign",
                    "post_softsign",
                    "branch_output",
                    "decoded_response",
                ):
                    directional = stage in {
                        "raw_gradient",
                        "post_aggregation",
                        "pre_softsign",
                        "post_softsign",
                    }
                    row = {
                        "case_id": record.case_id,
                        "family": record.family,
                        "anchor_index": record.anchor_index,
                        "phase": record.phase,
                        "resolution": list(resolution),
                        "layer": layer_trace.layer,
                        "stage": stage,
                        "statistics": _tensor_summary(
                            arrays[(layer_trace.layer, stage)],
                            grid,
                            record.case.feature_positions,
                            directional=directional,
                        ),
                    }
                    if stage == "pre_softsign":
                        row["saturation"] = layer_trace.saturation
                    rows.append(row)
            if previous is not None:
                coarse_resolution, coarse_arrays = previous
                for key, coarse_values in coarse_arrays.items():
                    restricted = _restrict_channels(
                        arrays[key],
                        fine_resolution=resolution,
                        coarse_resolution=coarse_resolution,
                    )
                    commutators.append(
                        {
                            "case_id": record.case_id,
                            "family": record.family,
                            "phase": record.phase,
                            "layer": key[0],
                            "stage": key[1],
                            "coarse_resolution": list(coarse_resolution),
                            "fine_resolution": list(resolution),
                            "normalized_rms_commutator": _normalized_commutator(
                                coarse_values, restricted
                            ),
                        }
                    )
            previous = (resolution, arrays)
            del trace, model_input, aux
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return rows, commutators, maximum_closure


def _commutator_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    keys = sorted(
        {
            (
                row["family"],
                row["layer"],
                row["stage"],
                tuple(row["coarse_resolution"]),
                tuple(row["fine_resolution"]),
            )
            for row in rows
        }
    )
    for family, layer, stage, coarse, fine in keys:
        selected = [
            float(row["normalized_rms_commutator"])
            for row in rows
            if row["family"] == family
            and row["layer"] == layer
            and row["stage"] == stage
            and tuple(row["coarse_resolution"]) == coarse
            and tuple(row["fine_resolution"]) == fine
        ]
        key = f"{family}_L{layer}_{stage}_{coarse[0]}x{coarse[1]}_to_{fine[0]}x{fine[1]}"
        result[key] = {
            "case_count": len(selected),
            "median": float(np.median(selected)),
            "maximum": float(np.max(selected)),
            "count_at_least_0p20": sum(value >= 0.20 for value in selected),
        }
    return result


def _plot_branch_masks(
    mask_aggregates: dict[str, Any], output_dir: Path
) -> list[Path]:
    metrics = (
        "relative_increment_l2",
        "oscillatory_mass_outside_front_band",
        "normalized_overshoot",
        "positive_total_variation_excess",
    )
    group = "held_discontinuous_phase_0p875"
    if mask_aggregates["111"][group]["case_count"] == 0:
        group = "held_discontinuous"
    values = np.asarray(
        [
            [mask_aggregates[mask][group]["mean"][metric] for mask in MASKS]
            for metric in metrics
        ],
        dtype=np.float64,
    )
    full = np.maximum(values[:, MASKS.index("111") : MASKS.index("111") + 1], 1e-12)
    ratios = np.log10(np.maximum(values / full, 1e-4))
    fig, axis = plt.subplots(figsize=(6.75, 2.8), constrained_layout=True)
    image = axis.imshow(ratios, cmap="coolwarm", vmin=-2.0, vmax=2.0, aspect="auto")
    axis.set_xticks(range(len(MASKS)), MASKS)
    axis.set_yticks(
        range(len(metrics)),
        ("Relative L2", "Oscillatory mass", "Overshoot", "TV excess"),
    )
    axis.set_xlabel("Global branch mask (S, P, D)")
    axis.set_title("Frozen dependence at the failed held phase (0.875)")
    fig.colorbar(image, ax=axis, label="log10(metric / full-PCNO metric)")
    return _save_figure(fig, output_dir / "fig_branch_mask_factorial")


def _plot_profiles(
    prediction_arrays: dict[str, np.ndarray],
    records: list[RegisteredCase],
    grid: StructuredCellGrid,
    output_dir: Path,
) -> list[Path]:
    selected: list[int] = []
    for family in ("step", "pulse"):
        candidates = [
            index
            for index, record in enumerate(records)
            if record.split == "held"
            and record.family == family
        ]
        selected.append(
            min(
                candidates,
                key=lambda index: (
                    abs(records[index].phase - 0.875),
                    abs(records[index].anchor_index - 1),
                ),
            )
        )
    masks = ("111", "011", "101", "110")
    titles = ("Full", "No spectral", "No pointwise", "No differential")
    x = grid.x_centers
    fig, axes = plt.subplots(2, 4, figsize=(9.0, 4.4), sharex=True, sharey=True, constrained_layout=True)
    for row_index, record_index in enumerate(selected):
        record = records[record_index]
        for column, (mask, title) in enumerate(zip(masks, titles, strict=True)):
            axis = axes[row_index, column]
            prediction = prediction_arrays[
                f"mask_{mask}_{grid.nx}x{grid.ny}"
            ][record_index]
            predicted_next = record.case.current + prediction
            axis.plot(x, np.mean(record.case.target, axis=0), "k--", label="Exact next")
            axis.plot(x, np.mean(predicted_next, axis=0), color="#0072B2", label="Prediction")
            error_axis = axis.twinx()
            error_axis.plot(
                x,
                np.mean(prediction - record.case.increment, axis=0),
                color="#D55E00",
                linestyle=":",
                linewidth=1.0,
            )
            error_axis.set_ylim(-0.12, 0.12)
            error_axis.grid(False)
            for front in record.case.target_fronts:
                axis.axvspan(
                    front.position - FRONT_BAND_WIDTH,
                    front.position + FRONT_BAND_WIDTH,
                    color="#F0E442",
                    alpha=0.12,
                    linewidth=0,
                )
            if row_index == 0:
                axis.set_title(title)
            if column == 0:
                axis.set_ylabel(f"{record.family.capitalize()} next state")
            if row_index == 1:
                axis.set_xlabel("x")
            axis.set_xlim(0.15, 0.9)
            axis.set_ylim(-0.12, 1.12)
    axes[0, 0].legend(loc="upper right", fontsize=7, frameon=False)
    fig.suptitle("Worst held subcell phase: acute frozen branch deletion")
    return _save_figure(fig, output_dir / "fig_worst_phase_branch_profiles")


def _plot_differential_scaling(
    rows: list[dict[str, Any]], output_dir: Path
) -> list[Path]:
    stages = (
        "raw_gradient",
        "post_aggregation",
        "pre_softsign",
        "post_softsign",
        "branch_output",
        "decoded_response",
    )
    resolutions = sorted({tuple(row["resolution"]) for row in rows})
    fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.9), constrained_layout=True)
    for stage in stages:
        medians = []
        for resolution in resolutions:
            values = [
                row["statistics"]["q99_abs"]
                for row in rows
                if tuple(row["resolution"]) == resolution
                and row["stage"] == stage
                and row["family"] in {"step", "pulse"}
            ]
            medians.append(float(np.median(values)))
        axes[0].plot([value[0] for value in resolutions], medians, marker="o", label=stage)
    saturation = []
    for resolution in resolutions:
        values = [
            row["saturation"]["front_band_volume_channel_fraction"]
            for row in rows
            if tuple(row["resolution"]) == resolution
            and row["stage"] == "pre_softsign"
            and row["family"] in {"step", "pulse"}
            and "front_band_volume_channel_fraction" in row["saturation"]
        ]
        saturation.append(float(np.median(values)))
    axes[1].plot([value[0] for value in resolutions], saturation, marker="o", color="#D55E00")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("nx")
    axes[0].set_ylabel("Median q99 absolute magnitude")
    axes[0].set_title("Differential stages")
    axes[0].legend(fontsize=6, frameon=False)
    axes[1].set_xlabel("nx")
    axes[1].set_ylabel("Front-band saturated fraction")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].set_title("Softsign saturation")
    return _save_figure(fig, output_dir / "fig_differential_path_scaling")


def _save_figure(fig: plt.Figure, stem: Path) -> list[Path]:
    pdf = stem.with_suffix(".pdf")
    png = stem.with_suffix(".png")
    fig.savefig(pdf, metadata={"Creator": "W26-L2 P1 analyzer", "CreationDate": None})
    fig.savefig(png, dpi=300, metadata={"Software": "W26-L2 P1 analyzer"})
    plt.close(fig)
    return [pdf, png]


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 8.5,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.bbox": "tight",
        }
    )


def run_analysis(
    p0_run_dir: Path,
    output_dir: Path,
    *,
    device_name: str,
    smoke: bool,
    batch_size: int,
) -> dict[str, Any]:
    device = _resolve_device(device_name)
    if not smoke and device.type != "cuda":
        raise RuntimeError("the production P1 contract requires CUDA")
    verified = _verify_p0_bundle(p0_run_dir.resolve(), smoke=smoke)
    config = verified["config"]
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing nonempty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir / "run_state.json",
        {"schema": SCHEMA, "working_run_id": WORKING_RUN_ID, "status": "running"},
    )
    checkpoint = torch.load(
        p0_run_dir / "checkpoint.pt", map_location=device, weights_only=False
    )
    if checkpoint["schema"] != P0_SCHEMA or checkpoint["config_digest"] != verified["contract"]["config_digest"]:
        raise ValueError("P0 checkpoint contract does not close")
    model = build_model(config, device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    initial_state = model_state_sha256(model)
    if initial_state != verified["summary"]["final_model_state_sha256"]:
        raise ValueError("P0 checkpoint model state hash does not match summary")
    source_hashes = _source_hashes(SOURCE_PATHS)
    provenance_hashes = _source_hashes(PROVENANCE_PATHS)
    resolutions = (
        (tuple(int(value) for value in config["resolution"]),)
        if smoke
        else REGISTERED_RESOLUTIONS
    )
    run_contract = {
        "schema": SCHEMA,
        "working_run_id": WORKING_RUN_ID + ("-SMOKE" if smoke else ""),
        "science_result_eligible": not smoke,
        "optimization_performed": False,
        "execution": {
            "requested_device": device_name,
            "resolved_device": str(device),
            "batch_size": int(batch_size),
        },
        "p0_input": {
            "path_name": p0_run_dir.name,
            "manifest_sha256": sha256_file(p0_run_dir / "manifest.json"),
            "checkpoint_sha256": sha256_file(p0_run_dir / "checkpoint.pt"),
            "model_state_sha256": initial_state,
        },
        "population": {
            "native": "all registered train and held cases",
            "nested": "common-physical held cases",
            "differential_trace": (
                "all held discontinuities plus anchor-1 held smooth controls"
                if not smoke
                else "one non-scientific held step"
            ),
            "resolutions": [list(value) for value in resolutions],
        },
        "branch_masks": {"bit_order": list(BRANCHES), "masks": list(MASKS)},
        "local_deletions": "every branch at every layer on the native population",
        "closure_tolerances": {
            "mask_111_native_bypass_max_abs": 0.0,
            "independent_noop_replay_vs_native_max_abs": 1.0e-4,
            "direct_batch_vs_stored_p0_max_abs": 1.0e-4,
            "differential_trace_replay_max_abs": 1.0e-4,
        },
        "source_sha256": source_hashes,
        "provenance_sha256": provenance_hashes,
        "environment": _environment(device),
    }
    write_json(output_dir / "run_contract.json", run_contract)

    started = time.perf_counter()
    prediction_arrays: dict[str, np.ndarray] = {}
    records_by_resolution: dict[str, list[RegisteredCase]] = {}
    mask_rows: list[dict[str, Any]] = []
    mask_aggregates: dict[str, Any] = {}
    for resolution in resolutions:
        grid = build_structured_cell_grid(resolution)
        key = f"{grid.nx}x{grid.ny}"
        if tuple(resolution) == tuple(config["resolution"]):
            records = build_registered_cases(grid, config, split="train")
            records += build_registered_cases(grid, config, split="held")
        else:
            records = build_common_physical_cases(grid, config, split="held")
        records_by_resolution[key] = records
        mask_aggregates[key] = {}
        for mask in MASKS:
            prediction = _predict_records(
                model,
                records,
                grid,
                device,
                batch_size=batch_size,
                mask=mask,
            )
            prediction_arrays[f"mask_{mask}_{key}"] = prediction
            rows = _case_rows(records, prediction, grid, mask=mask)
            mask_rows.extend(rows)
            mask_aggregates[key][mask] = _mask_aggregates(rows)
            print(
                json.dumps(
                    {
                        "stage": "global_mask",
                        "resolution": key,
                        "mask": mask,
                        "case_count": len(records),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    native_grid = verified["grid"]
    native_key = f"{native_grid.nx}x{native_grid.ny}"
    native_records = records_by_resolution[native_key]
    stored = np.load(p0_run_dir / "predictions.npz", allow_pickle=False)
    stored_predictions = stored["predicted_increment"]
    direct_predictions = prediction_arrays[f"mask_111_{native_key}"]
    noop_replay_predictions = _replay_predict_records(
        model,
        native_records,
        native_grid,
        device,
        batch_size=batch_size,
        mask="111",
    )
    noop_replay_floor = float(
        np.max(
            np.abs(noop_replay_predictions - direct_predictions)
        )
    )
    stored_drift = float(
        np.max(
            np.abs(direct_predictions - stored_predictions)
        )
    )
    if noop_replay_floor > 1.0e-4:
        raise ValueError(
            "independent no-op replay exceeds the registered CUDA floor: "
            f"{noop_replay_floor}"
        )
    if stored_drift > 1.0e-4:
        raise ValueError(
            f"direct batched model does not close on stored P0 predictions: {stored_drift}"
        )

    local_rows: list[dict[str, Any]] = []
    for layer, branch in itertools.product(range(len(model.ws)), BRANCHES):
        prediction = _predict_records(
            model,
            native_records,
            native_grid,
            device,
            batch_size=batch_size,
            disabled_branch=(layer, branch),
        )
        local_rows.extend(
            _case_rows(
                native_records,
                prediction,
                native_grid,
                local_deletion=(layer, branch),
            )
        )
        print(
            json.dumps(
                {"stage": "local_deletion", "layer": layer, "branch": branch},
                sort_keys=True,
            ),
            flush=True,
        )

    prediction_commutators = (
        []
        if smoke
        else _prediction_commutators(prediction_arrays, records_by_resolution)
    )
    scalar_gradient_rows = _scalar_gradient_rows(config, resolutions)
    stage_rows, stage_commutators, stage_closure = _differential_stage_rows(
        model, config, resolutions, device, smoke=smoke
    )
    if stage_closure > 1.0e-4:
        raise ValueError(f"differential trace replay does not close: {stage_closure}")
    if model_state_sha256(model) != initial_state:
        raise RuntimeError("frozen analysis mutated the PCNO state")

    native_aggregates = mask_aggregates[native_key]
    factorial = _factorial_effects(native_aggregates)
    deletion_gates = _frozen_deletion_gates(native_aggregates)
    local_aggregates: dict[str, Any] = {}
    for layer, branch in itertools.product(range(len(model.ws)), BRANCHES):
        selected = [
            row
            for row in local_rows
            if row["local_deletion"] == {"layer": layer, "branch": branch}
        ]
        local_aggregates[f"L{layer}_{branch}"] = _mask_aggregates(selected)

    write_json(output_dir / "global_mask_case_metrics.json", mask_rows)
    write_json(output_dir / "local_deletion_case_metrics.json", local_rows)
    write_json(output_dir / "prediction_commutators.json", prediction_commutators)
    write_json(output_dir / "scalar_gradient_scaling.json", scalar_gradient_rows)
    write_json(output_dir / "differential_stage_summaries.json", stage_rows)
    write_json(output_dir / "differential_stage_commutators.json", stage_commutators)
    metadata = {
        key: {
            "case_ids": [record.case_id for record in records],
            "families": [record.family for record in records],
            "splits": [record.split for record in records],
            "phases": [record.phase for record in records],
            "positions": [record.position for record in records],
        }
        for key, records in records_by_resolution.items()
    }
    write_json(output_dir / "prediction_metadata.json", metadata)
    write_npz(output_dir / "branch_mask_predictions.npz", **prediction_arrays)

    _style()
    figure_paths: list[Path] = []
    figure_paths += _plot_branch_masks(native_aggregates, output_dir)
    figure_paths += _plot_profiles(
        prediction_arrays, native_records, native_grid, output_dir
    )
    figure_paths += _plot_differential_scaling(stage_rows, output_dir)
    visual_manifest = {
        "schema": SCHEMA,
        "files": [path.name for path in figure_paths],
        "sha256": {path.name: sha256_file(path) for path in figure_paths},
    }
    write_json(output_dir / "visual_manifest.json", visual_manifest)

    held_decisions = [
        _case_guard(row, grid_spacing=native_grid.hx, held=True)
        for row in mask_rows
        if row["mask"] == "111"
        and row["resolution"] == list(native_grid.resolution)
        and row["split"] == "held"
        and row["family"] in {"step", "pulse"}
    ]
    summary = {
        "schema": SCHEMA,
        "working_run_id": run_contract["working_run_id"],
        "science_result": not smoke,
        "status": "completed",
        "optimization_performed": False,
        "p0_model_state_sha256": initial_state,
        "mask_111_native_bypass_max_abs_error": 0.0,
        "independent_noop_replay_max_abs_error": noop_replay_floor,
        "direct_batch_to_p0_stored_max_abs_error": stored_drift,
        "differential_trace_replay_max_abs_error": stage_closure,
        "global_mask_aggregates": mask_aggregates,
        "native_factorial_effects": factorial,
        "frozen_deletion_gates": deletion_gates,
        "local_deletion_aggregates": local_aggregates,
        "native_full_held_discontinuous_decisions": held_decisions,
        "prediction_commutator_case_count": len(prediction_commutators),
        "differential_commutator_summary": _commutator_summary(stage_commutators),
        "learned_stage_classification": {
            "automatic_mechanism_label_allowed": False,
            "reason": (
                "raw shock-gradient peak growth is expected; stage labels require "
                "joint review of grid-local distributional scaling, common-physical "
                "support, both adjacent-grid pairs, and decoded survival"
            ),
        },
        "visualizations": visual_manifest,
        "elapsed_seconds": time.perf_counter() - started,
        "maximum_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else None
        ),
    }
    write_json(output_dir / "summary.json", summary)
    write_json(
        output_dir / "run_state.json",
        {
            "schema": SCHEMA,
            "working_run_id": run_contract["working_run_id"],
            "status": "completed",
        },
    )
    output_paths = sorted(
        path
        for path in output_dir.iterdir()
        if path.is_file() and path.name != "manifest.json"
    )
    manifest = {
        "schema": SCHEMA,
        "working_run_id": run_contract["working_run_id"],
        "science_result": not smoke,
        "status": "completed",
        "p0_manifest_sha256": run_contract["p0_input"]["manifest_sha256"],
        "source_sha256": source_hashes,
        "output_hashes": {path.name: sha256_file(path) for path in output_paths},
        "output_count": len(output_paths),
    }
    write_json(output_dir / "manifest.json", manifest)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p0-run-dir", type=Path, default=DEFAULT_P0_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_analysis(
        args.p0_run_dir,
        args.output_dir,
        device_name=args.device,
        smoke=args.smoke,
        batch_size=args.batch_size,
    )
    print(json.dumps(summary, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
