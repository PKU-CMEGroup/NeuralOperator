#!/usr/bin/env python3
"""Verify, compare, and visualize the W26-L2-P2 matched training matrix."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.time_dependent_no.analyze_pcno_shock_pathways import _metric_scalars
from scripts.time_dependent_no.fit_pcno_shock_representation import sha256_file, write_json
from scripts.time_dependent_no.train_pcno_gradient_ablation import (
    SCHEMA as RUN_SCHEMA,
    SEEDS,
    VARIANTS,
)
from utility.time_dependent_no.pcno_shock_representation import (
    DISPLACEMENT,
    DOMAIN_LENGTHS,
    FRONT_BAND_WIDTH,
    PULSE_WIDTH,
    SMOOTH_FILTER_PASS_WAVENUMBER,
    SMOOTH_FILTER_STOP_WAVENUMBER,
    build_structured_cell_grid,
    physical_cosine_spectrum,
)

SCHEMA = "w26_l2_p2_matched_gradient_ablation_analysis_v1"
DEFAULT_MATRIX = ROOT / "artifacts" / "time_dependent_no" / "w26_l2_p2"
DEFAULT_OUTPUT = DEFAULT_MATRIX / "analysis"
COLORS = {
    "full": "#0072B2",
    "no_gradient": "#D55E00",
    "local_replacement": "#009E73",
}
LABELS = {
    "full": "Full PCNO",
    "no_gradient": "No gradient",
    "local_replacement": "Matched local",
}
LINESTYLES = {"full": "-", "no_gradient": "--", "local_replacement": ":"}
METRICS = (
    "relative_increment_l2",
    "oscillatory_mass_outside_front_band",
    "normalized_overshoot",
    "normalized_undershoot",
    "positive_total_variation_excess",
    "total_variation_deficit",
    "smooth_region_error",
    "increment_integral_error",
    "maximum_front_position_error",
    "maximum_front_strength_error",
    "maximum_front_thickness_excess",
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {name: payload[name] for name in payload.files}


def _stable_environment(environment: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in environment.items() if key != "pid"}


def _verify_run(run_dir: Path, variant: str, seed: int, *, smoke: bool) -> dict[str, Any]:
    required = (
        "manifest.json",
        "run_contract.json",
        "summary.json",
        "case_metrics.json",
        "nested_case_metrics.json",
        "predictions.npz",
        "history.json",
    )
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{run_dir.name} is missing {missing}")
    manifest = _read_json(run_dir / "manifest.json")
    contract = _read_json(run_dir / "run_contract.json")
    summary = _read_json(run_dir / "summary.json")
    if any(value.get("schema") != RUN_SCHEMA for value in (manifest, contract, summary)):
        raise ValueError(f"{run_dir.name} has the wrong schema")
    if any(value.get("status") != "completed" for value in (manifest, summary)):
        raise ValueError(f"{run_dir.name} is not complete")
    if bool(summary["science_result"]) == smoke:
        raise ValueError(f"{run_dir.name} scientific/smoke status differs")
    if summary["variant"] != variant or int(summary["seed"]) != seed:
        raise ValueError(f"{run_dir.name} arm or seed differs")
    if manifest["config_digest"] != contract["config_digest"]:
        raise ValueError(f"{run_dir.name} config digest does not close")
    if manifest["population_sha256"] != contract["population"]["sha256"]:
        raise ValueError(f"{run_dir.name} population digest does not close")
    if manifest["source_sha256"] != contract["source_sha256"]:
        raise ValueError(f"{run_dir.name} source hashes do not close")
    if manifest["output_count"] != len(manifest["output_hashes"]):
        raise ValueError(f"{run_dir.name} output count does not close")
    if summary["shared_initialization_sha256"] != contract[
        "shared_initialization_sha256"
    ]:
        raise ValueError(f"{run_dir.name} initialization digest does not close")
    if summary["parameter_summary"] != contract["parameter_summary"]:
        raise ValueError(f"{run_dir.name} parameter summary does not close")
    if summary["primary_metric"] != contract["config"]["primary_metric"]:
        raise ValueError(f"{run_dir.name} primary metric does not close")
    for relative, expected in manifest["output_hashes"].items():
        path = run_dir / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"{run_dir.name} output hash mismatch: {relative}")
    return {
        "run_dir": run_dir,
        "manifest": manifest,
        "contract": contract,
        "summary": summary,
        "native_rows": _read_json(run_dir / "case_metrics.json"),
        "nested_rows": _read_json(run_dir / "nested_case_metrics.json"),
        "history": _read_json(run_dir / "history.json"),
        "predictions": _load_npz(run_dir / "predictions.npz"),
    }


def _load_matrix(matrix_dir: Path, *, smoke: bool) -> dict[tuple[str, int], dict[str, Any]]:
    seeds = (SEEDS[0],) if smoke else SEEDS
    runs = {
        (variant, seed): _verify_run(
            matrix_dir / f"{variant}_s{seed}", variant, seed, smoke=smoke
        )
        for seed in seeds
        for variant in VARIANTS
    }
    source_sets = {
        json.dumps(run["manifest"]["source_sha256"], sort_keys=True)
        for run in runs.values()
    }
    if len(source_sets) != 1:
        raise ValueError("matrix runs do not share one executable source snapshot")
    provenance_sets = {
        json.dumps(run["contract"]["provenance_sha256"], sort_keys=True)
        for run in runs.values()
    }
    if len(provenance_sets) != 1:
        raise ValueError("matrix runs do not share one preregistration snapshot")
    population_sets = {
        run["contract"]["population"]["sha256"] for run in runs.values()
    }
    if len(population_sets) != 1:
        raise ValueError("matrix runs do not share one ordered population")
    environment_sets = {
        json.dumps(_stable_environment(run["contract"]["environment"]), sort_keys=True)
        for run in runs.values()
    }
    if len(environment_sets) != 1:
        raise ValueError("matrix runs do not share one execution environment")
    normalized_configs = set()
    for run in runs.values():
        config = dict(run["contract"]["config"])
        for key in ("seed", "variant", "working_run_id"):
            config.pop(key)
        normalized_configs.add(json.dumps(config, sort_keys=True))
    if len(normalized_configs) != 1:
        raise ValueError("matrix runs do not share one training/evaluation budget")
    for seed in seeds:
        shared = {
            runs[(variant, seed)]["summary"]["shared_initialization_sha256"]
            for variant in VARIANTS
        }
        if len(shared) != 1:
            raise ValueError(f"seed {seed} does not share common initialization")
        full = runs[("full", seed)]["summary"]["parameter_summary"]["trainable"]
        local = runs[("local_replacement", seed)]["summary"]["parameter_summary"][
            "trainable"
        ]
        no_gradient = runs[("no_gradient", seed)]["summary"]["parameter_summary"][
            "gradient_or_replacement_trainable"
        ]
        full_branch = runs[("full", seed)]["summary"]["parameter_summary"][
            "gradient_or_replacement_trainable"
        ]
        local_branch = runs[("local_replacement", seed)]["summary"][
            "parameter_summary"
        ]["gradient_or_replacement_trainable"]
        no_gradient_trainable = runs[("no_gradient", seed)]["summary"][
            "parameter_summary"
        ]["trainable"]
        if (
            local != full
            or local_branch != full_branch
            or no_gradient != 0
            or no_gradient_trainable != full - full_branch
        ):
            raise ValueError(f"seed {seed} parameter matching does not close")
    return runs


def _row_map(rows: list[dict[str, Any]]) -> dict[tuple[Any, ...], dict[str, Any]]:
    result = {}
    for row in rows:
        key = (
            tuple(row.get("resolution", (64, 32))),
            row["case_id"],
        )
        if key in result:
            raise ValueError(f"duplicate metric row {key}")
        result[key] = row
    return result


def _thresholds(row: dict[str, Any]) -> dict[str, float]:
    held = row["split"] == "held"
    resolution = row.get("resolution", [64, 32])
    h = 1.0 / int(resolution[0])
    return {
        "relative_increment_l2": 5.0e-3 if held else 1.0e-3,
        "normalized_overshoot": 0.02 if held else 0.01,
        "normalized_undershoot": 0.02 if held else 0.01,
        "oscillatory_mass_outside_front_band": 1.0e-3,
        "smooth_region_error": 5.0e-3 if held else 1.0e-3,
        "positive_total_variation_excess": 0.05 if held else 0.02,
        "total_variation_deficit": 0.05 if held else 0.02,
        "increment_integral_error": 0.01 if held else 0.005,
        "next_state_integral_error": 0.01 if held else 0.005,
        "maximum_front_position_error": h / (2.0 if held else 4.0),
        "maximum_front_strength_error": 0.05 if held else 0.02,
        "maximum_front_thickness_excess": h if held else h / 2.0,
    }


def _paired_control_failures(
    full_run: dict[str, Any], candidate_run: dict[str, Any]
) -> list[dict[str, Any]]:
    full_rows = _row_map(full_run["native_rows"] + full_run["nested_rows"])
    candidate_rows = _row_map(
        candidate_run["native_rows"] + candidate_run["nested_rows"]
    )
    if full_rows.keys() != candidate_rows.keys():
        raise ValueError("paired arms do not share metric rows")
    failures: list[dict[str, Any]] = []
    for key in full_rows:
        full_row = full_rows[key]
        candidate_row = candidate_rows[key]
        full = _metric_scalars(full_row["metrics"])
        candidate = _metric_scalars(candidate_row["metrics"])
        thresholds = _thresholds(full_row)
        primary = (
            tuple(full_row.get("resolution", (64, 32))) == (64, 32)
            and full_row["split"] == "held"
            and full_row["family"] in {"step", "pulse"}
            and math.isclose(float(full_row["phase"]), 0.875)
        )
        for metric, threshold in thresholds.items():
            if primary and metric == "relative_increment_l2":
                continue
            limit = max(threshold, 1.05 * full[metric])
            if candidate[metric] > limit:
                failures.append(
                    {
                        "resolution": list(key[0]),
                        "case_id": key[1],
                        "metric": metric,
                        "full": full[metric],
                        "candidate": candidate[metric],
                        "limit": limit,
                    }
                )
        full_valid = all(front["front_gate_valid"] for front in full_row["metrics"]["fronts"])
        candidate_valid = all(
            front["front_gate_valid"] for front in candidate_row["metrics"]["fronts"]
        )
        if full_valid and not candidate_valid:
            failures.append(
                {
                    "resolution": list(key[0]),
                    "case_id": key[1],
                    "metric": "front_gate_valid",
                    "full": True,
                    "candidate": False,
                    "limit": True,
                }
            )
    return failures


def _select_rows(
    run: dict[str, Any], predicate: Callable[[dict[str, Any]], bool]
) -> list[dict[str, Any]]:
    return [row for row in run["native_rows"] if predicate(row)]


def _primary_rows(run: dict[str, Any]) -> list[dict[str, Any]]:
    rows = _select_rows(
        run,
        lambda row: row["split"] == "held"
        and row["family"] in {"step", "pulse"}
        and math.isclose(float(row["phase"]), 0.875),
    )
    if rows:
        return rows
    return _select_rows(
        run,
        lambda row: row["split"] == "held"
        and row["family"] in {"step", "pulse"},
    )


def _aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        raise ValueError("cannot aggregate an empty metric population")
    scalars = [_metric_scalars(row["metrics"]) for row in rows]
    return {
        metric: float(np.mean([value[metric] for value in scalars]))
        for metric in METRICS
    }


def _comparison_rows(
    runs: dict[tuple[str, int], dict[str, Any]], seeds: Sequence[int]
) -> list[dict[str, Any]]:
    rows = []
    for seed in seeds:
        for variant in VARIANTS:
            run = runs[(variant, seed)]
            primary_rows = _primary_rows(run)
            train_rows = _select_rows(
                run,
                lambda row: row["split"] == "train"
                and row["family"] in {"step", "pulse"},
            )
            smooth_rows = _select_rows(
                run,
                lambda row: row["split"] == "held"
                and row["family"] in {"smooth_tanh", "smooth_sine"},
            )
            rows.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "primary_value": float(run["summary"]["primary_value"]),
                    "primary_population": _aggregate_metrics(primary_rows),
                    "primary_by_family": {
                        family: _aggregate_metrics(
                            [row for row in primary_rows if row["family"] == family]
                        )
                        for family in ("step", "pulse")
                    },
                    "primary_population_phases": sorted(
                        {float(row["phase"]) for row in primary_rows}
                    ),
                    "train_discontinuous": _aggregate_metrics(train_rows),
                    "held_smooth": _aggregate_metrics(smooth_rows),
                    "fixed_grid_capacity_pass": bool(
                        run["summary"]["capacity"]["fixed_grid_capacity_pass"]
                    ),
                    "nested_scores": run["summary"]["nested_scores"],
                    "elapsed_training_seconds": run["summary"][
                        "elapsed_training_seconds_this_invocation"
                    ],
                    "maximum_cuda_memory_bytes": run["summary"][
                        "maximum_cuda_memory_bytes"
                    ],
                }
            )
    return rows


def _pairing_closures(
    runs: dict[tuple[str, int], dict[str, Any]], seeds: Sequence[int]
) -> dict[str, Any]:
    first = runs[("full", seeds[0])]
    return {
        "all_passed": True,
        "executable_source_sha256": first["contract"]["source_sha256"],
        "preregistration_sha256": first["contract"]["provenance_sha256"],
        "ordered_population_sha256": first["contract"]["population"]["sha256"],
        "execution_environment": _stable_environment(first["contract"]["environment"]),
        "process_ids": {
            f"{variant}_s{seed}": runs[(variant, seed)]["contract"]["environment"][
                "pid"
            ]
            for seed in seeds
            for variant in VARIANTS
        },
        "shared_nonbranch_initialization_sha256": {
            str(seed): runs[("full", seed)]["summary"][
                "shared_initialization_sha256"
            ]
            for seed in seeds
        },
        "parameter_summaries": {
            f"{variant}_s{seed}": runs[(variant, seed)]["summary"][
                "parameter_summary"
            ]
            for seed in seeds
            for variant in VARIANTS
        },
        "verified_input_output_hashes": True,
    }


def _paired_decisions(
    runs: dict[tuple[str, int], dict[str, Any]], seeds: Sequence[int]
) -> dict[str, Any]:
    result = {}
    for candidate in ("no_gradient", "local_replacement"):
        improvements = []
        control_failures = {}
        for seed in seeds:
            full = float(runs[("full", seed)]["summary"]["primary_value"])
            value = float(runs[(candidate, seed)]["summary"]["primary_value"])
            improvements.append((full - value) / max(full, np.finfo(float).tiny))
            failures = _paired_control_failures(
                runs[("full", seed)], runs[(candidate, seed)]
            )
            control_failures[str(seed)] = failures
        win_count = sum(value > 0.0 for value in improvements)
        median = float(statistics.median(improvements))
        result[candidate] = {
            "paired_relative_improvements": {
                str(seed): float(value)
                for seed, value in zip(seeds, improvements, strict=True)
            },
            "paired_win_count": win_count,
            "seed_count": len(seeds),
            "median_relative_improvement": median,
            "primary_gate_pass": win_count >= 2 and median >= 0.05,
            "control_failure_count": sum(len(value) for value in control_failures.values()),
            "control_failures": control_failures,
            "causal_win": bool(
                win_count >= 2
                and median >= 0.05
                and all(not value for value in control_failures.values())
            ),
        }
    return result


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "savefig.bbox": "tight",
        }
    )


def _save(fig: plt.Figure, output_dir: Path, stem: str) -> list[Path]:
    pdf = output_dir / f"{stem}.pdf"
    png = output_dir / f"{stem}.png"
    fig.savefig(pdf, metadata={"Creator": "W26-L2 P2 analyzer", "CreationDate": None})
    fig.savefig(png, dpi=300, metadata={"Software": "W26-L2 P2 analyzer"})
    plt.close(fig)
    return [pdf, png]


def _paired_primary_figure(rows: list[dict[str, Any]], seeds: Sequence[int]) -> plt.Figure:
    fig, axis = plt.subplots(figsize=(4.8, 3.0), constrained_layout=True)
    x = np.arange(len(VARIANTS))
    for seed in seeds:
        values = [
            next(
                row["primary_value"]
                for row in rows
                if row["variant"] == variant and row["seed"] == seed
            )
            for variant in VARIANTS
        ]
        axis.plot(x, values, color="#8C8C8C", alpha=0.55, linewidth=1.0, zorder=1)
    for variant_index, variant in enumerate(VARIANTS):
        values = [
            next(
                row["primary_value"]
                for row in rows
                if row["variant"] == variant and row["seed"] == seed
            )
            for seed in seeds
        ]
        axis.scatter(
            np.full(len(seeds), variant_index),
            values,
            color=COLORS[variant],
            edgecolor="white",
            linewidth=0.5,
            s=28,
            label=LABELS[variant],
            zorder=3,
        )
        axis.scatter(
            [variant_index],
            [statistics.median(values)],
            marker="D",
            color="#222222",
            s=18,
            zorder=4,
        )
    axis.set_yscale("log")
    axis.set_xticks(x, ("Full", "No gradient", "Matched local"))
    axis.set_xlabel("Training arm; gray lines pair common seeds")
    axis.set_ylabel("Phase-0.875 relative increment L2")
    axis.set_title("Matched gradient ablation: locked primary")
    axis.legend(frameon=False)
    return fig


def _metric_figure(rows: list[dict[str, Any]], seeds: Sequence[int]) -> plt.Figure:
    specifications = (
        (
            "Relative L2",
            lambda row: row["primary_population"]["relative_increment_l2"],
        ),
        (
            "Step osc. mass",
            lambda row: row["primary_by_family"]["step"][
                "oscillatory_mass_outside_front_band"
            ],
        ),
        (
            "Pulse overshoot",
            lambda row: row["primary_by_family"]["pulse"]["normalized_overshoot"],
        ),
        (
            "Pulse undershoot",
            lambda row: row["primary_by_family"]["pulse"]["normalized_undershoot"],
        ),
        (
            "TV excess",
            lambda row: row["primary_population"][
                "positive_total_variation_excess"
            ],
        ),
        (
            "TV deficit",
            lambda row: row["primary_population"]["total_variation_deficit"],
        ),
        (
            "Smooth-region error",
            lambda row: row["primary_population"]["smooth_region_error"],
        ),
        (
            "Smooth-control L2",
            lambda row: row["held_smooth"]["relative_increment_l2"],
        ),
    )
    values = np.zeros((len(specifications), len(VARIANTS)), dtype=np.float64)
    errors = np.zeros_like(values)
    for metric_index, (_, accessor) in enumerate(specifications):
        for variant_index, variant in enumerate(VARIANTS):
            selected = [
                accessor(row)
                for row in rows
                if row["variant"] == variant
            ]
            values[metric_index, variant_index] = np.mean(selected)
            errors[metric_index, variant_index] = (
                np.std(selected, ddof=1) if len(selected) > 1 else 0.0
            )
    fig, axes = plt.subplots(2, 4, figsize=(8.8, 5.0), constrained_layout=True)
    for index, (axis, (title, _)) in enumerate(
        zip(axes.ravel(), specifications, strict=True)
    ):
        for variant_index, variant in enumerate(VARIANTS):
            axis.bar(
                variant_index,
                values[index, variant_index],
                yerr=errors[index, variant_index],
                color=COLORS[variant],
                width=0.7,
                capsize=2,
            )
        axis.set_xticks(range(len(VARIANTS)), ("Full", "No grad", "Local"), rotation=35, ha="right")
        axis.set_ylim(bottom=0.0)
        axis.set_title(title)
    fig.suptitle("Registered native primary population (mean and seed standard deviation)")
    return fig


def _primary_indices(payload: dict[str, np.ndarray]) -> np.ndarray:
    indices = np.flatnonzero(
        (payload["splits"] == "held")
        & np.isin(payload["families"], ("step", "pulse"))
        & np.isclose(payload["phases"], 0.875)
    )
    if indices.size:
        return indices
    return np.flatnonzero(
        (payload["splits"] == "held")
        & np.isin(payload["families"], ("step", "pulse"))
    )


def _representative_case_indices(payload: dict[str, np.ndarray]) -> list[int]:
    selected: list[int] = []
    for family in ("step", "pulse"):
        indices = np.flatnonzero(
            (payload["splits"] == "held")
            & (payload["families"] == family)
            & np.isclose(payload["phases"], 0.875)
        )
        if not indices.size:
            indices = np.flatnonzero(
                (payload["splits"] == "held") & (payload["families"] == family)
            )
        if not indices.size:
            raise ValueError(f"prediction payload has no held {family} case")
        selected.append(
            int(indices[np.argmin(np.abs(payload["anchor_indices"][indices] - 1))])
        )
    return selected


def _target_fronts(payload: dict[str, np.ndarray], case_index: int) -> tuple[float, ...]:
    position = float(payload["positions"][case_index])
    family = str(payload["families"][case_index])
    if family == "step":
        return (position + DISPLACEMENT,)
    if family == "pulse":
        return (
            position + DISPLACEMENT,
            position + DISPLACEMENT + PULSE_WIDTH,
        )
    raise ValueError(f"front visualization does not support {family}")


def _shade_front_bands(
    axis: plt.Axes, payload: dict[str, np.ndarray], case_index: int
) -> None:
    for front in _target_fronts(payload, case_index):
        axis.axvspan(
            front - FRONT_BAND_WIDTH,
            front + FRONT_BAND_WIDTH,
            color="#CC79A7",
            alpha=0.10,
            linewidth=0.0,
        )


def _profiles_figure(
    runs: dict[tuple[str, int], dict[str, Any]], seed: int
) -> plt.Figure:
    reference = runs[("full", seed)]["predictions"]
    selected = _representative_case_indices(reference)
    errors = []
    states = [reference["target_next"][index] for index in selected]
    for variant in VARIANTS:
        payload = runs[(variant, seed)]["predictions"]
        states.extend(
            payload["current"][index] + payload["predicted_increment"][index]
            for index in selected
        )
        errors.extend(
            np.mean(
                payload["predicted_increment"][selected]
                - payload["target_increment"][selected],
                axis=1,
            )
        )
    error_limit = max(float(np.max(np.abs(value))) for value in errors)
    state_min = min(float(np.min(value)) for value in states)
    state_max = max(float(np.max(value)) for value in states)
    state_margin = 0.05 * max(state_max - state_min, 1.0)
    nx = reference["current"].shape[-1]
    x = (np.arange(nx) + 0.5) / nx
    fig, axes = plt.subplots(2, 3, figsize=(8.2, 4.5), sharex=True, sharey=True, constrained_layout=True)
    for row_index, case_index in enumerate(selected):
        for column, variant in enumerate(VARIANTS):
            axis = axes[row_index, column]
            payload = runs[(variant, seed)]["predictions"]
            target_next = payload["target_next"][case_index]
            predicted_next = (
                payload["current"][case_index]
                + payload["predicted_increment"][case_index]
            )
            error = np.mean(
                payload["predicted_increment"][case_index]
                - payload["target_increment"][case_index],
                axis=0,
            )
            axis.plot(x, np.mean(target_next, axis=0), "k--", label="Exact next")
            axis.plot(x, np.mean(predicted_next, axis=0), color=COLORS[variant], label=LABELS[variant])
            _shade_front_bands(axis, payload, case_index)
            secondary = axis.twinx()
            secondary.plot(x, error, color="#777777", linestyle=":", linewidth=1.0, label="Increment error")
            secondary.set_ylim(-1.05 * error_limit, 1.05 * error_limit)
            secondary.grid(False)
            if column != 2:
                secondary.set_yticklabels([])
            else:
                secondary.set_ylabel("Increment error")
            if row_index == 0:
                axis.set_title(LABELS[variant])
            if column == 0:
                axis.set_ylabel(("Step" if row_index == 0 else "Pulse") + " next state")
            if row_index == 1:
                axis.set_xlabel("x")
            axis.set_xlim(0.15, 0.9)
            axis.set_ylim(state_min - state_margin, state_max + state_margin)
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.suptitle(f"Representative held-discontinuity profiles, paired seed {seed}")
    return fig


def _phase_maps_figure(
    runs: dict[tuple[str, int], dict[str, Any]], seed: int
) -> plt.Figure:
    reference = runs[("full", seed)]["predictions"]
    selected = _representative_case_indices(reference)
    state_fields = [reference["target_next"][index] for index in selected]
    error_fields = []
    for variant in VARIANTS:
        payload = runs[(variant, seed)]["predictions"]
        state_fields.extend(
            payload["current"][index] + payload["predicted_increment"][index]
            for index in selected
        )
        error_fields.extend(
            payload["predicted_increment"][index]
            - payload["target_increment"][index]
            for index in selected
        )
    state_min = min(float(np.min(field)) for field in state_fields)
    state_max = max(float(np.max(field)) for field in state_fields)
    error_limit = max(float(np.max(np.abs(field))) for field in error_fields)
    error_limit = max(error_limit, np.finfo(np.float64).tiny)
    extent = (0.0, DOMAIN_LENGTHS[0], 0.0, DOMAIN_LENGTHS[1])
    fig, axes = plt.subplots(
        2,
        7,
        figsize=(13.0, 3.8),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    prediction_columns = {"full": 1, "no_gradient": 3, "local_replacement": 5}
    state_image = None
    error_image = None
    for row_index, case_index in enumerate(selected):
        target_axis = axes[row_index, 0]
        state_image = target_axis.imshow(
            reference["target_next"][case_index],
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap="viridis",
            vmin=state_min,
            vmax=state_max,
        )
        _shade_front_bands(target_axis, reference, case_index)
        for variant in VARIANTS:
            payload = runs[(variant, seed)]["predictions"]
            prediction_axis = axes[row_index, prediction_columns[variant]]
            error_axis = axes[row_index, prediction_columns[variant] + 1]
            predicted_next = (
                payload["current"][case_index]
                + payload["predicted_increment"][case_index]
            )
            increment_error = (
                payload["predicted_increment"][case_index]
                - payload["target_increment"][case_index]
            )
            state_image = prediction_axis.imshow(
                predicted_next,
                origin="lower",
                extent=extent,
                aspect="auto",
                cmap="viridis",
                vmin=state_min,
                vmax=state_max,
            )
            error_image = error_axis.imshow(
                increment_error,
                origin="lower",
                extent=extent,
                aspect="auto",
                cmap="coolwarm",
                vmin=-error_limit,
                vmax=error_limit,
            )
            _shade_front_bands(prediction_axis, payload, case_index)
            _shade_front_bands(error_axis, payload, case_index)
        axes[row_index, 0].set_ylabel(
            ("Step" if row_index == 0 else "Pulse") + "\ny"
        )
        for axis in axes[row_index]:
            axis.set_xlim(0.15, 0.9)
            axis.set_ylim(0.0, DOMAIN_LENGTHS[1])
            if row_index == 1:
                axis.set_xlabel("x")
    titles = (
        "Exact next",
        "Full pred.",
        "Full error",
        "No-grad pred.",
        "No-grad error",
        "Local pred.",
        "Local error",
    )
    for axis, title in zip(axes[0], titles, strict=True):
        axis.set_title(title)
    if state_image is None or error_image is None:
        raise RuntimeError("map figure did not receive state and error fields")
    all_axes = list(axes.ravel())
    fig.colorbar(state_image, ax=all_axes, shrink=0.75, label="Next state")
    fig.colorbar(error_image, ax=all_axes, shrink=0.75, label="Increment error")
    fig.suptitle(
        f"Matched target/prediction/error maps with fixed physical front bands, seed {seed}"
    )
    return fig


def _resolution_figure(
    runs: dict[tuple[str, int], dict[str, Any]], seeds: Sequence[int]
) -> plt.Figure:
    resolutions = sorted(
        {64}
        | {
            int(row["resolution"][0])
            for run in runs.values()
            for row in run["nested_rows"]
        }
    )
    fig, axis = plt.subplots(figsize=(4.8, 3.0), constrained_layout=True)
    for variant in VARIANTS:
        means = []
        deviations = []
        for nx in resolutions:
            values = []
            for seed in seeds:
                run = runs[(variant, seed)]
                if nx == 64:
                    rows = [
                        row
                        for row in run["native_rows"]
                        if row["split"] == "held"
                        and row["family"] in {"step", "pulse"}
                    ]
                else:
                    rows = [
                        row
                        for row in run["nested_rows"]
                        if row["resolution"] == [nx, nx // 2]
                        and row["family"] in {"step", "pulse"}
                    ]
                if not rows:
                    raise ValueError(
                        f"missing held discontinuous rows for {variant}, seed {seed}, nx={nx}"
                    )
                values.append(
                    float(np.mean([row["metrics"]["relative_increment_l2"] for row in rows]))
                )
            means.append(float(np.mean(values)))
            deviations.append(float(np.std(values, ddof=1)) if len(values) > 1 else 0.0)
        axis.errorbar(
            resolutions,
            means,
            yerr=deviations,
            marker="o",
            color=COLORS[variant],
            label=LABELS[variant],
            capsize=2,
        )
    axis.set_yscale("log")
    axis.set_xticks(resolutions)
    axis.set_xlabel("nx (common physical held cases)")
    axis.set_ylabel("Mean discontinuous relative increment L2")
    axis.set_title("Resolution transfer")
    axis.legend(frameon=False)
    return fig


def _spectrum_figure(
    runs: dict[tuple[str, int], dict[str, Any]], seeds: Sequence[int]
) -> plt.Figure:
    sample = runs[("full", seeds[0])]["predictions"]
    ny, nx = sample["current"].shape[1:]
    grid = build_structured_cell_grid((nx, ny))
    q, _ = physical_cosine_spectrum(np.zeros(grid.array_shape), grid)
    edges = np.arange(0.0, math.ceil(float(np.max(q))) + 2.0, 1.0)
    centers = 0.5 * (edges[:-1] + edges[1:])
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 2.8), constrained_layout=True)
    all_energy: list[float] = []
    for variant_index, variant in enumerate(VARIANTS):
        high = []
        energy = []
        spectra = []
        for seed in seeds:
            run = runs[(variant, seed)]
            rows = _primary_rows(run)
            high.append(
                float(np.mean([row["metrics"]["physical_spectrum"]["high_fraction"] for row in rows]))
            )
            energy.append(
                float(np.mean([row["metrics"]["physical_spectrum"]["total_energy"] for row in rows]))
            )
            payload = run["predictions"]
            curves = []
            for index in _primary_indices(payload):
                error = (
                    payload["predicted_increment"][index]
                    - payload["target_increment"][index]
                )
                physical_q, coefficient_energy = physical_cosine_spectrum(error, grid)
                curves.append(
                    np.histogram(physical_q, bins=edges, weights=coefficient_energy)[0]
                )
            spectra.append(np.mean(curves, axis=0))
        all_energy.extend(energy)
        spectrum_values = np.asarray(spectra)
        spectrum_mean = np.mean(spectrum_values, axis=0)
        spectrum_deviation = (
            np.std(spectrum_values, axis=0, ddof=1)
            if len(spectra) > 1
            else np.zeros_like(spectrum_mean)
        )
        positive = spectrum_mean[spectrum_mean > 0.0]
        floor = (
            max(float(np.max(positive)) * 1.0e-12, np.finfo(np.float64).tiny)
            if positive.size
            else np.finfo(np.float64).tiny
        )
        axes[0].plot(
            centers,
            np.maximum(spectrum_mean, floor),
            color=COLORS[variant],
            linestyle=LINESTYLES[variant],
            label=LABELS[variant],
        )
        axes[0].fill_between(
            centers,
            np.maximum(spectrum_mean - spectrum_deviation, floor),
            np.maximum(spectrum_mean + spectrum_deviation, floor),
            color=COLORS[variant],
            alpha=0.15,
            linewidth=0.0,
        )
        axes[1].bar(
            variant_index,
            np.mean(energy),
            yerr=np.std(energy, ddof=1) if len(energy) > 1 else 0.0,
            color=COLORS[variant],
            capsize=2,
        )
        axes[2].bar(
            variant_index,
            np.mean(high),
            yerr=np.std(high, ddof=1) if len(high) > 1 else 0.0,
            color=COLORS[variant],
            capsize=2,
        )
    axes[0].axvline(
        SMOOTH_FILTER_PASS_WAVENUMBER, color="#666666", linestyle="--", linewidth=0.8
    )
    axes[0].axvline(
        SMOOTH_FILTER_STOP_WAVENUMBER, color="#666666", linestyle=":", linewidth=0.8
    )
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Physical cycles per unit length")
    axes[0].set_ylabel("Radial error energy")
    axes[0].set_title("Physical spectrum")
    axes[0].legend(frameon=False, fontsize=7)
    for axis in axes[1:]:
        axis.set_xticks(range(len(VARIANTS)), ("Full", "No grad", "Local"), rotation=25, ha="right")
    axes[1].set_yscale("log")
    axes[1].set_ylim(
        max(min(all_energy) * 0.5, np.finfo(np.float64).tiny),
        max(all_energy) * 2.0,
    )
    axes[1].set_ylabel("Physical spectral error energy")
    axes[1].set_title("Total error energy")
    axes[2].set_ylabel("High-wavenumber fraction")
    axes[2].set_ylim(0.0, 1.0)
    axes[2].set_title("Composition, not efficacy")
    fig.suptitle("Registered native primary-population error spectrum")
    return fig


def _convergence_figure(
    runs: dict[tuple[str, int], dict[str, Any]], seeds: Sequence[int]
) -> plt.Figure:
    fig, axis = plt.subplots(figsize=(4.8, 3.0), constrained_layout=True)
    for variant in VARIANTS:
        histories = [runs[(variant, seed)]["history"] for seed in seeds]
        steps = np.asarray([row["step"] for row in histories[0]])
        values = np.asarray(
            [
                [row["held"]["mean_relative_increment_l2"] for row in history]
                for history in histories
            ]
        )
        mean = np.mean(values, axis=0)
        axis.plot(
            steps,
            mean,
            color=COLORS[variant],
            linestyle=LINESTYLES[variant],
            label=LABELS[variant],
        )
        axis.fill_between(
            steps,
            np.min(values, axis=0),
            np.max(values, axis=0),
            color=COLORS[variant],
            alpha=0.15,
            linewidth=0,
        )
    axis.set_yscale("log")
    axis.set_xlabel("Optimizer update")
    axis.set_ylabel("Mean held relative increment L2")
    axis.set_title("Matched training convergence (mean and seed range)")
    axis.legend(frameon=False)
    return fig


def _representative_seed(
    runs: dict[tuple[str, int], dict[str, Any]], seeds: Sequence[int]
) -> int:
    values = {
        seed: float(runs[("full", seed)]["summary"]["primary_value"])
        for seed in seeds
    }
    median = float(statistics.median(values.values()))
    return min(seeds, key=lambda seed: (abs(values[seed] - median), seed))


def run_analysis(matrix_dir: Path, output_dir: Path, *, smoke: bool) -> dict[str, Any]:
    matrix_dir = matrix_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing nonempty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = (SEEDS[0],) if smoke else SEEDS
    runs = _load_matrix(matrix_dir, smoke=smoke)
    rows = _comparison_rows(runs, seeds)
    closures = _pairing_closures(runs, seeds)
    decisions = _paired_decisions(runs, seeds)
    representative_seed = _representative_seed(runs, seeds)
    input_manifest_sha256 = {
        f"{variant}_s{seed}": sha256_file(run["run_dir"] / "manifest.json")
        for (variant, seed), run in runs.items()
    }
    write_json(output_dir / "comparison_rows.json", rows)
    write_json(output_dir / "pairing_closures.json", closures)
    write_json(output_dir / "paired_decisions.json", decisions)
    _style()
    figures = []
    figures += _save(_paired_primary_figure(rows, seeds), output_dir, "fig_paired_primary")
    figures += _save(_metric_figure(rows, seeds), output_dir, "fig_structure_metrics")
    figures += _save(
        _profiles_figure(runs, representative_seed), output_dir, "fig_phase_profiles"
    )
    figures += _save(
        _phase_maps_figure(runs, representative_seed), output_dir, "fig_phase_maps"
    )
    figures += _save(_resolution_figure(runs, seeds), output_dir, "fig_resolution_transfer")
    figures += _save(_spectrum_figure(runs, seeds), output_dir, "fig_physical_spectrum")
    figures += _save(_convergence_figure(runs, seeds), output_dir, "fig_convergence")
    visual = {
        "schema": SCHEMA,
        "files": [path.name for path in figures],
        "sha256": {path.name: sha256_file(path) for path in figures},
    }
    write_json(output_dir / "visual_manifest.json", visual)
    summary = {
        "schema": SCHEMA,
        "status": "completed",
        "science_result": not smoke,
        "run_count": len(runs),
        "seeds": list(seeds),
        "representative_seed": representative_seed,
        "input_manifest_sha256": input_manifest_sha256,
        "pairing_closures": closures,
        "paired_decisions": decisions,
        "comparison_rows": rows,
        "visualizations": visual,
        "external_result_to_claim_review": "pending_private_scope_approval",
    }
    write_json(output_dir / "summary.json", summary)
    outputs = sorted(path for path in output_dir.iterdir() if path.name != "manifest.json")
    manifest = {
        "schema": SCHEMA,
        "status": "completed",
        "science_result": not smoke,
        "analysis_source_sha256": sha256_file(Path(__file__)),
        "input_manifest_sha256": input_manifest_sha256,
        "output_hashes": {path.name: sha256_file(path) for path in outputs},
        "output_count": len(outputs),
    }
    write_json(output_dir / "manifest.json", manifest)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-dir", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_analysis(args.matrix_dir, args.output_dir, smoke=args.smoke)
    print(json.dumps(summary, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
