"""Render solver-assisted defect landscapes from a verified ODE packet.

This script is a read-only consumer of ``run_corrective_ode_study.py``
artifacts.  It recomputes learned trajectories from the packet checkpoints and
compares two deliberately different one-step targets on ``S^1 x R``:

``flow_defect``
    distance from the deployed output to the trusted successor of the queried
    displaced state;

``clean_return_error``
    distance from the deployed output to the trusted successor of the clean
    anchor with the same phase.

Both are offline, solver-assisted diagnostics.  Neither defines ID/OOD and
neither is an online correction rule.  The parent study packet is never
modified.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.time_dependent_no.run_corrective_ode_study import (
    LearnedStudyConfig,
    ResidualMLP,
    _model_step,
    _test_phases,
    build_training_contract,
    chordal_error,
    lifted_coordinate_error,
    rollout,
    verify_study_packet,
)
from utility.time_dependent_no.pcno_artifacts import (
    atomic_write_json,
    git_state,
    sha256_file,
    write_csv,
)

MANIFEST_SCHEMA = "corrective_ode_landscape_manifest_v1"
SUMMARY_SCHEMA = "corrective_ode_landscape_summary_v1"
SOURCE_FILES = (
    "scripts/time_dependent_no/plot_corrective_ode_landscapes.py",
    "tests/time_dependent_no/test_corrective_ode_landscapes.py",
    "scripts/time_dependent_no/run_corrective_ode_study.py",
)
DEPLOYED_ARMS: dict[str, tuple[str, float]] = {
    "CLEAN": ("CLEAN", 1.0),
    "CLEAN_C0": ("CLEAN", 0.0),
    "RECOVERY": ("RECOVERY", 1.0),
    "DYN_RELABEL": ("DYN_RELABEL", 1.0),
}
PLOTTED_ARMS = ("CLEAN", "CLEAN_C0", "RECOVERY", "DYN_RELABEL")


@dataclass(frozen=True)
class LandscapeConfig:
    """Fixed evaluation and rendering choices for a derived figure packet."""

    theta_points: int = 257
    radius_points: int = 241
    clean_rollout_steps: int = 160
    impulse_rollout_steps: int = 40

    def validated(self, parent: LearnedStudyConfig) -> LandscapeConfig:
        if self.theta_points < 17 or self.radius_points < 17:
            raise ValueError("landscape grids must have at least 17 points per axis")
        if self.theta_points % 2 == 0 or self.radius_points % 2 == 0:
            raise ValueError("landscape grids must have odd point counts")
        if not 1 <= self.clean_rollout_steps <= parent.rollout_steps:
            raise ValueError("clean rollout horizon exceeds the parent contract")
        if not 1 <= self.impulse_rollout_steps <= parent.impulse_steps:
            raise ValueError("impulse rollout horizon exceeds the parent contract")
        return self


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected an object in {path}")
    return payload


def _parent_config(packet_dir: Path) -> LearnedStudyConfig:
    payload = _json(packet_dir / "config.json")
    if payload.get("schema") != "corrective_ode_config_v1":
        raise ValueError("unsupported parent ODE configuration schema")
    if payload.get("mode") != "all":
        raise ValueError("landscape rendering requires a learned all-mode packet")
    learned = dict(payload.get("learned", {}))
    learned["seeds"] = tuple(int(value) for value in learned.get("seeds", ()))
    learned["query_radii"] = tuple(
        float(value) for value in learned.get("query_radii", ())
    )
    return LearnedStudyConfig(**learned).validated()


def _validate_parent_contract(
    packet_dir: Path,
    *,
    require_scientific_pass: bool,
) -> tuple[LearnedStudyConfig, dict[str, Any]]:
    verification = verify_study_packet(packet_dir, verify_current_sources=True)
    manifest = _json(packet_dir / "manifest.json")
    learned_summary = _json(packet_dir / "learned_summary.json")
    if require_scientific_pass and verification.get("summary_status") != "pass":
        raise ValueError("parent ODE study did not close with status pass")
    if (
        require_scientific_pass
        and learned_summary.get("status") != "mechanism_realization_pass"
    ):
        raise ValueError("parent learned study did not pass its frozen mechanism checks")

    runner_name = "scripts/time_dependent_no/run_corrective_ode_study.py"
    runner_record = manifest.get("sources", {}).get(runner_name)
    if not isinstance(runner_record, dict):
        raise TypeError("parent manifest does not bind the ODE runner")
    current_runner = REPOSITORY_ROOT / runner_name
    if sha256_file(current_runner) != runner_record.get("sha256"):
        raise ValueError("current ODE runner differs from the parent-bound source")

    config = _parent_config(packet_dir)
    _, rebuilt_metadata = build_training_contract(config)
    recorded_metadata = learned_summary.get("paired_contract", {}).get("data")
    if rebuilt_metadata != recorded_metadata:
        raise ValueError("rebuilt training bank does not match the parent packet")
    return config, {
        "verification": verification,
        "manifest": manifest,
        "training_bank": rebuilt_metadata,
    }


def _load_models(
    packet_dir: Path, config: LearnedStudyConfig
) -> dict[int, dict[str, ResidualMLP]]:
    models: dict[int, dict[str, ResidualMLP]] = {}
    for seed in config.seeds:
        models[seed] = {}
        for arm in ("CLEAN", "RECOVERY", "DYN_RELABEL"):
            checkpoint = packet_dir / "checkpoints" / f"seed_{seed}_{arm.lower()}.pt"
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            model = ResidualMLP(
                width=config.hidden_width, hidden_layers=config.hidden_layers
            ).to(dtype=torch.float32)
            model.load_state_dict(
                torch.load(checkpoint, map_location="cpu", weights_only=True)
            )
            model.eval()
            models[seed][arm] = model
    return models


def _wrapped_phase(theta: np.ndarray | float) -> np.ndarray:
    values = np.asarray(theta, dtype=np.float64)
    return (values + np.pi) % (2.0 * np.pi) - np.pi


def periodic_path_for_plot(
    theta: Sequence[float] | np.ndarray,
    radius: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Wrap phase and insert NaNs across the periodic plotting seam."""

    wrapped = _wrapped_phase(np.asarray(theta, dtype=np.float64).reshape(-1))
    normal = np.asarray(radius, dtype=np.float64).reshape(-1)
    if wrapped.shape != normal.shape or not wrapped.size:
        raise ValueError("trajectory coordinates must be nonempty and aligned")
    if not np.isfinite(wrapped).all() or not np.isfinite(normal).all():
        raise ValueError("trajectory coordinates must be finite")
    split_after = np.flatnonzero(np.abs(np.diff(wrapped)) > np.pi)
    if not split_after.size:
        return wrapped, normal
    output_theta: list[float] = []
    output_radius: list[float] = []
    split_set = {int(index) for index in split_after}
    for index, (phase, value) in enumerate(zip(wrapped, normal, strict=True)):
        output_theta.append(float(phase))
        output_radius.append(float(value))
        if index in split_set:
            output_theta.append(float("nan"))
            output_radius.append(float("nan"))
    return np.asarray(output_theta), np.asarray(output_radius)


def _component_errors(
    output: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    phase = np.abs(_wrapped_phase(output[..., 0] - target[..., 0]))
    normal = np.abs(output[..., 1] - target[..., 1])
    return phase, normal


def compute_landscape_data(
    models: Mapping[int, Mapping[str, ResidualMLP]],
    parent: LearnedStudyConfig,
    landscape: LandscapeConfig,
) -> dict[str, np.ndarray]:
    """Evaluate every deployed arm on one common periodic state grid."""

    landscape.validated(parent)
    expected_seeds = set(parent.seeds)
    if set(models) != expected_seeds:
        raise ValueError("loaded model seeds do not match the parent contract")

    theta = np.linspace(-np.pi, np.pi, landscape.theta_points, endpoint=True)
    radius = np.linspace(
        -parent.max_displacement,
        parent.max_displacement,
        landscape.radius_points,
        endpoint=True,
    )
    theta_grid, radius_grid = np.meshgrid(theta, radius, indexing="xy")
    states = np.column_stack((theta_grid.reshape(-1), radius_grid.reshape(-1)))
    clean_anchors = states.copy()
    clean_anchors[:, 1] = 0.0
    flow_target = parent.flow.advance(states)
    return_target = parent.flow.advance(clean_anchors)

    arrays: dict[str, np.ndarray] = {
        "theta": theta,
        "radius": radius,
        "theta_grid": theta_grid,
        "radius_grid": radius_grid,
    }
    shape = theta_grid.shape
    for deployed_arm, (checkpoint_arm, rho) in DEPLOYED_ARMS.items():
        flow_lifted = []
        flow_chordal = []
        flow_phase = []
        flow_normal = []
        return_lifted = []
        return_chordal = []
        return_phase = []
        return_normal = []
        tube = []
        for seed in parent.seeds:
            model = models[seed][checkpoint_arm]
            output = _model_step(model, states, rho=rho)
            phase, normal = _component_errors(output, flow_target)
            return_phase_values, return_normal_values = _component_errors(
                output, return_target
            )
            flow_lifted.append(lifted_coordinate_error(output, flow_target))
            flow_chordal.append(chordal_error(output, flow_target))
            flow_phase.append(phase)
            flow_normal.append(normal)
            return_lifted.append(lifted_coordinate_error(output, return_target))
            return_chordal.append(chordal_error(output, return_target))
            return_phase.append(return_phase_values)
            return_normal.append(return_normal_values)
            tube.append(np.abs(output[:, 1]))

        fields = {
            "flow_defect_lifted": np.stack(flow_lifted),
            "flow_defect_chordal": np.stack(flow_chordal),
            "flow_phase_abs": np.stack(flow_phase),
            "flow_normal_abs": np.stack(flow_normal),
            "clean_return_error_lifted": np.stack(return_lifted),
            "clean_return_error_chordal": np.stack(return_chordal),
            "clean_return_phase_abs": np.stack(return_phase),
            "clean_return_normal_abs": np.stack(return_normal),
            "output_tube_distance": np.stack(tube),
        }
        for name, values in fields.items():
            if not np.isfinite(values).all():
                raise FloatingPointError(f"non-finite landscape: {deployed_arm}/{name}")
            arrays[f"{deployed_arm}_{name}_by_seed"] = values.reshape(
                len(parent.seeds), *shape
            )
            arrays[f"{deployed_arm}_{name}_mean"] = np.mean(values, axis=0).reshape(
                shape
            )
    return arrays


def _one_step_diagnostics(
    *,
    model: ResidualMLP,
    rho: float,
    flow: Any,
    state: np.ndarray,
) -> tuple[float, float, float]:
    query = np.asarray(state, dtype=np.float64).reshape(1, 2)
    output = _model_step(model, query, rho=rho)
    flow_target = flow.advance(query)
    clean_anchor = query.copy()
    clean_anchor[:, 1] = 0.0
    return_target = flow.advance(clean_anchor)
    return (
        float(lifted_coordinate_error(output, flow_target)[0]),
        float(lifted_coordinate_error(output, return_target)[0]),
        float(abs(output[0, 1])),
    )


def compute_trajectory_rows(
    models: Mapping[int, Mapping[str, ResidualMLP]],
    parent: LearnedStudyConfig,
    landscape: LandscapeConfig,
) -> list[dict[str, Any]]:
    """Recompute state-resolved trajectories from the parent checkpoints."""

    landscape.validated(parent)
    theta0 = float(_test_phases(parent)[0])
    assays = {
        "clean_start": (
            np.asarray([theta0, 0.0], dtype=np.float64),
            landscape.clean_rollout_steps,
        ),
        "normal_impulse": (
            np.asarray([theta0, parent.impulse_radius], dtype=np.float64),
            landscape.impulse_rollout_steps,
        ),
    }
    rows: list[dict[str, Any]] = []
    for deployed_arm, (checkpoint_arm, rho) in DEPLOYED_ARMS.items():
        for seed in parent.seeds:
            model = models[seed][checkpoint_arm]
            step = lambda value, current_model=model, current_rho=rho: _model_step(
                current_model, value, rho=current_rho
            )
            for assay, (initial, steps) in assays.items():
                predicted = rollout(step, initial, steps)
                clean_initial = initial.copy()
                clean_initial[1] = 0.0
                trusted_reference = rollout(parent.flow.advance, initial, steps)
                clean_reference = rollout(parent.flow.advance, clean_initial, steps)
                for index in range(steps + 1):
                    flow_defect, return_error, next_tube = _one_step_diagnostics(
                        model=model,
                        rho=rho,
                        flow=parent.flow,
                        state=predicted[index],
                    )
                    rows.append(
                        {
                            "arm": deployed_arm,
                            "seed": seed,
                            "assay": assay,
                            "step": index,
                            "theta_lifted": float(predicted[index, 0]),
                            "theta_wrapped": float(_wrapped_phase(predicted[index, 0])),
                            "normal": float(predicted[index, 1]),
                            "tube_distance": float(abs(predicted[index, 1])),
                            "inside_training_strip": bool(
                                abs(predicted[index, 1]) <= parent.max_displacement
                            ),
                            "inside_registered_tube": bool(
                                abs(predicted[index, 1]) <= parent.tube_radius
                            ),
                            "visited_flow_defect_lifted": flow_defect,
                            "visited_clean_return_error_lifted": return_error,
                            "next_output_tube_distance": next_tube,
                            "trusted_path_error_lifted": float(
                                lifted_coordinate_error(
                                    predicted[index], trusted_reference[index]
                                )
                            ),
                            "clean_path_error_lifted": float(
                                lifted_coordinate_error(
                                    predicted[index], clean_reference[index]
                                )
                            ),
                            "trusted_theta_lifted": float(
                                trusted_reference[index, 0]
                            ),
                            "trusted_normal": float(trusted_reference[index, 1]),
                            "clean_theta_lifted": float(clean_reference[index, 0]),
                            "clean_normal": float(clean_reference[index, 1]),
                        }
                    )
    return rows


def _first_exit(rows: Sequence[Mapping[str, Any]], key: str) -> int | None:
    for row in rows:
        if not bool(row[key]):
            return int(row["step"])
    return None


def _summarize(
    arrays: Mapping[str, np.ndarray],
    trajectory_rows: Sequence[Mapping[str, Any]],
    parent: LearnedStudyConfig,
    landscape: LandscapeConfig,
) -> dict[str, Any]:
    arms: dict[str, Any] = {}
    radius_grid = arrays["radius_grid"]
    nonzero = np.abs(radius_grid) > 1.0e-12
    for arm in DEPLOYED_ARMS:
        arm_rows = [row for row in trajectory_rows if row["arm"] == arm]
        clean_by_seed: dict[str, Any] = {}
        impulse_by_seed: dict[str, Any] = {}
        for seed in parent.seeds:
            clean = [
                row
                for row in arm_rows
                if row["seed"] == seed and row["assay"] == "clean_start"
            ]
            impulse = [
                row
                for row in arm_rows
                if row["seed"] == seed and row["assay"] == "normal_impulse"
            ]
            clean_by_seed[str(seed)] = {
                "first_tube_exit": _first_exit(clean, "inside_registered_tube"),
                "first_training_strip_exit": _first_exit(
                    clean, "inside_training_strip"
                ),
                "max_tube_distance": max(float(row["tube_distance"]) for row in clean),
                "max_visited_flow_defect": max(
                    float(row["visited_flow_defect_lifted"]) for row in clean
                ),
                "final_path_error": float(clean[-1]["trusted_path_error_lifted"]),
            }
            impulse_by_seed[str(seed)] = {
                "first_tube_exit": _first_exit(impulse, "inside_registered_tube"),
                "first_training_strip_exit": _first_exit(
                    impulse, "inside_training_strip"
                ),
                "final_trusted_path_error": float(
                    impulse[-1]["trusted_path_error_lifted"]
                ),
                "final_clean_path_error": float(
                    impulse[-1]["clean_path_error_lifted"]
                ),
            }
        tube_field = arrays[f"{arm}_output_tube_distance_mean"]
        contraction = tube_field[nonzero] / np.abs(radius_grid[nonzero])
        arms[arm] = {
            "grid_flow_defect_lifted_mean": float(
                np.mean(arrays[f"{arm}_flow_defect_lifted_mean"])
            ),
            "grid_clean_return_error_lifted_mean": float(
                np.mean(arrays[f"{arm}_clean_return_error_lifted_mean"])
            ),
            "grid_output_tube_distance_mean": float(np.mean(tube_field)),
            "grid_output_to_input_tube_ratio_mean_away_from_zero": float(
                np.mean(contraction)
            ),
            "clean_start_by_seed": clean_by_seed,
            "normal_impulse_by_seed": impulse_by_seed,
        }
    return {
        "schema": SUMMARY_SCHEMA,
        "scope": "offline_solver_assisted_ode_diagnostic_only",
        "defines_id_or_ood": False,
        "online_detector_or_corrector": False,
        "seed_aggregation": "arithmetic_mean_of_error_magnitudes",
        "state_chart": "theta_periodic_by_r_signed_normal",
        "landscape": asdict(landscape),
        "parent_learned_config": asdict(parent),
        "arms": arms,
    }


def _positive_limits(fields: Sequence[np.ndarray]) -> tuple[float, float]:
    values = np.concatenate([np.asarray(field, dtype=np.float64).reshape(-1) for field in fields])
    positive = values[values > 0.0]
    if not positive.size:
        raise ValueError("logarithmic landscape has no positive values")
    lower = max(float(np.min(positive)), 1.0e-8)
    upper = float(np.max(positive))
    if not lower < upper:
        upper = lower * 10.0
    return lower, upper


def _rows_for_path(
    rows: Sequence[Mapping[str, Any]], *, arm: str, assay: str, seed: int
) -> list[Mapping[str, Any]]:
    return [
        row
        for row in rows
        if row["arm"] == arm and row["assay"] == assay and row["seed"] == seed
    ]


def _plot_path(
    axis: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    radius_limit: float,
    color: str,
    linewidth: float,
    alpha: float,
    linestyle: str = "-",
) -> None:
    theta = np.asarray([row["theta_lifted"] for row in rows], dtype=np.float64)
    radius = np.asarray([row["normal"] for row in rows], dtype=np.float64)
    inside = np.abs(radius) <= radius_limit
    outside_indices = np.flatnonzero(~inside)
    first_out = int(outside_indices[0]) if outside_indices.size else None
    stop = first_out if first_out is not None else len(theta)
    if stop:
        theta_plot, radius_plot = periodic_path_for_plot(
            theta[:stop], radius[:stop]
        )
        axis.plot(
            theta_plot,
            radius_plot,
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            linestyle=linestyle,
            zorder=6,
        )
    if first_out is not None:
        axis.scatter(
            [_wrapped_phase(theta[first_out])],
            [math.copysign(radius_limit, radius[first_out])],
            marker="x",
            s=24,
            linewidths=1.1,
            color=color,
            zorder=8,
        )


def _plot_landscapes(
    *,
    arrays: Mapping[str, np.ndarray],
    trajectory_rows: Sequence[Mapping[str, Any]],
    training_inputs: Mapping[str, np.ndarray],
    parent: LearnedStudyConfig,
    output_pdf: Path,
    output_png: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.lines import Line2D

    flow_fields = [
        arrays[f"{arm}_flow_defect_lifted_mean"] for arm in PLOTTED_ARMS
    ]
    return_fields = [
        arrays[f"{arm}_clean_return_error_lifted_mean"] for arm in PLOTTED_ARMS
    ]
    flow_limits = _positive_limits(flow_fields)
    return_limits = _positive_limits(return_fields)
    theta = arrays["theta"]
    radius = arrays["radius"]
    radius_limit = parent.max_displacement

    fig, axes = plt.subplots(
        2,
        len(PLOTTED_ARMS),
        figsize=(12.0, 5.8),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    image_rows = []
    labels = {
        "CLEAN": "CLEAN",
        "CLEAN_C0": r"CLEAN + $C_0$",
        "RECOVERY": "RECOVERY",
        "DYN_RELABEL": "DYN-RELABEL",
    }
    seed_colors = ("#FFFFFF", "#56B4E9", "#F0E442")
    for column, arm in enumerate(PLOTTED_ARMS):
        top = axes[0, column]
        bottom = axes[1, column]
        top_image = top.pcolormesh(
            theta,
            radius,
            arrays[f"{arm}_flow_defect_lifted_mean"],
            shading="auto",
            cmap="magma",
            norm=LogNorm(vmin=flow_limits[0], vmax=flow_limits[1]),
            rasterized=True,
        )
        bottom_image = bottom.pcolormesh(
            theta,
            radius,
            arrays[f"{arm}_clean_return_error_lifted_mean"],
            shading="auto",
            cmap="magma",
            norm=LogNorm(vmin=return_limits[0], vmax=return_limits[1]),
            rasterized=True,
        )
        image_rows.append((top_image, bottom_image))

        support = training_inputs[arm]
        for axis in (top, bottom):
            axis.axhline(0.0, color="#00E5FF", linewidth=1.0, zorder=4)
            axis.axhline(
                parent.tube_radius,
                color="white",
                linewidth=0.7,
                linestyle=":",
                alpha=0.9,
                zorder=4,
            )
            axis.axhline(
                -parent.tube_radius,
                color="white",
                linewidth=0.7,
                linestyle=":",
                alpha=0.9,
                zorder=4,
            )
            axis.scatter(
                support[:, 0],
                support[:, 1],
                s=3.2,
                color="white",
                edgecolors="black",
                linewidths=0.15,
                alpha=0.65,
                zorder=5,
            )
            axis.set_xlim(-np.pi, np.pi)
            axis.set_ylim(-radius_limit, radius_limit)
            axis.set_xticks((-np.pi, 0.0, np.pi), (r"$-\pi$", "0", r"$\pi$"))
            axis.spines[["top", "right"]].set_visible(False)

        top.text(
            0.5,
            1.03,
            labels[arm],
            transform=top.transAxes,
            ha="center",
            va="bottom",
            fontsize=10,
        )
        for seed_index, seed in enumerate(parent.seeds):
            color = seed_colors[seed_index % len(seed_colors)]
            _plot_path(
                top,
                _rows_for_path(
                    trajectory_rows, arm=arm, assay="clean_start", seed=seed
                ),
                radius_limit=radius_limit,
                color=color,
                linewidth=1.0,
                alpha=0.9,
            )
            _plot_path(
                bottom,
                _rows_for_path(
                    trajectory_rows, arm=arm, assay="normal_impulse", seed=seed
                ),
                radius_limit=radius_limit,
                color=color,
                linewidth=1.0,
                alpha=0.9,
            )

        theta0 = float(_test_phases(parent)[0])
        impulse_horizon = max(
            int(row["step"])
            for row in trajectory_rows
            if row["assay"] == "normal_impulse"
        )
        trusted = rollout(
            parent.flow.advance,
            np.asarray([theta0, parent.impulse_radius]),
            min(parent.impulse_steps, impulse_horizon),
        )
        clean = rollout(
            parent.flow.advance,
            np.asarray([theta0, 0.0]),
            trusted.shape[0] - 1,
        )
        trusted_theta, trusted_radius = periodic_path_for_plot(
            trusted[:, 0], trusted[:, 1]
        )
        clean_theta, clean_radius = periodic_path_for_plot(clean[:, 0], clean[:, 1])
        bottom.plot(
            trusted_theta,
            trusted_radius,
            color="#009E73",
            linewidth=1.4,
            linestyle="--",
            zorder=7,
        )
        bottom.plot(
            clean_theta,
            clean_radius,
            color="#00E5FF",
            linewidth=1.2,
            linestyle="-.",
            zorder=7,
        )

    axes[0, 0].set_ylabel(r"flow defect; normal coordinate $r$")
    axes[1, 0].set_ylabel(r"clean-return error; normal coordinate $r$")
    for axis in axes[1]:
        axis.set_xlabel(r"periodic phase $\theta$")

    fig.colorbar(
        image_rows[-1][0],
        ax=list(axes[0]),
        location="right",
        shrink=0.86,
        label="mean trusted-flow defect",
    )
    fig.colorbar(
        image_rows[-1][1],
        ax=list(axes[1]),
        location="right",
        shrink=0.86,
        label="mean clean-return error",
    )
    legend_handles = [
        Line2D([0], [0], color="#00E5FF", linewidth=1.2, label=r"reference set $r=0$"),
        Line2D([0], [0], color="black", marker="o", markersize=3, linewidth=0, label="training input"),
        Line2D([0], [0], color="black", linewidth=1.0, label="learned rollout (three seeds)"),
        Line2D(
            [0],
            [0],
            color="black",
            marker="x",
            markersize=5,
            linewidth=0,
            label="first sample outside plotted strip",
        ),
        Line2D([0], [0], color="#009E73", linestyle="--", linewidth=1.4, label="trusted displaced path"),
        Line2D([0], [0], color="#00E5FF", linestyle="-.", linewidth=1.2, label="clean path"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.035),
        ncol=6,
        frameon=False,
        fontsize=7.5,
    )
    fig.savefig(output_pdf, bbox_inches="tight")
    fig.savefig(output_png, dpi=240, bbox_inches="tight")
    plt.close(fig)


def _training_inputs(
    parent: LearnedStudyConfig,
) -> dict[str, np.ndarray]:
    datasets, _ = build_training_contract(parent)
    count = parent.train_phases
    clean = np.asarray(datasets["CLEAN"][0][:count], dtype=np.float64)
    displaced = np.asarray(datasets["RECOVERY"][0][count:], dtype=np.float64)
    response_support = np.concatenate((clean, displaced), axis=0)
    return {
        "CLEAN": clean,
        "CLEAN_C0": clean,
        "RECOVERY": response_support,
        "DYN_RELABEL": response_support,
    }


def _source_records() -> dict[str, dict[str, Any]]:
    records = {}
    for relative_name in SOURCE_FILES:
        path = REPOSITORY_ROOT / relative_name
        if not path.is_file():
            raise FileNotFoundError(path)
        records[relative_name] = {
            "sha256": sha256_file(path),
            "bytes": int(path.stat().st_size),
        }
    return records


def _output_records(output_dir: Path) -> dict[str, dict[str, Any]]:
    records = {}
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path.name in {"manifest.json", "closeout.json"}:
            continue
        records[path.relative_to(output_dir).as_posix()] = {
            "sha256": sha256_file(path),
            "bytes": int(path.stat().st_size),
        }
    return records


def generate_landscape_packet(
    *,
    packet_dir: str | Path,
    output_dir: str | Path,
    landscape: LandscapeConfig | None = None,
    require_scientific_pass: bool = True,
) -> dict[str, Any]:
    parent_root = Path(packet_dir).resolve()
    output_root = Path(output_dir)
    parent, parent_contract = _validate_parent_contract(
        parent_root, require_scientific_pass=require_scientific_pass
    )
    landscape_config = (landscape or LandscapeConfig()).validated(parent)
    models = _load_models(parent_root, parent)
    output_root.mkdir(parents=True, exist_ok=False)

    arrays = compute_landscape_data(models, parent, landscape_config)
    rows = compute_trajectory_rows(models, parent, landscape_config)
    support = _training_inputs(parent)
    np.savez_compressed(output_root / "landscapes.npz", **arrays)
    write_csv(output_root / "trajectories.csv", rows)
    summary = _summarize(arrays, rows, parent, landscape_config)
    atomic_write_json(output_root / "summary.json", summary)
    _plot_landscapes(
        arrays=arrays,
        trajectory_rows=rows,
        training_inputs=support,
        parent=parent,
        output_pdf=output_root / "ode_defect_landscapes.pdf",
        output_png=output_root / "ode_defect_landscapes.png",
    )

    parent_manifest_path = parent_root / "manifest.json"
    parent_manifest = parent_contract["manifest"]
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "scope": "derived_read_only_visualization",
        "git": git_state(),
        "configuration": asdict(landscape_config),
        "parent_scientific_pass_required": require_scientific_pass,
        "parent": {
            "packet_name": parent_root.name,
            "manifest_sha256": sha256_file(parent_manifest_path),
            "summary_sha256": sha256_file(parent_root / "summary.json"),
            "verification": parent_contract["verification"],
            "runner_sha256": parent_manifest["sources"][
                "scripts/time_dependent_no/run_corrective_ode_study.py"
            ]["sha256"],
            "checkpoint_sha256": {
                name: record["sha256"]
                for name, record in parent_manifest["outputs"].items()
                if name.startswith("checkpoints/")
            },
            "training_bank": parent_contract["training_bank"],
        },
        "sources": _source_records(),
        "outputs": _output_records(output_root),
    }
    atomic_write_json(output_root / "manifest.json", manifest)
    atomic_write_json(
        output_root / "closeout.json",
        {
            "schema": "corrective_ode_landscape_closeout_v1",
            "status": "complete",
            "manifest_sha256": sha256_file(output_root / "manifest.json"),
        },
    )
    return verify_landscape_packet(
        output_root, packet_dir=parent_root, verify_current_sources=True
    )


def verify_landscape_packet(
    output_dir: str | Path,
    *,
    packet_dir: str | Path,
    verify_current_sources: bool = True,
) -> dict[str, Any]:
    root = Path(output_dir)
    parent_root = Path(packet_dir).resolve()
    manifest_path = root / "manifest.json"
    closeout_path = root / "closeout.json"
    if not manifest_path.is_file() or not closeout_path.is_file():
        raise FileNotFoundError("landscape packet lacks manifest.json or closeout.json")
    manifest = _json(manifest_path)
    closeout = _json(closeout_path)
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("unsupported landscape manifest schema")
    if closeout.get("manifest_sha256") != sha256_file(manifest_path):
        raise ValueError("landscape closeout manifest hash mismatch")
    for relative_name, record in manifest.get("outputs", {}).items():
        path = root / relative_name
        if not path.is_file() or sha256_file(path) != record.get("sha256"):
            raise ValueError(f"landscape output hash mismatch: {relative_name}")
        if int(path.stat().st_size) != int(record.get("bytes", -1)):
            raise ValueError(f"landscape output size mismatch: {relative_name}")
    if verify_current_sources:
        for relative_name, record in manifest.get("sources", {}).items():
            path = REPOSITORY_ROOT / relative_name
            if not path.is_file() or sha256_file(path) != record.get("sha256"):
                raise ValueError(f"current landscape source mismatch: {relative_name}")

    parent_verification = verify_study_packet(
        parent_root, verify_current_sources=verify_current_sources
    )
    parent_manifest_sha = sha256_file(parent_root / "manifest.json")
    if parent_manifest_sha != manifest.get("parent", {}).get("manifest_sha256"):
        raise ValueError("parent ODE manifest hash mismatch")
    if parent_root.name != manifest.get("parent", {}).get("packet_name"):
        raise ValueError("parent ODE packet identity mismatch")
    return {
        "status": "verified",
        "manifest_sha256": sha256_file(manifest_path),
        "outputs": len(manifest.get("outputs", {})),
        "sources": len(manifest.get("sources", {})),
        "current_sources_checked": verify_current_sources,
        "parent_manifest_sha256": parent_manifest_sha,
        "parent_status": parent_verification.get("summary_status"),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render or verify derived corrective-ODE defect landscapes. "
            "The parent study packet is read-only."
        )
    )
    parser.add_argument("--mode", choices=("generate", "verify"), required=True)
    parser.add_argument("--packet-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--theta-points", type=int, default=257)
    parser.add_argument("--radius-points", type=int, default=241)
    parser.add_argument(
        "--packet-only",
        action="store_true",
        help="with --mode verify, verify persisted packets without current sources",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.mode == "verify":
        result = verify_landscape_packet(
            args.output_dir,
            packet_dir=args.packet_dir,
            verify_current_sources=not args.packet_only,
        )
    else:
        if args.packet_only:
            raise ValueError("--packet-only is valid only with --mode verify")
        result = generate_landscape_packet(
            packet_dir=args.packet_dir,
            output_dir=args.output_dir,
            landscape=LandscapeConfig(
                theta_points=args.theta_points,
                radius_points=args.radius_points,
            ),
        )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
