#!/usr/bin/env python3
"""Render contract-labeled bump transfer metrics and admissible-prefix movies."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utility.time_dependent_no.pcno_artifacts import (
    atomic_write_json_with_paths as atomic_write_json,
)
from utility.time_dependent_no.pcno_artifacts import sha256_file
from utility.time_dependent_no.pcno_artifacts import (
    write_csv_with_paths as write_csv,
)

INPUT_SCHEMA = "pcno_d085_bump_fixed_fourier_rotation_v1"
LEGACY_INPUT_SCHEMA = "pcno_bump_geometric_consistency_v1"
OUTPUT_SCHEMA = "pcno_bump_geometric_consistency_visualization_v2"
PHASES = ("g1a", "g1b")
TYPE_ARMS = ("correct", "all_normal")
COMPONENT_NAMES = ("density", "x-momentum", "y-momentum", "energy")
METRIC_IDENTITY_FIELDS = (
    "case_id",
    "phase",
    "arm",
    "mode",
    "call",
    "metric",
    "frame",
    "region",
    "component",
)
INPUT_CONTRACTS = {
    LEGACY_INPUT_SCHEMA: {
        "study_id": "D083",
        "receipt_schema": "pcno_d083_terminal_receipt_v1",
        "fourier_policy": "transported_modes_and_transformed_phase_origin",
        "study_label": "D083 legacy transported-Fourier analytic covariance probe",
        "rotation_prediction_label": "inverse-rotated prediction (transported modes)",
        "rotation_panel_label": "transported-mode",
        "claim_boundary": (
            "analytic covariance under transported modes and transformed phase origin; "
            "not fixed-checkpoint-mode deployment, unseen geometry, an independently "
            "solved rotated PDE, or broad rotation equivariance"
        ),
    },
    INPUT_SCHEMA: {
        "study_id": "D085",
        "receipt_schema": "pcno_d085_terminal_receipt_v1",
        "fourier_policy": "checkpoint_native_modes_fixed_world_coordinates",
        "study_label": "D085 fixed-checkpoint-Fourier 90-degree transformed-input test",
        "rotation_prediction_label": "inverse-rotated prediction (fixed checkpoint modes)",
        "rotation_panel_label": "fixed-mode",
        "claim_boundary": (
            "fixed-checkpoint-mode transformed-input evidence on analytically rotated "
            "retained cases; not an independently solved rotated PDE, unseen-case or "
            "broad geometry generalization, or architecture-level equivariance"
        ),
    },
}


def _phase_study_label(contract: Mapping[str, str], phase: str) -> str:
    if phase == "g1a":
        return (
            f"{contract['study_id']} proxy-mass query graph "
            "(not PDE resolution transfer)"
        )
    return contract["study_label"]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--max-nodes", type=int, default=4000)
    parser.add_argument("--components", nargs="+", type=int, default=(0, 1, 2, 3))
    parser.add_argument("--phases", nargs="+", choices=PHASES)
    parser.add_argument("--skip-animations", action="store_true")
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(
            f"output directory already exists; choose a new path: {args.output_dir}"
        )
    if args.fps < 1 or args.max_nodes < 100:
        raise ValueError("fps must be positive and max-nodes at least 100")
    if tuple(sorted(set(args.components))) != tuple(args.components) or any(
        value not in range(4) for value in args.components
    ):
        raise ValueError("components must be unique increasing values in 0..3")
    return args


def _verify_input(root: Path) -> dict[str, Any]:
    summary_path = root / "summary.json"
    manifest_path = root / "artifact_manifest.json"
    receipt_path = root / "terminal_receipt.json"
    if (
        not summary_path.is_file()
        or not manifest_path.is_file()
        or not receipt_path.is_file()
    ):
        raise FileNotFoundError(
            "D083 summary, artifact manifest, or terminal receipt is missing"
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    contract = INPUT_CONTRACTS.get(summary.get("schema"))
    if contract is None:
        raise ValueError(
            f"unsupported bump visualization schema: {summary.get('schema')}"
        )
    if (
        summary.get("status") != "complete"
        or summary.get("scientific_interpretation_allowed") is not True
        or summary.get("declared_matrix_attempt_complete") is not True
    ):
        raise ValueError("visualization requires a complete scientific bump result")
    if manifest.get("schema") != "pcno_declared_artifact_manifest_v1":
        raise ValueError("unexpected bump artifact manifest")
    if receipt.get("schema") != contract["receipt_schema"]:
        raise ValueError("terminal receipt does not match the input Fourier contract")
    if receipt.get("summary_sha256") != sha256_file(summary_path):
        raise ValueError("terminal receipt does not bind summary.json")
    if receipt.get("artifact_manifest_sha256") != sha256_file(manifest_path):
        raise ValueError("terminal receipt does not bind artifact_manifest.json")
    if (
        receipt.get("execution_status") != summary["status"]
        or receipt.get("declared_matrix_attempt_complete")
        is not summary["declared_matrix_attempt_complete"]
        or receipt.get("declared_matrix_horizon_complete")
        is not summary["declared_matrix_horizon_complete"]
    ):
        raise ValueError("terminal receipt status differs from summary")
    for relative, expected in manifest["files"].items():
        path = root / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"bump artifact hash mismatch: {relative}")
    if sha256_file(manifest_path) != summary["artifact_manifest_sha256"]:
        raise ValueError("summary and artifact manifest differ")
    if summary["schema"] == INPUT_SCHEMA:
        actual_policy = summary.get("checkpoint_contract", {}).get("fourier_policy")
        if actual_policy != contract["fourier_policy"]:
            raise ValueError("D085 summary does not bind the fixed Fourier policy")
    return summary


def _read_metrics(path: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows = []
    seen: dict[tuple[Any, ...], dict[str, Any]] = {}
    removed = 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, Any] = dict(raw)
            for key in (
                "call",
                "physical_time",
                "numerator",
                "denominator",
                "value",
                "raw_physical_numerator",
                "raw_physical_denominator",
            ):
                value = row.get(key, "")
                row[key] = None if value in {"", "None"} else float(value)
            row["call"] = int(row["call"])
            key = tuple(row[field] for field in METRIC_IDENTITY_FIELDS)
            previous = seen.get(key)
            if previous is not None:
                if previous != row:
                    raise ValueError(f"conflicting duplicate metric identity: {key}")
                removed += 1
                continue
            seen[key] = row
            rows.append(row)
    return rows, {
        "input_rows": len(rows) + removed,
        "canonical_rows": len(rows),
        "exact_duplicate_rows_removed": removed,
    }


def _parse_bool(value: str, *, field: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"invalid completion boolean for {field}: {value!r}")


def _read_admissible_prefixes(
    path: Path,
) -> dict[tuple[str, str, str], dict[str, int | None]]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (str(row["case_id"]), str(row["geometry"]), str(row["type_arm"]))
            grouped[key].append(row)
    result: dict[tuple[str, str, str], dict[str, int | None]] = {}
    for key, rows in grouped.items():
        calls = [int(row["call"]) for row in rows]
        if calls != list(range(1, len(rows) + 1)):
            raise ValueError(f"noncontiguous or duplicate completion calls for {key}")
        admissible = [
            _parse_bool(row["admissible"], field="admissible") for row in rows
        ]
        failed = [index for index, passed in enumerate(admissible) if not passed]
        if len(failed) > 1 or (failed and failed[0] != len(rows) - 1):
            raise ValueError(
                f"completion rows continue after inadmissibility for {key}"
            )
        accepted = failed[0] if failed else len(rows)
        result[key] = {
            "accepted_calls": accepted,
            "first_inadmissible_call": None if not failed else failed[0] + 1,
            "recorded_calls": len(rows),
        }
    return result


def paired_admissible_prefix(
    prefixes: Mapping[tuple[str, str, str], Mapping[str, int | None]],
    *,
    case_id: str,
    phase: str,
    type_arm: str,
) -> dict[str, int | None]:
    transformed_geometry = "query" if phase == "g1a" else "rotated"
    native = prefixes[(case_id, "native", type_arm)]
    transformed = prefixes[(case_id, transformed_geometry, type_arm)]
    return {
        "accepted_calls": min(
            int(native["accepted_calls"]), int(transformed["accepted_calls"])
        ),
        "native_first_inadmissible_call": native["first_inadmissible_call"],
        "transformed_first_inadmissible_call": transformed["first_inadmissible_call"],
    }


def _case_first_curve(
    rows: Sequence[Mapping[str, Any]],
    *,
    phase: str,
    arms: Sequence[str],
    metric: str,
    mode: str,
    frame: str,
    field: str = "value",
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    selected: dict[str, dict[int, dict[str, float]]] = {
        arm: defaultdict(dict) for arm in arms
    }
    for row in rows:
        value = row.get(field)
        if (
            row["phase"] == phase
            and row["arm"] in selected
            and row["metric"] == metric
            and row["mode"] == mode
            and row["frame"] == frame
            and row["region"] == "all"
            and row["component"] == "all"
            and value is not None
            and math.isfinite(float(value))
        ):
            by_case = selected[str(row["arm"])][int(row["call"])]
            case_id = str(row["case_id"])
            if case_id in by_case:
                raise ValueError(
                    "case-first curve received a duplicate case contribution: "
                    f"{row['arm']}, call {row['call']}, case {case_id}"
                )
            by_case[case_id] = float(value)
    curves = {}
    for arm, by_call in selected.items():
        calls = np.asarray(sorted(by_call), dtype=np.int64)
        median = np.asarray(
            [np.median(list(by_call[value].values())) for value in calls]
        )
        lower = np.asarray(
            [np.quantile(list(by_call[value].values()), 0.25) for value in calls]
        )
        upper = np.asarray(
            [np.quantile(list(by_call[value].values()), 0.75) for value in calls]
        )
        curves[arm] = calls, median, lower, upper
    return curves


def _plot_curves(
    axis: plt.Axes,
    curves: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    *,
    title: str,
    ylabel: str,
) -> None:
    colors = plt.get_cmap("tab10")
    positive = []
    for index, (arm, (calls, median, lower, upper)) in enumerate(curves.items()):
        if not calls.size:
            continue
        color = colors(index)
        axis.plot(calls, median, label=arm, color=color, linewidth=1.8)
        axis.fill_between(calls, lower, upper, color=color, alpha=0.18)
        positive.extend(median[median > 0.0].tolist())
    if positive:
        axis.set_yscale("log")
    axis.set_title(title)
    axis.set_xlabel("rollout call")
    axis.set_ylabel(ylabel)
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(fontsize=7)


def _metric_figures(
    rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    *,
    phases: Sequence[str],
    contract: Mapping[str, str],
) -> list[Path]:
    rotation_label = contract["rotation_panel_label"]
    specifications = {
        "g1a": (
            (
                (
                    "native_correct",
                    "native_all_normal",
                    "query_correct",
                    "query_all_normal",
                ),
                "state_relative_l2",
                "free_rollout",
                "native",
                "State error",
            ),
            (
                (
                    "native_correct",
                    "native_all_normal",
                    "query_correct",
                    "query_all_normal",
                ),
                "predicted_increment_relative_error",
                "free_rollout",
                "native",
                "Increment error / true increment",
            ),
            (
                ("query_vs_native_correct", "query_vs_native_all_normal"),
                "increment_commutator",
                "free_rollout",
                "query",
                "Free query/native increment defect",
            ),
            (
                ("query_vs_native_correct", "query_vs_native_all_normal"),
                "accumulated_predicted_gap",
                "free_rollout",
                "query",
                "Accumulated predicted gap",
            ),
        ),
        "g1b": (
            (
                (
                    "native_correct",
                    "native_all_normal",
                    "rotated_correct",
                    "rotated_all_normal",
                ),
                "state_relative_l2",
                "free_rollout",
                "native",
                "State error",
            ),
            (
                (
                    "native_correct",
                    "native_all_normal",
                    "rotated_correct",
                    "rotated_all_normal",
                ),
                "predicted_increment_relative_error",
                "free_rollout",
                "native",
                "Increment error / true increment",
            ),
            (
                ("rotated_vs_native_correct", "rotated_vs_native_all_normal"),
                "increment_covariance_defect",
                "free_rollout",
                "native",
                f"{rotation_label} increment covariance defect",
            ),
            (
                ("rotated_vs_native_correct", "rotated_vs_native_all_normal"),
                "rotation_consistency_CQ",
                "free_rollout",
                "native",
                f"{rotation_label} state covariance C_Q",
            ),
        ),
    }
    paths = []
    for phase in phases:
        panels = specifications[phase]
        figure, axes = plt.subplots(2, 2, figsize=(12.0, 8.0), constrained_layout=True)
        for axis, (arms, metric, mode, frame, title) in zip(axes.flat, panels):
            curves = _case_first_curve(
                rows,
                phase=phase,
                arms=arms,
                metric=metric,
                mode=mode,
                frame=frame,
            )
            _plot_curves(axis, curves, title=title, ylabel="case-first median (IQR)")
        figure.suptitle(
            f"{_phase_study_label(contract, phase)} | {phase.upper()} metrics"
        )
        for suffix in ("png", "pdf"):
            path = output_dir / f"metrics_evolution_{phase}.{suffix}"
            figure.savefig(path, dpi=180)
            paths.append(path)
        plt.close(figure)

    absolute_specs = {
        "g1a": (
            "g1a",
            ("query_vs_native_correct", "query_vs_native_all_normal"),
            "free_total_defect",
            "query",
            "G1a raw defect RMS",
        ),
        "g1b": (
            "g1b",
            ("rotated_vs_native_correct", "rotated_vs_native_all_normal"),
            "free_total_defect",
            "native",
            "G1b raw defect RMS",
        ),
    }
    figure, axes = plt.subplots(
        1, len(phases), figsize=(6.0 * len(phases), 4.2), constrained_layout=True
    )
    axes_array = np.atleast_1d(axes)
    for axis, phase in zip(axes_array, phases):
        phase, arms, metric, frame, title = absolute_specs[phase]
        curves = _case_first_curve(
            rows,
            phase=phase,
            arms=arms,
            metric=metric,
            mode="free_rollout",
            frame=frame,
            field="raw_physical_numerator",
        )
        _plot_curves(axis, curves, title=title, ylabel="physical proxy-weighted RMS")
    figure.suptitle(" | ".join(_phase_study_label(contract, phase) for phase in phases))
    for suffix in ("png", "pdf"):
        path = output_dir / f"absolute_defect_numerators.{suffix}"
        figure.savefig(path, dpi=180)
        paths.append(path)
    plt.close(figure)
    return paths


def reference_color_scales(truth: np.ndarray, component: int) -> dict[str, float]:
    """Return fixed, reference-only physical scales for one component."""

    values = np.asarray(truth, dtype=np.float64)
    residual = np.diff(values[..., component], axis=0)
    cumulative = values[1:, :, component] - values[0, :, component]
    residual_limit = max(float(np.nanmax(np.abs(residual))), 1.0e-12)
    cumulative_limit = max(float(np.nanmax(np.abs(cumulative))), 1.0e-12)
    growth_limit = max(
        2.0 * cumulative_limit * residual_limit + residual_limit**2,
        1.0e-12,
    )
    return {
        "residual_limit": residual_limit,
        "cumulative_limit": cumulative_limit,
        "signed_growth_limit": growth_limit,
    }


def residual_animation_fields(
    artifact: Mapping[str, np.ndarray],
    *,
    phase: str,
    type_arm: str,
    component: int,
    rotation_prediction_label: str | None = None,
) -> tuple[list[str], list[np.ndarray], dict[str, float]]:
    """Derive the ten fixed-contract free-rollout fields for rendering."""

    truth = np.asarray(artifact["truth"], dtype=np.float64)
    native = np.asarray(artifact[f"native_{type_arm}"], dtype=np.float64)
    if phase == "g1a":
        transformed = np.asarray(
            artifact[f"query_{type_arm}_prolonged"], dtype=np.float64
        )
        mesh = np.asarray(artifact[f"query_mesh_{type_arm}"], dtype=np.float64)
        state = np.asarray(artifact[f"query_state_{type_arm}"], dtype=np.float64)
        transformed_name = "query"
    elif phase == "g1b":
        transformed = np.asarray(
            artifact[f"rotated_{type_arm}_inverse"], dtype=np.float64
        )
        mesh = np.asarray(artifact[f"rotation_mesh_{type_arm}"], dtype=np.float64)
        state = np.asarray(artifact[f"rotation_state_{type_arm}"], dtype=np.float64)
        transformed_name = rotation_prediction_label or "inverse-rotated"
    else:
        raise ValueError(f"unexpected phase: {phase}")
    true_residual = np.diff(truth, axis=0)[..., component]
    native_residual = np.diff(native, axis=0)[..., component]
    transformed_residual = np.diff(transformed, axis=0)[..., component]
    mesh_component = mesh[..., component]
    state_component = state[..., component]
    total = mesh_component + state_component
    cumulative = np.cumsum(total, axis=0)
    previous = cumulative - total
    signed_growth = 2.0 * previous * total + np.square(total)
    titles = [
        "true increment",
        "native predicted increment",
        f"{transformed_name} predicted increment",
        "native increment error",
        f"{transformed_name} increment error",
        "total cross-representation defect",
        "cumulative defect",
        "same-input mesh defect",
        "recurrent-state response",
        "signed local error growth",
    ]
    fields = [
        true_residual,
        native_residual,
        transformed_residual,
        native_residual - true_residual,
        transformed_residual - true_residual,
        total,
        cumulative,
        mesh_component,
        state_component,
        signed_growth,
    ]
    return titles, fields, reference_color_scales(truth, component)


def _render_animation(
    artifact_path: Path,
    output_path: Path,
    *,
    phase: str,
    type_arm: str,
    component: int,
    fps: int,
    max_nodes: int,
    frame_count: int,
    study_label: str,
    rotation_prediction_label: str,
) -> dict[str, Any]:
    with np.load(artifact_path, allow_pickle=False) as loaded:
        artifact = {name: np.asarray(loaded[name]) for name in loaded.files}
    titles, fields, scales = residual_animation_fields(
        artifact,
        phase=phase,
        type_arm=type_arm,
        component=component,
        rotation_prediction_label=rotation_prediction_label,
    )
    if frame_count < 1 or frame_count > fields[0].shape[0]:
        raise ValueError(f"invalid admissible animation prefix: {frame_count}")
    fields = [field[:frame_count] for field in fields]
    if any(not np.isfinite(field).all() for field in fields):
        raise ValueError("admissible animation prefix contains nonfinite values")
    nodes = np.asarray(artifact["nodes"], dtype=np.float64)
    node_count = nodes.shape[0]
    if node_count <= max_nodes:
        indices = np.arange(node_count, dtype=np.int64)
    else:
        indices = np.linspace(0, node_count - 1, max_nodes, dtype=np.int64)
    figure, axes = plt.subplots(2, 5, figsize=(18.0, 7.5), constrained_layout=True)
    residual_norm = Normalize(-scales["residual_limit"], scales["residual_limit"])
    cumulative_norm = Normalize(-scales["cumulative_limit"], scales["cumulative_limit"])
    growth_norm = Normalize(
        -scales["signed_growth_limit"], scales["signed_growth_limit"]
    )
    norms = (
        [residual_norm] * 6 + [cumulative_norm] + [residual_norm] * 2 + [growth_norm]
    )
    scatters = []
    for axis, title, field, norm in zip(axes.flat, titles, fields, norms):
        scatter = axis.scatter(
            nodes[indices, 0],
            nodes[indices, 1],
            c=field[0, indices],
            s=2.5,
            cmap="coolwarm",
            norm=norm,
            linewidths=0.0,
        )
        axis.set_title(title, fontsize=9)
        axis.set_aspect("equal")
        axis.set_xticks([])
        axis.set_yticks([])
        scatters.append(scatter)
    figure.colorbar(
        ScalarMappable(norm=residual_norm, cmap="coolwarm"),
        ax=[*axes.flat[:6], *axes.flat[7:9]],
        shrink=0.65,
        label="increment / defect (physical units)",
    )
    figure.colorbar(
        ScalarMappable(norm=cumulative_norm, cmap="coolwarm"),
        ax=axes.flat[6],
        shrink=0.65,
        label="cumulative defect (physical units)",
    )
    figure.colorbar(
        ScalarMappable(norm=growth_norm, cmap="coolwarm"),
        ax=axes.flat[9],
        shrink=0.65,
        label="signed growth (squared physical units)",
    )
    physical_times = np.asarray(artifact["physical_times"], dtype=np.float64)[1:]
    heading = figure.suptitle("")

    def update(frame: int) -> list[Any]:
        for scatter, field in zip(scatters, fields):
            scatter.set_array(field[frame, indices])
        heading.set_text(
            f"{study_label} | {phase.upper()} case {artifact['case_id'].item()} | "
            f"{type_arm} | {COMPONENT_NAMES[component]} | "
            f"call {frame + 1}, t={physical_times[frame]:.3f}"
        )
        return [*scatters, heading]

    movie = animation.FuncAnimation(
        figure, update, frames=frame_count, interval=1000.0 / fps, blit=False
    )
    movie.save(output_path, writer=animation.PillowWriter(fps=fps), dpi=105)
    plt.close(figure)
    return {
        "case_id": str(artifact["case_id"].item()),
        "phase": phase,
        "type_arm": type_arm,
        "component": component,
        "component_name": COMPONENT_NAMES[component],
        "frames": frame_count,
        "visualization_node_count": int(indices.size),
        **scales,
        "output": output_path.name,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    source_summary = _verify_input(args.input_dir)
    contract = INPUT_CONTRACTS[source_summary["schema"]]
    declared_phase = str(source_summary["phase"])
    available_phases = tuple(
        phase for phase in PHASES if declared_phase in {phase, "both"}
    )
    phases = available_phases if args.phases is None else tuple(args.phases)
    if not phases or any(phase not in available_phases for phase in phases):
        raise ValueError(
            f"requested phases {phases} are not available in {declared_phase}"
        )
    rows, metric_integrity = _read_metrics(args.input_dir / "metrics.csv")
    prefixes = _read_admissible_prefixes(args.input_dir / "completion.csv")
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "animations").mkdir()
    figure_paths = _metric_figures(
        rows, args.output_dir, phases=phases, contract=contract
    )
    scale_rows = []
    animation_paths = []
    requested_animation_count = 0
    skipped_animation_count = 0
    artifact_paths = sorted((args.input_dir / "arrays").glob("bump_*.npz"))
    if len(artifact_paths) != 3:
        raise ValueError("bump visualization requires the three frozen case bundles")
    if not args.skip_animations:
        for artifact_path in artifact_paths:
            case_id = artifact_path.stem.removeprefix("bump_")
            for phase in phases:
                for type_arm in TYPE_ARMS:
                    prefix = paired_admissible_prefix(
                        prefixes,
                        case_id=case_id,
                        phase=phase,
                        type_arm=type_arm,
                    )
                    for component in args.components:
                        requested_animation_count += 1
                        output_path = (
                            args.output_dir
                            / "animations"
                            / (
                                f"{artifact_path.stem}_{phase}_{type_arm}_"
                                f"component{component}.gif"
                            )
                        )
                        if int(prefix["accepted_calls"]) == 0:
                            skipped_animation_count += 1
                            scale_rows.append(
                                {
                                    "case_id": case_id,
                                    "phase": phase,
                                    "type_arm": type_arm,
                                    "component": component,
                                    "component_name": COMPONENT_NAMES[component],
                                    "status": "skipped_no_admissible_frames",
                                    "frames": 0,
                                    **prefix,
                                    "output": output_path.name,
                                }
                            )
                            continue
                        scale_rows.append(
                            {
                                **_render_animation(
                                    artifact_path,
                                    output_path,
                                    phase=phase,
                                    type_arm=type_arm,
                                    component=component,
                                    fps=args.fps,
                                    max_nodes=args.max_nodes,
                                    frame_count=int(prefix["accepted_calls"]),
                                    study_label=_phase_study_label(contract, phase),
                                    rotation_prediction_label=contract[
                                        "rotation_prediction_label"
                                    ],
                                ),
                                "status": "created_admissible_prefix",
                                **prefix,
                            }
                        )
                        animation_paths.append(output_path)
    else:
        scale_rows.append(
            {
                "status": "animations_skipped_by_request",
                "frames": 0,
            }
        )
    scale_path = args.output_dir / "fixed_scale_audit.csv"
    write_csv(scale_path, scale_rows)
    outputs = [*figure_paths, scale_path, *animation_paths]
    hashes = {
        path.relative_to(args.output_dir).as_posix(): sha256_file(path)
        for path in outputs
    }
    manifest_path = args.output_dir / "manifest.json"
    atomic_write_json(
        manifest_path,
        {
            "schema": OUTPUT_SCHEMA,
            "status": "complete",
            "source_summary_sha256": sha256_file(args.input_dir / "summary.json"),
            "source_artifact_manifest_sha256": source_summary[
                "artifact_manifest_sha256"
            ],
            "source_terminal_receipt_sha256": sha256_file(
                args.input_dir / "terminal_receipt.json"
            ),
            "visualizer_source_sha256": sha256_file(Path(__file__).resolve()),
            "case_ids": [path.stem.removeprefix("bump_") for path in artifact_paths],
            "phases": list(phases),
            "input_contract": contract,
            "metric_integrity": metric_integrity,
            "animation_contract": (
                "paired admissible free-rollout prefixes only; reference-only "
                "rollout-wide physical scales; no rejected proposal, per-frame, or "
                "model-outcome normalization"
            ),
            "figure_count": len(figure_paths),
            "animation_count": len(animation_paths),
            "animation_request_count": requested_animation_count,
            "animation_skipped_no_admissible_frames": skipped_animation_count,
            "files": hashes,
        },
    )
    print(
        json.dumps({"manifest": str(manifest_path), "animations": len(animation_paths)})
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
