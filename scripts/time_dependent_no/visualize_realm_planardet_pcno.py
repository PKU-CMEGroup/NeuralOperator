#!/usr/bin/env python3
"""Render verified PlanarDet PCNO rollout diagnostics and fixed-scale animations.

This is a visualization-only postprocessor for the completed D092-R1 open-
validation evaluation.  Scientific metrics are read from the full-resolution
evaluator result or recomputed on full-resolution decoded states.  Spatial
subsampling is used only for raster presentation, and every output is bound to
the retained checkpoint, evaluator result, open-data manifest, and normalizer.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import animation

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.time_dependent_no.evaluate_realm_planardet_pcno import (
    _validate_closed_training_tree,
)
from utility.time_dependent_no.realm_benchmark import (
    MagnitudeEnvelope,
    canonical_json_sha256,
    decoded_admissibility,
    decoded_boundedness,
    parse_manifest_payload,
)
from utility.time_dependent_no.realm_planardet import (
    PLANARDET_FIELDS,
    PLANARDET_GROUPS,
    PLANARDET_OPEN_MANIFEST_SHA256,
    PLANARDET_VAL_GROUPS,
    load_planardet_metadata,
    load_planardet_trajectory,
    sha256_file,
    trajectory_relative_path,
    validate_local_open_tree,
    validate_planardet_open_manifest,
)
from utility.time_dependent_no.realm_planardet_artifacts import (
    EXECUTABLE_ENTRYPOINTS,
    build_source_manifest,
)
from utility.time_dependent_no.realm_planardet_runtime import (
    BOUNDEDNESS_EXPANSION_FACTOR,
    CHANNELS,
    VALIDATION_HORIZON,
    PlanarDetNormalizerBundle,
    load_normalizer_bundle,
)

EVALUATION_SCHEMA = "w26_l4_planardet_pd0_a3_open_validation_evaluation_v1"
EVALUATION_SUMMARY_SCHEMA = "w26_l4_planardet_pd0_a3_evaluation_summary_v1"
EVALUATION_FINAL_MANIFEST_SCHEMA = (
    "w26_l4_planardet_pd0_a3_evaluation_final_hash_manifest_v1"
)
VISUALIZATION_SCHEMA = "w26_l4_planardet_pd0_a3_visualization_v3"
VISUALIZATION_FINAL_MANIFEST_SCHEMA = (
    "w26_l4_planardet_pd0_a3_visualization_final_hash_manifest_v3"
)
EVALUATION_FILES = frozenset(
    {
        "teacher_prediction_normalized.npy",
        "free_prediction_normalized.npy",
        "result.json",
        "source_manifest.json",
        "runtime_manifest.json",
        "final_hash_manifest.json",
        "summary.json",
    }
)
EVALUATION_HASHED_FILES = EVALUATION_FILES - {
    "final_hash_manifest.json",
    "summary.json",
}
DEFAULT_ANIMATION_FIELDS = ("pMax", "T", "H2O")
SNAPSHOT_CALLS = (1, 10, 25, 49)
TEACHER_COLOR = "#0072B2"
FREE_COLOR = "#D55E00"
GROUP_COLORS = {
    "chem": "#009E73",
    "T": "#CC79A7",
    "rho": "#56B4E9",
    "u": "#E69F00",
    "p": "#D55E00",
}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "figure.dpi": 150,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--normalizer-arrays", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--animation-fields",
        nargs="+",
        default=list(DEFAULT_ANIMATION_FIELDS),
    )
    parser.add_argument("--format", choices=("gif", "mp4"), default="gif")
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--dpi", type=int, default=85)
    parser.add_argument("--spatial-stride", type=int, default=2)
    return parser


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read JSON input: {path.name}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON input: {path.name}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise TypeError(f"JSON root must be an object: {path.name}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise ValueError(f"visualization output already exists: {path.name}")
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if path.exists() or path.is_symlink():
        raise ValueError(f"visualization output already exists: {path.name}")
    if not rows:
        raise ValueError("CSV output requires at least one row")
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        raise ValueError("CSV rows must have an identical ordered schema")
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _is_within(child: Path, parent: Path) -> bool:
    child_resolved = child.resolve(strict=False)
    parent_resolved = parent.resolve(strict=False)
    return (
        child_resolved == parent_resolved or parent_resolved in child_resolved.parents
    )


def _prepare_output_directory(
    output_dir: Path,
    *,
    evaluation_dir: Path,
    training_dir: Path,
    data_root: Path,
) -> None:
    if output_dir.exists() or output_dir.is_symlink():
        raise ValueError("visualization output directory must be absent")
    protected = (evaluation_dir, training_dir, data_root)
    if any(
        _is_within(output_dir, root) or _is_within(root, output_dir)
        for root in protected
    ):
        raise ValueError("visualization output must be disjoint from all inputs")
    output_dir.mkdir(parents=True)


def _verify_final_manifest(
    directory: Path,
    payload: Mapping[str, Any],
    *,
    schema: str,
    expected_files: frozenset[str],
) -> dict[str, str]:
    files = payload.get("files")
    if (
        payload.get("schema") != schema
        or payload.get("self_hash_excluded") is not True
        or payload.get("test_object_opened") is not False
        or not isinstance(files, Mapping)
        or set(files) != expected_files
    ):
        raise ValueError("final hash manifest schema or inventory differs")
    verified: dict[str, str] = {}
    for name, expected in files.items():
        if not isinstance(name, str) or not isinstance(expected, str):
            raise TypeError("final hash manifest entries must be string digests")
        actual = sha256_file(directory / name)
        if actual != expected:
            raise ValueError(f"visualization input hash differs: {name}")
        verified[name] = actual
    return verified


def _validate_evaluation_directory(
    evaluation_dir: Path,
    *,
    current_source_manifest: Mapping[str, Any],
    training_dir: Path,
) -> tuple[Mapping[str, Any], Mapping[str, Any], dict[str, str]]:
    if evaluation_dir.is_symlink() or not evaluation_dir.is_dir():
        raise ValueError(
            "evaluation directory must be an existing non-symlink directory"
        )
    actual: set[str] = set()
    for path in evaluation_dir.iterdir():
        if path.is_symlink() or not path.is_file():
            raise ValueError("evaluation directory contains a non-regular object")
        actual.add(path.name)
    if actual != EVALUATION_FILES:
        raise ValueError("evaluation directory inventory differs")

    manifest = _load_json(evaluation_dir / "final_hash_manifest.json")
    verified = _verify_final_manifest(
        evaluation_dir,
        manifest,
        schema=EVALUATION_FINAL_MANIFEST_SCHEMA,
        expected_files=EVALUATION_HASHED_FILES,
    )
    result = _load_json(evaluation_dir / "result.json")
    unsigned = {
        key: value for key, value in result.items() if key != "canonical_payload_sha256"
    }
    if (
        result.get("schema") != EVALUATION_SCHEMA
        or result.get("canonical_payload_sha256") != canonical_json_sha256(unsigned)
        or result.get("test_object_opened") is not False
        or result.get("evaluator_closure", {}).get("all_gates_pass") is not True
        or result.get("evaluator_closure", {}).get("test_object_opened") is not False
    ):
        raise ValueError("evaluation result closure or canonical digest differs")
    summary = _load_json(evaluation_dir / "summary.json")
    if (
        summary.get("schema") != EVALUATION_SUMMARY_SCHEMA
        or summary.get("result_sha256") != verified["result.json"]
        or summary.get("final_hash_manifest_sha256")
        != sha256_file(evaluation_dir / "final_hash_manifest.json")
        or summary.get("test_object_opened") is not False
    ):
        raise ValueError("evaluation summary differs from the verified result")
    source = _load_json(evaluation_dir / "source_manifest.json")
    if source != current_source_manifest:
        raise ValueError("current scientific sources differ from the evaluation")
    checkpoint = result.get("checkpoint")
    if (
        not isinstance(checkpoint, Mapping)
        or checkpoint.get("relative_path") != "best.pt"
        or checkpoint.get("sha256") != sha256_file(training_dir / "best.pt")
    ):
        raise ValueError("evaluation checkpoint binding differs")
    return result, summary, verified


def _validate_fields(values: Sequence[str]) -> tuple[str, ...]:
    fields = tuple(values)
    if not fields or len(set(fields)) != len(fields):
        raise ValueError("animation fields must be a nonempty unique sequence")
    unknown = set(fields) - set(PLANARDET_FIELDS)
    if unknown:
        raise ValueError(f"unknown PlanarDet animation fields: {sorted(unknown)}")
    return fields


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if not (math.isfinite(numerator) and math.isfinite(denominator)):
        return None
    if denominator == 0.0:
        return 0.0 if numerator == 0.0 else None
    value = numerator / denominator
    return value if math.isfinite(value) else None


def _nullable(value: Any) -> float:
    return math.nan if value is None else float(value)


def _error_rows(result: Mapping[str, Any], times: np.ndarray) -> list[dict[str, Any]]:
    teacher = result["views"]["ordered_teacher_forced"]["summary"]
    free = result["views"]["free_recurrence"]["summary"]
    rows: list[dict[str, Any]] = []
    for index in range(VALIDATION_HORIZON):
        teacher_npe = float(teacher["npe_total_by_call"][index])
        free_npe = float(free["npe_total_by_call"][index])
        row: dict[str, Any] = {
            "call": index + 1,
            "physical_time_s": float(times[index + 1]),
            "teacher_npe": teacher_npe,
            "free_npe": free_npe,
            "free_minus_teacher_npe": free_npe - teacher_npe,
            "free_over_teacher_npe": _safe_ratio(free_npe, teacher_npe),
            "teacher_decoded_correlation": float(
                teacher["decoded_correlation_by_call"][index]
            ),
            "free_decoded_correlation": float(
                free["decoded_correlation_by_call"][index]
            ),
            "teacher_pMax_correlation": float(
                teacher["pMax_decoded_correlation_by_call"][index]
            ),
            "free_pMax_correlation": float(
                free["pMax_decoded_correlation_by_call"][index]
            ),
        }
        for group in ("chem", "T", "rho", "u", "p"):
            row[f"teacher_{group}_npe"] = float(
                teacher["npe_group_by_call"][group][index]
            )
            row[f"free_{group}_npe"] = float(free["npe_group_by_call"][group][index])
        rows.append(row)
    return rows


def _event_record(
    state: torch.Tensor,
    current_pmax: torch.Tensor,
    *,
    envelope: MagnitudeEnvelope,
) -> dict[str, Any]:
    admissibility = decoded_admissibility(
        state,
        groups=PLANARDET_GROUPS,
        channel_axis=0,
    )
    boundedness = decoded_boundedness(
        state,
        envelope,
        expansion_factor=BOUNDEDNESS_EXPANSION_FACTOR,
        channel_axis=0,
    )
    increment = state[12] - current_pmax
    finite_increment = torch.isfinite(increment)
    decreases = finite_increment & (increment < 0.0)
    maximum_decrease = (
        float((-increment[decreases]).max().item()) if bool(decreases.any()) else 0.0
    )
    violations = tuple(
        "nonpositive_pMax" if value == "nonpositive_pressure" else value
        for value in admissibility.violations
    )
    return {
        "normalized_finite": True,
        "decoded_finite": admissibility.finite,
        "admissible": admissibility.admissible,
        "violations": violations,
        "min_species": admissibility.min_species,
        "min_temperature": admissibility.min_temperature,
        "min_density": admissibility.min_density,
        "min_pMax": admissibility.min_pressure,
        "bounded_10x_train_max": boundedness.bounded,
        "max_boundedness_ratio": max(boundedness.max_ratio_by_channel),
        "max_boundedness_ratio_by_channel": boundedness.max_ratio_by_channel,
        "pMax_nondecreasing": bool(finite_increment.all())
        and not bool(decreases.any()),
        "pMax_decrease_count": int(decreases.sum().item()),
        "pMax_max_decrease_pa": maximum_decrease,
    }


def _decode_prediction(
    values: np.ndarray,
    *,
    normalizer: Any,
) -> torch.Tensor:
    normalized = torch.from_numpy(np.array(values, dtype=np.float32, copy=True))
    if not bool(torch.isfinite(normalized).all()):
        return torch.full_like(normalized, torch.nan)
    return normalizer.decode(normalized, inverse_domain_policy="nan")


def _display_multiplier(field: str) -> tuple[float, str]:
    if field == "pMax":
        return 1.0e-6, "pMax [MPa]"
    if field == "T":
        return 1.0, "T [released units]"
    return 1.0, f"{field} [released units]"


def _decode_visualization_bundle(
    teacher_normalized: np.ndarray,
    free_normalized: np.ndarray,
    truth_native: np.ndarray,
    *,
    bundle: PlanarDetNormalizerBundle,
    fields: Sequence[str],
    spatial_stride: int,
) -> tuple[
    dict[str, dict[str, np.ndarray]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    if spatial_stride < 1:
        raise ValueError("spatial stride must be positive")
    expected = (VALIDATION_HORIZON, *truth_native.shape[1:])
    if teacher_normalized.shape != expected or free_normalized.shape != expected:
        raise ValueError("prediction array shape differs from the validation truth")
    if teacher_normalized.dtype != np.dtype(
        "float32"
    ) or free_normalized.dtype != np.dtype("float32"):
        raise ValueError("prediction arrays must be float32")

    indices = {field: PLANARDET_FIELDS.index(field) for field in fields}
    sampled_shape = truth_native.shape[-2::]
    sampled_shape = (
        len(range(0, sampled_shape[0], spatial_stride)),
        len(range(0, sampled_shape[1], spatial_stride)),
    )
    visual: dict[str, dict[str, np.ndarray]] = {}
    for field, index in indices.items():
        truth_values = np.asarray(
            truth_native[:, index, ::spatial_stride, ::spatial_stride],
            dtype=np.float32,
        )
        visual[field] = {
            "truth": truth_values.copy(),
            "teacher": np.empty((truth_native.shape[0], *sampled_shape), np.float32),
            "free": np.empty((truth_native.shape[0], *sampled_shape), np.float32),
        }
        visual[field]["teacher"][0] = truth_values[0]
        visual[field]["free"][0] = truth_values[0]

    normalizer = bundle.normalizer(0, dtype=torch.float32)
    envelope = MagnitudeEnvelope(
        max_abs=bundle.train_max_abs,
        quantile=1.0,
        channel_axis=0,
    )
    teacher_events: list[dict[str, Any]] = []
    free_events: list[dict[str, Any]] = []
    previous_free_pmax = torch.from_numpy(truth_native[0, 12].copy())
    for index in range(VALIDATION_HORIZON):
        teacher_state = _decode_prediction(
            teacher_normalized[index], normalizer=normalizer
        )
        free_state = _decode_prediction(free_normalized[index], normalizer=normalizer)
        teacher_events.append(
            _event_record(
                teacher_state,
                torch.from_numpy(truth_native[index, 12].copy()),
                envelope=envelope,
            )
        )
        free_events.append(
            _event_record(free_state, previous_free_pmax, envelope=envelope)
        )
        previous_free_pmax = free_state[12].detach().clone()
        teacher_numpy = teacher_state.numpy()
        free_numpy = free_state.numpy()
        for field, channel in indices.items():
            visual[field]["teacher"][index + 1] = teacher_numpy[
                channel, ::spatial_stride, ::spatial_stride
            ]
            visual[field]["free"][index + 1] = free_numpy[
                channel, ::spatial_stride, ::spatial_stride
            ]

    scales: dict[str, Any] = {}
    for field, arrays in visual.items():
        multiplier, label = _display_multiplier(field)
        for key in arrays:
            arrays[key] *= multiplier
        full_truth = (
            np.asarray(truth_native[:, indices[field]], dtype=np.float64) * multiplier
        )
        state_min = float(np.nanmin(full_truth))
        state_max = float(np.nanmax(full_truth))
        if state_max <= state_min:
            state_max = state_min + max(abs(state_min), 1.0) * 1.0e-6
        teacher_error = np.abs(arrays["teacher"][1:] - arrays["truth"][1:])
        free_error = np.abs(arrays["free"][1:] - arrays["truth"][1:])
        combined = np.concatenate((teacher_error.reshape(-1), free_error.reshape(-1)))
        finite = combined[np.isfinite(combined)]
        error_max = float(np.quantile(finite, 0.995)) if finite.size else 1.0
        if error_max <= 0.0:
            error_max = max(float(np.nanmax(finite)) if finite.size else 0.0, 1.0e-12)
        scales[field] = {
            "label": label,
            "display_multiplier": multiplier,
            "state_min": state_min,
            "state_max": state_max,
            "error_abs_q99_5": error_max,
            "error_scale_source": (
                "99.5th percentile of teacher/free absolute error on the "
                "visualization-only spatially subsampled grid"
            ),
            "teacher_state_saturation_fraction": float(
                np.mean(
                    (arrays["teacher"] < state_min) | (arrays["teacher"] > state_max)
                )
            ),
            "free_state_saturation_fraction": float(
                np.mean((arrays["free"] < state_min) | (arrays["free"] > state_max))
            ),
            "teacher_error_saturation_fraction": float(
                np.mean(teacher_error > error_max)
            ),
            "free_error_saturation_fraction": float(np.mean(free_error > error_max)),
        }
    return visual, teacher_events, free_events, scales


def _event_rows(
    teacher_events: Sequence[Mapping[str, Any]],
    free_events: Sequence[Mapping[str, Any]],
    times: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for view, events in (
        ("teacher", teacher_events),
        ("free", free_events),
    ):
        for index, event in enumerate(events):
            rows.append(
                {
                    "view": view,
                    "call": index + 1,
                    "physical_time_s": float(times[index + 1]),
                    "normalized_finite": event["normalized_finite"],
                    "decoded_finite": event["decoded_finite"],
                    "admissible": event["admissible"],
                    "violations": ";".join(event["violations"]),
                    "bounded_10x_train_max": event["bounded_10x_train_max"],
                    "pMax_nondecreasing": event["pMax_nondecreasing"],
                    "pMax_decrease_count": event["pMax_decrease_count"],
                    "pMax_max_decrease_pa": event["pMax_max_decrease_pa"],
                    "min_species": event["min_species"],
                    "min_temperature": event["min_temperature"],
                    "min_density": event["min_density"],
                    "min_pMax": event["min_pMax"],
                    "max_boundedness_ratio": event["max_boundedness_ratio"],
                }
            )
    return rows


def _first_event(
    events: Sequence[Mapping[str, Any]],
    key: str,
    *,
    expected: bool,
) -> int | None:
    for index, event in enumerate(events):
        if bool(event[key]) is expected:
            return index + 1
    return None


def _save_figure_pair(fig: Any, stem: Path) -> list[Path]:
    paths = [stem.with_suffix(".pdf"), stem.with_suffix(".png")]
    for path in paths:
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return paths


def _panel_label(axis: Any, label: str) -> None:
    axis.text(
        -0.12,
        1.04,
        label,
        transform=axis.transAxes,
        fontweight="bold",
        va="bottom",
    )


def render_error_accumulation(
    result: Mapping[str, Any],
    output_dir: Path,
) -> list[Path]:
    teacher = result["views"]["ordered_teacher_forced"]["summary"]
    free = result["views"]["free_recurrence"]["summary"]
    controls = result["controls"]
    calls = np.arange(1, VALIDATION_HORIZON + 1)
    teacher_npe = np.asarray(teacher["npe_total_by_call"], dtype=np.float64)
    free_npe = np.asarray(free["npe_total_by_call"], dtype=np.float64)
    ratio = np.divide(
        free_npe,
        teacher_npe,
        out=np.full_like(free_npe, np.nan),
        where=teacher_npe != 0.0,
    )

    fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.8), constrained_layout=True)
    axis = axes[0, 0]
    axis.semilogy(calls, teacher_npe, color=TEACHER_COLOR, label="Truth input")
    axis.semilogy(calls, free_npe, color=FREE_COLOR, label="Free recurrence")
    axis.axhline(
        controls["persistence"]["realm_npe_mean"],
        color="0.35",
        linestyle="--",
        linewidth=1.0,
        label="Persistence mean",
    )
    axis.axhline(
        controls["linear_normalized"]["realm_npe_mean"],
        color="0.55",
        linestyle=":",
        linewidth=1.0,
        label="Linear mean",
    )
    axis.set(xlabel="Call / time step", ylabel="Grouped normalized prediction error")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, ncol=2)
    _panel_label(axis, "a")

    axis = axes[0, 1]
    axis.plot(
        calls,
        teacher["decoded_correlation_by_call"],
        color=TEACHER_COLOR,
        label="Truth input",
    )
    axis.plot(
        calls,
        free["decoded_correlation_by_call"],
        color=FREE_COLOR,
        label="Free recurrence",
    )
    axis.set(
        xlabel="Call / time step",
        ylabel="Decoded spatial correlation",
        ylim=(-0.05, 1.02),
    )
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    _panel_label(axis, "b")

    axis = axes[1, 0]
    for group in ("chem", "T", "rho", "u", "p"):
        color = GROUP_COLORS[group]
        axis.semilogy(
            calls,
            free["npe_group_by_call"][group],
            color=color,
            label=f"{group} free",
        )
        axis.semilogy(
            calls,
            teacher["npe_group_by_call"][group],
            color=color,
            linestyle="--",
            linewidth=1.0,
            alpha=0.7,
        )
    axis.set(xlabel="Call / time step", ylabel="Group contribution to normalized error")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, ncol=2)
    _panel_label(axis, "c")

    axis = axes[1, 1]
    axis.semilogy(calls, ratio, color="#7A3E9D", label="Free / truth-input NPE")
    axis.axhline(1.0, color="0.4", linewidth=1.0, linestyle="--")
    axis.set(xlabel="Call / time step", ylabel="Error ratio")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    _panel_label(axis, "d")
    return _save_figure_pair(fig, output_dir / "error_accumulation")


def render_channel_heatmap(
    result: Mapping[str, Any],
    output_dir: Path,
) -> list[Path]:
    teacher = np.asarray(
        result["views"]["ordered_teacher_forced"]["summary"][
            "normalized_mse_by_call_channel"
        ],
        dtype=np.float64,
    ).T
    free = np.asarray(
        result["views"]["free_recurrence"]["summary"]["normalized_mse_by_call_channel"],
        dtype=np.float64,
    ).T
    floor = 1.0e-12
    logs = (np.log10(np.maximum(teacher, floor)), np.log10(np.maximum(free, floor)))
    lower = float(min(np.nanmin(logs[0]), np.nanmin(logs[1])))
    upper = float(max(np.nanmax(logs[0]), np.nanmax(logs[1])))
    ratio = np.log10(np.maximum(free, floor) / np.maximum(teacher, floor))

    fig, axes = plt.subplots(3, 1, figsize=(9.4, 7.6), constrained_layout=True)
    for axis, values, label in zip(
        axes[:2],
        logs,
        ("Truth-input log10 normalized MSE", "Free-recurrence log10 normalized MSE"),
        strict=True,
    ):
        image = axis.imshow(
            values,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            extent=(0.5, VALIDATION_HORIZON + 0.5, -0.5, CHANNELS - 0.5),
            cmap="viridis",
            vmin=lower,
            vmax=upper,
        )
        axis.set_yticks(range(CHANNELS), PLANARDET_FIELDS)
        axis.set_ylabel("Released field")
        axis.set_title(label, loc="left")
        fig.colorbar(image, ax=axis, shrink=0.82)
    image = axes[2].imshow(
        ratio,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=(0.5, VALIDATION_HORIZON + 0.5, -0.5, CHANNELS - 0.5),
        cmap="coolwarm",
        vmin=-max(abs(float(np.nanmin(ratio))), abs(float(np.nanmax(ratio)))),
        vmax=max(abs(float(np.nanmin(ratio))), abs(float(np.nanmax(ratio)))),
    )
    axes[2].set_yticks(range(CHANNELS), PLANARDET_FIELDS)
    axes[2].set(xlabel="Call / time step", ylabel="Released field")
    axes[2].set_title("log10 free / truth-input normalized MSE", loc="left")
    fig.colorbar(image, ax=axes[2], shrink=0.82)
    return _save_figure_pair(fig, output_dir / "channel_error_heatmap")


def _series_difference(
    structure: Mapping[str, Any],
    key: str,
) -> np.ndarray:
    prediction = np.asarray(
        [_nullable(value) for value in structure["prediction"][key]],
        dtype=np.float64,
    )
    truth = np.asarray(
        [_nullable(value) for value in structure["truth"][key]],
        dtype=np.float64,
    )
    return prediction - truth


def render_structure_evolution(
    result: Mapping[str, Any],
    output_dir: Path,
) -> list[Path]:
    teacher = result["structure"]["ordered_teacher_forced"]
    free = result["structure"]["free_recurrence"]
    calls = np.arange(VALIDATION_HORIZON + 1)
    panels = (
        ("pMax_relative_l2_per_frame", "pMax relative L2"),
        ("front_position_error_mm_per_frame", "Front-position error [mm]"),
        ("cell_size_proxy_error_mm_per_frame", "Cell-size proxy error [mm]"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.8), constrained_layout=True)
    for axis, (key, label), panel in zip(
        axes.flat[:3], panels, ("a", "b", "c"), strict=True
    ):
        axis.plot(
            calls,
            [_nullable(v) for v in teacher[key]],
            color=TEACHER_COLOR,
            label="Truth input",
        )
        axis.plot(
            calls,
            [_nullable(v) for v in free[key]],
            color=FREE_COLOR,
            label="Free recurrence",
        )
        axis.axhline(0.0, color="0.5", linewidth=0.8)
        axis.set(xlabel="Call / time step", ylabel=label)
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
        _panel_label(axis, panel)
    axis = axes.flat[3]
    axis.plot(
        calls,
        _series_difference(teacher, "shock_attached_highpass_fraction"),
        color=TEACHER_COLOR,
        label="Truth input",
    )
    axis.plot(
        calls,
        _series_difference(free, "shock_attached_highpass_fraction"),
        color=FREE_COLOR,
        label="Free recurrence",
    )
    axis.axhline(0.0, color="0.5", linewidth=0.8)
    axis.set(
        xlabel="Call / time step", ylabel="Shock-attached high-pass fraction error"
    )
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    _panel_label(axis, "d")
    return _save_figure_pair(fig, output_dir / "structure_evolution")


def render_training_history(history: Mapping[str, Any], output_dir: Path) -> list[Path]:
    rows = history.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("training history lacks rows")
    steps = np.asarray([row["completed_step"] for row in rows], dtype=np.int64)
    losses = np.asarray(
        [row["interval_mean_train_grouped_loss"] for row in rows], dtype=np.float64
    )
    npe = np.asarray(
        [row["validation"]["realm_npe_mean"] for row in rows], dtype=np.float64
    )
    correlation = np.asarray(
        [row["validation"]["decoded_correlation_case_first"] for row in rows],
        dtype=np.float64,
    )
    admissible = np.asarray(
        [row["validation"]["admissible_call_count"] for row in rows],
        dtype=np.float64,
    )
    best_index = int(np.argmin(npe))

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.6), constrained_layout=True)
    axis = axes[0]
    axis.semilogy(steps, losses, color="#009E73", label="Interval train loss")
    axis.semilogy(
        steps,
        npe,
        color=FREE_COLOR,
        label="Validation loss (truth-input NPE)",
    )
    axis.axvline(
        490,
        color="0.65",
        linestyle=":",
        linewidth=1.0,
        label="Two-call phase",
    )
    axis.axvline(
        steps[best_index],
        color="0.3",
        linestyle="--",
        linewidth=1.0,
        label="Selected step",
    )
    axis.set(xlabel="Optimizer step", ylabel="Grouped normalized loss")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    _panel_label(axis, "a")

    axis = axes[1]
    axis.plot(steps, correlation, color=TEACHER_COLOR, label="Decoded correlation")
    axis.plot(
        steps,
        admissible / VALIDATION_HORIZON,
        color="#CC79A7",
        label="Admissible-call fraction",
    )
    axis.axvline(steps[best_index], color="0.3", linestyle="--", linewidth=1.0)
    axis.set(
        xlabel="Optimizer step", ylabel="Fraction / correlation", ylim=(-0.02, 1.02)
    )
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    _panel_label(axis, "b")
    return _save_figure_pair(fig, output_dir / "training_validation_trajectory")


def _loss_rows(history: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = history.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("training history lacks rows")
    result: list[dict[str, Any]] = []
    for row in rows:
        train_loss = float(row["interval_mean_train_grouped_loss"])
        validation_loss = float(row["validation"]["realm_npe_mean"])
        result.append(
            {
                "optimizer_step": int(row["completed_step"]),
                "phase": str(row["phase"]),
                "calls": int(row["calls"]),
                "interval_train_grouped_loss": train_loss,
                "truth_input_validation_grouped_npe": validation_loss,
                "validation_minus_train": validation_loss - train_loss,
                "validation_over_train": (
                    validation_loss / train_loss if train_loss > 0.0 else None
                ),
                "learning_rate_used": float(row["learning_rate_used"]),
                "decoded_validation_correlation": float(
                    row["validation"]["decoded_correlation_case_first"]
                ),
                "strict_validation_improvement": bool(row["strict_improvement"]),
            }
        )
    return result


def _oriented(values: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    result = values
    if y[0] > y[-1]:
        result = result[..., ::-1, :]
    if x[0] > x[-1]:
        result = result[..., :, ::-1]
    return result


def render_snapshots(
    field: str,
    arrays: Mapping[str, np.ndarray],
    scale: Mapping[str, Any],
    *,
    x: np.ndarray,
    y: np.ndarray,
    output_dir: Path,
) -> list[Path]:
    truth = _oriented(arrays["truth"], x, y)
    teacher = _oriented(arrays["teacher"], x, y)
    free = _oriented(arrays["free"], x, y)
    extent = (
        float(np.min(x)) * 1.0e3,
        float(np.max(x)) * 1.0e3,
        float(np.min(y)) * 1.0e3,
        float(np.max(y)) * 1.0e3,
    )
    fig, axes = plt.subplots(
        len(SNAPSHOT_CALLS), 5, figsize=(12.4, 8.8), constrained_layout=True
    )
    state_image = None
    error_image = None
    for row, call in enumerate(SNAPSHOT_CALLS):
        error_teacher = np.abs(teacher[call] - truth[call])
        error_free = np.abs(free[call] - truth[call])
        values = (truth[call], teacher[call], free[call], error_teacher, error_free)
        for column, value in enumerate(values):
            is_error = column >= 3
            image = axes[row, column].imshow(
                value,
                origin="lower",
                extent=extent,
                aspect="auto",
                interpolation="nearest",
                cmap="magma" if is_error else "viridis",
                vmin=0.0 if is_error else scale["state_min"],
                vmax=scale["error_abs_q99_5"] if is_error else scale["state_max"],
            )
            if is_error:
                error_image = image
            else:
                state_image = image
            if row == 0:
                axes[row, column].set_title(
                    (
                        "Truth",
                        "Truth input",
                        "Free",
                        "Truth-input |error|",
                        "Free |error|",
                    )[column]
                )
            if column == 0:
                axes[row, column].set_ylabel(f"Call {call}\ny [mm]")
            else:
                axes[row, column].set_yticklabels([])
            if row == len(SNAPSHOT_CALLS) - 1:
                axes[row, column].set_xlabel("x [mm]")
            else:
                axes[row, column].set_xticklabels([])
    if state_image is not None:
        fig.colorbar(state_image, ax=axes[:, :3], shrink=0.78, label=scale["label"])
    if error_image is not None:
        fig.colorbar(
            error_image,
            ax=axes[:, 3:],
            shrink=0.78,
            label=f"Absolute error: {scale['label']}",
        )
    return _save_figure_pair(fig, output_dir / f"rollout_snapshots_{field}")


def render_animation(
    field: str,
    arrays: Mapping[str, np.ndarray],
    scale: Mapping[str, Any],
    result: Mapping[str, Any],
    teacher_events: Sequence[Mapping[str, Any]],
    free_events: Sequence[Mapping[str, Any]],
    times: np.ndarray,
    *,
    x: np.ndarray,
    y: np.ndarray,
    output_path: Path,
    fps: int,
    dpi: int,
) -> dict[str, Any]:
    truth = _oriented(arrays["truth"], x, y)
    teacher = _oriented(arrays["teacher"], x, y)
    free = _oriented(arrays["free"], x, y)
    extent = (
        float(np.min(x)) * 1.0e3,
        float(np.max(x)) * 1.0e3,
        float(np.min(y)) * 1.0e3,
        float(np.max(y)) * 1.0e3,
    )
    fig, axes = plt.subplots(2, 3, figsize=(10.8, 6.2), constrained_layout=True)
    images: list[Any] = []
    for axis, title in zip(
        axes.flat[:5],
        (
            "Truth",
            "Truth input",
            "Free recurrence",
            "Truth-input |error|",
            "Free |error|",
        ),
        strict=True,
    ):
        is_error = len(images) >= 3
        image = axis.imshow(
            np.zeros_like(truth[0]),
            origin="lower",
            extent=extent,
            aspect="auto",
            interpolation="nearest",
            cmap="magma" if is_error else "viridis",
            vmin=0.0 if is_error else scale["state_min"],
            vmax=scale["error_abs_q99_5"] if is_error else scale["state_max"],
        )
        axis.set_title(title)
        axis.set_xlabel("x [mm]")
        axis.set_ylabel("y [mm]")
        fig.colorbar(image, ax=axis, shrink=0.72)
        images.append(image)

    curve_axis = axes.flat[5]
    calls = np.arange(1, VALIDATION_HORIZON + 1)
    teacher_npe = np.asarray(
        result["views"]["ordered_teacher_forced"]["summary"]["npe_total_by_call"],
        dtype=np.float64,
    )
    free_npe = np.asarray(
        result["views"]["free_recurrence"]["summary"]["npe_total_by_call"],
        dtype=np.float64,
    )
    curve_axis.semilogy(calls, teacher_npe, color=TEACHER_COLOR, alpha=0.25)
    curve_axis.semilogy(calls, free_npe, color=FREE_COLOR, alpha=0.25)
    (teacher_line,) = curve_axis.semilogy(
        [], [], color=TEACHER_COLOR, label="Truth input"
    )
    (free_line,) = curve_axis.semilogy(
        [], [], color=FREE_COLOR, label="Free recurrence"
    )
    cursor = curve_axis.axvline(0, color="0.25", linestyle="--", linewidth=1.0)
    curve_axis.set(
        xlim=(1, VALIDATION_HORIZON),
        xlabel="Call / time step",
        ylabel="Normalized error",
    )
    curve_axis.grid(alpha=0.2)
    curve_axis.legend(frameon=False)
    status = fig.text(0.5, 0.005, "", ha="center", va="bottom", fontsize=8.5)

    def update(frame: int) -> list[Any]:
        values = (
            truth[frame],
            teacher[frame],
            free[frame],
            np.abs(teacher[frame] - truth[frame]),
            np.abs(free[frame] - truth[frame]),
        )
        artists: list[Any] = []
        for image, value in zip(images, values, strict=True):
            image.set_data(np.ma.masked_invalid(value))
            artists.append(image)
        if frame == 0:
            teacher_line.set_data([], [])
            free_line.set_data([], [])
            cursor.set_xdata([1, 1])
            event_text = "released initial state"
        else:
            teacher_line.set_data(calls[:frame], teacher_npe[:frame])
            free_line.set_data(calls[:frame], free_npe[:frame])
            cursor.set_xdata([frame, frame])
            teacher_event = teacher_events[frame - 1]
            free_event = free_events[frame - 1]
            event_text = (
                f"truth-input admissible/pMax-monotone="
                f"{int(teacher_event['admissible'])}/{int(teacher_event['pMax_nondecreasing'])}; "
                f"free={int(free_event['admissible'])}/{int(free_event['pMax_nondecreasing'])}"
            )
        status.set_text(
            f"open validation | call {frame}/{VALIDATION_HORIZON} | "
            f"t={times[frame]:.9g} s | {scale['label']} | {event_text}"
        )
        artists.extend((teacher_line, free_line, cursor, status))
        return artists

    if output_path.suffix.lower() == ".gif":
        writer: Any = animation.PillowWriter(fps=fps)
        encoder = {"writer": "matplotlib.PillowWriter", "fps": fps}
    elif output_path.suffix.lower() == ".mp4":
        if not animation.writers.is_available("ffmpeg"):
            raise RuntimeError("ffmpeg is required for MP4 output")
        writer = animation.FFMpegWriter(
            fps=fps,
            codec="libx264",
            bitrate=4500,
            extra_args=["-pix_fmt", "yuv420p"],
        )
        encoder = {
            "writer": "matplotlib.FFMpegWriter",
            "codec": "libx264",
            "pixel_format": "yuv420p",
            "fps": fps,
        }
    else:
        raise ValueError("animation output must end in .gif or .mp4")
    movie = animation.FuncAnimation(
        fig,
        update,
        frames=VALIDATION_HORIZON + 1,
        interval=1000 / fps,
        blit=False,
    )
    movie.save(
        output_path,
        writer=writer,
        dpi=dpi,
        savefig_kwargs={"facecolor": "white", "transparent": False},
    )
    plt.close(fig)
    return {
        "field": field,
        "path": output_path.name,
        "sha256": sha256_file(output_path),
        "size": output_path.stat().st_size,
        "rendered_frame_count": VALIDATION_HORIZON + 1,
        "all_released_frames_rendered": True,
        "fixed_scale": dict(scale),
        "encoder": encoder,
    }


def _one_step_residual_views(
    arrays: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return decoded truth-input residuals and one-step prediction error."""

    truth = np.asarray(arrays["truth"])
    prediction = np.asarray(arrays["teacher"])
    if truth.shape != prediction.shape:
        raise ValueError("truth and truth-input prediction arrays must match")
    if truth.ndim != 3 or truth.shape[0] < 2:
        raise ValueError(
            "one-step residual arrays must contain the initial state and every "
            "released adjacent pair"
        )
    true_residual = truth[1:] - truth[:-1]
    predicted_residual = prediction[1:] - truth[:-1]
    prediction_error = prediction[1:] - truth[1:]
    return true_residual, predicted_residual, prediction_error


def _fixed_symmetric_limit(*values: np.ndarray) -> float:
    finite_parts = [
        np.abs(np.asarray(value)[np.isfinite(value)]).reshape(-1) for value in values
    ]
    finite_parts = [part for part in finite_parts if part.size]
    if not finite_parts:
        raise RuntimeError("one-step residual animation has no finite values")
    finite = np.concatenate(finite_parts)
    limit = float(np.quantile(finite, 0.995))
    if not math.isfinite(limit) or limit <= 0.0:
        limit = max(float(np.max(finite)), 1.0e-12)
    return limit


def _one_step_residual_npe(
    true_residual: np.ndarray,
    prediction_error: np.ndarray,
) -> np.ndarray:
    values: list[float] = []
    for truth, error in zip(true_residual, prediction_error, strict=True):
        finite = np.isfinite(truth) & np.isfinite(error)
        denominator = float(np.sum(np.square(truth[finite]), dtype=np.float64))
        numerator = float(np.sum(np.square(error[finite]), dtype=np.float64))
        values.append(
            numerator / denominator if finite.any() and denominator > 0.0 else math.nan
        )
    return np.asarray(values, dtype=np.float64)


def render_one_step_residual_animation(
    field: str,
    arrays: Mapping[str, np.ndarray],
    scale: Mapping[str, Any],
    times: np.ndarray,
    *,
    x: np.ndarray,
    y: np.ndarray,
    output_path: Path,
    fps: int,
    dpi: int,
) -> dict[str, Any]:
    """Render decoded truth-input one-step residuals and prediction error."""

    true_residual, predicted_residual, prediction_error = _one_step_residual_views(
        arrays
    )
    if true_residual.shape[0] != VALIDATION_HORIZON:
        raise ValueError(
            "one-step residual frame count differs from validation horizon"
        )
    if np.asarray(times).shape != (VALIDATION_HORIZON + 1,):
        raise ValueError("time coordinates differ from released rollout frames")
    true_residual = _oriented(true_residual, x, y)
    predicted_residual = _oriented(predicted_residual, x, y)
    prediction_error = _oriented(prediction_error, x, y)
    extent = (
        float(np.min(x)) * 1.0e3,
        float(np.max(x)) * 1.0e3,
        float(np.min(y)) * 1.0e3,
        float(np.max(y)) * 1.0e3,
    )
    residual_limit = _fixed_symmetric_limit(true_residual, predicted_residual)
    error_limit = _fixed_symmetric_limit(prediction_error)
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.5), constrained_layout=True)
    true_image = axes[0].imshow(
        np.zeros_like(true_residual[0]),
        origin="lower",
        extent=extent,
        aspect="auto",
        interpolation="nearest",
        cmap="RdBu_r",
        vmin=-residual_limit,
        vmax=residual_limit,
    )
    predicted_image = axes[1].imshow(
        np.zeros_like(true_residual[0]),
        origin="lower",
        extent=extent,
        aspect="auto",
        interpolation="nearest",
        cmap="RdBu_r",
        vmin=-residual_limit,
        vmax=residual_limit,
    )
    error_image = axes[2].imshow(
        np.zeros_like(true_residual[0]),
        origin="lower",
        extent=extent,
        aspect="auto",
        interpolation="nearest",
        cmap="RdBu_r",
        vmin=-error_limit,
        vmax=error_limit,
    )
    for axis, title in zip(
        axes,
        (
            "True one-step residual",
            "Predicted one-step residual",
            "One-step prediction error",
        ),
        strict=True,
    ):
        axis.set_title(title)
        axis.set_xlabel("x [mm]")
        axis.set_ylabel("y [mm]")
    fig.colorbar(
        true_image,
        ax=axes[:2],
        shrink=0.82,
        label=f"One-step residual: {scale['label']}",
    )
    fig.colorbar(
        error_image,
        ax=axes[2],
        shrink=0.82,
        label=f"One-step prediction error: {scale['label']}",
    )
    residual_npe = _one_step_residual_npe(true_residual, prediction_error)
    status = fig.suptitle("", fontsize=8.5)

    def update(frame: int) -> list[Any]:
        true_image.set_data(np.ma.masked_invalid(true_residual[frame]))
        predicted_image.set_data(np.ma.masked_invalid(predicted_residual[frame]))
        error_image.set_data(np.ma.masked_invalid(prediction_error[frame]))
        npe = residual_npe[frame]
        npe_label = (
            f"one-step residual NPE={npe:.4g}"
            if math.isfinite(npe)
            else "one-step residual NPE=n/a"
        )
        status.set_text(
            f"open validation | truth-input pair {frame + 1}/{VALIDATION_HORIZON} | "
            f"input U(t_{frame}) | target U(t_{frame + 1}) | "
            f"t={times[frame + 1]:.9g} s | {npe_label} | fixed scales"
        )
        return [true_image, predicted_image, error_image, status]

    if output_path.suffix.lower() == ".gif":
        writer: Any = animation.PillowWriter(fps=fps)
        encoder = {"writer": "matplotlib.PillowWriter", "fps": fps}
    elif output_path.suffix.lower() == ".mp4":
        if not animation.writers.is_available("ffmpeg"):
            raise RuntimeError("ffmpeg is required for MP4 output")
        writer = animation.FFMpegWriter(
            fps=fps,
            codec="libx264",
            bitrate=4500,
            extra_args=["-pix_fmt", "yuv420p"],
        )
        encoder = {
            "writer": "matplotlib.FFMpegWriter",
            "codec": "libx264",
            "pixel_format": "yuv420p",
            "fps": fps,
        }
    else:
        raise ValueError("animation output must end in .gif or .mp4")
    movie = animation.FuncAnimation(
        fig,
        update,
        frames=VALIDATION_HORIZON,
        interval=1000 / fps,
        blit=False,
    )
    movie.save(
        output_path,
        writer=writer,
        dpi=dpi,
        savefig_kwargs={"facecolor": "white", "transparent": False},
    )
    plt.close(fig)
    finite_true = true_residual[np.isfinite(true_residual)]
    finite_predicted = predicted_residual[np.isfinite(predicted_residual)]
    finite_error = prediction_error[np.isfinite(prediction_error)]
    if not (finite_true.size and finite_predicted.size and finite_error.size):
        raise RuntimeError("one-step residual animation has a non-finite-only panel")
    return {
        "field": field,
        "path": output_path.name,
        "sha256": sha256_file(output_path),
        "size": output_path.stat().st_size,
        "view": "ordered_teacher_forced",
        "conditioning": "decoded_reference[n] restored before every prediction",
        "residual_space": "decoded_released_units",
        "true_residual_definition": "decoded_reference[n+1]-decoded_reference[n]",
        "predicted_residual_definition": (
            "decoded_truth_input_prediction[n+1]-decoded_reference[n]"
        ),
        "prediction_error_definition": (
            "decoded_truth_input_prediction[n+1]-decoded_reference[n+1]"
        ),
        "prediction_error_equals": "predicted_residual-true_residual",
        "accumulated_rollout_error_visualized": False,
        "network_raw_normalized_residual_visualized": False,
        "residual_colormap": "RdBu_r_centered_at_zero",
        "residual_shared_abs_q99_5_scale": residual_limit,
        "prediction_error_abs_q99_5_scale": error_limit,
        "scale_source": (
            "99.5th percentile of finite absolute values on the "
            "visualization-only spatially subsampled grid"
        ),
        "true_residual_saturation_fraction": float(
            np.mean(np.abs(finite_true) > residual_limit)
        ),
        "predicted_residual_saturation_fraction": float(
            np.mean(np.abs(finite_predicted) > residual_limit)
        ),
        "prediction_error_saturation_fraction": float(
            np.mean(np.abs(finite_error) > error_limit)
        ),
        "rendered_frame_count": VALIDATION_HORIZON,
        "all_truth_input_pairs_rendered": True,
        "encoder": encoder,
    }


def _artifact_records(paths: Sequence[Path]) -> list[dict[str, Any]]:
    return [
        {"path": path.name, "sha256": sha256_file(path), "size": path.stat().st_size}
        for path in paths
    ]


def _first_threshold(
    values: Sequence[float], threshold: float, *, below: bool
) -> int | None:
    for index, value in enumerate(values):
        if (float(value) < threshold) if below else (float(value) > threshold):
            return index + 1
    return None


def _analysis_summary(
    result: Mapping[str, Any],
    history: Mapping[str, Any],
    teacher_events: Sequence[Mapping[str, Any]],
    free_events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    teacher = result["views"]["ordered_teacher_forced"]["summary"]
    free = result["views"]["free_recurrence"]["summary"]
    teacher_npe = np.asarray(teacher["npe_total_by_call"], dtype=np.float64)
    free_npe = np.asarray(free["npe_total_by_call"], dtype=np.float64)
    ratio = np.divide(
        free_npe,
        teacher_npe,
        out=np.full_like(free_npe, np.nan),
        where=teacher_npe != 0.0,
    )
    rows = history["rows"]
    validation_npe = np.asarray(
        [row["validation"]["realm_npe_mean"] for row in rows], dtype=np.float64
    )
    train_loss = np.asarray(
        [row["interval_mean_train_grouped_loss"] for row in rows], dtype=np.float64
    )
    steps = np.asarray([row["completed_step"] for row in rows], dtype=np.int64)
    best_index = int(np.argmin(validation_npe))
    return {
        "training": {
            "best_step": int(steps[best_index]),
            "best_truth_input_npe": float(validation_npe[best_index]),
            "final_truth_input_npe": float(validation_npe[-1]),
            "final_over_best_npe": float(
                validation_npe[-1] / validation_npe[best_index]
            ),
            "post_best_train_loss_change_fraction": float(
                train_loss[-1] / train_loss[best_index] - 1.0
            ),
            "post_best_validation_npe_change_fraction": float(
                validation_npe[-1] / validation_npe[best_index] - 1.0
            ),
            "competence_all_gate_pass_count": int(
                sum(bool(row["competence_gate"]["all_gates_pass"]) for row in rows)
            ),
            "pMax_monotonicity_gate_pass_count": int(
                sum(
                    bool(row["competence_gate"]["all_pMax_nondecreasing_from_input"])
                    for row in rows
                )
            ),
        },
        "evaluation": {
            "truth_input_npe_mean": float(teacher["realm_npe_mean"]),
            "free_npe_mean": float(free["realm_npe_mean"]),
            "free_over_truth_input_mean_npe": float(
                free["realm_npe_mean"] / teacher["realm_npe_mean"]
            ),
            "free_endpoint_npe": float(free_npe[-1]),
            "truth_input_endpoint_npe": float(teacher_npe[-1]),
            "free_endpoint_over_truth_input": float(ratio[-1]),
            "free_max_npe": float(np.max(free_npe)),
            "free_max_npe_call": int(np.argmax(free_npe) + 1),
            "first_free_over_truth_input_ratio_gt_2": _first_threshold(
                ratio, 2.0, below=False
            ),
            "first_free_over_truth_input_ratio_gt_5": _first_threshold(
                ratio, 5.0, below=False
            ),
            "first_free_over_truth_input_ratio_gt_10": _first_threshold(
                ratio, 10.0, below=False
            ),
            "first_free_decoded_correlation_below_0_8": _first_threshold(
                free["decoded_correlation_by_call"], 0.8, below=True
            ),
            "first_free_decoded_correlation_below_0_5": _first_threshold(
                free["decoded_correlation_by_call"], 0.5, below=True
            ),
            "first_teacher_inadmissible_call": _first_event(
                teacher_events, "admissible", expected=False
            ),
            "first_free_inadmissible_call": _first_event(
                free_events, "admissible", expected=False
            ),
            "first_teacher_pMax_decrease_call": _first_event(
                teacher_events, "pMax_nondecreasing", expected=False
            ),
            "first_free_pMax_decrease_call": _first_event(
                free_events, "pMax_nondecreasing", expected=False
            ),
            "first_teacher_unbounded_call": _first_event(
                teacher_events, "bounded_10x_train_max", expected=False
            ),
            "first_free_unbounded_call": _first_event(
                free_events, "bounded_10x_train_max", expected=False
            ),
        },
        "structure": {
            "truth_input": {
                key: result["structure"]["ordered_teacher_forced"][key]
                for key in (
                    "front_speed_error_m_per_s",
                    "mean_cell_size_proxy_error_mm",
                    "mean_shock_attached_highpass_fraction_error",
                    "arrival_time_error_s",
                )
            },
            "free_recurrence": {
                key: result["structure"]["free_recurrence"][key]
                for key in (
                    "front_speed_error_m_per_s",
                    "mean_cell_size_proxy_error_mm",
                    "mean_shock_attached_highpass_fraction_error",
                    "arrival_time_error_s",
                )
            },
        },
        "claim_boundary": (
            "Open-validation description for one selected seed-0 checkpoint. "
            "The preregistered truth-input competence gate fails on cumulative-pMax "
            "monotonicity, so recurrent differences are diagnostic rather than a "
            "qualified PlanarDet or architecture-level mechanism claim."
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    fields = _validate_fields(args.animation_fields)
    if args.fps < 1 or args.dpi < 50 or args.spatial_stride < 1:
        raise ValueError("fps, dpi, and spatial stride must be positive and meaningful")

    source_manifest = build_source_manifest(
        REPO_ROOT,
        entrypoints=EXECUTABLE_ENTRYPOINTS,
    )
    contract, training_status, training_inputs, _training_config = (
        _validate_closed_training_tree(
            args.training_dir,
            current_source_manifest=source_manifest,
        )
    )
    result, evaluation_summary, verified_evaluation_files = (
        _validate_evaluation_directory(
            args.evaluation_dir,
            current_source_manifest=source_manifest,
            training_dir=args.training_dir,
        )
    )
    if (
        result.get("run_id") != contract.run_id
        or result.get("training_run_signature") != training_status.get("run_signature")
        or result.get("test_object_opened") is not False
    ):
        raise ValueError("evaluation and training identities differ")

    manifest_payload = _load_json(args.manifest)
    if canonical_json_sha256(manifest_payload) != contract.open_manifest_payload_sha256:
        raise ValueError("open manifest payload differs from preregistration")
    repository, revision, entries = parse_manifest_payload(manifest_payload)
    manifest_summary = validate_planardet_open_manifest(repository, revision, entries)
    inventory = validate_local_open_tree(args.data_root, entries)
    metadata = load_planardet_metadata(args.data_root / "data" / "data.npz")
    if (
        manifest_summary["manifest_sha256"] != PLANARDET_OPEN_MANIFEST_SHA256
        or training_inputs.get("test_object_opened") is not False
        or result.get("validation_case") != PLANARDET_VAL_GROUPS[0]
    ):
        raise ValueError("visualization data identity differs from evaluation")
    bundle = load_normalizer_bundle(
        args.normalizer_arrays,
        expected_sha256=contract.normalizer_arrays_sha256,
        expected_coordinates_yx=metadata.canonical_coords_yx,
    )
    truth_path = args.data_root.joinpath(
        *PurePosixPath(trajectory_relative_path("val", PLANARDET_VAL_GROUPS[0])).parts
    )
    truth_native = load_planardet_trajectory(truth_path, canonical=True)
    teacher_normalized = np.load(
        args.evaluation_dir / "teacher_prediction_normalized.npy",
        allow_pickle=False,
        mmap_mode="r",
    )
    free_normalized = np.load(
        args.evaluation_dir / "free_prediction_normalized.npy",
        allow_pickle=False,
        mmap_mode="r",
    )

    visual, teacher_events, free_events, scales = _decode_visualization_bundle(
        teacher_normalized,
        free_normalized,
        truth_native,
        bundle=bundle,
        fields=fields,
        spatial_stride=args.spatial_stride,
    )
    history = _load_json(args.training_dir / "history.json")
    analysis = _analysis_summary(result, history, teacher_events, free_events)
    error_rows = _error_rows(result, metadata.times)
    event_rows = _event_rows(teacher_events, free_events, metadata.times)
    loss_rows = _loss_rows(history)

    _prepare_output_directory(
        args.output_dir,
        evaluation_dir=args.evaluation_dir,
        training_dir=args.training_dir,
        data_root=args.data_root,
    )
    error_csv = args.output_dir / "error_accumulation.csv"
    event_csv = args.output_dir / "event_timeline.csv"
    loss_csv = args.output_dir / "training_validation_loss.csv"
    analysis_path = args.output_dir / "analysis_summary.json"
    _write_csv(error_csv, error_rows)
    _write_csv(event_csv, event_rows)
    _write_csv(loss_csv, loss_rows)
    _write_json(analysis_path, analysis)

    figure_paths = [
        *render_error_accumulation(result, args.output_dir),
        *render_channel_heatmap(result, args.output_dir),
        *render_structure_evolution(result, args.output_dir),
        *render_training_history(history, args.output_dir),
    ]
    for field in fields:
        figure_paths.extend(
            render_snapshots(
                field,
                visual[field],
                scales[field],
                x=metadata.x[:: args.spatial_stride],
                y=metadata.y[:: args.spatial_stride],
                output_dir=args.output_dir,
            )
        )
    figure_records = _artifact_records(figure_paths)

    extension = f".{args.format}"
    animation_records: dict[str, Any] = {}
    one_step_residual_animation_records: dict[str, Any] = {}
    for field in fields:
        output_path = args.output_dir / f"rollout_{field}{extension}"
        animation_records[field] = render_animation(
            field,
            visual[field],
            scales[field],
            result,
            teacher_events,
            free_events,
            metadata.times,
            x=metadata.x[:: args.spatial_stride],
            y=metadata.y[:: args.spatial_stride],
            output_path=output_path,
            fps=args.fps,
            dpi=args.dpi,
        )
        print(json.dumps(animation_records[field], sort_keys=True), flush=True)
        one_step_output_path = args.output_dir / f"one_step_residual_{field}{extension}"
        one_step_residual_animation_records[field] = render_one_step_residual_animation(
            field,
            visual[field],
            scales[field],
            metadata.times,
            x=metadata.x[:: args.spatial_stride],
            y=metadata.y[:: args.spatial_stride],
            output_path=one_step_output_path,
            fps=args.fps,
            dpi=args.dpi,
        )
        print(
            json.dumps(one_step_residual_animation_records[field], sort_keys=True),
            flush=True,
        )

    summary: dict[str, Any] = {
        "schema": VISUALIZATION_SCHEMA,
        "status": "completed",
        "run_id": contract.run_id,
        "checkpoint": dict(result["checkpoint"]),
        "validation_case": PLANARDET_VAL_GROUPS[0],
        "input_bindings": {
            "evaluation_result_sha256": verified_evaluation_files["result.json"],
            "evaluation_final_hash_manifest_sha256": sha256_file(
                args.evaluation_dir / "final_hash_manifest.json"
            ),
            "evaluation_summary_sha256": sha256_file(
                args.evaluation_dir / "summary.json"
            ),
            "training_history_sha256": sha256_file(args.training_dir / "history.json"),
            "normalizer_arrays_sha256": bundle.source_sha256,
            "open_manifest_payload_sha256": canonical_json_sha256(manifest_payload),
            "current_training_source_manifest_digest": source_manifest[
                "canonical_payload_sha256"
            ],
            "visualizer_source_sha256": sha256_file(Path(__file__)),
            "open_tree_file_count": len(inventory),
        },
        "evaluation_summary": dict(evaluation_summary),
        "analysis_summary": analysis,
        "visualization_contract": {
            "animation_fields": list(fields),
            "rendered_frame_count": VALIDATION_HORIZON + 1,
            "rendered_one_step_residual_frame_count": VALIDATION_HORIZON,
            "all_released_frames_rendered": True,
            "all_truth_input_pairs_rendered": True,
            "spatial_stride": args.spatial_stride,
            "visualization_only_subsampling": True,
            "scientific_metrics_full_resolution": True,
            "state_scale": "exact released-truth min/max shared by truth, truth-input, and free panels",
            "error_scale": "declared 99.5th percentile shared by truth-input/free error panels",
            "one_step_residual_definition": (
                "true=decoded_reference[n+1]-decoded_reference[n]; "
                "predicted=decoded_truth_input_prediction[n+1]-"
                "decoded_reference[n]"
            ),
            "one_step_prediction_error_definition": (
                "decoded_truth_input_prediction[n+1]-decoded_reference[n+1]"
            ),
            "one_step_conditioning": (
                "decoded reference state n restored before every model call"
            ),
            "accumulated_rollout_error_visualized": False,
            "one_step_residual_space": "decoded released units",
            "one_step_residual_scale": (
                "fixed zero-centered 99.5th-percentile scale shared by true and "
                "predicted residual panels; separate fixed zero-centered one-step "
                "prediction-error scale"
            ),
            "coordinate_units": "mm for display, converted from released m coordinates",
            "physical_time_units": "s from release metadata",
            "scales": scales,
        },
        "animations": animation_records,
        "one_step_residual_animations": one_step_residual_animation_records,
        "figures": figure_records,
        "tables": _artifact_records((error_csv, event_csv, loss_csv)),
        "analysis_artifact": _artifact_records((analysis_path,))[0],
        "truth_input_competence_pass": result["truth_input_competence_gate"][
            "all_gates_pass"
        ],
        "mechanism_interpretation_allowed": result["mechanism_interpretation_allowed"],
        "test_object_opened": False,
        "anti_claims": [
            "all media are open-validation diagnostics for one selected seed-0 checkpoint",
            "visualization-only spatial subsampling is not used for scientific metrics",
            "pMax is a released cumulative diagnostic, not instantaneous pressure",
            "released fields do not support a physical conservation claim",
            "the failed truth-input pMax monotonicity gate prevents a qualified PlanarDet mechanism claim",
        ],
    }
    _write_json(args.output_dir / "visualization_summary.json", summary)
    artifact_names = {
        *[record["path"] for record in animation_records.values()],
        *[record["path"] for record in one_step_residual_animation_records.values()],
        *[record["path"] for record in figure_records],
        "error_accumulation.csv",
        "event_timeline.csv",
        "training_validation_loss.csv",
        "analysis_summary.json",
        "visualization_summary.json",
    }
    final_manifest = {
        "schema": VISUALIZATION_FINAL_MANIFEST_SCHEMA,
        "files": {
            name: sha256_file(args.output_dir / name) for name in sorted(artifact_names)
        },
        "self_hash_excluded": True,
        "test_object_opened": False,
    }
    _write_json(args.output_dir / "final_hash_manifest.json", final_manifest)
    _verify_final_manifest(
        args.output_dir,
        final_manifest,
        schema=VISUALIZATION_FINAL_MANIFEST_SCHEMA,
        expected_files=frozenset(artifact_names),
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary = run(args)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(summary, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
