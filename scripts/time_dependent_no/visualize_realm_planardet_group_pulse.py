"""Visualize the frozen PlanarDet one-call field-group pulse diagnostic."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.time_dependent_no.evaluate_realm_planardet_group_feedback import (
    MATERIAL_EFFECT_RATIO,
)
from scripts.time_dependent_no.evaluate_realm_planardet_group_pulse import (
    ARM_SPECS,
    PULSE_CALLS,
    PULSE_GROUPS,
    RESULT_SCHEMA,
    arm_id,
)
from utility.time_dependent_no.realm_benchmark import canonical_json_sha256
from utility.time_dependent_no.realm_planardet_runtime import VALIDATION_HORIZON

VISUALIZATION_SCHEMA = "w26_l4_planardet_group_pulse_visualization_v1"
MATERIAL_THRESHOLD = 1.0 - MATERIAL_EFFECT_RATIO
WINDOW_ORDER = ("immediate", "early", "middle", "late")
METRIC_ORDER = (
    "all_group_ratio",
    "common_downstream_ratio",
    "own_group_ratio",
    "partner_ratio",
)

GROUP_COLORS = {"chem": "#009E73", "rho": "#0072B2"}
PULSE_COLORS = {4: "#56B4E9", 12: "#E69F00", 32: "#CC79A7"}
GROUP_MARKERS = {"chem": "o", "rho": "s"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _load_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"JSON root must be an object: {path.name}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_positive(value: Any, *, name: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return number


def _history(summary: Mapping[str, Any], key: str) -> np.ndarray:
    values = np.asarray(summary.get(key), dtype=np.float64)
    if values.shape != (VALIDATION_HORIZON,) or not np.isfinite(values).all():
        raise ValueError(f"{key} must be a finite H49 vector")
    return values


def _validate_result(result: Mapping[str, Any]) -> None:
    unsigned = {
        key: value for key, value in result.items() if key != "canonical_payload_sha256"
    }
    expected_arms = {arm_id(group, pulse_call) for group, pulse_call in ARM_SPECS}
    if (
        result.get("schema") != RESULT_SCHEMA
        or canonical_json_sha256(unsigned) != result.get("canonical_payload_sha256")
        or result.get("diagnostic_interpretation_allowed") is not True
        or result.get("evaluator_closure", {}).get("all_gates_pass") is not True
        or result.get("test_object_opened") is not False
        or set(result.get("arms", {})) != expected_arms
    ):
        raise ValueError("group-pulse result identity or closure differs")


def build_plot_data(result: Mapping[str, Any]) -> dict[str, Any]:
    _validate_result(result)
    baseline_total = _history(result["baseline_free_summary"], "npe_total_by_call")
    primary: dict[tuple[str, int], float] = {}
    partner: dict[tuple[str, int], float] = {}
    own: dict[tuple[str, int], float] = {}
    all_group_horizon: dict[tuple[str, int], float] = {}
    windows: dict[tuple[str, int], dict[str, float]] = {}
    lag_response: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    total_by_call: dict[tuple[str, int], np.ndarray] = {}

    for group, pulse_call in ARM_SPECS:
        current_arm_id = arm_id(group, pulse_call)
        arm = result["arms"][current_arm_id]
        if (
            arm.get("group") != group
            or arm.get("pulse_call") != pulse_call
            or arm.get("valid_length") != VALIDATION_HORIZON
            or arm.get("nonfinite_proposal_call") is not None
            or arm.get("closure", {}).get("all_gates_pass") is not True
        ):
            raise ValueError("group-pulse arm identity or closure differs")

        current_total = _history(arm["raw_summary"], "npe_total_by_call")
        if not np.array_equal(current_total[:pulse_call], baseline_total[:pulse_call]):
            raise ValueError("pulse rollout prefix differs from baseline")
        total_by_call[(group, pulse_call)] = current_total

        comparison = arm["comparison_to_unintervened_free_baseline"]
        primary[(group, pulse_call)] = _finite_positive(
            comparison["primary_postpulse_common_downstream"]["ratio"],
            name="primary ratio",
        )
        partner[(group, pulse_call)] = _finite_positive(
            comparison["postpulse_partner_group"]["ratio"], name="partner ratio"
        )
        own[(group, pulse_call)] = _finite_positive(
            comparison["postpulse_own_group"]["ratio"], name="own-group ratio"
        )
        all_group_horizon[(group, pulse_call)] = _finite_positive(
            comparison["all_group_realm_npe_mean"]["ratio"],
            name="all-group horizon ratio",
        )
        temporal = comparison["temporal_common_downstream"]
        if set(temporal) != set(WINDOW_ORDER):
            raise ValueError("temporal window inventory differs")
        windows[(group, pulse_call)] = {
            window: _finite_positive(temporal[window]["ratio"], name="window ratio")
            for window in WINDOW_ORDER
        }

        readout = arm["postpulse_lag_readout"]
        expected_length = VALIDATION_HORIZON - pulse_call
        if len(readout) != expected_length:
            raise ValueError("post-pulse lag inventory differs")
        if [row["lag"] for row in readout] != list(range(1, expected_length + 1)):
            raise ValueError("post-pulse lags must be consecutive")
        if [row["call"] for row in readout] != list(
            range(pulse_call + 1, VALIDATION_HORIZON + 1)
        ):
            raise ValueError("post-pulse calls must be consecutive")
        lag_response[(group, pulse_call)] = {
            "lag": np.arange(1, expected_length + 1, dtype=np.int64),
            **{
                metric: np.asarray(
                    [
                        _finite_positive(row[metric], name=f"lag {metric}")
                        for row in readout
                    ],
                    dtype=np.float64,
                )
                for metric in METRIC_ORDER
            },
        }

    directionality = result["directionality_by_pulse"]
    if set(directionality) != {str(call) for call in PULSE_CALLS}:
        raise ValueError("directionality pulse inventory differs")
    for pulse_call in PULSE_CALLS:
        current = directionality[str(pulse_call)]
        if not np.isclose(
            current["chem_to_rho_ratio"], partner[("chem", pulse_call)]
        ) or not np.isclose(current["rho_to_chem_ratio"], partner[("rho", pulse_call)]):
            raise ValueError("directionality and arm partner ratios differ")

    return {
        "baseline_total_by_call": baseline_total,
        "total_by_call": total_by_call,
        "primary_ratio": primary,
        "partner_ratio": partner,
        "own_group_ratio": own,
        "all_group_horizon_ratio": all_group_horizon,
        "window_ratio": windows,
        "lag_response": lag_response,
        "directionality": directionality,
    }


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "legend.fontsize": 7.5,
            "legend.frameon": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.16,
            "lines.linewidth": 1.7,
            "lines.markersize": 4.5,
        }
    )


def _save(fig: plt.Figure, output_dir: Path, stem: str) -> list[Path]:
    paths = [output_dir / f"{stem}.pdf", output_dir / f"{stem}.png"]
    fig.savefig(paths[0])
    fig.savefig(paths[1], dpi=300)
    plt.close(fig)
    return paths


def _reference_lines(ax: plt.Axes) -> None:
    ax.axhspan(
        MATERIAL_THRESHOLD,
        1.0,
        color="#D1D5DB",
        alpha=0.28,
        label="small/inconclusive band",
    )
    ax.axhline(MATERIAL_THRESHOLD, color="#009E73", ls="--", lw=1)
    ax.axhline(1.0, color="#374151", lw=1)


def _plot_effect_summary(data: Mapping[str, Any], output_dir: Path) -> list[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.75), sharex=True, sharey=True)
    panels = (
        ("primary_ratio", "Common downstream: T + u"),
        ("partner_ratio", "Directional partner: chem ↔ rho"),
    )
    for ax, (metric, title) in zip(axes, panels, strict=True):
        _reference_lines(ax)
        for group in PULSE_GROUPS:
            values = [data[metric][(group, call)] for call in PULSE_CALLS]
            label = (
                f"{group} pulse"
                if metric == "primary_ratio"
                else ("chem → rho" if group == "chem" else "rho → chem")
            )
            ax.plot(
                PULSE_CALLS,
                values,
                color=GROUP_COLORS[group],
                marker=GROUP_MARKERS[group],
                label=label,
            )
            for call, value in zip(PULSE_CALLS, values, strict=True):
                ax.annotate(
                    f"{value:.3f}",
                    (call, value),
                    xytext=(0, 6),
                    textcoords="offset points",
                    ha="center",
                    fontsize=7,
                )
        ax.set_title(title)
        ax.set_xlabel("Pulse call")
        ax.set_xticks(PULSE_CALLS)
        ax.set_ylim(0.62, 1.025)
        ax.legend(loc="lower left")
    axes[0].set_ylabel("Post-pulse error ratio vs raw baseline")
    fig.suptitle("One exact field-group pulse has phase-dependent effects", y=1.04)
    fig.text(
        0.5,
        -0.03,
        "Lower is better; ≤0.90 is the preregistered material-effect threshold.",
        ha="center",
        fontsize=8,
        color="#4B5563",
    )
    fig.tight_layout(w_pad=1.4)
    return _save(fig, output_dir, "pulse_effect_summary")


def _plot_lag_response(data: Mapping[str, Any], output_dir: Path) -> list[Path]:
    all_values = []
    for group, pulse_call in ARM_SPECS:
        response = data["lag_response"][(group, pulse_call)]
        all_values.extend(response["common_downstream_ratio"])
        all_values.extend(response["partner_ratio"])
    lower = max(0.0, float(np.min(all_values)) - 0.06)
    upper = max(1.08, float(np.max(all_values)) + 0.04)

    fig, axes = plt.subplots(2, len(PULSE_CALLS), figsize=(9.2, 5.0), sharey=True)
    for column, pulse_call in enumerate(PULSE_CALLS):
        for row, (metric, ylabel) in enumerate(
            (
                ("common_downstream_ratio", "T + u error ratio"),
                ("partner_ratio", "Partner error ratio"),
            )
        ):
            ax = axes[row, column]
            _reference_lines(ax)
            for group in PULSE_GROUPS:
                response = data["lag_response"][(group, pulse_call)]
                label = (
                    f"{group} pulse"
                    if metric == "common_downstream_ratio"
                    else ("chem → rho" if group == "chem" else "rho → chem")
                )
                ax.plot(
                    response["lag"],
                    response[metric],
                    color=GROUP_COLORS[group],
                    label=label,
                )
            ax.set_ylim(lower, upper)
            ax.set_xlim(1, VALIDATION_HORIZON - pulse_call)
            if row == 0:
                ax.set_title(f"Pulse at call {pulse_call}")
            else:
                ax.set_xlabel("Calls after pulse")
            if column == 0:
                ax.set_ylabel(ylabel)
            if column == len(PULSE_CALLS) - 1:
                ax.legend(loc="lower right")
    fig.suptitle("Temporal response after one exact oracle pulse", y=1.01)
    fig.tight_layout(h_pad=1.2, w_pad=1.0)
    return _save(fig, output_dir, "postpulse_lag_response")


def _plot_error_accumulation(data: Mapping[str, Any], output_dir: Path) -> list[Path]:
    calls = np.arange(1, VALIDATION_HORIZON + 1)
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 5.2), sharex=True, sharey=True)
    for ax, group in zip(axes, PULSE_GROUPS, strict=True):
        ax.plot(
            calls,
            data["baseline_total_by_call"],
            color="#4B5563",
            lw=2.1,
            label="raw baseline",
            zorder=5,
        )
        for pulse_call in PULSE_CALLS:
            ax.plot(
                calls,
                data["total_by_call"][(group, pulse_call)],
                color=PULSE_COLORS[pulse_call],
                label=f"pulse call {pulse_call}",
            )
            ax.axvline(
                pulse_call,
                color=PULSE_COLORS[pulse_call],
                ls=":",
                lw=0.9,
                alpha=0.75,
            )
        ax.set_title(f"One {group} pulse")
        ax.set_ylabel("All-group NPE per call")
        ax.set_ylim(bottom=0.0)
        ax.legend(ncol=2, loc="upper left")
    axes[-1].set_xlabel("Autoregressive call")
    axes[-1].set_xlim(1, VALIDATION_HORIZON)
    fig.suptitle("Absolute rollout-error accumulation", y=1.01)
    fig.text(
        0.5,
        -0.015,
        "Each pulse proposal is unchanged; the intervention first affects the next call.",
        ha="center",
        fontsize=8,
        color="#4B5563",
    )
    fig.tight_layout(h_pad=1.2)
    return _save(fig, output_dir, "rollout_error_accumulation")


def _write_csvs(data: Mapping[str, Any], output_dir: Path) -> list[Path]:
    summary_path = output_dir / "pulse_effect_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "pulse_group",
                "pulse_call",
                "common_downstream_ratio",
                "partner_ratio",
                "own_group_ratio",
                "all_group_horizon_ratio",
                "directionality_classification",
            ]
        )
        for group, pulse_call in ARM_SPECS:
            writer.writerow(
                [
                    group,
                    pulse_call,
                    data["primary_ratio"][(group, pulse_call)],
                    data["partner_ratio"][(group, pulse_call)],
                    data["own_group_ratio"][(group, pulse_call)],
                    data["all_group_horizon_ratio"][(group, pulse_call)],
                    data["directionality"][str(pulse_call)]["classification"],
                ]
            )

    windows_path = output_dir / "temporal_window_response.csv"
    with windows_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pulse_group", "pulse_call", *WINDOW_ORDER])
        for group, pulse_call in ARM_SPECS:
            writer.writerow(
                [
                    group,
                    pulse_call,
                    *[
                        data["window_ratio"][(group, pulse_call)][window]
                        for window in WINDOW_ORDER
                    ],
                ]
            )

    lag_path = output_dir / "postpulse_lag_response.csv"
    with lag_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pulse_group", "pulse_call", "call", "lag", *METRIC_ORDER])
        for group, pulse_call in ARM_SPECS:
            response = data["lag_response"][(group, pulse_call)]
            for index, lag in enumerate(response["lag"]):
                writer.writerow(
                    [
                        group,
                        pulse_call,
                        pulse_call + int(lag),
                        int(lag),
                        *[response[metric][index] for metric in METRIC_ORDER],
                    ]
                )

    rollout_path = output_dir / "rollout_error_accumulation.csv"
    with rollout_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        header = ["call", "baseline_all_group_npe"]
        for group, pulse_call in ARM_SPECS:
            header.append(f"{arm_id(group, pulse_call)}_all_group_npe")
        writer.writerow(header)
        for index in range(VALIDATION_HORIZON):
            writer.writerow(
                [
                    index + 1,
                    data["baseline_total_by_call"][index],
                    *[
                        data["total_by_call"][(group, pulse_call)][index]
                        for group, pulse_call in ARM_SPECS
                    ],
                ]
            )
    return [summary_path, windows_path, lag_path, rollout_path]


def run_visualization(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")
    result = _load_json(args.result)
    data = build_plot_data(result)
    args.output_dir.mkdir(parents=True)
    _style()
    outputs = []
    outputs.extend(_plot_effect_summary(data, args.output_dir))
    outputs.extend(_plot_lag_response(data, args.output_dir))
    outputs.extend(_plot_error_accumulation(data, args.output_dir))
    outputs.extend(_write_csvs(data, args.output_dir))
    manifest: dict[str, Any] = {
        "schema": VISUALIZATION_SCHEMA,
        "inputs": {"result_sha256": _sha256(args.result)},
        "outputs": {
            path.name: {"sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in sorted(outputs)
        },
        "material_effect_threshold": MATERIAL_THRESHOLD,
        "test_object_opened": False,
    }
    manifest["canonical_payload_sha256"] = canonical_json_sha256(manifest)
    manifest_path = args.output_dir / "visualization_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = run_visualization(args)
    print(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
