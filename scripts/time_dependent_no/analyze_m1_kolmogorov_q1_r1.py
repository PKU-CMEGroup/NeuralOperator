"""Analyze and plot the immutable M1-Q1-R1 solver diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_ID = "M1-KF-Q1-R1-ANALYSIS-20260827A"
MIXING_RUN_ID = "M1-KF-Q1-R1-MIX-20260826A"
SPATIAL_RUN_ID = "M1-KF-Q1-R1-SPAT-20260826A"
MIXING_RESULT_SHA256 = (
    "fe3e0232d5ea65ba8b96148b6828ac2f5fa599af90978ef6b83b076ba97e6335"
)
MIXING_MANIFEST_SHA256 = (
    "c3c91f915716c08c0c9045b097cb45578e3974c7bee8e475556718807f651a44"
)
SPATIAL_RESULT_SHA256 = (
    "31dd4bdfa2e22c2bd3cdaea69201361a1e42d7375e24a677751877cc9834938d"
)
SPATIAL_MANIFEST_SHA256 = (
    "22de2039374be156fde278b1b1fb5eeaf4c02b0f36de1582537bda19b63a63a5"
)
SOURCE_PATHS = (
    "scripts/time_dependent_no/analyze_m1_kolmogorov_q1_r1.py",
    "tests/time_dependent_no/test_analyze_m1_kolmogorov_q1_r1.py",
)
PAIR_COLORS = {"64_to_128": "#D55E00", "128_to_256": "#0072B2"}
METRIC_COLORS = {
    "energy_relative_difference": "#009E73",
    "enstrophy_relative_difference": "#E69F00",
    "palinstrophy_relative_difference": "#CC79A7",
    "spectrum_total_variation": "#56B4E9",
}


def sha256_file(path: Path) -> str:
    """Return the SHA256 digest of one file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_packet(
    root: Path,
    *,
    expected_run_id: str,
    expected_result_sha256: str,
    expected_manifest_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load one exact R1 packet and verify every manifest member."""

    result_path = root / "result.json"
    manifest_path = root / "artifact_manifest.json"
    if sha256_file(result_path) != expected_result_sha256:
        raise RuntimeError(f"result hash mismatch: {result_path}")
    if sha256_file(manifest_path) != expected_manifest_sha256:
        raise RuntimeError(f"manifest hash mismatch: {manifest_path}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if result.get("run_id") != expected_run_id:
        raise RuntimeError(f"run identity mismatch: {root}")
    if manifest.get("run_id") != expected_run_id:
        raise RuntimeError(f"manifest run identity mismatch: {root}")
    for relative, expected in manifest.get("artifacts", {}).items():
        member = (root / relative).resolve()
        if root.resolve() not in member.parents:
            raise RuntimeError(f"artifact path escapes packet: {relative}")
        if sha256_file(member) != expected:
            raise RuntimeError(f"artifact hash mismatch: {member}")
    return result, manifest


def candidate_table(diagnostic: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten registered mixing-window outcomes without changing gates."""

    rows = []
    for candidate in diagnostic["candidates"]:
        row: dict[str, Any] = {
            "burnin_calls": candidate["burnin_calls"],
            "observation_start_call": candidate["observation_start_call"],
            "observation_end_call": candidate["observation_end_call"],
            "candidate_pass": candidate["pass"],
        }
        for name in ("energy", "enstrophy"):
            metric = candidate[name]
            row.update(
                {
                    f"{name}_maximum_half_change": max(
                        item["half_change"] for item in metric["rows"]
                    ),
                    f"{name}_split_rhat": metric["split_rhat"],
                    f"{name}_pooled_ess": metric["pooled_effective_sample_size"],
                    f"{name}_half_change_pass": metric["half_change_pass"],
                    f"{name}_shared_drift_pass": metric["shared_drift"]["pass"],
                    f"{name}_rhat_pass": metric["rhat_pass"],
                    f"{name}_ess_pass": metric["ess_pass"],
                }
            )
        rows.append(row)
    return rows


def endpoint_table(diagnostic: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten the registered adjacent-resolution endpoint summaries."""

    rows = []
    horizon = max(int(row["horizon"]) for row in diagnostic["rows"])
    for pair, pair_values in diagnostic["endpoints"].items():
        for endpoint, call in (("h1", 1), ("horizon", horizon)):
            values = pair_values[endpoint]
            rows.append(
                {
                    "pair": pair,
                    "horizon": call,
                    "median": values["median"],
                    "maximum": values["maximum"],
                    "clean_median": values["families"]["clean"]["median"],
                    "clean_maximum": values["families"]["clean"]["maximum"],
                    "displaced_median": values["families"]["displaced"]["median"],
                    "displaced_maximum": values["families"]["displaced"]["maximum"],
                }
            )
    return rows


def metric_endpoint_table(
    spatial_rows: list[dict[str, Any]], pair: str, horizon: int
) -> list[dict[str, float | str]]:
    """Summarize structural discrepancies at one registered endpoint."""

    selected = [
        row
        for row in spatial_rows
        if row["pair"] == pair and int(row["horizon"]) == horizon
    ]
    rows = []
    for metric in METRIC_COLORS:
        values = np.asarray([row[metric] for row in selected], dtype=np.float64)
        rows.append(
            {
                "metric": metric,
                "median": float(np.median(values)),
                "maximum": float(np.max(values)),
            }
        )
    return rows


def build_summary(
    mixing_result: dict[str, Any], spatial_result: dict[str, Any]
) -> dict[str, Any]:
    """Build a compact evidence table from the two immutable result packets."""

    mixing = mixing_result["mixing_diagnostic"]
    spatial = spatial_result["spatial_diagnostic"]
    horizon = max(int(row["horizon"]) for row in spatial["rows"])
    return {
        "analysis_id": ANALYSIS_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mixing": {
            "classification": mixing_result["classification"],
            "candidate_found": mixing["candidate_found"],
            "selected_burnin_calls": mixing["selected_burnin_calls"],
            "candidate_table": candidate_table(mixing),
            "all_finite": mixing["all_finite"],
            "parent_state_replay_pass": mixing_result["parent_state_replay"]["pass"],
        },
        "spatial": {
            "classification": spatial_result["classification"],
            "gate_table": {
                key: spatial[key]
                for key in (
                    "absolute_pass",
                    "contraction_pass",
                    "all_finite",
                    "closure_pass",
                    "screen_pass",
                )
            },
            "endpoint_table": endpoint_table(spatial),
            "upper_to_lower_error_ratios": spatial["upper_to_lower_error_ratios"],
            "upper_pair_horizon_structure": metric_endpoint_table(
                spatial["rows"], "128_to_256", horizon
            ),
            "repeatability": spatial["repeatability"],
            "parent_state_replay_pass": spatial_result["parent_state_replay"]["pass"],
        },
        "scope": {
            "post_hoc_gate_change": False,
            "model_or_dataset_access": False,
            "new_solver_execution": False,
            "interpretation_boundary": (
                "R1 diagnoses this reference/population attempt; it neither "
                "qualifies M1-Q1 nor identifies a model mechanism."
            ),
        },
    }


def configure_matplotlib() -> None:
    """Apply a compact, colorblind-safe paper style."""

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.titleweight": "bold",
            "axes.labelsize": 8,
            "legend.fontsize": 6.8,
            "legend.frameon": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.16,
            "lines.linewidth": 1.4,
        }
    )


def plot_mixing(
    series: dict[str, np.ndarray],
    diagnostic: dict[str, Any],
    output_dir: Path,
) -> list[Path]:
    """Plot trajectory and gate diagnostics for all candidate windows."""

    calls = series["calls"]
    seeds = series["seeds"]
    colors = ("#0072B2", "#E69F00", "#009E73", "#CC79A7")
    fig, axes = plt.subplots(2, 3, figsize=(6.75, 4.25))
    for axis, metric, title in zip(
        axes[0],
        ("energy", "enstrophy", "palinstrophy"),
        ("Kinetic energy", "Enstrophy", "Palinstrophy"),
        strict=True,
    ):
        for index, seed in enumerate(seeds):
            axis.plot(
                calls, series[metric][index], color=colors[index], label=str(seed)
            )
        for burnin in diagnostic["candidates"]:
            axis.axvline(burnin["burnin_calls"], color="#808080", lw=0.55, alpha=0.35)
        axis.set_title(title)
        axis.set_xlabel("Solver calls")
    axes[0, 0].set_ylabel("Value")
    axes[0, 0].legend(ncol=2, title="Seed", title_fontsize=7)

    candidates = candidate_table(diagnostic)
    burnins = np.asarray([row["burnin_calls"] for row in candidates])
    for name, color, label in (
        ("energy", "#0072B2", "Energy"),
        ("enstrophy", "#D55E00", "Enstrophy"),
    ):
        axes[1, 0].plot(
            burnins,
            [row[f"{name}_maximum_half_change"] for row in candidates],
            marker="o",
            color=color,
            label=label,
        )
        axes[1, 1].plot(
            burnins,
            [row[f"{name}_split_rhat"] for row in candidates],
            marker="o",
            color=color,
            label=label,
        )
        axes[1, 2].plot(
            burnins,
            [row[f"{name}_pooled_ess"] for row in candidates],
            marker="o",
            color=color,
            label=label,
        )
    axes[1, 0].axhline(0.10, color="#202020", ls="--", lw=0.9, label="Gate")
    axes[1, 1].axhline(1.05, color="#202020", ls="--", lw=0.9, label="Gate")
    axes[1, 2].axhline(100.0, color="#202020", ls="--", lw=0.9, label="Gate")
    for axis, title, ylabel in zip(
        axes[1],
        ("Worst chain half-change", "Split R-hat", "Pooled ESS"),
        ("Relative change", "R-hat", "Effective samples"),
        strict=True,
    ):
        axis.set_title(title)
        axis.set_xlabel("Candidate burn-in calls")
        axis.set_ylabel(ylabel)
    axes[1, 0].legend()
    fig.suptitle("M1-Q1-R1 mixing diagnosis: no registered window passes", y=1.01)
    fig.tight_layout()
    paths = [
        output_dir / "fig_mixing_diagnostics.pdf",
        output_dir / "fig_mixing_diagnostics.png",
    ]
    fig.savefig(paths[0], metadata={"Creator": ANALYSIS_ID, "CreationDate": None})
    fig.savefig(paths[1], dpi=300, metadata={"Software": ANALYSIS_ID})
    plt.close(fig)
    return paths


def _horizon_values(
    rows: list[dict[str, Any]], pair: str, metric: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    horizons = np.asarray(sorted({int(row["horizon"]) for row in rows}))
    grouped = [
        np.asarray(
            [
                row[metric]
                for row in rows
                if row["pair"] == pair and int(row["horizon"]) == horizon
            ],
            dtype=np.float64,
        )
        for horizon in horizons
    ]
    return (
        horizons,
        np.asarray([np.median(values) for values in grouped]),
        np.asarray([np.min(values) for values in grouped]),
        np.asarray([np.max(values) for values in grouped]),
    )


def plot_spatial(diagnostic: dict[str, Any], output_dir: Path) -> list[Path]:
    """Plot path convergence, contraction, and structural discrepancies."""

    rows = diagnostic["rows"]
    fig, axes = plt.subplots(1, 3, figsize=(6.75, 2.35))
    pair_curves = {}
    for pair, color in PAIR_COLORS.items():
        horizon, median, minimum, maximum = _horizon_values(
            rows, pair, "state_relative_l2"
        )
        pair_curves[pair] = (horizon, median, maximum)
        label = pair.replace("_to_", r"$\rightarrow$")
        axes[0].plot(
            horizon,
            median,
            marker="o",
            markevery=3,
            color=color,
            label=label,
        )
        axes[0].fill_between(horizon, minimum, maximum, color=color, alpha=0.14)
    axes[0].scatter(
        [1, 16],
        [0.0125, 0.05],
        marker="x",
        color="#202020",
        zorder=5,
        label="Median gates",
    )
    axes[0].set_yscale("log")
    axes[0].set_title("Path discrepancy")
    axes[0].set_xlabel("Rollout horizon")
    axes[0].set_ylabel("Relative state L2")
    axes[0].legend()

    lower_h, lower_median, lower_maximum = pair_curves["64_to_128"]
    upper_h, upper_median, upper_maximum = pair_curves["128_to_256"]
    if not np.array_equal(lower_h, upper_h):
        raise RuntimeError("spatial pairs have different horizons")
    axes[1].plot(
        lower_h,
        upper_median / np.maximum(lower_median, np.finfo(np.float64).eps),
        marker="o",
        markevery=3,
        color="#0072B2",
        label="Median ratio",
    )
    axes[1].plot(
        lower_h,
        upper_maximum / np.maximum(lower_maximum, np.finfo(np.float64).eps),
        marker="s",
        markevery=3,
        color="#D55E00",
        label="Maximum ratio",
    )
    axes[1].axhline(0.5, color="#202020", ls="--", lw=0.9, label="Endpoint gate")
    axes[1].set_title("Adjacent-grid contraction")
    axes[1].set_xlabel("Rollout horizon")
    axes[1].set_ylabel("N128-to-N256 / N64-to-N128")
    axes[1].legend()

    axes[2].plot(upper_h, upper_median, color="#7A7A7A", ls="--", label="State L2")
    labels = {
        "energy_relative_difference": "Energy",
        "enstrophy_relative_difference": "Enstrophy",
        "palinstrophy_relative_difference": "Palinstrophy",
        "spectrum_total_variation": "Spectrum TV",
    }
    for metric, color in METRIC_COLORS.items():
        horizon, median, _, _ = _horizon_values(rows, "128_to_256", metric)
        axes[2].plot(horizon, median, color=color, label=labels[metric])
    axes[2].set_yscale("log")
    axes[2].set_title(r"N128$\rightarrow$N256 structure")
    axes[2].set_xlabel("Rollout horizon")
    axes[2].set_ylabel("Median discrepancy")
    axes[2].legend(ncol=2)

    fig.suptitle(
        "M1-Q1-R1 spatial diagnosis: contraction passes, H16 path gate fails", y=1.03
    )
    fig.tight_layout()
    paths = [
        output_dir / "fig_spatial_diagnostics.pdf",
        output_dir / "fig_spatial_diagnostics.png",
    ]
    fig.savefig(paths[0], metadata={"Creator": ANALYSIS_ID, "CreationDate": None})
    fig.savefig(paths[1], dpi=300, metadata={"Software": ANALYSIS_ID})
    plt.close(fig)
    return paths


def source_binding(source_commit: str) -> dict[str, Any]:
    """Require the analysis sources to equal one clean current commit."""

    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short", "--", *SOURCE_PATHS],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if source_commit != head:
        raise RuntimeError(f"--source-commit must equal current HEAD {head}")
    if status:
        raise RuntimeError("analysis sources differ from HEAD:\n" + status)
    return {
        "source_commit": source_commit,
        "sources": {
            relative: sha256_file(REPO_ROOT / relative) for relative in SOURCE_PATHS
        },
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--mixing-root", type=Path, required=True)
    result.add_argument("--spatial-root", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--source-commit", required=True)
    return result


def main() -> None:
    args = parser().parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")
    binding_start = source_binding(args.source_commit)
    mixing_result, _ = verify_packet(
        args.mixing_root,
        expected_run_id=MIXING_RUN_ID,
        expected_result_sha256=MIXING_RESULT_SHA256,
        expected_manifest_sha256=MIXING_MANIFEST_SHA256,
    )
    spatial_result, _ = verify_packet(
        args.spatial_root,
        expected_run_id=SPATIAL_RUN_ID,
        expected_result_sha256=SPATIAL_RESULT_SHA256,
        expected_manifest_sha256=SPATIAL_MANIFEST_SHA256,
    )
    with np.load(args.mixing_root / "series.npz") as archive:
        expected_keys = {
            "seeds",
            "calls",
            "energy",
            "enstrophy",
            "palinstrophy",
            "spectra",
            "substeps",
            "minimum_substep",
            "maximum_substep",
        }
        if set(archive.files) != expected_keys:
            raise RuntimeError("mixing series inventory mismatch")
        series = {name: np.asarray(archive[name]) for name in archive.files}
    if not all(np.all(np.isfinite(value)) for value in series.values()):
        raise RuntimeError("mixing archive contains nonfinite values")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()
    figures = [
        *plot_mixing(series, mixing_result["mixing_diagnostic"], args.output_dir),
        *plot_spatial(spatial_result["spatial_diagnostic"], args.output_dir),
    ]
    summary = build_summary(mixing_result, spatial_result)
    summary["input_bindings"] = {
        "mixing_result_sha256": MIXING_RESULT_SHA256,
        "mixing_manifest_sha256": MIXING_MANIFEST_SHA256,
        "spatial_result_sha256": SPATIAL_RESULT_SHA256,
        "spatial_manifest_sha256": SPATIAL_MANIFEST_SHA256,
    }
    summary_path = args.output_dir / "summary.json"
    write_json(summary_path, summary)
    binding_end = source_binding(args.source_commit)
    if binding_start != binding_end:
        raise RuntimeError("analysis source binding changed during execution")
    members = [summary_path, *figures]
    manifest = {
        "analysis_id": ANALYSIS_ID,
        **binding_end,
        "inputs": summary["input_bindings"],
        "artifacts": {path.name: sha256_file(path) for path in members},
        "self_exclusion": "artifact_manifest.json",
    }
    write_json(args.output_dir / "artifact_manifest.json", manifest)
    print(
        json.dumps(
            {"analysis_id": ANALYSIS_ID, "artifacts": manifest["artifacts"]},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
