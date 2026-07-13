"""Aggregate 1D Euler target-ladder experiment summaries.

The generic aggregate/rank outputs are useful for broad sweeps. For the current
Idea 2.1 selector, this script also adds diagnostic flags that keep us from
mistaking floor-clamped or heavily limited rollouts for structure-preserving
successes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Sequence


DEFAULT_METRIC = "rollout_relative_l2_final"
VERDICT_ORDER = {"candidate": 0, "caution": 1, "reject": 2}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Experiment directories containing summary.csv, or summary.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/time_dependent_no/euler1d_target_ladder_analysis"),
    )
    parser.add_argument(
        "--rank-metric",
        default=DEFAULT_METRIC,
        help="Numeric metric used to rank rows within each experiment group.",
    )
    parser.add_argument(
        "--group-by",
        choices=("experiment", "step_stride", "target", "all"),
        default="experiment",
        help="Column used for rank groups.",
    )
    parser.add_argument(
        "--floor-count-warning",
        type=int,
        default=0,
        help="Warn when density/pressure near-floor count is above this value.",
    )
    parser.add_argument(
        "--limiter-activation-warning",
        type=float,
        default=0.25,
        help="Warn when limiter activation fraction is above this value.",
    )
    parser.add_argument(
        "--flux-correction-saturation-warning",
        type=float,
        default=0.25,
        help="Warn when bounded flux-correction saturation is above this value.",
    )
    parser.add_argument(
        "--limiter-theta-min-warning",
        type=float,
        default=0.05,
        help="Warn when minimum limiter theta is below this value.",
    )
    return parser.parse_args(argv)


def summary_path(path: Path) -> Path:
    if path.is_dir():
        path = path / "summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing summary file: {path}")
    return path


def read_summary(path: Path) -> list[dict[str, str]]:
    path = summary_path(path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["experiment"] = path.parent.name
        row["summary_path"] = str(path)
    return rows


def as_float(row: dict[str, Any], key: str, default: float = math.nan) -> float:
    value = row.get(key, "")
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def finite_or_inf(row: dict[str, Any], key: str) -> float:
    value = as_float(row, key)
    return value if math.isfinite(value) else float("inf")


def as_int(row: dict[str, Any], key: str) -> int:
    value = as_float(row, key, default=0.0)
    if not math.isfinite(value):
        return 0
    return int(value)


def first_present_int(row: dict[str, Any], *keys: str) -> int:
    for key in keys:
        if str(row.get(key, "")).strip() != "":
            return as_int(row, key)
    return 0


def as_bool(row: dict[str, Any], key: str, default: bool = False) -> bool:
    value = str(row.get(key, "")).strip().lower()
    if value in {"true", "1", "yes"}:
        return True
    if value in {"false", "0", "no"}:
        return False
    return default


def infer_stride(row: dict[str, Any]) -> str:
    if row.get("step_stride"):
        return str(row["step_stride"])
    experiment = str(row.get("experiment", ""))
    marker = "stride"
    if marker not in experiment:
        return ""
    tail = experiment.split(marker, 1)[1]
    digits = []
    for char in tail:
        if char.isdigit():
            digits.append(char)
        elif digits:
            break
    return "".join(digits)


def rank_group_key(row: dict[str, Any], group_by: str) -> str:
    if group_by == "all":
        return "all"
    if group_by == "step_stride":
        return str(row.get("step_stride") or infer_stride(row))
    return str(row.get(group_by, ""))


def annotate_model_implementation(row: dict[str, Any]) -> None:
    implementation = str(row.get("model_implementation", "")).strip()
    model = str(row.get("model", "")).strip()
    deprecated = "no"
    reason = ""
    if not implementation:
        if model == "cpgnet":
            implementation = "deprecated_legacy_CPGStyleEuler1DHead"
            deprecated = "yes"
            reason = "unlabeled legacy cpgnet row used the old CPG-style pilot head"
        elif model == "fno":
            implementation = "legacy_FNOEuler1DHead_unspecified_config"
        else:
            implementation = "unspecified"
    if model == "cpg_style_pilot" or "CPGStylePilot" in implementation:
        deprecated = "yes"
        reason = reason or "explicit CPG-style pilot head"
    row["model_implementation"] = implementation
    row["deprecated_result"] = deprecated
    row["deprecated_reason"] = reason


def add_selector_diagnostics(row: dict[str, Any], args: argparse.Namespace) -> None:
    flags: list[str] = []
    status = str(row.get("status", "ok"))
    finite = as_bool(row, "finite", default=True)
    completed = as_bool(row, "completed_horizon", default=True)
    nonpositive_density = first_present_int(
        row,
        "num_nonpositive_proposed_density",
        "num_nonpositive_raw_density",
    ) + first_present_int(
        row,
        "one_step_num_nonpositive_proposed_density",
        "one_step_num_nonpositive_raw_density",
    )
    nonpositive_pressure = first_present_int(
        row,
        "num_nonpositive_proposed_pressure",
        "num_nonpositive_raw_pressure",
    ) + first_present_int(
        row,
        "one_step_num_nonpositive_proposed_pressure",
        "one_step_num_nonpositive_raw_pressure",
    )
    near_floor_density = first_present_int(
        row,
        "num_proposed_density_near_floor",
        "num_raw_density_near_floor",
    ) + first_present_int(
        row,
        "one_step_num_proposed_density_near_floor",
        "one_step_num_raw_density_near_floor",
    )
    near_floor_pressure = first_present_int(
        row,
        "num_proposed_pressure_near_floor",
        "num_raw_pressure_near_floor",
    ) + first_present_int(
        row,
        "one_step_num_proposed_pressure_near_floor",
        "one_step_num_raw_pressure_near_floor",
    )
    theta_min = as_float(row, "limiter_theta_min")
    theta_activation = as_float(row, "limiter_activation_fraction")
    flux_correction_saturation = as_float(row, "flux_correction_saturation_fraction")

    if status != "ok":
        flags.append("failed")
    if not finite:
        flags.append("nonfinite_rollout")
    if not completed:
        flags.append("incomplete_horizon")
    if nonpositive_density > 0:
        flags.append("nonpositive_density")
    if nonpositive_pressure > 0:
        flags.append("nonpositive_pressure")
    if near_floor_density > args.floor_count_warning:
        flags.append("density_floor_hugging")
    if near_floor_pressure > args.floor_count_warning:
        flags.append("pressure_floor_hugging")
    if (
        math.isfinite(theta_activation)
        and theta_activation > args.limiter_activation_warning
    ):
        flags.append("high_limiter_activation")
    if (
        math.isfinite(flux_correction_saturation)
        and flux_correction_saturation > args.flux_correction_saturation_warning
    ):
        flags.append("high_flux_correction_saturation")
    if math.isfinite(theta_min) and theta_min < args.limiter_theta_min_warning:
        flags.append("aggressive_limiter")

    reject_flags = {
        "failed",
        "nonfinite_rollout",
        "incomplete_horizon",
        "nonpositive_density",
        "nonpositive_pressure",
    }
    if any(flag in reject_flags for flag in flags):
        verdict = "reject"
    elif flags:
        verdict = "caution"
    else:
        verdict = "candidate"

    row["selector_verdict"] = verdict
    row["selector_flags"] = ";".join(flags)
    row["selector_nonpositive_total"] = nonpositive_density + nonpositive_pressure
    row["selector_near_floor_total"] = near_floor_density + near_floor_pressure


def limiter_theta_penalty(row: dict[str, Any]) -> float:
    theta_min = as_float(row, "limiter_theta_min")
    if not math.isfinite(theta_min):
        return 0.0
    return max(0.0, 1.0 - theta_min)


def limiter_activation_penalty(row: dict[str, Any]) -> float:
    activation = as_float(row, "limiter_activation_fraction")
    return activation if math.isfinite(activation) else 0.0


def selector_sort_key(row: dict[str, Any], rank_metric: str) -> tuple[Any, ...]:
    return (
        VERDICT_ORDER.get(str(row.get("selector_verdict", "reject")), 2),
        finite_or_inf(row, rank_metric),
        finite_or_inf(row, "shock_position_mae"),
        finite_or_inf(row, "conservative_total_error_final"),
        limiter_theta_penalty(row),
        limiter_activation_penalty(row),
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    preferred = [
        "experiment",
        "rank",
        "selector_rank",
        "rank_group",
        "model",
        "model_implementation",
        "deprecated_result",
        "deprecated_reason",
        "target",
        "target_type",
        "seed_count",
        "step_stride",
        "rollout_final_frame",
        "input_noise_std",
        "flux_correction_scale",
        "flux_correction_scale_floor",
        "selector_verdict",
        "selector_flags",
        "one_step_loss",
        "one_step_relative_l2",
        "test_relative_l2",
        "rollout_relative_l2_mean",
        "rollout_relative_l2_final",
        "shock_position_mae",
        "conservative_total_error_final",
        "min_density",
        "min_pressure",
        "proposed_min_density",
        "proposed_min_pressure",
        "raw_min_density",
        "raw_min_pressure",
        "num_nonpositive_proposed_density",
        "num_nonpositive_proposed_pressure",
        "num_proposed_density_near_floor",
        "num_proposed_pressure_near_floor",
        "num_nonpositive_raw_density",
        "num_nonpositive_raw_pressure",
        "num_raw_density_near_floor",
        "num_raw_pressure_near_floor",
        "limiter_theta_mean",
        "limiter_theta_min",
        "limiter_activation_fraction",
        "flux_correction_abs_over_bound_mean",
        "flux_correction_abs_over_bound_max",
        "flux_correction_saturation_fraction",
        "max_abs_primitive",
        "completed_horizon",
        "runtime_seconds",
        "summary_path",
    ]
    ordered = [key for key in preferred if key in fieldnames] + [
        key for key in fieldnames if key not in preferred
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, Any]], columns: list[str], limit: int) -> str:
    shown = rows[:limit]
    if not shown:
        return "No rows.\n"
    lines = ["| " + " | ".join(columns) + " |"]
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in shown:
        values = [str(row.get(column, "")) for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def write_report(
    path: Path, rows: list[dict[str, Any]], selector_rows: list[dict[str, Any]]
) -> None:
    columns = [
        "selector_rank",
        "model",
        "experiment",
        "target",
        "input_noise_std",
        "flux_correction_scale",
        "selector_verdict",
        "selector_flags",
        "deprecated_result",
        "rollout_relative_l2_final",
        "shock_position_mae",
        "conservative_total_error_final",
        "limiter_activation_fraction",
        "flux_correction_abs_over_bound_mean",
        "flux_correction_abs_over_bound_max",
        "flux_correction_saturation_fraction",
        "limiter_theta_min",
    ]
    verdict_counts = {
        verdict: sum(1 for row in rows if row.get("selector_verdict") == verdict)
        for verdict in ("candidate", "caution", "reject")
    }
    text = [
        "# 1D Euler Target-Ladder Analysis",
        "",
        "## Verdict Counts",
        "",
        json.dumps(verdict_counts, indent=2, sort_keys=True),
        "",
        "## Selector Ranking",
        "",
        markdown_table(selector_rows, columns, limit=20),
        "",
        "## Notes",
        "",
        "- `candidate` means no finite-horizon, positivity, floor-hugging, or aggressive-limiter warning was triggered by the configured thresholds.",
        "- `caution` means the rollout survived but may be relying on floors or limiter intervention.",
        "- `reject` means the row failed, became nonfinite, missed the horizon, or had nonpositive decoded conservative states.",
        "- `deprecated_result=yes` rows are legacy or explicit pilot runs and must not be mixed with corrected Section 1.2 CPGNet/FNO baselines.",
    ]
    path.write_text("\n".join(text) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    rows = [row for path in args.inputs for row in read_summary(path)]
    for row in rows:
        row["step_stride"] = row.get("step_stride") or infer_stride(row)
        annotate_model_implementation(row)
        add_selector_diagnostics(row, args)

    ranked: list[dict[str, Any]] = []
    selector_ranked: list[dict[str, Any]] = []
    for group_key in sorted({rank_group_key(row, args.group_by) for row in rows}):
        group = [row for row in rows if rank_group_key(row, args.group_by) == group_key]
        for rank, row in enumerate(
            sorted(group, key=lambda item: finite_or_inf(item, args.rank_metric)),
            start=1,
        ):
            ranked_row = dict(row)
            ranked_row["rank"] = rank
            ranked_row["rank_group"] = group_key
            ranked.append(ranked_row)
        for rank, row in enumerate(
            sorted(group, key=lambda item: selector_sort_key(item, args.rank_metric)),
            start=1,
        ):
            ranked_row = dict(row)
            ranked_row["selector_rank"] = rank
            ranked_row["rank_group"] = group_key
            selector_ranked.append(ranked_row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "aggregate.csv", rows)
    write_csv(args.output_dir / "ranked.csv", ranked)
    write_csv(args.output_dir / "selector_ranked.csv", selector_ranked)
    write_report(args.output_dir / "analysis.md", rows, selector_ranked)
    (args.output_dir / "aggregate.json").write_text(
        json.dumps(
            {
                "rows": rows,
                "rank_metric": args.rank_metric,
                "selector_thresholds": {
                    "floor_count_warning": args.floor_count_warning,
                    "limiter_activation_warning": args.limiter_activation_warning,
                    "limiter_theta_min_warning": args.limiter_theta_min_warning,
                    "flux_correction_saturation_warning": (
                        args.flux_correction_saturation_warning
                    ),
                },
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "aggregate_csv": str(args.output_dir / "aggregate.csv"),
                "ranked_csv": str(args.output_dir / "ranked.csv"),
                "selector_ranked_csv": str(args.output_dir / "selector_ranked.csv"),
                "analysis_md": str(args.output_dir / "analysis.md"),
                "rank_metric": args.rank_metric,
                "group_by": args.group_by,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
