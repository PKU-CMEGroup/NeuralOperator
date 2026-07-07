"""Aggregate 1D Euler target-ladder experiment summaries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence


DEFAULT_METRIC = "rollout_relative_l2_mean"


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


def as_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    try:
        return float(value)
    except ValueError:
        return float("inf")


def infer_stride(row: dict[str, str]) -> str:
    if row.get("step_stride"):
        return row["step_stride"]
    experiment = row.get("experiment", "")
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


def rank_group_key(row: dict[str, str], group_by: str) -> str:
    if group_by == "all":
        return "all"
    if group_by == "step_stride":
        return row.get("step_stride") or infer_stride(row)
    return row.get(group_by, "")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    preferred = [
        "experiment",
        "rank",
        "model",
        "target",
        "step_stride",
        "rollout_final_frame",
        "test_relative_l2",
        "rollout_relative_l2_mean",
        "rollout_relative_l2_final",
        "raw_min_pressure",
        "num_nonpositive_raw_pressure",
        "max_abs_primitive",
        "completed_horizon",
        "conservative_total_error_final",
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


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    rows = [row for path in args.inputs for row in read_summary(path)]
    for row in rows:
        row["step_stride"] = row.get("step_stride") or infer_stride(row)

    ranked: list[dict[str, Any]] = []
    for group_key in sorted({rank_group_key(row, args.group_by) for row in rows}):
        group = [row for row in rows if rank_group_key(row, args.group_by) == group_key]
        for rank, row in enumerate(
            sorted(group, key=lambda item: as_float(item, args.rank_metric)),
            start=1,
        ):
            ranked_row = dict(row)
            ranked_row["rank"] = rank
            ranked_row["rank_group"] = group_key
            ranked.append(ranked_row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "aggregate.csv", rows)
    write_csv(args.output_dir / "ranked.csv", ranked)
    (args.output_dir / "aggregate.json").write_text(
        json.dumps({"rows": rows, "rank_metric": args.rank_metric}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "aggregate_csv": str(args.output_dir / "aggregate.csv"),
                "ranked_csv": str(args.output_dir / "ranked.csv"),
                "rank_metric": args.rank_metric,
                "group_by": args.group_by,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
