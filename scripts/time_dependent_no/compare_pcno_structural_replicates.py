#!/usr/bin/env python3
"""Compare pathway and shock-profile aggregates across two PCNO replicates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import fmean
from typing import Any

SCHEMA = "pcno_structural_repeatability_v1"
INPUT_SCHEMA = "pcno_scale_separated_drift_diagnostic_v2"
PATHWAY_RELATIVE_METRICS = (
    "path_total_energy",
    "endpoint_total_energy",
    "path_energy_density_enrichment_within_band",
)
PATHWAY_ABSOLUTE_METRICS = (
    "path_mesh_symmetric_attribution_share",
    "path_state_symmetric_attribution_share",
    "endpoint_mesh_symmetric_attribution_share",
    "endpoint_state_symmetric_attribution_share",
    "negative_growth_fraction_total",
    "negative_growth_fraction_mesh",
    "negative_growth_fraction_state",
    "path_region_share_within_band",
    "endpoint_region_share_within_band",
)
SHOCK_POSITION_METRICS = (
    "median_rms_shock_position_error",
    "endpoint_rms_shock_position_error",
    "median_median_absolute_shock_position_error",
    "endpoint_median_absolute_shock_position_error",
)
SHOCK_LOG_METRICS = (
    "median_median_absolute_net_strength_log_error",
    "endpoint_median_absolute_net_strength_log_error",
    "median_median_absolute_total_variation_log_error",
    "endpoint_median_absolute_total_variation_log_error",
    "median_median_absolute_thickness_log_error",
    "endpoint_median_absolute_thickness_log_error",
)
SHOCK_FRACTION_METRICS = (
    "median_translation_only_energy_fraction",
    "endpoint_translation_only_energy_fraction",
    "median_joint_translation_dilation_amplitude_energy_fraction",
    "endpoint_joint_translation_dilation_amplitude_energy_fraction",
    "median_joint_unexplained_energy_fraction",
    "endpoint_joint_unexplained_energy_fraction",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-results-dir", type=Path, required=True)
    parser.add_argument("--confirmation-results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pathway-relative-tolerance", type=float, default=0.01)
    parser.add_argument("--pathway-absolute-tolerance", type=float, default=0.02)
    parser.add_argument("--shock-position-cell-tolerance", type=float, default=0.1)
    parser.add_argument("--shock-log-tolerance", type=float, default=0.02)
    parser.add_argument("--shock-fraction-tolerance", type=float, default=0.05)
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_and_verify(result_dir: Path) -> dict[str, Any]:
    summary_path = result_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("schema") != INPUT_SCHEMA:
        raise ValueError(f"unsupported result schema: {result_dir}")
    for name, expected in summary.get("output_hashes", {}).items():
        path = result_dir / str(name)
        if not path.is_file() or sha256_file(path) != str(expected):
            raise ValueError(f"registered result hash mismatch: {path}")
    return summary


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _cohort_means(
    rows: Sequence[Mapping[str, str]], keys: Sequence[str]
) -> dict[tuple[str, ...], dict[str, float]]:
    grouped: dict[tuple[str, ...], list[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row[key]) for key in keys)].append(row)
    return {
        key: {
            metric: fmean(float(row[metric]) for row in group)
            for metric in group[0]
            if metric not in {"case_id", *keys}
            and all(_is_float(row[metric]) for row in group)
        }
        for key, group in grouped.items()
    }


def _is_float(value: str) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _comparison(
    pilot: float,
    confirmation: float,
    *,
    mode: str,
    tolerance: float,
    scale: float = 1.0,
) -> tuple[float, bool]:
    if mode == "relative":
        denominator = max(abs(pilot), abs(confirmation), 1.0e-15)
        discrepancy = abs(confirmation - pilot) / denominator
    elif mode == "absolute":
        discrepancy = abs(confirmation - pilot) / scale
    else:
        raise ValueError(f"unsupported comparison mode: {mode}")
    return discrepancy, discrepancy <= tolerance


def _contract_projection(summary: Mapping[str, Any]) -> dict[str, Any]:
    contract = summary["contract"]
    return {
        "boundary_policy": summary["boundary_policy"],
        "population": summary["population"],
        "checkpoint_sha256": summary["input"]["checkpoint_sha256"],
        "normalization_digest": summary["input"]["normalization_digest"],
        "bands": contract["bands"],
        "component_scale": contract["component_scale"],
        "domain_lengths": contract["domain_lengths"],
        "gamma": contract["gamma"],
        "pathway_decomposition": contract["pathway_decomposition"],
        "shock_profile": contract["shock_profile"],
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    tolerances = {
        "pathway_relative": args.pathway_relative_tolerance,
        "pathway_absolute": args.pathway_absolute_tolerance,
        "shock_position_cells": args.shock_position_cell_tolerance,
        "shock_log_absolute": args.shock_log_tolerance,
        "shock_fraction_absolute": args.shock_fraction_tolerance,
    }
    if any(value <= 0.0 for value in tolerances.values()):
        raise ValueError("all repeatability tolerances must be positive")
    pilot_summary = _load_and_verify(args.pilot_results_dir)
    confirmation_summary = _load_and_verify(args.confirmation_results_dir)
    if pilot_summary.get("checks", {}).get("pathway_checks_passed") is not True:
        raise ValueError("pilot pathway closures did not pass")
    if (
        confirmation_summary.get("status") != "complete"
        or confirmation_summary.get("scientific_interpretation_allowed") is not True
    ):
        raise ValueError("confirmation result is not scientifically accepted")
    if _contract_projection(pilot_summary) != _contract_projection(
        confirmation_summary
    ):
        raise ValueError("replicate scientific contracts differ")

    pilot_pathway = _cohort_means(
        _read_rows(
            args.pilot_results_dir / "pathway_scale_region_aggregate_metrics.csv"
        ),
        ("target", "band", "region"),
    )
    confirmation_pathway = _cohort_means(
        _read_rows(
            args.confirmation_results_dir / "pathway_scale_region_aggregate_metrics.csv"
        ),
        ("target", "band", "region"),
    )
    pilot_shock = _cohort_means(
        _read_rows(args.pilot_results_dir / "shock_profile_aggregate_metrics.csv"),
        ("target", "comparison"),
    )
    confirmation_shock = _cohort_means(
        _read_rows(
            args.confirmation_results_dir / "shock_profile_aggregate_metrics.csv"
        ),
        ("target", "comparison"),
    )
    if pilot_pathway.keys() != confirmation_pathway.keys():
        raise ValueError("pathway aggregate cohorts differ")
    if pilot_shock.keys() != confirmation_shock.keys():
        raise ValueError("shock-profile aggregate cohorts differ")

    rows: list[dict[str, Any]] = []
    for key in sorted(pilot_pathway):
        for metric in PATHWAY_RELATIVE_METRICS:
            discrepancy, passed = _comparison(
                pilot_pathway[key][metric],
                confirmation_pathway[key][metric],
                mode="relative",
                tolerance=args.pathway_relative_tolerance,
            )
            rows.append(
                {
                    "family": "pathway",
                    "target": key[0],
                    "band": key[1],
                    "region_or_comparison": key[2],
                    "metric": metric,
                    "pilot_cohort_mean": pilot_pathway[key][metric],
                    "confirmation_cohort_mean": confirmation_pathway[key][metric],
                    "comparison": "relative",
                    "discrepancy": discrepancy,
                    "tolerance": args.pathway_relative_tolerance,
                    "passed": passed,
                }
            )
        for metric in PATHWAY_ABSOLUTE_METRICS:
            discrepancy, passed = _comparison(
                pilot_pathway[key][metric],
                confirmation_pathway[key][metric],
                mode="absolute",
                tolerance=args.pathway_absolute_tolerance,
            )
            rows.append(
                {
                    "family": "pathway",
                    "target": key[0],
                    "band": key[1],
                    "region_or_comparison": key[2],
                    "metric": metric,
                    "pilot_cohort_mean": pilot_pathway[key][metric],
                    "confirmation_cohort_mean": confirmation_pathway[key][metric],
                    "comparison": "absolute",
                    "discrepancy": discrepancy,
                    "tolerance": args.pathway_absolute_tolerance,
                    "passed": passed,
                }
            )
    for key in sorted(pilot_shock):
        coarse_nx = int(key[0].split("x", maxsplit=1)[0])
        coarse_dx = 2.0 / coarse_nx
        for metrics, mode, tolerance, scale in (
            (
                SHOCK_POSITION_METRICS,
                "absolute_cells",
                args.shock_position_cell_tolerance,
                coarse_dx,
            ),
            (SHOCK_LOG_METRICS, "absolute", args.shock_log_tolerance, 1.0),
            (
                SHOCK_FRACTION_METRICS,
                "absolute",
                args.shock_fraction_tolerance,
                1.0,
            ),
        ):
            for metric in metrics:
                discrepancy, passed = _comparison(
                    pilot_shock[key][metric],
                    confirmation_shock[key][metric],
                    mode="absolute",
                    tolerance=tolerance,
                    scale=scale,
                )
                rows.append(
                    {
                        "family": "shock_profile",
                        "target": key[0],
                        "band": "",
                        "region_or_comparison": key[1],
                        "metric": metric,
                        "pilot_cohort_mean": pilot_shock[key][metric],
                        "confirmation_cohort_mean": confirmation_shock[key][metric],
                        "comparison": mode,
                        "discrepancy": discrepancy,
                        "tolerance": tolerance,
                        "passed": passed,
                    }
                )

    args.output_dir.mkdir(parents=True)
    metrics_path = args.output_dir / "repeatability_metrics.csv"
    _write_csv(metrics_path, rows)
    passed = all(bool(row["passed"]) for row in rows)
    maxima = {
        comparison: max(
            float(row["discrepancy"]) for row in rows if row["comparison"] == comparison
        )
        for comparison in sorted({str(row["comparison"]) for row in rows})
    }
    summary = {
        "schema": SCHEMA,
        "status": "complete" if passed else "failed_repeatability",
        "scientific_interpretation_allowed": passed,
        "pilot_status_preserved": pilot_summary["status"],
        "pilot_scientific_interpretation_allowed_preserved": pilot_summary[
            "scientific_interpretation_allowed"
        ],
        "contract": _contract_projection(confirmation_summary),
        "tolerances": tolerances,
        "checks": {"maximum_discrepancy_by_comparison": maxima, "passed": passed},
        "row_count": len(rows),
        "input": {
            "pilot_summary_sha256": sha256_file(
                args.pilot_results_dir / "summary.json"
            ),
            "confirmation_summary_sha256": sha256_file(
                args.confirmation_results_dir / "summary.json"
            ),
        },
        "diagnostic_source_sha256": sha256_file(Path(__file__).resolve()),
        "output_hashes": {metrics_path.name: sha256_file(metrics_path)},
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    summary = run(parse_args(argv))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
