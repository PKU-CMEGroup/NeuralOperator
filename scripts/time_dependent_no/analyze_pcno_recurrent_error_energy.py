#!/usr/bin/env python3
"""Decompose recurrent correction benefit into additive error-energy views."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utility.time_dependent_no.pcno_artifacts import (
    atomic_write_json_with_paths as atomic_write_json,
)
from utility.time_dependent_no.pcno_artifacts import sha256_file, sha256_files
from utility.time_dependent_no.pcno_artifacts import (
    write_csv_with_paths as write_csv,
)
from utility.time_dependent_no.pcno_cross_resolution_teacher_forced import (
    verify_payload_sha256,
    with_payload_sha256,
)
from utility.time_dependent_no.pcno_residual_geometry import (
    ENERGY_COMPONENTS,
    ENERGY_LOCAL_REGIONS,
    recurrent_error_energy_rows,
)

WORKING_ID = "W26-L5-P6-RFB19-A25-RECURRENT-ERROR-ENERGY-PARTITION"
SCHEMA = "pcno_recurrent_error_energy_partition_v1"
SOURCE_PATHS = (
    "docs/time_dependent_no/W26_L5_RESPONSE_FILTERED_BLOCK_PREREGISTRATION.md",
    "utility/time_dependent_no/pcno_residual_geometry.py",
    "scripts/time_dependent_no/analyze_pcno_recurrent_error_energy.py",
    "tests/time_dependent_no/test_pcno_residual_geometry.py",
)
EXPECTED_HASHES = {
    "e12_result": "c6675dbb5481aea31a16fbd6aef8b7a71d7403a0fb199cd0120ff2cf343b868f",
    "e12_call_metrics": "6c736eb815987bd94a57e7a969e4b34d5b1706ca17d544c85b76e9a95a8c6f89",
    "e14_result": "aecc8a23c9b7c5e198b2948134ec2f901642684df9c66a04954308f31c9e195c",
    "e14_call_metrics": "6a60d6c7177cd3ef4d5e77df60636a6da992d66bea1f715369ee6126a79dfa2a",
}
EXPECTED_PAYLOAD_HASHES = {
    "e12": "bbe9abffbf1b4b699499eab72add357acf4ee6955a0fae9f1603579f0cd7f5dd",
    "e14": "8180306144be92f4cc059ba06dddcdf92fa674437fddee22da69b61ff46413ad",
}
POPULATIONS = {
    "e12": {
        "cases": tuple(f"sv_e12_y{index:02d}" for index in range(1, 8)),
        "policy": "relaxed_tether",
    },
    "e14": {
        "cases": tuple(f"sv_e14_y{index:02d}" for index in range(2, 7)),
        "policy": "buffered_relaxed_tether",
    },
}
BANDS = ((9, 14), (15, 20), (21, 26), (27, 30))
ADDITIVE_VIEWS = ("full", "rank8", "remainder")
COMPONENT_VIEWS = tuple(f"component_{name}" for name in ENERGY_COMPONENTS)
LOCAL_VIEWS = tuple(f"local_{name}" for name in ENERGY_LOCAL_REGIONS)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON payload must be an object: {path}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _git_status_short(paths: Sequence[str]) -> list[str]:
    try:
        result = subprocess.run(
            ["git", "status", "--short", "--", *paths],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return ["unknown"]
    return result.stdout.splitlines() if result.returncode == 0 else ["unknown"]


def _verify_inputs(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "e12_result": args.e12_result,
        "e12_call_metrics": args.e12_call_metrics,
        "e14_result": args.e14_result,
        "e14_call_metrics": args.e14_call_metrics,
    }
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    if hashes != EXPECTED_HASHES:
        raise ValueError("A25 input hashes differ from the frozen artifacts")
    results = {name: _read_json(getattr(args, f"{name}_result")) for name in POPULATIONS}
    for name, payload in results.items():
        verify_payload_sha256(payload)
        if payload.get("payload_sha256") != EXPECTED_PAYLOAD_HASHES[name]:
            raise ValueError(f"{name} payload differs from the frozen artifact")
        if payload.get("artifact_hashes", {}).get("rollout_call_metrics.csv") != (
            EXPECTED_HASHES[f"{name}_call_metrics"]
        ):
            raise ValueError(f"{name} result does not own its call metrics")
    return {"hashes": hashes, "results": results}


def _case_first_mean(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    cases = sorted({str(row["case_id"]) for row in rows})
    return float(
        np.mean(
            [
                np.mean(
                    [float(row[key]) for row in rows if row["case_id"] == case_id]
                )
                for case_id in cases
            ]
        )
    )


def _summary_row(
    rows: Sequence[Mapping[str, Any]],
    *,
    population: str,
    scope: str,
    first_output_call: int | None = None,
    last_output_call: int | None = None,
) -> dict[str, Any]:
    output = {
        "population": population,
        "scope": scope,
        "first_output_call": first_output_call,
        "last_output_call": last_output_call,
        "case_count": len({row["case_id"] for row in rows}),
        "row_count": len(rows),
    }
    for view in (*ADDITIVE_VIEWS, *COMPONENT_VIEWS, *LOCAL_VIEWS):
        key = f"{view}_benefit"
        output[f"{view}_case_first_mean_benefit"] = _case_first_mean(rows, key)
        output[f"{view}_help_count"] = sum(float(row[key]) > 0.0 for row in rows)
        output[f"{view}_harm_count"] = sum(float(row[key]) < 0.0 for row in rows)
    resolved = [
        float(row["rank8_benefit_share"])
        for row in rows
        if row["benefit_share_status"] == "ok"
    ]
    output["mean_rank8_benefit_share_over_resolved_rows"] = (
        float(np.mean(resolved)) if resolved else None
    )
    output["unresolved_benefit_share_count"] = len(rows) - len(resolved)
    return output


def _case_summaries(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for case_id in sorted({str(row["case_id"]) for row in rows}):
        selected = [row for row in rows if row["case_id"] == case_id]
        harmful = [int(row["output_call"]) for row in selected if row["full_benefit"] < 0.0]
        row = {
            "population": selected[0]["population"],
            "case_id": case_id,
            "full_help_count": sum(item["full_benefit"] > 0.0 for item in selected),
            "full_harm_count": len(harmful),
            "first_full_harmful_call": harmful[0] if harmful else None,
            "last_full_harmful_call": harmful[-1] if harmful else None,
        }
        for view in (*ADDITIVE_VIEWS, *COMPONENT_VIEWS):
            row[f"{view}_mean_benefit"] = float(
                np.mean([float(item[f"{view}_benefit"]) for item in selected])
            )
        output.append(row)
    return output


def _harm_classification(row: Mapping[str, Any]) -> str:
    rank8_harm = float(row["rank8_benefit"]) < 0.0
    remainder_harm = float(row["remainder_benefit"]) < 0.0
    if rank8_harm and remainder_harm:
        return "rank8_and_remainder_harm"
    if rank8_harm:
        return "rank8_harm_remainder_helps"
    if remainder_harm:
        return "remainder_harm_rank8_helps"
    return "unresolved_zero_or_closure"  # pragma: no cover - full harm forbids this


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    parents = _verify_inputs(args)
    populations = {
        name: recurrent_error_energy_rows(
            _read_csv(getattr(args, f"{name}_call_metrics")),
            population=name,
            expected_cases=contract["cases"],
            corrected_policy=contract["policy"],
        )
        for name, contract in POPULATIONS.items()
    }
    rows = [row for population in populations.values() for row in population]
    summaries = []
    cases = []
    for name, population in populations.items():
        summaries.append(_summary_row(population, population=name, scope="population"))
        for start, stop in BANDS:
            selected = [row for row in population if start <= row["output_call"] <= stop]
            summaries.append(
                _summary_row(
                    selected,
                    population=name,
                    scope="temporal_band",
                    first_output_call=start,
                    last_output_call=stop,
                )
            )
        cases.extend(_case_summaries(population))

    e14_harm = [row for row in populations["e14"] if row["full_benefit"] < 0.0]
    harm_rows = [
        {
            "population": row["population"],
            "case_id": row["case_id"],
            "output_call": row["output_call"],
            "classification": _harm_classification(row),
            **{
                f"{view}_benefit": row[f"{view}_benefit"]
                for view in (*ADDITIVE_VIEWS, *COMPONENT_VIEWS, *LOCAL_VIEWS)
            },
        }
        for row in e14_harm
    ]
    checks = {
        "input_hashes_exact": parents["hashes"] == EXPECTED_HASHES,
        "parent_payloads_exact": True,
        "e12_inventory_exact_7x22": len(populations["e12"]) == 154,
        "e14_inventory_exact_5x22": len(populations["e14"]) == 110,
        "maximum_component_energy_closure_at_most_1e_12": max(
            row["maximum_component_energy_closure_abs"] for row in rows
        )
        <= 1.0e-12,
        "maximum_rank8_remainder_benefit_closure_at_most_1e_12": max(
            row["rank8_remainder_benefit_closure_abs"] for row in rows
        )
        <= 1.0e-12,
        "remainder_energy_nonnegative": all(
            row[f"{arm}_remainder_energy"] >= 0.0
            for row in rows
            for arm in ("raw", "corrected")
        ),
        "e14_full_harm_inventory_exact_five": len(e14_harm) == 5,
        "no_model_built": True,
        "no_state_or_reference_array_loaded": True,
        "recurrence_not_executed": True,
        "no_fit_or_selector": True,
    }
    if not all(checks.values()):
        raise AssertionError(f"A25 validity checks failed: {checks}")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_csv(args.output_dir / "energy_rows.csv", rows)
    write_csv(args.output_dir / "energy_summaries.csv", summaries)
    write_csv(args.output_dir / "case_energy.csv", cases)
    write_csv(args.output_dir / "e14_terminal_harm.csv", harm_rows)
    classification_counts = {
        label: sum(row["classification"] == label for row in harm_rows)
        for label in sorted({row["classification"] for row in harm_rows})
    }
    payload = with_payload_sha256(
        {
            "schema": SCHEMA,
            "working_id": WORKING_ID,
            "status": "completed_retrospective_diagnostic",
            "source_hashes": sha256_files(SOURCE_PATHS, root=ROOT),
            "source_status": _git_status_short(SOURCE_PATHS),
            "input_hashes": parents["hashes"],
            "parent_payload_hashes": {
                name: payload["payload_sha256"]
                for name, payload in parents["results"].items()
            },
            "contract": {
                "populations": POPULATIONS,
                "output_calls": list(range(9, 31)),
                "temporal_bands": [list(band) for band in BANDS],
                "additive_views": list(ADDITIVE_VIEWS),
                "component_views": list(COMPONENT_VIEWS),
                "local_views": list(LOCAL_VIEWS),
                "benefit_sign": "raw squared RMS minus corrected squared RMS",
                "weighting": "equal calls within case then equal cases",
                "local_region_metrics_additive": False,
                "denominator_floor": 1.0e-12,
            },
            "checks": checks,
            "population_summaries": {
                row["population"]: row
                for row in summaries
                if row["scope"] == "population"
            },
            "e14_terminal_harm": {
                "row_count": len(harm_rows),
                "classification_counts": classification_counts,
                "rows": harm_rows,
            },
            "artifact_hashes": {
                name: sha256_file(args.output_dir / name)
                for name in (
                    "energy_rows.csv",
                    "energy_summaries.csv",
                    "case_energy.csv",
                    "e14_terminal_harm.csv",
                )
            },
            "claim_boundary": (
                "Retrospective zero-model additive state-error-energy decomposition "
                "on already-inspected E12/E14 scalar metrics only. Rank-8 is not "
                "the exact SP19 mask; local region RMS values are not globally "
                "additive. No selector, rollout, conservation, convergence, bump, "
                "direct off-grid, or Richardson-extrapolation claim is authorized."
            ),
        }
    )
    atomic_write_json(args.output_dir / "recurrent_error_energy.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e12-result", type=Path, required=True)
    parser.add_argument("--e12-call-metrics", type=Path, required=True)
    parser.add_argument("--e14-result", type=Path, required=True)
    parser.add_argument("--e14-call-metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    payload = analyze(parse_args())
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
