#!/usr/bin/env python3
"""Analyze recurrent cross-resolution benefit phase from immutable rollout CSVs."""

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
    PHASE_MODEL_FEATURES,
    fit_phase_ridge,
    leave_one_case_out_phase,
    predict_phase_ridge,
    recurrent_phase_records,
    score_phase_predictions,
)

WORKING_ID = "W26-L5-P6-RFB19-A24-RECURRENT-PHASE-GEOMETRY"
SCHEMA = "pcno_recurrent_phase_geometry_v1"
SOURCE_PATHS = (
    "docs/time_dependent_no/W26_L5_RESPONSE_FILTERED_BLOCK_PREREGISTRATION.md",
    "utility/time_dependent_no/pcno_residual_geometry.py",
    "scripts/time_dependent_no/analyze_pcno_recurrent_phase_geometry.py",
    "tests/time_dependent_no/test_pcno_residual_geometry.py",
)
EXPECTED_HASHES = {
    "e12_result": "c6675dbb5481aea31a16fbd6aef8b7a71d7403a0fb199cd0120ff2cf343b868f",
    "e12_call_metrics": "6c736eb815987bd94a57e7a969e4b34d5b1706ca17d544c85b76e9a95a8c6f89",
    "e12_tether_audits": "8542fe2606cb87508c73a31ce33582c9621a8810eeeebce8874e79746ae884f1",
    "e14_result": "aecc8a23c9b7c5e198b2948134ec2f901642684df9c66a04954308f31c9e195c",
    "e14_call_metrics": "6a60d6c7177cd3ef4d5e77df60636a6da992d66bea1f715369ee6126a79dfa2a",
    "e14_tether_audits": "6a0b2b31c7cd9a21692647ee10363c60832647fdec13853c28a5cec3fb83ba2b",
}
EXPECTED_PAYLOAD_HASHES = {
    "e12_result": "bbe9abffbf1b4b699499eab72add357acf4ee6955a0fae9f1603579f0cd7f5dd",
    "e14_result": "8180306144be92f4cc059ba06dddcdf92fa674437fddee22da69b61ff46413ad",
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


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON payload must be an object: {path}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


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


def _verify_inputs(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    paths = {
        "e12_result": args.e12_result,
        "e12_call_metrics": args.e12_call_metrics,
        "e12_tether_audits": args.e12_tether_audits,
        "e14_result": args.e14_result,
        "e14_call_metrics": args.e14_call_metrics,
        "e14_tether_audits": args.e14_tether_audits,
    }
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    if hashes != EXPECTED_HASHES:
        raise ValueError("A24 input hashes differ from the frozen artifacts")
    results = {
        "e12": _read_json(args.e12_result),
        "e14": _read_json(args.e14_result),
    }
    for population, payload in results.items():
        verify_payload_sha256(payload)
        if payload.get("payload_sha256") != EXPECTED_PAYLOAD_HASHES[
            f"{population}_result"
        ]:
            raise ValueError(f"{population} payload differs from the frozen artifact")
        expected_call_hash = EXPECTED_HASHES[f"{population}_call_metrics"]
        if payload.get("artifact_hashes", {}).get("rollout_call_metrics.csv") != (
            expected_call_hash
        ):
            raise ValueError(f"{population} result does not own its call metrics")
        expected_audit_hash = EXPECTED_HASHES[f"{population}_tether_audits"]
        if payload.get("artifact_hashes", {}).get("relaxed_tether_audits.csv") != (
            expected_audit_hash
        ):
            raise ValueError(f"{population} result does not own its tether audits")
    return {"hashes": hashes, "results": results}


def _record_row(record: Any) -> dict[str, Any]:
    return {
        "population": record.population,
        "case_id": record.case_id,
        "output_call": record.output_call,
        "raw_state_error": record.raw_state_error,
        "corrected_state_error": record.corrected_state_error,
        "benefit": record.benefit,
        **record.features,
    }


def _case_first_mean(records: Sequence[Any], key) -> float:
    cases = sorted({record.case_id for record in records})
    return float(
        np.mean(
            [
                np.mean([key(record) for record in records if record.case_id == case])
                for case in cases
            ]
        )
    )


def _population_descriptions(records: Sequence[Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    bands = []
    cases = []
    for start, stop in BANDS:
        selected = [record for record in records if start <= record.output_call <= stop]
        bands.append(
            {
                "population": selected[0].population,
                "first_output_call": start,
                "last_output_call": stop,
                "case_first_mean_benefit": _case_first_mean(
                    selected, lambda record: record.benefit
                ),
                "case_first_help_fraction": _case_first_mean(
                    selected, lambda record: float(record.benefit > 0.0)
                ),
            }
        )
    for case_id in sorted({record.case_id for record in records}):
        selected = [record for record in records if record.case_id == case_id]
        harmful = [record.output_call for record in selected if record.benefit < 0.0]
        cases.append(
            {
                "population": selected[0].population,
                "case_id": case_id,
                "mean_benefit": float(np.mean([record.benefit for record in selected])),
                "help_count": sum(record.benefit > 0.0 for record in selected),
                "harm_count": len(harmful),
                "first_harmful_call": harmful[0] if harmful else None,
                "last_harmful_call": harmful[-1] if harmful else None,
            }
        )
    return bands, cases


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    parents = _verify_inputs(args)
    records = {}
    for population in ("e12", "e14"):
        records[population] = recurrent_phase_records(
            _read_csv(getattr(args, f"{population}_call_metrics")),
            _read_csv(getattr(args, f"{population}_tether_audits")),
            population=population,
            expected_cases=POPULATIONS[population]["cases"],
            corrected_policy=POPULATIONS[population]["policy"],
        )

    row_output = [_record_row(record) for population in records.values() for record in population]
    band_output = []
    case_output = []
    for population in records.values():
        bands, cases = _population_descriptions(population)
        band_output.extend(bands)
        case_output.extend(cases)

    model_output = []
    crossfit = {}
    transfer = {}
    for model, features in PHASE_MODEL_FEATURES.items():
        crossfit[model] = {}
        for population in ("e12", "e14"):
            diagnostic = leave_one_case_out_phase(
                records[population], feature_names=features
            )
            crossfit[model][population] = diagnostic
            model_output.append(
                {
                    "evaluation": "leave_one_case_out",
                    "model": model,
                    "train_population": population,
                    "test_population": population,
                    **diagnostic["score"],
                }
            )
        frozen_fit = fit_phase_ridge(records["e12"], feature_names=features)
        predictions = predict_phase_ridge(frozen_fit, records["e14"])
        score = score_phase_predictions(records["e14"], predictions)
        transfer[model] = {"fit": frozen_fit, "score": score}
        model_output.append(
            {
                "evaluation": "frozen_e12_to_e14",
                "model": model,
                "train_population": "e12",
                "test_population": "e14",
                **score,
            }
        )

    combined = transfer["combined"]["score"]
    time_only = transfer["time_only"]["score"]
    stability = crossfit["combined"]["e12"]["coefficient_stability"]
    motivation_checks = {
        "combined_transfer_r2_positive": combined["r2_vs_zero"] is not None
        and combined["r2_vs_zero"] > 0.0,
        "combined_transfer_signed_cosine_positive": (
            combined["signed_cosine"] is not None and combined["signed_cosine"] > 0.0
        ),
        "combined_transfer_sign_accuracy_above_0p6": (
            combined["sign_accuracy"] is not None
            and combined["sign_accuracy"] > 0.6
        ),
        "combined_r2_exceeds_time_only_by_0p05": (
            combined["r2_vs_zero"] is not None
            and time_only["r2_vs_zero"] is not None
            and combined["r2_vs_zero"] >= time_only["r2_vs_zero"] + 0.05
        ),
        "e12_combined_feature_signs_stable": all(
            row["same_nonzero_sign"] for row in stability.values()
        ),
    }
    checks = {
        "input_hashes_exact": parents["hashes"] == EXPECTED_HASHES,
        "parent_payloads_exact": True,
        "e12_inventory_exact_7x22": len(records["e12"]) == 154,
        "e14_inventory_exact_5x22": len(records["e14"]) == 110,
        "populations_not_pooled": all(
            record.population == population
            for population, population_records in records.items()
            for record in population_records
        ),
        "all_features_and_targets_finite": all(
            np.isfinite([record.benefit, *record.features.values()]).all()
            for population in records.values()
            for record in population
        ),
        "state_correction_polarization_not_computed": True,
        "true_error_not_an_inference_feature": all(
            name not in feature
            for features in PHASE_MODEL_FEATURES.values()
            for feature in features
            for name in ("benefit", "error", "truth", "reference")
        ),
        "no_model_built": True,
        "no_state_or_reference_array_loaded": True,
        "recurrence_not_executed": True,
        "no_arm_switch_rollout_reconstructed": True,
    }
    if not all(checks.values()):
        raise AssertionError(f"A24 validity checks failed: {checks}")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_csv(args.output_dir / "phase_rows.csv", row_output)
    write_csv(args.output_dir / "temporal_bands.csv", band_output)
    write_csv(args.output_dir / "case_harm.csv", case_output)
    write_csv(args.output_dir / "model_scores.csv", model_output)
    payload = with_payload_sha256(
        _json_safe(
            {
                "schema": SCHEMA,
                "working_id": WORKING_ID,
                "status": "completed_retrospective_diagnostic",
                "source_hashes": sha256_files(SOURCE_PATHS, root=ROOT),
                "source_status": _git_status_short(SOURCE_PATHS),
                "input_hashes": parents["hashes"],
                "parent_payload_hashes": {
                    population: payload["payload_sha256"]
                    for population, payload in parents["results"].items()
                },
                "contract": {
                    "populations": POPULATIONS,
                    "output_calls": list(range(9, 31)),
                    "temporal_bands": [list(band) for band in BANDS],
                    "normalized_time": "output_call / 30",
                    "target": "1 - corrected_state_error^2 / raw_state_error^2",
                    "weighting": "equal calls within case then equal cases",
                    "models": PHASE_MODEL_FEATURES,
                    "ridge": 1.0e-6,
                    "denominator_floor": 1.0e-8,
                    "state_correction_polarization": (
                        "invalid and not computed because state and correction RMS "
                        "use different component scales"
                    ),
                },
                "checks": checks,
                "crossfit": crossfit,
                "frozen_e12_to_e14": transfer,
                "motivation_condition": {
                    "checks": motivation_checks,
                    "passed": all(motivation_checks.values()),
                    "authorizes_execution": False,
                },
                "artifact_hashes": {
                    name: sha256_file(args.output_dir / name)
                    for name in (
                        "phase_rows.csv",
                        "temporal_bands.csv",
                        "case_harm.csv",
                        "model_scores.csv",
                    )
                },
                "claim_boundary": (
                    "Retrospective zero-model recurrent-phase geometry on already-"
                    "inspected E12/E14 only. True-error benefit is diagnostic and "
                    "unavailable at inference. No selector, rollout, cross-family "
                    "coefficient, conservation, convergence, bump, direct off-grid, "
                    "or Richardson-extrapolation claim is authorized."
                ),
            }
        )
    )
    atomic_write_json(args.output_dir / "recurrent_phase_geometry.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e12-result", type=Path, required=True)
    parser.add_argument("--e12-call-metrics", type=Path, required=True)
    parser.add_argument("--e12-tether-audits", type=Path, required=True)
    parser.add_argument("--e14-result", type=Path, required=True)
    parser.add_argument("--e14-call-metrics", type=Path, required=True)
    parser.add_argument("--e14-tether-audits", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    payload = analyze(parse_args())
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
