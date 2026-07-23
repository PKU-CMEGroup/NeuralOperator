"""Analyze frozen Line-1 stride-attribution artifacts without checkpoint replay.

This companion to ``diagnose_euler1d_stride_attribution.py`` consumes only
already generated D032/D033/D035/D036 tables. It freezes the registered failure
and matched-control cohort, measures temporal precedence, and checks whether
truth-state or on-policy same-state defects distinguish the failures.

No model, reference solver, test split, or GPU is accessed by this script.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.time_dependent_no.diagnose_euler1d_stride_attribution import (  # noqa: E402
    event_start_frames,
    read_csv_rows,
    read_jsonl,
    select_cohort,
)
from scripts.time_dependent_no.evaluate_euler1d_flow_map_frontier import (  # noqa: E402
    _write_csv,
)
from scripts.time_dependent_no.train_euler1d_target_ladder import (  # noqa: E402
    json_ready,
    sha256_file,
)

SCHEMA = "euler1d_stride_attribution_frozen_analysis_v1"
PRIMARY_MODELS = {1: "s1", 2: "s2", 4: "s4", 8: "s8_seed20260707"}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct-curves", type=Path, required=True)
    parser.add_argument("--descriptors", type=Path, required=True)
    parser.add_argument("--same-state", type=Path, required=True)
    parser.add_argument("--ripple-metrics", type=Path, required=True)
    parser.add_argument("--full-batch-metrics", type=Path, required=True)
    parser.add_argument("--full-batch-terminations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-failures", type=int, default=4)
    parser.add_argument("--max-controls", type=int, default=2)
    parser.add_argument("--lookback-macros", type=int, default=2)
    args = parser.parse_args(argv)
    if not 1 <= args.max_failures <= 4:
        parser.error("--max-failures must lie in [1,4]")
    if not 1 <= args.max_controls <= 4:
        parser.error("--max-controls must lie in [1,4]")
    if args.lookback_macros < 0:
        parser.error("--lookback-macros must be nonnegative")
    if args.output_dir.exists():
        parser.error("--output-dir must not already exist")
    return args


def _float(row: Mapping[str, Any], name: str) -> float:
    value = float(row[name])
    return value if math.isfinite(value) else float("nan")


def _median(values: Sequence[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(statistics.median(finite)) if finite else float("nan")


def _first_event(
    rows: Sequence[Mapping[str, Any]],
    metric: str,
    threshold: float,
    *,
    relation: str,
) -> int | None:
    if relation not in {"above", "below"}:
        raise ValueError("event relation must be 'above' or 'below'")
    ordered = sorted(rows, key=lambda row: int(row["target_frame"]))
    for row in ordered:
        value = _float(row, metric)
        crossed = value > threshold if relation == "above" else value < threshold
        if crossed:
            return int(row["target_frame"])
    return None


def _case_entries(cohort: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [dict(row) for row in (*cohort["failures"], *cohort["controls"])]


def build_same_state_selection(
    cohort: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    lookback_macros: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    index = {
        (
            int(row["case_id"]),
            int(row["start_frame"]),
            int(row["stride"]),
            str(row["state_source"]),
        ): row
        for row in rows
    }
    for entry in _case_entries(cohort):
        final_start = event_start_frames(
            int(entry["matched_failure_frame"]),
            lookback_macros=0,
        )[0]
        for start_frame in event_start_frames(
            int(entry["matched_failure_frame"]),
            lookback_macros=lookback_macros,
        ):
            for stride in (1, 8):
                truth = index[(int(entry["case_id"]), start_frame, stride, "truth")]
                on_policy = index[
                    (int(entry["case_id"]), start_frame, stride, "on_policy")
                ]
                truth_defect = _float(
                    truth,
                    "model_reference_cons_scaled_rel_l2",
                )
                on_policy_defect = _float(
                    on_policy,
                    "model_reference_cons_scaled_rel_l2",
                )
                current_error = _float(
                    on_policy,
                    "current_truth_cons_scaled_rel_l2",
                )
                reference_error = _float(
                    on_policy,
                    "reference_truth_cons_scaled_rel_l2",
                )
                selected.append(
                    {
                        **entry,
                        "start_frame": start_frame,
                        "event_offset_frames": start_frame - final_start,
                        "stride": stride,
                        "truth_state_model_defect": truth_defect,
                        "on_policy_model_defect": on_policy_defect,
                        "on_policy_to_truth_defect_ratio": (
                            on_policy_defect / max(truth_defect, 1.0e-30)
                        ),
                        "on_policy_model_defect_per_physical_time": _float(
                            on_policy,
                            "model_reference_cons_scaled_rel_l2_per_physical_time",
                        ),
                        "current_state_error": current_error,
                        "reference_propagation_gain": (
                            reference_error / max(current_error, 1.0e-30)
                        ),
                        "same_state_correction_toward_truth_cosine": _float(
                            on_policy,
                            "correction_toward_truth_cosine",
                        ),
                        "on_policy_shock_model_defect": _float(
                            on_policy,
                            "model_reference_shock_cons_scaled_rel_l2",
                        ),
                        "on_policy_smooth_model_defect": _float(
                            on_policy,
                            "model_reference_smooth_cons_scaled_rel_l2",
                        ),
                    }
                )
    return selected


def summarize_same_state(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    metrics = (
        "truth_state_model_defect",
        "on_policy_model_defect",
        "on_policy_to_truth_defect_ratio",
        "on_policy_model_defect_per_physical_time",
        "current_state_error",
        "reference_propagation_gain",
        "same_state_correction_toward_truth_cosine",
        "on_policy_shock_model_defect",
        "on_policy_smooth_model_defect",
    )
    output: list[dict[str, Any]] = []
    for role in ("failure", "stable_control"):
        for stride in (1, 8):
            group = [
                row
                for row in rows
                if row["role"] == role and int(row["stride"]) == stride
            ]
            summary: dict[str, Any] = {
                "role": role,
                "stride": stride,
                "count": len(group),
                "case_count": len({int(row["case_id"]) for row in group}),
            }
            for metric in metrics:
                summary[f"{metric}_median"] = _median(
                    [_float(row, metric) for row in group]
                )
            output.append(summary)
    return output


def build_precursor_rows(
    cohort: Mapping[str, Any],
    ripple_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for entry in _case_entries(cohort):
        case_id = int(entry["case_id"])
        rollout = sorted(
            (
                row
                for row in ripple_rows
                if int(row["case_id"]) == case_id
                and row["model"] == "s1"
                and row["mode"] == "autoregressive"
            ),
            key=lambda row: int(row["target_frame"]),
        )
        teacher = [
            row
            for row in ripple_rows
            if int(row["case_id"]) == case_id
            and row["model"] == "s1"
            and row["mode"] == "teacher_forced"
        ]
        if not rollout or not teacher:
            raise ValueError(f"missing D036 rows for case {case_id}")
        invalid = [row for row in rollout if row["proposal_valid"] == "False"]
        last_valid = next(
            row for row in reversed(rollout) if row["proposal_valid"] == "True"
        )
        state_005 = _first_event(
            rollout,
            "state_cons_scaled_rel_l2",
            0.05,
            relation="above",
        )
        highpass_002 = _first_event(
            rollout,
            "error_high_25_64_rms",
            0.02,
            relation="above",
        )
        pressure_01 = _first_event(
            rollout,
            "min_pressure",
            0.1,
            relation="below",
        )
        output.append(
            {
                **entry,
                "batch1_failure_frame": (
                    int(invalid[0]["target_frame"]) if invalid else None
                ),
                "full_batch_failure_frame": (
                    int(entry["failure_frame"]) if entry["role"] == "failure" else None
                ),
                "first_state_error_above_0p02": _first_event(
                    rollout,
                    "state_cons_scaled_rel_l2",
                    0.02,
                    relation="above",
                ),
                "first_state_error_above_0p05": state_005,
                "first_state_error_above_0p10": _first_event(
                    rollout,
                    "state_cons_scaled_rel_l2",
                    0.1,
                    relation="above",
                ),
                "first_highpass_rms_above_0p02": highpass_002,
                "first_pressure_below_0p10": pressure_01,
                "first_pressure_below_0p05": _first_event(
                    rollout,
                    "min_pressure",
                    0.05,
                    relation="below",
                ),
                "highpass_lead_before_pressure_0p10": (
                    None
                    if highpass_002 is None or pressure_01 is None
                    else pressure_01 - highpass_002
                ),
                "state_0p05_lead_before_pressure_0p10": (
                    None
                    if state_005 is None or pressure_01 is None
                    else pressure_01 - state_005
                ),
                "last_valid_frame": int(last_valid["target_frame"]),
                "last_valid_state_error": _float(
                    last_valid,
                    "state_cons_scaled_rel_l2",
                ),
                "last_valid_highpass_rms": _float(
                    last_valid,
                    "error_high_25_64_rms",
                ),
                "last_valid_smooth_error_d2_rms": _float(
                    last_valid,
                    "smooth_error_d2_rms",
                ),
                "last_valid_min_pressure": _float(last_valid, "min_pressure"),
                "teacher_state_error_median": _median(
                    [_float(row, "state_cons_scaled_rel_l2") for row in teacher]
                ),
                "teacher_highpass_rms_median": _median(
                    [_float(row, "error_high_25_64_rms") for row in teacher]
                ),
            }
        )
    return output


def batch_presentation_audit(
    full_batch_rows: Sequence[Mapping[str, Any]],
    ripple_rows: Sequence[Mapping[str, Any]],
    termination_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    full = {
        (int(row["case_id"]), int(row["stride"]), int(row["frame"])): _float(
            row,
            "fixed_scale_conservative_relative_l2",
        )
        for row in full_batch_rows
        if str(row.get("fixed_scale_conservative_relative_l2", "")).strip()
    }
    batch1 = {
        (
            int(row["case_id"]),
            stride,
            int(row["target_frame"]),
        ): _float(row, "state_cons_scaled_rel_l2")
        for row in ripple_rows
        for stride, model in PRIMARY_MODELS.items()
        if row["mode"] == "autoregressive"
        and row["model"] == model
        and int(row["target_frame"]) % 8 == 0
        and row["proposal_valid"] == "True"
    }
    common = sorted(set(full).intersection(batch1))
    absolute = np.asarray([abs(full[key] - batch1[key]) for key in common])
    relative = np.asarray(
        [abs(full[key] - batch1[key]) / max(abs(full[key]), 1.0e-30) for key in common]
    )
    full_failures = {
        (int(row["case_id"]), int(row["stride"])): int(row["termination_frame"])
        for row in termination_rows
        if str(row.get("termination_frame", "")).strip()
        and math.isfinite(float(row["termination_frame"]))
    }
    batch1_failures = {
        (int(row["case_id"]), stride): int(row["target_frame"])
        for row in ripple_rows
        for stride, model in PRIMARY_MODELS.items()
        if row["mode"] == "autoregressive"
        and row["model"] == model
        and row["proposal_valid"] == "False"
    }
    mismatches = [
        {
            "case_id": case_id,
            "stride": stride,
            "full_batch_failure_frame": full_failures[(case_id, stride)],
            "batch1_failure_frame": batch1_failures[(case_id, stride)],
        }
        for case_id, stride in sorted(set(full_failures).intersection(batch1_failures))
        if full_failures[(case_id, stride)] != batch1_failures[(case_id, stride)]
    ]
    return {
        "common_metric_rows": len(common),
        "absolute_difference_median": float(np.median(absolute)),
        "absolute_difference_max": float(np.max(absolute)),
        "relative_difference_median": float(np.median(relative)),
        "relative_difference_max": float(np.max(relative)),
        "failure_frame_mismatches": mismatches,
        "interpretation": (
            "small batch-presentation numerical differences can shift a terminal "
            "pressure crossing; they do not change the 62/64 completion count"
        ),
    }


def _summary_lookup(
    rows: Sequence[Mapping[str, Any]],
    role: str,
    stride: int,
) -> Mapping[str, Any]:
    return next(
        row for row in rows if row["role"] == role and int(row["stride"]) == stride
    )


def build_decision(
    same_state_summary: Sequence[Mapping[str, Any]],
    precursor_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    failure_s1 = _summary_lookup(same_state_summary, "failure", 1)
    control_s1 = _summary_lookup(same_state_summary, "stable_control", 1)
    failure_s8 = _summary_lookup(same_state_summary, "failure", 8)
    failure_precursors = [row for row in precursor_rows if row["role"] == "failure"]
    values = {
        "failure_s1_on_policy_to_truth_defect_ratio_median": _float(
            failure_s1,
            "on_policy_to_truth_defect_ratio_median",
        ),
        "control_s1_on_policy_to_truth_defect_ratio_median": _float(
            control_s1,
            "on_policy_to_truth_defect_ratio_median",
        ),
        "failure_to_control_shift_enrichment": (
            _float(failure_s1, "on_policy_to_truth_defect_ratio_median")
            / max(
                _float(control_s1, "on_policy_to_truth_defect_ratio_median"),
                1.0e-30,
            )
        ),
        "failure_s1_to_s8_model_defect_per_time_ratio": (
            _float(
                failure_s1,
                "on_policy_model_defect_per_physical_time_median",
            )
            / max(
                _float(
                    failure_s8,
                    "on_policy_model_defect_per_physical_time_median",
                ),
                1.0e-30,
            )
        ),
        "failure_s1_reference_propagation_gain_median": _float(
            failure_s1,
            "reference_propagation_gain_median",
        ),
        "failure_to_control_truth_state_defect_ratio": (
            _float(failure_s1, "truth_state_model_defect_median")
            / max(
                _float(control_s1, "truth_state_model_defect_median"),
                1.0e-30,
            )
        ),
        "failure_highpass_leads_pressure_all": all(
            row["highpass_lead_before_pressure_0p10"] is not None
            and int(row["highpass_lead_before_pressure_0p10"]) >= 8
            for row in failure_precursors
        ),
        "failure_state_error_leads_pressure_all": all(
            row["state_0p05_lead_before_pressure_0p10"] is not None
            and int(row["state_0p05_lead_before_pressure_0p10"]) >= 8
            for row in failure_precursors
        ),
    }
    criteria = {
        "truth_state_fit_not_failure_specific": (
            values["failure_to_control_truth_state_defect_ratio"] <= 2.0
        ),
        "on_policy_defect_shift_enriched": (
            values["failure_s1_on_policy_to_truth_defect_ratio_median"] >= 5.0
            and values["failure_to_control_shift_enrichment"] >= 2.0
        ),
        "reference_flow_not_explosive": (
            values["failure_s1_reference_propagation_gain_median"] <= 1.25
        ),
        "pressure_is_terminal": (
            values["failure_highpass_leads_pressure_all"]
            and values["failure_state_error_leads_pressure_all"]
        ),
    }
    supported = all(criteria.values())
    return {
        "classification": (
            "recurrent_off_manifold_defect_with_terminal_pressure_supported_"
            "observationally_crossed_operator_fork_pending"
            if supported
            else "mixed_or_unresolved"
        ),
        "values": values,
        "criteria": criteria,
        "claim_boundary": (
            "The frozen same-state reference evidence supports recurrent "
            "distribution shift within the two failure cases. Missing crossed "
            "G1/G8 state-operator forks block a stronger map-versus-state causal "
            "claim; all cases contain shocks, so discontinuity necessity is untested."
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    direct_rows = read_jsonl(args.direct_curves)
    descriptor_rows = read_csv_rows(args.descriptors)
    same_state_rows = read_csv_rows(args.same_state)
    ripple_rows = read_csv_rows(args.ripple_metrics)
    full_batch_rows = read_csv_rows(args.full_batch_metrics)
    termination_rows = read_csv_rows(args.full_batch_terminations)
    cohort = select_cohort(
        direct_rows,
        descriptor_rows,
        max_failures=args.max_failures,
        max_controls=args.max_controls,
    )
    selected_same_state = build_same_state_selection(
        cohort,
        same_state_rows,
        lookback_macros=args.lookback_macros,
    )
    same_state_summary = summarize_same_state(selected_same_state)
    precursor_rows = build_precursor_rows(cohort, ripple_rows)
    batch_audit = batch_presentation_audit(
        full_batch_rows,
        ripple_rows,
        termination_rows,
    )
    decision = build_decision(same_state_summary, precursor_rows)
    report = {
        "schema": SCHEMA,
        "status": "complete",
        "training_runs": 0,
        "checkpoint_calls": 0,
        "reference_solver_calls": 0,
        "cohort": cohort,
        "decision": decision,
        "batch_presentation_audit": batch_audit,
        "source_sha256": {
            "direct_curves": sha256_file(args.direct_curves),
            "descriptors": sha256_file(args.descriptors),
            "same_state": sha256_file(args.same_state),
            "ripple_metrics": sha256_file(args.ripple_metrics),
            "full_batch_metrics": sha256_file(args.full_batch_metrics),
            "full_batch_terminations": sha256_file(args.full_batch_terminations),
            "script": sha256_file(Path(__file__).resolve()),
        },
        "same_state_summary": same_state_summary,
        "precursor_rows": precursor_rows,
    }
    args.output_dir.mkdir(parents=True)
    _write_csv(args.output_dir / "same_state_selected.csv", selected_same_state)
    _write_csv(args.output_dir / "same_state_summary.csv", same_state_summary)
    _write_csv(args.output_dir / "precursor_events.csv", precursor_rows)
    if batch_audit["failure_frame_mismatches"]:
        _write_csv(
            args.output_dir / "batch_failure_frame_mismatches.csv",
            batch_audit["failure_frame_mismatches"],
        )
    (args.output_dir / "report.json").write_text(
        json.dumps(json_ready(report), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        json.dumps(
            json_ready(
                {
                    "cohort": cohort,
                    "decision": decision,
                    "batch_presentation_audit": batch_audit,
                }
            ),
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return report


def main(argv: Sequence[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
