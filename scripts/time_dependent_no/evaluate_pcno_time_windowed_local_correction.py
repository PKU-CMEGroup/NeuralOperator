#!/usr/bin/env python3
"""D081: deterministic timing-versus-dose test for shock-normal correction."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.time_dependent_no.evaluate_pcno_local_correction_pilot as d080
import scripts.time_dependent_no.evaluate_pcno_native_residual_correction as parent
import scripts.time_dependent_no.evaluate_pcno_response_gain_controller as d076

SCHEMA = "pcno_time_windowed_local_correction_d081_v1"
EXPERIMENT_CONTRACT = "d081_shock_normal_timing_vs_nominal_dose"
EARLY_STOP_CALL = 20
LATE_START_CALL = 11
DOSE_MATCHED_CAP = 1.0 / 150.0
PRIMARY_ARM = "combined_shock_normal_early20"
REPLAY_ARMS = frozenset(("zero", "persistent", "combined_shock_normal"))
REPLAY_COMPARISONS = frozenset(
    (
        "persistent_vs_zero",
        "combined_shock_normal_vs_zero",
        "combined_shock_normal_vs_persistent",
    )
)

ARM_SPECS = (
    d080.LocalArmSpec("zero", None, 0.0, False),
    d080.LocalArmSpec("persistent", None, 0.0, True),
    d080.LocalArmSpec("combined_shock_normal", "shock_normal", 0.01, True),
    d080.LocalArmSpec(
        PRIMARY_ARM,
        "shock_normal",
        0.01,
        True,
        local_start_call=1,
        local_stop_call=EARLY_STOP_CALL,
    ),
    d080.LocalArmSpec(
        "combined_shock_normal_late20",
        "shock_normal",
        0.01,
        True,
        local_start_call=LATE_START_CALL,
        local_stop_call=30,
    ),
    d080.LocalArmSpec(
        "combined_shock_normal_dose_matched",
        "shock_normal",
        DOSE_MATCHED_CAP,
        True,
    ),
)
VISUAL_ARMS = tuple(arm.name for arm in ARM_SPECS)

D080_BINDINGS = {
    True: {
        "summary_sha256": (
            "def2edcc6bad175c5d63a3715879477560eaba732f372b88365a3627191163e9"
        ),
        "status": "smoke_complete",
        "selector_semantic_sha256": (
            "641d63e45e2cdbcbee46b29313b1ff4eb5fcae101f3afa63a7823f0a57967df6"
        ),
        "coefficient_semantic_sha256": (
            "b0a9e1ee116e39a9008981159f010fcf8acacb0a77497273958be9ec73b594af"
        ),
        "policy_semantic_sha256": (
            "5cdaaba34ccacea0b4bb3faa174c76bb5c0412368ff0991db3921a885d198f61"
        ),
        "gain_selection_sha256": (
            "490bb390ea67691bea05701a68a822c9d96db9b83614042ebb71ab5b97b0b4a9"
        ),
        "replay_digests": {
            "evaluation_case_summary.csv": (
                "9d6c9eabb38585ed67dea2bbdce998a34052cad3b0ae45f587b064529a22288f"
            ),
            "evaluation_call_metrics.csv": (
                "63e27bb28ea1d043a44e1dd9a6221ebb9fa74d86d63aa5abc1967d2826ba5c16"
            ),
            "evaluation_comparisons.csv": (
                "0e698e69d2572d69a2f78d7ede467f47aee60c33e4dc63995739c5672dc5e262"
            ),
        },
    },
    False: {
        "summary_sha256": (
            "c0f0d2420b046590a724842b3856ed3ee36b94e396e03be337d59d9aa3d0c18c"
        ),
        "status": "complete",
        "selector_semantic_sha256": (
            "874f649b4ca150627cfd8d5dde69f6531db36badbf418e27c87f44537c8f9813"
        ),
        "coefficient_semantic_sha256": (
            "f53e50514e4550fece263533169f4360423b0df14c6d62a79730187019dbacbc"
        ),
        "policy_semantic_sha256": (
            "51661825d019f04704ce438313a0460eb302ca2ed402ed8e4fd14e32a26e0ed4"
        ),
        "gain_selection_sha256": (
            "b2ca12b79eeea7dfbac45026bc9f744eab3fa9e4a396a2076c6423b86d693dd4"
        ),
        "replay_digests": {
            "evaluation_case_summary.csv": (
                "ca1eaf61685c51044edf4de5d1c61c72c835f771f879ce9f6694564917be116d"
            ),
            "evaluation_call_metrics.csv": (
                "601b00ef208b84bad509563e52dc431873f99fb1cc8d7c8c9770ab03a6f677ff"
            ),
            "evaluation_comparisons.csv": (
                "850b5b8650101570fa52c442601cce6fda184bd8982dffb7b31c12623131fd2a"
            ),
        },
    },
}
D080_SOURCE_MANIFEST_SHA256 = (
    "b70010320a0e6bc3bfbc605d8bdda0dfd7cc4ff4ebf1901fb3d5a06c67b5b997"
)


def _semantic_coefficient_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with np.load(path, allow_pickle=False) as payload:
        for key in sorted(set(payload.files) - {"schema"}):
            array = np.ascontiguousarray(payload[key])
            digest.update(key.encode("utf-8"))
            digest.update(str(array.dtype).encode("utf-8"))
            digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
            digest.update(array.tobytes())
    return digest.hexdigest()


def _semantic_json_sha256(path: Path, *, selector: bool) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.pop("schema", None)
    if selector:
        payload.pop("experiment_contract", None)
        payload.pop("artifact_sha256", None)
        payload["response_policy"].pop("schema", None)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalized_replay_rows(
    name: str, rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, str]]:
    if name == "evaluation_comparisons.csv":
        selected = [
            row for row in rows if str(row.get("comparison")) in REPLAY_COMPARISONS
        ]
    else:
        selected = [row for row in rows if str(row.get("arm")) in REPLAY_ARMS]
    return [
        {str(key): "" if value is None else str(value) for key, value in row.items()}
        for row in selected
    ]


def _replay_rows_sha256(name: str, rows: Sequence[Mapping[str, Any]]) -> str:
    normalized = _normalized_replay_rows(name, rows)
    encoded = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _parent_csv_digest(path: Path) -> str:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return _replay_rows_sha256(path.name, rows)


def _audit_row(check: str, expected: Any, observed: Any) -> dict[str, Any]:
    return {
        "check": check,
        "expected": expected,
        "observed": observed,
        "passed": bool(observed == expected),
    }


def _require_checks(rows: Sequence[Mapping[str, Any]], stage: str) -> None:
    failed = [str(row["check"]) for row in rows if not bool(row["passed"])]
    if failed:
        raise ValueError(f"D081 {stage} binding failed: {', '.join(failed)}")


def _validate_d080_parent(args: Any) -> dict[str, Any]:
    binding = D080_BINDINGS[bool(args.smoke)]
    root = Path(args.d080_parent_result_dir)
    summary_path = root / "summary.json"
    summary_sha = d076.sha256_file(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = [
        _audit_row("parent_summary_sha256", binding["summary_sha256"], summary_sha),
        _audit_row("parent_schema", d080.SCHEMA, summary.get("schema")),
        _audit_row("parent_status", binding["status"], summary.get("status")),
        _audit_row(
            "parent_source_manifest_sha256",
            D080_SOURCE_MANIFEST_SHA256,
            summary.get("provenance", {}).get("source_manifest_sha256"),
        ),
        _audit_row(
            "parent_rollout_calls",
            int(args.rollout_calls),
            summary.get("args", {}).get("rollout_calls"),
        ),
        _audit_row(
            "parent_boundary_policy",
            "model_all_nodes raw recurrence; base policy unchanged",
            summary.get("boundary_policy"),
        ),
        _audit_row(
            "parent_sealed_population_closed",
            False,
            summary.get("population", {}).get("sealed_populations_accessed"),
        ),
    ]
    required_files = (
        "selector.json",
        "frozen_rank8_coefficients.npz",
        "frozen_response_policy.json",
        "evaluation_gain_selections.csv",
        "evaluation_case_summary.csv",
        "evaluation_call_metrics.csv",
        "evaluation_comparisons.csv",
    )
    output_hashes = summary.get("output_hashes", {})
    for name in required_files:
        observed = d076.sha256_file(root / name)
        rows.append(
            _audit_row(
                f"parent_declared_output_sha256__{name}",
                output_hashes.get(name),
                observed,
            )
        )
    coefficient_digest = _semantic_coefficient_sha256(
        root / "frozen_rank8_coefficients.npz"
    )
    rows.extend(
        (
            _audit_row(
                "parent_coefficient_semantic_sha256",
                binding["coefficient_semantic_sha256"],
                coefficient_digest,
            ),
            _audit_row(
                "parent_policy_semantic_sha256",
                binding["policy_semantic_sha256"],
                _semantic_json_sha256(
                    root / "frozen_response_policy.json", selector=False
                ),
            ),
            _audit_row(
                "parent_selector_semantic_sha256",
                binding["selector_semantic_sha256"],
                _semantic_json_sha256(root / "selector.json", selector=True),
            ),
            _audit_row(
                "parent_gain_selection_sha256",
                binding["gain_selection_sha256"],
                d076.sha256_file(root / "evaluation_gain_selections.csv"),
            ),
        )
    )
    parent_replay_digests = {}
    for name, expected in binding["replay_digests"].items():
        observed = _parent_csv_digest(root / name)
        parent_replay_digests[name] = observed
        rows.append(_audit_row(f"parent_replay_rows__{name}", expected, observed))
    _require_checks(rows, "parent")
    return {
        "summary": summary,
        "summary_sha256": summary_sha,
        "replay_digests": parent_replay_digests,
        "audit_rows": rows,
    }


def _validate_generated_controller(
    args: Any,
    *,
    selector_sha: str,
    parent_binding: Mapping[str, Any],
) -> list[dict[str, Any]]:
    binding = D080_BINDINGS[bool(args.smoke)]
    rows = [
        _audit_row(
            "generated_selector_internal_sha256",
            selector_sha,
            d076.sha256_file(args.output_dir / "selector.json"),
        ),
        _audit_row(
            "generated_selector_semantic_sha256",
            binding["selector_semantic_sha256"],
            _semantic_json_sha256(args.output_dir / "selector.json", selector=True),
        ),
        _audit_row(
            "generated_coefficient_semantic_sha256",
            binding["coefficient_semantic_sha256"],
            _semantic_coefficient_sha256(
                args.output_dir / "frozen_rank8_coefficients.npz"
            ),
        ),
        _audit_row(
            "generated_policy_semantic_sha256",
            binding["policy_semantic_sha256"],
            _semantic_json_sha256(
                args.output_dir / "frozen_response_policy.json", selector=False
            ),
        ),
        _audit_row(
            "bound_parent_summary_sha256",
            binding["summary_sha256"],
            parent_binding["summary_sha256"],
        ),
    ]
    _require_checks(rows, "generated-controller")
    return rows


def _candidate_rows(
    comparisons: Sequence[Mapping[str, Any]],
    case_summaries: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    summaries = {(str(row["case_id"]), str(row["arm"])): row for row in case_summaries}
    case_ids = sorted(
        {
            str(row["case_id"])
            for row in case_summaries
            if str(row["arm"]) == "persistent"
        }
    )
    result = []
    for arm in ARM_SPECS[2:]:
        rows = [
            row
            for row in comparisons
            if row["comparison"] == f"{arm.name}_vs_persistent"
        ]
        if {str(row["case_id"]) for row in rows} != set(case_ids):
            raise ValueError(f"D081 comparison inventory is incomplete: {arm.name}")
        endpoint = np.asarray(
            [float(row["endpoint_state_ratio"]) for row in rows], dtype=np.float64
        )
        residual = np.asarray(
            [float(row["residual_rms_ratio"]) for row in rows], dtype=np.float64
        )
        cumulative = np.asarray(
            [
                parent._required_ratio(
                    float(summaries[(case_id, arm.name)]["net_defect_rms"]),
                    float(summaries[(case_id, "persistent")]["net_defect_rms"]),
                )
                for case_id in case_ids
            ],
            dtype=np.float64,
        )
        shock = np.asarray(
            [
                float(
                    json.loads(str(row["control_ratios_json"]))["endpoint_state__shock"]
                )
                for row in rows
            ],
            dtype=np.float64,
        )
        maximum_control = max(float(row["maximum_control_ratio"]) for row in rows)
        gates = {
            "median_endpoint_improves": float(np.median(endpoint)) < 1.0,
            "median_residual_improves": float(np.median(residual)) < 1.0,
            "median_cumulative_defect_improves": float(np.median(cumulative)) < 1.0,
            "endpoint_nonworse_four_of_six": int(np.count_nonzero(endpoint <= 1.0))
            >= 4,
            "maximum_endpoint_within_limit": float(endpoint.max())
            <= d080.ENDPOINT_WORST_LIMIT,
            "maximum_control_within_limit": maximum_control <= d080.CONTROL_WORST_LIMIT,
            "shock_endpoint_median_nonworse": float(np.median(shock)) <= 1.0,
        }
        result.append(
            {
                "arm": arm.name,
                "pathway": arm.pathway,
                "norm_cap": arm.cap,
                "local_start_call": arm.local_start_call,
                "local_stop_call": arm.local_stop_call,
                "median_endpoint_ratio": float(np.median(endpoint)),
                "median_residual_ratio": float(np.median(residual)),
                "median_cumulative_defect_ratio": float(np.median(cumulative)),
                "median_shock_endpoint_ratio": float(np.median(shock)),
                "maximum_endpoint_ratio": float(endpoint.max()),
                "maximum_control_ratio": maximum_control,
                "endpoint_nonworse_count": int(np.count_nonzero(endpoint <= 1.0)),
                "gates": gates,
                "efficacy_passed": bool(all(gates.values())),
            }
        )
    return result


def _timing_rows(
    comparisons: Sequence[Mapping[str, Any]],
    case_summaries: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    summaries = {(str(row["case_id"]), str(row["arm"])): row for row in case_summaries}
    comparison_rows = {
        (str(row["case_id"]), str(row["comparison"])): row for row in comparisons
    }
    case_ids = sorted(
        {
            str(row["case_id"])
            for row in case_summaries
            if str(row["arm"]) == PRIMARY_ARM
        }
    )
    result = []
    comparators = (
        "combined_shock_normal",
        "combined_shock_normal_late20",
        "combined_shock_normal_dose_matched",
    )
    for comparator in comparators:
        endpoint = []
        residual = []
        cumulative = []
        shock = []
        for case_id in case_ids:
            early_summary = summaries[(case_id, PRIMARY_ARM)]
            comparator_summary = summaries[(case_id, comparator)]
            endpoint.append(
                parent._required_ratio(
                    float(early_summary["final_state_error"]),
                    float(comparator_summary["final_state_error"]),
                )
            )
            residual.append(
                parent._required_ratio(
                    float(early_summary["residual_rms"]),
                    float(comparator_summary["residual_rms"]),
                )
            )
            cumulative.append(
                parent._required_ratio(
                    float(early_summary["net_defect_rms"]),
                    float(comparator_summary["net_defect_rms"]),
                )
            )
            early_controls = json.loads(
                str(
                    comparison_rows[(case_id, f"{PRIMARY_ARM}_vs_persistent")][
                        "control_ratios_json"
                    ]
                )
            )
            comparator_controls = json.loads(
                str(
                    comparison_rows[(case_id, f"{comparator}_vs_persistent")][
                        "control_ratios_json"
                    ]
                )
            )
            shock.append(
                parent._required_ratio(
                    float(early_controls["endpoint_state__shock"]),
                    float(comparator_controls["endpoint_state__shock"]),
                )
            )
        endpoint_array = np.asarray(endpoint, dtype=np.float64)
        cumulative_array = np.asarray(cumulative, dtype=np.float64)
        result.append(
            {
                "comparison": f"{PRIMARY_ARM}_vs_{comparator}",
                "comparator": comparator,
                "median_endpoint_ratio": float(np.median(endpoint_array)),
                "median_residual_ratio": float(np.median(residual)),
                "median_cumulative_defect_ratio": float(np.median(cumulative_array)),
                "median_shock_endpoint_ratio": float(np.median(shock)),
                "endpoint_nonworse_count": int(np.count_nonzero(endpoint_array <= 1.0)),
                "endpoint_timing_gate": float(np.median(endpoint_array)) < 1.0,
                "cumulative_timing_gate": float(np.median(cumulative_array)) < 1.0,
            }
        )
    return result


def _promotion(
    comparisons: Sequence[Mapping[str, Any]],
    case_summaries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    candidates = _candidate_rows(comparisons, case_summaries)
    timing = _timing_rows(comparisons, case_summaries)
    primary = next(row for row in candidates if row["arm"] == PRIMARY_ARM)
    timing_supported = bool(
        all(
            row["endpoint_timing_gate"] and row["cumulative_timing_gate"]
            for row in timing
        )
    )
    return {
        "adaptive_open_population": True,
        "d080_parent_summary_sha256": D080_BINDINGS[False]["summary_sha256"],
        "primary_arm": PRIMARY_ARM,
        "candidate_rows": candidates,
        "timing_rows": timing,
        "timing_mechanism_supported": timing_supported,
        "efficacy_passed": bool(primary["efficacy_passed"]),
        "selected_arm": PRIMARY_ARM if primary["efficacy_passed"] else None,
        "passed": bool(primary["efficacy_passed"]),
    }


def _schedule_rows(calls: int) -> list[dict[str, Any]]:
    rows = []
    for arm in ARM_SPECS:
        stop = calls if arm.local_stop_call is None else min(arm.local_stop_call, calls)
        active_calls = (
            0
            if arm.pathway is None or stop < arm.local_start_call
            else stop - arm.local_start_call + 1
        )
        rows.append(
            {
                "arm": arm.name,
                "pathway": arm.pathway or "none",
                "include_persistent": arm.include_persistent,
                "norm_cap": arm.cap,
                "local_start_call": arm.local_start_call,
                "local_stop_call": arm.local_stop_call,
                "rollout_calls": calls,
                "active_local_calls": active_calls,
                "nominal_cap_dose": float(active_calls * arm.cap),
            }
        )
    return rows


def _conditional_evaluation(
    args: Any,
    *,
    spec: d076.ResponseExperimentSpec,
    model: torch.nn.Module,
    checkpoint: Mapping[str, Any],
    store: Any,
    device: torch.device,
    evaluation_ids: Sequence[str],
    candidates: Sequence[parent.Candidate],
    frozen_coefficients: np.ndarray,
    frozen_policy: Mapping[str, Any],
    response_table_rows: Sequence[Mapping[str, Any]],
    selector_sha: str,
    phase_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    parent_binding = _validate_d080_parent(args)
    audit_rows = [
        *parent_binding["audit_rows"],
        *_validate_generated_controller(
            args,
            selector_sha=selector_sha,
            parent_binding=parent_binding,
        ),
    ]
    probe = d076._freeze_evaluation_probe_choices(
        args,
        spec=spec,
        model=model,
        checkpoint=checkpoint,
        store=store,
        device=device,
        evaluation_ids=evaluation_ids,
        candidates=candidates,
        frozen_coefficients=frozen_coefficients,
        frozen_policy=frozen_policy,
        response_table_rows=response_table_rows,
        selector_sha=selector_sha,
        phase_rows=phase_rows,
    )
    gain_row = _audit_row(
        "generated_gain_selection_sha256",
        D080_BINDINGS[bool(args.smoke)]["gain_selection_sha256"],
        d076.sha256_file(args.output_dir / "evaluation_gain_selections.csv"),
    )
    audit_rows.append(gain_row)
    _require_checks((gain_row,), "evaluation-choice")
    if d076.sha256_file(probe["record_path"]) != probe["record_sha"] or any(
        d076.sha256_file(args.output_dir / name) != digest
        for name, digest in probe["artifact_sha256"].items()
    ):
        raise ValueError("D081 evaluation choice bundle changed before rollout")

    evaluated = d080._evaluate_local_arms(
        args,
        model=model,
        device=device,
        probe=probe,
        arm_specs=ARM_SPECS,
        visual_arms=VISUAL_ARMS,
        schema=SCHEMA,
        promotion_runner=_promotion,
    )
    if d076.sha256_file(probe["record_path"]) != probe["record_sha"] or any(
        d076.sha256_file(args.output_dir / name) != digest
        for name, digest in probe["artifact_sha256"].items()
    ):
        raise ValueError("D081 evaluation choice bundle changed during rollout")

    replay_rows = []
    for name, expected in parent_binding["replay_digests"].items():
        observed = _replay_rows_sha256(name, evaluated["outputs"][name])
        row = _audit_row(f"generated_replay_rows__{name}", expected, observed)
        replay_rows.append(row)
        audit_rows.append(row)
    parent_replay_pass = bool(all(row["passed"] for row in replay_rows))
    evaluated["checks"]["closure_pass"] = bool(
        evaluated["checks"]["closure_pass"] and parent_replay_pass
    )
    evaluated["outputs"]["schedule_contract.csv"] = _schedule_rows(args.rollout_calls)
    evaluated["outputs"]["d080_parent_replay.csv"] = audit_rows
    evaluated["promotion"]["contract_passed"] = bool(
        evaluated["promotion"]["contract_passed"] and parent_replay_pass
    )
    evaluated["promotion"]["passed"] = bool(
        evaluated["promotion"]["efficacy_passed"]
        and evaluated["promotion"]["contract_passed"]
        and not args.smoke
    )
    if args.smoke:
        evaluated["promotion"]["status"] = "non_scientific_smoke"
        evaluated["promotion"]["selected_arm"] = None

    output_rows = {
        **probe["output_rows"],
        **d080._materialized_csv_rows(evaluated["outputs"]),
        "evaluation_reference_checks.csv": probe["reference_checks"],
    }
    checks = {
        "evaluation_choice_bundle_revalidated": True,
        "evaluation_inventory_pass": evaluated["checks"]["inventory_pass"],
        "evaluation_inventory_details": evaluated["checks"]["inventory"],
        "all_evaluation_complete": bool(
            evaluated["checks"]["all_evaluation_complete"]
            and evaluated["checks"]["completion_pass"]
        ),
        "probe_prefix_replay_pass": evaluated["checks"]["prefix_replay_pass"],
        "d080_parent_binding_pass": True,
        "d080_control_row_replay_pass": parent_replay_pass,
        "d080_parent_summary_sha256": parent_binding["summary_sha256"],
        **evaluated["checks"],
    }
    return {
        "hook": probe["hook"],
        "reference_checks": probe["reference_checks"],
        "case_contract_rows": probe["case_contract_rows"],
        "output_rows": output_rows,
        "promotion": evaluated["promotion"],
        "choice_bundle": {
            "record_path": probe["record_path"],
            "record_sha": probe["record_sha"],
            "artifact_sha256": probe["artifact_sha256"],
        },
        "checks": checks,
    }


D081_SPEC = replace(
    d080.D080_SPEC,
    experiment_id="D081",
    schema=SCHEMA,
    experiment_contract=EXPERIMENT_CONTRACT,
    description=__doc__ or "D081 shock-normal timing versus dose",
    method_claim=(
        "D078 persistent low-rank controller plus the exact D080 shock-normal "
        "correction under predeclared early, late, always-on, and nominal-dose "
        "schedules; adaptive open-validation mechanism test; not data assimilation"
    ),
    extra_source_paths=(*d080.D080_SPEC.extra_source_paths, Path(__file__)),
    evaluation_runner=_conditional_evaluation,
)


def parse_args(argv: Sequence[str] | None = None):
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--d080-parent-result-dir", type=Path, required=True)
    known, remaining = pre_parser.parse_known_args(argv)
    args = d076.parse_args_for_experiment(remaining, D081_SPEC)
    args.d080_parent_result_dir = known.d080_parent_result_dir
    return args


def run(args: Any):
    return d076.run(args, spec=D081_SPEC)


def main(argv: Sequence[str] | None = None) -> int:
    summary = run(parse_args(argv))
    print(
        f"D081 dynamic_fv status={summary['status']} "
        f"calibration_qualified={summary['calibration_qualification']['passed']} "
        f"promotion={summary['promotion'].get('passed')} "
        f"timing={summary['promotion'].get('timing_mechanism_supported')}",
        flush=True,
    )
    return 0 if summary["contract_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
