#!/usr/bin/env python3
"""Run the frozen constrained face-correction oracle on the FV PCNO baseline.

This is a truth-informed upper-bound diagnostic.  It does not define an
inference algorithm, train a corrector, or turn the PCNO state prediction into
a flux model.  The fixed contract is top-20% interior-face support, a 10%
global-update norm budget, raw Euler admissibility, exact interior
cancellation, and no more than 5% worsening of any anti-smearing metric.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.time_dependent_no.evaluate_pcno_shock_vortex_baseline import (  # noqa: E402
    SCHEMA as BASELINE_SCHEMA,
    atomic_write_json,
    endpoint_metrics,
    sha256_file,
    write_csv,
)
from utility.time_dependent_no.conservative_correction_oracle import (  # noqa: E402
    conservative_face_correction_oracle,
    euler2d_state_is_admissible,
    select_endpoint_error_faces,
)

SCHEMA = "pcno_shock_vortex_constrained_correction_oracle_v1"
DEFAULT_TRAJECTORY_KEYS = (
    "sv_e06_y00",
    "sv_e11_y00",
    "sv_e00_y00",
    "sv_e00_y08",
    "sv_e06_y08",
    "sv_e11_y08",
)
ENDPOINT_CALLS = (10, 30, 60)
MAX_FACE_FRACTION = 0.20
MAX_UPDATE_RATIO = 0.10
MAX_METRIC_WORSENING = 0.05
CONSERVATION_TOLERANCE = 1.0e-10
MIN_MEDIAN_STATE_REDUCTION = 0.15
MIN_MEDIAN_SMOOTH_HIGHPASS_REDUCTION = 0.20
MIN_JOINT_NONWORSE_FRACTION = 0.75
ANTI_SMEARING_METRICS = (
    "front_centroid_distance",
    "shock_strength_log_error",
    "shock_thickness_log_error",
    "vortex_core_density_relative_error",
    "smooth_region_graph_highpass_energy",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--d013-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--trajectory-keys",
        nargs="+",
        default=list(DEFAULT_TRAJECTORY_KEYS),
        help="Defaults to the predeclared validation diagnostic cohort.",
    )
    return parser.parse_args(argv)


def metric_acceptance(
    baseline_metrics: Mapping[str, Any],
    candidate_metrics: Mapping[str, Any],
    *,
    max_relative_worsening: float = MAX_METRIC_WORSENING,
) -> dict[str, dict[str, float | bool | None]]:
    """Evaluate the fixed lower-is-better anti-smearing contract."""

    if max_relative_worsening < 0.0 or not math.isfinite(max_relative_worsening):
        raise ValueError("max_relative_worsening must be finite and nonnegative")
    report: dict[str, dict[str, float | bool | None]] = {}
    for name in ANTI_SMEARING_METRICS:
        baseline = baseline_metrics.get(name)
        candidate = candidate_metrics.get(name)
        available = (
            baseline is not None
            and candidate is not None
            and math.isfinite(float(baseline))
            and math.isfinite(float(candidate))
            and float(baseline) >= 0.0
            and float(candidate) >= 0.0
        )
        if not available:
            report[name] = {
                "baseline": None if baseline is None else float(baseline),
                "candidate": None if candidate is None else float(candidate),
                "relative_change": None,
                "accepted": False,
            }
            continue
        baseline_value = float(baseline)
        candidate_value = float(candidate)
        absolute_tolerance = 1.0e-12 * max(1.0, baseline_value)
        limit = (1.0 + max_relative_worsening) * baseline_value + absolute_tolerance
        relative_change = (
            (candidate_value - baseline_value) / baseline_value
            if baseline_value > absolute_tolerance
            else (0.0 if candidate_value <= absolute_tolerance else None)
        )
        report[name] = {
            "baseline": baseline_value,
            "candidate": candidate_value,
            "relative_change": relative_change,
            "accepted": candidate_value <= limit,
        }
    return report


def metric_report_passed(report: Mapping[str, Mapping[str, Any]]) -> bool:
    return set(report) == set(ANTI_SMEARING_METRICS) and all(
        row.get("accepted") is True for row in report.values()
    )


def promotion_decision(
    rows: Sequence[Mapping[str, Any]], *, expected_rows: int
) -> dict[str, Any]:
    """Apply the predeclared oracle promotion gates to scalar rows."""

    complete = [row for row in rows if row.get("status") == "complete"]
    state_reductions = [float(row["state_error_reduction"]) for row in complete]
    smooth_reductions = [
        float(row["smooth_highpass_error_reduction"]) for row in complete
    ]
    all_rows_available = len(rows) == expected_rows and len(complete) == expected_rows
    constraints_pass = all_rows_available and all(
        row.get("raw_admissible") is True
        and row.get("conservation_pass") is True
        and row.get("support_cap_pass") is True
        and row.get("update_cap_pass") is True
        and row.get("anti_smearing_pass") is True
        for row in complete
    )
    median_state = float(np.median(state_reductions)) if state_reductions else None
    median_smooth = float(np.median(smooth_reductions)) if smooth_reductions else None
    joint_nonworse = (
        float(
            np.mean(
                [
                    bool(row["state_error_nonworse"])
                    and bool(row["smooth_highpass_error_nonworse"])
                    for row in complete
                ]
            )
        )
        if complete
        else 0.0
    )
    gates = {
        "all_rows_available": all_rows_available,
        "all_rows_admissible_conservative_bounded_and_anti_smearing": constraints_pass,
        "median_state_error_reduction_at_least_0p15": (
            median_state is not None and median_state >= MIN_MEDIAN_STATE_REDUCTION
        ),
        "median_smooth_highpass_error_reduction_at_least_0p20": (
            median_smooth is not None
            and median_smooth >= MIN_MEDIAN_SMOOTH_HIGHPASS_REDUCTION
        ),
        "joint_state_and_smooth_nonworse_fraction_at_least_0p75": (
            joint_nonworse >= MIN_JOINT_NONWORSE_FRACTION
        ),
    }
    passed = all(gates.values())
    return {
        "passed": passed,
        "decision": (
            "authorize_one_zero_initialized_antisymmetric_learned_correction"
            if passed
            else "reject_learned_local_correction_branch"
        ),
        "expected_rows": expected_rows,
        "complete_rows": len(complete),
        "median_state_error_reduction": median_state,
        "median_smooth_highpass_error_reduction": median_smooth,
        "joint_state_and_smooth_nonworse_fraction": joint_nonworse,
        "gates": gates,
    }


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _scalar(array: np.ndarray) -> Any:
    value = np.asarray(array)
    if value.shape != ():
        raise ValueError("expected a scalar artifact field")
    return value.item()


def _validate_inputs(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any], list[str], dict[str, Mapping[str, Any]]]:
    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")
    keys = [str(key) for key in args.trajectory_keys]
    if not keys or len(keys) != len(set(keys)):
        raise ValueError("trajectory keys must be nonempty and unique")
    baseline_path = args.baseline_dir / "summary.json"
    baseline = _load_json(baseline_path)
    if (
        baseline.get("schema") != BASELINE_SCHEMA
        or baseline.get("status") != "complete"
    ):
        raise ValueError("baseline summary is not a complete physical PCNO evaluation")
    evaluation = baseline.get("evaluation", {})
    if evaluation.get("split") != "validation":
        raise ValueError("oracle method selection is restricted to validation data")
    if evaluation.get("raw_recurrence") is not True:
        raise ValueError("oracle requires a raw-recurrence baseline")
    if evaluation.get("boundary_mode") != "model_all_nodes":
        raise ValueError("oracle requires the legal model_all_nodes boundary contract")
    if any(
        bool(value) for value in evaluation.get("inference_interventions", {}).values()
    ):
        raise ValueError("oracle baseline must not contain inference interventions")
    if int(evaluation.get("num_steps", 0)) < max(ENDPOINT_CALLS):
        raise ValueError("baseline horizon does not reach every oracle endpoint")
    available = set(map(str, evaluation.get("trajectory_keys", [])))
    if not set(keys).issubset(available):
        raise ValueError("requested oracle trajectories are absent from the baseline")

    d013 = _load_json(args.d013_summary)
    if d013.get("status") != "complete":
        raise ValueError("D013 must be frozen before the correction oracle")
    if d013.get("checkpoint", {}).get("sha256") != baseline["checkpoint"]["sha256"]:
        raise ValueError("D013 and physical baseline checkpoint digests differ")
    d013_keys = set(map(str, d013.get("evaluation", {}).get("trajectory_keys", [])))
    if not set(keys).issubset(d013_keys):
        raise ValueError("D013 does not cover every requested oracle trajectory")

    contracts = {
        str(row["trajectory"]): row for row in baseline.get("trajectory_artifacts", [])
    }
    if not set(keys).issubset(contracts):
        raise ValueError("baseline summary lacks a selected trajectory artifact")
    return baseline, d013, keys, contracts


def _expected_vortex_center(
    reference_config: Mapping[str, Any],
    parameters: Mapping[str, Any],
    *,
    absolute_time: float,
    gamma: float,
) -> tuple[float, float]:
    upstream_speed = float(reference_config["shock_mach"]) * math.sqrt(gamma)
    arrival_time = (
        float(reference_config["shock_x"]) - float(reference_config["vortex_x"])
    ) / upstream_speed
    if absolute_time <= arrival_time:
        x_position = (
            float(reference_config["vortex_x"]) + upstream_speed * absolute_time
        )
    else:
        x_position = float(reference_config["shock_x"]) + float(
            reference_config["right_u"]
        ) * (absolute_time - arrival_time)
    return x_position, float(parameters["vortex_y"])


def _highpass_reduction(baseline: float, corrected: float) -> float:
    if baseline <= 0.0:
        return 0.0 if corrected <= 1.0e-12 else -1.0
    return (baseline - corrected) / baseline


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    baseline, d013, keys, contracts = _validate_inputs(args)
    args.output_dir.mkdir(parents=True)
    correction_dir = args.output_dir / "corrections"
    correction_dir.mkdir()
    rows: list[dict[str, Any]] = []
    artifact_rows: list[dict[str, Any]] = []
    evaluation = baseline["evaluation"]
    start_frame = int(evaluation["start_frame"])
    step_stride = int(evaluation["step_stride"])

    for key in keys:
        contract = contracts[key]
        artifact_path = args.baseline_dir / str(contract["artifact"])
        if sha256_file(artifact_path) != str(contract["sha256"]):
            raise ValueError(f"baseline trajectory digest mismatch: {key}")
        with np.load(artifact_path, allow_pickle=False) as data:
            required = {
                "schema",
                "trajectory",
                "initial_state",
                "targets",
                "predictions",
                "positions",
                "edges",
                "physical_cell_volumes",
                "mesh_cell_to_graph_node",
                "face_to_directed_edge",
                "face_centers",
                "face_measures",
                "face_normals",
                "face_owner",
                "face_neighbor",
                "face_axis",
                "face_boundary_tag",
                "boundary_tag_names_json",
                "coordinate_convention",
                "face_orientation_convention",
                "physical_times",
                "parameters_json",
                "reference_config_json",
                "gamma",
                "state_component_scale",
                "shock_quantile",
                "valid_length",
                "failure_cause",
                "checkpoint_sha256",
                "data_manifest_digest",
                "geometry_digest",
                "normalization_digest",
                "boundary_mode",
                "raw_recurrence",
                "inference_interventions_json",
            }
            missing = sorted(required - set(data.files))
            if missing:
                raise ValueError(f"baseline artifact {key} lacks fields: {missing}")
            if str(_scalar(data["schema"])) != BASELINE_SCHEMA:
                raise ValueError(f"unexpected baseline artifact schema for {key}")
            if str(_scalar(data["trajectory"])) != key:
                raise ValueError(f"trajectory identity mismatch for {key}")
            if (
                str(_scalar(data["checkpoint_sha256"]))
                != baseline["checkpoint"]["sha256"]
            ):
                raise ValueError(f"checkpoint digest mismatch for {key}")
            if (
                str(_scalar(data["data_manifest_digest"]))
                != baseline["data_contract"]["data_manifest_digest"]
            ):
                raise ValueError(f"data manifest digest mismatch for {key}")
            if (
                str(_scalar(data["normalization_digest"]))
                != baseline["checkpoint"]["normalization_digest"]
            ):
                raise ValueError(f"normalization digest mismatch for {key}")
            if str(_scalar(data["boundary_mode"])) != "model_all_nodes" or not bool(
                _scalar(data["raw_recurrence"])
            ):
                raise ValueError(f"illegal baseline recurrence contract for {key}")
            interventions = json.loads(
                str(_scalar(data["inference_interventions_json"]))
            )
            if any(bool(value) for value in interventions.values()):
                raise ValueError(f"baseline artifact contains an intervention: {key}")

            initial = np.asarray(data["initial_state"], dtype=np.float64)
            predictions = np.asarray(data["predictions"], dtype=np.float64)
            targets = np.asarray(data["targets"], dtype=np.float64)
            positions = np.asarray(data["positions"], dtype=np.float64)
            edges = np.asarray(data["edges"], dtype=np.int64)
            volumes = np.asarray(data["physical_cell_volumes"], dtype=np.float64)
            owner = np.asarray(data["face_owner"], dtype=np.int64)
            neighbor = np.asarray(data["face_neighbor"], dtype=np.int64)
            mesh_cell_to_graph_node = np.asarray(
                data["mesh_cell_to_graph_node"], dtype=np.int64
            )
            face_to_directed_edge = np.asarray(
                data["face_to_directed_edge"], dtype=np.int64
            )
            face_centers = np.asarray(data["face_centers"], dtype=np.float64)
            face_measures = np.asarray(data["face_measures"], dtype=np.float64)
            face_normals = np.asarray(data["face_normals"], dtype=np.float64)
            face_axis = np.asarray(data["face_axis"], dtype=np.int64)
            face_boundary_tag = np.asarray(data["face_boundary_tag"], dtype=np.int64)
            boundary_tag_names_json = str(_scalar(data["boundary_tag_names_json"]))
            coordinate_convention = str(_scalar(data["coordinate_convention"]))
            face_orientation_convention = str(
                _scalar(data["face_orientation_convention"])
            )
            physical_times = np.asarray(data["physical_times"], dtype=np.float64)
            scale = np.asarray(data["state_component_scale"], dtype=np.float64)
            gamma = float(_scalar(data["gamma"]))
            shock_quantile = float(_scalar(data["shock_quantile"]))
            parameters = json.loads(str(_scalar(data["parameters_json"])))
            reference_config = json.loads(str(_scalar(data["reference_config_json"])))
            valid_length = int(_scalar(data["valid_length"]))
            failure_cause = str(_scalar(data["failure_cause"]))
            geometry_digest = str(_scalar(data["geometry_digest"]))

        for call in ENDPOINT_CALLS:
            row_identity = {"trajectory": key, "call": call}
            if valid_length < call:
                rows.append(
                    {
                        **row_identity,
                        "status": "baseline_incomplete_before_endpoint",
                        "baseline_valid_length": valid_length,
                        "baseline_failure_cause": failure_cause,
                    }
                )
                continue
            current = initial if call == 1 else predictions[call - 2]
            prediction = predictions[call - 1]
            target = targets[call - 1]
            saved_index = start_frame + call * step_stride
            absolute_time = float(physical_times[saved_index])
            vortex_center = _expected_vortex_center(
                reference_config,
                parameters,
                absolute_time=absolute_time,
                gamma=gamma,
            )
            metric_kwargs = {
                "positions": positions,
                "edges": edges,
                "volumes": volumes,
                "component_scale": scale,
                "gamma": gamma,
                "shock_quantile": shock_quantile,
                "vortex_center": vortex_center,
            }
            baseline_metrics = endpoint_metrics(prediction, target, **metric_kwargs)
            baseline_gate = metric_acceptance(baseline_metrics, baseline_metrics)
            if not metric_report_passed(baseline_gate):
                raise ValueError(
                    f"anti-smearing metrics unavailable for {key} call {call}"
                )

            support = select_endpoint_error_faces(
                prediction,
                target,
                face_owner=owner,
                face_neighbor=neighbor,
                state_scale=scale,
                max_face_fraction=MAX_FACE_FRACTION,
            )

            def anti_smearing_check(candidate: np.ndarray) -> bool:
                candidate_metrics = endpoint_metrics(candidate, target, **metric_kwargs)
                return metric_report_passed(
                    metric_acceptance(baseline_metrics, candidate_metrics)
                )

            result = conservative_face_correction_oracle(
                prediction,
                target,
                current,
                cell_volume=volumes,
                face_owner=owner,
                face_neighbor=neighbor,
                allowed_face_mask=support,
                state_scale=scale,
                max_face_fraction=MAX_FACE_FRACTION,
                max_update_ratio=MAX_UPDATE_RATIO,
                gamma=gamma,
                anti_smearing_check=anti_smearing_check,
            )
            corrected_metrics = endpoint_metrics(
                result.corrected_state, target, **metric_kwargs
            )
            gate_report = metric_acceptance(baseline_metrics, corrected_metrics)
            anti_smearing_pass = metric_report_passed(gate_report)
            baseline_highpass = float(
                baseline_metrics["smooth_region_graph_highpass_energy"]
            )
            corrected_highpass = float(
                corrected_metrics["smooth_region_graph_highpass_energy"]
            )
            smooth_reduction = _highpass_reduction(
                baseline_highpass, corrected_highpass
            )
            raw_admissible = euler2d_state_is_admissible(
                result.corrected_state, gamma=gamma
            )
            conservation_pass = (
                result.conservation_balance_max_abs <= CONSERVATION_TOLERANCE
            )
            support_cap_pass = (
                result.allowed_face_fraction <= MAX_FACE_FRACTION + 1.0e-12
                and result.active_face_fraction <= MAX_FACE_FRACTION + 1.0e-12
            )
            update_cap_pass = result.applied_update_ratio <= MAX_UPDATE_RATIO + 1.0e-12
            state_nonworse = result.corrected_error <= result.baseline_error + 1.0e-12
            smooth_nonworse = corrected_highpass <= baseline_highpass + 1.0e-12
            artifact_name = f"correction_{key}_call{call:03d}.npz"
            artifact_path = correction_dir / artifact_name
            np.savez_compressed(
                artifact_path,
                schema=np.asarray(SCHEMA),
                trajectory=np.asarray(key),
                call=np.asarray(call, dtype=np.int64),
                physical_time=np.asarray(absolute_time, dtype=np.float64),
                current_state=current.astype(np.float32),
                global_prediction=prediction.astype(np.float32),
                target=target.astype(np.float32),
                corrected_state=result.corrected_state.astype(np.float32),
                allowed_face_mask=support,
                face_impulse=result.face_impulse,
                cell_correction=result.cell_correction,
                physical_cell_volumes=volumes,
                mesh_cell_to_graph_node=mesh_cell_to_graph_node,
                face_to_directed_edge=face_to_directed_edge,
                face_centers=face_centers,
                face_measures=face_measures,
                face_normals=face_normals,
                face_owner=owner,
                face_neighbor=neighbor,
                face_axis=face_axis,
                face_boundary_tag=face_boundary_tag,
                boundary_tag_names_json=np.asarray(boundary_tag_names_json),
                coordinate_convention=np.asarray(coordinate_convention),
                face_orientation_convention=np.asarray(face_orientation_convention),
                state_component_scale=scale,
                baseline_metrics_json=np.asarray(
                    json.dumps(baseline_metrics, sort_keys=True)
                ),
                corrected_metrics_json=np.asarray(
                    json.dumps(corrected_metrics, sort_keys=True)
                ),
                anti_smearing_report_json=np.asarray(
                    json.dumps(gate_report, sort_keys=True)
                ),
                baseline_artifact_sha256=np.asarray(str(contract["sha256"])),
                checkpoint_sha256=np.asarray(baseline["checkpoint"]["sha256"]),
                data_manifest_digest=np.asarray(
                    baseline["data_contract"]["data_manifest_digest"]
                ),
                geometry_digest=np.asarray(geometry_digest),
                normalization_digest=np.asarray(
                    baseline["checkpoint"]["normalization_digest"]
                ),
                truth_informed_support=np.asarray(True),
                inference_method=np.asarray(False),
            )
            artifact_digest = sha256_file(artifact_path)
            artifact_rows.append(
                {
                    **row_identity,
                    "artifact": f"corrections/{artifact_name}",
                    "sha256": artifact_digest,
                    "baseline_artifact_sha256": contract["sha256"],
                }
            )
            row = {
                **row_identity,
                "status": "complete",
                "physical_time": absolute_time,
                "selected_alpha": result.selected_alpha,
                "state_error_baseline": result.baseline_error,
                "state_error_corrected": result.corrected_error,
                "state_error_reduction": result.relative_error_reduction,
                "smooth_highpass_error_baseline": baseline_highpass,
                "smooth_highpass_error_corrected": corrected_highpass,
                "smooth_highpass_error_reduction": smooth_reduction,
                "state_error_nonworse": state_nonworse,
                "smooth_highpass_error_nonworse": smooth_nonworse,
                "allowed_face_fraction": result.allowed_face_fraction,
                "active_face_fraction": result.active_face_fraction,
                "applied_update_ratio": result.applied_update_ratio,
                "conservation_balance_max_abs": result.conservation_balance_max_abs,
                "raw_admissible": raw_admissible,
                "conservation_pass": conservation_pass,
                "support_cap_pass": support_cap_pass,
                "update_cap_pass": update_cap_pass,
                "anti_smearing_pass": anti_smearing_pass,
                "anti_smearing_report": gate_report,
                "feasible_line_search_points": result.feasible_line_search_points,
                "lsqr_stop_codes": list(result.lsqr_stop_codes),
                "correction_artifact": f"corrections/{artifact_name}",
                "correction_artifact_sha256": artifact_digest,
            }
            for name in ANTI_SMEARING_METRICS:
                row[f"baseline_{name}"] = baseline_metrics[name]
                row[f"corrected_{name}"] = corrected_metrics[name]
            rows.append(row)
            print(
                json.dumps(
                    {
                        "trajectory": key,
                        "call": call,
                        "state_error_reduction": result.relative_error_reduction,
                        "smooth_highpass_error_reduction": smooth_reduction,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    promotion = promotion_decision(rows, expected_rows=len(keys) * len(ENDPOINT_CALLS))
    write_csv(args.output_dir / "oracle_rows.csv", rows)
    summary = {
        "schema": SCHEMA,
        "status": "complete",
        "baseline": {
            "summary_sha256": sha256_file(args.baseline_dir / "summary.json"),
            "checkpoint_sha256": baseline["checkpoint"]["sha256"],
            "checkpoint_config_digest": baseline["checkpoint"]["config_digest"],
            "data_contract": baseline["data_contract"],
            "evaluation": baseline["evaluation"],
        },
        "d013": {
            "summary_sha256": sha256_file(args.d013_summary),
            "checkpoint_sha256": d013["checkpoint"]["sha256"],
            "mechanism_screen": d013.get("mechanism_screen"),
        },
        "evaluation": {
            "split": "validation",
            "trajectory_keys": keys,
            "predeclared_default_trajectory_cohort": tuple(keys)
            == DEFAULT_TRAJECTORY_KEYS,
            "endpoint_calls": list(ENDPOINT_CALLS),
            "endpoint_roles": {
                "10": "early_interaction",
                "30": "peak_interaction",
                "60": "final_t_0p6",
            },
            "truth_informed_support": True,
            "inference_method": False,
            "support_ranking": (
                "top interior faces by maximum owner-neighbor Euclidean "
                "component-normalized endpoint truth error; stable face-index ties"
            ),
            "max_face_fraction": MAX_FACE_FRACTION,
            "max_update_ratio": MAX_UPDATE_RATIO,
            "max_anti_smearing_metric_worsening": MAX_METRIC_WORSENING,
            "conservation_tolerance": CONSERVATION_TOLERANCE,
            "raw_recurrence_source": True,
            "inference_interventions": baseline["evaluation"][
                "inference_interventions"
            ],
        },
        "promotion": promotion,
        "rows": rows,
        "correction_artifacts": artifact_rows,
        "artifact_schema": {
            "oracle_rows_csv": (
                "all requested rows, including zero-correction optima and unavailable endpoints"
            ),
            "correction_npz": (
                "frozen current/global/target/corrected states, selected physical-face support, "
                "antisymmetric owner-oriented impulses, decoded cell correction, geometry, "
                "mesh-to-graph and face mappings, face measures/normals/orientation, "
                "normalization, anti-smearing metrics, and provenance digests"
            ),
        },
        "claim_boundary": {
            "verified": (
                "truth-informed constrained-decomposition headroom for the frozen baseline "
                "on the declared validation rows"
            ),
            "unsupported": [
                "learnability",
                "inference-time face selection",
                "model-predicted flux",
                "test-set improvement",
                "a learned local correction unless every promotion gate passes",
            ],
        },
    }
    atomic_write_json(args.output_dir / "summary.json", summary)
    print(json.dumps({"promotion": promotion}, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
