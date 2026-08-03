#!/usr/bin/env python3
"""D071: evaluate frozen persistent and local residual corrections."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.time_dependent_no.evaluate_pcno_node_type_interventions import (
    CaseData,
    _bump_replay_gate,
    _hook_equivalence,
    _load_bump_cases,
    _load_dynamic_cases,
    _loaded_project_source_hashes,
    _region_masks,
    _verify_common_provenance,
)
from utility.time_dependent_no.pcno_artifacts import (
    atomic_write_json_with_paths as write_json,
)
from utility.time_dependent_no.pcno_artifacts import (
    git_state,
    jsonable_args,
    runtime_environment,
    sha256_file,
    write_csv_with_paths,
)
from utility.time_dependent_no.pcno_defect_corrections import (
    controlled_graph_dissipation,
    fit_error_coefficient_sequence,
    mean_calibration_coefficients,
    physical_cosine_basis,
    reconstruct_coefficient_sequence,
)
from utility.time_dependent_no.pcno_euler2d import PCNOEuler2DShardStore
from utility.time_dependent_no.pcno_resolution_transfer import (
    build_resolution_checkpoint_model,
    conservative_admissibility_summary,
    load_resolution_checkpoint,
    parse_resolution,
)
from utility.time_dependent_no.pcno_ripple_diagnostics import (
    conservative_to_primitive_raw,
    node_highpass_amplitude,
    node_highpass_field,
)
from utility.time_dependent_no.pcno_runtime import (
    build_checkpoint_model,
    load_checkpoint_payload,
    select_device,
)
from utility.time_dependent_no.pcno_scale_separated_drift import project_dct_bands

SCHEMA = "pcno_defect_correction_diagnostic_v1"
DYNAMIC_EVALUATION_CASES = (
    "sv_e00_y00",
    "sv_e00_y08",
    "sv_e06_y00",
    "sv_e06_y08",
    "sv_e11_y00",
    "sv_e11_y08",
)
BUMP_EVALUATION_CASES = ("128", "172", "58", "187")
ARMS = ("baseline", "persistent_rank8", "local_dissipation", "combined")
RANK = 8
FILTER_CAPS = (0.02, 0.05)
SENSOR_QUANTILE = 0.8
COMPONENTS = ("rho", "rho_u", "rho_v", "energy")
D071_HOOK_EQUIVALENCE_LIMITS = {
    "bump": {
        "native_repeat_absolute_limit": 2.0e-3,
        "shared_relative_l2_limit": 1.0e-5,
    },
    "dynamic_fv": {
        "native_repeat_absolute_limit": 2.0e-5,
        "shared_relative_l2_limit": 1.0e-7,
    },
}
HOOK_EQUIVALENCE_TOLERANCE_SOURCE = (
    "accepted_D068_family_specific_hook_equivalence_contract"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("bump", "dynamic_fv"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--normalization-json", type=Path, required=True)
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rollout-calls", type=int, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--amp", choices=("none",), default="none")
    parser.add_argument("--shock-quantile", type=float, default=0.9)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-normalization-sha256", required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--expected-data-manifest-digest", required=True)
    parser.add_argument("--expected-source-base-git-head", required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--expected-source-manifest-sha256", required=True)
    parser.add_argument(
        "--expected-source", action="append", default=[], metavar="PATH=SHA256"
    )
    parser.add_argument("--bump-replay-root", type=Path)
    parser.add_argument("--bump-replay-cases", nargs="*", default=())
    parser.add_argument("--bump-replay-calls", type=int, default=20)
    parser.add_argument("--bump-replay-absolute-limit", type=float)
    parser.add_argument("--bump-replay-relative-limit", type=float)
    parser.add_argument("--family-root", type=Path)
    parser.add_argument("--multires-reference-root", type=Path)
    parser.add_argument("--expected-family-manifest-sha256")
    parser.add_argument(
        "--resolutions", nargs="+", default=("125x50", "250x100", "500x200")
    )
    parser.add_argument("--training-resolution", default="250x100")
    parser.add_argument("--visualization-cases", nargs="*", default=())
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    expected_horizon = 20 if args.family == "bump" else 30
    if args.rollout_calls != expected_horizon:
        raise ValueError(f"D071 freezes {args.family} H{expected_horizon}")
    if args.family == "bump":
        if args.bump_replay_root is None or not args.bump_replay_cases:
            raise ValueError("bump requires the D041 open-validation replay gate")
        if (
            args.bump_replay_absolute_limit is None
            or args.bump_replay_relative_limit is None
        ):
            raise ValueError("bump replay limits are required")
        if args.family_root is not None or args.multires_reference_root is not None:
            raise ValueError("dynamic roots are invalid for bump")
    else:
        if args.family_root is None or args.multires_reference_root is None:
            raise ValueError("dynamic FV requires both reference roots")
        if args.expected_family_manifest_sha256 is None:
            raise ValueError("dynamic FV requires the family-manifest digest")
        if tuple(parse_resolution(value) for value in args.resolutions) != (
            (125, 50),
            (250, 100),
            (500, 200),
        ):
            raise ValueError("D071 freezes dynamic grids at 125x50, 250x100, 500x200")
        if parse_resolution(args.training_resolution) != (250, 100):
            raise ValueError("D071 freezes the dynamic training resolution at 250x100")
    for name in ("expected_source_base_git_head", "expected_source_manifest_sha256"):
        digest = str(getattr(args, name)).lower()
        size = 40 if name == "expected_source_base_git_head" else 64
        if len(digest) != size or any(
            value not in "0123456789abcdef" for value in digest
        ):
            raise ValueError(f"--{name.replace('_', '-')} must be a {size}-digit SHA")
    return args


def _evaluation_ids(family: str) -> tuple[str, ...]:
    return BUMP_EVALUATION_CASES if family == "bump" else DYNAMIC_EVALUATION_CASES


def _hook_equivalence_limits(family: str) -> dict[str, float]:
    try:
        return dict(D071_HOOK_EQUIVALENCE_LIMITS[family])
    except KeyError as error:
        raise ValueError(f"unsupported D071 family: {family}") from error


def _run_hook_equivalence(
    model: torch.nn.Module,
    case: CaseData,
    *,
    device: torch.device,
) -> dict[str, Any]:
    limits = _hook_equivalence_limits(case.family)
    result = _hook_equivalence(
        model,
        case,
        device=device,
        amp="none",
        absolute_limit=limits["native_repeat_absolute_limit"],
        relative_limit=limits["shared_relative_l2_limit"],
    )
    return {
        **result,
        "tolerance_source": HOOK_EQUIVALENCE_TOLERANCE_SOURCE,
        **limits,
    }


def _domain_bounds(family: str) -> tuple[float, float, float, float]:
    return (0.0, 6.0, 0.0, 2.0) if family == "bump" else (0.0, 2.0, 0.0, 1.0)


def _case_key(case: CaseData) -> tuple[str, str]:
    return case.case_id, case.resolution_name


def _group_key(case: CaseData) -> str:
    return case.resolution_name


@torch.inference_mode()
def _predict(model: torch.nn.Module, case: CaseData, state: np.ndarray) -> np.ndarray:
    current = torch.as_tensor(
        np.asarray(state, dtype=np.float32),
        dtype=torch.float32,
        device=case.sample["nodes"].device,
    ).unsqueeze(0)
    prediction = model(
        current,
        node_mask=case.sample["node_mask"],
        nodes=case.sample["nodes"],
        node_weights=case.sample["node_weights"],
        node_rhos=case.sample["node_rhos"],
        directed_edges=case.sample["directed_edges"],
        edge_gradient_weights=case.sample["edge_gradient_weights"],
        node_type=case.sample["node_type"],
        mach=case.sample["mach"],
    )
    return prediction[0].detach().float().cpu().numpy().astype(np.float64)


def _weighted_rms(
    value: np.ndarray,
    *,
    weights: np.ndarray,
    scale: np.ndarray,
    mask: np.ndarray | None = None,
) -> float | None:
    selected = (
        np.ones(weights.shape[0], dtype=bool)
        if mask is None
        else np.asarray(mask, dtype=bool)
    )
    if not np.any(selected):
        return None
    mass = np.asarray(weights, dtype=np.float64)[selected]
    field = np.asarray(value, dtype=np.float64)[selected] / np.asarray(scale)[None, :]
    return float(np.sqrt(np.dot(mass, np.sum(np.square(field), axis=-1)) / mass.sum()))


def _relative_error(
    error: np.ndarray,
    reference: np.ndarray,
    *,
    weights: np.ndarray,
    scale: np.ndarray,
    mask: np.ndarray,
) -> float | None:
    numerator = _weighted_rms(error, weights=weights, scale=scale, mask=mask)
    denominator = _weighted_rms(reference, weights=weights, scale=scale, mask=mask)
    if numerator is None or denominator in {None, 0.0}:
        return None
    return numerator / denominator


def _safe_admissibility_summary(state: np.ndarray, *, gamma: float) -> dict[str, Any]:
    values = np.asarray(state, dtype=np.float64)
    if not np.isfinite(values).all():
        return {
            "finite": False,
            "admissible": False,
            "admissible_fraction": 0.0,
            "minimum_density": None,
            "minimum_internal_energy": None,
            "minimum_pressure": None,
        }
    return conservative_admissibility_summary(values, gamma=gamma)


def _reference_mask_shock_proxy_features(
    case: CaseData, state: np.ndarray, mask: np.ndarray
) -> dict[str, float | None]:
    pressure = conservative_to_primitive_raw(state, gamma=case.gamma)[:, 3]
    score = node_highpass_amplitude(pressure, case.edges)
    selected = np.asarray(mask, dtype=bool)
    weighted_score = case.weights[selected] * score[selected]
    total = float(weighted_score.sum())
    if total <= np.finfo(float).tiny:
        return {
            "proxy_centroid_x": None,
            "proxy_strength": None,
            "proxy_thickness_x": None,
        }
    x = case.nodes[selected, 0]
    center = float(np.dot(weighted_score, x) / total)
    thickness = float(np.sqrt(np.dot(weighted_score, np.square(x - center)) / total))
    strength = float(
        np.sqrt(
            np.dot(case.weights[selected], np.square(score[selected]))
            / case.weights[selected].sum()
        )
    )
    return {
        "proxy_centroid_x": center,
        "proxy_strength": strength,
        "proxy_thickness_x": thickness,
    }


def _scale_band_fields(
    case: CaseData, field: np.ndarray
) -> tuple[dict[str, np.ndarray], float, str]:
    values = np.asarray(field, dtype=np.float64)
    if case.family == "dynamic_fv":
        if case.resolution is None:
            raise ValueError("dynamic scale bands require a structured resolution")
        projections, closure = project_dct_bands(
            values[None],
            resolution=case.resolution,
            component_scale=case.residual_scale,
            domain_lengths=(2.0, 1.0),
        )
        fields = {"total": values}
        fields.update({name: value[0] for name, value in projections.items()})
        return (
            fields,
            max(float(value) for value in closure.values()),
            "physical_dct_wavelength",
        )

    scaled = values / case.residual_scale[None, :]
    local = node_highpass_field(scaled, case.edges)
    first_average = scaled - local
    transition = node_highpass_field(first_average, case.edges)
    large = first_average - transition
    reconstructed = large + transition + local
    return (
        {
            "total": values,
            "large": large * case.residual_scale[None, :],
            "transition": transition * case.residual_scale[None, :],
            "local": local * case.residual_scale[None, :],
        },
        float(np.max(np.abs(reconstructed - scaled))),
        "nonorthogonal_two_level_graph_proxy",
    )


def _append_band_metrics(
    rows: list[dict[str, Any]],
    case: CaseData,
    *,
    mode: str,
    call: int,
    arm: str,
    current: np.ndarray,
    prediction: np.ndarray,
    target: np.ndarray,
    correction: np.ndarray,
) -> None:
    truth_update = target - case.reference_states[call - 1]
    predicted_update = prediction - current
    fields = {
        "update_error": predicted_update - truth_update,
        "correction": correction,
    }
    for field_name, field in fields.items():
        bands, closure, contract = _scale_band_fields(case, field)
        for band, values in bands.items():
            rows.append(
                {
                    "family": case.family,
                    "case_id": case.case_id,
                    "resolution": case.resolution_name,
                    "mode": mode,
                    "call": call,
                    "physical_time": float(case.physical_times[call]),
                    "arm": arm,
                    "field": field_name,
                    "band": band,
                    "band_contract": contract,
                    "rms_residual_scale": _weighted_rms(
                        values,
                        weights=case.weights,
                        scale=case.residual_scale,
                    ),
                    "maximum_reconstruction_closure": closure,
                }
            )


def _calibrate(
    model: torch.nn.Module,
    cases: Sequence[CaseData],
    calibration_ids: set[str],
) -> tuple[dict[str, np.ndarray], float, list[dict[str, Any]]]:
    coefficients: dict[str, list[np.ndarray]] = defaultdict(list)
    cap_scores: dict[float, dict[tuple[str, str], list[float]]] = {
        cap: defaultdict(list) for cap in FILTER_CAPS
    }
    rows: list[dict[str, Any]] = []
    bounds = _domain_bounds(cases[0].family)
    for case in cases:
        if case.case_id not in calibration_ids:
            continue
        basis, modes = physical_cosine_basis(
            case.nodes, rank=RANK, domain_bounds=bounds
        )
        errors = []
        for call in range(1, case.reference_states.shape[0]):
            current = case.reference_states[call - 1]
            target = case.reference_states[call]
            prediction = _predict(model, case, current)
            update = prediction - current
            truth = target - current
            error = update - truth
            errors.append(error)
            for cap in FILTER_CAPS:
                result = controlled_graph_dissipation(
                    update,
                    case.edges,
                    case.weights,
                    case.physical_node_type == 0,
                    component_scale=case.residual_scale,
                    sensor_quantile=SENSOR_QUANTILE,
                    norm_cap=cap,
                )
                score = _weighted_rms(
                    error + result.correction,
                    weights=case.weights,
                    scale=case.residual_scale,
                )
                if score is None:
                    raise AssertionError("calibration graph has no weighted nodes")
                cap_scores[cap][_case_key(case)].append(score)
        sequence = fit_error_coefficient_sequence(
            np.asarray(errors)[:, case.physical_node_type == 0],
            basis[case.physical_node_type == 0],
            case.weights[case.physical_node_type == 0],
            component_scale=case.residual_scale,
        )
        coefficients[_group_key(case)].append(sequence)
        rows.append(
            {
                "family": case.family,
                "case_id": case.case_id,
                "resolution": case.resolution_name,
                "role": "calibration",
                "calls": int(sequence.shape[0]),
                "rank": RANK,
                "modes": json.dumps(modes),
            }
        )
    frozen = {
        group: mean_calibration_coefficients(np.asarray(values))
        for group, values in coefficients.items()
    }
    cap_summary = {}
    for cap, by_case in cap_scores.items():
        per_case = [float(np.mean(values)) for values in by_case.values()]
        cap_summary[cap] = float(np.median(per_case))
        rows.append(
            {
                "family": cases[0].family,
                "case_id": "aggregate",
                "resolution": "all",
                "role": "filter_cap_selection",
                "filter_cap": cap,
                "median_per_case_residual_error_rms": cap_summary[cap],
                "case_resolution_count": len(per_case),
            }
        )
    selected_cap = min(FILTER_CAPS, key=lambda cap: (cap_summary[cap], cap))
    return frozen, float(selected_cap), rows


def _bias_sequence(
    case: CaseData, frozen_coefficients: Mapping[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    basis, _ = physical_cosine_basis(
        case.nodes, rank=RANK, domain_bounds=_domain_bounds(case.family)
    )
    coefficients = frozen_coefficients[_group_key(case)]
    reconstructed = reconstruct_coefficient_sequence(
        coefficients, basis, component_scale=case.residual_scale
    )
    reconstructed[:, case.physical_node_type != 0] = 0.0
    return basis, reconstructed


def _append_metrics(
    metric_rows: list[dict[str, Any]],
    component_rows: list[dict[str, Any]],
    front_rows: list[dict[str, Any]],
    case: CaseData,
    *,
    mode: str,
    call: int,
    arm: str,
    current: np.ndarray,
    prediction: np.ndarray,
    target: np.ndarray,
    correction: np.ndarray,
    shock_quantile: float,
) -> None:
    truth_update = target - case.reference_states[call - 1]
    predicted_update = prediction - current
    update_error = predicted_update - truth_update
    state_error = prediction - target
    regions = _region_masks(case, target, shock_quantile=shock_quantile)
    scaled_highpass = node_highpass_field(
        state_error / case.state_scale[None, :], case.edges
    )
    for region, mask in regions.items():
        metric_rows.append(
            {
                "family": case.family,
                "case_id": case.case_id,
                "resolution": case.resolution_name,
                "mode": mode,
                "call": call,
                "physical_time": float(case.physical_times[call]),
                "arm": arm,
                "region": region,
                "node_count": int(np.asarray(mask).sum()),
                "weight_mass": float(case.weights[np.asarray(mask)].sum()),
                "state_error_rms": _weighted_rms(
                    state_error,
                    weights=case.weights,
                    scale=case.state_scale,
                    mask=mask,
                ),
                "state_relative_l2": _relative_error(
                    state_error,
                    target,
                    weights=case.weights,
                    scale=case.state_scale,
                    mask=mask,
                ),
                "update_error_rms": _weighted_rms(
                    update_error,
                    weights=case.weights,
                    scale=case.residual_scale,
                    mask=mask,
                ),
                "update_relative_l2": _relative_error(
                    update_error,
                    truth_update,
                    weights=case.weights,
                    scale=case.residual_scale,
                    mask=mask,
                ),
                "truth_update_rms": _weighted_rms(
                    truth_update,
                    weights=case.weights,
                    scale=case.residual_scale,
                    mask=mask,
                ),
                "correction_rms": _weighted_rms(
                    correction,
                    weights=case.weights,
                    scale=case.residual_scale,
                    mask=mask,
                ),
                "state_error_graph_highpass_rms": _weighted_rms(
                    scaled_highpass,
                    weights=case.weights,
                    scale=np.ones(4),
                    mask=mask,
                ),
            }
        )
    all_nodes = np.ones(case.weights.shape[0], dtype=bool)
    for component, name in enumerate(COMPONENTS):
        component_rows.append(
            {
                "family": case.family,
                "case_id": case.case_id,
                "resolution": case.resolution_name,
                "mode": mode,
                "call": call,
                "physical_time": float(case.physical_times[call]),
                "arm": arm,
                "component": name,
                "state_error_rms": _weighted_rms(
                    state_error[:, component : component + 1],
                    weights=case.weights,
                    scale=case.state_scale[component : component + 1],
                    mask=all_nodes,
                ),
                "state_relative_l2": _relative_error(
                    state_error[:, component : component + 1],
                    target[:, component : component + 1],
                    weights=case.weights,
                    scale=case.state_scale[component : component + 1],
                    mask=all_nodes,
                ),
                "update_error_rms": _weighted_rms(
                    update_error[:, component : component + 1],
                    weights=case.weights,
                    scale=case.residual_scale[component : component + 1],
                    mask=all_nodes,
                ),
                "update_relative_l2": _relative_error(
                    update_error[:, component : component + 1],
                    truth_update[:, component : component + 1],
                    weights=case.weights,
                    scale=case.residual_scale[component : component + 1],
                    mask=all_nodes,
                ),
                "truth_update_rms": _weighted_rms(
                    truth_update[:, component : component + 1],
                    weights=case.weights,
                    scale=case.residual_scale[component : component + 1],
                    mask=all_nodes,
                ),
                "correction_rms": _weighted_rms(
                    correction[:, component : component + 1],
                    weights=case.weights,
                    scale=case.residual_scale[component : component + 1],
                    mask=all_nodes,
                ),
                "update_error_absolute_rms": _weighted_rms(
                    update_error[:, component : component + 1],
                    weights=case.weights,
                    scale=np.ones(1),
                    mask=all_nodes,
                ),
                "truth_update_absolute_rms": _weighted_rms(
                    truth_update[:, component : component + 1],
                    weights=case.weights,
                    scale=np.ones(1),
                    mask=all_nodes,
                ),
            }
        )
    reference_front = _reference_mask_shock_proxy_features(
        case, target, regions["shock"]
    )
    prediction_front = _reference_mask_shock_proxy_features(
        case, prediction, regions["shock"]
    )
    front_rows.append(
        {
            "family": case.family,
            "case_id": case.case_id,
            "resolution": case.resolution_name,
            "mode": mode,
            "call": call,
            "arm": arm,
            "feature_contract": "reference_shock_mask_graph_highpass_proxy_descriptive_only",
            **{f"reference_{name}": value for name, value in reference_front.items()},
            **{f"prediction_{name}": value for name, value in prediction_front.items()},
            "proxy_centroid_absolute_error": (
                None
                if reference_front["proxy_centroid_x"] is None
                or prediction_front["proxy_centroid_x"] is None
                else abs(
                    prediction_front["proxy_centroid_x"]
                    - reference_front["proxy_centroid_x"]
                )
            ),
            "proxy_strength_absolute_error": (
                None
                if reference_front["proxy_strength"] is None
                or prediction_front["proxy_strength"] is None
                else abs(
                    prediction_front["proxy_strength"]
                    - reference_front["proxy_strength"]
                )
            ),
            "proxy_thickness_absolute_error": (
                None
                if reference_front["proxy_thickness_x"] is None
                or prediction_front["proxy_thickness_x"] is None
                else abs(
                    prediction_front["proxy_thickness_x"]
                    - reference_front["proxy_thickness_x"]
                )
            ),
        }
    )


def _proposal_arms(
    case: CaseData,
    *,
    current: np.ndarray,
    base_prediction: np.ndarray,
    bias: np.ndarray,
    filter_cap: float,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    update = base_prediction - current
    local = controlled_graph_dissipation(
        update,
        case.edges,
        case.weights,
        case.physical_node_type == 0,
        component_scale=case.residual_scale,
        sensor_quantile=SENSOR_QUANTILE,
        norm_cap=filter_cap,
    )
    corrections = {
        "baseline": np.zeros_like(update),
        "persistent_rank8": -bias,
        "local_dissipation": local.correction,
        "combined": -bias + local.correction,
    }
    predictions = {
        arm: current + update + correction for arm, correction in corrections.items()
    }
    return (
        predictions,
        corrections,
        {
            "sensor": local.sensor,
            "local_correction": local.correction,
            "local_summary": local,
        },
    )


def _evaluate_case(
    model: torch.nn.Module,
    case: CaseData,
    *,
    frozen_coefficients: Mapping[str, np.ndarray],
    filter_cap: float,
    shock_quantile: float,
    metric_rows: list[dict[str, Any]],
    component_rows: list[dict[str, Any]],
    band_rows: list[dict[str, Any]],
    front_rows: list[dict[str, Any]],
    correction_rows: list[dict[str, Any]],
    completion_rows: list[dict[str, Any]],
    teacher_completion_rows: list[dict[str, Any]],
    save_visual: bool,
    visual_dir: Path,
) -> None:
    basis, bias_sequence = _bias_sequence(case, frozen_coefficients)
    calls = case.reference_states.shape[0] - 1
    band_calls = {1, 5, 15, calls} if case.family == "dynamic_fv" else {1, 5, 10, calls}
    for call in range(1, calls + 1):
        current = case.reference_states[call - 1]
        target = case.reference_states[call]
        base_prediction = _predict(model, case, current)
        if not np.isfinite(base_prediction).all():
            teacher_completion_rows.append(
                {
                    "family": case.family,
                    "case_id": case.case_id,
                    "resolution": case.resolution_name,
                    "call": call,
                    "finite": False,
                    "failure_stage": "base_prediction_nonfinite",
                }
            )
            correction_rows.append(
                {
                    "family": case.family,
                    "case_id": case.case_id,
                    "resolution": case.resolution_name,
                    "mode": "teacher_forced",
                    "call": call,
                    "arm": "proposal_diagnostic",
                    "filter_cap": filter_cap,
                    "failure_stage": "base_prediction_nonfinite",
                    "eligible_edge_count": None,
                    "applied_local_filter_relative_norm": None,
                    "maximum_local_weighted_mean_closure": None,
                }
            )
            continue
        predictions, corrections, local = _proposal_arms(
            case,
            current=current,
            base_prediction=base_prediction,
            bias=bias_sequence[call - 1],
            filter_cap=filter_cap,
        )
        error = (base_prediction - current) - (target - current)
        oracle_coefficients = fit_error_coefficient_sequence(
            error[None, case.physical_node_type == 0],
            basis[case.physical_node_type == 0],
            case.weights[case.physical_node_type == 0],
            component_scale=case.residual_scale,
        )
        oracle_projection = reconstruct_coefficient_sequence(
            oracle_coefficients,
            basis,
            component_scale=case.residual_scale,
        )[0]
        oracle_projection[case.physical_node_type != 0] = 0.0
        predictions["oracle_rank8_headroom"] = base_prediction - oracle_projection
        corrections["oracle_rank8_headroom"] = -oracle_projection
        all_predictions_finite = all(
            np.isfinite(prediction).all() for prediction in predictions.values()
        )
        teacher_completion_rows.append(
            {
                "family": case.family,
                "case_id": case.case_id,
                "resolution": case.resolution_name,
                "call": call,
                "finite": all_predictions_finite,
                "failure_stage": (
                    None if all_predictions_finite else "proposal_prediction_nonfinite"
                ),
            }
        )
        if not all_predictions_finite:
            continue
        for arm, prediction in predictions.items():
            _append_metrics(
                metric_rows,
                component_rows,
                front_rows,
                case,
                mode="teacher_forced",
                call=call,
                arm=arm,
                current=current,
                prediction=prediction,
                target=target,
                correction=corrections[arm],
                shock_quantile=shock_quantile,
            )
            if call in band_calls:
                _append_band_metrics(
                    band_rows,
                    case,
                    mode="teacher_forced",
                    call=call,
                    arm=arm,
                    current=current,
                    prediction=prediction,
                    target=target,
                    correction=corrections[arm],
                )
        correction_rows.append(
            {
                "family": case.family,
                "case_id": case.case_id,
                "resolution": case.resolution_name,
                "mode": "teacher_forced",
                "call": call,
                "arm": "proposal_diagnostic",
                "filter_cap": filter_cap,
                "eligible_edge_count": local["local_summary"].eligible_edge_count,
                "applied_local_filter_relative_norm": local[
                    "local_summary"
                ].applied_relative_norm,
                "maximum_local_weighted_mean_closure": float(
                    np.max(np.abs(local["local_summary"].weighted_mean_closure))
                ),
            }
        )

    states = {arm: np.array(case.reference_states[0], copy=True) for arm in ARMS}
    active = {arm: True for arm in ARMS}
    valid_length = {arm: 0 for arm in ARMS}
    visual: dict[str, list[np.ndarray]] = defaultdict(list)
    accumulated_correction = np.zeros_like(case.reference_states[0])
    accumulated_baseline_error = np.zeros_like(case.reference_states[0])
    accumulated_combined_error = np.zeros_like(case.reference_states[0])
    for call in range(1, calls + 1):
        target = case.reference_states[call]
        step_payload = {}
        for arm in ARMS:
            if not active[arm]:
                continue
            current = states[arm]
            base_prediction = _predict(model, case, current)
            base_admissibility = _safe_admissibility_summary(
                base_prediction, gamma=case.gamma
            )
            if not base_admissibility["finite"]:
                completion_rows.append(
                    {
                        "family": case.family,
                        "case_id": case.case_id,
                        "resolution": case.resolution_name,
                        "arm": arm,
                        "call": call,
                        "accepted": False,
                        "failure_stage": "base_prediction_nonfinite",
                        **base_admissibility,
                    }
                )
                correction_rows.append(
                    {
                        "family": case.family,
                        "case_id": case.case_id,
                        "resolution": case.resolution_name,
                        "mode": "free_rollout",
                        "call": call,
                        "arm": arm,
                        "filter_cap": filter_cap,
                        "failure_stage": "base_prediction_nonfinite",
                        "eligible_edge_count": None,
                        "applied_local_filter_relative_norm": None,
                        "maximum_local_weighted_mean_closure": None,
                    }
                )
                active[arm] = False
                continue
            predictions, corrections, local = _proposal_arms(
                case,
                current=current,
                base_prediction=base_prediction,
                bias=bias_sequence[call - 1],
                filter_cap=filter_cap,
            )
            prediction = predictions[arm]
            correction = corrections[arm]
            admissibility = _safe_admissibility_summary(prediction, gamma=case.gamma)
            accepted = bool(admissibility["finite"] and admissibility["admissible"])
            _append_metrics(
                metric_rows,
                component_rows,
                front_rows,
                case,
                mode="free_rollout",
                call=call,
                arm=arm,
                current=current,
                prediction=prediction,
                target=target,
                correction=correction,
                shock_quantile=shock_quantile,
            )
            if call in band_calls:
                _append_band_metrics(
                    band_rows,
                    case,
                    mode="free_rollout",
                    call=call,
                    arm=arm,
                    current=current,
                    prediction=prediction,
                    target=target,
                    correction=correction,
                )
            completion_rows.append(
                {
                    "family": case.family,
                    "case_id": case.case_id,
                    "resolution": case.resolution_name,
                    "arm": arm,
                    "call": call,
                    "accepted": accepted,
                    **admissibility,
                }
            )
            correction_rows.append(
                {
                    "family": case.family,
                    "case_id": case.case_id,
                    "resolution": case.resolution_name,
                    "mode": "free_rollout",
                    "call": call,
                    "arm": arm,
                    "filter_cap": filter_cap,
                    "eligible_edge_count": local["local_summary"].eligible_edge_count,
                    "applied_local_filter_relative_norm": (
                        local["local_summary"].applied_relative_norm
                        if arm in {"local_dissipation", "combined"}
                        else 0.0
                    ),
                    "maximum_local_weighted_mean_closure": (
                        float(
                            np.max(np.abs(local["local_summary"].weighted_mean_closure))
                        )
                        if arm in {"local_dissipation", "combined"}
                        else 0.0
                    ),
                }
            )
            step_payload[arm] = {
                "current": current,
                "base_prediction": base_prediction,
                "prediction": prediction,
                "correction": correction,
                "local": local,
                "accepted": accepted,
            }
            if accepted:
                states[arm] = prediction
                valid_length[arm] = call
            else:
                active[arm] = False
        if save_visual and "baseline" in step_payload and "combined" in step_payload:
            truth_increment = target - case.reference_states[call - 1]
            baseline_increment = (
                step_payload["baseline"]["prediction"]
                - step_payload["baseline"]["current"]
            )
            combined_increment = (
                step_payload["combined"]["prediction"]
                - step_payload["combined"]["current"]
            )
            baseline_error = baseline_increment - truth_increment
            combined_error = combined_increment - truth_increment
            accumulated_correction = (
                accumulated_correction + step_payload["combined"]["correction"]
            )
            accumulated_baseline_error = accumulated_baseline_error + baseline_error
            accumulated_combined_error = accumulated_combined_error + combined_error
            for name, value in (
                ("true_increment", truth_increment),
                ("baseline_increment", baseline_increment),
                ("combined_increment", combined_increment),
                ("baseline_residual_error", baseline_error),
                ("combined_residual_error", combined_error),
                ("persistent_correction", -bias_sequence[call - 1]),
                (
                    "local_correction",
                    step_payload["combined"]["local"]["local_correction"],
                ),
                ("accumulated_correction", accumulated_correction),
                ("accumulated_baseline_residual_error", accumulated_baseline_error),
                ("accumulated_combined_residual_error", accumulated_combined_error),
                ("sensor", step_payload["combined"]["local"]["sensor"]),
                (
                    "baseline_accepted",
                    np.asarray(step_payload["baseline"]["accepted"], dtype=bool),
                ),
                (
                    "combined_accepted",
                    np.asarray(step_payload["combined"]["accepted"], dtype=bool),
                ),
            ):
                visual[name].append(np.asarray(value, dtype=np.float32))
    for arm in ARMS:
        completion_rows.append(
            {
                "family": case.family,
                "case_id": case.case_id,
                "resolution": case.resolution_name,
                "arm": arm,
                "call": "summary",
                "accepted": valid_length[arm] == calls,
                "valid_length": valid_length[arm],
                "completion_fraction": valid_length[arm] / calls,
            }
        )
    if save_visual and visual:
        payload = {
            "schema": np.asarray(SCHEMA),
            "family": np.asarray(case.family),
            "case_id": np.asarray(case.case_id),
            "resolution": np.asarray(case.resolution_name),
            "mode": np.asarray("free_rollout"),
            "residual_error_contract": np.asarray(
                "free-input predicted increment minus reference-trajectory increment"
            ),
            "accumulated_correction_contract": np.asarray(
                "sum of direct combined-arm corrections; not the recurrent state gap"
            ),
            "nodes": case.nodes.astype(np.float64),
            "physical_times": case.physical_times[
                1 : len(visual["true_increment"]) + 1
            ],
            "residual_scale": case.residual_scale.astype(np.float64),
            "node_type": case.physical_node_type.astype(np.int64),
            "expected_rollout_calls": np.asarray(calls, dtype=np.int64),
            "baseline_valid_length": np.asarray(
                valid_length["baseline"], dtype=np.int64
            ),
            "combined_valid_length": np.asarray(
                valid_length["combined"], dtype=np.int64
            ),
            "rollout_complete": np.asarray(
                valid_length["baseline"] == calls and valid_length["combined"] == calls,
                dtype=bool,
            ),
        }
        payload.update({name: np.asarray(values) for name, values in visual.items()})
        np.savez_compressed(
            visual_dir / f"{case.family}_{case.case_id}_{case.resolution_name}.npz",
            **payload,
        )


def _promotion_summary(
    family: str,
    metric_rows: Sequence[Mapping[str, Any]],
    front_rows: Sequence[Mapping[str, Any]],
    completion_rows: Sequence[Mapping[str, Any]],
    horizon: int,
    expected_keys: Sequence[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    summaries = [row for row in completion_rows if row["call"] == "summary"]
    completed = {
        (str(row["case_id"]), str(row["resolution"]), str(row["arm"])): bool(
            row["accepted"]
        )
        for row in summaries
    }
    evaluation_keys = (
        sorted((str(case_id), str(resolution)) for case_id, resolution in expected_keys)
        if expected_keys is not None
        else sorted(
            {(str(row["case_id"]), str(row["resolution"])) for row in summaries}
        )
    )
    completion_pass = all(
        completed.get((*key, "baseline"), False)
        and completed.get((*key, "combined"), False)
        for key in evaluation_keys
    )

    def metric_ratios(
        rows: Sequence[Mapping[str, Any]], *, region: str | None, field: str
    ) -> tuple[dict[tuple[str, str], float], list[dict[str, Any]]]:
        indexed = {
            (str(row["case_id"]), str(row["resolution"]), str(row["arm"])): row
            for row in rows
            if row["mode"] == "free_rollout"
            and int(row["call"]) == horizon
            and (region is None or row.get("region") == region)
            and row["arm"] in {"baseline", "combined"}
        }
        ratios: dict[tuple[str, str], float] = {}
        invalid = []
        for key in evaluation_keys:
            baseline = indexed.get((*key, "baseline"))
            combined = indexed.get((*key, "combined"))
            if baseline is None or combined is None:
                invalid.append(
                    {"case_id": key[0], "resolution": key[1], "reason": "missing"}
                )
                continue
            denominator = baseline.get(field)
            numerator = combined.get(field)
            if denominator is None or numerator is None:
                invalid.append(
                    {"case_id": key[0], "resolution": key[1], "reason": "null"}
                )
                continue
            denominator_value = float(denominator)
            numerator_value = float(numerator)
            if not np.isfinite(denominator_value) or denominator_value <= 0.0:
                invalid.append(
                    {
                        "case_id": key[0],
                        "resolution": key[1],
                        "reason": "nonpositive_or_nonfinite_baseline",
                    }
                )
                continue
            if not np.isfinite(numerator_value):
                invalid.append(
                    {
                        "case_id": key[0],
                        "resolution": key[1],
                        "reason": "nonfinite_combined",
                    }
                )
                continue
            ratios[key] = numerator_value / denominator_value
        return ratios, invalid

    state_ratios, invalid_state = metric_ratios(
        metric_rows, region="all", field="state_error_rms"
    )
    control_specs = [
        ("shock_state", "shock", "state_error_rms"),
        ("smooth_highpass", "smooth", "state_error_graph_highpass_rms"),
        ("boundary_state", "boundary_nodes", "state_error_rms"),
    ]
    if family == "dynamic_fv":
        control_specs.append(("vortex_state", "vortex", "state_error_rms"))
    controls: dict[str, dict[tuple[str, str], float]] = {}
    invalid_controls: dict[str, list[dict[str, Any]]] = {}
    for name, region, field in control_specs:
        ratios, invalid = metric_ratios(metric_rows, region=region, field=field)
        controls[name] = ratios
        invalid_controls[name] = invalid

    descriptive_proxies: dict[str, dict[tuple[str, str], float]] = {}
    invalid_proxies: dict[str, list[dict[str, Any]]] = {}
    for output_name, field in (
        ("shock_proxy_centroid", "proxy_centroid_absolute_error"),
        ("shock_proxy_strength", "proxy_strength_absolute_error"),
        ("shock_proxy_thickness", "proxy_thickness_absolute_error"),
    ):
        ratios, invalid = metric_ratios(front_rows, region=None, field=field)
        descriptive_proxies[output_name] = ratios
        invalid_proxies[output_name] = invalid

    median_state_ratio = (
        None if not state_ratios else float(np.median(list(state_ratios.values())))
    )
    state_pass = bool(
        len(state_ratios) == len(evaluation_keys)
        and not invalid_state
        and all(value <= 0.95 for value in state_ratios.values())
    )
    control_pass = all(
        len(controls[name]) == len(evaluation_keys)
        and not invalid_controls[name]
        and all(value <= 1.05 for value in controls[name].values())
        for name in controls
    )
    promotion_pass = bool(completion_pass and state_pass and control_pass)
    label = lambda key: f"{key[0]}@{key[1]}"
    return {
        "promoted": promotion_pass,
        "completion_pass": completion_pass,
        "median_combined_to_baseline_final_state_error_ratio": median_state_ratio,
        "per_case_combined_to_baseline_final_state_error_ratio": {
            label(key): value for key, value in state_ratios.items()
        },
        "state_ratio_complete_and_per_case_pass": state_pass,
        "minimum_state_improvement": 0.05,
        "control_ratio_limit": 1.05,
        "control_median_ratios": {
            name: (None if not values else float(np.median(list(values.values()))))
            for name, values in controls.items()
        },
        "control_per_case_ratios": {
            name: {label(key): value for key, value in values.items()}
            for name, values in controls.items()
        },
        "control_complete_and_per_case_pass": control_pass,
        "descriptive_reference_mask_shock_proxy_median_ratios": {
            name: (None if not values else float(np.median(list(values.values()))))
            for name, values in descriptive_proxies.items()
        },
        "descriptive_reference_mask_shock_proxy_per_case_ratios": {
            name: {label(key): value for key, value in values.items()}
            for name, values in descriptive_proxies.items()
        },
        "invalid_state_ratios": invalid_state,
        "invalid_control_ratios": invalid_controls,
        "invalid_descriptive_proxy_ratios": invalid_proxies,
        "evaluation_case_resolution_count": len(evaluation_keys),
    }


def _rollout_completion_checks(
    completion_rows: Sequence[Mapping[str, Any]],
    evaluation_keys: Sequence[tuple[str, str]],
    horizon: int,
) -> dict[str, Any]:
    expected_steps = {
        (str(case_id), str(resolution), arm, call)
        for case_id, resolution in evaluation_keys
        for arm in ARMS
        for call in range(1, horizon + 1)
    }
    step_rows = [row for row in completion_rows if row["call"] != "summary"]
    observed_steps = [
        (
            str(row["case_id"]),
            str(row["resolution"]),
            str(row["arm"]),
            int(row["call"]),
        )
        for row in step_rows
    ]
    expected_summaries = {
        (str(case_id), str(resolution), arm)
        for case_id, resolution in evaluation_keys
        for arm in ARMS
    }
    summary_rows = [row for row in completion_rows if row["call"] == "summary"]
    observed_summaries = [
        (str(row["case_id"]), str(row["resolution"]), str(row["arm"]))
        for row in summary_rows
    ]
    step_matrix_complete = (
        len(observed_steps) == len(expected_steps)
        and set(observed_steps) == expected_steps
    )
    summary_matrix_complete = (
        len(observed_summaries) == len(expected_summaries)
        and set(observed_summaries) == expected_summaries
    )
    all_steps_accepted = step_matrix_complete and all(
        bool(row["accepted"]) for row in step_rows
    )
    all_summaries_accepted = summary_matrix_complete and all(
        bool(row["accepted"]) for row in summary_rows
    )
    return {
        "expected_step_rows": len(expected_steps),
        "observed_step_rows": len(observed_steps),
        "step_matrix_complete": step_matrix_complete,
        "all_steps_accepted": all_steps_accepted,
        "expected_summary_rows": len(expected_summaries),
        "observed_summary_rows": len(observed_summaries),
        "summary_matrix_complete": summary_matrix_complete,
        "all_summaries_accepted": all_summaries_accepted,
        "complete_and_admissible": bool(all_steps_accepted and all_summaries_accepted),
    }


def _teacher_forced_coverage_checks(
    rows: Sequence[Mapping[str, Any]],
    evaluation_keys: Sequence[tuple[str, str]],
    horizon: int,
) -> dict[str, Any]:
    expected = {
        (str(case_id), str(resolution), call)
        for case_id, resolution in evaluation_keys
        for call in range(1, horizon + 1)
    }
    observed = [
        (str(row["case_id"]), str(row["resolution"]), int(row["call"])) for row in rows
    ]
    matrix_complete = len(observed) == len(expected) and set(observed) == expected
    all_finite = matrix_complete and all(bool(row["finite"]) for row in rows)
    return {
        "expected_rows": len(expected),
        "observed_rows": len(observed),
        "matrix_complete": matrix_complete,
        "all_finite": all_finite,
        "complete_and_finite": bool(matrix_complete and all_finite),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = perf_counter()
    device = select_device(args.device)
    store = PCNOEuler2DShardStore(args.data_dir, max_cached_trajectories=2)
    if args.family == "dynamic_fv":
        checkpoint = load_resolution_checkpoint(args.checkpoint)
    else:
        checkpoint = load_checkpoint_payload(args.checkpoint)
    provenance = _verify_common_provenance(args, checkpoint, store)
    split = json.loads(args.split_json.read_text(encoding="utf-8"))
    validation_ids = tuple(str(value) for value in split.get("val_keys", ()))
    evaluation_ids = _evaluation_ids(args.family)
    if not validation_ids or any(
        value not in validation_ids for value in evaluation_ids
    ):
        raise ValueError("D071 evaluation cases are not in the frozen validation split")
    calibration_ids = set(validation_ids) - set(evaluation_ids)
    if not calibration_ids:
        raise ValueError("D071 requires a nonempty disjoint calibration subset")
    if args.family == "dynamic_fv":
        model, _ = build_resolution_checkpoint_model(checkpoint, device)
    else:
        model, _ = build_checkpoint_model(
            checkpoint, device, model_node_type_input="physical"
        )
    model.eval()
    args.case_ids = list(validation_ids)
    reference_checks: list[dict[str, Any]] = []
    if args.family == "dynamic_fv":
        cases, reference_checks = _load_dynamic_cases(
            args, checkpoint, store, device=device
        )
    else:
        cases = _load_bump_cases(args, checkpoint, store, device=device)
    cases_by_id = {case.case_id: case for case in cases if case.family == "bump"}
    bump_replay = (
        _bump_replay_gate(args, model, cases_by_id, device=device)
        if args.family == "bump"
        else None
    )
    evaluation_cases = [case for case in cases if case.case_id in set(evaluation_ids)]
    calibration_cases = [case for case in cases if case.case_id in calibration_ids]
    hook_equivalence = _run_hook_equivalence(
        model,
        evaluation_cases[0],
        device=device,
    )
    args.output_dir.mkdir(parents=True)
    visual_dir = args.output_dir / "visual_payloads"
    visual_dir.mkdir()
    registered_calibration_ids = set(calibration_ids)
    registered_evaluation_ids = evaluation_ids
    if args.smoke:
        calibration_cases = calibration_cases[:1]
        evaluation_cases = evaluation_cases[:1]
        calibration_ids = {calibration_cases[0].case_id}
        evaluation_ids = (evaluation_cases[0].case_id,)
        for case in (*calibration_cases, *evaluation_cases):
            case.reference_states = case.reference_states[:2]
            case.physical_times = case.physical_times[:2]
    expected_evaluation_keys = sorted({_case_key(case) for case in evaluation_cases})
    frozen_coefficients, filter_cap, calibration_rows = _calibrate(
        model, calibration_cases, calibration_ids
    )
    coefficient_payload = {
        "schema": np.asarray(SCHEMA),
        "family": np.asarray(args.family),
        "rank": np.asarray(RANK),
        "filter_cap": np.asarray(filter_cap),
    }
    coefficient_payload.update(
        {f"coefficients__{key}": value for key, value in frozen_coefficients.items()}
    )
    np.savez_compressed(
        args.output_dir / "calibration_coefficients.npz", **coefficient_payload
    )

    default_visual = (
        set(BUMP_EVALUATION_CASES)
        if args.family == "bump"
        else {DYNAMIC_EVALUATION_CASES[0], DYNAMIC_EVALUATION_CASES[-1]}
    )
    visual_ids = (
        set(args.visualization_cases) if args.visualization_cases else default_visual
    )
    metric_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    band_rows: list[dict[str, Any]] = []
    front_rows: list[dict[str, Any]] = []
    correction_rows: list[dict[str, Any]] = []
    completion_rows: list[dict[str, Any]] = []
    teacher_completion_rows: list[dict[str, Any]] = []
    try:
        for index, case in enumerate(evaluation_cases, start=1):
            save_visual = case.case_id in visual_ids and (
                case.family == "bump" or case.resolution_name == "250x100"
            )
            _evaluate_case(
                model,
                case,
                frozen_coefficients=frozen_coefficients,
                filter_cap=filter_cap,
                shock_quantile=args.shock_quantile,
                metric_rows=metric_rows,
                component_rows=component_rows,
                band_rows=band_rows,
                front_rows=front_rows,
                correction_rows=correction_rows,
                completion_rows=completion_rows,
                teacher_completion_rows=teacher_completion_rows,
                save_visual=save_visual,
                visual_dir=visual_dir,
            )
            print(
                f"evaluated {index}/{len(evaluation_cases)} {case.case_id} "
                f"{case.resolution_name}",
                flush=True,
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        store.close()

    maximum_balance_closure = max(
        (
            float(row["maximum_local_weighted_mean_closure"])
            for row in correction_rows
            if row.get("maximum_local_weighted_mean_closure") is not None
        ),
        default=0.0,
    )
    maximum_band_closure = max(
        (float(row["maximum_reconstruction_closure"]) for row in band_rows),
        default=0.0,
    )
    promotion = (
        {"promoted": False, "reason": "not_run_smoke"}
        if args.smoke
        else _promotion_summary(
            args.family,
            metric_rows,
            front_rows,
            completion_rows,
            args.rollout_calls,
            expected_evaluation_keys,
        )
    )
    outputs = {
        "calibration.csv": calibration_rows,
        "metrics.csv": metric_rows,
        "component_metrics.csv": component_rows,
        "band_metrics.csv": band_rows,
        "front_metrics.csv": front_rows,
        "correction_metrics.csv": correction_rows,
        "completion.csv": completion_rows,
        "teacher_completion.csv": teacher_completion_rows,
    }
    if reference_checks:
        outputs["reference_checks.csv"] = reference_checks
    for name, rows in outputs.items():
        write_csv_with_paths(args.output_dir / name, rows)
    output_hashes = {name: sha256_file(args.output_dir / name) for name in outputs}
    output_hashes["calibration_coefficients.npz"] = sha256_file(
        args.output_dir / "calibration_coefficients.npz"
    )
    output_hashes.update(
        {
            str(path.relative_to(args.output_dir)).replace("\\", "/"): sha256_file(path)
            for path in sorted(visual_dir.glob("*.npz"))
        }
    )
    provenance["loaded_project_source_sha256"] = _loaded_project_source_hashes()
    completion_checks = _rollout_completion_checks(
        completion_rows,
        expected_evaluation_keys,
        1 if args.smoke else args.rollout_calls,
    )
    teacher_checks = _teacher_forced_coverage_checks(
        teacher_completion_rows,
        expected_evaluation_keys,
        1 if args.smoke else args.rollout_calls,
    )
    reference_binding_pass = bool(
        args.family != "dynamic_fv" or len(reference_checks) == len(validation_ids)
    )
    replay_pass = bool(bump_replay is None or bump_replay.get("passed"))
    core_contract_pass = bool(
        maximum_balance_closure <= 1.0e-10
        and maximum_band_closure <= 2.0e-5
        and hook_equivalence["passed"]
        and reference_binding_pass
        and replay_pass
        and teacher_checks["complete_and_finite"]
    )
    smoke_ready = bool(
        core_contract_pass and completion_checks["complete_and_admissible"]
    )
    contract_pass = smoke_ready if args.smoke else core_contract_pass
    status = (
        "smoke_complete"
        if args.smoke and contract_pass
        else (
            "smoke_failed"
            if args.smoke
            else "complete" if contract_pass else "failed_contract"
        )
    )
    summary = {
        "schema": SCHEMA,
        "status": status,
        "contract_checks_passed": contract_pass,
        "scientific_interpretation_allowed": contract_pass and not args.smoke,
        "family": args.family,
        "args": jsonable_args(args),
        "population": {
            "split": "validation",
            "calibration_case_ids": sorted(calibration_ids),
            "evaluation_case_ids": list(evaluation_ids),
            "registered_calibration_case_ids": sorted(registered_calibration_ids),
            "registered_evaluation_case_ids": list(registered_evaluation_ids),
            "sealed_populations_accessed": False,
        },
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "normalization_digest": checkpoint.get("normalization_digest"),
        "boundary_policy": "model_all_nodes raw recurrence; base policy unchanged",
        "correction_contract": {
            "persistent_rank": RANK,
            "basis": "separable physical cosine ordered by physical wavenumber",
            "filter_cap_candidates": list(FILTER_CAPS),
            "selected_filter_cap": filter_cap,
            "sensor_quantile": SENSOR_QUANTILE,
            "dynamic_balance": "physical-volume mean neutral added correction",
            "bump_balance": "proxy-weight mean neutral added correction only",
            "oracle_scope": "teacher-forced headroom only; never free rollout",
            "band_calls": (
                [1, 5, 15, args.rollout_calls]
                if args.family == "dynamic_fv"
                else [1, 5, 10, args.rollout_calls]
            ),
            "dynamic_bands": "physical DCT wavelengths",
            "bump_bands": "nonorthogonal graph proxy only",
        },
        "checks": {
            "maximum_local_dissipation_weighted_mean_closure": (
                maximum_balance_closure
            ),
            "local_dissipation_weighted_mean_closure_pass": (
                maximum_balance_closure <= 1.0e-10
            ),
            "maximum_scale_band_reconstruction_closure": maximum_band_closure,
            "scale_band_reconstruction_closure_pass": (maximum_band_closure <= 2.0e-5),
            "hook_equivalence_pass": bool(hook_equivalence["passed"]),
            "reference_binding_pass": reference_binding_pass,
            "bump_replay_pass": replay_pass,
            "rollout_completion": completion_checks,
            "teacher_forced_coverage": teacher_checks,
            "core_contract_pass": core_contract_pass,
            "smoke_ready": smoke_ready,
        },
        "promotion": promotion,
        "claim_boundary": {
            "bump": "native graphs only; no resolution-transfer or physical-conservation claim",
            "dynamic": "audited volumes, but the base PCNO is not a conservative solver",
            "filter": "controlled proposal smoothing, not entropy-stable dissipation",
            "assimilation": "not implemented",
        },
        "provenance": provenance,
        "bump_open_validation_replay": bump_replay,
        "hook_equivalence": hook_equivalence,
        "runtime": runtime_environment(device),
        "git": git_state(),
        "row_counts": {name: len(rows) for name, rows in outputs.items()},
        "elapsed_seconds": perf_counter() - started,
        "output_hashes": output_hashes,
    }
    write_json(args.output_dir / "summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    summary = run(parse_args(argv))
    print(
        f"D071 {summary['family']} status={summary['status']} "
        f"promoted={summary['promotion']['promoted']}",
        flush=True,
    )
    return 0 if summary["status"] in {"complete", "smoke_complete"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
