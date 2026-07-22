"""Run frozen-checkpoint D013 ripple diagnostics for 2D Euler PCNO.

This entry point performs no training and applies no clipping, floors,
smoothing, or future-reference boundary replacement. It records raw model
recurrence and preserves whether the primary PCNO weights are validated physical
cell volumes or only reconstructed proxies; equal-node weights remain a proxy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pcno.pcno import compute_Fourier_modes  # noqa: E402
from utility.time_dependent_no.euler2d_metrics import (  # noqa: E402
    front_centroid_distance,
    front_distance_metrics,
    front_overlap_metrics,
    front_region_masks,
    local_shift_alignment_metrics,
    median_edge_length,
    near_shock_error,
    shift_grid,
    shock_front_masks,
    shock_smearing_metrics,
)
from utility.time_dependent_no.pcno_euler2d import (  # noqa: E402
    Euler2DNormalization,
    PCNOEuler2DResidual,
    PCNOEuler2DShardStore,
    reference_smooth_region_mask,
)
from utility.time_dependent_no.pcno_ripple_diagnostics import (  # noqa: E402
    RIPPLE_DIAGNOSTIC_SCHEMA,
    conservative_to_primitive_raw,
    estimate_generalized_laplacian_radius,
    fourier_gram_audit,
    fourier_reconstruction_audit,
    geometry_conditioning_features,
    graph_distance_to_mask,
    graph_spectral_bands,
    induced_subgraph,
    node_highpass_amplitude,
    normalized_node_weights,
    raw_admissibility_summary,
    spatial_correlation_summary,
    trace_pcno_branches,
    weighted_relative_l2_numpy,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trajectory-keys", nargs="*", default=None)
    parser.add_argument("--trajectory-count", type=int, default=5)
    parser.add_argument("--deep-trajectory-count", type=int, default=2)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument(
        "--diagnostic-calls",
        type=int,
        nargs="+",
        default=[1, 2, 5, 10, 15, 20],
    )
    parser.add_argument(
        "--trace-calls",
        type=int,
        nargs="+",
        default=[1, 10, 20],
    )
    parser.add_argument("--basis-call", type=int, default=10)
    parser.add_argument("--perturbation-call", type=int, default=10)
    parser.add_argument("--perturbation-fraction", type=float, default=0.25)
    parser.add_argument("--lanczos-steps", type=int, default=20)
    parser.add_argument("--shock-quantile", type=float, default=0.9)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=2403)
    parser.add_argument("--line2-d014-summary", type=Path)
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.trajectory_count < 1 or args.deep_trajectory_count < 0:
        raise ValueError("trajectory counts must be positive/nonnegative")
    if args.deep_trajectory_count > args.trajectory_count:
        raise ValueError("deep trajectory count cannot exceed trajectory count")
    if args.start_frame < 0 or args.num_steps < 1:
        raise ValueError("start frame and number of steps are invalid")
    for name in ("diagnostic_calls", "trace_calls"):
        values = sorted(set(int(value) for value in getattr(args, name)))
        if not values or values[0] < 1 or values[-1] > args.num_steps:
            raise ValueError(f"{name} must lie within the requested rollout")
        setattr(args, name, values)
    if args.num_steps not in args.trace_calls:
        args.trace_calls.append(args.num_steps)
        args.trace_calls.sort()
    for name in ("basis_call", "perturbation_call"):
        value = int(getattr(args, name))
        if value < 1 or value > args.num_steps:
            raise ValueError(f"{name} must lie within the requested rollout")
    if not 0.0 < args.perturbation_fraction < 1.0:
        raise ValueError("perturbation fraction must lie in (0,1)")
    if args.lanczos_steps < 2:
        raise ValueError("Lanczos steps must be at least two")
    if not 0.0 < args.shock_quantile < 1.0:
        raise ValueError("shock quantile must lie in (0,1)")


def select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(name)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    required = {
        "model_state",
        "model_config",
        "normalization",
        "data_manifest_digest",
        "step_stride",
        "val_keys",
        "boundary_mode",
        "raw_recurrence",
        "config_digest",
    }
    missing = sorted(required - set(checkpoint))
    if missing:
        raise ValueError(f"checkpoint is missing required fields: {missing}")
    return dict(checkpoint)


def build_model(
    checkpoint: Mapping[str, Any],
    device: torch.device,
) -> PCNOEuler2DResidual:
    config = checkpoint["model_config"]
    normalization = Euler2DNormalization.from_mapping(checkpoint["normalization"])
    model = PCNOEuler2DResidual(
        normalization=normalization,
        k_max=int(config["k_max"]),
        domain_lengths=config["domain_lengths"],
        layers=config["layers"],
        fc_dim=int(config["fc_dim"]),
        nmeasures=int(config["nmeasures"]),
        zero_initialize=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model


def select_trajectory_keys(
    args: argparse.Namespace,
    checkpoint: Mapping[str, Any],
    store: PCNOEuler2DShardStore,
) -> tuple[list[str], str]:
    if args.trajectory_keys:
        keys = [str(key) for key in args.trajectory_keys]
        source = "explicit_cli"
    else:
        split_path = args.checkpoint.parent / "split.json"
        if split_path.is_file():
            split = json.loads(split_path.read_text(encoding="utf-8"))
            keys = [str(key) for key in split.get("rollout_keys", [])]
            source = "checkpoint_sibling_split_rollout_keys"
        else:
            keys = [str(key) for key in checkpoint["val_keys"]]
            source = "checkpoint_validation_prefix"
    if len(keys) < args.trajectory_count:
        raise ValueError(
            f"only {len(keys)} trajectory keys available for requested "
            f"count {args.trajectory_count}"
        )
    keys = keys[: args.trajectory_count]
    missing = sorted(set(keys) - set(store.keys))
    if missing:
        raise KeyError(f"selected trajectory keys are absent from shards: {missing}")
    return keys, source


def line2_interface(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {
            "status": "pending_line2",
            "ownership": "Line 2 owns CPG D014; no competing reach computation",
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "status": "consumed",
        "sha256": sha256_file(path),
        "schema": payload.get("schema"),
        "dependency_trace": payload.get("dependency_trace"),
        "characteristic_contract": payload.get("characteristic_contract"),
        "rows_with_directed_cone_outside_support": payload.get(
            "rows_with_directed_cone_outside_support"
        ),
        "ownership": "Line 2 output consumed without recomputing CPG reach",
    }


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(json_safe(value), sort_keys=True, allow_nan=False) + "\n"
        )


def model_call(
    model: PCNOEuler2DResidual,
    sample: Mapping[str, torch.Tensor],
    current: torch.Tensor,
) -> torch.Tensor:
    return model(
        current,
        node_mask=sample["node_mask"],
        nodes=sample["nodes"],
        node_weights=sample["node_weights"],
        node_rhos=sample["node_rhos"],
        directed_edges=sample["directed_edges"],
        edge_gradient_weights=sample["edge_gradient_weights"],
        node_type=sample["node_type"],
        mach=sample["mach"],
    )


def _scalar(value: Any) -> float | int | None:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError("expected a scalar metric")
    item = array.reshape(-1)[0]
    if isinstance(item, (np.integer, int)):
        return int(item)
    numeric = float(item)
    return numeric if math.isfinite(numeric) else None


def shock_diagnostics(
    prediction: np.ndarray,
    target: np.ndarray,
    positions: np.ndarray,
    edges: np.ndarray,
    node_type: np.ndarray,
    proxy_weights: np.ndarray,
    component_scale: np.ndarray,
    *,
    gamma: float,
    quantile: float,
    phase_alignment: bool,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    pred_primitive = conservative_to_primitive_raw(prediction, gamma=gamma)
    target_primitive = conservative_to_primitive_raw(target, gamma=gamma)
    if not np.isfinite(pred_primitive).all():
        return {"status": "unavailable_nonfinite_primitive"}, {}
    interior = np.asarray(node_type).reshape(-1) == 0
    shock_domain = "interior"
    if np.count_nonzero(interior) < 3:
        interior = np.ones_like(interior)
        shock_domain = "all_nodes_small_fixture_fallback"
    fronts = shock_front_masks(
        pred_primitive,
        target_primitive,
        edges,
        quantile=quantile,
        node_mask=interior,
    )
    regions = front_region_masks(
        fronts["prediction_mask"],
        fronts["target_mask"],
        node_mask=interior,
    )
    region_contract = "prediction_target_front_union"
    if not np.any(regions["front_union"]) or not np.any(regions["smooth"]):
        target_scores = np.asarray(fronts["target_scores"], dtype=np.float64)
        selected_nodes = np.flatnonzero(interior)
        fallback_front = np.zeros_like(interior)
        fallback_front[
            selected_nodes[int(np.argmax(target_scores[selected_nodes]))]
        ] = True
        regions["front_union"] = fallback_front
        regions["target_front"] = fallback_front
        regions["smooth"] = interior & ~fallback_front
        region_contract = "single_max_target_score_small_fixture_fallback"
    overlap = front_overlap_metrics(fronts["prediction_mask"], fronts["target_mask"])
    distance = front_distance_metrics(
        fronts["prediction_mask"],
        fronts["target_mask"],
        positions,
    )
    pred_interior, interior_edges, _ = induced_subgraph(
        pred_primitive,
        edges,
        np.ones(pred_primitive.shape[0]),
        interior,
    )
    target_interior, _, _ = induced_subgraph(
        target_primitive,
        edges,
        np.ones(target_primitive.shape[0]),
        interior,
    )
    smearing = shock_smearing_metrics(
        pred_interior,
        target_interior,
        interior_edges,
        scalar_index=3,
    )
    near = near_shock_error(
        pred_primitive,
        target_primitive,
        regions["front_union"],
    )
    proxy_mass = normalized_node_weights(proxy_weights, name="shock error")
    scaled_error = (prediction - target) / component_scale.reshape(1, -1)
    error_energy = proxy_mass * np.sum(scaled_error**2, axis=-1)
    front_energy_fraction = float(
        error_energy[regions["front_union"]].sum()
        / max(float(error_energy.sum()), 1e-12)
    )
    summary: dict[str, Any] = {
        "status": "available",
        "shock_node_domain": shock_domain,
        "front_region_contract": region_contract,
        "median_edge_length": median_edge_length(positions, edges),
        "front_iou": _scalar(overlap["iou"]),
        "front_f1": _scalar(overlap["f1"]),
        "front_symmetric_chamfer": _scalar(distance["symmetric_chamfer_mean"]),
        "front_hausdorff": _scalar(distance["hausdorff"]),
        "front_centroid_distance_equal_node": _scalar(
            front_centroid_distance(
                fronts["prediction_mask"],
                fronts["target_mask"],
                positions,
            )
        ),
        "front_centroid_distance_proxy_weighted": _scalar(
            front_centroid_distance(
                fronts["prediction_mask"],
                fronts["target_mask"],
                positions,
                prediction_weights=proxy_mass,
                target_weights=proxy_mass,
            )
        ),
        "thickness_ratio": _scalar(smearing["thickness_ratio"]),
        "strength_ratio": _scalar(smearing["strength_ratio"]),
        "shock_relative_l2_primitive": _scalar(near["shock_relative_l2"]),
        "smooth_relative_l2_primitive": _scalar(near["smooth_relative_l2"]),
        "scaled_error_energy_fraction_on_front_union": front_energy_fraction,
        "front_union_fraction_interior": float(
            np.count_nonzero(regions["front_union"])
            / max(np.count_nonzero(interior), 1)
        ),
    }
    if phase_alignment:
        edge_scale = median_edge_length(positions, edges)
        alignment = local_shift_alignment_metrics(
            pred_primitive,
            target_primitive,
            positions,
            regions["front_union"],
            shift_grid(2.0 * edge_scale, grid_size=5, dim=2),
            scalar_index=3,
        )
        summary["local_shift_alignment"] = {
            "max_shift_in_median_edges": 2.0,
            "relative_pressure_rmse_reduction": _scalar(
                alignment["relative_rmse_reduction"]
            ),
            "best_shift_norm": _scalar(alignment["best_shift_norm"]),
            "best_shift": np.asarray(alignment["best_shift"]).reshape(-1).tolist(),
        }
    return summary, {
        "target_front": regions["target_front"],
        "predicted_front": regions["predicted_front"],
        "front_union": regions["front_union"],
        "smooth": regions["smooth"],
        "interior": interior,
    }


def call_diagnostics(
    *,
    trajectory: str,
    source: str,
    call_index: int,
    prediction: np.ndarray,
    target: np.ndarray,
    positions: np.ndarray,
    edges: np.ndarray,
    directed_edges: np.ndarray,
    node_type: np.ndarray,
    node_measures: np.ndarray,
    proxy_weights: np.ndarray,
    component_scale: np.ndarray,
    gamma: float,
    shock_quantile: float,
    lanczos_steps: int,
    deep_spectrum: bool,
    full_spectral_radii: Mapping[str, float],
    static_geometry_features: Mapping[str, np.ndarray],
    phase_alignment: bool,
    seed: int,
) -> dict[str, Any]:
    equal_weights = np.ones(positions.shape[0], dtype=np.float64)
    scaled_error = (prediction - target) / component_scale.reshape(1, -1)
    shock, masks = shock_diagnostics(
        prediction,
        target,
        positions,
        edges,
        node_type,
        proxy_weights,
        component_scale,
        gamma=gamma,
        quantile=shock_quantile,
        phase_alignment=phase_alignment,
    )
    row: dict[str, Any] = {
        "schema": RIPPLE_DIAGNOSTIC_SCHEMA,
        "trajectory": trajectory,
        "source": source,
        "call_index": int(call_index),
        "relative_l2": {
            "equal_node_proxy": weighted_relative_l2_numpy(
                prediction, target, equal_weights, component_scale
            ),
            "reconstructed_weight_proxy": weighted_relative_l2_numpy(
                prediction, target, proxy_weights, component_scale
            ),
        },
        "admissibility": raw_admissibility_summary(prediction, gamma=gamma),
        "shock": shock,
    }
    if not masks:
        return row

    highpass = node_highpass_amplitude(scaled_error, edges)
    proxy_mass = normalized_node_weights(proxy_weights, name="highpass")
    equal_mass = normalized_node_weights(equal_weights, name="highpass")
    smooth = masks["smooth"]
    row["smooth_region"] = {
        "node_count": int(np.count_nonzero(smooth)),
        "highpass_rms_equal_node_proxy": float(
            np.sqrt(
                np.sum(equal_mass[smooth] * highpass[smooth] ** 2)
                / np.sum(equal_mass[smooth])
            )
        ),
        "highpass_rms_reconstructed_weight_proxy": float(
            np.sqrt(
                np.sum(proxy_mass[smooth] * highpass[smooth] ** 2)
                / np.sum(proxy_mass[smooth])
            )
        ),
    }
    geometry = dict(static_geometry_features)
    geometry["shock_graph_distance"] = graph_distance_to_mask(
        positions, edges, masks["target_front"]
    )
    error_amplitude = np.linalg.norm(scaled_error, axis=-1)
    row["spatial_correlations"] = {
        "scaled_error_amplitude_interior": spatial_correlation_summary(
            error_amplitude, geometry, mask=masks["interior"]
        ),
        "highpass_amplitude_smooth": spatial_correlation_summary(
            highpass, geometry, mask=smooth
        ),
    }
    if deep_spectrum:
        spectra: dict[str, Any] = {}
        for weight_name, weights in (
            ("equal_node_proxy", equal_weights),
            ("reconstructed_weight_proxy", proxy_weights),
        ):
            full = graph_spectral_bands(
                scaled_error,
                edges,
                weights,
                lanczos_steps=lanczos_steps,
                radius=full_spectral_radii[weight_name],
                seed=seed + call_index,
            )
            smooth_field, smooth_edges, smooth_weights = induced_subgraph(
                scaled_error,
                edges,
                weights,
                smooth,
            )
            smooth_spectrum = graph_spectral_bands(
                smooth_field,
                smooth_edges,
                smooth_weights,
                lanczos_steps=lanczos_steps,
                seed=seed + 1000 + call_index,
            )
            spectra[weight_name] = {
                "full_graph": full,
                "smooth_induced_subgraph": smooth_spectrum,
            }
        row["graph_spectrum"] = spectra
    return row


def basis_audit(
    *,
    trajectory: str,
    model: PCNOEuler2DResidual,
    current: np.ndarray,
    target: np.ndarray,
    positions: np.ndarray,
    edges: np.ndarray,
    node_type: np.ndarray,
    proxy_weights: np.ndarray,
    output_dir: Path,
    shock_quantile: float,
) -> dict[str, Any]:
    modes = model.backbone.modes.detach().cpu().numpy()
    equal_weights = np.ones(positions.shape[0], dtype=np.float64)
    target_primitive = conservative_to_primitive_raw(target, gamma=model.gamma)
    interior = np.asarray(node_type).reshape(-1) == 0
    if np.count_nonzero(interior) < 3:
        interior = np.ones_like(interior)
    fronts = shock_front_masks(
        target_primitive,
        target_primitive,
        edges,
        quantile=shock_quantile,
        node_mask=interior,
    )
    regions = front_region_masks(
        fronts["prediction_mask"],
        fronts["target_mask"],
        node_mask=interior,
    )
    normalized_residual = (
        target - current
    ) / model.residual_scale.detach().cpu().numpy().reshape(1, -1)
    proxy_mass = normalized_node_weights(proxy_weights, name="basis pressure")
    pressure = target_primitive[:, 3]
    pressure_mean = float(np.sum(proxy_mass * pressure))
    pressure_scale = float(
        np.sqrt(np.sum(proxy_mass * (pressure - pressure_mean) ** 2))
    )
    normalized_pressure = ((pressure - pressure_mean) / max(pressure_scale, 1e-12))[
        :, None
    ]
    field_map = {
        "normalized_conservative_residual": normalized_residual,
        "normalized_target_pressure": normalized_pressure,
    }
    summaries: dict[str, Any] = {}
    coordinate_span = np.ptp(positions, axis=0)
    if np.any(coordinate_span <= 0.0):
        raise ValueError("basis audit requires positive coordinate spans")
    span_modes = compute_Fourier_modes(
        2,
        [model.k_max, model.k_max],
        coordinate_span.tolist(),
    )
    span_counterfactual: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {
        "target_front_mask": regions["target_front"].astype(np.uint8),
        "smooth_mask": regions["smooth"].astype(np.uint8),
    }
    for weight_name, weights in (
        ("equal_node_proxy", equal_weights),
        ("reconstructed_weight_proxy", proxy_weights),
    ):
        gram_summary, gram_arrays = fourier_gram_audit(positions, modes, weights)
        span_gram_summary, span_gram_arrays = fourier_gram_audit(
            positions,
            span_modes,
            weights,
        )
        weight_summary: dict[str, Any] = {"gram": gram_summary, "fields": {}}
        span_weight_summary: dict[str, Any] = {
            "gram": span_gram_summary,
            "fields": {},
        }
        for array_name, array in gram_arrays.items():
            arrays[f"{weight_name}_gram_{array_name}"] = array
        for array_name, array in span_gram_arrays.items():
            arrays[f"{weight_name}_coordinate_span_gram_{array_name}"] = array
        for field_name, field in field_map.items():
            reconstruction_summary, reconstruction_arrays = (
                fourier_reconstruction_audit(
                    positions,
                    modes,
                    weights,
                    field,
                    regions={
                        "target_front": regions["target_front"],
                        "smooth": regions["smooth"],
                    },
                )
            )
            weight_summary["fields"][field_name] = reconstruction_summary
            for array_name, array in reconstruction_arrays.items():
                arrays[f"{weight_name}_{field_name}_{array_name}"] = array.astype(
                    np.float32
                )
            span_reconstruction_summary, span_reconstruction_arrays = (
                fourier_reconstruction_audit(
                    positions,
                    span_modes,
                    weights,
                    field,
                    regions={
                        "target_front": regions["target_front"],
                        "smooth": regions["smooth"],
                    },
                )
            )
            span_weight_summary["fields"][field_name] = span_reconstruction_summary
            for array_name, array in span_reconstruction_arrays.items():
                arrays[f"{weight_name}_{field_name}_coordinate_span_{array_name}"] = (
                    array.astype(np.float32)
                )
        summaries[weight_name] = weight_summary
        span_counterfactual[weight_name] = span_weight_summary
    artifact_name = f"basis_{trajectory}.npz"
    np.savez_compressed(output_dir / artifact_name, **arrays)
    return {
        "trajectory": trajectory,
        "artifact": artifact_name,
        "fields": {
            "normalized_conservative_residual": (
                "(target-current)/training_residual_scale"
            ),
            "normalized_target_pressure": (
                "proxy_weight_centered_and_scaled_target_pressure"
            ),
        },
        "weights": summaries,
        "coordinate_span_basis_counterfactual": {
            "model_domain_lengths": list(model.domain_lengths),
            "coordinate_span": coordinate_span.tolist(),
            "weights": span_counterfactual,
            "claim_boundary": (
                "no-training information-contract counterfactual, not a model "
                "performance ablation"
            ),
        },
        "claim_boundary": (
            "basis stress test only; projection error does not prove the "
            "trained hidden representation uses the same coefficients"
        ),
    }


def _weighted_scaled_norm(
    value: np.ndarray,
    weights: np.ndarray,
    component_scale: np.ndarray,
) -> float:
    field = np.asarray(value, dtype=np.float64)
    mass = normalized_node_weights(weights, name="perturbation norm")
    scale = np.asarray(component_scale, dtype=np.float64).reshape(1, -1)
    return float(np.sqrt(np.sum(mass[:, None] * (field / scale) ** 2)))


@torch.no_grad()
def branch_trace(
    *,
    source: str,
    call_index: int,
    model: PCNOEuler2DResidual,
    sample: Mapping[str, torch.Tensor],
    current: torch.Tensor,
    paired_current: torch.Tensor | None = None,
    target: np.ndarray,
    proxy_weights: np.ndarray,
    include_counterfactuals: bool,
    smooth_reference: np.ndarray | None = None,
    shock_quantile: float = 0.9,
) -> dict[str, Any]:
    model_input = model.normalized_input(
        current,
        nodes=sample["nodes"],
        node_rhos=sample["node_rhos"],
        node_type=sample["node_type"],
        mach=sample["mach"],
    )
    paired_model_input = (
        None
        if paired_current is None
        else model.normalized_input(
            paired_current,
            nodes=sample["nodes"],
            node_rhos=sample["node_rhos"],
            node_type=sample["node_type"],
            mach=sample["mach"],
        )
    )
    aux = (
        sample["node_mask"],
        sample["nodes"],
        sample["node_weights"],
        sample["directed_edges"],
        sample["edge_gradient_weights"],
    )
    smooth_mask = None
    diagnostic_weight_maps = None
    smooth_mask_contract = None
    if smooth_reference is not None:
        reference_tensor = torch.as_tensor(
            np.array(smooth_reference, copy=True),
            dtype=current.dtype,
            device=current.device,
        ).unsqueeze(0)
        interior_mask = sample["node_type"] == 0
        if interior_mask.ndim == 2:
            interior_mask = interior_mask.unsqueeze(-1)
        interior_fallback = not bool(interior_mask.any(dim=1).all())
        if interior_fallback:
            interior_mask = sample["node_mask"].to(dtype=torch.bool)
        smooth_mask = reference_smooth_region_mask(
            reference_tensor,
            sample["directed_edges"],
            sample["node_mask"],
            interior_mask=interior_mask,
            shock_quantile=shock_quantile,
            dilation_hops=2,
            gamma=model.gamma,
        )
        diagnostic_weight_maps = {
            "equal_node_proxy": torch.ones_like(sample["node_mask"]),
            "reconstructed_weight_proxy": torch.as_tensor(
                np.array(proxy_weights, copy=True),
                dtype=current.dtype,
                device=current.device,
            ).reshape(1, -1, 1),
        }
        smooth_mask_contract = {
            "source": "same-call_reference_current_only",
            "shock_quantile": float(shock_quantile),
            "dilation_hops": 2,
            "interior_fallback_to_valid_nodes": interior_fallback,
            "selected_node_count": int(smooth_mask.sum().detach().cpu()),
            "claim_boundary": (
                "diagnostic region only; it does not alter the autonomous model call"
            ),
        }
    normalized_output, summaries = trace_pcno_branches(
        model.backbone,
        model_input,
        aux,
        paired_model_input=paired_model_input,
        reference_smooth_region_mask=smooth_mask,
        diagnostic_weight_maps=diagnostic_weight_maps,
    )
    traced_prediction = (current + normalized_output * model.residual_scale) * sample[
        "node_mask"
    ]
    direct_prediction = model_call(model, sample, current)
    replay_error = float(
        torch.max(torch.abs(traced_prediction - direct_prediction)).cpu()
    )
    prediction = traced_prediction[0].float().cpu().numpy()
    component_scale = model.state_scale.detach().cpu().numpy().reshape(-1)
    record: dict[str, Any] = {
        "source": source,
        "call_index": int(call_index),
        "exact_replay_max_absolute_error": replay_error,
        "layers": summaries,
        "output_relative_l2_reconstructed_weight_proxy": (
            weighted_relative_l2_numpy(
                prediction,
                target,
                proxy_weights,
                component_scale,
            )
        ),
    }
    if smooth_mask_contract is not None:
        record["smooth_highpass_mask"] = smooth_mask_contract
    if include_counterfactuals:
        if paired_current is not None:
            raise ValueError(
                "paired response and leave-one-branch-out are separate diagnostics"
            )
        final_layer = len(model.backbone.ws) - 1
        counterfactuals = {}
        for branch_name in ("spectral", "pointwise", "differential"):
            normalized_counterfactual, _ = trace_pcno_branches(
                model.backbone,
                model_input,
                aux,
                disabled_branch=(final_layer, branch_name),
                collect_summaries=False,
            )
            counterfactual_prediction = (
                (current + normalized_counterfactual * model.residual_scale)[0]
                .float()
                .cpu()
                .numpy()
            )
            counterfactuals[branch_name] = {
                "disabled_layer": final_layer,
                "relative_l2_reconstructed_weight_proxy": (
                    weighted_relative_l2_numpy(
                        counterfactual_prediction,
                        target,
                        proxy_weights,
                        component_scale,
                    )
                ),
                "prediction_change_norm": _weighted_scaled_norm(
                    counterfactual_prediction - prediction,
                    proxy_weights,
                    component_scale,
                ),
                "interpretation": (
                    "untrained leave-one-final-branch-out counterfactual"
                ),
            }
        record["counterfactuals"] = counterfactuals
    return record


@torch.no_grad()
def perturbation_diagnostic(
    *,
    call_index: int,
    model: PCNOEuler2DResidual,
    sample: Mapping[str, torch.Tensor],
    reference_current: torch.Tensor,
    rollout_current: torch.Tensor,
    target: np.ndarray,
    proxy_weights: np.ndarray,
    requested_fraction: float,
) -> tuple[dict[str, Any], torch.Tensor]:
    fraction = float(requested_fraction)
    direction = rollout_current - reference_current
    perturbed = reference_current + fraction * direction
    while fraction >= 1e-4:
        admissibility = raw_admissibility_summary(
            perturbed[0].float().cpu().numpy(),
            gamma=model.gamma,
        )
        if admissibility["all_admissible"]:
            break
        fraction *= 0.5
        perturbed = reference_current + fraction * direction
    else:
        raise RuntimeError(
            "could not construct an admissible rollout-error perturbation"
        )
    reference_prediction = model_call(model, sample, reference_current)
    perturbed_prediction = model_call(model, sample, perturbed)
    component_scale = model.state_scale.detach().cpu().numpy().reshape(-1)
    input_delta = (perturbed - reference_current)[0].float().cpu().numpy()
    output_delta = (
        (perturbed_prediction - reference_prediction)[0].float().cpu().numpy()
    )
    input_norm = _weighted_scaled_norm(input_delta, proxy_weights, component_scale)
    output_norm = _weighted_scaled_norm(output_delta, proxy_weights, component_scale)
    prediction = perturbed_prediction[0].float().cpu().numpy()
    return (
        {
            "source": "perturbed_reference",
            "call_index": int(call_index),
            "direction": "observed_rollout_error_direction",
            "requested_fraction": float(requested_fraction),
            "accepted_fraction": fraction,
            "input_perturbation_norm": input_norm,
            "output_perturbation_norm": output_norm,
            "local_gain": output_norm / max(input_norm, 1e-12),
            "trajectory_target_relative_l2_reconstructed_weight_proxy": (
                weighted_relative_l2_numpy(
                    prediction,
                    target,
                    proxy_weights,
                    component_scale,
                )
            ),
            "claim_boundary": (
                "sensitivity along an observed error direction, not a true PDE "
                "advance from the perturbed state"
            ),
        },
        perturbed,
    )


def _median(values: Sequence[float | None]) -> float | None:
    finite = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    return None if not finite else float(np.median(finite))


def _nonnegative_diagnostic_ratio(
    numerator: float | None,
    denominator: float | None,
) -> float | None:
    """Return a scale-free ratio without imposing an absolute energy floor."""

    if numerator is None or denominator is None:
        return None
    top = float(numerator)
    bottom = float(denominator)
    if not math.isfinite(top) or not math.isfinite(bottom) or top < 0.0 or bottom < 0.0:
        return None
    if bottom == 0.0:
        return 1.0 if top == 0.0 else None
    return top / bottom


def _nested(value: Mapping[str, Any], *keys: str) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


CANCELLATION_MIN_ABSOLUTE_ENERGY = 1e-16
CANCELLATION_MIN_RELATIVE_ENERGY = 1e-8
CANCELLATION_MAX_RELATIVE_IDENTITY_RESIDUAL = 1e-5


def _cancellation_metric_audit(
    metric: Any,
    *,
    reference_energy: float | None,
) -> dict[str, Any]:
    """Validate one cancellation identity before it can enter a screen."""

    if not isinstance(metric, Mapping) or metric.get("status") != "available":
        return {"valid": False, "reason": "metric_unavailable"}
    names = (
        "sum_branch_energy",
        "combined_energy",
        "relative_identity_residual",
        "cancellation_fraction",
    )
    values: dict[str, float] = {}
    for name in names:
        raw = metric.get(name)
        if raw is None:
            return {"valid": False, "reason": f"missing_{name}"}
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return {"valid": False, "reason": f"invalid_{name}"}
        if not math.isfinite(value):
            return {"valid": False, "reason": f"nonfinite_{name}"}
        values[name] = value
    if reference_energy is None or not math.isfinite(float(reference_energy)):
        return {"valid": False, "reason": "invalid_reference_energy"}
    scale = float(reference_energy)
    if scale < CANCELLATION_MIN_ABSOLUTE_ENERGY:
        return {"valid": False, "reason": "reference_energy_too_small"}
    if values["sum_branch_energy"] < 0.0 or values["combined_energy"] < 0.0:
        return {"valid": False, "reason": "negative_energy"}
    relative_energy = values["sum_branch_energy"] / scale
    if (
        values["sum_branch_energy"] < CANCELLATION_MIN_ABSOLUTE_ENERGY
        or relative_energy < CANCELLATION_MIN_RELATIVE_ENERGY
    ):
        return {
            "valid": False,
            "reason": "branch_highpass_energy_too_small",
            "relative_branch_highpass_energy": relative_energy,
        }
    if (
        values["relative_identity_residual"]
        > CANCELLATION_MAX_RELATIVE_IDENTITY_RESIDUAL
    ):
        return {
            "valid": False,
            "reason": "identity_residual_too_large",
            "relative_identity_residual": values["relative_identity_residual"],
        }
    if values["cancellation_fraction"] > 1.0 + 1e-5:
        return {"valid": False, "reason": "invalid_cancellation_fraction"}
    return {
        "valid": True,
        "reason": "available",
        **values,
        "reference_energy": scale,
        "relative_branch_highpass_energy": relative_energy,
    }


def paired_branch_response_selector(
    branch_rows: Sequence[Mapping[str, Any]],
    *,
    required_layer_fraction: float = 0.6,
    minimum_rows: int = 2,
) -> dict[str, Any]:
    """Route a bounded next experiment from repeated paired branch responses."""

    if not 0.0 < required_layer_fraction <= 1.0:
        raise ValueError("required_layer_fraction must lie in (0,1]")
    if minimum_rows < 2:
        raise ValueError("minimum_rows must be at least two")
    paired_rows = [
        row
        for row in branch_rows
        if row.get("source") == "perturbation_branch_response"
    ]
    branch_names = ("spectral", "pointwise", "differential")
    decisions = []
    for row in paired_rows:
        layers = [
            layer for layer in row.get("layers", []) if "paired_response" in layer
        ]
        gain_wins = {name: 0 for name in branch_names}
        roughness_wins = {name: 0 for name in branch_names}
        for layer in layers:
            response = layer["paired_response"]
            gains = response["branch_to_input_rms_gain"]
            roughness = {
                name: response["branches"][name]["edge_to_node_energy_ratio"]
                for name in branch_names
            }
            gain_winner = max(branch_names, key=lambda name: float(gains[name]))
            roughness_winner = max(
                branch_names, key=lambda name: float(roughness[name])
            )
            gain_wins[gain_winner] += 1
            roughness_wins[roughness_winner] += 1
        required_layers = math.ceil(required_layer_fraction * len(layers))
        candidates = [
            name
            for name in branch_names
            if gain_wins[name] >= required_layers
            and roughness_wins[name] >= required_layers
        ]
        decisions.append(
            {
                "trajectory": row.get("trajectory"),
                "call_index": row.get("call_index"),
                "layer_count": len(layers),
                "required_layers": required_layers,
                "gain_wins": gain_wins,
                "roughness_wins": roughness_wins,
                "selected_branch": candidates[0] if len(candidates) == 1 else None,
            }
        )

    complete = [row for row in decisions if row["layer_count"] > 0]
    selected = {row["selected_branch"] for row in complete}
    if len(complete) < minimum_rows:
        route = "insufficient_repeated_rows"
        branch = None
    elif None in selected or len(selected) != 1:
        route = "composite_or_unresolved"
        branch = None
    else:
        branch = next(iter(selected))
        route = {
            "spectral": "spectral_contract_repair",
            "pointwise": "local_pointwise_control",
            "differential": "local_differential_control",
        }[branch]
    return {
        "version": "paired_branch_selector_v1",
        "route": route,
        "selected_branch": branch,
        "required_layer_fraction": required_layer_fraction,
        "minimum_rows": minimum_rows,
        "row_count": len(paired_rows),
        "complete_row_count": len(complete),
        "row_decisions": decisions,
        "claim_boundary": (
            "experiment routing from repeated finite responses; not a causal "
            "branch attribution or an infinitesimal Jacobian estimate"
        ),
    }


def paired_branch_response_summary(
    branch_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate paired finite responses without treating them as Jacobians."""

    paired_rows = [
        row
        for row in branch_rows
        if row.get("source") == "perturbation_branch_response"
    ]
    if not paired_rows:
        return {
            "status": "not_requested",
            "row_count": 0,
            "layer_medians": [],
            "selector": paired_branch_response_selector(branch_rows),
        }
    layer_indices = sorted(
        {
            int(layer["layer"])
            for row in paired_rows
            for layer in row.get("layers", [])
            if "paired_response" in layer
        }
    )
    layer_medians = []
    branch_names = ("spectral", "pointwise", "differential")
    for layer_index in layer_indices:
        responses = []
        for row in paired_rows:
            for layer in row.get("layers", []):
                if int(layer["layer"]) == layer_index and "paired_response" in layer:
                    responses.append(layer["paired_response"])
        gains = {
            name: _median(
                [response["branch_to_input_rms_gain"][name] for response in responses]
            )
            for name in branch_names
        }
        roughness = {
            name: _median(
                [
                    response["branches"][name]["edge_to_node_energy_ratio"]
                    for response in responses
                ]
            )
            for name in branch_names
        }
        layer_medians.append(
            {
                "layer": layer_index,
                "row_count": len(responses),
                "branch_to_input_rms_gain": gains,
                "branch_delta_edge_to_node_energy_ratio": roughness,
                "dominant_rms_response_branch": max(
                    branch_names,
                    key=lambda name: (
                        -math.inf if gains[name] is None else float(gains[name])
                    ),
                ),
                "roughest_response_branch": max(
                    branch_names,
                    key=lambda name: (
                        -math.inf if roughness[name] is None else float(roughness[name])
                    ),
                ),
                "combined_update_to_input_rms_gain": _median(
                    [
                        response["combined_update_to_input_rms_gain"]
                        for response in responses
                    ]
                ),
                "next_hidden_to_input_rms_gain": _median(
                    [
                        response["next_hidden_to_input_rms_gain"]
                        for response in responses
                    ]
                ),
            }
        )
    return {
        "status": "complete",
        "row_count": len(paired_rows),
        "layer_medians": layer_medians,
        "selector": paired_branch_response_selector(branch_rows),
        "direction": "accepted fraction of observed rollout error",
        "claim_boundary": (
            "finite directional response, not an infinitesimal Jacobian or a "
            "causal branch ablation"
        ),
    }


def branch_cancellation_screen(
    branch_rows: Sequence[Mapping[str, Any]],
    *,
    minimum_cases: int = 2,
    minimum_layers: int = 3,
) -> dict[str, Any]:
    """Screen paired ripple responses; retain absolute outputs as context."""

    if minimum_cases != 2 or minimum_layers != 3:
        raise ValueError(
            "the branch-cancellation screen is frozen at two cases/three layers"
        )
    weight_names = ("equal_node_proxy", "reconstructed_weight_proxy")

    paired_rows = [
        row
        for row in branch_rows
        if row.get("source") == "perturbation_branch_response"
    ]
    trajectories = sorted({str(row.get("trajectory")) for row in paired_rows})
    case_decisions = []
    for trajectory in trajectories:
        by_call = {
            int(row["call_index"]): row
            for row in paired_rows
            if str(row.get("trajectory")) == trajectory
        }
        if not by_call:
            case_decisions.append(
                {
                    "trajectory": trajectory,
                    "status": "missing_paired_error_response",
                }
            )
            continue
        call_index = max(by_call)
        row = by_call[call_index]
        weight_decisions: dict[str, Any] = {}
        for weight_name in weight_names:
            layer_audits: dict[int, dict[str, Any]] = {}
            for layer in row.get("layers", []):
                layer_index = int(layer["layer"])
                input_rms = _nested(
                    layer,
                    "paired_response",
                    "input_hidden_delta",
                    "weighted_rms",
                )
                try:
                    reference_energy = float(input_rms) ** 2
                except (TypeError, ValueError):
                    reference_energy = None
                layer_audits[layer_index] = _cancellation_metric_audit(
                    _nested(
                        layer,
                        "paired_response",
                        "smooth_highpass_cancellation",
                        weight_name,
                    ),
                    reference_energy=reference_energy,
                )
            valid = {
                layer_index: audit
                for layer_index, audit in layer_audits.items()
                if audit["valid"]
            }
            cancellations = [
                float(audit["cancellation_fraction"]) for audit in valid.values()
            ]
            weight_decisions[weight_name] = {
                "available_layers": len(valid),
                "valid_layers": sorted(valid),
                "invalid_layer_reasons": {
                    str(layer_index): audit["reason"]
                    for layer_index, audit in layer_audits.items()
                    if not audit["valid"]
                },
                "cancellation_at_least_0.20": sum(
                    value >= 0.20 for value in cancellations
                ),
                "cancellation_at_most_0.05": sum(
                    value <= 0.05 for value in cancellations
                ),
            }
            weight_decisions[weight_name]["support_pass"] = (
                weight_decisions[weight_name]["cancellation_at_least_0.20"]
                >= minimum_layers
            )
            weight_decisions[weight_name]["falsification_pass"] = (
                weight_decisions[weight_name]["cancellation_at_most_0.05"]
                >= minimum_layers
            )
        case_decisions.append(
            {
                "trajectory": trajectory,
                "status": "complete",
                "paired_call": call_index,
                "weights": weight_decisions,
                "support_pass": all(
                    value["support_pass"] for value in weight_decisions.values()
                ),
                "falsification_pass": all(
                    value["falsification_pass"] for value in weight_decisions.values()
                ),
            }
        )

    complete_cases = [row for row in case_decisions if row["status"] == "complete"]
    repeated_cases = len(complete_cases) >= minimum_cases and len(
        complete_cases
    ) == len(case_decisions)
    support = repeated_cases and all(row["support_pass"] for row in complete_cases)
    falsified = repeated_cases and all(
        row["falsification_pass"] for row in complete_cases
    )
    if support:
        classification = "paired_error_response_cancellation_supported"
    elif falsified:
        classification = "paired_error_response_cancellation_falsified"
    else:
        classification = "unresolved"

    sources = ("teacher_forced", "rollout_state")
    absolute_rows = [row for row in branch_rows if row.get("source") in sources]
    absolute_trajectories = sorted(
        {str(row.get("trajectory")) for row in absolute_rows}
    )
    absolute_cases = []
    for trajectory in absolute_trajectories:
        by_source = {
            source: {
                int(row["call_index"]): row
                for row in absolute_rows
                if str(row.get("trajectory")) == trajectory
                and row.get("source") == source
            }
            for source in sources
        }
        common_calls = sorted(set(by_source[sources[0]]) & set(by_source[sources[1]]))
        if not common_calls:
            absolute_cases.append(
                {
                    "trajectory": trajectory,
                    "status": "missing_matched_teacher_rollout_call",
                }
            )
            continue
        late_call = common_calls[-1]
        rows = {source: by_source[source][late_call] for source in sources}
        weight_decisions = {}
        for weight_name in weight_names:
            metrics_by_source: dict[str, dict[int, dict[str, Any]]] = {}
            for source in sources:
                audits: dict[int, dict[str, Any]] = {}
                for layer in rows[source].get("layers", []):
                    branch_stats = layer.get("branches")
                    reference_energy = None
                    if isinstance(branch_stats, Mapping):
                        try:
                            reference_energy = sum(
                                float(branch_stats[name]["weighted_rms"]) ** 2
                                for name in (
                                    "spectral",
                                    "pointwise",
                                    "differential",
                                )
                            )
                        except (KeyError, TypeError, ValueError):
                            reference_energy = None
                    audits[int(layer["layer"])] = _cancellation_metric_audit(
                        _nested(
                            layer,
                            "smooth_highpass_cancellation",
                            weight_name,
                        ),
                        reference_energy=reference_energy,
                    )
                metrics_by_source[source] = {
                    layer_index: audit
                    for layer_index, audit in audits.items()
                    if audit["valid"]
                }
            source_counts = {}
            for source in sources:
                cancellations = [
                    float(metric["cancellation_fraction"])
                    for metric in metrics_by_source[source].values()
                ]
                source_counts[source] = {
                    "available_layers": len(cancellations),
                    "cancellation_at_least_0.20": sum(
                        value >= 0.20 for value in cancellations
                    ),
                    "cancellation_at_most_0.05": sum(
                        value <= 0.05 for value in cancellations
                    ),
                }
            matched_layers = sorted(
                set(metrics_by_source["teacher_forced"])
                & set(metrics_by_source["rollout_state"])
            )
            recurrence_break_layers = []
            for layer_index in matched_layers:
                teacher = metrics_by_source["teacher_forced"][layer_index]
                rollout = metrics_by_source["rollout_state"][layer_index]
                teacher_c = float(teacher["cancellation_fraction"])
                rollout_c = float(rollout["cancellation_fraction"])
                energy_ratio = _nonnegative_diagnostic_ratio(
                    rollout.get("combined_energy"),
                    teacher.get("combined_energy"),
                )
                if (
                    teacher_c >= 0.20
                    and rollout_c - teacher_c <= -0.10
                    and energy_ratio is not None
                    and energy_ratio >= 1.25
                ):
                    recurrence_break_layers.append(layer_index)
            weight_decisions[weight_name] = {
                "source_counts": source_counts,
                "matched_layers": matched_layers,
                "late_distribution_shift_layers": recurrence_break_layers,
                "support_pass": all(
                    source_counts[source]["cancellation_at_least_0.20"]
                    >= minimum_layers
                    for source in sources
                ),
                "falsification_pass": all(
                    source_counts[source]["cancellation_at_most_0.05"] >= minimum_layers
                    for source in sources
                ),
                "late_distribution_shift_pass": (
                    len(recurrence_break_layers) >= minimum_layers
                ),
            }
        absolute_cases.append(
            {
                "trajectory": trajectory,
                "status": "complete",
                "late_common_call": late_call,
                "weights": weight_decisions,
                "support_pass": all(
                    value["support_pass"] for value in weight_decisions.values()
                ),
                "falsification_pass": all(
                    value["falsification_pass"] for value in weight_decisions.values()
                ),
                "late_distribution_shift_pass": all(
                    value["late_distribution_shift_pass"]
                    for value in weight_decisions.values()
                ),
            }
        )
    complete_absolute = [row for row in absolute_cases if row["status"] == "complete"]
    repeated_absolute = len(complete_absolute) >= minimum_cases and len(
        complete_absolute
    ) == len(absolute_cases)
    absolute_support = repeated_absolute and all(
        row["support_pass"] for row in complete_absolute
    )
    absolute_falsified = repeated_absolute and all(
        row["falsification_pass"] for row in complete_absolute
    )
    late_shift = repeated_absolute and all(
        row["late_distribution_shift_pass"] for row in complete_absolute
    )
    if late_shift:
        absolute_classification = "late_teacher_rollout_cancellation_shift"
    elif absolute_support:
        absolute_classification = "absolute_output_cancellation_present"
    elif absolute_falsified:
        absolute_classification = "absolute_output_cancellation_absent"
    else:
        absolute_classification = "unresolved"
    return {
        "version": "smooth_highpass_branch_cancellation_v2",
        "classification": classification,
        "case_count": len(complete_cases),
        "minimum_cases": minimum_cases,
        "minimum_layers": minimum_layers,
        "thresholds": {
            "support_cancellation_fraction": 0.20,
            "falsification_cancellation_fraction": 0.05,
            "minimum_absolute_energy": CANCELLATION_MIN_ABSOLUTE_ENERGY,
            "minimum_relative_branch_highpass_energy": (
                CANCELLATION_MIN_RELATIVE_ENERGY
            ),
            "maximum_relative_identity_residual": (
                CANCELLATION_MAX_RELATIVE_IDENTITY_RESIDUAL
            ),
        },
        "case_decisions": case_decisions,
        "absolute_output_context": {
            "classification": absolute_classification,
            "case_count": len(complete_absolute),
            "thresholds": {
                "late_cancellation_drop": -0.10,
                "late_combined_energy_ratio": 1.25,
            },
            "case_decisions": absolute_cases,
            "claim_boundary": (
                "absolute branch-output cancellation at the latest common call; "
                "a teacher/rollout difference is a noncausal distribution-shift "
                "association, not a recurrence-formation attribution"
            ),
        },
        "claim_boundary": (
            "paired error-direction branch-delta cancellation under equal-node and "
            "reconstructed proxy weights; finite-response threshold screen, not "
            "causal attribution or an infinitesimal Jacobian"
        ),
    }


def mechanism_screen(
    call_rows: Sequence[Mapping[str, Any]],
    basis_rows: Sequence[Mapping[str, Any]],
    branch_rows: Sequence[Mapping[str, Any]],
    perturbation_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply conservative predeclared screens; ambiguity remains unresolved."""

    rollout_rows = [
        row
        for row in call_rows
        if row.get("source") == "rollout_state" and "graph_spectrum" in row
    ]
    teacher_rows = [
        row
        for row in call_rows
        if row.get("source") == "teacher_forced" and "graph_spectrum" in row
    ]
    late_call = max(
        [int(row["call_index"]) for row in rollout_rows],
        default=None,
    )
    first_call = min(
        [int(row["call_index"]) for row in rollout_rows],
        default=None,
    )

    def high_energy(row: Mapping[str, Any]) -> float | None:
        high = _nested(
            row,
            "graph_spectrum",
            "reconstructed_weight_proxy",
            "smooth_induced_subgraph",
            "high_band_energy",
        )
        total = _nested(
            row,
            "graph_spectrum",
            "reconstructed_weight_proxy",
            "smooth_induced_subgraph",
            "total_weighted_energy",
        )
        try:
            high_value = float(high)
            total_value = float(total)
        except (TypeError, ValueError):
            return None
        if (
            not math.isfinite(high_value)
            or not math.isfinite(total_value)
            or high_value < CANCELLATION_MIN_ABSOLUTE_ENERGY
            or total_value <= 0.0
            or high_value / total_value < CANCELLATION_MIN_RELATIVE_ENERGY
        ):
            return None
        return high_value

    late_ratios = []
    growth_ratios = []
    trajectories = sorted({str(row["trajectory"]) for row in rollout_rows})
    for trajectory in trajectories:
        rollout_by_call = {
            int(row["call_index"]): row
            for row in rollout_rows
            if str(row["trajectory"]) == trajectory
        }
        teacher_by_call = {
            int(row["call_index"]): row
            for row in teacher_rows
            if str(row["trajectory"]) == trajectory
        }
        if late_call in rollout_by_call and late_call in teacher_by_call:
            rollout_energy = high_energy(rollout_by_call[late_call])
            teacher_energy = high_energy(teacher_by_call[late_call])
            ratio = _nonnegative_diagnostic_ratio(
                rollout_energy,
                teacher_energy,
            )
            if ratio is not None:
                late_ratios.append(ratio)
        if first_call in rollout_by_call and late_call in rollout_by_call:
            first_energy = high_energy(rollout_by_call[first_call])
            late_energy = high_energy(rollout_by_call[late_call])
            ratio = _nonnegative_diagnostic_ratio(late_energy, first_energy)
            if ratio is not None:
                growth_ratios.append(ratio)

    gram_conditions = []
    gram_off_diagonal = []
    reconstruction_improvements = []
    orthogonal_shock_to_smooth = []
    for row in basis_rows:
        proxy = _nested(row, "weights", "reconstructed_weight_proxy")
        if not isinstance(proxy, Mapping):
            continue
        gram_conditions.append(_nested(proxy, "gram", "correlation_condition_number"))
        gram_off_diagonal.append(_nested(proxy, "gram", "off_diagonal_frobenius_ratio"))
        for field in (
            "normalized_conservative_residual",
            "normalized_target_pressure",
        ):
            field_summary = _nested(proxy, "fields", field)
            if not isinstance(field_summary, Mapping):
                continue
            reconstruction_improvements.append(
                field_summary.get("rmse_improvement_factor")
            )
            orthogonal = _nested(field_summary, "mass_orthogonal_projection", "regions")
            if isinstance(orthogonal, Mapping):
                shock_rmse = _nested(orthogonal, "target_front", "rmse")
                smooth_rmse = _nested(orthogonal, "smooth", "rmse")
                ratio = _nonnegative_diagnostic_ratio(shock_rmse, smooth_rmse)
                if ratio is not None:
                    orthogonal_shock_to_smooth.append(ratio)

    geometry_correlations = []
    phase_reductions = []
    front_energy_fractions = []
    for row in call_rows:
        if row.get("source") != "rollout_state":
            continue
        spatial = _nested(
            row,
            "spatial_correlations",
            "highpass_amplitude_smooth",
        )
        if isinstance(spatial, Mapping):
            for feature_name in (
                "mesh_density_proxy",
                "cell_size_proxy",
                "stencil_condition",
            ):
                correlation = _nested(spatial, feature_name, "spearman")
                if correlation is not None:
                    geometry_correlations.append(abs(float(correlation)))
        phase_reductions.append(
            _nested(
                row,
                "shock",
                "local_shift_alignment",
                "relative_pressure_rmse_reduction",
            )
        )
        front_energy_fractions.append(
            _nested(
                row,
                "shock",
                "scaled_error_energy_fraction_on_front_union",
            )
        )

    spectral_roughness_ratios = []
    for row in branch_rows:
        if row.get("source") != "rollout_state":
            continue
        for layer in row.get("layers", []):
            branches = layer["branches"]
            spectral = branches["spectral"]["edge_to_node_energy_ratio"]
            comparison = max(
                branches["pointwise"]["edge_to_node_energy_ratio"],
                branches["differential"]["edge_to_node_energy_ratio"],
                1e-12,
            )
            spectral_roughness_ratios.append(float(spectral / comparison))

    evidence = {
        "late_rollout_to_teacher_smooth_high_band_energy_ratio": _median(late_ratios),
        "rollout_first_to_late_smooth_high_band_growth": _median(growth_ratios),
        "perturbation_local_gain": _median(
            [row.get("local_gain") for row in perturbation_rows]
        ),
        "gram_condition_number": _median(gram_conditions),
        "gram_off_diagonal_frobenius_ratio": _median(gram_off_diagonal),
        "naive_to_orthogonal_reconstruction_rmse_ratio": _median(
            reconstruction_improvements
        ),
        "orthogonal_projection_shock_to_smooth_rmse_ratio": _median(
            orthogonal_shock_to_smooth
        ),
        "absolute_mesh_correlation": _median(geometry_correlations),
        "spectral_to_other_branch_roughness_ratio": _median(spectral_roughness_ratios),
        "front_local_shift_error_reduction": _median(phase_reductions),
        "front_union_error_energy_fraction": _median(front_energy_fractions),
    }

    def at_least(name: str, threshold: float) -> bool:
        value = evidence[name]
        return value is not None and float(value) >= threshold

    screens = {
        "recurrent_amplification": (
            at_least("late_rollout_to_teacher_smooth_high_band_energy_ratio", 2.0)
            and at_least("rollout_first_to_late_smooth_high_band_growth", 2.0)
            and at_least("perturbation_local_gain", 1.0)
        ),
        "quadrature_or_geometry_conditioning": (
            at_least("naive_to_orthogonal_reconstruction_rmse_ratio", 1.5)
            and (
                at_least("gram_condition_number", 100.0)
                or at_least("gram_off_diagonal_frobenius_ratio", 0.2)
            )
            and at_least("absolute_mesh_correlation", 0.3)
        ),
        "spectral_truncation_or_aliasing": (
            at_least("orthogonal_projection_shock_to_smooth_rmse_ratio", 2.0)
            and at_least("spectral_to_other_branch_roughness_ratio", 1.5)
        ),
        "front_phase_error": (
            at_least("front_local_shift_error_reduction", 0.3)
            and at_least("front_union_error_energy_fraction", 0.5)
        ),
    }
    supported = [name for name, passed in screens.items() if passed]
    if len(supported) == 1:
        classification = supported[0]
    elif supported:
        classification = "mixed_or_unresolved"
    else:
        classification = "unresolved"
    routing = {
        "spectral_truncation_or_aliasing": (
            "internal spectral shaping, de-aliasing, or mass orthogonalization"
        ),
        "quadrature_or_geometry_conditioning": (
            "repair quadrature, geometry scaling, or differential conditioning"
        ),
        "recurrent_amplification": (
            "short generated-state exposure after the one-step gate"
        ),
        "front_phase_error": (
            "front-factorized prediction with conservative remapping only when "
            "the evaluated artifact has a validated finite-volume geometry "
            "contract; otherwise restrict this to a representation diagnostic"
        ),
    }
    return {
        "classification": classification,
        "supported_screens": supported,
        "evidence": evidence,
        "predeclared_screens": screens,
        "selected_first_branch": routing.get(classification),
        "falsification": {
            "spectral_truncation_or_aliasing": (
                "falsified if the first high-band growth is absent from the "
                "spectral branch and appears only after local branches or "
                "autoregressive feedback"
            ),
            "quadrature_or_geometry_conditioning": (
                "falsified if mesh correlations remain weak under both proxy "
                "measures and an orthogonalized model replay does not reduce "
                "the first-step ripple; projection stress alone is insufficient"
            ),
            "recurrent_amplification": (
                "falsified if teacher-forced and rollout high-band energy are "
                "comparable, do not grow with call index, and perturbation "
                "gain stays below one"
            ),
            "front_phase_error": (
                "falsified if small spatial alignment removes little shock "
                "error while strength or thickness error remains"
            ),
        },
        "branch_cancellation": branch_cancellation_screen(branch_rows),
        "claim_boundary": (
            "threshold screen, not a calibrated statistical test; ambiguous "
            "or multiple passes remain unresolved"
        ),
    }


def _state_tensor(state: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(
        np.array(state, copy=True),
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    validate_args(args)
    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")
    device = select_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    checkpoint = load_checkpoint(args.checkpoint)
    store = PCNOEuler2DShardStore(args.data_dir)
    weight_provenance = str(
        store.manifest.get("weight_provenance", "missing_weight_provenance")
    )
    physical_volume_weights = (
        weight_provenance == "validated_physical_cell_volume_normalized"
    )
    if checkpoint["data_manifest_digest"] != store.manifest_digest:
        raise ValueError("checkpoint and shard manifest digests differ")
    if checkpoint["boundary_mode"] != "model_all_nodes":
        raise ValueError("D013 requires the legal model-all-node boundary contract")
    if not bool(checkpoint["raw_recurrence"]):
        raise ValueError("D013 requires raw recurrence")
    model = build_model(checkpoint, device)
    keys, key_source = select_trajectory_keys(args, checkpoint, store)
    args.output_dir.mkdir(parents=True)
    deep_keys = set(keys[: args.deep_trajectory_count])
    step_stride = int(checkpoint["step_stride"])
    dt = float(store.manifest["dt"]) * step_stride
    component_scale = model.state_scale.detach().cpu().numpy().reshape(-1)
    checkpoint_digest = sha256_file(args.checkpoint)

    call_rows: list[dict[str, Any]] = []
    basis_rows: list[dict[str, Any]] = []
    branch_rows: list[dict[str, Any]] = []
    perturbation_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    calls_path = args.output_dir / "calls.jsonl"
    branches_path = args.output_dir / "branches.jsonl"
    for trajectory_index, key in enumerate(keys):
        states = store.states(key)
        final_target_index = args.start_frame + args.num_steps * step_stride
        if final_target_index >= states.shape[0]:
            raise ValueError(
                f"trajectory {key} has only {states.shape[0]} frames for "
                f"requested target index {final_target_index}"
            )
        positions = np.array(store.array(key, "nodes"), copy=True)
        edges = np.array(store.array(key, "edges"), copy=True)
        directed_edges = np.array(store.array(key, "directed_edges"), copy=True)
        node_type = np.array(store.array(key, "node_type"), copy=True).reshape(-1)
        node_measures = np.array(store.array(key, "node_measures"), copy=True)
        proxy_weights = np.array(store.array(key, "node_weights"), copy=True).sum(
            axis=-1
        )
        equal_weights = np.ones(positions.shape[0], dtype=np.float64)
        sample = store.tensor_sample(
            key,
            args.start_frame,
            step_stride=step_stride,
            device=device,
        )
        static_geometry_features = geometry_conditioning_features(
            positions,
            edges,
            directed_edges,
            node_type,
            node_measures,
        )
        full_spectral_radii: dict[str, float] = {}
        if key in deep_keys:
            full_spectral_radii = {
                "equal_node_proxy": estimate_generalized_laplacian_radius(
                    edges,
                    equal_weights,
                    seed=args.seed + trajectory_index,
                ),
                "reconstructed_weight_proxy": (
                    estimate_generalized_laplacian_radius(
                        edges,
                        proxy_weights,
                        seed=args.seed + 100 + trajectory_index,
                    )
                ),
            }

        basis_current_index = args.start_frame + (args.basis_call - 1) * step_stride
        basis_target_index = basis_current_index + step_stride
        basis_record = basis_audit(
            trajectory=key,
            model=model,
            current=np.array(states[basis_current_index], copy=True),
            target=np.array(states[basis_target_index], copy=True),
            positions=positions,
            edges=edges,
            node_type=node_type,
            proxy_weights=proxy_weights,
            output_dir=args.output_dir,
            shock_quantile=args.shock_quantile,
        )
        basis_record["call_index"] = args.basis_call
        basis_rows.append(basis_record)

        current = _state_tensor(states[args.start_frame], device)
        rollout_currents: list[np.ndarray] = []
        reference_currents: list[np.ndarray] = []
        predictions: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        failure_cause = "completed"
        failed_proposal: np.ndarray | None = None
        failure_call: int | None = None
        for call_index in range(1, args.num_steps + 1):
            reference_current_index = args.start_frame + (call_index - 1) * step_stride
            target_index = reference_current_index + step_stride
            reference_current_np = np.array(states[reference_current_index], copy=True)
            target_np = np.array(states[target_index], copy=True)
            rollout_current_np = current[0].float().cpu().numpy()
            rollout_currents.append(rollout_current_np.copy())
            reference_currents.append(reference_current_np)
            with torch.no_grad():
                proposal = model_call(model, sample, current)
            proposal_np = proposal[0].float().cpu().numpy()
            admissibility = raw_admissibility_summary(proposal_np, gamma=model.gamma)
            if not admissibility["all_finite"]:
                failure_cause = "nonfinite_state"
            elif not admissibility["all_admissible"]:
                failure_cause = "inadmissible_state"
            if failure_cause != "completed":
                failed_proposal = proposal_np
                failure_call = call_index
                break
            predictions.append(proposal_np)
            targets.append(target_np)

            if call_index in args.diagnostic_calls:
                phase_alignment = call_index == args.num_steps
                rollout_row = call_diagnostics(
                    trajectory=key,
                    source="rollout_state",
                    call_index=call_index,
                    prediction=proposal_np,
                    target=target_np,
                    positions=positions,
                    edges=edges,
                    directed_edges=directed_edges,
                    node_type=node_type,
                    node_measures=node_measures,
                    proxy_weights=proxy_weights,
                    component_scale=component_scale,
                    gamma=model.gamma,
                    shock_quantile=args.shock_quantile,
                    lanczos_steps=args.lanczos_steps,
                    deep_spectrum=key in deep_keys,
                    full_spectral_radii=full_spectral_radii,
                    static_geometry_features=static_geometry_features,
                    phase_alignment=phase_alignment,
                    seed=args.seed + trajectory_index * 100,
                )
                call_rows.append(rollout_row)
                append_jsonl(calls_path, rollout_row)

                reference_current = _state_tensor(reference_current_np, device)
                with torch.no_grad():
                    teacher_prediction = model_call(model, sample, reference_current)
                teacher_prediction_np = teacher_prediction[0].float().cpu().numpy()
                teacher_row = call_diagnostics(
                    trajectory=key,
                    source="teacher_forced",
                    call_index=call_index,
                    prediction=teacher_prediction_np,
                    target=target_np,
                    positions=positions,
                    edges=edges,
                    directed_edges=directed_edges,
                    node_type=node_type,
                    node_measures=node_measures,
                    proxy_weights=proxy_weights,
                    component_scale=component_scale,
                    gamma=model.gamma,
                    shock_quantile=args.shock_quantile,
                    lanczos_steps=args.lanczos_steps,
                    deep_spectrum=key in deep_keys,
                    full_spectral_radii=full_spectral_radii,
                    static_geometry_features=static_geometry_features,
                    phase_alignment=False,
                    seed=args.seed + trajectory_index * 100 + 50,
                )
                call_rows.append(teacher_row)
                append_jsonl(calls_path, teacher_row)

            if key in deep_keys and call_index in args.trace_calls:
                teacher_current = _state_tensor(reference_current_np, device)
                for source, trace_current in (
                    ("teacher_forced", teacher_current),
                    ("rollout_state", current),
                ):
                    trace = branch_trace(
                        source=source,
                        call_index=call_index,
                        model=model,
                        sample=sample,
                        current=trace_current,
                        target=target_np,
                        proxy_weights=proxy_weights,
                        smooth_reference=reference_current_np,
                        shock_quantile=args.shock_quantile,
                        include_counterfactuals=(
                            source == "rollout_state"
                            and call_index == max(args.trace_calls)
                        ),
                    )
                    trace["trajectory"] = key
                    branch_rows.append(trace)
                    append_jsonl(branches_path, trace)
            current = proposal

        valid_length = len(predictions)
        if failure_call is not None and key in deep_keys:
            failed_input = _state_tensor(rollout_currents[failure_call - 1], device)
            failed_target_index = args.start_frame + failure_call * step_stride
            pre_failure_trace = branch_trace(
                source="pre_failure",
                call_index=failure_call,
                model=model,
                sample=sample,
                current=failed_input,
                target=np.array(states[failed_target_index], copy=True),
                proxy_weights=proxy_weights,
                smooth_reference=np.array(
                    states[args.start_frame + (failure_call - 1) * step_stride],
                    copy=True,
                ),
                shock_quantile=args.shock_quantile,
                include_counterfactuals=False,
            )
            pre_failure_trace["trajectory"] = key
            branch_rows.append(pre_failure_trace)
            append_jsonl(branches_path, pre_failure_trace)
        if valid_length >= args.perturbation_call and key in deep_keys:
            perturb_call = args.perturbation_call
            reference_np = reference_currents[perturb_call - 1]
            rollout_np = rollout_currents[perturb_call - 1]
            target_np = targets[perturb_call - 1]
            reference_tensor = _state_tensor(reference_np, device)
            rollout_tensor = _state_tensor(rollout_np, device)
            perturbation, perturbed_tensor = perturbation_diagnostic(
                call_index=perturb_call,
                model=model,
                sample=sample,
                reference_current=reference_tensor,
                rollout_current=rollout_tensor,
                target=target_np,
                proxy_weights=proxy_weights,
                requested_fraction=args.perturbation_fraction,
            )
            perturbation["trajectory"] = key
            perturbation_rows.append(perturbation)
            with torch.no_grad():
                perturbed_prediction = (
                    model_call(model, sample, perturbed_tensor)[0].float().cpu().numpy()
                )
            perturbed_row = call_diagnostics(
                trajectory=key,
                source="perturbed_reference",
                call_index=perturb_call,
                prediction=perturbed_prediction,
                target=target_np,
                positions=positions,
                edges=edges,
                directed_edges=directed_edges,
                node_type=node_type,
                node_measures=node_measures,
                proxy_weights=proxy_weights,
                component_scale=component_scale,
                gamma=model.gamma,
                shock_quantile=args.shock_quantile,
                lanczos_steps=args.lanczos_steps,
                deep_spectrum=False,
                full_spectral_radii=full_spectral_radii,
                static_geometry_features=static_geometry_features,
                phase_alignment=False,
                seed=args.seed + trajectory_index * 100 + 75,
            )
            perturbed_row["perturbation"] = perturbation
            call_rows.append(perturbed_row)
            append_jsonl(calls_path, perturbed_row)
            trace = branch_trace(
                source="perturbation_branch_response",
                call_index=perturb_call,
                model=model,
                sample=sample,
                current=reference_tensor,
                paired_current=perturbed_tensor,
                target=target_np,
                proxy_weights=proxy_weights,
                smooth_reference=reference_np,
                shock_quantile=args.shock_quantile,
                include_counterfactuals=False,
            )
            trace["trajectory"] = key
            branch_rows.append(trace)
            append_jsonl(branches_path, trace)

        artifact_name = f"trajectory_{key}.npz"
        artifact_arrays: dict[str, Any] = {
            "schema": np.asarray(RIPPLE_DIAGNOSTIC_SCHEMA),
            "trajectory_key": np.asarray(key),
            "positions": positions.astype(np.float32),
            "edges": edges.astype(np.int64),
            "node_type": node_type.astype(np.int64),
            "node_measures_proxy": node_measures.astype(np.float32),
            "reconstructed_node_weights_proxy": proxy_weights.astype(np.float32),
            "equal_node_weights_proxy": np.ones(positions.shape[0], dtype=np.float32),
            "rollout_currents": np.asarray(rollout_currents, dtype=np.float32),
            "reference_currents": np.asarray(reference_currents, dtype=np.float32),
            "predictions": np.asarray(predictions, dtype=np.float32),
            "targets": np.asarray(targets, dtype=np.float32),
            "valid_length": np.asarray(valid_length, dtype=np.int64),
            "failure_cause": np.asarray(failure_cause),
            "failure_call": np.asarray(
                -1 if failure_call is None else failure_call,
                dtype=np.int64,
            ),
            "start_frame": np.asarray(args.start_frame, dtype=np.int64),
            "step_stride": np.asarray(step_stride, dtype=np.int64),
            "delta_t": np.asarray(dt, dtype=np.float64),
            "physical_target_times": (
                np.arange(1, valid_length + 1, dtype=np.float64) * dt
                + args.start_frame * float(store.manifest["dt"])
            ),
            "mach": np.asarray(store.entry(key)["mach"], dtype=np.float64),
            "boundary_mode": np.asarray("model_all_nodes"),
            "coordinate_convention": np.asarray(
                store.manifest["coordinate_convention"]
            ),
            "state_convention": np.asarray(store.manifest["state_convention"]),
            "checkpoint_sha256": np.asarray(checkpoint_digest),
            "config_digest": np.asarray(str(checkpoint["config_digest"])),
            "weight_provenance": np.asarray(weight_provenance),
        }
        if physical_volume_weights:
            artifact_arrays["physical_cell_volume_weights"] = proxy_weights.astype(
                np.float32
            )
        if failed_proposal is not None:
            artifact_arrays["failed_proposal"] = failed_proposal.astype(np.float32)
        np.savez_compressed(args.output_dir / artifact_name, **artifact_arrays)
        trajectory_record = {
            "trajectory": key,
            "artifact": artifact_name,
            "requested_steps": args.num_steps,
            "valid_length": valid_length,
            "completed": valid_length == args.num_steps,
            "failure_cause": failure_cause,
            "failure_call": failure_call,
            "deep_diagnostics": key in deep_keys,
            "pre_failure_contract": (
                "last valid requested trace is preterminal because no failure occurred"
                if failure_cause == "completed"
                else "input immediately before failed proposal saved"
            ),
        }
        trajectory_rows.append(trajectory_record)
        print(
            json.dumps(
                {
                    "trajectory": key,
                    "valid_length": valid_length,
                    "failure_cause": failure_cause,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    screen = mechanism_screen(
        call_rows,
        basis_rows,
        branch_rows,
        perturbation_rows,
    )
    summary = {
        "schema": RIPPLE_DIAGNOSTIC_SCHEMA,
        "status": "complete",
        "checkpoint": {
            "sha256": checkpoint_digest,
            "epoch": checkpoint.get("epoch"),
            "best_epoch": checkpoint.get("best_epoch"),
            "config_digest": checkpoint["config_digest"],
            "model_config": checkpoint["model_config"],
            "data_manifest_digest": checkpoint["data_manifest_digest"],
            "boundary_mode": checkpoint["boundary_mode"],
            "raw_recurrence": checkpoint["raw_recurrence"],
        },
        "evaluation": {
            "trajectory_keys": keys,
            "trajectory_key_source": key_source,
            "deep_trajectory_keys": sorted(deep_keys),
            "start_frame": args.start_frame,
            "num_steps": args.num_steps,
            "step_stride": step_stride,
            "physical_delta_t": dt,
            "diagnostic_calls": args.diagnostic_calls,
            "trace_calls": args.trace_calls,
            "basis_call": args.basis_call,
            "perturbation_call": args.perturbation_call,
            "device": str(device),
            "diagnostic_precision": "float32",
        },
        "trajectories": trajectory_rows,
        "completion_rate": float(
            np.mean([row["completed"] for row in trajectory_rows])
        ),
        "basis_audits": basis_rows,
        "paired_branch_response": paired_branch_response_summary(branch_rows),
        "mechanism_screen": screen,
        "line2_d014_interface": line2_interface(args.line2_d014_summary),
        "architecture_reach": {
            "full_pcno": ("global spatial dependency through every spectral branch"),
            "finite_hop_scope": (
                "applies only to local differential/graph operations, not the "
                "full PCNO architecture"
            ),
            "paired_state_use_tests": (
                "teacher, rollout, and error-direction perturbation calls test "
                "whether the full architecture responds to nonlocal state"
            ),
        },
        "diagnostic_contract": {
            "primary_spectrum": "generalized graph Laplacian Lanczos bands",
            "fft": "not used",
            "weights": {
                "reconstructed_weight_proxy": (
                    "legacy diagnostic key containing normalized validated physical "
                    "cell-volume weights"
                    if physical_volume_weights
                    else "reconstructed vertex-lumped PCNO quadrature proxy"
                ),
                "equal_node_proxy": "equal-node diagnostic proxy",
                "manifest_weight_provenance": weight_provenance,
            },
            "physical_conservation": (
                "not evaluated inside D013; use the separate physical baseline "
                "evaluator with oriented faces and boundary exchange"
                if physical_volume_weights
                else "not evaluated; geometry contract invalid"
            ),
            "boundary": "model_all_nodes with no future-reference replacement",
            "recurrence": "raw, no clipping, floors, smoothing, or limiter",
            "branch_outputs": (
                "live summaries only; full prediction/target/current arrays saved"
            ),
            "branch_cancellation": (
                "paired error-direction branch-delta graph-high-pass cancellation is "
                "primary; absolute teacher/rollout outputs are noncausal context; both "
                "use equal-node and the manifest-declared primary weights"
            ),
        },
        "artifact_schema": {
            "trajectory_npz": (
                "prediction, target, rollout/reference current, positions, "
                "edges, node types, declared primary/equal weights, time, trajectory/Mach, "
                "boundary, failure, coordinate/state convention, digests"
            ),
            "calls_jsonl": "per-call shock, spectrum, geometry, admissibility",
            "branches_jsonl": (
                "branchwise, paired-delta, cancellation, and counterfactual summaries"
            ),
            "basis_npz": "Gram matrices and both basis reconstructions",
        },
        "claim_boundary": {
            "verified": (
                "frozen checkpoint behavior and exact discrete transforms "
                "recorded by this bundle"
            ),
            "plausible_mechanisms": (
                "mechanism screen remains a falsifiable diagnostic classification"
            ),
            "unsupported": (
                "physical conservation or flux from this D013 bundle alone, paper-faithful CPGNet, "
                "or a universal explanation of neural-operator ripples"
            ),
        },
    }
    write_json(args.output_dir / "summary.json", summary)
    store.close()
    print(json.dumps(json_safe(screen), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
