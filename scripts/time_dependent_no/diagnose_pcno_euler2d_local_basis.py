"""Compare one fixed local basis with saved full-resolution D013 projections.

This entry point consumes an existing D013 artifact bundle. It performs no
model loading, training, rollout, or GPU work, and all node measures remain
explicit diagnostic proxies rather than physical control-volume measures.
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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utility.time_dependent_no.euler2d_metrics import (  # noqa: E402
    front_centroid_distance,
    median_edge_length,
    shock_front_masks,
    shock_smearing_metrics,
)
from utility.time_dependent_no.pcno_ripple_diagnostics import (  # noqa: E402
    LOCAL_BASIS_CENTER_COUNT,
    LOCAL_BASIS_RADIUS_FACTOR,
    RIPPLE_DIAGNOSTIC_SCHEMA,
    conservative_to_primitive_raw,
    deterministic_farthest_point_centers,
    induced_subgraph,
    node_highpass_amplitude,
    normalized_node_weights,
    raw_admissibility_summary,
    weighted_basis_projection_audit,
    wendland_c2_partition_of_unity_basis,
)

LOCAL_BASIS_SCHEMA = "pcno_euler2d_local_basis_d042_v2"
EXPECTED_TRAJECTORY_COUNT = 5
MIN_FULL_RESOLUTION_NODES = 19_000
WEIGHT_NAMES = ("equal_node_proxy", "reconstructed_weight_proxy")
RESIDUAL_FIELD = "normalized_conservative_residual"
PRESSURE_FIELD = "normalized_target_pressure"
FIELD_NAMES = (RESIDUAL_FIELD, PRESSURE_FIELD)
CONTROL_NAMES = (
    "current_fourier_mass_orthogonal",
    "coordinate_span_fourier_mass_orthogonal",
)
FROZEN_GATES = {
    "center_count": LOCAL_BASIS_CENTER_COUNT,
    "radius_rule": "R=2h",
    "locality_column_support_fraction_median_max": 0.10,
    "locality_column_support_fraction_p90_max": 0.20,
    "partition_of_unity_max_row_error": 1e-10,
    "effective_rank_min": math.ceil(0.95 * LOCAL_BASIS_CENTER_COUNT),
    "residual_local_to_current_median_max_both_proxies": 0.80,
    "residual_local_no_worse_case_count_min_both_proxies": 4,
    "residual_reconstructed_local_to_span_median_max": 0.90,
    "smooth_residual_highpass_local_to_current_median_max_both_proxies": 0.70,
    "smooth_residual_highpass_local_no_worse_case_count_min_both_proxies": 4,
    "decoded_pressure_centroid_distance_median_edge_max": 1.0,
    "decoded_pressure_thickness_ratio_range": [0.95, 1.05],
    "decoded_pressure_strength_ratio_range": [0.95, 1.05],
    "decoded_pressure_overshoot_fraction_max": 0.05,
    "decoded_pressure_undershoot_fraction_max": 0.05,
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_scalar(value: np.ndarray, *, name: str) -> Any:
    array = np.asarray(value)
    if array.shape != ():
        raise ValueError(f"{name} must be a scalar")
    return array.item()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _required_array(
    archive: Mapping[str, np.ndarray],
    name: str,
    *,
    ndim: int | None = None,
) -> np.ndarray:
    if name not in archive:
        raise ValueError(f"artifact is missing required array {name}")
    value = np.asarray(archive[name])
    if ndim is not None and value.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    return value


def _load_source_contract(input_dir: Path) -> tuple[dict[str, Any], list[str]]:
    summary = _load_json(input_dir / "summary.json")
    if summary.get("schema") != RIPPLE_DIAGNOSTIC_SCHEMA:
        raise ValueError("input is not a D013 ripple diagnostic bundle")
    if summary.get("status") != "complete":
        raise ValueError("input D013 bundle is not complete")
    checkpoint = summary.get("checkpoint")
    evaluation = summary.get("evaluation")
    if not isinstance(checkpoint, Mapping) or not isinstance(evaluation, Mapping):
        raise ValueError("input summary lacks checkpoint/evaluation contracts")
    if checkpoint.get("boundary_mode") != "model_all_nodes":
        raise ValueError("input checkpoint does not use model_all_nodes boundaries")
    if checkpoint.get("raw_recurrence") is not True:
        raise ValueError("input checkpoint does not declare raw recurrence")
    for name in ("sha256", "config_digest", "data_manifest_digest"):
        if not isinstance(checkpoint.get(name), str) or not checkpoint[name]:
            raise ValueError(f"input checkpoint lacks {name}")
    keys = evaluation.get("trajectory_keys")
    if (
        not isinstance(keys, list)
        or len(keys) != EXPECTED_TRAJECTORY_COUNT
        or len({str(key) for key in keys}) != EXPECTED_TRAJECTORY_COUNT
    ):
        raise ValueError("local basis audit requires exactly five unique trajectories")
    keys = [str(key) for key in keys]
    basis_call = int(evaluation.get("basis_call", -1))
    if basis_call < 1:
        raise ValueError("input summary has an invalid basis call")

    trajectory_rows = summary.get("trajectories")
    basis_rows = summary.get("basis_audits")
    if not isinstance(trajectory_rows, list) or not isinstance(basis_rows, list):
        raise ValueError("input summary lacks trajectory or basis rows")
    trajectory_map = {str(row.get("trajectory")): row for row in trajectory_rows}
    basis_map = {str(row.get("trajectory")): row for row in basis_rows}
    if set(trajectory_map) != set(keys) or set(basis_map) != set(keys):
        raise ValueError("summary trajectory/basis rows do not match evaluation keys")
    for key in keys:
        trajectory_row = trajectory_map[key]
        basis_row = basis_map[key]
        if trajectory_row.get("artifact") != f"trajectory_{key}.npz":
            raise ValueError(f"trajectory artifact name mismatch for {key}")
        if basis_row.get("artifact") != f"basis_{key}.npz":
            raise ValueError(f"basis artifact name mismatch for {key}")
        if int(basis_row.get("call_index", -1)) != basis_call:
            raise ValueError(f"basis call mismatch for {key}")
    return summary, keys


def _shock_separated_smooth_mask(
    front_mask: np.ndarray,
    interior_mask: np.ndarray,
    edges: np.ndarray,
    *,
    dilation_hops: int = 2,
) -> np.ndarray:
    """Exclude the reference front and its graph neighborhood from smooth nodes."""

    front = np.asarray(front_mask, dtype=bool)
    interior = np.asarray(interior_mask, dtype=bool)
    edge_index = np.asarray(edges, dtype=np.int64)
    if front.shape != interior.shape or front.ndim != 1:
        raise ValueError("front and interior masks must be aligned vectors")
    if dilation_hops < 0:
        raise ValueError("dilation_hops must be nonnegative")
    if edge_index.ndim != 2 or edge_index.shape[1] != 2:
        raise ValueError("edges must have shape [E,2]")
    if edge_index.size and (edge_index.min() < 0 or edge_index.max() >= front.size):
        raise ValueError("edge index lies outside the masks")
    if np.any(front & ~interior):
        raise ValueError("reference front must remain inside the interior domain")

    excluded = front.copy()
    left = edge_index[:, 0]
    right = edge_index[:, 1]
    for _ in range(dilation_hops):
        expanded = excluded.copy()
        np.logical_or.at(expanded, left, excluded[right])
        np.logical_or.at(expanded, right, excluded[left])
        excluded = expanded
    smooth = interior & ~excluded
    if np.count_nonzero(smooth) < 2:
        raise ValueError("shock-separated smooth region contains fewer than two nodes")
    return smooth


def _weighted_reconstruction_metrics(
    reconstruction: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    *,
    smooth_mask: np.ndarray,
    front_mask: np.ndarray,
    edges: np.ndarray,
) -> dict[str, Any]:
    prediction = np.asarray(reconstruction, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    if prediction.shape != truth.shape or prediction.ndim != 2:
        raise ValueError("projection and target must have shape [N,C]")
    if not np.isfinite(prediction).all() or not np.isfinite(truth).all():
        raise ValueError("projection and target must be finite")
    mass = normalized_node_weights(weights, name="local basis comparison")

    def region(mask: np.ndarray) -> dict[str, float]:
        selected = np.asarray(mask, dtype=bool)
        if selected.shape != (truth.shape[0],) or not np.any(selected):
            raise ValueError("comparison region must select at least one node")
        local_mass = mass[selected]
        local_mass = local_mass / local_mass.sum()
        error = prediction[selected] - truth[selected]
        target_energy = float(np.sum(local_mass[:, None] * truth[selected] ** 2))
        error_energy = float(np.sum(local_mass[:, None] * error**2))
        return {
            "rmse": float(np.sqrt(error_energy)),
            "relative_l2": float(np.sqrt(error_energy / max(target_energy, 1e-12))),
            "max_absolute_error": float(np.max(np.abs(error))),
        }

    error = prediction - truth
    smooth = np.asarray(smooth_mask, dtype=bool)
    smooth_error, smooth_edges, smooth_mass = induced_subgraph(
        error,
        edges,
        mass,
        smooth,
    )
    highpass = node_highpass_amplitude(smooth_error, smooth_edges)
    smooth_highpass_rms = float(
        np.sqrt(
            np.sum(smooth_mass * highpass**2) / max(float(smooth_mass.sum()), 1e-12)
        )
    )
    target_min = np.min(truth, axis=0)
    target_max = np.max(truth, axis=0)
    target_range = np.maximum(target_max - target_min, 1e-12)
    return {
        **region(np.ones(truth.shape[0], dtype=bool)),
        "regions": {
            "target_front": region(front_mask),
            "smooth": region(smooth),
        },
        "smooth_error_highpass_rms": smooth_highpass_rms,
        "smooth_highpass_contract": (
            "induced subgraph after two-hop reference-front dilation"
        ),
        "overshoot_fraction_by_component": (
            np.maximum(np.max(prediction, axis=0) - target_max, 0.0) / target_range
        ).tolist(),
        "undershoot_fraction_by_component": (
            np.maximum(target_min - np.min(prediction, axis=0), 0.0) / target_range
        ).tolist(),
    }


def _finite_scalar(value: Any) -> float | None:
    array = np.asarray(value)
    if array.size != 1:
        return None
    number = float(array.reshape(-1)[0])
    return number if math.isfinite(number) else None


def _pressure_structure_metrics(
    reconstruction: np.ndarray,
    target: np.ndarray,
    *,
    positions: np.ndarray,
    edges: np.ndarray,
    interior: np.ndarray,
    saved_target_front: np.ndarray,
) -> dict[str, Any]:
    prediction = np.asarray(reconstruction, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    if prediction.shape != truth.shape or prediction.shape[1] != 1:
        return {"status": "unavailable_invalid_pressure_shape"}
    try:
        prediction_primitive = np.zeros((prediction.shape[0], 4), dtype=np.float64)
        target_primitive = np.zeros((truth.shape[0], 4), dtype=np.float64)
        prediction_primitive[:, 3] = prediction[:, 0]
        target_primitive[:, 3] = truth[:, 0]
        fronts = shock_front_masks(
            prediction_primitive,
            target_primitive,
            edges,
            scalar_index=3,
            quantile=0.9,
            node_mask=interior,
        )
        if not np.array_equal(fronts["target_mask"], saved_target_front):
            raise ValueError(
                "saved target-front mask does not match the fixed q0.9 reconstruction"
            )
        centroid = _finite_scalar(
            front_centroid_distance(
                fronts["prediction_mask"],
                fronts["target_mask"],
                positions,
            )
        )
        pred_interior, interior_edges, _ = induced_subgraph(
            prediction_primitive,
            edges,
            np.ones(prediction.shape[0]),
            interior,
        )
        target_interior, _, _ = induced_subgraph(
            target_primitive,
            edges,
            np.ones(truth.shape[0]),
            interior,
        )
        smearing = shock_smearing_metrics(
            pred_interior,
            target_interior,
            interior_edges,
            scalar_index=3,
        )
        thickness = _finite_scalar(smearing["thickness_ratio"])
        strength = _finite_scalar(smearing["strength_ratio"])
        edge_length = _finite_scalar(median_edge_length(positions, edges))
        if None in (centroid, thickness, strength, edge_length):
            return {"status": "unavailable_nonfinite_structure_metric"}
        return {
            "status": "available",
            "front_quantile": 0.9,
            "front_centroid_distance": centroid,
            "median_edge_length": edge_length,
            "front_centroid_distance_in_median_edges": centroid / edge_length,
            "thickness_ratio": thickness,
            "strength_ratio": strength,
        }
    except (ValueError, FloatingPointError) as error:
        return {"status": "unavailable", "reason": str(error)}


def _decoded_residual_state_metrics(
    normalized_residual_reconstruction: np.ndarray,
    reference_current: np.ndarray,
    physical_target: np.ndarray,
    residual_scale: np.ndarray,
    *,
    positions: np.ndarray,
    edges: np.ndarray,
    interior: np.ndarray,
    saved_target_front: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray]:
    """Decode the residual model contract and evaluate the raw next state."""

    reconstruction = np.asarray(
        normalized_residual_reconstruction,
        dtype=np.float64,
    )
    current = np.asarray(reference_current, dtype=np.float64)
    target = np.asarray(physical_target, dtype=np.float64)
    scale = np.asarray(residual_scale, dtype=np.float64).reshape(1, -1)
    if reconstruction.shape != current.shape or target.shape != current.shape:
        raise ValueError("residual reconstruction and paired states must align")
    if current.ndim != 2 or current.shape[1] != 4 or scale.shape != (1, 4):
        raise ValueError("decoded residual contract requires [N,4] states and scale")
    decoded = current + reconstruction * scale
    admissibility = raw_admissibility_summary(decoded)
    decoded_primitive = conservative_to_primitive_raw(decoded)
    target_primitive = conservative_to_primitive_raw(target)
    selected = np.asarray(interior, dtype=bool)
    decoded_pressure = decoded_primitive[selected, 3]
    target_pressure = target_primitive[selected, 3]
    pressure_range = float(np.ptp(target_pressure))
    if (
        pressure_range > 0.0
        and np.all(np.isfinite(decoded_pressure))
        and np.all(np.isfinite(target_pressure))
    ):
        overshoot = float(
            max(float(np.max(decoded_pressure) - np.max(target_pressure)), 0.0)
            / pressure_range
        )
        undershoot = float(
            max(float(np.min(target_pressure) - np.min(decoded_pressure)), 0.0)
            / pressure_range
        )
    else:
        overshoot = None
        undershoot = None
    pressure_structure = _pressure_structure_metrics(
        decoded_primitive[:, 3:4],
        target_primitive[:, 3:4],
        positions=positions,
        edges=edges,
        interior=selected,
        saved_target_front=saved_target_front,
    )
    return (
        {
            "decoder": "U_current + residual_scale * normalized_residual",
            "raw_admissibility": admissibility,
            "pressure_structure": pressure_structure,
            "pressure_overshoot_fraction": overshoot,
            "pressure_undershoot_fraction": undershoot,
        },
        decoded,
    )


def _validate_residual_pairing(
    normalized_residual: np.ndarray,
    reference_current: np.ndarray,
    target: np.ndarray,
) -> list[float]:
    residual = np.asarray(normalized_residual, dtype=np.float64)
    delta = np.asarray(target, dtype=np.float64) - np.asarray(
        reference_current,
        dtype=np.float64,
    )
    if residual.shape != delta.shape or residual.shape[1] != 4:
        raise ValueError("normalized residual does not match trajectory state")
    scales: list[float] = []
    for component in range(4):
        x = residual[:, component]
        y = delta[:, component]
        denominator = float(x @ x)
        if denominator <= 1e-20:
            raise ValueError("cannot identify residual scale from source pair")
        scale = float((x @ y) / denominator)
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("source pair implies an invalid residual scale")
        if not np.allclose(x * scale, y, rtol=2e-4, atol=2e-6):
            raise ValueError("basis residual target does not match trajectory pair")
        scales.append(scale)
    return scales


def _load_case(
    input_dir: Path,
    summary: Mapping[str, Any],
    key: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    trajectory_path = input_dir / f"trajectory_{key}.npz"
    basis_path = input_dir / f"basis_{key}.npz"
    if not trajectory_path.is_file() or not basis_path.is_file():
        raise FileNotFoundError(f"missing paired artifacts for trajectory {key}")
    with np.load(trajectory_path, allow_pickle=False) as source:
        trajectory = {name: np.array(source[name], copy=True) for name in source.files}
    with np.load(basis_path, allow_pickle=False) as source:
        basis = {name: np.array(source[name], copy=True) for name in source.files}

    checkpoint = summary["checkpoint"]
    evaluation = summary["evaluation"]
    if _json_scalar(trajectory["schema"], name="schema") != summary["schema"]:
        raise ValueError(f"trajectory schema mismatch for {key}")
    if str(_json_scalar(trajectory["trajectory_key"], name="trajectory_key")) != key:
        raise ValueError(f"trajectory key mismatch for {key}")
    if (
        _json_scalar(trajectory["checkpoint_sha256"], name="checkpoint_sha256")
        != checkpoint["sha256"]
    ):
        raise ValueError(f"checkpoint digest mismatch for {key}")
    if (
        _json_scalar(trajectory["config_digest"], name="config_digest")
        != checkpoint["config_digest"]
    ):
        raise ValueError(f"config digest mismatch for {key}")
    if (
        _json_scalar(trajectory["boundary_mode"], name="boundary_mode")
        != checkpoint["boundary_mode"]
    ):
        raise ValueError(f"boundary contract mismatch for {key}")

    positions = _required_array(trajectory, "positions", ndim=2).astype(np.float64)
    edges = _required_array(trajectory, "edges", ndim=2).astype(np.int64)
    node_type = _required_array(trajectory, "node_type", ndim=1)
    node_count = positions.shape[0]
    if node_count < MIN_FULL_RESOLUTION_NODES:
        raise ValueError(
            f"trajectory {key} has {node_count} nodes; full-resolution audit "
            f"requires at least {MIN_FULL_RESOLUTION_NODES}"
        )
    if positions.shape[1] != 2 or edges.shape[1] != 2:
        raise ValueError(f"invalid geometry shape for {key}")
    if node_type.shape != (node_count,):
        raise ValueError(f"node types do not align for {key}")
    if edges.size and (edges.min() < 0 or edges.max() >= node_count):
        raise ValueError(f"edge index lies outside trajectory {key}")
    weights = {
        "equal_node_proxy": _required_array(
            trajectory,
            "equal_node_weights_proxy",
            ndim=1,
        ).astype(np.float64),
        "reconstructed_weight_proxy": _required_array(
            trajectory,
            "reconstructed_node_weights_proxy",
            ndim=1,
        ).astype(np.float64),
    }
    if any(value.shape != (node_count,) for value in weights.values()):
        raise ValueError(f"diagnostic weights do not align for {key}")
    if not np.allclose(weights["equal_node_proxy"], 1.0):
        raise ValueError(f"equal-node proxy is not equal for {key}")
    for weight_name, value in weights.items():
        normalized_node_weights(value, name=f"{key} {weight_name}")

    basis_call = int(evaluation["basis_call"])
    call_index = basis_call - 1
    reference_currents = _required_array(
        trajectory,
        "reference_currents",
        ndim=3,
    )
    targets = _required_array(trajectory, "targets", ndim=3)
    if call_index >= reference_currents.shape[0] or call_index >= targets.shape[0]:
        raise ValueError(f"basis call is unavailable in trajectory {key}")
    reference_current = np.asarray(reference_currents[call_index], dtype=np.float64)
    physical_target = np.asarray(targets[call_index], dtype=np.float64)
    if reference_current.shape != (node_count, 4) or physical_target.shape != (
        node_count,
        4,
    ):
        raise ValueError(f"trajectory states do not align for {key}")

    front_mask = _required_array(basis, "target_front_mask", ndim=1).astype(bool)
    smooth_mask = _required_array(basis, "smooth_mask", ndim=1).astype(bool)
    if (
        front_mask.shape != (node_count,)
        or smooth_mask.shape != (node_count,)
        or not np.any(front_mask)
        or not np.any(smooth_mask)
        or np.any(front_mask & smooth_mask)
    ):
        raise ValueError(f"invalid saved front/smooth masks for {key}")
    interior = node_type.reshape(-1) == 0
    if np.count_nonzero(interior) < 3:
        raise ValueError(f"trajectory {key} lacks an interior-node domain")
    if np.any(front_mask & ~interior) or np.any(smooth_mask & ~interior):
        raise ValueError(f"saved masks leave the interior domain for {key}")
    shock_separated_smooth = _shock_separated_smooth_mask(
        front_mask,
        interior,
        edges,
        dilation_hops=2,
    )

    fields: dict[str, np.ndarray] = {}
    controls: dict[str, dict[str, dict[str, np.ndarray]]] = {
        weight_name: {} for weight_name in WEIGHT_NAMES
    }
    for field_name in FIELD_NAMES:
        canonical_target = _required_array(
            basis,
            f"equal_node_proxy_{field_name}_target",
            ndim=2,
        ).astype(np.float64)
        if canonical_target.shape[0] != node_count:
            raise ValueError(f"basis field {field_name} does not align for {key}")
        for weight_name in WEIGHT_NAMES:
            target_copy = _required_array(
                basis,
                f"{weight_name}_{field_name}_target",
                ndim=2,
            )
            span_target = _required_array(
                basis,
                f"{weight_name}_{field_name}_coordinate_span_target",
                ndim=2,
            )
            if not np.array_equal(target_copy, canonical_target) or not np.array_equal(
                span_target,
                canonical_target,
            ):
                raise ValueError(f"basis targets disagree for {key} {field_name}")
            controls[weight_name][field_name] = {
                "current_fourier_mass_orthogonal": _required_array(
                    basis,
                    f"{weight_name}_{field_name}_mass_orthogonal_reconstruction",
                    ndim=2,
                ).astype(np.float64),
                "coordinate_span_fourier_mass_orthogonal": _required_array(
                    basis,
                    (
                        f"{weight_name}_{field_name}_coordinate_span_"
                        "mass_orthogonal_reconstruction"
                    ),
                    ndim=2,
                ).astype(np.float64),
            }
        fields[field_name] = canonical_target

    inferred_residual_scale = _validate_residual_pairing(
        fields[RESIDUAL_FIELD],
        reference_current,
        physical_target,
    )
    physical_pressure = conservative_to_primitive_raw(physical_target)[:, 3]
    proxy_mass = normalized_node_weights(
        weights["reconstructed_weight_proxy"],
        name="source pressure pairing",
    )
    pressure_mean = float(np.sum(proxy_mass * physical_pressure))
    pressure_scale = float(
        np.sqrt(np.sum(proxy_mass * (physical_pressure - pressure_mean) ** 2))
    )
    expected_pressure = (
        (physical_pressure - pressure_mean) / max(pressure_scale, 1e-12)
    )[:, None]
    if not np.allclose(
        fields[PRESSURE_FIELD],
        expected_pressure,
        rtol=2e-4,
        atol=2e-5,
    ):
        raise ValueError(f"basis pressure target does not match trajectory {key}")

    metadata = {
        "trajectory": key,
        "node_count": int(node_count),
        "basis_call": basis_call,
        "checkpoint_sha256": checkpoint["sha256"],
        "config_digest": checkpoint["config_digest"],
        "data_manifest_digest": checkpoint["data_manifest_digest"],
        "trajectory_artifact_sha256": sha256_file(trajectory_path),
        "basis_artifact_sha256": sha256_file(basis_path),
        "inferred_residual_scale": inferred_residual_scale,
        "pressure_normalization_mean": pressure_mean,
        "pressure_normalization_scale": pressure_scale,
        "smooth_region_contract": (
            "reference q0.9 front dilated by two undirected graph hops, then "
            "evaluated on the induced remaining-interior subgraph"
        ),
        "smooth_region_node_count": int(np.count_nonzero(shock_separated_smooth)),
        "pairing_validation": (
            "key, schema, checkpoint/config digests, boundary, call, geometry, "
            "masks, residual target, pressure target, and duplicated basis targets"
        ),
        "pairing_limit": (
            "basis NPZ has no independent digest fields; pairing beyond these "
            "cross-checks cannot be established from the source schema"
        ),
    }
    arrays = {
        "positions": positions,
        "edges": edges,
        "node_type": node_type,
        "front_mask": front_mask,
        "saved_smooth_mask": smooth_mask,
        "smooth_mask": shock_separated_smooth,
        "interior": interior,
        "reference_current": reference_current,
        "physical_target": physical_target,
        "inferred_residual_scale": np.asarray(inferred_residual_scale),
        "physical_pressure": physical_pressure[:, None],
        **{f"weight_{name}": value for name, value in weights.items()},
        **{f"field_{name}": value for name, value in fields.items()},
    }
    for weight_name, weight_controls in controls.items():
        for field_name, field_controls in weight_controls.items():
            for control_name, value in field_controls.items():
                arrays[f"control_{weight_name}_{field_name}_{control_name}"] = value
    return metadata, arrays


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / max(denominator, 1e-15))


def evaluate_frozen_gates(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(rows) != EXPECTED_TRAJECTORY_COUNT:
        raise ValueError("frozen gates require exactly five trajectory rows")

    locality_case_pass = []
    rank_case_pass: dict[str, list[bool]] = {name: [] for name in WEIGHT_NAMES}
    residual_current_ratios: dict[str, list[float]] = {
        name: [] for name in WEIGHT_NAMES
    }
    residual_current_wins: dict[str, int] = {name: 0 for name in WEIGHT_NAMES}
    residual_span_ratios: list[float] = []
    highpass_current_ratios: dict[str, list[float]] = {
        name: [] for name in WEIGHT_NAMES
    }
    highpass_current_wins: dict[str, int] = {name: 0 for name in WEIGHT_NAMES}
    decoded_state_case_pass: list[bool] = []

    for row in rows:
        basis = row["local_basis"]
        locality_case_pass.append(
            bool(basis["finite"])
            and int(basis["covered_node_count"]) == int(basis["node_count"])
            and float(basis["partition_of_unity_max_row_error"])
            <= FROZEN_GATES["partition_of_unity_max_row_error"]
            and float(basis["column_support_fraction_median"])
            <= FROZEN_GATES["locality_column_support_fraction_median_max"]
            and float(basis["column_support_fraction_p90"])
            <= FROZEN_GATES["locality_column_support_fraction_p90_max"]
        )
        for weight_name in WEIGHT_NAMES:
            fields = row["weights"][weight_name]["fields"]
            residual = fields[RESIDUAL_FIELD]
            local = residual["local_wendland_pou_mass_projection"]
            current = residual["current_fourier_mass_orthogonal"]
            span = residual["coordinate_span_fourier_mass_orthogonal"]
            rank_case_pass[weight_name].append(
                int(row["weights"][weight_name]["numerical_rank"])
                >= FROZEN_GATES["effective_rank_min"]
            )
            residual_current_ratios[weight_name].append(
                _ratio(float(local["rmse"]), float(current["rmse"]))
            )
            residual_current_wins[weight_name] += int(
                float(local["rmse"]) <= float(current["rmse"])
            )
            highpass_current_ratios[weight_name].append(
                _ratio(
                    float(local["smooth_error_highpass_rms"]),
                    float(current["smooth_error_highpass_rms"]),
                )
            )
            highpass_current_wins[weight_name] += int(
                float(local["smooth_error_highpass_rms"])
                <= float(current["smooth_error_highpass_rms"])
            )
            if weight_name == "reconstructed_weight_proxy":
                residual_span_ratios.append(
                    _ratio(float(local["rmse"]), float(span["rmse"]))
                )

            decoded = local.get("decoded_next_state", {})
            structure = decoded.get("pressure_structure", {})
            admissibility = decoded.get("raw_admissibility", {})
            centroid = _finite_scalar(
                structure.get("front_centroid_distance_in_median_edges")
            )
            thickness = _finite_scalar(structure.get("thickness_ratio"))
            strength = _finite_scalar(structure.get("strength_ratio"))
            overshoot = _finite_scalar(decoded.get("pressure_overshoot_fraction"))
            undershoot = _finite_scalar(decoded.get("pressure_undershoot_fraction"))
            decoded_state_case_pass.append(
                bool(admissibility.get("all_finite"))
                and bool(admissibility.get("all_admissible"))
                and structure.get("status") == "available"
                and None not in (centroid, thickness, strength, overshoot, undershoot)
                and centroid
                <= FROZEN_GATES["decoded_pressure_centroid_distance_median_edge_max"]
                and FROZEN_GATES["decoded_pressure_thickness_ratio_range"][0]
                <= thickness
                <= FROZEN_GATES["decoded_pressure_thickness_ratio_range"][1]
                and FROZEN_GATES["decoded_pressure_strength_ratio_range"][0]
                <= strength
                <= FROZEN_GATES["decoded_pressure_strength_ratio_range"][1]
                and overshoot <= FROZEN_GATES["decoded_pressure_overshoot_fraction_max"]
                and undershoot
                <= FROZEN_GATES["decoded_pressure_undershoot_fraction_max"]
            )

    locality_pass = all(locality_case_pass)
    rank_pass = all(all(values) for values in rank_case_pass.values())
    residual_vs_current_pass = all(
        float(np.median(residual_current_ratios[name]))
        <= FROZEN_GATES["residual_local_to_current_median_max_both_proxies"]
        and residual_current_wins[name]
        >= FROZEN_GATES["residual_local_no_worse_case_count_min_both_proxies"]
        for name in WEIGHT_NAMES
    )
    residual_vs_span_pass = (
        float(np.median(residual_span_ratios))
        <= FROZEN_GATES["residual_reconstructed_local_to_span_median_max"]
    )
    smooth_highpass_pass = all(
        float(np.median(highpass_current_ratios[name]))
        <= FROZEN_GATES[
            "smooth_residual_highpass_local_to_current_median_max_both_proxies"
        ]
        and highpass_current_wins[name]
        >= FROZEN_GATES[
            "smooth_residual_highpass_local_no_worse_case_count_min_both_proxies"
        ]
        for name in WEIGHT_NAMES
    )
    decoded_state_pass = all(decoded_state_case_pass)
    gates = {
        "locality_and_partition": locality_pass,
        "effective_rank": rank_pass,
        "residual_vs_current_fourier": residual_vs_current_pass,
        "residual_vs_coordinate_span_fourier": residual_vs_span_pass,
        "smooth_residual_highpass": smooth_highpass_pass,
        "decoded_residual_admissibility_and_anti_smearing": decoded_state_pass,
    }
    return {
        "passed": all(gates.values()),
        "gates": gates,
        "evidence": {
            "locality_case_pass": locality_case_pass,
            "rank_case_pass": rank_case_pass,
            "residual_local_to_current_rmse_ratio_median": {
                name: float(np.median(values))
                for name, values in residual_current_ratios.items()
            },
            "residual_local_no_worse_case_count": residual_current_wins,
            "residual_reconstructed_local_to_span_rmse_ratio_median": float(
                np.median(residual_span_ratios)
            ),
            "smooth_residual_highpass_local_to_current_ratio_median": {
                name: float(np.median(values))
                for name, values in highpass_current_ratios.items()
            },
            "smooth_residual_highpass_local_no_worse_case_count": (
                highpass_current_wins
            ),
            "decoded_residual_case_pass": decoded_state_case_pass,
        },
        "thresholds": FROZEN_GATES,
        "fail_closed": (
            "missing, nonfinite, or inadmissible decoded residual states fail "
            "the anti-smearing gate"
        ),
    }


def run_case(
    input_dir: Path,
    output_dir: Path,
    summary: Mapping[str, Any],
    key: str,
) -> dict[str, Any]:
    metadata, source = _load_case(input_dir, summary, key)
    positions = source["positions"]
    edges = source["edges"]
    front_mask = source["front_mask"]
    smooth_mask = source["smooth_mask"]
    interior = source["interior"]
    center_summary, center_arrays = deterministic_farthest_point_centers(
        positions,
        center_count=LOCAL_BASIS_CENTER_COUNT,
    )
    basis_summary, local_basis = wendland_c2_partition_of_unity_basis(
        center_arrays["normalized_positions"],
        center_arrays["normalized_centers"],
        fill_distance=float(center_summary["fill_distance_h"]),
        radius_factor=LOCAL_BASIS_RADIUS_FACTOR,
    )
    artifact_arrays: dict[str, np.ndarray] = {
        "schema": np.asarray(LOCAL_BASIS_SCHEMA),
        "trajectory_key": np.asarray(key),
        "source_checkpoint_sha256": np.asarray(metadata["checkpoint_sha256"]),
        "source_config_digest": np.asarray(metadata["config_digest"]),
        "source_data_manifest_digest": np.asarray(metadata["data_manifest_digest"]),
        "center_indices": center_arrays["center_indices"],
        "normalized_centers": center_arrays["normalized_centers"].astype(np.float32),
        "minimum_center_distance": center_arrays["minimum_center_distance"].astype(
            np.float32
        ),
        "fill_distance_h": np.asarray(center_summary["fill_distance_h"]),
        "radius": np.asarray(basis_summary["radius"]),
        "reference_front_mask": front_mask,
        "shock_separated_smooth_mask": smooth_mask,
        "reference_current": source["reference_current"].astype(np.float32),
        "physical_target": source["physical_target"].astype(np.float32),
        "inferred_residual_scale": source["inferred_residual_scale"],
    }
    weight_rows: dict[str, Any] = {}
    for weight_name in WEIGHT_NAMES:
        weights = source[f"weight_{weight_name}"]
        combined = np.concatenate(
            [source[f"field_{name}"] for name in FIELD_NAMES],
            axis=1,
        )
        projection_summary, projection_arrays = weighted_basis_projection_audit(
            local_basis,
            weights,
            combined,
            regions={"target_front": front_mask, "smooth": smooth_mask},
        )
        combined_reconstruction = projection_arrays["reconstruction"]
        residual_width = source[f"field_{RESIDUAL_FIELD}"].shape[1]
        local_reconstructions = {
            RESIDUAL_FIELD: combined_reconstruction[:, :residual_width],
            PRESSURE_FIELD: combined_reconstruction[:, residual_width:],
        }
        field_rows: dict[str, Any] = {}
        for field_name in FIELD_NAMES:
            target = source[f"field_{field_name}"]
            reconstruction_map = {
                "local_wendland_pou_mass_projection": local_reconstructions[field_name],
                **{
                    control_name: source[
                        f"control_{weight_name}_{field_name}_{control_name}"
                    ]
                    for control_name in CONTROL_NAMES
                },
            }
            method_rows: dict[str, Any] = {}
            for method_name, reconstruction in reconstruction_map.items():
                metrics = _weighted_reconstruction_metrics(
                    reconstruction,
                    target,
                    weights,
                    smooth_mask=smooth_mask,
                    front_mask=front_mask,
                    edges=edges,
                )
                if field_name == RESIDUAL_FIELD:
                    decoded_metrics, _ = _decoded_residual_state_metrics(
                        reconstruction,
                        source["reference_current"],
                        source["physical_target"],
                        source["inferred_residual_scale"],
                        positions=positions,
                        edges=edges,
                        interior=interior,
                        saved_target_front=front_mask,
                    )
                    metrics["decoded_next_state"] = decoded_metrics
                elif field_name == PRESSURE_FIELD:
                    physical_reconstruction = (
                        reconstruction * metadata["pressure_normalization_scale"]
                        + metadata["pressure_normalization_mean"]
                    )
                    metrics["pressure_structure"] = _pressure_structure_metrics(
                        physical_reconstruction,
                        source["physical_pressure"],
                        positions=positions,
                        edges=edges,
                        interior=interior,
                        saved_target_front=front_mask,
                    )
                    metrics["pressure_structure_claim"] = (
                        "context-only full-pressure projection; not used by the "
                        "residual-model promotion gate"
                    )
                method_rows[method_name] = metrics
                artifact_arrays[f"{weight_name}_{field_name}_{method_name}"] = (
                    np.asarray(reconstruction, dtype=np.float32)
                )
            field_rows[field_name] = method_rows
        weight_rows[weight_name] = {
            "numerical_rank": projection_summary["numerical_rank"],
            "rank_fraction": projection_summary["rank_fraction"],
            "correlation_condition_number": projection_summary[
                "correlation_condition_number"
            ],
            "fields": field_rows,
        }
        artifact_arrays[f"{weight_name}_correlation_eigenvalues"] = projection_arrays[
            "correlation_eigenvalues"
        ]

    artifact_name = f"local_basis_{key}.npz"
    np.savez_compressed(output_dir / artifact_name, **artifact_arrays)
    return {
        **metadata,
        "artifact": artifact_name,
        "artifact_sha256": sha256_file(output_dir / artifact_name),
        "center_selection": center_summary,
        "local_basis": basis_summary,
        "weights": weight_rows,
        "claim_boundary": (
            "full-resolution, fixed no-training projection evidence only; "
            "proxy weights do not support physical conservation claims"
        ),
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")
    summary, keys = _load_source_contract(args.input_dir)
    args.output_dir.mkdir(parents=True)
    rows = [run_case(args.input_dir, args.output_dir, summary, key) for key in keys]
    gate_result = evaluate_frozen_gates(rows)
    output = {
        "schema": LOCAL_BASIS_SCHEMA,
        "status": "complete",
        "source": {
            "bundle": str(args.input_dir),
            "summary_sha256": sha256_file(args.input_dir / "summary.json"),
            "schema": summary["schema"],
            "checkpoint_sha256": summary["checkpoint"]["sha256"],
            "config_digest": summary["checkpoint"]["config_digest"],
            "data_manifest_digest": summary["checkpoint"]["data_manifest_digest"],
            "trajectory_keys": keys,
        },
        "method": {
            "center_count": LOCAL_BASIS_CENTER_COUNT,
            "center_selection": (
                "deterministic geometry-only FPS on span-normalized positions; "
                "first center nearest (0.5,0.5)"
            ),
            "basis": "Wendland-C2 row-normalized partition of unity",
            "radius": "R=2h where h is the FPS fill distance",
            "coefficient_count_match": (
                "289 local columns versus 289 saved real Fourier columns"
            ),
            "fit": "same weighted normal-equation pseudoinverse, rcond=1e-10",
            "smooth_region": (
                "reference q0.9 front plus two graph hops excluded; high-pass "
                "evaluated on the induced smooth subgraph"
            ),
            "anti_smearing_decoder": (
                "U_current + residual_scale * projected normalized residual; "
                "raw conservative-to-primitive conversion without floors"
            ),
        },
        "frozen_gate_result": gate_result,
        "trajectories": rows,
        "artifact_schema": {
            "local_basis_npz": (
                "center indices/geometry, local and saved-control "
                "reconstructions, paired current/target states, residual scale, "
                "reference masks, and weighted-Gram eigenvalues"
            ),
        },
        "claim_boundary": {
            "verified": (
                "fixed projection and decoded conservative-residual behavior on "
                "paired full-resolution D013 artifact arrays under two "
                "diagnostic proxy measures"
            ),
            "unsupported": (
                "training or rollout improvement, physical conservation, "
                "physical flux, or a universal Gibbs explanation"
            ),
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(gate_result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
