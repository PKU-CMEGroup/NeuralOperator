#!/usr/bin/env python3
"""Run the registered reference-free PCFNO rollout through H320."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.time_dependent_no.evaluate_pcno_euler2d_residual import (
    NATIVE_CAUSAL_BOUNDARY_SCOPE,
    PCNOEuler2DShardStore,
    apply_causal_boundary_conservative,
    build_graph_causal_boundary_policy,
    build_model,
    load_checkpoint,
    model_call,
    preprocessing_contract_audit,
    select_device,
    sha256_file,
    synchronize,
)
from scripts.time_dependent_no.train_pcno_bump_gradient_ablation import (
    B1_DATA_MANIFEST_SHA256,
    B1_SOURCE_SET_SHA256,
    checkpoint_differential_branch_mode,
    verify_frozen_b1_sources,
)

SCHEMA = "w26_l1_pcfno_h320_rollout_v1"
ANIMATION_SCHEMA = "w26_l1_pcfno_h320_animation_bundle_v1"
WORKING_ID = "W26-L1-PCFNO-H320-A1-S20260718"
CHECKPOINT_SHA256 = "ff5c24fb8d1be813b97d44ebdfde8693d196c467adf4a9d5ddc7c1d4d28f9d0d"
STRICT_SUMMARY_SHA256 = (
    "5625eef2b22763c19f5189febcd5fb6e59f547ba8aea3ef072e8e3d5eb114396"
)
STRICT_TRAJECTORY_SET_SHA256 = (
    "01d0f1da7870e553bd646afe815178e91d583a3c5d1b83b075735a945d61ae3f"
)
CONTINUATION_SUMMARY_SHA256 = (
    "8e9aa4cb1c49b0fd1ad34ad8120994d1c8953e86bcd122d868d8da9d9bf93ab4"
)
CONTINUATION_MANIFEST_SHA256 = (
    "e299fc62b43c8b8767f33b84e90f1f434509115c3b9c64d2a598ca85aa928867"
)
TRAJECTORY_KEYS = (
    "7",
    "16",
    "18",
    "23",
    "47",
    "54",
    "58",
    "60",
    "72",
    "82",
    "101",
    "103",
    "112",
    "120",
    "126",
    "128",
    "141",
    "145",
    "150",
    "172",
    "187",
    "188",
    "190",
    "211",
    "227",
    "233",
    "235",
    "251",
    "287",
    "296",
)
CONTINUATION_KEYS = frozenset(("54", "227", "233"))
ANIMATION_KEYS = ("187", "54", "227", "233")
TRUTH_STEPS = 79
PRODUCTION_STEPS = 320
EVENT_NAMES = ("admissible", "bounded", "finite")
FINAL_HASH_MANIFEST_SCHEMA = "pcno_long_horizon_final_hash_manifest_v1"
FINAL_HASH_MANIFEST_NAME = "final_hash_manifest.json"
SOURCE_PATHS = (
    "docs/time_dependent_no/W26_L1_PCFNO_H320_PREREGISTRATION.md",
    "scripts/time_dependent_no/evaluate_pcfno_h320_rollout.py",
    "scripts/time_dependent_no/visualize_pcfno_h320_rollout.py",
    "tests/time_dependent_no/test_pcfno_h320_rollout.py",
    "scripts/time_dependent_no/evaluate_pcno_long_horizon_stability.py",
    "scripts/time_dependent_no/evaluate_pcno_euler2d_residual.py",
    "scripts/time_dependent_no/train_pcno_bump_gradient_ablation.py",
    "scripts/time_dependent_no/visualize_pcno_inadmissibility_continuation.py",
    "utility/time_dependent_no/pcno_inadmissibility.py",
)


@dataclass(frozen=True)
class StabilityThresholds:
    """Frozen D087 thresholds used by the three reference-free events."""

    accurate_relative_l2_max: float = 0.05
    amplitude_ratio_max: float = 100.0
    scaled_rms_ratio_max: float = 10.0

    def __post_init__(self) -> None:
        for name, value in self.as_dict().items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")

    def as_dict(self) -> dict[str, float]:
        return {
            "accurate_relative_l2_max": self.accurate_relative_l2_max,
            "amplitude_ratio_max": self.amplitude_ratio_max,
            "scaled_rms_ratio_max": self.scaled_rms_ratio_max,
        }


THRESHOLDS = StabilityThresholds()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "production"), required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--training-data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--strict-rollout-dir", type=Path, required=True)
    parser.add_argument("--continuation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    return args


def mapping_sha256(value: Any) -> str:
    payload = json.dumps(
        _json_safe(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    values = list(rows)
    columns: list[str] = []
    for row in values:
        for name in row:
            if name not in columns:
                columns.append(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in values:
            writer.writerow(
                {
                    name: (
                        json.dumps(_json_safe(item), sort_keys=True)
                        if isinstance(item, (dict, list, tuple))
                        else _json_safe(item)
                    )
                    for name, item in row.items()
                }
            )


def load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.array(archive[name], copy=True) for name in archive.files}


def write_npz_atomic(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Write an uncompressed NPZ without retaining a compression workspace."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _closed_top_level_artifact_names(artifact_names: Sequence[str]) -> list[str]:
    names = list(artifact_names)
    if not names:
        raise ValueError("at least one closed artifact file is required")
    for name in names:
        if not isinstance(name, str):
            raise TypeError("artifact names must be strings")
        path = PurePosixPath(name)
        if (
            not name
            or "\\" in name
            or path.is_absolute()
            or len(path.parts) != 1
            or path.name != name
            or name in {".", ".."}
        ):
            raise ValueError("each artifact must use one top-level POSIX file name")
        if name == FINAL_HASH_MANIFEST_NAME:
            raise ValueError("the final hash manifest must not include itself")
    if len(set(names)) != len(names):
        raise ValueError("artifact names must be unique")
    return sorted(names)


def _require_exact_top_level_inventory(root: Path, names: Sequence[str]) -> None:
    expected = set(names)
    observed = {
        entry.name for entry in root.iterdir() if entry.is_file() or entry.is_symlink()
    }
    observed.discard(FINAL_HASH_MANIFEST_NAME)
    missing = sorted(expected.difference(observed))
    unexpected = sorted(observed.difference(expected))
    if missing or unexpected:
        raise ValueError(
            "artifact root differs from the declared closed inventory; "
            f"missing files={missing}; unregistered top-level files={unexpected}"
        )


def build_final_hash_manifest(
    artifact_root: str | Path, artifact_names: Sequence[str]
) -> dict[str, Any]:
    """Build the exact closed flat-file manifest defined by D087."""

    root = Path(artifact_root)
    if not root.is_dir():
        raise ValueError("artifact_root must be an existing directory")
    root = root.resolve()
    names = _closed_top_level_artifact_names(artifact_names)
    _require_exact_top_level_inventory(root, names)
    files: list[dict[str, Any]] = []
    for name in names:
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"artifact must be a regular non-symlink file: {name}")
        before = path.stat()
        digest = sha256_file(path)
        after = path.stat()
        before_identity = (before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        after_identity = (after.st_size, after.st_mtime_ns, after.st_ctime_ns)
        if before_identity != after_identity:
            raise ValueError(
                f"artifact changed while hashing; close its writer: {name}"
            )
        files.append({"path": name, "size": after.st_size, "sha256": digest})
    _require_exact_top_level_inventory(root, names)
    return {"schema": FINAL_HASH_MANIFEST_SCHEMA, "files": files}


def verify_final_hash_manifest(
    artifact_root: str | Path, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """Rebuild and verify the exact closed flat-file manifest defined by D087."""

    if manifest.get("schema") != FINAL_HASH_MANIFEST_SCHEMA:
        raise ValueError("unsupported final hash manifest schema")
    files = manifest.get("files")
    if not isinstance(files, list) or not all(
        isinstance(row, Mapping) for row in files
    ):
        raise TypeError("final hash manifest files must be a list of mappings")
    try:
        names = [str(row["path"]) for row in files]
    except KeyError as exc:
        raise ValueError("every final hash manifest row requires path") from exc
    rebuilt = build_final_hash_manifest(artifact_root, names)
    if rebuilt != dict(manifest):
        raise ValueError("artifact root does not match the closed-file manifest")
    return {"schema": FINAL_HASH_MANIFEST_SCHEMA, "verified_file_count": len(files)}


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def source_hashes() -> dict[str, str]:
    missing = [relative for relative in SOURCE_PATHS if not (ROOT / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"registered source files are absent: {missing}")
    return {relative: sha256_file(ROOT / relative) for relative in SOURCE_PATHS}


def canonical_npz_inventory(root: Path) -> tuple[list[dict[str, Any]], str]:
    paths = sorted((root / "trajectories").glob("trajectory_*.npz"))
    rows = [
        {"path": path.name, "size": path.stat().st_size, "sha256": sha256_file(path)}
        for path in paths
    ]
    return rows, mapping_sha256(rows)


def verify_prefix_roots(strict_root: Path, continuation_root: Path) -> dict[str, Any]:
    strict_summary = strict_root / "summary.json"
    continuation_summary = continuation_root / "summary.json"
    continuation_manifest = continuation_root / "manifest.json"
    checks = {
        "strict_summary": sha256_file(strict_summary) == STRICT_SUMMARY_SHA256,
        "continuation_summary": (
            sha256_file(continuation_summary) == CONTINUATION_SUMMARY_SHA256
        ),
        "continuation_manifest": (
            sha256_file(continuation_manifest) == CONTINUATION_MANIFEST_SHA256
        ),
    }
    inventory, inventory_digest = canonical_npz_inventory(strict_root)
    checks["strict_trajectory_count"] = len(inventory) == len(TRAJECTORY_KEYS)
    checks["strict_trajectory_set"] = inventory_digest == STRICT_TRAJECTORY_SET_SHA256
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(f"H79 prefix roots failed closure: {failed}")
    continuation_payload = json.loads(continuation_manifest.read_text(encoding="utf-8"))
    return {
        "checks": checks,
        "strict_inventory_sha256": inventory_digest,
        "strict_inventory": inventory,
        "continuation_output_sha256": continuation_payload["output_sha256"],
    }


def expected_prefix(
    key: str,
    *,
    strict_root: Path,
    continuation_root: Path,
    continuation_output_sha256: Mapping[str, str],
) -> tuple[np.ndarray, dict[str, Any]]:
    if key in CONTINUATION_KEYS:
        relative = f"trajectories/trajectory_{key}.npz"
        path = continuation_root / relative
        if sha256_file(path) != continuation_output_sha256.get(relative):
            raise ValueError(f"continuation prefix hash changed for trajectory {key}")
        arrays = load_npz(path)
        states = np.asarray(arrays["deployed_states_conservative"], dtype=np.float32)
        if states.shape[0] != TRUTH_STEPS + 1:
            raise ValueError(f"continuation prefix for {key} does not end at H79")
        if str(np.asarray(arrays["checkpoint_sha256"]).item()) != CHECKPOINT_SHA256:
            raise ValueError(f"continuation prefix checkpoint changed for {key}")
        return states, {
            "kind": "finite_invalid_continuation",
            "path": relative,
            "sha256": sha256_file(path),
        }

    relative = f"trajectories/trajectory_{key}.npz"
    path = strict_root / relative
    arrays = load_npz(path)
    initial = np.asarray(arrays["initial_conservative"], dtype=np.float32)
    predictions = np.asarray(
        arrays["pcno_baseline_predictions_conservative"], dtype=np.float32
    )
    valid_length = int(np.asarray(arrays["baseline_valid_length"]).item())
    checkpoint = str(np.asarray(arrays["checkpoint_sha256"]).item())
    if valid_length != TRUTH_STEPS or predictions.shape[0] != TRUTH_STEPS:
        raise ValueError(f"strict prefix for {key} is not a complete H79 rollout")
    if checkpoint != CHECKPOINT_SHA256:
        raise ValueError(f"strict prefix checkpoint changed for {key}")
    states = np.concatenate((initial[None], predictions), axis=0)
    return states, {
        "kind": "strict_h79_survivor",
        "path": relative,
        "sha256": sha256_file(path),
    }


def _check_euler_state(state: torch.Tensor) -> None:
    if state.ndim < 2 or state.shape[-1] != 4:
        raise ValueError("Euler state must have shape [..., N, 4]")


def _active_mask(state: torch.Tensor, node_mask: torch.Tensor | None) -> torch.Tensor:
    if node_mask is None:
        return torch.ones(state.shape[:-1], dtype=torch.bool, device=state.device)
    mask = node_mask.to(device=state.device)
    if mask.shape == (*state.shape[:-1], 1):
        mask = mask[..., 0]
    if mask.shape != state.shape[:-1]:
        raise ValueError("node_mask must match state shape without components")
    active = mask > 0
    if not bool(active.any()):
        raise ValueError("node_mask must select at least one active node")
    return active


def _finite_minimum(value: torch.Tensor, active: torch.Tensor) -> float | None:
    selected = value[active]
    finite = selected[torch.isfinite(selected)]
    return None if finite.numel() == 0 else float(finite.min().cpu())


def diagnose_euler_state(
    state: torch.Tensor,
    *,
    representation: str,
    gamma: float = 1.4,
    node_mask: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Apply the frozen D087 native-finiteness and Euler-admissibility view."""

    _check_euler_state(state)
    if representation not in {
        "primitive_rho_v1_v2_p",
        "conservative_rho_m1_m2_E",
    }:
        raise ValueError(f"unsupported state representation: {representation}")
    if gamma <= 1.0:
        raise ValueError("gamma must be greater than one")
    native = state.detach().to(dtype=torch.float64)
    active = _active_mask(native, node_mask)
    rho = native[..., 0]
    if representation == "primitive_rho_v1_v2_p":
        v1, v2, pressure = native[..., 1], native[..., 2], native[..., 3]
        internal_energy = pressure / (gamma - 1.0)
        derived = torch.stack(
            (
                rho * v1,
                rho * v2,
                internal_energy + 0.5 * rho * (v1.square() + v2.square()),
                internal_energy,
            ),
            dim=-1,
        )
    else:
        momentum_x, momentum_y, energy = (
            native[..., 1],
            native[..., 2],
            native[..., 3],
        )
        v1 = momentum_x / rho
        v2 = momentum_y / rho
        internal_energy = (
            energy - 0.5 * (momentum_x.square() + momentum_y.square()) / rho
        )
        pressure = (gamma - 1.0) * internal_energy
        derived = torch.stack((v1, v2, internal_energy, pressure), dim=-1)

    active_components = active.unsqueeze(-1).expand_as(native)
    native_components_finite = bool(
        torch.isfinite(native)[active_components].all().item()
    )
    derived_fields_finite = bool(
        torch.isfinite(derived)[active_components].all().item()
    )
    rho_bad = active & torch.isfinite(rho) & (rho <= 0.0)
    internal_bad = active & torch.isfinite(internal_energy) & (internal_energy <= 0.0)
    pressure_bad = active & torch.isfinite(pressure) & (pressure <= 0.0)
    violation_fields: list[str] = []
    if not native_components_finite:
        violation_fields.append("native_components_nonfinite")
    if not derived_fields_finite:
        violation_fields.append("derived_fields_nonfinite")
    if bool(rho_bad.any()):
        violation_fields.append("nonpositive_density")
    if bool(internal_bad.any()):
        violation_fields.append("nonpositive_internal_energy")
    if bool(pressure_bad.any()):
        violation_fields.append("nonpositive_pressure")

    admissible = (
        native_components_finite
        and derived_fields_finite
        and not bool(rho_bad.any())
        and not bool(internal_bad.any())
        and not bool(pressure_bad.any())
    )
    if admissible:
        primary_failure = "admissible"
    elif not native_components_finite:
        primary_failure = "nonfinite_native_components"
    elif bool(rho_bad.any()):
        primary_failure = "nonpositive_density"
    elif not derived_fields_finite:
        primary_failure = "nonfinite_derived_fields"
    elif representation == "primitive_rho_v1_v2_p" and bool(pressure_bad.any()):
        primary_failure = "nonpositive_pressure"
    elif bool(internal_bad.any()):
        primary_failure = "nonpositive_internal_energy"
    else:
        primary_failure = "nonpositive_pressure"

    return {
        "state_representation": representation,
        "active_node_count": int(active.sum().item()),
        "native_components_finite": native_components_finite,
        "derived_fields_finite": derived_fields_finite,
        "finite": native_components_finite,
        "thermodynamic_admissible": admissible,
        "admissible": admissible,
        "violation_fields": violation_fields,
        "primary_failure": primary_failure,
        "nonpositive_density_node_count": int(rho_bad.sum().item()),
        "nonpositive_internal_energy_node_count": int(internal_bad.sum().item()),
        "nonpositive_pressure_node_count": int(pressure_bad.sum().item()),
        "min_density": _finite_minimum(rho, active),
        "min_internal_energy": _finite_minimum(internal_energy, active),
        "min_pressure": _finite_minimum(pressure, active),
    }


def conservative_to_primitive_numpy(state: np.ndarray, *, gamma: float) -> np.ndarray:
    value = np.asarray(state, dtype=np.float64)
    rho = value[..., 0]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        v1 = value[..., 1] / rho
        v2 = value[..., 2] / rho
        internal = value[..., 3] - 0.5 * (value[..., 1] ** 2 + value[..., 2] ** 2) / rho
        pressure = (gamma - 1.0) * internal
    return np.stack((rho, v1, v2, pressure), axis=-1)


def internal_energy_numpy(state: np.ndarray) -> np.ndarray:
    value = np.asarray(state, dtype=np.float64)
    rho = value[..., 0]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        return value[..., 3] - 0.5 * (value[..., 1] ** 2 + value[..., 2] ** 2) / rho


def node_weights(store: PCNOEuler2DShardStore, key: str) -> np.ndarray:
    weights = np.asarray(store.array(key, "node_weights"), dtype=np.float64)
    if weights.ndim == 1:
        result = weights
    else:
        result = weights.sum(axis=-1)
    if result.ndim != 1 or not np.isfinite(result).all() or np.any(result <= 0.0):
        raise ValueError(f"trajectory {key} has invalid reconstructed weights")
    return result


def _weighted_component_mean_square(
    value: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    return np.sum(weights[:, None] * np.square(value), axis=0) / np.sum(weights)


def presentation_scales(
    states: np.ndarray, *, gamma: float, state_scale: np.ndarray
) -> dict[str, float]:
    primitive = conservative_to_primitive_numpy(states, gamma=gamma)
    density = primitive[..., 0]
    pressure = primitive[..., 3]
    internal = internal_energy_numpy(states)
    pressure_increment = np.diff(pressure, axis=0)
    scaled_increment = np.linalg.norm(
        np.diff(states.astype(np.float64), axis=0) / state_scale[None, None, :],
        axis=-1,
    )

    def limits(value: np.ndarray) -> tuple[float, float]:
        finite = value[np.isfinite(value)]
        if finite.size == 0:
            raise ValueError("reference scale field has no finite values")
        lower, upper = np.quantile(finite, (0.005, 0.995))
        if not upper > lower:
            upper = lower + max(abs(lower), 1.0) * 1.0e-6
        return float(lower), float(upper)

    density_min, density_max = limits(density)
    pressure_min, pressure_max = limits(pressure)
    internal_min, internal_max = limits(internal)
    residual = pressure_increment[np.isfinite(pressure_increment)]
    increment = scaled_increment[np.isfinite(scaled_increment)]
    residual_abs = max(float(np.quantile(np.abs(residual), 0.995)), 1.0e-8)
    increment_max = max(float(np.quantile(increment, 0.995)), 1.0e-8)
    return {
        "density_min": density_min,
        "density_max": density_max,
        "pressure_min": pressure_min,
        "pressure_max": pressure_max,
        "internal_energy_min": internal_min,
        "internal_energy_max": internal_max,
        "pressure_increment_abs_max": 5.0 * residual_abs,
        "scaled_increment_max": 5.0 * increment_max,
        "quantiles": [0.005, 0.995],
        "multiplier_for_increment_limits": 5.0,
        "source": "matching H0-H79 reference for fixed presentation scales only",
    }


def build_calibration_contract(
    store: PCNOEuler2DShardStore,
    keys: Sequence[str],
    *,
    gamma: float,
    state_scale: np.ndarray,
) -> dict[str, Any]:
    component_sum = np.zeros(4, dtype=np.float64)
    frame_count = 0
    for key in keys:
        states = np.asarray(store.states(key)[: TRUTH_STEPS + 1], dtype=np.float64)
        if states.shape[0] != TRUTH_STEPS + 1:
            raise ValueError(f"trajectory {key} lacks H79 reference calibration")
        primitive = conservative_to_primitive_numpy(states[1:], gamma=gamma)
        if not np.isfinite(primitive).all():
            raise ValueError(f"trajectory {key} reference primitive is nonfinite")
        weights = node_weights(store, key)
        for frame in primitive:
            component_sum += _weighted_component_mean_square(frame, weights)
            frame_count += 1
    component_scale = np.sqrt(component_sum / frame_count)
    component_scale = np.maximum(component_scale, 1.0e-12)

    case_envelopes: dict[str, Any] = {}
    animation_scales: dict[str, Any] = {}
    for key in keys:
        states = np.asarray(store.states(key)[: TRUTH_STEPS + 1], dtype=np.float64)
        weights = node_weights(store, key)
        primitive = conservative_to_primitive_numpy(states[1:], gamma=gamma)
        scaled = primitive / component_scale[None, None, :]
        amplitude = float(np.max(np.abs(scaled)))
        rms = float(
            np.max(
                np.sqrt(
                    np.sum(weights[None, :, None] * np.square(scaled), axis=(1, 2))
                    / np.sum(weights)
                )
            )
        )
        integral = np.sum(weights[None, :, None] * states, axis=1)
        integral_scale = np.maximum(np.max(np.abs(integral), axis=0), 1.0e-12)
        case_envelopes[key] = {
            "maximum_abs_scaled_primitive": amplitude,
            "maximum_proxy_scaled_rms": rms,
            "proxy_integral_scale": integral_scale.tolist(),
        }
        if key in ANIMATION_KEYS:
            animation_scales[key] = presentation_scales(
                states, gamma=gamma, state_scale=state_scale
            )
    payload = {
        "schema": f"{SCHEMA}_calibration_v1",
        "truth_calls_used": [1, TRUTH_STEPS],
        "truth_use": (
            "prefix identity, fixed event envelopes, and fixed presentation scales; "
            "never recurrence and never H80-H320 error"
        ),
        "component_order": ["density", "velocity_x", "velocity_y", "pressure"],
        "component_scale": component_scale.tolist(),
        "component_scale_formula": (
            "sqrt(mean_case_call(sum_node w*q^2/sum_node w)), floor 1e-12"
        ),
        "case_envelopes": case_envelopes,
        "animation_scales": animation_scales,
    }
    payload["payload_sha256"] = mapping_sha256(payload)
    return payload


def common_envelope_metrics(
    state: np.ndarray,
    *,
    weights: np.ndarray,
    component_scale: np.ndarray,
    envelope: Mapping[str, Any],
    gamma: float,
) -> tuple[float | None, float | None]:
    primitive = conservative_to_primitive_numpy(state, gamma=gamma)
    if not np.isfinite(primitive).all():
        return None, None
    scaled = primitive / component_scale[None, :]
    amplitude = float(np.max(np.abs(scaled)))
    rms = float(np.sqrt(np.sum(weights[:, None] * np.square(scaled)) / np.sum(weights)))
    return (
        amplitude / float(envelope["maximum_abs_scaled_primitive"]),
        rms / float(envelope["maximum_proxy_scaled_rms"]),
    )


def weighted_scaled_rms(
    value: np.ndarray, weights: np.ndarray, scale: np.ndarray
) -> float | None:
    array = np.asarray(value, dtype=np.float64)
    if not np.isfinite(array).all():
        return None
    return float(
        np.sqrt(
            np.sum(weights[:, None] * np.square(array / scale[None, :]))
            / np.sum(weights)
        )
    )


def proxy_integrals(state: np.ndarray, weights: np.ndarray) -> np.ndarray:
    array = np.asarray(state, dtype=np.float64)
    if not np.isfinite(array).all():
        return np.full(4, np.nan, dtype=np.float64)
    return np.sum(weights[:, None] * array, axis=0)


def normalizer_metrics(
    state: np.ndarray, *, state_mean: np.ndarray, state_scale: np.ndarray
) -> dict[str, float | None]:
    value = np.asarray(state, dtype=np.float64)
    if not np.isfinite(value).all():
        return {
            "maximum_normalizer_excursion": None,
            "normalizer_fraction_gt_6": None,
            "normalizer_fraction_gt_10": None,
        }
    excursion = np.abs((value - state_mean[None, :]) / state_scale[None, :])
    return {
        "maximum_normalizer_excursion": float(np.max(excursion)),
        "normalizer_fraction_gt_6": float(np.mean(excursion > 6.0)),
        "normalizer_fraction_gt_10": float(np.mean(excursion > 10.0)),
    }


def stage_fields(prefix: str, diagnosis: Mapping[str, Any]) -> dict[str, Any]:
    return {
        f"{prefix}_finite": diagnosis["finite"],
        f"{prefix}_admissible": diagnosis["admissible"],
        f"{prefix}_failure": diagnosis["primary_failure"],
        f"{prefix}_min_density": diagnosis["min_density"],
        f"{prefix}_min_internal_energy": diagnosis["min_internal_energy"],
        f"{prefix}_min_pressure": diagnosis["min_pressure"],
        f"{prefix}_invalid_density_count": diagnosis["nonpositive_density_node_count"],
        f"{prefix}_invalid_internal_energy_count": diagnosis[
            "nonpositive_internal_energy_node_count"
        ],
        f"{prefix}_invalid_pressure_count": diagnosis[
            "nonpositive_pressure_node_count"
        ],
    }


def invalid_mask_numpy(state: np.ndarray, *, gamma: float) -> np.ndarray:
    primitive = conservative_to_primitive_numpy(state, gamma=gamma)
    internal = internal_energy_numpy(state)
    return (
        ~np.isfinite(state).all(axis=-1)
        | ~np.isfinite(primitive).all(axis=-1)
        | (primitive[..., 0] <= 0.0)
        | (internal <= 0.0)
        | (primitive[..., 3] <= 0.0)
    )


def invalid_type_counts(
    state: np.ndarray, node_type: np.ndarray, *, gamma: float
) -> dict[str, int]:
    invalid = invalid_mask_numpy(state, gamma=gamma)
    names = {0: "normal", 1: "wall", 2: "outflow", 3: "inflow"}
    result = {"deployed_invalid_node_count": int(np.count_nonzero(invalid))}
    for value, name in names.items():
        result[f"deployed_invalid_{name}_count"] = int(
            np.count_nonzero(invalid & (node_type == value))
        )
    return result


def transition(before: Mapping[str, Any], after: Mapping[str, Any], stage: str) -> str:
    return stage_transition_label(
        str(before["primary_failure"]), str(after["primary_failure"]), stage=stage
    )


@torch.no_grad()
def contract_call(
    model: torch.nn.Module,
    sample: Mapping[str, torch.Tensor],
    current: torch.Tensor,
    policy: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    model_current = apply_causal_boundary_conservative(
        current,
        policy,
        gamma=float(model.gamma),
        scope=NATIVE_CAUSAL_BOUNDARY_SCOPE,
    )
    raw = model_call(model, sample, model_current, branch_gains=None)
    deployed = apply_causal_boundary_conservative(
        raw,
        policy,
        gamma=float(model.gamma),
        scope=NATIVE_CAUSAL_BOUNDARY_SCOPE,
    )
    return model_current, raw, deployed


def retained_prefix_observations(
    prefix: np.ndarray,
    *,
    key: str,
    weights: np.ndarray,
    component_scale: np.ndarray,
    envelope: Mapping[str, Any],
    gamma: float,
    node_mask: torch.Tensor,
    state_mean: np.ndarray,
    state_scale: np.ndarray,
    device: torch.device,
) -> dict[str, Any]:
    """Measure the immutable H1--H79 source without replaying model calls."""

    if prefix.shape[0] != TRUTH_STEPS + 1:
        raise ValueError("retained-prefix continuation requires exact H0-H79")
    events: dict[str, list[bool]] = {name: [] for name in EVENT_NAMES}
    amplitude: list[float | None] = []
    scaled_rms: list[float | None] = []
    normalizer_excursion: list[float | None] = []
    internal_energy_minimum: list[float | None] = []
    state_hashes: list[str] = []
    for call in range(1, TRUTH_STEPS + 1):
        deployed_np = np.asarray(prefix[call], dtype=np.float32)
        deployed_state = torch.as_tensor(
            deployed_np[None], dtype=torch.float32, device=device
        )
        diagnosis = diagnose_euler_state(
            deployed_state,
            representation="conservative_rho_m1_m2_E",
            gamma=gamma,
            node_mask=node_mask,
        )
        amplitude_ratio, scaled_rms_ratio = common_envelope_metrics(
            deployed_np,
            weights=weights,
            component_scale=component_scale,
            envelope=envelope,
            gamma=gamma,
        )
        finite = bool(diagnosis["finite"])
        if not finite:
            raise ValueError(
                f"trajectory {key} retained prefix is nonfinite at call {call}"
            )
        events["finite"].append(True)
        events["admissible"].append(bool(diagnosis["admissible"]))
        events["bounded"].append(
            bool(
                amplitude_ratio is not None
                and scaled_rms_ratio is not None
                and amplitude_ratio <= THRESHOLDS.amplitude_ratio_max
                and scaled_rms_ratio <= THRESHOLDS.scaled_rms_ratio_max
            )
        )
        amplitude.append(amplitude_ratio)
        scaled_rms.append(scaled_rms_ratio)
        normalizer_excursion.append(
            normalizer_metrics(
                deployed_np, state_mean=state_mean, state_scale=state_scale
            )["maximum_normalizer_excursion"]
        )
        internal_energy_minimum.append(diagnosis["min_internal_energy"])
        state_hashes.append(array_sha256(deployed_np))
    return {
        "events": events,
        "amplitude": amplitude,
        "scaled_rms": scaled_rms,
        "normalizer_excursion": normalizer_excursion,
        "internal_energy_minimum": internal_energy_minimum,
        "state_hashes": state_hashes,
    }


def gradient_zero_audit(model: torch.nn.Module) -> dict[str, Any]:
    backbone = getattr(model, "backbone", model)
    rows = []
    for layer, module in enumerate(backbone.gws):
        weight = module.gw2.weight.detach()
        rows.append(
            {
                "layer": layer,
                "element_count": int(weight.numel()),
                "nonzero_count": int(torch.count_nonzero(weight).cpu()),
                "max_abs": float(torch.max(torch.abs(weight)).cpu()),
            }
        )
    if any(row["nonzero_count"] != 0 for row in rows):
        raise ValueError("PCFNO checkpoint has a nonzero serialized gw2 weight")
    return {"exact_zero_gw2": True, "layers": rows}


def accepted_prefix_event(
    pass_values: Sequence[bool | None],
    *,
    requested_horizon: int,
    event_name: str,
) -> dict[str, Any]:
    """Apply the frozen D087 first-failure and censoring semantics."""

    if event_name not in EVENT_NAMES:
        raise ValueError(f"unknown event: {event_name}")
    if requested_horizon < 1:
        raise ValueError("requested_horizon must be positive")
    values = list(pass_values)
    if len(values) > requested_horizon:
        raise ValueError("event sequence exceeds requested_horizon")
    if any(value is not None and not isinstance(value, bool) for value in values):
        raise TypeError("event values must be bool or None")
    unavailable = next((i for i, value in enumerate(values) if value is None), None)
    if unavailable is not None and any(
        value is not None for value in values[unavailable:]
    ):
        raise ValueError("unavailable event values must form a terminal suffix")
    observed = len(values) if unavailable is None else unavailable
    first_failure_offset = next(
        (i for i, value in enumerate(values[:observed]) if value is False), None
    )
    if first_failure_offset is not None:
        first_failure_call = first_failure_offset + 1
        return {
            "first_failure_call": first_failure_call,
            "accepted_prefix_calls": first_failure_call - 1,
            "right_censored": False,
            "censor_call": None,
            "censor_reason": None,
            "observed_call_count": observed,
            "recovered_after_failure": any(
                value is True for value in values[first_failure_offset + 1 : observed]
            ),
        }
    if unavailable is not None:
        censor_reason = "measurement_unavailable"
    elif observed == requested_horizon:
        censor_reason = "requested_horizon"
    else:
        censor_reason = "execution_horizon"
    return {
        "first_failure_call": None,
        "accepted_prefix_calls": observed,
        "right_censored": True,
        "censor_call": observed,
        "censor_reason": censor_reason,
        "observed_call_count": observed,
        "recovered_after_failure": False,
    }


def stage_transition_label(
    before_failure: str, after_failure: str, *, stage: str
) -> str:
    """Apply the maintained introduce, persist, and recovery labels."""

    before_valid = before_failure == "admissible"
    after_valid = after_failure == "admissible"
    if before_valid and after_valid:
        return f"{stage}_admissible"
    if before_valid and not after_valid:
        return f"{stage}_introduced_inadmissibility"
    if not before_valid and after_valid:
        return f"{stage}_recovery"
    return f"{stage}_persistent_inadmissibility"


def event_summary(values: Mapping[str, Sequence[bool]], horizon: int) -> dict[str, Any]:
    return {
        name: accepted_prefix_event(
            list(values[name]), requested_horizon=horizon, event_name=name
        )
        for name in EVENT_NAMES
    }


def finite_max(values: Sequence[float | None]) -> float | None:
    finite = [
        float(value) for value in values if value is not None and math.isfinite(value)
    ]
    return max(finite) if finite else None


def nan_if_none(value: float | None) -> float:
    return float("nan") if value is None else float(value)


def survival_rows(
    case_rows: Sequence[Mapping[str, Any]], horizon: int
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for event_name in EVENT_NAMES:
        survival = 1.0
        for call in range(1, horizon + 1):
            at_risk = 0
            failures = 0
            censored = 0
            for row in case_rows:
                event = row["events"][event_name]
                failure = event["first_failure_call"]
                censor = event["censor_call"]
                endpoint = int(failure if failure is not None else censor)
                if call <= endpoint:
                    at_risk += 1
                if failure == call:
                    failures += 1
                if failure is None and censor == call:
                    censored += 1
            if at_risk == 0:
                break
            conditional = 1.0 - failures / at_risk
            survival *= conditional
            output.append(
                {
                    "event": event_name,
                    "call": call,
                    "cohort_size": len(case_rows),
                    "at_risk_count": at_risk,
                    "failure_count": failures,
                    "censored_after_call_count": censored,
                    "conditional_survival": conditional,
                    "kaplan_meier_survival": survival,
                }
            )
    return output


def run_case(
    model: torch.nn.Module,
    store: PCNOEuler2DShardStore,
    key: str,
    *,
    horizon: int,
    checkpoint: Mapping[str, Any],
    device: torch.device,
    prefix: np.ndarray,
    prefix_record: Mapping[str, Any],
    calibration: Mapping[str, Any],
    retain_states: bool,
    resume_from_retained_prefix: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, np.ndarray] | None]:
    states = np.asarray(store.states(key), dtype=np.float32)
    initial = np.array(states[0], copy=True)
    initial_internal = internal_energy_numpy(initial)
    if not np.isfinite(initial).all() or not np.isfinite(initial_internal).all():
        raise ValueError(f"trajectory {key} has a nonfinite initial state")
    if not np.array_equal(initial, prefix[0]):
        raise ValueError(f"trajectory {key} frame-zero prefix mismatch")
    sample = store.tensor_sample(key, 0, step_stride=1, device=device)
    native = checkpoint["boundary_contract"]
    policy, policy_metadata = build_graph_causal_boundary_policy(
        store,
        key,
        device=device,
        max_source_hops=int(native["max_source_hops"]),
        rho_inf=float(native["rho_inf"]),
        p_inf=float(native["p_inf"]),
    )
    expected_policy = native.get("policy_digests", {}).get(key)
    if (
        expected_policy is not None
        and policy_metadata["policy_digest"] != expected_policy
    ):
        raise ValueError(f"trajectory {key} boundary-policy digest changed")

    weights = node_weights(store, key)
    node_type = np.asarray(store.array(key, "node_type"), dtype=np.int64).reshape(-1)
    positions = np.asarray(store.array(key, "nodes"), dtype=np.float32)
    state_scale = np.asarray(
        checkpoint["normalization"]["state_scale"], dtype=np.float64
    )
    state_mean = np.asarray(checkpoint["normalization"]["state_mean"], dtype=np.float64)
    component_scale = np.asarray(calibration["component_scale"], dtype=np.float64)
    envelope = calibration["case_envelopes"][key]
    integral_scale = np.asarray(envelope["proxy_integral_scale"], dtype=np.float64)
    initial_integral = proxy_integrals(initial, weights)
    gamma = float(model.gamma)
    node_mask = sample["node_mask"]
    rows: list[dict[str, Any]] = []
    events: dict[str, list[bool]] = {name: [] for name in EVENT_NAMES}
    terminal_call: int | None = None
    prefix_hashes: list[str] = []
    output_recoveries: list[int] = []
    model_recoveries: list[int] = []
    amplitude_history: list[float | None] = []
    scaled_rms_history: list[float | None] = []
    normalizer_excursion_history: list[float | None] = []
    internal_energy_minimum_history: list[float | None] = []

    if resume_from_retained_prefix:
        if horizon <= TRUTH_STEPS:
            raise ValueError("retained-prefix continuation must extend beyond H79")
        observations = retained_prefix_observations(
            prefix,
            key=key,
            weights=weights,
            component_scale=component_scale,
            envelope=envelope,
            gamma=gamma,
            node_mask=node_mask,
            state_mean=state_mean,
            state_scale=state_scale,
            device=device,
        )
        events = observations["events"]
        amplitude_history = observations["amplitude"]
        scaled_rms_history = observations["scaled_rms"]
        normalizer_excursion_history = observations["normalizer_excursion"]
        internal_energy_minimum_history = observations["internal_energy_minimum"]
        prefix_hashes = observations["state_hashes"]
        current = torch.as_tensor(
            prefix[TRUTH_STEPS][None], dtype=torch.float32, device=device
        )
        previous = np.array(prefix[TRUTH_STEPS], copy=True)
        retained = (
            [np.array(state, copy=True) for state in prefix] if retain_states else None
        )
        start_call = TRUTH_STEPS + 1
    else:
        current = torch.tensor(initial[None], dtype=torch.float32, device=device)
        previous = initial
        retained = [initial] if retain_states else None
        start_call = 1

    for call in range(start_call, horizon + 1):
        model_current, raw, deployed = contract_call(model, sample, current, policy)
        synchronize(device)
        diagnoses = {
            "current": diagnose_euler_state(
                current,
                representation="conservative_rho_m1_m2_E",
                gamma=gamma,
                node_mask=node_mask,
            ),
            "model_input": diagnose_euler_state(
                model_current,
                representation="conservative_rho_m1_m2_E",
                gamma=gamma,
                node_mask=node_mask,
            ),
            "raw": diagnose_euler_state(
                raw,
                representation="conservative_rho_m1_m2_E",
                gamma=gamma,
                node_mask=node_mask,
            ),
            "deployed": diagnose_euler_state(
                deployed,
                representation="conservative_rho_m1_m2_E",
                gamma=gamma,
                node_mask=node_mask,
            ),
        }
        raw_np = raw[0].detach().float().cpu().numpy().copy()
        deployed_np = deployed[0].detach().float().cpu().numpy().copy()
        model_current_np = model_current[0].detach().float().cpu().numpy().copy()
        deployed_digest = array_sha256(deployed_np)
        prefix_equal: bool | None = None
        if call <= TRUTH_STEPS:
            prefix_equal = bool(np.array_equal(deployed_np, prefix[call]))
            if not prefix_equal:
                difference = float(np.max(np.abs(deployed_np - prefix[call])))
                raise ValueError(
                    f"trajectory {key} call {call} bitwise prefix mismatch; "
                    f"max_abs={difference}"
                )
            prefix_hashes.append(deployed_digest)

        amplitude_ratio, scaled_rms_ratio = common_envelope_metrics(
            deployed_np,
            weights=weights,
            component_scale=component_scale,
            envelope=envelope,
            gamma=gamma,
        )
        finite = bool(diagnoses["deployed"]["finite"])
        admissible = bool(diagnoses["deployed"]["admissible"])
        bounded = bool(
            finite
            and amplitude_ratio is not None
            and scaled_rms_ratio is not None
            and amplitude_ratio <= THRESHOLDS.amplitude_ratio_max
            and scaled_rms_ratio <= THRESHOLDS.scaled_rms_ratio_max
        )
        events["finite"].append(finite)
        events["admissible"].append(admissible)
        events["bounded"].append(bounded)
        amplitude_history.append(amplitude_ratio)
        scaled_rms_history.append(scaled_rms_ratio)
        input_transition = transition(
            diagnoses["current"], diagnoses["model_input"], "input_boundary"
        )
        model_transition = transition(
            diagnoses["model_input"], diagnoses["raw"], "model"
        )
        output_transition = transition(
            diagnoses["raw"], diagnoses["deployed"], "output_boundary"
        )
        if model_transition == "model_recovery":
            model_recoveries.append(call)
        if output_transition == "output_boundary_recovery":
            output_recoveries.append(call)

        integral = proxy_integrals(deployed_np, weights)
        integral_drift = (integral - initial_integral) / integral_scale
        normalizer = normalizer_metrics(
            deployed_np, state_mean=state_mean, state_scale=state_scale
        )
        normalizer_excursion_history.append(normalizer["maximum_normalizer_excursion"])
        internal_energy_minimum_history.append(
            diagnoses["deployed"]["min_internal_energy"]
        )
        row: dict[str, Any] = {
            "trajectory": key,
            "call": call,
            "physical_time": float(call * float(store.manifest["dt"])),
            "truth_available_for_accuracy": False,
            "prefix_expected": call <= TRUTH_STEPS,
            "prefix_bitwise_equal": prefix_equal,
            "deployed_state_sha256": deployed_digest,
            "event_admissible": admissible,
            "event_bounded": bounded,
            "event_finite": finite,
            "common_amplitude_ratio": amplitude_ratio,
            "common_scaled_rms_ratio": scaled_rms_ratio,
            "input_transition": input_transition,
            "model_transition": model_transition,
            "output_transition": output_transition,
            "input_boundary_correction_scaled_rms": weighted_scaled_rms(
                model_current_np - previous, weights, state_scale
            ),
            "output_boundary_correction_scaled_rms": weighted_scaled_rms(
                deployed_np - raw_np, weights, state_scale
            ),
            "deployed_increment_scaled_rms": weighted_scaled_rms(
                deployed_np - previous, weights, state_scale
            ),
            "proxy_integral_density": integral[0],
            "proxy_integral_momentum_x": integral[1],
            "proxy_integral_momentum_y": integral[2],
            "proxy_integral_energy": integral[3],
            "proxy_integral_density_drift_scaled": integral_drift[0],
            "proxy_integral_momentum_x_drift_scaled": integral_drift[1],
            "proxy_integral_momentum_y_drift_scaled": integral_drift[2],
            "proxy_integral_energy_drift_scaled": integral_drift[3],
            **stage_fields("current", diagnoses["current"]),
            **stage_fields("model_input", diagnoses["model_input"]),
            **stage_fields("raw", diagnoses["raw"]),
            **stage_fields("deployed", diagnoses["deployed"]),
            **invalid_type_counts(deployed_np, node_type, gamma=gamma),
            **normalizer,
        }
        rows.append(row)
        if retained is not None:
            retained.append(deployed_np)
        previous = deployed_np
        if not finite:
            terminal_call = call
            break
        current = deployed

    recorded = start_call - 1 + len(rows)
    summaries = event_summary(events, horizon)
    case_summary = {
        "trajectory": key,
        "requested_horizon": horizon,
        "recorded_call_count": recorded,
        "termination_call": terminal_call,
        "terminal_call": recorded,
        "terminal_events": {name: bool(events[name][-1]) for name in EVENT_NAMES},
        "events": summaries,
        "model_recovery_calls": model_recoveries,
        "output_boundary_recovery_calls": output_recoveries,
        "prefix": {
            **dict(prefix_record),
            "frame_zero_bitwise_equal": True,
            "checked_calls": 0
            if resume_from_retained_prefix
            else min(recorded, TRUTH_STEPS),
            "all_checked_calls_bitwise_equal": None
            if resume_from_retained_prefix
            else True,
            "source_bound": resume_from_retained_prefix,
            "loaded_retained_calls": TRUTH_STEPS if resume_from_retained_prefix else 0,
            "resume_call": TRUTH_STEPS + 1 if resume_from_retained_prefix else None,
            "digest_sequence_kind": "retained_source"
            if resume_from_retained_prefix
            else "replayed_deployed_state",
            "deployed_digest_sequence_sha256": mapping_sha256(prefix_hashes),
            "resume_state_sha256": array_sha256(prefix[TRUTH_STEPS])
            if resume_from_retained_prefix
            else None,
        },
        "boundary_policy_digest": policy_metadata["policy_digest"],
        "terminal_common_amplitude_ratio": rows[-1]["common_amplitude_ratio"],
        "terminal_common_scaled_rms_ratio": rows[-1]["common_scaled_rms_ratio"],
        "terminal_min_density": rows[-1]["deployed_min_density"],
        "terminal_min_internal_energy": rows[-1]["deployed_min_internal_energy"],
        "terminal_min_pressure": rows[-1]["deployed_min_pressure"],
        "maximum_common_amplitude_ratio": finite_max(amplitude_history),
        "maximum_common_scaled_rms_ratio": finite_max(scaled_rms_history),
        "maximum_normalizer_excursion": finite_max(normalizer_excursion_history),
        "claim_boundary": (
            "reference-free descriptive PCFNO recurrence; not accuracy, physical "
            "validity, conservation, or asymptotic stability"
        ),
    }
    bundle = None
    if retained is not None:
        retained_array = np.stack(retained, axis=0).astype(np.float32, copy=False)
        bundle = {
            "schema": np.asarray(ANIMATION_SCHEMA),
            "working_id": np.asarray(WORKING_ID),
            "trajectory": np.asarray(key),
            "checkpoint_sha256": np.asarray(CHECKPOINT_SHA256),
            "requested_horizon": np.asarray(horizon, dtype=np.int64),
            "recorded_call_count": np.asarray(recorded, dtype=np.int64),
            "termination_call": np.asarray(
                -1 if terminal_call is None else terminal_call, dtype=np.int64
            ),
            "physical_times": np.arange(recorded + 1, dtype=np.float64)
            * float(store.manifest["dt"]),
            "positions": positions,
            "node_type": node_type.astype(np.int64, copy=False),
            "deployed_states_conservative": retained_array,
            "state_scale": state_scale.astype(np.float64, copy=False),
            "presentation_scales_json": np.asarray(
                json.dumps(calibration["animation_scales"][key], sort_keys=True)
            ),
            "event_admissible": np.asarray(
                [True, *events["admissible"]], dtype=np.bool_
            ),
            "event_bounded": np.asarray([True, *events["bounded"]], dtype=np.bool_),
            "event_finite": np.asarray([True, *events["finite"]], dtype=np.bool_),
            "common_amplitude_ratio": np.asarray(
                [1.0, *[nan_if_none(value) for value in amplitude_history]],
                dtype=np.float64,
            ),
            "common_scaled_rms_ratio": np.asarray(
                [1.0, *[nan_if_none(value) for value in scaled_rms_history]],
                dtype=np.float64,
            ),
            "minimum_internal_energy": np.asarray(
                [
                    float(np.min(initial_internal)),
                    *[nan_if_none(value) for value in internal_energy_minimum_history],
                ],
                dtype=np.float64,
            ),
        }
    return rows, case_summary, bundle


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    horizon = 2 if args.mode == "smoke" else PRODUCTION_STEPS
    keys = ("187",) if args.mode == "smoke" else TRAJECTORY_KEYS
    retained_keys = ("187",) if args.mode == "smoke" else ANIMATION_KEYS
    device = select_device(args.device)
    if args.mode == "production" and device.type != "cuda":
        raise RuntimeError("registered production H320 requires CUDA")

    frozen_sources = verify_frozen_b1_sources()
    checkpoint_hash = sha256_file(args.checkpoint)
    if checkpoint_hash != CHECKPOINT_SHA256:
        raise ValueError("checkpoint SHA-256 differs from the registered PCFNO")
    checkpoint = load_checkpoint(args.checkpoint)
    if checkpoint_differential_branch_mode(checkpoint) != "no_gradient":
        raise ValueError("checkpoint is not the registered PCFNO mode")
    if int(checkpoint["step_stride"]) != 1:
        raise ValueError("registered PCFNO requires step_stride=1")

    training_store = PCNOEuler2DShardStore(args.training_data_dir)
    test_store = PCNOEuler2DShardStore(args.data_dir)
    if training_store.manifest_digest != B1_DATA_MANIFEST_SHA256:
        raise ValueError("training data manifest differs from registered B1")
    missing = sorted(set(TRAJECTORY_KEYS) - set(test_store.keys))
    if missing:
        raise KeyError(f"test store lacks registered trajectories: {missing}")
    preprocessing = preprocessing_contract_audit(checkpoint, training_store, test_store)
    prefix_contract = verify_prefix_roots(
        args.strict_rollout_dir, args.continuation_dir
    )
    model = build_model(checkpoint, device)
    model.eval()
    gradient_audit = gradient_zero_audit(model)
    state_scale = np.asarray(
        checkpoint["normalization"]["state_scale"], dtype=np.float64
    )
    calibration = build_calibration_contract(
        test_store,
        TRAJECTORY_KEYS,
        gamma=float(model.gamma),
        state_scale=state_scale,
    )
    sources = source_hashes()

    args.output_dir.mkdir(parents=True)
    run_contract = {
        "schema": SCHEMA,
        "working_id": WORKING_ID,
        "status": "running",
        "mode": args.mode,
        "requested_horizon": horizon,
        "trajectory_keys": list(keys),
        "full_registered_population": list(TRAJECTORY_KEYS),
        "animation_keys": list(retained_keys),
        "precision": "fp32_batch_one",
        "accuracy_evaluated": False,
        "truth_free_calls": [80, PRODUCTION_STEPS],
        "recurrence": (
            "fresh frame zero; input causal boundary; PCFNO; output causal boundary; "
            "deployed feedback; finite invalid continues; returned nonfinite stops"
        ),
        "checkpoint": {
            "sha256": checkpoint_hash,
            "differential_branch_mode": "no_gradient",
        },
        "data": {
            "training_manifest_digest": training_store.manifest_digest,
            "evaluation_manifest_digest": test_store.manifest_digest,
        },
        "frozen_b1_source_set_sha256": B1_SOURCE_SET_SHA256,
        "frozen_b1_sources": frozen_sources,
        "extension_source_sha256": sources,
        "preprocessing": preprocessing,
        "prefix_contract": prefix_contract,
        "gradient_audit": gradient_audit,
        "thresholds": THRESHOLDS.as_dict(),
        "calibration_payload_sha256": calibration["payload_sha256"],
    }
    write_json(args.output_dir / "run_contract.json", run_contract)
    write_json(args.output_dir / "calibration_contract.json", calibration)

    all_rows: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    for key in keys:
        prefix, prefix_record = expected_prefix(
            key,
            strict_root=args.strict_rollout_dir,
            continuation_root=args.continuation_dir,
            continuation_output_sha256=prefix_contract["continuation_output_sha256"],
        )
        rows, case, bundle = run_case(
            model,
            test_store,
            key,
            horizon=horizon,
            checkpoint=checkpoint,
            device=device,
            prefix=prefix,
            prefix_record=prefix_record,
            calibration=calibration,
            retain_states=key in retained_keys,
        )
        all_rows.extend(rows)
        cases.append(case)
        if bundle is not None:
            write_npz_atomic(args.output_dir / f"trajectory_{key}.npz", bundle)
        print(
            json.dumps(
                {
                    "trajectory": key,
                    "recorded_calls": case["recorded_call_count"],
                    "terminal_events": case["terminal_events"],
                    "first_failures": {
                        name: case["events"][name]["first_failure_call"]
                        for name in EVENT_NAMES
                    },
                },
                sort_keys=True,
            ),
            flush=True,
        )

    case_rows: list[dict[str, Any]] = []
    for case in cases:
        row: dict[str, Any] = {
            "trajectory": case["trajectory"],
            "recorded_call_count": case["recorded_call_count"],
            "termination_call": case["termination_call"],
            "terminal_admissible": case["terminal_events"]["admissible"],
            "terminal_bounded": case["terminal_events"]["bounded"],
            "terminal_finite": case["terminal_events"]["finite"],
            "terminal_common_amplitude_ratio": case["terminal_common_amplitude_ratio"],
            "terminal_common_scaled_rms_ratio": case[
                "terminal_common_scaled_rms_ratio"
            ],
            "terminal_min_density": case["terminal_min_density"],
            "terminal_min_internal_energy": case["terminal_min_internal_energy"],
            "terminal_min_pressure": case["terminal_min_pressure"],
            "maximum_common_amplitude_ratio": case["maximum_common_amplitude_ratio"],
            "maximum_common_scaled_rms_ratio": case["maximum_common_scaled_rms_ratio"],
            "maximum_normalizer_excursion": case["maximum_normalizer_excursion"],
            "model_recovery_calls": case["model_recovery_calls"],
            "output_boundary_recovery_calls": case["output_boundary_recovery_calls"],
            "prefix_calls_checked": case["prefix"]["checked_calls"],
            "prefix_bitwise_equal": case["prefix"]["all_checked_calls_bitwise_equal"],
        }
        for event_name in EVENT_NAMES:
            event = case["events"][event_name]
            row[f"{event_name}_first_failure_call"] = event["first_failure_call"]
            row[f"{event_name}_accepted_prefix_calls"] = event["accepted_prefix_calls"]
            row[f"{event_name}_recovered_after_failure"] = event[
                "recovered_after_failure"
            ]
        case_rows.append(row)

    survival = survival_rows(cases, horizon)
    write_csv(args.output_dir / "call_metrics.csv", all_rows)
    write_csv(args.output_dir / "case_metrics.csv", case_rows)
    write_csv(args.output_dir / "survival.csv", survival)
    summary = {
        "schema": SCHEMA,
        "working_id": WORKING_ID,
        "status": "completed",
        "mode": args.mode,
        "requested_horizon": horizon,
        "trajectory_count": len(cases),
        "accuracy_evaluated": False,
        "terminal_counts": {
            name: sum(bool(case["terminal_events"][name]) for case in cases)
            for name in EVENT_NAMES
        },
        "ever_failed_counts": {
            name: sum(
                case["events"][name]["first_failure_call"] is not None for case in cases
            )
            for name in EVENT_NAMES
        },
        "all_prefixes_bitwise_equal": all(
            case["prefix"]["all_checked_calls_bitwise_equal"] for case in cases
        ),
        "cases": {case["trajectory"]: case for case in cases},
        "claim_boundary": (
            "H80-H320 is reference-free descriptive recurrence only; no accuracy, "
            "defect, physical conservation, physical validity, general gradient-"
            "ablation, or asymptotic-stability claim"
        ),
    }
    write_json(args.output_dir / "summary.json", summary)
    run_contract["status"] = "completed"
    write_json(args.output_dir / "run_contract.json", run_contract)

    artifact_names = [
        "run_contract.json",
        "calibration_contract.json",
        "call_metrics.csv",
        "case_metrics.csv",
        "survival.csv",
        "summary.json",
        *[f"trajectory_{key}.npz" for key in retained_keys],
    ]
    manifest = build_final_hash_manifest(args.output_dir, artifact_names)
    write_json(args.output_dir / "final_hash_manifest.json", manifest)
    verification = verify_final_hash_manifest(args.output_dir, manifest)
    summary["final_hash_manifest"] = {
        "sha256": sha256_file(args.output_dir / "final_hash_manifest.json"),
        **verification,
    }
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run(args)
    print(json.dumps(_json_safe(summary), sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
