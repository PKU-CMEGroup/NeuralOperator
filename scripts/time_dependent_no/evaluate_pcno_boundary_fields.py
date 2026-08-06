#!/usr/bin/env python3
"""Evaluate matched D072 boundary-field models on frozen open populations.

The evaluator keeps the deployed physical boundary policy fixed.  It compares
the N0/G1/S1 training arms, applies frozen zero-field interventions to G1/S1,
checks that read-only hooks do not perturb inference, and exports all-frame
rollout bundles for outcome-independent representative cases.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utility.time_dependent_no.euler2d_metrics import shock_indicator
from utility.time_dependent_no.pcno_artifacts import (
    atomic_write_json_with_paths as atomic_write_json,
)
from utility.time_dependent_no.pcno_artifacts import (
    git_state,
    jsonable_args,
    runtime_environment,
    sha256_file,
)
from utility.time_dependent_no.pcno_artifacts import (
    write_csv_with_paths as write_csv,
)
from utility.time_dependent_no.pcno_euler2d import (
    PCNOEuler2DResidual,
    PCNOEuler2DShardStore,
    build_graph_causal_boundary_policy,
    conservative_admissibility,
)
from utility.time_dependent_no.pcno_node_type_interpretability import (
    PCNOActivationRecorder,
    activation_differences,
    tensor_equivalence_metrics,
)
from utility.time_dependent_no.pcno_residual_structure import shock_vortex_regions
from utility.time_dependent_no.pcno_ripple_diagnostics import (
    conservative_to_primitive_raw,
    graph_distance_to_mask,
    node_highpass_field,
)
from utility.time_dependent_no.pcno_rollout import (
    boundary_outflow_normal_mach,
    close_boundary,
    endpoint_diagnostics,
)
from utility.time_dependent_no.pcno_runtime import (
    autocast_context,
    build_checkpoint_model,
    load_checkpoint,
    select_device,
    synchronize,
)

SCHEMA = "pcno_boundary_field_evaluation_v1"
ARMS = ("N0", "G1", "S1")
MODES = ("teacher_forced", "free_rollout")
RUN_PATTERN = re.compile(r"^(?P<family>dynamic|bump)_s(?P<seed>\d+)_(?P<arm>N0|G1|S1)$")
PHYSICAL_WAVELENGTH_BANDS = ((0.05, 0.125), (0.125, 0.25))

FAMILY_CONTRACTS: dict[str, dict[str, Any]] = {
    "dynamic_fv": {
        "directory_prefix": "dynamic",
        "step_stride": 2,
        "start_frame": 0,
        "rollout_calls": 30,
        "endpoints": (20, 30),
        "boundary_mode": "model_all_nodes",
        "raw_recurrence": True,
        "input_dims": {"N0": 8, "G1": 9, "S1": 10},
        "semantic_names": ("y_symmetry", "x_extrapolation"),
        "hook_abs_limit": 2.0e-5,
        "hook_rel_limit": 1.0e-7,
    },
    "bump": {
        "directory_prefix": "bump",
        "step_stride": 1,
        "start_frame": 0,
        "rollout_calls": 79,
        "endpoints": (20, 40, 60, 79),
        "boundary_mode": "causal_nodal_physical",
        "raw_recurrence": False,
        "input_dims": {"N0": 8, "G1": 9, "S1": 11},
        "semantic_names": ("wall", "outflow", "inflow"),
        "hook_abs_limit": 2.0e-3,
        "hook_rel_limit": 1.0e-5,
    },
}


@dataclass(frozen=True)
class Variant:
    arm: str
    intervention: str
    zero_field_indices: tuple[int, ...] = ()

    @property
    def name(self) -> str:
        return f"{self.arm}_{self.intervention}"


@dataclass
class LoadedArm:
    arm: str
    run_dir: Path
    summary: dict[str, Any]
    split: dict[str, Any]
    run_contract: dict[str, Any]
    checkpoint: dict[str, Any]
    model: PCNOEuler2DResidual
    checkpoint_sha256: str


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=tuple(FAMILY_CONTRACTS), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case-ids", nargs="*")
    parser.add_argument("--trace-cases", nargs="+", required=True)
    parser.add_argument("--trace-calls", nargs="+", type=int, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--amp", choices=("none", "bf16", "fp16"), default="none")
    parser.add_argument("--shock-quantile", type=float, default=0.9)
    parser.add_argument("--animation-max-nodes", type=int, default=25_000)
    parser.add_argument("--expected-data-manifest-digest", required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.seed < 0:
        raise ValueError("--seed must be nonnegative")
    if not 0.0 < args.shock_quantile < 1.0:
        raise ValueError("--shock-quantile must lie in (0,1)")
    if args.animation_max_nodes < 1:
        raise ValueError("--animation-max-nodes must be positive")
    contract = FAMILY_CONTRACTS[args.family]
    invalid_calls = [
        call
        for call in args.trace_calls
        if call < 1 or call > contract["rollout_calls"]
    ]
    if invalid_calls:
        raise ValueError(f"trace calls lie outside the rollout: {invalid_calls}")
    digest = args.expected_data_manifest_digest.lower()
    if len(digest) != 64 or any(value not in "0123456789abcdef" for value in digest):
        raise ValueError("--expected-data-manifest-digest must be a SHA-256")
    return args


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError(f"expected a JSON mapping: {path}")
    return dict(value)


def _require_equal(actual: Any, expected: Any, name: str) -> None:
    if actual != expected:
        raise ValueError(f"{name} differs: {actual!r} != {expected!r}")


def _run_dir(root: Path, family: str, seed: int, arm: str) -> Path:
    prefix = FAMILY_CONTRACTS[family]["directory_prefix"]
    result = root / f"{prefix}_s{seed}_{arm}"
    if not result.is_dir():
        raise FileNotFoundError(result)
    match = RUN_PATTERN.fullmatch(result.name)
    if match is None:
        raise ValueError(f"unexpected run-directory name: {result.name}")
    return result


def _load_arms(args: argparse.Namespace, device: torch.device) -> dict[str, LoadedArm]:
    contract = FAMILY_CONTRACTS[args.family]
    result: dict[str, LoadedArm] = {}
    reference_split: dict[str, Any] | None = None
    reference_normalization: str | None = None
    reference_stream: str | None = None
    for arm in ARMS:
        run_dir = _run_dir(args.run_root, args.family, args.seed, arm)
        summary = _read_json(run_dir / "summary.json")
        split = _read_json(run_dir / "split.json")
        run_contract = _read_json(run_dir / "run_contract.json")
        checkpoint_path = run_dir / "best.pt"
        checkpoint_sha256 = sha256_file(checkpoint_path)
        _require_equal(
            checkpoint_sha256,
            summary["artifact_sha256"]["best_checkpoint"],
            f"{run_dir.name} best-checkpoint digest",
        )
        _require_equal(
            summary["data_manifest_digest"],
            args.expected_data_manifest_digest,
            f"{run_dir.name} data manifest",
        )
        _require_equal(split.get("test_keys", []), [], f"{run_dir.name} sealed split")
        _require_equal(
            summary["model_config"]["in_dim"],
            contract["input_dims"][arm],
            f"{run_dir.name} input dimension",
        )
        _require_equal(
            summary["boundary_mode"],
            contract["boundary_mode"],
            f"{run_dir.name} physical boundary policy",
        )
        _require_equal(
            summary["raw_recurrence"],
            contract["raw_recurrence"],
            f"{run_dir.name} recurrence contract",
        )
        training_args = run_contract["args"]
        for key, expected in (
            ("step_stride", contract["step_stride"]),
            ("rollout_start_frame", contract["start_frame"]),
            ("rollout_steps", contract["rollout_calls"]),
        ):
            _require_equal(training_args[key], expected, f"{run_dir.name} {key}")
        _require_equal(
            tuple(training_args["rollout_checkpoints"]),
            contract["endpoints"],
            f"{run_dir.name} endpoint contract",
        )
        checkpoint = load_checkpoint(checkpoint_path)
        _require_equal(
            checkpoint["data_manifest_digest"],
            args.expected_data_manifest_digest,
            f"{run_dir.name} checkpoint data manifest",
        )
        model, _ = build_checkpoint_model(
            checkpoint,
            device,
            model_node_type_input=str(checkpoint["model_node_type_input"]),
        )
        if reference_split is None:
            reference_split = split
            reference_normalization = str(summary["normalization_digest"])
            reference_stream = str(
                run_contract["exposure_contract"]["presentation_stream_sha256"]
            )
        else:
            _require_equal(split, reference_split, "matched split")
            _require_equal(
                summary["normalization_digest"],
                reference_normalization,
                "matched normalization",
            )
            _require_equal(
                run_contract["exposure_contract"]["presentation_stream_sha256"],
                reference_stream,
                "matched presentation stream",
            )
        result[arm] = LoadedArm(
            arm=arm,
            run_dir=run_dir,
            summary=summary,
            split=split,
            run_contract=run_contract,
            checkpoint=checkpoint,
            model=model,
            checkpoint_sha256=checkpoint_sha256,
        )
    return result


def _verify_training_source_snapshot(arms: Mapping[str, LoadedArm]) -> dict[str, Any]:
    """Verify every retained training/provenance file in the evaluation source."""

    snapshots = [loaded.run_contract["source_snapshot"] for loaded in arms.values()]
    reference = snapshots[0]
    for snapshot in snapshots[1:]:
        _require_equal(snapshot, reference, "matched training source snapshot")
    verified: dict[str, dict[str, Any]] = {}
    for category in ("files", "provenance_files"):
        entries = reference.get(category)
        if not isinstance(entries, Mapping) or not entries:
            raise ValueError(f"training source snapshot lacks {category}")
        for relative_name, metadata in entries.items():
            path = ROOT / str(relative_name)
            if not path.is_file():
                raise FileNotFoundError(path)
            actual_sha256 = sha256_file(path)
            _require_equal(
                actual_sha256,
                metadata["sha256"],
                f"training source {relative_name}",
            )
            _require_equal(
                path.stat().st_size,
                int(metadata["bytes"]),
                f"training source byte count {relative_name}",
            )
            verified[str(relative_name)] = {
                "sha256": actual_sha256,
                "bytes": path.stat().st_size,
                "category": category,
            }
    return {
        "schema": reference["schema"],
        "source_set_digest": reference["source_set_digest"],
        "provenance_set_digest": reference["provenance_set_digest"],
        "verified_files": verified,
        "evaluator": {
            "path": str(Path(__file__).resolve().relative_to(ROOT)),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
    }


def _variants(semantic_names: Sequence[str]) -> tuple[Variant, ...]:
    result = [
        Variant("N0", "correct"),
        Variant("G1", "correct"),
        Variant("G1", "zero_all", tuple(range(len(semantic_names)))),
        Variant("S1", "correct"),
        Variant("S1", "zero_all", tuple(range(len(semantic_names)))),
    ]
    result.extend(
        Variant("S1", f"zero_{name}", (index,))
        for index, name in enumerate(semantic_names)
    )
    return tuple(result)


def _features_for_variant(
    sample: Mapping[str, torch.Tensor], variant: Variant
) -> torch.Tensor | None:
    fields = sample.get("boundary_features")
    if variant.arm == "N0":
        return None
    if fields is None:
        raise ValueError(f"{variant.arm} requires boundary features")
    if not variant.zero_field_indices:
        return fields
    result = fields.clone()
    result[..., list(variant.zero_field_indices)] = 0.0
    return result


@torch.inference_mode()
def _model_call(
    model: PCNOEuler2DResidual,
    sample: Mapping[str, torch.Tensor],
    current: np.ndarray | torch.Tensor,
    *,
    boundary_features: torch.Tensor | None,
    boundary_policy: Mapping[str, Any] | None,
    device: torch.device,
    amp: str,
    trace: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor] | None]:
    if isinstance(current, np.ndarray):
        current_tensor = torch.as_tensor(
            np.asarray(current, dtype=np.float32), dtype=torch.float32, device=device
        ).unsqueeze(0)
    else:
        current_tensor = current
    model_current = close_boundary(current_tensor, boundary_policy, gamma=model.gamma)
    recorder = PCNOActivationRecorder(model.backbone) if trace else None
    synchronize(device)
    with autocast_context(device, amp):
        if recorder is None:
            raw_prediction = model(
                model_current,
                node_mask=sample["node_mask"],
                nodes=sample["nodes"],
                node_weights=sample["node_weights"],
                node_rhos=sample["node_rhos"],
                directed_edges=sample["directed_edges"],
                edge_gradient_weights=sample["edge_gradient_weights"],
                node_type=sample["node_type"],
                mach=sample["mach"],
                boundary_features=boundary_features,
            )
        else:
            with recorder:
                raw_prediction = model(
                    model_current,
                    node_mask=sample["node_mask"],
                    nodes=sample["nodes"],
                    node_weights=sample["node_weights"],
                    node_rhos=sample["node_rhos"],
                    directed_edges=sample["directed_edges"],
                    edge_gradient_weights=sample["edge_gradient_weights"],
                    node_type=sample["node_type"],
                    mach=sample["mach"],
                    boundary_features=boundary_features,
                )
    prediction = close_boundary(raw_prediction, boundary_policy, gamma=model.gamma)
    synchronize(device)
    snapshot = None if recorder is None else recorder.snapshot(device="cpu")
    return prediction, raw_prediction, model_current, snapshot


def _passes_equivalence(
    metrics: Mapping[str, Any], *, absolute_limit: float, relative_limit: float
) -> bool:
    return (
        bool(metrics["same_shape"])
        and bool(metrics["same_dtype"])
        and float(metrics["max_abs"]) <= absolute_limit
        and float(metrics["relative_l2"]) <= relative_limit
    )


def _hook_equivalence(
    arm: LoadedArm,
    sample: Mapping[str, torch.Tensor],
    current: np.ndarray,
    *,
    boundary_policy: Mapping[str, Any] | None,
    device: torch.device,
    amp: str,
    absolute_limit: float,
    relative_limit: float,
) -> dict[str, Any]:
    variant = Variant(arm.arm, "correct")
    fields = _features_for_variant(sample, variant)
    calls = []
    traces = []
    for trace in (False, True, True, False):
        prediction, _, _, snapshot = _model_call(
            arm.model,
            sample,
            current,
            boundary_features=fields,
            boundary_policy=boundary_policy,
            device=device,
            amp=amp,
            trace=trace,
        )
        calls.append(prediction)
        traces.append(snapshot)
    comparisons = {
        "native_repeat": tensor_equivalence_metrics(calls[0], calls[3]),
        "unhooked0_hooked0": tensor_equivalence_metrics(calls[0], calls[1]),
        "hooked0_hooked1": tensor_equivalence_metrics(calls[1], calls[2]),
        "hooked1_unhooked1": tensor_equivalence_metrics(calls[2], calls[3]),
    }
    passed = all(
        _passes_equivalence(
            value, absolute_limit=absolute_limit, relative_limit=relative_limit
        )
        for value in comparisons.values()
    )
    if device.type == "cpu":
        passed = passed and all(
            bool(value["exact_equal"]) for value in comparisons.values()
        )
    result = {
        "arm": arm.arm,
        "amp": amp,
        "absolute_limit": absolute_limit,
        "relative_limit": relative_limit,
        "comparisons": comparisons,
        "passed": passed,
    }
    if traces[1] is None or traces[2] is None:
        raise RuntimeError("hook-equivalence calls did not record activations")
    if not passed:
        raise RuntimeError(f"hook equivalence failed: {json.dumps(result)}")
    return result


def _weights(store: PCNOEuler2DShardStore, key: str) -> np.ndarray:
    value = np.array(store.array(key, "node_weights"), copy=True, dtype=np.float64)
    if value.ndim != 2:
        raise ValueError("node weights must have shape [N,M]")
    result = value.sum(axis=-1)
    if not np.all(np.isfinite(result)) or np.any(result < 0.0) or result.sum() <= 0.0:
        raise ValueError("node weights must define positive finite mass")
    return result / result.sum()


def _dynamic_resolution(nodes: np.ndarray) -> tuple[int, int]:
    positions = np.asarray(nodes, dtype=np.float64)
    nx = int(np.unique(positions[:, 0]).size)
    ny = int(np.unique(positions[:, 1]).size)
    if nx * ny != positions.shape[0]:
        raise ValueError("dynamic FV nodes do not form a tensor-product grid")
    expected = positions.reshape(ny, nx, 2)
    if not (
        np.allclose(expected[:, :, 0], expected[0, :, 0][None, :])
        and np.allclose(expected[:, :, 1], expected[:, 0, 1][:, None])
    ):
        raise ValueError("dynamic FV nodes are not stored in row-major grid order")
    return nx, ny


def _boundary_distance(
    family: str, nodes: np.ndarray, edges: np.ndarray, node_type: np.ndarray
) -> np.ndarray:
    if family == "dynamic_fv":
        positions = np.asarray(nodes, dtype=np.float64)
        return np.minimum.reduce(
            (
                positions[:, 0] - 0.0,
                2.0 - positions[:, 0],
                positions[:, 1] - 0.0,
                1.0 - positions[:, 1],
            )
        )
    return graph_distance_to_mask(nodes, edges, np.asarray(node_type) != 0)


def _region_masks(
    family: str,
    reference: np.ndarray,
    *,
    nodes: np.ndarray,
    edges: np.ndarray,
    node_type: np.ndarray,
    boundary_distance: np.ndarray,
    resolution: tuple[int, int] | None,
    gamma: float,
    shock_quantile: float,
) -> dict[str, np.ndarray]:
    if family == "dynamic_fv":
        if resolution is None:
            raise ValueError("dynamic FV region masks require a grid resolution")
        masks, _, _ = shock_vortex_regions(
            reference,
            nodes,
            resolution=resolution,
            gamma=gamma,
            boundary_width=0.05,
            shock_core_width=0.02,
            shock_envelope_width=0.05,
            vortex_radius=0.18,
        )
        return {
            "all": np.ones(nodes.shape[0], dtype=bool),
            "boundary_nodes": np.asarray(node_type).reshape(-1) != 0,
            "boundary_distance_le_0.02": masks["boundary_le_0.02"],
            "boundary_distance_le_0.05": masks["boundary_le_0.05"],
            "shock": masks["partition_shock"],
            "vortex": masks["partition_vortex"],
            "smooth": masks["partition_smooth"],
        }
    primitive = conservative_to_primitive_raw(reference, gamma=gamma)
    shock_seed = shock_indicator(primitive, edges, quantile=shock_quantile)
    shock_distance = graph_distance_to_mask(nodes, edges, shock_seed)
    boundary_005 = boundary_distance <= 0.05
    boundary_010 = boundary_distance <= 0.10
    shock = (shock_distance <= 0.05) & ~boundary_010
    return {
        "all": np.ones(nodes.shape[0], dtype=bool),
        "boundary_nodes": np.asarray(node_type).reshape(-1) != 0,
        "boundary_distance_le_0.05": boundary_005,
        "boundary_distance_le_0.10": boundary_010,
        "shock": shock,
        "smooth": ~(boundary_010 | shock),
    }


def _scaled_rms(
    value: np.ndarray,
    *,
    weights: np.ndarray,
    scale: np.ndarray,
    mask: np.ndarray,
) -> float | None:
    selected = np.asarray(mask, dtype=bool)
    if not np.any(selected):
        return None
    mass = np.asarray(weights, dtype=np.float64)[selected]
    if mass.sum() <= 0.0:
        return None
    scaled = np.asarray(value, dtype=np.float64)[selected] / np.asarray(scale)[None, :]
    return float(np.sqrt(np.einsum("n,nc,nc->", mass, scaled, scaled) / mass.sum()))


def _scaled_relative_l2(
    prediction: np.ndarray,
    reference: np.ndarray,
    *,
    weights: np.ndarray,
    scale: np.ndarray,
    mask: np.ndarray,
) -> float | None:
    selected = np.asarray(mask, dtype=bool)
    if not np.any(selected):
        return None
    mass = np.asarray(weights, dtype=np.float64)[selected]
    error = (
        np.asarray(prediction, dtype=np.float64)[selected]
        - np.asarray(reference, dtype=np.float64)[selected]
    ) / np.asarray(scale)[None, :]
    target = (
        np.asarray(reference, dtype=np.float64)[selected] / np.asarray(scale)[None, :]
    )
    numerator = float(np.einsum("n,nc,nc->", mass, error, error))
    denominator = float(np.einsum("n,nc,nc->", mass, target, target))
    return None if denominator <= 1.0e-30 else float(np.sqrt(numerator / denominator))


def _physical_wavelength_metrics(
    prediction: np.ndarray,
    reference: np.ndarray,
    *,
    resolution: tuple[int, int],
    component_scale: np.ndarray,
    wavelength_min: float,
    wavelength_max: float,
) -> dict[str, Any]:
    nx, ny = resolution
    scale = np.asarray(component_scale, dtype=np.float64)
    pred = np.asarray(prediction, dtype=np.float64).reshape(ny, nx, -1) / scale
    truth = np.asarray(reference, dtype=np.float64).reshape(ny, nx, -1) / scale
    error_spectrum = np.fft.rfftn(pred - truth, axes=(0, 1), norm="ortho")
    truth_spectrum = np.fft.rfftn(truth, axes=(0, 1), norm="ortho")
    frequency_x = np.fft.rfftfreq(nx, d=2.0 / nx)
    frequency_y = np.fft.fftfreq(ny, d=1.0 / ny)
    radial = np.hypot(frequency_y[:, None], frequency_x[None, :])
    selected = (radial >= 1.0 / wavelength_max) & (radial < 1.0 / wavelength_min)
    if not np.any(selected):
        raise ValueError("physical wavelength band has no modes")
    error_energy = float(np.sum(np.abs(error_spectrum[selected]) ** 2))
    truth_energy = float(np.sum(np.abs(truth_spectrum[selected]) ** 2))
    return {
        "metric_kind": "physical_wavelength_band",
        "wavelength_min": wavelength_min,
        "wavelength_max": wavelength_max,
        "mode_count": int(np.count_nonzero(selected)),
        "error_spectral_l2": float(np.sqrt(error_energy)),
        "reference_spectral_l2": float(np.sqrt(truth_energy)),
        "relative_spectral_l2": (
            None
            if truth_energy <= 1.0e-30
            else float(np.sqrt(error_energy / truth_energy))
        ),
    }


def _reference_scales(
    store: PCNOEuler2DShardStore,
    keys: Sequence[str],
    *,
    start_frame: int,
    step_stride: int,
    rollout_calls: int,
    gamma: float,
) -> dict[str, Any]:
    fields = {
        "density": {"minimum": math.inf, "maximum": -math.inf, "update_abs_max": 0.0},
        "pressure": {"minimum": math.inf, "maximum": -math.inf, "update_abs_max": 0.0},
    }
    for key in keys:
        states = store.states(key)
        indices = start_frame + np.arange(rollout_calls + 1) * step_stride
        selected = np.asarray(states[indices], dtype=np.float64)
        primitive = conservative_to_primitive_raw(selected, gamma=gamma)
        values = {"density": primitive[..., 0], "pressure": primitive[..., 3]}
        for name, value in values.items():
            fields[name]["minimum"] = min(fields[name]["minimum"], float(value.min()))
            fields[name]["maximum"] = max(fields[name]["maximum"], float(value.max()))
            fields[name]["update_abs_max"] = max(
                fields[name]["update_abs_max"],
                float(np.max(np.abs(np.diff(value, axis=0)))),
            )
    for value in fields.values():
        span = value["maximum"] - value["minimum"]
        if not math.isfinite(span) or span <= 0.0:
            raise ValueError("reference field has no positive finite range")
        value["error_abs_max"] = 0.05 * span
        value["difference_abs_max"] = value["update_abs_max"]
        value["scale_source"] = (
            "open-validation reference states only; fixed before model inference"
        )
    return {
        "scope": "complete frozen open-validation rollout cohort",
        "outcome_independent": True,
        "includes_all_reference_frames": True,
        "fields": fields,
    }


def _admissibility(
    prediction: torch.Tensor,
    *,
    model: PCNOEuler2DResidual,
    policy: Mapping[str, Any] | None,
) -> dict[str, Any]:
    diagnostics = conservative_admissibility(prediction.float(), gamma=model.gamma)
    finite = bool(diagnostics["finite_components"].all())
    density = diagnostics["density"]
    internal = diagnostics["internal_energy"]
    pressure = diagnostics["pressure"]

    def finite_minimum(value: torch.Tensor) -> float | None:
        selected = value[torch.isfinite(value)]
        return None if selected.numel() == 0 else float(selected.min().cpu())

    result = {
        "finite": finite,
        "admissible": bool(diagnostics["admissible"].all()),
        "min_density": finite_minimum(density),
        "min_internal_energy": finite_minimum(internal),
        "min_pressure": finite_minimum(pressure),
        "minimum_outflow_normal_mach": None,
        "outflow_admissible": True,
    }
    if policy is not None and result["admissible"]:
        outflow = boundary_outflow_normal_mach(prediction.float(), policy)
        if outflow is None or not bool(torch.isfinite(outflow).all()):
            result["outflow_admissible"] = False
        else:
            minimum = float(outflow.min().cpu())
            result["minimum_outflow_normal_mach"] = minimum
            result["outflow_admissible"] = minimum > 1.0
    result["deployment_admissible"] = bool(
        result["admissible"] and result["outflow_admissible"]
    )
    return result


def _activation_rows_and_maps(
    model: PCNOEuler2DResidual,
    correct: Mapping[str, torch.Tensor],
    intervened: Mapping[str, torch.Tensor],
    *,
    correct_features: torch.Tensor,
    intervened_features: torch.Tensor,
    weights: np.ndarray,
    regions: Mapping[str, np.ndarray],
    common: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    differences = activation_differences(correct, intervened)
    rows: list[dict[str, Any]] = []
    maps: dict[str, np.ndarray] = {}
    for field, tensor in differences.items():
        difference = tensor[0].float().numpy().astype(np.float64)
        correct_value = correct[field][0].float().numpy().astype(np.float64)
        maps[field.replace(".", "_") + "_difference_norm"] = np.linalg.norm(
            difference, axis=-1
        ).astype(np.float32)
        for region_name, mask in regions.items():
            selected = np.asarray(mask, dtype=bool)
            if not np.any(selected):
                continue
            mass = weights[selected]
            correct_rms = float(
                np.sqrt(
                    np.dot(mass, np.mean(np.square(correct_value[selected]), axis=-1))
                    / mass.sum()
                )
            )
            difference_rms = float(
                np.sqrt(
                    np.dot(mass, np.mean(np.square(difference[selected]), axis=-1))
                    / mass.sum()
                )
            )
            rows.append(
                {
                    **common,
                    "field": field,
                    "region": region_name,
                    "node_count": int(selected.sum()),
                    "weight_mass": float(mass.sum()),
                    "correct_rms": correct_rms,
                    "difference_rms": difference_rms,
                    "difference_to_correct_rms": difference_rms
                    / max(correct_rms, 1.0e-30),
                    "interpretation": "diagnostic propagation norm, not causal share",
                }
            )
    correct_lift = model.boundary_lift_contribution(correct_features)
    intervened_lift = model.boundary_lift_contribution(intervened_features)
    analytical_boundary = (correct_lift - intervened_lift).detach().float().cpu()
    actual = differences["post_lift"].float()
    input_difference = (
        correct["model_input"].float() - intervened["model_input"].float()
    )
    lift_weight = model.backbone.fc0.weight.detach().float().cpu()
    analytical_full = torch.matmul(input_difference, lift_weight.transpose(0, 1))
    full_error = actual - analytical_full
    boundary_start = 7
    boundary_stop = boundary_start + model.boundary_field_input_count
    nonboundary_difference = input_difference.clone()
    nonboundary_difference[..., boundary_start:boundary_stop] = 0.0
    same_nonboundary_input = bool(torch.count_nonzero(nonboundary_difference) == 0)
    boundary_only_error = actual - analytical_boundary
    maps["analytical_boundary_lift_norm"] = (
        torch.linalg.vector_norm(analytical_boundary[0], dim=-1)
        .numpy()
        .astype(np.float32)
    )
    maps["analytical_full_lift_error_norm"] = (
        torch.linalg.vector_norm(full_error[0], dim=-1).numpy().astype(np.float32)
    )
    maps["analytical_boundary_only_error_norm"] = (
        torch.linalg.vector_norm(boundary_only_error[0], dim=-1)
        .numpy()
        .astype(np.float32)
    )
    rows.append(
        {
            **common,
            "field": "analytical_full_post_lift_check",
            "region": "all",
            "node_count": int(weights.size),
            "weight_mass": float(weights.sum()),
            "correct_rms": None,
            "difference_rms": float(torch.sqrt(torch.mean(full_error.square()))),
            "difference_to_correct_rms": None,
            "maximum_absolute_error": float(full_error.abs().max()),
            "same_nonboundary_input": same_nonboundary_input,
            "interpretation": (
                "the complete correct-minus-intervened post-lift difference equals "
                "W_fc0 times the complete model-input difference"
            ),
        }
    )
    rows.append(
        {
            **common,
            "field": "analytical_boundary_only_post_lift_check",
            "region": "all",
            "node_count": int(weights.size),
            "weight_mass": float(weights.sum()),
            "correct_rms": None,
            "difference_rms": (
                float(torch.sqrt(torch.mean(boundary_only_error.square())))
                if same_nonboundary_input
                else None
            ),
            "difference_to_correct_rms": None,
            "maximum_absolute_error": (
                float(boundary_only_error.abs().max())
                if same_nonboundary_input
                else None
            ),
            "same_nonboundary_input": same_nonboundary_input,
            "interpretation": (
                "when state and all non-boundary inputs match, the post-lift "
                "difference equals the exact additive boundary term W_B Delta B; "
                "later free-rollout calls intentionally leave this unevaluated"
            ),
        }
    )
    return rows, maps


def _visualization_indices(
    family: str,
    nodes: np.ndarray,
    resolution: tuple[int, int] | None,
    maximum_nodes: int,
) -> np.ndarray:
    count = nodes.shape[0]
    if count <= maximum_nodes:
        return np.arange(count, dtype=np.int64)
    if family == "dynamic_fv":
        if resolution is None:
            raise ValueError("dynamic visualization requires a grid resolution")
        nx, ny = resolution
        stride = math.ceil(math.sqrt(count / maximum_nodes))
        return (
            np.arange(count, dtype=np.int64)
            .reshape(ny, nx)[::stride, ::stride]
            .reshape(-1)
        )
    stride = math.ceil(count / maximum_nodes)
    return np.arange(0, count, stride, dtype=np.int64)


def _aggregate_rows(
    outcome_rows: Sequence[Mapping[str, Any]],
    completion_rows: Sequence[Mapping[str, Any]],
    *,
    endpoints: Sequence[int],
) -> dict[str, Any]:
    endpoint_rows = [
        row
        for row in outcome_rows
        if row["mode"] == "free_rollout"
        and row["region"] == "all"
        and row["call"] in endpoints
    ]
    endpoint_summary = []
    groups: defaultdict[tuple[str, str, int], list[float]] = defaultdict(list)
    for row in endpoint_rows:
        value = row.get("prediction_relative_l2")
        if value is not None:
            groups[(row["arm"], row["intervention"], row["call"])].append(float(value))
    for (arm, intervention, call), values in sorted(groups.items()):
        endpoint_summary.append(
            {
                "arm": arm,
                "intervention": intervention,
                "call": call,
                "population": len(values),
                "mean_relative_l2": float(np.mean(values)),
                "sample_std": (
                    None if len(values) < 2 else float(np.std(values, ddof=1))
                ),
            }
        )
    variants = sorted({(row["arm"], row["intervention"]) for row in completion_rows})
    completion_summary = []
    for arm, intervention in variants:
        rows = [
            row
            for row in completion_rows
            if row["mode"] == "free_rollout"
            and row["arm"] == arm
            and row["intervention"] == intervention
        ]
        cases = sorted({str(row["case_id"]) for row in rows})
        valid_by_case = {
            case: max(
                [
                    int(row["call"])
                    for row in rows
                    if row["case_id"] == case and row["deployment_admissible"]
                ],
                default=0,
            )
            for case in cases
        }
        requested = max(endpoints)
        completion_summary.append(
            {
                "arm": arm,
                "intervention": intervention,
                "cases": len(cases),
                "completed": sum(
                    value == requested for value in valid_by_case.values()
                ),
                "completion_rate": (
                    sum(value == requested for value in valid_by_case.values())
                    / len(cases)
                ),
                "mean_survival_fraction": float(
                    np.mean([value / requested for value in valid_by_case.values()])
                ),
            }
        )
    return {
        "endpoint_summary": endpoint_summary,
        "completion_summary": completion_summary,
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    contract = FAMILY_CONTRACTS[args.family]
    device = select_device(args.device)
    args.output_dir.mkdir(parents=True)
    for name in ("animations", "traces"):
        (args.output_dir / name).mkdir()

    store = PCNOEuler2DShardStore(args.data_dir, max_cached_trajectories=2)
    _require_equal(
        store.manifest_digest,
        args.expected_data_manifest_digest,
        "loaded data manifest digest",
    )
    _require_equal(
        store.boundary_field_names,
        contract["semantic_names"],
        "boundary semantic field order",
    )
    arms = _load_arms(args, device)
    source_verification = _verify_training_source_snapshot(arms)
    split = arms["N0"].split
    open_keys = [str(value) for value in split["rollout_keys"]]
    if args.case_ids:
        requested = [str(value) for value in args.case_ids]
        missing = sorted(set(requested) - set(open_keys))
        if missing:
            raise ValueError(f"requested cases are outside the open split: {missing}")
        keys = requested
    else:
        keys = open_keys
    trace_cases = [str(value) for value in args.trace_cases]
    missing_traces = sorted(set(trace_cases) - set(keys))
    if missing_traces:
        raise ValueError(
            f"trace cases are outside the evaluated cohort: {missing_traces}"
        )

    gamma = float(store.manifest.get("gamma", 1.4))
    reference_scales = _reference_scales(
        store,
        keys,
        start_frame=contract["start_frame"],
        step_stride=contract["step_stride"],
        rollout_calls=contract["rollout_calls"],
        gamma=gamma,
    )
    atomic_write_json(args.output_dir / "reference_scales.json", reference_scales)

    policies: dict[str, Mapping[str, Any] | None] = {}
    policy_metadata: dict[str, Any] = {}
    if args.family == "bump":
        training_args = arms["N0"].run_contract["args"]
        max_hops = int(training_args.get("boundary_max_source_hops", 3))
        rho_inf = float(training_args.get("boundary_rho_inf", 1.4))
        p_inf = float(training_args.get("boundary_p_inf", 1.0))
        for key in keys:
            policy, metadata = build_graph_causal_boundary_policy(
                store,
                key,
                device=device,
                max_source_hops=max_hops,
                rho_inf=rho_inf,
                p_inf=p_inf,
            )
            policies[key] = policy
            policy_metadata[key] = metadata
    else:
        policies = {key: None for key in keys}

    first_key = keys[0]
    first_sample = store.tensor_sample(
        first_key,
        contract["start_frame"],
        step_stride=contract["step_stride"],
        device=device,
    )
    first_current = np.array(
        store.states(first_key)[contract["start_frame"]], copy=True
    )
    hook_equivalence = {
        arm: _hook_equivalence(
            loaded,
            first_sample,
            first_current,
            boundary_policy=policies[first_key],
            device=device,
            amp=args.amp,
            absolute_limit=contract["hook_abs_limit"],
            relative_limit=contract["hook_rel_limit"],
        )
        for arm, loaded in arms.items()
    }
    atomic_write_json(args.output_dir / "hook_equivalence.json", hook_equivalence)

    variants = _variants(contract["semantic_names"])
    animation_names = {
        "N0_correct",
        "G1_correct",
        "G1_zero_all",
        "S1_correct",
        "S1_zero_all",
    }
    outcome_rows: list[dict[str, Any]] = []
    completion_rows: list[dict[str, Any]] = []
    frequency_rows: list[dict[str, Any]] = []
    structure_rows: list[dict[str, Any]] = []
    activation_rows: list[dict[str, Any]] = []
    trace_files: list[Path] = []
    animation_files: list[Path] = []

    for key_index, key in enumerate(keys, 1):
        states = store.states(key)
        frame_indices = (
            contract["start_frame"]
            + np.arange(contract["rollout_calls"] + 1) * contract["step_stride"]
        )
        if int(frame_indices[-1]) >= states.shape[0]:
            raise ValueError(f"{key} lacks the requested rollout frames")
        reference_states = np.array(states[frame_indices], copy=True, dtype=np.float64)
        physical_times = np.array(
            store.array(key, "physical_times")[frame_indices],
            copy=True,
            dtype=np.float64,
        )
        nodes = np.array(store.array(key, "nodes"), copy=True, dtype=np.float64)
        edges = np.array(store.array(key, "edges"), copy=True, dtype=np.int64)
        node_type = np.array(store.array(key, "node_type"), copy=True).reshape(-1)
        weights = _weights(store, key)
        boundary_features_np = np.array(
            store.array(key, "boundary_features"), copy=True, dtype=np.float64
        )
        boundary_distance = _boundary_distance(args.family, nodes, edges, node_type)
        resolution = _dynamic_resolution(nodes) if args.family == "dynamic_fv" else None
        sample = store.tensor_sample(
            key,
            contract["start_frame"],
            step_stride=contract["step_stride"],
            device=device,
        )
        policy = policies[key]
        case_is_traced = key in set(trace_cases)
        animation_series: dict[str, np.ndarray] = {}
        if case_is_traced:
            for mode in MODES:
                for variant in variants:
                    if variant.name in animation_names:
                        value = np.full(
                            (contract["rollout_calls"] + 1, nodes.shape[0], 4),
                            np.nan,
                            dtype=np.float32,
                        )
                        value[0] = reference_states[0].astype(np.float32)
                        animation_series[f"{mode}_{variant.name}"] = value

        for mode in MODES:
            free_states = {
                variant.name: reference_states[0].copy() for variant in variants
            }
            active = {variant.name: True for variant in variants}
            for call in range(1, contract["rollout_calls"] + 1):
                target = reference_states[call]
                regions = _region_masks(
                    args.family,
                    target,
                    nodes=nodes,
                    edges=edges,
                    node_type=node_type,
                    boundary_distance=boundary_distance,
                    resolution=resolution,
                    gamma=gamma,
                    shock_quantile=args.shock_quantile,
                )
                predictions: dict[str, np.ndarray] = {}
                currents: dict[str, np.ndarray] = {}
                call_inputs: dict[str, np.ndarray] = {}
                for variant in variants:
                    if mode == "free_rollout" and not active[variant.name]:
                        continue
                    current = (
                        reference_states[call - 1]
                        if mode == "teacher_forced"
                        else free_states[variant.name]
                    )
                    prediction_tensor, _, model_current_tensor, _ = _model_call(
                        arms[variant.arm].model,
                        sample,
                        current,
                        boundary_features=_features_for_variant(sample, variant),
                        boundary_policy=policy,
                        device=device,
                        amp=args.amp,
                        trace=False,
                    )
                    prediction = (
                        prediction_tensor[0]
                        .detach()
                        .float()
                        .cpu()
                        .numpy()
                        .astype(np.float64)
                    )
                    model_current = (
                        model_current_tensor[0]
                        .detach()
                        .float()
                        .cpu()
                        .numpy()
                        .astype(np.float64)
                    )
                    admissibility = _admissibility(
                        prediction_tensor,
                        model=arms[variant.arm].model,
                        policy=policy,
                    )
                    completion_rows.append(
                        {
                            "family": args.family,
                            "seed": args.seed,
                            "case_id": key,
                            "mode": mode,
                            "call": call,
                            "physical_time": float(physical_times[call]),
                            "arm": variant.arm,
                            "intervention": variant.intervention,
                            **admissibility,
                        }
                    )
                    if (
                        mode == "free_rollout"
                        and not admissibility["deployment_admissible"]
                    ):
                        active[variant.name] = False
                        continue
                    predictions[variant.name] = prediction
                    currents[variant.name] = model_current
                    call_inputs[variant.name] = np.asarray(current, dtype=np.float64)
                    if mode == "free_rollout":
                        free_states[variant.name] = prediction
                    if case_is_traced and variant.name in animation_names:
                        animation_series[f"{mode}_{variant.name}"][call] = prediction

                for variant in variants:
                    if variant.name not in predictions:
                        continue
                    prediction = predictions[variant.name]
                    current = currents[variant.name]
                    state_scale = (
                        arms[variant.arm]
                        .model.state_scale.detach()
                        .cpu()
                        .numpy()
                        .reshape(-1)
                    )
                    residual_scale = (
                        arms[variant.arm]
                        .model.residual_scale.detach()
                        .cpu()
                        .numpy()
                        .reshape(-1)
                    )
                    correct_name = f"{variant.arm}_correct"
                    correct_prediction = predictions.get(correct_name)
                    correct_current = currents.get(correct_name)
                    for region_name, mask in regions.items():
                        outcome_rows.append(
                            {
                                "family": args.family,
                                "seed": args.seed,
                                "case_id": key,
                                "mode": mode,
                                "call": call,
                                "physical_time": float(physical_times[call]),
                                "arm": variant.arm,
                                "intervention": variant.intervention,
                                "region": region_name,
                                "node_count": int(np.asarray(mask).sum()),
                                "weight_mass": float(weights[np.asarray(mask)].sum()),
                                "prediction_error_rms": _scaled_rms(
                                    prediction - target,
                                    weights=weights,
                                    scale=state_scale,
                                    mask=mask,
                                ),
                                "prediction_relative_l2": _scaled_relative_l2(
                                    prediction,
                                    target,
                                    weights=weights,
                                    scale=state_scale,
                                    mask=mask,
                                ),
                                "state_gap_to_arm_correct_rms": (
                                    None
                                    if correct_prediction is None
                                    else _scaled_rms(
                                        prediction - correct_prediction,
                                        weights=weights,
                                        scale=state_scale,
                                        mask=mask,
                                    )
                                ),
                                "increment_gap_to_arm_correct_rms": (
                                    None
                                    if correct_prediction is None
                                    or correct_current is None
                                    else _scaled_rms(
                                        (prediction - current)
                                        - (correct_prediction - correct_current),
                                        weights=weights,
                                        scale=residual_scale,
                                        mask=mask,
                                    )
                                ),
                                "normalized_volume_weighted_total_error_rms": float(
                                    np.sqrt(
                                        np.mean(
                                            np.square(
                                                np.sum(
                                                    weights[:, None]
                                                    * (prediction - target),
                                                    axis=0,
                                                )
                                                / state_scale
                                            )
                                        )
                                    )
                                ),
                                "total_metric_claim": (
                                    "normalized physical cell-volume state-total error"
                                    if args.family == "dynamic_fv"
                                    else "reconstructed graph quadrature proxy only"
                                ),
                            }
                        )
                    scale = state_scale
                    if args.family == "dynamic_fv":
                        if resolution is None:
                            raise AssertionError("dynamic resolution is missing")
                        for wavelength_min, wavelength_max in PHYSICAL_WAVELENGTH_BANDS:
                            frequency_rows.append(
                                {
                                    "family": args.family,
                                    "seed": args.seed,
                                    "case_id": key,
                                    "mode": mode,
                                    "call": call,
                                    "arm": variant.arm,
                                    "intervention": variant.intervention,
                                    "comparison": "prediction_vs_reference_state",
                                    **_physical_wavelength_metrics(
                                        prediction,
                                        target,
                                        resolution=resolution,
                                        component_scale=scale,
                                        wavelength_min=wavelength_min,
                                        wavelength_max=wavelength_max,
                                    ),
                                }
                            )
                    else:
                        highpass = node_highpass_field(prediction - target, edges)
                        frequency_rows.append(
                            {
                                "family": args.family,
                                "seed": args.seed,
                                "case_id": key,
                                "mode": mode,
                                "call": call,
                                "arm": variant.arm,
                                "intervention": variant.intervention,
                                "comparison": "prediction_vs_reference_state",
                                "metric_kind": (
                                    "graph_highpass_proxy_not_physical_frequency"
                                ),
                                "error_highpass_rms": _scaled_rms(
                                    highpass,
                                    weights=weights,
                                    scale=scale,
                                    mask=np.ones(nodes.shape[0], dtype=bool),
                                ),
                            }
                        )
                    if call in contract["endpoints"]:
                        structure_rows.append(
                            {
                                "family": args.family,
                                "seed": args.seed,
                                "case_id": key,
                                "mode": mode,
                                "call": call,
                                "arm": variant.arm,
                                "intervention": variant.intervention,
                                **endpoint_diagnostics(
                                    prediction,
                                    target,
                                    positions=nodes,
                                    edges=edges,
                                    node_type=node_type,
                                    proxy_weights=weights,
                                    component_scale=scale,
                                    gamma=gamma,
                                    shock_quantile=args.shock_quantile,
                                ),
                            }
                        )

                if case_is_traced and call in set(args.trace_calls):
                    correct_variant = Variant("S1", "correct")
                    zero_variant = Variant(
                        "S1", "zero_all", tuple(range(len(contract["semantic_names"])))
                    )
                    for variant in (correct_variant, zero_variant):
                        if variant.name not in predictions:
                            raise RuntimeError(
                                f"trace variant {variant.name} is unavailable at {key}:{call}"
                            )
                    trace_payloads = {}
                    for variant in (correct_variant, zero_variant):
                        traced, _, _, trace = _model_call(
                            arms["S1"].model,
                            sample,
                            call_inputs[variant.name],
                            boundary_features=_features_for_variant(sample, variant),
                            boundary_policy=policy,
                            device=device,
                            amp=args.amp,
                            trace=True,
                        )
                        equivalence = tensor_equivalence_metrics(
                            torch.as_tensor(
                                predictions[variant.name],
                                dtype=torch.float32,
                                device=device,
                            ).unsqueeze(0),
                            traced,
                        )
                        if not _passes_equivalence(
                            equivalence,
                            absolute_limit=contract["hook_abs_limit"],
                            relative_limit=contract["hook_rel_limit"],
                        ):
                            raise RuntimeError(
                                f"traced call perturbed output: {equivalence}"
                            )
                        if trace is None:
                            raise RuntimeError("traced call produced no activations")
                        trace_payloads[variant.name] = trace
                    correct_features = _features_for_variant(sample, correct_variant)
                    zero_features = _features_for_variant(sample, zero_variant)
                    if correct_features is None or zero_features is None:
                        raise AssertionError("S1 trace lacks boundary fields")
                    common = {
                        "family": args.family,
                        "seed": args.seed,
                        "case_id": key,
                        "mode": mode,
                        "call": call,
                        "physical_time": float(physical_times[call]),
                        "arm": "S1",
                        "intervention": "zero_all",
                    }
                    rows, maps = _activation_rows_and_maps(
                        arms["S1"].model,
                        trace_payloads["S1_correct"],
                        trace_payloads["S1_zero_all"],
                        correct_features=correct_features,
                        intervened_features=zero_features,
                        weights=weights,
                        regions=regions,
                        common=common,
                    )
                    activation_rows.extend(rows)
                    trace_path = (
                        args.output_dir
                        / "traces"
                        / f"{args.family}_{key}_{mode}_call{call:03d}_S1_zero_all.npz"
                    )
                    np.savez_compressed(
                        trace_path,
                        schema=np.asarray(SCHEMA),
                        family=np.asarray(args.family),
                        seed=np.asarray(args.seed),
                        case_id=np.asarray(key),
                        mode=np.asarray(mode),
                        call=np.asarray(call),
                        physical_time=np.asarray(physical_times[call]),
                        nodes=nodes.astype(np.float32),
                        physical_node_type=node_type.astype(np.int64),
                        weights=weights.astype(np.float64),
                        boundary_distance=boundary_distance.astype(np.float64),
                        boundary_features=boundary_features_np.astype(np.float32),
                        correct_prediction=predictions["S1_correct"].astype(np.float32),
                        intervened_prediction=predictions["S1_zero_all"].astype(
                            np.float32
                        ),
                        target=target.astype(np.float32),
                        **{f"region_{name}": value for name, value in regions.items()},
                        **maps,
                    )
                    trace_files.append(trace_path)

        if case_is_traced:
            indices = _visualization_indices(
                args.family, nodes, resolution, args.animation_max_nodes
            )
            animation_path = (
                args.output_dir / "animations" / f"{args.family}_{key}_s{args.seed}.npz"
            )
            np.savez_compressed(
                animation_path,
                schema=np.asarray(SCHEMA),
                family=np.asarray(args.family),
                seed=np.asarray(args.seed),
                case_id=np.asarray(key),
                resolution=np.asarray(
                    "native_graph"
                    if resolution is None
                    else f"{resolution[0]}x{resolution[1]}"
                ),
                physical_times=physical_times,
                nodes=nodes[indices].astype(np.float32),
                physical_node_type=node_type[indices].astype(np.int64),
                boundary_distance=boundary_distance[indices].astype(np.float32),
                boundary_features=boundary_features_np[indices].astype(np.float32),
                boundary_field_names=np.asarray(contract["semantic_names"]),
                reference_states=reference_states[:, indices].astype(np.float32),
                visualization_node_indices=indices,
                visualization_only_subsampling=np.asarray(
                    indices.size < nodes.shape[0]
                ),
                requested_frame_count=np.asarray(contract["rollout_calls"] + 1),
                all_reference_frames_included=np.asarray(True),
                post_failure_values=np.asarray(
                    "NaN; no recurrence state was fabricated after first failure"
                ),
                **{name: value[:, indices] for name, value in animation_series.items()},
            )
            animation_files.append(animation_path)
        print(
            json.dumps(
                {"case": key, "completed": key_index, "population": len(keys)},
                sort_keys=True,
            ),
            flush=True,
        )

    write_csv(args.output_dir / "outcomes.csv", outcome_rows)
    write_csv(args.output_dir / "completion.csv", completion_rows)
    write_csv(args.output_dir / "frequency.csv", frequency_rows)
    write_csv(args.output_dir / "structure.csv", structure_rows)
    write_csv(args.output_dir / "activations.csv", activation_rows)
    aggregate = _aggregate_rows(
        outcome_rows, completion_rows, endpoints=contract["endpoints"]
    )
    summary = {
        "schema": SCHEMA,
        "status": "complete",
        "scope": "frozen open-validation population only; sealed test population absent",
        "family": args.family,
        "seed": args.seed,
        "args": jsonable_args(args),
        "contract": contract,
        "evaluated_cases": keys,
        "open_population_count": len(keys),
        "variants": [variant.__dict__ | {"name": variant.name} for variant in variants],
        "checkpoints": {
            arm: {
                "path": str(loaded.run_dir / "best.pt"),
                "sha256": loaded.checkpoint_sha256,
                "best_epoch": int(loaded.summary["best_epoch"]),
                "model_config": loaded.summary["model_config"],
            }
            for arm, loaded in arms.items()
        },
        "data_manifest_digest": store.manifest_digest,
        "boundary_field_contract": store.boundary_field_contract,
        "source_verification": source_verification,
        "physical_boundary_policy": contract["boundary_mode"],
        "physical_boundary_policy_changed": False,
        "policy_metadata": policy_metadata,
        "hook_equivalence": hook_equivalence,
        "reference_scales": reference_scales,
        "aggregate": aggregate,
        "artifacts": {
            "animation_files": [
                str(path.relative_to(args.output_dir)) for path in animation_files
            ],
            "trace_files": [
                str(path.relative_to(args.output_dir)) for path in trace_files
            ],
        },
        "claim_boundary": {
            "training_arm_contrast": (
                "effect of access to the representation under one matched recipe; "
                "not optimality"
            ),
            "zero_field_intervention": (
                "causal effect within the frozen checkpoint and open cohort only"
            ),
            "activation_norms": "mechanistic diagnostics, not causal branch shares",
            "bump_weights": "quadrature proxies; no physical conservation claim",
            "dynamic_total_metric": (
                "reference state-total discrepancy, not a boundary-flux balance proof"
            ),
        },
        "environment": runtime_environment(device),
        "git": git_state(),
    }
    atomic_write_json(args.output_dir / "summary.json", summary)
    store.close()
    print(f"wrote {args.output_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
