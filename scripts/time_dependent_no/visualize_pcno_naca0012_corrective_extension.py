#!/usr/bin/env python3
"""Render the frozen NACA corrective-extension development comparison.

Static figures consume only the verified evaluator packet.  Animations consume
an independent qualitative replay of the predeclared development anchor.  The
two sources are kept distinct because independent numerical rollouts are not exact
long-horizon reproductions.  Prospective and sealed populations are refused.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import shutil
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import torch

MODULE_PATH = Path(__file__)
if MODULE_PATH.is_symlink():
    raise RuntimeError("corrective-extension visualization source is aliased")
REPO_ROOT = MODULE_PATH.resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _source_record(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    if path.read_bytes() != payload:
        raise RuntimeError("visualization source changed while reading")
    return {"bytes": len(payload), "sha256": sha256(payload).hexdigest()}


VISUALIZATION_SOURCE_AT_IMPORT = _source_record(MODULE_PATH.resolve())

from scripts.time_dependent_no import (
    evaluate_pcno_naca0012_corrective_extension as evaluator,
)
from utility.time_dependent_no.pcno_naca0012 import recurrent_step
from utility.time_dependent_no.pcno_naca0012_successor import (
    corrected_recurrent_step,
)

if _source_record(MODULE_PATH.resolve()) != VISUALIZATION_SOURCE_AT_IMPORT:
    raise RuntimeError("visualization source changed during bootstrap")

EXPERIMENT_ID = evaluator.EXPERIMENT_ID
SEEDS = evaluator.SEEDS
ANCHOR = 1234
STATIC_HORIZONS = (35, 104, 208)
FULL_HORIZONS = tuple(range(evaluator.FULL_TRACE[1] + 1))
FIELD = "Density"
FIELD_INDEX = 0
X_LIMITS = (-1.0, 10.0)
Y_LIMITS = (-5.0, 5.0)
REFINER_SAMPLER_SEEDS = evaluator.REFINER_SAMPLER_SEEDS

REPLAY_SCHEMA = "time_dependent_no.naca_corrective_extension_visualization_replay.v1"
RENDER_SCHEMA = "time_dependent_no.naca_corrective_extension_visualization.v2"
REPLAY_FILE = "extension_rollout_replay.npz"
REPLAY_MANIFEST_FILE = "replay_manifest.json"
RENDER_MANIFEST_FILE = "visualization_manifest.json"
SIGNED_ERROR_LINEAR_THRESHOLD = 1.0e-2


@dataclass(frozen=True)
class MethodSpec:
    """One predeclared deployed-map view."""

    slug: str
    display: str
    arm: str
    deployment: str
    state_stage: str
    sampler_seed: int | None
    packet_family: str

    def record(self) -> dict[str, Any]:
        return {
            "slug": self.slug,
            "display": self.display,
            "arm": self.arm,
            "deployment": self.deployment,
            "state_stage": self.state_stage,
            "sampler_seed": self.sampler_seed,
            "packet_family": self.packet_family,
        }


METHODS = (
    MethodSpec("clean", "Clean", "CLEAN", "raw", "raw", None, "inherited"),
    MethodSpec(
        "clean_ema", "Clean + EMA", "CLEAN_EMA", "ema", "raw", None, "extension"
    ),
    MethodSpec(
        "detached",
        "Detached prefix",
        "DETACHED_PUSHFORWARD",
        "raw",
        "raw",
        None,
        "inherited",
    ),
    MethodSpec(
        "mp_pde",
        "MP-PDE",
        "MP_PDE_PUSHFORWARD_M01",
        "online",
        "raw",
        None,
        "extension",
    ),
    MethodSpec(
        "curriculum",
        "Curriculum + EMA",
        "CURRICULUM_EMA_PUSHFORWARD_K13",
        "ema",
        "raw",
        None,
        "extension",
    ),
    MethodSpec(
        "recovery",
        "Recovery",
        "PAIRED_RECOVERY",
        "online",
        "raw",
        None,
        "extension",
    ),
    MethodSpec(
        "relabel",
        "Dynamics relabeling",
        "DYNAMICS_RELABEL",
        "online",
        "raw",
        None,
        "extension",
    ),
    MethodSpec(
        "pderefiner",
        "PDE-Refiner (sampler 101)",
        evaluator.REFINER_ARM,
        "ema",
        "raw",
        101,
        "extension",
    ),
    MethodSpec(
        "path_projection",
        "Path projection",
        "PATH_PROJECTION",
        "path_projection",
        "corrected",
        None,
        "path_projection",
    ),
)

METRIC_SPECS = (
    (
        "normalized_state_error_auc",
        "Rollout state-error AUC",
    ),
    (
        "late_window_path_distance_median",
        "Late empirical train-path proximity proxy",
    ),
    (
        "late_window_graph_dirichlet_error_energy_median",
        "Late spatial roughness error energy",
    ),
)

METHOD_COLORS = (
    "#000000",
    "#56B4E9",
    "#009E73",
    "#0072B2",
    "#999999",
    "#E69F00",
    "#D55E00",
    "#F0E442",
    "#CC79A7",
)

QUALITATIVE_CONTRACT = {
    "population_role": "development",
    "anchor": ANCHOR,
    "seeds": list(SEEDS),
    "methods": [method.record() for method in METHODS],
    "field": FIELD,
    "full_horizons_inclusive": [FULL_HORIZONS[0], FULL_HORIZONS[-1]],
    "static_horizons": list(STATIC_HORIZONS),
    "view": {"x": list(X_LIMITS), "y": list(Y_LIMITS)},
    "spatial_representation": "flat_native_quadrilateral_vertex_mean",
    "interpolation": "none",
    "static_method_panels": (
        "reference_density_plus_pointwise_seed_median_signed_normalized_density_error"
    ),
    "static_source": "exact_verified_evaluator_snapshots",
    "animation_source": "independent_qualitative_replay",
    "animation_is_exact_evaluator_reproduction": False,
    "summary_aggregation": {
        "within_seed": "median_over_eight_development_anchors",
        "across_seed_center": "median_over_seeds_17_29_43",
        "across_seed_range": "minimum_to_maximum_over_seeds_17_29_43",
        "range_is_confidence_interval": False,
    },
    "refiner_sampler_snapshot": {
        "horizon": 208,
        "deployment": "ema",
        "training_seeds": list(SEEDS),
        "sampler_seeds": list(REFINER_SAMPLER_SEEDS),
    },
    "prospective_opened": False,
    "sealed_opened": False,
}
QUALITATIVE_CONTRACT_SHA256 = evaluator._canonical_sha256(QUALITATIVE_CONTRACT)


@dataclass(frozen=True)
class ReplayBundle:
    """Portable arrays used by the outcome-blind rendering layer."""

    horizons: np.ndarray
    frame_indices: np.ndarray
    coordinates: np.ndarray
    quads: np.ndarray
    airfoil_mask: np.ndarray
    reference_density: np.ndarray
    predicted_density: np.ndarray
    valid_mask: np.ndarray
    static_horizons: np.ndarray
    exact_reference_density: np.ndarray
    exact_predicted_density: np.ndarray
    exact_valid_mask: np.ndarray
    refiner_sampler_density_h208: np.ndarray
    refiner_sampler_valid_h208: np.ndarray
    density_state_scale: float
    per_seed_summary_metrics: np.ndarray
    manifest: Mapping[str, Any]


def _reverify_visualization_source() -> None:
    if _source_record(MODULE_PATH.resolve()) != VISUALIZATION_SOURCE_AT_IMPORT:
        raise ValueError("visualization source changed during execution")


def _claim_boundary() -> dict[str, Any]:
    return {
        "development_only": True,
        "static_figures_use_exact_evaluator_snapshots": True,
        "animations_use_independent_qualitative_replay": True,
        "animation_is_exact_long_horizon_reproduction": False,
        "path_distance_is_empirical_train_path_proximity_proxy": True,
        "path_projection_corrected_path_distance_is_zero_by_construction": True,
        "spatial_ripples_are_called_signed_density_error_or_spatial_roughness": True,
        "off_manifold_claimed": False,
        "trusted_displaced_state_response_claimed": False,
        "descriptive_ranges_are_confidence_intervals": False,
        "online_solver_calls": False,
        "online_defect_trigger": False,
        "prospective_opened": False,
        "sealed_opened": False,
    }


def _array_sha256(value: np.ndarray) -> str:
    return sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def _method_records() -> list[dict[str, Any]]:
    return [method.record() for method in METHODS]


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_replay_provenance(
    manifest: Mapping[str, Any], *, expected_visualization_source_sha256: str
) -> None:
    source = manifest.get("source")
    if not isinstance(source, Mapping) or set(source) != {
        "visualization_script",
        "evaluator_source_set_sha256",
    }:
        raise ValueError("extension visualization replay source schema differs")
    visual_source = source["visualization_script"]
    if (
        not isinstance(visual_source, Mapping)
        or set(visual_source) != {"bytes", "sha256"}
        or isinstance(visual_source["bytes"], bool)
        or not isinstance(visual_source["bytes"], int)
        or visual_source["bytes"] <= 0
        or not _is_sha256(expected_visualization_source_sha256)
        or not _is_sha256(visual_source["sha256"])
        or visual_source["sha256"] != expected_visualization_source_sha256
        or source["evaluator_source_set_sha256"] != evaluator.SOURCE_SET_SHA256
    ):
        raise ValueError("extension visualization replay source binding differs")

    inputs = manifest.get("inputs")
    if not isinstance(inputs, Mapping) or set(inputs) != {
        "extension_contract_sha256",
        "successor_contract_sha256",
        "dataset_final_hash_manifest_sha256",
        "calibration_final_hash_manifest_sha256",
        "relabel_pilot_final_hash_manifest_sha256",
        "paired_bank_final_hash_manifest_sha256",
        "evaluation_final_hash_manifest_sha256",
        "extension_training_packets",
        "inherited_training_packets",
    }:
        raise ValueError("extension visualization replay input schema differs")
    fixed_inputs = {
        "extension_contract_sha256": evaluator.EXTENSION_CONTRACT_SHA256,
        "successor_contract_sha256": evaluator.trainer.SUCCESSOR_CONTRACT_SHA256,
        "dataset_final_hash_manifest_sha256": evaluator.trainer.DATASET_FINAL_SHA256,
        "calibration_final_hash_manifest_sha256": (
            evaluator.trainer.SUCCESSOR_CALIBRATION_SHA256
        ),
    }
    if any(inputs.get(key) != value for key, value in fixed_inputs.items()) or any(
        not _is_sha256(inputs.get(key))
        for key in (
            "relabel_pilot_final_hash_manifest_sha256",
            "paired_bank_final_hash_manifest_sha256",
            "evaluation_final_hash_manifest_sha256",
        )
    ):
        raise ValueError("extension visualization replay fixed input binding differs")

    extension_records = inputs["extension_training_packets"]
    expected_extension_keys = [
        (arm, seed) for arm in evaluator.LEARNED_ARMS for seed in SEEDS
    ]
    if not isinstance(extension_records, list) or len(extension_records) != len(
        expected_extension_keys
    ):
        raise ValueError("extension visualization training inventory differs")
    observed_extension_keys = []
    for record in extension_records:
        if not isinstance(record, Mapping) or set(record) != {
            "arm",
            "seed",
            "final_hash_manifest_sha256",
            "checkpoint_sha256",
            "source_set_sha256",
        }:
            raise ValueError("extension visualization training record differs")
        observed_extension_keys.append((record["arm"], record["seed"]))
        if (
            not _is_sha256(record["final_hash_manifest_sha256"])
            or not _is_sha256(record["checkpoint_sha256"])
            or record["source_set_sha256"] != evaluator.trainer.SOURCE_SET_SHA256
        ):
            raise ValueError("extension visualization training binding differs")
    if observed_extension_keys != expected_extension_keys:
        raise ValueError("extension visualization training grid differs")

    inherited_records = inputs["inherited_training_packets"]
    expected_inherited_keys = [
        (arm, seed) for arm in ("CLEAN", "DETACHED_PUSHFORWARD") for seed in SEEDS
    ]
    if not isinstance(inherited_records, list) or len(inherited_records) != len(
        expected_inherited_keys
    ):
        raise ValueError("extension visualization inherited inventory differs")
    observed_inherited_keys = []
    inherited_source_sets: set[str] = set()
    for record in inherited_records:
        if not isinstance(record, Mapping) or set(record) != {
            "arm",
            "seed",
            "final_hash_manifest_sha256",
            "checkpoint_sha256",
            "source_set_sha256",
        }:
            raise ValueError("extension visualization inherited record differs")
        observed_inherited_keys.append((record["arm"], record["seed"]))
        inherited_source_sets.add(record["source_set_sha256"])
        if (
            not _is_sha256(record["final_hash_manifest_sha256"])
            or not _is_sha256(record["checkpoint_sha256"])
            or not _is_sha256(record["source_set_sha256"])
        ):
            raise ValueError("extension visualization inherited binding differs")
    if observed_inherited_keys != expected_inherited_keys:
        raise ValueError("extension visualization inherited grid differs")
    if len(inherited_source_sets) != 1:
        raise ValueError("extension visualization inherited source identity differs")


def _snapshot_key(
    method: MethodSpec,
    seed: int,
    horizon: int,
    *,
    sampler_seed: int | None = None,
) -> str:
    """Return the exact member name written by the common evaluator."""

    if seed not in SEEDS or horizon not in evaluator.HORIZONS:
        raise ValueError("snapshot identity is outside the frozen evaluator grid")
    if method.arm == "PATH_PROJECTION":
        return f"path_projection_seed{seed}_corrected_a{ANCHOR}_h{horizon}"
    base = f"{method.arm.lower()}_seed{seed}_raw_a{ANCHOR}_h{horizon}"
    if method.packet_family == "inherited":
        return base
    selected_sampler = method.sampler_seed if sampler_seed is None else sampler_seed
    suffix = f"_{method.deployment}"
    if selected_sampler is not None:
        suffix += f"_sampler{selected_sampler}"
    return base + suffix


def _parse_packet_csv(packet: Any, name: str) -> list[dict[str, str]]:
    _, payload = evaluator.parent._verify_record(packet.root, packet.files[name], name)
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{name} is not UTF-8") from error
    rows = list(csv.DictReader(io.StringIO(text)))
    if not rows:
        raise ValueError(f"{name} is empty")
    return rows


def _parse_packet_npz(packet: Any, name: str) -> dict[str, np.ndarray]:
    _, payload = evaluator.parent._verify_record(packet.root, packet.files[name], name)
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        return {key: np.array(archive[key], copy=True) for key in archive.files}


def _expected_extension_training_records(
    packets: Sequence[Any],
) -> list[dict[str, Any]]:
    return [
        {
            "arm": packet.arm,
            "seed": packet.seed,
            "final_hash_manifest_sha256": packet.packet.final_file_sha256,
            "checkpoint_sha256": packet.checkpoint_record["sha256"],
            "source_set_sha256": packet.summary["source_set_sha256"],
        }
        for packet in packets
    ]


def _verify_evaluation_packet(
    root: Path,
    *,
    extension_packets: Sequence[Any],
    inherited_packets: Sequence[Any],
    successor_sha256: str,
    dataset_sha256: str,
    dataset_payload_sha256: str,
    calibration_sha256: str,
    calibration_payload_sha256: str,
    pilot_sha256: str,
    paired_bank_sha256: str,
) -> tuple[Any, dict[str, np.ndarray], list[dict[str, str]]]:
    """Verify the common evaluator and bind it to the supplied model packets."""

    packet = evaluator._verify_output_packet(root)
    inputs = evaluator.parent._packet_json(
        packet,
        "input_manifest.json",
        schema=evaluator.INPUT_MANIFEST_SCHEMA,
        label="extension visualization evaluation inputs",
    )
    result = evaluator.parent._packet_json(
        packet,
        "result.json",
        schema=evaluator.RESULT_SCHEMA,
        label="extension visualization evaluation result",
    )
    expected = {
        "experiment_id": EXPERIMENT_ID,
        "extension_contract_sha256": evaluator.EXTENSION_CONTRACT_SHA256,
        "extension_preregistration_sha256": evaluator.EXTENSION_PREREGISTRATION_SHA256,
        "inherited_r0_contract_sha256": evaluator.trainer.R0_CONTRACT_SHA256,
        "inherited_successor_contract_sha256": successor_sha256,
        "dataset_final_hash_manifest_sha256": dataset_sha256,
        "dataset_manifest_payload_sha256": dataset_payload_sha256,
        "successor_calibration_final_hash_manifest_sha256": calibration_sha256,
        "successor_calibration_payload_sha256": calibration_payload_sha256,
        "relabel_pilot_final_hash_manifest_sha256": pilot_sha256,
        "paired_bank_final_hash_manifest_sha256": paired_bank_sha256,
        "training_packets": _expected_extension_training_records(extension_packets),
        "evaluator_source_set_sha256": evaluator.SOURCE_SET_SHA256,
        "population_role": "development",
        "anchors": list(evaluator.DEVELOPMENT_ANCHORS),
        "horizons": list(evaluator.HORIZONS),
        "full_trace_inclusive": list(evaluator.FULL_TRACE),
        "online_solver_calls": False,
        "online_defect_trigger": False,
        "prospective_opened": False,
        "sealed_opened": False,
    }
    for key, value in expected.items():
        if inputs.get(key) != value:
            raise ValueError(f"extension evaluation input binding differs: {key}")

    inherited = inputs.get("inherited_baselines")
    if not isinstance(inherited, Mapping):
        raise TypeError("extension evaluation lacks inherited-baseline provenance")
    by_arm = {
        arm: [item for item in inherited_packets if item.arm == arm]
        for arm in ("CLEAN", "DETACHED_PUSHFORWARD")
    }
    for arm, key in (
        ("CLEAN", "clean_training_packets"),
        ("DETACHED_PUSHFORWARD", "detached_pushforward_training_packets"),
    ):
        records = inherited.get(key)
        expected_compact = [
            (
                item.arm,
                item.seed,
                item.training.packet.final_file_sha256,
                item.training.checkpoint_record["sha256"],
                item.training.summary["source_set_sha256"],
            )
            for item in by_arm[arm]
        ]
        if not isinstance(records, list):
            raise TypeError(f"extension evaluation {key} is not a list")
        observed_compact = [
            (
                record.get("arm"),
                record.get("seed"),
                record.get("final_hash_manifest_sha256"),
                record.get("checkpoint_sha256"),
                record.get("source_set_sha256"),
            )
            for record in records
            if isinstance(record, Mapping)
        ]
        if observed_compact != expected_compact:
            raise ValueError(f"extension evaluation {key} binding differs")

    claim = result.get("claim_boundary")
    if (
        result.get("experiment_id") != EXPERIMENT_ID
        or result.get("status") != "complete"
        or result.get("classification") != "SCIENTIFIC_DEVELOPMENT_RESULT"
        or result.get("population_role") != "development"
        or result.get("input_manifest_payload_sha256")
        != inputs.get("canonical_payload_sha256")
        or result.get("seeds") != list(SEEDS)
        or result.get("learned_arms") != list(evaluator.LEARNED_ARMS)
        or not isinstance(claim, Mapping)
        or claim.get("development_only") is not True
        or claim.get("online_solver_calls") != 0
        or claim.get("online_defect_trigger") is not False
        or claim.get("prospective_opened") is not False
        or claim.get("sealed_opened") is not False
    ):
        raise ValueError("extension evaluation result claim boundary differs")
    snapshots = _parse_packet_npz(packet, "rollout_snapshots.npz")
    summary_rows = _parse_packet_csv(packet, "method_summary.csv")
    evaluator.parent._reverify_packet(packet)
    return packet, snapshots, summary_rows


def _sampler_value(value: str) -> int | None:
    return None if value == "" else int(value)


def _aggregate_summary_rows(rows: Sequence[Mapping[str, str]]) -> np.ndarray:
    """Median eight anchors within seed; retain three seeds for min--max bars."""

    output = np.empty((len(METHODS), len(SEEDS), len(METRIC_SPECS)), dtype=np.float64)
    for method_offset, method in enumerate(METHODS):
        for seed_offset, seed in enumerate(SEEDS):
            selected = [
                row
                for row in rows
                if row.get("arm") == method.arm
                and row.get("seed") == str(seed)
                and row.get("deployment") == method.deployment
                and row.get("state_stage") == method.state_stage
                and _sampler_value(row.get("sampler_seed", "")) == method.sampler_seed
            ]
            anchors = [int(row["anchor"]) for row in selected]
            if len(selected) != 8 or sorted(anchors) != list(
                evaluator.DEVELOPMENT_ANCHORS
            ):
                raise ValueError(
                    f"method summary lacks eight common anchors: {method.slug}:{seed}"
                )
            for metric_offset, (metric, _) in enumerate(METRIC_SPECS):
                values = np.asarray([float(row[metric]) for row in selected])
                if (
                    values.shape != (8,)
                    or np.any(np.isnan(values))
                    or np.any(np.isneginf(values))
                    or np.any(values < 0.0)
                ):
                    raise ValueError(f"method summary metric is invalid: {metric}")
                output[method_offset, seed_offset, metric_offset] = float(
                    np.median(values)
                )
    return output


def _load_exact_snapshots(
    snapshots: Mapping[str, np.ndarray], node_count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    references = np.empty((len(STATIC_HORIZONS), node_count), dtype=np.float64)
    predictions = np.full(
        (len(METHODS), len(SEEDS), len(STATIC_HORIZONS), node_count),
        np.nan,
        dtype=np.float32,
    )
    valid = np.zeros((len(METHODS), len(SEEDS), len(STATIC_HORIZONS)), dtype=np.bool_)
    for horizon_offset, horizon in enumerate(STATIC_HORIZONS):
        key = f"reference_a{ANCHOR}_h{horizon}"
        reference = np.asarray(snapshots.get(key))
        if (
            reference.shape != (node_count, 5)
            or reference.dtype != np.float64
            or not np.all(np.isfinite(reference))
        ):
            raise ValueError("exact evaluator reference snapshot differs")
        references[horizon_offset] = reference[:, FIELD_INDEX]
        for method_offset, method in enumerate(METHODS):
            for seed_offset, seed in enumerate(SEEDS):
                candidate = snapshots.get(_snapshot_key(method, seed, horizon))
                if candidate is None:
                    continue
                candidate = np.asarray(candidate)
                if candidate.shape != (node_count, 5) or candidate.dtype != np.float32:
                    raise ValueError("exact evaluator prediction snapshot differs")
                if np.all(np.isnan(candidate)):
                    continue
                if not np.all(np.isfinite(candidate)):
                    raise ValueError("exact evaluator prediction snapshot differs")
                predictions[method_offset, seed_offset, horizon_offset] = candidate[
                    :, FIELD_INDEX
                ]
                valid[method_offset, seed_offset, horizon_offset] = True

    refiner = next(method for method in METHODS if method.arm == evaluator.REFINER_ARM)
    sampler_density = np.full(
        (len(SEEDS), len(REFINER_SAMPLER_SEEDS), node_count),
        np.nan,
        dtype=np.float32,
    )
    sampler_valid = np.zeros((len(SEEDS), len(REFINER_SAMPLER_SEEDS)), dtype=np.bool_)
    for seed_offset, seed in enumerate(SEEDS):
        for sampler_offset, sampler in enumerate(REFINER_SAMPLER_SEEDS):
            candidate = snapshots.get(
                _snapshot_key(refiner, seed, 208, sampler_seed=sampler)
            )
            if candidate is None:
                continue
            candidate = np.asarray(candidate)
            if candidate.shape != (node_count, 5) or candidate.dtype != np.float32:
                raise ValueError("exact PDE-Refiner sampler snapshot differs")
            if np.all(np.isnan(candidate)):
                continue
            if not np.all(np.isfinite(candidate)):
                raise ValueError("exact PDE-Refiner sampler snapshot differs")
            sampler_density[seed_offset, sampler_offset] = candidate[:, FIELD_INDEX]
            sampler_valid[seed_offset, sampler_offset] = True
    return references, predictions, valid, sampler_density, sampler_valid


def _rollout_selected(
    *,
    method: MethodSpec,
    transition: torch.nn.Module,
    previous: torch.Tensor,
    current: torch.Tensor,
    selected_offset: int,
    geometry: Any,
    normalization: Any,
    projector: Any | None,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, int | None]:
    """Replay one selected deployment on the evaluator's eight-anchor batch."""

    if method not in METHODS:
        raise ValueError("qualitative replay method is not predeclared")
    if method.arm == "PATH_PROJECTION" and projector is None:
        raise ValueError("path projection requires the train-only projector")
    batch_size, node_count, _ = current.shape
    if not 0 <= selected_offset < batch_size:
        raise ValueError("selected anchor offset differs")
    density = np.full((len(FULL_HORIZONS), node_count), np.nan, dtype=np.float32)
    valid = np.zeros(len(FULL_HORIZONS), dtype=np.bool_)
    density[0] = current[selected_offset, :, FIELD_INDEX].detach().cpu().numpy()
    valid[0] = True
    active = torch.ones(batch_size, dtype=torch.bool, device=device)
    geometry_batch = geometry.expand(batch_size, device)
    fourier = transition.prepare_fourier_tensors(geometry_batch)  # type: ignore[attr-defined]
    device_projector = (
        projector.to(device, dtype=torch.float32) if projector is not None else None
    )
    first_nonfinite: int | None = None
    transition.eval()
    with torch.no_grad():
        for horizon in FULL_HORIZONS[1:]:
            if method.arm == evaluator.REFINER_ARM:
                step = transition.refine(  # type: ignore[attr-defined]
                    previous,
                    current,
                    geometry_batch,
                    fourier_tensors=fourier,
                )
                candidate = step.next_state
                recurrent_previous = step.recurrent_previous
                retained = active & torch.all(torch.isfinite(candidate), dim=(1, 2))
            elif method.arm == "PATH_PROJECTION":
                retained_masks: list[torch.Tensor] = []

                def apply_projection(
                    raw_state: torch.Tensor,
                    *,
                    active_now: torch.Tensor = active,
                    current_now: torch.Tensor = current,
                    masks: list[torch.Tensor] = retained_masks,
                ) -> torch.Tensor:
                    raw_finite = torch.all(torch.isfinite(raw_state), dim=(1, 2))
                    retained_now = active_now & raw_finite
                    safe_query = torch.where(
                        retained_now[:, None, None], raw_state, current_now
                    )
                    assert device_projector is not None
                    corrected = device_projector.project(safe_query).corrected_state
                    masks.append(retained_now)
                    return torch.where(
                        retained_now[:, None, None], corrected, current_now
                    )

                step = corrected_recurrent_step(
                    transition,
                    previous,
                    current,
                    geometry_batch,
                    normalization,
                    apply_projection,
                    fourier_tensors=fourier,
                )
                if len(retained_masks) != 1:
                    raise AssertionError("path projection call count differs")
                retained = retained_masks[0]
                candidate = step.corrected_next_state
                if torch.any(
                    retained & ~torch.all(torch.isfinite(candidate), dim=(1, 2))
                ):
                    raise FloatingPointError(
                        "path projection produced a nonfinite state"
                    )
                recurrent_previous = step.recurrent_previous
            else:
                recurrent_previous, candidate = recurrent_step(
                    transition,
                    previous,
                    current,
                    geometry_batch,
                    normalization,
                    fourier_tensors=fourier,
                )
                retained = active & torch.all(torch.isfinite(candidate), dim=(1, 2))

            if bool(retained[selected_offset]):
                density[horizon] = (
                    candidate[selected_offset, :, FIELD_INDEX]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32, copy=False)
                )
                valid[horizon] = True
            elif bool(active[selected_offset]) and first_nonfinite is None:
                first_nonfinite = horizon
            active = retained
            safe_candidate = torch.where(active[:, None, None], candidate, current)
            previous, current = recurrent_previous, safe_candidate
    return density, valid, first_nonfinite


def _reverify_replay_inputs(
    *,
    arguments: argparse.Namespace,
    source: Mapping[str, Any],
    dataset_sha256: str,
    calibration_packet: Any,
    pilot: Any,
    paired_bank: Any,
    extension_packets: Sequence[Any],
    inherited_packets: Sequence[Any],
    evaluation_packet: Any,
) -> None:
    _reverify_visualization_source()
    evaluator._reverify_source(source)
    checks = (
        (arguments.contract, evaluator.trainer.R0_CONTRACT_SHA256, "R0 contract"),
        (
            arguments.successor_contract,
            evaluator.trainer.SUCCESSOR_CONTRACT_SHA256,
            "successor contract",
        ),
        (
            arguments.successor_preregistration,
            evaluator.trainer.SUCCESSOR_PREREGISTRATION_SHA256,
            "successor preregistration",
        ),
        (
            arguments.extension_contract,
            evaluator.EXTENSION_CONTRACT_SHA256,
            "extension contract",
        ),
        (
            arguments.extension_preregistration,
            evaluator.EXTENSION_PREREGISTRATION_SHA256,
            "extension preregistration",
        ),
        (
            arguments.dataset_dir / "final_hash_manifest.json",
            dataset_sha256,
            "dataset final manifest",
        ),
        (
            pilot.final_path,
            pilot.final_sha256,
            "relabel pilot final manifest",
        ),
        (
            paired_bank.final_path,
            paired_bank.final_sha256,
            "paired bank final manifest",
        ),
    )
    for path, expected, label in checks:
        if evaluator._file_sha256(path) != expected:
            raise ValueError(f"{label} changed during visualization replay")
    evaluator.parent._reverify_packet(calibration_packet)
    evaluator.parent._reverify_packet(evaluation_packet)
    for packet in extension_packets:
        evaluator.parent._reverify_packet(packet.packet)
    for packet in inherited_packets:
        evaluator.parent._reverify_packet(packet.training.packet)


def replay(arguments: argparse.Namespace) -> dict[str, Any]:
    """Create the independent qualitative replay plus exact static inputs."""

    evaluator._validate_population_role(arguments.population_role)
    inputs = [
        arguments.contract,
        arguments.successor_contract,
        arguments.successor_preregistration,
        arguments.extension_contract,
        arguments.extension_preregistration,
        arguments.dataset_dir,
        arguments.calibration_dir,
        arguments.paired_bank_dir,
        arguments.relabel_pilot_final_hash_manifest,
        arguments.evaluation_dir,
        *arguments.training_dir,
        *arguments.inherited_clean_training_dir,
        *arguments.inherited_detached_training_dir,
    ]
    evaluator.parent._validate_output_disjoint(
        arguments.output_dir,
        [("visualization replay input", path) for path in inputs],
        label="corrective-extension visualization replay output",
    )
    _reverify_visualization_source()
    source = evaluator._source_snapshot()
    evaluator._reverify_source(source)

    extension = evaluator.load_extension_contract(
        arguments.extension_contract, arguments.extension_preregistration
    )
    evaluator.trainer._validate_training_contract(extension.payload())
    successor, successor_sha256, baseline = evaluator.parent._load_successor_contract(
        arguments.successor_contract,
        arguments.contract,
        arguments.successor_preregistration,
    )
    dataset_root = evaluator.parent._safe_directory(
        arguments.dataset_dir, label="visualization open dataset packet"
    )
    dataset_manifest, geometry, normalization, roles, dataset_sha256 = (
        evaluator.parent._load_dataset(dataset_root, baseline)
    )
    if dataset_sha256 != evaluator.trainer.DATASET_FINAL_SHA256 or set(roles) != {
        "train",
        "development",
    }:
        raise ValueError("visualization dataset differs or exposes a protected role")
    train_states, train_indices, _ = roles["train"]
    development_states, development_indices, _ = roles["development"]
    evaluator.parent._validate_population(
        train_states,
        train_indices,
        role="train",
        expected_frames=range(955, 1195),
        node_count=geometry.num_nodes,
    )
    evaluator.parent._validate_population(
        development_states,
        development_indices,
        role="development",
        expected_frames=range(1233, 1473),
        node_count=geometry.num_nodes,
    )
    calibration_packet, calibration, projector = evaluator.parent._load_calibration(
        arguments.calibration_dir,
        successor=successor,
        successor_sha256=successor_sha256,
        dataset_sha256=dataset_sha256,
        normalization=normalization,
        train_states=train_states,
        train_indices=train_indices,
    )
    pilot, paired_bank = evaluator._load_solver_label_provenance(
        arguments.paired_bank_dir,
        arguments.relabel_pilot_final_hash_manifest,
        expected_coordinates=geometry.native_coordinates,
        train_states=train_states,
        train_frame_indices=train_indices,
        normalization=normalization,
    )
    extension_packets = evaluator._load_training_packets(
        arguments.training_dir,
        dataset_sha256=dataset_sha256,
        dataset_payload_sha256=dataset_manifest["canonical_payload_sha256"],
        calibration_sha256=calibration_packet.final_file_sha256,
        calibration_payload_sha256=calibration["canonical_payload_sha256"],
        paired_bank_sha256=paired_bank.final_sha256,
        pilot_sha256=pilot.final_sha256,
    )
    inherited_packets = evaluator._load_inherited_baseline_packets(
        arguments.inherited_clean_training_dir,
        arguments.inherited_detached_training_dir,
        successor=successor,
        successor_sha256=successor_sha256,
        dataset_sha256=dataset_sha256,
        dataset_payload_sha256=dataset_manifest["canonical_payload_sha256"],
        calibration_sha256=calibration_packet.final_file_sha256,
        calibration_payload_sha256=calibration["canonical_payload_sha256"],
        baseline=baseline,
        geometry=geometry,
    )
    evaluation_packet, snapshots, summary_rows = _verify_evaluation_packet(
        arguments.evaluation_dir,
        extension_packets=extension_packets,
        inherited_packets=inherited_packets,
        successor_sha256=successor_sha256,
        dataset_sha256=dataset_sha256,
        dataset_payload_sha256=dataset_manifest["canonical_payload_sha256"],
        calibration_sha256=calibration_packet.final_file_sha256,
        calibration_payload_sha256=calibration["canonical_payload_sha256"],
        pilot_sha256=pilot.final_sha256,
        paired_bank_sha256=paired_bank.final_sha256,
    )
    (
        exact_reference,
        exact_predictions,
        exact_valid,
        refiner_sampler,
        refiner_sampler_valid,
    ) = _load_exact_snapshots(snapshots, geometry.num_nodes)
    summary_metrics = _aggregate_summary_rows(summary_rows)

    if tuple(evaluator.DEVELOPMENT_ANCHORS) != (
        1234,
        1238,
        1243,
        1247,
        1251,
        1256,
        1260,
        1264,
    ):
        raise ValueError("development anchor contract differs")
    positions = {int(index): offset for offset, index in enumerate(development_indices)}
    if any(
        required not in positions
        for anchor in evaluator.DEVELOPMENT_ANCHORS
        for required in (anchor - 1, anchor, anchor + FULL_HORIZONS[-1])
    ):
        raise ValueError("development role lacks the frozen qualitative replay")
    selected_offset = evaluator.DEVELOPMENT_ANCHORS.index(ANCHOR)
    previous_numpy = np.stack(
        [
            development_states[positions[anchor - 1]]
            for anchor in evaluator.DEVELOPMENT_ANCHORS
        ]
    ).astype(np.float32)
    current_numpy = np.stack(
        [
            development_states[positions[anchor]]
            for anchor in evaluator.DEVELOPMENT_ANCHORS
        ]
    ).astype(np.float32)
    frame_indices = np.asarray([ANCHOR + horizon for horizon in FULL_HORIZONS])
    reference_density = np.stack(
        [
            development_states[positions[int(index)], :, FIELD_INDEX]
            for index in frame_indices
        ]
    ).astype(np.float32)
    predicted_density = np.full(
        (len(METHODS), len(SEEDS), len(FULL_HORIZONS), geometry.num_nodes),
        np.nan,
        dtype=np.float32,
    )
    valid_mask = np.zeros(
        (len(METHODS), len(SEEDS), len(FULL_HORIZONS)), dtype=np.bool_
    )
    first_nonfinite: dict[str, int | None] = {}
    extension_lookup = {
        (packet.arm, packet.seed): packet for packet in extension_packets
    }
    inherited_lookup = {
        (packet.arm, packet.seed): packet.training for packet in inherited_packets
    }
    device = torch.device(arguments.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    for method_offset, method in enumerate(METHODS):
        for seed_offset, seed in enumerate(SEEDS):
            base_model: torch.nn.Module
            if method.packet_family == "extension":
                packet = extension_lookup[(method.arm, seed)]
                deployment = evaluator.Deployment(
                    packet, method.deployment, method.sampler_seed
                )
                base_model = evaluator._instantiate_deployment_model(
                    deployment,
                    extension=extension,
                    baseline=baseline,
                    geometry=geometry,
                    device=device,
                )
                transition: torch.nn.Module = base_model
                if method.arm == evaluator.REFINER_ARM:
                    assert method.sampler_seed is not None
                    transition = evaluator.RefinerTransitionAdapter(
                        base_model,
                        normalization,
                        sampler_seed=method.sampler_seed,
                        device=device,
                    ).to(device)
            else:
                parent_arm = "CLEAN" if method.arm == "PATH_PROJECTION" else method.arm
                packet = inherited_lookup[(parent_arm, seed)]
                base_model = evaluator.parent._instantiate_model(
                    packet, baseline=baseline, geometry=geometry, device=device
                )
                transition = base_model
            density, valid, failed = _rollout_selected(
                method=method,
                transition=transition,
                previous=torch.as_tensor(previous_numpy, device=device),
                current=torch.as_tensor(current_numpy, device=device),
                selected_offset=selected_offset,
                geometry=geometry,
                normalization=normalization,
                projector=projector if method.arm == "PATH_PROJECTION" else None,
                device=device,
            )
            predicted_density[method_offset, seed_offset] = density
            valid_mask[method_offset, seed_offset] = valid
            first_nonfinite[f"{method.slug}:{seed}"] = failed
            del transition, base_model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if not np.all(valid_mask[:, :, 0]):
        raise AssertionError("every qualitative replay must retain the initial state")
    density_state_scale = float(normalization.state_scale[FIELD_INDEX])
    coordinates = np.asarray(geometry.native_coordinates, dtype=np.float64)
    quads = np.asarray(geometry.elements[:, 1:], dtype=np.int64)
    airfoil_mask = np.asarray(geometry.boundary_one_hot[:, 1], dtype=np.uint8)

    _reverify_replay_inputs(
        arguments=arguments,
        source=source,
        dataset_sha256=dataset_sha256,
        calibration_packet=calibration_packet,
        pilot=pilot,
        paired_bank=paired_bank,
        extension_packets=extension_packets,
        inherited_packets=inherited_packets,
        evaluation_packet=evaluation_packet,
    )

    output = arguments.output_dir.resolve()
    if output.exists() or arguments.output_dir.is_symlink():
        raise FileExistsError(f"visualization replay output already exists: {output}")

    def build(staging: Path) -> dict[str, Any]:
        arrays = {
            "schema": np.asarray(REPLAY_SCHEMA),
            "qualitative_contract_sha256": np.asarray(QUALITATIVE_CONTRACT_SHA256),
            "horizons": np.asarray(FULL_HORIZONS, dtype=np.int64),
            "frame_indices": frame_indices.astype(np.int64),
            "method_slugs": np.asarray([method.slug for method in METHODS]),
            "seeds": np.asarray(SEEDS, dtype=np.int64),
            "coordinates": coordinates,
            "quads": quads,
            "airfoil_mask": airfoil_mask,
            "reference_density": reference_density,
            "predicted_density": predicted_density,
            "valid_mask": valid_mask,
            "static_horizons": np.asarray(STATIC_HORIZONS, dtype=np.int64),
            "exact_reference_density": exact_reference,
            "exact_predicted_density": exact_predictions,
            "exact_valid_mask": exact_valid,
            "refiner_sampler_density_h208": refiner_sampler,
            "refiner_sampler_valid_h208": refiner_sampler_valid,
            "density_state_scale": np.asarray(density_state_scale, dtype=np.float64),
            "metric_names": np.asarray([metric for metric, _ in METRIC_SPECS]),
            "per_seed_summary_metrics": summary_metrics,
        }
        np.savez_compressed(staging / REPLAY_FILE, **arrays)
        replay_record = evaluator.parent._file_record(staging / REPLAY_FILE, staging)
        manifest: dict[str, Any] = {
            "schema": REPLAY_SCHEMA,
            "status": "complete",
            "classification": "VISUALIZATION_ONLY_REPLAY",
            "experiment_id": EXPERIMENT_ID,
            "qualitative_contract": QUALITATIVE_CONTRACT,
            "qualitative_contract_sha256": QUALITATIVE_CONTRACT_SHA256,
            "population_role": "development",
            "anchor": ANCHOR,
            "seeds": list(SEEDS),
            "methods": _method_records(),
            "online_solver_calls": False,
            "online_defect_trigger": False,
            "prospective_opened": False,
            "sealed_opened": False,
            "first_nonfinite_horizon": first_nonfinite,
            "source": {
                "visualization_script": dict(VISUALIZATION_SOURCE_AT_IMPORT),
                "evaluator_source_set_sha256": source["source_set_sha256"],
            },
            "inputs": {
                "extension_contract_sha256": evaluator.EXTENSION_CONTRACT_SHA256,
                "successor_contract_sha256": successor_sha256,
                "dataset_final_hash_manifest_sha256": dataset_sha256,
                "calibration_final_hash_manifest_sha256": (
                    calibration_packet.final_file_sha256
                ),
                "relabel_pilot_final_hash_manifest_sha256": pilot.final_sha256,
                "paired_bank_final_hash_manifest_sha256": paired_bank.final_sha256,
                "evaluation_final_hash_manifest_sha256": (
                    evaluation_packet.final_file_sha256
                ),
                "extension_training_packets": _expected_extension_training_records(
                    extension_packets
                ),
                "inherited_training_packets": [
                    {
                        "arm": packet.arm,
                        "seed": packet.seed,
                        "final_hash_manifest_sha256": (
                            packet.training.packet.final_file_sha256
                        ),
                        "checkpoint_sha256": packet.training.checkpoint_record[
                            "sha256"
                        ],
                        "source_set_sha256": packet.training.summary[
                            "source_set_sha256"
                        ],
                    }
                    for packet in inherited_packets
                ],
            },
            "replay_file": replay_record,
            "array_sha256": {
                key: _array_sha256(value) for key, value in arrays.items()
            },
            "claim_boundary": _claim_boundary(),
        }
        manifest["canonical_payload_sha256"] = evaluator._canonical_sha256(manifest)
        evaluator._write_json(staging / REPLAY_MANIFEST_FILE, manifest)
        return manifest

    def verify(staging: Path, manifest: dict[str, Any]) -> None:
        del manifest
        _reverify_replay_inputs(
            arguments=arguments,
            source=source,
            dataset_sha256=dataset_sha256,
            calibration_packet=calibration_packet,
            pilot=pilot,
            paired_bank=paired_bank,
            extension_packets=extension_packets,
            inherited_packets=inherited_packets,
            evaluation_packet=evaluation_packet,
        )
        load_replay(staging)

    return _publish_directory_atomically(output, build, before_publish=verify)


def _validate_replay_arrays(arrays: Mapping[str, np.ndarray]) -> None:
    required = {
        "schema",
        "qualitative_contract_sha256",
        "horizons",
        "frame_indices",
        "method_slugs",
        "seeds",
        "coordinates",
        "quads",
        "airfoil_mask",
        "reference_density",
        "predicted_density",
        "valid_mask",
        "static_horizons",
        "exact_reference_density",
        "exact_predicted_density",
        "exact_valid_mask",
        "refiner_sampler_density_h208",
        "refiner_sampler_valid_h208",
        "density_state_scale",
        "metric_names",
        "per_seed_summary_metrics",
    }
    if set(arrays) != required:
        raise ValueError("extension visualization replay array inventory differs")
    if arrays["schema"].shape != () or arrays["schema"].item() != REPLAY_SCHEMA:
        raise ValueError("extension visualization replay schema differs")
    if (
        arrays["qualitative_contract_sha256"].shape != ()
        or arrays["qualitative_contract_sha256"].item() != QUALITATIVE_CONTRACT_SHA256
    ):
        raise ValueError("extension visualization qualitative contract differs")
    coordinates = arrays["coordinates"]
    if coordinates.ndim != 2 or coordinates.shape[1:] != (2,):
        raise ValueError("extension visualization coordinate shape differs")
    node_count = coordinates.shape[0]
    shapes = {
        "horizons": (len(FULL_HORIZONS),),
        "frame_indices": (len(FULL_HORIZONS),),
        "method_slugs": (len(METHODS),),
        "seeds": (len(SEEDS),),
        "coordinates": (node_count, 2),
        "airfoil_mask": (node_count,),
        "reference_density": (len(FULL_HORIZONS), node_count),
        "predicted_density": (
            len(METHODS),
            len(SEEDS),
            len(FULL_HORIZONS),
            node_count,
        ),
        "valid_mask": (len(METHODS), len(SEEDS), len(FULL_HORIZONS)),
        "static_horizons": (len(STATIC_HORIZONS),),
        "exact_reference_density": (len(STATIC_HORIZONS), node_count),
        "exact_predicted_density": (
            len(METHODS),
            len(SEEDS),
            len(STATIC_HORIZONS),
            node_count,
        ),
        "exact_valid_mask": (len(METHODS), len(SEEDS), len(STATIC_HORIZONS)),
        "refiner_sampler_density_h208": (
            len(SEEDS),
            len(REFINER_SAMPLER_SEEDS),
            node_count,
        ),
        "refiner_sampler_valid_h208": (len(SEEDS), len(REFINER_SAMPLER_SEEDS)),
        "metric_names": (len(METRIC_SPECS),),
        "per_seed_summary_metrics": (len(METHODS), len(SEEDS), len(METRIC_SPECS)),
    }
    if any(arrays[name].shape != shape for name, shape in shapes.items()):
        raise ValueError("extension visualization replay array shape differs")
    exact_values = {
        "horizons": np.asarray(FULL_HORIZONS),
        "frame_indices": ANCHOR + np.asarray(FULL_HORIZONS),
        "method_slugs": np.asarray([method.slug for method in METHODS]),
        "seeds": np.asarray(SEEDS),
        "static_horizons": np.asarray(STATIC_HORIZONS),
        "metric_names": np.asarray([metric for metric, _ in METRIC_SPECS]),
    }
    if any(
        not np.array_equal(arrays[name], value) for name, value in exact_values.items()
    ):
        raise ValueError("extension visualization replay selection differs")
    expected_dtypes = {
        "horizons": np.int64,
        "frame_indices": np.int64,
        "seeds": np.int64,
        "coordinates": np.float64,
        "quads": np.int64,
        "airfoil_mask": np.uint8,
        "reference_density": np.float32,
        "predicted_density": np.float32,
        "valid_mask": np.bool_,
        "static_horizons": np.int64,
        "exact_reference_density": np.float64,
        "exact_predicted_density": np.float32,
        "exact_valid_mask": np.bool_,
        "refiner_sampler_density_h208": np.float32,
        "refiner_sampler_valid_h208": np.bool_,
        "density_state_scale": np.float64,
        "per_seed_summary_metrics": np.float64,
    }
    if any(arrays[name].dtype != dtype for name, dtype in expected_dtypes.items()):
        raise ValueError("extension visualization replay array dtype differs")
    quads = arrays["quads"]
    if (
        quads.ndim != 2
        or quads.shape[1] != 4
        or quads.size == 0
        or np.any(quads < 0)
        or np.any(quads >= node_count)
    ):
        raise ValueError("extension visualization native quads differ")
    if (
        not np.all(np.isfinite(coordinates))
        or not np.all(np.isfinite(arrays["reference_density"]))
        or not np.all(np.isfinite(arrays["exact_reference_density"]))
        or arrays["density_state_scale"].shape != ()
        or not np.isfinite(arrays["density_state_scale"].item())
        or arrays["density_state_scale"].item() <= 0
        or np.count_nonzero(arrays["airfoil_mask"]) < 3
        or not np.all(np.isin(arrays["airfoil_mask"], (0, 1)))
    ):
        raise ValueError(
            "extension visualization finite geometry/data contract differs"
        )
    summary_metrics = arrays["per_seed_summary_metrics"]
    if (
        np.any(np.isnan(summary_metrics))
        or np.any(np.isneginf(summary_metrics))
        or np.any(summary_metrics < 0.0)
    ):
        raise ValueError("extension visualization summary metric contract differs")
    for values_name, mask_name in (
        ("predicted_density", "valid_mask"),
        ("exact_predicted_density", "exact_valid_mask"),
        ("refiner_sampler_density_h208", "refiner_sampler_valid_h208"),
    ):
        values = arrays[values_name]
        mask = arrays[mask_name]
        if not np.all(np.isfinite(values[mask])) or not np.all(np.isnan(values[~mask])):
            raise ValueError(f"{values_name} validity mask differs")
    if not np.all(arrays["valid_mask"][:, :, 0]):
        raise ValueError("extension visualization initial-state validity differs")


def _validate_first_nonfinite(
    manifest: Mapping[str, Any], valid_mask: np.ndarray
) -> None:
    records = manifest.get("first_nonfinite_horizon")
    expected_keys = {f"{method.slug}:{seed}" for method in METHODS for seed in SEEDS}
    if not isinstance(records, Mapping) or set(records) != expected_keys:
        raise ValueError("extension visualization nonfinite inventory differs")
    for method_offset, method in enumerate(METHODS):
        for seed_offset, seed in enumerate(SEEDS):
            invalid = np.flatnonzero(~valid_mask[method_offset, seed_offset])
            expected = int(invalid[0]) if invalid.size else None
            observed = records[f"{method.slug}:{seed}"]
            valid_observed = (
                observed is None
                if expected is None
                else (isinstance(observed, int) and not isinstance(observed, bool))
            )
            if not valid_observed or observed != expected:
                raise ValueError(
                    "extension visualization first-nonfinite record differs"
                )


def load_replay(
    root: Path,
    *,
    expected_visualization_source_sha256: str | None = None,
) -> ReplayBundle:
    """Load and fully validate one portable replay packet."""

    if root.is_symlink():
        raise ValueError("extension visualization replay root is absent or aliased")
    root = root.resolve()
    if not root.is_dir():
        raise ValueError("extension visualization replay root is absent or aliased")
    entries = list(root.iterdir())
    if {entry.name for entry in entries} != {REPLAY_FILE, REPLAY_MANIFEST_FILE} or any(
        entry.is_symlink() or not entry.is_file() for entry in entries
    ):
        raise ValueError("extension visualization replay inventory differs")
    manifest_path = root / REPLAY_MANIFEST_FILE
    payload = manifest_path.read_bytes()
    manifest = json.loads(payload)
    if not isinstance(manifest, dict):
        raise TypeError("extension visualization replay manifest is not an object")
    if set(manifest) != {
        "schema",
        "status",
        "classification",
        "experiment_id",
        "qualitative_contract",
        "qualitative_contract_sha256",
        "population_role",
        "anchor",
        "seeds",
        "methods",
        "online_solver_calls",
        "online_defect_trigger",
        "prospective_opened",
        "sealed_opened",
        "first_nonfinite_horizon",
        "source",
        "inputs",
        "replay_file",
        "array_sha256",
        "claim_boundary",
        "canonical_payload_sha256",
    }:
        raise ValueError("extension visualization replay manifest schema differs")
    unsigned = dict(manifest)
    digest = unsigned.pop("canonical_payload_sha256", None)
    expected_source = (
        VISUALIZATION_SOURCE_AT_IMPORT["sha256"]
        if expected_visualization_source_sha256 is None
        else expected_visualization_source_sha256
    )
    _validate_replay_provenance(
        manifest, expected_visualization_source_sha256=expected_source
    )
    if (
        digest != evaluator._canonical_sha256(unsigned)
        or manifest.get("schema") != REPLAY_SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("classification") != "VISUALIZATION_ONLY_REPLAY"
        or manifest.get("experiment_id") != EXPERIMENT_ID
        or manifest.get("qualitative_contract") != QUALITATIVE_CONTRACT
        or manifest.get("qualitative_contract_sha256") != QUALITATIVE_CONTRACT_SHA256
        or manifest.get("population_role") != "development"
        or manifest.get("anchor") != ANCHOR
        or manifest.get("seeds") != list(SEEDS)
        or manifest.get("methods") != _method_records()
        or manifest.get("claim_boundary") != _claim_boundary()
        or manifest.get("online_solver_calls") is not False
        or manifest.get("online_defect_trigger") is not False
        or manifest.get("prospective_opened") is not False
        or manifest.get("sealed_opened") is not False
    ):
        raise ValueError(
            "extension visualization replay selection or provenance differs"
        )
    replay_path = root / REPLAY_FILE
    replay_record = evaluator.parent._file_record(replay_path, root)
    if replay_record != manifest.get("replay_file"):
        raise ValueError("extension visualization replay file binding differs")
    with np.load(replay_path, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    _validate_replay_arrays(arrays)
    _validate_first_nonfinite(manifest, arrays["valid_mask"])
    hashes = manifest.get("array_sha256")
    if not isinstance(hashes, Mapping) or hashes != {
        key: _array_sha256(value) for key, value in arrays.items()
    }:
        raise ValueError("extension visualization replay array hash differs")
    if evaluator.parent._file_record(replay_path, root) != replay_record:
        raise ValueError("extension visualization replay changed while loading")
    if manifest_path.read_bytes() != payload:
        raise ValueError(
            "extension visualization replay manifest changed while loading"
        )
    return ReplayBundle(
        horizons=arrays["horizons"],
        frame_indices=arrays["frame_indices"],
        coordinates=arrays["coordinates"],
        quads=arrays["quads"],
        airfoil_mask=arrays["airfoil_mask"],
        reference_density=arrays["reference_density"],
        predicted_density=arrays["predicted_density"],
        valid_mask=arrays["valid_mask"],
        static_horizons=arrays["static_horizons"],
        exact_reference_density=arrays["exact_reference_density"],
        exact_predicted_density=arrays["exact_predicted_density"],
        exact_valid_mask=arrays["exact_valid_mask"],
        refiner_sampler_density_h208=arrays["refiner_sampler_density_h208"],
        refiner_sampler_valid_h208=arrays["refiner_sampler_valid_h208"],
        density_state_scale=float(arrays["density_state_scale"]),
        per_seed_summary_metrics=arrays["per_seed_summary_metrics"],
        manifest=manifest,
    )


def _visible_quads(bundle: ReplayBundle) -> np.ndarray:
    vertices = bundle.coordinates[bundle.quads]
    overlaps = (
        (np.max(vertices[:, :, 0], axis=1) >= X_LIMITS[0])
        & (np.min(vertices[:, :, 0], axis=1) <= X_LIMITS[1])
        & (np.max(vertices[:, :, 1], axis=1) >= Y_LIMITS[0])
        & (np.min(vertices[:, :, 1], axis=1) <= Y_LIMITS[1])
    )
    selected = bundle.quads[overlaps]
    if selected.size == 0:
        raise ValueError("fixed NACA view contains no native quadrilaterals")
    return selected


def _quad_face_values(values: np.ndarray, quads: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim != 1 or np.any(quads < 0) or np.any(quads >= values.size):
        raise ValueError("native node values and quadrilaterals do not align")
    return np.mean(values[quads], axis=1)


def _airfoil_outline(bundle: ReplayBundle) -> np.ndarray:
    airfoil = bundle.coordinates[bundle.airfoil_mask.astype(bool)]
    centroid = np.mean(airfoil, axis=0)
    angles = np.arctan2(airfoil[:, 1] - centroid[1], airfoil[:, 0] - centroid[0])
    ordered = airfoil[np.argsort(angles)]
    return np.vstack((ordered, ordered[0]))


def _format_field_axis(axis: Any, outline: np.ndarray) -> None:
    axis.set_xlim(*X_LIMITS)
    axis.set_ylim(*Y_LIMITS)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xticks([])
    axis.set_yticks([])
    axis.plot(outline[:, 0], outline[:, 1], color="#1A1A1A", linewidth=0.65)


def _error_cmap_and_norm(error_max: float):
    from matplotlib.colors import LinearSegmentedColormap, SymLogNorm

    linthresh = max(min(SIGNED_ERROR_LINEAR_THRESHOLD, 0.02 * error_max), 1.0e-12)
    cmap = LinearSegmentedColormap.from_list(
        "okabe_ito_blue_white_orange", ("#0072B2", "#FFFFFF", "#E69F00")
    )
    norm = SymLogNorm(
        linthresh=linthresh,
        linscale=1.0,
        vmin=-error_max,
        vmax=error_max,
        base=10,
    )
    return cmap, norm, linthresh


def _plot_style() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _color_limits(
    bundle: ReplayBundle, quads: np.ndarray, *, exact: bool
) -> dict[str, tuple[float, float]]:
    nodes = np.unique(quads)
    error_max = 1.0e-12
    if exact:
        reference = bundle.exact_reference_density[:, nodes]
        for horizon_offset in range(len(STATIC_HORIZONS)):
            exact_reference = bundle.exact_reference_density[horizon_offset]
            for method_offset in range(len(METHODS)):
                if not np.all(
                    bundle.exact_valid_mask[method_offset, :, horizon_offset]
                ):
                    continue
                signed = np.median(
                    (
                        bundle.exact_predicted_density[method_offset, :, horizon_offset]
                        - exact_reference[None]
                    )
                    / bundle.density_state_scale,
                    axis=0,
                )
                face_values = _quad_face_values(signed, quads)
                error_max = max(error_max, float(np.max(np.abs(face_values))))
        exact_reference = bundle.exact_reference_density[-1]
        for seed_offset in range(len(SEEDS)):
            for sampler_offset in range(len(REFINER_SAMPLER_SEEDS)):
                if not bundle.refiner_sampler_valid_h208[seed_offset, sampler_offset]:
                    continue
                signed = (
                    bundle.refiner_sampler_density_h208[seed_offset, sampler_offset]
                    - exact_reference
                ) / bundle.density_state_scale
                face_values = _quad_face_values(signed, quads)
                error_max = max(error_max, float(np.max(np.abs(face_values))))
    else:
        reference = bundle.reference_density[:, nodes]
        for method_offset in range(len(METHODS)):
            for seed_offset in range(len(SEEDS)):
                for horizon in np.flatnonzero(
                    bundle.valid_mask[method_offset, seed_offset]
                ):
                    signed = (
                        bundle.predicted_density[method_offset, seed_offset, horizon]
                        - bundle.reference_density[horizon]
                    ) / bundle.density_state_scale
                    face_values = _quad_face_values(signed, quads)
                    error_max = max(error_max, float(np.max(np.abs(face_values))))
    density_min = float(np.min(reference))
    density_max = float(np.max(reference))
    if not density_max > density_min:
        density_max = density_min + 1.0e-12
    return {
        "density": (density_min, density_max),
        "signed_error": (-error_max, error_max),
    }


def _field_artist(
    axis: Any, vertices: np.ndarray, values: np.ndarray, *, cmap: Any, norm: Any
):
    from matplotlib.collections import PolyCollection

    artist = PolyCollection(
        vertices,
        array=np.ma.masked_invalid(values),
        cmap=cmap,
        norm=norm,
        edgecolors="none",
        antialiased=False,
        rasterized=True,
    )
    axis.add_collection(artist)
    return artist


def _method_comparison_figure(
    bundle: ReplayBundle,
    *,
    horizon: int,
    quads: np.ndarray,
    limits: Mapping[str, tuple[float, float]],
):
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    if horizon not in STATIC_HORIZONS:
        raise ValueError("static method-comparison horizon is not frozen")
    offset = STATIC_HORIZONS.index(horizon)
    reference = bundle.exact_reference_density[offset]
    vertices = bundle.coordinates[quads]
    outline = _airfoil_outline(bundle)
    density_norm = Normalize(*limits["density"])
    error_cmap, error_norm, _ = _error_cmap_and_norm(limits["signed_error"][1])
    figure, axes = plt.subplots(2, 5, figsize=(14.0, 5.3), constrained_layout=True)
    reference_artist = _field_artist(
        axes.flat[0],
        vertices,
        _quad_face_values(reference, quads),
        cmap="viridis",
        norm=density_norm,
    )
    del reference_artist
    axes.flat[0].set_title("Reference density")
    _format_field_axis(axes.flat[0], outline)
    for axis, method_offset, method in zip(
        axes.flat[1:], range(len(METHODS)), METHODS, strict=True
    ):
        valid = bundle.exact_valid_mask[method_offset, :, offset]
        if np.all(valid):
            signed = (
                bundle.exact_predicted_density[method_offset, :, offset]
                - reference[None]
            ) / bundle.density_state_scale
            values = np.median(signed, axis=0)
            face_values = _quad_face_values(values, quads)
        else:
            face_values = np.full(quads.shape[0], np.nan)
            axis.text(
                0.5,
                0.5,
                f"finite seeds: {int(np.count_nonzero(valid))}/3",
                transform=axis.transAxes,
                ha="center",
                va="center",
            )
        _field_artist(axis, vertices, face_values, cmap=error_cmap, norm=error_norm)
        axis.set_title(method.display)
        _format_field_axis(axis, outline)
    figure.colorbar(
        ScalarMappable(norm=density_norm, cmap="viridis"),
        ax=[axes.flat[0]],
        label=r"Density $\rho$",
        fraction=0.045,
        pad=0.02,
    )
    figure.colorbar(
        ScalarMappable(norm=error_norm, cmap=error_cmap),
        ax=list(axes.flat[1:]),
        label=r"Median signed normalized density error $(\hat\rho-\rho)/s_\rho$",
        fraction=0.018,
        pad=0.012,
    )
    return figure


def _refiner_sampler_figure(
    bundle: ReplayBundle,
    *,
    quads: np.ndarray,
    limits: Mapping[str, tuple[float, float]],
):
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable

    vertices = bundle.coordinates[quads]
    outline = _airfoil_outline(bundle)
    cmap, norm, _ = _error_cmap_and_norm(limits["signed_error"][1])
    reference = bundle.exact_reference_density[-1]
    figure, axes = plt.subplots(3, 3, figsize=(8.8, 7.0), constrained_layout=True)
    for row, seed in enumerate(SEEDS):
        for column, sampler in enumerate(REFINER_SAMPLER_SEEDS):
            axis = axes[row, column]
            if bundle.refiner_sampler_valid_h208[row, column]:
                signed = (
                    bundle.refiner_sampler_density_h208[row, column] - reference
                ) / bundle.density_state_scale
                values = _quad_face_values(signed, quads)
            else:
                values = np.full(quads.shape[0], np.nan)
                axis.text(0.5, 0.5, "nonfinite", transform=axis.transAxes, ha="center")
            _field_artist(axis, vertices, values, cmap=cmap, norm=norm)
            _format_field_axis(axis, outline)
            if row == 0:
                axis.set_title(f"sampler {sampler}")
            if column == 0:
                axis.set_ylabel(f"PCNO seed {seed}")
    figure.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        ax=list(axes.flat),
        label=r"Signed normalized density error $(\hat\rho-\rho)/s_\rho$",
        fraction=0.02,
        pad=0.012,
    )
    return figure


def _rollout_path_roughness_figure(bundle: ReplayBundle):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(12.8, 4.5), constrained_layout=True)
    y = np.arange(len(METHODS))[::-1]
    for metric_offset, ((_, label), axis) in enumerate(
        zip(METRIC_SPECS, axes, strict=True)
    ):
        values = bundle.per_seed_summary_metrics[:, :, metric_offset]
        center = np.median(values, axis=1)
        lower = np.min(values, axis=1)
        upper = np.max(values, axis=1)
        finite = values[np.isfinite(values)]
        positive = finite[finite > 0]
        maximum = max(float(np.max(finite)) if finite.size else 0.0, 1.0e-12)
        linthresh = max(
            float(np.min(positive)) * 0.25 if positive.size else 1.0e-12,
            maximum * 1.0e-6,
        )
        axis.set_xscale("symlog", linthresh=linthresh, base=10)
        infinity_cap = maximum * 2.0
        for offset, color in enumerate(METHOD_COLORS):
            finite_seed_values = values[offset, np.isfinite(values[offset])]
            if np.isfinite(center[offset]):
                finite_upper = float(np.max(finite_seed_values))
                axis.errorbar(
                    center[offset],
                    y[offset],
                    xerr=np.asarray(
                        [
                            [center[offset] - lower[offset]],
                            [finite_upper - center[offset]],
                        ]
                    ),
                    fmt="o",
                    color=color,
                    ecolor=color,
                    capsize=2.5,
                    markersize=4.5,
                )
            if np.isposinf(upper[offset]):
                start = (
                    float(np.max(finite_seed_values))
                    if finite_seed_values.size
                    else infinity_cap
                )
                axis.plot(
                    [start, infinity_cap],
                    [y[offset], y[offset]],
                    color=color,
                    linestyle=":",
                    linewidth=0.8,
                )
                axis.plot(
                    infinity_cap,
                    y[offset],
                    marker=">",
                    color=color,
                    markersize=5.0,
                )
                if np.isposinf(center[offset]):
                    axis.annotate(
                        r"median $\infty$",
                        (infinity_cap, y[offset]),
                        xytext=(-3, 5),
                        textcoords="offset points",
                        ha="right",
                        fontsize=6.5,
                        color=color,
                    )
        if np.any(np.isposinf(values)):
            axis.set_xlim(left=0.0, right=infinity_cap * 1.15)
        axis.set_xlabel(label)
        axis.grid(axis="x", color="#DDDDDD", linewidth=0.6)
        axis.set_ylim(-0.7, len(METHODS) - 0.3)
        if metric_offset == 0:
            axis.set_yticks(y, [method.display for method in METHODS])
        else:
            axis.set_yticks(y, [])
    return figure


def animation_horizons(frame_stride: int) -> tuple[int, ...]:
    if isinstance(frame_stride, bool) or frame_stride < 1:
        raise ValueError("animation frame stride must be a positive integer")
    selected = list(range(FULL_HORIZONS[0], FULL_HORIZONS[-1] + 1, frame_stride))
    if selected[-1] != FULL_HORIZONS[-1]:
        selected.append(FULL_HORIZONS[-1])
    return tuple(selected)


def _animation_figure(
    bundle: ReplayBundle,
    *,
    seed_offset: int,
    quads: np.ndarray,
    limits: Mapping[str, tuple[float, float]],
):
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    vertices = bundle.coordinates[quads]
    outline = _airfoil_outline(bundle)
    density_norm = Normalize(*limits["density"])
    error_cmap, error_norm, _ = _error_cmap_and_norm(limits["signed_error"][1])
    figure, axes = plt.subplots(2, 5, figsize=(14.0, 5.5), constrained_layout=True)
    reference_artist = _field_artist(
        axes.flat[0],
        vertices,
        np.zeros(quads.shape[0]),
        cmap="viridis",
        norm=density_norm,
    )
    axes.flat[0].set_title("Reference density")
    _format_field_axis(axes.flat[0], outline)
    error_artists = []
    invalid_labels = []
    for axis, method in zip(axes.flat[1:], METHODS, strict=True):
        artist = _field_artist(
            axis,
            vertices,
            np.zeros(quads.shape[0]),
            cmap=error_cmap,
            norm=error_norm,
        )
        axis.set_title(method.display)
        _format_field_axis(axis, outline)
        invalid = axis.text(
            0.5,
            0.5,
            "nonfinite",
            transform=axis.transAxes,
            ha="center",
            va="center",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85},
        )
        invalid.set_visible(False)
        error_artists.append(artist)
        invalid_labels.append(invalid)
    figure.colorbar(
        ScalarMappable(norm=density_norm, cmap="viridis"),
        ax=[axes.flat[0]],
        label=r"Density $\rho$",
        fraction=0.045,
        pad=0.02,
    )
    figure.colorbar(
        ScalarMappable(norm=error_norm, cmap=error_cmap),
        ax=list(axes.flat[1:]),
        label=r"Signed normalized density error $(\hat\rho-\rho)/s_\rho$",
        fraction=0.018,
        pad=0.012,
    )
    frame_label = figure.text(0.985, 0.018, "", ha="right", va="bottom")
    figure.text(
        0.5,
        0.012,
        "Independent qualitative replay; not an exact long-horizon reproduction of the evaluator.",
        ha="center",
        fontsize=8,
    )

    def update(horizon: int):
        reference = bundle.reference_density[horizon]
        reference_artist.set_array(_quad_face_values(reference, quads))
        for method_offset, (artist, invalid) in enumerate(
            zip(error_artists, invalid_labels, strict=True)
        ):
            is_valid = bool(bundle.valid_mask[method_offset, seed_offset, horizon])
            if is_valid:
                signed = (
                    bundle.predicted_density[method_offset, seed_offset, horizon]
                    - reference
                ) / bundle.density_state_scale
                artist.set_array(_quad_face_values(signed, quads))
            else:
                artist.set_array(np.ma.masked_all(quads.shape[0]))
            invalid.set_visible(not is_valid)
        frame_label.set_text(f"seed {SEEDS[seed_offset]}   h = {horizon:03d}")
        return [reference_artist, *error_artists, *invalid_labels, frame_label]

    update(0)
    return figure, update


def _publish_directory_atomically(
    output: Path,
    build: Callable[[Path], dict[str, Any]],
    *,
    before_publish: Callable[[Path, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"visualization output already exists: {output}")
    if not output.parent.is_dir() or output.parent.is_symlink():
        raise ValueError("visualization output parent is absent or aliased")
    staging = evaluator.parent._make_staging_directory(
        output.parent, prefix=f".{output.name}.visualization-staging-"
    )
    owned = True
    try:
        manifest = build(staging)
        if before_publish is not None:
            before_publish(staging, manifest)
        evaluator.parent._rename_directory_no_replace(staging, output)
        owned = False
    except Exception:
        if owned:
            shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def _verify_render_stage(staging: Path, manifest: Mapping[str, Any]) -> None:
    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping):
        raise TypeError("visualization output records are not an object")
    expected = set(outputs) | {RENDER_MANIFEST_FILE}
    entries = list(staging.iterdir())
    if {entry.name for entry in entries} != expected or any(
        entry.is_symlink() or not entry.is_file() for entry in entries
    ):
        raise ValueError("visualization staged inventory differs")
    for name, record in outputs.items():
        if evaluator.parent._file_record(staging / name, staging) != record:
            raise ValueError(f"visualization staged output changed: {name}")
    path = staging / RENDER_MANIFEST_FILE
    payload = path.read_bytes()
    loaded = json.loads(payload)
    if loaded != dict(manifest):
        raise ValueError("visualization staged manifest content differs")
    unsigned = dict(loaded)
    digest = unsigned.pop("canonical_payload_sha256", None)
    if digest != evaluator._canonical_sha256(unsigned):
        raise ValueError("visualization staged manifest self hash differs")
    if path.read_bytes() != payload:
        raise ValueError("visualization staged manifest changed while verifying")


def render(arguments: argparse.Namespace) -> dict[str, Any]:
    """Render the frozen exact PDFs and three qualitative seed animations."""

    if arguments.output_dir.exists() or arguments.output_dir.is_symlink():
        raise FileExistsError(
            f"visualization output already exists: {arguments.output_dir}"
        )
    evaluator.parent._validate_output_disjoint(
        arguments.output_dir,
        [("extension visualization replay input", arguments.replay_dir)],
        label="corrective-extension visualization render output",
    )
    _reverify_visualization_source()
    import matplotlib as mpl

    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter, FuncAnimation

    if arguments.ffmpeg is not None:
        if arguments.ffmpeg.is_symlink():
            raise ValueError("ffmpeg path is absent or aliased")
        ffmpeg = arguments.ffmpeg.resolve()
        if not ffmpeg.is_file():
            raise ValueError("ffmpeg path is absent or aliased")
        mpl.rcParams["animation.ffmpeg_path"] = str(ffmpeg)
    if not FFMpegWriter.isAvailable():
        raise RuntimeError("Matplotlib cannot locate ffmpeg")
    if arguments.replay_dir.is_symlink():
        raise ValueError("extension visualization replay root is absent or aliased")
    replay_root = arguments.replay_dir.resolve()
    replay_manifest_record = evaluator.parent._file_record(
        replay_root / REPLAY_MANIFEST_FILE, replay_root
    )
    replay_file_record = evaluator.parent._file_record(
        replay_root / REPLAY_FILE, replay_root
    )
    bundle = load_replay(
        arguments.replay_dir,
        expected_visualization_source_sha256=arguments.expected_replay_producer_sha256,
    )
    if (
        evaluator.parent._file_record(replay_root / REPLAY_MANIFEST_FILE, replay_root)
        != replay_manifest_record
        or evaluator.parent._file_record(replay_root / REPLAY_FILE, replay_root)
        != replay_file_record
    ):
        raise ValueError("visualization replay changed immediately after loading")
    _plot_style()
    quads = _visible_quads(bundle)
    exact_limits = _color_limits(bundle, quads, exact=True)
    qualitative_limits = _color_limits(bundle, quads, exact=False)
    _, _, exact_linthresh = _error_cmap_and_norm(exact_limits["signed_error"][1])
    _, _, qualitative_linthresh = _error_cmap_and_norm(
        qualitative_limits["signed_error"][1]
    )
    frames = animation_horizons(arguments.frame_stride)
    output = arguments.output_dir

    captions = {
        **{
            f"corrective_exposure_targets_h{horizon:03d}.pdf": (
                "Reference density and the nine predeclared deployed-map signed "
                "normalized density errors at the frozen horizon. Error panels use "
                "pointwise medians across the three model seeds and exact evaluator snapshots."
            )
            for horizon in STATIC_HORIZONS
        },
        "rollout_path_roughness.pdf": (
            "Rollout error, empirical train-path proximity proxy, and spatial "
            "roughness. Each seed is first aggregated by the median over eight "
            "development anchors; dots and bars show the median and descriptive "
            "minimum--maximum across three seeds, not confidence intervals. "
            "A right triangle denotes an infinite seed maximum. "
            "Path projection's corrected path distance is zero by construction."
        ),
        "pderefiner_sampler_h208.pdf": (
            "Signed normalized density error for all three registered PDE-Refiner "
            "sampler seeds and all three model seeds at h=208, using exact evaluator snapshots."
        ),
        **{
            f"signed_density_error_seed{seed}.mp4": (
                "Reference density plus nine signed normalized density-error fields "
                "from an independent qualitative replay; not an exact long-horizon evaluator reproduction."
            )
            for seed in SEEDS
        },
    }

    def build(staging: Path) -> dict[str, Any]:
        names: list[str] = []
        for horizon in STATIC_HORIZONS:
            figure = _method_comparison_figure(
                bundle, horizon=horizon, quads=quads, limits=exact_limits
            )
            name = f"corrective_exposure_targets_h{horizon:03d}.pdf"
            figure.savefig(staging / name, format="pdf", dpi=300)
            plt.close(figure)
            names.append(name)
        figure = _rollout_path_roughness_figure(bundle)
        name = "rollout_path_roughness.pdf"
        figure.savefig(staging / name, format="pdf", dpi=300)
        plt.close(figure)
        names.append(name)
        figure = _refiner_sampler_figure(bundle, quads=quads, limits=exact_limits)
        name = "pderefiner_sampler_h208.pdf"
        figure.savefig(staging / name, format="pdf", dpi=300)
        plt.close(figure)
        names.append(name)
        for seed_offset, seed in enumerate(SEEDS):
            figure, update = _animation_figure(
                bundle,
                seed_offset=seed_offset,
                quads=quads,
                limits=qualitative_limits,
            )
            animation = FuncAnimation(
                figure,
                update,
                frames=frames,
                interval=1000.0 / arguments.fps,
                blit=False,
                repeat=True,
            )
            name = f"signed_density_error_seed{seed}.mp4"
            writer = FFMpegWriter(
                fps=arguments.fps,
                codec="libx264",
                bitrate=arguments.bitrate,
                extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
            )
            animation.save(staging / name, writer=writer, dpi=arguments.dpi)
            plt.close(figure)
            names.append(name)
        manifest: dict[str, Any] = {
            "schema": RENDER_SCHEMA,
            "status": "complete",
            "classification": "VISUALIZATION_ONLY",
            "experiment_id": EXPERIMENT_ID,
            "qualitative_contract_sha256": QUALITATIVE_CONTRACT_SHA256,
            "online_solver_calls": False,
            "online_defect_trigger": False,
            "prospective_opened": False,
            "sealed_opened": False,
            "visualization_script": dict(VISUALIZATION_SOURCE_AT_IMPORT),
            "replay_manifest": replay_manifest_record,
            "replay_file": replay_file_record,
            "outputs": {
                name: evaluator.parent._file_record(staging / name, staging)
                for name in names
            },
            "captions": captions,
            "rendering": {
                "paper_figures_have_no_overall_titles": True,
                "spatial_representation": "flat native-quadrilateral vertex mean",
                "interpolation": "none",
                "fixed_x_limits": list(X_LIMITS),
                "fixed_y_limits": list(Y_LIMITS),
                "static_horizons": list(STATIC_HORIZONS),
                "animation_horizons": list(frames),
                "animation_frame_stride": arguments.frame_stride,
                "static_signed_error_norm": "common_symmetric_log",
                "animation_signed_error_norm": "common_symmetric_log",
                "static_signed_error_limits": list(exact_limits["signed_error"]),
                "animation_signed_error_limits": list(
                    qualitative_limits["signed_error"]
                ),
                "static_signed_error_linthresh": exact_linthresh,
                "animation_signed_error_linthresh": qualitative_linthresh,
                "signed_error_limit_basis": (
                    "maximum_absolute_rendered_native_quadrilateral_value"
                ),
                "signed_error_clipping": False,
                "visible_native_quad_count": int(quads.shape[0]),
            },
            "claim_boundary": _claim_boundary(),
        }
        manifest["canonical_payload_sha256"] = evaluator._canonical_sha256(manifest)
        evaluator._write_json(staging / RENDER_MANIFEST_FILE, manifest)
        return manifest

    def verify(staging: Path, manifest: dict[str, Any]) -> None:
        _reverify_visualization_source()
        if (
            evaluator.parent._file_record(
                replay_root / REPLAY_MANIFEST_FILE, replay_root
            )
            != replay_manifest_record
            or evaluator.parent._file_record(replay_root / REPLAY_FILE, replay_root)
            != replay_file_record
        ):
            raise ValueError("visualization replay changed while rendering")
        _verify_render_stage(staging, manifest)

    return _publish_directory_atomically(output, build, before_publish=verify)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    replay_parser = subparsers.add_parser("replay", help="run qualitative replay")
    replay_parser.add_argument("--population-role", default="development")
    replay_parser.add_argument("--contract", type=Path, required=True)
    replay_parser.add_argument("--successor-contract", type=Path, required=True)
    replay_parser.add_argument("--successor-preregistration", type=Path, required=True)
    replay_parser.add_argument("--extension-contract", type=Path, required=True)
    replay_parser.add_argument("--extension-preregistration", type=Path, required=True)
    replay_parser.add_argument("--dataset-dir", type=Path, required=True)
    replay_parser.add_argument("--calibration-dir", type=Path, required=True)
    replay_parser.add_argument("--paired-bank-dir", type=Path, required=True)
    replay_parser.add_argument(
        "--relabel-pilot-final-hash-manifest", type=Path, required=True
    )
    replay_parser.add_argument(
        "--training-dir", type=Path, action="append", required=True
    )
    replay_parser.add_argument(
        "--inherited-clean-training-dir", type=Path, action="append", required=True
    )
    replay_parser.add_argument(
        "--inherited-detached-training-dir", type=Path, action="append", required=True
    )
    replay_parser.add_argument("--evaluation-dir", type=Path, required=True)
    replay_parser.add_argument("--output-dir", type=Path, required=True)
    replay_parser.add_argument("--device", default="cuda")

    render_parser = subparsers.add_parser("render", help="render verified replay")
    render_parser.add_argument("--replay-dir", type=Path, required=True)
    render_parser.add_argument("--output-dir", type=Path, required=True)
    render_parser.add_argument("--ffmpeg", type=Path)
    render_parser.add_argument(
        "--expected-replay-producer-sha256",
        default=VISUALIZATION_SOURCE_AT_IMPORT["sha256"],
    )
    render_parser.add_argument("--fps", type=int, default=12)
    render_parser.add_argument("--frame-stride", type=int, default=2)
    render_parser.add_argument("--dpi", type=int, default=120)
    render_parser.add_argument("--bitrate", type=int, default=4200)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "replay":
            result = replay(arguments)
        else:
            if (
                arguments.fps <= 0
                or arguments.frame_stride <= 0
                or arguments.dpi <= 0
                or arguments.bitrate <= 0
            ):
                raise ValueError("fps, frame stride, dpi, and bitrate must be positive")
            result = render(arguments)
    except evaluator.parent.ProtectedPopulationError as error:
        print(
            f"NACA extension visualization protected access refused: {error}",
            file=sys.stderr,
        )
        return 4
    except torch.cuda.OutOfMemoryError as error:
        print(
            f"NACA extension visualization infrastructure failure: {error}",
            file=sys.stderr,
        )
        return 3
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"NACA extension visualization failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
