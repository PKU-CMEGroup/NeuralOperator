#!/usr/bin/env python3
"""Replay and render the frozen clean-PCNO NACA0012 development rollout.

The replay and render phases are separate so inference can run on AutoDL while
the publication-facing animation is rendered locally.  This script is a
visualization-only consumer of the already evaluated R0 baseline; it does not
open protected populations, extend the registered horizon, or compare a
corrective mechanism.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.time_dependent_no.evaluate_pcno_naca0012 import (
    EVALUATION_SCHEMA,
    SEEDS,
    _reverify_packet_files,
    _verify_training_packet,
)
from scripts.time_dependent_no.train_pcno_naca0012 import (
    _load_dataset,
    _load_self_hashed_snapshot,
    _self_hashed,
    _write_json,
)
from utility.time_dependent_no.pcno_naca0012 import (
    BASELINE_CONTRACT_SHA256,
    NACANormalization,
    VerifiedNACAGeometry,
    load_naca_baseline_contract,
    recurrent_step,
)
from utility.time_dependent_no.su2_restart_contract import sha256_file

REPLAY_SCHEMA = "time_dependent_no.naca_pcno_rollout_visualization_replay.v1"
MANIFEST_SCHEMA = "time_dependent_no.naca_pcno_rollout_visualization.v1"
REPLAY_FILE = "rollout_replay.npz"
REPLAY_MANIFEST_FILE = "replay_manifest.json"
FINAL_MANIFEST_FILE = "manifest.json"
DEFAULT_ANCHOR = 1234
HORIZONS = tuple(range(209))
STATIC_HORIZONS = (1, 35, 104, 208)
SNAPSHOT_REPLAY_RTOL = 1.0e-6
SNAPSHOT_REPLAY_ATOL = 1.0e-7
X_LIMITS = (-1.0, 10.0)
Y_LIMITS = (-5.0, 5.0)
SEED_COLORS = ("#0072B2", "#D55E00", "#009E73")


@dataclass(frozen=True)
class ReplayBundle:
    """Validated arrays required by the native-quad density visualization."""

    anchor: int
    horizons: np.ndarray
    frame_indices: np.ndarray
    seeds: np.ndarray
    coordinates: np.ndarray
    quads: np.ndarray
    airfoil_mask: np.ndarray
    reference_density: np.ndarray
    predicted_density: np.ndarray
    density_state_scale: float
    density_error_rmse: np.ndarray
    manifest: Mapping[str, Any]


def _file_record(path: Path, root: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"visualization artifact is absent or aliased: {path.name}")
    resolved = path.resolve()
    return {
        "relative_path": resolved.relative_to(root.resolve()).as_posix(),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _claim_boundary() -> dict[str, bool]:
    return {
        "visualization_only_replay": True,
        "clean_PCNO_only": True,
        "causal_failure_mechanism_identified": False,
        "manifold_drift_measured": False,
        "corrective_mechanisms_compared": False,
        "registered_rollout_horizon_extended": False,
        "prospective_opened": False,
        "sealed_opened": False,
    }


def _verify_evaluation_result(
    root: Path, dataset_packet_sha256: str
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if root.is_symlink():
        raise ValueError("evaluation packet is aliased")
    evaluation_root = root.resolve()
    final_path = evaluation_root / "final_hash_manifest.json"
    if final_path.is_symlink() or not final_path.is_file():
        raise ValueError("evaluation final-hash manifest is absent or aliased")
    final_sha256 = sha256_file(final_path)
    final = json.loads(final_path.read_bytes())
    files = final.get("files", {}) if isinstance(final, dict) else {}
    if (
        final.get("contract_sha256") != BASELINE_CONTRACT_SHA256
        or final.get("self_hash_excluded") is not True
        or final.get("prospective_opened") is not False
        or final.get("sealed_opened") is not False
        or not {"result.json", "rollout_snapshots.npz"}.issubset(files)
    ):
        raise ValueError("evaluation final-hash manifest differs")
    _reverify_packet_files(evaluation_root, files, final_sha256)
    result = _load_self_hashed_snapshot(
        evaluation_root,
        files["result.json"],
        "result.json",
        EVALUATION_SCHEMA,
    )
    claim = result.get("claim_boundary", {})
    if (
        result.get("status") != "complete"
        or result.get("classification") != "SCIENTIFIC_RESULT"
        or result.get("contract_sha256") != BASELINE_CONTRACT_SHA256
        or result.get("dataset_final_hash_manifest_sha256") != dataset_packet_sha256
        or result.get("population_role") != "development"
        or result.get("seeds") != list(SEEDS)
        or claim.get("R0_development_baseline_evaluated") is not True
        or claim.get("corrective_mechanism_performance_claimed") is not False
        or claim.get("prospective_opened") is not False
        or claim.get("sealed_opened") is not False
    ):
        raise ValueError("evaluation result is not the frozen clean-PCNO R0 result")
    snapshots_path = evaluation_root / "rollout_snapshots.npz"
    with np.load(snapshots_path, allow_pickle=False) as archive:
        snapshots = {name: archive[name] for name in archive.files}
    return (
        {
            "directory_name": evaluation_root.name,
            "final_hash_manifest_sha256": final_sha256,
            "result_json_sha256": files["result.json"]["sha256"],
            "rollout_snapshots_sha256": files["rollout_snapshots.npz"]["sha256"],
            "aggregate_decision": result.get("aggregate_decision", {}).get("decision"),
        },
        snapshots,
    )


def _snapshot_replay_check(
    *,
    seed: int,
    anchor: int,
    prediction: np.ndarray,
    reference: np.ndarray,
    snapshots: Mapping[str, np.ndarray],
) -> list[dict[str, Any]]:
    """Compare replay density with retained snapshots without hiding CUDA drift.

    Reference states are data and must match bitwise.  Recurrent predictions can
    differ across CUDA executions, so their exact and tight-tolerance outcomes
    are recorded as diagnostics rather than silently replaced or interpolated.
    """

    checks: list[dict[str, Any]] = []
    for horizon in STATIC_HORIZONS:
        reference_key = f"reference_anchor_{anchor}_h{horizon:03d}"
        prediction_key = f"pcno_seed_{seed}_anchor_{anchor}_h{horizon:03d}"
        if reference_key not in snapshots or prediction_key not in snapshots:
            raise ValueError("retained evaluation snapshots lack a replay checkpoint")
        expected_reference = np.asarray(snapshots[reference_key])[:, 0]
        expected_prediction = np.asarray(snapshots[prediction_key])[:, 0]
        observed_reference = np.asarray(reference[horizon])
        observed_prediction = np.asarray(prediction[horizon])
        reference_max_abs = float(
            np.max(np.abs(observed_reference - expected_reference))
        )
        prediction_max_abs = float(
            np.max(np.abs(observed_prediction - expected_prediction))
        )
        reference_exact = bool(np.array_equal(observed_reference, expected_reference))
        prediction_exact = bool(
            np.array_equal(observed_prediction, expected_prediction)
        )
        prediction_tolerance_pass = bool(
            np.allclose(
                observed_prediction,
                expected_prediction,
                rtol=SNAPSHOT_REPLAY_RTOL,
                atol=SNAPSHOT_REPLAY_ATOL,
            )
        )
        prediction_max_relative = float(
            np.max(
                np.abs(observed_prediction - expected_prediction)
                / np.maximum(np.abs(expected_prediction), np.finfo(np.float32).tiny)
            )
        )
        checks.append(
            {
                "seed": seed,
                "anchor": anchor,
                "horizon": horizon,
                "field": "Density",
                "reference_max_abs_difference": reference_max_abs,
                "prediction_max_abs_difference": prediction_max_abs,
                "reference_bitwise_equal": reference_exact,
                "prediction_bitwise_equal": prediction_exact,
                "prediction_max_relative_difference": prediction_max_relative,
                "prediction_tolerance_pass": prediction_tolerance_pass,
                "prediction_rtol": SNAPSHOT_REPLAY_RTOL,
                "prediction_atol": SNAPSHOT_REPLAY_ATOL,
            }
        )
        if not reference_exact:
            raise ValueError(
                "visualization reference differs from retained evaluation density "
                f"snapshot for anchor={anchor}, h={horizon}: "
                f"reference_max_abs={reference_max_abs:.9g}"
            )
    return checks


def _rollout_density(
    *,
    model: torch.nn.Module,
    states: np.ndarray,
    positions: Mapping[int, int],
    anchors: Sequence[int],
    selected_anchor: int,
    geometry: VerifiedNACAGeometry,
    normalization: NACANormalization,
    device: torch.device,
) -> np.ndarray:
    """Return h=0,...,208 density predictions from one exact clean checkpoint."""

    prediction = np.empty((len(HORIZONS), geometry.num_nodes), dtype=np.float32)
    anchor_offset = tuple(anchors).index(selected_anchor)
    previous = torch.as_tensor(
        np.stack([states[positions[anchor - 1]] for anchor in anchors], axis=0).astype(
            np.float32
        ),
        device=device,
    )
    current = torch.as_tensor(
        np.stack([states[positions[anchor]] for anchor in anchors], axis=0).astype(
            np.float32
        ),
        device=device,
    )
    prediction[0] = current[anchor_offset, :, 0].detach().cpu().numpy()
    geometry_batch = geometry.expand(len(anchors), device)
    fourier = model.prepare_fourier_tensors(geometry_batch)  # type: ignore[attr-defined]
    model.eval()
    with torch.no_grad():
        for horizon in HORIZONS[1:]:
            previous, current = recurrent_step(
                model,
                previous,
                current,
                geometry_batch,
                normalization,
                fourier_tensors=fourier,
            )
            if not bool(torch.all(torch.isfinite(current))):
                raise FloatingPointError(
                    f"clean-PCNO visualization replay became nonfinite at h={horizon}"
                )
            prediction[horizon] = current[anchor_offset, :, 0].detach().cpu().numpy()
    return prediction


def replay(arguments: argparse.Namespace) -> dict[str, Any]:
    """Run the registered 208-step clean-PCNO replay and write portable arrays."""

    output = arguments.output_dir.resolve()
    if arguments.output_dir.is_symlink() or output.exists():
        raise FileExistsError(f"visualization output already exists: {output}")
    contract = load_naca_baseline_contract(arguments.contract)
    (
        dataset_manifest,
        geometry,
        normalization,
        roles,
        dataset_packet_sha256,
    ) = _load_dataset(arguments.dataset_dir, contract)
    evaluation_record, retained_snapshots = _verify_evaluation_result(
        arguments.evaluation_dir, dataset_packet_sha256
    )
    device = torch.device(arguments.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if len(arguments.training_dir) != len(SEEDS):
        raise ValueError("replay requires exactly three training directories")

    states, frame_indices, _ = roles["development"]
    positions = {int(index): offset for offset, index in enumerate(frame_indices)}
    anchors = tuple(
        contract["phase_population"]["roles"]["development"]["anchor_input_indices"]
    )
    anchor = int(arguments.anchor)
    if anchor not in anchors:
        raise ValueError("anchor must be one of the frozen development anchors")
    required = tuple(anchor + horizon for horizon in HORIZONS)
    if anchor - 1 not in positions or any(index not in positions for index in required):
        raise ValueError("development packet does not cover the registered replay")
    reference_density64 = np.stack(
        [states[positions[index], :, 0] for index in required], axis=0
    )
    reference_density = reference_density64.astype(np.float32)

    seed_rollouts: list[tuple[int, np.ndarray, dict[str, Any]]] = []
    replay_checks: list[dict[str, Any]] = []
    for training_dir in arguments.training_dir:
        (
            seed,
            model,
            summary,
            final_sha256,
            directory_name,
            _,
            files,
        ) = _verify_training_packet(
            training_dir,
            dataset_manifest,
            dataset_packet_sha256,
            contract,
            geometry,
            device,
        )
        density = _rollout_density(
            model=model,
            states=states,
            positions=positions,
            anchors=anchors,
            selected_anchor=anchor,
            geometry=geometry,
            normalization=normalization,
            device=device,
        )
        replay_checks.extend(
            _snapshot_replay_check(
                seed=seed,
                anchor=anchor,
                prediction=density,
                reference=reference_density64,
                snapshots=retained_snapshots,
            )
        )
        seed_rollouts.append(
            (
                seed,
                density,
                {
                    "seed": seed,
                    "directory_name": directory_name,
                    "final_hash_manifest_sha256": final_sha256,
                    "best_checkpoint_sha256": files["best.pt"]["sha256"],
                    "best_epoch": summary["best_epoch"],
                },
            )
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    seed_rollouts.sort(key=lambda item: item[0])
    if [item[0] for item in seed_rollouts] != list(SEEDS):
        raise ValueError("training packets must contain exactly seeds 17, 29, and 43")

    predicted_density = np.stack([item[1] for item in seed_rollouts], axis=0)
    scale = float(normalization.state_scale[0])
    signed_error = (predicted_density.astype(np.float64) - reference_density) / scale
    density_error_rmse = np.sqrt(np.mean(np.square(signed_error), axis=2))
    quads = np.asarray(geometry.elements[:, 1:], dtype=np.int64)
    coordinates = np.asarray(geometry.native_coordinates, dtype=np.float64)
    airfoil_mask = np.asarray(geometry.boundary_one_hot[:, 1], dtype=np.uint8)

    staging = output.with_name(f".{output.name}.replay-staging")
    if staging.exists() or staging.is_symlink():
        raise FileExistsError(f"stale visualization staging path exists: {staging}")
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        staging.mkdir()
        np.savez_compressed(
            staging / REPLAY_FILE,
            schema=np.asarray(REPLAY_SCHEMA),
            anchor=np.asarray(anchor, dtype=np.int64),
            horizons=np.asarray(HORIZONS, dtype=np.int64),
            frame_indices=np.asarray(required, dtype=np.int64),
            seeds=np.asarray(SEEDS, dtype=np.int64),
            coordinates=coordinates,
            quads=quads,
            airfoil_mask=airfoil_mask,
            reference_density=reference_density,
            predicted_density=predicted_density,
            density_state_scale=np.asarray(scale, dtype=np.float64),
            density_error_rmse=density_error_rmse,
        )
        replay_record = _file_record(staging / REPLAY_FILE, staging)
        manifest = _self_hashed(
            {
                "schema": REPLAY_SCHEMA,
                "status": "complete",
                "classification": "VISUALIZATION_ONLY_REPLAY",
                "anchor": anchor,
                "horizons_inclusive": [0, 208],
                "population_role": "development",
                "seeds": list(SEEDS),
                "field": "Density",
                "contract_sha256": BASELINE_CONTRACT_SHA256,
                "dataset_manifest_payload_sha256": dataset_manifest[
                    "canonical_payload_sha256"
                ],
                "dataset_final_hash_manifest_sha256": dataset_packet_sha256,
                "source_set_sha256": dataset_manifest["source_manifest"][
                    "source_set_sha256"
                ],
                "evaluation_packet": evaluation_record,
                "training_packets": [item[2] for item in seed_rollouts],
                "retained_snapshot_replay_checks": replay_checks,
                "visualization_script_sha256": sha256_file(Path(__file__).resolve()),
                "replay_file": replay_record,
                "arrays": {
                    "reference_density": list(reference_density.shape),
                    "predicted_density": list(predicted_density.shape),
                    "coordinates": list(coordinates.shape),
                    "quads": list(quads.shape),
                    "airfoil_mask": list(airfoil_mask.shape),
                },
                "claim_boundary": _claim_boundary(),
            }
        )
        _write_json(staging / REPLAY_MANIFEST_FILE, manifest)
        os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def load_replay(root: Path) -> ReplayBundle:
    """Load a replay only after verifying its self-hash, file hash, and arrays."""

    if root.is_symlink():
        raise ValueError("visualization replay directory is aliased")
    replay_root = root.resolve()
    manifest_path = replay_root / REPLAY_MANIFEST_FILE
    replay_path = replay_root / REPLAY_FILE
    if manifest_path.is_symlink() or replay_path.is_symlink():
        raise ValueError("visualization replay artifact is aliased")
    manifest = json.loads(manifest_path.read_bytes())
    if not isinstance(manifest, dict) or manifest.get("schema") != REPLAY_SCHEMA:
        raise ValueError("unsupported visualization replay manifest")
    observed_hash = manifest.pop("canonical_payload_sha256", None)
    expected_hash = _self_hashed(manifest).pop("canonical_payload_sha256")
    manifest["canonical_payload_sha256"] = observed_hash
    if observed_hash != expected_hash:
        raise ValueError("visualization replay manifest self-hash differs")
    record = manifest.get("replay_file", {})
    if (
        record.get("relative_path") != REPLAY_FILE
        or not replay_path.is_file()
        or replay_path.stat().st_size != record.get("bytes")
        or sha256_file(replay_path) != record.get("sha256")
    ):
        raise ValueError("visualization replay file differs from its manifest")

    with np.load(replay_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    expected_names = {
        "schema",
        "anchor",
        "horizons",
        "frame_indices",
        "seeds",
        "coordinates",
        "quads",
        "airfoil_mask",
        "reference_density",
        "predicted_density",
        "density_state_scale",
        "density_error_rmse",
    }
    if (
        set(arrays) != expected_names
        or np.asarray(arrays["schema"]).item() != REPLAY_SCHEMA
    ):
        raise ValueError("visualization replay arrays differ")
    anchor = int(np.asarray(arrays["anchor"]).item())
    horizons = np.asarray(arrays["horizons"])
    frame_indices = np.asarray(arrays["frame_indices"])
    seeds = np.asarray(arrays["seeds"])
    coordinates = np.asarray(arrays["coordinates"])
    quads = np.asarray(arrays["quads"])
    airfoil_mask = np.asarray(arrays["airfoil_mask"])
    reference = np.asarray(arrays["reference_density"])
    predicted = np.asarray(arrays["predicted_density"])
    scale = float(np.asarray(arrays["density_state_scale"]).item())
    errors = np.asarray(arrays["density_error_rmse"])
    node_count = coordinates.shape[0] if coordinates.ndim == 2 else -1
    if (
        anchor != manifest.get("anchor")
        or horizons.dtype != np.int64
        or not np.array_equal(horizons, np.asarray(HORIZONS, dtype=np.int64))
        or frame_indices.dtype != np.int64
        or not np.array_equal(frame_indices, anchor + horizons)
        or seeds.dtype != np.int64
        or not np.array_equal(seeds, np.asarray(SEEDS, dtype=np.int64))
        or coordinates.dtype != np.float64
        or coordinates.shape != (node_count, 2)
        or not np.all(np.isfinite(coordinates))
        or quads.dtype != np.int64
        or quads.ndim != 2
        or quads.shape[1] != 4
        or np.any(quads < 0)
        or np.any(quads >= node_count)
        or airfoil_mask.dtype != np.uint8
        or airfoil_mask.shape != (node_count,)
        or not np.all((airfoil_mask == 0) | (airfoil_mask == 1))
        or np.count_nonzero(airfoil_mask) == 0
        or reference.dtype != np.float32
        or reference.shape != (len(HORIZONS), node_count)
        or predicted.dtype != np.float32
        or predicted.shape != (len(SEEDS), len(HORIZONS), node_count)
        or errors.dtype != np.float64
        or errors.shape != (len(SEEDS), len(HORIZONS))
        or not np.all(np.isfinite(reference))
        or not np.all(np.isfinite(predicted))
        or not np.all(np.isfinite(errors))
        or not np.isfinite(scale)
        or scale <= 0.0
    ):
        raise ValueError("visualization replay array contract differs")
    recomputed = np.sqrt(
        np.mean(
            np.square((predicted.astype(np.float64) - reference[None, ...]) / scale),
            axis=2,
        )
    )
    if not np.allclose(errors, recomputed, rtol=1.0e-13, atol=1.0e-15):
        raise ValueError("stored density error curves differ from replay states")
    return ReplayBundle(
        anchor=anchor,
        horizons=horizons,
        frame_indices=frame_indices,
        seeds=seeds,
        coordinates=coordinates,
        quads=quads,
        airfoil_mask=airfoil_mask,
        reference_density=reference,
        predicted_density=predicted,
        density_state_scale=scale,
        density_error_rmse=errors,
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
        raise ValueError("fixed near-body/wake view contains no native quads")
    return selected


def _quad_face_values(node_values: np.ndarray, quads: np.ndarray) -> np.ndarray:
    """Flat native-quad colors; no spatial interpolation is performed."""

    values = np.asarray(node_values)
    if values.ndim != 1 or np.any(quads < 0) or np.any(quads >= values.size):
        raise ValueError("node values and native quads do not align")
    return np.mean(values[quads], axis=1)


def color_limits(
    bundle: ReplayBundle, quads: np.ndarray
) -> dict[str, tuple[float, float]]:
    """Compute one fixed, unclipped scale for every frame and seed."""

    nodes = np.unique(quads)
    density_min = float(
        min(
            np.min(bundle.reference_density[:, nodes]),
            np.min(bundle.predicted_density[:, :, nodes]),
        )
    )
    density_max = float(
        max(
            np.max(bundle.reference_density[:, nodes]),
            np.max(bundle.predicted_density[:, :, nodes]),
        )
    )
    signed = (
        bundle.predicted_density[:, :, nodes].astype(np.float64)
        - bundle.reference_density[None, :, nodes]
    ) / bundle.density_state_scale
    error_max = float(np.max(np.abs(signed)))
    curve_max = float(np.max(bundle.density_error_rmse))
    if not density_max > density_min:
        density_max = density_min + 1.0e-12
    error_max = max(error_max, 1.0e-12)
    curve_max = max(curve_max, 1.0e-12)
    return {
        "density": (density_min, density_max),
        "signed_error": (-error_max, error_max),
        "error_curve": (0.0, 1.05 * curve_max),
    }


def animation_horizons(frame_stride: int) -> tuple[int, ...]:
    """Subsample frames without interpolation and always retain the terminal state."""

    if isinstance(frame_stride, bool) or frame_stride < 1:
        raise ValueError("frame stride must be a positive integer")
    selected = list(range(HORIZONS[0], HORIZONS[-1] + 1, frame_stride))
    if selected[-1] != HORIZONS[-1]:
        selected.append(HORIZONS[-1])
    return tuple(selected)


def _build_figure(bundle: ReplayBundle):
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import LinearSegmentedColormap, Normalize, SymLogNorm

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    quads = _visible_quads(bundle)
    vertices = bundle.coordinates[quads]
    limits = color_limits(bundle, quads)
    density_norm = Normalize(*limits["density"])
    error_max = limits["signed_error"][1]
    error_linthresh = 0.02 * error_max
    error_norm = SymLogNorm(
        linthresh=error_linthresh,
        linscale=1.0,
        vmin=-error_max,
        vmax=error_max,
        base=10,
    )
    error_cmap = LinearSegmentedColormap.from_list(
        "okabe_ito_blue_white_orange",
        ("#0072B2", "#FFFFFF", "#E69F00"),
    )
    limits["signed_error_linthresh"] = (-error_linthresh, error_linthresh)
    figure, axes = plt.subplots(2, 4, figsize=(14.0, 7.2), constrained_layout=True)
    top_titles = ["Reference", *(f"Clean PCNO (seed {seed})" for seed in SEEDS)]
    density_arrays = [bundle.reference_density, *bundle.predicted_density]
    density_artists = []
    error_artists = []
    for column, (title, values) in enumerate(
        zip(top_titles, density_arrays, strict=True)
    ):
        axis = axes[0, column]
        artist = PolyCollection(
            vertices,
            array=_quad_face_values(values[0], quads),
            cmap="viridis",
            norm=density_norm,
            edgecolors="none",
            antialiased=False,
            rasterized=True,
        )
        axis.add_collection(artist)
        density_artists.append(artist)
        axis.set_title(title)
    curve_axis = axes[1, 0]
    curve_lines = []
    for offset, (seed, color) in enumerate(zip(SEEDS, SEED_COLORS, strict=True)):
        (line,) = curve_axis.plot(
            [], [], color=color, linewidth=1.8, label=f"seed {seed}"
        )
        curve_lines.append(line)
    curve_axis.set_xlim(HORIZONS[0], HORIZONS[-1])
    curve_axis.set_ylim(*limits["error_curve"])
    curve_axis.set_xlabel("Horizon step")
    curve_axis.set_ylabel(r"Density error $\mathrm{RMSE}[(\hat\rho-\rho)/s_\rho]$")
    curve_axis.set_title("Clean-PCNO density error history")
    curve_axis.grid(color="#D9D9D9", linewidth=0.6, alpha=0.8)
    curve_axis.legend(loc="upper left", frameon=False)
    for offset, (seed, color) in enumerate(zip(SEEDS, SEED_COLORS, strict=True)):
        axis = axes[1, offset + 1]
        signed = (
            bundle.predicted_density[offset, 0] - bundle.reference_density[0]
        ) / bundle.density_state_scale
        artist = PolyCollection(
            vertices,
            array=_quad_face_values(signed, quads),
            cmap=error_cmap,
            norm=error_norm,
            edgecolors="none",
            antialiased=False,
            rasterized=True,
        )
        axis.add_collection(artist)
        error_artists.append(artist)
        axis.set_title(rf"Seed {seed}: $(\hat\rho-\rho)/s_\rho$")
    for row in range(2):
        for column in range(4):
            if row == 1 and column == 0:
                continue
            axis = axes[row, column]
            axis.set_xlim(*X_LIMITS)
            axis.set_ylim(*Y_LIMITS)
            axis.set_aspect("equal", adjustable="box")
            if row == 1:
                axis.set_xlabel("x")
            if column == 0 or (row == 1 and column == 1):
                axis.set_ylabel("y")
            else:
                axis.set_yticklabels([])
            if row == 0:
                axis.set_xticklabels([])
            airfoil = bundle.coordinates[bundle.airfoil_mask.astype(bool)]
            centroid = np.mean(airfoil, axis=0)
            angles = np.arctan2(
                airfoil[:, 1] - centroid[1], airfoil[:, 0] - centroid[0]
            )
            order = np.argsort(angles)
            outline = np.vstack((airfoil[order], airfoil[order][0]))
            axis.plot(
                outline[:, 0],
                outline[:, 1],
                color="#1A1A1A",
                linewidth=0.65,
                zorder=3,
            )
    figure.colorbar(
        ScalarMappable(norm=density_norm, cmap="viridis"),
        ax=list(axes[0, :]),
        label=r"Density $\rho$",
        fraction=0.018,
        pad=0.015,
    )
    figure.colorbar(
        ScalarMappable(norm=error_norm, cmap=error_cmap),
        ax=list(axes[1, 1:]),
        label=r"Signed normalized density error $(\hat\rho-\rho)/s_\rho$",
        fraction=0.024,
        pad=0.015,
    )
    title = figure.suptitle("")

    def update(horizon: int):
        density_artists[0].set_array(
            _quad_face_values(bundle.reference_density[horizon], quads)
        )
        for offset in range(len(SEEDS)):
            density_artists[offset + 1].set_array(
                _quad_face_values(bundle.predicted_density[offset, horizon], quads)
            )
            signed = (
                bundle.predicted_density[offset, horizon]
                - bundle.reference_density[horizon]
            ) / bundle.density_state_scale
            error_artists[offset].set_array(_quad_face_values(signed, quads))
            curve_lines[offset].set_data(
                bundle.horizons[: horizon + 1],
                bundle.density_error_rmse[offset, : horizon + 1],
            )
        title.set_text(
            "NACA0012 clean-PCNO rollout — "
            f"development anchor {bundle.anchor}, h={horizon:03d}"
        )
        return [*density_artists, *error_artists, *curve_lines, title]

    update(0)
    return figure, update, limits, quads.shape[0]


def render(arguments: argparse.Namespace) -> dict[str, Any]:
    """Render one 2x4 MP4 and the four frozen-horizon static frames."""

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter, FuncAnimation

    output = arguments.output_dir.resolve()
    if not output.is_dir() or output.is_symlink():
        raise ValueError("render requires an existing unaliased replay directory")
    if (output / FINAL_MANIFEST_FILE).exists():
        raise FileExistsError("final visualization manifest already exists")
    bundle = load_replay(output)
    replay_script_sha256 = bundle.manifest.get("visualization_script_sha256")
    current_script_sha256 = sha256_file(Path(__file__).resolve())
    if replay_script_sha256 != current_script_sha256:
        raise ValueError(
            "renderer source differs from replay-bound visualization source"
        )
    if arguments.ffmpeg is not None:
        ffmpeg = arguments.ffmpeg.resolve()
        if ffmpeg.is_symlink() or not ffmpeg.is_file():
            raise ValueError("ffmpeg path is absent or aliased")
        mpl.rcParams["animation.ffmpeg_path"] = str(ffmpeg)
    if not FFMpegWriter.isAvailable():
        raise RuntimeError("Matplotlib cannot locate ffmpeg")

    staging = output.with_name(f".{output.name}.render-staging")
    if staging.exists() or staging.is_symlink():
        raise FileExistsError(f"stale render staging path exists: {staging}")
    staging.mkdir()
    animation_name = f"density_rollout_anchor{bundle.anchor}.mp4"
    selected_animation_horizons = animation_horizons(arguments.frame_stride)
    try:
        figure, update, limits, visible_quad_count = _build_figure(bundle)
        static_names: list[str] = []
        for horizon in STATIC_HORIZONS:
            update(horizon)
            name = f"density_rollout_anchor{bundle.anchor}_h{horizon:03d}.png"
            figure.savefig(staging / name, dpi=180)
            static_names.append(name)
        animation = FuncAnimation(
            figure,
            update,
            frames=selected_animation_horizons,
            interval=1000.0 / arguments.fps,
            blit=False,
            repeat=True,
        )
        writer = FFMpegWriter(
            fps=arguments.fps,
            codec="libx264",
            bitrate=arguments.bitrate,
            extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        )
        animation.save(staging / animation_name, writer=writer, dpi=arguments.dpi)
        plt.close(figure)

        rendered_names = [animation_name, *static_names]
        for name in rendered_names:
            os.replace(staging / name, output / name)
        manifest = _self_hashed(
            {
                "schema": MANIFEST_SCHEMA,
                "status": "complete",
                "classification": "VISUALIZATION_ONLY",
                "renderer_script_sha256": current_script_sha256,
                "replay_manifest": _file_record(output / REPLAY_MANIFEST_FILE, output),
                "replay_file": _file_record(output / REPLAY_FILE, output),
                "outputs": {
                    name: _file_record(output / name, output) for name in rendered_names
                },
                "layout": {
                    "grid": [2, 4],
                    "top_row": [
                        "reference_density",
                        *[f"seed_{seed}_density" for seed in SEEDS],
                    ],
                    "bottom_row": [
                        "density_error_history",
                        *[
                            f"seed_{seed}_signed_normalized_density_error"
                            for seed in SEEDS
                        ],
                    ],
                },
                "rendering": {
                    "spatial_representation": (
                        "flat native-quadrilateral cell mean of native vertex values"
                    ),
                    "interpolation": "none",
                    "fixed_x_limits": list(X_LIMITS),
                    "fixed_y_limits": list(Y_LIMITS),
                    "fixed_color_limits": {
                        key: list(value) for key, value in limits.items()
                    },
                    "visible_native_quad_count": visible_quad_count,
                    "animation_frames": len(selected_animation_horizons),
                    "animation_horizons": list(selected_animation_horizons),
                    "frame_stride": arguments.frame_stride,
                    "fps": arguments.fps,
                    "dpi": arguments.dpi,
                    "codec": "libx264/yuv420p",
                    "static_horizons": list(STATIC_HORIZONS),
                },
                "claim_boundary": _claim_boundary(),
            }
        )
        _write_json(output / FINAL_MANIFEST_FILE, manifest)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    replay_parser = subparsers.add_parser("replay", help="run GPU/CPU inference")
    replay_parser.add_argument("--contract", type=Path, required=True)
    replay_parser.add_argument("--dataset-dir", type=Path, required=True)
    replay_parser.add_argument("--evaluation-dir", type=Path, required=True)
    replay_parser.add_argument(
        "--training-dir", type=Path, action="append", required=True
    )
    replay_parser.add_argument("--output-dir", type=Path, required=True)
    replay_parser.add_argument("--anchor", type=int, default=DEFAULT_ANCHOR)
    replay_parser.add_argument("--device", default="cuda")

    render_parser = subparsers.add_parser("render", help="render portable replay")
    render_parser.add_argument("--output-dir", type=Path, required=True)
    render_parser.add_argument("--ffmpeg", type=Path)
    render_parser.add_argument("--fps", type=int, default=12)
    render_parser.add_argument("--frame-stride", type=int, default=2)
    render_parser.add_argument("--dpi", type=int, default=120)
    render_parser.add_argument("--bitrate", type=int, default=3600)
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
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"NACA0012 rollout visualization failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
