#!/usr/bin/env python3
"""Export and render comparable PCNO resolution-rollout animations.

The ``collect`` command replays one checkpoint on one physical case and stores a
small, provenance-bound NPZ bundle. The ``render`` command combines the serious
physical-input bundle with the matched all-normal-training bundle and writes:

1. a reference/prediction/error comparison across resolutions; and
2. a node-type ablation comparison with one shared signed-density-error scale.

The renderer deliberately fixes physical extent, physical time, color limits,
and panel geometry. Each animation uses its own unclipped symmetric-log error
scale, shared by every error panel in that animation. It does not interpolate
one grid into another for metrics; ``imshow`` only expands native
finite-volume cells for display.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import animation, colors

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utility.time_dependent_no.pcno_artifacts import sha256_file
from utility.time_dependent_no.pcno_euler2d import PCNOEuler2DShardStore
from utility.time_dependent_no.pcno_resolution_transfer import (
    NODE_TYPE_PROTOCOLS,
    Resolution,
    as_model_state,
    build_resolution_checkpoint_model,
    build_resolution_geometry,
    initial_states_from_common_source,
    load_resolution_checkpoint,
    load_resolution_reference,
    make_model_sample,
    native_geometry_audit,
    node_types_for_protocol,
    parse_resolution,
    predict_resolution_sample,
    reference_at_resolution,
    resolution_label,
    validate_resolution_rollout_contract,
    weighted_scaled_relative_l2,
)
from utility.time_dependent_no.pcno_runtime import (
    checkpoint_model_node_type_input,
    select_device,
)
from utility.time_dependent_no.shock_vortex_family import (
    config_for_family_case,
    family_case_provenance,
    load_shock_vortex_family_manifest,
)

BUNDLE_SCHEMA = "pcno_resolution_animation_bundle_v1"
MANIFEST_SCHEMA = "pcno_resolution_animation_manifest_v2"
DEFAULT_RESOLUTIONS = ("125x50", "250x100", "500x200")
DEFAULT_CASE_ID = "sv_e06_y00"
PHYSICAL_PROTOCOLS = ("physical", "all_normal", "swapped_boundary_kinds")
SIGNED_LOG_LINEAR_FRACTION = 0.02
DENSITY_CMAP = "viridis"
SIGNED_ERROR_CMAP_NAME = "Okabe-Ito blue-white-orange"
SIGNED_ERROR_CMAP = colors.LinearSegmentedColormap.from_list(
    "okabe_ito_blue_white_orange",
    ("#0072B2", "#56B4E9", "#F7F7F7", "#E69F00", "#D55E00"),
)
ABLATION_COLUMNS = (
    ("Correct types", "physical", "physical"),
    ("All-normal input", "physical", "all_normal"),
    ("x/y meanings swapped", "physical", "swapped_boundary_kinds"),
    ("All-normal retrained", "all_normal", "all_normal"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_scalar(value: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(json.dumps(value, sort_keys=True, separators=(",", ":")))


def _array_key(kind: str, resolution: str, protocol: str | None = None) -> str:
    parts = [kind]
    if protocol is not None:
        parts.append(protocol)
    parts.append(resolution)
    return "__".join(parts)


def animation_frame_indices(frame_count: int, frame_stride: int) -> list[int]:
    """Return ordered animation indices and always retain the final frame."""

    if frame_count < 1:
        raise ValueError("frame_count must be positive")
    if frame_stride < 1:
        raise ValueError("frame_stride must be positive")
    indices = list(range(0, frame_count, frame_stride))
    if indices[-1] != frame_count - 1:
        indices.append(frame_count - 1)
    return indices


def _strict_output(path: Path) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)


def _collect_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser("collect", help="Replay one checkpoint into an NPZ")
    parser.add_argument("--family-root", type=Path, required=True)
    parser.add_argument("--multires-reference-root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--case-id", default=DEFAULT_CASE_ID)
    parser.add_argument("--resolutions", nargs="+", default=list(DEFAULT_RESOLUTIONS))
    parser.add_argument(
        "--protocols", nargs="+", choices=NODE_TYPE_PROTOCOLS, required=True
    )
    parser.add_argument("--source-resolution", default="1000x400")
    parser.add_argument("--training-resolution", default="250x100")
    parser.add_argument("--rollout-calls", type=int)
    parser.add_argument("--quadrature-order", type=int)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-normalization-digest", required=True)
    parser.add_argument("--expected-step-stride", type=int, default=2)
    parser.add_argument("--expected-k-max", type=int, default=8)
    parser.add_argument(
        "--expected-domain-lengths", type=float, nargs=2, default=(2.0, 1.0)
    )
    parser.add_argument("--expected-final-metrics-csv", type=Path, required=True)
    parser.add_argument("--metric-absolute-tolerance", type=float, default=5.0e-5)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--amp", choices=("none", "bf16", "fp16"), default="none")


def _render_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser("render", help="Render two comparable GIFs")
    parser.add_argument("--physical-bundle", type=Path, required=True)
    parser.add_argument("--all-normal-bundle", type=Path, required=True)
    parser.add_argument("--resolution-output", type=Path, required=True)
    parser.add_argument("--ablation-output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--dpi", type=int, default=90)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    _collect_parser(subparsers)
    _render_parser(subparsers)
    return parser.parse_args(argv)


def _contract_namespace(args: argparse.Namespace) -> argparse.Namespace:
    """Adapt collection arguments to the frozen evaluator validator."""

    return argparse.Namespace(
        checkpoint=args.checkpoint,
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
        expected_normalization_digest=args.expected_normalization_digest,
        expected_k_max=args.expected_k_max,
        expected_domain_lengths=args.expected_domain_lengths,
        expected_step_stride=args.expected_step_stride,
        protocols=args.protocols,
        rollout_calls=args.rollout_calls,
        endpoint_physical_times=(0.6,),
        shock_quantile=0.9,
        repeat_forward=1,
        boundary_band_widths=(0.02,),
    )


def _expected_metric_lookup(
    path: Path,
    *,
    case_id: str,
    protocols: Sequence[str],
    resolutions: Sequence[str],
    final_call: int,
) -> dict[tuple[str, str], float]:
    expected: dict[tuple[str, str], float] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if (
                row["case_id"] != case_id
                or row["node_type_protocol"] not in protocols
                or row["resolution"] not in resolutions
                or int(row["call"]) != final_call
            ):
                continue
            key = (row["node_type_protocol"], row["resolution"])
            if key in expected:
                raise ValueError(f"duplicate retained metric row: {key}")
            expected[key] = float(row["scaled_relative_l2_physical_volume"])
    required = {
        (protocol, resolution) for protocol in protocols for resolution in resolutions
    }
    missing = sorted(required - set(expected))
    if missing:
        raise ValueError(f"retained metrics are missing final rows: {missing}")
    return expected


def collect_bundle(args: argparse.Namespace) -> int:
    _strict_output(args.output)
    resolutions = [parse_resolution(value) for value in args.resolutions]
    ordered = sorted(resolutions, key=lambda value: value[0] * value[1])
    resolution_names = [resolution_label(value) for value in ordered]
    source_resolution = parse_resolution(args.source_resolution)
    training_resolution = parse_resolution(args.training_resolution)
    protocols = list(dict.fromkeys(args.protocols))

    checkpoint = load_resolution_checkpoint(args.checkpoint)
    manifest = load_shock_vortex_family_manifest(
        args.family_root / "family_manifest.json"
    )
    store = PCNOEuler2DShardStore(args.data_dir, max_cached_trajectories=1)
    try:
        stride, rollout_calls, physical_dt, _ = validate_resolution_rollout_contract(
            _contract_namespace(args),
            checkpoint,
            manifest,
            store,
            ordered,
            source_resolution,
            training_resolution,
        )
        provenance = family_case_provenance(manifest, args.case_id)
        if provenance["split"] != "validation":
            raise ValueError("animation case must belong to the validation split")
        case_config = config_for_family_case(manifest, args.case_id)
        device = select_device(args.device)
        if args.amp != "none" and device.type != "cuda":
            raise ValueError("mixed precision requires CUDA")
        model, normalization = build_resolution_checkpoint_model(checkpoint, device)

        geometry_by_resolution: dict[Resolution, Any] = {}
        sample_by_key: dict[tuple[Resolution, str], Mapping[str, torch.Tensor]] = {}
        for resolution in ordered:
            config, geometry = build_resolution_geometry(case_config, resolution)
            geometry_by_resolution[resolution] = geometry
            for protocol in protocols:
                node_type = node_types_for_protocol(
                    geometry,
                    config,
                    protocol,
                    training_resolution=training_resolution,
                )
                sample_by_key[(resolution, protocol)] = make_model_sample(
                    geometry,
                    node_type,
                    mach=case_config.shock_mach,
                    device=device,
                )
        native_geometry_audit(
            geometry_by_resolution[training_resolution], store, args.case_id
        )

        reference, reference_check = load_resolution_reference(
            args.family_root,
            args.multires_reference_root,
            store,
            manifest,
            args.case_id,
            training_resolution=training_resolution,
        )
        reference_resolution = tuple(int(v) for v in reference["retained_resolution"])
        _, common_states = initial_states_from_common_source(
            case_config,
            ordered,
            source_resolution=source_resolution,
            dtype=torch.float64,
            quadrature_order=args.quadrature_order,
        )
        stored_t0 = np.asarray(store.states(args.case_id)[0])
        if not np.array_equal(
            np.asarray(common_states[training_resolution], dtype=np.float32), stored_t0
        ):
            raise ValueError("common-source native initial state differs from shards")

        physical_times = np.arange(rollout_calls + 1, dtype=np.float64) * physical_dt
        arrays: dict[str, np.ndarray] = {
            "physical_times": physical_times,
            "state_scale": np.asarray(normalization.state_scale, dtype=np.float64),
        }
        computed_final: dict[tuple[str, str], float] = {}
        for resolution in ordered:
            label = resolution_label(resolution)
            geometry = geometry_by_resolution[resolution]
            reference_frames = []
            for call in range(rollout_calls + 1):
                frame = call * stride
                expected_time = physical_times[call]
                actual_time = float(reference["physical_times"][frame])
                if not math.isclose(
                    actual_time, expected_time, rel_tol=0.0, abs_tol=1.0e-12
                ):
                    raise ValueError("reference and rollout physical times differ")
                state = reference_at_resolution(
                    reference["conservative_states"][frame],
                    reference_resolution=reference_resolution,
                    target_resolution=resolution,
                )
                if state is None:
                    raise ValueError(f"reference is unavailable at {label}")
                reference_frames.append(np.asarray(state, dtype=np.float32))
            reference_array = np.stack(reference_frames)
            arrays[_array_key("reference", label)] = reference_array
            arrays[_array_key("volumes", label)] = np.asarray(
                geometry.node_measures, dtype=np.float64
            ).reshape(-1)

            for protocol in protocols:
                current = as_model_state(common_states[resolution])
                predictions = [np.asarray(current, dtype=np.float32)]
                metrics = [
                    weighted_scaled_relative_l2(
                        current,
                        reference_array[0],
                        volumes=geometry.node_measures,
                        component_scale=normalization.state_scale,
                    )
                ]
                for _ in range(rollout_calls):
                    current, _ = predict_resolution_sample(
                        model,
                        sample_by_key[(resolution, protocol)],
                        current,
                        device=device,
                        amp=args.amp,
                        repeats=1,
                    )
                    predictions.append(np.asarray(current, dtype=np.float32))
                    metrics.append(
                        weighted_scaled_relative_l2(
                            current,
                            reference_array[len(predictions) - 1],
                            volumes=geometry.node_measures,
                            component_scale=normalization.state_scale,
                        )
                    )
                prediction_array = np.stack(predictions)
                metric_array = np.asarray(metrics, dtype=np.float64)
                arrays[_array_key("prediction", label, protocol)] = prediction_array
                arrays[_array_key("metric", label, protocol)] = metric_array
                computed_final[(protocol, label)] = float(metric_array[-1])
                print(
                    f"{args.label} {protocol} {label}: "
                    f"final={100.0 * metric_array[-1]:.6f}%",
                    flush=True,
                )

        expected = _expected_metric_lookup(
            args.expected_final_metrics_csv,
            case_id=args.case_id,
            protocols=protocols,
            resolutions=resolution_names,
            final_call=rollout_calls,
        )
        metric_differences = {
            f"{protocol}/{resolution}": abs(
                computed_final[(protocol, resolution)]
                - expected[(protocol, resolution)]
            )
            for protocol in protocols
            for resolution in resolution_names
        }
        worst_metric_difference = max(metric_differences.values())
        if worst_metric_difference > args.metric_absolute_tolerance:
            raise ValueError(
                "animation replay differs from retained metrics: "
                f"max_abs={worst_metric_difference:.6e}, "
                f"tolerance={args.metric_absolute_tolerance:.6e}"
            )

        metadata = {
            "schema": BUNDLE_SCHEMA,
            "label": args.label,
            "case_id": args.case_id,
            "case_provenance": provenance,
            "resolutions": resolution_names,
            "source_resolution": resolution_label(source_resolution),
            "training_resolution": resolution_label(training_resolution),
            "protocols": protocols,
            "model_node_type_input": checkpoint_model_node_type_input(checkpoint),
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "normalization_digest": checkpoint["normalization_digest"],
            "family_manifest_digest": manifest["manifest_digest_sha256"],
            "reference_check": reference_check,
            "stride": stride,
            "physical_dt": physical_dt,
            "rollout_calls": rollout_calls,
            "boundary_policy": "model_all_nodes raw recurrence",
            "metric_contract": "physical-volume component-scaled relative L2",
            "retained_final_metrics_csv": str(args.expected_final_metrics_csv),
            "maximum_retained_metric_absolute_difference": worst_metric_difference,
            "retained_metric_absolute_tolerance": args.metric_absolute_tolerance,
            "final_metrics": {
                f"{protocol}/{resolution}": computed_final[(protocol, resolution)]
                for protocol in protocols
                for resolution in resolution_names
            },
        }
        arrays["metadata_json"] = _json_scalar(metadata)
        np.savez_compressed(args.output, **arrays)
        print(f"wrote {args.output} ({args.output.stat().st_size} bytes)", flush=True)
    finally:
        store.close()
        if "model" in locals():
            del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return 0


@dataclass(frozen=True)
class AnimationBundle:
    path: Path
    metadata: Mapping[str, Any]
    physical_times: np.ndarray
    state_scale: np.ndarray
    references: Mapping[str, np.ndarray]
    predictions: Mapping[tuple[str, str], np.ndarray]
    metrics: Mapping[tuple[str, str], np.ndarray]

    @property
    def resolutions(self) -> tuple[str, ...]:
        return tuple(str(value) for value in self.metadata["resolutions"])

    @property
    def protocols(self) -> tuple[str, ...]:
        return tuple(str(value) for value in self.metadata["protocols"])


def load_bundle(path: Path) -> AnimationBundle:
    with np.load(path, allow_pickle=False) as artifact:
        metadata = json.loads(artifact["metadata_json"].item())
        if metadata.get("schema") != BUNDLE_SCHEMA:
            raise ValueError(f"unsupported animation bundle: {path}")
        resolutions = tuple(str(value) for value in metadata["resolutions"])
        protocols = tuple(str(value) for value in metadata["protocols"])
        times = np.asarray(artifact["physical_times"], dtype=np.float64)
        state_scale = np.asarray(artifact["state_scale"], dtype=np.float64)
        references = {
            resolution: np.asarray(
                artifact[_array_key("reference", resolution)], dtype=np.float32
            )
            for resolution in resolutions
        }
        predictions = {
            (protocol, resolution): np.asarray(
                artifact[_array_key("prediction", resolution, protocol)],
                dtype=np.float32,
            )
            for protocol in protocols
            for resolution in resolutions
        }
        metrics = {
            (protocol, resolution): np.asarray(
                artifact[_array_key("metric", resolution, protocol)], dtype=np.float64
            )
            for protocol in protocols
            for resolution in resolutions
        }
    if times.ndim != 1 or times.size < 2:
        raise ValueError("bundle physical times must be a nontrivial vector")
    for resolution in resolutions:
        nx, ny = parse_resolution(resolution)
        expected_shape = (times.size, nx * ny, 4)
        if references[resolution].shape != expected_shape:
            raise ValueError(
                f"reference shape at {resolution} is not {expected_shape}: "
                f"{references[resolution].shape}"
            )
        for protocol in protocols:
            if predictions[(protocol, resolution)].shape != expected_shape:
                raise ValueError(f"prediction shape mismatch: {protocol}/{resolution}")
            if metrics[(protocol, resolution)].shape != (times.size,):
                raise ValueError(f"metric shape mismatch: {protocol}/{resolution}")
    return AnimationBundle(
        path=path,
        metadata=metadata,
        physical_times=times,
        state_scale=state_scale,
        references=references,
        predictions=predictions,
        metrics=metrics,
    )


def _density_frames(states: np.ndarray, resolution: str) -> np.ndarray:
    nx, ny = parse_resolution(resolution)
    return np.asarray(states[..., 0], dtype=np.float64).reshape(-1, ny, nx)


def animation_color_limits(
    physical: AnimationBundle,
    all_normal: AnimationBundle,
) -> tuple[float, float, float, float]:
    """Return density limits and exact error limits for the two animations."""

    state_min = math.inf
    state_max = -math.inf
    resolution_error_limit = 0.0
    ablation_error_limit = 0.0
    for resolution in physical.resolutions:
        reference = _density_frames(physical.references[resolution], resolution)
        prediction = _density_frames(
            physical.predictions[("physical", resolution)], resolution
        )
        state_min = min(state_min, float(np.min(reference)), float(np.min(prediction)))
        state_max = max(state_max, float(np.max(reference)), float(np.max(prediction)))
        resolution_error_limit = max(
            resolution_error_limit,
            float(np.max(np.abs(prediction - reference))),
        )
        for _, source_name, protocol in ABLATION_COLUMNS:
            source = physical if source_name == "physical" else all_normal
            prediction = _density_frames(
                source.predictions[(protocol, resolution)], resolution
            )
            ablation_error_limit = max(
                ablation_error_limit,
                float(np.max(np.abs(prediction - reference))),
            )
    if (
        not state_min < state_max
        or resolution_error_limit <= 0.0
        or ablation_error_limit <= 0.0
    ):
        raise ValueError("animation color limits are degenerate")
    return state_min, state_max, resolution_error_limit, ablation_error_limit


def _signed_log_norm(limit: float) -> colors.SymLogNorm:
    if not math.isfinite(limit) or limit <= 0.0:
        raise ValueError("signed-log limit must be positive and finite")
    return colors.SymLogNorm(
        linthresh=SIGNED_LOG_LINEAR_FRACTION * limit,
        linscale=1.0,
        vmin=-limit,
        vmax=limit,
        base=10.0,
    )


def _format_signed_log_colorbar(colorbar: Any, limit: float) -> None:
    threshold = SIGNED_LOG_LINEAR_FRACTION * limit
    ticks = (-limit, -threshold, 0.0, threshold, limit)
    colorbar.set_ticks(ticks)
    colorbar.set_ticklabels(tuple(f"{value:.3g}" for value in ticks))
    colorbar.ax.tick_params(labelsize=7)


def _configure_axis(axis: Any, *, row: int, row_label: str) -> None:
    axis.set_xlim(0.0, 2.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_aspect("equal")
    axis.set_xticks((0.0, 1.0, 2.0) if row == 2 else ())
    axis.set_yticks((0.0, 0.5, 1.0))
    axis.tick_params(labelsize=6, length=2)
    axis.set_ylabel(row_label, fontsize=8, fontweight="bold")


def _save_resolution_animation(
    bundle: AnimationBundle,
    path: Path,
    *,
    frame_indices: Sequence[int],
    state_min: float,
    state_max: float,
    error_limit: float,
    fps: int,
    dpi: int,
) -> None:
    _strict_output(path)
    resolutions = bundle.resolutions
    figure, axes = plt.subplots(
        len(resolutions),
        3,
        figsize=(10.8, 5.8),
        constrained_layout=True,
        squeeze=False,
    )
    state_norm = colors.Normalize(vmin=state_min, vmax=state_max)
    error_norm = _signed_log_norm(error_limit)
    images: list[list[Any]] = []
    annotations: list[Any] = []
    for row, resolution in enumerate(resolutions):
        reference = _density_frames(bundle.references[resolution], resolution)
        prediction = _density_frames(
            bundle.predictions[("physical", resolution)], resolution
        )
        row_images = [
            axes[row, 0].imshow(
                reference[0],
                origin="lower",
                extent=(0.0, 2.0, 0.0, 1.0),
                interpolation="nearest",
                cmap=DENSITY_CMAP,
                norm=state_norm,
            ),
            axes[row, 1].imshow(
                prediction[0],
                origin="lower",
                extent=(0.0, 2.0, 0.0, 1.0),
                interpolation="nearest",
                cmap=DENSITY_CMAP,
                norm=state_norm,
            ),
            axes[row, 2].imshow(
                prediction[0] - reference[0],
                origin="lower",
                extent=(0.0, 2.0, 0.0, 1.0),
                interpolation="nearest",
                cmap=SIGNED_ERROR_CMAP,
                norm=error_norm,
            ),
        ]
        images.append(row_images)
        for axis in axes[row]:
            _configure_axis(axis, row=row, row_label=resolution)
        annotations.append(
            axes[row, 1].text(
                0.02,
                0.94,
                "",
                transform=axes[row, 1].transAxes,
                ha="left",
                va="top",
                fontsize=7,
                color="white",
                bbox={
                    "facecolor": "black",
                    "alpha": 0.55,
                    "pad": 2,
                    "edgecolor": "none",
                },
            )
        )
    for column, title in enumerate(
        (
            "Common-source reference $\\rho$",
            "PCNO prediction $\\rho$",
            "Signed-log error",
        )
    ):
        axes[0, column].set_title(title, fontsize=10, fontweight="bold")
    figure.colorbar(
        images[0][0],
        ax=axes[:, :2],
        orientation="horizontal",
        shrink=0.58,
        pad=0.02,
        label="Density $\\rho$ (shared linear scale)",
    )
    error_colorbar = figure.colorbar(
        images[0][2],
        ax=axes[:, 2],
        orientation="horizontal",
        shrink=0.82,
        pad=0.02,
        label="Signed density error (shared SymLog; no clipping)",
    )
    _format_signed_log_colorbar(error_colorbar, error_limit)
    title = figure.suptitle("", fontsize=11, fontweight="bold")

    def update(frame_index: int) -> list[Any]:
        artists: list[Any] = []
        for row, resolution in enumerate(resolutions):
            reference = _density_frames(bundle.references[resolution], resolution)
            prediction = _density_frames(
                bundle.predictions[("physical", resolution)], resolution
            )
            images[row][0].set_data(reference[frame_index])
            images[row][1].set_data(prediction[frame_index])
            images[row][2].set_data(prediction[frame_index] - reference[frame_index])
            annotations[row].set_text(
                "state rel. $L^2$\n"
                f"{100.0 * bundle.metrics[('physical', resolution)][frame_index]:.3f}%"
            )
            artists.extend(images[row])
            artists.append(annotations[row])
        title.set_text(
            f"Zero-shot resolution transfer | {bundle.metadata['case_id']} | "
            f"$t={bundle.physical_times[frame_index]:.2f}$"
        )
        artists.append(title)
        return artists

    movie = animation.FuncAnimation(
        figure,
        update,
        frames=list(frame_indices),
        blit=False,
        repeat=True,
    )
    movie.save(path, writer=animation.PillowWriter(fps=fps), dpi=dpi)
    plt.close(figure)


def _save_ablation_animation(
    physical: AnimationBundle,
    all_normal: AnimationBundle,
    path: Path,
    *,
    frame_indices: Sequence[int],
    error_limit: float,
    fps: int,
    dpi: int,
) -> None:
    _strict_output(path)
    resolutions = physical.resolutions
    figure, axes = plt.subplots(
        len(resolutions),
        len(ABLATION_COLUMNS),
        figsize=(12.0, 5.2),
        constrained_layout=True,
        squeeze=False,
    )
    error_norm = _signed_log_norm(error_limit)
    images: list[list[Any]] = []
    annotations: list[list[Any]] = []
    for row, resolution in enumerate(resolutions):
        reference = _density_frames(physical.references[resolution], resolution)
        row_images = []
        row_annotations = []
        for column, (_, source_name, protocol) in enumerate(ABLATION_COLUMNS):
            source = physical if source_name == "physical" else all_normal
            prediction = _density_frames(
                source.predictions[(protocol, resolution)], resolution
            )
            image = axes[row, column].imshow(
                prediction[0] - reference[0],
                origin="lower",
                extent=(0.0, 2.0, 0.0, 1.0),
                interpolation="nearest",
                cmap=SIGNED_ERROR_CMAP,
                norm=error_norm,
            )
            row_images.append(image)
            _configure_axis(axes[row, column], row=row, row_label=resolution)
            row_annotations.append(
                axes[row, column].text(
                    0.02,
                    0.94,
                    "",
                    transform=axes[row, column].transAxes,
                    ha="left",
                    va="top",
                    fontsize=6.5,
                    color="black",
                    bbox={
                        "facecolor": "white",
                        "alpha": 0.72,
                        "pad": 1.5,
                        "edgecolor": "none",
                    },
                )
            )
        images.append(row_images)
        annotations.append(row_annotations)
    for column, (label, source_name, _) in enumerate(ABLATION_COLUMNS):
        suffix = (
            "frozen intervention"
            if source_name == "physical" and column
            else "checkpoint"
        )
        axes[0, column].set_title(
            f"{label}\n({suffix})", fontsize=8.5, fontweight="bold"
        )
    error_colorbar = figure.colorbar(
        images[0][0],
        ax=axes,
        orientation="horizontal",
        shrink=0.5,
        pad=0.025,
        label="Signed density error (shared SymLog; no clipping)",
    )
    _format_signed_log_colorbar(error_colorbar, error_limit)
    title = figure.suptitle("", fontsize=11, fontweight="bold")

    def update(frame_index: int) -> list[Any]:
        artists: list[Any] = []
        for row, resolution in enumerate(resolutions):
            reference = _density_frames(physical.references[resolution], resolution)
            for column, (_, source_name, protocol) in enumerate(ABLATION_COLUMNS):
                source = physical if source_name == "physical" else all_normal
                prediction = _density_frames(
                    source.predictions[(protocol, resolution)], resolution
                )
                images[row][column].set_data(
                    prediction[frame_index] - reference[frame_index]
                )
                annotations[row][column].set_text(
                    "state rel. $L^2$: "
                    f"{100.0 * source.metrics[(protocol, resolution)][frame_index]:.3f}%"
                )
                artists.extend((images[row][column], annotations[row][column]))
        title.set_text(
            f"Node-type encoding audit | {physical.metadata['case_id']} | "
            f"$t={physical.physical_times[frame_index]:.2f}$"
        )
        artists.append(title)
        return artists

    movie = animation.FuncAnimation(
        figure,
        update,
        frames=list(frame_indices),
        blit=False,
        repeat=True,
    )
    movie.save(path, writer=animation.PillowWriter(fps=fps), dpi=dpi)
    plt.close(figure)


def render_animations(args: argparse.Namespace) -> int:
    for path in (
        args.resolution_output,
        args.ablation_output,
        args.manifest_output,
    ):
        if path.exists():
            raise FileExistsError(path)
    if args.fps < 1 or args.dpi < 30:
        raise ValueError("fps must be positive and dpi must be at least 30")

    physical = load_bundle(args.physical_bundle)
    all_normal = load_bundle(args.all_normal_bundle)
    if physical.metadata["case_id"] != all_normal.metadata["case_id"]:
        raise ValueError("animation bundles use different physical cases")
    if physical.resolutions != all_normal.resolutions:
        raise ValueError("animation bundles use different resolutions")
    if not np.array_equal(physical.physical_times, all_normal.physical_times):
        raise ValueError("animation bundles use different physical times")
    for resolution in physical.resolutions:
        if not np.array_equal(
            physical.references[resolution], all_normal.references[resolution]
        ):
            raise ValueError(f"bundle references differ at {resolution}")
    missing_physical = sorted(set(PHYSICAL_PROTOCOLS) - set(physical.protocols))
    if missing_physical:
        raise ValueError(f"physical bundle is missing protocols: {missing_physical}")
    if all_normal.protocols != ("all_normal",):
        raise ValueError("matched-training bundle must contain only all_normal")

    frame_indices = animation_frame_indices(
        physical.physical_times.size, args.frame_stride
    )
    state_min, state_max, resolution_error_limit, ablation_error_limit = (
        animation_color_limits(physical, all_normal)
    )
    _save_resolution_animation(
        physical,
        args.resolution_output,
        frame_indices=frame_indices,
        state_min=state_min,
        state_max=state_max,
        error_limit=resolution_error_limit,
        fps=args.fps,
        dpi=args.dpi,
    )
    _save_ablation_animation(
        physical,
        all_normal,
        args.ablation_output,
        frame_indices=frame_indices,
        error_limit=ablation_error_limit,
        fps=args.fps,
        dpi=args.dpi,
    )
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "case_selection": (
            "sv_e06_y00 is the median final-error validation case at every "
            "evaluated resolution for the physical Call_30 checkpoint"
        ),
        "case_id": physical.metadata["case_id"],
        "physical_times": [float(value) for value in physical.physical_times],
        "rendered_frame_indices": list(frame_indices),
        "resolutions": list(physical.resolutions),
        "density_limits": [state_min, state_max],
        "density_colormap": DENSITY_CMAP,
        "resolution_signed_density_error_scale": {
            "normalization": "symmetric_log",
            "limits": [-resolution_error_limit, resolution_error_limit],
            "linear_threshold": (SIGNED_LOG_LINEAR_FRACTION * resolution_error_limit),
            "clipped": False,
            "colormap": SIGNED_ERROR_CMAP_NAME,
        },
        "ablation_signed_density_error_scale": {
            "normalization": "symmetric_log",
            "limits": [-ablation_error_limit, ablation_error_limit],
            "linear_threshold": SIGNED_LOG_LINEAR_FRACTION * ablation_error_limit,
            "clipped": False,
            "colormap": SIGNED_ERROR_CMAP_NAME,
        },
        "error_scale_contract": (
            "each animation uses its own exact, unclipped symmetric-log range; "
            "the scale is shared by every error panel inside that animation, "
            "so colors are comparable within but not across the two animations"
        ),
        "physical_extent": [0.0, 2.0, 0.0, 1.0],
        "interpolation": "nearest native finite-volume cells; display only",
        "resolution_animation": {
            "path": str(args.resolution_output),
            "sha256": _sha256(args.resolution_output),
            "bytes": args.resolution_output.stat().st_size,
        },
        "ablation_animation": {
            "path": str(args.ablation_output),
            "sha256": _sha256(args.ablation_output),
            "bytes": args.ablation_output.stat().st_size,
            "columns": [value[0] for value in ABLATION_COLUMNS],
        },
        "physical_bundle": {
            "path": str(physical.path),
            "sha256": _sha256(physical.path),
            "metadata": physical.metadata,
        },
        "all_normal_bundle": {
            "path": str(all_normal.path),
            "sha256": _sha256(all_normal.path),
            "metadata": all_normal.metadata,
        },
    }
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.resolution_output}", flush=True)
    print(f"wrote {args.ablation_output}", flush=True)
    print(f"wrote {args.manifest_output}", flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "collect":
        return collect_bundle(args)
    if args.command == "render":
        return render_animations(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
