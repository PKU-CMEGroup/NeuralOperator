#!/usr/bin/env python3
"""Render registered raw-shadow correction rollout animations."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation, colors

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.time_dependent_no.evaluate_pcno_response_filtered_block import (
    BUFFERED_OFFSET_ANIMATION_CONTRACT,
    BUFFERED_OFFSET_SCHEMA,
    BUFFERED_OFFSET_WORKING_ID,
    BUFFERED_RELAXED_TETHER_E14_ANIMATION_CONTRACT,
    BUFFERED_RELAXED_TETHER_E14_SCHEMA,
    BUFFERED_RELAXED_TETHER_E14_WORKING_ID,
    BUFFERED_RELAXED_TETHER_REPLAY_ANIMATION_CONTRACT,
    BUFFERED_RELAXED_TETHER_REPLAY_SCHEMA,
    BUFFERED_RELAXED_TETHER_REPLAY_WORKING_ID,
    FIXED_LATE_RAMP_ANIMATION_CONTRACT,
    FIXED_LATE_RAMP_SCHEMA,
    FIXED_LATE_RAMP_WORKING_ID,
    FROZEN_OFFSET_ANIMATION_CONTRACT,
    FROZEN_OFFSET_SCHEMA,
    FROZEN_OFFSET_WORKING_ID,
    PERSISTENCE_GAIN_ANIMATION_CONTRACT,
    PERSISTENCE_GAIN_SCHEMA,
    PERSISTENCE_GAIN_WORKING_ID,
    PROJECTED_COAST_ANIMATION_CONTRACT,
    PROJECTED_COAST_SCHEMA,
    PROJECTED_COAST_WORKING_ID,
    PROSPECTIVE_BUFFERED_OFFSET_ANIMATION_CONTRACT,
    PROSPECTIVE_BUFFERED_OFFSET_SCHEMA,
    PROSPECTIVE_BUFFERED_OFFSET_WORKING_ID,
    PROSPECTIVE_STRUCTURAL_ANIMATION_CONTRACT,
    PROSPECTIVE_STRUCTURAL_SCHEMA,
    PROSPECTIVE_STRUCTURAL_WORKING_ID,
    RELAXED_TETHER_ANIMATION_CONTRACT,
    RELAXED_TETHER_SCHEMA,
    RELAXED_TETHER_WORKING_ID,
    SLEW_LIMITED_TETHER_ANIMATION_CONTRACT,
    SLEW_LIMITED_TETHER_SCHEMA,
    SLEW_LIMITED_TETHER_WORKING_ID,
    STRENGTH_OOD_ANIMATION_CONTRACT,
    STRENGTH_OOD_SCHEMA,
    STRENGTH_OOD_WORKING_ID,
    STRUCTURAL_GATE_ANIMATION_CONTRACT,
    STRUCTURAL_GATE_SCHEMA,
    STRUCTURAL_GATE_WORKING_ID,
    TERMINAL_RAMP_ANIMATION_CONTRACT,
    TERMINAL_RAMP_SCHEMA,
    TERMINAL_RAMP_WORKING_ID,
    WARM_START_ANIMATION_CONTRACT,
    WARM_START_SCHEMA,
    WARM_START_WORKING_ID,
)
from utility.time_dependent_no.pcno_artifacts import (
    atomic_write_json_with_paths as atomic_write_json,
)
from utility.time_dependent_no.pcno_artifacts import sha256_file
from utility.time_dependent_no.pcno_cross_resolution_teacher_forced import (
    verify_payload_sha256,
    with_payload_sha256,
)

RENDER_SCHEMA = "pcno_response_filtered_block_animation_render_v1"
FPS = 5
DPI = 100
FIELDS = ("density", "pressure")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON payload must be an object: {path}")
    return value


def _load_contract(
    rollout_path: Path, manifest_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    rollout = _read_json(rollout_path)
    manifest = _read_json(manifest_path)
    verify_payload_sha256(rollout)
    verify_payload_sha256(manifest)
    contracts = {
        STRENGTH_OOD_WORKING_ID: (
            STRENGTH_OOD_SCHEMA,
            STRENGTH_OOD_ANIMATION_CONTRACT,
        ),
        STRUCTURAL_GATE_WORKING_ID: (
            STRUCTURAL_GATE_SCHEMA,
            STRUCTURAL_GATE_ANIMATION_CONTRACT,
        ),
        PROSPECTIVE_STRUCTURAL_WORKING_ID: (
            PROSPECTIVE_STRUCTURAL_SCHEMA,
            PROSPECTIVE_STRUCTURAL_ANIMATION_CONTRACT,
        ),
        WARM_START_WORKING_ID: (
            WARM_START_SCHEMA,
            WARM_START_ANIMATION_CONTRACT,
        ),
        PROJECTED_COAST_WORKING_ID: (
            PROJECTED_COAST_SCHEMA,
            PROJECTED_COAST_ANIMATION_CONTRACT,
        ),
        FROZEN_OFFSET_WORKING_ID: (
            FROZEN_OFFSET_SCHEMA,
            FROZEN_OFFSET_ANIMATION_CONTRACT,
        ),
        BUFFERED_OFFSET_WORKING_ID: (
            BUFFERED_OFFSET_SCHEMA,
            BUFFERED_OFFSET_ANIMATION_CONTRACT,
        ),
        PROSPECTIVE_BUFFERED_OFFSET_WORKING_ID: (
            PROSPECTIVE_BUFFERED_OFFSET_SCHEMA,
            PROSPECTIVE_BUFFERED_OFFSET_ANIMATION_CONTRACT,
        ),
        PERSISTENCE_GAIN_WORKING_ID: (
            PERSISTENCE_GAIN_SCHEMA,
            PERSISTENCE_GAIN_ANIMATION_CONTRACT,
        ),
        TERMINAL_RAMP_WORKING_ID: (
            TERMINAL_RAMP_SCHEMA,
            TERMINAL_RAMP_ANIMATION_CONTRACT,
        ),
        FIXED_LATE_RAMP_WORKING_ID: (
            FIXED_LATE_RAMP_SCHEMA,
            FIXED_LATE_RAMP_ANIMATION_CONTRACT,
        ),
        SLEW_LIMITED_TETHER_WORKING_ID: (
            SLEW_LIMITED_TETHER_SCHEMA,
            SLEW_LIMITED_TETHER_ANIMATION_CONTRACT,
        ),
        RELAXED_TETHER_WORKING_ID: (
            RELAXED_TETHER_SCHEMA,
            RELAXED_TETHER_ANIMATION_CONTRACT,
        ),
        BUFFERED_RELAXED_TETHER_REPLAY_WORKING_ID: (
            BUFFERED_RELAXED_TETHER_REPLAY_SCHEMA,
            BUFFERED_RELAXED_TETHER_REPLAY_ANIMATION_CONTRACT,
        ),
        BUFFERED_RELAXED_TETHER_E14_WORKING_ID: (
            BUFFERED_RELAXED_TETHER_E14_SCHEMA,
            BUFFERED_RELAXED_TETHER_E14_ANIMATION_CONTRACT,
        ),
    }
    working_id = rollout.get("working_id")
    if working_id not in contracts:
        raise ValueError("rollout is not a registered animation execution")
    expected_schema, expected_contract = contracts[working_id]
    if (
        rollout.get("schema") != expected_schema
        or rollout.get("recurrence_executed") is not True
    ):
        raise ValueError("rollout schema or recurrence status is invalid")
    if (
        manifest.get("schema") != "pcno_response_filtered_block_animation_bundles_v1"
        or manifest.get("working_id") != working_id
        or manifest.get("contract") != expected_contract
        or manifest.get("inference_or_gate_input") is not False
        or rollout.get("animation_bundle_manifest") != manifest
        or rollout.get("animation_bundle_manifest_sha256") != sha256_file(manifest_path)
    ):
        raise ValueError("animation manifest differs from the registered contract")
    return rollout, manifest


def _primitive_field(
    conservative: np.ndarray, *, field: str, gamma: float
) -> np.ndarray:
    state = np.asarray(conservative, dtype=np.float64)
    if state.ndim != 3 or state.shape[-1] != 4:
        raise ValueError("conservative animation state must have shape [T,N,4]")
    density = state[..., 0]
    if field == "density":
        return density
    if field != "pressure":
        raise ValueError(f"unsupported animation field: {field}")
    kinetic = 0.5 * (state[..., 1] ** 2 + state[..., 2] ** 2) / density
    return (gamma - 1.0) * (state[..., 3] - kinetic)


def _relative_improvement(
    raw_error: np.ndarray, corrected_error: np.ndarray, *, floor: float
) -> tuple[np.ndarray, np.ndarray]:
    magnitude = np.maximum(raw_error, corrected_error)
    resolved = magnitude > floor
    denominator = np.maximum(magnitude, floor)
    improvement = (raw_error - corrected_error) / denominator
    improvement = np.where(resolved, improvement, 0.0)
    return np.clip(improvement, -1.0, 1.0), resolved


def _load_bundle(
    path: Path, expected: Mapping[str, Any], contract: Mapping[str, Any]
) -> dict[str, Any]:
    if sha256_file(path) != expected["sha256"]:
        raise ValueError(f"animation bundle hash mismatch: {path.name}")
    with np.load(path, allow_pickle=False) as data:
        payload = {name: np.array(data[name], copy=True) for name in data.files}
    case_id = str(payload["case_id"].item())
    group_id = str(payload["split_group_id"].item())
    expected_case_id = str(expected["case_id"])
    case_parts = expected_case_id.split("_")
    if len(case_parts) != 3 or not case_parts[1].startswith("e"):
        raise ValueError(f"invalid registered animation case: {expected_case_id}")
    expected_group_id = f"strength_ood_{case_parts[1]}"
    resolution = tuple(int(value) for value in payload["native_resolution"])
    calls = [int(value) for value in payload["output_calls"]]
    times = [float(value) for value in payload["physical_times"]]
    frame_count = len(contract["output_calls"])
    expected_shape = (frame_count, resolution[0] * resolution[1], 4)
    if (
        case_id != expected_case_id
        or group_id != expected_group_id
        or resolution != tuple(contract["native_resolution"])
        or calls != contract["output_calls"]
        or not np.allclose(times, contract["physical_times"], atol=0.0, rtol=0.0)
        or payload["nodes"].shape != (resolution[0] * resolution[1], 2)
        or payload["volumes"].shape != (resolution[0] * resolution[1],)
        or any(
            payload[name].shape != expected_shape
            for name in (
                "truth_conservative",
                "raw_shadow_conservative",
                "corrected_conservative",
            )
        )
        or not all(
            np.isfinite(payload[name]).all()
            for name in (
                "truth_conservative",
                "raw_shadow_conservative",
                "corrected_conservative",
            )
        )
    ):
        raise ValueError(f"animation bundle contract mismatch: {path.name}")
    return payload


def _animation_arrays(
    bundle: Mapping[str, Any], *, field: str, contract: Mapping[str, Any]
) -> dict[str, np.ndarray]:
    gamma = float(np.asarray(bundle["gamma"]).item())
    truth = _primitive_field(bundle["truth_conservative"], field=field, gamma=gamma)
    raw = _primitive_field(bundle["raw_shadow_conservative"], field=field, gamma=gamma)
    corrected = _primitive_field(
        bundle["corrected_conservative"], field=field, gamma=gamma
    )
    raw_error = np.abs(raw - truth)
    corrected_error = np.abs(corrected - truth)
    improvement, resolved = _relative_improvement(
        raw_error,
        corrected_error,
        floor=float(contract["relative_improvement_denominator_floor"]),
    )
    nx, ny = (int(value) for value in bundle["native_resolution"])

    def grid(value: np.ndarray) -> np.ndarray:
        return value.reshape(value.shape[0], ny, nx)

    return {
        "truth": grid(truth),
        "raw": grid(raw),
        "corrected": grid(corrected),
        "raw_error": grid(raw_error),
        "corrected_error": grid(corrected_error),
        "improvement": grid(improvement),
        "resolved": grid(resolved),
    }


def _saturation_summary(
    arrays: Mapping[str, np.ndarray], *, field: str, contract: Mapping[str, Any]
) -> dict[str, float]:
    field_min, field_max = (float(value) for value in contract["field_limits"][field])
    error_min, error_max = (float(value) for value in contract["absolute_error_limits"])
    states = np.concatenate(
        [arrays[name].reshape(-1) for name in ("truth", "raw", "corrected")]
    )
    errors = np.concatenate(
        [arrays[name].reshape(-1) for name in ("raw_error", "corrected_error")]
    )
    return {
        "field_below_fraction": float(np.mean(states < field_min)),
        "field_above_fraction": float(np.mean(states > field_max)),
        "error_below_fraction": float(np.mean(errors < error_min)),
        "error_above_fraction": float(np.mean(errors > error_max)),
        "improvement_floor_fraction": float(np.mean(~arrays["resolved"].astype(bool))),
    }


def _render_one(
    bundle: Mapping[str, Any],
    *,
    field: str,
    contract: Mapping[str, Any],
    output_dir: Path,
) -> tuple[list[Path], dict[str, float]]:
    arrays = _animation_arrays(bundle, field=field, contract=contract)
    field_min, field_max = (float(value) for value in contract["field_limits"][field])
    error_min, error_max = (float(value) for value in contract["absolute_error_limits"])
    improvement_min, improvement_max = (
        float(value) for value in contract["relative_improvement_limits"]
    )
    candidate_label = str(contract.get("candidate_label", "anchored correction"))
    output_stem = str(contract.get("output_stem", "shadow_anchored_h30"))
    error_norm = colors.LogNorm(vmin=error_min, vmax=error_max, clip=True)
    case_id = str(bundle["case_id"].item())
    times = np.asarray(bundle["physical_times"], dtype=np.float64)
    extent = (0.0, 2.0, 0.0, 1.0)
    fig, axes = plt.subplots(2, 3, figsize=(12.4, 5.5), constrained_layout=True)
    state_images = [
        axes[0, column].imshow(
            arrays[name][0],
            origin="lower",
            extent=extent,
            cmap="viridis",
            vmin=field_min,
            vmax=field_max,
            interpolation="nearest",
            animated=True,
        )
        for column, name in enumerate(("truth", "raw", "corrected"))
    ]
    error_images = [
        axes[1, column].imshow(
            np.maximum(arrays[name][0], error_min),
            origin="lower",
            extent=extent,
            cmap="magma",
            norm=error_norm,
            interpolation="nearest",
            animated=True,
        )
        for column, name in enumerate(("raw_error", "corrected_error"))
    ]
    improvement_image = axes[1, 2].imshow(
        arrays["improvement"][0],
        origin="lower",
        extent=extent,
        cmap="RdBu_r",
        vmin=improvement_min,
        vmax=improvement_max,
        interpolation="nearest",
        animated=True,
    )
    titles = (
        f"truth {field}",
        f"raw-shadow {field}",
        f"{candidate_label} {field}",
        "raw absolute error",
        "corrected absolute error",
        "signed relative improvement",
    )
    for axis, title in zip(axes.flat, titles, strict=True):
        axis.set_title(title)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
    fig.colorbar(state_images[0], ax=axes[0, :], shrink=0.86, label=field)
    fig.colorbar(error_images[0], ax=axes[1, :2], shrink=0.86, label="absolute error")
    fig.colorbar(
        improvement_image,
        ax=axes[1, 2],
        shrink=0.86,
        label="positive means correction helps",
    )
    title = fig.suptitle(f"{case_id} | {field} | t={times[0]:.2f}")

    def update(frame: int) -> list[Any]:
        for image, name in zip(
            state_images, ("truth", "raw", "corrected"), strict=True
        ):
            image.set_data(arrays[name][frame])
        for image, name in zip(
            error_images, ("raw_error", "corrected_error"), strict=True
        ):
            image.set_data(np.maximum(arrays[name][frame], error_min))
        improvement_image.set_data(arrays["improvement"][frame])
        title.set_text(f"{case_id} | {field} | t={times[frame]:.2f}")
        return [*state_images, *error_images, improvement_image, title]

    movie = animation.FuncAnimation(
        fig,
        update,
        frames=len(times),
        interval=1000 / FPS,
        blit=False,
    )
    gif_path = output_dir / f"{case_id}_{field}_{output_stem}.gif"
    movie.save(gif_path, writer=animation.PillowWriter(fps=FPS), dpi=DPI)
    update(len(times) - 1)
    final_path = output_dir / f"{case_id}_{field}_{output_stem}_final.png"
    fig.savefig(final_path, dpi=160)
    plt.close(fig)
    return [gif_path, final_path], _saturation_summary(
        arrays, field=field, contract=contract
    )


def render(rollout_path: Path, manifest_path: Path, output_dir: Path) -> dict[str, Any]:
    rollout, manifest = _load_contract(rollout_path, manifest_path)
    output_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    generated = []
    for bundle_row in manifest["bundles"]:
        bundle_path = manifest_path.parent / bundle_row["path"]
        bundle = _load_bundle(bundle_path, bundle_row, manifest["contract"])
        for field in FIELDS:
            paths, saturation = _render_one(
                bundle,
                field=field,
                contract=manifest["contract"],
                output_dir=output_dir,
            )
            generated.extend(paths)
            rows.append(
                {
                    "case_id": bundle_row["case_id"],
                    "field": field,
                    "saturation": saturation,
                    "files": [
                        {"path": path.name, "sha256": sha256_file(path)}
                        for path in paths
                    ],
                }
            )
    payload = with_payload_sha256(
        {
            "schema": RENDER_SCHEMA,
            "working_id": rollout["working_id"],
            "rollout_sha256": sha256_file(rollout_path),
            "rollout_payload_sha256": rollout["payload_sha256"],
            "bundle_manifest_sha256": sha256_file(manifest_path),
            "bundle_manifest_payload_sha256": manifest["payload_sha256"],
            "contract": manifest["contract"],
            "fps": FPS,
            "dpi": DPI,
            "rendered": rows,
            "generated_file_count": len(generated),
            "selection_or_gate_effect": False,
            "interpretation": (
                "Positive signed relative improvement means the displayed candidate "
                "has lower pointwise absolute field error than the raw shadow."
            ),
        }
    )
    atomic_write_json(output_dir / "animation_render_manifest.json", payload)
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--bundle-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = render(args.rollout, args.bundle_manifest, args.output_dir)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
