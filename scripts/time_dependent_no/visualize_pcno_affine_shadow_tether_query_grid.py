#!/usr/bin/env python3
"""Render the registered A33 query-grid comparison bundles."""

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
from matplotlib import animation

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.time_dependent_no.evaluate_pcno_affine_shadow_tether_query_grid import (
    ANIMATION_CONTRACT,
    ANIMATION_SCHEMA,
    RESULT_SCHEMA,
    WORKING_ID,
)
from scripts.time_dependent_no.visualize_pcno_response_filtered_block import (
    DPI,
    FIELDS,
    FPS,
    _primitive_field,
    _relative_improvement,
)
from utility.time_dependent_no.pcno_artifacts import (
    atomic_write_json_with_paths as atomic_write_json,
)
from utility.time_dependent_no.pcno_artifacts import sha256_file
from utility.time_dependent_no.pcno_cross_resolution_teacher_forced import (
    verify_payload_sha256,
    with_payload_sha256,
)

RENDER_SCHEMA = "pcno_sp19_affine_shadow_tether_query_grid_render_v1"
STATE_ARRAYS = (
    "truth_conservative",
    "direct_query_conservative",
    "matched_direct_conservative",
    "raw_transfer_conservative",
    "corrected_transfer_conservative",
)


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
    if (
        rollout.get("schema") != RESULT_SCHEMA
        or rollout.get("working_id") != WORKING_ID
        or rollout.get("recurrence_executed") is not True
        or rollout.get("animations_rendered") is not False
    ):
        raise ValueError("rollout is not the registered unrendered A33 execution")
    if (
        manifest.get("schema") != ANIMATION_SCHEMA
        or manifest.get("working_id") != WORKING_ID
        or manifest.get("contract") != ANIMATION_CONTRACT
        or manifest.get("inference_or_gate_input") is not False
        or rollout.get("animation_bundle_manifest") != manifest
        or rollout.get("animation_bundle_manifest_sha256") != sha256_file(manifest_path)
    ):
        raise ValueError("animation manifest differs from the registered A33 contract")
    return rollout, manifest


def _load_bundle(
    path: Path, expected: Mapping[str, Any], contract: Mapping[str, Any]
) -> dict[str, Any]:
    if sha256_file(path) != expected["sha256"]:
        raise ValueError(f"animation bundle hash mismatch: {path.name}")
    with np.load(path, allow_pickle=False) as data:
        payload = {name: np.array(data[name], copy=True) for name in data.files}
    case_id = str(payload["case_id"].item())
    parts = case_id.split("_")
    expected_group = f"strength_{parts[1]}" if len(parts) == 3 else None
    resolution = tuple(int(value) for value in payload["query_resolution"])
    calls = [int(value) for value in payload["output_calls"]]
    times = [float(value) for value in payload["physical_times"]]
    frame_count = len(contract["output_calls"])
    expected_shape = (frame_count, resolution[0] * resolution[1], 4)
    if (
        case_id != expected["case_id"]
        or str(payload["split_group_id"].item()) != expected_group
        or resolution != tuple(contract["query_resolution"])
        or calls != contract["output_calls"]
        or not np.allclose(times, contract["physical_times"], atol=0.0, rtol=0.0)
        or payload["nodes"].shape != (resolution[0] * resolution[1], 2)
        or payload["volumes"].shape != (resolution[0] * resolution[1],)
        or any(payload[name].shape != expected_shape for name in STATE_ARRAYS)
        or not all(np.isfinite(payload[name]).all() for name in STATE_ARRAYS)
    ):
        raise ValueError(f"animation bundle contract mismatch: {path.name}")
    return payload


def _animation_arrays(
    bundle: Mapping[str, Any], *, field: str, contract: Mapping[str, Any]
) -> dict[str, np.ndarray]:
    gamma = float(np.asarray(bundle["gamma"]).item())
    fields = {
        name.removesuffix("_conservative"): _primitive_field(
            bundle[name], field=field, gamma=gamma
        )
        for name in STATE_ARRAYS
    }
    raw_error = np.abs(fields["raw_transfer"] - fields["truth"])
    corrected_error = np.abs(fields["corrected_transfer"] - fields["truth"])
    improvement, resolved = _relative_improvement(
        raw_error,
        corrected_error,
        floor=float(contract["relative_improvement_denominator_floor"]),
    )
    nx, ny = (int(value) for value in bundle["query_resolution"])

    def grid(value: np.ndarray) -> np.ndarray:
        return value.reshape(value.shape[0], ny, nx)

    return {
        **{name: grid(value) for name, value in fields.items()},
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
        [
            arrays[name].reshape(-1)
            for name in (
                "truth",
                "direct_query",
                "matched_direct",
                "raw_transfer",
                "corrected_transfer",
            )
        ]
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
    improvement_min, improvement_max = (
        float(value) for value in contract["relative_improvement_limits"]
    )
    case_id = str(bundle["case_id"].item())
    times = np.asarray(bundle["physical_times"], dtype=np.float64)
    extent = (0.0, 2.0, 0.0, 1.0)
    fig, axes = plt.subplots(2, 3, figsize=(12.4, 5.5), constrained_layout=True)
    state_names = (
        "truth",
        "direct_query",
        "matched_direct",
        "raw_transfer",
        "corrected_transfer",
    )
    state_axes = (*axes[0, :], axes[1, 0], axes[1, 1])
    state_images = [
        axis.imshow(
            arrays[name][0],
            origin="lower",
            extent=extent,
            cmap="viridis",
            vmin=field_min,
            vmax=field_max,
            interpolation="nearest",
            animated=True,
        )
        for axis, name in zip(state_axes, state_names, strict=True)
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
        "direct 500x200",
        "direct after R/P initialization",
        "raw native then prolong",
        "A32 native then prolong",
        "A32 vs raw signed improvement",
    )
    for axis, title_text in zip(axes.flat, titles, strict=True):
        axis.set_title(title_text)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
    fig.colorbar(state_images[0], ax=state_axes, shrink=0.86, label=field)
    fig.colorbar(
        improvement_image,
        ax=axes[1, 2],
        shrink=0.86,
        label="positive means A32 transfer helps",
    )
    title = fig.suptitle(f"{case_id} | {field} | t={times[0]:.2f}")

    def update(frame: int) -> list[Any]:
        for image, name in zip(state_images, state_names, strict=True):
            image.set_data(arrays[name][frame])
        improvement_image.set_data(arrays["improvement"][frame])
        title.set_text(f"{case_id} | {field} | t={times[frame]:.2f}")
        return [*state_images, improvement_image, title]

    movie = animation.FuncAnimation(
        fig,
        update,
        frames=len(times),
        interval=1000 / FPS,
        blit=False,
    )
    stem = str(contract["output_stem"])
    gif_path = output_dir / f"{case_id}_{field}_{stem}.gif"
    movie.save(gif_path, writer=animation.PillowWriter(fps=FPS), dpi=DPI)
    update(len(times) - 1)
    final_path = output_dir / f"{case_id}_{field}_{stem}_final.png"
    fig.savefig(final_path, dpi=160)
    plt.close(fig)
    return [gif_path, final_path], _saturation_summary(
        arrays, field=field, contract=contract
    )


def render(rollout_path: Path, manifest_path: Path, output_dir: Path) -> dict[str, Any]:
    rollout, manifest = _load_contract(rollout_path, manifest_path)
    output_dir.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []
    generated: list[Path] = []
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
                        {
                            "path": path.name,
                            "sha256": sha256_file(path),
                            "bytes": path.stat().st_size,
                        }
                        for path in paths
                    ],
                }
            )
    payload = with_payload_sha256(
        {
            "schema": RENDER_SCHEMA,
            "working_id": WORKING_ID,
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
                "Positive signed relative improvement means A32 transfer has lower "
                "pointwise absolute field error than raw native transfer. Direct arms "
                "are visual comparators only."
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
