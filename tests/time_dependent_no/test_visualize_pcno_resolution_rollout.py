from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.time_dependent_no.visualize_pcno_resolution_rollout import (
    BUNDLE_SCHEMA,
    SIGNED_LOG_LINEAR_FRACTION,
    AnimationBundle,
    _signed_log_norm,
    animation_color_limits,
    animation_frame_indices,
    load_bundle,
)


def _states(frames: int, nx: int, ny: int, offset: float) -> np.ndarray:
    x = np.linspace(0.0, 1.0, nx, dtype=np.float32)
    y = np.linspace(0.0, 1.0, ny, dtype=np.float32)
    field = x[None, :] + y[:, None]
    result = np.ones((frames, ny, nx, 4), dtype=np.float32)
    for frame in range(frames):
        result[frame, ..., 0] = field + offset * frame
    return result.reshape(frames, nx * ny, 4)


def _write_bundle(
    path: Path,
    *,
    protocols: tuple[str, ...],
    prediction_offset: float,
) -> None:
    times = np.asarray([0.0, 0.1, 0.2], dtype=np.float64)
    resolutions = ("4x2", "8x4")
    arrays: dict[str, np.ndarray] = {
        "physical_times": times,
        "state_scale": np.ones(4, dtype=np.float64),
        "metadata_json": np.asarray(
            json.dumps(
                {
                    "schema": BUNDLE_SCHEMA,
                    "case_id": "synthetic",
                    "resolutions": resolutions,
                    "protocols": protocols,
                }
            )
        ),
    }
    for resolution in resolutions:
        nx, ny = (int(value) for value in resolution.split("x"))
        reference = _states(times.size, nx, ny, 0.01)
        arrays[f"reference__{resolution}"] = reference
        arrays[f"volumes__{resolution}"] = np.ones(nx * ny, dtype=np.float64)
        for index, protocol in enumerate(protocols, start=1):
            prediction = reference.copy()
            prediction[..., 0] += prediction_offset * index
            arrays[f"prediction__{protocol}__{resolution}"] = prediction
            arrays[f"metric__{protocol}__{resolution}"] = np.full(
                times.size, prediction_offset * index, dtype=np.float64
            )
    np.savez_compressed(path, **arrays)


def test_animation_frame_indices_keeps_final_frame() -> None:
    assert animation_frame_indices(7, 2) == [0, 2, 4, 6]
    assert animation_frame_indices(8, 3) == [0, 3, 6, 7]


def test_load_bundle_and_color_limits(tmp_path: Path) -> None:
    physical_path = tmp_path / "physical.npz"
    all_normal_path = tmp_path / "all_normal.npz"
    _write_bundle(
        physical_path,
        protocols=("physical", "all_normal", "swapped_boundary_kinds"),
        prediction_offset=0.02,
    )
    _write_bundle(
        all_normal_path,
        protocols=("all_normal",),
        prediction_offset=0.03,
    )

    physical = load_bundle(physical_path)
    all_normal = load_bundle(all_normal_path)
    state_min, state_max, resolution_limit, ablation_limit = animation_color_limits(
        physical, all_normal
    )

    assert isinstance(physical, AnimationBundle)
    assert physical.resolutions == ("4x2", "8x4")
    assert state_min == 0.0
    assert state_max > 2.0
    np.testing.assert_allclose(resolution_limit, 0.02, rtol=0.0, atol=1.0e-6)
    np.testing.assert_allclose(ablation_limit, 0.06, rtol=0.0, atol=1.0e-6)
    norm = _signed_log_norm(ablation_limit)
    np.testing.assert_allclose(
        (norm(-ablation_limit), norm(0.0), norm(ablation_limit)),
        (0.0, 0.5, 1.0),
        rtol=0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        norm.linthresh,
        SIGNED_LOG_LINEAR_FRACTION * ablation_limit,
        rtol=0.0,
        atol=1.0e-12,
    )
