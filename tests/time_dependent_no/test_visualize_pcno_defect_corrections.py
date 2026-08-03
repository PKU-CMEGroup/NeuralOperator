from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.time_dependent_no.visualize_pcno_defect_corrections import (
    CUMULATIVE_LIMIT,
    INSTANT_LIMIT,
    RESULT_SCHEMA,
    _series,
    _verify_results,
    animate_component,
)
from utility.time_dependent_no.pcno_artifacts import sha256_file


def test_series_aggregates_case_rows_at_each_time() -> None:
    rows = [
        {
            "physical_time": str(time),
            "value": str(value),
            "arm": "combined",
            "resolution": "250x100",
        }
        for time, value in ((1.0, 1.0), (1.0, 3.0), (2.0, 2.0), (2.0, 6.0))
    ]
    times, median, lower, upper = _series(
        rows, "value", arm="combined", resolution="250x100"
    )
    np.testing.assert_allclose(times, (1.0, 2.0))
    np.testing.assert_allclose(median, (2.0, 4.0))
    np.testing.assert_allclose(lower, (1.5, 3.0))
    np.testing.assert_allclose(upper, (2.5, 5.0))


def test_animation_uses_fixed_residual_scale_limits(tmp_path: Path) -> None:
    nx, ny = 4, 2
    x = (np.arange(nx, dtype=np.float64) + 0.5) / nx
    y = (np.arange(ny, dtype=np.float64) + 0.5) / ny
    xx, yy = np.meshgrid(x, y)
    nodes = np.column_stack((xx.reshape(-1), yy.reshape(-1)))
    frames = 2
    vector = np.zeros((frames, nx * ny, 4), dtype=np.float32)
    vector[1, :, 0] = 2.0
    payload = {
        "schema": np.asarray(RESULT_SCHEMA),
        "family": np.asarray("dynamic_fv"),
        "case_id": np.asarray("synthetic"),
        "resolution": np.asarray("4x2"),
        "mode": np.asarray("free_rollout"),
        "nodes": nodes,
        "physical_times": np.asarray([0.1, 0.2]),
        "residual_scale": np.asarray([2.0, 1.0, 1.0, 1.0]),
        "node_type": np.zeros(nx * ny, dtype=np.int64),
        "sensor": np.zeros((frames, nx * ny), dtype=np.float32),
        "baseline_accepted": np.ones(frames, dtype=bool),
        "combined_accepted": np.ones(frames, dtype=bool),
        "expected_rollout_calls": np.asarray(frames),
        "baseline_valid_length": np.asarray(frames),
        "combined_valid_length": np.asarray(frames),
        "rollout_complete": np.asarray(True),
    }
    for name in (
        "true_increment",
        "baseline_increment",
        "combined_increment",
        "baseline_residual_error",
        "combined_residual_error",
        "persistent_correction",
        "local_correction",
        "accumulated_correction",
        "accumulated_baseline_residual_error",
        "accumulated_combined_residual_error",
    ):
        payload[name] = vector
    payload_path = tmp_path / "payload.npz"
    np.savez_compressed(payload_path, **payload)
    output, rows = animate_component(payload_path, tmp_path / "animations", component=0)
    assert output.is_file()
    limits = {row["field"]: row["fixed_limit"] for row in rows}
    assert limits["true_increment"] == INSTANT_LIMIT
    assert limits["accumulated_correction"] == CUMULATIVE_LIMIT
    true_row = next(row for row in rows if row["field"] == "true_increment")
    assert true_row["maximum_absolute_scaled_value"] == 1.0


def test_verify_results_rejects_unlisted_visual_payload(tmp_path: Path) -> None:
    results = tmp_path / "results"
    payload_dir = results / "visual_payloads"
    payload_dir.mkdir(parents=True)
    required = (
        "metrics.csv",
        "component_metrics.csv",
        "band_metrics.csv",
        "front_metrics.csv",
        "correction_metrics.csv",
        "completion.csv",
        "teacher_completion.csv",
        "calibration.csv",
        "calibration_coefficients.npz",
        "reference_checks.csv",
        "visual_payloads/listed.npz",
    )
    for relative in required:
        path = results / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode())
    output_hashes = {relative: sha256_file(results / relative) for relative in required}
    (results / "summary.json").write_text(
        json.dumps(
            {
                "schema": RESULT_SCHEMA,
                "status": "complete",
                "scientific_interpretation_allowed": True,
                "family": "dynamic_fv",
                "output_hashes": output_hashes,
            }
        ),
        encoding="utf-8",
    )
    _verify_results(results)
    (payload_dir / "unlisted.npz").write_bytes(b"unlisted")
    with pytest.raises(ValueError, match="payload inventory"):
        _verify_results(results)
