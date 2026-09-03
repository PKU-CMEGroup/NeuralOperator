from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.time_dependent_no.train_pcno_naca0012 import _self_hashed
from scripts.time_dependent_no.visualize_pcno_naca0012_rollout import (
    HORIZONS,
    REPLAY_FILE,
    REPLAY_MANIFEST_FILE,
    REPLAY_SCHEMA,
    SEEDS,
    STATIC_HORIZONS,
    ReplayBundle,
    _claim_boundary,
    _file_record,
    _quad_face_values,
    _snapshot_replay_check,
    animation_horizons,
    color_limits,
    load_replay,
)


def _synthetic_replay(root: Path) -> None:
    coordinates = np.asarray(
        [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]],
        dtype=np.float64,
    )
    quads = np.asarray([[0, 1, 2, 3]], dtype=np.int64)
    horizons = np.asarray(HORIZONS, dtype=np.int64)
    reference = np.broadcast_to(
        np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        (len(HORIZONS), 4),
    ).copy()
    predicted = np.stack(
        [
            reference + np.float32((offset + 1) * horizons[:, None] / 208.0)
            for offset in range(3)
        ],
        axis=0,
    ).astype(np.float32)
    scale = 2.0
    errors = np.sqrt(
        np.mean(
            np.square((predicted.astype(np.float64) - reference[None, ...]) / scale),
            axis=2,
        )
    )
    root.mkdir()
    np.savez_compressed(
        root / REPLAY_FILE,
        schema=np.asarray(REPLAY_SCHEMA),
        anchor=np.asarray(1234, dtype=np.int64),
        horizons=horizons,
        frame_indices=1234 + horizons,
        seeds=np.asarray(SEEDS, dtype=np.int64),
        coordinates=coordinates,
        quads=quads,
        airfoil_mask=np.asarray([1, 1, 1, 1], dtype=np.uint8),
        reference_density=reference,
        predicted_density=predicted,
        density_state_scale=np.asarray(scale, dtype=np.float64),
        density_error_rmse=errors,
    )
    manifest = _self_hashed(
        {
            "schema": REPLAY_SCHEMA,
            "anchor": 1234,
            "replay_file": _file_record(root / REPLAY_FILE, root),
            "claim_boundary": _claim_boundary(),
        }
    )
    (root / REPLAY_MANIFEST_FILE).write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )


def test_quad_face_values_are_flat_native_quad_means() -> None:
    values = np.asarray([1.0, 3.0, 5.0, 7.0, 11.0])
    quads = np.asarray([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int64)
    np.testing.assert_allclose(_quad_face_values(values, quads), [4.0, 6.5])
    assert animation_horizons(3)[-2:] == (207, 208)
    assert animation_horizons(2)[-1] == 208


def test_snapshot_replay_check_records_cuda_roundoff() -> None:
    reference = np.ones((len(HORIZONS), 3), dtype=np.float64)
    prediction = np.ones((len(HORIZONS), 3), dtype=np.float32)
    snapshots: dict[str, np.ndarray] = {}
    for horizon in STATIC_HORIZONS:
        snapshots[f"reference_anchor_1234_h{horizon:03d}"] = np.ones(
            (3, 5), dtype=np.float64
        )
        snapshots[f"pcno_seed_17_anchor_1234_h{horizon:03d}"] = np.ones(
            (3, 5), dtype=np.float32
        )
    prediction[1, 0] = np.nextafter(
        np.float32(1.0), np.float32(2.0), dtype=np.float32
    )
    checks = _snapshot_replay_check(
        seed=17,
        anchor=1234,
        prediction=prediction,
        reference=reference,
        snapshots=snapshots,
    )
    assert checks[0]["prediction_bitwise_equal"] is False
    assert checks[0]["prediction_tolerance_pass"] is True

    prediction[1, 0] = np.float32(1.001)
    checks = _snapshot_replay_check(
        seed=17,
        anchor=1234,
        prediction=prediction,
        reference=reference,
        snapshots=snapshots,
    )
    assert checks[0]["prediction_tolerance_pass"] is False


def test_load_replay_verifies_canonical_horizons_and_error_curve(
    tmp_path: Path,
) -> None:
    root = tmp_path / "replay"
    _synthetic_replay(root)
    bundle = load_replay(root)

    assert isinstance(bundle, ReplayBundle)
    assert bundle.anchor == 1234
    assert bundle.reference_density.shape == (209, 4)
    assert bundle.predicted_density.shape == (3, 209, 4)
    assert bundle.manifest["claim_boundary"]["clean_PCNO_only"] is True
    assert (
        bundle.manifest["claim_boundary"]["causal_failure_mechanism_identified"]
        is False
    )

    with np.load(root / REPLAY_FILE, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays["horizons"] = np.arange(1, 210, dtype=np.int64)
    np.savez_compressed(root / REPLAY_FILE, **arrays)
    manifest = json.loads((root / REPLAY_MANIFEST_FILE).read_text(encoding="utf-8"))
    manifest.pop("canonical_payload_sha256")
    manifest["replay_file"] = _file_record(root / REPLAY_FILE, root)
    manifest = _self_hashed(manifest)
    (root / REPLAY_MANIFEST_FILE).write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="array contract differs"):
        load_replay(root)


def test_color_limits_cover_every_seed_and_are_symmetric_for_error(
    tmp_path: Path,
) -> None:
    root = tmp_path / "replay"
    _synthetic_replay(root)
    bundle = load_replay(root)
    limits = color_limits(bundle, bundle.quads)

    assert limits["density"] == (1.0, 7.0)
    assert limits["signed_error"] == (-1.5, 1.5)
    assert limits["error_curve"] == (0.0, 1.5750000000000002)
