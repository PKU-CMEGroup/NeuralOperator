from __future__ import annotations

import hashlib
import io
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.time_dependent_no import audit_realm_planardet
from scripts.time_dependent_no.acquire_realm_planardet import acquire_entry
from utility.time_dependent_no.realm_benchmark import ManifestEntry
from utility.time_dependent_no.realm_planardet import (
    CANONICAL_SPATIAL_SHAPE_YX,
    METADATA_KEYS,
    PLANARDET_FIELDS,
    PLANARDET_TEST_GROUPS_METADATA_ONLY,
    PLANARDET_TRAIN_GROUPS,
    PLANARDET_VAL_GROUPS,
    RELEASED_SPATIAL_SHAPE_XY,
    TIME_CADENCE,
    TIME_FIRST,
    load_planardet_metadata,
    sha256_file,
    validate_local_open_tree,
)
from utility.time_dependent_no.realm_planardet_metrics import (
    compare_planardet_structure,
    shock_attached_transverse_spectrum,
)
from utility.time_dependent_no.realm_planardet_runtime import (
    PlanarDetNormalizerBundle,
    PlanarDetTrainingContract,
    competence_gate,
    grouped_planardet_mse,
    inverse_box_cox_min_margin,
    load_normalizer_bundle,
    planardet_boundary_mask,
    scheduled_step_window,
    summarize_planardet_predictions,
    validation_control_summary,
)


def _metadata_payload() -> dict[str, np.ndarray]:
    nx, ny = RELEASED_SPATIAL_SHAPE_XY
    x = (0.1298875 - np.arange(nx) * 2.5e-5).astype(np.float32)
    y = (0.0097875 - np.arange(ny) * 2.5e-5).astype(np.float32)
    coords = np.empty((2, nx, ny), dtype=np.float32)
    coords[0] = x[:, None]
    coords[1] = y[None, :]
    times = np.asarray([str(TIME_FIRST + frame * TIME_CADENCE) for frame in range(50)])
    return {
        "coords": coords,
        "times": times,
        "variables": np.asarray([f"{name}.npy" for name in PLANARDET_FIELDS]),
        "train_groups": np.asarray(PLANARDET_TRAIN_GROUPS),
        "val_groups": np.asarray(PLANARDET_VAL_GROUPS),
        "test_groups": np.asarray(PLANARDET_TEST_GROUPS_METADATA_ONLY),
        "spatial_size": np.asarray(RELEASED_SPATIAL_SHAPE_XY),
        "num_chemical": np.asarray(8),
        "num_temperature": np.asarray(1),
        "num_density": np.asarray(1),
        "num_velocity": np.asarray(2),
        "num_pressure": np.asarray(1),
    }


def _contract() -> PlanarDetTrainingContract:
    digest = "1" * 64
    return PlanarDetTrainingContract(
        run_id="d000_planardet_unit",
        seed=7,
        width=96,
        total_steps=200,
        one_call_steps=49,
        validation_interval=50,
        checkpoint_eligible_from_step=100,
        competence_npe_ceiling=0.5,
        max_wall_seconds=3600.0,
        smoke_result_sha256=digest,
        normalizer_arrays_sha256=digest,
        data_audit_final_manifest_sha256=digest,
        open_manifest_payload_sha256=digest,
        persistence_npe=1.0,
        linear_extrapolation_npe=2.0,
    )


def test_data_audit_source_manifest_covers_package_time_imports() -> None:
    manifest = audit_realm_planardet._source_manifest()
    paths = {row["path"] for row in manifest["files"]}
    assert {
        "utility/__init__.py",
        "utility/adam.py",
        "utility/losses.py",
        "utility/normalizer.py",
        "utility/time_dependent_no/__init__.py",
        "utility/time_dependent_no/realm_benchmark.py",
        "utility/time_dependent_no/realm_planardet.py",
        "scripts/time_dependent_no/audit_realm_planardet.py",
        "scripts/time_dependent_no/acquire_realm_planardet.py",
    } == paths


def test_metadata_loader_resolves_released_and_canonical_axes(tmp_path: Path) -> None:
    path = tmp_path / "data.npz"
    payload = _metadata_payload()
    assert set(payload) == METADATA_KEYS
    np.savez(path, **payload)

    metadata = load_planardet_metadata(path)

    assert metadata.released_coords_xy.shape == (2, 832, 384)
    assert metadata.canonical_coords_yx.shape == (2, 384, 832)
    assert metadata.canonical_coords_yx.flags.c_contiguous
    assert np.array_equal(
        metadata.canonical_coords_yx[0], metadata.released_coords_xy[1].T
    )
    assert np.array_equal(
        metadata.canonical_coords_yx[1], metadata.released_coords_xy[0].T
    )
    assert metadata.x[0] > metadata.x[-1]
    assert metadata.y[0] > metadata.y[-1]
    assert metadata.dx == pytest.approx(2.5e-5, rel=1.0e-3)
    assert metadata.dy == pytest.approx(2.5e-5, rel=1.0e-3)


def test_closed_open_tree_rejects_any_test_component(tmp_path: Path) -> None:
    root = tmp_path / "release"
    (root / "data" / "test").mkdir(parents=True)
    (root / "data" / "test" / "forbidden.npz").write_bytes(b"sealed")
    with pytest.raises(ValueError, match="sealed test path"):
        validate_local_open_tree(root, ())


def test_atomic_acquisition_verifies_hash_and_cleans_failed_partial(
    tmp_path: Path,
) -> None:
    content = b"registered open object"
    entry = ManifestEntry(
        path="data/train/example.npz",
        size=len(content),
        oid=hashlib.sha256(content).hexdigest(),
        lfs=True,
    )
    row, reused = acquire_entry(
        tmp_path,
        entry,
        open_url=lambda _: io.BytesIO(content),
    )
    assert not reused
    assert row["content_sha256"] == entry.oid
    assert (tmp_path / "data" / "train" / "example.npz").read_bytes() == content

    bad = ManifestEntry(
        path="data/train/bad.npz",
        size=len(content),
        oid="0" * 64,
        lfs=True,
    )
    with pytest.raises(ValueError, match="object mismatch"):
        acquire_entry(tmp_path, bad, open_url=lambda _: io.BytesIO(content))
    assert not (tmp_path / "data" / "train" / "bad.npz").exists()
    assert not list((tmp_path / "data" / "train").glob(".bad.npz.part.*"))


def test_preregistration_roundtrip_and_fail_closed_contract() -> None:
    contract = _contract()
    payload = contract.payload()
    assert PlanarDetTrainingContract.from_payload(payload) == contract
    assert contract.frozen_training_config()["model"]["layers"] == [96] * 5

    changed = dict(payload)
    changed["width"] = 128
    with pytest.raises(ValueError, match="digest differs"):
        PlanarDetTrainingContract.from_payload(changed)

    with pytest.raises(ValueError, match="strictly beat"):
        PlanarDetTrainingContract(
            **{
                **contract.__dict__,
                "competence_npe_ceiling": contract.persistence_npe,
            }
        )


def test_phase_schedules_cover_every_window_before_repeating() -> None:
    one_call = [
        scheduled_step_window(step, one_call_steps=49, seed=3) for step in range(1, 50)
    ]
    two_call = [
        scheduled_step_window(step, one_call_steps=49, seed=3) for step in range(50, 98)
    ]
    assert {row.frame_start for row in one_call} == set(range(49))
    assert {row.frame_start for row in two_call} == set(range(48))
    assert {row.calls for row in one_call} == {1}
    assert {row.calls for row in two_call} == {2}
    assert all(sorted(row.case_indices) == list(range(7)) for row in one_call)
    assert scheduled_step_window(50, one_call_steps=49, seed=3) == two_call[0]


def test_grouped_loss_and_controls_have_exact_group_weighting() -> None:
    truth = torch.zeros(1, 13, 2, 3)
    prediction = torch.arange(1, 14, dtype=torch.float32)[None, :, None, None].expand(
        -1, -1, 2, 3
    )
    loss, groups = grouped_planardet_mse(prediction, truth)
    expected = {
        "chem": torch.arange(1, 9, dtype=torch.float32).square().mean(),
        "T": torch.tensor(9.0).square(),
        "rho": torch.tensor(10.0).square(),
        "u": torch.tensor([11.0, 12.0]).square().mean(),
        "p": torch.tensor(13.0).square(),
    }
    assert set(groups) == set(expected)
    assert all(torch.equal(groups[name], value) for name, value in expected.items())
    assert torch.equal(loss, sum(expected.values()))

    increments = torch.arange(1, 14, dtype=torch.float32)[None, :, None, None]
    sequence = torch.arange(4, dtype=torch.float32)[:, None, None, None] * increments
    sequence = sequence.expand(-1, -1, 2, 3).clone()
    summary = validation_control_summary(sequence)
    assert summary["linear_normalized"]["realm_npe_mean"] == pytest.approx(
        summary["persistence"]["realm_npe_mean"] / 3.0
    )


def test_normalizer_bundle_binds_coordinates_and_inverse_margin(
    tmp_path: Path,
) -> None:
    coordinates = np.zeros((2, *CANONICAL_SPATIAL_SHAPE_YX), dtype=np.float32)
    arrays = {
        "primary_mean": np.zeros(13, dtype=np.float64),
        "primary_std": np.ones(13, dtype=np.float64),
        "primary_scale": np.ones(13, dtype=np.float64),
        "source_sensitivity_mean": np.zeros(13, dtype=np.float64),
        "source_sensitivity_std": np.ones(13, dtype=np.float64),
        "source_sensitivity_scale": np.ones(13, dtype=np.float64),
        "raw_train_mean": np.zeros(13, dtype=np.float64),
        "raw_train_std": np.ones(13, dtype=np.float64),
        "train_max_abs": np.ones(13, dtype=np.float64),
        "canonical_coordinates_yx": coordinates,
    }
    path = tmp_path / "normalizer.npz"
    np.savez(path, **arrays)
    bundle = load_normalizer_bundle(
        path,
        expected_sha256=sha256_file(path),
        expected_coordinates_yx=coordinates,
    )
    state = torch.zeros(1, 13, 2, 3)
    state[:, 0] = -11.0
    assert inverse_box_cox_min_margin(state, bundle, channel_axis=1) == pytest.approx(
        -0.1
    )

    drifted = coordinates.copy()
    drifted[0, 0, 0] = 1.0
    with pytest.raises(ValueError, match="coordinates differ"):
        load_normalizer_bundle(
            path,
            expected_sha256=sha256_file(path),
            expected_coordinates_yx=drifted,
        )


def test_prediction_summary_and_competence_gate_include_pmax_memory() -> None:
    calls, height, width = 3, 5, 6
    truth = torch.zeros(calls, 13, height, width)
    truth[:, 8:10] = 1.0
    truth[:, 12] = torch.arange(2, 5, dtype=torch.float32)[:, None, None]
    current = truth.clone()
    current[:, 12] -= 1.0
    bundle = PlanarDetNormalizerBundle(
        mean=torch.zeros(13, dtype=torch.float64),
        scale=torch.ones(13, dtype=torch.float64),
        train_max_abs=torch.full((13,), 10.0, dtype=torch.float64),
        canonical_coordinates_yx=torch.empty(0),
        source_sha256="1" * 64,
    )
    x = torch.arange(width, dtype=torch.float32) * 1.0e-3
    y = torch.arange(height, dtype=torch.float32) * 1.0e-3
    boundary = planardet_boundary_mask(x, y)
    summary = summarize_planardet_predictions(
        truth,
        truth,
        truth,
        truth,
        current,
        bundle=bundle,
        boundary_mask=boundary,
    )
    assert summary["realm_npe_mean"] == 0.0
    assert summary["all_pMax_nondecreasing_from_input"]
    assert competence_gate(summary, _contract())["all_gates_pass"]

    violating = truth.clone()
    violating[1, 12] = current[1, 12] - 1.0
    violation_summary = summarize_planardet_predictions(
        violating,
        truth,
        violating,
        truth,
        current,
        bundle=bundle,
        boundary_mask=boundary,
    )
    assert not violation_summary["all_pMax_nondecreasing_from_input"]
    assert not competence_gate(violation_summary, _contract())["all_gates_pass"]


def _synthetic_detonation_sequence() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.arange(96, dtype=np.float64) * 1.0e-4
    x = 0.1091 + np.arange(209, dtype=np.float64) * 1.0e-4
    sequence = np.zeros((4, 13, y.size, x.size), dtype=np.float64)
    transverse = 1.0 + 0.2 * np.sin(2.0 * np.pi * 5.0 * y / 0.0096)
    for frame, front in enumerate((0.116, 0.118, 0.120, 0.122)):
        pmax = np.full((y.size, x.size), 1.0e5)
        active = x <= front
        pmax[:, active] = 3.0e6 * transverse[:, None]
        sequence[frame, 12] = pmax
    return sequence, x, y


def test_structure_metrics_recover_moving_front_and_transverse_mode() -> None:
    sequence, x, y = _synthetic_detonation_sequence()
    spectrum = shock_attached_transverse_spectrum(sequence[-1, 12], x=x, y=y)
    assert spectrum.status == "ok"
    assert spectrum.dominant_mode == 5
    assert spectrum.dominant_wavelength_m == pytest.approx(0.00192)

    times = np.arange(sequence.shape[0], dtype=np.float64) * 1.0e-6
    comparison = compare_planardet_structure(
        sequence,
        sequence,
        x=x,
        y=y,
        times=times,
    )
    assert comparison["front_speed_error_m_per_s"] == pytest.approx(0.0)
    assert comparison["mean_cell_size_proxy_error_mm"] == pytest.approx(0.0)
    assert comparison["mean_shock_attached_highpass_fraction_error"] == pytest.approx(
        0.0
    )
    assert comparison["prediction"]["pMax_decrease_count"] == 0
    assert np.nanmax(comparison["pMax_relative_l2_per_frame"]) == pytest.approx(0.0)
