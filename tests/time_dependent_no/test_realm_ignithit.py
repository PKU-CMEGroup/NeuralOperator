from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest

from scripts.time_dependent_no.acquire_realm_ignithit import (
    acquire_entry,
    acquire_open_manifest,
    object_url,
)
from scripts.time_dependent_no.acquire_realm_ignithit import (
    main as acquire_main,
)
from scripts.time_dependent_no.audit_realm_benchmark import main as audit_main
from utility.time_dependent_no.realm_benchmark import IGNITHIT_FIELDS, ManifestEntry
from utility.time_dependent_no.realm_ignithit import (
    BOX_COX_LAMBDA,
    COORDINATE_UNIFORMITY_RTOL,
    METADATA_KEYS,
    PRIMARY_BOX_COX_EPSILON,
    TEST_GROUPS,
    TRAIN_GROUPS,
    TRAJECTORY_DTYPE,
    TRAJECTORY_SHAPE,
    VAL_GROUPS,
    StreamingChannelMoments,
    git_blob_oid,
    load_ignithit_metadata,
    load_ignithit_trajectory,
    trajectory_diagnostics,
    trajectory_relative_path,
    validate_local_open_tree,
    verify_manifest_file,
)


@pytest.fixture(scope="module")
def valid_state() -> np.ndarray:
    state = np.zeros(TRAJECTORY_SHAPE, dtype=TRAJECTORY_DTYPE)
    for channel in range(8):
        state[:, channel] = np.float32(10.0 ** (-channel - 2))
    state[:, 8] = 900.0
    state[:, 9] = 1.2
    state[:, 10] = 4.0
    state[:, 11] = -0.5
    return state


def _write_metadata(path: Path, **overrides: np.ndarray) -> None:
    axis = np.arange(128, 0, -1, dtype=np.float32) * np.float32(0.25)
    coords = np.empty((2, 128, 128), dtype=np.float32)
    coords[0] = axis[:, None]
    coords[1] = axis[None, :]
    payload: dict[str, np.ndarray] = {
        "coords": coords,
        "num_chemical": np.asarray(8, dtype=np.int64),
        "num_density": np.asarray(1, dtype=np.int64),
        "num_pressure": np.asarray(0, dtype=np.int64),
        "num_temperature": np.asarray(1, dtype=np.int64),
        "num_velocity": np.asarray(2, dtype=np.int64),
        "spatial_size": np.asarray([128, 128], dtype=np.int64),
        "test_groups": np.asarray(TEST_GROUPS),
        "times": np.asarray([f"{index * 1.0e-5:.8g}" for index in range(1, 31)]),
        "train_groups": np.asarray(TRAIN_GROUPS),
        "val_groups": np.asarray(VAL_GROUPS),
        "variables": np.asarray([f"{field}.npy" for field in IGNITHIT_FIELDS]),
    }
    payload.update(overrides)
    np.savez(path, **payload)


def _entry_for(path: Path, relative: str, *, lfs: bool) -> ManifestEntry:
    oid = hashlib.sha256(path.read_bytes()).hexdigest() if lfs else git_blob_oid(path)
    return ManifestEntry(path=relative, size=path.stat().st_size, oid=oid, lfs=lfs)


def test_pinned_metadata_closes_axes_splits_fields_and_cadence(tmp_path: Path) -> None:
    path = tmp_path / "data.npz"
    _write_metadata(path)

    metadata = load_ignithit_metadata(path)

    assert metadata.train_groups == TRAIN_GROUPS
    assert metadata.val_groups == VAL_GROUPS
    assert metadata.test_groups == TEST_GROUPS
    assert metadata.dx == pytest.approx(0.25)
    assert metadata.dy == pytest.approx(0.25)
    assert metadata.times[0] == pytest.approx(1.0e-5)
    assert metadata.times[-1] == pytest.approx(3.0e-4)
    summary = metadata.summary()
    assert summary["coordinate_order"] == ["y", "x"]
    assert summary["x_descending"] is True
    assert summary["y_descending"] is True
    assert summary["coordinate_units"] is None


def test_metadata_rejects_key_and_open_split_drift(tmp_path: Path) -> None:
    path = tmp_path / "data.npz"
    _write_metadata(path, extra=np.asarray(1))
    with pytest.raises(ValueError, match="keys"):
        load_ignithit_metadata(path)
    _write_metadata(path, train_groups=np.asarray(tuple(reversed(TRAIN_GROUPS))))
    with pytest.raises(ValueError, match="membership or order"):
        load_ignithit_metadata(path)


def test_metadata_uniformity_tolerance_covers_float32_jitter_but_not_grid_drift(
    tmp_path: Path,
) -> None:
    assert COORDINATE_UNIFORMITY_RTOL == 3.0e-4
    path = tmp_path / "data.npz"
    axis = np.arange(128, 0, -1, dtype=np.float32) * np.float32(0.25)
    coords = np.empty((2, 128, 128), dtype=np.float32)
    coords[0] = axis[:, None]
    coords[1] = axis[None, :]
    coords[0, 64:, :] -= np.float32(0.01)
    _write_metadata(path, coords=coords)
    with pytest.raises(ValueError, match="y spacing is not uniform"):
        load_ignithit_metadata(path)


def test_metadata_key_set_is_explicit() -> None:
    assert METADATA_KEYS == {
        "coords",
        "num_chemical",
        "num_density",
        "num_pressure",
        "num_temperature",
        "num_velocity",
        "spatial_size",
        "test_groups",
        "times",
        "train_groups",
        "val_groups",
        "variables",
    }


def test_trajectory_loader_requires_exact_key_shape_dtype_and_finiteness(
    tmp_path: Path, valid_state: np.ndarray
) -> None:
    path = tmp_path / "trajectory.npz"
    np.savez_compressed(path, data=valid_state)
    loaded = load_ignithit_trajectory(path)
    assert loaded.shape == TRAJECTORY_SHAPE
    assert loaded.dtype == TRAJECTORY_DTYPE

    np.savez_compressed(path, data=valid_state, extra=np.asarray(1))
    with pytest.raises(ValueError, match="only 'data'"):
        load_ignithit_trajectory(path)
    np.savez_compressed(path, data=valid_state.astype(np.float64))
    with pytest.raises(ValueError, match="float32"):
        load_ignithit_trajectory(path)
    invalid = valid_state.copy()
    invalid[0, 0, 0, 0] = np.nan
    np.savez_compressed(path, data=invalid)
    with pytest.raises(ValueError, match="native-nonfinite"):
        load_ignithit_trajectory(path)


def test_released_admissibility_stays_distinct_from_finiteness(
    valid_state: np.ndarray,
) -> None:
    valid = trajectory_diagnostics(valid_state)
    assert valid["finite"] is True
    assert valid["admissible_released_state"] is True

    invalid = valid_state.copy()
    invalid[0, 0, 0, 0] = -1.0
    invalid[0, 8, 0, 1] = 0.0
    invalid[0, 9, 0, 2] = 0.0
    result = trajectory_diagnostics(invalid)
    assert result["finite"] is True
    assert result["admissible_released_state"] is False
    assert result["negative_species_count_by_channel"][0] == 1
    assert result["nonpositive_temperature_count"] == 1
    assert result["nonpositive_density_count"] == 1


def test_streaming_train_moments_match_direct_float64_calculation(
    valid_state: np.ndarray,
) -> None:
    state = valid_state.copy()
    state[0, 10, 0, 0] = 8.0
    moments = StreamingChannelMoments(box_cox_epsilon=PRIMARY_BOX_COX_EPSILON)
    moments.update(state)
    moments.update(state)
    result = moments.finalize()

    velocity = np.concatenate((state[:, 10].reshape(-1).astype(np.float64),) * 2)
    species = np.concatenate((state[:, 0].reshape(-1).astype(np.float64),) * 2)
    transformed_species = (
        np.maximum(species, PRIMARY_BOX_COX_EPSILON) ** BOX_COX_LAMBDA - 1.0
    ) / BOX_COX_LAMBDA
    assert result["sample_count_per_channel"] == 2 * 30 * 128 * 128
    assert result["mean"][10] == pytest.approx(velocity.mean(), rel=1.0e-14)
    assert result["std"][10] == pytest.approx(velocity.std(ddof=1), rel=1.0e-12)
    assert result["mean"][0] == pytest.approx(transformed_species.mean(), rel=1.0e-14)
    assert result["scale"][8] == 1.0


def test_group_to_path_mapping_is_exact_and_test_is_unavailable() -> None:
    assert trajectory_relative_path("train", TRAIN_GROUPS[0]) == (
        f"data/train/{TRAIN_GROUPS[0]}.npz"
    )
    assert trajectory_relative_path("val", VAL_GROUPS[0]) == (
        f"data/val/{VAL_GROUPS[0]}.npz"
    )
    with pytest.raises(ValueError, match="train or val"):
        trajectory_relative_path("test", TEST_GROUPS[0])
    with pytest.raises(ValueError, match="unregistered"):
        trajectory_relative_path("train", VAL_GROUPS[0])


def test_lfs_and_git_blob_files_are_verified_exactly(tmp_path: Path) -> None:
    root = tmp_path / "data_root"
    root.mkdir()
    lfs_path = root / "large.bin"
    lfs_path.write_bytes(b"registered-lfs")
    git_path = root / "metadata.txt"
    git_path.write_bytes(b"registered-git")
    lfs_entry = _entry_for(lfs_path, "large.bin", lfs=True)
    git_entry = _entry_for(git_path, "metadata.txt", lfs=False)

    assert verify_manifest_file(root, lfs_entry)["oid"] == lfs_entry.oid
    assert verify_manifest_file(root, git_entry)["oid"] == git_entry.oid
    git_path.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="size mismatch|object mismatch"):
        verify_manifest_file(root, git_entry)


def test_local_tree_is_exact_and_rejects_any_sealed_test_path(tmp_path: Path) -> None:
    root = tmp_path / "data_root"
    train = root / "data" / "train" / "a.bin"
    train.parent.mkdir(parents=True)
    train.write_bytes(b"train")
    entry = _entry_for(train, "data/train/a.bin", lfs=True)
    assert len(validate_local_open_tree(root, (entry,))) == 1

    extra = root / "unexpected.txt"
    extra.write_text("extra", encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected"):
        validate_local_open_tree(root, (entry,))
    extra.unlink()
    sealed = root / "data" / "test" / "never.bin"
    sealed.parent.mkdir()
    sealed.write_bytes(b"sealed")
    with pytest.raises(ValueError, match="sealed test path"):
        validate_local_open_tree(root, (entry,))


def test_acquisition_is_atomic_hash_checked_and_idempotent(tmp_path: Path) -> None:
    content = b"pinned object bytes"
    entry = ManifestEntry(
        path="data/train/a.bin",
        size=len(content),
        oid=hashlib.sha256(content).hexdigest(),
        lfs=True,
    )
    opened: list[str] = []

    def open_url(url: str) -> BytesIO:
        opened.append(url)
        return BytesIO(content)

    root = tmp_path / "data_root"
    summary = acquire_open_manifest(root, (entry,), open_url=open_url)
    assert summary["downloaded_bytes_this_invocation"] == len(content)
    assert summary["reused_verified_file_count"] == 0
    assert (root / "data" / "train" / "a.bin").read_bytes() == content
    assert len(opened) == 1

    repeated = acquire_open_manifest(root, (entry,), open_url=open_url)
    assert repeated["downloaded_bytes_this_invocation"] == 0
    assert repeated["reused_verified_file_count"] == 1
    assert len(opened) == 1


def test_failed_acquisition_leaves_no_target_or_part_file(tmp_path: Path) -> None:
    content = b"wrong bytes"
    entry = ManifestEntry(
        path="data/val/a.bin",
        size=len(content),
        oid="0" * 64,
        lfs=True,
    )
    root = tmp_path / "data_root"
    with pytest.raises(ValueError, match="object mismatch"):
        acquire_entry(root, entry, open_url=lambda _: BytesIO(content))
    assert not (root / "data" / "val" / "a.bin").exists()
    assert not list(root.rglob("*.part.*"))


def test_existing_test_path_stops_before_any_network_open(tmp_path: Path) -> None:
    content = b"registered"
    entry = ManifestEntry(
        path="data/train/a.bin",
        size=len(content),
        oid=hashlib.sha256(content).hexdigest(),
        lfs=True,
    )
    root = tmp_path / "data_root"
    sealed = root / "data" / "test" / "sealed.bin"
    sealed.parent.mkdir(parents=True)
    sealed.write_bytes(b"never open")
    network_calls = 0

    def open_url(_: str) -> BytesIO:
        nonlocal network_calls
        network_calls += 1
        return BytesIO(content)

    with pytest.raises(ValueError, match="before acquisition"):
        acquire_open_manifest(root, (entry,), open_url=open_url)
    assert network_calls == 0


def test_object_url_binds_official_repository_revision_and_path() -> None:
    entry = ManifestEntry(
        path="data/train/phi=c_5_1_c.npz",
        size=1,
        oid="a" * 64,
        lfs=True,
    )
    url = object_url(entry)
    assert "realm-bench/realm-bench-IgnitHIT" in url
    assert "a0736b4d8c6c58a2688127e32addc30085e824c3" in url
    assert url.endswith("data/train/phi=c_5_1_c.npz")


def test_p1b_clis_expose_only_explicit_paths(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as acquire_exit:
        acquire_main(["--help"])
    assert acquire_exit.value.code == 0
    acquire_help = capsys.readouterr().out
    assert "--manifest" in acquire_help
    assert "--data-root" in acquire_help
    assert "--summary" in acquire_help

    with pytest.raises(SystemExit) as audit_exit:
        audit_main(["--help"])
    assert audit_exit.value.code == 0
    audit_help = capsys.readouterr().out
    assert "--data-root" in audit_help
    assert "--report-dir" in audit_help


@pytest.mark.parametrize("output_location", ("data", "reports"))
def test_replay_output_cannot_contaminate_closed_roots(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    output_location: str,
) -> None:
    data_root = tmp_path / "data_root"
    report_dir = tmp_path / "reports"
    output_root = data_root if output_location == "data" else report_dir
    with pytest.raises(SystemExit) as exit_info:
        audit_main(
            [
                "--manifest",
                str(tmp_path / "manifest.json"),
                "--data-root",
                str(data_root),
                "--report-dir",
                str(report_dir),
                "--output",
                str(output_root / "summary.json"),
            ]
        )
    assert exit_info.value.code == 2
    assert "output must be outside" in capsys.readouterr().err


def test_metadata_audit_output_cannot_overwrite_manifest(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest = tmp_path / "manifest.json"
    with pytest.raises(SystemExit) as exit_info:
        audit_main(["--manifest", str(manifest), "--output", str(manifest)])
    assert exit_info.value.code == 2
    assert "must not overwrite" in capsys.readouterr().err
