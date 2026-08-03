import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

from scripts.time_dependent_no.audit_pcno_bump_training_geometry import (
    _sha256_arrays,
    _undirected_unique_edges,
    audit_bump_training_geometry,
)
from scripts.time_dependent_no.augment_pcno_bump_boundary_fields import (
    ARRAY_NAMES as BUMP_ARRAY_NAMES,
)
from scripts.time_dependent_no.augment_pcno_bump_boundary_fields import (
    augment_open_bump_shards,
)
from scripts.time_dependent_no.augment_pcno_dynamic_boundary_fields import (
    augment_open_dynamic_shards,
)
from scripts.time_dependent_no.prepare_pcno_shock_vortex_shards import (
    _case_ids_for_splits,
)
from scripts.time_dependent_no.train_pcno_euler2d_residual import (
    NO_TYPE_CHANNEL_CONTROL,
    build_model,
)
from utility.time_dependent_no.pcno_boundary_fields import (
    compact_cubic_collar,
    factorized_rectangle_boundary_fields,
    tagged_polyline_boundary_fields,
)
from utility.time_dependent_no.pcno_euler2d import (
    BOUNDARY_FIELD_GEOMETRY_COLLAR,
    BOUNDARY_FIELD_SEMANTIC_COLLAR,
    NODE_TYPE_FEATURE_OMITTED,
    Euler2DNormalization,
    PCNOEuler2DResidual,
    copy_no_boundary_initialization_to_boundary_field_model,
)


def _unit_normalization() -> Euler2DNormalization:
    return Euler2DNormalization(
        state_mean=np.zeros(4),
        state_scale=np.ones(4),
        residual_scale=np.ones(4),
        mach_mean=0.0,
        mach_scale=1.0,
    )


def _cell_centers(nx: int, ny: int) -> np.ndarray:
    x = (np.arange(nx, dtype=np.float64) + 0.5) * 2.0 / nx
    y = (np.arange(ny, dtype=np.float64) + 0.5) / ny
    xx, yy = np.meshgrid(x, y, indexing="xy")
    return np.stack((xx.reshape(-1), yy.reshape(-1)), axis=-1)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_open_shard_source(root: Path) -> str:
    entries = []
    split_keys = {"train": [], "validation": [], "test": []}
    for key, split, folder in (
        ("train_a", "train", "traj_train_a"),
        ("validation_a", "validation", "traj_validation_a"),
        ("sealed_a", "test", "traj_sealed_missing"),
    ):
        entry = {
            "key": key,
            "folder": folder,
            "split": split,
            "num_nodes": 8,
            "array_sha256": {},
        }
        if split != "test":
            shard = root / folder
            shard.mkdir(parents=True)
            np.save(shard / "nodes.npy", _cell_centers(4, 2).astype(np.float32))
            np.save(shard / "states_conservative.npy", np.ones((2, 8, 4), np.float32))
            entry["array_sha256"] = {
                name: _file_sha256(shard / f"{name}.npy")
                for name in ("nodes", "states_conservative")
            }
            (shard / "metadata.json").write_text(
                json.dumps({"manifest_entry": entry}), encoding="utf-8"
            )
        entries.append(entry)
        split_keys[split].append(key)
    manifest = {
        "schema_version": 1,
        "dataset": "shock_vortex_fv_family",
        "boundary_field_contract": None,
        "declared_split_counts": {"train": 1, "validation": 1, "test": 1},
        "prepared_split_counts": {"train": 1, "validation": 1, "test": 1},
        "splits": split_keys,
        "trajectories": entries,
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
    )
    return _file_sha256(manifest_path)


def _write_bump_geometry_source(
    tmp_path: Path,
) -> tuple[Path, Path, str, Path, str]:
    source_h5 = tmp_path / "train.h5"
    nodes = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        dtype=np.float32,
    )
    raw_edges = np.asarray([[0, 1], [1, 0], [1, 2], [2, 3], [3, 0]], np.int64)
    node_type = np.asarray([[3], [1], [2], [1]], dtype=np.int32)
    mach = np.full((4, 1), 3.0, dtype=np.float64)
    with h5py.File(source_h5, "w") as handle:
        group = handle.create_group("train_a")
        for name, value in (
            ("pos", nodes),
            ("edges", raw_edges),
            ("node_type", node_type),
            ("Mach", mach),
        ):
            group.create_dataset(name, data=np.stack((value, value), axis=0))

    source_root = tmp_path / "shards"
    source_root.mkdir()
    train_folder = source_root / "traj_train_a"
    train_folder.mkdir()
    unique_edges = _undirected_unique_edges(raw_edges, num_nodes=4)
    np.save(train_folder / "nodes.npy", nodes)
    np.save(train_folder / "edges.npy", unique_edges.astype(np.int64))
    np.save(train_folder / "node_type.npy", node_type[:, 0].astype(np.int64))
    entries = [
        {
            "key": "train_a",
            "folder": "traj_train_a",
            "num_steps": 2,
            "num_nodes": 4,
            "mach": 3.0,
            "geometry_digest": _sha256_arrays(
                nodes.astype(np.float64),
                unique_edges,
                node_type[:, 0].astype(np.int64),
                np.asarray([3.0]),
            ),
        },
        {
            "key": "validation_missing",
            "folder": "traj_validation_missing",
            "num_steps": 2,
            "num_nodes": 4,
            "mach": 3.0,
            "geometry_digest": "unused_validation_digest",
        },
    ]
    stat = source_h5.stat()
    manifest_path = source_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_size_bytes": stat.st_size,
                "source_mtime_ns": stat.st_mtime_ns,
                "trajectories": entries,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_digest = _file_sha256(manifest_path)
    split_path = tmp_path / "split.json"
    split_path.write_text(
        json.dumps(
            {
                "data_manifest_digest": manifest_digest,
                "split_seed": 19,
                "train_keys": ["train_a"],
                "val_keys": ["validation_missing"],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return (
        source_h5,
        source_root,
        manifest_digest,
        split_path,
        _file_sha256(split_path),
    )


def test_dynamic_shard_split_filter_excludes_sealed_cases_by_construction() -> None:
    manifest = {
        "cases": [
            {"case_id": "train_b", "split": "train"},
            {"case_id": "test_a", "split": "test"},
            {"case_id": "validation_a", "split": "validation"},
            {"case_id": "train_a", "split": "train"},
        ]
    }
    audited_order = ["train_b", "test_a", "validation_a", "train_a"]

    assert _case_ids_for_splits(audited_order, manifest, ("train", "validation")) == [
        "train_b",
        "validation_a",
        "train_a",
    ]
    with pytest.raises(ValueError, match="distinct split names"):
        _case_ids_for_splits(audited_order, manifest, ("train", "train"))
    with pytest.raises(ValueError, match="contain no cases"):
        _case_ids_for_splits(["train_a"], manifest, ("validation",))


def test_dynamic_open_shard_augmentation_never_visits_sealed_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    source_digest = _write_open_shard_source(source)
    output = tmp_path / "derived"
    real_load = np.load
    opened_arrays: list[Path] = []

    def recording_load(path: object, *args: object, **kwargs: object) -> np.ndarray:
        opened_arrays.append(Path(path))
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(
        "scripts.time_dependent_no.augment_pcno_dynamic_boundary_fields.np.load",
        recording_load,
    )
    manifest_path = augment_open_dynamic_shards(
        source_shard_dir=source,
        source_manifest_sha256=source_digest,
        output_dir=output,
        boundary_collar_width=0.25,
        x_min=0.0,
        x_max=2.0,
        y_min=0.0,
        y_max=1.0,
    )

    assert [path.name for path in opened_arrays] == ["nodes.npy", "nodes.npy"]
    assert all("sealed" not in str(path) for path in opened_arrays)
    assert not (source / "traj_sealed_missing").exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["requested_splits"] == ["train", "validation"]
    assert manifest["prepared_split_counts"] == {
        "train": 1,
        "validation": 1,
        "test": 0,
    }
    assert manifest["splits"]["test"] == []
    assert manifest["derived_shard_provenance"] == {
        "schema": "pcno_open_shard_boundary_field_derivation_v1",
        "generator_sha256": _file_sha256(
            Path("scripts/time_dependent_no/augment_pcno_dynamic_boundary_fields.py")
        ),
        "source_manifest_sha256": source_digest,
        "selected_splits": ["train", "validation"],
        "sealed_test_arrays_opened": False,
        "source_array_reuse": "hardlink_preserving_npy_bytes",
        "decoded_source_arrays": ["nodes"],
        "generated_arrays": ["boundary_features"],
    }
    assert manifest["boundary_field_contract"]["domain_bounds"] == {
        "x_min": 0.0,
        "x_max": 2.0,
        "y_min": 0.0,
        "y_max": 1.0,
    }
    for entry in manifest["trajectories"]:
        source_folder = source / entry["folder"]
        target_folder = output / entry["folder"]
        assert os.path.samefile(
            source_folder / "states_conservative.npy",
            target_folder / "states_conservative.npy",
        )
        fields = real_load(target_folder / "boundary_features.npy")
        assert fields.shape == (8, 2)
        assert np.all((0.0 <= fields) & (fields <= 1.0))


def test_dynamic_open_shard_augmentation_fails_before_publication_on_digest_drift(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _write_open_shard_source(source)
    output = tmp_path / "derived"
    with pytest.raises(ValueError, match="source manifest digest mismatch"):
        augment_open_dynamic_shards(
            source_shard_dir=source,
            source_manifest_sha256="0" * 64,
            output_dir=output,
            boundary_collar_width=0.25,
            x_min=0.0,
            x_max=2.0,
            y_min=0.0,
            y_max=1.0,
        )
    assert not output.exists()


def test_bump_geometry_audit_uses_training_geometry_only(tmp_path: Path) -> None:
    source_h5, source_root, manifest_digest, split_path, split_digest = (
        _write_bump_geometry_source(tmp_path)
    )
    output = tmp_path / "geometry_audit.json"
    audit_bump_training_geometry(
        source_h5=source_h5,
        source_shard_dir=source_root,
        source_manifest_sha256=manifest_digest,
        split_json=split_path,
        split_sha256=split_digest,
        output_json=output,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["training_keys"] == ["train_a"]
    assert payload["training_key_count"] == 1
    assert payload["open_validation_key_count"] == 1
    assert payload["validation_geometry_opened"] is False
    assert payload["state_or_target_arrays_opened"] is False
    assert payload["width_rule"]["primary_physical_width"] == pytest.approx(0.05)
    assert payload["training_geometry"][0]["mixed_endpoint_segment_count"] == 4
    assert (
        payload["training_geometry"][0]["semantic_segments"]["wall"]["segment_count"]
        == 4
    )
    assert not (source_root / "traj_validation_missing").exists()


def test_bump_augmentation_freezes_audited_width_before_validation_geometry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_h5, source_root, manifest_digest, split_path, split_digest = (
        _write_bump_geometry_source(tmp_path)
    )
    audit_path = tmp_path / "geometry_audit.json"
    audit_bump_training_geometry(
        source_h5=source_h5,
        source_shard_dir=source_root,
        source_manifest_sha256=manifest_digest,
        split_json=split_path,
        split_sha256=split_digest,
        output_json=audit_path,
    )
    audit_digest = _file_sha256(audit_path)
    manifest = json.loads((source_root / "manifest.json").read_text(encoding="utf-8"))
    nodes = np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], np.float32)
    edges = np.asarray([[0, 1], [1, 2], [2, 3], [0, 3]], np.int64)
    node_type = np.asarray([3, 1, 2, 1], np.int64)
    for entry in manifest["trajectories"]:
        folder = source_root / entry["folder"]
        folder.mkdir(exist_ok=True)
        values = {
            "states_conservative": np.ones((2, 4, 4), np.float32),
            "nodes": nodes,
            "edges": edges,
            "elements": np.asarray([[0, 1, 2, 3]], np.int32),
            "node_type": node_type,
            "node_measures": np.ones((4, 1), np.float32),
            "node_weights": np.full((4, 1), 0.25, np.float32),
            "node_rhos": np.full((4, 1), 0.25, np.float32),
            "directed_edges": np.asarray([[0, 1], [1, 0], [1, 2], [2, 1]], np.int64),
            "edge_gradient_weights": np.ones((4, 2), np.float32),
        }
        for name in BUMP_ARRAY_NAMES:
            np.save(folder / f"{name}.npy", values[name])
        (folder / "metadata.json").write_text(
            json.dumps({"manifest_entry": entry}), encoding="utf-8"
        )

    real_load = np.load
    opened_arrays: list[Path] = []

    def recording_load(path: object, *args: object, **kwargs: object) -> np.ndarray:
        opened_arrays.append(Path(path))
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(
        "scripts.time_dependent_no.augment_pcno_bump_boundary_fields.np.load",
        recording_load,
    )
    output = tmp_path / "derived"
    target_manifest_path = augment_open_bump_shards(
        source_shard_dir=source_root,
        source_manifest_sha256=manifest_digest,
        split_json=split_path,
        split_sha256=split_digest,
        geometry_audit_json=audit_path,
        geometry_audit_sha256=audit_digest,
        output_dir=output,
    )

    assert {path.name for path in opened_arrays} == {
        "nodes.npy",
        "edges.npy",
        "node_type.npy",
    }
    assert all(path.name != "states_conservative.npy" for path in opened_arrays)
    target = json.loads(target_manifest_path.read_text(encoding="utf-8"))
    assert target["splits"] == {
        "train": ["train_a"],
        "validation": ["validation_missing"],
        "test": [],
    }
    assert target["boundary_field_contract"]["physical_width"] == pytest.approx(0.05)
    assert target["boundary_field_contract"]["channel_names"] == [
        "wall",
        "outflow",
        "inflow",
    ]
    for entry in target["trajectories"]:
        assert os.path.samefile(
            source_root / entry["folder"] / "states_conservative.npy",
            output / entry["folder"] / "states_conservative.npy",
        )
        assert set(entry["array_sha256"]) == set(BUMP_ARRAY_NAMES) | {
            "boundary_features"
        }


def test_compact_cubic_collar_has_bounded_c1_endpoints() -> None:
    width = 0.2
    distances = np.asarray([0.0, 0.05, width, 2.0 * width])
    values = compact_cubic_collar(distances, width)
    np.testing.assert_allclose(values, [1.0, 0.84375, 0.0, 0.0])
    epsilon = 1.0e-7
    near_boundary_slope = (
        compact_cubic_collar(np.asarray([epsilon]), width)[0] - values[0]
    ) / epsilon
    near_support_slope = (
        compact_cubic_collar(np.asarray([width - epsilon]), width)[0] / epsilon
    )
    assert abs(near_boundary_slope) < 1.0e-4
    assert abs(near_support_slope) < 1.0e-4


def test_factorized_rectangle_fields_keep_physical_mass_under_refinement() -> None:
    width = 0.2
    masses = []
    affected_fractions = []
    for nx, ny in ((80, 40), (160, 80)):
        fields = factorized_rectangle_boundary_fields(
            _cell_centers(nx, ny),
            x_min=0.0,
            x_max=2.0,
            y_min=0.0,
            y_max=1.0,
            physical_width=width,
        )
        assert fields.contract["channel_names"] == [
            "y_symmetry",
            "x_extrapolation",
        ]
        masses.append(fields.values.mean(axis=0))
        affected_fractions.append((fields.values > 0.0).mean(axis=0))

    # Integral rho(d/ell) through one side of the cubic collar is ell/2.
    np.testing.assert_allclose(masses[-1], [0.2, 0.1], atol=2.0e-4, rtol=0.0)
    np.testing.assert_allclose(masses[0], masses[1], atol=6.0e-4, rtol=0.0)
    np.testing.assert_allclose(
        affected_fractions[0], affected_fractions[1], atol=1.0e-12, rtol=0.0
    )

    corner = factorized_rectangle_boundary_fields(
        np.asarray([[0.0, 0.0]]),
        x_min=0.0,
        x_max=2.0,
        y_min=0.0,
        y_max=1.0,
        physical_width=width,
    )
    np.testing.assert_array_equal(corner.values, np.ones((1, 2), dtype=np.float32))


def test_tagged_polyline_corner_overlap_is_rotation_invariant() -> None:
    points = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.5, 0.0],
        ],
        dtype=np.float64,
    )
    boundary_edges = np.asarray([[0, 1], [1, 2], [2, 3], [3, 0]])
    node_type = np.asarray([3, 1, 2, 1, 0])
    kwargs = {
        "semantic_codes": {"wall": 1, "outflow": 2, "inflow": 3},
        "physical_width": 0.25,
    }
    fields = tagged_polyline_boundary_fields(
        points, boundary_edges, node_type, **kwargs
    )
    # The midpoint of the mixed inflow-wall edge belongs to both finite-mesh
    # semantic subsets; no arbitrary precedence is imposed.
    np.testing.assert_array_equal(fields.values[4], [1.0, 0.0, 1.0])

    angle = 0.71
    rotation = np.asarray(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    rotated = tagged_polyline_boundary_fields(
        points @ rotation.T + np.asarray([1.7, -0.4]),
        boundary_edges,
        node_type,
        **kwargs,
    )
    np.testing.assert_allclose(rotated.values, fields.values, atol=1.0e-7, rtol=0.0)


def test_model_field_layout_and_matched_initial_function() -> None:
    common = {
        "normalization": _unit_normalization(),
        "k_max": 1,
        "domain_lengths": (1.0, 1.0),
        "layers": (8, 8),
        "fc_dim": 8,
        "zero_initialize": False,
        "node_type_feature_mode": NODE_TYPE_FEATURE_OMITTED,
    }
    torch.manual_seed(19)
    no_boundary = PCNOEuler2DResidual(**common)
    semantic = PCNOEuler2DResidual(
        **common,
        boundary_field_mode=BOUNDARY_FIELD_SEMANTIC_COLLAR,
        boundary_field_names=("wall", "outflow", "inflow"),
    )
    audit = copy_no_boundary_initialization_to_boundary_field_model(
        no_boundary, semantic
    )
    assert audit["mathematical_initial_function_match"] is True

    current = torch.tensor([[[1.0, 0.2, 0.1, 2.6], [0.9, 0.3, -0.1, 2.4]]])
    nodes = torch.tensor([[[0.25, 0.25], [0.75, 0.75]]])
    node_rhos = torch.full((1, 2, 1), 0.5)
    node_type = torch.tensor([[0, 3]], dtype=torch.int64)
    mach = torch.tensor([1.1])
    raw_fields = torch.tensor([[[0.2, 0.0, 0.8], [0.0, 0.4, 0.1]]], dtype=torch.float32)
    source_input = no_boundary.normalized_input(
        current,
        nodes=nodes,
        node_rhos=node_rhos,
        node_type=node_type,
        mach=mach,
    )
    semantic_input = semantic.normalized_input(
        current,
        nodes=nodes,
        node_rhos=node_rhos,
        node_type=node_type,
        mach=mach,
        boundary_features=raw_fields,
    )
    assert source_input.shape[-1] == 8
    assert semantic_input.shape[-1] == 11
    torch.testing.assert_close(semantic_input[..., 7:10], raw_fields)
    torch.testing.assert_close(
        no_boundary.backbone.fc0(source_input),
        semantic.backbone.fc0(semantic_input),
        rtol=0.0,
        atol=2.0e-7,
    )
    with torch.no_grad():
        semantic.backbone.fc0.weight[:, 7:10].copy_(
            torch.linspace(-0.3, 0.4, 24).reshape(8, 3)
        )
    zero_field_input = semantic.normalized_input(
        current,
        nodes=nodes,
        node_rhos=node_rhos,
        node_type=node_type,
        mach=mach,
        boundary_features=torch.zeros_like(raw_fields),
    )
    observed_lift_delta = semantic.backbone.fc0(semantic_input) - semantic.backbone.fc0(
        zero_field_input
    )
    torch.testing.assert_close(
        observed_lift_delta,
        semantic.boundary_lift_contribution(raw_fields),
        rtol=0.0,
        atol=2.0e-7,
    )

    geometry = PCNOEuler2DResidual(
        **common,
        boundary_field_mode=BOUNDARY_FIELD_GEOMETRY_COLLAR,
        boundary_field_names=("wall", "outflow", "inflow"),
    )
    geometry_input = geometry.normalized_input(
        current,
        nodes=nodes,
        node_rhos=node_rhos,
        node_type=node_type,
        mach=mach,
        boundary_features=raw_fields,
    )
    assert geometry_input.shape[-1] == 9
    torch.testing.assert_close(
        geometry_input[..., 7:8], raw_fields.amax(dim=-1, keepdim=True)
    )
    with pytest.raises(ValueError, match="requires boundary_features"):
        semantic.normalized_input(
            current,
            nodes=nodes,
            node_rhos=node_rhos,
            node_type=node_type,
            mach=mach,
        )
    invalid_fields = raw_fields.clone()
    invalid_fields[..., 0] = 1.01
    with pytest.raises(ValueError, match=r"lie in \[0,1\]"):
        semantic.normalized_input(
            current,
            nodes=nodes,
            node_rhos=node_rhos,
            node_type=node_type,
            mach=mach,
            boundary_features=invalid_fields,
        )
    with pytest.raises(ValueError, match=r"lie in \[0,1\]"):
        semantic.boundary_lift_contribution(invalid_fields)


def test_trainer_field_model_matches_no_boundary_seed_and_rng() -> None:
    common = {
        "k_max": 1,
        "domain_lengths": (1.0, 1.0),
        "layers": (8, 8),
        "fc_dim": 8,
        "model_node_type_input": "physical",
        "node_type_channel_control": NO_TYPE_CHANNEL_CONTROL,
    }
    torch.manual_seed(211)
    no_boundary = build_model(
        SimpleNamespace(
            **common,
            boundary_field_mode="none",
            boundary_field_names=[],
        ),
        _unit_normalization(),
        zero_initialize=False,
    )
    rng_after_no_boundary = torch.get_rng_state().clone()

    torch.manual_seed(211)
    semantic = build_model(
        SimpleNamespace(
            **common,
            boundary_field_mode=BOUNDARY_FIELD_SEMANTIC_COLLAR,
            boundary_field_names=["wall", "outflow", "inflow"],
        ),
        _unit_normalization(),
        zero_initialize=False,
    )
    assert torch.equal(torch.get_rng_state(), rng_after_no_boundary)
    assert (
        semantic.initialization_control["cpu_rng_state_matches_no_boundary_arm"] is True
    )
    source_state = no_boundary.state_dict()
    target_state = semantic.state_dict()
    for name in source_state:
        if name != "backbone.fc0.weight":
            assert torch.equal(source_state[name], target_state[name]), name
