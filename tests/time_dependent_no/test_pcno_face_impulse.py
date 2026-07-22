from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from scripts.time_dependent_no.train_pcno_euler2d_residual import (
    tiny_fit_snapshot,
    train_canonical_face_epoch,
    main as train_main,
)
from utility.time_dependent_no.pcno_euler2d import (
    Euler2DNormalization,
    PCNOEuler2DShardStore,
    SCHEMA_VERSION,
)
from utility.time_dependent_no.pcno_face_impulse import (
    DIVERGENCE_ACTIVE_TARGET_KIND,
    PCNOEuler2DSharedFaceImpulse,
    TARGET_KIND,
    decode_owner_oriented_face_impulse_torch,
    face_contract_summary,
    impulse_balance_residual,
    load_fixed_fv_face_geometry,
    load_fv_reference_face_trajectory,
    winv_face_loss_and_relative_l2,
)
from utility.time_dependent_no.pcno_fv_geometry import (
    build_pcno_finite_volume_geometry,
)
from utility.time_dependent_no.shock_vortex_family import (
    REFERENCE_ARTIFACT_SCHEMA,
    build_shock_vortex_family_manifest,
    family_case_provenance,
)


def _normalization() -> Euler2DNormalization:
    return Euler2DNormalization(
        state_mean=np.array([1.0, 1.0, 0.0, 3.0]),
        state_scale=np.array([0.2, 0.3, 0.2, 0.5]),
        residual_scale=np.array([0.02, 0.03, 0.02, 0.05]),
        mach_mean=1.1,
        mach_scale=0.1,
        gamma=1.4,
        weight_provenance="validated_physical_cell_volume_normalized",
    )


def _mesh() -> dict[str, np.ndarray | tuple[str, ...]]:
    cells = np.array(
        [[0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]],
        dtype=np.float64,
    )
    volume = np.full(4, 0.25, dtype=np.float64)
    owner = np.array([0, 2, 0, 1, 0, 2, 1, 3, 0, 1, 2, 3], dtype=np.int64)
    neighbor = np.array([1, 3, 2, 3, -1, -1, -1, -1, -1, -1, -1, -1])
    centers = np.array(
        [
            [0.5, 0.25],
            [0.5, 0.75],
            [0.25, 0.5],
            [0.75, 0.5],
            [0.0, 0.25],
            [0.0, 0.75],
            [1.0, 0.25],
            [1.0, 0.75],
            [0.25, 0.0],
            [0.75, 0.0],
            [0.25, 1.0],
            [0.75, 1.0],
        ],
        dtype=np.float64,
    )
    normals = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [-1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, -1.0],
            [0.0, -1.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=np.float64,
    )
    tags = np.array([0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4], dtype=np.int64)
    names = ("interior", "x_min", "x_max", "y_min", "y_max")
    return {
        "cells": cells,
        "volume": volume,
        "face_center": centers,
        "face_measure": np.full(12, 0.5, dtype=np.float64),
        "face_normal": normals,
        "face_owner": owner,
        "face_neighbor": neighbor,
        "face_boundary_tag": tags,
        "boundary_tag_names": names,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_arrays(*values: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in values:
        array = np.ascontiguousarray(value)
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _prepare_face_training_fixture(
    tmp_path: Path,
    *,
    include_reference_impulses: bool = False,
) -> tuple[Path, Path]:
    family = build_shock_vortex_family_manifest()
    family_root = tmp_path / "family"
    family_root.mkdir()
    (family_root / "family_manifest.json").write_text(
        json.dumps(family, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    selected = {
        split: next(case for case in family["cases"] if case["split"] == split)
        for split in ("train", "validation", "test")
    }

    mesh = _mesh()
    geometry = build_pcno_finite_volume_geometry(
        cell_centers=mesh["cells"],
        cell_volume=mesh["volume"],
        face_owner=mesh["face_owner"],
        face_neighbor=mesh["face_neighbor"],
        face_boundary_tag=mesh["face_boundary_tag"],
        boundary_tag_names=mesh["boundary_tag_names"],
    )
    geometry_arrays = {
        "nodes": geometry.nodes,
        "edges": geometry.edges,
        "node_type": geometry.node_type,
        "node_measures": geometry.node_measures,
        "node_weights": geometry.node_weights,
        "node_rhos": geometry.node_rhos,
        "directed_edges": geometry.directed_edges,
        "edge_gradient_weights": geometry.edge_gradient_weights,
        "mesh_cell_to_graph_node": geometry.mesh_cell_to_graph_node,
        "face_to_directed_edge": geometry.face_to_directed_edge,
    }
    geometry_digest = _sha256_arrays(*geometry_arrays.values())
    times = np.arange(4, dtype=np.float64) * 0.01
    train_key = str(selected["train"]["case_id"])
    source_dir = family_root / train_key
    source_dir.mkdir()
    source_path = source_dir / "reference.npz"
    source_payload = {
        "schema": np.asarray(REFERENCE_ARTIFACT_SCHEMA),
        "physical_times": times,
        "cell_centers": mesh["cells"],
        "cell_volume": mesh["volume"],
        "face_centers": mesh["face_center"],
        "face_measure": mesh["face_measure"],
        "face_normal": mesh["face_normal"],
        "face_owner": mesh["face_owner"],
        "face_neighbor": mesh["face_neighbor"],
        "face_boundary_tag": mesh["face_boundary_tag"],
        "boundary_tag_names_json": np.asarray(json.dumps(mesh["boundary_tag_names"])),
        "family_contract_json": np.asarray(
            json.dumps(family_case_provenance(family, train_key))
        ),
    }
    np.savez(source_path, **source_payload)
    source_digest = _sha256_file(source_path)

    shard_root = tmp_path / "shards"
    shard_root.mkdir()
    entries = []
    splits = {}
    centers = np.asarray(mesh["cells"])
    for case_index, split in enumerate(("train", "validation", "test")):
        case = selected[split]
        key = str(case["case_id"])
        folder = shard_root / f"trajectory_{case_index}"
        folder.mkdir()
        states = []
        for time_index in range(times.size):
            rho = 1.0 + 0.02 * case_index + 0.01 * time_index + 0.005 * centers[:, 0]
            velocity_x = 0.5 + 0.01 * time_index + 0.005 * centers[:, 1]
            velocity_y = 0.02 * centers[:, 1] + 0.004 * time_index
            pressure = 1.0 + 0.02 * time_index + 0.01 * centers[:, 0]
            energy = pressure / 0.4 + 0.5 * rho * (velocity_x**2 + velocity_y**2)
            states.append(
                np.stack((rho, rho * velocity_x, rho * velocity_y, energy), axis=-1)
            )
        states_array = np.asarray(states, dtype=np.float32)
        if include_reference_impulses and key == train_key:
            reference_impulses = np.zeros(
                (times.size - 1, len(mesh["face_owner"]), 4), dtype=np.float64
            )
            np.savez(
                source_path,
                **source_payload,
                conservative_states=states_array.astype(np.float64),
                cumulative_accepted_substep_face_impulses=reference_impulses,
            )
            source_digest = _sha256_file(source_path)
        arrays = {
            "states_conservative": states_array,
            "physical_times": times,
            "parameters": np.asarray(
                [
                    case["parameters"]["vortex_epsilon"],
                    case["parameters"]["vortex_y"],
                ],
                dtype=np.float32,
            ),
            **geometry_arrays,
        }
        for name, value in arrays.items():
            if np.issubdtype(np.asarray(value).dtype, np.floating):
                dtype = np.float64 if name == "physical_times" else np.float32
            else:
                dtype = np.int64
            np.save(folder / f"{name}.npy", np.asarray(value, dtype=dtype))
        entries.append(
            {
                "key": key,
                "folder": folder.name,
                "num_steps": int(times.size),
                "num_nodes": int(centers.shape[0]),
                "num_edges": int(geometry.edges.shape[0]),
                "num_directed_edges": int(geometry.directed_edges.shape[0]),
                "num_elements": 0,
                "mach": 1.1,
                "split": split,
                "split_group_id": case["split_group_id"],
                "parameters": case["parameters"],
                "geometry_digest": geometry_digest,
                "source_reference_sha256": (
                    source_digest if key == train_key else "0" * 64
                ),
                "weight_provenance": ("validated_physical_cell_volume_normalized"),
            }
        )
        splits[split] = [key]

    shard_manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset": "shock_vortex_fv_family",
        "source_family_manifest_relative_to_family_root": "family_manifest.json",
        "source_family_id": family["family_id"],
        "source_family_manifest_digest": family["manifest_digest_sha256"],
        "gamma": 1.4,
        "dt": 0.01,
        "state_convention": "conservative_[rho,rho_u,rho_v,E]",
        "coordinate_convention": "FV_row_major_cell_centers_xy",
        "weight_provenance": "validated_physical_cell_volume_normalized",
        "mesh_to_graph_map": "identity_FV_cell_index_to_PCNO_node_index",
        "gradient_rcond": 1.0e-12,
        "declared_split_counts": {split: 1 for split in splits},
        "prepared_split_counts": {split: 1 for split in splits},
        "splits": splits,
        "trajectories": entries,
    }
    (shard_root / "manifest.json").write_text(
        json.dumps(shard_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return family_root, shard_root


def _model(
    *,
    zero_initialize: bool,
    target_kind: str = TARGET_KIND,
) -> PCNOEuler2DSharedFaceImpulse:
    mesh = _mesh()
    return PCNOEuler2DSharedFaceImpulse(
        normalization=_normalization(),
        cell_volume=mesh["volume"],
        face_center=mesh["face_center"],
        face_measure=mesh["face_measure"],
        face_normal=mesh["face_normal"],
        face_owner=mesh["face_owner"],
        face_neighbor=mesh["face_neighbor"],
        face_boundary_tag=mesh["face_boundary_tag"],
        boundary_tag_names=mesh["boundary_tag_names"],
        geometry_digest="synthetic-2x2",
        fixed_delta_t=0.01,
        target_kind=target_kind,
        supervision=(
            "direct_canonical_face_winv_loss"
            if target_kind == DIVERGENCE_ACTIVE_TARGET_KIND
            else "decoded_next_state_loss_only"
        ),
        reference_impulse_supervision=(target_kind == DIVERGENCE_ACTIVE_TARGET_KIND),
        k_max=1,
        domain_lengths=(1.0, 1.0),
        layers=(8, 8),
        fc_dim=8,
        latent_dim=8,
        face_hidden_dim=16,
        zero_initialize=zero_initialize,
    )


def _sample() -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    mesh = _mesh()
    geometry = build_pcno_finite_volume_geometry(
        cell_centers=mesh["cells"],
        cell_volume=mesh["volume"],
        face_owner=mesh["face_owner"],
        face_neighbor=mesh["face_neighbor"],
        face_boundary_tag=mesh["face_boundary_tag"],
        boundary_tag_names=mesh["boundary_tag_names"],
    )
    current = torch.tensor(
        [
            [
                [1.0, 1.0, 0.0, 3.0],
                [1.02, 1.01, 0.01, 3.02],
                [0.99, 0.98, -0.01, 2.98],
                [1.01, 1.00, 0.02, 3.01],
            ]
        ],
        dtype=torch.float32,
    )
    return current, {
        "node_mask": torch.ones(1, 4, 1),
        "nodes": torch.as_tensor(geometry.nodes, dtype=torch.float32).unsqueeze(0),
        "node_weights": torch.as_tensor(
            geometry.node_weights, dtype=torch.float32
        ).unsqueeze(0),
        "node_rhos": torch.as_tensor(geometry.node_rhos, dtype=torch.float32).unsqueeze(
            0
        ),
        "directed_edges": torch.as_tensor(
            geometry.directed_edges, dtype=torch.int64
        ).unsqueeze(0),
        "edge_gradient_weights": torch.as_tensor(
            geometry.edge_gradient_weights, dtype=torch.float32
        ).unsqueeze(0),
        "node_type": torch.as_tensor(geometry.node_type, dtype=torch.int64).unsqueeze(
            0
        ),
        "mach": torch.tensor([1.1], dtype=torch.float32),
    }


def test_face_decoder_closes_exactly_against_boundary_exchange() -> None:
    mesh = _mesh()
    generator = torch.Generator().manual_seed(7)
    impulse = 0.01 * torch.randn(2, 12, 4, generator=generator)
    update = decode_owner_oriented_face_impulse_torch(
        impulse,
        cell_volume=torch.as_tensor(mesh["volume"]),
        face_owner=torch.as_tensor(mesh["face_owner"]),
        face_neighbor=torch.as_tensor(mesh["face_neighbor"]),
    )
    residual = impulse_balance_residual(
        update,
        impulse,
        cell_volume=torch.as_tensor(mesh["volume"]),
        face_neighbor=torch.as_tensor(mesh["face_neighbor"]),
    )
    assert torch.max(torch.abs(residual)).item() < 2.0e-8


def test_zero_initialized_face_model_is_exact_persistence() -> None:
    model = _model(zero_initialize=True)
    current, sample = _sample()
    prediction, impulse = model.forward_with_face_impulse(current, **sample)
    torch.testing.assert_close(prediction, current, rtol=0.0, atol=0.0)
    torch.testing.assert_close(impulse, torch.zeros_like(impulse), rtol=0.0, atol=0.0)
    config = model.model_config()
    assert config["target_kind"] == TARGET_KIND
    assert config["geometry_digest"] == "synthetic-2x2"
    contract = face_contract_summary(model)
    assert contract["reference_impulse_supervision"] is False
    assert contract["decoded_state_conservative_up_to_predicted_boundary_exchange"]


def test_face_model_accepts_bfloat16_autocast_outputs() -> None:
    model = _model(zero_initialize=False)
    current, sample = _sample()
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        prediction, impulse = model.forward_with_face_impulse(current, **sample)
    assert prediction.dtype == torch.float32
    assert impulse.dtype == torch.float32
    assert bool(torch.isfinite(prediction).all())
    assert bool(torch.isfinite(impulse).all())


def test_face_model_enforces_wall_exchange_and_backpropagates_state_loss() -> None:
    model = _model(zero_initialize=True)
    current, sample = _sample()
    target = current.clone()
    target[:, :, 0] += torch.tensor([[0.01, -0.01, 0.005, -0.005]])
    prediction, impulse = model.forward_with_face_impulse(current, **sample)
    loss = torch.mean((prediction - target) ** 2)
    loss.backward()
    assert model.interior_head[-1].weight.grad is not None
    assert model.boundary_head[-1].weight.grad is not None
    assert torch.linalg.vector_norm(model.interior_head[-1].weight.grad).item() > 0.0
    assert torch.linalg.vector_norm(model.boundary_head[-1].weight.grad).item() > 0.0
    wall = torch.tensor([8, 9, 10, 11])
    nonnormal = impulse[:, wall][:, :, [0, 1, 3]]
    torch.testing.assert_close(
        nonnormal, torch.zeros_like(nonnormal), rtol=0.0, atol=0.0
    )


def test_canonical_face_loss_is_unit_at_zero_and_backpropagates() -> None:
    generator = torch.Generator().manual_seed(13)
    target = 0.1 + torch.rand(2, 12, 4, generator=generator)
    prediction = torch.zeros_like(target, requires_grad=True)
    weight = torch.linspace(0.5, 1.5, 12)
    interior = torch.tensor([True] * 4 + [False] * 8)

    interior_loss, interior_relative = winv_face_loss_and_relative_l2(
        prediction,
        target,
        face_weight=weight,
        face_mask=interior,
    )
    boundary_loss, boundary_relative = winv_face_loss_and_relative_l2(
        prediction,
        target,
        face_weight=weight,
        face_mask=~interior,
    )
    loss = 0.5 * (interior_loss + boundary_loss)
    loss.backward()

    torch.testing.assert_close(interior_loss, torch.ones_like(interior_loss))
    torch.testing.assert_close(boundary_loss, torch.ones_like(boundary_loss))
    torch.testing.assert_close(interior_relative, torch.ones_like(interior_relative))
    torch.testing.assert_close(boundary_relative, torch.ones_like(boundary_relative))
    assert prediction.grad is not None
    assert torch.linalg.vector_norm(prediction.grad).item() > 0.0


def test_direct_face_supervision_changes_contract_not_parameterization() -> None:
    state_model = _model(zero_initialize=True)
    direct_model = _model(
        zero_initialize=True,
        target_kind=DIVERGENCE_ACTIVE_TARGET_KIND,
    )

    assert sum(p.numel() for p in state_model.parameters()) == sum(
        p.numel() for p in direct_model.parameters()
    )
    config = direct_model.model_config()
    contract = face_contract_summary(direct_model)
    assert config["target_kind"] == DIVERGENCE_ACTIVE_TARGET_KIND
    assert config["supervision"] == "direct_canonical_face_winv_loss"
    assert config["reference_impulse_supervision"] is True
    assert contract["target_kind"] == DIVERGENCE_ACTIVE_TARGET_KIND
    assert contract["reference_impulse_supervision"] is True


def test_reference_face_loader_is_explicit_and_digest_bound(tmp_path: Path) -> None:
    family_root, shard_root = _prepare_face_training_fixture(
        tmp_path,
        include_reference_impulses=True,
    )
    store = PCNOEuler2DShardStore(shard_root)
    train_key = str(store.manifest["splits"]["train"][0])
    geometry = load_fixed_fv_face_geometry(
        family_root,
        store,
        source_key=train_key,
    )
    reference = load_fv_reference_face_trajectory(
        family_root,
        store,
        geometry,
        key=train_key,
    )

    assert reference.case_id == train_key
    assert reference.split == "train"
    assert reference.conservative_states.shape == (4, 4, 4)
    assert reference.cumulative_face_impulses.shape == (3, 12, 4)
    assert (
        reference.source_reference_sha256
        == store.entry(train_key)["source_reference_sha256"]
    )


def test_canonical_face_epoch_optimizes_face_loss_only(tmp_path: Path) -> None:
    _, shard_root = _prepare_face_training_fixture(tmp_path)
    store = PCNOEuler2DShardStore(shard_root)
    key = str(store.manifest["splits"]["train"][0])
    model = _model(
        zero_initialize=True,
        target_kind=DIVERGENCE_ACTIVE_TARGET_KIND,
    )
    face_target = torch.full((12, 4), 1.0e-3, dtype=torch.float32)
    face_target[8:12, 0] = 0.0
    face_target[8:12, 1] = 0.0
    face_target[8:12, 3] = 0.0
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-5)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    scaler = torch.amp.GradScaler("cpu", enabled=False)

    metrics = train_canonical_face_epoch(
        model,
        store,
        [(key, 0)],
        canonical_targets={(key, 0): face_target},
        face_weight=np.full(12, 0.25, dtype=np.float64),
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        step_stride=1,
        batch_size=1,
        batch_rng=np.random.default_rng(19),
        device=torch.device("cpu"),
        amp="none",
        gradient_clip=1.0,
    )

    assert np.isclose(metrics["loss"], 1.0)
    assert np.isclose(metrics["interior_face_relative_l2"], 1.0)
    assert np.isclose(metrics["boundary_face_relative_l2"], 1.0)
    assert metrics["gradients_finite"] == 1.0
    assert metrics["wall_forbidden_exchange_absolute_max"] == 0.0
    assert metrics["generated_loss"] is None


def test_canonical_tiny_fit_requires_all_face_gates_at_one_snapshot() -> None:
    initial = {"loss": 1.0}
    current = {
        "loss": 5.0e-3,
        "relative_l2": 5.0e-4,
        "admissible_fraction": 1.0,
        "face_relative_l2": 0.05,
        "interior_face_relative_l2": 0.06,
        "boundary_face_relative_l2": 0.07,
        "wall_forbidden_exchange_absolute_max": 0.0,
    }
    passed = tiny_fit_snapshot(
        initial,
        current,
        relative_l2_threshold=1.0e-3,
        loss_ratio_threshold=1.0e-2,
        face_relative_l2_threshold=0.1,
    )
    assert passed["passed"] is True

    failed_metrics = dict(current)
    failed_metrics["boundary_face_relative_l2"] = 0.1001
    failed = tiny_fit_snapshot(
        initial,
        failed_metrics,
        relative_l2_threshold=1.0e-3,
        loss_ratio_threshold=1.0e-2,
        face_relative_l2_threshold=0.1,
    )
    assert failed["passed"] is False


def test_interior_face_head_is_orientation_antisymmetric() -> None:
    first = _model(zero_initialize=False)
    mesh = _mesh()
    swapped_owner = np.array(mesh["face_owner"], copy=True)
    swapped_neighbor = np.array(mesh["face_neighbor"], copy=True)
    swapped_normal = np.array(mesh["face_normal"], copy=True)
    interior = swapped_neighbor >= 0
    swapped_owner[interior], swapped_neighbor[interior] = (
        swapped_neighbor[interior].copy(),
        swapped_owner[interior].copy(),
    )
    swapped_normal[interior] *= -1.0
    second = PCNOEuler2DSharedFaceImpulse(
        normalization=_normalization(),
        cell_volume=mesh["volume"],
        face_center=mesh["face_center"],
        face_measure=mesh["face_measure"],
        face_normal=swapped_normal,
        face_owner=swapped_owner,
        face_neighbor=swapped_neighbor,
        face_boundary_tag=mesh["face_boundary_tag"],
        boundary_tag_names=mesh["boundary_tag_names"],
        geometry_digest="synthetic-2x2-swapped",
        fixed_delta_t=0.01,
        k_max=1,
        domain_lengths=(1.0, 1.0),
        layers=(8, 8),
        fc_dim=8,
        latent_dim=8,
        face_hidden_dim=16,
        zero_initialize=False,
    )
    first_parameters = dict(first.named_parameters())
    with torch.no_grad():
        for name, parameter in second.named_parameters():
            parameter.copy_(first_parameters[name])
    generator = torch.Generator().manual_seed(17)
    latent = torch.randn(2, 4, 8, generator=generator)
    state = torch.randn(2, 4, 4, generator=generator)
    first_output = first._interior_normalized_impulse(latent, state)
    second_output = second._interior_normalized_impulse(latent, state)
    torch.testing.assert_close(second_output, -first_output, rtol=1.0e-6, atol=1.0e-6)


def test_shared_face_target_cpu_training_records_closed_contract(
    tmp_path: Path,
) -> None:
    family_root, shard_root = _prepare_face_training_fixture(tmp_path)
    output_dir = tmp_path / "run"
    train_main(
        [
            "--data-dir",
            str(shard_root),
            "--family-root",
            str(family_root),
            "--output-dir",
            str(output_dir),
            "--target-kind",
            TARGET_KIND,
            "--split-mode",
            "manifest",
            "--epochs",
            "1",
            "--presentations-per-epoch",
            "1",
            "--val-presentations",
            "1",
            "--batch-size",
            "1",
            "--tiny-pairs",
            "1",
            "--tiny-fit-rel-l2",
            "100",
            "--tiny-fit-loss-ratio",
            "100",
            "--stop-on-tiny-fit",
            "--k-max",
            "1",
            "--domain-lengths",
            "1",
            "1",
            "--layers",
            "8",
            "8",
            "--fc-dim",
            "8",
            "--face-latent-dim",
            "8",
            "--face-hidden-dim",
            "16",
            "--rollout-every",
            "1",
            "--rollout-val-count",
            "1",
            "--rollout-steps",
            "1",
            "--device",
            "cpu",
            "--amp",
            "none",
        ]
    )

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(
        output_dir / "best.pt", map_location="cpu", weights_only=False
    )
    contract = summary["target_contract"]
    assert summary["target_kind"] == TARGET_KIND
    assert summary["tiny_fit"]["passed"] is True
    assert contract["supervision"] == "decoded_next_state_loss_only"
    assert contract["reference_face_impulse_arrays_loaded"] is False
    assert contract["reference_impulse_supervision"] is False
    assert contract["model_face_contract"][
        "decoded_state_conservative_up_to_predicted_boundary_exchange"
    ]
    assert contract["cycle_component_identifiability"].startswith("not_identified")
    assert len(contract["physical_geometry_digest"]) == 64
    assert len(contract["graph_geometry_digest"]) == 64
    assert checkpoint["target_contract"] == contract
    assert checkpoint["model_config"]["target_kind"] == TARGET_KIND
    with np.load(
        family_root / contract["source_geometry_key"] / "reference.npz",
        allow_pickle=False,
    ) as source:
        assert "cumulative_accepted_substep_face_impulses" not in source.files
