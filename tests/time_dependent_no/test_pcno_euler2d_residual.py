from __future__ import annotations

import csv
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from scripts.time_dependent_no.evaluate_pcno_euler2d_residual import (
    main as evaluate_main,
)
from scripts.time_dependent_no.prepare_pcno_euler2d_shards import main as prepare_main
from scripts.time_dependent_no.train_pcno_euler2d_residual import (
    main as train_main,
    manifest_train_val_test_split,
)
from utility.time_dependent_no.pcno_euler2d import (
    PCNOEuler2DResidual,
    PCNOEuler2DShardStore,
    apply_admissible_primitive_noise,
    balanced_presentations,
    conservative_admissibility,
    conservative_to_primitive_torch,
    directional_highpass_stability,
    fit_normalization,
    graph_neighbor_highpass,
    homogeneous_presentation_batches,
    primitive_to_conservative_torch,
    reference_smooth_region_mask,
    stratified_train_val_split,
    weighted_scaled_mse,
)


def _write_raw_trajectory(group: h5py.Group, trajectory_index: int) -> None:
    num_steps = 4
    nodes = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        dtype=np.float32,
    )
    edges = np.asarray([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int64)
    node_type = np.asarray([3, 1, 2, 1], dtype=np.int64)[:, None]
    mach = np.full((nodes.shape[0], 1), 1.5 + 0.1 * trajectory_index, dtype=np.float32)

    group.create_dataset("pos", data=np.repeat(nodes[None, ...], num_steps, axis=0))
    group.create_dataset("edges", data=np.repeat(edges[None, ...], num_steps, axis=0))
    group.create_dataset(
        "node_type",
        data=np.repeat(node_type[None, ...], num_steps, axis=0),
    )
    group.create_dataset("Mach", data=np.repeat(mach[None, ...], num_steps, axis=0))

    time = np.arange(num_steps, dtype=np.float32)[:, None, None]
    x = nodes[None, :, 0:1]
    y = nodes[None, :, 1:2]
    group.create_dataset(
        "rho", data=1.0 + 0.02 * trajectory_index + 0.01 * time + 0.01 * x
    )
    group.create_dataset("v1", data=0.8 + 0.02 * time + 0.01 * x)
    group.create_dataset("v2", data=0.03 * y + 0.005 * time)
    group.create_dataset("pres", data=1.0 + 0.03 * time + 0.01 * y)


def _prepare_synthetic_shards(tmp_path: Path) -> Path:
    source = tmp_path / "raw.h5"
    with h5py.File(source, "w") as handle:
        _write_raw_trajectory(handle.create_group("0"), 0)
        _write_raw_trajectory(handle.create_group("1"), 1)
    output = tmp_path / "shards"
    prepare_main(
        [
            "--source-h5",
            str(source),
            "--output-dir",
            str(output),
            "--static-check",
            "all",
        ]
    )
    return output


def _prepare_manifest_synthetic_shards(tmp_path: Path) -> Path:
    source = tmp_path / "raw_manifest.h5"
    with h5py.File(source, "w") as handle:
        for trajectory_index in range(3):
            _write_raw_trajectory(
                handle.create_group(str(trajectory_index)), trajectory_index
            )
    output = tmp_path / "manifest_shards"
    prepare_main(
        [
            "--source-h5",
            str(source),
            "--output-dir",
            str(output),
            "--static-check",
            "all",
        ]
    )
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split_names = ("train", "validation", "test")
    splits = {
        split_name: [manifest["trajectories"][index]["key"]]
        for index, split_name in enumerate(split_names)
    }
    for split_name in split_names:
        manifest["trajectories"][split_names.index(split_name)]["split"] = split_name
    manifest["splits"] = splits
    manifest["declared_split_counts"] = {name: 1 for name in split_names}
    manifest["prepared_split_counts"] = {name: 1 for name in split_names}
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output


def _prepare_synthetic_test_shards(tmp_path: Path) -> Path:
    source = tmp_path / "raw_test.h5"
    with h5py.File(source, "w") as handle:
        _write_raw_trajectory(handle.create_group("00"), 2)
        _write_raw_trajectory(handle.create_group("01"), 3)
    output = tmp_path / "test_shards"
    prepare_main(
        [
            "--source-h5",
            str(source),
            "--output-dir",
            str(output),
            "--static-check",
            "all",
        ]
    )
    return output


def _write_synthetic_boundary_audit(tmp_path: Path, shard_dir: Path) -> Path:
    store = PCNOEuler2DShardStore(shard_dir)
    audit_dir = tmp_path / "mesh_audit"
    geometry_dir = audit_dir / "geometry"
    geometry_dir.mkdir(parents=True)
    cases = []
    for raw_case_id, key in enumerate(store.keys, start=1):
        positions = np.array(store.array(key, "nodes"), copy=True)
        edges = np.array(store.array(key, "edges"), copy=True)
        node_type = np.array(store.array(key, "node_type"), copy=True).reshape(-1)
        artifact_name = f"trajectory_{key}_case_{raw_case_id}.npz"
        np.savez_compressed(
            geometry_dir / artifact_name,
            schema=np.asarray("cpg_bump_mesh_contract_v1"),
            trajectory_key=np.asarray(key),
            raw_case_id=np.asarray(raw_case_id, dtype=np.int64),
            mach=np.asarray(store.entry(key)["mach"], dtype=np.float64),
            pos=positions.astype(np.float64),
            primal_edges=edges.astype(np.int64),
            node_type=node_type.astype(np.int64),
            node_normal=np.asarray(
                [[-1.0, 0.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0]],
                dtype=np.float64,
            ),
            node_boundary_normal_coherence=np.ones(4, dtype=np.float64),
            stencil_target_nodes=np.asarray([1, 2, 3], dtype=np.int64),
            stencil_target_rows=np.asarray([0, 1, 2], dtype=np.int64),
            stencil_source_nodes=np.asarray([0, 0, 0], dtype=np.int64),
            stencil_weights=np.ones(3, dtype=np.float64),
        )
        cases.append(
            {
                "trajectory_key": key,
                "raw_case_id": raw_case_id,
                "geometry_artifact": f"geometry/{artifact_name}",
                "config": {"gamma": 1.4, "rho_inf": 1.4, "p_inf": 1.0},
                "boundary": {
                    "boundary_stencil_target_count": 3,
                    "boundary_stencil_entry_count": 3,
                    "boundary_stencil_fallback_target_count": 0,
                    "sharp_wall_corner_count": 0,
                },
            }
        )
    store.close()
    audit = {
        "schema": "cpg_bump_mesh_provenance_audit_v1",
        "status": "valid",
        "boundary_geometry": "verified_all_selected_cases",
        "graph_mesh_identity": "verified_all_selected_cases",
        "legal_boundary_counterfactual": (
            "enabled_causal_nodal_projection_not_exact_dg_boundary_replay"
        ),
        "cases": cases,
    }
    path = audit_dir / "summary.json"
    path.write_text(json.dumps(audit), encoding="utf-8")
    return path


def _write_synthetic_checkpoint(tmp_path: Path, training_dir: Path) -> Path:
    training_store = PCNOEuler2DShardStore(training_dir)
    normalization = fit_normalization(training_store, training_store.keys)
    torch.manual_seed(19)
    model = PCNOEuler2DResidual(
        normalization=normalization,
        k_max=1,
        domain_lengths=(1.0, 1.0),
        layers=(8, 8, 8),
        fc_dim=8,
    )
    with torch.no_grad():
        model.backbone.fc2.weight.normal_(mean=0.0, std=1e-4)
    checkpoint = {
        "checkpoint_schema_version": 4,
        "epoch": 2,
        "best_epoch": 2,
        "model_state": model.state_dict(),
        "model_config": model.model_config(),
        "normalization": normalization.to_dict(),
        "train_keys": training_store.keys,
        "val_keys": training_store.keys,
        "data_manifest_digest": training_store.manifest_digest,
        "step_stride": 1,
        "config_digest": "a" * 64,
        "best_selection": [1.0, -0.1, -0.01],
        "boundary_mode": "model_all_nodes",
        "raw_recurrence": True,
    }
    checkpoint_path = tmp_path / "best.pt"
    torch.save(checkpoint, checkpoint_path)
    training_store.close()
    return checkpoint_path


def test_conservative_round_trip_and_admissible_noise() -> None:
    primitive = torch.tensor(
        [[[1.0, 0.7, -0.1, 0.9], [0.8, 1.2, 0.2, 1.1]]],
        dtype=torch.float32,
    )
    conservative = primitive_to_conservative_torch(primitive)
    recovered = conservative_to_primitive_torch(conservative)
    torch.testing.assert_close(recovered, primitive)

    generator = torch.Generator(device="cpu").manual_seed(7)
    noisy = apply_admissible_primitive_noise(conservative, 0.1, generator=generator)
    assert not torch.equal(noisy, conservative)
    assert bool(conservative_admissibility(noisy)["admissible"].all())
    torch.testing.assert_close(
        apply_admissible_primitive_noise(conservative, 0.0), conservative
    )


def test_proxy_weighted_loss_ignores_masked_nodes() -> None:
    target = torch.zeros((1, 3, 4), dtype=torch.float32)
    prediction = target.clone()
    prediction[:, 2, :] = 100.0
    node_weights = torch.full((1, 3, 1), 1.0 / 3.0)
    node_mask = torch.tensor([[[1.0], [1.0], [0.0]]])
    scale = torch.ones(4)
    assert (
        weighted_scaled_mse(prediction, target, node_weights, node_mask, scale).item()
        == 0.0
    )


def test_graph_highpass_and_directional_stability_exclude_reference_shock() -> None:
    num_nodes = 7
    undirected = torch.stack(
        (torch.arange(num_nodes - 1), torch.arange(1, num_nodes)), dim=-1
    )
    directed = torch.cat((undirected, undirected.flip(-1)), dim=0).unsqueeze(0)
    node_mask = torch.ones((1, num_nodes, 1))
    node_weights = torch.full((1, num_nodes, 1), 1.0 / num_nodes)

    constant = torch.ones((1, num_nodes, 1))
    torch.testing.assert_close(
        graph_neighbor_highpass(constant, directed, node_mask),
        torch.zeros_like(constant),
    )
    alternating = ((-1.0) ** torch.arange(num_nodes)).reshape(1, num_nodes, 1)
    assert (
        float(graph_neighbor_highpass(alternating, directed, node_mask).square().sum())
        > 0.0
    )

    pressure = torch.tensor([1.0, 1.0, 1.0, 4.0, 4.0, 4.0, 4.0])
    primitive = torch.zeros((1, num_nodes, 4))
    primitive[..., 0] = 1.0
    primitive[..., 3] = pressure
    reference = primitive_to_conservative_torch(primitive)
    smooth = reference_smooth_region_mask(
        reference,
        directed,
        node_mask,
        interior_mask=node_mask,
        shock_quantile=0.7,
        dilation_hops=1,
    )
    assert smooth[0, :, 0].tolist() == [True, False, False, False, False, True, True]

    perturbation = torch.zeros_like(reference)
    perturbation[..., 0] = alternating[..., 0]
    reference_prediction = torch.zeros_like(reference)
    perturbed_prediction = (2.0 * perturbation).requires_grad_()
    stability = directional_highpass_stability(
        reference_prediction,
        perturbed_prediction,
        perturbation,
        directed_edges=directed,
        node_weights=node_weights,
        node_mask=node_mask,
        smooth_mask=smooth,
        component_scale=torch.ones(4),
        gain_cap=1.0,
    )
    assert stability["gain"].item() == pytest.approx(2.0)
    assert stability["loss"].item() == pytest.approx(1.0)
    assert 0.0 < stability["smooth_proxy_mass_fraction"].item() < 1.0
    stability["loss"].backward()
    assert perturbed_prediction.grad is not None
    assert bool(torch.isfinite(perturbed_prediction.grad).all())


def test_reference_smooth_mask_threshold_uses_interior_nodes_only() -> None:
    undirected = torch.tensor([[0, 1], [1, 2], [2, 3], [4, 5]])
    directed = torch.cat((undirected, undirected.flip(-1)), dim=0).unsqueeze(0)
    node_mask = torch.ones((1, 6, 1))
    interior = torch.tensor([True, True, True, True, False, False]).reshape(1, 6, 1)
    primitive = torch.zeros((1, 6, 4))
    primitive[..., 0] = 1.0
    primitive[..., 3] = torch.tensor([1.0, 1.0, 2.0, 2.0, 1.0, 101.0])

    smooth = reference_smooth_region_mask(
        primitive_to_conservative_torch(primitive),
        directed,
        node_mask,
        interior_mask=interior,
        shock_quantile=0.8,
        dilation_hops=0,
    )

    assert smooth[0, :, 0].tolist() == [True, False, False, True, False, False]


def test_shard_contract_normalization_sampling_and_identity_model(
    tmp_path: Path,
) -> None:
    output = _prepare_synthetic_shards(tmp_path)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["state_convention"] == "conservative_[rho,rho_v1,rho_v2,E]"
    assert manifest["weight_provenance"] == "reconstructed_vertex_lumped_proxy"
    assert len(manifest["trajectories"]) == 2

    evicting_store = PCNOEuler2DShardStore(output, max_cached_trajectories=1)
    first_states = evicting_store.states(evicting_store.keys[0])
    assert not first_states._mmap.closed
    second_states = evicting_store.states(evicting_store.keys[1])
    assert first_states._mmap.closed
    evicting_store.close()
    assert second_states._mmap.closed

    store = PCNOEuler2DShardStore(output)
    train_keys, val_keys = stratified_train_val_split(store, val_count=1, seed=3)
    assert len(train_keys) == len(val_keys) == 1
    normalization = fit_normalization(store, train_keys, step_stride=2)
    assert np.isfinite(normalization.state_scale).all()
    assert np.all(normalization.state_scale > 0.0)
    assert np.all(normalization.residual_scale > 0.0)
    assert normalization.mach_scale == pytest.approx(0.1)

    pairs = balanced_presentations(
        store,
        store.keys,
        step_stride=2,
        count=5,
        rng=np.random.default_rng(11),
    )
    counts = {key: sum(pair_key == key for pair_key, _ in pairs) for key in store.keys}
    assert max(counts.values()) - min(counts.values()) <= 1
    generated_pairs = balanced_presentations(
        store,
        train_keys,
        step_stride=1,
        count=4,
        rng=np.random.default_rng(8),
        minimum_time_index=1,
    )
    assert all(time_index >= 1 for _, time_index in generated_pairs)
    batches = homogeneous_presentation_batches(
        pairs,
        batch_size=2,
        rng=np.random.default_rng(5),
    )
    assert sum(len(time_indices) for _, time_indices in batches) == len(pairs)
    assert all(len(time_indices) <= 2 for _, time_indices in batches)

    sample = store.tensor_sample(
        store.keys[0], 0, step_stride=2, device=torch.device("cpu")
    )
    expected_target = torch.from_numpy(
        np.array(store.states(store.keys[0])[2], copy=True)
    )
    torch.testing.assert_close(sample["target"][0], expected_target)
    assert sample["node_weights"].sum().item() == pytest.approx(1.0)
    batched_sample = store.tensor_batch(
        store.keys[0],
        [0, 1],
        step_stride=2,
        device=torch.device("cpu"),
    )
    assert batched_sample["current"].shape == (2, 4, 4)
    torch.testing.assert_close(
        batched_sample["target"][1],
        torch.from_numpy(np.array(store.states(store.keys[0])[3], copy=True)),
    )

    model = PCNOEuler2DResidual(
        normalization=normalization,
        k_max=1,
        domain_lengths=(1.0, 1.0),
        layers=(8, 8),
        fc_dim=8,
    )
    prediction = model(
        sample["current"],
        node_mask=sample["node_mask"],
        nodes=sample["nodes"],
        node_weights=sample["node_weights"],
        node_rhos=sample["node_rhos"],
        directed_edges=sample["directed_edges"],
        edge_gradient_weights=sample["edge_gradient_weights"],
        node_type=sample["node_type"],
        mach=sample["mach"],
    )
    torch.testing.assert_close(prediction, sample["current"], rtol=0.0, atol=0.0)
    batched_prediction = model(
        batched_sample["current"],
        node_mask=batched_sample["node_mask"],
        nodes=batched_sample["nodes"],
        node_weights=batched_sample["node_weights"],
        node_rhos=batched_sample["node_rhos"],
        directed_edges=batched_sample["directed_edges"],
        edge_gradient_weights=batched_sample["edge_gradient_weights"],
        node_type=batched_sample["node_type"],
        mach=batched_sample["mach"],
    )
    torch.testing.assert_close(
        batched_prediction,
        batched_sample["current"],
        rtol=0.0,
        atol=0.0,
    )


def test_normalization_retains_manifest_weight_provenance(tmp_path: Path) -> None:
    data_dir = _prepare_synthetic_shards(tmp_path)
    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["weight_provenance"] = "validated_physical_cell_volume_normalized"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    store = PCNOEuler2DShardStore(data_dir)
    normalization = fit_normalization(store, store.keys)
    assert normalization.weight_provenance == (
        "validated_physical_cell_volume_normalized"
    )
    store.close()


def test_manifest_split_is_exact_and_rejects_incomplete_family(tmp_path: Path) -> None:
    data_dir = _prepare_manifest_synthetic_shards(tmp_path)
    store = PCNOEuler2DShardStore(data_dir)
    train_keys, val_keys, test_keys = manifest_train_val_test_split(store)
    assert train_keys == ["0"]
    assert val_keys == ["1"]
    assert test_keys == ["2"]
    store.close()

    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["declared_split_counts"]["train"] = 84
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    incomplete_store = PCNOEuler2DShardStore(data_dir)
    with pytest.raises(ValueError, match="declared_split_counts"):
        manifest_train_val_test_split(incomplete_store)
    incomplete_store.close()


def test_cpu_training_manifest_split_records_sealed_test_partition(
    tmp_path: Path,
) -> None:
    data_dir = _prepare_manifest_synthetic_shards(tmp_path)
    output_dir = tmp_path / "manifest_run"
    train_main(
        [
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(output_dir),
            "--split-mode",
            "manifest",
            "--epochs",
            "1",
            "--presentations-per-epoch",
            "1",
            "--val-presentations",
            "1",
            "--tiny-pairs",
            "1",
            "--tiny-fit-rel-l2",
            "10",
            "--tiny-fit-loss-ratio",
            "10",
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
    split = json.loads((output_dir / "split.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(
        output_dir / "best.pt", map_location="cpu", weights_only=False
    )
    assert summary["split_mode"] == "manifest"
    assert summary["train_trajectories"] == 1
    assert summary["validation_trajectories"] == 1
    assert summary["test_trajectories"] == 1
    assert len(summary["normalization_digest"]) == 64
    assert len(summary["data_contract"]["grouped_split_digest"]) == 64
    assert summary["data_contract"]["line4_handoff"] == {
        "status": "prerequisites_only_baseline_not_frozen",
        "line4_training_truth_authorized": None,
        "line4_front_candidate_available": None,
    }
    assert summary["inference_interventions"] == {
        "future_reference_boundary_values": False,
        "clipping": False,
        "primitive_floors": False,
        "limiter": False,
        "decode_reencode_projection": False,
    }
    assert split["test_keys"] == ["2"]
    assert checkpoint["test_keys"] == ["2"]
    assert checkpoint["split_mode"] == "manifest"
    assert checkpoint["data_contract"] == summary["data_contract"]
    assert len(summary["artifact_sha256"]["best_checkpoint"]) == 64
    assert len(summary["artifact_sha256"]["last_checkpoint"]) == 64


def test_cpu_training_smoke_uses_requested_stride_and_writes_strict_json(
    tmp_path: Path,
) -> None:
    data_dir = _prepare_synthetic_shards(tmp_path)
    output_dir = tmp_path / "run"
    train_main(
        [
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(output_dir),
            "--val-count",
            "1",
            "--step-stride",
            "2",
            "--epochs",
            "2",
            "--presentations-per-epoch",
            "2",
            "--val-presentations",
            "1",
            "--tiny-pairs",
            "1",
            "--tiny-fit-rel-l2",
            "10",
            "--tiny-fit-loss-ratio",
            "10",
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
    summary_text = (output_dir / "summary.json").read_text(encoding="utf-8")
    assert "Infinity" not in summary_text
    assert "NaN" not in summary_text
    summary = json.loads(summary_text)
    split = json.loads((output_dir / "split.json").read_text(encoding="utf-8"))
    assert summary["step_stride"] == 2
    assert summary["boundary_mode"] == "model_all_nodes"
    assert summary["raw_recurrence"] is True
    assert summary["batch_size"] == 4
    assert summary["optimizer"]["scheduler"] == "constant"
    assert len(summary["config_digest"]) == 64
    assert summary["parent_checkpoint"] is None
    assert summary["tiny_fit"]["passed"] is True
    assert len(split["rollout_keys"]) == 1
    assert set(split["rollout_keys"]).issubset(split["val_keys"])
    assert (output_dir / "best.pt").is_file()
    assert (output_dir / "last.pt").is_file()
    assert (output_dir / "tiny_best.pt").is_file()


def test_cpu_generated_state_exposure_keeps_clean_anchor_and_raw_contract(
    tmp_path: Path,
) -> None:
    data_dir = _prepare_synthetic_shards(tmp_path)
    output_dir = tmp_path / "generated"
    train_main(
        [
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(output_dir),
            "--val-count",
            "1",
            "--epochs",
            "1",
            "--presentations-per-epoch",
            "2",
            "--val-presentations",
            "1",
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
            "--generated-state-exposure-weight",
            "0.5",
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
    exposure = summary["generated_state_exposure"]
    assert exposure == {
        "weight": 0.5,
        "depth": 1,
        "first_call_gradient": "detached",
        "clean_one_step_anchor": 0.5,
        "raw_generated_state": True,
    }
    assert summary["last_train"]["clean_loss"] is not None
    assert summary["last_train"]["generated_loss"] is not None
    assert summary["last_train"]["generated_input_relative_l2"] > 0.0
    checkpoint = torch.load(
        output_dir / "best.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert checkpoint["generated_state_exposure"] == exposure


def test_official_residual_evaluator_uses_heldout_targets_and_raw_gain_replay(
    tmp_path: Path,
) -> None:
    training_dir = _prepare_synthetic_shards(tmp_path)
    test_dir = _prepare_synthetic_test_shards(tmp_path)
    checkpoint_path = _write_synthetic_checkpoint(tmp_path, training_dir)

    output_dir = tmp_path / "evaluation"
    evaluate_main(
        [
            "--data-dir",
            str(test_dir),
            "--training-data-dir",
            str(training_dir),
            "--checkpoint",
            str(checkpoint_path),
            "--output-dir",
            str(output_dir),
            "--expected-trajectory-count",
            "2",
            "--num-steps",
            "3",
            "--endpoint-calls",
            "1",
            "2",
            "3",
            "--device",
            "cpu",
        ]
    )

    summary_text = (output_dir / "summary.json").read_text(encoding="utf-8")
    assert "NaN" not in summary_text
    assert "Infinity" not in summary_text
    summary = json.loads(summary_text)
    assert summary["status"] == "complete"
    assert summary["evaluation"]["ground_truth"] == "held-out test shard states"
    assert summary["evaluation"]["future_reference_boundary_values"] is False
    assert summary["evaluation"]["clipping_floors_smoothing_limiter"] is False
    assert summary["preprocessing_contract"]["status"] == (
        "compatible_with_declared_legacy_gaps"
    )
    assert summary["diagnostic_gate"]["learned_method_authorized"] is False
    assert summary["variants"]["pcno_pointwise_tail_gain_0p75"][
        "pointwise_layer_gains"
    ] == [1.0, 0.75]
    assert len(summary["gain_one_replay"]) == 2
    assert max(row["relative_l2"] for row in summary["gain_one_replay"]) <= 1e-5

    trajectory_rows = list(
        csv.DictReader(
            (output_dir / "trajectory_metrics.csv").open(encoding="utf-8", newline="")
        )
    )
    assert len(trajectory_rows) == 6
    assert {row["variant"] for row in trajectory_rows} == {
        "persistence",
        "pcno_baseline",
        "pcno_pointwise_tail_gain_0p75",
    }
    assert (
        len(
            list(
                csv.DictReader(
                    (output_dir / "call_metrics.csv").open(encoding="utf-8", newline="")
                )
            )
        )
        == 18
    )

    test_store = PCNOEuler2DShardStore(test_dir)
    with np.load(output_dir / "trajectories" / "trajectory_00.npz") as artifact:
        np.testing.assert_array_equal(
            artifact["reference_targets_conservative"],
            np.asarray(test_store.states("00")[1:4]),
        )
        assert artifact["pcno_baseline_predictions_conservative"].shape == (3, 4, 4)
        assert artifact["pointwise_gain_predictions_conservative"].shape == (3, 4, 4)
        assert artifact["boundary_mode"].item() == "model_all_nodes"
        np.testing.assert_allclose(
            artifact["pointwise_layer_gains"], np.asarray([1.0, 0.75])
        )
    test_store.close()


@pytest.mark.parametrize(
    (
        "counterfactual",
        "variant",
        "boundary_mode",
        "prediction_field",
        "metadata_field",
        "expected_policy",
    ),
    [
        (
            "causal_nodal_boundary",
            "pcno_causal_nodal_boundary_sensitivity",
            "causal_nodal_physical",
            "causal_boundary_predictions_conservative",
            "causal_boundary_metadata",
            "fixed freestream inflow, current-interior slip wall, and current-interior supersonic outflow",
        ),
        (
            "fixed_freestream_inflow",
            "pcno_fixed_freestream_inflow_sensitivity",
            "fixed_freestream_inflow_model_other_nodes",
            "fixed_inflow_predictions_conservative",
            "fixed_inflow_metadata",
            "fixed freestream inflow; model-predicted wall and outflow nodes",
        ),
    ],
)
def test_official_residual_evaluator_reuses_validated_boundary_contract(
    tmp_path: Path,
    counterfactual: str,
    variant: str,
    boundary_mode: str,
    prediction_field: str,
    metadata_field: str,
    expected_policy: str,
) -> None:
    training_dir = _prepare_synthetic_shards(tmp_path)
    test_dir = _prepare_synthetic_test_shards(tmp_path)
    checkpoint_path = _write_synthetic_checkpoint(tmp_path, training_dir)
    mesh_audit = _write_synthetic_boundary_audit(tmp_path, test_dir)
    output_dir = tmp_path / f"boundary_evaluation_{counterfactual}"

    evaluate_main(
        [
            "--data-dir",
            str(test_dir),
            "--training-data-dir",
            str(training_dir),
            "--checkpoint",
            str(checkpoint_path),
            "--output-dir",
            str(output_dir),
            "--expected-trajectory-count",
            "2",
            "--num-steps",
            "3",
            "--endpoint-calls",
            "1",
            "2",
            "3",
            "--counterfactual",
            counterfactual,
            "--boundary-mesh-audit",
            str(mesh_audit),
            "--device",
            "cpu",
        ]
    )

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["evaluation"]["baseline_boundary_mode"] == "model_all_nodes"
    assert summary["evaluation"]["candidate_boundary_mode"] == boundary_mode
    assert summary["evaluation"]["future_reference_boundary_values"] is False
    candidate = summary["variants"][variant]
    assert candidate["exact_dg_boundary_replay"] is False
    assert candidate["future_reference_boundary_values"] is False
    assert summary["diagnostic_gate"]["matched_boundary_training_authorized"] is False
    assert summary["diagnostic_gate"]["status"] == (
        "uninformative_no_baseline_inflow_failure"
    )

    trajectory_rows = list(
        csv.DictReader(
            (output_dir / "trajectory_metrics.csv").open(encoding="utf-8", newline="")
        )
    )
    assert len(trajectory_rows) == 6
    assert {row["variant"] for row in trajectory_rows} == {
        "persistence",
        "pcno_baseline",
        variant,
    }

    with np.load(output_dir / "trajectories" / "trajectory_00.npz") as artifact:
        assert artifact["candidate_variant"].item() == variant
        assert artifact["candidate_boundary_mode"].item() == boundary_mode
        assert "pointwise_gain_predictions_conservative" not in artifact.files
        predictions = artifact[prediction_field]
        primitive = conservative_to_primitive_torch(
            torch.from_numpy(predictions)
        ).numpy()
        mach = float(artifact["mach"].item())
        np.testing.assert_allclose(
            primitive[:, 0],
            np.repeat(np.asarray([[1.4, mach, 0.0, 1.0]]), 3, axis=0),
            rtol=1e-6,
            atol=1e-6,
        )
        metadata = json.loads(artifact[metadata_field].item())
        assert metadata["trajectory_key"] == "00"
        assert metadata["future_reference_boundary_values"] is False
        assert metadata["applied_scope"] == counterfactual
        assert metadata["policy"] == expected_policy
