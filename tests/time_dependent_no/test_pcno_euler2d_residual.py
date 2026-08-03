from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

from scripts.time_dependent_no.evaluate_pcno_euler2d_boundary_protocol import (
    main as evaluate_boundary_protocol_main,
)
from scripts.time_dependent_no.evaluate_pcno_euler2d_residual import (
    main as evaluate_main,
)
from scripts.time_dependent_no.prepare_pcno_euler2d_shards import main as prepare_main
from scripts.time_dependent_no.train_pcno_euler2d_residual import (
    NO_TYPE_CHANNEL_CONTROL,
    RAW_BOUNDARY_REFERENCE_AUXILIARY,
    RAW_TO_CAUSAL_INIT_BOUNDARY_TRANSITION,
    ZERO_CHANNEL_ORDINARY_CONTROL,
    ZERO_CHANNEL_MATCHED_CONTROL,
    assert_resume_training_args,
    boundary_auxiliary_loss,
    build_model,
    manifest_train_val_test_split,
    parse_args,
    selection_tuple,
    validate_args,
    warmup_cosine_factor,
)
from scripts.time_dependent_no.train_pcno_euler2d_residual import (
    main as train_main,
)
from tests.time_dependent_no._pcno_test_support import (
    prepare_boundary_synthetic_shards as _prepare_boundary_synthetic_shards,
)
from utility.time_dependent_no.cpg_mesh_contract import (
    INFLOW_NODE,
    NORMAL_NODE,
    OUTFLOW_NODE,
)
from utility.time_dependent_no.pcno_artifacts import (
    jsonable_args,
    verify_source_snapshot,
    write_source_snapshot,
)
from utility.time_dependent_no.pcno_euler2d import (
    NODE_TYPE_FEATURE_CONSTANT_ZERO,
    NODE_TYPE_FEATURE_OMITTED,
    NODE_TYPE_FEATURE_ONE_HOT,
    Euler2DNormalization,
    PCNOEuler2DResidual,
    PCNOEuler2DShardStore,
    apply_admissible_primitive_noise,
    apply_causal_boundary_conservative_batch,
    apply_minimum_change_boundary_conservative_batch,
    balanced_presentations,
    boundary_band_normal_node_mask,
    build_graph_causal_boundary_policy,
    build_graph_minimum_change_boundary_policy,
    conservative_admissibility,
    conservative_to_primitive_torch,
    copy_no_type_initialization_to_zero_channels,
    directional_highpass_stability,
    fit_normalization,
    full_coverage_presentations,
    graph_neighbor_highpass,
    homogeneous_optimizer_step_count,
    homogeneous_presentation_batches,
    normal_node_mask,
    primitive_to_conservative_torch,
    reference_smooth_region_mask,
    stratified_train_val_split,
    weighted_scaled_mse,
)
from utility.time_dependent_no.pcno_rollout import (
    LEARNED_DOFS_CLOSED_PRIMARY_OBJECTIVE,
    MINIMUM_CHANGE_BOUNDARY_MODE,
    NORMAL_CLOSED_PRIMARY_OBJECTIVE,
    RAW_ALL_NODES_PRIMARY_OBJECTIVE,
    close_boundary,
    primary_training_metrics,
    rollout_trajectory,
)
from utility.time_dependent_no.pcno_runtime import forward_sample


def test_all_normal_training_input_does_not_mutate_physical_type_tensor() -> None:
    class RecordingModel(torch.nn.Module):
        model_node_type_input = "all_normal"

        def __init__(self) -> None:
            super().__init__()
            self.seen_node_type: torch.Tensor | None = None

        def forward(
            self, current: torch.Tensor, **kwargs: torch.Tensor
        ) -> torch.Tensor:
            self.seen_node_type = kwargs["node_type"].detach().clone()
            return current

    model = RecordingModel()
    current = torch.zeros(1, 3, 4)
    physical_type = torch.tensor([[0, 1, 3]], dtype=torch.int64)
    sample = {
        "node_mask": torch.ones(1, 3, 1),
        "nodes": torch.zeros(1, 3, 2),
        "node_weights": torch.ones(1, 3, 1) / 3.0,
        "node_rhos": torch.ones(1, 3, 1),
        "directed_edges": torch.zeros(1, 1, 2, dtype=torch.int64),
        "edge_gradient_weights": torch.zeros(1, 1, 2),
        "node_type": physical_type,
        "mach": torch.ones(1),
    }

    assert torch.equal(forward_sample(model, sample, current), current)
    assert torch.equal(model.seen_node_type, torch.zeros_like(physical_type))
    assert torch.equal(sample["node_type"], physical_type)


def test_no_type_channels_do_not_change_the_frozen_physical_boundary_policy() -> None:
    common = [
        "--data-dir",
        "unused",
        "--output-dir",
        "unused",
        "--amp",
        "none",
        "--boundary-mode",
        "causal_nodal_physical",
    ]
    no_type = parse_args(
        [*common, "--node-type-channel-control", NO_TYPE_CHANNEL_CONTROL]
    )
    validate_args(no_type, torch.device("cpu"))

    constant_zero = parse_args(
        [*common, "--node-type-channel-control", ZERO_CHANNEL_ORDINARY_CONTROL]
    )
    with pytest.raises(ValueError, match="constant-zero-channel controls"):
        validate_args(constant_zero, torch.device("cpu"))


def _unit_normalization() -> Euler2DNormalization:
    return Euler2DNormalization(
        state_mean=np.zeros(4),
        state_scale=np.ones(4),
        residual_scale=np.ones(4),
        mach_mean=0.0,
        mach_scale=1.0,
    )


def test_node_type_channel_controls_have_exact_input_layouts() -> None:
    common = {
        "normalization": _unit_normalization(),
        "k_max": 1,
        "domain_lengths": (1.0, 1.0),
        "layers": (8, 8),
        "fc_dim": 8,
    }
    one_hot = PCNOEuler2DResidual(
        **common, node_type_feature_mode=NODE_TYPE_FEATURE_ONE_HOT
    )
    zero_four = PCNOEuler2DResidual(
        **common, node_type_feature_mode=NODE_TYPE_FEATURE_CONSTANT_ZERO
    )
    no_type = PCNOEuler2DResidual(
        **common, node_type_feature_mode=NODE_TYPE_FEATURE_OMITTED
    )
    current = torch.tensor(
        [[[1.0, 0.2, 0.1, 2.6], [0.9, 0.3, -0.1, 2.4]]]
    )
    nodes = torch.tensor([[[0.25, 0.25], [0.75, 0.75]]])
    node_rhos = torch.full((1, 2, 1), 0.5)
    node_type = torch.tensor([[0, 3]], dtype=torch.int64)
    mach = torch.tensor([1.1])

    def model_input(model: PCNOEuler2DResidual) -> torch.Tensor:
        return model.normalized_input(
            current,
            nodes=nodes,
            node_rhos=node_rhos,
            node_type=node_type,
            mach=mach,
        )

    one_hot_input = model_input(one_hot)
    zero_input = model_input(zero_four)
    no_type_input = model_input(no_type)
    assert one_hot_input.shape[-1] == 12
    assert zero_input.shape[-1] == 12
    assert no_type_input.shape[-1] == 8
    torch.testing.assert_close(
        one_hot_input[..., 7:11],
        torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]]),
    )
    assert torch.count_nonzero(zero_input[..., 7:11]).item() == 0
    torch.testing.assert_close(no_type_input[..., :7], zero_input[..., :7])
    torch.testing.assert_close(no_type_input[..., 7], zero_input[..., 11])


def test_matched_zero_channel_initialization_copies_no_type_lift_exactly() -> None:
    common = {
        "normalization": _unit_normalization(),
        "k_max": 1,
        "domain_lengths": (1.0, 1.0),
        "layers": (8, 8),
        "fc_dim": 8,
        "zero_initialize": False,
    }
    torch.manual_seed(31)
    no_type = PCNOEuler2DResidual(
        **common, node_type_feature_mode=NODE_TYPE_FEATURE_OMITTED
    )
    torch.manual_seed(97)
    zero_four = PCNOEuler2DResidual(
        **common, node_type_feature_mode=NODE_TYPE_FEATURE_CONSTANT_ZERO
    )
    audit = copy_no_type_initialization_to_zero_channels(no_type, zero_four)
    assert audit["mathematical_initial_function_match"] is True
    assert audit["active_weight_rescale_factor"] == 1.0

    active = torch.linspace(-0.7, 0.8, 48).reshape(2, 3, 8)
    expanded = torch.cat(
        (active[..., :7], torch.zeros(2, 3, 4), active[..., 7:8]), dim=-1
    )
    assert torch.equal(
        no_type.backbone.fc0(active), zero_four.backbone.fc0(expanded)
    )
    source_state = no_type.state_dict()
    target_state = zero_four.state_dict()
    for name in source_state:
        if name != "backbone.fc0.weight":
            assert torch.equal(source_state[name], target_state[name]), name


def test_trainer_matched_control_restores_no_type_rng_and_parameters() -> None:
    common = {
        "k_max": 1,
        "domain_lengths": (1.0, 1.0),
        "layers": (8, 8),
        "fc_dim": 8,
        "model_node_type_input": "physical",
    }
    torch.manual_seed(211)
    no_type = build_model(
        SimpleNamespace(
            **common, node_type_channel_control=NO_TYPE_CHANNEL_CONTROL
        ),
        _unit_normalization(),
        zero_initialize=False,
    )
    rng_after_no_type = torch.get_rng_state().clone()
    assert no_type.initialization_control["ordinary_initialization"] is True

    torch.manual_seed(211)
    matched = build_model(
        SimpleNamespace(
            **common, node_type_channel_control=ZERO_CHANNEL_MATCHED_CONTROL
        ),
        _unit_normalization(),
        zero_initialize=False,
    )
    assert torch.equal(torch.get_rng_state(), rng_after_no_type)
    assert matched.initialization_control["cpu_rng_state_matches_no_type_arm"] is True
    assert matched.initialization_control["mathematical_initial_function_match"] is True
    source_state = no_type.state_dict()
    target_state = matched.state_dict()
    for name in source_state:
        if name != "backbone.fc0.weight":
            assert torch.equal(source_state[name], target_state[name]), name


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


def test_bump_shards_publish_manifest_bound_semantic_collars(
    tmp_path: Path,
) -> None:
    source = tmp_path / "raw_boundary_fields.h5"
    with h5py.File(source, "w") as handle:
        _write_raw_trajectory(handle.create_group("0"), 0)
    output = tmp_path / "boundary_field_shards"
    prepare_main(
        [
            "--source-h5",
            str(source),
            "--output-dir",
            str(output),
            "--static-check",
            "all",
            "--boundary-collar-width",
            "0.25",
        ]
    )

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    contract = manifest["boundary_field_contract"]
    assert contract["continuum_object"] == "bounded_volume_descriptor"
    assert contract["surface_measure_scaling"] is False
    assert contract["physical_width"] == pytest.approx(0.25)
    assert contract["channel_names"] == ["wall", "outflow", "inflow"]
    store = PCNOEuler2DShardStore(output)
    assert store.boundary_field_names == ("wall", "outflow", "inflow")
    features = np.asarray(store.array("0", "boundary_features"))
    assert features.shape == (4, 3)
    assert np.all((0.0 <= features) & (features <= 1.0))
    sample = store.tensor_batch(
        "0", [0], step_stride=1, device=torch.device("cpu")
    )
    assert sample["boundary_features"].shape == (1, 4, 3)
    model = build_model(
        SimpleNamespace(
            k_max=1,
            domain_lengths=(1.0, 1.0),
            layers=(8, 8),
            fc_dim=8,
            model_node_type_input="physical",
            node_type_channel_control=NO_TYPE_CHANNEL_CONTROL,
            boundary_field_mode="semantic_collar",
            boundary_field_names=list(store.boundary_field_names),
        ),
        _unit_normalization(),
        zero_initialize=False,
    )
    prediction = forward_sample(model, sample, sample["current"])
    assert prediction.shape == sample["current"].shape
    assert torch.isfinite(prediction).all()


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


def test_full_coverage_and_warmup_cosine_have_exact_step_semantics(
    tmp_path: Path,
) -> None:
    data_dir = _prepare_synthetic_shards(tmp_path)
    store = PCNOEuler2DShardStore(data_dir)
    pairs = full_coverage_presentations(
        store,
        store.keys,
        step_stride=1,
        rng=np.random.default_rng(23),
    )
    expected = {
        (key, time_index)
        for key in store.keys
        for time_index in range(int(store.entry(key)["num_steps"]) - 1)
    }
    assert set(pairs) == expected
    assert len(pairs) == len(expected)
    assert len(pairs) == len(set(pairs))
    assert homogeneous_optimizer_step_count(pairs, batch_size=2) == 4
    store.close()

    factors = [
        warmup_cosine_factor(
            step,
            total_steps=10,
            warmup_steps=3,
            start_factor=0.1,
            minimum_factor=0.02,
        )
        for step in range(10)
    ]
    assert factors[0] == pytest.approx(0.1)
    assert factors[2] == pytest.approx(1.0)
    assert factors[3] == pytest.approx(1.0)
    assert factors[-1] == pytest.approx(0.02)
    assert all(left <= right for left, right in zip(factors[:2], factors[1:3]))
    assert all(left >= right for left, right in zip(factors[3:-1], factors[4:]))


def test_parity_selection_uses_historical_all_node_metric() -> None:
    rollout = {
        "completion_rate": 1.0,
        "mean_survival_fraction": 1.0,
        "mean_selection_relative_l2": 0.03,
        "mean_endpoint_relative_l2": {"20": 0.02},
    }
    assert selection_tuple(rollout, {"relative_l2": 0.004}) == pytest.approx(
        (1.0, 1.0, 1.0, -0.03, -0.02, -0.004)
    )

    parity_rollout = {
        **rollout,
        "parity": {
            "completion_rate": 1.0,
            "mean_relative_l2": 0.024,
        },
    }
    thresholds = {
        "parity_max_rollout_relative_l2": 0.0243,
        "parity_max_one_step_relative_l2": 0.0054,
    }
    with pytest.raises(ValueError, match="all-node one-step"):
        selection_tuple(parity_rollout, {"relative_l2": 0.004}, **thresholds)

    one_step = {"relative_l2": 0.004, "all_relative_l2": 0.005}
    passed = selection_tuple(parity_rollout, one_step, **thresholds)
    assert passed[:3] == (1.0, 1.0, 1.0)

    rollout_miss = {
        **parity_rollout,
        "parity": {
            "completion_rate": 1.0,
            "mean_relative_l2": 0.025,
        },
    }
    assert selection_tuple(rollout_miss, one_step, **thresholds)[0] == 0.0
    one_step_miss = {"relative_l2": 0.004, "all_relative_l2": 0.006}
    assert selection_tuple(parity_rollout, one_step_miss, **thresholds)[0] == 0.0


def test_resume_contract_and_source_snapshot_reject_scientific_drift(
    tmp_path: Path,
) -> None:
    args = parse_args(
        [
            "--data-dir",
            str(tmp_path / "data"),
            "--output-dir",
            str(tmp_path / "run"),
        ]
    )
    checkpoint = {"training_args": jsonable_args(args)}
    args.resume_checkpoint = tmp_path / "run" / "last.pt"
    args.max_wall_hours = 23.0
    assert_resume_training_args(checkpoint, args)
    args.learning_rate *= 2.0
    with pytest.raises(ValueError, match="frozen training contract"):
        assert_resume_training_args(checkpoint, args)

    snapshot = write_source_snapshot(tmp_path / "snapshot")
    verify_source_snapshot(snapshot)
    corrupted = json.loads(json.dumps(snapshot))
    first_source = next(iter(corrupted["files"]))
    corrupted["files"][first_source]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="current source differs"):
        verify_source_snapshot(corrupted)


def test_rollout_refuses_to_censor_a_requested_horizon(tmp_path: Path) -> None:
    data_dir = _prepare_synthetic_shards(tmp_path)
    store = PCNOEuler2DShardStore(data_dir)
    normalization = fit_normalization(store, store.keys)
    model = PCNOEuler2DResidual(
        normalization=normalization,
        k_max=1,
        domain_lengths=(1.0, 1.0),
        layers=(8, 8),
        fc_dim=8,
    )
    with pytest.raises(ValueError, match="4 were requested"):
        rollout_trajectory(
            model,
            store,
            store.keys[0],
            step_stride=1,
            start_frame=0,
            num_steps=4,
            device=torch.device("cpu"),
            amp="none",
        )
    store.close()


def test_causal_boundary_closure_is_current_state_only_and_differentiable(
    tmp_path: Path,
) -> None:
    data_dir = _prepare_boundary_synthetic_shards(tmp_path)
    store = PCNOEuler2DShardStore(data_dir)
    key = store.keys[0]
    policy, metadata = build_graph_causal_boundary_policy(
        store,
        key,
        device=torch.device("cpu"),
    )
    state = torch.tensor(
        np.array(store.states(key)[0:2], copy=True),
        dtype=torch.float32,
        requires_grad=True,
    )
    closed = apply_causal_boundary_conservative_batch(state, policy, gamma=1.4)
    primitive = conservative_to_primitive_torch(closed)
    torch.testing.assert_close(closed[:, 4], state[:, 4], rtol=0.0, atol=0.0)

    inflow_nodes = policy["inflow_nodes"]
    expected_inflow = (
        policy["freestream"]
        .reshape(1, 1, 4)
        .expand(state.shape[0], inflow_nodes.numel(), 4)
    )
    torch.testing.assert_close(
        primitive[:, inflow_nodes], expected_inflow, rtol=1e-6, atol=1e-6
    )
    wall_rows = policy["wall_rows"]
    wall_nodes = policy["target_nodes"][wall_rows]
    wall_normals = policy["target_normals"][wall_rows]
    wall_normal_velocity = (
        primitive[:, wall_nodes, 1:3] * wall_normals.unsqueeze(0)
    ).sum(dim=-1)
    torch.testing.assert_close(
        wall_normal_velocity,
        torch.zeros_like(wall_normal_velocity),
        rtol=0.0,
        atol=1e-6,
    )
    node_type = torch.tensor(
        np.array(store.array(key, "node_type"), copy=True), dtype=torch.int64
    ).reshape(1, -1, 1)
    mask = normal_node_mask(
        node_type,
        torch.ones((1, state.shape[1], 1), dtype=torch.float32),
    )
    assert int(mask.sum()) == 1
    closed.square().mean().backward()
    assert state.grad is not None
    assert bool(torch.isfinite(state.grad).all())
    assert float(state.grad[:, 4].abs().sum()) > 0.0
    assert metadata["fallback_target_count"] == 0
    assert metadata["future_reference_boundary_values"] is False
    assert metadata["exact_dg_boundary_replay"] is False
    store.close()


def test_minimum_change_boundary_preserves_deployed_free_dofs(
    tmp_path: Path,
) -> None:
    data_dir = _prepare_boundary_synthetic_shards(tmp_path)
    store = PCNOEuler2DShardStore(data_dir)
    key = store.keys[0]
    policy, metadata = build_graph_minimum_change_boundary_policy(
        store,
        key,
        device=torch.device("cpu"),
    )
    state = torch.tensor(
        np.array(store.states(key)[0:2], copy=True),
        dtype=torch.float32,
        requires_grad=True,
    )
    before = conservative_to_primitive_torch(state)
    closed_state = apply_minimum_change_boundary_conservative_batch(
        state, policy, gamma=1.4
    )
    closed = conservative_to_primitive_torch(closed_state)

    # The sole interior node and the ordinary (non-junction) outflow node are
    # untouched. Wall density, pressure, and tangential velocity remain learned.
    torch.testing.assert_close(closed_state[:, 4], state[:, 4], rtol=0.0, atol=0.0)
    torch.testing.assert_close(closed_state[:, 5], state[:, 5], rtol=0.0, atol=0.0)
    wall_node = 1
    torch.testing.assert_close(
        closed[:, wall_node, (0, 1, 3)],
        before[:, wall_node, (0, 1, 3)],
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    torch.testing.assert_close(
        closed[:, wall_node, 2],
        torch.zeros_like(closed[:, wall_node, 2]),
        rtol=0.0,
        atol=1.0e-6,
    )
    expected_inflow = policy["freestream"].reshape(1, 1, 4).expand(2, 3, 4)
    torch.testing.assert_close(
        closed[:, policy["inflow_nodes"]],
        expected_inflow,
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    torch.testing.assert_close(
        apply_minimum_change_boundary_conservative_batch(
            closed_state, policy, gamma=1.4
        ),
        closed_state,
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        autocast_closed = apply_minimum_change_boundary_conservative_batch(
            state, policy, gamma=1.4
        )
    assert autocast_closed.dtype == state.dtype
    torch.testing.assert_close(autocast_closed, closed_state, rtol=1.0e-6, atol=1.0e-6)

    loss = (
        closed[:, wall_node, 0].sum()
        + closed[:, wall_node, 1].sum()
        + closed[:, wall_node, 2].sum()
        + closed[:, wall_node, 3].sum()
        + closed[:, 5].sum()
        + closed[:, policy["inflow_nodes"]].sum()
    )
    loss.backward()
    assert state.grad is not None
    assert float(state.grad[:, wall_node].abs().sum()) > 0.0
    assert float(state.grad[:, 5].abs().sum()) > 0.0
    assert float(state.grad[:, policy["inflow_nodes"]].abs().sum()) == 0.0
    assert metadata["rank_two_wall_corner_count"] == 0
    assert metadata["incident_wall_junction_count"] == 4
    assert metadata["future_reference_boundary_values"] is False
    assert metadata["physical_conservation_claim"] is False
    store.close()


def test_learned_dof_objective_is_dense_only_on_deployed_components(
    tmp_path: Path,
) -> None:
    data_dir = _prepare_boundary_synthetic_shards(tmp_path)
    store = PCNOEuler2DShardStore(data_dir)
    key = store.keys[0]
    normalization = fit_normalization(store, store.keys)
    model = PCNOEuler2DResidual(
        normalization=normalization,
        k_max=1,
        domain_lengths=(1.0, 1.0),
        layers=(8, 8),
        fc_dim=8,
    )
    policy, _ = build_graph_minimum_change_boundary_policy(
        store, key, device=torch.device("cpu")
    )
    sample = store.tensor_batch(key, [0], step_stride=1, device=torch.device("cpu"))
    perturbation = torch.zeros_like(sample["target"])
    perturbation[:, 0, :] = 0.5  # fixed inflow: must not train the proposal
    perturbation[:, 1, 0] = 0.05  # wall density: deployed and supervised
    perturbation[:, 4, 3] = 0.05  # interior energy: deployed and supervised
    perturbation[:, 5, 0] = 0.05  # outgoing outflow: deployed and supervised
    raw_prediction = (sample["target"] + perturbation).detach().requires_grad_(True)
    prediction = close_boundary(raw_prediction, policy, gamma=model.gamma)

    loss, relative_l2 = primary_training_metrics(
        prediction,
        raw_prediction,
        sample["target"],
        sample,
        model,
        boundary_policy=policy,
        primary_objective=LEARNED_DOFS_CLOSED_PRIMARY_OBJECTIVE,
    )
    assert float(loss.detach()) > 0.0
    assert float(relative_l2.detach()) > 0.0
    loss.backward()
    assert raw_prediction.grad is not None
    assert float(raw_prediction.grad[:, 0].abs().sum()) == 0.0
    assert float(raw_prediction.grad[:, 1].abs().sum()) > 0.0
    assert float(raw_prediction.grad[:, 4].abs().sum()) > 0.0
    assert float(raw_prediction.grad[:, 5].abs().sum()) > 0.0
    store.close()


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


def test_boundary_band_normal_node_mask_respects_graph_hops() -> None:
    node_type = torch.tensor(
        [[INFLOW_NODE, NORMAL_NODE, NORMAL_NODE, NORMAL_NODE, OUTFLOW_NODE]]
    )
    node_mask = torch.ones((1, 5, 1))
    undirected = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 4]])
    directed = torch.cat((undirected, undirected.flip(-1)), dim=0).unsqueeze(0)

    one_hop = boundary_band_normal_node_mask(node_type, node_mask, directed, max_hops=1)
    two_hop = boundary_band_normal_node_mask(node_type, node_mask, directed, max_hops=2)

    assert one_hop[0, :, 0].tolist() == [0.0, 1.0, 0.0, 1.0, 0.0]
    assert two_hop[0, :, 0].tolist() == [0.0, 1.0, 1.0, 1.0, 0.0]


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
    store.close()


def test_shard_store_geometry_cache_matches_uncached_and_is_bounded(
    tmp_path: Path,
) -> None:
    output = _prepare_synthetic_shards(tmp_path)
    device = torch.device("cpu")
    static_names = (
        "node_mask",
        "nodes",
        "node_measures",
        "node_weights",
        "node_rhos",
        "directed_edges",
        "edge_gradient_weights",
        "node_type",
    )

    with PCNOEuler2DShardStore(
        output, max_cached_trajectories=1
    ) as cached_store:
        with PCNOEuler2DShardStore(
            output, max_cached_geometry_bytes=0
        ) as uncached_store:
            key = cached_store.keys[0]
            cached_first = cached_store.tensor_batch(
                key, [0, 1], step_stride=1, device=device
            )
            uncached_first = uncached_store.tensor_batch(
                key, [0, 1], step_stride=1, device=device
            )
            assert cached_first.keys() == uncached_first.keys()
            for name in cached_first:
                torch.testing.assert_close(
                    cached_first[name],
                    uncached_first[name],
                    rtol=0.0,
                    atol=0.0,
                )

            assert cached_store.cached_geometry_entries == 1
            entry_bytes = cached_store.cached_geometry_bytes
            assert entry_bytes > 0
            assert uncached_store.cached_geometry_entries == 0
            assert uncached_store.cached_geometry_bytes == 0

            cached_store.states(cached_store.keys[1])
            cached_second = cached_store.tensor_batch(
                key, [1], step_stride=1, device=device
            )
            uncached_second = uncached_store.tensor_batch(
                key, [1], step_stride=1, device=device
            )
            for name in static_names:
                assert (
                    cached_first[name].untyped_storage().data_ptr()
                    == cached_second[name].untyped_storage().data_ptr()
                )
                assert (
                    uncached_first[name].untyped_storage().data_ptr()
                    != uncached_second[name].untyped_storage().data_ptr()
                )
            assert (
                cached_first["current"].untyped_storage().data_ptr()
                != cached_second["current"].untyped_storage().data_ptr()
            )
            assert set(cached_store._arrays[key]) == {"states_conservative"}

            cached_store.clear_geometry_cache(device)
            assert cached_store.cached_geometry_entries == 0
            assert cached_store.cached_geometry_bytes == 0

    assert cached_store.cached_geometry_entries == 0
    assert cached_store.cached_geometry_bytes == 0

    with PCNOEuler2DShardStore(
        output, max_cached_geometry_bytes=entry_bytes
    ) as evicting_store:
        evicting_store.tensor_sample(
            evicting_store.keys[0], 0, step_stride=1, device=device
        )
        evicting_store.tensor_sample(
            evicting_store.keys[1], 0, step_stride=1, device=device
        )
        assert evicting_store.cached_geometry_entries == 1
        assert evicting_store.cached_geometry_bytes <= entry_bytes
        assert list(evicting_store._geometry_tensors[device]) == [
            evicting_store.keys[1]
        ]

    with PCNOEuler2DShardStore(
        output, max_cached_geometry_bytes=entry_bytes - 1
    ) as undersized_store:
        undersized_store.tensor_sample(
            undersized_store.keys[0], 0, step_stride=1, device=device
        )
        assert undersized_store.cached_geometry_entries == 0
        assert undersized_store.cached_geometry_bytes == 0

    with pytest.raises(ValueError, match="max_cached_geometry_bytes"):
        PCNOEuler2DShardStore(output, max_cached_geometry_bytes=-1)


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


def test_manifest_split_accepts_an_explicit_open_only_population(
    tmp_path: Path,
) -> None:
    data_dir = _prepare_manifest_synthetic_shards(tmp_path)
    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["requested_splits"] = ["train", "validation"]
    manifest["trajectories"] = manifest["trajectories"][:2]
    manifest["splits"]["test"] = []
    manifest["prepared_split_counts"]["test"] = 0
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    store = PCNOEuler2DShardStore(data_dir)
    train_keys, val_keys, test_keys = manifest_train_val_test_split(store)
    assert train_keys == ["0"]
    assert val_keys == ["1"]
    assert test_keys == []
    store.close()

    manifest["splits"]["test"] = ["2"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    inconsistent_store = PCNOEuler2DShardStore(data_dir)
    with pytest.raises(ValueError, match="outside requested_splits"):
        manifest_train_val_test_split(inconsistent_store)
    inconsistent_store.close()


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
        "smoothing": False,
        "boundary_decode_reencode_closure": False,
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
    assert (
        summary["boundary_contract"]["training_objective"][
            "primary_node_population"
        ]
        == "all_valid_nodes"
    )
    assert (
        summary["boundary_contract"]["training_objective"]["primary_prediction"]
        == "raw_model_proposal_equal_to_deployed_proposal"
    )
    assert summary["batch_size"] == 4
    assert summary["optimizer"]["scheduler"] == "constant"
    assert len(summary["config_digest"]) == 64
    assert set(summary["code_sha256"]) == {
        "trainer",
        "pcno_euler2d",
        "pcno_core",
        "cpg_mesh_contract",
        "pcno_artifacts",
        "euler2d_metrics",
        "pcno_runtime",
        "pcno_rollout",
        "pcno_ripple_diagnostics",
        "evaluator",
    }
    assert all(len(value) == 64 for value in summary["code_sha256"].values())
    assert summary["parent_checkpoint"] is None
    assert summary["tiny_fit"]["passed"] is True
    assert len(split["rollout_keys"]) == 1
    assert set(split["rollout_keys"]).issubset(split["val_keys"])
    assert (output_dir / "best.pt").is_file()
    assert (output_dir / "last.pt").is_file()
    assert (output_dir / "tiny_best.pt").is_file()


def test_cpu_serious_contract_uses_full_coverage_and_matched_boundary(
    tmp_path: Path,
) -> None:
    data_dir = _prepare_boundary_synthetic_shards(tmp_path)
    output_dir = tmp_path / "serious_contract"
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
            "--presentation-mode",
            "full_coverage",
            "--batch-size",
            "2",
            "--val-presentations",
            "2",
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
            "--scheduler",
            "warmup_cosine",
            "--min-learning-rate",
            "0.0001",
            "--rollout-every",
            "1",
            "--rollout-val-count",
            "1",
            "--rollout-steps",
            "3",
            "--rollout-checkpoints",
            "1",
            "2",
            "3",
            "--boundary-mode",
            "causal_nodal_physical",
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
    exposure = summary["exposure_contract"]
    assert exposure["presentation_mode"] == "full_coverage"
    assert exposure["coverage_passes_requested"] == 1
    assert exposure["unique_eligible_train_transitions"] == 3
    assert exposure["presentations_per_epoch"] == 3
    assert exposure["optimizer_steps_per_epoch"] == 2
    assert exposure["requested_presentations"] == 3
    assert exposure["requested_optimizer_steps"] == 2
    assert exposure["microbatch_size"] == 2
    assert exposure["gradient_accumulation_steps"] == 1
    assert exposure["effective_batch_size"] == 2
    assert summary["actual_presentations"] == 3
    assert summary["actual_optimizer_steps"] == 2
    assert summary["boundary_mode"] == "causal_nodal_physical"
    assert summary["raw_recurrence"] is False
    assert summary["autonomous_recurrence"] is True
    assert summary["boundary_contract"]["teacher_input_closure"] is True
    assert summary["boundary_contract"]["proposal_closure"] is True
    assert summary["boundary_contract"]["recurrence_closure"] is True
    assert summary["boundary_contract"]["state_loss_mask"] == "normal_nodes_only"
    assert summary["boundary_contract"]["future_reference_boundary_values"] is False
    assert summary["inference_interventions"]["decode_reencode_projection"] is True
    assert summary["last_validation"]["normal_relative_l2"] is not None
    assert summary["last_validation"]["all_relative_l2"] is not None
    assert summary["last_validation"]["boundary_relative_l2"] is not None
    assert split["validation_pair_seed"] == 20261709
    assert len(split["validation_pairs"]) == 2
    assert summary["last_rollout"]["endpoint_population_count"] == {
        "1": 1,
        "2": 1,
        "3": 1,
    }
    assert checkpoint["boundary_contract"] == summary["boundary_contract"]
    assert checkpoint["exposure_contract"] == exposure
    assert checkpoint["source_snapshot"] == summary["source_snapshot"]
    assert len(checkpoint["source_snapshot"]["source_set_digest"]) == 64
    assert summary["gpu_max_memory_bytes"] == 0
    assert summary["gpu_max_reserved_memory_bytes"] == 0
    assert (output_dir / "run_contract.json").is_file()
    assert (output_dir / "source_snapshot" / "manifest.json").is_file()


def test_cpu_projected_boundary_auxiliary_is_recorded_and_optimized(
    tmp_path: Path,
) -> None:
    data_dir = _prepare_boundary_synthetic_shards(tmp_path)
    output_dir = tmp_path / "projected_boundary_auxiliary"
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
            "--batch-size",
            "2",
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
            "--boundary-mode",
            "causal_nodal_physical",
            "--boundary-auxiliary",
            "projected_target",
            "--boundary-auxiliary-weight",
            "0.1",
            "--device",
            "cpu",
            "--amp",
            "none",
        ]
    )

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    objective = summary["boundary_contract"]["training_objective"]
    assert objective["auxiliary"] == "projected_target"
    assert objective["auxiliary_weight"] == pytest.approx(0.1)
    assert objective["projected_target_boundary_nodes"] == "wall_and_outflow"
    assert objective["fixed_inflow_auxiliary_gradient"] == "excluded"
    assert summary["last_train"]["boundary_auxiliary_loss"] is not None
    assert summary["last_train"]["loss"] >= summary["last_train"]["clean_loss"]


def test_raw_boundary_reference_auxiliary_has_only_raw_boundary_gradient() -> None:
    target = torch.ones((1, 3, 4), dtype=torch.float32)
    prediction = target.clone()
    raw_prediction = target.clone()
    raw_prediction[:, 0, :] = 3.0
    raw_prediction[:, 2, :] = 2.0
    raw_prediction.requires_grad_()
    sample = {
        "node_mask": torch.ones((1, 3, 1), dtype=torch.float32),
        "node_type": torch.tensor([[NORMAL_NODE, NORMAL_NODE, OUTFLOW_NODE]]),
        "node_weights": torch.ones((1, 3, 1), dtype=torch.float32),
    }

    class ScaleOnlyModel:
        state_scale = torch.ones(4, dtype=torch.float32)

    loss = boundary_auxiliary_loss(
        prediction,
        target,
        sample,
        ScaleOnlyModel(),
        boundary_policy={},
        kind=RAW_BOUNDARY_REFERENCE_AUXILIARY,
        near_boundary_hops=3,
        raw_prediction=raw_prediction,
    )
    assert loss > 0.0
    loss.backward()
    assert raw_prediction.grad is not None
    assert torch.count_nonzero(raw_prediction.grad[:, :2, :]) == 0
    assert torch.count_nonzero(raw_prediction.grad[:, 2:, :]) == 4


def test_cpu_raw_boundary_reference_auxiliary_is_recorded_and_keeps_causal_recurrence(
    tmp_path: Path,
) -> None:
    data_dir = _prepare_boundary_synthetic_shards(tmp_path)
    output_dir = tmp_path / "raw_boundary_reference_auxiliary"
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
            "--batch-size",
            "2",
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
            "--boundary-mode",
            "causal_nodal_physical",
            "--boundary-auxiliary",
            RAW_BOUNDARY_REFERENCE_AUXILIARY,
            "--boundary-auxiliary-weight",
            "0.0018733749",
            "--device",
            "cpu",
            "--amp",
            "none",
        ]
    )

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    objective = summary["boundary_contract"]["training_objective"]
    assert objective["primary_kind"] == NORMAL_CLOSED_PRIMARY_OBJECTIVE
    assert objective["auxiliary"] == RAW_BOUNDARY_REFERENCE_AUXILIARY
    assert objective["auxiliary_weight"] == pytest.approx(0.0018733749)
    assert (
        objective["auxiliary_prediction"] == "raw_model_proposal_before_causal_closure"
    )
    assert objective["auxiliary_node_population"] == "valid_non_normal_nodes"
    assert objective["reference_boundary_targets"] == "training_and_validation_only"
    assert summary["boundary_contract"]["state_loss_mask"] == "normal_nodes_only"
    assert summary["last_train"]["boundary_auxiliary_loss"] is not None
    assert summary["last_train"]["loss"] >= summary["last_train"]["clean_loss"]
    assert summary["last_validation"]["raw_boundary_relative_l2"] is not None
    assert summary["raw_recurrence"] is False
    assert summary["boundary_contract"]["recurrence_closure"] is True
    assert (
        summary["inference_interventions"]["future_reference_boundary_values"] is False
    )


def test_raw_all_node_primary_objective_backpropagates_through_raw_boundary() -> None:
    target = torch.ones((1, 3, 4), dtype=torch.float32)
    prediction = target.clone()
    raw_prediction = target.clone()
    raw_prediction[:, 2, :] = 2.0
    raw_prediction.requires_grad_()
    sample = {
        "node_mask": torch.ones((1, 3, 1), dtype=torch.float32),
        "node_type": torch.tensor([[NORMAL_NODE, NORMAL_NODE, OUTFLOW_NODE]]),
        "node_weights": torch.ones((1, 3, 1), dtype=torch.float32),
    }

    class ScaleOnlyModel:
        state_scale = torch.ones(4, dtype=torch.float32)

    normal_loss, _ = primary_training_metrics(
        prediction,
        raw_prediction,
        target,
        sample,
        ScaleOnlyModel(),
        boundary_policy={},
        primary_objective=NORMAL_CLOSED_PRIMARY_OBJECTIVE,
    )
    raw_loss, _ = primary_training_metrics(
        prediction,
        raw_prediction,
        target,
        sample,
        ScaleOnlyModel(),
        boundary_policy={},
        primary_objective=RAW_ALL_NODES_PRIMARY_OBJECTIVE,
    )
    assert normal_loss == pytest.approx(0.0)
    assert raw_loss > 0.0

    raw_loss.backward()
    assert raw_prediction.grad is not None
    assert torch.count_nonzero(raw_prediction.grad[:, :2, :]) == 0
    assert torch.count_nonzero(raw_prediction.grad[:, 2:, :]) == 4


@pytest.mark.parametrize(
    "extra_args, message",
    [
        ([], "requires causal boundary mode"),
        (
            [
                "--boundary-mode",
                "causal_nodal_physical",
                "--boundary-auxiliary",
                "projected_target",
                "--boundary-auxiliary-weight",
                "0.1",
            ],
            "boundary auxiliaries are separate studies",
        ),
    ],
)
def test_raw_all_node_primary_objective_rejects_confounded_contracts(
    tmp_path: Path,
    extra_args: list[str],
    message: str,
) -> None:
    args = parse_args(
        [
            "--data-dir",
            str(tmp_path / "data"),
            "--output-dir",
            str(tmp_path / "output"),
            "--primary-objective",
            RAW_ALL_NODES_PRIMARY_OBJECTIVE,
            *extra_args,
        ]
    )
    with pytest.raises(ValueError, match=message):
        validate_args(args, torch.device("cpu"))


def test_minimum_change_normal_closed_is_a_legal_diagnostic_control(
    tmp_path: Path,
) -> None:
    args = parse_args(
        [
            "--data-dir",
            str(tmp_path / "data"),
            "--output-dir",
            str(tmp_path / "output"),
            "--boundary-mode",
            MINIMUM_CHANGE_BOUNDARY_MODE,
            "--primary-objective",
            NORMAL_CLOSED_PRIMARY_OBJECTIVE,
            "--amp",
            "none",
        ]
    )
    validate_args(args, torch.device("cpu"))


def test_cpu_raw_all_node_supervision_is_recorded_and_keeps_causal_recurrence(
    tmp_path: Path,
) -> None:
    data_dir = _prepare_boundary_synthetic_shards(tmp_path)
    output_dir = tmp_path / "raw_all_node_primary"
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
            "--batch-size",
            "2",
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
            "--boundary-mode",
            "causal_nodal_physical",
            "--primary-objective",
            RAW_ALL_NODES_PRIMARY_OBJECTIVE,
            "--device",
            "cpu",
            "--amp",
            "none",
        ]
    )

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    objective = summary["boundary_contract"]["training_objective"]
    assert objective["primary_kind"] == RAW_ALL_NODES_PRIMARY_OBJECTIVE
    assert objective["primary_prediction"] == "raw_model_proposal_before_causal_closure"
    assert objective["primary_node_population"] == "all_valid_nodes"
    assert objective["reference_boundary_targets"] == "training_and_validation_only"
    assert summary["boundary_contract"]["state_loss_mask"] == "all_valid_nodes"
    assert summary["last_train"]["primary_objective"] == RAW_ALL_NODES_PRIMARY_OBJECTIVE
    assert summary["last_train"]["boundary_auxiliary_loss"] is None
    assert summary["last_validation"]["raw_boundary_relative_l2"] is not None
    assert summary["raw_recurrence"] is False
    assert (
        summary["inference_interventions"]["future_reference_boundary_values"] is False
    )


def test_cpu_minimum_change_training_and_validation_protocol(tmp_path: Path) -> None:
    data_dir = _prepare_boundary_synthetic_shards(tmp_path)
    output_dir = tmp_path / "minimum_change_training"
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
            "--batch-size",
            "2",
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
            "--boundary-mode",
            MINIMUM_CHANGE_BOUNDARY_MODE,
            "--primary-objective",
            LEARNED_DOFS_CLOSED_PRIMARY_OBJECTIVE,
            "--device",
            "cpu",
            "--amp",
            "none",
        ]
    )
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    objective = summary["boundary_contract"]["training_objective"]
    assert summary["boundary_mode"] == MINIMUM_CHANGE_BOUNDARY_MODE
    assert objective["primary_kind"] == LEARNED_DOFS_CLOSED_PRIMARY_OBJECTIVE
    assert objective["fixed_inflow_target_gradient"] == "zero_through_projection"
    assert objective["wall_tangential_density_pressure_and_outflow_supervision"] == (
        "enabled"
    )
    assert summary["boundary_contract"]["projection_metric"] == (
        "euclidean_primitive_variables"
    )
    assert summary["last_validation"]["raw_boundary_relative_l2"] is not None
    assert summary["raw_recurrence"] is False

    evaluation_dir = tmp_path / "minimum_change_protocol_evaluation"
    evaluate_boundary_protocol_main(
        [
            "--data-dir",
            str(data_dir),
            "--checkpoint",
            str(output_dir / "last.pt"),
            "--output-dir",
            str(evaluation_dir),
            "--one-step-presentations",
            "1",
            "--rollout-count",
            "1",
            "--rollout-steps",
            "1",
            "--rollout-checkpoints",
            "1",
            "--device",
            "cpu",
            "--amp",
            "none",
        ]
    )
    evaluation = json.loads(
        (evaluation_dir / "summary.json").read_text(encoding="utf-8")
    )
    assert evaluation["schema"] == "pcno_euler2d_boundary_protocol_validation_v2"
    assert evaluation["selection_population"] == "checkpoint_validation_split_only"
    assert evaluation["test_split_opened"] is False
    assert evaluation["evaluation"]["start_frame"] == 0
    assert (
        "utility/time_dependent_no/cpg_mesh_contract.py" in evaluation["source_sha256"]
    )
    assert "utility/time_dependent_no/pcno_rollout.py" in evaluation["source_sha256"]
    assert "utility/time_dependent_no/euler2d_metrics.py" in evaluation["source_sha256"]
    assert (
        "utility/time_dependent_no/pcno_ripple_diagnostics.py"
        in evaluation["source_sha256"]
    )
    assert (
        "scripts/time_dependent_no/train_pcno_euler2d_residual.py"
        not in evaluation["source_sha256"]
    )
    assert (
        "scripts/time_dependent_no/evaluate_pcno_euler2d_residual.py"
        not in evaluation["source_sha256"]
    )
    assert evaluation["native"]["rollout"]["completion_rate"] == 1.0
    assert evaluation["causal"]["rollout"]["completion_rate"] == 1.0
    assert evaluation["minimum_change"]["rollout"]["completion_rate"] == 1.0
    assert evaluation["native"]["structure"]["completion_rate"] == 1.0
    assert evaluation["causal"]["structure"]["completion_rate"] == 1.0
    assert evaluation["minimum_change"]["structure"]["completion_rate"] == 1.0
    assert evaluation["native"]["structure"]["endpoints"]["1"]["population_count"] == 1
    assert evaluation["paired_structure_ratios"]["ratio"] == (
        "minimum_change_over_native"
    )
    assert evaluation["paired_causal_structure_ratios"]["ratio"] == (
        "causal_over_native"
    )
    assert (
        evaluation["causal"]["projection_decomposition"]["wall_constraint_max_abs"]
        <= 1.0e-6
    )
    assert (
        evaluation["minimum_change"]["projection_decomposition"][
            "wall_constraint_max_abs"
        ]
        <= 1.0e-6
    )


def test_raw_checkpoint_requires_explicit_causal_initialization_transition(
    tmp_path: Path,
) -> None:
    data_dir = _prepare_boundary_synthetic_shards(tmp_path)
    checkpoint_path = _write_synthetic_checkpoint(tmp_path, data_dir)
    output_dir = tmp_path / "causal_continuation"
    common_args = [
        "--data-dir",
        str(data_dir),
        "--output-dir",
        str(output_dir),
        "--init-checkpoint",
        str(checkpoint_path),
        "--val-count",
        "1",
        "--epochs",
        "1",
        "--presentations-per-epoch",
        "2",
        "--val-presentations",
        "1",
        "--batch-size",
        "2",
        "--k-max",
        "1",
        "--domain-lengths",
        "1",
        "1",
        "--layers",
        "8",
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
        "--boundary-mode",
        "causal_nodal_physical",
        "--device",
        "cpu",
        "--amp",
        "none",
    ]

    with pytest.raises(ValueError, match="boundary modes differ without the exact"):
        train_main(common_args)

    train_main(
        [
            *common_args,
            "--init-boundary-mode-transition",
            RAW_TO_CAUSAL_INIT_BOUNDARY_TRANSITION,
        ]
    )

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    run_contract = json.loads(
        (output_dir / "run_contract.json").read_text(encoding="utf-8")
    )
    checkpoint = torch.load(
        output_dir / "last.pt", map_location="cpu", weights_only=False
    )
    transition = summary["initialization_transition"]
    assert transition["kind"] == RAW_TO_CAUSAL_INIT_BOUNDARY_TRANSITION
    assert transition["source_boundary_mode"] == "model_all_nodes"
    assert transition["target_boundary_mode"] == "causal_nodal_physical"
    assert transition["model_state_loaded_strictly"] is True
    assert transition["optimizer_state_loaded"] is False
    assert transition["scheduler_state_loaded"] is False
    assert transition["future_reference_boundary_values"] is False
    assert len(transition["parent_checkpoint_sha256"]) == 64
    assert len(transition["target_boundary_contract_digest"]) == 64
    assert (
        transition["parent_checkpoint_sha256"] == summary["parent_checkpoint"]["sha256"]
    )
    assert run_contract["initialization_transition"] == transition
    assert checkpoint["initialization_transition"] == transition
    assert checkpoint["boundary_mode"] == "causal_nodal_physical"


def test_gradient_accumulation_matches_the_effective_batch_update(
    tmp_path: Path,
) -> None:
    data_dir = _prepare_synthetic_shards(tmp_path)
    full_batch_dir = tmp_path / "full_batch"
    accumulated_dir = tmp_path / "accumulated"
    common_args = [
        "--data-dir",
        str(data_dir),
        "--val-count",
        "1",
        "--epochs",
        "1",
        "--presentations-per-epoch",
        "4",
        "--val-presentations",
        "2",
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
    train_main(
        [
            *common_args,
            "--output-dir",
            str(full_batch_dir),
            "--batch-size",
            "4",
        ]
    )
    train_main(
        [
            *common_args,
            "--output-dir",
            str(accumulated_dir),
            "--batch-size",
            "2",
            "--gradient-accumulation-steps",
            "2",
        ]
    )

    full_checkpoint = torch.load(
        full_batch_dir / "last.pt", map_location="cpu", weights_only=False
    )
    accumulated_checkpoint = torch.load(
        accumulated_dir / "last.pt", map_location="cpu", weights_only=False
    )
    for name, full_parameter in full_checkpoint["model_state"].items():
        torch.testing.assert_close(
            accumulated_checkpoint["model_state"][name],
            full_parameter,
            rtol=1e-5,
            atol=1e-7,
        )

    full_summary = json.loads(
        (full_batch_dir / "summary.json").read_text(encoding="utf-8")
    )
    accumulated_summary = json.loads(
        (accumulated_dir / "summary.json").read_text(encoding="utf-8")
    )
    assert full_summary["actual_optimizer_steps"] == 1
    assert accumulated_summary["actual_optimizer_steps"] == 1
    assert full_summary["effective_batch_size"] == 4
    assert accumulated_summary["microbatch_size"] == 2
    assert accumulated_summary["gradient_accumulation_steps"] == 2
    assert accumulated_summary["effective_batch_size"] == 4
    assert accumulated_summary["last_train"]["loss"] == pytest.approx(
        full_summary["last_train"]["loss"], rel=1e-6, abs=1e-8
    )


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


def test_official_residual_evaluator_runs_native_causal_checkpoint(
    tmp_path: Path,
) -> None:
    data_dir = _prepare_boundary_synthetic_shards(tmp_path)
    checkpoint_path = _write_synthetic_checkpoint(tmp_path, data_dir)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint["boundary_mode"] = "causal_nodal_physical"
    checkpoint["raw_recurrence"] = False
    checkpoint["boundary_contract"] = {
        "mode": "causal_nodal_physical",
        "max_source_hops": 3,
        "rho_inf": 1.4,
        "p_inf": 1.0,
        "policy_digests": {},
    }
    torch.save(checkpoint, checkpoint_path)

    output_dir = tmp_path / "native_causal_evaluation"
    evaluate_main(
        [
            "--data-dir",
            str(data_dir),
            "--training-data-dir",
            str(data_dir),
            "--checkpoint",
            str(checkpoint_path),
            "--output-dir",
            str(output_dir),
            "--trajectory-keys",
            "0",
            "--expected-trajectory-count",
            "1",
            "--num-steps",
            "3",
            "--endpoint-calls",
            "1",
            "2",
            "3",
            "--counterfactual",
            "none",
            "--device",
            "cpu",
        ]
    )

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["checkpoint"]["boundary_mode"] == "causal_nodal_physical"
    assert summary["evaluation"]["baseline_boundary_mode"] == ("causal_nodal_physical")
    assert summary["evaluation"]["candidate_boundary_mode"] is None
    assert summary["evaluation"]["raw_recurrence"] is False
    assert summary["evaluation"]["boundary_decode_reencode_closure"] is True
    assert set(summary["variants"]) == {"persistence", "pcno_baseline"}
    assert all(summary["diagnostic_gate"]["checks"].values())

    trajectory_rows = list(
        csv.DictReader(
            (output_dir / "trajectory_metrics.csv").open(encoding="utf-8", newline="")
        )
    )
    assert len(trajectory_rows) == 2
    assert {row["variant"] for row in trajectory_rows} == {
        "persistence",
        "pcno_baseline",
    }

    with np.load(output_dir / "trajectories" / "trajectory_0.npz") as artifact:
        assert artifact["baseline_boundary_mode"].item() == ("causal_nodal_physical")
        assert "candidate_variant" not in artifact.files
        metadata = json.loads(artifact["native_boundary_metadata"].item())
        assert metadata["applied_scope"] == "native_causal_nodal_physical"
        assert metadata["fallback_target_count"] == 0
        primitive = conservative_to_primitive_torch(
            torch.from_numpy(artifact["pcno_baseline_predictions_conservative"])
        ).numpy()
        inflow = artifact["node_type"].reshape(-1) == 3
        mach = float(artifact["mach"].item())
        expected = np.asarray([1.4, mach, 0.0, 1.0])
        np.testing.assert_allclose(
            primitive[:, inflow],
            np.broadcast_to(expected, primitive[:, inflow].shape),
            rtol=1e-6,
            atol=1e-6,
        )


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
            "fixed freestream inflow, current-interior slip wall, and "
            "current-interior supersonic outflow",
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
