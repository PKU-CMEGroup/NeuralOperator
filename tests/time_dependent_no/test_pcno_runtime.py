from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

import scripts.time_dependent_no.evaluate_pcno_euler2d_residual as bump_evaluator
import scripts.time_dependent_no.evaluate_pcno_shock_vortex_baseline as dynamic_evaluator
import scripts.time_dependent_no.train_pcno_euler2d_residual as trainer
from utility.time_dependent_no.pcno_euler2d import (
    NODE_TYPE_FEATURE_CONSTANT_ZERO,
    Euler2DNormalization,
    PCNOEuler2DResidual,
)
from utility.time_dependent_no.pcno_resolution_transfer import (
    build_resolution_checkpoint_model,
    load_resolution_checkpoint,
    predict_resolution_batch,
    predict_resolution_sample,
)
from utility.time_dependent_no.pcno_runtime import (
    autocast_context,
    build_checkpoint_model,
    checkpoint_model_node_type_input,
    forward_sample,
    load_checkpoint,
    load_checkpoint_payload,
    select_device,
    synchronize,
    timed_model_call,
)

GOLDEN_PHYSICAL_OUTPUT = np.asarray(
    [
        [1.0113818645, 0.3576550186, -0.0397413671, 2.5700204372],
        [1.0364042521, 0.3822389245, -0.0405711792, 2.5898470879],
        [1.0214073658, 0.3612489402, -0.0299054012, 2.5703670979],
        [1.0464519262, 0.3859620392, -0.0304360874, 2.5902380943],
    ],
    dtype=np.float32,
)
GOLDEN_ALL_NORMAL_OUTPUT = np.asarray(
    [
        [1.0114350319, 0.3576509356, -0.0398596935, 2.5700271130],
        [1.0364141464, 0.3822233379, -0.0406324230, 2.5898437500],
        [1.0214591026, 0.3612388074, -0.0300366059, 2.5703718662],
        [1.0464557409, 0.3859447837, -0.0305147767, 2.5902330875],
    ],
    dtype=np.float32,
)


def _normalization() -> Euler2DNormalization:
    return Euler2DNormalization(
        state_mean=np.asarray([1.0, 0.4, 0.0, 2.8]),
        state_scale=np.asarray([0.2, 0.2, 0.1, 0.4]),
        residual_scale=np.asarray([0.05, 0.04, 0.03, 0.02]),
        mach_mean=1.1,
        mach_scale=0.2,
        weight_provenance="synthetic_runtime_fixture",
    )


def _deterministic_model(
    *, node_type_feature_mode: str = "one_hot"
) -> PCNOEuler2DResidual:
    model = PCNOEuler2DResidual(
        normalization=_normalization(),
        k_max=1,
        domain_lengths=(2.0, 1.0),
        layers=(8, 8, 8),
        fc_dim=8,
        nmeasures=1,
        zero_initialize=False,
        node_type_feature_mode=node_type_feature_mode,
    )
    with torch.no_grad():
        for parameter_index, parameter in enumerate(model.parameters(), start=1):
            values = torch.arange(parameter.numel(), dtype=parameter.dtype)
            values = ((values.remainder(19) - 9.0) * 2.0e-2) + (
                parameter_index * 1.0e-3
            )
            parameter.copy_(values.reshape_as(parameter))
    return model


def _synthetic_checkpoint(
    *,
    node_type_mode: str = "physical",
    node_type_feature_mode: str = "one_hot",
) -> dict[str, object]:
    model = _deterministic_model(node_type_feature_mode=node_type_feature_mode)
    return {
        "checkpoint_schema_version": 4,
        "model_state": model.state_dict(),
        "model_config": model.model_config(),
        "normalization": _normalization().to_dict(),
        "normalization_digest": "synthetic-normalization",
        "data_manifest_digest": "synthetic-data",
        "data_contract": {
            "dataset": "synthetic_runtime_fixture",
            "step_stride": 1,
        },
        "step_stride": 1,
        "config_digest": "synthetic-config",
        "boundary_mode": "model_all_nodes",
        "raw_recurrence": True,
        "inference_interventions": {
            "future_reference_boundary_values": False,
            "clipping": False,
            "smoothing": False,
        },
        "model_node_type_input": node_type_mode,
        "training_args": {
            "amp": "none",
            "model_node_type_input": node_type_mode,
        },
    }


def test_runtime_reconstructs_literal_zero_type_channels() -> None:
    checkpoint = _synthetic_checkpoint(
        node_type_feature_mode=NODE_TYPE_FEATURE_CONSTANT_ZERO
    )
    model, _ = build_checkpoint_model(
        checkpoint,
        torch.device("cpu"),
        model_node_type_input="physical",
    )
    sample, current = _synthetic_sample()
    baseline = forward_sample(model, sample, current)
    changed = dict(sample)
    changed["node_type"] = (sample["node_type"] + 1).remainder(4)
    assert model.node_type_feature_mode == NODE_TYPE_FEATURE_CONSTANT_ZERO
    assert torch.equal(baseline, forward_sample(model, changed, current))


def _synthetic_sample() -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    nodes = torch.tensor(
        [[[0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]]],
        dtype=torch.float32,
    )
    directed_edges = torch.tensor(
        [
            [
                [0, 1],
                [0, 2],
                [1, 0],
                [1, 3],
                [2, 0],
                [2, 3],
                [3, 1],
                [3, 2],
            ]
        ],
        dtype=torch.int64,
    )
    edge_gradient_weights = torch.tensor(
        [
            [
                [2.0, 0.0],
                [0.0, 2.0],
                [-2.0, 0.0],
                [0.0, 2.0],
                [0.0, -2.0],
                [2.0, 0.0],
                [0.0, -2.0],
                [-2.0, 0.0],
            ]
        ],
        dtype=torch.float32,
    )
    sample = {
        "node_mask": torch.ones((1, 4, 1), dtype=torch.float32),
        "nodes": nodes,
        "node_weights": torch.full((1, 4, 1), 0.25, dtype=torch.float32),
        "node_rhos": torch.ones((1, 4, 1), dtype=torch.float32),
        "directed_edges": directed_edges,
        "edge_gradient_weights": edge_gradient_weights,
        "node_type": torch.tensor([[3, 1, 2, 0]], dtype=torch.int64),
        "mach": torch.tensor([1.1], dtype=torch.float32),
    }
    nodes = sample["nodes"]
    rho = 1.0 + 0.05 * nodes[..., 0:1] + 0.02 * nodes[..., 1:2]
    velocity_x = 0.35 + 0.03 * nodes[..., 0:1]
    velocity_y = -0.04 + 0.02 * nodes[..., 1:2]
    pressure = 1.0 + 0.01 * nodes[..., 0:1]
    energy = pressure / 0.4 + 0.5 * rho * (velocity_x.square() + velocity_y.square())
    current = torch.cat(
        (rho, rho * velocity_x, rho * velocity_y, energy),
        dim=-1,
    )
    return sample, current


def _write_checkpoint(
    tmp_path: Path,
    *,
    node_type_mode: str = "physical",
) -> Path:
    path = tmp_path / f"{node_type_mode}.pt"
    torch.save(_synthetic_checkpoint(node_type_mode=node_type_mode), path)
    return path


def test_physical_runtime_paths_match_frozen_golden(tmp_path: Path) -> None:
    path = _write_checkpoint(tmp_path)
    sample, current = _synthetic_sample()
    device = torch.device("cpu")

    resolution_checkpoint = load_resolution_checkpoint(path)
    resolution_model, _ = build_resolution_checkpoint_model(
        resolution_checkpoint, device
    )
    resolution_output, timing = timed_model_call(
        resolution_model,
        sample,
        current,
        device=device,
        amp="none",
        repeats=2,
    )

    bump_model = bump_evaluator.build_model(
        bump_evaluator.load_checkpoint(path), device
    )
    bump_output = bump_evaluator.model_call(bump_model, sample, current)
    dynamic_model = dynamic_evaluator.build_model(
        dynamic_evaluator.load_checkpoint(path), device
    )
    dynamic_output = dynamic_evaluator.model_call(
        dynamic_model,
        sample,
        current,
        device=device,
        amp="none",
    )
    trainer_output, _, _ = trainer.contract_forward_sample(
        resolution_model,
        sample,
        current,
        boundary_policy=None,
    )
    runtime_checkpoint = load_checkpoint(path)
    runtime_model, runtime_normalization = build_checkpoint_model(
        runtime_checkpoint,
        device,
        model_node_type_input="physical",
    )
    runtime_forward = forward_sample(runtime_model, sample, current)
    runtime_output, runtime_timing = timed_model_call(
        runtime_model,
        sample,
        current,
        device=device,
        amp="none",
        repeats=2,
    )

    np.testing.assert_allclose(
        resolution_output,
        bump_output[0].detach().numpy(),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        resolution_output,
        dynamic_output[0].detach().numpy(),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        resolution_output,
        trainer_output[0].detach().numpy(),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        resolution_output,
        runtime_forward[0].detach().numpy(),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        resolution_output,
        runtime_output,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        runtime_normalization.residual_scale,
        _normalization().residual_scale,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        resolution_output,
        GOLDEN_PHYSICAL_OUTPUT,
        rtol=0.0,
        atol=0.0,
    )
    assert timing["repeat_max_abs"] == pytest.approx(0.0)
    assert runtime_timing["repeat_max_abs"] == pytest.approx(0.0)
    assert runtime_output.dtype == np.float64


def test_homogeneous_batch_matches_sequential_and_is_order_isolated(
    tmp_path: Path,
) -> None:
    path = _write_checkpoint(tmp_path)
    sample, first = _synthetic_sample()
    second = first + torch.tensor([0.01, -0.02, 0.015, 0.03])
    model, _ = build_resolution_checkpoint_model(
        load_resolution_checkpoint(path), torch.device("cpu")
    )
    states = (first[0].numpy(), second[0].numpy())
    sequential = np.stack(
        [
            predict_resolution_sample(
                model,
                sample,
                state,
                device=torch.device("cpu"),
                amp="none",
                repeats=1,
            )[0]
            for state in states
        ]
    )
    batched, timing = predict_resolution_batch(
        model,
        sample,
        states,
        device=torch.device("cpu"),
        amp="none",
        repeats=2,
    )
    reversed_batch, _ = predict_resolution_batch(
        model,
        sample,
        states[::-1],
        device=torch.device("cpu"),
        amp="none",
        repeats=1,
    )

    np.testing.assert_allclose(batched, sequential, rtol=0.0, atol=2.0e-7)
    np.testing.assert_allclose(reversed_batch[::-1], batched, rtol=0.0, atol=2.0e-7)
    assert timing["repeat_max_abs"] == pytest.approx(0.0)


def test_node_type_behavior_matches_frozen_golden(tmp_path: Path) -> None:
    path = _write_checkpoint(tmp_path, node_type_mode="all_normal")
    sample, current = _synthetic_sample()
    original_node_type = sample["node_type"].clone()
    device = torch.device("cpu")

    checkpoint = load_resolution_checkpoint(path)
    all_normal_model, _ = build_resolution_checkpoint_model(checkpoint, device)
    resolution_output, _ = timed_model_call(
        all_normal_model,
        sample,
        current,
        device=device,
        amp="none",
        repeats=1,
    )
    trainer_output, _, _ = trainer.contract_forward_sample(
        all_normal_model,
        sample,
        current,
        boundary_policy=None,
    )
    runtime_model, _ = build_checkpoint_model(
        load_checkpoint(path),
        device,
        model_node_type_input="all_normal",
    )
    runtime_output = forward_sample(runtime_model, sample, current)

    historical_bump_model = bump_evaluator.build_model(
        bump_evaluator.load_checkpoint(path), device
    )
    historical_bump_output = bump_evaluator.model_call(
        historical_bump_model, sample, current
    )
    historical_dynamic_model = dynamic_evaluator.build_model(
        dynamic_evaluator.load_checkpoint(path), device
    )
    historical_dynamic_output = dynamic_evaluator.model_call(
        historical_dynamic_model,
        sample,
        current,
        device=device,
        amp="none",
    )

    np.testing.assert_allclose(
        resolution_output,
        trainer_output[0].detach().numpy(),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        resolution_output,
        runtime_output[0].detach().numpy(),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        resolution_output,
        GOLDEN_ALL_NORMAL_OUTPUT,
        rtol=0.0,
        atol=0.0,
    )
    assert not np.array_equal(
        resolution_output,
        historical_bump_output[0].detach().numpy(),
    )
    np.testing.assert_allclose(
        historical_bump_output.detach().numpy(),
        historical_dynamic_output.detach().numpy(),
        rtol=0.0,
        atol=0.0,
    )
    assert torch.equal(sample["node_type"], original_node_type)


def test_runtime_base_contract_is_policy_neutral_and_strict(
    tmp_path: Path,
) -> None:
    path = _write_checkpoint(tmp_path)
    checkpoint = load_checkpoint(path)

    assert checkpoint_model_node_type_input(checkpoint) == "physical"
    legacy = dict(checkpoint)
    legacy.pop("model_node_type_input")
    legacy["training_args"] = {}
    assert checkpoint_model_node_type_input(legacy) == "physical"
    from_training_args = dict(legacy)
    from_training_args["training_args"] = {"model_node_type_input": "all_normal"}
    assert checkpoint_model_node_type_input(from_training_args) == "all_normal"

    disagreement = dict(checkpoint)
    disagreement["model_node_type_input"] = "physical"
    disagreement["training_args"] = {"model_node_type_input": "all_normal"}
    with pytest.raises(ValueError, match="declarations disagree"):
        checkpoint_model_node_type_input(disagreement)

    unsupported = dict(checkpoint)
    unsupported["model_node_type_input"] = "boundary_only"
    unsupported["training_args"] = {}
    with pytest.raises(ValueError, match="unsupported checkpoint node-type input"):
        checkpoint_model_node_type_input(unsupported)

    missing_state = dict(checkpoint)
    missing_state["model_state"] = dict(checkpoint["model_state"])
    missing_state["model_state"].pop(next(iter(missing_state["model_state"])))
    with pytest.raises(RuntimeError):
        build_checkpoint_model(
            missing_state,
            torch.device("cpu"),
            model_node_type_input="physical",
        )
    with pytest.raises(ValueError, match="unsupported model node-type input"):
        build_checkpoint_model(
            checkpoint,
            torch.device("cpu"),
            model_node_type_input="boundary_only",
        )

    wrong_schema = tmp_path / "wrong_schema.pt"
    torch.save({"checkpoint_schema_version": 3}, wrong_schema)
    assert load_checkpoint_payload(wrong_schema) == {"checkpoint_schema_version": 3}
    with pytest.raises(ValueError, match="unsupported checkpoint schema"):
        load_checkpoint(wrong_schema)
    with pytest.raises(ValueError, match="missing required fields"):
        bump_evaluator.load_checkpoint(wrong_schema)
    with pytest.raises(ValueError, match="missing frozen contract fields"):
        load_resolution_checkpoint(wrong_schema)
    with pytest.raises(ValueError, match="missing frozen contract fields"):
        dynamic_evaluator.load_checkpoint(wrong_schema)
    nonmapping = tmp_path / "nonmapping.pt"
    torch.save(["not", "a", "checkpoint"], nonmapping)
    with pytest.raises(TypeError, match="payload must be a mapping"):
        load_checkpoint_payload(nonmapping)


def test_runtime_device_amp_and_cpu_synchronization_are_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = select_device("cpu")
    assert device == torch.device("cpu")
    with autocast_context(device, "none"):
        value = torch.ones(1)
    assert value.item() == pytest.approx(1.0)

    monkeypatch.setattr(
        torch.cuda,
        "synchronize",
        lambda *_args, **_kwargs: pytest.fail("CPU synchronize called CUDA"),
    )
    synchronize(device)

    with pytest.raises(ValueError, match="unsupported device selection"):
        select_device("gpu")
    with pytest.raises(ValueError, match="mixed-precision PCNO execution"):
        autocast_context(device, "bf16")
    with pytest.raises(ValueError, match="unsupported AMP mode"):
        autocast_context(device, "tf32")


def test_evaluator_checkpoint_policies_remain_distinct(tmp_path: Path) -> None:
    checkpoint = _synthetic_checkpoint()
    checkpoint["boundary_mode"] = "causal_nodal_physical"
    checkpoint["raw_recurrence"] = False
    checkpoint["boundary_contract"] = {"mode": "causal_nodal_physical"}
    path = tmp_path / "causal.pt"
    torch.save(checkpoint, path)

    assert bump_evaluator.load_checkpoint(path)["boundary_mode"] == (
        "causal_nodal_physical"
    )
    with pytest.raises(ValueError, match="model_all_nodes"):
        load_resolution_checkpoint(path)
    with pytest.raises(ValueError, match="model_all_nodes"):
        dynamic_evaluator.load_checkpoint(path)
