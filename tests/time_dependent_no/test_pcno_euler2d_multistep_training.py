from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

import scripts.time_dependent_no.train_pcno_euler2d_residual as training
from tests.time_dependent_no.test_pcno_euler2d_residual import (
    _prepare_boundary_synthetic_shards,
)
from utility.time_dependent_no.pcno_euler2d import PCNOEuler2DShardStore


class _ScaleStep(torch.nn.Module):
    gamma = 1.4

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("state_scale", torch.ones(4))
        self.scale = torch.nn.Parameter(torch.tensor(1.01))


def test_multistep_terms_retain_tail_anchor_and_attach_first_call_gradient(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = _prepare_boundary_synthetic_shards(tmp_path)
    store = PCNOEuler2DShardStore(data_dir)
    key = store.keys[0]
    model = _ScaleStep()

    def scaled_step(
        model: _ScaleStep,
        sample: dict[str, torch.Tensor],
        current: torch.Tensor,
        *,
        boundary_policy: object,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del sample, boundary_policy
        prediction = current * model.scale
        return prediction, prediction, current

    monkeypatch.setattr(training, "contract_forward_sample", scaled_step)
    initial_sample = store.tensor_batch(
        key,
        [0, 2],
        step_stride=1,
        device=torch.device("cpu"),
    )
    first_prediction = initial_sample["current"].clone().requires_grad_(True)

    pairs = [(key, 0), (key, 2)]
    assert (
        training.multistep_future_comparison_count(
            store,
            pairs,
            step_stride=1,
            rollout_steps=3,
        )
        == 2
    )
    terms = training.differentiable_multistep_normal_terms(
        model,
        store,
        key=key,
        time_indices=[0, 2],
        first_prediction=first_prediction,
        step_stride=1,
        rollout_steps=3,
        device=torch.device("cpu"),
        boundary_policy=None,
        recurrent_input=training.ATTACHED_PREDICTION_MULTISTEP_INPUT,
    )

    assert terms["comparisons"] == 2
    assert terms["admissible_comparisons"] == 2
    terms["loss_sum"].backward()
    assert first_prediction.grad is not None
    assert float(first_prediction.grad.abs().sum()) > 0.0
    assert model.scale.grad is not None

    model.scale.grad = None
    teacher_first_prediction = initial_sample["current"].clone().requires_grad_(True)
    teacher_terms = training.differentiable_multistep_normal_terms(
        model,
        store,
        key=key,
        time_indices=[0, 2],
        first_prediction=teacher_first_prediction,
        step_stride=1,
        rollout_steps=3,
        device=torch.device("cpu"),
        boundary_policy=None,
        recurrent_input=training.PROJECTED_TEACHER_MULTISTEP_INPUT,
    )
    teacher_terms["loss_sum"].backward()
    assert teacher_first_prediction.grad is None
    assert model.scale.grad is not None
    assert float(model.scale.grad.abs()) > 0.0


def test_fixed_horizon_blocks_are_deterministic_and_call_complete(
    tmp_path: Path,
) -> None:
    data_dir = _prepare_boundary_synthetic_shards(tmp_path)
    store = PCNOEuler2DShardStore(data_dir)
    pairs = training.fixed_horizon_block_presentations(
        store,
        store.keys,
        step_stride=1,
        rollout_steps=2,
        count=2,
        block_size=2,
        seed=19,
    )
    repeated = training.fixed_horizon_block_presentations(
        store,
        store.keys,
        step_stride=1,
        rollout_steps=2,
        count=2,
        block_size=2,
        seed=19,
    )

    assert pairs == repeated
    assert training.presentation_stream_sha256(pairs) == (
        training.presentation_stream_sha256(repeated)
    )
    assert len(pairs) == 2
    assert len({key for key, _ in pairs}) == 1
    assert len({time_index for _, time_index in pairs}) == 2
    assert training.multistep_future_comparison_count(
        store,
        pairs,
        step_stride=1,
        rollout_steps=2,
    ) == len(pairs)
    assert training.homogeneous_optimizer_step_count(pairs, batch_size=2) == 1


@pytest.mark.parametrize(
    ("recurrent_input", "first_call_gradient", "uses_future_reference_input"),
    (
        (
            training.ATTACHED_PREDICTION_MULTISTEP_INPUT,
            "attached",
            False,
        ),
        (
            training.PROJECTED_TEACHER_MULTISTEP_INPUT,
            "not_connected_projected_teacher_control",
            True,
        ),
    ),
)
def test_cpu_multistep_training_records_exact_exposure_contract(
    tmp_path: Path,
    recurrent_input: str,
    first_call_gradient: str,
    uses_future_reference_input: bool,
) -> None:
    data_dir = _prepare_boundary_synthetic_shards(tmp_path)
    output_dir = tmp_path / f"multistep_{recurrent_input}"

    training.main(
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
            "fixed_horizon_blocks",
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
            "--multistep-loss-steps",
            "2",
            "--multistep-loss-weight",
            "1",
            "--multistep-recurrent-input",
            recurrent_input,
            "--boundary-mode",
            "minimum_change_nodal_physical",
            "--primary-objective",
            "learned_dofs_closed",
            "--rollout-every",
            "1",
            "--rollout-val-count",
            "1",
            "--rollout-steps",
            "1",
            "--rollout-checkpoints",
            "1",
            "--selection-mode",
            "interior_rollout",
            "--selection-short-horizon",
            "1",
            "--device",
            "cpu",
            "--amp",
            "none",
        ]
    )

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    contract = summary["differentiable_multistep"]
    train = summary["last_train"]
    assert contract["enabled"] is True
    assert contract["rollout_steps"] == 2
    assert contract["future_loss_weight"] == 1.0
    assert contract["recurrent_input"] == recurrent_input
    assert contract["first_call_gradient"] == first_call_gradient
    assert contract["future_reference_as_model_input"] is uses_future_reference_input
    assert contract["future_node_population"] == "normal_nodes_only"
    assert contract["future_coverage_fraction"] == 1.0
    assert contract["clean_one_step_coefficient"] + contract[
        "realized_future_coefficient"
    ] == pytest.approx(1.0)
    assert (
        train["multistep_future_comparisons"]
        == contract["future_comparisons_per_epoch"]
    )
    assert train["model_calls"] == contract["model_calls_per_epoch"]
    assert train["model_calls"] == 4
    assert train["optimizer_steps"] == 1
    assert train["multistep_loss"] is not None
    assert train["multistep_first_call_gradient"] == first_call_gradient
    assert train["multistep_recurrent_input"] == recurrent_input
    assert train["parameters_finite"] is True
    exposure = summary["exposure_contract"]
    assert exposure["presentation_mode"] == "fixed_horizon_blocks"
    assert exposure["fixed_horizon_block_size"] == 2
    assert exposure["fixed_horizon_block_count"] == 1
    assert len(exposure["presentation_stream_sha256"]) == 64
    selection = summary["checkpoint_selection_contract"]
    assert selection["mode"] == "interior_rollout"
    assert selection["boundary_reference_error_is_primary"] is False

    checkpoint = torch.load(
        output_dir / "best.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert checkpoint["differentiable_multistep"] == contract
    assert checkpoint["checkpoint_selection_contract"] == selection


def test_interior_selection_prioritizes_normal_rollout_over_boundary_trace() -> None:
    rollout = {
        "completion_rate": 1.0,
        "mean_survival_fraction": 1.0,
        "mean_selection_relative_l2": 0.08,
        "mean_endpoint_relative_l2": {"20": 0.07},
        "mean_final_normal_relative_l2": 0.03,
        "mean_endpoint_normal_relative_l2": {"20": 0.02},
        "parity": None,
    }
    one_step = {
        "relative_l2": 0.06,
        "normal_relative_l2": 0.004,
        "all_relative_l2": 0.06,
    }
    selected = training.selection_tuple(
        rollout,
        one_step,
        mode=training.INTERIOR_SELECTION_MODE,
        short_horizon=20,
    )
    assert selected == pytest.approx(
        (1.0, 1.0, 1.0, -0.03, -0.02, -0.004, -0.08, -0.06)
    )


def test_multistep_training_rejects_nondeterministic_presentations(
    tmp_path: Path,
) -> None:
    args = training.parse_args(
        [
            "--data-dir",
            str(tmp_path / "data"),
            "--output-dir",
            str(tmp_path / "output"),
            "--multistep-loss-steps",
            "2",
            "--multistep-loss-weight",
            "1",
            "--device",
            "cpu",
            "--amp",
            "none",
        ]
    )
    with pytest.raises(
        ValueError,
        match="requires full coverage or fixed horizon blocks",
    ):
        training.validate_args(args, torch.device("cpu"))


def test_legacy_boundary_metadata_compatibility_is_narrow_and_auditable() -> None:
    current = {
        "schema": "pcno_euler2d_boundary_contract_v1",
        "mode": training.CAUSAL_BOUNDARY_MODE,
        "state_loss_mask": "normal_nodes_only",
        "teacher_input_closure": True,
        "proposal_closure": True,
        "recurrence_closure": True,
        "future_reference_boundary_values": False,
        "policy_set_digest": "same-policy-set",
        "policy_digests": {"0": "same-policy"},
        "reference_endpoint_boundary_primitive_rms_by_component": [
            0.1,
            0.2,
            0.3,
            0.4,
        ],
        "outflow_characteristic_treatment": (
            "current-interior nodal extrapolation"
        ),
        "physical_conservation_claim": False,
        "projection_metric": None,
        "training_objective": {
            "primary_kind": training.NORMAL_CLOSED_PRIMARY_OBJECTIVE,
            "auxiliary": training.NO_BOUNDARY_AUXILIARY,
        },
    }
    saved = {
        key: value
        for key, value in current.items()
        if key
        not in {
            "outflow_characteristic_treatment",
            "physical_conservation_claim",
            "projection_metric",
            "training_objective",
        }
    }
    saved["reference_endpoint_boundary_primitive_rms_by_component"] = [
        0.1,
        0.2,
        0.3,
        0.400000001,
    ]

    compatible = training.compare_initialized_boundary_contracts(
        saved,
        current,
        allow_legacy_metadata=True,
    )
    assert compatible["compatible"] is True
    assert compatible["comparison"] == "legacy_metadata_compatible"
    assert {item["field"] for item in compatible["accepted_differences"]} == {
        "outflow_characteristic_treatment",
        "physical_conservation_claim",
        "projection_metric",
        "training_objective",
        "reference_endpoint_boundary_primitive_rms_by_component",
    }

    strict = training.compare_initialized_boundary_contracts(
        saved,
        current,
        allow_legacy_metadata=False,
    )
    assert strict["compatible"] is False

    policy_drift = dict(saved)
    policy_drift["policy_set_digest"] = "different-policy-set"
    rejected = training.compare_initialized_boundary_contracts(
        policy_drift,
        current,
        allow_legacy_metadata=True,
    )
    assert rejected["compatible"] is False
