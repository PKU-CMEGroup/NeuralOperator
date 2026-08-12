from __future__ import annotations

import random
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from scripts.time_dependent_no.train_realm_ignithit_ffno import (
    ACCUMULATION_STEPS,
    BEST_CHECKPOINT_SCHEMA,
    EFFECTIVE_BATCH_SIZE,
    LAST_CHECKPOINT_SCHEMA,
    SCHEDULER_TOTAL_STEPS,
    TOTAL_STEPS,
    _best_checkpoint_matches,
    _prepare_output_directory,
    _recoverable_current_best_validation,
    _validate_stop_after_step,
    build_best_checkpoint,
    build_last_checkpoint,
    build_optimizer_and_scheduler,
    direct_rollout,
    draw_step_sample,
    frozen_training_contract,
    is_strict_improvement,
    legal_adjacent_frame_pairs,
    restore_last_checkpoint,
    scaled_microbatch_loss,
    structured_state_sha256,
    summarize_validation_predictions,
    validation_steps,
)
from scripts.time_dependent_no.train_realm_ignithit_ffno import main as train_main
from utility.time_dependent_no.realm_benchmark import canonical_json_sha256
from utility.time_dependent_no.realm_ffno import grouped_next_state_mse
from utility.time_dependent_no.realm_ignithit import validate_local_open_tree


def test_frozen_contract_covers_all_adjacent_pairs_and_exact_budget() -> None:
    pairs = legal_adjacent_frame_pairs()
    assert pairs == tuple((start, start + 1) for start in range(29))
    assert pairs[-1] == (28, 29)

    schedule = validation_steps()
    assert schedule[0] == 1
    assert schedule[1] == 50
    assert schedule[-1] == TOTAL_STEPS
    assert len(schedule) == 101

    contract = frozen_training_contract()
    assert contract["paper_faithful"] is False
    assert contract["exposure"] == "one_call_all_29_adjacent_pairs"
    assert contract["total_presentations"] == 130_000
    assert contract["effective_batch_size"] == 26
    assert contract["scheduler"]["total_steps"] == SCHEDULER_TOTAL_STEPS
    assert contract["optimizer"]["weight_decay"] == 0.0


def test_step_sampling_is_deterministic_and_state_resumable() -> None:
    first = random.Random(0)
    second = random.Random(0)
    first_rows = [draw_step_sample(first) for _ in range(12)]
    second_rows = [draw_step_sample(second) for _ in range(12)]
    assert first_rows == second_rows
    assert all(sorted(row.case_indices) == list(range(26)) for row in first_rows)
    assert all(0 <= row.frame_start <= 28 for row in first_rows)

    generator = random.Random(0)
    prefix = [draw_step_sample(generator) for _ in range(4)]
    state = generator.getstate()
    suffix = [draw_step_sample(generator) for _ in range(5)]
    resumed = random.Random(999)
    resumed.setstate(state)
    assert [draw_step_sample(resumed) for _ in range(5)] == suffix
    assert prefix + suffix == first_rows[:9]


def test_accumulated_case_loss_matches_full_effective_batch_value_and_gradient() -> (
    None
):
    torch.manual_seed(7)
    truth = torch.randn(EFFECTIVE_BATCH_SIZE, 12, 2, 2)
    full_prediction = torch.randn_like(truth, requires_grad=True)
    accumulated_prediction = full_prediction.detach().clone().requires_grad_(True)

    full_loss, _ = grouped_next_state_mse(full_prediction, truth)
    full_loss.backward()
    for case in range(EFFECTIVE_BATCH_SIZE):
        case_loss, _ = grouped_next_state_mse(
            accumulated_prediction[case : case + 1],
            truth[case : case + 1],
        )
        scaled_microbatch_loss(
            case_loss,
            effective_batch_size=EFFECTIVE_BATCH_SIZE,
        ).backward()

    assert ACCUMULATION_STEPS == EFFECTIVE_BATCH_SIZE == 26
    assert torch.allclose(full_prediction.grad, accumulated_prediction.grad, atol=1e-7)
    assert full_loss.item() == pytest.approx(
        sum(
            grouped_next_state_mse(
                accumulated_prediction[case : case + 1],
                truth[case : case + 1],
            )[0].item()
            for case in range(EFFECTIVE_BATCH_SIZE)
        )
        / EFFECTIVE_BATCH_SIZE
    )


class _RecordingIncrement(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inputs: list[torch.Tensor] = []

    def forward(
        self,
        state: torch.Tensor,
        coordinates: torch.Tensor,
    ) -> torch.Tensor:
        del coordinates
        self.inputs.append(state.detach().clone())
        return state + 1.0


def test_direct_rollout_feeds_each_returned_proposal_back_exactly() -> None:
    model = _RecordingIncrement()
    initial = torch.zeros(2, 12, 2, 2)
    coordinates = torch.zeros(1, 2, 2, 2)
    prediction = direct_rollout(model, initial, coordinates, calls=3)

    assert prediction.shape == (2, 3, 12, 2, 2)
    assert torch.equal(model.inputs[0], initial)
    assert torch.equal(model.inputs[1], prediction[:, 0])
    assert torch.equal(model.inputs[2], prediction[:, 1])
    assert torch.equal(prediction[:, 2], torch.full_like(initial, 3.0))


def test_validation_summary_is_case_first_over_five_cases() -> None:
    spatial = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    truth = spatial.view(1, 1, 1, 2, 2).expand(5, 2, 12, 2, 2).clone()
    offsets = torch.arange(5, dtype=torch.float32).view(5, 1, 1, 1, 1)
    prediction = truth + offsets
    summary = summarize_validation_predictions(
        prediction,
        truth,
        prediction,
        truth,
        case_keys=tuple(f"case-{index}" for index in range(5)),
        train_max_abs=torch.full((12,), 100.0),
    )

    expected_per_case = [4.0 * float(index**2) for index in range(5)]
    assert [row["realm_npe_mean"] for row in summary["per_case"]] == pytest.approx(
        expected_per_case
    )
    assert summary["realm_npe_mean"] == pytest.approx(np.mean(expected_per_case))
    assert summary["realm_npe_sum_source"] == pytest.approx(
        2.0 * np.mean(expected_per_case)
    )
    assert summary["decoded_correlation_case_first"] == pytest.approx(1.0)
    assert summary["case_count"] == 5
    assert summary["all_released_state_admissible"] is True
    assert summary["all_bounded_10x_train_max"] is True


def test_closed_training_tree_rejects_any_test_path(tmp_path) -> None:
    forbidden = tmp_path / "data" / "test"
    forbidden.mkdir(parents=True)
    (forbidden / "trajectory.npz").write_bytes(b"not opened")
    with pytest.raises(ValueError, match="sealed test path"):
        validate_local_open_tree(tmp_path, ())


def test_best_and_last_checkpoints_have_distinct_ownership() -> None:
    torch.manual_seed(3)
    model = nn.Linear(2, 1)
    optimizer, scheduler = build_optimizer_and_scheduler(
        model,
        scheduler_total_steps=4,
    )
    provenance = {
        "config_digest": "a" * 64,
        "input_digest": "b" * 64,
        "source_digest": "c" * 64,
        "runtime_digest": "d" * 64,
    }
    run_signature = canonical_json_sha256(provenance)
    validation = {"realm_npe_mean": 1.25}
    best = build_best_checkpoint(
        model,
        run_signature=run_signature,
        completed_step=1,
        validation=validation,
        normalizer_state={"mean": torch.zeros(12), "scale": torch.ones(12)},
        provenance=provenance,
    )
    last = build_last_checkpoint(
        model,
        optimizer,
        scheduler,
        random.Random(0),
        run_signature=run_signature,
        completed_step=1,
        best_step=1,
        best_score=1.25,
        best_model_state_sha256=best["model_state_sha256"],
        history=({"completed_step": 1, "validation": validation},),
        provenance=provenance,
    )

    assert best["schema"] == BEST_CHECKPOINT_SCHEMA
    assert last["schema"] == LAST_CHECKPOINT_SCHEMA
    assert best["resume_supported"] is False
    assert last["resume_supported"] is True
    assert "normalizer_state" in best and "optimizer_state" not in best
    assert "optimizer_state" in last and "normalizer_state" not in last
    assert "order_rng_state" in last and "scheduler_state" in last
    assert best["model_state_sha256"] == structured_state_sha256(best["model_state"])
    assert best["normalizer_state_sha256"] == structured_state_sha256(
        best["normalizer_state"]
    )
    assert last["model_state_sha256"] == structured_state_sha256(last["model_state"])
    assert _best_checkpoint_matches(
        best,
        run_signature=run_signature,
        best_step=1,
        best_score=1.25,
        best_model_state_sha256=last["best_model_state_sha256"],
        normalizer_state_sha256=best["normalizer_state_sha256"],
        provenance=provenance,
    )
    damaged_best = dict(best)
    damaged_best["model_state"] = {
        name: value.clone() for name, value in best["model_state"].items()
    }
    damaged_best["model_state"]["weight"][0, 0] += 1.0
    assert not _best_checkpoint_matches(
        damaged_best,
        run_signature=run_signature,
        best_step=1,
        best_score=1.25,
        best_model_state_sha256=last["best_model_state_sha256"],
        normalizer_state_sha256=best["normalizer_state_sha256"],
        provenance=provenance,
    )
    assert (
        _recoverable_current_best_validation(
            last["history"],
            completed_step=1,
            best_step=1,
            best_score=1.25,
            current_model_state_sha256=last["model_state_sha256"],
            best_model_state_sha256=last["best_model_state_sha256"],
        )
        == validation
    )
    assert (
        _recoverable_current_best_validation(
            last["history"],
            completed_step=2,
            best_step=1,
            best_score=1.25,
            current_model_state_sha256=last["model_state_sha256"],
            best_model_state_sha256=last["best_model_state_sha256"],
        )
        is None
    )
    assert is_strict_improvement(1.0, 1.1) is True
    assert is_strict_improvement(1.0, 1.0) is False


def _tiny_updates(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    order_generator: random.Random,
    *,
    start: int,
    stop: int,
) -> None:
    values = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
    for _ in range(start, stop):
        sample = draw_step_sample(order_generator, case_count=4, frame_count=3)
        optimizer.zero_grad(set_to_none=True)
        for case_index in sample.case_indices:
            prediction = model(values[case_index : case_index + 1])
            target = 2.0 * values[case_index : case_index + 1] + sample.frame_start
            loss = (prediction - target).square().mean()
            scaled_microbatch_loss(loss, effective_batch_size=4).backward()
        optimizer.step()
        scheduler.step()


def _assert_nested_equal(left: object, right: object) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert torch.equal(left, right)
    elif isinstance(left, Mapping):
        assert isinstance(right, Mapping)
        assert set(left) == set(right)
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert isinstance(right, type(left))
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right, strict=True):
            _assert_nested_equal(left_item, right_item)
    else:
        assert left == right


def test_last_checkpoint_resume_matches_uninterrupted_cpu_updates_exactly() -> None:
    torch.manual_seed(11)
    initial = nn.Linear(1, 1)
    initial_state = {
        name: value.detach().clone() for name, value in initial.state_dict().items()
    }

    full = nn.Linear(1, 1)
    full.load_state_dict(initial_state)
    full_optimizer, full_scheduler = build_optimizer_and_scheduler(
        full,
        scheduler_total_steps=7,
    )
    full_order = random.Random(0)
    _tiny_updates(
        full,
        full_optimizer,
        full_scheduler,
        full_order,
        start=0,
        stop=6,
    )

    partial = nn.Linear(1, 1)
    partial.load_state_dict(initial_state)
    partial_optimizer, partial_scheduler = build_optimizer_and_scheduler(
        partial,
        scheduler_total_steps=7,
    )
    partial_order = random.Random(0)
    _tiny_updates(
        partial,
        partial_optimizer,
        partial_scheduler,
        partial_order,
        start=0,
        stop=3,
    )
    provenance = {"config_digest": "a" * 64}
    run_signature = canonical_json_sha256(provenance)
    checkpoint = build_last_checkpoint(
        partial,
        partial_optimizer,
        partial_scheduler,
        partial_order,
        run_signature=run_signature,
        completed_step=3,
        best_step=1,
        best_score=2.0,
        best_model_state_sha256="0" * 64,
        history=({"completed_step": 1},),
        provenance=provenance,
    )

    resumed = nn.Linear(1, 1)
    resumed_optimizer, resumed_scheduler = build_optimizer_and_scheduler(
        resumed,
        scheduler_total_steps=7,
    )
    resumed_order = random.Random(999)
    completed, best_step, best_score, best_digest, history = restore_last_checkpoint(
        checkpoint,
        resumed,
        resumed_optimizer,
        resumed_scheduler,
        resumed_order,
        expected_run_signature=run_signature,
    )
    assert (completed, best_step, best_score, best_digest, history) == (
        3,
        1,
        2.0,
        "0" * 64,
        [{"completed_step": 1}],
    )
    _tiny_updates(
        resumed,
        resumed_optimizer,
        resumed_scheduler,
        resumed_order,
        start=3,
        stop=6,
    )

    _assert_nested_equal(full.state_dict(), resumed.state_dict())
    _assert_nested_equal(full_optimizer.state_dict(), resumed_optimizer.state_dict())
    _assert_nested_equal(full_scheduler.state_dict(), resumed_scheduler.state_dict())
    assert draw_step_sample(
        full_order, case_count=4, frame_count=3
    ) == draw_step_sample(
        resumed_order,
        case_count=4,
        frame_count=3,
    )


def test_output_isolation_and_registered_stop_guards(tmp_path: Path) -> None:
    data_root = tmp_path / "data-root"
    data_root.mkdir()
    with pytest.raises(ValueError, match="outside the exact data root"):
        _prepare_output_directory(data_root / "runs" / "attempt", data_root, None)

    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "foreign.txt").write_text("preserve", encoding="utf-8")
    with pytest.raises(ValueError, match="absent or empty"):
        _prepare_output_directory(occupied, data_root, None)

    resume_dir = tmp_path / "resume"
    resume_dir.mkdir()
    last_path = resume_dir / "last.pt"
    last_path.write_bytes(b"fixture")
    _prepare_output_directory(resume_dir, data_root, last_path)
    assert not (resume_dir / "best.pt").exists()

    assert _validate_stop_after_step(None) == TOTAL_STEPS
    assert _validate_stop_after_step(1) == 1
    assert _validate_stop_after_step(50) == 50
    for invalid in (0, 49, TOTAL_STEPS, TOTAL_STEPS + 1):
        with pytest.raises(ValueError, match="registered pre-final"):
            _validate_stop_after_step(invalid)


def test_training_cli_help_is_safe(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        train_main(["--help"])
    assert exit_info.value.code == 0
    rendered = capsys.readouterr().out
    assert "--manifest" in rendered
    assert "--data-root" in rendered
    assert "--normalizer-arrays" in rendered
    assert "--output-dir" in rendered
    assert "--resume-checkpoint" in rendered
    assert "--stop-after-step" in rendered
