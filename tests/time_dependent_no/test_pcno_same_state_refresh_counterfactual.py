from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.time_dependent_no import (
    evaluate_pcno_same_state_refresh_counterfactual as evaluator,
)
from utility.time_dependent_no.pcno_cross_resolution_teacher_forced import (
    verify_payload_sha256,
)
from utility.time_dependent_no.pcno_same_state_refresh_counterfactual import (
    PAIRED_CALL_ROLES,
    BranchCandidate,
    LogicalCallRole,
    OrderedRolePredictor,
    array_sha256,
    build_same_state_branch_pair,
    score_same_state_branch_pair,
)


def _states() -> tuple[np.ndarray, np.ndarray]:
    accepted = np.arange(12, dtype=np.float64).reshape(4, 3) / 10.0 + 1.0
    shadow = accepted - 0.2
    return accepted, shadow


def _builders(
    *,
    exact_shift: float = 0.1,
    coast_shift: float = -0.1,
):
    def exact(
        accepted: np.ndarray,
        shadow: np.ndarray,
        shared: np.ndarray,
        predict: OrderedRolePredictor,
    ) -> BranchCandidate:
        del accepted, shadow
        predict("accepted_native", shared)
        predict("accepted_fine", shared)
        return BranchCandidate(
            next_native_state=shared + exact_shift,
        )

    def coast(
        accepted: np.ndarray,
        shadow: np.ndarray,
        shared: np.ndarray,
    ) -> BranchCandidate:
        del accepted, shadow
        return BranchCandidate(
            next_native_state=shared + coast_shift,
        )

    return exact, coast


def test_pair_shares_immutable_inputs_and_one_shadow_prediction() -> None:
    accepted, shadow = _states()
    accepted_before = np.array(accepted, copy=True)
    shadow_before = np.array(shadow, copy=True)
    prediction_calls: list[LogicalCallRole] = []
    branch_inputs: dict[str, tuple[str, str, str, bool]] = {}

    def predictor(role: LogicalCallRole, value: np.ndarray) -> np.ndarray:
        prediction_calls.append(role)
        assert not value.flags.writeable
        return value + 0.05

    def record_inputs(
        name: str,
        accepted_input: np.ndarray,
        shadow_input: np.ndarray,
        shared_input: np.ndarray,
    ) -> None:
        branch_inputs[name] = (
            array_sha256(accepted_input),
            array_sha256(shadow_input),
            array_sha256(shared_input),
            not any(
                item.flags.writeable
                for item in (accepted_input, shadow_input, shared_input)
            ),
        )

    def coast_builder(
        accepted_input: np.ndarray,
        shadow_input: np.ndarray,
        shared_input: np.ndarray,
    ) -> BranchCandidate:
        record_inputs("coast", accepted_input, shadow_input, shared_input)
        return BranchCandidate(shared_input - 0.1)

    def exact_builder(
        accepted_input: np.ndarray,
        shadow_input: np.ndarray,
        shared_input: np.ndarray,
        predict: OrderedRolePredictor,
    ) -> BranchCandidate:
        record_inputs("exact", accepted_input, shadow_input, shared_input)
        predict("accepted_native", accepted_input)
        predict("accepted_fine", accepted_input)
        return BranchCandidate(shared_input + 0.1)

    pair = build_same_state_branch_pair(
        accepted,
        shadow,
        master_route="exact",
        model_predictor=predictor,
        exact_builder=exact_builder,
        coast_builder=coast_builder,
    )

    assert tuple(prediction_calls) == PAIRED_CALL_ROLES
    assert pair.logical_call_roles == PAIRED_CALL_ROLES
    assert branch_inputs["exact"] == branch_inputs["coast"]
    assert branch_inputs["exact"][3]
    assert pair.accepted_input_sha256 == array_sha256(accepted_before)
    assert pair.shadow_input_sha256 == array_sha256(shadow_before)
    assert np.array_equal(accepted, accepted_before)
    assert np.array_equal(shadow, shadow_before)
    assert np.array_equal(pair.master_next_native_state, pair.exact_next_native_state)
    assert np.array_equal(
        pair.next_shadow_native_state,
        pair.shared_shadow_prediction,
    )
    assert pair.maximum_input_mutation_abs == 0.0
    assert pair.maximum_master_identity_abs == 0.0
    assert pair.maximum_shadow_identity_abs == 0.0
    assert not pair.master_next_native_state.flags.writeable


def test_branch_inputs_are_read_only() -> None:
    accepted, shadow = _states()
    exact, _ = _builders()

    def mutating_coast(
        accepted_input: np.ndarray,
        shadow_input: np.ndarray,
        shared_input: np.ndarray,
    ) -> BranchCandidate:
        del shadow_input
        accepted_input[0, 0] = 10.0
        return BranchCandidate(shared_input)

    with pytest.raises(ValueError, match="read-only"):
        build_same_state_branch_pair(
            accepted,
            shadow,
            master_route="exact",
            model_predictor=lambda role, value: value,
            exact_builder=exact,
            coast_builder=mutating_coast,
        )


@pytest.mark.parametrize(
    ("mode", "match"),
    [
        ("missing", "incomplete"),
        ("wrong_order", "must be accepted_native"),
        ("extra", "inventory exceeded"),
    ],
)
def test_pair_rejects_wrong_logical_call_inventory(
    mode: str,
    match: str,
) -> None:
    accepted, shadow = _states()

    def exact(
        accepted_input: np.ndarray,
        shadow_input: np.ndarray,
        shared_input: np.ndarray,
        predict: OrderedRolePredictor,
    ) -> BranchCandidate:
        del accepted_input, shadow_input
        if mode == "wrong_order":
            predict("accepted_fine", shared_input)
        if mode != "missing":
            predict("accepted_native", shared_input)
            predict("accepted_fine", shared_input)
        if mode == "extra":
            predict("accepted_fine", shared_input)
        return BranchCandidate(shared_input)

    def coast(
        accepted_input: np.ndarray,
        shadow_input: np.ndarray,
        shared_input: np.ndarray,
    ) -> BranchCandidate:
        del accepted_input, shadow_input
        return BranchCandidate(shared_input)

    with pytest.raises(ValueError, match=match):
        build_same_state_branch_pair(
            accepted,
            shadow,
            master_route="exact",
            model_predictor=lambda role, value: value,
            exact_builder=exact,
            coast_builder=coast,
        )


@pytest.mark.parametrize(
    "bad_value",
    [
        np.ones(3),
        np.asarray([[1.0, np.nan], [2.0, 3.0]]),
        np.empty((0, 3)),
    ],
)
def test_pair_rejects_invalid_state(bad_value: np.ndarray) -> None:
    accepted, shadow = _states()
    exact, coast = _builders()
    with pytest.raises(ValueError):
        build_same_state_branch_pair(
            bad_value,
            shadow,
            master_route="exact",
            model_predictor=lambda role, value: value,
            exact_builder=exact,
            coast_builder=coast,
        )
    with pytest.raises(ValueError):
        build_same_state_branch_pair(
            accepted,
            bad_value,
            master_route="exact",
            model_predictor=lambda role, value: value,
            exact_builder=exact,
            coast_builder=coast,
        )


def test_pair_rejects_bad_route_and_candidate_shape() -> None:
    accepted, shadow = _states()
    exact, coast = _builders()
    with pytest.raises(ValueError, match="master_route"):
        build_same_state_branch_pair(
            accepted,
            shadow,
            master_route="raw",  # type: ignore[arg-type]
            model_predictor=lambda role, value: value,
            exact_builder=exact,
            coast_builder=coast,
        )

    def bad_exact(
        accepted_input: np.ndarray,
        shadow_input: np.ndarray,
        shared_input: np.ndarray,
        predict: OrderedRolePredictor,
    ) -> BranchCandidate:
        del accepted_input, shadow_input, shared_input
        predict("accepted_native", accepted)
        predict("accepted_fine", accepted)
        return BranchCandidate(np.ones((2, 3)))

    with pytest.raises(ValueError, match="shape"):
        build_same_state_branch_pair(
            accepted,
            shadow,
            master_route="exact",
            model_predictor=lambda role, value: value,
            exact_builder=bad_exact,
            coast_builder=coast,
        )


def test_signed_utility_matches_physical_volume_scaled_mse() -> None:
    accepted, shadow = _states()

    def exact(
        accepted_input: np.ndarray,
        shadow_input: np.ndarray,
        shared_input: np.ndarray,
        predict: OrderedRolePredictor,
    ) -> BranchCandidate:
        del accepted_input, shadow_input, shared_input
        predict("accepted_native", accepted)
        predict("accepted_fine", accepted)
        return BranchCandidate(
            np.asarray(
                [
                    [1.0, 2.0, 3.0],
                    [2.0, 3.0, 4.0],
                    [3.0, 4.0, 5.0],
                    [4.0, 5.0, 6.0],
                ]
            )
        )

    def coast(
        accepted_input: np.ndarray,
        shadow_input: np.ndarray,
        shared_input: np.ndarray,
    ) -> BranchCandidate:
        del accepted_input, shadow_input, shared_input
        return BranchCandidate(
            np.asarray(
                [
                    [2.0, 3.0, 4.0],
                    [3.0, 4.0, 5.0],
                    [4.0, 5.0, 6.0],
                    [5.0, 6.0, 7.0],
                ]
            )
        )

    pair = build_same_state_branch_pair(
        accepted,
        shadow,
        master_route="coast",
        model_predictor=lambda role, value: value,
        exact_builder=exact,
        coast_builder=coast,
    )
    reference = np.zeros_like(accepted)
    volumes = np.asarray((1.0, 2.0, 3.0, 4.0))
    scale = np.asarray((1.0, 2.0, 4.0))
    score = score_same_state_branch_pair(
        pair,
        reference,
        volumes=volumes,
        component_scale=scale,
    )

    def expected_mse(value: np.ndarray) -> float:
        node_squared = np.square(value / scale[None, :]).sum(axis=1)
        return float(np.dot(volumes, node_squared) / volumes.sum())

    exact_mse = expected_mse(pair.exact_next_native_state)
    coast_mse = expected_mse(pair.coast_next_native_state)
    assert score.exact_mse == pytest.approx(exact_mse)
    assert score.coast_mse == pytest.approx(coast_mse)
    assert score.master_mse == pytest.approx(coast_mse)
    assert score.signed_exact_refresh_utility == pytest.approx(coast_mse - exact_mse)
    assert score.relative_exact_refresh_utility == pytest.approx(
        (coast_mse - exact_mse) / coast_mse
    )
    assert score.relative_denominator_status == "resolved"
    assert score.preferred_branch == "exact"


def test_zero_utility_and_small_denominator_are_explicit() -> None:
    accepted, shadow = _states()
    exact, coast = _builders(exact_shift=0.0, coast_shift=0.0)
    pair = build_same_state_branch_pair(
        accepted,
        shadow,
        master_route="exact",
        model_predictor=lambda role, value: value,
        exact_builder=exact,
        coast_builder=coast,
    )
    score = score_same_state_branch_pair(
        pair,
        pair.exact_next_native_state,
        volumes=np.ones(accepted.shape[0]),
        component_scale=np.ones(accepted.shape[1]),
    )

    assert score.signed_exact_refresh_utility == 0.0
    assert score.relative_exact_refresh_utility is None
    assert score.relative_denominator == 0.0
    assert score.relative_denominator_status == "unresolved_small_coast_mse"
    assert score.preferred_branch == "tie"


@pytest.mark.parametrize(
    ("volumes", "scale", "match"),
    [
        (np.asarray((1.0, 1.0)), np.ones(3), "volumes"),
        (np.asarray((1.0, 1.0, 1.0, 0.0)), np.ones(3), "positive"),
        (np.ones(4), np.asarray((1.0, 0.0, 1.0)), "positive"),
        (np.ones(4), np.ones(2), "components"),
    ],
)
def test_score_rejects_invalid_physical_metric(
    volumes: np.ndarray,
    scale: np.ndarray,
    match: str,
) -> None:
    accepted, shadow = _states()
    exact, coast = _builders()
    pair = build_same_state_branch_pair(
        accepted,
        shadow,
        master_route="exact",
        model_predictor=lambda role, value: value,
        exact_builder=exact,
        coast_builder=coast,
    )
    with pytest.raises(ValueError, match=match):
        score_same_state_branch_pair(
            pair,
            np.zeros_like(accepted),
            volumes=volumes,
            component_scale=scale,
        )


def test_nonselected_counterfactual_is_not_fed_to_next_probe() -> None:
    accepted, shadow = _states()
    exact, coast = _builders(exact_shift=0.3, coast_shift=-0.4)
    first = build_same_state_branch_pair(
        accepted,
        shadow,
        master_route="coast",
        model_predictor=lambda role, value: value + 0.05,
        exact_builder=exact,
        coast_builder=coast,
    )
    second = build_same_state_branch_pair(
        first.master_next_native_state,
        first.next_shadow_native_state,
        master_route="exact",
        model_predictor=lambda role, value: value + 0.05,
        exact_builder=exact,
        coast_builder=coast,
    )

    assert second.accepted_input_sha256 == array_sha256(first.coast_next_native_state)
    assert second.accepted_input_sha256 != array_sha256(first.exact_next_native_state)
    assert second.shadow_input_sha256 == array_sha256(first.shared_shadow_prediction)


def test_synthetic_summary_closes_without_real_execution(monkeypatch) -> None:
    monkeypatch.setattr(
        evaluator,
        "OWNED_SOURCE_PATHS",
        ("tests/time_dependent_no/test_pcno_same_state_refresh_counterfactual.py",),
    )
    summary = evaluator.synthetic_summary()

    assert summary["status"] == "passed"
    assert all(summary["checks"].values())
    assert summary["call_contract"]["paired_logical_call_roles"] == list(
        PAIRED_CALL_ROLES
    )
    assert summary["call_contract"]["diagnostic_calls_per_h30_active_case"] == 90
    boundary = summary["authorization_boundary"]
    assert boundary["checkpoint_model_calls"] == 0
    assert boundary["dataset_state_arrays_loaded"] is False
    assert boundary["dataset_truth_arrays_loaded"] is False
    assert boundary["remote_access"] is False
    assert boundary["new_population_opened"] is False
    assert boundary["recurrent_controller_simulated"] is False


def test_cli_is_synthetic_only_and_writes_bound_payload(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        evaluator,
        "OWNED_SOURCE_PATHS",
        ("tests/time_dependent_no/test_pcno_same_state_refresh_counterfactual.py",),
    )
    with pytest.raises(SystemExit, match="only --synthetic"):
        evaluator.main([])

    output = tmp_path / "synthetic.json"
    assert evaluator.main(["--synthetic", "--output", str(output)]) == 0
    capsys.readouterr()
    payload = json.loads(output.read_text(encoding="utf-8"))
    verify_payload_sha256(payload)
    assert payload["status"] == "passed"
    assert payload["working_id"] == evaluator.WORKING_ID
