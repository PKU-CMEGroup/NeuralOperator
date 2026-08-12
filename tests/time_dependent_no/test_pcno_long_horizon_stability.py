from __future__ import annotations

import pytest
import torch

from scripts.time_dependent_no.evaluate_pcno_long_horizon_stability import (
    DESCRIPTIVE_SURVIVAL_REQUIRED_FIELDS,
    DIGEST_FIELDS,
    EVENT_NAMES,
    FINAL_HASH_MANIFEST_NAME,
    FINAL_HASH_MANIFEST_SCHEMA,
    PROVENANCE_FIELDS,
    TRAINING_SIDE_PROVENANCE_FIELDS,
    StabilityThresholds,
    StepTrace,
    accepted_prefix_event,
    build_final_hash_manifest,
    classify_call_events,
    common_primitive_to_native,
    decompose_common_error,
    diagnose_euler_state,
    event_survival_table,
    main,
    native_to_common_primitive,
    recovery_policy_attribution,
    summarize_stability_events,
    validate_provenance_row,
    validate_trace_sequence,
    verify_final_hash_manifest,
    weighted_error_energy_closure,
)


def _primitive(
    *, rho: float = 1.0, v1: float = 0.2, v2: float = -0.1, pressure: float = 1.0
) -> torch.Tensor:
    return torch.tensor([[[rho, v1, v2, pressure]]], dtype=torch.float64)


def _event_row(call: int, **overrides: bool | None) -> dict[str, object]:
    events: dict[str, bool | None] = {name: True for name in EVENT_NAMES}
    events.update(overrides)
    return {"call": call, "events": events}


def test_equivalent_native_states_share_a_common_primitive_view() -> None:
    primitive = _primitive(rho=1.4, v1=0.7, v2=-0.2, pressure=0.9)
    conservative = common_primitive_to_native(
        primitive,
        representation="conservative_rho_m1_m2_E",
    )

    recovered = native_to_common_primitive(
        conservative,
        representation="conservative_rho_m1_m2_E",
    )

    assert torch.allclose(recovered, primitive, rtol=0.0, atol=1.0e-14)
    assert diagnose_euler_state(
        primitive,
        representation="primitive_rho_v1_v2_p",
    )["admissible"]
    assert diagnose_euler_state(
        conservative,
        representation="conservative_rho_m1_m2_E",
    )["admissible"]


def test_primitive_zero_density_is_finite_but_inadmissible() -> None:
    metrics = diagnose_euler_state(
        _primitive(rho=0.0),
        representation="primitive_rho_v1_v2_p",
    )

    assert metrics["finite"] is True
    assert metrics["derived_fields_finite"] is True
    assert metrics["admissible"] is False
    assert metrics["primary_failure"] == "nonpositive_density"


def test_primitive_zero_pressure_is_not_misclassified_as_nonfinite() -> None:
    metrics = diagnose_euler_state(
        _primitive(pressure=0.0),
        representation="primitive_rho_v1_v2_p",
    )

    assert metrics["finite"] is True
    assert metrics["admissible"] is False
    assert metrics["primary_failure"] == "nonpositive_pressure"
    assert "nonpositive_internal_energy" in metrics["violation_fields"]


def test_conservative_negative_internal_energy_remains_distinct() -> None:
    conservative = torch.tensor([[[1.0, 2.0, 0.0, 1.0]]])
    metrics = diagnose_euler_state(
        conservative,
        representation="conservative_rho_m1_m2_E",
    )

    assert metrics["finite"] is True
    assert metrics["admissible"] is False
    assert metrics["primary_failure"] == "nonpositive_internal_energy"
    assert metrics["min_internal_energy"] == pytest.approx(-1.0)


def test_inactive_nonfinite_padding_is_ignored() -> None:
    state = torch.cat((_primitive(), _primitive()), dim=1)
    state[0, 1] = torch.nan
    metrics = diagnose_euler_state(
        state,
        representation="primitive_rho_v1_v2_p",
        node_mask=torch.tensor([[[1.0], [0.0]]]),
    )

    assert metrics["active_node_count"] == 1
    assert metrics["finite"] is True
    assert metrics["admissible"] is True


def test_first_failure_and_recovery_use_the_accepted_prefix() -> None:
    event = accepted_prefix_event(
        [False, True, True],
        requested_horizon=3,
        event_name="admissible",
    )

    assert event["first_failure_call"] == 1
    assert event["accepted_prefix_calls"] == 0
    assert event["right_censored"] is False
    assert event["recovered_after_failure"] is True


def test_accuracy_is_censored_at_the_last_truth_call() -> None:
    event = accepted_prefix_event(
        [True, True, True, None, None],
        requested_horizon=5,
        event_name="accurate",
    )

    assert event["first_failure_call"] is None
    assert event["accepted_prefix_calls"] == 3
    assert event["right_censored"] is True
    assert event["censor_call"] == 3
    assert event["censor_reason"] == "truth_horizon"


def test_unavailable_accuracy_must_be_a_terminal_suffix() -> None:
    with pytest.raises(ValueError, match="terminal suffix"):
        accepted_prefix_event(
            [True, None, True],
            requested_horizon=3,
            event_name="accurate",
        )


def test_event_values_must_be_literal_booleans_or_none() -> None:
    with pytest.raises(TypeError, match="bool or None"):
        accepted_prefix_event(
            [True, 1],
            requested_horizon=2,
            event_name="finite",
        )


def test_four_event_times_remain_distinct_after_recovery() -> None:
    rows = [
        _event_row(1),
        _event_row(2, accurate=False),
        _event_row(3, accurate=True, bounded=False),
        _event_row(4, accurate=True, bounded=True, admissible=False),
        _event_row(
            5,
            accurate=True,
            bounded=True,
            admissible=True,
            finite=False,
        ),
    ]

    summary = summarize_stability_events(rows, requested_horizon=5, case_id="case")

    assert summary["events"]["accurate"]["accepted_prefix_calls"] == 1
    assert summary["events"]["bounded"]["accepted_prefix_calls"] == 2
    assert summary["events"]["admissible"]["accepted_prefix_calls"] == 3
    assert summary["events"]["finite"]["accepted_prefix_calls"] == 4
    assert summary["events"]["accurate"]["recovered_after_failure"] is True


def test_simultaneous_failures_share_one_first_event_call() -> None:
    summary = summarize_stability_events(
        [
            _event_row(1),
            _event_row(
                2,
                accurate=False,
                admissible=False,
                bounded=False,
                finite=False,
            ),
        ],
        requested_horizon=2,
        case_id="simultaneous",
    )

    assert {event["first_failure_call"] for event in summary["events"].values()} == {2}
    assert {event["accepted_prefix_calls"] for event in summary["events"].values()} == {
        1
    }


def test_truth_free_calls_cannot_smuggle_in_accuracy_values() -> None:
    metrics = diagnose_euler_state(
        _primitive(),
        representation="primitive_rho_v1_v2_p",
    )
    with pytest.raises(ValueError, match="truth-free"):
        classify_call_events(
            state_diagnostics=metrics,
            truth_available=False,
            common_relative_l2=0.0,
            common_amplitude_ratio=1.0,
            common_scaled_rms_ratio=1.0,
            thresholds=StabilityThresholds(),
        )


def test_finite_calls_require_nonnegative_observed_metrics() -> None:
    metrics = diagnose_euler_state(
        _primitive(),
        representation="primitive_rho_v1_v2_p",
    )
    with pytest.raises(ValueError, match="lacks required common_relative_l2"):
        classify_call_events(
            state_diagnostics=metrics,
            truth_available=True,
            common_relative_l2=None,
            common_amplitude_ratio=1.0,
            common_scaled_rms_ratio=1.0,
            thresholds=StabilityThresholds(),
        )
    with pytest.raises(ValueError, match="nonnegative and not NaN"):
        classify_call_events(
            state_diagnostics=metrics,
            truth_available=True,
            common_relative_l2=0.01,
            common_amplitude_ratio=-1.0,
            common_scaled_rms_ratio=1.0,
            thresholds=StabilityThresholds(),
        )


def test_positive_infinite_metric_is_an_observed_failure_not_missing_data() -> None:
    conservative_zero_density = torch.tensor([[[0.0, 1.0, 0.0, 1.0]]])
    metrics = diagnose_euler_state(
        conservative_zero_density,
        representation="conservative_rho_m1_m2_E",
    )

    events = classify_call_events(
        state_diagnostics=metrics,
        truth_available=True,
        common_relative_l2=float("inf"),
        common_amplitude_ratio=float("inf"),
        common_scaled_rms_ratio=float("inf"),
        thresholds=StabilityThresholds(),
    )

    assert events == {
        "accurate": False,
        "admissible": False,
        "bounded": False,
        "finite": True,
    }


@pytest.mark.parametrize(
    ("truth_available", "expected_accurate"),
    [(True, False), (False, None)],
)
def test_returned_native_nonfinite_state_has_explicit_event_semantics(
    truth_available: bool,
    expected_accurate: bool | None,
) -> None:
    nonfinite = _primitive()
    nonfinite[..., 3] = torch.inf
    metrics = diagnose_euler_state(
        nonfinite,
        representation="primitive_rho_v1_v2_p",
    )

    events = classify_call_events(
        state_diagnostics=metrics,
        truth_available=truth_available,
        common_relative_l2=None,
        common_amplitude_ratio=None,
        common_scaled_rms_ratio=None,
        thresholds=StabilityThresholds(),
    )

    assert events == {
        "accurate": expected_accurate,
        "admissible": False,
        "bounded": False,
        "finite": False,
    }


def test_event_thresholds_are_inclusive_and_admissibility_is_independent() -> None:
    inadmissible = diagnose_euler_state(
        _primitive(pressure=0.0),
        representation="primitive_rho_v1_v2_p",
    )

    events = classify_call_events(
        state_diagnostics=inadmissible,
        truth_available=True,
        common_relative_l2=0.05,
        common_amplitude_ratio=100.0,
        common_scaled_rms_ratio=10.0,
        thresholds=StabilityThresholds(),
    )

    assert events == {
        "accurate": True,
        "admissible": False,
        "bounded": True,
        "finite": True,
    }


def test_d019_padding_policy_feeds_invalid_active_proposal_back() -> None:
    valid = torch.cat((_primitive(), torch.zeros_like(_primitive())), dim=1)
    proposal = torch.cat(
        (_primitive(pressure=0.0), torch.full_like(_primitive(), 9.0)),
        dim=1,
    )
    node_mask = torch.tensor([[[1.0], [0.0]]])
    deployed = proposal * node_mask
    traces = [
        StepTrace(
            call=1,
            current_native=valid,
            model_input_native=valid,
            model_proposal_native=proposal,
            deployed_native=deployed,
            representation="primitive_rho_v1_v2_p",
            recurrence_source="deployed",
        ),
        StepTrace(
            call=2,
            current_native=deployed,
            model_input_native=deployed,
            model_proposal_native=deployed,
            deployed_native=deployed,
            representation="primitive_rho_v1_v2_p",
            recurrence_source="deployed",
        ),
    ]

    assert (
        validate_trace_sequence(traces, node_mask=node_mask)["exact_recurrence_closure"]
        is True
    )
    attribution = recovery_policy_attribution(traces, node_mask=node_mask)
    assert attribution[0]["proposal_full_state_fed_back"] is False
    assert attribution[0]["proposal_invalid_active_nodes_fed_back"] is True
    assert attribution[0]["deployed_invalid_fed_back"] is True
    assert attribution[0]["proposal_deployed_padding_only_difference"] is True
    assert attribution[0]["output_transition"] == (
        "output_projection_persistent_inadmissibility"
    )
    assert attribution[1]["feedback_observed"] is False
    assert attribution[1]["proposal_full_state_fed_back"] is None
    assert attribution[1]["proposal_invalid_active_nodes_fed_back"] is None
    assert attribution[1]["deployed_invalid_fed_back"] is None


def test_output_policy_recovery_feeds_deployed_not_invalid_proposal() -> None:
    valid = _primitive()
    invalid = _primitive(pressure=0.0)
    traces = [
        StepTrace(
            call=1,
            current_native=valid,
            model_input_native=valid,
            model_proposal_native=invalid,
            deployed_native=valid,
            representation="primitive_rho_v1_v2_p",
            recurrence_source="deployed",
        ),
        StepTrace(
            call=2,
            current_native=valid,
            model_input_native=valid,
            model_proposal_native=valid,
            deployed_native=valid,
            representation="primitive_rho_v1_v2_p",
        ),
    ]

    attribution = recovery_policy_attribution(traces)
    assert attribution[0]["output_transition"] == "output_projection_recovery"
    assert attribution[0]["proposal_full_state_fed_back"] is False
    assert attribution[0]["proposal_invalid_active_nodes_fed_back"] is False
    assert attribution[0]["deployed_invalid_fed_back"] is False


def test_wrong_recurrence_link_is_rejected() -> None:
    valid = _primitive()
    other = _primitive(v1=0.3)
    traces = [
        StepTrace(
            call=1,
            current_native=valid,
            model_input_native=valid,
            model_proposal_native=valid,
            deployed_native=valid,
            representation="primitive_rho_v1_v2_p",
        ),
        StepTrace(
            call=2,
            current_native=other,
            model_input_native=other,
            model_proposal_native=other,
            deployed_native=other,
            representation="primitive_rho_v1_v2_p",
        ),
    ]

    with pytest.raises(ValueError, match="recurrence mismatch"):
        validate_trace_sequence(traces)


def test_exact_recurrence_rejects_a_dtype_change() -> None:
    valid64 = _primitive()
    valid32 = valid64.float()
    traces = [
        StepTrace(
            call=1,
            current_native=valid64,
            model_input_native=valid64,
            model_proposal_native=valid64,
            deployed_native=valid64,
            representation="primitive_rho_v1_v2_p",
        ),
        StepTrace(
            call=2,
            current_native=valid32,
            model_input_native=valid32,
            model_proposal_native=valid32,
            deployed_native=valid32,
            representation="primitive_rho_v1_v2_p",
        ),
    ]

    with pytest.raises(ValueError, match="recurrence dtype mismatch"):
        validate_trace_sequence(traces)


def test_trace_sequence_rejects_empty_or_changing_contracts() -> None:
    valid = _primitive()
    with pytest.raises(ValueError, match="at least one"):
        validate_trace_sequence([])

    changing = [
        StepTrace(
            call=1,
            current_native=valid,
            model_input_native=valid,
            model_proposal_native=valid,
            deployed_native=valid,
            representation="primitive_rho_v1_v2_p",
            recurrence_source="deployed",
        ),
        StepTrace(
            call=2,
            current_native=valid,
            model_input_native=valid,
            model_proposal_native=valid,
            deployed_native=valid,
            representation="primitive_rho_v1_v2_p",
            recurrence_source="model_proposal",
        ),
    ]
    with pytest.raises(ValueError, match="recurrence source"):
        validate_trace_sequence(changing)

    mixed_representation = [
        changing[0],
        StepTrace(
            call=2,
            current_native=valid,
            model_input_native=valid,
            model_proposal_native=valid,
            deployed_native=valid,
            representation="conservative_rho_m1_m2_E",
            recurrence_source="deployed",
        ),
    ]
    with pytest.raises(ValueError, match="state representation"):
        validate_trace_sequence(mixed_representation)


def test_native_nonfinite_state_must_terminate_but_finite_invalid_may_continue() -> (
    None
):
    valid = _primitive()
    finite_invalid = _primitive(pressure=0.0)
    finite_invalid_traces = [
        StepTrace(
            1,
            valid,
            valid,
            finite_invalid,
            finite_invalid,
            "primitive_rho_v1_v2_p",
        ),
        StepTrace(
            2,
            finite_invalid,
            finite_invalid,
            finite_invalid,
            finite_invalid,
            "primitive_rho_v1_v2_p",
        ),
    ]
    assert (
        validate_trace_sequence(finite_invalid_traces)["recurrence_links_checked"] == 1
    )

    nonfinite = finite_invalid.clone()
    nonfinite[..., 3] = torch.inf
    nonfinite_traces = [
        StepTrace(
            1,
            valid,
            valid,
            nonfinite,
            nonfinite,
            "primitive_rho_v1_v2_p",
        ),
        StepTrace(
            2,
            nonfinite,
            nonfinite,
            nonfinite,
            nonfinite,
            "primitive_rho_v1_v2_p",
        ),
    ]
    with pytest.raises(ValueError, match="must terminate"):
        validate_trace_sequence(nonfinite_traces)


@pytest.mark.parametrize(
    "representation",
    ["primitive_rho_v1_v2_p", "conservative_rho_m1_m2_E"],
)
def test_common_coordinate_decomposition_closes_for_both_representations(
    representation: str,
) -> None:
    truth = _primitive()
    fresh = _primitive(v1=0.25)
    rollout = _primitive(v1=0.4, pressure=1.1)
    common = []
    for state in (rollout, fresh, truth):
        native = common_primitive_to_native(state, representation=representation)
        common.append(native_to_common_primitive(native, representation=representation))

    result = decompose_common_error(
        prediction_from_rollout_input=common[0],
        prediction_from_truth_input=common[1],
        truth_next=common[2],
    )

    assert result["closure_max_abs"] <= 1.0e-12
    assert result["closure_relative_l2"] <= 1.0e-12


def test_weighted_energy_and_cross_term_closure_is_cancellation_robust() -> None:
    propagated = torch.tensor([[[1.0e8, -2.0e8], [3.0e8, -4.0e8]]], dtype=torch.float64)
    fresh = -propagated + torch.tensor(
        [[[1.0, -2.0], [3.0, -4.0]]], dtype=torch.float64
    )
    total = propagated + fresh

    result = weighted_error_energy_closure(
        total_error=total,
        propagated_input_error=propagated,
        fresh_one_step_defect=fresh,
        proxy_weights=torch.tensor([[1.0, 3.0]], dtype=torch.float64),
        component_scales=torch.tensor([2.0, 0.5], dtype=torch.float64),
    )

    assert result["weighted_vector_closure_relative"] <= 1.0e-12
    assert result["weighted_energy_closure_relative"] <= 1.0e-12
    assert result["weighted_propagated_fresh_cross_term"] < 0.0
    assert result["weighted_energy_closure_denominator"] == pytest.approx(
        max(
            result["weighted_total_energy"],
            result["weighted_propagated_energy"]
            + result["weighted_fresh_energy"]
            + 2.0 * abs(result["weighted_propagated_fresh_cross_term"]),
        )
    )
    assert result["weighted_energy_closure_denominator"] > (
        1.0e6 * result["weighted_total_energy"]
    )


def test_weighted_energy_closure_rejects_invalid_shapes_and_values() -> None:
    total = torch.ones((1, 2, 2), dtype=torch.float64)
    valid = {
        "total_error": total,
        "propagated_input_error": 0.25 * total,
        "fresh_one_step_defect": 0.75 * total,
        "proxy_weights": torch.ones((1, 2), dtype=torch.float64),
        "component_scales": torch.ones(2, dtype=torch.float64),
    }

    with pytest.raises(ValueError, match="same shape"):
        weighted_error_energy_closure(
            **{**valid, "fresh_one_step_defect": torch.ones((1, 1, 2))}
        )
    with pytest.raises(ValueError, match="finite values"):
        nonfinite = total.clone()
        nonfinite[0, 0, 0] = torch.nan
        weighted_error_energy_closure(**{**valid, "total_error": nonfinite})
    with pytest.raises(ValueError, match="strictly positive"):
        weighted_error_energy_closure(
            **{**valid, "component_scales": torch.tensor([1.0, 0.0])}
        )
    with pytest.raises(ValueError, match="nonnegative"):
        weighted_error_energy_closure(
            **{**valid, "proxy_weights": torch.tensor([[1.0, -1.0]])}
        )
    with pytest.raises(ValueError, match="positive total weight"):
        weighted_error_energy_closure(**{**valid, "proxy_weights": torch.zeros((1, 2))})


def test_survival_table_uses_case_first_risk_sets() -> None:
    failed = summarize_stability_events(
        [_event_row(1), _event_row(2, accurate=False), _event_row(3, accurate=True)],
        requested_horizon=3,
        case_id="failed",
    )
    censored = summarize_stability_events(
        [_event_row(1), _event_row(2), _event_row(3)],
        requested_horizon=3,
        case_id="censored",
    )

    rows = event_survival_table([failed, censored])
    accurate = {row["call"]: row for row in rows if row["event"] == "accurate"}

    assert accurate[2]["at_risk_count"] == 2
    assert accurate[2]["failure_count"] == 1
    assert accurate[2]["kaplan_meier_survival"] == pytest.approx(0.5)
    assert accurate[3]["at_risk_count"] == 1
    assert accurate[3]["kaplan_meier_survival"] == pytest.approx(0.5)


def test_survival_support_ends_at_an_early_failure_despite_recovery_rows() -> None:
    first = summarize_stability_events(
        [_event_row(1), _event_row(2, accurate=False), _event_row(3)],
        requested_horizon=3,
        case_id="first",
    )
    second = summarize_stability_events(
        [_event_row(1), _event_row(2, accurate=False), _event_row(3)],
        requested_horizon=3,
        case_id="second",
    )

    survival = event_survival_table([first, second])

    assert [row["call"] for row in survival if row["event"] == "accurate"] == [1, 2]


def test_accuracy_survival_stops_when_truth_support_ends() -> None:
    rows = [
        _event_row(1),
        _event_row(2),
        _event_row(3, accurate=None),
        _event_row(4, accurate=None),
    ]
    summary = summarize_stability_events(rows, requested_horizon=4, case_id="truth2")

    survival = event_survival_table([summary])

    assert [row["call"] for row in survival if row["event"] == "accurate"] == [1, 2]
    assert [row["call"] for row in survival if row["event"] == "finite"] == [
        1,
        2,
        3,
        4,
    ]


def test_survival_table_rejects_duplicate_case_rows() -> None:
    summary = summarize_stability_events(
        [_event_row(1)],
        requested_horizon=1,
        case_id="duplicate",
    )
    with pytest.raises(ValueError, match="unique case_id"):
        event_survival_table([summary, summary])


def test_scientific_trajectory_summary_requires_one_observed_call() -> None:
    with pytest.raises(ValueError, match="at least one call row"):
        summarize_stability_events([], requested_horizon=3, case_id="empty")


def _provenance_row() -> dict[str, object]:
    row: dict[str, object] = {field: f"known-{field}" for field in PROVENANCE_FIELDS}
    for field in DIGEST_FIELDS:
        row[field] = "a" * 64
    row.update(
        {
            "evaluator_digest": "b" * 64,
            "checkpoint_size_bytes": 1,
            "state_representation": "primitive_rho_v1_v2_p",
            "training_seed": 7,
            "training_presentations": 10,
            "optimizer_steps": 5,
            "unresolved_fields": [],
            "claim_boundary": "descriptive native-system comparison only",
        }
    )
    return row


def test_exact_comparison_kinds_enforce_claim_scope() -> None:
    training_gaps = sorted(TRAINING_SIDE_PROVENANCE_FIELDS)
    descriptive_gaps = [*training_gaps, "native_map_equivalence_digest"]
    descriptive_row = _provenance_row()
    for field in descriptive_gaps:
        descriptive_row[field] = None
    descriptive_row["unresolved_fields"] = descriptive_gaps

    result = validate_provenance_row(
        descriptive_row,
        comparison_kind="descriptive_survival",
    )

    assert result["provenance_complete"] is False
    assert result["unresolved_fields"] == sorted(descriptive_gaps)
    assert "native_map_equivalence_digest" not in DESCRIPTIVE_SURVIVAL_REQUIRED_FIELDS

    inference_row = _provenance_row()
    for field in training_gaps:
        inference_row[field] = None
    inference_row["unresolved_fields"] = training_gaps
    inference = validate_provenance_row(
        inference_row,
        comparison_kind="inference_map_diagnostic",
    )
    assert inference["provenance_complete"] is False

    with pytest.raises(ValueError, match="separate registered intervention/validator"):
        validate_provenance_row(
            _provenance_row(),
            comparison_kind="training_factor_causal_attribution",
        )


def test_inference_map_diagnostic_rejects_an_inference_side_gap() -> None:
    row = _provenance_row()
    row["native_map_equivalence_digest"] = None
    row["unresolved_fields"] = ["native_map_equivalence_digest"]

    with pytest.raises(ValueError, match="inference-side provenance"):
        validate_provenance_row(row, comparison_kind="inference_map_diagnostic")


def test_descriptive_survival_rejects_a_required_execution_gap() -> None:
    row = _provenance_row()
    row["runtime_manifest_digest"] = None
    row["unresolved_fields"] = ["runtime_manifest_digest"]

    with pytest.raises(ValueError, match="executable survival provenance"):
        validate_provenance_row(row, comparison_kind="descriptive_survival")


@pytest.mark.parametrize(
    "comparison_kind",
    [
        "descriptive_survival",
        "inference_map_diagnostic",
        "training_factor_causal_attribution",
    ],
)
def test_every_comparison_kind_requires_an_explicit_claim_boundary(
    comparison_kind: str,
) -> None:
    row = _provenance_row()
    row["claim_boundary"] = ""

    with pytest.raises(ValueError, match="nonempty claim_boundary"):
        validate_provenance_row(row, comparison_kind=comparison_kind)


def test_provenance_rejects_old_or_unknown_modes_and_malformed_known_digest() -> None:
    row = _provenance_row()
    for comparison_kind in ("descriptive", "mechanistic", "exploratory"):
        with pytest.raises(ValueError, match="unsupported comparison_kind"):
            validate_provenance_row(row, comparison_kind=comparison_kind)

    row["evaluation_data_digest"] = "not-a-digest"
    with pytest.raises(ValueError, match="evaluation_data_digest"):
        validate_provenance_row(row, comparison_kind="descriptive_survival")

    row = _provenance_row()
    row["training_source_digest"] = None
    with pytest.raises(ValueError, match="must match exactly"):
        validate_provenance_row(row, comparison_kind="descriptive_survival")


@pytest.mark.parametrize(
    "field", ["recurrence", "boundary_policy", "optimizer_history"]
)
def test_provenance_rejects_empty_resolved_contract(field: str) -> None:
    row = _provenance_row()
    row[field] = ""

    with pytest.raises(ValueError, match=f"{field} must be a nonempty contract"):
        validate_provenance_row(row, comparison_kind="descriptive_survival")


def test_dry_run_cli_is_synthetic_only_and_writes_nothing(
    tmp_path, monkeypatch, capsys
) -> None:
    monkeypatch.chdir(tmp_path)

    assert main(["--dry-run"]) == 0

    payload = capsys.readouterr().out
    assert '"scope": "synthetic_contract_only"' in payload
    assert '"provenance_validator_scope": "scientific_result_rows_only"' in payload
    assert '"training_factor_causal_attribution_supported": false' in payload
    assert '"executed_inference_source_digest"' in payload
    assert '"native_map_equivalence_digest"' in payload
    assert '"runtime_manifest_digest"' in payload
    assert '"returned_stage_manifest_digest"' in payload
    assert '"inference_source_digest"' not in payload
    assert '"final_hash_manifest_schema"' in payload
    assert '"live_transport_log": "outside_scientific_root_until_closed"' in payload
    assert '"post_write_reverification": "required"' in payload
    assert '"checkpoint_loading_supported": false' in payload
    assert list(tmp_path.iterdir()) == []


def test_help_cli_writes_nothing(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])

    assert exc_info.value.code == 0
    assert "--dry-run" in capsys.readouterr().out
    assert list(tmp_path.iterdir()) == []


def test_final_hash_manifest_requires_a_closed_exact_top_level_inventory(
    tmp_path,
) -> None:
    contaminated = tmp_path / "contaminated"
    contaminated.mkdir()
    (contaminated / "preflight.json").write_text("{}", encoding="utf-8")
    (contaminated / "h2_screen.log").write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="unregistered top-level files"):
        build_final_hash_manifest(contaminated, ["preflight.json"])

    run_root = tmp_path / "run"
    run_root.mkdir()
    (run_root / "preflight.json").write_text("{}", encoding="utf-8")
    (run_root / "execution_log.json").write_text("[]", encoding="utf-8")
    transport_root = tmp_path / "transport"
    transport_root.mkdir()
    (transport_root / "screen.log").write_text("closed", encoding="utf-8")
    (transport_root / "screen.exit").write_text("0\n", encoding="utf-8")

    manifest = build_final_hash_manifest(
        run_root,
        ["preflight.json", "execution_log.json"],
    )

    assert manifest["schema"] == FINAL_HASH_MANIFEST_SCHEMA
    assert [entry["path"] for entry in manifest["files"]] == [
        "execution_log.json",
        "preflight.json",
    ]
    assert verify_final_hash_manifest(run_root, manifest) == {
        "schema": FINAL_HASH_MANIFEST_SCHEMA,
        "verified_file_count": 2,
    }
    assert not (run_root / FINAL_HASH_MANIFEST_NAME).exists()


def test_final_hash_manifest_detects_mutation_and_rejects_unsafe_names(
    tmp_path,
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    result = run_root / "result.json"
    result.write_text('{"status":"closed"}', encoding="utf-8")
    manifest = build_final_hash_manifest(run_root, ["result.json"])

    result.write_text('{"status":"changed"}', encoding="utf-8")
    with pytest.raises(ValueError, match="does not match the closed-file manifest"):
        verify_final_hash_manifest(run_root, manifest)

    with pytest.raises(ValueError, match="top-level POSIX file name"):
        build_final_hash_manifest(run_root, ["../result.json"])
    with pytest.raises(ValueError, match="must not include itself"):
        build_final_hash_manifest(run_root, [FINAL_HASH_MANIFEST_NAME])
