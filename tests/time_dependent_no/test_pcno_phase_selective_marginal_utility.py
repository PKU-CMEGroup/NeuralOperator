from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from scripts.time_dependent_no import (
    analyze_pcno_phase_selective_marginal_utility as analysis,
)


def _fake_tables(packet_id: str, *, error_offset: float = 0.0):
    metrics: list[dict[str, object]] = []
    routes: list[dict[str, object]] = []
    positions: list[dict[str, object]] = []
    views: list[dict[str, object]] = []
    cases = analysis.PACKET_CASES[packet_id]
    for case_index, case_id in enumerate(cases):
        positions.append(
            {
                "case_id": case_id,
                "position_selected": True,
                "expected_selected": True,
            }
        )
        for input_call in analysis.INPUT_CALLS:
            route = analysis._expected_route(input_call)
            base = 0.2 + 0.01 * case_index + 0.002 * input_call + error_offset
            corrected = base - 0.0005 * (1 + input_call % 3)
            for policy, error in (
                (analysis.ZERO_POLICY, base),
                (analysis.CANDIDATE_POLICY, corrected),
            ):
                metrics.append(
                    {
                        "case_id": case_id,
                        "policy": policy,
                        "input_call": str(input_call),
                        "state_error": error,
                        "rank8_state_error": 0.5 * error,
                        "increment_defect": 2.0 * error,
                        "cumulative_defect": 3.0 * error,
                        "finite": True,
                        "admissible": True,
                        "position_selected": True,
                        "correction_active": route == "exact_a32_window",
                        "route": route if policy == analysis.CANDIDATE_POLICY else "",
                        "retained_displacement_rms": (
                            0.01 * (input_call + 1)
                            if policy == analysis.CANDIDATE_POLICY
                            else ""
                        ),
                        "correction_rms": (
                            0.02 * (input_call + 1)
                            if policy == analysis.CANDIDATE_POLICY
                            else 0.0
                        ),
                    }
                )
            routes.append(
                {
                    "case_id": case_id,
                    "input_call": str(input_call),
                    "position_selected": True,
                    "correction_active": route == "exact_a32_window",
                    "route": route,
                }
            )
        for band_index, band in enumerate(analysis.BANDS):
            for view_index, view in enumerate(analysis.STRUCTURAL_VIEWS):
                zero_sse = 1.0 + 0.1 * case_index + 0.01 * band_index
                views.append(
                    {
                        "scope": "case",
                        "case_id": case_id,
                        "cell": band,
                        "view": view,
                        "zero_sse": zero_sse,
                        "corrected_sse": zero_sse
                        - 0.001 * (1 + view_index + band_index),
                    }
                )
    return {
        "metrics": metrics,
        "routes": routes,
        "positions": positions,
        "views": views,
    }


def _patch_tables(monkeypatch: pytest.MonkeyPatch, *, error_offset: float = 0.0):
    tables = {
        packet_id: _fake_tables(packet_id, error_offset=error_offset)
        for packet_id in analysis.PACKET_CASES
    }
    monkeypatch.setattr(
        analysis,
        "_load_packet_tables",
        lambda packet: tables[str(packet["packet_id"])],
    )
    return tables


def _fake_packets():
    return ({"packet_id": "a43"}, {"packet_id": "a44_r1"})


def test_synthetic_contract_freezes_population_features_and_nonexecution() -> None:
    summary = analysis.synthetic_summary()

    assert summary["status"] == "passed"
    assert all(summary["checks"].values())
    assert summary["model_calls"] == 0
    assert summary["truth_arrays_loaded"] is False
    assert summary["recurrent_policy_simulated"] is False
    assert analysis.FEATURES == (
        "lag_displacement",
        "lag_intervention",
        "lag_displacement_change",
    )


def test_routes_and_bands_are_total_and_disjoint() -> None:
    assert [analysis._expected_route(call) for call in analysis.INPUT_CALLS] == [
        *("exact_a32_window" for _ in range(8)),
        *("surrogate_coast" for _ in range(14)),
        *("exact_a32_window" for _ in range(8)),
    ]
    assert sorted(call for calls in analysis.BANDS.values() for call in calls) == list(
        analysis.INPUT_CALLS
    )
    with pytest.raises(ValueError, match="outside"):
        analysis._expected_route(30)


def test_inventory_verification_rejects_tampering(tmp_path: Path) -> None:
    for name in analysis.REQUIRED_INPUT_FILES:
        (tmp_path / name).write_text(name, encoding="utf-8")
    inventory = {
        name: {
            "sha256": analysis.sha256_file(tmp_path / name),
            "bytes": (tmp_path / name).stat().st_size,
        }
        for name in analysis.REQUIRED_INPUT_FILES
    }

    verified = analysis._verify_inventory(tmp_path.resolve(), inventory)
    assert set(verified) == set(analysis.REQUIRED_INPUT_FILES)

    (tmp_path / "view_scores.csv").write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="mismatch"):
        analysis._verify_inventory(tmp_path.resolve(), inventory)


def test_call_features_are_strictly_lagged(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_tables(monkeypatch)
    rows = analysis.build_call_rows(_fake_packets())
    case_rows = [row for row in rows if row["case_id"] == "sv_e00_y01"]

    assert len(rows) == len(analysis.ACTIVE_CASES) * len(analysis.INPUT_CALLS)
    assert case_rows[0]["lag_displacement"] == 0.0
    assert case_rows[0]["lag_intervention"] == 0.0
    assert case_rows[0]["lag_displacement_change"] == 0.0
    assert case_rows[1]["lag_displacement"] == pytest.approx(0.01)
    assert case_rows[1]["lag_intervention"] == pytest.approx(0.02)
    assert case_rows[1]["lag_displacement_change"] == pytest.approx(0.01)
    assert case_rows[2]["lag_displacement"] == pytest.approx(0.02)
    assert case_rows[2]["lag_displacement_change"] == pytest.approx(0.01)


def test_error_changes_cannot_change_predictors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_tables(monkeypatch, error_offset=0.0)
    first = analysis.build_call_rows(_fake_packets())
    _patch_tables(monkeypatch, error_offset=3.0)
    second = analysis.build_call_rows(_fake_packets())

    first_features = [tuple(row[name] for name in analysis.FEATURES) for row in first]
    second_features = [tuple(row[name] for name in analysis.FEATURES) for row in second]
    assert first_features == second_features
    assert [row["full_state"] for row in first] != [row["full_state"] for row in second]


def test_build_call_rows_fails_closed_on_route_tampering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tables = _patch_tables(monkeypatch)
    candidate = next(
        row
        for row in tables["a43"]["metrics"]
        if row["policy"] == analysis.CANDIDATE_POLICY and row["input_call"] == "8"
    )
    candidate["route"] = "exact_a32_window"

    with pytest.raises(ValueError, match="route"):
        analysis.build_call_rows(_fake_packets())


def _regression_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for group_index, (group_id, cases) in enumerate(analysis.GROUP_CASES.items()):
        for case_id in cases:
            for input_call in range(6):
                displacement = 0.2 + 0.03 * input_call + 0.01 * group_index
                intervention = 0.1 + 0.02 * input_call
                change = 0.01 * ((input_call % 3) - 1)
                exact = float(input_call < 3)
                rows.append(
                    {
                        "row_id": f"{case_id}|{input_call}",
                        "group_id": group_id,
                        "case_id": case_id,
                        "input_call": input_call,
                        "band": "band_0_7",
                        "route": ("exact_a32_window" if exact else "surrogate_coast"),
                        "exact_phase": exact,
                        "lag_displacement": displacement,
                        "lag_intervention": intervention,
                        "lag_displacement_change": change,
                        "full_state": (
                            0.4 * displacement
                            - 0.2 * intervention
                            + 0.1 * change
                            + 0.03 * exact
                        ),
                    }
                )
    return rows


def test_grouped_crossfit_is_permutation_invariant_and_leak_free() -> None:
    rows = _regression_rows()
    first_predictions, first_coefficients = analysis.crossfit_predictions(
        rows,
        targets=("full_state",),
        split_key="group_id",
        crossfit="leave_strength_group_out",
    )
    second_predictions, second_coefficients = analysis.crossfit_predictions(
        list(reversed(rows)),
        targets=("full_state",),
        split_key="group_id",
        crossfit="leave_strength_group_out",
    )

    assert first_predictions == second_predictions
    assert first_coefficients == second_coefficients
    assert len(first_predictions) == len(rows) * len(analysis.MODELS)
    assert all(row["holdout"] == row["group_id"] for row in first_predictions)


def test_fit_model_rejects_unresolved_feature_scale() -> None:
    rows = _regression_rows()
    for row in rows:
        row["lag_displacement"] = 1.0
    with pytest.raises(ValueError, match="feature scale"):
        analysis._fit_model(rows, "full_state", "history")


def test_metric_denominators_and_single_sign_statuses_are_explicit() -> None:
    resolved = analysis._metric_summary(np.asarray([1.0, 2.0]), np.asarray([1.0, 2.0]))
    unresolved = analysis._metric_summary(np.zeros(2), np.zeros(2))

    assert resolved["skill_vs_zero"] == pytest.approx(1.0)
    assert resolved["cosine"] == pytest.approx(1.0)
    assert resolved["pearson"] == pytest.approx(1.0)
    assert resolved["balanced_accuracy"] is None
    assert resolved["balanced_accuracy_status"] == "single_sign"
    assert unresolved["skill_vs_zero"] is None
    assert unresolved["skill_vs_zero_status"] == "small_denominator"
    assert unresolved["cosine_denominator"] == 0.0
    assert unresolved["pearson_denominator"] == 0.0


def _coefficient_rows(values: tuple[float, ...]):
    return [
        {
            "crossfit": "leave_strength_group_out",
            "holdout": group_id,
            "target": "full_state",
            "model": "phase_history",
            "feature": "lag_displacement",
            "coefficient_standardized": value,
        }
        for group_id, value in zip(analysis.GROUP_CASES, values, strict=True)
    ]


def test_coefficient_stability_requires_common_sign_and_bounded_scale() -> None:
    stable = analysis.coefficient_stability(
        _coefficient_rows((1.0, 2.0, 3.0, 4.0)),
        "full_state",
        "lag_displacement",
    )
    sign_flip = analysis.coefficient_stability(
        _coefficient_rows((1.0, 2.0, -3.0, 4.0)),
        "full_state",
        "lag_displacement",
    )
    scale_fail = analysis.coefficient_stability(
        _coefficient_rows((1.0, 2.0, 3.0, 4.01)),
        "full_state",
        "lag_displacement",
    )

    assert stable["stable"] is True
    assert stable["scale_ratio"] == pytest.approx(4.0)
    assert sign_flip["common_sign"] is False
    assert sign_flip["stable"] is False
    assert scale_fail["stable"] is False


def test_structural_rows_use_only_band_averaged_lagged_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_tables(monkeypatch)
    calls = analysis.build_call_rows(_fake_packets())
    rows = analysis.build_structural_rows(_fake_packets(), calls)
    first = next(
        row
        for row in rows
        if row["case_id"] == "sv_e00_y01"
        and row["band"] == "band_0_7"
        and row["view"] == "full"
    )

    assert len(rows) == (
        len(analysis.ACTIVE_CASES)
        * len(analysis.BANDS)
        * len(analysis.STRUCTURAL_VIEWS)
    )
    expected = np.mean([0.0, *[0.01 * call for call in range(1, 8)]])
    assert first["lag_displacement"] == pytest.approx(expected)
    assert first["route"] == "exact_a32_window"
    assert first["utility"] > 0.0


def _gate_metric(target: str, scope: str, *, skill: float = 0.1):
    return {
        "crossfit": "leave_strength_group_out",
        "target": target,
        "scope": scope,
        "model": "phase_history",
        "skill_vs_zero_status": "ok",
        "skill_vs_zero": skill,
        "skill_vs_phase_status": "ok",
        "skill_vs_phase": skill,
        "cosine_status": "ok",
        "cosine": skill,
        "pearson_status": "ok",
        "pearson": skill,
    }


def test_gate_uses_strict_improvement_and_cannot_be_rescued_by_controls() -> None:
    call_metrics = []
    for target in analysis.PRIMARY_OUTCOMES:
        call_metrics.append(_gate_metric(target, "overall"))
        call_metrics.extend(
            _gate_metric(target, f"group:{group_id}")
            for group_id in analysis.GROUP_CASES
        )
        call_metrics.extend(
            _gate_metric(target, f"route:{route}")
            for route in ("exact_a32_window", "surrogate_coast")
        )
    structural_metrics = [
        _gate_metric(view, "overall") for view in analysis.REQUIRED_STRUCTURAL_VIEWS
    ]
    stabilities = [
        {"stable": True}
        for _ in range(len(analysis.PRIMARY_OUTCOMES) * len(analysis.FEATURES))
    ]

    passed = analysis.prospective_gate(
        call_metrics,
        structural_metrics,
        stabilities,
        structural_checks={"integrity": True},
    )
    equality_rows = deepcopy(call_metrics)
    equality_rows[0]["skill_vs_phase"] = 0.0
    equality = analysis.prospective_gate(
        equality_rows,
        structural_metrics,
        stabilities,
        structural_checks={"integrity": True},
    )

    assert passed["status"] == "qualified"
    assert all(passed["checks"].values())
    assert equality["status"] == "failed"
    assert "primary_overall_predictability" in equality["failed_checks"]
