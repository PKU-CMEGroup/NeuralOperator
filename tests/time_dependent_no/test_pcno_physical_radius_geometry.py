from __future__ import annotations

from itertools import pairwise
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.time_dependent_no.analyze_pcno_fine_grained_pathways import (
    CASE_IDS,
    RESOLUTIONS,
    _band_fields,
    _pair_label,
)
from scripts.time_dependent_no.analyze_pcno_physical_radius_geometry import (
    ARMS,
    BANDS,
    D070_FROZEN_ARTIFACT_ARGUMENTS,
    D070_INTENTIONAL_SOURCE_DIFFERENCES,
    D070_REQUIRED_UNCHANGED_SOURCES,
    MODES,
    VISUAL_CASE_IDS,
    _all_numeric_values_finite,
    _case_aggregates,
    _contract_masked_decision,
    _d070_compatibility_checks,
    _edge_reciprocity,
    _gate_summary,
    _generic_band_fields,
    _manifest_science_result,
    _operator_sha256,
    _reference_binding_checks,
    _visual_payload_inventory,
)
from utility.time_dependent_no.pcno_resolution_pathways import (
    build_graph_ball_operator,
)


def _passing_aggregates() -> list[dict[str, object]]:
    rows = []
    for case_id in CASE_IDS:
        for coarse, fine in pairwise(RESOLUTIONS):
            pair = _pair_label(coarse, fine)
            for mode in MODES:
                for arm in ARMS:
                    for band in BANDS:
                        value = 1.0
                        if arm == "A2" and band == "large":
                            value = 0.7
                        rows.append(
                            {
                                "case_id": case_id,
                                "pair": pair,
                                "mode": mode,
                                "pathway_level": "decoded_residual",
                                "arm": arm,
                                "band": band,
                                "call_rms_aggregate": value,
                            }
                        )
    return rows


def test_d073_gate_is_conjunctive_over_pair_and_mode_strata() -> None:
    rows, decision = _gate_summary(_passing_aggregates(), smoke=False)

    assert len(rows) == 4
    assert all(row["mechanism_pass"] for row in rows)
    assert all(row["native_relevance_pass"] for row in rows)
    assert decision == {
        "evaluated": True,
        "mechanism_pass": True,
        "native_relevance_pass": True,
        "advance_d073b": True,
    }

    target = next(
        row
        for row in rows
        if row["pair"] == "125x50->250x100" and row["mode"] == "teacher_forced"
    )
    assert target["median_large_reduction_a2_vs_a1"] == pytest.approx(0.3)
    assert target["positive_large_cases"] == 6


def test_d073_gate_rejects_one_failed_stratum_and_small_denominator() -> None:
    aggregates = _passing_aggregates()
    for row in aggregates:
        if (
            row["pair"] == "250x100->500x200"
            and row["mode"] == "baseline_free_input"
            and row["arm"] == "A2"
            and row["band"] == "large"
        ):
            row["call_rms_aggregate"] = 1.0
    gate_rows, decision = _gate_summary(aggregates, smoke=False)
    failed = [row for row in gate_rows if not row["mechanism_pass"]]
    assert len(failed) == 1
    assert not decision["mechanism_pass"]
    assert not decision["advance_d073b"]

    for row in aggregates:
        if row["arm"] == "A1" and row["band"] == "large":
            row["call_rms_aggregate"] = 1.0e-10
            break
    gate_rows, decision = _gate_summary(aggregates, smoke=False)
    assert any(not row["a1_denominator_valid"] for row in gate_rows)
    assert not decision["advance_d073b"]


def test_d073_gate_keeps_mechanism_and_native_relevance_separate() -> None:
    aggregates = _passing_aggregates()
    for row in aggregates:
        if (
            row["pair"] == "125x50->250x100"
            and row["mode"] == "teacher_forced"
            and row["arm"] == "A0"
            and row["band"] == "large"
        ):
            row["call_rms_aggregate"] = 0.5
    gate_rows, decision = _gate_summary(aggregates, smoke=False)
    target = next(
        row
        for row in gate_rows
        if row["pair"] == "125x50->250x100" and row["mode"] == "teacher_forced"
    )
    assert target["mechanism_pass"]
    assert not target["native_relevance_pass"]
    assert decision["mechanism_pass"]
    assert not decision["native_relevance_pass"]
    assert not decision["advance_d073b"]


def test_d073_gate_requires_four_of_six_positive_case_signs() -> None:
    pair = "125x50->250x100"
    mode = "teacher_forced"
    aggregates = _passing_aggregates()
    for case_index, case_id in enumerate(CASE_IDS):
        for row in aggregates:
            if (
                row["case_id"] == case_id
                and row["pair"] == pair
                and row["mode"] == mode
                and row["arm"] == "A2"
                and row["band"] == "large"
            ):
                row["call_rms_aggregate"] = 0.5 if case_index < 4 else 1.0
    gate_rows, _ = _gate_summary(aggregates, smoke=False)
    target = next(
        row for row in gate_rows if row["pair"] == pair and row["mode"] == mode
    )
    assert target["positive_large_cases"] == 4
    assert target["mechanism_pass"]

    for row in aggregates:
        if (
            row["case_id"] == CASE_IDS[3]
            and row["pair"] == pair
            and row["mode"] == mode
            and row["arm"] == "A2"
            and row["band"] == "large"
        ):
            row["call_rms_aggregate"] = 1.0
    gate_rows, _ = _gate_summary(aggregates, smoke=False)
    target = next(
        row for row in gate_rows if row["pair"] == pair and row["mode"] == mode
    )
    assert target["positive_large_cases"] == 3
    assert target["median_large_reduction_a2_vs_a1"] >= 0.2
    assert not target["mechanism_pass"]


@pytest.mark.parametrize("band", ("total", "local"))
def test_d073_gate_enforces_nonlarge_band_guard(band: str) -> None:
    aggregates = _passing_aggregates()
    for row in aggregates:
        if (
            row["pair"] == "125x50->250x100"
            and row["mode"] == "teacher_forced"
            and row["arm"] == "A2"
            and row["band"] == band
        ):
            row["call_rms_aggregate"] = 1.2
    gate_rows, decision = _gate_summary(aggregates, smoke=False)
    target = next(
        row
        for row in gate_rows
        if row["pair"] == "125x50->250x100" and row["mode"] == "teacher_forced"
    )
    assert not target["mechanism_pass"]
    assert not decision["advance_d073b"]


def test_d073_gate_checks_only_registered_bands_and_both_denominator_floors() -> None:
    aggregates = _passing_aggregates()
    for row in aggregates:
        if row["arm"] == "A1" and row["band"] == "transition":
            row["call_rms_aggregate"] = 1.0e-12
    gate_rows, decision = _gate_summary(aggregates, smoke=False)
    assert all(row["a1_denominator_valid"] for row in gate_rows)
    assert decision["advance_d073b"]

    for row in aggregates:
        if row["arm"] == "A1" and row["band"] == "total":
            row["call_rms_aggregate"] = 1.0e-12
            break
    gate_rows, decision = _gate_summary(aggregates, smoke=False)
    assert any(not row["a1_denominator_valid"] for row in gate_rows)
    assert not decision["advance_d073b"]

    aggregates = _passing_aggregates()
    for row in aggregates:
        if row["arm"] == "A0" and row["band"] == "local":
            row["call_rms_aggregate"] = 1.0e-12
            break
    gate_rows, decision = _gate_summary(aggregates, smoke=False)
    assert any(not row["a0_denominator_valid"] for row in gate_rows)
    assert decision["mechanism_pass"]
    assert not decision["native_relevance_pass"]
    assert not decision["advance_d073b"]


def test_contract_failure_suppresses_scientific_decisions_and_manifest_flag() -> None:
    raw = {
        "evaluated": True,
        "mechanism_pass": True,
        "native_relevance_pass": True,
        "advance_d073b": True,
    }
    masked = _contract_masked_decision(raw, contract_pass=False, smoke=False)
    assert masked == {
        "evaluated": False,
        "mechanism_pass": None,
        "native_relevance_pass": None,
        "advance_d073b": False,
        "reason": "scientific_decision_suppressed_by_contract_failure",
    }
    assert not _manifest_science_result(contract_pass=False, smoke=False)
    assert not _manifest_science_result(contract_pass=True, smoke=True)
    assert _manifest_science_result(contract_pass=True, smoke=False)


def test_case_aggregate_is_rms_over_calls_not_a_median_or_node_pool() -> None:
    rows = [
        {
            "case_id": "case",
            "mode": "teacher_forced",
            "pair": "coarse->fine",
            "call": call,
            "pathway_level": "decoded_residual",
            "arm": "A1",
            "band": "large",
            "defect_scaled_rms": value,
        }
        for call, value in ((1, 3.0), (5, 4.0))
    ]
    aggregate = _case_aggregates(rows, required_calls=(1, 5))
    assert len(aggregate) == 1
    assert aggregate[0]["call_rms_aggregate"] == pytest.approx(np.sqrt(12.5))
    with pytest.raises(ValueError, match="incomplete"):
        _case_aggregates(rows[:-1], required_calls=(1, 5))


def test_operator_digest_is_deterministic_and_edge_reciprocity_is_explicit() -> None:
    nodes = np.asarray(((0.0, 0.0), (0.5, 0.0), (1.0, 0.0)))
    reciprocal_edges = np.asarray(((0, 1), (1, 0), (1, 2), (2, 1)))
    weights = np.asarray((1.0, 2.0, 3.0))
    left = build_graph_ball_operator(nodes, reciprocal_edges, weights, radius=0.51)
    right = build_graph_ball_operator(
        nodes, reciprocal_edges[::-1], weights, radius=0.51
    )

    assert _operator_sha256(left) == _operator_sha256(right)
    assert _edge_reciprocity(reciprocal_edges)["reciprocal"]
    nonreciprocal = _edge_reciprocity(np.asarray(((0, 1), (1, 2))))
    assert not nonreciprocal["reciprocal"]
    assert nonreciprocal["missing_reverse_edge_count"] == 2


def test_generic_dct_bands_match_d070_for_four_channels_and_support_latents() -> None:
    rng = np.random.default_rng(73)
    resolution = (8, 4)
    field = rng.normal(size=(resolution[0] * resolution[1], 4))
    scale = np.asarray((0.2, 0.4, 0.8, 1.6))
    expected, expected_closure = _band_fields(
        field, resolution=resolution, residual_scale=scale
    )
    observed, observed_closure = _generic_band_fields(
        field, resolution=resolution, component_scale=scale
    )
    assert observed.keys() == expected.keys()
    for band in observed:
        np.testing.assert_allclose(observed[band], expected[band], atol=1.0e-12)
    for name in observed_closure:
        assert observed_closure[name] == pytest.approx(
            expected_closure[name], abs=1.0e-12
        )

    latent = rng.normal(size=(resolution[0] * resolution[1], 64))
    latent_bands, closure = _generic_band_fields(
        latent, resolution=resolution, component_scale=np.ones(64)
    )
    assert all(value.shape == latent.shape for value in latent_bands.values())
    np.testing.assert_allclose(
        latent_bands["large"] + latent_bands["transition"] + latent_bands["local"],
        latent,
        atol=1.0e-12,
    )
    assert max(closure.values()) < 1.0e-12


def test_d070_compatibility_binds_artifacts_contract_and_unchanged_sources() -> None:
    artifact_values = {
        name: ("a" * 40 if name == "expected_source_base_git_head" else "b" * 64)
        for name in D070_FROZEN_ARTIFACT_ARGUMENTS
    }
    args = SimpleNamespace(
        **artifact_values,
        d067_source_contract="current_core_self_consistent_pathway_contract",
    )
    required_sources = {
        path: f"{index:064x}"
        for index, path in enumerate(D070_REQUIRED_UNCHANGED_SOURCES, start=1)
    }
    intentional_sources = {
        path: f"{index + 100:064x}"
        for index, path in enumerate(D070_INTENTIONAL_SOURCE_DIFFERENCES, start=1)
    }
    d070_sources = {**required_sources, **intentional_sources}
    active_sources = {
        **required_sources,
        **{path: "f" * 64 for path in D070_INTENTIONAL_SOURCE_DIFFERENCES},
    }
    summary = {
        "scientific_interpretation_allowed": True,
        "args": artifact_values,
        "normalization_digest": "normalization-mapping",
        "d067_binding": {
            "source_contract": "current_core_self_consistent_pathway_contract"
        },
        "provenance": {"source_sha256": d070_sources},
    }
    checkpoint = {"normalization_digest": "normalization-mapping"}
    checks = _d070_compatibility_checks(args, summary, checkpoint, active_sources)
    assert checks["passed"]
    assert all(checks["artifact_arguments"].values())
    assert all(checks["required_unchanged_sources"].values())
    assert all(
        row["difference_allowed"]
        for row in checks["intentional_source_differences"].values()
    )

    incompatible_sources = dict(active_sources)
    incompatible_sources[D070_REQUIRED_UNCHANGED_SOURCES[0]] = "0" * 64
    assert not _d070_compatibility_checks(
        args, summary, checkpoint, incompatible_sources
    )["passed"]


def test_d070_reference_binding_is_exact_for_all_six_cases() -> None:
    expected = [
        {
            "case_id": case_id,
            "frozen_training_reference_sha256": f"{index + 1:064x}",
            "active_reference_artifact_sha256": f"{index + 101:064x}",
            "retained_resolution": "500x200",
            "state_dtype": "float64",
            "state_shape": "[61,100000,4]",
        }
        for index, case_id in enumerate(CASE_IDS)
    ]
    active = [
        {
            **row,
            "state_shape": [61, 100000, 4],
            "restriction_crosscheck_max_abs": 1.0e-15,
        }
        for row in expected
    ]
    assert _reference_binding_checks(expected, active)["passed"]

    mismatched = [dict(row) for row in active]
    mismatched[0]["active_reference_artifact_sha256"] = "f" * 64
    checks = _reference_binding_checks(expected, mismatched)
    assert not checks["passed"]
    assert checks["mismatches"] == {CASE_IDS[0]: ["active_reference_artifact_sha256"]}


def _valid_visual_payload_inventory() -> tuple[dict, dict]:
    payloads = {}
    grouped = {
        case_id: {resolution: object() for resolution in RESOLUTIONS}
        for case_id in VISUAL_CASE_IDS
    }
    for case_id in VISUAL_CASE_IDS:
        for coarse, fine in pairwise(RESOLUTIONS):
            pair = _pair_label(coarse, fine)
            field = np.zeros((coarse[0] * coarse[1], 4), dtype=np.float32)
            payload = {
                "call": [1, 5, 15, 30],
                "physical_time": [0.02, 0.1, 0.3, 0.6],
            }
            for arm in ("A0", "A1", "A2"):
                for band in ("total", "large", "local"):
                    payload[f"{arm}_{band}"] = [field] * 4
            for band in ("total", "large", "local"):
                payload[f"A2_minus_A1_{band}"] = [field] * 4
            for mode in MODES:
                payloads[(case_id, pair, mode)] = payload
    return payloads, grouped


def test_visual_payload_inventory_is_exact_and_finite() -> None:
    payloads, grouped = _valid_visual_payload_inventory()
    checked = _visual_payload_inventory(
        payloads,
        active_case_ids=VISUAL_CASE_IDS,
        diagnostic_calls=(1, 5, 15, 30),
        grouped=grouped,
    )
    assert checked["passed"]

    missing = dict(payloads)
    missing.pop(next(iter(missing)))
    assert not _visual_payload_inventory(
        missing,
        active_case_ids=VISUAL_CASE_IDS,
        diagnostic_calls=(1, 5, 15, 30),
        grouped=grouped,
    )["passed"]

    nonfinite = dict(payloads)
    key = next(iter(nonfinite))
    damaged = dict(nonfinite[key])
    damaged_fields = list(damaged["A2_large"])
    damaged_field = damaged_fields[0].copy()
    damaged_field[0, 0] = np.nan
    damaged_fields[0] = damaged_field
    damaged["A2_large"] = damaged_fields
    nonfinite[key] = damaged
    assert not _visual_payload_inventory(
        nonfinite,
        active_case_ids=VISUAL_CASE_IDS,
        diagnostic_calls=(1, 5, 15, 30),
        grouped=grouped,
    )["passed"]
    assert _all_numeric_values_finite([{"value": [1.0, {"nested": 2.0}]}])
    assert not _all_numeric_values_finite([{"value": np.nan}])
