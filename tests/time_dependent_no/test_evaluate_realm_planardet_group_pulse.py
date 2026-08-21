from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.time_dependent_no import (
    evaluate_realm_planardet_group_pulse as pulse,
)
from utility.time_dependent_no.realm_benchmark import canonical_json_sha256


def _trajectory_fixture() -> tuple[np.ndarray, torch.Tensor]:
    baseline = np.zeros((49, 13, 2, 3), dtype=np.float32)
    for index in range(49):
        baseline[index].fill(float(index))
    truth = torch.zeros((50, 13, 2, 3), dtype=torch.float32)
    truth[4, :8].fill_(10.0)
    truth[4, 8:].fill_(3.0)
    return baseline, truth


def _summary(*, after_call: int | None = None) -> dict[str, object]:
    grouped = {group: [1.0] * 49 for group in pulse.feedback.ALL_METRIC_GROUPS}
    if after_call is not None:
        for index in range(after_call, 49):
            grouped["chem"][index] = 2.0
            grouped["T"][index] = 0.5
            grouped["rho"][index] = 0.8
            grouped["u"][index] = 0.5
            grouped["p"][index] = 1.0
    return {
        "call_count": 49,
        "npe_group_by_call": grouped,
        "realm_npe_mean": 5.0 if after_call is None else 4.8,
        "realm_npe_sum_source": 245.0 if after_call is None else 235.2,
        "decoded_correlation_case_first": 0.5,
        "admissible_call_count": 7,
        "bounded_call_count": 49,
    }


def test_arm_id_closes_group_and_schedule() -> None:
    assert pulse.arm_id("chem", 4) == "chem_call4"
    assert pulse.arm_id("rho", 32) == "rho_call32"
    with pytest.raises(ValueError, match="unknown pulse arm"):
        pulse.arm_id("T", 4)
    with pytest.raises(ValueError, match="unknown pulse arm"):
        pulse.arm_id("chem", 5)


def test_preregistration_binds_source_selection_and_claim_contract() -> None:
    payload = pulse.build_preregistration()
    source = pulse.build_diagnostic_source_manifest()
    unsigned = {
        key: value
        for key, value in payload.items()
        if key != "canonical_payload_sha256"
    }
    assert payload["canonical_payload_sha256"] == canonical_json_sha256(unsigned)
    assert (
        payload["diagnostic_source_manifest_digest"]
        == source["canonical_payload_sha256"]
    )
    assert payload["selection_evidence"]["result_sha256"] == pulse.G0B_RESULT_SHA256
    assert payload["selection_evidence"]["pulse_calls"] == {
        str(call): pulse.PULSE_SELECTION[call] for call in pulse.PULSE_CALLS
    }
    assert len(payload["arms"]) == 6
    assert payload["primary_causal_readout"].startswith("for each arm")
    assert payload["sealed_test_object_opened"] is False
    assert [row["path"] for row in source["files"]] == [
        "scripts/time_dependent_no/evaluate_realm_planardet_group_pulse.py",
        "scripts/time_dependent_no/evaluate_realm_planardet_group_feedback.py",
        "scripts/time_dependent_no/evaluate_realm_planardet_pmax_projection.py",
    ]


def test_write_preregistration_is_create_only(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "preregistration.json"
    summary = pulse._write_preregistration(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert summary["canonical_payload_sha256"] == payload["canonical_payload_sha256"]
    with pytest.raises(ValueError, match="already exists"):
        pulse._write_preregistration(path)


def test_validate_g0b_evidence_closes_hash_and_ratios(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = {
        "schema": pulse.feedback.RESULT_SCHEMA,
        "experiment_id": pulse.G0B_EXPERIMENT_ID,
        "diagnostic_interpretation_allowed": True,
        "test_object_opened": False,
        "arms": {
            group: {
                "comparison_to_unintervened_free_baseline": {
                    "primary_untouched_non_pMax_groups": {"ratio": ratio}
                }
            }
            for group, ratio in pulse.G0B_PRIMARY_RATIOS.items()
        },
    }
    result["canonical_payload_sha256"] = canonical_json_sha256(result)
    final = {
        "files": {"result.json": pulse.G0B_RESULT_SHA256},
        "test_object_opened": False,
    }
    (tmp_path / "result.json").write_text(json.dumps(result), encoding="utf-8")
    (tmp_path / "final_hash_manifest.json").write_text(
        json.dumps(final), encoding="utf-8"
    )

    def fake_sha256(path: Path) -> str:
        return (
            pulse.G0B_RESULT_SHA256
            if path.name == "result.json"
            else pulse.G0B_FINAL_MANIFEST_SHA256
        )

    monkeypatch.setattr(pulse, "sha256_file", fake_sha256)
    monkeypatch.setattr(
        pulse,
        "G0B_RESULT_PAYLOAD_SHA256",
        result["canonical_payload_sha256"],
    )
    assert pulse._validate_g0b_evidence(tmp_path)["experiment_id"] == (
        pulse.G0B_EXPERIMENT_ID
    )
    result["diagnostic_interpretation_allowed"] = False
    result["canonical_payload_sha256"] = canonical_json_sha256(
        {
            key: value
            for key, value in result.items()
            if key != "canonical_payload_sha256"
        }
    )
    (tmp_path / "result.json").write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(ValueError, match="contract differs"):
        pulse._validate_g0b_evidence(tmp_path)


def test_prepare_pulse_prefix_is_exact_and_single_group_only() -> None:
    baseline, truth = _trajectory_fixture()
    baseline_copy = baseline.copy()
    truth_copy = truth.clone()
    predictions, recurrent_inputs, accepted, activity = pulse._prepare_pulse_prefix(
        baseline,
        truth,
        group="chem",
        pulse_call=4,
    )
    expected_prefix = torch.from_numpy(baseline[:4])
    assert torch.equal(predictions[:4], expected_prefix)
    assert torch.equal(recurrent_inputs[1:4], expected_prefix[:3])
    assert torch.equal(accepted[:, :8], truth[4, :8].unsqueeze(0))
    assert torch.equal(accepted[:, 8:], expected_prefix[-1:, 8:])
    assert torch.equal(recurrent_inputs[4], accepted[0])
    assert activity["truth_injection_count"] == 1
    assert activity["baseline_prefix_bitwise_exact"] is True
    assert activity["changed_selected_elements"] > 0
    assert np.array_equal(baseline, baseline_copy)
    assert torch.equal(truth, truth_copy)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("dtype", "baseline pulse array"),
        ("call", "pulse call differs"),
        ("nonfinite", "pulse truth"),
    ],
)
def test_prepare_pulse_prefix_rejects_contract_drift(
    mutation: str, message: str
) -> None:
    baseline, truth = _trajectory_fixture()
    pulse_call = 4
    if mutation == "dtype":
        baseline = baseline.astype(np.float64)
    elif mutation == "call":
        pulse_call = 5
    else:
        truth[0, 0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match=message):
        pulse._prepare_pulse_prefix(
            baseline,
            truth,
            group="chem",
            pulse_call=pulse_call,
        )


def test_predict_pulse_recurrence_has_one_pulse_and_raw_suffix() -> None:
    baseline, truth = _trajectory_fixture()

    def predictor(
        model: torch.nn.Module,
        current: torch.Tensor,
        coordinates: torch.Tensor,
        *,
        parameterization: str,
    ) -> torch.Tensor:
        del model, coordinates
        assert parameterization == "residual"
        return current + 1.0

    predictions, inputs, nonfinite, activity, timings = pulse._predict_pulse_recurrence(
        torch.nn.Identity(),
        truth,
        torch.zeros((1, 2, 2, 3)),
        baseline,
        group="chem",
        pulse_call=4,
        device=torch.device("cpu"),
        predictor=predictor,
    )
    assert nonfinite is None
    assert torch.equal(predictions[:4], torch.from_numpy(baseline[:4]))
    assert torch.equal(inputs[4, :8], truth[4, :8])
    assert torch.equal(predictions[4, :8], truth[4, :8] + 1.0)
    assert torch.equal(predictions[5], predictions[4] + 1.0)
    assert activity["truth_injection_count"] == 1
    assert len(timings) == 45


def test_window_and_pulse_comparison_use_only_postpulse_calls() -> None:
    baseline = _summary()
    candidate = _summary(after_call=4)
    assert pulse.mean_grouped_npe_window(
        candidate,
        ("T", "u"),
        start_call=5,
        end_call=49,
    ) == pytest.approx(1.0)
    comparison = pulse.compare_pulse_summary(
        baseline,
        candidate,
        group="chem",
        pulse_call=4,
    )
    assert comparison["primary_postpulse_common_downstream"]["ratio"] == 0.5
    assert comparison["postpulse_partner_group"]["ratio"] == 0.8
    assert comparison["postpulse_own_group"]["ratio"] == 2.0
    assert comparison["temporal_common_downstream"]["immediate"]["ratio"] == 0.5
    assert comparison["temporal_common_downstream"]["late"]["start_call"] == 17
    assert comparison["primary_effect_classification"] == "materially_helpful"


def test_lag_readout_starts_after_pulse() -> None:
    rows = pulse.build_lag_readout(
        _summary(),
        _summary(after_call=12),
        group="rho",
        pulse_call=12,
    )
    assert rows[0]["call"] == 13
    assert rows[0]["lag"] == 1
    assert rows[-1]["call"] == 49
    assert len(rows) == 37


@pytest.mark.parametrize(
    ("chem_to_rho", "rho_to_chem", "complete", "expected"),
    [
        (0.8, 0.7, True, "bidirectional_material_partner_reduction"),
        (0.8, 1.0, True, "chem_to_rho_only_material_partner_reduction"),
        (1.0, 0.8, True, "rho_to_chem_only_material_partner_reduction"),
        (1.0, 1.0, True, "no_material_single_pulse_partner_reduction"),
        (0.8, 0.8, False, "not_interpretable"),
    ],
)
def test_directionality_routing(
    chem_to_rho: float,
    rho_to_chem: float,
    complete: bool,
    expected: str,
) -> None:
    assert (
        pulse.classify_directionality(
            chem_to_rho,
            rho_to_chem,
            complete=complete,
        )
        == expected
    )


def test_cli_help_is_safe(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        pulse.main(["--help"])
    assert exc_info.value.code == 0
    assert "one exact chemistry or density pulse" in capsys.readouterr().out
