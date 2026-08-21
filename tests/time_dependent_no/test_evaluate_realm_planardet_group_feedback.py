from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from scripts.time_dependent_no import (
    evaluate_realm_planardet_group_feedback as feedback,
)


def _summary(scale: float = 1.0) -> dict[str, object]:
    return {
        "call_count": 49,
        "realm_npe_mean": 15.0 * scale,
        "realm_npe_sum_source": 30.0 * scale,
        "npe_group_by_call": {
            "chem": [1.0 * scale, 2.0 * scale],
            "T": [2.0 * scale, 4.0 * scale],
            "rho": [3.0 * scale, 6.0 * scale],
            "u": [4.0 * scale, 8.0 * scale],
            "p": [5.0 * scale, 10.0 * scale],
        },
        "decoded_correlation_case_first": 0.5,
        "admissible_call_count": 1,
        "bounded_call_count": 2,
    }


@pytest.mark.parametrize(
    ("group", "channels"),
    [("chem", range(8)), ("T", (8,)), ("rho", (9,)), ("u", (10, 11))],
)
def test_truth_feedback_changes_only_registered_group(
    group: str, channels: object
) -> None:
    raw = torch.arange(13 * 2 * 3, dtype=torch.float32).reshape(1, 13, 2, 3)
    truth = -torch.arange(13 * 2 * 3, dtype=torch.float32).reshape(13, 2, 3) - 1
    original = raw.clone()

    accepted, stats = feedback.inject_truth_feedback(raw, truth, group=group)

    selected = tuple(channels)
    untouched = tuple(index for index in range(13) if index not in selected)
    assert torch.equal(raw, original)
    assert torch.equal(accepted[:, list(selected)], truth[list(selected)].unsqueeze(0))
    assert torch.equal(accepted[:, list(untouched)], original[:, list(untouched)])
    assert stats["selected_group_truth_exact"] is True
    assert stats["other_channels_bitwise_unchanged"] is True
    assert stats["raw_proposal_bitwise_unchanged"] is True


@pytest.mark.parametrize(
    "raw,truth,group",
    [
        (torch.zeros(1, 12, 2, 2), torch.zeros(12, 2, 2), "T"),
        (torch.zeros(1, 13, 2, 2, dtype=torch.float64), torch.zeros(13, 2, 2), "T"),
        (torch.zeros(1, 13, 2, 2), torch.zeros(13, 2, 3), "T"),
        (torch.zeros(1, 13, 2, 2), torch.zeros(13, 2, 2), "p"),
    ],
)
def test_truth_feedback_rejects_contract_drift(
    raw: torch.Tensor, truth: torch.Tensor, group: str
) -> None:
    with pytest.raises(ValueError):
        feedback.inject_truth_feedback(raw, truth, group=group)


def test_truth_feedback_rejects_nonfinite_state() -> None:
    raw = torch.zeros(1, 13, 2, 2)
    truth = torch.zeros(13, 2, 2)
    raw[0, 0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        feedback.inject_truth_feedback(raw, truth, group="chem")


def test_primary_readout_excludes_intervened_group_and_pmax() -> None:
    baseline = _summary()
    candidate = _summary(0.5)
    candidate["npe_group_by_call"]["chem"] = [1.0e9, 1.0e9]  # type: ignore[index]
    candidate["npe_group_by_call"]["p"] = [2.0e9, 2.0e9]  # type: ignore[index]
    comparison = feedback.compare_feedback_summary(baseline, candidate, group="chem")
    primary = comparison["primary_untouched_non_pMax_groups"]
    assert primary["groups"] == ["T", "rho", "u"]
    assert primary["ratio"] == pytest.approx(0.5)
    assert comparison["primary_effect_classification"] == "materially_helpful"
    assert comparison["raw_own_group"]["ratio"] > 1.0e6
    assert comparison["secondary_untouched_groups_including_pMax"]["ratio"] > 1.0e6


@pytest.mark.parametrize(
    ("ratio", "complete", "expected"),
    [
        (0.90, True, "materially_helpful"),
        (0.94, True, "small_or_inconclusive"),
        (0.95, True, "near_null"),
        (1.05, True, "near_null"),
        (1.06, True, "small_or_inconclusive"),
        (1.10, True, "materially_harmful"),
        (0.5, False, "not_interpretable"),
    ],
)
def test_effect_bands(ratio: float, complete: bool, expected: str) -> None:
    assert feedback.classify_effect(ratio, complete=complete) == expected


def test_preregistration_binds_source_and_metric_contract() -> None:
    payload = feedback.build_preregistration()
    source = feedback.build_diagnostic_source_manifest()
    assert payload["status"] == "frozen_before_gpu_execution"
    assert (
        payload["diagnostic_source_manifest_digest"]
        == source["canonical_payload_sha256"]
    )
    assert payload["metric_contract"]["required_sum_to_mean_factor"] == 49
    assert set(payload["arms"]) == {"chem", "T", "rho", "u"}
    assert payload["sealed_test_object_opened"] is False
    feedback._validate_preregistration(payload)
    changed = json.loads(json.dumps(payload))
    changed["status"] = "changed"
    with pytest.raises(ValueError, match="preregistration"):
        feedback._validate_preregistration(changed)


def test_open_manifest_loader_decodes_mapping_before_validation(
    tmp_path: Path,
) -> None:
    payload = {
        "schema": "realm_huggingface_manifest_v1",
        "repository": "owner/repository",
        "revision": "a" * 40,
        "entries": [
            {
                "path": "data/file.npz",
                "size": 1,
                "oid": "b" * 64,
                "lfs": True,
            }
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    digest = feedback.canonical_json_sha256(payload)

    loaded, repository, revision, entries = feedback._load_open_manifest(
        path, expected_payload_sha256=digest
    )

    assert loaded == payload
    assert repository == "owner/repository"
    assert revision == "a" * 40
    assert len(entries) == 1
    with pytest.raises(ValueError, match="differs"):
        feedback._load_open_manifest(path, expected_payload_sha256="0" * 64)


def test_source_sum_identity_uses_full_horizon() -> None:
    assert feedback._source_sum_identity(
        {"realm_npe_mean": 2.0, "realm_npe_sum_source": 98.0}
    )
    assert not feedback._source_sum_identity(
        {"realm_npe_mean": 2.0, "realm_npe_sum_source": 2.0}
    )


def test_entrypoint_help_is_safe(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        feedback.main(["--help"])
    assert exc.value.code == 0
    assert "oracle field-group feedback" in capsys.readouterr().out
