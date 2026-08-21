from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from scripts.time_dependent_no import evaluate_realm_planardet_pmax_projection as pmax
from utility.time_dependent_no.realm_benchmark import canonical_json_sha256
from utility.time_dependent_no.realm_planardet import sha256_file


def _summary(scale: float = 1.0) -> dict[str, object]:
    return {
        "npe_group_by_call": {
            "T": [1.0 * scale, 2.0 * scale],
            "chem": [2.0 * scale, 3.0 * scale],
            "p": [100.0 * scale, 200.0 * scale],
            "rho": [3.0 * scale, 4.0 * scale],
            "u": [4.0 * scale, 5.0 * scale],
        }
    }


def test_projection_changes_only_pmax_and_reports_exact_activity() -> None:
    proposal = torch.arange(78, dtype=torch.float32).reshape(1, 13, 2, 3)
    current_pmax_pa = torch.tensor([[[90.0, 120.0, 130.0], [140.0, 150.0, 160.0]]])
    proposal[:, 12] = torch.tensor([[[-2.0, 3.0, 1.0], [4.0, 8.0, 0.0]]])
    original = proposal.clone()

    projected, stats = pmax.project_pmax_nondecreasing(
        proposal,
        current_pmax_pa,
        pmax_mean_pa=100.0,
        pmax_scale_pa=10.0,
    )

    assert torch.equal(projected[:, :12], original[:, :12])
    assert torch.equal(proposal, original)
    assert torch.equal(
        projected[:, 12],
        torch.tensor([[[-1.0, 3.0, 3.0], [4.0, 8.0, 6.0]]]),
    )
    assert stats["corrected_cells"] == 3
    assert stats["total_cells"] == 6
    assert stats["corrected_fraction"] == pytest.approx(0.5)
    assert stats["correction_sum_pa"] == pytest.approx(90.0)
    assert stats["correction_max_pa"] == pytest.approx(60.0)
    assert stats["correction_mean_active_pa"] == pytest.approx(30.0)
    assert stats["encoded_floor_nextafter_updates"] == 0
    assert stats["encoded_floor_max_nextafter_steps"] == 0
    assert stats["other_channels_bitwise_unchanged"] is True
    assert stats["projected_pMax_nondecreasing"] is True


def test_projection_noop_is_bitwise_exact() -> None:
    proposal = torch.ones(2, 13, 2, 2)
    current_pmax_pa = torch.full((2, 2, 2), 5.0)
    projected, stats = pmax.project_pmax_nondecreasing(
        proposal,
        current_pmax_pa,
        pmax_mean_pa=5.0,
        pmax_scale_pa=2.0,
    )
    assert torch.equal(projected, proposal)
    assert stats["corrected_cells"] == 0
    assert stats["correction_mean_active_pa"] == 0.0


def test_projection_closes_float32_physical_round_trip_upward() -> None:
    proposal = torch.zeros(1, 13, 1, 1)
    proposal[:, 12] = -3.0
    current_pmax_pa = torch.tensor([[[-162_140.0]]])

    projected, stats = pmax.project_pmax_nondecreasing(
        proposal,
        current_pmax_pa,
        pmax_mean_pa=100_000.0,
        pmax_scale_pa=100_000.0,
    )
    decoded = projected[:, 12] * 100_000.0 + 100_000.0

    assert bool((decoded >= current_pmax_pa).all())
    assert stats["encoded_floor_nextafter_updates"] == 1
    assert stats["encoded_floor_max_nextafter_steps"] == 1
    assert stats["projected_pMax_nondecreasing"] is True


@pytest.mark.parametrize(
    ("proposal", "current_pmax_pa", "scale"),
    (
        (torch.zeros(13, 2, 2), torch.zeros(1, 2, 2), 1.0),
        (torch.zeros(1, 12, 2, 2), torch.zeros(1, 2, 2), 1.0),
        (torch.zeros(1, 13, 2, 2), torch.zeros(1, 2, 3), 1.0),
        (
            torch.zeros(1, 13, 2, 2),
            torch.zeros(1, 2, 2, dtype=torch.float64),
            1.0,
        ),
        (torch.zeros(1, 13, 2, 2), torch.zeros(1, 2, 2), 0.0),
    ),
)
def test_projection_rejects_contract_drift(
    proposal: torch.Tensor, current_pmax_pa: torch.Tensor, scale: float
) -> None:
    with pytest.raises(ValueError):
        pmax.project_pmax_nondecreasing(
            proposal,
            current_pmax_pa,
            pmax_mean_pa=0.0,
            pmax_scale_pa=scale,
        )


def test_projection_rejects_nonfinite_states() -> None:
    proposal = torch.zeros(1, 13, 2, 2)
    current_pmax_pa = torch.zeros(1, 2, 2)
    proposal[0, 12, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        pmax.project_pmax_nondecreasing(
            proposal,
            current_pmax_pa,
            pmax_mean_pa=0.0,
            pmax_scale_pa=1.0,
        )


def test_non_pmax_readout_excludes_pressure_group() -> None:
    assert pmax.mean_non_pmax_grouped_npe(_summary()) == pytest.approx(12.0)
    changed = _summary()
    changed["npe_group_by_call"]["p"] = [1.0e9, 2.0e9]  # type: ignore[index]
    assert pmax.mean_non_pmax_grouped_npe(changed) == pytest.approx(12.0)


def test_startup_environment_must_precede_python(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    with pytest.raises(RuntimeError, match="before Python startup"):
        pmax._validate_startup_environment()
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    pmax._validate_startup_environment()


@pytest.mark.parametrize(
    ("ratio", "complete", "expected"),
    (
        (0.90, True, "materially_helpful"),
        (1.10, True, "materially_harmful"),
        (0.95, True, "near_null"),
        (1.05, True, "near_null"),
        (0.93, True, "small_or_inconclusive"),
        (1.07, True, "small_or_inconclusive"),
        (0.5, False, "not_interpretable"),
    ),
)
def test_effect_classification_is_frozen(
    ratio: float, complete: bool, expected: str
) -> None:
    assert pmax.classify_non_pmax_effect(ratio, complete=complete) == expected


def test_preregistration_binds_source_baseline_and_causal_readout() -> None:
    payload = pmax.build_preregistration()
    unsigned = {
        key: value
        for key, value in payload.items()
        if key != "canonical_payload_sha256"
    }
    assert payload["canonical_payload_sha256"] == canonical_json_sha256(unsigned)
    assert payload["baseline"]["checkpoint_sha256"] == (pmax.BASELINE_CHECKPOINT_SHA256)
    assert payload["intervention"]["changed_channels"] == [12]
    assert payload["corrects_failed_attempt"]["result_sha256"] == (
        pmax.FAILED_P0_RESULT_SHA256
    )
    assert payload["runtime"]["CUBLAS_WORKSPACE_CONFIG"].endswith(
        "before Python startup"
    )
    assert "excluding the pMax group" in payload["primary_causal_readout"]
    source = pmax.build_projection_source_manifest()
    assert (
        payload["projection_source_manifest_digest"]
        == source["canonical_payload_sha256"]
    )
    assert source["files"][0]["sha256"] == sha256_file(Path(pmax.__file__))
    pmax._validate_projection_preregistration(payload)

    changed = copy.deepcopy(payload)
    changed["effect_bands"]["materially_helpful_ratio_at_most"] = 0.91
    with pytest.raises(ValueError, match="preregistration"):
        pmax._validate_projection_preregistration(changed)


def test_projection_entrypoint_help_is_safe(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        pmax.main(["--help"])
    assert exc_info.value.code == 0
    assert "usage:" in capsys.readouterr().out
