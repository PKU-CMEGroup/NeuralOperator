from __future__ import annotations

import copy

import numpy as np
import pytest

from scripts.time_dependent_no import visualize_realm_planardet_group_pulse as visualize
from utility.time_dependent_no.realm_benchmark import canonical_json_sha256


def _summary(scale: float = 1.0) -> dict[str, object]:
    return {"npe_total_by_call": (np.arange(1, 50, dtype=float) * scale).tolist()}


def _result() -> dict[str, object]:
    baseline = _summary()
    arms = {}
    for group, pulse_call in visualize.ARM_SPECS:
        scale = 0.8 if group == "chem" else 0.9
        total = np.arange(1, 50, dtype=float)
        total[pulse_call:] *= scale
        ratio = scale + pulse_call / 1000.0
        lag_rows = [
            {
                "call": call,
                "lag": call - pulse_call,
                "all_group_ratio": ratio,
                "common_downstream_ratio": ratio,
                "own_group_ratio": ratio,
                "partner_ratio": ratio,
            }
            for call in range(pulse_call + 1, 50)
        ]
        arms[visualize.arm_id(group, pulse_call)] = {
            "group": group,
            "pulse_call": pulse_call,
            "valid_length": 49,
            "nonfinite_proposal_call": None,
            "closure": {"all_gates_pass": True},
            "raw_summary": {"npe_total_by_call": total.tolist()},
            "comparison_to_unintervened_free_baseline": {
                "primary_postpulse_common_downstream": {"ratio": ratio},
                "postpulse_partner_group": {"ratio": ratio + 0.01},
                "postpulse_own_group": {"ratio": ratio + 0.02},
                "all_group_realm_npe_mean": {"ratio": ratio + 0.03},
                "temporal_common_downstream": {
                    window: {"ratio": ratio + index / 100.0}
                    for index, window in enumerate(visualize.WINDOW_ORDER)
                },
            },
            "postpulse_lag_readout": [
                {**row, "partner_ratio": row["partner_ratio"] + 0.01}
                for row in lag_rows
            ],
        }
    payload: dict[str, object] = {
        "schema": visualize.RESULT_SCHEMA,
        "baseline_free_summary": baseline,
        "arms": arms,
        "directionality_by_pulse": {
            str(call): {
                "chem_to_rho_ratio": arms[visualize.arm_id("chem", call)][
                    "comparison_to_unintervened_free_baseline"
                ]["postpulse_partner_group"]["ratio"],
                "rho_to_chem_ratio": arms[visualize.arm_id("rho", call)][
                    "comparison_to_unintervened_free_baseline"
                ]["postpulse_partner_group"]["ratio"],
                "classification": "synthetic",
            }
            for call in visualize.PULSE_CALLS
        },
        "diagnostic_interpretation_allowed": True,
        "evaluator_closure": {"all_gates_pass": True},
        "test_object_opened": False,
    }
    payload["canonical_payload_sha256"] = canonical_json_sha256(payload)
    return payload


def test_plot_data_preserves_registered_effects_and_lags() -> None:
    data = visualize.build_plot_data(_result())
    for group, pulse_call in visualize.ARM_SPECS:
        expected = (0.8 if group == "chem" else 0.9) + pulse_call / 1000.0
        assert data["primary_ratio"][(group, pulse_call)] == expected
        assert data["partner_ratio"][(group, pulse_call)] == expected + 0.01
        assert data["window_ratio"][(group, pulse_call)]["middle"] == expected + 0.02
        response = data["lag_response"][(group, pulse_call)]
        np.testing.assert_array_equal(response["lag"], np.arange(1, 50 - pulse_call))
        np.testing.assert_allclose(response["common_downstream_ratio"], expected)
        np.testing.assert_allclose(response["partner_ratio"], expected + 0.01)
        np.testing.assert_array_equal(
            data["total_by_call"][(group, pulse_call)][:pulse_call],
            data["baseline_total_by_call"][:pulse_call],
        )


@pytest.mark.parametrize(
    "mutation",
    ["schema", "closure", "interpretation", "test", "digest", "arm_closure"],
)
def test_plot_data_rejects_result_contract_drift(mutation: str) -> None:
    result = _result()
    if mutation == "schema":
        result["schema"] = "changed"
    elif mutation == "closure":
        result["evaluator_closure"]["all_gates_pass"] = False  # type: ignore[index]
    elif mutation == "interpretation":
        result["diagnostic_interpretation_allowed"] = False
    elif mutation == "test":
        result["test_object_opened"] = True
    elif mutation == "arm_closure":
        result["arms"]["chem_call4"]["closure"]["all_gates_pass"] = False  # type: ignore[index]
        unsigned = {
            key: value
            for key, value in result.items()
            if key != "canonical_payload_sha256"
        }
        result["canonical_payload_sha256"] = canonical_json_sha256(unsigned)
    else:
        result["canonical_payload_sha256"] = "0" * 64
    match = (
        "arm identity or closure"
        if mutation == "arm_closure"
        else "identity or closure"
    )
    with pytest.raises(ValueError, match=match):
        visualize.build_plot_data(result)


def test_plot_data_detects_tampering_after_digest() -> None:
    result = _result()
    changed = copy.deepcopy(result)
    changed["arms"]["chem_call12"]["postpulse_lag_readout"][0][  # type: ignore[index]
        "partner_ratio"
    ] = 9.0
    with pytest.raises(ValueError, match="identity or closure"):
        visualize.build_plot_data(changed)


def test_entrypoint_help_is_safe(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        visualize.main(["--help"])
    assert exc.value.code == 0
    assert "one-call field-group pulse" in capsys.readouterr().out
