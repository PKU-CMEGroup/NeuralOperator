from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from scripts.time_dependent_no import (
    visualize_realm_planardet_group_feedback as visualize,
)
from utility.time_dependent_no.realm_benchmark import canonical_json_sha256


def _summary(scale_by_group: dict[str, float] | None = None) -> dict[str, object]:
    scales = scale_by_group or {}
    grouped = {
        group: (np.arange(1, 50, dtype=float) * scales.get(group, 1.0)).tolist()
        for group in visualize.ALL_METRIC_GROUPS
    }
    total = np.sum(list(grouped.values()), axis=0)
    return {
        "npe_group_by_call": grouped,
        "npe_total_by_call": total.tolist(),
        "decoded_correlation_by_call": np.linspace(1.0, 0.5, 49).tolist(),
    }


def _result() -> dict[str, object]:
    baseline = _summary()
    arms = {}
    for arm_index, arm in enumerate(visualize.ARM_ORDER, start=1):
        summary = _summary(
            {group: float(arm_index) for group in visualize.ALL_METRIC_GROUPS}
        )
        primary_groups = [group for group in visualize.ARM_ORDER if group != arm]
        arms[arm] = {
            "raw_summary": summary,
            "comparison_to_unintervened_free_baseline": {
                "primary_untouched_non_pMax_groups": {
                    "groups": primary_groups,
                    "ratio": float(arm_index),
                }
            },
        }
    payload: dict[str, object] = {
        "schema": visualize.RESULT_SCHEMA,
        "baseline_free_summary": baseline,
        "arms": arms,
        "diagnostic_interpretation_allowed": True,
        "evaluator_closure": {"all_gates_pass": True},
        "test_object_opened": False,
    }
    payload["canonical_payload_sha256"] = canonical_json_sha256(payload)
    return payload


def test_plot_data_preserves_group_and_time_ratios() -> None:
    data = visualize.build_plot_data(_result())
    expected = np.arange(1, 5, dtype=float)[:, None] * np.ones((4, 5))
    np.testing.assert_allclose(data["group_ratio_matrix"], expected)
    for index, arm in enumerate(visualize.ARM_ORDER, start=1):
        assert data["primary_ratios"][arm] == float(index)
        np.testing.assert_allclose(data["primary_ratio_by_call"][arm], index)
    np.testing.assert_allclose(data["total_by_call"]["baseline"], np.arange(1, 50) * 5)


@pytest.mark.parametrize(
    "mutation",
    ["schema", "closure", "interpretation", "test", "digest"],
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
    else:
        result["canonical_payload_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="identity or closure"):
        visualize.build_plot_data(result)


def test_plot_data_detects_tampering_after_digest() -> None:
    result = _result()
    changed = copy.deepcopy(result)
    changed["arms"]["chem"]["raw_summary"]["npe_total_by_call"][0] = 9.0  # type: ignore[index]
    with pytest.raises(ValueError, match="identity or closure"):
        visualize.build_plot_data(changed)


def test_entrypoint_help_is_safe(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        visualize.main(["--help"])
    assert exc.value.code == 0
    assert "field-group recurrence" in capsys.readouterr().out


def test_manifest_json_is_finite_serializable() -> None:
    result = _result()
    serialized = json.dumps(result, allow_nan=False)
    assert "NaN" not in serialized
