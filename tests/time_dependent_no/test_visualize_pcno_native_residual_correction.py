from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

import scripts.time_dependent_no.visualize_pcno_native_residual_correction as visual


def _summary(
    *,
    candidate: str = "rank8_gain1",
    rank: int = 8,
    gain: float | None = None,
    correction_policy: str = "raw",
    experiment_contract: str = "d074_raw",
) -> dict[str, object]:
    if gain is None:
        gain = 0.0 if rank == 0 else 1.0
    return {
        "schema": visual.RESULT_SCHEMA,
        "experiment_contract": experiment_contract,
        "family": "dynamic_fv",
        "status": "smoke_complete",
        "contract_checks_passed": True,
        "scientific_interpretation_allowed": False,
        "selector": {
            "selected": {
                "key": candidate,
                "rank": rank,
                "gain": gain,
                "correction_policy": correction_policy,
            }
        },
    }


def _write_payload(
    path: Path,
    *,
    case_id: str = "case_a",
    candidate: str = "rank8_gain1",
    selected_rank: int | None = None,
    selected_gain: float | None = None,
    selected_policy: str = "raw",
    multiplier: float = 1.0,
    corrupt: str | None = None,
) -> Path:
    if selected_rank is None:
        selected_rank = 0 if candidate == "zero" else 8
    if selected_gain is None:
        selected_gain = 0.0 if selected_rank == 0 else 1.0
    calls, node_count, components = 3, 6, len(visual.COMPONENTS)
    xx, yy = np.meshgrid(np.arange(3), np.arange(2))
    nodes = np.column_stack((xx.ravel(), yy.ravel())).astype(np.float64)
    weights = np.arange(1, node_count + 1, dtype=np.float64)
    scale = np.arange(1, components + 1, dtype=np.float64)
    values = np.arange(1, calls * node_count * components + 1, dtype=np.float64)
    true_increment = multiplier * values.reshape(calls, node_count, components) / 100.0
    base_defect = -0.12 * true_increment
    correction = -0.25 * base_defect
    corrected_defect = base_defect + correction
    cumulative_error = np.cumsum(corrected_defect, axis=0)
    previous = np.concatenate(
        (np.zeros_like(cumulative_error[:1]), cumulative_error[:-1]), axis=0
    )
    growth = (
        2.0 * previous * corrected_defect + np.square(corrected_defect)
    ) / np.square(scale[None, None, :])
    contribution = growth * weights[None, :, None] / weights.sum()
    arrays = {
        "cumulative_error": cumulative_error,
        "signed_growth_density_scaled": growth,
        "signed_growth_contribution": contribution,
    }
    if corrupt is not None:
        arrays[corrupt] = arrays[corrupt].copy()
        arrays[corrupt][1, 2, 3] += 0.1
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        schema=np.asarray(visual.RESULT_SCHEMA),
        family=np.asarray("dynamic_fv"),
        case_id=np.asarray(case_id),
        resolution=np.asarray("3x2"),
        selected_candidate=np.asarray(candidate),
        selected_rank=np.asarray(selected_rank),
        selected_gain=np.asarray(selected_gain),
        selected_correction_policy=np.asarray(selected_policy),
        expected_calls=np.asarray(calls),
        baseline_complete=np.asarray(True),
        selected_complete=np.asarray(True),
        component_names=np.asarray(visual.COMPONENTS),
        nodes=nodes,
        weights=weights,
        node_type=np.asarray([3, 2, 2, 3, 1, 1]),
        physical_times=np.asarray([0.1, 0.2, 0.3]),
        residual_scale=scale,
        true_increment=true_increment.astype(np.float32),
        baseline_defect=(1.1 * base_defect).astype(np.float32),
        corrected_input_base_defect=base_defect.astype(np.float32),
        correction=correction.astype(np.float32),
        corrected_defect=corrected_defect.astype(np.float32),
        cumulative_error=arrays["cumulative_error"].astype(np.float32),
        signed_growth_density_scaled=arrays["signed_growth_density_scaled"].astype(
            np.float32
        ),
        signed_growth_contribution=arrays["signed_growth_contribution"].astype(
            np.float32
        ),
    )
    return path


def _write_csv(
    path: Path, fieldnames: list[str], rows: list[dict[str, object]]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_derived_fields_frame_selection_and_structured_extent(tmp_path: Path) -> None:
    path = _write_payload(tmp_path / "payload.npz")
    payload = visual._load_payload(path, _summary())
    fields = visual._derived_fields(payload)
    scale = payload["residual_scale"][None, None, :]
    assert np.allclose(
        fields["corrected_increment"],
        (payload["true_increment"] + payload["corrected_defect"]) / scale,
    )
    assert np.allclose(
        fields["base_defect"], payload["corrected_input_base_defect"] / scale
    )
    assert visual._frame_indices(6, 4) == (0, 4, 5)
    assert visual._structured_shape("3x2", 6) == (3, 2)
    assert visual._structured_shape("4x2", 6) is None
    assert visual._structured_extent(payload["nodes"], (3, 2)) == (
        -0.5,
        2.5,
        -0.5,
        1.5,
    )


@pytest.mark.parametrize(
    "corrupt",
    ("cumulative_error", "signed_growth_density_scaled", "signed_growth_contribution"),
)
def test_payload_replay_rejects_corruption(tmp_path: Path, corrupt: str) -> None:
    path = _write_payload(tmp_path / f"{corrupt}.npz", corrupt=corrupt)
    with pytest.raises(ValueError, match="replay failed"):
        visual._load_payload(path, _summary())


@pytest.mark.parametrize(
    ("selected_rank", "selected_gain", "match"),
    ((8, 0.0, "rank differs"), (0, 1.0, "gain differs")),
)
def test_zero_payload_binds_frozen_rank_and_gain(
    tmp_path: Path, selected_rank: int, selected_gain: float, match: str
) -> None:
    path = _write_payload(
        tmp_path / "zero_mismatch.npz",
        candidate="zero",
        selected_rank=selected_rank,
        selected_gain=selected_gain,
    )
    with pytest.raises(ValueError, match=match):
        visual._load_payload(path, _summary(candidate="zero", rank=0))


def test_payload_binds_frozen_correction_policy(tmp_path: Path) -> None:
    path = _write_payload(
        tmp_path / "policy_mismatch.npz",
        candidate="rank8_gain0p5_all_integrals_neutral",
        selected_gain=0.5,
        selected_policy="all_integrals_neutral",
    )
    with pytest.raises(ValueError, match="correction policy differs"):
        visual._load_payload(
            path,
            _summary(
                candidate="rank8_gain0p5_all_integrals_neutral",
                gain=0.5,
                correction_policy="energy_integral_neutral",
                experiment_contract="d075_integral_neutral",
            ),
        )


def test_population_limits_are_fixed_across_cases_and_field_groups(
    tmp_path: Path,
) -> None:
    paths = [
        _write_payload(tmp_path / "a.npz", case_id="a", multiplier=1.0),
        _write_payload(tmp_path / "b.npz", case_id="b", multiplier=3.0),
    ]
    limits, selected = visual._population_limits(paths, _summary(), None)
    assert selected == paths
    expected = {group: np.zeros(4) for group in limits}
    for path in paths:
        fields = visual._derived_fields(visual._load_payload(path, _summary()))
        for name, _, group in visual.PANEL_FIELDS:
            expected[group] = np.maximum(
                expected[group], np.max(np.abs(fields[name]), axis=(0, 1))
            )
    for group in limits:
        assert np.array_equal(limits[group], expected[group])
    assert np.all(limits["residual"] >= limits["defect"])


def test_time_join_uses_each_case_physical_clock(tmp_path: Path) -> None:
    _write_csv(
        tmp_path / "case_contracts.csv",
        ["family", "case_id", "resolution", "time_step", "calls"],
        [
            {
                "family": "dynamic_fv",
                "case_id": "a",
                "resolution": "3x2",
                "time_step": 0.025,
                "calls": 2,
            }
        ],
    )
    _write_csv(
        tmp_path / "sequence_time_metrics.csv",
        ["family", "case_id", "resolution", "step"],
        [
            {
                "family": "dynamic_fv",
                "case_id": "a",
                "resolution": "3x2",
                "step": 4,
            }
        ],
    )
    rows = visual._time_joined_sequence_rows(tmp_path)
    assert rows[0]["physical_time"] == pytest.approx(0.1)
    assert visual._minimum_case_calls(tmp_path) == 2


def test_selector_display_medians_include_ineligible_candidates(tmp_path: Path) -> None:
    selector = [
        {
            "candidate": "zero",
            "median_endpoint_state_ratio": "1.0",
            "median_residual_rms_ratio": "1.0",
        },
        {
            "candidate": "rank8_gain0p25",
            "median_endpoint_state_ratio": "",
            "median_residual_rms_ratio": "",
        },
    ]
    _write_csv(
        tmp_path / "selector_rows.csv",
        [
            "candidate",
            "complete",
            "endpoint_state_ratio",
            "residual_rms_ratio",
        ],
        [
            {
                "candidate": "zero",
                "complete": True,
                "endpoint_state_ratio": 1.0,
                "residual_rms_ratio": 1.0,
            },
            {
                "candidate": "zero",
                "complete": True,
                "endpoint_state_ratio": 1.0,
                "residual_rms_ratio": 1.0,
            },
            {
                "candidate": "rank8_gain0p25",
                "complete": True,
                "endpoint_state_ratio": 0.9,
                "residual_rms_ratio": 0.98,
            },
            {
                "candidate": "rank8_gain0p25",
                "complete": True,
                "endpoint_state_ratio": 0.8,
                "residual_rms_ratio": 1.02,
            },
        ],
    )
    medians = visual._selector_display_medians(tmp_path, selector)
    assert medians["zero"] == pytest.approx((1.0, 1.0))
    assert medians["rank8_gain0p25"] == pytest.approx((0.85, 1.0))


def test_result_verifier_requires_exact_hashed_inventory(tmp_path: Path) -> None:
    results = tmp_path / "results"
    payload = _write_payload(
        results / "visual_payloads" / "dynamic_fv_case_a_3x2.npz",
        candidate="zero",
    )
    payload_hash = visual.sha256_file(payload)
    for name in visual.REQUIRED_TABLES - {"visual_payload_inventory.csv"}:
        (results / name).write_text("value\n1\n", encoding="utf-8")
    _write_csv(
        results / "visual_payload_inventory.csv",
        ["relative_path", "sha256"],
        [
            {
                "relative_path": payload.relative_to(results).as_posix(),
                "sha256": payload_hash,
            }
        ],
    )
    output_paths = [results / name for name in visual.REQUIRED_TABLES] + [payload]
    summary = _summary(candidate="zero", rank=0)
    summary["output_hashes"] = {
        path.relative_to(results).as_posix(): visual.sha256_file(path)
        for path in output_paths
    }
    (results / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    verified, paths = visual._verify_results(results)
    assert verified["selector"]["selected"]["key"] == "zero"
    assert paths == [payload.resolve()]
    assert paths[0].relative_to(results.resolve()).as_posix() == (
        "visual_payloads/dynamic_fv_case_a_3x2.npz"
    )

    summary["experiment_contract"] = "d075_integral_neutral"
    (results / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="omit required tables"):
        visual._verify_results(results)
    summary["experiment_contract"] = "d074_raw"
    (results / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    (results / "selector_summary.csv").write_text("corrupt\n", encoding="utf-8")
    with pytest.raises(ValueError, match="digest mismatch"):
        visual._verify_results(results)


def test_smoke_watermark_is_explicit() -> None:
    figure = plt.figure()
    visual._watermark(figure, True)
    assert [text.get_text() for text in figure.texts] == ["H2 SMOKE — NON-SCIENTIFIC"]
    plt.close(figure)
