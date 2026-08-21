from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Self

import numpy as np
import pytest

from scripts.time_dependent_no import (
    analyze_pcno_shadow_response_coast_geometry as analysis,
)


def test_synthetic_contract_freezes_all_coast_calls_and_cost() -> None:
    summary = analysis.synthetic_summary()

    assert summary["status"] == "passed"
    assert all(summary["checks"].values())
    assert analysis.SEGMENT_START_CALLS == (8, 10, 12, 14, 16, 18, 20)
    assert analysis.COAST_CALLS == tuple(range(8, 22))
    assert analysis.EXPECTED_SEGMENTS == 28
    assert analysis.EXPECTED_ROWS == 56
    assert analysis.NATIVE_LOGICAL_PREDICTIONS == 112


def test_coast_band_is_total_and_disjoint_on_registered_calls() -> None:
    assert [analysis._coast_band(call) for call in analysis.COAST_CALLS] == [
        *("band_8_14" for _ in range(7)),
        *("band_15_21" for _ in range(7)),
    ]
    with pytest.raises(ValueError, match="outside"):
        analysis._coast_band(7)
    with pytest.raises(ValueError, match="outside"):
        analysis._coast_band(22)


def test_load_segments_selects_active_cases_and_never_requests_truth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = [
        {
            "case_id": case_id,
            "path": f"{case_id}.npz",
            "sha256": "frozen",
            "storage_dtype": "float32_visualization_only",
        }
        for case_id in (*analysis.CASE_IDS, "sv_e12_y04")
    ]
    manifest = {"bundles": rows}
    calls = np.asarray(analysis.a32.ANIMATION_CALLS, dtype=np.int64)

    class Bundle:
        def __init__(self, path: Path) -> None:
            self.case_id = path.stem

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def __getitem__(self, key: str) -> np.ndarray:
            if key == "truth_conservative":
                raise AssertionError("A42 indexed bundled truth")
            raise KeyError(key)

    def state_arrays(bundle: Bundle):
        case_offset = float((*analysis.CASE_IDS, "sv_e12_y04").index(bundle.case_id))
        raw = np.stack(
            [np.full((2, 4), case_offset + call, dtype=np.float32) for call in calls]
        )
        corrected = raw + np.float32(0.25)
        return bundle.case_id, calls, raw, corrected

    monkeypatch.setattr(analysis, "sha256_file", lambda path: "frozen")
    monkeypatch.setattr(analysis.np, "load", lambda path, **kwargs: Bundle(path))
    monkeypatch.setattr(analysis.a40, "_state_arrays_from_bundle", state_arrays)

    manifest_path = tmp_path / "bundle_manifest.json"
    segments = analysis._load_segments(manifest_path, manifest)

    assert len(segments) == analysis.EXPECTED_SEGMENTS
    assert {row["case_id"] for row in segments} == set(analysis.CASE_IDS)
    assert {row["start_call"] for row in segments} == set(analysis.SEGMENT_START_CALLS)
    first = segments[0]
    np.testing.assert_array_equal(first["shadow"], np.full((2, 4), 8.0))
    np.testing.assert_array_equal(first["accepted"], np.full((2, 4), 8.25))
    np.testing.assert_array_equal(first["expected_shadow_end"], np.full((2, 4), 10.0))
    np.testing.assert_array_equal(
        first["expected_accepted_end"], np.full((2, 4), 10.25)
    )


def test_update_baseline_requires_exact_case_call_inventory() -> None:
    zero_rows = [
        {
            "case_id": case_id,
            "input_call": call,
            "update_error_sse": float(call + 1),
        }
        for case_id in analysis.CASE_IDS
        for call in analysis.COAST_CALLS
    ]
    candidate_rows = [
        {"case_id": row["case_id"], "input_call": row["input_call"]}
        for row in zero_rows
    ]

    analysis._attach_update_baseline(zero_rows, candidate_rows)
    assert candidate_rows[0]["update_zero_sse"] == pytest.approx(9.0)
    assert candidate_rows[-1]["update_zero_sse"] == pytest.approx(22.0)

    with pytest.raises(ValueError, match="inventory"):
        analysis._attach_update_baseline(zero_rows[:-1], candidate_rows)


def _score_rows(value: float) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for scope, case_ids in (
        ("e12", analysis.a41.CALIBRATION_CASES),
        ("e14", analysis.a41.TRANSFER_CASES),
    ):
        for band in ("overall", *analysis.COAST_BANDS):
            rows.append(
                {
                    "scope": scope,
                    "band": band,
                    "response_skill": value,
                    "response_skill_status": "ok",
                    "update_skill": value,
                    "update_skill_status": "ok",
                    "median_response_cosine": 0.999,
                    "median_response_cosine_status": "ok",
                    "median_response_norm_ratio": 1.0,
                    "median_response_norm_ratio_status": "ok",
                    "case_metrics": [
                        {
                            "case_id": case_id,
                            "response_skill": value,
                            "response_skill_status": "ok",
                            "update_skill": value,
                            "update_skill_status": "ok",
                            "cosine": 0.999,
                            "cosine_status": "ok",
                        }
                        for case_id in case_ids
                    ],
                }
            )
    return rows


def _candidate_rows(ratio: float) -> list[dict[str, object]]:
    return [
        {
            "case_id": case_id,
            "input_call": call,
            "update_error_to_exact_offset_status": "ok",
            "update_error_to_exact_offset": ratio,
            "exact_offset_rms": 1.0,
            "update_error_rms": ratio,
        }
        for case_id in analysis.CASE_IDS
        for call in analysis.COAST_CALLS
    ]


def test_gate_passes_only_strict_prospective_thresholds() -> None:
    passed = analysis._gate(
        _score_rows(0.999),
        _candidate_rows(0.1),
        structural_checks={"structural": True},
    )
    equality = analysis._gate(
        _score_rows(0.99),
        _candidate_rows(0.5),
        structural_checks={"structural": True},
    )

    assert passed["status"] == "qualified"
    assert all(passed["checks"].values())
    assert equality["status"] == "failed"
    assert "overall_response_and_update_skill" in equality["failed_checks"]
    assert "per_row_tether_error_ratio" in equality["failed_checks"]


def test_gate_allows_only_exact_zero_small_denominator_rows() -> None:
    rows = _candidate_rows(0.1)
    rows[0].update(
        {
            "update_error_to_exact_offset_status": "small_denominator",
            "update_error_to_exact_offset": None,
            "exact_offset_rms": analysis.a41.DENOMINATOR_FLOOR,
            "update_error_rms": analysis.a41.DENOMINATOR_FLOOR,
        }
    )
    exact_zero = analysis._gate(
        _score_rows(0.999), rows, structural_checks={"structural": True}
    )
    rows[0]["update_error_rms"] = 2.0 * analysis.a41.DENOMINATOR_FLOOR
    nonzero_error = analysis._gate(
        _score_rows(0.999), rows, structural_checks={"structural": True}
    )

    assert exact_zero["status"] == "qualified"
    assert nonzero_error["status"] == "failed"
    assert nonzero_error["failed_checks"] == ["per_row_tether_error_ratio"]


def test_frozen_diagonal_matches_a41_result_inventory() -> None:
    values = np.asarray(analysis.FROZEN_DIAGONAL)
    assert values.shape == (len(analysis.a41.FROZEN_ACTIVE_CELLS),)
    assert np.all(values > 0.0)
    assert float(np.min(values)) == pytest.approx(0.7860447848257047)
    assert float(np.max(values)) == pytest.approx(1.0228484831912932)


def test_new_execution_starts_with_no_hidden_fine_or_recurrent_work() -> None:
    execution = analysis._new_execution()
    assert execution["segments"] == 0
    assert execution["response_rows"] == 0
    assert execution["native_logical_predictions"] == 0
    assert execution["fine_logical_predictions"] == 0
    assert execution["fine_actual_forward_passes"] == 0


def test_collect_records_executes_exact_two_step_segment_bookkeeping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shape = (2, 4)
    runtime = SimpleNamespace(
        model=object(),
        sample_by_resolution={analysis.a32.NATIVE_RESOLUTION: {}},
        device=SimpleNamespace(type="cpu"),
        geometry_by_resolution={
            analysis.a32.NATIVE_RESOLUTION: SimpleNamespace(
                node_measures=np.ones(shape[0])
            )
        },
        normalization=SimpleNamespace(
            state_scale=np.ones(shape[1]),
            residual_scale=np.ones(shape[1]),
            gamma=1.4,
        ),
        native_projector=object(),
    )
    calls: list[np.ndarray] = []

    def predict(model, sample, state, **kwargs):
        del model, sample, kwargs
        calls.append(np.array(state, copy=True))
        return np.asarray(state) + 1.0, {
            "forward_seconds": [0.1, 0.1],
            "peak_gpu_memory_bytes": 7,
            "repeat_max_abs": 0.0,
        }

    def step(accepted, shadow, *, predictor, **kwargs):
        del kwargs
        shadow_prediction = predictor(
            analysis.a32.NATIVE_RESOLUTION, np.asarray(shadow)
        )
        accepted_prediction = predictor(
            analysis.a32.NATIVE_RESOLUTION, np.asarray(accepted)
        )
        retained = accepted_prediction - shadow_prediction
        return SimpleNamespace(
            shadow_prediction=shadow_prediction,
            proposal=SimpleNamespace(
                predictions={analysis.a32.NATIVE_RESOLUTION: accepted_prediction}
            ),
            retained_displacement=retained,
            tether_audit=SimpleNamespace(
                maximum_integral_difference_abs=0.0,
                maximum_boundary_difference_abs=0.0,
                maximum_projection_idempotence_abs=0.0,
            ),
            next_native_state=shadow_prediction + retained,
            logical_call_count=2,
            maximum_update_identity_abs=0.0,
        )

    monkeypatch.setattr(analysis, "predict_resolution_sample", predict)
    monkeypatch.setattr(
        analysis, "synchronized_shadow_tethered_binary_affine_step", step
    )
    monkeypatch.setattr(
        analysis.a41,
        "_tether_offset",
        lambda runtime, **kwargs: (
            kwargs["accepted_prediction"] - kwargs["shadow_prediction"],
            {},
        ),
    )
    monkeypatch.setattr(
        analysis.a41,
        "_active_coordinates",
        lambda field, runtime: np.full(19, float(np.mean(field))),
    )
    monkeypatch.setattr(
        analysis,
        "conservative_admissibility_summary",
        lambda state, gamma: {
            "admissible": True,
            "minimum_density": 1.0,
            "minimum_pressure": 1.0,
            "minimum_internal_energy": 1.0,
        },
    )
    segment = {
        "case_id": analysis.CASE_IDS[0],
        "start_call": 8,
        "shadow": np.zeros(shape),
        "accepted": np.full(shape, 0.25),
        "expected_shadow_end": np.full(shape, 2.0),
        "expected_accepted_end": np.full(shape, 2.25),
    }

    records, segments, execution = analysis._collect_records(
        runtime,
        [segment],
        start_coefficients=np.zeros((19, 19)),
        end_coefficients=np.zeros((19, 19)),
    )

    assert [row["input_call"] for row in records] == [8, 9]
    assert all(row["call_order_exact"] for row in records)
    assert len(calls) == 4
    np.testing.assert_array_equal(calls[0], np.zeros(shape))
    np.testing.assert_array_equal(calls[1], np.full(shape, 0.25))
    np.testing.assert_array_equal(calls[2], np.full(shape, 1.0))
    np.testing.assert_array_equal(calls[3], np.full(shape, 1.25))
    assert execution["segments"] == 1
    assert execution["response_rows"] == 2
    assert execution["native_logical_predictions"] == 4
    assert execution["native_actual_forward_passes"] == 8
    assert execution["fine_logical_predictions"] == 0
    assert execution["maximum_shadow_end_serialization_abs"] == pytest.approx(0.0)
    assert execution["maximum_accepted_end_serialization_abs"] == pytest.approx(0.0)
    assert segments == [
        {
            "case_id": analysis.CASE_IDS[0],
            "segment_start_call": 8,
            "segment_end_call": 10,
            "shadow_end_serialization_abs": 0.0,
            "accepted_end_serialization_abs": 0.0,
        }
    ]
