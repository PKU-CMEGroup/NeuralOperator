from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

import scripts.time_dependent_no.visualize_pcno_response_gain_controller as visual
from utility.time_dependent_no.pcno_artifacts import sha256_file


def _write_csv(path: Path, fieldnames: tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()


def _failed_result(tmp_path: Path) -> Path:
    results = tmp_path / "results"
    results.mkdir()
    fields = {
        "calibration_controller_selections.csv": ("case_id",),
        "evaluation_case_summary.csv": ("case_id",),
        "evaluation_comparisons.csv": ("case_id",),
        "evaluation_gain_selections.csv": (
            "case_id",
            "selected_candidate",
            "selected_gain",
        ),
        "evaluation_probe_replay.csv": ("case_id",),
        "sequence_time_metrics.csv": ("case_id",),
        "visual_payload_inventory.csv": (
            "case_id",
            "relative_path",
            "selected_candidate",
            "selected_gain",
            "sha256",
        ),
    }
    for name, fieldnames in fields.items():
        _write_csv(results / name, fieldnames)
    output_hashes = {
        name: sha256_file(results / name) for name in sorted(fields)
    }
    summary = {
        "schema": "pcno_strength_grouped_response_controller_diagnostic_v1",
        "experiment_contract": "d077_strength_grouped_response_probe",
        "status": "failed_contract",
        "contract_checks_passed": False,
        "scientific_interpretation_allowed": False,
        "family": "dynamic_fv",
        "output_hashes": output_hashes,
    }
    (results / "summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )
    return results


def test_failed_contract_requires_explicit_diagnostic_opt_in(tmp_path: Path) -> None:
    results = _failed_result(tmp_path)
    with pytest.raises(ValueError, match="not eligible"):
        visual._verify_results(results, allow_failed_contract=False)

    summary, payloads, watermark = visual._verify_results(
        results, allow_failed_contract=True
    )
    assert summary["status"] == "failed_contract"
    assert payloads == []
    assert watermark == "FAILED CONTRACT - DIAGNOSTIC ONLY"


def test_complete_d078_identity_is_supported(tmp_path: Path) -> None:
    results = _failed_result(tmp_path)
    summary_path = results / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update(
        {
            "schema": "pcno_deterministic_response_controller_diagnostic_v1",
            "experiment_contract": (
                "d078_deterministic_strength_grouped_response_probe"
            ),
            "status": "complete",
            "contract_checks_passed": True,
            "scientific_interpretation_allowed": True,
        }
    )
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    observed, payloads, watermark = visual._verify_results(
        results, allow_failed_contract=False
    )
    assert observed["experiment_contract"].startswith("d078_")
    assert payloads == []
    assert watermark is None


def _payload(path: Path, *, cumulative_offset: float = 0.0) -> dict:
    calls = 2
    nodes = 4
    shape = (calls, nodes, len(visual.COMPONENTS))
    defect = np.full(shape, 0.01, dtype=np.float32)
    cumulative = np.cumsum(defect, axis=0)
    cumulative[-1, 0, 0] += cumulative_offset
    scale = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    previous = np.concatenate(
        (np.zeros_like(cumulative[:1]), cumulative[:-1]), axis=0
    )
    growth = (2.0 * previous * defect + np.square(defect)) / np.square(
        scale[None, None, :]
    )
    values = {
        "schema": np.asarray(
            "pcno_strength_grouped_response_controller_diagnostic_v1"
        ),
        "family": np.asarray("dynamic_fv"),
        "case_id": np.asarray("case"),
        "resolution": np.asarray("2x2"),
        "selected_candidate": np.asarray("rank8_gain0p5_raw"),
        "selected_rank": np.asarray(8),
        "selected_gain": np.asarray(0.5),
        "selected_correction_policy": np.asarray("raw"),
        "expected_calls": np.asarray(calls),
        "baseline_complete": np.asarray(True),
        "selected_complete": np.asarray(True),
        "component_names": np.asarray(visual.COMPONENTS),
        "nodes": np.asarray(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        ),
        "weights": np.ones(nodes),
        "node_type": np.arange(nodes),
        "physical_times": np.asarray([0.1, 0.2]),
        "residual_scale": scale,
        "true_increment": np.zeros(shape, dtype=np.float32),
        "baseline_defect": np.zeros(shape, dtype=np.float32),
        "corrected_input_base_defect": np.zeros(shape, dtype=np.float32),
        "correction": np.zeros(shape, dtype=np.float32),
        "corrected_defect": defect,
        "cumulative_error": cumulative,
        "signed_growth_density_scaled": growth,
        "signed_growth_contribution": growth / nodes,
        "static_defect": np.zeros(shape, dtype=np.float32),
        "static_cumulative_error": np.zeros(shape, dtype=np.float32),
        "selected_minus_zero_state": np.zeros(shape, dtype=np.float32),
        "selected_minus_static_state": np.zeros(shape, dtype=np.float32),
    }
    np.savez_compressed(path, **values)
    return values


def test_payload_replays_cumulative_defect_and_dynamic_node_types(
    tmp_path: Path,
) -> None:
    path = tmp_path / "payload.npz"
    _payload(path)
    summary = {
        "schema": "pcno_strength_grouped_response_controller_diagnostic_v1",
        "family": "dynamic_fv",
    }
    payload = visual._load_payload(path, summary)
    assert tuple(payload["component_names"]) == visual.COMPONENTS

    _payload(path, cumulative_offset=1e-3)
    with pytest.raises(ValueError, match="cumulative-defect replay"):
        visual._load_payload(path, summary)
