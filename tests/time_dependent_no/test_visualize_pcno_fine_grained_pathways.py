from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.time_dependent_no.visualize_pcno_fine_grained_pathways import (
    REQUIRED_RESULT_FILES,
    RESULT_SCHEMA,
    _arm_label,
    _candidate,
    _matrix,
    _verify_results,
    plot_heatmap,
)
from utility.time_dependent_no.pcno_artifacts import sha256_file


def test_multiple_supported_display_scores_only_qualified_candidates() -> None:
    qualified = {
        "arm": "qualified",
        "layer": 1,
        "pairs": [{"median_large_reduction": 0.3}],
    }
    nonqualifying = {
        "arm": "failed_guard",
        "layer": 2,
        "pairs": [{"median_large_reduction": 0.9}],
    }
    arm, layer, decision = _candidate(
        {
            "mechanism_selection": {
                "decision": "multiple_supported",
                "qualified": [qualified],
                "candidates": [qualified, nonqualifying],
            }
        }
    )
    assert (arm, layer, decision) == (
        "qualified",
        "1",
        "best_descriptive_multiple_supported",
    )


def test_pathway_matrix_uses_case_level_medians() -> None:
    rows = [
        {
            "arm": "fourier_quadrature",
            "layer": "0",
            "call": str(call),
            "value": str(value),
        }
        for call, value in ((1, 0.1), (1, 0.3), (5, 0.2), (5, 0.6))
    ]
    labels, calls, matrix = _matrix(rows, row_label=_arm_label, value_field="value")
    assert labels == ["fourier_quadrature@L0"]
    assert calls == [1, 5]
    np.testing.assert_allclose(matrix, [[0.2, 0.4]])


def test_fixed_heatmap_writes_png_pdf_and_saturation(tmp_path: Path) -> None:
    rows = [
        {
            "arm": "fourier_quadrature",
            "layer": "0",
            "call": "1",
            "value": "1.5",
        }
    ]
    paths, saturation = plot_heatmap(
        rows,
        tmp_path / "heatmap",
        row_label=_arm_label,
        value_field="value",
        title="synthetic",
        cmap="RdBu",
        vmin=-1.0,
        vmax=1.0,
    )
    assert {path.suffix for path in paths} == {".png", ".pdf"}
    assert all(path.is_file() for path in paths)
    assert saturation["fraction_outside_limits"] == 1.0


def _v2_result_dir(tmp_path: Path) -> Path:
    results = tmp_path / "results"
    results.mkdir()
    hashes = {}
    for relative in REQUIRED_RESULT_FILES:
        path = results / relative
        path.write_text("column\n", encoding="utf-8")
        hashes[relative] = sha256_file(path)
    (results / "summary.json").write_text(
        json.dumps(
            {
                "schema": RESULT_SCHEMA,
                "status": "complete",
                "scientific_interpretation_allowed": True,
                "output_hashes": hashes,
            }
        ),
        encoding="utf-8",
    )
    return results


def test_verify_v2_results_requires_exact_inventory_and_all_hashes(
    tmp_path: Path,
) -> None:
    results = _v2_result_dir(tmp_path)
    _verify_results(results)
    (results / REQUIRED_RESULT_FILES[-1]).write_text("changed\n", encoding="utf-8")
    with np.testing.assert_raises_regex(ValueError, "digest mismatch"):
        _verify_results(results)


def test_verify_v2_results_rejects_extra_hash_listed_output(tmp_path: Path) -> None:
    results = _v2_result_dir(tmp_path)
    summary_path = results / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    extra = results / "unexpected.csv"
    extra.write_text("column\n", encoding="utf-8")
    summary["output_hashes"][extra.name] = sha256_file(extra)
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with np.testing.assert_raises_regex(ValueError, "inventory mismatch"):
        _verify_results(results)
