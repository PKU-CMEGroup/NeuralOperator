from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scripts.time_dependent_no.analyze_pcno_fine_grained_pathways import (
    CALLS,
    CASE_IDS,
)
from scripts.time_dependent_no.analyze_pcno_physical_radius_geometry import (
    MODES,
    VISUAL_CASE_IDS,
)
from scripts.time_dependent_no.analyze_pcno_physical_radius_geometry import (
    SCHEMA as RESULT_SCHEMA,
)
from scripts.time_dependent_no.visualize_pcno_physical_radius_geometry import (
    REQUIRED_RESULT_FILES,
    RESULT_MANIFEST_SCHEMA,
    _expected_payloads,
    _expected_rendered_outputs,
    _load_payload,
    _physical_extent,
    _verify_results,
    animate_payload,
    plot_band_evolution,
    plot_case_signs,
    plot_radius_support_inventory,
)
from utility.time_dependent_no.pcno_artifacts import sha256_file


def _synthetic_payload(path: Path) -> dict[str, np.ndarray]:
    frame_count = len(CALLS)
    nx, ny = 125, 50
    node_count = nx * ny
    x = (np.arange(nx, dtype=np.float64) + 0.5) * (2.0 / nx)
    y = (np.arange(ny, dtype=np.float64) + 0.5) * (1.0 / ny)
    xx, yy = np.meshgrid(x, y)
    arrays: dict[str, np.ndarray] = {
        "schema": np.asarray("pcno_physical_radius_visual_payload_v1"),
        "case_id": np.asarray("sv_e00_y00"),
        "pair": np.asarray("125x50->250x100"),
        "mode": np.asarray("teacher_forced"),
        "resolution": np.asarray("125x50"),
        "nodes": np.column_stack((xx.ravel(), yy.ravel())),
        "weights": np.ones(node_count),
        "node_type": np.zeros(node_count, dtype=np.int64),
        "residual_scale": np.asarray((1.0, 2.0, 3.0, 4.0)),
        "call": np.asarray(CALLS),
        "physical_time": np.asarray((0.02, 0.1, 0.3, 0.6)),
    }
    base = np.arange(frame_count * node_count * 4, dtype=np.float32).reshape(
        frame_count, node_count, 4
    )
    base = 0.01 * (base - base.mean())
    for band_scale, band in ((1.0, "total"), (0.7, "large"), (0.3, "local")):
        arrays[f"A0_{band}"] = band_scale * base
        arrays[f"A1_{band}"] = 0.9 * band_scale * base
        arrays[f"A2_{band}"] = 0.7 * band_scale * base
        arrays[f"A2_minus_A1_{band}"] = -0.2 * band_scale * base
    np.savez_compressed(path, **arrays)
    return arrays


def test_payload_contract_and_fixed_scale_movies(tmp_path: Path) -> None:
    payload = tmp_path / "sv_e00_y00__125x50_to_250x100__teacher_forced.npz"
    _synthetic_payload(payload)
    loaded = _load_payload(payload)
    assert tuple(loaded["call"].tolist()) == CALLS
    _load_payload(payload, expected_residual_scale=(1.0, 2.0, 3.0, 4.0))
    with pytest.raises(ValueError, match="summary residual scales differ"):
        _load_payload(payload, expected_residual_scale=(1.0, 2.0, 3.0, 5.0))
    np.testing.assert_allclose(
        _physical_extent(loaded), (0.0, 2.0, 0.0, 1.0), atol=1.0e-14
    )

    outputs, saturation = animate_payload(
        payload,
        tmp_path / "full.gif",
        tmp_path / "scale_split.gif",
        fps=2,
        dpi=25,
    )
    assert all(path.is_file() for path in outputs)
    with Image.open(outputs[0]) as image:
        assert image.n_frames == len(CALLS)
    with Image.open(outputs[1]) as image:
        assert image.n_frames == len(CALLS)
    assert {row["movie"] for row in saturation} == {"full", "scale_split"}
    assert all(row["fixed_physical_limit"] > 0.0 for row in saturation)


def test_payload_rejects_metadata_geometry_time_weight_and_closure_errors(
    tmp_path: Path,
) -> None:
    wrong_name = tmp_path / "sv_e11_y08__125x50_to_250x100__teacher_forced.npz"
    arrays = _synthetic_payload(wrong_name)
    with pytest.raises(ValueError, match="filename and intrinsic metadata"):
        _load_payload(wrong_name)

    payload = tmp_path / "sv_e00_y00__125x50_to_250x100__teacher_forced.npz"
    damaged = dict(arrays)
    damaged_nodes = damaged["nodes"].copy()
    damaged_nodes[1, 0] = damaged_nodes[0, 0]
    damaged["nodes"] = damaged_nodes
    np.savez_compressed(payload, **damaged)
    with pytest.raises(ValueError, match="structured FV order"):
        _load_payload(payload)

    damaged = dict(arrays)
    damaged["physical_time"] = np.asarray((0.02, 0.1, 0.1, 0.6))
    np.savez_compressed(payload, **damaged)
    with pytest.raises(ValueError, match="physical times"):
        _load_payload(payload)

    damaged = dict(arrays)
    damaged_weights = damaged["weights"].copy()
    damaged_weights[0] = 0.0
    damaged["weights"] = damaged_weights
    np.savez_compressed(payload, **damaged)
    with pytest.raises(ValueError, match="weights"):
        _load_payload(payload)

    damaged = dict(arrays)
    damaged_closure = damaged["A2_minus_A1_large"].copy()
    damaged_closure[0, 0, 0] += 1.0
    damaged["A2_minus_A1_large"] = damaged_closure
    np.savez_compressed(payload, **damaged)
    with pytest.raises(ValueError, match="closure failed"):
        _load_payload(payload)


def test_radius_evolution_and_case_sign_figures_write_png_pdf(
    tmp_path: Path,
) -> None:
    geometry_rows = [
        {
            "resolution": resolution,
            "local_two_hop_radius": str(radius),
            "fixed_training_radius": "0.02",
        }
        for resolution, radius in (
            ("125x50", 0.04),
            ("250x100", 0.02),
            ("500x200", 0.01),
        )
    ]
    operator_rows = []
    for resolution in ("125x50", "250x100", "500x200"):
        for arm in ("A1", "A2"):
            row = {
                "resolution": resolution,
                "arm": arm,
                "neighbor_count_median": "9",
                "support_weight_median": "0.1",
                "changed_row_fraction": "0" if resolution == "250x100" else "0.8",
                "type0_mean_neighbor_count": "9",
                "non_type0_mean_neighbor_count": "6",
            }
            for source_type in (0, 1, 2, 3):
                row[f"target_type_0_source_type_{source_type}_coefficient_share"] = (
                    "0.01" if source_type else "0.97"
                )
            operator_rows.append(row)
    radius_paths = plot_radius_support_inventory(
        geometry_rows, operator_rows, tmp_path / "radius"
    )
    assert {path.suffix for path in radius_paths} == {".png", ".pdf"}

    arm_rows = []
    aggregate_rows = []
    pair = "125x50->250x100"
    mode = MODES[0]
    for case_id in CASE_IDS:
        for arm in ("A0", "A1", "A2"):
            for band in ("total", "large", "transition", "local"):
                aggregate_value = "0.7" if arm == "A2" else "1.0"
                if arm == "A1" and band == "large" and case_id == CASE_IDS[0]:
                    aggregate_value = "0.0"
                aggregate_rows.append(
                    {
                        "case_id": case_id,
                        "pair": pair,
                        "mode": mode,
                        "pathway_level": "decoded_residual",
                        "arm": arm,
                        "band": band,
                        "call_rms_aggregate": aggregate_value,
                    }
                )
                for call in CALLS:
                    value = "0.7" if arm == "A2" else "1.0"
                    if arm == "A1" and band == "large" and call == CALLS[0]:
                        value = "0.0"
                    arm_rows.append(
                        {
                            "case_id": case_id,
                            "pair": pair,
                            "mode": mode,
                            "pathway_level": "decoded_residual",
                            "arm": arm,
                            "band": band,
                            "call": str(call),
                            "defect_scaled_rms": value,
                        }
                    )
    evolution_paths, saturation = plot_band_evolution(
        arm_rows, tmp_path / "evolution", pair=pair, mode=mode
    )
    sign_paths = plot_case_signs(
        aggregate_rows, tmp_path / "signs", pair=pair, mode=mode
    )
    assert {path.suffix for path in evolution_paths + sign_paths} == {".png", ".pdf"}
    assert any(
        row["panel"] == "ratio_large_A2_A1"
        and row["invalid_denominator_count"] == len(CASE_IDS)
        for row in saturation
    )


def _synthetic_result_dir(tmp_path: Path) -> Path:
    results = tmp_path / "results"
    results.mkdir()
    hashes = {}
    for relative in (*REQUIRED_RESULT_FILES, *_expected_payloads()):
        path = results / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"registered\n")
        hashes[relative] = sha256_file(path)
    summary_path = results / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "schema": RESULT_SCHEMA,
                "status": "complete",
                "scientific_interpretation_allowed": True,
                "contract": {
                    "diagnostic_calls": list(CALLS),
                    "modes": list(MODES),
                    "visual_case_ids": list(VISUAL_CASE_IDS),
                },
                "population": {"case_ids": list(CASE_IDS)},
                "output_hashes": hashes,
            }
        ),
        encoding="utf-8",
    )
    manifest_hashes = {**hashes, "summary.json": sha256_file(summary_path)}
    (results / "manifest.json").write_text(
        json.dumps(
            {
                "schema": RESULT_MANIFEST_SCHEMA,
                "status": "complete",
                "result_schema": RESULT_SCHEMA,
                "science_result": True,
                "summary_sha256": sha256_file(summary_path),
                "output_count": len(manifest_hashes),
                "output_hashes": manifest_hashes,
            }
        ),
        encoding="utf-8",
    )
    return results


def test_visualizer_verifies_exact_result_and_payload_inventory(
    tmp_path: Path,
) -> None:
    results = _synthetic_result_dir(tmp_path)
    _verify_results(results)
    target = results / _expected_payloads()[0]
    target.write_bytes(b"changed\n")
    with np.testing.assert_raises_regex(ValueError, "digest mismatch"):
        _verify_results(results)


def test_visualizer_freezes_exact_35_output_names() -> None:
    expected = {
        "radius_support_inventory.png",
        "radius_support_inventory.pdf",
        "visual_scale_saturation_by_call.csv",
    }
    for pair in ("125x50_to_250x100", "250x100_to_500x200"):
        for mode in MODES:
            for stem in ("a2_vs_a1_band_evolution", "a2_vs_a1_case_signs"):
                expected.add(f"{stem}_{pair}_{mode}.png")
                expected.add(f"{stem}_{pair}_{mode}.pdf")
    for relative in _expected_payloads():
        stem = Path(relative).stem
        expected.add(f"{stem}__full.gif")
        expected.add(f"{stem}__scale_split.gif")
    assert len(expected) == 35
    assert _expected_rendered_outputs() == expected
