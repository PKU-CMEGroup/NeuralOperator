from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.time_dependent_no.analyze_pcno_shock_pathways import (
    BRANCHES,
    MASKS,
    SCHEMA,
    _prediction_commutators,
    branch_gains,
    build_common_physical_cases,
    run_analysis,
)
from scripts.time_dependent_no.fit_pcno_shock_representation import (
    build_registered_cases,
    registered_config,
    run_experiment,
)
from utility.time_dependent_no.pcno_shock_representation import (
    build_structured_cell_grid,
    restrict_nested_cell_averages,
)


def test_registered_global_mask_cube_has_exact_spd_semantics() -> None:
    assert MASKS == ("000", "001", "010", "011", "100", "101", "110", "111")
    gains = branch_gains("101", 2)

    assert len(gains) == 2 * len(BRANCHES)
    for layer in range(2):
        assert gains[(layer, "spectral")] == 1.0
        assert gains[(layer, "pointwise")] == 0.0
        assert gains[(layer, "differential")] == 1.0


def test_common_physical_cases_close_under_conservative_restriction() -> None:
    config = registered_config(smoke=False, device_type="cuda")
    coarse = build_structured_cell_grid((32, 16))
    fine = build_structured_cell_grid((64, 32))
    coarse_records = build_common_physical_cases(coarse, config, split="held")
    fine_records = build_common_physical_cases(fine, config, split="held")

    assert [record.position for record in coarse_records] == [
        record.position for record in fine_records
    ]
    for coarse_record, fine_record in zip(coarse_records, fine_records, strict=True):
        restricted = restrict_nested_cell_averages(
            fine_record.case.increment,
            fine_resolution=fine.resolution,
            coarse_resolution=coarse.resolution,
        )
        np.testing.assert_allclose(restricted, coarse_record.case.increment, atol=1e-14)


def test_prediction_commutators_intersect_mixed_native_population() -> None:
    config = registered_config(smoke=False, device_type="cuda")
    grids = {
        "32x16": build_structured_cell_grid((32, 16)),
        "64x32": build_structured_cell_grid((64, 32)),
        "128x64": build_structured_cell_grid((128, 64)),
    }
    held = {
        key: build_common_physical_cases(grid, config, split="held")[:1]
        for key, grid in grids.items()
    }
    train_native = build_registered_cases(grids["64x32"], config, split="train")[:1]
    records = {
        "32x16": held["32x16"],
        "64x32": train_native + held["64x32"],
        "128x64": held["128x64"],
    }
    predictions = {
        f"mask_{mask}_{key}": np.zeros(
            (len(records[key]), grid.ny, grid.nx), dtype=np.float32
        )
        for mask in MASKS
        for key, grid in grids.items()
    }

    rows = _prediction_commutators(predictions, records)

    assert len(rows) == len(MASKS) * 2
    assert all(row["analytic_restriction_floor"] < 1e-12 for row in rows)


def test_p1_smoke_closes_checkpoint_masks_traces_and_visuals(tmp_path: Path) -> None:
    p0_dir = tmp_path / "p0"
    p1_dir = tmp_path / "p1"
    p0_summary = run_experiment(p0_dir, smoke=True, device_name="cpu")
    summary = run_analysis(
        p0_dir,
        p1_dir,
        device_name="cpu",
        smoke=True,
        batch_size=2,
    )

    assert p0_summary["science_result"] is False
    assert summary["schema"] == SCHEMA
    assert summary["science_result"] is False
    assert summary["optimization_performed"] is False
    assert summary["mask_111_native_bypass_max_abs_error"] == 0.0
    assert summary["independent_noop_replay_max_abs_error"] <= 1e-4
    assert summary["direct_batch_to_p0_stored_max_abs_error"] <= 1e-4
    assert summary["differential_trace_replay_max_abs_error"] <= 1e-4
    native_key = "16x8"
    assert set(summary["global_mask_aggregates"][native_key]) == set(MASKS)
    assert len(summary["local_deletion_aggregates"]) == 3
    assert (p1_dir / "branch_mask_predictions.npz").is_file()
    assert (p1_dir / "differential_stage_summaries.json").is_file()
    manifest = json.loads((p1_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["science_result"] is False
    assert manifest["output_count"] == len(manifest["output_hashes"])
    contract = json.loads((p1_dir / "run_contract.json").read_text(encoding="utf-8"))
    assert contract["execution"] == {
        "requested_device": "cpu",
        "resolved_device": "cpu",
        "batch_size": 2,
    }
    assert len(summary["visualizations"]["files"]) == 6
    assert all((p1_dir / name).stat().st_size > 1_000 for name in summary["visualizations"]["files"])
    json.dumps(summary, allow_nan=False)


def test_native_registered_cases_are_not_redefined_by_common_builder() -> None:
    config = registered_config(smoke=False, device_type="cuda")
    grid = build_structured_cell_grid(config["resolution"])
    native = build_registered_cases(grid, config, split="held")
    common = build_common_physical_cases(grid, config, split="held")

    assert [record.case_id for record in native] == [record.case_id for record in common]
    assert [record.position for record in native] == [record.position for record in common]
