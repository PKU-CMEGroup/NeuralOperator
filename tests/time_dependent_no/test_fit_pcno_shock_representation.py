from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts.time_dependent_no.fit_pcno_shock_representation import (
    SCHEMA,
    SOURCE_PATHS,
    _git_state,
    build_registered_cases,
    capacity_summary,
    case_metric_rows,
    registered_config,
    run_experiment,
    sha256_file,
)
from utility.time_dependent_no.pcno_shock_representation import (
    build_structured_cell_grid,
)


def test_registered_production_contract_is_full_pcno_only() -> None:
    config = registered_config(smoke=False, device_type="cuda")
    grid = build_structured_cell_grid(config["resolution"])
    train = build_registered_cases(grid, config, split="train")
    held = build_registered_cases(grid, config, split="held")

    assert config["working_run_id"] == "W26-L2-P0-FULL-S1701"
    assert config["variant"] == "full_pcno"
    assert config["seed"] == 1701
    assert config["steps"] == 20_000
    assert config["width"] == 128
    assert config["block_count"] == 4
    assert config["k_max"] == 8
    assert len(train) == 48
    assert len(held) == 48
    assert len({record.case_id for record in train + held}) == 96
    assert {
        "pcno/__init__.py",
        "pcno/geo_utility.py",
        "pcno/pcno.py",
        "utility/adam.py",
        "utility/losses.py",
        "utility/normalizer.py",
        "utility/time_dependent_no/pcno_fv_geometry.py",
        "utility/time_dependent_no/pcno_resolution_pathways.py",
        "utility/time_dependent_no/pcno_shock_representation.py",
        "utility/time_dependent_no/shock_vortex_coarse_cfd.py",
        "utility/time_dependent_no/shock_vortex_fv.py",
        "scripts/time_dependent_no/fit_pcno_shock_representation.py",
        "scripts/time_dependent_no/visualize_pcno_shock_overfit.py",
    }.issubset(SOURCE_PATHS)


def test_git_state_is_nullable_for_an_isolated_source_snapshot() -> None:
    error = subprocess.CalledProcessError(128, ["git", "rev-parse", "HEAD"])
    with patch(
        "scripts.time_dependent_no.fit_pcno_shock_representation.subprocess.run",
        side_effect=error,
    ):
        state = _git_state()

    assert state == {
        "available": False,
        "head": None,
        "branch": None,
        "status_short": None,
        "unavailable_reason": "CalledProcessError",
    }


def test_exact_predictions_pass_the_fixed_grid_capacity_gate() -> None:
    config = registered_config(smoke=False, device_type="cuda")
    config["families"] = ["step", "pulse"]
    grid = build_structured_cell_grid(config["resolution"])
    train = build_registered_cases(grid, config, split="train")
    held = build_registered_cases(grid, config, split="held")
    train_predictions = np.stack([record.case.increment for record in train])
    held_predictions = np.stack([record.case.increment for record in held])
    rows = case_metric_rows(train, train_predictions, grid)
    rows.extend(case_metric_rows(held, held_predictions, grid))

    decision = capacity_summary(rows, grid_spacing=grid.hx)

    assert decision["fixed_grid_capacity_pass"] is True
    assert decision["train_discontinuous_case_count"] == 24
    assert decision["train_discontinuous_pass_count"] == 24
    assert (
        decision["held_phase_descriptive"]["by_family"]["step"]["ten_of_twelve_gate"]
        is True
    )
    assert (
        decision["held_phase_descriptive"]["by_family"]["pulse"]["ten_of_twelve_gate"]
        is True
    )


def test_smoke_run_writes_parseable_results_checkpoint_and_visuals(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "w26_l2_smoke"

    summary = run_experiment(
        output_dir,
        smoke=True,
        device_name="cpu",
    )

    assert summary["schema"] == SCHEMA
    assert summary["science_result"] is False
    assert summary["variant"] == "full_pcno"
    assert summary["gradient_ablation"]["executed"] is False
    assert summary["completed_steps"] == 2
    assert (output_dir / "checkpoint.pt").is_file()
    assert not (output_dir / "checkpoint_latest.pt").exists()
    assert (output_dir / "predictions.npz").is_file()
    assert (output_dir / "case_metrics.json").is_file()
    json.dumps(summary, allow_nan=False)

    figure_paths = [
        output_dir / relative for relative in summary["visualizations"]["files"]
    ]
    assert {path.suffix for path in figure_paths} == {".pdf", ".png"}
    assert len(figure_paths) == 4
    assert all(path.stat().st_size > 1_000 for path in figure_paths)
    assert (
        next(path for path in figure_paths if path.suffix == ".pdf")
        .read_bytes()
        .startswith(b"%PDF")
    )
    assert (
        next(path for path in figure_paths if path.suffix == ".png")
        .read_bytes()
        .startswith(b"\x89PNG")
    )

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["science_result"] is False
    assert manifest["status"] == "completed"
    assert manifest["output_count"] == len(manifest["output_hashes"])
    for relative, expected in manifest["output_hashes"].items():
        assert sha256_file(output_dir / relative) == expected

    visual_manifest = json.loads(
        (output_dir / "figures" / "visual_manifest.json").read_text(encoding="utf-8")
    )
    assert visual_manifest["style"]["pdf_vector"] is True
    assert visual_manifest["style"]["png_dpi"] == 300
    assert visual_manifest["style"]["common_profile_scales"] is True
