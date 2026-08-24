from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from scripts.time_dependent_no.visualize_pcno_bump_scaling_surface import (
    ANALYSIS_SCHEMA,
    ARCHITECTURES,
    ARTIFACT_SCHEMA,
    COMPARABLE_STEPS,
    COUNTS,
    METRICS,
    RAW_STEPS,
    build_parser,
    load_observed_grid,
    run,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_analysis_packet(root: Path, *, drop_value: bool = False) -> None:
    root.mkdir()
    analysis = {
        "schema": ANALYSIS_SCHEMA,
        "status": "complete",
        "scientific_scope": "single_seed_bump_development_scaling_analysis",
        "historical_test_population_accessed": False,
        "counts": list(COUNTS),
        "architectures": list(ARCHITECTURES),
        "optimizer_steps": RAW_STEPS[-1],
    }
    (root / "analysis.json").write_text(
        json.dumps(analysis, sort_keys=True), encoding="utf-8"
    )
    fields = [
        "trajectory_count",
        "architecture",
        "epoch",
        "optimizer_step",
        "window_equivalent_exposure",
        "online_train_one_step_relative_l2",
        *(name for name, _ in METRICS),
    ]
    with (root / "training_curves.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for architecture_index, architecture in enumerate(ARCHITECTURES):
            for count in COUNTS:
                for epoch, step in enumerate(RAW_STEPS):
                    comparable = step in COMPARABLE_STEPS
                    base = 1.0 + architecture_index + count / 1_000 + step / 1e6
                    row = {
                        "trajectory_count": count,
                        "architecture": architecture,
                        "epoch": epoch,
                        "optimizer_step": step,
                        "window_equivalent_exposure": step / (79 * count),
                        "online_train_one_step_relative_l2": base,
                        "fixed_seen_train_one_step_relative_l2": (
                            base if comparable else ""
                        ),
                        "fixed_validation_one_step_relative_l2": base * 1.1,
                        "rollout_h79_relative_l2": base * 2.0 if comparable else "",
                    }
                    if (
                        drop_value
                        and architecture == "pcno"
                        and count == 8
                        and step == COMPARABLE_STEPS[0]
                    ):
                        row["rollout_h79_relative_l2"] = ""
                    writer.writerow(row)
    files = {}
    for name in ("analysis.json", "training_curves.csv"):
        path = root / name
        files[name] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    manifest = {
        "schema": ARTIFACT_SCHEMA,
        "historical_test_population_accessed": False,
        "files": files,
    }
    (root / "artifact_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )


def test_observed_grid_uses_only_exact_common_checkpoints(tmp_path: Path) -> None:
    analysis_dir = tmp_path / "analysis"
    _write_analysis_packet(analysis_dir)

    observed = load_observed_grid(analysis_dir / "training_curves.csv")

    assert len(observed) == len(ARCHITECTURES) * len(COUNTS) * len(COMPARABLE_STEPS)
    assert ("pcno", 8, 256) in observed
    assert ("pcfno", 256, 20_480) in observed
    assert ("pcno", 8, 512) not in observed
    assert observed[("pcno", 8, 256)]["window_equivalent_exposure"] == 256 / (
        79 * 8
    )


def test_visualizer_writes_observed_maps_ratios_slices_and_manifest(
    tmp_path: Path,
) -> None:
    analysis_dir = tmp_path / "analysis"
    output_dir = tmp_path / "figures"
    _write_analysis_packet(analysis_dir)

    result = run(
        ["--analysis-dir", str(analysis_dir), "--output-dir", str(output_dir)]
    )

    assert result["observed_grid"]["total_points"] == 204
    assert result["observed_grid"]["seed_count"] == 1
    assert result["claim_boundary"]["interpolation_used"] is False
    assert len(result["figures"]) == 6
    assert all((output_dir / record["path"]).stat().st_size > 0 for record in result["figures"])
    assert (output_dir / "observed_surface_points.csv").is_file()
    assert (output_dir / "manifest.json").is_file()


def test_missing_common_rollout_value_fails_closed(tmp_path: Path) -> None:
    analysis_dir = tmp_path / "analysis"
    _write_analysis_packet(analysis_dir, drop_value=True)

    with pytest.raises(ValueError, match="missing or nonnumeric"):
        load_observed_grid(analysis_dir / "training_curves.csv")


def test_parser_has_no_historical_test_surface() -> None:
    destinations = {action.dest for action in build_parser()._actions}

    assert "test" not in destinations
    assert "test_root" not in destinations
    assert {"analysis_dir", "output_dir"} <= destinations
