from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

from scripts.time_dependent_no.plot_pcno_boundary_field_results import render


def test_result_figure_renders_from_declared_training_and_intervention_rows(
    tmp_path: Path,
) -> None:
    metrics = (
        "H30_relative_l2",
        "selected_one_step_relative_l2",
        "selected_one_step_boundary_relative_l2",
        "selected_one_step_normal_relative_l2",
    )
    records = []
    aggregate = []
    means = {"N0": 0.008, "G1": 0.0084, "S1": 0.0078}
    for arm in ("N0", "G1", "S1"):
        for seed_index, seed in enumerate((20260718, 20260719, 20260720)):
            records.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "H30_relative_l2": means[arm] + seed_index * 0.0001,
                }
            )
        for metric_index, metric in enumerate(metrics):
            aggregate.append(
                {
                    "arm": arm,
                    "metric": metric,
                    "mean": means[arm] / (metric_index + 1),
                }
            )
    training = tmp_path / "training.json"
    training.write_text(
        json.dumps({"records": records, "aggregate": aggregate}), encoding="utf-8"
    )

    endpoint_summary = []
    for intervention, value in (
        ("correct", 0.0078),
        ("zero_all", 0.024),
        ("zero_y_symmetry", 0.022),
        ("zero_x_extrapolation", 0.011),
    ):
        endpoint_summary.append(
            {
                "arm": "S1",
                "intervention": intervention,
                "call": 30,
                "mean_relative_l2": value,
            }
        )
    evaluation = tmp_path / "evaluation.json"
    evaluation.write_text(
        json.dumps(
            {
                "family": "dynamic_fv",
                "contract": {
                    "endpoints": [20, 30],
                    "semantic_names": ["y_symmetry", "x_extrapolation"],
                },
                "aggregate": {"endpoint_summary": endpoint_summary},
            }
        ),
        encoding="utf-8",
    )
    outcomes = tmp_path / "outcomes.csv"
    rows = []
    for case_index in range(2):
        for region_index, region in enumerate(
            ("boundary_distance_le_0.05", "shock", "vortex", "smooth"), 1
        ):
            rows.append(
                {
                    "case_id": f"case_{case_index}",
                    "mode": "free_rollout",
                    "call": 30,
                    "arm": "S1",
                    "intervention": "zero_all",
                    "region": region,
                    "state_gap_to_arm_correct_rms": 0.01 * region_index,
                }
            )
    with outcomes.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    output = tmp_path / "summary"
    manifest = render(
        SimpleNamespace(
            training_analysis=training,
            evaluation_summary=evaluation,
            outcomes_csv=outcomes,
            output_stem=output,
            overwrite=False,
        )
    )
    assert manifest["schema"] == "pcno_boundary_field_result_figure_v1"
    assert output.with_suffix(".png").is_file()
    assert output.with_suffix(".pdf").is_file()
    assert (tmp_path / "summary_manifest.json").is_file()
