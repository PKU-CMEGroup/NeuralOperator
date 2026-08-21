from __future__ import annotations

import json
from pathlib import Path

from scripts.time_dependent_no.analyze_pcno_p2_frozen_branch_cube import (
    BRANCH_MASKS,
    SCHEMA,
    branch_mask_name,
    run_analysis,
)
from scripts.time_dependent_no.train_pcno_gradient_ablation import (
    VARIANTS,
    run_experiment,
)


def test_p2_f_smoke_closes_all_three_architectures(tmp_path: Path) -> None:
    for variant in VARIANTS:
        run_experiment(
            tmp_path / f"{variant}_s1701",
            variant=variant,
            seed=1701,
            smoke=True,
            device_name="cpu",
        )

    output_dir = tmp_path / "p2_f"
    summary = run_analysis(
        tmp_path,
        output_dir,
        device_name="cpu",
        batch_size=2,
        smoke=True,
    )
    cube = json.loads((output_dir / "branch_cube_metrics.json").read_text())
    invariance = json.loads(
        (output_dir / "functional_no_gradient_invariance.json").read_text()
    )
    visual = json.loads((output_dir / "visual_manifest.json").read_text())
    manifest = json.loads((output_dir / "manifest.json").read_text())

    assert summary["schema"] == SCHEMA
    assert summary["science_result"] is False
    assert summary["optimization_performed"] is False
    assert summary["checkpoint_count"] == 3
    assert summary["population_count_per_checkpoint"] == 1
    assert summary["branch_mask_count"] == 8
    assert summary["closures"]["all_passed"] is True
    assert summary["closures"]["maximum_all_on_replay_max_abs"] <= 1.0e-5
    assert summary["closures"]["maximum_functional_no_gradient_third_bit_max_abs"] == 0.0
    assert set(cube) == {f"{variant}_s1701" for variant in VARIANTS}
    assert all(
        set(run["populations"]["native"]) == {
            *(branch_mask_name(mask) for mask in BRANCH_MASKS),
            "factorial_effects",
        }
        for run in cube.values()
    )
    assert invariance["no_gradient_s1701"]["native"] == 0.0
    assert len(visual["files"]) == 12
    assert set(visual["files"]) == set(visual["sha256"])
    assert manifest["output_count"] == len(manifest["output_hashes"])
    json.dumps(summary, allow_nan=False)
