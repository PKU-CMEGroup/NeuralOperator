from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")


def _load_module():
    root = Path(__file__).resolve().parents[2]
    path = root / "scripts" / "time_dependent_no" / "diagnose_cpg_rollout_mechanisms.py"
    spec = importlib.util.spec_from_file_location(
        "diagnose_cpg_rollout_mechanisms", path
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_result_fixture(path: Path) -> None:
    steps = 3
    nodes = 6
    target = np.ones((steps, nodes, 4), dtype=np.float32)
    prediction = target.copy()
    for step in range(steps):
        target[step, :, 3] = np.array([1.0, 1.0, 4.0, 4.0, 4.0, 4.0])
        prediction[step, :, 3] = np.array([1.0, 1.0, 1.0, 4.0, 4.0, 4.0])
    pos = np.stack(
        (np.arange(nodes, dtype=np.float32), np.zeros(nodes, dtype=np.float32)), axis=-1
    )
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]], dtype=np.int64)
    node_type = np.array([0, 0, 0, 0, 0, 1], dtype=np.int32)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("predicteds", data=prediction)
        handle.create_dataset("targets", data=target)
        handle.create_dataset("pos", data=pos)
        handle.create_dataset("edges", data=edges)
        handle.create_dataset("node_type", data=node_type)


def _artifact_root() -> Path:
    root = Path("artifacts") / "time_dependent_no" / "diagnostic_fixture"
    try:
        root.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        if not root.is_dir():
            pytest.skip("cannot create ignored artifact directory in this environment")
    return root


def test_diagnostic_script_writes_summary_from_combined_hdf5():
    module = _load_module()
    root = _artifact_root()
    result_file = root / "1.h5"
    output_dir = root / "diagnostic"
    _write_result_fixture(result_file)

    module.main(
        [
            "--run",
            f"fixture={result_file}",
            "--output-dir",
            str(output_dir),
            "--quantiles",
            "0.5",
            "--alignment-quantile",
            "0.5",
            "--alignment-grid-size",
            "3",
            "--alignment-max-shift",
            "1.0",
        ]
    )

    summary_path = output_dir / "analysis_summary.json"
    report_path = output_dir / "diagnostic_report.md"
    per_time_path = output_dir / "per_time_metrics.csv"
    assert summary_path.is_file()
    assert report_path.is_file()
    assert per_time_path.is_file()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    run = summary["runs"]["fixture"]
    assert run["trajectory_count"] == 1
    assert run["mean_per_trajectory_overall_rmse"]["pres"] > 0.0
    shock = summary["trajectory_summaries"]["fixture"][0]["shock_quantiles"]["q0.50"]
    assert shock["overlap"]["iou"]["mean"] < 1.0
    assert shock["alignment"]["relative_rmse_reduction"]["mean"] > 0.0
