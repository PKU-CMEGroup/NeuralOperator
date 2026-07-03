from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
pytest.importorskip("matplotlib")


def _load_module():
    root = Path(__file__).resolve().parents[2]
    path = root / "scripts" / "time_dependent_no" / "visualize_official_cpg_rollout.py"
    spec = importlib.util.spec_from_file_location("visualize_official_cpg_rollout", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _artifact_root() -> Path:
    root = Path("artifacts") / "time_dependent_no"
    try:
        root.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        if not root.is_dir():
            pytest.skip("cannot create ignored artifact directory in this environment")
    return root


def _write_fixture(root: Path) -> tuple[Path, Path]:
    dataset_file = root / "visualization_fixture_test.h5"
    result_file = root / "visualization_fixture_result.h5"
    with h5py.File(dataset_file, "w") as handle:
        group = handle.create_group("00")
        nodes = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [0.5, 0.5],
            ],
            dtype=np.float32,
        )
        group.create_dataset("pos", data=np.tile(nodes[None, :, :], (4, 1, 1)))
        group.create_dataset(
            "node_type",
            data=np.array([0, 0, 1, 2, 3], dtype=np.int32)[None, :, None].repeat(
                4, axis=0
            ),
        )

    truth = np.ones((3, 5, 4), dtype=np.float32)
    prediction = truth.copy()
    prediction[:, :, 3] += np.linspace(0.0, 0.2, 3, dtype=np.float32)[:, None]
    with h5py.File(result_file, "w") as handle:
        handle.create_dataset("targets", data=truth)
        handle.create_dataset("predicteds", data=prediction)
    return dataset_file, result_file


def test_save_visualization_from_official_rollout_fixture():
    module = _load_module()
    root = _artifact_root()
    dataset_file, result_file = _write_fixture(root)

    summary = module.save_official_rollout_visualization(
        dataset_file=dataset_file,
        result_file=result_file,
        output_dir=root,
        trajectory_index=0,
        feature="pres",
        node_filter="normal",
        no_animation=True,
        snapshot_steps="1,3",
    )

    assert summary["animation_path"] is None
    assert summary["trajectory_key"] == "00"
    assert summary["feature"] == "pres"
    assert summary["num_nodes_plotted"] == 2
    assert Path(summary["relative_error_path"]).is_file()
    assert len(summary["snapshot_paths"]) == 2
    for path in summary["snapshot_paths"]:
        assert Path(path).is_file()


def test_result_geometry_node_count_mismatch_is_rejected():
    module = _load_module()
    root = _artifact_root()
    dataset_file, result_file = _write_fixture(root)
    with h5py.File(dataset_file, "a") as handle:
        del handle["00"]["pos"]
        handle["00"].create_dataset("pos", data=np.zeros((4, 4, 2), dtype=np.float32))

    with pytest.raises(ValueError, match="node count"):
        module.save_official_rollout_visualization(
            dataset_file=dataset_file,
            result_file=result_file,
            output_dir=root,
            no_animation=True,
        )
