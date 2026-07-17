import argparse
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from scripts.time_dependent_no.euler1d_weno_hllc_ader_dataset import (
    CaseConfig,
    conservative_to_primitive,
    generate_dataset,
    initialize_case,
)
from scripts.time_dependent_no.build_euler1d_restriction_family import (
    exact_initial_average_error,
)
from scripts.time_dependent_no.train_euler1d_multiresolution import (
    batched_rollout_selection_summary,
    balanced_batch_counts,
    evaluate_one_step_selection,
    make_balanced_loader,
)
from scripts.time_dependent_no.train_euler1d_target_ladder import (
    PrimitiveNormalizer,
    evaluate_one_step,
    rollout_cases,
    summarize_rollouts,
)
from utility.time_dependent_no.euler1d_data import (
    Euler1DNPZ,
    Euler1DTimePairDataset,
    collate_euler1d_pairs,
    load_euler1d_npz,
    restrict_euler1d_source,
    restriction_commutation_metrics,
    save_euler1d_npz,
)
from utility.time_dependent_no.euler1d_targets import make_target_adapter


def _fine_source() -> Euler1DNPZ:
    case = CaseConfig(
        x_left=0.0,
        x_right=1.0,
        x_disc=0.3,
        left_state=np.array([1.0, 0.4, 1.2]),
        right_state=np.array([0.2, -0.1, 0.15]),
        t_final=0.2,
    )
    x, initial, _ = initialize_case(
        case,
        nx=8,
        gamma=1.4,
        ng=3,
        initialization_mode="exact_cell_average",
    )
    primitive = conservative_to_primitive(initial[3:-3], 1.4)
    frames = np.stack(
        (
            primitive,
            primitive + np.array([0.01, 0.005, 0.02]),
            primitive + np.array([0.02, 0.010, 0.04]),
        )
    ).astype(np.float32)
    return Euler1DNPZ(
        data=np.stack((frames, frames)),
        x=np.stack((x, x)).astype(np.float32),
        t=np.broadcast_to(np.array([0.0, 0.1, 0.2], dtype=np.float32), (2, 3)).copy(),
        left_states=np.stack((case.left_state, case.left_state)).astype(np.float32),
        right_states=np.stack((case.right_state, case.right_state)).astype(np.float32),
        gamma=1.4,
        metadata={
            "domains": np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32),
            "x_disc": np.array([0.3, 0.3], dtype=np.float32),
            "t_final": np.array([0.2, 0.2], dtype=np.float32),
            "initialization_mode": np.array("exact_cell_average"),
            "nx": np.array(8, dtype=np.int32),
        },
    )


def test_exact_cell_average_initialization_commutes_with_nested_restriction():
    case = CaseConfig(
        x_left=0.0,
        x_right=1.0,
        x_disc=0.3,
        left_state=np.array([1.0, 0.5, 1.0]),
        right_state=np.array([0.25, -0.2, 0.1]),
        t_final=0.1,
    )
    _, coarse, _ = initialize_case(
        case,
        nx=4,
        gamma=1.4,
        ng=3,
        initialization_mode="exact_cell_average",
    )
    _, fine, _ = initialize_case(
        case,
        nx=8,
        gamma=1.4,
        ng=3,
        initialization_mode="exact_cell_average",
    )

    restricted = fine[3:-3].reshape(4, 2, 3).mean(axis=1)

    np.testing.assert_allclose(restricted, coarse[3:-3], rtol=1.0e-12, atol=1.0e-12)
    left_conservative = coarse[3]
    right_conservative = coarse[5]
    np.testing.assert_allclose(
        coarse[4],
        0.2 * left_conservative + 0.8 * right_conservative,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_restricted_source_round_trip_and_increment_commutation(tmp_path: Path):
    fine = _fine_source()
    coarse = restrict_euler1d_source(fine, 4)
    path = tmp_path / "restricted.npz"

    save_euler1d_npz(path, coarse)
    restored = load_euler1d_npz(path)
    metrics = restriction_commutation_metrics(
        restored,
        fine,
        strides=(1, 2),
    )

    assert restored.num_cells == 4
    assert restored.data.dtype == np.float64
    assert (
        Euler1DTimePairDataset(restored)[0]["current_primitive"].dtype == torch.float32
    )
    assert restored.metadata["restriction_source_nx"] == 8
    assert metrics["state_global_relative_l2"] < 1.0e-6
    assert all(item["global_relative_l2"] < 1.0e-6 for item in metrics["updates"])


def test_balanced_loader_keeps_batches_homogeneous_and_budget_exact():
    datasets = {
        "nx4": [torch.full((4,), float(index)) for index in range(20)],
        "nx8": [torch.full((8,), float(index)) for index in range(20)],
        "nx16": [torch.full((16,), float(index)) for index in range(20)],
    }
    loader, counts = make_balanced_loader(
        datasets,
        total_samples=24,
        batch_size=4,
        generator=torch.Generator().manual_seed(7),
        collate_fn=torch.stack,
    )

    shapes = [tuple(batch.shape) for batch in loader]

    assert counts == {"nx4": 2, "nx8": 2, "nx16": 2}
    assert shapes == [
        (4, 4),
        (4, 8),
        (4, 16),
        (4, 4),
        (4, 8),
        (4, 16),
    ]
    assert balanced_batch_counts(8, list(datasets), offset=1) == {
        "nx4": 2,
        "nx8": 3,
        "nx16": 3,
    }


def test_parallel_dataset_generation_matches_serial(tmp_path: Path):
    serial_path = tmp_path / "serial.npz"
    parallel_path = tmp_path / "parallel.npz"
    common = {
        "n_cases": 2,
        "n_steps": 2,
        "nx": 8,
        "t_final": 0.02,
        "cfl": 0.2,
        "seed": 2401,
        "verbose": False,
        "initialization_mode": "exact_cell_average",
        "storage_dtype": "float64",
    }

    generate_dataset(str(serial_path), num_workers=1, **common)
    generate_dataset(str(parallel_path), num_workers=2, **common)

    with (
        np.load(serial_path, allow_pickle=False) as serial,
        np.load(parallel_path, allow_pickle=False) as parallel,
    ):
        compared = (
            "data",
            "x",
            "t",
            "left_states",
            "right_states",
            "domains",
            "x_disc",
            "t_final",
            "fallback_counts",
        )
        for key in compared:
            np.testing.assert_array_equal(parallel[key], serial[key])
        assert serial["data"].dtype == np.float64

    source = load_euler1d_npz(serial_path)
    assert exact_initial_average_error(source)["relative_l2_max"] < 1.0e-8


class _ZeroResidual(torch.nn.Module):
    def forward(self, batch):
        return torch.zeros_like(batch.current_primitive)


def test_batched_selection_rollout_matches_casewise_rollout():
    source = _fine_source()
    cases = np.array([0, 1], dtype=np.int64)
    args = argparse.Namespace(
        rollout_steps=2,
        step_stride=1,
        rollout_final_frame=2,
        recurrent_coordinates="conservative",
    )
    model = _ZeroResidual()
    adapter = make_target_adapter("residual", positive_transform="none")

    batched = batched_rollout_selection_summary(
        model,
        adapter,
        source,
        cases,
        args,
        torch.device("cpu"),
    )
    casewise = summarize_rollouts(
        rollout_cases(model, adapter, source, cases, args, torch.device("cpu"))
    )

    for key in (
        "finite",
        "admissible",
        "num_cases",
        "num_completed_cases",
        "completion_fraction",
        "survival_fraction_mean",
        "survival_fraction_min",
        "completed_horizon",
    ):
        assert batched[key] == casewise[key]
    np.testing.assert_allclose(
        batched["rollout_relative_l2_mean"],
        casewise["rollout_relative_l2_mean"],
        rtol=1.0e-7,
        atol=1.0e-9,
    )
    np.testing.assert_allclose(
        batched["rollout_relative_l2_final"],
        casewise["rollout_relative_l2_final"],
        rtol=1.0e-7,
        atol=1.0e-9,
    )


def test_deferred_selection_evaluation_matches_full_evaluation():
    source = _fine_source()
    loader = DataLoader(
        Euler1DTimePairDataset(source),
        batch_size=2,
        shuffle=False,
        collate_fn=collate_euler1d_pairs,
    )
    model = _ZeroResidual()
    adapter = make_target_adapter("residual", positive_transform="none")
    normalizer = PrimitiveNormalizer(
        mean=torch.zeros(1, 1, 3),
        std=torch.ones(1, 1, 3),
        coordinates="conservative",
        normalization="fixed_physical",
    )
    full = evaluate_one_step(
        model,
        adapter,
        loader,
        normalizer,
        torch.device("cpu"),
        loss_coordinates="conservative",
        target_supervision="state",
    )
    deferred = evaluate_one_step_selection(
        model,
        adapter,
        loader,
        normalizer,
        torch.device("cpu"),
    )

    for key in ("loss", "state_loss", "relative_l2"):
        assert deferred[key] == pytest.approx(full[key], rel=1.0e-7)
