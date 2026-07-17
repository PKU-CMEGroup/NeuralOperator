"""Train one residual FNO on balanced homogeneous batches from multiple grids."""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler

if __package__:
    from scripts.time_dependent_no.evaluate_euler1d_resolution_transfer import (
        validate_resolution_family,
    )
    from scripts.time_dependent_no.train_euler1d_target_ladder import (
        PrimitiveNormalizer,
        build_model,
        clone_state_dict_cpu,
        evaluate_one_step,
        json_ready,
        load_state_dict_cpu,
        relative_l2_torch,
        rollout_cases,
        rollout_selection_score,
        select_device,
        set_seed,
        split_cases,
        supervised_loss,
        summarize_rollouts,
        train_one_epoch,
        train_unrolled_epoch,
    )
else:
    from evaluate_euler1d_resolution_transfer import validate_resolution_family
    from train_euler1d_target_ladder import (
        PrimitiveNormalizer,
        build_model,
        clone_state_dict_cpu,
        evaluate_one_step,
        json_ready,
        load_state_dict_cpu,
        relative_l2_torch,
        rollout_cases,
        rollout_selection_score,
        select_device,
        set_seed,
        split_cases,
        supervised_loss,
        summarize_rollouts,
        train_one_epoch,
        train_unrolled_epoch,
    )
from utility.time_dependent_no.euler1d import (
    conservative_to_primitive,
    make_euler1d_batch,
    primitive_to_conservative,
)
from utility.time_dependent_no.euler1d_data import (
    Euler1DNPZ,
    Euler1DRolloutWindowDataset,
    Euler1DTimePairDataset,
    collate_euler1d_pairs,
    collate_euler1d_rollout_windows,
    load_euler1d_npz,
    restriction_commutation_metrics,
)
from utility.time_dependent_no.euler1d_targets import make_target_adapter


class AlternatingResolutionLoader:
    """Round-robin batches from loaders whose individual tensors have one grid."""

    def __init__(self, loaders: dict[str, DataLoader]) -> None:
        if not loaders:
            raise ValueError("at least one resolution loader is required")
        self.loaders = loaders

    def __len__(self) -> int:
        return sum(len(loader) for loader in self.loaders.values())

    def __iter__(self) -> Iterator[Any]:
        labels = list(self.loaders)
        iterators = {label: iter(self.loaders[label]) for label in labels}
        active = set(labels)
        while active:
            for label in labels:
                if label not in active:
                    continue
                try:
                    yield next(iterators[label])
                except StopIteration:
                    active.remove(label)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        nargs=2,
        action="append",
        metavar=("LABEL", "PATH"),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--unroll-epochs", type=int, default=10)
    parser.add_argument("--unroll-steps", type=int, default=4)
    parser.add_argument("--unroll-lr-factor", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--evaluation-batch-size", type=int, default=128)
    parser.add_argument("--samples-per-epoch", type=int, default=0)
    parser.add_argument("--unroll-samples-per-epoch", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument("--split-seed", type=int, default=20260707)
    parser.add_argument("--train-cases", type=int, default=64)
    parser.add_argument("--val-cases", type=int, default=16)
    parser.add_argument("--test-cases", type=int, default=16)
    parser.add_argument("--step-stride", type=int, default=1)
    parser.add_argument("--rollout-final-frame", type=int, default=20)
    parser.add_argument("--replay-horizons", type=int, nargs="+", default=[20, 50, 100])
    parser.add_argument("--fno-width", type=int, default=64)
    parser.add_argument("--fno-modes", type=int, default=24)
    parser.add_argument("--fno-layers", type=int, default=4)
    parser.add_argument("--fno-fc-dim", type=int, default=128)
    parser.add_argument("--fno-pad-ratio", type=float, default=0.0)
    parser.add_argument("--commutation-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--torch-threads", type=int, default=0)
    return parser.parse_args()


def labeled_paths(entries: list[list[str]]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for label, raw_path in entries:
        if label in result:
            raise ValueError(f"duplicate data label: {label}")
        result[label] = Path(raw_path)
    return result


def balanced_batch_counts(
    total_batches: int,
    labels: list[str],
    *,
    offset: int = 0,
) -> dict[str, int]:
    """Divide a fixed update budget as evenly as possible among resolutions."""

    if total_batches < len(labels):
        raise ValueError("total batches must give every resolution at least one batch")
    base, remainder = divmod(total_batches, len(labels))
    counts = {label: base for label in labels}
    for index in range(remainder):
        label = labels[(offset + index) % len(labels)]
        counts[label] += 1
    return counts


def _sample_indices(
    dataset_size: int,
    sample_count: int,
    generator: torch.Generator,
) -> list[int]:
    if dataset_size < 1 or sample_count < 1:
        raise ValueError("dataset and sample counts must be positive")
    if sample_count <= dataset_size:
        return torch.randperm(dataset_size, generator=generator)[:sample_count].tolist()
    return torch.randint(
        dataset_size,
        (sample_count,),
        generator=generator,
    ).tolist()


def make_balanced_loader(
    datasets: dict[str, Any],
    *,
    total_samples: int,
    batch_size: int,
    generator: torch.Generator,
    collate_fn: Any,
    offset: int = 0,
) -> tuple[AlternatingResolutionLoader, dict[str, int]]:
    """Construct one exact-budget epoch of homogeneous-resolution batches."""

    if batch_size < 1 or total_samples % batch_size:
        raise ValueError("total samples must be a positive multiple of batch size")
    labels = list(datasets)
    counts = balanced_batch_counts(total_samples // batch_size, labels, offset=offset)
    loaders: dict[str, DataLoader] = {}
    for label, dataset in datasets.items():
        indices = _sample_indices(len(dataset), counts[label] * batch_size, generator)
        sampler = SubsetRandomSampler(indices, generator=generator)
        loaders[label] = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=True,
            collate_fn=collate_fn,
        )
    return AlternatingResolutionLoader(loaders), counts


def _full_pair_loaders(
    sources: dict[str, Euler1DNPZ],
    cases: np.ndarray,
    *,
    step_stride: int,
    batch_size: int,
) -> dict[str, DataLoader]:
    return {
        label: DataLoader(
            Euler1DTimePairDataset(
                source,
                case_indices=cases,
                step_stride=step_stride,
            ),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_euler1d_pairs,
        )
        for label, source in sources.items()
    }


def _model_contract(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "fno_width": args.fno_width,
        "fno_modes": args.fno_modes,
        "fno_layers": args.fno_layers,
        "fno_fc_dim": args.fno_fc_dim,
        "fno_pad_ratio": args.fno_pad_ratio,
        "input_coordinates": "conservative",
        "input_normalization": "fixed_physical",
        "loss_coordinates": "conservative",
        "loss_normalization": "fixed_physical",
        "recurrent_coordinates": "conservative",
        "positive_transform": "none",
        "target_supervision": "state",
        "flux_gauge_mode": "raw",
        "interface_flux_mode": "rusanov",
        "flux_correction_scale": 1.0,
        "flux_correction_scale_floor": 1.0e-6,
        "step_stride": args.step_stride,
        "rollout_steps": args.rollout_final_frame // args.step_stride,
        "rollout_final_frame": args.rollout_final_frame,
        "epochs": args.epochs,
        "unroll_epochs": args.unroll_epochs,
        "unroll_steps": args.unroll_steps,
        "unroll_lr_factor": args.unroll_lr_factor,
        "batch_size": args.batch_size,
        "evaluation_batch_size": args.evaluation_batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "split_seed": args.split_seed,
        "train_cases": args.train_cases,
        "val_cases": args.val_cases,
        "test_cases": args.test_cases,
    }


def batched_rollout_selection_summary(
    model: torch.nn.Module,
    adapter: torch.nn.Module,
    source: Euler1DNPZ,
    cases: np.ndarray,
    model_args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    """Compute the raw checkpoint rollout metric with cases batched by grid."""

    case_ids = np.asarray(cases, dtype=np.int64)
    step_stride = int(model_args.step_stride)
    final_frame = int(model_args.rollout_final_frame)
    max_steps = final_frame // step_stride
    current_primitive = torch.from_numpy(
        np.asarray(source.data[case_ids, 0], dtype=np.float32)
    ).to(device)
    current_conservative = primitive_to_conservative(
        current_primitive,
        gamma=source.gamma,
    )
    x = torch.from_numpy(source.x[case_ids]).to(device)
    left = torch.from_numpy(source.left_states[case_ids]).to(device)
    right = torch.from_numpy(source.right_states[case_ids]).to(device)
    active = np.ones(case_ids.size, dtype=np.bool_)
    finite = np.ones(case_ids.size, dtype=np.bool_)
    num_steps = np.zeros(case_ids.size, dtype=np.int64)
    error_sum = np.zeros(case_ids.size, dtype=np.float64)
    final_error = np.full(case_ids.size, np.nan, dtype=np.float64)

    model.eval()
    with torch.no_grad():
        for step in range(max_steps):
            current_frame = step * step_stride
            next_frame = current_frame + step_stride
            model_primitive = conservative_to_primitive(
                current_conservative,
                gamma=source.gamma,
            )
            dt = torch.from_numpy(
                (
                    source.t[case_ids, next_frame] - source.t[case_ids, current_frame]
                ).astype(np.float32)
            ).to(device)
            batch = make_euler1d_batch(
                model_primitive,
                x,
                dt,
                current_conservative_state=current_conservative,
                gamma=source.gamma,
                left_boundary_primitive=left,
                right_initial_primitive=right,
            )
            decoded = adapter(model(batch), batch)
            next_state = decoded.primitive
            state_finite = (
                torch.isfinite(next_state).all(dim=(1, 2)).detach().cpu().numpy()
            )
            state_admissible = (
                ((next_state[..., 0] > 0.0) & (next_state[..., 2] > 0.0))
                .all(dim=1)
                .detach()
                .cpu()
                .numpy()
            )
            raw_recurrence = bool(decoded.aux.get("raw_recurrence", False))
            valid = state_finite & (state_admissible if raw_recurrence else True)
            newly_invalid = active & ~valid
            finite[newly_invalid & ~state_finite] = False
            accepted = active & valid

            prediction = next_state.detach().cpu().numpy().astype(np.float64)
            truth = source.data[case_ids, next_frame].astype(np.float64)
            difference = (prediction - truth).reshape(case_ids.size, -1)
            denominator = np.linalg.norm(truth.reshape(case_ids.size, -1), axis=1)
            relative = np.linalg.norm(difference, axis=1) / np.maximum(
                denominator,
                1.0e-12,
            )
            error_sum[accepted] += relative[accepted]
            final_error[accepted] = relative[accepted]
            num_steps[accepted] += 1

            accepted_tensor = torch.as_tensor(
                accepted,
                device=device,
                dtype=torch.bool,
            ).reshape(-1, 1, 1)
            current_primitive = torch.where(
                accepted_tensor,
                next_state.detach(),
                current_primitive,
            )
            current_conservative = torch.where(
                accepted_tensor,
                decoded.conservative.detach(),
                current_conservative,
            )
            active = accepted
            if not np.any(active):
                break

    completed = num_steps == max_steps
    survival = num_steps / max_steps
    has_prediction = num_steps > 0
    per_case_mean = np.full(case_ids.size, np.nan, dtype=np.float64)
    per_case_mean[has_prediction] = (
        error_sum[has_prediction] / num_steps[has_prediction]
    )
    return {
        "selection_rollout_mode": "batched_raw_residual",
        "finite": bool(np.all(finite)),
        "admissible": bool(np.all(completed)),
        "num_cases": int(case_ids.size),
        "num_completed_cases": int(np.sum(completed)),
        "completion_fraction": float(np.mean(completed)),
        "survival_fraction_mean": float(np.mean(survival)),
        "survival_fraction_min": float(np.min(survival)),
        "num_steps_min": int(np.min(num_steps)),
        "num_steps_max": int(np.max(num_steps)),
        "final_frame_min": int(np.min(num_steps) * step_stride),
        "final_frame_max": int(np.max(num_steps) * step_stride),
        "completed_horizon": bool(np.all(completed)),
        "num_nonpositive_terminations": int(np.sum(~completed & finite)),
        "num_nonfinite_terminations": int(np.sum(~finite)),
        "rollout_relative_l2_mean": (
            float(np.nanmean(per_case_mean)) if np.any(has_prediction) else float("nan")
        ),
        "rollout_relative_l2_final": (
            float(np.nanmean(final_error)) if np.any(has_prediction) else float("nan")
        ),
    }


def _pool_selection_summaries(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    if not summaries:
        raise ValueError("at least one resolution summary is required")
    case_counts = np.asarray([row["num_cases"] for row in summaries], dtype=np.float64)
    weights = case_counts / np.sum(case_counts)
    return {
        "selection_rollout_mode": "batched_raw_residual",
        "finite": all(bool(row["finite"]) for row in summaries),
        "admissible": all(bool(row["admissible"]) for row in summaries),
        "num_cases": int(np.sum(case_counts)),
        "num_completed_cases": int(
            sum(int(row["num_completed_cases"]) for row in summaries)
        ),
        "completion_fraction": float(
            np.sum(
                weights
                * np.asarray(
                    [row["completion_fraction"] for row in summaries],
                    dtype=np.float64,
                )
            )
        ),
        "survival_fraction_mean": float(
            np.sum(
                weights
                * np.asarray(
                    [row["survival_fraction_mean"] for row in summaries],
                    dtype=np.float64,
                )
            )
        ),
        "survival_fraction_min": float(
            min(row["survival_fraction_min"] for row in summaries)
        ),
        "completed_horizon": all(bool(row["completed_horizon"]) for row in summaries),
        "rollout_relative_l2_mean": float(
            np.mean([row["rollout_relative_l2_mean"] for row in summaries])
        ),
        "rollout_relative_l2_final": float(
            np.mean([row["rollout_relative_l2_final"] for row in summaries])
        ),
    }


def evaluate_one_step_selection(
    model: torch.nn.Module,
    adapter: torch.nn.Module,
    loader: DataLoader,
    normalizer: PrimitiveNormalizer,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate the exact state loss while synchronizing metrics once."""

    model.eval()
    losses: list[torch.Tensor] = []
    state_losses: list[torch.Tensor] = []
    relative_errors: list[torch.Tensor] = []
    total_samples = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            if batch.target_primitive is None:
                raise RuntimeError("evaluation batch is missing target_primitive")
            raw = model(batch)
            prediction = adapter(raw, batch)
            loss, state_loss, _flux_loss, _boundary_loss = supervised_loss(
                raw,
                prediction,
                batch,
                normalizer,
                loss_coordinates="conservative",
                target_supervision="state",
                flux_normalizer=None,
                flux_loss_weight=1.0,
            )
            batch_size = int(batch.current_primitive.shape[0])
            losses.append(loss * batch_size)
            state_losses.append(state_loss * batch_size)
            relative_errors.append(
                relative_l2_torch(
                    prediction.primitive,
                    batch.target_primitive,
                ).sum()
            )
            total_samples += batch_size
    totals = torch.stack(
        (
            torch.stack(losses).sum(),
            torch.stack(state_losses).sum(),
            torch.stack(relative_errors).sum(),
        )
    )
    if not torch.isfinite(totals).all():
        raise FloatingPointError("non-finite selection evaluation metrics")
    values = totals.cpu().numpy() / total_samples
    return {
        "loss": float(values[0]),
        "state_loss": float(values[1]),
        "relative_l2": float(values[2]),
        "num_samples": float(total_samples),
        "evaluation_mode": "deferred_state_selection",
    }


def evaluate_family(
    model: torch.nn.Module,
    adapter: torch.nn.Module,
    sources: dict[str, Euler1DNPZ],
    loaders: dict[str, DataLoader],
    cases: np.ndarray,
    normalizer: PrimitiveNormalizer,
    model_args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    per_resolution: dict[str, Any] = {}
    rollout_summaries: list[dict[str, Any]] = []
    losses: list[float] = []
    for label, source in sources.items():
        one_step = evaluate_one_step_selection(
            model,
            adapter,
            loaders[label],
            normalizer,
            device,
        )
        rollout = batched_rollout_selection_summary(
            model,
            adapter,
            source,
            cases,
            model_args,
            device,
        )
        rollout_summaries.append(rollout)
        per_resolution[label] = {
            "one_step": one_step,
            "rollout": rollout,
        }
        losses.append(float(one_step["loss"]))
    pooled = _pool_selection_summaries(rollout_summaries)
    mean_loss = float(np.mean(losses))
    return {
        "mean_one_step_loss": mean_loss,
        "per_resolution": per_resolution,
        "pooled_rollout": pooled,
        "selection_score": rollout_selection_score(pooled, mean_loss),
    }


def evaluate_one_step_family(
    model: torch.nn.Module,
    adapter: torch.nn.Module,
    loaders: dict[str, DataLoader],
    normalizer: PrimitiveNormalizer,
    device: torch.device,
) -> dict[str, Any]:
    per_resolution = {
        label: evaluate_one_step_selection(
            model,
            adapter,
            loader,
            normalizer,
            device,
        )
        for label, loader in loaders.items()
    }
    return {
        "mean_loss": float(
            np.mean([metrics["loss"] for metrics in per_resolution.values()])
        ),
        "per_resolution": per_resolution,
    }


def evaluate_replay_horizons(
    model: torch.nn.Module,
    adapter: torch.nn.Module,
    sources: dict[str, Euler1DNPZ],
    loaders: dict[str, DataLoader],
    cases: np.ndarray,
    normalizer: PrimitiveNormalizer,
    contract: dict[str, Any],
    horizons: list[int],
    device: torch.device,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for label, source in sources.items():
        one_step = evaluate_one_step(
            model,
            adapter,
            loaders[label],
            normalizer,
            device,
            loss_coordinates="conservative",
            target_supervision="state",
        )
        rollouts: dict[str, Any] = {}
        for horizon in sorted(set(horizons)):
            if horizon < contract["step_stride"] or horizon >= source.num_frames:
                continue
            if horizon % contract["step_stride"]:
                continue
            horizon_contract = dict(contract)
            horizon_contract["rollout_final_frame"] = horizon
            horizon_contract["rollout_steps"] = horizon // contract["step_stride"]
            horizon_args = argparse.Namespace(**horizon_contract)
            rows = rollout_cases(
                model,
                adapter,
                source,
                cases,
                horizon_args,
                device,
            )
            rollouts[str(horizon)] = {
                "summary": summarize_rollouts(rows),
                "cases": rows,
            }
        result[label] = {"one_step": one_step, "rollout": rollouts}
    return result


def _write_summary_csv(path: Path, results: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    for label, evaluated in results.items():
        for horizon, rollout in evaluated["rollout"].items():
            rows.append(
                {
                    "data": label,
                    "horizon_frame": int(horizon),
                    "one_step_relative_l2": evaluated["one_step"]["relative_l2"],
                    **rollout["summary"],
                }
            )
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.unroll_epochs < 0:
        raise ValueError("epoch counts are invalid")
    if args.evaluation_batch_size < 1:
        raise ValueError("evaluation batch size must be positive")
    if args.unroll_steps < 2 or args.unroll_lr_factor <= 0.0:
        raise ValueError("unroll settings are invalid")
    if args.step_stride < 1:
        raise ValueError("step stride must be positive")
    if args.rollout_final_frame % args.step_stride:
        raise ValueError("rollout final frame must be divisible by step stride")
    if args.commutation_tolerance <= 0.0:
        raise ValueError("commutation tolerance must be positive")
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    set_seed(args.seed)
    device = select_device(args)

    paths = labeled_paths(args.data)
    loaded = {label: load_euler1d_npz(path) for label, path in paths.items()}
    validate_resolution_family(loaded)
    ordered_items = sorted(loaded.items(), key=lambda item: item[1].num_cells)
    sources = dict(ordered_items)
    if len(sources) < 2:
        raise ValueError("shared training requires at least two resolutions")
    reference = ordered_items[-1][1]
    train_cases, val_cases, test_cases = split_cases(reference, args)
    selected_cases = np.concatenate((train_cases, val_cases, test_cases))
    commutation = [
        restriction_commutation_metrics(
            coarse,
            fine,
            case_indices=selected_cases,
            strides=(1, 2),
        )
        for (_, coarse), (_, fine) in zip(ordered_items, ordered_items[1:])
    ]
    commutation_max = max(
        [item["state_global_relative_l2"] for item in commutation]
        + [
            update["global_relative_l2"]
            for item in commutation
            for update in item["updates"]
        ]
    )
    if commutation_max > args.commutation_tolerance:
        raise ValueError(
            f"commutation error {commutation_max:.6e} exceeds "
            f"{args.commutation_tolerance:.6e}"
        )

    contract = _model_contract(args)
    model_args = argparse.Namespace(**contract)
    normalizer = PrimitiveNormalizer.from_source(
        reference,
        train_cases,
        step_stride=args.step_stride,
        coordinates="conservative",
        normalization="fixed_physical",
    ).to(device)
    input_normalizer = PrimitiveNormalizer.from_source_inputs(
        reference,
        train_cases,
        step_stride=args.step_stride,
        coordinates="conservative",
        normalization="fixed_physical",
    ).to(device)
    model = build_model("fno", "residual", model_args, input_normalizer).to(device)
    adapter = make_target_adapter("residual", positive_transform="none").to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    train_datasets = {
        label: Euler1DTimePairDataset(
            source,
            case_indices=train_cases,
            step_stride=args.step_stride,
        )
        for label, source in sources.items()
    }
    val_loaders = _full_pair_loaders(
        sources,
        val_cases,
        step_stride=args.step_stride,
        batch_size=args.evaluation_batch_size,
    )
    test_loaders = _full_pair_loaders(
        sources,
        test_cases,
        step_stride=args.step_stride,
        batch_size=args.evaluation_batch_size,
    )
    train_evaluation_loaders = _full_pair_loaders(
        sources,
        train_cases,
        step_stride=args.step_stride,
        batch_size=args.evaluation_batch_size,
    )
    pair_presentations = args.samples_per_epoch or len(
        next(iter(train_datasets.values()))
    )
    if pair_presentations % args.batch_size:
        raise ValueError("one-step sample budget must be divisible by batch size")
    generator = torch.Generator().manual_seed(args.seed)

    initial_training = evaluate_one_step_family(
        model,
        adapter,
        train_evaluation_loaders,
        normalizer,
        device,
    )
    initial_validation = evaluate_family(
        model,
        adapter,
        sources,
        val_loaders,
        val_cases,
        normalizer,
        model_args,
        device,
    )
    history: list[dict[str, Any]] = []
    best_state = clone_state_dict_cpu(model)
    best_epoch = 0
    best_score = float("inf")
    best_validation: dict[str, Any] | None = None
    started = time.perf_counter()
    labels = list(sources)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        loader, batch_counts = make_balanced_loader(
            train_datasets,
            total_samples=pair_presentations,
            batch_size=args.batch_size,
            generator=generator,
            collate_fn=collate_euler1d_pairs,
            offset=(epoch - 1) % len(labels),
        )
        train_metrics = train_one_epoch(
            model,
            adapter,
            loader,
            normalizer,
            optimizer,
            device,
            args.grad_clip,
            input_noise_std=0.0,
            loss_coordinates="conservative",
            target_supervision="state",
            defer_metric_sync=True,
        )
        validation = evaluate_family(
            model,
            adapter,
            sources,
            val_loaders,
            val_cases,
            normalizer,
            model_args,
            device,
        )
        score = float(validation["selection_score"])
        if score < best_score or best_epoch == 0:
            best_score = score
            best_epoch = epoch
            best_state = clone_state_dict_cpu(model)
            best_validation = validation
        history.append(
            {
                "epoch": epoch,
                "stage": "one_step",
                "train": train_metrics,
                "batch_counts": batch_counts,
                "validation": validation,
                "is_best_checkpoint": epoch == best_epoch,
            }
        )
        (args.output_dir / "history.json").write_text(
            json.dumps(json_ready(history), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        train_loss = train_metrics["loss"]
        validation_loss = validation["mean_one_step_loss"]
        validation_rollout = validation["pooled_rollout"]["rollout_relative_l2_final"]
        print(
            f"shared residual epoch {epoch:03d}/{args.epochs:03d} "
            f"train_loss={train_loss:.4e} "
            f"val_loss={validation_loss:.4e} "
            f"val_rollout={validation_rollout:.4e} "
            f"best_epoch={best_epoch:03d}",
            flush=True,
        )

    if args.unroll_epochs > 0:
        load_state_dict_cpu(model, best_state, device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr * args.unroll_lr_factor,
            weight_decay=args.weight_decay,
        )
        unroll_datasets = {
            label: Euler1DRolloutWindowDataset(
                source,
                case_indices=train_cases,
                step_stride=args.step_stride,
                rollout_steps=args.unroll_steps,
            )
            for label, source in sources.items()
        }
        unroll_presentations = args.unroll_samples_per_epoch or len(
            next(iter(unroll_datasets.values()))
        )
        if unroll_presentations % args.batch_size:
            raise ValueError("unroll sample budget must be divisible by batch size")
        unroll_generator = torch.Generator().manual_seed(args.seed + 1)
        for stage_epoch in range(1, args.unroll_epochs + 1):
            epoch = args.epochs + stage_epoch
            loader, batch_counts = make_balanced_loader(
                unroll_datasets,
                total_samples=unroll_presentations,
                batch_size=args.batch_size,
                generator=unroll_generator,
                collate_fn=collate_euler1d_rollout_windows,
                offset=(stage_epoch - 1) % len(labels),
            )
            train_metrics = train_unrolled_epoch(
                model,
                adapter,
                loader,
                normalizer,
                optimizer,
                device,
                args.grad_clip,
                input_noise_std=0.0,
                loss_coordinates="conservative",
                defer_metric_sync=True,
            )
            validation = evaluate_family(
                model,
                adapter,
                sources,
                val_loaders,
                val_cases,
                normalizer,
                model_args,
                device,
            )
            score = float(validation["selection_score"])
            if score < best_score:
                best_score = score
                best_epoch = epoch
                best_state = clone_state_dict_cpu(model)
                best_validation = validation
            history.append(
                {
                    "epoch": epoch,
                    "stage": "autoregressive",
                    "stage_epoch": stage_epoch,
                    "train": train_metrics,
                    "batch_counts": batch_counts,
                    "validation": validation,
                    "is_best_checkpoint": epoch == best_epoch,
                }
            )
            (args.output_dir / "history.json").write_text(
                json.dumps(json_ready(history), indent=2, sort_keys=True),
                encoding="utf-8",
            )
            train_loss = train_metrics["loss"]
            validation_loss = validation["mean_one_step_loss"]
            validation_rollout = validation["pooled_rollout"][
                "rollout_relative_l2_final"
            ]
            print(
                f"shared residual autoregressive "
                f"{stage_epoch:03d}/{args.unroll_epochs:03d} "
                f"train_loss={train_loss:.4e} "
                f"val_loss={validation_loss:.4e} "
                f"val_rollout={validation_rollout:.4e} "
                f"best_epoch={best_epoch:03d}",
                flush=True,
            )
    else:
        unroll_presentations = 0

    if best_validation is None:
        raise RuntimeError("training did not produce a selectable checkpoint")
    load_state_dict_cpu(model, best_state, device)
    test_results = evaluate_replay_horizons(
        model,
        adapter,
        sources,
        test_loaders,
        test_cases,
        normalizer,
        contract,
        args.replay_horizons,
        device,
    )
    runtime = time.perf_counter() - started
    checkpoint_path = args.output_dir / "checkpoint.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model": "fno",
            "model_implementation": "FNOEuler1DHead",
            "parameter_count": parameter_count,
            "target": "residual",
            "args": contract,
            "train_cases": train_cases,
            "val_cases": val_cases,
            "test_cases": test_cases,
            "best_epoch": best_epoch,
            "selection_metric": (
                "pooled_completed_val_rollout_then_survival_then_one_step"
            ),
            "input_coordinates": "conservative",
            "recurrent_coordinates": "conservative",
            "predicted_quantity": "conservative_increment",
            "target_supervision": "state",
            "loss_coordinates": "conservative",
            "input_normalization": "fixed_physical",
            "loss_normalization": "fixed_physical",
            "normalizer_mean": normalizer.mean.detach().cpu(),
            "normalizer_std": normalizer.std.detach().cpu(),
            "input_normalizer_mean": input_normalizer.mean.detach().cpu(),
            "input_normalizer_std": input_normalizer.std.detach().cpu(),
            "training_mode": "balanced_restriction_consistent_multiresolution",
        },
        checkpoint_path,
    )
    payload = {
        "status": "ok",
        "training_mode": "balanced_restriction_consistent_multiresolution",
        "data_paths": {key: str(value) for key, value in paths.items()},
        "parameter_count": parameter_count,
        "contract": contract,
        "commutation": commutation,
        "commutation_max_global_relative_l2": commutation_max,
        "train_cases": train_cases,
        "val_cases": val_cases,
        "test_cases": test_cases,
        "one_step_samples_per_epoch": pair_presentations,
        "unroll_samples_per_epoch": unroll_presentations,
        "initial_training": initial_training,
        "initial_validation": initial_validation,
        "one_step_training_loss_decreased": min(
            row["train"]["loss"] for row in history if row["stage"] == "one_step"
        )
        < initial_training["mean_loss"],
        "best_epoch": best_epoch,
        "best_score": best_score,
        "best_validation": best_validation,
        "history": history,
        "test": test_results,
        "checkpoint_path": checkpoint_path,
        "runtime_seconds": runtime,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(json_ready(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_summary_csv(args.output_dir / "summary.csv", test_results)
    print(
        json.dumps(
            {
                "status": "ok",
                "best_epoch": best_epoch,
                "checkpoint": str(checkpoint_path),
                "runtime_seconds": runtime,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
