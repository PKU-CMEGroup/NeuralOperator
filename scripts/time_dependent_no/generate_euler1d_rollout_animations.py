"""Train 1D Euler target heads and save rollout animations.

This is a visualization pass for the solver-facing target ladder. The main
training harness intentionally keeps rollout arrays out of git-tracked outputs;
this script writes generated GIFs and summaries under ignored artifacts.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.time_dependent_no import train_euler1d_target_ladder as ladder
from utility.time_dependent_no.euler1d import make_euler1d_batch
from utility.time_dependent_no.euler1d_data import (
    Euler1DNPZ,
    Euler1DTimePairDataset,
    collate_euler1d_pairs,
    load_euler1d_npz,
)
from utility.time_dependent_no.euler1d_models import Euler1DTarget
from utility.time_dependent_no.euler1d_targets import make_target_adapter


VARIABLE_NAMES = ("density", "velocity", "pressure")
MODEL_CHOICES = ("cpgnet", "fno")
ARG_MODEL_CHOICES = (*MODEL_CHOICES, "cpg_style_pilot")
TARGET_CHOICES = (
    "residual",
    "primitive_residual",
    "limited_residual",
    "flux",
    "limited_flux",
    "physical_flux_correction",
    "interface",
    "positive_limited_interface",
)
EPS = 1.0e-12


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts" / "time_dependent_no" / "euler1d_rollout_animations",
    )
    parser.add_argument("--model", choices=(*ARG_MODEL_CHOICES, "all"), default="all")
    parser.add_argument("--target", choices=(*TARGET_CHOICES, "all"), default="all")
    parser.add_argument("--step-stride", type=int, default=4)
    parser.add_argument("--rollout-final-frame", type=int, default=80)
    parser.add_argument("--case-ids", nargs="*", type=int)
    parser.add_argument("--num-cases", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--train-cases", type=int, default=384)
    parser.add_argument("--test-cases", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--flux-correction-scale",
        type=float,
        default=1.0,
        help="Multiplier for bounded physical_flux_correction outputs.",
    )
    parser.add_argument(
        "--flux-correction-scale-floor",
        type=float,
        default=1.0e-6,
        help="Minimum componentwise physical flux scale for correction bounds.",
    )
    parser.add_argument("--input-noise-std", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260707)
    parser.add_argument(
        "--positive-transform", choices=("none", "softplus", "exp"), default="softplus"
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--torch-threads", type=int, default=0)
    parser.add_argument("--cpg-hidden-dim", type=int, default=128)
    parser.add_argument("--cpg-message-passing-steps", type=int, default=12)
    parser.add_argument("--cpg-mlp-layers", type=int, default=3)
    parser.add_argument("--fno-width", type=int, default=64)
    parser.add_argument("--fno-modes", type=int, default=16)
    parser.add_argument("--fno-layers", type=int, default=4)
    parser.add_argument("--fno-fc-dim", type=int, default=128)
    parser.add_argument("--fno-pad-ratio", type=float, default=0.0)
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument(
        "--plot-scale",
        choices=("truth", "combined"),
        default="truth",
        help="Use truth-scale y limits for readability or combined truth/prediction limits.",
    )
    parser.add_argument(
        "--save-checkpoints",
        action="store_true",
        help="Save per-variant state_dict checkpoints under the artifact output directory.",
    )
    return parser.parse_args(argv)


def requested(value: str, choices: tuple[str, ...]) -> list[str]:
    return list(choices) if value == "all" else [value]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_cases(
    source: Euler1DNPZ, args: argparse.Namespace
) -> tuple[np.ndarray, np.ndarray]:
    split_args = SimpleNamespace(
        train_cases=args.train_cases,
        test_cases=args.test_cases,
        seed=args.seed,
    )
    return ladder.split_cases(source, split_args)


def select_case_ids(test_cases: np.ndarray, args: argparse.Namespace) -> list[int]:
    if args.case_ids:
        test_set = {int(case_id) for case_id in test_cases.tolist()}
        missing = [case_id for case_id in args.case_ids if int(case_id) not in test_set]
        if missing:
            raise ValueError(
                f"case ids are not in the deterministic test split: {missing}"
            )
        return [int(case_id) for case_id in args.case_ids]
    if args.num_cases < 1:
        raise ValueError("--num-cases must be >= 1")
    if args.num_cases >= len(test_cases):
        return [int(case_id) for case_id in test_cases.tolist()]
    positions = np.linspace(0, len(test_cases) - 1, args.num_cases, dtype=np.int64)
    return [int(test_cases[pos]) for pos in positions]


def make_loaders(
    source: Euler1DNPZ,
    train_cases: np.ndarray,
    test_cases: np.ndarray,
    args: argparse.Namespace,
) -> tuple[DataLoader, DataLoader, ladder.PrimitiveNormalizer]:
    train_dataset = Euler1DTimePairDataset(
        source,
        case_indices=train_cases,
        step_stride=args.step_stride,
    )
    test_dataset = Euler1DTimePairDataset(
        source,
        case_indices=test_cases,
        step_stride=args.step_stride,
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_euler1d_pairs,
        generator=generator,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_euler1d_pairs,
    )
    normalizer = ladder.PrimitiveNormalizer.from_source(
        source,
        train_cases,
        step_stride=args.step_stride,
    )
    return train_loader, test_loader, normalizer


def relative_l2_by_frame(prediction: np.ndarray, truth: np.ndarray) -> np.ndarray:
    diff = prediction.reshape(prediction.shape[0], -1) - truth.reshape(
        truth.shape[0], -1
    )
    denom = np.linalg.norm(truth.reshape(truth.shape[0], -1), axis=1)
    return np.linalg.norm(diff, axis=1) / np.maximum(denom, EPS)


def train_variant(
    source: Euler1DNPZ,
    train_cases: np.ndarray,
    test_cases: np.ndarray,
    model_name: str,
    target_name: str,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[nn.Module, nn.Module, dict[str, Any]]:
    train_loader, test_loader, normalizer_cpu = make_loaders(
        source, train_cases, test_cases, args
    )
    normalizer = normalizer_cpu.to(device)
    target = cast(Euler1DTarget, target_name)
    model = ladder.build_model(model_name, target, args).to(device)
    adapter = make_target_adapter(
        target_name,
        positive_transform=args.positive_transform,
        flux_correction_scale=args.flux_correction_scale,
        flux_correction_scale_floor=args.flux_correction_scale_floor,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    history: list[dict[str, Any]] = []
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_metrics = ladder.train_one_epoch(
            model,
            adapter,
            train_loader,
            normalizer,
            optimizer,
            device,
            args.grad_clip,
            args.input_noise_std,
        )
        test_metrics = ladder.evaluate_one_step(
            model, adapter, test_loader, normalizer, device
        )
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_relative_l2": train_metrics["relative_l2"],
            "test_loss": test_metrics["loss"],
            "test_relative_l2": test_metrics["relative_l2"],
            "test_proposed_min_pressure": test_metrics["proposed_min_pressure"],
            "test_raw_min_pressure": test_metrics["raw_min_pressure"],
            "input_noise_std": args.input_noise_std,
            "flux_correction_scale": args.flux_correction_scale,
            "flux_correction_scale_floor": args.flux_correction_scale_floor,
            "test_num_nonpositive_proposed_pressure": test_metrics[
                "num_nonpositive_proposed_pressure"
            ],
            "test_num_nonpositive_raw_pressure": test_metrics[
                "num_nonpositive_raw_pressure"
            ],
        }
        history.append(row)
        print(
            f"{model_name:6s} {target_name:9s} epoch {epoch:03d}/{args.epochs:03d} "
            f"train_loss={row['train_loss']:.4e} test_rel_l2={row['test_relative_l2']:.4e}",
            flush=True,
        )

    final_eval = ladder.evaluate_one_step(
        model, adapter, test_loader, normalizer, device
    )
    return (
        model,
        adapter,
        {
            "model": model_name,
            "target": target_name,
            "history": history,
            "one_step": final_eval,
            "runtime_seconds": time.perf_counter() - start,
        },
    )


def rollout_arrays(
    model: nn.Module,
    adapter: nn.Module,
    source: Euler1DNPZ,
    case_id: int,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    if args.rollout_final_frame % args.step_stride != 0:
        raise ValueError("--rollout-final-frame must be divisible by --step-stride")
    max_steps = args.rollout_final_frame // args.step_stride
    x_np = source.x[case_id]
    current = torch.from_numpy(source.data[case_id, 0]).unsqueeze(0).to(device)
    x = torch.from_numpy(x_np).unsqueeze(0).to(device)
    left = torch.from_numpy(source.left_states[case_id]).unsqueeze(0).to(device)

    predictions = [source.data[case_id, 0].astype(np.float64)]
    proposed_accumulator = ladder.new_conservative_safety_accumulator()
    limiter_accumulator = ladder.new_limiter_accumulator()
    flux_correction_accumulator = ladder.new_flux_correction_accumulator()
    completed = True

    model.eval()
    with torch.no_grad():
        for step in range(max_steps):
            current_frame = step * args.step_stride
            next_frame = current_frame + args.step_stride
            dt = torch.tensor(
                [source.t[case_id, next_frame] - source.t[case_id, current_frame]],
                dtype=current.dtype,
                device=device,
            )
            batch = make_euler1d_batch(
                current,
                x,
                dt,
                gamma=source.gamma,
                left_boundary_primitive=left,
            )
            decoded = adapter(model(batch), batch)
            next_state = decoded.primitive
            ladder.update_conservative_safety_accumulator(
                proposed_accumulator,
                ladder.proposed_conservative(decoded),
                gamma=source.gamma,
            )
            ladder.update_limiter_accumulator(limiter_accumulator, decoded.aux)
            ladder.update_flux_correction_accumulator(
                flux_correction_accumulator, decoded.aux
            )
            if not torch.isfinite(next_state).all():
                completed = False
                break
            predictions.append(
                next_state.squeeze(0).detach().cpu().numpy().astype(np.float64)
            )
            current = next_state.detach()

    prediction = np.stack(predictions, axis=0)
    frame_ids = np.arange(prediction.shape[0], dtype=np.int64) * args.step_stride
    truth = source.data[case_id, frame_ids].astype(np.float64)
    rel_l2 = relative_l2_by_frame(prediction, truth)
    proposed_metrics = ladder.proposed_safety_metrics(proposed_accumulator)
    limiter_metrics = ladder.finalize_limiter_accumulator(limiter_accumulator)
    flux_correction_metrics = ladder.finalize_flux_correction_accumulator(
        flux_correction_accumulator
    )
    if prediction.shape[0] - 1 < max_steps:
        completed = False
    return {
        "case_id": int(case_id),
        "x": x_np.astype(np.float64),
        "frame_ids": frame_ids,
        "prediction": prediction,
        "truth": truth,
        "relative_l2_by_frame": rel_l2,
        "completed_horizon": bool(completed),
        **proposed_metrics,
        **limiter_metrics,
        **flux_correction_metrics,
        "max_abs_primitive": float(np.nanmax(np.abs(prediction))),
        "final_relative_l2": float(rel_l2[-1]),
        "mean_relative_l2": float(np.nanmean(rel_l2)),
    }


def variable_limits(
    truth: np.ndarray, prediction: np.ndarray, variable: int, plot_scale: str
) -> tuple[float, float]:
    values = truth[..., variable]
    if plot_scale == "combined":
        values = np.concatenate(
            (values.reshape(-1), prediction[..., variable].reshape(-1))
        )
        values = values[np.isfinite(values)]
        if values.size > 0:
            lo = float(np.nanpercentile(values, 1.0))
            hi = float(np.nanpercentile(values, 99.0))
        else:
            lo, hi = -1.0, 1.0
    else:
        lo = float(np.nanmin(values))
        hi = float(np.nanmax(values))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        center = 0.0 if not np.isfinite(lo) else lo
        lo, hi = center - 1.0, center + 1.0
    pad = 0.08 * (hi - lo)
    return lo - pad, hi + pad


def save_rollout_gif(
    rollout: dict[str, Any],
    *,
    model_name: str,
    target_name: str,
    output_path: Path,
    fps: int,
    dpi: int,
    plot_scale: str,
) -> None:
    x = rollout["x"]
    truth = rollout["truth"]
    prediction = rollout["prediction"]
    frame_ids = rollout["frame_ids"]
    rel = rollout["relative_l2_by_frame"]
    limits = [variable_limits(truth, prediction, i, plot_scale) for i in range(3)]

    fig, axes = plt.subplots(
        3, 1, figsize=(8.5, 8.0), sharex=True, constrained_layout=True
    )
    truth_lines = []
    pred_lines = []
    for i, ax in enumerate(axes):
        (truth_line,) = ax.plot(
            x, truth[0, :, i], color="black", linewidth=1.8, label="truth"
        )
        (pred_line,) = ax.plot(
            x, prediction[0, :, i], color="#d62728", linewidth=1.4, label="prediction"
        )
        ax.set_ylabel(VARIABLE_NAMES[i])
        ax.set_ylim(*limits[i])
        ax.grid(True, alpha=0.25)
        truth_lines.append(truth_line)
        pred_lines.append(pred_line)
    axes[-1].set_xlabel("x")
    axes[0].legend(loc="upper right")
    title = fig.suptitle("")

    def update(frame: int):
        for i in range(3):
            truth_lines[i].set_ydata(truth[frame, :, i])
            pred_lines[i].set_ydata(prediction[frame, :, i])
        title.set_text(
            f"{model_name} / {target_name} | case {rollout['case_id']} | "
            f"saved frame {int(frame_ids[frame])} | rel L2={rel[frame]:.3e}"
        )
        return [*truth_lines, *pred_lines, title]

    animation = FuncAnimation(fig, update, frames=prediction.shape[0], blit=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    animation.save(output_path, writer=PillowWriter(fps=max(fps, 1)), dpi=dpi)
    plt.close(fig)


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(k): json_ready(v)
            for k, v in value.items()
            if k not in {"x", "prediction", "truth"}
        }
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.input_noise_std < 0.0:
        raise ValueError("--input-noise-std must be nonnegative")
    if args.flux_correction_scale < 0.0:
        raise ValueError("--flux-correction-scale must be nonnegative")
    if args.flux_correction_scale_floor <= 0.0:
        raise ValueError("--flux-correction-scale-floor must be positive")
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    set_seed(args.seed)
    device = ladder.select_device(args)
    source = load_euler1d_npz(args.data_path)
    train_cases, test_cases = split_cases(source, args)
    case_ids = select_case_ids(test_cases, args)
    models = requested(args.model, MODEL_CHOICES)
    targets = requested(args.target, TARGET_CHOICES)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "data_path": str(args.data_path),
        "data_shape": list(source.data.shape),
        "device": str(device),
        "step_stride": args.step_stride,
        "rollout_final_frame": args.rollout_final_frame,
        "input_noise_std": args.input_noise_std,
        "flux_correction_scale": args.flux_correction_scale,
        "flux_correction_scale_floor": args.flux_correction_scale_floor,
        "case_ids": case_ids,
        "models": models,
        "targets": targets,
        "runs": [],
    }
    print(
        json.dumps({k: v for k, v in manifest.items() if k != "runs"}, indent=2),
        flush=True,
    )

    for model_name in models:
        for target_name in targets:
            run_dir = (
                args.output_dir
                / f"stride{args.step_stride}"
                / f"{model_name}_{target_name}"
            )
            run_dir.mkdir(parents=True, exist_ok=True)
            print(
                f"[visualize] training {model_name}/{target_name} -> {run_dir}",
                flush=True,
            )
            model, adapter, train_summary = train_variant(
                source,
                train_cases,
                test_cases,
                model_name,
                target_name,
                args,
                device,
            )
            if args.save_checkpoints:
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "model": model_name,
                        "target": target_name,
                        "args": vars(args),
                    },
                    run_dir / "checkpoint.pt",
                )

            case_summaries = []
            for case_id in case_ids:
                rollout = rollout_arrays(model, adapter, source, case_id, args, device)
                gif_path = run_dir / f"case_{case_id:04d}.gif"
                save_rollout_gif(
                    rollout,
                    model_name=model_name,
                    target_name=target_name,
                    output_path=gif_path,
                    fps=args.fps,
                    dpi=args.dpi,
                    plot_scale=args.plot_scale,
                )
                summary = json_ready(rollout)
                summary["animation_path"] = str(gif_path)
                case_summaries.append(summary)
                print(
                    f"[visualize] saved {gif_path} final_rel={rollout['final_relative_l2']:.4e} "
                    f"completed={rollout['completed_horizon']}",
                    flush=True,
                )

            run_summary = {
                **train_summary,
                "step_stride": args.step_stride,
                "rollout_final_frame": args.rollout_final_frame,
                "input_noise_std": args.input_noise_std,
                "flux_correction_scale": args.flux_correction_scale,
                "flux_correction_scale_floor": args.flux_correction_scale_floor,
                "case_summaries": case_summaries,
                "run_dir": str(run_dir),
            }
            (run_dir / "animation_summary.json").write_text(
                json.dumps(json_ready(run_summary), indent=2, sort_keys=True),
                encoding="utf-8",
            )
            manifest["runs"].append(run_summary)

    manifest_path = args.output_dir / "animation_manifest.json"
    manifest_path.write_text(
        json.dumps(json_ready(manifest), indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"manifest": str(manifest_path)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
