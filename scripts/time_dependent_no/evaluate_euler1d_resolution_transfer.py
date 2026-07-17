"""Evaluate frozen 1D Euler residual FNOs on matched mesh resolutions.

This entry point is intentionally evaluation-only.  It verifies that every
dataset represents the same physical cases and saved times, restores the exact
checkpoint architecture and normalizers, and reports native-grid one-step and
raw autoregressive metrics without updating model weights.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

if __package__:
    from scripts.time_dependent_no.train_euler1d_target_ladder import (
        PrimitiveNormalizer,
        build_model,
        evaluate_one_step,
        json_ready,
        relative_l2_torch,
        rollout_case,
        summarize_rollouts,
    )
else:
    from train_euler1d_target_ladder import (
        PrimitiveNormalizer,
        build_model,
        evaluate_one_step,
        json_ready,
        relative_l2_torch,
        rollout_case,
        summarize_rollouts,
    )
from utility.time_dependent_no.euler1d_data import (
    Euler1DNPZ,
    Euler1DTimePairDataset,
    collate_euler1d_pairs,
    conservative_restrict,
    load_euler1d_npz,
    primitive_to_conservative_np as _primitive_to_conservative_np,
)
from utility.time_dependent_no.euler1d import make_euler1d_batch
from utility.time_dependent_no.euler1d_targets import make_target_adapter


REQUIRED_PHYSICAL_METADATA = ("domains", "x_disc", "t_final")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        nargs=2,
        action="append",
        metavar=("LABEL", "PATH"),
        required=True,
        help="Frozen checkpoint label and path; repeat for multiple models.",
    )
    parser.add_argument(
        "--data",
        nargs=2,
        action="append",
        metavar=("LABEL", "PATH"),
        required=True,
        help="Resolution label and matched dataset path; repeat for each mesh.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=[2, 20, 50, 100],
        help="Physical saved-frame horizons; incompatible strides are skipped.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--torch-threads", type=int, default=1)
    return parser.parse_args()


def _labeled_paths(entries: list[list[str]], kind: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for label, raw_path in entries:
        if label in result:
            raise ValueError(f"duplicate {kind} label: {label}")
        result[label] = Path(raw_path)
    return result


def _metadata_array(source: Euler1DNPZ, key: str) -> np.ndarray:
    if key not in source.metadata:
        raise ValueError(f"dataset is missing required physical metadata: {key}")
    return np.asarray(source.metadata[key])


def _uniform_geometry_summary(source: Euler1DNPZ) -> dict[str, float | int]:
    dx = np.diff(source.x.astype(np.float64), axis=1)
    if np.any(dx <= 0.0):
        raise ValueError("cell centers must be strictly increasing")
    mean_dx = dx.mean(axis=1, keepdims=True)
    uniformity_error = float(np.max(np.abs(dx - mean_dx)))
    domains = _metadata_array(source, "domains").astype(np.float64)
    left = source.x[:, 0].astype(np.float64) - 0.5 * dx[:, 0]
    right = source.x[:, -1].astype(np.float64) + 0.5 * dx[:, -1]
    endpoint_error = float(
        max(np.max(np.abs(left - domains[:, 0])), np.max(np.abs(right - domains[:, 1])))
    )
    scale = float(max(1.0, np.max(np.abs(domains))))
    if uniformity_error > 2.0e-6 * scale or endpoint_error > 2.0e-6 * scale:
        raise ValueError(
            "dataset coordinates do not reconstruct the serialized uniform domains"
        )
    return {
        "num_cells": source.num_cells,
        "uniformity_max_abs": uniformity_error,
        "domain_endpoint_max_abs": endpoint_error,
        "dx_min": float(dx.min()),
        "dx_max": float(dx.max()),
    }


def validate_resolution_family(
    sources: dict[str, Euler1DNPZ],
) -> dict[str, Any]:
    """Require identical physical cases/times and valid native uniform grids."""

    if not sources:
        raise ValueError("at least one dataset is required")
    reference_label = next(iter(sources))
    reference = sources[reference_label]
    reference.validate()
    details: dict[str, Any] = {}
    seen_resolutions: set[int] = set()

    for label, source in sources.items():
        source.validate()
        if source.num_cells in seen_resolutions:
            raise ValueError(f"duplicate mesh resolution: {source.num_cells}")
        seen_resolutions.add(source.num_cells)
        if source.num_cases != reference.num_cases:
            raise ValueError(f"{label}: case count differs from {reference_label}")
        if source.num_frames != reference.num_frames:
            raise ValueError(f"{label}: frame count differs from {reference_label}")
        if source.gamma != reference.gamma:
            raise ValueError(f"{label}: gamma differs from {reference_label}")
        for name, value, expected in (
            ("saved times", source.t, reference.t),
            ("left states", source.left_states, reference.left_states),
            ("right states", source.right_states, reference.right_states),
        ):
            if not np.array_equal(value, expected):
                raise ValueError(f"{label}: {name} differ from {reference_label}")
        for key in REQUIRED_PHYSICAL_METADATA:
            if not np.array_equal(
                _metadata_array(source, key),
                _metadata_array(reference, key),
            ):
                raise ValueError(
                    f"{label}: physical metadata {key} differs from {reference_label}"
                )
        details[label] = _uniform_geometry_summary(source)

    return {
        "reference_label": reference_label,
        "physical_case_identity_exact": True,
        "saved_time_identity_exact": True,
        "datasets": details,
    }


def native_reference_gaps(
    sources: dict[str, Euler1DNPZ],
    case_ids: np.ndarray,
    frame_ids: list[int],
) -> list[dict[str, Any]]:
    """Measure adjacent-grid solver/reference differences after FV restriction."""

    ordered = sorted(sources.items(), key=lambda item: item[1].num_cells)
    rows: list[dict[str, Any]] = []
    for (coarse_label, coarse), (fine_label, fine) in zip(ordered, ordered[1:]):
        if fine.num_cells % coarse.num_cells != 0:
            continue
        for frame in frame_ids:
            native = coarse.data[case_ids, frame].astype(np.float64)
            restricted = conservative_restrict(
                fine.data[case_ids, frame].astype(np.float64),
                coarse.num_cells,
                coarse.gamma,
            )
            diff = (restricted - native).reshape(case_ids.size, -1)
            denom = np.linalg.norm(native.reshape(case_ids.size, -1), axis=1)
            relative = np.linalg.norm(diff, axis=1) / np.maximum(denom, 1.0e-12)
            rows.append(
                {
                    "coarse_label": coarse_label,
                    "fine_label": fine_label,
                    "coarse_cells": coarse.num_cells,
                    "fine_cells": fine.num_cells,
                    "frame": frame,
                    "relative_l2_mean": float(relative.mean()),
                    "relative_l2_median": float(np.median(relative)),
                    "relative_l2_max": float(relative.max()),
                }
            )
    return rows


def native_update_gaps(
    sources: dict[str, Euler1DNPZ],
    case_ids: np.ndarray,
    strides: list[int],
) -> list[dict[str, Any]]:
    """Compare adjacent-grid conservative increments along paired trajectories.

    This is not a same-input operator commutator because each native solver
    evolves its own grid state.  It measures the update-label mismatch that a
    frozen model encounters when it is moved between those native trajectories.
    """

    ordered = sorted(sources.items(), key=lambda item: item[1].num_cells)
    rows: list[dict[str, Any]] = []
    for (coarse_label, coarse), (fine_label, fine) in zip(ordered, ordered[1:]):
        if fine.num_cells % coarse.num_cells != 0:
            continue
        for stride in sorted(set(strides)):
            for input_frame in range(coarse.num_frames - stride):
                target_frame = input_frame + stride
                coarse_current = _primitive_to_conservative_np(
                    coarse.data[case_ids, input_frame].astype(np.float64),
                    coarse.gamma,
                )
                coarse_target = _primitive_to_conservative_np(
                    coarse.data[case_ids, target_frame].astype(np.float64),
                    coarse.gamma,
                )
                fine_current = _primitive_to_conservative_np(
                    conservative_restrict(
                        fine.data[case_ids, input_frame].astype(np.float64),
                        coarse.num_cells,
                        coarse.gamma,
                    ),
                    coarse.gamma,
                )
                fine_target = _primitive_to_conservative_np(
                    conservative_restrict(
                        fine.data[case_ids, target_frame].astype(np.float64),
                        coarse.num_cells,
                        coarse.gamma,
                    ),
                    coarse.gamma,
                )
                coarse_delta = coarse_target - coarse_current
                fine_delta = fine_target - fine_current
                difference = (fine_delta - coarse_delta).reshape(case_ids.size, -1)
                target_norm = np.linalg.norm(
                    coarse_target.reshape(case_ids.size, -1), axis=1
                )
                update_norm = np.linalg.norm(
                    coarse_delta.reshape(case_ids.size, -1), axis=1
                )
                difference_norm = np.linalg.norm(difference, axis=1)
                relative_state = difference_norm / np.maximum(target_norm, 1.0e-12)
                relative_update = difference_norm / np.maximum(update_norm, 1.0e-12)
                rows.append(
                    {
                        "coarse_label": coarse_label,
                        "fine_label": fine_label,
                        "coarse_cells": coarse.num_cells,
                        "fine_cells": fine.num_cells,
                        "stride": stride,
                        "input_frame": input_frame,
                        "target_frame": target_frame,
                        "state_normalized_mean": float(relative_state.mean()),
                        "state_normalized_median": float(np.median(relative_state)),
                        "update_normalized_mean": float(relative_update.mean()),
                        "update_normalized_median": float(np.median(relative_update)),
                    }
                )
    return rows


def _checkpoint_normalizer(
    checkpoint: dict[str, Any],
    prefix: str,
    coordinates: str,
    normalization: str,
) -> PrimitiveNormalizer:
    mean = torch.as_tensor(checkpoint[f"{prefix}_mean"], dtype=torch.float32).reshape(
        1, 1, 3
    )
    std = torch.as_tensor(checkpoint[f"{prefix}_std"], dtype=torch.float32).reshape(
        1, 1, 3
    )
    return PrimitiveNormalizer(
        mean=mean,
        std=std,
        coordinates=coordinates,
        normalization=normalization,
    )


def load_frozen_residual_checkpoint(
    path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, torch.nn.Module, PrimitiveNormalizer, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("model") != "fno" or checkpoint.get("target") != "residual":
        raise ValueError("resolution gate requires an FNO residual checkpoint")
    checkpoint_args = dict(checkpoint.get("args", {}))
    required_args = (
        "fno_modes",
        "fno_width",
        "fno_layers",
        "fno_fc_dim",
        "fno_pad_ratio",
        "input_coordinates",
        "input_normalization",
        "loss_coordinates",
        "loss_normalization",
        "recurrent_coordinates",
        "step_stride",
        "target_supervision",
    )
    missing = [key for key in required_args if key not in checkpoint_args]
    if missing:
        raise ValueError(f"checkpoint is missing required arguments: {missing}")
    expected_contract = {
        "input_coordinates": "conservative",
        "input_normalization": "fixed_physical",
        "loss_coordinates": "conservative",
        "loss_normalization": "fixed_physical",
        "recurrent_coordinates": "conservative",
        "target_supervision": "state",
    }
    for key, expected in expected_contract.items():
        if checkpoint_args[key] != expected:
            raise ValueError(
                f"checkpoint {key}={checkpoint_args[key]!r}, expected {expected!r}"
            )
    namespace = argparse.Namespace(**checkpoint_args)
    input_normalizer = _checkpoint_normalizer(
        checkpoint,
        "input_normalizer",
        checkpoint_args["input_coordinates"],
        checkpoint_args["input_normalization"],
    )
    loss_normalizer = _checkpoint_normalizer(
        checkpoint,
        "normalizer",
        checkpoint_args["loss_coordinates"],
        checkpoint_args["loss_normalization"],
    )
    model = build_model("fno", "residual", namespace, input_normalizer).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != int(checkpoint["parameter_count"]):
        raise ValueError("restored parameter count differs from checkpoint metadata")
    adapter = make_target_adapter("residual").to(device)
    return model, adapter, loss_normalizer, checkpoint


def evaluate_one_step_by_input_frame(
    model: torch.nn.Module,
    adapter: torch.nn.Module,
    source: Euler1DNPZ,
    test_cases: np.ndarray,
    step_stride: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    """Return native-grid teacher-forced relative error for each input time."""

    model.eval()
    rows: list[dict[str, Any]] = []
    case_ids = np.asarray(test_cases, dtype=np.int64)
    x = torch.from_numpy(source.x[case_ids]).to(device)
    left = torch.from_numpy(source.left_states[case_ids]).to(device)
    right = torch.from_numpy(source.right_states[case_ids]).to(device)
    with torch.no_grad():
        for input_frame in range(source.num_frames - step_stride):
            target_frame = input_frame + step_stride
            current = torch.from_numpy(
                np.asarray(source.data[case_ids, input_frame], dtype=np.float32)
            ).to(device)
            target = torch.from_numpy(
                np.asarray(source.data[case_ids, target_frame], dtype=np.float32)
            ).to(device)
            dt = torch.from_numpy(
                source.t[case_ids, target_frame] - source.t[case_ids, input_frame]
            ).to(device)
            batch = make_euler1d_batch(
                current,
                x,
                dt,
                gamma=source.gamma,
                left_boundary_primitive=left,
                right_initial_primitive=right,
            )
            prediction = adapter(model(batch), batch)
            relative = relative_l2_torch(prediction.primitive, target).cpu().numpy()
            rows.append(
                {
                    "input_frame": input_frame,
                    "target_frame": target_frame,
                    "relative_l2_mean": float(relative.mean()),
                    "relative_l2_median": float(np.median(relative)),
                    "relative_l2_max": float(relative.max()),
                    "min_density": float(prediction.primitive[..., 0].min().cpu()),
                    "min_pressure": float(prediction.primitive[..., 2].min().cpu()),
                }
            )
    return rows


def evaluate_checkpoint_on_source(
    model: torch.nn.Module,
    adapter: torch.nn.Module,
    loss_normalizer: PrimitiveNormalizer,
    checkpoint: dict[str, Any],
    source: Euler1DNPZ,
    horizons: list[int],
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint_args = checkpoint["args"]
    test_cases = np.asarray(checkpoint["test_cases"], dtype=np.int64)
    stride = int(checkpoint_args["step_stride"])
    modes = int(checkpoint_args["fno_modes"])
    if source.num_cells // 2 + 1 < modes:
        raise ValueError(
            f"{source.num_cells} cells do not provide {modes} requested Fourier modes"
        )
    dataset = Euler1DTimePairDataset(
        source,
        case_indices=test_cases,
        step_stride=stride,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_euler1d_pairs,
    )
    one_step = evaluate_one_step(
        model,
        adapter,
        loader,
        loss_normalizer.to(device),
        device,
        loss_coordinates=checkpoint_args["loss_coordinates"],
        target_supervision="state",
    )
    one_step["by_input_frame"] = evaluate_one_step_by_input_frame(
        model,
        adapter,
        source,
        test_cases,
        stride,
        device,
    )
    rollout: dict[str, Any] = {}
    for horizon in sorted(set(horizons)):
        if horizon < stride or horizon >= source.num_frames or horizon % stride:
            continue
        rows = [
            rollout_case(
                model,
                adapter,
                source,
                int(case_id),
                horizon // stride,
                stride,
                device,
                final_frame=horizon,
                recurrent_coordinates=checkpoint_args["recurrent_coordinates"],
            )
            for case_id in test_cases.tolist()
        ]
        rollout[str(horizon)] = {
            "summary": summarize_rollouts(rows),
            "cases": rows,
        }
    return {
        "num_cells": source.num_cells,
        "step_stride": stride,
        "test_cases": test_cases,
        "one_step": one_step,
        "rollout": rollout,
    }


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.torch_threads < 1:
        raise ValueError("batch size and torch thread count must be positive")
    if any(horizon < 0 for horizon in args.horizons):
        raise ValueError("horizons must be nonnegative saved-frame indices")
    torch.set_num_threads(args.torch_threads)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    checkpoint_paths = _labeled_paths(args.checkpoint, "checkpoint")
    data_paths = _labeled_paths(args.data, "data")
    sources = {label: load_euler1d_npz(path) for label, path in data_paths.items()}
    contract = validate_resolution_family(sources)

    started = time.perf_counter()
    results: dict[str, Any] = {}
    summary_rows: list[dict[str, Any]] = []
    common_test_cases: np.ndarray | None = None
    checkpoint_strides: set[int] = set()
    for checkpoint_label, checkpoint_path in checkpoint_paths.items():
        model, adapter, normalizer, checkpoint = load_frozen_residual_checkpoint(
            checkpoint_path,
            device,
        )
        checkpoint_strides.add(int(checkpoint["args"]["step_stride"]))
        test_cases = np.asarray(checkpoint["test_cases"], dtype=np.int64)
        if common_test_cases is None:
            common_test_cases = test_cases
        elif not np.array_equal(common_test_cases, test_cases):
            raise ValueError("all checkpoints must use the same frozen test cases")
        results[checkpoint_label] = {}
        for data_label, source in sources.items():
            evaluated = evaluate_checkpoint_on_source(
                model,
                adapter,
                normalizer,
                checkpoint,
                source,
                args.horizons,
                args.batch_size,
                device,
            )
            results[checkpoint_label][data_label] = evaluated
            for horizon, horizon_result in evaluated["rollout"].items():
                summary_rows.append(
                    {
                        "checkpoint": checkpoint_label,
                        "data": data_label,
                        "num_cells": source.num_cells,
                        "step_stride": evaluated["step_stride"],
                        "horizon_frame": int(horizon),
                        "one_step_relative_l2": evaluated["one_step"]["relative_l2"],
                        **horizon_result["summary"],
                    }
                )

    if common_test_cases is None:
        raise RuntimeError("no checkpoints were evaluated")
    frames = (
        sorted(
            set(
                [
                    0,
                    *[
                        horizon
                        for horizon in args.horizons
                        if horizon < reference.num_frames
                    ],
                ]
            )
        )
        if (reference := next(iter(sources.values())))
        else []
    )
    reference_gaps = native_reference_gaps(sources, common_test_cases, frames)
    update_gaps = native_update_gaps(
        sources,
        common_test_cases,
        sorted(checkpoint_strides),
    )
    payload = {
        "native_update_gaps": update_gaps,
        "status": "ok",
        "device": str(device),
        "checkpoint_paths": checkpoint_paths,
        "data_paths": data_paths,
        "contract": contract,
        "native_reference_gaps": reference_gaps,
        "results": results,
        "runtime_seconds": time.perf_counter() - started,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(json_ready(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_summary_csv(args.output_dir / "summary.csv", summary_rows)
    _write_summary_csv(args.output_dir / "native_reference_gaps.csv", reference_gaps)
    print(json.dumps(json_ready(payload["contract"]), indent=2, sort_keys=True))
    print(f"wrote resolution-transfer results to {args.output_dir}")


if __name__ == "__main__":
    main()
