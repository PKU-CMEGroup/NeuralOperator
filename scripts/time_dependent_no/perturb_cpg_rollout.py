"""Controlled perturbation amplification probe for official CPGNet rollouts."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utility.time_dependent_no.euler2d import PRIMITIVE_NAMES  # noqa: E402

MODE_TO_INDEX = {name: index for index, name in enumerate(PRIMITIVE_NAMES)}
MODE_TO_INDEX["all"] = -1
EPS = 1.0e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--run", action="append", required=True, help="NAME=run_dir")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--trajectory-indices", nargs="+", type=int, default=[0, 6, 11, 13, 17]
    )
    parser.add_argument("--start-frames", nargs="+", type=int, default=[0, 20, 40, 58])
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--milestones", nargs="+", type=int, default=[1, 5, 10, 20])
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=sorted(MODE_TO_INDEX),
        default=["all", "rho", "v1", "v2", "pres"],
    )
    parser.add_argument(
        "--profile", action="append", default=None, help="NAME=s_rho,s_v1,s_v2,s_pres"
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def parse_profiles(values: Sequence[str] | None) -> dict[str, np.ndarray]:
    if not values:
        return {
            "train_noise": np.array([2.0e-2, 2.0e-2, 1.0e-2, 2.0e-2], dtype=np.float64),
            "tf_rmse": np.array([7.6e-3, 4.1e-3, 3.1e-3, 1.45e-2], dtype=np.float64),
        }
    profiles = {}
    for raw in values:
        name, payload = raw.split("=", 1)
        parts = [float(item) for item in payload.replace(",", " ").split()]
        if len(parts) != len(PRIMITIVE_NAMES):
            raise ValueError(f"profile {raw!r} must provide four scales")
        profiles[name.strip()] = np.array(parts, dtype=np.float64)
    return profiles


def by_var(values: Sequence[float] | np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {name: float(arr[index]) for index, name in enumerate(PRIMITIVE_NAMES)}


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(gpu: int) -> torch.device:
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu)
        return torch.device("cuda")
    return torch.device("cpu")


def make_transformer():
    return T.Compose(
        [T.NormalizeScale(), T.Cartesian(norm=False), T.Distance(norm=False)]
    )


def make_model(run_dir: Path, device: torch.device):
    from modelEdgeUpd.simulator import Simulator

    checkpoint = run_dir / "checkpoint" / "simulator.pth"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    model = Simulator(
        message_passing_num=12,
        node_input_size=6,
        edge_input_size=5,
        device=device,
        model_dir=str(checkpoint),
    ).to(device)
    model.load_checkpoint(str(checkpoint))
    model.eval()
    return model, checkpoint


def load_graphs(
    dataset, loader, transformer, trajectory_index: int, max_step: int
) -> list[Any]:
    from utils.to_undirected import make_edges_undirected

    dataset.change_file(trajectory_index)
    graphs = []
    for step, graph in enumerate(loader):
        if step > max_step:
            break
        graph = make_edges_undirected(graph)
        graph = transformer(graph)
        graphs.append(graph.cpu())
    return graphs


def normal_boundary_masks(
    graph, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    from utils.utils import NodeType

    node_type = graph.x[:, 0].to(device)
    normal = node_type == NodeType.NORMAL
    boundary = torch.logical_not(normal)
    return normal, boundary


def initial_delta(
    graph,
    normal: torch.Tensor,
    profile: np.ndarray,
    mode: str,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    noise = torch.randn(
        (graph.x.shape[0], len(PRIMITIVE_NAMES)), generator=generator, device=device
    )
    scales = torch.as_tensor(profile, dtype=torch.float32, device=device).reshape(1, -1)
    delta = noise * scales
    if mode != "all":
        selected = MODE_TO_INDEX[mode]
        mask = torch.zeros_like(delta)
        mask[:, selected] = delta[:, selected]
        delta = mask
    delta[torch.logical_not(normal)] = 0.0
    return delta


def rollout_from(
    model,
    graphs: list[Any],
    start: int,
    horizon: int,
    device: torch.device,
    delta0: torch.Tensor | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
    predicted_prims = None
    predictions = []
    targets = []
    first_graph = graphs[start].clone().to(device)
    normal, boundary = normal_boundary_masks(first_graph, device)

    with torch.no_grad():
        for offset in range(horizon):
            graph = graphs[start + offset].clone().to(device)
            next_v = graph.y
            if predicted_prims is not None:
                graph.x[:, 1:5] = predicted_prims.detach()
            elif delta0 is not None:
                graph.x[:, 1:5] = graph.x[:, 1:5] + delta0
            graph.x[boundary, 1:5] = next_v[boundary]
            predicted_prims = model(graph, sequence_noise=None)
            predicted_prims[boundary] = next_v[boundary]
            predictions.append(predicted_prims.detach().cpu())
            targets.append(next_v.detach().cpu())
    return predictions, targets, normal.detach().cpu()


def l2_norm(values: np.ndarray, mask: np.ndarray) -> float:
    selected = values[mask]
    return float(np.sqrt(np.sum(selected * selected)))


def rmse_by_var(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    selected = values[mask]
    return np.sqrt(np.mean(selected * selected, axis=0))


def summarize_case(
    *,
    run_name: str,
    trajectory_index: int,
    start_frame: int,
    profile_name: str,
    mode: str,
    seed: int,
    initial: np.ndarray,
    base_predictions: list[torch.Tensor],
    pert_predictions: list[torch.Tensor],
    targets: list[torch.Tensor],
    normal: torch.Tensor,
    milestones: Sequence[int],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    normal_np = normal.numpy().astype(bool)
    initial_norm = l2_norm(initial, normal_np)
    initial_rmse = rmse_by_var(initial, normal_np)
    rows = []
    amplification = []
    base_target_rmse = []
    pert_target_rmse = []
    for step, (base, pert, target) in enumerate(
        zip(base_predictions, pert_predictions, targets, strict=True), start=1
    ):
        base_np = base.numpy().astype(np.float64)
        pert_np = pert.numpy().astype(np.float64)
        target_np = target.numpy().astype(np.float64)
        delta_np = pert_np - base_np
        base_err = base_np - target_np
        pert_err = pert_np - target_np
        amp = l2_norm(delta_np, normal_np) / max(initial_norm, EPS)
        base_rmse = rmse_by_var(base_err, normal_np)
        pert_rmse = rmse_by_var(pert_err, normal_np)
        amplification.append(amp)
        base_target_rmse.append(base_rmse)
        pert_target_rmse.append(pert_rmse)
        if step in milestones:
            row = {
                "run": run_name,
                "trajectory_index": int(trajectory_index),
                "start_frame": int(start_frame),
                "profile": profile_name,
                "mode": mode,
                "seed": int(seed),
                "step": int(step),
                "amplification_l2": float(amp),
            }
            for index, name in enumerate(PRIMITIVE_NAMES):
                row[f"initial_rmse_{name}"] = float(initial_rmse[index])
                row[f"base_target_rmse_{name}"] = float(base_rmse[index])
                row[f"perturbed_target_rmse_{name}"] = float(pert_rmse[index])
                row[f"excess_target_rmse_{name}"] = float(
                    pert_rmse[index] - base_rmse[index]
                )
            rows.append(row)
    amp_arr = np.asarray(amplification, dtype=np.float64)
    summary = {
        "run": run_name,
        "trajectory_index": int(trajectory_index),
        "start_frame": int(start_frame),
        "profile": profile_name,
        "mode": mode,
        "seed": int(seed),
        "initial_l2": float(initial_norm),
        "initial_rmse": by_var(initial_rmse),
        "amplification_by_step": {
            str(step): float(amp_arr[step - 1])
            for step in milestones
            if step <= amp_arr.size
        },
        "max_amplification": float(np.max(amp_arr)),
        "final_amplification": float(amp_arr[-1]),
        "recovered_vs_step1": bool(amp_arr[-1] < amp_arr[0]),
    }
    return summary, rows


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["run"], row["profile"], row["mode"], int(row["step"]))
        grouped.setdefault(key, []).append(row)
    out = []
    for (run, profile, mode, step), items in sorted(grouped.items()):
        values = np.array(
            [item["amplification_l2"] for item in items], dtype=np.float64
        )
        entry = {
            "run": run,
            "profile": profile,
            "mode": mode,
            "step": int(step),
            "count": int(len(items)),
            "amplification_mean": float(np.mean(values)),
            "amplification_median": float(np.median(values)),
            "amplification_max": float(np.max(values)),
        }
        for name in PRIMITIVE_NAMES:
            excess = np.array(
                [item[f"excess_target_rmse_{name}"] for item in items], dtype=np.float64
            )
            entry[f"excess_target_rmse_{name}_mean"] = float(np.mean(excess))
        out.append(entry)
    return {"by_run_profile_mode_step": out}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    seed_all(args.seed)
    if str(args.official_root) not in sys.path:
        sys.path.insert(0, str(args.official_root))
    from dataset.fpcMulti import FPC_ROLLOUT

    profiles = parse_profiles(args.profile)
    max_milestone = max(args.milestones)
    if args.horizon < max_milestone:
        raise SystemExit("--horizon must be at least the largest milestone")
    max_required_step = max(args.start_frames) + args.horizon - 1
    device = select_device(args.gpu)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    run_specs = [parse_run(raw) for raw in args.run]
    dataset = FPC_ROLLOUT(str(args.dataset_root), split="test")
    loader = DataLoader(dataset=dataset, batch_size=1)
    transformer = make_transformer()
    all_rows: list[dict[str, Any]] = []
    case_summaries: list[dict[str, Any]] = []
    start_time = time.time()

    for run_name, run_dir in run_specs:
        model, checkpoint = make_model(run_dir, device)
        for trajectory_index in args.trajectory_indices:
            graphs = load_graphs(
                dataset, loader, transformer, trajectory_index, max_required_step
            )
            if len(graphs) <= max_required_step:
                raise RuntimeError(
                    f"trajectory {trajectory_index} yielded {len(graphs)} graphs, need index {max_required_step}"
                )
            for start_frame in args.start_frames:
                base_predictions, targets, normal = rollout_from(
                    model, graphs, start_frame, args.horizon, device, delta0=None
                )
                first_graph = graphs[start_frame].clone().to(device)
                normal_device, _ = normal_boundary_masks(first_graph, device)
                for profile_name, profile in profiles.items():
                    for mode in args.modes:
                        delta = initial_delta(
                            first_graph,
                            normal_device,
                            profile,
                            mode,
                            args.seed
                            + 100000 * trajectory_index
                            + 1000 * start_frame
                            + len(case_summaries),
                            device,
                        )
                        pert_predictions, _, _ = rollout_from(
                            model,
                            graphs,
                            start_frame,
                            args.horizon,
                            device,
                            delta0=delta,
                        )
                        summary, rows = summarize_case(
                            run_name=run_name,
                            trajectory_index=trajectory_index,
                            start_frame=start_frame,
                            profile_name=profile_name,
                            mode=mode,
                            seed=args.seed,
                            initial=delta.detach().cpu().numpy().astype(np.float64),
                            base_predictions=base_predictions,
                            pert_predictions=pert_predictions,
                            targets=targets,
                            normal=normal,
                            milestones=args.milestones,
                        )
                        case_summaries.append(summary)
                        all_rows.extend(rows)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    payload = {
        "mode": "perturbation_amplification",
        "dataset_root": str(args.dataset_root),
        "runs": {name: str(path) for name, path in run_specs},
        "trajectory_indices": [int(value) for value in args.trajectory_indices],
        "start_frames": [int(value) for value in args.start_frames],
        "horizon": int(args.horizon),
        "milestones": [int(value) for value in args.milestones],
        "profiles": {name: by_var(values) for name, values in profiles.items()},
        "modes": list(args.modes),
        "seed": int(args.seed),
        "elapsed_seconds": float(time.time() - start_time),
        "case_summaries": case_summaries,
        "aggregate": aggregate(all_rows),
    }
    summary_path = args.output_dir / "perturbation_amplification_summary.json"
    csv_path = args.output_dir / "perturbation_amplification_milestones.csv"
    summary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_csv(csv_path, all_rows)
    print(
        json.dumps(
            {"summary": str(summary_path), "milestones": str(csv_path)}, indent=2
        )
    )


def parse_run(raw: str) -> tuple[str, Path]:
    if "=" in raw:
        name, value = raw.split("=", 1)
        return name.strip(), Path(value)
    path = Path(raw)
    return path.name, path


if __name__ == "__main__":
    main()
