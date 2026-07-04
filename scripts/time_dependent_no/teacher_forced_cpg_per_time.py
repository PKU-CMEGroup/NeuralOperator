"""Teacher-forced per-time evaluation for official CPGNet checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utility.time_dependent_no.euler2d import PRIMITIVE_NAMES  # noqa: E402


def by_var(values: Sequence[float] | np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {name: float(arr[index]) for index, name in enumerate(PRIMITIVE_NAMES)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--rollout-num", type=int, default=20)
    parser.add_argument("--total-steps", type=int, default=80)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


def rmse(sse: np.ndarray, count: np.ndarray | int) -> np.ndarray:
    return np.sqrt(sse / np.maximum(np.asarray(count, dtype=np.float64), 1.0))


def main() -> None:
    args = parse_args()
    if args.total_steps < 2:
        raise SystemExit("--total-steps must be at least 2")
    if str(args.official_root) not in sys.path:
        sys.path.insert(0, str(args.official_root))

    from dataset.fpcMulti import FPC_ROLLOUT
    from modelEdgeUpd.simulator import Simulator
    from utils.to_undirected import make_edges_undirected
    from utils.utils import NodeType

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    checkpoint = args.checkpoint or (args.run_dir / "checkpoint" / "simulator.pth")
    model = Simulator(
        message_passing_num=12,
        node_input_size=6,
        edge_input_size=5,
        device=device,
        model_dir=str(checkpoint),
    ).to(device)
    model.load_checkpoint(str(checkpoint))
    model.eval()

    dataset = FPC_ROLLOUT(str(args.dataset_root), split="test")
    loader = DataLoader(dataset=dataset, batch_size=1)
    transformer = T.Compose(
        [T.NormalizeScale(), T.Cartesian(norm=False), T.Distance(norm=False)]
    )
    num_steps = args.total_steps - 1
    all_sse_by_time = np.zeros((num_steps, len(PRIMITIVE_NAMES)), dtype=np.float64)
    all_count_by_time = np.zeros(num_steps, dtype=np.int64)
    normal_sse_by_time = np.zeros((num_steps, len(PRIMITIVE_NAMES)), dtype=np.float64)
    normal_count_by_time = np.zeros(num_steps, dtype=np.int64)
    per_trajectory: list[dict[str, Any]] = []
    start = time.time()

    with torch.no_grad():
        for trajectory_index in range(args.rollout_num):
            dataset.change_file(trajectory_index)
            traj_all_sse = np.zeros((num_steps, len(PRIMITIVE_NAMES)), dtype=np.float64)
            traj_all_count = np.zeros(num_steps, dtype=np.int64)
            traj_normal_sse = np.zeros(
                (num_steps, len(PRIMITIVE_NAMES)), dtype=np.float64
            )
            traj_normal_count = np.zeros(num_steps, dtype=np.int64)
            iterator = enumerate(loader)
            if args.progress:
                iterator = enumerate(tqdm(loader, total=args.total_steps))

            for step, graph in iterator:
                if step >= num_steps:
                    break
                graph = make_edges_undirected(graph)
                graph = transformer(graph)
                graph = graph.to(device)

                node_type = graph.x[:, 0]
                normal_mask = node_type == NodeType.NORMAL
                boundary_mask = torch.logical_not(normal_mask)
                target = graph.y
                graph.x[boundary_mask, 1:5] = target[boundary_mask]
                prediction = model(graph, sequence_noise=None)
                prediction[boundary_mask] = target[boundary_mask]

                err = (prediction - target).detach().cpu().numpy().astype(np.float64)
                normal = normal_mask.detach().cpu().numpy().astype(bool)
                squared = err * err
                traj_all_sse[step] = np.sum(squared, axis=0)
                traj_all_count[step] = err.shape[0]
                traj_normal_sse[step] = np.sum(squared[normal], axis=0)
                traj_normal_count[step] = int(np.count_nonzero(normal))

            all_sse_by_time += traj_all_sse
            all_count_by_time += traj_all_count
            normal_sse_by_time += traj_normal_sse
            normal_count_by_time += traj_normal_count
            per_trajectory.append(
                {
                    "trajectory_index": int(trajectory_index),
                    "all_node_count": int(np.sum(traj_all_count)),
                    "normal_node_count": int(np.sum(traj_normal_count)),
                    "all_node_rmse": by_var(
                        rmse(np.sum(traj_all_sse, axis=0), int(np.sum(traj_all_count)))
                    ),
                    "normal_node_rmse": by_var(
                        rmse(
                            np.sum(traj_normal_sse, axis=0),
                            int(np.sum(traj_normal_count)),
                        )
                    ),
                    "normal_node_rmse_by_time": [
                        by_var(values)
                        for values in rmse(traj_normal_sse, traj_normal_count[:, None])
                    ],
                }
            )

    all_rmse_by_time = rmse(all_sse_by_time, all_count_by_time[:, None])
    normal_rmse_by_time = rmse(normal_sse_by_time, normal_count_by_time[:, None])
    summary = {
        "mode": "teacher_forced_per_time_eval",
        "run_dir": str(args.run_dir),
        "checkpoint": str(checkpoint),
        "dataset_root": str(args.dataset_root),
        "device": str(device),
        "rollout_num": int(args.rollout_num),
        "total_steps_requested": int(args.total_steps),
        "num_evaluated_steps": int(num_steps),
        "elapsed_seconds": float(time.time() - start),
        "all_node_count": int(np.sum(all_count_by_time)),
        "normal_node_count": int(np.sum(normal_count_by_time)),
        "all_node_rmse": by_var(
            rmse(np.sum(all_sse_by_time, axis=0), int(np.sum(all_count_by_time)))
        ),
        "normal_node_rmse": by_var(
            rmse(np.sum(normal_sse_by_time, axis=0), int(np.sum(normal_count_by_time)))
        ),
        "all_node_rmse_by_time": [by_var(values) for values in all_rmse_by_time],
        "normal_node_rmse_by_time": [by_var(values) for values in normal_rmse_by_time],
        "all_node_count_by_time": [int(value) for value in all_count_by_time],
        "normal_node_count_by_time": [int(value) for value in normal_count_by_time],
        "per_trajectory": per_trajectory,
    }
    output = args.output or (
        args.run_dir / "metrics" / "teacher_forced_per_time_summary.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {"output": str(output), "normal_node_rmse": summary["normal_node_rmse"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
