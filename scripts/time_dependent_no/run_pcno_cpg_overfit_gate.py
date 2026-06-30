from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from timeit import default_timer
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pcno.pcno import PCNO, compute_Fourier_modes
from utility.time_dependent_no.euler2d import make_cpg_graph_frame
from utility.time_dependent_no.euler2d_synthetic import (
    SyntheticEuler2DConfig,
    make_synthetic_cpg_trajectory,
)
from utility.time_dependent_no.pcno_adapter import PcnoFrameBatch, make_pcno_frame_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a small PCNO one-trajectory overfit gate on CPG Euler frames."
    )
    parser.add_argument("path", type=Path, nargs="?", help="Path to train.h5 or test.h5")
    parser.add_argument("--trajectory", default=None, help="Trajectory group name")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--k-max", type=int, default=4)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--rcond", type=float, default=1.0e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Training device. 'auto' uses CUDA when available.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only build tensors and run one forward pass.",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use a small synthetic fixture instead of a real HDF5 file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/time_dependent_no/pcno_cpg_overfit_gate.json"),
        help="JSON summary path. Keep this under an ignored directory for real data.",
    )
    parser.add_argument(
        "--save-model",
        type=Path,
        default=None,
        help="Optional model state_dict path. Keep this under an ignored directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _validate_args(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    frames, source = _load_frames(args)
    batch = _stack_batches(
        [
            make_pcno_frame_batch(frame, rcond=args.rcond)
            for frame in frames
        ]
    )
    device = _select_device(args.device)
    model = _make_model(batch, args).to(device)
    tensors = _to_tensors(batch, device)

    t0 = default_timer()
    initial = _evaluate(model, tensors, args.batch_size)
    history: list[dict[str, float | int]] = []
    if args.dry_run:
        final = initial
    else:
        history = _train(model, tensors, args)
        final = _evaluate(model, tensors, args.batch_size)
    elapsed = default_timer() - t0

    payload: dict[str, Any] = {
        "kind": "pcno_cpg_overfit_gate",
        "source": source,
        "dry_run": bool(args.dry_run),
        "device": str(device),
        "seed": int(args.seed),
        "config": {
            "num_frames": int(args.num_frames),
            "start_frame": int(args.start_frame),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "k_max": int(args.k_max),
            "width": int(args.width),
            "num_layers": int(args.num_layers),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "rcond": float(args.rcond),
        },
        "shapes": {
            "x": list(batch.x.shape),
            "y": list(batch.y.shape),
            "node_mask": list(batch.node_mask.shape),
            "nodes": list(batch.nodes.shape),
            "node_weights": list(batch.node_weights.shape),
            "directed_edges": list(batch.directed_edges.shape),
            "edge_gradient_weights": list(batch.edge_gradient_weights.shape),
        },
        "adapter_metadata": batch.metadata,
        "initial": initial,
        "final": final,
        "history": history,
        "elapsed_seconds": float(elapsed),
        "model_parameter_count": int(sum(p.numel() for p in model.parameters())),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True))
    if args.save_model is not None:
        args.save_model.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), args.save_model)
        payload["saved_model"] = str(args.save_model)
        args.output.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True))

    print(f"source: {source}")
    print(f"device: {device}")
    print(f"x: {payload['shapes']['x']}  y: {payload['shapes']['y']}")
    print(f"initial rel_l2: {initial['rel_l2']:.6g}  mse: {initial['mse']:.6g}")
    print(f"final rel_l2: {final['rel_l2']:.6g}  mse: {final['mse']:.6g}")
    print(f"saved: {args.output}")


def _validate_args(args: argparse.Namespace) -> None:
    if not args.synthetic and args.path is None:
        raise SystemExit("path is required unless --synthetic is set")
    if args.num_frames <= 0:
        raise SystemExit("--num-frames must be positive")
    if args.epochs < 0:
        raise SystemExit("--epochs must be nonnegative")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    if args.k_max <= 0:
        raise SystemExit("--k-max must be positive")
    if args.width <= 0 or args.num_layers <= 0:
        raise SystemExit("--width and --num-layers must be positive")


def _load_frames(args: argparse.Namespace) -> tuple[list[dict[str, np.ndarray]], dict[str, Any]]:
    if args.synthetic:
        group = make_synthetic_cpg_trajectory(
            SyntheticEuler2DConfig(nx=8, ny=6, num_steps=args.start_frame + args.num_frames + 2)
        )
        trajectory = "synthetic"
        path = None
    else:
        try:
            import h5py  # type: ignore[import-not-found]
        except ModuleNotFoundError as exc:
            raise RuntimeError("h5py is required for real HDF5 overfit gates") from exc
        handle = h5py.File(args.path, "r")
        trajectory = args.trajectory or sorted(handle.keys())[0]
        group = handle[trajectory]

    try:
        frames = [
            make_cpg_graph_frame(group, frame, num_steps=1)
            for frame in range(args.start_frame, args.start_frame + args.num_frames)
        ]
    finally:
        if not args.synthetic:
            handle.close()

    return frames, {
        "path": str(path if args.synthetic else args.path),
        "synthetic": bool(args.synthetic),
        "trajectory": trajectory,
        "frames": list(range(args.start_frame, args.start_frame + args.num_frames)),
    }


def _stack_batches(batches: list[PcnoFrameBatch]) -> PcnoFrameBatch:
    first = batches[0]
    for batch in batches[1:]:
        _require_same_tail_shape("x", first.x, batch.x)
        _require_same_tail_shape("y", first.y, batch.y)
        _require_same_tail_shape("node_mask", first.node_mask, batch.node_mask)
        _require_same_tail_shape("nodes", first.nodes, batch.nodes)
        _require_same_tail_shape("node_weights", first.node_weights, batch.node_weights)
        _require_same_tail_shape("directed_edges", first.directed_edges, batch.directed_edges)
        _require_same_tail_shape(
            "edge_gradient_weights",
            first.edge_gradient_weights,
            batch.edge_gradient_weights,
        )

    return PcnoFrameBatch(
        x=np.concatenate([batch.x for batch in batches], axis=0),
        y=np.concatenate([batch.y for batch in batches], axis=0),
        node_mask=np.concatenate([batch.node_mask for batch in batches], axis=0),
        nodes=np.concatenate([batch.nodes for batch in batches], axis=0),
        node_weights=np.concatenate([batch.node_weights for batch in batches], axis=0),
        directed_edges=np.concatenate([batch.directed_edges for batch in batches], axis=0),
        edge_gradient_weights=np.concatenate(
            [batch.edge_gradient_weights for batch in batches],
            axis=0,
        ),
        metadata={
            **first.metadata,
            "num_samples": len(batches),
        },
    )


def _require_same_tail_shape(name: str, left: np.ndarray, right: np.ndarray) -> None:
    if left.shape[1:] != right.shape[1:]:
        raise ValueError(f"{name} shapes must match after batch axis: {left.shape} vs {right.shape}")


def _make_model(batch: PcnoFrameBatch, args: argparse.Namespace) -> PCNO:
    nodes = batch.nodes
    lengths = np.ptp(nodes.reshape(-1, nodes.shape[-1]), axis=0)
    lengths = np.maximum(lengths, 1.0e-6)
    ndims = int(nodes.shape[-1])
    modes = compute_Fourier_modes(ndims, [args.k_max] * ndims, lengths.tolist())
    mode_tensor = torch.as_tensor(modes, dtype=torch.float32)
    return PCNO(
        ndims,
        mode_tensor,
        nmeasures=int(batch.node_weights.shape[-1]),
        layers=[args.width] * args.num_layers,
        fc_dim=args.width,
        in_dim=int(batch.x.shape[-1]),
        out_dim=int(batch.y.shape[-1]),
        inv_L_scale_hyper=None,
        act="gelu",
    )


def _select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return torch.device(name)


def _to_tensors(batch: PcnoFrameBatch, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "x": torch.as_tensor(batch.x, dtype=torch.float32, device=device),
        "y": torch.as_tensor(batch.y, dtype=torch.float32, device=device),
        "node_mask": torch.as_tensor(batch.node_mask, dtype=torch.float32, device=device),
        "nodes": torch.as_tensor(batch.nodes, dtype=torch.float32, device=device),
        "node_weights": torch.as_tensor(batch.node_weights, dtype=torch.float32, device=device),
        "directed_edges": torch.as_tensor(batch.directed_edges, dtype=torch.long, device=device),
        "edge_gradient_weights": torch.as_tensor(
            batch.edge_gradient_weights,
            dtype=torch.float32,
            device=device,
        ),
    }


def _train(
    model: PCNO,
    tensors: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> list[dict[str, float | int]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history: list[dict[str, float | int]] = []
    num_samples = int(tensors["x"].shape[0])
    for epoch in range(args.epochs):
        model.train()
        permutation = torch.randperm(num_samples, device=tensors["x"].device)
        epoch_loss = 0.0
        for start in range(0, num_samples, args.batch_size):
            idx = permutation[start : start + args.batch_size]
            mini = _slice_tensors(tensors, idx)
            optimizer.zero_grad(set_to_none=True)
            pred = _predict(model, mini)
            loss = _masked_mse(pred, mini["y"], mini["node_mask"])
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item()) * int(idx.numel())
        metrics = _evaluate(model, tensors, args.batch_size)
        metrics["epoch"] = epoch
        metrics["train_mse_epoch"] = epoch_loss / float(num_samples)
        history.append(metrics)
        print(
            f"epoch {epoch}: rel_l2={metrics['rel_l2']:.6g} "
            f"mse={metrics['mse']:.6g}",
            flush=True,
        )
    return history


def _evaluate(
    model: PCNO,
    tensors: dict[str, torch.Tensor],
    batch_size: int,
) -> dict[str, float]:
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, int(tensors["x"].shape[0]), batch_size):
            idx = torch.arange(
                start,
                min(start + batch_size, int(tensors["x"].shape[0])),
                device=tensors["x"].device,
            )
            preds.append(_predict(model, _slice_tensors(tensors, idx)).detach())
    pred = torch.cat(preds, dim=0)
    target = tensors["y"]
    mask = tensors["node_mask"]
    err = (pred - target) * mask
    denom = torch.sum((target * mask) ** 2).clamp_min(1.0e-12)
    rel_l2 = torch.sqrt(torch.sum(err**2) / denom)
    mse = _masked_mse(pred, target, mask)
    return {
        "rel_l2": float(rel_l2.item()),
        "mse": float(mse.item()),
        "finite_prediction": bool(torch.isfinite(pred).all().item()),
        "pred_rho_min": float(pred[..., 0].min().item()),
        "pred_pres_min": float(pred[..., 3].min().item()),
    }


def _predict(model: PCNO, tensors: dict[str, torch.Tensor]) -> torch.Tensor:
    return model(
        tensors["x"],
        (
            tensors["node_mask"],
            tensors["nodes"],
            tensors["node_weights"],
            tensors["directed_edges"],
            tensors["edge_gradient_weights"],
        ),
    )


def _slice_tensors(
    tensors: dict[str, torch.Tensor],
    idx: torch.Tensor,
) -> dict[str, torch.Tensor]:
    return {key: value.index_select(0, idx) for key, value in tensors.items()}


def _masked_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    node_mask: torch.Tensor,
) -> torch.Tensor:
    err = (pred - target) * node_mask
    denom = node_mask.sum().clamp_min(1.0) * pred.shape[-1]
    return torch.sum(err**2) / denom


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


if __name__ == "__main__":
    main()
