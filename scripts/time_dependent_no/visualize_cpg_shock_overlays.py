"""Create shock-front overlay snapshots from CPGNet rollout HDF5 files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utility.time_dependent_no.euler2d import EulerNodeType, PRIMITIVE_NAMES  # noqa: E402
from utility.time_dependent_no.euler2d_metrics import (  # noqa: E402
    front_centroid,
    front_centroid_distance,
    front_distance_metrics,
    front_overlap_metrics,
    front_region_masks,
    shock_front_masks,
)

FEATURE_INDEX = {name: index for index, name in enumerate(PRIMITIVE_NAMES)}
FEATURE_INDEX.update(
    {"pressure": FEATURE_INDEX["pres"], "density": FEATURE_INDEX["rho"]}
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        action="append",
        required=True,
        help="Case spec NAME=PATH_TO_RESULT_H5. Repeat for multiple cases.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", nargs="+", type=int, default=[1, 20, 40, 60, 79])
    parser.add_argument("--quantile", type=float, default=0.90)
    parser.add_argument("--feature", default="pres", choices=sorted(FEATURE_INDEX))
    parser.add_argument("--node-filter", choices=("normal", "all"), default="normal")
    parser.add_argument("--point-size", type=float, default=None)
    parser.add_argument("--error-quantile", type=float, default=0.995)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def parse_case(raw: str) -> tuple[str, Path]:
    if "=" in raw:
        name, value = raw.split("=", 1)
        return name.strip(), Path(value)
    path = Path(raw)
    return path.stem, path


def read_rollout(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as handle:
        required = ("predicteds", "targets", "pos", "edges", "node_type")
        missing = [key for key in required if key not in handle]
        if missing:
            raise KeyError(f"{path} is missing datasets: {missing}")
        pred = np.asarray(handle["predicteds"], dtype=np.float64)
        target = np.asarray(handle["targets"], dtype=np.float64)
        pos = np.asarray(handle["pos"], dtype=np.float64)
        edges = np.asarray(handle["edges"], dtype=np.int64)
        node_type = np.asarray(handle["node_type"], dtype=np.int64).reshape(-1)
    if pred.shape != target.shape:
        raise ValueError(
            f"predicteds and targets differ: {pred.shape} vs {target.shape}"
        )
    if pred.ndim != 3 or pred.shape[-1] != len(PRIMITIVE_NAMES):
        raise ValueError(
            f"rollout arrays must have shape (steps, nodes, 4), got {pred.shape}"
        )
    if pos.shape != (pred.shape[1], 2):
        raise ValueError(
            f"pos shape {pos.shape} does not match node count {pred.shape[1]}"
        )
    if node_type.shape != (pred.shape[1],):
        raise ValueError("node_type must have one value per node")
    return {
        "prediction": pred,
        "target": target,
        "pos": pos,
        "edges": edges,
        "node_type": node_type,
    }


def node_mask(node_type: np.ndarray, node_filter: str) -> np.ndarray:
    if node_filter == "all":
        return np.ones(node_type.shape[0], dtype=bool)
    return node_type == int(EulerNodeType.NORMAL)


def auto_point_size(num_nodes: int) -> float:
    return float(np.clip(48_000.0 / max(num_nodes, 1), 0.25, 3.0))


def finite_limits(
    values: np.ndarray, *, quantile: float | None = None
) -> tuple[float, float]:
    flat = values[np.isfinite(values)]
    if flat.size == 0:
        return -1.0, 1.0
    if quantile is None:
        lo = float(np.min(flat))
        hi = float(np.max(flat))
    else:
        bound = float(np.quantile(np.abs(flat), quantile))
        lo, hi = -bound, bound
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return lo - 0.5, hi + 0.5
    return lo, hi


def step_list(requested: Sequence[int], max_step: int) -> list[int]:
    out = []
    for step in requested:
        if step < 1 or step > max_step:
            continue
        out.append(int(step))
    if not out:
        raise ValueError(f"no requested steps are within [1, {max_step}]")
    return sorted(set(out))


def scatter_field(
    ax: plt.Axes,
    pos: np.ndarray,
    values: np.ndarray,
    *,
    mask: np.ndarray,
    point_size: float,
    cmap: str,
    limits: tuple[float, float],
    title: str,
) -> Any:
    artist = ax.scatter(
        pos[mask, 0],
        pos[mask, 1],
        c=values[mask],
        s=point_size,
        cmap=cmap,
        vmin=limits[0],
        vmax=limits[1],
        linewidths=0,
    )
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    return artist


def overlay_fronts(
    ax: plt.Axes,
    pos: np.ndarray,
    *,
    target_front: np.ndarray,
    prediction_front: np.ndarray,
    point_size: float,
) -> None:
    overlap = target_front & prediction_front
    target_only = target_front & ~prediction_front
    pred_only = prediction_front & ~target_front
    if np.any(target_only):
        ax.scatter(
            pos[target_only, 0],
            pos[target_only, 1],
            s=point_size * 4.0,
            c="#1f77b4",
            marker="o",
            linewidths=0,
            label="target front only",
        )
    if np.any(pred_only):
        ax.scatter(
            pos[pred_only, 0],
            pos[pred_only, 1],
            s=point_size * 4.0,
            c="#d62728",
            marker="x",
            linewidths=0.7,
            label="predicted front only",
        )
    if np.any(overlap):
        ax.scatter(
            pos[overlap, 0],
            pos[overlap, 1],
            s=point_size * 4.5,
            c="#2ca02c",
            marker=".",
            linewidths=0,
            label="front overlap",
        )


def add_centroids(
    ax: plt.Axes,
    target_centroid: np.ndarray,
    prediction_centroid: np.ndarray,
) -> None:
    if np.all(np.isfinite(target_centroid)):
        ax.scatter(
            target_centroid[0],
            target_centroid[1],
            s=35,
            c="#003f8c",
            marker="P",
            edgecolors="white",
            linewidths=0.5,
        )
    if np.all(np.isfinite(prediction_centroid)):
        ax.scatter(
            prediction_centroid[0],
            prediction_centroid[1],
            s=35,
            c="#9c1111",
            marker="X",
            edgecolors="white",
            linewidths=0.5,
        )


def save_overlay(
    *,
    case_name: str,
    data: dict[str, np.ndarray],
    output_dir: Path,
    step: int,
    feature_index: int,
    feature_name: str,
    quantile: float,
    mask: np.ndarray,
    point_size: float,
    field_limits: tuple[float, float],
    error_limits: tuple[float, float],
    dpi: int,
) -> dict[str, Any]:
    index = step - 1
    pred = data["prediction"]
    target = data["target"]
    pos = data["pos"]
    edges = data["edges"]
    fronts = shock_front_masks(
        pred[index : index + 1],
        target[index : index + 1],
        edges,
        scalar_index=FEATURE_INDEX["pres"],
        quantile=quantile,
        node_mask=mask,
    )
    pred_front = fronts["prediction_mask"][0]
    target_front = fronts["target_mask"][0]
    overlap = front_overlap_metrics(pred_front, target_front)
    distances = front_distance_metrics(pred_front, target_front, pos)
    centroid_distance = front_centroid_distance(pred_front, target_front, pos)
    pred_centroid = front_centroid(pred_front, pos)
    target_centroid = front_centroid(target_front, pos)
    regions = front_region_masks(pred_front, target_front, node_mask=mask)

    target_values = target[index, :, feature_index]
    pred_values = pred[index, :, feature_index]
    error_values = pred_values - target_values

    fig, axes = plt.subplots(2, 2, figsize=(10, 6.8), constrained_layout=True)
    artists = [
        scatter_field(
            axes[0, 0],
            pos,
            target_values,
            mask=mask,
            point_size=point_size,
            cmap="viridis",
            limits=field_limits,
            title=f"target {feature_name}",
        ),
        scatter_field(
            axes[0, 1],
            pos,
            pred_values,
            mask=mask,
            point_size=point_size,
            cmap="viridis",
            limits=field_limits,
            title=f"predicted {feature_name}",
        ),
        scatter_field(
            axes[1, 0],
            pos,
            error_values,
            mask=mask,
            point_size=point_size,
            cmap="coolwarm",
            limits=error_limits,
            title=f"signed error {feature_name}",
        ),
    ]
    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        overlay_fronts(
            ax,
            pos,
            target_front=target_front,
            prediction_front=pred_front,
            point_size=point_size,
        )
        add_centroids(ax, target_centroid, pred_centroid)

    ax = axes[1, 1]
    ax.scatter(
        pos[mask, 0],
        pos[mask, 1],
        s=point_size,
        c="#d9d9d9",
        linewidths=0,
    )
    overlay_fronts(
        ax,
        pos,
        target_front=target_front,
        prediction_front=pred_front,
        point_size=point_size,
    )
    add_centroids(ax, target_centroid, pred_centroid)
    ax.set_title("shock-front overlay", fontsize=9)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])

    for ax, artist in zip((axes[0, 0], axes[0, 1], axes[1, 0]), artists, strict=True):
        fig.colorbar(artist, ax=ax, fraction=0.046, pad=0.02)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8)
    fig.suptitle(
        f"{case_name} step {step} pressure-front q={quantile:.2f} "
        f"IoU={float(overlap['iou']):.3f}, Chamfer={float(distances['symmetric_chamfer_mean']):.4f}",
        fontsize=11,
    )

    path = output_dir / f"{case_name}_step_{step:04d}_shock_overlay.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)

    region_counts = {
        name: int(np.count_nonzero(value)) for name, value in regions.items()
    }
    return {
        "case": case_name,
        "step": int(step),
        "path": str(path),
        "front_iou": float(overlap["iou"]),
        "front_f1": float(overlap["f1"]),
        "front_precision": float(overlap["precision"]),
        "front_recall": float(overlap["recall"]),
        "symmetric_chamfer_mean": float(distances["symmetric_chamfer_mean"]),
        "centroid_distance": float(centroid_distance),
        "target_front_count": int(np.count_nonzero(target_front)),
        "prediction_front_count": int(np.count_nonzero(pred_front)),
        "region_counts": region_counts,
    }


def write_gallery(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    lines = ["# CPGNet Shock Overlay Gallery", ""]
    current = None
    for row in rows:
        if row["case"] != current:
            current = row["case"]
            lines.extend([f"## {current}", ""])
        image = Path(row["path"]).name
        lines.append(
            f"- step {row['step']}: IoU={row['front_iou']:.3f}, "
            f"F1={row['front_f1']:.3f}, Chamfer={row['symmetric_chamfer_mean']:.4f}, "
            f"centroid={row['centroid_distance']:.4f}"
        )
        lines.append(f"  ![]({image})")
        lines.append("")
    (output_dir / "shock_overlay_gallery.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    feature_index = FEATURE_INDEX[args.feature]
    feature_name = PRIMITIVE_NAMES[feature_index]
    rows: list[dict[str, Any]] = []

    for case_name, path in [parse_case(raw) for raw in args.case]:
        data = read_rollout(path)
        mask = node_mask(data["node_type"], args.node_filter)
        if not np.any(mask):
            raise ValueError(f"case {case_name} selected no nodes")
        steps = step_list(args.steps, data["target"].shape[0])
        values = np.concatenate(
            [
                data["target"][:, mask, feature_index].reshape(-1),
                data["prediction"][:, mask, feature_index].reshape(-1),
            ]
        )
        errors = (
            data["prediction"][:, mask, feature_index]
            - data["target"][:, mask, feature_index]
        )
        field_limits = finite_limits(values)
        error_limits = finite_limits(errors, quantile=args.error_quantile)
        point_size = args.point_size or auto_point_size(int(np.count_nonzero(mask)))
        for step in steps:
            rows.append(
                save_overlay(
                    case_name=case_name,
                    data=data,
                    output_dir=args.output_dir,
                    step=step,
                    feature_index=feature_index,
                    feature_name=feature_name,
                    quantile=args.quantile,
                    mask=mask,
                    point_size=point_size,
                    field_limits=field_limits,
                    error_limits=error_limits,
                    dpi=args.dpi,
                )
            )

    payload = {
        "quantile": float(args.quantile),
        "feature": feature_name,
        "node_filter": args.node_filter,
        "rows": rows,
    }
    summary_path = args.output_dir / "shock_overlay_summary.json"
    summary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_gallery(args.output_dir, rows)
    print(
        json.dumps({"summary": str(summary_path), "num_figures": len(rows)}, indent=2)
    )


if __name__ == "__main__":
    main()
