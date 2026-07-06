from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import torch

from rollout_visualization import (
    DEFAULT_DATA_PATH,
    FEATURE_NAMES,
    _load_euler_forward_data,
    build_euler2d_pcno_from_data,
    rollout_euler2d_sample,
)


# Edit this block directly, then run:
#   python rollout_error_summary.py
INDEX_START = 0
INDEX_END = 99  # inclusive
STEPS = (10, 30, 50, 60)
EPS = 1.0e-6
PLOT_FEATURE_GROUPS = (
    ("rho_p", ("rho", "p"), False),  # (output name, features, use log y-axis)
    ("u_v", ("u", "v"), True),
)

MODEL_PATH = "PCNO_forward_euler_exp_model.pth"
DATA_PATH = DEFAULT_DATA_PATH
START_TIME = 0
DEVICE = None  # None uses cuda:0 if available, otherwise cpu; or set "cuda:0"/"cpu"
EQUAL_WEIGHTS = False
K_MAX = 12  # forward_train.py uses 12
DOMAIN_LENGTHS = (6.0, 2.0)

OUTPUT_DIR = "euler2d_rollout_error_summary"
OUTPUT_PREFIX = None  # None creates a prefix from INDEX_START/INDEX_END/EPS


def plot_features() -> tuple[str, ...]:
    features = []
    for _, group_features, _ in PLOT_FEATURE_GROUPS:
        for feature in group_features:
            if feature not in features:
                features.append(feature)
    return tuple(features)


def _as_path(path: Union[str, Path]) -> Path:
    return Path(path).expanduser().resolve()


def relative_l2(truth: np.ndarray, prediction: np.ndarray) -> float:
    numerator = float(np.linalg.norm(prediction - truth))
    denominator = float(np.linalg.norm(truth))
    if denominator > 1.0e-12:
        return numerator / denominator
    return 0.0 if numerator <= 1.0e-12 else float("nan")


def feature_index(feature: str, nfeatures: int) -> int:
    if feature in FEATURE_NAMES:
        index = FEATURE_NAMES.index(feature)
    else:
        index = int(feature)
    if index < 0 or index >= nfeatures:
        raise IndexError(f"feature={feature!r} maps to index {index}, outside [0, {nfeatures}).")
    return index


def relative_l2_by_rollout_step(
    truth: np.ndarray,
    prediction: np.ndarray,
    feature_idx: int,
) -> np.ndarray:
    errors = []
    for step in range(1, truth.shape[0]):
        errors.append(relative_l2(truth[step, :, feature_idx], prediction[step, :, feature_idx]))
    return np.asarray(errors, dtype=np.float64)


def first_step_below(min_by_step: np.ndarray, eps: float) -> Optional[int]:
    hits = np.flatnonzero(min_by_step < eps)
    return None if hits.size == 0 else int(hits[0])


def format_optional_step(step: Optional[int]) -> str:
    return "" if step is None else str(step)


def sample_threshold_summary(prediction: np.ndarray, *, eps: float) -> dict[str, Union[int, float, str]]:
    rho_min_by_step = np.min(prediction[..., 0], axis=1)
    p_min_by_step = np.min(prediction[..., 3], axis=1)

    rho_min_step = int(np.argmin(rho_min_by_step))
    p_min_step = int(np.argmin(p_min_by_step))
    rho_first_below = first_step_below(rho_min_by_step, eps)
    p_first_below = first_step_below(p_min_by_step, eps)

    return {
        "pred_rho_first_below_eps_step": format_optional_step(rho_first_below),
        "pred_p_first_below_eps_step": format_optional_step(p_first_below),
        "pred_rho_global_min": float(rho_min_by_step[rho_min_step]),
        "pred_rho_global_min_step": rho_min_step,
        "pred_p_global_min": float(p_min_by_step[p_min_step]),
        "pred_p_global_min_step": p_min_step,
    }


def compute_sample_row(
    *,
    index: int,
    truth: np.ndarray,
    prediction: np.ndarray,
    steps: Sequence[int],
    eps: float,
) -> dict[str, Union[int, float, str]]:
    row: dict[str, Union[int, float, str]] = {"index": index}
    nfeatures = truth.shape[-1]

    for step in steps:
        for feature_idx in range(nfeatures):
            feature_name = (
                FEATURE_NAMES[feature_idx]
                if feature_idx < len(FEATURE_NAMES)
                else f"f{feature_idx}"
            )
            row[f"rel_l2_step_{step}_{feature_name}"] = relative_l2(
                truth[step, :, feature_idx],
                prediction[step, :, feature_idx],
            )

    row.update(sample_threshold_summary(prediction, eps=eps))
    return row


def aggregate_error_rows(
    sample_rows: Sequence[dict[str, Union[int, float, str]]],
    *,
    steps: Sequence[int],
    nfeatures: int,
) -> list[dict[str, Union[int, float, str]]]:
    rows = []
    for step in steps:
        for feature_idx in range(nfeatures):
            feature_name = (
                FEATURE_NAMES[feature_idx]
                if feature_idx < len(FEATURE_NAMES)
                else f"f{feature_idx}"
            )
            key = f"rel_l2_step_{step}_{feature_name}"
            values = np.asarray([float(row[key]) for row in sample_rows], dtype=np.float64)
            finite = values[np.isfinite(values)]
            rows.append(
                {
                    "step": step,
                    "feature": feature_name,
                    "mean_relative_l2": float(np.mean(finite)) if finite.size else float("nan"),
                    "min_relative_l2": float(np.min(finite)) if finite.size else float("nan"),
                    "max_relative_l2": float(np.max(finite)) if finite.size else float("nan"),
                    "median_relative_l2": float(np.median(finite)) if finite.size else float("nan"),
                }
            )
    return rows


def aggregate_time_error_rows(
    error_series_by_feature: dict[str, Sequence[np.ndarray]],
) -> list[dict[str, Union[int, float, str]]]:
    rows = []
    for feature, series_list in error_series_by_feature.items():
        if not series_list:
            continue
        values_by_sample = np.asarray(series_list, dtype=np.float64)
        for step_index in range(values_by_sample.shape[1]):
            values = values_by_sample[:, step_index]
            finite = values[np.isfinite(values)]
            rows.append(
                {
                    "step": step_index + 1,
                    "feature": feature,
                    "mean_relative_l2": float(np.mean(finite)) if finite.size else float("nan"),
                    "min_relative_l2": float(np.min(finite)) if finite.size else float("nan"),
                    "max_relative_l2": float(np.max(finite)) if finite.size else float("nan"),
                    "median_relative_l2": float(np.median(finite)) if finite.size else float("nan"),
                }
            )
    return rows


def write_csv(path: Path, rows: Sequence[dict[str, Union[int, float, str]]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {path}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_average_error_table(rows: Sequence[dict[str, Union[int, float, str]]]) -> None:
    print("\nRelative L2 summary over selected indices")
    print("step,feature,mean,min,max,median")
    for row in rows:
        print(
            f"{row['step']},{row['feature']},"
            f"{float(row['mean_relative_l2']):.8e},"
            f"{float(row['min_relative_l2']):.8e},"
            f"{float(row['max_relative_l2']):.8e},"
            f"{float(row['median_relative_l2']):.8e}"
        )


def save_error_summary_plot(
    rows: Sequence[dict[str, Union[int, float, str]]],
    output_path: Path,
    *,
    features: Sequence[str],
    value_key: str,
    ylabel: str,
    title: str,
    log_y: bool = False,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    steps = sorted({int(row["step"]) for row in rows})
    selected_features = list(features)

    values_by_feature = {
        feature: {
            int(row["step"]): float(row[value_key])
            for row in rows
            if row["feature"] == feature
        }
        for feature in selected_features
    }

    fig, ax = plt.subplots(figsize=(7.0, 4.6), constrained_layout=True)
    for feature in selected_features:
        values = np.asarray(
            [values_by_feature[feature].get(step, np.nan) for step in steps],
            dtype=np.float64,
        )
        if log_y:
            values = np.where(values > 0.0, values, np.nan)
        ax.plot(steps, values, marker="o", linewidth=1.8, label=feature)

    ax.set_xlabel("Rollout step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if log_y:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.35)
    ax.legend()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_average_error_plot(
    rows: Sequence[dict[str, Union[int, float, str]]],
    output_path: Path,
    *,
    features: Sequence[str],
    log_y: bool = False,
) -> None:
    save_error_summary_plot(
        rows,
        output_path,
        features=features,
        value_key="mean_relative_l2",
        ylabel="Mean relative L2 error",
        title=f"Mean relative error over rollout steps, indices {INDEX_START}-{INDEX_END}",
        log_y=log_y,
    )


def save_median_error_plot(
    rows: Sequence[dict[str, Union[int, float, str]]],
    output_path: Path,
    *,
    features: Sequence[str],
    log_y: bool = False,
) -> None:
    save_error_summary_plot(
        rows,
        output_path,
        features=features,
        value_key="median_relative_l2",
        ylabel="Median relative L2 error",
        title=f"Median relative error over rollout steps, indices {INDEX_START}-{INDEX_END}",
        log_y=log_y,
    )


def print_threshold_table(rows: Sequence[dict[str, Union[int, float, str]]]) -> None:
    print("\nPredicted rho/p minimum diagnostics")
    print("index,rho_first_below_eps,p_first_below_eps,rho_min,rho_min_step,p_min,p_min_step")
    for row in rows:
        print(
            f"{row['index']},"
            f"{row['pred_rho_first_below_eps_step']},"
            f"{row['pred_p_first_below_eps_step']},"
            f"{float(row['pred_rho_global_min']):.8e},"
            f"{row['pred_rho_global_min_step']},"
            f"{float(row['pred_p_global_min']):.8e},"
            f"{row['pred_p_global_min_step']}"
        )


def validate_config() -> list[int]:
    if INDEX_END < INDEX_START:
        raise ValueError("INDEX_END must be >= INDEX_START.")
    steps = sorted(set(int(step) for step in STEPS))
    if not steps:
        raise ValueError("STEPS must contain at least one step.")
    if steps[0] < 1:
        raise ValueError("Rollout error steps must be >= 1.")
    if EPS <= 0:
        raise ValueError("EPS should be positive.")
    return steps


def output_prefix() -> str:
    if OUTPUT_PREFIX is not None:
        return OUTPUT_PREFIX
    eps_tag = f"{EPS:g}".replace(".", "p").replace("-", "m")
    return f"indices_{INDEX_START}_{INDEX_END}_eps_{eps_tag}"


def main() -> None:
    steps = validate_config()
    n_time = max(steps)

    device = (
        torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if DEVICE is None
        else torch.device(DEVICE)
    )

    data = _load_euler_forward_data(DATA_PATH, equal_weights=EQUAL_WEIGHTS)
    model = build_euler2d_pcno_from_data(
        data,
        model_path=MODEL_PATH,
        device=device,
        k_max=K_MAX,
        domain_lengths=DOMAIN_LENGTHS,
    )

    sample_rows = []
    nfeatures = int(data["features"].shape[-1])
    plot_feature_indices = {
        feature: feature_index(feature, nfeatures)
        for feature in plot_features()
    }
    full_step_errors = {feature: [] for feature in plot_features()}
    print(
        f"Running rollout summary for indices {INDEX_START}..{INDEX_END} "
        f"on {device}, steps={steps}, eps={EPS}."
    )

    for index in range(INDEX_START, INDEX_END + 1):
        truth, prediction = rollout_euler2d_sample(
            model,
            data,
            index=index,
            n_time=n_time,
            start_time=START_TIME,
            device=device,
        )
        row = compute_sample_row(
            index=index,
            truth=truth,
            prediction=prediction,
            steps=steps,
            eps=EPS,
        )
        sample_rows.append(row)
        for feature, feature_idx in plot_feature_indices.items():
            full_step_errors[feature].append(
                relative_l2_by_rollout_step(truth, prediction, feature_idx)
            )
        print(
            f"index={index} done; "
            f"rho_first_below_eps={row['pred_rho_first_below_eps_step'] or 'never'}, "
            f"p_first_below_eps={row['pred_p_first_below_eps_step'] or 'never'}"
        )

    average_rows = aggregate_error_rows(sample_rows, steps=steps, nfeatures=nfeatures)
    full_step_rows = aggregate_time_error_rows(full_step_errors)

    output_dir = _as_path(OUTPUT_DIR)
    prefix = output_prefix()
    per_sample_path = output_dir / f"{prefix}_per_sample.csv"
    average_path = output_dir / f"{prefix}_summary.csv"
    full_step_path = output_dir / f"{prefix}_plot_features_full_steps.csv"
    write_csv(per_sample_path, sample_rows)
    write_csv(average_path, average_rows)
    write_csv(full_step_path, full_step_rows)

    plot_paths = []
    for group_name, group_features, log_y in PLOT_FEATURE_GROUPS:
        average_plot_path = output_dir / f"{prefix}_{group_name}_average.png"
        median_plot_path = output_dir / f"{prefix}_{group_name}_median.png"
        save_average_error_plot(
            full_step_rows,
            average_plot_path,
            features=group_features,
            log_y=log_y,
        )
        save_median_error_plot(
            full_step_rows,
            median_plot_path,
            features=group_features,
            log_y=log_y,
        )
        plot_paths.extend([average_plot_path, median_plot_path])

    print_average_error_table(average_rows)
    print_threshold_table(sample_rows)
    print(f"\nPer-sample CSV saved to: {per_sample_path}")
    print(f"Summary CSV saved to: {average_path}")
    print(f"Full-step plot-feature CSV saved to: {full_step_path}")
    for plot_path in plot_paths:
        print(f"Error plot saved to: {plot_path}")


if __name__ == "__main__":
    main()
