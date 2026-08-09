import argparse
import csv
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from mpcno_eval_errors import (
    SCRIPT_DIR,
    build_model,
    load_preprocessed_data,
    prepare_tensors,
    resolve_user_path,
    str_to_bool,
)
from utility.normalizer import UnitGaussianNormalizer


def default_model_paths():
    return [SCRIPT_DIR / "model" / "mpcno.pth"] + [
        SCRIPT_DIR / "model" / f"mpcno_{index}.pth" for index in range(1, 8)
    ]


def rankdata(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def pearson_corr(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)
    denominator = np.linalg.norm(x_centered) * np.linalg.norm(y_centered)
    if denominator == 0:
        return float("nan")
    return float(np.dot(x_centered, y_centered) / denominator)


def spearman_corr(x, y):
    return pearson_corr(rankdata(x), rankdata(y))


def auroc_score(scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    n_pos = int(np.sum(labels))
    n_neg = int(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores) + 1.0
    pos_rank_sum = float(np.sum(ranks[labels]))
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def load_models(args, x_train, y_train, device):
    models = []
    for model_path in args.model_paths:
        model_args = SimpleNamespace(**vars(args))
        model_args.model_path = model_path
        print(f"Loading model: {model_path}", flush=True)
        models.append(build_model(model_args, x_train, y_train, device))
    return models


def predict_sample(models, x_test, y_test, aux_test, y_normalizer, device, local_index):
    x = x_test[[local_index]].to(device)
    y_encoded = y_test[[local_index]].to(device)
    aux = tuple(item[[local_index]].to(device) for item in aux_test)
    mask = aux[0].to(torch.bool)

    predictions = []
    with torch.no_grad():
        y = y_normalizer.decode(y_encoded) * mask.to(y_encoded.dtype)
        for model in models:
            pred = model(x, aux)
            pred = y_normalizer.decode(pred)
            pred = pred * mask.to(pred.dtype)
            predictions.append(pred)

    y_np = y.detach().cpu().numpy()[0, :, 0]
    mask_np = mask.detach().cpu().numpy()[0, :, 0].astype(bool)
    predictions_np = np.stack(
        [pred.detach().cpu().numpy()[0, :, 0] for pred in predictions],
        axis=0,
    )
    return predictions_np, y_np, mask_np


def relative_l2(prediction, reference):
    denominator = np.linalg.norm(reference)
    if denominator < 1.0e-12:
        denominator = 1.0e-12
    return float(np.linalg.norm(prediction - reference) / denominator)


def absolute_l2(prediction, reference):
    return float(np.linalg.norm(prediction - reference))


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_uncertainty_vs_error(path, uncertainty, errors):
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.scatter(uncertainty, errors, s=34, alpha=0.75, edgecolors="none")
    ax.set_title("M-PCNO ensemble uncertainty vs error", fontsize=16)
    ax.set_xlabel("Ensemble uncertainty", fontsize=14)
    ax.set_ylabel(r"Rel. $L^2$ error", fontsize=14)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_uncertainty_bins(path, uncertainty, errors, num_bins):
    order = np.argsort(uncertainty)
    bins = np.array_split(order, num_bins)
    bin_centers = [float(np.mean(uncertainty[indices])) for indices in bins if len(indices)]
    bin_errors = [float(np.mean(errors[indices])) for indices in bins if len(indices)]

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.plot(bin_centers, bin_errors, marker="o", linewidth=1.8)
    ax.set_title("Error binned by ensemble uncertainty", fontsize=16)
    ax.set_xlabel("Mean uncertainty in bin", fontsize=14)
    ax.set_ylabel(r"Mean rel. $L^2$ error", fontsize=14)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_risk_coverage(path, uncertainty, errors):
    order = np.argsort(uncertainty)
    sorted_errors = errors[order]
    coverage = np.arange(1, len(errors) + 1, dtype=np.float64) / len(errors)
    risk = np.cumsum(sorted_errors) / np.arange(1, len(errors) + 1)

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.plot(coverage, risk, linewidth=1.8)
    ax.set_title("Risk-coverage by ensemble uncertainty", fontsize=16)
    ax.set_xlabel("Coverage", fontsize=14)
    ax.set_ylabel(r"Mean rel. $L^2$ error", fontsize=14)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate M-PCNO deep ensemble uncertainty.")
    parser.add_argument("--model_paths", type=Path, nargs="+", default=default_model_paths())
    parser.add_argument("--data_path", type=Path, default=SCRIPT_DIR / "../../data/car_shapenet")
    parser.add_argument("--output_dir", type=Path, default=SCRIPT_DIR / "eval_mpcno_ensemble")
    parser.add_argument("--n_train", type=int, default=500)
    parser.add_argument("--n_test", type=int, default=111)
    parser.add_argument("--ndata", type=int, default=611)
    parser.add_argument("--k_max", type=int, default=16)
    parser.add_argument("--Ls", type=str, default="4.0,4.0,12.0")
    parser.add_argument("--layer_sizes", type=str, default="64,64,64,64,64,64,64")
    parser.add_argument("--act", type=str, default="gelu")
    parser.add_argument("--geo_act", type=str, default="softsign")
    parser.add_argument("--grad", type=str_to_bool, default=True)
    parser.add_argument("--geo", type=str_to_bool, default=True)
    parser.add_argument("--geointegral", type=str_to_bool, default=True)
    parser.add_argument("--equal_weights", action="store_true")
    parser.add_argument("--preprocess_data", action="store_true")
    parser.add_argument("--num_bins", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    args.model_paths = [resolve_user_path(path) for path in args.model_paths]
    args.data_path = resolve_user_path(args.data_path)
    args.output_dir = resolve_user_path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data = load_preprocessed_data(args.data_path, args.ndata, args.preprocess_data)
    x_train, x_test, y_train, y_test, aux_test, nnodes_test = prepare_tensors(
        data,
        args.data_path,
        args.n_train,
        args.n_test,
        args.equal_weights,
    )
    y_normalizer = UnitGaussianNormalizer(y_train, non_normalized_dim=0, normalization_dim=[])
    y_test = y_normalizer.encode(y_test)
    y_normalizer.to(device)

    models = load_models(args, x_train, y_train, device)
    model_names = [path.stem for path in args.model_paths]

    sample_rows = []
    model_rows = []
    pointwise_variances = []
    masks = []

    for local_index in range(args.n_test):
        data_index = args.n_train + local_index
        predictions, reference, mask = predict_sample(
            models,
            x_test,
            y_test,
            aux_test,
            y_normalizer,
            device,
            local_index,
        )
        valid_predictions = predictions[:, mask]
        valid_reference = reference[mask]
        ensemble_mean = np.mean(valid_predictions, axis=0)
        point_variance = np.var(valid_predictions, axis=0)
        padded_point_variance = np.zeros(predictions.shape[1], dtype=np.float32)
        padded_point_variance[mask] = point_variance.astype(np.float32)
        pointwise_variances.append(padded_point_variance)
        masks.append(mask.astype(np.uint8))

        per_model_rel_l2 = []
        per_model_abs_l2 = []
        for model_index, model_name in enumerate(model_names):
            pred = valid_predictions[model_index]
            rel_error = relative_l2(pred, valid_reference)
            abs_error = absolute_l2(pred, valid_reference)
            per_model_rel_l2.append(rel_error)
            per_model_abs_l2.append(abs_error)
            model_rows.append(
                {
                    "test_local_index": local_index,
                    "data_index": data_index,
                    "model_index": model_index,
                    "model_name": model_name,
                    "relative_l2_error": rel_error,
                    "absolute_l2_error": abs_error,
                }
            )

        ensemble_rel_l2 = relative_l2(ensemble_mean, valid_reference)
        ensemble_abs_l2 = absolute_l2(ensemble_mean, valid_reference)
        mean_prediction_variance = float(np.mean(point_variance))
        max_prediction_variance = float(np.max(point_variance))
        rms_prediction_std = float(np.sqrt(mean_prediction_variance))
        reference_rms = float(np.sqrt(np.mean(valid_reference**2)))
        mean_prediction_rms = float(np.sqrt(np.mean(ensemble_mean**2)))
        relative_uncertainty_ref = rms_prediction_std / max(reference_rms, 1.0e-12)
        relative_uncertainty_pred = rms_prediction_std / max(mean_prediction_rms, 1.0e-12)

        sample_rows.append(
            {
                "test_local_index": local_index,
                "data_index": data_index,
                "ensemble_relative_l2_error": ensemble_rel_l2,
                "ensemble_absolute_l2_error": ensemble_abs_l2,
                "mean_model_relative_l2_error": float(np.mean(per_model_rel_l2)),
                "std_model_relative_l2_error": float(np.std(per_model_rel_l2)),
                "min_model_relative_l2_error": float(np.min(per_model_rel_l2)),
                "max_model_relative_l2_error": float(np.max(per_model_rel_l2)),
                "mean_model_absolute_l2_error": float(np.mean(per_model_abs_l2)),
                "std_model_absolute_l2_error": float(np.std(per_model_abs_l2)),
                "mean_prediction_variance": mean_prediction_variance,
                "max_prediction_variance": max_prediction_variance,
                "rms_prediction_std": rms_prediction_std,
                "relative_uncertainty_ref": relative_uncertainty_ref,
                "relative_uncertainty_pred": relative_uncertainty_pred,
            }
        )
        print(
            f"test_local_index={local_index:03d} data_index={data_index:05d} "
            f"ensemble_rel_l2={ensemble_rel_l2:.8e} "
            f"uncertainty_ref={relative_uncertainty_ref:.8e}",
            flush=True,
        )

    per_model_csv = args.output_dir / "per_model_test_errors.csv"
    sample_csv = args.output_dir / "ensemble_uq_by_sample.csv"
    write_csv(
        per_model_csv,
        model_rows,
        [
            "test_local_index",
            "data_index",
            "model_index",
            "model_name",
            "relative_l2_error",
            "absolute_l2_error",
        ],
    )
    write_csv(
        sample_csv,
        sample_rows,
        [
            "test_local_index",
            "data_index",
            "ensemble_relative_l2_error",
            "ensemble_absolute_l2_error",
            "mean_model_relative_l2_error",
            "std_model_relative_l2_error",
            "min_model_relative_l2_error",
            "max_model_relative_l2_error",
            "mean_model_absolute_l2_error",
            "std_model_absolute_l2_error",
            "mean_prediction_variance",
            "max_prediction_variance",
            "rms_prediction_std",
            "relative_uncertainty_ref",
            "relative_uncertainty_pred",
        ],
    )

    pointwise_npz = args.output_dir / "ensemble_pointwise_variance.npz"
    np.savez_compressed(
        pointwise_npz,
        pointwise_variance=np.asarray(pointwise_variances, dtype=np.float32),
        node_mask=np.asarray(masks, dtype=np.uint8),
    )

    ensemble_errors = np.asarray(
        [row["ensemble_relative_l2_error"] for row in sample_rows],
        dtype=np.float64,
    )
    uncertainty_ref = np.asarray(
        [row["relative_uncertainty_ref"] for row in sample_rows],
        dtype=np.float64,
    )
    uncertainty_variance = np.asarray(
        [row["mean_prediction_variance"] for row in sample_rows],
        dtype=np.float64,
    )

    high_error_threshold = float(np.quantile(ensemble_errors, 0.8))
    high_error_labels = ensemble_errors >= high_error_threshold
    summary = {
        "model_paths": [str(path) for path in args.model_paths],
        "n_models": len(args.model_paths),
        "n_test": args.n_test,
        "ensemble_error_mean": float(np.mean(ensemble_errors)),
        "ensemble_error_median": float(np.median(ensemble_errors)),
        "ensemble_error_max": float(np.max(ensemble_errors)),
        "relative_uncertainty_ref_mean": float(np.mean(uncertainty_ref)),
        "mean_prediction_variance_mean": float(np.mean(uncertainty_variance)),
        "pearson_uncertainty_error": pearson_corr(uncertainty_ref, ensemble_errors),
        "spearman_uncertainty_error": spearman_corr(uncertainty_ref, ensemble_errors),
        "auroc_top_20_percent_error": auroc_score(uncertainty_ref, high_error_labels),
        "high_error_threshold_top_20_percent": high_error_threshold,
        "per_model_errors_csv": str(per_model_csv),
        "sample_uq_csv": str(sample_csv),
        "pointwise_variance_npz": str(pointwise_npz),
    }
    summary_path = args.output_dir / "ensemble_uq_summary.json"
    with summary_path.open("w") as file:
        json.dump(summary, file, indent=2)

    plot_uncertainty_vs_error(
        args.output_dir / "uncertainty_vs_error.png",
        uncertainty_ref,
        ensemble_errors,
    )
    plot_uncertainty_bins(
        args.output_dir / "uncertainty_bins.png",
        uncertainty_ref,
        ensemble_errors,
        args.num_bins,
    )
    plot_risk_coverage(
        args.output_dir / "risk_coverage.png",
        uncertainty_ref,
        ensemble_errors,
    )

    print(f"Wrote per-model errors to {per_model_csv}", flush=True)
    print(f"Wrote sample-level UQ to {sample_csv}", flush=True)
    print(f"Wrote pointwise variances to {pointwise_npz}", flush=True)
    print(f"Wrote summary to {summary_path}", flush=True)
    print(
        "Spearman uncertainty-error correlation: "
        f"{summary['spearman_uncertainty_error']:.8e}",
        flush=True,
    )
    print(
        "AUROC for detecting top-20% high-error samples: "
        f"{summary['auroc_top_20_percent_error']:.8e}",
        flush=True,
    )


if __name__ == "__main__":
    main()
