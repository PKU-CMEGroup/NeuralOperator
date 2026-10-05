"""Working figures for corrective rollouts, their local defects and a clean probe.

Run with --input EVALUATION --output FIGURES [--comparison-root TRAINING_RUNS].
Incomplete training/evaluation remains explicitly pending in the displays.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


CONDITIONS = ("native_one", "native_mixed", "gaussian")
TITLES = ("Native one-step errors", "Native errors: 1, 2, 4, 8 steps", "Gaussian noise")
LABELS = {"pretrained": "Pretrained", "clean_continuation": "Clean continuation",
          "recovery": "Recovery", "dynamics": "Dynamics relabeling"}
COLORS = {"pretrained": "#333333", "clean_continuation": "#777777",
          "recovery": "#0072B2", "dynamics": "#D55E00", "truth": "#009E73"}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_metrics(root):
    metadata = json.loads((root / "result.json").read_text())
    truth_file = root / "ground_truth.npz"
    if digest(truth_file) != metadata["ground_truth_sha256"]:
        raise ValueError("ground-truth metric hash mismatch")
    with np.load(truth_file) as saved:
        truth = {k: saved[k] for k in saved.files}
    data = {}
    for name in metadata["completed_methods"]:
        receipt = json.loads((root / name / "result.json").read_text())
        filename = root / name / "metrics.npz"
        if receipt["status"] != "completed" or digest(filename) != receipt["artifacts"]["metrics.npz"]:
            raise ValueError("incomplete or unauthenticated method metrics")
        with np.load(filename) as saved:
            data[name] = {k: saved[k] for k in saved.files}
        h = metadata["horizon"]
        for key, values in data[name].items():
            expected = (2, h+1, 4) if key == "physical_values" else (2, h+1) if key == "rollout_mse" else (2, h)
            if values.shape != expected or not np.isfinite(values).all():
                raise ValueError(f"invalid metric shape/values: {name}/{key}")
    return metadata, truth, data


def save(fig, output, stem):
    for suffix in ("pdf", "png"):
        fig.savefig(output / (stem + "." + suffix), dpi=160, facecolor="white")
    plt.close(fig)


def axes_for(title, explanation, metadata):
    fig, axes = plt.subplots(2, 3, figsize=(13.8, 7.3), sharex=True, sharey=True)
    fig.subplots_adjust(left=.085, right=.985, top=.77, bottom=.16, wspace=.12, hspace=.3)
    fig.suptitle(title, y=.98, fontsize=16)
    fig.text(.5, .928, explanation, ha="center", fontsize=10)
    for c, label in enumerate(TITLES):
        axes[0, c].set_title(label, fontsize=11)
    for r, population in enumerate(("Training initial condition", "Validation initial condition")):
        axes[r, 0].text(-.18, .5, population, transform=axes[r, 0].transAxes,
                        rotation=90, va="center", ha="center", fontsize=10)
        for axis in axes[r]:
            axis.spines[["top", "right"]].set_visible(False)
            axis.grid(axis="y", alpha=.22)
            axis.set_xlim(0, metadata["horizon"] * metadata["macro_dt"])
    fig.text(.5, .025, "One training and one validation initial condition; one trained model per condition. "
             "s is the fixed training normalization scale.\nPending notices indicate models without completed evaluation.",
             ha="center", fontsize=9)
    return fig, axes


def pending(axis, names, data):
    missing = [LABELS[name.rsplit("_", 1)[-1]] for name in names if name not in data]
    if missing:
        axis.text(.03, .94, "Pending: " + ", ".join(missing), transform=axis.transAxes,
                  va="top", fontsize=8, color="#777777")


def plot(root, output, comparison_root=None):
    root, output = Path(root), Path(output)
    metadata, truth, data = read_metrics(root)
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42, "ps.fonttype": 42})
    t = np.arange(metadata["horizon"]) * metadata["macro_dt"]
    rollout_t = np.arange(metadata["horizon"] + 1) * metadata["macro_dt"]
    handles = [Line2D([0], [0], color=COLORS[key], linestyle=":" if key == "clean_continuation" else "-",
                      label=LABELS[key], linewidth=2) for key in LABELS]
    for metric, stem, title, explanation in (
            ("rollout_mse", "rollout_errors", "Which training choices improve rollout?",
             "Each model predicts autonomously from the same initial state; compare its state with the ground-truth state."),
            ("probe_defect_mse", "probe_errors", "One fixed clean probe evaluated on every rollout",
             "Each color identifies the model generating the probe's input states. Compare the probe and solver next states at each input.")):
        fig, axes = axes_for(title, explanation, metadata)
        legend = handles.copy()
        if metric == "probe_defect_mse":
            legend = [Line2D([0], [0], color=COLORS[key],
                             linestyle=":" if key == "clean_continuation" else "-",
                             label=label, linewidth=2)
                      for key, label in zip(LABELS, ("Pretrained rollout", "Clean-continuation rollout",
                                                    "Recovery rollout", "Dynamics rollout"))]
            legend.append(Line2D([0], [0], color=COLORS["truth"], label="Ground-truth inputs", linestyle="--", linewidth=2))
        fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(.5, .88),
                   ncol=3 if metric == "probe_defect_mse" else len(legend), frameon=False)
        for c, condition in enumerate(CONDITIONS):
            methods = ["pretrained", "clean_continuation", condition + "_recovery", condition + "_dynamics"]
            for r in range(2):
                axis = axes[r, c]
                for name in methods:
                    if name not in data:
                        continue
                    key = name if name in LABELS else name.rsplit("_", 1)[-1]
                    axis.plot(rollout_t if metric == "rollout_mse" else t,
                              np.sqrt(data[name][metric][r]), color=COLORS[key],
                              linestyle=":" if key == "clean_continuation" else "-", linewidth=1.8)
                if metric == "probe_defect_mse":
                    axis.plot(t, np.sqrt(truth["probe_clean_mse"][r]), "--", color=COLORS["truth"], linewidth=1.8)
                    axis.set_yscale("log")
                pending(axis, methods[2:], data)
                if r == 1:
                    axis.set_xlabel("State time" if metric == "rollout_mse" else "Input-state time")
                if c == 0:
                    axis.set_ylabel("Rollout error / s" if metric == "rollout_mse" else "Probe one-step defect / s")
        if metric == "rollout_mse":
            axes[0, 0].set_ylim(bottom=0)
        save(fig, output, stem)

    fig, axes = axes_for("How do clean-input and own-rollout one-step errors differ?",
                        "For each trained model: compare its prediction with the archived clean next state or the solver's next state from its rollout input.", metadata)
    fig.legend(handles=[handles[0], handles[2], handles[3], Line2D([0], [0], color="black", label="Own-rollout inputs"),
                        Line2D([0], [0], color="black", linestyle="--", label="Ground-truth inputs")],
               loc="upper center", bbox_to_anchor=(.5, .88), ncol=5, frameon=False)
    for c, condition in enumerate(CONDITIONS):
        for r in range(2):
            axis = axes[r, c]
            for name in ("pretrained", condition + "_recovery", condition + "_dynamics"):
                if name not in data:
                    continue
                key = name if name == "pretrained" else name.rsplit("_", 1)[-1]
                for metric, style in (("own_defect_mse", "-"), ("clean_mse", "--")):
                    axis.plot(t, np.sqrt(data[name][metric][r]), color=COLORS[key], linestyle=style,
                              linewidth=1.8, marker=".", markersize=3)
            axis.set_yscale("log")
            pending(axis, [condition + "_" + key for key in ("recovery", "dynamics")], data)
            if r == 1:
                axis.set_xlabel("Input-state time")
            if c == 0:
                axis.set_ylabel("Model one-step error / s")
    save(fig, output, "one_step_errors")

    fig, axes = axes_for("Do local defects cancel or reinforce propagated trajectory errors?",
                        "Cosine between the model's local dynamics defect and the solver-propagated trajectory error: negative values indicate cancellation.", metadata)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .88), ncol=4, frameon=False)
    for c, condition in enumerate(CONDITIONS):
        for r in range(2):
            axis = axes[r, c]
            for name in ("pretrained", "clean_continuation", condition + "_recovery", condition + "_dynamics"):
                if name not in data:
                    continue
                a = data[name]
                denominator = 2 * np.sqrt(a["own_defect_mse"][r] * a["propagated_mse"][r])
                cosine = np.divide(a["cross_term"][r], denominator, out=np.full_like(denominator, np.nan),
                                   where=(denominator > 0) & (a["rollout_mse"][r, :-1] > 0))
                key = name if name in LABELS else name.rsplit("_", 1)[-1]
                axis.plot(t, cosine, color=COLORS[key], linestyle=":" if key == "clean_continuation" else "-", linewidth=1.5)
            axis.axhline(0, color="black", linewidth=.5)
            axis.set_ylim(-1.02, 1.02)
            pending(axis, [condition + "_" + key for key in ("recovery", "dynamics")], data)
            if r == 1:
                axis.set_xlabel("Input-state time")
            if c == 0:
                axis.set_ylabel("Alignment cosine")
    save(fig, output, "error_alignment")

    fig, axes = axes_for("Does correction preserve the scale of the motion?",
                        "Enstrophy along each predicted trajectory divided by ground-truth enstrophy at the same time.", metadata)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .88), ncol=4, frameon=False)
    for c, condition in enumerate(CONDITIONS):
        for r in range(2):
            axis = axes[r, c]
            for name in ("pretrained", "clean_continuation", condition + "_recovery", condition + "_dynamics"):
                if name not in data:
                    continue
                key = name if name in LABELS else name.rsplit("_", 1)[-1]
                axis.plot(rollout_t, data[name]["physical_values"][r, :, 2] / truth["physical_truth"][r, :, 2],
                          color=COLORS[key], linestyle=":" if key == "clean_continuation" else "-", linewidth=1.8)
            axis.axhline(1, color=COLORS["truth"], linestyle="--", linewidth=1)
            pending(axis, [condition + "_" + key for key in ("recovery", "dynamics")], data)
            if r == 1:
                axis.set_xlabel("State time")
            if c == 0:
                axis.set_ylabel("Enstrophy / ground truth")
    axes[0, 0].set_ylim(bottom=0)
    save(fig, output, "enstrophy")

    rows = []
    for name, values in data.items():
        for r, population in enumerate(metadata["populations"]):
            row = dict(method=name, population=population)
            for h in (8, 32):
                if h <= metadata["horizon"]:
                    row[f"rollout_rms_H{h}"] = float(np.sqrt(np.mean(values["rollout_mse"][r, 1:h+1])))
                    row[f"clean_rms_H{h}"] = float(np.sqrt(np.mean(values["clean_mse"][r, :h])))
                    row[f"own_defect_rms_H{h}"] = float(np.sqrt(np.mean(values["own_defect_mse"][r, :h])))
                    row[f"probe_defect_rms_H{h}"] = float(np.sqrt(np.mean(values["probe_defect_mse"][r, :h])))
            row["final_error"] = float(np.sqrt(values["rollout_mse"][r, -1]))
            rows.append(row)
    with (output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(json.dumps(dict(completed_methods=list(data), rows=rows,
        probe_clean_rms=[float(np.sqrt(np.mean(a))) for a in truth["probe_clean_mse"]]), indent=2) + "\n")
    if comparison_root:
        receipt = json.loads((Path(comparison_root) / "displacements/result.json").read_text())
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        times = np.array([r["input_step"] for r in receipt["rows"]]) * metadata["macro_dt"]
        for condition, label, color in zip(CONDITIONS, TITLES, ("#0072B2", "#D55E00", "#009E73")):
            magnitude = np.array(receipt["displacement_rms_scaled"][condition])
            axes[0].scatter(times, magnitude, s=5, alpha=.4, color=color, label=label)
            axes[1].plot(np.sort(magnitude), np.arange(1, len(magnitude)+1) / len(magnitude), color=color, label=label)
        axes[0].set(xlabel="Clean-center time", ylabel="Input displacement RMS / s", yscale="log")
        axes[1].set(xlabel="Input displacement RMS / s", ylabel="Fraction of training inputs", xscale="log")
        axes[1].legend(fontsize=8)
        fig.suptitle("Training exposure: pooled RMS matching does not match the magnitude distribution")
        fig.tight_layout()
        save(fig, output, "input_exposure")
    stems = ["rollout_errors", "one_step_errors", "probe_errors", "error_alignment", "enstrophy"]
    if comparison_root:
        stems.append("input_exposure")
    items = "".join(f'<h2>{html.escape(s.replace("_", " ").capitalize())}</h2><a href="{s}.pdf">PDF</a><img src="{s}.png">' for s in stems)
    (output / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>Corrective rollout comparison</title>'
        '<style>body{font:16px system-ui;max-width:1450px;margin:30px auto;padding:0 20px}img{width:100%}h2{margin-top:40px}</style>'
        '<h1>Corrective rollout comparison</h1><p>Completed: ' + html.escape(", ".join(data)) + '</p>'
        '<p>Two initial conditions, horizon 32. The probe is an independently initialized clean model. Its error is a diagnostic, not a manifold distance.</p>'
        '<p><a href="summary.csv">Numerical table (CSV)</a></p>' + items, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--comparison-root", type=Path)
    args = parser.parse_args()
    plot(args.input, args.output, args.comparison_root)


if __name__ == "__main__":
    main()
