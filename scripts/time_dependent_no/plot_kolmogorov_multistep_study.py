"""Render the October 4 study: --input RUN_OUTPUT --output FIGURE_DIRECTORY."""
from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path

from scripts.time_dependent_no.plot_kolmogorov_corrective_followup import curve, digest, plt, save
import numpy as np

STYLES = {
    "pretrained": ("Pretrained", "#777777", ":"),
    "clean_continuation": ("Original clean continuation", "#222222", "--"),
    "clean_centers_u4096": ("Clean center control", "#888888", "--"),
    "native_mixed_recovery": ("REC", "#0072B2", "-"),
    "mixed_midpoint": ("Half REC + half DYN", "#009E73", "-"),
    "native_mixed_dynamics": ("DYN", "#D55E00", "-"),
    "clean4": ("Clean windows", "#222222", "--"),
    "fixed4": ("Frozen inputs", "#0072B2", "-"),
    "detached4": ("Current inputs, detached", "#009E73", "-"),
    "attached4": ("Current inputs, attached", "#CC79A7", "-"),
}
GROUPS = {
    "midpoint": ("pretrained", "native_mixed_recovery", "mixed_midpoint", "native_mixed_dynamics"),
    "multistep": ("pretrained", "clean4", "fixed4", "detached4", "attached4"),
}


def load_arrays(path, expected=None):
    if expected is not None and digest(path) != expected:
        raise ValueError("figure source hash mismatch")
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def outcomes(root, output, meta, data):
    rows = []
    for name, values in data.items():
        for role in ("train", "validation"):
            mask = np.array(meta["populations"]) == role
            row = dict(method=name, population=role, cases=int(mask.sum()))
            for horizon in (8, 32):
                row[f"rollout_H{horizon}"] = float(np.sqrt(values["rollout_mse"][mask, 1:horizon+1].mean()))
                row[f"clean_H{horizon}"] = float(np.sqrt(values["clean_mse"][mask, :horizon].mean()))
                row[f"own_defect_H{horizon}"] = float(np.sqrt(values["own_defect_mse"][mask, :horizon].mean()))
            rows.append(row)
    with (output/"outcomes.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    costs = []
    for path in sorted((root/"training").glob("*/result.json")):
        r = json.loads(path.read_text())
        costs.append(dict(method=path.parent.name, status=r["status"], updates=r["updates_completed"],
                          seconds=r.get("seconds"), peak_memory_bytes=r.get("peak_memory_bytes")))
    if costs:
        with (output/"training_cost.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(costs[0]))
            writer.writeheader(); writer.writerows(costs)


def rollout_figures(output, meta, data, truth, captions):
    time = np.arange(meta["horizon"])*meta["macro_dt"]
    state_time = np.arange(meta["horizon"]+1)*meta["macro_dt"]
    for group, candidates in GROUPS.items():
        names = [n for n in candidates if n in data]
        if len(names) < 2 and names != ["mixed_midpoint"]:
            continue
        for role in ("train", "validation"):
            mask = np.array(meta["populations"]) == role
            caption = (f"Each model generates its own rollout from the same {int(mask.sum())} {role} initial conditions. "
                "Curves pool squared errors across these initial conditions before taking the square root; they do not show training-seed uncertainty. "
                "Local errors use input time; rollout error uses output time. All errors use RMS / fixed training scale s, without percentages. ")
            fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.0), layout="constrained")
            for ax, metric, title in zip(axes, ("clean_mse", "own_defect_mse", "rollout_mse"),
                    ("One-step error: ground-truth inputs", "One-step error: own rollout inputs", "Accumulated rollout error")):
                for name in names:
                    label, color, style = STYLES[name]
                    start = 1 if metric == "rollout_mse" else 0
                    x = state_time if metric == "rollout_mse" else time
                    curve(ax, x[start:], data[name][metric][:, start:], mask, label, color, style)
                ax.set_title(title, fontsize=9); ax.set_yscale("log")
                ax.set_xlabel("Output time" if metric == "rollout_mse" else "Input time")
                ax.set_ylabel("RMS / $s$")
            axes[0].legend(fontsize=6.5)
            save(fig, output, f"{group}_errors_{role}", caption+
                "The own-input error compares the learned next state with a solver restart from that same reached input.", captions)
            fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.0), layout="constrained")
            for name in names:
                label, color, style = STYLES[name]
                curve(axes[0], time, data[name]["probe_defect_mse"], mask, label, color, style)
                axes[1].plot(time, data[name]["cross_term"][mask].mean(0), color=color, linestyle=style, label=label)
                # Existing evaluator: mean vorticity, energy, enstrophy, palinstrophy.
                axes[2].plot(state_time, data[name]["physical_values"][mask, :, 2].mean(0),
                             color=color, linestyle=style, label=label)
            if "seed29_pretrained_clean_mse" in truth:
                curve(axes[0], time, truth["seed29_pretrained_clean_mse"], mask,
                      "Probe on ground truth", "#000000", "-.")
            axes[0].legend(fontsize=6)
            axes[0].set_yscale("log"); axes[0].set_ylabel("RMS / $s$")
            axes[0].set_title("Independent clean probe on each rollout", fontsize=9)
            axes[1].axhline(0, color="#aaaaaa", linewidth=.7)
            axes[1].set_ylabel("Signed cross term / $s^2$"); axes[1].set_title("Interaction of local and propagated errors", fontsize=9)
            axes[2].plot(state_time, truth["physical_truth"][mask, :, 2].mean(0), "k-.", label="Ground truth")
            axes[2].set_ylabel("Enstrophy"); axes[2].set_title("Physical evolution", fontsize=9)
            for ax in axes[:2]: ax.set_xlabel("Input time")
            axes[2].set_xlabel("State time"); axes[2].legend(fontsize=6)
            save(fig, output, f"{group}_probe_physics_{role}", caption+
                "A single independent clean probe is evaluated on each model's reached states against solver labels. "
                "The cross term is twice the inner product of local dynamics defect and propagated input error; negative values indicate cancellation. "
                "Enstrophy is averaged across initial conditions in physical units.", captions)


def gradient_figures(root, output, captions):
    records = {n: json.loads((root/"training"/n/"result.json").read_text())
               for n in GROUPS["multistep"][1:] if (root/"training"/n/"result.json").exists()}
    names = [n for n, record in records.items() if record["status"] == "completed"]
    if not names:
        return
    fig, axes = plt.subplots(2, len(names), figsize=(2.6*len(names), 5), squeeze=False, layout="constrained")
    fig2, axes2 = plt.subplots(1, 2, figsize=(7, 3), layout="constrained")
    for column, name in enumerate(names):
        ax = axes[0, column]
        result = records[name]
        steps = sorted(map(int, result["gradients"]))
        cosine, ratios = [], []
        for step in steps:
            record = result["gradients"][str(step)]
            data = load_arrays(root/"training"/name/record["file"], record["sha256"])
            ax.plot(np.arange(1, 4), np.nanmean(data["state_cosine"][:, :3], axis=0), marker="o", label=str(step))
            direct = data["state_direct_rms"][:, :3]
            relative = np.divide(data["state_future_rms"][:, :3], direct,
                                 out=np.full_like(direct, np.nan), where=direct > 0)
            axes[1, column].plot(np.arange(1, 4), np.nanmean(relative, axis=0), marker="o", label=str(step))
            cosine.append(float(data["parameter_cosine"]))
            ratios.append(float(data["parameter_attached_l2"]/data["parameter_detached_l2"]))
        ax.set_title(STYLES[name][0], fontsize=8); ax.set_ylim(-1.05, 1.05)
        ax.axhline(0, color="#aaaaaa", linewidth=.6); ax.set_xlabel("Intermediate state j")
        ax.set_ylabel("Immediate / future gradient cosine"); ax.legend(title="Updates", fontsize=6)
        axes[1, column].set_xlabel("Intermediate state j")
        axes[1, column].set_ylabel("Future / immediate gradient RMS")
        axes[1, column].set_yscale("log")
        label, color, style = STYLES[name]
        axes2[0].plot(steps, cosine, color=color, linestyle=style, marker="o", label=label)
        axes2[1].plot(steps, ratios, color=color, linestyle=style, marker="o", label=label)
    save(fig, output, "state_gradient_alignment", "Cosine between immediate-loss and future-loss state gradients, averaged over four fixed training onset windows. "
         "The lower row measures the ratio of their magnitudes. No future loss exists at the final state, so it is omitted. Negative cosine alone does not establish that attached training is inferior.", captions)
    axes2[0].set_ylabel("Attached / detached gradient cosine"); axes2[0].set_ylim(-1.05, 1.05)
    axes2[1].set_ylabel("Attached / detached gradient norm ratio"); axes2[1].set_yscale("log")
    for ax in axes2: ax.set_xlabel("Training updates")
    axes2[0].legend(fontsize=6)
    save(fig2, output, "parameter_gradient_comparison", "Both parameter gradients are evaluated at the same checkpoint on identical forward trajectories. "
         "The four fixed training onset windows are common across methods and checkpoints.", captions)


def midpoint_figures(root, output, meta, data, captions):
    path = root/"midpoint_analysis/result.json"
    names = ("native_mixed_recovery", "mixed_midpoint", "native_mixed_dynamics")
    if not path.exists() or any(name not in data for name in names):
        return
    record = json.loads(path.read_text())
    if record["status"] != "completed":
        return
    if record["case_ids"] != meta["case_ids"] or record["populations"] != meta["populations"]:
        raise ValueError("midpoint diagnostic population differs from rollout evaluation")
    training = {name: load_arrays(path.parent/(name+"_training_targets.npz"),
        record["artifacts"][name+"_training_targets.npz"]) for name in names}
    targets = load_arrays(path.parent/"common_targets.npz", record["artifacts"]["common_targets.npz"])
    x = np.arange(3)
    labels = ("REC", "Half REC\n+ half DYN", "DYN")
    paired = []
    cases = []
    for role in ("train", "validation"):
        mask = np.array(meta["populations"]) == role
        for horizon in (8, 32):
            errors = {n: np.sqrt(data[n]["rollout_mse"][mask, 1:horizon+1].mean(axis=1)) for n in names}
            for name in names:
                for case, value in zip(np.array(meta["case_ids"])[mask], errors[name]):
                    cases.append(dict(method=name, population=role, case_id=case, horizon=horizon, rollout_rms=value))
            for endpoint in (names[0], names[2]):
                ratio = np.sqrt(np.mean(errors[names[1]]**2)/np.mean(errors[endpoint]**2))
                paired.append(dict(population=role, horizon=horizon, endpoint=endpoint,
                    midpoint_to_endpoint=ratio, midpoint_wins=int(np.sum(errors[names[1]] < errors[endpoint])),
                    cases=int(mask.sum())))
    for filename, rows in (("midpoint_cases.csv", cases), ("midpoint_paired_comparisons.csv", paired)):
        with (output/filename).open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    mask = np.array(meta["populations"]) == "validation"
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.0), layout="constrained")
    for metric, label, color in (("rec_mse", "Against recovery target", "#0072B2"),
                                 ("dyn_mse", "Against dynamics target", "#D55E00")):
        axes[0].plot(x, [np.sqrt(training[n][metric].mean()) for n in names],
                     marker="o", color=color, label=label)
    axes[0].set_title("(a) Same displaced training inputs", fontsize=9)
    axes[0].set_ylabel("Target error: RMS / $s$")
    axes[0].legend(fontsize=7)
    for ax, horizon, letter in zip(axes[1:], (8, 32), ("b", "c")):
        values = np.stack([data[n]["rollout_mse"][mask, 1:horizon+1].mean(axis=1) for n in names])
        for i in range(values.shape[1]):
            ax.plot(x, np.sqrt(values[:, i]), color="#999999", alpha=.6, linewidth=.8,
                    marker="o", markersize=3, label="One initial condition" if i == 0 else None)
        pooled = np.sqrt(values.mean(axis=1))
        ax.plot(x, pooled, color="#009E73", marker="D", linewidth=2, label="Pooled error")
        ax.set_title(f"({letter}) Validation rollout through step {horizon}", fontsize=9)
        ax.set_ylabel("Rollout error: RMS / $s$")
        ax.legend(fontsize=7)
    for ax in axes:
        ax.set_xticks(x, labels, fontsize=8)
        ax.set_xlabel("Corrective training target")
        ax.set_ylim(bottom=0)
    save(fig, output, "midpoint_targets_and_outcomes",
        "(a) Each trained model is queried on the same 1,024 displaced training inputs, with native errors from lengths 1, 2, 4 and 8. "
        "The blue and orange curves measure errors against the recovery and dynamics targets, respectively. "
        "(b,c) Each model generates its own rollout from the same four validation initial conditions. Thin lines pair the same initial condition across models; "
        "diamonds pool squared errors over states 1 through H and all four cases before taking the square root. "
        "The midpoint uses an equally weighted recovery/dynamics target on the corrective branch. Lines connect three trained models, not a continuous weight sweep. "
        "Each condition has one training run, so the case variation does not estimate training-seed uncertainty.", captions)
    for role in ("train", "validation"):
        mask = np.array(meta["populations"]) == role
        fig, axes = plt.subplots(1, 2, figsize=(7, 3), layout="constrained")
        for ax, metric in zip(axes, ("rec_mse", "dyn_mse")):
            for name in ("pretrained", *names):
                curve(ax, np.arange(meta["horizon"])*meta["macro_dt"], targets[name+"__"+metric], mask, *STYLES[name])
            ax.set_yscale("log"); ax.set_xlabel("Input time on pretrained rollout"); ax.set_ylabel("RMS / $s$")
            ax.set_title("Recovery target error" if metric == "rec_mse" else "Dynamics target error", fontsize=9)
        axes[0].legend(fontsize=7)
        save(fig, output, f"midpoint_common_targets_{role}",
            f"All models are queried on the same saved pretrained-model rollout inputs from {int(mask.sum())} {role} initial conditions. "
            "The recovery target is the ground-truth next state at the corresponding time; the dynamics target is a solver restart from the reached input. "
            "These are common-input measurements, distinct from each model's autonomous rollout errors. Curves pool squared errors across cases before taking the square root.", captions)


def diagnostic_figures(root, output, meta, captions):
    path = root/"diagnostics/result.json"
    if not path.exists():
        return
    record = json.loads(path.read_text())
    if record["status"] != "completed":
        return
    target = load_arrays(path.parent/"common_targets.npz", record["artifacts"]["common_targets.npz"])
    geometry = load_arrays(path.parent/"geometry.npz", record["artifacts"]["geometry.npz"])
    response = load_arrays(path.parent/"responses.npz", record["artifacts"]["responses.npz"])
    requests = json.loads((path.parent/"response_requests.json").read_text())["requests"]
    for group, names in GROUPS.items():
        for role in ("train", "validation"):
            mask = np.array(meta["populations"]) == role
            fig, axes = plt.subplots(1, 2, figsize=(7, 3), layout="constrained")
            for ax, metric in zip(axes, ("rec_mse", "dyn_mse")):
                for name in names:
                    curve(ax, np.arange(32)*.05, target[name+"__"+metric], mask, *STYLES[name])
                ax.set_yscale("log"); ax.set_xlabel("Input time"); ax.set_ylabel("RMS / $s$")
                ax.set_title("Recovery target error" if metric == "rec_mse" else "Dynamics target error")
            axes[0].legend(fontsize=6)
            save(fig, output, f"{group}_common_targets_{role}",
                 "All maps are evaluated on identical saved pretrained-rollout inputs. The evaluating model does not generate these inputs. "
                 f"Curves pool squared error over the {role} initial conditions before taking the square root.", captions)
            fig, axes = plt.subplots(1, 3, figsize=(10.2, 3), layout="constrained")
            for ax, metric, title in zip(axes, ("nearest_mse", "pca_mse_r8", "pca_mse_r16"),
                                        ("Nearest training state", "PCA residual: rank 8", "PCA residual: rank 16")):
                for name in names:
                    curve(ax, np.arange(33)*.05, geometry[name+"__"+metric], mask, *STYLES[name])
                curve(ax, np.arange(33)*.05, geometry["ground_truth__"+metric], mask, "Ground-truth coverage", "#000000", "-.")
                ax.set_title(title); ax.set_xlabel("State time"); ax.set_ylabel("RMS / $s$")
            axes[0].legend(fontsize=6)
            save(fig, output, f"{group}_geometry_{role}",
                 "Each curve uses that model's own rollout. The representation contains only training states at the same time; the evaluated training trajectory is excluded. "
                 "Ground-truth curves calibrate finite-sample coverage. These are empirical distances and subspace residuals, not exact manifold distances.", captions)
        for kind in ("finite", "central"):
            fig, axes = plt.subplots(2, 3, figsize=(10.2, 5.6), layout="constrained")
            for column, direction in enumerate(("actual_error", "pca_parallel", "pca_orthogonal")):
                mask = np.array([r["population"] == "validation" and r["direction"] == direction and
                                 r["kind"] == kind and (kind == "finite" or r["amplitude"] == .005) for r in requests], dtype=bool)
                for i, name in enumerate(names):
                    label, color, _ = STYLES[name]
                    for ax, key in zip(axes[:, column], ("response_error_gain", "cosine")):
                        values = response[name+"__"+key][mask]
                        if not len(values):
                            ax.text(.5, .5, "No resolved directions", ha="center", transform=ax.transAxes)
                            continue
                        resolved = response[name+"__local_response_resolved"][mask] if kind == "central" else np.ones(len(values), dtype=bool)
                        ax.scatter(np.full(int(resolved.sum()), i), values[resolved], color=color, alpha=.3, s=7)
                        ax.scatter(np.full(int((~resolved).sum()), i), values[~resolved], color=color, marker="x", alpha=.6, s=12)
                        ax.scatter(i, np.nanmean(values), color=color, marker="D", s=28)
                        ax.set_xticks(range(len(names)), [STYLES[n][0] for n in names], rotation=60, ha="right", fontsize=6)
                axes[0, column].set_title(direction.replace("_", " "), fontsize=9)
                axes[0, column].set_ylabel("Response error / input displacement")
                axes[1, column].set_ylabel("Learned / true response cosine"); axes[1, column].set_ylim(-1.05, 1.05)
            save(fig, output, f"{group}_responses_{kind}",
                 "Responses along fixed pretrained-rollout errors and their empirical PCA components, using validation cases and input steps 1, 4, 8, 16 and 31. "
                 "Dots are individual measurements, diamonds arithmetic means. "+
                 ("Finite responses retain each component's actual amplitude. " if kind == "finite" else
                  "Centered responses use RMS amplitude 0.005s; crosses fail the recorded amplitude-agreement or solver-refinement qualification. Diamonds still average all measurements. ")+
                 "PCA components do not establish true normal or tangent directions, and these samples are not uniform bounds.", captions)


def plot(root, output):
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    meta = json.loads((root/"evaluation/result.json").read_text())
    data = {}
    for name in meta["completed_methods"]:
        record = json.loads((root/"evaluation"/name/"result.json").read_text())
        data[name] = load_arrays(root/"evaluation"/name/"metrics.npz", record["artifacts"]["metrics.npz"])
    truth = load_arrays(root/"evaluation/ground_truth.npz", meta.get("ground_truth_sha256"))
    captions = {}
    outcomes(root, output, meta, data)
    rollout_figures(output, meta, data, truth, captions)
    midpoint_figures(root, output, meta, data, captions)
    gradient_figures(root, output, captions)
    diagnostic_figures(root, output, meta, captions)
    (output/"captions.json").write_text(json.dumps(captions, indent=2), encoding="utf-8")
    body = "\n".join(f'<figure><a href="{name}.pdf"><img src="{name}.png" style="max-width:1100px;width:100%"></a>'
                     f'<figcaption>{html.escape(caption)}</figcaption></figure>' for name, caption in captions.items())
    (output/"index.html").write_text('<!doctype html><meta charset="utf-8"><title>Controlled corrective study</title>'
        '<body style="font:16px sans-serif;max-width:1150px;margin:30px auto"><h1>Controlled corrective study</h1>'
        '<p><a href="outcomes.csv">Numerical summary</a>. Click a figure for its vector PDF. '
        'Partial galleries show only completed evaluations.</p>'+body+'</body>', encoding="utf-8")
    sources = [root/"evaluation/result.json", root/"evaluation/ground_truth.npz"]
    sources += [root/"evaluation"/n/"metrics.npz" for n in meta["completed_methods"]]
    sources += list((root/"training").glob("*/gradients_*.npz"))
    sources += list((root/"diagnostics").glob("*.npz"))
    sources += list((root/"midpoint_analysis").glob("*.npz"))
    if (root/"midpoint_analysis/result.json").exists():
        sources.append(root/"midpoint_analysis/result.json")
    manifest = dict(renderer_sha256=digest(__file__), figures=list(captions),
        source_hashes={p.relative_to(root).as_posix(): digest(p) for p in sources},
        files={p.name: digest(p) for p in sorted(output.iterdir()) if p.name != "manifest.json" and p.is_file()})
    (output/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(args.input, args.output)
