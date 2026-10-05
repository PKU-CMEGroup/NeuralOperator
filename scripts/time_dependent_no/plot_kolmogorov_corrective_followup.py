"""Render reproducible Section 4.2 figures from the corrective follow-up arrays.

python -m scripts.time_dependent_no.plot_kolmogorov_corrective_followup
--input FOLLOWUP_OUTPUT --output FIGURE_DIRECTORY
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
import numpy as np

REC, DYN, GRAY, PURPLE = "#0072B2", "#D55E00", "#737373", "#CC79A7"
CONDITIONS = (("native_one", "One-step errors"), ("native_mixed", "Mixed rollout errors"), ("gaussian", "Gaussian noise"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pool(values, mask):
    return np.sqrt(np.mean(np.asarray(values)[mask], axis=0))


def curve(ax, x, values, mask, label, color, style="-", paths=False, omit_first=False):
    values = np.array(np.asarray(values)[mask], dtype=float, copy=True)
    if omit_first:
        values[:, 0] = np.nan
    if paths and len(values) > 1:
        for row in values:
            ax.plot(x, np.sqrt(row), color=color, alpha=.16, linewidth=.65, linestyle=style)
    ax.plot(x, np.sqrt(np.mean(values, axis=0)), color=color, linestyle=style, label=label, linewidth=1.5)


def save(fig, output, stem, caption, captions):
    for ax in fig.axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=.15, linewidth=.5)
    fig.savefig(output/(stem+".pdf"), bbox_inches="tight")
    fig.savefig(output/(stem+".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)
    captions[stem] = caption


def target_figures(root, output, folder, prefix, captions):
    p = root/folder
    if not (p/"result.json").exists():
        return
    r = json.loads((p/"result.json").read_text())
    if digest(p/"metrics.npz") != r["metrics_sha256"]:
        raise ValueError("target figure source changed")
    with np.load(p/"metrics.npz") as f:
        data = {k: f[k] for k in f.files}
    time = np.arange(r["horizon"])*r["macro_dt"]
    for role in ("train", "validation"):
        mask = np.array(r["populations"]) == role
        fig, axes = plt.subplots(2, 3, figsize=(7.1, 4.8), sharex=True, layout="constrained")
        for column, (condition, title) in enumerate(CONDITIONS):
            axes[0, column].set_title(title)
            for row, metric in enumerate(("rec_mse", "dyn_mse")):
                ax = axes[row, column]
                for name, label, color, style in (
                        ("pretrained", "Pretrained", GRAY, ":"),
                        ("clean_continuation", "Clean continuation", "#222222", "--"),
                        (condition+"_recovery", "REC", REC, "-"),
                        (condition+"_dynamics", "DYN", DYN, "-")):
                    curve(ax, time, data[name+"__pretrained__"+metric], mask, label, color, style,
                          paths=name.endswith(("recovery", "dynamics")))
                curve(ax, time, data[condition+"_recovery__pretrained__separation_mse"], mask,
                      "Target separation", "#009E73", "-.", omit_first=True)
                ax.set_yscale("log")
                ax.set_ylabel((r"$E_{\rm rec}$" if row == 0 else r"$E_{\rm dyn}$")+" (RMS / $s$)")
                if row == 1:
                    ax.set_xlabel("Input time")
        axes[0, 0].legend(fontsize=6.4)
        save(fig, output, f"{prefix}_targets_{role}",
             f"Recovery and dynamics errors on identical inputs from the pretrained rollout. "
             f"Each curve identifies the evaluating model; no evaluated prediction changes the input path. "
             f"The green curve is the separation between the solver next state and ground-truth next state. "
             f"Thick curves pool squared error over {int(mask.sum())} {role} trajectories before taking the square root; "
             "faint curves show individual corrective-model cases, not training-seed uncertainty. "
             "All local errors are indexed by input time, with targets one time step later. "
             "Target separation starts after the common initial state, where it is zero up to numerical precision. "
             "The fixed training scale is used without a percentage multiplier.", captions)
        for condition, title in CONDITIONS:
            fig, axes = plt.subplots(2, 2, figsize=(6.7, 4.5), sharex=True, layout="constrained")
            for column, generator in enumerate((condition+"_recovery", condition+"_dynamics")):
                axes[0, column].set_title(("REC" if column == 0 else "DYN")+" generates the input trajectory")
                for row, metric in enumerate(("rec_mse", "dyn_mse")):
                    ax = axes[row, column]
                    for target, color in (("recovery", REC), ("dynamics", DYN)):
                        name = condition+"_"+target
                        curve(ax, time, data[name+"__"+generator+"__"+metric], mask,
                              "REC evaluates" if target == "recovery" else "DYN evaluates", color, paths=True)
                    curve(ax, time, data[condition+"_recovery__"+generator+"__separation_mse"], mask,
                          "Target separation", "#009E73", "-.", omit_first=True)
                    ax.set_yscale("log")
                    ax.set_ylabel((r"$E_{\rm rec}$" if row == 0 else r"$E_{\rm dyn}$")+" (RMS / $s$)")
                    if row == 1:
                        ax.set_xlabel("Input time")
            axes[0, 0].legend(fontsize=7)
            save(fig, output, f"{prefix}_cross_{condition}_{role}",
                 f"{title}: each column fixes a rollout generator and compares REC and DYN on those identical inputs. "
                 "Top: error against the ground-truth next state. Bottom: error against the solver next state from the same input. "
                 "When evaluator and generator coincide, the recovery curve equals that generator's next-step rollout error. "
                 f"Aggregation and normalization use the same {int(mask.sum())} {role} cases as the common-input comparison. "
                 "Target separation at the common initial state is omitted from the logarithmic display because its ideal value is zero.", captions)
        fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.6), layout="constrained")
        for ax, (condition, title) in zip(axes, CONDITIONS):
            for target, color in (("recovery", REC), ("dynamics", DYN)):
                key=condition+"_"+target+"__pretrained__"
                change=data[key+"rec_mse"]-data[key+"separation_mse"]
                ax.plot(time,change[mask].mean(axis=0),color=color,label="REC" if target=="recovery" else "DYN")
            ax.axhline(0,color=GRAY,linestyle=":")
            ax.set(title=title,xlabel="Input time",ylabel="Change in squared error / $s^2$")
        axes[0].legend()
        save(fig, output, f"{prefix}_signed_effect_{role}",
             "Change in squared trajectory error relative to taking one solver step from the identical pretrained-rollout input. The plotted quantity is recovery error squared minus target separation squared, equivalently dynamics error squared plus its signed cross term with the propagated displacement. Negative values mean the learned prediction is closer to the ground-truth next state than the solver next state is. The mean is taken over trajectories after computing the signed quantity; no square root is applied.", captions)


def comparison_figure(output, stem, panels, data, roles, role, captions, caption, clean=False):
    mask = np.array(roles) == role
    if not all(name in data for _, lines in panels for name, *_ in lines):
        return
    rows = 2 if clean else 1
    fig, axes = plt.subplots(rows, len(panels), figsize=(7.1, 4.5 if clean else 2.7),
                             squeeze=False, layout="constrained")
    for col, (title, lines) in enumerate(panels):
        axes[0, col].set_title(title)
        for row in range(rows):
            metric = "clean_mse" if clean and row == 0 else "rollout_mse"
            ax = axes[row, col]
            for name, label, color, style in lines:
                values = data[name][metric]
                time = np.arange(values.shape[-1])*.05
                curve(ax, time, values, mask, label, color, style, paths=False)
            ax.set_xlabel("Input time" if metric == "clean_mse" else "Rollout time")
            ax.set_ylabel("Clean one-step error / $s$" if metric == "clean_mse" else "Rollout error / $s$")
            if metric == "clean_mse":
                ax.set_yscale("log")
            else:
                ax.set_ylim(bottom=0)
            ax.legend(fontsize=6.5)
    save(fig, output, stem+"_"+role, caption+f" Curves pool squared errors across {int(mask.sum())} {role} initial conditions before taking the square root. All models roll out autonomously from the same initial conditions.", captions)


def autonomous_diagnostics(root, output, data, meta, role, captions):
    names = (("clean_centers_u4096", "Clean control"),
             ("native_mixed_recovery", "Mixed-length REC"),
             ("native_mixed_dynamics", "Mixed-length DYN"))
    if not all(name in data for name, _ in names):
        return
    mask = np.array(meta["populations"]) == role
    with np.load(root/"evaluation/ground_truth.npz") as f:
        probe_clean = f["seed29_pretrained_clean_mse"]
    fig, axes = plt.subplots(2, 3, figsize=(7.3, 4.7), sharey="row", layout="constrained")
    time = np.arange(meta["horizon"])*meta["macro_dt"]
    for col, (name, title) in enumerate(names):
        axes[0, col].set_title(title)
        for metric, label, color, style in (
                ("clean_mse", "Model on ground truth", "#009E73", "--"),
                ("own_defect_mse", "Model on its own rollout", REC, "-"),
                ("probe_defect_mse", "Probe on this rollout", PURPLE, "-")):
            curve(axes[0, col], time, data[name][metric], mask, label, color, style)
        curve(axes[0, col], time, probe_clean, mask, "Probe on ground truth", GRAY, ":")
        axes[0, col].set(yscale="log", xlabel="Input time")
        for generator, label, color, style in ((name, "This model", REC, "-"),
                ("pretrained", "Unchanged pretrained model", GRAY, "--")):
            curve(axes[1, col], np.arange(meta["horizon"]+1)*meta["macro_dt"],
                  data[generator]["rollout_mse"], mask, label, color, style)
        axes[1, col].set(xlabel="Rollout time", ylim=(0, None))
    axes[0, 0].set_ylabel("One-step dynamics error\n(RMS / $s$)")
    axes[1, 0].set_ylabel("Rollout error\n(RMS / $s$)")
    axes[1, 0].legend(loc="upper left", fontsize=6.5)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside upper center", ncol=2, fontsize=7)
    save(fig, output, "autonomous_diagnostics_"+role,
         f"Accuracy along autonomous trajectories from the same {int(mask.sum())} {role} initial conditions. "
         "Each column fixes a model trained for 4096 additional updates. The clean control uses the same clean centers and sample sequence as the corrective pair. "
         "Top: one-step dynamics error against the solver next state, evaluated on ground-truth inputs and on that model's own rollout inputs. "
         "The fixed independently initialized probe evaluates the same two input populations; its calibration on ground truth is shown in gray. "
         "Bottom: accumulated error against the evolving ground-truth trajectory, with the unchanged pretrained model retained. "
         "Curves pool squared errors across initial conditions before taking the square root. "
         "Local errors use input time; rollout errors use predicted-state time. Probe error measures predictive competence, not manifold distance.", captions)


def training_target_figure(root, output, captions):
    paths = [root/"training"/(c+"_"+t)/"training_004096.npz"
             for c, _ in CONDITIONS for t in ("recovery", "dynamics")]
    if not all(p.exists() for p in paths) or not (root/"all_targets/result.json").exists():
        return
    receipt = json.loads((root/"all_targets/result.json").read_text())
    if digest(root/"all_targets/metrics.npz") != receipt["metrics_sha256"]:
        raise ValueError("target transfer source changed")
    with np.load(root/"all_targets/metrics.npz") as f:
        reached = {k: f[k] for k in f.files}
    mask = np.array(receipt["populations"]) == "validation"
    fig, axes = plt.subplots(2, 3, figsize=(7.3, 4.7), layout="constrained")
    for col, (condition, title) in enumerate(CONDITIONS):
        axes[0, col].set_title(title)
        for target, color, offset in (("recovery", REC, -.18), ("dynamics", DYN, .18)):
            name = condition+"_"+target
            with np.load(root/"training"/name/"training_004096.npz") as f:
                trained = {k: f[k] for k in f.files}
            for row in range(2):
                values = trained if row == 0 else {
                    k: reached[name+"__pretrained__"+k][mask]
                    for k in ("rec_mse", "dyn_mse", "separation_mse")}
                heights = [np.sqrt(values[k].mean()) for k in ("rec_mse", "dyn_mse")]
                axes[row, col].bar(np.arange(2)+offset, heights, width=.32,
                                   color=color, label="REC model" if target == "recovery" else "DYN model")
                if target == "recovery":
                    axes[row, col].axhline(np.sqrt(values["separation_mse"].mean()),
                                          color="#009E73", linestyle="--", label="Target separation")
                if row == 1:
                    for x, key in enumerate(("rec_mse", "dyn_mse")):
                        axes[row, col].scatter(np.full(int(mask.sum()), x+offset),
                                              np.sqrt(values[key].mean(axis=1)),
                                              s=10, facecolor="white", edgecolor="#222222", linewidth=.5, zorder=3)
        for row in range(2):
            axes[row, col].set(xticks=[0, 1], xticklabels=["Recovery", "Dynamics"],
                              xlabel="Evaluation target", ylabel=("Training inputs" if row == 0 else "Pretrained rollout inputs")+"\nError (RMS / $s$)")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside upper center", ncol=3, fontsize=7)
    save(fig, output, "target_specialization_and_transfer",
         "Training-target specialization and transfer to common rollout inputs at 4096 updates. "
         "Blue and orange identify the REC and DYN models; horizontal positions identify the target used to measure error. "
         "Top: the same 1024 displaced training inputs for each REC/DYN pair. Bottom: identical inputs from the unchanged pretrained model's four validation rollouts through step 32. "
         "Bars pool squared error over inputs before taking the square root; white markers show individual validation trajectories. "
         "The dashed line is the RMS separation of the two targets on the corresponding inputs. "
         "The rows differ in input distribution and time weighting, so their difference is a transfer diagnostic rather than a controlled training-validation gap. "
         "The plotted models are fixed terminals, not checkpoints selected for these outcomes.", captions)


def comparison_tables(root, output, data, meta):
    pairs = [("Clean continuation", "clean_continuation", "pretrained"),
             ("Exact clean control", "clean_centers_u4096", "pretrained")]
    for condition, _ in CONDITIONS:
        for target in ("recovery", "dynamics"):
            name = condition+"_"+target
            pairs += [("Target and inputs", name, "clean_centers_u4096"),
                      ("Training duration", name+"_u8192", name)]
    for target in ("recovery", "dynamics"):
        pairs += [("Early coverage", "early_one_"+target+"_u4096", "native_one_"+target),
                  ("Early corrective effect", "early_one_"+target+"_u4096", "early_one_clean_u4096"),
                  ("Gaussian amplitudes", "gaussian_profile_"+target+"_u4096", "gaussian_"+target),
                  ("Second initialization", "seed29_one_"+target+"_u4096", "seed29_one_clean_u4096")]
    rows = []
    for question, model, baseline in pairs:
        if model not in data or baseline not in data:
            continue
        for role in ("train", "validation"):
            mask = np.array(meta["populations"]) == role
            for horizon in (8, 32):
                a, b = [data[n]["rollout_mse"][mask, 1:horizon+1].mean(axis=1) for n in (model, baseline)]
                rows.append(dict(question=question, model=model, baseline=baseline, population=role,
                                 horizon=horizon, model_RMS=float(np.sqrt(a.mean())),
                                 baseline_RMS=float(np.sqrt(b.mean())), ratio=float(np.sqrt(a.mean()/b.mean())),
                                 improved_cases=int((a < b).sum()), cases=int(mask.sum())))
    if rows:
        with (output/"comparisons.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    if not (root/"all_targets/result.json").exists():
        return
    receipt = json.loads((root/"all_targets/result.json").read_text())
    with np.load(root/"all_targets/metrics.npz") as f:
        targets = {k: f[k] for k in f.files}
    rows = []
    for request in receipt["requests"]:
        prefix = request["predictor"]+"__"+request["input_generator"]+"__"
        for role in ("train", "validation"):
            mask = np.array(receipt["populations"]) == role
            row = dict(**request, population=role)
            for key in ("rec_mse", "dyn_mse", "separation_mse"):
                row[key.replace("_mse", "_RMS")] = float(np.sqrt(targets[prefix+key][mask].mean()))
            row["signed_change"] = float((targets[prefix+"rec_mse"]-targets[prefix+"separation_mse"])[mask].mean())
            rows.append(row)
    with (output/"target_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def plot(root, output):
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family":"serif", "font.size":8, "axes.titlesize":8,
                         "axes.labelsize":8, "legend.fontsize":7, "pdf.fonttype":42, "ps.fonttype":42})
    captions = {}
    target_source = "original_targets" if (root/"original_targets/result.json").exists() else "pilot_targets"
    target_figures(root, output, target_source, "mechanisms", captions)
    if (root/"sampling.npz").exists():
        receipt = json.loads((root/"sampling.json").read_text())
        if digest(root/"sampling.npz") != receipt["metrics_sha256"]:
            raise ValueError("sampling data changed")
        with np.load(root/"sampling.npz") as f:
            sampling = {k:f[k] for k in f.files}
        fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.7), layout="constrained")
        for (key, label), color in zip(CONDITIONS, (GRAY, REC, DYN)):
            x = np.sort(sampling[key])
            axes[0].plot(x, np.arange(1, len(x)+1)/len(x), label=label, color=color)
        axes[0].set(xscale="log", xlabel="Training displacement RMS / $s$", ylabel="Fraction of inputs")
        axes[0].legend()
        for offset, metric, label, color in ((-.18,"rec_mse","REC loss",REC),(.18,"dyn_mse","DYN loss",DYN)):
            shares = [sampling[metric][sampling["length"]==k].sum()/sampling[metric].sum() for k in (1,2,4,8)]
            axes[1].bar(np.arange(4)+offset, shares, width=.35, label=label, color=color)
        axes[1].axhline(.25, color=GRAY, linestyle=":", label="Equal input counts")
        axes[1].set(xticks=np.arange(4), xticklabels=["1","2","4","8"], xlabel="Generating rollout length", ylabel="Fraction of displaced training loss", ylim=(0,1))
        axes[1].legend()
        save(fig, output, "sampling_and_loss", "Left: empirical distributions of the original 1,024 training displacement magnitudes. Right: contributions of each generating length to the displaced-input MSE at update 4,096. Each length supplies one quarter of inputs. These are training-set measurements; loss contributions are not gradient contributions.", captions)
    eroot = root/"evaluation"
    if (eroot/"result.json").exists():
        meta = json.loads((eroot/"result.json").read_text())
        data, probe_names = {}, {}
        for name in meta["completed_methods"]:
            r = json.loads((eroot/name/"result.json").read_text())
            if digest(eroot/name/"metrics.npz") != r["artifacts"]["metrics.npz"]:
                raise ValueError("evaluation metrics changed")
            with np.load(eroot/name/"metrics.npz") as f:
                data[name] = {k:f[k] for k in f.files}
            probe_names[name] = r["probe_name"]
        rows = []
        for name, values in data.items():
            for i, (case, role) in enumerate(zip(meta["case_ids"], meta["populations"])):
                row = dict(method=name, case_id=case, population=role, probe=probe_names[name])
                for h in (8,32):
                    row[f"rollout_H{h}"] = float(np.sqrt(values["rollout_mse"][i,1:h+1].mean()))
                    row[f"clean_H{h}"] = float(np.sqrt(values["clean_mse"][i,:h].mean()))
                row["own_dynamics_RMS"] = float(np.sqrt(values["own_defect_mse"][i].mean()))
                row["probe_dynamics_RMS"] = float(np.sqrt(values["probe_defect_mse"][i].mean()))
                rows.append(row)
        if rows:
            with (output/"summary.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        for role in ("train", "validation"):
            autonomous_diagnostics(root, output, data, meta, role, captions)
            panels = [(title, [("pretrained","Pretrained",GRAY,":"), ("clean_continuation","Clean continuation","#222222","--"),
                       (c+"_recovery","REC",REC,"-"),(c+"_dynamics","DYN",DYN,"-")]) for c,title in CONDITIONS]
            comparison_figure(output,"clean_and_rollout",panels,data,meta["populations"],role,captions,
                "Clean next-state accuracy and autonomous rollout error for the original fixed-budget models. Columns identify displaced training inputs; each REC/DYN pair shares exactly the same inputs.",clean=True)
            if all(c+"_"+t in data for c,_ in CONDITIONS for t in ("recovery","dynamics")):
                with np.load(eroot/"ground_truth.npz") as f:
                    physical_truth=f["physical_truth"]
                    probe_clean=f["seed29_pretrained_clean_mse"]
                mask=np.array(meta["populations"])==role
                fig,axes=plt.subplots(2,3,figsize=(7.1,4.8),layout="constrained")
                for col,(condition,title) in enumerate(CONDITIONS):
                    axes[0,col].set_title(title)
                    for name,label,color,style in (("pretrained","Pretrained path",GRAY,":"),
                            ("clean_continuation","Clean-continuation path","#222222","--"),
                            (condition+"_recovery","REC path",REC,"-"),(condition+"_dynamics","DYN path",DYN,"-")):
                        curve(axes[0,col],np.arange(32)*.05,data[name]["probe_defect_mse"],mask,label,color,style)
                        ratio=data[name]["physical_values"][:,:,2]/physical_truth[:,:,2]
                        axes[1,col].plot(np.arange(33)*.05,ratio[mask].mean(axis=0),label=label,color=color,linestyle=style)
                    curve(axes[0,col],np.arange(32)*.05,probe_clean,mask,"Ground-truth inputs","#009E73","-.")
                    axes[0,col].set(yscale="log",xlabel="Input time",ylabel="Fixed probe dynamics error / $s$")
                    axes[1,col].axhline(1,color="#009E73",linestyle="-.")
                    axes[1,col].set(xlabel="Rollout time",ylabel="Enstrophy / ground-truth enstrophy",ylim=(0,None))
                axes[0,0].legend(fontsize=6.3)
                save(fig,output,"probe_and_enstrophy_"+role,
                     "Top: one fixed independently initialized clean probe evaluates the inputs generated by each named model; its target is the solver next state from that input. The green curve calibrates the same probe on ground-truth inputs. These are predictive-competence measurements, not manifold distances. Bottom: mean of the per-trajectory enstrophy ratios against the evolving ground truth; a ratio near one does not establish spatial accuracy.",captions)
            panels = [(title, [(c+"_recovery","REC: 4096",REC,"--"),(c+"_recovery_u8192","REC: 8192",REC,"-"),
                              (c+"_dynamics","DYN: 4096",DYN,"--"),(c+"_dynamics_u8192","DYN: 8192",DYN,"-")]) for c,title in CONDITIONS]
            comparison_figure(output,"training_duration",panels,data,meta["populations"],role,captions,
                "Training-duration comparison with the original inputs, losses, optimizer states and sample sequences preserved. Both fixed checkpoints are shown; neither is selected using validation performance.")
            panels = []
            for target,color in (("clean",GRAY),("recovery",REC),("dynamics",DYN)):
                old = "clean_centers_u4096" if target == "clean" else "native_one_"+target
                panels.append((target.upper(),[(old,"Original input times",GRAY,"--"),("early_one_"+target+"_u4096","More early inputs",color,"-")]))
            comparison_figure(output,"early_coverage",panels,data,meta["populations"],role,captions,
                "One-step-exposure coverage comparison at 4096 updates. Both designs use 1024 inputs. The revised design allocates half to input times 1--31. The clean controls use the same associated ground-truth states and sample tapes.",clean=True)
            panels = [(target.upper(),[("gaussian_"+target,"Constant Gaussian scale",GRAY,"--"),
                     ("gaussian_profile_"+target+"_u4096","Native amplitude profile",PURPLE,"-"),
                     ("native_mixed_"+target,"Native mixed inputs",REC,"-")]) for target in ("recovery","dynamics")]
            comparison_figure(output,"gaussian_amplitudes",panels,data,meta["populations"],role,captions,
                "Gaussian amplitude control at 4096 updates. The Gaussian directions and clean centers are preserved; the expected Gaussian RMS at each center is set to its native mixed-error magnitude. Native errors are unchanged.",clean=True)
            panels = []
            for target in ("clean","recovery","dynamics"):
                old = "clean_centers_u4096" if target == "clean" else "native_one_"+target
                panels.append((target.upper(),[("pretrained","Original before correction",GRAY,":"),(old,"Original after 4096",GRAY,"-"),
                    ("seed29_pretrained","Independent before correction",PURPLE,":"),("seed29_one_"+target+"_u4096","Independent after 4096",PURPLE,"-")]))
            comparison_figure(output,"initialization",panels,data,meta["populations"],role,captions,
                "Replication of the one-step-exposure comparison from a second clean-model initialization. Each model generates its own frozen one-step training errors at the same clean centers. This tests the one-step question, not every corrective-method ranking.")
        comparison_tables(root, output, data, meta)
        training_target_figure(root, output, captions)
    training=root/"training"
    if training.exists():
        records={p.parent.name:json.loads(p.read_text()) for p in training.glob("*/result.json")}
        fig,axes=plt.subplots(2,3,figsize=(7.1,4.5),layout="constrained")
        populated=False
        for col,(condition,title) in enumerate(CONDITIONS):
            axes[0,col].set_title(title)
            for target,color in (("recovery",REC),("dynamics",DYN)):
                record=records.get(condition+"_"+target,{})
                measurements=record.get("measurements",{})
                if not measurements: continue
                updates=sorted(map(int,measurements))
                values=[measurements[str(n)] for n in updates]
                metric="rec_mse" if target=="recovery" else "dyn_mse"
                for row,key in enumerate(("clean_mse",metric)):
                    axes[row,col].plot(updates,[v[key] for v in values],marker="o",color=color,label="REC" if target=="recovery" else "DYN")
                populated=True
            for row in range(2):
                axes[row,col].set(xlabel="Training updates",yscale="log",ylabel="Clean training RMS / $s$" if row==0 else "Objective-target training RMS / $s$")
        if populated:
            axes[0,0].legend()
            save(fig,output,"training_progress","Fixed-input evaluation during continued training: the same 1024 clean centers and displaced inputs are scored at each saved checkpoint. Bottom panels use each model's own training target. These are evaluation measurements of frozen checkpoints on training inputs, not averages of changing minibatch losses. Curves show available checkpoints; terminal selection is fixed in advance.",captions)
        else:
            plt.close(fig)
    if (root/"all_targets/result.json").exists():
        receipt=json.loads((root/"all_targets/result.json").read_text())
        if digest(root/"all_targets/metrics.npz") != receipt["metrics_sha256"]:
            raise ValueError("follow-up target metrics changed")
        with np.load(root/"all_targets/metrics.npz") as f:
            targets={k:f[k] for k in f.files}
        for role in ("train","validation"):
            mask=np.array(receipt["populations"])==role
            fig,axes=plt.subplots(2,3,figsize=(7.1,4.8),layout="constrained")
            controls=(("Training duration","native_mixed","native_mixed","_u8192"),
                      ("Early coverage","native_one","early_one","_u4096"),
                      ("Gaussian amplitude","gaussian","gaussian_profile","_u4096"))
            for col,(title,old,new,suffix) in enumerate(controls):
                axes[0,col].set_title(title)
                for row,metric in enumerate(("rec_mse","dyn_mse")):
                    ax=axes[row,col]
                    for target,color in (("recovery",REC),("dynamics",DYN)):
                        label="REC" if target=="recovery" else "DYN"
                        for name,style,tag in ((old+"_"+target,"--","original"),(new+"_"+target+suffix,"-","follow-up")):
                            curve(ax,np.arange(32)*.05,targets[name+"__pretrained__"+metric],mask,label+": "+tag,color,style)
                    ax.set(yscale="log",xlabel="Input time",ylabel=(r"$E_{\rm rec}$" if row==0 else r"$E_{\rm dyn}$")+" (RMS / $s$)")
            axes[0,0].legend(fontsize=6.5)
            save(fig,output,"controlled_target_changes_"+role,
                 "Effects of the follow-up controls on identical saved pretrained-rollout inputs. Columns change training duration, input-time coverage, or the Gaussian amplitude profile. Blue evaluates REC and orange evaluates DYN. Dashed curves are the original model and solid curves the corresponding follow-up. Targets and input paths are unchanged within each comparison.",captions)
    # The gallery makes every figure usable outside the paper without hidden conventions.
    body = ["<!doctype html><html><head><meta charset='utf-8'><title>Kolmogorov corrective study</title><style>body{max-width:1050px;margin:32px auto;font:16px/1.55 system-ui}img{width:100%;height:auto}section{margin:3em 0}code{font-size:.9em}</style></head><body><h1>Corrective mechanisms: Section 4.2 figures</h1>",
            "<p>Errors use RMS divided by the fixed training scale. Local errors are plotted at input time. Rollouts are autonomous. Figures appear only when all data required for that comparison are complete.</p>"]
    body.append("<p>Start with the autonomous diagnostics and target-specialization figures below. The remaining figures document individual controls. "
                "<a href='summary.csv'>Per-trajectory measurements</a> · <a href='comparisons.csv'>Paired rollout comparisons</a> · "
                "<a href='target_summary.csv'>Common-input target measurements</a></p>")
    lead = ("autonomous_diagnostics_validation", "target_specialization_and_transfer", "clean_and_rollout_validation",
            "mechanisms_cross_native_mixed_validation", "controlled_target_changes_validation")
    captions = {k: captions[k] for k in list(lead)+list(captions) if k in captions}
    tex = []
    for stem, caption in captions.items():
        body.append(f"<section><h2>{html.escape(stem.replace('_',' '))}</h2><a href='{stem}.pdf'>Vector PDF</a><img src='{stem}.png'><p>{html.escape(caption)}</p></section>")
        tex += [f"% {stem}.pdf", caption.replace("%",r"\%").replace("_",r"\_"), ""]
    body.append("</body></html>")
    (output/"index.html").write_text("\n".join(body),encoding="utf-8")
    (output/"captions.tex").write_text("\n".join(tex),encoding="utf-8")
    (output/"manifest.json").write_text(json.dumps(dict(renderer_sha256=digest(__file__),figures={k:dict(caption=v,pdf_sha256=digest(output/(k+".pdf")),png_sha256=digest(output/(k+".png"))) for k,v in captions.items()}),indent=2),encoding="utf-8")


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    a=p.parse_args(); plot(a.input,a.output)
