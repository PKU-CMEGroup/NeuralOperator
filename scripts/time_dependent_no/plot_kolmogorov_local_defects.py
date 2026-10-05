"""Plot clean and reached-state flow defects from the first Kolmogorov study.

Usage: python scripts/time_dependent_no/plot_kolmogorov_local_defects.py \
    --input <evaluation_directory> --output <figure_directory>

The input contains result.json and metrics.npz. MSE arrays are already divided
by the fixed clean-training scale squared. This script pools cases at each
time before taking a square root; it never averages over time or drops cases.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


MODEL_NAMES = ("pretrained", "clean_continuation")
COLORS = ("#0072B2", "#D55E00")
LINESTYLES = ("-", "--")
PATH_LABELS = ("Ground-truth trajectory", "Pretrained rollout", "Clean-continuation rollout")
PATH_COLORS = ("#666666", *COLORS)
PATH_LINESTYLES = (":", *LINESTYLES)
POPULATIONS = ("train", "validation")


def load_metrics(input_dir: Path) -> tuple[dict, dict[str, np.ndarray]]:
    metadata = json.loads((input_dir / "result.json").read_text(encoding="utf-8"))
    with np.load(input_dir / "metrics.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    names = arrays["model_names"].tolist()
    if names not in [list(MODEL_NAMES), [*MODEL_NAMES, "probe"]]:
        raise ValueError("model_names must be pretrained, clean_continuation, then optional probe")
    cases = arrays["case_ids"]
    populations = arrays["populations"]
    steps = arrays["input_steps"]
    if cases.ndim != 1 or len(cases) == 0 or populations.shape != cases.shape:
        raise ValueError("case_ids and populations must be aligned nonempty vectors")
    if not set(populations.tolist()).issubset(POPULATIONS):
        raise ValueError("populations must contain only train or validation")
    if steps.ndim != 1 or len(steps) == 0 or not np.array_equal(steps, np.arange(len(steps))):
        raise ValueError("input_steps must contain consecutive steps starting at zero")
    expected_shapes = {
        "clean_mse": (len(names), len(cases), len(steps)),
        "local_mse": (len(names), len(cases), 2, len(steps)),
        "rollout_mse": (2, len(cases), len(steps) + 1),
    }
    for key, shape in expected_shapes.items():
        values = arrays[key]
        if values.shape != shape:
            raise ValueError(f"{key}: expected shape {shape}, got {values.shape}")
        if np.isinf(values).any() or (values < 0).any():
            raise ValueError(f"{key} must contain nonnegative MSE values or explicit missing values")
    for key in ("macro_dt", "train_scale"):
        if not np.isfinite(float(metadata[key])) or float(metadata[key]) <= 0:
            raise ValueError(f"{key} must be finite and positive")
    return metadata, arrays


def pooled_percent(mse: np.ndarray) -> np.ndarray:
    """Spatial MSE is already pooled; pool cases before taking the RMS."""
    return 100.0 * np.sqrt(np.mean(mse, axis=0))


def _draw_error(axis, times, mse, *, color, linestyle, label):
    if len(mse) > 1:
        for values in mse:
            axis.plot(times, 100.0 * np.sqrt(values), color=color,
                      linestyle=linestyle, linewidth=0.6, alpha=0.22)
    axis.plot(times, pooled_percent(mse), color=color, linestyle=linestyle,
              linewidth=1.9, label=label)


def _format_axis(axis, *, end_time: float, logarithmic: bool):
    axis.set_xlim(0.0, end_time)
    if logarithmic:
        axis.set_yscale("log")
    else:
        axis.set_ylim(bottom=0.0)
    axis.grid(axis="y", which="major", alpha=0.22, linewidth=0.6)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(labelsize=8)


def _legend(fig, *, include_ground_truth: bool, y: float):
    start = 0 if include_ground_truth else 1
    handles = [Line2D([0], [0], color=color, linestyle=style, linewidth=1.9,
                      label=label)
               for label, color, style in zip(PATH_LABELS[start:], PATH_COLORS[start:],
                                              PATH_LINESTYLES[start:])]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, y),
               ncol=len(handles), frameon=False, title="Input trajectory" if include_ground_truth else None)


def _footer(metadata, arrays) -> str:
    counts = [int(np.count_nonzero(arrays["populations"] == name)) for name in POPULATIONS]
    stage = "Pilot" if metadata.get("pilot", False) else "Evaluation"
    counts_text = f"{counts[0]} training / {counts[1]} validation initial conditions"
    aggregation = ("Thick curves pool RMS over cases; thin curves show individual cases."
                   if max(counts) > 1 else "One initial condition per population; no averaging over time.")
    missing = any(np.isnan(arrays[key]).any() for key in ("clean_mse", "local_mse", "rollout_mse"))
    missing_text = " Missing measurements remain gaps; cases are not dropped." if missing else ""
    return f"{stage}: {counts_text}.\n{aggregation}{missing_text}"


def _save(fig, output_dir: Path, stem: str) -> list[str]:
    paths = []
    for suffix in ("pdf", "png"):
        path = output_dir / f"{stem}.{suffix}"
        fig.savefig(path, dpi=300, facecolor="white")
        paths.append(str(path))
    plt.close(fig)
    return paths


def plot_metrics(input_dir: Path, output_dir: Path) -> list[str]:
    metadata, arrays = load_metrics(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 11,
        "axes.labelsize": 10, "legend.fontsize": 9, "pdf.fonttype": 42,
        "ps.fonttype": 42, "text.usetex": False,
    })
    dt = float(metadata["macro_dt"])
    input_times = arrays["input_steps"] * dt
    rollout_times = np.arange(len(input_times) + 1) * dt
    end_time = float(rollout_times[-1])
    has_probe = "probe" in arrays["model_names"].tolist()
    columns = 3 if has_probe else 2
    predictor_titles = ("Pretrained predictor", "Clean-continuation predictor",
                        "Independent clean probe")

    fig, axes = plt.subplots(2, columns, figsize=(4.6 * columns, 7.7), squeeze=False)
    fig.subplots_adjust(left=0.10 if columns == 2 else 0.07, right=0.985,
                        bottom=0.25, top=0.72, hspace=0.45, wspace=0.18)
    fig.suptitle("One-step prediction error along three trajectories", y=0.98, fontsize=14)
    fig.text(0.5, 0.925,
             "Take a trajectory state  \u2192  apply predictor and solver to that state  \u2192  compare their next states",
             ha="center", fontsize=9)
    _legend(fig, include_ground_truth=True, y=0.885)
    for row, population in enumerate(POPULATIONS):
        mask = arrays["populations"] == population
        count = int(mask.sum())
        name = "Training" if population == "train" else "Validation"
        axes[row, 0].set_ylabel(f"{name}\nOne-step error (%)")
        positive_values = []
        for model in range(columns):
            axis = axes[row, model]
            if row == 0:
                axis.set_title(predictor_titles[model], pad=13)
            axis.set_xlabel("Time of input state")
            if not count:
                axis.text(0.5, 0.5, "No cases evaluated", transform=axis.transAxes,
                          ha="center", va="center")
                continue
            for source in range(3):
                mse = (arrays["clean_mse"][model, mask] if source == 0 else
                       arrays["local_mse"][model, mask, source - 1])
                _draw_error(axis, input_times, mse, color=PATH_COLORS[source],
                            linestyle=PATH_LINESTYLES[source], label=PATH_LABELS[source])
                values = 100.0 * np.sqrt(mse)
                positive_values.extend(values[np.isfinite(values) & (values > 0)].tolist())
            _format_axis(axis, end_time=end_time, logarithmic=True)
        if positive_values:
            lower, upper = min(positive_values), max(positive_values)
            padding = max(0.08, 0.06 * np.log10(upper / lower))
            for axis in axes[row]:
                axis.set_ylim(lower / 10**padding, upper * 10**padding)
    fig.text(0.5, 0.177,
             "Each panel fixes the evaluating predictor; the three curves change only the input trajectory.",
             ha="center", fontsize=9)
    fig.text(0.5, 0.135,
             r"At state $z_n$, error = $100\,\|M(z_n)-\Phi(z_n)\|_{\rm RMS}/s$; "
             "s is the fixed training scale.",
             ha="center", fontsize=9)
    fig.text(0.5, 0.080, _footer(metadata, arrays),
             ha="center", va="center", fontsize=8, linespacing=1.6)
    fig.text(0.5, 0.025,
             r"Errors are placed at input time $n\Delta t$; predictions target $(n+1)\Delta t$. "
             "Vertical scales agree within each row.",
             ha="center", fontsize=8)
    paths = _save(fig, output_dir, "trajectory_one_step_errors")

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.6), squeeze=False)
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.25, top=0.72, wspace=0.25)
    fig.suptitle("Accumulated rollout error against the ground-truth trajectory",
                 y=0.98, fontsize=13)
    _legend(fig, include_ground_truth=False, y=0.915)
    maximum = 0.0
    for column, population in enumerate(POPULATIONS):
        axis = axes[0, column]
        mask = arrays["populations"] == population
        count = int(mask.sum())
        name = "Training" if population == "train" else "Validation"
        cases_label = "initial condition" if count == 1 else "initial conditions"
        axis.set_title(f"{name}: {count} {cases_label}", pad=11)
        axis.set_xlabel("Time")
        if column == 0:
            axis.set_ylabel("Rollout error (% of training scale)")
        if not count:
            axis.text(0.5, 0.5, "No cases evaluated", transform=axis.transAxes,
                      ha="center", va="center")
            continue
        for source in range(2):
            mse = arrays["rollout_mse"][source, mask]
            _draw_error(axis, rollout_times, mse, color=COLORS[source],
                        linestyle=LINESTYLES[source], label=PATH_LABELS[source + 1])
            values = 100.0 * np.sqrt(mse)
            finite = values[np.isfinite(values)]
            if len(finite):
                maximum = max(maximum, float(finite.max()))
        _format_axis(axis, end_time=end_time, logarithmic=False)
    if maximum > 0:
        for axis in axes[0]:
            axis.set_ylim(0.0, 1.05 * maximum)
    fig.text(0.5, 0.13,
             r"At time $n\Delta t$, compare the autonomous rollout state $x_n$ with the ground-truth state $u_n$.",
             ha="center", fontsize=9)
    fig.text(0.5, 0.045, _footer(metadata, arrays),
             ha="center", va="center", fontsize=8, linespacing=1.6)
    paths.extend(_save(fig, output_dir, "trajectory_rollout_errors"))
    return paths


def animation_fields(truth, reached, solver_next, scale, predictions=None):
    """Signed fields in stored (x,y) order; no inference or numerical stepping."""
    truth, reached, solver_next = [np.asarray(a, dtype=np.float64)
                                  for a in (truth, reached, solver_next)]
    horizon, n = len(truth) - 1, truth.shape[-1]
    if (truth.shape != (horizon + 1, n, n)
            or reached.shape != (2, horizon + 1, n, n)
            or solver_next.shape != (2, horizon, n, n)
            or not np.isfinite(scale) or scale <= 0
            or not all(np.isfinite(a).all() for a in (truth, reached, solver_next))):
        raise ValueError("invalid animation states, solver targets, or normalization")
    error_a, error_b = (reached - truth[None]) / scale
    rollout_fields = [truth / scale, reached[0] / scale, reached[1] / scale,
                      error_a, error_b, (reached[1] - reached[0]) / scale]
    own = (reached[:, 1:] - solver_next) / scale
    views = {"rollout": rollout_fields, "own": [own[0], own[1], own[1] - own[0]]}
    if predictions is not None:
        predictions = np.asarray(predictions, dtype=np.float64)
        if predictions.shape != (2, 3, horizon, n, n) or not np.isfinite(predictions).all():
            raise ValueError("predictions must have axes [model,source,time,x,y]")
        targets = np.stack([truth[1:], *solver_next])
        for source, name in enumerate(("truth", "pretrained", "continued")):
            errors = (predictions[:, source] - targets[source]) / scale
            views[name] = [errors[0], errors[1], errors[1] - errors[0]]
    return views


def display_xy(field):
    """Rows on screen run from high y to low y; columns run from low x to high x."""
    return np.asarray(field).T[::-1]


def _field_png(field, limit):
    from PIL import Image

    values = display_xy(field)
    if np.max(np.abs(values)) > limit * (1 + 1e-12):
        raise ValueError("color limits would clip the field")
    normalized = (values / limit + 1.0) / 2.0 if limit > 0 else np.full_like(values, 0.5)
    indices = np.minimum((normalized * 256).astype(np.uint16), 255).astype(np.uint8)
    palette = matplotlib.colormaps["RdBu_r"](np.arange(256), bytes=True)[:, :3]
    image = Image.fromarray(indices)
    image.putpalette(palette.ravel().tolist())
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


ANIMATION_HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Kolmogorov trajectories and one-step errors</title><style>
*{box-sizing:border-box}body{margin:0;background:#f5f6f8;color:#202631;font:15px system-ui,sans-serif}
main{max-width:1320px;margin:auto;padding:20px}h1{font-size:25px;margin:0 0 6px}p{line-height:1.45;margin:8px 0}
.controls{position:sticky;top:0;z-index:2;background:#f5f6f8f2;padding:12px 0;border-bottom:1px solid #d7dce2}
.row{display:flex;flex-wrap:wrap;align-items:center;gap:10px;margin:8px 0}label{font-size:13px}
select,button{font:inherit;padding:6px 10px;border:1px solid #bbc3ce;border-radius:5px;background:white;color:inherit}
button{cursor:pointer}#play{min-width:84px;background:#1e567a;color:white;border-color:#1e567a}
input[type=range]{flex:1;min-width:220px}#time{font-variant-numeric:tabular-nums;font-weight:600}
.note{color:#475669;font-size:13px}#scaleNote{font-weight:600}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin-top:14px}
.panel{background:white;border:1px solid #dce1e7;border-radius:7px;padding:10px}h2{font-size:15px;margin:0 0 3px;font-weight:600}
canvas{width:100%;height:auto;display:block}.numbers{font-size:12px;color:#526171;font-variant-numeric:tabular-nums}
footer{font-size:12px;color:#526171;padding:14px 0}details{margin-top:8px}summary{cursor:pointer}
@media(max-width:800px){.grid{grid-template-columns:repeat(2,minmax(0,1fr))}main{padding:12px}}
@media(max-width:500px){.grid{grid-template-columns:1fr}}
</style></head><body><main>
<h1>Kolmogorov trajectories and one-step errors</h1>
<p>Follow the ground truth and both autonomous rollouts, then inspect their signed one-step errors.</p>
<div class="controls"><div class="row">
<label>Initial condition <select id="case"></select></label>
<label>View <select id="view"><option value="rollout">Rollout fields and accumulated errors</option><option value="defect">One-step defects</option></select></label>
<label id="sourceLabel">Input states <select id="source"><option value="own">Each model's own rollout</option></select></label>
<label>Colors <select id="scaling"><option value="fixed">Fixed over time</option><option value="adaptive">Rescale each frame</option></select></label>
</div><div class="row"><button id="play">Play</button><button id="previous" aria-label="Previous frame">&#9664;</button>
<input id="step" type="range" min="0" value="0" step="1" aria-label="Time step"><button id="next" aria-label="Next frame">&#9654;</button>
<label>Speed <select id="speed"><option value="2">2 fps</option><option value="4" selected>4 fps</option><option value="8">8 fps</option></select></label>
</div><div id="time"></div><p id="meaning"></p><div class="note" id="scaleNote"></div></div>
<div id="panels" class="grid"></div><footer>
<div id="scope"></div><p>All fields and RMS values are divided by the fixed training scale s; these are not percentages.
Blue is negative, red is positive. The B&#8722;A panel has its own labeled color scale.</p>
<p>Coordinates: x runs left to right and y runs bottom to top on the periodic square [0, 2&#960;)&#178;.
RMS values use the original numerical fields. PNG color maps use the full spatial grid; no values are clipped.</p>
<details><summary>What the views measure</summary><p>The rollout view compares predicted and ground-truth states at the same time.
In the defect view, each predicted next state is compared with the numerical true-flow next state from the same input.
On the ground-truth trajectory, the target is the archived ground-truth next state. In own-rollout mode, A and B start from different reached inputs.
In common-input modes, the input and target are identical for both predictors. These are empirical prediction errors, not exact manifold distances.</p></details>
</footer></main><script id="data" type="application/json">__DATA__</script><script>
const data=JSON.parse(document.getElementById('data').textContent);
const $=id=>document.getElementById(id), cases=$('case'), view=$('view'), source=$('source'), scaling=$('scaling'), slider=$('step');
const names={rollout:['Ground truth','A: pretrained rollout','B: continued rollout','A minus ground truth','B minus ground truth','B minus A'],
defect:['A: signed one-step defect','B: signed one-step defect','Defect difference: B minus A']};
for(const [i,c] of data.cases.entries()){const option=document.createElement('option');option.value=i;option.textContent=c.label;cases.append(option)}
if(data.common){for(const [value,label] of [['truth','Ground-truth trajectory'],['pretrained','Pretrained rollout (same input)'],['continued','Continuation rollout (same input)']]){
const option=document.createElement('option');option.value=value;option.textContent=label;source.append(option)}}
$('scope').textContent=`${data.pilot?'Pilot':'Evaluation'}: ${data.cases.length} initial conditions. Fixed training scale s = ${data.train_scale.toPrecision(7)}. No new rollouts are generated by this viewer.`;
let playing=false,timer=null,ticket=0,panelCount=0;const images=new Map();
const format=x=>Math.abs(x)>0&&Math.abs(x)<.001?x.toExponential(2):Number(x.toPrecision(4)).toString();
function pause(){playing=false;clearTimeout(timer);$('play').textContent='Play'}
function schedule(){clearTimeout(timer);if(playing)timer=setTimeout(()=>{slider.value=(Number(slider.value)+1)%(Number(slider.max)+1);render();schedule()},1000/Number($('speed').value))}
function imageFor(src){if(!images.has(src)){const image=new Image();const pending=new Promise(resolve=>{image.onload=()=>resolve(image)});image.src=src;images.set(src,pending)}return images.get(src)}
async function paint(canvas,src,limit,revision){const image=await imageFor(src);if(revision!==ticket)return;
const ctx=canvas.getContext('2d'),x=34,y=8,size=256;ctx.clearRect(0,0,330,324);ctx.imageSmoothingEnabled=false;ctx.drawImage(image,x,y,size,size);
ctx.strokeStyle='#66717f';ctx.lineWidth=.5;ctx.strokeRect(x,y,size,size);ctx.font='11px system-ui';ctx.fillStyle='#475669';ctx.textAlign='right';
ctx.fillText('2π',x-5,y+8);ctx.fillText('0',x-5,y+size+3);ctx.fillText('y',x-16,y+size/2);ctx.textAlign='center';
ctx.fillText('0',x,y+size+16);ctx.fillText('x',x+size/2,y+size+16);ctx.fillText('2π',x+size,y+size+16);
const gradient=ctx.createLinearGradient(x,0,x+size,0);data.palette.forEach((rgb,i)=>gradient.addColorStop(i/(data.palette.length-1),rgb));
ctx.fillStyle=gradient;ctx.fillRect(x,290,size,10);ctx.fillStyle='#475669';ctx.textAlign='left';ctx.fillText('-'+format(limit),x,315);
ctx.textAlign='center';ctx.fillText('0',x+size/2,315);ctx.textAlign='right';ctx.fillText('+'+format(limit),x+size,315)}
function render(){ticket++;const revision=ticket,rollout=view.value==='rollout',key=rollout?'rollout':source.value,c=data.cases[Number(cases.value)],frames=c.views[key];
$('sourceLabel').style.display=rollout?'none':'';slider.max=frames.length-1;if(Number(slider.value)>Number(slider.max))slider.value=slider.max;
const n=Number(slider.value),frame=frames[n],mode=scaling.value,limit=frame[mode+'_limits'];
$('time').textContent=rollout?`State step ${n} / ${frames.length-1} · time ${(n*data.dt).toFixed(2)}`:
`Step ${n} → ${n+1} · input time ${(n*data.dt).toFixed(2)} · next-state time ${((n+1)*data.dt).toFixed(2)}`;
$('meaning').textContent=rollout?'Top: synchronized vorticity states. Bottom: accumulated trajectory errors and the difference between the two rollouts.':
key==='own'?"Each predictor uses its own rollout input. Inputs differ: each defect is its saved next prediction minus the solver's next state from that input.":
`Both predictors use the same ${key==='truth'?'ground-truth':key==='pretrained'?'pretrained-rollout':'continued-rollout'} input. The target is also identical for A and B.`;
$('scaleNote').textContent=mode==='fixed'?'For this view, color limits are fixed across all times and initial conditions; A and B share a scale.':'Color limits CHANGE each frame; A and B still share a scale within the frame.';
if(panelCount!==frame.images.length){$('panels').replaceChildren();panelCount=frame.images.length;for(let i=0;i<panelCount;i++){
const panel=document.createElement('section');panel.className='panel';panel.innerHTML='<h2></h2><div class="numbers"></div><canvas width="330" height="324"></canvas>';$('panels').append(panel)}}
[...$('panels').children].forEach((panel,i)=>{panel.querySelector('h2').textContent=names[rollout?'rollout':'defect'][i];
panel.querySelector('.numbers').textContent=`RMS / s: ${format(frame.rms[i])} · field / s range: ±${format(limit[i])}`;
paint(panel.querySelector('canvas'),frame.images[i][mode],limit[i],revision)})}
$('play').onclick=()=>{if(playing)pause();else{playing=true;$('play').textContent='Pause';schedule()}};
$('previous').onclick=()=>{pause();slider.value=Math.max(0,Number(slider.value)-1);render()};
$('next').onclick=()=>{pause();slider.value=Math.min(Number(slider.max),Number(slider.value)+1);render()};
slider.oninput=()=>{pause();render()};[cases,view,source,scaling].forEach(el=>el.onchange=render);$('speed').onchange=schedule;render();
</script></body></html>'''


def _animation_mp4(case, output_dir, dt):
    from PIL import Image
    from matplotlib.animation import FFMpegWriter
    from matplotlib.colors import Normalize

    if not FFMpegWriter.isAvailable():
        raise RuntimeError("FFmpeg is unavailable; render HTML without --mp4")
    paths = []
    for key, layout, figsize in (("rollout", (2, 3), (10.8, 7.4)),
                                 ("own", (1, 3), (10.8, 4.2))):
        frames = case["views"][key]
        titles = (["Ground truth", "A: pretrained", "B: continued", "A minus truth", "B minus truth", "B minus A"]
                  if key == "rollout" else ["A: own-rollout defect", "B: own-rollout defect", "Defect difference B minus A"])
        fig, axes = plt.subplots(*layout, figsize=figsize, layout="constrained", squeeze=False)
        artists = []
        for i, axis in enumerate(axes.flat):
            pixels = np.array(Image.open(io.BytesIO(base64.b64decode(frames[0]["images"][i]["fixed"].split(",", 1)[1]))).convert("RGBA"))
            artists.append(axis.imshow(pixels, extent=(0, 2*np.pi, 0, 2*np.pi), origin="upper", interpolation="nearest"))
            axis.set_xticks([0, np.pi, 2*np.pi], ["0", "π", "2π"])
            axis.set_yticks([0, np.pi, 2*np.pi], ["0", "π", "2π"])
            axis.set_xlabel("x"); axis.set_ylabel("y")
            limit = frames[0]["fixed_limits"][i]
            bar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=Normalize(-limit, limit), cmap="RdBu_r"),
                               ax=axis, orientation="horizontal", shrink=.85, pad=.12)
            bar.set_label("signed field / s", fontsize=8)
            bar.ax.tick_params(labelsize=7)
        path = output_dir / f"{case['id']}_{key}.mp4"
        writer = FFMpegWriter(fps=4, codec="libx264", bitrate=1800,
                             extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])
        with writer.saving(fig, str(path), dpi=100):
            for n, frame in enumerate(frames):
                for i, (axis, artist) in enumerate(zip(axes.flat, artists)):
                    pixels = np.array(Image.open(io.BytesIO(base64.b64decode(frame["images"][i]["fixed"].split(",", 1)[1]))).convert("RGBA"))
                    artist.set_data(pixels)
                    axis.set_title(f"{titles[i]}\nRMS / s = {frame['rms'][i]:.4g}", fontsize=9)
                time_text = f"state time {n*dt:.2f}" if key == "rollout" else f"input {n*dt:.2f} → next state {(n+1)*dt:.2f}; inputs differ"
                fig.suptitle(f"{case['label']} | {time_text}\nFixed color scales; B−A uses its own scale", fontsize=11)
                writer.grab_frame()
        plt.close(fig)
        paths.append(str(path))
    return paths


def render_animation(input_dir: Path, output_dir: Path, common_input_dir=None, *, mp4=False):
    metadata, metrics = load_metrics(input_dir)
    if metadata.get("status") != "completed":
        raise ValueError("animation requires a completed evaluation")
    scale = float(metadata["train_scale"])
    if _digest(input_dir / "metrics.npz") != metadata["metrics_sha256"]:
        raise ValueError("evaluation metrics hash mismatch")
    input_hashes = {"result.json": _digest(input_dir / "result.json"),
                    "metrics.npz": metadata["metrics_sha256"]}
    rms_checks = []
    common = None
    if common_input_dir is not None:
        common_input_dir = Path(common_input_dir)
        common = json.loads((common_input_dir / "result.json").read_text(encoding="utf-8"))
        if (common["status"] != "completed" or common["train_scale"] != scale
                or common["case_ids"] != metrics["case_ids"].tolist()
                or common["input_hashes"]["pilot_result"] != _digest(input_dir / "result.json")
                or not all(case["replay_pass"] for case in common["cases"])):
            raise ValueError("common predictions do not bind the evaluated pilot")
    cases, limits = [], {}
    groups = {"rollout": (0, 0, 0, 1, 1, 2), "defect": (0, 0, 1)}
    for case_index, (case_id, population) in enumerate(zip(metrics["case_ids"].tolist(), metrics["populations"].tolist())):
        input_hashes[f"{case_id}.npz"] = _digest(input_dir / f"{case_id}.npz")
        with np.load(input_dir / f"{case_id}.npz", allow_pickle=False) as saved:
            truth, reached, labels = [saved[key] for key in ("ground_truth", "reached", "solver_next")]
        predictions = None
        if common is not None:
            path = common_input_dir / f"{case_id}.npz"
            if (_digest(path) != common["artifacts"][path.name]
                    or _digest(input_dir / path.name) != common["input_hashes"][case_id]):
                raise ValueError("saved prediction or input-state hash mismatch")
            with np.load(path, allow_pickle=False) as saved:
                if (saved["model_names"].tolist() != list(MODEL_NAMES)
                        or saved["input_names"].tolist() != ["ground_truth", "pretrained_rollout", "clean_continuation_rollout"]
                        or not np.array_equal(saved["input_steps"], metrics["input_steps"])):
                    raise ValueError("saved prediction axes mismatch")
                predictions = saved["predictions"]
        views = animation_fields(truth, reached, labels, scale, predictions)
        own_rms = np.stack([np.sqrt(np.mean(field**2, axis=(-2, -1))) for field in views["own"][:2]])
        rollout_rms = np.stack([np.sqrt(np.mean(field**2, axis=(-2, -1))) for field in views["rollout"][3:5]])
        expected_own = np.sqrt(metrics["local_mse"][np.arange(2), case_index, np.arange(2)])
        expected_rollout = np.sqrt(metrics["rollout_mse"][:, case_index])
        if (not np.allclose(own_rms, expected_own, atol=1e-12, rtol=1e-12)
                or not np.allclose(rollout_rms, expected_rollout, atol=1e-12, rtol=1e-12)):
            raise ValueError("rendered fields do not reproduce saved RMS metrics")
        rms_checks.append(dict(case_id=case_id,
                               maximum_own_rms_difference=float(np.max(np.abs(own_rms - expected_own))),
                               maximum_rollout_rms_difference=float(np.max(np.abs(rollout_rms - expected_rollout)))))
        for key, fields in views.items():
            group = groups["rollout" if key == "rollout" else "defect"]
            fixed = limits.setdefault(key, np.zeros(max(group) + 1))
            for field, index in zip(fields, group):
                fixed[index] = max(fixed[index], float(np.max(np.abs(field))))
        cases.append(dict(id=case_id, label=f"{population.capitalize()}: {case_id}", views=views))
    for case in cases:
        for key, fields in list(case["views"].items()):
            group = groups["rollout" if key == "rollout" else "defect"]
            frames = []
            for n in range(len(fields[0])):
                current = [field[n] for field in fields]
                adaptive = [max(float(np.max(np.abs(field))) for field, g in zip(current, group) if g == index)
                            for index in range(max(group) + 1)]
                fixed_limits = [float(limits[key][index]) for index in group]
                adaptive_limits = [adaptive[index] for index in group]
                images = [{"fixed": _field_png(field, fixed), "adaptive": _field_png(field, changing)}
                          for field, fixed, changing in zip(current, fixed_limits, adaptive_limits)]
                frames.append(dict(images=images, rms=[float(np.sqrt(np.mean(field**2))) for field in current],
                                   fixed_limits=fixed_limits, adaptive_limits=adaptive_limits))
            case["views"][key] = frames
        print(f"encoded animation case={case['id']}", flush=True)
    palette = matplotlib.colormaps["RdBu_r"](np.linspace(0, 1, 65), bytes=True)
    payload = dict(cases=cases, train_scale=scale, dt=float(metadata["macro_dt"]),
                   pilot=bool(metadata.get("pilot")), common=common is not None,
                   input_metrics_sha256=_digest(input_dir / "metrics.npz"),
                   renderer_sha256=_digest(Path(__file__)),
                   palette=[f"rgb({r},{g},{b})" for r, g, b, _ in palette])
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "trajectory_viewer.html"
    encoded = json.dumps(payload, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    path.write_text(ANIMATION_HTML.replace("__DATA__", encoded), encoding="utf-8")
    paths = [str(path)]
    if mp4:
        for case in cases:
            paths.extend(_animation_mp4(case, output_dir, payload["dt"]))
    receipt = dict(status="completed", input_hashes=input_hashes,
                   common_prediction_manifest_sha256=_digest(common_input_dir / "result.json") if common else None,
                   renderer_sha256=payload["renderer_sha256"], rms_checks=rms_checks,
                   normalization="signed fields and RMS divided by fixed training scale; no percent multiplier",
                   orientation="stored [x,y]; displayed field.T[::-1] so x increases rightward and y upward",
                   time_semantics="rollout frame n is time n*dt; defect frame n is transition n to n+1",
                   color_scales="symmetric; fixed across all cases/time by default; optional per-frame shared scaling",
                   no_clipping=True, model_calls=0, solver_calls=0,
                   outputs={Path(name).name: _digest(Path(name)) for name in paths})
    receipt_path = output_dir / "animation_manifest.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, allow_nan=False), encoding="utf-8")
    paths.append(str(receipt_path))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Directory containing result.json and metrics.npz")
    parser.add_argument("--output", required=True, type=Path, help="Output directory for PDF and PNG figures")
    parser.add_argument("--animate", action="store_true", help="Render a self-contained HTML viewer from saved states")
    parser.add_argument("--common-input-dir", type=Path, help="Optional authenticated saved predictions for common-input defect views")
    parser.add_argument("--mp4", action="store_true", help="With --animate, also render rollout and own-defect MP4 previews")
    args = parser.parse_args()
    if not args.animate and (args.common_input_dir is not None or args.mp4):
        parser.error("--common-input-dir and --mp4 require --animate")
    paths = (render_animation(args.input, args.output, args.common_input_dir, mp4=args.mp4)
             if args.animate else plot_metrics(args.input, args.output))
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
