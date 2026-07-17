# Mechanistic Diagnostic Tracker

Date: 2026-07-04

This file is an evidence ledger. `TODO` or `BLOCKED` rows are not automatically
approved next work; current method-design precedence is
`RESEARCH_DIRECTION_DECISION.md`, then `HANDOFF.md`.

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D001 | M0 | Recompute current summaries from rollout arrays | CPGNet one-step bs2 | bump test full | paper RMSE, per-time RMSE, VPT | MUST | DONE | Full 20 trajectories analyzed on AutoDL under `artifacts/time_dependent_no/cpg_mechanistic_diagnostic_20260704_full/`; reproduced user-provided AR summary (`rho=0.200242`, `pres=0.384017`). |
| D002 | M0 | Recompute current summaries from rollout arrays | CPGNet two-stage bs2 | bump test full | paper RMSE, per-time RMSE, VPT | MUST | DONE | Full 20 trajectories analyzed on AutoDL; reproduced user-provided AR summary (`rho=0.158637`, `pres=0.293593`). |
| D003 | M1 | Teacher-forced vs autoregressive error by time | CPGNet one-step and two-stage | bump test full | TF RMSE, AR RMSE, AR/TF ratio | MUST | DONE | Per-time teacher-forced evaluator run on AutoDL. TF remains small over time; AR/TF ratios are ~22-26x for one-step and ~16-28x for two-stage depending on variable. |
| D004 | M1 | State-distribution drift | CPGNet one-step and two-stage | bump test/train stats | z-score drift, range violations | MUST | DONE | Sampled train stats from 30 train trajectories at stride 10. Final predictions almost never leave sampled train min/max ranges except tiny pressure fractions; gross range-OOD drift is not the main explanation. |
| D005 | M1 | Perturbation amplification | CPGNet one-step and two-stage | selected bump test | amplification factor, excess target RMSE | MUST | DONE | Sampled AutoDL probe over 5 trajectories, 4 start frames, two perturbation scales, and 5 channel modes. One-step is mostly locally damped; two-stage is more velocity-sensitive, but induced excess target RMSE is tiny versus full AR errors. |
| D006 | M2 | Shock position and region decomposition | CPGNet one-step and two-stage | bump test full | shock-front IoU/F1, Chamfer/front distance, centroid, near-shock/smooth error, thickness, strength | MUST | DONE | Full q=0.85/0.90/0.95 shock diagnostics run on AutoDL. Shock-local error dominates smooth error, but best-shift alignment explains only a small fraction, so failure is shock-local shape/amplitude/stability plus phase, not pure displacement. |
| D007 | M2 | Rollout animations and overlays | CPGNet one-step and two-stage | selected bump test | qualitative phase/smear/drift labels | MUST | DONE | Generated 32 selected-subset pressure/shock overlay PNGs plus JSON/Markdown gallery under `artifacts/time_dependent_no/cpg_shock_overlay_gallery_20260704/`. |
| D008 | M3 | Equal-node physics diagnostics | CPGNet one-step and two-stage | bump test full | conservation drift, positivity, TV proxy | MUST | DONE | Equal-node normal-node total mismatch, positivity, and clamped-boundary leakage computed for full split. No positivity failures; boundary error is exactly zero under clamped rollout. |
| D009 | M3 | Approximate-weight physics diagnostics | CPGNet one-step and two-stage | bump test | weighted conservation drift | NICE | BLOCKED | Requires mesh/cell weights or a validated geometric approximation; current diagnostics intentionally report equal-node totals only. |
| D010 | M4 | Direct state-predictor control | Direct GNN or simplest available comparable model | bump train/test | B1-B5 metrics | NICE | BLOCKED | Wait for dominant defect |
| D011 | M5 | PCNO diagnostic replay | PCNO completed checkpoint via corrected preprocessing contract | selected bump test | AR RMSE, shock masks, positivity, boundary leakage, animations | MUST | DONE | D019 completed for trajectories 0, 6, 11, 13, 17 using the corrected HDF5-to-npy-to-reconstructed-npz path. The retired raw-HDF5 graph adapter should not be used for model conclusions. |
| D012 | M6 | Correlation-time and geometric rollout aggregation | CPGNet one-step and two-stage | bump test full | high-correlation time, geometric relative error aggregation | MUST | TODO | Add APEBench/PDE-Refiner-style temporal metrics to existing raw-array diagnostic reports. |
| D013 | M6 | Scale/spectral residual diagnostics | 1D residual FNO, flux FNO, and CPGNet mp28 first; then 2D CPGNet/PCNO | frozen 1D split, then bump full/subset | divergence-active flux spectra, first/second differences, characteristic and shock/smooth splits, pre-failure high-frequency growth; graph-native bins in 2D | MUST | DONE (1D) | The frozen 1D run classifies the flux failure as recurrent high-frequency growth rather than deficient one-step spectral fit. The graph-native 2D extension remains separate future work; do not use interpolation-to-grid FFT as primary 2D evidence until interpolation error is audited. |
| D014 | M6 | Effective-CFL / receptive-field audit | CPGNet first, then PCNO | bump test full/subset | shock-front motion per step, median-edge-length units, message-passing/hop coverage, correlation with hard trajectories | MUST | TODO | Tests whether hard trajectories exceed stable information-propagation capacity. |
| D015 | M7 | Recurrent/unrolled stabilization control | CPGNet or PCNO after D013/D014 | bump train/test | TF error, AR error, VPT, shock metrics, scale residuals | NICE | BLOCKED | Do not launch until scale/resolution and effective-CFL diagnostics identify which mechanism to target. |
| D016 | M6 | Interface-state latent instrumentation | CPGNet one-step and two-stage bs2 | selected bump test frames | `reconstruct_prims`, one-sided trace metrics, LLF central/dissipation split, induced FV update, wave-type strata | MUST | DONE | Historical probe completed. The one-off implementation was retired from the active tree after the result was recorded; recover from git history only if exact reproduction is needed. |
| D017 | M6 | Interface-state latent selected-frame run | CPGNet one-step and two-stage bs2 | trajectories 0, 6, 11, 13, 17; frames 0, 20, 40, 58, 78 | admissibility, trace-likeness, flux/update match, dissipation localization, speed projection, sampled edge table | MUST | DONE | AutoDL run completed under `artifacts/time_dependent_no/cpg_interface_latent_diagnostic_20260705_full/`; latents are admissible but not physical one-sided traces, induced update matches model delta but not true update exactly, and dissipation is only weakly shock-localized. |
| D018 | M6 | Interface-latent mechanism probe | CPGNet one-step and two-stage bs2 | selected trajectories/frames; teacher-forced and autoregressive state sources | physical projection sensitivity, constrained inverse flux fit, TF-vs-AR latent drift | MUST | DONE | AutoDL run completed under `artifacts/time_dependent_no/cpg_interface_mechanism_probe_20260705_full/`; physical projections do not preserve learned flux/update, constrained physical inverse fits remain poor, and AR mode greatly increases learned-update error against the target next state. |
| D019 | M5 | PCNO corrected preprocessed rollout replay | PCNO Euler checkpoint through collaborator-compatible preprocessing | trajectories 0, 6, 11, 13, 17 | AR RMSE, positivity, velocity blow-up, GIF gallery | MUST | DONE | AutoDL selected replay completed under ignored corrected PCNO rollout artifacts. Visual readout: PCNO initially tracks shock position better than CPGNet, but Fourier-style ripples grow and can trigger long-rollout crash; pressure mean RMSE across selected trajectories is about 2.13 and velocity errors can overflow. |
| D020 | M6 | 1D Euler effective-CFL / receptive-field intervention | corrected CPGNet h128, mp12 versus mp28 | 384/64/64, stride 4, frame 80 | one-step fit, raw completion, survival, CFL correlations, shock, conservation | MUST | DONE | mp28 completed 64/64 raw test rollouts versus 34/64 for mp12; first-rollout-step error fell 88.6% and the initial-CFL/error Pearson correlation fell from 0.90 to -0.12. Depth and parameter count changed together, so the claim remains partial pending matched-parameter controls. |
| D021 | Idea 2.1 | Staged target-family optimization and solver screen | coordinate-selected FNO: next state, residual, state-loss-only flux, direct cumulative impulse, and joint supervision | frozen 1D ADER split, then stride ladder | label/endpoint/boundary closure, tiny-set fit, training floor, seed variance, supervised-objective/decoded error, divergence-active impulse error for state-loss-only flux, shock/smooth curves, direct horizon, raw rollout, physics, transfer | MUST | IN PROGRESS | Exact face-MSE stopped at midscale and the recurrent flux head falls from 57/64 completion at 20 calls to 1/64 at 50. Direct next state failed tiny fit on two seeds. Projection improves pooled 100-call completion from 182/192 to 187/192 but not seed-stable accuracy. Generated burn-in improves H50/H100 rollout but D023 finds no material same-state PDE-map gain; stride and resolution gates remain. |
| D022 | Idea 2.1 | Separate later-time sampling from generated-state exposure | plain-residual FNO: clean 0+4, teacher-offset 8+4, generated 8+4 | fixed 64/16/16 split, seed 20260708 | matched one-step history/update count, H20/H50/H100 raw rollout, common-endpoint error, conservation, top-two front position/strength | MUST | DONE | Teacher offset gives no H50/H100 state benefit over clean. Generated exposure beats teacher by 13.9%, 33.9%, and 50.8% at H20/H50/H100 and completes 16/16 versus 14/16 at H100. It still regresses top-two front position versus clean at H20/H50, so full-scale promotion remains paused. |
| D023 | Idea 2.1 | Solver-consistency diagnostic from generated states | learned residual versus WENO-HLLC-ADER advancement initialized from the same generated state | all 16 D022 test cases, starts 0:10:80, prefix depths 0/2/4/8 | learned-vs-reference next-state defect, truth-next defect, shock/smooth and characteristic components, error-versus-prefix depth | MUST | DONE | At depth eight, generated/teacher error is 0.992 to original truth but 1.017 to the same-state solver continuation; correction alignment is only 0.074. Clean is closest to the solver. The result is mixed and rejects a large local PDE-map explanation for the rollout gain. |
| D024 | M6 | Frozen-checkpoint conservative-dissipation probe | state-loss-only flux FNO plus small local interior diffusive face flux | full frozen split | H20/H50 survival and positivity, front position/strength, shock width, TV excess, conservation and boundary exchange | MUST | DONE | The paired five-coefficient probe gives no material H50 stability gain. Small diffusion leaves completion unchanged while worsening state/tail error; larger diffusion shortens survival even when modes 25--64 decrease. Boundary correction is exactly zero. |
| D025 | Idea 2.1 | Global implicit-interface FNO pilot | face-grid FNO with two relative directed traces, shared Rusanov or central decoder, exact FV update | tiny fit then 64/16/16 | one-step fit, raw H20/H50 rollout, positivity, conservation, top-two front geometry, decoder ablation | MUST | DONE (1D pilot) | The 317,126-parameter Rusanov model reaches selected test one-step relative L2 `0.00786` after frame-zero weighting and a training-only barrier, but every 16-case H50 rollout still becomes inadmissible within five calls. Short unrolling, temporal reweighting, and barrier weight `0.1` fail the `0.10` survival gate. Do not promote this parameterization to full scale or four-step training. |
| D026 | Idea 2.1 | Identifiable boundary-exchange supervision | projected-residual FNO plus RMS-normalized net solver boundary-impulse loss | matched 64/16/16 stride-1 gate, H20 | one-step state, boundary exchange, raw rollout, shock, conservation, closure | MUST | DONE | Weights `0.1` and `0.01` improve boundary-exchange and conserved-total errors but worsen one-step, H20 state, and shock errors. Both retain 16/16 completion but fail the joint-accuracy gate. Stop without a full seed sweep. |
| D027 | Idea 2.1 | Cold stride-2 transfer gate | plain-residual FNO, fixed stride 2, compared with composed frozen stride-1 model | matched 64/16/16, H20 selection then frame-100 replay | native fit, same-frame H20/H50/H100 state, survival, shock, conservation | MUST | DONE (partial) | Direct stride 2 improves common-case H20/H50/H100 state error by 25.0%/36.2%/48.6% and completes 16/16 at H100, but its direct frame-2 error is 1.216 times stride-1 composition and misses the 1.15 gate. Activate the continuation control before stride 4. |
| D028 | Idea 2.1 | Stride-1 to stride-2 continuation gate | plain-residual FNO initialized from the frozen stride-1 weights, fresh stride-2 optimizer | matched D027 split/schedule, H20 selection then frame-100 replay | final-target fit floor, frame-2 defect, H20/H50/H100 state, survival, shock, conservation | MUST | DONE (partial) | Continuation repairs frame 2, lowers one-step/recurrent training floors by 42%/44%, and improves H100 state and pressure margin, but H50 state is 1.095 times cold and fails the 1.05 gate. Do not run the conditional total-exposure control or claim a uniformly better solver. |
| D029 | Idea 2.1 | Frozen cross-resolution transfer gate | frozen plain-residual stride-1 and cold stride-2 FNOs trained at 256 cells | identical 512 physical cases at 128/256/512 cells; frozen 16-case test split | native-grid one-step, frame-2/H20/H50/H100 raw rollout, shock, conservation, completion, solver restriction gap | MUST | DONE (partial) | Neither checkpoint passes native-map resolution transfer: off-grid one-step error is 5.5--8.3 times nx256, high-resolution shock metrics regress, and cold stride 2 loses one case off-grid. The larger-step advantage itself transfers bidirectionally: stride 2 beats stride-1 composition at H20/H50/H100 on nx128 and nx512 with equal same-grid completion. |
| D030 | Idea 2.1 | Restriction-consistent shared-resolution gate | one shared 64/24/4 residual FNO versus equal-presentation single-resolution oracles and the frozen native-nx256 baseline | exact-cell-average nx512 reference conservatively restricted to nx256/nx128; matched 64/16/16 split | label commutation, per-grid one-step, H20/H50/H100 raw rollout, shock, conservation, completion, native-solver diagnostic | MUST | DONE (partial) | The primary representation and frozen-baseline usefulness gates pass. The shared row stays within `1.418x` same-grid-oracle state error and `1.296x` one-step error, with no completion loss. It does not reproduce independently evolved native coarse-grid maps and still loses three pressure-limited cases by H100. Classify as `shared_restriction_operator_without_native_solver_equivalence`. |

## 2026-07-15 Integrated Solver-Flux Result

The validated ADER artifact contains 512 trajectories, 101 saved frames, 256
cells, and 257 owner-oriented faces. Its accepted-substep impulses close the
state transition before serialization to about `7.1e-14`; float32 decoded
closure is at most about `2.90e-5`. The state arrays match the original dataset
bitwise and no fallback was used. This rules out a mislabeled-flux or endpoint
closure failure for the tested rows.

The original tiny absolute gate was too strict for the shared FNO: even
state-only reached minimum decoded train relative L2 about `0.0060`, not the
declared `0.002`. The failed gate was preserved. A repeat seed reproduced the
ordering, a 100-epoch `3e-4` control improved all native losses, and the matched
raw/gauge-canonical by joint-weight controls identified gauge-canonical joint
weight `0.1` as the only solver-flux row that kept tiny-set state fit within
`1.10x` while improving identifiable flux fit.

That candidate did not survive the 64/16/16 successive-halving gate. Against
the state-loss-only flux head, its held-out active-flux MSE improved from
`1.0006e-3` to `3.0068e-4`, but one-step relative L2 worsened from `0.004612` to
`0.006226` and mean raw survival from `0.51875` to `0.300`. Every one of the 16
paired test trajectories failed earlier under joint supervision. State-only
survived 10.375 saved steps on average; joint survived 6.0. All terminations
were raw nonpositive states, with pressure responsible in 15/16 joint cases.

Final-prefix relative, shock, and conserved-total errors are not directly
comparable because the joint prefixes are shorter. At each common valid saved
step, joint relative error was already higher; at each joint trajectory's last
valid step, its error exceeded state-only for all 16 cases. The failure is
therefore classified as a decoded-state/generalization tradeoff plus recurrent
stability failure, not label invalidity or inability to optimize the native
flux target.

In this 1D chain, projecting out the one constant-flux null mode makes face flux
uniquely recoverable from its divergence. Gauge-canonical face-flux MSE is thus
an inverse-divergence reweighting of the state increment. It emphasizes global,
low-frequency residual modes and can reduce active face error while weakening
the local/high-frequency update accuracy required at shocks. Do not promote
this exact loss to the full split. The next bounded diagnostic is a good
state-loss-only flux model followed by short unrolled training, with a smooth
training-only admissibility barrier as a matched second-stage intervention.

## 2026-07-15 Short-Unroll Stability Result

The 64/16/16 stability screen used the same 316,739-parameter FNO and identical
50-epoch one-step histories for both rows. Each row then used ten epochs of a
four-step differentiable conservative rollout, a fresh AdamW optimizer at
`3e-5`, and rollout-based validation checkpoint selection. No limiter, floor,
or positive transform was used in training recurrence or inference.

Unroll-only passed the predeclared gate. Relative to the reproduced state-only
checkpoint, test one-step relative L2 changed from `0.004612` to `0.003438`,
mean survival from `0.51875` to `0.759375` (`1.464x`), and completion from 0/16
to 2/16. It ran longer on 14/16 paired trajectories, with a median gain of five
saved steps. At every case's common endpoint its relative error was lower; the
paired survival improvement has a trajectory bootstrap 95% interval of about
`[0.159, 0.306]`. The epoch-60 unroll checkpoint was selected.

The weight-`0.1` admissibility row reached one-step relative L2 `0.003418`,
survival `0.75625`, completion 2/16, and 14 nonpositive terminations. Compared
with unroll-only, its survival ratio is `0.996`; four trajectories are longer,
eight tied, and four shorter. It therefore fails both barrier-attribution
conditions: no `1.10x` survival gain and no reduction in nonpositive failures.
The barrier contributes about 4.5% of state loss only in stage epoch 1 and less
than 0.1% thereafter. This is a negative result for this loss specification, not
for all differentiable invariant-domain penalties.

Result-to-claim classification is `partial`: short recurrent training is a
supported stability intervention on this fixed midscale 1D Euler screen, but
14/16 test trajectories still terminate early and no target-family superiority,
long-horizon robustness, or cross-stride claim is established. Promote only the
unroll-only row to 384/64/64.

## 2026-07-15 Full-Scale Conservative FNO Result

The promoted row used the 316,739-parameter FNO on the complete 512-trajectory,
101-frame, 256-cell cumulative-impulse dataset. The fixed split was 384/64/64,
with 38,400/6,400/6,400 stride-1 one-step pairs. Fifty one-step epochs supplied
240,000 optimizer updates; ten four-step recurrent epochs supplied another
46,560 optimizer updates. No input noise, limiter, floor, positive transform, or
admissibility barrier was active. Runtime was about 1 hour 54 minutes.

The epoch-59 checkpoint was selected by validation survival, then one-step loss.
On test it reaches one-step relative L2 `0.001305`, mean raw survival `0.97734`,
and 57/64 completed 20-call rollouts. Mean valid length is 19.55 calls. All seven
terminations are raw nonpositive proposals: three pressure failures and four
density failures; no nonfinite failure occurred. Mean rollout relative L2 is
`0.01444`, final-prefix relative L2 `0.03097`, shock-position MAE `0.00928`, and
final conserved-total error `0.00373`. Mean initial effective CFL is `3.84` and
the maximum is `5.64`.

The original scale-confirmation and 20-call promotion gates pass. The strict
strong-baseline gate requires at least 90% completion and fails by one case:
57/64 is `0.890625`, while 58/64 would pass. The causal within-run evidence is
the fixed validation set: the best one-step checkpoint has survival `0.635` and
5/64 completion, while the selected recurrent checkpoint has survival `0.984`
and 56/64 completion. A midscale-to-full comparison uses a different held-out
split and must not be presented as a paired data-scale effect.

Failure onset is late rather than an obvious high-CFL subgroup. Failed cases
have lower mean errors than completed cases through about call 8, then separate
rapidly after calls 10--12. The three pressure failures have much larger shock
error than the four density failures, indicating at least two residual modes:
shock-front/pressure breakdown and late density undershoot or ripple. The
result-to-claim classification remains `partial`: this is strong evidence that
a recurrently trained, global, exactly conservative FNO can learn this fixed
large-step 1D Euler map without inference-time positivity repair, but seven raw
positivity failures prevent a general no-constraint claim. Long-horizon,
target-family, seed, stride, resolution, and 2D transfer evidence is absent.

## 2026-07-15 Long-Horizon Flux-Checkpoint Result

The saved epoch-59 full-scale flux checkpoint was evaluated without retraining
at 20, 50, and 100 raw calls on the same 64 test trajectories. The 20-call
metrics reproduce exactly: 57/64 completion and mean survival `0.97734`. At 50
calls, only 1/64 completes and mean survival is `0.565625`; 63 cases terminate
on raw nonpositive proposals. At 100 calls, 0/64 completes, mean survival is
`0.28297`, and all cases have terminated by call 51. There are no nonfinite
terminations.

This fails the predeclared 50-call gate of survival at least `0.85` and
completion at least `0.50`. The full-scale flux row is therefore classified as
a strong short-horizon result and a medium-horizon recurrent-stability failure.
It does not support a claim that training accuracy can replace positivity or
stability mechanisms over arbitrary horizons.

Termination time at the 100-call request has median 27 and range 12--51. Density
becomes nonpositive in 56/64 cases and pressure in 13/64, with five cases showing
both; thus the later failure population is predominantly a density-undershoot
mode rather than the pressure-heavy earliest failures. Initial effective CFL is
not a sufficient explanation: its correlation with valid length is about
`-0.36` at this horizon, and the seven cases that already failed by call 20 had
essentially no initial-CFL association. The next stability intervention should
therefore expose the model to longer recurrent distributions before adding a
generic CFL-conditioned or whole-sample limiter.

## 2026-07-15 Strict State-Target Result

Direct next conservative state and conservative residual labels both pass the
float32 closure gate. The direct target is an uncentered map; the residual has
zero output as the identity update. With the same eight-case, 50-epoch tiny-fit
budget, direct state misses both thresholds on seed `20260707` (minimum train
relative L2 `0.01658`, normalized loss `4.33e-4`) and the declared repeat seed
`20260708` (`0.01711`, `4.84e-4`). Residual passes on the first seed at
`0.00648` and `8.08e-5`. Direct state is stopped as an optimization/centering
failure, not a label or rollout failure.

The 316,419-parameter residual head then advanced to the 64/16/16 matched screen.
After the frozen 50+10 schedule it reaches test one-step relative L2 `0.002649`,
mean survival `1.0`, 16/16 completion, zero nonpositive terminations, final
rollout relative L2 `0.04895`, and shock-position MAE `0.0240`. The matched
316,739-parameter state-loss-only flux head has `0.003438`, `0.7594`, 2/16,
14 nonpositive terminations, truncated final-prefix L2 `0.1547`, and shock MAE
`0.05184`.

Residual survives longer on 14/16 paired cases and ties on the two flux
completions. It has lower relative error on all 16 cases at their common
endpoint; the mean endpoint-error ratio is `0.316`. The paired survival gain is
`0.2406`, with trajectory-bootstrap 95% interval about `[0.134, 0.369]`. This
passes the strict midscale gate and promotes residual to 384/64/64. The causal
claim is still `partial`: this establishes a target-parameterization effect on
one split and seed, but residual does not enforce facewise conservation and has
not yet passed seed, stride, or resolution transfer.

## 2026-07-15 Full-Scale Residual and 100-Call Result

The promoted 316,419-parameter residual FNO used the same 512-trajectory ADER
artifact, 384/64/64 split, 50 one-step plus ten four-step recurrent epochs,
conservative coordinates, and unconstrained inference contract as the flux
comparison. Runtime was `6014 s` (about 1 hour 40 minutes). Epoch 60 was selected.
On test, one-step relative L2 is `0.001123`; all 64 trajectories complete 20
calls with final relative L2 `0.01215`, shock-position MAE `0.00428`, and
conserved-total error `0.00316`. Recurrent training improves the best validation
20-call error by `39.3%` relative to the best one-step-only checkpoint while
preserving full survival.

Without retraining, 62/64 cases complete 50 calls (survival `0.990`) and 61/64
complete 100 (survival `0.97781`). Completed-case 100-call final error has mean
`0.0551`, median `0.0482`, and p90 `0.104`; aggregate conserved-total error is
`0.00927`. No density or nonfinite termination occurs. The three pressure
failures occur at calls 33, 37, and 91 near interior moving waves, after local
velocity overshoot and pressure/internal-energy collapse. Valid length has
essentially zero truth-CFL correlation (`r` about `0.01`).

Against the matched flux checkpoint, residual is lower-error on 58/64 cases at
their 20-call common endpoint; the paired mean difference is `-0.0189` with
bootstrap 95% interval about `[-0.0304, -0.00975]`. At the longer common
endpoint it wins 63/64 and has median error ratio `0.165`. The legacy pressure-
argmax shock mean is affected by multi-wave rank switching: focused replay shows
the large residual outliers retain both fronts at the correct locations but
mis-rank their gradient amplitudes. Report top-k wave matching and amplitude
error alongside the scalar argmax metric.

Result-to-claim remains `partial`. This is a strong unconstrained baseline on one
fixed split/seed and a decisive target-parameterization result against the
matched state-loss-only face-flux head. It is not structurally facewise
conservative and does not establish seed, stride, resolution, or 2D transfer.

## 2026-07-04 Diagnostic Update

Artifacts created under ignored `artifacts/time_dependent_no/`:

| Artifact | Purpose |
| --- | --- |
| `cpg_mechanistic_subset_20260704/` | Local compact subset pulled from AutoDL: trajectories 0, 6, 11, 13, 17 for one-step and two-stage, with `predicteds`, `targets`, `pos`, `edges`, and `node_type`. |
| `cpg_mechanistic_diagnostic_20260704_subset/` | First selected-subset diagnostic and key findings. |
| `cpg_mechanistic_diagnostic_20260704_full/` | Full 20-trajectory AR rollout diagnostics for one-step and two-stage CPGNet. |
| `cpg_teacher_forced_per_time_20260704/` | Per-time teacher-forced RMSE for both bs2 checkpoints. |
| `cpg_state_drift_20260704_full/` | Sampled train-stat range/z-score drift diagnostic for full rollouts. |
| `cpg_perturbation_amplification_20260704/` | Controlled perturbation amplification probe for one-step and two-stage CPGNet. |
| `cpg_shock_overlay_gallery_20260704/` | Selected-subset pressure/shock overlay gallery for qualitative phase/shape readout. |
| `cpg_cloud_diagnostics_20260704_summary.md` | Compact interpretation of the full cloud diagnostics. |


## 2026-07-04 Literature-Informed Update

The diagnostic plan now incorporates lessons from four neural-operator failure-mode papers: spectral FNO analysis, APEBench, PDE-Refiner, and Recurrent Neural Operators. The main additions are:

- Treat one-step accuracy and autoregressive stability as separate quantities in every report.
- Add correlation-time and geometric rollout aggregation so long-horizon behavior is visible even when final-step averages are noisy.
- Add graph-native scale/spectral residual diagnostics to test whether low-amplitude or high-frequency shock-local residuals drive rollout failure.
- Add an effective-CFL/receptive-field audit to compare shock-front motion against graph spacing and model propagation depth.
- Defer recurrent/unrolled control experiments until the scale/resolution and propagation diagnostics identify a concrete target.

Current blockers:

- Approximate/mesh-weighted conservation remains blocked until geometric weights are validated.
- Corrected PCNO replay is complete through the collaborator-compatible preprocessing path. Future PCNO conclusions should use that path, not the retired raw-HDF5 adapter.
- The 1D scale/spectral diagnostic and D025 pilot are complete. The graph-native
  2D scale extension remains pending but does not reopen D025.
- New 2D CPG/PCNO recurrent controls remain blocked on D014.

## 2026-07-05 Interface-Latent Run Result

D017 completed on AutoDL for the selected trajectories/frames. Compact outputs are under ignored `artifacts/time_dependent_no/cpg_interface_latent_diagnostic_20260705_full/`, including `summary.json`, `per_frame_summary.csv`, `per_wave_type_summary.csv`, `sampled_edges.csv`, and `aggregate_analysis.md`.

Main evidence: decoded interface states remain density/pressure-admissible but are not physical one-sided traces; pressure boundedness between adjacent node states is near zero and owner-closer fractions are only about 0.52. The LLF/FV path is internally consistent with the model delta, but its true-update error remains nontrivial, especially around shocks. Dissipation is enriched near shocks only weakly: top-decile dissipative edges are about 17% target shock-front edges versus an 8.6% base shock-edge fraction. Frame-motion/wave-speed diagnostics are noisy and do not support a single too-fast or too-slow explanation.

## 2026-07-05 Interface-Latent Priority Update

Idea 2.2 is now the next diagnostic priority before Idea 2.1 target-ladder training. The new script should be run first on the selected hard/representative trajectories and frames for both bs2 checkpoints. Its immediate purpose is to decide whether learned `reconstruct_prims` behaves like physical one-sided traces, space-time averaged predictors, flux coordinates, hidden dissipation, or a wrong wave-speed / shock-shape mechanism.

Local implementation status:

- The exact one-off implementation was retired from the active tree during cleanup after the D017/D018 results were recorded.
- Full D017 evidence has been generated under ignored interface-latent diagnostic artifacts.

## 2026-07-05 Interface Mechanism Probe Result

D018 completed on AutoDL for the same selected trajectories/frames as D017, now in both teacher-forced and autoregressive state-source modes. Compact outputs are under ignored `artifacts/time_dependent_no/cpg_interface_mechanism_probe_20260705_full/`, including `summary.json`, `per_frame_summary.csv`, `projection_summary.csv`, `inverse_fit_summary.csv`, `report.md`, and `aggregate_analysis.md`.

Main evidence: replacing learned interface states with physical candidates or componentwise-bounded projections changes the induced model update substantially. The best physical projection (`expanded_clip`) still has update-vs-model energy relative L2 about 1.7-1.9 and learned-flux relative L2 about 7.8-7.9. Constrained inverse fits inside strict local owner-neighbor boxes almost never match learned fluxes within 10%; one-jump-expanded boxes improve the median residual but still leave relative L2 medians around 1.4-3.9 and high p90 residuals. AR-mode inputs increase learned true-update energy relative L2 from about 0.075-0.080 teacher-forced to about 0.79-0.99 on frames >0, while the internal learned update still matches the model delta at roughly 1e-6.

Interpretation: `reconstruct_prims` is not just a strangely scaled physical trace. It behaves as a functional nonphysical flux-control coordinate; projection to plausible physical states destroys the model update, and the learned flux is generally not close to a local physical LLF flux manifold. This strengthens the case for Idea 2.1 target-ladder training with explicit physical/interface/flux targets rather than supervising PCNO-FV from raw CPGNet latents.

## 2026-07-05 Corrected PCNO Preprocessed Rollout Result

After collaborator feedback, D019 uses the corrected PCNO replay path: HDF5-to-npy conversion and reconstruction before loading the Euler PCNO `.npz` contract. The active script is `scripts/time_dependent_no/rollout_pcno_preprocessed.py`; the raw-HDF5 graph adapter was retired from the active tree.

Corrected selected-rollout artifacts are under ignored `artifacts/time_dependent_no/`, including `pcno_corrected_rollout_20260705_selected/` and `pcno_corrected_animation_gallery_20260706_selected/`.

Main evidence: PCNO initially places the shock front more accurately than CPGNet, but nonphysical ripple artifacts grow over time and can trigger rollout crash. In the selected corrected replay, pressure mean RMSE is about 2.13 and final pressure RMSE about 3.48; velocity errors can reach overflow scale. The next useful diagnostics are direct ripple-energy and shock-front-position metrics, not time-lag curves.
## 2026-07-06 Active-Tree Cleanup

The tracked `time_dependent_no` surface was reduced to reusable utilities, active rollout/visualization diagnostics, corrected PCNO replay, and compact documentation. One-off inspection, smoke, state-drift, perturbation, time-alignment, raw-HDF5 PCNO adapter, and CPGNet interface-latent probe implementations were removed from the active tree after their conclusions were recorded here. Use git history for exact reproduction; do not treat retired paths as current entry points.
## 2026-07-09 Idea 2.1 Priority Update

The active weekly objective is now Idea 2.1: solver-facing target diagnostics. Use 1D Euler as the fast pilot to compare target parameterizations before transferring only the useful stabilized variants to CPGNet-style and 2D bump runs.

The current target ladder should remain compact: choose one nonzero training-noise level from the active follow-up batch, then run the FNO stride-4 selector over `limited_residual`, `limited_flux`, and `positive_limited_interface` with noise `0` and the selected nonzero noise. Seed-confirm only the best one or two variants.

This target-ladder work also supports later Idea 2.2. By making models predict residuals, fluxes, or interface states through explicit adapters, the learned quantities become inspectable as physical traces, flux corrections, dissipation controls, or nonphysical update coordinates.

## 2026-07-09 1D Euler Noise Follow-up Result

The `euler1d_noise_followups_v2` batch completed on AutoDL. Compact local analysis was written under ignored `artifacts/time_dependent_no/euler1d_noise_followups_v2_analysis_20260709/`.

Main evidence: for `FNO + limited_residual + stride 4`, three-seed final rollout L2 was `0.0861 +/- 0.0329` with no noise, `0.0754 +/- 0.0173` with noise `0.003`, and `0.0755 +/- 0.0055` with noise `0.02`. Noise `0.003` gave the best mean rollout/conservation without pressure-floor hugging. Noise `0.02` gave much better shock MAE (`0.0058 +/- 0.0015`) but had worse one-step error and minimum pressure at the floor.

Limiter control: raw conservative residual did not crash in the small stride-4 control and beat limited residual at noise `0` for one seed, but at noise `0.02` limited residual had better final rollout and shock metrics. Interpret this as evidence that the current limiter may be conservative/diffusive, not that admissibility control is unnecessary.

Stride-2 check: zero-noise stride 2 had nonpositive raw-pressure counts and final L2 `0.1318`; noise `0.02` improved final L2 to `0.0962`, conservation to `0.0286`, and shock MAE to `0.0061`. Smaller stride does not automatically stabilize rollout because it increases the number of learned operator applications.

Decision: use noise `0.003` as the default nonzero setting for the first stabilized target selector. Keep noise `0.02` as a shock-stability stress setting, especially when a target family is shock-unstable after the `0`/`0.003` selector.

## 2026-07-09 1D Euler Stabilized Target Selector Launch

Implemented the shared `ConservativeUpdateLimiter`, `limited_flux`, and `positive_limited_interface` target adapters. `limited_flux` and `positive_limited_interface` use samplewise limiting so a single limiter coefficient scales the whole finite-volume update for each trajectory sample, preserving interior face-pair conservation accounting. Existing `limited_residual` keeps cellwise limiting for backward compatibility with the prior residual experiments.

Local verification before launch: `uvx ruff check` passed for the touched Python files, `tests/time_dependent_no` passed, and tiny FNO CPU smoke runs completed for both new targets. Remote CPU smoke runs on the real 1D Euler dataset also completed for `limited_flux` and `positive_limited_interface`.

The compact AutoDL selector `euler1d_target_selector_v1` was launched on 2026-07-09 at 23:08 CST. Matrix: FNO, stride 4, 40 epochs, 384 train cases, 64 test cases, targets `limited_residual`, `limited_flux`, and `positive_limited_interface`, with training noise `0` and `0.003`. The first run (`limited_residual`, noise `0`) reached epoch 2/40 with finite training and test metrics. Output directories use the relative pattern `artifacts/time_dependent_no/target_selector_v1_fno_*`; the latest log path is recorded in `artifacts/time_dependent_no/euler1d_target_selector_v1_latest_log.txt`.
## 2026-07-10 1D Euler Target Selector V1 Result

The compact FNO stride-4 stabilized-target selector completed all six runs. Ignored local analysis artifacts are under `artifacts/time_dependent_no/target_selector_v1_analysis_20260710/`, with downloaded lightweight metrics under `artifacts/time_dependent_no/target_selector_v1_metrics_20260710/`.

Main evidence: one-step relative L2 was similar across targets (`0.0059` to `0.0081`), but rollout separated sharply. `limited_residual` with noise `0.003` had the best final rollout L2 (`0.0933`), shock MAE (`0.0175`), and conservation error (`0.0457`), improving over zero-noise `limited_residual` (`0.1231`, `0.0305`, `0.0663`). However, it touched the pressure floor in a few rollout states, so it remains a cautionary baseline rather than a clean structure-preserving solution.

Current flux/interface target forms failed the selector. `limited_flux` final rollout L2 was `0.6138` at noise `0` and `0.4182` at noise `0.003`; `positive_limited_interface` was `1.3574` at noise `0` and `0.3594` at noise `0.003`. Both families had large shock MAE, pressure-floor hugging, nonpositive raw pressure after conservative decoding, and high limiter activation during rollout, despite stable one-step metrics.

Interpretation: conservation-form decoding alone is not enough. The current absolute flux and absolute interface-state parameterizations are structurally conservative but dynamically wrong under autoregressive distribution shift. The limiter prevents immediate crash but becomes an emergency clamp. Next target work should prioritize physical base flux plus bounded correction, direct macro-step face-flux supervision from the generator, and bounded interface corrections around local/Riemann base states.
## 2026-07-10 Physical Flux-Correction Target Launch

Implemented `physical_flux_correction` for the 1D Euler target ladder: the network predicts a bounded correction around the current-state Rusanov face flux, then the finite-volume update and shared samplewise conservative admissibility limiter decode the next state. The target exposes correction-scale diagnostics: mean/max absolute correction over bound and saturation fraction.

Local verification: focused solver-target tests passed, `tests/time_dependent_no` passed, script help/smoke checks passed, and ruff passed with the established E402 ignore for script entry points.

Remote smoke on the real 1D dataset passed. The zero-correction/base-Rusanov audit at stride 4 was poor (`one_step_l2` about `0.198`, final rollout L2 about `0.818`, shock MAE about `0.410`, and rollout limiter activation about `0.999`), so the learned correction must do real macro-step work rather than lightly polishing a good classical step.

The first full scale-1 launch was interrupted after early live diagnostics showed test relative L2 near `0.10` and correction saturation around `0.18-0.20` with no teacher-forced limiter activation. Decision: run a short bound-scale probe over correction scales `1`, `2`, and `4` before spending the full training budget. Active ignored outputs use the relative pattern `artifacts/time_dependent_no/physical_flux_scale_probe_v1_*`, with log `artifacts/time_dependent_no/logs/euler1d_physical_flux_scale_probe_v1.log`.
## 2026-07-10 Physical Flux-Correction Scale Probe Result

The short scale probe over correction scales `1`, `2`, and `4` completed on AutoDL. Ignored local analysis artifacts are under `artifacts/time_dependent_no/physical_flux_scale_probe_v1_analysis_20260710/`, with lightweight downloaded metrics under `artifacts/time_dependent_no/physical_flux_scale_probe_v1_metrics_20260710/`.

Main evidence: all three physical-flux-correction scale settings were rejected. Scale `1` had one-step L2 `0.0991` and final rollout L2 `1.1100`; scale `2` had one-step L2 `0.0850` and final rollout L2 `0.8259`; scale `4` had one-step L2 `0.0805` and final rollout L2 `0.8568`. All reached pressure floor, had nonpositive raw pressure counts, and had rollout limiter activation about `0.95-0.96` with minimum theta `0`.

Interpretation: increasing the correction bound improves one-step fit but does not stabilize rollout. Scale `1` is correction-bound-limited, while scale `4` largely removes correction saturation but activates the teacher-forced limiter heavily. The base audit showed the core issue: one explicit Rusanov flux over the stride-4 macro step is a poor large-timestep anchor, so the network must cancel and replace the base flux rather than learn a small physical correction.

Decision: do not run the full noise `0`/`0.003` selector for this exact target family. Next flux-target work should prioritize direct macro-step time-integrated face-flux supervision or a stable/data-derived macro flux base.

## 2026-07-13 Corrected Solver-Level CPGNet M0

The earlier `CPGNetEuler1DHead` rows are now deprecated: that generic directed target head did not execute the paper's solver-level interface-state/FV recurrence. The corrected M0 is `CPGNetEuler1D` with the exclusive `cpg_interface` target. It uses physical ghost nodes, directed geometry-only edge encoding, 12 unshared message-passing layers, target-node interface reconstruction, positive density/pressure interface decoders, one shared oriented Rusanov flux per face, and an exact finite-volume update. Left inflow is anchored to the case state; the right-wall exterior interface is the reflected owner-side prediction. Exact 1D geometry replaces the release model's learned positive geometry factor.

No post-update cell-state floor or admissibility limiter is used in the corrected recurrence. Invalid raw density/pressure terminates and is counted by rollout diagnostics. Training now supports the paper-compatible two-stage schedule: one-step standardized next-state loss followed by three-step fully differentiable autoregressive fine-tuning with reduced learning rate and additive Gaussian primitive-input noise. The existing FNO path retains its admissibility-preserving log-normal density/pressure noise. Checkpoint selection prioritizes completed admissible validation rollouts by final error; if every candidate fails the horizon, one-step validation loss selects the best-fit failure for diagnosis rather than silently freezing on epoch 1.

Local verification: all `tests/time_dependent_no` tests passed (`55 passed`), and a small synthetic CPU gate with the additive CPG noise path completed without nonpositive raw states. With a 16-hidden-channel, 2-layer smoke model trained for five one-step epochs plus one three-step autoregressive epoch, one-step relative L2 was `0.00177` and four-step final-rollout L2 was `0.00696`; final conservation error was `0.00129`. These numbers validate the implementation path only and are not benchmark evidence.

This M0 launch gate is now closed by D020. The h128/mp28 run supplies the
competitive-fit, raw-recurrence reference; remaining CPG work is limited to its
shock-tail diagnostic and two matched-parameter locality controls.

## 2026-07-14 1D Euler Receptive-Field Result

The matched h128 intervention completed on AutoDL with the same data split,
stride-4 operator, seed, noise, optimizer, 15 one-step epochs, and five
three-step unrolled epochs. The only architectural change was unshared directed
message-passing depth, although that also increased parameter count.

| Variant | Parameters | Best Epoch | One-Step Loss | One-Step Rel L2 | Completion | Mean Survival | Rollout Mean L2 | Final L2 | Shock MAE | Conservation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CPGNet h128/mp12 | 1,500,931 | 3 | 3.49e-03 | 3.34e-02 | 34/64 | 0.695 | 2.14e-01* | 2.68e-01* | 1.68e-01* | 2.75e-02* |
| CPGNet h128/mp28 | 3,350,275 | 20 | 6.75e-05 | 6.14e-03 | 64/64 | 1.000 | 2.06e-02 | 2.98e-02 | 2.57e-02 | 1.03e-02 |
| FNO 64/24/4, limited residual | 316,419 | 79 | 6.43e-05 | 5.10e-03 | 64/64 | 1.000 | 2.16e-02 | 3.72e-02 | 2.51e-03 | 1.89e-02 |

`*` mp12 aggregates use variable-length valid prefixes because 30 raw
rollouts terminate; they are not full-horizon estimates.

Mechanistic evidence:

- mp28 lowered the initial-rollout error on all 64 paired test cases. The mean
  fell from 0.1366 to 0.01557, with paired bootstrap mean difference 95% CI
  [-0.1326, -0.1095].
- All 30 mp12 failures were rescued by mp28. mp12 completion was 10% for cases
  with truth effective CFL in [20, 24), and zero above 24; mp28 completed every
  bin through the observed maximum 28.44.
- Initial effective CFL strongly predicted mp12 first-step error
  (Pearson/Spearman 0.901/0.900) but not mp28 error (-0.120/-0.197).
- Against FNO, mp28 is slightly worse at early steps but accumulates error more
  slowly. Its final L2 is 19.9% lower (paired p=0.020), while mean rollout L2
  differs by only 4.4% and is not significant (paired p=0.199).
- mp28 conservation error is 45.7% lower than FNO and wins on 75% of cases.
  Both methods complete all raw rollouts without active limiting.
- mp28 shock median is 0.00234, close to FNO's 0.00176, but its tail is much
  worse: p95 0.0747 and maximum 0.5785 versus FNO p95 0.00684 and maximum
  0.00801. Three cases dominate the mp28 mean shock error.
- Saved-checkpoint replay of those three cases shows a real but
  argmax-amplified failure: mp28 retains a strong nearly stationary pressure
  gradient near the original discontinuity while the target front moves.
  Predicted second/first gradient ratios are about 0.65-0.93, so a largest-front
  metric switches between two comparable fronts, but the persistent ghost
  front is not a metric artifact.

Result-to-claim gate: `partial` with high internal confidence, pending an
independent Codex review because unpublished results were not sent to an
external tool without approval. The supported statement is narrow: on this
fixed 1D Euler stride-4 dataset, a CPGNet whose hop depth covers the observed
macro-step domain of dependence can learn a stable implicit macro-operator,
whereas mp12 underfits and fails high-CFL cases. The result does not yet prove
that hop depth rather than added capacity causes the full gain, nor does it
establish timestep/resolution transfer or robust shock tracking.

Completion update, 2026-07-15:

- The mp12/h193 and mp28/h85 controls are complete. Width does not recover the
  shallow model and the narrow deep model retains the gain, supporting hop depth
  as the primary mechanism. Freeze h128/mp28 and stop the CPG architecture
  sweep.
- The four-way FNO coordinate matrix is complete. Conservative input, loss, and
  recurrence with fixed physical scaling are the provisional D021 contract.
- The 512-case ADER cumulative-impulse dataset and serialized closure checks are
  complete. The exact face-value supervision screen then reached the midscale
  stop documented above; it is no longer queued or waiting.
- The matched stability screen is complete. Unroll-only passed its midscale
  promotion gate; the weight-`0.1` barrier failed attribution.
- The frozen full-scale unroll-only confirmation is complete at 57/64 raw
  completion and survival `0.97734`. It passes scale and 20-call promotion gates
  but misses the strict 90% completion gate by one trajectory.
- The frozen flux checkpoint then fails the 50-call extension at 1/64
  completion. Direct next state fails tiny fit on both declared seeds. The
  completed full-scale residual checkpoint reaches 64/64, 62/64, and 61/64 at
  20/50/100 calls and wins 63/64 longer common-endpoint comparisons. D021 remains
  open for seed confirmation, conservation-compatible residual projection,
  stride, and resolution gates.

## 2026-07-16 Three-Seed Residual Projection Result

The matched full-scale comparison used the frozen 512-trajectory split,
conservative coordinates, 50 one-step plus ten four-step recurrent epochs, and
20/50/100 raw calls. The residual FNO has 316,419 parameters; the projected
variant has 316,806. Neither uses noise, a limiter, a floor, a positive
transform, or an admissibility loss. The projected decoder separates a
volume-zero cell increment from one learned three-component boundary budget;
its decoded increment closes to that budget at about `1.0e-7` absolute error.

| Seed | One-step L2 | 100-call completed | 100-call final L2 | Conserved-total error | Shock MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20260707 | 0.001133 | 62/64 | 0.04730 | 0.00720 | 0.00435 |
| 20260708 | 0.001060 | 61/64 | 0.05940 | 0.00834 | 0.00622 |
| 20260709 | 0.001113 | 64/64 | 0.06136 | 0.00904 | 0.00355 |

Projection raises pooled completion from 187/192 to 189/192 at 50 calls and
from 182/192 to 187/192 at 100 calls, so the preregistered stability-
noninferiority gate passes. The other family gates fail: only one seed improves
100-call state error; the three-seed mean state-error ratio is `1.016`; only one
seed improves conservation at both 50 and 100 calls; and the mean conservation
ratios are `0.953` and `0.871`, short of the prespecified robust criterion.
Both new seeds also miss the broad frame-20 gate because shock MAE is worse.

Variable-length prefix averages are optimistic for different models on
different cases. Among the 181 seed-case pairs completed by both methods at 100
calls, projection is 9.8% worse in final state L2, 5.9% better in conserved-total
error, and 4.8% worse in shock MAE. At each trajectory's last common valid step,
its pooled state L2 is 2.5% worse. All projected failures are raw nonpositive-
pressure events on the same hard case family as residual; the parameterization
changes which seeds enter that failure basin rather than eliminating it.

Classification is `partial`: the low-dimensional boundary-budget factorization
is an exact and useful conservation coordinate and gives a real stability
signal, but it is not a seed-stable accuracy or shock improvement. Keep plain
residual as the strong accuracy baseline and projected residual as a structural
ablation. The next bounded intervention is generated burn-in plus detached
short BPTT; do not reopen full face-value supervision or add an inference
limiter.

## 2026-07-16 Generated-State Burn-In Pilot

The matched 64/16/16 pilot used the 316,806-parameter projected-residual FNO,
seed `20260708`, the fixed conservative-coordinate contract, and no noise,
limiter, floor, positive transform, or admissibility loss. Both rows used 50
one-step epochs and four supervised recurrent steps. The control used no
burn-in for ten recurrent epochs (`7,760` updates); the intervention rolled
eight detached model steps before supervision for eleven epochs (`7,832`
updates, `1.009x`). Their one-step histories match exactly and both select
recurrent epoch 58. Total wall time, including sanity and three horizons per
row, was about 41 minutes.

The generated states are a materially harder training distribution without
being inadmissible. Burn-in relative L2 falls from `0.01057` to `0.00897` over
the second stage; no burn-in sample has nonpositive density or pressure, and
the observed minima are `0.0831` and `0.0675`. The supervised four-step train
loss starts about an order of magnitude above the clean-start control and then
falls by about 47%. This confirms that the intervention is active and that
clean four-step BPTT underexposes the model to later accumulated errors.

| Calls | Clean completed | Burn-in completed | Survival clean / burn-in | State-L2 ratio | Conserved-total ratio | Shock ratio | Paired burn-in wins |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | 16/16 | 16/16 | 1.000 / 1.000 | 1.021 | 0.924 | 1.382 | 8/16 |
| 50 | 15/16 | 15/16 | 0.9988 / 0.9950 | 0.768 | 0.659 | 1.068 | 15/16 |
| 100 | 14/16 | 14/16 | 0.9650 / 0.9394 | 0.638 | 0.537 | 0.936 | 14/16 |

The predeclared primary gate fails on 20-call shock regression and the 0.01
100-call survival tolerance. The aggregate failure is not broad. Burn-in
rescues case 21, whose clean pressure becomes nonpositive at call 96, but case
234 changes from completed to a density/pressure failure at call 58 and case
238's pressure failure advances from call 50 to 47. The 20-call shock increase
is also concentrated: case 238 explains about 56% of the mean increase, and
cases 183 and 348 bring the concentration to about 85%.

Classification is `partial recurrent-distribution improvement with rare-case
stability regression`. Later-state exposure strongly reduces medium/long-
horizon error on most cases, but eight burn-in calls do not sample every later
failure basin and are not a positivity mechanism. Do not promote projected
residual to a full-scale accuracy row. The matched plain-residual exposure gate
below is the relevant accuracy-baseline continuation; defer a pressure/internal-
energy penalty until its mechanism controls resolve the active failure mode.

## 2026-07-16 Plain-Residual Generated-State Exposure Result

The matched 64/16/16 plain-residual gate used the 316,419-parameter FNO, seed
`20260708`, conservative coordinates, no noise or constraint, 50 identical
one-step epochs, and four supervised recurrent steps. The clean control used
ten recurrent epochs and 7,760 optimizer updates. The intervention used eight
detached generated steps followed by four-step BPTT for eleven epochs and 7,832
updates (`1.009x`). Generated prefix error fell from `0.01008` to
`0.00922`, with no nonpositive prefix states.

| Calls | Clean completed | Burn-in completed | Survival clean / burn-in | Common-endpoint state ratio | Conserved-total ratio | Legacy shock ratio | Burn-in wins |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | 16/16 | 16/16 | 1.000 / 1.000 | 0.954 | 0.923 | 1.456 | 9/16 |
| 50 | 15/16 | 16/16 | 0.9975 / 1.000 | 0.678 | 0.592 | 1.145 | 11/16 |
| 100 | 15/16 | 16/16 | 0.9675 / 1.000 | 0.486 | 0.526 | 1.066 | 14/16 |

The formal primary gate fails only the 20-call legacy shock criterion. Three
cases explain nearly all of that mean increase. Cases 114 and 21 contain
important front-strength/rank changes; case 238 is a real short-horizon
regression with an extra or displaced strong pressure front. Do not erase the
failed gate after this adjudication. Subsequent runs report both the legacy
argmax and separated top-two position/strength metrics.

Classification is `strong later-state-exposure signal with unresolved causal
source and one real shock regression`. The current generated-burn objective
supervises the reference next state after a perturbed model prefix. For a
generated error `e`, the consistent PDE target is the reference solver
advanced from the generated state, which differs to first order by the solver
Jacobian acting on `e`. The present label can therefore train a trajectory
correction or denoiser. D022 holds the later-time window fixed and replaces the
generated prefix with the exact offset state; D023 then advances generated
states with the reference solver. Full-scale promotion and a pressure penalty
wait for those controls.

## 2026-07-16 Teacher-Offset Mechanism Result

D022 used the same 316,419-parameter residual FNO, 64/16/16 split, seed
`20260708`, 50 one-step epochs, and four supervised recurrent steps as the
plain-residual gate. Teacher offset and generated burn-in both use the same
eight-step-offset windows, eleven recurrent epochs, and 7,832 optimizer
updates. Teacher starts from the exact reference state at the offset; generated
uses eight detached model calls. One-step histories match exactly. Teacher
prefix L2 and nonpositive fraction are both zero.

| Calls | Teacher / clean state ratio | Generated / teacher state ratio | Generated / teacher state-ratio 95% interval | Generated wins vs teacher | Completion clean / teacher / generated |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | 1.109 | 0.861 | [0.782, 0.948] | 13/16 | 16 / 16 / 16 |
| 50 | 1.017 | 0.661 | [0.470, 0.854] | 13/16 | 15 / 16 / 16 |
| 100 | 1.006 | 0.492 | [0.348, 0.669] | 15/16 | 15 / 14 / 16 |

Teacher offset is significantly worse than clean at H20 and statistically tied
at H50/H100. It postpones case 21's pressure failure from call 49 to 51 but
adds a case-114 pressure failure at call 89. Generated exposure completes every
case and cuts conserved-total error relative to teacher by 24%, 46%, and 50%.
The selected teacher checkpoint has one-step relative L2 `0.002660`, better
than generated's `0.002886` despite much worse H50/H100 behavior. This is
another direct inversion between one-step and rollout ranking.

The causal fixed-setting conclusion is that generated off-manifold state
exposure contributes beyond later physical-time sampling. The method-level
result remains partial. Relative to clean, generated top-two front-position
error is 17.8% worse at H20 and 10.2% worse at H50; trajectory-bootstrap
intervals exclude parity. At H100 that front-position difference reverses in
mean but is not significant. Also, the generated-state objective uses the
original reference next state, not the reference solver advanced from the
generated state. D023 must quantify that operator-consistency gap before
full-scale promotion.

Local result-to-claim verdict is `partial` with high internal confidence:
the narrow D022 mechanism claim is supported, while a PDE-consistent burn-in
method claim is not. Independent Codex review is pending because unpublished
results were not sent to an external tool without approval.

## 2026-07-16 D023 Frozen Solver-Consistency Protocol

D023 freezes the clean, teacher-offset, and generated-burn-in residual FNOs.
The generated checkpoint constructs one shared state bank over all 16 held-out
cases, start frames `0,10,...,80`, and prefix depths `0,2,4,8`. Each state is
advanced one saved interval by the dataset WENO-HLLC-ADER solver with its CFL
substeps, retry logic, shock flattening, HLLE troubled-face fallback, inflow,
and reflective wall. All three learned maps are evaluated on that identical
state. Prefix depth zero controls for replaying float32 serialized snapshots.

The primary readout is fixed-physical-scale conservative RMSE to (a) the
same-state solver continuation and (b) the original stored next state. It also
records primitive error, shock/smooth splits, three linearized characteristic
families, separated top-two pressure-front position/strength, and the cosine
between the learned correction from the solver continuation and the direction
back to the original trajectory. Ratios use 2,000 bootstrap replicates over
held-out case IDs, keeping start frames clustered within each case.

At prefix depth eight, call the result PDE-map improvement only if the
generated/teacher ratios to both targets are at most `0.90`. Call it trajectory
correction or denoising only if the original-trajectory ratio is at most
`0.90`, the solver-continuation ratio is at least `0.95`, and the generated
correction-alignment cosine is positive. A solver ratio at most `0.90` with a
truth ratio at least `0.95` is PDE consistency without trajectory recovery;
anything else is mixed or inconclusive. This diagnostic identifies the
existing objective's mechanism and does not by itself justify full-scale or
multi-seed promotion.

## 2026-07-16 D023 Solver-Consistency Result

The 576 same-state solver advances and 1,728 learned-map evaluations completed
in `56.9 s`. Prefix-zero replay RMSE is `1.17e-7` (maximum `4.02e-7`), versus
`1.90e-2` after an eight-step generated prefix, and none of the solver advances
uses a retry or first-order fallback. Serialization or replay failure therefore
does not explain the result.

| Prefix depth | Generated / teacher truth RMSE | 95% interval | Generated / teacher solver RMSE | 95% interval |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 1.147 | [1.086, 1.215] | 1.147 | [1.080, 1.223] |
| 2 | 1.037 | [1.011, 1.069] | 1.019 | [0.997, 1.045] |
| 4 | 1.011 | [0.996, 1.027] | 1.024 | [1.007, 1.044] |
| 8 | 0.992 | [0.984, 0.999] | 1.017 | [1.006, 1.028] |

Generated burn-in gradually removes its clean-state one-step disadvantage and
is 0.8% better than teacher against the original trajectory at depth eight,
but it is 1.7% worse against the actual continuation from that state. Clean is
2.6% closer to the solver than generated. The depth-eight generated solver
defect is 6.3% worse than teacher in the shock region and 8.0% worse than clean;
its truth-front strength error is also 5.9% worse than teacher. The generated
correction from the solver continuation has mean alignment `0.074` toward the
original trajectory (median `0.054`) and norm `0.407` of the solver-to-truth
defect. This is weak trajectory bias, not a dominant denoising map.

The preregistered classification is `mixed_or_inconclusive`. D022 still shows
that exposure to generated states changes closed-loop behavior, but D023 rules
out the simple explanation that it materially improves the local PDE map on
those states. A small systematic modified-equation effect can compound over
100 calls. The completed D013 and D024 results below identify its spectral
signature and reject blind local dissipation as the remedy.

## 2026-07-16 D013 Scale/Spectral Result

The exact frozen-checkpoint run covered the full residual and state-loss-only
flux FNOs, corrected CPGNet mp28, and the D022 clean, teacher-offset, and
generated-exposure residual models. The full residual model completes 61/64
raw H100 rollouts. The flux model completes 0/64 by H60; first failures occur
at call 13 and last failures at call 52, with mean valid survival 28.30 calls.

Teacher-forced fit does not expose a high-frequency disadvantage for the flux
head. Relative to residual, its state error is 1.031, modes-25--64 error is
0.982, second-difference error is 1.013, and its divergence-active update
high-band projection gain is essentially identical (0.9862 versus 0.9860).
The instability appears only under recurrence. From eight calls before failure
to the invalid proposal, flux-model state error grows 10.23 times,
modes-25--64 error 7.45 times, Nyquist-tail error 27.43 times, first-difference
error 23.84 times, and second-difference error 26.23 times. At the same
case/frame, the failure has 10.37 times the residual model's high-band error,
254.7 times its Nyquist-tail error, and 154.2 times its second-difference
error. Error starts shock-local but spreads into the smooth region by failure.

The preregistered classification is
`recurrent_high_frequency_growth_not_one_step_fit`. D022 generated exposure
reduces state error while increasing the error spectral centroid, so it is not
a clean learned-viscosity mechanism. This result justified D024 as a narrow
causal probe, not blanket smoothing as a method.

## 2026-07-16 D024 Conservative-Dissipation Result

D024 freezes the full stride-1 flux FNO and adds only the interior correction
`-kappa (dx / dt) (U_R - U_L)`; the added boundary flux is exactly zero.
The five paired coefficients are 0, 0.0025, 0.005, 0.01, and 0.02. No model is
retrained and no limiter or floor is used.

| Kappa | Mean valid calls | H20 complete | H50 complete | H20 state error | H20 high-band error | H20 Nyquist-tail error |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 28.28 | 57/64 | 1/64 | 0.02022 | 0.00418 | 0.00943 |
| 0.0025 | 27.86 | 57/64 | 0/64 | 0.02234 | 0.00440 | 0.01202 |
| 0.005 | 28.06 | 57/64 | 1/64 | 0.02206 | 0.00420 | 0.01162 |
| 0.01 | 27.30 | 57/64 | 0/64 | 0.02222 | 0.00409 | 0.01249 |
| 0.02 | 26.23 | 55/64 | 0/64 | 0.02416 | 0.00397 | 0.01562 |

For the selected small coefficient 0.005, the paired mean-survival change is
-0.22 calls with a 95% interval spanning zero; H20 state, tail, and shock-width
errors are 1.091, 1.231, and 1.045 times the no-correction control. Larger
diffusion slightly lowers the broad modes-25--64 metric but amplifies the
Nyquist tail, worsens front/width metrics, and shortens survival. The
classification is `no_material_stability_gain`: blind current-state
Laplacian viscosity does not repair the learned nonlinear recurrence.

## 2026-07-16 D025 Global Implicit-Interface Pilot

D025 adds one global face-grid FNO target with six outputs per face. Two
directed traces use multiplicative exponential density/pressure corrections
and sound-speed-scaled additive velocity corrections around the local physical
states. The exterior traces obey the fixed inflow and reflective wall
conditions. One shared Rusanov or central flux then feeds the exact finite-
volume update. Training uses decoded state loss only; recurrence is raw, with
no interface label, limiter, state floor, or positivity penalty.

The matched 8/4/4 tiny runs use 317,126 parameters, 100 one-step epochs, and
the same seed and split.

| Decoder | Minimum train relative L2 | Minimum validation relative L2 | Selected test one-step relative L2 | Raw H20 completion | Failure |
| --- | ---: | ---: | ---: | ---: | --- |
| Rusanov | 0.00734 | 0.01328 | 0.01429 | 0/4 | all invalid on call 2 |
| Central | 0.00823 | 0.01351 | 0.03069 | 0/4 | all invalid on call 3 |

Thus the Rusanov row passes the preregistered fit gate, while central narrowly
misses the selected-checkpoint test threshold of 0.03. Neither is a viable
tiny-set recurrent model. A per-frame frozen analysis localizes the mismatch:
Rusanov conservative relative error is 0.130 at frame 0 versus 0.0087 over
frames 20--99; central is 0.157 versus 0.0186. The frame-0 error is dominated
by shock cells. Both models remain admissible on all 400 truth-state
evaluations, but recurrent tail/high-band error grows within two to three
calls before pressure or density failure. Rusanov's divergence-active flux
MSE is lower than central's in every time bin despite the absence of direct
flux supervision.

The 64/16/16 run localizes a genuine recurrent failure. Its 50-epoch one-step
stage reaches train/validation relative L2 `0.00559/0.00544`, but four-step
training overflows at sequence step three after the generated input has already
become inadmissible (`min rho=-0.919`, `min p=-300.8`). The network output is
still finite (`max |raw|=49.5`); decoding the invalid state produces a finite
prediction of order `1e35`. Skipping a zero-weight barrier fixed one real
trainer bug, but the identical rerun proves that the remaining failure is not
that implementation defect.

A matched two-step curriculum is finite, so three recovery controls were run
on the same 64/16/16 split and 50-call validation horizon:

| Recovery control | Selected test one-step relative L2 | Test mean H50 survival | H50 completion |
| --- | ---: | ---: | ---: |
| Uniform time sampling, barrier `0` | 0.01538 | 0.04875 | 0/16 |
| Frame-zero weight `15`, barrier `0` | 0.00791 | 0.04625 | 0/16 |
| Frame-zero weight `15`, barrier `0.1` | 0.00786 | 0.04750 | 0/16 |

Frame weighting hits its intended mechanism: it lowers initial-call
conservative error from `0.1113` to `0.0304`, shock RMSE from `0.572` to
`0.151`, and top-two front-position error from `0.0683` to `0.0135`. It does
not move the admissibility threshold: all cases still fail on calls two through
four. The barrier lowers its recurrent training value from `0.00443` to about
`1e-7` and slightly improves call-one/two errors, but mean valid calls move only
from `2.3125` to `2.375`; 0/16 complete.

The classification is `well_fitted_but_recurrently_inadmissible`. This is a
negative result for the tested state-loss-only relative-trace parameterization,
not a proof that implicit interface operators or neural operators cannot work.
The likely design liabilities are the underidentified six-trace-to-three-update
map and a zero-output anchor equal to an explicit Rusanov macro-step at effective
CFL about four to six. The FNO has global receptive field, so CPGNet locality is
not the leading explanation. Stop this row at successive halving; do not add an
inference limiter or sweep more barrier/viscosity coefficients.

## 2026-07-16 D026 Boundary-Exchange Supervision Result

D026 tests the smallest identifiable solver-flux quantity that is not affected
by the interior face gauge: the three-component net boundary exchange over one
saved interval. The target is derived directly from accepted-substep,
owner-oriented solver impulses. The projected-residual decoder predicts the
same exchange through its learned boundary budget, and the auxiliary uses a
zero-centered per-channel RMS normalization. Inference remains the exact raw
projected-residual update, with no limiter, floor, positive transform, or
post-hoc projection.

The matched gate uses the 316,806-parameter FNO, 64/16/16 cases, seed
`20260707`, 50 one-step plus ten four-step recurrent epochs, and 20 raw calls.
The zero-weight row is the frozen matched projected-residual control.

| Boundary weight | One-step relative L2 | Boundary-exchange relative L2 | H20 state L2 | Conserved-total error | Shock-position MAE | H20 completion |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.002646 | 0.01892 | 0.04377 | 0.01175 | 0.02789 | 16/16 |
| 0.01 | 0.003048 | 0.01518 | 0.05016 | 0.00744 | 0.03481 | 16/16 |
| 0.1 | 0.003538 | 0.00821 | 0.05242 | 0.00323 | 0.03370 | 16/16 |

Relative to the control, weight `0.01` improves boundary exchange by 19.8%
and conserved-total error by 36.7%, but worsens one-step, H20 state, and shock
errors by 15.2%, 14.6%, and 24.8%. Weight `0.1` improves the two structural
metrics by 56.6% and 72.5%, but worsens the three accuracy metrics by 33.7%,
19.8%, and 20.8%. Both runs preserve full H20 survival and decoder closure near
`1e-7`.

The auxiliary therefore reaches its intended coordinate but exposes a
conservation--accuracy Pareto tradeoff rather than a joint improvement. In 1D,
the net exchange is already identifiable from the endpoint state balance, so
this auxiliary primarily reweights existing information. Moreover, the
projected decoder distributes budget correction globally, while the dominant
remaining errors are shock-local. The supported claim is narrow: direct
boundary supervision can reduce budget error, but it is not a better joint
accuracy/stability objective for this fixed projected-residual setting. Do not
search more weights or promote to the full three-seed scale; proceed to the
predeclared stride and resolution gates for the plain residual baseline.

## 2026-07-16 D027 Cold Stride-2 Gate

D027 tests whether the frozen plain-residual FNO can learn a two-saved-frame
operator directly before adding a continuation mechanism. The gate keeps the
316,419-parameter 64/24/4 FNO, 64/16/16 cases, model seed `20260708`, split
seed `20260707`, conservative coordinates, fixed physical scaling, batch size
8, learning rate `3e-4`, 50 one-step plus ten four-step recurrent epochs, and
raw inference. Only `step_stride` changes from 1 to 2. Checkpoints are selected
at physical frame 20, exactly as for the frozen stride-1 control. The selected
checkpoint is then replayed without retraining to frames 50 and 100, so the
direct model uses 10/25/50 calls and is compared with 20/50/100 calls of the
frozen stride-1 control.

Before launch, the cold row is declared viable without continuation only if:

- selected native one-step relative L2 is at most `0.005`;
- mean direct frame-2 error from the initial condition is at most 1.15 times
  the frozen stride-1 two-call composition error at the same frame;
- common-case state-error ratios at frames 20, 50, and 100 are at most 1.10;
- frame-100 completion does not decrease and mean survival drops by at most
  0.01; and
- frame-20 shock-position and conserved-total error ratios are at most 1.15.

This first cold screen uses the same epoch schedule, which gives about 1% fewer
one-step and 4% fewer recurrent sample presentations because stride-2 windows
are shorter. If the row fails and activates continuation, the final comparison
must instead match final-target sample presentations exactly, add the
total-exposure-matched cold control, compose the saved stride-1 model, transfer
weights only, and reset optimizer and scheduler state. Do not infer a
continuation-sensitive optimization effect from validation or rollout alone;
that claim additionally requires a lower final-target training-loss floor than
both cold controls.

### D027 result

The corrected run selected recurrent epoch 59 and finished in 938.8 s. Its
native test one-step relative L2 is 0.004139, below the absolute 0.005 fit
threshold but 1.61 times the stride-1 value. From the common initial condition,
the direct frame-2 error is 0.01487 versus 0.01223 for two stride-1 calls, a
ratio of 1.216; only 3/16 cases favor the direct jump. This misses the
predeclared 1.15 initial-jump gate.

The longer same-frame comparison strongly favors direct stride 2:

| Physical frame | Stride-1 calls | Stride-2 calls | Common cases | State-error ratio | Direct wins | Completion, stride 1 / 2 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | 20 | 10 | 16 | 0.750 | 15/16 | 16/16 / 16/16 |
| 50 | 50 | 25 | 15 | 0.638 | 13/15 | 15/16 / 16/16 |
| 100 | 100 | 50 | 15 | 0.514 | 11/15 | 15/16 / 16/16 |

Trajectory-bootstrap 95% intervals for the three common-case ratios are
[0.616, 0.866], [0.430, 0.827], and [0.337, 0.744]. At frame 20, direct
stride 2 also improves conserved-total error from 0.00924 to 0.00722
(ratio 0.781) and shock-position MAE from 0.03903 to 0.03557 (ratio
0.911). All 16 direct rollouts remain finite and admissible through frame 100,
although minimum pressure falls to 0.00193, so the result is not a broad
positivity guarantee. The mean truth effective-CFL maximum is about 9.89,
with maximum 13.32.

The classification is large_step_capacity_with_initial_jump_defect. A global
fixed-step FNO can learn a useful stride-2 coarse propagator and substantially
outperform repeated small-step composition over medium horizons. The formal
gate remains partial because the rare nonsmooth initial jump is worse. Follow
the preregistered continuation route: initialize the stride-2 model from the
frozen stride-1 weights, reset optimizer state, and keep the final stride-2
training presentations and all other contracts fixed. Only if that arm improves
the failed initial-jump metric should it advance to the total-exposure-matched
cold control and a continuation-sensitive optimization claim.

## 2026-07-16 D028 Stride Continuation Gate

D028 changes only initialization relative to the corrected D027 cold row. The
316,419-parameter residual FNO is initialized from the rollout-selected
stride-1 epoch-59 weights. Model family, target, architecture, coordinate and
normalization contract, case IDs, model and split seeds, stride-2 labels, batch
size, learning rate, weight decay, 50+10 final-stage schedule, H20 checkpoint
selection, and raw inference are unchanged. The loader checks those contracts,
requires a smaller source stride, verifies identical train/validation/test case
IDs and input-normalizer buffers, loads model weights only, and constructs a
fresh optimizer afterward.

The historical-checkpoint CUDA smoke passed. After one stride-2 epoch,
continuation gives training loss 3.78e-4 and H20 validation error 0.0506 with
full survival, compared with 8.77e-3 and 0.1948 for cold epoch one.

Before the full launch, continuation is declared successful at this
successive-halving stage only if:

- mean direct frame-2 error is at most 1.15 times stride-1 composition, repairing
  the only failed D027 gate;
- selected native one-step error is at most 1.05 times the cold stride-2 value;
- H20/H50/H100 common-case state error is at most 1.05 times the cold stride-2
  value at each horizon;
- H100 completion does not decrease and survival drops by at most 0.01; and
- H20 shock-position and conserved-total errors are at most 1.15 times the cold
  stride-2 values.

A pass justifies the total-exposure-matched cold stride-2 control. Only a lower
final-target training-loss floor than both the original and total-exposure cold
controls supports a continuation-sensitive optimization claim. A rollout gain
without that fit-floor gain is instead a curriculum-induced generalization or
stability result.

### D028 result

The matched run selected recurrent epoch 59 and finished in 909.7 s. Weight
provenance confirms source stride 1, source epoch 59, identical model and split
seeds, and no optimizer-state loading.

Continuation fixes the D027 initial-jump defect. Native test one-step relative
L2 improves from 0.004139 to 0.003623 (ratio 0.875). Mean direct frame-2 error
falls from 0.01487 to 0.01058, which is 0.866 times the two-call stride-1
composition error and 0.712 times cold stride 2. It wins 13/16 cases against
composition and all 16 against cold. The one-step training-loss floor falls
from 4.82e-5 to 2.80e-5 (ratio 0.581); the recurrent floor falls from 4.44e-5
to 2.49e-5 (ratio 0.562).

The solver comparison is nonuniform:

| Physical frame | State ratio, continuation / cold | Continuation wins | Conservation ratio | Shock ratio | Completion, cold / continuation |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | 1.000 | 7/16 | 0.986 | 0.515 | 16/16 / 16/16 |
| 50 | 1.095 | 7/16 | 1.079 | 1.565 | 16/16 / 16/16 |
| 100 | 0.888 | 12/16 | 0.981 | 1.217 | 16/16 / 16/16 |

The paired-bootstrap 95% state-ratio intervals are [0.924, 1.105],
[0.861, 1.537], and [0.725, 1.046]. Continuation also raises the minimum
pressure through frame 100 from 0.00193 to 0.0452, indicating a materially
safer trajectory on this split. Nevertheless, the point H50 state ratio exceeds
the preregistered 1.05 limit, and its H50 shock regression is substantial.

The classification is continuation_repairs_fit_but_not_uniform_rollout.
Warm initialization clearly accelerates and deepens final-target optimization,
repairs the rare initial discontinuity, and moves the model to a safer-pressure
trajectory. It does not yield a uniformly better fixed-step solver. Because the
successive-halving solver gate fails, do not spend the conditional
total-exposure cold run and do not claim continuation-sensitive optimization;
the missing total-exposure control remains an explicit claim boundary. Preserve
cold stride 2 as the large-step accuracy baseline and continuation as a
fit/stability tradeoff ablation.

## 2026-07-16 D029 Frozen Cross-Resolution Gate

D029 asks whether the two frozen 316,419-parameter FNOs trained only at 256
cells define useful zero-shot operators at 128 and 512 cells. It changes no
weights, normalizers, timestep, physical cases, saved times, or test IDs. The
input, loss, and recurrent coordinates remain conservative with fixed physical
scaling. Inference is native-grid and interpolation-free. The stride-1 model
uses 1/20/50/100 calls to reach frames 1/20/50/100; cold stride 2 uses
1/10/25/50 calls to reach frames 2/20/50/100.

This is a discrete-operator transfer test, not an assertion that the predicted
cell residual is itself resolution invariant. The FNO keeps 24 learned Fourier
modes at every mesh, so the 512-cell evaluation tests interpolation and operator
transfer without granting additional learned shock bandwidth. Native reference
differences are reported after conservative finite-volume restriction from
512 to 256 and 256 to 128; they contextualize, but are not subtracted from,
model error.

Before looking at off-grid model results, the gate requires:

- exact identity of left/right states, domains, discontinuity positions, and
  saved times across all 512 cases;
- uniform-grid endpoint reconstruction within `2e-6` and enough grid modes for
  the frozen 24-mode FNO; and
- nx256 replay of the frozen one-step and rollout metrics to numerical print
  precision.

A checkpoint has usable zero-shot resolution transfer only if, on both nx128
and nx512 relative to its own nx256 replay:

- native one-step relative L2 and H20/H50/H100 final state error are each at
  most `1.5x`;
- completion loses at most one of 16 cases at each horizon and mean survival
  falls by at most `0.02`; and
- H20 shock-position and conserved-total errors are each at most `1.5x`.

The D027 larger-step advantage transfers only if cold stride 2 also has no
larger H20/H50/H100 final state error than stride-1 composition on each new
mesh and does not reduce completion. Passing only the lower resolution is a
coarse-grid interpolation result, not bidirectional mesh robustness. Failure at
512 with stable completion will be classified separately from recurrent
instability because fixed spectral bandwidth may impose an accuracy floor near
shocks.

### D029 result

The implementation contract passes. The current generator configuration and
seed reproduce all 512 serialized physical case parameters exactly. All three
datasets have identical saved times and reconstruct their uniform domains within
`6e-8`. The nx256 replay reproduces every D027 one-step, rollout, completion,
shock, and conservation value to printed precision. This rules out checkpoint
restoration, FFT shape handling, split drift, or metric drift as explanations.

Neither frozen checkpoint passes the preregistered native-grid transfer gate:

| Checkpoint | Grid | One-step ratio | H20 state ratio | H50 state ratio | H100 state ratio | H100 completion, off / nx256 | H20 shock ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stride 1 | 128 | 5.692 | 1.345 | 0.984 | 0.961 | 15/15 | 0.651 |
| stride 1 | 512 | 8.340 | 1.306 | 1.077 | 1.017 | 15/15 | 2.098 |
| cold stride 2 | 128 | 5.527 | 1.735 | 1.168 | 0.910 | 15/16 | 0.656 |
| cold stride 2 | 512 | 5.619 | 1.419 | 1.127 | 1.057 | 15/16 | 2.994 |

State ratios use cases that complete on both the off-grid and nx256 runs;
completion is evaluated separately. Stride 1 remains remarkably stable: the
same case 21 loses pressure positivity at call 49--50 on every resolution.
Cold stride 2 exposes a resolution-sensitive margin: case 190 completes all 50
calls at nx256 with minimum pressure 0.00193, but becomes nonpositive at calls
19 and 18 on nx128 and nx512. Thus the off-grid completion loss is small but
mechanistically real, not a mixed-prefix accounting artifact.

High-resolution shock degradation is also real rather than only primary-front
detector switching. At H20, stride-1 top-two position error rises from 0.0129
to 0.0274 and strength error from 0.344 to 0.732; cold stride 2 rises from
0.0172 to 0.0353 and from 0.353 to 0.827. Conserved-total error stays near its
nx256 value. The fixed 24-mode network therefore transfers the bulk field and
global budget much better than fine-grid shock geometry and strength.

The important positive result is that the D027 larger-step advantage transfers
on both new meshes:

| Grid | H20 stride-2 / stride-1 | H50 ratio | H100 ratio | H100 completion, stride 2 / 1 |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 0.968 | 0.762 | 0.474 | 15/15 |
| 256 | 0.750 | 0.638 | 0.514 | 16/15 |
| 512 | 0.815 | 0.643 | 0.523 | 15/15 |

The H50 and H100 paired-bootstrap intervals exclude one on both off-grid
meshes. The H20 interval narrowly includes one at nx128 but excludes one at
nx512. This supports a resolution-robust reduction in recurrent composition
error from the learned larger step, even though neither frozen network matches
the native off-grid one-step map well.

The label audit explains much of the negative transfer result. Along paired
native trajectories, the conservative one-frame increment mismatch between
nx128 and restricted nx256 is 2.70% of the next-state norm and 57.9% of the
coarse increment norm. Between nx256 and restricted nx512 it is 2.91% and
47.5%. For stride 2 the corresponding pairs are 3.85%/44.8% and 3.03%/28.3%.
The initial condition is also center sampled rather than an exact cut-cell
average; its restricted state mismatch reaches 5.66% on average for 256 to 128
and 2.09% for 512 to 256. The frozen FNO is therefore asked to reproduce a
different grid-dependent numerical flow map. Its cell features contain state
and position but no explicit cell width or timestep, and the fixed 24-mode
checkpoint does not exploit the added 512-cell shock resolution. The bandwidth
explanation is consistent with the result but is not isolated by D029.

The classification is
`stable_large_step_advantage_without_native_resolution_transfer`. Do not call
the current residual FNO mesh invariant, but do retain cold stride 2 as a strong
large-step baseline. Before stride 4 or another constraint sweep, construct a
restriction-consistent state/residual contract and test one shared
multi-resolution FNO with matched sample presentations. First alternate
uniform-resolution batches without adding architecture features; only then
ablate explicit cell-width/timestep channels or higher spectral bandwidth.

## 2026-07-16 D030 Restriction-Consistent Shared-Resolution Gate

D030 asks whether the D029 failure comes primarily from changing the numerical
flow map with the grid. Generate one 512-cell WENO-HLLC-ADER reference with
exact conservative cell averages at the initial discontinuity, then obtain the
256- and 128-cell trajectories only by finite-volume restriction. The primary
truth is this common restriction-consistent family. Separately generated native
coarse-solver trajectories remain a diagnostic and are never substituted for
the primary ground truth.

The shared row keeps the D027 stride-1 contract: a 316,419-parameter 64/24/4
FNO, conservative input/loss/recurrent coordinates, fixed physical scaling,
model seed `20260708`, split seed `20260707`, 64/16/16 cases, batch size 8,
learning rate `3e-4`, 50 one-step epochs, ten four-step recurrent epochs, H20
rollout selection, and raw inference. Each shared epoch has the same total
one-step or recurrent sample presentations as one single-resolution row; those
presentations are divided as evenly as possible among homogeneous 128-, 256-,
and 512-cell batches. Equal-presentation single-resolution models are fit as
same-grid oracle controls. No cell-width/timestep feature, additional Fourier
mode, limiter, positivity transform, viscosity, or smoothing is allowed.

Before full training:

- physical case parameters and saved times must match exactly across all three
  files;
- conservative restriction of the 512-cell states must reproduce both coarse
  files with global conservative relative L2 at most `1e-6` on the selected
  cases;
- restriction must commute with stride-1 and stride-2 increments in the same
  global norm to the same tolerance; per-case/time maxima remain diagnostics;
  and
- a 2-epoch 8/4/4 sanity row must remain finite, write a restorable checkpoint,
  and lower its training loss from the zero-update initialization.

The shared representation gate passes only if, on restriction-consistent test
truth at every resolution:

- one-step and H20/H50/H100 final state error are each at most `1.5x` the
  corresponding same-grid oracle;
- completion loses at most one of 16 cases and mean survival falls by at most
  `0.02` relative to that oracle; and
- H20 shock-position and conserved-total errors are each at most `1.5x` the
  oracle.

Improvement over the frozen native-nx256 checkpoint is a separate usefulness
gate: on restriction-consistent nx128 and nx512 truth, shared one-step error
must fall by at least 25%, H20 state error must not increase, and completion
must not decrease. Native-coarse evaluation is reported without imposing this
target-fit gate. Passing the primary gate but failing native-coarse equivalence
will be classified as `shared_restriction_operator_without_native_solver_equivalence`.
If same-grid oracles fit but the shared row fails, route next to explicit cell
width or higher bandwidth. If the oracles themselves fail, diagnose target fit
before changing the shared architecture.

Preflight amendment before any model result: primitive float32 serialization on
the 12-case synthetic smoke gives state global relative errors below `2.4e-8`
and stride-1 update global relative errors below `8.0e-7`, but a per-case/time
ratio reaches `3.29e-6` when its update denominator is small. The contract
therefore uses the global conservative relative norm at the original `1e-6`
tolerance and records the unstable local maximum separately. This changes the
norm definition, not the tolerance or any learned-model gate.

Execution amendment chosen before inspecting any learned metric: PowerShell
buffered the remote trainer's stdout, so stopped timing attempts were later
found to have written partial histories even though no epoch line had reached
the monitor. None of those metric values was read while choosing the changes;
the decisions used only wall time, GPU utilization, isolated phase benchmarks,
and equivalence tests. Training remains batch 8 with the same sample
presentations. Teacher-forced evaluation uses batch 128,
and the raw H20 checkpoint-selection rollout batches the same 16 cases within
each resolution. A focused equivalence test matches the original casewise
completion, survival, and rollout-error summaries. Final H20/H50/H100 research
evaluation remains casewise with the full shock, conservation, and positivity
diagnostics; this amendment changes execution overhead, not selection data or
the model/training contract. The same execution audit found repeated
GPU-to-CPU scalar reads inside every generic training batch. D030 therefore
uses the trainer's opt-in deferred-metric path, reducing detached loss and
relative-error tensors once per epoch and replacing its remaining host-side
finite-loss branch with a CUDA-stream asynchronous assertion. One-step and
recurrent equivalence tests match the original optimizer updates exactly and
the reported metrics within floating-point summation tolerance; a focused
nonfinite test still aborts, and the default trainer path is unchanged.
An exact-shape AutoDL benchmark rejected `torch.compile`: Inductor does not
generate complex FFT kernels here and was 1.8 times slower than eager
forward/backward. Eager optimization steps are already about 5.6 ms; the
remaining overhead came from the generic validation diagnostics. D030
checkpoint selection therefore also defers its state loss and relative-error
reduction to one synchronization per resolution. A focused test matches the
generic evaluator, while the final paper-facing evaluator remains unchanged.

### Result

The full AutoDL row completed under the ignored artifact root
`artifacts/time_dependent_no/d030_restriction_consistent_f64_20260716/`.
The fine file contains 512 trajectories, 101 saved frames, and 512 cells. The
shared and oracle rows all use the same 316,419-parameter 64/24/4 FNO, 6,400
one-step sample presentations per epoch, 6,208 recurrent-window presentations
per recurrent epoch, and the frozen 64/16/16 split. The shared row selected
epoch 58; the nx128/nx256/nx512 oracles selected epochs 56/60/60.

The exact-data gate is substantially tighter than required. Maximum analytic
initialization error is `2.93e-10`. State restriction and stride-1/stride-2
increment commutation are at machine precision; the largest reported global
relative error is `1.63e-15`, versus the preregistered `1e-6` tolerance.

On the common restriction-consistent evaluator, the shared row has one-step
relative L2 `0.004331/0.004040/0.004664` at nx128/nx256/nx512, versus
`0.003342/0.003524/0.003619` for the corresponding single-grid oracles. The
shared/oracle final-state results are:

| Grid | Horizon | Shared error | Oracle error | Ratio | Shared complete | Oracle complete |
|---|---:|---:|---:|---:|---:|---:|
| nx128 | 20 | 0.050765 | 0.054775 | 0.927 | 16/16 | 15/16 |
| nx128 | 50 | 0.102588 | 0.106591 | 0.962 | 14/16 | 13/16 |
| nx128 | 100 | 0.262138 | 0.244347 | 1.073 | 13/16 | 13/16 |
| nx256 | 20 | 0.048955 | 0.051311 | 0.954 | 16/16 | 15/16 |
| nx256 | 50 | 0.096231 | 0.083663 | 1.150 | 14/16 | 14/16 |
| nx256 | 100 | 0.254927 | 0.179770 | 1.418 | 13/16 | 13/16 |
| nx512 | 20 | 0.049461 | 0.053850 | 0.918 | 16/16 | 15/16 |
| nx512 | 50 | 0.096530 | 0.095497 | 1.011 | 14/16 | 14/16 |
| nx512 | 100 | 0.255026 | 0.182126 | 1.400 | 13/16 | 13/16 |

The worst state and one-step ratios are `1.418` and `1.296`; completion is
never lower than the oracle, the worst mean-survival deficit is `0.0075`, and
the worst H20 shock-position and conserved-total ratios are `1.123` and
`1.138`. The primary representation gate therefore passes. Against the frozen
native-nx256 checkpoint on restriction-consistent nx128/nx512 truth, shared
one-step error falls by 68.1%/78.3%, H20 error falls by 8.9%/13.3%, and
completion remains 16/16, so the separate usefulness gate also passes.

The native-coarse diagnostic does not pass equivalence. Moving the shared row
from restriction-consistent to independently evolved native truth multiplies
one-step error by `5.27x` at nx128 and `3.65x` at nx256, while changing it by
only `1.002x` at nx512. Native paired-grid state gaps are already several
percent and include frame-zero discretization differences. This supports target
inconsistency as a major cause of D029, rather than an FFT or shared-capacity
failure, but it does not establish one model for multiple numerical flow maps.

Raw recurrent stability remains unresolved. The same held-out cases 190, 238,
and 418 terminate on negative pressure across all three shared-grid rollouts at
approximately calls 21--25, 35--39, and 60--66. D030 therefore supports
resolution sharing for one identifiable restricted flow map, not positivity-free
H100 reliability. Do not add cell width or more modes merely to repair D029.
Route the next stability experiment to a separately controlled post-fit
tail-risk/admissibility stage on the strong residual baseline; do not repeat the
rejected blind-viscosity or implicit-interface rows unchanged.
