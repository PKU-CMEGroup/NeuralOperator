# Handoff

Use this page when transferring work between human collaborators or AI agents.

## Current Objective

This week's active objective is Idea 2.1: the solver-facing representation
diagnostic. Use 1D Euler to isolate four design axes: state coordinates,
predicted quantity, supervision graph, and enforcement mechanism. The completed
primitive-loss target ladder is evidence and baseline history; it is not a
license to bundle all four axes into one new method. Medium-horizon open-loop
forecasting remains the gate before data assimilation.

Empirical target-family exploration remains central. The coordinate matrix and
the first flux-supervision matrix are causal screens within that program, not
replacements for it. Treat theory as a source of predictions and falsifiers;
classify both positive and negative results by label validity, target fit,
held-out fit, and induced rollout before accepting or rejecting a family.

## Current Diagnostic State

CPGNet did not reproduce the paper's Supersonic Bump Table 2 autoregressive rollout errors. Completed diagnostics show teacher-forced one-step errors near the paper scale, while autoregressive errors are roughly one to two orders of magnitude larger. The most defensible explanation is shock-local phase/shape/amplitude or multi-scale structure degradation under autoregressive rollout, not simple one-step underfitting, gross train-range drift, or small random-noise amplification.

The CPGNet interface-latent investigation concluded that `reconstruct_prims` behaves like a nonphysical flux-control coordinate rather than a physical one-sided trace. That code was retired from the active tree after the result was recorded; recover it from git history only if the exact probe must be rerun.

For PCNO, use `scripts/time_dependent_no/rollout_pcno_preprocessed.py`, not the retired raw-HDF5 graph adapter. The corrected PCNO rollouts show good early shock placement but visible Fourier-style rippling, followed by instability/crash in longer autoregressive rollout.

The active 1D Euler target-ladder work is the bridge from these diagnostics to method design. The current evidence says raw conservation structure is not enough: flux and interface objectives need admissibility, rollout, shock, conservation, and noise diagnostics before they can support a structure-preserving claim. The target ladder also makes learned quantities more interpretable for later Idea 2.2 investigation.

The two Section 1.2 rows formerly labeled CPGNet were produced by a generic
directed target head with `limited_residual`; they did not execute the paper's
interface reconstruction, Rusanov flux, and finite-volume update. They are now
deprecated. The corrected M0 path is `CPGNetEuler1D + cpg_interface`: positive
directed interface states, one shared face flux, exact 1D geometry, physical
ghost boundaries, no post-update limiter, and raw failure at the first
inadmissible recurrent state. AutoDL validation is complete. The paper-faithful
h128/mp12 reference remains locality-limited on the stride-4 dataset: 34/64
test rollouts complete and one-step relative L2 is 0.0334. Increasing only
unshared message depth to 28 gives one-step relative L2 0.00614 and 64/64 raw
completion, with final L2 0.0298 and no nonpositive recurrent states. This is a
strong baseline, but depth also raises parameters from 1.50M to 3.35M, so
matched-parameter controls remain before a causal depth claim.

The two parameter-matched controls are complete. The wide mp12 model does not
recover the deep model's behavior, while the narrow mp28 model retains the
important gain. Receptive-field depth, not parameter count alone, is therefore
the supported mechanism. Freeze h128/mp28 as the strong corrected CPGNet
reference and do not append another graph-depth or width sweep.

The mp28 mean shock error is tail-dominated. Replay of the three outliers shows
a persistent nearly stationary pressure gradient near the original
discontinuity while the target front travels. The argmax front metric magnifies
switching between two comparable gradients, but the ghost front is genuine.
Track the top two fronts; do not add blanket TV regularization to repair this
secondary baseline.

The accepted direction is recorded in
`docs/time_dependent_no/RESEARCH_DIRECTION_DECISION.md`. The four-way FNO
coordinate matrix is complete. Conservative input, conservative loss, fixed
physical scaling, and conservative recurrence are the provisional contract for
the target screen; this is a within-screen choice, not a universal claim that
primitive coordinates are unsuitable.

The prediction-target path is implemented and validated. The active
ADER generator can optionally save the owner-oriented conservative face-flux
impulse summed over every accepted CFL substep. Rejected retry attempts
contribute zero; the final accepted retry or first-order fallback contributes
exactly once. The loader aggregates those impulses over the requested macro
stride and exposes their time average to the existing conservative flux
decoder. The cumulative conservative face impulse is the solver-exported
semantic label; its time average is an exactly invertible normalization for
training. Do not confuse that label contract with the separate gauge-canonical
representative used in one loss ablation.
The FNO trainer now distinguishes state-only, direct-flux, and joint flux/state
supervision with component normalization. The state-loss-only flux row is the
underidentification/conditioning control, while direct and joint rows isolate
solver-facing supervision. Direct solver-flux labels
require clean inputs because adding input noise without recomputing the solver
flux makes the label inconsistent with the decoded update. The 512-case ADER
flux dataset has shape `[512, 101, 256, 3]` with `[512, 100, 257, 3]` face
impulses. It preserves the original state arrays bitwise, has no solver fallback,
closes before serialization to about `7.1e-14`, and has maximum float32 decoded
closure about `2.90e-5`.

The stride-1 supervision screen is complete through a midscale stop decision.
Two tiny-set seeds, a lower-learning-rate/longer control, and a matched
gauge-mode by joint-weight ablation established that raw reference-gauge fitting
is an optimization nuisance but not the main rollout bottleneck. Gauge-canonical
joint supervision with flux weight `0.1` was the only solver-flux row that
preserved tiny-set state fit while materially improving divergence-active flux
error, so it alone advanced with the state-loss-only flux head.

The promoted midscale screen used the same 316,739-parameter FNO 64/24/4,
64/16/16 trajectories, 6,400/1,600/1,600 one-step pairs, 50 epochs, and 40,000
optimizer updates per row. Gauge-canonical joint supervision reduced held-out
divergence-active flux MSE to `0.301x` state-only, but test one-step relative L2
rose from `0.00461` to `0.00623` (`1.350x`) and mean raw rollout survival fell
from `0.51875` to `0.300` (`0.578x`). Both rows completed 0/16 rollouts, and the
joint row failed earlier on every paired trajectory: 6.0 valid steps on average
versus 10.375. All failures were raw nonpositive states, predominantly pressure.
The lower joint prefix error, shock error, and total-state error are not
long-horizon wins because they are evaluated on a much shorter surviving prefix.

This result rejects full-scale promotion of this exact face-flux MSE objective;
it does not reject conservative flux-form FNOs or all solver-facing targets. In
1D, after removing the single constant-flux null mode, the face field is fixed
by its divergence. Gauge-canonical face-flux MSE is therefore an
inverse-divergence reweighting of the same state increment, emphasizing global
low-frequency modes. The observed active-flux gain with worse shock-adjacent
state and rollout behavior is consistent with that loss geometry.

The matched second-stage stability screen is complete. Both rows reproduced an
identical 50-epoch one-step trajectory, then reset AdamW at `3e-5` for ten
epochs of four-step conservative recurrence. Unroll-only improved test one-step
relative L2 from `0.004612` to `0.003438` and mean raw survival from `0.51875`
to `0.759375` (`1.464x`), with 2/16 completed rollouts instead of 0/16. It
survived longer on 14/16 paired cases, by a median five saved steps, and had
lower relative error on all 16 cases at their common endpoint. The
rollout-selected checkpoint is epoch 60, so the intervention passes its
predeclared promotion gate.

The weight-`0.1` training-only density/pressure barrier reached one-step
relative L2 `0.003418`, survival `0.75625`, and the same 2/16 completion and 14
nonpositive terminations. Its survival is `0.996x` unroll-only, with four cases
longer, eight tied, and four shorter. The weighted barrier was about 4.5% of
state loss in its first stage epoch and below 0.1% thereafter. This tested
barrier therefore fails attribution; do not carry it into the full-scale run.
Both midscale unrolled rows remain incomplete-horizon models.

The full 384/64/64 unroll-only confirmation is complete. It used the frozen
316,739-parameter FNO, 50 one-step epochs plus ten four-step recurrent epochs,
and no noise, limiter, floor, positive transform, or barrier. The selected
epoch-59 checkpoint reaches test one-step relative L2 `0.001305`, mean survival
`0.97734`, and 57/64 completed 20-call raw rollouts. All seven terminations are
nonpositive proposals. The initial effective CFL is `3.84` on average and
`5.64` at maximum. The run passes the original scale and 20-call promotion
gates but misses the strict 90% completion gate by exactly one trajectory. On
the same validation cases, four-step recurrence raises completion from 5/64 at
the best one-step checkpoint to 56/64 at the selected recurrent checkpoint.
This supports recurrent-distribution training as the decisive intervention on
this fixed dataset, but it does not establish unconditional positivity,
long-horizon robustness, or timestep/resolution transfer.

The frozen epoch-59 flux checkpoint has now been evaluated without retraining at
20, 50, and 100 raw calls. The 20-call metrics reproduce exactly. At 50 calls,
mean survival falls to `0.566`, only 1/64 cases completes, and 63 terminate on
raw nonpositive proposals. At 100 calls, 0/64 completes and every case has
terminated by call 51. The earlier 57/64 result is a strong short-horizon
baseline, not evidence of medium-horizon stability without positivity control.
Median termination is call 27; density is nonpositive in 56/64 cases and
pressure in 13/64, with five overlaps. This later population is predominantly a
density-undershoot failure, not simply the three early pressure/shock failures.

The strict target-parameterization screen now contains a faithful direct next
conservative-state head. Direct state and residual labels both pass float32
closure. Direct state fails the fixed tiny-fit gate on both seeds: minimum train
relative L2 is `0.01658` and `0.01711`, versus the `0.01` threshold.
Conservative residual passes on the first seed at `0.00648`; this is an
optimization/centering failure for direct state, not a rollout-based rejection.

The promoted residual row is now complete at full 384/64/64 scale. It uses the
316,419-parameter FNO cell head, the same 50+10 schedule and conservative
coordinate contract as the flux head, and no noise, limiter, floor, positive
transform, or admissibility penalty. The rollout-selected epoch-60 checkpoint
reaches test one-step relative L2 `0.001123`, final 20-call relative L2
`0.01215`, survival `1.0`, 64/64 completion, shock-position MAE `0.00428`, and
conserved-total error `0.00316`. Four-step recurrence lowers the best validation
20-call error from `0.01883` after one-step training to `0.01143` (`39.3%`) while
retaining full survival.

The frozen checkpoint also passes the actual horizon test: 62/64 trajectories
complete 50 calls with mean survival `0.990`, and 61/64 complete 100 calls with
mean survival `0.97781`. At 100 calls, completed-case final relative L2 has mean
`0.0551`, median `0.0482`, and p90 `0.104`; conserved-total error is `0.00927`
over all valid prefixes. There is no density or nonfinite termination. The three
failures occur at calls 33, 37, and 91 when pressure collapses near an interior
moving wave after velocity overshoot; valid length is essentially uncorrelated
with truth effective CFL (`r` about `0.01`). This supports longer generated-state
exposure and a training-only pressure/internal-energy penalty before any
inference limiter.

The residual advantage is paired, not just aggregate. At the 20-call common
endpoint it has lower relative error on 58/64 cases; the mean error difference
is `-0.0189` with trajectory-bootstrap 95% interval about
`[-0.0304, -0.00975]`. At the longer common endpoint, which is usually set by
the flux model's failure, residual is lower on 63/64 cases and its median error
ratio is `0.165`. The pressure-argmax shock metric has a known multi-wave caveat:
its largest residual outliers are rank swaps between two correctly located
gradient peaks, reflecting relative wave-amplitude error rather than a missing
or displaced front. Keep top-k wave matching alongside the legacy scalar.

The cell residual head is not structurally facewise conservative, so
conserved-total error remains an outcome metric. The next method should retain
this well-conditioned residual objective while projecting the increment onto a
boundary-budget-compatible conservative subspace and reconstructing one shared
canonical face impulse, rather than returning to full face-value MSE.

That projected-residual comparison is now complete over seeds `20260707--09`.
The 316,806-parameter model enforces decoded closure to its learned boundary
budget at about `1e-7` and predicts that budget with mean one-step relative L2
`0.00722`. It improves pooled completion from 182/192 to 187/192 at 100 calls,
but does not improve accuracy robustly: the three-seed mean 100-call state-error
ratio is `1.016`, and only one seed improves state error. Among the 181
seed-case pairs completed by both methods, projected residual is 9.8% worse in
state L2, 5.9% better in conserved-total error, and 4.8% worse in shock MAE.
Only stability noninferiority passes the preregistered family gates. Keep raw
residual as the accuracy baseline and projected residual as a stability/
conservation ablation; neither result supports a general positivity claim.

The first generated-state exposure pilot is also complete on projected
residual at 64/16/16 scale. Its 50 one-step epochs reproduce the matched clean
control exactly. The clean second stage uses four-step BPTT; the intervention
uses eight detached model steps followed by the same four supervised steps,
with 10 versus 11 recurrent epochs to match optimizer updates within 1%. Both
rows select recurrent epoch 58. Burn-in changes 20/50/100-call state-L2 ratios
to `1.021`, `0.768`, and `0.638`; conserved-total ratios are `0.924`, `0.659`,
and `0.537`; shock ratios are `1.382`, `1.068`, and `0.936`. At 100 calls both
complete 14/16, but survival falls from `0.9650` to `0.9394`. It rescues case 21,
which fails under the control at call 96, while creating a call-58 failure on
case 234 and advancing case 238's failure from call 50 to 47. Training burn-in
states remain admissible: zero nonpositive samples, minimum density `0.0831`,
and minimum pressure `0.0675`. Classify this as partial evidence that later-
state exposure repairs broad error drift without controlling rare long-horizon
failure basins. Do not full-scale the projected row. The completed matched
plain-residual gate is recorded next.

The matched plain-residual exposure gate is now complete at the same 64/16/16
scale and seed. Its one-step histories again match exactly; the clean row uses
7,760 recurrent optimizer updates and the eight-step generated-burn row uses
7,832. Generated exposure lowers final state L2 by 4.6% at 20 calls and 32% at
50 calls. At 100 calls it lowers common-endpoint state error by 51.4%, wins
14/16 paired cases, and changes completion from 15/16 to 16/16. Conserved-total
error falls by 7.7%, 40.8%, and 47.4% at 20/50/100 calls. The formal gate still
fails because legacy pressure-argmax shock MAE rises 45.6% at 20 calls.
Focused replay attributes most of that aggregate increase to three cases:
two contain front-strength rank changes, while case 238 has a genuine
spurious/displaced-front regression. Keep the formal failure; report separated
top-two front position and strength metrics in subsequent gates.

The teacher-offset mechanism control is complete. It uses the identical 8+4
windows and 7,832 recurrent optimizer updates but initializes the supervised
segment from the exact offset state. One-step histories match both frozen
controls exactly; its prefix error is zero and all prefix states are
admissible. Teacher offset is 10.9% worse than clean at 20 calls, essentially
tied at 50/100 calls, and changes H100 completion from 15/16 to 14/16. It
postpones clean case 21's pressure failure from call 49 to 51 but introduces a
case-114 pressure failure at call 89.

Generated exposure decisively beats that time-offset control. Its
generated/teacher common-endpoint state-error ratios are `0.861`, `0.661`,
and `0.492` at 20/50/100 calls; trajectory-bootstrap 95% intervals are
approximately `[0.782,0.948]`, `[0.470,0.854]`, and `[0.348,0.669]`.
It wins 13/16, 13/16, and 15/16 cases and completes all H100 cases versus 14/16
teacher. Conserved-total error improves by 24%, 46%, and 50%. This supports
off-manifold generated-state exposure as a real mechanism beyond later-time
sampling on this fixed split and seed. The one-step ordering is opposite:
teacher reaches `0.002660` versus generated `0.002886`, reinforcing that
one-step fit does not select the better recurrent operator.

Do not full-scale yet. Against clean, generated exposure still worsens separated
top-two front-position error by 17.8% at H20 and 10.2% at H50; both paired
bootstrap intervals exclude one. D023 is complete. At generated-prefix depth
eight, generated/teacher fixed-scale conservative errors are `0.992` to the
original next state and `1.017` to the same-state WENO-HLLC-ADER continuation;
clean is `0.974` of generated on the solver-continuation metric. The generated
correction-to-original-trajectory cosine is only `0.074`. Its shock-region
solver defect is 6.3% worse than teacher and 8.0% worse than clean. This rejects
a large local PDE-map improvement and does not meet the preregistered strong
denoising threshold. The long-rollout gain is more consistent with a small
closed-loop modified-dynamics or self-consistency effect. Keep full-scale
promotion paused and run the frozen scale/spectral diagnostic next.

D029 is complete and separates two claims that were previously entangled. The
frozen stride-1 and cold stride-2 residual FNOs do not satisfy native-grid
zero-shot resolution transfer: native one-step error rises by roughly 5.5 to
8.3 times at 128 and 512 cells, and the 512-cell H20 shock error rises by 2.10
times for stride 1 and 2.99 times for stride 2. This is a persistent discrete
closure mismatch rather than a first-frame artifact. Along separately evolved
paired native trajectories, update labels differ by 28% to 58% of the coarse
increment norm, while the current inputs contain absolute cell centers but no
cell width or timestep channel. This is strong evidence of grid-dependent
labels, not an exact same-input operator commutator measurement.

The large-step result is nevertheless robust. At 128 and 512 cells, cold
stride 2 beats stride 1 on common-case H50/H100 state error, with H100 ratios
of 0.474 and 0.523 and no lower common-case completion. The correct
classification is therefore stable large-step advantage without native
resolution transfer. Next build restriction-consistent cell-average targets
and train one shared multi-resolution FNO before changing architecture.

## Active Code Surface

Reusable utilities are under `utility/time_dependent_no/`. The active 1D Euler target-ladder utilities are:

- `utility/time_dependent_no/fv.py`
- `utility/time_dependent_no/euler1d.py`
- `utility/time_dependent_no/euler1d_data.py`
- `utility/time_dependent_no/euler1d_models.py`
- `utility/time_dependent_no/euler1d_targets.py`

Active 1D Euler scripts are:

- `scripts/time_dependent_no/euler1d_weno_hllc_rk3_dataset.py`
- `scripts/time_dependent_no/euler1d_weno_hllc_ader_dataset.py`
- `scripts/time_dependent_no/train_euler1d_target_ladder.py`
- `scripts/time_dependent_no/analyze_euler1d_target_ladder.py`
- `scripts/time_dependent_no/build_euler1d_restriction_family.py`
- `scripts/time_dependent_no/train_euler1d_multiresolution.py`
- `scripts/time_dependent_no/evaluate_euler1d_resolution_transfer.py`
- `scripts/time_dependent_no/diagnose_euler1d_generated_state_consistency.py`
- `scripts/time_dependent_no/diagnose_euler1d_scale_spectra.py`
- `scripts/time_dependent_no/probe_euler1d_conservative_dissipation.py`
- `scripts/time_dependent_no/generate_euler1d_rollout_animations.py`

Active 2D diagnostic scripts are:

- `scripts/time_dependent_no/run_euler_fixture_diagnostics.py`
- `scripts/time_dependent_no/diagnose_cpg_rollout_mechanisms.py`
- `scripts/time_dependent_no/visualize_official_cpg_rollout.py`
- `scripts/time_dependent_no/visualize_cpg_shock_overlays.py`
- `scripts/time_dependent_no/rollout_pcno_preprocessed.py`

Generated reports, HDF5 rollout arrays, GIFs, checkpoints, and large logs remain ignored under `artifacts/time_dependent_no/`.

## Key References

- `docs/time_dependent_no/README.md`
- `docs/time_dependent_no/RESEARCH_DIRECTION_DECISION.md`
- `docs/time_dependent_no/CPG_EULER_DATASET_CONTRACT.md`
- `docs/time_dependent_no/MECHANISTIC_DIAGNOSTIC_TRACKER.md`
- `docs/time_dependent_no/MECHANISTIC_DIAGNOSTIC_PLAN.md`
- `docs/time_dependent_no/BUMP_300_DATASET_AUDIT.md`
- `docs/time_dependent_no/CPGGNSPDES_REFERENCE_AUDIT.md`

## Next Steps

1. Keep h128/mp28 frozen as the strong corrected CPGNet reference. The matched
   controls support a receptive-field explanation; do not run another CPG
   architecture sweep.
2. Keep the conservative-input/conservative-loss/recurrent contract frozen for
   the remaining strict target comparisons. Record it as provisional rather
   than claiming universal coordinate superiority.
3. Stop scaling the tested gauge-canonical joint face-flux MSE row. Preserve it
   as a label-valid negative result: it fits the identifiable flux target but
   loses decoded-state accuracy and raw rollout survival at midscale.
4. The matched second-stage stability screen is complete. Unroll-only passes;
   the tested admissibility barrier does not. Preserve inference as the exact
   unmodified flux divergence with no limiter, floor, or positive transform.
5. Freeze the completed flux-form row as a strong 20-call result and a negative
   50-call stability result. Do not call 57/64 at 20 calls a medium-horizon or
   positivity-preserving baseline; only 1/64 reaches call 50.
6. Stop the uncentered direct-next-state row in the strict matched regime after
   its two-seed tiny-fit failure. Treat conservative residual as the surviving
   information-equivalent parameterization; an explicit identity bypass would
   make direct state algebraically a residual head rather than rescue a distinct
   target.
7. Freeze the completed three-seed residual FNO as the strong fixed-setting
   accuracy baseline. Its 100-call pooled completion is 182/192; report both
   mixed-prefix and completed/common-case metrics.
8. Freeze projected residual as a structural ablation, not the new accuracy
   baseline. It improves pooled 100-call completion to 187/192 and exact budget
   closure, but fails the state, conservation-robustness, and shock family gates.
9. The teacher-offset 8+4 control is complete. Later-time sampling alone does
   not improve H50/H100 error, while generated exposure decisively does, so the
   off-manifold intervention is active on this split. D023 is complete and does
   not find a material same-state PDE-map gain: generated is 1.7% worse than
   teacher against the solver continuation at depth eight while only 0.8%
   better against the original trajectory. Treat the rollout gain as a
   closed-loop dynamics effect. D013 finds recurrent high-frequency growth but
   no teacher-forced spectral-fit deficit, and D024 rejects blind conservative
   Laplacian viscosity as its remedy. Keep full-scale burn-in paused because
   H20/H50 top-two front position remains worse than clean.
10. D013 is complete for 1D Euler. The state-loss-only flux head has
    teacher-forced high-band fit comparable to residual, but its failure state
    has 10.4 times the matched residual high-band error and 154 times its
    second-difference error. Error begins shock-local and spreads into smooth
    cells. Keep any later 2D extension graph-native.
11. D024 is complete and negative. No tested small interior diffusive flux
    improves H50 completion; small coefficients worsen state and Nyquist-tail
    error, and larger coefficients shorten survival. Do not promote global
    smoothing or repeat blind current-state viscosity sweeps.
12. D025 is complete and fails its successive-halving gate. The tested global
    relative-interface FNO reaches selected test one-step relative L2 `0.00786`,
    but uniform two-step training, frame-zero weight `15`, and a training-only
    barrier all give 0/16 H50 completion and mean survival below `0.05`.
    Frame weighting improves the initial shock/front error substantially, so
    rarity was real but not sufficient. Four-step loss becomes nonfinite only
    after recurrence has already produced negative density/pressure; raw model
    outputs remain finite. Do not full-scale or add an inference limiter.
13. The pressure/internal-energy-loss control is complete for D025. Barrier
    weight `0.1` drives its two-step training violation from `0.00443` to about
    `1e-7`, but does not improve held-out H50 survival. Do not sweep blanket TV,
    smoothing, or more mean barrier weights. Route new conservative-target work
    toward centered residual/projection or identifiable cumulative-impulse
    supervision rather than another underidentified interface-state row.
14. D026 is complete at the matched 64/16/16 gate. Directly supervising the
    identifiable net solver boundary exchange improves boundary and
    conserved-total errors at weights `0.01` and `0.1`, but worsens one-step,
    H20 state, and shock errors while leaving completion at 16/16. Treat it as
    a negative Pareto ablation; do not search more weights or run full seeds.
15. D027 is complete and partially positive. Cold stride 2 beats stride-1
    composition by 25.0%, 36.2%, and 48.6% in common-case H20/H50/H100 state
    error and finishes 16/16 at H100, despite mean truth effective-CFL maxima
    near 9.9. It misses only the direct frame-2 gate at 1.216 times composition.
    Preserve it as the large-step accuracy baseline, not a timestep-robust model.
16. D028 is complete and partial. Weight-only continuation fixes frame 2 and
    lowers final-target training floors by more than 40%, but H50 state and
    shock errors regress to 1.095 and 1.565 times cold. Do not run the
    conditional total-exposure control or claim continuation-sensitive
    optimization.
17. D029 is complete and partial. Neither frozen checkpoint passes native-grid
    zero-shot transfer: one-step error is 5.5 to 8.3 times its 256-cell value,
    and the 512-cell shock metric regresses. Cold stride 2 still beats stride 1
    at H50/H100 on both off-grid resolutions, with no lower common-case
    completion. Preserve the large-step result, but do not call the present
    residual FNO mesh invariant.
18. D030 is complete and partial. Exact cell-average restriction and stride-1/
    stride-2 update commutation pass at machine precision. The equal-presentation
    shared 128/256/512 residual FNO passes every same-grid-oracle gate and cuts
    frozen off-grid one-step error by 68.1%/78.3% at nx128/nx512 on common
    restriction-consistent truth. It does not match independently evolved
    native nx128/nx256 flow maps, so preserve the classification
    `shared_restriction_operator_without_native_solver_equivalence`; do not add
    cell width, timestep, or modes as a response to D029 alone.
19. The shared row still completes only 13/16 at H100. The same cases fail by
    negative pressure at similar calls on all three resolutions. Predeclare a
    post-fit tail-risk/admissibility stage on the strong residual baseline with
    an equal-update, equal-generated-exposure zero-penalty control. Change one
    constraint family at a time, keep inference raw, and do not repeat blanket
    Laplacian smoothing or the failed implicit-interface parameterization.
20. Evaluate every finalist with teacher-forced one-step, direct horizon, at
    least 100 raw autoregressive calls, timestep/resolution transfer, and a
    coarse-CFD comparison. Only after this gate should the method transfer to
    CPG geometry and a dynamic 2D shock benchmark.

## Agent Execution Rules

- Read `RESEARCH_DIRECTION_DECISION.md` before changing the trainer, target
  adapters, generator contract, or experiment matrix.
- Change one causal axis per primary comparison. A multi-component engineering
  prototype is not a substitute for the coordinate and supervision diagnostics.
- Record input/recurrent coordinates, predicted quantity, loss coordinates,
  normalization, timestep, effective CFL, geometry scaling, and all hidden
  floors or clamps in every run summary. Also record the comparison regime,
  canonical label semantics, closure error, initialization stride, stride
  curriculum, tiny-set fit result, and final failure classification when
  applicable.
- Predeclare closure and tiny-set pass thresholds, per-row diagnostic and
  optimizer budgets, data exposure, and the failure-repeat seed before a target
  batch starts. Do not infer gates from completed results.
- Do not conclude that a target family failed from rollout alone. Separate
  label/implementation failure, inability to fit, continuation-sensitive
  optimization, train/validation generalization, conditional closure ambiguity,
  and recurrent stability or distribution-shift failure.
- Preserve one primary inference path. Do not add independent state and flux
  heads that can disagree; compute derived states through the conservative
  update.
- New auxiliary labels require a documented source, identifiability argument,
  inference role, normalization, and ablation. Do not launch a broad dense-loss
  sweep by default.

## 1D Euler Experiment Todo

This section preserves experiment history. Unchecked historical items are not
automatically approved work; the ordered `Next Steps` and accepted research
direction above take precedence.

Completed follow-up batch on AutoDL:

- [x] Analyze `euler1d_noise_followups_v2`. Use only `v2` output directories; the earlier non-v2 launch was stopped because it did not pass the intended `--seed` argument.
- [x] Check seed stability for `FNO + limited_residual + stride 4` at noise `0`, `0.003`, and `0.02` over seeds `20260707`, `20260708`, and `20260709`.
- [x] Test whether noise can replace the limiter: compare `FNO + residual + stride 4` at noise `0` and `0.02` against `limited_residual`.
- [x] Test whether denoising helps shorter-step accumulation: compare `FNO + limited_residual + stride 2` at noise `0`, `0.003`, and `0.02`.

Result summary: over three stride-4 seeds, noise `0.003` and `0.02` both improved final rollout L2 by about 12% relative to zero noise. Noise `0.003` is the safer default for the next selector because it gives the best mean rollout/conservation without pressure-floor hugging; noise `0.02` is a useful shock-stability stress setting because it sharply improves shock MAE but worsens one-step error and sits at the pressure floor. Stride 2 also needed noise: zero-noise stride 2 had raw-pressure violations, while noise `0.02` greatly improved final, conservation, and shock metrics.

Ignored compact analysis artifacts are under `artifacts/time_dependent_no/euler1d_noise_followups_v2_analysis_20260709/`.

Target-objective extensions to implement after the active batch:

- [x] Select one noise level from the active follow-up before expanding the target matrix: use `0.003` as the default nonzero-noise selector value, and keep `0.02` only as a shock-stability stress setting.
- [x] Factor a shared conservative admissibility limiter for decoded updates so residual, flux, and interface adapters can all keep updated density and pressure admissible. Flux/interface targets use samplewise scaling to preserve finite-volume face-pair accounting.
- [x] Add `limited_flux`: predict face fluxes, apply the finite-volume update, then limit the induced conservative update to keep updated density and pressure admissible.
- [x] Add `positive_limited_interface`: decode positive interface density/pressure, compute Rusanov fluxes, apply the finite-volume update, then limit the updated cell averages.
- [x] Keep training noise objective-independent: perturb only the current primitive input during training, keep the clean next state as the supervised target, and evaluate clean rollouts.
- [x] Log limiter diagnostics for all limited objectives: activation fraction, mean/min `theta`, pressure-floor cases, and whether the limiter preserved the intended conservation accounting.
- [x] Run the first extended target selector on `FNO` at stride 4 with only noise `0` and `0.003`: `limited_residual`, `limited_flux`, and `positive_limited_interface`. Result: only `limited_residual` remained viable; current absolute flux/interface targets failed under rollout.
- [ ] Fully seed-confirm only the best one or two stabilized target variants;
  losing variants receive the required predeclared diagnostic repeat, not a
  full seed sweep.
- [x] Completed the corrected `cpg_interface` CPGNet baseline and matched
  parameter controls. mp28 is the frozen strong reference; do not spend GPU on
  the deprecated CPG residual head or another depth sweep.
- [ ] Generate animations for the best and worst noise/limiter cases, especially trajectories where pressure reaches the limiter floor.

Flux-indeterminacy and Hodge-style follow-ups:

- [x] The ADER generator exports the solver's exact cumulative conservative face
  impulse and validates endpoint/boundary closure. The raw/gauge-canonical and
  direct/joint D021 screens are complete through the midscale stop. Do not scale
  the rejected face-value MSE objective or treat flux reconstructed only from
  state differences as a unique physical target.
- [x] Add a physical-flux-plus-correction objective before more exotic gauges: `F_pred = F_Rusanov(current) + delta_F`, with regularization/diagnostics on the learned correction rather than on the whole physical flux.
- [ ] For 2D only, test curl or loop-circulation regularization on the learned flux correction after mesh face orientations, areas, and cell volumes are validated.
- [ ] Treat scalar-potential flux prediction as an ablation rather than the first mainline method, because a gradient-only gauge may over-constrain shocks and multidimensional flow.
- [ ] Do not combine Hodge regularization with the broad noise/positivity target ladder until the stabilized FNO selector identifies at least one viable target family.


Physical flux-correction status:

- [x] Implemented `physical_flux_correction` as current-state Rusanov base flux plus bounded learned correction, decoded by the FV update and shared samplewise limiter.
- [x] Added live correction diagnostics: correction/bound mean, correction/bound max, and saturation fraction.
- [x] Ran zero-correction base audit on the real 1D dataset; stride-4 base Rusanov is a poor macro-step anchor and heavily activates the rollout limiter.
- [x] Finished the short correction-scale probe over scales `1`, `2`, and `4`; all variants were rejected, so do not run the full noise selector for this exact macro-step Rusanov-base target.
Engineering controls to consider after the selector:

- [x] Replace the mislabeled CPG residual head with the solver-level paper adaptation: geometry-only edge encoder, 12 directed flow layers, positive interface decoding, shared Rusanov flux, exact 1D geometry, physical ghosts, and 15+5 training.
- [x] Evaluate that baseline without a cell-state limiter or recurrence floor; mp12 completes 34/64, while the mp28 receptive-field control completes 64/64 with clean raw positivity.
- [ ] For FNO, keep fixed-step operators without `dt` parameterization and keep identity/zero-update initialization for residual-like heads; treat spectral smoothing or high-frequency penalties as later ablations, not part of the first target selector.

## Do Not Do

- Do not touch collaborator-owned checkpoint or preprocessing artifacts unless explicitly asked.
- Do not use the retired raw-HDF5 PCNO adapter path for model conclusions.
- Do not launch broad new training sweeps before ripple and shock-position
  diagnostics identify the mechanism to target. This does not block the bounded,
  gated target-family screen approved in `RESEARCH_DIRECTION_DECISION.md`.
- Do not combine coordinate, target, dense-supervision, constraint, and
  architecture changes in one primary experiment.
- Do not impose blanket componentwise TVD or global spectral/variation damping
  on Euler without a smooth/shock-region contract.
- Do not make mesh-weighted conservation claims until cell/mesh weights are validated.
- Do not treat boundary leakage as evidence of model quality under clamped official rollout.
- Do not commit private paths, credentials, raw data, checkpoints, extracted rollout arrays, heavy figures, or local machine paths.
