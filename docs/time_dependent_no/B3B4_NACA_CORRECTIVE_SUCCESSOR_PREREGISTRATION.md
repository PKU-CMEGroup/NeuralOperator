# B3/B4 NACA Corrective-Mechanism Successor

Updated: 2026-09-01

Experiment ID: `B3B4_NACA_CM_20260901A`

Status: owner-authorized implementation and open-population execution. The
prospective and sealed populations remain closed pending separate owner
decisions.

## 1. Scientific Role

This successor makes SU2 Unsteady NACA0012 the primary fixed-PCNO PDE case for
a revised question:

> Does the structured error created under clean self-composition contain a
> deployment-relevant response component, and do representative corrective
> mechanisms change their declared mediator and the resulting rollout?

The completed R0 verdict
`R0_REJECT_NACA_AS_HERO_CORRECTION_NECESSITY_CASE` is unchanged. R0 rejected
the stronger phenotype “accurate for a nontrivial early window and then
escapes.” This successor does not relabel that screen as a pass. It studies the
verified, narrower phenomenon: small teacher-forced next-state error followed
by severe finite rollout degradation with seed-consistent, grid-aligned error
bands.

The physical bands are not called state-space-normal or off-manifold before a
model-independent geometry assay supports that interpretation. Grid or
representation error remains a live alternative.

## 2. Fixed Information Boundary

- Solver, mesh, state, timestep, boundary conditions, normalization, PCNO
  architecture, optimizer family, and seeds inherit the verified R0 contract.
- The complete BDF2 state is
  `[Density, Momentum_x, Momentum_y, Energy, Nu_Tilde]` at two consecutive
  frames.
- Training uses only frames `955--1194`. Development uses only frames
  `1233--1472` for checkpoint selection, diagnostics, and open comparison.
- The prospective frames `1511--1750` remain unopened until a named reveal.
- The sealed/test frames `1755--1994` remain unopened until a later, separate
  owner decision.
- No arbitrary displaced-state SU2 restart has been qualified. Therefore this
  study does not claim to measure the trusted normal response
  `Phi(u + eta)`. SU2 supplies clean reference trajectories; displaced-input
  assays are model-side and offline.
- Deployed mechanisms use only the learned transition, the fixed training
  reference bank, and their declared random input. There is no online solver
  call, defect threshold, or OOD detector.

All successor source, data, parent-checkpoint, normalizer, evaluator, and
result identities must be hash-bound. The old execution authorization and old
source manifest are not reusable.

## 3. Fixed Training And Selection Contract

Use seeds `17`, `29`, and `43`, batch size `4`, `100` epochs, AdamW with initial
learning rate `1e-3`, and the inherited cosine schedule. Each learned arm is
initialized from the same seed-specific parameter state and receives the same
clean sample order. Select the earliest checkpoint attaining the lowest clean
development one-step normalized-residual RMSE. No rollout metric selects a
checkpoint, noise scale, rank, loss weight, or arm.

The model predicts a normalized residual. For a possibly displaced current
state `u_tilde_n`, recovery toward clean `u_(n+1)` therefore uses target

```text
(u_(n+1) - u_tilde_n) / residual_scale,
```

not the clean residual. Any implementation that leaves this displacement out
of the target fails the contract.

## 4. Retained Arms

### 4.1 `CLEAN`

The inherited teacher-forced residual objective, rerun under the successor
identity for exact paired initialization and source parity. The verified R0
checkpoints remain immutable parents and may be used only to construct the
train-only perturbation calibration below.

### 4.2 `IID_RECOVERY`

For every clean sample, average one clean loss and one recovery loss. Perturb
both BDF2 history states in state-normalized coordinates using independent,
zero-mean Gaussian noise. For each field, the standard deviation is the
train-only RMS teacher-forced error of the three frozen R0 predictors in that
field. The multiplier is exactly `1.0`; there is no scale search. Draw fresh
noise each epoch from a recorded deterministic stream.

This arm tests generic Gaussian recovery at a displacement scale supplied by
the baseline's own clean error rather than a rollout-tuned hyperparameter.

### 4.3 `ERROR_SUBSPACE_RECOVERY`

Form consecutive two-state teacher-forced error pairs from the three frozen R0
predictors on the training frames only, in state-normalized coordinates. Fit a
centered rank-`16` PCA basis before successor training. Sample a zero-mean
Gaussian in this basis and rescale its expected squared norm to equal the
`IID_RECOVERY` history-pair noise energy. Average one clean loss and one
recovery loss, with the same recovery target semantics as above.

This is the framework-derived method. It asks whether spending the same noise
energy along empirically observed model-error directions creates a more useful
controlled buffer than spatially independent noise. The basis, captured
variance, rescaling, parent hashes, and realized perturbation statistics are
recorded. Rank `16` is fixed and is not selected by rollout.

### 4.4 `DETACHED_PUSHFORWARD`

For each eligible center `n`, first compute

```text
u_hat_n = stop_gradient(Psi(u_(n-2), u_(n-1))).
```

The exposure input is `(u_(n-1), u_hat_n)` and its residual target is

```text
(u_(n+1) - u_hat_n) / residual_scale.
```

Average the clean and exposure losses equally. The first training center,
which lacks the preceding BDF2 pair inside the training role, receives only
the clean loss. Gradients do not pass through `u_hat_n`.

This is detached model-prefix exposure toward a stored clean future. It is not
dynamics relabeling at an arbitrary displaced state and is not described as
learning `Phi(u_hat_n)`.

### 4.5 `PATH_PROJECTION`

Attach one deterministic operational corrector to each `CLEAN` predictor. Fit
an ordered piecewise-linear reference path from normalized training states
only. Candidate nearest segments are selected in a train-PCA embedding; use
the smallest rank reaching `99.9%` training variance, capped at `32`, and
record whether the cap is active. The corrected state is the full-state linear
interpolant on the selected segment. Apply the corrector after every raw
prediction and feed the corrected state into the next BDF2 step.

The identity corrector must reproduce the raw recurrence. Report raw and
corrected predictions separately. This arm tests explicit empirical return;
its main risk is phase snapping or stable but wrong path following.

No diffusion refiner, learned sampler, solver-feedback corrector, dynamics
relabeling, architecture sweep, noise-scale sweep, or second PDE is added to
this deadline-bounded study.

## 5. Offline Diagnostics

Fit the reference geometry from clean training states independently of every
tested model. For common clean and one-prefix inputs, report:

1. clean teacher-forced forcing error;
2. the change in the next prediction caused by replacing the clean current
   state with the model-generated current state;
3. tangent and transverse components relative to declared finite-difference
   reference-path directions, only where those directions are numerically
   stable;
4. distance to the train-only piecewise-linear reference path;
5. projected-path discrepancy, kept separate from path distance;
6. graph-Dirichlet or equivalent declared high-frequency error energy to test
   the grid-aligned-ripple alternative; and
7. field error, positivity/finite-state checks, boundary-region error, and
   reference-relative integrated-state drift.

If local direction estimates are unstable, item 3 is dropped and the paper
uses finite-amplitude response language. A lower path distance is not a
success unless field and projected-path errors are not harmed.

## 6. Frozen Outcomes And Aggregation

Evaluate the common anchors at horizons `1`, `8`, `35`, `104`, and `208`, and
retain the full `1--208` traces. Primary rollout summaries are the normalized
state-error AUC and the late-window median over horizons `174--208`. Report
each seed and anchor. Because the anchors share one trajectory, use
deterministic finite-population summaries rather than IID confidence claims.

For a pairwise metric ratio in `[0.95, 1.05]`, record a tie. Cost includes
training calls, inference calls, stored reference-bank size, and wall time.

## 7. Signed Predictions And Falsifiers

These predictions are frozen before any successor-arm rollout is opened:

1. `CLEAN` will show a larger one-prefix response and transverse or
   high-frequency error component than its clean one-step error alone suggests.
2. `IID_RECOVERY` will reduce model-side displaced-input gain and reference-path
   distance relative to `CLEAN`, with a risk of damping path dynamics.
3. At matched expected noise energy, `ERROR_SUBSPACE_RECOVERY` will reduce
   response along baseline-error directions and late rollout error more than
   `IID_RECOVERY`.
4. `DETACHED_PUSHFORWARD` will reduce defect on model-prefix inputs and improve
   path accuracy relative to `CLEAN`, but need not be the most contractive arm.
5. `PATH_PROJECTION` will give the lowest post-correction path distance. It is
   a scientific failure, not a success, if this is accompanied by worse field
   or projected-path error.

The mechanism explanation fails for any arm that improves rollout without the
predicted mediator change. The broader C2 claim fails if the frozen diagnostics
do not distinguish intervention value beyond clean one-step error. A result in
which all learned arms retain the same structured grid mode supports a
representation-limited alternative and must be reported as such.

## 8. Execution And Reveal Gates

1. Implement focused target, perturbation, recurrence, projection, leakage,
   determinism, and identity-parity tests.
2. Pass an independent code-to-intent audit.
3. Run a full-resolution open-data resource smoke with exact successor code.
4. Freeze source, input-lineage, calibration, smoke, and execution manifests.
5. Train and evaluate only training/development data.
6. Close and independently verify the open result packet.
7. Freeze the prospective evaluator and prediction packet.
8. Request a separate owner decision before opening prospective data.
9. Request another separate owner decision before any sealed/test access.

The Friday `2026-09-04` target is a complete open-population PDE line,
prospective-ready freeze, figures/animations from permitted data, and an honest
manuscript section. Prospective or sealed claims are not fabricated if their
separate gates have not opened.
