# M1 Kolmogorov Information-Source Comparison Preregistration

Date: 2026-08-26

Status: **A1 DESIGN AND SYNTHETIC PLUMBING ONLY**. This document does not
authorize scientific solver execution, dataset generation, checkpoint access,
training, remote execution, or sealed evaluation. Every later stage requires
the named authorization in the execution ladder.

## 1. Decision Being Registered

The experiment asks one narrow question:

> When a learned evolution map visits displaced states, does a trusted
> dynamics label `Phi(x)` provide rollout-relevant information beyond clean
> one-step supervision, detached stored-target exposure, and solver-free
> recovery toward the clean trajectory?

The primary contribution is an information comparison, not a new backbone.
One fixed residual FNO is used throughout. Architecture, optimizer, data,
normalization, parameter count, clean-pair exposure, update count, and
checkpoint rules are held fixed as far as the intervention permits.

### Claim map

| ID | Registered claim or question | Minimum convincing evidence |
| --- | --- | --- |
| M1-C1 | Dynamics-consistent labels can matter when clean or recovery targets do not identify the deployed off-state response. | On an identical off-state input bank, the dynamics-labelled arm improves both `Psi(x)` versus `Phi(x)` and autonomous rollout relative to the recovery-target arm, across three seeds, without material clean/tangent harm. |
| M1-C2 | A deployment-response observable can explain rollout differences left unresolved by clean one-step error. | A frozen response observable improves cross-fitted prediction/ranking of H64 rollout beyond clean one-step error and moves in the same direction as the successful intervention. |

The decisive anti-claim is that any gain comes from different inputs, more
parameters, more optimizer steps, a different normalization, projection, or
checkpoint selection. The common-bank recovery/dynamics pair exists primarily
to rule out that explanation.

### Outcomes that are scientifically useful

- If dynamics labels beat the identical-input recovery arm, proceed to one
  selective cached-label correction under M2.
- If recovery or stored-target pushforward matches dynamics labels, prefer the
  solver-free mechanism and do not claim that solver relabeling is required.
- If all arms match, the tested regime supplies no empirical need for an
  off-state intervention; either the clean map is adequate or the assay is too
  weak.
- If off-state fidelity improves without rollout improvement, the measured
  response defect is not sufficient for the deployed horizon.
- If rollout improves while clean/tangent behavior regresses materially, the
  result is a stability--accuracy trade rather than a successful correction.

## 2. Why This Testbed

The mechanism test uses damped, forced two-dimensional Kolmogorov flow in
vorticity form on the periodic square. It is selected because:

1. scalar vorticity is a closed Markov state once forcing, viscosity, drag,
   clock, and grid are fixed;
2. arbitrary finite displaced states can be advanced without a graph-to-solver
   inverse;
3. periodicity removes boundary reconstruction as a competing explanation;
4. the flow can exhibit long-horizon turbulent decorrelation while retaining
   interpretable energy, enstrophy, palinstrophy, and spectral diagnostics; and
5. a standard public regime uses `nu=0.001`, forcing proportional to
   `sin(4y)`, linear drag `0.1`, and a 16-native-step prediction interval in
   [PDE-Refiner](https://openreview.net/forum?id=Qv6468llWS). The present study
   does **not** reproduce that paper's 2048-grid finite-volume DNS or its
   released data.

The disqualified dynamic-FV native-coarse map is not reused: P0-RS-A2 showed
that its bias is comparable to the model defect and between-model separation.
REALM is also not the primary mechanism test because the released fixed data do
not provide arbitrary-state solver labels.

## 3. Reference-Map Contract

The state is a real array

    omega in R^(N x N)

representing a mean-zero, 2/3-dealiased Fourier polynomial on `[0, 2 pi)^2`.
Axis order is `(x, y)`. The fixed-grid reference equation is

    omega_t + u dot grad(omega)
      = nu Laplacian(omega) - alpha omega - A q cos(q y),
    u = (psi_y, -psi_x),
    -Laplacian(psi) = omega.

For the registered candidate, `N=64`, `nu=0.001`, `alpha=0.1`, `A=1`,
integer forcing mode `k=4`, `q=k`, and macro interval

    Delta T = 16 * 0.0070125 = 0.1122.

The numerical map `Phi_64` is adaptive-step classical RK4 with a maximum
substep `0.002`, CFL factor `0.4`, and a 2/3 mask applied to every accepted
stage output. The solver is FP64. This is a declared finite-dimensional
reference map, not the continuum Navier--Stokes semigroup.

The canonical projection `P` removes the mean and discarded Fourier modes.
It is never hidden:

- trusted states use `advance_canonical` and fail if their relative projection
  change exceeds `1e-11`;
- deliberately arbitrary proposals use `advance_projected`, and the projection
  norm is recorded;
- the deployed learned transition is evaluated both before and after `P`;
- the primary comparison uses canonical common-bank inputs so `P` cannot create
  an arm-specific input advantage.

Maintained source surfaces:

- `utility/time_dependent_no/kolmogorov_reference.py` -- reusable reference map;
- `tests/time_dependent_no/test_kolmogorov_reference.py` -- synthetic contracts;
- `scripts/time_dependent_no/run_m1_kolmogorov_a1_preflight.py` -- synthetic-only
  dry run.

The older exploratory generator under `scripts/navier_stokes_square/` is not a
scientific source binding for M1.

## 4. Qualification Before Data Or Models

No arm may train until the following results are frozen on an open calibration
population.

### Q0: analytic and plumbing closure

Required:

- Taylor--Green vorticity follows its exact viscous-plus-drag exponential
  decay;
- the configured laminar Kolmogorov solution remains steady;
- canonicalization is mean-zero, dealiased, idempotent to FP64 tolerance, and
  explicit;
- same-process restart is bitwise repeatable;
- single-step and repeated-rollout paths close;
- shape, dtype, nonfinite, noncanonical, and substep-limit drift fail closed.

Current A1 status: all 12 focused CPU tests pass and the synthetic preflight
closes. This is implementation evidence only.

### Q1: numerical-reference qualification

Under a separately authorized small scientific execution:

1. **Time refinement:** compare `dt_max={0.002,0.001,0.0005}` on fixed clean
   and displaced states. The candidate must have median/family-maximum one-step
   relative differences below `1e-5/1e-4` against `0.0005`, and H16 differences
   below `1e-3/5e-3`. At H64, where chaotic phase separation may make pointwise
   convergence inappropriate, kinetic-energy, enstrophy, and shell-spectrum
   distribution discrepancies must each remain below 2%.
2. **Repeatability:** two fresh FP64 processes must agree below scaled RMS
   `1e-13`; accepted substep counts must agree exactly.
3. **State closure:** all outputs remain finite and canonical; no input
   projection exceeds the declared tolerance unless the projected API was
   requested.
4. **Spatial context:** zero-pad canonical 64-grid Fourier states to 128,
   advance them under the correspondingly refined solver, and truncate them
   back. This discrepancy is reported, not silently called ground truth.
   Before a PDE-level claim, its median and maximum must each be below one
   quarter of both the clean-model defect and the recovery-versus-dynamics
   target separation on the matched bank. Otherwise claims remain explicitly
   about `Phi_64`, or the resolution is increased before training.
5. **Stationarity:** after the frozen burn-in, first-half versus second-half
   means of kinetic energy and enstrophy differ by at most 10% on the
   calibration population, with no monotone drift shared by more than 75% of
   trajectories.

Failure stops M1 at qualification. Parameters are not tuned after arm outcomes.

### Q2: phenomenon qualification

Using only the open calibration/validation split, the clean baseline must:

- attain validation one-step normalized RMSE at or below `0.05`;
- remain finite and complete through H128; and
- exhibit nontrivial composition difficulty: median H64 error at least three
  times H1 error or a predeclared H64 structure drift event in at least half the
  validation trajectories.

If the setting is too easy, one predeclared adjustment may be chosen using only
calibration data: increase the rollout horizon to H256 or reduce viscosity to
`0.00075`. If it is numerically unqualified, increase resolution or stop. The
choice is frozen before any information-arm result is opened.

## 5. Data And Split Contract

Subject to Q1 approval, generate independent trajectories from the qualified
reference map:

| Split | Trajectories | Use |
| --- | ---: | --- |
| train | 128 | optimization and train normalizers only |
| validation | 16 | recipe choice, checkpoint selection, and gate decisions |
| development audit | 16 | common-bank and diagnostic development before freeze |
| test | 32 | one final sealed comparison after explicit authorization |

Each retained trajectory contains 129 macro states after a qualified burn-in.
Initial-condition seeds, burn-in length, every solver/config hash, and split
membership are written to immutable manifests before training. No adjacent
states cross split boundaries. Normalization uses train trajectories only.

The model predicts the normalized residual `omega_(t+1)-omega_t`; all losses
and metrics reconstruct the next physical state before comparison. Coordinates
are deterministic and are not normalized with state statistics.

## 6. Fixed Model And Recipe Selection

The candidate backbone is one periodic residual FNO:

- input: normalized vorticity plus `(x,y)` coordinates;
- output: one normalized vorticity increment;
- four Fourier blocks, width 64, `16 x 16` retained modes, GELU, pointwise
  branch, and a width-128 output MLP;
- no padding, projection, filtering, gradient branch, attention, memory, or
  architecture-specific corrector inside the trainable network.

Before arm outcomes, a clean-only calibration may choose one of at most three
registered learning-rate schedules. The winning recipe is selected by mean
validation H64 error, then H1 error, then earlier checkpoint. Architecture,
normalizer, optimizer family, update budget, batch size, initialization seeds,
and checkpoint cadence are then locked for all arms.

The registered maximum is 20,480 updates with checkpoints every 1,280 updates.
A 40,960-update continuation is not automatic. Three training seeds are
required for the confirmatory comparison.

## 7. Four Information Arms

Every minibatch contains 50% clean pairs and 50% auxiliary pairs. The clean arm
uses additional clean pairs in the auxiliary slots. All arms therefore receive
the same updates, examples per update, train-only normalization, and parameter
count.

| Arm | Input | Reconstructed target | Additional information |
| --- | --- | --- | --- |
| `CLEAN` | clean `u_t` | stored `u_(t+1)=Phi(u_t)` | clean trace only |
| `PF-STORED` | detached model-owned prefix state `x=Psi^k(u_t)`, `k in {1,2,4}` | stored clean future `u_(t+k+1)` | deployment exposure, but no `Phi(x)` label |
| `RECOVERY` | frozen common-bank state `x=P(u_t+eta)` | clean next state `Phi(u_t)` | solver-free restoration toward the reference path |
| `DYN-RELABEL` | the identical frozen common-bank state `x` | cached `Phi(x)` | dynamics-consistent displaced-state label |

Targets are converted to residuals relative to the actual arm input. For
example, `RECOVERY` predicts `Phi(u_t)-x`, while `DYN-RELABEL` predicts
`Phi(x)-x`. This prevents an independently rescaled delta from changing the
deployed map.

### Common-bank construction

The bank is frozen before arm training and contains equal cells over:

- perturbation origin: synthetic and frozen clean-pilot rollout deviation;
- spectral band: low, middle, and high retained modes;
- relative amplitude: `0.01`, `0.03`, and `0.10` of state RMS; and
- prefix depth for pilot deviations: `1`, `2`, and `4` calls.

Synthetic directions are real, mean-zero, band-limited, and orthogonalized
against the declared local tangent probes. Pilot deviations come from a frozen
clean model and are never regenerated per information arm. For a depth-`k`
pilot row, the clean base is `u_(t+k)` and the deviation is
`Psi_pilot^k(u_t)-u_(t+k)`; thus the perturbed input and both targets share the
same physical clock. Each bank row stores the clean base `u`, `x`, `Phi(u)`,
`Phi(x)`, perturbation metadata, projection norm, and source hashes. `RECOVERY`
and `DYN-RELABEL` use identical row order and batch order for each seed.

`PF-STORED` is intentionally on-policy and therefore cannot share every input.
Its generated-state count, prefix lengths, extra forward passes, and wall time
are reported separately.

## 8. Checkpoint And Evaluation Contract

Two views are mandatory:

1. **Common-step view:** all arms at the fixed terminal 20,480-update
   checkpoint. This is the primary controlled comparison.
2. **Practical selected view:** each arm selects the checkpoint with the lowest
   validation H64 state error. Nonfinite/incomplete rollouts are ineligible;
   otherwise error is prioritized over structural admissibility. Ties within
   0.5% use H1 error, then the earlier checkpoint.

No checkpoint is selected on the test population. Selected and terminal
results are both retained; a favorable selected result cannot erase terminal
behavior.

### Required metrics

Report case-level values before aggregation.

**Clean map**

- online train, fixed-seen, fixed-validation, and test H1 state/increment error;
- residual scale, raw proposal scale, and projection norm.

**Autonomous rollout**

- state and increment error at H1/H8/H16/H32/H64/H128;
- all-call area, completion, finiteness, and first registered error event;
- fresh/common-input versus propagated-path decomposition at H16/H64;
- kinetic energy, enstrophy, palinstrophy, shell spectra, and dissipation-rate
  errors.

**Common-bank response**

- dynamics defect `||Psi(x)-Phi(x)||`;
- recovery/tube displacement `||Psi(x)-Phi(u)||`;
- response mismatch
  `||(Psi(x)-Psi(u))-(Phi(x)-Phi(u))||`;
- learned and reference secant gains and their ratio;
- defect--perturbation and learned--reference response alignment;
- pre/post-projection output and projection norm.

**Declared tangent/path probes**

- time-shift tangent from centered short reference steps;
- exact periodic `x`-translation tangent;
- spectral perturbations orthogonal to their span as declared normal probes.

These are probe families, not an estimator of the full data-manifold tangent
space. Normal-probe contraction and tangent/path no-harm are reported
separately.

**Cost**

- parameters, optimizer updates, presentations, generated prefixes, trusted
  solver calls, cached bytes, GPU/CPU time, peak memory, and evaluation time.

## 9. Primary Statistics And Gates

Primary ratios use paired trajectory-level errors with the same denominator
floor frozen from the validation split. Aggregate with the geometric mean;
report arithmetic means, medians, per-seed directions, and cluster bootstrap
intervals by independent trajectory as secondary views.

### M1-C1 success gate

At both common-step and practical-selected views, `DYN-RELABEL` versus
`RECOVERY` must satisfy:

1. common-bank dynamics-defect geometric-mean ratio at most `0.80`;
2. H64 rollout-error geometric-mean ratio at most `0.90`;
3. the H64 direction favors dynamics labels in all three training seeds;
4. clean H1 error ratio at most `1.05`;
5. declared tangent response-mismatch ratio at most `1.10`; and
6. no more than 10% regression in each of kinetic-energy, enstrophy,
   palinstrophy, and spectral-distribution error.

Items 1--3 establish benefit; items 4--6 are no-harm gates. Failing any
no-harm gate yields a tradeoff result, not promotion.

`DYN-RELABEL` is not declared necessary if either `PF-STORED` or `RECOVERY`
lies within 5% of its H64 error while passing the same no-harm gates.

### M1-C2 predictive gate

Freeze one scalar response score on validation data before test access. The
candidate starts as log common-bank response mismatch, equally averaged across
origin, band, amplitude, and prefix cells. Compare:

    base: log(H64 error) ~ log(clean H1 error)
    augmented: base + log(response score).

Use leave-one-trajectory-out cross-fitting with model/seed clustered
uncertainty. The response score qualifies only if it:

- reduces held-out median absolute log error by at least 15%;
- increases held-out Spearman correlation by at least 0.10; and
- preserves the sign in every training seed.

This is prospective prediction within the registered model family, not a
universal causal mediator claim.

## 10. Run Order, Cost, And Stop Decisions

| Stage | Purpose | Runs | Gate | Authorization | Cost cap |
| --- | --- | --- | --- | --- | ---: |
| M1-Q0 | analytic/synthetic closure | focused CPU tests plus dry-run CLI | all fail-closure and analytic checks pass | A1, complete | under 5 CPU-min |
| M1-Q1 | numerical and stationarity qualification | fixed solver-only calibration bank | all Q1 gates pass | new named approval | 8 GPU-h or 24 CPU-h |
| M1-Q2 | baseline/difficulty qualification | clean FNO recipe screen, seed 0 | Q2 phenomenon gate and resource envelope pass | A3 required | 8 GPU-h |
| M1-B0 | freeze common bank and both targets | one immutable bank build | balance, replay, solver, and manifest checks pass | A3 required | 8 GPU-h |
| M1-I0 | four-arm screen | four seed-0 cells | correctness plus mediator movement; no test access | A3 required | 12 GPU-h |
| M1-I1 | confirmatory replication | remaining eight cells | three-seed complete matrix | new continuation decision | 24 GPU-h |
| M1-E0 | final open evaluation and figures | selected and terminal checkpoints | all metrics and cost accounting close | A2 required | 4 GPU-h |
| M1-T0 | one sealed test opening | 12 selected/terminal cells under frozen evaluator | one-shot receipt and no reselection | A4 required | 4 GPU-h |

The pre-test programme is capped at 60 GPU-hours. M1-Q1 must replace these
planning caps with measured throughput and memory estimates before M1-Q2.
Unused budget does not authorize extra architectures, schedules, horizons, or
seeds.

## 11. Paper Outputs

Main-paper targets if M1 reaches confirmation:

- one four-arm table containing H1, H64, response mismatch, structure, and cost;
- one scatter/residual plot comparing H1-only and H1-plus-response prediction;
- one rollout/error-growth figure with common-step and selected views; and
- one qualitative vorticity/spectral sequence showing whether stability is
  gained through faithful response or smoothing.

Appendix targets:

- numerical qualification and projection audit;
- complete per-seed/per-trajectory tables;
- tangent/normal probe cells;
- training curves and all checkpoint trajectories; and
- exact manifests, source hashes, and failure receipts.

Cut unless separately justified: new attention kernels, gradient branches,
multiple backbone families, HydroGym, longer-training ladders, and sealed
parameter OOD. Those belong after M1, not inside its causal contrast.

## 12. Current Closeout

The A1 deliverable is complete when the maintained source, focused tests,
synthetic CLI, this preregistration, and the compact tracker share one reviewed
source snapshot. The next permitted action is a concrete request for M1-Q1
solver-only qualification. No data or experiment is launched by this file.
