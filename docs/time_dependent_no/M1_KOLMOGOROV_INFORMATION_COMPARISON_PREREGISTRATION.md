# M1 Kolmogorov Information-Source Comparison Preregistration

Date: 2026-08-26

Status: **M1-Q1-R2 REFERENCE QUALIFICATION AUTHORIZED; Q2 NOT AUTHORIZED**. This document does not
authorize dataset-scale generation, checkpoint access, model training, remote
execution, or sealed evaluation. Every later stage requires the named
authorization in the execution ladder.

Owner continuation on 2026-08-26 authorizes the local solver-only M1-Q1
qualification specified below. It does not authorize M1-Q2 model training,
dataset-scale generation, remote execution, or test access.

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

#### Frozen M1-Q1 execution amendment

The first Q1 execution uses run ID `M1-KF-Q1-20260826A` and the following
deterministic calibration population:

- four independent Gaussian initial arrays with seeds
  `2026082601--2026082604`, canonicalized and rescaled to vorticity RMS `4.0`;
- 256 candidate-map burn-in calls followed by 128 observation calls;
- one predeclared contingency only: if the stationarity gate fails, discard the
  first observation window, extend total burn-in to 512 calls, and observe one
  new 128-call window; and
- no further burn-in, parameter, forcing, resolution, or threshold change.

Time-refinement inputs are the first three post-burn-in clean states and three
matched 3%-RMS perturbations. Perturbation seeds are
`2026082701--2026082703`; their radial integer-mode bands are respectively
`[1,4]`, `[5,10]`, and `[11,20]`. All directions are real, canonical, and
rescaled after band selection. Candidate `dt_max` values are
`{0.002,0.001,0.0005}`; `0.0005` is the comparison reference.

Exact aggregation is:

- one-step and H16 state discrepancy: relative physical-grid L2, with the
  median and maximum evaluated separately for clean and displaced families;
- H64 kinetic-energy and enstrophy distribution discrepancy: empirical
  one-Wasserstein distance computed by sorting the 64 call values, normalized
  by the fine-step mean absolute value;
- H64 spectral discrepancy: total variation between time-averaged normalized
  kinetic-energy shell spectra; and
- stationarity: per-trajectory first-half versus second-half relative mean
  change for energy and enstrophy, divided by the full-window mean absolute
  value, plus Spearman time correlation. A shared drift event requires
  `|rho| >= 0.5` with the same sign in all four trajectories; three of four is
  allowed by the registered 75% boundary.

The spatial-context assay zero-pads each of the same six 64-grid states to 128,
advances at `dt_max=0.0005`, truncates back to the canonical 64-grid space, and
reports one-step and H16 discrepancy from the fine-step 64-grid map. This row
cannot pass its model-relative factor-of-four gate until M1-Q2 supplies a clean
model defect and M1-B0 supplies the recovery/dynamics target separation.

Two spawned FP64 Python processes advance the first clean state independently.
The output packet records bitwise equality, scaled RMS difference, substep
accounting, environment, elapsed time, all case rows, source hashes, and final
artifact hashes. The fixed-grid Q1 classification is
`qualified_fixed_grid` only if time refinement, stationarity, process
repeatability, finiteness, and canonical closure all pass. Spatial qualification
remains explicitly pending rather than being inferred from the fixed-grid gate.

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

## 13. M1-Q1-R1 Reference And Population Diagnosis

The owner continuation on 2026-08-26 authorizes two local FP64 solver-only
diagnostics after the failed `M1-KF-Q1-20260826A` attempt. The parent result is
immutable. R1 does not reclassify it, relax its gates, or authorize Q2.

The parent bindings are:

- source commit `d396c45f2acd0bbea971b3285b98f406bcea4e74`;
- `result.json` SHA256
  `9ffc7a23c62704dad04af03996336ebff793d1cb3d8fb25da6f6f01176018002`;
  and
- `artifact_manifest.json` SHA256
  `d1ef9b4e77cfe9866ab4beee82129972b2f1f6cd6ae3ee6ca8ba93a9653317ae`.

Both R1 stages must verify these bytes before computation, reproduce the parent
initial or post-burn-in state hashes where applicable, bind one clean source
commit at start and end, use at most four local worker processes, and retain
only scalar/spectral diagnostics and hashes. They may not retain full state
trajectories.

### R1-MIX: mixing-window diagnosis

Run ID: `M1-KF-Q1-R1-MIX-20260826A`.

Recreate the four Q1 Gaussian initial conditions under the unchanged N=64,
`dt_max=0.002` map and advance each for 2,048 macro calls. Record after every
call:

- kinetic energy, enstrophy, and palinstrophy;
- kinetic-energy shell spectrum;
- accepted-substep accounting; and
- canonical-state hashes at calls `0,256,512,768,1024,1280,1536,2048`.

The candidate burn-ins are `B={512,768,1024,1280,1536}`. For each B, use the
next 512 calls and report, separately for energy and enstrophy:

1. every chain's first-256 versus second-256 relative mean change;
2. every chain's Spearman time correlation;
3. four 128-call block means and their relative range;
4. split-R-hat across the four chains; and
5. pooled effective sample size from a Geyer initial-positive-sequence
   autocorrelation estimate.

A provisional burn-in candidate is the earliest B for which:

- every chain has energy and enstrophy half-window change at most `0.10`;
- no same-sign `|rho| >= 0.5` drift occurs in all four chains;
- energy and enstrophy split-R-hat are each at most `1.05`; and
- pooled effective sample size is at least `100` for each quantity.

If no B passes, R1 reports `no_burnin_candidate`. If one passes, it is a design
choice only. A future confirmatory qualification would use the frozen fresh
seeds `2026083001--2026083004`, the selected B, one 512-call window, and the
same gates with no contingency. That confirmation is not authorized here.

### R1-SPAT: adjacent-resolution diagnosis

Run ID: `M1-KF-Q1-R1-SPAT-20260826A`.

Recreate and hash-check the six parent Q1 calibration inputs at burn-in 512.
Lift each same N=64 Fourier polynomial directly to N=128 and N=256. Advance all
three resolutions for H16 with `dt_max=0.0005`, then compare adjacent pairs
after restricting the finer result:

- N64 versus restricted N128; and
- N128 versus restricted N256.

For every call and case, retain relative state L2, relative energy,
enstrophy, and palinstrophy difference, normalized-spectrum total variation,
finiteness, and canonical closure. Two fresh N=256 processes must also replay
the first clean H1 call with scaled RMS at most `1e-13` and identical substep
accounting.

The N128 pre-model screen requires all of:

- N128-to-N256 H1 median/maximum state discrepancy at most `0.0125/0.025`;
- N128-to-N256 H16 median/maximum at most `0.05/0.10`; and
- each H1/H16 median and maximum no greater than one half of its corresponding
  N64-to-N128 value.

The absolute H1 screen is derived from the preregistered Q2 maximum clean H1
error `0.05`; it is not a substitute for the later model-relative
factor-of-four gate. Passing labels N=128 only `provisional_spatial_candidate`.
Failure labels the tested N=128 grid `spatial_screen_failed`; it does not
automatically authorize N256-to-N512, a different PDE, or a fixed-grid claim.

### R1 outputs and stop rule

Each stage writes a separate ignored result packet with raw scalar/spectral
series where applicable, source and artifact hashes, environment, timing, and
explicit false flags for data, checkpoint, training, remote, and test access.
The combined cost cap is eight local CPU-hours.

After both stages, record one of three routes:

1. preregister a fresh-seed Q1 confirmation if a burn-in candidate exists and
   the owner accepts either a provisional spatial candidate or an explicitly
   fixed-grid claim;
2. revise the candidate grid under a new numerical qualification; or
3. pivot the restartable testbed.

No route is automatic, and no learned model may run under R1.

## 14. M1-Q1-R1 Immutable Closeout

Both registered stages completed locally on 2026-08-27 from source commit
`26e8ef870b7b93f46326d5be451ebb4d2023f234`.

`M1-KF-Q1-R1-MIX-20260826A` returned `no_burnin_candidate`. Its `result.json`
SHA256 is
`fe3e0232d5ea65ba8b96148b6828ac2f5fa599af90978ef6b83b076ba97e6335`,
and its `artifact_manifest.json` SHA256 is
`c3c91f915716c08c0c9045b097cb45578e3974c7bee8e475556718807f651a44`.
All five candidates failed both registered R-hat and ESS gates. Shared drift
passed in every window, so this result rejects the tested sampling law without
establishing persistent physical nonstationarity.

`M1-KF-Q1-R1-SPAT-20260826A` returned `spatial_screen_failed`. Its
`result.json` SHA256 is
`31dd4bdfa2e22c2bd3cdaea69201361a1e42d7375e24a677751877cc9834938d`,
and its `artifact_manifest.json` SHA256 is
`22de2039374be156fde278b1b1fb5eeaf4c02b0f36de1582537bda19b63a63a5`.
The N128-to-N256 H1 median/maximum were `0.00209/0.00265`; H16 was
`0.07442/0.08179`. Contraction, H1, H16 maximum, finiteness, closure, and
repeatability passed. The sole spatial gate failure was the H16 median limit
`0.05`.

The result-bound analysis and figures live under
`artifacts/time_dependent_no/m1_kolmogorov_q1_r1_analysis_20260827a/`. Their
source commit is `c2ec748cddfc4e13cb741de3f178626144084641`, and the analysis
manifest SHA256 is
`65574314c427eb98d0da4b12561ed76ef8de7fc6c6f16d46ed79f3becac7b093`.

R1 therefore closes without a fresh-seed candidate, a provisional N128 grid,
or authority to proceed to Q2. A later owner decision may preregister a new
population/reference design or pivot the testbed. It may not reinterpret this
failed attempt as a qualification.

## 15. M1-Q1-R2 Candidate-Grid And Sampling-Law Qualification

The owner continuation on 2026-08-27 authorizes one new staged, local,
solver-only qualification. R2 has a new identity and does not reclassify Q1 or
R1. It may not access a dataset, checkpoint, learned model, remote process, or
sealed population, and it does not authorize Q2.

R2 addresses the two R1 failures in authority order:

1. qualify the complete numerical reference contract at a candidate grid; then
2. only if that reference packet passes and is hash-frozen in this document,
   test a fresh-seed sampling law for that same numerical map.

This order is mandatory. Population statistics measured for `Phi_64` may not
qualify a later `Phi_256` training population.

### R2-REF: N256 candidate-reference qualification

Run ID: `M1-KF-Q1-R2-REF-20260827A`.

The stage verifies the immutable parent Q1 packet and both R1 diagnostic
packets, including every artifact and commit-source hash. It reconstructs the
same six parent Q1 inputs at N64 burn-in call 512 and requires their exact state
hashes before lifting them. It then performs two complementary tests.

**Adjacent spatial refinement.** Lift each identical N64 Fourier polynomial
directly to `N={128,256,512}` and advance every resolution for H16 at
`dt_max=0.0005`. Retain the same state-L2, energy, enstrophy, palinstrophy,
normalized-spectrum, finiteness, and canonical-closure rows as R1. The complete
N128-to-N256 overlap must reproduce R1 with matching categorical fields and a
maximum absolute numeric difference at most `1e-13`.

N256 passes the spatial screen only if all of the following hold:

- N256-to-restricted-N512 H1 median/maximum state discrepancy is at most
  `0.0125/0.025`;
- N256-to-restricted-N512 H16 median/maximum is at most `0.05/0.10`;
- every H1/H16 median and maximum is at most one half of the corresponding
  N128-to-N256 value;
- all states are finite and canonical; and
- two fresh N512 processes agree below scaled RMS `1e-13` with identical
  accepted-substep accounting.

These are unchanged R1 screen thresholds applied to the next adjacent pair.
Passing is bounded finite-grid evidence, not proof of continuum convergence.

**Candidate-grid time refinement.** Lift the same six hash-matched inputs to
N256 and compare `dt_max={0.002,0.001,0.0005}`. Candidate `0.002` is compared
with `0.0005`. The state-path gates remain the original Q1 values: separately
for clean and displaced families, H1 median/maximum must be below
`1e-5/1e-4`, and H16 median/maximum below `1e-3/5e-3`. Across H64, normalized
one-Wasserstein energy and enstrophy discrepancies and normalized-spectrum
total variation must each be below `0.02`. All trajectories must be finite and
canonical. Two fresh N256 candidate-map processes must additionally satisfy
the same repeatability gate.

The registered classification is `reference_candidate_qualified` only if the
spatial screen, shared-row replay, candidate-grid time refinement, both process
repeatability checks, finiteness, and canonical closure all pass. Any failure
stops R2 before population execution. The stage retains scalar/structure rows,
hashes, and accounting only; no full state trajectory is retained. Its wall
time cap is four hours with at most three solver workers.

### R2-POP: conditional same-map population qualification

Run ID: `M1-KF-Q1-R2-POP-20260827A`.

This stage is conditionally authorized only after R2-REF qualifies and its
exact result and manifest SHA256 values are amended into this section in a
clean source commit. Until then, its execution is fail-closed.

If opened, R2-POP uses `Phi_256` with `dt_max=0.002`, four fresh canonical
Gaussian initial arrays with seeds `2026083101--2026083104`, and vorticity RMS
`4.0`. Each chain has exactly 1,024 burn-in calls followed by one fixed
4,096-call observation window. The length is prospective: R1's worst observed
IAT of approximately 155 calls implies a nominal pooled ESS of approximately
`4*4096/155 = 105.7`, only slightly above the registered minimum. No alternate
burn-in, shorter subwindow, seed substitution, or contingency may select a
passing result.

For energy and enstrophy separately, retain per-chain split-half relative mean
change, Spearman time correlation, eight 512-call block means, split-R-hat,
Geyer IAT, per-chain ESS, and pooled ESS. The unchanged gates are:

- every split-half relative mean change at most `0.10`;
- no same-sign `|rho| >= 0.5` drift in all four chains;
- split-R-hat at most `1.05`; and
- pooled ESS at least `100`.

Passing qualifies only this finite `Phi_256` sampling law. Failure distinguishes
insufficient population precision from the already separate reference-grid
question, but it does not prove physical nonstationarity. The population stage
retains scalar/spectral series and state hashes at calls `0,1024,5120`; it does
not retain full state trajectories. Its wall time cap is eight hours with at
most four solver workers.

### R2 stop and continuation rule

R2 opens Q2 only if both exact stages pass under their frozen contracts and a
closeout records their final artifact/source hashes. A reference pass with a
population failure routes only to a new population decision. A reference
failure routes to a new target-semantic decision or a testbed pivot. Aggregate
structure convergence may motivate a future tubular/statistical target, but it
cannot replace the registered deterministic-path gate post hoc.
