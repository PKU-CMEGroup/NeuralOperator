# B2-GN Gaussian Normal-Noise ODE Study

Updated: 2026-08-30

Status: owner-authorized, bounded local-CPU ODE extension. This successor has
its own source, training, checkpoint, and artifact identity. It does not alter
the frozen B2-NL packet, authorize PDE or remote work, or imply an ODE-to-PDE
ranking.

## 1. Question And Role

B2-GN asks how the scale of Gaussian training-input corruption changes the
learned response, clean-path fidelity, and forced rollout of recovery and
dynamics-relabeling maps:

> When corruption is restricted to the known normal coordinate, which response
> properties change with its scale, and do those changes match the framework's
> qualitative predictions?

The study calibrates mechanisms. It does not estimate a universally optimal
noise variance. Every registered scale is reported, and no scale is selected
from a long-rollout outcome.

## 2. Trusted System And Fixed Learner

Use the exact nonlinear ODE, C/M/E response regimes, float64 RK4 transition,
solver qualification, residual MLP, recurrent forcing, tube, and safety strip
defined by
[B2_NL_NONLINEAR_ODE_STRESS_PREREGISTRATION.md](B2_NL_NONLINEAR_ODE_STRESS_PREREGISTRATION.md).
The frozen parent packet is
`corrective_ode_nonlinear_stress_20260830a`, with manifest SHA-256
`8c1458d44dedb5dfa170d7e2acd7b84efb74541c2d87eaa49e0276373abf9f40`.

The learner remains a three-hidden-layer, width-64 `tanh` residual MLP with
inputs `(cos(theta), sin(theta), r)` and lifted outputs `(Delta theta, Delta r)`.
Training uses Adam, learning rate `2e-3`, batch size 128, 3000 fixed updates,
and paired seeds `17`, `29`, and `43`. Terminal checkpoints are retained; no
validation or rollout metric selects a checkpoint.

## 3. Gaussian Training Contract

### 3.1 Geometry and scales

Only the input normal coordinate is corrupted:

\[
\widetilde x=(\theta,\sigma_{\rm train}z),
\qquad z\sim\mathcal N(0,1).
\]

There is no tangent noise, label noise, ambient-coordinate noise, clipping,
rejection sampling, or training-time recurrent forcing. Normal corruption can
still induce a trusted tangential output response; DYN must learn that response
rather than suppress it.

The registered values are standard deviations

\[
\sigma_{\rm train}\in\{0.005,0.01,0.02,0.04\},
\]

with recorded variances
`{2.5e-5, 1e-4, 4e-4, 1.6e-3}`. `CLEAN` is the exact zero-noise control. No
redundant zero-scale RECOVERY or DYN checkpoint is trained because its target
and input law collapse to CLEAN.

### 3.2 Paired fresh-per-update schedule

For each paired seed, pre-generate and digest:

- clean phase indices with shape `(3000, 64)`;
- noisy phase indices with shape `(3000, 64)`; and
- standardized normal draws with shape `(3000, 64)`.

The clean phases are the same 64-point grid as B2-NL. Every update consumes
exactly 64 clean and 64 noisy rows. Draws are fresh across updates but fully
deterministic under the registered seed. Within a seed, every arm and scale
uses the identical indexed phases and `z` values; a scale changes only
`r=sigma_train*z`. Every checkpoint starts from the same initial parameter
bytes within that seed.

`CLEAN` uses both phase schedules as 128 clean rows. The other targets are:

- `RECOVERY`: clean and noisy inputs both target the clean successor
  `Phi(theta,0)`;
- `DYN-C`, `DYN-M`, and `DYN-E`: clean inputs target `Phi(theta,0)`, while a
  noisy input targets the trusted successor `Phi_j(theta,sigma_train z)`.

This gives 3 CLEAN checkpoints and
`4 scales x 3 seeds x (RECOVERY + DYN-C/M/E) = 48` scale-specific checkpoints,
for 51 checkpoints total.

### 3.3 Gaussian tails

The Gaussian is unbounded and must not be described as a hard training tube.
For every seed and scale, record the unique draw count, repeated model
presentations, maximum `|r|`, the `0.5`, `0.9`, `0.95`, `0.99`, and `0.999`
quantiles of `|r|`, and realized and
theoretical fractions beyond the response-tube radius `0.15` and safety radius
`0.5`. If the frozen realized schedule contains any `|r|>0.5`, fail before
solver labeling; do not resample the tail away.

## 4. Frozen Evaluation

Reuse the B2-NL held-out half-grid response banks:

- clean states at `r=0`;
- inner radii `{-0.10,-0.025,0.025,0.10}`; and
- outer diagnostic radii `{-0.20,0.20}`.

For each checkpoint report clean one-step error, fresh normal forcing,
finite-amplitude normal and normal-to-phase response, trusted-flow defect,
clean-return error, output distance to the clean set, and the 32-step local
secant log-gain. Inner and outer banks remain separately labelled.

A separate frozen profile bank uses both signs of
`|r| in {0.025,0.05,0.075,0.10,0.15,0.20}` at the held-out phases. It reports
phase-and-sign RMS clean-return error for RECOVERY and trusted-flow defect for
DYN at each absolute radius. No binary low-error width or hard Gaussian tube is
defined.

The recurrent assay remains post-map Rademacher forcing with
`sigma_force in {0,1e-3,5e-3}`, 192 steps, 32 held-out phases, and 16 paired
sequences. Training records use `sigma_train` and `variance_train`; rollout
records use `sigma_force`. Their laws are never conflated. Every learned
rollout is compared with both the unforced clean path and the independently
evolved same-forcing trusted path.

## 5. Predictions And Falsifiers

The following scale-response predictions are frozen before the canonical
long-rollout results are inspected. At a fixed nonzero radius, Gaussian density
need not increase monotonically with `sigma_train`; every nonzero Gaussian also
has full support. The tests therefore use local probability mass rather than a
monotone notion of coverage.

1. **Recovery density--fidelity signature.** For each seed and registered
   absolute profile radius, compute the Spearman correlation across the four
   scales between Gaussian log density at that radius and negative RECOVERY
   clean-return RMS error. Aggregate over seed--radius pairs. The signature
   passes when the median correlation is positive and more than half of the
   correlations are positive.
2. **Dynamics density--fidelity signature.** Repeat the same calculation for
   negative DYN trusted-flow RMS defect, separately retaining regime in the
   sampling unit. Aggregate over seed--regime--radius rows. The signature
   passes under the same positive-median and majority-positive rule.
3. **Fidelity--retention tradeoff.** Larger recovery noise may improve normal
   retention while worsening clean or phase fidelity. DYN may be preferable in
   the contractive regime yet remain vulnerable when the trusted normal
   response is expansive. If one intervention and one scale dominates every
   response, path, and forcing endpoint, the anticipated tradeoff is not
   observed.
4. **Mediator before outcome.** A claimed rollout improvement is mechanistically
   interpretable only if the corresponding fixed-bank response or retention
   mediator moves first in the predicted direction. Rollout-only improvement
   does not validate the explanation.

Report all four scales and all three seeds. Scale curves, paired seed values,
and negative results remain visible. Long-rollout error is not used to choose a
displayed scale or to change the contract.

### 5.1 Response qualification before rollout interpretation

For each seed and scale, the intended learned-response comparison qualifies
only if all of the following fixed-bank checks pass:

- in every regime, DYN inner trusted-flow RMS defect divided by trusted
  displaced-response RMS is at most `0.15`;
- every DYN clean lifted one-step RMS error is at most `2e-3`, and the largest
  across-regime DYN clean error is at most twice the smallest;
- in every regime, DYN has lower trusted-response defect than RECOVERY, while
  RECOVERY has lower clean-return error and smaller normal response magnitude;
  and
- the learned 32-step clean-reference local-secant means satisfy
  `C < -0.5`, `|M| <= 0.5`, and `E > 0.5`, hence `C < M < E`.

A scale is rollout-qualified only when every paired seed passes. All forcing
assays are still retained, but a scale that misses this gate has descriptive
rollout outcomes only. The profile signatures above remain tests of how the
training distribution changes fitted response; they do not rescue an
unqualified long-rollout mechanism claim.

## 6. Implementation, Audit, And Packet

New long-lived files are limited to:

- `scripts/time_dependent_no/run_corrective_ode_gaussian_normal_noise.py`; and
- `tests/time_dependent_no/test_corrective_ode_gaussian_normal_noise.py`.

The runner may import the frozen nonlinear and affine ODE implementations but
must not edit them. Before training it verifies the exact parent manifest,
current imported sources, inherited solver qualification, and regenerated
forcing-bank digests. The final manifest binds this preregistration, both new
files, every imported source dependency, the recorded independent audit,
schedules, checkpoints, results, and figures. The closeout then binds the final
manifest hash; the manifest does not circularly bind the closeout.

Focused tests must close normal-only corruption, exact 64/64 mixtures, paired
schedules and initial bytes, target semantics, zero-scale collapse, unclipped
tails, safety failure, sigma-name separation, checkpoint round trips, and
manifest-tamper refusal. A separate agent must issue `AUDIT_PASS` for target,
schedule, solver, recurrence, metric, and provenance semantics before the
canonical run.

The canonical output identity is
`artifacts/time_dependent_no/corrective_ode_gaussian_normal_noise_20260830a`.
It is a local CPU study. No dataset, GPU, remote host, PDE solver, or sealed
population is involved.

## 7. Scope Boundary

Included: one known normal coordinate, four Gaussian standard deviations,
three paired seeds, recovery and C/M/E dynamics relabeling, fixed response
banks, and the separate frozen recurrent-forcing assay.

Excluded: tangent or ambient noise, label noise, clipping, architecture or
optimizer sweeps, post-outcome scale tuning, learned correctors, PDE analogues,
online OOD detection, and claims of a universally optimal variance.
