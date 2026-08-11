# Codex Kick-Start Prompts: Week Of 2026-08-11

Updated: 2026-08-11

These prompts start the five lines in
[WEEKLY_RESEARCH_PLAN.md](WEEKLY_RESEARCH_PLAN.md). Each is self-contained and
ready to paste into a separate Codex session.

All five prompts authorize A0 read-only preflight only. They may run in parallel
because they do not edit the shared worktree. After review, authorize A1 or a
named scientific run line by line. Do not let multiple agents edit shared core
files such as `pcno/pcno.py`, `utility/time_dependent_no/pcno_euler2d.py`, or
`scripts/time_dependent_no/train_pcno_euler2d_residual.py` concurrently in one
worktree.

## Prompt 1: W26-L1 Long-Horizon Stability

```text
We are doing research on Neural Operators for long horizon time-dependent PDE prediction. We are starting W26-L1: long-horizon PCNO stability and failure forensics.

Repository:
the current workspace root (resolve it with `git rev-parse --show-toplevel`)

Branch:
time-dependent-no

Start read-only, inspect this codebase carefully and return an execution-ready preregistration
proposal and the smallest next authorization request.

Read in authority order:
1. AGENTS.md
2. docs/time_dependent_no/RESEARCH_DIRECTION_DECISION.md
3. docs/time_dependent_no/HANDOFF.md
4. docs/time_dependent_no/MECHANISTIC_DIAGNOSTIC_TRACKER.md
5. docs/time_dependent_no/WEEKLY_RESEARCH_PLAN.md, especially W26-L1
6. docs/time_dependent_no/README.md
7. LOCAL_CONTEXT.md privately if it exists; never quote or commit it
8. git status --short --branch

Verify current source and retained artifact manifests rather than repeating
historical statements. Read only bounded tracker/archive sections selected by
run ID. Preserve user-owned changes.

Scientific questions:
- Which exact artifacts are meant by the mentor's stable and unstable
  checkpoints? B1 is the intended serious PCNO checkpoint, with boundary
  information used during both training and inference. Bind its exact
  checkpoint, training protocol, inference protocol, normalizer, recurrence,
  evaluator, and data contract before comparison.
- Why do some evaluated checkpoints remain finite through H79 while older ones
  fail or become inadmissible near H60?
- Is the difference in T_accurate, T_admissible, T_bounded, or T_finite?
- Is failure driven by fresh one-step defect, propagated input error, boundary
  feedback, high-frequency growth, local map amplification, or invariant-domain
  margins?
- When a row “recovers,” was the raw invalid state fed back, or did an input,
  output, or boundary policy repair it?
- Which longer horizons have matching truth? Treat H160/H320 without truth as
  survival/admissibility evidence only.

Tasks:
1. Build exact provenance matrices for every proposed comparison, including
   checkpoint, source, data, split, normalizer, precision, optimizer/history,
   recurrence, boundary policy, evaluator, and selection rule.
2. Reconcile D019, D041, B1/K2, N0, D082, and D084 only where identities are
   exact. Keep bump and dynamic FV separate.
3. Define event semantics and accepted-prefix rules for T_accurate,
   T_admissible, T_bounded, and T_finite.
4. Audit reusable code before proposing a new evaluator, especially:
   utility/time_dependent_no/pcno_rollout.py
   utility/time_dependent_no/pcno_inadmissibility.py
   scripts/time_dependent_no/decompose_pcno_euler2d_rollout_error.py
   scripts/time_dependent_no/evaluate_pcno_inadmissibility_continuation.py
5. Design the matched survival evaluator, fresh/propagated decomposition,
   localized perturbation/JVP study, pre-failure spectra, and recovery-policy
   attribution. JVPs are sensitivity diagnostics, not causal proof.
6. Use primary literature only for any refreshed stability claims. Keep
   boundedness, accuracy, admissibility, conservation, and physical validity
   distinct.
7. Give exact stop/go gates, CPU tests, artifact inventory, cost class, and the
   reference-backed versus truth-free rollout plan.

Required final report:
1. Resolved and unresolved checkpoint identities.
2. Verified evidence, plausible mechanisms, missing evidence, alternatives,
   claim implications, and the minimum decisive experiment.
3. Stability-event schema and metric table.
4. A proposed line-specific preregistration filename and noncolliding stable ID
   request; do not allocate or write it yet.
5. The exact files that A1 would touch and focused CPU checks it would run.
6. The smallest A1 or A2 authorization requested next.
```

## Prompt 2: W26-L2 Shock Representation And Differential Pathway

```text
We are doing research on Neural Operators for long horizon time-dependent PDE prediction. We are starting W26-L2: shock representation, ripple, PCNO differential-path
scaling, gradient ablation, intermediate filtering, loss, and exposure.

Repository:
the current workspace root (resolve it with `git rev-parse --show-toplevel`)

Branch:
time-dependent-no

Start read-only, inspect this codebase carefully and return an execution-ready preregistration
proposal and the smallest next authorization request.

Read in authority order:
1. AGENTS.md
2. docs/time_dependent_no/RESEARCH_DIRECTION_DECISION.md
3. docs/time_dependent_no/HANDOFF.md
4. docs/time_dependent_no/MECHANISTIC_DIAGNOSTIC_TRACKER.md
5. docs/time_dependent_no/WEEKLY_RESEARCH_PLAN.md, especially W26-L2
6. docs/time_dependent_no/README.md
7. LOCAL_CONTEXT.md privately if present; never quote or commit it
8. git status --short --branch

Then read the local paper identified by the private
`GIBBS_PHENOMENON_PAPER` alias in `LOCAL_CONTEXT.md`. Never quote or commit its
machine-specific path.

Verify current code and artifact manifests. Search only bounded D042, D043,
D052--D056, D070C, D073-A, and D080--D081 sections needed for this question.
Do not rerun a completed negative intervention under a new name.

Scientific questions:
- Can full PCNO fit a discontinuous moving-front increment without ripple on a
  fixed discrete grid?
- If so, does failure first appear under subcell phase, resolution transfer,
  optimization, or recurrent exposure?
- Where does the expected O(h^-1) raw discrete shock gradient become an
  inconsistent learned feature: least-squares stencil, fixed-hop aggregation,
  learned scaling, Softsign saturation, or decoding?
- Does removing the gradient branch help, or only remove useful capacity?
- Which intermediate filtering location has causal headroom without shock
  blurring?
- Would front-aware loss or endogenous rollout corruption address a measured
  defect that plain L2/teacher forcing underweights?

Tasks:
1. Audit current `pcno/pcno.py` and the maintained ripple/pathway utilities.
   Record the exact least-squares gradient, graph aggregation, `gw1`, Softsign,
   `gw2`, branch summation, activation, and residual-connection order.
2. Design exact cell-average translated-step/rectangular-increment and smooth
   controls on nested grids with fixed physical displacement and width.
3. Specify fixed-grid overfit, held-out shock phase, held-out resolution, and
   branch-restricted/full-PCNO comparisons.
4. Require normalized overshoot/undershoot, oscillatory mass outside a fixed
   physical front band, front position/strength/thickness, smooth-region error,
   TV excess, pulse integral, and physical spectrum.
5. State the observation required to call a result classical spectral Gibbs,
   full-PCNO capacity failure, phase failure, grid inconsistency, or recurrent
   regeneration. Do not equate a shock-local bump or high-frequency energy with
   Gibbs/ripple.
6. Design frozen gradient instrumentation across nested grids: raw gradient,
   post-aggregation, pre-Softsign, saturation fraction, post-Softsign, branch
   output, and decoded contribution. Distinguish expected distributional peak
   scaling from changing physical support.
7. Design, but do not launch, the three-arm matched training study: full PCNO,
   functional no-gradient, and parameter-matched pointwise/local replacement,
   with shared initialization/data order/budget and three seeds.
8. Design a zero-inclusive frozen filter screen: smooth physical-wavenumber
   spectral filter, hard-cutoff negative control, fixed-physical differential
   support, and diagnosed-layer preactivation. Do not stack an unregistered
   smoother on the existing graph average. Retrain at most two passing choices.
9. Make loss/noise studies conditional: baseline L2; target-derived front loss;
   TV-of-error or target-TV discrepancy; IID noise control; structured
   conservative/shock-phase corruption; short endogenous rollout exposure.
10. Coordinate any D073-B recurrent test with W26-L5's transfer-native
    comparator. D073-A already passed its same-hidden question and must not be
    repeated.

Inspect and prefer existing surfaces such as:
utility/time_dependent_no/pcno_ripple_diagnostics.py
utility/time_dependent_no/pcno_resolution_pathways.py
scripts/time_dependent_no/analyze_pcno_local_correctability.py
scripts/time_dependent_no/analyze_pcno_physical_radius_geometry.py
scripts/time_dependent_no/evaluate_pcno_time_windowed_local_correction.py
tests/time_dependent_no/test_pcno_ripple_diagnostics.py
tests/time_dependent_no/test_pcno_physical_radius_geometry.py

Required final report:
1. Gibbs-to-PCNO claim boundary and current implementation audit.
2. Synthetic target/capacity and gradient-instrumentation registrations.
3. A hypothesis decision tree with decisive observations.
4. Minimal no-gradient, filter, loss, and exposure matrices with gates.
5. At most two variants eligible for later training.
6. Proposed preregistration filenames/ID requests, exact A1 file scope, CPU
   checks, shared-file conflicts, and cost class.
7. The smallest authorization requested next.

Stop after the report.
```

## Prompt 3: W26-L3 Boundary Conditions And Finite Propagation

```text
We are doing research on Neural Operators for long horizon time-dependent PDE prediction. We are starting W26-L3: systematic boundary-condition integration and the
finite-propagation/top-left-error diagnosis.

Repository:
the current workspace root (resolve it with `git rev-parse --show-toplevel`)

Branch:
time-dependent-no

Start read-only, inspect this codebase carefully for context.

Read in authority order:
1. AGENTS.md
2. docs/time_dependent_no/RESEARCH_DIRECTION_DECISION.md
3. docs/time_dependent_no/HANDOFF.md
4. docs/time_dependent_no/MECHANISTIC_DIAGNOSTIC_TRACKER.md
5. docs/time_dependent_no/WEEKLY_RESEARCH_PLAN.md, especially W26-L3
6. docs/time_dependent_no/README.md
7. LOCAL_CONTEXT.md privately if present; never quote or commit it
8. git status --short --branch

Verify current source and artifact manifests. D068--D069, D072, D082/D084,
BG0, RB0, projection, and splice experiments are bounded prior evidence. Do not
reopen their exact static-label/collar or boundary-objective questions under a
new identity.

Scientific questions:
- Which boundary and forcing data are causally known at deployment for
  periodic, Dirichlet/inflow, Neumann/flux, wall, characteristic outflow, and
  time-dependent forcing cases?
- Which information should be encoded, which constraints should be enforced,
  and which combinations improve recurrent interior prediction?
- Does enforcement improve the raw model proposal or merely overwrite it?
- Why does bump error accumulate in the top-left corner?
- Does PCNO response materially exceed the trusted solver's numerical domain of
  dependence, and which branch/layer carries the excess?

Tasks:
1. Build a family-by-boundary taxonomy for existing advection, Burgers, 1D
   Euler, bump, dynamic FV, and planned REALM tasks. Record patch semantics,
   normals, prescribed traces/fluxes, time schedules, geometry, and validated
   boundary exchange. Keep family-local semantics separate.
2. Separate three axes: information, enforcement, and training exposure.
3. Design the minimal information-versus-enforcement 2x2: neither, information
   only, enforcement only, both. Include a periodic/boundary-free control, one
   time-dependent prescribed-inflow task, and one mixed hyperbolic geometry.
4. Inputs may use only information known over [t_n,t_(n+1)]. Future solution
   traces are forbidden. A nodal projection is not exact DG replay or physical
   conservation.
5. Freeze physical boundary policy within representation comparisons and
   representation within enforcement comparisons. Add an exact-function-
   matched or parameter-matched information control when needed.
6. Require six-channel reporting: constraint residual, corrected boundary-
   reference error, raw proposal error, intervention norm, near-boundary normal
   error, and recurrent interior error. Also require admissibility, front
   controls, and all-frame fixed-scale visuals.
7. Design the top-left/domain-of-dependence experiment: paired admissible states
   with compact shock, smooth, boundary, and corner perturbations; one trusted
   solver step; characteristic plus numerical-stencil response cone; full-model
   and pathway-resolved outside-cone response; recurrent correlation with later
   top-left error.
8. Do not attribute the corner defect to the Fourier path unless a controlled
   intervention reduces both early out-of-cone response and later corner error
   without harming the physical response region.

Reuse before proposing new code:
utility/time_dependent_no/pcno_boundary_fields.py
scripts/time_dependent_no/evaluate_pcno_euler2d_boundary_protocol.py
scripts/time_dependent_no/evaluate_pcno_euler2d_boundary_splice.py
scripts/time_dependent_no/evaluate_pcno_boundary_fields.py
tests/time_dependent_no/test_pcno_boundary_fields.py

Required final report:
1. Boundary taxonomy, causal-input contract, and missing-data requests.
2. Existing evidence/nonclaim table showing what must not be duplicated.
3. The minimal benchmark-by-2x2 preregistration and six-channel evaluator.
4. Finite-propagation/top-left preregistration with cone definition and
   alternatives.
5. Seeds, metrics, gates, stop rules, artifacts, cost, and proposed stable-ID
   requests.
6. Recommendation for the single first boundary experiment.
7. Exact A1/A2/A3 authorization requested next.

Stop after the report.
```

## Prompt 4: W26-L4 REALM And Paper-Level Validation

```text
We are doing research on Neural Operators for long horizon time-dependent PDE prediction. We are starting W26-L4: external benchmark adoption and paper-level validation
of the residual neural time-stepper.

Repository:
the current workspace root (resolve it with `git rev-parse --show-toplevel`)

Branch:
time-dependent-no

Start read-only, inspect this codebase carefully for context.

Authorization for this kickoff: A0 READ-ONLY ONLY. You may browse public
primary sources and inspect existing local metadata, but do not edit, run
tests/models, load scientific dataset arrays, download multi-GB data, generate
artifacts, access remote/GPU resources, open sealed populations, stage, commit,
or change the plan.

Read in authority order:
1. AGENTS.md
2. docs/time_dependent_no/RESEARCH_DIRECTION_DECISION.md
3. docs/time_dependent_no/HANDOFF.md
4. docs/time_dependent_no/MECHANISTIC_DIAGNOSTIC_TRACKER.md
5. docs/time_dependent_no/WEEKLY_RESEARCH_PLAN.md, especially W26-L4
6. docs/time_dependent_no/README.md
7. LOCAL_CONTEXT.md privately if present; never quote or commit it
8. git status --short --branch

Read the exact local paper identified by the private `REALM_BENCHMARK_PAPER`
alias in `LOCAL_CONTEXT.md`. Never quote or commit its machine-specific path.

Use official/public primary sources to verify the current paper, code, and
dataset manifests. Do not send private repository material to an external AI.
Do not download the corpus during this kickoff.

Scientific questions:
- Does residual prediction improve long-horizon behavior over a matched direct-
  next-state model on realistic reacting flows?
- Are gains backbone-independent or PCNO-specific?
- Do normalized benchmark metrics hide phase, front, integral, boundary, or
  admissibility failure?
- Can the released data support a boundary-condition study, or are patch
  semantics/forcing schedules missing?
- If the simple model is already competitive, can we retain it and make the
  algorithmic/boundary/evaluation framework the paper contribution?

Tasks:
1. Verify the exact REALM paper version, official code revision, licenses,
   dataset repositories, released splits, file manifests, channels, grids,
   trajectories/frames, cadence, mesh, boundary metadata, and preprocessing for
   IgnitHIT, PlanarDet, and ObstacleDet or SupCavityFlame.
2. Reconcile paper/README/website inconsistencies against actual file manifests
   and record exact missing-data requests.
3. Confirm that REALM is a reacting-flow suite and that its standard recurrent
   models generally predict the next state, with one/two-step unrolling and
   transformed/normalized grouped loss. Do not summarize it as universal neural-
   operator blow-up evidence.
4. Inspect `utility/time_dependent_no/realpde_track2.py` and
   `scripts/time_dependent_no/train_realpde_track2_pcno.py` only to establish
   reuse boundaries. They implement RealPDE Track 2, not REALM; do not conflate
   dataset schemas or claims.
5. Select the smallest staged portfolio: IgnitHIT pipeline/baseline,
   PlanarDet shock test, then one irregular case only if metadata and cost pass.
   Defer 3D and the remaining cases.
6. Design the faithful direct-next-state baseline before the residual variant.
   Strong baseline families are published FFNO/direct-state, matched residual
   FNO or FFNO, and PCNO where point-cloud/irregular geometry is relevant.
7. Match split, preprocessing, target coordinates, model budget, optimizer,
   presentations, rollout exposure, checkpoint selection, precision, and three
   seeds. State whether the residual is physical conservative state or merely a
   transformed-feature increment.
8. Require one-step and rollout error, phase/front/structure, admissibility,
   boundedness, physical summaries, boundary leakage where defined, runtime,
   memory, and parameters. Longer truth-free rollout is survival only.
9. Define the decision: retain the simple framework if it passes; otherwise
   route only a diagnosed failure to W26-L1--L3 rather than proposing an
   architecture sweep.

Required final report:
1. Maximum-two-claim paper map.
2. Verified benchmark/provenance/availability matrix.
3. Must-run versus deferred portfolio and exact data/compute requests.
4. Faithful-baseline and matched-residual contracts, metrics, seeds, and gates.
5. Proposed preregistration filename/ID request and A1 code ownership plan.
6. Shared-file conflicts with the other lines.
7. The smallest authorization requested next, including any download scope.

Stop after the report.
```

## Prompt 5: W26-L5 Cross-Resolution Correlation And Correction

```text
We are doing research on Neural Operators for long horizon time-dependent PDE prediction. We are starting W26-L5: dynamic-FV resolution consistency and cross-resolution
correction inspired by, but not assumed to be, Richardson extrapolation.

Repository:
the current workspace root (resolve it with `git rev-parse --show-toplevel`)

Branch:
time-dependent-no

Start read-only, inspect this codebase carefully for context.

Read in authority order:
1. AGENTS.md
2. docs/time_dependent_no/RESEARCH_DIRECTION_DECISION.md
3. docs/time_dependent_no/HANDOFF.md
4. docs/time_dependent_no/MECHANISTIC_DIAGNOSTIC_TRACKER.md
5. docs/time_dependent_no/WEEKLY_RESEARCH_PLAN.md, especially W26-L5
6. docs/time_dependent_no/README.md
7. LOCAL_CONTEXT.md privately if present; never quote or commit it
8. git status --short --branch

Verify current source and artifact manifests. Read only the bounded D063--D067,
D070C, D073-A, and D074--D081 evidence needed. Keep dynamic FV and bump
separate. Bump query-node dropping is not PDE resolution transfer, and bump
proxy weights do not support conservation claims.

Scientific questions:
- Do synchronized coarse/native/fine predicted-increment differences predict
  the signed native reference error across held-out cases and time?
- Is predictability concentrated in the persistent low-rank large-scale defect
  rather than the shock-local component that often cancels during recurrence?
- Can a case-independent correction improve native rollout safely?
- Does direct off-grid physical-radius PCNO beat transfer to 250x100, native
  rollout, and transfer back after transfer floors and compute are included?

Tasks:
1. Audit and reuse:
   utility/time_dependent_no/pcno_resolution_transfer.py
   utility/time_dependent_no/pcno_resolution_pathways.py
   scripts/time_dependent_no/evaluate_pcno_resolution_transfer.py
   scripts/time_dependent_no/evaluate_pcno_resolution_rollout.py
   scripts/time_dependent_no/analyze_pcno_physical_radius_geometry.py
   tests/time_dependent_no/test_pcno_resolution_transfer.py
   tests/time_dependent_no/test_pcno_physical_radius_geometry.py
2. Bind the common-source dynamic-FV contract: physical volumes, nested maps,
   evolved references, conservative restriction/prolongation, boundary policy,
   normalization, component scales, and transfer floors.
3. Build all coarse/native/fine one-step predictions from representations of
   the same physical state. Do not use independently diverged rollouts for the
   teacher-forced correlation gate.
4. Design synthetic CPU closure tests for restriction/prolongation, mapped
   state-increment conservation, zero correction, and recurrence bookkeeping.
5. Evaluate this candidate only as a registered starting point:
   r_corr = r_native
          + alpha * (r_native - P(r_coarse))
          + beta  * (R(r_fine) - r_native).
6. Fit fixed scalar coefficients first on calibration cases. Test on disjoint
   cases and times. Report signed correlation/cosine, cross-validated R2,
   component/band/region stability, and small-denominator diagnostics. Include
   zero and a per-case oracle diagnostic; the oracle is not deployable.
7. Stop unless coefficient sign/scale is stable and a held-out preregistered
   error component improves without any front/integral/boundary/admissibility
   control violation.
8. Only after that gate, design synchronized recurrent fusion updating one
   native state per call. Compare raw native, corrected native, direct off-grid,
   and transfer-native-transfer-back on the same query-grid truth and cost.
9. Coordinate D073-B with W26-L2. D073-A already answers the same-hidden physical-
   support question; do not rerun it. If direct off-grid cannot beat transfer-
   native, retain it as mechanism evidence rather than a deployment method.
10. Use “cross-resolution correction” or “multiresolution ensemble.” Do not use
    “Richardson extrapolation” unless a stable asymptotic order and leading-
    error model are demonstrated across at least three refinements.

Required final report:
1. Exact common-source and transfer contract.
2. Teacher-forced predictability preregistration and prospective gate.
3. Cross-fitted correction and synchronized-rollout design.
4. D073-B versus transfer-native dependency/decision table.
5. Metrics, stop rules, cost comparators, artifact inventory, and claim limits.
6. Proposed preregistration filename/ID request, A1 file scope, focused CPU
   tests, and shared-file conflicts.
7. The smallest authorization requested next.

Stop after the report.
```
