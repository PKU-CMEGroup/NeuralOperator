# Weekly Research Plan: Long-Horizon Residual Neural Operators

Updated: 2026-08-12

Status: current owner-selected planning and coordination surface. This document
does not by itself authorize checkpoint execution, dataset-scale training,
remote access, or sealed-population access. Each research line advances through
the authorization ladder below.

This plan is subordinate to [RESEARCH_DIRECTION_DECISION.md](RESEARCH_DIRECTION_DECISION.md)
for evidence and claim boundaries. Ready-to-paste agent instructions are in
[CODEX_KICKSTART_PROMPTS.md](CODEX_KICKSTART_PROMPTS.md).

The frozen D072 [experiment plan](history/d072_refine_logs_frozen_2026-08-03/EXPERIMENT_PLAN.md)
and [execution tracker](history/d072_refine_logs_frozen_2026-08-03/EXPERIMENT_TRACKER.md)
retain their 2026-08-03 forward-looking status text as history; they are not the
current weekly plan or queue.

## Problem Anchor And Method Thesis

The paper studies neural operators as time steppers for difficult time-dependent
PDEs. The common algorithmic skeleton is

    delta_u_hat_n = G_theta(u_n, mesh, dt, known boundary data, known forcing)
    u_hat_(n+1) = P_B(u_hat_n + delta_u_hat_n),

where `G_theta` predicts a one-step flow-map increment and `P_B` is an explicit,
causal boundary-handling operator when the problem requires one.

Residual prediction plus autoregression is the backbone, not a sufficient
novelty claim by itself. The intended contribution is a reproducible framework
that identifies which state, boundary, representation, and stability contracts
are necessary for accurate long-horizon use.

The paper is intentionally non-frontier-model work. No LLM, diffusion, or RL
component is needed unless a later result establishes that a simpler mechanism
cannot solve a registered problem.

## Claim Map

At most two claims are primary.

| Claim | Minimum convincing evidence | Main supporting lines |
| --- | --- | --- |
| C1. A residual neural time-stepper can remain accurate and operationally stable across representative smooth, shock-dominated, and irregular-geometry time-dependent PDE regimes. | Matched direct-next-state versus residual comparison; reference-backed horizons; at least three seeds when training variance matters; separate accurate, admissible, bounded, and finite horizons; front, integral, boundary, and spectral diagnostics. | W26-L1, W26-L2, W26-L4 |
| C2. Boundary handling is a separable algorithmic component: causal boundary information and boundary enforcement have distinguishable effects, and an appropriate combination improves recurrent interior prediction for more than one boundary class. | Information-versus-enforcement controls under frozen physical boundary policies; at least three representative boundary classes; raw and corrected proposal metrics; recurrent interior benefit rather than constraint satisfaction alone. | W26-L3, W26-L4 |

Anti-claims that must be ruled out or stated explicitly:

- the gain is merely more parameters, a different optimizer, a different
  normalization, or more generated-state exposure;
- a bounded rollout is necessarily accurate, admissible, conservative, or on
  the correct attractor;
- lower high-frequency energy is necessarily less ripple rather than a blurred
  shock;
- an `O(h^-1)` gradient peak at a discontinuity is itself evidence of a wrong
  derivative;
- fixed weights or finite unseen-grid tests prove operator convergence;
- a learned cross-grid combination is Richardson extrapolation without a stable
  asymptotic error expansion;
- a boundary projection is exact DG replay or physical conservation;
- bump proxy weights support finite-volume conservation claims; or
- results on bump, dynamic FV, REALM, or 1D families can be pooled without a
  family-specific transfer experiment.

## Five Coordinated Research Lines

| Line | Core question | This-week must-run scope | Conditional continuation |
| --- | --- | --- | --- |
| W26-L1: Long-horizon stability | Why do some PCNO checkpoints remain finite through H79 while others fail near H60, and what does stability mean? | D087 is complete and terminal at H79: exact identities, four event horizons, feedback attribution, and paired fresh/propagated decomposition passed. | No implicit continuation. H160/H320, spectra/JVPs, policy counterfactuals, and more bump truth require a new owner-selected claim and stable identity. |
| W26-L2: Shock representation and differential pathway | Are shock-adjacent defects classical spectral retrieval error, finite-grid capacity/phase error, gradient-path inconsistency, or recurrent exposure error? | Analytic moving-step capacity harness; frozen gradient-scaling instrumentation; branch/filter/loss/noise preregistration. | Three-seed no-gradient training, selected intermediate filter, selected loss/noise retraining, D073-B recurrent physical-radius test. |
| W26-L3: Boundary conditions and finite propagation | Which boundary information and enforcement mechanisms are useful for each boundary class, and does global mixing seed the top-left error outside the physical domain of dependence? | Boundary taxonomy and task audit; information-versus-enforcement matrix; synthetic/local finite-propagation probe. | Multi-seed boundary training on selected classes and recurrent top-left causal test. |
| W26-L4: Benchmark and paper validation | Does the residual framework survive a strong external benchmark under a fair direct-state baseline? | Verify REALM data/manifests/metadata and boundary availability; define the exact IgnitHIT baseline contract; do not download multi-GB data without approval. | IgnitHIT baseline and residual comparison, PlanarDet, one irregular case, then 3D only after 2D passes. |
| W26-L5: Cross-resolution correction | Do coarse/native/fine prediction differences predict native error out of case, and can they improve a synchronized native update safely? | Teacher-forced common-source formulation, synthetic transfer tests, and preregistered correlation gate on dynamic FV only. | Cross-fitted scalar correction, synchronized recurrent fusion, and direct off-grid versus transfer-native comparison. |

These lines may run in parallel through preflight and synthetic CPU checks. They
do not create five independent GPU queues. Training and real-checkpoint phases
advance only after their line-specific gate is reviewed.

## Shared Authorization Ladder

| Level | Authorized work |
| --- | --- |
| A0 | Read-only authority, source, artifact-manifest, literature, and provenance audit. |
| A1 | Minimal local implementation, synthetic fixtures, CPU tests, dry-run CLI validation, and a line-specific preregistration. No real checkpoint or dataset-array execution. |
| A2 | Named open-population checkpoint evaluation under an approved immutable contract. No training, remote access, or sealed population. |
| A3 | Named training or dataset-scale run, including an explicit machine/data/compute contract. AutoDL access must be explicitly authorized. |
| A4 | Named sealed/OOD/test evaluation. Not authorized by this weekly plan. |

The kick-start prompts grant A0 only. An agent must end with a concrete A1, A2,
or A3 request rather than silently crossing the boundary.

## Shared Experimental Contract

### Provenance

Every scientific row binds checkpoint, source snapshot, data manifest, split,
normalizer, precision, seed, training presentations, optimizer, selection rule,
recurrence, boundary policy, evaluator, and artifact hashes. Changed attempts
receive new IDs; failed historical attempts are not rewritten.

`B1` is the intended serious PCNO checkpoint, with boundary information used
during both training and inference. W26-L1 must bind its exact checkpoint,
training protocol, inference protocol, normalizer, recurrence, evaluator, and
data contract before using it in a comparison.

### Horizon semantics

Report four event times separately:

- `T_accurate`: reference-backed error/structure threshold is met;
- `T_admissible`: density, internal energy, pressure, and any family-specific
  invariant-domain checks pass;
- `T_bounded`: registered amplitude/norm/control thresholds pass; and
- `T_finite`: no NaN or Inf occurs.

Truth-free continuation beyond the available reference supports only boundedness,
finiteness, admissibility, and declared long-time statistics. It is not a
longer-horizon accuracy result.

### Required metric families

Use case-level records before aggregation. At minimum, report:

- teacher-forced one-step state and increment error;
- free-rollout state error and the fresh/propagated decomposition;
- completion, survival, first invalid proposal, first nonfinite state, and
  amplitude/norm events;
- density, internal energy, and pressure minima;
- shock/front position, strength, and thickness where applicable;
- smooth-region and fixed-physical-wavenumber errors, with anti-smearing
  controls whenever filtering or TV is involved;
- boundary residual, corrected trace error, raw proposal error, intervention
  norm, near-boundary normal error, and recurrent interior error;
- physical-volume integral/boundary-leakage diagnostics only where audited
  finite-volume geometry exists; and
- runtime, memory, parameter count, and evaluation cost.

Three seeds are the default for new training comparisons. A one-seed frozen
intervention may diagnose a mechanism but cannot establish training robustness.

## W26-L1: Long-Horizon Stability And Failure Forensics

### Questions

1. Which exact checkpoints are being compared as “stable” and “unstable”?
2. Does the newer model remain accurate, merely admissible, merely bounded, or
   merely finite after error saturation?
3. Are older failures driven primarily by fresh one-step defect, amplification
   of incoming error, boundary feedback, high-frequency growth, invariant-domain
   loss, or a configuration/provenance difference?
4. When an inadmissible state appears to recover, did the raw state enter the
   next call, or did an input/output/boundary policy repair it?
5. Do conclusions survive longer horizons and multiple cases/seeds?

### Current evidence to preserve

- D041 is the exact historical bump comparator. B1 is the intended serious
  boundary-informed PCNO checkpoint, subject to exact protocol and artifact
  binding before comparison.
- D084 shows 360 finite H79 continuations and rejects “inadmissibility implies
  global blow-up” under its exact contract. Recovery is sparse, policy-mixed,
  and not durable.
- D053b finds full-state error mostly propagated while smooth high-pass defect
  is mostly freshly regenerated on the dynamic family.
- Historical D019-versus-D041 outcome differences are real but causally
  confounded.
- D087 exactly binds B1 and owner-designated D019 on one common declared
  30-case bump population. B1 remains admissible, bounded, and finite through
  H79. D019 loses accuracy first, then becomes inadmissible and unbounded while
  remaining finite. Its late error is dominated by propagated-input response,
  not a rising fresh one-step defect.

### Experiment ladder

**L1-P0: identity and contract audit — COMPLETE**

- Bind the exact B1 checkpoint and its boundary-information training and
  inference protocols.
- Construct a comparison matrix for D019, D041, B1/N0/D082 only when exact
  source/configuration identities exist.
- Record which raw, projected, or repaired state enters every recurrent call.

Gate: no mechanistic old-versus-new claim if the comparison cannot be aligned.
The line may still produce a bounded descriptive survival comparison.

**L1-P1: common stability evaluator — COMPLETE**

- Reuse the maintained rollout, inadmissibility, boundary, and visualization
  utilities before creating a new evaluator.
- Implement synthetic CPU tests for all four horizon semantics and recovery
  attribution.
- Predeclare H79 reference-backed evaluation separately from H160/H320
  truth-free continuation.

Gate: event identities, accepted-prefix semantics, and recurrence closure must
pass before any checkpoint is loaded.

**L1-P2: open-population replay — COMPLETE A2**

- Evaluate matched cases with identical precision and boundary policy.
- Report survival curves rather than only the longest successful trajectory.
- Preserve every comparable frame and mark the first excluded invalid proposal.

**L1-P3: amplification diagnosis — COMPLETE AT REGISTERED DECOMPOSITION SCOPE**

- Estimate finite-time response to localized shock, smooth, boundary, and
  spectral-band perturbations.
- Use the fresh/propagated decomposition and optional JVP norms; label JVPs as
  sensitivities, not causal proof.
- Inspect physical and hidden-branch spectra before failure.

D087 completed the fresh/propagated branch with exact H79 replay and algebraic
closure. Spectra, localized perturbations, and JVPs were not needed for the
accepted inference-map claim and are not queued. They require a new
preregistration if a later causal question makes them decisive.

**L1-P4: longer reference — NOT SELECTED**

- Generate or acquire longer truth only after the solver/data contract and cost
  are approved. Never infer accuracy from a truth-free extension.

### Paper role and decision

D087 supplies a main-paper-ready event taxonomy and a bounded inference-map
failure diagnostic. It does not provide a matched causal factor contrast or a
general stability claim. The line is closed at H79; the next benchmark-facing
step is W26-L4 REALM rather than truth-free bump continuation.

## W26-L2: Shock Representation, Ripple, Gradient, Filtering, Loss, And Noise

### Questions

1. Can full PCNO represent a translated discontinuous increment without ringing
   on a fixed grid?
2. If it can, does failure arise from held-out shock phase, resolution transfer,
   optimization, or recurrence?
3. Where does raw `O(h^-1)` gradient scaling become an inconsistent learned
   feature: least-squares stencil, graph aggregation, learned scaling,
   `softsign` saturation, or decoding?
4. Does removing the gradient branch improve stability/resolution behavior, or
   merely remove useful front information/capacity?
5. Which intermediate filtering location has causal headroom without shock
   blurring?
6. Do front-aware losses or rollout-shaped noise reduce fresh shock defects and
   improve recurrence?

### Current evidence to preserve

- Classical Gibbs theory applies directly to a truncated global spectral
  reconstruction, not automatically to full nonlinear PCNO.
- D052 found no safe uniform branch attenuation.
- D070C implicates the differential pathway and a substantial fixed-hop
  composite, not one unique additive subpath.
- D073-A shows that a fixed physical graph-ball radius improves every registered
  same-hidden cross-grid stratum; it does not modify the least-squares gradient
  or establish recurrent benefit.
- D080--D081 show small early local-filter/timing headroom with late reversal
  and no promoted endpoint correction.

### Experiment ladder

**L2-P0: analytic moving-front capacity test — MUST**

- Generate exact cell-average translated steps/rectangular increments and
  smooth controls on nested meshes with fixed physical displacement and width.
- Compare spectral-only, pointwise-only, differential-only, branch combinations,
  and full PCNO at fixed-grid overfit, held-out shock phase, and held-out
  resolution stages.
- Measure normalized overshoot/undershoot, oscillatory mass outside a fixed
  physical front band, front position/thickness, smooth-region error, TV excess,
  Fourier tail, and pulse integral.

Gate: do not call a generic shock-local bump “Gibbs.” Require persistent
normalized overshoot with narrowing support as spectral resolution increases,
or use the more general representation/phase/grid terminology.

**L2-P1: frozen gradient scaling audit — MUST**

- On synthetic discontinuities, record the raw least-squares gradient,
  post-neighbor aggregation, learned pre-`softsign` value, saturation fraction,
  post-`softsign` value, branch output, and decoded contribution across grids.
- Distinguish expected distributional `1/h` peak scaling from changing physical
  support and downstream nonlinear treatment.
- Compare fixed graph hops, fixed physical radius, locally scaled `h_i grad_h`,
  and a fixed-physical-scale mollified derivative as diagnostics.

**L2-P2: trained gradient ablation — A3 CONDITIONAL**

- Three arms: full PCNO, functional no-gradient PCNO, and a parameter-matched
  pointwise/local-capacity control.
- Freeze shared initialization, data order, optimizer, presentation budget,
  selection, recurrence, and boundary policy; use three seeds.
- Evaluate native rollout, resolution transfer, ripple, front, admissibility,
  and branch activation.

**L2-P3: filtering screen — A2 THEN A3 CONDITIONAL**

- Frozen-checkpoint screen first: smooth physical-wavenumber spectral filter;
  hard cutoff negative control; replacement of the existing differential
  average by fixed-physical support; summed-preactivation filter at the
  diagnosed layer.
- Do not stack a second unregistered smoother on the existing two-hop average.
- Retrain at most one or two variants selected by preregistered anti-smearing
  gates. Do not select from final H79 outcomes.

**L2-P4: loss and exposure — A3 CONDITIONAL**

- Compare baseline normalized `L2`, `L2` plus target-derived front/phase loss,
  and `L2` plus TV-of-error or target-TV discrepancy. Raw TV of the prediction
  is not the default because it penalizes the physical jump.
- Compare clean training, IID noise as a control, endogenous short-rollout
  states, and conservative/front/boundary-shaped perturbations.
- Trigger only if L2 underweights a measured front defect or if fresh defect
  remains after the representation/pathway screen.

### Paper role and decision

The static capacity result and gradient pathway are main mechanistic evidence.
Filtering, loss, and noise are main-paper material only if one passes recurrent
state/front/admissibility gates; otherwise they are bounded negative ablations.

## W26-L3: Boundary Conditions And Finite Propagation

### Questions

1. Which information is known causally for Dirichlet/inflow, Neumann/flux,
   wall, characteristic outflow, periodic, and time-dependent forcing cases?
2. Should that information be encoded, enforced, or both?
3. Does a boundary method improve raw proposals and recurrent interior dynamics,
   or only overwrite the boundary trace?
4. Why does error accumulate in the bump top-left corner?
5. Does the learned one-step response extend materially outside the reference
   PDE/numerical domain of dependence, and which pathway carries it?

### Current evidence to preserve

- Hard causal/minimum-change projection is a legal nodal deployment operator,
  not exact DG replay or physical conservation.
- D068--D069 prove use of family-local categorical semantics but not optimality.
- D072 continuous collars are complete and not promoted; do not rerun the same
  N0/G1/S1 question under a new name.
- D084 exposes a checkpoint-bound completion/error/local-growth tradeoff; it
  does not promote D082 or prove that invalidity causes blow-up.
- A global Fourier path has global receptive field, but a corner error alone is
  not causal evidence of finite-propagation violation.

### Experiment ladder

**L3-P0: boundary contract taxonomy — MUST**

- Audit the existing advection, Burgers, 1D Euler, bump, dynamic FV, and planned
  REALM cases for boundary class, known deployment data, node/face metadata,
  time dependence, and reference boundary exchange.
- For each class, choose the simplest legal mechanism: periodic topology;
  Dirichlet lifting/projection; incoming-characteristic or flux treatment;
  weak residual only when hard enforcement is not defined.
- Treat known boundary/forcing values over `[t_n,t_(n+1)]` as causal exogenous
  inputs; never use future solution traces.

**L3-P1: information-versus-enforcement matrix — MUST DESIGN, A3 CONDITIONAL**

- Core 2x2: neither, information only, enforcement only, both.
- Start with one controlled time-dependent inflow problem, then bump mixed
  boundaries, then one irregular external case only if boundary metadata exist.
- Parameter-match information arms or include an exact-function-matched control.
- Hold the physical boundary policy fixed within representation comparisons;
  hold representation fixed within enforcement comparisons.

Gate: constraint satisfaction alone is insufficient. Promotion requires raw
proposal improvement or a material recurrent interior gain under the six-channel
boundary decomposition and no front/admissibility harm.

**L3-P2: finite-propagation/top-left probe — MUST**

- Construct paired admissible states differing only in a compact shock,
  interior, boundary, or corner region.
- Compare one trusted solver step and one model step on one declared mesh.
- Use maximum characteristic speed plus the reference stencil width to define a
  numerical response cone.
- Measure full-model and pathway-resolved sensitivity outside that cone via
  finite differences or JVPs, then test whether the early leakage predicts
  recurrent top-left accumulation.

Gate: do not attribute the corner error to the Fourier path unless a controlled
pathway intervention reduces both early out-of-cone response and later corner
error without worsening the physical response region.

### Paper role and decision

This line owns C2. If only enforcement satisfies the trace while raw/interior
behavior is unchanged, narrow the claim to a deployment contract rather than a
rollout-improvement method.

## W26-L4: REALM Benchmark And Paper-Level Validation

### Questions

1. Does the residual framework outperform a matched direct-next-state model on
   realistic reactive-flow rollouts?
2. Are any gains architecture-independent or specific to PCNO?
3. Do standard normalized metrics hide front, integral, boundary, or
   admissibility failures?
4. Can REALM support the boundary claim, or does its released data omit the
   required patch/forcing metadata?
5. Is architectural novelty necessary after the benchmark result is known?

### Current evidence to preserve

- The supplied 52-page local paper is `arXiv:2512.18595v2`, revised
  2026-02-02, with SHA-256
  `148971e6eef3782f1562b0eaa045ce8608ded673ffa13c0bfd99acbc39f5b137`;
  no peer-reviewed venue is identified in the paper.
- REALM contains 11 stiff reacting-flow cases rather than 11 unrelated PDE
  families. Its reported failures are error growth, phase/structure loss,
  smoothing, and biased physical summaries, not a nonfinite-stability study.
- FFNO is the strongest reported regular-2D baseline; DeepONet is often strongest
  on irregular cases under a different temporal mechanism.
- The paper generally trains direct next-state models with one- or two-step
  unrolling and grouped transformed/normalized MSE.
- The v2 PDF and public metadata contain inconsistent grid/frame counts. Actual
  files and manifests must be authoritative.
- Existing `realpde_track2.py` and `train_realpde_track2_pcno.py` implement
  RealPDE Track 2, not REALM. Do not conflate the datasets or reuse semantics
  without an explicit adapter audit.

### Experiment ladder

**L4-P0: source/data/metric audit — MUST**

- Bind the exact REALM paper version, official code revision, dataset repository,
  license, file manifests, released splits, dimensions, channels, time cadence,
  mesh, boundary metadata, and preprocessing for IgnitHIT, PlanarDet, and
  ObstacleDet/SupCavityFlame.
- Reconcile prose inconsistencies against actual files without downloading the
  full corpus during A0--A1.
- Decide whether the residual target lives in physical conservative variables,
  released primitive/species variables, or transformed normalized variables;
  never call the latter physically conservative.

**L4-P1: faithful baseline — A3 CONDITIONAL**

- Reproduce the published direct-next-state training/evaluation contract on
  IgnitHIT before changing the target.
- Strong baseline families: published FFNO/direct-state, matched residual FNO
  or FFNO, and PCNO only where geometry support is scientifically relevant.
- Keep model capacity, data, preprocessing, rollout exposure, optimizer budget,
  and selection matched.

**L4-P2: residual comparison — A3 CONDITIONAL**

- Compare direct-state and residual targets on IgnitHIT with three seeds.
- Advance to PlanarDet only after data/metric replay and a registered baseline
  tolerance pass.
- Advance to one irregular case only after patch/mesh semantics and feasible
  compute are verified.

**L4-P3: horizon and structure extension — A2/A3 CONDITIONAL**

- Add survival/admissibility, front position/thickness, integral quantities,
  physical-frequency summaries, and boundary leakage where defined.
- Rolling beyond released truth tests survival only. A longer-accuracy claim
  requires additional reference trajectories.

### Paper role and decision

If the residual framework is already competitive across these regimes, retain
the simple architecture and emphasize the algorithmic/boundary/evaluation
framework. If it fails a registered mechanism gate, route only the diagnosed
fix from W26-L1--L3; do not begin an architecture sweep.

## W26-L5: Cross-Resolution Correlation And Correction

### Questions

1. Do coarse/native/fine residual discrepancies predict the signed native
   residual error across held-out cases and time?
2. Is the predictable component the persistent low-rank large-scale defect,
   rather than the shock-local component that often cancels during recurrence?
3. Can a case-independent correction improve native rollout safely?
4. Does direct off-grid PCNO beat transfer to native, native rollout, and
   transfer back after cost and transfer floors are included?

### Current evidence to preserve

- D063--D067 provide the common-source dynamic-FV contract and identify
  persistent large-scale mesh drift plus high-rank local cancellation.
- D073-A supports fixed physical differential support only at same-hidden level.
- D074--D081 show that plausible corrections can improve aggregate endpoints
  yet fail energy, local, or late-time no-harm gates.
- The bump bundle lacks the validated physical geometry required for this line.

### Experiment ladder

Let `r_2h`, `r_h`, and `r_h2` denote synchronized coarse, native, and fine
predicted increments mapped to the native grid. A minimal learned combination is

    r_corr = r_h + alpha * (r_h - P(r_2h))
                   + beta  * (R(r_h2) - r_h).

**L5-P0: common-source and transfer contract — MUST**

- Dynamic FV only; bind physical volumes, nested maps, common-source states,
  boundary policy, normalization, and transfer floors.
- Build all three predictions from representations of the same physical state.
  Do not use independently diverged rollouts in the one-step correlation test.
- Add synthetic CPU tests for restriction/prolongation, conservation of mapped
  state increments, zero correction, and closure.

**L5-P1: teacher-forced predictability gate — A2 CONDITIONAL**

- On calibration cases, measure signed correlation/cosine and regression of the
  two cross-grid differences against the native reference error.
- Report stability across case, time, component, physical band, and shock/smooth
  region. Use leave-one-case-out or a fixed calibration/evaluation split.
- Compare zero, one scalar coefficient, two scalar coefficients, and a per-case
  oracle upper bound. The oracle is never a deployable result.

Go gate: a case-independent coefficient/sign must improve held-out one-step
error materially in a preregistered component/band and avoid every registered
control. Otherwise retain the cross-grid differences as diagnostics and stop.

**L5-P2: synchronized recurrent fusion — A2 CONDITIONAL**

- Update one native state per call after evaluating all three synchronized
  representations.
- Compare raw native, corrected native, direct off-grid, and transfer-native-
  transfer-back on query-grid truth.
- Report state/front/integral/boundary metrics and evaluation cost.

Gate: no “Richardson” claim without a stable empirical order and leading-error
model. The bounded claim is cross-resolution correction or ensemble.

### Paper role and decision

Appendix/diagnostic unless the held-out recurrent correction beats raw native
and direct off-grid baselines after cost and no-harm controls.

## Weekly Run Order And Review Gates

| Milestone | Goal | Parallel work | Gate before next stage | Cost class |
| --- | --- | --- | --- | --- |
| M0 | Bind scope and avoid duplicate experiments | All five A0 audits | Every line returns current evidence, exact gaps, proposed files, and a noncolliding identity request. | Low, CPU/read-only |
| M1 | Validate the diagnostic surfaces | L1 event semantics; L2 synthetic front and gradient harness; L3 synthetic boundary/cone probe; L4 metadata audit; L5 synthetic transfer closure | Focused CPU tests pass; no current result is contradicted or silently rerun. | Low |
| M2 | Human causal review | Joint review of all five preregistrations | Select at most one new training matrix and at most two open-checkpoint evaluations. | Human decision |
| M3 | Open scientific runs | Approved L1/L2/L3/L5 checkpoint diagnostics and/or L4 baseline | Exact provenance, open-only population, smoke, and artifact manifest gates pass. | Medium; measure from smoke |
| M4 | Training confirmation | Approved three-seed no-gradient, boundary, or REALM comparison | Main metric improves and every structure/admissibility/no-harm gate passes. | High; explicit AutoDL approval |
| M5 | Integration | Claim table, figures, negative results, and next decision | No claim exceeds its family, horizon, population, or metric evidence. | Low |

The first-week target is M0--M2 plus any synthetic M1 result. M3--M4 are not
assumed to fit in the week and must not be launched merely to keep GPUs busy.

## Compact Coordination Tracker

Working labels are coordination identifiers, not stable D-series run IDs.

| Working ID | Owner surface | Priority | Initial status | Required first deliverable |
| --- | --- | --- | --- | --- |
| W26-L1-P0 | stability | MUST | COMPLETE | D087 checkpoint/provenance, representation, recurrence, policy, and event-semantics matrix |
| W26-L1-P1 | stability | MUST | COMPLETE | D087 evaluator contract; 46 focused CPU tests and Ruff checks pass |
| W26-L1-P2 | stability | A2 | COMPLETE | Paired 30-case H79 event/survival replay with exact feedback attribution |
| W26-L1-P3 | stability | A2 | COMPLETE AT DECOMPOSITION SCOPE | Paired H79 fresh/propagated result; spectra/JVP branches not selected |
| W26-L2-P0 | shock representation | MUST | TODO | Analytic moving-front capacity report |
| W26-L2-P1 | differential path | MUST | TODO | Cross-grid gradient-path instrumentation report |
| W26-L2-P2 | no-gradient training | CONDITIONAL | BLOCKED ON P0/P1 | Three-arm preregistration and costed launch request |
| W26-L2-P3 | filtering | CONDITIONAL | BLOCKED ON P0/P1 | Frozen-intervention selector |
| W26-L2-P4 | loss/noise | CONDITIONAL | BLOCKED ON DEFECT RESULT | Measured loss/exposure rationale |
| W26-L3-P0 | boundary taxonomy | MUST | TODO | Family-by-boundary contract table |
| W26-L3-P1 | boundary 2x2 | MUST-DESIGN | TODO | One controlled-task preregistration |
| W26-L3-P2 | finite propagation | MUST | TODO | Synthetic cone/leakage pilot |
| W26-L4-P0 | REALM audit | MUST | TODO | Exact-data and metric feasibility report |
| W26-L4-P1 | REALM baseline | CONDITIONAL | BLOCKED ON P0 | IgnitHIT smoke/launch request |
| W26-L4-P2 | residual comparison | CONDITIONAL | BLOCKED ON BASELINE | Matched three-seed contract |
| W26-L5-P0 | multiresolution | MUST | TODO | Common-source/transfer closure tests |
| W26-L5-P1 | correlation gate | CONDITIONAL | BLOCKED ON P0 | Cross-fitted teacher-forced preregistration |
| W26-L5-P2 | recurrent fusion | CONDITIONAL | BLOCKED ON P1 | Costed synchronized-rollout contract |

## Mentor-Agenda Coverage

| Mentor point or question | Owned by |
| --- | --- |
| Residual flow-map prediction and autoregressive framework | Claim map, W26-L4 |
| Systematic use and enforcement of boundary conditions | W26-L3 |
| Existing advection/Burgers/1D Euler/bump/dynamic benchmarks | W26-L3-P0, W26-L4 |
| REALM benchmark problems | W26-L4 |
| Publish without deliberate architecture changes if performance is strong | Claim map, W26-L4 decision |
| Literature meaning of rollout stability and whether PCNO stability is surprising | W26-L1 |
| Extend horizons and distinguish stable from unstable checkpoints | W26-L1 |
| Grid versus model cause of residual ripple/Gibbs | W26-L2-P0 |
| No-gradient-layer ablation | W26-L2-P2 |
| Classical Gibbs remedies | W26-L2-P0/P3 |
| Filtering at intermediate gradient/local/activation states | W26-L2-P3 |
| Shock-position loss and alternatives to plain L2 | W26-L2-P4 |
| Grid-noise or rollout-error denoising | W26-L2-P4 |
| Top-left error and finite propagation | W26-L3-P2 |
| Resolution-inconsistent gradient behavior and fixed physical support | W26-L2-P1/P2, W26-L5 |
| Coarse/native/fine correlation and Richardson-inspired correction | W26-L5 |

## Human Decisions Required After Kickoff

1. Confirm the exact B1 artifact and its boundary-information training and
   inference protocols.
2. Approve at most two A2 open-checkpoint evaluations after M1 review.
3. Select at most one first A3 training matrix: no-gradient, boundary 2x2, or
   REALM residual comparison. Their causal questions are different and should
   not share an opportunistic run.
4. Approve any multi-GB REALM download and every AutoDL launch explicitly.
5. Keep strength-OOD/test populations sealed until a later named A4 decision.
