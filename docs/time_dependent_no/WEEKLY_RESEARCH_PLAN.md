# Weekly Research Plan: Long-Horizon Residual Neural Operators

Updated: 2026-08-21

Status: current owner-selected planning and coordination surface. This document
does not by itself authorize checkpoint execution, dataset-scale training,
remote access, or sealed-population access. Each research line advances through
the authorization ladder below.

This plan is subordinate to [RESEARCH_DIRECTION_DECISION.md](RESEARCH_DIRECTION_DECISION.md)
for evidence and claim boundaries. Agent instructions must be derived from the
current authority documents and explicit human direction; no separate kickoff
prompt file is maintained.

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
| W26-L1: Long-horizon stability | Why do some PCNO checkpoints remain finite through H79 while others fail near H60, and what does stability mean? | D087 is complete at H79. A separately registered same-population H320 comparison is also closed: active-gradient PCNO reaches reference-free failure events later than PCFNO more often, but H80--H320 has no truth. | No implicit continuation. A separate H160 identity, spectra/JVPs, policy counterfactuals, and more bump truth require a new owner-selected claim and stable identity. |
| W26-L2: Shock representation and differential pathway | Are shock-adjacent defects spectral-retrieval-like error, finite-grid capacity/phase error, gradient-path inconsistency, or recurrent exposure error? | Review P2-C0/P2-F/P2-W0: necessary spectral transport, robust pointwise cancellation, trained-architecture reorganization, a held-phase recurrent wake, and failed global gradient scaling. | Select at most one justified selective-limiter, loss/exposure, or bump-specific no-gradient A1 preregistration after human review; D073-B remains coordinated with W26-L5. |
| W26-L3: Boundary conditions and finite propagation | Which boundary information and enforcement mechanisms are useful for each boundary class, and does global mixing seed the top-left error outside the physical domain of dependence? | Boundary taxonomy and task audit; information-versus-enforcement matrix; synthetic/local finite-propagation probe. | Multi-seed boundary training on selected classes and recurrent top-left causal test. |
| W26-L4: Benchmark and paper validation | After baseline insufficiency is removed, what fails first in one strongest-PCNO realistic reactive-flow rollout: the fresh map, recurrent on-manifold behavior, condition generalization, or a dynamically decisive structure hidden by aggregate metrics? | PD0 through G1 and D093 are closed. D094 B1-A and the 28-trajectory B1-B audit are locally retained and rehashed; B1-B selects stretched, and the fresh paired seed-0 `n={8,16,32,64,128}` ladder is running. | Execute D094 B1-C0/B1-C1 only after the isolated ladder finishes: rehash the exact matrix, then compare online train, fixed seen, fixed validation, selected/terminal rollout, and outside-audit scaling against both optimizer steps and corrected exposure. Seeds 1--2, compute/capacity controls, PlanarDet FFNO, and test remain gated. |
| W26-L5: Cross-resolution correction | Do coarse/native/fine prediction differences predict native error out of case, and can they improve a synchronized native update safely? | A43/A44-R1 qualify one bounded same-family correction protocol; A45 stops the scalar-history router. A46-A1 closes synthetic plumbing, A46-A2 fails closed on checkpoint/new-case compatibility, and A46-A2-R1 stops before generation or model construction. | No implicit branch inference or resource retry. A decisive continuation needs a compatible independent physical-node-type checkpoint and frozen new branch-label/recurrence manifests under separate authorization. |

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

The authority ladder starts at A0. An agent must end with a concrete A1, A2,
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
- A separately registered H320 continuation reuses the same 30 open cases.
  All 43 members named by five retained final manifests rehash. PCNO has later
  first inadmissibility/boundedness events on 27/30 and 28/30 pairs and remains
  finite through H320 on 30/30 cases versus PCFNO 17/30. Truth ends at H79, so
  these are descriptive recurrence events rather than later-horizon accuracy
  or physical-validity results.

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

**W26-L1-H320: separately registered reference-free continuation — COMPLETE**

The matched PCNO/PCFNO packets close at one seed and 30 open cases, with truth
supported only through H79. The
[bounded record](W26_L1_H320_REFERENCE_FREE_ROLLOUT_RECORD.md) retains exact
packet hashes, event summaries, and the two unrecoverable preregistration
identities. H80--H320 supports no accuracy, conservation, physical-validity,
asymptotic-stability, causal-gradient, seed-general, or general-operator claim.

**L1-P4: longer reference — NOT SELECTED**

- Generate or acquire longer truth only after the solver/data contract and cost
  are approved. Never infer accuracy from a truth-free extension.

### Paper role and decision

D087 supplies a main-paper-ready event taxonomy and a bounded inference-map
failure diagnostic. The later H320 packet adds only reference-free recurrence
events. Neither provides a matched causal-factor contrast or general stability
claim. W26-L1 is closed at these registered scopes; no further bump continuation
is implicit.

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

The full-PCNO fit, fixed-scale PDF/PNG visualizations, P1 instrumentation, and
frozen branch attribution are complete. They establish fixed-grid fitting plus
phase/resolution failure and show that every acute branch deletion destroys
useful capacity. Frozen attribution is not a substitute for the separately
trained architectures evaluated in P2.

**L2-P1: frozen gradient scaling audit — MUST**

- On synthetic discontinuities, record the raw least-squares gradient,
  post-neighbor aggregation, learned pre-`softsign` value, saturation fraction,
  post-`softsign` value, branch output, and decoded contribution across grids.
- Distinguish expected distributional `1/h` peak scaling from changing physical
  support and downstream nonlinear treatment.
- Compare fixed graph hops, fixed physical radius, locally scaled `h_i grad_h`,
  and a fixed-physical-scale mollified derivative as diagnostics.

**L2-P2: trained gradient ablation — COMPLETE A3**

- Three arms: full PCNO, functional no-gradient PCNO, and a parameter-matched
  pointwise/local-capacity control.
- Freeze shared initialization, data order, optimizer, presentation budget,
  selection, recurrence, and boundary policy; use three seeds.
- Evaluate registered native one-step populations, resolution transfer, ripple,
  front, and branch activation. P2 itself has no recurrent or sealed evaluation.
- Before training, evaluate the finalized P0 full-PCNO checkpoint with all eight
  frozen on/off combinations of spectral, pointwise-linear, and differential
  branches on its registered one-step populations. Report branch-local
  same-hidden responses separately from end-to-end masked responses; the latter
  allow downstream hidden states to change and therefore measure acute network
  dependence, not a unique additive cause. Repeat the frozen cube on the P2
  checkpoints for the registered one-step populations. Any recurrent cube stays
  coordinated with D073-B and W26-L5 rather than being inferred from P2.
- Score every mask and trained arm with normalized over/undershoot, outside-band
  oscillatory mass, front position/strength/thickness, smooth-region error, TV
  excess, pulse integral, physical spectrum, state error, admissibility, and
  boundary leakage. A lower high-wavenumber tail alone is not a ripple win.
- Produce matched target/prediction/error/front-normal-cut panels, physical
  spectra, paired-seed metric plots, and a frozen branch-factorial effect plot
  with common axes and physical front bands.

P2 completed all nine matched runs. No-gradient and parameter-matched local
replacement improve the locked primary in 3/3 seeds, with median gains `53.47%`
and `49.47%`, but both fail the strict causal/no-harm gate. Most violations are
coarse transfer controls; each arm also has five native phase-`0.875` step
oscillatory-mass regressions. Across seeds the restricted arms roughly halve
L2 and reduce extrema/TV, while mean step outside-band oscillatory mass rises.
The result is therefore partial: active parameter count is not the explanation,
but the differential path is not established as the ripple source. Filtering,
loss/exposure, boundary 2x2, and REALM training were not part of P2.

`W26-L2-P2-C0` is complete. Reactivating zero-output differential branches
lowers final training objective in `3/3` seeds by only `0.24%--0.74%`, but held
phase-`0.875` changes are `-0.141%`, `-0.121%`, and `+0.034%`; the `5%` rescue
gate fails. Both arms worsen their exact no-gradient parent by roughly
`2.6%--6.7%`. The pre-closure descriptive rule is therefore gradient-capacity
train refit with slight held-phase harm, not optimization rescue. The registered
verdict remains activation failure/inconclusive because CUDA step-zero shared
gradient hashes differ across every pair. A posthoc same-process audit bounds
that deterministic graph-path difference to `2.26e-7--4.46e-7` relative vector
L2 with cosine above `0.9999999999999`; it supports descriptive comparability
but cannot repair the exact gate.

The final cube shows spectral transport is necessary, pointwise/local decoding
has a `3/3` cancellation role, and the differential branch is only a small,
mixed correction with cancellation in `2/3` seeds. Learned `gw2` is material,
while `gw1` collapses from `0.01` to near-zero mostly negative values. Neither
activation nor deletion repairs coarse/fine transfer. This result does not
select raw Sobolev supervision: exact-shock gradients retain distributional
`1/h` scaling. P2 still gives P4 conditional headroom because plain L2 can
improve while a registered ripple channel worsens, but any gradient-aware loss
must be fixed-physical or shock-normalized. No additional W26-L2 GPU launch is
authorized by completion alone.

`W26-L2-P2-W0` is complete. It compares exact residual, frozen PCNO residual,
and residual error for three consecutive translated calls. The old pulse edges
near `0.389/0.639` are exact one-step residual discontinuities, not automatically
ripple. No native arm exhibits a two-phase path-wide wake. Full PCNO has a
held-pulse-only wake/regeneration pass in `2/3` seeds; no-gradient passes none.
At held phase, no-gradient lowers call-2 recurrent L2 by `44.4%` (step) and
`56.9%` (pulse), while the full model is slightly better at the trained phase.
The first defect is subcell phase and recurrence amplifies it. Frozen `/10`
gradient scaling is catastrophic (`43.8--64.6x` discontinuous L2 and
`516--601x` sine L2), so it is not a limiter. No-gradient also wins every paired
smooth-sine control, leaving the mentor's smooth-region-gradient hypothesis
unsupported for these trained architectures. Selective limiter, raw Sobolev,
and exposure training remain separate conditional studies.

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

Current owner direction on 2026-08-15 supersedes the earlier IgnitHIT-first
sequence. D088--D091 remain terminal evidence at their exact scopes; this line
does not resume, relabel, or repair them. `W26-L4-PD0` A1--A3 and its registered
causal follow-ups are complete. D092-R1 finished the repaired seed-0 width-96
5,000-step run and selected step 950. P0b then removed cumulative-`pMax`
violations without materially changing non-`pMax` recurrence error. Corrected
G0b then localized strong recurrent sensitivity to chemistry and density and a
compensating temperature interaction. G1 one-call pulses then resolve an
asymmetric mid/late chemistry-to-density persistence signature while leaving
state phase confounded with intervention magnitude. D093 subsequently closes
  the PCNO/PCFNO/residual-FFNO comparison at three and seven unique supervised
  conditions. Each seven-condition cell has lower selected truth-input error
  than its three-condition counterpart, but only FFNO has a lower selected
  free-rollout sum; one seed, one open validation trajectory, incomplete
  training cells, distinct D092/D093 source inventories, and the
  non-paper-faithful FFNO contract bound the result. The released test remains
  absent, and no further run is implicit.

### Problem anchor and claim map

Primary problem-discovery question:

> After one fixed, carefully trained residual-PCNO checkpoint passes a strong
> truth-input competence gate on released PlanarDet, does it remain accurate,
> structure-faithful, and physically valid over all 49 recurrent calls? If not,
> does the first failure lie in the fresh map, propagated-input recurrence,
> held-condition generalization, or disagreement between normalized accuracy
> and detonation observables?

The checkpoint is the best-engineered PCNO system, not a one-factor architecture
comparison: it predicts an increment in transformed normalized released fields,
begins from an identity map, uses all released training transitions, and may
use the maintained exactly implemented two-call
generated-state-exposure phase after the full-grid memory gate. The learned
increment is not a PDE-equation residual, conservative update, or physical flux.
The pinned public `deepflame-ai/REALM@7d00523dbda7823efa03c20be36692c947a417b5`
runner forms an all-trajectory batch at one shared random time index, but its
`totalTimeStep=2` loop executes only one model call.
PD0 preserves the all-seven/shared-time batching contract while making the
one-call warm-up and detached true-two-call phase explicit; it is therefore a
best-engineered baseline, not an exact executable-code reproduction.

Minimum interpretable evidence is one checkpoint that:

1. beats frozen persistence and linear-extrapolation controls on truth-input
   adjacent pairs;
2. passes finite decoding and the released-state validity contract under truth
   inputs;
3. produces complete fresh-pair, ordered teacher-forced, and free-recurrent
   records under one decoder/evaluator; and
4. reaches a preregistered competence tolerance on the official validation
   trajectory before any long-horizon mechanism interpretation.

The source-independent competence numbers are now frozen before GPU execution:
persistence `realm_npe_mean=1.1872532937574853`, normalized-state linear
extrapolation `2.7430130485367235`, and checkpoint ceiling
`0.5936266468787427` (one half of persistence). A checkpoint must meet the
ceiling, strictly beat both controls, and pass every finite/decode/admissibility/
10x-envelope/`pMax`-monotonicity gate. The GPU profile may choose only the
registered width and cost budget; it may not relax these accuracy gates.

If the competence gate fails, the result is only that this checkpoint is not a
strong PlanarDet baseline. One run and one validation trajectory cannot support
a generic PCNO/neural-operator/shock limitation, architecture superiority, seed
robustness, boundary learning, resolution transfer, conservation, solver
equivalence, asymptotic stability, or REALM-wide claim.

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
- The current public PlanarDet release is
  `realm-bench/realm-bench-PlanarDet@b084b6fc2e624e4ee5e44b88dd87d628bcbb9a4b`.
  The exact open train/validation manifest is `2,940,759,467` bytes with SHA-256
  `b53f8195f931ac46b112a76adc9c303b0c8b312cd8b054fb3fb0dbaaebbcc31e`;
  it contains seven train and one validation trajectory, while the named test
  object is physically absent. The arrays have 50 frames, 13 fields, and
  released storage `(time,channel,x,y)=(50,13,832,384)`. The fields are
  `H,H2,H2O,H2O2,HO2,O,O2,OH,T,rho,Ux,Uy,pMax`. These facts supersede the
  paper's contradictory grid, frame, trajectory, and field descriptions.
- Public accessibility is verified, but neither the REALM repository nor the
  Hugging Face dataset currently supplies an explicit code/data license. Do not
  redistribute source, derived data, or checkpoints without clarification.
- The audited model layout is `(time,channel,y,x)=(50,13,384,832)`, with
  `(Lx,Ly)=(0.0208,0.0096)` m and `2e-7` s cadence. Train-only Box--Cox/z-score
  replay uses one consistent species epsilon `1e-8`. The eight released species
  are an incomplete composition and cannot support species-sum conservation.
  `pMax` is operationally a positive cellwise-nondecreasing cumulative
  maximum-pressure memory field, not instantaneous pressure. Boundary/source
  semantics remain unavailable, so the PCNO graph is explicitly nonwrapping
  and uniform weights are an operational measure only.
- Full-resolution cost is now measured rather than inferred. At `832x384` with
  `(8,8)` modes, PCNO has 319,488 nodes, 1,275,520 directed nonwrapping edges,
  and 144 Fourier modes. The registered width-128 envelope OOMs on an idle RTX
  5090; the width-96 envelope completes the exact two-call backward and optimizer
  step with 20.58% effective headroom and a 2.42968-second measured step.
- The first width-96 diagnostic exposed a correctness defect rather than a model
  failure: BF16 autocast reused no-grad cached parameter casts in the second,
  grad-enabled call. The reviewed source disables that cache only around the
  detached first call and uses an exact chunked CUDA graph-gradient
  forward/adjoint. The source was rehashed and the width-128-then-width-96 ladder
  rerun from the beginning; this repair is runtime/resource evidence, not a
  scientific result or hidden architecture search.

### One-checkpoint evaluation contract

Use only the seven released training trajectories and one validation trajectory.
The test object remains physically unfetched and fail-closed. Fit Box--Cox and
z-score statistics and train-only boundedness
envelopes from the seven training trajectories. Freeze data order, seed,
architecture, mode periods, precision, optimizer, presentation budget,
one-/two-call schedule, validation cadence, checkpoint eligibility, and metric
algorithms before training. Finish the registered budget even if intermediate
validation is poor; retain exactly one deployed checkpoint by the frozen
eligible-validation rule. No manual retry or hyperparameter change after seeing
the validation rollout belongs to the same identity.

The three views use the same checkpoint and decoder:

| View | Input at call `t` | Scientific role |
| --- | --- | --- |
| Fresh pair | Independent truth `Z_t` for every adjacent pair | Intrinsic one-call error by time, channel, and region |
| Ordered teacher forcing | Truth is restored before every call while event order is retained | First late-state local-map failure and event localization |
| Free recurrence | Truth only at frame 0; every prediction is fed back | Deployed propagation and accepted-prefix behavior through H49 |

Fresh and ordered teacher-forced calls are the same maps with different
aggregation. Their gap from free recurrence establishes propagated-input
necessity for a threshold crossing, not a causal training mechanism. Feed back
finite invalid proposals without repair so later recovery or growth remains
visible. Stop only the affected trajectory at a nonfinite proposal and censor,
rather than fabricate, subsequent values.

Primary metrics are the independently reproduced REALM grouped normalized
prediction error and decoded spatial Pearson correlation. Always add per-call,
per-channel transformed error; decoded relative L2; free-minus-teacher gap;
normalized/decoded finiteness; inverse-transform margin; nonnegative released
species; positive density and temperature; and componentwise train-envelope
boundedness. Add pressure validity or species-sum error only if the release
semantics justify them.

PlanarDet structure metrics are `pMax` error/correlation/peak bias, detonation
front position, fixed-station arrival time and derived speed, front thickness,
mean cell size, and a fixed shock-attached spectral/high-pass view behind the
leading front. Validate every algorithm on training truth and synthetic fields
before validation. Lower high-frequency energy is not improvement if front
position, strength, thickness, or cell size regresses. Do not report
conservation without complete conservative fields, physical cell measures, and
authoritative boundary/source accounting.

### Execution ladder and authorization gates

| Stage | Exact work | Gate / disposition | Authorization |
| --- | --- | --- | --- |
| `PD0-A1` | Reusable 12/13-channel REALM Cartesian geometry, explicit Fourier periods, raw PCNO compatible with external residual reconstruction, static-memory estimate, and focused synthetic CPU tests | Linear-field gradient, identity initialization, finite backward, serialization, and fail-closed contract tests pass | Authorized by current owner direction; no real arrays/checkpoints/GPU |
| `PD0-A2` | Pin open train/validation object manifest; resolve axes, fields/groups, cadence, units, `pMax`, transform, and periods; close source/data bindings; execute the registered full-grid inference/backward/optimizer memory ladder | Complete. Width 128 OOMs; corrected width 96 is the first envelope passing the 20% headroom gate. All data gates pass and test remains absent. | Owner-authorized AutoDL preflight complete; no training occurred |
| `PD0-A3` / `D092-R1` | One seed-0, 5,000-step width-96 champion residual-PCNO history; all seven train trajectories; registered validation and eligibility; no test access | Complete. Step 950 is selected. Truth-input/free H49 NPE is `0.1244744/1.7987391`; all calls are finite and bounded, free recurrence is admissible on 7/49 calls, and truth competence fails only nondecreasing cumulative `pMax`. | Owner-authorized launch and open-validation evaluation complete |
| `PD0-P0/P0b` | Same step-950 checkpoint, exact raw-call replay, and causal cumulative-`pMax` projection before scoring/feedback | P0 is a retained CPU-metric closure failure. P0b passes all gates and removes every `pMax` decrease, but its non-`pMax` recurrence-error ratio is `0.9925744` (`near_null`); total/pressure ratios are `0.9972079/1.02203`. | Owner-authorized inference complete; projection not promoted; no automatic retry |
| `PD0-G0/G0b` | Same step-950 checkpoint and open trajectory; independently replace chemistry, temperature, density, or velocity with exact next-frame truth only after scoring the raw proposal and only for recurrent feedback | G0 is a retained pre-inference manifest-parser launch failure. Corrected G0b passes replay/isolation/H49/metric gates. Primary untouched-group ratios are chemistry `0.48403`, density `0.34435`, temperature `1.13895`, and velocity `1.06437`. | Owner-authorized inference complete; coupling diagnosis only; no oracle arm promoted and no automatic continuation |
| `PD0-G1` | Same checkpoint/trajectory; preserve the raw proposal at calls 4/12/32, use one exact chemistry or density group only in the next input, then return to raw recurrence | All six arms pass replay/isolation/H49/metric gates. Chemistry-to-density partner ratios are `0.950/0.854/0.677` versus density-to-chemistry `0.991/0.942/0.936`; only chemistry-to-density is materially persistent at calls 12/32. Phase and pulse dose remain confounded. | Owner-authorized inference complete; no oracle arm promoted and no automatic continuation |
| `D093` | Reuse D092-R1 as PCNO-7 and compare PCNO/PCFNO/residual FFNO with three or seven unique supervised trajectories at fixed seven presentations per optimizer step | Closed partial evidence. Each seven-condition cell has a lower selected truth-input sum than its three-condition counterpart; only FFNO has a lower selected free-rollout sum, and FFNO-7 is best at `1.36466/32.53910`. Three cells are incomplete; no free rollout is fully admissible. | Owner-authorized study closed at the available selected-checkpoint scope for one seed/open trajectory; no paper-faithful baseline, automatic replication, or test access |
| `D094` | Native-mesh bump PCNO/PCFNO scaling over the nested `n=8..256` ladder, with a separately qualified hundreds-case PlanarDet population required for the hard-benchmark claim | B1-A and B1-B are locally retained and rehashed. B1-B selects stretched on 28 trajectories outside checkpoint selection; all four selected checkpoints complete H79 with zero hard failures. The fresh paired seed-0 `n={8,16,32,64,128}` ladder is running and reuses the winning B1-A `n=256` endpoints. | Retrieve and rehash the complete ladder before interpretation; distinguish online train, fixed seen, fixed open-validation, selected rollout, and terminal rollout metrics; no implicit test access |
| `PD0-A4` | One frozen checkpoint/evaluator test opening | One shot after conclusions and hashes are frozen | Separate named sealed-test approval |

Current data/compute budget: the seven-train/one-validation open subset is
exactly `2,940,759,467` bytes. Its normalized-train plus normalized/native-
validation resident tensors require about 6.96 GiB of host memory before
validation workspaces, model/checkpoint state, or the evaluator's two retained
normalized rollout arrays. A2 cost one bounded profiling session, not a training
sweep. D092-R1 fixes 5,000 optimizer steps with seven microbatches per step, 490
one-call warm-up steps, 4,510 detached true-two-call steps, validation at step 1,
every 50 steps, and step 5,000, plus checkpoint eligibility from step 550. Using
the rebound 2.364844-second two-call optimizer step conservatively for every
training microbatch and validation call gives about 22.99 training GPU-hours
plus 3.25 validation GPU-hours, or 26.24 hours before I/O and metric overhead.
The frozen 36-hour wall cap leaves about 37.2% margin over that estimate. Because
the measured memory margin is
only 0.58 percentage points above the 20% gate, A3 must use an otherwise idle
matching GPU, microbatch one, and the exact frozen source/configuration.

The A2 memory gate tests exactly two same-cutoff envelopes in order, both with
`mode_counts_xy=(8,8)`, five equal-width stages, `fc_dim=128`, microbatch one,
and the intended mixed-precision/two-call execution: width 128 first
(`19,157,393` trainable parameters for 13 fields), then width 96 only if the
first lacks registered memory headroom (`10,780,401` parameters). Select the
first envelope that completes forward/backward with the frozen safety margin;
if neither does, stop before data training. The ladder executed exactly in that
order after the reviewed correctness/resource repair: width 128 still OOMs and
width 96 passes. This is a resource choice, not a validation-result
hyperparameter sweep. The A3 source binding uses a distinct REALM-PlanarDet
schema covering the adapter, PCNO core and geometry, normalization/metrics,
trainer, evaluator, and transitive imports; it does not reuse or reinterpret
`pcno_euler2d_source_snapshot_v5`.

### D092/G0 failure provenance, corrected bindings, and A2 closure

The ignored source-of-truth directory is
`artifacts/time_dependent_no/w26_l4_planardet_pd0_a2_20260815a/`. The trainer's
own fail-closed binding validator accepts the following exact chain without
opening dataset arrays or starting optimization:

| Binding | Frozen value |
| --- | --- |
| Run identity | `d092_r1_planardet_residual_pcno_seed0_width96_5000_20260815a`; seed 0; width 96; 5,000 steps; 36-hour cap |
| Preregistration | file SHA-256 `69c05a0230c881d21415c634363a956d62a8278dfa84d98c7098875fb630a5e8`; canonical payload `3b75166f954c768abc6b02ba18727388241c611a24cac273f6f726c775355747` |
| Executable source | 17-file canonical manifest `1c48afecdde31d5998695438a39b19ddb328f91b63e272c26ecaed0c30e6f502` |
| Open population | manifest payload `c52627f0b26391722899bd441331e3972917ca2b8a302fd292016d583135d4e2`; test object absent |
| Data closure | rebound final audit `2b4f6a9a87db524fe575efd92dfc1f2e357faa7cedb029c29b95f700ec35c278`; normalizer `ac2c898cd9a99ae1b1fa2f5c6a05d94d71c2180e71b7324e41a1045c0f84f223` |
| Resource closure | width-96 smoke `ea549947aa4e449c754915062a0b24f5c363a77aab0d393e938b269a3914729d`; width-128 OOM `12196282999897a0e7176ee0889fe00af3e233cedc95d7c81f80adbb778324b0` |
| Selection/competence | validation every 50 steps; eligible from step 550; ceiling `0.5936266468787427`; persistence `1.1872532937574853`; linear extrapolation `2.7430130485367235` |
| G0 failed launch | run `w26_l4_pd0_group_feedback_g0_d092r1_step950_20260817a`; source digest `b766c9aae979cc74e275a25dbc6fec976d6d5d88eb891d13e908053fe67660c1`; preregistration payload `70de092200818f4345e6ed44c406e336c4a212f0a6b008964e3931b3895e7a7c`; log/exit SHA-256 `80b87f543536b97f5d3ade0183ebc2290297d7d4927662812848ee818265c9dc` / `4355a46b19d348dc2f57c046f8ef63d4538ebb936000f3c9ee954a27460dd865` |
| G0b corrected binding | run `w26_l4_pd0_group_feedback_g0b_d092r1_step950_20260817a`; source digest `0cfb642abb2b347d47eea78bea4ced354de17aeccf98eeffaa77dc2077563cea`; evaluator SHA-256 `9edde7cfa66421c7afff51b60191f4558b98b5ec205cbaf18af77a83a69008b6`; preregistration file/payload `4a261adce15e317c762fb7bc82b8e64b3113c3e78180c8a2832c427ebdc51dc0` / `777bb7ce80f1de2ea433a2972b41df2533dbeecc5d61e2a96362bf77750ab688` |
| G0b result closure | result file/payload `a05e44a23529e7883802315c47b7fa41c3476c37a6e57106e187e8516c0260a3` / `a4e4dbf841321acde666ca882a7dd47b49d6de645de7d45904affb6290dd1259`; final manifest `385b5f7fea016f3704ccb4e96d2f0dee8558d5681585c087ee4032f25a592ce6`; all gates pass; test false |
| G0b visualization | visualizer SHA-256 `bee702b69cecfe679e52290fb198b99f3f1b93398d0f900e69e54a3af792454d`; manifest file/payload `79a84036513a24dc7635fe6718b55a4e3add04b07a6410bfa8894183b348b0b0` / `b9f863a352ba9f7c698e2f699628f1125488d02fb2b71b02676696cdc02d17cc` |
| G1 pulse binding | run `w26_l4_pd0_group_pulse_g1_d092r1_step950_20260817a`; diagnostic source `b68f4fbac053539d0d634391e9f4de74cc4cd7d5b5bb28fa1b81d8e165559d85`; evaluator SHA-256 `7b3fb15c9d608bbcb26fc81e8d77c53bd13e5a2c7f2fe00b9a362e6f07685627`; preregistration file/payload `b5f02ffecc45fbff7657b3c7a8b80080637b40369047b259ec52e16339625680` / `5fbb074f78b29a43c5b84fa6191336fb003cc53827c407828ef22d4a43efc2aa` |
| G1 result closure | result file/payload `2005c6b620a608955102c21f7b20a265fbce29c568d939e4662039d7dc5ab14d` / `0e8f9b99db53b5518d037623b28aae2cbd82d2f8fd6b9be2cca20c1a795ce423`; final manifest `bef239780ce3e54e5ea7ebd733cea7c9f1a14c5485936fb511cf1eb3d6d158e2`; all six arm/global gates pass; test false |
| G1 visualization | visualizer SHA-256 `5a4901e96022b20a871f0aad388a958c5ac314622db4ed845077716c16b47fa8`; manifest file/payload `6ea2b6bb5d9935d3704750543a64d6c1091f37bba5495178ab05ab4ecc3594ff` / `976c01828b6f531a7db08069f13d2e178d6514ccb37ca0c96ca6350350998cb6` |

The A2 rebound audit was required because its source manifest includes the
earlier corrected benchmark helper. Its schema report, train statistics, replay,
normalizer, and runtime manifest are byte-identical to the prior audit; only
source/final provenance changed. D092 then failed before its first history row
on scalar checkpoint hashing. D092-R1 preserves that failure, completed all
5,000 steps, and selected step 950. The closed evaluator exactly binds the
checkpoint and source and reports truth-input/free H49 mean NPE
`0.1244744/1.7987391`; its truth gate fails only cumulative-`pMax` monotonicity.
The public REALM `train_rollout.py` evaluator sums its five grouped per-step MSE
values across all 49 transitions without dividing by the horizon. D092-R1's
comparable free validation sum is therefore `88.1382141`, not its mean
`1.7987391`. The REALM v2 table reports PlanarDet FFNO validation `12.577`; the
PCNO value is 7.01x worse despite 20.63% more trainable parameters. This is a
same-nominal-scenario released-code comparison, not a paired reproduction or a
sealed-test ranking. The truth-input sum `6.09925` is oracle-input evidence and
cannot rank deployable autoregressive models; the paper's reported correlations
are test quantities and are not directly comparable to D092 validation
correlation.

P0 is retained as a noninterpretable metric-decoder closure failure. P0b uses a
new source/preregistration, replays all 49 raw truth-input calls plus free call 1
bitwise, changes only `pMax`, and passes all truth/no-harm/evaluator gates. It
finds `6,187,272` baseline `pMax` decreases, makes `6,201,819` corrections on
its altered recurrent path, and leaves zero. The primary non-`pMax`
ratio is `0.9925744`, total-NPE ratio is `0.9972079`, pressure-group ratio is
`1.02203`, and nonpositive-temperature calls change `21 -> 25`. The registered
effect is near-null, so `pMax` monotonicity is not a material cross-channel drift
driver here and the projection is not promoted.

G0 is retained as a pre-inference implementation failure: the launcher supplied
serialized JSON to a parser requiring an already decoded mapping, before release
data, checkpoint loading, learned calls, or output creation. Corrected G0b
preserves a distinct identity. It scores every raw model proposal before using
one exact next-frame group only in the following recurrent input. All four arms
replay baseline call 1 bitwise, isolate the selected feedback group, finish H49
with finite outputs, persist their arrays, and close the official sum identity.
The primary mean error on untouched non-`pMax` groups changes by ratios
`0.48403/0.34435/1.13895/1.06437` for chemistry/density/temperature/velocity
feedback. Chemistry and density are materially helpful and improve every
raw-scored group. Temperature improves its own next-field prediction
(`0.55372`) but worsens all untouched fields, while velocity improves itself but
has an inconclusive mildly harmful cross-group effect. These are oracle hybrid-
state interventions: they diagnose recurrent coupling and compensation but do
not establish a deployable correction or an architecture-specific cause.

G1 preserves and scores the raw proposal at each registered pulse call, inserts
one exact chemistry or density group only into the following input, and then
returns to raw recurrence. All six arms close. Post-pulse common `T+u` ratios
are chemistry `0.96534/0.92635/0.83864` and density
`0.98336/0.90248/0.86301` at calls 4/12/32. The directional partner ratios are
chemistry-to-density `0.95009/0.85390/0.67713` versus density-to-chemistry
`0.99150/0.94230/0.93567`, so only chemistry-to-density is materially persistent
at calls 12 and 32. Both directions respond immediately; their decay differs.
The result does not isolate state phase because the selected-group baseline
error, and hence the full-truth pulse dose, increases sharply with call. Error
improvement also does not imply validity improvement: chemistry call 32 has
only 4/49 admissible calls and density call 12 only 46/49 bounded calls.

### D093 architecture/exposure closeout

D093 reuses D092-R1 as `PCNO-7` and adds five cells spanning PCNO, PCFNO, and
residual FFNO at three or seven unique supervised trajectories. Every optimizer
step still accumulates seven presentations at one shared time window; the
three-condition arm repeats active conditions and both arms use the seven-
trajectory normalizer. This is condition-diversity evidence with a transductive
normalizer control, not a clean dataset-size or compute-scaling law.

Each seven-condition cell has a lower selected truth-input H49 sum than its
three-condition counterpart. Only FFNO's seven-condition cell also has a lower
selected free H49 sum (`51.50276 -> 32.53910`); PCNO changes `85.74666 ->
88.13821` and PCFNO `67.19994 -> 167.42062`. FFNO-7 is best in the matrix, but
uses the common residual contract and remains 2.59x the paper's
numerical validation value under a noncomparable contract. PCFNO is PCNO with
the gradient branch disabled, not vanilla FNO. Three cells stop incomplete
after later decoded-validation failures, and no selected free rollout is fully
admissible. The [D093 record](D093_W26_L4_PLANARDET_SCALING_RECORD.md) binds the
full table, source manifests, missing-array replay boundary, and claim limits.

### D094 hundreds-trajectory scaling registration

D094 retires the `n<=7` factorial sweep as the main scaling experiment. Its
[preregistration](D094_BUMP_SCALING_PREREGISTRATION.md) and immutable
[split](D094_BUMP_SCALING_SPLIT_MANIFEST.json) bind the 300-case bump metadata
population, a 256/44 field-blind development partition, and nested
`n=8..256` subsets. The v6 strict source set registers the adapter,
differential-branch utility, split builder, preregistration, and split. Neither
state arrays nor the historical test population were opened to construct it.

B1-A and the preregistered B1-B outside-selection audit are locally retained
and rehashed. B1-B evaluates the four already-selected `n=256` checkpoints on
the other 28 open-validation trajectories and selects stretched: paired
geometric-mean all-call error is `0.0640001` versus `0.0967684` for
prefix-tail, with zero hard failures in all four cells. This is one-seed
schedule-selection evidence, not an architecture or data-scaling conclusion.
The fresh paired stretched-schedule seed-0 `n={8,16,32,64,128}` ladder is
running serially; its B1-A `n=256` endpoints are reused. Initial CUDA and metric
health passed. The next action is retrieval and rehash after completion, not
continuous polling or partial-checkpoint interpretation.

B1-B also records a metric-scope warning that governs the scaling analysis.
For PCFNO stretched, step 15,360 to 20,480 improves fixed seen and fixed
validation one-step error by about 11% while selection rollout worsens 22% and
H79 worsens 13%. This is recurrent-objective divergence, not classical
seen/held-out one-step overfitting. The 28 audit trajectories have now informed
schedule choice and remain development evidence rather than an untouched final
holdout.

The registered B1-C route is staged. B1-C0/B1-C1 finish, rehash, and analyze
the exact seed-0 six-count PCNO/PCFNO surface. If that gate closes, B1-C2
proposes the same full ladder for seeds 1 and 2 whether the seed-0 interaction
is positive, null, or reversed. B1-C3 adds exact matched-exposure checkpoint
retention for future seeds. B1-C4 routes to extra compute only when high-count
fixed-validation and rollout curves are still improving together, or to an
objective/recurrence diagnostic when one-step improves while rollout worsens;
capacity controls follow only after extra compute plateaus. Native bump PCFNO
is not relabeled as vanilla FFNO. Matched-objective and paper-faithful FFNO
remain separate PlanarDet arms after generator/population qualification.

### Failure-decision table

| Observed pattern | Supported interpretation | Next action |
| --- | --- | --- |
| Fresh/teacher poor and free poor | Baseline, preprocessing, representation, capacity, optimization, or data problem; not a recurrence result | Stop and audit competence; do not escalate |
| Train truth-input good but validation truth-input poor | Held-condition/data-sparsity hypothesis | Report as one-case validation evidence; do not call it a long-horizon mechanism |
| Teacher good but free degrades materially earlier/faster | Propagated input is required for deployed failure | Preserve recurrence-amplification hypothesis; localize first channel/region/event |
| Aggregate metric good but front/cell/thickness wrong under teacher forcing | Standard objective/metric neglects a decisive local structure | Candidate metric--physics thesis signal |
| Teacher structure good but free structure fails before aggregate error | Recurrence destroys decisive structure before nominal metrics expose it | Strongest PlanarDet problem candidate; seek independent confirmation |
| Validity fails under teacher forcing | Local output/representation/domain problem | Diagnose before any harder case |
| Validity passes under teacher forcing but fails only in free recurrence | Recurrent physical-domain robustness problem | Record first event and accepted prefix; no repair |
| Accuracy, structure, and validity pass through H49 | PlanarDet is not a tuned-PCNO bottleneck | Escalate to ReactTGV |
| Normalized error is high but decisive observables remain correct | Metric may overweight secondary channels | Do not label the rollout a physical failure |

Observed PD0 route: truth-input competence is blocked only by `pMax`, free
recurrence has a 14.45x truth/free mean-NPE gap, and exact P0b projection is
near-null on the primary cross-channel readout. G0b now localizes a strong
chemistry-density recurrent coupling signature and a compensating temperature
interaction. G1 further localizes an asymmetric persistence signature: the
chemistry-to-density effect survives mid/late pulses, while the reverse partner
effect decays below the aggregate material threshold. It still does not
determine whether the cause is architecture, rollout exposure, loss weighting,
optimization, data coverage, state phase, or correction magnitude. Do not
escalate, propose an oracle correction, or make a bitter-lesson/architecture
  claim from this single checkpoint and trajectory. In D093, each
  seven-condition cell has lower selected truth-input error than its
  three-condition counterpart without a consistent free-rollout ordering, and
  PCNO-versus-PCFNO free ordering changes with the condition count. The minimum
  next decision is to close the line or separately choose a paper-faithful
  direct-state FFNO control, free-rollout-aware checkpoint selection, or
  seed/condition replication. None is automatically queued.

### ReactTGV escalation rule

Escalate only if PlanarDet either passes H49 cleanly or produces a competent,
sharply localized fresh-good/free-bad or metric-good/structure-bad hypothesis
that ReactTGV can test without a moving detonation front. Do not escalate after
a competence failure, unresolved release semantics, preprocessing bug,
resource failure, or unexplained single-validation-condition gap. ReactTGV
requires its own 3D/downsampling/compute/manifest preregistration and approval.

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
- W26-L5 P0/P1 establish exact common-source transfer closure, cross-fitted
  teacher-forced predictability, and a persistent low-rank residual-error map.
  A32 is the first synchronized H30 wrapper in this line to pass the reused-open
  E12/E14 state/front/integral/boundary/admissibility gate. Its active full and
  rank-8 trajectory ratios are `0.97929` and `0.93287`; audited candidate
  inference costs `2.15238x` raw logical calls.
- A33/A33-R1 preserve that correction on six reused D074 cases after mapping to
  retained 500x200 truth: full/rank-8 trajectory ratios are
  `0.98887/0.94507`, H30 ratios are `0.97770/0.89007`, all six cases improve in
  trajectory and endpoint, and maximum maintained control is `1.00459`.
  Direct fixed-hop 500x200 is about `2.01x` worse over the trajectory and has
  median H30 ratio `2.08418` versus corrected transfer; matched-information
  initialization is nearly neutral, so call-zero subcell information does not
  explain the direct failure.
- A34--A36 close three simple coefficient-transfer hypotheses. Shock-vortex
  calibration/teacher discrepancy directions agree (`0.99410` cosine), but
  their cosines with Euler1D are negative and Euler1D joint explainability is
  only `0.004301`. A fixed Euler1D early/late schedule has `-0.03422` nested
  skill and 7/16 wins; a nested three-static-descriptor map improves this only
  to `-0.02678` and 9/16, with an excessive `3.5100` coefficient and unstable
  slope norms. Neither map authorizes a capped replay or recurrence.
- A37 tests the next frozen dynamic observable without new model calls. A
  normalized causal EMA selects decay `0.5` in 8/9 strength folds and improves
  ordinary nested skill from `0.95716` to `0.96599`, but fails endpoint phase
  exclusion: 16/36 dual-held cells fail and late leave-band-out teacher skill
  is `-1.18757`. It authorizes no interior execution or recurrence.
- A38 tests the sole local signed phase portrait `[x_n,x_n-x_(n-1)]`. It raises
  ordinary nested/teacher skill to `0.98567/0.97562` with no harms and removes
  every early dual-held failure, but all nine late cells still fail; late
  leave-band-out teacher skill is `-1.72491`. A local tangent is informative
  but is not a safe late-regime clock, so no A38 model run is authorized.
- A39 tests one bounded causal path-length event clock. Ordinary nested/teacher
  skill rises to `0.99258/0.98549`, but all early and late dual-held cells fail
  (`18/36`, minimum `-5.40467`) and temporal coefficient scale varies by
  `3.07970x`. This closes further observer fitting on the open A2 rows absent
  independent evidence or a new physical measurement.
- A40 rejects paired-native batching as a faithful deployment shortcut. A41
  and A42 instead find a nearly diagonal accepted-shadow native response during
  correction-inactive coast, with unchanged E12-to-E14 response skill
  `0.99850/0.99845`. A43 freezes that response map and removes the second native
  call on calls `8--21`. It qualifies prospectively: active full/rank-8/H30
  ratios are `0.97877/0.93117/0.96249`, all four active trajectory/endpoint
  pairs win, maximum control is `1.01141`, and the candidate uses 548 calls
  versus optimized A32's 604. Measured forward time remains `1.80245x` raw, so
  this is lower logical cost rather than a demonstrated latency advantage.
- A44-R1 keeps A43 unchanged and opens 14 disjoint correction cases from E00
  and E11 inside checkpoint training support. It qualifies: active
  full/rank-8/H30 ratios are `0.98216/0.95586/0.97873`, all four active
  trajectory/endpoint pairs win, and maximum control is `1.04361`. Large/rank-8
  views improve, whereas transition/local views are `1.00095/1.00253`; this
  confirms a modest broad low-rank recurrent benefit without establishing
  independent checkpoint/test or cross-family transfer. Candidate forward time
  is still `1.79933x` raw.
- A45 uses no new model call and strictly lags three inference-available
  displacement/response scalars. Pooled full/rank-8 logged-utility skill over
  zero/phase is `0.57844/0.49571`, and all coefficient signs/scales are stable,
  but E00 skill over phase is `-0.30058` and coast is `-3.03163`; the
  every-group/both-route gate fails. Across 32 case-band cells, full/rank-8/
  large ratios are `0.98037/0.94410/0.97853`, transition/local are
  `1.00097/1.00258`, and local utility is negative in all 32. This supports a
  predictable broad component plus local harm, not a router: the packets lack
  same-state exact/coast counterfactuals and no recurrent policy was simulated.
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

Current P2 state: A33-R1 qualifies A32 transfer on reused open 500x200 truth and
rejects direct fixed-hop 500x200 as a deployment path at this scope. The state
round-trip floor is substantial (`0.08964` scaled RMS), but the mapped evolution
error is numerically orthogonal to it; correction reduces the evolution term
without floor cancellation. The audited corrected path costs 456 calls and
`432.70` forward seconds versus raw transfer's 180 calls and `102.25` seconds;
the separately reported raw-shadow cache lower bound is 456 unique calls for
raw plus corrected, not a measured optimized implementation. Direct physical-
radius PCNO remains absent unless W26-L2 supplies a deployable D073-B recurrent
wrapper; D073-A is not a rollout arm. Independent confirmation and every sealed
population remain outside this ladder until an explicit named authorization.
The A34--A39 failures further rule out treating the vortex coefficients,
elapsed time alone, or three static Euler1D initial-state descriptors as a
deployable router, and show that neither stationary smoothing nor a local
discrepancy tangent nor the registered bounded event clock solves endpoint
phase transfer. Further fitting on these open A2 rows is closed; true residual
error remains offline scoring information. Re-entry requires independent
evidence or a new physical measurement, while practical work may reduce the
cost of the already-qualified A32/A33 shadow protocol. A43 reduces that
protocol to 548 calls, and A44-R1 confirms it unchanged on disjoint correction
cases in E00/E11 checkpoint training support. A43/A44-R1 are frozen. A45 then
rejects the registered scalar-history router despite strong pooled prediction:
E00 and coast do not improve over the known phase baseline. Further fitting on
these eight opened active cases is closed. The minimum decisive continuation
is a newly registered same-state exact/coast branch experiment: predictors
must be available before the fine call, true error remains label-only, and no
recurrent controller may run until a held-group gate passes. A later recurrent
test requires newly generated, checkpoint-independent dynamic-FV evidence.
A46-A1 now freezes the exact same-state estimand, an always-active exact-refresh
probe, the cheap post-shadow/pre-accepted-native feature clock, grouped gate,
and `30 + 2*k` scheduler cost identity. Fifteen synthetic checks and 18 focused
CPU tests pass. No scientific row was generated: the independent compatible
checkpoint and new branch-label/recurrence manifests remain deliberately
unbound. The completed A46-A2 metadata preflight authenticates the D060
reference and 52-case opened inventory but rejects all ten available checkpoint
candidates: literal-zero or omitted node-type channels (plus G1/S1 boundary
inputs) do not match the physical dynamic-FV interface, checkpoint bytes are
not locally available to rehash, and no new-case manifest exists. Its 27 CPU
tests and Ruff checks pass, with zero model/data-array/remote/controller
activity. Branch inference remains closed. A46-A2-R1 then stopped twice on broad
isolated-source imports and once on interpreter environment discovery, all
before data, reference generation, model construction, training, or outcome
opening. Its latest 18-case plan remains frozen with every `generated` and
`outcome_opened` flag false. These are infrastructure receipts, not a scientific
continuation. Any new executable scope must first bind one fresh D060-compatible
checkpoint and freeze newly generated, disjoint branch-label and recurrence
case manifests without opening outcomes.

The A46 A1/A2 source records also bind
`W26_L5_SAME_STATE_REFRESH_COUNTERFACTUAL_PREREGISTRATION.md` at SHA-256
`3bc057aa3b337ce09ae6c80c36cf23c6d863d15c73ed56d62ce13b802ca2419f`,
but its bytes are absent from the checkout, Git history, and retained archives.
This limits exact old-identity replay without retracting the bounded
synthetic/preflight outcomes. Do not reconstruct it; any future activation needs
a new preregistration, identity, and source manifest. See the complete
[HANDOFF gap table](HANDOFF.md#unrecoverable-preregistration-gaps).

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
| W26-L1-H320 | stability | SEPARATELY REGISTERED | COMPLETE; REFERENCE-FREE AFTER H79 | Matched 30-case PCNO/PCFNO recurrence through H320; all five retained packets rehash, but no H80--H320 truth or causal-gradient claim |
| W26-L2-P0 | shock representation | MUST | COMPLETE | Analytic moving-front capacity report and fixed-scale visualizations |
| W26-L2-P1 | differential path | MUST | COMPLETE | Cross-grid gradient-path instrumentation and frozen 2^3 branch attribution |
| W26-L2-P2 | no-gradient training | FIRST A3 MATRIX | COMPLETE; PARTIAL RESULT | Three paired seeds passed the primary but failed the strict no-harm gate |
| W26-L2-P2-C0 | zero-output gradient continuation | A3 | COMPLETE; STRICT INCONCLUSIVE, DESCRIPTIVE PHASE OVERFIT | Six runs, final frozen branch cube, nine PDF/PNG figure pairs, and posthoc numerical pairing audit |
| W26-L2-P2-F | all-nine frozen branch cube | A2 | COMPLETE; STRICT REPLAY FAIL, DESCRIPTIVE COMPATIBILITY PASS | 216 finite predictions and seven PDF/PNG figure pairs; spectral and pointwise paths are essential, full differential cancellation passes 3/3, matched local is near-neutral |
| W26-L2-P2-W0 | moving-front residual wake | A2 | COMPLETE; PHASE-SPECIFIC RECURRENT DEFECT | Three-call true/predicted residual and error decomposition; full held pulse passes wake/regeneration in 2/3, no-gradient passes none, global `/10` scaling fails |
| W26-L2-P3 | filtering | CONDITIONAL | DEFERRED BEHIND P2 | Frozen-intervention selector |
| W26-L2-P4 | loss/noise | CONDITIONAL | DEFERRED BEHIND P2 AND DEFECT RESULT | Measured loss/exposure rationale |
| W26-L3-P0 | boundary taxonomy | MUST | TODO | Family-by-boundary contract table |
| W26-L3-P1 | boundary 2x2 | MUST-DESIGN | TODO | One controlled-task preregistration |
| W26-L3-P2 | finite propagation | MUST | TODO | Synthetic cone/leakage pilot |
| W26-L4-P0 | REALM audit | MUST | COMPLETE FOR PLANNING | Paper, public release, local source, and retained-manifest audit; D088--D091 remain terminal at their exact IgnitHIT scopes |
| W26-L4-P1 | IgnitHIT direct baseline | HISTORICAL | TERMINAL INCOMPLETE UNDER D088--D090 | Sparse decode and later boundedness failures; no completed baseline or residual comparison |
| W26-L4-P2 | IgnitHIT residual comparison | HISTORICAL | NOT RUN; SUPERSEDED BY CURRENT PD0 DIRECTION | No result and no implicit queue |
| W26-L4-PD0 | PlanarDet champion PCNO | MUST | A1--A3 + P0b + G0b + G1 COMPLETE; ASYMMETRIC RECURRENT PERSISTENCE LOCALIZED | Exact 2,940,759,467-byte open tree, resource ladder, D092-R1 training, step-950 selection, H49 evaluation, and three causal diagnostics are closed. Truth/free mean NPE is `0.1244744/1.7987391`; the comparable free horizon sum `88.13821` is 7.01x the paper FFNO validation value. P0b is near-null. G0b localizes chemistry/density sensitivity; G1 finds chemistry-to-density-only material partner persistence at calls 12/32, with phase/dose confounded. No intervention is promoted and test remains absent. |
| D093 | PlanarDet architecture/exposure | REGISTERED FOLLOW-UP | CLOSED; PARTIAL SINGLE-SEED/OPEN-VALIDATION RESULT | PCNO-7 result anchor plus five PCNO/PCFNO/residual-FFNO cells at three/seven unique conditions. Every seven-condition cell has lower selected truth-input error than its three-condition counterpart; only FFNO has a lower selected free-rollout sum. Three cells are incomplete, no free rollout is fully admissible, the normalizer control is transductive, D092/D093 source inventories differ, and test remains absent. |
| D094 | hundreds-trajectory scaling | REGISTERED; SEED-0 LADDER RUNNING | B1-A/B1-B RETAINED; B1-C ROUTE REGISTERED | Bump 300-case metadata population and 256/44 development split remain bound without historical-test opening. B1-B selects stretched on the 28 trajectories outside checkpoint selection and exposes one-step/recurrent divergence. Fresh paired `n={8,16,32,64,128}` cells run serially and reuse the winning B1-A `n=256` endpoints. Next: exact B1-C0 retrieval/rehash and B1-C1 fixed-compute/exposure analysis; seeds 1--2 and conditional controls require review after that gate. |
| W26-L5-P0 | multiresolution | MUST | COMPLETE | Common-source/transfer closure tests and frozen gradient-policy decision |
| W26-L5-P1 | correlation gate | CONDITIONAL | COMPLETE FOR REUSED OPEN POPULATIONS | Cross-fitted teacher-forced gate plus low-rank residual-error structure analysis |
| W26-L5-P2 | recurrent fusion | CONDITIONAL | A44-R1 QUALIFIED; A45 STOPPED; A46 BLOCKED BEFORE SCIENTIFIC EXECUTION | A43/A44-R1 remain frozen bounded evidence. A45 fails E00/coast router transfer. A46-A1 closes synthetic same-state plumbing, A46-A2 finds no compatible independent checkpoint/new-case manifest, and three A46-A2-R1 infrastructure attempts stop before generation/model construction. No branch result or automatic resource retry. |

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

## Current Human Decisions

1. Decide whether the closed W26-L2 evidence warrants at most one separately
   preregistered follow-up.
2. Decide whether to start the still-unexecuted W26-L3 boundary taxonomy and
   synthetic finite-propagation line.
3. Treat D093 and the H320 recurrence as closed at their bounded scopes. Let the
   authorized D094 seed-0 ladder finish, then execute B1-C0 retrieval/rehash and
   B1-C1 analysis before interpreting the data--architecture--optimization
   surface or reviewing the seeds 1--2 proposal.
4. Close A46 at the independent-checkpoint/new-case block or separately repair
   and re-register its portable resource build.
5. Keep sealed and strength-OOD populations closed unless a later named
   one-shot decision opens them.
