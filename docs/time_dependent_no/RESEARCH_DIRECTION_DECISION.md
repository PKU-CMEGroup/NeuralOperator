# Time-Dependent Neural Operators: Current Research State

Updated: 2026-08-21

Status: current human-review snapshot; evidence and planning, not executable
authorization

## Authority And Change Control

Current explicit human direction takes precedence over this repository snapshot.
This file records the last reconciled scientific state, claim boundaries, and
standing owner constraints. It does not grant or deny permission for a future
experiment by itself.

Documentation roles are:

- [HANDOFF.md](HANDOFF.md): replaceable operational snapshot;
- [MECHANISTIC_DIAGNOSTIC_TRACKER.md](MECHANISTIC_DIAGNOSTIC_TRACKER.md):
  compact experiment index;
- [WEEKLY_RESEARCH_PLAN.md](WEEKLY_RESEARCH_PLAN.md): current research lines,
  dependencies, gates, and prospective work;
- [D093_W26_L4_PLANARDET_SCALING_RECORD.md](D093_W26_L4_PLANARDET_SCALING_RECORD.md):
  bounded closed evidence for the PlanarDet architecture/exposure comparison;
- [W26_L1_H320_REFERENCE_FREE_ROLLOUT_RECORD.md](W26_L1_H320_REFERENCE_FREE_ROLLOUT_RECORD.md):
  bounded closed evidence for the reference-free H320 bump recurrence;
- [D094_BUMP_SCALING_PREREGISTRATION.md](D094_BUMP_SCALING_PREREGISTRATION.md):
  registered scaling plan with a reported remote pilot but no locally retained
  or rehashed calibration packet;
- [README.md](README.md): onboarding and maintained-code navigation; and
- [the 2026-08-11 decision archive](history/RESEARCH_DIRECTION_DECISION_through_2026-08-11.md):
  byte-preserved exact evidence, contracts, terminal narratives, and prior
  decision language replaced by this compact surface.

Historical gates close their exact registered attempts, not entire method
families. Codex-proposed sequencing, stop rules, or next steps are nonbinding
unless the human owner adopts them. New scoped owner direction supersedes older
direction; documentation should then be reconciled. Private paths, hosts,
credentials, and machine-specific dataset locations stay in ignored local
context.

## Current Verdict By Evidence Family

The overall result-to-claim verdict is **partial, with high confidence**. The
campaign supports useful fixed-family baselines and a bounded mechanistic map.
It does not establish a generally improved, shock-stable, geometry-aware neural
operator solver.

| Program | Current supported scope | Binding nonclaim |
| --- | --- | --- |
| 1D flow maps | Larger learned macro steps can win after fewer recurrent compositions; the preferred stride depends on horizon and metric. | No universal stride, learned CFL limit, timestep transfer, native-solver resolution transfer, or mesh-invariant solver. |
| CPGNet | Corrected 1D controls support message reach rather than width alone; causal boundary training improves the local 2D release-bundle result without closing the oracle gap. | No paper-table reproduction, exact DG replay, physical interface trace, conservation, seed robustness, or architecture transfer. |
| Supersonic bump | D041 remains the exact historical comparator. B1 misses replacement parity; attached-K2 improves B1 recurrence, mostly through a better propagated path. D084 closes finite inadmissibility without a registered global explosion. D087 then separates four event horizons on a common declared B1/D019 population: B1 stays admissible and bounded through H79, while D019 loses accuracy first and later becomes inadmissible and unbounded without becoming nonfinite. Exact deployed-map decomposition shows propagated-input response dominates D019's realized late error. | No same-contract K2 comparison with D041, attached-gradient causality, fresh-ripple cure, causal training/representation/boundary mechanism, general stability, promotable collar encoding, physical conservation, PDE resolution transfer, broad geometry generalization, independent rotated PDE solve, or cross-family transfer. |
| Dynamic finite-volume shock-vortex | D044 is a useful one-seed baseline; D060 improves state error but fails joint front/high-pass promotion. D063--D081 establish bounded common-source transfer, a composite mesh-defect diagnosis, differential-support evidence, and one deterministic open-validation correction result with exact same-population process reproduction. W26-L5 A32/A33 show that a target-free raw-shadow tether retains a modest low-rank recurrent gain after transfer to retained 500x200 truth on all six reused D074 cases: full/rank-8 trajectory ratios are `0.98887/0.94507`, all six trajectory/endpoint pairs improve, and the maximum maintained control is `1.00459`. Direct fixed-hop 500x200 loses decisively to corrected transfer (median H30 ratio `2.084`, zero of six wins), while matched-information initialization is nearly neutral. A34--A39 reject a universal vortex/Euler1D coefficient geometry, two simple Euler1D routers, and three prospective phase observers despite strong ordinary same-family fits. A41/A42 instead identify a nearly diagonal accepted-shadow native response on correction-inactive coast (`0.99850/0.99845` E12/E14 skill). A43 freezes that response map, removes the second native call on coast, and qualifies prospectively: active full/rank-8/H30 ratios are `0.97877/0.93117/0.96249`, all four active trajectory/endpoint pairs win, maximum control is `1.01141`, and candidate cost falls from 604 to 548 logical calls versus optimized A32. A44-R1 then applies A43 unchanged to 14 disjoint correction cases in E00/E11 checkpoint training support: active full/rank-8/H30 ratios are `0.98216/0.95586/0.97873`, all four active trajectory/endpoint pairs win, and maximum control is `1.04361`. Large-scale/rank-8 views improve while transition/local views are neutral to slightly harmful. A45 finds stable pooled prediction of logged utility from lagged scalar displacement/response history, but E00 and coast fail the registered transfer gate. Its 32 case-band cells strengthen the broad/rank-8 benefit versus local-harm decomposition without qualifying a router or causal fine-refresh estimate. Node-type and always-on local-filter questions are closed at their registered scopes. | No resolution invariance, proof that PCNO is an operator, unique native fixed-hop cause, safety beyond the opened cases, independent checkpoint/test or statistical confirmation, family-independent coefficients or response maps, deployable scalar-history router, causal exact-versus-coast estimate, a deployable Euler1D correction, practical latency advantage for the correction, physical conservation, general node-type necessity, optimal boundary encoding, native-solver equivalence, other-PDE transfer, or sealed performance. |
| Latent forecast | Discontinuous decoder regularity improves front fidelity, but the tested family is decoder-capacity limited. | No latent transition, autonomous recurrence, geometry transfer, or data-assimilation result. |
| REALM IgnitHIT | D088 shows that lower mean fresh-map error can hide a sparse inverse-domain tail. D089's species-domain link prevents that registered species-domain failure under its separate history, but D090 shows the H29 map still becomes unbounded while remaining finite and admissible; its sampled-pair training loss and rollout quality move in opposite directions. | No completed direct baseline, residual comparison, sealed test result, PlanarDet result, causal link effect, offending-channel attribution, general FFNO/direct-map rejection, or benchmark-wide claim. |
| REALM PlanarDet | D092-R1 completed the seed-0 width-96 residual-PCNO run and selected step 950. On the one released validation trajectory, truth-input/free H49 mean NPE is `0.1244744/1.7987391`; the corresponding released-code horizon sums are `6.09925/88.13821`. P0b is near-null on non-`pMax` drift, while G0b/G1 localize a checkpoint-specific chemistry-to-density persistence asymmetry. D093 reuses this PCNO-7 result anchor and adds PCFNO/FFNO at three and seven unique supervised conditions. Each seven-condition cell has lower selected truth-input error than its three-condition counterpart, but only FFNO has a lower selected free-rollout sum; FFNO-7 is best in the matrix at `1.36466/32.53910` truth-input/free H49 sums. | The D092 paper-value ratio is validation-to-validation context, not a paired reproduction or sealed ranking. D092 and D093 use distinct source inventories, and D093 has one seed and one open validation trajectory; the three-condition arms still use seven-condition normalization and seven presentations per optimizer step. PCFNO is not vanilla FNO and the residual FFNO is not paper-faithful. The results do not establish a physical causal graph, clean data scaling, architecture superiority, gradient-branch stability, seed robustness, conservation, or a REALM-wide claim. |

The supersonic-bump and Mach-1.1 dynamic finite-volume programs remain separate.
D053b's mostly fresh smooth-high-pass defect is dynamic-family evidence;
K2D0's bump high-pass improvement is mainly propagated-path evidence. Neither
may be used as evidence for the other without a new matched experiment.

## Bounded Line 3 Evidence Index

Exact metrics, hashes, contracts, alternatives, and terminal run narratives are
recoverable by ID from the
[2026-08-11 decision archive](history/RESEARCH_DIRECTION_DECISION_through_2026-08-11.md)
and the [compact tracker](MECHANISTIC_DIAGNOSTIC_TRACKER.md).

| Amendment | Bounded evidence retained here | Binding limit / disposition |
| --- | --- | --- |
| A-01: B1, attached-K2, K2D0 | B1 is serious training but fails historical D041 parity. Attached-K2 improves B1 H20--H79 recurrence; K2D0 attributes most endpoint improvement to a smaller propagated term on an already-better incoming-error path, with little fresh-defect change and essentially unchanged finite-amplitude gain. K2T0 records are harness failures. | No training-factor causality, exact D041 same-contract comparison, stronger contraction, or one-step ripple cure. |
| A-02: D063 resolution | One unchanged D060 checkpoint runs on three nested grids under a common-source restriction contract; native training resolution is sharply best. Exact-input commutators are small while free-rollout commutators grow with trajectory gaps; off-grid front, boundary, and physical-band errors regress. | Useful bounded zero-shot transfer, not resolution invariance, a general operator, or node-type necessity. |
| A-03: D064--D067 structure | Same-run identities support a composite defect: persistent low-rank large-physical-scale mesh drift plus high-rank shock/vortex-local structure that mostly cancels or corrects during recurrence. State response is often corrective; endpoint attribution is mesh dominated. | No unique branch/kernel cause, pure translation account, accepted independent replicate, correction result, or resolution-consistent learning claim. |
| A-04: D068--D069 node types | Frozen interventions show checkpoint use of family-local type semantics; wall labels dominate the tested native bump effect. Matched training does not support a benefit from appending permanently zero channels. | Closed at this scope; no universal necessity, optimal encoding, boundary-condition improvement, or geometry generalization. |
| A-05: D072 semantic fields | Fixed-width union/semantic collars and exact matched initialization are verified. One dynamic FP32 seed causally uses the fields, but neither the three-seed dynamic BF16 matrix nor bump matrix establishes the registered N0-relative rollout gain. | D072 is complete and not promoted. Re-entry needs a new owner claim and same-precision multi-seed evidence. |
| A-06: D084 inadmissibility | All 360 H79 continuations remain finite. Exact repairs occur but are not durable; inadmissibility precedes some local growth yet neither implies nor equals global blow-up. D082 fields materially affect recurrence without passing the completion/error conjunction. | D084 is terminal; D082 is not promoted; no queue or implicit repair follow-up remains. |
| A-07: D083/D085/D086 representation | Severe query compression is strongly negative. D083 is a transported-Fourier analytic covariance control; D085's fixed-world Fourier test fails at call 1 through a material same-input transformed-representation defect; D086 shows broad failure morphology and recurrent sensitivity. | No physical conservation, independent rotated PDE solve, Fourier-only causality, accepted rollout, broad geometry/rotation generalization, or general coarse-mesh result. |
| A-08: D087 stability forensics and H320 recurrence | On the exact common declared 30-case bump population, B1 has later `T_accurate` on every pair and is admissible/bounded/finite through H79. D019 loses accuracy 29--57 calls before inadmissibility, becomes unbounded 3--5 calls after inadmissibility, and remains finite through H79. Its invalid active-node state is fed back without repair; exact common-coordinate decomposition shows propagated response dominates while fresh defect stays small through H60. A separately registered H320 comparison then records later active-gradient PCNO events than PCFNO more often than the reverse: 27/30 admissibility, 28/30 boundedness, and 13 later plus 17 tied finiteness events. | H80--H320 has no truth and is reference-free recurrence only. D019 training membership/provenance is unresolved, and representation, learned map, training, recurrence, and boundary policy differ. No H320 accuracy, physical validity, OOD, causal gradient factor, Lipschitz/JVP, asymptotic stability, conservation, seed generality, or cross-family claim. See the [bounded H320 record](W26_L1_H320_REFERENCE_FREE_ROLLOUT_RECORD.md); no separate H160 identity, spectra/JVP, or other mechanism branch is queued. |

## Compact Unified Claim Register

RC-01--RC-15 retain their registered substance. The exact prior decision surface
is byte-preserved in the 2026-08-11 archive; later amendments do not silently
create new claim IDs.

| ID | Verdict and allowed claim | Binding nonclaim |
| --- | --- | --- |
| RC-01 | Yes, bounded: larger learned macro steps can beat repeated smaller steps, consistent with a tradeoff between one-call difficulty and recurrence count. | The winner depends on horizon and metric; no universal stride, CFL, timestep-transfer, ripple, or mesh-invariance result. |
| RC-02 | Yes, bounded: corrected 1D CPGNet stability depends materially on message reach rather than width alone. | No paper-faithful 2D mechanism, physical trace, exact DG replay, conservation, seed, or architecture-transfer result. |
| RC-03 | Yes, bounded: D044 is a useful raw-rollout macro-map baseline on one frozen dynamic family. | One seed; strength/test sealed; recurrence is not conservative by construction. |
| RC-04 | Partial for state; no for joint promotion: D060 lowers H60 state error with fewer recurrent calls. | Front/high-pass conjunction fails; D060 is not uniformly better, ripple-stable, or promoted. |
| RC-05 | Bounded composite mechanism: no isolated Fourier, pointwise, differential, boundary-only, or strong-cancellation account explains the tested ripple evidence. | No exact layerwise cause, exclusion of all aliasing, universal mechanism, or cross-family transfer. |
| RC-06 | Yes, bounded: on D053's dynamic cohort, full-state error is mostly propagated while smooth high-pass error is mostly freshly regenerated. | Not a bump result or universal neural-operator law; no matched smooth control. |
| RC-07 | No for exact D048: native face-target fit is insufficient until the decoded update is accurate and admissible. | Other projected, divergence-conditioned, or impulse representations remain untested. |
| RC-08 | Locator yes; exact correction no: D055 identifies a legal sparse defect-support signal, while D056 lacks safe late-horizon correction headroom. | The result does not reject every conservative local correction. |
| RC-09 | No safe late-OOD continuation for exact static directions: jointly descendable objectives can transfer with geometry-dependent sign. | No generic optimizer-failure claim; finite-step and state-dependent weighting remain open. |
| RC-10 | No for exact D061: scalar state blending does not combine D044 and D060's complementary gains. | Learned state-dependent multirate maps remain untested. |
| RC-11 | No for exact D062: independent-row argmax front identity is unstable when pressure-jump branches exchange identity. | Connected multi-front, level-set, or topology-aware charts remain untested. |
| RC-12 | Yes, bounded: the tested latent route is decoder-capacity limited; discontinuous regularity helps but is insufficient. | No transition, autonomous recurrence, geometry transfer, assimilation, or latent-operator result. |
| RC-13 | Diagnostics yes; learned solver no: physical balance diagnostics require audited volumes, faces, normals, orientation, and boundary accounting. | D044/D060 are not flux-conservative solvers; bump proxy weights are not physical conservation evidence. |
| RC-14 | No: a general shock-stable geometry-aware neural operator is not established. | Independent seeds, new geometries, matched smooth controls, sealed evidence, and a promoted method are absent. |
| RC-15 | Partial, high confidence: the campaign gives a coherent bounded map of useful baselines, distinct defects, and failed necessary conditions. | No positive replacement architecture and no isolated discontinuity-specificity claim. |

## Standing Human Constraints

These constraints protect data, claim validity, and reproducibility until later
explicit owner direction changes them.

1. Strength-OOD and test populations remain sealed until the owner explicitly
   opens a named evaluation.
2. Private paths, remote-machine details, credentials, datasets, checkpoints,
   and large generated outputs remain outside committed documentation.
3. Bump, dynamic FV, CPG, and latent populations are not pooled; mechanisms do
   not transfer between them without evidence.
4. Bump equal-node or reconstructed weights are diagnostic proxies. Physical
   conservation requires validated volumes, faces, normals, orientation, and
   boundary accounting.
5. Validation results do not become test or state-of-the-art claims.
6. Historical attempts retain their checkpoint, source, data, optimizer,
   recurrence, evaluator, population, metric, threshold, cost, and stop-rule
   identities. A changed attempt receives a new identity.
7. One-step fit is an entry diagnostic, not a rollout proxy. Jointly report
   admissibility, state error, front position/strength/thickness, smooth
   high-pass behavior, and boundary behavior.
8. High-frequency energy is a band diagnostic, not synonymous with ripple or
   instability. Filtering claims require anti-smearing front controls.
9. Truth-free continuation supports qualitative finiteness/admissibility only,
   not longer-horizon accuracy.
10. Failed harnesses are provenance, not negative method evidence.
11. Boundary-representation comparisons separately freeze the physical
    boundary-condition family and the checkpoint/evaluator recurrence and
    boundary-handling policy. Encoding work cannot select or modify clamping,
    splicing, closure correction, or boundary objectives owned by the boundary
    workstream. D060 and dynamic D072 use `model_all_nodes`; bump D072 uses
    `causal_nodal_physical`.
12. Required animations and spatial views complement, not replace, per-case
    metrics. They bind case, resolution, physical time, recurrence,
    intervention, units, color scale, and visualization-only subsampling; model
    comparisons use common predeclared per-field scales.
13. Available local or AutoDL compute is not authorization. Planning, preflight,
    synthetic CPU validation, training, checkpoint evaluation, remote work,
    downloads, and sealed-population access retain their explicit boundaries.

## Shared Reporting Semantics

- Hk means k recurrent calls joining k+1 saved states; completion means
  `valid_length == k`.
- Here residual/update is the denormalized one-call conservative-state increment
  `F_h(U) = N_h(U) - U`, not a PDE-equation residual.
- Form commutators and state/update views after denormalization. Use audited
  physical-volume metrics where available and label bump proxies explicitly.
- Mean survival is `mean(valid_length / k)` and need not equal completion.
- Error curves contain only the accepted prefix; mark the first excluded
  inadmissible proposal.
- Report the six channels needed for interpretation: state, shock/front,
  conservation or proxy balance, positivity/admissibility, boundary leakage,
  and spectral/high-pass behavior.
- Every animation records trajectory, physical horizon, learned stride, encoded
  states, recurrence, units/scales, and visualization-only subsampling.
- Checkpoint selection, precision, seed, sample presentations, optimizer
  history, recurrence, evaluator, and source/artifact digests are part of the
  result.
- A sharp physical front legitimately contains high-frequency content; lower
  high-pass energy is not improvement when caused by blurring or lost strength.

## Current Owner-Selected Research Program

The 2026-08-11 mentor-derived program is defined in
[WEEKLY_RESEARCH_PLAN.md](WEEKLY_RESEARCH_PLAN.md):

| Line | Current question |
| --- | --- |
| W26-L1: long-horizon stability | Why do some PCNO checkpoints remain finite while others fail, and how should accurate, admissible, bounded, and finite horizons be separated? |
| W26-L2: shock representation and differential pathway | Are shock defects spectral retrieval, finite-grid capacity/phase, gradient-path inconsistency, or recurrent exposure effects? |
| W26-L3: boundary conditions and finite propagation | Which causal boundary information and enforcement mechanisms improve recurrent interior dynamics, and why does top-left error accumulate? |
| W26-L4: benchmark and paper validation | After baseline insufficiency is removed, what fails first in one strongest-PCNO realistic reactive-flow rollout: the fresh map, recurrence, held-condition generalization, or decisive structure hidden by aggregate metrics? |
| W26-L5: cross-resolution correction | Can coarse/native/fine discrepancy predict and safely correct synchronized native updates out of case? |

Agent work begins under repository-root `AGENTS.md` and current explicit human
direction. No planning document independently launches training, checkpoint
evaluation, remote work, dataset download, or sealed evaluation.

On 2026-08-15 the owner superseded the earlier IgnitHIT-first sequence and
selected the `W26-L4-PD0` PlanarDet problem-discovery ladder in
[WEEKLY_RESEARCH_PLAN.md](WEEKLY_RESEARCH_PLAN.md). The selected system is a
best-engineered residual PCNO rather than a one-factor architecture comparison.
The exact seven-train/one-validation open tree, A1 implementation, A2 full-grid
resource ladder, and seed-0 D092-R1 5,000-step training are complete. Width 128
OOMs; width 96 passed with 20.58% effective memory headroom. D092-R1 selected
step 950 under the frozen validation rule. The closed open-validation evaluator
reports truth-input/free H49 NPE `0.1244744/1.7987391`; all 49 calls in both
views are finite and bounded, truth inputs are otherwise competent, and free
recurrence is admissible on 7/49 calls. The only truth-input competence failure
is nondecreasing cumulative `pMax`, so the original evaluator correctly withheld
mechanism interpretation.

The first frozen `pMax` projection attempt P0 is retained as a noninterpretable
implementation failure: GPU-side comparison did not close the exact CPU metric
decoder. P0b corrects that path under a new source and preregistration, exactly
replays all 49 raw truth-input calls plus free call 1, changes only `pMax`, and
passes every evaluator, truth-competence, no-harm, finiteness, and physical-floor
gate. Baseline free recurrence contains `6,187,272` `pMax` decreases; P0b makes
`6,201,819` causal corrections on its altered path and leaves zero. Yet the primary
non-`pMax` grouped-NPE ratio is `0.9925744` and total-NPE ratio is `0.9972079`;
the former is preregistered near-null. Pressure-group NPE rises by 2.20%, the
final-call non-`pMax` ratio is `1.01260`, and nonpositive-temperature calls rise
from 21 to 25. Thus `pMax` monotonicity is not a material causal driver of the
observed cross-channel recurrence drift on this checkpoint/trajectory, and its
removal is not a promotable correction. The released test object remains
physically absent; no sealed evaluation occurred.

The metric comparison is now closed against the public REALM implementation.
The released evaluator accumulates the five groupwise per-step MSE values over
all 49 transitions and does not divide by the horizon. D092-R1's comparable free
validation value is therefore `88.1382141`, not the separately reported mean
`1.7987391`. The REALM v2 PlanarDet table reports FFNO validation `12.577`, so
this PCNO checkpoint is 7.01x worse on the same nominal validation horizon and
released-code aggregation despite having 20.63% more parameters. The paper's
reported correlations are test-set quantities; D092's validation correlation
`0.59354` is not a direct ranking statistic. The truth-input sum `6.09925` is an
oracle-input diagnostic and is likewise not an autoregressive model comparison.

The registered recurrence-localization attempt G0 failed before any learned
call or output creation because its launcher passed serialized JSON where the
manifest parser required a mapping. That failure is preserved as infrastructure
provenance. The corrected G0b attempt uses the same D092-R1 checkpoint and open
validation trajectory. Each arm scores the raw learned proposal, then replaces
only one selected group with exact next-frame truth for the next recurrent input.
All replay, isolation, finiteness, persistence, and metric-identity gates pass.
The primary untouched non-`pMax` error ratios are chemistry `0.48403`, density
`0.34435`, temperature `1.13895`, and velocity `1.06437`; the frozen bands label
the first two materially helpful, temperature materially harmful, and velocity
small/inconclusive. Chemistry and density feedback improve every raw-scored
group. Temperature feedback improves its own raw next-temperature error
(`0.55372` ratio) while worsening every untouched group, which is evidence of a
compensating or co-adapted multi-field rollout rather than evidence that true
temperature is physically harmful.

G1 executes the separately selected one-shot directionality study without
changing the raw proposal at the pulse call. One exact chemistry or density
group is used only in the next recurrent input at calls 4, 12, or 32; all six
arms then return to raw recurrence and pass replay, isolation, H49 finiteness,
persistence, and metric-identity gates. The preregistered common-downstream
`T+u` post-pulse ratios for chemistry pulses are `0.96534/0.92635/0.83864` and
for density pulses `0.98336/0.90248/0.86301`. Directional partner ratios are
chemistry-to-density `0.95009/0.85390/0.67713` versus density-to-chemistry
`0.99150/0.94230/0.93567`. The frozen rule therefore finds no material partner
reduction at call 4 and chemistry-to-density-only material reduction at calls
12 and 32. The response is transient in both directions immediately after a
pulse; the asymmetry is in persistence, not the existence of a one-step effect.

The strongest bounded interpretation is now an asymmetric, state-dependent
recurrent persistence signature hidden by the aggregate score. It is not yet an
intrinsic PCNO bottleneck or a learned physical causal graph. The selected-group
baseline error also grows sharply from call 4 to 32 (chemistry NPE
`0.0107 -> 0.2621`; density `0.0800 -> 0.6246`), so phase and intervention dose
remain confounded. Accuracy gains do not repair physical validity: chemistry
call 32 reduces all-group mean error to `0.85884` of baseline but lowers
admissible calls from 7 to 4, while density call 12 leaves only 46/49 bounded
calls. Do not invent a method from these oracle hybrid states.

D093 subsequently closes the single-seed architecture/exposure matrix on the
same open validation trajectory. Each seven-condition cell has a lower selected
truth-input H49 sum than its three-condition counterpart. Only FFNO's
seven-condition cell also has a lower selected free H49 sum. FFNO-7 is best in
the matrix at `1.36466/32.53910`; all six free rollouts remain incompletely
admissible, and three training cells are incomplete after later decoded-
validation failures. The bounded [D093 record](D093_W26_L4_PLANARDET_SCALING_RECORD.md)
contains the exact table and manifests. The next decision is human-owned:
close this line, or separately choose a paper-faithful direct-state FFNO
control, free-rollout-aware selection study, or seed/condition replication.
None is implied or authorized by this record, and the test object remains
absent.

The D088 IgnitHIT direct-baseline attempt and its registered diagnostics are now
terminal. The paired fresh-map audit shows that the exact step-100 checkpoint
reduces the five-case, one-call normalized error by 60.38% relative to step 50
while creating 327 new `H2O` inverse-domain violations; matching truth and step
50 create none. This is validation/model-selection evidence that a sparse
physical-decoding tail can worsen while the global squared-error channel
improves. It does not identify a particular training cause, complete a direct
baseline, open test, or compare direct and residual models. Re-entry starts
with a separately authorized domain-compatible direct-baseline contract, not
an automatic resume or an inference-only clipping repair.

That re-entry is terminal under `D089`. Its one authorized fresh seed-0 A3-P0
run retained eligible step-1 and step-50 rows, then stopped at the step-100
eligibility gate before serializing the row. Best remains step 1
(`realm_npe_mean=4.5477657318`) and last remains step 50 (`4.9248557091`).

The separately authorized `D090` replay now resolves the missing row without
changing those identities. At step 100 all normalized and decoded proposals are
finite and all 145 case-calls are admissible, but every validation case contains
unbounded calls. The maximum registered-envelope ratio is `2.2733771801` and
H29 `realm_npe_mean=5.7428269386`. From step 50 to 100 the recorded
sampled-pair one-step train loss falls 81.60% while H29 NPE rises 16.61% and the
envelope ratio grows 10.09x. The training rows use different registered samples,
so this motivates rather than proves an objective/exposure-mismatch hypothesis.
It does establish a boundedness-versus-admissibility separation under the exact
history, not an offending-channel, fresh-versus-propagated, optimizer,
output-link, or general instability cause.
No resume, retry, selection, repair, sealed test access, residual comparison,
or PlanarDet run occurred under D090. Full D089 continuation and its residual
arm remain terminal; they are not resumed or relabeled as the new PD0 line.

`D091` A1 is now complete without new scientific model output. It freezes the
minimum matched step-50/step-100 teacher-forced/free-recurrence diagnostic,
including per-channel first events, spatial maxima, decoded-finiteness versus
normalized-recurrence semantics, exact checkpoint/runtime/source gates, and
truth-input-reset claim boundaries. Synthetic and maintained REALM CPU tests
pass. Its one exact A2 inference execution remains separately unauthorized;
therefore offending-channel and fresh-versus-propagated attribution remain
missing evidence, not findings.

D072, D084, D087--D090, and D092 are complete and terminal at their registered
scopes. D093 is closed and terminal as partial single-seed,
single-open-validation evidence. None establishes boundary-condition
improvement, resolution consistency, geometry/rotation generalization,
conservation, operator convergence, or general Euler stability. Re-entry
requires a new owner-selected claim and preregistration.

The separately registered H320 bump study is also closed at its single-seed,
30-open-case, reference-free scope. D094 B1-A is locally retained and rehashed.
Its preregistered B1-B audit evaluated the four already-selected `n=256`
checkpoints on the other 28 open-validation trajectories and selected the
stretched schedule: paired geometric-mean all-call error is `0.0640001` versus
`0.0967684` for prefix-tail, with zero hard failures in all four cells. The
stretched/prefix all-call ratios are `0.4843` for PCNO and `0.9032` for PCFNO.
This is one-seed schedule--architecture interaction and schedule-selection
evidence, not a completed scaling curve, seed-robust architecture ranking, or
conservation result. The authorized fresh paired seed-0
`n={8,16,32,64,128}` stretched-schedule ladder is now running; its B1-A
`n=256` endpoints will be reused. The 20 historical test trajectories remain
sealed.

W26-L5 A46-A1 closes synthetic plumbing only. A46-A2 remains fail-closed because
no compatible independent physical-node-type checkpoint or new-case manifest
is bound. Its A46-A2-R1 resource-build continuation stopped twice on broad
isolated-source imports and once on interpreter environment discovery, all
before data, model construction, training, generation, or outcome opening.
Those receipts are infrastructure provenance, not same-state branch evidence;
A43/A44-R1 remain the bounded scientific result.

Historical exact-source replay is additionally limited by nine source-bound
preregistration documents that are absent from the checkout, Git history, and
retained archives, including the A46 preregistration. Retained artifacts remain
bounded outcome evidence, but these old identities must not be reconstructed;
future reruns require new preregistrations and source manifests. The exact gap
table is maintained in [HANDOFF.md](HANDOFF.md#unrecoverable-preregistration-gaps).

Native-resolution residual correction is one component of W26-L5, not the sole
practical queue or standing default. Data assimilation remains reserved pending
open-loop diagnosis and explicit human review. Mixed-resolution training is
likewise not implied by current evidence.

## Recovery And Retained Records

- [2026-08-11 decision archive](history/RESEARCH_DIRECTION_DECISION_through_2026-08-11.md):
  exact pre-compaction decision text and terminal narratives.
- [2026-08-11 evidence archive](history/MECHANISTIC_DIAGNOSTIC_TRACKER_through_2026-08-11.md):
  exact detailed contracts, results, hashes, and run narratives through D086.
- [Earlier evidence ledger](history/MECHANISTIC_DIAGNOSTIC_TRACKER_through_2026-08-01.md):
  byte-preserved state through D063.
- [Compact diagnostic tracker](MECHANISTIC_DIAGNOSTIC_TRACKER.md): current
  run-ID and topic routing.
- [D093 PlanarDet scaling record](D093_W26_L4_PLANARDET_SCALING_RECORD.md):
  closed single-seed/open-validation results and provenance.
- [H320 reference-free rollout record](W26_L1_H320_REFERENCE_FREE_ROLLOUT_RECORD.md):
  closed 30-case descriptive recurrence and exact packet hashes.
- [D094 bump-scaling preregistration](D094_BUMP_SCALING_PREREGISTRATION.md):
  locally retained B1-A/B1-B schedule audit, exact routed-ladder provenance,
  and the running fresh seed-0 `n={8,16,32,64,128}` paired sweep.
- [Hash-bound W26-L5 derivation source](../../DERIVATION_PACKAGE.md): retained
  byte-for-byte at its artifact-bound root path.
- [HANDOFF.md](HANDOFF.md): current operational recovery snapshot.
- [SECTION_1_2_CORRECTED_BASELINES.md](SECTION_1_2_CORRECTED_BASELINES.md):
  corrected 1D labels and frozen results.
- [CPG_EULER_DATASET_CONTRACT.md](CPG_EULER_DATASET_CONTRACT.md),
  [BUMP_300_DATASET_AUDIT.md](BUMP_300_DATASET_AUDIT.md), and
  [CPGGNSPDES_REFERENCE_AUDIT.md](CPGGNSPDES_REFERENCE_AUDIT.md): live schema
  and frozen dataset/reference provenance.
