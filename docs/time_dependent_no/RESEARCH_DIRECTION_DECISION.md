# Time-Dependent Neural Operators: Current Research State

Updated: 2026-08-03

Status: current human-review snapshot; evidence and planning, not executable
authorization

## Authority And Change Control

Current explicit human direction takes precedence over this repository snapshot.
This file records the last reconciled scientific state, claim boundaries, and
standing owner constraints. It does not grant or deny permission for a future
experiment by itself.

The other documentation roles are:

- [HANDOFF.md](HANDOFF.md): replaceable operational snapshot;
- [MECHANISTIC_DIAGNOSTIC_TRACKER.md](MECHANISTIC_DIAGNOSTIC_TRACKER.md):
  compact experiment index;
- [historical tracker](history/MECHANISTIC_DIAGNOSTIC_TRACKER_through_2026-08-01.md):
  exact historical contracts, results, hashes, and dated decisions; and
- [README.md](README.md): onboarding, data pointers, and active-code inventory.

Historical gates close their exact registered attempts. They do not create a
permanent prohibition on a method family. A Codex-proposed next step, stop rule,
or sequencing rule is nonbinding unless the human owner explicitly adopts it.
Newer scoped owner direction supersedes older direction without requiring a
documentation edit first; the documents should then be reconciled afterward.

Private paths, hosts, credentials, and machine-specific dataset locations
remain in ignored local context.

## Current Verdict By Evidence Family

The overall result-to-claim verdict is partial, with high confidence. The
campaign supports useful fixed-family baselines and a bounded mechanistic
account. It does not establish a generally improved, shock-stable,
geometry-aware neural-operator solver.

| Program | Current supported scope | Binding nonclaim |
| --- | --- | --- |
| Line 1: 1D flow maps | Larger learned macro steps can win after fewer recurrent compositions; the preferred stride depends on horizon and metric. | No universal stride, learned CFL limit, timestep transfer, native-solver resolution transfer, or mesh-invariant solver follows. |
| Line 2: CPGNet | Corrected 1D controls support message reach rather than width alone; causal boundary training improves the local 2D release-bundle result without closing the oracle gap. | No paper-table reproduction, exact DG replay, physical interface trace, conservation, seed robustness, or architecture-transfer claim follows. |
| Line 3 bump | D041 remains the exact historical comparator. B1 is a replacement miss; attached-K2 improves B1 recurrence, and K2D0 decomposes most of that endpoint difference into a smaller propagated term on an already-better incoming-error path. | No exact same-contract attached-K2 comparison with D041, attached-gradient causality, more-contractive map, fresh-ripple cure, physical conservation, or cross-family transfer is established. |
| Line 3 dynamic FV | D044 is a useful one-seed baseline. D060 reduces H60 state error but fails the joint front/high-pass promotion field. D063 gives bounded zero-shot coarse/fine transfer for one unchanged D060 checkpoint under one common-source contract. D064--D067 show persistent large-scale mesh drift plus high-rank local structure that mostly cancels or corrects during recurrence. D070C and D073-A narrow the cross-grid pathway to differential geometry and physical support. D071 additionally shows that persistent low-rank correction improves every tested native `250x100` H30 endpoint, while its local filter is harmful there. Owner-prioritized, registered D074 makes native correction and a transfer-native comparator the primary practical track. D068--D069 close the node-type mechanism question. | No resolution invariance, proof that PCNO is an operator, unique native fixed-hop cause, independently confirmed correction, general node-type necessity, optimal boundary encoding, boundary-condition improvement, native-solver equivalence, other-PDE transfer, or sealed performance follows. |
| Line 4 latent forecast | Discontinuous decoder regularity improves front fidelity, but the tested latent family remains decoder-capacity limited. | No latent transition, autonomous recurrence, geometry transfer, or data-assimilation result exists. |

The supersonic-bump and Mach-1.1 dynamic finite-volume programs remain separate.
In particular, D053b's mostly fresh smooth-high-pass defect is dynamic-family
evidence; K2D0's bump high-pass improvement is mainly propagated-path evidence.

## Current Line 3 Evidence Amendments

### A-01. Bump serious-training and K2 evidence

Verified evidence:

- B1 completed 34 of 40 full-coverage passes, with 725,220 transition
  presentations and 183,600 optimizer steps.
- No B1 checkpoint passed its historical parity conjunction, so B1 supplies no
  selected replacement for exact D041.
- The retained B1 epoch-22 checkpoint improves normal-node one-step error but
  has a much larger causal-boundary stratum than raw D041.
- The attached-K2 continuation improves B1 FP32 H20/H40/H60/H79 state error by
  7.42%/13.70%/21.33%/27.23% on the same 30 validation trajectories.
- K2D0 closes the exact additive decomposition for both checkpoints. At H79,
  normal-node total and propagated error improve by about 30%, fresh defect
  improves only 2.43%, and finite-amplitude propagation gain is essentially
  unchanged.
- The two projected-teacher K2T0 records stopped before training. They are
  harness/preflight failures and supply no method result.

Claim implication: the attached-K2 continuation follows a better compounded
error path on this parent checkpoint and validation cohort. The bundled evidence
does not identify which training change caused that difference, establish an
exact same-contract comparison with D041, show a more contractive map, or cure
the one-step ripple source.

### A-02. D063 dynamic resolution evidence

Verified evidence:

- One unchanged D060 checkpoint completes all 24 already-open validation cases
  at 125x50, 250x100, and 500x200, using restrictions of case-matched 1000x400
  evolutions.
- Mean H60 errors are 0.0149776/0.00748436/0.0161704; the training resolution
  remains a sharp optimum.
- Final state-scaled exact-input prediction commutators are
  0.00157577/0.00137169; free-rollout commutators rise to
  0.0121705/0.0129033 and are numerically close to accumulated
  input-trajectory gaps.
- For aligned exact inputs, the prediction-state and predicted-update
  commutators derive from the same underlying physical difference field because
  the restricted base states cancel, up to recorded restriction/rounding
  tolerance. Their historical scalar numerators are not identical:
  prediction-state metrics use `state_scale`, whereas update metrics use
  `residual_scale`. The larger 0.206303/0.175978 ratios therefore combine a much
  smaller one-step-update denominator with different component weighting. D063
  does not isolate those effects or determine how signed discrepancies
  accumulate during rollout.
- Off-grid boundary, front, and physical-wavelength-band errors regress.
- Replacing the frozen checkpoint's node types is harmful, so the checkpoint
  uses those channels. The separately trained all-normal row changes source and
  loss population and is capability evidence rather than a causal ablation.
- The confirmation is pilot-informed and uses FP32; it is not a bitwise replay
  of the historical BF16 D060 selection.

Claim implication: a single resolution-independent parameter vector provides
useful bounded zero-shot transfer and approximate exact-state commutation under
this frozen common-source contract. This is not a general operator,
resolution-invariance, or node-type-necessity result.

### A-03. D064--D067 predicted-increment structure

Verified contract:

- D067 binds the unchanged D060 checkpoint
  `95e6c180a3298c4b38662d8c6cb77301c0e7fd0286e9a53571bddc487b8379c9`
  and normalizer
  `9df7efb2f1aadf315f399dcabf3b366bf1b2f46794b026cbc5b7d80ba22e1a1a`
  on six already-open validation cases, both adjacent grid pairs, 30 raw
  recurrent calls, physical `dt=0.02`, FP32 without AMP, and the frozen
  `model_all_nodes` boundary policy.
- The unified payload stores the free rollout, mesh/state pathway terms,
  common-source references, restriction, and residual scale from the same
  execution. Reference, increment, cumulative, and relative replay errors are
  zero at stored float64 precision. The maximum total mesh/state
  reconstruction residual is `1.35e-17` residual-scale units; band-energy and
  region/pathway closures are below `3e-14`.
- The paired-input increment/output identity, free recurrence
  `e_(n+1)=e_n+delta_n`, and signed-growth identity all pass. Nodes are never
  pooled across meshes before per-case aggregation.

Verified evidence:

| Pair | Large path -> endpoint share | Local path -> endpoint share | Local coherence | State share of path | Mesh share of endpoint |
| --- | ---: | ---: | ---: | ---: | ---: |
| 125x50 -> 250x100 | 50.26% -> 88.85% | 21.29% -> 3.20% | 0.0928 | 73.35% | 81.15% |
| 250x100 -> 500x200 | 14.55% -> 64.86% | 75.86% -> 22.11% | 0.0596 | 69.52% | 92.09% |

- The small one-step-update denominator explains why a modest absolute
  physical defect can have a large update-relative ratio, but it does not
  explain the accumulation pattern. At the high pair, the local-band aggregate
  signed interaction is `-0.9067` of local path energy and its lag-1
  correlation is `-0.3473`; the large band has positive interaction and
  dominates the endpoint.
- Residual-scaled POD/SVD supports a low-rank-drift/broadband-local split. At
  250x100 -> 500x200, the first three modes contain 72.9% of large-band energy
  and 40.0% of local energy; 8 versus 20 modes are required for 95%.
- State response dominates path energy but is negative-growth/corrective on
  97.78% of high-pair steps. The surviving endpoint is mesh-attribution
  dominated. Symmetric attribution splits the mesh/state cross term equally,
  so fractions above 100% and negative fractions denote cancellation rather
  than probabilities.
- The shock region occupies about 4.32% of physical volume but contains 32.85%
  of high-pair total path energy, a 7.60x enrichment. The boundary occupies
  14.32% and contains 14.61%, about neutral enrichment; this result does not
  select a boundary-condition change.
- The high-pair endpoint shock difference is not primarily translation:
  translation alone explains 4.79%, while translation+dilation+amplitude
  explain 24.46%. Roughly 75.54% remains profile/structure error. The
  restricted 500x200 rollout is worse than the 250x100 rollout against the
  reference in shock position, total variation, and thickness.
- Density and energy dominate persistent endpoint energy. Local momentum
  defects, especially transverse momentum, largely disappear by the endpoint.

Evidence limits:

- D065's pointwise/differential/spectral pathway probe remains composite or
  unresolved; D067 does not isolate the responsible PCNO branch or kernel.
- At the 2026-08-03 reconciliation, all 125 output hashes registered by the
  accepted D064--D067 summaries and final visual manifest verify. D067's
  diagnostic runner and metric utility also match their registered hashes, but
  the current dirty worktree differs from five of eight inherited evaluator
  source hashes. The stored same-run payload remains interpretable; the current
  checkout is not an exact replay surface until the manifest-bound source bytes
  are retained together.
- A first cross-process 500x200 replay developed recurrent elementwise drift
  from a tiny numerical difference and remains `failed_closure`. It is not
  accepted as a standalone scientific result. Its aggregate structure agrees
  with D067 within the predeclared repeatability gates, but a second fully
  accepted unified replicate is still absent.
- The six-case cohort is open validation evidence. Strength-OOD and test remain
  sealed, and no mixed-resolution training, correction, or assimilation result
  follows.

Claim implication: the large update-relative discrepancy is not a uniform
`1.3*dU`-style bias and is not made harmless by normalization alone. Under this
frozen contract it is a composite of a persistent, low-rank, large-physical-
scale mesh defect and a high-rank, shock/vortex-local recurrent component that
mostly cancels or corrects. This supports a two-channel correction or
data-assimilation hypothesis, but it does not yet identify the architecture
change or establish resolution-consistent operator learning.

### A-04. D068--D069 node-type mechanism closeout

Verified contract and signal path:

- The zero-based PCNO input layout is coordinates `0:2`, quadrature density
  `2`, normalized conservative state `3:7`, family-local one-hot node type
  `7:11`, and normalized Mach `11`.
- Dynamic-FV meanings are `0=interior`, `1=y-symmetry contact`,
  `2=x-extrapolation contact`, and `3=contact with both`. Bump meanings remain
  separate: `0=normal`, `1=wall`, `2=outflow`, and `3=inflow`.
- With `Delta` defined as correct-type minus type-`k`-replaced-by-normal, the
  post-lift difference at an affected node is exactly
  `Delta z_i = W_type (e_k-e_0)`. Node types enter the learned model through
  this initial lift. Pointwise paths initially preserve locality, graph
  differential paths spread through the bound neighborhood, and integral
  paths can spread globally; their outputs then mix in every PCNO block and the
  decoder. Downstream branch and hidden-state differences are propagated
  intervention responses, not causal branch shares.
- Non-mutating instrumentation records post-lift state, all four PCNO blocks,
  pointwise/integral/differential branch outputs, and decoder state. Hooked and
  unhooked inference passed the predeclared equivalence checks: dynamic
  tolerance `2e-5` absolute and `1e-7` relative with observed maxima at most
  about `1.31e-5` and `5e-8`; bump tolerance `2e-3` absolute and `1e-5`
  relative with observed maxima at most about `5.94e-4` and `8.5e-7`. JVP and
  activation norms remain diagnostics, not causal proof.

Frozen-checkpoint evidence:

| Dynamic grid | Replaced-node fraction | Teacher H30 correct / all-normal | Free H30 correct / all-normal |
| --- | ---: | ---: | ---: |
| 125x50 | 5.536% | 0.002263 / 0.004052 | 0.015773 / 0.036416 |
| 250x100 | 2.784% | 0.001532 / 0.004039 | 0.008058 / 0.036419 |
| 500x200 | 1.396% | 0.002370 / 0.004416 | 0.016811 / 0.036954 |

- These are means over three open-validation physical cases. Every dynamic arm
  completes 30 calls and remains admissible. Per-affected-node lift changes do
  not shrink with grid refinement, while the domain-average post-lift relative
  difference falls `0.1285 -> 0.0905 -> 0.0638` with the boundary-node
  fraction. Decoder-level relative differences remain `0.237/0.259/0.233`.
  This distinguishes input dilution from downstream propagation; it is not
  resolution invariance or PDE resolution transfer.
- Dynamic physical-frequency summaries show a larger relative all-normal
  effect in the `0.125--0.25` physical-frequency band than in the
  `0.05--0.125` band by call 30.
  Single-type replacement is population-ordered: y-symmetry is largest,
  x-extrapolation is intermediate, and the combined type occurs only at four
  corners. This does not establish an intrinsic per-node importance ordering.
- On four native open-validation bump geometries, wall-to-normal tracks the
  all-normal intervention. Three of four all-normal and wall arms become
  inadmissible at calls 12 or 13; correct, inflow-to-normal, and
  outflow-to-normal complete H20 on all four. Mean teacher H20 error is
  `0.005809` correct, `0.011458` all-normal, `0.011437` wall-to-normal,
  `0.005837` inflow-to-normal, and `0.005810` outflow-to-normal. Wall dominance
  reflects label population, location, and wall/shock interaction rather than
  lift-vector magnitude alone. Because the evidence uses native graphs, it is
  distinguishable from query-mesh resampling; it still does not prove general
  geometry dependence or boundary-condition improvement.
- The registered visual bundle contains 14 MP4s and all 317 comparable rollout
  frames: 30/30 for every dynamic rollout and every admissible bump frame up to
  the first invalid proposal. Invalid bump states are not fabricated. Views
  include baseline, intervention, difference, residual, boundary distance,
  post-lift sensitivity, and selected hidden-feature sensitivity.

Controlled-training evidence:

| Dynamic-FV arm, three matched seeds | Mean H30 error | Sample SD | Within-seed wins |
| --- | ---: | ---: | ---: |
| No type channels | 0.008108 | 0.001322 | 2/3 |
| Four constant-zero channels, ordinary initialization | 0.008272 | 0.000064 | 1/3 |
| Four constant-zero channels, exact-function-matched lift | 0.009332 | 0.001001 | 0/3 |

All nine runs use the same within-seed data order, optimizer, 50-epoch budget,
51,200 presentations, 13,400 optimizer steps, and open-validation evaluation
population. All selected checkpoints pass completion and admissibility gates,
and every exact-function-matched initialization passes the registered identity
check. The claim that four permanently zero channels improve training is not
supported. Ordinary-init variation is compatible with fan-in,
initialization/parameterization, and downstream RNG effects; the experiment
does not isolate scalar fan-in alone. The short-horizon mean and variance
differences are hypothesis-generating only, with `n=3` and no independent
selection/evaluation population.

Result-to-claim decision:

- **Supported, bounded:** under the frozen source, normalizer, geometry,
  recurrence, and physical boundary policy, replacing family-local node types
  causes checkpoint-output changes. Bump wall semantics are the decisive
  category for the four tested native geometries.
- **Not supported:** node types are universally necessary for PCNO, the present
  encoding is optimal, boundary conditions are improved, bump generalizes
  across geometry because of the labels, or constant-zero channels improve
  training.
- **Closure:** no additional node-type experiment is required for these bounded
  claims. Re-entry requires a separately authorized question that actually
  needs a new physical descriptor, an independent population, or a changed
  model family. Rotation, new encodings, and boundary-policy changes are not
  prerequisites for this closeout.

## Compact Unified Claim Register

The exact RC-01--RC-15 wording through 2026-07-23 is preserved in the historical
decision archive. This table is the current compact rendering; later evidence
amendments above do not silently create a new claim ID.

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

These constraints persist because they protect data, claim validity, or
reproducibility. They may be changed by later explicit owner direction.

1. Strength-OOD and test populations remain sealed until the owner explicitly
   opens a named evaluation.
2. Private paths, remote-machine details, credentials, datasets, checkpoints,
   and large generated outputs remain outside committed documentation.
3. Bump, dynamic FV, CPG, and latent populations are not pooled. A mechanism
   established on one family is not transferred to another without evidence.
4. Bump equal-node or reconstructed weights are diagnostic proxies. Physical
   conservation language requires validated control volumes, faces, normals,
   orientation, and boundary accounting.
5. Validation results do not become test or state-of-the-art claims.
6. Exact historical attempts retain their checkpoint, source, data, optimizer,
   recurrence, evaluator, population, metric, threshold, cost, and stop-rule
   identities. A changed attempt receives a new identity rather than rewriting
   the old outcome.
7. One-step fit is an entry diagnostic, not a rollout proxy. Report
   admissibility, state error, front position/strength/thickness, smooth
   high-pass, and boundary behavior together.
8. High-frequency energy is a band diagnostic, not synonymous with ripple or
   instability. Any filtering claim requires anti-smearing front controls.
9. A truth-free continuation is qualitative finiteness/admissibility evidence,
   not longer-horizon accuracy.
10. Failed harnesses are provenance rather than negative method evidence.
11. Boundary-representation comparisons freeze separately (i) the physical
    boundary-condition family and (ii) the explicitly bound checkpoint/evaluator
    recurrence and boundary-handling policy. Every representation arm uses the
    same pair, and the encoding line may neither select nor modify them. D060
    uses `model_all_nodes`; a bump comparison declares its policy before
    evaluation. Changing input descriptors does not authorize boundary
    clamping, splicing, closure corrections, or boundary-objective changes
    owned by the separate boundary-condition workstream.
12. Animations and spatial visualizations are required evidence for the active
    resolution and boundary lines, not optional illustrations. They complement
    rather than replace per-case quantitative metrics and must bind case,
    resolution, physical time, recurrence, intervention, units, color scale,
    and any visualization-only subsampling. Compared state/update panels use
    common per-field scales fixed without model-result-dependent autoscaling.

## Shared Reporting Semantics

- Hk means k recurrent calls joining k+1 saved states.
- Completion at Hk means valid_length equals k.
- Here residual/update means the denormalized one-call conservative-state
  increment `F_h(U) = N_h(U) - U`, not a PDE equation residual. Commutators and
  state/update visual fields are formed after denormalization. Use audited
  physical-volume metrics where they exist and label any bump proxy explicitly.
- Mean survival is mean(valid_length / k) and need not equal completion
  fraction.
- Error curves contain the accepted prefix; a failure marker identifies the
  first excluded inadmissible proposal.
- Every animation records trajectory ID, physical horizon, learned stride,
  encoded states, and visualization-only subsampling.
- Checkpoint selection, precision, seed, sample presentations, optimizer
  history, recurrence, evaluator, and source/artifact digests are part of the
  reported result.
- A sharp physical front legitimately contains high-frequency content; lower
  high-pass energy is not an improvement when caused by shock blurring or lost
  wave strength.

## Current Owner-Selected Research Program

An explicit owner update on 2026-08-03 authorized D072. D072 asks whether
fixed-physical-width semantic boundary fields give recurrent PCNOs
resolution-consistent access to boundary meaning. D070--D071 subsequently
completed under their own frozen contracts and neither promoted; they do not
pause or broaden D072.
The completed D068--D069 node-type diagnosis supplies motivation and controls;
it is not being reopened or reinterpreted as a new-encoding result.

The dominant D072 contribution target is representation consistency for
long-horizon time-dependent operators. Geometry variation and rigid rotation
are controlled stress tests of the representation, not a claim to solve broad
geometric generalization. This keeps the project anchored to autonomous Euler
rollouts while making the geometric input contract mathematically defensible.

### Native correction and resolution generalization

This line owns the cross-resolution target, restriction, commutator, recurrence,
and aggregation contract. D064--D067 close its first frozen-checkpoint question:
the large update-relative discrepancy combines a demanding one-step
normalization, persistent low-rank large-scale mesh drift, and high-rank
shock/vortex-local structure that mostly cancels or corrects during recurrence.
The common-source reference floor, ordinary boundary localization, and a pure
shock-translation account do not explain the result.

D070C narrows the causal resolution question: Fourier quadrature, subcell,
synthesis, and pointwise controls remain small, while the differential branch
and its fixed-hop composite are supported. D071's frozen low-rank plus local
correction does not promote on either benchmark. D073-A then fixes the layer-3
graph-ball radius at the training-grid physical width; all four dynamic-FV
same-hidden mechanism and native-relevance strata pass, including both
contraction on `125x50` and expansion on `500x200`. This makes differential
physical support the leading diagnosed mesh pathway, without uniquely
attributing the native repeated two-hop kernel.

The later explicit owner direction changes the practical priority. D074 first
tests systematic correction at the checkpoint's native `250x100` dynamic grid,
starting with the D071-supported persistent low-rank channel and an exact zero
control. D071's current local dissipation is not carried forward: at native
resolution it worsens both aggregate residual error and local-band error. Bump
is evaluated separately on native graphs as a portability/safety control and
may legitimately select zero.

The operational cross-resolution comparator is now frozen as query-grid cell
averages transferred to `250x100`, raw or corrected native rollout, and transfer
back to the query grid. Exact block averaging and piecewise-constant injection
are used for the nested dynamic grids, with transfer floors and initial-
information loss reported explicitly. A direct D073-style off-grid rollout is
a practical method only if it beats the best transfer-native arm under the same
query-grid truth, horizon, structure controls, and compute accounting. If it
does not, D073 remains valuable causal/theoretical evidence but is not the
deployment path. Data assimilation remains reserved; mixed-resolution training
remains conditional on human review and is not implied by this diagnosis.

### Completed node-type mechanism diagnosis

D068--D069 close the currently authorized boundary-representation question.
The frozen interventions establish actual checkpoint use of family-local type
semantics, with wall labels decisive for the tested bump geometries. The
controlled training matrix rejects a general benefit from appending four
permanently zero channels. The physical boundary-condition family and evaluator
boundary policy remained frozen throughout; no boundary-condition improvement,
new encoding, rotation, or general geometry claim was tested.

The reusable intervention, instrumentation, equivalence-test, and all-frame
visualization surfaces remain maintained for audit or a future separately
authorized question. Dynamic-FV and bump meanings must remain separate, and
bump node dropping remains query-mesh resampling unless connectivity,
quadrature, tags, targets, and provenance establish a genuine physical
resolution contract. Nothing in this closeout selects a new boundary
descriptor or blocks the active resolution-pathway work.

The later explicit D072 authorization now selects one minimal descriptor for a
new comparison. For each semantic boundary subset Gamma_k, the stored channel
is the bounded volume field

    B_k^ell(x) = rho(distance(x, Gamma_k) / ell),

where ell is fixed in physical coordinate units and
rho(r) = 1 - 3 r^2 + 2 r^3 for 0 <= r < 1, with zero outside the collar. This
is not a diffuse surface measure and receives no 1/ell amplitude scaling.
Dynamic FV factors the four exclusive contact codes into overlapping
y-symmetry and x-extrapolation fields. Bump retains separate wall, outflow, and
inflow fields; finite-graph mixed endpoint edges belong to both adjoining
semantic subsets rather than using wall precedence. The bump boundary polyline
remains a mesh-derived geometry proxy, so node dropping is still not a
physical resolution experiment.

The first training ladder is deliberately limited to no boundary field, the
union geometry collar, and separate semantic collars. All field models omit
categorical type channels and start from an exact mathematically matched
no-boundary initialization: active lift columns and every non-lifting
parameter are copied, inserted field columns are zero, and post-build RNG state
matches the no-boundary arm. The exact lift mechanism is
Delta h_lift(x) = W_B B(x); later blockwise differences and rollout effects are
causal only under frozen field interventions, while JVPs remain sensitivity
diagnostics. No learned extender, surface-delta channel, normal/tangent field,
new physical boundary policy, or broad geometry architecture enters the first
ladder.

### Shared sequence and visual evidence

The completed D068--D069 boundary diagnosis is now reconciled with the
resolution program. D072 is the separately authorized boundary-descriptor
question; it does not authorize a boundary-policy change or opening a sealed
population. Any method beyond its minimal ladder, or any mixed-resolution
checkpoint outside its preregistration, still requires human review.

Every decisive evaluation predeclares the quantitative figures and animations
applicable to its family and stage. In the resolution stage, teacher-forced mesh
discrepancies and free-rollout discrepancies are animated separately.
Cross-resolution fields are compared on one declared physical mesh through the
conservative restriction, never image resizing. One synchronized, fixed-scale
layout shows the coarse true increment, coarse predicted increment,
restricted-fine true increment, restricted-fine predicted increment, both model
errors, and their cross-model discrepancy. In D068, correct-type versus
all-normal animations are labeled frozen-checkpoint OOD interventions showing
channel use, not causal training ablations.

For a coarse comparison mesh, define the teacher-forced discrepancy
`delta_n^TF = F_c(U_c^n) - R F_f(U_f^n)` with `U_c^n = R U_f^n`. For free
rollouts, define `e_n = Uhat_c^n - R Uhat_f^n` and
`delta_n^FR = F_c(Uhat_c^n) - R F_f(Uhat_f^n)`. Initial predictions satisfy
`Uhat_c^0 = R Uhat_f^0` up to a recorded numerical tolerance. The resolution
line requires a four-panel temporal diagnostic: the teacher-forced physical
difference field and its state- and update-relative ratios under both the
historical scales and one common declared component scale; the free-rollout
cumulative-change curve `||e_n|| / ||U_c^n-U_c^0||` for `n >= 1`, together with
the unnormalized state gap, reference discretization floor, and endpoint errors;
signed growth
`||e_(n+1)||^2 - ||e_n||^2 = 2<e_n, delta_n^FR> + ||delta_n^FR||^2`; and
cumulative coherence
`||sum_(j<n) delta_j^FR|| / sum_(j<n) ||delta_j^FR||`. A recurrence-closure
preflight checks `e_(n+1) - e_n - delta_n^FR`. The signed local growth map uses
the declared state-channel scaling and volume-weighted contributions
`w_i (2 <e_(n,i), delta_(n,i)^FR> + ||delta_(n,i)^FR||^2)`, whose sum must recover
the global growth numerator.

The D068 views include fixed-distance boundary overlays, nonperturbing post-lift
and per-block representation differences, and bump shock evolution. All
comparable recurrent frames are present; bump sequences stop at the first
inadmissible proposal. Representation plots remain descriptive unless paired
with a controlled intervention, and Jacobian/JVP norms remain diagnostics.
State/update panels identify physical versus normalized units and use common
per-field scales fixed before inspecting model outcomes. Every comparison
annotates trajectory, mesh, physical time, horizon, recurrence, intervention,
admissibility, and visualization-only subsampling.

Other historical questions remain recoverable from the archived decision
surface. They are deferred rather than silently rejected.

## Retained Records

- [MECHANISTIC_DIAGNOSTIC_TRACKER.md](MECHANISTIC_DIAGNOSTIC_TRACKER.md):
  compact run-ID and topic index.
- [Historical evidence ledger](history/MECHANISTIC_DIAGNOSTIC_TRACKER_through_2026-08-01.md):
  byte-preserved detailed evidence through D063.
- [Historical decision surface](history/RESEARCH_DIRECTION_DECISION_through_2026-08-01.md):
  byte-preserved prior decision and claim language.
- [SECTION_1_2_CORRECTED_BASELINES.md](SECTION_1_2_CORRECTED_BASELINES.md):
  corrected 1D labels and frozen results.
- [CPG_EULER_DATASET_CONTRACT.md](CPG_EULER_DATASET_CONTRACT.md),
  [BUMP_300_DATASET_AUDIT.md](BUMP_300_DATASET_AUDIT.md), and
  [CPGGNSPDES_REFERENCE_AUDIT.md](CPGGNSPDES_REFERENCE_AUDIT.md):
  live schema and frozen dataset/reference provenance.
