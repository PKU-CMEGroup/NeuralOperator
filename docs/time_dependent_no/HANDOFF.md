# Time-Dependent Neural Operators: Handoff

Updated: 2026-08-03

This is a replaceable operational snapshot. It should describe the current
workspace, the latest supported conclusions, and the next human decisions
without repeating historical run narratives.

## Read Order

1. [RESEARCH_DIRECTION_DECISION.md](RESEARCH_DIRECTION_DECISION.md) for the
   current scientific state and standing owner constraints.
2. This handoff for the current workspace and active sequencing.
3. [MECHANISTIC_DIAGNOSTIC_TRACKER.md](MECHANISTIC_DIAGNOSTIC_TRACKER.md) for
   compact experiment-ID routing.
4. [README.md](README.md) for navigation and maintained code.
5. [BOUNDARY_FIELD_DERIVATION_PACKAGE.md](BOUNDARY_FIELD_DERIVATION_PACKAGE.md)
   and [BOUNDARY_FIELD_PRIOR_ART_AUDIT.md](BOUNDARY_FIELD_PRIOR_ART_AUDIT.md)
   for D072's mathematical and public-prior-art boundaries.
6. One bounded section of the
   [historical evidence ledger](history/MECHANISTIC_DIAGNOSTIC_TRACKER_through_2026-08-01.md)
   only when exact provenance or prior evidence is needed.

Current explicit human direction outranks this snapshot. Historical Codex
recommendations are evidence or advice, not permanent research policy.

## Current Scientific State

The branch supports a partial, high-confidence mechanistic contribution:

- a useful fixed-family 1D residual-FNO program;
- corrected evidence that CPGNet's 1D gain depends materially on message reach;
- a useful one-seed dynamic-FV residual-PCNO baseline;
- a strong but not D041-replacing bump training lineage; and
- separated evidence for propagated state error, fresh smooth high-pass defect,
  front error, boundary effects, admissibility loss, resolution transfer, and
  the scale/coherence structure of cross-resolution predicted-increment defects.

It does not support a generally improved shock-stable geometry-aware neural
operator, a learned conservative finite-volume solver, or a state-of-the-art
claim.

| Program | Operational evidence status |
| --- | --- |
| Line 1 | Completed evidence program; larger-step advantage is horizon- and metric-dependent. |
| Line 2 | Completed bounded CPGNet program; paper parity and exact DG replay remain unresolved. |
| Line 3 bump | D041 remains the historical comparator. B1 is a replacement miss; attached-K2 improves B1 recurrence, but no exact same-contract D041 comparison is bound and it is not promoted as a replacement. |
| Line 3 dynamic | D044 is the useful baseline; D060 improves state error without passing the front/high-pass conjunction; D063 gives bounded common-source transfer; D064--D067 identify persistent large-scale mesh drift plus locally cancelling/corrective structure; D070C and D073-A narrow the leading mesh pathway to layer-3 differential geometry and physical support, while D068--D069 close the node-type mechanism question. |
| Line 4 | Stopped before a viable latent forecast; decoder capacity remains insufficient. |

The bump and dynamic-FV conclusions remain family-specific. Bump node weights
are proxies; physical conservation diagnostics apply only to the audited
dynamic finite-volume contract.

## Latest Line 3 Results

### Bump serious-training lineage

- B1 completed 34/40 passes and retained epoch 22, but no checkpoint passed the
  historical parity conjunction against exact D041.
- Boundary-objective rows BG0 and RB0 did not improve the long-horizon
  state/front conjunction. RA0P failed its BF16 repeatability gate, so RA0 was
  never launched.
- On all 30 validation trajectories, minimum-change D041 projection `P_B^*`
  improves H20 state error and H79 completion from 28/30 to 30/30, but fails the
  1.05 shock-thickness no-harm envelope. The tested boundary-only splice does not
  explain the main interior gain.
- Attached-K2 improves B1 H79 state error by 27.23% on its 30 validation
  trajectories.
- K2D0 shows that the attached checkpoint's normal-node H79 state gain on the
  shared 30-trajectory validation cohort lies overwhelmingly in the propagated
  term. Fresh normal-state defect and finite-amplitude propagation gain change
  little; the bump smooth-high-pass decomposition is mixed.
- Both projected-teacher K2T0 attempts are zero-step implementation-provenance
  records, not negative method results.

### Dynamic resolution evidence

D063 evaluates one unchanged D060 vector on 125x50, 250x100, and 500x200
restrictions of common 1000x400 evolutions:

- all 24 open validation cases complete and remain admissible on every grid;
- mean H60 state error is 0.014978/0.007484/0.016170;
- state-scaled exact-input prediction commutators remain small but nonzero;
- exact-input update commutators are 0.206303/0.175978 relative to the
  fine-grid update. For aligned inputs, both derive from the same underlying
  difference field up to restriction/rounding tolerance, but their historical
  metrics use `state_scale` and `residual_scale`, respectively. At D063, the
  effects of component weighting, the smaller update denominator, and signed
  accumulation remained unresolved;
- the training grid is the clear state/front/boundary optimum; and
- node-type interventions show channel use, while the all-normal-trained row is
  confounded by source and loss-population changes.

This is bounded zero-shot transfer, not resolution invariance or proof that
PCNO is an operator. The final-call free commutators being close to their
incoming trajectory gaps does not establish harmlessness or error cancellation.

### D064--D067 residual-structure closeout

D064--D067 close the first frozen-checkpoint accumulation diagnosis under one
unchanged D060 vector and normalizer, six preregistered open-validation cases,
the adjacent 125x50/250x100 and 250x100/500x200 pairs, 30 raw recurrent calls,
`dt=0.02`, `model_all_nodes`, and conservative common-source restriction. No
sealed population is opened.

- The paired-input increment/output identity, free recurrence
  `e_(n+1)=e_n+delta_n`, signed-growth identity, and
  `delta_free=delta_mesh+delta_state` all pass. The accepted D067 payload has
  zero stored float64 replay error; maximum pathway and partition closure is
  `2.842e-14`.
- The small `O(dt)` increment denominator materially enlarges instantaneous
  ratios, but it does not explain the endpoint structure. Large physical scales
  rise from 50.26% to 88.85% of path-to-endpoint energy on the lower pair and
  from 14.55% to 64.86% on the upper pair. Local scales fall from 21.29% to
  3.20% and from 75.86% to 22.11%, with local temporal coherence
  `kappa=0.0928/0.0596`.
- The state-response pathway carries 73.35%/69.52% of path energy, but the
  fresh mesh pathway carries 81.15%/92.09% of endpoint energy. On the upper
  pair the state response is corrective on 97.78% of steps. This is structured
  recurrence, not a uniform `1.3*dU` amplitude bias.
- The defect is spatially concentrated but not boundary-dominated. The shock
  band occupies 4.32% of physical volume and carries 32.85% of upper-pair path
  energy (7.60x enrichment); the fixed boundary band occupies 14.32% and carries
  14.61%. Translation, dilation, and amplitude modes explain only 24.46% of the
  upper-pair endpoint defect, leaving most of it as profile/structure error.
- POD confirms a low-rank large-scale drift and a higher-rank local field: the
  first three upper-pair modes explain 72.9% of large-scale energy but 40.0% of
  local energy, and 95% requires 8 versus 20 modes.

The bounded mechanism is therefore persistent large-scale mesh inconsistency
coexisting with broadband shock/vortex-local noise that often cancels or
corrects during recurrence. The exact responsible pointwise, differential, or
spectral branch remains composite/unresolved. The failed-closure pilot is
retained as provenance, not a second scientific replicate, and no correction,
assimilation, training, or operator-invariance claim follows yet.

### D068--D069 node-type mechanism closeout

D068 verifies the 12-channel PCNO layout and the exact lift identity
`Delta z_i = W_type(e_k-e_0)` for correct-minus-normal replacement. Types enter
through the lift; non-mutating probes then trace their propagated effects
through pointwise, integral, differential, four-block, and decoder states.
Hooked and unhooked inference pass the family-specific equivalence tolerances.

- Across three dynamic open-validation cases, all-normal H30 free error is
  `0.036416/0.036419/0.036954` at 125x50/250x100/500x200, versus
  `0.015773/0.008058/0.016811` with correct types. The affected-node fraction
  halves with each refinement, explaining lift-level dilution but not the
  persistent decoder effect.
- Across four native bump geometries, wall-to-normal reproduces the all-normal
  damage: three arms fail admissibility at calls 12 or 13. Inflow-to-normal and
  outflow-to-normal complete H20 and remain close to correct. This is checkpoint
  use of bump wall semantics, not general geometry or boundary-condition
  improvement.
- All 14 animations contain every comparable frame: 180 dynamic frames and 137
  admissible bump frames. Bump movies stop at the first invalid proposal.
- D069's three-seed controlled dynamic training matrix gives mean H30 error
  `0.008108` without type channels, `0.008272` with ordinary constant-zero
  channels, and `0.009332` with exact-function-matched constant-zero channels.
  The zero-channel improvement claim is not supported.

The current node-type line is safely closed as a bounded mechanism diagnosis.
Do not promote it to universal necessity or an optimal encoding claim. The
subsequent explicit owner direction authorizes D072 as a separate semantic
boundary-field question; it does not retroactively change D068--D069 or
authorize changed boundary policies or sealed-population evaluation.

All 125 output hashes registered across the accepted D064--D067 summaries and
final visual manifest verify at this handoff. The current dirty checkout is not
an exact evaluator replay surface: five of D067's eight inherited evaluator
source hashes have drifted, although its diagnostic runner and metric utility
still match. The exact source retrieval request is recorded in the compact
tracker and must be satisfied before claiming a byte-identical rerun.

## Current Workspace Task

The node-type mechanism line is complete through D068--D069. D072 separately
compares no boundary field, one union geometry collar, and separate semantic
collars under a fixed physical width and exact no-boundary matched
initialization. D070--D071 have completed under their separate frozen
contracts and neither correction promoted. Under the later explicit owner
direction, D073-A now also completed: fixed training-grid physical radius in
the layer-3 graph-ball family passes every same-hidden dynamic-FV mechanism and
native-relevance stratum. The latest owner direction makes D074 native-grid
residual correction the primary practical track. D073-B is secondary until a
direct off-grid rollout can be compared against transfer to `250x100`, native
rollout, and transfer back. Data assimilation remains reserved and is not an
active experiment.

D072's implementation, open-only collar manifests, and both three-arm CUDA
smokes pass. The final touched boundary/source/trainer surfaces pass 59 CPU
tests and the maintained runtime surface passes six more. All nine dynamic
training summaries now exist, but their run contracts and aggregate outcomes
remain unreviewed. On bump, seed-20260718 N0 completed and the preserved resume
queue is running the remaining eight arms after correcting only a wrapper
assertion about causal recurrence. The resumed G1 arm passed process, contract,
disk, and CUDA health checks. Measured N0 throughput gives a nominal queue
finish near 2026-08-05 17:00 CST and a conservative window through early
2026-08-06. Per the owner's instruction, do not poll the healthy bump queue
merely for status; wait for completion or an explicit status request.

The public-literature audit falsifies novelty of the primitive
boundary-to-domain extension and of bounded geometry masks. The defensible
D072 target is the narrower codimension-to-PCNO-pathway scaling diagnosis and
its mechanistic validation in autonomous Euler rollouts. Both families now use
the preregistered primary `ell=0.05`: dynamic from validated rectangle geometry
and bump from a training-only 270-geometry audit that opened no state/target
array and no validation geometry. The exact manifests, source snapshot,
smoke-stream hashes, run-contract hash, and launch evidence are in
`refine-logs/EXPERIMENT_TRACKER.md`. No core comparison outcome has been
accepted: dynamic training is complete but unreviewed, and bump is incomplete.

D064--D067 remain the quantitative and visual contract for resolution work:
teacher-forced and free-rollout fields are separate, common-source comparisons
use conservative restriction rather than image resizing, and state/update
scales are fixed before model outcomes. D068 separately preserves all-frame
correct-type/all-normal, residual, boundary-distance, and hidden-sensitivity
views as frozen-checkpoint channel-use evidence. Scalar metrics and
completion/admissibility remain mandatory alongside every visualization.

For recovery, all-normal means replacing each family-local category by code 0
and one-hot `[1, 0, 0, 0]`; zeroing all four channels is a different
intervention. Dynamic-FV and bump meanings must never be pooled. Bump node
dropping is query-mesh resampling unless connectivity, quadrature, boundary
tags, targets, and provenance establish a physical resolution contract. These
rules remain standing safeguards, not an active request for more node-type
experiments.

The reviewed code is preserved in dependency order by commits `136e288`,
`ef07ae3`, `d399828`, `2c3e7d0`, and `41b106f`; large generated artifacts remain
ignored and outside Git.
The D072 implementation adds one boundary-field utility, one focused test
file, two retained-shard derivation adapters, and one bump training-geometry
audit; it surgically extends the existing shard preparation, PCNO
input/runtime, resolution sample builder, trainer, and maintained documentation.

## Standing Boundaries

- Strength-OOD and test populations remain sealed until explicitly opened by
  the human owner.
- Private machine details and credentials stay in ignored local context.
- Large datasets, checkpoints, raw rollouts, and generated figures stay under
  ignored artifact storage.
- Bump and dynamic evidence are not pooled.
- Physical conservation language requires audited finite-volume geometry and
  boundary exchange.
- Historical run outcomes are immutable; a changed run receives a new identity.
- Boundary-representation comparisons freeze separately the physical
  boundary-condition family and the explicitly bound checkpoint/evaluator
  recurrence and boundary-handling policy; every arm uses the same pair. D060
  uses `model_all_nodes`, while bump declares its policy before comparison.
- Visualizations bind trajectory, resolution, physical time, horizon,
  recurrence, intervention, units, common per-field scale fixed without
  result-dependent autoscaling, and visualization-only subsampling; they do not
  replace quantitative metrics.

## Active Sequencing And Decision Gates

1. Treat the D067 unified payload and scale/profile analysis as the accepted
   frozen-checkpoint mechanism closeout; do not promote the failed-closure pilot.
2. Treat D068--D069 as the terminal categorical node-type mechanism record;
   D072 is a separately authorized continuous-field study.
3. Treat the completed dynamic D072 artifacts as unreviewed, and let the healthy
   unattended bump N0/G1/S1 queue finish without polling unless explicitly
   requested. Then verify every run contract and compare open populations only.
4. Require field covariance, no-hook/hooked equivalence, completion,
   admissibility, region/frequency metrics, and all-frame animations before
   interpreting D072. Keep dynamic resolution and bump geometry evidence
   separate.
5. Keep completed D070--D071 as bounded negative/mechanistic evidence; do not
   reopen them as part of D072.
6. The narrow D074-A evaluator is implemented, has passed Ruff/compile checks,
   44 focused and 120 broader CPU tests, and three independent reviews. Its
   92-file source manifest is frozen as `3fbd8059...5237`. The next authorized
   integration step is only the non-scientific dynamic H2 contract smoke on an
   explicitly selected resource. Dynamic H30 stays closed until regenerated
   `250x100` geometry/weights and selector repeatability are exact; bump stays
   closed until the D041 replay cases, limits, and artifact hashes are frozen.
   Preserve the historical D071 evaluator and never pool family coefficients
   or node meanings.
7. Freeze the dynamic practical comparator as transfer to `250x100`, native raw
   or corrected rollout, and transfer back. A direct D073-style rollout must
   beat the best transfer-native arm on the query grid to retain practical
   priority; otherwise preserve it as mechanism/theory evidence. Do not reuse
   the currently harmful D071 local filter without a new zero-inclusive signed
   no-harm selector.
8. Log low-rank-coefficient and localized-innovation assimilation ideas, but do
   not implement them in this stage.
9. Consider mixed-resolution training only after the causal and correction
   reviews, with its held-out grid declared in advance.

Projected-teacher K2, other front-aware methods, and longer-reference questions
remain deferred historical topics rather than rejected method families.

## Evidence And Recovery

- [Compact experiment index](MECHANISTIC_DIAGNOSTIC_TRACKER.md)
- [Current research state](RESEARCH_DIRECTION_DECISION.md)
- [Full historical evidence ledger](history/MECHANISTIC_DIAGNOSTIC_TRACKER_through_2026-08-01.md)
- [Full prior decision surface](history/RESEARCH_DIRECTION_DECISION_through_2026-08-01.md)
- [Corrected 1D baseline record](SECTION_1_2_CORRECTED_BASELINES.md)
- [Bump data audit](BUMP_300_DATASET_AUDIT.md)
- [CPG schema](CPG_EULER_DATASET_CONTRACT.md)
- [Public-reference audit](CPGGNSPDES_REFERENCE_AUDIT.md)

Commit ae1f402 preserves the D063 tracker addition. Commit e2070f6 preserves
the active documentation immediately before this compaction and the reusable
PCNO infrastructure cleanup. Retired one-off scripts remain recoverable from
the historical commits named by the compact index.
