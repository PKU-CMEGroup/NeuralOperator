# Time-Dependent Neural Operators: Handoff

Updated: 2026-08-11

This is a replaceable operational snapshot. It should describe the current
workspace, the latest supported conclusions, and the next human decisions
without repeating historical run narratives.

## Read Order

1. [RESEARCH_DIRECTION_DECISION.md](RESEARCH_DIRECTION_DECISION.md) for the
   current scientific state and standing owner constraints.
2. This handoff for the current workspace and active sequencing.
3. [MECHANISTIC_DIAGNOSTIC_TRACKER.md](MECHANISTIC_DIAGNOSTIC_TRACKER.md) for
   compact experiment-ID routing.
4. [WEEKLY_RESEARCH_PLAN.md](WEEKLY_RESEARCH_PLAN.md) for the current
   owner-selected research lines, dependencies, and prospective gates.
5. [CODEX_KICKSTART_PROMPTS.md](CODEX_KICKSTART_PROMPTS.md) for read-only agent
   kickoff instructions after the plan is reviewed.
6. [README.md](README.md) for navigation and maintained code.
7. [BOUNDARY_FIELD_DERIVATION_PACKAGE.md](BOUNDARY_FIELD_DERIVATION_PACKAGE.md)
   and [BOUNDARY_FIELD_PRIOR_ART_AUDIT.md](BOUNDARY_FIELD_PRIOR_ART_AUDIT.md)
   for D072's mathematical and public-prior-art boundaries.
8. One bounded section of the
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
| Line 3 bump | D041 remains the historical comparator. B1 is a replacement miss; attached-K2 improves B1 recurrence, but no exact same-contract D041 comparison is bound and it is not promoted as a replacement. D072 G1/S1 improve neither N0 completion nor common-case H79. D084 closes finite-inadmissibility continuation without a causal blow-up claim or D082 promotion. D083 is negative proxy-mass query-graph evidence plus a transported-Fourier rotation control. D085 is terminal negative fixed-world-Fourier transformed-orientation evidence: all 60 rotated proposals fail at call 1 before recurrence, with a material raw same-input defect. |
| Line 3 dynamic | D044 is the useful baseline; D060 improves state error without passing the front/high-pass conjunction; D063 gives bounded common-source transfer; D064--D067 identify persistent large-scale mesh drift plus locally cancelling/corrective structure; D070C and D073-A narrow the leading mesh pathway to layer-3 differential geometry and physical support; D068--D069 close the node-type mechanism question; D072 proves one-seed FP32 continuous-field use but establishes no promotable three-seed BF16 N0-relative gain and does not clear structure gates; D074-A and D075 show consistent native H30 endpoint benefit from low-rank correction, but neither raw scaling nor integral neutralization satisfies the frozen efficacy-and-safety contract. |
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
subsequent explicit owner direction authorized D072 as a separate semantic
boundary-field question, now closed below; it does not retroactively change
D068--D069 or authorize changed boundary policies or sealed-population
evaluation.

### D072 semantic boundary-field closeout

The first N0/G1/S1 ladder is complete and not promoted. Dynamic FV remained
`model_all_nodes`; bump remained `causal_nodal_physical`. Across three BF16
seeds, dynamic H30 is `0.007602/0.007951/0.007522` for N0/G1/S1, so neither
field arm establishes a promotable N0-relative gain: S1's 1.05% mean reduction
misses the registered `0.90` gate. Bump H79 completion is
`0.95556/0.94444/0.86667`, and common-case H79 ratios are
G1/N0 `1.10787` and S1/N0 `1.12396`.

One seed-20260718 FP32 dynamic evaluator proves checkpoint use of the fields:
zeroing all fields raises G1/S1 H30 from `0.008253/0.008582` to
`0.033767/0.033867`. The effect spreads beyond the collar, but the field arms
do not clear N0-relative front/high-pass controls, S1 is 3.98% worse than G1 in
the same FP32 endpoint evaluation, and BF16/FP32 rankings differ. Do not combine
the three-seed BF16 scalar contrast and one-seed FP32 structure rows into a
single promotion gate.

The bound artifact root is
`artifacts/time_dependent_no/d072_boundary_fields_20260803a`. Bump retrieval
verifies all 18 checkpoints and 330 declared arrays. The compact dynamic
retrieval omits its 18 training checkpoint payloads; its bound analysis checked
them remotely, and the FP32 evaluator rechecked the exact seed-20260718
checkpoints. The dynamic package verifies 25 visual manifests and 62 outputs,
including 12 fixed-scale 31-frame MP4s. At D072 closeout no bump checkpoint
intervention/structure/animation package existed. D084 later supplies that
separate evidence for D082, not for the D072 bump arms. No D072 resolution,
geometry-transform, or rotation result exists.

Supported scope is exact descriptor/matched-initialization construction and
causal continuous-field use on one dynamic FP32 seed under the frozen physical
policy. No benchmark-wide gain, causal use by the D072 bump arms, optimal
encoding, boundary-condition improvement, conservation,
resolution/geometry/rotation generalization, or operator-convergence claim
follows. D084's later D082 evidence does not retroactively promote D072.
Re-entry requires a new owner-selected claim and same-precision multi-seed
evidence; resolution and rotation require separate contracts.

### D084 finite-inadmissibility and bump-field closeout

D084 is terminal on the same 30 open bump validation cases under exact D082/N0
checkpoint, normalizer, source, split, and `causal_nodal_physical` policy
bindings. All 360 six-variant/two-repeat H79 continuations remain finite and
avoid every registered global explosion event. Ninety-eight rows become
inadmissible, always first through nonpositive internal energy in the semantic
collar. Eight model and 20 output-policy calls exactly repair an invalid
input/proposal, but every naturally invalid row is invalid again at call 79.
Inadmissibility is therefore a local-growth warning here, not a sufficient cause
or synonym for blow-up.

The D082 frozen field interventions expose a tradeoff. Zeroing inflow raises
strict completion from 19/30 to 23/30 at nearly unchanged error but retains
large local excursions; zeroing all also reaches 23/30 while raising error about
23%; zeroing wall leaves completion at 19/30 and raises error about 23%. D082
versus N0 is only a trained-model comparison. D082 is not promoted.

The canonical case-23/54/128 publication bundle has six 80-frame MP4s at
1430-by-638 and 5 fps, full-node smooth interpolation, fixed reference-only
scales, no mesh/nodal/boundary overlay, and maximum saturation 0.0607%.
Its manifest hash is `e6296fcc...fd344`. D084 leaves no active queue. Re-entry
requires explicit owner authorization and first a deterministic FP32 audit
before any matched minimal-admissibility-repair counterfactual.

All 125 output hashes registered across the accepted D064--D067 summaries and
final visual manifest verify at this handoff. The current dirty checkout is not
an exact evaluator replay surface: five of D067's eight inherited evaluator
source hashes have drifted, although its diagnostic runner and metric utility
still match. The exact source retrieval request is recorded in the compact
tracker and must be satisfied before claiming a byte-identical rerun.

## Current Workspace Task

The 2026-08-11 owner-selected weekly program has five coordinated lines:
long-horizon stability; shock representation and the differential pathway;
boundary conditions and finite propagation; REALM benchmark and paper
validation; and cross-resolution correction. The live claim map, experiment
ladders, gates, dependencies, and weekly run order are in
[WEEKLY_RESEARCH_PLAN.md](WEEKLY_RESEARCH_PLAN.md); ready-to-paste A0-only agent
instructions are in [CODEX_KICKSTART_PROMPTS.md](CODEX_KICKSTART_PROMPTS.md).
The present planning request does not authorize implementation, checkpoint or
dataset execution, training, remote work, downloads, or sealed-population
access.

Cleanup gate: `utility/time_dependent_no/__init__.py` retains one legacy
`fixture` label because that file is bound by the v4/v5 continuation hashes.
Remove the label only as part of an explicitly versioned v6 source-set change.

The categorical node-type line is complete through D068--D069, and the
separate D072 N0/G1/S1 continuous-field ladder is also complete and not
promoted. D084 separately closes the D082 bump collar and
finite-inadmissibility question without method promotion or an active follow-up.
D070--D071 completed under their own frozen contracts and neither correction
promoted. Under the later explicit owner direction, D073-A also
completed: fixed training-grid physical radius in the layer-3 graph-ball
family passes every same-hidden dynamic-FV mechanism and native-relevance
stratum. Native-grid residual correction is retained within W26-L5 rather than
serving as the sole practical track.
D073-B is secondary until a direct off-grid rollout can be compared against
transfer to `250x100`, native rollout, and transfer back. Data assimilation
remains reserved and is not an active experiment.

D083's exact 30-case H79 package is now locally retrieved and verified. All
240 attempts terminate, but only 61 reach H79. G1a retains about 8.22% of
native nodes and strongly degrades state and residual behavior; it is
query-graph resampling, not PDE resolution transfer. G1b directly used rotated
coordinates, state, graph, and differential geometry but also transported the
Fourier modes and phase origin, so it is only an analytic covariance negative
control. The owner-corrected D085 contract keeps the checkpoint's original
`[6,2]` Fourier lattice fixed on rotated coordinates, permits inverse rotation
only in postprocessing, and requires a fresh case-128 H2 before any H79 launch.
That H2 and the terminal H79 package pass all provenance, direct-path, replay,
Fourier, geometry, uniqueness, and algebra gates. All 30 cases fail the first
finite rotated proposal in both type arms, so no accepted rotated rollout frame
exists. At call 1 the correct-type median same-input defect is `2.3226`
residual units with raw RMS `2.4057`, versus `1.5134` and `1.7846` for D083's
transported-basis control under the identical raw denominator. Fixed-world
Fourier representation materially worsens the defect, but D083's remaining
large error shows that it is not the sole coordinate-dependent pathway. The
failure is architecturally unsurprising: the checkpoint has absolute coordinate
features, fixed world-axis Fourier phases, componentwise state normalization,
and unconstrained learned x/y differential-channel mixing, but no enforced
90-degree covariance. Thus D085 stresses the complete frozen representation on
an unseen orientation of retained cases; it is not a new-case experiment or a
Fourier-only attribution. D086 now closes the selected-case visualization gap
for cases `172/58/187`: both type arms and two ordinary-CUDA repeats are
inadmissible from call 1 but remain finite through call 79. Correct-type
repeat-0 median state error grows from `0.172` at call 1 to `2.756` at call 10
and `7.83e5` at call 79; the raw increment-error RMS grows from `2.258` to
`2.35e6`, so denominator shrinkage is not the main failure. Six verified
shared-reference-scale movies show localized truth updates versus broad
interior/lower-domain predicted updates in all conservative components. The
two repeats agree initially but diverge strongly in the invalid regime. D086
is a failure-morphology continuation, not accepted-rollout, rotated-PDE,
geometry-generalization, stability, conservation, or deterministic-late-field
evidence.

D074's original-source and post-validator dynamic H2 smokes are complete. The
post-validator 93-file source identity is `42210950...4b07` at base
`b219394`; all three loaded cases reproduce the canonical 25,000-node,
99,300-directed-edge geometry and exact model tensors. H2 remains explicitly
non-scientific. Run `d074_dynamic_h30_r1` is an infrastructure-only failure;
the summary-only retention repair passes 24 focused and 569 full CPU tests under
immutable source `892745de...86e8`, and the four H2-run equivalence audit is
`b1dda067...0bcb`. Replacement `d074_dynamic_h30_r2` completes in 448.66 s
after preflight `de190303...fa1d`; its summary is `bb7b1c3...57ab`, and all
23 declared hashes verify. The frozen selector retains `zero` and promotion
fails. All 288 nonzero candidate/case pairs improve endpoint error, but every
candidate fails at least one complete-case no-harm control. The closest arm,
`rank8_gain0p25`, has median endpoint/residual ratios `0.95338/0.99177` and
passes 323/324 rows; its sole miss is `sv_e10_y00` energy-integral RMS at
`1.09823`. The six held-out evaluation cases consequently reproduce the raw
baseline and do not supply nonzero-arm efficacy evidence. Bump remains closed
until its D041 replay population, limits, and hashes are exact.

D075, the dynamic-only integral-neutral native-correction follow-up, is now
complete. It retains the D074 checkpoint, population, rank-8 fit, recurrence,
physical boundary policy, selector separation, and all no-harm thresholds. The
26-output H30 run selects `rank8_gain0p125_raw`, not a projected arm. All six
endpoint and residual ratios improve, with medians `0.974925/0.995263`, but the
registered endpoint target is missed and two strong-energy cases fail controls
at final energy-integral ratios `1.51847/1.69402`; promotion is false. Energy-
only neutralization is nearly tied with raw at gain `0.125` and reduces the
worst gain-`0.5` control from `3.07718` to `1.20361`, but remains unsafe. All-
integral projection removes useful density/transverse-momentum constant modes
and is already ineligible at gain `0.125`. The applied correction is exactly
case-independent across the two visual payloads, so their divergent energy
response is recurrent state sensitivity. Next work should predeclare a state-
aware gain/short-window response constraint and keep local shock/vortex
dissipation as a separate capped channel; data assimilation remains reserved.

D076 is complete and not qualified. The first H2 attempt is a preserved
pre-rollout parser failure; repaired H2 r2 passes every non-scientific contract
gate. Under the exact D075 checkpoint, normalizer, open split, native `250x100`
geometry, raw recurrence, type-0 correction support, and `model_all_nodes`
policy, H30 selects positive gains on all 18 calibration cases and reaches
median endpoint/residual ratios `0.93272/0.98622`. Sixteen choices are safe.
Both highest-strength `e10` choices fail one real global-energy control:
`y00` gain `0.25` has energy-integral RMS `1.09825`, and `y08` gain `1.0`
has final absolute energy-integral ratio `2.00237`. The run therefore stops at
`calibration_not_qualified` before any conditional evaluation target loads.

The result and analysis summaries are under
`artifacts/time_dependent_no/pcno_response_gain_controller_d076_20260805b/`
with SHA-256 `baaa4c16...53e7` and `0e797316...96bba`. Exact case-LOO replay,
all declared result hashes, the reference floor, and the source contract pass.
The H5 response changes by less than `0.04%` at the `e09 -> e10` transition
while the worst H30 control grows by factors `1.084/1.970`; endpoint prediction
is useful, but H5 response is not a late energy-safety certificate. A post-hoc
strength-grouped fold replay changes only `e10_y00/y08` to zero and passes the
old numeric gates (`18/18` safe, 16 nonzero, endpoint/residual medians
`0.94914/0.99587`), but it cannot retroactively qualify D076 or authorize
target loading. Bump, sealed populations, local shock filtering, and data
assimilation remain outside this result.

D077 is registered as the prospective leakage-free test. For each physical
strength, both `y00/y08` cases are excluded from the rank-8 correction fit and
from response-policy statistics/neighbors. The unchanged target-free upper-
amplitude support rule must therefore abstain on held-out `e10`; no late-energy
threshold, distance cutoff, gain relaxation, or scientific gate changes. The
exact contract and H2-before-H30 order are in the tracker. The 95-file isolated
source passes 57 tests. H2 passes its non-scientific contract with summary
SHA-256 `b5f26464...3cc2`, exact phase order, zero fold/neighbor leakage, and
no sealed access. H30 is terminal `failed_contract` with summary SHA-256
`1406ff87...81b7`; all 40 declared outputs verify. Grouped calibration passes
(`18/18` safe, 16 nonzero, endpoint/residual medians
`0.94867/0.99590`) and the evaluation numbers satisfy every promotion
inequality (`0.93440/0.99482` selected/zero medians, worst control `1.03810`,
four nonzero). These are diagnostic only. All six independent H5-prefix/H30
replays miss the unchanged `2e-5/1e-7` tolerances, reaching
`5.507e-5/5.340e-7`, including both zero-correction `e11` cases. Every other
closure is near machine precision. Do not promote or reinterpret D077.

The structure is nevertheless clear enough to preserve as diagnostic evidence:
on the four nonzero cases, endpoint and final cumulative-defect medians fall to
`0.91115/0.92044` while residual RMS falls only to `0.97552`. The selected
correction removes `25.8--60.6%` of rank-8 parallel energy and exactly preserves
orthogonal energy, matching the persistent-drift/local-noise decomposition.
D078 is terminal `complete` under the deterministic-runtime-only replay, with
every D077 scientific gate and tolerance unchanged. Its first source attempt
stopped before data/CUDA and has no scientific result. Corrected immutable
source B passes H2-r2 and H30-r1; H30 summary SHA-256 is `959d3398...e86e`, all
40 declared outputs verify, and all six prefix discrepancies are exactly zero.
Calibration qualifies and promotion passes at `0.93441/0.99483` median
endpoint/residual ratios with worst control `1.03811` and four nonzero cases.
On those cases, endpoint/cumulative medians are `0.91116/0.92045`; rank-8
parallel energy falls to `0.394--0.742` while orthogonal energy is unchanged.
This is accepted adaptive dynamic-FV open-validation evidence, not a D077 pass
or independent confirmation. The scientific 12-output figure and 27-output
fixed-scale response-animation bundles are hash-exact, unwatermarked, and have
zero saturation. Bump remains closed on the missing immutable D041 replay
binding.

D079 is terminal `complete` as the required deterministic process confirmation.
Its fresh process and clean output path recompute D078 from scratch. Summary
SHA-256 is `f11c5882...6aa7`; all 40 within-run hashes pass and every declared
scientific payload, including the three visual payloads, is byte-identical to
D078. The summaries differ only in output path and elapsed time (`2363.85 s`
versus `2365.73 s`); all scientific fields and all six zero-prefix rows are
exact. Choices, promotion numbers, low-rank energy removal, and local
vortex/high-pass worsenings are consequently exact. The machine-readable audit
SHA-256 is `12a82ffb...2b9f`. This confirms deterministic repeatability on the
same open population only, not independent-data or statistical robustness.

D080 is now terminal `complete` on the same dynamic-FV open population. D080-A
finds small frozen-direction headroom, but the exact recurrent D080-B H30 pilot
promotes no local arm. Shock-isotropic/normal corrections reduce aggregate
residual RMS by about `0.17%`, reach their best median state ratio near call 4,
and help through roughly calls 22/23 before reversing. They worsen the shock
endpoint in all six cases while improving smooth high-pass error in six/five,
and slightly increase temporal coherence and first-three POD concentration.
The vortex arm is placement-dependent: all `y08` target regions improve and all
`y00` target regions worsen; its worst global-integral control is `1.08594`.
All result hashes, 48 H5 prefixes, recurrence/growth/mean/support closures, and
the 40-output fixed-scale analysis package verify. This favors causal
state/time-conditioned activation or an early-window stop over a stronger
always-on filter. It is dynamic-only adaptive open-validation evidence; bump,
sealed populations, and data assimilation remain outside the run.

D072's implementation, open-only manifests, all 18 training summaries, and the
dynamic FP32 intervention/visual package are now reviewed and closed above.
The primitive boundary-to-domain extension and bounded mask are prior art; the
tested `ell=0.05` first ladder does not promote even under the narrower
PCNO-pathway framing. There is no live D072 queue and no pending artifact whose
mere retrieval would change the decision.

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
  and dynamic D072 use `model_all_nodes`; bump D072 uses
  `causal_nodal_physical`.
- Visualizations bind trajectory, resolution, physical time, horizon,
  recurrence, intervention, units, common per-field scale fixed without
  result-dependent autoscaling, and visualization-only subsampling; they do not
  replace quantitative metrics.

## Active Sequencing And Decision Gates

The live cross-line sequencing is the milestone order in
[WEEKLY_RESEARCH_PLAN.md](WEEKLY_RESEARCH_PLAN.md). The numbered items below
remain binding evidence-specific gates and nonclaims; they are not, by
themselves, the current weekly queue.

1. Treat the D067 unified payload and scale/profile analysis as the accepted
   frozen-checkpoint mechanism closeout; do not promote the failed-closure pilot.
2. Treat D068--D069 as the terminal categorical node-type mechanism record and
   D072 as the terminal, not-promoted first continuous-field ladder. Preserve
   their separate claims and frozen physical boundary policies.
3. Treat D084 as the terminal checkpoint-bound D082 bump mechanism and
   finite-inadmissibility record. Do not turn its field-zeroing tradeoff into a
   general stabilizer, causal blow-up result, or D072/D082 promotion.
4. Re-enter D072 or D084 only after an explicit new owner-selected claim and
   preregistration. D084 repair work first requires deterministic FP32;
   common-source resolution, native-graph geometry transforms, and rotation
   each require their own contract.
5. Keep completed D070--D071 as bounded negative/mechanistic evidence; do not
   reopen them as part of a boundary-representation closeout.
6. D074-A replacement H30 r2 completes under the unchanged
   `model_all_nodes`/raw-recurrence/FP32 policy. All result and visualization
   hashes, arithmetic closures, and fixed-scale saturation checks pass. Zero is
   the frozen selection: the tested low-rank direction consistently improves
   endpoints, but no arm satisfies all complete-case controls. Preserve the
   `rank8_gain0p25` one-row energy near miss as mechanism evidence; do not relax
   the `1.05` gate after inspection. Bump stays closed until the D041 replay
   cases, limits, and artifact hashes are frozen.
   Preserve the historical D071 evaluator and never pool family coefficients
   or node meanings.
7. D075's float64 type-0-supported integral projection, per-call normalized
   integral closure at most `1e-12`, idempotence, and policy-bound selector rows
   pass focused qualification. The H30 result and all 26 hashes now pass audit,
   but `rank8_gain0p125_raw` misses the endpoint target and two energy controls.
   Do not promote static raw or integral-neutral correction. Preserve the
   useful coherent-channel reduction and the high-gain energy-neutral safety
   improvement as mechanism evidence. Cross-source D074/D075 H30 replay is
   tolerance-level, not bitwise exact. The next native-correction contract must
   control state-dependent response rather than only the instantaneous
   correction integral; keep the local shock/vortex channel separate.
8. Preserve D076 as `calibration_not_qualified`; do not load its six conditional
   evaluation targets or promote the controller. The exact failure is two
   highest-strength energy controls, not endpoint or aggregate-residual harm.
   Implement the registered D077 contract without changing D076: grouped
   exclusion must cover both correction fitting and response-policy fitting,
   and the existing target-free upper-strength support rule is the only new
   effective abstention. Run focused CPU/isolated-source checks and the six-case
   H2 smoke before any H30 launch. D077 is now terminal `failed_contract` even
   though its calibration and numerical promotion inequalities pass: every
   independent prefix replay misses the unchanged tolerance at a nearly
   uniform scale, including zero arms. Preserve all numerical outcomes as
   explicitly diagnostic only. D078 H2-r2 and H30-r1 pass under the registered
   deterministic runtime, including bit-identical replay for all six prefixes,
   all 40 H30 hashes, and the full D077 scientific conjunction. Preserve D077
   as failed and D078 as the new adaptive open-validation result. D079 now
   passes the registered fresh-process gate with exact equality for all 40
   declared D078 scientific payloads and all scientific summary fields.
   Preserve D078/D079's low-rank endpoint, cumulative-defect, and control gains;
   treat D079 as same-open-population process repeatability, not independent-data
   confirmation. D080 now closes the first separate zero-inclusive local
   shock/vortex intervention: no combined arm promotes. Preserve its genuine
   early shock benefit, late reversal, shock-versus-smooth split, and
   placement-dependent vortex signal as mechanism evidence. A next local step
   must predeclare state/time-conditioned or early-window activation and retain
   target-free construction plus complete-case regional and integral controls;
   do not infer that a larger fixed cap will help. D081 is now terminal. Its
   Source-A H2 stopped correctly on an overbroad coefficient-digest guard;
   Source B narrowed only that guard, passed H2, and completed the exact-bound
   H30 matrix with all 41 output hashes and every parent, prefix, inventory,
   closure, support, and cap gate passing. Calls 1--20 beat always-on, calls
   11--30, and the dose-matched control on the registered timing comparisons,
   but improve the persistent endpoint in only `3/6` cases. Early-versus-
   persistent median endpoint/residual/cumulative/shock ratios are
   `0.999971/0.993392/0.999677/0.997485`; no arm promotes. Preserve the timing
   and `y00`/`y08` placement split as mechanism evidence, not a safe or
   materially effective correction.
9. Freeze the dynamic practical comparator as transfer to `250x100`, native raw
   rollout, and transfer back; omit corrected transfer because D074-A selected
   zero. A direct D073-style rollout must
   beat the best transfer-native arm on the query grid to retain practical
   priority; otherwise preserve it as mechanism/theory evidence. Do not reuse
   the currently harmful D071 local filter without a new zero-inclusive signed
   no-harm selector.
10. Log low-rank-coefficient and localized-innovation assimilation ideas, but do
   not implement them in this stage.
11. Consider mixed-resolution training only after the causal and correction
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
