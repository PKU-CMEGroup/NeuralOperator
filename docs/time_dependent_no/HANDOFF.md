# Handoff

Updated: 2026-07-30

This page is a concise operational snapshot derived from
[`RESEARCH_DIRECTION_DECISION.md`](RESEARCH_DIRECTION_DECISION.md). The decision
file is authoritative; exact run records and artifact hashes live in
[`MECHANISTIC_DIAGNOSTIC_TRACKER.md`](MECHANISTIC_DIAGNOSTIC_TRACKER.md). The
[README](README.md) is non-authoritative onboarding and the sole active-code
inventory.

## Current Verdict

The 2026-07-23 result-to-claim gate is `partial`, with high confidence. The
campaign supports a useful one-seed residual-PCNO baseline on one frozen dynamic
finite-volume family and a bounded mechanistic decomposition. It does not
support the intended claim of a generally improved, shock-stable,
geometry-aware neural-operator solver.

Following explicit mentor-directed human guidance, Line 3 was reopened at the
scientific-planning level on 2026-07-26. On 2026-07-27 the human owner changed
the next-stage premise: skip historical reproduction/attribution and use the
accumulated evidence for one serious bump baseline. L3R-R0 and L3R-A0 are
superseded as prerequisites, not failed. L3R-B0P has now failed its frozen tiny-
fit error gate, so conditional L3R-B0 was not launched. Later that day the owner
explicitly authorized the new L3R-B1P/B1 replacement contract and required
strict outcome parity with exact R/D041. B1P has now passed exact comparator
replay, all-300 boundary parity, local/AutoDL tests, and one full-coverage health
and throughput pass. B1 then completed 34/40 passes under its 23-hour guard,
with no parity-eligible checkpoint and therefore no D041 holdout or sealed
access. Its epoch-22 checkpoint improves exact-D041 normal-node one-step 21.79%
but misses all-node parity because the causal-boundary stratum is 10.48 times
the raw-D041 value. This sacrifices a causal explanation of the historical
checkpoint gain. Dynamic, smoothing, larger-backbone, longer-reference, and
sealed work remain unauthorized.

The owner has now also authorized L3R-BC0P and conditional paired L3R-BC0 on
zw-gpu: exact D041 raw continuation control versus a causal-boundary
continuation, without changing the live B1 process. The local implementation
and 65-test gate pass. The exact source, complete 300-shard tree, replay,
all-policy audit, and paired sample stream were then bound successfully. The
registered raw real-graph preflight reserved 14.493 GiB against the frozen
14 GiB ceiling, so BC0P failed/closed before causal timing and neither BC0 arm
was launched.

On 2026-07-28 the owner authorized and completed the separate L3R-BG0P/BG0
zw-gpu study. Both candidate gradients passed preflight and all three matched
rows completed. Relative to the selected causal normal-node-only control,
projected-target and three-hop band training increased BF16 H79 state error by
`3.70%/4.13%` and H20 state error by `5.04%/2.77%`. Both preserved 30/30
completion and slightly reduced boundary-node error, but worsened normal-node
and deployed near-boundary error. Both fail the frozen utility gate; neither is
selected and BG0 is closed.

L3R-RB0P passed and the one L3R-RB0 zw-gpu arm completed. Direct raw all-node
supervision reduced fixed-pair raw-boundary error `84.00%` and BF16 H20 state
error `10.97%`, but selected H79 state and normal-node errors worsened `9.30%`
and `10.50%`. Completion remained 30/30; FP32 diagnostics instead show higher
smooth high-pass and worse front-shape geometry. The exact objective is
rejected, RB0P/RB0 are closed, and no holdout or sealed population was opened.

L3R-RA0P subsequently bound its implementation and identities but missed the
frozen BF16 gradient-ratio repeatability tolerance (`0.0006482` relative versus
`0.0001`). RA0P is failed/closed; its smoke and the conditional RA0 arm were not
launched. No new, holdout, or sealed population was opened.

On 2026-07-29 the owner authorized one minimum-change hard-boundary audit on
zw-gpu using exact D041 and retained B1. D041 plus `P_B^*` improves all-30 H20
state error `2.61%`, completes `30/30` rather than `28/30` at H79, and wins
26/28 common-survivor H79 state comparisons with median ratio `0.9259`. It
worsens corrected boundary/reference error `30.27%` and H79 shock-thickness
log error to ratio `1.1028`, so it is the preferred validation state/completion
deployment protocol with an explicit anti-smearing caveat, not an unconditional
replacement. B1 plus `P_B^*` fails 5/5 sampled rollouts, supporting the need to
match training and recurrent closure. Two 256-presentation D041 continuation
pilots, dense deployed-degree supervision and a matched normal-only control,
both sharply regress H20. No serious checkpoint was launched and no test,
holdout, or sealed population was opened.

The boundary-information subline is now closed with conditional re-entry. Its
output-splice diagnostic shows that only `18.81%/17.20%` of the projected-
teacher H20 all/normal gain follows the boundary proposal, while the reverse
interior splice retains `136.82%/137.45%`; both fail shock-shape gates. The
owner subsequently authorized general-PCNO K2 attribution on 2026-07-30. The
full-scale attached-K2 continuation from exact B1 epoch 22 improves FP32 H79
validation state error `27.23%` with 30/30 completion and a broadly favorable
H79 front/high-pass hierarchy. L3R-K2D0 now closes every contract gate. At H79,
normal-state total and propagated error improve `29.78%/29.95%`, fresh defect
improves only `2.43%`, and finite-amplitude normal propagation gain is unchanged;
the map follows a better incoming-error path rather than a verified more-
contractive map. Smooth high-pass improves `6.91%` through its propagated term,
while fresh high-pass is unchanged. The principal gain is interior/front, not
boundary-local. L3R-K2T0 attempt 20260730a stopped in argument parsing before
training because the staged frozen trainer lacked the registered projected-
teacher option. It has zero sample presentations, no checkpoint, and no method
result. A human-reviewed isolated repair then passed 40/40 local focused tests,
but its 20260730b launcher duplicated a 62-character expected tracker hash,
omitting `ef`, and stopped during remote source preflight before staged tests or
training. Both records remain zero-step harness failures. No retry is authorized,
and no holdout or sealed population was opened.

## Four-Line Status

| Line | Operational status | Frozen conclusion |
| --- | --- | --- |
| 1: large-step flow maps | Closed | On the frozen 1D Euler study, direct larger-step maps trade harder one-call approximation against fewer recurrent compositions. The preferred stride changes with horizon and metric. This is not a universal stride, CFL, timestep-transfer, or resolution-transfer result. |
| 2: CPGNet validity and mechanism | Closed | Corrected 1D controls support message reach rather than width alone as the main gain. The 2D release-bundle legal-boundary run improves all four primitive variables on 19/20 trajectories, but remains roughly `1.6--2.4x` worse than the oracle-boundary row. Dataset/checkpoint/evaluator parity with the paper remains unresolved, and the result is one seed without a validation or grouped geometry holdout. |
| 3: geometry-aware 2D rollout | Boundary-information subline closed; K2D0 closed; K2T0 remains zero-step and no retry is authorized | D041 plus minimum-change projection retains its thickness caveat. Full-scale attached K2 improves B1 H79; K2D0 attributes the endpoint state gain primarily to a smaller propagated term on a better incoming-error path and finds no fresh smooth-high-pass cure. Both projected-teacher execution records are harness failures, so the matched control has no result and attached-gradient attribution remains open. |
| 4: latent forecasting and assimilation | Stopped before forecast training | Smooth-decoder and fixed-Haar capacity tests do not pass the reconstruction/front hierarchy, even though discontinuous regularity helps. No latent transition, recurrent forecast, geometry-transfer, test, or filtering result exists. |

## Current Authorization

The frozen-artifact report queue remains authorized. L3R-B0P is spent and B0 was
not launched. B1P passed; B1 is now done/closed without an eligible checkpoint.
BC0P, BG0P/BG0, RB0P/RB0, and RA0P are also spent and closed; BC0 and RA0 were
not launched. The boundary-information line remains closed and L3R-K2D0 is
complete. L3R-K2T0 has a zero-step 20260730a argument failure and a zero-step
20260730b source-identity preflight failure; neither supplies a control result.
The human-reviewed isolated scientific source passes local tests, but no retry
is authorized. Minimum recovery requires human review of the one-line expected-
tracker correction, a complete launcher-constant-versus-manifest/input audit,
a new unique attempt, and a new launcher hash. No other retry, continuation,
changed tolerance, allocator, coefficient, batch, environment, host, second
seed, learned boundary module, or training change is authorized. The report
queue may:

- compare D044 and D060 state, high-pass, front, and geometry-stratum curves;
- show D053 propagated-versus-fresh shares beside the D052 branch
  interventions;
- place D048/D049 discrete-decoder amplification beside D062 front-identity
  failure;
- tabulate objective, sample presentations, selected epoch, intervention,
  population, evidence grade, and non-claim; and
- preserve failed implementation attempts only as provenance.

L3R-M0 and its later explicitly authorized read-only retrieval are complete.
P/D019 is now bound to the exact 76,634,327-byte checkpoint, SHA-256
`31364c48...796b5`, and to the recovered preprocessing/evaluation/visualization
chain; its collaborator-side training epoch, source, command, optimizer history,
and selection rule remain missing. R/D041's base and continuation best/last
hashes, exact commands, logs, split/normalization sidecars, processed-data
manifest, and already-open output hashes are recovered; its exact dirty runtime
source is not. No material backbone-capacity change exists: the residual model
adds only 640 input-projection parameters. The four attribution headings remain
qualified bundles, with checkpoint selection separated analytically from
continuation. During M0 no checkpoint was executed, no model was trained, no
new or sealed split was opened, and zero GPU-hours were used. The later U0/U1
diagnostic exception executed only the exact selected checkpoint on already-open
trajectory `05`; it performed no training or new/sealed population access.

The report-only failure-mode decomposition is also complete. On the bump
family, D041 verifies enormous reference-smooth graph-high-pass growth and
three finite local negative-internal-energy/pressure terminations, but it does
not retain the exact propagated-versus-fresh map split. Boundary policy changes
failure timing without eliminating the failure class. On the separate dynamic
family, corrected D053b verifies that full-state error is mostly propagated
while smooth high-pass defect is mostly freshly regenerated; D052 finds no safe
uniform branch attenuation, and D044/D060 show that fewer calls can improve
state error while worsening the registered high-pass/front conjunction. These
categories must not be collapsed into “instability,” and the dynamic source
split must not be transferred to the bump checkpoint.

The report-only ripple/high-frequency audit is now also closed. Operationally,
a ripple requires nonphysical oscillatory error in a reference-smooth region;
high-frequency energy is only a band diagnostic; and instability is separately
global growth, loss of admissibility, or termination. On all 20 D041 bump
holdouts, the post-hoc call-20 failure-ranking AUC is `0.392` for global smooth
high-pass energy versus `0.922` for both state and shock-thickness error. The
three failed nodes are locally high-pass and all lie on the boundary, but exact
U1 traces stay strongly high-pass during admissibility recovery. The bump
causal role is therefore unresolved and composite. On the dynamic family,
D053b/D054 identify a genuine smooth oscillatory defect that is mostly freshly
regenerated by the one-step map, but every registered H60 proposal is
admissible; D055 localizes defect rather than failure, and D056 rejects its
exact bounded correction. Generic smoothing is not selected. The tracker holds
the hashes, raw comparison table, missing evidence, and the mandatory anti-blur
contract for any later reviewed proposal.

The follow-on visual-phenomenon audit is also complete. On dynamic validation
case `sv_e06_y00`, the two right-moving error traces measure
`1.115/2.330` for D044 and `1.112/2.335` for D060, matching the downstream
entropy/contact and right-acoustic speeds `1.1133/2.3343`. They are seeded
before vortex impact and should not be collapsed into one ripple. The upward
left-region band is already present in the finite-volume reference, is an
acoustic boundary-associated transient under the extrapolated supersonic-inflow
plus symmetry contract, and D044 follows it with median phase lag `0.0125`;
the D044 vortex core is slightly early, not late. On the exact bump prefix,
`94.1--94.3%` of front-excluded pressure-error energy is downstream and its
signed high-pass alternates, while the thin front still owns most total
pressure-error energy. U1 re-enters admissibility on the current-source splice
but is not a learned limiter or guaranteed self-correction. Exact tables,
figures, alternatives, and minimum controls are in the tracker and the ignored
`l3r_wave_kinematics_audit_20260727` bundle. No run authorization changed.

The later U0/U1 continuation request is also closed. U0 stopped at its frozen
call-7 replay gate and its tolerance was not relaxed. U1 preserved D041 calls
1--33 byte-for-byte, continued only calls 34--79 with the exact checkpoint, and
encoded all 80 states. No nonfinite, nonpositive-density, `100x` amplitude, or
proxy-L2-`10` event occurs; final/max proxy L2 is `0.0317686`. Inadmissibility
is intermittent and always limited to one node: upper-left inflow node `6` on
calls `33--47` and `70--79`, and wall/outflow-front neighbor node `294` on
calls `49--51`, with full admissibility on call `48` and calls `52--69`.

Node `6` has a constant reference inflow state but is predicted recurrently,
has only one inflow label at the wall corner, an extreme mixed stencil, and a
`1.11273e-5` proxy-weight share that also downweights its training loss. Total
energy falls while kinetic energy rises through the first crossing. These are
verified vulnerability factors, not a sufficient causal attribution: the
lower-left corner has similar geometry/weight and stays healthy, and no
same-process `G(u_t)`/`G(uhat_t)` or branch trace exists. The transient node
`294` failure coincides with a reference pressure-front arrival and is kept
separate. The tracker binds the raw arrays, metrics, environment, exact hashes,
and a corrected all-80-frame animation whose red marker follows the actual
failed node.

The horizon-extension audit is conditionally positive but presently deferred.
H79 exhausts the evaluator-bound 80 states. The three D041 failures are
observed events, whereas the other 17 trajectories are right-censored at H79;
their fixed-cohort H40--H79 mean error is still approximately linearly rising.
No post-H79 current state appears in training, and state-distribution overlap
is unaudited. The raw bundle preserves representative Trixi lineage and one
unbound `sol_80.vtu`, but the exact Julia environment, per-case generator,
integrator, and HDF5 conversion contract needed for an identical extension are
missing.

Do not generate a longer reference before an eligible serious bump checkpoint
passes clean validation and its post-selection audit is frozen. After that
pass, a separate human review
may retrieve the literal generator contract and authorize
one nonsealed prefix-parity/runtime pilot. Only a passing pilot can unlock the
proposed H159 evaluation: 159 learned calls through `t=3.975`, full matching
truth, H20/H40/H60/H79/H99/H119/H139/H159 checkpoints, and joint
teacher/propagated/state/front/high-pass/boundary/admissibility/support metrics.
The purpose is to distinguish smooth accumulation, an abrupt delayed recurrent
basin, post-H79 state-distribution shift, and learned fixed-point bias. A
truth-free extension may be described only as qualitative finiteness or
admissibility.

No training branch is now running. B1 stopped exactly under its prospective
wall guard after 34/40 passes, 725,220 presentations, and 183,600 steps. No
checkpoint passes the three-part eligibility rule; no D041 holdout or sealed
population was opened. The retained epoch-22 checkpoint has SHA-256
`221d12c3cd5f3df65546bb02537354fce334b481ab36478e7ff9f35f8e4323dc`,
30/30 H79 completion, five-key H20 `0.02434854879975319`, and fixed-128
all/normal/boundary one-step
`0.010221920889307512/0.004201662173727527/0.07026912062428892`. Exact-source
FP32 validation confirms an H20--H60/front-position gain over BG0 that erodes to
a 0.41% H79 state regression, with mixed shock strength/thickness behavior.

RA0P independently fails its frozen BF16 ratio-repeatability gate and closes
before smoke or training. L3R-MR0 remains deferred, L3R-MR1 blocked, L3R-L0
mechanism-conditional, and L3R-F0 sealed. Historical R0/A0 evidence gaps remain
visible.

The replacement is deliberately training-first rather than an architecture
scale-up: exhaustive transition coverage, AdamW with 2% warmup and cosine decay,
BF16, and the same legal boundary closure at input, output, validation, and
recurrence. No soft boundary loss or smoothing is bundled. B0P remains failed;
B1P uses the newly reviewed comparator-replay and full-coverage gates rather than
relaxing its tiny-fit threshold. Exact hashes and metrics are in the decision
file and tracker.

BC0P preserves exact D041 weights, split, normalization, 19.16M architecture,
seed, and historical low-rate exposure. It adds only an explicit raw-to-causal
initialization transition and memory-safe microbatch `2` times accumulation `2`
for effective batch `4`; a matched raw continuation controls for extra
optimization. Both use fresh AdamW state, constant `2e-4`, ten times 2,048
presentations, and the same sample stream. This is a complete native boundary-
contract block, not a pure projection-only ablation. No multistep, soft boundary,
noise, smoothing, filtering, limiter, floor, or capacity change is bundled. Its
source/data/replay/policy/stream gates passed, but the finite raw preflight epoch
reserved `15,562,964,992` bytes against the registered `15,032,385,536`-byte
limit. Causal timing and both arms were therefore not run. This is engineering
preflight evidence, not a test of the boundary intervention.

The MR0 planning control now matches both retained serious parents exactly:
`51,200` total transition presentations, `13,400` trajectory-homogeneous
optimizer steps, and `19,155,720` parameters. The equal stride mixture reuses
the fixed Mach-1.1 family's constant standardized Mach input for normalized
physical `Delta t`, rather than appending capacity. Primary evaluation is the
raw 30-call stride-2 path through physical H60, with the same checkpoint's
60-call stride-1 path and direct/composed consistency reported. This is a
prospective contract detail, not execution authorization.

After clean bump reproduction and attribution, the preferred planning order is
algorithm before scale. A later review may register one bounded screen of at
most three distinct candidates: differentiable two-call recurrence, a
shock-excluded smooth-region high-frequency error objective, and a
primitive/admissibility auxiliary objective with conservative recurrence. This
is not an active run or permission to reopen the stopped D015 generated-state
exposure, generic smoothing/noise, or post-hoc branch-control routes. Only an
individually passing candidate may receive a clean from-scratch confirmation;
otherwise scale optimizer presentations, capacity/data, and reference horizon
one axis at a time.

The decision file also records a deferred front-capture substitution candidate
from the visual-phenomenon audit. Its smallest test keeps the reproduced setup
fixed and adds only a current-state-front-local weighting of outgoing
prediction-minus-target Euler characteristic error, with front geometry,
unweighted modes, admissibility, boundary, and state-error guards. Shock-phase
augmentation and transport/local-flux architectures remain separate later
escalations rather than a bundled method. This note does not add a fourth screen
row, authorize training, or change the registered L3R order.

## Sealed Boundaries

- D044 and D060 are single-seed, position-OOD-validation results. Strength-OOD
  and test splits remain sealed; casewise repetition is not seed robustness.
- The serious PCNO rows predict conservative-state residuals. They emit no
  learned face exchange and are not conservative by construction.
- Physical balance and reference-impulse diagnostics are valid only on the
  audited shock-vortex finite-volume contract. Equal-node or reconstructed
  weights on the bump bundle are proxies, not physical control volumes.
- The public CPG release uses next-reference boundary injection. Its oracle row
  is not a fair autonomous baseline, and the local bundle is not an established
  paper-table reproduction.
- Line 1 establishes a bounded operating-envelope tradeoff, not a learned CFL
  theorem or a timestep-conditioned, mesh-invariant solver.
- Line 4 has not passed representation identity and closure. Transition
  training and data assimilation remain unauthorized.

## Frozen Evidence Pointers

- The completed Frozen Cross-Line Claim Matrix and Result-to-Claim Gate are in
  [`RESEARCH_DIRECTION_DECISION.md`](RESEARCH_DIRECTION_DECISION.md).
- Exact run contracts, outcomes, hashes, and stopped routes are in
  [`MECHANISTIC_DIAGNOSTIC_TRACKER.md`](MECHANISTIC_DIAGNOSTIC_TRACKER.md),
  especially the L3R-M0 provenance audit, the 2026-07-26 failure-mode
  decomposition, D031-D039, D014 and the legal-boundary closeout, D037, D044,
  D048-D053, D060-D062, and L4A-001-L4A-004.
- Historical corrected 1D baseline labels, final results, and the frozen-run
  analyzer contract are in
  [`SECTION_1_2_CORRECTED_BASELINES.md`](SECTION_1_2_CORRECTED_BASELINES.md).
- The live bump schema and frozen provenance records are
  [`CPG_EULER_DATASET_CONTRACT.md`](CPG_EULER_DATASET_CONTRACT.md),
  [`BUMP_300_DATASET_AUDIT.md`](BUMP_300_DATASET_AUDIT.md), and
  [`CPGGNSPDES_REFERENCE_AUDIT.md`](CPGGNSPDES_REFERENCE_AUDIT.md).
- Exact generated reports, arrays, figures, checkpoints, and source manifests
  remain under ignored `artifacts/time_dependent_no/`; compact documentation is
  not a substitute for those evidence bundles.

## Repository State

- Branch: `time-dependent-no`.
- Commit `31e5765` preserves the complete pre-document-cleanup documentation
  surface.
- Commit `729091b` preserves the exact pre-code-cleanup diagnostic source;
  commit `cf6cbe1` prunes the closed experiment scaffolding.
- The post-code-cleanup `tests/time_dependent_no` suite passed `306` tests with
  two existing Torch JIT deprecation warnings.
- The U0/U1 continuation surfaces pass eight focused CPU tests. Their runtime
  and post-run visualization-correction hashes are frozen in the tracker.
- The maintained code surface and the ADER-generator invocation warning are
  listed only in [`README.md`](README.md).
- Machine-specific paths and credentials remain in ignored local context;
  generated research outputs remain ignored under `artifacts/time_dependent_no/`.

## Next Human Review

BC0P is closed and BC0 stays unlaunched. BG0P/BG0 are also complete and closed:
both candidate soft objectives fail the frozen state-error gate despite a small
boundary-node improvement. RB0P/RB0 answer the narrower question: direct raw
reference supervision does improve raw boundary fit and H20, but most of the
gain is overwritten by hard closure and the selected H79/front-high-pass joint
gate fails. The exact full-weight objective is rejected rather than promoted as
a partial method win. RA0P then fails its numerical-repeatability gate before
testing the weak auxiliary, so RA0 supplies no utility evidence.

The B1 review is complete: no parity-eligible checkpoint exists, so the D041
holdout remains closed. The requested validation-only boundary-contract
decomposition separates physical residual, corrected reference trace, raw
proposal, intervention, near-boundary, and future-interior errors. The later
authorized call-matched `K=2` gate is also complete. Its attached recurrent-
gradient arm fails one-step and H20; its projected-teacher control improves
H20 all/normal by `5.32%/5.44%` and H79 all/normal by `6.33%/6.48%`, with
30/30 completion. However, the paired H79 front-centroid ratio is `1.15102`,
so the checkpoint fails the registered `1.05` structural envelope and is not
selected. Training summary SHA-256 values are
`0a9b6e5df6fcb2054dffe9e68fbc3ebd8b446db5a99723e0ec853e26e7d310d9`
and `ef7724fd07adfba1b77acb619ccd48fc59082c2b9eb43376773846fe49369d6c`;
teacher H79 summary SHA-256 is
`c48c7b63dad42cbc051483a830fd323999505415d609fc93734d3e0ef8894341`.

The no-training boundary/interior output splice is complete. Its source archive
SHA-256 is
`eedf3793dcdc7c854a71e279fbcd1a41748c5f28608e5838921763e415aa6a99`;
its matched H20 summary SHA-256 is
`3e3b33ce25f4b9f0fd8f5ee288a5b43ce379628206a5597741f82fe51b3ab17a`.
D041 normal-node output plus teacher boundary output retains only
`18.81%/17.20%` of the teacher's all/normal state gain and fails the thickness
and strength structural gates. The reverse hybrid retains
`136.82%/137.45%`, showing that the teacher's state gain is primarily in its
normal-node/interior output, but it also fails thickness and strength. Both are
30/30 complete and admissible. The boundary gate is closed: no H79, frozen-base
boundary adapter, or new boundary training is justified for this pair. Route
the interior signal and structural failures to general PCNO work; boundary
localization may re-enter only after a one-call checkpoint passes joint state
and structure gates.

A post-closeout implementation review leaves the frozen H20 result unchanged
and bound to its recorded v1 source hash. The maintained splice evaluator is
now schema v2: it is frame-zero-only because the v1 references omit start-frame
provenance, validates reference stride and every policy digest, freezes both
child modules literally, records the start frame, and requires 30/30 paired
structural values. Its evaluator/test SHA-256 values are
`ea48ce218f0bbeb29e5686cd248803d628af6e559895f290ea569d0fc1477a38`
and `2beeb3f60d01480efd65f8b143838cf878d5b766089cccf9578ad68069693c69`;
9 focused and 50 combined relevant CPU tests pass. No GPU replay, new result,
run authorization, or scientific disposition follows from this hardening.

The boundary-information line is therefore safe to close for the current bump-
PCNO lineage. Its result-to-claim verdict is `partial`, high confidence: causal
minimum-change projection is retained as a deployment invariant with an anti-
smearing caveat, while no learned boundary objective or adapter is selected.
This does not close exact DG replay, characteristic outflow, corner fluxes,
conservation, repeat-seed, sealed, or cross-family questions. Re-entry requires
a future one-call checkpoint that first passes joint state/structure gates and
then shows a material boundary-local residual under the six-channel
decomposition. Optional independent Codex review remains
`[pending Codex review]` because no private-result transmission was approved.

A completed miss in B1, the bounded pilots, RA0P, BC0P, BG0, RB0, attached
`K=2`, or the output splice is a result, not permission to retry, tune the
optimizer, smooth/filter, add a seed, transfer family, or access sealed data.

Do not claim a paper-level CPGNet comparison until dataset, split,
checkpoint/evaluator, boundary, horizon, stride, and primitive-metric parity are
bound. The algorithmic screen, any subsequent scale ladder, and novelty review
all require later explicit contracts; none is authorized by this handoff.

Line 4 remains stopped. No representation, latent transition, test access, or
data-assimilation action is revived by the Line 3 decision.
