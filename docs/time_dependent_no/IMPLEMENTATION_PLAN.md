# Implementation Plan: From Paper Skeleton To Prospective PDE Evidence

Updated: 2026-09-05

Status: active implementation roadmap. The completed R0 and five-arm successor
remain immutable parent evidence. The extension under
[B3B4_NACA_CORRECTIVE_EXTENSION_PREREGISTRATION.md](B3B4_NACA_CORRECTIVE_EXTENSION_PREREGISTRATION.md)
and its JSON contract has completed implementation, training, and development
evaluation. Its claim-bounded paper integration and rebuild are complete;
the corrected derived presentation is also complete. Prospective and sealed
roles remain unopened. The selected B5 bump transfer now authorizes only local
Stage-0A/B identity/reducibility and synthetic implementation; no remote resource
or dataset-scale launch is selected.

## 1. Design Rules

- Primary-PDE readiness comes before a broad intervention implementation.
- Keep PCNO fixed and express project code under `utility/time_dependent_no/`,
  `scripts/time_dependent_no/`, and `tests/time_dependent_no/`.
- Reuse current runtime, rollout, solver, and artifact utilities before adding
  abstractions.
- Separate clean-data geometry, trusted solver response, training targets,
  deployed composition, diagnostics, and outcome analysis.
- Fit geometry and drift instruments on clean training/reference states only.
- Keep solver-relative response offline. Core deployment makes no solver call
  and uses no OOD trigger.
- Build only the representatives retained by the closed parent and active
  extension contracts.
- Every arm receives a code-to-intent audit before a scientific smoke.

## 2. Verified Live Starting Point

The 2026-08-30 source/artifact audit established:

### Kolmogorov

- `utility/time_dependent_no/kolmogorov_reference.py` and
  `tests/time_dependent_no/test_kolmogorov_reference.py` match both retained
  spatial/temporal qualification source manifests.
- Both retained result bytes match their artifact manifests; the temporal
  manifest itself rehashes to the recorded value.
- The solver exposes projected arbitrary-state advancement, but the
  canonical mean-zero/dealiased projection may itself act as a structural
  correction.
- No current Kolmogorov PCNO dataset builder, model adapter, trainer,
  evaluator, or result manifest exists.

### Shock-vortex

- The live fine/coarse solver, family builder, shard preparation, and baseline
  evaluator match the retained D074 source manifest.
- The current `pcno_euler2d.py` and residual trainer do not match that old
  manifest, so old results are not presumed checkout-compatible.
- The fine solver starts from its parameterized clean initial condition and
  does not accept arbitrary predicted states. The native coarse solver does,
  but its prior fidelity gate did not qualify it as the trusted displaced-state
  target.

### Reusable diagnostics and runtime

| Existing file | Reuse | Limit |
| --- | --- | --- |
| `utility/time_dependent_no/path_conditioned_tube.py` | Path-conditioned secants and algebra. | Does not evaluate `Phi(u+eta)` and cannot measure trusted response. |
| `utility/time_dependent_no/pcno_propagated_sensitivity.py` | Model lookahead and gain diagnostics. | Model-side only until paired with a qualified solver response. |
| `utility/time_dependent_no/pcno_runtime.py` | Canonical model loading/runtime support. | Must bind a new study identity. |
| `utility/time_dependent_no/pcno_rollout.py` | Shared recurrence support. | Must preserve raw versus corrected recurrence explicitly. |
| `utility/time_dependent_no/p0_restart_sufficiency_a2.py` | Restart provenance and bias-gate patterns. | Historical result/identity is not reused. |
| `scripts/time_dependent_no/decompose_pcno_euler2d_rollout_error.py` | Exact error-decomposition pattern. | Family/state contract must be rebound. |
| `utility/time_dependent_no/pcno_artifacts.py` | Source/result/closeout manifest patterns. | Packet integrity and checkout compatibility remain separate. |

No current generic train-only PCA/kNN support estimator, qualified
tangent/normal projector, displaced solver-response evaluator, or empirical
`q`, `rho`, and complete-map gain estimator was found.

## 3. Phase I0: Frozen NACA Interface And Stage-0 Audit

Phase I0 is complete for byte identity and state-interface inspection:

- [R0_PRIMARY_PDE_READINESS_CONTRACT.md](R0_PRIMARY_PDE_READINESS_CONTRACT.md)
  freezes the scientific role, gates, state, recurrence, and claim boundary;
- [R0_SU2_NACA_RESOURCE_MANIFEST.json](R0_SU2_NACA_RESOURCE_MANIFEST.json)
  binds the public license, configuration, mesh, restart histories 497/498,
  target 499, and intended SU2 release;
- `utility/time_dependent_no/su2_restart_contract.py` strictly validates the
  pinned configuration, mesh, native binary layout, field order, coordinates,
  and complete RANS--SA state; and
- `scripts/time_dependent_no/audit_su2_naca0012_resources.py` exposes that
  read-only audit, with focused synthetic tests.

The evolved state is `[Density, Momentum_x, Momentum_y, Energy, Nu_Tilde]` on
the fixed mesh. Second-order dual-time continuation binds both histories 497
and 498. Stage 0 alone does not establish executable compatibility, restart
sufficiency, solver accuracy, throughput, or a trusted transition.

The preparation-only replay harness is complete. It stages a clean case
atomically, binds the configuration/input hashes, refuses to copy or modify the
pinned target, stages only histories 497/498, uses a distinct replay output
stem, explicitly sets `WRT_RESTART_COMPACT=NO`, and fails closed on authority
drift or an interrupted copy. It never invokes SU2.

The narrow native executor/evaluator/comparator is now complete. It binds the
official executable identity, requires exactly output index 499, enforces the
frozen native 19-field output schema, name-aligns the canonical 17 fields,
rehashes target and sources, records runtime and determinism, and fails closed
on identity or completion mismatch. Two independently prepared final cases
execute successfully and produce bitwise-identical outputs. Their five evolved
fields have component-balanced relative L2 discrepancy `4.3264e-7` under
physical-area weighting, `5.9447e-4` under uniform-node weighting, and
`8.9081e-4` in the fixed near-body/wake box, with `1.92`--`1.96`
transitions/s observed locally.

The later model-relative quarter-margin evaluation failed the uniform-node and
near-body/wake views while passing the physical-area view. No
intervention-relative margin was opened because no NACA corrector ran, and it
is not pending under this closed identity. The qualification addendum records
that original producer-integrated coefficients are unavailable, freezes the
common surface functional and auxiliary-field scope, and projects storage.

The maintained multi-step path is
`utility/time_dependent_no/su2_naca_trajectory.py` with entry point
`scripts/time_dependent_no/run_su2_naca0012_trajectory.py`. It produced 1,501
native states over indices 499--1999 in `5631.17 s`, with an ordered-state
aggregate SHA256
`9adbcfd53c4812009d3c5e5081730b709a2f63f48237760390e2c8559ab200ef`.
The one-step smoke prefix is byte-identical to the full run, but its short-run
throughput is not extrapolated to production.

The solver-only phase analyzer is
`utility/time_dependent_no/su2_naca_phase_pilot.py`, exposed by
`scripts/time_dependent_no/analyze_su2_naca0012_phase_pilot.py`. Under the
clarified pre-analysis contract SHA256
`f442065172beb014ebcd4f3823abe1e2764fb2b74bbad348a8dd94d6b4c903f5`,
it returned `PHASE_ONLY_SUPPORTED`: period `34.6543` steps, 31 complete cycles,
and passing uniform-node plus near-body/wake recurrence. The required
physical-area secondary view fails and remains reported. The runner/analyzer
suite closes 43 focused tests, Ruff check/format, and independent audit.

The frozen train/development-only materializer, trainer, and aggregate evaluator
are implemented and independently audited. Their dataset, full-grid smoke,
three registered training packets, evaluation packet, and exact negative R0
decision are complete. Preserve these source and artifact identities; do not
reuse them to authorize a changed horizon, candidate, or correction study.

## 4. Phase I1: Closed-Parent Geometry And Response Diagnostics

The closed implementation is deliberately tied to the frozen NACA successor;
there is no generic corrective-response module. Calibration, training,
evaluation, replay, and rendering are owned by
`calibrate_pcno_naca0012_successor.py`,
`train_pcno_naca0012_successor.py`,
`evaluate_pcno_naca0012_successor.py`, and
`visualize_pcno_naca0012_successor.py`, with shared state/model logic in
`utility/time_dependent_no/pcno_naca0012_successor.py`.

The implemented diagnostics are exactly those needed by the successor
contract: train-path PCA and nearest-segment projection, matched-energy IID and
error-subspace recovery calibration, model-generated-prefix forcing and
finite-amplitude gain, recurrent state/path/projected-path error, graph and
boundary structure measures, physical admissibility, and complete deployed-map
accounting. Solver-relative response at arbitrary displaced states is not
claimed, because no such trusted SU2 query population was generated.

Focused synthetic tests cover normalization, path projection, arm semantics,
metric aggregation, manifest closure, input/output disjointness, protected-role
authorization, one-shot access receipts, and non-clobbering packet publication.
All solver-derived measurements remain offline diagnostics; none is an online
OOD rule or a deployment input.

## 5. Parked Fallback I2A: Minimal Kolmogorov PCNO Screen

The Kolmogorov route remains parked while the NACA successor is active. Reopen
it only through a new owner-approved contract.

Add only the missing path needed for the readiness question:

| Proposed file | Purpose |
| --- | --- |
| `utility/time_dependent_no/pcno_kolmogorov.py` | State normalization, periodic PCNO graph/geometry, residual transition, and canonical-projection accounting. |
| `scripts/time_dependent_no/build_pcno_kolmogorov_dataset.py` | Immutable population/split generation from the qualified reference. |
| `scripts/time_dependent_no/train_pcno_kolmogorov.py` | One fixed clean PCNO baseline; no method flags. |
| `scripts/time_dependent_no/evaluate_pcno_kolmogorov.py` | Teacher-forced and autonomous rollout, energy/enstrophy/spectrum, support drift, and projection residual. |
| `tests/time_dependent_no/test_pcno_kolmogorov.py` | Representation, graph, recurrence, tiny-fit, split, and manifest checks. |

Required checks:

1. dataset restart equality and split immutability;
2. periodic wraparound edges, weights, and graph invariance;
3. normalization/state round trip and residual identity;
4. autonomous recurrence consumes the previous model output;
5. raw versus canonicalized state and projection norm are always retained;
6. teacher-forced/free-rollout decomposition closes;
7. tiny CPU overfit and two-step checkpoint round trip; and
8. manifest tampering fails closed.

Readiness stops:

- qualify the actual model-grid lift/restrict/reference contract;
- projection residual is negligible or declared as part of every deployed map;
- production data cost fits the measured envelope;
- clean support has stable structure under train-only diagnostics; and
- an honestly trained PCNO has good clean H1 but severe endogenous rollout
  degradation.

Failure of any item rejects Kolmogorov as the primary case without tuning the
data or training to manufacture failure.

## 6. Conditional Phase I2B: Time-Boxed Shock Restart Audit

Shock-vortex is currently a proposed tangential-dynamics/no-harm control, not
the default primary case. Activate this audit only if the owner selects that
role and its solver-response measurements remain necessary.

Do not retrain the shock PCNO first. Reuse the existing fine/coarse solvers and
P0 audit patterns under a new source identity.

Audit three possible common-state routes:

1. solver-native advancement on the PCNO grid;
2. qualified coarse-to-fine lift, fine evolve, and restrict; and
3. a requalified native coarse transition.

For clean and controlled displaced states, verify:

- byte/semantic identity between model and solver state;
- admissibility and boundary/forcing closure;
- clean one-step closure;
- native or lifted bias against fine/restricted advancement;
- boundary and flux balance under displacement;
- current checkpoint reconstruction or explicit incompatibility; and
- train-only support fitting.

Hard stop: restart bias must be smaller than both the baseline discrepancy and
the smallest intervention separation the paper intends to interpret. Boundary
or lift artifacts that dominate the response reject shock-vortex before
training.

## 7. Completed Phase I3: NACA Primary-Case Decision

After native replay and population qualification, the frozen development-only
question was whether NACA0012 should become the primary case.

The baseline state is the complete BDF2 pair
`z_n = (u_{n-1}, u_n)`, with five evolved fields per history. A narrow adapter
around the generic `pcno.PCNO` predicts a five-field residual relative to
`u_n`; autonomous recurrence shifts
`(u_{n-1}, u_n) -> (u_n, u_n + Delta_theta(z_n))`. It predicts every node and
uses no oracle boundary replacement, clipping, flooring, or smoothing. A
one-frame baseline is inadmissible unless a separate representation gate first
shows that dropping numerical history is negligible.

The pre-outcome plan listed:

1. fixed-PCNO clean H1;
2. ID autonomous rollout and failure timing;
3. train-only support structure;
4. temporal order of support drift, solver-response deterioration, and rollout
   failure;
5. structure/validity failure channels; and
6. data, training, response-query, and reveal cost.

The promotion rule required the full conjunction. A positive result would have
frozen NACA0012 as the hard case. Any failure preserved the exact negative
result, returned candidate selection to the owner/mentor, and prohibited
intervention code on the failed hero case.

The registered screen completed with three adequately trained seeds. All three
208-step rollouts are finite and severe late, but early train-state-scale window
errors `0.2669/0.2368/0.2458` all fail the strict `<0.15` gate. Native replay
also fails the uniform-node and near-body/wake strict quarter-margin views. The
exact decision is `R0_REJECT_NACA_AS_HERO_CORRECTION_NECESSITY_CASE`. NACA is
closed as the primary case and retained only as supplementary discrepancy and
structure-diagnostic evidence. Decision precedence closed promotion at the
early phenotype gate before train-only support and solver-response ordering
were qualified. The screen therefore does not establish manifold drift or a
transverse corrective mechanism as the cause.

Current owner direction subsequently selected a new NACA successor for a
narrower question. That successor is not an extension of this R0 identity and
does not inherit its unmeasured causal claims.

## 8. Completed Phase I4: NACA Calibration And Implementation Freeze

For the selected PDE, the successor implementation has:

- built a train-only calibration packet containing the three parent
  teacher-forced error banks, matched IID field scales, the rank-16 parent-error
  pair basis, and the train-path projector;
- distinguished IID and parent-error-subspace recovery from detached
  model-prefix exposure;
- run only permitted train/development calibration and resource checks;
- frozen signed intervention-mediator and ranking predictions; and
- bound the exact five-arm contract before development evaluation.

The calibration metadata and arrays bind their source, dataset, parent,
normalizer, contract, and array-member hashes under a final hash manifest.

## 9. Phase I5: Implemented Arms And Verified Development Evaluation

The maintained implementation is:

- `utility/time_dependent_no/pcno_naca0012_successor.py` for recovery
  presentations, detached-pushforward presentations, train-path projection,
  identity parity, and corrected-state recurrence;
- `scripts/time_dependent_no/calibrate_pcno_naca0012_successor.py` for the
  parent-error subspace and train-path calibration packet;
- `scripts/time_dependent_no/train_pcno_naca0012_successor.py` for the four
  learned arms; and
- `scripts/time_dependent_no/evaluate_pcno_naca0012_successor.py` for the
  development common-bank, rollout, structure, cost, and path-projection
  packet, plus the safety-hardening prospective-freeze path.

The exact comparison is `CLEAN`, `IID_RECOVERY`,
`ERROR_SUBSPACE_RECOVERY`, `DETACHED_PUSHFORWARD`, and the operational
`PATH_PROJECTION` deployment on each clean predictor. Always retain raw
prediction, correction displacement, corrected state, recurrent input,
schedule, conditioning, and cost. No learned operational corrector, dynamics
relabeling arm, structural arm, or hybrid is part of this frozen successor.

Focused tests must verify:

- IID and error-subspace recovery use their declared input laws and the same
  displaced-current-to-clean-future recovery target;
- detached pushforward uses a generated, gradient-detached prefix with its
  stored clean future target;
- identity corrector is byte/parity equivalent to raw recurrence;
- corrected state, not raw prediction, is fed to the next step;
- predictor/corrector composition order and schedule;
- PCNO architecture, seeds, data order, and checkpoint selection equality;
- no validation/test truth or designated horizon enters fitting or selection;
- stochastic reproducibility; and
- source/data/model/corrector/evaluator mismatch refusal.

All twelve learned-arm/seed training packets and the development-only result
packet are locally verified. The result packet's final-manifest SHA256 is
`09e3ae7e5896039e2230073c1ead5e5f9808c0cc9b47d0b15ff337b26b3e0ed1`.
All evaluated rollouts are finite. `PATH_PROJECTION` improves both primary
metrics in all 24 paired comparisons and improves pressure and graph-Dirichlet
diagnostics; its zero corrected path residual is construction-only.
`IID_RECOVERY` helps, `ERROR_SUBSPACE_RECOVERY` improves `CLEAN` but not IID
recovery, and the registered `DETACHED_PUSHFORWARD` mechanism is falsified with
seed-unstable rollout effects. No physical off-manifold or arbitrary
displaced-state SU2 response claim follows.

## 9A. Completed Phase I5E: NACA Corrective Extension

`B3B4_NACA_CM_EXT_20260902A` was implemented under new NACA-specific source and
packet identities while reusing the verified parent dataset/model loading,
normalization, BDF2 recurrence, graph, and evaluation semantics. Do not modify
or overwrite the parent trainer, evaluator, checkpoints, or packets.

The extension must implement exactly:

- `MP_PDE_PUSHFORWARD_M01`: clean epoch 1, then batch-shared depth uniform on
  `{0,1}`, current-online stopped-gradient prefixes, terminal loss only, and
  logged requested/realized depth caps at the train-role boundary;
- `CURRICULUM_EMA_PUSHFORWARD_K13`: registered exposure ramp, EMA evaluation-
  mode prefixes at depth `1--3`, online/EMA reporting, and `CLEAN_EMA` control;
- one frozen 476-input displaced-BDF2 bank shared byte-for-byte by
  `PAIRED_RECOVERY` and `DYNAMICS_RELABEL`, changing only the target; and
- `PCNO_PDEREFINER_K3_VPRED`: normalized-residual DDPM velocity prediction at
  four scheduler levels, one shared PCNO initialized from the matched seeded
  law with an exactly zero output head, EMA deployment, and four model calls
  per physical step.

The first implementation seam is a native-restart writer that templates both
BDF2 histories, replaces exactly the five evolved fields, preserves coordinates
and all auxiliary fields, uses absolute indices, and records round trips and
changed-field lists. Focused CPU tests cover writer/index alignment, clean
versus relabeled targets, paired-bank identity/sign schedule, prefix depth and
target alignment including boundary caps, stopped gradients, EMA
update/deployment, DDPM targets and
reverse recurrence, fixed stochastic replay, four-call accounting, protected-
role refusal, and manifest mismatch failure. An independent agent must trace
every mathematical input, target, information source, recurrence, and cost
before scientific execution.

Only after those tests and audit pass may the exact eleven-call train-only SU2
pilot run. Any replay, convergence, admissibility, repeatability, auxiliary-
column, scale, response-separation, or solver-completion failure stops the
paired bank without dropping or replacing a case. A full pilot pass unlocks
bank generation, then full-resolution resource smokes, registered training,
and common development evaluation. Offline solver labels are allowed; online
solver calls, online defect triggers, prospective reads, and sealed reads are
not.

Closeout: the extension core, restart path, corrected pilot R1, 476-input
paired bank, full-resolution smokes, 18 training packets, and common evaluator
all passed their registered gates. Attempt E reused Attempt-D training without
mutation and closed the development evaluation under final-manifest SHA256
`15475da78b4182754fe959eadc175434ffca18ef1558a45ea06cbf9c93ca7370`.
The locally retrieved training packet contains verified metadata and histories,
not duplicate checkpoints. No online solver call, online defect trigger, or
protected-role access occurred.

## 10. Phase I6: Compact ODE Study

Maintained entry points:

- scripts/time_dependent_no/run_corrective_ode_study.py generates and verifies
  the exact and learned calibration packet; and
- scripts/time_dependent_no/plot_corrective_ode_landscapes.py is a read-only
  derived renderer for the canonical learned checkpoints.

The renderer verifies its parent manifest and the runner source bound by that
parent, reconstructs the recorded training bank, and then evaluates all
checkpoints on one common phase--radius grid. It writes mean trusted-flow
defect and clean-return-error fields, recomputed clean-start and normal-impulse
trajectories, the exact training supports, and a derived manifest. The parent
packet is never modified.

Focused tests cover the exact flow, recurrence, ranking reversal, wrong phase,
false attractor, recovery/relabeling semantics, proxy geometry, deterministic
splits, packet closure, one-step landscape semantics, periodic-seam handling,
training-support reconstruction, and parent/source mismatch refusal.

The ODE phase is complete supporting calibration. It does not authorize a
harder ODE setting or imply that its method ranking transfers to B3/B4.

No executed ODE study contains genuine pushforward, multistep-loss, or
model-prefix-exposure training. `DYN-RELABEL` uses prescribed displaced inputs
with trusted successor labels. The completed PDE extension supplied the literal
MP-PDE and curriculum/EMA exposure comparisons; the closed
`DETACHED_PUSHFORWARD` arm remains a one-prefix parent stress test.

The owner subsequently authorized one separate bounded extension while the
primary-PDE choice was pending:
[B2_NL_NONLINEAR_ODE_STRESS_PREREGISTRATION.md](B2_NL_NONLINEAR_ODE_STRESS_PREREGISTRATION.md).
`B2-NL` uses a new runner, test, source identity, and artifact packet; it does
not edit the source-bound B1/B2 experiment plan, runner, tests, or landscape
renderer. It tests only the preregistered contractive/mixed/expansive response
interaction under shared recurrent forcing. Canonical local-CPU execution
begins only after focused tests and an independent `AUDIT_PASS`.

Closeout: the audited canonical packet is complete and verified, with formal
classification `response_qualification_failed`. Its P3--P5 rollout ordering is
descriptive only. Do not rerun, relax its gates, or tune its model under this
identity; a successor would require a new preregistration and authorization.

The fixed-support packet is retained as an internal response diagnostic, not as
the PDE-style noise-augmentation experiment. The separate read-only renderer
`scripts/time_dependent_no/plot_corrective_ode_nonlinear_diagnostics.py` loads
its verified checkpoints and produces dense trusted-flow-defect,
clean-return-error, and output clean-set-distance landscapes together with the
actual training support, learned/trusted trajectories, and visited-state
defects. It must bind every output to the parent manifest and checkpoint hashes
and must not write into the canonical packet.

The owner-authorized successor `B2-GN` replaces the fixed offsets with Gaussian input
corruption. Before code or training, its contract must freeze:

- covariance geometry and normalization; for the ODE laboratory, normal-only
  corruption is the current simplest proposal, while ambient or spatially
  correlated perturbations remain separate PDE choices;
- whether the swept scale is recorded as standard deviation, variance, or both;
- a preregistered scale grid and Gaussian-tail summary, without treating an
  unbounded Gaussian as a hard training tube;
- clean/noisy sampling proportions, resampling schedule, paired random draws,
  and targets for recovery versus dynamics relabeling; and
- `sigma_train` separately from recurrent `sigma_force`, whose law must be
  frozen independently.

The framework predicts scale-dependent tradeoffs, not a universal optimum. At
a fixed radius, the registered test relates local Gaussian density to fitted
response error; it does not treat larger standard deviation as monotone
coverage because every Gaussian has full support and its density at that radius
can be nonmonotone. All preregistered scales are reported; no scale is selected
post hoc from long-rollout performance.

Frozen B2-GN contract: use normal-coordinate corruption
`eta = (0, sigma_train z)`, `z ~ N(0,1)`, with
`sigma_train in {0.005, 0.01, 0.02, 0.04}` and report the corresponding
variances. Mix clean and freshly resampled corrupted inputs equally, reuse the
same indexed draws across arms and scales, train recovery toward the clean
successor and relabeling toward the trusted successor from the corrupted state,
and retain the registered recurrent forcing as a separate diagnostic law unless
the owner changes it. The exact schedule, tail, audit, and packet rules are in
[B2_GN_GAUSSIAN_NORMAL_NOISE_PREREGISTRATION.md](B2_GN_GAUSSIAN_NORMAL_NOISE_PREREGISTRATION.md).
The owner authorized implementation and the bounded local-CPU run on
2026-08-30; no PDE, GPU, remote, or sealed action is included.

Closeout: the independently audited implementation and canonical packet are
complete. Manifest
`8902430e479caae18e877407cfbc40c8ddd231bb55dc310bd765990a5af46172`
verifies 72 outputs, 51 checkpoints, six current sources, the frozen parent, and
the inherited forcing signs. Both registered fixed-radius density--fidelity
signatures pass. No scale passes the all-seed response qualification, so the
forcing outcomes remain descriptive and cannot select a scale or support a
mechanism-qualified rollout ranking. Preserve this packet unchanged; any new
threshold, model, scale, covariance, or training objective requires a successor
identity and owner authorization.

The inherited `base.training_radii` field in `config.json` is unused by B2-GN;
the Gaussian schedule and targets are bound by the schedule summary and training
digests. CLEAN and oracle rows are repeated under `comparison_sigma_train` only
for paired comparisons, so cross-scale summaries must deduplicate them.

## 10A. Active B5 Supersonic-Bump Transfer

The implementation authority is
[B5_BUMP_SOLVER_FREE_TRANSFER_PREREGISTRATION.md](B5_BUMP_SOLVER_FREE_TRANSFER_PREREGISTRATION.md)
and its JSON contract. Add only B5-scoped code under the maintained utility,
script, and test directories; reuse the current bump shard/runtime/artifact
paths but bind a fresh source/evaluator identity. Historical D094 packets are
parent evidence, not mutable implementation surfaces.

The immediate local scope is Stage 0A/B: fail-closed identity/leakage checks and
native-mesh train-only oracle and prefix POD diagnostics. Focused synthetic
tests must cover weighted POD, prefix/future separation, rank and reconstruction
metrics, variable-node graphs, admissibility/structure diagnostics, manifest
mismatch, and protected-role refusal.

Stage 0C phenotype replay and Stage 1 are specified but blocked pending their
contract amendments and a named resource. Stage 1 defines `CLEAN`, `IID_RECOVERY`,
`CURRICULUM_EMA_PREFIX_K13`, and `PREFIX_ERROR_CORRECTOR_K13`, with corrected-
state feedback and explicit cost accounting. A bump PDE-Refiner port remains a
conditional extension. Do not implement a cross-mesh PCA/kNN projector or add
method variants. Remote or dataset-scale execution requires a selected resource
and launch receipt; the historical test stays sealed.

## 11. Closed Parent Packet Architecture And Extension Rule

Each closed-parent learned-arm training packet contains exactly:

```text
best.pt
config.json
history.json
input_manifest.json
last.pt
runtime_manifest.json
status.json
summary.json
final_hash_manifest.json
```

The development evaluator writes eleven declared payload files plus its final
manifest:

```text
cost_metrics.csv
input_manifest.json
method_summary.csv
offline_diagnostics.csv
pairwise_summary.csv
result.json
rollout_metrics.csv
rollout_snapshots.npz
rollout_structure.csv
runtime_manifest.json
source_manifest.json
final_hash_manifest.json
```

For the parent, development is the only executed scientific evaluation. Its
audited evaluator and portable source-bound data-free prediction freeze are closed.
The immutable replay, historical v2 visualization, and final v3 presentation
packet are retained as distinct provenance layers. Any later prospective reveal
must validate the frozen predictions and separate owner authorization before
resolving or reading a protected scientific path. Prospective and sealed data
remain unopened.

The extension receives new training, pilot, bank, evaluator, result, and final
manifests. It may reuse packet helpers but must not claim the parent's exact
inventory if the registered EMA, stochastic-replay, intermediate-refinement,
solver-label, or cost records require additional files.

## 12. Code-To-Intent Audit

An agent other than the implementer traces:

1. mathematical input, target, objective, inference action, information, and
   expected mediator;
2. actual entry-point path through query, target, loss/corrector, recurrence,
   and metrics;
3. normalization, units, timestep, precision, forcing, boundaries, and one
   hand-checkable fixture;
4. identity/no-correction parity, raw/deployed separation, corrected-state
   recurrence, leakage, and reproducibility; and
5. separate predictor, corrector, bank, feature, evaluator, and result
   identities.

Verdict: `AUDIT_PASS`, `NEEDS_OWNER_CLARIFICATION`, or `AUDIT_FAIL`. Passing
unit tests alone is not an audit pass.

## 13. Test And Execution Ladder

1. analytical unit tests;
2. synthetic geometry, query-bank, and target tests;
3. composition, recurrence, parity, and leakage tests;
4. tiny CPU overfit and checkpoint round trip;
5. source/manifest mismatch failures;
6. independent code-to-intent audit;
7. bounded open-data read-only preflight under named approval;
8. measured GPU memory/throughput smoke under named approval;
9. registered training/evaluation; and
10. prediction freeze followed by separately approved reveal.

No stage is authorized merely because the prior stage passes.

## 14. Immediate Implementation Order

| Order | Work | Exit |
| ---: | --- | --- |
| 1 | Preserve the complete NACA R0 sources, contracts, packets, and exact negative verdict. | Complete: immutable parent evidence. |
| 2 | Implement `CLEAN`, the three learned interventions, and `PATH_PROJECTION` in NACA-specific files. | Complete: focused local tests pass. |
| 3 | Complete independent code-to-intent audit, calibration, authorizations, and full-resolution open-data smokes. | Complete: bound audit/smoke packets. |
| 4 | Train all four learned arms at seeds `17/29/43` and evaluate the open development population. | Complete: training and development result packets independently verified. |
| 5 | Freeze prospective evaluator and predictions. | Complete: source-bound prospective-ready packet with no protected access. |
| 6 | By Friday `2026-09-04`, render permitted verified figures/animations, finish the honest PDE manuscript section, and leave any prospective decision separate. | Complete: final presentation packet, clean 35-page parent paper, and exact mounted-checkout WSL verification. |

The table above is the closed parent. The completed extension order is:

| Order | Work | Exit |
| ---: | --- | --- |
| E1 | Add extension-scoped implementation and focused synthetic/CPU tests without changing parent sources or packets. | Complete: core, relabel path, generator, trainer, and common evaluator pass focused regression. |
| E2 | Complete independent code-to-intent audit and freeze source identities. | Complete: `AUDIT_PASS`; trainer and evaluator bind the full behavior-bearing source closure. |
| E3 | Run the eleven-call train-only SU2 relabeling pilot. | Complete under corrected pilot R1; every registered gate passed. |
| E4 | Generate and verify the 476-input paired bank, then run full-resolution smokes. | Complete: bank, manifest audit, and resource smokes passed. |
| E5 | Train registered seeds and evaluate only development roles with the common extension evaluator. | Complete: all 18 training final manifests and the Attempt-E development result are bound and independently verified. |
| E6 | Integrate verified evidence into the paper. | Complete: claim-bounded integration, corrected rendering, 34-page rebuild, and direct figure inspection pass. |

The next active implementation milestone is B5 Stage 0A/B only: add and test the
contract-bound local identity/reducibility path, then stop at the resource gate.

The extension presentation derivative
`naca_corrective_extension_visualization_results_20260903b` closes E6. Its
static panels are exact evaluator snapshots; its three seed animations are
independent qualitative rerolls. The plotting correction changes presentation
only, not the scientific replay or evaluation.

The parent scientific evaluation in order 4 and the derived replay-render path in
order 6 are complete. The immutable replay and historical v2 visualization are
preserved; final presentation uses `development_visualization_v3_e2eccdbf`,
whose 11 outputs reuse the replay byte-for-byte and pin the historical producer.
Static figures use the verified development evaluation and the three seed
animations are qualitative only. The portable source-bound data-free freeze and
then-current 35-page parent paper are complete. Windows checks and exact mounted-checkout WSL Ruff
and 68-test verification pass; exact identities and environment versions are in
`EXPERIMENT_TRACKER.md`. No prospective or sealed reveal is authorized.

## 15. Maintenance And Authorization

Do not add a generic experiment framework, method registry, dashboard, or
configuration hierarchy. Remove task-created helpers with no caller. Preserve
all failed attempts under unique identities. Do not commit raw datasets,
checkpoints, rollout arrays, large logs, credentials, private hosts, or local
machine paths.

This document records the completed NACA R0 lineage, successor, and
open-development extension. The successor's twelve learned packets and the
extension's 18 learned packets plus development result are locally verified.
No new remote or dataset-scale training is opened. If its claim is retained, the identical-bank
recovery--relabel comparison requires a separately frozen evaluation-only
assay. Completion at development scope adds no authority for prospective or
sealed reveal.
