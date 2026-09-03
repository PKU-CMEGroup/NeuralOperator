# R0 Primary-PDE Readiness Contract

Updated: 2026-08-31

Status: **Unsteady NACA0012 is frozen as the first R0 screen and conditional
primary case study. Stage 0, native replay/determinism, the 499--1999 solver
trajectory, and the solver-only phase analysis are complete and independently
validated. The analysis returned `PHASE_ONLY_SUPPORTED` in the two gated
views. The 32-anchor population, 208-step horizon, complete-BDF2 PCNO baseline,
metrics, and access rules are now frozen and independently audited. Dataset
materialization, code audit, the model-relative solver ceiling, support
diagnostics, and clean-PCNO outcome remain open.**

## 1. Decision

No surveyed SU2 case is clearly better overall for the first screen.
Unsteady NACA0012 is fixed-geometry, autonomous, solver-backed, and visually
structured. Its separated periodic wake gives us a plausible low-dimensional
clean support and a nontrivial chance of delayed transverse drift.

This is a readiness decision, not a result. NACA0012 becomes the paper's
primary case only if vanilla PCNO has honest small clean one-step error,
initially accurate rollout, and later failure with a measurable transverse
component. We do not tune the data or model to manufacture that phenomenon.

The bounded fallback order is:

1. turbulent square cylinder, only if a production SU2 contract can be made
   reproducible without changing the scientific question;
2. laminar von Karman cylinder as a cheap solver/control case; and
3. shock-vortex or supersonic bump only as tangential/no-harm controls under
   their existing limitations.

Transonic buffet is scientifically attractive but lacks a turnkey pinned SU2
case. Pitching/gust, FSI, and CHT are parked because forcing, moving geometry,
or multiphysics would confound the first demonstration. Inviscid bump is
steady as published.

## 2. What R0 Must Establish

R0 produces no intervention evidence. It asks whether this benchmark can
support the paper's prospective PDE case study. A pass requires:

1. exact public-resource identity and a complete state interface;
2. a native SU2 replay whose numerical discrepancy is safely below the model
   and intervention effects we intend to interpret;
3. an immutable ID trajectory population with disjoint train, development,
   prospective, and sealed/test members;
4. train-only evidence of structured, low-dimensional support;
5. a fixed, adequately trained clean PCNO with small clean one-step error and
   severe delayed autonomous degradation; and
6. evidence that the degradation is not explained by undertraining, a wrong
   recurrence, missing state, solver bias, boundary leakage, or simple phase
   error alone.

If the clean model remains near support and only accumulates tangential or
phase error, NACA0012 is useful as a no-harm control but fails as the hero
corrective-necessity case.

## 3. Frozen Solver And State Contract

The pinned tutorial case is two-dimensional compressible URANS with the
Spalart--Allmaras model on a fixed NACA0012 mesh:

- Mach `0.3`, angle of attack `17 deg`, Reynolds number `1000`;
- second-order dual time stepping with physical `dt = 5e-4`;
- `10` inner iterations in the published tutorial configuration;
- `14,336` quadrilateral elements, `14,576` points, and markers `airfoil` and
  `farfield`; and
- binary restart histories `497` and `498`, with restart `499` as the first
  replay target.

The exact source files and SHA256 values are frozen in
[R0_SU2_NACA_RESOURCE_MANIFEST.json](R0_SU2_NACA_RESOURCE_MANIFEST.json).
The first replay target is SU2 `v8.5.0`, commit
`12eb826f049ef7f67df974dfcb44cf36ee07c0f8`. Compatibility is not inferred
from source inspection; it must be demonstrated by execution.

One binary restart contains coordinates followed by the evolved state

```text
[Density, Momentum_x, Momentum_y, Energy, Nu_Tilde]
```

at every mesh point. Pressure, temperature, Mach, coefficients, viscosities,
and wall quantities in the file are derived or auxiliary outputs, not extra
recurrent state. SU2 associates restart records with mesh coordinate row
order; the optional trailing labels in the ASCII mesh are not the point order.

The canonical target has the frozen 17-field schema. SU2 `v8.5.0` emits an
exact 19-field non-compact replay schema with derived `Velocity_x` and
`Velocity_y` inserted after `Pressure_Coefficient`. Evaluation aligns fields by
their exact names and excludes only those two extra derived fields from the
canonical-target metrics; whole-file hashes still cover every output byte.

Second-order dual-time continuation needs two complete histories. A one-frame
parser, one-frame solver injection, or flow-only state is incomplete. The
learned map may later use a first-order state representation only after the
reference recurrence and any displaced-state labeling interface are defined
without silently dropping required history.

## 4. Stage 0: Resource And Interface Audit

Stage 0 is complete.

- All downloaded public resources match the frozen sizes and SHA256 values.
- Each restart has the exact little-endian native layout
  `(535532, 17, 14576, 0, 0)`, 17 NUL-padded 33-byte field names, and
  point-major float64 values with no trailer.
- All values are finite and all three coordinate arrays match the mesh row
  order exactly.
- The five evolved RANS--SA fields are present in the same order in all three
  states.

The maintained audit is
`scripts/time_dependent_no/audit_su2_naca0012_resources.py`, backed by
`utility/time_dependent_no/su2_restart_contract.py` and focused synthetic
tests. Passing Stage 0 establishes byte identity and interface completeness
only. It does not establish that SU2 accepts the files, that 497/498 reproduce
499, or that the transition is accurate enough for scientific use.

## 5. Stage 1: Native Replay And Throughput

The pinned native replay and repeatability substage is complete. The maintained
runner creates an isolated case atomically, stages only histories 497/498,
keeps canonical target 499 external, binds the official SU2 `v8.5.0`
executable and release identity, records the selected one-rank/one-thread
environment, invokes the executable without a shell, and fails closed on
source, input, output, or authority drift. It does not claim to pin the entire
Windows loader or DLL image. The evaluator and
comparator passed 47 focused tests, Ruff, and three independent code-to-intent
audits before the final pair was executed.

The final cases `su2_naca0012_native_replay_20260831_c` and
`su2_naca0012_native_replay_20260831_d` both completed execution and evaluation.
Their 2,216,199-byte outputs have the same SHA256
`a39f0b4a2462d54817995e6fbb40a9d384af444224f6a8b8439fed99c7747fab`.
The independently verified comparison records matching identities, bitwise
determinism, and zero native-to-native numerical discrepancy. Runtime was
`0.5205 s` and `0.5090 s`, or `1.92` and `1.96` transitions/s locally.

Against canonical target 499, the five evolved fields have component-balanced
relative L2 error `4.3264e-7` under physical vertex-area weighting. That view is
farfield dominated: the fixed near-body/wake box contains 8,392 nodes but only
`1.4052e-4` of physical area. The required uniform-node and near-body/wake views
give `5.9447e-4` and `8.9081e-4`, respectively. Farfield evolved-state errors
are at numerical precision. These three frozen views establish pinned
executable compatibility and a small observed recurrent-state replay
discrepancy without hiding the dynamically active region; they do not yet
establish the complete Stage-1 sufficiency claim.

The execution followed this frozen protocol:

1. preserve and hash canonical target 499 outside the solver directory, then
   stage only mesh plus histories 497/498 in a fresh case directory;
2. derive and hash a replay configuration with `RESTART_ITER=499`,
   `TIME_ITER=500`, `SOLUTION_FILENAME=restart_flow`, the distinct output stem
   `RESTART_FILENAME=replay_flow`, convergence stopping disabled, restart-only
   output, and `WRT_RESTART_COMPACT=NO` so the comparison schema is explicit;
3. run the pinned SU2 executable and produce exactly
   `replay_flow_00499.dat`; refuse any path or filename that can overwrite the
   canonical `restart_flow_00499.dat`, and rehash the canonical target after
   execution;
4. compare all five evolved fields, derived physical fields, a clearly labelled
   surface body-force proxy, and boundary behavior against the pinned 499
   target;
5. repeat the replay and record executable identity, configuration bytes,
   command, runtime, output hashes, and determinism; and
6. measure states/second and projected storage before choosing population size.

The remaining model-relative gate is that replay discrepancy must be smaller
than one quarter of the vanilla model
discrepancy at the same interface. Before intervention comparisons, it must
also be smaller than one quarter of the smallest separation the paper will
interpret. Failure closes this exact NACA route; the discrepancy is never
subtracted post hoc from favorable model results.

That comparison cannot close until the clean PCNO and the smallest interpreted
intervention separation exist. The canonical target contains nodal fields but
no producer-bound history/objective record, and its producer version is
unknown. Original producer-integrated `CD`/`CL` are therefore unavailable by
construction rather than a recoverable pending measurement. The qualified
replay retains its orientation-normalized pressure-plus-skin-friction
functional. Five-state PCNO evaluation instead uses a separately frozen
pressure-only surface functional derived from the conserved state. These
diagnostics are never conflated and neither is labelled canonical
SU2-integrated lift or drag. Candidate-only SU2 `v8.5.0` histories are
byte-identical across C/D and provide a repeatable consistency check.

The auxiliary wall `Heat_Flux` and `Y_Plus` fields show order-one relative
discrepancies despite the small recurrent-state error. They are retained as
postprocessing-sensitive diagnostics and excluded from the recurrent-state
gate; no claim that every derived channel reproduces is allowed. The immutable
C/D receipts keep `restart_sufficiency_claimed=false`,
`boundary_and_force_gate_complete=false`, and
`trusted_transition_claimed=false`. A qualification addendum records the
unavailable producer-integrated target, the common surface functional, and the
still-deferred model-relative margin without rewriting those receipts.

Measured storage is `2,216,199` bytes per native 19-field frame,
`583,040` bytes per five-field float64 state, or `291,520` bytes per five-field
float32 state, plus static coordinates and mesh once. For `T` BDF2 trajectories
with `H` transitions, retain `T(H+2)` distinct frames; do not duplicate three
files per transition. The completed 1,501-frame native trajectory occupies
`3,326,514,699` bytes including its bound case and receipt files; its storage
manifest SHA256 is
`1c0d83027bed64df14f41a620fd4689eda2d341aee54b1e5b9f4d84b71f9d519`.
The population contract fixes train/development-only float64 materialization;
prospective and sealed dynamic values remain unopened until separate reveal
authorization. Actual dataset bytes and hashes do not exist yet.

Exact artifact hashes, field roles, force-history availability, and storage
constants are frozen in
[R0_SU2_NACA_NATIVE_REPLAY_QUALIFICATION.json](R0_SU2_NACA_NATIVE_REPLAY_QUALIFICATION.json).

Arbitrary/noisy-state restart is a separate gate. It is required for dynamic
relabeling and solver-relative response diagnostics, but not for clean PCNO or
purely learned/parameter-free correctors. Any writer must preserve coordinates,
the five evolved fields, both BDF2 histories, field order, native encoding,
and boundary/physical admissibility.

## 6. Stage 2: Population And Clean PCNO

The population is fixed before outcomes are opened. `ID` means membership in
the declared initial-condition law, not low defect or proximity to training
states. Complete rollout windows—not shuffled frames—are assigned to
content-disjoint roles. When all windows lie on one attractor, their dependence
is disclosed: phase anchors are deterministic finite-population quadrature
points, not independent physical trajectories. Only train data fit
normalization, support geometry, tangent estimates, or density models.

The first population hypothesis is deliberately narrow: one fixed autonomous
system at Mach `0.3`, angle of attack `17 deg`, and Reynolds number `1000`, with
ID defined by phase on a qualified post-transient attractor. The required open
solver-only continuation from the pinned BDF2 histories through index `1999`
is complete, with every native state and final-inner history row retained and
hash-bound.

The trajectory contains 1,501 outputs and occupies `3,326,514,699` bytes. Its
solver runtime was `5631.17 s`, or `0.2665` written states/s; the earlier
single-transition replay rate is not used as a dataset-scale throughput
estimate. The trajectory receipt SHA256 is
`3d857d6ddca4dca111ecc716b2c85ff23f727a4fafc97d39f24db340d2b2672b`,
and the ordered restart-record aggregate SHA256 is
`9adbcfd53c4812009d3c5e5081730b709a2f63f48237760390e2c8559ab200ef`.

The unchanged solver-only analyzer returned `PHASE_ONLY_SUPPORTED`. The CL
clock gives period `34.6543086` steps, the selected burn boundary leaves 31
complete cycles, and the maximum normalized template shift and Theil--Sen
drift are `0.0257830` and `0.0475450`. Uniform-node recurrence passes with
median/q90 component-balanced errors `0.0038451/0.0040757`; the near-body/wake
view passes with `0.0038275/0.0040379`. The required physical-area secondary
view fails (`0.1109955/0.1464302`, q90 cycle-to-template `1.2558969`) because
the farfield-weighted orbit amplitudes are extremely small. This failed
secondary view is disclosed and cannot independently qualify or overturn the
two frozen gated views. The analysis file SHA256 is
`b14ef6d66c5e520633ea94954ff3b243a47c3185743de43a5ff10d1c7f74069f`.

The first sandboxed analysis attempt failed before reading the packet because
its temporary directory was inaccessible. Its `INVALID_ARTIFACT` receipt is
preserved as infrastructure evidence; the same source and packet then ran
unchanged outside that sandbox. It is not counted as a scientific attempt.

The audited baseline contract fixes a global phase offset and 32 stratified
phase anchors, eight per train, development, prospective, and sealed role. It
uses content-disjoint 240-frame role blocks and a 208-step horizon, approximately
six shedding periods. The anchors are deterministic phase quadrature on one
attractor, not independent trajectories. Stage-2 materialization may read only
train and development states; prospective and sealed values remain unopened.
The claim is single-system, single-attractor ID closed-loop prediction, not
generalization across parameters or physical realizations.

If meaningful cycle-to-cycle modulation remains, reject the phase-only law and
represent that modulation explicitly before population freeze. Perturbed-state
restarts are excluded from the clean population; they belong to the separate
arbitrary-state response/relabeling gate. A parameter-conditioned population
would change the scientific problem and requires a new owner-approved contract.

The exact solver-only signals, period/stationarity/recurrence thresholds,
extension rule, resource cap, outcomes, and prospective phase-anchor rule were
frozen before scientific analysis in
[R0_NACA_PHASE_PILOT_CONTRACT.json](R0_NACA_PHASE_PILOT_CONTRACT.json), current
SHA256 `f442065172beb014ebcd4f3823abe1e2764fb2b74bbad348a8dd94d6b4c903f5`.
The phase result supported the subsequent population freeze; it did not itself
create or claim that population.

The population, representation, training, evaluation, and R0 rules are frozen
in [R0_NACA_PCNO_BASELINE_CONTRACT.json](R0_NACA_PCNO_BASELINE_CONTRACT.json),
SHA256 `94069e65ef520d31860735b6c17f2b7515da1c4acd34d68518a2a16b3181fa87`.
Independent audit recomputed every anchor, crossing, phase error, BDF2 window,
transition count, horizon ratio, and channel count and closed the metric,
access, replay-margin, measure, precision, and failure-accounting semantics.
This freezes the experiment before any PCNO outcome; it is not a dataset or
model-result receipt.

The baseline uses the full 2D PCNO with one common architecture and resource
envelope across all later information arms. It predicts a residual in the five
evolved fields on the fixed mesh. The clean recurrence has no noise, exposure,
multistep loss, hidden projection, clipping, flooring, smoothing, oracle
boundary replacement, or future truth.

Checkpoint selection uses dense development one-step normalized-residual error
only. Every seed must also meet the frozen next-state one-step gates. Initial
accuracy is the per-rollout RMS error over steps 1--35, with both an absolute
threshold and at least 20% improvement over persistence. Severe degradation is
tested over steps 174--208 rather than at the nearly period-aliased terminal
endpoint; failed windows count as infinite, and full-anchor completion is
reported separately from the finite state-time fraction. Exact equations,
floors, aggregation order, three seeds, and thresholds are in the JSON
contract above.

Train-only reconstruction curves, effective dimension, neighborhood
stability, and physical features must independently show structured support.
Model defect never defines the data manifold or ID/OOD status.

## 7. Stage 3: Diagnosis And Promotion

If Stages 1 and 2 pass, freeze the B3 query bank and signed predictions before
opening designated rollout outcomes. Diagnostics separate:

- on-reference/tangential defect;
- transverse forcing and response;
- support drift measured without solver defect;
- solver-relative off-state defect as an offline diagnostic only;
- conservation, positivity, turbulence-state validity, and boundary leakage;
  and
- phase error versus genuine transverse departure.

NACA0012 is promoted to the primary PDE only if these measurements support a
nontrivial two-axis story: good on-reference evolution is present, but rollout
also needs retention or recovery. A stable wrong trajectory, a tube-retained
phase error, and an accurate but unstable trajectory are reported separately.

## 8. Current Authorization Boundary

Authorized now:

- the frozen public-resource manifest;
- the Stage-0 parser and audit entry point;
- the source-bound replay preparation, native runner, evaluator, comparator,
  shared contract utilities, multi-step trajectory runner, and solver-only
  phase analyzer;
- the verified claim that the pinned 497/498-to-499 recurrent-state replay is
  successful, small-error, and bitwise deterministic under the recorded local
  execution identity;
- the verified 499--1999 trajectory packet and the bounded
  `PHASE_ONLY_SUPPORTED` result, including the failed physical-area secondary
  view;
- the independently audited 32-anchor/208-step PCNO baseline contract at the
  hash above; and
- focused CPU tests and independent code-to-intent review for these paths.

Not yet claimed or opened:

- complete model-relative replay sufficiency, unavailable original
  producer-integrated force agreement, or trusted arbitrary/noisy-state
  transitions;
- a materialized PCNO dataset or opened prospective/sealed state value;
- PCNO training or baseline rollout outcomes;
- B3 response queries; or
- B4 intervention comparisons and prospective/sealed reveals.

Each later stage begins only after the preceding evidence and code-to-intent
audit pass. The next milestone is to finish and independently audit the
train/development-only materializer, trainer, and evaluator, then run a bounded
full-resolution resource smoke. The model-relative replay ceiling remains a
paired post-training readiness check. The unavailable original
producer-integrated coefficients, distinct replay and five-state surface
functionals, auxiliary-field scope, and measured storage are preserved; none
is silently converted into a stronger claim.
