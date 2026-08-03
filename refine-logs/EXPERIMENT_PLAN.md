# D072 Boundary-Field Experiment Plan

Status: preregistered and executing on open populations; the dynamic core has
finished training without aggregate interpretation, and the bump core remains
active. This file is subordinate to
`docs/time_dependent_no/RESEARCH_DIRECTION_DECISION.md` and
`docs/time_dependent_no/HANDOFF.md`.

Date frozen: 2026-08-03

## Question and claim boundary

D072 asks whether bounded, fixed-physical-width boundary fields give recurrent
PCNOs a more resolution-consistent and dynamically useful representation than
omitting boundary information. It does not change or improve the physical
boundary condition, and it does not claim an optimal encoding or a general
geometry solver.

The two preregistered claims are:

- C1, primary: under a fixed family-local physical boundary policy, recurrence,
  optimizer, data stream, and exact matched initialization, G1 or S1 improves
  open-validation autonomous Euler rollouts over N0 without worsening
  admissibility or anti-smearing controls.
- C2, supporting: S1 minus G1 identifies incremental semantic value, and any
  useful effect is consistent with a nonvanishing physical collar signal whose
  influence strengthens near boundary interaction. Dynamic resolution evidence
  and bump unseen-geometry evidence are reported separately.

Nonclaims include optimality, boundary-condition enforcement, physical
conservation by architecture, broad geometric generalization, rotation
equivariance, and bump PDE resolution transfer.

## Frozen representation contract

For each semantic subset `Gamma_k`,

    B_k^ell(x) = rho(distance(x, Gamma_k) / ell),
    rho(r) = 1 - 3 r^2 + 2 r^3 for 0 <= r < 1,
    rho(r) = 0 otherwise.

`ell` is in source physical coordinates and stays fixed across discretizations.
The channel is bounded in `[0, 1]`; there is no `1/ell` surface-measure
scaling.

| Arm | Model input after the four state-independent geometry slots | Input width | Role |
| --- | --- | ---: | --- |
| N0 | coordinates (2), quadrature density, normalized state (4), normalized Mach | 8 | no-boundary reference |
| G1 | N0 plus `max_k B_k^ell` before Mach | 9 | boundary-location control |
| S1 dynamic | N0 plus y-symmetry and x-extrapolation collars before Mach | 10 | factorized semantic fields |
| S1 bump | N0 plus wall, outflow, and inflow collars before Mach | 11 | semantic fields on native graphs |

Categorical type channels are omitted from all three arms. Dynamic contact code 3
belongs to both independent descriptors. Bump mixed endpoint edges belong to
both adjoining semantic subsets. Family-local node-type meanings are never
pooled.

Within a seed, each G1/S1 process first constructs the identical N0 model, copies
every non-lifting parameter and every active lifting column, inserts exactly zero
field columns, and restores the post-N0 CPU RNG state. The preactivation identity
gate checks copied tensors, zero columns, RNG state, post-lift features, selected
hidden states, and the model output in FP32; output identity alone is
insufficient because the residual decoder starts at zero.

The primary width rule is outcome independent:

    ell_family = 0.05 * median_train_geometry min(x_span, y_span).

Dynamic FV is frozen at `ell = 0.05`. Bump `ell` remains blocked until the
training-only geometry manifest described below is retrieved. Validation states,
targets, model outputs, and sealed populations may not enter the width.

## Why this kernel, and bounded alternatives

The compact cubic is the primary choice because it has unit boundary amplitude,
exact physical support, zero slope at both support endpoints, and one-sided mass
`integral_0^1 rho(r) dr = 1/2`. The zero boundary slope also avoids coupling the
nondifferentiability of unsigned distance at the boundary directly into the
differential branch.

The only first-order kernel alternative is the compact linear collar
`rho_L(r) = max(1-r, 0)`. It matches support, boundary amplitude, and one-sided
mass `1/2`, but is only C0 and has a derivative jump at the support edge. It is
therefore a clean conditional smoothness control rather than a search over
geometry methods.

| Alternative | Decision in this campaign | Reason |
| --- | --- | --- |
| compact linear, same `ell` | conditional one-seed control after the core matrix | isolates kernel smoothness with amplitude/support/mass matched |
| cubic at `ell/2` and `2 ell` | conditional sensitivity, never model selection | checks width robustness without claiming an optimum |
| top-hat band | omit | discontinuity and doubled one-sided mass confound the comparison |
| exponential or global SDF | omit | changes locality, support, and total volume mass |
| diffuse surface measure | omit | a different codimension-scaled mathematical object |
| normals, characteristics, prescribed state/flux | defer | add distinct physical information, not merely a minimal semantic extension |
| harmonic or learned extension | defer | introduces a new solver or learned subsystem and a larger causal surface |

The linear and width controls are triggered only after N0/G1/S1 are complete.
They cannot replace the primary cubic result. A one-seed difference of at least
5% in the family primary metric, or a change in completion/front controls,
triggers the remaining two matched seeds; otherwise it remains a sensitivity
check.

## Experiment matrix and run order

### P0: provenance and real-data field gate

- Bind an exact source snapshot, source-family manifest, open split, normalizer,
  boundary policy, recurrence, and generated collar manifest before training.
- Dynamic generation uses one augmented shard tree shared by N0/G1/S1 and the
  explicit population contract `--splits train validation`. Test arrays are not
  opened, decoded, copied, rehashed, or evaluated.
- Verify field range, channel order, contract digest, rigid-motion covariance on
  synthetic geometry, fixed-`ell` mass convergence, and exact G1 equals the
  pointwise maximum of S1.
- Run one real train shard and one real open-validation shard through N0/G1/S1.
  Require exact active-array digests across arms and the matched-initialization
  gate before an optimizer step.

### P1: dynamic-FV core

Frozen numerical contract follows the completed D069 control:

- seeds `20260718`, `20260719`, `20260720`;
- manifest split: 84 train and 24 open validation; 27 test remain sealed;
- step stride 2, 50 epochs, 1,024 presentations/epoch, 256 validation
  presentations, batch size 4, 13,400 optimizer steps;
- `k_max=8`, domain lengths `(2, 1)`, five width-128 blocks, FC width 128;
- AdamW optimizer, learning rate
  `1e-3`, weight decay `1e-5`, constant schedule, clipping 1.0, input noise
  0.003, BF16;
- raw self-recurrence, `model_all_nodes`, no boundary auxiliary, historical
  all-node checkpoint selection, all 24 open-validation trajectories, H30.

Arm order is rotated while each seed block remains on one unchanged
machine/software/data deployment:

- seed 20260718: N0, G1, S1;
- seed 20260719: G1, S1, N0;
- seed 20260720: S1, N0, G1.

A one-epoch, 64-presentation CUDA smoke for all arms precedes the serious runs
and is excluded from scientific comparisons.

### P2: bump core

Execution begins only after the bump geometry/provenance gate. The fixed physical
policy is `causal_nodal_physical` for every arm; this is a frozen evaluator and
recurrence policy, not an encoding change.

The optimization contract is the strong B1 lineage with evaluation cadence
reduced uniformly to avoid making validation rollout cost the training budget:

- same three seeds and rotated arm orders as P1;
- one fixed 270-train/30-open-validation split with disjoint geometry digests;
- stride 1, 40 full-coverage epochs, 853,200 presentations and 216,000 requested
  optimizer steps, batch size 4;
- `k_max=8`, domain lengths `(6, 2)`, five width-128 blocks, FC width 128;
- learning rate `1e-3`, weight decay `1e-5`, warmup-cosine schedule,
  warmup fraction 0.02, floor `2e-5`, clipping 1.0, no input noise, BF16;
- raw self-recurrence, no boundary auxiliary, H20/H40/H60/H79, all 30 open
  validation geometries, rollout selection every five epochs.

The first seed is an infrastructure and numerical-health stage, not a result
selection stage. Seeds two and three proceed automatically unless a registered
stop condition fires.

### P3: conditional design controls

After each family core is frozen:

- S1 with compact linear `rho_L` at the primary `ell`;
- S1 cubic at `ell/2`;
- S1 cubic at `2 ell`.

Run seed 20260718 first. Expand a control to all three seeds only under the
predeclared 5%/completion/front trigger. No width or kernel is selected for the
primary result after seeing these outcomes.

## Metrics and decision rules

All metrics are case-level first, then aggregated within family. Seeds are the
training replicate unit. Paired trajectory bootstrap intervals are supporting,
not substitutes for seed replication.

Required metrics:

- teacher-forced one-step relative state and update error;
- free-rollout state error at every call and declared horizons;
- completion, mean survival, density/pressure minima, first invalid proposal;
- boundary, fixed-physical-distance near-boundary, shock, vortex, and smooth
  region errors using truth-defined masks;
- shock position, strength, and thickness; vortex amplitude/circulation proxies;
- fixed physical-wavenumber bands and smooth high-pass error, always paired with
  anti-smearing front controls;
- dynamic physical-volume state-total mismatch and boundary leakage, without
  calling the model conservative;
- runtime, memory, parameter count, source/data/normalizer/presentation digests.

Dynamic primary success for an arm requires 24/24 H30 completion for every seed,
at least two of three within-seed H30 wins over N0, a three-seed mean H30 error
ratio at most 0.90, and no required front/strength/thickness metric ratio above
1.05. G1 versus N0 measures geometry-location value; S1 versus G1 measures
semantic value, with a supporting semantic threshold of ratio at most 0.95 and
at least two seed wins.

Bump comparison is lexicographic: H79 completion fraction, mean survival, then
H79 error on the common admissible population. A positive arm must improve
completion, or tie within five percentage points and achieve an error ratio at
most 0.90, with at least two seed wins and no front-control ratio above 1.05.
Errors on different accepted prefixes are not compared as if they shared H79.

A mixed outcome includes G1 improvement without S1 incremental value, a
family-specific benefit, better one-step error without rollout benefit, or state
gain with failed front/admissibility controls. A negative outcome is no
consistent open-validation benefit after all registered seeds; it does not prove
that every boundary extension is ineffective.

## Resolution, geometry, and rotation stress tests

Dynamic resolution evaluation uses only retained, provenance-complete
common-source pairs with evolved targets and conservative restriction to the
declared comparison mesh. Teacher-forced commutators and autonomous rollouts are
separate. A finite-resolution support result requires a commutator/error ratio
no worse than 1.05 on every eligible pair and at most 0.90 on at least one pair.
No missing 500x200 evolved target is inferred or generated after outcomes.

Bump uses native open-validation graphs only. The verified split has 270 unique
training and 30 unique validation geometry digests with zero overlap. This is an
unseen-native-geometry test, not resolution transfer. Node dropping remains
query-mesh resampling.

Rigid-motion tests first verify the scalar collar construction itself under
transformed geometry. A frozen-checkpoint model test must rotate coordinates and
momentum vectors, rotate the physical boundary semantics, transform predictions
back, and report equivariance error. It is diagnostic because the underlying
PCNO Fourier basis is not asserted equivariant.

## Frozen interventions and instrumentation

For selected N0/G1/S1 checkpoints:

- baseline fields versus all-zero fields;
- one semantic field zeroed at a time where physically meaningful;
- G1 union field versus zero;
- teacher-forced one step and autonomous rollout;
- post-lift `W_B B`, per-block hidden differences, and pointwise,
  Fourier/integral, and differential branch outputs;
- optional JVP norms, labeled sensitivity diagnostics rather than causal proof.

No activation evidence is used unless no-hook and hooked inference agree. The
maximum tolerances are dynamic `2e-5` absolute and `1e-7` relative, and bump
`2e-3` absolute and `1e-5` relative. A failure stops activation
interpretation rather than relaxing the tolerance.

## Visualization contract

Every comparable rollout frame is included. Bump movies stop at the first
inadmissible proposal and never fabricate later states. Each selected case has:

- truth/baseline state, zero-field state, and their difference;
- truth, baseline, and intervened denormalized increments and both errors;
- boundary-distance and truth-defined shock/vortex/smooth overlays;
- post-lift and selected hidden-feature sensitivity maps;
- boundary-interaction timing and fixed physical-frequency summaries.

Case, geometry/resolution, physical time, recurrence, intervention, units, and
visualization-only subsampling are in the manifest. Compared panels use common
per-field scales fixed from truth/training statistics before model outcomes.

## Stop conditions

Stop the affected stage if:

- source, split, normalizer, boundary-policy, or source-snapshot provenance is
  incomplete or mismatched;
- exact matched initialization, RNG, or within-seed presentation-stream identity
  fails;
- any operation would access a sealed population;
- bump training-only geometry spans or tagged boundary provenance are missing;
- hooks perturb outputs beyond tolerance;
- nonfinite training or inadmissibility requires changing the frozen contract;
- a machine/software/data deployment changes within a seed block;
- bump evidence cannot be separated from query-mesh resampling.

## Compute estimate and ownership

The dynamic core is nine runs and approximately 14--16 GPU-hours from the
verified D069 timing. Bump requests nine runs at 216,000 steps each; wall time is
re-estimated from the mandatory CUDA smoke because historical rollout overhead
made the old wall-clock estimate unreliable. Seed blocks may run in parallel on
different machines, but arms within one seed block remain sequential on one
unchanged deployment.

Long-lived source ownership is limited to:

- boundary construction: `utility/time_dependent_no/pcno_boundary_fields.py`;
- open-only retained-shard publication:
  `scripts/time_dependent_no/prepare_pcno_shock_vortex_shards.py` and
  `scripts/time_dependent_no/augment_pcno_dynamic_boundary_fields.py`;
- bump training-geometry audit and retained-shard publication:
  `scripts/time_dependent_no/audit_pcno_bump_training_geometry.py`,
  `scripts/time_dependent_no/prepare_pcno_euler2d_shards.py`, and
  `scripts/time_dependent_no/augment_pcno_bump_boundary_fields.py`;
- model/runtime/training contract:
  `utility/time_dependent_no/pcno_euler2d.py`,
  `utility/time_dependent_no/pcno_runtime.py`, and
  `scripts/time_dependent_no/train_pcno_euler2d_residual.py`;
- focused tests: `tests/time_dependent_no/test_pcno_boundary_fields.py` plus
  existing runtime/artifact tests;
- evaluators and all-frame visualization reuse the maintained D063/D068
  resolution and intervention surfaces. New code is added only when those
  surfaces cannot express a preregistered metric.

Large shards, checkpoints, logs, rollouts, and media remain ignored artifacts.
No core PCNO API promotion, new boundary policy, or generic geometry framework is
owned by D072.
