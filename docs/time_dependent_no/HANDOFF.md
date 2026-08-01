# Time-Dependent Neural Operators: Handoff

Updated: 2026-08-01

This is a replaceable operational snapshot. It should describe the current
workspace, the latest supported conclusions, and the next human decisions
without repeating historical run narratives.

## Read Order

1. [README.md](README.md) for navigation and maintained code.
2. [RESEARCH_DIRECTION_DECISION.md](RESEARCH_DIRECTION_DECISION.md) for the
   current scientific state and standing owner constraints.
3. [MECHANISTIC_DIAGNOSTIC_TRACKER.md](MECHANISTIC_DIAGNOSTIC_TRACKER.md) for
   compact experiment-ID routing.
4. One bounded section of the
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
  front error, boundary effects, admissibility loss, and resolution transfer.

It does not support a generally improved shock-stable geometry-aware neural
operator, a learned conservative finite-volume solver, or a state-of-the-art
claim.

| Program | Operational evidence status |
| --- | --- |
| Line 1 | Completed evidence program; larger-step advantage is horizon- and metric-dependent. |
| Line 2 | Completed bounded CPGNet program; paper parity and exact DG replay remain unresolved. |
| Line 3 bump | D041 remains the historical comparator. B1 is a replacement miss; attached-K2 improves B1 recurrence, but no exact same-contract D041 comparison is bound and it is not promoted as a replacement. |
| Line 3 dynamic | D044 is the useful baseline; D060 improves state error without passing the front/high-pass conjunction; D063 gives bounded common-source resolution transfer. |
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
- exact-input commutators remain small but nonzero;
- the training grid is the clear state/front/boundary optimum; and
- node-type interventions show channel use, while the all-normal-trained row is
  confounded by source and loss-population changes.

This is bounded zero-shot transfer, not resolution invariance or proof that
PCNO is an operator.

## Current Workspace Task

The current task is repository documentation and infrastructure cleanup:

- preserve D063 as a separate evidence milestone;
- archive the full historical decision and tracker byte-for-byte;
- keep compact active decision, handoff, experiment index, and README files;
- remove historical authorization language from default context; and
- make future PCNO source snapshots depend on executable/scientific source,
  while recording mutable documentation only as provenance.

No training, checkpoint execution, remote-machine access, dataset mutation, or
sealed evaluation is part of this cleanup.

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

## Next Human Review Topics

These topics are nonbinding and may be selected, reordered, or rejected:

1. whether the projected-teacher K2 control is still the smallest useful
   recurrent-training attribution;
2. whether to build a serious checkpoint around multistep exposure, front-aware
   loss, admissibility information, capacity, or another mechanism;
3. whether D063 motivates a one-factor node-type comparison or
   mixed-resolution training;
4. whether a clean matching-reference horizon extension is worth its solver
   cost; and
5. which frozen evaluation conjunction defines a complete advantage over D041.

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
