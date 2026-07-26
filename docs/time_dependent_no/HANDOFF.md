# Handoff

Updated: 2026-07-26

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

The active queue is report-only. No learned method, oracle, checkpoint run,
sealed-split read, or data-assimilation experiment is automatically next.

## Four-Line Status

| Line | Operational status | Frozen conclusion |
| --- | --- | --- |
| 1: large-step flow maps | Closed | On the frozen 1D Euler study, direct larger-step maps trade harder one-call approximation against fewer recurrent compositions. The preferred stride changes with horizon and metric. This is not a universal stride, CFL, timestep-transfer, or resolution-transfer result. |
| 2: CPGNet validity and mechanism | Closed | Corrected 1D controls support message reach rather than width alone as the main gain. The 2D release-bundle legal-boundary run improves all four primitive variables on 19/20 trajectories, but remains roughly `1.6--2.4x` worse than the oracle-boundary row. Dataset/checkpoint/evaluator parity with the paper remains unresolved, and the result is one seed without a validation or grouped geometry holdout. |
| 3: geometry-aware 2D rollout | Closed without promotion | D044 completes 24/24 raw H60 position-OOD validation rollouts on the frozen Mach-1.1 family at mean physical-volume state error `0.00834190`. D060 also completes 24/24 and lowers that error to `0.87434x`, but its six-case high-pass RMS is `1.0255x` D044 and its front-centroid distance is `1.4351x`. The joint stabilization claim fails. |
| 4: latent forecasting and assimilation | Stopped before forecast training | Smooth-decoder and fixed-Haar capacity tests do not pass the reconstruction/front hierarchy, even though discontinuous regularity helps. No latent transition, recurrent forecast, geometry-transfer, test, or filtering result exists. |

## Authorized Report-Only Work

Using frozen artifacts only, the report may:

- compare D044 and D060 state, high-pass, front, and geometry-stratum curves;
- show D053 propagated-versus-fresh shares beside the D052 branch
  interventions;
- place D048/D049 discrete-decoder amplification beside D062 front-identity
  failure;
- tabulate objective, sample presentations, selected epoch, intervention,
  population, evidence grade, and non-claim; and
- preserve failed implementation attempts only as provenance.

No report task may execute a checkpoint, alter a threshold, consume GPU time,
open a sealed split, or silently become a method experiment.

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
  especially D031-D039, D014 and the legal-boundary closeout, D037, D044,
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
- The maintained code surface and the ADER-generator invocation warning are
  listed only in [`README.md`](README.md).
- Machine-specific paths and credentials remain in ignored local context;
  generated research outputs remain ignored under `artifacts/time_dependent_no/`.

## Required Human Direction

There is no automatic experiment left in the queue. The next human decision is
one of:

1. close the campaign around the bounded mechanism results and produce the
   frozen-artifact report; or
2. separately preregister one zero-training capacity oracle for a genuinely new
   discontinuous representation with connected front identity and transverse
   regularity.

The second option is not current authorization. Such an oracle must define its
identity and closure contract and pass reconstruction and closure before any
learned encoder, latent transition, test access, or filter is considered.
