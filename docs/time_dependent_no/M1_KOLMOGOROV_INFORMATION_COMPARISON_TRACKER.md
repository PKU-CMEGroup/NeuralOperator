# M1 Kolmogorov Information-Source Comparison Tracker

Date: 2026-08-26

This is the compact execution surface for
[the M1 preregistration](M1_KOLMOGOROV_INFORMATION_COMPARISON_PREREGISTRATION.md).
The preregistration is scientific authority; this file records status without
redefining its gates.

| ID | Milestone | Purpose | Scope | Priority | Status | Continuation requirement |
| --- | --- | --- | --- | --- | --- | --- |
| M1-Q0-SRC | A1 | reusable fixed-grid reference map | synthetic only | MUST | COMPLETE LOCALLY | focused commit and source hash |
| M1-Q0-TEST | A1 | analytic, projection, restart, and fail-closure tests | 12 CPU tests | MUST | PASS | retain exact command/result |
| M1-Q0-CLI | A1 | dry-run invocation and JSON accounting | synthetic random and laminar states | MUST | PASS | retain exact command/result |
| M1-Q1-NUM | qualification | time refinement, process repeatability, spatial context | solver-only calibration | MUST | COMPLETE: TEMPORAL/REPEAT PASS; SPATIAL PENDING | exact attempt closed; do not infer PDE-level spatial qualification |
| M1-Q1-STAT | qualification | burn-in and stationarity | four deterministic generated calibration states | MUST | FAIL AT REGISTERED 512-CALL CONTINGENCY | stop current ladder; new owner-approved qualification design only |
| M1-Q1-R1-MIX | diagnosis | mixing-window uncertainty and fresh-seed burn-in design | four parent calibration seeds; scalar/spectral series only | MUST | AUTHORIZED; SOURCE PREPARATION | immutable R1 source plus parent-hash replay |
| M1-Q1-R1-SPAT | diagnosis | adjacent N64/N128/N256 spatial context | six hash-matched parent inputs | MUST | AUTHORIZED; SOURCE PREPARATION | same R1 source; N128 screen is provisional only |
| M1-Q2-CLEAN | baseline | clean FNO recipe and phenomenon gate | seed 0, open validation only | MUST | NOT AUTHORIZED | a new Q1 pass and A3 contract; current failed attempt is insufficient |
| M1-B0-BANK | bank freeze | paired recovery/dynamics inputs and targets | open train/development only | MUST | NOT AUTHORIZED | Q2 pass and immutable manifests |
| M1-I0 | information screen | four arms at seed 0 | open validation only | MUST | NOT AUTHORIZED | B0 replay and source gates |
| M1-I1 | replication | remaining two seeds for four arms | open validation only | MUST | NOT AUTHORIZED | I0 correctness review and explicit continuation |
| M1-E0 | open closeout | common-step/selected response, rollout, structure, and cost | development/open evaluation | MUST | NOT AUTHORIZED | frozen evaluator and named A2 approval |
| M1-T0 | sealed confirmation | one-shot 32-trajectory test | sealed test | CONDITIONAL | NOT AUTHORIZED | full prereg closeout and A4 approval |
| M2 | selective correction | cached informative solver labels | future method study | CONDITIONAL | BLOCKED ON M1 | M1 dynamics-label benefit and no-harm pass |

## A1 Verification Record

Commands are run from the repository root with the configured local Python:

```text
python -m pytest -q -p no:cacheprovider tests/time_dependent_no/test_kolmogorov_reference.py
python -m scripts.time_dependent_no.run_m1_kolmogorov_a1_preflight
```

Observed A1 results:

- focused tests: `12 passed`;
- same-process synthetic replay: bitwise equal;
- single-step/repeated-rollout closure: exact;
- laminar relative L2 closure: approximately `1.82e-16`;
- final canonical projection change: approximately `4.02e-16`;
- no dataset, checkpoint, remote process, training run, or sealed population was
  accessed.

Exact source commit and file hashes are recorded at the focused A1 closeout;
any later source change requires a new source snapshot and tracker amendment.

## M1-KF-Q1-20260826A Closeout

The registered local solver-only qualification completed on 2026-08-26 with
classification `failed_fixed_grid_qualification`. This is a scientific gate
failure, not an infrastructure failure.

### Provenance and integrity

- immutable execution source: `d396c45f2acd0bbea971b3285b98f406bcea4e74`;
- ignored local packet:
  `artifacts/time_dependent_no/m1_kolmogorov_q1_20260826a/`;
- `result.json` SHA256:
  `9ffc7a23c62704dad04af03996336ebff793d1cb3d8fb25da6f6f01176018002`;
- `artifact_manifest.json` SHA256:
  `d1ef9b4e77cfe9866ab4beee82129972b2f1f6cd6ae3ee6ca8ba93a9653317ae`;
- both start and end bindings saw the exact clean source commit, every
  registered source hash reverified, and the result hash matches the manifest;
- runtime: `2130.47 s`; and
- no dataset, checkpoint, model training, remote execution, or sealed
  population was accessed.

### Raw gate record

At the initial 256-call burn-in, only seed `2026082604` passed both registered
10% half-window mean-change limits. The sole registered contingency extended
burn-in to 512 calls. The final observation window was:

| Seed | Energy half-change | Enstrophy half-change | Energy Spearman | Enstrophy Spearman | Half-change gate |
| ---: | ---: | ---: | ---: | ---: | --- |
| 2026082601 | 0.00742 | 0.06584 | -0.07356 | -0.26712 | pass |
| 2026082602 | 0.14061 | 0.19079 | -0.93528 | -0.92061 | **fail** |
| 2026082603 | 0.01919 | 0.02741 | 0.28776 | 0.33163 | pass |
| 2026082604 | 0.03405 | 0.02388 | -0.10532 | 0.06993 | pass |

The shared-drift gate passed: only one of four trajectories had a registered
same-sign monotone event. The stationarity gate nevertheless fails because it
requires every trajectory to satisfy both half-window limits.

The candidate `dt_max=0.002` versus `0.0005` results all pass:

| Family | H1 median / maximum | H16 median / maximum |
| --- | ---: | ---: |
| clean | 5.639e-8 / 7.779e-8 | 5.255e-7 / 7.299e-7 |
| displaced | 5.877e-8 / 7.938e-8 | 5.327e-7 / 7.473e-7 |

Maximum H64 distribution discrepancies were `8.847e-9` for kinetic energy,
`1.304e-8` for enstrophy, and `3.526e-8` for the normalized shell spectrum.
Maximum canonical projection change was `3.296e-16`. Two spawned FP64
processes were bitwise equal, had scaled RMS difference `0.0`, and agreed on
all 57 accepted substeps.

Spatial context remains unresolved rather than passed. Restricting the N=128
fine-step evolution back to N=64 gave overall median/maximum discrepancies of
`0.09786/0.11758` at H1 and `0.27965/0.31487` at H16. Clean and displaced
families were nearly identical, so this is a systematic resolution effect, not
an effect of the registered 3% perturbations. Its model-relative factor-of-four
gate was not evaluable because Q2 and B0 never opened.

### Bounded interpretation and decision

The temporal integrator is qualified for the declared finite-dimensional
`Phi_64`, and deterministic implementation error is not the observed
bottleneck. The registered population did **not** establish stationarity. One
remaining drifting window cannot distinguish a persistent transient from a
large correlated fluctuation, so the result does not prove that the underlying
forced system is nonstationary.

The much larger N=64 versus N=128 discrepancy independently warns that
`Phi_64` must not be presented as a spatially converged PDE reference. It may
only be treated as its declared fixed-grid dynamical system unless a later
resolution study closes the PDE-level gap.

Per preregistration, this exact Q1 attempt is closed and the current M1 ladder
stops before Q2. Extending burn-in again, changing seeds or thresholds, or
training a clean model would be post-hoc continuation and is not authorized by
this result. The minimum defensible next stage, if the owner chooses to retain
M1, is a newly preregistered solver-only reference/population diagnosis that
separates mixing-window uncertainty from spatial-resolution error before any
model data are generated.

### Result-to-claim routing

Internal verdict, pending any separately approved external Codex review:

- `claim_supported: no` for M1-C1/M1-C2 on the current evidence, because the
  information arms and learned rollout comparison were never opened; this is
  absence of qualifying evidence, not a negative test of dynamics relabeling;
- supported: the declared FP64 temporal integration and implementation replay
  are sufficiently accurate for the fixed-grid `Phi_64` map on the registered
  calibration inputs;
- not supported: stationary-population qualification, spatially converged PDE
  reference status, phenomenon qualification, or any learned-model claim;
- missing evidence: a defensible stationary sampling law and a spatial
  resolution study whose candidate grid closes before model training; and
- route: reference/population supplement or testbed pivot, not architecture
  ablation, training, or paper-claim promotion.

Confidence is high in the stop decision and moderate in the interpretation of
the single drifting trajectory, because the packet retained registered summary
statistics but not the underlying per-call diagnostic series.
