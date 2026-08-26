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
| M1-Q1-NUM | qualification | time refinement, process repeatability, spatial context | solver-only calibration | MUST | NOT AUTHORIZED | named solver-execution approval |
| M1-Q1-STAT | qualification | burn-in and stationarity | open generated calibration states | MUST | NOT AUTHORIZED | Q1-NUM pass plus data-generation approval |
| M1-Q2-CLEAN | baseline | clean FNO recipe and phenomenon gate | seed 0, open validation only | MUST | NOT AUTHORIZED | Q1 pass and A3 contract |
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
