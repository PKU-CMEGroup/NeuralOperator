# Implementation Plan: Corrective-Mechanism Experiments

Updated: 2026-08-29

Status: active implementation design. No code change or scientific execution
is authorized by this document.

## 1. Design Principles

- Reuse existing branch-local utilities before adding abstractions.
- Keep reusable code under `utility/time_dependent_no/`, entry points under
  `scripts/time_dependent_no/`, and tests under `tests/time_dependent_no/`.
- Keep the PDE backbone fixed to PCNO; do not modify core `pcno/` APIs unless a
  later reviewed implementation cannot be expressed branch-locally.
- Separate exact reference dynamics, training targets, deployed transition,
  diagnostics, evaluation, and analysis.
- Select and qualify the mandatory primary PDE before substantial ODE or model
  implementation. ODEs are supporting laboratories, not the application.
- Support the small representative set of embedded, training-time, explicit,
  and hybrid interventions selected in the experiment contract; do not build a
  method zoo in advance.
- Use one shared query-bank format and one shared result schema across ODE and
  PDE experiments where meanings truly coincide.
- Do not force exact ODE geometry and PDE proxy geometry into one misleading
  abstraction.
- Add the minimum long-lived files needed for the active blocks.

## 2. Existing Surfaces To Reuse

| Existing path | Reuse role | Boundary |
| --- | --- | --- |
| `utility/time_dependent_no/kolmogorov_reference.py` | Trusted periodic Kolmogorov reference stepping and refinement support. | Existing finite-grid reference evidence does not qualify the population or PCNO model stage. |
| `scripts/time_dependent_no/run_m1_kolmogorov_q1_r2_diagnostic.py` | Reference/parent packet verification patterns and current population failure provenance. | Do not rerun or modify under the old R2-POP identity. |
| `utility/time_dependent_no/path_conditioned_tube.py` | Algebra, finite-amplitude secants, and supporting path-conditioned diagnostics. | It compares against archived clean targets and is not trusted `Phi(x)` response. |
| `utility/time_dependent_no/pcno_artifacts.py` | Source snapshot, manifest, and closeout patterns. | Preserve schema-specific historical meaning; a new study gets a new source identity. |
| Existing PCNO rollout/state/metric utilities | Backbone execution, denormalization, recurrence, and structure metrics. | Bind exact family/state semantics before reuse. |
| Existing M1, D094, and correction tests | Examples of fail-closed contracts and synthetic fixtures. | Historical fixed hashes are not updated in place. |

The repository already contains many historical correction scripts. Presence
does not make them active. The new implementation must not wrap the whole
historical zoo behind a generic dispatcher.

## 3. Minimal New Code Surface

Proposed files are created only after owner review of this plan.

### 3.1 Shared diagnostics

`utility/time_dependent_no/corrective_response.py`

Responsibilities:

- typed records for clean forcing, finite-amplitude response, and optional
  tangent/normal blocks;
- exact two-channel recurrence and ranking score;
- recovery versus dynamics-relabeling target construction from an already bound
  query bank;
- model-independent kNN and local-reconstruction scores on a supplied frozen
  feature representation;
- tube/proxy exit and survival summaries; and
- JSON-serializable summary records.

Non-responsibilities:

- no model training;
- no solver launch;
- no per-method latent feature extraction;
- no online OOD decision; and
- no global “manifold estimator” class.

Create this module only if both ODE and PDE code have real callers. Until then,
keep exact ODE helpers in the ODE runner to avoid a speculative abstraction.

### 3.2 Operational correctors

`utility/time_dependent_no/operational_correctors.py`

Use one small interface:

```text
correct(predicted_state, allowed_context, rng) -> corrected_state, metadata
```

Implement only the representative deterministic and learned correctors chosen
in the reviewed experiment contract. The module must also provide identity
behavior for parity tests and a single composition path for the raw predictor
and deployed corrected map. It records the raw prediction, corrected state,
correction displacement, allowed conditioning, and method-specific cost.

Do not create a generic diffusion library, search framework, or method
registry. Add a separate training entry point only if the selected learned
corrector requires it. The hybrid uses the same composition path: a
dynamics-relabelled predictor followed by a separately auditable recovery
corrector.

`tests/time_dependent_no/test_operational_correctors.py`

Required tests:

- identity corrector exactly recovers the raw predictor;
- the corrected state, not the raw prediction, is fed into the next step;
- raw and deployed outputs and metrics remain separate;
- correction order, frequency, and allowed context match the contract;
- projection/reference banks use only the registered training population;
- stochastic correctors are reproducible under the registered seed; and
- predictor and corrector identities fail closed on a manifest mismatch.

### 3.3 ODE runner

`scripts/time_dependent_no/run_corrective_ode_study.py`

Subcommands:

- `exact`: evaluate analytical scenarios and recurrence identities;
- `train`: train one matched residual learner for one registered arm/seed;
- `evaluate`: evaluate fixed checkpoints on the frozen common bank and rollout
  population; and
- `analyze`: combine only manifest-verified arm packets under the frozen rule.

Long training cells still receive unique run identities and output roots. A
single CLI does not permit outputs from different arms or seeds to overwrite
one another.

`tests/time_dependent_no/test_corrective_ode_study.py`

Required tests:

- exact flow formula for `kappa=0` and `kappa!=0`;
- closed-form recurrence versus explicit iteration;
- tangent/normal block identities;
- smaller-H1/worse-rollout construction;
- wrong-phase and false-attractor controls;
- recovery and relabeling target semantics;
- response-shrinkage endpoints `lambda=0,1`;
- exact circle distance versus convex-hull counterexample;
- deterministic split/query-bank hashing; and
- result/manifest fail-closure.

### 3.4 PCNO case study

Only after the PDE readiness gate passes:

`scripts/time_dependent_no/train_pcno_information_arms.py`

- one frozen PCNO constructor;
- the reviewed representative training interventions, with CLEAN,
  PF-STORED, RECOVERY, and DYN-RELABEL retained when selected;
- common initialization/data order/update/selection contracts;
- query-bank identity and target-source checks;
- checkpoint and metric receipts; and
- resume only from a manifest-compatible cell.

`scripts/time_dependent_no/evaluate_pcno_information_arms.py`

- one shared raw-predictor and optional-corrector recurrence;
- H1/common-bank evaluation without designated long-horizon opening;
- frozen state-only diagnostics;
- prediction-freeze packet;
- separately gated H16/H64 or contract-specific reveal;
- path/tube/proxy, state, structure, validity, boundary/conservation, and cost
  output; and
- exact separation between pre-reveal and post-reveal artifacts.

`tests/time_dependent_no/test_pcno_information_arms.py`

- objective target semantics;
- identical RECOVERY/DYN-RELABEL inputs;
- identity-corrector parity and raw/deployed separation;
- corrected-state recurrence and composition-order checks;
- no target or long-horizon leakage into diagnostic fitting;
- PCNO architecture/config equality across arms;
- paired seed/data-order equality;
- checkpoint selection independence from designated long horizon;
- state/normalizer/forcing/boundary closure;
- metric and recurrence identities;
- prediction-freeze immutability; and
- manifest/source verification.

Do not create these files while the old residual-FNO M1 model contract is still
being treated as an executable continuation. The new fixed-PCNO study requires
a new preregistration and source schema or an explicitly reviewed extension of
an existing schema.

### 3.5 Analysis and figures

Prefer one bounded result-bound analyzer per completed study over a broad
dashboard. Reuse general plotting helpers only when they already have multiple
callers. Planned outputs:

- exact ODE regime diagram;
- learned ODE response/rollout paired-seed figure;
- H1-versus-long-horizon PCNO ranking plot with discordant pairs;
- diagnostic-score-versus-rollout ranking plot;
- recovery/relabeling mediator table; and
- main paper hero figure assembled only from verified result packets.

Figure scripts must bind input packet hashes and write a figure manifest.

## 4. Blocking Code-To-Intent Audit

Every intervention must receive `AUDIT_PASS` before a scientific smoke or run.
A Codex agent other than the implementer should perform the final trace.

The audit records:

1. the mathematical input, target, training objective, inference action,
   composition order and frequency, recurrent state, permitted information,
   expected diagnostic changes, and falsifier;
2. the actual path from entry point and configuration through query
   construction, target, loss or corrector, recurrence, and metrics;
3. focused tests for target semantics, no-corrector parity, raw/deployed
   separation, corrected-state recurrence, frozen-bank leakage, and stochastic
   reproducibility where applicable;
4. normalization, units, timestep, precision, boundary/forcing, and solver
   restart semantics, including one hand-checkable fixture; and
5. separate predictor, corrector, data/bank, feature, evaluator, and output
   identities.

The verdict is `AUDIT_PASS`, `NEEDS_OWNER_CLARIFICATION`, or `AUDIT_FAIL`.
If ambiguity changes the scientific meaning, the agent stops and asks the
owner before implementing or running anything. Passing unit tests alone is not
an audit pass.

## 5. Data And Query-Bank Contracts

Every query-bank row records:

- system and population identity;
- trajectory/initial-condition identity;
- time/call and conditioning variables;
- clean anchor `u` hash;
- displacement `eta` hash and construction family;
- displaced input `x=u+eta` hash;
- clean target `Phi(u)` hash;
- trusted displaced target `Phi(x)` hash when available;
- admissibility and restart-closure result;
- feature/normalizer identity; and
- split role.

RECOVERY and DYN-RELABEL consume the identical `x` rows. Their target columns
differ. PF-STORED uses a distinct model-prefix query law and is never described
as using `Phi(x)` unless the trusted solver actually evaluated that input.

Query banks are built from open train/development populations only. The bank,
its balancing rule, and perturbation scales are frozen before arm outcomes.

An operational-corrector contract additionally binds its training or reference
bank, permitted conditioning, checkpoint or rule, correction schedule,
stochastic seed and steps where applicable, and inference cost. The evaluator
stores both raw and corrected states. No validation/test truth or future state
may enter the corrector unless the method is explicitly defined as using that
additional information and is reported in a separate comparison class.

## 6. Diagnostic Implementation

### 6.1 Exact ODE geometry

- compute the exact circle/manifold projection;
- compute exact tangent and normal coordinates;
- verify derivative blocks analytically and by centered finite differences;
- use exact distance as the primary tube metric; and
- compare proxies only as sensitivity/negative controls.

### 6.2 PDE reference-proximity diagnostics

Use one frozen state-only representation shared across methods:

- physical mass-weighted low modes or PCA fitted on clean states;
- energy/enstrophy or family-valid invariant summaries;
- fixed spectral bands;
- gradient/vorticity/front summaries as appropriate;
- positivity/admissibility features; and
- boundary leakage or constraint residuals.

Then implement:

- time-/condition-matched normalized kNN distance;
- local-PCA reconstruction residual with dimension/spectral-gap and bootstrap
  qualification;
- MMD on the frozen physical feature map; and
- feature-wise standardized drift.

Fit preprocessing on clean training/reference states only. Calibrate thresholds
on trajectory-disjoint clean states. Never use model-specific PCNO latents or
refit the measuring instrument for an intervention.

## 7. Result And Artifact Schema

Each run directory contains, at minimum:

```text
run_contract.json
source_manifest.json
data_or_query_manifest.json
progress.jsonl
result.json
artifact_manifest.json
closeout.json
```

Training runs additionally retain checkpoints and metric histories under
ignored artifact storage. Prediction-freeze and long-horizon-reveal packets are
separate directories with parent hashes.

Required identity fields:

- run and attempt ID;
- source commit and executable-source digest;
- preregistration hash;
- data, split, population, and query-bank hashes;
- architecture/config and initial-parameter hash;
- predictor and corrector checkpoint/rule hashes, reference-bank hash, and
  composition schedule;
- normalizer and feature-map hashes;
- seed, data order, optimizer, update/presentation count, and selection rule;
- checkpoint hash;
- evaluator hash, precision, recurrence, and boundary policy;
- parent packet hashes;
- sealed/test access booleans; and
- final result and artifact-manifest hashes.

Packet integrity and current-checkout compatibility are verified separately.

## 8. Prospective Freeze Architecture

The evaluator must enforce two phases:

1. `diagnostic`: H1 and common-bank diagnostics only; writes the immutable
   prediction-freeze packet; and
2. `reveal`: verifies the freeze hash, then opens only the named long-horizon
   population and writes a separate result packet.

The reveal command refuses to run when:

- the source or evaluator differs from the freeze;
- any model/checkpoint/query/feature hash differs;
- any corrector, reference-bank, seed, or composition-schedule identity differs;
- the predicted ranking or analysis rule is missing;
- the output path exists;
- the population is not the registered open prospective population; or
- test/sealed access is requested without a separate approval token/contract.

An analyzer reads the freeze and reveal packets together and recomputes ranking
metrics without fitting new parameters.

## 9. Test Ladder

Run the narrowest tests first:

1. analytical unit tests;
2. synthetic query-bank and target-semantics tests;
3. operational-corrector composition, parity, and recurrence tests;
4. synthetic diagnostic and no-leakage tests;
5. tiny CPU model overfit and checkpoint round trip;
6. manifest/source/corrector mismatch failures;
7. prospective freeze/reveal refusal tests;
8. an independent code-to-intent audit pass;
9. one bounded real-data read-only preflight under a named open-population
   evaluation approval; and
10. one measured GPU memory/throughput smoke under a separately named resource
   approval before estimating training.

No dataset-scale run begins because unit tests pass. Each level requires its own
scope and approval.

## 10. Implementation Sequence

| Phase | Work | Exit criterion |
| --- | --- | --- |
| I0 | Owner reviews the plans, then selects the mandatory primary PDE and candidate intervention roles. | One bounded case-study contract; exact choices remain open until that review. |
| I1 | Audit reference, population, restart, PCNO, and reusable intervention code for the selected PDE. Switch candidates promptly if a hard readiness condition fails. | Qualified primary-PDE contract and code-to-intent gap list. |
| I2 | Implement only the missing shared diagnostics, state/query contracts, clean-baseline path, and focused tests. | Synthetic semantics, leakage, recurrence, and fail-closure tests pass. |
| I3 | Qualify or train CLEAN, run the B3 development diagnosis, and freeze problem-specific intervention hypotheses and retained B4 roles. | Signed B3 packet with all parent hashes. |
| I4 | Implement the retained PDE training and correction paths, including deployed-corrector composition. | Independent `AUDIT_PASS` plus tiny CPU and measured resource smokes. |
| I5 | Execute the registered B4 training cells under named approval. | Complete predictor/corrector packets with all registered replications. |
| I6 | Implement and run the compact exact/learned ODE laboratory in parallel under separate approval. | Supporting mechanism packets verify without delaying the PDE. |
| I7 | Compute only preregistered early diagnostics and freeze trained-model rankings before long-rollout reveal. | Immutable prediction packet with all parent hashes. |
| I8 | Reveal and analyze the registered primary-PDE rollouts. | Verified ranking, mediator, no-harm, structure, and cost results. |
| I9 | Generate figures and manuscript tables. | Inputs rehash and figure manifest passes. |

## 11. Cleanup And Maintenance Rules

- Do not duplicate old W26 scripts into new names.
- Do not add a generic experiment framework, database, web dashboard, or YAML
  hierarchy for this bounded study.
- Delete only task-created scratch/cache files after verifying exact resolved
  paths; never delete historical artifacts by age or ignore status.
- Remove any helper created by this implementation if it has no remaining
  caller.
- Keep failed attempts and their logs under distinct identities.
- Report every new long-lived file and its invocation path at handoff.

## 12. Authorization Boundary

This file describes future code. The current authorized action is documentation
rewrite only. No implementation, test execution involving scientific arrays,
solver call, checkpoint load, training, remote work, download, or sealed access
follows automatically.
