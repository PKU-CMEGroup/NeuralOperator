# D089 W26-L4 REALM Domain-Linked Direct Baseline Preregistration

Status: **A3-P0 terminal failure at the registered step-100 eligibility gate;
no continuation or residual arm authorized**

Date: 2026-08-13

Stable ID: `D089`

Frozen attempt:
`d089_realm_ignithit_p1d_domain_link_direct_seed0_5000_20260813a`

This line-specific registration amends the failed D088 direct baseline with one
parameter-free train-time model intervention and an explicit claim-validity
eligibility gate. It does not rewrite D088, resume either
D088 checkpoint, open the sealed test population, or authorize the residual
arm. The archived
[W26-L4 REALM benchmark plan](history/W26_L4_REALM_BENCHMARK_PLAN_through_D091.md)
was the W26-L4 source of truth for this attempt; this file owns only D089's
changed contract.

## Question, Evidence, And Claim Boundary

D088 established that its exact step-100 direct map improved the case-first
fresh one-call normalized error from `0.4941208959` at step 50 to
`0.1957884878`, while introducing 327 `H2O` inverse-domain violations across
all five open validation cases. Matching truth and the step-50 map introduced
none. The first step-100 event had inverse-domain margin `-0.0034511089`.
Those are checkpoint-history-associated validation observations. They do not
show that Box--Cox normalization, direct prediction, FFNO, or the optimizer
caused the tail.

D089 asks a narrower prospective question:

> Can a parameter-free, domain-compatible output parameterization produce an
> evaluable direct FFNO training attempt under the otherwise frozen D088
> contract, without hiding a nonfinite raw proposal or relaxing accuracy,
> decoded-admissibility, or boundedness selection gates?

The prospectively allowed pass claim was only that this amended direct
baseline completed the registered open-validation contract. D089 is not the
original raw direct baseline; its output parameterization and hypothesis class
have changed. A pass does not retrospectively repair D088, identify the cause
of D088's failure, establish conservation, open held-out generalization, or
support direct-versus-residual comparison. A failure does not reject all
domain-compatible maps or REALM.

## Frozen Comparison And Provenance Matrix

| Field | D088 provenance arm | D089 amended arm |
| --- | --- | --- |
| stable identity | `D088` | `D089` |
| attempt | `d088_realm_ignithit_p1c_direct_seed0_5000_20260812a` | `d089_realm_ignithit_p1d_domain_link_direct_seed0_5000_20260813a` |
| role | terminal historical raw-direct attempt | prospective domain-linked direct amendment |
| starting checkpoint | none | none; D088 checkpoints are forbidden |
| model | released-source reconstruction of FFNO-M; 8,936,460 trainable parameters | identical FFNO-M backbone and trainable-parameter count; parameter-free link follows the head |
| input/state | 12-channel normalized state plus two static normalized coordinate channels | identical |
| target | direct next normalized state | identical |
| training population | 26 ordered released IgnitHIT train groups, all 29 adjacent pairs | identical exact allowlist and order |
| selection population | five ordered released validation groups: `phi=_t_15_3_t`, `phi=c_5_4_c`, `phi=_t_15_1_t`, `phi=c_15_3_c`, `phi=c_5_2_c` | identical; model-selection evidence, not untouched test evidence |
| test population | sealed and unavailable | sealed and unavailable; any test-path existence/open is a stop |
| tensor contract | float32 `[30,12,128,128]`; frame cadence `1e-5`; field order fixed by `realm_ignithit.py` | identical |
| normalizer | primary P1b train-only arrays SHA-256 `368d243b5e0f71b6380ee5f49fb9f5cf2724baa94ddece620d3ceae55285ca20`; Box--Cox `lambda=0.1`, epsilon `1e-8`, then z-score | identical arrays and transform; link buffers derive only from these train statistics |
| precision/runtime | one CUDA device, FP32, autocast off, TF32 off, deterministic algorithms, seed 0 | identical; a new runtime manifest is mandatory |
| optimizer/history | Adam, LR `1e-3`, betas `(0.9,0.999)`, epsilon `1e-8`, zero weight decay; OneCycleLR; 5,000 steps; effective batch 26; 130,000 presentations | identical from a fresh seed-0 initialization |
| objective | equal-weight grouped normalized one-step MSE | identical, evaluated on the linked deployed proposal |
| recurrence | deployed normalized model output is the next input; H29 from frame-0 truth | identical; the linked proposal, not the raw head output or decoded state, is fed back |
| output/boundary policy | raw normalized head output; no clipping, floor, smoothing, or boundary intervention | registered smooth species-domain link; no input, decoded-output, recurrence, or boundary repair |
| validation eligibility | retained step 50 happened to pass all four conditions and the terminal step-100 decode stopped before a row, but the parent selector itself did not reject finite inadmissible/unbounded rows | normalized finite, decoded finite, released-admissible, and bounded are mandatory before a row or checkpoint may participate in selection |
| selection | minimum case-first H29 `realm_npe_mean`; strict improvement retains the earliest step | identical metric and tie rule among eligible rows only |
| data/source anchor | open manifest SHA-256 `85e8dcdaf4a5f7726ac5a7ad18a1bc131785fc1212acc9318ff69b94be965205`; D088 config digest `9949335070d23ecd8719a94c67d327e0aa53ce421ea1dffa27892b7fba599fd7`; frozen parent trainer SHA-256 `a3fa0c0e831765ce24c6ae3aa11e3d2afcf67cc242de0a508381dc0b25b43f1e` | same input and parent anchors plus the D089 source manifest, runtime manifest, and distinct config/run signatures |

Coordinate and time units, physical centering, omitted-species semantics, and
complete reactive conservation remain unavailable. Released-state
admissibility is therefore not a composition-conservation claim.

## Registered Output Link

Only released species channels 0--7 are transformed. Temperature, density, and
both velocity channels pass through bit-for-bit. Let `m_c` and `s_c>0` be the
frozen P1b transformed-space mean and scale, `lambda=0.1`, `epsilon=1e-8`, and

```
q_floor = epsilon**lambda = 0.15848931924611134
l_c = ((q_floor - 1) / lambda - m_c) / s_c
d_c = -l_c
a_c = softplus_inverse(d_c)
z_c(r) = l_c + softplus(r + a_c)
```

The implementation rejects any nonfinite raw normalized head tensor before the
link. For finite `r`, `z_c(r) >= l_c` numerically and its inverse Box--Cox base
is positive; the configured floor decodes to species value `epsilon`. Floating
roundoff may realize equality at the floor for very negative `r`, so the tested
contract is positive inverse base and decoded species `>= epsilon`, not strict
separation from the floor. The shift makes raw zero map to normalized zero,
avoiding a gratuitous offset at initialization. The link is monotone, smooth,
truth-independent, and has no trainable parameters.

The deployed chain is:

```
current normalized state
  -> FFNO raw normalized proposal
  -> reject if raw proposal is nonfinite
  -> species-domain link (channels 0..7 only)
  -> deployed normalized proposal
  -> decode only for metrics
  -> feed the same deployed normalized proposal into the next call
```

`DomainLinkedMap.forward_raw` exposes the unlinked proposal for a separately
authorized diagnostic. Training and standard validation use `forward`; no raw
proposal is silently substituted into recurrence. D089 records the configured
and realized floor and the zero-mapping error without performing a second model
call. It does not currently run a raw-versus-linked validation study.

A penalty-only alternative is rejected for this attempt because a finite
penalty does not guarantee the inverse-transform domain. Hard clipping is
rejected because it is nonsmooth and would conflate training parameterization
with an inference repair. Exponential/log-space replacement is broader than
necessary and would not preserve the normalized zero center.

## Metrics, Events, And Selection

| Channel | Definition | Role |
| --- | --- | --- |
| normalized accuracy | paper-formula case-first H29 `realm_npe_mean`, plus grouped/per-call curves | primary selection among eligible rows |
| inverse-domain margin | `1 + 0.1 * (z*s_c + m_c)` for species channels; configured and realized floor recorded | parameterization audit; not accuracy |
| decoded finiteness | every returned decoded value finite | mandatory eligibility/stop |
| released-state admissibility | all released species `>=0`, temperature `>0`, density `>0` | mandatory eligibility/stop; not conservation |
| boundedness | every decoded channel remains within 10 times the train-only maximum-magnitude envelope | mandatory eligibility/stop; not accuracy |
| recurrence | next input exactly equals the preceding deployed normalized proposal | contract gate |

The model intervention is only the output link. The stronger eligibility rule
is a claim-safety change: it does not change gradients, but it can change which
checkpoint is selectable. D089 is therefore an amended operational baseline,
not a one-factor causal intervention on D088.

All five validation cases and all 29 calls are evaluated before aggregation.
There is no accepted-prefix averaging and no row-dependent cohort. A returned
nonfinite raw/deployed tensor, decode failure, inadmissible state, unbounded
state, missing metric, or nonfinite selection score stops before that
validation row and its checkpoints are written. Infrastructure failure before
a returned tensor produces no scientific event time. Validation remains a
model-selection population.

## Execution Stages And Gates

### A1 — source and synthetic CPU work

Scope is exactly one reusable utility, one isolated entry point, one focused
test, this preregistration, and compact index/plan routing. It may not open the
real normalizer array, trajectory tree, D088 checkpoint, CUDA, remote host, or
test object.

A1 passes only if:

1. `D089` and its attempt label do not collide;
2. the D088 parent trainer remains byte-identical at its registered SHA-256;
3. selected species inverse bases stay positive over extreme finite synthetic
   proposals and channels 8--11 remain exact identities;
4. normalized zero maps to zero within `1e-6` in FP32 and `1e-14` in FP64;
5. gradients are finite and nonzero in the synthetic adverse tail;
6. nonfinite raw proposals raise before the link;
7. the link adds zero trainable parameters and the default model remains
   8,936,460 trainable parameters;
8. a three-call synthetic recurrence and backward pass are finite;
9. validation eligibility fails closed for each finite, decoded, admissible,
   bounded, or accuracy field;
10. D089 temporarily binds distinct run/checkpoint/runtime/history/status
    identities and restores every D088 parent symbol even on failure;
11. the executed-source manifest binds every imported model, adapter, trainer,
    data-contract, and metric source; and
12. Ruff, safe CLI help, the focused D089/D088 suite, and the full maintained
    REALM CPU suite pass.

### A3-P0 — prospective GPU step-100 gate; executed once and failed

The prospectively authorized GPU request was one fresh seed-0 run through
registered validation step 100 only, using `--stop-after-step 100`. It could run
only after a fresh-process preflight verified the committed source hashes, exact open
manifest/tree, P1b normalizer, test absence, one visible idle GPU, deterministic
runtime flags, and a new noncolliding output directory. It did not load or
resume D088.

The frozen promotion rule required validations at steps 1, 50, and 100 to
produce eligible rows, the step-100 case-first H29 `realm_npe_mean` to be
strictly below D088's last eligible step-50 value `4.5625491142`, recurrence and
model/checkpoint digests to verify, and no test object, raw-output repair, retry,
or runtime drift to occur.
Any gate failure was terminal for this attempt. A P0 pass would have been
viability evidence, not a completed baseline; resumption toward 5,000 steps
would have required a separate A3 authorization and exact restoration of all
registered identities.

The `4.5625491142` P0 cutoff is explicitly pilot-informed by the already known
D088 step-50 result. It is a prospective viability rule only for the new D089
row, not an independent confirmatory threshold. Continuous case-first and
per-case/per-call metrics remain primary.

The residual arm remains blocked until a D089 direct amendment completes the
full registered budget and owner review accepts it as the baseline for a new
matched comparison. PlanarDet remains conditional on that later comparison.

Result interpretation is frozen prospectively:

| Observation | Supported implication | Live alternatives |
| --- | --- | --- |
| species margins pass but temperature/density admissibility fails | the registered species link addressed only its stated domain | unconstrained non-species heads, optimization, or recurrence can still fail |
| all structure gates pass but normalized error misses the P0 cutoff | domain compatibility alone is insufficient for a viable amended baseline | gradient saturation, changed hypothesis class, optimization, or ordinary estimation error |
| P0 passes | the amended direct system is viable through step 100 on open validation | it remains partial, seed-0, model-selection evidence |
| full D089 later completes | owner may consider it as an amended direct comparator | original D088 and causal failure attribution remain unresolved |

## A1 Artifact Inventory And Cost

Long-lived files:

| File | Reason to exist |
| --- | --- |
| `utility/time_dependent_no/realm_domain_link.py` | reusable parameter-free domain map and explicit raw-proposal API |
| `scripts/time_dependent_no/train_realm_ignithit_domain_linked_ffno.py` | isolated D089 entry point that reuses and restores the byte-frozen D088 loop |
| `tests/time_dependent_no/test_realm_ignithit_domain_linked_ffno.py` | synthetic algebra, gradient, recurrence, provenance, and fail-closed gates |
| this file | line-specific changed contract and authorization boundary |

No configuration file is added because the attempt is intentionally knob-free.
No generated report, array, checkpoint, log, or artifact directory belongs in
Git. A1 cost is CPU-small. Conditional A3-P0 is one-GPU small relative to the
full 5,000-step run and stops at step 100.

## A1 Closeout

Status: **COMPLETE; ALL A1 GATES PASS** on 2026-08-13.

Exact executable identities before the reviewed-file commit:

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `utility/time_dependent_no/realm_domain_link.py` | 6,592 | `faf70ee35ba04ccd81805c91e1f32b8af233662c342a8276b44be1d51004c2a0` |
| `scripts/time_dependent_no/train_realm_ignithit_domain_linked_ffno.py` | 15,100 | `cc0a7027da296fb9d805af20674b2eb6ae954911f20df3bccfd221a0080720ec` |
| `tests/time_dependent_no/test_realm_ignithit_domain_linked_ffno.py` | 11,787 | `f19a0ec746a4f8706681f4c7f23f37c7b06fb3dcd4de05f08a7043ca5d50e77f` |

The frozen D088 parent trainer remains byte-identical at
`a3fa0c0e831765ce24c6ae3aa11e3d2afcf67cc242de0a508381dc0b25b43f1e`.
The D089 config digest is
`9e1edc140c29ec79fbc99e7c65dffa295e045e01c526b077664a4314ccb7a161`.
The canonical executed-source manifest digest is
`b301cab59e53870f46a5ac2ae0252646e51939e3c567d0ce85f3910b919e31ec`;
the executed A3 preflight recomputed and matched it after checkout.

`ruff check` and `ruff format --check` pass for all three D089 Python files.
The complete maintained REALM synthetic CPU suite passes `109/109`, including
the frozen D088 trainer and both D088 diagnostic descendants. Safe CLI help
exits zero. No real trajectory or normalizer array, scientific checkpoint,
CUDA context, remote host, test object, residual arm, or generated scientific
artifact was opened or created.

## A3-P0 Result And Claim Decision

The owner authorized exactly one fresh seed-0 D089 run through validation step
100. The attempt ran once on 2026-08-13 from `09:51:53+08:00` to
`09:53:00+08:00` and
terminated with process exit code `2` when the registered validation
eligibility check raised at step 100. It was not resumed or retried. The sealed
test object remained absent, no D088 checkpoint was loaded, and no raw-output
repair was used.

The fresh-process preflight passed before the first model call. It verified all
34 open-manifest entries totaling 552,023,019 bytes, open-manifest digest
`85e8dcdaf4a5f7726ac5a7ad18a1bc131785fc1212acc9318ff69b94be965205`,
the P1b normalizer digest, exact D089 source and config digests, 8,936,460
trainable parameters, one visible idle GPU, and deterministic FP32 runtime. The
executed run signature is
`d5e0acb53973ec9c89ad51aeb718bd91e84a4f192971ff8d47f1235a3efbd5e1`.

The two retained eligible rows are:

| Step | Train grouped loss | Case-first H29 `realm_npe_mean` | Maximum 10x-envelope ratio | Finite / decoded finite / admissible / bounded |
| ---: | ---: | ---: | ---: | --- |
| 1 | 6.1849659406 | 4.5477657318 | 0.1403203458 | pass / pass / pass / pass |
| 50 | 1.3774454662 | 4.9248557091 | 0.2253825963 | pass / pass / pass / pass |

At step 100, at least one of normalized finiteness, decoded finiteness,
released-state admissibility, or 10x-envelope boundedness failed. D089 itself
did not retain the exact flag or continuous metrics: it called the fail-closed
eligibility check before appending the validation row or writing the checkpoint.
Consequently, the D089 last checkpoint remains the eligible step-50 state and
the selected best checkpoint remains step 1. The separately registered and
authorized D090 diagnostic later reconstructed the missing row without changing
those D089 identities; its result is summarized below.

The exact retained identities are:

| Artifact | Step | SHA-256 / structured-state SHA-256 |
| --- | ---: | --- |
| `best.pt` | 1 | file `c25f519f294ec8a11b8b863d7d00b8724b18ef67304cfa632ea954344973ef56`; model `efcbdac2e5d85f653ac0bee8452a39435f857f2ba1b5eedd176f12e0e19f8c1f` |
| `last.pt` | 50 | file `e0dc58273727fb07e5cb698df805e27ea3f43e68a6a259c1c1581649ebd047bf`; model `6bb34b88b082e7e6d5668c651703707949a4a7d8b2ef4334ed903655c0deb7c0` |
| terminal audit | n/a | canonical payload `d9b00215fd83cfc67a9e3c7448e4e8dd9146468125919ae71ae13afadd3235c7` |

Local result-to-claim verdict:
**`claim_supported: no` for P0 viability, confidence high**. The unpublished
result was not disclosed to an external reviewer. D089 alone supports only that
the domain-linked direct map remains eligible through step 50 under the
registered open-validation contract. They do not support step-100 viability,
full-baseline completion, a residual comparison, or a mechanism claim about
which state channel failed. D090 subsequently resolves the gate as boundedness
only but still does not identify the responsible state channel. This single
failure also does not reject all
domain-compatible maps, direct neural operators, FFNO, or REALM.

The attempt is terminal and full 5,000-step continuation and the residual arm
remain blocked. That smallest A1 is now complete under the distinct `D090`
contract in
[D090_W26_L4_REALM_FAILURE_CAPTURE_PREREGISTRATION.md](D090_W26_L4_REALM_FAILURE_CAPTURE_PREREGISTRATION.md).
At A1 closeout it changed no D089 result and executed no real data, checkpoint,
GPU, test population, PlanarDet, or residual training. The owner later
authorized its exact one-replay A3 scope.

## Subsequent D090 Diagnostic Resolution

D090 ran once on 2026-08-13, reconstructed exactly D089 steps 51--100, wrote
the full pre-gate row, called D089's unchanged gate, and stopped. The D089
failure is boundedness alone: normalized and decoded finiteness pass,
released-state admissibility passes for all 145 case-calls, and the maximum
registered-envelope ratio is `2.2733771801`, while H29
`realm_npe_mean=5.7428269386` misses the pilot-informed cutoff. Every validation
case contains at least one unbounded call. D090 did not retry, continue, select,
open test, repair output, run PlanarDet, or execute a residual arm.

This resolves the missing D089 gate identity without making the diagnostic
model a D089 checkpoint. It supports a boundedness-versus-admissibility
separation under the exact D089 checkpoint history, not a causal account of the
training dynamics or offending state channel. Exact metrics, hashes, per-case
counts, alternatives, and the next minimum diagnostic are in
[D090_W26_L4_REALM_FAILURE_CAPTURE_PREREGISTRATION.md](D090_W26_L4_REALM_FAILURE_CAPTURE_PREREGISTRATION.md).
