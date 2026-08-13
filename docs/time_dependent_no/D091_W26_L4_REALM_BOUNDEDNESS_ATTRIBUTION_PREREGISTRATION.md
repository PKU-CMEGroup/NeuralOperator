# D091 W26-L4 REALM Boundedness Attribution Preregistration

Status: **A1 complete; no scientific execution performed or authorized**

Date: 2026-08-13

Stable ID: `D091`

Frozen prospective attempt:
`d091_realm_ignithit_p1d_boundedness_attribution_20260813a`

This line-specific registration follows the terminal D089 direct-baseline
amendment and the terminal D090 step-100 failure capture. It owns one matched
inference-side diagnostic only. It does not resume training, select a model,
repair a state, access a test object, run the residual arm, or start PlanarDet.
[W26_L4_REALM_BENCHMARK_PLAN.md](W26_L4_REALM_BENCHMARK_PLAN.md) remains the
W26-L4 routing source of truth.

## Question And Existing Evidence

D090 established that the exact D089 step-100 deployed map remains finite and
passes the registered released-state admissibility test on all 145 open
validation case-calls, but only 96 calls are bounded. Every one of the five
cases contains an unbounded call, and the population maximum is 2.2733771801
times the registered 10x-train-maximum limit. Its retained aggregate row does
not identify the offending channel, first event, spatial support, or whether
the excursion is already present under exact truth input.

D091 asks:

> For the exact D089 step-50 and D090 step-100 deployed maps, on the same five
> open validation trajectories, which channels first exceed the registered
> envelope, where do they exceed it, and does the same checkpoint cross the
> threshold when the recurrent input is replaced by matching truth?

The minimum decisive contrast is a checkpoint-matched teacher-forced/free-
recurrence evaluation. It is an inference-input intervention, not a training-
factor intervention. It can establish whether propagated input is required
for a particular registered threshold crossing. It cannot identify why the
checkpoint learned that map or causally isolate the optimizer, training
objective, domain link, architecture, or representation.

## Frozen Provenance And Comparison Matrix

| Field | Step-50 map | Step-100 map |
| --- | --- | --- |
| source attempt | D089 `d089_realm_ignithit_p1d_domain_link_direct_seed0_5000_20260813a` | D090 `d090_realm_ignithit_p1d_step100_failure_capture_20260813a` reconstructed from the same D089 history |
| checkpoint role | D089 retained `last.pt`; non-best, completed step 50 | D090 `step100_model.pt`; diagnostic-only, nonresumable, nonselectable |
| checkpoint file SHA-256 | `e0dc58273727fb07e5cb698df805e27ea3f43e68a6a259c1c1581649ebd047bf` | `ee6bd1e6dc21897fb602ab0cb024f36190fe2b83dc20854b30727070d5e801b0` |
| model-state SHA-256 | `6bb34b88b082e7e6d5668c651703707949a4a7d8b2ef4334ed903655c0deb7c0` | `1af738ece2b2f4167e51905cd67622a052c0fc7b0e96714251130af0745c2f93` |
| model | exact D089 domain-linked FFNO-M; 8,936,460 trainable parameters | identical deployed architecture |
| training history | fresh seed-0 D089 history through step 50 | exact D090 reconstruction of the same history through step 100 |
| optimizer history | Adam/OneCycleLR D089 state through step 50; not restored for D091 inference | exact restored D089 optimizer/RNG trajectory through step 100; not restored for D091 inference |
| input/state | float32 normalized 12-channel state plus two static normalized coordinate channels | identical |
| normalizer | P1b train-only arrays SHA-256 `368d243b5e0f71b6380ee5f49fb9f5cf2724baa94ddece620d3ceae55285ca20`; Box--Cox then z-score | exact same state embedded in the diagnostic checkpoint and independently reconstructed from the same arrays |
| inference precision | exact retained single-CUDA-device FP32 runtime; autocast and TF32 off; deterministic algorithms; seed 0 | identical |
| recurrence | frame-0 truth, then each finite deployed normalized proposal is the next input through call 29 | identical |
| output/boundary policy | parameter-free D089 species-domain link; no input, decoded-output, recurrence, or boundary repair | identical |
| evaluator | D091 executed-source manifest including model, adapter, normalization, metric, and attribution sources | identical manifest and process |
| selection rule | none; checkpoint identity is frozen before D091 | none; diagnostic artifact is never selectable |

The D089 parent identity remains additionally bound by config digest
`9e1edc140c29ec79fbc99e7c65dffa295e045e01c526b077664a4314ccb7a161`,
input digest
`08d80fc972ae5ace8f4c3c5392c4fd36b53187f2d2a7655fff820257a2ec0074`,
executed-source digest
`b301cab59e53870f46a5ac2ae0252646e51939e3c567d0ce85f3910b919e31ec`,
runtime digest
`e92f794237558837f4d0cef5c23aed535213dc47245092a01208703e8dc9e0f2`,
and D089 run signature
`d5e0acb53973ec9c89ad51aeb718bd91e84a4f192971ff8d47f1235a3efbd5e1`.
The distinct D090 diagnostic run signature is
`f9012ab69410f418c9405bce2f69a8f68597241cc44f5374d3021e1d2e5aa04b`.
D091 must validate every registered D089 and D090 parent file and inventory;
loading only a matching-looking model state is insufficient.

### Population and data contract

The ordered evaluation population is exactly:

1. `phi=_t_15_3_t`
2. `phi=c_5_4_c`
3. `phi=_t_15_1_t`
4. `phi=c_15_3_c`
5. `phi=c_5_2_c`

These are released validation/model-selection trajectories, not untouched test
evidence. Each starts at frame 0 and has matching truth through frame 29, so
calls 1--29 map frame `h-1` to frame `h`. The exact released IgnitHIT open
manifest SHA-256 is
`85e8dcdaf4a5f7726ac5a7ad18a1bc131785fc1212acc9318ff69b94be965205`.
The field order is exactly `H, H2, H2O, H2O2, HO2, O, O2, OH, T, rho, Ux,
Uy`. A test-path existence check is mandatory before any scientific model
call; any test object or population drift stops the run.

## Frozen Inference Interventions

For checkpoint map `G_z`, with static normalized coordinates `z` held fixed:

- **teacher forced:** call `h` returns `G_z(u_{h-1})`, where `u_{h-1}` is the
  exact normalized validation truth; calls are independent and no proposal is
  fed into another call;
- **free recurrence:** `u_hat_0=u_0` and call `h` returns
  `u_hat_h=G_z(u_hat_{h-1})`; every finite deployed proposal is fed back
  unchanged, even if its decoded state is unbounded;
- **truth context:** decoded truth `u_h` is evaluated against the same envelope
  but is never substituted for a model proposal except in the registered
  teacher-forced input intervention.

The deployed normalized proposal is the post-link model return. The raw FFNO
head output is not part of D091. There is no clipping, floor beyond the frozen
D089 output parameterization, smoothing, input projection, decoded-state
repair, boundary splice, or invalid-stop policy. If a finite unbounded free
state later returns within the envelope, that is a raw recurrent re-entry: the
unbounded deployed state was still fed into the next call.

## Stability Event Schema

Let `M_c` be the exact train-only maximum absolute decoded magnitude for
channel `c`, and let the registered inclusive limit be `L_c=10 M_c`. For case
`i`, call `h`, channel `c`, and mode `q`, define

```
m[q,i,h,c] = max_(y,x) |proposal_decoded[q,i,h,c,y,x]|
r[q,i,h,c] = m[q,i,h,c] / L_c
```

If `L_c=0` and `m=0`, the ratio is defined as zero. If `L_c=0` and `m>0`, the
ratio is positive infinite and is represented in JSON as `null` with an exact
reason code; NaN/Infinity JSON tokens are forbidden. A channel is bounded iff
all of its returned values are native-finite and `m<=L_c`. Equality passes.
Any-channel boundedness is the conjunction over all 12 channels.

For predicate `P`, the first failure event is
`E_P=min{h: P(h)=false}`. The accepted prefix is `T_P=E_P-1`; if no failure is
observed through call 29, `T_P=29` with right-censoring. Event calls are
1-based. Later re-entry never changes the accepted prefix. D091 reports:

| Event/metric | Exact meaning | Truth requirement | Aggregation |
| --- | --- | --- | --- |
| `T_decoded_finite` | accepted prefix before the first decoded-nonfinite model state | none | per checkpoint/mode/case |
| `T_bounded` | accepted prefix before the first any-channel registered-envelope failure | none | per checkpoint/mode/case |
| channel `T_bounded` | accepted prefix before that channel first fails its limit | none | per checkpoint/mode/case/channel |
| envelope ratio | per-call/channel `r`, signed row-major argmax value and `(row,column)`, strict-exceedance count, and row/column bounding box | none | no mixed-prefix mean |
| truth envelope ratio | same calculation on matching decoded target frame | matching truth | per case/call/channel, checkpoint-independent |
| first-event attribution | teacher-forced status at each channel failing at the free first event | matching truth input | per checkpoint/case/event channel |
| paper-compatible error | unchanged case-first H29 `realm_npe_mean` and grouped per-call NPE, used only as replay/evidence context | matching truth | existing D089 evaluator |

A decoded-nonfinite model state fails decoded finiteness and boundedness on
that call. Decoding is metrics-only: if the deployed normalized proposal is
finite, that exact normalized tensor is still fed back and later decoded calls
remain observable. The D089 domain link raises on a nonfinite raw normalized
proposal before deployment, so such an exception returns no deployed state and
produces no scientific event time. Teacher-forced calls are independent and no
teacher-forced proposal is ever fed back. A load, hash, source, runtime, OOM,
or exception before a deployed state is returned is likewise infrastructure
failure.

The spatial maximum uses the first row-major index attaining the channel's
maximum absolute value. The record also stores the signed decoded value, the
number of cells strictly beyond the inclusive limit, and the smallest
axis-aligned row/column bounding box containing that support. This is
localization evidence, not a shock, conservation, or causal spatial-support
claim.

### Fresh-versus-propagated attribution

For every channel that fails at a free trajectory's first any-channel event
call `E`, D091 evaluates the already-registered teacher-forced proposal for
the same checkpoint, case, call, channel, and static coordinates:

- teacher-forced bounded and free unbounded means the propagated input is
  required for that exact threshold crossing under the truth-input-reset
  intervention;
- teacher-forced unbounded means an exact-truth input is already sufficient
  for a threshold crossing in that channel at that call;
- teacher-only crossings and different first-event times remain separate
  observations and are not forced into either label.

This intervention does not algebraically decompose error, prove recurrent
instability, or identify a training cause. Ratios and event times remain the
primary outputs; categorical labels are deterministic summaries of them.

## A1 Implementation And Gates

The owner authorized D091 A1 preregistration, reusable evaluator code, an exact
entry point, and synthetic CPU tests on 2026-08-13. A1 excludes real
trajectory/normalizer arrays, loading or executing either checkpoint, CUDA or
remote access, the test population, training/replay, model selection, repair,
the residual arm, and PlanarDet.

A1 passes only if focused synthetic tests establish:

1. stable ID, attempt label, and registered filenames are noncolliding;
2. tensor axes, unique ordered case/channel identities, and call count fail
   closed;
3. `r=1` passes and `r>1` fails, including exact zero-limit reason codes;
4. row-major spatial argmax and signed-value records are deterministic;
5. any-channel and per-channel accepted prefixes use first failure minus one,
   censor at the observed horizon, and do not reset after re-entry;
6. decoded nonfiniteness fails decoded finiteness and boundedness on that call
   without silently changing the finite normalized recurrence policy;
7. finite unbounded free proposals are fed back unchanged;
8. a synthetic fresh failure and a synthetic propagated-input-required failure
   receive the frozen labels only under their exact input interventions;
9. truth records are derived from the passed truth tensor and never from a
   model arm;
10. JSON payloads are strict and contain no NaN or Infinity token;
11. parent/source/runtime/checkpoint/output inventories fail closed; and
12. Ruff, safe CLI `--help`, the focused D091 suite, and maintained REALM CPU
    tests pass.

The reviewed A1 contract digest is
`d4969ab61d4391958e34a00b01e176a879f8132addf953d0bfffd819a908294c`.
The reviewed canonical executed-source manifest digest is
`f0543bce94c23a285dd125d4079ccf1142b8228d8a6d58b2e093202795e2facb`.
The latter must match the required A2 provenance argument before the first
model call.

Reviewed A1 file SHA-256 identities are:

| File | SHA-256 |
| --- | --- |
| `utility/time_dependent_no/realm_boundedness_attribution.py` | `f31c808d869d2489d78289403ea6a3104b17e6a7e24f2f2f3a2a3a6ca27e929b` |
| `scripts/time_dependent_no/diagnose_realm_ignithit_boundedness_attribution.py` | `da5308649e5eabf27f891a30f3e0305f1da833f9415433bb01638850b8d94dd7` |
| `tests/time_dependent_no/test_realm_boundedness_attribution.py` | `7756e407c82701bf62ff0fb4dd9f60af6787129ec82deb1444dfcb422f30df62` |

## Prospective A2 Scientific Execution

A2 is **not authorized by A1**. If separately authorized, it consists of one
exact inference-only execution on the registered GPU/runtime and new empty
output directory. It evaluates both frozen checkpoints, both modes, all five
cases, and calls 1--29 exactly once. There is no retry or exploratory knob.

Before the first scientific model call, stop unless:

1. the exact D089 seven-file parent and D090 twelve-file diagnostic inventories
   and every file hash match;
2. checkpoint schemas, diagnostic/selection/resume flags, model-state hashes,
   normalizer states, parameter count, and training-history identities match;
3. before the first model call, current model/adapter/normalizer/metric/
   diagnostic sources match the exact reviewed digest supplied through the
   required provenance-only `--expected-source-digest` argument;
4. manifest, open data tree, validation order, tensor shapes, frame count,
   channel order, normalizer arrays, and test absence match;
5. PyTorch/CUDA/cuDNN/device, FP32 model/input dtype, autocast, TF32,
   deterministic flags, and all inference seeds match the exact D089 runtime;
6. the output directory is absent or an already-empty dedicated directory and
   registered input/output paths are pairwise distinct; and
7. no model call has occurred during preflight.

Post-return equivalence gates require the step-100 free normalized and decoded
prediction hashes and aggregate boundedness row to equal D090 exactly. The
step-50 free row must reproduce D089's retained H29 NPE
`4.924855709075928`, all 145 bounded calls, and maximum ratio
`0.22538259625434875`. Any mismatch invalidates D091 before attribution.

The registered compact A2 inventory is exactly:

| File | Purpose |
| --- | --- |
| `contract.json` | frozen D091 question, modes, metrics, gates, and anti-claims |
| `checkpoint_identity.json` | exact D089 step-50 and D090 step-100 identities |
| `input_manifest.json` | exact open-data/population/normalizer binding |
| `source_manifest.json` | canonical executed model/adapter/normalizer/evaluator chain |
| `runtime_manifest.json` | exact deterministic inference runtime |
| `stage_manifest.json` | input/truth/coordinate and per-map prediction tensor hashes only |
| `ratio_records.json` | strict compact case/call/channel ratios and spatial maxima |
| `event_summary.json` | per-case/channel event prefixes and frozen attribution labels |
| `summary.json` | equivalence gates, continuous aggregate context, and verdict |
| `final_hash_manifest.json` | self-excluding exact output inventory and hashes |

No prediction array, decoded trajectory, checkpoint copy, animation, optimizer
state, or generated dataset is retained by D091. Cost class is A1 `XS` CPU;
the prospective A2 is one-GPU `XS`, expected under two minutes with a hard
ten-minute cap.

## Claim Policy And Decision Rule

**Allowed if A2 passes provenance/equivalence gates:** exact open-validation,
checkpoint-history-associated statements about which channels and calls cross
the frozen envelope under teacher forcing and free recurrence; whether the
truth-input-reset intervention removes a particular free threshold crossing;
and whether a later bounded row is raw recurrent re-entry after an unbounded
state was fed back.

**Forbidden:** a completed baseline; untouched-test or benchmark-wide
performance; physical conservation; full reactive admissibility; general
stability; a causal optimizer/objective/domain-link/architecture result; or a
direct-versus-residual conclusion.

The minimum decision is descriptive. If teacher forcing is bounded at the
free first events, propagated input is required for those exact threshold
crossings and a later training proposal may target recurrent exposure. If
teacher forcing already fails at those events, fresh deployed-map behavior is
already sufficient and recurrence-only repair is not justified. Mixed cases
remain mixed; no majority label is promoted to a universal cause. Either
outcome leaves D089 continuation, residual comparison, sealed test access, and
PlanarDet blocked pending owner review.

## A1 Long-Lived Files

| File | Reason to exist |
| --- | --- |
| `utility/time_dependent_no/realm_boundedness_attribution.py` | reusable exact channel/event extraction absent from the aggregate REALM evaluator |
| `scripts/time_dependent_no/diagnose_realm_ignithit_boundedness_attribution.py` | knob-free D091 provenance and matched-map entry point |
| `tests/time_dependent_no/test_realm_boundedness_attribution.py` | synthetic event, recurrence, attribution, provenance, and CLI gates |
| this file | line-specific contract, authorization, evidence, and claim source of truth |

## A1 Closeout

A1 completed locally on 2026-08-13. Ruff, safe CLI help, 12 focused D091
synthetic CPU tests, and the complete maintained 134-test REALM CPU suite pass.
The gates cover inclusive/zero-limit ratios, deterministic row-major maxima,
decoded-nonfinite events, first-failure accepted prefixes, raw re-entry,
normalized recurrence equality, matched truth-input intervention labels,
truth/model separation, strict JSON, exact inventories, output isolation,
checkpoint roles, source coverage, and CUDA-safe help.

No real trajectory or normalizer array was opened. Neither checkpoint was
loaded into a model or executed. No CUDA context, remote host, test object,
training/replay, retry, continuation, selection, output repair, residual arm,
or PlanarDet execution occurred. D091 A1 therefore adds no scientific model
result and does not resolve the offending channel or fresh-versus-propagated
mechanism. The smallest next request is one exact A2 inference-only execution
under this contract.
