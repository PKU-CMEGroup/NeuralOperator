# D090 W26-L4 REALM Step-100 Failure-Capture Preregistration

Status: **A1 complete; diagnostic replay not executed and not authorized**

Date: 2026-08-13

Stable ID: `D090`

Frozen attempt:
`d090_realm_ignithit_p1d_step100_failure_capture_20260813a`

This line-specific registration owns one deterministic reconstruction of the
missing D089 step-100 validation row. It does not reopen D089, continue its
training, create a selectable checkpoint, open the sealed test population, or
authorize the residual arm. The D089 result and checkpoint identities remain
unchanged.

## Question And Evidence Boundary

D089 retained eligible validation rows at steps 1 and 50, then its step-100
validation raised `validation failed the finite/admissible/bounded gate` before
the row or checkpoint was serialized. Its exact failing flag and continuous
step-100 metrics are therefore unknown.

The maintained call chain gives one additional, bounded inference. Before D089
raises that message, the parent validation routine has completed its H29
rollout and metric summary. That routine would already have raised distinct
errors for a nonfinite normalized proposal or a nonfinite decoded proposal.
Therefore D090 prospectively expects the captured failing set to contain
`all_released_state_admissible` and/or `all_bounded_10x_train_max`, with both
finite flags true. This is a preregistered expectation, not a D089 observation.

D090 asks only:

> Under an exact reconstruction of the D089 step-50 state and optimizer/RNG
> history, which registered step-100 eligibility flags fail, and what are the
> corresponding already-defined H29 validation metrics?

The answer is checkpoint-history-associated diagnostic evidence. It cannot
identify why training produced the state, establish a completed baseline,
select step 100, support a residual comparison, or generalize beyond the five
open validation cases.

## Exact Parent Provenance

| Field | Frozen D089 binding |
| --- | --- |
| parent run | `d089_realm_ignithit_p1d_domain_link_direct_seed0_5000_20260813a` |
| executed run signature | `d5e0acb53973ec9c89ad51aeb718bd91e84a4f192971ff8d47f1235a3efbd5e1` |
| config digest | `9e1edc140c29ec79fbc99e7c65dffa295e045e01c526b077664a4314ccb7a161` |
| input digest | `08d80fc972ae5ace8f4c3c5392c4fd36b53187f2d2a7655fff820257a2ec0074` |
| executed-source digest | `b301cab59e53870f46a5ac2ae0252646e51939e3c567d0ce85f3910b919e31ec` |
| runtime digest | `e92f794237558837f4d0cef5c23aed535213dc47245092a01208703e8dc9e0f2` |
| retained best | step 1; file SHA-256 `c25f519f294ec8a11b8b863d7d00b8724b18ef67304cfa632ea954344973ef56`; model-state SHA-256 `efcbdac2e5d85f653ac0bee8452a39435f857f2ba1b5eedd176f12e0e19f8c1f`; H29 `realm_npe_mean=4.547765731811523` |
| retained last | step 50; file SHA-256 `e0dc58273727fb07e5cb698df805e27ea3f43e68a6a259c1c1581649ebd047bf`; model-state SHA-256 `6bb34b88b082e7e6d5668c651703707949a4a7d8b2ef4334ed903655c0deb7c0`; H29 `realm_npe_mean=4.924855709075928` |
| retained output inventory | exact seven regular files: `best.pt`, `last.pt`, `config.json`, `history.json`, `input_manifest.json`, `source_manifest.json`, `runtime_manifest.json`; every file hash is frozen in source |

The D090 preflight must recompute every file hash, reconstruct the current D089
config/input/source/runtime payloads, restore the step-50 model, optimizer,
scheduler, Python/NumPy/PyTorch/CUDA RNG states, and validate the retained
step-1 best checkpoint. Any mismatch stops before optimizer step 51.

## Frozen Data, Model, And Replay Contract

| Field | D090 binding |
| --- | --- |
| data | exact D089/D088 IgnitHIT open manifest and P1b normalizer; train and validation only |
| training population | 26 frozen ordered train groups; all 29 adjacent pairs |
| validation population | five frozen ordered validation groups; selection population, not untouched test evidence |
| test population | unavailable; no object may be opened |
| model | exact D089 domain-linked direct FFNO-M, 8,936,460 trainable parameters |
| state/target | normalized 12-channel direct next state plus two static normalized coordinate channels |
| precision/runtime | exact retained D089 single-GPU FP32 runtime; autocast and TF32 off; deterministic algorithms; seed 0 |
| optimizer/history | restored D089 Adam, OneCycleLR, sampler and RNG states at completed step 50 |
| replay | exactly optimizer steps 51--100 through the maintained registered replay helper |
| validation | one fresh frame-0 H29 rollout on the five ordered validation cases |
| recurrence | deployed domain-linked normalized proposal is fed back directly |
| boundary/output repair | none; no clipping, floor, smoothing, boundary intervention, or post-hoc repair |
| selection | none; the already registered D089 pilot-informed cutoff `realm_npe_mean < 4.5625491142` is reported descriptively but cannot promote or select this row |
| stop | stop after the single step-100 validation in every eligible, ineligible, or infrastructure outcome |

The step-100 model snapshot is retained only to bind the captured validation
row. It omits optimizer/scheduler/RNG state, declares `resume_supported=false`
and `selection_eligible=false`, and is not a continuation checkpoint.

## Capture And Event Semantics

The write order is immutable:

1. Execute the exact pre-gate D089 H29 validation and existing domain-link
   diagnostics.
2. Construct the complete step-100 training/validation row.
3. Reject missing flags, non-boolean flags, NaN/Inf JSON, or non-object payloads
   as instrumentation failures.
4. Atomically write `validation_row.json`, read it back, and require exact
   payload equality.
5. Call the unchanged `d089.require_eligible_validation` on that validation.
6. Write the eligibility result and stop. An unexpectedly eligible row also
   stops; it cannot enter selection or authorize continuation.

The registered flags remain distinct:

| Flag | Pass semantics | Failure semantics |
| --- | --- | --- |
| `all_normalized_finite` | every recurrent normalized proposal is finite through H29 | at least one normalized proposal is nonfinite; the parent normally stops before a complete row |
| `all_decoded_finite` | every decoded deployed proposal is finite through H29 | at least one decoded proposal is nonfinite; the parent normally stops before a complete row |
| `all_released_state_admissible` | released species are nonnegative and temperature/density are positive at every case/call | at least one registered released-state invariant-domain condition fails |
| `all_bounded_10x_train_max` | every decoded channel/case/call remains within the frozen 10x train-maximum envelope | at least one channel/case/call exceeds the envelope |

Returned scientific flag failures are not infrastructure failures. A load,
hash, source, runtime, replay, OOM, or exception before a complete returned row
produces no scientific event claim. Released-state admissibility is not full
composition conservation or complete physical validity. Boundedness is not
accuracy; H29 `realm_npe_mean` is not admissibility or finiteness.

## Retained Artifact Inventory For A Later Replay

Only these outputs are registered:

| Artifact | Purpose |
| --- | --- |
| `contract.json` | frozen D090 replay and stop policy |
| `parent_identity.json` | exact D089 artifact/checkpoint identities |
| `input_manifest.json` | exact open-data binding |
| `source_manifest.json` | canonical executed source chain including adapters and replay helper |
| `runtime_manifest.json` | exact D089 runtime replay binding |
| `replay_trace.json` | step-51--100 sample/LR/loss and final model identity |
| `step100_model.pt` | nonresumable, nonselectable diagnostic model identity |
| `stage_manifest.json` | frame-0, truth, prediction, coordinate, shape, dtype, and recurrence digests |
| `validation_row.json` | full pre-gate step-100 validation row |
| `eligibility.json` | unchanged D089 gate outcome and exact failed-flag list |
| `summary.json` | compact claim-bounded outcome |
| `final_hash_manifest.json` | SHA-256 inventory of all preceding retained outputs |

No rollout arrays, animations, test outputs, optimizer checkpoint, repair arm,
or residual artifact may be retained.

## Stop/Go Gates

Before optimizer step 51, stop if any parent file/hash/inventory, checkpoint,
model-state, normalizer, history, input, source, runtime, device, dtype,
parameter-count, output-directory, or test-absence check differs.

During replay, stop without a scientific row for any sample-order drift,
nonfinite proposal/loss/gradient/parameter, model/data device mismatch, or
unexpected exception. At validation, write and verify the full row before the
eligibility call. Stop after that call regardless of outcome. There is no retry,
resume, continuation, selection, population change, horizon change, threshold
change, output repair, or residual arm.

## A1 Authorization And Closeout

The owner authorized A1 preregistration, implementation, and synthetic CPU
testing on 2026-08-13. A1 explicitly excluded real trajectory and normalizer
arrays, loading a checkpoint into a model or executing it, CUDA initialization,
remote work, the sealed test population, PlanarDet, and the residual arm. Local
retained JSON, file hashes, and checkpoint mapping metadata were audited
read-only to bind the parent identity; no tensor was evaluated.

Long-lived A1 files:

| File | Reason to exist |
| --- | --- |
| `scripts/time_dependent_no/capture_realm_ignithit_domain_link_failure.py` | isolated, knob-free replay/capture entry point; existing trainers cannot persist the row before D089's gate |
| `tests/time_dependent_no/test_capture_realm_ignithit_domain_link_failure.py` | synthetic provenance, persistence-order, event, stage, and stop-policy gates |
| this file | line-specific scientific and authorization source of truth |

Exact executable identities before the reviewed-file commit:

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `scripts/time_dependent_no/capture_realm_ignithit_domain_link_failure.py` | 29,027 | `ba34d70d65ac3b573d77a57fd9a5e3e9286149f018db5478ac21c3ae207387b3` |
| `tests/time_dependent_no/test_capture_realm_ignithit_domain_link_failure.py` | 15,866 | `32221d9abb3037c2377a67a360e00c9a53f611ce5f2b26550b19749363eb3c16` |

The D090 contract digest is
`a7a72e3a00fc801440cf8ab828be3dfd76db1d1582e09876d50cd65b7761f382`.
The canonical D090 executed-source manifest digest is
`0ea9cbd4f7ed8bfc42d516c841fd9e4402903125302d1b16059341e17354400c`.

Focused synthetic CPU checks pass `13/13`; Ruff check and format-check pass and
safe CLI help exits zero. The complete maintained REALM synthetic CPU suite,
including D090, passes `122/122`. No scientific result exists under D090 yet.

## Smallest Next Authorization Request

The smallest next request is **one A3 diagnostic replay only** of this exact
D090 contract on an owner-selected idle GPU with the retained D089 parent
outputs. It runs exactly steps 51--100 and one H29 validation, then stops. It
does not authorize a retry, a second seed, continuation past step 100, model
selection, test access, PlanarDet, output repair, or the residual arm.

Cost class: one GPU, expected under five minutes, with a hard 15-minute
wall-clock cap. The retained D089 runtime payload must match byte-for-byte
before step 51; a different software stack or GPU identity is a preflight stop,
not permission to relax the runtime binding. The registered command form is:

```text
python scripts/time_dependent_no/capture_realm_ignithit_domain_link_failure.py \
  --manifest <exact-open-manifest.json> \
  --data-root <exact-open-IgnitHIT-root> \
  --normalizer-arrays <exact-P1b-normalizer-arrays.npz> \
  --parent-output-dir <exact-D089-output-directory> \
  --output-dir <new-empty-D090-output-directory>
```
