# P0: Restart Sufficiency And Native-Coarse Solver-Bias Gate

Date: 2026-08-25; A0--A2 closeout updated 2026-08-26

Status: A0/A1 are complete; the explicitly authorized A2 execution completed
and failed both factor-of-four solver-bias gates on all three fixed cases. The
native-coarse map is closed as the trusted target proxy for this registration;
A3, training, checkpoint reselection, test, and strength-OOD access did not run

Owner: time-dependent neural-operator project

## 1. Decision

The proposed deployment-response diagnostic requires a numerical map that can
advance the exact state queried by a learned model. That map is not presently
available for the supersonic-bump bundle: the released graph-node primitives do
not determine the Trixi-DG volume and surface state needed for an arbitrary-state
restart.

The first fallback candidate is the dynamic shock-vortex native-coarse finite-
volume map. Its state is the same row-major conservative cell-average array used
by the neural models, but it is not automatically the training target. The
models learn fine-grid evolution followed by restriction; the candidate solver
evolves the restricted state directly on the `250 x 100` grid. This document
asks only whether that native-coarse map is accurate enough to serve as a
solver-relative response proxy.

A pass authorizes design of a separate response-bank experiment. It is not a
response-diagnostic result, a model ranking, a training decision, or evidence
that the native-coarse solver equals the fine-evolve/restrict map. A failure
closes this proxy without rejecting the deployment-response question.

## 2. Current Read-Only Audit

The current implementation accepts an arbitrary finite, admissible native-grid
state with shape `[25000, 4]`, advances it with WENO5--HLLC--SSPRK3 and adaptive
CFL, and records rejected steps, reconstruction fallbacks, boundary exchange,
and density/pressure minima. It applies no state clipping, floor, projection, or
decode--encode repair. Invalid reconstructed face states can invoke an explicit
first-order face fallback; inadmissible time steps are retried at half the step
size. Both mechanisms are part of the declared map and must be reported.

Current audited source hashes are:

| Member | SHA-256 |
| --- | --- |
| `utility/time_dependent_no/shock_vortex_fv.py` | `619abb3688137594682af8a43e298c1b2eaf16d59f211fc1d60058c57f058091` |
| `utility/time_dependent_no/shock_vortex_coarse_cfd.py` | `0c4cfd116fed1018990b7239c13976086a144f773900ad52a955c07582e83cfd` |
| `tests/time_dependent_no/test_shock_vortex_coarse_cfd.py` | `188f003ab3e6a2962e7491ccff158304101c94fbef17c5818f4f734ee8336a97` |

These are audit-time hashes, not an execution manifest. Any implementation
change must produce a fresh source manifest and make the change explicit before
an A2 launch.

Historical D051 is feasibility evidence only. Its retained summary has schema
`pcno_shock_vortex_coarse_cfd_error_cost_v1` and SHA-256
`9e2208d2b4ad3450f78a603129887d16bb2bd83963e00fc2527b89bde0af857e`.
At `250 x 100`, its mean full-horizon native-coarse error was `0.00808583`
versus D044 PCNO's `0.00834190`, at about `30.14x` the calibrated PCNO cost.
D051 started from initial conditions and did not test arbitrary stored-state,
one-stride restart. Its own strict error--cost claim was false, so it cannot
satisfy this gate.

The candidate model records are:

| Map | Common-horizon definition | Expected checkpoint SHA-256 | Retained summary SHA-256 |
| --- | --- | --- | --- |
| D044 | compose the stride-1 residual map twice | `c5e468c7045bf5ff8ccdd5f222c19bd63dab15af54b0461a58e26af5e17f678f` | `ded10f92fb6eba3a3ee665a66c2ce37ecb9ec9173654305877c2f3e8b52a2612` |
| D060 | apply the stride-2 residual map once | `95e6c180a3298c4b38662d8c6cb77301c0e7fd0286e9a53571bddc487b8379c9` | `50ad14177f1efd1154124c56a74224102331af5cc9d1da914e179847a2d1f2e2` |

Both summaries bind data-manifest digest
`f8d228ae3c6e08fe6697f37df21476fe621abc354d0b0a6eed2a623ed2b9f96c`,
open grouped-split digest
`ae4be7f0e5fcfb305106173625c62c9a96634a716061207afb35fc9bda724d5a`,
and source-artifact-set digest
`88915ecaa742a04f9dc5e470b7beb347ffcbde68f57400c3066307a4a35e5ac3`.
Their shared training-only conservative-state scales are
`[0.0752340287, 0.0546168404, 0.0435805767, 0.2345080528]`.

At the local-only audit boundary, the compact records did **not** retain the
checkpoint bytes or trajectory arrays, and their recorded checkpoint paths were
not available locally. D044 also lacked a complete executable source hash
inventory, while D060 was produced from a dirty historical checkout whose
recorded core and wrapper hashes differ from the current checkout. Those were
hard blockers for the local attempt, not details to infer or waive.

### 2.1 P0-A0 local continuation audit (2026-08-26)

The owner authorized continuation of the documented plan, so the local part of
A0 was repeated by content and inventory rather than by historical filename.
It remains a provenance stop:

- all 52 standalone checkpoint-like files under the retained artifact root
  were SHA-256 hashed (`6,873,764,799` total bytes); neither expected D044 nor
  D060 checkpoint digest is present;
- all 88 retained archive inventories were opened read-only. Eighty-seven
  contain no checkpoint-like member. The remaining archive contains 72
  explicitly named D094 B1-C2 checkpoints and no advertised D044/D060 member;
- the local artifact tree contains 882 array-like files, including derived
  traces, visualization payloads, and partial dynamic-family bundles, but no
  immutable full trajectory root bound to the frozen D044/D060 data and open-
  split contract;
- the compact family manifest, grouped split, D060 normalizer, model configs,
  and downstream summary bindings survive and rehash, but they cannot replace
  the absent checkpoint and trajectory bytes; and
- of the eight historical evaluator-source members named by the retained
  D044/D060 resolution summaries, only three match the current checkout. A
  compatible complete historical executable source tree has not been
  recovered.

The three preregistered native-coarse solver members still match the audit-time
hashes in the table above. No case was selected, no checkpoint or state array
was loaded, no model was constructed, and no A1/A2 numerical work ran. The A0
receipt therefore has status `failed_missing_exact_resources`, and the next
legal action is a read-only recovery search on an explicitly selected retained
compute resource. AutoDL is the recommended first search target because it was
the active high-capacity environment for this lineage; no remote search is
implicitly authorized by this local audit.

### 2.2 P0-A0 AutoDL recovery (2026-08-26)

The owner explicitly selected AutoDL for the next read-only recovery step. The
search recovered and independently rehashed:

- exact D044 and D060 `best.pt` bytes at the expected hashes, together with
  each run's frozen normalization, split, and training summary;
- the complete `135`-trajectory root: `1,893` files and `3,954,339,212` bytes,
  including all `1,755` manifest-bound arrays, with top-level manifest SHA-256
  `f8d228ae3c6e08fe6697f37df21476fe621abc354d0b0a6eed2a623ed2b9f96c`
  and exact `84/24/27` train/validation/test counts;
- separate completed evaluator receipts for D044 and D060, with SHA-256
  `66a7d30197e861820acad2ff178949e2301b4c66830c779f5b0575070bac8a14`
  and `625a07b91d8cd93354375e08493aaf2efdd62f00ad9d0308d1d8436032ff1dee`.
  Each receipt jointly binds its exact checkpoint, the data/split contract, and
  the same eight registered evaluator-source members; and
- one recovery-time static project-import closure of those eight members: `23`
  files, no unresolved project import, no dynamic import site, and canonical
  mapping SHA-256
  `9bda94598e09cdabb56731ccbcfe8c52bd1a829b6cb169b478ed0952f946e675`.
  This 23-file closure is a recovery-time extension, not a retroactive claim
  that all 23 hashes were registered during historical execution.

The current native-solver source was frozen in a distinct `10`-file transitive
root with canonical mapping SHA-256
`293af994cfd3656f9ea72120bec9d32a921c6f3a4939f71daa9ae67fdd48e8cf`.
The evaluator and solver roots must remain separate because they bind different
registered revisions of `shock_vortex_fv.py`.

Field-blind sorting selected `(sv_e06_y00, 10)`, `(sv_e11_y08, 30)`, and
`(sv_e05_y00, 50)` before their frame bytes were read. All are open-validation
members with frames through `f+4`; the promoted case-manifest file has SHA-256
`1452e03f905b1fd7a50ba7f2c09143faec259b5cab08e12a0d2538deebb554f7`
and references no historical-test or strength-OOD member.

The fresh ignored input root contains `1,945` manifested members totaling
`4,415,212,852` bytes. Its final artifact-manifest SHA-256 is
`14b5b48c8d6ea754341ece82cb1e3cf26ef53c350807b2620df89c1eb77d8eb7`;
the canonical member-mapping digest is
`393a5035ee9f32b38df8700d8f6808a005397a9fb3d46f6a498ea45139d29bda`.
Every input member was rehashed after promotion and has no write bit; the three
fresh output roots are empty.

No checkpoint was deserialized, no model was constructed, no solver or project
source was executed, and no training or scientific metric ran. The immutable
resource and compatibility recovery was complete. The formal A0 gate was not
yet closed at that recovery boundary because the receipt deliberately left A1/A2
commands null until an owner-authorized implementation is source-hashed and
reviewed; this literal command-binding requirement may not be waived. The next
legal action at that boundary was owner review of this recovery record, followed
only if approved by A1 implementation and command freezing. A2 remains
separately unauthorized at that recovery boundary.

### 2.3 P0-A1 synthetic and exact-input closeout (2026-08-26)

After explicit owner approval, a bounded A1 implementation was frozen and
executed on AutoDL with CUDA disabled and one CPU thread. Post-run review of the
first attempt found that it semantically validated `execution_contract.json`
and its frozen artifact record but did not independently rehash the physical
file. Its immutable closeout manifest
`686c71b185dee6449838ef651f5bb4fb93b857e62b975c169aac27d0bdc25302`
is retained as superseded infrastructure provenance and is not the promoted A1
receipt.

The corrected `P0-A1-v2` attempt adds the physical-member rehash and a regression
test under a fresh source, command, output, and attempt root. Its `13`-member
source manifest has SHA-256
`6ee2a5d27da37875e63b6b5cd35fcbf5e625914db292a3761fbafdb7a404a740`
and canonical mapping digest
`6027b7e0e802deae56d30e0f718f05fbc072f2ba659ddca54fbf151f36cd3d69`.
The exact two-command binding, which runs the focused test before the data-bound
preflight and records A2 as unauthorized, has SHA-256
`1ae74bd9cb4b480915b3ee07c9df271fc8ea08d69053d3f0bd09cd3be551f131`.

All `11` focused CPU tests pass. They cover row-major shape/channel replay,
dtype and shape rejection, admissibility rejection, the registered one-stride
timing, boundary-balance pass/fail behavior, fixed-case and population closure,
physical artifact-member rehash, source-hash drift, unmanifested Python, and
path traversal. The data-bound preflight then rehashed the unchanged A0
artifact manifest, physical execution contract, and only the three fixed
selected state files. The physical execution-contract file SHA-256 is
`b3d88b16d33b26797a7f1fdd324d5c8681797cd984e1de70f759331d82a963d2`.
For every case, the C-contiguous solver-input copy is still little-endian
float32 and its raw byte hash exactly equals the selected stored-frame hash:

- `sv_e06_y00/f10`:
  `6681c8c8008180d3a31ce3863ed608bb50b1872eb26fd03e70bb699336c6cbec`;
- `sv_e11_y08/f30`:
  `e7f4f4ffd8b4cf270a509034a1b6539df86e68b6393f0cb9e83283fd77772896`;
  and
- `sv_e05_y00/f50`:
  `b4a807dca89e71694c571bbb8bc979e0f1fabb64d27f87292dc7fdd6dea20878`.

The A1 summary file SHA-256 is
`1a57413f10040013034b09570cd6315580a5fe504ccdf67688d74fe8129cf182`,
with payload SHA-256
`addff7cb47f129d850226b7fdd1b596e91951d40e7d5b82b5290af9b64aaf330`.
The final execution receipt has SHA-256
`2fac2a8a1c5ad33d7de8796801dc3ceacecce450fa0eea3d90d394d1c742d21e`;
the `9`-member closeout manifest has SHA-256
`8fd2192d84e287da98c890a653dda2f367791ec8d3b8ca889cd13626257f93c6`
and canonical mapping digest
`35dfc745f07f19a9a3c2a2be4ad04e3ca2296835eb145f77864591f0a02b4c32`.
All `13` source members and all `9` closeout members were independently
rehashed after execution, and the execution/output roots have no write bit.

A1 opened exactly the three fixed state files. It did not open or deserialize a
checkpoint, construct a model, advance the native solver, train, field-select a
case, or compute a scientific metric. The formerly open A0 downstream
source/command-binding requirement is now satisfied by the immutable A1
receipts without rewriting the historical A0 receipt. A0 and A1 are complete;
A2 remained a separate owner decision at that A1 closeout boundary.

### 2.4 P0-A2 stored-state restart and bias-gate closeout (2026-08-26)

After explicit owner approval, A2 was executed on AutoDL under fresh primary
and repeat processes. The promoted `P0-A2-v4` source is commit
`4e22c7022df4d23a3376758bcce54cdfb0b44641`. Its `14`-member native source
manifest has SHA-256
`acb6d163c7fca679eba4e5a69b14aca96cfec83244b16e82df423406ec52548e`
and mapping digest
`cbe3625694cd01bdf46d33568c812c3c44283ce2c1c1f2c853d45af4e7d3cc78`.
The `26`-member historical-evaluator source manifest has SHA-256
`7ed7b15a67f4706719cd42de45925ce3b4763860a68a894efd056f929d3508ae`
and mapping digest
`1e013788494fb8ab4385473b02dda1830bda3892a94fddbc3a91463592cf2e53`.
The exact command binding, including deterministic cuBLAS workspace
configuration, has SHA-256
`e27c1ce8944a212c5f415d9837d4dce4ca641384afcd982ec381aab7eeb3dfb4`.
All `18` focused tests and the historical evaluator import gate pass.

Both solver processes made two calls per case. Same-process states and boundary
exchange were byte-identical. Primary and fresh-process counts agreed, and all
fresh-process FP64 state distances were exactly zero:

| Case | Accepted steps | Rejected / fallbacks | Boundary residual | Fresh-process scaled RMS |
| --- | ---: | ---: | ---: | ---: |
| `sv_e06_y00/f10` | 28 | 0 / 0 | `7.483e-15` | `0` |
| `sv_e11_y08/f30` | 28 | 0 / 0 | `1.315e-14` | `0` |
| `sv_e05_y00/f50` | 27 | 0 / 0 | `1.326e-14` | `0` |

The solver therefore passed completion, finiteness, admissibility,
repeatability, retry/fallback, and boundary-accounting gates. It failed the
registered accuracy qualification on every case; all denominators were finite
and resolved:

| Case | Solver bias `b` | D044 defect | D060 defect | D044--D060 separation `s` | `b/min(d044,d060)` | `b/s` | Pass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `sv_e06_y00/f10` | `0.01607056` | `0.01292325` | `0.01184779` | `0.01466799` | `1.35642` | `1.09562` | no |
| `sv_e11_y08/f30` | `0.02369133` | `0.02311327` | `0.02203681` | `0.02213283` | `1.07508` | `1.07042` | no |
| `sv_e05_y00/f50` | `0.02109930` | `0.02563764` | `0.02598512` | `0.02715236` | `0.82298` | `0.77707` | no |

This is not merely an admissibility failure: every solver, D044, and D060
proposal is finite and admissible. Descriptively, the solver has higher front
IoU than either learned map on all three cases (`0.945/0.950/0.947`), but it
produces thicker (`1.055/1.129/1.110`) and weaker
(`0.840/0.805/0.813`) shocks. The learned maps have near-unit thickness and
strength ratios but distinct bulk proposals. These family-local diagnostics do
not rank D044 against D060; they show why coarse-solver error cannot be treated
as a negligible common target bias.

Four implementation-only stops are retained under distinct attempt identities:
JSON serialization of solver diagnostics, missing deterministic cuBLAS workspace
configuration, an exact-float volume-contract error, and post-inference payload
serialization. No failed attempt produced a scientific gate result.
The six model proposals retained by the last stop are byte-identical to the
promoted v4 proposals, providing an independent deterministic rerun check.

The final model summary payload is
`392b6873d6eb154edb4d160be88c9caa1d4fad1f798ca7791ec4906fd394202d`.
The closeout receipt has SHA-256
`f02b2f608ce6ffd0da88fe990aa3fe6d85005651ff6c7ee70c67434f9a61fb9b`;
the `111`-member immutable closeout manifest has SHA-256
`fad5b6c39b1c619c348a719f624a5d830809d34a8dc317ac198229c66b351825`
and mapping digest
`a582d2d92717f844d4fe5385e756398b83ae8a35ea05689201cb002eac16455f`.
Exactly the two fixed checkpoints were deserialized; D044 made six logical calls
and D060 made three. No checkpoint selection, training, test/strength-OOD
access, or A3 execution occurred. Under the registered decision rule, this
native-coarse target proxy is closed and only descriptive solver-relative
evidence is retained.

Result-to-claim verdict (`[pending independent Codex review]` because unpublished
results were not sent to an external tool): `claim_supported=no` for the
intended native-coarse trusted-target qualification, with high confidence. The
results support a deterministic, admissible, boundary-accounted implementation
of the declared coarse map. They do not support treating that map as the learned
target or using it for A3 response attribution. The missing evidence is a
higher-fidelity arbitrary-state restart map, or a newly justified diagnostic
whose claim does not require one; the route is a scoped pivot, not an A3
supplement.

## 3. Frozen State And Time Contract

The common physical horizon is `Delta t = 0.02`:

- reference target: stored fine-grid evolution followed by restriction from
  frame `f` to frame `f+2`, with saved spacing `0.01`;
- native-coarse proxy: one `run_coarse_cfd_rollout` call with
  `output_times=(0.0, 0.02)` and `t_final=0.02`;
- D044 map: two raw residual calls with its own frozen normalizer; and
- D060 map: one raw stride-2 residual call with its own frozen normalizer.

The state order is `[rho, rho*u, rho*v, E]`, flattened by
`index = y * 250 + x` on `[0,2] x [0,1]`, with uniform cell volume, `gamma=1.4`,
and no model or solver intervention. The physical start time `0.01 f` is
recorded even though the audited x-extrapolation/y-symmetry boundary map is
autonomous and the solver uses elapsed time beginning at zero.

No primitive round trip, interpolation, smoothing, clipping, floor, limiter,
future-reference boundary value, or state repair may occur before either model
or solver sees the stored conservative array.

## 4. Field-Blind Open-Case Selection

A0 must recover the bound open-validation split without opening the historical
test or strength-OOD populations. Before reading any field values:

1. sort the eligible open-validation trajectory identifiers by
   `sha256("P0-RS-20260825|" + identifier)`;
2. take the first three distinct identifiers; and
3. assign frames `10`, `30`, and `50` in that order.

Every selected trajectory must contain frames through `f+4`. Failure of this
metadata-only requirement stops the attempt; identifiers or frames may not be
substituted after fields or outcomes are inspected. The resulting three
identifier/frame pairs and their input/reference byte hashes become the fixed
case manifest.

## 5. Metric And Gate

For conservative states `a` and `b`, define the primary physical-volume,
channel-scaled RMS distance

```text
E(a,b)^2 = sum_j V_j sum_c ((a_jc-b_jc)/s_c)^2
           / sum_j V_j,
```

where `s` is the shared training-only scale vector in Section 2. No case or
outcome-dependent rescaling is allowed.

For case `i`, let `u_i` be the stored frame, `u_i+` its frame-`f+2` target,
`S_c` the native-coarse map, `G_44` two D044 calls, and `G_60` one D060 call.
Record

```text
b_i  = E(S_c(u_i), u_i+)
d44_i = E(G_44(u_i), u_i+)
d60_i = E(G_60(u_i), u_i+)
s_i  = E(G_44(u_i), G_60(u_i)).
```

The proxy passes only if **every** case satisfies

```text
b_i / min(d44_i, d60_i) <= 0.25
b_i / s_i                <= 0.25.
```

A zero or numerically unresolved denominator fails the corresponding gate; it
is not replaced by an epsilon-selected pass. The factor-of-four margin is fixed
before outcomes because a proxy comparable to the model defect or model
separation cannot resolve the proposed response mechanism.

Also report, but do not substitute for the primary gate:

- unscaled and per-channel physical-volume relative L2;
- primitive density/velocity/pressure error;
- density, internal-energy, and pressure minima;
- integrated conservative change and recorded boundary exchange; and
- shock position, strength, thickness, and smooth-region error under the
  already audited family-local definitions.

For each baseline solver call, completion and all states must be finite and
admissible; rejected attempts and face fallbacks must both be zero. The relative
boundary-balance residual

```text
||Delta integral(U) + boundary_exchange||_2
/ max(||Delta integral(U)||_2, ||boundary_exchange||_2, 1)
```

must not exceed `1e-10` in FP64.

## 6. Staged Execution

### P0-A0: provenance recovery and preflight

A0 is read-only. It must:

- recover the exact two checkpoint files, full trajectory arrays, data/split
  manifests, normalizers, model configs, and source members into a fresh
  immutable input root;
- reproduce every expected digest above and create a transitive source and
  artifact manifest;
- establish an exact evaluator-compatibility receipt for both historical
  checkpoints; absence of the dirty D044 source state is a failure, not license
  to use the current model implementation;
- create the field-blind three-case manifest;
- bind OS, Python, PyTorch, CPU, thread count, dtype, deterministic settings,
  solver configuration, commands, and fresh output roots; and
- prove no historical-test or strength-OOD member is reachable through the
  selected manifest.

No numerical A1/A2 result is valid if A0 is incomplete.

### P0-A1: synthetic and exact-input plumbing

After owner approval of the completed A0 receipt, run focused CPU/FP64 tests for
shape/order/channel replay, admissibility rejection, one-stride timing, boundary
accounting, and manifest fail-closure. The solver input bytes must equal the
selected stored-state bytes. A1 may not load model checkpoints or field-select
cases.

### P0-A2: three stored-state restarts

After A1 passes, execute exactly the three baseline cases once in a primary
process and once in a fresh repeat process. The two processes must have equal
input/config/source hashes, accepted-step counts, rejected-attempt counts, and
fallback counts. Their FP64 outputs must have primary distance at most `1e-12`;
same-process repeated calls must be byte-identical. Any failure stops A2 without
tuning CFL, thresholds, WENO epsilon, retry policy, cases, or frames.

Compute the model defects and separation only after solver outputs and their
hashes are frozen. Do not select or reselect either checkpoint from A2 results.

Closeout: A2 completed, but both factor-of-four bias ratios failed on all three
cases. The registered native-coarse target proxy is closed.

### P0-A3: admissible displaced-state smoke

A3 is conditional on a complete A2 pass. For each model and case, define the
one-common-stride generated displacement

```text
eta_m,i = G_m(u_i) - u_i+.
```

Query the solver once from `u_i+ + alpha*eta_m,i` for
`alpha in {0.5, 1.0}`. Each fixed amplitude is either already finite and
admissible or is recorded as failed; no clipping, projection, rescaling,
backtracking, or replacement amplitude is allowed. Require the same
repeatability, completion, zero-retry, zero-fallback, and boundary-balance gates
as A2.

A3 establishes only that the proxy can advance a small, registered set of
reachable model deviations. It has no fine-reference truth at those displaced
states and cannot validate response accuracy.

A3 was not executed because A2 failed its prerequisite accuracy gate; it remains
unauthorized.

## 7. Required Outputs

Use one fresh ignored root per stage. Each completed stage must retain:

- preregistration and source-commit identifiers;
- transitive source, input, checkpoint, and output manifests;
- case-selection receipt and explicit no-test-access receipt;
- exact command/environment/configuration records;
- case-level metrics before aggregates;
- solver states, model proposals, and A3 displacements as ignored arrays;
- completion/failure records, including all retry/fallback counts; and
- a final artifact manifest that independently rehashes every retained member.

No raw fields, checkpoints, generated arrays, machine paths, credentials, or
private host information may be committed.

## 8. Decision Rule

| Outcome | Decision |
| --- | --- |
| A0 cannot recover exact checkpoint/data/source bindings | Stop. The historical pair is not executable evidence. |
| A1 fails replay, ordering, admissibility, accounting, or manifest closure | Fix infrastructure under a new attempt identity; do not run A2. |
| Any A2 completion, repeatability, retry/fallback, or boundary gate fails | Close the current native-coarse implementation as a response proxy. |
| A2 completes but either factor-of-four bias gate fails for any case | Close the native-coarse map as the trusted target proxy; retain only descriptive solver-relative evidence. |
| A2 passes and A3 fails | Do not build the reachable-deviation bank on this proxy. |
| A2 and A3 pass | The owner may review a separate diagnostic preregistration; no model training or target-H79 access follows automatically. |

The historical test population, strength-OOD cases, new training, longer D094
compute, FFNO/component/attention ladders, checkpoint reselection, and HydroGym
remain outside this authorization boundary.
