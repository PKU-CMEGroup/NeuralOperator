# D087 W26-L1 PCNO Long-Horizon Stability Preregistration

Status: **closed at the registered H79 scope on 2026-08-12. The A1 synthetic
contract, separately authorized H2 preflight, paired H79 survival evaluation,
and paired H79 fresh/propagated decomposition are complete. Immutable
replacement attempt `20260812e` passed every provenance, bitwise-prefix, and
algebraic-closure gate after the first decomposition attempt stopped before a
forward call. H160/H320, spectra, JVPs, training, bump-reference extension,
policy counterfactuals, and cross-benchmark work were not selected and form no
implicit queue. Reopening any of them requires a new preregistration and stable
identity.**

## Question, owner binding, and authorization boundary

D087 asks when and how two owner-designated bump PCNO systems cease to be
accurate, Euler-admissible, bounded, or numerically finite under autonomous
rollout. The owner bound the intended systems on 2026-08-11:

- stable system: the retained serious B1 epoch-22 checkpoint;
- unstable system: the recovered D019 positive-primitive checkpoint
  `artifacts/time_dependent_no/d019_recovered_20260810a/recovered/artifacts/time_dependent_no/pcno_ckpt/PCNO_forward_euler_exp_model.pth`.

The word *unstable* is an owner operational label. The animation observation
that D019 becomes out of distribution and blows up soon after an inadmissible
proposal is a hypothesis to time-resolve. It is not a frozen premise. D019's
training population and normalizer are unresolved, so this line uses
*reference-scale excursion*, not *OOD*, unless a later A2 contract first binds
an actual training-envelope statistic.

The qualitative observation is artifact-bound by
`artifacts/time_dependent_no/pcno_corrected_animation_gallery_20260706_selected/gallery_summary.json`,
SHA-256 `6a547abb78ad0efd81a4c828c409249121f066e88f649a80624342929f4b252a`.
Those stride-two normal-node GIFs are hypothesis-generating visuals, not an OOD
metric or a causal timing result.

This file originally registered A1 only: a pure synthetic event/recurrence
kernel, its CPU tests, and future A2 gates. The owner later authorized and
completed the exact H2 preflight and paired H79 evaluation recorded below. The
tracked A1 amendment itself loaded no checkpoint, dataset, rollout array, or
sealed split and changed no event, threshold, recurrence, population, or claim
definition. Every additional B1 or D019 forward call remains separately
authorized A2.

## Identity reconciliation

Names are not interchangeable:

| Name | Exact identity or role | D087 disposition |
| --- | --- | --- |
| P/D019 | Owner-designated unstable primitive direct-state checkpoint, 76,634,327 bytes, SHA-256 `31364c48b45b052ff45e2ca349d1ea711c29cb429d2b5cd68f223851472796b5` | Primary unstable system. Exact inference/checkpoint contract; incomplete training provenance. |
| R/D041 | Conservative-residual continuation epoch 4, SHA-256 `2bb5ee3ca831a6ffc498f01e309ae7ea957b2df0f411023fb3812919a6732964` | Historical comparator only. It is neither D019 nor B1 and is not run in the primary matrix. |
| L3R-B1 | Serious causal-boundary conservative-residual epoch 22, SHA-256 `221d12c3cd5f3df65546bb02537354fce334b481ab36478e7ff9f35f8e4323dc` | Primary stable system. Retained despite missing the historical outer D041-parity conjunction. |
| attached K2 | B1 continuation epoch 1, SHA-256 `830f81c4ff8cfb4a09b44919201a7dc64f0acd13341712e0c28469fff914e0b9`; parent is exact B1 | Separate derived checkpoint. K2D0 is useful prior propagated-path evidence, not part of B1's identity or the primary matrix. |
| D072 N0 | Exact D082 control checkpoint epoch 4, SHA-256 `c34c598f311b7636bbe21cdfbce521dbd16ca7972be10605f5dc3fe00e785fc3` | Separate same-family matched control; not the mentor's stable checkpoint. |
| D082 BR1 | Semantic boundary-residual checkpoint, SHA-256 `2ae807a89cf3de9f4e26d1c14997cc052c4babf1f6be4f19d4b199a89988d4b1` | Separate trained system; not B1 and not a repair of D019. |
| D084 | Frozen H79 continuation evaluator over N0/D082 and input-field interventions | Prior evidence and code audit only; not a checkpoint. Its hard-coded population, BF16 precision, and thresholds are not silently reused. |

The supersonic-bump systems above remain separate from every dynamic
finite-volume shock-vortex checkpoint and result.

## Exact native-system provenance matrix

The primary B1--D019 comparison is intentionally descriptive. Representation,
prediction target, boundary policy, training history, and architecture differ;
therefore it cannot identify which factor caused a survival difference.

| Field | B1 stable system | D019 unstable system | Comparison consequence |
| --- | --- | --- | --- |
| Checkpoint | `artifacts/time_dependent_no/l3r_b1_serious_20260728a/best.pt`; 229,972,835 bytes; epoch 22; SHA-256 `221d12c3cd5f3df65546bb02537354fce334b481ab36478e7ff9f35f8e4323dc` | Repository-relative path above; 76,634,327 bytes; SHA-256 `31364c48b45b052ff45e2ca349d1ea711c29cb429d2b5cd68f223851472796b5` | Exact bytes resolved. |
| Model/runtime source | Training source-set digest `61ea3d2c650a0d5d67b2020c4a97a08dc5ffdfb55498da0a3b26129437c5b730`; config digest `191748e7c1d08e7f59ef4c6882ca2565ca47efff27192b8cc7a75c1d412d2936`; runnable minimal-source archive `artifacts/time_dependent_no/l3r_bc0p_minimal_source_20260727a.tar.gz`, SHA-256 `14c80f64b7132f72526da6fb74d6b617d218e3e65cd7700e7cbbe4df62b52a67` | Historical inference core blob `pcno/pcno.py` SHA-256 `c5bb98fe736b370f277de935c88f6fb22efc23417a0dbab1bb9a08e979a6298b`, recoverable at Git commit `3ee2087dd8cd5a03d447a3b84eea96b43a43bb45`; exact checkpoint training source remains unresolved | The current core SHA-256 is `5b66c3574513422e85aeea10f2fd4f2a05033d6ea248b56d46fe3cb5e2519bbf` and is not called byte-identical. A2 must use the retained/reconstructed native bytes or pass an explicit map-equivalence gate. |
| Architecture and target | `PCNOEuler2DResidual`, 12 inputs, 19,155,720 parameters; normalized conservative residual added to current state | `euler2d_PCNO`, seven inputs `[x,y,node_rho,rho,v1,v2,p]`, 19,155,080 parameters; direct next primitive state; returned rho/p are post-head exponentials | Never send D019 tensors to a conservative-only diagnostic. The pre-exponential logits are not a physical proposal and are not collected. |
| Training data and split | 300 bump trajectories, 270/30 grouped split; data-manifest digest `5d5373fdcc682544bf330fba6d54fe65509dabe936baee7443c71a8c0c8d9fa7`; grouped-split digest `a887c43990265fdf30b04a0745056f2ad9df3db39135f4489a797fe7c44f786d` | Exact training data, split, and membership remain unresolved. Recovered evaluation NPZ SHA-256 `664ffaaaeaff82519b15617458ca6c05da72c44ec53e1342cfff648305c210b5`; mapping SHA-256 `57cbc497efe8f9716b801b3eaadee573e95674119e03085f089e18a315f26dbe` binds samples 0--4 to trajectories 0/6/11/13/17 | B1's 30 open keys are a *common declared evaluation population*, not known held-out cases for D019. The historical five D019 cases do not pair with B1's historical 30-case result. |
| Normalizer | Logical digest `d2d07a4000acc3cfd45ff105a19e7437177d553cee4c18d50858f5572de7efec`; retained file SHA-256 `717a948a1f219af9eabd6aafeacbdb8e2d00a362133e6a2b6b4408e60b71b5af` | Recovered evaluator applies no input/target normalizer; exact training normalization contract remains unresolved | Do not infer D019 OOD status. Cross-system metrics use a separately frozen common primitive view. |
| Precision | BF16 training; future primary evaluation is FP32, batch one | Training precision unresolved; recovered CUDA evaluator returned float32 arrays without a declared autocast path; future primary evaluation is FP32, batch one | Same A2 inference precision is required; historical BF16 D084 case timing is corroboration only. |
| Seed, optimizer, and history | Seed/split seed `20260718`; AdamW, peak LR `1e-3`, weight decay `1e-5`, clip `1`, 2% warmup then cosine to `2e-5`; 34/40 passes, 725,220 presentations, 183,600 steps | Exact seed, optimizer, epoch, parent, continuation, presentations, optimizer steps, and history unresolved. Historical source candidates are not checkpoint bindings | Training factors are confounded and cannot receive causal credit. |
| Recurrence | Conservative residual map with causal input closure, model proposal, causal output closure, then deployed-state recurrence: `P_B(M(P_B(U)))` | Model-returned post-exponential primitive proposal is multiplied by the padding mask; that deployed tensor is recurred. Proposal and deployed are exact on active nodes, so an invalid active-node proposal is fed back unchanged | Native recurrence is never performed through the common analysis adapter; padding zeroing is not a physical repair. |
| Boundary policy | `causal_nodal_physical` in teacher input, proposal, and recurrence; policy-set digest `c7d9f92dadeaef46bcb0c62c0f26458f08b7bfc6d6ec4e124aa840e9a246c13c`; no future truth, floor, clip, limiter, or smoothing | No explicit boundary encoding or repair; all active nodes predicted recurrently; no future truth, floor, clip, limiter, smoothing, or invalid-stop | Boundary policy is part of each native system. A policy-equalized counterfactual would be a new A2 registration, not the primary result. |
| Historical evaluator | FP32 B1 post-evaluation used evaluator SHA-256 `cf4da6cdbe508a34ebef5d2fa486350108ade098807e738ea6f0d7b0b72f8259`, H79, 30 open keys | Historical D019 driver blob `scripts/time_dependent_no/rollout_pcno_preprocessed.py`, SHA-256 `acd42fe099df9b416b253b72df6032e27037b9dd9a577462394da8fc5fb2a0f1`, recoverable at commit `f61f9133dda8294d068f5ef1084b5a95b8aa1710`; recurrence helper `scripts/2d_Euler_eq/rollout_visualization.py`, SHA-256 `9b4aee0e19427fc9ced3d722deb6ceca06a2dccae233bea31f7e6bf2d247465f`, recoverable at commit `ab407d1e85d9b0999405008c660125d8d8eb5130`; retained summary SHA-256 `9c37b07addcbc26a170c34b280ddae3d1869ada65eea5796aa8ed51a83280899` | The historical driver is absent from the current tree. Historical aggregate rows are not paired. Future comparison uses one new frozen D087 analysis evaluator but must preserve or prove equivalence to each native map. |
| Selection rule | Internal lexicographic validation/rollout `best.pt`; epoch 22 retained for study although no checkpoint passed the separate historical outer parity conjunction | Actual checkpoint selection rule unresolved | Call B1 “serious retained,” not a D041 replacement; call D019 “owner-designated unstable,” not a selected worst model. |

Any future scientific result row must serialize all provenance fields enforced
by the A1 kernel. Training source/data/split/normalizer/precision, exact executed
inference source, historical-native-map equivalence, evaluation data/evaluation
population, inference normalizer/precision, runtime manifest, and returned-stage
manifest are separate fields; one digest cannot conceal a gap in another.
`None` is accepted only when the same field is listed in `unresolved_fields`,
the row declares one of the permissions below, and a nonempty claim boundary is
present. Preflight records are separate artifacts and are not scientific result
rows.

### Provenance permissions

Every row declares exactly one ordered permission; a result cannot be promoted
after execution by relabeling its permission:

| Permission | Minimum exact provenance | Allowed interpretation |
| --- | --- | --- |
| `descriptive_survival` | Exact checkpoint bytes; executed-source manifest; evaluation data, population, and order; inference normalizer/precision; state representation; recurrence and boundary policy; evaluator; runtime manifest; and returned-stage manifest. Every unresolved training field and any missing historical-native-map equivalence are explicit. | Observed continuous trajectories, raw event calls, accepted prefixes, and survival under the executed system. No map-internal, mechanism, or training-factor attribution. |
| `inference_map_diagnostic` | Everything above plus an exact historical-native-map identity or a passing bitwise-equivalence report. | Exact deployed-map fresh/propagated, feedback, recovery-policy, finite-perturbation, or other inference-side diagnostics under the bound map. No training-factor attribution. |
| `training_factor_causal_attribution` | Everything above plus a registered matched or randomized training intervention with exact training source, data membership and order, split, normalizer, precision, seed, optimizer/scheduler/history, presentations, parentage, and selection rule. | Only the causal contrast identified by that separately frozen intervention. |

D087 registers no matched or randomized training intervention. Its A1 result-
row validator therefore rejects `training_factor_causal_attribution`
unconditionally; a later intervention requires a separate stable ID,
preregistration, and intervention-aware validator rather than a relabeled D087
row.

D019's unresolved training fields permanently forbid
`training_factor_causal_attribution` in D087. They do not forbid exact
`inference_map_diagnostic` rows once every inference-side field is bound or
passes the registered bitwise map-equivalence gate. Only an unresolved
historical-native-map equivalence may downgrade an otherwise fully execution-
bound row to `descriptive_survival`. An unresolved executed source, adapter,
normalizer, evaluation data/population, recurrence, boundary policy, evaluator,
runtime, or returned-stage binding invalidates the scientific row rather than
downgrading it. The B1--D019 cross-system contrast is never training-factor
causal under this ID.

`executed_inference_source_digest` is the SHA-256 of a canonical
`d087_executed_source_manifest_v1` JSON object. Its `files` entries are sorted by
POSIX repository-relative path and contain `path`, byte `size`, and lowercase
SHA-256 for every project-local model/core, driver, rollout/recurrence helper,
boundary/normalizer implementation, and primitive/conservative analysis adapter
actually used by the process. Serialize UTF-8 without BOM or trailing newline,
with sorted object keys and JSON separators `(',', ':')`, then hash those exact
bytes. Package/runtime versions belong in `runtime_manifest.json`.
`native_map_equivalence_digest` separately hashes the H2 identity/equivalence
report; it cannot substitute for the executed-source manifest.

## Verified prior evidence and the causal gap

The recovered D019 arrays contain 395/395 finite stored primitive predictions
through call 79. Density and pressure never become negative; the exponential
head underflows to exact zero. The order below is exact for trajectories
0/6/11/13/17:

| Event or proxy | First-call sequence | Accepted-prefix sequence |
| --- | --- | --- |
| unweighted global relative L2 at least 1 | 59/50/55/57/40 | 58/49/54/56/39 |
| fixed-reference amplitude at least 10 | 58/50/54/56/40 | 57/49/53/55/39 |
| first rho or pressure at most zero | 61/53/57/59/42 | 60/52/56/58/41 |
| fixed-reference amplitude at least 100 | 65/57/61/63/46 | 64/56/60/62/45 |
| native nonfiniteness | not observed through 79 | at least 79 |

Speed first exceeds 10 exactly 12 calls before the first zero in every case;
10x amplitude and unit relative error precede the zero by two or three calls;
100x amplitude follows it by four calls. This is compatible with the owner's
visual observation of rapid post-inadmissibility blow-up, while also
establishing within these five retained traces that large reference-scale
growth was already underway before positivity underflow. It does not establish
whether invalid feedback accelerates the later growth.

Prior B1, D041, N0, D082, and D084 rows use different checkpoints, policies,
precisions, populations, or accepted-prefix rules. In particular, a strict row
that stops before appending its first invalid proposal measures
`T_admissible`, not `T_finite`. D084's 360 continuations remaining finite
through H79 and D019's 395 finite stored states therefore remove the apparent
contradiction between “failed near H60” and “finite through H79.”

## Stability event schema

One model call maps frame `h-1` to proposal/deployed frame `h` and is indexed
from one. At every call retain four physical stages in checkpoint-native
coordinates:

1. `current`: the state actually fed from the preceding recurrence;
2. `model_input`: after the declared input/boundary policy;
3. `model_proposal`: the model-returned post-head physical proposal;
4. `deployed`: after the declared output/boundary policy.

For D019, `model_proposal` is already `[rho,v1,v2,p]` after exponentiation and
equals `deployed` on active nodes; `deployed=model_proposal*node_mask` on the
full padded tensor. No pre-exponential tensor is called a physical proposal and
padding zeroing is not a physical recovery. For B1, `deployed` is the causal
boundary-closed conservative state and is the only state recurred.

Let `E_X` be the first call whose deployed state fails event `X`. The accepted
prefix is always `T_X = E_X - 1`; failure at call one gives `T_X=0`. A later
pass is recorded as recovery but never reopens the accepted prefix. Independent
events may fail on the same call. If no failure is observed, `T_X` is right
censored at the last call on which `X` is observable.

The numeric `0.05`, `100`, and `10` cutoffs below are pilot-informed by prior
bump evidence. They are frozen prospectively only for newly executed paired
D087 rows; they are not confirmatory thresholds for the evidence used to choose
them. Continuous error, amplitude, scaled-RMS, and invariant-margin curves and
their raw crossing calls are primary. The historical D019 table above retains
its original unit-error and 10x/100x amplitude proxy events and must not be
retroactively labeled D087 `T_accurate` or D087's conjunctive `T_bounded`.

| Quantity | Deployed-state pass rule | Availability and censoring | What it does not imply |
| --- | --- | --- | --- |
| `T_accurate` | common-primitive proxy-weighted, component-scaled relative L2 at most `0.05` | Truth-backed returned calls only; a returned native nonfinite fails. H160/H320 calls without matching truth carry `null`, never a fabricated error, and censor at the last matching-truth call | Admissibility, boundedness, conservation, or physical validity |
| `T_admissible` | all active nodes have finite derived fields and strictly positive density, internal-energy density, and pressure | Available for every returned native state; a returned native nonfinite fails. Primitive channels are tested directly before any division | Accuracy, boundedness, boundary-contract validity, or conservation |
| `T_bounded` | native state finite, common-primitive fixed-reference amplitude ratio at most `100`, and fixed-reference scaled-RMS ratio at most `10` | Reference envelope is frozen from matching truth through H79, then reused unchanged for truth-free calls | Accuracy, admissibility, conservation, or asymptotic stability |
| `T_finite` | every active component of the native deployed state is finite | Available at every attempted call; the first native-nonfinite state is recorded and terminates that trajectory | Accuracy, admissibility, boundedness, or physical validity |

If a native model call returns a deployed tensor with any active-node
nonfinite component, that returned call fails `T_finite`, `T_admissible`, and
`T_bounded`, and also fails `T_accurate` when matching truth makes accuracy
applicable. At a truth-free call, `T_accurate` remains inapplicable rather than
being invented. An infrastructure failure before a deployed tensor is returned
-- for example process loss, OOM, unreadable input, or runtime exception -- is
an execution failure, produces no scientific event or censoring time, and
invalidates that run until owner-reviewed re-execution under the same frozen
contract.

Boundary-policy validity and proxy conservation drift are separate diagnostic
channels. They do not enter `T_admissible`. Bump weights are reconstructed
diagnostic proxy quadrature, not validated physical control volumes, so their
weighted sums are not called physical conservation.

Also report each checkpoint's established native paper-compatible rollout
error as a secondary channel: B1's proxy-weighted, state-scale-normalized
conservative relative L2 and D019's historical primitive per-channel/global
relative L2. Preserve each native formula and population label. These native
errors are not ratioed across checkpoints; the common-primitive metric above is
the registered cross-system accuracy event.

The common accuracy view is `[rho,v1,v2,p]` in float64 reductions. For component
`c`, case `j`, call `h`, active node `i`, and positive proxy weight `w_ji`, freeze

`s_c = max(sqrt(mean_(j,h)[sum_i w_ji q_jhic^2 / sum_i w_ji]), 1e-12)`

over the common H1--H79 truth. The callwise accuracy metric is

`sqrt(sum_(i,c) w_i ((qhat_ic-q_ic)/s_c)^2 /
      sum_(i,c) w_i (q_ic/s_c)^2)`.

For each case, freeze the largest H1--H79 truth `max(abs(q/s))` and the largest
proxy-weighted RMS of `q/s`. Prediction amplitude and scaled-RMS ratios divide
the corresponding callwise statistic by that fixed positive case envelope.
Serialize scales, envelopes, formulas, population keys, and their digest before
the first scientific model call. These truth statistics are metric-only and
never enter recurrence. Conservative-to-primitive and primitive-to-
conservative active-node round trips must close to `1e-12` in float64 on
admissible synthetic and sampled reference states.

The survival table is case-first. For each event and call it reports cohort
size, cases at risk, failures on that call, censorings after that call,
conditional survival, and Kaplan--Meier survival. It never switches to a mean
over whatever shorter prefix happens to remain valid.

## Matched evaluator and mechanism plan

### Primary native-system survival matrix

The future A2 primary population is the exact 30 B1 open `rollout_keys`, but it
may be used only if the D019 seven-input primitive/geometry contract can be
constructed for every same key without unstated metadata. The population is
called *common declared evaluation*, because D019 training membership is
unknown. Both checkpoints run FP32, batch one, start frame zero, stride one,
under their registered native recurrence and boundary policy.

The population order is frozen exactly as

```text
[7,16,18,23,47,54,58,60,72,82,101,103,112,120,126,128,141,145,
 150,172,187,188,190,211,227,233,235,251,287,296]
```

It is bound by B1 `split.json` SHA-256
`c3da16cbabc48bfa1038c775a1c99c5621900c7c6743e15e435afae964d6b073`
and `rollout_selection_seed=20262709`. These are B1 validation/rollout-
selection keys: B1's internal lexicographic validation and rollout rule used
this open population when selecting `best.pt`, and the separate outer D041-
parity conjunction was not passed. `test_keys=[]`; D087 opens no sealed or test
population. D019's training membership for every key is unknown, so D087 does
not call this population held out for D019 or use it to claim cross-system
generalization.

H79 is reference-backed. H160 and H320 use no matching retained truth and
therefore supply only `T_admissible`, `T_bounded`, `T_finite`, reference-envelope
excursions, boundary diagnostics where defined, and proxy internal statistics.
They supply no accuracy, defect, conservation, or physical-validity result.

H79, H160, and H320 are separate fresh frame-0 replays, not chained jobs. Each
longer replay must reproduce every retained native stage and deployed-state
digest in the shorter prefix bitwise before its suffix is interpreted.
Aggregate summaries, event tables, compressed diagnostics, and terminal scalar
records are never recurrence sources. H160 attempts every trajectory whose H79
native deployed state remained finite, even if it was inaccurate,
inadmissible, or unbounded; H320 applies the same rule to every H160-terminal
finite trajectory. A trajectory terminated by a returned native-nonfinite
deployed state remains terminated and is never restarted at a longer horizon.

### Fresh defect and propagated input error

For each truth-backed call and case, let `z` denote the fixed coordinates,
graph, proxy weights, node metadata, Mach/static channels, and boundary-policy
metadata. Define the full deployed native map, including the checkpoint's
input/output policy with `z` held identical, as `F_z`, and its common-primitive
analysis map as `G_z = C F_z C^{-1}`. Evaluate the same frozen checkpoint once
on its rollout input and once on the matching truth input. Record the exact
identity

`G_z(u_hat_(h-1)) - u_h =
 [G_z(u_hat_(h-1)) - G_z(u_(h-1))] +
 [G_z(u_(h-1)) - u_h]`.

The first bracket is propagated-input response and the second is fresh
one-step defect. Let `t`, `r`, and `d` denote the total, propagated-response,
and fresh-defect vectors, and define the active-node common-scale inner product

`<a,b>_W = sum_(i,c) w_i (a_ic/s_c)(b_ic/s_c) / sum_i w_i`.

In addition to the relative vector closure, require the exact weighted energy/
cross-term identity

`<t,t>_W = <r,r>_W + <d,d>_W + 2<r,d>_W`.

The registered relative energy residual is

`abs(<t,t>_W - (<r,r>_W + <d,d>_W + 2<r,d>_W)) /
 max(abs(<t,t>_W),
     abs(<r,r>_W) + abs(<d,d>_W) + 2 abs(<r,d>_W),
     float64_tiny)`

and must be at most `1e-12` after float64 reduction. The vector residual uses
the same weights/scales and the denominator
`max(||t||_W, ||r||_W + ||d||_W, float64_tiny)` and must also be at most
`1e-12`. Report absolute residuals, all three squared energies, the signed
`2<r,d>_W` cross term, magnitudes, and cosine by call, component, and frozen
region. If any required native output is nonfinite, or any required native-to-
common conversion is undefined, every decomposition field for that call is
`null` with an exact reason code; no clipping, imputation, or partial closure is
allowed. This decomposition is also unavailable without matching truth and
does not by itself identify a causal training factor.

### Authorized paired-decomposition execution amendment

On 2026-08-12 the owner separately authorized only the truth-backed paired H79
fresh/propagated decomposition. The immutable ignored attempt is
`artifacts/time_dependent_no/d087_w26_l1_pcno_stability_20260812d/`. It uses
the same two checkpoints, exact ordered 30-case population, frame-zero start,
79-call truth, source reconstruction, FP32 batch-one runtime, common component
scales, proxy weights, native recurrence, and native boundary policies as the
closed H79 packet. It executes no H160/H320 continuation, spectrum, filter,
perturbation, JVP, policy counterfactual, training, sealed population, new bump
truth, or REALM data/model operation.

Each system runs in one fresh worker. At every case and call, the worker first
evaluates `F_z(u_hat_(h-1))` on the autonomous current state and then evaluates
the same deployed map `F_z(u_(h-1))` on the matching truth state, holding all
coordinates, graph tensors, proxy weights, node metadata, Mach/static channels,
and boundary-policy metadata fixed. Thus the maximum authorized model-call
count is exactly `2 systems * 30 cases * 79 calls * 2 inputs = 9,480`. The
rollout branch's `current`, `model_input`, `model_proposal`, and `deployed`
records must match the completed H79 packet bitwise for every system, case, and
call. Call one must additionally have a bitwise-identical deployed prediction
from rollout and truth inputs and exactly zero propagated term. Any mismatch,
runtime/source/data/checkpoint drift, closure failure, worker failure, or
unexpected file closes this attempt without scientific interpretation or an
automatic retry.

The frozen node regions are `all_active`, `normal` (`node_type == 0`), and
`boundary` (`node_type != 0`); the zero-forward preflight requires both normal
and boundary nodes in every case. For each call, metrics are retained for all
components jointly and for each of `rho`, `v1`, `v2`, and `p` within every
region. Region/component cells retain the registered component scaling and use
the *full active-node proxy-weight sum* as denominator. Consequently the eight
atomic normal/boundary-by-component squared-energy and cross-term cells add to
the all-active/all-component value. That additivity, the vector identity, and
the weighted energy/cross-term identity must each close to relative tolerance
`1e-12`. The proxy weights remain diagnostic and are not physical control
volumes.

Alongside total, propagated, and fresh magnitudes and energies, record the
incoming common-coordinate rollout-input error norm and the pathwise secant
response ratio `||r||_W / ||u_hat_(h-1)-u_(h-1)||_W`. A zero denominator yields
an explicit unavailable value. This ratio is an observed response along the
realized rollout-error direction, not a JVP, Lipschitz estimate, stability
certificate, or causal attribution. `||r||_W > ||d||_W` is the sole frozen
strict propagated-magnitude dominance rule; equality is not dominance. Report
its first call per case, values at each first H79 event, continuous per-call
rows, event-aligned offsets `-8..+4`, and display-only fixed calls
`[1,5,10,20,40,60,79]`. No thresholded separation is promoted over the
continuous curves.

The closed packet contains only exact source/parent/population/reference/runtime
bindings, compact per-call stage digests and decomposition metrics, full-H79
prefix identity, provenance, summaries, execution receipts, and a closed final
hash manifest. It retains no full state tensor. An undefined or nonfinite
native-to-common conversion produces a reason-coded null decomposition row;
it is never clipped, imputed, or partially decomposed.

#### Closed pre-forward attempt `20260812d`

The zero-forward preflight passed, including exact parent-H79 rehash,
checkpoint/source/data/population/reference identity, the preregistration hash,
the normal/boundary partition, and the synthetic closure gate. The detached
launch then stopped before either worker or any model forward because the
process environment did not already export the retained deterministic-runtime
requirement `CUBLAS_WORKSPACE_CONFIG=:4096:8`. The external transport exit is
exactly `1`; `preflight.json` records `failed_no_scientific_result` and the
attempt's no-retry rule.

The ignored failure packet is the attempt root above. It contains exactly six
top-level JSON files plus separately bound staging: the executed-source,
parent, population, reference, source/data-preflight, and failure records. It
has no runtime manifest, worker output, returned-stage record, decomposition
row, scientific result, or final hash manifest. The exact bindings are:

- pre-forward preregistration SHA-256
  `2a20a7c02b0ab7378ab639341ec71fb834926dfebf2bd126100627d19ccd5a60`;
- executed-source internal digest
  `818f31f67ac60fba7b5ab8d5f713e4e2a8f5c7dfd82b959c8b00d76c7396f611`;
- run-local driver SHA-256
  `24226e585dd08f79ebd30803012e2c01c5fba6d04a53d04755b4760e38583c39`;
- zero-forward preflight SHA-256
  `a16a061a93b5fe9c4f53bbf02ffc7e9e2cfc9b79dac9233823c2439267692dea`;
  and
- failure-record SHA-256
  `b8d1eb5a9b9ac4161a50c32293d51c5fbe6f83b731ace83dceba3bbf09c07a4b`.

This is a harness failure, not evidence about B1, D019, fresh defect,
propagated error, or stability. A replacement requires a new immutable attempt,
a new preregistration hash, an idle-GPU check, and the same preflight with the
required cuBLAS environment exported before Python starts. No scientific or
metric definition is changed by that proposed correction.

#### Authorized replacement attempt `20260812e`

After reviewing the closed `20260812d` harness failure, the owner separately
authorized one new immutable attempt on 2026-08-12. Its ignored root is
`artifacts/time_dependent_no/d087_w26_l1_pcno_stability_20260812e/`. It freezes
the same two checkpoints, exact ordered 30-case population, frame-zero start,
79-call matching truth, source reconstruction, FP32 batch-one deterministic
runtime, common component scales, proxy weights, native recurrence, native
boundary policies, decomposition identity, closure tolerances, artifact
contract, and maximum `9,480` model calls registered above.

The only allowed launch correction is to export
`CUBLAS_WORKSPACE_CONFIG=:4096:8` in the detached process environment before the
Python interpreter starts. The attempt necessarily receives new immutable
attempt, preregistration, and executed-source hashes. It repeats the entire
zero-forward preflight, including the idle-GPU, parent-H79, checkpoint, source,
data, population, reference, partition, and synthetic-closure gates, before any
worker or model forward. No scientific definition, threshold, population,
horizon, seed, precision, recurrence, or boundary policy changes.

This authorization covers only the paired H79 fresh/propagated decomposition.
It does not authorize an automatic retry, H160/H320, spectra, perturbations,
JVPs, policy counterfactuals, training, sealed data, new bump-reference
generation, or any REALM data/model operation. Any failure closes `20260812e`
and returns to owner review without another model call.

### Recovery-policy attribution

At `current -> model_input -> model_proposal -> deployed -> next current`,
record each stage's native finiteness, common admissibility, margins, and exact
state digest. Classify input-policy, model, and output-policy transitions as
introducing, preserving, or repairing inadmissibility. A “recovery” row must
state whether an invalid model proposal or invalid deployed state was exactly
fed into the next call.

D019's padding-only policy predicts that a finite invalid active-node proposal
is deployed and fed back unchanged even though the full padded proposal is not
identical to the deployed tensor. Record full-tensor equality, active-node
equality, and whether proposal/deployed differences are padding-only. B1
predicts that only its boundary-projected deployed state is fed back. Finite-
invalid states continue; a native-nonfinite deployed state is recorded once and
stops. No floor, clip, smoothing, limiter, or hidden replacement is permitted.

### Pre-failure margins and spectra

For every case with an event, retain calls `E-8` through `E+4`, clipped to the
available range, plus matched calls from the other checkpoint. Report minima
and 1/5/50-percentiles of density, internal energy, and pressure; invalid-node
counts; common accuracy/boundedness channels; and boundary-local versus normal-
node strata. These scalar margins come from already retained primary-stage rows
and do not authorize an additional model call.

Spectra are exploratory and are not executable under this base contract. Before
any spectrum-specific or perturbation/JVP-specific model forward call, a
separate no-forward-call mechanism addendum must be reviewed, hash-bound, and
frozen. It must fix, without discretionary choices during execution:

- the state/update/error/defect quantities and exact graph operator;
- every band edge and endpoint-inclusion rule;
- Lanczos depth, tolerance, reorthogonalization, degeneracy handling, and
  implementation hash;
- radius-estimation method, iterations, initialization, and seed;
- boundary, shock, smooth, largest-error, and control masks, including the
  source fields, thresholds, morphology, empty-mask behavior, and all tie-
  breaking rules;
- event-onset precedence, case/control selection, offsets, clipping, and ties;
- every perturbation component/direction, normalization, seed, sign, epsilon,
  and direction-reuse rule; and
- perturbation-admissibility predicates, backtracking/rejection behavior, and
  the exact condition that makes a row unavailable.

The addendum must also bind source/evaluator hashes and a cost/output cap. Error,
fresh-defect, and propagated-error spectra are available only through H79 where
matching truth exists. Truth-free H160/H320 may carry a separately declared
state or update spectrum, but never an error or defect spectrum. All such rows
remain graph-spectral proxy diagnostics, not Fourier causality, physical
dissipation, or confirmatory mechanism evidence.

### Local perturbations and JVPs

Localized finite perturbations and JVPs are exploratory and remain blocked on
the no-forward-call mechanism addendum above. The addendum, not an outcome-time
choice, freezes the calls, regions, directions, signs, epsilon ladder,
admissibility-preserving rule, controls, and onset alignment. When the exact
native map supports `torch.func.jvp`, compare its JVP with the addendum's frozen
centered finite differences and apply its frozen linearization-validity rule;
otherwise emit an unavailable reason rather than substitute another direction
or scale. JVP norms and localized gains are local sensitivity diagnostics, not
causal proof, stability certificates, global Lipschitz bounds, or evidence that
one branch caused a rollout failure. Because the addendum may be informed by
the primary survival result, its output remains exploratory under D087 even
when every numerical gate passes.

## Stop/go gates

### A1 synthetic gate

All must pass before any A2 request:

1. primitive and conservative valid states share a common view;
2. primitive zero density/pressure is finite but inadmissible and is never sent
   to the conservative-only diagnostic;
3. first-event, simultaneous-event, recovery, and censoring semantics obey
   `T=E-1`, and missing truth is a terminal suffix;
4. D019 deployed recurrence proves invalid active-node proposal feedback while
   separating padding-only changes; a synthetic physical output repair proves
   recurrence from deployed rather than the invalid proposal;
5. finite-invalid continuation succeeds; a returned native-nonfinite deployed
   tensor fails all applicable events and terminates, while an infrastructure
   failure before return produces no scientific event;
6. fresh/propagated vector and weighted energy/cross-term closures are each at
   most `1e-12` in both representations, and nonfinite or undefined conversion
   produces reason-coded null decomposition fields;
7. fixed-risk, case-first survival denominators are correct;
8. the three ordered provenance permissions accept only their exact declared
   fields: D019 training gaps reject training-factor attribution, an otherwise
   exact inference map permits deployed-map diagnostics, and any inference gap
   permits descriptive survival only;
9. `--help` and `--dry-run` expose no checkpoint/data path and write nothing;
10. focused CPU tests, Ruff, and `git diff --check` pass.

### A2 identity and H2 preflight

Stop before science if any resolved checkpoint, source/config/data/split/
normalizer/policy/evaluator binding differs from its registered value, if an
unresolved D019 training field is silently assumed rather than emitted as
unresolved, or for any mismatch in population keys, active-node order,
geometry, state representation, gamma, precision, or recurrence. The explicit
D019 training gaps forbid training-factor attribution but do not block an
otherwise exact deployed-map diagnostic.

The H2 smoke case is frozen as trajectory key `7`, the first key in the ordered
common population, at start frame zero and calls one and two. Run each
checkpoint twice in separate fresh processes under the identical manifest; no
other key or horizon is opened by this gate.

Before the first H2 forward call, write and hash-bind `runtime_manifest.json`.
The runtime contract fixes model parameters, model inputs, native proposals,
and deployed tensors to FP32; disables autocast; disables both CUDA-matmul and
cuDNN TF32; enables deterministic PyTorch algorithms; sets cuDNN deterministic
true and benchmark false; performs conversions and metric reductions in
float64; and sets Python `random`, NumPy, Torch CPU, and every Torch CUDA seed to
`20262709`. The manifest records the exact Python, NumPy, Torch, CUDA runtime,
CUDA driver, and cuDNN versions; OS/platform; device name, UUID, compute
capability, and count; checkpoint parameter dtype; every input, adapter,
proposal, deployed, derived-field, and reduction dtype; autocast enabled state
and dtype; both TF32 flags; deterministic-algorithm and warn-only states; cuDNN
deterministic/benchmark flags; `CUBLAS_WORKSPACE_CONFIG`; every seed and RNG-
state digest; and the evaluator/process start method. A missing, unsupported,
or changed runtime field fails H2 rather than being filled after execution.

Also stop if:

- D019's seven inputs cannot be built exactly on all common keys;
- the B1 minimal-source archive and the three historical D019 runtime blobs
  cannot be reconstructed in an ignored immutable staging tree at their exact
  registered hashes, or a proposed maintained replacement fails bitwise H2
  native-map equivalence against those bytes, **when requesting
  `inference_map_diagnostic` permission**. If the executed evaluator and
  exact executed source, runtime, and returned-stage tensors remain hash-bound,
  a missing historical-native-map equivalence binding may instead produce a
  `descriptive_survival` row with that gap named; all map-diagnostic and
  mechanism outputs are then disabled;
- any common key is sealed or future truth enters recurrence;
- active-node common/native round trip exceeds `1e-12` in float64;
- two fresh-process deterministic FP32 H2, batch-one executions are not
  bitwise identical at every retained physical stage, or any stage link fails
  exact declared recurrence;
- the reference metric envelope is nonfinite, zero, or not hash-bound before
  H79 execution.

Only a passing H2 preflight permits the paired H79 matrix. A contract failure
produces a preflight report, no scientific result, and no retry under D087
without owner review. The sole downgrade path is the explicit inference-gap
path above: it permits only a newly executed descriptive survival record, never
an exact-native-map or training-factor claim.

### Executed H2 preflight and post-close packaging amendment

The owner separately authorized the exact H2 gate on 2026-08-11. The ignored
packet is
`artifacts/time_dependent_no/d087_w26_l1_pcno_stability_20260811a/`.
`preflight.json`, SHA-256
`a34666359829fd1c7fb982c935ca65a02b835522858ca6de957902fbfb3eb379`,
records `pass_h79_permitted_but_not_authorized` and
`scientific_result=false`. The zero-forward source/data preflight bound the
executed-source digest
`9032c93e0c2fbaba6b8f74a485150262721b2e39f63dd128ebd89199e837a952`,
population digest
`f614ceae2fcc803ab3245d7b454862fbcab0b7dc30e0fd12b4993ab80d7333ea`,
and evaluation-data digest
`6ca2a20b1d3f7b425e0f9b670e60509233d1cd4a41313fa8d7d2305e98845cf0`.

The runtime manifest SHA-256 is
`658fb7021b04b0a23286f1b042d4a61f0f6aeaa7333b306940fba9f72198daaa`.
On one RTX 5090 under the registered deterministic FP32 contract, B1 repeat
one, B1 repeat two, D019 repeat one, and D019 repeat two each completed calls
one and two in a fresh Python process with return code zero and empty stderr.
The returned-stage manifest SHA-256 is
`ff17aaa68ea213d389472435f1b1e36e35d78f3dc54a7b1132c9d0c2d33224c5`;
its internal digest is
`14f385004d19ea4332330c211f19473a70ba5bd5a408f2c121959a84fed6d1ce`.
It records bitwise duplicate identity. Independent revalidation established
that every recorded tensor was finite FP32, every call-two current digest
equaled its call-one deployed digest, D019 proposal and deployment were exact
on all active nodes, and B1 recurred only its boundary-closed deployed state.
H2 computed no event horizon, accuracy row, or checkpoint-comparison result.

One auxiliary packaging defect does not alter those hash-bound scientific
JSONs but must not recur. The original final manifest scanned the artifact root
while `tee` still held `h2_screen.log` open. Fourteen of its fifteen entries
rehash; the sole mismatch is that log, recorded as the empty-file SHA-256 and
zero bytes before the final 80-byte status line was appended. The wrapper exit
receipt was written after the manifest and was not listed. The completed H2
packet is retained byte-for-byte with this discrepancy declared; it is not
retroactively rewritten or rerun.

For every later D087 attempt, the scientific root has a closed, explicit, flat
top-level file inventory. Live `tee` logs and wrapper exit receipts remain in a
sibling ignored transport directory until the process exits and are not
scientific-result files. Immutable source staging remains separately bound by
the executed-source manifest. Only after every scientific writer closes may
the caller build `final_hash_manifest.json`; the manifest excludes itself,
rejects duplicates, path traversal, symlinks, missing files, and every
unregistered top-level file, checks that each file stays unchanged while it is
hashed, and is rebuilt and compared after writing. A mismatch produces an
infrastructure failure, not a scientific event. This amendment changes artifact
finalization only and therefore does not allocate a new scientific ID or alter
the completed H2 map/evaluator identity.

### Executed paired H79 result

The owner separately authorized the paired H79 matrix on 2026-08-11. The
ignored scientific packet is
`artifacts/time_dependent_no/d087_w26_l1_pcno_stability_20260811c/`.
`final_hash_manifest.json`, SHA-256
`ce3ec1a962162088069b1f0d293ffdcea5a83716a59ed6d8da8cf117b26d1e9b`,
closes exactly 18 listed scientific files plus the self-excluded manifest. A
second local rehash found no missing, extra, size-mismatched, or digest-
mismatched file. The external transport receipt is exactly zero. The compact
aggregate is `scientific_result.json`, SHA-256
`b56866307d87d5b93de68afb58da9fcebcbdbd117bb0794b75eb07899fa6d706`;
the case-first event table is `trajectory_events.json`, SHA-256
`dbc25f86f1d1f528420ed9e207426bbecfddc11fc4e546381288e0b878125181`.

Both fresh workers completed the exact ordered 30-case population with 79
calls per case under the frozen batch-one FP32 runtime. The H79 prefix gate is
bitwise exact against retained H2 repeat one for key `7`, calls one and two,
and every `current`, `model_input`, `model_proposal`, and `deployed` tensor.
Independent post-close checks found zero case-order, row-count, call-sequence,
event-formula, observed-recurrence, or terminal-feedback-semantics mismatch.

| System/event | Failed cases | Right-censored cases | Minimum / median / maximum accepted prefix |
| --- | ---: | ---: | --- |
| B1 `T_accurate` | 9 | 21 | `66 / 79 / 79` |
| B1 `T_admissible` | 0 | 30 | `79 / 79 / 79` |
| B1 `T_bounded` | 0 | 30 | `79 / 79 / 79` |
| B1 `T_finite` | 0 | 30 | `79 / 79 / 79` |
| D019 `T_accurate` | 30 | 0 | `5 / 9.5 / 12` |
| D019 `T_admissible` | 30 | 0 | `38 / 51 / 67` |
| D019 `T_bounded` | 30 | 0 | `42 / 55 / 71` |
| D019 `T_finite` | 0 | 30 | `79 / 79 / 79` |

B1 has the later accepted prefix on all 30 paired cases for accuracy,
admissibility, and boundedness; finiteness is tied and right-censored at H79 on
all 30. This is a descriptive native-system comparison, not a matched causal
factor contrast. B1's common-coordinate endpoint relative L2 has mean
`0.0472728`, median `0.0469773`, and range `0.0364968--0.0572195`; its separate
native paper-compatible mean endpoint is `0.0328470`. D019's common-coordinate
endpoint relative L2 has median `2.89767e11` and range
`60.5821--8.64229e16`; its differently defined native paper-compatible mean
endpoint is `3.37126e15`. The two native errors are not ratioed.

The event order refines the animation hypothesis. D019 first fails the frozen
accuracy cutoff at calls `6--13`, then first becomes inadmissible at calls
`39--68`: accuracy departure leads inadmissibility by `29--57` calls, median
`41.5`. Every first inadmissible row has nonpositive pressure/internal energy
from exponential underflow; density is also nonpositive in four cases. The
first invalidity is at the boundary in every case, while normal-node
inadmissibility follows after `0--5` calls, median `2.5`. The first boundedness
failure follows inadmissibility after `3--5` calls, median `4`, and is triggered
by the `100x` amplitude rule in all 30 cases, not by the `10x` scaled-RMS rule.
All D019 deployed states remain native-finite through H79. Thus rapid
post-inadmissibility reference-scale growth is verified, but predictive
departure begins much earlier and neither OOD nor numerical nonfiniteness is
established.

Recovery attribution is exact. D019 has 839 invalid deployed rows. For all 809
whose next call is observed, the finite invalid active-node proposal/deployed
state is fed back unchanged; the 30 terminal rows correctly retain unknown
feedback rather than false. There is no input, output, boundary, clipping, or
limiting repair. B1 instead has 137 raw model-proposal inadmissibility rows
across ten cases, all boundary-only pressure/internal-energy violations at
calls `2--44`; its registered causal output boundary closure repairs all 137
before deployment, so no invalid deployed B1 state is fed back. B1 case `296`
crosses the accuracy cutoff at call 70, passes it again at call 71, and fails
again at call 72. That is threshold recrossing of an always admissible,
bounded, finite deployed trajectory, not recovery from invalid feedback, and
its accepted prefix remains 69.

These observations establish an inference-side pathway: unrepaired D019
boundary inadmissibility is recurrently propagated and is followed quickly by
interior inadmissibility and large amplitude. They do not establish that this
feedback uniquely causes the growth. Representation, learned map, training
history, and native boundary policy remain confounded. The paired
fresh/propagated result below localizes the realized error pathway but does not
remove those causal confounds. Pre-failure spectra, local perturbations/JVPs,
and any policy-equalized counterfactual were not executed and receive no result
by implication.

### Executed paired H79 fresh/propagated result

After the owner reviewed the closed pre-forward `20260812d` harness failure,
immutable replacement `20260812e` completed on 2026-08-12. Its ignored packet is
`artifacts/time_dependent_no/d087_w26_l1_pcno_stability_20260812e/`.
`final_hash_manifest.json`, SHA-256
`1479a69cc7fe93b7b5ee48be4ddb959231587c903c2d9772300e54b4c5bbba39`,
closes exactly 15 listed scientific files plus the self-excluded manifest. A
local post-download rehash passed for every listed size and SHA-256. The
external preflight and decomposition exit receipts are both exactly zero. The
compact result is `scientific_result.json`, SHA-256
`4c221d9ba48205266be2b949dc49bd769db7c42bc4e2d0b58f1caaae58f7f2e1`.
The attempt was bound before its first forward to preregistration SHA-256
`aedb764310c096f78e75759053dc0c114c2758385cd47a5de57f94e604c1ba75`
and run-local driver SHA-256
`5575c8e48fa67e74190361e7d54311f99d6f04fcdcc0960ae85a2d0af14d1517`;
this post-result section necessarily changes the live documentation hash but
does not change the executed contract.

The two sequential fresh workers each completed 30 cases, 79 calls, and 4,740
model forwards, for the registered total of 9,480. All 2,370 decomposition rows
per system are available. The 18,960 replay-stage comparisons and all 60 equal-
input call-one predictions are bitwise exact against the closed H79 parent.
The full-prefix record SHA-256 is
`56430faeab09eead9dae2fd7433893f2a012e88129b13ef8b0e737da9ee67a01`;
the returned-stage manifest SHA-256 is
`36cc4183cd8a2adb3c9db6aee656fc39be910e8bfebec3f9c33911855e1c0389`.
The maximum weighted vector, weighted-energy, and atomic-additivity relative
residuals are respectively `2.41095e-16`, `8.64602e-16`, and `3.83085e-15`,
all below the frozen `1e-12` gate. The exact runtime record SHA-256 is
`b064e914bfc217a4eb5924b3d30c4cdc0fc6c2dc78b7e334ecb9d6a65dc6d0ba`.

| System / call | Mean fresh magnitude | Mean propagated magnitude | Mean total magnitude | Mean propagated share | Mean pathwise secant ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| B1 / H5 | `0.02043` | `0.02104` | `0.03072` | `0.5007` | `0.7777` |
| B1 / H20 | `0.02287` | `0.04887` | `0.05437` | `0.6833` | `0.9359` |
| B1 / H60 | `0.02129` | `0.08306` | `0.08575` | `0.7958` | `0.9807` |
| B1 / H79 | `0.02112` | `0.10257` | `0.10485` | `0.8291` | `0.9941` |
| D019 / H5 | `0.01809` | `0.03853` | `0.04268` | `0.6759` | `1.1029` |
| D019 / H20 | `0.01430` | `0.27775` | `0.27815` | `0.9493` | `1.0497` |
| D019 / H60 | `0.01587` | `4.19848e5` | `4.19848e5` | `0.9982` | `2.1992` |
| D019 / H79 | `0.06188` | `7.37004e15` | `7.37004e15` | `0.999995` | `3.4253` |

Propagated magnitude first strictly exceeds fresh magnitude at calls `4--8`
for B1 (median `6`) and calls `2--3` for D019 (median `3`). D019's mean fresh
defect remains only `0.0138--0.0181` at the frozen H5--H60 display calls, while its propagated
term grows by more than seven orders of magnitude over the same interval. At
the D019 accuracy event (calls `6--13`), the mean propagated share is `0.8414`
and the realized pathwise secant ratio is `1.1453`. At first inadmissibility the
corresponding values are `0.9957` and `1.3701`; at first boundedness failure they
are `0.9988` and `1.4096`. This directly supports the registered interpretation
that compounded response to the already erroneous recurrent input, not a rising
fresh one-step defect, dominates D019's realized long-horizon error pathway.

The registered spatial/component summaries refine, but do not causally close,
that interpretation. At D019's first inadmissibility and boundedness events,
the boundary carries mean fractions `0.6850` and `0.6966` of total error energy,
and `v1` carries `0.9230` and `0.9359`; the fresh-defect boundary fractions are
only `0.0372` and `0.0410`. B1's nine accuracy-event rows instead have a mean
fresh-defect boundary fraction `0.7286`, but only `0.0374` of total error energy
at the boundary, and B1 remains admissible and bounded through H79. Therefore a
boundary-local fresh defect alone is not sufficient to explain the native-
system separation. The exact learned map response plus recurrence/boundary
policy determines how that defect is propagated.

These are exact deployed-map diagnostics, not a JVP, Lipschitz estimate,
asymptotic stability certificate, conservation result, OOD test, or causal
training-factor attribution. D019 training provenance remains unresolved and
the two systems differ simultaneously in representation, learned map, training,
recurrence, and boundary policy. A brief separately owned GPU process started
after D087's idle-GPU launch gate and ended during the B1 worker. It did not
alter the exact runtime manifest, and every retained D087 rollout stage still
matched the closed H79 parent bitwise; it is nevertheless retained as a timing/
resource caveat rather than omitted.

### Result-to-claim closeout

The local result-to-claim verdict is **yes, with high confidence, for the
registered inference-map diagnostic**: under the two exactly executed native
systems on the common declared population, B1 remains accurate much longer and
remains admissible and bounded through H79, whereas D019's realized error path
is dominated by response to an already erroneous recurrent input rather than a
rising fresh one-step defect. The verdict is **no** for a causal claim about
training, state representation, residual prediction, boundary information, or
general neural-operator stability. Those factors were not independently varied.
This local verdict remains pending any separately approved independent review
of the unpublished result packet.

The minimum decisive paired H79 experiment and its registered decomposition
follow-up are complete. More bump frames without matching truth would add only
survival/admissibility evidence and are not required to support the accepted
claim. D087 is therefore terminal rather than an active queue.

### H79, H160, H320, and mechanism disposition

- H79: completed on both checkpoints and every common key under the frozen
  contract. All four deployed-state events and stage transitions are retained;
  no contract mismatch occurred. The paired fresh/propagated decomposition also
  completed for every H79 row with exact parent-prefix and algebraic closure.
- H160/H320: not selected under D087. Truth ends at H79, so these extensions
  would provide survival/admissibility evidence only. Any future extension must
  receive a new stable identity and preserve the registered fresh-frame-zero,
  bitwise-prefix, finite-terminal continuation rules rather than resume a
  compact summary.
- Spectra and perturbations/JVPs: not selected under D087. A future mechanism
  study must freeze graph/radius/bands, masks, onset, tie-breaking, directions,
  seeds, signs, scales, admissibility behavior, and linearization checks in a
  new no-forward-call preregistration. JVPs remain sensitivity diagnostics, not
  causal proof.

No outcome authorizes training, repair, policy counterfactuals, sealed data, or
cross-family claims.

D087 introduces no refreshed general stability claim from secondary
literature. Any later theoretical interpretation must cite the relevant
primary paper directly and must still keep finite-horizon boundedness,
accuracy, Euler admissibility, proxy conservation, and physical validity
separate.

## Decision table and minimum decisive experiment

| Evidence pattern on the common population | Supported interpretation | Still unsupported |
| --- | --- | --- |
| `T_accurate` separates before the other three | early predictive departure with later numerical/physical survival | cause of the departure |
| `T_admissible` separates while `T_finite` does not | finite physical-domain exit, explaining strict-stop differences | numerical blow-up or necessary instability |
| `T_bounded` separates before `T_admissible` | reference-scale growth precedes positivity loss | OOD without a training-envelope contract |
| `T_finite` separates | native numerical survival differs under the registered systems | asymptotic stability or physical validity |
| propagated term rises before fresh defect | compounded-input response dominates that common-coordinate error window | global contraction or training-factor causality |
| fresh defect rises first | one-step defect dominates that window | unique architecture/representation cause |
| output/input repair precedes a reported recovery | policy-mediated recovery | raw-map self-recovery |
| high-band fraction or localized gain rises first | spectral/local sensitivity warning | causal proof; JVPs remain diagnostic |

The minimum decisive experiment was one paired FP32 H79 evaluation of exact B1
and D019 on one hash-bound common declared population, with native recurrence,
four stages, four event times, and exact feedback attribution. It is complete,
as is the truth-backed fresh/propagated follow-up on the same traces. Boundary
policy, representation, and training history remain confounded and receive no
causal conclusion.

## Artifact inventory, code ownership, and cost

A1 touches exactly these tracked files:

- `scripts/time_dependent_no/evaluate_pcno_long_horizon_stability.py` -- pure
  synthetic schema and dry-run entry point;
- `tests/time_dependent_no/test_pcno_long_horizon_stability.py` -- focused CPU
  contract tests;
- this preregistration.

It does not modify `pcno_rollout.py`, `pcno_inadmissibility.py`, the historical
decomposition script, or the frozen D084 evaluator. The new kernel reuses the
maintained Euler conversions and `stage_transition_label`. It replaces neither
the strict rollout evaluator nor D084. In particular, the existing rollout
aggregate's mixed-valid-prefix fallback is not a matched survival estimator,
and D084's H79/N0/D082/hard-coded-threshold surface is not generalized in
place.

The executed and historical ignored A2 roots follow the pattern
`artifacts/time_dependent_no/d087_w26_l1_pcno_stability_<date><attempt>/` and
contains only `preflight.json`, `provenance_matrix.json`,
`runtime_manifest.json`, `population_contract.json`, `reference_contract.json`,
`returned_stage_manifest.json`, per-call compact stage/event rows and state
digests, trajectory event summaries,
survival tables, per-horizon `prefix_identity.json`, decomposition/spectrum/JVP
summaries where authorized, execution log, native paper-compatible error rows,
and a final hash manifest. The final manifest uses the evaluator's explicit
closed-file builder and post-write verifier; it is never made by scanning a
directory that contains a live writer. Screen logs and wrapper exit receipts
live in a sibling ignored transport directory, outside this scientific root.
Full rollout arrays are not retained by default; any bounded pre-failure tensor
bundle needs an explicit size cap and manifest. Nothing under either ignored
root is committed.

Cost classes:

| Phase | Cost | Authorization |
| --- | --- | --- |
| A1 synthetic implementation and focused CPU checks | XS local CPU | Authorized and completed, including post-H2 packaging amendment, on 2026-08-11 |
| A2 exact binding plus one-case H2 duplicate smoke for two checkpoints | S local/AutoDL GPU inference | Separately authorized and completed on 2026-08-11; preflight only |
| paired 30-case H79 plus compact diagnostics | M single-GPU inference | Separately authorized and completed on 2026-08-11 |
| paired 30-case H79 fresh/propagated decomposition | M single-GPU inference; exactly 9,480 model forwards | Separately authorized and completed as immutable replacement `20260812e` on 2026-08-12 |
| truth-free fresh frame-0 H160/H320 replays | M single-GPU inference; longer prefixes are recomputed for bitwise identity | Conditional and not authorized |
| selected pre-failure spectra/JVP directions | M, capped before launch | Conditional and not authorized |

## A1 freeze hashes and verification

The post-H2 packaging amendment freezes:

- evaluator SHA-256: `a27cea96e87bfc661bff15dc81e92c4f57e1c8225a1c8ab3b0b41ba430b6691e`;
- focused-test SHA-256: `b0614334c787878a1efe88284de7f71854cda34c0727035b4268fd4a7ec850d6`;
- base repository HEAD before A1: `4d90436ffd4d9fd27797a8551b232196253ea623`.

Verification on 2026-08-11: `46 passed` in the focused synthetic CPU suite;
Ruff lint and format checks passed. The new manifest tests prove that a live
top-level transport log is rejected, an external closed transport log does not
enter the scientific inventory, post-hash mutation is detected, unsafe names
and manifest self-inclusion are rejected, and the dry-run still writes nothing.
No checkpoint, dataset array, rollout, GPU, remote process, or sealed population
was accessed by this A1 amendment.

Closeout verification on 2026-08-12 reproduced `46 passed` in `7.36s` with an
explicit ignored basetemp. Ruff lint and format checks passed, the module help
path returned without writing, `git diff --check` passed, and all 15 files in
the closed `20260812e` scientific manifest rehashed with zero mismatch. The
evaluator and focused-test hashes above remained unchanged.

Changing the event definitions, thresholds, representation adapters,
recurrence semantics, population rule, or claim boundary after the first real
forward call requires a new stable ID.
