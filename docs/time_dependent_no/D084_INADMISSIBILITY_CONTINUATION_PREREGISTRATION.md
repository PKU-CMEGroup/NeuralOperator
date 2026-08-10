# D084 Finite-Inadmissibility Continuation Preregistration

Status: **complete and closed; contract frozen before GPU execution and executed unchanged**.

## Question and scope

D084 asks whether Euler inadmissibility in the current serious bump checkpoints
is followed by recovery, persistent finite error, or a registered numerical or
global-error explosion. It does not assume that a negative density, internal
energy, or pressure is itself a blow-up. It does not change the physical
boundary policy, add a positivity method, or reopen a sealed/test population.

The diagnostic keeps the checkpoint-native causal nodal policy fixed and
records its existing computation explicitly:

1. close the recurrent input under the registered boundary policy;
2. evaluate the frozen PCNO raw proposal;
3. close that raw proposal under the same policy;
4. feed the deployed proposal back, even when it is finite but inadmissible;
5. stop only after a deployed nonfinite conservative state.

There are no floors, clipping, smoothing, limiters, future-reference boundary
values, forward hooks, or altered boundary-condition arms. The bump
reconstructed node weights remain quadrature proxies, not validated physical
volumes.

## Exact population and binding gates

- Population: the existing 30-case open bump validation `rollout_keys` shared
  exactly by D072 `bump_s20260718_N0` and D082
  `bump_s20260718_BR1`; `test_keys` must be empty.
- Horizon: initial state plus all 79 stride-one model calls.
- Precision and batching: checkpoint-selected BF16, batch one.
- Repeats: two complete executions of every case/variant.
- Checkpoints: exact `best.pt` bytes bound by each run summary.
- Normalizer: D082 and N0 normalization dictionaries and digests must match.
- Boundary policy: D082 and N0 boundary contracts must match, and every rebuilt
  per-case policy digest must equal the stored digest.
- Source: active PCNO, rollout, runtime, and model files must hash-match the
  immutable D082 source snapshot. The new diagnostic files receive separate
  hashes.

Stop without scientific interpretation for any checkpoint, split, data
manifest, normalizer, source, or boundary-policy mismatch. Because no hooks are
registered, a hook/no-hook activation-equivalence gate is not applicable.
Two-repeat disagreement is retained as numerical-sensitivity evidence rather
than silently selecting one favorable repeat.

## Frozen matrix

| Variant | Checkpoint | Only changed quantity |
| --- | --- | --- |
| `N0_correct` | D072 N0 | none; no boundary representation |
| `D082_correct` | D082 BR1 | none; all semantic fields present |
| `D082_zero_all` | D082 BR1 | wall, outflow, and inflow fields set to zero before every call |
| `D082_zero_wall` | D082 BR1 | wall field set to zero before every call |
| `D082_zero_outflow` | D082 BR1 | outflow field set to zero before every call |
| `D082_zero_inflow` | D082 BR1 | inflow field set to zero before every call |

The D082 field-zeroing rows are checkpoint-local frozen input interventions.
D082 versus N0 is a trained-model comparison and is not a frozen intervention.

## Events and metrics

At `current`, `model_current`, `raw`, and `deployed`, record finiteness,
admissibility cause, minimum density/internal energy/pressure, invalid-node
count, proxy-weighted state-scaled relative L2, physical error RMS numerator,
maximum conservative amplitude relative to the full reference rollout, and
normalizer-scale excursions above 6 and 10. Normalizer excursions are
diagnostics, not proof of membership outside an unknown training distribution.

For every call, classify:

- input-projection recovery or introduction;
- model recovery or introduction;
- output-projection recovery or introduction;
- recurrence re-entry from an invalid deployed input to an admissible deployed
  output;
- first invalid node, bump node type, coordinate, and membership in the fixed
  semantic collar;
- raw and deployed increment error relative to the reference increment;
- input/output boundary-correction magnitude and outflow normal Mach.

Maximal inadmissibility episodes record start, end, duration, and recovery.
Registered explosion events retain the prior U1 diagnostic thresholds:

- definitive numerical blow-up: any nonfinite deployed conservative state;
- severe amplitude explosion: finite maximum conservative amplitude reaches
  100 times the case reference maximum;
- severe global-error explosion: proxy-weighted state-scaled relative L2
  reaches 10.

These thresholds are diagnostic, not physical. Report conditional descriptive
rates such as registered explosion among ever-inadmissible trajectories, but do
not call that association a causal effect of inadmissibility.

The strict prefix reconstructed from each unchecked continuation must also be
compared case by case with the checkpoint-selection rollout record. Differences
are reported; they are not repaired by choosing another repeat.

## Frozen animation rule

Retain all 80 temporal frames for cases `23`, `54`, and `128`:

- `54`: known D082 strict failure case;
- `23`: known D082 completion where N0 previously failed;
- `128`: prior cross-encoding bump visualization case with a late strict D082
  failure.

For these cases retain reference, `N0_correct`, `D082_correct`, and
`D082_zero_inflow` deployed states, plus D082 raw proposals and model inputs.
Animations must use all frames, fixed reference-only rollout-wide scales, and
explicit unavailable-frame markers after any nonfinite stop. They show truth,
predictions, errors, raw-to-deployed boundary correction, semantic fields, and
inadmissible-node markers. Per-frame or prediction-dependent autoscaling is
forbidden.

## Causal-language policy and decisions

Allowed statements distinguish exact stage attribution from temporal
association:

- If a raw proposal is invalid and its deterministic deployed projection is
  admissible, the existing output projection recovered that proposal.
- If an invalid model input produces an admissible raw proposal, the frozen
  model recovered that input on that call.
- Field-zeroing differences are causal only for that frozen checkpoint, input
  intervention, population, and recurrence.
- Recovery or explosion rates across naturally arising cases are descriptive;
  they do not identify the independent causal effect of inadmissibility.

Do not claim exact DG replay, physical conservation, a general Euler stability
result, or that inadmissibility is necessary or sufficient for instability
without the observed conjunction supporting it. Historical D019 evidence stays
provenance-limited, and historical L3R-B0 cannot be used because L3R-B0 was not
launched.

## Terminal result (2026-08-10)

All provenance gates passed for the exact D082 and N0 checkpoint bytes, the
shared open-data manifest and normalizer, the 30-case open-validation split,
all per-case boundary-policy digests, and the immutable D082 model/rollout
source. The six-variant, two-repeat H79 matrix finished in 651.530 seconds.
Every one of the 360 continuations executed all 79 calls with finite deployed
conservative states. No continuation reached a registered event: no nonfinite
state, no 100-times reference-amplitude threshold, and no proxy-scaled relative
L2 of 10.

| Variant | Strict completion r0/r1 | Mean survival r0/r1 | Strict error r0/r1 | Ever inadmissible r0/r1 | Maximum amplitude ratio r0/r1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `N0_correct` | 28/28 | 0.98354/0.98354 | 0.070582/0.070589 | 2/2 | 1.011/1.011 |
| `D082_correct` | 19/19 | 0.93333/0.93291 | 0.020166/0.020166 | 11/11 | 15.961/14.981 |
| `D082_zero_all` | 23/23 | 0.95105/0.95232 | 0.024857/0.024903 | 7/7 | 1.035/1.039 |
| `D082_zero_wall` | 19/19 | 0.94135/0.94304 | 0.024758/0.024836 | 11/11 | 1.034/1.035 |
| `D082_zero_outflow` | 19/19 | 0.93165/0.94008 | 0.020116/0.020298 | 11/11 | 7.287/38.993 |
| `D082_zero_inflow` | 23/23 | 0.94937/0.94810 | 0.020193/0.020191 | 7/7 | 25.939/31.194 |

All 98 repeat/variant/case rows that ever became inadmissible first failed
through nonpositive internal energy. Their first invalid node was always in the
fixed semantic collar: 30 were normal nodes and 68 were wall nodes; none was an
inflow or outflow node. Eight rows had at least one temporary recovery episode,
but all 98 were inadmissible again at call 79. Across both repeats there were
zero input-projection recovery calls, eight exact model recovery calls, and 20
exact output-projection recovery calls. Thus the frozen model or output policy
can repair a finite invalid state/proposal, but the observed repairs did not
guarantee durable admissibility.

Every trajectory whose amplitude reached 2, 5, or 10 times its reference
maximum had already become inadmissible, and the threshold crossing followed
the first invalid call. Many inadmissible trajectories never developed such a
large local excursion, however, and no trajectory developed a registered
global explosion. D084 therefore supports inadmissibility as a checkpoint-local
warning for possible later local growth, not as a sufficient cause or synonym
for blow-up. Historical U1 strengthens the counterexample: its one-node
negative-pressure episodes at calls 33--47 and 49--51 recovered at calls 48 and
52, and it remained finite through call 79 with final proxy error 0.03177.
Historical D019 remains only an association, but its raw evidence was recovered
on 2026-08-10: all five H79 HDF5 prediction/target pairs, the exact mapped
preprocessing arrays, and the named checkpoint bytes now pass hash and target-
binding checks. All 395 stored predictions remain finite. Density and pressure
never become negative; the positive-primitive exponential head instead
underflows exactly to zero. In every trajectory, speed first exceeds 10 twelve
calls before the first zero, fixed-reference amplitude first exceeds 10 two or
three calls before it, and unweighted global relative L2 first exceeds 1 two or
three calls before it. Thus D019's zero underflow is downstream of an already
large rollout error, not evidence that positivity loss initiated the instability.
This remains noncausal because the checkpoint's training epoch, parent,
selection rule, and exact training source are missing and no matched repair
counterfactual was run.

The frozen field interventions expose a tradeoff rather than a monotone
stability gain. Zeroing inflow gives five D082 rescues and one harm in each
repeat, for a net four additional strict completions, while strict error changes
only +0.13%/+0.12%. It does not remove local growth: its repeat maxima reach
25.94/31.19. Zeroing all fields also gains four completions but raises strict
error by 23.3%/23.5%; zeroing wall leaves completion unchanged and raises error
by 22.8%/23.2%. D082 versus N0 is not a frozen intervention: D082 reduces mean
strict error by about 71.4% but has nine fewer strict completions.

Population counts and mean errors are stable across the two repeats, but exact
case identities and calls are not. Only 141/180 repeat pairs match first-invalid
call and strict valid length; no deployed trajectory is bitwise identical, and
the largest common-finite state difference is 2024.27. Fresh execution shifts
D082/N0 completion by one case relative to checkpoint selection while changing
mean strict error by less than 8.4e-5. Exact case/call conclusions must therefore
be labeled BF16/CUDA-sensitive; a deterministic FP32 audit is the minimum next
test before causal state-repair interventions.

The canonical publication bundle contains six fixed-scale MP4s for cases 23,
54, and 128 with exactly all 80 frames, decoded 5 fps, 1430-by-638 output, and
no temporal or nodal-source subsampling. Full-node piecewise-linear fields use
21,629, 21,232, and 21,636 source nodes respectively. Mesh edges, nodal
markers, and ordinary boundary-tag overlays are not drawn; only inadmissible
nodes receive small markers. Maximum field saturation under reference-only
scales is 0.0607%. The canonical result-summary, result-manifest, and
publication-visualization-manifest SHA-256 values are respectively
`c3a4c20225fd9ca5be159f81bf07a143e36797d425027e2939fa837a385bc971`,
`45aefa9db4ff2c013007582a700b718f4cbf9afecdc42188df299e83505ebb7c`,
and
`e6296fcc4e49ebfe9f160956733989b2e34879afee953b36e105faaf72bfd344`.
The earlier 3,500-node diagnostic-scatter bundle
`39b28aa285b3428f74e0a919f830d057280836e5310ee57c71f549cbf18ba18c`
is retained as historical, not canonical, visualization evidence.
The recovered D019 retrieval manifest, frame-metric table, and analysis summary
have SHA-256 values
`54c8158c86a57b1d7aae5d3e1e94e538c35387d9cce137c39c39a47ad5807deb`,
`ccb9b6681c3164165669d7f8357b122e9ea883737a9d13c9a23d46223d36bad9`,
and
`86d95c5c6336c820ffdf4a0cfad44c9e3bc1449ce32aef3bf1f1f81074890aa9`.

D084 is terminal. It does not promote the D082 collar side branch and leaves no
active queue. Re-entry requires new owner authorization and, at minimum, a
deterministic FP32 audit before any matched minimal-admissibility-repair
counterfactual; neither follow-up is implied by this closeout.
