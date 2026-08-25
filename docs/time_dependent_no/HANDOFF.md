# Time-Dependent Neural Operators: Handoff

Updated: 2026-08-26

This is a replaceable operational snapshot. It records the current scientific
position, workspace boundary, and decisions still owned by the human researcher.
Run-level history belongs in the dated archives, not here.

## Authority Read Order

1. Repository-root `AGENTS.md` for branch rules and authorization boundaries.
2. [RESEARCH_DIRECTION_DECISION.md](RESEARCH_DIRECTION_DECISION.md) for the
   current claim surface and standing human constraints.
3. This handoff for the current workspace and unresolved decisions.
4. [MECHANISTIC_DIAGNOSTIC_TRACKER.md](MECHANISTIC_DIAGNOSTIC_TRACKER.md) for
   compact experiment-ID and topic routing.
5. [WEEKLY_RESEARCH_PLAN.md](WEEKLY_RESEARCH_PLAN.md) for the owner-selected
   phased programme, evidence registers, dependencies, and prospective gates.
6. [README.md](README.md) for onboarding, provenance, and maintained-code
   navigation.
7. Read one bounded section of the
   [2026-08-11 decision archive](history/RESEARCH_DIRECTION_DECISION_through_2026-08-11.md)
   or [evidence archive](history/MECHANISTIC_DIAGNOSTIC_TRACKER_through_2026-08-11.md)
   only when exact historical language, metrics, hashes, or contracts are
   required.

Current explicit human direction outranks repository planning snapshots.
Historical outcomes remain evidence; old forward-looking recommendations are
not permanent authorization or prohibition.

## Active Thesis And Priority

The active thesis is now **deployment-response fidelity of neural operators
under self-composition**, within the broader time-dependent-PDE project. The
working title is *Beyond One-Step Accuracy: Deployment-Response Fidelity of
Neural Operators under Self-Composition*.

The single intended main contribution is a solver-relative response diagnostic
measured on a common, cross-fitted bank of short-prefix reachable deviations
and tested prospectively against long-horizon rankings. It must add information
beyond fixed-validation one-step error and the longest early rollout prefix used
to construct the bank. Architecture, data diversity, exposure, compute, and
optimization phase are controlled levers, not independent thesis claims.

Immediate priority order:

1. review the
   [P0 restart-sufficiency contract](P0_RESTART_SUFFICIENCY_PREREGISTRATION.md)
   and recover its exact inputs before any solver execution;
2. freeze and test the diagnostic first on open development evidence, then on
   genuinely unseen model/seed panels before target H79 access;
3. only after that passes, select one mediator-targeted causal intervention;
   and
4. treat one HydroGym cylinder contract as an optional feedback capstone, not a
   new RL-algorithm programme.

The retained bump contract is a no-go for direct graph-state-to-Trixi-DG
restart; its graph-node primitives do not recover the DG volume/surface state.
Bump therefore remains the primary phenomenon system, and no graph-to-DG
reconstruction project is implied. The dynamic-FV fallback accepts conservative
native-grid cell averages, but its native coarse map is not automatically the
fine-evolve/restrict target used for training. It requires a solver-bias gate
before being described as anything stronger than native-coarse-solver-relative.
Boundary handling remains a frozen deployment contract unless the diagnostic
localizes a material boundary-response defect. HydroGym enters the thesis title
only after an unchanged diagnostic predicts true-CFD policy transfer beyond
one-step and reward baselines.

## Compact Scientific State

The credible paper direction is a mechanism-first study of why one-step
flow-map accuracy fails to determine behavior under repeated composition, and
whether solver-relative response on reachable deviations supplies the missing
predictor. It is not a generic residual-architecture, boundary-method, or
benchmark programme.

| Evidence family | Current bounded conclusion |
| --- | --- |
| 1D residual FNO | A useful fixed-family program exists. Larger-step advantages depend on horizon and metric; they are not a universal stability result. |
| CPGNet reproduction | The released 1D gain depends materially on message reach, and causal boundary training improves the bounded local 2D release-bundle comparison while leaving the oracle gap. Paper-faithful parity and exact DG replay remain unresolved. |
| Supersonic bump PCNO | D041 remains the historical comparator. D087 closes its exact H79 scope on B1 and owner-designated D019: B1 stays admissible/bounded/finite, while D019 loses accuracy first, later becomes inadmissible and unbounded, and remains finite. Exact decomposition shows propagated-input response dominates D019's realized late error. A separately registered H320 comparison on the same 30 open cases records later active-gradient PCNO first events than PCFNO more often than the reverse, but H80--H320 has no truth. These are bounded recurrence diagnostics, not causal factor attribution, physical validity, accuracy past H79, or general stability. |
| Bump data--architecture--optimization response surface | D094 B1-A/B1-B, the seed-0 ladder, B1-C4, B1-C5-A/B/C, B1-C2, B1-C3, and B1-C3-R1 are complete at their registered development scopes. On the three-seed exact-`64n` diagonal, mean online-train, fixed-seen, and fixed-validation one-step errors decrease at every adjacent count for both PCNO and PCFNO; PCNO has lower one-step error in all 18 seed-by-count comparisons in both precisions. Mean PCNO outside H79 is also monotone on this diagonal, but PCFNO outside H79 is not; per-seed H79 favors PCFNO in 3/3 seeds at `n=8`, PCNO in 3/3 at `n=16/32/256`, and is seed-inconsistent at `n=64/128`. This supports predictable one-step improvement under joint data/exposure/optimization scaling plus one-step/rollout non-equivalence, not pure data causality or a generic monotone rollout law. FFNO/components, conservation, 81,920 steps, and test remain separately gated. |
| Dynamic shock-vortex PCNO | D044 is the useful baseline and D060 is a useful one-seed improvement with unresolved structure error. Common-source resolution studies identify persistent large-scale mesh inconsistency plus locally cancelling shock/vortex defect. W26-L5 A32/A33 qualify one target-free raw-shadow-tethered affine protocol on reused open populations: on six retained-500x200 D074 cases, corrected versus raw transfer has full/rank-8 trajectory ratios `0.98887/0.94507`, full/rank-8 H30 ratios `0.97770/0.89007`, six of six trajectory/endpoint wins, and maximum control `1.00459`. Direct fixed-hop 500x200 has median H30 ratio `2.08418` versus corrected transfer and zero wins. A34--A39 reject universal coefficients/simple routers despite strong ordinary same-family fits. A41/A42 then identify a nearly diagonal accepted-shadow native response during coast. A43 uses that frozen response to remove 56 accepted-native calls and qualifies on the reused E12/E14 population: active full/rank-8/H30 ratios are `0.97877/0.93117/0.96249`, all four active trajectory/endpoint pairs win, and maximum control is `1.01141`. A44-R1 confirms the unchanged protocol on 14 new E00/E11 correction cases inside checkpoint training support: active full/rank-8/H30 ratios are `0.98216/0.95586/0.97873`, all four active pairs win, maximum control is `1.04361`, and large/rank-8 benefit coexists with neutral-to-slightly-harmful local bands. A45 then finds pooled logged-utility prediction and stable lag-history coefficients, but fails the required E00 and coast transfer cells; it strengthens the broad/rank-8 versus local-harm diagnosis without qualifying a router. This is bounded same-family correction-protocol evidence, not resolution invariance, independent checkpoint/test confirmation, family-independent coefficients/response, causal fine-refresh utility, physical conservation, an Euler1D deployment result, or a measured latency win. |
| Synthetic shock representation | W26-L2 P0/P1/P2 establish fixed-grid capacity followed by subcell-phase and finite-grid transfer failure. Retrained no-gradient and parameter-matched local arms roughly halve held L2 and reduce extrema/TV, but increase registered step-ripple mass and fail strict transfer/no-harm controls. P2-C0 is complete: zero-output gradient activation slightly lowers train loss but is a held-phase near tie, so it does not rescue the original full-PCNO path. P2-F is descriptive because its `1e-5` replay gate fails while the maintained `1e-4` ceiling passes: spectral transport is necessary, pointwise cancellation passes every seed, and the full model's differential path is acutely essential/cancelling in `3/3`, so the trained no-gradient gain is architecture reorganization rather than acute deletion. P2-W0 then isolates a moving-front wake: no native arm leaves error everywhere the front traveled; full PCNO has a held-pulse-only path-wide/recurrent wake in `2/3` seeds, no-gradient passes no wake stratum, and posthoc `/10` gradient scaling is catastrophic. The first defect is phase-sensitive and recurrence amplifies it. |
| REALM PlanarDet | D092-R1 completed one seed-0 width-96 residual-PCNO run and selected step 950. Truth-input/free H49 sums are `6.09925/88.13821`; P0b is near-null, and G0b/G1 localize a checkpoint-specific chemistry-to-density persistence asymmetry. D093 reuses the PCNO-7 result anchor and adds PCFNO/FFNO at three and seven unique supervised conditions. Each seven-condition cell has lower selected truth-input error than its three-condition counterpart, but only FFNO has a lower selected free-rollout sum; FFNO-7 is best at `1.36466/32.53910`. This is one seed and one open trajectory under a common residual contract but distinct D092/D093 source inventories, not clean data scaling, a paper-faithful baseline, sealed ranking, physical causal graph, architecture cause, or general PlanarDet claim. |
| Latent forecasting and assimilation | The tested latent forecast was not viable. Assimilation ideas remain reserved until an open-loop failure mode and target claim justify them. |

The branch therefore supports a mechanism-first neural-operator study, not a
learned conservative finite-volume solver, a generally shock-stable
geometry-aware architecture, state-of-the-art performance, or operator
convergence.

Keep the bump and dynamic-FV evidence separate. Bump node weights are diagnostic
proxies. Physical conservation language is allowed only for the audited dynamic
finite-volume contract with geometry and boundary exchange accounted for.

## Current Workspace And Authorization

The reviewed D094 source and launch infrastructure is committed through
`c0eef62` in the following focused stages; the local artifact transaction is
recorded separately below:

1. commits `b9bc125` and `180f70c` retire three orphan analyzers and require an
   explicit ADER dataset-generation flag;
2. commits `451e8dd` through `05f1989` integrate the v6 PCNO source-snapshot
   contract and the maintained REALM/D093 infrastructure;
3. commits `e100286`, `c8a2f21`, `5357694`, and `2504b65` preserve the H320,
   W26-L2, cross-resolution derivation, and A43--A46 provenance sources before
   any further pruning;
4. commit `f44ce92` anchors 28 superseded diagnostic files and `120617e`
   retires those same paths; and
5. commits `127f4f0`, `8f674ac`, and `901a839` integrate the stable local D094
   split, source registry, full-horizon selector, schedule, sentinel, and resume
   contracts without launching the next scientific gate; and
6. commit `066238b` adds the B1-A analyzer, explicit selected-versus-terminal
   metric receipt, B1-B outside-selection evaluator, finite-only structure
   diagnostics, focused tests, and the preregistered route to the full seed-0
   ladder; and
7. commits `110e659`, `8762e6d`, and `1a2a859` record the B1-B result, launch
   the isolated ladder, add the unbiased 12-checkpoint evaluator, and freeze the
   seed-0 result/route before retrieval; and
8. commits `35efd96`, `ef46167`, `1aa21e9`, `3baa6cd`, and `8cbf93c` close the
   seed-0 analysis, register B1-C4, add its fixed-checkpoint outside audit and
   visual diagnostics, make tensor export copy-safe, and verify portable H.264
   rendering; and
9. commits `a94c1aa`, `db3b4f7`, `fd3c7db`, `f105223`, `e3671ab`,
   `9440ea6`, and `b0be12c` close
   B1-C4, register and close B1-C5-A, preregister the paired FP32 B1-C5-B
   fixed-map evaluator/analyzer, and preserve its first engineering stop plus
   the tested retained-scale shape fix and corrected launch identity; and
10. commits `45bfc24` and `e5c8b70` preregister and implement the paired FP32
    B1-C5-C symmetric map--path decomposition; and
11. commits `0da7ad5`, `8a020c8`, `e90d7bb`, and `c0eef62` close B1-C3,
    register and close the B1-C3-R1 training invariant gap, and bind the
    exact-sentinel evaluator with its corrected source hash.

The audited artifact transaction removed six targets containing 81
archive-covered files and 21 directories, including nine empty D094 pytest
directories (`61,459,631` bytes). The ignored append-only receipt is
`artifacts/time_dependent_no/RETENTION_MANIFEST_20260821b.json`, SHA-256
`de4d33468b7501188b9bf18cb3cea94dc50b4354455de4d38f935684407598c6`.
All 40 checkpoint files and canonical evidence packets remain. Ambiguous or
source-bound artifact material was preserved; the exact replay gaps are
recorded below.
Use `git status` and the current checkout, not this prose, as the source of truth
for file status.

The cleanup authorization did not itself authorize training, dataset-scale or
checkpoint execution, remote work, downloads, sealed-population access, or new
scientific claims. The owner subsequently authorized and completed the W26-L2
P0 full-PCNO synthetic fit, P1 frozen checkpoint analysis, and exact nine-run P2
gradient-ablation matrix. P2 passed both restricted-arm primary gates but failed
both strict causal/no-harm gates. That approval does not extend to filtering,
loss/exposure, non-synthetic data, recurrent D073-B, REALM, or sealed
populations. On 2026-08-12 the owner separately authorized and completed the
exact six-run synthetic `W26-L2-P2-C0` continuation, its frozen final branch
cube, visualizations, monitoring, retrieval, and analysis on the available
cloud machine. The result is strict-inconclusive because its exact
first-gradient hash gate failed; the explicit post-closure descriptive analysis
shows slight train refit/held-phase harm and no resolution rescue. That
authorization does not expand the excluded scopes above. The owner then
separately authorized and completed the inference-only P2-F cube across the
nine original P2 checkpoints, including retrieval, manifest verification, and
seven PDF/PNG figure pairs. It performs no optimization and does not authorize
another GPU study. The owner subsequently authorized and completed the
inference-only synthetic P2-W0 moving-front wake diagnostic, including
residual truth/error plots, manifest-verified retrieval, and checkpoint-free
phase-separated visual analysis. It adds no training authorization.
The owner also separately authorized and completed D087's exact H2
preflight, paired 30-case H79 event/survival evaluation, and paired H79
fresh/propagated decomposition. The closed packet passes bitwise replay,
algebraic closure, and retained-manifest rehash gates. A separately registered
follow-up then completed matched PCNO/PCFNO recurrence through H320 on the same
30 open cases. All 43 members named by its five retained final manifests rehash,
but truth ends at H79. The [bounded H320 record](W26_L1_H320_REFERENCE_FREE_ROLLOUT_RECORD.md)
therefore closes that work as descriptive reference-free recurrence, not
H80--H320 accuracy, physical validity, conservation, asymptotic stability, or a
causal gradient result. No separate H160 identity, spectra/JVP, policy
counterfactual, or new-bump-truth branch is an implicit queue.

Machine paths, hosts, credentials, and private dataset locations stay in ignored
`LOCAL_CONTEXT.md`. Raw datasets, checkpoints, rollout arrays, media, and large
logs stay under ignored artifact storage unless a specific reviewed deletion is
authorized.

## Human Decisions Still Required

D094 B1-C3-R1 is now closed retrospective development evidence. The owner
reviewed the
[P0 restart-sufficiency and solver-bias gate](P0_RESTART_SUFFICIENCY_PREREGISTRATION.md)
and explicitly selected AutoDL for read-only A0 recovery. Exact D044/D060
checkpoints, all `1,755` manifest-bound arrays, the frozen normalizers/config
records, and separate complete checkpoint--evaluator compatibility receipts are
now recovered in a fresh read-only ignored root. The reconstructed historical
evaluator closure has 23 members and canonical digest `9bda9459...e675`; the
separate current solver closure has 10 members and digest `293af994...e8cf`.
The three field-blind cases are fixed at `sv_e06_y00/f10`,
`sv_e11_y08/f30`, and `sv_e05_y00/f50`. Formal A0 closure still requires exact
downstream command/source binding; no A1/A2 solver or model execution and no new
training is authorized. Boundary work remains conditional on a measured
boundary-response defect, and HydroGym remains an optional capstone.

D092-R1, P0b, corrected G0b, G1, and the D093 architecture/exposure matrix are
terminal at their registered scopes. In D093, each seven-condition cell has
lower selected truth-input error than its three-condition counterpart, but only
FFNO has a lower selected free-rollout sum. The three-condition
arms retain seven-condition normalization, all results use one seed and one
open validation trajectory, and three cells have incomplete training closure.
There is no live PlanarDet process or automatic retry. The line is closed at
its bounded scope. A paper-faithful direct-state FFNO control,
free-rollout-aware selection study, or seed/condition replication can re-enter
only if the active diagnostic programme later selects that family and mediator;
none is a current queue. Do not promote an oracle intervention or architecture
claim. The released test
trajectory remains physically unfetched until a separately named one-shot
evaluation decision. See the bounded
[D093 record](D093_W26_L4_PLANARDET_SCALING_RECORD.md).

D094 B1-A/B1-B, B1-C0/B1-C1, and B1-C4 are locally retained and rehashed. The
B1-C4 cells each complete 40,960 steps, 160 metric rows, and all finite H79
rollouts with zero hard failures. Both select step 38,400 and fail every frozen
strict late-decrease criterion, so neither automatically proposes 81,920 steps.
That routing result does not establish convergence of optimization,
representations, or the deployed recurrent map.
The clean outside audit evaluates four fixed checkpoints per count on the same
28 development trajectories without reselection or historical-test access. At
selection, fixed one-step and internal H79 are near ties, but outside H79 favors
`n=128` by 14.46% and on 22 of 28 cases. At exact `X=64/79`, `n=256` is 41.0%
better outside but has twice the updates; at the same 20,480 updates it is 13.7%
worse but has half the passes. These are interacting data--compute--schedule--
recurrence views, not isolated data or capacity effects.

The clean B1-C4 evaluator packet is bound to commit `3baa6cd`; all 12 artifact
and 28 Git-archive source members rehash. A current-checkout verifier flags
exactly two Windows line-ending differences, while byte comparison to the bound
Git commit passes every source member.

B1-C5-A is complete on the retained selected and terminal checkpoints. Three
fresh bfloat16 processes and one float32 control complete all 448 primary
rollouts with zero hard failures. All four preserve selected `n=128 < n=256`
H79; bfloat16 means are `0.038809/0.044743` and float32 is
`0.039575/0.045282`. The largest within-cell H79 range is 15.58% of the
smallest cross-count gap, so the preregistered aggregate floor gate passes.
Selected-to-terminal H79 worsens for both counts in all four executions.
Bfloat16 H79 rises 1.98%/4.06%, even though H20 improves 4.06%/5.15%, all-call
error improves 0.90%/0.11%, and fixed validation changes only -0.30%/+0.02%.
This early-help/tail-harm crossover is the main new signal.

Per-case and physical-event claims remain local: 23/28 count winners are stable
(19 `n=128`, four `n=256`), five flip; the maximum-gap case is trajectory 83 in
three executions and 152 in one; and eight of 112 event rows are precision-
sensitive. Admissibility counts vary despite complete finite rollouts, so they
do not override rollout error for selection.

B1-C5-B is complete on the retained step-20,480, selected, and terminal maps.
Both fresh FP32 executions close 26,544 rows, all finite/path-completion gates,
the `1e-4` pooled same-map replay ceiling, and `1e-10` quadratic identity. On
the selected path, terminal versus selected improves one-call error at calls
20/79 by `0.166%/0.130%` for `n=128` and `0.294%/0.117%` for `n=256`. Its own
FP32 autonomous path instead improves H20 by `4.37%/8.40%` and worsens H79 by
`1.71%/2.10%`. At H79,
common-input response improves in 24/28 and 23/28 cases, and 14 cases per count
flip from that help to harm on the terminal checkpoint's own path. The frozen
classification is
`resolved_common_input_response_does_not_match_full_crossover`.

The selected-path map result rules out the simple explanation that the terminal
map is pointwise worse at late selected-path states. B1-C5-C now closes the
registered symmetric square with two fresh FP32 processes. At H79, terminal-map
effects on the selected path are `-0.00005151/-0.00005298` for `n=128/256`,
while selected-map effects of terminal-path displacement are
`+0.00068861/+0.00096228`. Even with interaction set to zero, the exact totals
are `+0.00063710/+0.00090930`; the frozen classification is therefore
`resolved_path_displacement_sufficient` for both counts. The positive
interaction contributes only 5.74%/4.48% of the realized total and is not
required to reverse the sign.

The path effect turns permanently harmful only at calls 61/57 and the total at
calls 61/56 for `n=128/256`. Terminal remains better over all calls by
1.55%/3.24% while becoming worse at H79 by 1.71%/2.10%. All 14 stable parent
sign-flip cases per count are path-sufficient, but only six case identities
overlap across counts. This is a one-seed, retained-map reachable-path result,
not an explanation of why optimization changed the path. Do not auto-run
81,920/163,840, PCFNO/FFNO, another seed, or the historical test; the
representation-evolution study remains deferred, not rejected, under a separate
schedule-aware contract.
The ignored local recovery roots are `d094_b1_c4_closeout_20260822a`,
`d094_b1_c4_outside_audit_20260822b`,
`d094_b1_c4_audit_repeatability_20260822a`, and
`d094_b1_c4_visualizations_20260822c` under `artifacts/time_dependent_no/`.
The new ignored roots are `d094_b1_c5_repeatability_20260823a`,
`d094_b1_c5_repeatability_visualizations_20260823a`,
`d094_b1_c5_fixed_map_20260823b`, and
`d094_b1_c5_map_path_20260823a`; their retained manifests rehash. The fixed-map
retrieval binds 115 files and `125,482,211` bytes at SHA-256
`d749ba95bca4fa084ca6a7d760a4b13adc5058e97650fb5f9ee185fab7b9b8c7`.
The original result manifest preserves one finalizer-log self-reference mismatch;
all scientific artifact/source manifests close. The visualization renderer is
`scripts/time_dependent_no/visualize_pcno_bump_b1_c5_repeatability.py`.
The map--path result manifest binds 116 files and `70,825,545` bytes; its local
retrieval manifest binds 123 files and `70,862,988` bytes with SHA-256
`2c936241f3156a981fdf560d6e320dfd4235414a92270697ea3da4d519ce4281`.
An independent local replay reproduces the three scientific CSVs byte-for-byte
and the same classification.
B1-C2 attempt `d094_b1_c2_seed_replication_20260824a` closed before source
staging or training when its conservative old-payload storage gate failed; no
existing output was deleted or compacted. Replacement `20260824b` binds source
commit `2372b8b` and archive SHA-256
`ff029e6321686b1c2733f6d66dca237c45b323353eab4c2a38555d216ddd355b`.
Its 30 focused remote tests, two extreme registered-cell preflights, data/split
hashes, and one BF16 CUDA forward passed. The 24-cell serial queue completed at
2026-08-25 00:14:22 CST with all receipts passing and no historical-test
access. The 1,700-file, 5.607-GB regular payload is locally retained in a
streamed tar whose SHA-256 is
`215a4adea5f850a9c1a79c7fe934fbcb77c22a305dd151cb42ff123c52fa9c92`;
every regular member matches the independent remote retrieval manifest. The
common-process 72-checkpoint audit `20260825c` also completed and rehashed:
all cells reached H79 with zero hard failures, no reselection, and no historical-
test access. Selected seed-specific outside H79 means for PCNO/PCFNO are
`0.14047/0.14896`, `0.07918/0.09867`, `0.06389/0.10353`,
`0.05808/0.10840`, `0.05115/0.09778`, and `0.06515/0.12108` for
`n={8,16,32,64,128,256}`. PCNO is lower in all three seeds at `n>=16` and in
every paired outside case at `n=128`. Both architectures worsen from `n=128`
to `n=256`; this is not monotone scaling.

The fixed-validation/fixed-seen ratio reaches about `1.21` at `n=8` but stays
near `1.01--1.03` at `n>=32`, whereas the recurrent architecture gap keeps
growing. PCNO is initially recurrently worse at step 256 for every count, then
crosses and remains better after about 7,680--16,640 updates. This is the main
mechanistic signal: one-step generalization saturation conceals an optimization-
and-data-dependent difference in self-composition. A post-hoc same-state check
also bounds BF16 interpretation: 14 selected/terminal pairs have bit-identical
model tensors, yet repeated PCNO H79 differs by up to `3.11%`; PCFNO repeats
exactly. Small late differences are within the gradient-path evaluator floor,
while the `46--48%` high-data architecture gap is not.

The ignored result roots are `d094_b1_c2_audit_closeout_20260825a`,
`d094_b1_c2_analysis_20260825a`, and
`d094_b1_c2_visualizations_20260825b` under
`artifacts/time_dependent_no/`. The B1-C2 retrieval/artifact/analysis-manifest
SHA-256 values are
`2090796f2f23e9f89493834fb3ae7e3956e075c9b46604fadb19bbe242a9bea1`,
`0645155a3d09e4eac642568f6491a542a51089252a7ffb0f37408855c96eef90`,
and `9d91a5f4dbcf133e29ceb8967d08e67da38bbe063bf349889a4a391b46caed9b`.

B1-C3 is complete for all 24 retained replication-seed sentinels in BF16 and
FP32. PCNO has lower exact fixed-validation error in every count/seed/precision
cell, with geometric ratios increasing from about `0.56` at `n=8--32` to
`0.80` at `n=256`. H79 does not follow that ordering: PCFNO wins both seeds at
`n=8` and `n=128`, PCNO wins both at `n=16/32/256`, and the seeds disagree at
`n=64`. The signs are identical in BF16 and FP32. At `n=256`, PCNO's exact
step-16,384 sentinel is only 3.2% worse than terminal on one-step validation but
36.6% worse on H79, whereas PCFNO's sentinel is 6.0% worse on validation and
about 2% better on H79. This is an optimization-phase/self-composition signal,
not a smooth architecture-versus-data threshold.

The ignored B1-C3 closeout, analysis, and visualization roots are
`d094_b1_c3_audit_closeout_20260825a`,
`d094_b1_c3_analysis_20260825a`, and
`d094_b1_c3_visualizations_20260825a` under `artifacts/time_dependent_no/`.
Their retrieval, analysis-manifest, and visualization-manifest SHA-256 values
are `8fc529f5dc2fd1519d66ae5cee08ea29c8afd1e0f41b56f0452019f6a0a5c44f`,
`eb66f10751dc7e02406e87617f0f1868fcbe424019dbf46556a9e1b19b885927`, and
`57d1176c73a4a2287b8f71d5ff42da3511fa5f58166ea3aed211946e003413f7`.

B1-C3-R1 training, paired-precision exact-sentinel evaluation, and three-seed
combined analysis are locally retained. The corrected matrix receipt and
closeout artifact-manifest
SHA-256 values are
`cc935ce0495d9c153152ca280bbd650d47b4589cd7ddaf3c1afd6d03e05756b9`
and `e74281466fba618bd0c472a5261eba3a34b8df260bb063467ebeb42144951360`.
The BF16 summary/artifact-manifest hashes are
`b5880b7e9c6c7112e49e21384ee96285e8aa189466389f9b1a9c36b8d98fa714`
and `2a7df14a50527150d2f869529ef995a1a2b24e7715e008de91b896beb7fe769a`;
the FP32 values are
`f29ab183ba0ced7c00c6314bd6d3d953b54564d08b2ed716a2f584f496751e2b`
and `0796f50fc729ebe6b4ab790305109e4ac4c4883669bfbbc9f4f643c4c0321c56`.
The precision-pair receipt passes, all 12 checkpoints match across precisions,
every selection/outside H79 rollout completes, and test access is false.

PCNO has lower fixed-validation one-step error at all six counts. Outside H79
favors PCFNO at `n=8/64` and PCNO at `n=16/32/128/256`, identically in BF16
and FP32. The combined 12-entry artifact manifest rehashes at
`5ff3bb80ba0342ff1e9caa2d548227cb15fb67aaaf24873bdb5d5cbb084f6fd8`.
Across all three seeds, PCNO wins one-step in all 18 seed-by-count comparisons
in both precisions; outside H79 is 3/3
PCFNO at `n=8`, 3/3 PCNO at `n=16/32/256`, and seed-inconsistent at
`n=64/128`. Because these target rollouts are already open, R1 is retrospective
development evidence for any future response diagnostic. PCFNO replay is exact,
whereas PCNO diverges from step 256 despite an exact learning-rate trace; the
historical/replay source inventory also has two changed members. The captured
training source matches only 22/25 current strict-source files, so internal
packet integrity is not current-checkout replay compatibility. No
checkpoint reselection, resume, 81,920-step continuation, FFNO/component study,
attention arm, or test access is automatic. Native bump PCFNO remains a trained
no-gradient ablation, not vanilla or paper-faithful FFNO. See the
[D094 preregistration](D094_BUMP_SCALING_PREREGISTRATION.md).
The next contract is the owner-reviewed
[P0 restart-sufficiency preregistration](P0_RESTART_SUFFICIENCY_PREREGISTRATION.md).
Its exact inputs and compatibility surface have been recovered and made
read-only, but its formal A0 command-binding clause remains open. No numerical
stage may begin until the owner reviews that receipt and separately authorizes
an exact, source-hashed A1 implementation and command.

The original D092 scalar-hash failure, P0 metric-decoder closure failure, and G0
manifest-parser launch failure remain provenance, not scientific evidence. P0b
and G0b each have distinct source, preregistration, result, and final manifests
and must not overwrite their failed attempts.

There is no automatic W26-L2 GPU queue after P2-W0. At most one conditional
study may re-enter only if P1 selects its mediator. P2-F shows that the original full checkpoint's
differential path is acutely useful, while independently training without it
allows spectral-plus-pointwise reorganization; it does not show that gradients
simply cause ripple. A supersonic-bump no-gradient study is therefore a useful
cross-family test only as a matched trained-architecture comparison. Current
source hashes differ from frozen B1, but the retained AutoDL B1 staging tree
still closes all seven frozen source hashes. The earlier line-local
recommendation was an isolated A1 preregistration and CPU-plumbing study against
that immutable tree. Under the active programme it is deferred unless P1
selects the differential path as the mediator; it is not a current
authorization.
Plain L2 still has a measured structure tradeoff from P2, so a target-derived
front or fixed-physical/shock-normalized derivative loss is more directly
motivated than raw Sobolev supervision. Previously rejected always-on output
smoothing and D073-A must not be repeated under new names. Filtering,
loss/exposure, boundary 2x2, recurrent D073-B, and REALM training remain deferred
until separately selected and authorized.

P2-W0 sharpens that choice. It rejects posthoc global `/10` gradient scaling,
finds no general wake at the trained phase, and locates a propagated held-phase
defect, strongest for the full pulse. No-gradient also wins the smooth-sine
control, so the present synthetic evidence does not support “gradient is useful
in smooth regions.” A matched bump full/no-gradient study, selective limiter,
or Sobolev/exposure study remains only a conditional intervention source after
P1; none is the next step or an active registration.

D087 closes W26-L1 at its registered H79 scope. W26-L4 PD0 is now complete
through P0b and the corrected G0b recurrence-localization diagnostic. The
D092-R1 17-file source digest is
`1c48afecdde31d5998695438a39b19ddb328f91b63e272c26ecaed0c30e6f502`;
the exact run completed 5,000 steps and selected step 950 with checkpoint SHA-256
`3ecb4ef90c9800812323f388b699ea639fc283ac8da15b801ae6283442ab9924`.
The closed evaluator gives truth-input/free H49 mean NPE
`0.1244744/1.7987391` and a 14.45x free/truth gap. The released-code sums are
`6.09925/88.13821`; the PCNO free validation sum is 7.01x the REALM paper's FFNO
validation value `12.577`. Its truth gate fails only nondecreasing cumulative
`pMax`.
P0b then exactly replays the learned calls, changes only `pMax`, closes every
gate, and removes all free-rollout `pMax` decreases. The primary non-`pMax`
grouped-NPE ratio is nevertheless `0.9925744` and total-NPE ratio is
`0.9972079`, both practically near-null; pressure-group NPE worsens 2.20% and
nonpositive-temperature calls increase from 21 to 25. This is a bounded causal
negative result: `pMax` monotonicity is not the material driver of cross-channel
drift here. G0b subsequently scores each raw proposal before using one exact
next-frame group only as recurrent feedback. Chemistry and density feedback
materially reduce untouched-group error (`0.48403/0.34435`), temperature
feedback is harmful (`1.13895`), and velocity feedback is small/inconclusive
(`1.06437`). Chemistry/density improve every raw-scored group; temperature
improves itself while worsening every untouched group. This localizes a
chemistry-density feedback/compensation signature but does not identify whether
its cause is architecture, rollout exposure, objective, optimization, or data.
G0b result/final-manifest SHA-256 values are
`a05e44a23529e7883802315c47b7fa41c3476c37a6e57106e187e8516c0260a3` and
`385b5f7fea016f3704ccb4e96d2f0dee8558d5681585c087ee4032f25a592ce6`.
G1 then preserves the raw pulse-call proposal and injects one exact chemistry
or density group only into the next recurrent input at calls 4/12/32. Its
common `T+u` ratios are chemistry `0.96534/0.92635/0.83864` and density
`0.98336/0.90248/0.86301`; partner ratios are chemistry-to-density
`0.95009/0.85390/0.67713` and density-to-chemistry
`0.99150/0.94230/0.93567`. All six arms and the global closure pass. Result,
payload, and final-manifest SHA-256 values are
`2005c6b620a608955102c21f7b20a265fbce29c568d939e4662039d7dc5ab14d`,
`0e8f9b99db53b5518d037623b28aae2cbd82d2f8fd6b9be2cca20c1a795ce423`, and
`bef239780ce3e54e5ea7ebd733cea7c9f1a14c5485936fb511cf1eb3d6d158e2`.
The phase trend is dose-confounded, and lower error does not imply greater
admissibility or boundedness. This is not a correction method, seed result,
sealed ranking, or general PlanarDet claim. Test remains absent. W26-L5 remains bounded
completed mechanism evidence rather than the active execution line. A33/A33-R1
completed the frozen 500x200 common-query-truth comparator without opening a new
population. The correction survives transfer floors and every registered
control, while direct
fixed-hop 500x200 loses to both raw and corrected transfer; matched-information
initialization is nearly neutral. A34 then found no family-independent
two-discrepancy direction. A35 and A36 tested, without new model calls, whether
Euler1D could use elapsed time or three static Riemann descriptors; both failed
prospective readiness, although the state map moved from 7 to 9 case wins.
A37 then tested one frozen causal EMA on retained shock-vortex modal rows.
A38 added the causal local discrepancy direction. It improves ordinary nested
skill to `0.98567`, removes A37's early-band failures, and retains zero harm in
ordinary teacher scoring, but all nine late dual-held cells still fail and
late leave-band-out teacher skill is `-1.72491`; no A37/A38 interior model run
or recurrence is authorized. A39 then tested the single preregistered bounded
path-length clock. It improves ordinary nested skill to `0.99258` but makes
endpoint exclusion worse: 18/36 cells fail, minimum skill is `-5.40467`, and
temporal coefficient norm ratio is `3.07970`. Further observer fitting on
these open A2 rows is stopped. A40 then rejected paired-native batching as a
faithful shortcut. A41/A42 found that the correction-inactive coast response of
the accepted-shadow SP19 offset is almost diagonal and transfers unchanged from
E12 to E14 (`0.99850/0.99845` response skill). A43 froze that map and replaced
the accepted-native call only on calls `8--21`. It qualifies with active
full/rank-8/H30 ratios `0.97877/0.93117/0.96249`, all four active
trajectory/endpoint wins, maximum control `1.01141`, exact inactive/shadow
closure, and 548 candidate calls versus optimized A32's 604. Forward time is
still `1.80245x` raw, so the result is a lower-logical-call protocol, not a
latency claim. A44-R1 then applies A43 unchanged to 14 disjoint E00/E11
correction cases inside the checkpoint training-support split. It qualifies
with active full/rank-8/H30 ratios `0.98216/0.95586/0.97873`, four of four
trajectory/endpoint wins, and maximum control `1.04361`. Large/rank-8 views
improve (`0.98063/0.95586`), while transition/local views are neutral to
slightly harmful (`1.00095/1.00253`). A43/A44-R1 are now frozen. No Euler1D
capped replay or further coefficient/response refit on E00/E06/E11/E12/E14 is
queued. A45's CPU-only, no-new-call audit then stops at its prospective gate:
pooled full/rank-8 skill over zero/phase is `0.57844/0.49571`, but E00 skill
over phase is `-0.30058` and coast skill is `-3.03163`. Across 32 case-band
cells, rank-8/large improve while local utility is negative in all 32. This is
mechanism evidence only because the packets contain no same-state exact/coast
counterfactual. Further scalar-history fitting on these eight active cases is
closed. The minimum decisive continuation is a separately preregistered
same-state exact/coast branch diagnostic followed, only after a held-group
gate, by synchronized recurrence on newly generated, checkpoint-independent
dynamic-FV evidence.
A46-A1 now preregisters that diagnostic and passes 15 synthetic closure checks
with 18 focused CPU tests. Its exact and coast probes share one shadow-native
prediction and byte-identical accepted/shadow inputs; truth is accepted only by
the separate scorer, and only the frozen A43 route advances the synthetic
master. The small closure artifact SHA-256 is
`e68aafa58265facd15683fe6675779fa500eaae03195ccc8fa00d686824c0a9a`
with payload SHA-256
`3f6787bc35f6cea95edf60e9687784ba89346e863be9ca5324401a3fb574eb44`.
This is plumbing, not branch evidence: no compatible independent checkpoint or
new-case manifest is bound, and A1 made no checkpoint call, loaded no dataset
array, opened no population, accessed no remote, and ran no controller.
A46-A2's manifest-only preflight now authenticates the frozen reference
contract, binds 52 opened A28--A45 case IDs, and audits ten independent
checkpoint candidates. None qualifies: the closest 12-input checkpoint uses
four literal-zero node-type channels rather than regenerated physical dynamic-
FV types, while all nine D072 candidates omit those channels and the G1/S1
arms also add boundary inputs. No candidate checkpoint bytes are present
locally for rehashing, and no new branch-label/recurrence case manifest exists.
The fail-closed report is therefore `blocked` on exactly
`no_compatible_independent_checkpoint` and
`new_case_population_manifest_missing`. Its artifact SHA-256 is
`edbe1f3d17dca7388ea51246dd079b59b9f77f8f41be0a19283e2a229a755801`
with payload SHA-256
`9069c3664601beae5843eec2f41e685dfd4594948b2fff97c02f061aca4307c2`;
27 focused CPU tests and Ruff pass. This A2 audit made zero checkpoint-tensor,
model, state/truth-array, outcome, remote, or controller calls. The minimum
continuation is a separately authorized resource-build stage for one fresh
D060-compatible physical-node-type checkpoint and a frozen newly generated
dynamic-FV branch/recurrence manifest; A46 branch inference remains closed.
Three subsequent A46-A2-R1 infrastructure attempts stopped before data,
generation, model construction, training, or outcome opening: two exposed broad
imports missing from the isolated source snapshot, and one broke Python
environment discovery through a run-local interpreter symlink. The frozen
18-case plan still has `generated=false` and `outcome_opened=false` for every
case. These stops do not clear either A46-A2 block or create scientific evidence.
D073-B remains omitted because no recurrent physical-radius wrapper exists.
Strength-OOD and test populations remain sealed pending an explicit named
decision.

Do not convert working labels in the weekly plan into stable D-series IDs until
scope, source, population, metrics, and noncollision have been reviewed.

## Standing Scientific Boundaries

- Report state error, residual error, admissibility/completion, shock or front
  diagnostics, conservation where physically defined, and boundary leakage as
  separate channels.
- A finite invalid rollout is not an admissible rollout; inadmissibility is not
  automatically the cause of a later blow-up.
- A finite set of resolution tests is bounded transfer evidence, not proof of
  operator learning or resolution invariance.
- Node dropping is query-graph resampling unless connectivity, quadrature,
  boundary tags, targets, and provenance establish a physical grid contract.
- Boundary-representation studies must freeze the physical boundary policy and
  evaluator recurrence separately from the learned representation.
- Visualizations require fixed physical meaning and scales and supplement, but
  never replace, scalar and structure diagnostics.
- A failed registered attempt closes that attempt, not an entire method family,
  unless the evidence and current owner direction support the broader claim.

## Artifact And Source-Snapshot Gate

Artifact deletion must start from a regenerated inventory. Delete only outputs
classified as reproducible scratch, superseded duplicate smoke material, or
cache data after checking that no active document, manifest, registered hash,
or recovery contract binds them. Ambiguous scientific payloads are retained.
Record what was deleted and whether it can be regenerated; never infer safety
from age, directory name, or ignore status alone.

Euler2D residual-PCNO runs use source-snapshot schemas v2 through v6, with v6
the latest registered schema in `pcno_artifacts.py`. Every schema retains its
own file inventory and meaning. V2 binds the then-current decision and tracker
inside strict source equality; v3 through v6 separate executable/scientific
sources from provenance-only documents so later documentation edits do not by
themselves invalidate continuation. V5 freezes the boundary-field-era base
inventory. V6 retains that base and adds a sorted, unique, run-specific
`extra_source_files` registry to the strict source set. REALM PlanarDet uses the
separate `realm_planardet_pcno_source_snapshot_v1` schema; D092 and D093 share
that schema name but bind distinct inventories and payload digests.

V6 closes the governance gap created when a v5-bound core source changed after
v5 was registered. Do not reinterpret v2--v5 as v6. An archived run is
compatible with the current checkout only when its schema-specific recorded
hashes still match. The compatibility tests exercise each schema's inventory
and verification branch; they do not assert byte equality between historical
snapshots and current source. Archived snapshot integrity and current-checkout
continuation compatibility remain distinct checks.

## Unrecoverable Preregistration Gaps

A scan of the checkout, all Git history, and 51 retained `.tar.gz`, `.tgz`, and
`.zip` archives found no exact bytes for the nine source-bound preregistration
documents below. Representative retained records still bind their historical
identities. Those records remain evidence at their bounded outcome scopes, but
exact old-identity source replay is unavailable. Do not reconstruct any missing
document under a historical hash; a future run requires a new preregistration,
identity, and source manifest.

| Missing document | Bound SHA-256 history | Representative retained binding |
| --- | --- | --- |
| `W26_L1_PCFNO_H320_PREREGISTRATION.md` | `db5cc052a470d06fd3360b84f20c569421c949914ace6774bc4dfaf13894eb3b` | `artifacts/time_dependent_no/w26_l1_pcfno_h320_s20260718_20260814a_r1/run_contract.json` |
| `W26_L1_PCFNO_PCNO_H320_COMPARISON_PREREGISTRATION.md` | `a25aad5a9ce2c35b6b4968ebe0e6f3225a70f1f692fd6fb8abd4dcc42c0480a7` | `artifacts/time_dependent_no/w26_l1_pcno_h320_s20260718_20260814a_r1/run_contract.json` |
| `W26_L2_PCFNO_INADMISSIBILITY_CONTINUATION_PREREGISTRATION.md` | `d77aebf69a5fc790be906b6934b5af3a8f70454c52cd66f1e078ecbd0746ddfc` | `artifacts/time_dependent_no/w26_l2_pcfno_inadmissibility_continuation_s20260718_20260813a/run_contract.json` |
| `W26_L5_CROSS_RESOLUTION_PREREGISTRATION.md` | `157f02824b05eb346576fc90845edbd7fcb8b4fe2247b48e53d9168bc13fc2b7` | `artifacts/time_dependent_no/w26_l5_rank8_projected_a2_20260811a/remote/calibration/source_manifest.json` |
| `W26_L5_FINE_DISCREPANCY_ROLLOUT_PREREGISTRATION.md` | two historical bindings: `1de56f26f9fcb53ec8571e09ea8f3af6a461bf9a5bcaaf79231074554d35b599`; `5f82858aadceeeea680b4ab61f4872323fe868e89bf53c4b6d5ecba669985d6a` | `artifacts/time_dependent_no/a18_preflight_check.json`; `artifacts/time_dependent_no/w26_l5_fine_discrepancy_a2_20260812a/readiness.json` |
| `W26_L5_PROPAGATED_SENSITIVITY_PREREGISTRATION.md` | `07bc027a33ce320e87c0dca55ec8975df4f6c83556961110e5bdf9ad5514f4f5` | `artifacts/time_dependent_no/w26_l5_p5_psj_a2_20260812b/autodl_runs/preflight.json` |
| `W26_L5_RANK8_PROJECTED_CORRECTION_PREREGISTRATION.md` | `a4e80a1482b0c650ed3cda679ef5f71eaa0bef96e2d2ade07bf7de981d3a4664` | `artifacts/time_dependent_no/w26_l5_rank8_projected_a2_20260811a/remote/calibration/source_manifest.json` |
| `W26_L5_RESPONSE_FILTERED_BLOCK_PREREGISTRATION.md` | 38 historical bindings; current frozen constant `a940a2d7fe0f219e97d742e30bbf1d352687f0b63ccbd4cdcb258b3b1028021b` | `artifacts/time_dependent_no/w26_l5_p6_rfb19_a14_shardtruth_20260813c/w26_l5_p6_rfb19_a14_shardtruth_20260813c_runs/a14_shard_native_rollout/source_manifest.json` |
| `W26_L5_SAME_STATE_REFRESH_COUNTERFACTUAL_PREREGISTRATION.md` | `3bc057aa3b337ce09ae6c80c36cf23c6d863d15c73ed56d62ce13b802ca2419f` | `artifacts/time_dependent_no/w26_l5_p6_rfb19_a46_a2_metadata_preflight_20260814a/source_manifest.json` |

## Additional Exact-Source Replay Gaps

The latest `W26_L2_SHOCK_PATHWAY_PREREGISTRATION.md` bytes were recovered from
the retained `w26_l2_p2_source_v2.tar.gz` archive, rehashed at
`e634653e73275a0b7b8f3691010b3dab77229e08af38f3f438b37b213a9805ce`,
and preserved in commit `c8a2f21`. Four older artifact-bound versions remain
unrecoverable from the checkout, Git history, and 51 retained archives:

- P0: `a9119beff2c91a4605655c43ca14a46f67b65c7d11b9e69f20433862712d5d6e`;
- P1: `126170512c6a0a813d007ea8b621797848e4801b848a2e70f5504f47e81f66cc`;
- P2-C0: `c7836068931dffc2a9455b067330fab32c8a202998ee79ac0f7c9326195e004c`;
  and
- P2 frozen cube: `cc2ffa794337dc6d490282cdc86cb5df041641a80ef34f62150e7649685dce7b`.

The historical B1 source verifier reused by W26-L2 and all three H320
evaluators also binds three exact source versions absent from the checkout, Git
history, and every retained archive:

| Frozen source path | Expected SHA-256 |
| --- | --- |
| `docs/time_dependent_no/MECHANISTIC_DIAGNOSTIC_TRACKER.md` | `c3014df67a4e7d5ed8f1d48b1c76f664b5a7604557d7ce0b783ca77a19cc7ee4` |
| `docs/time_dependent_no/RESEARCH_DIRECTION_DECISION.md` | `dafa88e4bca3b174ff9711272259e6a449dab9d744dac24d0254e6a61d7c2de3` |
| `scripts/time_dependent_no/train_pcno_euler2d_residual.py` | `7f7733b421f74411a420d753b51a5c56a798f578462806a37054bee16bc82318` |

Their strongest original binding is
`artifacts/time_dependent_no/l3r_b1_serious_20260728a/b1_20260727a_serious/source_snapshot/manifest.json`;
the H320 and W26-L2 run contracts repeat it. Retained output packets remain
bounded outcome evidence when their own manifests rehash, but exact historical
source replay against the mutable repository necessarily fails closed. Do not
replace the missing bytes or relabel a future run under an old identity.

## Recovery

- [Archive guide](history/README.md)
- [Full decision state through 2026-08-11](history/RESEARCH_DIRECTION_DECISION_through_2026-08-11.md)
- [Full evidence ledger through 2026-08-11](history/MECHANISTIC_DIAGNOSTIC_TRACKER_through_2026-08-11.md)
- [Frozen D072 experiment plan](history/d072_refine_logs_frozen_2026-08-03/EXPERIMENT_PLAN.md)
- [Frozen D072 execution tracker](history/d072_refine_logs_frozen_2026-08-03/EXPERIMENT_TRACKER.md)
- [Corrected 1D baseline record](SECTION_1_2_CORRECTED_BASELINES.md)
- [Bump data audit](BUMP_300_DATASET_AUDIT.md)
- [CPG dataset contract](CPG_EULER_DATASET_CONTRACT.md)
- [Public-reference audit](CPGGNSPDES_REFERENCE_AUDIT.md)
- [H320 reference-free rollout record](W26_L1_H320_REFERENCE_FREE_ROLLOUT_RECORD.md)
- [D094 bump-scaling preregistration](D094_BUMP_SCALING_PREREGISTRATION.md)
- [Hash-bound W26-L5 derivation source](../../DERIVATION_PACKAGE.md)

Commit `5646bfb` preserves the full active decision and tracker immediately
before this compaction. Commits `ebf210a` and `3e646ac` preserve the weekly-plan
milestone and first isolated-scaffolding prune. Generated artifacts remain
outside Git and require their own manifests for recovery.
