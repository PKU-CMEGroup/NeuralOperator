# Time-Dependent Experiment Index

Updated: 2026-08-03

Status: compact routing index; not an execution queue

## Purpose

This file routes stable experiment IDs to the byte-preserved historical ledger:

[history/MECHANISTIC_DIAGNOSTIC_TRACKER_through_2026-08-01.md](history/MECHANISTIC_DIAGNOSTIC_TRACKER_through_2026-08-01.md)

Historical ledger SHA-256:
D055F2EEB398224CEFDBDD86D3A230C501DFA4069D65DF188C7F8183F0F821CE

The archive preserves dated contracts, metrics, hashes, stop decisions, and
forward-looking language exactly as recorded. That language describes the
campaign state at the time. It does not authorize or prohibit current work.
Current explicit human direction and
[RESEARCH_DIRECTION_DECISION.md](RESEARCH_DIRECTION_DECISION.md) govern the
current scientific framing.

Normal startup should read this compact index, not the full archive. Retrieve
only the section needed for the active question. For example:

    rg -n "^## .*D053|^\| D053" docs/time_dependent_no/history/MECHANISTIC_DIAGNOSTIC_TRACKER_through_2026-08-01.md

Then read a bounded line range around the matching heading. Do not infer missing
checkpoint, source, command, data, environment, or artifact identity from
memory; use the archive's exact retrieval request.

## Stable ID Rules

- The primary run universe contains 103 IDs: D001--D074, 23 L3R IDs,
  L4A-001--L4A-004, L1-OOD, and XLINE-001.
- D043b/D043d are attempts under D043; D053a/D053b are attempts under D053.
- P/D019 maps to D019 and R/D041 maps to D041.
- RC-01--RC-15 and historical L3R-C1/L3R-C2 are claim IDs, not run IDs.
- A failed attempt retains its identity. A revised experiment receives a new
  attempt or run ID rather than rewriting the old result.

## Line 3 Restart IDs

| ID | Historical question | Recorded disposition |
| --- | --- | --- |
| L3R-M0 | PCNO provenance and reporting semantics | Completed; residual collaborator/source gaps declared. |
| L3R-U0 | Exact unchecked D041 replay | Stopped at replay gate; no continuation artifact. |
| L3R-U1 | Exact-state-spliced unchecked continuation | Completed through all 80 states; no registered numerical blow-up. |
| L3R-R0 | Clean D041 reproduction | Superseded as a prerequisite; not executed. |
| L3R-A0 | Historical checkpoint attribution | Superseded as a prerequisite; attribution remains unresolved. |
| L3R-B0P | First serious-baseline preflight | Tiny-fit gate failed. |
| L3R-B0 | First serious baseline | Not launched after B0P. |
| L3R-B1P | Replacement-baseline readiness | Passed. |
| L3R-B1 | Serious causal-boundary bump baseline | Completed 34/40 passes; no parity-eligible checkpoint. |
| L3R-BC0P | Exact-D041 continuation preflight | Memory ceiling failed after identity gates. |
| L3R-BC0 | Paired raw/causal continuation | Not launched after BC0P. |
| L3R-BG0P | Boundary-gradient preflight | Passed. |
| L3R-BG0 | Matched boundary-objective study | Completed; both auxiliaries missed utility gate. |
| L3R-RB0P | Raw-boundary supervision preflight | Passed. |
| L3R-RB0 | Raw all-node continuation | Completed; boundary/H20 improved but H79/structure regressed. |
| L3R-RA0P | Gradient-normalized auxiliary preflight | Repeatability tolerance missed. |
| L3R-RA0 | Weak raw-boundary continuation | Not launched after RA0P. |
| L3R-K2D0 | Exact B1/attached-K2 error decomposition | Completed; propagated-path gain dominates. |
| L3R-K2T0 | Matched projected-teacher control | Two zero-step harness/preflight failures; no method result. |
| L3R-MR0 | Shared stride-conditioned dynamic PCNO | Historical proposal; no result. |
| L3R-MR1 | Seed confirmation of MR0 | Historical conditional proposal; no result. |
| L3R-L0 | Sensor-gated local residual branch | Historical mechanism-conditional proposal; no result. |
| L3R-F0 | One-time sealed evaluation | Historical proposal; sealed population remains unopened. |

## D-Series Index

Every D-series ID is contiguous from D001 through D074. Exact D001--D063
populations, metrics, thresholds, artifacts, and claims remain in the
historical ledger. D064--D074 are routed in the post-archive section below.

| ID | Family and topic | Terminal evidence state |
| --- | --- | --- |
| D001 | CPG bump one-step rollout summaries | Completed. |
| D002 | CPG bump two-stage rollout summaries | Completed. |
| D003 | Teacher-forced versus autoregressive error | Completed. |
| D004 | State-distribution drift | Completed. |
| D005 | Perturbation amplification | Completed. |
| D006 | Shock position and region decomposition | Completed. |
| D007 | Rollout animations and overlays | Completed. |
| D008 | Equal-node bump physics proxies | Completed; proxy-only. |
| D009 | Approximate-weight bump diagnostics | Not run; legacy plan closed. |
| D010 | Direct state-predictor control | Not run; legacy plan closed. |
| D011 | Corrected PCNO diagnostic replay | Completed through D019 path. |
| D012 | Correlation-time/geometric aggregation | Not run; legacy plan closed. |
| D013 | Scale, spectral, and roughness diagnostics | Completed across bounded regimes. |
| D014 | Effective-CFL and receptive-field audit | Completed with bounded/partial claim. |
| D015 | Generated-state exposure pilot | Completed bounded pilot; promotion gate missed. |
| D016 | Interface-state latent instrumentation | Completed. |
| D017 | Selected interface-latent run | Completed. |
| D018 | Interface-latent mechanism probe | Completed. |
| D019 | Legacy positive-primitive PCNO replay | Completed; exact provenance remains incomplete. |
| D020 | Corrected 1D CPGNet reach intervention | Completed. |
| D021 | 1D target-family optimization screen | Completed. |
| D022 | Later-time versus generated-state exposure | Completed. |
| D023 | Same-state solver-consistency diagnostic | Completed; mixed result. |
| D024 | Conservative-dissipation probe | Completed; no useful stabilization. |
| D025 | Global interface-latent FNO pilot | Completed pilot; recurrent viability failed. |
| D026 | Boundary-exchange supervision | Completed; joint utility failed. |
| D027 | Cold stride-2 gate | Completed; partial macro-step result. |
| D028 | Stride continuation gate | Completed; partial result. |
| D029 | Native-map cross-resolution gate | Completed; native transfer failed. |
| D030 | Restriction-consistent shared-resolution gate | Completed; bounded shared-map result. |
| D031 | Full-split fixed-stride frontier | Completed. |
| D032 | Dense reliability/checkpoint audit | Completed. |
| D033 | Multi-stride same-state decomposition | Completed. |
| D034 | Runtime and Pareto closeout | Completed; contract-specific timing. |
| D035 | Error-geometry and visualization closeout | Completed. |
| D036 | Ripple and roughness conflict audit | Completed. |
| D037 | Dynamic finite-volume reference audit | Completed. |
| D038 | Modal-error evolution | Completed. |
| D039 | Frozen operating-envelope closeout | Completed. |
| D040 | CPGNet legal-boundary closeout | Completed; bounded release-bundle evidence. |
| D041 | Official bump PCNO holdout and boundary routing | Completed; exact historical comparator. |
| D042 | Compact local-basis counterfactual | Completed; exact fixed span rejected. |
| D043 | Paired branch-cancellation audit | Completed; strong cancellation falsified. |
| D044 | Dynamic FV residual-PCNO baseline | Completed; useful one-seed baseline. |
| D045 | Shared-face impulse tiny fit | Completed; tiny-fit gate failed. |
| D046 | Canonical face-target preflight | Completed; numerical closure gate failed. |
| D047 | Direct canonical projector | Completed; preflight gates passed. |
| D048 | Canonical-face supervised tiny fit | Completed; decoded update failed. |
| D049 | Divergence-conditioning audit | Completed; conditioning gates passed. |
| D050 | Residual-to-face lift and boundary headroom | Completed; artifact binding failed and headroom was small. |
| D051 | PCNO/coarse-CFD error-cost comparison | Completed; bounded descriptive comparison only. |
| D052 | Branch-gain sensitivity | Completed; no safe uniform branch attenuation. |
| D053 | Exact fresh/propagated decomposition | Corrected attempt completed; full state mostly propagated, smooth high-pass mostly fresh. |
| D054 | Fresh-defect locality | Completed; localized but not safely shock-localized. |
| D055 | Frozen-proposal self-sensor | Completed; legal locator candidate only. |
| D056 | Causal-support balanced correction | Completed; late-horizon utility failed. |
| D057 | Joint-gradient compatibility | Completed; naive scalarization not geometry robust. |
| D058 | Geometry-group MGDA direction | Completed zero-step audit; late-OOD direction failed. |
| D059 | Direct stride-2 PCNO tiny fit | Completed; fitability routed D060. |
| D060 | Matched serious stride-2 PCNO | Completed; state improved and joint promotion failed. |
| D061 | Scalar multirate blend | Completed; exact blend lacked joint headroom. |
| D062 | Independent-row front chart oracle | Completed; exact chart rejected. |
| D063 | Common-source resolution and node-type audit | Completed; bounded transfer passed and broad claim remains partial. |
| D064 | Residual-scale accumulation and error structure | Completed; 24-case temporal metrics and six registered field bundles expose signed accumulation beyond the small-denominator effect. |
| D065 | Instrumented resolution pathways | Completed; algebra and replay passed, but no pointwise, differential, or spectral family alone dominates. |
| D066 | Physical-scale and temporal-coherence decomposition | Completed; persistent large-scale drift separates from broadband local cancellation on six open cases. |
| D067 | Unified cumulative/pathway/shock confirmation | Completed; same-run replay and all closures pass, establishing the bounded two-component defect mechanism. |
| D068 | Frozen node-type interventions and nonperturbing instrumentation | Completed; both checkpoints use family-local semantics, with bump wall labels dominating the tested native geometries. |
| D069 | Controlled constant-zero-channel training comparison | Completed; three matched seeds do not support an H30 benefit from four permanently zero channels. |
| D070 | Fine-grained PCNO mesh-pathway surgery | Completed under D070C; the differential branch is supported, with multiple nested full/fixed-hop arms rather than a unique additive attribution. |
| D071 | Two-channel frozen residual correction | Completed on dynamic FV and bump; both contracts pass and neither benchmark promotes the combined correction. |
| D072 | Fixed-physical-width semantic boundary fields | Active; open-only manifests and both N0/G1/S1 CUDA smokes pass. All nine dynamic training summaries exist but remain unreviewed. Bump N0 completed and the unchanged eight-arm resume queue is healthy. No core comparison outcome is accepted yet. |
| D073 | Physical-radius differential geometry | D073-A completed; all four dynamic same-hidden mechanism and native-relevance strata pass, authorizing D073-B modified-rollout testing. Bump safety has not run. |
| D074 | Native-resolution residual correction and transfer-native comparator | Owner-prioritized and registered; persistent-only dynamic correction at `250x100` is primary, local dissipation is stopped pending redesign, and direct cross-resolution rollout must beat the frozen transfer-native pipeline. The fail-closed evaluator has passed 120 CPU dependency tests and three independent reviews for a non-scientific dynamic H2 smoke. Source manifest `3fbd8059...5237` binds 92 Python files at base `c6d959f`; no D074 model compute has launched. Dynamic H30 remains closed pending exact regenerated-geometry and selector-repeatability contracts; bump remains closed pending an immutable D041 replay binding. |

## Post-Archive D064--D074 Evidence

These records postdate the byte-preserved historical ledger. Generated arrays,
figures, and logs remain ignored; the registered summaries below bind their
checkpoint, normalizer, population, source, output hashes, and claim scope.

| ID | Registered summary | Terminal interpretation |
| --- | --- | --- |
| D064 | `artifacts/time_dependent_no/pcno_residual_structure_d064_20260802a/results_allval24_d060_20260802b/summary.json` | One frozen D060 checkpoint completes residual-scale, cumulative, recurrence, spectral, spatial, characteristic, and signed-growth diagnostics on all 24 open validation cases. Six representative bundles support fixed-scale fields. |
| D065 | `artifacts/time_dependent_no/pcno_resolution_pathways_d065_20260802a/results_sixval30_d060_deterministic_20260802b/summary.json` | Exact paired-input, commutator, replay, wrapper, and float64 pointwise checks pass. Pointwise and differential families exchange prominence by pair; the branch-level source remains composite or unresolved. |
| D066 | `artifacts/time_dependent_no/pcno_scale_separated_drift_d066_20260802a/results_d064_sixcase_free_teacher_20260802b/summary.json` | Residual-scaled DCT-II wavelength bands, temporal coherence, lag correlation, signed growth, POD/SVD, and conservative-component partitions pass closure. Large scales accumulate; local scales are much less coherent. |
| D067 | `artifacts/time_dependent_no/pcno_cumulative_pathway_structure_d067_20260802a/results_d067_unified_r1d_20260803a/summary.json` and `results_d067_unified_r1d_scale_profile_20260803a/summary.json` under the same root | One self-consistent six-case payload gives zero stored replay error and closes `delta_free=delta_mesh+delta_state`. State response dominates path energy but is mostly corrective; mesh inconsistency dominates the endpoint. Shock-local high-rank noise coexists with persistent low-rank large-scale drift. |
| D068 | `artifacts/time_dependent_no/pcno_node_type_visualizations_20260802a/inputs/dynamic/one_step/summary.json`, `artifacts/time_dependent_no/pcno_node_type_visualizations_20260802a/inputs/dynamic/rollout/summary.json`, and `artifacts/time_dependent_no/node_type_bump_trace_lmh_calls20_20260802b/summary.json` | Exact lift differences and hook equivalence pass. Dynamic all-normal error remains harmful across three grids; bump wall-to-normal tracks all-normal and causes three early admissibility failures. Fourteen animations retain all 317 comparable frames. |
| D069 | `artifacts/time_dependent_no/pcno_zero_channel_training_20260803a/analysis/analysis_summary.json` | Three seeds compare no-type, ordinary constant-zero, and exact-function-matched constant-zero inputs under matched order, optimizer, budget, and population. Mean H30 is `0.008108/0.008272/0.009332`; the proposed zero-channel advantage is unsupported. |
| D070 | `artifacts/time_dependent_no/pcno_fine_grained_pathways_d070_20260803a/d070c_full_r1_scalar/summary.json` | Every D070C hard gate passes. Same-hidden decoded interventions support the differential branch and a substantial fixed-hop composite contribution, while Fourier quadrature/subcell/synthesis and pointwise controls do not qualify. The registered decision is `multiple_supported`. |
| D071 | `artifacts/time_dependent_no/pcno_defect_corrections_d071_20260803a/d071_dynamic_full_r1/summary.json` and `d071_bump_full_r1/summary.json` under the same root | Both family-local contracts pass, but neither combined correction promotes. Dynamic benefit reverses sign at `125x50` and the local arm amplifies vortex error; bump gains split by case. |
| D073 | `artifacts/time_dependent_no/pcno_physical_radius_geometry_d073_20260803d/d073a_full_r1/summary.json` and sibling `d073a_full_r1_visuals/manifest.json` | Fixing the layer-3 graph-ball radius at the training-grid physical width reduces every registered dynamic-FV decoded commutator stratum and passes native relevance. This is adaptive same-hidden pathway evidence, not yet a modified rollout or bump result. |

D067's structural-repeatability comparison is registered at
`results_d067_structural_repeatability_r1_vs_r1d_20260803a/summary.json` under
the same root. It passes the predeclared aggregate gates, but preserves the
first pilot as `failed_closure`; it is not a second accepted scientific
replicate. The final fixed-scale visual manifest is
`visuals_d067_unified_r1d_representative_final_20260803a/visual_summary.json`.

D068 summary SHA-256 values are
`3717da5dafaebc3bf13d8c045de89d01237492bd0eef9817383a2d83da5073e2`
for dynamic one-step,
`fab3e694756e4cbbec520dab1b3582c8d6bf7932a9e554ffe43917219aebb47e`
for dynamic rollout, and
`c7e7afbd776f9141a47b30c52475738f2a08d7f91fa186d1e3f38c46fe9f58c4`
for the four-geometry bump trace. All-frame media and JSON sidecars are under
`pcno_node_type_visualizations_20260802b_all_frames` and
`pcno_node_type_visualizations_20260802c_cross_geometry`; every declared
comparable call equals its rendered and encoded frame count.

D069's analysis-summary SHA-256 is
`d5be4c888321408fe478d017c886395a43768a3633d311870b9444bb07121fce`.
Its source, data-manifest, and normalizer digests are respectively
`09b3735107416d7041f5db589cfab2c95ebb4cbe8bb659883e08762c07311481`,
`f8d228ae3c6e08fe6697f37df21476fe621abc354d0b0a6eed2a623ed2b9f96c`,
and `9df7efb2f1aadf315f399dcabf3b366bf1b2f46794b026cbc5b7d80ba22e1a1a`.

## Registered D072 Boundary-Field Stage

D072 is a separately authorized representation study, not a
boundary-condition improvement and not a continuation of categorical
node-type interventions. It asks whether a bounded semantic collar of fixed
physical width improves autonomous time-dependent PCNO rollouts while avoiding
the vanishing-mass and shrinking-support behavior of a one-cell tag band.
All nine dynamic data-scale training summaries now exist, but aggregate
verification and interpretation have not begun. Bump seed-20260718 N0 completed;
an unchanged eight-arm resume queue is running unattended after the original
wrapper stopped on an incorrect `raw_recurrence=true` assertion. Dynamic FV is
assigned to the workstation and bump to AutoDL, with each dirty remote checkout
excluded in favor of an isolated, digest-bound source snapshot. D070--D071 are
complete and do not pause this stage.

The final touched boundary/source/trainer regression passes 59 CPU tests and
the maintained runtime surface passes six more. Dynamic publication derives
only the 84 train and 24 open-validation trajectories from the retained D069
shard manifest; the 27 test trajectories are filtered before any folder path
is constructed. Bump width selection decodes only HDF5 geometry for the 270
training keys. Its source geometry reproduces the legacy manifest digest;
the retained shards intentionally differ at the digest level because positions
were serialized from float64 to float32 and canonical edges from int32 to
int64, while their values match exactly after those declared casts. D072 binds
both representations instead of treating dtype serialization as geometry
drift.

The artifact contract stores one compact cubic volume descriptor per semantic
boundary subset. The width, kernel, channel order, continuum-object label,
geometry provenance, corner rule, and family-local factorization are bound in
the shard manifest. Dynamic FV stores overlapping y-symmetry and
x-extrapolation fields from validated rectangular bounds. Bump stores wall,
outflow, and inflow fields from the released tagged boundary-cycle polyline;
mixed endpoint edges belong to both adjoining subsets. That bump polyline is a
mesh-derived proxy, so its fields do not convert node dropping into physical
resolution transfer.

The preregistered first ladder is:

| Arm | Categorical types at model input | Continuous input | Initialization |
| --- | --- | --- | --- |
| N0 | omitted | none | ordinary eight-input source |
| G1 | omitted | maximum over all semantic collars | exact N0 function match |
| S1 | omitted | separate family-local semantic collars | exact N0 function match |

All arms retain physical node types outside the model for frozen masks,
metrics, and boundary policy. G1 separates domain-boundary geometry from
semantic identity; S1-G1 is the incremental semantic-information contrast.
No learned extension network, normals, tangents, surface-delta scaling, or
alternative geometry architecture is in this ladder. The primary physical
width is selected from geometry-only training-manifest statistics and frozen
before any outcome is inspected. At most one narrower and one wider
outcome-independent sensitivity are allowed after the primary matrix; they are
robustness checks, not architecture search.

Each family is trained and reported separately with matched seeds, presentation
order, optimizer, budget, recurrence, physical boundary policy, and open
evaluation population. Dynamic FV owns common-source conservative resolution
tests because its connectivity, cell volumes, tags, and targets are validated.
Bump owns native-graph geometry variation only. Rigid transforms first test
descriptor covariance numerically; model rotation performance is a later
stress test and cannot be described as broad geometric generalization.

Mechanistic analysis starts with the exact pre-activation lift
Delta h_lift(x) = W_B B(x). Frozen zero-field, union-field, and
single-semantic interventions then propagate through the maintained
nonperturbing branch/block instrumentation. Post-lift and per-block
differences, pointwise/integral/differential branch contributions, region and
physical-frequency effects, teacher-forced calls, and every free-rollout frame
are retained. Hooked inference must match hook-free inference at a declared
tolerance before activation evidence is used. JVP norms are sensitivity
diagnostics, not causal proof.

Causal language is restricted to frozen interventions on a fixed trained
checkpoint and fixed evaluator. Matched-training differences support an
effect of granting a model access to the representation under the tested
training recipe; they do not prove optimality. Resolution consistency means
the channel definition has fixed pointwise amplitude, physical width, and
continuum target as h changes. Finite grids and a bump boundary-polyline proxy
do not prove operator convergence.

The maintained implementation is owned by
`utility/time_dependent_no/pcno_boundary_fields.py`, the optional shard
fields in both PCNO preparation entry points, the generic input and exact
lifting diagnostic in `pcno_euler2d.py`, manifest-aware runtime loading, and
the existing trainer. Focused synthetic tests cover collar mass under 2x
refinement, factorized dynamic corners, bump semantic overlap, rigid-motion
invariance, exact parameter copying, the numerical lift equivalence tolerance,
and bump shard-to-runtime inference. The 2026-08-03 relevant CPU regression
suite passes 68 tests. No dataset-scale shard generation, checkpoint training,
remote compute, or scientific outcome is recorded by this implementation
milestone.

Stop before training if source, boundary-geometry, normalizer, split, or
checkpoint provenance is incomplete. Stop interpretation if descriptor
covariance fails, a field contract changes across arms, hooks perturb outputs,
or bump evidence cannot be separated from query-mesh resampling.

## Registered D070--D071 Corrective Stage

D070 isolates the large-scale mesh defect only on the dynamic finite-volume
family, where nested physical grids, common-source references, physical cell
volumes, connectivity, tags, and conservative restriction are all bound. It
reuses the D060 checkpoint and normalizer, D067's six open-validation cases,
both adjacent grid pairs, calls 1/5/15/30 in teacher-forced and free-rollout
states, FP32 without AMP, raw recurrence, and `model_all_nodes`. A native
output replay against D067 is a stop gate before interpretation. D070 also
requires the active isolated snapshot to match all eight inherited source
hashes recorded by the accepted D067 summary. The exact-source contract is
named exact_registered_d067_inherited_sources; summary and bundle hashes alone
are insufficient.

For same-hidden paired fields `h_c = R h_f`, let `P` be blockwise-constant
injection from the coarse grid to the fine grid. The spectral mesh gap is
closed as

`K_c(h_c) - R K_f(h_f) = [K_c(h_c) - S_c M A_f(P h_c)] +
[S_c M A_f(P h_c) - S_c M A_f(h_f)] +
[S_c M A_f(h_f) - R S_f M A_f(h_f)]`.

The first term is a controlled Fourier quadrature/analysis-grid response for
the same injected coarse hidden values, not an exact continuum quadrature
error. The second is fine subcell hidden content lost by restriction; the third
is Fourier synthesis/restriction aliasing. Their aggregate remains the original
analysis-versus-synthesis split. For the differential pathway, with `G` the
least-squares gradient and `Q_h` the checkpoint's two-hop neighbor average,
the pre-nonlinear closure is

`Q_c G_c(h_c) - R Q_f G_f(h_f) =
Q_c[G_c(h_c) - R G_f(h_f)] + [Q_c R G_f(h_f) - R Q_f G_f(h_f)]`.

These terms distinguish the gradient stencil from fixed-hop geometry before
`gw1`, Softsign, and `gw2`; the corresponding decoded responses retain those
nonlinearities and are labeled composite. Float64 pointwise commutation is a
negative control. Single-layer and pathway-wide coarse-branch replacements
with restricted native-fine branch values are causal representation
interventions, not deployable models. Their primary endpoint is reduction of
the D066 physical large band (`wavelength >= 0.125`) in the decoded increment
commutator. The primary selector uses only same-hidden, single-layer rows;
native all-layer replacements are supportive diagnostics. Its complete
six-case by four-call matrix must pass separately in teacher-forced and
free-rollout modes on both adjacent pairs. A pathway qualifies only if its
median large-band reduction is at least 20%, the sign agrees in at least four
of six cases, and neither full-field nor local-band defect rises by more than
10% in every pair and mode. Exactly one qualifying pathway is selected;
multiple nested qualifying arms are reported as `multiple_supported`, not
forced into a unique attribution. Otherwise the source remains composite or
unresolved.

D071 tests two corrections without changing checkpoint weights or physical
boundary handling. Calibration and evaluation cases are disjoint subsets of
the already-open validation populations: the six D067 dynamic cases and four
D068 bump cases are evaluation cases; the remaining open-validation cases are
calibration only. No Strength-OOD or test trajectory is opened. Family results
are reported separately and nodewise rows are aggregated within case before
any across-case summary.
Dynamic FV is frozen to `125x50`, `250x100`, and `500x200`, with checkpoint
training resolution `250x100`; bump remains native-graph only. D070 and D071
also require a digest-bound isolated-source manifest whose inventory exactly
equals every Python file under `pcno/`, `utility/`, and
`scripts/time_dependent_no/`, together with the registered 40-digit base HEAD.

The persistent channel is an offline, target-free-at-application bias map. In
each family, conservative residual errors are divided by the frozen residual
component scale and projected by proxy/volume-weighted least squares onto the
first eight separable physical cosine modes, ordered by physical wavenumber
and including the constant mode. A call-indexed mean coefficient vector is
estimated from teacher-forced calibration cases and frozen before evaluation.
Both fitting and direct correction support use only family-local type-0 nodes;
the reconstruction is exactly zero on every boundary-contact node. The
evaluated correction subtracts only this reconstruction; an evaluation-truth
projection onto the same support and subspace is reported separately as
one-step headroom and is never used in free rollout.

The local channel is a legal proposal-only controlled-dissipation map. Its
sensor is the continuous upper 20% of the graph-high-pass amplitude of the
scaled predicted increment on the family-local type-0 induced graph. Only edges
joining type-0 nodes are eligible, and the correction-norm cap uses the type-0
predicted update on that same support. Symmetric edge conductances make the
added correction exactly
physical-volume-mean neutral for dynamic FV and only proxy-weight-mean neutral
for bump. Candidate correction-norm caps of 2% and 5% of the predicted update
are selected on calibration data, then frozen. This is neither an entropy-
stable solver nor physical conservation of the base PCNO. It differs from
D024's blind uniform diffusion and D056's discontinuous sparse-support
correction by being proposal-localized, symmetric, continuously weighted, and
norm capped.

The registered arms are baseline, rank-eight persistent correction, local
controlled dissipation, and their combination; subspace headroom is
teacher-forced descriptive evidence only. Report absolute and residual-scaled
update errors (physical absolute RMS is component-wise),
large/transition/local bands, correction norms, free-rollout
state error and admissibility, component and region errors, shock
behavior, vortex-core behavior where defined, smooth high-pass behavior, and
boundary leakage. The reference-shock-mask graph-high-pass centroid, strength,
and thickness quantities are descriptive proxies only: they cannot reliably
detect a front that has moved outside the reference mask and are not promotion
gates. The selected combined arm must improve final state error by at least 5%
for every registered case-resolution without loss of completion, while every
registered shock-region state, vortex, smooth-high-pass, and boundary control
ratio is present and no larger than `1.05`; missing, zero-baseline, or nonfinite
ratios fail promotion. Bump balance and graph bands remain explicitly
proxy-only, and bump is not a resolution experiment.
Component trajectories are recorded at every call. Scale bands are recorded at
calls 1/5/15/30 for dynamic FV and 1/5/10/20 for bump: dynamic uses the frozen
physical DCT wavelength thresholds, while bump uses an exactly reconstructing
but nonorthogonal two-level graph proxy (`large=S^2x`,
`transition=Sx-S^2x`, `local=x-Sx`).

Figures show per-case-aggregated temporal metrics. Animations use one
rollout-wide physical scale per component and synchronize true increment,
baseline and corrected increments, their residual errors, persistent
correction, local correction, sensor, accumulated correction, and cumulative
baseline and corrected residual errors. Truncated or inadmissible sequences are
visibly labeled. No per-frame normalization is permitted. Before any GPU
ladder, the D071 smoke gate requires exact arm-by-call coverage, admissibility
of all four arms, an exact finite teacher-forced case-resolution-call matrix,
hook equivalence, family replay/reference binding, and both closure gates; a
failed arm or missing/nonfinite teacher-forced frame therefore returns nonzero
rather than `smoke_complete`. The same teacher-forced completeness gate is part
of the full scientific-interpretation contract. Hook equivalence inherits the
accepted D068 family-specific native-repeat contracts rather than generic
defaults: bump uses absolute native-repeat limit 2e-3 and shared relative-L2
limit 1e-5, while dynamic FV uses 2e-5 and 1e-7. The separately computed hook
absolute limit may be larger because it includes the observed native repeat and
FP32 numerical floor; these limits are not replay-error thresholds.

Data assimilation is reserved for a later research line. Possible future uses
include observing only persistent subspace coefficients, shock-phase
registration before coefficient updates, and treating local residual
innovations with robust covariance or localization. D070--D071 do not
implement nudging, Kalman updates, target injection during rollout, or learned
observation operators.

Artifact-integrity reconciliation on 2026-08-03 verifies all 125 output hashes
registered by the accepted D064--D067 summaries and final visual manifest.
D067's registered diagnostic runner and metric-utility hashes also match the
current files. The current dirty worktree differs from five of its eight
inherited evaluator-source hashes, so it must not be called a byte-identical
replay surface.

The five differing inherited files were recovered on 2026-08-03 from the
preserved NeuralOperator-pcno-resolution-20260730b AutoDL source tree and
verified against the registered hashes. Exact replay requires placing them in
one ignored, immutable D067 source snapshot without overwriting the current
worktree:

- `pcno/pcno.py` at
  `c5bb98fe736b370f277de935c88f6fb22efc23417a0dbab1bb9a08e979a6298b`;
- `scripts/time_dependent_no/evaluate_pcno_resolution_rollout.py` at
  `b5c3dcf6a55f6adec320698a7c5f819bc22d9ad0c4118fdcba051b37a49f4b34`;
- `utility/time_dependent_no/pcno_euler2d.py` at
  `ba17a75daaaa38d8172f9c619f5a7bae48cd385e434175a56b3fa002d7b319d4`;
- `utility/time_dependent_no/pcno_resolution_transfer.py` at
  `552dc3279bdc988df5520618879ef9ec79df08b0b311d3a662f03feeeec68f10`;
  and
- `utility/time_dependent_no/shock_vortex_family.py` at
  `6c2d4ba7bb18c611feed55faf6c580fd371f44e8bd63575d91f42c131442e164`.

The other three inherited files already match D067. D070 now verifies the
complete eight-file map before model construction. A bounded exact-source trial
is registered for one case, one call, both adjacent pairs, and both
teacher-forced and free-rollout paths. It may advance only if imports,
checkpoint construction, all source hashes, the original 2e-6 historical
output replay gate, hook equivalence, every pathway/DCT/replacement closure,
admissibility, and smoke row coverage pass. Failure stops exact-source D070;
any current-core compatibility fallback must be separately registered and
must not claim exact replay.

Two current-core engineering smokes are preserved as non-scientific failures.
The digest-bound source-D run gave D067 output max-absolute discrepancy
3.0398369e-6 > 2e-6, while pathway closure was 3.0572186e-6 and hook
equivalence passed. Replacing only pcno/pcno.py by its exact D067 version left
the replay discrepancy unchanged and gave pathway closure 3.0953997e-6; this
rules out the PCNO core alone and does not authorize pathway interpretation.

The first correctly populated bump D071 smoke also stopped before correction
interpretation: D071 accidentally used generic hook limits and observed native
repeat max-absolute 3.2043457e-4 and relative L2 4.585e-7. Both lie within the
accepted D068 bump contract (2e-3, 1e-5). This is an evaluator wiring failure,
not correction-method evidence. The failed log is retained, the family-specific
binding above is required in the rerun, and a fresh output directory must be
used.

The complete eight-file exact-source snapshot passed its manifest/hash audit
but failed the bounded import preflight before model or GPU execution:
the D067 resolution-transfer module predates the as_model_state API required by
the maintained case loader. This closes the exact-source D070 route without
scientific interpretation; no compatibility functions will be backported into
the historical files.

A separate fallback is therefore registered as
historical_output_compatibility_current_core, with exact_replay_claimed=false.
It retains the digest-bound current source and compares each current free
next-state output with the retained D067 output through H30. The state guard
uses max absolute physical error at most 2e-5 and physical-state relative L2 at
most 1e-7. Increment drift is reported separately in frozen residual-scale RMS.
At every call, the large-band residual-scale RMS of the current-versus-D067
commutator drift is divided by the current baseline large-band mesh-defect RMS;
the maximum ratio must be at most 0.01. The original 2e-6 mixed-output field is
retained only as descriptive historical evidence and is not relabeled exact.
The smoke may cover H1, but full scientific interpretation requires all H30
compatibility rows, both adjacent pairs, all six cases, the state gates, the
science-scale ratio gate, pathway closures, hook equivalence, and
admissibility. Failure of either absolute or relative state compatibility or
the science-scale ratio stops pathway interpretation.

The full historical-output fallback D070B run
`d070_full_r1_current_compat` is retained as a failed contract, not pathway
evidence. Its exact 360-row compatibility inventory, finiteness, hook,
admissibility, band reconstruction, and pathway closure gates passed, with
maximum pathway/DCT/replacement closure `9.89062e-6`. Historical compatibility
failed: maximum state error was `4.604536e-2` against `2e-5`, maximum relative
state L2 was `4.288598e-4` against `1e-7`, and maximum large-band commutator
drift ratio was `1.02316856e-2` against `1e-2`. The summary is
`failed_contract`, scientific interpretation is false, and its arm/pathway
rows are not reused. This records material recurrence/source sensitivity but
does not identify a PCNO pathway.

A fresh dynamic-FV-only contract is registered as
`current_core_self_consistent_pathway_contract` (D070C). It asks which pathway
supports the mesh defect produced by the frozen checkpoint under one
digest-bound current evaluator; it neither rescues D070B nor reproduces or
explains the exact D067 implementation. Checkpoint, normalizer, split,
common-source family/reference artifacts, conservative restriction, all Python
source hashes, base HEAD, raw recurrence, FP32/no AMP, `model_all_nodes`, six
cases, both adjacent pairs, H30, and calls 1/5/15/30 remain frozen. D067 summary
and bundle hashes may be verified and reported descriptively, but
`exact_d067_replay_claimed=false` and
`cross_version_comparability_claimed=false`; no D067 difference enters a D070C
gate or selector.

D070C predeclares the following fail-closed checks. At every case, pair, mode,
and diagnostic call it verifies

`N_c-RN_f = (U_c-RU_f) + (F_c-RF_f)`

with maximum physical absolute closure and residual-scale RMS closure no larger
than `2e-5`. The shorter paired-input identity is reported only when the
teacher-forced common-source input restriction floor independently has maximum
physical error at most `2e-6` and residual-scale RMS at most `2e-5`. At every
diagnostic call it also verifies

`delta_free = delta_mesh + delta_state`,

where `delta_mesh=F_c(R U_f)-R F_f(U_f)` and
`delta_state=F_c(U_c)-F_c(R U_f)`, to the same dual `2e-5` limits. Every free
call verifies `e_(n+1)=e_n+delta_n` to the same limits. These are numerical
contract checks, not evidence of cancellation by themselves.

Independent trace replay covers every case, pair, mode, and diagnostic call:
direct backbone outputs against `trace_backbone_output` at both resolutions;
no-replacement native-branch traces against the direct coarse output;
restricted traced fine outputs against direct conservative restriction; and
same-hidden fine replay at every layer. Every comparison records maximum
physical absolute error, residual-scaled RMS numerator and denominator, and
relative error. The absolute and relative replay limits are both `2e-5`, and
the direct/trace large-band mismatch must be at most `0.01` of the corresponding
baseline large-band defect before a 20% arm reduction can be interpreted.

Arm, pathway-term, typed-closure, direct-output, commutator, recurrence, and
trace-replay inventories must equal their fully enumerated expected key sets,
with no missing, duplicate, extra, or nonfinite row. Selector baseline
total/large/local residual-scaled RMS denominators must exceed `1e-8`.
Pathway latent closures record absolute latent RMS and relative latent RMS and
must each be at most `2e-5`; physical trace closures use the replay limits;
DCT reconstruction absolute residual-scaled and energy-relative closures must
each be at most `1e-10`. A SHA-256 of every named state-dict tensor must be
identical before and after all interventions. Common-source teacher floors,
baseline admissibility, hook nonperturbation, reference binding, exact source
inventory, and all typed closures remain hard gates.

The selector executes only after every D070C hard gate passes. A missing or
numerically unresolved target, failed closure, or incomplete inventory yields
`failed_contract`, `mechanism_selection=null`, and a nonzero process exit.

Because the interventions act on the decoded residual head
`H_h=s_res*F_tilde_h`, D070C also binds that target to the stipulated increment
`F_h=N_h-U_h`. At every case, pair, mode, and diagnostic call it records the
native bridge

`(F_c-R F_f) - (H_c-R H_f)`

and the paired mesh bridge

`delta_mesh - [H_c(R U_f)-R H_f(U_f)]`.

Each bridge must have maximum physical absolute and residual-scale RMS closure
at most `2e-5`. Its large-band closure must be at most `0.01` of the
corresponding resolved `F` defect, whose large-band residual-scale RMS must
exceed `1e-8`. The bridge rows share the exact commutator inventory and must be
unique, complete, and finite. Thus the selector targets decoded residual-head
defects and may be interpreted for `F=N-U` only after this bridge passes;
generalized `N/U/F` algebra alone is insufficient under float32 recurrence.

The independently approved H1 D070C smoke source was frozen under snapshot label
`pcno_d070_d071_source_20260803h_d070c`: 86 Python files, base HEAD
`c6d959fabd9ce667d8efb92dd025fcd248a82553`, manifest SHA-256
`043afd3b1e1ceb5c394e13ed894aaa1badddbc5deb615e219d59b1e69c84300e`,
and archive SHA-256
`330d4f75349e91dfa843a2dd36ade3f479afc593b62781a2cbbc95d6c1515d4b`.
The analyzer and visualizer hashes are respectively
`458823c76a54582dca990e54ec681de81792b184e0e5f17d61db762bbf936e91`
and
`228dd0f6a62be72d67df0356a791fe01aa70bea2163cf36b936cec57f2921153`.
Two independent static reviews approve the smoke after the active-reference,
v2 visual-inventory, and residual-head bridge fixes; 64 focused CPU tests pass
and Ruff is clean. The smoke must use a fresh output directory and the explicit
`current_core_self_consistent_pathway_contract` selector.

The first real-checkpoint smoke from that snapshot stopped before science: the
model-state checksum attempted a byte view of a scalar state tensor and raised
before any case completed. The evaluator now flattens each contiguous tensor
before its byte view, and a scalar-buffer regression was added. The focused
suite again passes 64 tests and Ruff is clean. The replacement immutable
snapshot is `pcno_d070_d071_source_20260803i_d070c_scalar`, with the same 86
Python files and base HEAD, manifest SHA-256
`18afb8680a8b64fbb1933344217c055635a15a6fe9b3886c9c8d41f725fd488b`,
archive SHA-256
`e68a78e3245a587064424184cdf5b4cb660b4a75cda692bca13b7164e095c87d`,
and analyzer SHA-256
`cc1e937cf4f18b329f160a6d7f96ceb0e563bc5e5f196e883c82579a2a7fa5f3`.
An independent rereview approved the exact smoke-qualified snapshot for the
full run.

`multiple_supported` and `composite_or_unresolved` remain valid scientific
outcomes after a passing contract. The bounded claim is causal pathway support
within the frozen current evaluator and controlled same-hidden interventions;
it is not additive causal attribution, exact historical replay, a deployable
correction, or bump resolution transfer.

The scalar-safe D070C H1 smoke and six-case full run both complete. The full
summary has SHA-256
`20ecdf0da89fd36723b1084053a6a501a332d5171e3c68b85219e2939477f899`;
all ten registered output hashes verify after retrieval. Every identity,
decoded-head bridge, inventory, replay, closure, reference, admissibility,
hook, and model-immutability gate passes. The selector denominator minimum is
`0.03461593`, well above `1e-8`.

The terminal selector is `multiple_supported`. At layer 3, replacing the full
differential branch reduces the median large-band decoded defect by
`81.53/82.48%` on `125x50->250x100` and `87.78/87.76%` on
`250x100->500x200` in teacher/free modes. Replacing the fixed-hop composite at
that layer reduces it by `39.47/40.00%` and `55.06/62.18%`; fixed-hop layers 2
and 3 and full differential layers 1--3 qualify. The gradient-only arm does
not pass both-pair gates. Every same-hidden Fourier sub-arm, including
quadrature, subcell, analysis, synthesis, and full spectral replacement,
changes the pair-aggregated large defect by at most `1.45%` in magnitude;
pointwise is the commuting negative control. These nested nonlinear responses
cannot be added or interpreted as shares. The result supports the
differential pathway and makes fixed-hop geometry a priority intervention,
but does not isolate a unique subpath. The 21-output visualization manifest
SHA-256 is
`569ecf1b733be913f9185871414dc2a342cfa8b2459aa8ee22574b26a094ef97`.

D071 also completes under both family-local contracts, with every registered
output hash verified after retrieval. Dynamic FV uses selected cap `0.05`.
Its combined median final-state ratio is `0.91635` across all 18
case-resolutions, but every `125x50` case is worse; resolution medians are
`1.11538`, `0.90191`, and `0.87361`. The persistent-only medians are
`1.02438/0.89020/0.90167`, whereas local-only medians are
`1.09862/1.01633/0.96671`. The coarse-grid vortex-region median is `2.37616`
for local dissipation and `2.29525` for the combination. Thus the useful
dynamic effect is chiefly persistent and resolution dependent; a median win
does not satisfy the per-case or control gates. The dynamic summary and
39-output visual-manifest hashes are respectively
`2bdf1939a03cba5aa0006a47f711c214c441ce0833b8ba84cd45806dff770188`
and
`72f3464e82f1bd4ff3eb53f0b26eb7feb17dfc4b644e5a36e3b2a9466713ad6f`.

Bump uses selected cap `0.02` and remains native-graph/proxy-only. Combined
final-state ratios for cases `58/128/172/187` are
`0.86910/1.31349/0.92325/1.19447`, with median `1.05886`; the persistent arm
is nearly neutral and the local arm drives the case split. Its combined shock
control median is `1.05945 > 1.05`. The bump summary and verified 27-output
visual-manifest hashes are respectively
`53540b06810707e9dbb8c7c827c58e2add133ab5fb6282c558a618685c77fff7`
and
`c2346f4015ee56e201ebb7cab07787516f16571bde81fb6b7fdded623bd12c30`.
Neither family promotes D071. Both calibrations lacked a zero-cap candidate,
so cap selection forced a nonzero local filter; that evaluator limitation is
now a required control for any follow-up.

## Registered D073 Physical-Radius Differential Stage

D073 follows D070C before any revised D071 correction. It tests whether the
checkpoint differential branch is sensitive to the shrinking physical support
of its fixed two-hop average. It does not change the least-squares gradient,
`gw1`, Softsign, `gw2`, Fourier or pointwise paths, base graph topology,
physical-boundary connectivity, boundary tags, physical boundary policy,
recurrence, checkpoint weights, or targets.
The first stage is a frozen, same-hidden causal diagnostic at zero-indexed
layer 3 only.

D070C selected the pathway, layer, and this follow-up using the same six open
cases. D073 is therefore bounded adaptive/exploratory evidence, not a fresh
confirmatory experiment. More precisely, the base graph topology and physical-
boundary connectivity remain unchanged; the diagnostic intentionally changes
the smoothing support induced on that graph.

For graph-geodesic distance `d_G`, positive node weight `V_j`, and radius `r`,
the registered replacement is

`Q_ball(r,h) z_i = sum_{j:d_G(i,j)<=r} V_j z_j /
sum_{j:d_G(i,j)<=r} V_j`,

including the center node. Ball membership uses unrestricted shortest-path
distance `d_G`. Let `d_G^(<=2)` be the shortest accumulated physical length
among paths constrained to at most two hops, let `r_h^(2)` be the median over
family-local type-0 nodes of the maximum `d_G^(<=2)`, and let
`r_star=r_250x100^(2)`. Geometry alone fixes every radius before model outputs
are read. The dynamic arms are:

- `A0`: native repeated two-hop average;
- `A1`: volume-weighted ball kernel at the mesh-local `r_h^(2)`;
- `A2`: the same ball kernel at fixed physical radius `r_star`.

`A0-A1` measures the confounded kernel change. `A2-A1` is the primary
physical-width contrast within one ball-kernel family. At `250x100`, A1 and A2
must have identical neighborhood/weight operators and outputs within the
declared floating-point tolerance. The layer-3 fine prefix is native; its
restriction supplies the common coarse hidden field. Every arm modifies both
members of an adjacent pair consistently, while the other branches and final
decoder remain unchanged.

D073-A inherits the exact D070C checkpoint, normalizer, data, restriction, and
evaluator-provenance bindings, but requires a new immutable D073 source
manifest because the pathway utility changes. The evaluator compares the
checkpoint, normalizer, split, data manifest, family manifest, D067 binding,
base Git head, normalization mapping, and an explicit inventory of unchanged
imported sources against the accepted D070C summary. Two common sources are
explicitly allowed to differ: the registered physical-radius pathway utility,
and the artifact helper's source-snapshot-v5/backward-v4 compatibility change,
which does not alter D073 mathematics. It uses the same six open cases,
`125x50/250x100/500x200`, common-source conservative restriction, calls
1/5/15/30, teacher-forced inputs, and current-core self-consistent baseline-free
inputs. It reports
physical-volume and residual-component-scaled total, large
(`wavelength>=0.125`), transition, and local commutator RMS; conservative
components; arm response magnitude; and exact operator/output inventories.

For field `x`, residual-component scale `S`, and coarse physical volumes `V`,
the primary norm is

`||x||_(V,S)^2 = sum_i V_i sum_k (x_ik/S_k)^2 / sum_i V_i`.

For arm `a`, case `s`, pair `p`, input mode `m`, band projector `P_b`, calls
`T={1,5,15,30}`, and decoded commutator `d`, define

`D(a,s,p,m,b) = sqrt(mean_{t in T} ||P_b d(a,s,p,m,t)||_(V,S)^2)`

and `rho(s,p,m,b)=D(A2,...)/D(A1,...)`. Thus calls are repeated observations,
not replicates; each case is aggregated by RMS over its four calls before any
across-case median. An A1 denominator at or below `1e-8` in total, large, or
local bands invalidates the A2/A1 stratum rather than receiving an epsilon. An
A0 denominator at or below `1e-8` in those same bands independently invalidates
the A2/A0 native-relevance gate. Transition is reported but is not a denominator
or promotion gate. Node or call rows are never pooled across meshes or treated
as independent cases.

The D073-A contract stops interpretation unless all ball weights are finite and
nonnegative, every exact float64 row sum is one within `1e-12`, and executed
float32 row-sum and constant-field maximum errors are at most `2e-6`. It also
requires `r_125^(2)>r_250^(2)>r_500^(2)`, nonzero A1/A2 changed-row fractions
on both non-anchor grids, exact reciprocity of the supplied directed edge set,
complete radius/operator inventories, and exact sparse-operator hashes.
Inventories include row/NNZ counts; neighbor-count, realized-distance,
support-weight, and coefficient quantiles; boundary truncation; and family-
local source/target node-type composition. No symmetrized execution is
interpreted if reciprocity fails.

`A0` is the single native layer-3 execution reused by identity inside the
same-hidden trace; it is not a second CUDA replay. Independently built A1 and A2
sparse arrays at `250x100` must be exactly equal before the identical prepared
operator and result are shared by identity. A focused CPU regression separately
compares the constructed A0 path with the independent native-subpath trace.
Model state must remain immutable, reference/restriction and hook contracts must
pass, all outputs must be finite, and the existing D070C decoded-head, trace,
DCT, and closure tolerances remain satisfied.

This within-run A0 native-execution reuse is distinct from replaying the older D067
bundle under current source. D070C already recorded that cross-version D067
exact replay is false (the current-core discrepancy is about `0.070` maximum
absolute), so D073 audits that replay row for completeness and finiteness but
does not put it in the scientific contract. Scientific interpretation instead
depends on the accepted exact-hash D070C result and the explicit compatibility
checks above. The active recurrence itself is current-core, not read from the
D067 bundle: the evaluator gates raw recurrence, `model_all_nodes`, the exact
state handed to each call, and `prediction = state + increment` within `2e-6`.

The physical-width mechanism gate is applied conjunctively to all four
adjacent-pair by teacher/baseline-free strata: A2 versus A1 requires
`median_s(1-rho_large)>=0.20`, `rho_large<1` in at least four of six cases, and
median total and local `rho` no larger than `1.10`. D073-B additionally requires
A2 to remain competitive with native A0 in every stratum: median A2/A0 large
ratio at most `1.05` and median total/local ratios at most `1.10`. Passing
A2/A1 alone establishes sensitivity within the ball-kernel family; it does not
prove that shrinking native two-hop support uniquely caused the A0 defect. A
negative result rejects only this layer-3 ball-radius replacement, not physical-
scale graph operators generally.

Only a passing D073-A mechanism gate may advance to D073-B modified free
rollouts. That stage compares A2 with both A0 and A1 and requires no completion
or admissibility loss and no more than 5% worsening in any per-case truth
residual, final state, shock, vortex, smooth-high-pass, or boundary control.
Layer 2 and joint layers 2+3 are later sensitivities, not part of the first
run.

D073-A writes exact-hashed `geometry_inventory.csv`, `operator_checks.csv`,
`arm_metrics.csv`, `component_metrics.csv`, `case_aggregates.csv`,
`gate_summary.csv`, `closure_checks.csv`, `completion.csv`,
`reference_checks.csv`, `replay_metrics.csv`, selected same-hidden visual
payloads, and
`summary.json`. Its visual contract includes radius/support inventories,
per-case plus median/IQR absolute and ratio curves, case-sign panels, and
shared-residual-scale four-call spatial views for preregistered cases
`sv_e00_y00` and `sv_e11_y08`. These are labeled same-hidden interventions,
not modified rollouts, and cannot establish temporal cancellation. If D073-B
is reached, every-call payloads add instantaneous and cumulative total/large/
local defects, temporal coherence, and signed-growth fields; only that stage
may make accumulation or cancellation claims. Every result and visual artifact
uses an exact v2 manifest rather than directory globbing.

Implementation audit on 2026-08-03 completed the unrestricted `d_G` and
hop-constrained `d_G^(<=2)` constructions, freezes geometry and all radii before
the first D073 model output, supports the checkpoint's 64-channel latent DCT
path, records A2-minus-A1 response and boundary-distance/type support strata,
and fail-closes decisions plus manifest science flags on any contract failure.
The pre-smoke focused D070/D073 synthetic CPU suite passed `87/87`. The visualizer binds
the exact result manifest and exact 35-file output inventory: 18 static
PNG/PDF files, 16 four-frame GIFs with one physical scale and a 2:1 domain
aspect, and one per-call scale-saturation audit CSV. The immutable smoke source
snapshot contains 91 Python
files at base `c6d959fabd9ce667d8efb92dd025fcd248a82553`; its manifest and archive
SHA-256 digests are respectively
`5bc3620144ba9bfe5cc3a1512bc4cfcdf0c3035dd8d71101c0fbda42aec42a5b`
and
`5342440a79298fd6b76a40248bfe21a46d533f1a7972de694b2c9079a9718c83`.
The evaluator additionally loads D070C's exact-hashed `reference_checks.csv`
and, before any D073 output, requires all six frozen-training and active
common-source reference artifact hashes, retained resolution, state dtype, and
state shape to agree exactly.

`D073A-SMOKE-C-R1` was the registered one-case/H1 deployment smoke. It stopped
correctly with `smoke_failed`; no scientific decision was emitted and no full
run was launched. Runtime was `79.65 s`, including `29.4 s` for the active
case. Checkpoint, normalizer, source, all-six common-source references, radius
ordering, sparse operators, active recurrence, model immutability, completion,
exact inventories, visual payloads, numerical finiteness, and DCT closure all
passed; maximum DCT closure was `2.27e-13`. Eight of 96 execution-closure rows
failed. The failures were separate CUDA scatter executions of algebraically
identical paths: anchor A1/A2 smoothed-gradient maximum absolute differences
were `7.63e-6` to `2.29e-5` with relative scaled errors only `2.45e-8` to
`2.74e-8`; separately recomputed A0/native decoded outputs differed by
`2.81e-5` to `1.51e-4`, with relative scaled errors `6.94e-7` to `4.68e-6`.
This is an evaluator replay/atomic-reduction defect, not D073 mechanism
evidence.

The minimum repair reuses the already computed native final-layer path for A0
and the same prepared/result object for exactly identical A1/A2 anchor
operators. It does not alter graph support, weights, checkpoint parameters,
inputs, recurrence, boundary policy, or non-anchor interventions. A focused
CPU regression now asserts object reuse and passed `34/34`; a new immutable
source snapshot, independent review, and a fresh smoke directory are required
before any full run.

The repaired immutable snapshot is `D073-SOURCE-D`, still 91 Python files at
the same base. Its manifest and archive SHA-256 digests are
`06c6398f5af864b886bbf3c37a138a0e1bd562fb7dc420f64638e9ae1fb9a580`
and
`35dfa6f20ea9bfd188f888aed8e5c8b522d51594ee671b38edcf49fbd49f4fd0`.
The complete focused D070/D073 suite passed `87/87`. This does not retroactively
validate the failed smoke; snapshot D requires independent review and a fresh
smoke run.

Three independent reviews accepted `D073-SOURCE-D` for a fresh smoke after
confirming that A0 and the anchor A1/A2 paths reuse only identical executions,
while both non-anchor A1/A2 contrasts remain distinct and scientifically active.
`D073A-SMOKE-D-R1` then completed with `smoke_complete` in `77.99 s`, including
`27.7 s` for the active case. All 96 execution-closure rows, active recurrence,
all-six common-source references, D070C compatibility, geometry/operator,
immutability, completion, finite-value, visual-payload, and exact-inventory
checks passed. The maximum DCT closure was `2.27e-13`; the explicitly non-gating
historical D067 current-source replay remained inexact at only `2.92e-6` maximum
absolute error. The exact 15-output smoke manifest records summary SHA-256
`de1db5b13a793f2465dd96dd92e7d99b87553d1cb709ea45ab801aba0f19ecb7`.
This authorizes one full D073-A run from the same immutable snapshot; it does not
authorize D073-B or establish a physical-radius mechanism result.

`D073A-FULL-D-R1` was launched from that exact snapshot at
`2026-08-03T09:32:49Z`. The single 20-second launch-health audit found the
evaluator alive with an active CUDA context and no traceback or failure marker;
stdout was still buffered during the preamble. No mechanism decision is recorded
until the exact full result manifest and all contract gates are verified.

The full run then completed in `770.14 s`. Its exact 19-output result manifest
and summary SHA-256 digests are respectively
`4b872a6e2310166daaf90bb449eed21dccda0f64850baa64d62d611cf51a4d1d`
and
`bff1f83ee797d8b29de662130211734cecaf0375255b61b8666423e98ef88a77`.
All 2,304 closure rows pass, including exact A0 reuse and anchor A1/A2 identity;
maximum DCT closure is `6.82e-13`. The maximum active common-source restriction
cross-check is `1.78e-15`, operator row-sum and constant-field errors are at most
`5.96e-7`, all six baseline trajectories remain admissible, and all result,
payload, finite-value, recurrence, reference, and exact-inventory gates pass.
The separately audited historical D067 current-source replay remains non-gating:
its maximum discrepancy reaches `0.0905` on `250x100->500x200` and is not used
as a scientific D073 input.

For `r21=D(A2)/D(A1)` and `r20=D(A2)/D(A0)`, lower is better. Calls
`1/5/15/30` are RMS-aggregated inside each case before the six-case median:

| Pair | Input | `r21` total/large/local | `r20` total/large/local | Large reduction | Positive cases |
| --- | --- | --- | --- | ---: | ---: |
| `125x50->250x100` | teacher | `0.561/0.590/0.684` | `0.622/0.677/0.621` | `41.0%` | `6/6` |
| `125x50->250x100` | baseline-free | `0.563/0.600/0.686` | `0.624/0.681/0.624` | `40.0%` | `6/6` |
| `250x100->500x200` | teacher | `0.350/0.439/0.309` | `0.366/0.435/0.328` | `56.1%` | `6/6` |
| `250x100->500x200` | baseline-free | `0.221/0.336/0.205` | `0.287/0.336/0.274` | `66.4%` | `6/6` |

The leave-one-case-out minimum large-band reductions are `39.8/39.3/55.5/66.1%`.
The smallest gated denominator is `0.0479`, far above `1e-8`. Median absolute
A1-to-A2 total defects fall `0.129->0.072`, `0.127->0.071`, `0.179->0.063`,
and `0.468->0.104` across the four rows above, so the result is not a
small-denominator artifact. A2/A1 is below one in all 96 decoded
case-by-band rows, including the non-gating transition band, and in all 288
gated conservative-component rows.

The response is introduced exactly after the graph-ball operation: A2/A1 is
one before the ball and below one at the smoothed-gradient, Softsign, `gw2`, and
decoded levels. Contracting the coarse `125x50` radius from `0.04` to `0.02`
and expanding the fine `500x200` radius from `0.01` to `0.02` both help. This
opposite-direction result supports physical-support alignment rather than a
generic preference for more or less smoothing. Teacher and baseline-free rows
agree at call 1; the much larger later benefit on `250x100->500x200`
baseline-free inputs shows an additional interaction with already-diverged
states. It does not show that an A2 recurrent rollout prevents that divergence.

The exact eight-payload post hoc signed audit, limited to the two registered
visual cases, finds `cos(A1,A2-A1)<0` for every call and total/large/local field:
the ranges are `-0.946` to `-0.738` for `125x50->250x100` and `-0.992` to
`-0.881` for `250x100->500x200`. The corresponding relative squared-defect
growth is always negative. Thus the intervention response opposes the A1
commutator rather than merely moving its energy between components. This signed
audit is descriptive, not an additional promotion gate.

The frozen visualizer passed `5/5` focused tests and wrote the exact 35-output
suite plus manifest. Its manifest SHA-256 is
`9ce160fb67ccba3c5958f23640948ddd32c2c7fd77e5c5ae4bdf3408ceb65244`;
all 288 payload/CSV RMS comparisons agree within `4.45e-10`, and A2-minus-A1
closure is within `3.73e-9`. Static scales and all large-band movie panels avoid
clipping. Local extrema reach `5.26` residual scales and at worst clip `6.24%`
of nodes under the deliberately shared plus/minus-one-scale view; the exact
1,616-row saturation CSV records every frame.

D073-A therefore passes both the physical-width mechanism and native-relevance
gates and authorizes D073-B. It establishes a strong causal sensitivity to
layer-3 physical radius within the graph-ball family and makes differential
geometry the leading diagnosed resolution pathway. It does not uniquely prove
that native repeated two-hop support caused the defect because A0 uses a
different kernel; radius also changes cardinality, weights, boundary truncation,
and node-type composition. It is same-hidden evidence from one checkpoint and
six adaptively reused open dynamic-FV cases, not corrected-rollout truth error,
shock/vortex improvement, conservation, resolution invariance, general operator
learning, or bump portability.

The bump arm, if reached, uses the same graph-geodesic construction with
separate native-graph radii and proxy weights. A graph-local radius is compared
with a geometry-only fixed family radius frozen from the disjoint open
calibration geometries. Solid-boundary connectivity is never augmented. The
four D068 evaluation geometries retain bump meanings `0 normal, 1 wall,
2 outflow, 3 inflow` and report native-graph residual/state, shock, smooth,
boundary, completion, and admissibility controls only. This is a portability
and safety control, not conservative restriction, a commutator experiment,
PDE resolution transfer, or physical conservation.

Any later D071B is separately adaptive because D071 evaluation outcomes shaped
its design. It must include exact zero gain/cap, keep persistent and local
channels separate, use case-level recurrent cross-fitting inside the disjoint
calibration population, and must not load evaluation targets during
calibration. Reusing the existing evaluation cases can support only bounded
adaptive evidence; a confirmatory claim requires a fresh preregistered
nonsealed population. Data assimilation remains reserved.

## Registered D074 Native-Resolution Correction Stage

The 2026-08-03 owner direction makes native-resolution residual correction the
primary practical track. Cross-resolution commutators and D073 remain important
mechanistic evidence, but a direct off-grid rollout is not a useful practical
method unless it beats a frozen pipeline that transfers the initial state to
the checkpoint's training grid, rolls out there, and transfers predictions back
to the query grid.

D071 already supplies the development signal for the first arm. At the dynamic
training grid `250x100`, the persistent rank-eight correction improves all six
H30 endpoint errors, with median corrected/baseline ratio `0.89020`. Its
RMS-over-call residual-error ratio is `0.9749`, again with six of six cases
improving. The local-dissipation arm instead has native ratios `1.01633` for
H30 state and `1.1118` for RMS-over-call residual error; its local-band ratio is
`1.2349`. The combined arm reaches endpoint ratio `0.90191` while worsening the
RMS-over-call residual error to `1.0777`, confirming that recurrent signed
accumulation matters. On bump native graphs, persistent correction is nearly
neutral (median H20 state ratio `0.9893`) and local/combined corrections are
harmful in the median. These are adaptively reused open-validation results, not
a fresh confirmation.

### D074-A: native correction

Implementation qualification on 2026-08-04 found no remaining blocker for the
contract-only dynamic H2 smoke. The final evaluator separates candidate-matrix
inventory from eligibility, rejects unequal-horizon ratios, freezes and
revalidates the complete selector bundle before and after evaluation, gates
exact row inventories, and stores per-call corrected-defect and same-corrected-
input base-defect sequences. Its projection audit uses family-neutral
`type0_fit_support` and `excluded_non_type0_support` language, componentwise
weighted-QR closures, constant/nonconstant cross energy, exact zero-arm object
identity, and replay-checked visual payloads. Ruff and compile checks pass; 44
focused tests and 120 broader CPU dependency tests pass. Three independent
read-only reviews return GO for H2 only. This is implementation evidence, not a
model result.

The immutable source manifest has schema
`pcno_isolated_source_manifest_v1`, SHA-256
`3fbd80595374587ba193c7694a5b894d21f5664dc44e63fa4e200666e0ef5237`,
base HEAD `c6d959fabd9ce667d8efb92dd025fcd248a82553`, and 92 bound Python
files under `pcno/`, `utility/`, and `scripts/time_dependent_no/`. The exact
source copy is retained beside the manifest in the ignored artifact tree. No
checkpoint inference has run under this source identity yet.

D074-A inherits exact family-local artifacts rather than the shorthand "D071
checkpoint." Dynamic binds checkpoint SHA-256
`95e6c180a3298c4b38662d8c6cb77301c0e7fd0286e9a53571bddc487b8379c9`,
normalization-file SHA-256
`4c931c813d318f9a3012814c9803cf85fbb30ce4c68739535047b2aff6f0faf4`,
normalization-mapping digest
`9df7efb2f1aadf315f399dcabf3b366bf1b2f46794b026cbc5b7d80ba22e1a1a`,
split SHA-256
`1be17494eaac3902159763a9e9a6c562d39c31957739899f50c28e37b48a921d`,
data-manifest digest
`f8d228ae3c6e08fe6697f37df21476fe621abc354d0b0a6eed2a623ed2b9f96c`,
and family-manifest SHA-256
`2c0d0c516d38dc826c23edda76e1a0ef668ff4692caf72f1bd710150629c7dc8`.
Its calibration cases are exactly
`sv_e{01,02,03,04,05,07,08,09,10}_{y00,y08}` and its evaluation cases are
`sv_e{00,06,11}_{y00,y08}`. It uses only `250x100`, H30, and
`t_n=0.02 n` for `n=0,...,30`.

Bump binds checkpoint SHA-256
`2bb5ee3ca831a6ffc498f01e309ae7ea957b2df0f411023fb3812919a6732964`,
normalization-file SHA-256
`717a948a1f219af9eabd6aafeacbdb8e2d00a362133e6a2b6b4408e60b71b5af`,
split SHA-256
`ba648aa0bf404f61f5f8ceabf2be963efda9935ca325908435c3c52bad88519f`,
and data-manifest digest
`5d5373fdcc682544bf330fba6d54fe65509dabe936baee7443c71a8c0c8d9fa7`.
Its calibration cases are exactly
`7/16/18/23/47/54/60/72/82/101/103/112/120/126/141/145/150/188/190/211/227/233/235/251/287/296`,
its evaluation cases are `128/172/58/187`, and it uses H20 with
`t_n=0.025 n` for `n=0,...,20`. Both families retain FP32/no AMP,
`model_all_nodes`, raw recurrence, unchanged physical boundary handling, and
their own node meanings. Before any smoke, a new immutable D074 source snapshot
must bind every Python file under `pcno/`, `utility/`, and
`scripts/time_dependent_no/`, the 40-digit base HEAD, and every runtime-loaded
source. The prior D071 source-manifest SHA-256
`77f347db8b81d6058f6aa477e33523e805fd2d4b14b97d359679dcd25ff1af8f`
is inherited provenance, not the future D074 source identity.

For support `S`, component scale `D`, and weights `w_i`, define

`||x||_(W,D,S)^2 = sum_(i in S) w_i sum_c (x_ic/D_c)^2 /
                    sum_(i in S) w_i`.

Dynamic `W` contains physical cell volumes. Bump `W` contains only the frozen
proxy weights. `D_s` is the frozen state scale and `D_r` the frozen residual
scale. All reported ratios are formed only after within-case, within-support
aggregation. A required baseline denominator at or below `1e-8`, or a missing
or nonfinite numerator or denominator, makes that required ratio undefined and
fails the contract; no epsilon silently repairs it.

Calibration fits the teacher-forced residual error

`epsilon_n = F(U^n) - r_n`,  where `r_n=U^(n+1)-U^n`,

on family-local type-0 nodes. For the first `r` physical cosine modes `Phi_r`,
weighted least squares produces call-indexed coefficients `a_(j,n)`. A frozen
candidate correction is

`C_(r,alpha)^n(x) = -alpha D_r Phi_r(x) mean_j(a_(j,n))`,

with exact zero support on all non-type-0 nodes. Application is the clock-
conditioned, target-free recurrence
`Uhat^(n+1)=Uhat^n+F(Uhat^n)+C_(r,alpha)^n`. It is an offline climatological
bias schedule, not state feedback, data assimilation, conservation, or an
autonomous operator correction.

The unique candidate inventory is one canonical `ZERO` arm plus the 16 pairs
`r in {2,4,8,16}` and `alpha in {0.25,0.5,0.75,1}`. `ZERO` reuses the exact
baseline predictions bit-for-bit and is never rerun once per nominal rank.
Folds are the lexicographically sorted calibration case IDs. In each fold, the
held-out case contributes neither teacher-forced errors nor fitted coefficients;
the candidate is fit on every other calibration case and then recurrently
rolled out on the held-out case. The complete 17-candidate-by-case out-of-fold
matrix is an inventory gate.

For arm `a`, define on its own recurrent input

`b_n^a = F(Uhat_a^n)-r_n`,  `delta_n^a=b_n^a+C_n^a`, and
`e_n^a=Uhat_a^n-U^n`.

Native starts are aligned, so every call must close
`e_(n+1)^a=e_n^a+delta_n^a`. Per-case residual RMS is
`[H^-1 sum_n ||delta_n^a||_(W,D_r)^2]^(1/2)`; its aggregate relative form is
`[sum_n ||delta_n^a||_(W,D_r)^2 /
  sum_n ||r_n||_(W,D_r)^2]^(1/2)`. Also report instantaneous
`||delta_n||_(W,D_r)/||r_n||_(W,D_r)`, absolute `D_r`-scaled cumulative error
`||e_N||_(W,D_r)`, cumulative truth-relative error
`||e_N||_(W,D_r)/||sum_(n<N) r_n||_(W,D_r)`, and coherence
`kappa_N=||e_N||_(W,D_r)/sum_(n<N)||delta_n||_(W,D_r)`. Raw numerators accompany
every true-increment denominator.

A nonzero calibration candidate is eligible only when every fold is complete,
admissible, finite, has H30/H20 state-error ratio no larger than `1.00`, has
residual-RMS ratio no larger than `1.00`, and has every primary structure,
component, and global-quantity error ratio no larger than `1.05`. `ZERO` is the
fallback provided the baseline contract passes. Among `ZERO` and all eligible
nonzero candidates, select lexicographically by median endpoint state ratio,
worst endpoint state ratio, median residual-RMS ratio, lower rank, then lower
gain. The full float64 score tuple and candidate key are serialized; selection
is never case- or metric-specific. After selection, refit exactly once on all
calibration cases, hash the frozen coefficients and selector record, and only
then load evaluation targets.

At evaluation, the dynamic gate requires complete admissible H30 rollouts,
median selected/baseline H30 `D_s` state-error ratio at most `0.92`, every case
ratio at most `1.00`, and every case's residual-RMS ratio at most `1.00`.
Primary no-harm controls are endpoint shock-region state error, vortex-core
state error, smooth-region high-pass state error, fixed-physical-distance
boundary-band state error, per-component residual RMS, and per-component
physical-volume integral truth error over calls and at H30. Every required
ratio must be present, finite, and no larger than `1.05`. DCT
large/transition/local curves remain secondary mechanism diagnostics rather
than duplicate promotion gates.

The signed analysis separates `2<e_n,delta_n>`, `||delta_n||^2`, and net growth
under one `W,D_r` inner product. Correction efficacy is evaluated at the same
corrected input through `2<C_n,b_n>+||C_n||^2` and `cos(C_n,b_n)`; comparison
with the baseline arm's different trajectory is not substituted. Zero-vector
cosines are null with an explicit validity flag. An unregularized weighted-QR
projection, separate from the ridge fit, partitions type-0 error into the
selected cosine subspace and its exact weighted orthogonal complement; non-type-0
energy is a third family-local partition, not called orthogonal. Dynamic
presentation may name those nodes contacts; bump presentation must retain
wall/outflow/inflow meanings. Store orthogonality, energy,
recurrence, signed-growth, and reconstruction closures, plus constant versus
nonconstant modes and component/global budgets.

The minimum D074-A visual suite contains all-case state-scale error,
instantaneous `D_r` residual error, cumulative `D_r` error and coherence;
separate interaction, innovation, and net-growth curves with same-input
correction efficacy; selected-subspace, type-0-orthogonal, and excluded-support
energy before/after correction; constant/nonconstant and component/global-budget
curves; and one compact no-harm ratio chart. Two representative all-call
animations use one rollout-wide physical scale per component and show true
increment, uncorrected defect at the corrected input, correction, corrected
defect, cumulative error, and signed local growth density. Raw predictions and
DCT-band movies are added only when they answer a nonredundant question. No
per-frame normalization is permitted.

The bump arm repeats the selector with bump meanings `0 normal, 1 wall,
2 outflow, 3 inflow` and proxy weights only. Selecting `ZERO` is a valid
negative portability result. A nonzero bump arm promotes only with complete
admissible H20 rollouts, median H20 state ratio at most `0.95`, at least three
of four case ratios no larger than `1.00`, every case state and residual-RMS
ratio no larger than `1.05`, and every proxy-weighted shock, smooth/high-pass,
boundary, component, and proxy-mean control ratio no larger than `1.05`. No
dynamic DCT, physical-conservation, or resolution language transfers to bump.

Because the present local-dissipation proposal is harmful at the dynamic
native grid and heterogeneous on bump, it is not a D074-A arm. A later local
method requires its own canonical zero selector and a target-free signed
no-harm signal before GPU evaluation. The first smoke is dynamic only,
contract-only, and non-scientific: one cross-fit fold, one evaluation case,
`ZERO` plus one nonzero candidate, and H1/H2 coverage. A full run remains
unauthorized until the selector, target-loading order, projection closures,
metric inventories, and zero/baseline identity pass focused CPU tests.

### D074-B: transfer-native practical baseline

D074-B is dynamic-FV only and rejects bump at argument validation. For a query
grid `q`, let `T_q` map query conservative cell averages to native `250x100`
and `S_q` map native predictions back. At `q=500x200`, `T=R` is exact block
averaging and `S=P` is piecewise-constant injection. At `q=125x50`, `T=P` and
`S=R`. Transfer only the four conservative state channels. Geometry, volumes,
node types, boundary masks, graph edges, gradients, and Fourier structures are
regenerated independently on every model/query grid; dynamic meanings remain
`0 interior, 1 y-symmetry contact, 2 x-extrapolation contact, 3 both`.

All conservative maps execute in float64. `as_model_state` is called once on
the actual model grid, and predictions return to float64 before `S_q`.
NumPy/Torch map agreement, exact float64 mapping closure, and the realized FP32
model-input gap are separate contract rows. Prolongating `125x50` creates a
piecewise-constant, potentially out-of-distribution native input; copying a
coarse boundary-cell state does not copy its type label to interior children.
Boundary/type-band mapping errors are therefore reported rather than hidden.

The direction-specific truth quantities are

`o_q^n = S_q T_q U_q^n - U_q^n`                 (query round-trip loss),
`m_q^n = T_q U_q^n - U_250^n`                   (native information mismatch),
`p_q^n = S_q U_250^n - U_q^n`                   (pipeline truth floor).

They must close `o_q^n=S_q m_q^n+p_q^n`. For `500x200`, `m_q` should be near
roundoff under the common source while `o_q` and `p_q` contain the fine subcell
loss. For `125x50`, `o_q` and `p_q` should be near roundoff while `m_q` exposes
the coarse-to-native information deficit. Thus a generic remapping-oracle norm
is not used for both directions. Report all three raw fields and norms, their
closure, the analogous mapped true-residual floor, and the initial value
`o_q^0`; plot only nonredundant curves when two floors coincide. No floor is
subtracted as if orthogonal.

For a transfer-native prediction `Utilde_q^n=S_q Uhat_250^n`, define
`e_n=Utilde_q^n-U_q^n` and
`delta_n=S_q[Uhat_250^(n+1)-Uhat_250^n]-(U_q^(n+1)-U_q^n)`. The exact query
identity is `e_n=e_0+sum_(j<n)delta_j`; at `500x200`, `e_0=o_q^0` is generally
nonzero. Also decompose the query error as

`e_n = S_q[Uhat_250^n-T_q U_q^n] + o_q^n`,

and report both vector closure and the inner product/cosine between the
evolution and round-trip terms. Norm-squared subtraction is prohibited absent
an observed orthogonality result.

The arms are direct query-grid raw rollout, raw transfer-native rollout, and a
matched-information direct `500x200` control initialized with
`P R U_500^0`. The selected-corrected transfer-native arm is primary only if
D074-A has already passed its frozen evaluation gate; otherwise raw transfer-
native is the predeclared primary comparator and corrected transfer is omitted
from selection. A direct D073 physical-radius rollout enters only after it is
implemented and passes its own recurrence/identity checks. No arm or comparator
is chosen per case or metric after evaluation.

For each query grid, success is predeclared as

`median_i E_direct(i,H30) / E_primary_transfer(i,H30) <= 0.95`,

with at least four of six paired case wins. `E` is the query-grid `W,D_s`
state error. Report all six ratios, leave-one-case-out median sensitivity,
trajectory AUC, residual-scale cumulative diagnostics, endpoint shock, vortex,
smooth/high-pass, boundary and component ratios, completion/admissibility, wall
time, peak memory, mapping overhead, and error-versus-cost. Every primary
control denominator must exceed `1e-8` and every direct/primary-transfer no-harm
ratio must be at most `1.05`. Passing yields bounded adaptive practical evidence
on the reused open cases, not confirmation. Failure leaves D073 as bounded
mechanism/theory evidence.

D074-B figures keep `125x50` and `500x200` separate and show all-call query
error with the three direction-specific floors, all six H30 paired ratios, and
the cost table. One representative fixed-scale animation per query grid compares
direct raw, raw transfer-native, selected-corrected transfer-native when
eligible, direct physical-radius when implemented, and the nonredundant truth
floor against the same query truth.

Piecewise-constant injection remains the primary map because it is already
implemented and conservative for nested cell averages. Before execution,
focused tests must cover both production directions, per-component integral
preservation, `R P=I`, `P R` idempotence, impulse ordering, NumPy/Torch and
dtype-order agreement, all directional identities above, regenerated node
types, fixed comparator selection, target-loading separation, gate arithmetic,
cost inventory, nonzero `e_0` at the fine query, and hard bump rejection. A
higher-order reconstruction is not introduced unless this simple baseline is
shown to be decision-limiting.

## Other Stable IDs

| ID | Family and topic | Terminal evidence state |
| --- | --- | --- |
| L1-OOD | Existing mild-support 1D OOD artifacts | Completed, provenance-limited. |
| XLINE-001 | Cross-line rollout-instability attribution | Not run; historical mixed plan closed. |
| L4A-001 | Zero-training latent representation/closure preflight | Completed; route stopped before learned forecast. |
| L4A-002 | Frozen 2D autoencoder capacity smoke | Completed; reconstruction/front gates failed. |
| L4A-003 | Decoder code-reachability oracle | Completed; decoder manifold rejected. |
| L4A-004 | Conservative local-Haar capacity preflight | Completed; discontinuous regularity helped but remained insufficient. |

## Topic Routing

| Question | First IDs to inspect |
| --- | --- |
| D019 versus D041 provenance | L3R-M0, D019, D041 |
| Boundary descriptor use and node-type scaling | D063, D068, D069 |
| Constant-zero channels and initialization | D069 |
| Boundary closure and objective history | D040, D041, L3R-BG0, L3R-RB0, L3R-RA0P, L3R-K2D0 |
| Fresh versus propagated error | D053, L3R-K2D0 |
| Ripple/high-frequency mechanisms | D013, D036, D038, D043, D052--D056, D060 |
| Front position/identity | D006, D035, D060--D062 |
| Large learned steps | D027--D035, D060--D061 |
| Resolution transfer | D029, D030, D063--D067 |
| Residual/update commutator and accumulation structure | D063--D067 |
| Large-scale drift versus local cancellation | D064, D066, D067 |
| Native residual correction and transfer-native baseline | D071, D074 |
| PCNO resolution-pathway attribution | D052, D065, D067 |
| Physical conservation limits | D008, D037, D045--D051, RC-13 in the decision file |
| Latent forecasting readiness | L4A-001--L4A-004 |
| Longer horizon and admissibility | L3R-U0, L3R-U1, L3R-B1 |

## Archive And Recovery Anchors

- Historical decision SHA-256:
  5B446C93FA7487F7B97C895FC3CC38309E549F496656FDD87FC6FBC60BC6A538.
- Historical tracker SHA-256:
  D055F2EEB398224CEFDBDD86D3A230C501DFA4069D65DF188C7F8183F0F821CE.
- Commit ae1f402 preserves the D063 evidence addition.
- Commit e2070f6 preserves the complete active documentation immediately before
  this compaction and the reusable PCNO infrastructure cleanup.
- Commit ce5d6a2 preserves the source state immediately before that code cleanup.
- Commit 31e5765 preserves an earlier expanded documentation surface.
- Retired diagnostic entry points remain recoverable from commit ce5d6a2 or the
  earlier source anchor named by the relevant historical record.

Generated arrays, figures, logs, checkpoints, and manifests remain under the
ignored artifacts/time_dependent_no tree. This index is not a substitute for an
artifact digest or an exact retrieval request.
