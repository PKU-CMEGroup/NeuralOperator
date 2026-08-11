# Time-Dependent Experiment Index

Updated: 2026-08-11

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

- The primary run universe contains 109 IDs: D001--D080, 23 L3R IDs,
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

Every D-series ID is contiguous from D001 through D080. Exact D001--D063
populations, metrics, thresholds, artifacts, and claims remain in the
historical ledger. D064--D080 are routed in the post-archive section below.

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
| D072 | Fixed-physical-width semantic boundary fields | Complete, not promoted. The dynamic frozen-checkpoint intervention proves field use on one FP32 seed, but the three-seed BF16 ladder establishes no promotable N0-relative gain. The bump ladder is seed-unstable and neither field arm improves completion or common-case H79. No boundary-condition, resolution-transfer, or optimal-encoding claim follows. |
| D073 | Physical-radius differential geometry | D073-A completed; all four dynamic same-hidden mechanism and native-relevance strata pass, authorizing D073-B modified-rollout testing. Bump safety has not run. |
| D074 | Native-resolution residual correction and transfer-native comparator | Dynamic H30 r2 completes under the repaired source and frozen physical contract; all 23 result hashes pass. Every nonzero candidate/case pair improves endpoint error, but every arm violates at least one complete-case no-harm gate. The selector retains zero and promotion fails. `rank8_gain0p25` is the closest arm, failing only one energy-integral RMS row at `1.09823` versus `1.05`. H2 remains non-scientific; bump remains NO-GO pending an immutable D041 replay binding. |
| D075 | Conservative-integral-neutral native correction | Completed on dynamic FV. H30 selects `rank8_gain0p125_raw`; all six endpoint and residual ratios improve, but the median endpoint target and two energy controls fail, so promotion is false. Static integral neutralization is not the selected mechanism. Bump is outside this contract. |
| D076 | Short-window state-response gain pilot | Completed on dynamic FV under repaired immutable source. H2 r2 passes its non-scientific contract gate. H30 stops at `calibration_not_qualified` before evaluation targets load: 16/18 leave-one-case-out selections are safe, while both `e10` selections fail one energy-integral control. No correction promotes. |
| D077 | Strength-grouped response controller | Terminal `failed_contract`. Grouped calibration qualifies (`18/18` safe, 16 nonzero; endpoint/residual medians `0.94867/0.99590`) and the evaluation numbers pass every promotion inequality, but all six independent H5-prefix/H30 replays fail the unchanged FP32 tolerances, including both zero-correction cases. No scientific interpretation or promotion is accepted. |
| D078 | Deterministic CUDA replay attribution | Terminal `complete`. Corrected immutable source B passes H2-r2 and H30-r1 under deterministic CUDA. All six independent prefixes are bit-identical, all 40 H30 outputs verify, calibration qualifies, and the unchanged adaptive open-validation promotion conjunction passes. D077 remains failed; D078 is not independent confirmation or bump evidence. |
| D079 | Deterministic process confirmation | Terminal `complete`. A fresh process and clean output path recompute the exact D078 dynamic-FV H30 contract. All 40 declared scientific payloads are byte-identical to D078, both within-run hash inventories pass, all scientific summary fields match exactly, and all six prefix rows remain exactly zero. This confirms same-open-population deterministic process repeatability, not independent-data confirmation. |
| D080 | Causal local-channel correctability and bounded rollout pilot | Terminal `complete` on dynamic FV. D080-A supplies frozen-direction headroom, but no D080-B combined arm passes the recurrent H30 promotion gates. Shock corrections reduce residual RMS by about `0.17%` and help through roughly call 22, then reverse; both worsen the median shock endpoint. The vortex arm is placement-dependent and violates the global-control limit. The result supports time/state-conditioned local correction, not a stronger fixed filter. |
| D083 | Bump query-graph and transported-Fourier rotation consistency | Terminal execution complete: all 240 attempts are present, but only 61 reach H79. G1a is strong negative proxy-mass query-representation evidence under about 92% node compression, never PDE resolution transfer. G1b used `k'=Qk` and is retained only as a transported-Fourier analytic covariance negative control, not the owner-requested fixed-mode unseen-orientation test. |
| D085 | Fixed-checkpoint-Fourier 90-degree bump transformation | Terminal `complete` attempted matrix under the exact H2 source: all 120 phase-arm attempts and all declared artifacts verify. Every fixed-mode rotated proposal fails at call 1 (`0/30` accepted in each type arm), before recurrence. The correct-type call-1 median same-input defect is `2.3226` residual units with raw RMS `2.4057` against raw reference RMS `1.2343`. D083's transported-basis control is smaller but still large; fixed-world Fourier representation exacerbates rather than solely explains the failure. |
| D086 | Rotated finite-invalid failure visualization | Terminal `continuation_complete` on the three frozen visualization cases, both type arms, and two ordinary-CUDA repeats. All 12 continuations are physically inadmissible from call 1 but remain numerically finite through call 79. Shared-reference-scale movies show a domain-wide residual/pressure failure rather than a localized shock-only defect. These are failure continuations, never accepted rollouts. |

## Post-Archive D064--D080 Evidence

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
| D074 | `artifacts/time_dependent_no/pcno_native_residual_correction_d074_20260804c/d074_dynamic_h30_r2/summary.json` and sibling `d074_dynamic_h30_r2_visualization/manifest.json` | The scientific dynamic H30 run selects zero and does not promote. All 288 nonzero calibration pairs improve endpoints, while none passes the complete-case safety gates. The closest arm misses one energy-integral RMS row. The 27-output fixed-scale visualization bundle passes all declared hashes and has zero saturation in all 64 audited scale rows. The six evaluation cases exactly replay baseline because zero was selected. |
| D075 | `artifacts/time_dependent_no/pcno_integral_neutral_correction_d075_20260804a/d075_dynamic_h30_r1/summary.json`; sibling analysis and visualization manifests | The 26-output H30 result passes every contract/hash audit and selects raw rank 8 at gain `0.125`. Median endpoint/residual ratios are `0.97493/0.99526`; endpoints and residuals improve in all six cases, but only four pass control no-harm and promotion fails. Registered and structure-only visuals separate absolute defect magnitude from the smooth case-independent correction. This is adaptive open-validation evidence only. |
| D076 | `artifacts/time_dependent_no/pcno_response_gain_controller_d076_20260805b/d076_dynamic_h30_r1/summary.json` and sibling `d076_dynamic_h30_r1_analysis/analysis_summary.json` | The registered selector improves median endpoint/residual ratios to `0.93272/0.98622`, but selects unsafe gains on `sv_e10_y00/y08`; H30 terminates before conditional evaluation. A post-hoc strength-grouped fold replay passes the numeric calibration gates but is diagnostic only and cannot retroactively authorize evaluation. |
| D077 | `artifacts/time_dependent_no/pcno_strength_grouped_response_controller_d077_20260805a/d077_dynamic_h30_r1/summary.json`, sibling `d077_dynamic_h30_r1_analysis_v2/manifest.json`, and `d077_dynamic_h30_r1_response_animations_stride2/manifest.json` | All 40 declared H30 outputs and three visual payloads verify. The target-free selector chooses gains `0.25/1.0/0.5/1.0/0/0`; numerical selected/zero medians are `0.93440/0.99482`, worst control is `1.03810`, and the highest-strength pair safely abstains. The immutable contract nevertheless fails because prefix replay reaches `5.507e-5` absolute and `5.340e-7` relative versus `2e-5/1e-7`. The 27-output response animation suite is hash-exact, fixed-scale, unsaturated, and watermarked failed-contract diagnostic only. These values are diagnostic only, not a promoted result. |
| D078 | `artifacts/time_dependent_no/pcno_deterministic_response_controller_d078_20260805b/d078_dynamic_h30_r1/summary.json`, sibling `d078_dynamic_h30_r1_analysis/manifest.json`, and `d078_dynamic_h30_r1_response_animations_stride2/manifest.json` | H30-r1 is terminal `complete` with summary SHA-256 `959d3398...e86e`; every one of 40 declared outputs verifies. All six prefixes are exactly zero. The unchanged selector chooses `0.25/1.0/0.5/1.0/0/0`, and promotion medians are `0.93441/0.99483` with worst control `1.03811`. The 12-output figure and 27-output response-animation bundles are scientific, hash-exact, unwatermarked, fixed-scale, and unsaturated. |
| D079 | `artifacts/time_dependent_no/pcno_deterministic_process_confirmation_d079_20260805a/d079_vs_d078_repeatability_audit.json` plus the exact D078/D079 H30 summaries | All 40 declared D079 scientific payloads are byte-identical to D078 and all scientific summary fields match exactly after removing output-path and elapsed-time leaves. The six deterministic prefix discrepancies remain zero. This establishes same-open-population process repeatability only. |
| D080 | D080-A summary above; `artifacts/time_dependent_no/pcno_local_correction_pilot_d080_20260805b/d080_dynamic_h30_r1/summary.json`; sibling `d080_dynamic_h30_r1_analysis/manifest.json` | D080-B H30-r1 is terminal `complete` with summary SHA-256 `c0f0d242...c18c`; all 39 declared source outputs and every H5 prefix/hash/closure gate verify. No arm promotes. Shock-isotropic/normal endpoint-residual-cumulative medians are `1.00039/0.99834/1.00114` and `1.00040/0.99835/1.00099`; target-shock medians are `1.00830/1.00583`. Vortex medians are `0.99898/1.00004/0.99926`, but its worst global control is `1.08594`. The 40-output derived analysis package contains 12 fixed-scale, unsaturated GIFs and four PDF/PNG figure pairs. Dynamic-only adaptive open-validation evidence; bump and sealed populations remain unopened. |

| D081 | `artifacts/time_dependent_no/d081_runs_20260806b/d081_dynamic_h30_r1/summary.json` | H30-r1 is terminal `complete`; all 41 declared outputs rehash exactly and every parent, prefix, inventory, closure, support, and cap gate passes. Calls 1--20 outperform always-on, calls 11--30, and dose-matched controls on the registered paired timing gates, but improve the persistent endpoint in only 3/6 cases. Timing is supported; absolute efficacy and promotion fail. Dynamic adaptive open-validation only. |

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
the vanishing-mass and shrinking-support behavior of a one-cell tag band. The
first three-arm ladder is now complete and not promoted. Dynamic FV retains
`model_all_nodes`; bump retains `causal_nodal_physical`. All training and
frozen-checkpoint evidence uses only the open populations, with strength-OOD
and test populations still sealed.

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

### D072 closeout evidence (2026-08-09)

The bound artifact root is
`artifacts/time_dependent_no/d072_boundary_fields_20260803a`. It contains all
nine dynamic and all nine bump training summaries plus one dedicated
dynamic-FV FP32 frozen-field evaluation on seed 20260718. The source-set and
provenance digests are respectively
`5c31b97d840397b000e7bf60ffed8a0d2b748d28816f97a575a522019f78a194`
and
`6012dd888560dbb9b8c6eebfcfa676e6f750b77313b2bf5b0d718a986bef2687`.
Dynamic data/normalizer digests are
`6d190423b7f367c16c3080bda2ef5a27840463c4a6421c23864f870c941f9339`
and
`9df7efb2f1aadf315f399dcabf3b366bf1b2f46794b026cbc5b7d80ba22e1a1a`;
bump data/normalizer digests are
`6b7176d5f7f926af2fe05dc7a2e62a76292501461e47ac721782de4c8431ab4a`
and
`d2d07a4000acc3cfd45ff105a19e7437177d553cee4c18d50858f5572de7efec`.

The dynamic training analysis, dynamic FP32 evaluator, dynamic evaluation
analysis, and corrected bump training audit have SHA-256 values
`e5999a35b8efbc4ab6745a4b183d9586b3554f528ab2c330f4c1c1fb981d20e9`,
`8769859f0c5465f1083f99dd436388ad50d5f1f51d2c8c2cd5d733210bca346a`,
`08d8df0a022e362b65b282df90bc72e15c192536979847f78f8631c50febd0bd`,
and
`d75ed7e545318298059e7f7e2a07f265d9fe5324e723997ec91facf9471c926c`.
The corrected bump audit supersedes the generic analyzer's bump H79 column:
the frozen bump trainer's tuple index 3 is mixed-prefix final error, not H79.

Independent retrieval verification found zero source-snapshot hash mismatches
in either family. The bump package also verifies all 18 checkpoint hashes and
all 330 declared arrays (30 keys; 1,000,607,108 bytes). The compact dynamic
retrieval intentionally omits its 18 training checkpoint payloads: their hashes
were checked by the bound training analysis on the remote source at analysis
time, and the dedicated evaluator rechecked the exact seed-20260718 checkpoints
before inference, but the other two seeds cannot be rerun from the compact
local package alone. Seven dynamic evaluation core inputs and the training-
analysis binding verify exactly.

Matched BF16 training gives:

| Family and arm | Primary open metric | Comparison |
| --- | ---: | --- |
| Dynamic N0 | H30 `0.0076021 +/- 0.0008601` | reference; 24/24 complete in every seed |
| Dynamic G1 | H30 `0.0079510 +/- 0.0003622` | G1/N0 `1.04590`; 1/3 seed wins |
| Dynamic S1 | H30 `0.0075220 +/- 0.0003433` | S1/N0 `0.98947`; 2/3 seed wins |
| Bump N0 | H79 completion `0.95556`; survival `0.99058` | reference |
| Bump G1 | H79 completion `0.94444`; survival `0.97539` | common-case H79 G1/N0 `1.10787`; 1/3 seed wins |
| Bump S1 | H79 completion `0.86667`; survival `0.95935` | common-case H79 S1/N0 `1.12396`; 1/3 seed wins |

Dynamic uses 84 train and 24 open-validation trajectories, 50 epochs, 51,200
presentations, 13,400 optimizer steps, raw recurrence, stride 2, and H30. Bump
uses 270 train and 30 open-validation trajectories, 40 epochs, 853,200
presentations, 216,000 optimizer steps, non-raw recurrence, stride 1, and H79.
Within each seed, G1/S1 copy every active and non-lifting N0 parameter, zero the
inserted columns, and restore the matched CPU RNG. Dynamic S1/G1 is
`0.94605`, narrowly passing the provisional scalar `0.95` contrast, but S1
lowers the N0 arm mean by only 1.05%, misses the registered `0.90` gate, and
does not establish a promoted N0-relative gain. Selected checkpoints occur only
among the final three rollout evaluations. On bump, neither field arm improves
N0 completion or common-case H79; raw arm means are not directly comparable
because completion populations differ.

The dedicated FP32 dynamic evaluation uses all 24 open cases for 30 raw calls:

| Frozen input intervention | Mean H30 error | Comparison |
| --- | ---: | ---: |
| N0 correct | `0.00900287` | 1 |
| G1 correct | `0.00825326` | G1/N0 `0.91674`; 13/24 case wins |
| G1 zero all | `0.03376748` | `4.09141` |
| S1 correct | `0.00858169` | S1/N0 `0.95322`; 12/24 case wins |
| S1 zero all | `0.03386694` | `3.94642` |
| S1 zero y / zero x | `0.03185012 / 0.01162399` | `3.71140 / 1.35451` |

Thus the continuous fields are causally used by these two trained checkpoints
under frozen intervention. The effect is not uniformly better one-step
prediction: mean teacher-forced error is `0.00056306/0.00066704/0.00065902`
for N0/G1/S1. FP32 also reverses the seed-20260718 BF16 endpoint rankings, so
the stored analysis leaf `combined_gate_passed=true` must not be read as one
coherent promotion gate: it combines a three-seed BF16 scalar contrast with
one-seed FP32 structural controls.

Neither field arm clears the dynamic anti-smear/no-harm interpretation against
N0. G1/N0 shock-strength and smooth-highpass ratios are `1.4563/1.1564`;
S1/N0 ratios are `1.0541/1.0853`. S1 improves all five registered front and
highpass controls relative to G1, yet its same-FP32 H30 endpoint is 3.98% worse
than G1 and its vortex-region H30 error is about twice N0. Zero-field damage is
largest near the boundary but propagates into shock, smooth, vortex, and
interior physical bands. Teacher-forced call-1 activation responses are largest
in early differential branches, while call-30 free responses are already mixed;
these are propagation diagnostics, not causal branch shares.

The dynamic visual package has 25 verified manifests and 62 verified outputs,
including 12 MP4s with all 31 frames and reference-only, rollout-wide physical
scales. Its summary-manifest SHA-256 is
`8e9fa2f7837a6a57719328425da33babc0119135d5ab7899fe790e6a578270fe`.
No checkpoint-bound bump intervention, region/frequency decomposition, or
animation package exists, and no D072 common-source resolution, native-graph
geometry-transform, or rotation experiment ran.

Result-to-claim decision:

- **Supported, bounded:** exact matched initialization and descriptor
  construction; on one dynamic FP32 seed, frozen field interventions prove that
  G1/S1 use the continuous boundary fields and that their effects propagate
  beyond the collar under the unchanged physical boundary policy.
- **Not supported:** G1 or S1 improves N0 across either benchmark; S1 is an
  optimal semantic encoding; bump checkpoints causally use the fields; the
  representation improves boundary conditions, resolution consistency,
  geometry or rotation generalization, conservation, or PCNO operator
  convergence.
- **Closure:** the first D072 ladder is complete and not promoted. Re-entry for
  a stronger training claim requires same-precision multi-seed dynamic
  checkpoint evaluation and a checkpoint-bound bump intervention/structure/
  visualization package. Resolution or rotation claims require separately
  preregistered common-source/native-graph stress tests; they are not loose ends
  of this closeout.

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
full-H30 checkpoint inference has run under this source identity.

Two exact-command dynamic H2 contract smokes, `d074_dynamic_h2_smoke_r1` and
`d074_dynamic_h2_smoke_r2`, did run under that original source identity on the
personal GPU workstation. Both return `smoke_complete`, retain
`scientific_interpretation_allowed=false`, select `rank8_gain1`, preserve the
same candidate eligibility vector, pass the smoke-form promotion arithmetic,
and close all registered output inventories. On the one smoke evaluation case,
the first run has endpoint-state ratio `0.84744168`, residual-RMS ratio
`0.91676662`, and worst primary-control ratio `1.00469914`; these values are
contract-path observations only, not H30 evidence. The exact repeat is stable
but not bitwise identical under workstation CUDA: maximum drift is
`2.06e-6` in selector-score scalars, `3.66e-5` in selector metric ratios,
`3.95e-6` in evaluation primary ratios, `2.42e-5` in evaluation controls,
`1.49e-7` in per-case coefficient entries, and `7.60e-8` in frozen coefficient
entries. Therefore artifact-hash equality is not a valid GPU repeatability
criterion.

Before any H2 run under the post-validator source, the repeatability gate is
frozen as follows. Replicates must have exact source/input bindings, candidate
key, candidate eligibility vector, phase order, completion/admissibility and
row inventories, and contract booleans. Every finite selector score or selector
metric ratio and every finite evaluation primary/control ratio must agree with
absolute drift at most `1e-4`. Calibration and frozen coefficient arrays must
agree under `allclose(rtol=1e-3, atol=5e-7)`. Recurrence, projection,
signed-growth, and reconstruction closures must independently pass their
evaluator tolerances. This gate was selected above the observed original-source
CUDA drift and is frozen before the new-source repeats; it is not permission to
interpret H2 scientifically.

The post-validator immutable source manifest has SHA-256
`422109500770a0e341c4cc237c89a2492852fcf8b236f6cf90a642c43c704b07`,
base HEAD `b2193946dd20a350053520e5406b98bfee3c4ae3`, and 93 exact Python
files. The evaluator now regenerates the canonical `250x100` dynamic geometry
and requires 25,000 nodes, 99,300 directed edges, identity mesh-to-graph
ordering, exact FP32 model-sample tensors, and fixed digests for nodes, edges,
physical measures, normalized weights, node radii, and family-local node types.
The three post-validator H2 cases all pass those checks. The current source has
22 focused evaluator tests and eight renderer tests; the final combined D074
and maintained-dependency command passes 128 tests. The workstation lacks
`pytest`, so its exact copied bytes were instead compiled and exercised through
the production geometry digest gate before inference.

Post-validator runs `d074_dynamic_h2_smoke_r3` and
`d074_dynamic_h2_smoke_r4` both verify all 24 output hashes and visual-payload
inventories, preserve exact categorical phase/completion contracts, select
`rank8_gain1`, and pass the frozen repeatability gate. Maximum observed drift is
`7.97e-6` in selector-score scalars, `2.16e-5` in selector metric ratios,
`8.43e-6` in evaluation primary ratios, `6.53e-5` in evaluation controls,
`1.22e-7` in per-case coefficients, and `6.03e-8` in frozen coefficients.
The first post-validator evaluation ratios are `0.84744333` endpoint state,
`0.91676714` residual RMS, and `1.00469756` worst control. Original-to-new
source drift also stays below `1e-4`, so the validation code did not materially
change the smoke outcome. Summary SHA-256 values are
`f446cbeebc6b3997b88400ab754ffc16f1c4d38b5db13a4ff41ad772bbcad12b`
and
`354322958f84f0911f84a032837d0eb0d3bf8c5a2058e280fe5c78c288b91dfe`.

The deterministic machine-readable comparison is
`artifacts/time_dependent_no/pcno_native_residual_correction_d074_20260804b/repeatability_r3_vs_r4.json`
with SHA-256
`1a80d6c6b4f5187b1226262a33e93c4a142e2ebf0c94e8e33e41b4a51f9530a1`;
it recomputes every frozen drift, coefficient all-close gate, categorical
binding, and both 24-output within-run hash audits and records `PASS`.

The maintained renderer is
`scripts/time_dependent_no/visualize_pcno_native_residual_correction.py`.
It verifies every input hash and replay identity before drawing, aggregates
case-first, uses population-exact fixed scales by physical field group and
component, forbids clipping and per-frame normalization, and watermarks H2 as
non-scientific. Its original-source H2 bundle contains five PNG/PDF figure
pairs, four two-frame component GIFs plus final PNGs, and a saturation audit;
manifest SHA-256 is
`9c2f2219ef5cc0863c3e6936fee7fa82bca89e06f16c02f718d2033e7282c781`.
Because two calls force trivial lag/POD structure, that H2 panel is explicitly
N/A; temporal coherence, lag correlation, and POD become interpretable only at
H30.

Independent pre-launch review freezes four postprocessing-only supplements for
the retained H30 payloads: a spatial recurrence-sensitivity view comparing the
baseline-own defect with the base defect on the selected trajectory and their
difference/cumulative errors; centered and uncentered full POD spectra; explicit
per-payload time, frame, unit, and weight metadata; and case-traceable curves.
Focused tests must also cover case-first aggregation, component summation, the
H30 temporal/POD branch, two-payload output counts, and the same-input
correction identity. These additions do not change or delay the frozen
checkpoint rollout.

These results close the exact dynamic geometry and selector-repeatability
prerequisites. Dynamic H30 is GO under the same open population, immutable
source/input identities, FP32/no-AMP recurrence, and physical boundary policy.
This authorizes execution, not a correction claim. Bump remains NO-GO until the
D041 replay cases, limits, and artifact hashes are frozen exactly.

The first full-run identity is `d074_dynamic_h30_r1`. Before its outcomes are
available, its two visual-payload cases are frozen as `sv_e00_y00` and
`sv_e11_y08`, spanning the low-energy/no-offset and high-energy/offset corners
of the declared evaluation set. Their inclusion is for all-call structure
inspection, not case selection or promotion weighting.

On 2026-08-04, the open-only H30 preflight passed before launch. It verifies
exactly 24 validation directories and no train/test directories, all 312 shard
array hashes, all 73 reference-bundle payload hashes, all 48 D063-ledger
reference bindings, all 24 multires solver summaries, the six top-level
artifact/source bindings, and all 93 active source files. The retained record is
`artifacts/time_dependent_no/pcno_native_residual_correction_d074_20260804b/d074_dynamic_h30_r1_preflight.json`
with SHA-256
`7c766b5476ffbf18dd5fcb05a75e560bf7225fc389d80d5dba4c9602c240b3dd`.
The exact no-smoke process then launched and showed sustained GPU work at the
initial health check. At 03:28:46 +08:00 the kernel OOM-killed PID 971524:
anonymous RSS was 28,422,396 KiB and system swap had only 244 KiB free. The
dedicated log remained empty and no output directory or selector artifact was
created, so this attempt supplies no scientific correction evidence. The
machine-readable incident record is
`artifacts/time_dependent_no/pcno_native_residual_correction_d074_20260804b/d074_dynamic_h30_r1_failure.json`
with SHA-256
`b35ea99140e318499ae84cfd98cc3b8e3a7c4719ea943bd91cc1042374ae7e2e`.

The cause is deterministic evaluator retention: a complete H30
`RolloutResult` holds 120,800,000 bytes across state, base-defect, correction,
defect, and truth-increment float64 arrays. Retaining 18 cases by 17 candidates
therefore approaches 34.43 GiB before cases, the model, and ordinary runtime
memory, although `select_candidate` consumes only completion,
`final_state_error`, `residual_rms`, and controls. The minimum repair drops
each full calibration rollout immediately after copying exactly those summary
fields; evaluation payloads and every scientific formula remain unchanged. Its
24 focused tests and full 569-test time-dependent suite pass. The new immutable
source manifest has SHA-256
`892745de57ebe1d7b81074353974eb0b7f3e3c11c05365ec6ed317dbac686e8e`,
binds the same 93-file inventory and base `b219394`, and differs from
`42210950...4b07` only in the evaluator, whose new SHA-256 is
`215a20340bd62f80e700b4c46e1a09ab0cd438c4ef2215c1ed979e9efc45c1f2`.
Its verification record has SHA-256
`3770e381d45b9faa14544cd7a6af932b90bcbccfda6417dd9e095bc96c836bce`.

The two post-repair H2 identities are `d074_dynamic_h2_smoke_r5` and
`d074_dynamic_h2_smoke_r6`. They reuse the exact r3/r4 checkpoint, normalizer,
split, three loaded cases, geometry, candidate inventory, FP32/no-AMP
recurrence, boundary policy, and visualization case; only source-manifest and
output identities change. Both pass. They select `rank8_gain1`, retain exact
categorical and contract fields, and verify all 48 within-run output hashes.
Their endpoint-state ratios are `0.8474389503/0.8474385622`, residual-RMS
ratios are `0.9167682930/0.9167618299`, and maximum primary-control ratios are
`1.0046743813/1.0046947925`. The four-run r3/r4/r5/r6 audit is
`artifacts/time_dependent_no/pcno_native_residual_correction_d074_20260804c/equivalence_r3_r4_vs_r5_r6.json`
with SHA-256
`b1dda0672039c08cfec53a6c23419ed44757848d1d2ce9a8df145dc34fcf0bcb`.
Its maximum scalar drift is `7.87e-5` against limit `1e-4`; maximum coefficient
drift is `2.32e-7` under `allclose(rtol=1e-3, atol=5e-7)`. The source delta is
exactly the evaluator retention repair. This closes evaluator equivalence only:
H2 remains contract-only and scientifically non-interpretable.

The replacement full-run identity is `d074_dynamic_h30_r2`; r1 remains
permanently failed. Its new preflight must
record exact argv, interpreter, checkout, source manifest, checkpoint, shard,
family, multires, output and log paths, prelaunch absence, launch PID/time, and
the r1 preflight/failure hashes. Initial calibration health additionally
requires bounded external RSS samples rather than GPU utilization alone.

That r2 preflight passes with SHA-256
`de1903038ec3d0c5f4baaff252320cf9394ca69b0fd47f2357e6887eeffbfa1d`
and launches PID 1030473 at 11:17:45 +08:00 under the exact recorded argv.
The bounded early-launch health artifact is
`artifacts/time_dependent_no/pcno_native_residual_correction_d074_20260804c/d074_dynamic_h30_r2_launch_health.json`
with SHA-256
`aa3f91d8468786c76c1bd995b3deec7669046d31193216035cdef1559de55c4c`.
RSS is 2.88 GiB at both 106 and 176 seconds, versus r1's 27.11 GiB anonymous
RSS at kill; 26.27 GiB host memory and 6.77 GiB swap remain free at the final
bounded sample. The evaluator is therefore left unattended. The two equal RSS
samples span only 70 seconds and end at 176 seconds, whereas r1 failed after
about 17 minutes; they are launch-health evidence only.

The run subsequently completes in 448.664 s. The registered summary is
`artifacts/time_dependent_no/pcno_native_residual_correction_d074_20260804c/d074_dynamic_h30_r2/summary.json`
with SHA-256
`bb7b1c38097c92c6866b3f8853ceeab28668fe0bf38e4fa469cc6033c36a57ab`.
All 23 summary-declared output hashes, the exact recurrence/growth closures, and
the no-target selector contract pass. Across 18 calibration cases and 16
nonzero candidates, all 288 endpoint ratios are below one. The eligibility-first
selector nevertheless retains `zero`, and the frozen promotion decision is
false because every nonzero arm fails at least one complete-case no-harm gate.
The closest complete-case arm is `rank8_gain0p25`: median endpoint-state ratio
`0.953384`, median residual-RMS ratio `0.991771`, and 17/18 eligible cases. Its
only failed case-metric row is
`physical_volume_integral_state_scale_rms__energy` on `sv_e10_y00`:
`0.00450793 / 0.00410474 = 1.098226`, above the frozen `1.05` threshold. The
denominator is far above the registered `1e-8` floor, so this is a small but
real global energy-integral harm, not division noise. Higher ranks and gains
improve median endpoint and residual ratios further but introduce larger
case-specific energy and later `rho_v` integral errors. The six held-out
evaluation cases run only the selected zero arm and reproduce baseline exactly;
they supply no nonzero-arm efficacy evidence.

The raw baseline sequence analysis preserves the defect-structure diagnosis.
Across the six evaluation cases, median aggregate relative residual energy is
`0.172753`, median net defect relative to cumulative truth change is `0.126450`,
and median temporal coherence is `0.218555`. Lag correlation is positive at lag
one (`0.3721` median), near zero at lag two, and negative from lag three onward.
All 180 squared-error increments remain positive, so cancellation reduces the
path-sum accumulation rather than reversing rollout-error growth. Centered POD
needs 14--18 modes for 95% energy, showing that the full defect is not globally
low-rank even though a persistent correctable component exists.

The fixed-scale renderer manifest is
`artifacts/time_dependent_no/pcno_native_residual_correction_d074_20260804c/d074_dynamic_h30_r2_visualization/manifest.json`
with SHA-256
`e8171b25433031ca94ea864b1e726b45a03ea260a9cc068f4ba5ef2cf393fb72`.
All 27 declared figure/animation hashes pass; all 64 scale-saturation rows are
zero, and the selected-zero correction panels explicitly report N/A rather than
an efficacy curve. This supports a useful low-rank direction plus broad
temporally cancelling structure, but not a safe promoted correction. D074-B
therefore uses raw transfer-native rollout as its primary practical comparator.

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
`ZERO` plus one nonzero candidate, and H1/H2 coverage. The full-run
prerequisites were focused CPU tests of the selector, target-loading order,
projection closures, metric inventories, and zero/baseline identity. Those
prerequisites are now fulfilled by the post-validator source, focused tests,
and H2 repeats recorded above, so the first frozen dynamic full run is
authorized. This does not authorize bump.

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

## Registered D075 Conservative-Integral-Neutral Correction

D075 is the smallest new native-resolution correction experiment justified by
D074-A. It does not rewrite D074, relax its `1.00/1.05` calibration gates, or
claim that the base PCNO is conservative. It asks whether the persistent rank-8
direction can retain its endpoint benefit after the correction itself is
constrained not to inject selected physical-volume integrals. The dynamic-FV
checkpoint, normalizer, split, 18 calibration cases, six evaluation cases,
native `250x100` geometry, H30 clock, FP32/no AMP, raw recurrence,
`model_all_nodes`, physical boundary handling, truth-loading order, case-first
aggregation, denominator floor, metrics, and promotion thresholds are exactly
those of D074-A. Strength-OOD and test remain sealed. Reuse of the same open
population makes this adaptive mechanism/development evidence, not independent
confirmation.

Let `S={i:type_i=0}` be the dynamic interior support and let `V_i` be the
audited physical cell volume. For the raw rank-8 fitted bias `B_n`, define

`mu_c(B_n) = sum_(i in S) V_i B_(n,i,c) / sum_(i in S) V_i`.

Three target-free policies are frozen:

- `raw`: `P_raw B=B`;
- `energy_integral_neutral`: subtract `mu_energy(B)` from the energy
  component on `S` only; and
- `all_integrals_neutral`: subtract `mu_c(B)` for all four conservative
  components on `S` only.

Every policy remains exactly zero on non-type-0 contacts. The applied
correction is `C_n=-alpha P B_n`, formed in denormalized conservative-update
units and float64 after cosine reconstruction. For every neutralized component,
the full-domain residual-scale-normalized physical-volume mean of `C_n` must
be at most `1e-12` at every call. The operator must be idempotent within
`1e-12`; energy-only neutralization must leave the other three components
bitwise unchanged; and the raw policy must be exactly identical to the D074
correction construction. These are correction-budget identities only. Open
boundaries and later nonlinear recurrence mean they neither impose physical
conservation on PCNO nor guarantee zero future state-integral error.

The unique full candidate inventory is one canonical `ZERO` plus rank 8 at
`alpha in {0.125,0.25,0.5,1.0}` under each of the three policies: 13 arms
total. The matched raw arms distinguish a smaller-gain explanation from an
integral-projection explanation. Rank is not reswept because D074 identified
rank 8 as the closest complete-case arm and the question is policy isolation,
not renewed capacity search. Gain `0.25` is the exact D074 near-miss anchor;
`0.125` tests a smaller raw intervention; `0.5` and `1.0` test whether
neutralization can safely retain stronger endpoint correction. Policy
complexity for an otherwise exact score tie is `raw < energy < all`, after
the existing lower-rank/lower-gain ordering.

Calibration repeats D074's leave-one-case-out coefficient fit and complete-case
selector. A projected candidate is ineligible if any required integral closure
is missing, nonfinite, or exceeds `1e-12`. All existing endpoint, residual,
shock, vortex, smooth-high-pass, boundary, component, and global-quantity
eligibility limits remain unchanged. The selector record binds correction
policy explicitly rather than inferring it from the candidate key. After the
selector and coefficient bundle are frozen and rehashed, evaluation targets
may load. Evaluation uses the unchanged D074 promotion conjunction: complete
admissible H30 rollouts, median endpoint ratio at most `0.92`, every endpoint
and residual ratio at most `1.00`, and every required control ratio at most
`1.05`.

Interpretation is predeclared:

- a selected/promoted neutralized arm supports a bounded target-free
  conservative-budget correction on this one checkpoint and reused open
  population;
- a selected/promoted raw arm shows that reduced gain, not projection, was
  sufficient;
- a safe selected arm that misses only the evaluation `0.92` threshold is a
  safety/mechanism success but not a promoted rollout method;
- selecting zero means none of the registered policies is safe; and
- larger endpoint gains accompanied by failed integral or structure controls
  remain negative deployment evidence.

The non-scientific H2 gate contains `ZERO` and one gain-`0.5` rank-8 arm
from each policy so that every projection branch executes. It requires exact
candidate/case/metric inventories, raw-path identity, all projection closures,
zero/non-type-0 support, recurrence and signed-growth closure, immutable model
state, exact dynamic geometry and provenance, and two repeat runs under one
immutable source before H30. The full run uses summary-only calibration
retention. Stop before interpretation on any source, checkpoint, normalizer,
split, common-source reference, geometry, target-loading, selector, closure,
inventory, or hash mismatch.

Execution status on 2026-08-04: the 93-file immutable source manifest is
`7b1760db...ce7b` and its scoped verification record is
`fc13a1f5...e293`. The maintained focused and full local suites pass
`40/40` and `581/581`; the exact frozen source passes the same `40/40`
focused gate. Its broad copied-checkout run passes 573 tests and leaves only
four unrelated CPG history tests unable to resolve the remote checkout's
nonportable alternate Git object store; the unchanged D074 parent snapshot has
its independent 569-test full pass. On the actual GPU runtime, H2 r1 and r2
both complete, remain non-scientific, select `rank8_gain0p5_raw`, and rehash
all 25 declared outputs. Their maximum scalar drift is `4.4336e-5` against
`1e-4`; coefficient drifts are at most `1.0514e-7` and all-close under
`atol=5e-7, rtol=1e-3`. H30 `d075_dynamic_h30_r1` then launched from
preflight `f31484ca...ab67` and completed in `413.929` seconds. The retrieved
summary is `dc011715...c5ceb`; all 26 declared output hashes pass locally and
remotely, and the retrieval archive is `27aa7dd4...f183`. The eligibility-first
selector chooses `rank8_gain0p125_raw`. `ZERO`, raw gain `0.125`, and energy-
neutral gain `0.125` are the only complete-case eligible calibration
candidates.

On the six evaluation cases, the selected arm improves every endpoint and
residual RMS ratio. The endpoint median/maximum are `0.974925/0.982984`; the
residual median/maximum are `0.995263/0.998266`. Net accumulated-defect ratios
are `0.97012--0.98115`, while instantaneous residual ratios are
`0.99048--0.99827`, and temporal-coherence ratios are `0.97650--0.98538`.
This is a small but consistent reduction of the coherent accumulating channel,
not removal of the local defect field. Promotion nevertheless fails: the
endpoint median is above the registered `0.92` target and only four of six
cases pass control no-harm. `sv_e11_y00` and `sv_e11_y08` have final absolute
energy-integral ratios `1.51847` and `1.69402`; the former also has energy-
integral RMS `1.05932`.

Matched-policy evidence rejects simple integral projection as the standalone
remedy. At gain `0.125`, energy-neutral and raw are nearly tied; energy
neutralization changes the median endpoint/residual ratios by only
`+3.45e-5/-4.78e-5` and worsens the worst control by `0.00579`. All-integral
neutralization is already ineligible at gain `0.125`, worsens the median
endpoint from `0.97517` to `0.98643`, and raises the worst control from
`1.03478` to `1.08345`. At gain `0.5`, raw and energy-neutral arms reach the
endpoint target (`0.91826/0.91842`), and energy neutralization materially
reduces the worst control from `3.07718` to `1.20361`, but neither is safe.
The selected raw correction's summed constant-mode energy fractions are
`0.491/0.057/0.850/0.046` for `rho/rho_u/rho_v/energy`; blanket projection
therefore removes much of the useful density and transverse-momentum channel.

The frozen correction is exactly array-identical across both same-grid visual
cases, yet their energy-control response differs sharply. This isolates
state-dependent recurrence as the route from one case-independent input to
different global-energy outcomes. A supplemental movie shows the smooth
instantaneous correction and its direct cumulative sum on separate rollout-
fixed structure scales; the registered eight-panel movies remain the absolute-
magnitude reference. This direct sum is not recurrent causal attribution.
The next correction design should therefore use state-aware gain or short-
window response constraints and keep shock-aware local dissipation as a
separate capped channel; data assimilation remains reserved.

One stronger cross-run audit does not establish bitwise D074/D075 H30 replay.
The nominal raw gain-`0.25` selector rows differ by at most `5.07e-4` in one
sensitive energy control, while baseline numerators differ relatively by at
most `5.68e-5` and rank-8 fitted coefficients by `3.20e-7`. The within-source
raw identity and focused tests pass, but cross-source H30 values must be
treated as tolerance-level replay rather than exact equality.

Required outputs add correction-policy and per-call physical-integral audit
rows to the D074 tables. Figures retain the candidate no-harm matrix,
time-resolved residual/cumulative/coherence curves, signed growth, and
centered/uncentered POD. They additionally compare matched-gain raw versus
neutralized global budgets and the norm of the removed constant field.
Representative all-call animations retain one rollout-wide physical scale and
show truth increment, base defect at the corrected input, applied correction,
corrected defect, cumulative error, and signed local growth density. No
per-frame normalization or model-outcome-dependent scale selection is allowed.
The registered visualization manifest is `4accda31...e31be`: all 27 hashes
pass and all 64 saturation rows have zero fraction outside their fixed limits.
Compact policy, gate, and component-structure figures are bound by
`448958ee...e5ca`. The one case-independent structure-only animation is bound
separately by `56c89a03...fe6a`. Shock-aware filtering and data assimilation
remain outside D075.

## Registered D076 Short-Window State-Response Gain Pilot

D076 was predeclared as the smallest follow-up to D075 that tests recurrent
response rather than the instantaneous integral of a case-independent
correction. Its registered dynamic-FV calibration run is now complete and
fails the exact safety qualification. It remains adaptive open-validation
development evidence, not an independent confirmation.

### Frozen scope and provenance

- Family: dynamic finite volume only, native `250x100` geometry, 30 raw
  recurrent calls, FP32 inference, no AMP.
- Checkpoint, normalizer, split, retained shards, common-source references,
  regenerated geometry, rank-8 physical cosine basis, type-0 support, source
  manifest, and case provenance must bind exactly to the accepted D075 parent.
- Physical boundary policy remains `model_all_nodes` with raw recurrence. This
  is boundary representation under the existing solver policy, not a boundary-
  condition intervention.
- Dynamic node meanings remain `0 interior`, `1 y-symmetry contact`,
  `2 x-extrapolation contact`, and `3 contact with both`. Bump meanings are not
  loaded or reused.
- Calibration remains the 18 D075 open-validation cases
  `e01--e05,e07--e10` at `y00/y08`. Conditional evaluation remains the six
  already inspected cases `e00,e06,e11` at `y00/y08`. Strength-OOD/test cases
  stay sealed. Evaluation targets may load only after the response policy and
  every calibration artifact hash are frozen.
- Candidate gains are exactly `g in {0, 0.125, 0.25, 0.5, 1.0}`. Positive arms
  use the raw rank-8 persistent correction only. D075's integral projections
  and D071's local dissipation are not mixed into this experiment.

For calibration fold `k`, let `B_{-k,n}` be the call-indexed rank-8 defect bias
fit from every other calibration case. The positive-gain rollout is

    Uhat_g^{n+1} = N(Uhat_g^n) - g B_{-k,n},

while the shadow baseline is

    Uhat_0^{n+1} = N(Uhat_0^n).

The correction remains zero on non-type-0 nodes. The exact same-input identity
and recurrence closure from D074/D075 remain mandatory at floating-point
tolerance.

### Target-free five-call probe

The probe horizon is fixed at `p=5` learned calls (`t=0.10`). For the initial
state, on type-0 physical-volume support, define

    m_v = sum_i V_i (rho v)_i / sum_i V_i,

    a_0 = sqrt(sum_i V_i ((rho v)_i-m_v)^2 / sum_i V_i) / s_{rho v},

    y_0 = sum_i V_i y_i ((rho v)_i-m_v)^2
          / sum_i V_i ((rho v)_i-m_v)^2.

The denominator of `y_0` must exceed a frozen floating-point floor. These two
features are computed from the supplied initial condition, not from a target.
For each positive gain, define the induced probe response

    d_g^p = Uhat_g^p - Uhat_0^p,

    r_g = ||d_g^p||_{V,s_state},

    j_{g,c} = sum_i V_i d_{g,i,c}^p / s_{state,c},

    q_g = ||d_g^p||_{V,s_state}
          / (||sum_{n<p} (-g B_{-k,n})||_{V,s_state} + epsilon),

    chi_g = <d_g^p, sum_{n<p}(-gB_{-k,n})>_{V,s_state}
            / ((||d_g^p||_{V,s_state}+epsilon)
               (||sum_{n<p}(-gB_{-k,n})||_{V,s_state}+epsilon)).

Here `||X||_{V,s}` is the physical-volume weighted RMS of `X/s`, averaged over
the four conservative components, and `j_{g,c}` deliberately uses the same
signed physical-volume-integral/state-scale normalization as the existing
dynamic no-harm control. The response feature vector is exactly
`(a_0, y_0, r_g, j_g,rho, j_g,rho_u, j_g,rho_v, j_g,E, q_g, chi_g)`.
No truth state, true increment, residual error, future target, or per-frame
normalization enters the runtime feature.

### Calibration labels and deterministic controller

A gain/case calibration row is recurrently safe only if its H30 rollout is
complete and admissible, its endpoint-state and aggregate-residual-RMS ratios
against that case's zero arm are at most `1.00`, and every frozen
structure/component/global-control ratio is at most `1.05`. Denominators at or
below the existing numerical floor stop the experiment.

For each gain separately, features are standardized using calibration-only
means and population standard deviations. A zero-variance feature is retained
as zero after a fixed `1e-12` scale floor. The response policy uses `k=3`
Euclidean nearest neighbors from the same gain. A positive gain is eligible
only when all three neighbors are recurrently safe. Its predicted endpoint
ratio is their median endpoint ratio; the policy chooses the eligible gain with
the smallest prediction, then the smaller gain, and otherwise chooses zero.

Because D075 already exposed abrupt high-strength recurrence failure that was
not predicted by ordinary calibration trends, positive gains fail closed when
`a_0` exceeds the maximum calibration support for the active fold. Lower-
amplitude extrapolation remains allowed only in this adaptive open-population
pilot. This asymmetry is part of the predeclared D076 rule and cannot be changed
after evaluation targets load.

Leave-one-case-out calibration qualifies the controller only if all 18 selected
arms are recurrently safe, at least 12 select a positive gain, the median
endpoint ratio is at most `0.95`, and the median residual ratio is at most
`1.00`. Failure freezes zero and stops before evaluation-target loading.

### Conditional evaluation and promotion

If calibration qualifies, the response table, standardization constants,
neighbor rule, frozen rank-8 coefficients, source manifest, and their hashes
are written and revalidated before evaluation targets load. Each evaluation
case then runs target-free H5 probes for all four positive gains, freezes one
case-specific gain, and only afterward evaluates H30. The selected H30 replay
must reproduce its chosen H5 probe prefix within the existing family-specific
repeatability tolerance. A contemporaneous zero arm and static raw gain-`0.125`
arm are mandatory comparators.

Adaptive promotion requires all six selected rollouts to complete and remain
admissible, every endpoint and residual ratio to be at most `1.00`, every
structure/component/global control ratio to be at most `1.05`, at least four
cases to select a positive gain, an all-case median endpoint ratio at most
`0.95`, and selected/static-`0.125` median endpoint and residual ratios at most
`0.98` and `1.00`, respectively. These gates do not establish independent
confirmation, conservation, resolution transfer, or other-PDE generality.

Required outputs include per-case/per-gain probe features, neighbor identities
and distances, calibration safety labels, leave-one-out selections, the frozen
response table, target-loading phase order, evaluation gain choices, H5 replay
checks, complete H30 ratios, and all prior closure/admissibility inventories.
Case-level metrics are aggregated only after per-case computation; nodewise
values are never pooled across cases or meshes.

Figures must show gain choice and safety by case, probe-feature neighborhoods,
time-resolved endpoint/residual/cumulative/coherence metrics, signed component
budgets, and controller versus zero/static controls. Representative animations
use one rollout-wide physical scale and show true increment, zero-arm defect,
static defect, selected defect, selected-minus-zero response, cumulative
selected defect, selected-minus-zero state, and signed local error-growth
density. Per-frame or outcome-dependent scale selection is forbidden. Bump
requires a separate family-local contract after its immutable D041 replay
binding exists; shock-aware local filtering and data assimilation remain
outside D076.

The first H2 launch, `d076_dynamic_h2_smoke_r1`, stopped before model rollout
because the evaluator had not populated the inherited dynamic loader's
`resolutions` field. It is a preserved parser-contract failure with no
scientific result. The repaired immutable source passes `11/11` focused tests,
`51/51` combined evaluator/visualization regressions, and `11/11` focused tests
when imported only from the isolated snapshot outside the repository. Ruff
check and format checks pass. The manifest still binds the accepted 93-file
D075 runtime plus only `evaluate_pcno_response_gain_controller.py`, for 94
files total. The repaired manifest, bundle, verification, and launcher SHA-256
values are `c0494165...1a75`, `6d6317c1...a5d5`, `7fcfeab1...ae50`, and
`0691d990...f05c`.

H2 r2 is `smoke_complete`, passes every contract/hash/inventory check, remains
scientifically non-interpretable, and is bound by summary SHA-256
`6f103295...5287`. The exact H30 result is `calibration_not_qualified`; its
summary SHA-256 is `baaa4c16...53e7`. All 18 selected gains are positive and
the median endpoint/residual ratios against each case's zero arm are
`0.93272065/0.98621848`, passing the aggregate efficacy thresholds. Sixteen
selections are recurrently safe. `sv_e10_y00` selects gain `0.25` and fails
energy-integral RMS at `0.00450777/0.00410449 = 1.09825`; `sv_e10_y08`
selects gain `1.0` and fails final absolute energy integral at
`0.00504187/0.00251795 = 2.00237`. Both denominators are resolved, so these are
late global-energy harms rather than restriction-floor or division noise.
Reference restriction crosschecks remain `1.77636e-15`.

The five-call features forecast endpoint ratios accurately but do not reveal
the abrupt highest-strength energy transition. From `e09` to `e10`, response
RMS changes only `0.03896%` for `y00` gain `0.25` and `0.03143%` for `y08`
gain `1.0`, while the worst H30 control grows by factors `1.08413` and
`1.97035`. Across the four positive gains, the neighbor classifier makes two
false-positive safety decisions out of 50 predicted-eligible decisions; the
endpoint-oriented selector chooses both. This supports a useful state- and
position-dependent gain signal, but the H5 endpoint response is insufficient
as a late energy-safety certificate.

The hash-bound analysis summary is `0e797316...96bba`. Its registered selector
figure is `7e907d20...b1dd` and the fixed-axis high-strength transition figure
is `ee57e4a1...0814`. A clearly post-hoc replay that groups the paired
`y00/y08` cases by physical strength changes only the two `e10` choices to
zero and yields 18/18 safe, 16 nonzero, median endpoint/residual
`0.949135/0.995869`, and worst control `1.02620`. It passes the old numeric
calibration gates but is not D076 qualification, evaluation authorization, or
confirmation. The registered stop therefore stands: evaluation targets,
rollout metrics, visual payloads, and animations were not generated. Sealed
populations and bump were not accessed, and `model_all_nodes` remains the
unchanged physical boundary policy.

## Registered D077 Strength-Grouped Response Controller

D077 is the smallest prospective follow-up to D076. It tests whether D076's
two highest-strength false-positive selections arose from leakage between the
paired `y00/y08` cases at one physical vortex strength. The post-hoc D076
selector replay is design evidence only: D077 changes both the correction fit
and the controller fold, so its outcome is not implied by that replay.

### Frozen scope, groups, and intervention

- Family, checkpoint, normalizer, split, retained shards, common-source
  references, regenerated native `250x100` geometry, raw 30-call FP32/no-AMP
  recurrence, type-0 correction support, rank-8 physical cosine basis, and
  source/provenance bindings remain exactly D076/D075. The physical boundary
  policy remains `model_all_nodes`; this is not boundary-condition improvement.
- Dynamic meanings remain `0 interior`, `1 y-symmetry contact`,
  `2 x-extrapolation contact`, and `3 contact with both`. Bump is not loaded.
- Calibration is the same 18 open cases `e01--e05,e07--e10` at `y00/y08`.
  They form exactly nine provenance-defined physical-strength groups
  `sv_e01`, ..., `sv_e05`, `sv_e07`, ..., `sv_e10`; each group contains exactly
  its `y00` and `y08` cases. The case manifest and target-free initial-vortex
  amplitude must agree with this grouping within floating-point tolerance.
- For held-out group `G`, the rank-8 bias is fit only from the other 16 cases:

      B_{-G,n} = mean_{k not in G} B_{k,n}.

  Every case in `G` is rolled out with this same group-excluded bias. Feature
  standardization, maximum-amplitude support, safety labels, and `k=3`
  response neighbors also exclude every case in `G`. No coefficient, feature,
  label, normalizer statistic, support bound, or neighbor from the query group
  may enter its selection.
- Gains remain exactly `g in {0, 0.125, 0.25, 0.5, 1.0}` with raw rank-8
  correction. The H5 feature vector, denominators, safety labels, deterministic
  nearest-neighbor tie breaks, static `g=0.125` comparator, H30 metrics, and all
  closure/admissibility tolerances remain exactly D076.
- The sole new target-free abstention mechanism is physical-strength grouped
  upper-support rejection: a positive gain is ineligible when the query's
  initial-vortex amplitude exceeds the maximum among the remaining groups.
  Consequently both held-out `sv_e10` cases must select zero because `sv_e09`
  is the largest permissible training group. Lower-strength extrapolation
  remains the already declared adaptive-pilot behavior. No target-derived
  late-energy threshold, gain relaxation, or post-hoc distance cutoff is added.

### Gates, phase order, and outputs

Calibration qualification is unchanged: all 18 selected arms must be safe, at
least 12 must be nonzero, median endpoint ratio must be at most `0.95`, and
median residual ratio at most `1.00`. Failure freezes the selector and stops
before any of the six conditional evaluation targets `e00,e06,e11` at
`y00/y08` load. A stopped run must report evaluation checks and closure maxima
as not run/null, never as vacuous true/zero. Passing calibration freezes and
hashes the full 18-case response table, full-calibration rank-8 coefficients,
group inventory, fold training inventories, neighbor rows, selector, source,
and phase order before evaluation-target loading.

Conditional evaluation and promotion use every D076 rule unchanged: target-
free H5 choice followed by exact-prefix H30 replay; zero and static `g=0.125`
comparators; all-case endpoint/residual no-harm; `1.05` control no-harm; at
least four nonzero selections; median endpoint at most `0.95`; and selected-
versus-static median endpoint/residual at most `0.98/1.00`. Case metrics are
aggregated only after per-case computation; nodewise values are never pooled
across cases or meshes. Passing remains adaptive open-validation development
evidence, not confirmation, conservation, resolution transfer, or other-PDE
generality. Sealed populations and data assimilation remain outside D077.

The non-scientific H2 smoke uses the first three complete strength groups
(`e01--e03`, six cases), one-call probes, and only zero/`0.125`/`0.5` arms, so
each group-excluded policy still has at least three neighbors. It may use the
existing explicit smoke-only target-loading bypass after the selector bundle
freezes; its metrics cannot qualify the method. A scientific H30 launch is
authorized only after focused CPU tests, isolated-source import tests, source
manifest verification, and this H2 contract pass.

Required new audit fields are the group ID and both excluded case IDs for each
query; coefficient-training case/group IDs; policy-training IDs; maximum
training amplitude; and neighbor case/group IDs. Tests must prove exact
two-case group construction, exclusion from both fitting paths, high-strength
fail-closed behavior, no neighbor leakage, smoke inventory sufficiency,
target-loading phase order, null reporting when evaluation does not run, and
unchanged D076 defaults. If calibration passes and evaluation runs, the D076
fixed-scale figures and animations remain required without per-frame or
outcome-dependent normalization.

The isolated runtime derives from the accepted 94-file D076 archive and
contains 95 files after overlaying only the reusable D076 experiment seam and
the D077 entry point. Source-manifest, verification, bundle, and launcher
SHA-256 values are `a9cc0795...f2bb`, `3bc8d254...fdd9`,
`9f2a96ad...10ea`, and `be4cb7c7...2aa8`. Both the working tree and isolated
snapshot pass the 57-test native-correction/visualization/D076/D077 suite.
All 18 immutable D076 leave-one-case-out choices replay exactly through the
default seam.

H2 `d077_dynamic_h2_smoke_r1` is `smoke_complete`, contract-clean, and
scientifically non-interpretable; its summary SHA-256 is
`b5f26464...3cc2`. It uses six cases in three complete strength groups. Every
declared output hash passes, all six coefficient/policy fold inventories and
all 36 neighbor rows are leakage-free, and held-out `sv_e03_y00/y08` both
select zero under upper support. The selector freezes before the one explicit
smoke-only evaluation target loads. Sealed arrays are not accessed.

H30 `d077_dynamic_h30_r1` is terminal `failed_contract`; its summary SHA-256 is
`1406ff87...81b7`. All 40 declared outputs are local and SHA-256 exact,
including the three `250x100` visual payloads. The source, checkpoint,
normalizer, split, common-source references, physical geometry, group/fold
inventories, target-loading phase order, evaluation-choice freeze, closures,
and absence of sealed access all pass. Calibration qualifies with `18/18`
safe selections, 16 nonzero selections, endpoint/residual medians
`0.94867470/0.99590446`, and required zero abstention on both held-out `e10`
cases.

Conditional evaluation selects gains `0.25/1.0/0.5/1.0/0/0` for
`e00_y00/e00_y08/e06_y00/e06_y08/e11_y00/e11_y08`. Considered only as
diagnostic numbers, every promotion inequality passes: selected/zero endpoint
and residual medians are `0.93440472/0.99481783`, worst registered control is
`1.03809984`, four cases are nonzero, and selected/static medians are
`0.96014872/0.99738543`. In the four nonzero cases, median endpoint,
residual, and final cumulative-defect ratios are `0.91115`, `0.97552`, and
`0.92044`. The correction reduces rank-8 parallel defect energy to
`0.394--0.742` of the same-input baseline while orthogonal energy remains
`1.000` in every case. This directly supports the intended structure: the
smooth persistent subspace controls accumulated error, while most local
orthogonal residual energy remains untreated.

The frozen H5-prefix/H30 identity gate fails in all six cases. Maximum absolute
differences are `3.743e-5--5.507e-5` against `2e-5`; relative L2 differences
are `4.690e-7--5.340e-7` against `1e-7`. Both zero-correction `e11` cases fail
at the same scale, while recurrence, same-input, projection, component-energy,
and visual replays close between `2.44e-17` and `1.55e-14`. Therefore D077
does not scientifically interpret the evaluation, does not promote the
controller, and does not relax or retroactively change its gate. The exact
immutable PCNO path uses CUDA `scatter_add_` in the gradient and fixed-hop
averaging and did not enable deterministic algorithms. A new prospective
deterministic-runtime replay is required to separate GPU reduction
repeatability from an evaluator/state-contract bug.

## Registered D078 Deterministic CUDA Replay Attribution

D078 is the smallest prospective follow-up to D077. It does not rerun a looser
gate or retroactively rescue D077. It tests whether the all-case prefix replay
failure is removed when the same mathematical PCNO is evaluated with PyTorch's
deterministic algorithms enabled.

- Scope, checkpoint, normalizer, split, retained shards, common-source
  references, native `250x100` geometry, raw H30 FP32/no-AMP recurrence,
  type-0 rank-8 correction support, physical-strength grouping, H5 features,
  gains, controller neighbors, upper-support abstention, evaluation cases,
  static/zero comparators, and every D077 efficacy, safety, closure, and prefix
  tolerance remain unchanged. Dynamic node meanings remain family-local;
  `model_all_nodes` remains frozen. Bump, sealed populations, boundary-condition
  changes, local filtering, and data assimilation are not loaded.
- The sole intervention is runtime determinism, configured before model
  construction and any target load with
  `torch.use_deterministic_algorithms(True)`, cuDNN benchmarking disabled, and
  `CUBLAS_WORKSPACE_CONFIG=:4096:8` set before CUDA initialization. The summary
  must record and contract-check all three settings. D076 and D077 defaults
  remain nondeterministic-runtime bytes and behavior.
- D078 recomputes grouped calibration and every target-free evaluation choice
  from scratch. It may not import D077's evaluation choices or target outcomes.
  Selector freeze and target-loading phase order remain exact.
- Independent H5 and H30 executions must still satisfy maximum absolute
  `2e-5` and relative L2 `1e-7` prefix identity in all six cases. No tolerance
  is changed. All 40-style output inventories, within-run hashes, algebraic
  closures, and case-first aggregation remain mandatory.
- H2 is non-scientific and must precede H30 under a new immutable source
  manifest. H30 interpretation requires both the complete D077 scientific
  conjunction and the new deterministic-runtime contract. Passing D078 would
  be a new deterministic-runtime adaptive open-validation result, not a D077
  pass, independent confirmation, conservation, resolution transfer, or
  other-PDE evidence.

Decision rule: if D078 passes all prefix replays while retaining exact source
and data bindings, the evidence supports normally nondeterministic CUDA
execution as the cause of D077's contract failure; the differential
`scatter_add_` path is then the leading code-supported source, but unique
branch attribution still requires a same-hidden branch repeat if needed. If
D078 still fails, stop scientific interpretation and audit the probe/H30 state,
normalization, recurrence, and evaluator identities before any further
correction experiment.

### Execution status (2026-08-05)

Attempt A used source-manifest SHA-256 `ab4f7876...a3de` and stopped before any
data load, checkpoint inference, or CUDA work because the new direct script
entry point did not bootstrap the repository namespace. It produced no summary
and has no scientific interpretation. The failed source/launcher/log provenance
is preserved under
`artifacts/time_dependent_no/pcno_deterministic_response_controller_d078_20260805a/`.
The minimum repair copies D077's repository-root bootstrap into the D078 entry
point and adds a direct-execution regression test; the focused controller suite
then passes 8 tests.

Corrected immutable source B has manifest, verification, bundle, and launcher
SHA-256 values `ae02af9e...e748`, `7bf95b62...5425`,
`9e511385...c1e8`, and `114f1472...8337`. H2
`d078_dynamic_h2_smoke_r2` is `smoke_complete`, contract-clean, and explicitly
non-scientific; its summary SHA-256 is `0fef77a5...633`. All 38 declared outputs
verify, the deterministic-runtime settings are recorded and pass, and maximum
independent prefix absolute/relative discrepancies are both exactly zero. H30
`d078_dynamic_h30_r1` was launched only after binding that exact H2 summary.
It is terminal `complete` and scientifically interpretable within the frozen
adaptive open-validation scope. Summary SHA-256 is
`959d33989e5c03a4f581af848ec746aa25aa26de64c82db97624c5d90138e86e`;
all 40 declared hashes pass locally, including three visual payloads. Source,
checkpoint, normalizer, split, common-source references, geometry, phase order,
group/fold exclusions, choice freeze, deterministic-runtime settings, closures,
and absence of sealed access all pass. Every one of the six independent prefix
rows has exactly zero absolute and relative discrepancy.

Calibration qualifies with `18/18` safe choices, 16 nonzero choices, endpoint
and residual medians `0.94867699/0.99590733`, and required `e10` abstention.
Evaluation choices are `0.25/1.0/0.5/1.0/0/0`. The unchanged promotion
conjunction passes with selected/zero endpoint and residual medians
`0.93441454/0.99483269`, worst control `1.03811492`, four nonzero cases, and
selected/static medians `0.96015362/0.99741356`. D078 therefore qualifies this
controller as adaptive dynamic-FV open-validation evidence; it does not
retroactively pass D077 or establish independent confirmation.

The mechanism result is stable. On the four nonzero cases, median endpoint,
residual, and final cumulative-defect ratios are
`0.91115853/0.97551396/0.92045299`. Aggregate rank-8-parallel energy ratios are
`0.74213/0.39376/0.63277/0.46959`, while orthogonal energy is unchanged to
floating-point precision. Median shock and boundary endpoint ratios improve to
`0.93033/0.88985`, but two cases worsen vortex error to `1.01778/1.03811` and
smooth high-pass error to `1.00773/1.02031`. The controller removes persistent
low-rank drift; it does not solve the local-noise channel.

D077-to-D078 calibration and evaluation gain choices are categorically exact.
Scientific ratio changes are at most `6.10e-5`, and frozen rank-8 coefficients
move by relative L2 `1.13e-7`. Five near-tied neighbor sets change membership,
but no neighbor safety label or final choice changes; the smallest evaluation
winner margin is `0.00782`, versus maximum predicted-ratio drift `2.28e-5`.
Combined with the zero prefixes, this supports ordinarily nondeterministic CUDA
execution as the cause of D077's contract failure. The immutable code makes
differential/fixed-hop `scatter_add_` the leading source, but deterministic mode
covers other CUDA operations too, so unique branch attribution is not claimed.
Deterministic H30 takes `2365.73 s`, `7.92x` D077's runtime.

The static scientific figure bundle is
`artifacts/time_dependent_no/pcno_deterministic_response_controller_d078_20260805b/d078_dynamic_h30_r1_analysis/manifest.json`
with SHA-256 `8f0741ae...97eb`; all 12 outputs pass, nodewise errors are never
pooled across cases, and no failed-contract watermark is present. The matching
three-case, four-component response animation bundle is
`d078_dynamic_h30_r1_response_animations_stride2/manifest.json` under the same
root with SHA-256 `c5ef2f04...b6b2`; all 27 declared outputs pass, all 96 scale
rows have zero saturation, and no per-frame normalization, clipping, or
watermark is used.

## D079 Deterministic Process Confirmation

D079 is the minimum independent-process confirmation required before adding a
local shock/vortex correction. It tests deterministic process repeatability,
not a new population, seed, checkpoint, method, or physical contract.

- The exact D078 corrected source B, evaluator, checkpoint, normalizer, split,
  retained open-validation shards, common-source references, native `250x100`
  geometry, raw H30 FP32/no-AMP recurrence, type-0 rank-8 support,
  physical-strength groups and exclusions, H5 features, gains, `k=3` response
  controller, upper-support abstention, six evaluation cases, static/zero
  comparators, formulas, denominators, case-first aggregation, tolerances, and
  every efficacy, safety, closure, admissibility, and prefix gate are frozen.
  The boundary policy remains `model_all_nodes`; dynamic node meanings remain
  `0` interior, `1` y-symmetry contact, `2` x-extrapolation contact, and `3`
  contact with both. Bump, sealed populations, local filtering, boundary-
  condition changes, and data assimilation are outside D079.
- The runtime contract remains exactly
  `torch.use_deterministic_algorithms(True)`, cuDNN benchmarking disabled,
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `PYTHONHASHSEED=0`, and one visible GPU.
  D079 must use a fresh detached process, a separately verified checkout, and a
  previously absent output/log/preflight path. It binds D078 H2-r2 summary
  SHA-256 `0fef77a54ad086dc10157160120202da8f798e1b4b5e9c39e3635740507a3633`
  and D078 H30-r1 summary SHA-256
  `959d33989e5c03a4f581af848ec746aa25aa26de64c82db97624c5d90138e86e`.
- D079 recomputes calibration, correction bases, response features, safety
  labels, neighbors, gains, and every evaluation rollout from scratch. It may
  not import D078 choices, coefficients, targets, or outcomes. D078 is used
  only after D079 completes as the frozen repeatability comparator.
- The primary repeatability gate is exact SHA-256 equality between D079 and
  D078 for all 40 files in D078's declared `output_hashes` inventory. The two
  run-level `summary.json` files are not required to be byte-identical because
  they include output-path and elapsed-time metadata; their scientific fields,
  categorical choices, contract results, and declared output inventories must
  nevertheless agree exactly. D079 must also pass all 40 within-run hashes and
  retain exactly zero absolute and relative discrepancy in every one of the six
  independent H5-prefix/H30 rows.
- Any missing or unequal declared payload, nonzero prefix row, provenance or
  runtime mismatch, scientific-field disagreement, incomplete rollout,
  inadmissibility, closure failure, sealed access, or changed choice makes D079
  `failed_contract` or inconclusive. Do not relax a tolerance or substitute a
  rounded comparison. Instead perform a bounded per-file numeric/categorical
  audit before further correction work.

Decision rule: only an exact D079 pass confirms that D078 is reproducible across
fresh deterministic processes on this same open-validation population and
authorizes registration of a separate zero-inclusive local shock/vortex
correction experiment. It does not establish statistical robustness,
independent-data confirmation, bump transfer, conservation, resolution
invariance, boundary-condition improvement, other-PDE transfer, or sealed
performance.

### Execution result (2026-08-05)

Run `d079_dynamic_h30_process_confirmation_r1` is terminal `complete`. Launcher
and launch-preflight SHA-256 values are `ae214b58...d769` and
`3bc09dfe...703f`; summary SHA-256 is `f11c5882...6aa7`. The preflight verifies
all 96 source files, 24 retained validation cases, 312 staged shard arrays, 73
reference payloads, both exact D078 summary bindings, and no sealed access.

The retrieved directory contains exactly the 40 declared payloads plus
`summary.json`; all 40 D079 within-run hashes pass. D078 and D079 declare the
same 40 paths and the same SHA-256 value for every path, including the three
large visual payloads. A recursive summary comparison finds exactly two
different leaves: `args.output_dir` and `elapsed_seconds`. After removing those
run-metadata leaves, the summaries are exactly equal. D079 takes `2363.8545 s`
versus D078's `2365.7278 s`, a ratio of `0.999208`.

All six H5-prefix/H30 rows again have exactly zero maximum absolute and relative
L2 discrepancy. Calibration remains `18/18` safe with 16 nonzero choices and
endpoint/residual medians `0.94867699/0.99590733`. Evaluation gains remain
`0.25/1.0/0.5/1.0/0/0`; promotion remains true at endpoint/residual medians
`0.93441454/0.99483269`, maximum control `1.03811492`, four nonzero cases, and
selected/static medians `0.96015362/0.99741356`.

On the four nonzero cases, endpoint, residual, final cumulative-defect, and
temporal-coherence ratio medians remain
`0.91115853/0.97551396/0.92045299/0.95309911`. Aggregate parallel-energy ratios
are `0.74213/0.39376/0.63277/0.46959`; orthogonal energy is unchanged. Median
boundary/shock ratios are `0.88985/0.93033`. The same two local-control
worsenings remain: vortex ratios `1.01778/1.03811` and smooth-high-pass ratios
`1.00773/1.02031`. D079 therefore confirms the persistent-low-rank correction
and untreated local-channel result across fresh deterministic processes; it
adds no new population or robustness axis.

The exact audit is
`artifacts/time_dependent_no/pcno_deterministic_process_confirmation_d079_20260805a/d079_vs_d078_repeatability_audit.json`
with SHA-256 `12a82ffb...2b9f`; its private audit program has SHA-256
`10c49717...b044` and passes Ruff. D079 satisfies the registered decision rule
and authorizes registration of a separate zero-inclusive local shock/vortex
correction experiment. It does not authorize a filter chosen from evaluation
targets, change the frozen boundary policy, open sealed data, or remove the
independent-data/bump evidence requirements.

## D080 Causal Local-Channel Correctability And Bounded Rollout Pilot

D080 separates cheap directional evidence from an actual recurrent method test.
D080-A is a frozen-trajectory audit; D080-B, if routed, is a separate
zero-inclusive deterministic rollout pilot. The owner explicitly permits a
bounded signal-seeking GPU run when D080-A is mixed. The launch rule below is
therefore intentionally weaker than a promotion gate, but it is fixed before
reading D080-A outcomes.

### D080-A registered contract

- Inputs are only the six retained D067 bundles `sv_e00_y00`, `sv_e00_y08`,
  `sv_e06_y00`, `sv_e06_y08`, `sv_e11_y00`, and `sv_e11_y08` under
  `pcno_cumulative_pathway_structure_d067_20260802a/results_d067_unified_r1d_20260803a`.
  The summary and all six bundle SHA-256 values, metadata checkpoint SHA-256
  `95e6c180...79c9`, normalizer digest `9df7efb2...e1a1`, physical times,
  residual scales, native nodes, and volumes must verify before metrics are
  interpreted. Only the upper `250x100_to_500x200` fields are used. The exact
  structured-FV geometry must reconstruct those native nodes and volumes.
- The baseline remains the frozen D060 checkpoint, raw H30 recurrence,
  `dt=0.02`, FP32/no AMP, and `model_all_nodes`. Dynamic node codes remain
  family-local: `0` interior, `1` y-symmetry contact, `2` x-extrapolation
  contact, and `3` contact with both. Corrections use type-0-to-type-0 edges
  only; contact-node updates are untouched. This is boundary representation
  under a frozen physical boundary policy, not boundary-condition improvement.
  Bump, sealed populations, reference generation, training, checkpoint
  changes, and data assimilation are outside D080-A.
- For physical cell volumes `V_i` and the fixed D067 residual component scale
  `s_c`, define

      <a,b>_{V,s} = sum_i V_i sum_c (a_ic/s_c)(b_ic/s_c) / sum_i V_i,
      ||a||_{V,s} = sqrt(<a,a>_{V,s}).

  At call `n`, let `p_n` be stored `coarse_increment_free`, `r_n` stored
  `true_increment`, and `d_n=p_n-r_n`. A candidate correction `c_n` is added to
  the predicted increment, so its scored defect is `d_n+c_n`. Reference states
  and defects may score a direction but may not construct it.
- The target-free local directions are: the D071 global-isotropic negative
  control; shock-envelope isotropic; shock-normal; shock-tangential;
  shock-tangential with positive temporal coherence; vortex-isotropic; and
  vortex-isotropic with positive temporal coherence. Shock/vortex masks and
  shock normals are inferred causally from the stored baseline free state at
  the current call. For edge unit vector `l_ij` and the normalized mean shock
  normal `n_ij`, normal and tangential conductances are
  `(l_ij dot n_ij)^2` and `1-(l_ij dot n_ij)^2`. The coherence gate is the
  positive part of the nodewise cosine between graph-high-pass, residual-scaled
  predicted increments at calls `n` and `n-1`; it is zero at the first call.
- Every direction uses the existing quantile-0.8 graph sensor and symmetric
  equal-and-opposite volume-weighted edge flux. Node gates enter through the
  geometric mean of their endpoint values. Registered relative-norm caps are
  `0.005`, `0.01`, `0.02`, and `0.05`, measured against the type-0
  residual-scaled RMS of `p_n`. Each correction must be finite, touch no
  non-type-0 node, satisfy its cap within `1e-12`, and close every component's
  volume-weighted mean within `1e-12` in residual-scaled absolute units.
- The rank-eight physical-cosine decomposition uses exact weighted QR on type-0
  nodes over `[0,2] x [0,1]`; non-type-0 nodes remain a separate partition.
  Per-call metrics include

      q_res(n) = ||d_n+c_n||_{V,s} / ||r_n||_{V,s},
      rho(n)   = ||d_n+c_n||_{V,s} / ||d_n||_{V,s},
      cos(n)   = <-d_n,c_n>_{V,s} / (||d_n||_{V,s} ||c_n||_{V,s}),
      g(n)     = (||d_n+c_n||^2_{V,s}-||d_n||^2_{V,s}) / ||d_n||^2_{V,s}.

  The same quantities are reported for rank-eight parallel, orthogonal, and
  excluded-contact partitions and for fixed boundary, shock, vortex, and smooth
  regions. Reference-derived region masks are diagnostic scoring masks only.
- Per-case trajectory metrics include the residual-scale aggregate and the
  cumulative defect measures

      Q_res = sqrt(sum_n ||d_n+c_n||^2 / sum_n ||r_n||^2),
      R_res = sqrt(sum_n ||d_n+c_n||^2 / sum_n ||d_n||^2),
      Q_cum = ||sum_n(d_n+c_n)|| / ||sum_n r_n||,
      R_cum = ||sum_n(d_n+c_n)|| / ||sum_n d_n||,
      kappa = ||sum_n(d_n+c_n)|| / sum_n ||d_n+c_n||.

  Raw numerators and denominators are retained. These are frozen-trajectory
  directional-headroom quantities, not a corrected recurrent rollout. Cases
  are reduced to scalars before median, minimum, maximum, or count aggregation;
  nodewise values are never pooled across cases.
- The registered D080-A output root is
  `artifacts/time_dependent_no/pcno_local_correctability_d080_20260805a/d080a_d067_native_sixcase_r1`.
  It must contain a hash-declared `summary.json`, per-call and per-case CSVs,
  and no undeclared scientific payload. Any input-hash, metadata, shape, time,
  geometry, decomposition, cap, support, mean-closure, finiteness, or output-hash
  failure stops scientific interpretation.

### D080-A execution result (2026-08-05)

Run `d080a_d067_native_sixcase_r1` is terminal `complete` after `124.332 s`.
Summary SHA-256 is `8570ac47...767d`; the declared call/case CSV hashes are
`9210d5f7...dac4` and `1951c3e6...b11`. The inventory is exactly those two
CSVs plus `summary.json`, with 5,220 per-call rows and 174 case-first rows.
All six input bundle hashes, metadata, 30-call increment replays, native nodes,
physical volumes, time grid, and rank-eight partitions verify. Maximum scaled
weighted-mean closure is `1.064e-18`, cap excess is `6.939e-18`, and contact-node
correction is exactly zero. The audit source and correction-utility SHA-256
values are `8e63b90b...8129` and `f1b11542...2061`.

The uncorrected aggregate residual-scale error `Q_res` has case median
`0.172783` and range `0.162268--0.200741`, confirming that the scored numerator
is materially large relative to the true residual rather than only to the
state. The main case-first frozen-direction results are:

| Direction | Median `R_res` | Median `R_cum` | Median shock ratio | Cases improving `R_res/R_cum` |
| --- | ---: | ---: | ---: | ---: |
| shock isotropic, cap `0.01` | `0.997680` | `0.997443` | `0.985642` | `6/6`, `4/6` |
| shock normal, cap `0.01` | `0.998311` | `0.997411` | `0.988338` | `6/6`, `4/6` |
| shock tangential, cap `0.01` | `0.997669` | `1.000756` | `0.988692` | `6/6`, `0/6` |
| vortex isotropic, cap `0.005` | `0.999956` | `1.000124` | n/a | `4/6`, `2/6` |
| global isotropic control, cap `0.02` | `0.991734` | `0.996779` | `0.987012` | `6/6`, `4/6` |

The shock-local corrections change the rank-eight aggregate energy ratio by at
most `2.31e-5`, so they are empirically complementary to D078's persistent
low-rank correction in this audit. The normal/tangential split is decisive:
both directions reduce instantaneous residual energy, but every tangential
case has `R_cum>1`, whereas the normal direction supplies the median cumulative
gain. Positive-update-coherence gating does not improve this result and often
worsens it; coherence of the predicted update is not a reliable proxy for
coherence of the unknown defect. The small vortex arm improves its local ratio
in four cases but worsens both `e00` cases, so it is a heterogeneous exploratory
signal, not a robust correction.

The stronger frozen global-isotropic control does not supersede D071: D071
already showed that a generic cap-`0.05` recurrent filter can harm vortex and
smooth controls. D080-A instead authorizes a new bounded rollout test of the
smaller, causally localized directions. None of these numbers is a corrected
rollout, independent-data confirmation, conservation result, or bump evidence.

### D080-A to D080-B routing

A bounded D080-B GPU pilot is authorized if every D080-A contract check passes
and at least one nonzero path/cap has any one of these predeclared signals after
case-first aggregation: median `R_res<1`; median `R_cum<1`; median shock- or
vortex-region residual ratio below `0.98` with global median `R_res<=1.10`; or
positive case-aggregated correction-to-negative-defect cosine in at least two
of six cases with global median `R_res<=1.20`. This is a launch rule, not an
efficacy claim. Mixed cases, a missed median target, or failure to beat D078 in
the frozen audit do not by themselves veto the signal-seeking pilot.

D080-B may carry at most three nonzero local choices selected by the registered
metrics, plus exact zero, the frozen D078 persistent controller, and their
predeclared combinations. It must use D078/D079 deterministic CUDA, exact
checkpoint/normalizer/data/recurrence provenance, a one-case H2 smoke, and then
the same six open H30 evaluation cases with raw zero prefixes. Selection remains
adaptive open-validation evidence. Promotion requires separately registered
rollout efficacy, no-harm, conservation-proxy, admissibility, closure, and
repeatability gates; the permissive launch rule cannot be cited as promotion.
If no routing signal is present, D080 stops after A and reports the failed local
directions rather than inventing an unregistered filter.

### D080-B selected rollout contract

D080-A passes the routing rule. Before any D080-B result is read, the three
selected local choices are frozen as shock-isotropic cap `0.01`, shock-normal
cap `0.01`, and vortex-isotropic cap `0.005`. This keeps the best shock-local
aggregate direction, the direction carrying the cumulative signal, and the
weaker heterogeneous vortex signal. Shock-tangential and positive-coherence
arms stop at D080-A because their frozen cumulative behavior is adverse.

- D080-B binds D080-A summary SHA-256 `8570ac47...767d` and inherits the exact
  D078/D079 checkpoint, normalizer, split, 24 open-validation shards,
  common-source references, native `250x100` geometry, H30 raw recurrence,
  FP32/no AMP, deterministic CUDA, strength-grouped H5 response controller,
  gains, abstention, and D078 efficacy/control definitions. It recomputes the
  D078 calibration and target-free gain choices; no future evaluation metric
  constructs a local correction. Sealed populations, bump, training, reference
  generation, resolution transfer, boundary-condition changes, and data
  assimilation remain outside the run.
- The eight evaluation arms are exact zero; the recomputed D078 persistent
  controller; the three local-only choices; and persistent plus each of the
  three local choices. Each arm has its own recurrent state, causal masks, and
  model update. For a combined arm, the persistent and local corrections are
  both computed from the same unmodified base PCNO proposal and then added
  simultaneously; the local filter does not smooth the persistent correction.
- At each call the shock/vortex mask and shock normal are inferred from that
  arm's current predicted state. The local direction uses the residual-scaled
  base predicted increment, quantile-`0.8` sensor, type-0-to-type-0 symmetric
  volume-weighted flux, and the D080-A conductance formula. Dynamic contact
  nodes remain untouched. The local correction must be finite, respect its cap
  within `1e-12`, and close every residual-scaled physical-volume mean within
  `1e-12`. This is controlled proposal smoothing, not entropy-stable physical
  dissipation or a boundary-condition intervention.
- A fresh one-case H2 deterministic smoke must pass source/data/checkpoint,
  hook, admissibility, closure, local cap/support/mean, output-inventory, and
  visual-payload hashes before H30. The scientific run uses all 18 calibration
  and six evaluation cases. Every one of the eight H30 arms receives an
  independently recomputed prefix from the exact initial state; H5/full-H30
  replay keeps D078's unchanged `2e-5` absolute and `1e-7` relative limits.
- Report state-scale endpoint error, residual-scale aggregate error, cumulative
  defect, temporal coherence, all D078 controls, conservative-component signed
  budgets, recurrence and signed-growth closures, and separate persistent/local
  correction norms and physical-volume means. Cases are aggregated before any
  median or count. Three registered visual cases `sv_e00_y00`, `sv_e06_y08`,
  and `sv_e11_y08` retain reference states plus zero, persistent, and all three
  combined states, defects, base increments, persistent corrections, and local
  corrections for fixed-scale residual animations.
- Promotion is evaluated separately for each combined arm against the exact
  same-case persistent arm. A combined arm passes efficacy only if median
  endpoint, residual-RMS, and final cumulative-defect ratios are all strictly
  below `1`; at least four of six endpoint ratios are at most `1`; the worst
  endpoint ratio is at most `1.02`; the worst registered control ratio is at
  most `1.05`; and the median `endpoint_state__shock` ratio is at most `1` for
  a shock arm or median `endpoint_state__vortex` is at most `1` for the vortex
  arm. Promotion additionally requires every source/runtime/inventory,
  completion, admissibility, H5 replay, recurrence/growth, cap, support,
  volume-mean, and payload-hash gate. Arms missing promotion remain informative
  signal-seeking results; thresholds are not relaxed after launch.

The initially registered draft roots were
`artifacts/time_dependent_no/pcno_local_correctability_d080_20260805a/d080b_dynamic_h2_smoke_r1`
and sibling `d080b_dynamic_h30_r1`. The retained execution roots are instead
`artifacts/time_dependent_no/pcno_local_correction_pilot_d080_20260805b/d080_dynamic_h2_smoke_r2`
and sibling `d080_dynamic_h30_r1`: source revision A stopped after all eight
rollouts at a strict-writer error on an intentionally empty projection table,
before summary creation or scientific interpretation. Revision B changes only
that output boundary and omits empty inapplicable tables; the registered arms,
caps, recurrence, cases, metrics, and gates are unchanged. D080-B remains
adaptive development on the same open population even if it passes. It cannot
establish independent confirmation, statistical robustness, bump transfer,
resolution invariance, physical conservation, entropy stability, other-PDE
transfer, or sealed performance.

### D080-B source and H2 execution (2026-08-05)

- Source revision A manifest SHA-256 is `b942f6cd...f4218`; its H2-r1 evaluated
  the one registered case and all eight arms, then stopped while the inherited
  strict CSV writer attempted to materialize empty `projection_metrics.csv`.
  This is an evaluator-output contract failure, not method evidence. The failed
  source, preflight, partial outputs, and log are retained and are not promoted.
- Revision B adds a tested output-boundary rule that omits only empty,
  inapplicable tables. Its isolated 97-file source manifest, verification, and
  bundle SHA-256 values are `b7001032...b997`, `0c226d4a...9f41`, and
  `1ea76a8e...966`. The D080 entry-point SHA-256 is `25630841...a339`; the exact
  D078 parent manifest remains `ae02af9e...e748`. The full local parent and
  callback compatibility suite passes `67/67`.
- H2-r2 is terminal `smoke_complete`, non-scientific, and bound to D080-A
  summary `8570ac47...767d`. Summary SHA-256 is `def2edcc...3e9`; all 37 declared
  outputs independently hash-verify. The eight arm prefixes replay exactly
  (`0` absolute and relative discrepancy). Maximum recurrence, signed-growth,
  same-input energy, same-input pointwise, visual cumulative, local
  physical-volume-mean, cap-excess, and non-type-0 support closures are
  respectively `1.513e-14`, `1.691e-17`, `1.396e-18`, `6.641e-19`,
  `4.400e-16`, `9.779e-21`, `8.674e-19`, and exactly `0`.
- The one-case H2 efficacy values are routing diagnostics only. Against the
  persistent arm, combined shock-isotropic has endpoint/residual/cumulative
  ratios `0.9901/0.9766/0.9688`; combined shock-normal has
  `0.9894/0.9756/0.9655`. Combined vortex-isotropic has
  `0.9963/1.0007/0.9989` and a worst control ratio `1.2423`, reinforcing the
  preregistered need for six-case no-harm aggregation.
- The exact-H2-bound H30-r1 passed preflight and completed all 18 calibration
  and six evaluation cases. Its terminal result and interpretation follow.

### D080-B H30 execution and local-effect analysis (2026-08-06)

Run `d080_dynamic_h30_r1` is terminal `complete` after `3151.017 s`. Summary
SHA-256 is `c0f0d242...c18c`; all 39 declared outputs independently
hash-verify. The run binds revision-B source manifest `b7001032...b997`, the
exact H2 summary `def2edcc...3e9`, checkpoint `95e6c180...79c9`, normalization
mapping `9df7efb2...e1a1`, the registered open-validation population, raw H30
recurrence, and `model_all_nodes`. Sealed populations and bump are not opened.
All 48 arm/case H5 prefixes replay with exactly zero absolute and relative
discrepancy. Maximum recurrence, signed-growth, same-input-energy,
same-input-pointwise, visual-cumulative, cap-excess, and local-volume-mean
closures are respectively `1.542e-14`, `1.776e-15`, `4.441e-16`,
`1.388e-17`, `3.704e-15`, `5.204e-18`, and `5.008e-20`; non-type-0 local
correction is exactly zero.

No combined arm passes the registered promotion conjunction:

| Combined arm | Median endpoint | Median residual RMS | Median cumulative defect | Median target region | Endpoint wins | Worst control |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| shock isotropic, cap `0.01` | `1.000388` | `0.998343` | `1.001136` | shock `1.008300` | `3/6` | `1.035352` |
| shock normal, cap `0.01` | `1.000404` | `0.998349` | `1.000994` | shock `1.005825` | `2/6` | `1.029995` |
| vortex isotropic, cap `0.005` | `0.998982` | `1.000041` | `0.999260` | vortex `0.992995` | `4/6` | `1.085941` |

The shock arms are useful but mistimed directions, not merely ineffective
ones. Their local-only same-input defect-energy ratios are `0.99407/0.99391`
with median correction/base-defect cosines `-0.0837/-0.0885`. The combined
state-error median reaches its minimum at call 4 (`0.99188/0.99120`), remains
below one through calls 22/23, and then reverses. At H20 the case-first median
change in squared state error is `-0.254%/-0.390%`; the late interval changes
that comparison by `+0.102%/+0.128%` of the persistent H30 squared error,
leaving a small final regression. Both shock arms improve smooth high-pass
endpoint error (`0.99683/0.99670`, six/five wins) while worsening shock
endpoint error in all six cases. This is consistent with removal of some local
noise plus added front phase/profile bias.

The same separation appears in rank and time. Shock filtering increases median
temporal coherence by `0.64%/0.70%` and the uncentered first-three POD energy
fraction by `0.00577/0.00607` absolute, with no material change in modes needed
for 95% energy. Thus the filter slightly reduces cancellable orthogonal
structure while leaving the remaining defect more organized; it does not
isolate the persistent large-scale recurrent mode. The shock and vortex masks
cover about `4.7%` of native cells. Registered caps are reached on
`96.7--100%` of calls, so simply increasing strength is not supported by the
late reversal.

The vortex arm is conditional rather than robust. Its local-only same-input
energy ratio is `0.99954` with cosine `-0.0273`. All three `y08` target-vortex
ratios improve (`0.96848/0.98023/0.98444`), whereas all three `y00` ratios
worsen (`1.00155/1.00810/1.01266`). The worst case, `sv_e06_y08`, reaches a
final absolute transverse-momentum integral ratio `1.08594`. Instantaneous
local corrections close every physical-volume mean, but recurrent state
sensitivity can still create later global-integral drift; mean neutrality is
not a rollout-safety certificate.

The derived analysis package is
`d080_dynamic_h30_r1_analysis/manifest.json`, SHA-256
`537fa53b...992e`. Its 40 declared outputs independently verify: six
case-first/temporal/structure tables, four PDF/PNG figure pairs, 12 GIFs with
16 fixed-scale frames each, 12 final PNGs, and the analysis summary. All 288
scale-audit rows have zero values outside the exact common limits. Residual,
prediction, defect, correction, cumulative-defect, and signed-growth fields use
frozen component residual scales without per-frame normalization.

D080 therefore stops with no promoted local arm. The evidence rejects these
three fixed always-on recurrent implementations at their exact caps, not local
correction as a family. A next attempt should predeclare state/time-conditioned
activation or an early-window stop and must retain complete-case integral and
regional controls. Data assimilation remains logged but reserved. Bump remains
outside D080 until its immutable D041 replay population, checkpoint,
normalizer, recurrence, and case provenance can be bound exactly.

### D081 predeclaration: shock-normal timing versus dose (2026-08-06)

D081 asks whether D080's late reversal is caused by *when* the local correction
is applied rather than only by its total nominal strength. This is an adaptive
mechanism experiment on the same open validation population, not independent
confirmation. It changes no checkpoint, normalizer, reference, recurrence,
boundary policy, persistent controller, shock sensor, graph, or regional
definition.

The six frozen arms are: zero; the D078 persistent controller; the exact D080
always-on shock-normal correction at cap 0.01; that correction on calls 1--20;
the equal-duration correction on calls 11--30; and an always-on nominal-dose
control at cap 0.01 * 20 / 30 = 1/150. The fixed call-20 cutoff uses D080's
already reported H20 checkpoint and is not tuned further. The late-window and
dose controls distinguish timing from application count and nominal cap-dose.
Caps are relative residual norms, not state-scale normalizations.

For every case and arm, retain D080's state-scale endpoint error, residual-scale
per-call and RMS defect, cumulative-defect norm, shock/vortex/smooth and
fixed-boundary-band errors, component metrics, physical-volume integral
controls, admissibility, recurrence and signed-growth closures, correction
support/mean/cap audits, lag/coherence, and POD tables. Aggregate within each
case first and then take the six-case median; never pool nodes between cases.
All ratios use the case-matched persistent arm as denominator. In particular,

\[
q_E(a)=\operatorname{median}_i
\frac{\|\widehat U_{i,a}^{30}-U_i^{30}\|_{V,S_U}}
     {\|\widehat U_{i,p}^{30}-U_i^{30}\|_{V,S_U}},\qquad
q_C(a)=\operatorname{median}_i
\frac{\|\sum_{n<30}\delta_{i,a,n}\|_{V,S_r}}
     {\|\sum_{n<30}\delta_{i,p,n}\|_{V,S_r}}.
\]

The calls-1--20 arm promotes only if the unchanged D080 conjunction passes:
median endpoint, residual RMS, and cumulative-defect ratios below one; at least
four of six endpoint ratios at most one; maximum endpoint ratio at most 1.02;
maximum registered control ratio at most 1.05; and median shock-endpoint ratio
at most one. Timing is supported separately only if the early arm's paired
case-first endpoint and cumulative-defect medians are both below those of the
exact always-on, calls-11--30, and dose-matched arms. A failed timing gate does
not erase any arm that independently passes the efficacy gate.

The run must stop before scientific interpretation if the D080 parent summary,
rank-8 coefficient content, frozen response policy, evaluation gain choices,
or the zero/persistent/exact-always control rows fail their registered exact
digests; or if any prefix, recurrence, signed-growth, mean, cap, support,
admissibility, reference, manifest, or inventory check fails. Execute one H2
smoke first, then one H30 run only after exact H2 replay. Visual payloads freeze
the three D080 cases and all six arms; result-dependent or per-frame scaling is
forbidden. Bump, sealed populations, training, new reference generation,
boundary-condition changes, and data assimilation remain out of scope.

The first isolated-source H2 attempt, source A run r1, stopped before evaluation
rollout as designed. Its coefficient guard included the experiment-schema
string stored beside the scientific rank and coefficient arrays, so D081's new
wrapper label differed from D080 even though rank and coefficients were exactly
equal (zero maximum difference). It has no scientific result and is preserved
as an evaluator-contract stop. Revision B changes only this guard: the semantic
coefficient digest excludes the wrapper schema while continuing to hash key
names, dtypes, shapes, rank, and every coefficient byte. Policy and selector
guards already used the analogous schema-independent scientific digest.

Source-B H2-r2 is terminal smoke_complete, non-scientific, with summary
SHA-256 10ab335f...8726 and elapsed time 71.385 s. All 39 declared outputs
independently hash-verify and all 30 D080 parent, controller, choice, and
control-row audit records pass. All six arm prefixes are bit-exact. Maximum
recurrence, signed-growth, same-input energy, same-input pointwise, visual
cumulative, local-mean, cap-excess, and non-type-0 support closures are
1.513e-14, 7.806e-18, 1.518e-18, 6.641e-19, 4.400e-16, 6.490e-21,
8.674e-19, and exactly zero. The schedule inventory also closes: at H2 the
always, early, and dose arms are active twice and the late arm is inactive.
Exact-H2-bound source-B H30-r1 is terminal `complete` after `2883.73` s, with
summary SHA-256
`eb6e2826b4c9dd59ca2a8053decfae203d93e6dd16e91f5ca7ffc8772f6d553a`.
All 41 declared output hashes reverify, the 108-file loaded-source binding is
complete, and every D080 parent/control-row, selector, inventory, reference,
hook-equivalence, prefix, recurrence, signed-growth, support, mean, cap, and
visual-payload gate passes. Maximum recurrence closure is `1.542e-14`, maximum
growth closure is `1.887e-15`, maximum local-cap excess is `3.469e-18`, and
non-type-0 local correction is exactly zero.

The registered timing mechanism passes. Relative to exact always-on,
calls 11--30, and the nominal-dose-matched arm, the calls-1--20 endpoint medians
are `0.999461/0.999242/0.999634` and cumulative-defect medians are
`0.998469/0.997527/0.998981`; all six paired endpoints are nonworse in every
comparison. Residual medians are `0.995719/0.986063/0.995571`, and shock-endpoint
medians are `0.991634/0.987851/0.994101`. This distinguishes placement in time
from duration and nominal cap-dose under the exact registered controls.

The practical efficacy conjunction nevertheless fails. Against the persistent
D078 controller, calls 1--20 have median endpoint/residual/cumulative/shock
ratios `0.999971/0.993392/0.999677/0.997485`, maximum endpoint `1.000221`, and
maximum control `1.031082`, but only `3/6` endpoints are nonworse rather than the
required four. The three improvements are the `y08` cases; all three `y00`
cases worsen slightly, reinforcing placement-dependent recurrent sensitivity.
No arm is selected and promotion is false. D081 therefore supports a small
early-window timing effect, not a safe or materially effective correction,
independent confirmation, data assimilation, bump transfer, or a larger-cap
recommendation.

### D083 predeclaration: bump query-graph transfer and rigid rotation (2026-08-09)

D083 evaluates the exact frozen D041 categorical PCNO on the 30-case open
validation population. The checkpoint, schema-4 normalizer, raw stride-1
recurrence, `model_all_nodes` physical policy, 80 retained frames, and bump
meanings `0 normal, 1 wall, 2 outflow, 3 inflow` are frozen. The existing D068
FP32 open-validation replay is a hard prerequisite: checkpoint, normalizer,
split, shard manifest, replay population, recurrence, and current isolated
inference source must bind before any new scientific inference. Sealed or test
cases, training, boundary clamping, and reference generation are excluded.
The replay payload SHA-256 is
`550c8357ae4f3d53624a9a6ed0a4ee880d6d3d582e4f07768495e2f2d029a204`;
its case-128 schema, 20-call sequence, raw recurrence, geometry, types, proxy
weights, and times must match exactly. Every project Python source loaded by the
evaluator must have an explicitly supplied matching digest. Scientific arms use
separate batch-one inference. The deterministic batch-two audit must agree with
sequential calls on the predicted-increment scale under the scale-aware limits
frozen below.

One case-128 H2 smoke is non-scientific and precedes H79. The scientific run is
the complete 30-case validation sequence in frozen split order. Inadmissibility
is retained as a scientific survival/truncation outcome rather than aborting the
process, but an arm first failing at H79 is not an H79 endpoint. Execution
completion, per-arm horizon admissibility, and full declared-matrix completion
are recorded separately; endpoint aggregates include only fully admissible arms
with a produced horizon output. Scientific interpretation and visualization
require every declared arm/case attempt to have terminal metadata, while
inadmissible arms remain valid survival/truncation evidence and animations show
only their available prefixes. A terminal receipt must bind both `summary.json`
and the artifact manifest.

Source revision A used the 23-file loaded-source manifest SHA-256
`09aa722a78e6e2eda7e16c5c659515bbec3fc261dbef3a02d6731ca62fedb646`.
H2-r1 passed frozen provenance and the 20-call D068 replay, then stopped before
new rollout interpretation because an added ordinary-wrapper versus
precomputed-Fourier identity gate used an unrealistically exact tolerance under
ordinary nondeterministic CUDA execution. Its maximum cross-path difference was
`1.07193e-3`, relative state L2 `1.08997e-6`. The retained log and launch
receipt SHA-256 values are `5a8ddb68...35f5` and `6b10f420...c716`; H2-r1 has no
scientific result.

A bounded case-128 probe found same-path ordinary-wrapper variation up to
`4.95911e-4`, same-path precomputed-Fourier variation up to `1.06430e-3`, and
cross-path variation up to `1.07193e-3`; all were only `3.12e-6` to `9.51e-6`
relative to the component-scaled predicted-increment norm. Under the established
deterministic CUDA setting `CUBLAS_WORKSPACE_CONFIG=:4096:8` plus deterministic
algorithms, all six repeat and cross-path comparisons were exactly zero. Source
revision B therefore requires that environment from process start, enables
deterministic algorithms only around the custom-wrapper and batch-two identity
gates, and then restores the ordinary runtime mode before scientific inference.
The ordinary 20-call replay and H2/H79 rollouts remain nondeterministic FP32;
this change isolates evaluator-path equivalence rather than changing the frozen
model's rollout runtime. Source B's 23-file loaded-source manifest SHA-256 is
`569c5513ddd8ae14de1c690a1d3120957ae0857fa465c0957ffe4f6f1931a7bb`.
H2-r2 passed the deterministic custom-wrapper identity exactly, then stopped
before new rollout interpretation at the separate batch-two control. Correct
batch-two versus sequential had raw maximum `5.22614e-4` but only `8.54124e-6`
component-scaled increment-relative error. The retained log and launch receipt
SHA-256 values are `7f6c4959...1664` and `724b1b01...9b36`; H2-r2 has no
scientific result.

The deterministic batch probe found all-normal raw maximum `1.84441e-3` and
increment-relative error `9.68111e-6`. The maximum pointwise component-scaled
difference was only `2.894e-5` of the correct-arm reference pointwise increment
maximum and `6.391e-5` for all-normal. The rejected raw maximum mixed physical
units across conservative components and was stricter than ordinary same-path
CUDA variation. Source revision C therefore makes every scientific correct and
all-normal inference a separate batch-one call, exactly matching the frozen
D068 replay batch size. Batch two remains only a deterministic audit control;
it must have component-scaled proxy-weighted increment-relative error at most
`2e-5` and pointwise component-scaled relative maximum at most `1e-4`. Raw
component maxima remain reported but are not the gate. A fresh H2-r3 under a
new immutable source is required before H79. Source C's 23-file loaded-source
manifest SHA-256 is
`56f2b52d0034bcab65e5f58c6bc3dfb1ecc5f9175b170795fe2aacba46b42074`.

Source-C H2-r3 is terminal `smoke_complete` after 25.002 s. Its summary,
artifact-manifest, terminal-receipt, launch-receipt, and log SHA-256 values are
respectively `64f1a8d6...532e`, `6837985c...8516`, `0f576d42...4ee2`,
`009991fa...11c`, and `8f803416...3b85`; the manifest and terminal bindings
independently verify. The exact 20-call replay passes at maximum absolute
`1.67656e-3` and relative L2 `7.05740e-7`. The deterministic custom-wrapper
identity is exactly zero. Correct/all-normal batch-two audit controls have
increment-relative errors `8.54124e-6/9.68111e-6` and pointwise scaled maxima
`2.89413e-5/6.39101e-5`; deterministic mode is restored and every scientific
inference remains batch one. All regenerated query and rotation geometry gates
pass. Native and query arms remain admissible through H2, while both rotated
arms first become inadmissible at call 2 with negative internal energy and only
`0.96104/0.96192` admissible fractions. The analytic rotation floor is zero,
but call-1 correct/all-normal `C_Q` is `0.13951/0.13229` and the corresponding
increment covariance defect is `1.51806/1.44163`. These one-case values are
non-scientific smoke routing evidence, not population estimates. The H2
summary's per-arm `endpoint_claim_allowed` values mean only that an admissible
smoke endpoint exists; top-level `scientific_interpretation_allowed=false`
governs, and its 30-case claim-boundary text describes the intended H79
population rather than the case-128 smoke. H79 is
authorized only through an exact hash binding to this H2 package and must retain
rotated inadmissibility as survival/truncation evidence rather than suppressing
or replacing it.

Exact-H2-bound `d083_bump_g1ab_h79_r1` was then launched from launcher SHA-256
`8519705d...edcb`. Its live launch receipt binds all five H2 prerequisite files,
the Source-C manifest, and the launcher itself. The first frozen validation case
completed and the detached evaluator remained healthy; no H79 result is
interpreted until all 30 declared case attempts terminate and the result package
passes local receipt, inventory, closure, and provenance verification.

G1a is explicitly a **proxy-mass query-graph representation transfer**, not PDE
resolution transfer. The retained bump package has reconstructed native
elements, proxy vertex lumps, and least-squares differential weights, but no
coarse solver mesh, coarse trajectory, native-to-coarse physical restriction,
or coarse solver provenance. For each case, deterministic ascending-index
maximal anchors cover each family-local node-type component to graph radius
two. A same-type multi-source graph assignment defines clusters. Coarse nodes
retain anchor coordinates and physical types; coarse proxy measure is the sum
of native reconstructed vertex lumps; restriction is the corresponding
componentwise proxy-mass average; prolongation is piecewise-constant cluster
injection. Quotient connectivity is regenerated from mapped native adjacency,
self edges are removed, symmetry/connectivity/rank are checked, and 2D
least-squares differential weights are recomputed. Correct-type and all-normal
arms differ only in the model-facing categorical tensor; physical masks and
semantics never change.

The frozen G1a matrix is native/correct, native/all-normal, coarse-query/correct,
and coarse-query/all-normal. Native rollouts and the restricted-native query
reference are evaluated through H79. The query reference is derived linearly
from the retained native trajectory,
so the algebraic restriction identity floor and the distinct native-grid
information-loss floor `||P R U-U||` are both reported. Metrics include raw
numerators and proxy-mass state-scaled rollout error on the arm geometry,
prolongated coarse-to-native error, teacher-forced and free increment
commutators, reference accumulated-change gap, recurrence closure, fixed
physical boundary bands, reference-defined shock structure, component rows,
admissibility, and final-horizon error. Aggregation is per case before the
30-case summary; nodes are never pooled between meshes or cases.

All component-scaled norms use
`||v||_{W,S}=[sum_i w_i sum_c (v_ic/S_c)^2 / sum_i w_i]^(1/2)`;
every ratio also stores the physical RMS numerator and denominator obtained with
`S=1`. State ratios use the frozen D041 state scale. Increment errors,
commutators, and covariance defects use the frozen residual scale and divide by
the same-representation reference increment norm. Boundary rows use physical
edge-path distances `0.05` and `0.10`. The bump shock proxy is the top 10% of
reference pressure-jump scores, expanded by physical edge-path distance `0.05`
and excluding the `0.10` boundary band; no vortex-core mask is inferred. The
visualization cases are frozen in order as `172`, `58`, and `187` before model
outcomes.

G1b asks only a 90-degree rigid-rotation consistency question. With
`Q=[[0,-1],[1,0]]` and casewise bounding-box center `c`, raw coordinates obey
`x'=c+Q(x-c)` and raw conservative state obeys
`(rho,m,E)'=(rho,Qm,E)` before the frozen normalizer. Physical bump types travel
with their subsets; density, energy, proxy measure, scalar Mach, and physical
policy are unchanged. D041 has no normal or vector boundary-data input, so the
audit records their absence rather than fabricating such channels. Connectivity
is regenerated and must equal the rigidly transported graph; differential
weights are recomputed from rotated coordinates and must satisfy
`W_Q=W Q^T` within the declared numerical tolerance.

The Fourier lattice is transported as a physical lattice, not rescaled by node
count or silently reconstructed from swapped scalar periods. Original learned
mode indices remain fixed, wavevectors obey `k'=Qk`, and the transformed phase
origin is `o'=c-Qc`, so `(x'-o') dot k'=x dot k`. The frozen spectral weights
therefore retain their exact index association. Synthetic gates require inverse
round trips for geometry/state/residual, conservative energy and proxy-mass
invariance, type counts and semantics, pairwise/edge metric invariance,
differential-weight covariance, Fourier phase/basis invariance, and exactly one
raw-state transform before normalization.

The frozen G1b matrix is native/correct, native/all-normal, rotated/correct, and
rotated/all-normal. For each type arm, report

\[
C_Q(t)=\frac{\|T_Q^{-1}\widehat U_Q(t)-\widehat U(t)\|_{W,S_U}}
              {\|U(t)\|_{W,S_U}},
\]

its raw numerator, the analytic reference transformation floor, inverse-frame
truth error, increment covariance, structure/boundary/component rows,
admissibility, and H79 error. This analytic transformed reference is not an
independent rotated PDE solve; D083 can establish only frozen-checkpoint
geometric consistency under the declared transformation. Fixed-scale figures
and animations use reference-only, rollout-wide physical scales and include
truth, predictions, residuals, residual errors, cross-arm defects, accumulated
defects, and signed local growth. Per-frame normalization and result-selected
cases/scales are forbidden.

### D083 terminal closeout and claim correction (2026-08-10)

The exact Source-C H79 package is terminal `complete` after 2635.921 s. All
seven declared payload hashes, the terminal summary/manifest receipt, the
23-source inventory, and the D068 replay binding verify locally. The summary
and artifact-manifest SHA-256 values are `ac7592a...a99be` and
`205ec70f...7a00`. All 240 declared phase/arm/case attempts have terminal
metadata and no output is missing, but only 61 reach H79; 179 terminate on
inadmissibility. Closure maxima remain at numerical precision: at most
`3.515e-14` absolute and `8.497e-15` relative.

The raw `metrics.csv` has 571,797 rows but only 546,573 semantic identities.
There are 25,224 extra rows, all exact multiplicity-two duplicates with no
conflicting payload. They come from overlapping all-component/all-region
emission by the general and native-region helpers. Endpoint summary rows are
unique, so the signed result remains interpretable after exact semantic-key
deduplication. All population curves and statistics below use one row per
`(case,phase,arm,mode,call,time,metric,frame,region,component)` and aggregate
cases before population summaries; nodes are never pooled. Future evaluators
must emit disjoint global/region rows and fail before writing on any duplicate
scientific identity.

G1a retains a median 1,745.5 of 21,187 nodes (`0.08223`). Proxy mass, constant
restriction, type stratification, regenerated rank, and all-frame proxy
integrals pass, but this remains query-graph resampling. At call 10, the
30-case median correct-type state error is `0.01731` on native geometry and
`0.24088` after query representation. The teacher-forced query/native increment
commutator is `0.50443`; its median raw physical numerator is `0.47799` against
`0.92802` for the reference increment. The free commutator is about `1.096`,
and same-input mesh and recurrent-state terms are both material. The
accumulated-change information-loss floor is `0.313` at call 10, compared with
`0.0797` for the full-state information-loss floor. This failure is therefore
not a small-increment-denominator artifact. Correct tags do not rescue the
severely compressed representation. Native/correct, native/all-normal,
query/correct, and query/all-normal have respectively `28/1/3/0` H79 survivors;
the three query/correct endpoints are survivor-only and cannot estimate
population H79 performance.

The executed G1b transform gates pass: reference floor zero, differential
covariance at most `8.527e-14` absolute, and transported Fourier-basis mismatch
at most `3.816e-6` absolute (`8.373e-7` relative). Nevertheless the call-1
30-case median teacher-forced increment covariance defect is `1.51337` for
correct types and `1.45294` for all-normal. Their raw physical numerators are
`1.78457/1.73322` against a `1.23426` reference-increment denominator. Fourteen
of 30 rotated cases fail on call 1 and the other 16 fail on call 2, identically
for both type arms; no rotated H79 endpoint exists.

Owner review exposed a decisive claim mismatch. D083 sent rotated coordinates,
raw transformed state, regenerated connectivity, and regenerated differential
weights directly through PCNO, with no inverse transform before inference.
However it also supplied `k'=Qk` and the transformed phase origin, making every
Fourier phase native-equivalent. D083 G1b therefore tests a hybrid analytic
covariance contract and remains useful as a negative control, but it does not
test deployment of the frozen checkpoint's native Fourier lattice on an unseen
domain orientation. It must not be labeled unseen-geometry generalization.

The bounded deduplicated analysis package contains survival, G1a mechanism, and
explicit legacy-G1b figures in PNG/PDF plus case-first CSV/JSON tables and a
hash manifest. The original visualizer is not used unchanged: it neither
deduplicates metrics nor reads `completion.csv`, and would include the first
rejected proposal followed by NaN frames. Corrected animations use the minimum
accepted native/transformed prefix, omit the first inadmissible proposal, and
skip zero-prefix requests with an explicit audit row.

### D084 terminal finite-inadmissibility result (2026-08-10)

D084 completed the exact-bound six-variant, two-repeat H79 matrix on all 30 open
bump validation cases. All checkpoint, normalizer, data, split, source, and
per-case frozen causal-boundary-policy gates passed. All 360 continuations
executed 79 calls with finite deployed conservative states; none reached the
registered nonfinite, 100-times-amplitude, or proxy-L2-at-least-10 event.

N0 and D082 strict completion are `28/30` and `19/30` in both repeats, with
mean strict errors about `0.07059` and `0.02017`. Zeroing D082 inflow fields
gives `23/30` completion in both repeats at essentially unchanged strict error;
zeroing all fields also gives `23/30` but raises error about 23%. Zeroing the
wall field preserves only `19/30` completion and raises error about 23%.
These are frozen-checkpoint field interventions; D082 versus N0 remains a
trained-model comparison. Zero-inflow is not a general stabilizer because its
maximum local amplitude ratios reach `25.94/31.19` across repeats.

All 98 repeat/variant/case inadmissibility events first arise from nonpositive
internal energy inside the fixed semantic collar, on 30 normal and 68 wall
nodes. Eight rows recover temporarily, with zero input-policy, eight model, and
20 output-policy recovery calls in total, but every naturally invalid row is
invalid again at call 79. All local amplitude crossings of 2, 5, or 10 follow
inadmissibility, but many invalid rows never grow that far and none explodes
globally. Together with U1's two recovered negative-pressure episodes and
finite H79 continuation, D084 rejects “inadmissibility implies blow-up” while
retaining inadmissibility as a local-growth warning. L3R-B0 was never launched.

The exact D019 five-trajectory HDF5 arrays, mapped preprocessing arrays, and
named checkpoint bytes were recovered on 2026-08-10. All 395 predictions are
finite; stored density and pressure are never negative and first reach exact
zero through positive-primitive exponential underflow. In all five trajectories,
speed exceeds 10 twelve calls before the first zero, fixed-reference amplitude
exceeds 10 two or three calls before it, and unweighted global relative L2
exceeds 1 two or three calls before it. D019 therefore shows rollout growth
before positivity underflow, not positivity underflow initiating growth. It
remains observational because checkpoint training/selection provenance is
incomplete and no matched repair counterfactual exists. Retrieval-manifest,
frame-metric, and analysis-summary hashes are `54c8158c...07deb`,
`ccb9b668...bad9`, and `86d95c5c...90aa9`.

Aggregate repeat results are stable, but only 141/180 pairs agree on the exact
first-invalid call/strict length, no state trajectory is bitwise identical, and
the largest repeat difference is 2024.27. Use repeat envelopes and label exact
case timing BF16/CUDA-sensitive. The canonical six case-23/54/128 publication
MP4s contain all 80 frames at 1430-by-638 and 5 fps under fixed reference-only
scales. They interpolate from every retained node, draw no mesh edges, nodal
cloud, or ordinary boundary-tag overlay, and have maximum saturation 0.0607%.
Result-summary and publication-visualization-manifest hashes are
`c3a4c202...bc971` and `e6296fcc...fd344`; the earlier
`39b28aa2...ba18c` scatter bundle is historical only. D084 is terminal and
closes without a causal blow-up claim or promotion of D082. It leaves no active
queue. Re-entry requires explicit owner authorization and first a deterministic
FP32 audit before any matched minimal-admissibility-repair counterfactual.

### D085 predeclaration: fixed-Fourier unseen-orientation stress (2026-08-10)

D085 is the owner-directed correction to D083 G1b; it does not overwrite D083.
The distinct ID is mandatory because D084 is already frozen for the unrelated
finite-inadmissibility continuation study.
It reuses the exact frozen D041 checkpoint, schema-4 normalizer, raw stride-1
recurrence, 30 open-validation cases, D068 case-128 replay, bump meanings
`0 normal, 1 wall, 2 outflow, 3 inflow`, and `model_all_nodes` physical policy.
No training, reference generation, sealed/test access, boundary clamping, or
boundary-condition change is permitted. D083 G1a is already closed and is not
rerun. D085's only scientific phase is G1b with native/correct,
native/all-normal, rotated/correct, and rotated/all-normal.

For casewise center `c` and the frozen 90-degree matrix `Q`, D085 still forms
`x'=c+Q(x-c)` and transforms raw conservative fields as
`(rho,m,E)'=(rho,Qm,E)` before normalization. Physical bump tags travel with
their subsets. Connectivity and differential geometry are regenerated from
`x'`. The checkpoint has no explicit normal or vector boundary input; scalar
Mach remains scalar and freestream transport occurs only through raw momentum.
Inverse rotation is forbidden in preprocessing and inference and is used only
after prediction for native-frame metrics.

The scientific Fourier contract is now deliberately non-covariant: the
checkpoint's original mode tensor, mode ordering, physical periods `[6,2]`, and
ordinary zero phase origin remain unchanged. The rotated arm uses the standard
PCNO preparation directly on `x'`, hence phases are `k dot x'`. No `Qk`, no
transformed origin, no period swap, no node-count scaling, and no
rotate-to-native inference shortcut is allowed. This is a zero-shot
fixed-representation stress on an analytically transformed orientation. Its
reference is still an analytic transform of retained trajectories, not an
independent rotated PDE solve; D085 cannot establish broad geometry
generalization or architecture-level rotational equivariance.

Before any scientific run, hard gates require: exact checkpoint, normalizer,
split, shard, replay, recurrence, and complete loaded-source bindings; raw
coordinate/state/residual round trips; type, proxy-mass, edge-metric, pressure,
and differential covariance; raw transform before normalization; checkpoint
mode hashes unchanged before and after preparation; ordinary fixed-mode tensors
on rotated coordinates; a diagnostic-only D083 transported basis that is
materially distinct and never attached to an inference arm; deterministic
custom/precomputed versus ordinary-wrapper identity on both native and rotated
case-128 inputs; direct rotated-node/raw-state preprocessing tests; batch-one
scientific inference; metric-identity uniqueness; and restoration of ordinary
CUDA mode after deterministic gates.

Metrics, denominators, reference-only scales, boundary/shock regions,
admissibility, signed growth, and mesh/state decomposition remain those frozen
for D083 G1b. The transformed reference floor and

\[
C_Q(t)=\frac{\|T_Q^{-1}\widehat U_Q(t)-\widehat U(t)\|_{W,S_U}}
              {\|U(t)\|_{W,S_U}}
\]

remain reported, together with raw physical numerators and teacher-forced
increment covariance. Animations use shared rollout-wide physical scales and
only the paired accepted prefix. A first inadmissible proposal may appear only
in a separately labeled failure diagnostic, never as an accepted rollout
frame. Case-128 H2 is non-scientific and must pass every new gate under a fresh
source manifest. H79 is GO only after independent review of that exact H2
package; otherwise scientific interpretation stops.

#### D085 H2 gate and H79 launch (2026-08-10)

The fresh case-128 H2 passed and two independent read-only reviews found no
contract or runtime blocker. All 23 loaded-source hashes, frozen checkpoint,
normalizer, split, open-validation manifest, and D068 replay bindings match.
The 20-call replay has maximum absolute error `1.0567e-3` and relative L2
`6.9563e-7`, within the frozen `2e-3/1e-5` limits. Native and rotated
precomputed-Fourier versus ordinary-wrapper identities are exactly zero, all
324 metric identities are unique, and commutator, recurrence, signed-growth,
and mesh/state closures hold to at most `1.11e-16` absolute error.

The corrected Fourier contract is nonvacuous. The checkpoint mode hash is
unchanged, the fixed basis on rotated coordinates differs from both the native
and transported bases by relative L2 `1.4058`, and the diagnostic transported
control remains native-equivalent to `8.2143e-7` relative error while being
excluded from inference. The analytic reference transformation floor and the
declared state, residual, pressure, proxy-mass, edge-metric, and differential
covariance floors pass; differential covariance is `6.39e-14` absolute.

Both rotated case-128 proposals remain finite but become inadmissible at call 1
through negative internal energy and pressure. Correct/all-normal admissible
fractions are `0.70993/0.71279`; minimum internal energies are
`-17.4128/-14.4256`. The teacher-forced call-1 increment covariance defects are
`2.39852/2.35286`, with raw physical numerators `1.93096/1.88494` against raw
true-increment RMS `0.99819`; free `C_Q` is `0.22162/0.21757`. Because inputs
are still aligned, the state-response term is exactly zero and the total defect
equals the same-input transformed-representation term. The initial failure is
therefore neither recurrence nor a small-denominator-only effect, although H2
does not isolate Fourier from the other coordinate-dependent PCNO pathways.

The exact-bound 30-case H79 matrix was authorized by both reviews and launched
under the same source snapshot, batch-one inference, fixed `[6,2]` lattice, and
process-start deterministic-gate environment. Its purpose is population
survival and full teacher-forced structure characterization, not confirmation
that rotated rollout succeeds. H2 remains non-scientific and supplies no
population estimate.

#### D085 H79 terminal result (2026-08-10)

The retrieved H79 package is terminal `complete` as an attempted matrix after
`2007.84` s. All seven declared result artifacts, terminal and launch receipts,
the exact H2 prerequisites, all 23 source bindings, checkpoint, normalizer,
split, open-validation manifest, and replay payload rehash correctly. D068
replay remains inside tolerance (`9.613e-4` maximum absolute and `6.853e-7`
relative L2). The fixed versus transported Fourier-basis relative difference
is `1.4047--1.4064`; checkpoint mode hashes remain unchanged, the diagnostic
transported control is never used for inference, and the analytic reference
transformation floor is zero. Differential covariance is at most `8.53e-14`
absolute. All `217404` D085 metric identities and `4800` closure identities are
unique; the largest paired commutator closure error is `4.44e-16`, while
recurrence, signed-growth, and mesh/state closures are exact at stored
precision.

All `120/120` phase-arm attempts are present. Native/correct and
native/all-normal reach H79 in `28/30` and `1/30` cases. Both rotated type arms
reach H79 in `0/30`: all `60/60` first proposals are finite but inadmissible,
and every case has negative internal energy and pressure. Correct/all-normal
median admissible-node fractions are `0.6121/0.6135`; nonpositive density is
secondary (`7/30` and `4/30`). There is therefore no accepted rotated free
rollout frame or rotated endpoint claim.

Case-first teacher forcing establishes a material same-input defect rather than
a small-denominator-only artifact. For correct types, D085 fixed-world versus
D083 transported-basis median defect ratios at calls `1/10/40/79` are
`2.3226/2.0332/3.9885/5.7616` versus
`1.5134/1.3529/2.7676/3.5465`. Their raw physical defect RMS values are
`2.4057/2.5378/2.7950/2.3431` versus
`1.7846/1.8398/2.1983/1.8628`; the identical raw true-residual RMS declines
from `1.2343` at call 1 to `0.4605` at call 79. Thus the late ratio increase is
partly denominator amplification, but the fixed-mode raw numerator remains
order `2--3`. The fixed relative defect exceeds the transported control in all
`4740/4740` matched case-arm-call rows; its raw numerator is larger in
`4697/4740`. At call 1 the free state-response term is exactly zero and total
equals the same-input transformed-representation term, so recurrence cannot
cause the initial failure.

The call-1 correct-type defect is largest relatively in `my` (`4.8358`) and
largest in raw RMS in energy (`2.2411`). Raw regional RMS is not confined to
the boundary: the `0.05` boundary band, shock, and smooth-region values are
`4.1493`, `3.5304`, and `1.5310`. Shock and smooth relative ratios are
denominator-conditioned and must not be quoted without these raw values; local
region RMS is also not a global energy share. Selected-case fields show a
broad, sign-coherent lower/interior drift plus local structure and widespread
negative pressure/internal energy. This is consistent with a large-scale
orientation/Fourier-basis mismatch, but no low-rank, POD, or spectral claim is
made.

D085 therefore closes as severe negative fixed-world-Fourier transformed-input
evidence on analytically rotated retained cases. D083 transports the combined
Fourier basis and phase representation and remains a matched negative control,
not the deployment policy. Since D083 itself retains a large defect and fails
by call 2, Fourier policy is a major exacerbating pathway rather than the sole
identified cause; learned differential/component mixing, anisotropic
normalization, and coordinate-branch interaction remain alternatives.
Quadrature/proxy-mass, connectivity, differential regeneration, reference
transformation, recurrence, and node-type semantics do not explain the initial
failure under their passing gates. This is not an independent rotated PDE
solve, broad or unseen-case geometry generalization, architecture-level
equivariance, boundary-condition improvement, or physical-conservation result.

This negative result is architecturally expected. The frozen checkpoint was
trained on one orientation and has no enforced rotation representation: raw
absolute coordinates enter the lift, Fourier phases use an axis-fixed physical
lattice, conservative momentum components use the frozen componentwise
normalizer, and the two regenerated least-squares gradient channels are mixed by
unconstrained learned weights. The scalar Mach input supplies no separately
rotated freestream vector. Regenerating a covariant graph and differential
stencil therefore fixes the geometric inputs but does not make the learned map
covariant. D085 tests the complete frozen coordinate representation on an
unseen orientation of the same retained cases; it is neither a new-case test nor
an isolation of the Fourier branch. The matched D083/D085 difference identifies
the combined Fourier basis/phase policy as one exacerbating pathway only.

The corrected local analysis bundle is
`artifacts/time_dependent_no/d085_bump_fixed_fourier_rotation_20260810a/h79_r1_analysis_20260810b`;
its 21 declared payloads rehash exactly. The
standard fixed-scale visualizer correctly creates no accepted-prefix GIFs and
records all 24 requests as `skipped_no_admissible_frames`. Separately labeled
first-rejected-proposal panels exist for cases `172/58/187`. The remaining
decisive artifact request is a selected-case, calls-1--79 teacher-forced replay
retaining native, fixed-mode, and D083 transported residual fields plus nodes,
proxy weights, types, boundary distances, shock masks, and reference-only
scales for shared-scale animation, invalid-node overlays, regional energy
shares, POD, and physical-wavelength analysis.

### D086 predeclaration: rotated finite-invalid visualization continuation (2026-08-10)

D086 is a bounded visualization follow-up to terminal D085. It does not reopen
D085's accepted-rollout matrix and does not promote its first rejected proposal
to a valid state. The exact D085 checkpoint, normalizer, split, open-validation
shard manifest, D068 replay, `model_all_nodes` policy, bump type meanings
`0 normal, 1 wall, 2 outflow, 3 inflow`, 90-degree raw state/coordinate
transform, regenerated graph/differential geometry, fixed checkpoint Fourier
modes, `[6,2]` periods, zero phase origin, and batch-one direct inference remain
frozen. There is still no inverse transform before model inference. No new PDE
reference, training, projection, boundary change, sealed/test access, or
transported Fourier basis is permitted.

The visualization population is exactly cases `172/58/187`, chosen before this
continuation by D085, with correct-type and all-normal arms and two ordinary-CUDA
repeats. Starting from the analytically rotated frame-0 state, raw recurrence
continues through finite inadmissibility for at most 79 calls and stops an arm
only after its deployed conservative state becomes nonfinite. Unavailable later
slots remain explicit NaNs. Call 1 must replay the signed D085 rejected proposal
within the frozen replay tolerance and must remain finite/inadmissible; failure
of that binding stops interpretation. Repeat differences, first nonfinite call,
admissible-node fraction, invalid-node count, minimum density/internal energy/
pressure, component-scaled proxy-weighted state error, predicted-increment
error in true-residual units, and raw physical numerators are retained per case,
arm, repeat, and call. Bump weights remain proxy weights, not physical volumes,
and nodewise values are never pooled across cases.

For repeat 0, each case receives two rotated-coordinate animations. The state
movie compares analytic transformed-reference, correct-type, and all-normal
pressure plus their errors and the between-arm difference. The residual movie
shows all four conservative true increments, correct-type predicted increments,
and their errors. Invalid nodes and the first rejected call are overlaid. Every
state/update scale is fixed from the full transformed reference only; scales are
shared across compared panels and never normalized per frame or selected from a
model result. Movies include the common finite prefix only and label themselves
`finite-invalid diagnostic -- not an accepted rollout`. D086 can show how a
failed frozen representation evolves numerically; it cannot establish rotated
PDE accuracy, geometric generalization, stability, conservation, or a causal
failure mechanism.

#### D086 terminal result (2026-08-11)

The exact-bound continuation completed with return code zero in `119.86` s.
The source-binding and source-snapshot digests are `1c81b197...cd847a` and
`798f3d78...ad780e`; the retrieved run summary, artifact manifest, and terminal
receipt have SHA-256 `fd166149...bd4a10`, `c798ddc0...f54981`, and
`28ad7e04...4fcfdb`. All ten manifest payloads rehash exactly, the terminal
receipt binds the summary and manifest, and the run contains exactly `948`
unique call rows (`3 cases x 2 type arms x 2 repeats x 79 calls`) plus 12 unique
terminal-event rows. The D068 replay, direct rotated-input geometry/source
gates, and signed D085 call-1 replay all pass. Across the six case/arm call-1
replays, state relative L2 is at most `6.08e-7` and component-scaled proxy RMS
at most `1.29e-5`.

Every one of the 12 raw recurrences is inadmissible at call 1, yet all remain
finite through call 79. Numerical finiteness therefore does not rescue the
physical rollout. Using repeat 0 and aggregating cases before medians, the
correct-type admissible-node fraction at calls `1/10/20/40/79` is
`0.635/0.417/0.309/0.295/0.295`. The corresponding state-error ratios are
`0.172/2.756/16.695/631.725/782783.776`. Raw physical predicted-increment error
RMS grows from `2.258` to `6.993/41.907/1722.128/2351205.464`, while the raw
true-increment RMS is `1.220/1.217/1.092/0.727/0.417`. The late ratio is
amplified by the smaller true increment, but the exploding raw numerator rules
out a denominator-only explanation. All-normal medians are nearly the same
(`0.170` state error at call 1 and `779568.407` at call 79), so removing tags
does not explain or repair the initiating transformed-representation failure.

The family-local spatial audit keeps bump meanings separate. At call 1, the
correct arm already invalidates a case-first median `36.2%` of type-0 normal
nodes. Type-0 nodes carry `96.7%` of the component-scaled proxy-weighted
increment-error energy; this is a proxy diagnostic, not physical conservation.
Wall and inflow subsets have larger local RMS, but the error and invalidity are
not confined to them. Under ordinary CUDA, the two repeats agree on call-1
rejection and H79 finiteness but are not field-identical. Their correct-arm
case-first median state gap relative to the analytic-reference state scale grows
from `3.79e-7` at call 1 to `2.79e-5` at call 10, `4.93e-3` at call 40, and
`5.67` at call 79 (`6.22` for all-normal). The initiating failure is repeatable;
the detailed late invalid trajectory is path-sensitive.

The verified visualization package is
`artifacts/time_dependent_no/d086_bump_rotated_invalid_continuation_20260810a/visualizations/d086_selected_failure_movies_r2/`.
Its manifest and terminal receipt hashes are `621c2d8f...e685e3` and
`523f629c...f623e0`. All six MP4 hashes and the metric-figure hash verify; state
movies have 80 frames and residual movies 79, with reference-only rollout-wide
scales, no inverse rotation, no per-frame normalization, and explicit
invalid-node overlays. Across all three cases, true increments stay localized
while predicted increments and errors saturate broad interior/lower regions in
every conservative component. This supports a large-scale orientation/
representation failure followed by explosive recurrence, but not a unique
deterministic late field, a low-rank/POD claim, accepted rollout, independent
rotated PDE, broad geometry generalization, stability, or conservation.

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
| Native residual correction and transfer-native baseline | D071, D074, D075, D076, D077 |
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
