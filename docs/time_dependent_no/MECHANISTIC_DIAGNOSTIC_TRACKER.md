# Time-Dependent Experiment Index

Updated: 2026-08-11

Status: compact routing index; not an execution queue

## Purpose

This file routes stable experiment IDs to the byte-preserved detailed ledger:

[history/MECHANISTIC_DIAGNOSTIC_TRACKER_through_2026-08-11.md](history/MECHANISTIC_DIAGNOSTIC_TRACKER_through_2026-08-11.md)

Archive SHA-256:
`CA8CF201FD2CD76AE086241BCF858C36EC7F20375A2AD5E9F97A811D213CBC03`

The archive preserves the expanded tracker through D086 exactly as it existed
before this compaction: dated contracts, metrics, hashes, stop decisions,
artifact paths, and historical forward-looking language. Historical language
records the campaign state at that time; it does not authorize or prohibit
current work. Current explicit human direction and
[RESEARCH_DIRECTION_DECISION.md](RESEARCH_DIRECTION_DECISION.md) govern the
scientific framing.

Normal startup should read this index, not the full archive. Search by stable
ID, then read only a bounded range around the matching row or heading. Example:

    rg -n "\bD081\b" docs/time_dependent_no/history/MECHANISTIC_DIAGNOSTIC_TRACKER_through_2026-08-11.md

Do not reconstruct missing checkpoint, source, command, data, environment, or
artifact identity from memory. Retrieve the exact archived contract and verify
the current source and artifact manifests before execution or interpretation.

## Stable ID Rules

- The registered run universe through D086 contains 115 IDs: D001--D086,
  23 L3R IDs, L4A-001--L4A-004, L1-OOD, and XLINE-001.
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

Every D-series ID is contiguous from D001 through D086. Use the archive for
exact populations, contracts, metrics, thresholds, source and artifact hashes,
attempt histories, and claim language.

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
| D064 | Residual-scale accumulation and error structure | Completed; signed accumulation extends beyond small-denominator effects. |
| D065 | Instrumented resolution pathways | Completed; no single pointwise, differential, or spectral family dominates. |
| D066 | Physical-scale and temporal-coherence decomposition | Completed; persistent large-scale drift separates from local cancellation. |
| D067 | Unified cumulative/pathway/shock confirmation | Completed; bounded two-component defect mechanism with closure-qualified replay. |
| D068 | Node-type interventions and nonperturbing instrumentation | Completed; family-local semantics, with bump wall labels dominant on tested geometries. |
| D069 | Constant-zero-channel training comparison | Completed; three seeds do not support an H30 benefit from zero channels. |
| D070 | Fine-grained PCNO mesh-pathway surgery | Completed under D070C; differential and nested fixed-hop pathways supported, not uniquely attributed. |
| D071 | Frozen residual correction | Completed on dynamic FV and bump; neither family-local correction promoted. |
| D072 | Fixed-physical-width semantic boundary fields | Completed, not promoted; no general boundary-condition or resolution-transfer claim. |
| D073 | Physical-radius differential geometry | D073-A completed and supports same-hidden pathway relevance; D073-B rollout and bump safety remain unrun. |
| D074 | Native residual correction and transfer-native comparator | Dynamic H30 completed; nonzero arms helped endpoints but failed no-harm gates, so zero was selected. |
| D075 | Conservative-integral-neutral correction | Dynamic H30 completed; selected raw correction improved endpoints/residuals but failed promotion controls. |
| D076 | Short-window state-response gain pilot | Completed; calibration did not qualify and evaluation stopped before targets loaded. |
| D077 | Strength-grouped response controller | Failed immutable repeatability contract; favorable numbers remain diagnostic only. |
| D078 | Deterministic CUDA replay attribution | Completed on dynamic open validation; registered adaptive promotion conjunction passed. |
| D079 | Deterministic process confirmation | Completed; byte-identical same-population process repeat, not independent confirmation. |
| D080 | Causal local-channel correction pilot | Completed on dynamic FV; early small gains reversed and no recurrent arm promoted. |
| D081 | Shock-normal timing versus dose | Completed on dynamic FV; early-window timing supported, practical efficacy and promotion failed. |
| D082 | Semantic collar residual side branch | Bump checkpoint exists; later frozen interventions show field-use tradeoffs, but D082 is not promoted. |
| D083 | Bump query-graph transfer and transported-Fourier rotation | Completed; negative proxy-mass compression evidence and analytic covariance control, not PDE transfer. |
| D084 | Finite-inadmissibility and bump-field continuation | Completed; inadmissibility did not imply global blow-up and field zeroing did not promote D082. |
| D085 | Fixed-Fourier bump orientation stress | Completed; every rotated proposal failed at call 1, before recurrence. |
| D086 | Rotated finite-invalid visualization continuation | Completed; all continuations stayed finite through H79 but were inadmissible from call 1. |

## Evidence And Claim Boundaries

- Keep 1D, bump, dynamic finite-volume, and future REALM results in separate
  family-local tables. Do not pool them or infer transfer without a registered
  cross-family experiment.
- Bump quadrature-density weights are proxy weights. Bump integral summaries do
  not establish physical conservation or exact DG replay.
- Dynamic FV physical-volume diagnostics support only their frozen mesh,
  population, horizon, precision, and recurrence contracts.
- D063--D067 provide bounded common-source and pathway evidence, not proof of
  continuum operator learning or arbitrary-resolution generalization.
- D068--D072 and D082--D084 concern representation, routing, and frozen policy
  use. They do not establish a generally optimal boundary encoding or improved
  physical boundary condition.
- D083 query compression is not PDE resolution transfer. Its transported basis
  is an analytic covariance control; D085 is the distinct fixed-Fourier
  orientation stress, and D086 contains invalid failure continuations only.
- D077's failed contract cannot be repaired by its favorable aggregate metrics.
  D078 and D079 establish deterministic same-open-population replay, not new-data
  confirmation.
- D084 rejects "inadmissibility implies blow-up" for its exact matrix; it does
  not show inadmissibility is harmless or identify a unique instability cause.
- REALM has no registered D-series result here. Route current audit and baseline
  planning to the weekly plan, and do not conflate REALM with RealPDE Track 2.
- Populations marked sealed by their contracts remain sealed. In particular,
  D064--D086 do not authorize dynamic strength-OOD/test, bump test, new reference
  generation, or the historical L3R-F0 one-time evaluation.

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

| Question | First IDs or document to inspect |
| --- | --- |
| D019 versus D041 provenance | L3R-M0, D019, D041 |
| Long-horizon stability and admissibility | D019, D041, D084, D086, L3R-U0, L3R-U1, L3R-B1 |
| Boundary closure and objective history | D040, D041, L3R-BG0, L3R-RB0, L3R-RA0P, L3R-K2D0 |
| Boundary descriptor/routing use | D063, D068, D069, D072, D082, D084 |
| Fresh versus propagated error | D053, L3R-K2D0 |
| Ripple/high-frequency mechanisms | D013, D036, D038, D043, D052--D056, D060 |
| Front position and shock identity | D006, D035, D060--D062, D080--D081 |
| Large learned steps | D027--D035, D060--D061 |
| Resolution transfer | D029, D030, D063--D067, D073--D075 |
| PCNO pathway attribution | D052, D065, D067, D070, D073 |
| Native residual correction/controllers | D071, D074--D081 |
| Orientation and graph transformation | D083, D085, D086 |
| Physical-conservation limits | D008, D037, D045--D051, RC-13 in the decision file |
| Latent forecasting readiness | L4A-001--L4A-004 |
| REALM benchmark validation | [WEEKLY_RESEARCH_PLAN.md](WEEKLY_RESEARCH_PLAN.md), W26-L4 |

## Archive And Recovery Anchors

- The through-2026-08-11 archive above is the authoritative detailed ledger for
  D001--D086 and all registered non-D IDs in this index.
- Commit `5646bfb` adds the byte-preserved pre-compaction tracker state; the
  archive, not the commit diff, is the normal retrieval target.
- The earlier through-2026-08-01 archive has SHA-256
  `D055F2EEB398224CEFDBDD86D3A230C501DFA4069D65DF188C7F8183F0F821CE`.
- Commit `ae1f402` preserves the D063 evidence addition. Commits `e2070f6` and
  `ce5d6a2` preserve the prior documentation and source cleanup anchors.
- Retired entry points remain recoverable from the source anchor named by the
  corresponding archived record. Recovery does not imply current maintenance
  status or execution authorization.

Generated arrays, figures, logs, checkpoints, and manifests remain under the
ignored `artifacts/time_dependent_no/` tree. This index is not a substitute for
an artifact digest, exact retrieval request, or current manifest verification.
