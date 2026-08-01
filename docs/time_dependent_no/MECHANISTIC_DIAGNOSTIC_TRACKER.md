# Time-Dependent Experiment Index

Updated: 2026-08-01

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

- The primary run universe contains 92 IDs: D001--D063, 23 L3R IDs,
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

Every D-series ID is contiguous from D001 through D063. Exact populations,
metrics, thresholds, artifacts, and claims remain in the historical ledger.

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
| Boundary information and hard closure | D040, D041, L3R-BG0, L3R-RB0, L3R-RA0P, L3R-K2D0 |
| Fresh versus propagated error | D053, L3R-K2D0 |
| Ripple/high-frequency mechanisms | D013, D036, D038, D043, D052--D056, D060 |
| Front position/identity | D006, D035, D060--D062 |
| Large learned steps | D027--D035, D060--D061 |
| Resolution transfer | D029, D030, D063 |
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
