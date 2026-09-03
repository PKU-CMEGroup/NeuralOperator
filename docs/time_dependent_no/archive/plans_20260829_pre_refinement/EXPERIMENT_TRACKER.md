# Active Experiment Tracker

Updated: 2026-08-29

Status: current block/authorization tracker. This is not the historical
experiment ledger and does not authorize execution.

Historical IDs and evidence are routed through
[MECHANISTIC_DIAGNOSTIC_TRACKER.md](MECHANISTIC_DIAGNOSTIC_TRACKER.md). Exact
old contracts are recovered from their Git or artifact anchors when needed.

## Status Vocabulary

| Status | Meaning |
| --- | --- |
| `PLAN_REVIEW` | Design exists and awaits owner review. |
| `PLANNED_NOT_AUTHORIZED` | Scientifically selected, but implementation or execution has no current approval. |
| `BLOCKED_BY_PREREQUISITE` | A named scientific/provenance prerequisite is absent; no automatic retry or pivot. |
| `EVIDENCE_ONLY` | Completed historical result relevant to the story, but not a prospective active block. |
| `PARKED` | Preserved but outside the current critical path. |
| `COMPLETE` | The exact block and its required packet are complete. |

Authorization is tracked separately from scientific status.

## Core Blocks

| Block | Claim role | Scientific status | Authorization | Next admissible action |
| --- | --- | --- | --- | --- |
| T0 theory and terminology | C1 | High-level claims and corrector terminology approved; exact document wording awaits owner inspection. | Documentation only. | Owner inspects active plans; then align manuscript claims/terms. |
| B1 exact ODE laboratory | C1 boundary cases; diagnostic ground truth | `PLAN_REVIEW` | Not authorized to implement or execute. | Review exact system, scenarios, endpoints, and stop rule. |
| B2 learned ODE intervention sandbox | Supporting mechanism realization; not the prospective PDE test | `PLANNED_NOT_AUTHORIZED` | No implementation/training approval. | Select simple representative slots only after B1 closure and named approval. |
| B3 primary complex-PDE diagnosis and prediction freeze | Mandatory C2 case-study diagnosis | `BLOCKED_BY_PREREQUISITE` | PDE qualification and scientific execution are not authorized; old M1 Q2 remains closed separately. | Select the PDE and representative implementations, then qualify reference/population/restart/resource contracts under new identities. |
| B4 primary complex-PDE intervention study and reveal | Mandatory C2 prospective test | `BLOCKED_BY_PREREQUISITE` | No training, corrector execution, checkpoint evaluation, or reveal approval. | Begins only after the B3 freeze and blocking intent-to-code audits. |
| B5 secondary-PDE transfer | Conditional regime-transfer evidence | `PARKED` | None. | Consider a reduced matrix only after the primary B3/B4 case study is complete. |

## Current Parent Evidence

| Evidence | Live verified state | Use in current paper |
| --- | --- | --- |
| D094 bump response surface | Closed retrospective development evidence; long-horizon outcomes already open. | Motivation and ranking-reversal phenomenon only; never the prospective C2 test. |
| M1 R2-REF spatial | `spatial_candidate_qualified`; result `f32b2d4d...16805ed4`; manifest `9f8de40a...8dd67c84`. | Finite-grid parent evidence on six fixed trajectories. |
| M1 R2-REF temporal R1 | `reference_candidate_qualified`; result `460eb8d3...c847196`; manifest `af0b7bb1...ebb5412`; launcher exit-code capture caveat preserved. | Completes registered finite-grid N256 reference qualification with spatial packet. |
| M1 R2-POP launch 0 | `incomplete_8h_wall_time_cap_before_packet`; no result/manifest/series. | Infrastructure provenance only. |
| M1 R2-POP R1 | Receipt `runner_failed_confirmed_nonzero`, exit `1`, elapsed `22307.064 s`; no result/manifest/series/output directory; no traceback; infrastructure classification null. | No scientific result and no stationarity conclusion. Q2 remains closed. |
| P0 native-coarse restart | Solver repeatability passed; registered solver-bias qualification failed. | Explains why the dynamic-FV coarse map is not a trusted displaced-state target. |
| P1 path-conditioned tube synthetic stage | Synthetic algebra/plumbing only; real checkpoint and prospective evaluation absent. | Supporting diagnostic vocabulary only. |

The verified M1 reference packets do not make the old residual-FNO M1 Q2
contract the active B3 study. The current PDE empirical decision fixes PCNO;
any reuse requires a new model/source identity.

## Immediate Review Checklist

- [x] Owner approves the high-level two claims and corrector terminology; exact
      document wording remains open for inspection.
- [ ] Owner approves B1 exact ODE system and scenario matrix.
- [ ] Owner selects the simple B2 mechanism representatives and three-seed
      contract; no ODE ranking is transferred directly to a PDE.
- [ ] Owner selects one primary complex PDE and the exact implementations for
      the embedded, structural, and explicit slots, and decides whether the
      hybrid contract is ready for PDE promotion.
- [ ] Every retained B4 slot has a written intent contract and a passing
      code-path audit before execution.
- [ ] No manuscript section treats dynamics relabeling as the preselected
      winner.
- [ ] The qualitative signature table is converted into signed,
      problem-specific predictions before the B4 reveal.
- [ ] No active document calls D094 prospective.
- [ ] No active document implies another R2-POP retry is authorized.
- [ ] No active document treats REALM as a candidate for this paper.

## Execution Log

No experiment, model, solver, remote, sealed, or scientific-data execution was
performed as part of the 2026-08-29 documentation rewrite.
