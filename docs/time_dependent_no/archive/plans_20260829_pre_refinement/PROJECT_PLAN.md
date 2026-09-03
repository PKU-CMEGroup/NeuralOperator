# Project Plan: Corrective Mechanisms For Long-Horizon Neural Operators

Updated: 2026-08-29

Status: active master plan. Planning only; no experiment execution is
authorized by this document.

## Objective

Deliver a rigorous and empirically testable framework for explaining why clean
one-step accuracy can fail to predict autoregressive rollout quality, and for
selecting corrective mechanisms for long-horizon time-dependent PDE prediction.

The project is successful when it provides:

1. a qualified mathematical account of the clean-trace/deployment-response
   gap;
2. a common, non-tautological vocabulary for corrective mechanisms;
3. measurable, prospective predictions that separate models beyond one-step
   error;
4. a cutoff-bounded, comprehensive, dated field study and taxonomy with representative
   empirical tests;
5. controlled evidence for when embedded and operational corrective mechanisms
   are useful in a mandatory complex-PDE case study; and
6. a serious submittable paper whose claims match its verified evidence.

## Two Claims And Their Owners

| Claim | Required deliverable | Main falsifier |
| --- | --- | --- |
| C1: clean one-step supervision does not identify deployment-relevant transverse response in a sufficiently rich class; uniform control requires target-separating information or a justified restriction. | Exact self-composition identity, conditional non-identifiability theorem, minimax corollary, tube/path bounds, counterexamples, and precise scope. | The argument omits the hypothesis/world class, confuses a finite cloud with a manifold, or claims necessity when structure already identifies the response. |
| C2: frozen response/forcing/drift diagnostics predict rollout ranking reversals and regime-dependent intervention value beyond clean error for a fixed PCNO backbone. | ODE calibration plus a mandatory complex-PDE case study with a prospective freeze, representative embedded and operational mechanisms, and outcome-blind ranking evaluation. | The score works only after outcomes are known, does not improve held-out ranking, or changes without the predicted rollout effect. |

The field study, ODE laboratories, case study, and proposed intervention support
these claims. They do not become extra headline contributions.

## Deliverables

### D1. Current research package

- [RESEARCH_DIRECTION_DECISION.md](RESEARCH_DIRECTION_DECISION.md)
- this project plan
- [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md)
- [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)
- [EXPERIMENT_TRACKER.md](EXPERIMENT_TRACKER.md)
- [PARKED_EXPERIMENTS.md](PARKED_EXPERIMENTS.md)
- [WEEKLY_RESEARCH_PLAN.md](WEEKLY_RESEARCH_PLAN.md)
- [the active paper plan](../../paper/PAPER_PLAN.md)

Acceptance: every active document uses the same scope, two claims,
terminology, empirical blocks, authorization boundaries, and M1 readiness
state.

### D2. Theory and terminology freeze

- Retain the complete deployed transition as the invariant object.
- Make the Gronwall qualification explicit.
- Keep the clean-trace theorem conditional on normal richness.
- Define a response-controlled region as one where response is identified by
  data, constrained by structure, or actively corrected or conditioned.
- Separate tube retention, tangent/path accuracy, admissibility, boundedness,
  finiteness, and statistical fidelity.
- Define corrective mechanism, embedded corrective mechanism, and operational
  corrector without retrospective relabeling.
- State the exact recovery and dynamics-relabeling targets.

Acceptance: every theorem and proposition is labelled as an identity,
conditional theorem, sufficient condition, or empirical hypothesis; every
manifold/tube assumption is explicit.

### D3. Supporting ODE laboratory

- Implement an analytically solvable tangent-normal flow with controllable
  forcing, normal response, and normal-to-tangent coupling.
- Demonstrate smaller-one-step/worse-rollout reversals.
- Produce predeclared regimes in which recovery and dynamics relabeling win for
  different reasons.
- Compare exact geometry with kNN, local-PCA, and convex-hull proxies.
- Train one shared residual learner under matched target semantics for three
  paired seeds.

Acceptance: exact formulas, code identities, and predicted regime boundaries
agree; learned results are reported separately from exact results and are not
retrofitted after outcome inspection. These experiments calibrate the
framework and its measurements; they do not replace the PDE case study.

### D4. Mandatory fixed-PCNO complex-PDE case study

- Select and qualify one reasonably complex, restartable PDE reference law and
  population without changing the fixed PCNO backbone.
- Freeze a common query bank and diagnostic estimator.
- Compare a frozen representative set spanning embedded training/structure
  mechanisms and deterministic and learned operational correctors; include the
  relabeling-plus-recovery hybrid if its contract and audit close.
- Write predicted changes in response, forcing, drift, path accuracy, physical
  diagnostics, and cost before results are opened.
- Freeze H1/common-bank diagnostics and predicted rankings before designated
  long rollouts.
- Evaluate ranking, mediator movement, path/tube behavior, structure, no-harm,
  and cost.
- Before any run, audit each arm's mathematical target, information source,
  deployed composition, recurrent feedback state, and identity ablation. Stop
  and ask the owner if any scientific intent is ambiguous.

Acceptance: source, split, population, checkpoint, evaluator, prediction, and
result identities are separately hash-bound; three paired seeds are complete;
the prospective analysis follows the frozen rule.

Current constraint: the M1 finite-grid reference is qualified, but its
fresh-population sampling law is not. The R2-POP R1 process exited nonzero
without a packet. It is one possible parent reference, not a selected PDE.
Q2 and D4 execution remain closed pending a new explicit owner decision and
contract.

### D5. Manuscript and review package

- Reconcile `paper/main.tex` and sections to the active paper plan.
- Replace the one-sided relabeling expectation with an outcome-neutral
  regime-dependent prediction.
- Integrate only manifest-verified results.
- Produce the hero figure, core result figures/tables, limitations, literature
  ledger, and a claim-evidence checklist.
- Build and inspect the PDF after each structural milestone.

Acceptance: no planned result is written as a finding; every numerical claim
maps to a verified packet; every citation maps to the source ledger; the PDF
builds without unresolved cross-references or red evidence placeholders in the
claimed-complete version. The mentor-review draft includes the complete primary
PDE case study and is serious enough to submit after normal internal review.

## Critical Path

```text
scope and vocabulary freeze
        |
        +--> paper story and theory alignment
        |
        +--> supporting exact/learned ODE laboratory
        |
        +--> immediate primary-PDE selection and readiness
                  |
                  +--> freeze B3/B4 --> audit --> train/evaluate --> reveal
        |
        +--> integrate verified evidence --> full draft audit
```

## Work Packages

| Work package | Scope | Output | Dependency |
| --- | --- | --- | --- |
| WP0: plan freeze | Align active documentation and remove stale navigation. | D1. | None. |
| WP1: theory and field study | C1, terminology, theorem ladder, counterexamples, and cutoff-bounded family-level literature audit. | Revised manuscript Sections 1–5, taxonomy ledger, and appendices. | WP0. |
| WP2: supporting ODE evidence | Exact B1 and exploratory learned B2. | Reproducible packets and one compact main-text result. | WP0; execution requires separate authorization. |
| WP3: PCNO case-study readiness | Select and qualify the primary PDE and freeze the representative intervention set. | Qualified reference/population and immutable B3/B4 preregistration. | WP0; immediate owner decision. |
| WP4: primary PDE evidence | Audited representative arms and prospective ranking reveal. | B3/B4 result packets, tables, figures, and case-study walkthrough. | WP3; separately named training and reveal approvals. |
| WP5: synthesis | Integrate evidence, limitations, and review package. | D5. | WP1–WP4. |

## Two-Week Completion Definition

“First complete draft” means a serious submittable manuscript with:

- coherent prose in every section rather than outline fragments;
- complete theory and a cutoff-bounded field study across the declared
  mechanism families;
- verified exact and learned ODE evidence used as a compact supporting
  laboratory;
- a completed, verified primary complex-PDE case study with representative
  embedded and operational corrective mechanisms;
- a prospective PCNO diagnostic freeze and long-rollout analysis completed
  under the registered contract;
- all figures and tables shown as results are backed by verified packets; and
- the PDF builds and passes a claim/provenance consistency review.

If the mandatory PDE evidence is unavailable, the project may still deliver a
useful progress draft, but it has not met the two-week complete-draft gate. The
deadline never justifies an unqualified PDE or invented result.

## Decision Gates

### Gate G0: owner review of this documentation

The owner confirms the two claims, terminology, five bounded empirical blocks,
parked set, and two-week cutoff before manuscript or implementation work
proceeds.

### Gate G1: exact ODE mechanism closure

The analytical recurrence, regime boundaries, and code agree. Failure narrows
the framework before any PDE claim.

### Gate G2: learned ODE realization

The matched learner changes the declared response components in the expected
directions. Failure is reported as a representation/optimization gap, not
hidden by new architectures.

### Gate G3: PDE readiness

One PDE must have a qualified reference, population, state closure, restart
contract, solver-bias margin, open split, and resource envelope early enough to
complete WP4. A failed candidate triggers an immediate owner decision on the
remaining solver-ready candidates, not an ODE-only completion claim.

### Gate G4: prospective freeze

Before designated long-rollout outcomes are opened, freeze the source,
population, models, query bank, diagnostics, horizons, rankings, statistical
rules, and no-harm/cost gates.

### Gate G5: claim decision

C2 is retained only if the prospective result adds held-out value beyond clean
error and the mediator/outcome relationship follows the registered rule.

## Risk Register

| Risk | Consequence | Mitigation / stop rule |
| --- | --- | --- |
| “Near the manifold” becomes tautological. | Weak novelty. | Make prospective separation and intervention prediction—not contemporaneous distance/error correlation—the empirical contribution. |
| Smooth-manifold assumptions fail for a PDE. | Invalid tangent/normal language. | Use exact geometry only in ODEs; call PDE quantities reference-proximity or local reconstruction diagnostics unless geometry qualifies. |
| Recovery/relabeling comparison is biased toward one target. | Circular result. | Same inputs and budgets; score each declared target plus ID rollout; predict ranking before reveal. |
| Drift detector changes with the model. | Instrument confounding. | Fit one state-only estimator on clean reference data and freeze it across methods. |
| M1 population remains unqualified. | One candidate route is unavailable. | Select the primary PDE from solver-ready candidates immediately; do not treat M1 or REALM as mandatory. |
| PCNO compute exceeds the schedule. | Draft delay. | Measure one smoke contract before estimating; run only the frozen representative arms and no architecture sweep. |
| Tube retention improves but phase worsens. | Stable but inaccurate method. | Tangent/path no-harm is mandatory. |
| Literature coverage is incomplete. | The unifying claim is unconvincing. | Run a dated, reproducible field study across the declared mechanism families; keep empirical implementations representative rather than exhaustive. |

## Parking And Reactivation

The parking ledger is [PARKED_EXPERIMENTS.md](PARKED_EXPERIMENTS.md). A parked
line may re-enter only when all are true:

1. it directly tests C1 or C2 or fills a named required evidence gap;
2. its target, information source, population, metric, and cost are declared;
3. it does not duplicate a closed historical experiment;
4. it receives a new identity and preregistration where required; and
5. the owner explicitly authorizes the next execution level.

Parking never deletes code, data, manifests, failures, or evidence.
