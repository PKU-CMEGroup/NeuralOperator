# Project Plan: A Useful Framework for Corrective Mechanisms

Updated: 2026-09-28

## Objective and current phase

Produce a concise research paper explaining the clean-error/rollout discrepancy,
comparing corrective mechanisms under one framework, and assessing how
quantitative diagnostics can guide design. The [research decision](RESEARCH_DIRECTION_DECISION.md)
owns the thesis and scientific scope.

The selected experiments and independent confirmation are complete. The
September 24 draft contains the existing scientific material; the present task
is to consolidate it into the version for Daniel before October 1. This does not
declare the paper submission-ready. Venue and any stronger methodological
contribution remain open for the mentor discussion.

The abstract and Sections 1--3 are approved and locked. Section 4 is next.
The private [paper plan](../../paper/PAPER_PLAN.md) owns the exact editing
boundary, section structure, displays and mentor TODOs.

## Remaining work from existing material

| Stage | Purpose | Completion condition |
| --- | --- | --- |
| Section 4 | Show what the framework explains through controlled comparisons. | Each selected result answers a stated question; setup, finding and interpretation are clear. Preserve contrary outcomes and the distinction between matched contrasts and limited method adaptations. |
| Section 5 | Assess the framework's usefulness for intervention and prediction. | Separate observed protection against clean-only adaptation from qualified short-horizon forecasts; compare with simpler controls and retain failures and limits. |
| Conclusion and appendices | State the contribution at the supported scope. | Conclusions match the evidence; proofs, recipes, costs and supporting cases have clear locations without repeating the main argument. |
| Displays and manuscript checks | Make the existing argument readable and reviewable. | Use selected existing evidence, explicit placeholders where needed, consistent labels and terminology, and an appropriately checked PDF. |
| Mentor discussion | Decide how to strengthen the paper. | Agree on the engineering decision to improve, any additional method/evidence, venue and a realistic second-version scope. |

The main empirical story uses clean-continuation/recovery, matched recovery
versus dynamics targets, exposure versus temporal gradients, and the finite
range of acquired corrections. Existing ODEs provide known geometry; NACA and
Bump supply selective supporting evidence. Preserve the full method comparison
and its limitations without making every historical result a main-text result.

## Completion standard and open questions

For the first version, each contribution must have a clear argument and a
traceable supporting result. Separate theoretical sufficient conditions from
sampled diagnostics, and clean competence at onset from later-time behavior.
Keep essential unresolved scientific questions visible for Daniel rather than
filling them with conjectural results.

The main open question is practical usefulness: what decision does the
framework improve beyond simpler measurements? The current response-preserving
test uses an established penalty. Its value must rest on the insight and evidence,
without claiming invention of response distillation or forecasting training.
A stronger intervention and additional evidence are options for discussion,
not prerequisites silently added to the current writing phase.

## Ownership, schedule and authorization

- [PAPER_PLAN.md](../../paper/PAPER_PLAN.md): manuscript argument and editorial tasks.
- [WEEKLY_RESEARCH_PLAN.md](WEEKLY_RESEARCH_PLAN.md): dates and dependencies.
- [EXPERIMENT_TRACKER.md](EXPERIMENT_TRACKER.md): exact dated results and qualifications.
- [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md): completed recipes and original expectations.
- [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md): preservation and future change discipline.
- [HANDOFF.md](HANDOFF.md): current checkpoint.

No new fitting, experiment, evaluation population or theory programme is selected
before the mentor meeting. Existing workstation access persists without an
implicit launch instruction. Privacy, protected-population and publication
boundaries remain in force. The second version's scope and October 8/15 target
will be reassessed with Daniel; submission preparation follows the venue decision.
The preceding plan is retained at Git commit 669f7fc.
