# Research Direction: Corrective Mechanisms for Predictive Design

Updated: 2026-09-24

Status: current owner-approved scientific direction. The owner has authorized
planning, implementation and execution of the fixed-data corrective study in
[EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md), with full access to the owner-selected
workstation configured in ignored `LOCAL_CONTEXT.md`. No repeated resource
permission is needed. Privacy and protected-population boundaries persist.

## Current decision

September 23: the owner selects **64 independent evaluation trajectories**
for confirmation, with the existing training data and fitted models fixed.
The nine-map comparison, forecast rules and outcome order in Experiment Plan
Block 4 remain unchanged. This authorizes fresh evaluation-only ID starts;
historical protected populations remain closed. No new fitting is selected.

September 22: the owner closes representative-method coverage and selects a
matched response-preserving clean-bias reduction test for Section 5. Adapt the
same bank-DYN terminal with clean-only fitting versus clean fitting plus a
penalty on changes to its finite displaced-input responses. The experiment plan
owns the fixed recipe, pre-outcome forecast and stopping point. No new method
roster, buffer-target sweep or protected-population opening is selected.

September 21 owner direction: **finish representative-method coverage first**.
After those comparisons, discuss intervention design and quantitative prediction
with the owner before selecting that work. This supersedes instructions to
overlap design with the remaining baseline comparisons. The fixed-data scope
and eventual paper ambition remain unchanged. The experiment plan owns the
shell-prior, architectural restriction, physical-cap and stationary-data decisions.

September 20: keep the paper plan central and allow creative mechanism
hypotheses. Exploratory comparisons do not require a successful local quantitative
forecast before their rollouts can be informative. Keep prospective prediction,
retrospective interpretation and hypotheses distinct when writing the paper;
this distinction should clarify results, not prevent useful experiments.
The planned ACDM-style and fixed-terminal Refiner comparisons on fixed data
are complete; the handoff owns their distinct response/accuracy tradeoffs
and remaining coverage/design decisions.

September 19: the owner selects **Journal of Computational Physics** as the
publication target. Retain the six-section argument and fixed-data main PDE.
The paper plan owns JCP presentation priorities; the delivery schedule records
the early-October rough draft, October 8 polished draft and weekly mentor feedback.
This decision does not require more theory, a new architecture or a larger PDE
benchmark, and does not change the quantitative design/confirmation obligation.

September 17 owner direction: resume work and refine the plan after completing
the approved six-section manuscript refactor and primary-source field study.
The draft retains missing evidence as red TODOs; its predecessor is preserved.
The refined sequence is a bounded reached-state diagnostic, staged mechanism
contrasts, one quantitative intervention prediction and independent confirmation.
Keep the present fixed-data Kolmogorov case; do not reopen regime/data selection
without a demonstrated paper-critical reason. The paper must compare
all representative corrective methods identified in the literature review.
The focused recovery diagnostic and one framework-guided design change are
parts of that comparison, not replacements for it. The method-coverage table
in [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md) owns representatives and remaining
evidence; the [paper plan](../../paper/PAPER_PLAN.md) owns manuscript structure,
claim/figure gaps and the linked primary-source literature ledger. This planning
refinement does not close the quantitative prediction or method comparisons.

Build a framework that closes the loop from a one-step/rollout discrepancy to
a quantitative prediction and an effective application-facing intervention:

**diagnose -> predict quantitatively -> choose and intervene -> verify.**

A useful paper explains mechanisms; the intended standard here also predicts
a measurable benefit and its limits before the corresponding outcomes are
inspected. A universal intervention ranking is unnecessary. A new corrector
may follow from the framework, but no specific algorithm is selected yet.

The [project plan](PROJECT_PLAN.md) owns scope, deliverables and completion
criteria. Its work packages replace the previous experiment queue. The owner's
current manuscript targets and weekly mentor-feedback cadence are recorded in
[the delivery schedule](WEEKLY_RESEARCH_PLAN.md); no venue submission deadline
or new compute budget is agreed. Older dates and prospective ranking milestones
are historical planning, not immutable requirements.

## Scientific commitments

The owner explicitly fixes the data and assumes a data-rich regime sufficient
to identify behavior near the reference manifold. Severe clean one-step
overfitting is not the explanation we seek. Hold the clean dataset fixed across
interventions and check clean training/development competence by relevant time
window. Do not turn that eligibility check into a data-scaling programme.
Finite approximation error, its propagation, recovery bias and preservation of
meaningful dynamics remain consequential even when nearby behavior is identifiable.

The complete deployed transition is $\Psi_n$, including numerical restrictions,
history, correction and stochastic inputs where applicable. When a fixed
predictor/corrector interface exists, $\Psi_n=C_{n+1}\circ F_n$; this
factorization is nonunique and is not required for every embedded mechanism.

Gronwall bounds are valid under their premises. Small clean-law defect does
not by itself provide a defect or response bound throughout the states reached
by self-composition. Conditional clean-trace non-identifiability remains a
theoretical boundary statement; it is not the empirical premise that our fixed,
rich dataset cannot identify nearby dynamics.

Keep two obligations separate: accurate on-reference/path dynamics and
appropriate treatment of accumulated displacement. Retention alone is not
accuracy; faithful displaced dynamics can propagate an unwanted displacement.
Normal contraction, explicit projection and a separate correction network are
not universally necessary. Clean one-step error is not a tangent Jacobian
measurement, and OOD generalization is not mathematically impossible.

Reference laws and any claimed geometry are defined independently of the tested
model. A finite dataset, smooth manifold and physical admissible set are distinct.
Model defect, Fourier bands and empirical PCA do not define manifold normals
or ID/OOD. The manifold/tube picture remains qualified intuition.

Recovery at $x=u+\eta$ targets $\Phi(u)$; dynamics relabeling targets $\Phi(x)$.
Compare identical inputs, target acquisition, clean forcing, trusted/learned
response, path error and information cost. Neither target is universally better.
A method is not declared a hidden corrector because its rollout succeeded.

## Predictive framework and case selection

The existing theory is sufficiently developed. The remaining framework work
connects its identities to measurable, falsifiable predictions; further theorem
generalization is not a default task.

Local response measurements, motivated in part by the spectral audit of Gao,
Yang and Karniadakis, are instruments for comparing corrections. The completed
tangent pilots demonstrated finite-amplitude limits; improving a baseline error
forecaster is not a separate project or prerequisite. Predict the effect of an
intervention on useful rollout accuracy, preserving forcing, phase and physical
tradeoffs. A sampled gain is not a stability bound.

Kolmogorov is the leading mentor-supported PDE family. Its exact regime,
population, horizon and pending paired adaptation are not the final paper plan.
Select the numerical case carefully: clean competence, trusted response,
meaningful dynamics to preserve, an informative tradeoff and an affordable
independent test. Do not require blow-up or engineer a desired winner.

Existing ODE, NACA and Bump material is retained evidence. Include only what
advances a distinct paper claim; move to an appendix or omit otherwise. Do not
reopen ODE work or rerun all corrective families on all cases by default.

## Scope

- Autonomous rollouts from in-distribution initial conditions; PCNO is the main
  empirical backbone.
- At least one substantive PDE study that demonstrates quantitative guidance
  and intervention verification, supported by concise interpretable examples.
- Trusted solvers for data and offline diagnostics/relabeling: SU2 for NACA,
  the dedicated spectral reference for current Kolmogorov work.
- Representative training, structural and deployed corrective mechanisms,
  with all representative literature methods covered in the empirical study
  and compared by their actual information, inputs, targets, maps and costs.
  Establish the named roster from primary sources before selecting missing
  runs; do not reduce the paper to recovery versus clean continuation.
- No assumed online solver, online OOD detector, exogenous-OOD programme,
  architecture zoo, broad PDE campaign or automatic protected-data reveal.

## Evidence and claim boundaries

Current results and their qualifications are in the
[experiment tracker](EXPERIMENT_TRACKER.md), with historical routing in the
[experiment index](MECHANISTIC_DIAGNOSTIC_TRACKER.md). Clean32's matched rollout
reversal, common-response pilot and paired bank support planning; they do not
establish the new quantitative design loop. Prepared adaptation is not training
evidence. The refactored manuscript now includes the fixed-data Kolmogorov
comparison and selective ODE/NACA/Bump support. Selected mechanism coverage is
complete. The response-preserving test and September 23 independent 64-trajectory
confirmation establish H32 protection against clean-only adaptation and qualified
prospective H8 magnitude predictions. A universal ranking and sustained-fidelity
algorithm are not established. The handoff owns exact status.

Legacy C1 remains conditional clean-trace non-identifiability. Legacy C2's
prospective ranking claim is unestablished. A new narrower prediction must be
named and evaluated at its own scope, not used to retroactively validate C2.
Descriptive explanations, genuinely pre-outcome predictions and deployment-time
quantities remain distinct.

Preserve failed scientific qualifications and incomplete infrastructure attempts
under their original identities. No historical result, gate or proposed
continuation is silently rewritten by this decision.

## Authority, preservation and next action

Authority order: current explicit human direction and repository AGENTS.md;
this decision; [HANDOFF.md](HANDOFF.md) for current status; compact experiment
routing; [README.md](README.md) for maintained code and retention navigation.
Planning documents do not reopen historical launch or protected-access scopes.

The matched clean-continuation/online-recovery comparison is complete; its fixed
recipe remains in the experiment plan and the September 17 evidence is in the
dated tracker. Frozen-map replay and common-amplitude diagnosis are also complete,
with the response/fidelity panel incorporated in Section 4. The same-input target
and matched-gradient contrasts are complete. The fixed Refiner acquisition and
blend/switch forecast checks are closed with their limitations retained. The
owner-selected response-preserving test is now complete at its declared scope.
The September 22 continuation's two additional paired fitting seeds are complete
under the unchanged recipe, with forecasts frozen before autonomous outcomes.
Response retention and H32 protection against clean-only adaptation replicate;
clean forcing and H8 benefit vary. The owner-selected 64-trajectory confirmation
is complete with fixed training data and fitted maps. Qualified forecasts track
both benefit and harm; H32 protection transfers to every matched confirmation
path, while H128 fidelity remains poor and some maps reach the amplitude guard.
This separates optimization-order sensitivity on one parent from independent
initial-condition confirmation without selecting a best seed. The September 24
writing pass completes the scientific draft and supporting-case methods. Next
incorporate mentor feedback and prepare submission material. No further fitting or
historical protected reveal is selected. Confirmation outcomes remain separately
identified from development work.
[HANDOFF.md](HANDOFF.md) owns the current checkpoint.
The old Clean8/Clean32 proposal and solver-bank launch queue remain historical.
The refined scientific questions, not prepared payloads, select the new work.

Private context, unpublished manuscript, datasets and credentials remain private.
No external AI review receives private material without exact scope approval.
Preserve other agents' work, scientific artifacts, source snapshots and recovery
copies. Sealed/prospective/test roles remain closed until their named decisions.

The [pre-reset archive](archive/pre_predictive_reset_20260916a/INDEX.md)
preserves exact previous document bytes and their historical instructions.
The [theory source](CORRECTIVE_MECHANISMS_THEORY_AND_TAXONOMY.md) retains its
qualified statements; the ignored [paper plan](../../paper/PAPER_PLAN.md) owns
narrative and figure selection. Neither source is rewritten into new results.
