# Research Direction: Corrective Mechanisms for Neural-Operator Rollouts

Updated: 2026-09-28

## Current phase

Consolidate the existing theory, literature and experiments into the first
mentor-facing version. The abstract and Sections 1--3 are approved and locked;
Section 4 is next. The [paper plan](../../paper/PAPER_PLAN.md) owns detailed
editing boundaries, figure choices and unresolved questions for Daniel.

No new experiment, intervention, evaluation population or expanded theory is
selected before the mentor discussion. Existing workstation access authorization
persists, but is not a current launch instruction. Venue remains under discussion;
the [delivery schedule](WEEKLY_RESEARCH_PLAN.md) owns manuscript targets.

## Scientific question and contribution

Why can a neural operator with small clean one-step error have poor long-horizon
rollout, and how can understanding this discrepancy guide corrective design?

Even when one-step prediction generalizes well on ground-truth states,
accumulated errors can drive predictions away from the training-data manifold,
where clean supervision alone does not control the learned dynamics.
Gronwall arguments remain valid; clean one-step error is not a uniform defect
bound on displaced inputs.

The paper develops three connected contributions:
1. A theoretical framework separating control of departure from the manifold
   and control of trajectory error.
2. A common interpretation of representative corrective mechanisms through
   their inputs, targets, deployed flow maps and information costs.
3. Quantitative diagnostics whose measured range of validity determines how
   they can inform intervention design.

The framework must provide useful distinctions, not claim priority for error
accumulation, distribution shift, response matching or local error forecasts.
A stronger method and its practical value remain topics for the mentor meeting.

## Scope and scientific boundaries

- Autonomous rollouts from in-distribution initial conditions; PCNO and the
  existing fixed-data Kolmogorov regime are the main empirical setting.
- The data-rich premise concerns adequate clean competence in the relevant
  time window. Preserve measured later-time gaps; do not reopen data scaling.
- Accurate dynamics and appropriate handling of accumulated displacement both
  matter. Retention alone is not trajectory accuracy; neither explicit
  projection nor normal contraction is universally necessary.
- At the same displaced input x = u + eta, recovery targets Phi(u), while
  dynamics relabeling targets Phi(x). Neither target is universally superior.
- Geometry is defined independently of the tested model. Finite-bank distances,
  PCA directions and sampled responses are not exact manifold coordinates or
  uniform stability bounds.
- Trusted solvers supply offline data, diagnosis and labels. No online solver,
  online OOD detector, exogenous-OOD programme or broad benchmark is assumed.
- ODE, NACA and Bump evidence enters only where it serves a distinct paper claim.

## Evidence, authority and preservation

The selected empirical programme is complete. Exact findings, qualifications
and attempt identities belong in [EXPERIMENT_TRACKER.md](EXPERIMENT_TRACKER.md);
[EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md) routes completed recipes.
Keep retrospective explanations, pre-outcome predictions and deployment
quantities distinct. Do not infer a universal ranking or sustained fidelity.

Current explicit owner direction and AGENTS.md govern; [HANDOFF.md](HANDOFF.md)
owns the checkpoint. Protected populations remain closed under their contracts.
Keep private context and unpublished material private; external AI review needs
explicit approval of the material sent. Preserve source and artifact identities.
The preceding working documents are retained at Git commit 669f7fc.
