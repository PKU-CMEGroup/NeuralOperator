# Research Direction: Corrective Mechanisms for Neural-Operator Rollouts

Updated: 2026-10-05

## Current phase

The September 30 discussion with Daniel sets version 2's direction: a rigorous
comparison of corrective methods that explains their behavior through a common
framework and provides practical guidance. The owner manually revised and locked
the [paper plan](../../paper/PAPER_PLAN.md) on October 1; do not modify it without
explicit permission. Experiments follow owner review before advancing each
stage, starting with Section 4.2's mechanistic questions.
Comprehensive comparison planning follows later. The owner approved implementing
the first existing-pair study, producing pilot visualizations, and subsequently
training a reusable independent clean probe. The owner then authorized the
six-run recovery/dynamics comparison across Gaussian, native one-step and
mixed-length native displacements, queued behind the probe. This six-run batch
and its two-case rollout/probe evaluation are now complete and locally verified.
At the owner's request, interpretation follows completion of the full batch;
[HANDOFF.md](HANDOFF.md) records the next review boundary. After reviewing the
results and design weaknesses, the owner approved the October 2 controlled
follow-up in [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md), including paired target
diagnostics, training/sampling controls and the original twelve-case evaluation.
That follow-up is complete and analyzed on the original twelve-case population.
The handoff routes its report, figures and verification receipts. The findings
support useful distinctions between training inputs, target learning and rollout
outcomes; they do not establish universal rankings or exact manifold retention.
On October 4 the owner approved implementing and running the next first stage:
a REC/DYN midpoint and a controlled four-step comparison of clean, frozen,
detached and attached inputs. The experiment plan owns its fixed protocol;
the handoff owns execution status. This authorization includes offline response
and empirical-geometry measurements, not longer-horizon sweeps or new methods.

Version 2 is due before October 8, with October 7 as the completion day. This
supersedes older October 8/15 alternatives. Venue remains undecided. The paper
plan owns the proposed structure and open decisions. The manuscript preserves
the approved abstract and Sections 1--4.1. At the owner's request, Section 4.2
now includes the clean-training, REC/DYN, target-mixture and four-step Kolmogorov
comparisons with working figures and supporting appendix details. The remaining
mechanistic evidence is for discussion with the owner. Other planned revisions
are not source edits.

## Scientific question and contribution

Why can a neural operator with small clean one-step error have poor long-horizon
rollout, and which corrective choices help, under what conditions and at what cost?

Even when one-step prediction generalizes well on ground-truth states,
accumulated errors can drive predictions away from the training-data manifold,
where clean supervision alone does not control the learned dynamics.
Gronwall arguments remain valid; clean one-step error is not a uniform defect
bound on displaced inputs.

The intended contributions are a common analytical framework, controlled
explanations alongside fair method comparisons, and evidence-based practical
guidance. Offline diagnostics serve the scientific explanation; rollout
validation remains the practical model-selection procedure. A new method is
optional and belongs in Section 3 if justified. The old standalone Section 5
intervention/forecast study is removed from the plan; a final discussion
synthesizes the empirical findings.

Use Daniel's "minimal energy route": retain mathematics and motivation in
proportion to their role in understanding the final results. The framework must
make useful, testable distinctions. Do not claim priority for error accumulation,
distribution shift or response matching.

## Scope and scientific boundaries

- Autonomous rollout remains the setting. Kolmogorov flow is the main PDE study;
  a second PDE is planned but not selected. Preserve the dynamical-system
  illustration. Data quantity, model scale and distribution are proposed
  comparison axes whose protocols remain for experiment planning.
- Establish clean competence over the time window being interpreted. Continued
  optimization on fixed data and increasing data quantity are different tests.
  The former does not establish that data scaling cannot help.
- Accurate dynamics and appropriate handling of accumulated displacement both
  matter. Retention alone is not trajectory accuracy; neither explicit
  projection nor normal contraction is universally necessary.
- Departure and error accumulation can interact. Universal OOD causality,
  universal necessity of added correction and general inferiority of full
  multistep differentiation are not established conclusions.
- At the same displaced input x = u + eta, recovery targets Phi(u), while
  dynamics relabeling targets Phi(x). Neither target is universally superior.
- Geometry is defined independently of the tested model. Finite-bank distances,
  PCA directions and sampled responses are not exact manifold coordinates or
  uniform stability bounds.
- Independent clean/REC/DYN probes are candidate diagnostics; their errors do
  not automatically measure geometry or the generating model's behavior.
- Trusted solvers supply offline data, diagnosis and labels. No online solver,
  online OOD detector or broad benchmark is selected. "Plug and play" concerns
  limited problem-specific engineering, with access and training costs explicit.
- NACA and Bump studies remain historical candidates, not an automatic choice
  for the second PDE.

## Evidence, authority and preservation

The previous empirical programme is preserved as historical evidence rather
than the plan for the revised paper. Exact findings, qualifications and attempt
identities belong in [EXPERIMENT_TRACKER.md](EXPERIMENT_TRACKER.md);
[EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md) routes completed recipes.
Keep retrospective explanations, pre-outcome predictions and deployment
quantities distinct. Do not infer a universal ranking or sustained fidelity.

Current explicit owner direction and AGENTS.md govern; [HANDOFF.md](HANDOFF.md)
owns the checkpoint. Protected populations remain closed under their contracts.
Keep private context and unpublished material private; external AI review needs
explicit approval of the material sent. Preserve source and artifact identities.
The preceding working documents are retained at Git commit 669f7fc.
