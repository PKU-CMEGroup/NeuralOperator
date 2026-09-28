# Project Plan: Predictive Design of Corrective Mechanisms

Updated: 2026-09-24

Status: the authorized fixed-data corrective comparison is complete; the
September 17 [result ledger](EXPERIMENT_TRACKER.md) records its onset benefit and
long-horizon failure. The fixed recipe remains in [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md).
The six-section manuscript refactor and primary-source field study are complete.
The reached-state diagnostic and its Section 4 response/fidelity panel are now
complete, as are the selected representative-method comparison and the three
paired response-preserving design fits. Independent confirmation on 64 fresh
trajectories is complete with fixed data and models. Supporting-case methods,
variability/cost reporting and manuscript consolidation now complete the first
scientific draft. Remaining work is mentor revision and submission preparation. The
current fixed-data Kolmogorov case is selected for this work; conditional method
applicability and final submission scope remain explicit decisions.

## Objective and completion standard

Publication target: **Journal of Computational Physics**, selected by the owner
on September 19. The [paper plan](../../paper/PAPER_PLAN.md) owns the JCP reader's
argument, section/display budget and writing priorities. Venue selection changes
presentation, not the fixed-data scope or the evidence standard below.

Explain why small clean one-step error can coexist with poor autoregressive
PDE prediction, and turn that explanation into a useful design decision:

**diagnose -> predict quantitatively -> choose and intervene -> verify.**

The empirical completion standard is a substantive PDE case in which the
framework identifies a consequential response error, chooses a corrective
action, predicts a quantitative benefit and its limits before the corresponding
outcomes are inspected, and survives an independent test of that prediction.

A taxonomy and retrospective explanation alone do not complete this objective.
The empirical comparison must cover all representative literature methods,
with the roster, reusable evidence and missing comparisons made explicit in
[EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md). A single recovery improvement is not
the complete comparison.
A universal method ranking, a new architecture, or a theorem guaranteeing all
long-horizon predictions is not required. A new corrector can follow naturally
from the diagnosis; algorithmic novelty is a separate claim to check if pursued.

The theoretical framework is sufficiently developed. Improve its accessibility
and connect its existing identities and assumptions to measurable predictions;
do not expand the theorem programme without a specific paper-critical gap.

## Paper claims and decisive evidence

| Intended claim | Evidence needed | What would weaken or defeat it |
| --- | --- | --- |
| Clean accuracy leaves deployment-relevant response unspecified under a declared rich class. | Existing conditional theory, exact error identity, and a concise existing ODE illustration. | Overstating the class assumptions, equating finite data with a smooth manifold, or claiming every successful model needs explicit correction. |
| Corrective families change different parts of the prediction problem. | Common definitions of inputs, targets, complete deployed maps and information cost; measured signatures for the selected representatives. | Assigning a mechanism from its name or its rollout rank without measuring the intended response. |
| The framework guides a useful engineering choice. | A frozen quantitative prediction that selects an intervention, its direction/strength, or another application-relevant choice, followed by independent comparison with the forecast. | Post-outcome tuning, no advantage over a relevant simpler decision rule, or a predicted gain that damages required physical/path fidelity. |
| The guidance has an identifiable range of validity. | Amplitude/time/condition checks and one informative limit or contrast where the model predicts reduced benefit, harm, or abstention. | Hiding failures, claiming a local diagnostic is uniform, or interpreting natural physical growth as model error. |

Legacy C1 denotes the qualified theoretical claim. Legacy C2 proposed a
prospective ranking comparison beyond clean error; that remains unestablished.
The present quantitative design goal may use a narrower decision than a full
ranking, such as forecast horizon or correction strength. Success on that
decision must not be relabelled as completion of the old C2 protocol.

## Scope and standing boundaries

- Fix the clean dataset in a regime with enough data to identify nearby
  reference behavior. Severe clean one-step overfitting is excluded as the
  intended mechanism; check competence without launching a data-scaling study.
- Autonomous prediction from in-distribution initial conditions, with PCNO as
  the principal empirical backbone and a declared complete numerical state.
- Trusted solvers supply reference data and offline diagnosis/relabeling.
  NACA uses SU2; current Kolmogorov code uses a dedicated spectral solver.
  The deployed surrogate does not assume online solver access.
- At least one carefully selected substantive PDE design study. ODEs remain
  supporting illustrations. Existing cases enter the paper only when they
  establish a distinct point.
- Representative mechanisms: clean training, recovery/noise, faithful
  prefix/multistep exposure, dynamics relabeling, empirical projection and
  learned explicit refinement, plus distinct representative methods identified
  by the literature review. Cover them empirically; a taxonomy alone is
  insufficient. Not every method must run on every PDE, but omissions from a
  common-case comparison need a scientific rationale, not an implicit scope cut.
- Reference law and any claimed geometry are specified independently of the
  tested model. Solver defect and a model's linearization-validity region do not
  define ID/OOD or a data manifold. A learned model may represent the continuous
  reference family between stored trajectories. Large finite-bank distance does
  not establish distance from that family; geometric resolution is a diagnostic
  obligation, not a reason to reopen data scaling.
- Preserve privacy, immutable attempts and protected-population boundaries.
  No online OOD detector, exogenous-OOD programme, broad architecture/PDE sweep,
  new rental, protected reveal, or automatic revival of a parked line.

## Evidence available for planning

The [dated result tracker](EXPERIMENT_TRACKER.md) owns exact metrics, identities,
hashes and historical numerical audits. The September 15--16 context inspection
verified source/manifest bindings and selected result files; it did not rerun
the scientific analyses. Do not describe that integrity check as independent
numerical recomputation.

| Material | Reusable evidence | Limitation and present role |
| --- | --- | --- |
| Theory and ODEs | Conditional trace ambiguity, distinct target semantics, retention versus path accuracy, existing figures. | Supporting calibration; no new ODE work without a specific essential gap. |
| NACA0012 | Strong path projection/refinement and clean-error/rollout discordance; existing paired solver labels. | Favorable nearly linear path; trained paired maps were not scored against both targets on the common bank. Inclusion is selective. |
| Supersonic Bump | Different intervention rankings, typical-versus-tail tradeoffs, physical failures despite finite rollouts. | One-seed pilot, unlike projector/metric contracts, no qualified arbitrary-state target solver. Do not infer geometry alone causes the contrast. |
| Kolmogorov | Fixed-data onset reversal, common-input response assays, matched targets, representative corrections and the response-preserving design with 64-start confirmation. | Selected main case. The finite-horizon design/prediction loop is complete; later clean gaps, local forecast qualification, poor H128 fidelity and one-parent scope remain explicit. |
| Historical solver-bank adaptation | Three-arm fitter, matched samplers/control, implementation tests and train-only validation. | Available implementation, not a scientific result or the selected fixed-data study. Its September 12 resource payload had no local launch receipt at the inspected checkpoint. |

Archived numerical outcomes are not changed by this plan. NACA R0's failed
promotion, Bump's combined-rule failures and incomplete M1 attempts retain
their original meanings; neither failure nor successful infrastructure alone
decides the new scientific case.

## Diagnostic Instruments, Not A Separate Forecasting Programme

For the complete deployed transition, along a declared clean reference path,

$$
b_n=\Psi_n(u_n)-\Phi_n(u_n),\qquad J_n=D\Psi_n(u_n),\qquad
e_{n+1}=b_n+J_ne_n+r_n(e_n).
$$

A candidate local forecast is
$\widetilde e_{n+1}=b_n+J_n\widetilde e_n$, starting from
$\widetilde e_0=0$. The completed calibration found a short validity window.
Retain this as supporting evidence about finite-amplitude response; do not make
improving this forecast a prerequisite for comparing corrective mechanisms.
Trusted response measurements can distinguish physical dynamics from response
error when that distinction is needed for the selected claim.

A credible forecast retains forcing direction, phase/coupling and time-ordered
propagation. Individual gains or eigenvalues do not characterize changing,
potentially nonnormal products. If a reduced basis is used, check omitted-space
leakage and return; if linearization fails, retain finite-amplitude response
and symmetric/even components rather than claiming a Jacobian explains them.
Sampled checks support empirical validity, not an unproved uniform bound.

Measure or fit the response model on declared calibration inputs, freeze it,
then predict without querying the complete learned map along every newly
predicted state. Such recursive exact-map queries would reproduce a rollout,
not supply an independent predictive diagnostic.

The spectral-audit inspiration is Gao, Yang and Karniadakis,
[Spectral Audit of In-Context Operator Networks](https://arxiv.org/abs/2606.02427),
v1, 2026-06-01, Sections 3.2--6 and Appendix C. It compares local gains, phase and
coupling against a reference tangent map; it does not establish autoregressive
correction selection. Fourier directions are not manifold normals, directional
gains are not an eigenvalue spectrum, and valid sine/cosine phase rotations
must not be counted as spurious cross-frequency mixing.

The engineering objective is finite-horizon application error with required
physical/path fidelity, not contraction alone. A correction can change both
forcing and response. Selective damping and a relabeling/recovery buffer are
illustrations to assess after diagnosis; neither is a selected algorithm.
Any new method must have a distinct predicted advantage and an identity/control
comparison, with training, inference and trusted-label costs reported.

## Main case and its validity limits

Keep the existing Kolmogorov parameters, initial law, full clean dataset and
start-0 evaluation. Its adequately learned onset already supplies the intended
discrepancy; changing the regime to obtain a preferred ranking would weaken the
comparison. Reconsider the case only if a paper-critical validity problem is
demonstrated, using these criteria:

1. Small, comparable clean training/development errors in relevant time windows;
   ordinary overfitting must not dominate the intended discrepancy.
2. A complete restartable state and numerically resolved responses at the actual
   states, directions and amplitudes that support the mechanism claim.
3. Meaningful dynamics to preserve as well as an error to correct. Finite but
   inaccurate rollout is sufficient; blow-up and a preferred method winner
   are not admission criteria.
4. A quantitative observable that matters to use: error growth over a declared
   interval, a tolerance-crossing time/range, or intervention-strength effect.
5. An informative physics-motivated contrast and affordable qualification,
   training, diagnosis and independent confirmation.

The completed comparison held the full 32-path clean dataset and parent fixed.
Clean continuation controlled for additional optimization; recovery used perturbed
inputs with their exact archived clean successors. Initial-time evaluation and
the later transient gap were retained. Preserve that distinction in the paper;
do not tune a regime until a preferred result appears.

The historical solver-bank adaptation is not selected: its directions differ
from endogenous errors and its anchors begin at input step 15. The completed
online recovery comparison sampled every clean time and reused exact archived
labels. Its acquisition/transfer assays preceded autonomous evaluation; their
early benefit and late exceptions remain part of the evidence.

## Work packages, deliverables and decisions

| Order | Scientific work | Concrete deliverable / completion check |
| --- | --- | --- |
| Completed foundation | Six-section manuscript and source survey; fixed-data eligibility and paired clean/recovery experiment. | Working draft, verified evidence bindings and explicit red gaps. These development results are not independent confirmation. |
| 1. Completed diagnosis | Frozen-map replay and common-amplitude responses; independent physical-family envelope. | Section 4 panel and dated evidence identify finite useful range, directional exceptions and clean-forcing cost, with late family exclusion. This retrospective result does not complete the quantitative prediction. |
| 2. Completed selected mechanism comparisons | Target/gradient contrasts, conditional refinement/generation, shell prior, channel-matrix restriction and physical cap are complete. Retain the fixed-data stationary-law scope decision, NACA projection support and component-adaptation limits. | Sections 3--4 have measured mechanisms, unfavorable outcomes, fidelity and separate information/training/deployment costs. This closes the selected comparison block, not independent confirmation or optimal tuning of every family. |
| 3. Completed bounded design test and fitting repeats | Three paired fits retain response and improve H32 over clean-only; onset benefit varies and H128 fidelity remains poor. Qualified forecasts track both improvement and harm. | Section 5 reports every seed, failed clean-only forecast checks and the parent-relative decision's limited horizon. Fitting repeats share one parent. |
| 4. Completed independent confirmation | All nine fixed maps evaluated on the owner-selected 64 independent trajectories; forecasts and controls frozen before adapted outcomes. | H32 protection transfers to every paired path; qualified H8 forecasts agree within 0.25% relative discrepancy. Trajectory-level uncertainty and guarded outcomes are retained. |
| 5. First scientific draft complete; revision next | Supporting-case methods, seed/case variability, cost reporting, compact Section 4 and checked source/figure package are complete. | No unresolved essential evidence TODOs. Review with the mentor, refine presentation and finish author/funding/availability and submission material. No best-seed selection, refitting or expanded coverage. |

[EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md) is the single source for the roster,
qualitative expectations, common inputs/targets/gradients and staged order.
Coverage is at the level of representative corrective operations, with distinct
conditional families kept visible. There is no full method-by-PDE-by-strength
matrix and no automatic return to the old solver bank.

The reached-state diagnostic and matched gradient/target contrasts are complete.
They identify a concrete tradeoff: broader response control can keep trajectories
finite and improve later error while increasing clean bias and early path error.
The conditional Refiner/ACDM, shell-prior, channel-cap and physical-cap
comparisons are complete, with their clean-error and physical-fidelity costs.
September 22: the owner selected a matched clean-only versus response-preserving
adaptation of bank DYN. It asks whether reducing clean forcing while retaining
the useful displaced-input response improves composition. The experiment plan
owns the recipe and pre-outcome H8 prediction; broader rollout/physical limits
remain measured. The buffer alternative and failed blend/switch checks are not
active work. The completed three-seed comparison supplies repeated conditional
quantitative onset predictions, with a consistent H32 advantage over clean-only
and poor H128 fidelity. The 64-trajectory confirmation establishes transfer of
the H32 contrast and conditional onset predictions, with some H128 guards and
no uniform parent-relative ranking. HANDOFF.md owns the current checkpoint.

The paper's display sequence is phenomenon; representative mechanisms and
measured response/bias/cost; predicted versus observed intervention; fidelity
and limits. Use one compact ODE illustration and NACA's favorable geometry
contrast. Bump is supporting only if it adds a distinct, qualified point.
The [paper plan](../../paper/PAPER_PLAN.md) owns layout.

## Prediction, confirmation and failure policy

- Separate retrospective explanation, predictions frozen before new outcomes,
  and quantities available during deployment. Offline reference access is
  legitimate for diagnosis but is not a deployed capability.
- Freeze the observable, prediction/uncertainty, decision rule, tolerances,
  information inputs and confirmation scope before the relevant outcome.
  Compare with clean-error-only or another appropriate simpler decision rule.
- Reused development paths with new models can test a new pre-outcome forecast;
  they do not establish fresh-population generalization. Use independent
  confirmation at the scope of the final claim. Existing sealed roles stay closed.
- Match inputs and budgets for target comparisons; measure whether the model
  acquired the intended target before interpreting rollout. Distinguish model
  seed variation from trajectory variation and frames from independent samples.
- Negative results remain useful and visible. An unclosed predictive design
  loop is not called achieved; revise the explanation, scope or completion
  claim with the owner instead of quietly lowering the standard.
- Replicate the central benefit/limit proportionately to the claim. Neither a
  fixed three-seed matrix nor a single parent is an automatic adequacy rule.

## Schedule and effort

The owner requests a rough draft by the start of October and a carefully
polished version by October 8, with weekly mentor feedback alongside writing.
The working dates, measured compute costs and scientific dependencies are
maintained in [WEEKLY_RESEARCH_PLAN.md](WEEKLY_RESEARCH_PLAN.md). These are
manuscript targets, not a venue submission deadline. JCP is selected; the agreed
confirmation is complete. Supporting-case reproducibility and clear synthesis
are the remaining delivery dependencies; historical protected roles stay closed.
The September 6--13 dates, old method cutoffs and 72-hour envelope remain
superseded. Neither the calendar
nor a completed training job lowers the paper's evidence standard.

## Documentation and execution discipline

This file owns scope, work packages and completion criteria.
[EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md) owns the scientific comparison;
[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) owns minimal implementation
and proportionate verification; [HANDOFF.md](HANDOFF.md) owns current status.
The tracker owns dated evidence. Avoid duplicating numerical histories.

The current owner instruction requests retrieval, careful analysis and next
steps tied to the paper. Prior authorization for coding, focused checks and
workstation use persists; no repeated resource permission is needed. The initial
manuscript-driven refinement preceded the now-completed causal comparisons.
Historical approvals do not resume the old queue after this reset. Preserve
source-bound recipes and use new identities for changed attempts. No additional
review round, generic framework or infrastructure refactor is required merely
because a new plan exists.

The [pre-reset snapshot](archive/pre_predictive_reset_20260916a/INDEX.md)
preserves the exact preceding working-tree documentation, including uncommitted
changes. Frozen preregistrations, results, private context and manuscript source
are untouched by this planning reset.
