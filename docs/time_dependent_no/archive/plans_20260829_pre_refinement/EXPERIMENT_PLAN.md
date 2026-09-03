# Experiment Plan: From Response Identification To Prospective Rollout Prediction

Updated: 2026-08-29

Status: active claim-driven design. No block is authorized to execute by this
document. Each scientific run requires a block-specific preregistration,
identity, resource choice, and owner approval.

## 1. Research Questions

The empirical programme tests two questions:

1. Can one-step-matched learned maps have predictably different rollouts because
   their forcing and response away from the clean trace differ?
2. Can the framework predict, before long-rollout evaluation, which embedded,
   structural, or operational corrective mechanism will be useful for a fixed
   neural-operator backbone?

These questions correspond to C1 and C2 in
[RESEARCH_DIRECTION_DECISION.md](RESEARCH_DIRECTION_DECISION.md). The experiment
does not evaluate exogenous OOD generalization or deploy an OOD detector.

## 2. Bounded Design

There are four required blocks and one conditional transfer block:

| Block | Role | Status at plan freeze |
| --- | --- | --- |
| B1: exact ODE laboratory | Establish exact counterexamples, response geometry, and diagnostic identities. | Required supporting block; planned, not authorized. |
| B2: learned ODE intervention sandbox | Test whether representative interventions produce their expected signatures in a cheap setting with exact geometry. | Required supporting block; planned, not authorized. |
| B3: primary complex-PDE diagnosis and prediction freeze | Diagnose one qualified PDE, then freeze problem-specific intervention predictions before designated outcomes are opened. | Required main-paper case study; PDE and methods not yet selected. |
| B4: primary complex-PDE intervention study and reveal | Test representative embedded, structural, and explicit mechanisms under the B3 freeze; promote the hybrid only if its contract closes. | Required main-paper evidence; blocked by B3 and implementation audits. |
| B5: secondary-PDE transfer | Test a reduced comparison on a contrasting qualified PDE. | Conditional; must not delay the first serious submission draft. |

The core paper fixes PCNO and does not add architectures, PDEs, or method slots
after seeing designated outcomes. The exact primary PDE and the implementation
used for each representative slot are decisions for the next planning stage.

## 3. Representative Mechanism Slots

The literature study is broad by mechanism class. The empirical study uses one
preselected implementation per slot rather than reproducing every paper:

1. **clean control:** CLEAN;
2. **embedded recovery:** noise or recovery training;
3. **endogenous exposure:** pushforward or multistep training;
4. **trusted response:** dynamics-consistent relabeling;
5. **structural control:** one constraint or invariant appropriate to the PDE;
6. **deterministic operational correction:** one projection-style method;
7. **learned operational correction:** one sampler or refinement method; and
8. **hybrid:** faithful response in an inner buffer with corrective return
   outside it.

B2 uses compact analogues of all slots, including the hybrid. B4 must contain
representative embedded and operational correctors; it includes the hybrid only
if B3 closes its target, composition, cost, and audit contract. Operational arms
may share a trained predictor; they are not automatically separate training
runs.

### 3.1 Qualitative predictions to freeze

These are framework-level expectations, not universal rankings. B3 must turn
them into signed, problem-specific predictions before the B4 reveal.

| Mechanism | Expected diagnostic signature | Main failure mode |
| --- | --- | --- |
| Recovery/noise | Lower transverse forcing or gain; longer tube residence. | Tangential damping or suppression of legitimate response. |
| Pushforward/multistep | Lower defect along model-generated error directions. | No generic normal contraction or horizon-specific overfitting. |
| Dynamics relabeling | Lower trusted off-state response defect; wider accurate region. | Faithfully propagates an expansive displacement. |
| Structural control | Lower violation of the encoded invariant or constraint. | Physical validity improves without path accuracy. |
| Deterministic projection | Immediate reduction in post-correction proximity or constraint residual. | Quantization, phase snapping, or a stable but wrong rollout. |
| Learned sampler/refiner | Better frozen distributional-proximity score. | Smoothing, mode switching, conditional inconsistency, or high cost. |
| Hybrid | Faithful response inside an inner buffer and corrective return outside it. | Conflicting targets, poor gating, or erased valid modes. |

For every slot, report clean accuracy, the declared mediator, long-rollout
accuracy, physical no-harm, and information/compute cost. A rollout gain without
the predicted mediator change rejects the proposed explanation.

## 4. B1: Exact ODE Laboratory

### 4.1 Reference flow

Use tangent-normal coordinates `(theta,r)` with in-distribution reference set

\[
\mathcal M=\{(\theta,r):r=0\}
\]

and ODE

\[
\dot r=\kappa r,
\qquad
\dot\theta=\omega+c r.
\]

For step size `h` and `kappa != 0`, the exact flow is

\[
r^+=a_*r,
\qquad
\theta^+=\theta+\omega h+b_*r,
\]

where

\[
a_*=e^{\kappa h},
\qquad
b_*=c\frac{e^{\kappa h}-1}{\kappa}.
\]

For `kappa=0`, use `a_*=1` and `b_*=ch`. This system gives an exact normal
response and an exact normal-to-tangent coupling.

Embed the periodic coordinate as `(cos(theta),sin(theta),r)` when comparing
geometry proxies. The clean reference set is then a curved circle rather than
an affine line.

### 4.2 Learned/deployed map family

Use the transparent affine local model

\[
\widehat r^+=\epsilon_N+a\widehat r,
\qquad
\widehat\theta^+=\widehat\theta+\omega h+\epsilon_T+b\widehat r.
\]

On the clean set, one-step error depends on `epsilon_N` and `epsilon_T`; the
deployment response depends additionally on `a` and `b`. This makes exact
smaller-one-step/worse-rollout constructions possible.

### 4.3 Predeclared scenarios

| Scenario | Construction | Required observation |
| --- | --- | --- |
| S1: ranking reversal | Model A has smaller clean forcing but `abs(a_A)>1`; Model B has larger clean forcing but `abs(a_B)<1`. | A wins H1 and loses at a predicted finite horizon. |
| S2: on-manifold wrong phase | Normal error remains zero while `epsilon_T` or tangent response is wrong. | Tube distance is zero but path error grows. |
| S3: bounded false attractor | `abs(a)<1` with nonzero `epsilon_N`, producing a bounded displaced fixed point. | Boundedness does not imply controlled reference tracking. |
| S4: recovery/relabeling crossover | Vary `a_*`, `b_*`, forcing, perturbation amplitude, and horizon. | Recovery and relabeling each win in at least one preregistered regime. |
| S5: partial shrinkage | Interpolate response target `A_lambda=lambda A_*`. | A conditional optimum may lie between recovery and full relabeling; appendix unless decisive. |

The crossover must include a regime where recovery wins despite substantial
trusted normal response. This tests the user's safety hypothesis without
presupposing it is universal.

### 4.4 Exact endpoints

- clean one-step error;
- exact tangent and normal forcing;
- exact `TT`, `TN`, `NT`, and `NN` response blocks;
- normal distance, tube-exit time, and survival;
- phase/path error;
- finite-horizon error predicted from the exact recurrence;
- recovery-target and dynamics-relabeling-target errors; and
- predicted versus observed intervention ranking.

### 4.5 B1 gate

The closed-form recurrence and numerical implementation must agree to a
precision-derived tolerance on every scenario and parameter cell. If a
predeclared ranking or boundary case fails, stop and revise the framework before
using it to motivate the learned/PDE experiments.

B1 establishes mechanism possibilities and calibrates measurements. It does
not determine which intervention will win on a PDE.

## 5. B2: Learned ODE Intervention Sandbox

### 5.1 Learner and data

- Use one residual MLP flow map on `(cos(theta),sin(theta),r)`.
- Use the same layer widths, initialization policy, optimizer, data order,
  update count, checkpoint rule, and three paired seeds for every arm.
- Generate clean trajectories only from `r=0` for CLEAN.
- Freeze one perturbation bank with separate tangent, normal, and mixed
  directions before training the response arms.
- Use trajectory-disjoint train, calibration, development, and prospective
  evaluation splits.
- Do not tune the network architecture after inspecting long-rollout results.

### 5.2 Experimental slots

| Slot | Defining contract |
| --- | --- |
| CLEAN | Clean `u_n -> Phi(u_n)` supervision. |
| EXPOSURE | One pushforward or multistep analogue using model-generated prefixes. |
| RECOVERY | Common displaced `x=u+eta -> Phi(u)`. |
| DYN-RELABEL | The identical `x=u+eta -> Phi(x)`. |
| STRUCTURE | One simple exact invariant or constraint appropriate to the ODE. |
| EXPLICIT-PROJECTION | A fixed post-step projection attached to a shared predictor. |
| LEARNED-REFINER | A learned post-step map or kernel attached to a shared predictor. |
| HYBRID | A relabel-trained predictor plus a corrector that is near identity in an inner buffer and corrective outside it. |

The exact exposure, projection, refiner, and hybrid implementations are not
fixed by this document. They must be selected before implementation and bound
to an intent contract. RECOVERY and DYN-RELABEL use exactly the same input bank.
Operational correctors are evaluated both as raw `F` and deployed `C o F`.
Oracle calls, stored data, and inference cost are reported separately.

### 5.3 Diagnostics

Primary geometry is exact. Compare it with:

- time-conditioned normalized kNN distance;
- local-PCA reconstruction residual, called a normal residual only after a
  stable local dimension and spectral-gap check; and
- convex-hull membership as a deliberately coarse negative proxy.

The circle's convex hull contains the entire disk, so convex-hull membership
cannot be treated as manifold distance.

### 5.4 B2 success and failure

B2 is a supporting sandbox, not the prospective C2 test. Its shared bank,
learner contract, and analysis provide a controlled comparison, but only B4
may support the outcome-blind prospective ranking claim.

Success requires:

- the retained intervention slots move their declared mediators in the
  predicted direction;
- recovery reduces transverse retention/forcing without unacceptable phase
  harm in its predicted regimes;
- relabeling reduces trusted response defect in its predicted regimes;
- the predeclared exploratory framework score ranks the learned rollouts better
  than clean H1;
  and
- results are reported over all three paired seeds.

If exact B1 predictions hold but the learned arms do not realize them, report a
learning/optimization failure separately from a theory failure. Do not add a
new architecture to repair the story. Dense ODE coverage may prevent meaningful
drift; report that boundary rather than engineering a desired failure. No ODE
ranking is transferred to the PDE without a separate B3 prediction and B4 test.

## 6. B3: Primary Complex-PDE Diagnosis And Prediction Freeze

### 6.1 Role and backbone

B3 is the mandatory end-to-end case-study diagnosis. Select one reasonably
complex, restartable PDE after owner review. Use one fixed PCNO backbone for
every learned arm. The theory remains map-agnostic, but the empirical comparison
avoids architecture confounding.

The existing M1 preregistration fixed a residual FNO for its later model stage.
That model contract is not silently continued under this plan. Its verified
Kolmogorov reference packets may be reused only as parent evidence in a new
PCNO identity whose model mesh, geometry, memory envelope, source set, and
evaluation contract are explicitly rebound.

### 6.2 PDE readiness gate

Before model work, the selected PDE must have:

- a qualified trusted transition at the model's state representation;
- a fresh-population or split contract with immutable membership;
- arbitrary-state restart and boundary/forcing closure;
- temporal/spatial error and solver bias below the model/arm effects to be
  interpreted;
- a model-mesh and PCNO memory/runtime envelope;
- open development populations and sealed test separation; and
- exact source/data/result manifest conventions.

Current M1 state does not pass this gate. The spatial and temporal N256
reference packets qualify only the frozen finite-grid reference. R2-POP launch
0 and R1 produced no scientific packet; R1 exited code 1 without an
infrastructure classification. Q2 is closed, and no retry is implicit.

### 6.3 Diagnostic protocol

Using open development populations only:

1. establish the clean one-step/rollout discrepancy for the fixed PCNO;
2. probe response profiles over frozen perturbation radii and directions;
3. measure model-owned defect, reference-proximity drift, and physical failure
   channels;
4. determine which response, forcing, coupling, or constraint mediator appears
   relevant; and
5. freeze the retained B4 implementations, their expected signatures, the
   predicted ranking, and all no-harm/cost rules for a separate prospective
   population.

The exact PDE and method implementations remain unresolved until the next
owner discussion. B3 may compare candidate diagnostics on development data,
but it may not use designated B4 long-horizon outcomes to select them.

### 6.4 Required endpoints

- clean H1 state and increment error;
- common-bank learned response and trusted response defect;
- model-owned/on-policy defect;
- qualified response blocks or finite-amplitude secant gains;
- response profiles as functions of perturbation radius, rather than only one
  local scalar;
- time-/condition-matched reference-proximity and population drift;
- first tube/proxy exit and survival;
- H16/H64 or contract-specific path error;
- energy, enstrophy, spectrum, front/shock, positivity, boundary, and
  conservation channels where physically valid;
- accurate, admissible, bounded, and finite event times separately;
- solver queries, training time, inference time, memory, and parameters; and
- clean/tangent and structure no-harm.

No single global normalized error substitutes for the structure and validity
channels.

### 6.5 B3 output

B3 closes only with a signed prediction-freeze packet containing the selected
PDE and method identities, diagnostic definitions, expected mediator changes,
ranking predictions, prospective population, horizons, statistics, costs, and
hashes. Without this packet, B4 cannot begin.

## 7. B4: Primary Complex-PDE Intervention Study And Reveal

### 7.1 Representative comparison

B4 is the mandatory main-paper intervention study on the same primary PDE as
B3. It retains one implementation for each framework slot selected in the B3
freeze:

| Slot | Minimum comparison role |
| --- | --- |
| CLEAN | Uncorrected fixed-PCNO control. |
| RECOVERY | Embedded return-to-path training. |
| EXPOSURE | Pushforward or multistep training on model-generated states. |
| DYN-RELABEL | Trusted dynamics from displaced states. |
| STRUCTURE | One PDE-relevant structural restriction or correction. |
| EXPLICIT-PROJECTION | One deterministic operational corrector. |
| LEARNED-REFINER | One learned sampling or refinement corrector. |
| HYBRID | Conditional: relabel-trained predictor plus an attributable recovery corrector, retained only if its B3 contract and audit close. |

The exact algorithms remain unresolved here. They are selected once, before
implementation, and are not replaced after designated outcomes are opened.
Freeze the PCNO recipe, clean data, normalization, seeds, query banks,
preprocessing, evaluator, and selection rules across comparable arms. When
exact budget matching is impossible, report equal-information and equal-cost
views rather than claiming perfect parity.

Operational correctors use a fixed schedule, never an online OOD trigger.
Evaluate the raw predictor `F` and deployed transition `C o F` separately, and
report training labels, solver calls, reference-bank size, parameters,
correction iterations, memory, and latency.

### 7.2 Blocking implementation audit

Before any run, each slot needs a written intent contract covering its input
law, target, information source, composition order, rollout feedback state,
normalization, timestep, boundary/state representation, correction schedule,
expected signature, cost, and leakage prohibitions. A Codex agent must trace the
implemented code path and verify the contract with focused tests. If the
scientific meaning is ambiguous, implementation and execution stop for owner
clarification.

### 7.3 Freeze before reveal

Before opening the designated long-horizon outcomes, freeze:

- source commit and source manifest;
- data, trajectory split, and population manifest;
- model/checkpoint identities and hashes;
- retained method identities and implementation-audit receipts;
- evaluator and numerical-repeatability receipt;
- common query bank and perturbation amplitudes;
- state-only feature map, normalization, kNN `k`, local-PCA rank rule, MMD
  kernel/bandwidth, and physical feature blocks;
- clean H1 and all early response/drift metrics;
- primary long horizons and tie margin;
- predicted model ranking and predicted recovery-versus-relabeling ranking;
- Kendall/Spearman, all-pair, and discordant-pair analysis;
- trajectory/seed bootstrap procedure; and
- no-harm, failure, and cost thresholds.

D094 outcomes cannot enter this freeze because their H79 results are already
known. They are retrospective motivation only.

### 7.4 Transparent framework score

When local geometry qualifies, estimate clean forcing `b_hat` and a two-channel
tangent/normal response matrix `G_hat` on the common bank, then propagate

\[
\widehat z_{n+1}=\widehat G_n\widehat z_n+\widehat b_n,
\qquad
\widehat z_0=0,
\]

and rank models by

\[
\widehat R_H=\|\widehat z_H\|.
\]

Do not fit a high-capacity ranking predictor. If local geometry does not pass
its qualification gate, retain the finite-amplitude response and drift
diagnostics, drop tangent/normal language, and preregister a transparent scalar
risk aggregation before outcome reveal.

### 7.5 Drift diagnostics

Fit every diagnostic on clean reference/training data only and freeze it across
all models:

- normalized time-/condition-matched kNN distance;
- qualified local-PCA reconstruction residual;
- MMD on a frozen physical/state feature map;
- individual physical feature deviations; and
- tube/proxy survival and early-score time integral.

Do not use each model's latent representation. Split by initial condition or
trajectory, not by individual frames. Condition on time, forcing, parameters,
and boundary regime. Diagnostics never modify the deployed rollout.

### 7.6 Primary ranking criteria

The block-specific preregistration will freeze exact numerical thresholds after
evaluator-repeatability calibration and before long-rollout outcomes. The
default strong-evidence targets are:

- leave-one-seed-out Kendall improvement over H1 of at least `0.20`;
- all-pair accuracy improvement over H1 of at least `0.15`;
- at least `0.70` accuracy on H1/long-horizon discordant pairs; and
- a trajectory/seed-clustered 95% bootstrap lower bound above zero for the
  improvement in pairwise accuracy.

Report Spearman correlation, all ties, and all individual pairs regardless of
whether the gate passes. Thresholds cannot be changed after reveal.

### 7.7 Mediator criteria

- RECOVERY should reduce the declared transverse retention/forcing mediator.
- DYN-RELABEL should reduce trusted response defect on the identical bank.
- EXPOSURE should change response along model-prefix directions without being
  described as a trusted `Phi(x)` label.
- A structural arm should change its declared physical mediator before any
  rollout claim is credited to that mechanism.
- An operational corrector should change the post-correction mediator while
  preserving the raw-predictor measurement.
- The hybrid should exhibit its preregistered inner-faithful/outer-corrective
  response profile.
- The preregistered problem diagnosis must predict which mediator matters for
  the ID rollout.
- A rollout change without the predicted mediator change rejects the proposed
  explanation.
- A mediator change without rollout benefit indicates poor coverage or an
  irrelevant local response object.

No universal winner is required or claimed.

## 8. B5: Conditional Secondary-PDE Transfer

B5 tests whether the framework remains useful under a contrasting PDE regime.
It uses a reduced, preregistered matrix: CLEAN plus the smallest set needed to
compare the primary case-study diagnosis with an embedded, operational, or
hybrid alternative. The secondary PDE and retained methods are selected only
after B4 is complete.

B5 must satisfy the same state-closure, reference, split, implementation-audit,
and provenance requirements as B3/B4. It may strengthen regime-dependent
generality, but it does not delay the first serious submission draft. A failed
transfer is reported as a boundary of the framework or diagnostics, not repaired
with a benchmark sweep.

## 9. Statistics And Reporting

- Three paired seeds are required for every new stochastic learned comparison.
- The trajectory/initial condition is the primary sampling unit.
- Use paired differences wherever methods share a trajectory or seed.
- Use hierarchical or cluster bootstrap intervals over trajectories and seeds;
  do not treat time frames as independent samples.
- Report mean, median, quantiles, worst cases, and per-seed effects.
- Separate numerical repeatability from scientific seed variability.
- Report every registered failure and incomplete run under its original
  identity.
- Exploratory analyses are labelled post hoc and cannot repair a failed frozen
  gate.

## 10. Cost And Resource Plan

| Block | Planned scale | Resource policy |
| --- | --- | --- |
| B1 | Exact formulas and small grids; minutes. | CPU only after implementation authorization. |
| B2 | Small learned-ODE representatives; three paired seeds where trained. | CPU or one small GPU after a measured smoke; no architecture sweep. |
| PDE readiness | Reference, population, restart, and PCNO memory qualification. | Stage-specific local/remote choice requires owner approval; no estimate replaces a measured envelope. |
| B3/B4 | One primary PDE and the frozen representative slots; operational correctors may share predictors. | Measure each training and inference contract before launch; no post-reveal method additions. |
| B5 | Reduced transfer matrix on one secondary PDE. | Conditional and separately authorized. |

Every retained run writes a unique output root, immutable run contract, source
manifest, split/population manifest, progress log, result packet, artifact
manifest, and closeout receipt. Failed attempts are never overwritten.

## 11. Run Order And Stop Rules

1. Owner reviews the active plans.
2. Select the primary PDE and exact representative implementations in a
   separate owner discussion.
3. Qualify the primary PDE reference, population, restart, and resource
   contract.
4. Establish the clean PDE baseline and begin B3 development diagnosis. In
   parallel, implement and verify B1 identities, then run B1 only after
   authorization; stop if its predictions fail.
5. Complete B3 and bind the problem-specific intervention-hypothesis freeze.
6. Implement and run B2 under a separate contract; report all three seeds.
   B2 is required supporting evidence but must not delay primary-PDE readiness.
7. Implement every retained B4 slot, then complete its blocking intent-to-code
   audit.
8. Train or attach the frozen representatives under separately approved named
   contracts.
9. Reveal only the designated long horizons and run the frozen B4 analysis.
10. Consider B5 only after the primary case study is complete.

If PDE readiness is unresolved at the calendar cutoff, do not activate another
architecture or broad PDE search. Revisit the PDE choice or schedule explicitly;
B1/B2 alone do not constitute the planned serious submission draft.

## 12. Claim-to-Block Matrix

| Evidence | C1 | C2 | Supporting only |
| --- | ---: | ---: | ---: |
| Theory and counterexamples | Primary |  |  |
| B1 exact ODE | Boundary/illustration | Mechanism prediction |  |
| B2 learned ODE |  | Supporting mechanism realization |  |
| D094 retrospective evidence |  |  | Phenomenon motivation |
| B3 primary PDE diagnosis/freeze |  | Problem-specific prediction |  |
| B4 primary PDE intervention/reveal |  | Primary |  |
| Literature taxonomy |  |  | Unifying language and coverage |
| B5 secondary PDE transfer |  | Conditional transfer |  |

## 13. What Is Not An Experiment Result

- an unexecuted plan or preregistration;
- a successful synthetic unit test;
- an incomplete process with no result packet;
- an artifact packet whose manifest does not rehash;
- a validation result relabelled as test;
- a retrospective score computed after the target horizon was seen; or
- a drift proxy described as the true data manifold.
