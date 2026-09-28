# Experiment Plan: Corrective Mechanisms With Fixed, Rich Data

Updated: 2026-09-24

Current status: the selected comparisons, design test and 64-trajectory
confirmation are complete, as is the first scientific draft. The stages,
recipes and forecasts below document that completed programme; they are not an
active launch queue. [PAPER_PLAN.md](../../paper/PAPER_PLAN.md) owns the remaining
mentor revision and submission preparation.

The six-section manuscript and primary-source survey guide the empirical work
for the owner-selected JCP target. Venue selection does not expand the roster.
The reached-state diagnostic below is complete and incorporated in Section 4;
the gradient and target contrasts are complete. The owner-authorized workstation
remains the execution resource. Privacy and protected-population boundaries persist.
[PROJECT_PLAN.md](PROJECT_PLAN.md) owns completion;
[PAPER_PLAN.md](../../paper/PAPER_PLAN.md) owns the figures;
[the source ledger](../../paper/LITERATURE_AUDIT.md) owns literature semantics.
The completed comparison's recipe and original hypotheses are retained below,
byte for byte. Its numerical records remain in the dated experiment tracker.

## Selected Scientific Work (completed)

Keep the existing Kolmogorov regime, all 32 clean training trajectories, four
already-open development trajectories, PCNO backbone and autonomous start at
step 0. Do not vary data count, retune the PDE for a preferred winner, or reopen
ODEs. Onset clean competence is established; the later clean generalization gap
must stay visible. Fixed clean data does not mean equal total information:
trusted displaced-state labels and analytic priors are additional information.

Two empirical questions organize these comparisons:

| Question | Decisive evidence | Paper destination |
| --- | --- | --- |
| Which property does each correction change, and what does it sacrifice? | Matched target/gradient contrasts; clean forcing, finite response, bias, path/physical fidelity and cost. Rollout rank alone is insufficient. | Sections 3--4: mechanism and comparison tables; response and fidelity panels. |
| Can those measurements guide a useful quantitative intervention choice? | A numerical application prediction frozen before the corresponding new rollout, followed by independent confirmation and a useful-range check. | Section 5: predicted versus observed effect, uncertainty, failures and limits. |

Existing theory and the completed clean/recovery reversal supply the motivation.
The completed reached-state diagnostic is retrospective; it cannot become a
prospective prediction merely because new response measurements were produced.

September 20 owner direction: pursue useful mechanism hypotheses without making
every exploratory experiment pass a quantitative forecast screen. The forecast
requirements in Block 3 apply to a claimed prospective prediction; they do not
prohibit finite-rollout comparison or retrospective discovery in Block 2. Clean
acquisition error is a measurement, not an automatic rejection of a finite map.

### Representative Coverage And Selected Roles

The selected representative comparison is closed. September 22: the owner
selected response-preserving reduction of clean bias for Section 5, against a
matched clean-only continuation. Block 3 below owns this new test. The completed
coverage recipes and unfavorable results remain evidence; no additional baseline
or tuning sweep follows from this choice.

Cover distinct corrective operations, with explicit representatives and source
fidelity. This is not a requirement to reproduce every architecture in the
40-source survey or run every method on every PDE. Applicable families cannot
disappear after an unfavorable result. A citation does not close an empirical
coverage gap. The following is the working roster for Sections 3--4:

| Mechanism / representative | Evidence to use or obtain | Expected signature and informative contrary outcome |
| --- | --- | --- |
| Clean supervision / matched clean continuation | Reuse the fixed Clean32 parent and completed continuation. | Small clean error need not rank composition; later clean gaps qualify the horizon of that interpretation. |
| Noise/recovery / direct-state adaptation informed by GNS, MGN and DPOT | Reuse online Gaussian recovery; add its matched bank arm for the target contrast below. | Reduced response to trained displacements can help early rollout, with finite range and recovery bias. Failure to transfer to native directions limits that recipe. |
| Faithful displaced dynamics / offline APEBench-STAP-inspired relabeling | Completed Kolmogorov REC-bank versus DYN-bank at identical inputs, with unused-probe transfer and costs. Reuse ODE/NACA as support. | DYN should improve actual-input dynamics if acquired; REC may remove displacement more strongly. Either can win rollout. No acquired target difference means no mechanism conclusion. |
| Model-prefix exposure / stopped temporal gradients | Completed K=4 PCNO arm with stored future targets. | Native-input exposure may improve transfer without reducing every gain. No transfer benefit weakens exposure as the relevant missing ingredient. |
| Differentiable multistep training / full temporal gradients | Completed paired K=4 arm with identical loss and depth; only the temporal gradient path changes. | Extra benefit can reflect coordinated composition/bias changes rather than contraction. Similar outcomes indicate little added value from temporal gradients at this depth. |
| Empirical projection / train-derived path reconstruction | Use verified NACA projection as the principal geometry contrast. No automatic new KF nearest-bank projection. | Favorable near-linear geometry can make correction effective; this does not establish universal superiority or an exact manifold. A KF projection requires adequate independent geometric resolution first. |
| Learned conditional refinement / PDE-Refiner style | Completed fixed EMA comparison under `kf_refiner_20260919c/evaluation`; reuse NACA/Bump selectively. | Limited early response improvement and clean bias coexist with later amplitude growth. This qualifies the four-call adaptation, not the family. |
| Conditional generative prediction / ACDM style | Completed separate one-state PCNO comparison under `kf_acdm_20260920b/evaluation_v2`. | Strong response attenuation and finite evolution coexist with large clean error and physical damping. Three draws are not fitting-seed replication. |
| Imposed dissipative response / MNO-style shell-target training | Completed and independently reduced in `kf_remaining_coverage_20260921a`: fresh-shell output/input RMS 0.5027 for target 0.5; unchanged clean sample tapes. Development onset/H8/H32/H128 errors 0.08037/5.9320/68.8438/163.9019%. Early response remains close to continuation; all paths finish inside both RMS envelopes with poor physical fidelity. | Acquired far-shell behavior and finite evolution do not ensure early response control or path accuracy. Shell targets are an analytic prior, not trusted displaced dynamics or a uniform stability certificate. |
| Architectural response restriction / McCabe et al. | Completed and independently reduced as `kf_spectral_control_20260921b`: all effective matrix caps verify; early parent-error gains decrease but Gaussian gains remain above one. Development onset/H8/H32/H128 is 0.46284/3.5109/73.0125/212.0748%. This is one matched-budget PCNO normalization component, not full ReFNO or optimal constrained fitting. | Selective response benefit coexists with clean-accuracy cost and later physical distortion. Residual sums and graph derivatives prevent a whole-map contraction conclusion. |
| Physical constraint correction | Completed and independently reduced zero-fit online REC plus radial output projection, in `kf_remaining_coverage_20260921a`: all 12 paths reach H128; development error 332.7542%; first development activation 57--65. H8/H32 and common-response vectors are unchanged. | Enforcing this coarse ball does not enforce the tighter time-dependent envelope, proximity, or path accuracy. The observed late activation matches the qualitative expectation. |
| Statistical correction / DySLIM; unconditional denoising / Thermalizer | Training-only late-block checks show enstrophy/palinstrophy drift of 13.26%/12.90%. The present pool is not supported as invariant-law data; retain the published mechanisms in Section 3, without an empirical ranking in this transient case. | This scopes the fixed case, not the method families. A time-conditioned adaptation or stationary dataset would be additional research. Thermalizer also uses an online classifier outside the current deployment contract. |

The September 21 checks resolve the last three applicability decisions above.
All selected comparisons above are now complete at their declared component,
fitting-budget and population scope; the stationary-data and deployment
exceptions remain explicit. The response-preserving design test and independent
64-trajectory confirmation are also complete. Next consolidate the manuscript
and supporting-case methods. Earlier recipes and pre-outcome expectations below
remain evidence, not a reopened queue.
Do not acquire a new stationary dataset or substitute a different PDE silently.
Online solver/observation hybrids remain framework context outside the declared
deployment contract.

### Shared Comparison Contract

- Same clean training population, normalizer, state restriction and evaluation
  starts. Fit and tune only on permitted training/development roles.
- For paired target or gradient contrasts, match parent, sampler, optimizer,
  update budget, loss weights and terminal-selection rule wherever applicable.
  The completed 4,096-update recipe is a starting cost reference, not a universal
  adequacy rule for a new diffusion objective or architecture.
- Record actual targets, history, gradients, random inputs and the complete
  deployed transition. A PCNO objective adaptation is not a full reproduction.
  Use one-state conditioning for the main Markov-state ACDM adaptation to avoid
  introducing a true extra future/history state at evaluation; disclose the
  published two-state variant. A two-state reproduction would need a matched
  history control and a separately declared initialization contract.
- Compare paired methods on shared physical displaced inputs and both targets.
  For stochastic maps, share noise draws across paired probes and report
  per-realization variation; do not equate an ensemble mean with a deployed path.
- Separate fit updates, reference exposure, solver labels/calls, inference calls,
  wall time and peak memory. Match information where possible; disclose it where
  the mechanism itself changes information or compute.
- Report clean errors by time window; finite-amplitude forcing/odd/even response,
  alignment and target-specific bias; H8/H32 and H128 error/failure accounting;
  phase/path and energy/enstrophy diagnostics. No fabricated H128 average after
  censoring. A numerical amplitude guard is not an accuracy criterion.
- Start with a development comparison for mechanism identification. Replicate
  the central contrasts and final design, including contradictory cases, rather
  than every row automatically. Trajectories are not optimization seeds; shared-
  parent fine-tuning replication is not independent end-to-end training.

### 1. Completed: Reached-State Response Without Refitting

`kf_reached_response_20260917a` completes this block. The dated
[tracker](EXPERIMENT_TRACKER.md) binds the verified report and Section 4 panel.
Common-amplitude/native probes show finite useful range, a step-4 directional
exception and late clean-forcing cost. The independent PDE/Galerkin envelope
establishes eventual exclusion, with sampled numerical qualification; it does
not identify early tangent/normal error. No further geometry work is required
before the selected target and gradient contrasts. The original diagnostic
contract is retained below.

Replay the three frozen maps through step 32 on the existing eight training and
four open development paths, retaining dense states and checking saved anchors.
At input steps 4, 8, 12, 16 and 31, compare parent-native, recovery-native and
Gaussian directions using identical physical perturbations for every map.
Normalize directions to unit RMS, then use common amplitudes 0.01, 0.03 and 0.1
times the inherited training scale plus the two native donor magnitudes.
Use both signs. A zero native direction has no defined unit direction.

Measure clean forcing b, odd/even response, signed alignment, positive-direction
recovery error and gain; keep acquisition on Gaussian inputs distinct from
transfer to recovery's own reached directions. Use the complete deployed map.
Retain compact vectors sufficient to recompute the decomposition. This separates
amplitude, direction and clean-bias effects without fitting another model.

Alongside this assay, compute the cheap reference-family amplitude envelope.
For mean-zero periodic vorticity with viscosity nu, drag gamma and forcing f,
lambda = nu*(2*pi/L)^2 + gamma gives
W(t) <= exp(-lambda*t)*W(0) + ||f||_RMS/lambda*(1-exp(-lambda*t)).
Verify the IC-family norm and continuous/discrete numerical allowance using the
existing reference states. Violating a valid envelope excludes that family;
satisfying it does not establish proximity or path accuracy. Do not convert it
into a deployed clipping rule without evaluating that rule separately.

Only if the intervention choice remains ambiguous, use a bounded trusted
response assay to distinguish physical amplification from learned response
error, or a local phase-family tangent check to resolve a specific geometric
claim. No global manifold fit or tangent/normal attribution from PCA alone.
Stop this diagnostic when it can distinguish range, direction, bias or temporal
composition as the actionable limitation. Its deliverable is one response/
fidelity panel and a stated intervention hypothesis, not a new diagnostic suite.

### 2. Fill The Missing Comparisons In Stages

First complete the two causal contrasts; do not restart the old solver-bank
queue just because its code exists. The completed diagnosis directs attention
to early reached directions and finite amplitudes, with clean forcing retained
as a cost. It does not select stronger noise or a coarse amplitude cap as a
sufficient remedy; the latter would act well after accuracy is lost.

For REC-bank versus DYN-bank, choose a small training-only displaced-input bank
from the diagnostic's relevant amplitudes/directions and onset times. Freeze
exactly the same x = u + eta for both arms. REC targets the archived successor
Phi(u); DYN targets the trusted restart Phi(x). Both keep the same clean branch.
Qualify clean restarts and numerical target differences, report every added
solver label, and test unused perturbations to separate acquisition from bank
memorization. The existing online REC arm alone is not a matched comparator
for a finite-bank DYN arm. The old bank beginning at step 15 does not answer the
onset question automatically. NACA's trained maps can also be scored against
both retained target types without new fits, if the contracts match.

The completed `kf_target_pair_20260918a` comparison fixed training inputs
at steps 4/8/12/16 on all 32 paths: Gaussian and parent-native unit directions,
amplitudes 0.01/0.1 training scales, both signs (1,024 inputs). Unused probes
use steps 6/10/14 on the original eight training and four open-development
paths, with new Gaussian draws (288 inputs). Two fresh-Adam fits share the
parent, clean batch 8, bank batch 8, equal branch weights, LR 1e-4, 4,096
updates and sampler tapes. Only REC versus DYN bank targets differ. Retain
the canonical-projection/FP32-label qualification, four train refinement
sentinels, exact solver cost, and acquisition on both banks. The ignored
attempt plan binds details before outcomes. Failed qualification stops the
attempt; no automatic bank/threshold changes. Assays precede new autonomous
outcomes, and no REC/DYN rollout ranking is assumed in advance.

September 18 closeout: the K=4 comparison is complete as
`kf_unroll_pair_20260918a`, with fixed fits, independently reduced assays and
rollouts. Detached/full development clean onset error is 0.16583%/0.20160%,
H8 is 1.1694%/1.7871% and H32 is 59.7540%/62.6150%. Detachment improves H8 on
all twelve paths and H32 on eleven, with 3.55/11.93 GiB peak memory and about
1.80 hours per fit. Both qualitative assay-informed H8 forecasts pass.
Twenty-three of 24 new rollouts hit the amplitude guard before H128; the one
completed path is inaccurate. No universal ranking, optimization-convergence
claim or quantitative design success follows. The ignored analysis.md and
dated tracker own verification and costs; Section 4 contains the comparison.
The same-input REC-bank/DYN-bank contrast is now also complete. The retained
recipe below describes the gradient test; it does not authorize a depth sweep.

For exposure versus temporal gradients, begin with K=4 transition steps and
the same averaged reference-path loss over those steps. One arm detaches each
recurrent input; the other differentiates through the same sequence. Use the
same sequence starts and loss weights. Detachment changes gradients without
changing the forward construction for fixed parameters; the trained maps will
subsequently induce different inputs. This is a controlled short-unroll
adaptation, not a claim to reproduce every MP-PDE pushforward detail. K=4 gives
several displaced-input exposures within the observed early-error window.
No horizon sweep or curriculum is selected; adjust depth only for a stated
scientific limitation, retaining the first result.

Finish representative coverage, then discuss the quantitative design test in
Block 3 with the owner. The September 21 instruction supersedes the earlier
parallel design work. Do not automatically reopen a blend, switch, or buffer method.

Add conditional refinement, conditional generation and the qualified
response-prior representative. Fix their faithful objectives, conditioning,
noise schedules and training sufficiency checks before looking at rollout
rankings. Reuse published schedules where compatible and measure one short
resource batch for unfamiliar operations. Do not declare an entire family
ineffective from an unacquired objective or an arbitrary inherited update cap.
Resolve the conditional roster rows alongside these comparisons. Keep all
outcomes and costs; no method-by-PDE-by-strength-by-seed Cartesian sweep.

### September 19 Closeout And Immediate Next Work

Both causal contrasts now have fixed fits, verified mechanism assays and rollout
outcomes. In the target pair, unused native probes separate removal from faithful
dynamics; both suppress excessive parent response, with a large clean-error cost.
All 24 bank-arm rollouts complete H128 and satisfy the coarse physical envelope,
but neither sustains path accuracy. DYN is better at H8 and pooled H32; REC has
better pooled H128 error and scalar fidelity. The dated tracker and ignored
analysis own exact values. This closes the main PDE target-semantics gap, not
quantitative prediction or independent confirmation.

The preceding empirical decision was acquisition of the implemented conditional
Refiner comparison; its outcome and the current next comparison are below.
Its question is whether successor refinement preserves clean dynamics while
obtaining useful displaced-input response control. Freeze faithful conditioning,
targets and the complete inference recurrence; measure acquisition and model-call
cost before interpreting H8/H32, the 10% crossing and H128 path/physical errors.
Use the fixed clean dataset. Improvement at a smaller bias cost supports the
design lever; no benefit or lost conditional fidelity is an informative limit.
Do not assume the old 4,096-update budget proves acquisition of a new objective.

Acquisition decision, September 20: the fixed 8,192-update pilot and its one
unchanged-recipe continuation to 16,384 updates are complete. The continuation
does not resolve the clean-fidelity and internally generated candidate mismatch;
close that recipe at its registered review stop. The dated tracker and handoff
own exact cohorts, reductions, source/replay checks and costs. No additional
clean data, labels, development or autonomous outcome entered these attempts.
A larger clean error alone does not establish worse rollout. The main-case
method comparison is now complete for that fixed EMA terminal under the shared
teacher/response/H8/H32/H128 protocol. Development clean onset/H8 errors are
0.8971/5.8652%; all 36 path/draw combinations hit the numerical amplitude guard
before H128. Early native response is reduced relative to continuation but
remains above one and rises at input 31; clean forcing also increases. The
ignored `kf_refiner_20260919c/evaluation/analysis.md` and tracker own the verified
results and pre-outcome expectations. This closes the comparison without a
new fit or quantitative-design success. Resolve the planned response-prior
and conditional coverage decisions next; any later refinement revision must
address actual feedback errors and preserve clean dynamics. No automatic
further extension, schedule search or intermediate-checkpoint selection.

The completed quantitative-design feasibility checks used the acquired maps'
forcing/response tradeoff: low clean forcing versus broader response control.
The retained decision rules below govern an explicitly justified revision. A fixed strength or
schedule of existing maps is a simpler candidate than a new geometry-dependent
buffer. Select one candidate only if a calibrated, forcing/alignment-aware model
supports a numerical useful-horizon or error forecast with an explicit range.
Previously inspected outcomes may calibrate/check the relation; they cannot be
counted as prospective success. No candidate rollout precedes its frozen forecast.
If the available assays do not support that prediction, state the missing
quantity and bound the next check; do not revive an open-ended baseline-forecast
programme or automatically launch a target-mixture/noise-strength sweep.

The September 19 saved-vector parent/DYN blend check closes the simplest local
selection rule: recovery-target MSE on training parent-native inputs 4/5/6 is
minimized at the existing DYN endpoint (unconstrained alpha 1.0673), retaining
about nine times the parent's clean query MSE. This retrospective calculation
does not select a new blend. The missing quantity is how forcing, direction and
alignment evolve under the candidate's composition. That gap motivated the
subsequent time-ordered calibration below; it did not justify a strength sweep.

Completed feasibility candidate, with no selected correction: blend
from the bank-DYN map toward the lower-clean-bias online recovery map with one
fixed coefficient. Along an already-observed bank-DYN trajectory, propagate
the derivative with respect to that coefficient using the baseline Jacobian and
the difference between the two maps on the same baseline states. This retains
forcing direction and time ordering that the static query-loss check omitted.
It is sensitivity of an intervention about an observed trajectory, not a new
general predictor of baseline errors.

Use training calibration paths and a small set of finite-amplitude checks to
assess a local coefficient range. Do not evaluate the candidate on every
forecast state or inspect its autonomous outcomes to choose that range. Only
propose a coefficient if the measured sensitivity and its validity support a
useful numerical error forecast against both unchanged endpoints, with clean
bias, physical fidelity and the two-call deployment cost explicit. The quadratic
loss of a linearized state forecast is a surrogate, not an exact second-order
error expansion. If no useful interior choice or credible range is found, stop
this candidate and record the limitation; do not turn it into a coefficient
sweep. The single bounded revision is now closed below.

Closeout: `kf_design_sensitivity_20260919a` does not justify advancing this
fixed blend. Its H16 surrogate favors about 18% online recovery, beyond the
supported local range; all three registered sampled remainder screens fail,
the 1% case narrowly. That small coefficient predicts under 1% relative gain
for twice the inference calls. H32 sensitivity also disfavors positive blending.
The eight DYN training 10% crossings are actually steps 20--22; the frozen plan's
15--18 rationale was corrected without changing its preselected H16 endpoint.
No candidate rollout or coefficient sweep follows. Keep this feasibility limit
in planning records, not a new main-text panel. Its bounded revision addressed
deployment cost and temporal composition with a fixed switch, as recorded below.
Neither check closes Section 5 or initiates another forecast programme.

Completed revision: `kf_prefix_switch_20260920a` fixed online REC for the first
eight steps and bank-DYN thereafter, with a step-16 endpoint forecast on the
first eight training paths. This avoids continuing two-call blending but still
depends on the second map's response to the prefix displacement. Its signed
remainder screen fails on 5/8 paths (maximum 1.2460, limit 0.10); the linear
surrogate is worse than DYN and its curvature diagnostic gives no useful range.
REC also has slightly worse pooled step-8 endpoint error despite improving 7/8
paths. The independently checked result rejects this prediction rule, not an
unevaluated hybrid rollout. Close this revision without candidate rollout,
switch-time sweep, path exclusion or relaxed threshold. Detailed evidence stays
in the dated tracker and ignored analysis, not a new main-text panel.

The next work remains the representative comparison below. Assess a design
lever from those response/bias measurements; do not keep extending local blend
or switch forecasts. The quantitative decision and independent confirmation
are still essential scientific gaps. Review their feasibility at the delivery
schedule's September 23 checkpoint rather than silently downgrading the claim.

### Completed Comparison: Conditional Generation

The selected representative is the one-state PCNO adaptation of ACDM. Ask whether
noisy conditioning plus repeated successor generation preserves flow structure
under accumulated input error, and what it costs in path accuracy and sampling
variation. Conditioning augmentation may reduce sensitivity; the successor prior
may sustain plausible fields. Neither effect guarantees temporal fidelity.

The frozen recipe uses joint current/successor epsilon prediction with mean
Huber loss, 20 DDPM levels and the released linear beta schedule. At each reverse
level replace the conditioning with the supplied state at that noise level;
keep its noise field fixed across levels as in the source. Use one observed
vorticity field, exact forcing, the inherited scale and PCNO body, a new joint
head, and the common restriction only on the final physical successor. Raw
parameters are used without EMA. The source ledger owns the disclosed differences
from the published two-state U-Net method.

Fit 8,192 updates on the unchanged 32 paths, after a 16-update resource check.
The fixed terminal, not the best probe, enters the comparison. Training probes
measure denoising and full generated transitions with three fixed noise tapes.
Then compare H8/H32/H128 on eight training/four already-open development paths,
three sampling realizations, shared noise for paired response probes, phase/path
error, energy/enstrophy, time-resolved spectra and actual inference cost.
No clean-error cutoff or successful local forecast is required before evaluating
a finite model. Poor path accuracy with better physical fidelity is informative;
do not infer stationary sampling from these transient trajectories.

The ignored `kf_acdm_20260920b/plan.md` and manifest freeze implementation and
budget; the dated tracker owns launch/status evidence. Section 3 describes the
mechanism, Section 4 receives the shared comparison, and Appendix B records the
implementation. A design idea emerging from this comparison can be explored;
its later prospective test must use outcomes that have not already selected it.

September 20 closeout: the fixed fit and evaluation are complete. Response
attenuation coexists with large clean bias and excessive physical damping;
all sampled paths complete H128 but exceed 10% error at step 2. The handoff and
dated tracker own verified results; Section 4 uses one added shared-table row,
and Appendix B records sampling and implementation. No automatic acquisition
extension or diffusion tuning follows this recipe's limited performance.

The qualified response-prior decision follows this comparison;
architectural, physical and stationary-statistical applicability decisions remain
explicit. Complete coverage alongside the one design test, then replicate the
central contrast/design and obtain an agreed untouched confirmation population.
Do not expand the PDE/data roster or add a figure per completed attempt. Section 4
uses one shared eight-map table and the consolidated comparison figure; Section 5
still needs its quantitative prediction-versus-outcome display.

### 3. Use The Diagnosis To Select One Quantitative Design Test

#### Selected September 22: Reduce Clean Forcing While Preserving Response

The owner selected this direction over the optional inner-DYN/outer-REC buffer.
First-pair checkpoint (seed 17): the fits, assays, bounded forecast and frozen-map outcomes
are complete and independently reduced. Response retention transfers to unused
probes, and the preserving arm's qualified H8 prediction matches its outcome.
Clean-only adaptation wins H8 but loses H32; preserving is worse than DYN at
H128. The clean-only forecast failed its local checks before outcome inspection.
This closes a conditional onset prediction, with independent confirmation still
open. The dated tracker owns exact results. The fixed recipe and pre-outcome
forecast specification below record the completed test, not a new launch queue.

Question: can the bank-DYN map's useful finite-displacement response be retained
while its state-dependent clean forcing is reduced? A training-only saved-query
check ruled out a promising *shared additive* explanation: the constant mean
field explains 1.98% of clean forcing energy across 32 paths and four anchor
times, but leave-trajectory-out subtraction reduces MSE only 0.33%. This does
not rule out state-dependent correction. Exact evidence belongs in the tracker.

Let D denote the frozen bank-DYN terminal, G the adapted map, and
R_G(u,eta)=G(u+eta)-G(u). Fit two arms from the identical D checkpoint:

- Clean-only: the existing two-stream clean objective L_clean(G).
- Response-preserving: L_clean(G) + E_bank ||R_G(u,eta)-R_D(u,eta)||^2 / s^2.

Here s is the inherited training RMS. Both evaluations of G in the response
difference receive gradients; the archived D response is fixed. This penalizes
changes in learned response, not its magnitude and not a presumed normal
component. The useful D response is itself an empirical approximation, not a
trusted dynamics target. The penalty may also preserve unwanted behavior.

Use all fixed 32*512 clean pairs, the existing 1,024 signed training-bank
inputs and their already-recorded D outputs. No new solver labels or trajectories.
Both arms use fresh Adam, learning rate 1e-4, 4,096 updates, two clean batches
of eight, independent sampler seeds 17 and 1701, and identical clean tapes.
The preserving arm adds eight bank pairs per update, sampled with seed 1703;
its response coefficient is fixed at one, without a weight search. Both start
from the same bank-DYN terminal, retain the common state restriction, and deploy
one unchanged PCNO call per step. This matches clean information and update
count, not total fitting FLOPs: the extra response branch costs two forwards and
their gradients. No checkpoint selection or automatic continuation is included.

Before fitting, use synthetic checks of the centered gradient, zero-penalty
control, immutable teacher/bank identity and train-only scope, then a 16-update
resource check. Provisional wall allowances are three hours for clean-only and
five hours for response-preserving; measure actual memory and time before launch.

After fitting, separate three questions: does clean forcing decrease; how well
are signed finite responses retained on training and unused probes; and does
that combination improve self-composition? Common probes retain both REC/DYN
target errors, odd/even response and alignment, including any acquired defects.
These are measurements, not an automatic promise of rollout improvement.

The primary quantitative target is the new maps' H8 relative L2 error and its
change against the frozen D baseline and matched clean-only arm. This onset
window has established clean competence. Use the existing directional,
forcing-aware diagnostic on declared calibration inputs; freeze numerical
predictions, their checked range and uncertainty before opening either new
autonomous outcome. Clean-error-only selection is the simpler decision rule.
Do not substitute a scalar gain or recursively query the exact candidate map
along a forecast and call that a prediction. H32/H128, first 10% crossing in
steps and physical time, and physical fidelity remain required outcomes.
No long-horizon quantitative accuracy claim follows from an H8 prediction.

September 22 forecast specification (before new autonomous outcomes): use the
saved DYN states v_0,...,v_8 as fixed expansion points for each frozen candidate
G. Starting at z_0=0, propagate z_{n+1}=G(v_n)-v_{n+1}+DG(v_n)z_n and predict
the pooled error of v_n+z_n against u_n, n=1,...,8. Record G(u_n)-u_{n+1}
separately; the forcing on the DYN path is not a clean-reference defect. Run
the same calculation on the already-observed bank-REC control and report its
qualification and observed discrepancy. Use eight training and the four
already-open development paths; development H8 is primary.

Signed probes at v_n +/- z_n diagnose remainders r_n without updating z.
Propagate the error proxy q_{n+1}=DG(v_n)q_n+r_n^+, q_0=0. Freeze the empirical
range p +/- max(0.1p,2Q), truncated below at zero, where p is predicted pooled
error and Q is pooled q relative to reference energy. This is a tolerance,
not a confidence interval or stability bound. Local qualification requires
both signed remainder/linear-response ratios <=0.1, range half-width <=0.25p,
and AD/no-grad forward offsets <=1e-6 of training RMS. Report all paths and
failed checks. An unqualified calculation remains a recorded prospective
surrogate, followed by explicitly exploratory outcomes, without revising the
forecast method or intervention. Select the preserving arm over clean-only
only if their qualified ranges separate; a claimed improvement over DYN must
also put its upper range below 0.95 times the DYN H8 error. The clean-error-only
rule selects the clean-only arm. Physical diagnostics and H32/H128 remain
required observations, with no predicted physical no-harm guarantee.

The first execution stops after fitting and nonrecurrent assays. A failed local
forecast does not prohibit an informative exploratory rollout, but requires an
explicit change in claim status before that outcome is opened. Do not rescue
this attempt with another loss weight, new architecture, or data change.
Independent fitting/population confirmation remains separately scoped; protected
roles stay closed. Section 5 will show one prediction-versus-outcome display,
with recipe and cost details in the appendix. A new algorithmic-novelty claim
is not part of this bounded test.

#### Retained Design Alternatives And Prediction Boundaries

| Decision-bearing observation | Candidate change | Main risk/control |
| --- | --- | --- |
| Benefit disappears mainly as amplitude increases. | Broaden recovery amplitudes over the diagnosed range. | Clean/recovery bias; compare with the unchanged recovery law. |
| Recovery-native directions fail at matched amplitude. | Use native/prefix exposure or recovery directions. | Donor dependence and bias; retain Gaussian and common-input controls. |
| Suppression removes legitimate trusted dynamics. | Weaker/selective correction or a measured REC/DYN mixture. | Faithful dynamics may preserve unwanted error; compare the pure target endpoints. |
| Tested response looks favorable but composition remains poor. | Use the short-unroll comparison to test temporal coordination. | A local assay may omit relevant states/coupling; no scalar-gain forecast. |

These are alternatives, not a strength/architecture matrix. A two-region buffer
or named new corrector is optional and needs a distinct advantage over its simpler
endpoint controls. This work is not limited to rescuing the original REC recipe:
a different representative may reveal the more useful design lever.

Preferred application observable: the first crossing of 10% relative L2 path
error. If crossing-time sensitivity lies outside the measured validity range,
use error over a fixed useful horizon selected from the existing baseline and
reference diagnostics before candidate outcomes. H8/H32, the crossing and
physical fidelity remain reported in either case. This is
a new prospective target; its already-observed values are calibration evidence.
Use a bounded forcing-aware finite-response model, retaining directions and
alignment, and check its useful range on existing controls. It must yield a
numerical interval or strength/horizon decision, not merely a response-gain
change. Do not query the exact learned map recursively on the forecast states
and call the resulting rollout an independent prediction.

After fitting a selected new candidate, its fixed clean/common-input assays
may inform the forecast before its autonomous outcomes are inspected. Freeze
the prediction, uncertainty rule, validity range, clean/physical no-harm
tolerances and decision versus a relevant simpler rule at that point. This
forecasts composition from a measured map; it does not forecast what training
will learn. New candidate-native directions belong after its forecast, not in
its calibration. Evaluate through H128 regardless of a shorter forecast window.

A successful rank or delayed guard alone does not close this block. An accurate
forecast of harm/limited range is valuable, but the paper's positive design claim
also needs a useful action. If the proposed relation cannot support a credible
application prediction, report that gap and reconsider the claim with the owner.
The blend/schedule branch has used its one bounded revision. A different design
needs a scientific rationale from the representative comparison; repeated local
forecast failures are not a reason for an indefinite search for agreement.

### 4. Confirm The Claim And Finish The Paper

September 23 closeout: the two additional pairs are complete, retrieved and
independently reduced. Response retention and H32 protection against clean-only
replicate on all twelve matched paths in every seed. Clean forcing and H8
ordering vary; the expected uniformly delayed 10% crossing does not replicate.
The three preserving H8 forecasts all qualify and agree within 0.27% relative
point error on development, including seed 18's harm relative to DYN. Every
clean-only forecast fails its local checks. All preserving fits lose to DYN
on every H128 path. The tracker owns exact values; Section 5 now reports every
seed in a consolidated table and figure. No further fitting is selected.
The approved 64-path confirmation below is also complete as
`kf_confirmation64_20260923c`. All three H32 contrasts transfer to every matched
path; the predeclared geometric ratio is 0.515 [0.499, 0.532]. Qualified preserving
H8 forecasts agree within 0.25% relative discrepancy, including harm. H128 retains
poor physical fidelity and guarded outcomes; the development-population universal
loss against DYN does not transfer. The dated tracker owns the independent
reduction and exact values. No further fitting or evaluation population is
selected. The following specifications are retained pre-outcome records.

#### Completed final confirmation: fixed maps on 64 fresh ID starts (September 23)

The owner revised the evaluation population to **64 independent trajectories**,
keeping training data and fitted models fixed. Generate 64 evaluation-only
initial conditions from the exact
existing phase law, with the same reference discretization and macro timestep.
Retain steps 0--128. Freeze their seed list before generation and verify that it
is disjoint from the existing train/development/registered role identities using
metadata only. Historical protected populations remain closed. Training data,
normalizer, checkpoints, fitting budgets and all forecast tolerances stay fixed.
This tests new initial conditions, not additional training coverage.

Use **nine existing maps**: matched clean continuation and online recovery;
bank DYN; and the three clean-only/preserving pairs (fitting seeds 17,18,19).
The first pair confirms the clean-error/composition discrepancy in Section 4;
the seven DYN/design maps confirm Section 5. Include all three pairs, without
selecting the best seed or refitting another representative. The remaining
method comparisons keep their explicitly descriptive development scope.

| Question | Frozen comparison and measurement | Consequence for the paper |
| --- | --- | --- |
| Does the clean-competent discrepancy transfer to fresh starts? | Continuation versus online recovery: clean inputs 0--7, 8--15 and 16--31; pooled H8/H32; retain existing training scores alongside the fresh scores. | A recurrence supports the main illustration beyond four reused paths. A large fresh clean gap or changed ordering limits its population claim; do not blame displacement without checking clean competence. |
| Does preserving response protect composition after clean fitting? | Each of the three matched pairs: H32 error, H8 tradeoff, H128 completion/physical fidelity, always retaining DYN as the unadapted baseline. | Confirm protection against clean-only adaptation if supported; do not translate that into superiority to DYN or sustained accuracy. |
| Does the measured response add quantitative predictive information? | All six unchanged H8 forecasts and checks; identity-response control and a scalar clean-error control, defined below. Freeze values before any adapted-map autonomous outcomes. | Report qualification, prediction discrepancy and empirical-range coverage separately. Failed checks or accurate simpler controls narrow the diagnostic claim; no tolerance adjustment or substitute forecast. |

The scalar control is `k * clean_H8`, where `k = 3.6330056115889438` is taken at
full precision from `kf_bias_replication_20260922a/forecast_controls.json`:
original DYN's **old development** H8 error divided by its clean inputs-0--7
error. Do not recalibrate on fresh rollout outcomes. Identity response uses
`z_next = G(v_n) - v_(n+1) + z_n` and the same saved DYN path/reference as the
full forecast. These are point controls, without borrowing the full forecast's
qualification or empirical tolerance. Their development comparison is
retrospective; their fresh-population comparison will be prospective.

Execution order: generate/reference-check the new trajectories; evaluate the
frozen DYN baseline and nonrecurrent clean measurements; compute and independently
freeze every candidate forecast, control, qualification and parent-relative
decision; only then evaluate continuation, recovery and the six adapted maps.
The existing decision screen is unchanged: recommend a candidate over DYN at
H8 only when it qualifies and its upper empirical range is below 0.95 times
DYN's H8 error. For comparison, record the clean-error-only recommendation
`clean(candidate) < clean(DYN)` before outcomes. These are six separate
candidate-versus-parent decisions, not a best-seed selection. Measure actual
effect and false recommendations, whether favorable or unfavorable. Direct
rollout remains cheaper; the claim concerns explanation, not acceleration.

Report each fit and each trajectory. For the H32 paired effect, average the
three log error ratios within each trajectory, then average across the 64
trajectories. Resample **whole trajectory identities jointly across all maps**
(10,000 draws, analysis seed 2026092317) for a descriptive 95% bootstrap interval.
Also give each fit's pooled ratio and paired win count. An interval overlapping
zero leaves the population-average benefit uncertain; a per-fit sign reversal
requires a seed-dependent statement. Do not count 192 seed-path cells or time
frames as independent samples. If any H32 outcome is guarded/incomplete, report
completion first and leave its unconditional numerical contrast unresolved;
do not replace missing outcomes by survivor averages. H128 is a limit/physical
fidelity check, not another optimized endpoint. Sixty-four starts are a bounded
confirmation sample, not a power guarantee or independent parent replication.

Reuse the solver, `rollout_case`, `forecast_on_baseline`, and `summarize` kernels.
The historical command wrappers hard-code train/development roles, so the new
evaluation packet needs a small explicit confirmation loader; do not relabel
new paths as old development or weaken historical validators. CPU fixtures
should check role separation, unchanged forecast arithmetic, paired reduction
and guard accounting before deployment. Bind the selected checkpoints/source
and freeze the scope before any new population is generated. No new training
module, sweep, generic experiment framework or external review is needed.
The maintained entry point is `scripts/time_dependent_no/confirm_kolmogorov.py`,
with focused checks in `tests/time_dependent_no/test_confirm_kolmogorov.py`.
Its explicit phases implement the above order; the isolated run packet owns
machine-specific input paths and frozen identities.

Cost estimate from existing manifests: the first 128 reference transitions
took 122--137 seconds per path in the 24-path generation packet (about 138 serial
minutes for 64). The six forecasts price at about 60 minutes and nine rollouts
at roughly 60 minutes from measured twelve-path costs. Allow **4--8 workstation
hours** including clean measurements, numerical sentinels, file I/O and current
machine variability; this is an estimate, not a reserved completion time.
Reference qualification remains sampled, not a full-H128 continuum certificate.
Before model evaluation, replay one reference transition from steps 0, 16 and
32 of the first two frozen identities. Compare halved internal timestep/CFL
and doubled spatial resolution: require exact base replay, temporal relative
error at most 1e-5, spatial relative error at most 1e-3 and discarded fine-grid
content at most 1e-3. A numerical failure stops the run; do not replace paths.
One consolidated confirmation table/panel should extend Sections 4--5; no new
method-by-PDE grid. Retain all outcomes and stop after this fixed comparison.

#### Retained September 22 specification (completed; not an active queue)

September 22 continuation: run two additional paired fine-tuning seeds, 18 and
19, for the completed Section 5 recipe (`kf_bias_replication_20260922a`). Together
with seed 17 this gives three paired fits, all conditional on the same frozen
bank-DYN teacher. For seed k, the two clean sampler seeds are k and 100k+1;
the response sampler seed is 100k+3 and is advanced in both arms. All data,
teacher responses, optimizer settings, 4,096-update terminal rule, unit response
weight and deployed maps remain fixed. This tests optimization-order sensitivity,
not independent end-to-end training. No other representative is refit.

Before these fits, the qualitative expectation is that preservation retains more
of DYN's unused finite responses, while clean-only reduces onset clean error
more. We expect the observed H8/H32 tradeoff to recur, but do not assume an H128
gain over DYN. Report each paired seed, all trajectories and incomplete outcomes;
do not select the best seed. If the H32 advantage changes sign across seeds,
narrow the intervention claim to a seed-dependent tradeoff. Even consistent
H32 improvement over clean-only does not establish superiority to original DYN.

Run the four fits sequentially on the selected workstation, followed by the
existing clean and signed-response assays and the unchanged full-candidate H8
forecast on the unchanged training/open-development inputs. Expected fitting
time is about eleven hours from the first pair's measured costs. Stop before
new autonomous outcomes. Independently reduce and freeze each numerical
prediction under the empirical qualification rule above before evaluating
H8/H32/H128 and physical fidelity.
Forecast failures remain visible and their subsequent outcomes exploratory.
Keep fitting-seed variation separate from variation across reused paths.

The untouched-population confirmation is still a separate, unselected scope.
Neither these repeats nor their reused development assays open a protected role.
While fits run, complete the common-reference field display and numerical-method
details from existing source/manifests. These tasks fill current paper gaps
without changing the scientific comparison.

Repeat the central mechanism/design contrast with proportionate paired fitting
replication and a separately agreed untouched evaluation population. Size this
after the development effect and actual method costs are known; do not impose
an automatic full seed matrix. Keep model-seed and trajectory uncertainty
separate. The four reused development paths do not provide fresh-population
confirmation. Protected roles stay closed until their named scope is agreed.

The minimum display sequence is discrepancy -> representative response/bias and
cost comparison -> predicted versus observed intervention effect -> fidelity
and limits. Reuse one compact ODE illustration and the favorable NACA geometry
contrast; Bump enters only for a distinct qualified point. Fill the manuscript's
red TODOs with evidence or explicit scope statements, then finalize the abstract,
claims and reproducibility details. Do not put execution histories in the paper.

### Effort And Stopping Rules

The initial evaluation-only diagnostic and both causal contrasts are closed.
Reuse existing loading, response and rollout accounting for the next selected
method; preserve historical phases. No generic experiment framework, new review
loop or data migration is required.

Measured reference costs on the selected workstation are about 1.8 hours per
completed 4,096-update fit, 13.5 minutes per existing clean/response assay and
63--68 seconds per twelve-path rollout. These are not cost estimates for full
unrolling, solver labels or diffusion. Time their relevant operations once,
then price the selected comparison and confirmation work. The completed frozen
replay/response stages took 7.89 wall minutes. A decision-critical trusted or
geometry check, if needed, requires its own cost estimate. The owner-selected
draft dates are in [WEEKLY_RESEARCH_PLAN.md](WEEKLY_RESEARCH_PLAN.md); they do not
override scientific adequacy or protected-population boundaries.

End each block with its paper panel, interpretation and next decision. Essential
work is applicable mechanism coverage, the predictive design test, proportionate
confirmation and manuscript completion. Optional improvements are a new named
algorithm, extra geometrical reconstruction, additional PDEs, a larger benchmark
or additional strength sweeps. No calendar or new experimental outcome is
created by this planning refinement.

## Completed Comparison: Retained Recipe And Hypotheses

## Question And Working Regime

With the clean dataset fixed and sufficiently rich to identify nearby reference
behavior, which corrections improve autonomous prediction, through what response
change, and at what cost to accurate dynamics? Severe clean one-step overfitting
must not be the dominant explanation. Data size is not an experimental variable.
Clean one-step error alone does not certify Jacobian accuracy or a uniform tube.

The first comparison asks whether recovery acquired from perturbed clean inputs
transfers to endogenous errors and improves self-composition, beyond extra clean
optimization. It is a mechanism experiment, not a new baseline-error forecaster.
The eventual paper needs a quantitative intervention-design prediction and its
independent confirmation; this first paired fit does not itself complete that goal.

## Eligibility And Fixed Information

Use the existing 32-path Kolmogorov training population, fixed Clean32 terminal
49,152, original normalizer, PCNO architecture and complete output restriction.
All 32*512 clean pairs remain eligible for both arms. No new trajectory, solver
label, initial-condition law, architecture or data-count comparison is included.

The saved terminal metrics resolve the crucial onset window: clean relative L2
on inputs 0..7 is 0.08545% on all 32 training paths and 0.09835% on the four open
development paths. Existing H8 autonomous error is 5.6849% on the original eight
training paths and 5.7579% on those four development paths. Thus severe ordinary
overfitting does not explain the initial composition failure. The exact audit
and source identities live in the ignored current attempt and dated tracker.

Do not hide the later clean generalization gap: inputs 16..31 have approximately
a 2.17 development/training error ratio. Keep start 0, report H8 onset separately,
and retain H32/H128 application evaluation with the later limitation explicit.
Do not discard the transient or select a late easy window to obtain a winner.

## Selected Comparison And Frozen Recipe

| Map | Input and target | Purpose |
| --- | --- | --- |
| Untouched Clean32 parent | Existing fitted map; no updates | Reference for additional fitting and possible regression. |
| Matched clean continuation | Two clean-pair streams, each targeting its exact archived successor | Controls optimization, clean sampling and compute. |
| Online recovery | First stream clean; second stream u+eta targets the same archived clean successor | Changes response to displacement without extra trusted labels. |

Both fits start afresh from the identical parent, with fresh Adam at 1e-4,
betas=(0.9,0.999), eps=1e-8, zero weight decay, seed 17, 4,096 optimizer updates,
two batches of eight per update and equal branch weights. Independent
without-replacement stream seeds 17 and 1701 are replayed across arms. Each
stream visits all 16,384 training pairs twice. Use FP32, no AMP/TF32, clipping,
checkpoint selection, early stopping or extension. Retain the fixed terminal.
This is one paired optimization seed; trajectory variation is not seed replication.

Recovery perturbations are fresh mean-zero rectangular 2/3-band-limited Gaussian
fields. Their ensemble expected RMS is 0.01 times the inherited training scale;
a separate CPU RNG with seed 1702 leaves pair sampling unaffected. Scale by the
retained-mode variance, not by each draw's norm; record realized amplitudes.
The scale matches the previously measured actual displacement near input 5
(0.010013 training-scale RMS). It is fixed from diagnosis before corrective
outcomes, not selected to improve a new rollout. The Fourier support defines a
perturbation law, not manifold-normal geometry. Gaussian-to-endogenous transfer
is a hypothesis, not an assumed property of recovery.

The original solver bank is not used: it starts at input 15 and includes newly
computed targets. Here targets are byte-identical archived clean successors.
Dynamics relabeling remains a framework family with extra information cost,
not an equal-data arm in this first comparison. Projection, multistep exposure
and learned explicit refinement remain possible distinct mechanism contrasts;
none is an automatic next training queue.

## Measurements And Expectations Before New Outcomes

Let b=Psi(u)-u_next, O=[Psi(u+eta)-Psi(u-eta)]/2 and
E=[Psi(u+eta)+Psi(u-eta)]/2-Psi(u). The paired recovery loss is exactly
||b+E||^2+||O||^2. The actual positive-direction error is ||b+E+O||^2;
antithetic averaging removes an alignment term that matters for rollout.
Measure these finite-amplitude quantities; do not substitute a derivative gain.

| Question | Frozen directional hypothesis | What a contrary result means |
| --- | --- | --- |
| Acquired recovery? | Paired loss on unused Gaussian inputs is lower than matched continuation. | The chosen objective/budget did not generalize even to its declared perturbation law. |
| Relevant transfer? | Positive-direction error on common native parent-error inputs is lower, with clean forcing separately measured. | Gaussian response learning did not reach the endogenous direction, or clean bias outweighed it. |
| Useful composition? | Recovery/continuation H8 and H32 pooled squared-error ratios are below one. | Local response improvement did not translate through composition or damaged useful dynamics. |
| Limits and costs? | Report clean accuracy, longer-horizon outcomes and physical distortion, including harm. | A smaller response gain alone cannot justify the intervention. |

No evidence yet supports a credible percentage rollout improvement. Do not
invent one. This experiment tests acquisition, transfer and application effect;
a later magnitude/strength prediction must have its own measured justification.

After fitting, run a separate assay phase before inspecting new rollouts:

- Clean teacher errors for all 32 training and four open development paths,
  retaining per-pair sums for the onset and subsequent time windows.
- Shared signed Gaussian probes with unused seed 90017 and native parent-error
  probes at inputs 4,5,6,31, on the original eight training and four development
  paths. Same physical inputs and exact clean successor targets for every map.
- Forcing, odd/even response, signed alignment, response gain, paired and positive
  recovery scores, and realized displacement amplitudes. Save response vectors
  for independent recomputation. No new recurrent outputs enter these assays.

Then evaluate unchanged deployment from initial step 0 on those twelve paths,
through H128. Report H8/H32/H128 pooled relative L2 and per-path variation,
error curves, mean vorticity, kinetic energy and enstrophy errors. Keep the
existing numerical amplitude guard at 1e6 times training scale and honest
censoring; it is not an accuracy tolerance. Preserve all failures. Retain onset
and selected later state snapshots without storing redundant full trajectories.
These open paths support development evidence, not fresh-population confirmation.

## Execution And Completion Of This Bounded Step

Use one lean, hash-bound float32 repacking of the fixed clean states, exactly
matching the original fit's concatenated state-store identity. Training and open
development are separate files/manifests; fitting reads training only. Old data,
source snapshots, checkpoint bytes and result packets remain unchanged.

A separate 16-update resource run qualifies batch memory, numerical behavior
and runtime. Discard that model; start both fits from the original parent.
Proceed if the recipe fits with GPU headroom and projects below three hours per
arm; otherwise revise resource implementation without silently changing the
scientific recipe. Run arms sequentially on the available GPU. No repeated
resource permission or external review is required by the current owner direction.

This step is complete when both fixed terminals, mechanism assays and autonomous
comparisons are verified and interpreted, including unfavorable outcomes. Use
that result to select the next distinct mechanism/strength test, not to launch
an architecture or data-scaling sweep. Independent confirmation and manuscript
integration remain subsequent work packages. The tracker owns exact execution
status; do not report a launched or incomplete fit as a scientific outcome.
