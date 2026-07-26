# Research Direction Decision: Four-Line Transition To 2D Macro-Solvers

Date: 2026-07-12
Updated: 2026-07-23
Status: Accepted working direction

## Decision

The immediate objective remains a credible neural-operator-based algorithm for
medium-horizon, open-loop prediction of time-dependent PDEs before adding data
assimilation. The project is now organized into four coordinated research
lines:

1. close the remaining 1D large-timestep question and extract a small theory
   package about causal reach and learned finite-time flow maps;
2. audit and explain the 1D/2D CPGNet mechanism before treating the released 2D
   result as solver evidence;
3. make a geometry-aware neural-operator macro-solver roll out reliably in 2D;
   and
4. determine whether discontinuity-aware latent coordinates support sharp,
   stable open-loop rollout and, only after that forecast gate passes,
   statistically meaningful data assimilation.

Line 3 is the flagship algorithmic line. Line 2 is a forensic mechanism line
that can change how Line 3 uses CPGNet. Line 1 is a bounded 1D closure and
theory-seed line, not a new open-ended 1D architecture program. Line 4 is a
cross-cutting representation line: it studies the encoder-transition-decoder
contract as the causal variable and consumes Line 3's shared dynamic-testbed
contract and frozen physical-space baseline rather than duplicating Line 3's
shock stabilization work. A learned 2D pilot waits for Line 3 to authorize the
dynamic reference as training truth.

Data assimilation remains central to the long-term program, particularly for
chaotic systems. It will not be used to compensate for forecast defects already
visible on nonchaotic or weakly chaotic benchmarks.

The completed Idea 2.1 target-family program remains the empirical foundation:
it treats state coordinates, predicted quantity, supervision graph, and
enforcement mechanism as separate causal axes. Its 1D evidence is now mature
enough to stop broad target and structure sweeps. Future structure-preserving
work must answer a named 2D mechanism rather than repeat rejected 1D variants.

## Position In The Grand Plan

The long-term framework is a learned numerical forecast-analysis system:

1. A neural operator predicts a solver-facing quantity such as a macro residual,
   time-integrated numerical flux, interface predictor, closure, or dissipation.
2. A numerical scaffold converts it into a forecast state while enforcing
   selected structure.
3. Reliability diagnostics expose nonphysical behavior, forecast error, and
   distribution shift.
4. Data assimilation conditions later forecasts on observations once the
   forecast model itself is credible.
5. The resulting framework supports many-query inverse problems, control, and
   design optimization.

The forecast layer has the factorization

z_n = T_theta(U_n, parameters, geometry, dt),

U_(n+1) = Phi_dt(U_n, z_n).

T_theta may use global neural-operator context while Phi_dt is a local or
structured numerical decoder. One feed-forward evaluation is still an explicit
evaluation of a learned finite-time map, even when it spans many reference-
solver substeps. Input-output behavior alone does not identify an implicit
numerical scheme or nonlinear solve. Likewise, an unconstrained latent is not a
physical interface state merely because a flux decoder consumes it.

Line 4 introduces a distinct latent-state factorization. To avoid confusing the
latent state with the solver-facing quantity `z_n`, write

a_n = E_psi(U_n, geometry),

a_(n+1) = G_theta(a_n, parameters, geometry, dt),

Uhat_(n+1) = D_phi(a_(n+1), geometry).

The latent state may combine front or phase coordinates, a coarse conservative
spatial field, and learned local features. A fixed finite-dimensional latent
vector defines a reduced dynamical model; it earns a neural-operator transfer
claim only when the complete encoder-transition-decoder contract is shown to
transfer across discretizations or geometries.

## Why Forecasting Comes First

- The current 1D Euler and supersonic-bump failures are not established
  consequences of chaos. They expose one-step/rollout mismatch, shock-local
  degradation, target non-identifiability, and time-discretization mismatch.
- Assimilation could make a poor forecast appear usable by repeatedly resetting
  it, obscuring what the model learned and why it failed.
- A trustworthy assimilation study needs a fixed forecast model whose open-loop
  error growth, physical defects, and predictability horizon are already known.
- Medium-horizon rollout is long enough to expose accumulated numerical error
  without demanding pointwise prediction beyond a chaotic predictability limit.

Open-loop rollout remains a diagnostic after assimilation is introduced. Later
chaotic-system evaluation will distinguish short-range forecast skill, filtered
state accuracy, and long-time statistical fidelity.

## What Current Work Has Established

- Teacher-forced one-step accuracy is not a reliable measure of autoregressive
  stability.
- Errors in the available local-bundle CPGNet run are dominated by shock-local
  phase, shape, amplitude, and stability defects rather than simple one-step
  underfitting. Dataset and checkpoint parity with the paper remain unresolved.
- Instrumented local bs2 reproduction checkpoints use interface latents as
  nonphysical flux-control coordinates rather than verified one-sided physical
  traces. This has not been established on an identified paper checkpoint.
- Residual, flux, and interface parameterizations with similar one-step errors
  can have sharply different rollout behavior.
- Input noise can regularize residual rollout, but may act through denoising or
  damping rather than improved wave physics.
- The legacy toy CPG-style rows are diagnostic failures, not CPGNet evidence.
  Their interface rollouts survive mainly because a whole-sample limiter
  suppresses most proposed updates.
- Conservation-form decoding, positive interface variables, or a limiter alone
  do not guarantee an accurate or stable learned scheme.
- On the fixed 1D Euler stride-4 dataset, corrected CPGNet becomes a competitive
  raw macro-step solver when message depth is raised from 12 to 28: completion
  rises from 34/64 to 64/64 and the effective-CFL/error correlation disappears.
  Parameter-matched wide-shallow and narrow-deep controls support receptive
  field rather than parameter count as the primary mechanism. Its remaining
  shock tail is a persistent stationary ghost front, magnified but not created
  by argmax front detection.
- Exact accepted-substep ADER face impulses close the saved state transition and
  can supervise a conservative flux-form FNO without an implementation-level
  label mismatch.
- In 1D, removing the one constant-flux null mode makes the remaining face field
  information-equivalent to the state increment. Its MSE changes the residual
  norm toward an inverse-divergence, low-frequency weighting; it is not an
  independent physical target after projection.
- The first midscale solver-flux screen confirms that distinction: the
  gauge-canonical joint row reduces divergence-active face error but worsens
  held-out state error and fails all paired rollouts earlier than state-only.
  Better native-target fit is therefore not sufficient evidence for a better
  learned solver.
- Short differentiable recurrence is a useful optimization intervention for the
  same conservative flux-form FNO. At 64/16/16 scale it improves one-step fit,
  raises raw survival by `1.464x`, and yields the first 2/16 completed rollouts.
  A matched smooth admissibility barrier neither improves survival nor reduces
  the 14 remaining nonpositive terminations, so the supported mechanism is
  recurrent-distribution training rather than that barrier.
- At the full 384/64/64 scale, the same recurrent recipe reaches one-step
  relative L2 `0.001305`, mean survival `0.97734`, and 57/64 completed 20-call
  raw rollouts at mean initial effective CFL `3.84`. It passes the scale and
  20-call promotion gates but misses the strict 90% completion gate by one
  trajectory. Seven nonpositive proposals remain, so this is not evidence that
  positivity constraints are unnecessary in general.
- The frozen flux checkpoint does not pass the actual 50-call capability gate.
  Only 1/64 cases completes 50 calls, mean survival is `0.566`, and no case
  reaches 100 calls; all terminate on raw nonpositive proposals by call 51.
- Direct next conservative state and conservative residual are label-valid and
  information-equivalent, but not optimization-equivalent. Direct state misses
  the tiny-fit threshold on both declared seeds, while residual passes. At
  64/16/16, recurrent residual reaches one-step relative L2 `0.002649`, 16/16
  completion, and zero nonpositive failures, versus `0.003438`, 2/16, and 14
  for the matched flux head. This supports output centering and the identity
  bypass as an important target-parameterization effect.
- At full 384/64/64 scale, recurrent residual reaches one-step relative L2
  `0.001123` and completes 64/64, 62/64, and 61/64 raw rollouts at 20, 50, and
  100 calls. Mean 100-call survival is `0.97781`; the three failures are
  interior pressure collapses at calls 33, 37, and 91, with no density or
  nonfinite termination. Residual is lower-error than the matched flux head on
  63/64 cases at their longer common endpoint. This establishes a strong
  fixed-setting baseline, but not a seed-, stride-, resolution-, or
  architecture-independent target ordering.
- The matched three-seed residual/projection screen resolves the first seed
  gate. Plain residual completes 182/192 pooled 100-call rollouts; projected
  residual completes 187/192 and closes exactly to a learned low-dimensional
  boundary budget. That stability gain does not become a robust accuracy gain:
  the projected three-seed mean state-error ratio is `1.016`, and among 181
  common completers its state L2 is 9.8% worse, conserved-total error 5.9%
  better, and shock MAE 4.8% worse. Only stability noninferiority passes the
  preregistered family gates. Projection is therefore a structural ablation,
  not the replacement accuracy baseline.
- A matched 64/16/16 generated-burn-in pilot on projected residual gives a
  strong but nonuniform distribution-exposure signal. With identical one-step
  histories and near-matched recurrent update counts, eight detached burn-in
  calls before four-step BPTT reduce 50- and 100-call state L2 by 23% and 36%.
  It wins 15/16 and 14/16 paired common-endpoint comparisons, respectively.
  The primary gate still fails because 20-call shock MAE rises 38% and 100-call
  survival drops from `0.9650` to `0.9394`: one clean failure is rescued, one
  new call-58 failure appears, and a shared failure moves three calls earlier.
  This supports later-state exposure as an error-drift intervention, not as a
  complete stability mechanism or a reason to promote projected residual.
- The same generated-exposure gate on plain residual gives a stronger
  medium-horizon signal. It lowers final state L2 by 4.6% and 32% at 20 and 50
  calls; at 100 calls it lowers common-endpoint state error by 51.4%, wins
  14/16 cases, and raises completion from 15/16 to 16/16. Conserved-total error
  also improves at all horizons. The preregistered gate remains formally
  failed because the legacy pressure-argmax shock MAE rises 45.6% at 20 calls.
  Focused replay separates two front-strength rank changes from one genuine
  spurious/displaced-front regression. Generated burn-in therefore remains a
  promising recurrent-distribution intervention, but a teacher-offset control
  is required to distinguish off-manifold exposure from later-time sampling.
- The matched teacher-offset control resolves that ambiguity on the fixed
  64/16/16 split and seed. Later-time sampling alone is 10.9% worse than clean
  at 20 calls and tied at 50/100 calls. Generated exposure beats teacher by
  13.9%, 33.9%, and 50.8% in common-endpoint state error at 20/50/100 calls,
  wins 15/16 H100 cases, and completes 16/16 versus 14/16. The bootstrap
  intervals exclude parity at all three horizons. Thus model-generated
  off-manifold exposure contributes beyond time offset in this setting.
  Method promotion remains blocked: generated versus clean top-two
  front-position error is 17.8% worse at H20 and 10.2% worse at H50, and the
  original-trajectory label is not the reference PDE advance from a perturbed
  generated state.
- The restriction-consistent resolution gate separates shared representation
  from grid-dependent numerical targets. One equal-presentation 64/24/4 FNO
  stays within `1.296x` same-grid-oracle one-step error and `1.418x` H20/H50/
  H100 state error on exact 128/256/512-cell restrictions, never loses
  completion, and improves frozen off-grid one-step error by 68.1%/78.3% at
  nx128/nx512. The same weights do not reproduce independently evolved native
  coarse maps: native nx128/nx256 one-step error is 5.27/3.65 times the
  restriction-consistent value. D029 was therefore substantially a target-
  consistency problem, not an FFT-capacity result. The shared model still loses
  three pressure-limited cases by H100, so neither native-solver invariance nor
  positivity-free medium-horizon reliability is established.

These results motivate method design but do not show that directly supervised
flux or interface targets outperform residual prediction.

## Central Scientific Hypothesis

The learning target is a first-class algorithmic design variable. Target choice
acts like coordinate selection or preconditioning in operator space: it changes
identifiability, optimization conditioning, inductive bias, and error
amplification through the numerical decoder.

> Solver-facing learning improves medium-horizon prediction when the target is
> identifiable, aligned with the macro time integrator, representable by the
> decoder, and coupled to appropriate local admissibility control.

This conditional hypothesis is more defensible than assuming that flux or
interface prediction is intrinsically superior.

## Optimization Lens

There is no architecture-independent ordering of easy and hard targets. For a
fixed model and optimizer, target difficulty depends on the target's alignment
with the model parameterization, the loss metric, the data distribution, and
the supervision path. Representability does not imply that gradient descent
will find the represented operator on the available budget.

For a flux head linearized as `F_theta = F_0 + J delta_theta` and a conservative
decoder `U_next = U - A F_theta`, the local Gauss-Newton matrices are

`H_flux = J^T J`,

`H_state = J^T A^T A J`.

State-through-divergence supervision is blind to `ker(A)` and weights visible
flux modes by the singular spectrum of `A`. Direct flux supervision changes
both identifiability and conditioning. This is a local optimization statement,
not a proof that full flux loss gives a better solver: an exactly
divergence-free error is harmless under the same fixed decoder, and direct
loss may waste capacity on a solver-specific gauge.

The primary conditioning diagnostic should distinguish

1. raw decoded-state loss, `||A e||^2`;
2. whitened divergence-active or projected loss,
   `||(A A^T)^(dagger/2) A e||^2 = ||P_range(A^T) e||^2`; and
3. full direct-flux loss, `||e||^2`.

Raw versus divergence-active loss tests conditioning. Divergence-active versus
full flux tests whether selecting the reference solver's nullspace
representative adds useful information. These controls are analysis tools; do
not add all three to the first production matrix if the simpler comparison is
already decisive.

Every apparent failure must be classified before changing the method:

- failure to overfit a tiny supervised subset indicates an implementation,
  effective-capacity, scaling, or optimization problem;
- cold-start failure followed by a lower final-target training-loss floor after
  a horizon curriculum supports continuation-sensitive optimization only when
  the final-stage and total-exposure cold controls below are satisfied;
- low training error with poor held-out error indicates generalization, data
  coverage, or closure failure rather than optimizer failure;
- different reference labels for nearly identical model inputs indicate a
  non-Markovian or underresolved closure, for which deterministic MSE learns a
  conditional mean; and
- good teacher-forced and held-out one-step error with bad free rollout
  indicates recurrent distribution shift or unstable error propagation.

Record observations first and mechanisms second. Kernel alignment, spectral
bias, gradient conflict, shock rarity, and denoising are hypotheses until an
intervention changes the predicted optimization behavior.

### Cross-Line Research-Process Correction (2026-07-22)

The combined Lines 1--4 evidence does not support the shorthand conclusion
that capacity and physical inductive bias are irrelevant and optimization is
the remaining bottleneck. The project must distinguish at least six causal
classes before changing a model:

1. supervised fit or centering, as in the direct-state tiny-fit failure;
2. objective/decoder conditioning, as in D048's successful face-space fit and
   catastrophic divergence-amplified state decode;
3. recurrent objective mismatch, when one-step validation improves while raw
   rollout worsens;
4. representation-manifold failure, as isolated by L4A-003's per-state decoder
   oracle;
5. geometry-conditioned generalization, as exposed by D057--D058's repeated
   lower-boundary `y00` conflict; and
6. implementation or metric binding, as exposed and preserved by D046 and
   D050.

Every learned proposal must therefore predeclare a mechanism variable, the
expected diagnostic curve if that mechanism is active, the smallest matched
control, case strata, promotion and kill thresholds, the strongest alternative
explanation, seed scope, and full cost. Every threshold must bind an exact
artifact field, aggregation, population, and direction before output is
inspected. Learning curves must place the optimized loss beside raw physical
rollout, front, admissibility, and budget metrics. If
the proposal names a gate, attention map, latent variable, local branch, or
physical coordinate, a readout or intervention must verify that the trained
model actually uses it. A lower reconstruction or target-space loss cannot
promote a row when front strength, thickness, decoded conditioning, closure, or
recurrence fails.

The current routing is deliberately narrow. Lines 1 and 2 are closed evidence
lines. D060 closes Line 3's temporal-step comparison as a partial mechanism
result without promotion: fewer calls lower state error, but not ripple energy
or every front-position metric. D061 then rejects scalar multirate state
blending: it misses the state/ripple gates and is jointly nonworse plus anti-
smearing in 0/6 H60 cases. Its positive disagreement-localization readout does
not overcome D056's correction-realizability failure. No multirate tiny fit or
other learned Line-3 continuation is authorized. Line 4 remains stopped before
serious representation training because its smooth-decoder oracle and fixed
discontinuous chart both fail capability gates. Any later 2D
latent reopening requires a new, separately registered discontinuous-decoder
or front-chart oracle, followed by closure; neither a learned transition nor
data assimilation can precede those passes.

D062 closes the registered zero-training exception without promotion. Its
target-informed row-wise pressure-jump chart, exact conservative remap, and
four strength coefficients preserve row totals and improve the scalar
front-curve error, yet worsen state and high-pass error and pass no H60 joint
case. Under stronger shock--vortex interaction, comparable pressure-jump
branches exchange argmax identity in isolated rows; the resulting phase field
has jumps of up to `10.51` cells between adjacent rows. This is a
representation-contract failure before learning, not an invitation to tune
the encoder or optimizer. It rejects this exact chart without rejecting every
level-set or multi-chart representation.

No learned or oracle method experiment is now authorized. The four-line causal
synthesis below is the completed next action: it states what has been ruled
out, which mechanism evidence transfers, and what remains missing. Any later
reopening needs a separately approved zero-training oracle that defines
connected front identity and transverse regularity before results are visible;
transition training and data assimilation remain downstream of reconstruction
and closure.

#### Immediate Frozen-Artifact Audit Before Any New Training

| Line | Mechanistic question and expected observation | Frozen answer | Decision |
| --- | --- | --- | --- |
| 1: macro flow maps | If fewer compositions are the benefit, truth-state one-call error should worsen with stride while on-policy long-horizon error eventually improves; front and roughness metrics need not share the L2 crossover. | D033/D035/D038 and D060 show exactly that harder-map/fewer-calls trade. The winner changes with horizon and metric; large stride injects a rougher early defect even when later global L2 is lower. | Preserve the operating-envelope result. Do not search for one universally optimal stride or infer ripple control from endpoint L2. |
| 2: CPGNet mechanism | If causal reach rather than width is active, a deep narrow control should retain the gain and a wide shallow control should not. If interface codes are physical traces, physical projection should preserve the update. | The mp12/h193 and mp28/h85 controls support hop coverage as the 1D mechanism. Physical projection of the interface code destroys the learned update, and legal-boundary 2D evaluation remains materially worse than oracle evaluation. | Describe CPGNet as a feed-forward macro map with useful causal reach and functional, nonphysical interface coordinates; do not call it an implicit scheme or reopen its architecture sweep. |
| 3: structured PCNO | If a named branch or local support causes the failure, a matched intervention should improve the targeted metric without damaging the state/front hierarchy. If optimization is the only issue, a common training descent direction should transfer across geometry groups. | Branch attenuation passes no complete D052 row; D053 separates propagated shock-supported state error from freshly generated smooth high-pass error; D055 locates the latter but D056 cannot correct enough safely. D057/D058 directions are valid on training tasks yet repeatedly fail late `y00` clean-state transfer. D048 fits face targets while divergence amplification destroys decoded states. D060 lowers recurrent state L2 but not high-pass error; D061 cannot combine its gain with D044 by state averaging; D062 conserves exactly and aligns a target-informed row front while worsening the field through branch switching. | The bottleneck is split across target/decode conditioning, recurrence, geometry-conditioned transfer, and representation identity. Stop local repairs, multirate composition, this front chart, and static joint-objective continuation; do not summarize the evidence as generic optimizer failure. |
| 4: latent representation | If encoder optimization is the main problem, fitting each held-out state directly in the frozen decoder code should recover a sharp admissible front. If discontinuous decoder regularity matters, the fixed Haar control should improve the front hierarchy. | Direct code fitting still leaves nine of ten states above the L2 gate and broadens/weakens fronts. Haar improves L2, strength, thickness, and IoU, so decoder regularity matters, but it remains far outside the physical gate and below the privileged fitted-code control. | The tested chart is capacity-limited before forecast optimization. Do not train a transition or filter; a future reopening needs a qualitatively new front/discontinuous chart oracle first. |

#### Frozen Cross-Line Claim Matrix (completed 2026-07-23)

| Claim candidate | Optimized object and learning evidence | Matched intervention and evidence grade | Systematic strata and robustness | Supported claim | Missing evidence and live alternative |
| --- | --- | --- | --- | --- | --- |
| Fewer macro-map compositions can outweigh a harder one-call map. | Conservative-residual next-state objectives under the fixed D031 schedules and the matched D060 contract. Learning curves select frozen checkpoints; they are not themselves mechanism evidence. | D033 same-state decomposition, D035/D038 frozen evolution, and the stride-only D060 comparison; matched ablation plus frozen causal decomposition. | Sixty-four 1D test cases through H96, three stride-8 seeds, 24 2D position-OOD validation cases through H60, and the six-case D013 subset. The 2D row has one seed and sealed strength OOD. | The operating point trades one-call approximation difficulty against recurrence count. Lower endpoint state error can coexist with a rougher early defect and worse front or high-pass metrics. | No universal best stride, learned CFL limit, or ripple cure follows. Independent 2D seeds, strength OOD, other geometries, and longer uncensored horizons remain missing. |
| CPGNet stability depends materially on message reach, while its interface coordinates are functional rather than verified physical traces. | Primitive next-state loss through the fixed message-passing/interface decoder. Parameter-matched controls use the frozen training schedule; no paper-level sample-efficiency claim is available. | The mp12/h193 and mp28/h85 depth-width controls, physical projection of learned interface codes, and legal-boundary training; matched ablations and decoder interventions. | The corrected 1D 384/64/64 split and 64 raw test rollouts, plus one legal-boundary-trained seed on 20 release-bundle 2D trajectories. Paper dataset/checkpoint identity and exact DG replay remain open. | Hop coverage is the supported 1D mechanism, not width alone. Physical projection does not preserve the learned update, and boundary training helps without closing the legal-versus-oracle gap. | This does not establish a paper-faithful CPG mechanism, an implicit scheme, physical interface states, conservation, or transfer of the architecture to PCNO. |
| Serious-PCNO ripple formation is composite rather than a pure Fourier/Gibbs, pointwise-only, cancellation, or boundary-only effect. | Conservative-variable residual state loss for D044/D060. D041--D043 and D052--D053 add no optimization and read frozen branch and recurrence responses. | Branch-gain intervention, paired branch-response energy identity, and teacher/rollout source decomposition; causal intervention plus matched frozen diagnostics. | Twenty bump holdouts for boundary/failure routing and six dynamic-FV validation trajectories over early, middle, and late calls. The dynamic result is one model seed on one family. | The dominant state error is largely propagated and shock-supported, while smooth-region high-pass error is freshly injected by the one-call map. The spectral branch is globally coupled and comparatively smooth; no isolated branch explains the failure. | The exact layerwise source and any true spectral-aliasing contribution remain unresolved. Adaptive local bases, different global maps, and transfer beyond the frozen family have not been tested. |
| Native face-target fit is not sufficient for a viable conservative update. | D048 directly minimizes the canonical shared-face `W_f^{-1}` target loss for 3,200 updates; 15 epochs satisfy all native face-space gates. | Exact discrete-divergence replay of the fitted face field; direct objective intervention. | Four immutable training pairs only, with validation excluded from selection and test sealed. | A roughly 10% face-field error can become a 1001--1581x relative divergence amplification, non-admissible decoded states, and state error far worse than persistence. | This rejects the exact canonical face-value objective, not flux prediction in general. Divergence-conditioned/projected supervision, a different decoder, and full reference-impulse supervision remain scientifically distinct but are not authorized continuations. |
| A sparse legal locator for fresh ripple does not imply a safe local correction exists. | D044, D055, and D056 are zero-training capacity tests; learning-curve sample efficiency is not applicable. | Target-informed corrections on a frozen proposal-selected support, with fixed support, balance, update-norm, admissibility, and anti-smearing constraints; constrained oracle intervention. | Six validation trajectories at calls 1/10/30/60, with strength OOD sealed. | The proposal locates fresh high-pass energy, but the bounded correction cannot remove enough late state and high-pass error without violating the physical hierarchy. | This rejects that support-limited detail head, not every conservative local correction or an end-to-end change of the global map. |
| One static joint-objective direction does not transfer safely across geometry groups at late horizon. | D057/D058 take zero optimizer steps. They form clean-state, smooth-high-pass, and generated-state gradients under fixed objectives. | Equal-gradient and eight-task minimum-norm directions; matched gradient intervention. | Four-pair then full 84-trajectory training construction, followed by six position-OOD validation cases at calls 30/60. Late `y00` clean-state cosines fail repeatedly while `y08` remains positive. | The objectives are jointly descendable on training data, but the frozen full-model direction is geometry-conditioned and not a safe continuation. | Finite-step optimization, state-dependent weighting, and other task parameterizations are untested; the result is not a generic optimizer-failure claim. |
| A row-wise scalar front coordinate is not a stable representation through shock--vortex interaction. | D062 is a zero-training, target-informed capacity oracle with an exact conservative finite-volume remap and four fitted strength coefficients. | Phase-only and phase-plus-strength decomposition under exact row-total conservation; target-informed oracle intervention. | Twelve frozen validation rows from six trajectories at H30/H60. The failure is weak for `e00` and large for interacting `e06/e11` rows, with up to 10.54-cell displacement and 10.51-cell adjacent-row jumps. | Comparable pressure-jump branches exchange argmax identity. Improving the scalar front error can shear the 2D field, increase high-pass error by an order of magnitude, and worsen state accuracy despite exact conservation and admissibility. | This rejects the exact independent-row argmax chart. A connected multi-front atlas, level-set representation, or topology-aware shock chart first needs a separately registered zero-training identity and closure oracle. |
| The tested latent family is decoder-capacity limited before transition learning. | L4A-002 uses an 800-update matched autoencoder smoke; L4A-003 fits each held-out code directly; L4A-004 is a zero-training fixed-Haar capacity test. | Matched generic/conservative decoders, privileged per-state code fitting, and a fixed discontinuous-basis intervention. | Ten position-OOD validation states from two trajectories; strength OOD and test remain sealed. | Exact moments improve budgets but smooth decoders blur and weaken fronts even with privileged codes. Discontinuous regularity improves L2 and front fidelity, yet the fixed 16-detail chart remains far outside reconstruction and sharpness gates. | No latent transition, recurrence, geometry transfer, or assimilation has been tested. A qualitatively different discontinuous chart must pass reconstruction and closure first. |
| Physical conservation claims remain conditional on the validated finite-volume data contract. | D037 validates volumes, oriented faces, normals, boundary accounting, and cumulative accepted-substep impulses. D044 predicts state residuals; D062 only post-processes frozen states. | Contract verification rather than a learned-method intervention. | The Mach-1.1 shock--vortex family and its audited 250x100 finite-volume mapping. | Physical totals and boundary exchange are valid outcome diagnostics on this benchmark, and D062 preserves its post-hoc row totals exactly. | Neither D044/D060 recurrence nor D062 is a predicted-flux conservative neural solver. Equal-node bump sums and unvalidated bump quadrature still cannot support physical 2D conservation. |

The transferable findings are methodological rather than architectural:
target-space fit must be checked after the physical decoder; fewer recurrent
calls and lower state error do not imply less roughness; conservation is
necessary but does not identify a correct shock representation; and a proposed
latent, gate, branch, or front coordinate needs an intervention showing that it
retains identity and improves the full physical hierarchy.

The non-transferable findings stay equally explicit. CPGNet's reach result does
not make PCNO local, the bump evidence does not inherit the dynamic benchmark's
finite-volume contract, the failed 1D or four-pair flux rows do not reject all
2D flux models, and no validation-only oracle result licenses test, strength
OOD, or cross-geometry claims.

This completes the frozen-artifact synthesis. It authorizes no new run. The
next research action is a direction decision: close the current campaign around
the bounded mechanism results, or separately preregister a zero-training
capacity oracle for a representation with connected front identity and
transverse regularity. AutoDL access does not relax that scientific gate.

#### Result-to-Claim Gate (completed 2026-07-23)

Overall verdict: `partial`, with high confidence. The intended method claim --
that the current campaign identifies and validates a generally improved,
shock-stable geometry-aware neural-operator solver -- is not supported. The
bounded scientific claim -- that a serious residual PCNO is useful on one
validated dynamic finite-volume family and that matched interventions separate
several distinct failure mechanisms -- is supported within the frozen scope.

| Intended claim | Verdict | Decisive evidence | Allowed report wording | Missing evidence or disqualifier |
| --- | --- | --- | --- | --- |
| D044 is a useful fixed-family 2D macro-map baseline. | yes, bounded | One seed completes 24/24 raw H60 position-OOD validation rollouts at mean physical-volume state error `0.00834190` and beats persistence plus both train-manifold controls on every paired case. | A conservative-coordinate residual PCNO gives a strong raw-rollout baseline on the frozen Mach-1.1 shock--vortex family. | Strength OOD and test are sealed; seed uncertainty and other geometry/dataset families are missing. The recurrence is not conservative by construction. |
| D060 is a uniformly better or ripple-stable PCNO. | partial for state; no for the joint claim | D060 lowers H60 state error on 24/24 cases to `0.87434x` D044, but six-case high-pass RMS is `1.0255x`, all-24 high-pass energy is worse in every case, and front-centroid distance is `1.4351x`. | Halving recurrent calls improves accumulated state L2 while leaving the fresh high-pass source and some front errors unresolved. | One seed, no strength OOD, no extra stride, and failed physical conjunction. Do not call D060 a promoted solver. |
| PCNO ripple is caused primarily by Fourier/Gibbs behavior or one local branch. | no | The Fourier Gram matrix is well-conditioned; the spectral response is the smoothest branch; strong paired cancellation is absent; D052 finds no safe repeated branch attenuation; D053 separates full-state and high-pass sources. | Pure spectral, pure pointwise, pure differential, and strong cancellation explanations are falsified for the frozen checkpoint; the mechanism remains composite. | Exact layerwise causation and any residual aliasing contribution remain unresolved. |
| Long-horizon state error and smooth ripple have distinct formation channels. | yes, bounded | At calls 30/60 D053 gives median full-field propagated shares `0.88995/0.89043`, but smooth-high-pass shares only `0.15619/0.30733`; additive identities close to numerical precision. | On the frozen six-case cohort, shock-supported state error is carried mainly through recurrence while smooth-region high-pass error is predominantly regenerated by the one-step map. | One checkpoint, one shock-bearing family, and no matched smooth control. This is not a universal neural-operator law. |
| Native face-target accuracy is sufficient for a viable conservative solver. | no for the tested objective | D048 satisfies native face-space gates but yields no admissible state; post-hoc replay gives `1001--1581x` relative divergence amplification. D049 independently confirms frequency-dependent divergence conditioning. | Target-space error must be assessed after the physical discrete decoder; the tested canonical face-value objective is stopped. | No serious divergence-conditioned or full reference-impulse row was run. The result does not reject every flux representation. |
| A sparse gate, static joint objective, scalar multirate blend, or independent-row front chart supplies the missing stabilization. | no for the exact tested branches | D056 cannot correct enough safely after D055 localizes error; D057/D058 fail late geometry transfer; D061 is jointly nonworse in 0/6; D062 improves its scalar front metric while worsening state and high-pass error. | These necessary-condition failures rule out the exact registered branches before serious training. | End-to-end global-map changes, state-dependent objectives, connected multi-front charts, and other representations remain untested, not disproved. |
| The project has established a general shock-stable geometry-aware neural operator. | no | All serious 2D method evidence is one-seed and family-bounded; every proposed stabilization misses a predeclared joint gate; test and strength OOD remain sealed. | No general solver or state-of-the-art claim is allowed. | Multi-seed confirmation, new geometries, test OOD, and a promoted method are absent. |
| The completed work supports a mechanistic research contribution. | partial, high confidence | The validated benchmark and baseline, exact source decomposition, decoder-conditioning audit, systematic geometry strata, and failed capacity oracles form a coherent bounded account. | The contribution is an evidence-graded map of what works, what fails, and why several plausible shock remedies do not pass necessary conditions. | The central positive replacement architecture is missing, and discontinuity specificity is not isolated by a matched smooth control. |

The report should use one causal thesis: on this shock-bearing finite-volume
family, the residual PCNO has a useful global map, but its error is not governed
by one removable branch. Propagated shock-supported state error, freshly
generated smooth high-pass error, geometry-conditioned transfer, discrete-
decoder conditioning, and unstable front identity are distinct coupled
obstacles. This is stronger and more precise than describing the outcome as
generic instability, optimization failure, or Gibbs ringing.

The following limitations must remain visible rather than becoming implicit
follow-up authorization:

1. D044 bundles conservative coordinates, residual prediction, primitive
   training noise, and rollout-based selection; their individual 2D
   contributions were not isolated by matched trained ablations.
2. D044 and D060 are single-seed serious runs. Existing casewise repetition is
   not seed robustness.
3. Selected epochs 44 and 34 and the saved loss curves are training provenance,
   not a controlled sample-efficiency comparison between different targets.
4. Branch traces and proposal scores are not a learned hidden-state probe of
   stable front topology, future failure, or correction sign.
5. Every current 2D cohort is shock-bearing; discontinuity necessity remains
   untested against a matched smooth problem.
6. The sealed strength-OOD and test splits cannot be used to strengthen any
   current claim after method selection failed.

The active queue is report-only and uses frozen artifacts:

- show D044/D060 state, high-pass, front, and geometry-stratum curves together;
- show D053 propagated-versus-fresh shares over horizon beside the D052 branch
  falsification;
- show D048/D049 discrete-decoder amplification beside D062's front-identity
  failure;
- tabulate objective, presentations, selected epoch, intervention, population,
  evidence grade, and non-claim for every reported result; and
- retain failed implementation attempts only as provenance, not as scientific
  trials.

No item in this report queue executes a checkpoint, opens a sealed split,
changes a threshold, or consumes GPU time.

## Macro-Flux Target Contract

Use `cumulative conservative face impulse` for the solver-facing quantity

`I_f[n:n+K] = sum_(k in accepted substeps) dt_k Fbar_(f,k)`,

where `Fbar_(f,k)` is the solver's owner-oriented normal numerical flux density
averaged with the actual quadrature for accepted substep `k`. For an RK solver
this average is the weighted stage flux; for ADER it is the returned
space-time-predictor quadrature flux. Rejected retry attempts contribute zero.
The final accepted retry or first-order fallback contributes exactly once.

Use `time-averaged face flux` only for `I_f / Delta T_sample`, where
`Delta T_sample = sum_k dt_k` is the actual saved-interval duration. Avoid the
unqualified term `mean field`, which can be confused with an ensemble or
statistical mean.

The current 1D artifact stores conservative flux density times time, oriented
owner-to-neighbor on interior faces and outward on boundaries; the discrete
decoder owns the face signs and `1 / dx` factor. For higher dimensions, record
whether face measure is stored in `I` or in `D_h`, and never apply it twice.

Before training, require serialized closure to numerical precision:

`U_(n+K) - U_n = -D_h I[n:n+K]`,

where `D_h` includes the documented face orientation, geometry, mass, and
boundary convention. Do not separately add boundary terms when boundary-face
impulses are already included in `I`. The current 1D gate is maximum float64
pre-serialization closure below `1e-12`, followed by float32 decoded closure at
`rtol=3e-5`, `atol=3e-6`; any changed dtype or scale requires a documented
replacement tolerance before data generation.

Temporal averaging can suppress fast temporal oscillations, but it does not
guarantee an easier operator from the initial state. Larger horizons expand the
domain of dependence, increase sensitivity to unresolved state, and introduce
kinks when shocks cross faces or solver branches change. Horizon difficulty is
therefore an empirical axis, not an assumed benefit of averaging.

## State-Coordinate Decision

Status note, 2026-07-18: this section and the target-family section below
preserve the preregistered Idea 2.1 logic that produced the completed 1D
evidence. Their imperative and future-tense wording is a historical decision
record, not authorization for another broad 1D screen. The active extension is
the bounded 2D structured-target study under the Four-Line Research Program.

At preregistration time, the 1D and reference CPG paths consumed primitive
variables, used conservative updates for residual/flux/interface decoders,
recurred through primitive states, and optimized standardized next-primitive
loss. They did not answer whether conservative-variable coordinates improved
optimization or rollout.

The completed coordinate matrix kept the predicted quantity fixed as the
conservative increment and separated input coordinates from loss coordinates:

1. primitive input with primitive loss: current residual control;
2. conservative input with primitive loss: input-coordinate effect;
3. primitive input with conservative loss: loss-coordinate effect;
4. conservative input with conservative loss: fully conservative form; and
5. conservative plus primitive features with joint loss only after the causal
   four-way comparison.

The four-way comparison selected conservative input, conservative loss,
conservative recurrence, and fixed physical scaling as the provisional 1D
contract. This is a benchmark-supported contract, not a universal conclusion
that primitive or characteristic features are inferior.

New conservative-form variants should keep the recurrent state in conservative
variables and derive primitive features without hidden floors. Use fixed
physical nondimensional scales as the main normalization and retain empirical
standardization only as a documented control. A pure conservative formulation
is a hypothesis, not a foregone conclusion: conservative variables align with
weak-form updates, while primitive or characteristic features can make wave
speeds, pressure, and admissibility easier to represent.

Every run must record `input_coordinates`, `recurrent_coordinates`,
`predicted_quantity`, `loss_coordinates`, target normalization, timestep,
geometry scaling, and whether primitive conversion used a floor. Do not label a
conservative decoder trained through primitive loss as a conservative-space
model.

Apply the same failure taxonomy to the coordinate matrix. Select a provisional
coordinate contract from matched train/validation and rollout evidence, but do
not treat a coordinate as intrinsically unsuitable when it failed label,
tiny-set-fit, or optimization checks.

## Completed Target-Family Empirical Program Record

The completed 1D target exploration used theory for prior predictions,
controls, and falsifiers rather than removing a family before a valid test.
Its positive and negative results remain evidence only within the tested
coordinates, decoder, stride, data regime, normalization, and optimization
budget.

Ask two questions for every family:

1. Can the final model optimize the target? Report tiny-set fit, training-loss
   floor, convergence time, seed variance, full-batch versus minibatch
   behavior, and per-target gradient norms or conflicts.
2. Does a well-fitted target induce a better solver? Report held-out
   supervised-objective error, decoded-state error, direct-horizon error, raw rollout,
   shock, conservation, positivity, boundary, limiter, and transfer
   diagnostics. For directly supervised rows, also report native-target error.
   For the state-loss-only flux row, decoded-state loss is its native supervised
   metric; report divergence-active impulse error separately and label full
   reference-impulse error as gauge-dependent.

The core matched screen is

1. direct next conserved state;
2. conservative residual or increment;
3. a flux head trained only through decoded-state loss;
4. directly supervised cumulative conservative face impulse; and
5. joint cumulative-impulse and decoded-state supervision.

Rows 1 and 2 isolate output centering and the identity bypass for two
information-equivalent state targets. Rows 3 and 4 isolate the supervision
graph while keeping the flux-form architecture fixed. Use a common decoded
state metric so native target scaling is not mistaken for target superiority.
The joint row is a prespecified supervision control, not the later
dense-supervision stage.

Conditional second-stage families are

- uniquely exported solver-stage interface states, or states selected by an
  explicit inference-relevant rule, decoded by one fixed Riemann solver;
  endpoint closure alone does not identify an interface-state label;
- a cumulative coarse-solver or other inference-computable base impulse plus a
  normalized learned correction, but only after auditing base error and
  correlation; the base may use the current coarse state, parameters, and
  geometry, but not future reference states or solver stages, while
  training-set statistics may be used only for normalization;
- characteristic wave or fluctuation targets when the basis and degeneracies
  are controlled; and
- divergence-active or gauge-aware flux targets when the full reference gauge
  is shown to be unnecessary or harmful.

Pre-register tendencies and their falsifiers rather than retrospective
mechanistic stories:

- residual prediction may optimize faster than direct next-state prediction
  because it centers a near-identity map; loss whitening or an explicit skip
  that removes the gap would attribute the effect to parameterization;
- direct impulse labels should improve common decoded-state fit,
  divergence-active impulse error, or data efficiency over the state-loss-only
  flux head if missing local supervision is important; full impulse error for
  the state-loss-only row is gauge-dependent, and no matched gain weakens the
  local-supervision mechanism;
- a fixed-Riemann interface target should help only when its label-selection
  rule is unique and inference-relevant, and its proposed states remain
  credible without constant emergency limiting;
- base-plus-correction should help only when the cumulative base is correlated
  with the reference macro impulse; otherwise the network must cancel the base
  before learning the target; and
- characteristic or dense targets may amplify rare wave and shock signals but
  can lose through basis degeneracy or conflicting gradients.

Surprises are expected. Update the hypothesis after the controlled result; do
not retrofit the mechanism to make the run agree with the prior.

The rejected absolute flux, absolute interface, and instantaneous stride-4
Rusanov-correction rows remain negative controls for those exact formulations.
They do not reject identifiable cumulative-impulse supervision, a
uniquely selected fixed-Riemann interface target, or a credible macro-flux
base.
Do not rerun the rejected formulations unchanged.

Run two distinct comparison regimes:

- In the `strict matched` regime, freeze coordinates, data, backbone capacity,
  seeds, compute, clean-input contract, and decoded-state metric. Declare the
  deterministic decoder required by each target family. Within the
  state-loss-only, direct, and joint flux supervision block, also freeze the
  exact decoder and output parameterization. Use this regime for causal
  statements about target and supervision choice.
- In the `best engineered` regime, give surviving families equal tuning budget
  for target-specific normalization, baseline correction, horizon curriculum,
  and optimizer settings. Use this regime to select the practical algorithm,
  not to claim intrinsic target superiority.

Use successive halving. Before a batch starts, its run manifest must declare
dtype-aware closure tolerances, normalized supervised-objective and decoded-state
tiny-set pass thresholds, the maximum diagnostic interventions, optimizer
updates and data exposure per row, and the seed policy. Do not choose pass
thresholds after seeing the results. First require applicable label closure and
tiny-set fit; then run matched train/validation curves; then direct-horizon and
raw rollout tests. An apparent failure receives one predeclared repeat
initialization before rejection. Only finalists receive full seed confirmation,
timestep/resolution transfer, and 2D evaluation. A one-step winner is not
promoted when raw rollout or structural diagnostics fail.

The following continuation protocol was preregistered and executed for D028.
It is retained as experimental provenance and is not authorization for a
warm-start arm in the bounded Line-1 closeout. The protocol sent any label-valid family, with closure also validated where its decoder
contract requires it, that fits at short stride but fails cold at a larger
stride into the continuation diagnostic, regardless of its large-stride
ranking. The progressive arm uses sequential fixed-stride checkpoints over
`Delta t -> 2 Delta t -> 4 Delta t -> ...`; it is not mixed-stride training or
a newly timestep-conditioned model. Here `Delta t` is one saved reference
interval, not an internal accepted CFL substep. Transfer model weights only,
and reset the optimizer and scheduler at each stride; carrying optimizer state is a labeled
secondary ablation. At the final stride, hold architecture, data cases,
final-horizon labels, normalization, loss weights, optimizer/scheduler contract,
and final-stage sample presentations identical to a cold final-target arm.
Define sample presentations as batch size times optimizer updates, including
repeated examples, rather than epochs or nominal trajectory cases. Add a
total-exposure-matched cold arm whose final-target sample presentations equal
all continuation stages, plus composition of a small-step model trained on
`Delta t` labels. A lower final-target training-loss floor than both cold
controls supports a continuation-sensitive optimization claim; validation or
rollout gains without that fit gain indicate curriculum-induced generalization
or stability instead. None of these outcomes shows that the large-step operator
is intrinsically easier or physically better.

## Structured Dense Supervision

Dense supervision is approved as a staged representation-learning strategy,
not as a kitchen-sink loss. Preserve one primary solver path and derive
algebraically related outputs through known operators. For a flux model, the
network predicts one shared oriented face flux, while conservative increment,
next conservative state, and next primitive state are computed from that flux;
independent state heads must not bypass the conservative decoder at inference.

Classify each proposed auxiliary target before implementation:

- information-adding labels include exact time-integrated face flux, uniquely
  defined stage traces, shock/contact labels, front motion, wave speed,
  entropy production, admissibility margin, and solver error/fallback labels;
- redundant but potentially conditioning-improving losses include next
  conservative state, next primitive state, conservative increment, and flux
  divergence when they are linked by the deterministic decoder; and
- unsafe labels include underidentified macro interface states, arbitrary flux
  gauges, or solver-specific internal decisions that do not serve the intended
  inference algorithm.

Direct and joint impulse supervision belong to the primary target screen, not
to this dense stage. After selecting a primary target and validating its
reference label, add only auxiliary losses not already present in that
objective, in this order: decoded conservative state, decoded primitive state,
uniquely defined stage or partial-impulse traces, shock/front supervision, then
short unrolled supervision.
Semigroup, cross-resolution, entropy, and error-estimator losses are later
additions. Add one loss family at a time and report target scales, gradient
norms, and gradient conflicts. Auxiliary accuracy alone does not establish a
better representation; require data-efficiency, transfer, or rollout gains.

## Constraint Hierarchy

Enforce inexpensive algebraic and convex properties; regularize conditional or
representation-dependent properties:

- enforce interior conservation with one shared face flux, validated geometry,
  and exact owner/neighbor signs;
- enforce known boundary conditions in the solver, while reporting free versus
  clamped boundary behavior explicitly;
- use local conservative invariant-domain or antidiffusive-flux limiting for
  density and internal-energy admissibility instead of whole-sample scaling;
- prefer entropy-stable flux parameterizations or bounded entropy corrections
  when they can be made compatible with the actual time integrator; and
- use soft losses for shock geometry, short rollout, semigroup composition,
  cross-resolution commutation, and calibrated failure prediction.

Do not impose blanket componentwise TVD on multidimensional Euler. TVD is an
appropriate hard diagnostic for scalar 1D problems and a possible constraint
on characteristic or face-correction variables. For Euler systems, prioritize
conservation, invariant domains, entropy admissibility, and local oscillation
control. Global spectral or total-variation penalties can erase physical shocks
and must not be introduced without smooth/shock-region separation.

## Capability And Escalation Gate

The project must test whether a strong neural operator can handle genuinely
hard time-dependent PDE rollout before accumulating method complexity. A
50-step result must state the physical horizon, characteristic travel
distance, timestep, effective CFL, and number of autoregressive model calls.
For each serious candidate, measure:

1. teacher-forced one-step accuracy;
2. direct prediction to selected future horizons as a representation oracle;
3. at least 50 raw autoregressive calls where the data contract supports them;
4. timestep, resolution, and parameter transfer; and
5. performance against a coarse classical solver at matched runtime or error.

The 1D ladder must include shocks, contacts, rarefactions, reflections, and wave
interactions. The CPG bump remains an irregular-geometry attached-shock transfer
test, but it is not sufficient evidence for dynamic moving shocks. A credible
story ultimately needs a dynamic 2D shock problem in addition to the bump.

Escalate engineering only in response to a named failure:

- use a stable coarse/subcycled solver plus learned time-integrated conservative
  correction when raw proposals are nonphysical or worse than coarse CFD;
- use a physics-routed facewise expert or dissipation model when error is
  localized to shocks/contacts, with one gate shared by both adjacent cells;
- use an explicit front/level-set plus one-sided smooth fields when shock phase
  and smearing remain dominant after local-global controls; and
- use INR space-time representations for smooth pieces, front geometry, or
  adaptive querying, not as an assumed cure for a raw discontinuous field.

Adaptive time stepping should be driven by a measurable defect such as one
full step versus two half steps, semigroup inconsistency, entropy violation, or
admissibility risk. Front tracking must use Rankine-Hugoniot-compatible motion
and conservative remapping; generic per-cell MoE and visually sharp but
nonconservative front prediction do not qualify.

## Four-Line Research Program

### Program Hierarchy

| Line | Role | Immediate decision target |
| --- | --- | --- |
| 1. Large-step flow maps | Bounded 1D closure and theory seed | Determine the empirical large-step operating envelope and formalize what causal reach does and does not imply for a global operator. |
| 2. CPGNet mechanism audit | Forensic enabling line | Determine whether the reported 2D behavior comes from genuine causal propagation, benchmark conditioning, decoder flexibility, or evaluation assistance. |
| 3. Geometry-aware 2D rollout | Flagship algorithmic line | Establish a strong residual baseline, classify the ripple/instability mechanism, and promote one targeted stabilization method. |
| 4. Latent forecast and assimilation | Cross-cutting representation line | Determine whether phase-amplitude or hybrid spatial latent coordinates preserve sharp fronts, improve Markov closure and raw recurrence, and subsequently support calibrated assimilation. |

The four lines do not define their own incompatible
benchmarks. Line 1 supplies common causal-reach and timestep language. Line 2
decides how CPGNet may be used as evidence or as a baseline. Line 3 can begin
baseline and diagnostic work immediately, but method selection waits for its
diagnostic gate. Line 4 may begin its read-only representation and literature
audit immediately, but a learned 2D pilot must wait for Line 3's dynamic
reference promotion and then consume that testbed and the frozen physical-space
baseline. Its data-assimilation stage is blocked until its raw latent open-loop
forecast passes the declared forecast gates. A coordinating owner maintains the
shared evaluation contract and reconciles claims across lines.

### Shared Terminology

Retain `C_eff = max_x(|u| + c) Delta t / h` as a descriptive property
of a case, not as a universal learned-model stability condition. On a uniform
1D grid, a local architecture can compare the physical characteristic reach
`r_phys(Delta t)` with the architecture state-propagation radius `r_arch`:

`chi = r_phys(Delta t) / r_arch`.

For arbitrary hyperbolic inputs, `chi > 1` exposes a worst-case causal
obstruction. On an irregular 2D graph, do not collapse this to one scalar:
report node-, direction-, state-, and time-dependent predecessor-set coverage
in a common physical metric. Dataset conditioning can hide an obstruction, so
nominal performance on a narrow solution family does not disprove it.

A domain-global spatial operator remains global for every timestep; its spatial
receptive field does not cease to be global when characteristics cross the
domain. At longer horizons, the information contract may instead become
insufficient because boundary or forcing history is absent from the input.
There is no proposed universal neural CFL limit. Report a learned flow-map
operating envelope conditioned on architecture, information contract, data
distribution, resolution, training budget, physical horizon, error tolerance,
admissibility rate, shock gates, and cost.

Use semigroup defect and perturbation amplification only as supporting temporal
diagnostics. Interpret a defect such as `T_8 - T_4 o T_4` beside the truth error
of both paths because shared bias can make the defect small. Normalize
amplification per physical time, for example `log(L_Delta) / Delta`, under a
fixed perturbation distribution and norm. These quantities describe a learned
finite-time propagator; they do not turn a feed-forward evaluation into an
implicit numerical solve.

### Line 1: Large-Step Flow Maps And Their Limits

#### Core question

How large a finite-time Euler flow map can a fixed global neural operator learn
accurately, stably, and efficiently, and which observed limits arise from
causal reach, target-map complexity, data coverage, optimization, or recurrent
amplification?

#### Work package 1A: small theory and definitions

Limit the initial theory package to three results:

1. A local receptive-field lower bound for linear advection, using two inputs
   that agree inside an `r`-local receptive field but differ at an upstream
   point transported into the target cell whose periodic modular grid distance
   from the target exceeds `r`.
2. A global spectral counterexample to a universal CFL barrier: on band-limited
   periodic linear advection, a Fourier multiplier realizes the exact
   finite-time shift for any fixed `Delta t`.
3. A consistency-amplification bound of the form
   `e_N <= epsilon_Delta sum_(j=0)^(N-1) L_Delta^j`, with
   `N = T / Delta`, plus a semigroup-defect definition.

Stop after these statements and proof sketches are rigorous. Do not open an
unbounded nonlinear neural-CFL theorem project inside the 1D closeout. Broader
approximation theory becomes a separate future direction only if the empirical
frontier exposes a quantity that this small theory predicts.

Use the following precise statements.

1. **Finite-receptive-field obstruction.** Let the exact periodic linear-
   advection flow be `S_Delta u(x) = u(x - a Delta)`. Suppose an operator's
   output at grid point `x_j` depends only on input values within `r` grid hops
   of `x_j` on an `n`-point periodic grid. For the clean discrete statement,
   assume `a Delta = q Delta x` for an integer `q`, and let
   `d_n(q,0) = min(q mod n, n - (q mod n))`. If `d_n(q,0) > r` and the admissible
   input class contains two grid states that agree on the local neighborhood
   but differ by `delta` at `x_{j-q mod n}`, the learned outputs for the two
   states must agree while the exact shifted outputs differ by `delta`. The
   triangle inequality therefore gives worst-case point error at least
   `delta/2` for one input. A resolved non-grid preimage admits the analogous
   continuous-input construction, and the same argument applies on an interior
   interval before boundary influence arrives. The modular condition is
   essential: a displacement larger than `r Delta x` can wrap back into the
   local neighborhood. This is a hard causal lower bound for local
   architectures and rich input classes, not for global operators or narrow
   trajectory manifolds.
2. **Band-limited spectral counterexample.** On a periodic domain, assume
   `u(x) = sum_{|k| <= K} u_hat_k exp(i k x)` and at least `2K+1` non-aliased
   grid samples. The Fourier multiplier `m_k = exp(-i k a Delta)` realizes
   `S_Delta` exactly for every fixed `Delta`. A linear Fourier layer retaining
   all modes through `K`, with the conjugate symmetry needed for real outputs,
   realizes this map exactly; an FNO hypothesis class that contains this
   undistorted linear case therefore contains the exact map. Thus no universal
   CFL-style timestep barrier follows from spatial causal reach for a domain-
   global spectral operator.
   This statement does not cover nonlinear shock formation, truncated modes,
   missing boundary/forcing history, finite data, or optimization.
3. **Consistency, amplification, and semigroup defect.** Let `G_Delta` be a
   learned map and `S_Delta` the exact flow on a forward-invariant tube `D`.
   If `sup_D ||G_Delta(v)-S_Delta(v)|| <= epsilon_Delta` and `S_Delta` is
   `L_S`-Lipschitz on `D`, then, while both recurrences stay in `D`,

   `||G_Delta^N(u)-S_Delta^N(u)|| <= epsilon_Delta sum_{j=0}^{N-1} L_S^j`.

   Alternatively, a truth-state consistency bound plus an `L_G`-Lipschitz
   learned map gives the same expression with `L_G`; this distinction is why
   truth-state and same-on-policy-state errors must both be measured. For
   `m Delta`, define

   `d_m(u) = ||G_{m Delta}(u)-G_Delta^m(u)||`.

   The triangle and reverse-triangle inequalities give

   `|e_direct(u)-e_composed(u)| <= d_m(u) <= e_direct(u)+e_composed(u)`,

   and the consistency bound gives

   `d_m(u) <= epsilon_{m Delta} + epsilon_Delta sum_{j=0}^{m-1} L_S^j`.

   A small defect can therefore coexist with two inaccurate paths, while a
   large defect does not by itself say which path is closer to truth.

#### Work package 1B: bounded empirical operating envelope

Freeze the 64/24/4 conservative-variable residual architecture, conservative
input/loss/recurrence, zero-update initialization, raw inference, and physical
scaling. Do not conflate the evidence regimes: the strong accuracy baseline
uses the full 384/64/64 split and three seeds, while D027 used a 64/16/16
midscale split, one model seed, and H20 checkpoint selection.

Before Phase 1, select one frontier regime. The preferred paper-facing regime
is the full split; if its matched references cannot fit the budget, keep the
entire frontier midscale and label it accordingly. A matched frontier has at
most one cold row for each stride 1, 2, 4, and 8 plus two additional seeds at
the scientifically relevant pass/fail boundary. Treat those as two
confirmation slots: normally both are seed repeats, but one may instead isolate
the recurrent call-depth versus physical-horizon confound. If so, explicitly
weaken the seed-stability claim. A historical checkpoint may be reused only
when its split, exposure, normalization, and selection rule match. Any required
stride-1 or stride-2 retraining counts against the six-run cap.

Never exceed six new training runs. Do not add stride 16, architecture scaling,
warm-start rescue, a constraint sweep, or a timestep-conditioned model to this
closeout. Separate fixed-step models answer fixed-flow-map capacity; they do
not establish one reusable timestep-conditioned solver.

Use common physical endpoints divisible by every tested stride and chosen
before waves leave the open domain. Compare each direct map with every
available smaller-step composition. Match sample presentations rather than
nominal epochs where stride changes the number of windows. Select checkpoints
at one physical frame divisible by eight. The historical H20-selected D027
rows are unmatched context unless saved epochs can be reselected without
retraining; any rerun counts against the cap. Predeclare whether recurrent
training holds call depth or physical unroll horizon fixed, report the other as
a confound, and spend a boundary repeat on a matched control if the conclusion
depends on that choice.

Required readouts are tiny/full training floors; direct first-jump and
teacher-forced error; raw completion and failure type; the full shock hierarchy
and smooth-region error; conserved-total and boundary-exchange error; effective
CFL and characteristic travel over the macro interval; semigroup defect beside
both paths' truth errors; per-physical-time perturbation amplification;
batch-1 latency and amortized throughput after warm-up and synchronization;
accuracy-matched wall-clock Pareto comparison with documented preprocessing
and transfer boundaries; paired error-cost against the same reference solver;
and a modest held-out Riemann-state suite varying states and discontinuity
position.

Stop when the first failing stride has been repeated and classified, when
direct large steps are dominated by smaller-step composition, when apparent
gains disappear on the held-out suite, or when an efficiency gain exists only
in call count rather than wall-clock cost. A successful stride-8 result gives a
lower bound on the tested operating envelope, not proof of no timestep limit.

#### Phase-0 closeout decision, 2026-07-18

The full-split paper-facing frontier is complete and consumes the entire
six-training-run budget: one cold row at strides 1, 2, 4, and 8 with seed
`20260707`, plus stride-8 confirmation seeds `20260708` and `20260709`. Every
row uses the 384/64/64 trajectory split, 1,920,000 one-step sample
presentations, 372,480 recurrent windows, four recurrent calls per window, and
H32 validation checkpoint selection. The selected epochs for strides
1/2/4/8 and the two stride-8 repeats are 47/60/59/58/57/57.

This contract fixes recurrent call depth, so the physical unroll horizon grows
with stride. It does not isolate matched physical recurrent exposure. The
stride-1 checkpoint was selected before its recurrent stage, whereas all
larger-stride checkpoints were selected during recurrent training; report both
facts whenever the frontier is interpreted.

The Phase-0 prose originally attached the labels H8/H16/H32 to the following
triplets. The frozen D032 replay corrects that labeling: these are common
physical frames H32/H64/H96. Under exact conservative recurrence, the selected
mean fixed-scale conservative relative-L2 errors are:

- stride 1: `0.02680 / 0.06588 / 0.12706`;
- stride 2: `0.01204 / 0.02243 / 0.03008`;
- stride 4: `0.00917 / 0.01853 / 0.02646`; and
- stride 8, pooled across three seeds: `0.00891 / 0.01256 / 0.01798`, with
  192/192 raw H96 completions.

The corrected dense evaluation resolves the short-horizon crossover. At H8,
selected endpoint error is best at stride 2 (`0.00536`); at H16 it is best at
stride 4 (`0.00647`); and at H32/H64/H96 it is best at stride 8. Time-mean error
selects stride 8 at H8, stride 4 at H16/H32, and stride 8 at H64/H96. The
empirical sweet spot is therefore objective- and horizon-dependent even under
one fixed model/data contract. Active waves remain in the domain at H96: all
64 cases retain an interior active front, the median primary front is at face
191.5 of 256, and the median relative state change over the final audited
eight-frame interval H88--H96 is `0.236`.

This result demonstrates an observed accuracy-and-call-count advantage for the
tested large-step maps at medium horizon under the frozen contract. It does not
establish monotone
superiority at every endpoint: the first direct stride-8 jump is worse than
stride-4 composition and its shock-position error is worse. The supported
object is a metric-, horizon-, distribution-, architecture-, training-
contract-, and error-budget-dependent learned flow-map operating envelope, not
a universal optimal timestep or neural CFL threshold.

#### Phase-1A frozen-evaluation closeout, 2026-07-18

Phase 1A is complete with zero new learned training.

The six-run bundle was registered before evaluation. The selected-checkpoint
registry contains six hashes, the candidate registry contains 360 hashes, and
the full registry contains 608 files while excluding failed attempts. The
verified archive is 931,943,871 bytes with SHA256
`81151ee33478935b53ba650dc9f57b4c90c42de158db71069e2ad6bc7491bb55`.
Large artifacts remain ignored and must not be deleted merely because compact
summaries are present in the repository.

D032 evaluated primary selections and the predeclared epoch-50/60 candidates
at every stride-compatible endpoint through H96. At H96, the fractions staying
raw-admissible and below fixed-scale conservative error budgets `0.02 / 0.05 /
0.10` are `0/64, 4/64, 16/64` for stride 1; `24/64, 52/64, 63/64` for stride 2;
`19/64, 59/64, 64/64` for stride 4; and `43/64, 64/64, 64/64` for stride 8.
The stride-8 selected repeats are tight: H32 error ranges `0.00860--0.00907`
and H96 error ranges `0.01767--0.01821`, all with 64/64 completion.

Candidate evaluation exposes real selection sensitivity but does not authorize
test-set reselection. The stride-1 epoch-60 candidate lowers H96 mean error from
`0.12706` to `0.03933`, while completion changes from 62/64 to 61/64.
Epoch-50 is worse at H32/H64 and for every larger stride; its lower stride-1
H96 survivor mean comes with only 59/64 completion. Epoch-60 effects are not
uniform across the larger strides. The candidate shrinks the apparent stride
gap and reverses the stride-1/stride-2 test ranking at H32. Strides 4 and 8
remain better at H32, and every larger stride remains better than stride 1 at
H64/H96. Primary validation selections remain immutable, and the pre-recurrent
stride-1 selection remains an explicit checkpoint-stage confound.

The D032 field historically named `conservative_budget_relative_l2` is
algebraically the relative mismatch between predicted and truth
domain-integrated conserved totals. It does not use boundary fluxes and is not a
conservation-closure residual. The maintained evaluator now emits the explicit
name `global_conserved_total_mismatch_relative_l2` while retaining the old field
only as a compatibility alias. Because the residual FNO is not structurally
facewise conservative, conserved totals remain outcome diagnostics and the
planned boundary-exchange conservation readout is still unfulfilled.

D033 passed the reference replay gate only after a documented pre-model gate
revision. The initial max-only `1e-5` gate stopped before learned evaluation.
The 1,024 replay rows had mean `1.18e-7`, p99 `3.20e-7`, and one maximum
`2.09e-5`, consistent with the known float32 serialization scale. The revised,
pre-model gate requires mean and p99 at most `1e-6` and maximum at most `5e-5`;
the full evaluation then ran in a new preserved artifact directory.

The same-state result directly exposes the timestep tradeoff. At start frame
32, truth-state learned-versus-reference one-call error rises from
`0.00082/0.00094/0.00156/0.00273` across strides 1/2/4/8, while the accumulated
on-policy state error falls from `0.02680/0.01204/0.00917/0.00907`. The larger
map is harder locally, but it is applied fewer times. One reference replacement
at frame 32 or 64 changes mean H96 paired error by only about `-5.8%` to `+1.6%`
and changes no completion count. Per-physical-time perturbation amplification
and semigroup defect do not reveal a monotone instability threshold; retain
them as supporting diagnostics beside both paths' truth errors.

D034 times complete rollouts on one documented hardware/software contract,
with 10 warm-up and 50 synchronized learned repeats, batch-1 and batch-16
boundaries, explicit host-transfer/preprocessing scopes, and a reference solver
limited to one CPU thread per case. On the fixed 16-case subset at H96, direct
stride-2/4/8 host-to-host batch-1 times are `60.40/30.75/15.78` ms. Their
accuracy-matched reference times yield `14.55x/65.53x/127.71x` speedups. The
stride-8 batch-16 host-to-host throughput is 991 cases/s versus 0.277 cases/s
for the matched 256-cell reference. These are measured Pareto points against
this documented Python reference implementation, not claims against optimized
production CFD or across hardware.

#### Frozen error-geometry addendum, 2026-07-19

D035 closes the requested visual and phase-resolved analysis with zero new
learned training. Across truth-state starts `0,8,...,88`, absolute one-call
error rises strictly with stride at all 12 starts, but error per physical step
falls strictly. The fitted four-stride log-error/log-stride exponent ranges
from `0.488` to `0.696`. Treat this as contract-specific empirical scaling
across separately trained maps, not as an approximation order.

At H96, stride 8 is best on fixed-scale conservative L1 and L2, separated-
front-region L2, smooth-region L2, and post-hoc global-translation-oracle L2.
Stride 2 is best on top-two-front position MAE. The translation oracle removes
only `10.8%` of stride-8 fixed-interior error on average, versus `55.4%` for
stride 1, and does not reverse the large-step ranking. The result is therefore
not merely a global phase-lag artifact, although front position remains a real
metric conflict.

At error budget `0.05`, H96 reliable counts are `4/64`, `52/64`, `59/64`, and
`64/64` for strides 1/2/4/8. Casewise H96 winners among strides 2/4/8 are
`14/15/35`; their post-hoc oracle improves mean error by `20.61%` over always
using stride 8. A predeclared six-descriptor nearest-centroid router performs
worse than its fold-majority control (`0.328` versus `0.547` accuracy) and
fails every signal gate. Stop adaptive routing rather than adding a controller
or descriptor sweep.

The artifact includes common-time error curves, L1/L2 and shock/smooth
decompositions, budget survival, phase-aligned diagnostics, a runtime Pareto
figure, and five exact-recurrence GIF/NPZ rollouts. It reinforces the
harder-map/fewer-calls mechanism and preserves the existing claim boundary:
the tested medium-horizon optimum remains right-censored at stride 8, so there
is no universal optimal timestep or neural CFL claim.

#### Frozen ripple addendum, 2026-07-19

D036 tests the qualitative observation that stride 8 has stronger ripples than
stride 4 while retaining lower final state error. It reuses all four primary
checkpoints and both frozen stride-8 repeats, adds no learned training, and
measures derivative and total-variation errors outside four-cell
neighborhoods of the two strongest truth pressure fronts.

At H96, primary stride 8 has `0.668x` stride-4 global L2 with paired-bootstrap
ratio interval `[0.571, 0.780]`, but it has `1.197x` smooth-region
second-derivative error (`[1.147, 1.244]`) and `1.094x` smooth-region
predicted/truth TV (`[1.069, 1.120]`). Its modes-65-to-Nyquist share of error
energy is also `1.197x` stride 4. All three stride-8 seeds preserve lower
global L2 and higher smooth roughness than the single stride-4 row.

For sensitivity only, define ripple-budget failure as the first common
endpoint that is raw-inadmissible or exceeds a smooth predicted/truth TV ratio
of `1.25`, `1.5`, or `2.0`. At `1.5`, median first exceedance is H68 for stride
4 and H56/H56/H48 for the three stride-8 seeds; H96 survival is 14/64 versus
6/64, 7/64, and 3/64. The ordering persists across the threshold grid. This
supports the user's narrower hypothesis that stride 8 can have a shorter
*ripple-budget* lifetime even while its global-error lifetime is longer. It
does not establish earlier loss of positivity: every stride-4/8 path is still
raw-admissible at H96.

This also sharpens the comparison with
[PDE-Refiner](https://proceedings.neurips.cc/paper_files/paper/2023/file/d529b943af3dba734f8a7d49efcb6d09-Paper-Conference.pdf).
Its Appendix E.3 U-Net study on periodic Kuramoto--Sivashinsky reports longer
high-correlation times for smaller output steps; the MSE sweep spans 1--64
solver steps, while the PDE-Refiner points shown span 1--8. Our shock-dominated
Euler/FNO result is directionally different for H96 global L2, but D036 agrees
with its central frequency warning: a favorable aggregate error can conceal
degraded low-amplitude/high-frequency structure. Treat this as a
PDE-, architecture-, training-contract-, horizon-, and metric-dependent regime
contrast, not a refutation of autoregressive neural PDE solvers.

#### Frozen modal mechanism addendum, 2026-07-19

D038 resolves why the stride-8 animation can look more oscillatory even when
its later global error is lower. On common truth starts, stride-8 one-call
truth-normalized update error exceeds stride 4 in every spectral band across
all three seeds. The excess is `2.465--2.744x` in modes 17--24,
`1.308--1.427x` in modes 25--64, and `1.325--1.384x` in modes 65--Nyquist.
The historical artifact key `resolved_17_24` straddles the 24-mode FNO cutoff:
`k=17--23` are retained and `k=24` is the first omitted mode. Interpret the
reported band numerically, not as eight wholly retained modes.
At H8, the primary stride-8 rollout has `1.218x` total modal error and all 101
truth-resolved modes are worse; its spectral centroid, normalized `k^4` shape,
and tail share are also higher for all three stride-8 seeds.

By H32, primary stride 8 has crossed to `0.917x` total modal error while its
centroid, normalized `k^4` shape, and tail share remain
`1.203/1.327/1.857x` stride 4. By H96 its total modal error is only `0.522x`;
the primary seed still has a slightly rougher global error shape, but that
global shape ordering is not preserved by both stride-8 repeats. D036's
away-front derivative/TV ordering *is* preserved across all repeats. The
supported interpretation is therefore: a larger map injects a harder,
broadband defect; fewer recurrent calls later reduce total error; and the
remaining late ripple is localized rather than a global high-frequency energy
blow-up.

This favors algorithmic separation of a strong global macro map from any
future localized conservative roughness correction. It does not authorize a
Line-1 architecture or loss sweep, and it argues against blind global spectral
damping, which could erase physical shocks while attacking a late error that
is spatially localized. The conditional D037 oracle remains the gate for
testing whether such localized correction headroom exists in the flagship 2D
setting.

#### Frozen operating-envelope synthesis addendum, 2026-07-20

D039 combines the registered D032--D038 source tables without adding training,
checkpoint evaluation, solver calls, or selection. Its ten-invariant audit
passes: the four primary checkpoint hashes, 64-case identity, 101 saved-frame
contract, H96 wave activity, H32 harder-map/fewer-calls ordering, and
seed-consistent stride-8 teacher-modal deficit all reproduce; the 72 scalar
rows duplicated between D036 and D038 agree exactly.

The resulting operating envelope distinguishes an aggregate winner from a
statistically separated one. Endpoint global error selects stride 2 at H8,
stride 4 at H16, and stride 8 at H32/H64/H96. Paired held-out-case bootstrap
intervals cross equality for the H8 stride-2/stride-4 and H32
stride-8/stride-4 comparisons, but not for H16, H64, or H96. Stride 8 first and
sustainably beats stride 4 at H32 for endpoint error; its time-mean error does
not stay below stride 4 until H56. At H96, stride 8 wins global error,
reliability at budget `0.05`, and measured latency, whereas stride 2 wins
front-position, away-front second-derivative, and away-front TV-deviation
metrics. Strides 2/4/8 remain tied on raw admissibility.

Therefore an empirical optimum must be written as
`stride*(horizon, metric, error budget, admissibility rule, runtime boundary)`.
The D039 map is descriptive within the frozen contract; its paired intervals
are post-selection held-out-case uncertainty, not training-seed uncertainty or
confirmatory uncertainty for a predeclared winning stride. It strengthens the
operating-envelope conclusion and closes empirical Line 1 without resolving
the right-censored global optimum.

#### Retrospective mild-OOD artifact audit, 2026-07-22

This audit adds no model training, checkpoint selection, or evaluation. Three
existing directories under `artifacts/time_dependent_no/line1_frontier_20260718/`
share OOD data SHA256
`e378ca6df947116f7847376edbc3af140e82a33db3d3ec55edea0f2a6e249b67`, the
label `mild_support_extrapolation_v1`, shape `32x101x256x3`, and evaluator source
SHA256 `d528fdb2d80d907ee8d3850168aa5ef735718f5dae9ebe3bf594a573d7f595bf`.
At H96, the frozen primary stride-2/4/8 direct errors are
`0.07938/0.04287/0.04890`, with completion `31/32`, `32/32`, and `32/32`.
Replacing only the stride-8 checkpoint by its two confirmation seeds gives
H96 errors `0.04074` and `0.03611`, again with `32/32` completion.

The supported OOD conclusion is narrow: stride 4 and every observed stride-8
seed improve substantially over the single stride-2 row on this mild suite, but
stride 8 does not beat stride 4 for every model seed. Only stride 8 has seed
repeats, so there is no matched seed-stability comparison across strides and no
universal extrapolation claim. Do not pool these 32 OOD cases with the 64-case
matched-ID D032 summaries.

The historical OOD path rows omit per-case regime labels, the serialized OOD
NPZ is not present in the local evidence bundle, and the old contract records
the user-supplied suite label but not the generator-source hash. Under the
current generator's concatenation order, cases `0--15` would be `high_inflow`
and cases `16--31` `low_ambient`; that mapping is therefore an inference, not
independently verified provenance. It must not support regime-specific claims.
For orientation only, the primary inferred block means are stride 4 versus
stride 8 `0.04638` versus `0.06374` on cases `0--15`, and `0.03936` versus
`0.03407` on cases `16--31`; the ordering changes across stride-8 seeds.

The maintained evaluator now requires the embedded `ood_contract_json` and
`ood_regime`, requires and records the generator SHA256, checks the declared
regime counts, attaches the regime to every case row, and writes regime-specific
summaries. Those changes secure future runs; they do not retroactively upgrade
the historical artifact's provenance.

The Line-1 empirical ladder is now closed. Do not add stride 16, matched-
exposure retraining, new targets, limiters, barriers, or timestep-conditioned
training to this line. Strides 1/2/4/8 are sufficient to establish the bounded
tradeoff and a stride-8 lower bound on the tested envelope, but not to bracket a
global optimal timestep because the best medium-horizon row lies at the tested
right boundary. The 101 saved frames are sufficient for common endpoints
through H96 (`t=0.48`, twelve stride-8 calls) with active waves still present;
they are not a long-horizon plateau or a basis for a stride-16 optimum claim.

#### Cross-line frozen rollout-instability attribution decision (registered 2026-07-22)

The observation that a larger stride can remain raw-admissible longer does not
identify why a smaller-stride FNO or a PCNO rollout fails. In this project,
instability must be split into four outcomes: first global-error-budget
exceedance, first roughness/ripple-budget exceedance, first raw-admissibility
failure, and excess learned perturbation amplification relative to a reference
flow. D036 already shows that these lifetimes can rank strides differently.
Do not use raw completion alone to call one learned flow map more stable.

Register one bounded cross-line attribution diagnostic, `XLINE-001`. This is a
documentation decision only: it authorizes no training, checkpoint selection,
architecture or target change, limiter, smoothing rule, GPU run, or test-split
access. It does not reopen Line 1 and must not interrupt or alter an active
Line-3 run. Reuse, without regeneration, D033 and D035--D038 for the frozen
D031 FNO frontier. For new case-level forks, use at most four stride-1/2 raw
failures ordered by first-failure time then case ID, their same-case stride-4/8
paths, and at most four stable matched controls. Use the three D041 bump-PCNO
H79 failures and at most three matched completed paths as a separate failure
cohort; use only D044's frozen six-case D013 validation cohort as a separate
stable control. Do not pool statistics across the 1D, bump, and shock--vortex
regimes. Select matched controls before new diagnostics by deterministic nearest
matching on frozen initial physical descriptors, with case ID as the tie-break;
do not use later rollout traces for matching.

For a reference flow `S_Delta`, learned map `G_Delta`, truth state `u`, and
on-policy state `v`, record the vector terms, their norms, and their pairwise
alignment in both exact decompositions

`G_Delta(v) - S_Delta(u)`
`= [G_Delta(v) - S_Delta(v)] + [S_Delta(v) - S_Delta(u)]`

and

`G_Delta(v) - S_Delta(u)`
`= [G_Delta(v) - G_Delta(u)] + [G_Delta(u) - S_Delta(u)]`.

The first separates same-state model defect from reference-flow propagation of
the existing state error; the second separates learned amplification from
truth-state consistency. Scalar norms alone are insufficient because the two
terms can align or cancel. On the 1D multi-stride family, also evaluate, from
the identical start state and at identical physical endpoints,
`G_8`, `G_4^2`, `G_2^4`, and `G_1^8`, reporting every path's truth error beside
each semigroup defect. Cross the state source and operator by applying each
native-stride map to truth states and to states generated by the small- and
large-stride paths at common start frames. This is the missing test that
separates map difficulty from state-distribution shift and call count.

Estimate finite perturbation growth under one frozen conservative physical
norm and one recorded perturbation bank. Report

`lambda_G(Delta; v, delta) = Delta^(-1) log((||G_Delta(v+delta)-G_Delta(v)||+eta) / (||delta||+eta))`

and the identical `lambda_S` reference quantity, with fixed perturbation norm,
distribution, numerical safeguard `eta`, and physical-time normalization.
Include smooth low-mode, front-phase, front-strength, high-pass, boundary, and
pressure-margin directions only when each can be defined reproducibly on that
regime. A large learned gain is evidence of learned dynamical amplification
only relative to the same-state reference gain; it is not a universal neural
CFL threshold.

Align every diagnostic to the first outcome event and record the earliest
precursor among front position/strength/thickness, shock- and smooth-region
state error, Fourier or graph-native high-pass error, total variation or second
difference, raw update norm, minimum density/internal energy/pressure,
boundary error, and a declared training-distribution-distance proxy. Temporal
precedence alone selects a candidate mechanism. Causal attribution additionally
requires either the reference-flow decomposition or one preregistered bounded
intervention: a single truth reset, a single validated reference-solver
replacement, a shock-versus-smooth truth-state transplant, or a boundary reset
on a boundary-triggered path. Low-pass filtering and positivity projection may
be used only as labeled mechanism falsifiers, never as hidden rollout fixes.
After the observational pass, preregister at most two intervention types per
regime and one intervention time per failing path before seeing intervention
outcomes.

The exact same-state reference fork already exists for the 1D solver. The
shock--vortex implementation exposes lower-level stepping from a supplied
conservative state, but its generated-state restart, time/boundary, saved-time,
and closure contract must pass a focused preflight before D044 causal use. No
generated-state restart is currently validated on the released bump/DG
artifact. Therefore the D041 bump-PCNO branch may support precursor,
crossed-state model-response, and intervention evidence, but it must not claim a
solver-versus-model causal decomposition until an audited restart contract
exists. Record these as blockers rather than substituting the next saved truth
frame for `S_Delta(v)`.

All registered cohorts contain shocks or sharp fronts. XLINE-001 can therefore
identify a shock-local precursor or intervention response within these regimes,
but it cannot establish that discontinuities are necessary for instability. A
matched smooth-solution control would require separate data and authorization.

Classify a mechanism within one frozen regime only when its precursor leads the
failure in the failing paths but not the matched controls and a same-state fork
or targeted intervention points in the same direction. Otherwise report a
mixed or unresolved mechanism. In particular:

- rising on-policy model defect before reference amplification supports
  recurrent distribution shift;
- learned gain exceeding same-state reference gain supports learned dynamical
  amplification;
- front-phase error preceding broadband growth supports transport-error
  injection, while the reverse ordering supports ripple growth upstream;
- pressure collapse after already-large state error is a terminal failure mode,
  not the root cause;
- a boundary-timed precursor repaired by the boundary intervention supports a
  boundary-contract mechanism; and
- larger one-call defect together with lower per-time gain and fewer calls
  supports the approximation--recurrence tradeoff, not unconditional
  large-step robustness.

Stop after this frozen attribution report. If no mechanism passes the within-
regime evidence rule, preserve the composite diagnosis and do not manufacture a
single cause. A mechanism-matched algorithm or architecture experiment would be
a separate Line-3 authorization; no such experiment follows from XLINE-001.
Any future artifact must record case and checkpoint identity, state source,
direct/composed path, physical start and endpoint, `Delta`, event times,
component norms and alignments, perturbation identity and norm, intervention,
failure cause, data/evaluator/reference digests, and whether `S_Delta(v)` came
from a validated restart.

A separate 2D extension is conditionally recommended only after the Line-3
baseline and legal data contract pass their gates: choose two physically
interpretable output intervals and train at most three seeds per interval.
That work belongs to Line 3 and requires separate authorization.

### Line 2: CPGNet Validity, Causal Reach, And Decoder Mechanism

#### Core question

Under what conditions can a state-loss-trained latent-interface/Rusanov decoder
produce a stable macro flow map, and how much of the released 2D result comes
from causal propagation, a low-dimensional benchmark manifold, flexible
directed update factors, or future-boundary assistance?

The default hypotheses, in order of discriminating power, are:

1. Dataset provenance or the released rollout/evaluation contract explains an
   important part of the reported performance.
2. The audited final-output dependency radius covers the physical
   characteristic cone, so DG substep count was the wrong proxy for graph
   distance. Twelve processor layers are only a nominal starting point because
   endpoint decoding and aggregation may add a dependency hop.
3. The cone is not covered, but geometry, Mach number, absolute position, and a
   narrow family of anchored flows make the next update predictable locally.

4. State-loss training makes interface outputs correlated, divergence-benign
   flux-control coordinates rather than physical traces.
5. Learned directed geometry factors, Rusanov dissipation, positivity,
   recurrent training, and boundary assistance supply the observed stability.
6. Only after rejecting the above should two-dimensionality itself be treated
   as a possible advantage for interface or flux targets.

Do not call CPGNet an implicit scheme. Current evidence supports a feed-forward
coarse flow map with ADER-inspired or implicit-like latent controls. Existing
instrumentation on local bs2 reproduction checkpoints shows admissible but
nonphysical latents whose learned update is not preserved by the tested
physical projection; it does not establish the paper checkpoint's mechanism.

#### Ordered phases

1. **Validity and provenance, frozen models only.** Establish dataset identity,
   train/test provenance, parameter distances, checkpoint/configuration
   identity, and the exact boundary contract. Compare released next-reference
   boundary injection with physically legal boundaries and stratify error by
   graph distance from the boundary. The locally audited bump bundle is not
   assumed identical to the paper dataset.
2. **Causal reach, without retraining.** First verify the mesh-to-graph mapping
   and audit the exact cell-output dependency radius with code tracing or
   autograd. Then compute time-aware, orientation-consistent characteristic
   graph travel from validated edge lengths, normals, local velocity, and sound
   speed. Compare the physical predecessor set with the audited dependency
   set. When the cone is uncovered, construct paired admissible states that
   agree inside the dependency set but differ in the uncovered region.

3. **Conditional manifold versus evolution operator.** Measure conditional
   ambiguity given an `L`-hop patch, geometry, and Mach number. Add
   geometry-plus-parameter-plus-time and nearest-trajectory controls that do
   not consume the current state. Test same-geometry counterfactual initial
   conditions before using the word memorization.
4. **Frozen decoder instrumentation.** After verifying that graph nodes map to
   control volumes and edges map to oriented physical faces, measure
   interface-state physicality against declared trace reconstructions, learned-
   update covariance, graph-native divergence/cycle components, Rusanov
   central/dissipative contributions, true-volume face-pair balance, learned
   directed-factor magnitude, and recurrent Jacobian-vector amplification.
   Call a quantity flux error only when validated reference cumulative impulses
   and their graph mapping exist. Saved rollout arrays alone are insufficient.

5. **One minimal matched ablation, only after phases 1--4 isolate a
   mechanism.** Under one shared backbone, state loss, exposure, legal
   boundary contract, and parameter budget, compare a conservative-variable
   cell-residual decoder, interface/Rusanov with exact geometry, and
   interface/Rusanov with learned directed geometry. Flux-supervised training is a separate
   diagnostic, not another broad axis.

Boundary validity precedes causal claims; causal coverage precedes a
memorization claim; manifold ambiguity precedes decoder retraining. If the
exact paper dataset/checkpoint cannot be established, report a mechanism audit
of the available release and local bundle rather than a Table-2 reproduction.
Parameter count alone is never evidence of memorization.

#### Legal-boundary closeout decision (2026-07-20)

Validity was resolved before mechanism claims on one locked release-bundle
contract. The same 20 test trajectories, 79 rollout calls, dataset, and archived
closeout evaluator give the following mean-per-trajectory normal-node RMSE:

| Checkpoint and boundary contract | rho | v1 | v2 | pres |
| --- | ---: | ---: | ---: | ---: |
| Public checkpoint, released next-reference oracle | 0.177032 | 0.073653 | 0.082204 | 0.319612 |
| Public checkpoint, causal nodal sensitivity | 0.563023 | 0.263983 | 0.242344 | 1.137027 |
| One legal-boundary-trained checkpoint, causal nodal rollout | 0.368161 | 0.157947 | 0.133379 | 0.753202 |

The archived closeout-evaluator oracle reproduces the earlier oracle aggregate
to within `8.5e-7`. Legal-boundary training recovers `34--45%` of the four-variable
frozen-sensitivity RMSE and improves all four variables on 19/20 trajectories,
while remaining `1.6--2.4x` worse than oracle. It also improves q0.90
shock-centroid and symmetric-Chamfer error on 20/20 trajectories. The three
runs are finite and positive on normal nodes, but the trained row improves
IoU/F1 on only 11/20 and has a worse maximum post-policy boundary
outlier. This is evidence that the release architecture can support a useful
causal-nodal rollout under the recorded one-seed training configuration; it is
not evidence that the released oracle score is an autonomous result or that the
released training process was reproduced exactly.

| Outcome | Supported claim | Unsupported or unresolved claim |
| --- | --- | --- |
| Reproduction | Pinned-runtime-file release replay under oracle boundary behavior | Paper table, paper dataset/checkpoint/splits/seeds, or compared-baseline evaluator parity |
| Boundary validity | Future-reference injection materially assists the released row; causal-nodal retraining recovers part of the loss | Exact DG or universally legal CFD boundary treatment |
| Causal reach | Audited current-state radius is 13 and none of 800 endpoint-sampled characteristic rows exceeds it | Integrated DG-substep cone coverage or a paired-state causal impossibility result |
| Conditional manifold | No positive claim | Dataset-manifold interpolation, nonlocal state dependence, or memorization |
| Decoder stability | One legal-boundary-trained seed remains finite and positive for all 20 test rollouts; local bs2 probes show nonphysical control coordinates | Isolation of Rusanov/interface decoding from exposure, depth, boundary training, and data-family effects |
| Conservation | Antisymmetric raw edge updates are implementation evidence | Exact physical finite-volume conservation without control volumes, face measures, orientations, and reference impulses |

The legal-trained row uses one seed, final-epoch selection, no validation split,
overlapping train/test Mach ranges, and no grouped geometry holdout. Its causal
nodal policy is physically motivated but not exact DG replay. The frozen public
checkpoint under legal boundaries remains a sensitivity counterfactual, not a
fairly trained autonomous baseline. The oracle-assisted public row must not be
used as a fair baseline against autonomous methods unless every method receives
the same future boundary information.

Provenance remains tiered. The historical manifest's public evaluation-runtime
pins are complete, but its training closure omits `utils/lossCompute.py` and
`utils/noise.py`; its exact trainer and boundary-utility hashes match archival
commit `4654225e`, while local provenance/Euler hashes and a verified training
worktree are absent. The run also predates the hardened all-frame mesh/outflow,
exact stencil/audit binding, finite-gradient, optimizer-state,
recurrent-outflow, and physical admissible-prefix checks; these are not
retroactive evidence. Its normalizers reached the
configured cap before multistep training, and the final rollouts are positive,
but earlier teacher-forced raw pressure was negative. Treat positivity as an
observed checkpoint result, not a guaranteed property.

Line 2 is closed at this bounded result. Phase 5 is omitted because phases
2--4 did not isolate a decoder mechanism and the required physical graph
mapping is absent. Do not run the three-way decoder ablation, another seed
sweep, conditional-manifold campaign, or additional CPG training under the
current research budget. Preserve the unresolved items as claim limits and
transfer the legal-boundary, recurrent-exposure, message-depth, shared-face,
and shock-diagnostic lessons to Line 3.

### Line 3: Geometry-Aware 2D Rollout And Targeted Stabilization

#### Core question

Can a global geometry-aware neural operator provide a stable and accurate
medium-horizon 2D macro-solver for shocks and moving complex flow structures,
and which representation or local correction is required once the actual
failure mechanism is known?

#### Two-testbed rule

1. The CPG supersonic bump is a reproduction-audit, attached-shock,
   causal-reach, boundary-contract, and low-dimensional-manifold test. Its present HDF5
   contract lacks a validated control-volume mesh mapping, so it cannot support
   strong physical-conservation claims. It also lacks cumulative reference face
   impulses, which separately blocks direct physical-flux supervision and
   reference-flux error claims.
2. A dynamic 2D finite-volume benchmark must provide validated volumes, face
   measures, normals, orientation, and boundary accounting before physical
   conservation claims. It must additionally export cumulative accepted-
   substep face impulses before direct flux-target claims. It must contain
   moving shocks or interactions, not only a geometry-anchored pattern.

#### Baseline contract

On identical physical trajectories and splits, temporal stride, state
convention, legal boundary treatment, raw recurrence, and evaluation horizons,
establish the following. Audit architecture-specific preprocessing rather than
forcing PCNO, MPCNO, and CPGNet through one internal mesh representation:

- the available released or local-bundle CPGNet reference under released and
  legal-boundary evaluation, without calling it paper-faithful until Line 2
  closes provenance;
- a PCNO conservative-variable residual baseline, with conserved totals kept as
  outcome metrics rather than guaranteed structure;
- MPCNO conservative-variable residual as a later local-global comparator;
- persistence, geometry/time-only, and nearest-trajectory controls; and
- coarse CFD at matched runtime or matched error.

The current PCNO checkpoint predicts positive primitive next state; it is not
the required residual baseline. There is no active time-dependent MPCNO Euler
trainer, and its geometry contract needs normals not present in the current
Euler artifact. Recover checkpoint modes, layers, split, weights, node order,
and preprocessing before comparing models. Do not revive the raw-HDF5 PCNO
adapter or call an equal-node sum physical conservation.

#### Diagnostic gate before method design

Complete graph-native D013 first and consume the CPG D014 result owned by Line
2. Line 3 adds only architecture-specific PCNO/MPCNO reach analysis:

- verify mesh-to-graph topology before interpreting characteristic paths;
- distinguish finite-hop local branches from PCNO/MPCNO's globally dependent
  spectral branch, for which the relevant question is information, basis, and
  approximation quality rather than a local receptive-field obstruction;
- use graph spectral bands rather than interpolation-to-grid FFT as primary
  evidence. If physical masses are unavailable, label equal-node or
  reconstructed weights as proxies and report sensitivity to both;
- audit the PCNO Fourier-basis Gram matrix, conditioning, and quadrature
  leakage in addition to graph-Laplacian bands;
- compare teacher-forced, perturbed-reference, rollout-state, and pre-failure
  calls;
- instrument the spectral, pointwise, local differential, and residual branches
  of PCNO/MPCNO; and
- separate shock position, strength, and thickness from smooth-region ripple
  energy, admissibility, conservation, and boundary leakage.

Correlate ripple growth with mesh density, cell size, local stencil condition,
boundary distance, and shock distance. Classify the dominant failure as
aliasing, quadrature/geometry conditioning, recurrent amplification, or
front-phase error.

#### Failure-to-method routing

Promote exactly one first method branch:

- aliasing born inside spectral layers -> internal spectral shaping,
  de-aliasing, or mass-orthogonalization;
- localized correctable ripples on the validated finite-volume testbed -> a
  global residual operator plus a small shock-aware antisymmetric conservative
  edge correction, preceded by an oracle constrained-decomposition test;
- dominant front displacement on the validated finite-volume testbed ->
  front-factorized representation with conservative remapping rather than
  smoothing. On the current bump artifact, restrict this to a front-
  representation diagnostic;
- mesh-correlated error -> repair quadrature, geometry scaling, or local
  differential conditioning before changing the network; or
- clean teacher-forced behavior with recurrent growth -> short generated-state
  exposure only after the one-step gate.

Reject a filter that reduces high-frequency energy by broadening, weakening, or
displacing the shock. Reject a local corrector active over most of the domain or
with update norm comparable to the global model. Reject gains confined to the
attached-shock bump that disappear on dynamic or grouped-OOD tests.

#### Separate 2D structured-target study

On validated finite-volume data, compare four distinct rows under one matched
backbone: conservative-variable cell residual, state-loss-only shared-face
impulse, divergence-active or projected impulse supervision, and full direct
reference-impulse supervision. In 2D, face flux has a large cycle-space
nullspace, not only the single 1D constant-flux gauge. Keeping the last three
rows separate tests whether the cycle component is redundant or harmful. On
the CPG bump, an interface experiment is a decoder ablation, not evidence that
a physical interface target works.

No broad basis, smoothing, flux, or architecture sweep is approved before
D013-2D and D014 classify the failure.

#### Current Line-3 gate result

The frozen D013 paired branch-response extension is complete on the selected
clean 19,155,720-parameter residual PCNO. All five historical validation
trajectories complete 20/20 raw calls. At call 10, both repeated trajectory
rows select the pointwise branch: it has the largest finite-response RMS gain
in 3/4 layers and the largest response roughness in 4/4 layers. The frozen
selector therefore routes `local_pointwise_control`. This is next-experiment
routing from a finite perturbation along the observed rollout-error direction;
it is neither a Jacobian estimate nor causal branch attribution.

The bump artifact cannot complete the required oracle correction gate because
its reconstructed node weights are diagnostic proxies and it lacks validated
control-volume volumes and oriented physical faces. Do not substitute proxy
mass balance for conservation, and do not train a pointwise corrector or sweep
branch gains on this artifact. D037 has now closed the dynamic finite-volume
testbed contract. The next method-facing authorization is limited to freezing
its narrow perturbation split and serious global baseline, then running the
predeclared conservation-, locality-, norm-, admissibility-, and
anti-smearing-constrained oracle decomposition there. A learned correction
remains unauthorized until that oracle passes.

#### Selected method hypothesis and dynamic oracle contract (2026-07-19)

**Verified:** the clean global conservative-residual PCNO is the current strong
2D baseline; D013 routes the next diagnostic toward the local pointwise path;
turning that path off worsens truth error; and the bump bundle cannot support a
physical conservation test. **Inferred:** the most economical hypothesis is
that the global macro-step map is useful but leaves a small shock-generated,
locally rough residual that may be representable by shared conservative face
impulses. This is not yet a causal explanation. **Must be checked before any
learned run:** reference-solver convergence, accepted-step and saved-time
provenance, physical control volumes and oriented faces, raw baseline behavior
on the new benchmark, and the frozen oracle gates below.

The problem anchor is deliberately narrow: preserve the fewer-call benefit of
a large-step global flow map while suppressing its local ripple without
broadening, weakening, or displacing a shock or degrading the transported
vortex. This does not reopen Line 1, approve a timestep sweep, or make generic
local-global or conservation claims. Recent local-global neural operators
already combine Fourier and local multiresolution paths
([LGNO](https://arxiv.org/abs/2606.18221)); hybrid finite-volume/neural-operator
methods already impose flux structure
([Flux Neural Operators](https://arxiv.org/abs/2605.05488)); and iterative
frequency refinement is already represented by
[PDE-Refiner](https://papers.neurips.cc/paper_files/paper/2023/hash/d529b943af3dba734f8a7d49efcb6d09-Abstract-Conference.html).
Any contribution here must therefore be the measured macro-step/call-count
tradeoff and the constrained interface between one global macro map and one
small conservative correction, not the existence of a local branch.

The first dynamic finite-volume gate uses the canonical Mach-1.1
shock--isentropic-vortex interaction on `[0,2] x [0,1]`: shock at `x=0.5`,
left primitive state `(1, 1.1*sqrt(gamma), 0, 1)`, right state
`(1.1691, 1.1133, 0, 1.245)`, and the published vortex centered at
`(0.25,0.5)` with `epsilon=0.3`, `alpha=0.204`, and `r_c=0.05`. Use symmetry
boundaries in `y`, extrapolation in `x`, and evaluate through `t=0.6`, matching
the public benchmark contract
([benchmark source](https://link.springer.com/article/10.1007/s10915-021-01743-1)).
Generate a documented WENO5-HLLC-SSPRK3 reference, or a numerically equivalent
documented finite-volume reference, at `1000 x 400`; require convergence and
agreement with an independent public solver before conservative restriction to
`250 x 100`. Save every accepted-step time and rejection count. Do not invent a
parameter distribution until the canonical case closes; held-out cases are a
later requirement for promotion, not a reason to start a broad generator now.
After canonical closure, predeclare one narrow perturbation family and grouped
train/validation/test split before fitting the frozen global baseline.

For frozen current state `U_n`, global prediction `U_g=G_DeltaT(U_n)`, cell
volumes `V`, and owner-oriented cumulative interior-face impulses `I`, decode
one candidate correction as

```text
delta U_owner    = -I_f / V_owner
delta U_neighbor = +I_f / V_neighbor
U_corr           = U_g + alpha * delta U .
```

The truth-informed oracle may choose the top 20% of interior faces ranked by
the maximum normalized endpoint truth error of their two cells. It then solves
a volume- and component-normalized least-squares problem on only that frozen
support. Line search must enforce raw positive density and pressure, zero
boundary correction, exact interior cancellation to numerical tolerance, and
`||alpha delta U|| <= 0.10 ||U_g-U_n||` in the same normalized weighted norm.
The anti-smearing callback must reject a correction that worsens any frozen
shock-position, shock-strength, shock-thickness, vortex-core, or smooth-region
high-pass diagnostic by more than 5% relative to `U_g`. Truth-selected support
makes this an upper-bound mechanism diagnostic, never an inference algorithm.

Run the oracle on predeclared early-interaction, peak-interaction, and `t=0.6`
snapshots after freezing the baseline and diagnostic definitions. Promotion
requires all rows to remain raw-admissible and conservative, correction support
and norm to respect the 20%/10% caps, no anti-smearing violation, at least 15%
median reduction in normalized state error, at least 20% median reduction in
smooth-region high-pass error, and at least 75% of rows non-worse on both error
families. Report all rows, including zero-correction optima. Failure kills the
learned corrector. If the remaining error is primarily front displacement,
route once to a front-factorized prediction with conservative remapping;
otherwise leave the mechanism unresolved rather than opening an architecture
sweep.

Only an oracle pass authorizes one learned design. Keep the frozen global
backbone and predict one stored impulse per physical interior face from
owner/neighbor current states, global predicted states and updates, normalized
jumps or graph high-pass features, face geometry, `DeltaT`, and characteristic
travel. Enforce orientation antisymmetry by
`I_f = 0.5*(phi(z_o,z_n,n)-phi(z_n,z_o,-n))`, initialize the correction at
zero, derive one orientation-invariant activity score per physical face, and
retain a deterministic top 20% of interior faces. After decoding, rescale every
impulse by the common factor
`min(1, 0.10*||U_g-U_n||/(||delta U||+epsilon))`; this preserves cancellation
while imposing the oracle update budget. Leave boundary impulses zero
initially. Train the corrector first with the backbone
frozen, one macro correction per call, raw recurrence, and one fixed
state-plus-smooth-graph-difference objective. No learned micro-rollouts,
clipping, primitive floors, coefficient sweep, or generic pointwise damping is
authorized. If two output intervals are later needed, choose them by a
predeclared `p99` characteristic travel of approximately two versus eight
coarse cells per call, rounded to divisors of `t=0.6`; do not repeat the
1/2/4/8 Line-1 frontier.

The reusable oracle scaffold now lives in
`utility/time_dependent_no/conservative_correction_oracle.py`, with synthetic
CPU tests in `tests/time_dependent_no/test_conservative_correction_oracle.py`.
The scaffold remains a truth-informed diagnostic rather than an inference
method; its completed dynamic result is recorded below.

#### Dynamic reference execution and closeout decision (2026-07-20)

**Verified canonical contract:** the final ignored audit
`shock_vortex_fv_convergence_cellavg_sharpclaw_matchedbc_20260720c` has schema
`shock_vortex_fv_convergence_audit_v3`, status `benchmark_contract_closed`, no
failed checks, and both benchmark and direct-reference-impulse closure booleans
true. The WENO5-HLLC-SSPRK3 generator in
`utility/time_dependent_no/shock_vortex_fv.py` emits certified conservative cell
averages, physical cell volumes, face measures, unit normals, oriented
owner/neighbor connectivity, exact accepted-step and saved-time provenance,
conservative 250x100 restriction, and cumulative accepted-substep face
impulses. The 250x100, 500x200, and 1000x400 runs reach `t=0.6` with 822, 1642,
and 3279 accepted steps, no rejected attempts or reconstruction fallbacks, and
maximum interval balance-closure error `9.38609e-13`. Final-state,
time-mean-state, and centerline-density successive ratios are `0.484166`,
`0.481540`, and `0.397123`.

**Verified independent state agreement:** pinned Clawpack 5.9.0 build
`py311h3d4ca6a_1` SharpClaw uses fifth-order WENO, SSP33, the four-wave Euler
Riemann solver, custom linear primitive-variable x extrapolation, and reflecting
y walls. It matches the primary initial cell averages to `3.18165e-16`. Its
final and time-mean component-normalized state errors are
`0.0315534 < 0.0466208` and `0.0156538 < 0.0233192`; final shock-position
difference is zero and vortex-core density relative difference is `0.00215749`.
This closes one frozen, boundary-matched state envelope. SharpClaw is state-only
and exports no reference face impulses.

**Verified impulse-label boundary:** after conservatively restricting the
medium/fine states and impulses to the common 250x100 mesh, primary-ladder
full-face-vector, divergence-active, cycle, and boundary impulse-error
contraction ratios are `0.463428`, `0.463362`, `0.463482`, and `0.464003`; all
componentwise ratios are at most `0.464399`. Maximum decomposition
reconstruction, cycle divergence,
weighted orthogonality, and decoded-transition residuals are `2.22363e-17`,
`2.71952e-9`, `3.29544e-15`, and `1.31988e-17`. The v3 audit uses a tightened
`1e-11` decomposition solve tolerance with the unchanged `1e-8` cycle gate.
Active boundary exchanges contract, while analytically zero reflecting-wall
components and their actual primary exchanges remain below `1e-14`. These
checks close same-primary-solver full common-mesh face-impulse labels, not
native fine-grid face vectors, cross-solver flux agreement, or a unique
physical face field. The 2D cycle component remains discretization-specific
and non-identifiable.

**Superseded and historical evidence:** preserve
`shock_vortex_fv_convergence_cellavg_sharpclaw_matchedbc_20260720b`, schema
`shock_vortex_fv_convergence_audit_v2`, as fail-closed provenance. Its
`failed_convergence` status came from one-ULP saved-time/config equality, an
insufficient `1e-10` LSMR solve tolerance for the unchanged `1e-8` cycle gate,
and relative contraction ratios on analytically zero wall components; it was
not physical nonconvergence or independent-solver disagreement. The pinned
[Pyro](https://github.com/python-hydro/pyro2) 4.5.0 CTU/HLLC state mismatch is
also retained as a historical low-order comparison. Its final/time-mean errors
`0.200259/0.092201` exceeded `0.047538/0.025736`, without identifying which
discretization was more accurate.

**Decision:** the canonical benchmark and same-primary-solver direct-impulse
data contract are promoted. Next predeclare and freeze one narrow perturbation
family and grouped train/validation/test split, then fit one serious dynamic
conservative-residual global baseline. Freeze the baseline and diagnostics
before running the predeclared 20%-support/10%-update oracle. This closeout is
not evidence of neural baseline quality, oracle correction headroom,
learnability, or a learned method. Only an oracle pass may authorize the one
zero-initialized orientation-antisymmetric correction defined above; an oracle
failure retains the existing front-displacement routing and kill criteria.

#### Dynamic residual baseline and method-routing closeout (2026-07-21)

**Verified global baseline:** the frozen 135-case family
`shock_vortex_eps_y_15x9_fine1000_to_250_dt001_v1` passes 135/135 artifact
audits and uses the grouped 84/24/27 train/validation/strength-OOD split. The
full-resolution seed-20260718 residual PCNO has 19,155,720 parameters and
predicts conservative-variable state residuals with training-only primitive
noise. Raw all-24 H60 validation selected epoch 44, checkpoint SHA-256 beginning
`c5e468c7`, rather than the final epoch. Physical-volume state error is
`0.00028009` at call 1 and `0.00834190` at call 60; all 24 trajectories remain
finite and admissible without a reference boundary, floor, clip, smoother, or
limiter. It beats persistence, nearest-training-trajectory, and four-neighbor
train-only parameter/time interpolation at H60 on 24/24 paired cases; their
mean errors are `0.0732949`, `0.0231132`, and `0.0231196`.

This is not a conservative neural update. The model emits no shared-face
impulse, and its H60 normalized physical-total mismatch relative to reference
boundary exchange is `0.0513185`, worse than the train-manifold controls.
Conserved totals and boundary exchange remain outcome diagnostics rather than
architectural guarantees; no neural flux claim is authorized.

**Verified D013 formation evidence:** the predeclared six validation cases all
complete H60 and consume the frozen Line-2 D014 interface. The actual basis
Gram condition number is `1.000003`, off-diagonal Frobenius ratio is `4.80e-7`,
and naive-to-mass-orthogonal reconstruction RMSE ratio is one to numerical
precision. The spectral response is only `0.0516` as rough as the other branch
responses. Pointwise has the largest paired RMS gain in all four layers while
differential is roughest in three; no one branch wins both criteria, and the
selector returns `composite_or_unresolved`. Strong paired error-response
cancellation is falsified. Smooth high-pass error grows by roughly four orders
of magnitude from first to late interaction in the deep rows, but late rollout
to teacher high-band energy is only `1.0458` and the accepted perturbation gain
is `0.9419`. All predeclared spectral-aliasing, quadrature/geometry,
front-phase, and recurrent-amplification screens are false. The mechanism is
`unresolved`: the data do not support a pure Gibbs explanation or a single
local-branch culprit.

**Verified constrained-oracle rejection:** on the frozen six-case cohort at
calls 10/30/60, all 18 truth-informed rows are raw-admissible, internally
conservative to at most `4.75e-19`, within the 20% support and 10% update caps,
and pass every anti-smearing gate. Median state-error reduction is only
`0.05135` and median smooth-high-pass reduction is only `0.04164`, below the
predeclared `0.15/0.20` gates. The learned sparse local correction is rejected;
do not relax its caps or start a support/norm sweep. The front-phase screen is
also false, so no front-factorized method is selected. Keep the strength-OOD
test split sealed.

**Decision:** no stabilization training run is authorized from D013. Do not
replace the global Fourier branch with a fixed local basis merely because the
solution contains a shock: the measured basis contract is well-conditioned
and the spectral branch is the smoothest path. The next eligible model-facing
question is the separately predeclared structured-target test on this validated
finite-volume family, beginning with contract/tiny-fit closure and at most one
serious state-loss-only shared-face-impulse row. Divergence-active and full
reference-impulse supervision remain separate later rows; do not launch them as
a sweep. This target study tests whether exact shared-face decoding changes the
failure, not whether the rejected local correction can be rescued.

#### State-loss-only shared-face tiny-fit closeout (D045, 2026-07-21)

**Verified contract:** the bounded implementation adds a 19,210,028-parameter
PCNO shared-face model on the unchanged full 250x100 finite-volume mesh. One
interior-face value is decoded with exact owner/neighbor cancellation; boundary
heads use only the current state and validated geometry, and reflecting y walls
permit only y-momentum exchange. The source training artifact, physical face
arrays, PCNO graph, and mesh-to-graph mapping are digest-bound and reconstructed
before use. Training is state-loss-only: cumulative reference face impulses are
neither loaded nor supervised. Zero initialization exactly recovers persistence,
and focused CPU/remote tests plus the corrected execution smoke pass.

**Verified tiny-fit result:** the four fixed full-resolution training pairs do
not pass the predeclared joint gate. With 800 optimizer updates, best mean state
relative error is `0.00166965` and the best-to-initial loss ratio is
`0.0730244`. The one permitted retry changes only presentations of those same
four pairs, reaching 3,200 updates; its best relative error is `0.000955675`,
but its loss ratio is `0.0254377`. It therefore passes the absolute `0.001`
error threshold and fails the required `0.01` ratio. All pairs improve and
remain admissible. A read-only replay gives mean relative errors `0.00123513`
for the trained bfloat16 path, `0.00121919` with float32 face heads, and
`0.00127140` in full float32, so precision does not explain the measured floor.

**Decision:** the serious state-loss-only shared-face row is stopped. Do not
relax the tiny-fit gate, add another exposure retry, downsample a purportedly
serious model, or open the 27-case test split. Exact decoder balance means only
that the learned cell update equals the model's predicted boundary exchange;
it does not establish accurate physical boundary exchange or a conservative
neural solver. This result rejects the current direct state-loss-only
parameterization at its entry gate, not all flux/interface formulations. Any
divergence-active or minimum-norm row requires a separate bounded authorization
and should begin with a zero-training canonical-projection preflight. Full
reference-impulse supervision remains a distinct identifiability row rather
than a sweep companion. D045 does not modify the frozen Line-4 handoff:
training truth remains authorized only for the audited family, no front
candidate is available, and transition training remains false.

#### Divergence-active canonical-target preflight (D046, authorized 2026-07-21)

D046 is authorized only as a zero-training identifiability and numerical-
conditioning preflight. It uses the full 250x100 validated finite-volume mesh,
the fixed weighted complement of the interior cycle space, and the accepted
reference boundary impulses. The canonical interior target is the minimum
`W_f^{-1}`-norm owner-oriented face field whose divergence matches the cell
transition after subtracting physical boundary exchange. It must be reproduced
independently from reference states plus boundary impulses; the full reference
interior cycle field may be measured but is not a supervision target.

The frozen cohort is 24 rows: train cases `sv_e00_y04`, `sv_e03_y04`,
`sv_e07_y04`, and `sv_e11_y04`; position-OOD validation cases `sv_e00_y00`,
`sv_e03_y08`, `sv_e07_y00`, and `sv_e11_y08`; and saved calls 1, 30, and 60.
The strength-OOD test split is forbidden. Before results are read, promotion
requires every provenance, finiteness, and split check plus all of these maxima:
reference-impulse/reference-state closure `1e-10`, canonical/reference-state
closure `1e-8`, canonical/float32-shard increment closure `1e-4`, independent
state-plus-boundary/canonical field disagreement `1e-5`, compatibility residual
`1e-10`, cycle-divergence relative residual `1e-8`, canonical/full weighted-
norm ratio `1.000001`, and forbidden reflecting-wall exchange `1e-14`.
Every sparse solve must terminate with LSMR code 1 or 2.

Attempt `fv_divergence_active_target_preflight_20260721b` completed all 24 rows
but exposed a semantic implementation error before promotion: the code labeled
the projection of the full reference face field as the canonical target and
used the state-plus-boundary minimum-norm solve only as a cross-check. The
registered definition above is the reverse. That route reached maximum
reference-state closure `1.82559e-8` and therefore remains a failed artifact.
One implementation-corrected rerun may make the state-constrained solution the
canonical target and retain the reference projection only as the independent
agreement check. The cohort, gates, LSMR tolerance, iteration cap, and all
promotion restrictions are immutable; this is not a tolerance retry.

The corrected attempt `fv_divergence_active_target_preflight_20260721c`
completed all 24 rows in 111.4 seconds. It failed only the unchanged canonical/
reference-state closure gate: every row is between `1.67298e-7` and
`4.94497e-7`, versus the required maximum `1e-8`, with overlapping train and
position-OOD validation ranges. The other maxima pass: reference-state closure
`1.23427e-12`, float32-shard closure `9.36905e-6`, independent canonical-field
disagreement `1.61785e-8`, compatibility `2.94020e-12`, cycle divergence
`5.18524e-10`, canonical/full weighted-norm ratio `0.999440`, and forbidden
wall exchange `3.10460e-20`; every sparse-solve code is accepted. The reference
cycle fraction is `0.00112859--0.00669102` of interior `W_f^{-1}` energy, with
median `0.00264596`, and the canonical/full weighted-norm ratio ranges from
`0.996677` to `0.999440`.

**Decision:** D046 fails its preregistered proof-level numerical gate, so no
divergence-active supervised tiny fit, serious run, loss-weight sweep, full-
cycle supervision, or test-split evaluation is authorized. The passing
geometry, boundary, compatibility, reference-closure, cycle, and independent-
field checks show that the finite-volume contract and practical target
identifiability are not the measured blocker; the current iterative projector
is. Do not relax the gate or make another LSMR/tolerance retry. A fixed-mesh
direct-factorization projector may be considered only as a separately
preregistered numerical-method experiment with the same sealed test boundary;
it does not inherit training authorization. D046 does not change any Line-4
handoff flag.

#### Fixed-mesh direct canonical projector (D047, authorized 2026-07-22)

D047 is one zero-training numerical-method preflight, not a tolerance retry or
a learned-method row. It tests the specific D046 diagnosis that iterative LSMR
accuracy, rather than face-target incompatibility, blocked proof-level closure.
The data cohort is unchanged: the same four train and four position-OOD
validation trajectories at saved calls 1, 30, and 60, for 24 rows total. The
27 strength-OOD test trajectories remain forbidden.

For interior incidence `B_i`, interior dual-volume weights `W_i`, reference
boundary impulse `I_b`, and requested cell integral `q`, define
`r = q - B_b I_b`. Within each connected component, explicitly project `r`
onto the incidence range by subtracting its componentwise arithmetic mean and
record the removed compatibility component. Form the weighted graph Laplacian
`L = B_i W_i B_i^T`, anchor the lowest global cell index in each component to
fix only the potential gauge, and factor each reduced float64 sparse matrix
once. Reuse those factors for every variable and row, then recover the unique
minimum-`W_i^{-1}`-norm interior impulse as `I_i = W_i B_i^T phi`. No diagonal
jitter, regularization, iterative refinement, alternative anchor, tolerance
change, or solver sweep is allowed. The gauge anchor cannot change `I_i`.

D046's projection of the full accepted reference field remains the independent
field check; it does not define the direct target. All D046 maxima remain
unchanged: reference/canonical/shard closure `1e-10/1e-8/1e-4`, independent
field disagreement `1e-5`, compatibility `1e-10`, cycle divergence `1e-8`,
canonical/full weighted-norm ratio `1.000001`, and forbidden wall exchange
`1e-14`. D047 additionally requires the explicit compatibility projection to
be at most `1e-10` relative to the unprojected interior right-hand side, the
reduced-Laplacian solve residual to be at most `1e-10`, exact factor reuse, and
finite factors, potentials, face fields, and decoded states. Provenance,
source-digest, geometry, orientation, split, and test-nonaccess checks remain
mandatory.

Only synthetic local and remote CPU checks may precede the single frozen-data
execution. An infrastructure failure that produces no scientific rows may be
restarted; once numerical rows exist, a failed gate stops D047 without another
factorization variant. A complete pass authorizes only a separately registered
four-pair canonical-target tiny fit. It does not authorize a serious model,
full-cycle supervision, a loss/solver sweep, test access, or any physical-
conservation claim. Failure kills the projected-target training route. D047
does not change any Line-4 handoff flag.

The single artifact `fv_direct_canonical_target_preflight_20260722a` completes
all 24 registered rows in 41.7 seconds and passes every gate. Canonical/
reference-state closure is `1.09848e-9--3.40147e-9` with median
`1.96139e-9`, improving every matched D046 row by `144.97--157.64x`. Train and
position-OOD validation ranges overlap. Maximum reference closure, float32-
shard closure, independent-field disagreement, compatibility, compatibility-
projection, reduced-solve residual, cycle divergence, weighted-norm ratio, and
forbidden wall exchange are respectively `1.23427e-12`, `9.35801e-6`,
`8.42560e-10`, `2.94020e-12`, `1.85905e-14`, `1.13351e-12`, `5.18537e-10`,
`0.999440`, and `3.10460e-20`. One 1,130,356-nonzero float64 factor is reused
for all rows without regularization or refinement. The test split is untouched.

**Decision:** D047 confirms that D046's failure was numerical-projector
accuracy under the tested implementation, not target incompatibility. It
authorizes only the D048 tiny-fit contract below; it does not establish learned
flux accuracy or physical conservation and does not authorize a serious run.

#### Canonical-face supervised tiny fit (D048, completed 2026-07-22)

D048 uses exactly the D045 architecture and maximum tiny-fit exposure on the
unchanged 250x100 mesh: seed `20260718`, `kmax=8`, five width-128 layers,
width-128 fully connected tail, face latent width 32, face-head width 128,
domain `2 x 1`, zero output initialization, batch size 1, bfloat16 autocast
with float32 loss reductions, AdamW at constant `1e-3`, weight decay `1e-5`,
gradient clip 1, 50 epochs, and 64 presentations per epoch (3,200 optimizer
updates). Input noise, generated-state exposure, clipping, floors, limiters,
future-reference boundaries, downsampling, and checkpoint initialization are
forbidden.

The immutable training bank is `sv_e08_y07@8`, `sv_e03_y03@7`,
`sv_e03_y01@44`, and `sv_e05_y07@41`, where each index denotes the current
saved state and the target is the next saved state. For each pair, construct
the float64 canonical field from the reference state increment and accepted
reference boundary impulse using the exact D047 factor; validate the D047
closure and wall gates; then cast the label once to float32 and require its
decoded/reference-state closure to be at most `1e-4`. Record every source,
label, factorization, geometry, split, normalization, configuration, and code
digest. No test case may be opened.

The optimized objective contains no decoded-state term:

`L_face = 0.5 L_interior + 0.5 L_boundary`,

where each term is the mean across the four conservative components of squared
relative `W_f^{-1}` error on that face subset. This directly constrains the
minimum-norm divergence-active interior field and the accepted physical
boundary exchange while leaving the decoded state as an outcome metric. The
zero-output initial checkpoint must reproduce persistence and finite unit native
relative error. Promotion requires, at one common saved epoch, native loss
ratio at most `0.01`, full/interior/boundary `W_f^{-1}` relative errors each at
most `0.10`, decoded-state relative L2 at most `0.001`, 4/4 admissibility,
exact structural zero on forbidden reflecting-wall components, and finite
gradients, predictions, labels, and metrics.

There is one run and no optimization retry, loss-weight change, state-loss
mixture, target recasting, precision variant, or validation-selected rescue.
Failure stops the canonical-face target branch. Passing only permits a later
decision about one matched serious row; it does not authorize that row, full-
reference-cycle supervision, test evaluation, or a conservation claim. D048
does not change any Line-4 flag.

The single artifact
`pcno_canonical_face_tinyfit_s20260718_20260722a` completes all 3,200 updates
in 195.4 seconds and fails the joint gate. Its summary and epoch-table SHA-256
digests are `cbfa1e90993e...` and `0652829297ce...`; the exact label-set digest
is `a84242bd67d1...`, and the test split remains unopened. The real-data label
preflight passes before training: maximum float64 canonical/reference closure
is `2.77918e-9`, the one-time float32 label closure is `3.34619e-5`, and
forbidden wall exchange is `2.33260e-20`.

Face-space optimization succeeds under the registered metric. Fifteen saved
epochs satisfy the loss-ratio and full/interior/boundary face-error gates, first
at epoch 24. The minimum loss ratio is `0.00707672` at epoch 28; minimum full
and interior relative errors are `0.0950046/0.0953408` at epoch 31, and minimum
boundary error is `0.0201020` at epoch 43. No saved epoch is admissible or
reaches decoded-state error `0.001`. The best decoded-state error is `0.474958`
at epoch 33, versus zero-output persistence error `0.00628209`; the final error
is `0.500998`, `79.75x` persistence, with 0/4 admissible predictions.

A descriptive, post-hoc replay of the final checkpoint explains why the
native fit is insufficient without changing the gate. Across the four frozen
training pairs, full `W_f^{-1}` face error is `0.09772--0.09877`, but the
physical-volume cell-increment error is `98.91--155.90` times the true
increment. The error field's weighted face-to-divergence gain is
`1.995--2.002`, while the canonical target gain is only
`0.001264--0.001999`, yielding `1001--1581x` relative divergence
amplification. Minimum density/internal energy/pressure are
`-0.61196/-80.7938/-32.3175`. This replay is descriptive training-pair
evidence, not a preregistered promotion metric; its digest-bound JSON has
SHA-256 `c71bbff63265...`.

**Decision:** D048 is a native-face-fit but discrete-divergence-conditioned
failure. It rules out this exact face-value `W_f^{-1}` objective and stops the
canonical-face target branch; do not retry it with a state-loss mixture, tune a
weight, run a serious row, inspect test, or claim physical conservation. The
result is consistent with high-divergence face-error modes being amplified by
the finite-volume difference update, but it does not identify their
architectural source or prove Fourier/Gibbs causation. In particular, D044's
spectral branch was the smoothest traced branch, so this is not authorization
for a Fourier-to-local-basis sweep. The conservative-residual D044 model remains
the flagship 2D PCNO baseline. Any later face-form proposal needs a new
zero-training conditioning contract, such as an `H(div)`-aligned error or a
deterministic face lifting of an already accurate cell residual, before a new
training authorization. D048 changes no Line-4 flag.

#### D049 divergence-conditioning closeout and D050 registration (2026-07-22)

D049 is complete in
`fv_divergence_conditioning_d049_20260722a`. Physical perturbations use only
the accepted 250x100 dynamic finite-volume artifact. The 500x200 and 1000x400
rows rebuild analytic Cartesian geometry and are algebraic decoder controls;
they contain no native fine-grid shock states or face-impulse truth. Three
seeded graph bands have median normalized frequencies
`0.01523/0.11917/0.63531` and decoded gains
`0.34903/0.97641/2.25444`. Every seed orders low below mid below high, the
minimum high/low ratio is `6.3866`, and the minimum high-band gain is
`369.06x` the largest physical canonical-target component gain. The exact
reference cycle has only `0.0837--0.2074%` of full `W_f^-1` energy and decodes
with gain at most `1.24e-10`.

All six frozen D049 gates pass. Canonical closure is at most `6.61e-9`; fixed-
macro-step flux gain doubles by `2.00009x/2.00002x` on the two analytic 2x
refinements, while cumulative-impulse top gain remains approximately `2.8284`.
This fixed-`Delta t` result is an effective-CFL/algebraic statement, not a
claim that physical refinement is unstable. D048's documented error-field
gain `1.995--2.002` lies between the D049 mid/high bands. Thus D049 supports a
face-norm/discrete-divergence-conditioning diagnosis, but it does not locate
the learned architectural source, prove Fourier/Gibbs causation, or authorize
a local-basis, smoothing, learned-face, or loss sweep.

D050 is registered before reading its frozen trajectory arrays. It is one
zero-training residual-to-face preflight on the exact D013 validation cohort
`sv_e00/e06/e11 x y00/y08` at calls `1/10/30/60`; all 27 strength-OOD test
cases remain sealed. The legal control deterministically allocates the D044
predicted total cell integral over `x_min/x_max` faces by the minimum-
`W_f^-1` rule, then applies the once-factored direct interior lift. It uses no
future reference, but that allocation is algebraic and has no physical-
boundary-flux claim. A separate oracle row uses accepted cumulative reference
boundary impulses only to measure headroom and is never an autonomous result.

The legal reconstruction, legal lift closure, and oracle lift closure must be
at most `1e-8`; forbidden legal wall exchange must be at most `1e-14`; and all
raw/legal/oracle endpoints must remain admissible. Oracle headroom additionally
requires median H60 state-error reduction at least `0.15`, median physical-
budget-defect reduction at least `0.80`, correction norm at most `0.10` of the
raw cumulative update, and median H60 front-chamfer, shock-strength,
shock-thickness, and smooth-error ratios no greater than `1.05`. Structural
failure kills D050. Structural passage with insufficient headroom stops the
face route and keeps D044. Passage of both sets permits only a new legal
current-state boundary-budget preflight; it does not authorize training. D050
changes no Line-4 flag.

The single D050 artifact
`pcno_residual_face_lift_d050_20260722a` is preserved as failed provenance.
All 24 paired and 72 state rows complete and remain admissible. The legal row
reconstructs D044 to `5.37e-12`, has reported target residual at most
`7.58e-11`, and puts exactly zero impulse on non-x boundary faces. However, the
implementation bound the registered oracle-lift closure field to the
intentional difference from the *unprojected* incompatible D044 residual,
rather than the reduced direct-solve residual after compatibility projection.
That field reaches `0.07610` and formally fails the frozen `1e-8` structural
gate. The code now separates compatible-system closure from target-projection
magnitude, but D050 is not rerun or relabeled.

The saved future-reference oracle readout is independently below its material-
headroom gate. At H60 it reduces median physical-budget defect from `0.05206`
to `4.04e-12` without an anti-smearing regression and uses at most `0.05467` of
the raw cumulative-update norm, but median state error improves only
`7.41%` (`0.008673` to `0.008055`) against the required `15%`; all six cases
lie between `5.09%` and `9.54%`. Thus even a corrected closure field would not
promote this route. Stop D050 without a retry, learned boundary model, learned
face model, test access, or conservation claim. Keep D044 as the flagship 2D
state-residual baseline and leave every Line-4 flag unchanged.

#### D051 paired coarse-CFD error--cost registration (2026-07-22)

D051 closes the still-unverified `paired_coarse_cfd_error_cost` field in the
frozen D044 summary. It trains no model and reads no strength-OOD test case.
Accuracy remains the digest-bound 24-case validation rollout selected at epoch
44; a fresh same-host timing-only PCNO replay must use the identical checkpoint,
configuration, normalization, CUDA/bfloat16 mode, raw recurrence, and zero
inference interventions.

The only CFD candidates are nested `25x10`, `50x20`, `125x50`, and `250x100`
float32 WENO5--HLLC--SSPRK3 runs over the same H60 physical horizon. All four
run first on `sv_e00/e06/e11 x y00/y08`. The closest log-cost and log-error
members are then extended to all 24 validation cases; if those are identical,
the combined-distance runner-up is added. The maximum full-validation count is
two and no post-result grid insertion is allowed. Initial states are exact
uniform block restrictions of D044 truth, output remapping is conservative
piecewise-constant prolongation, and restrict--prolong truth is reported as the
oracle information floor.

The synchronized CFD timer includes adaptive-CFL reductions, WENO/HLLC/SSPRK
evolution, retries, save-state retention, and its own physical boundary sums,
but excludes initialization, host export, remapping, metric evaluation, and
artifact I/O. The PCNO comparison uses the analogous synchronized forward
boundary and 60 calls. Strict matched-cost evidence requires timing
p95/median at most `2`, an observed symmetric cost ratio at most `2`, complete
raw admissibility, and maximum own-boundary balance error at most `5e-5`.
Matched-error evidence requires an observed symmetric H60 error ratio at most
`1.25`. Paired intervals resample cases, not model seeds. Any failed gate makes
that result descriptive only and authorizes neither added CFD grids, new neural
training, test access, production-CFD claims, nor a Line-4 coordination change.

#### D051 paired coarse-CFD closeout (2026-07-22)

The completed artifact `pcno_shock_vortex_coarse_cfd_d051_20260722a` selects
`25x10` and `250x100` exactly as registered. All 60 solver trajectories are raw,
finite, and admissible; none uses a rejected step, face fallback, future
boundary, clip, floor, limiter, or smoother. All table and endpoint-artifact
hashes verify, and the test split remains unopened.

No observed cost-matched member exists. The closest-cost 25x10 row takes
`1.938` s versus the calibrated PCNO H60 median estimate `0.449` s and has H60
error `0.53185` versus `0.0083419`. Its H60 restrict--prolong floor is only
`0.022814`, so the `23.3x` excess over that floor is solver evolution error, not
cross-mesh remapping. The 50x20 and 125x50 pilot rows are likewise more
expensive and less accurate than PCNO on the six-case cohort; they are not
extended post hoc.

The 250x100 row passes only the observed error-match gate: H60 error is
`0.00808583`, `3.07%` below PCNO with symmetric ratio `1.0317`. The paired mean
CFD-minus-PCNO difference is `-0.0002561`, but its case-bootstrap 95% interval
`[-0.0008367, 0.0003231]` contains zero and PCNO is lower on 10/24 cases. CFD
has better front IoU/chamfer and vortex-core error; PCNO has better shock-
strength, shock-thickness, and smooth-region scaled error, while CFD has lower
smooth high-pass energy. This is evidence of a shock-smearing/ripple tradeoff,
not evidence that either solver is uniformly more accurate.

Strict error--cost and physical-balance claims fail. The fresh PCNO calibration
has per-call median/p95 `0.007485/0.087588` s, ratio `11.70` versus the frozen
`2` gate. Descriptive 250x100 CFD time is `13.53` s versus PCNO `0.449` s from
the calibrated median and `2.225` s from the frozen observed median. The coarse
solver's maximum float32 own-balance residual is `2.99e-4`, above `5e-5`.
Neither timing nor balance threshold may be retuned after seeing this result.
Close D051 with matched-error and descriptive-cost evidence only. Keep D044 as
the flagship baseline; do not add a CFD grid, claim production-CFD speedup or
physical conservation, access test, start new training, or change Line-4 flags.

Result-to-claim is therefore `partial`: D044 is state-error competitive with
this 250x100 implementation on the frozen validation family, with a different
shock/ripple tradeoff and descriptively lower measured cost. Strict speed,
physical conservation, general CFD superiority, seed uncertainty, and test/OOD
claims remain unsupported. The post-run source edit narrows claim wording only;
the artifact is neither rerun nor relabeled.


#### D052 frozen branch-gain attribution closeout (2026-07-22)

D052 consumes the same six D013 validation trajectories and calls
`1/10/30/60` without training or opening strength OOD. Its code-frozen
one-sided `1.00 -> 0.99` gain probe covers every layer and spectral, pointwise,
and differential branch under both teacher and rollout inputs. Promotion
requires a branch to improve smooth high-pass error by elasticity at least
`0.10`, keep state/front/strength/thickness elasticities above `-0.05`, remain
admissible, pass three of four calls on five of six cases, and repeat under both
input regimes.

Artifact `pcno_shock_vortex_branch_sensitivity_d052_20260722a` is complete:
48/48 rows, maximum exact-replay difference `4.76837e-7`, and no inadmissible
base or attenuated output. Every branch has zero passing cases. Spectral
attenuation has negative median smooth-high-pass elasticity under teacher and
rollout (`-0.0208/-0.0050`). Differential attenuation has only weak positive
values (`0.0324/0.0249`) and simultaneously worsens teacher state/front and
rollout front error. Pointwise is likewise inconsistent and trades ripple
changes against shock strength.

At call 60 the rollout/teacher state-error ratio is `8.95`, but the analogous
smooth-high-pass ratio is only `1.095`. The spectral response stays smoother
than both local branches, and rollout hidden-band magnitudes do not exhibit a
unique branch explosion. The selector returns `composite_or_unresolved` and
`stop_line3_architecture_tinkering`. This rejects a gain sweep and a single-
branch or pure-Gibbs repair for this checkpoint; it does not prove causal
independence or reject every local/global architecture.

#### D053 exact rollout-error source decomposition (authorized 2026-07-22)

D053 is one zero-training diagnostic on D052's six saved validation rollouts.
It evaluates the frozen legal map `G` on each reference current and decomposes
every raw next-step error exactly as

`e_(t+1) = G(uhat_t) - u_(t+1) = [G(uhat_t)-G(u_t)] + [G(u_t)-u_(t+1)]`.

The first term is propagated input error and the second is fresh teacher-
forced model defect. Record physical-volume/component-scaled norms, cosine and
cross energy, shock/smooth support, and the same quantities after the linear
graph-high-pass operator. All 360 case/call rows, raw admissibility, a call-1
zero-propagation control, and maximum full/high-pass relative identity residual
`1e-6` are mandatory.

At both calls 30 and 60, propagation dominance requires
`||p||/(||p||+||d||) >= 0.65` for both full and smooth-high-pass fields on at
least five of six cases. Fresh-defect dominance replaces `>= 0.65` with
`<= 0.35`; everything else is mixed. Propagation dominance alone may route one
matched short generated-state-exposure capacity test. Fresh-defect dominance
routes target/representation diagnosis without recurrent training. A split or
mixed result authorizes no learned row. Do not rerun D052, tune thresholds,
open test, or begin a basis/gain/smoothing sweep.


Attempt A is fail-closed before interpretation. All 360 rows and admissibility
checks complete and the additive identity closes to `4.91408e-15`, but the
call-1 zero control fails because a saved prior-process rollout output was
subtracted from a fresh same-input GPU replay. The absolute discrepancy is only
about `1.65e-7--1.84e-7`, yet its high-pass propagated share exceeds `1e-6`.
This is a replay-binding defect, not evidence of physical propagation.

Authorize exactly one corrected attempt B. Re-evaluate both map outputs in one
process, reuse the same output for bit-identical call-1 currents, and require
maximum absolute agreement with the saved D052 proposal at `1e-5`. No source
state, cohort, mask, decomposition, dominance threshold, or route changes.
Attempt A remains preserved and no additional retry is authorized.


Corrected attempt B passes every gate on 360/360 rows. Maximum identity and
source-replay errors are `4.93441e-15` and `4.76837e-7`; call-1 propagation is
zero, all masks are available without fallback, and every raw output is
admissible. Median full propagated shares at calls 30/60 are
`0.88995/0.89043`, but smooth-high-pass shares are only `0.15619/0.30733`.
Full propagation gain is near neutral (`0.9878/1.0026`), whereas smooth-high-
pass gain is contractive (`0.1889/0.4056`). Full error energy is about 91%
propagated at both calls; smooth-high-pass energy is `94.75%/84.72%` fresh
teacher defect.

The classification is `mixed_or_split` and the frozen route is
`no_learned_method`. Recurrent exposure is not a ripple-specific answer, while
a target-only repair does not address the propagated full state error. Together
with D052 and the failed conservative local-correction oracle, this closes the
current gain, basis, smoothing, local-corrector, and generated-exposure routes.
Do not open strength OOD or spend seed-confirmation/serious-method compute until
a new target or representation hypothesis is explicitly authorized with a
zero-training falsifier. D044 remains the flagship residual baseline and all
Line-4 flags remain unchanged.

#### D054 current-state shock-conditioned target falsifier (authorized 2026-07-22)

The next hypothesis changes the target decomposition rather than attenuating
an existing PCNO branch. Retain a global conservative-residual path and expose
one local detail tied to a nonlinear current-state troubled-cell sensor, in the
spirit of WENO's local stencil adaptation. Before any training, D054 must test
the necessary locality conditions on D053's fresh teacher defect. This is
scientifically distinct from the rejected fixed local span in D042 and the
rejected inference correction in D044.

D054 is fixed to the six D052/D053 validation trajectories, calls `1--60`, the
D044 component scales, validated cell-volume weights, the linear graph high-
pass, and D053's two-hop reference-current/target smooth mask. Its legal sensor
is computed only from the current input state: top-decile interior pressure-
jump nodes followed by exactly four graph hops. A separate truth oracle selects
the best 20% of smooth nodes only to bound spatial compressibility; it is never
an autonomous sensor.

At calls 30 and 60, the truth oracle must capture at least 70% of smooth high-
pass fresh-defect energy on at least five of six cases. Independently, the
causal halo must capture at least 50% of that energy while covering no more
than 25% of interior nodes, again on at least five of six cases at both calls.
The artifact must contain exactly 360 finite rows, retain every required mask,
keep all raw teacher outputs admissible, and replay both D053 fresh norms within
`2e-4` relative error.

A joint pass authorizes only one later matched tiny-fit contract for a zero-
initialized shock-conditioned detail target; it does not authorize the run.
Truth-oracle locality without causal-halo locality rejects the simple shock
gate. Failure of the truth oracle rejects a bounded local target. Do not sweep
support fraction, halo radius, pressure-jump quantile, local basis, or sensor;
do not open strength OOD, claim conservation/flux prediction, change D044, or
change any Line-4 coordination flag.

#### D054 closeout and D055 proposal self-sensor (authorized 2026-07-22)

D054 closes with all 360 contract rows valid. The best 20% smooth-node oracle
captures median `98.741%/92.536%` of fresh high-pass energy at calls 30/60 on
all six cases, but the four-hop current-shock halo captures only
`18.751%/20.290%` and passes zero cases. Roughly `55.838%/49.105%` of the late
energy lies beyond eight hops; neighbor cosine is negative and x-momentum is
the dominant scaled component. This supports sparse oscillatory pockets, not a
diffuse defect or a narrow shock-band mechanism. The fixed shock gate is
rejected and its radius/quantile must not be tuned post hoc.

D055 tests exactly one remaining causal locator. For reference current `u_t`
and frozen legal proposal `G(u_t)`, compute the graph-high-pass amplitude of
`(G(u_t)-u_t)/scale`. Remove the two-hop halo of the current input's top-decile
pressure-jump nodes, then select the top 20% of the remaining eligible interior
nodes. This support is fixed before comparison with the target. Use only the
six validation trajectories and calls `1/10/30/60`.

Every row must remain raw-admissible, selected support must be at most 20% of
the interior, and D053 fresh norms plus D054 oracle capture must replay within
`2e-4` relative. At calls 30 and 60, at least five of six cases must capture at
least 50% of fresh smooth-high-pass energy. A pass can authorize only the
design of one matched tiny-fit target/detail contract. Failure rejects the
sensor-gated local-detail route. Do not vary the score, exclusion hops, support
fraction, or thresholds; do not train, open strength OOD, claim physical
conservation, modify D044, or alter Line-4 flags under D055.

#### D055 closeout and D056 realizability gate (authorized 2026-07-22)

D055 passes every preregistered gate on all 24 rows. At calls 30/60, the legal
proposal score captures median `91.224%/80.212%` of fresh smooth-high-pass
energy with `17.304%/16.866%` interior support; the six-case minima are
`90.494%/75.534%`. It recovers `92.347%/87.245%` of the truth-oracle energy,
and its median Spearman correlation with fresh amplitude is
`0.87482/0.73318`. No target, rollout state, or future boundary enters support
selection. The score is therefore a credible legal troubled-proposal sensor,
but the result does not establish a usable correction direction.

Before any learned detail head, D056 must test realizability under the original
local-correction safeguards. Reconstruct D055's support exactly on the same six
validation trajectories and calls `1/10/30/60`. Truth may specify the desired
state correction only after support is fixed. Subtract its validated-volume-
weighted support mean independently for each conservative component, cap the
result at 10% of the frozen global update norm, and use a fixed one-dimensional
line search to retain raw admissibility and minimize teacher state error. This
zero-integral projection preserves the frozen prediction's global totals; it
is not an oriented face flux and supports no physical local-conservation claim.

Every corrected row must use at most 20% interior support, at most 10% update
norm, and have volume-integral correction no larger than `1e-10`. Density,
pressure, and internal energy must remain raw-admissible, and front position,
shock strength/thickness, vortex-core error, and smooth high-pass energy may
worsen by at most 5%. At calls 30 and 60, median state-error reduction must be
at least 15%, median smooth-high-pass reduction at least 20%, and at least five
of six cases must be jointly nonworse. Passing authorizes implementation of one
zero-initialized tiny local-detail head with the D044 global model frozen;
failure stops the route. D056 itself trains nothing and may not access test OOD,
tune caps/line search, claim flux conservation, change D044, or alter Line-4
flags.

#### D056 closeout: sparse detail correction rejected (2026-07-22)

D056 completes all 24 fixed validation rows in
`pcno_shock_vortex_proposal_correction_oracle_d056_20260722a`. Summary and row
digests begin `90f5256a` and `016e1ec5`. Every source-replay, support,
zero-volume-integral, raw-admissibility, and anti-smearing check passes; the
largest balance residual is `3.61508e-17` and support remains
`16.767--17.474%` of interior nodes.

Promotion fails at both late calls. Median state-error reduction is
`7.185%/7.010%` at calls 30/60, below `15%`. Smooth-high-pass reduction is
`23.763%/-3.592%`; the required joint-nonworse count is six at call 30 but only
two at call 60. The norm cap does not explain the failure. Before capping, the
balanced truth direction uses only median `2.361%/5.783%` of the frozen global
update norm; all call-30 rows accept it completely. At call 60 the median
applied ratio falls to `2.750%` as the fixed front/high-pass guards become
active.

**Decision:** D055's proposal score is a valid troubled-region locator, but a
sparse, balanced post-hoc state correction on that support has insufficient
state-error headroom and can create late support-boundary high-pass cost.
Reject the zero-initialized learned detail head. Do not reinterpret this as a
request for more support, a larger cap, weaker anti-smearing, or a sensor
sweep. D044 remains the flagship baseline. No new Line-3 learned run is
authorized by D056; an eligible successor must change the global map and
jointly confront D053's propagated shock-supported state error and fresh
smooth-region defect under a new preregistered falsifier. Test OOD remains
sealed and all Line-4 coordination flags remain unchanged.

#### D057 joint-objective gradient falsifier (authorized 2026-07-22)

The next hypothesis changes the learned global map rather than relaxing the
rejected local correction. D053 requires a coupled objective: preserve the
clean state map, penalize the teacher-forced smooth-region high-pass defect,
and expose the map to the shock-supported state error carried by one detached
raw generated call. D057 asks only whether those objectives are locally
compatible in D044's parameterization. It performs no optimization and leaves
the checkpoint byte-identical.

The training cohort is D048's immutable four-pair bank
`(sv_e08_y07,8)`, `(sv_e03_y03,7)`, `(sv_e03_y01,44)`, and
`(sv_e05_y07,41)`. For each objective, accumulate one full-model float32
gradient under validated physical cell-volume weights. The smooth objective is
the graph-neighbor high-pass MSE of `G(u_t)-u_(t+1)` on the intersection of
the current/target reference-smooth masks, each using the frozen top-decile
pressure-jump and two-hop exclusion. The generated objective evaluates
`G(stopgrad(G(u_(t-1))))` against `u_(t+1)`. No stochastic noise is sampled in
the audit; the parent D044 checkpoint retains its noise-training provenance.

Normalize the three aggregate training gradients to unit norm and sum them;
there is no alternative weighting or gradient-surgery row. Each training
objective must have directional cosine at least `0.05` with this candidate.
Transfer is evaluated separately on `sv_e00/e06/e11 x y00/y08` at calls 30 and
60. At each call, at least five of six cases must have cosine at least `0.02`
for clean state, smooth high-pass, and generated-state gradients. All losses,
gradients, and layerwise norms must be finite; generated inputs must be raw-
admissible; smooth masks must have positive physical volume; checkpoint,
configuration, manifest, split, pairs, and calls must match exactly; optimizer
steps and state mutations must be zero.

A joint pass authorizes only drafting one short, full-resolution continuation
contract whose scalar weights are fixed from these gradient norms. It does not
authorize the continuation itself, a serious run, a coefficient sweep, test
access, an architecture change, or a conservation claim. Failure rejects this
naive equal-gradient scalarization while leaving more explicit constrained
multiobjective algorithms as untested. D057 cannot alter D044 or any Line-4
coordination flag.

#### D057 closeout and D058 geometry-group preflight (authorized 2026-07-22)

D057 completes with every integrity gate true: 48 backward calls, zero
optimizer creation/steps, unchanged parameter-state digest, no AMP/noise/test
access, raw-admissible generated inputs, and positive validated-volume smooth
masks. The summary and validation-row digests begin `72ed8121` and
`c78187ea`; peak allocation is 1.991 GiB.

The four-pair aggregate is locally compatible. Candidate-direction cosines for
clean state, smooth high-pass, and generated state are
`0.90133/0.41047/0.87180`. Clean and generated gradients have cosine
`0.98776`; high-pass has only `-0.01961/-0.08409` cosine with them. Yet only
4/6 validation cases pass all three transfer gates at call 30 and 2/6 at call
60. The high-pass cosine is positive on every row, so it is not the repeated
failure. At call 60, clean/generated cosines are both negative on all three
held-out `y00` cases; two `y08` cases pass jointly. Thus the exact D057
direction is not geometry-group robust. Reject its continuation and do not
retune scalar weights.

D058 tests the explicitly distinct constrained alternative left open by D057.
It uses all 84 full-resolution training trajectories, with equal presentations
at calls 30 and 60. For each `y_index` in `1--7`, average clean and detached-
generated gradients across all 12 training strengths and both calls; normalize
the two aggregates, sum them, and normalize again to form one geometry-group
state/recurrence task. Average and normalize the smooth-high-pass gradient over
all 84 trajectories and both calls as the eighth task. Validation and test data
cannot enter task construction.

Compute the minimum-norm convex combination of these eight unit tasks on the
simplex using exactly 500 Frank-Wolfe iterations, lexicographic initialization,
and analytic line minimization. The solver gap must be at most `1e-6`, and the
resulting direction must have cosine at least `0.02` with every training task.
On the same six validation trajectories, at least five of six cases must have
cosine at least `0.02` with clean, high-pass, and generated gradients at each
of calls 30 and 60. Finiteness, raw generated admissibility, positive smooth
mass, exact full-resolution split/provenance, unchanged parameters, zero
optimizer steps, and unopened strength OOD are mandatory.

A pass authorizes only a separately specified short full-resolution MGDA
continuation. Failure stops the current joint-objective route. D058 may not
train, change groups/tasks/calls/iterations after seeing results, use AMP or
stochastic noise in the audit, access test, claim conservation, modify D044,
or alter Line-4 coordination flags.

#### D058 closeout (2026-07-22)

D058 completes its exact contract. The canonical artifact
`pcno_shock_vortex_group_robust_gradients_d058_20260722a` has summary/row
SHA-256 values `45fe9019...` and `57ccc449...`. All structural checks pass:
the complete 84-case train split and 12 validation rows are bound to D044 and
D057, every smooth mask has positive physical volume, all generated inputs are
raw-admissible, the test split is unopened, no optimizer/noise/AMP is used, and
the parameter-state digest is unchanged across 288 backward calls. Runtime is
`153.708` seconds with 2.875 GiB peak CUDA allocation.

The constrained solve itself succeeds. Its Frank-Wolfe gap is `5.55e-17`,
and the eight training-task cosines span `0.563997--0.824737`, far above the
`0.02` gate. Joint validation transfer rises to 6/6 cases at call 30 but only
3/6 at call 60, so promotion fails. At call 60, high-pass cosines span
`0.04286--0.52502` and generated-state cosines
`0.04511--0.26262`; both objectives pass on every row. Clean-state cosines
are negative on all lower boundary-nearest `y00` cases
(`-0.01409-- -0.00916`) and positive on all upper `y08` cases
(`0.03534--0.11023`). The remaining conflict is therefore a repeated
long-horizon geometry-OOD state-fit reversal, not failed Frank-Wolfe
convergence, insufficient training-group coverage, or a high-pass-only
conflict.

Classify D058 as `geometry_group_joint_objective_insufficient` and stop this
joint-objective route without training. Do not retry weights, groups, calls,
iterations, or another constrained-gradient solver after seeing the result.
This local first-order audit does not prove that all multiobjective training,
finite-step retraining, or neural operators fail. It does show that another
D044 continuation is not the next flagship experiment. Any later Line-3
candidate must be a new target/representation or solver-coupling hypothesis
that jointly addresses D053's propagated shock-supported state error and fresh
smooth-region defect, with a predeclared physical headroom test, anti-smearing
gate, raw recurrence, and measured cost. Strength OOD remains sealed and no
Line-4 flag changes.

#### D059 direct-stride-2 tiny-fit gate (authorized 2026-07-22)

The next hypothesis changes the temporal target, not D044's architecture or
loss. D053 attributes `84.7--94.8%` of late smooth-high-pass error energy to
the newly generated one-call defect, so reducing the number of learned calls
is a direct causal lever. Line 1 independently shows that a harder direct map
can outperform repeated shorter maps at the same physical endpoint. Neither
fact proves transfer to 2D PCNO; D059 first asks only whether the direct
stride-2 target is fit-compatible at full resolution.

Freeze the 250x100 family, 84/24/27 grouped split, seed/split seed
`20260718`, conservative residual coordinates, validated physical weights,
model-all-node boundary, and 19,155,720-parameter D044 architecture. The only
model/data change is `step_stride=2`. Use AdamW `1e-3`, weight decay
`1e-5`, unit clipping, constant schedule, BF16, batch one, no input noise,
and no generated-state exposure. Train at most 50 epochs with four
presentations each and early stop. The deterministic pair bank is
`(sv_e08_y07,8)`, `(sv_e03_y03,7)`, `(sv_e03_y01,44)`, and
`(sv_e05_y07,40)`.

Pass only if the best four-pair relative L2 and loss ratio are each at most
`0.01`, every decoded state is finite and admissible, and the fixed single
position-OOD validation smoke completes 30 raw calls to `t=0.6`. Record
source/configuration/data/normalization/split digests, the pair bank, valid
length, failure cause, and every intervention flag. Future-reference
boundaries, clipping, floors, limiters, smoothing, projection, initialization,
downsampling, and test-state access are forbidden. The budget is one run and
0.05 GPU-hour.

A pass authorizes only drafting a serious stride-2 comparison with D044 under
matched physical horizon, sample presentations, validation selection,
parameter count, shock/smooth hierarchy, physical totals, and cost. It does
not authorize that run. Failure stops this direct-stride branch without a warm
start, relaxed gate, longer tiny fit, stride-4 fallback, or attached method
change. D059 cannot alter Line-4 coordination flags.

#### D059 closeout and D060 serious-run draft (2026-07-22)

D059 passes its full-resolution gate at epoch 35. The canonical artifact
`pcno_shock_vortex_stride2_tinyfit_d059_20260722a` has summary/split hashes
`40a15670...` and `b2d2fc78...`, records all three source-code hashes, and
matches the frozen 84/24/27 split and four-pair bank. It consumes
`69.4304` seconds (0.0193 GPU-hour), below budget.

The tiny-bank relative L2 improves from `0.0124691` to `0.00115519`; loss
improves from `0.0409940` to `0.000400830`, ratio `0.00977779`. All
decoded states are raw-admissible. The selected position-OOD smoke completes
30/30 calls to `t=0.6`, with final relative L2 `0.0524226`, minimum density
`0.854982`, and minimum pressure `0.780739`. Every inference intervention
is false and strength OOD remains unopened.

This is the intended fit classification. Stride-2 residual scales are
`1.95--1.98x` stride 1, and initial/best tiny-fit errors are
`2.083/1.994x`; the relative loss reduction is slightly better than the
stride-1 tiny fit. Thus the doubled physical map is harder but shows no new
four-pair optimization obstruction. Do not compare the tiny rollout errors:
the stride-1 and stride-2 smoke selectors chose different validation cases.
No ripple, rollout-accuracy, conservation, or OOD gain is established.

D060 completed its one authorized cold training run on 2026-07-22. It changes
only D044 `step_stride=1` to `2` and H60 rollout calls from 60 to 30. All other
training fields are exact D044 metadata: seed/split seed `20260718`, 50 epochs,
1,024 training presentations, 256 validation presentations, batch four,
rollout selection over all 24 validation trajectories every five epochs, five
width-128 layers, `k_max=8`, `fc_dim=128`, AdamW `1e-3`, weight decay `1e-5`,
unit clipping, constant schedule, BF16, primitive input noise `0.003`, zero
generated exposure, and cold initialization. The retained run uses 0.347
GPU-hour by recorded wall time and remains within the one-seed/2.5-GPU-hour
budget.

Promotion is conjunctive: 24/24 raw finite/admissible H60 completion; mean H60
physical-volume state error at most `0.9` times D044; six-case D013 H60
smooth-high-pass RMS at most `0.8` times D044; shock position, strength,
thickness, vortex-core error, and physical-total mismatch no worse than
`1.05` times D044; direct frame-2 error at most `1.15` times two composed
D044 calls; and same-host H60 wall time at most `0.65` times D044. Preserve
mixed-prefix, completed-case, common-endpoint, raw failure, and boundary-
exchange accounting. Test access, extra seeds, warm starts, stride 4, loss or
architecture changes, and Line-4 flag changes remain forbidden. The completed
artifact is `pcno_shock_vortex_stride2_serious_d060_s20260718_20260722a`; its
50-record log selects epoch 34 with checkpoint SHA-256 `95e6c180a329...`.
Summary, metrics, configuration, and data-manifest digests begin
`b92d8bef4dab...`, `231027ecb52b...`, `0064571b5414...`, and
`f8d228ae3c6e...`.

The frozen physical comparison is complete. D060 is 24/24 complete and its
mean H60 physical-volume state error is `0.00729369`, a `0.87434x` D044 ratio;
it improves every paired case. Direct frame 2 is `1.03246x` two composed D044
calls. Mean shock-strength, shock-thickness, vortex-core, and physical-total-
mismatch ratios are `0.78994/0.86185/0.34240/0.55000`, all inside their
registered envelopes.

The conjunctive claim fails on the targeted mechanism. Six-case H60 D013
median rollout smooth-high-pass RMS is `0.00548836` versus `0.00535199`
(`1.02548x` rather than `<=0.8x`). Across all 24 cases, endpoint smooth-region
graph-high-pass energy is worse on every case; the ratio of mean energies is
`1.36252`, or `1.16727` after the declared RMS readout. Front-centroid distance,
the direct stored position field, is `0.0260282` versus `0.0181368`
(`1.43511x`), although symmetric chamfer is `0.99522x`. The centroid failure
separates cleanly by geometry: `2.74982x` for lower-position `y00` and
`0.74853x` for `y08`. Thus an aggregate alone would hide an opposing conditional
effect. D060 physical and D013 summary SHA-256 values begin
`50ad14177f1e...` and `2f2b5d8c6414...`; the paired D044 summaries begin
`ded10f92fb6e...` and `cf8815020303...`.

The learning and time curves identify what changed. D060 is `1.032x` D044
composition at `t=0.02`, first becomes better near `t=0.08`, reaches a minimum
state ratio `0.708` near `t=0.26`, and ends at `0.874`. On the six-case D013
cohort at H60, its teacher-forced relative L2 is `1.432x` D044 while its raw-
rollout relative L2 is `0.854x`; teacher-forced and rollout high-pass RMS remain
`1.033x` and `1.025x`. The larger macro-step therefore trades a harder local
map for fewer recurrent compositions. It reduces accumulated state error but
does not suppress the measured high-frequency source. Epoch 44's better one-
step validation and worse rollout further show that local optimization cannot
select the recurrent model alone.

The synchronized median-call measurements imply an H60 forward-time ratio of
`0.480`, below `0.65`, while raw measured trajectory time gives `0.103`.
However, D044's batch-1 p95/median ratio is anomalously large, so retain timing
as descriptive rather than making a strict cost claim. A timing rerun cannot
rescue the already failed physical conjunction. D060 is closed without
promotion; do not retry, add seeds or strides, open strength OOD, or attach a
second method.

### Line 4: Discontinuity-Aware Latent Forecasting And Assimilation

#### Core question

Can an encoder expose coordinates in which discontinuous Eulerian dynamics are
easier to predict recurrently without erasing the discontinuity, and can those
same coordinates later support a statistically meaningful analysis update?
This is not the weaker question of whether an autoencoder can reconstruct held-
out snapshots. The representation itself is the causal object: it must preserve
the information needed for the future, condition the decoder well enough to
place fronts accurately, and remain useful under raw recurrence.

#### Representation hypothesis and limits

A moving discontinuity has an algebraic Fourier tail in fixed Eulerian
coordinates, whereas its position and wave strength may evolve smoothly in
phase-amplitude coordinates. This motivates factoring front geometry from a
smoother background field. It does not make the physical problem smooth: a
small phase error can still produce a large decoded state error, front creation
or interaction can change chart topology, and discarded fine scales can feed
back into the resolved future.

For deterministic latent recurrence, approximate Markov closure is necessary:

E(U) = E(V) implies E(Phi_dt(U)) approximately equals E(Phi_dt(V)).

Equivalently, a large conditional future variance
`E[Var(E(U_(n+1)) | E(U_n))]` is an irreducible error for a deterministic
one-state latent transition. Low reconstruction error does not establish this
condition. A bounded history or stochastic closure is justified only after the
one-state ambiguity is measured.

The leading representation hypothesis is therefore hybrid rather than a tiny
monolithic code. A candidate latent state may contain

`a = (a_front, U_coarse, a_local)`,

where explicit phase or front variables carry shock geometry, a coarse
conservative spatial field carries global budgets and slowly varying content,
and local spatial tokens carry geometry-dependent residual structure. Memory or
stochastic closure is added only if measured latent ambiguity requires it. The
exact candidate remains a Phase-0 decision, not authorization to implement all
of these components.

Line 4 may test structure through an admissible or conservative decoder or
manifold rather than through an ill-conditioned face-flux target. Such structure
is not assumed beneficial: the causal comparison must hold the latent
transition and effective capacity fixed and verify sharp decoded states,
budgets, and raw recurrence.

Latent smoothness must be operationally qualified as spatial, temporal,
parameterwise, spectral, or manifold smoothness. A global latent vector has no
intrinsic spatial Fourier spectrum, and visually smooth latent trajectories are
coordinate-gauge dependent. The relevant evidence is lower forecast ambiguity,
better-conditioned decoded perturbations, and improved raw rollout under
matched cost.

Line 3 owns direct physical-space ripple control, local-basis methods, and the
front-factorized conservative-remapping route if its measured error is
primarily front displacement. Line 4 owns coordinate transformation, latent
propagation, and later latent analysis. It must consume Line 3's shared dynamic-
testbed contract and diagnostics, and it may train on that testbed only after
Line 3 authorizes the reference states as training truth. If Line 3 validates a
front representation, Line 4 may inherit it as a candidate; it must not
independently rebuild the same physical remapping experiment.

#### Current analytic result and required Line-3 handoff (2026-07-20)

A separately authorized zero-training Line-4 preflight has completed on the
frozen 512-case, 256-cell Line-1 dataset. The matched rank-43 POD control fails
the hierarchical front reconstruction screen. The truth-derived oracle chart
does not improve state or encoded conditional-future ambiguity over POD, and a
fixed-size one-step history materially improves encoded-future ambiguity. The
current-state Rankine--Hugoniot/contact-speed augmentation also fails its matched
closure and reconstruction gates.

The full-rank oracle remap fails even on its preregistered favorable cohort of
243 snapshots from 64 validation trajectories. Its grouped-bootstrap state-L2
p95 upper bound is `0.0250` against the `0.010` gate, front-recall and
front-precision lower bounds are `0.6827/0.6474` against `0.95/0.95`, and the
front-thickness symmetric-distortion p95 upper bound is `7.90` against `1.05`.
Admissibility remains one. This rejects the existing 1D chart/remap, not every
front representation. Line 4 therefore stops before a generic autoencoder,
learned front encoder, learned latent transition, or assimilation experiment;
it must not rescue the failed pilot with a representation sweep.

This result does not add a Line-3 method run or change Line 3's independently
selected causal route. When Line 3 freezes its dynamic global baseline, its
handoff must nevertheless make the dependency machine-readable or explicit in
the frozen report:

- record `line4_training_truth_authorized = true/false` for the promoted
  reference states, and bind the reference, perturbation-family, grouped-split,
  geometry, resolution, timestep, and normalization digests;
- bind the selected physical baseline checkpoint/configuration and report raw
  rollout completion, failure causes, front hierarchy, smooth-region error,
  budgets, latency, and every inference intervention;
- if and only if Line 3's own error classification selects and validates the
  front-factorized route, export the candidate's fixed and learned variables,
  front-topology convention, conservative remap, geometry/resolution contract,
  encoder/decoder digests, valid lengths, and explicit intervention flags; and
- otherwise record `line4_front_candidate_available = false` and the causal
  reason. Line 3 is not required to invent a front candidate to unblock Line 4.

This requirement is now frozen in
`line3_to_line4_handoff_s20260718_20260721b.json`, schema
`line3_to_line4_physical_baseline_handoff_v1`, SHA-256 beginning `4b56baef`.
The strengthened finalizer verifies equality of the complete training and
evaluation data contracts and reconstructs the exact source-artifact mapping
from the 135-case family audit. It records
`line4_training_truth_authorized = true` only for this frozen family,
`line4_front_candidate_available = false` because D013 is unresolved and no
front/remap contract exists, and `line4_transition_training_authorized = false`.

Line 4 may reopen reconstruction and closure tests only after the first two
items are frozen and, for a front/hybrid candidate, the third item exists.
Latent transition training still requires those representation gates to pass;
Line 4B assimilation remains separately blocked by raw open-loop promotion.

#### Authorized frozen-2D representation gate (L4A-002, 2026-07-21)

The exact frozen handoff, SHA-256
`4b56baefe0f61fd635668c51e7dd71e82a783450a51131de74bcb7c7875cc9c5`, has
now been read directly and passes the Line-4 input audit. Its training-truth
contract is complete; its physical baseline contains nonempty aggregates, 24
front/smooth hierarchy rows, grouped parameter summaries, cost, reference
geometry and characteristic travel; and every declared inference intervention
is false. The handoff still records no front candidate and keeps latent
transition training false.

L4A-002 may therefore implement and train representations only on the frozen
84 training and 24 position-OOD validation trajectories, using all 61 saved
250x100 states and the inherited training-only normalization and physical cell
volumes. The 27 strength-OOD trajectories remain sealed. The fixed ladder is:

1. identity coordinates as the no-compression control;
2. physical-volume POD at exactly 5,000 retained scalars;
3. one generic 25x10 spatial-token autoencoder with 20 channels per token; and
4. one matched conservative-moment spatial-token candidate on the same lattice.

The 1D oracle-front result remains the required negative mechanism falsifier.
There is no 2D front-coordinate row: L4A-002 contains no front variable, front
fit, front loss, or conservative front remap. The spatial-token candidate uses
four exact volume-weighted conservative averages and 16 learned residual
channels per token. Its decoder subtracts the volume-weighted residual mean in
each physical macrocell before adding the exact average. This is a declared
decoder layer, not a physical-space stabilization or recurrent projection. The
generic control has the same token count, scalar budget, network parameters,
optimizer, state-reconstruction loss, seed, and schedule, but all 20 channels
are learned and no conservative projection is applied.

Promotion requires one checkpoint to pass every gate simultaneously:

- 100% raw validation admissibility and no hidden clipping, floor, limiter,
  smoothing, front fit, or decode--reencode;
- mean physical-volume reconstruction error at most `0.0021`, with no greater
  than 5% shock-strength or thickness distortion, no increased overshoot, and
  each front/smooth metric consuming at most 25% of the corresponding frozen
  D044 H60 error budget;
- encoded conditional-future ambiguity at most `0.8` times matched POD, while
  adding exactly one previous code improves the ambiguity by less than 20%;
- empirical decoder perturbation gain at doubled query resolution at most
  `1.25` times its 250x100 value, with nearby same-regime interpolations
  decoding to one admissible sharp state; and
- complete parameter count, retained-state size, encoder/decode latency,
  throughput, physical totals, intervention flags, and failure causes.

The conservative-moment candidate is promoted over the generic control only if
both pass and it improves closure or conditioning by at least 20% without a
front, smooth-region, or admissibility regression. If only the generic row
passes, it may continue without a conservation claim. Front blurring, a
material history benefit, resolution-growing decoder sensitivity, or failure
of both learned rows stops this representation contract; do not rescue it with
a latent-dimension, architecture, or loss sweep.

The implementation budget is two tiny-fit smokes and two one-seed
representation jobs, no more than 24 GPU-hours total. No latent transition,
raw decoded rollout, joint fine-tuning, strength-OOD test access, or filtering
is authorized. A later decision must explicitly set
`line4_transition_training_authorized = true` after these representation gates
pass.

#### L4A-002 engineering-smoke result and stop (2026-07-21)

The first actual matched smoke is complete. A strengthened preflight verified
the frozen handoff and normalization, hashed all 78 arrays in the staged
four-train/two-position-OOD-validation cohort, loaded frames 0 and 60, built
the declared 250-token geometry on the real `[N,1]` physical measures, and
confirmed that no strength-OOD test array was staged or read. A prior launch
found the missing `[N,1]` integration shape and stopped before model
construction and zero optimizer steps; it is an implementation preflight
failure, not a training result. The completed run uses one seed, 800 matched
updates, frames 0/15/30/45/60, and 10 paired validation rows. Its summary
SHA-256 begins `e742ccf4`.

The conservative-moment row is causally better than the parameter-matched
generic row on several bounded properties: it lowers mean physical relative
L2 from `0.007605` to `0.005430` (`28.6%`, lower on 10/10 paired rows), lowers
smooth-region scaled error by `36.6%`, improves mean front IoU from `0.283` to
`0.390`, and reduces global-budget error by `99.78%`. This supports only the
claim that exact macrocell moments and a zero-mean residual decoder improve
matched reconstruction and budget fidelity under this smoke contract.

The joint representation capability fails. The structured decoder retains
only `0.539` mean shock strength and produces `2.672x` mean thickness; the
generic row retains `0.463` and produces `3.540x`. Both exceed the `0.0021`
reconstruction ceiling and miss the 5% strength/thickness gates by large
margins. From update 400 through 800, L2 continues falling while decoded shock
strength generally falls and thickness remains near `2.65--2.80x`. Therefore
the lower L2 is not evidence of a sharp manifold, and the `38.6%` lower
smooth-region high-pass energy cannot be counted as ripple reduction.

The remaining diagnostics do not reverse that decision. Raw admissibility is
100% for these 10 rows, but admissibility does not establish front fidelity.
The structured code's conditional-future ambiguity proxy is only `5.74%`
below generic, with no rank-5000 POD denominator; one previous code changes
the ambiguity by less than 1% for either row. Doubled-query-resolution decoder
gain is stable (`1.002` structured, `1.000` generic) and query consistency is
below `8e-5`, which supports only local decoder-conditioning evidence without
fine-resolution truth. There is one seed and only two validation trajectories,
so even the consistent paired advantage is engineering evidence, not a
population or uncertainty-calibration claim.

Stop this smooth spatial-token decoder/state-L2 contract. Do not spend the
remaining smoke on longer identical optimization, run rank-5000 POD or a full
representation job to rescue it, change latent size, add a front loss, train a
transition, open the test split, or begin filtering. The only admissible next
proposal is one separately authorized frozen-decoder code-reachability oracle
on the same 10 rows: hold the checkpoint and exact four moment channels fixed,
fit only each state's remaining code, and reapply the same L2, strength,
thickness, overshoot, and admissibility gates. Its matched control is the
amortized encoder code; pass would localize the defect to amortized encoding,
while failure would reject the decoder manifold. Cap it at 10 states, one
fixed optimizer contract, less than 0.1 GPU-hour, and no test access. Because
it uses per-state fitting, it is mechanism evidence only and can never support
an autonomous-forecast or assimilation claim. This oracle is proposed, not
authorized by the completed smoke.

#### L4A-003 frozen-decoder code-reachability result and family stop (2026-07-22)

After explicit authorization, the bounded oracle reused the exact L4A-002
checkpoint, SHA-256
`58a04862dc3d6b630a32b8b792e14d9280bfacde810d47fe54e41b7fbf86ba69`,
and the same 10 position-OOD validation states. It held the four conservative
moment channels and every decoder weight fixed, optimized only the 16 free
channels for each state with one preregistered L-BFGS contract, and never read
the strength-OOD test split. The corrected preflight rehashed all 78 staged
arrays and reverified the handoff, manifest, normalization, selected keys, and
800-update checkpoint provenance. Total cost was `0.00308` GPU-hour.

The first integration attempt invalidated a non-owning edge-array view when the
bounded shard cache evicted its source. It stopped before emitting any
reconstruction row or summary and is not model evidence. The corrected
implementation owns that array, has focused CPU coverage, and produced the
complete artifact `line4a_code_reachability_20260722b`; its summary SHA-256 is
`e9577b437824069ff0db9dc28b9072368754eac2ba14453ef4039a41083a798a`.

Per-state fitting is a stronger capacity oracle than amortized encoding but is
not a deployable representation. It reduces mean relative L2 from `0.005430`
to `0.003226` (`40.6%`) and mean thickness ratio from `2.672x` to `2.109x`,
while all 10 raw decoded states remain admissible. Those gains still fail the
joint physical gate: nine of 10 states exceed the `0.0021` L2 ceiling, mean
shock strength falls from `0.539` to `0.490` of truth, mean thickness remains
`110.9%` too broad, and maximum overshoot increases by `16.2%`. The `78.1%`
lower smooth-region high-pass energy is not ripple reduction because the same
states retain materially blurred and weakened fronts.

The fitted codes are also not nearby data-manifold corrections. Their
training-scale-normalized free-code RMS displacements span `1.528--4.419`
(mean `2.550`), so all 10 fail the at-most-one-scale neighborhood gate. The
registered classification is therefore `decoder_manifold_rejected`, with
`physical_reachability_pass=false`, `nearby_code_pass=false`, and
`encoder_remediation_eligible=false`.

Stop this smooth point-decoder/state-L2 representation family. Do not run its
serious representation job, rank-5000 POD as a rescue, encoder distillation,
transition training, test evaluation, or data assimilation. This result does
not prove that no latent representation can preserve discontinuities or that
no code exists under any optimizer; it rejects the declared frozen decoder
manifold under the bounded local reachability contract. The exact conservative
moment channels remain a useful structural component. Any future Line-4 row
must introduce a separately authorized representation hypothesis that directly
changes sharp-front capacity, with a deterministic capacity preflight before
training; it is not an automatic retry or sweep.

#### L4A-004 conservative local-Haar capacity preflight (registered 2026-07-22)

L4A-004 tests one causal redesign only: whether the failed decoder's smooth
point basis, rather than the 5,000-scalar spatial budget itself, prevents sharp
reconstruction. It keeps the frozen 25x10 physical token lattice and 20 scalars
per token. Four coordinates remain exact normalized conservative token means;
the other 16 are coefficients of a fixed local discontinuous chart. There is
no front variable, front fit, learned transition, rollout, or analysis step.

The chart starts from tensor-product Haar step functions through level three in
normalized token coordinates. On the frozen geometry, each token dictionary is
volume-centered and deterministically orthonormalized. Using only the same four
spread training trajectories and frames 0/15/30/45/60, each spatial mode's
four-component coefficient matrix receives one deterministic SVD. The 16
mode/direction atoms with greatest component-scaled training energy are frozen
per token with deterministic tie and sign conventions. Validation encoding is
then one weighted linear projection, not per-state optimization. The selected
atoms, directions, geometry, normalization, training keys, and frame set must
be hashed before any validation metric is read.

The strongest matched capacity control is L4A-003's per-state fitted smooth
decoder on the identical 10 validation states; the amortized conservative code
remains the deployment-style control. L4A-004 has the same 5,000-scalar state
and exact token-moment contract, but it changes the decoder basis and therefore
does not isolate an encoder or loss effect. It makes no geometry, resolution,
closure, forecast, or neural-operator claim.

One CPU-only run is authorized. The strengthened preflight must rehash all 78
staged arrays, bind the completed L4A-003 artifact and exact validation targets,
and confirm that the strength-OOD test split is absent and unread. Promotion
requires every gate simultaneously: 10/10 raw admissibility; mean physical-
volume relative L2 at most `0.0021`; mean shock-strength and thickness ratios
within 5% of one; maximum density/pressure overshoot no greater than the frozen
amortized conservative control's `0.0708514`; maximum token-moment and global-
budget relative L2 at most `1e-5`; and exactly 16 finite independent detail
coordinates in every token. Lower graph high-pass energy is not a pass when a
front gate fails.

Passing all gates records only `capacity_only_pass` and authorizes one separate
zero-training closure/conditioning proposal; it does not authorize serious
representation or transition training. Failing any gate records
`local_haar_capacity_rejected` and stops this exact chart without changing Haar
levels, atom count, token lattice, selection objective, or data cohort. Test
access and Line 4B remain blocked in either case.

#### L4A-004 local-Haar result and stop (completed 2026-07-22)

The sole registered run completed on CPU in `9.92` seconds (`0.00276` CPU-hour).
Its preflight rehashed all 78 staged arrays, verified bitwise equality with the
10 L4A-003 validation targets, used exactly 20 declared training states for
dictionary selection, and read no strength-OOD test array. The complete result
is `line4a_local_haar_capacity_20260722a`; summary SHA-256 is
`f91b78d5605c9d02041e81a9b523d8a21959b7a86effa453b94604c4d5927123`.

The discontinuous chart improves the amortized smooth encoder on several
matched metrics. Mean relative L2 falls from `0.005430` to `0.004461` (`17.9%`,
better on 10/10 states), mean shock strength rises from `0.539` to `0.830` of
truth, mean thickness falls from `2.672x` to `1.554x`, and mean front IoU rises
from `0.390` to `0.500`. Shock strength is closer on 9/10 states and thickness
is closer on 9/10. All decoded states are raw admissible, maximum overshoot is
`0.06840`, and maximum token/global moment errors are `1.04e-6/1.53e-6`.
This supports the narrow mechanism inference that discontinuous local decoder
regularity affects front fidelity at fixed state size.

The registered capability nevertheless fails. Mean L2 is `2.12x` the `0.0021`
ceiling and only one of 10 states reaches that ceiling. Mean strength remains
`17.0%` weak and mean thickness remains `55.4%` broad, both far outside the 5%
gates. The Haar row is worse in L2 than L4A-003's privileged per-state fitted
smooth decoder on 10/10 states, although that control itself fails by blurring
and weakening fronts. Higher Haar graph-high-pass energy is reported but is not
interpreted as either ripple or success without a passing front hierarchy.

The preregistered classification is `local_haar_capacity_rejected`. Stop this
fixed level-three, 16-detail-per-token chart without changing levels, atom
count, lattice, selection rule, or cohort. Do not run a closure test, serious
representation training, transition, test evaluation, or assimilation. The
result does not reject adaptive sparse multiresolution or every discontinuous
representation, but those would be new representation hypotheses with new
support/topology and transition contracts, not rescues of L4A-004. With no
active candidate passing reconstruction, Line 4 returns to a stopped state.

#### Stage 4A: bounded latent open-loop screen

Use 1D only as a controlled representation and closure diagnostic. The minimal
matched ladder is:

1. identity coordinates with the frozen physical residual baseline;
2. POD at matched retained state size;
3. one generic autoencoder;
4. oracle front or phase coordinates; and
5. at most one selected learned or hybrid phase-field representation.

The oracle front coordinate is a mandatory mechanism falsifier. If removing
front translation with truth-derived coordinates does not materially improve
the forecast problem, do not escalate to a complex learned front encoder. Do
not open a broad autoencoder, latent-dimension, dynamics-model, or loss sweep.

A Line-4A method must pass all of the following gates before it is considered a
credible latent forecast:

- **Hierarchical reconstruction:** report admissibility, front position,
  strength, thickness, overshoot, contact behavior, smooth-region error, and
  global budgets. Lower aggregate L2 caused by broader or weaker fronts fails.
- **Closure:** measure future ambiguity among nearby or identical codes and run
  one bounded history test if ambiguity is material.
- **Decoder conditioning:** measure physical perturbation amplification through
  the decoder, including front-position sensitivity and scaling with
  resolution. Compression that becomes increasingly singular under refinement
  does not establish transfer.
- **Raw recurrence:** advance `a_(n+1)=G(a_n, ...)` without hidden physical
  correction. Any decode-reencode projection, clipping, floor, limiter, or
  periodic truth reset is an intervention and must be declared and separately
  ablated.
- **Timestep behavior:** compare direct and composed steps, truth errors, and
  semigroup defect over physical horizons; a continuous-time latent model is
  not automatically timestep robust.
- **Transfer:** test unseen front location, strength, interaction, resolution,
  and geometry under a declared representation contract.
- **Matched utility:** compare rollout accuracy, completion, physical metrics,
  batch-1 latency, throughput, encoding cost, every required decode, and state
  size against the frozen physical residual FNO or PCNO baseline.

After the bounded 1D falsifiers, the primary evidence must come from Line 3's
promoted dynamic 2D shock benchmark. The strengthened Line-3 handoff now
authorizes its frozen reference states as training truth only for the audited
135-case family and exact 250x100/time contract; this does not establish
resolution or geometry transfer. The CPG bump may support a representation
diagnostic, but its current geometry contract cannot support physical
conservation claims.
Stable latent trajectories alone are not a pass; the decoded rollout must
satisfy the shared physical hierarchy.

#### Data-assimilation readiness gate

Line 4B remains blocked until Line 4A and the medium-horizon promotion gate
pass. Before filtering, the representation must preserve dynamically relevant
perturbations on a declared data-manifold, dynamically relevant, or observation-
relevant subspace. A useful local target on that subspace is

`J_D J_G J_E delta U approximately equals J_(Phi_dt) delta U`,

or an empirical ensemble analogue. Report the discarded and encoder-nullspace
directions rather than requiring equality for every physical perturbation.
Also report decoded covariance and perturbation-growth fidelity, latent
observability under changing sparse sensors, forecasts from analysis-like off-
manifold states, innovation behavior, sensitivity to equivalent latent
reparameterizations, and a localization contract when the latent state is
spatial. Nearby same-regime or ensemble-relevant code interpolations should
decode to one plausible intermediate front rather than two fronts or a diffuse
mixture; arbitrary interpolation across different front topologies is not a
forecast requirement.

#### Stage 4B: conditional latent data assimilation

Freeze the accepted forecast and begin with the physical observation model

`y = H(D(a)) + eta`,

using a standard EnKF or LETKF. A learned observation encoder, learned
covariance, learned analysis operator, score model, or particle method is a
later extension, not the first test. A compact latent code does not guarantee
Gaussian forecast errors or posterior uncertainty; shock-location or topology
uncertainty can be strongly non-Gaussian and multimodal.

Run two comparisons and do not merge their attribution. First, freeze one
accepted forecast operator and compare physical-space, oracle-front-coordinate,
generic-autoencoder, and selected-hybrid analysis coordinates under the same
observations, ensemble budget, initialization, and cadence. This isolates the
analysis coordinates. Second, compare complete forecast-analysis systems; this
changes both forecast and filter and must be labeled end to end. Include a
credible numerical forecast and a weak persistence forecast under the same
filter as controls. Report observation-interval curves, sensor density and
layout, noise level, an observation-blackout continuation, spread-error and
innovation calibration, shock-position uncertainty, admissibility,
conservation, ripple energy, and complete online cost.

Assimilation that succeeds only at very frequent cadence establishes forecast
dependence, not a successful latent macro-solver. Analysis updates must not hide
raw forecast failures, and assimilation performance must be reported separately
from open-loop solver evidence.

#### Phase-0 status and kill criteria

Only the read-only Line-4 proposal is authorized by this decision update. A
later Phase 1 must separately authorize latent-forecast implementation and
Line-4B assimilation. Kill or redesign the candidate if:

- reconstruction gains come from shock blurring or lost wave strength;
- oracle phase coordinates do not improve predictability;
- closure defect remains high at the intended compression;
- decoder sensitivity worsens materially with resolution;
- stable rollout depends on repeated re-encoding or an undeclared correction;
- the latent method cannot compete with the identity-space residual baseline;
  or
- assimilation gains disappear under observation blackout, require nearly
  continuous resets, or come from an oracle forecast or per-cycle latent fit.

### Shared Evaluation And Claim Contract

Every line must:

- keep one-step fit as an entry gate rather than a rollout proxy;
- evaluate raw recurrence without hiding failure behind clipping, primitive
  floors, future-reference boundaries, or a nearly frozen limiter;
- report admissible completion before oscillation, shock position, strength,
  thickness, smooth-region, and global-budget accuracy;
- distinguish mixed-prefix, completed-case, and common-endpoint statistics;
- use grouped geometry/parameter OOD splits where possible;
- select checkpoints by a declared rollout rule; and
- report physical horizon, calls, characteristic travel, effective CFL,
  batch-1 latency, and amortized throughput together.

Line 4 must additionally report the latent-state dimension or spatial-token
contract, hierarchical reconstruction, closure defect, decoder conditioning,
and whether recurrence includes decode-reencode. Any later assimilation claim
must use the same observation operator, ensemble budget, cadence, and forecast
initialization across physical and latent filters, and must include calibration
and observation-blackout evidence.

For 2D artifacts, preserve `predicteds`, `targets`, `pos`, `edges`, and
`node_type` when available, together with current or initial state, physical
time and `Delta t`, trajectory/sample mapping, parameters or Mach number,
boundary mode, valid length and failure cause, coordinate convention,
checkpoint/configuration digest, and the weights used by each diagnostic. Add
a validated mesh-to-graph map, volumes, face measures, normals, and oriented
face connectivity before physical conservation conclusions; add reference
cumulative impulses before reference-flux conclusions.

### Codex Agent Workflow

The approved Phase-0 kickoff prompts are:

- `prompts/LINE_1_LARGE_STEP_FLOW_MAP_PROMPT.md`;
- `prompts/LINE_2_CPGNET_MECHANISM_PROMPT.md`;
- `prompts/LINE_3_2D_ROLLOUT_PROMPT.md`; and
- `prompts/LINE_4_LATENT_FORECAST_ASSIMILATION_PROMPT.md`.

Phase 0 is read-only. Each agent returns a bounded proposal, evidence audit,
dependencies, run budget, claim table, and kill criteria without editing files
or launching experiments. The coordinating agent reconciles shared metrics,
data contracts, and dependencies before Phase 1. Implementation agents then
use isolated branches or worktrees with explicit ownership, compute limits,
artifact schemas, and stopping rules.

For Line 4, Phase 0 may design both stages, but Phase 1 authorization is split:
latent forecasting comes first, and assimilation implementation remains
unauthorized until the forecast gates pass.

## Completed And Superseded 1D Experimental Path

The numbered path below records how the 1D evidence was obtained. It is not an
active queue. The four-line program above supersedes any unchecked or
forward-looking item in this historical list.

1. Freeze h128/mp28 as the strong corrected CPGNet reference. The matched
   controls support a locality mechanism; do not turn this into a broad graph
   architecture sweep or target-superiority claim.
2. Freeze conservative input, conservative loss, fixed physical scaling, and
   conservative recurrence as the provisional coordinate contract within D021.
   The coordinate matrix is complete; do not reopen it during the target screen.
3. Preserve the validated cumulative-impulse dataset and the completed
   tiny/repeat/optimization/gauge controls. Stop the exact gauge-canonical joint
   face-value MSE row at midscale because it fails the decoded-state and rollout
   gates despite fitting its native target.
4. The named stability intervention is complete. Unroll-only passes the
   predeclared midscale gate; the tested training-only barrier fails attribution.
   Keep inference limiter-free and the conservative flux decoder fixed.
5. Freeze the full-scale flux-form row as a strong 20-call result and a failed
   50-call baseline. The 1/64 completion at call 50 rules out a medium-horizon
   stability claim for this checkpoint.
6. Stop uncentered direct next-state prediction in the strict matched regime
   after its two-seed tiny-fit failure. Residual is the surviving centered
   parameterization; adding an identity skip to direct state would make the two
   parameterizations algebraically equivalent.
7. Freeze the completed three-seed residual family as the strong fixed-setting
   accuracy baseline. Preserve pooled completion and matched common-case
   readouts; do not collapse failed prefixes into a single accuracy claim.
8. Freeze the completed projected-residual family as a structural ablation. Its
   exact learned-budget closure and pooled stability gain are useful, but its
   state, robust-conservation, and shock gates fail; do not promote it as the
   more accurate target or return to full face-value MSE.
9. The teacher-offset 8+4 control is complete. It rules out later physical-time
   sampling as the source of the generated-burn gain on this split and seed.
   D023 is complete on all 16 test cases, starts 0:10:80, and prefix depths
   0/2/4/8. At depth eight, generated/teacher error ratios are `0.992` to the
   original next state and `1.017` to the same-state solver continuation, with
   correction alignment `0.074`. It therefore passes neither the PDE-map nor
   strong trajectory-correction threshold. Treat generated burn-in as a useful
   closed-loop intervention with unresolved modified-dynamics mechanism, not a
   demonstrated PDE-consistent operator improvement. Do not full-scale while
   H20/H50 top-two front position remains worse than clean.
10. D013 is complete for 1D. Teacher-forced flux and residual fits have similar
    high-band content, but recurrent flux failure has 10.4 times the matched
    residual high-band error and 154 times its second-difference error.
    Classify this as recurrent spectral growth, not one-step spectral
    underfitting. A 2D extension must use audited graph-native scale bins.
11. D024 rejects blind conservative Laplacian viscosity on the frozen flux
    model. No tested coefficient improves H50 completion; small coefficients
    worsen state and Nyquist-tail error, and larger coefficients shorten
    survival. Do not repeat broad current-state smoothing sweeps.
12. D025 is complete and fails its successive-halving gate. The faithful global
    face-grid FNO uses relative positive traces, a shared Rusanov or central
    flux, physical boundaries, exact FV update, and raw recurrence. At 64/16/16
    scale, frame-zero weighting lowers selected test one-step relative L2 to
    `0.00791` and initial-call conservative error by 73%, but all 16 H50
    rollouts still fail within four calls. This rules out shock rarity as a
    sufficient explanation. Do not promote this parameterization to full scale
    or four-step training.
13. The D025 training-only pressure/internal-energy control is also complete.
    Barrier weight `0.1` eliminates the measured violation inside two-step
    training windows but leaves test survival below `0.05` with 0/16 complete.
    This does not reject risk-sensitive constraint losses in general, but it
    rejects another mean-barrier sweep for this row. Keep inference limiter-free
    and do not add blanket TV.
14. D026 closes the direct boundary-exchange auxiliary at the matched gate.
    Weights `0.01` and `0.1` improve the intended boundary and conserved-total
    metrics, but both worsen one-step, H20 state, and shock errors while leaving
    completion unchanged at 16/16. This is a structural Pareto ablation, not a
    promoted method. Do not search more coefficients or run full seed repeats.
15. D027 establishes a useful cold stride-2 residual operator at effective CFL
    maxima up to about 13.3. It improves common-case H20/H50/H100 state error by
    25.0%, 36.2%, and 48.6% over stride-1 composition and completes 16/16 at
    H100. Its direct frame-2 error is 1.216 times composition, so classify it as
    large-step capacity with an initial-jump defect, not unconditional timestep
    robustness.
16. D028 weight-only continuation repairs frame 2 and lowers final-target
    training floors by more than 40%, but it is not a uniformly better solver:
    H50 state and shock errors are 1.095 and 1.565 times cold. Stop before the
    conditional total-exposure control and make no continuation-sensitive
    optimization claim. Preserve cold stride 2 as the large-step accuracy
    baseline and warm stride 2 as a fit/stability tradeoff ablation.
17. D029 completes the frozen 128/256/512-cell transfer gate. Exact physical
    case identity and nx256 metric replay pass, but neither checkpoint matches
    the native off-grid one-step map: error rises by 5.5--8.3 times, fine-grid
    shock geometry regresses, and cold stride 2 loses one near-pressure-margin
    case on both new meshes. Do not call the current residual FNO mesh invariant.
    The larger-step advantage is more robust: cold stride 2 still beats
    stride-1 composition at H20/H50/H100 on both off-grid meshes with equal
    same-grid completion.
18. D030 resolves the immediate representation question. Exact restriction and
    update-label commutation pass, and the unchanged 24-mode shared FNO passes
    the equal-presentation same-grid-oracle and frozen-baseline usefulness
    gates. Do not add explicit cell width, timestep, or bandwidth to explain
    D029. The row fails native coarse-solver equivalence and still has three
    consistent pressure terminations by H100; it remains a fixed-step operator
    for one restriction-consistent flow map, not one timestep-conditioned or
    arbitrary-discretization model.
19. Isolate post-fit tail stability on the strong residual baseline. Compare an
    equal-update, equal-generated-exposure continuation control with one
    training-only tail-risk pressure/internal-energy penalty, keep raw inference,
    and predeclare accuracy, shock, conservation, and completion noninferiority.
    Do not bundle TV, viscosity, projection, or an interface decoder into the
    first row.
20. Give survivors an equal-budget best-engineered comparison, then add uniquely
    defined stage/partial-impulse and shock/front auxiliaries one family at a
    time. Derive redundant states through the primary decoder.
21. Evaluate every finalist under the one-step/direct-horizon/100-call raw
    capability protocol, timestep/resolution transfer, and a coarse-CFD
    comparison before transfer to CPG geometry and a dynamic 2D shock benchmark.

The CPG benchmark is a secondary geometry and shock-transfer test. Its current
HDF5 contract lacks validated control-volume geometry and reference face fluxes,
so it cannot be the primary causal target diagnostic without additional
solver-side data.

## Medium-Horizon Promotion Gate

Data-assimilation experiments begin only after a forecast method demonstrates:

- a documented one-step/direct-horizon/autoregressive capability decomposition;
- at least 50 raw autoregressive calls over a meaningful physical horizon where
  the dataset supports that protocol;
- completed rollouts for nearly all held-out trajectories at a fixed horizon;
- rollout-curve and final-error gains over a matched residual baseline;
- consistent gains across fixed data splits and multiple model seeds;
- no nonphysical raw states hidden only by primitive floors;
- infrequent, nondegenerate limiter intervention;
- boundary-aware conservation accounting;
- accurate shock position, strength, thickness, and contact behavior;
- robustness to timestep and resolution changes; and
- successful transfer to both irregular geometry and a credible dynamic 2D
  shock setting.

Thresholds should follow reference-solver scales and baseline variability. A
method that remains finite only because its limiter nearly freezes the update
does not pass.

## Later Data-Assimilation Stage

Line 4B begins only after this gate. Freeze the accepted forecast and compare a
classical physical-space filter with the corresponding latent-space filter under
the same observation operator, sensor layout, noise, ensemble budget,
initialization, and cadence. Start with EnKF or LETKF and evaluate physical
observations through `H(D(a))`; this isolates the representation and learned
forecast from a learned analysis mechanism.

Initial questions include the useful observation interval, required sensor and
ensemble counts, analysis-increment magnitude, uncertainty calibration,
shock-position uncertainty, behavior under an observation blackout, and which
physical scales or unstable directions remain poorly modeled or observed.

Learned observation encoders, covariance, localization, analysis operators,
adaptive observation, generative filters, and solver fallback belong after this
controlled forecast-plus-classical-DA comparison. Data-assimilation accuracy is
not evidence that the underlying macro-solver passed the open-loop forecast
gate.

## Paper-Level Interpretation

The paper story is not fixed to a conservative target. The strongest current
framing is global reach, target conditioning, and localized stabilization for
geometry-aware neural macro-solvers. Line 1 may become a bounded empirical and
theoretical result about learned flow-map operating envelopes. Line 2 may
become a causal failure/reproduction analysis. Line 3 must supply the main 2D
algorithmic result. Line 4 can support a distinct representation result only if
it shows that sharper, approximately closed latent coordinates improve raw
forecast recurrence under matched cost. A later forecast-analysis result also
requires calibrated latent assimilation under sparse observations and blackout,
not merely frequent analysis resets.

Any paper should establish causal comparisons rather than claim that every
component is necessary. It needs medium-horizon 1D and 2D evidence, measured
efficiency, and a mechanistic account of how representation, geometry, and
decoder errors amplify under rollout.

The target program has two legitimate outcomes. A positive algorithmic result
requires a strict matched gain, a competitive best-engineered method, and
seed-stable structural and rollout improvements. A negative result is useful
only after applicable label closure, target-fit checks, continuation checks
where short-stride fit and large-stride cold failure make them relevant, and a
clear failure classification across more than one credible architecture or PDE
setting. A failed toy head or one optimizer setting cannot reject a target
family.

A subsequent paper can place the resulting forecast model in a closed-loop
data-assimilation system for genuinely chaotic PDEs. Autoencoder plus neural
dynamics plus EnKF is not itself the contribution; the missing joint capability
is a discontinuity-preserving, transferable, forecastable, and observable
latent representation.

## Work To Stop Or Defer

- Do not reopen a broad 1D target, limiter, barrier, graph-depth, or resolution
  sweep; only the bounded Line-1 stride frontier is active.
- Do not call a separately trained fixed-step model timestep-conditioned,
  implicit, or CFL-free.
- Do not treat the released 2D CPG score as autonomous-solver evidence before
  dataset provenance, legal boundaries, and characteristic graph reach pass.
- Do not make physical 2D flux or conservation claims from the current bump
  artifact without validated control volumes and face geometry.
- Do not launch a broad PCNO/MPCNO basis or smoothing sweep before the
  graph-native ripple diagnostic.
- Do not use the legacy toy CPG-style model as research evidence.
- Do not broadly sweep absolute flux or interface latents supervised only
  through next-state loss.
- Do not continue the instantaneous stride-4 Rusanov correction.
- Do not treat heavy whole-sample limiting as structure-preserving success.
- Do not scale architectures broadly before fixing target identifiability and
  the time-integration contract.
- Do not combine coordinate changes, dense supervision, constraint layers,
  MoE routing, and front tracking in the first experiment.
- Do not train independent state and flux heads that can disagree at inference;
  derive redundant solver quantities through the primary update path.
- Do not impose global componentwise TVD, total-variation, or spectral damping
  on Euler without separating smooth and discontinuous regions.
- Do not describe a grid residual as resolution invariant without a verified
  cross-resolution operator contract.
- Do not implement generic per-cell MoE or raw-field INR as a shock solution
  before simpler conservative local-global controls fail.
- Do not start a full shock-fitting platform before phase/smearing diagnostics
  establish that moving-front representation is the dominant bottleneck.
- Do not infer latent Markov closure, forecastability, or stability from low
  reconstruction error or visually smooth latent trajectories.
- Do not count a broader or weaker shock as ripple reduction, and do not start a
  broad latent-dimension, autoencoder, dynamics-model, or filter sweep.
- Do not call a fixed global latent vector a neural operator without a verified
  discretization- or geometry-transfer contract for encoder, transition, and
  decoder together.
- Do not use decode-reencode projection as hidden rollout stabilization; declare
  and ablate it as an inference intervention.
- Do not introduce observations into current 1D or bump experiments to hide
  forecast defects.
- Line-4B assimilation may be designed read-only, but defer its implementation,
  learned data assimilation, uncertainty-driven fallback, and inverse-design
  integration until the medium-horizon forecast gate is passed.

## Related Records

- README.md
- HANDOFF.md
- MECHANISTIC_DIAGNOSTIC_TRACKER.md
- MECHANISTIC_DIAGNOSTIC_PLAN.md
- SECTION_1_2_CORRECTED_BASELINES.md
- BUMP_300_DATASET_AUDIT.md
- CPGGNSPDES_REFERENCE_AUDIT.md
- CPG_EULER_DATASET_CONTRACT.md
- prompts/LINE_1_LARGE_STEP_FLOW_MAP_PROMPT.md
- prompts/LINE_2_CPGNET_MECHANISM_PROMPT.md
- prompts/LINE_3_2D_ROLLOUT_PROMPT.md
- prompts/LINE_4_LATENT_FORECAST_ASSIMILATION_PROMPT.md
