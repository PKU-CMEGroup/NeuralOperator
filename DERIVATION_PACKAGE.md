# Derivation Package

## Target

Explain when synchronized predictions on native and neighboring grids can
predict the native one-step increment error, why fitted scalar coefficients
need not transfer across populations or PDE families, and what restricted
correction can plausibly improve an autoregressive rollout.

The immediate output is a mechanism interpretation and a bounded inference
rule. It is not a convergence theorem or a proof of recurrent improvement.

## Status

COHERENT AFTER REFRAMING / EXTRA ASSUMPTION

The exact cross-grid error identities are coherent without extra assumptions.
Inferring the native error from those identities requires a statistical latent-
error assumption. A classical leading-order power law is one possible special
case, but the current dynamic-FV evidence contradicts its sign pattern. The
inference rule below is therefore a calibrated cross-resolution error model,
not Richardson extrapolation.

## Invariant Object

The invariant object is the native-grid one-call predicted-increment error in
the audited physical-volume and component-scaled Hilbert space. For native
fields `a` and `b`, define

\[
\langle a,b\rangle_W
=\frac{1}{\sum_i V_i}\sum_i V_i\sum_{k=1}^4
 \frac{a_{ik}b_{ik}}{s_k^2},
\qquad
\|a\|_W^2=\langle a,a\rangle_W.
\]

Here `V_i` are native physical cell volumes and `s_k` are the frozen residual
component scales. All mapped predictions, fits, projections, and rollout
budgets use this same object.

## Assumptions

- Coarse, native, and fine model inputs are derived from one native physical
  conservative state before the model-boundary FP32 conversion.
- Predicted increments are denormalized before conservative mapping or scoring.
- `P` maps coarse increments to native by piecewise-constant prolongation and
  `R` maps fine increments to native by conservative block restriction.
- The native physical-volume metric, residual component scales, normalization,
  boundary policy, checkpoint, and grid triple are fixed within an experiment.
- The first eight physical-cosine modes are constructed by weighted QR on
  dynamic node type 0. Corrections remain zero on node types 1--3.
- A useful part of the native error lies in a low-dimensional, temporally
  persistent subspace whose response to grid representation is observable in
  synchronized cross-grid discrepancies. The remaining local component may be
  weakly predictable or cancelling.
- Linearized rollout analysis is used only to explain a possible accumulation
  mechanism. The actual recurrent result must be measured.

## Notation

- `r_*`: native reference increment for one physical step.
- `r_c`, `r_h`, `r_f`: coarse, native, and fine predicted increments after
  mapping each prediction to the native grid.
- `e_c=r_c-r_*`, `e_h=r_h-r_*`, `e_f=r_f-r_*`: offline prediction errors.
- `y=r_*-r_h=-e_h`: ideal additive native correction, available only offline.
- `x_c=r_h-r_c` and `x_f=r_f-r_h`: deployable synchronized discrepancies.
- `Pi_8`: fixed weighted projector onto the first eight physical-cosine modes.
- `Pi_7`: `Pi_8` with the weighted constant spatial mode removed.
- `delta_n`: recurrent state error at call `n`.

## Derivation Strategy

Start from exact mapped-error identities. Then derive the population least-
squares estimator and expose its dependence on the error covariance. Separate
the stable fine-discrepancy direction from the small coarse innovation by an
exact orthogonalization. Finally, insert the correction into a first-order
rollout error recurrence to state why teacher-forced skill is necessary but not
sufficient.

## Derivation Map

1. Common-source mapping gives exact relations among `x_c`, `x_f`, and the
   three offline errors.
2. A linear correction is the orthogonal projection of `y` onto the span of
   the deployable discrepancies under a declared population measure.
3. Its coefficients depend on population covariance and therefore are not
   universal constants.
4. A leading-order mesh power law would impose a distinctive collinearity and
   coefficient-sign pattern; the observed sign pattern does not satisfy it.
5. Orthogonalizing the coarse discrepancy against the fine discrepancy
   isolates the incremental information carried by the unstable coefficient.
6. The retained evidence shows that this coarse innovation is negligible,
   motivating a fixed fine-only rule rather than another two-scalar fit.
7. A recurrent correction helps only when the propagated sum of corrected
   low-rank defects decreases without creating state, front, integral,
   boundary, or admissibility harm.

## Main Derivation

### Step 1: exact synchronized-error identities

By definition,

\[
x_c=r_h-r_c=e_h-e_c,
\qquad
x_f=r_f-r_h=e_f-e_h,
\qquad
y=-e_h.
\]

These are identities. They require no convergence model. They also show why
independently diverged coarse and fine rollouts are invalid for this inference:
their differences would additionally contain input-state error.

### Step 2: population-optimal scalar correction

For `X=(x_c,x_f)` and `theta=(alpha,beta)`, the no-intercept population problem
is

\[
\theta_{\mathcal P}
=\arg\min_\theta
 \mathbb E_{\mathcal P}\|y-X\theta\|_W^2.
\]

Define

\[
G_{ij}=\mathbb E_{\mathcal P}\langle x_i,x_j\rangle_W,
\qquad
b_i=\mathbb E_{\mathcal P}\langle x_i,y\rangle_W.
\]

When `G` is nonsingular, the exact least-squares solution is

\[
\theta_{\mathcal P}=G^{-1}b.
\]

This identity makes the transfer limitation explicit: changing the PDE family,
checkpoint, grid triple, state distribution, time distribution, boundary
policy, or metric changes `G`, `b`, or both. Equal numerical coefficients
across those changes are an empirical hypothesis, not a consequence of the
cross-grid construction.

### Step 3: the leading-power-law special case

Suppose, only as a special approximation, that mapped prediction error has one
leading field `a` and one stable order `p>0`:

\[
e(h)=a h^p+o(h^p).
\]

Ignoring higher-order and mapping terms gives

\[
x_c\simeq a h^p(1-2^p),
\qquad
x_f\simeq a h^p(2^{-p}-1).
\]

The two discrepancies must then be positively collinear, their norm ratio is
approximately `2^p`, and either discrepancy yields a positive coefficient for
the correction `y=-a h^p`. The retained dynamic-FV fits instead have a stable
negative fine coefficient, and the two-term coarse coefficient changes sign
between the early calibration and all-open populations. Therefore the current
evidence does not support this leading-error model or the term Richardson
extrapolation.

### Step 4: exact coarse-innovation decomposition

Define the population projection of the coarse discrepancy onto the fine
discrepancy,

\[
\lambda=\frac{\mathbb E\langle x_c,x_f\rangle_W}
              {\mathbb E\|x_f\|_W^2},
\qquad
v=x_c-\lambda x_f.
\]

Then `E< v,x_f >_W=0`, and any two-term correction can be written exactly as

\[
\alpha x_c+\beta x_f
=\alpha v+(\beta+\alpha\lambda)x_f.
\]

The extra normalized target-energy reduction available from the coarse
innovation after fitting the fine direction is

\[
\Delta S_v=
\frac{\mathbb E\langle v,y\rangle_W^2}
     {\mathbb E\|v\|_W^2\,\mathbb E\|y\|_W^2}.
\]

On the retained adaptive dynamic-FV statistics, `Delta S_v` is `0.004694` on
the 18-case calls-0--19 calibration block and `0.000266` on all 24 open cases
and calls. The two-term versus fine-only cross-fitted skill differences are
similarly small (`0.004671` and `0.000039`). The unstable coarse coefficient is
therefore attached to a low-value innovation rather than to the dominant
predictable direction.

### Step 5: fixed fine-discrepancy hypothesis

The fine-only coefficient is negative in every retained calibration fold and
every all-open strength-pair fold. A simple fixed hypothesis close to both
population optima is

\[
c_8=-\frac12\Pi_8 x_f
=\frac12\Pi_8(r_h-r_f).
\]

This rule moves the native prediction away from the mapped fine prediction in
the observed low-rank direction. It is cross-resolution error cancellation,
not averaging and not evidence that either grid is closer to a continuum
operator. The coefficient `-1/2` is an adaptive hypothesis chosen from the
open dynamic-FV evidence; its fixed form reduces fitted degrees of freedom but
does not make it family universal.

For an integral-budget safety arm, let `q_0` be the weighted constant column of
the QR basis and define

\[
\Pi_7=\Pi_8-q_0q_0^T_W,
\qquad
c_7=-\frac12\Pi_7 x_f.
\]

Because subsequent QR columns are weighted-orthogonal to the constant column,
`c_7` has zero type-0 physical-volume integral in each conservative component
up to numerical closure and is zero on excluded contact nodes. This is an
identity for the applied correction only. It neither makes PCNO conservative
nor guarantees equal future integrals after nonlinear recurrence.

Both nonzero rules use the target-free trust region

\[
c\leftarrow \sigma c,
\qquad
\sigma=\min\left(1,
\frac{0.10\,\|r_h\|_W}{\|c\|_W}\right),
\]

with an explicit denominator status. This bounds intervention size but does
not certify correctness.

### Step 6: recurrent error accumulation

Let the raw learned one-step map be `N` and the exact sampled flow map be
`Phi`. Around the reference state, write

\[
N(U_n+\delta_n)-\Phi(U_n)
\simeq A_n\delta_n+e_n,
\]

where `A_n` is the local response to incoming state error and `e_n` is the
fresh teacher-forced defect. With correction `c_n`,

\[
\delta_{n+1}^{c}\simeq A_n\delta_n^{c}+e_n+c_n.
\]

Iterating this approximation gives

\[
\delta_N^{c}
\simeq A_{N-1:0}\delta_0
+\sum_{k=0}^{N-1}A_{N-1:k+1}(e_k+c_k).
\]

A small systematic improvement in a temporally persistent low-rank component
can therefore reduce a coherent accumulated forcing. Conversely, a correction
that perturbs a locally cancelling shock component, integral budget, or state
distribution can be amplified even when its teacher-forced squared error is
smaller. This is why one-step skill is an entry gate rather than a rollout
claim.

## Remarks and Interpretation

- The retained feature-feature cosine is high, so a two-feature coefficient can
  be numerically stable within folds while its low-value innovation coefficient
  changes sign under a population shift.
- Dropping the coarse recurrent call is justified only as an adaptive compute-
  saving hypothesis for this experiment. Teacher-forced collection still
  evaluates all three grids and reports the omitted innovation.
- Numeric coefficients should be calibrated and validated independently for a
  new family. What may transfer is the protocol: common-source discrepancies,
  weighted modal analysis, zero-inclusive trust regions, and recurrent
  no-harm gates.
- A second-family test should use a family with real nested physical grids and
  reference evolution. Supersonic-bump query-node dropping is not such a test.

## Boundaries and Non-Claims

- No stable asymptotic order or leading mesh-error expansion is demonstrated.
- The derivation does not establish resolution invariance, operator
  convergence, or a conservative learned solver.
- The `-1/2` gain is not claimed to transfer across PDE families, architectures,
  checkpoints, timesteps, or grid ratios.
- True residual error appears only in offline calibration and evaluation.
- Zero physical-volume mean of `c_7` is not a physical conservation theorem.
- Teacher-forced improvement does not imply autoregressive improvement.
- Dynamic-FV and bump evidence remain separate.

## Open Risks

- The fine-only sign may change on a genuinely new family or checkpoint.
- The fixed cosine basis may not align with the persistent error subspace after
  recurrent state drift.
- Removing the constant mode may remove useful correctable error, while keeping
  it may perturb integral controls.
- The intervention cap may activate selectively and introduce its own state-
  dependent bias.
- The available dynamic open population has already influenced method choice;
  any further result on it is adaptive development evidence.
