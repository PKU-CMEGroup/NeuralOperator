# Corrective Mechanisms Under Self-Composition: Theory And Taxonomy

Updated: 2026-08-26

Status: T0 theory and field-coverage freeze; no experiment authorization

## Target

Formalize the narrow claim that clean trajectory supervision does not, by
itself, identify deployment-relevant off-reference response in a sufficiently
rich model class. Then separate that identification result from sufficient
stability bounds, path-accuracy bounds, and the empirical question of which
information source is useful in practice.

The invariant scientific object is the **complete deployed transition**. For a
trusted step `Phi_n`, raw predictor `F_n`, and an optional explicit corrector
`C_(n+1)`, write

\[
\Psi_n=C_{n+1}\circ F_n,
\qquad
\widehat u_{n+1}=\Psi_n(\widehat u_n).
\]

The factorization is non-unique. Accuracy, stability, response, and rollout
claims therefore attach to `Psi_n`. A separate component is called a corrector
only when it has a declared, falsifiable contract. A generic Fourier,
attention, MLP, or gradient-path change is an architectural choice, not a
corrector by retrospective naming.

## Status

**COHERENT AFTER REFRAMING / EXTRA ASSUMPTIONS**

The exact rollout identities require no smooth-manifold assumption. The
clean-trace non-identifiability result is valid after explicitly restricting it
to a normal-rich hypothesis/world class. The local normal-cocycle result needs
a smooth tubular neighborhood, exact trace invariance, and differentiability.
The finite-amplitude tube and tangent recurrences are sufficient bounds under
declared inequalities; they are not universal properties of neural operators.

The strongest defensible necessity statement is:

> If the admissible world or hypothesis class can vary its off-reference
> extension while preserving every clean trajectory observation, then clean
> trajectory data do not identify that extension. A guarantee uniform over
> the clean-data-equivalence class needs equivalence-breaking information or a
> justified structural restriction.

This is **not** a theorem that every successful model needs a separate
post-step correction network. Exact predictors, correct structural priors, and
appropriately contractive reference dynamics are immediate counterexamples to
that stronger claim.

## Claim Labels

Every statement below has one of four labels.

| Label | Meaning |
| --- | --- |
| **Identity** | Algebraic equality under the displayed definitions. |
| **Conditional theorem** | A proved implication under explicit assumptions. |
| **Sufficient bound** | A usable guarantee that need not be necessary. |
| **Empirical hypothesis** | A proposition that requires an experiment and is not implied by the mathematics. |

## Invariant Object And Notation

### Deterministic state evolution

Work first at one fixed spatial discretization in a finite-dimensional weighted
Hilbert space `(X, <.,.>_W)`. This avoids silently promoting a
fixed-discretization statement to an operator-convergence theorem.

- `M_n` is the reference-state set at call `n`.
- `Phi_n : U_n -> X` is the trusted transition on a neighborhood `U_n`.
- `Psi_n : U_n -> X` is the complete deployed learned transition.
- `u_(n+1)=Phi_n(u_n)` is a reference trajectory.
- `u_hat_(n+1)=Psi_n(u_hat_n)` is an autonomous learned trajectory.
- `e_n=u_hat_n-u_n` is the pathwise state error.
- `Delta_n(x)=Psi_n(x)-Phi_n(x)` is the complete deployed defect.
- `mu_n` is a clean-state law supported on `M_n`.
- `mu_hat_n` is the on-policy law induced by repeated `Psi` composition.

When a smooth tube is assumed, `pi_n` is the nearest-point projection,
`T_u M_n` and `N_u M_n` are the tangent and normal spaces, and `P^T_n(u)` and
`P^N_n(u)` are their orthogonal projectors. Define

\[
r_n=\operatorname{dist}(\widehat u_n,M_n),
\qquad
m_n=\pi_n(\widehat u_n),
\qquad
a_n=\|m_n-u_n\|_W.
\]

Here `r_n` measures normal/tube displacement and `a_n` measures displacement
along the reference set. Their sum bounds total path error, but neither one
determines the other.

### Controlled, observed, stochastic, and history-dependent evolution

The field-complete object is a conditional transition kernel on an augmented
state,

\[
K_\theta(dz_{n+1}\mid z_n,a_n,y_{n+1}),
\]

where `z_n` contains every variable needed for closure: the physical state and,
when applicable, history, clock, actuator state, controller memory, solver
mode, or refinement state. `a_n` is an action and `y_(n+1)` is an observation.
Randomness is represented by the kernel rather than by pretending every method
has a deterministic `C o F` form.

Training-only interventions select `theta`; they are not deployed correctors.
Observation assimilation is a deployed information channel; it is not an
autonomous-rollout guarantee. Memory augmentation changes the state on which a
Markov transition is learned.

## Assumptions

The results use only the assumptions named beside them.

### Exact identities

- **I1.** `Phi_n` and `Psi_n` are defined at the states in the displayed
  trajectory. No tube or differentiability is required.

### Clean-trace non-identifiability

- **N1 (fixed discretization).** `X` is finite dimensional.
- **N2 (smooth tube).** `M_n` is a compact embedded `C^r` submanifold,
  `r >= 2`, with positive codimension and a tubular neighborhood of radius
  `rho_n`.
- **N3 (reference trace).** `Phi_n` is `C^r` on that neighborhood and maps
  `M_n` into `M_(n+1)`.
- **N4 (clean information).** The learner observes only trajectories starting
  on `M_0` and remaining on the `M_n`. The theorem grants the learner the
  stronger oracle information `Phi_n|M_n`; finite samples reveal no more.
- **N5 (normal richness).** The admissible world or hypothesis class contains
  smooth tube extensions that agree on `M_n` while varying a nonzero normal
  bundle direction. The theorem does not assert this for every fixed neural
  architecture.

### Normal cocycle and local stability

- **S1 (exact invariant trace).** `Psi_n(M_n)` is a subset of `M_(n+1)`.
- **S2 (regularity).** `Psi_n` is `C^2` on a uniform tube around `M_n`.
- **S3 (compatible iterates).** Every intermediate iterate used by a block
  theorem remains inside the declared tubes.
- **S4 (uniform geometry).** Tube curvature and second-derivative constants
  are bounded on the compact reference sets.

### Tangent/path accuracy

- **T1.** The projected learned transition is well defined on the relevant
  tube.
- **T2.** The trusted trace map is Lipschitz along `M_n` in the chosen path
  metric.
- **T3.** Off-manifold input has a bounded effect on the projected output.
- **T4.** The projected learned trace has a separately bounded tangent defect.

Shocks, contact discontinuities, topology changes, nonsmooth admissible sets,
and an empirical cloud of trajectories do not automatically satisfy N2 or S2.
For those regimes, the smooth-tube theorem is an explanatory local model until
a stratified, cone, reach, or set-valued replacement is justified.

## Dependency Map

```text
complete deployed map Psi
    |
    +--> exact rollout identity --> on-policy defect is the relevant defect
    |
clean trace + normal-rich class
    +--> indistinguishable extensions --> minimax information requirement
    |
exact invariant trace + C2 tube
    +--> normal cocycle --> local tube contraction (sufficient)
    |
finite-amplitude tube inequality
    +--> product-sum tube bound
    |
tangent Lipschitz + cross-coupling + trace defect
    +--> separate path-accuracy recurrence
```

The non-identifiability theorem does not imply contraction. The contraction
theorem does not imply path accuracy. None of these fixed-grid results implies
continuum convergence.

## Main Derivation

### Proposition 1: exact rollout-defect identity

**Label: Identity.** Under I1,

\[
\begin{aligned}
e_{n+1}
&=\Psi_n(u_n+e_n)-\Phi_n(u_n)\\
&=\Phi_n(u_n+e_n)-\Phi_n(u_n)
  +\Delta_n(u_n+e_n).
\end{aligned}
\]

With the clean defect and response defect

\[
d_n(u)=\Delta_n(u),
\qquad
q_n(u,\eta)=\Delta_n(u+\eta)-\Delta_n(u),
\]

the learned contribution is exactly

\[
d_n(u_n)+q_n(u_n,e_n)=\Delta_n(u_n+e_n).
\]

**Proof.** Add and subtract `Phi_n(u_n+e_n)`, then add and subtract
`Delta_n(u_n)`. No linearization is used. \(\square\)

**Interpretation.** One-step clean risk probes `Delta_n` under `mu_n`; an
autonomous rollout probes it under `mu_hat_n`. Small clean defect does not bound
the response term without another assumption or information source.

### Proposition 2: exact law mismatch under self-composition

**Label: Identity.** Reference and learned laws satisfy

\[
\mu_{n+1}=(\Phi_n)_\#\mu_n,
\qquad
\widehat\mu_{n+1}=(\Psi_n)_\#\widehat\mu_n.
\]

Therefore a one-step objective

\[
\mathbb E_{u\sim\mu_n}\|\Delta_n(u)\|_W^2
\]

does not, as an algebraic matter, equal the deployment objective

\[
\mathbb E_{x\sim\widehat\mu_n}\|\Delta_n(x)\|_W^2.
\]

Equality requires an additional condition, such as equal laws, a uniform defect
bound, or a valid change-of-measure argument.

### Lemma 1: smooth trace-preserving normal perturbations

**Label: Conditional theorem.** Assume N1--N3. Let

\[
B_n(u):N_uM_n\longrightarrow N_{\Phi_n(u)}M_{n+1}
\]

be a `C^(r-1)` normal-bundle map. For sufficiently small norm, there exists a
`C^(r-1)` perturbation `H_B` supported inside the tube such that

\[
H_B(u)=0,
\qquad
DH_B(u)\tau=0,
\qquad
DH_B(u)\nu=B_n(u)\nu
\]

for `tau in T_uM_n` and `nu in N_uM_n`.

**Proof.** Tubular coordinates write every nearby point uniquely as
`x=u+v`, with `v in N_uM_n`. Let `chi` be a smooth cutoff equal to one near
zero and supported before the tube boundary. Define

\[
H_B(u+v)=\chi(\|v\|_W)B_n(u)v.
\]

At `v=0`, the value is zero. Derivatives of `B_n(u)` and the cutoff are
multiplied by `v`, so the tangent derivative vanishes there; the normal
derivative is `B_n(u)`. Compactness supplies a common support radius.
\(\square\)

### Theorem 1: clean-trace non-identifiability

**Label: Conditional theorem.** Assume N1--N5. For any two admitted bundle
maps `B^0_n` and `B^1_n`, define two trusted worlds

\[
\Phi_n^i=\Phi_n+H_{B^i_n},\qquad i\in\{0,1\}.
\]

Then:

1. `Phi_n^0(u)=Phi_n^1(u)=Phi_n(u)` for every `u in M_n`;
2. every clean-start trajectory and every clean one-step pair has the same law
   in the two worlds; but
3. their normal response blocks differ by

\[
P^N_{n+1}D\Phi_n^1(u)P^N_n
-P^N_{n+1}D\Phi_n^0(u)P^N_n
=B^1_n(u)-B^0_n(u).
\]

For a finite perturbation `t nu`,

\[
\Phi_n^1(u+t\nu)-\Phi_n^0(u+t\nu)
=t(B^1_n-B^0_n)\nu+O(t^2).
\]

**Proof.** Lemma 1 gives equality on `M_n` and the derivative formula. Since
both worlds map the same point of `M_n` to the same point of `M_(n+1)`, induction
over `n` gives identical clean trajectories. Taylor expansion gives the
finite-amplitude statement. \(\square\)

**Strength of the result.** The theorem grants exact knowledge of the entire
clean trace, not merely a finite training set. It therefore isolates
off-reference ambiguity from ordinary sample error.

**Boundary of the result.** N5 is essential. A hypothesis class that fixes the
off-reference extension by construction can remove the ambiguity. The theorem
does not prove that a particular PCNO, FNO, FFNO, or attention operator realizes
every `B_n`.

### Corollary 1: minimax need for equivalence-breaking information

**Label: Conditional theorem.** Let `R_i` be any target response object that
differs between the two worlds in Theorem 1, measured in a normed space. For
any possibly randomized estimator `R_hat(D)` based only on clean data `D`,

\[
\max_{i\in\{0,1\}}
\mathbb E_i\|\widehat R(D)-R_i\|
\geq \frac12\|R_1-R_0\|.
\]

**Proof.** The data law is identical in both worlds. For every realized
estimate `r`, the triangle inequality gives
`||R_1-R_0|| <= ||r-R_0||+||r-R_1||`. Take expectation under the common data
law and then the larger of the two risks. \(\square\)

If the worlds lie on opposite sides of a declared local-response threshold,
any clean-data-only binary certificate has worst-case error at least `1/2`.
This is an indistinguishability result, not an assertion of global nonlinear
instability.

**What can break the equivalence.** At least one of the following must enter:

- trusted labels at displaced states;
- a justified structural architecture or physical parameterization that fixes
  the extension;
- an analytic projection, invariant, boundary, or admissibility rule that is
  valid for the claimed response object;
- deployment feedback or a solver in the loop;
- deployment observations;
- a long-time statistical target; or
- another oracle that contains the missing response information.

The theorem says one of these kinds of information or restriction is needed
for a **uniform guarantee over the normal-rich equivalence class**. It does not
say all are needed, or that solver labels always improve finite-data practice.
Stored-target multistep feedback can select a recovery behavior and improve a
finite rollout objective, but it does not distinguish two trusted worlds that
agree on every clean trace. It therefore does not, by itself, identify
`Phi(u+eta)`.

### Proposition 3: normal cocycle of the complete deployed map

**Label: Identity under S1 and differentiability.** Differentiating the
inclusion of `Psi_n(M_n)` in `M_(n+1)` gives

\[
D\Psi_n(u)T_uM_n\subseteq T_{\Psi_n(u)}M_{n+1}.
\]

The induced one-step normal block is

\[
A_n(u)=P^N_{n+1}(\Psi_n(u))D\Psi_n(u)P^N_n(u).
\]

Along a reference trace of `Psi`, the `J`-step normal cocycle is

\[
A_{n:J}=A_{n+J-1}\cdots A_n.
\]

Tangent components created by a normal perturbation do not later re-enter the
normal quotient because tangent inputs remain tangent under the exact trace
map.

#### Explicit predictor/corrector factorization

Suppose `Z_(n+1)=F_n(M_n)` is an embedded intermediate reference set,
`F_n(M_n)` is contained in `Z_(n+1)`, and `C_(n+1)(Z_(n+1))` is contained in
`M_(n+1)`. Then

\[
A_n^\Psi=A_{n+1}^C A_n^F,
\]

with the normal spaces taken relative to `M_n`, `Z_(n+1)`, and `M_(n+1)`.
This follows from the chain rule and tangent invariance at both stages.

Without a compatible intermediate set, the only unconditional derivative
identity is

\[
D(C\circ F)=DC\,DF.
\]

After inserting arbitrary tangent/normal projectors, cross terms remain. It is
then invalid to multiply a quoted predictor gain by a quoted corrector gain and
call the product the deployed normal response.

Because `C o F` is non-unique, the component blocks are explanatory only under
a fixed factorization contract. `A_n^Psi` is the invariant deployed object.

### Theorem 2: local normal attraction from the normal derivative

**Label: Conditional theorem and sufficient bound.** Assume S1--S4. There are
constants `K_n` and tube radii small enough that

\[
\operatorname{dist}(\Psi_n(x),M_{n+1})
\leq
\|A_n(\pi_nx)\|\operatorname{dist}(x,M_n)
+K_n\operatorname{dist}(x,M_n)^2.
\]

If `sup_u ||A_n(u)|| <= q < 1`, choose a tube radius `r` such that
`q+K_n r <= alpha < 1`. Then

\[
\operatorname{dist}(\Psi_n(x),M_{n+1})
\leq \alpha\operatorname{dist}(x,M_n)
\]

inside that tube.

**Proof sketch.** Write `x=u+v` in tubular coordinates. Taylor expand `Psi_n`
at `u`. Its tangent derivative lies in `T M_(n+1)`. The first-order normal
distance is therefore the normal block applied to `v`; bounded second
derivatives and tube curvature contribute `O(||v||^2)` uniformly.
\(\square\)

For a `J`-step block, replace `Psi_n` by its composition. If the complete
normal cocycle has norm below one and every intermediate iterate stays inside
its tube, the same argument gives local block contraction. One-step
contraction is sufficient, not necessary: some steps may expand while the
block cocycle contracts.

Conversely, if a differentiable exact-trace block satisfies a uniform
nonlinear tube contraction with factor `q` to first order, division by the
perturbation amplitude and passage to zero imply that its normal derivative
norm is at most `q`. An empirical negative average Lyapunov estimate is weaker
than either uniform statement.

### Proposition 4: finite-amplitude tube recurrence

**Label: Sufficient bound.** Suppose the actually deployed map satisfies

\[
\operatorname{dist}(\Psi_n(x),M_{n+1})
\leq \alpha_n\operatorname{dist}(x,M_n)+\zeta_n
\]

on the entire reached tube. Then

\[
r_N\leq
\left(\prod_{j=0}^{N-1}\alpha_j\right)r_0
+\sum_{k=0}^{N-1}
\zeta_k\prod_{j=k+1}^{N-1}\alpha_j.
\]

This follows by induction. If `alpha_n <= alpha_bar < 1` and
`zeta_n <= zeta_bar`,

\[
r_N\leq \bar\alpha^N r_0
+\frac{1-\bar\alpha^N}{1-\bar\alpha}\bar\zeta.
\]

For one fixed explicit factorization, sufficient component contracts are

\[
\begin{aligned}
\operatorname{dist}(F_n(x),Z_{n+1})
&\leq q_n\operatorname{dist}(x,M_n)+\epsilon_n,\\
\operatorname{dist}(C_{n+1}(z),M_{n+1})
&\leq \rho_{n+1}\operatorname{dist}(z,Z_{n+1})+\delta_{n+1}.
\end{aligned}
\]

They imply

\[
\alpha_n=\rho_{n+1}q_n,
\qquad
\zeta_n=\rho_{n+1}\epsilon_n+\delta_{n+1}.
\]

These constants are factorization-dependent; the resulting bound on `Psi` is
the deployed claim. Pointwise `alpha < 1` is not necessary for bounded finite
horizon behavior, and bounded tube distance is not path accuracy.

### Proposition 5: separate tangent/path-accuracy recurrence

**Label: Sufficient bound.** Let

\[
T_n(m)=\pi_{n+1}(\Psi_n(m)).
\]

Assume, on the reached tube,

\[
\begin{aligned}
\|\Phi_n(m)-\Phi_n(u)\|_W
&\leq L_n^T\|m-u\|_W,\\
\|T_n(m)-\Phi_n(m)\|_W
&\leq \epsilon_n^T,\\
\|\pi_{n+1}(\Psi_n(x))-T_n(\pi_nx)\|_W
&\leq \gamma_n\operatorname{dist}(x,M_n).
\end{aligned}
\]

Then

\[
a_{n+1}\leq L_n^T a_n+\gamma_n r_n+\epsilon_n^T,
\qquad
\|\widehat u_n-u_n\|_W\leq r_n+a_n.
\]

**Proof.** Insert `T_n(m_n)` and `Phi_n(m_n)` between
`pi_(n+1)(Psi_n(u_hat_n))` and `Phi_n(u_n)`, then apply the three displayed
bounds and the triangle inequality. \(\square\)

Consequences:

- normal contraction can coexist with tangent drift;
- exact admissibility or projection can improve `r_n` while harming `a_n`;
- in chaotic dynamics, positive tangent growth can make long pathwise tracking
  impossible even when the invariant measure is reproduced; and
- long-time statistical fidelity, phase accuracy, and tube stability require
  separate metrics and claims.

### Proposition 6: local objectives identify different response objects

**Label: Identity to second order; interpretation thereafter.** Near a clean
state, let the trusted normal response be `A_* eta`, the learned response be
`b+A eta`, and let centered perturbations have covariance `Sigma`. Ignoring
quadratic remainders,

\[
\begin{aligned}
L_{\mathrm{clean}}
&=\|b\|_W^2,\\
L_{\mathrm{dyn}}
&=\|b\|_W^2+
\operatorname{tr}((A-A_*)\Sigma(A-A_*)^*),\\
L_{\mathrm{recover}}
&=\|b\|_W^2+\operatorname{tr}(A\Sigma A^*).
\end{aligned}
\]

Clean targets identify the on-trace bias. Dynamics-consistent displaced-state
labels identify `A_*` on excited directions. Denoise/recover-to-clean targets
favor retraction. These objectives coincide only under additional structure,
for example `A_*=0` on the excited directions.

This calculation explains why solver-free recovery can stabilize a rollout yet
learn the wrong physical response from a legitimate displaced state. It does
not prove that recovery is harmful in a given PDE.

## Pushforward And Solver Semantics

The word `pushforward` must not hide the target source.

| Contract | Model input | Target | Trusted solver call on that input? | Identified object |
| --- | --- | --- | --- | --- |
| Clean one-step | `u_n` | stored `u_(n+1)` | No | Trace value on clean states |
| Detached self-input | `stopgrad(Psi(u_(n-1)))` | stored `u_(n+1)` | No | Recovery/cancellation along generated prefix directions |
| Supervised unrolling | repeated learned states | stored future clean states | No | Finite composed path relative to the stored trajectory |
| Solver-free recovery | `u+eta` | clean state or clean next state | No | Retraction/recovery target chosen by the designer |
| Dynamics-consistent relabeling | `x=u+eta` or `Psi(u)` | `Phi(x)` | Yes, unless another off-state oracle exists | Trusted response from the actual displaced state |

The solver need not be differentiable when off-state labels are cached or when
the relevant branch is detached. It must be differentiable only when gradients
are propagated through its computation. Conversely, stored future frames do
not become displaced-state solver labels merely because the model generated
the input.

## Stochastic And Controlled Extension

For an augmented stochastic state, a mean-square tube condition can replace the
deterministic inequality. If

\[
\left(
\mathbb E[d(Z_{n+1},M_{n+1})^2\mid Z_n=z]
\right)^{1/2}
\leq \alpha_n d(z,M_n)+\zeta_n,
\]

then Minkowski's inequality gives the same product-sum recurrence for
`R_n=(E d(Z_n,M_n)^2)^(1/2)`. A Wasserstein kernel comparison can instead target
conditional distributional fidelity.

Actions and observations must be included in the conditioning law. A model
that omits actuator state, history needed for closure, or observation timing is
not being tested as the same transition object. This is why a controlled or
data-assimilative application cannot be reduced silently to deterministic
`C o F`.

## Counterexamples And Nonclaims

### Exact predictor: no separate corrector is necessary

Set `Psi=Phi`. The complete deployed defect is zero and no separate correction
network is needed. This refutes any universal module-necessity claim.

### Structural prior can remove ambiguity

If the hypothesis class fixes the normal derivative to the trusted one, the two
worlds in Theorem 1 are not both admitted. The missing information has been
supplied as a structural restriction rather than as displaced-state labels.

### Perfect tube contraction can create path error

Let `M` be the unit circle. In polar coordinates define, near `M`,

\[
\Phi(r,\theta)=(1,\theta+\omega),
\qquad
\Psi(r,\theta)=(1,\theta+\omega+\gamma(r-1)).
\]

Both maps agree on `M`, and `Psi` sends every nearby state exactly onto `M`, so
its next-step tube distance is zero. But an off-manifold deviation produces a
tangent phase error `gamma(r-1)` that can persist or grow under later
composition. Normal stability is not path fidelity.

### Projection can be physically wrong off the data manifold

If the trusted dynamics carries a legitimate normal perturbation forward, a
projection that maps every displaced input back to the clean trace has perfect
recovery loss but does not approximate `Phi(u+eta)`. Constraint satisfaction
does not uniquely determine dynamics.

### Contractive reference dynamics is not a learned-correction result

If `Phi` itself attracts the relevant tube and `Psi` approximates it uniformly,
rollout stability may follow without a special correction mechanism. The
reference dynamics and learned defect must be separated.

### Fixed-grid theory is not operator convergence

A continuum claim additionally needs a family of spaces, discretization and
reconstruction maps, solver consistency, and normal/tangent defects that vanish
under independently controlled spatial and temporal refinement.

## Literature Coverage Protocol

### Search contract

- **Cutoff:** 2026-08-26.
- **Scope:** time-dependent learned PDE evolution, autoregressive physical
  simulators, neural operators, hybrid learned/numerical solvers, long-time
  statistical objectives, data assimilation, memory, and methods that reduce
  recurrence.
- **Primary-source rule:** method claims were checked against official
  proceedings, OpenReview records, journal pages, or arXiv abstracts. Survey
  prose and search-result summaries were not used as scientific authority.
- **Status rule:** peer-reviewed papers and preprints are marked separately.
- **Inclusion:** a work must alter the information source, learning rule,
  deployed transition, state closure, constraint set, or recursion pattern in a
  way relevant to repeated evolution.
- **Exclusion:** a generic backbone replacement, benchmark, or robustness test
  is not called a corrective mechanism unless it declares such an intervention
  and target. Static operator learning is included only when its structural
  contract transfers directly to a time-stepper.

Public query families included combinations of:

```text
autoregressive neural operator stability long horizon PDE
neural PDE solver pushforward unrolled training detached
solver in the loop learned PDE correction
noise injection recovery learned physical simulator rollout
neural operator conservation boundary energy constraint
neural operator invariant measure chaotic attractor
neural operator data assimilation observation correction
memory time-dependent PDE neural operator
stochastic refinement neural PDE rollout
hybrid PDE solver neural corrector
space-time neural operator direct trajectory prediction
```

The search was seeded by existing public citations in the branch, then updated
with targeted 2025--2026 searches. The claim is a reproducible, search-bounded
taxonomy, not literal coverage of every future method.

### Four-axis taxonomy

Every method is described by

\[
(\text{intervention locus},\ \text{correction target},\
  \text{information source},\ \text{guarantee}).
\]

| Family and representative primary sources | Intervention locus | Target | Information source | Strongest warranted guarantee category |
| --- | --- | --- | --- | --- |
| Clean one-step neural evolution, including [FNO](https://openreview.net/forum?id=c8P9NQVtmnO) (ICLR 2021) | Supervised objective | Clean trace | Stored clean pairs | Empirical one-step/operator approximation; no automatic rollout guarantee |
| [Scheduled Sampling](https://proceedings.neurips.cc/paper_files/paper/2015/hash/e995f98d56967d946471af29d7bf99f1-Abstract.html) (NeurIPS 2015) and [Professor Forcing](https://proceedings.neurips.cc/paper/2016/hash/16026d60ff9b54410b3435b403afd226-Abstract.html) (NeurIPS 2016) | Training input law or latent-distribution alignment | Teacher/free-running mismatch | Model samples plus stored sequence targets | Empirical sequence robustness; no PDE off-state fidelity |
| Noise and recovery in [Graph Network-based Simulators](https://proceedings.mlr.press/v119/sanchez-gonzalez20a.html) (ICML 2020) and [MeshGraphNets](https://openreview.net/forum?id=roNqYL0_XP) (ICLR 2021) | Input corruption and supervised target | Recovery from accumulated-like noise | Synthetic noise plus clean trajectories | Empirical rollout robustness |
| Detached pushforward in [MP-PDE](https://openreview.net/forum?id=vSix3HPYKSU) (ICLR 2022) | Stopped-gradient generated input | Exposure/recovery on generated prefixes | Model state plus stored future target | Empirical finite-horizon improvement; no `Phi(x)` label at generated `x` |
| Supervised unrolling and its optimization variants in [APEBench](https://proceedings.neurips.cc/paper_files/paper/2024/hash/d9875ebcf74bccdc5076acab0dbee62c-Abstract-Datasets_and_Benchmarks_Track.html) (NeurIPS 2024) and [stabilized BPTT](https://openreview.net/forum?id=bozbTTWcaw) (ICLR 2024) | Multistep training objective and gradient path | Finite-prefix composition | Stored trajectories; model-generated intermediate states | Empirical finite-horizon behavior and optimization evidence |
| Diverted-chain relabeling in [APEBench](https://proceedings.neurips.cc/paper_files/paper/2024/file/d9875ebcf74bccdc5076acab0dbee62c-Paper-Datasets_and_Benchmarks_Track.pdf) and [Solver-in-the-Loop](https://proceedings.neurips.cc/paper/2020/hash/43e4e6a6f341e00671e123714de019a8-Abstract.html) (NeurIPS 2020) | Training-time solver coupling | Dynamics-consistent response or correction interaction | Trusted solver evaluated on model-affected states | Empirical solver-relative improvement; differentiability depends on gradient contract |
| Stability-oriented architecture in [Towards Stability of Autoregressive Neural Operators](https://openreview.net/forum?id=RFfUUtKYOG) (TMLR 2023) | Architecture and spectral operations | Selected amplification/aliasing mechanisms | Structural prior plus clean data | Analysis and empirical long-rollout improvement, not universal stability |
| [SGNO](https://arxiv.org/abs/2602.18801) (arXiv preprint, 2026) | Generator parameterization, gating, filtering | Latent amplification and high-frequency feedback | Structural spectral prior | Stated one-step and finite-horizon sufficient bounds; preprint evidence |
| Boundary, conservation, and energy structure in [BOON](https://openreview.net/forum?id=gfWNItGOES6) (ICLR 2023), [clawNO](https://proceedings.mlr.press/v235/liu24p.html) (ICML 2024), and [Energy-Consistent Neural Operators](https://proceedings.mlr.press/v258/tanaka25a.html) (AISTATS 2025) | Architecture or analytic output parameterization | Exact/structured physical constraints | Boundary law, conservation law, or energy structure | Exact stated constraint or structural consistency; not unique dynamics fidelity |
| Hybrid numerical correction in [Solver-in-the-Loop](https://proceedings.neurips.cc/paper/2020/hash/43e4e6a6f341e00671e123714de019a8-Abstract.html) and [INC](https://papers.nips.cc/paper_files/paper/2025/hash/9facb952d4152f1ce2f21a979ab5b420-Abstract-Conference.html) (NeurIPS 2025) | Post-step or equation-level learned/numerical composition | Discretization or solver defect | Coarse and/or fine numerical solver | Method-specific error analysis plus empirical hybrid rollout evidence |
| Stochastic refinement in [PDE-Refiner](https://proceedings.neurips.cc/paper_files/paper/2023/hash/d529b943af3dba734f8a7d49efcb6d09-Abstract-Conference.html) (NeurIPS 2023) | Iterative denoising/refinement at each step | Frequency-resolved state and uncertainty | Noise-conditioned trajectory data | Empirical long-rollout and uncertainty results; not pathwise contraction |
| Long-time statistical objectives in [Invariant-measure neural operators](https://proceedings.neurips.cc/paper_files/paper/2023/hash/57d7e7e1593ad1ab6818c258fa5654ce-Abstract-Conference.html) (NeurIPS 2023) and [DySLIM](https://proceedings.mlr.press/v235/schiff24b.html) (ICML 2024) | Training loss/regularizer | Invariant measure and attractor statistics | Long trajectories or reference statistics | Statistical-fidelity objective and empirical evidence; not phase tracking |
| Observation feedback in [Semilinear Neural Operators](https://openreview.net/forum?id=ZMv6zKYYUs) (ICLR 2024) | Deployed recursive prediction/correction | Online state estimate | Sparse noisy observations | Data-assimilation/observer contract; not autonomous surrogate stability |
| Memory augmentation in [MemNO](https://openreview.net/forum?id=o9kqa5K3tB) (ICLR 2025) | Augmented state and architecture | Markov closure under partial observation | State history | Expressivity examples and empirical gains; changes the learned state object |
| Geometry and symmetry restrictions in [SFNO](https://proceedings.mlr.press/v202/bonev23a.html) (ICML 2023) and [INO](https://proceedings.mlr.press/v206/liu23f.html) (AISTATS 2023) | Transform/kernel geometry | Equivariance, spherical artifacts, or momentum structure | Geometric/physical prior | Exact or architecture-specific invariance plus empirical accuracy |
| Recursion reduction via direct space--time or continuous time-shift maps, represented by [FNO](https://openreview.net/forum?id=c8P9NQVtmnO) and [Khatri--Rao Neural Operators](https://proceedings.mlr.press/v267/dama25a.html) (ICML 2025) | Temporal parameterization | Fewer recurrent compositions or irregular-time forecasting | Stored trajectories and time coordinates/history | Empirical forecasting; partly outside the one-step corrector core |
| Active perturbation and input denoising in [Beyond Uniform Sampling](https://arxiv.org/abs/2604.13316) (arXiv preprint, 2026) | Data acquisition plus denoising architecture | Adversarial/local robustness | Model-guided queries and synthetic attacks | Empirical robustness on the tested setting; preprint evidence |

### Capacity changes that are not automatically corrective mechanisms

[FFNO](https://openreview.net/forum?id=tmIiMPl4IPa), generic MLP changes, and
attention operators such as [GNOT](https://proceedings.mlr.press/v202/hao23c.html)
can materially alter approximation, optimization, and recurrent behavior. Under
this taxonomy they remain architecture/capacity choices until a separate target
and information contract is identified. This preserves the value of later
component ablations without making the word `corrector` tautological.

### Evaluation-only evidence

Benchmarks and stress tests can reveal where correction is needed without being
correction methods. [APEBench](https://proceedings.neurips.cc/paper_files/paper/2024/hash/d9875ebcf74bccdc5076acab0dbee62c-Abstract-Datasets_and_Benchmarks_Track.html)
also implements training interventions, whereas the 2026 TMLR study
[Diagnosing Failure Modes of Neural Operators](https://openreview.net/forum?id=0S1LWZHQYn)
primarily supplies structured-shift evaluation evidence. These roles should not
be conflated.

## Field Synthesis

The taxonomy supports five conclusions.

1. Existing methods do not all estimate the same object. Recovery, stored-target
   unrolling, displaced-state solver labels, exact projection, invariant-measure
   training, and observation assimilation use different information and target
   different errors.
2. Explicit post-step correction is only one intervention locus. Structural
   restrictions, training-law changes, memory, feedback, hybrid solvers, and
   recursion reduction belong in the field map without being forced into a
   literal `C o F` module.
3. Exact constraint satisfaction is stronger than empirical validity for that
   constraint but weaker than dynamics fidelity. Statistical fidelity is not
   phase accuracy. Online assimilation is not autonomous stability.
4. The literature contains many successful remedies for rollout degradation,
   so the paper should not claim that long-horizon failure is unsolved in the
   generic sense. The sharper open question is which information source
   controls the deployment-relevant response while preserving tangent/path
   fidelity and cost.
5. Within this search, no single controlled study supplies the exact planned
   comparison of clean supervision, detached stored-target exposure,
   solver-free recovery, and dynamics-consistent relabeling while measuring the
   complete deployed map, tube behavior, tangent/path error, structure, and
   equal-cost no-harm. This is a search-bounded gap, not a proof of novelty.

## Experimental Implications

These are design consequences, not execution authorization.

### Minimum decisive information-source comparison

Use one qualified restartable system and one fixed base architecture. Compare:

1. clean one-step supervision;
2. detached pushforward or stored-target supervised unrolling;
3. solver-free recovery from the same perturbation bank; and
4. dynamics-consistent labels `x -> Phi(x)` on model-owned states.

Match parameters, clean data, normalization, optimizer family, presentations,
and evaluation populations as closely as the objectives allow. Report solver
queries and wall-clock cost separately from optimizer updates.

### Required endpoints

- clean one-step error;
- common-bank and on-policy complete deployed defect;
- finite-amplitude tube distance and normal-response estimate;
- tangent/path displacement;
- autonomous state and structure error over time;
- admissibility, boundedness, and finiteness separately;
- family-valid conservation or boundary metrics where defined;
- equal-cost and clean/tangent no-harm controls; and
- seeds and numerical repeatability.

The current P1 path-conditioned score can remain a phenotype or intervention
selector. Because it compares a displaced-input prediction with the archived
clean next state rather than with `Phi(u+eta)`, it cannot substitute for arm 4.

### Decision rule

- If dynamics-consistent labels reproducibly improve the response and rollout
  tradeoff beyond solver-free controls, proceed to one selective cached-label
  correction.
- If stored-target exposure or recovery is equivalent, prefer the simpler
  solver-free mechanism and drop the solver-information contribution.
- If all arms are null, retain the conditional theory and bounded negative
  mechanism result; do not add architectures or PDEs merely to rescue a method.
- If normal stability improves but tangent/path accuracy worsens, the method
  fails the complete deployed-map claim even if rollouts remain bounded.

## Open Risks And Verification Checklist

### Theory risks

- Prove or explicitly assume normal richness for any named architecture before
  making an architecture-specific corollary.
- Replace the smooth manifold with a justified local set model before applying
  the theorem to shock-dominated empirical states.
- Keep exact trace invariance separate from nonzero clean defect; the latter
  enters as forcing in the finite-amplitude recurrence.
- Use the complete normal cocycle. Do not multiply predictor and corrector gains
  without an invariant intermediate set.
- Define the path metric and projection uniqueness before using tangent bounds.
- Add discretization and solver-bias terms before any continuum statement.

### Literature risks

- Re-run the public search near submission and record the new cutoff.
- Verify venue status separately for 2026 preprints.
- Read full papers before attributing theorem strength; abstracts establish
  scope, not every assumption.
- Search adjacent control, reduced-order modeling, differentiable simulation,
  and data-assimilation venues for mechanisms described under different names.
- Phrase the four-arm gap as search-bounded until a formal systematic-review
  protocol and duplicate screening log are complete.

### Empirical risks

- Solver restart must preserve the physical state and boundary/actuator closure.
- Trusted off-state labels must have solver bias materially below the model
  differences being interpreted.
- Perturbation directions must be frozen before target outcomes and cover the
  response object claimed.
- Common-bank assays and on-policy rollouts answer different questions and must
  both be retained.
- A stable projection can hide phase error, smearing, or loss of legitimate
  expanding/high-frequency dynamics.

## T0 Closeout Verdict

The mathematical spine is coherent at the intended conditional scope:

1. exact self-composition exposes an on-policy defect not fixed by clean risk;
2. a constructive normal-rich equivalence class proves clean-trace
   non-identifiability and a minimax need for equivalence-breaking information
   or structure;
3. normal cocycle and finite-amplitude tube bounds give sufficient stability
   conditions for the complete deployed map; and
4. a separate tangent recurrence prevents stable-but-wrong correction from
   being labeled successful.

The next scientific uncertainty is empirical, not definitional: whether
dynamics-consistent off-state information adds value beyond solver-free
exposure and recovery on one qualified restartable PDE. No model, dataset,
solver, remote resource, or sealed population is opened by this closeout.
