# B2-NL Nonlinear ODE Response-Regime Stress Test

Updated: 2026-08-30

Status: owner-authorized, bounded local-CPU ODE extension. This is a new
identity after the completed B1/B2 calibration. It does not amend or invalidate
that packet, authorize any PDE work, or promote an ODE ranking to a PDE claim.

## 1. Question And Role

The completed affine ODE study shows that dynamics relabeling can learn a wide
region of trusted displaced-state response and can roll out well when the
trusted transverse dynamics are contracting. This stress test asks the next
question:

> Does the value of faithful dynamics relabeling change when the trusted
> normal response changes from contractive to neutral to expansive, especially
> under small recurrent normal forcing?

Here, “larger normal dynamics” means a larger clean-manifold transverse
multiplier or cocycle. It does not mean a larger initial displacement. The
experiment separately measures fidelity to a same-forcing trusted path and
robustness relative to the unforced clean path. Neither target is allowed to
stand in for the other.

B2-NL is supporting mechanism calibration. The primary fixed-PCNO PDE case
study remains mandatory and has priority once the owner selects its candidate.

## 2. Trusted Nonlinear System

Use lifted coordinates `(theta, r)` on `S^1 x R` and the continuous ODE

\[
\dot\theta
=1+0.2\sin\theta +(0.5+0.2\cos\theta)r+0.1r^2,
\]

\[
\dot r=k_j(\theta)r-4r^3.
\]

The three registered regimes are

\[
\begin{aligned}
k_{\rm C}(\theta)&=-0.4+0.2\cos\theta,
&\text{uniformly contractive},\\
k_{\rm M}(\theta)&=0.35\cos\theta,
&\text{locally mixed and cycle-neutral},\\
k_{\rm E}(\theta)&=0.4+0.2\cos\theta,
&\text{uniformly expansive near the circle}.
\end{aligned}
\]

The trusted transition `Phi_j` is the time-`h` map with `h=0.2`, evaluated on
the continuous phase lift by deterministic float64 RK4 with 32 substeps. Before
training, the implementation must pass a 32-versus-64-substep convergence
check on the full frozen state bank and an independent clean-manifold response
check against the analytic variational multiplier.

The design is safe and interpretable:

- `r=0` is invariant because the normal vector field contains a factor `r`;
- all regimes have the same nonuniform clean phase dynamics;
- phase remains monotone on `|r| <= 0.5`;
- the cubic term points inward at `|r|=0.5`, so deterministic truth stays in
  the declared safety strip; and
- along the clean circle, the full-cycle transverse log multiplier has the
  sign of the constant part of `k_j`: negative for C, zero for M, and positive
  for E. The cubic term prevents “expansive” from meaning numerical blow-up.

The exact clean-manifold finite-step multiplier is also available from the
variational equation and is used only to qualify the trusted implementation.

## 3. Matched Learned Maps

Every learned map is a generic residual MLP

\[
\Psi_\vartheta(\theta,r)
=(\theta,r)+g_\vartheta(\cos\theta,\sin\theta,r),
\]

with three width-64 `tanh` hidden layers and two lifted-coordinate outputs. It
receives no analytic response coefficient, regime label, projection, or
closed-form solution.

### 3.1 Frozen data and optimization

| Item | Contract |
| --- | --- |
| Clean phases | 64 uniformly spaced phases. Evaluation phases use the disjoint half-grid. |
| Displaced inputs | The identical four radii `{-0.15, -0.05, 0.05, 0.15}` at every clean phase. |
| Mixture | 50% clean anchors and 50% displaced rows. Clean rows are duplicated only to match row counts. |
| Held-out in-tube bank | Half-grid phases at radii `{-0.10, -0.025, 0.025, 0.10}`. |
| Outer diagnostic bank | The same phases at radii `{-0.20, 0.20}`, reported separately with no generalization claim. |
| Objective | Equal-coordinate lifted MSE. |
| Optimizer | Adam, learning rate `2e-3`, batch size 128, 3000 fixed updates. |
| Seeds | Paired seeds `17`, `29`, and `43`. |
| Fairness | Identical initial parameter bytes and minibatch indices for every arm within a seed. |
| Checkpoint | Terminal update only; no validation or rollout selection. |

The learned checkpoints are:

- `CLEAN`: clean `(theta,0) -> Phi(theta,0)`;
- `RECOVERY`: clean rows plus displaced `(theta,r) -> Phi(theta,0)`; and
- `DYN-C`, `DYN-M`, `DYN-E`: the identical clean and displaced inputs with
  displaced target `Phi_j(theta,r)` for the corresponding regime.

Because clean and recovery targets are regime-independent, exactly one CLEAN
and one RECOVERY checkpoint are trained per seed and their bytes are reused in
all three regimes. Only the three dynamics-relabeling checkpoints differ. This
gives five trained checkpoints per seed. Any regime-dependent CLEAN or
RECOVERY trajectory under identical recurrent forcing is an implementation
failure.

## 4. Recurrent Normal-Forcing Assay

Use the complete learned map followed by paired bounded normal forcing:

\[
\widehat x_{n+1}=\Psi(\widehat x_n)+(0,\xi_{n+1}),
\qquad
\xi_n=\sigma s_n,
\qquad
s_n\in\{-1,+1\}.
\]

The forcing is applied after the complete map so that every mechanism must
handle the new displacement on the next step. It represents controlled
recurrent process/model forcing, not observational noise, label noise, a
physical stochastic PDE, or an OOD detector.

Frozen contract:

- `sigma in {0, 1e-3, 5e-3}`;
- 192 recurrent steps;
- 32 held-out initial phases;
- 16 fixed Rademacher sequences per phase, shared by every arm, regime, and
  seed;
- response tube `|r| <= 0.15`; and
- safety envelope `|r| <= 0.5`.

No clipping, defect trigger, online solver query, or post-exit rescue is used.

Every learned rollout is compared with two references:

1. the unforced clean path `u_{n+1}=Phi_j(u_n)`, measuring clean-path
   retention and accuracy; and
2. the same-forcing trusted path
   `x^xi_{n+1}=Phi_j(x^xi_n)+(0,xi_{n+1})`, measuring faithful response to
   an independently evolved trusted trajectory under the same forcing.

The second quantity is called **same-forcing trusted-path fidelity**. It is not
the solver defect `Psi(x)-Phi(x)` evaluated at each learned reached state.

The exact trusted map under the same forcing is retained as an oracle dynamics
control. It reveals inherent sensitivity of the true normal dynamics without
mixing it with neural approximation error.

## 5. Frozen Diagnostics And Endpoints

Diagnostics computed before interpreting long-horizon ordering:

- held-out clean one-step lifted and chordal error;
- fresh normal forcing on clean states;
- phase-dependent finite-difference normal gain and normal-to-phase response;
- trusted-flow defect and clean-return error on the common in-tube bank;
- the same quantities on the separately labelled outer bank;
- 32-step clean-reference local-secant log-gain; and
- exact solver convergence, clean invariance, and analytic-response closure.

Long-horizon endpoints:

- first response-tube exit and survival;
- RMS and maximum normal distance;
- unwrapped phase and lifted error relative to the clean path;
- lifted error relative to the same-forcing trusted path;
- safety-envelope exit and nonfinite rates; and
- the paired regime-by-method interaction.

Initial phase and forcing sequence are the trajectory sampling units. Time
steps are not treated as independent. Post-tube-exit behavior is descriptive;
the primary mechanism comparison is also reported up to first exit.

## 6. Registered Predictions And Falsifiers

### P1. Target realization

On every paired seed and the held-out in-tube bank, DYN-j should have lower
trusted-response defect than RECOVERY, while RECOVERY should have lower
clean-return error and smaller normal response magnitude than DYN-j.

Relative ordering is not enough to qualify DYN. For every seed and regime,
the DYN in-tube trusted-flow RMS defect divided by the trusted displaced-
response RMS must be at most `0.15`, and its clean lifted one-step RMS error
must be at most `2e-3`. Within each seed, the largest DYN clean error across
the three regimes may be at most twice the smallest. Failure makes rollout
comparisons descriptive even if DYN remains less wrong than RECOVERY.

Falsifier: either signed target effect fails. The optimization/representation
link is then incomplete and the noisy rollout cannot validate the intended
mechanism.

### P2. Response-regime ordering

The response-qualified learned DYN maps should reproduce the trusted ordering

\[
\lambda_{\rm DYN-C}<\lambda_{\rm DYN-M}<\lambda_{\rm DYN-E},
\]

with negative, near-zero, and positive 32-step clean-reference local-secant
log-gain matching the three trusted regimes. This is an offline product of
local response secants evaluated along the trusted clean path, not the
Jacobian cocycle along the learned deployed trajectory. The trusted 32-step
quantity must pass the same ordering and thresholds before training.

For the decision rule, negative means a seedwise phase-mean log-gain below
`-0.5`, near-zero means absolute phase-mean log-gain at most `0.5`, and
positive means a phase-mean log-gain above `0.5`.

Falsifier: the ordering or signs fail after the trusted solver checks pass.

### P3. Robustness interaction

Under nonzero recurrent forcing, clean-path tube retention for DYN should
degrade from C to M to E. RECOVERY should be materially less regime-sensitive
and should outperform DYN-E on clean-path tube survival or lifted error.

The primary interaction is evaluated at `sigma=5e-3` using only first-exit
quantities. It passes if DYN-C, DYN-M, and DYN-E have nonincreasing mean tube-
residence fractions with a C-to-E difference of at least `0.20`. RECOVERY
beats DYN-E if its mean tube-residence fraction is higher by at least `0.20`.
Full post-exit error AUC is descriptive and cannot validate this prediction.

Falsifier: a response-qualified DYN-E is no more sensitive than DYN-C, or no
regime-by-method interaction is observed.

### P4. Dual-target tradeoff

Before the first exit of DYN-E, RECOVERY, or the same-forcing trusted path,
DYN-E should be closer than RECOVERY to that trusted path even when RECOVERY
is closer to the unforced clean path. The median common prefix must contain at
least 16 recurrent steps; otherwise this prediction is insufficiently covered
and is reported as uninterpretable.

Falsifier: RECOVERY matches both targets without learning the trusted response,
or DYN-E departs from the same-forcing trusted path before its claimed
response-regime effect.

### P5. Contractive control

Response-qualified DYN-C should remain inside the response tube under the mild
registered forcing for most trajectories and should retain its trusted-path
fidelity advantage. Relabeling is not predicted to fail universally.

“Most” means at least `0.80` survival at each of the two nonzero forcing
levels, aggregated over the frozen phases, forcing sequences, and paired model
seeds. At each level, DYN-C must also retain lower common-prefix same-forcing
trusted-path error than RECOVERY.

Falsifier: DYN-C has substantial early tube failure at both nonzero forcing
levels despite matching the trusted in-tube response.

These are directional mechanism predictions, not a universal best-method
ranking. A negative outcome is retained and changes the manuscript
interpretation rather than triggering a parameter or architecture rescue.

## 7. Implementation, Audit, And Packet Contract

New long-lived files are limited to:

- `scripts/time_dependent_no/run_corrective_ode_nonlinear_stress.py`, the
  bounded run/verify entry point; and
- `tests/time_dependent_no/test_corrective_ode_nonlinear_stress.py`, its
  analytical, target, recurrence, parity, and provenance tests.

The new script may import the frozen residual-MLP implementation from the
completed affine runner, but must not modify that runner or its tests. The new
manifest therefore binds this preregistration, both new files, and every
imported source dependency. It also records the independently verified B1/B2
and landscape manifest hashes as compatibility anchors; it is not a derived
rewrite of either packet.

The packet contains the frozen config and predictions, solver qualification,
training and response records, recurrent curves, aggregate endpoints,
trajectory-unit exits, paired common-prefix errors, checkpoints, summary
figure, summary, manifest, and closeout. Verification checks every output and
current source hash independently. A changed contract receives a new output
identity.

Before the canonical run, an agent other than the implementer must audit:

1. ODE, regime, RK4, and clean-invariance semantics;
2. identical inputs, paired initialization/batches, and target construction;
3. shared CLEAN/RECOVERY checkpoint reuse;
4. process-noise placement and sharing;
5. clean-path versus same-forcing-truth references;
6. tube/safety/first-exit logic; and
7. packet/source mismatch refusal.

The verdict must be `AUDIT_PASS`. Unit tests alone do not authorize the
canonical scientific run.

## 8. Scope Boundary

Included: one nonlinear ODE, three normal-response regimes, one neural
architecture, three paired seeds, five learned checkpoints per seed, three
fixed forcing levels, and exact trusted controls.

Excluded: label noise, architecture sweeps, learned or buffer hybrids,
additional manifolds, PDE analogues, online detection, adaptive correction,
post-outcome tuning, and any claim that the outer bank measures general OOD
performance.
