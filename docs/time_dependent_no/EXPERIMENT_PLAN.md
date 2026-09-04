# Experiment Plan: Prospective Diagnosis And Corrective Mechanisms

Updated: 2026-09-05

Status: active claim-driven design. B1/B2, the first NACA R0 screen, the
five-arm successor, and the corrective extension are complete at their
registered open-development scopes. The extension remains frozen in
[B3B4_NACA_CORRECTIVE_EXTENSION_PREREGISTRATION.md](B3B4_NACA_CORRECTIVE_EXTENSION_PREREGISTRATION.md)
and its JSON contract. The bounded Supersonic-Bump B5 transfer is selected for
local Stage-0A/B work under its own preregistration; no remote resource is selected.
Prospective and sealed outcomes remain unopened.

## 1. Questions

The empirical programme asks:

1. Can learned maps with similar or reversed clean one-step rankings have
   predictably different rollouts because their forcing and response differ?
2. Can frozen offline diagnostics predict which corrective mechanism will help
   a fixed PCNO on one qualified complex PDE?

The first question supports the boundary of C1. The second is C2. ODEs provide
controlled calibration; the primary PDE supplies the decisive evidence.

## 2. Experimental Boundary

Keep these objects distinct:

| Object | Fitted or measured with | Use |
| --- | --- | --- |
| Reference geometry | Clean training/reference states only, conditioned on time and problem parameters. | Model-independent support/drift proxy. |
| Solver-relative response | Qualified solver calls at frozen displaced states. | Offline defect, trusted response, and mechanism validation. |
| Deployed transition | Only declared inference inputs. | Online rollout. No core online solver query or OOD trigger. |

`ID` refers to the initial-condition and clean-data law. Solver-relative defect
does not define the tube or OOD.

Any solver call at deployment creates a separately costed solver-feedback
baseline and is not silently compared with solver-free methods.

## 3. Blocks And Priority

| Block | Role | Priority |
| --- | --- | --- |
| **R0: primary-PDE readiness screen** | Select one solver-backed, restartable problem with structured clean support and an honest clean-H1/severe-rollout PCNO discrepancy. | Immediate prerequisite. |
| **B1: exact ODE laboratory** | Validate identities, response measurements, ranking reversals, and target semantics. | Required supporting evidence. |
| **B2: learned ODE sandbox** | Check that matched learners instantiate the predicted recovery/relabeling/explicit mechanism changes. | Required but compact. |
| **B3: primary-PDE diagnosis and prediction freeze** | Diagnose the selected PDE and freeze interventions, mediators, rankings, and falsifiers before designated outcomes. | Main-paper prerequisite. |
| **B4: primary-PDE intervention and reveal** | Test representative embedded and operational mechanisms under the B3 freeze. | Main C2 evidence. |
| **B5: reduced second-PDE contrast** | Test solver-free correction beyond NACA's special linear low-rank geometry. | Active local Stage 0A/B; remote execution unselected. |

No architecture, PDE, diagnostic, or method slot is added after designated B4
outcomes are opened merely to rescue a claim.

## 4. R0: Primary-PDE Readiness Screen

### 4.1 Hard gates

The hero-route candidate proceeds only if all gates pass. The separately
authorized NACA successor is not a retroactive R0 pass; it proceeds only under
its narrower successor contract:

1. **Trusted transition.** A numerically qualified solver can generate clean
   data at adequate throughput in the exact model representation, with
   boundary/forcing/state closure. Arbitrary-state advancement is a separate
   prerequisite for methods or diagnostics that actually consume it.
2. **Immutable populations.** Train, development, prospective, and sealed/test
   role windows are content-disjoint and hash-bound. A single-attractor route
   must disclose common-trajectory dependence and use deterministic
   finite-phase summaries rather than IID confidence intervals.
3. **Structured clean support.** Train-only diagnostics show a reproducible,
   interpretable concentration near a low-dimensional or otherwise structured
   reference family. Use reconstruction curves, effective dimension,
   neighborhood stability, and physical features; do not use model defect to
   establish this gate.
4. **Honest baseline discordance.** A fixed, adequately trained vanilla PCNO
   has acceptable clean H1 error but reproducible severe long-rollout
   degradation from ID starts after a nontrivial window.
5. **Failure validity.** Normalization, recurrent feedback, timestep,
   discretization, projection, boundaries, solver bias, undertraining, and
   numerical inadmissibility do not trivially explain the phenomenon.
6. **Diagnostic ordering.** On development data, reference drift or an adverse
   response profile can be measured before or during the onset of failure.
7. **Resource fit.** Data generation, three-seed training, response queries,
   operational correction, open-development evaluation, prospective-ready
   freeze, permitted figures/animations, and manuscript integration fit the
   Friday `2026-09-04` envelope. Protected reveals retain separate gates.

Literal blow-up is desirable for demonstration but not mandatory. A stable but
severely wrong rollout is admissible and protects the paper from depending on
a pathological exploding baseline.

For the primary correction-necessity case, severe error alone is not enough:
the baseline must expose a defensible transverse-retention or recovery problem
in addition to nontrivial on-reference dynamics. A problem that stays near the
reference support while accumulating phase or transport error is retained as a
tangential/no-harm control rather than promoted as the hero case.

Candidate screening may use open development outcomes. The selection rule and
its use as a hard-case demonstration must be disclosed. C2 is tested only on a
disjoint frozen prospective population.

### 4.2 Completed first screen and candidate order

| Candidate | Scientific role | Verified state | Next gate |
| --- | --- | --- | --- |
| **SU2 Unsteady NACA0012** | Completed negative R0 delayed-failure screen; primary case for the distinct corrective successor and extension. | R0, the five-arm parent, and `B3B4_NACA_CM_EXT_20260902A` all completed and passed their open-development packet audits. R0 retains its negative verdict; the later identities answer a narrower corrective-mechanism question. | Preserve every identity unchanged. Derived presentation and manuscript integration are allowed; any prospective or sealed reveal is a separate decision. |
| **Turbulent square cylinder** | Unselected candidate after NACA. | Public SU2 material exists, but the production configuration/provenance route is not yet sufficiently pinned. | Activate only if the owner/mentor selects it and a reproducible production solver contract is frozen. |
| **Laminar von Karman cylinder** | Cheap solver/control candidate. | Official unsteady SU2 tutorial route. | Use only after owner/mentor selection; do not assume it will expose delayed transverse drift. |
| **Euler shock-vortex / supersonic bump** | Shock-vortex remains a control; bump is the selected B5 transfer/no-harm study. | Bump has a new solver-free contract; neither route has a qualified arbitrary-state trusted restart. | Run only B5's local Stage-0A/B scope until the required amendments and remote resource are separately selected. |

The literature screen found no clearly better overall first benchmark, so NACA
received an honest attempt. Its negative result closes that exact delayed-
failure route without weakened thresholds, post-outcome horizon changes, or
manufactured model drift. The later successor changes the question and
identity, not the R0 verdict. More ambitious buffet, pitching/gust, FSI, or CHT
cases remain outside the Friday critical path.

## 5. B1: Exact ODE Laboratory

Use tangent-normal coordinates `(theta, r)` with reference set `r = 0`:

\[
\dot r=\kappa r,
\qquad
\dot\theta=\omega+c r.
\]

For step size `h`, the exact map is

\[
r^+=a_*r,
\qquad
\theta^+=\theta+\omega h+b_*r,
\]

with

\[
a_*=\exp(\kappa h),
\qquad
b_*=
\begin{cases}
c\,\operatorname{expm1}(\kappa h)/\kappa,&\kappa\neq0,\\
ch,&\kappa=0.
\end{cases}
\]

The exact state lies on `S^1 x R`. The embedding
`(cos(theta), sin(theta), r)` is used only for chordal error and proxy-geometry
calibration; it does not add another exact normal direction.

Use the transparent learned-map family

\[
\widehat r^+=\epsilon_N+a\widehat r,
\qquad
\widehat\theta^+=\widehat\theta+\omega h+\epsilon_T+b\widehat r.
\]

For a clean start `r_0=0`, define

\[
S_n(a)=\frac{1-a^n}{1-a},
\qquad
T_n(a)=\frac{n}{1-a}-\frac{1-a^n}{(1-a)^2},
\]

with `S_n(1)=n` and `T_n(1)=n(n-1)/2`. The registered recurrences are

\[
\widehat r_n=\epsilon_N S_n(a),
\qquad
\widehat\theta_n-\theta_n=n\epsilon_T+b\epsilon_NT_n(a).
\]

Clean one-step error therefore depends on `(epsilon_T, epsilon_N)` and is
independent of the displaced-input response coefficients `(a, b)`; it is not
called a pure tangent-error measurement.

### 5.1 Frozen exact scenarios

All analytical work uses float64 and an unwrapped continuous phase lift.
Lifted-coordinate rollout error is `sqrt(phase_error^2 + r_error^2)`; chordal
embedded state error is reported separately.

| Scenario | Frozen contract | Required observation |
| --- | --- | --- |
| Ranking reversal | Trusted `a_*=0.8`, `b_*=0.5`; A: `epsilon_N=1e-3`, `a=1.15`, `b=0.5`; B: `epsilon_N=5e-3`, `a=b=0`; both `epsilon_T=0`, `N=30`. | A has smaller clean one-step error and larger final lifted-coordinate rollout error. |
| Retention without accuracy | `epsilon_N=0`, `epsilon_T=0.02`, `a=b=0`, `N=40`. | `|r_n|=0` for every step while unwrapped phase error reaches `0.8`. |
| Bounded false attractor | `epsilon_N=0.002`, `a=0.98`, `b=epsilon_T=0`, `N=300`, tube radius `R=0.05`. | The rollout remains finite, exits the tube, and approaches `r_infty=0.1`. |
| Recovery/relabeling crossover | `h=0.2`, `c=2`, `N=80`, `a_*` swept on `[0.05,0.98]`, and exact `b_*`. RECOVERY uses `(epsilon_T,epsilon_N,a,b)=(0.002,0.001,0.05,0)`; DYN-RELABEL uses `(0.0005,0.001,a_*,b_*)`. | Relabeling wins when its smaller clean-path bias dominates; recovery wins for a registered substantial-response regime despite larger trusted-response defect. This is a conditional bias--response tradeoff, not an inherent ordering. |
| Fixed-interface correction | Raw predictor `epsilon_N=0.002`, `a=1.2`, `b=0.5`; compare `C_rho(theta,r)=(theta,rho r)` for `rho in {1,0.5,0}`. | Complete-map normal forcing and gain equal `rho epsilon_N` and `rho a`; normal-to-tangent coupling remains `b`. Identity parity and corrected-state recurrence close exactly. |

Required endpoints are exact forcing, the complete-map response block
`[[1,b],[0,a]]`, trusted-response defect, tube exit/survival, unwrapped phase
and lifted-coordinate path error, chordal error, fixed-point location, and predicted
versus observed ranking. kNN, local-PCA residual, and sampled-convex-hull
membership are appendix-only proxy calibrations against exact geometry.

Gate: every registered inequality holds without parameter changes, and direct
iteration agrees with the closed recurrence at a scale- and horizon-aware
multiple of float64 machine precision. Failure revises the framework before
PDE interpretation.

## 6. B2: Learned ODE Sandbox

Use the substantial-response flow `omega h=0.25`, `a_*=0.8`, `b_*=0.5`. The
learner receives `(cos(theta), sin(theta), r)` and predicts lifted-coordinate residuals
`(Delta theta, Delta r)`; it never predicts or projects a three-dimensional
ambient state.

### 6.1 Frozen training contract

| Item | Value |
| --- | --- |
| Architecture | Two-hidden-layer width-32 `tanh` residual MLP, 3 inputs and 2 residual outputs. |
| Clean phases | 256 uniformly spaced phases; held-out phases use the half-grid offset. |
| Displacements | One deterministic balanced displacement per phase in `[-0.2,0.2]`; the exact same rows feed RECOVERY and DYN-RELABEL. |
| Mixture | 50% clean anchors and 50% displaced rows for response arms; CLEAN duplicates clean rows to match row and update counts. |
| Objective | Equal-weight MSE on lifted phase and normal residuals. |
| Optimizer | Adam, learning rate `3e-3`, batch size 128, 1500 fixed updates. |
| Seeds | Paired seeds `17`, `29`, and `43`; identical initial parameter bytes and batch-index order across training arms within a seed. |
| Checkpoint | Terminal update only; no validation or rollout selection. |
| Query bank | Held-out phases at `r in {0, +/-0.05, +/-0.1, +/-0.2}`. |
| Rollouts | 160-step ID clean starts plus a separately labelled `r=0.1` normal-impulse assay; tube radius `0.1`. |

The three trained arms are:

- `CLEAN`: clean `u` to `Phi(u)`;
- `RECOVERY`: clean rows plus displaced `x=u+eta` to `Phi(u)`; and
- `DYN-RELABEL`: the identical clean and displaced rows, with displaced target
  `Phi(x)`.

The fourth deployed arm is `CLEAN+C_0`, where the oracle ODE corrector
`C_0(theta,r)=(theta,0)` is attached after the same CLEAN predictor and the
corrected state is fed back. `C_1` is the identity ablation. No hybrid,
pushforward, architecture sweep, or PDE analogue is included.

Therefore no genuine pushforward, multistep-loss, or model-prefix-exposure arm
was run in the ODE studies. `DYN-RELABEL` is trusted relabeling on prescribed
displaced inputs, not pushforward training. The completed PDE extension supplies
the source-faithful MP-PDE and separately named curriculum/EMA exposure
contrasts; the closed `DETACHED_PUSHFORWARD` arm remains a one-prefix stress
test rather than a general verdict on multi-step training.

### 6.2 Frozen predictions and decision rule

| Prediction | Falsifier |
| --- | --- |
| RECOVERY moves `(a,b)` toward `(0,0)`. | It is not closer to `(0,0)` than DYN-RELABEL on every paired seed. |
| DYN-RELABEL moves `(a,b)` toward `(a_*,b_*)` and lowers trusted response defect. | Its trusted response defect is not below RECOVERY on every paired seed. |
| `CLEAN+C_0` has deployed normal gain zero while retaining the raw predictor's phase coupling. | Raw/deployed separation, identity parity, or corrected feedback fails. |
| In the impulse assay, DYN-RELABEL best tracks the displaced trusted path, while recovery mechanisms return more strongly toward the clean path. | The signed paired response effects do not appear. |

Autonomous ID rollout separation is reported but is not a pass condition. Dense
coverage may make every arm accurate; data coverage will not be reduced to
manufacture drift. Clean error, trusted-response defect, tube distance,
unwrapped phase error, lifted-coordinate/chordal rollout error, and raw/deployed cost
remain separate. B2 supports target-mechanism realization only and does not
establish an ODE-to-PDE ranking transfer.

## 7. B3: Primary-PDE Diagnosis And Freeze

For the separately authorized NACA successor--or for a future candidate that
passes its own R0--use one fixed PCNO backbone and open development populations
to:

1. reproduce and audit the clean-H1/rollout discrepancy;
2. fit one train-only reference-geometry instrument shared by every model;
3. build a common bank of qualified clean anchors and physically admissible
   tangent, normal, mixed, and model-prefix perturbations where meaningful;
4. measure complete-map forcing, finite-amplitude response, solver-relative
   response defect, coupling, drift, path/phase, and physical failure channels;
5. compare clean proximity with solver-relative fidelity without defining either by the
   other; and
6. freeze the smallest nonredundant B4 representatives and signed predictions.

### 7.1 Offline response quantities

At a fixed predictor-corrector interface, estimate predictor normal gain `q`,
corrector return gain `rho`, additive forcing, and the complete deployed gain.
Use

\[
r_{n+1}\leq \rho q r_n+\rho\epsilon+\delta
\]

only as a qualified local diagnostic. Tube closure requires

\[
(\rho q)R+\rho\epsilon+\delta\leq R.
\]

For embedded mechanisms, report complete-map gain rather than inventing `q`
and `rho`. Always compare learned response with trusted response. Suppressing
normal motion can improve retention while increasing normal-response defect.

If local dimension or tangent estimates are unstable across reasonable
neighborhoods, drop tangent/normal terminology and use finite-amplitude
state/feature responses.

### 7.2 B3 freeze packet

Before B4, bind:

- selected PDE, solver, state, population, and PCNO identities;
- common bank and geometry/feature definitions;
- retained intervention implementations and fairness views;
- expected mediator changes and main failure mode of each arm;
- predicted clean-H1, long-horizon, recovery/relabeling, and overall rankings;
- horizons, structure/validity endpoints, statistical rules, tie handling,
  no-harm gates, and cost accounting; and
- source, data, model, evaluator, and packet hashes.

No B4 run begins without this packet.

## 8. B4: Representative PDE Intervention Study

The closed five-arm NACA successor is frozen and must not expand after outcome:

| Role | Required comparison |
| --- | --- |
| Clean control | `CLEAN`: unmodified fixed-PCNO deployment. |
| Isotropic embedded recovery | `IID_RECOVERY`: IID history-pair corruption with the stored clean successor target. |
| Error-directed embedded recovery | `ERROR_SUBSPACE_RECOVERY`: matched-energy rank-16 parent-error-subspace corruption with the same recovery target. |
| Model-prefix exposure | `DETACHED_PUSHFORWARD`: one detached generated current-state proposal with stored clean-future target. |
| Deterministic operational correction | `PATH_PROJECTION`: train-only path projection attached to each `CLEAN` predictor with corrected-state feedback. |

Dynamics relabeling, structural arms, learned refiners, and hybrids remain in
the field taxonomy but are not missing arms from that frozen parent. Three
mechanism gaps are now tested under a distinct identity rather than appended to
the old result.

When exact budget matching is impossible, report equal-information and
equal-cost views rather than claiming perfect parity. Operational methods use
a fixed correction schedule, not an online OOD trigger. Report raw predictor
`F` and deployed `C o F` separately.

Every arm needs a code-to-intent audit covering input law, target, information,
composition order, recurrent feedback state, normalization, timestep,
boundary/state semantics, identity ablation, leakage, expected signature, and
cost.

### 8.1 Predeclared mechanism expectations

| Frozen arm | Expected mediator | Main falsifier or risk |
| --- | --- | --- |
| `IID_RECOVERY` | Lower displaced-input gain or forcing and longer reference residence. | Legitimate path response is suppressed. |
| `ERROR_SUBSPACE_RECOVERY` | Stronger correction along directions actually injected by clean parents at matched expected energy. | The fitted error basis is irrelevant, unstable, or over-specialized. |
| `DETACHED_PUSHFORWARD` | Lower stored-future error on model-prefix inputs and lower projected-path discrepancy. | No generic contraction; partial-history exposure is insufficient or horizon-specific. |
| `PATH_PROJECTION` | Lowest post-correction projection residual with immediate corrected-state feedback. | Path snapping, quantization, or stable wrong path dynamics. |

A rollout gain without the predicted mediator change rejects that mechanism
explanation. A mediator change without rollout benefit indicates poor coverage
or an irrelevant response object.

### 8.2 Verified development outcome

The development packet is bound by final-manifest SHA256
`09e3ae7e5896039e2230073c1ead5e5f9808c0cc9b47d0b15ff337b26b3e0ed1`.
All evaluated rollouts are finite. `PATH_PROJECTION` improves both primary
rollout metrics in all 24 paired anchor-seed comparisons and improves pressure
and graph-Dirichlet diagnostics; its zero post-correction path residual is a
construction check, not empirical support. `IID_RECOVERY` helps relative to
`CLEAN`. `ERROR_SUBSPACE_RECOVERY` improves `CLEAN` but not `IID_RECOVERY`.
The registered `DETACHED_PUSHFORWARD` mechanism prediction is falsified and its
rollout effect is seed-unstable.

This development result measures learned-map and deployed-corrector behavior.
It does not identify visible flow features as physical off-manifold drift and
does not measure trusted SU2 response from arbitrary displaced states.
Prospective and sealed populations remain unopened.

### 8.3 Frozen and completed open-development extension

`B3B4_NACA_CM_EXT_20260902A` preserves the parent and reuses its fixed PDE,
complete BDF2 state, PCNO width/modes, roles, seeds, horizons, optimizer family,
and evaluator semantics. Its nonredundant contrasts are:

| Contrast | Frozen role |
| --- | --- |
| `MP_PDE_PUSHFORWARD_M01` | Literal released MP-PDE: clean epoch 1, then batchwise `m` uniform on `{0,1}`, online stopped-gradient prefixes, terminal clean-future loss. |
| `CURRICULUM_EMA_PUSHFORWARD_K13` | Separate ramped EMA-prefix protocol with exposed depth uniform on `{1,2,3}`; `CLEAN_EMA` isolates EMA deployment. |
| `PAIRED_RECOVERY` vs `DYNAMICS_RELABEL` | Identical frozen displaced BDF2 bank and clean/displaced weighting; only the clean-future versus one-step SU2 target changes. |
| `PCNO_PDEREFINER_K3_VPRED` | Shared-PCNO four-level DDPM velocity prediction with four model calls per physical step and EMA deployment. |

Before the paired bank exists, an eleven-call train-only SU2 pilot must pass
restart round-trip, clean replay, realized-scale, convergence, admissibility,
repeatability, auxiliary-column, and response-separation gates. Any failed
case stops the bank; no case is silently dropped or replaced. A full pass
unlocks exactly one antithetic Gaussian direction for each of 238 train
centers, or 476 paired inputs. Focused CPU tests and an independent code-to-
intent audit precede scientific execution. Development evaluation occurs only
after training and uses one common extension evaluator. Solver labels are
offline training/diagnostic information; online solver calls and defect
triggers are forbidden. Prospective and sealed roles remain closed.

All preregistered open-role stages completed. Corrected pilot R1 passed every
gate; the 476-input paired bank, 18 arm/seed training packets, and common
development evaluator were independently verified. Attempt E is the canonical
evaluation packet under final-manifest SHA256
`15475da78b4182754fe959eadc175434ffca18ef1558a45ea06cbf9c93ca7370`.

### 8.4 Verified extension outcome

Path projection ranks first and PDE-Refiner second on both primary rollout
metrics at every seed. Paired recovery improves or ties inherited clean on all
24 anchor--seed units and is the safest one-call learned intervention. Dynamics
relabeling is seed-dependent. Literal MP-PDE improves only 8 of 24 units and
worsens 16; curriculum exposure helps seeds 17 and 29 but fails severely at
seed 43. These outcomes do not reject pushforward or multistep training as a
family.

The registered recovery--relabel target-semantics prediction was not directly
assayed. The common evaluator measures each model's own one-prefix response; it
does not compare trained models on identical frozen-bank inputs against both
the clean future and the stored SU2 continuation. Relabeling also incurs large
clean-path and optimization harm. A trusted-response claim therefore requires
a new frozen evaluation-only assay, not reinterpretation of the rollout table.
Likewise, MP-PDE gives mixed early-gradient changes and the curriculum retains
large gradient spikes; their realized prefix displacements are about 3--11
times the recovery-noise scale. PDE-Refiner's rollout gain is real, but its
reverse steps cease to be a monotone projection at later reached states.

## 9. Reveal, Statistics, And Reporting

Before the designated reveal, freeze:

- source, split/population, model/checkpoint, normalizer, feature map, query
  bank, evaluator, corrector, schedule, and prediction hashes;
- primary horizons and tie rule;
- Kendall/Spearman, all-pair, and H1-discordant-pair analyses;
- trajectory/seed-clustered bootstrap procedure; and
- structure, validity, no-harm, solver-information, and compute-cost endpoints.

Exact numerical success thresholds are calibrated after PDE/evaluator
selection and frozen in B3. This master plan does not invent universal values.

Use three paired seeds for stochastic learned comparisons when budget allows;
the trajectory/initial condition is the primary sampling unit. Do not treat
time frames as independent. Report all registered failures, ties, negative
results, and incomplete attempts.

D094 remains retrospective motivation and cannot enter the prospective freeze.

## 10. B5: Selected Supersonic-Bump Transfer

`B5_BUMP_SOLVER_FREE_TRANSFER_20260905A` is a bounded secondary
open-development study, not prospective C2. Its source of truth is
[B5_BUMP_SOLVER_FREE_TRANSFER_PREREGISTRATION.md](B5_BUMP_SOLVER_FREE_TRANSFER_PREREGISTRATION.md)
and the accompanying JSON contract. Authorized Stage 0A/B verifies
identity/leakage and measures train-only native-mesh POD reducibility. Stage 0C
clean-phenotype replay remains blocked pending its artifact-identity amendment
and resource. Raw global PCA is not a valid bump arm because trajectories
use different native meshes; per-trajectory POD is diagnostic only.

If the frozen Stage-0 gates pass, Stage 1 compares exactly `CLEAN`,
`IID_RECOVERY`, `CURRICULUM_EMA_PREFIX_K13`, and
`PREFIX_ERROR_CORRECTOR_K13`. A bump `PCNO_PDEREFINER_K3_VPRED` port is
conditional on the registered promotion decision. Local Stage-0A/B and
synthetic implementation are authorized. Stage 0C and Stage 1 require their
registered amendments; any dataset-scale or remote launch also needs a named
resource decision and receipt. The historical test remains sealed.

## 11. Execution Order

1. Preserve the complete NACA R0 packet and exact negative verdict.
2. Implement only the arms and diagnostics frozen in the closed five-arm NACA
   successor.
3. Pass focused tests, an independent code-to-intent audit, and a
   full-resolution resource smoke.
4. Train paired seeds and evaluate only train/development roles.
5. Preserve the closed open-development packet and freeze the prospective
   evaluator and predictions.
6. Request a separate owner decision before the prospective reveal.
7. Integrate verified B1/B2/B3/B4 evidence into the manuscript.
8. Implement and audit only the local B5 Stage-0A/B identity/reducibility path;
   stop before remote or dataset-scale execution until its resource is selected.

Extension order:

1. preserve every parent source and artifact under its old identity;
2. implement the registered schedules, targets, EMA, refinement recurrence,
   restart writer, cost accounting, and new packet identity;
3. pass focused CPU tests and independent code-to-intent audit;
4. run the eleven-call train-only SU2 pilot and stop on any failed gate;
5. after a full pilot pass, generate and verify the 476-input paired bank;
6. pass full-resolution resource smokes, train the registered seeds, and run
   the common development evaluator; and
7. close the result packet before revising empirical manuscript claims.

Current state: parent steps 1--5 and manuscript integration are complete. The
successor's immutable replay, historical v2 visualization, and final v3
presentation packet remain valid provenance. Exact mounted-checkout WSL source
and test verification passes for that parent. Extension steps 1--7 and the
subsequent presentation/manuscript pass are complete on development roles. Its
corrected local derivative is bound to the remote Attempt-E visualization
receipt and exact replay; static figures use exact evaluator snapshots, while
the three seed animations are independent qualitative rerolls. Every
protected-access flag is false. Parent step 6 still requires a separate owner
decision. Prospective and sealed populations are unopened, and prospective C2
generalization is unclaimed.

If the successor cannot distinguish response, drift, path, and representation
alternatives, report that falsification and use the fallback candidate route;
do not rescue the claim by adding arms after outcome.

## 12. What Is Not Evidence

- an unexecuted plan or passing synthetic unit test;
- an incomplete process without a result packet;
- a packet whose hashes do not verify;
- a validation result relabelled as test;
- a retrospective score described as prospective;
- an on-rollout defect measured after failure presented as prediction; or
- a reference-proximity proxy called the true data manifold.
