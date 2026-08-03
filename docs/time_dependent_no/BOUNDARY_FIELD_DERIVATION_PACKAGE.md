# Boundary-Field Derivation Package

Updated: 2026-08-03

## Target

Define and analyze a resolution-independent boundary-input contract for the
maintained PCNO Euler model. The immediate target is not a theorem that one
boundary encoding is optimal. It is a coherent theory line that:

1. distinguishes a mesh-local categorical band from a continuum boundary
   extension;
2. predicts how each representation enters the pointwise, integral, and
   differential PCNO pathways;
3. states the assumptions needed for resolution consistency; and
4. yields falsifiable mechanisms for teacher-forced and autonomous rollouts.

## Status

COHERENT AFTER REFRAMING / EXTRA ASSUMPTION

The resolution-consistency diagnosis and the exact lifting identity are
coherent. A claim of optimality for rollout is not derivable from the
representation alone. It would require a task distribution, a hypothesis
class, an optimization procedure, and a risk criterion. The defensible target
is therefore a minimal, controlled representation study plus explicit
approximation and stability obligations.

## Invariant Object

The invariant object is the continuum boundary descriptor supplied to a
time-step solution operator, together with its discretization on each query
mesh. A node count, a one-cell mask, or a particular PCNO activation is only a
discrete observation of that object.

For a semantic boundary subset Gamma_k contained in the boundary of a domain
Omega, the primary descriptor is

    B_k^ell(x) = rho(d_k(x) / ell),
    d_k(x) = distance(x, Gamma_k),

where ell is a positive physical length and rho is bounded and compactly
supported. The maintained implementation uses

    rho(r) = 1 - 3 r^2 + 2 r^3,    0 <= r < 1,
             0,                    r >= 1.

Thus B_k^ell is a bounded volume field. It is not a surface quadrature weight.

## Assumptions

- A1. Omega is a bounded Lipschitz domain in R^d. Smooth-boundary expansions
  additionally require a tubular neighborhood with positive reach.
- A2. Each semantic subset Gamma_k is relatively closed in the boundary after
  an explicit corner convention is fixed.
- A3. The physical collar width ell is fixed before outcome inspection and is
  unchanged across training and inference resolutions within a family.
- A4. For rate statements only, the mesh family is shape regular and locally
  quasi-uniform near Gamma_k, with representative spacing h. Without this
  assumption, exact sums over local cell measures replace powers of h.
- A5. Volume quadrature weights are consistent for bounded sufficiently
  regular functions. This is validated physical cell volume for dynamic FV
  and only reconstructed proxy quadrature for bump.
- A6. The sampled boundary geometry converges to the same Gamma_k. This is
  exact for the declared rectangular dynamic domain. It is an unproved
  mesh-derived polyline approximation for bump.
- A7. The PCNO Fourier modes are fixed in physical coordinates while h changes.
- A8. Rollout statements condition on one frozen physical boundary policy,
  recurrence, normalizer, and evaluator.

## Notation

- d is the spatial dimension; the maintained benchmarks use d = 2.
- h is a representative local mesh spacing only under A4.
- I_Gamma,h is a one-cell categorical boundary band.
- w_i is the volume quadrature weight at sample x_i.
- M_h(A) = sum_i w_i A(x_i) is the discrete volume mass of a scalar field A.
- ell is the physical collar width.
- q is a fixed graph-hop count.
- W_B is the slice of the PCNO lifting matrix multiplying boundary fields.
- K_h, W_h, and D_h denote the integral, pointwise, and differential PCNO
  branches at resolution h.
- R_h samples or restricts a continuum field to the mesh.
- I_h reconstructs or compares a discrete output in the chosen continuum norm.
- T is the target physical time-step operator under the frozen boundary policy.

## Derivation Strategy

Start from codimension and volume measure, not from node count. First derive the
support and amplitude of each representation. Then pass those quantities
through the exact discrete branches in the maintained PCNO. Finally separate
learning, discretization, and extension errors and place the one-step mechanism
inside an autoregressive stability relation.

## Derivation Map

1. A codimension-one one-cell band has volume proportional to local mesh
   thickness, even though its pointwise amplitude is one.
2. A bounded collar with fixed ell converges to one fixed volume function as
   h tends to zero.
3. A diffuse surface measure needs a different normalization and a
   distributional interpretation.
4. The PCNO lift receives the field pointwise; its Fourier analysis receives
   the quadrature-weighted field; its differential branch receives a discrete
   gradient.
5. The exact lift contribution is algebraic. Later block and rollout effects
   require interventions or stability assumptions.
6. Total error is decomposed into learned discrete error, representation and
   quadrature mismatch, and finite-width extension bias.

## Main Derivation

### 1. One-cell categorical band

Let I_Gamma,h(x_i) equal one on cells or vertices classified as touching
Gamma, and zero elsewhere.

Its pointwise amplitude is exactly order one:

    norm(I_Gamma,h)_infinity = 1.

Under A4, the number of tagged samples is order h to the power minus d plus
one, while the total number is order h to the power minus d. Therefore the
tagged node fraction is order h. More importantly, under consistent volume
quadrature,

    M_h(I_Gamma,h)
      = sum over tagged i of w_i
      = order(measure_(d-1)(Gamma) h).

Without A4, the correct statement is only the exact sum of local boundary-cell
weights. A single global h can be misleading on graded or anisotropic meshes.

For bounded per-node loss l_i, a volume-weighted loss assigns the band at most

    sum over tagged i of w_i l_i = order(h)

under A4. An unweighted node mean also gives order h only for a quasi-uniform
family. Trajectory-level batching can add a separate node-count or
presentation-frequency effect and must be audited independently.

### 2. Branch-scaling table

The following rates assume A4 and bounded hidden features before the listed
operation. They describe the direct representation signal, not the fully
trained nonlinear response.

| Quantity | One-cell tag I_Gamma,h | Fixed collar B_Gamma^ell | Diffuse surface field |
| --- | --- | --- | --- |
| Pointwise amplitude | order 1 | order 1 | order 1/ell after normalization |
| Volume-quadrature mass | order h | order ell | order 1 |
| Bounded-mode Fourier coefficient | order h | order ell | order 1 |
| Raw gradient scale | order 1/h across the tag jump | order 1/ell | order 1/ell^2 |
| Fixed-q-hop physical width | order q h | collar support ell plus order q h | layer ell plus order q h |
| Bounded volume-weighted loss share | order h | order ell | not a bounded-channel loss without care |

For the maintained PCNO integral branch, a Fourier analysis coefficient has
the form

    a_m,h = sum_i w_i v_i phi_m(x_i).

If v_i is the direct lifted contribution of the tag and phi_m is a bounded
fixed physical mode, then a_m,h is order M_h(I_Gamma,h), hence order h. A fixed
collar gives an order-ell coefficient that converges to a continuum integral.
This is the direct contribution; subsequent pointwise mixing can put
boundary-derived information into the whole domain before later Fourier
blocks.

For the least-squares differential branch, a unit jump across an edge of length
h produces weights and raw gradients of order 1/h, multiplied by mesh
conditioning constants. The maintained branch then applies a learned scalar,
a two-hop neighbor average, Softsign, and a one-by-one channel map. Softsign can
saturate a growing raw gradient, so order 1/h is not a claim that the final
branch output diverges.

The differential input dependency extends through the gradient edge plus two
neighbor-average hops. Its fixed-hop physical radius therefore contracts like
order h on a shape-regular refinement. The Fourier branch remains global.

### 3. Fixed ell and the sharp-boundary limit are different limits

Fix ell greater than zero. Under A5 and A6,

    M_h(B_Gamma^ell)
      tends to integral_Omega B_Gamma^ell(x) dx

as h tends to zero. The pointwise amplitude, physical support, and continuum
target do not change with resolution. If the quadrature has order p and the
collar and sampled geometry are sufficiently regular, a schematic rate is

    absolute quadrature error
      <= C_q (h / ell)^p + C_g epsilon_Gamma(h) / ell,

where epsilon_Gamma(h) is the boundary-geometry approximation error. The
constant deteriorates when the distance field loses regularity near corners
or the medial axis.

Now instead fix h conceptually and let ell tend to zero. The bounded collar
converges to zero almost everywhere, and

    norm(B_Gamma^ell)_Lp = order(ell^(1/p))

for finite p in a regular tubular neighborhood. It does not converge to a
nonzero surface measure. Thus a bounded collar is suitable as a volume
descriptor, but the sharp limit recreates the vanishing-mass conditioning
problem.

### 4. Diffuse surface measure

Let C_rho be the one-sided kernel mass

    C_rho = integral_0^1 rho(r) dr.

For the maintained cubic kernel, C_rho = 1/2. A one-sided diffuse surface
measure therefore begins with

    delta_Gamma^ell(x)
      = rho(d_Gamma(x) / ell) / (ell C_rho).

In tubular coordinates x = y - s n(y), the volume element includes the
Jacobian

    J(y,s) = product_j (1 - s kappa_j(y)).

An accurate surface integral either accepts an order-ell curvature error or
includes the corresponding geometric correction. Corners, nonunique closest
points, and finite reach require a partition or a diffuse-domain theorem.
Therefore simply multiplying the collar by 1/ell is not a complete surface
measure contract. D072 does not use this object as an ordinary model channel.

### 5. Exact maintained PCNO pathway

The input layout is coordinates, quadrature density, normalized conservative
state, one selected boundary representation, and normalized Mach. Let b_i be
the selected boundary vector at node i. The linear lift is

    h_i^0 = W_base a_i + W_B b_i + c.

Comparing the same input with b_i replaced by zero gives the identity

    Delta h_i^0 = W_B b_i.

For a single semantic replacement b_i to b_i prime,

    Delta h_i^0 = W_B (b_i - b_i prime).

The implementation exposes this exact pre-activation contribution without a
hook. It is an algebraic identity, not a statement that the final prediction
depends linearly on b.

At PCNO block r, the maintained update is

    h^(r+1) = h^r + sigma(K_r h^r + W_r h^r + D_r h^r)

except at the last block, which omits the residual-plus-activation form. Here
K_r is the quadrature-weighted Fourier branch, W_r is a pointwise one-by-one
map, and D_r is least-squares gradient, two-hop averaging, learned scaling,
Softsign, and channel projection. The decoder is a pointwise linear layer,
activation, and final linear layer. Therefore boundary information can:

1. remain local through W_r;
2. become globally available through K_r;
3. emphasize collar transitions and orientation-dependent state variation
   through D_r; and
4. be nonlinearly mixed and recurrently amplified across calls.

This branch audit explains plausible routes. It does not identify which route
a trained checkpoint uses; frozen interventions and nonperturbing
instrumentation are required.

### 6. Family-local semantic factorization

For dynamic FV, define two Boolean contact variables:

    s_y = 1 for y-symmetry contact,
    s_x = 1 for x-extrapolation contact.

The current exclusive code is exactly

    type = s_y + 2 s_x.

The continuum representation should therefore use independent fields B_y and
B_x. At a corner both are nonzero. A four-way precedence code is an artifact
of categorical storage, not a mathematical requirement of the boundary.

For bump, wall, outflow, and inflow remain three independent semantic subsets.
At a semantic transition, both collars may overlap. On the released graph, a
mixed-endpoint boundary edge is included in both finite-mesh subsets. Its
duplicated tangential length is a proxy error of boundary-edge scale, not an
arbitrary wall precedence. A physical refinement claim still requires common
geometry and mesh provenance.

### 7. Rigid-motion behavior

For a rigid map g(x) = Qx + a with orthogonal Q,

    distance(gx, g Gamma_k) = distance(x, Gamma_k).

Thus scalar semantic collars are rigid-motion invariant under jointly
transformed queries and geometry. If normals are later supplied, they must
transform covariantly as n(gx) = Q n(x). This proves descriptor covariance
only. The present PCNO is not thereby rotation equivariant: its coordinate
features, fixed Fourier modes, gradient weights, data distribution, and learned
parameters must be analyzed separately.

### 8. Schematic error decomposition

Let F_ell be an ideal continuum operator conditioned on the chosen finite-width
extension, and let F_ell,h be its consistent discrete counterpart. Then

    norm(I_h N_theta,h(R_h u, B_h^ell) - T(u))
      <= E_learn(theta,h,ell)
         + E_disc(h,ell)
         + E_ext(ell),

where

    E_learn
      = norm(I_h N_theta,h - I_h F_ell,h),

    E_disc
      = norm(I_h F_ell,h - F_ell),

    E_ext
      = norm(F_ell - T).

A schematic resolved-collar bound is

    E_disc
      <= C_0 h^p + C_1 (h/ell)^q
         + C_2 epsilon_Gamma(h)/ell.

The first term collects base state/operator discretization, the second collar
quadrature and differentiation, and the third geometry approximation. The
extension bias E_ext is not known to vanish merely because ell tends to zero.
It depends on whether the finite-width field is a sufficient, stable encoding
for the target operator and on how prescribed boundary data are extended.

This decomposition is a proof obligation, not a proved error theorem for the
current nonlinear learned PCNO.

### 9. Autoregressive mechanism

Let Phi be the exact one-call map and N_B the learned map with boundary field B.
For rollout error e_n,

    e_(n+1)
      = N_B(Uhat_n) - Phi(U_n)
      = [N_B(Uhat_n) - N_B(U_n)]
        + [N_B(U_n) - Phi(U_n)].

If N_B is locally L_n-Lipschitz on the visited states,

    norm(e_(n+1))
      <= L_n norm(e_n) + tau_n(B),

where tau_n is one-step model error. Iteration gives a weighted sum of past
tau_j terms. A boundary representation can therefore have a small early
effect but a large late effect if it reduces tau_j when a shock or vortex
reaches the wall, or if the recurrence amplifies earlier differences. This is
why one-step accuracy alone cannot establish rollout competence.

## Proposition-Level Statements and Proof Obligations

### Proposition P1: codimension-one dilution

Under A4 and consistent positive volume weights, the quadrature mass and
bounded weighted-loss share of a one-cell codimension-one tag are order h.

Proof obligation: establish local boundary-cell counting and weight bounds for
the actual mesh family. Do not apply this rate to unvalidated bump proxy
weights as physical volume.

### Proposition P2: fixed-collar consistency

Under A1--A6 with fixed ell, discrete samples and quadrature of B_k^ell
converge to the same bounded volume field and its volume integrals as h tends
to zero.

Proof obligation: specify quadrature order, distance-field regularity, corner
partition, and boundary-geometry error. For bump, first establish a common
geometry/refinement provenance.

### Proposition P3: exact lifting intervention

For any frozen checkpoint and fixed non-boundary inputs, replacing b_i by
b_i prime changes the pre-activation lift exactly by W_B(b_i-b_i prime).

Proof status: exact linear algebra; numerical replay still requires a stated
floating-point tolerance when different input widths use different matrix
kernels.

### Proposition P4: descriptor rigid-motion invariance

Scalar distance collars are invariant under joint rigid transformation of the
query points and semantic boundary subsets.

Proof status: follows from Euclidean distance invariance. It does not imply
PCNO rotation equivariance or rotation-generalized rollout accuracy.

### Proposition P5: blockwise perturbation bound

If each PCNO branch and activation is Lipschitz on the visited hidden-state
set, then the hidden difference caused by a frozen boundary intervention is
bounded recursively by the lift difference times a product of block
Lipschitz factors.

Proof obligation: bound the Fourier operator using quadrature mass and learned
weights, the pointwise maps by spectral norms, and the differential branch by
mesh conditioning, learned scaling, Softsign, and graph averaging. Empirical
JVP norms may estimate local sensitivity but do not prove the global bound.

### Proposition P6: rollout accumulation

Under a local Lipschitz bound for the learned recurrent map, the final
intervention effect and model error are bounded by the time-weighted
accumulation of one-call differences.

Proof obligation: define the admissible visited-state tube and verify that both
baseline and intervened rollouts stay within it. An inadmissible proposal ends
the comparison rather than being clipped into the tube.

## Remarks and Interpretation

- The collar is not claimed optimal. It is the smallest representation that
  fixes amplitude, support width, and volume mass in physical coordinates.
- The geometry-only union collar is essential. If it matches the semantic arm,
  the gain comes from locating the boundary, not learning its type.
- Separate prescribed state, prescribed flux, characteristic, SDF, normal, and
  tangent fields remain possible inputs. They should not be conflated with
  semantic identity.
- A learned boundary extension can be compared later only if the minimal
  ladder leaves a specific failure that justifies its added capacity.
- Dynamic FV and bump results answer different questions. Dynamic FV can test
  a validated resolution commutator; bump can test native-graph geometry
  variation under proxy geometry.

## Boundaries and Non-Claims

- No result here proves neural-operator discretization convergence.
- No bump node-dropping result is physical resolution transfer.
- A bounded collar is not a boundary integral and not a diffuse delta.
- Descriptor rotation covariance is not architecture equivariance.
- Frozen field ablation supports checkpoint dependence under that
  intervention; it does not prove the training representation is uniquely
  necessary.
- Matched-training differences are conditional on the tested data, optimizer,
  budget, seeds, recurrence, and evaluation population.
- The physical boundary-condition family and evaluator policy remain frozen.

## Open Risks

1. The primary ell is a physical hyperparameter and currently lacks a
   problem-derived optimality principle.
2. A collar truncated at ell does not encode global distance far from the
   boundary; positions and Fourier mixing may or may not compensate.
3. Bump semantic subsets and corner locations inherit released graph-tag
   discretization error.
4. The differential branch can respond strongly when ell is under-resolved or
   mesh stencils are ill-conditioned.
5. A model can exploit correlation between semantics and absolute position
   rather than learn transferable boundary dynamics.
6. Long-horizon improvements can arise from optimization regularization rather
   than the proposed continuum mechanism; matched initialization and
   geometry-only controls reduce but do not eliminate this risk.
