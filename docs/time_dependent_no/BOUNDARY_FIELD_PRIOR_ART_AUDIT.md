# Boundary-Field Prior-Art and Claim Audit

Status: preregistration support for D072, 2026-08-03. This document is
subordinate to RESEARCH_DIRECTION_DECISION.md and HANDOFF.md. It records a
public-literature search, the representation contract, claim boundaries, and
the gates that must be satisfied before data-scale work.

## Decision in one paragraph

Extending boundary information to functions over the domain is established
prior art. A bounded distance collar is also not, by itself, a novel method.
The plausible contribution is the narrower combination of: (i) diagnosing the
codimension-dependent failure of a one-cell categorical band through each
maintained PCNO pathway; (ii) replacing it with a manifest-bound continuum
field; and (iii) validating the mechanism in long autoregressive Euler rollouts
under genuine resolution comparisons, native bump geometry variation, and
rigid-coordinate stress tests. The project should therefore be presented first
as a time-dependent representation-consistency and rollout-stability study.
Geometry variation is a controlled stress test, not the primary claim.

A bounded search of public primary sources through 2026-08-03 found no paper
combining semantic boundary extensions, autoregressive time-dependent Euler,
PCNO branch-level interventions, resolution commutators, and rotation tests.
This negative search is not a proof of absence. Discovery of such a paper is an
immediate stop-and-reframe event.

## Exact representation contract

### Continuum object

For semantic boundary subset Gamma_k, D072 uses

    B_k^ell(x) = rho(distance(x, Gamma_k) / ell),

where

    rho(r) = 1 - 3 r^2 + 2 r^3,  0 <= r < 1,
             0,                    r >= 1.

The contract is pcno_bounded_semantic_collar_v1:

- distance is computed in physical coordinates before model-coordinate
  normalization;
- ell is a positive physical length and is fixed within a family across mesh
  resolutions and geometries;
- amplitude is bounded in [0, 1] and is not multiplied by 1/ell;
- each channel is an ordinary volume descriptor, not a surface measure and not
  a boundary-condition enforcement term;
- channel order, kernel, width, coordinate convention, geometry provenance,
  semantic factorization, and corner rule are stored and digested in the shard
  manifest;
- categorical types are omitted from the model input, but retained outside the
  model for the unchanged physical boundary policy, masks, and metrics;
- the G1 geometry control is max_k B_k^ell; S1 retains separate B_k^ell;
- under a rigid motion x -> Qx + a, both points and boundary geometry transform
  and the scalar channels must agree numerically. This descriptor covariance
  does not make PCNO rotation equivariant.

### Family-local semantics

The families remain mathematically separate.

- Dynamic FV uses two overlapping fields: y-symmetry contact and
  x-extrapolation contact. Its stored exclusive code is the derived pair
  type = s_y + 2 s_x, so code 3 activates both fields.
- Supersonic bump retains wall, outflow, and inflow fields. Boundary-cycle edges
  with unlike endpoint labels contribute to both adjoining semantic subsets;
  arbitrary wall precedence is forbidden.
- Bump boundary polylines are mesh-derived proxies. Their use does not convert
  node dropping or query-mesh resampling into PDE resolution transfer.

### Width rule

The outcome-independent primary rule is

    L_family = median over training geometries of
               min(x-span, y-span),
    ell_family = 0.05 L_family.

Only training geometry metadata may enter this calculation; states, targets,
validation outcomes, and sealed populations may not. The width is then one
family-level scalar. Sensitivities ell/2 and 2ell are allowed only after the
primary matrix as declared robustness checks, not a search.

For the validated dynamic rectangle [0, 2] x [0, 1], this gives ell = 0.05.
The bump value is intentionally not frozen: the retained workstation bundle
does not contain all training geometries needed to evaluate this rule. If
physical scale varies materially across geometries, a geometry-normalized width
would be a different contract and requires a new decision before outcomes are
seen.

### Independent fields, not a single overloaded encoding

The following mathematical objects must remain independently selectable:

| Object | Meaning | First D072 ladder? |
| --- | --- | --- |
| Domain indicator or signed distance | Geometry and inside/outside membership | G1 only through the union collar |
| Semantic subset collar | Where a named physical boundary class is located | Yes, S1 |
| Closest-point normal or tangent | Local orientation, undefined or multivalued at medial axes/corners without a rule | No |
| Prescribed state data | Dirichlet-like physical data with its own units and normalization | No |
| Prescribed flux data | Neumann/Robin-like data with its own units and surface convention | No |
| Characteristic information | Incoming/outgoing Euler characteristics relative to the local normal and state | No |

The current bump labels describe semantics under a frozen policy; they are not
prescribed wall states or fluxes. Adding any of the last four rows would expand
the study beyond D072.

## Branch-scaling audit

Let h denote a representative mesh spacing, ell a fixed physical collar width,
and q a fixed graph-hop count. The rates below require shape-regular,
quasi-uniform sampling near a codimension-one boundary and consistent volume
quadrature. They are analytical diagnostics, not claims about unvalidated bump
proxy weights.

| Quantity | One-cell tag I_Gamma,h | Bounded collar B_k^ell, fixed ell | Diffuse surface field delta_k^ell |
| --- | --- | --- | --- |
| Pointwise amplitude | O(1) | O(1) | O(ell^-1) |
| Volume-quadrature mass | O(h) | O(ell) | O(1), after normalization |
| Bounded Fourier/integral coefficient | O(h) | O(ell) | O(1), after normalization |
| Raw discrete gradient scale | O(h^-1) | O(ell^-1) | O(ell^-2) |
| Fixed-q physical support/reach | O(qh) | ell + O(qh) | ell + O(qh) |
| Consistent volume-weighted loss share | O(h) | O(ell) | not an ordinary bounded loss channel |
| Unweighted batching share | sampling-policy dependent | sampling-policy dependent | sampling-policy dependent |

Consequences for maintained PCNO are branch specific:

- Initial lift: the exact preactivation contribution is W_B B(x). For a
  categorical type replacement it was W_type(e_k-e_0); for a field intervention
  it is W_B times the field difference.
- Pointwise branch: preserves pointwise amplitude but cannot by itself prevent a
  one-cell signal from occupying a shrinking physical region.
- Fourier/integral branch: bounded kernels and physical quadrature see the
  vanishing O(h) mass of a one-cell band; a fixed collar retains O(ell) mass.
- Differential branch: a one-cell jump produces O(h^-1) raw gradients while a
  fixed smooth collar supplies O(ell^-1) scale. The maintained fixed-hop average
  has physical reach O(qh), so fixed q is not a fixed physical receptive width.
- Repeated PCNO blocks mix these three effects nonlinearly. Per-block activation
  differences are diagnostic pathways, not additive causal shares.
- Decoder: it is pointwise and can only decode the representation delivered by
  the preceding blocks; it does not restore a vanished integral signal or a
  fixed physical receptive width.
- Loss and batching: physical volume weighting makes a one-cell boundary band's
  direct contribution vanish like h. An unweighted node mean can hide, reverse,
  or exaggerate that rate when boundary nodes are oversampled.

The fixed-ell limit h -> 0 approximates a bounded volume function. The separate
sharp-boundary limit ell -> 0 makes that function vanish in volume. To represent
a surface measure instead, one needs

    delta_k^ell(x) = [ell C_rho]^-1 rho(d(x, Gamma_k)/ell),

with C_rho = integral_0^1 rho(r) dr = 1/2 for this one-sided kernel, plus the
appropriate tubular-neighborhood Jacobian, reach/corner treatment, and surface
quadrature convention. Multiplication by 1/ell alone is not a complete surface
integral contract.

Detailed propositions and proof obligations are in
BOUNDARY_FIELD_DERIVATION_PACKAGE.md.

## Primary-literature matrix

Each row distinguishes geometry from boundary type/data, physical from
mesh-dependent width, volume from surface objects, and actual temporal evidence.

| Work and status | Geometry versus boundary information | Mathematical representation | Width and measure | Evidence | Limitation relative to D072 |
| --- | --- | --- | --- | --- | --- |
| [ReNO](https://proceedings.neurips.cc/paper_files/paper/2023/hash/dc35c593e61f6df62db541b976d09dcf-Abstract-Conference.html), NeurIPS 2023 | General discretized function spaces; not a boundary encoder | Representation equivalence, operator aliasing, commuting discrete representations | Not applicable | Theory and examples of discretization mismatch | Supplies the correct resolution language, not semantic boundary fields or Euler rollouts |
| [DAFNO](https://papers.neurips.cc/paper_files/paper/2023/hash/940a7634dab556b67af15bacd337f7db-Abstract-Conference.html), NeurIPS 2023 | Domain geometry; no independent semantic boundary data | Smoothed characteristic function embedded in FNO integral layers | Bounded volume mask; smoothing controlled by a physical-coordinate beta, tuned rather than frozen as an operator contract | Hyperelasticity, steady transonic airfoil, and fracture evolution; the hyperelasticity study includes 41 x 41 training and 161 x 161 testing | Very close geometry-mask prior art; no semantic subsets, autoregressive Euler, PCNO pathway audit, or rotation intervention |
| [GINO](https://proceedings.neurips.cc/paper_files/paper/2023/hash/70518ea42831f02afc3a2828993935ad-Abstract-Conference.html), NeurIPS 2023 | Variable 3D geometry plus inlet velocity | SDF and point cloud, GNO to latent grid, FNO core, GNO decoder | SDF is a bounded/physical volume descriptor; graph neighborhoods use physical radii | Steady vehicle RANS surface pressure on varying geometries and resolutions | Strong geometry architecture; no semantic boundary-class extension or autoregressive time |
| [BENO](https://openreview.net/forum?id=ZZTkLDRmkg), ICLR 2024 | Complex geometry and inhomogeneous boundary values | Interior-source and boundary-value GNN branches plus a transformer boundary-geometry latent | Boundary samples, not a declared fixed physical collar or diffuse volume field | Elliptic PDEs with complex boundary conditions | Static and architecture-specific; no Euler rollout or resolution commutator |
| [Boundary-Augmented Neural Operators](https://openreview.net/forum?id=DqZoWaDwfN), NeurIPS 2025 AI4Science workshop poster | Domain and boundary functions, geometry OOD | Separate domain and boundary operators; low-rank boundary-to-domain interaction with surface-element weights | Explicit surface integral rather than a bounded collar | Poisson and steady transonic Euler airfoil/flap geometry; discretization and point-distribution robustness | Closest surface-operator prior; no time dependence, rotation, or PCNO frozen mechanism study |
| [Learned Function Extensions](https://arxiv.org/abs/2602.04923), arXiv v2, 2026; accessible primary page does not itself certify a venue | Boundary type and data; geometry is fixed within each dataset, with transfer studies across datasets | Zero, harmonic, and learned attention pseudo-extensions of boundary functions to the whole domain; separate geometry features such as distance | Domain extensions; zero extension is mesh-local, harmonic/learned extensions are nonlocal; not a surface measure | 18 static Poisson/elasticity/hyperelasticity datasets; boundary-node tests at about 286, 401, 669, and 1004 nodes; regular attention is resolution sensitive, while masking improves transfer | Strongly establishes the primitive and reports resolution/high-frequency failures. No autoregressive hyperbolic rollout, PCNO branch scaling, common-source commutator, or rotation test |
| [LP-FNO](https://arxiv.org/abs/2406.16740), arXiv 2024 and ICML 2024 AI for Science workshop | Boundary functions on a fixed rectangular domain | Two lower-dimensional FNOs lifted to the domain by a product layer | Boundary-to-volume lifting; no fixed physical collar | Static 2D Poisson; models trained at 32, 64, or 128 and tested across all three grid resolutions, including 32 to 128 zero-shot super-resolution | Establishes boundary-to-domain operator lifting; no variable geometry or time, and visible checkerboard artifacts remain |
| [Diffuse-domain method](https://pmc.ncbi.nlm.nih.gov/articles/PMC3097555/), Communications in Mathematical Sciences 2009 | Complex and moving domain geometry with boundary conditions | Phase-field/domain indicator and diffuse boundary source terms on an embedding domain | Diffuse layer; singular terms approximate surface effects | Numerical PDE method for stationary and moving geometries | Numerical enforcement method, not a learned input channel; nevertheless establishes diffuse boundary representations |
| [Diffuse-domain analysis](https://arxiv.org/abs/1407.7480), Communications in Mathematical Sciences 2015 | Complex dynamic geometry with Dirichlet, Neumann, and Robin data | Matched-asymptotic analysis of diffuse source approximations | epsilon is typically tied to minimum grid size; first- or second-order error depends on correction | Theory plus numerical confirmation | Warns that normalization and geometric correction determine the sharp-limit object; not a fixed-ell neural input |
| [Closest Point Method](https://www.sciencedirect.com/science/article/pii/S002199910700441X), Journal of Computational Physics 2008 | Embedded surface geometry | Closest-point extension from a surface into a narrow Cartesian band | Physical embedding band chosen for the numerical stencil; not a surface-density input | Surface PDEs including time stepping | Classical extension machinery, not semantic BC conditioning or operator learning |
| [Distance-function boundary enforcement](https://www.sciencedirect.com/science/article/pii/S0045782521006186), CMAME 2022 | Geometry and prescribed boundary data | Approximate distance/R-functions and transfinite interpolation in the trial ansatz | Physical distance field; not a diffuse surface measure | Static PINN boundary-value problems on complex domains | Changes/enforces the physical BC and is outside D072's frozen-policy scope |
| [Geo-FNO](https://arxiv.org/abs/2207.05209), JMLR 2023 | Variable geometry | Learned deformation between physical and latent domains | Global coordinate map, not a collar | Multiple variable-domain PDEs including steady Euler | Geometry generalization prior; no semantic-boundary mechanism or autoregressive time |
| [piG-Sp2GNO](https://arxiv.org/abs/2508.09627), arXiv 2025 | Geometry-aware and time-dependent tasks | Boundary interpolation or learned coordinate/BC/geometry encoder with a spatio-spectral graph operator | No single frozen physical-width contract | Variable-geometry static Darcy/plate tests; fixed-geometry time tests including long autoregression | Geometry and time capabilities are demonstrated in separate settings, not semantic Euler geometry rollouts |
| [G-FNO](https://proceedings.mlr.press/v202/helwig23a.html), ICML 2023 | Coordinate symmetries, not boundary semantics | Group-equivariant Fourier layers for rotations, translations, and reflections | Not applicable | Autoregressive PDEs, resolution changes, and rotation/reflection groups on regular grids | Supplies rotation controls and an architectural comparator, not a boundary encoding |
| [INO](https://proceedings.mlr.press/v206/liu23f.html), AISTATS 2023 | Invariance/equivariance of physical response | Relative-coordinate invariant kernels | Not applicable | Material-response benchmarks and rigid transformations | No boundary extension or long Euler rollout |
| [EqGINO](https://arxiv.org/abs/2606.03260), ICML 2026 per arXiv comments | 3D complex geometry under rigid transformations | Isotropic spectral construction with exact discrete equivariance | Geometry representation rather than collar | Rotation and sampling/resolution evidence on 3D PDEs | Strong current rotation prior; no time-dependent semantic boundaries |
| [PCNO](https://arxiv.org/abs/2501.14475), CMAME 2025 | Complex variable point-cloud domains | Point-cloud Fourier, pointwise, and differential branches using weights/connectivity | Mesh and quadrature enter each branch; no continuum semantic collar contract | Static parametric PDEs, variable geometries, adaptive point clouds | Architecture under study; original work does not supply the requested time-dependent boundary mechanism |
| [Boundary-indexed operator families](https://arxiv.org/abs/2603.01406), arXiv 2026 | Varying boundary-condition distributions | Conditional-risk and non-identifiability view of a family indexed by BCs | Not an encoding | Theory and static Poisson shifts | Supports explicit conditioning and claim caution; no representation construction or time |
| [Generalized Neural Operator](https://arxiv.org/abs/2607.21932), arXiv 2026 | PDE parameters and arbitrary boundary constraints | Boundary transfer to a unified latent Dirichlet representation plus gated kernels and stability objective | Learned latent transfer, not a fixed collar | Reported heterogeneous parameter/BC generalization | Very recent adjacent method; no verified match to the full PCNO Euler/resolution/rotation combination |

### Closest-overlap matrix

Y means explicit coverage, P means partial/adjacent coverage, and N means absent
from the paper's demonstrated combination.

| Work | Semantic boundary subsets/data | Boundary-to-domain extension | Codimension scaling diagnosis | Autoregressive time-dependent Euler | PCNO branch interventions | Resolution commutator | Rotation test |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DAFNO | N | P | N | N | N | N | N |
| GINO | N | P | N | N | N | N | N |
| BENO | Y | P | N | N | N | N | N |
| Boundary-Augmented NO | P | Y | P: explicit surface quadrature | N | N | N | N |
| Learned Function Extensions | Y | Y | P: observes resolution and spectral failures | N | N | N | N |
| LP-FNO | Y | Y | N | N | N | N | N |
| Diffuse/closest-point methods | P | Y | Y for numerical sharp/diffuse limits | P | N | N | N |
| piG-Sp2GNO | P | P | N | P: time and geometry tested separately | N | N | N |
| G-FNO / EqGINO | N | N | N | P | N | P | Y |
| D072 target | Y | Y: fixed bounded collar | Y | Y | Y | Y on dynamic FV | Y stress test |

The novelty hypothesis is therefore partly falsified. The primitive
representation is known, and learned extensions already connect rough/zero
extensions to resolution sensitivity and high Fourier content. What remains
plausibly underexplored is the codimension-to-branch scaling diagnosis and its
frozen mechanistic validation in autonomous PCNO Euler rollouts. That narrower
hypothesis must be rechecked immediately before submission.

## Borrow versus invent

Borrow directly:

- ReNO's commuting-representation and aliasing language;
- DAFNO's use of smooth physical-coordinate geometry functions;
- the extension-space formulation, mixed-BC separation, and resolution warning
  from Learned Function Extensions;
- explicit surface quadrature, if a later surface operator is justified, from
  Boundary-Augmented Neural Operators;
- diffuse-domain normalization and geometric-correction obligations;
- closest-point geometry and normal/tangent definitions;
- rotation baselines and equivariance language from G-FNO, INO, and EqGINO.

Project-owned contribution candidates, not yet novelty claims:

- an exact family-local semantic contract for dynamic FV and bump corners;
- the one-cell codimension scaling table carried through the actual PCNO lift,
  pointwise, Fourier/integral, differential, block, loss, and decoder paths;
- exact N0-to-field matched initialization with zero inserted lift columns and
  identical non-lift state/RNG;
- frozen field interventions with hook-equivalence gates and branch/block
  instrumentation;
- physical-time-to-boundary interaction analysis over every rollout frame;
- separating genuine dynamic common-source resolution evidence from bump
  native-geometry evidence and query resampling;
- the interaction between boundary representation, shock-wall contact, and
  long-horizon error growth.

Do not invent a learned extender, new geometry architecture, or surface operator
before the minimal ladder identifies a failure that requires it.

## Minimal representation ladder and escalation rule

| Arm | Model input | Purpose |
| --- | --- | --- |
| N0 | no categorical type and no boundary field | information-free baseline |
| G1 | one union geometry collar | tests boundary location without semantics |
| S1 | separate family-local semantic collars | tests incremental semantic information |

G1 and S1 must be exact-function matches to the same N0 initialization. All
other training and evaluation choices are matched. This three-arm ladder is
sufficient to answer whether location and semantics help the tested rollout;
it does not answer which encoding is globally optimal.

Only if S1 fails a predefined representation gate while frozen interventions
show unresolved boundary sensitivity may a second stage compare one borrowed
learned/harmonic extension. Only if the failure is specifically an inability to
transport boundary information into the domain should a BNO-style surface
architecture be considered. Those are new preregistrations, not silent D072
variants.

## Mechanistic interpretation and measurements

The collar has an analyzable entry point but no guaranteed optimality. At the
lift, its contribution is exactly W_B B(x). A fixed physical support gives the
integral branch nonvanishing quadrature mass, regularizes differential scale
from h^-1 to ell^-1, and exposes a macroscopic region to pointwise nonlinear
mixing. During recurrence, the relevant question is whether those differences
remain dormant until a wave reaches the boundary, then alter the correct
wall/inflow/outflow response without degrading smooth interior evolution.

Required measurements are family separate and resolution stratified:

- teacher-forced one-step and free-rollout error at every physical time;
- boundary, fixed physical near-boundary, shock, vortex, and smooth-region
  metrics;
- physical-frequency-band state and residual effects;
- completion, density/pressure admissibility, conservation proxies only where
  physical quadrature is validated, and boundary leakage;
- post-lift differences, per-block hidden differences, and pointwise,
  integral, and differential branch contributions;
- optional JVP/intervention norms labeled as sensitivity diagnostics;
- dynamic common-source conservative restriction and resolution commutators;
- bump native-geometry strata, never relabeled as resolution transfer;
- rigid descriptor-covariance checks before model rotation stress tests.

Hook-free and hooked inference must agree at the declared tolerance before any
activation evidence is admitted. Animations must contain all frames and show
baseline, intervention, their difference, baseline/intervened residuals,
boundary-distance overlays, and selected post-lift/hidden sensitivity maps. The
late frames after shock-wall contact are mandatory; a highlight-only animation
is not an admissible replacement.

## Proposition and proof obligations

The derivation package records the formal statements. Before a paper claim,
the following obligations remain:

1. Sampling consistency: show discrete collar samples converge to the same
   B_k^ell under stated geometry and distance-approximation assumptions.
2. Quadrature consistency: bound the PCNO integral error in terms of h/ell and
   geometry approximation, without applying quasi-uniform rates to unvalidated
   bump weights.
3. Extension bias: define the target sharp or trace-aware operator and bound
   the bias induced by fixed ell; do not treat ell -> 0 as the same experiment
   as h -> 0.
4. Dynamic factorization: prove that the two overlapping dynamic descriptors
   recover all four family-local codes without changing the physical policy.
5. Corner geometry: specify reach, closest-point nonuniqueness, and overlap for
   bump junctions.
6. Lift identity: verify numerically and analytically that the only initial
   change is W_B B and that the matched model equals N0 at initialization.
7. Rotation covariance: prove the scalar distance fields commute with rigid
   motions; separately test, but never infer, model equivariance.
8. Rollout propagation: state a local Lipschitz/stability condition under which
   one-step representation/discretization errors accumulate, rather than
   claiming that a small one-step error guarantees a stable horizon.

The schematic decomposition is

    total error <= learning error
                   + discretization error(h/ell)
                   + geometry approximation error
                   + extension bias(ell).

Its terms must be operationalized; it is not yet a proved convergence theorem
for trained PCNO.

## Causal-language policy

- A frozen field intervention establishes checkpoint-output dependence under
  that checkpoint, evaluator, and intervention only.
- A matched-training contrast estimates the effect of granting the
  representation under the tested data order, optimizer, budget, and seeds.
- Hidden activations and JVPs are mechanism diagnostics, not causal shares or
  proofs of physical reasoning.
- Representation consistency means that the input samples a fixed continuum
  object. It does not establish model resolution transfer.
- Descriptor covariance does not establish architectural equivariance.
- Native bump geometry variation does not establish broad geometry
  generalization, and bump query resampling does not establish PDE resolution
  transfer.
- No finite ladder establishes optimality. Negative S1 results do not show that
  boundary information is useless; positive S1 results do not show that collars
  dominate learned extensions or surface operators.

## Missing artifacts and exact retrieval requests

Data-scale execution is closed until all applicable requests are satisfied.

1. Supersonic bump: retrieve either the complete source HDF5 named by the
   retained 300-trajectory manifest or complete prepared shards for all 300
   trajectories. Supply SHA-256, byte size, source modification time, source
   manifest, split JSON and digest, normalization JSON and digest, per-geometry
   coordinates, tagged boundary arrays, connectivity, targets, and the exact
   meaning/provenance of proxy point weights. Do not access the sealed test
   population. This is required to freeze ell from all training geometries.
2. Dynamic FV: retrieve the source family_manifest.json and require its digest
   to equal
   150c589af9c7291674f502dfe30ca77930d9429fb2115b3f2a0f22042a08c69e.
   Retain the source references and array hashes for every open train/validation
   resolution used by the common-source comparison.
3. For both families: bind source, split, normalizer, checkpoint-initialization,
   collar contract, prepared-array, code-commit, and evaluator digests in each
   run manifest. A matching filename is not provenance.
4. Before remote execution: select the authorized compute machine and provide a
   working noninteractive secure access route if the selected machine lacks
   one. Do not embed a password in commands or documentation.

## Exact later code ownership

| Responsibility | Maintained owner |
| --- | --- |
| Collar construction and schema | utility/time_dependent_no/pcno_boundary_fields.py |
| Dynamic and bump artifact creation | the two maintained prepare_pcno shard scripts under scripts/time_dependent_no |
| Input layout, exact lift diagnostic, and matched initialization | utility/time_dependent_no/pcno_euler2d.py |
| Legacy-safe checkpoint reconstruction | utility/time_dependent_no/pcno_runtime.py |
| Trainer flags and run-manifest binding | scripts/time_dependent_no/train_pcno_euler2d_residual.py |
| Genuine dynamic resolution sample construction | utility/time_dependent_no/pcno_resolution_transfer.py |
| Frozen interventions and nonperturbing instrumentation | extend utility/time_dependent_no/pcno_node_type_interpretability.py only through a neutral field interface; do not duplicate hook logic |
| Resolution/region/frequency metrics and all-frame visualizations | extend existing evaluators/visualizers only when the data-scale stage is authorized |
| Contract/regression tests | tests/time_dependent_no |

No core pcno API change is required for the first ladder.

## Conditional paper claims

Positive outcome:

> Under fixed physical boundary policies and matched training, bounded semantic
> collars improve long-horizon PCNO Euler rollouts over both no-field and
> geometry-only controls on the tested families. Frozen interventions and
> branch-resolved diagnostics localize the gain to nonvanishing boundary
> information whose effect grows after physical boundary interaction, while
> dynamic common-source tests show better finite-resolution commutation.

Mixed outcome:

> A fixed physical geometry collar improves the tested rollouts, but separate
> semantic collars add no consistent benefit across seeds/families/resolutions.
> The evidence therefore supports resolution-consistent boundary localization,
> not a general benefit from semantic boundary typing; family-specific effects
> remain descriptive.

Negative outcome:

> Replacing one-cell categorical tags with bounded physical collars removes a
> clear input-scaling defect but does not consistently improve autonomous PCNO
> Euler rollouts under the tested training recipe. This separates continuum
> representation consistency from rollout competence and motivates, but does
> not itself validate, learned extensions or boundary-surface architectures.

## Go/no-go

- GO: retain the minimal implementation and synthetic contract tests; the code
  has a clear continuum object, exact matched initialization, and bounded scope.
- GO, conditional: prepare open-population shards only after the source and
  geometry manifests above are complete and the primary width is frozen.
- NO-GO now: data-scale training, bump scientific interpretation, sealed
  evaluation, an optimal-encoding claim, or a broad geometry-generalization
  claim.
- STOP: checkpoint/source/normalizer provenance is incomplete, hooks change
  outputs, descriptor covariance fails, or bump evidence cannot be separated
  from query-mesh resampling.
