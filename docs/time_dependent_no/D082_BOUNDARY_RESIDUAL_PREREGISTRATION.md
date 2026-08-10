# D082 Semantic Boundary-Residual Preregistration

Status: frozen before the first AutoDL CUDA outcome, 2026-08-09.

## Question and scope

D082 tests one explanation for D072's negative bump result: appending a bounded
collar to the shared PCNO lift changes the representation at every collar node
and lets every PCNO branch propagate that change, even when the useful boundary
effect may be local. D082 instead preserves the ordinary eight-input no-type
PCNO exactly and grants a small residual branch access to the semantic collars.

This is a representation/routing experiment. It does not change or improve the
physical boundary condition, does not add categorical node types, and does not
test K2, P_B*, rotation, resolution transfer, or a general BNO architecture.

## Frozen mathematical contract

For bump semantics k in {wall, outflow, inflow}, the input fields are the D072
fixed-physical-width bounded volume collars

    b_k(x) = rho(d(x, Gamma_k) / ell),  ell = 0.05.

Their values lie in [0, 1]. Mixed corners retain overlap; there is no semantic
precedence. These are volume descriptors from the released tagged boundary
polyline proxy, not diffuse surface measures or physical boundary quadrature.

Let R_theta be the normalized conservative residual from the ordinary no-type
PCNO. Its lift remains exactly

    [x_1, x_2, quadrature_density, q_hat_1, ..., q_hat_4, Mach_hat],

with width eight. No b_k or derivative of b_k enters this lift or any PCNO
pointwise, Fourier/integral, or differential branch.

The existing least-squares graph derivative D_h constructs a bounded direction

    g_k^h = D_h b_k / max(||D_h b_k||_2, 1e-12),

with the zero vector used where the norm is at most 1e-12. The learned side-path
input at node i is

    s_i = [q_hat_i, Mach_hat, b_1(i), ..., b_3(i),
           g_1x^h(i), g_1y^h(i), ..., g_3x^h(i), g_3y^h(i)].

It has 14 columns and deliberately excludes absolute coordinates. Gradient
components are coordinate-frame components, so this contract is not itself
rotation equivariant.

With a width-64 two-hidden-layer pointwise MLP A_phi and union gate
a_i = max_k b_k(i), D082 predicts

    q_i^{n+1} = q_i^n
                  + sigma_R [R_theta(q^n)_i + a_i A_phi(s_i)].

The learned correction is therefore exactly zero outside the collar. It is a
bounded collar-volume residual, not a learned surface integral or BNO claim.
The branch adds 5,380 parameters, about 0.0281% relative to the 19,155,208
parameter D072 N0 model.

The final affine layer of A_phi is initialized to exact zero. Every shared PCNO
and normalization tensor is copied exactly from the same-seed N0 construction,
and the CPU RNG is restored to the post-N0 state. Hence D082 and N0 implement
the same initial function while preserving the same data order for later
training.

## Frozen bump training protocol

D082 uses the D072 bump S1 shards and no sealed/test population. The following
items remain unchanged from D072 N0:

- seed and split seed 20260718, manifest split with 270 train and 30 open
  validation trajectories;
- stride 1, 40 full-coverage passes, 21,330 transitions per pass, batch size 4,
  and 853,200 total presentations;
- five width-128 PCNO layers, k_max 8, domain lengths (6, 2), and projection
  width 128;
- AdamW, learning rate 1e-3, weight decay 1e-5, gradient clipping 1, and the
  same 2% warmup-cosine schedule down to 2e-5;
- BF16 autocast, no input noise, no generated-state exposure, and one-step
  training loss;
- causal_nodal_physical closure before and after every model call,
  normal_closed primary objective, no boundary auxiliary, raw recurrence
  disabled, and no clipping or limiter;
- open-validation H79 rollout selection, with H20/H40/H60/H79 checkpoints and
  all 30 validation cases.

The exact control is D072 bump_s20260718_N0: data-manifest digest
6b7176d5f7f926af2fe05dc7a2e62a76292501461e47ac721782de4c8431ab4a,
normalizer digest
d2d07a4000acc3cfd45ff105a19e7437177d553cee4c18d50858f5572de7efec,
presentation-stream digest
5d5932560d2cd6f125141dfc9fc88a9bbb8f8a1e48a1c624a34548362dd3075b,
and best-checkpoint digest
c34c598f311b7636bbe21cdfbce521dbd16ca7972be10605f5dc3fe00e785fc3.
That selected epoch is 4; 27/30 cases complete H79, with reported H20/H79
errors 0.0536458/0.107729 on their respective surviving populations.

P_B* and attached K2 are excluded. P_B* did not satisfy the prior thickness and
closure-robustness gates, while the bounded matched K2 screen was harmful.
Bundling either would destroy the representation-only contrast.

## Preflight and stop conditions

Before the full run:

1. The synthetic CPU suite must prove an unchanged eight-column lift, exact
   copied backbone state, exact-zero side output, bitwise-equal complete FP32
   forward output, semantic sensitivity, exact zero outside the collar, RNG
   parity, strict checkpoint reload, and rejection of mixed input routes.
2. The isolated AutoDL source bundle must use D072's exact pcno/pcno.py plus the
   reviewed D082 overlays, and all staged hashes must be recorded.
3. A real-shard CUDA smoke run must produce finite train/validation/rollout
   metrics, a schema-valid checkpoint, the declared 8/14 feature layouts,
   exact initialization audit, nonzero side-head learning signal, and safe GPU
   memory headroom.

Stop before full training if provenance is incomplete, the semantic field
contract differs, initial outputs differ, the PCNO lift is not width eight,
metrics are nonfinite, gradients do not reach the side output, or the smoke run
exhausts safe memory.

## Frozen evaluation and decision matrix

The primary comparison is the same-seed D082 checkpoint against the exact D072
N0 checkpoint on the same 30 open-validation trajectories.

| Evidence | Promote to two additional matched seeds | Mixed; inspect once | Stop this routing |
| --- | --- | --- | --- |
| H79 completion | at least 27/30 | at least 27/30 | below 27/30 |
| Common-survivor H79 all/normal error | ratio at most 0.98 | ratio in (0.98, 1.02] | ratio above 1.02 |
| H20 all/normal and one-step normal error | each ratio at most 1.02 | one bounded miss with a clear regional gain | material or repeated miss |
| Shock-front centroid, thickness, strength, and smooth high-pass errors | every ratio at most 1.05 | one ratio above 1.05 with otherwise coherent gain | broad structure degradation |
| Admissibility and failure cause | no new failure mode | unchanged aggregate with case turnover | worse completion or new failure mode |

Promotion requires the conjunction, not one favorable scalar. Report
boundary, near-boundary, shock, smooth, and remaining-interior errors even
though bump weights and regions are proxies rather than physical conservation
measures. Compare H79 only on the exact common surviving population and report
the population size; raw means over different completion sets are not
comparable.

If the trained branch passes the training gate, frozen-checkpoint evaluation
must compare correct fields, all-zero fields, and one-semantic-at-a-time zeroing.
Those interventions can establish checkpoint-local field use. They cannot
prove that the branch, descriptor, or width is optimal. JVPs or activation
norms, if later used, are sensitivity diagnostics and require a no-hook versus
hooked inference equivalence test first.

Every later rollout animation must include the initial state and every one of
the 79 autoregressive frames without temporal subsampling. At minimum show
reference, N0, D082, D082-minus-reference, N0-minus-reference, D082-minus-N0,
boundary distance/collar overlays, and residual views through and after the
shock-wall interaction.

## Claim policy and next branch

A positive one-seed result licenses replication, not a paper claim. A
multi-seed positive result supports that localized access to semantic
boundary-volume descriptors improves the tested bump PCNO recipe under the
unchanged causal projection. It does not establish resolution transfer,
rotation generalization, physical conservation, or boundary-condition
improvement.

A mixed result supports mechanistic evaluation and one prespecified refinement,
not width search. A negative result rejects this pointwise collar-volume
residual routing under the frozen protocol. It does not show that boundary
information is useless; the next mathematically distinct candidate is a
surface-aware boundary operator with explicit quadrature and extension
contracts.

## Code ownership

- utility/time_dependent_no/pcno_euler2d.py owns the model contract and exact
  matched initialization.
- utility/time_dependent_no/pcno_runtime.py owns strict checkpoint rebuilding.
- scripts/time_dependent_no/train_pcno_euler2d_residual.py owns CLI,
  provenance, training, selection, and summary fields.
- tests/time_dependent_no/test_pcno_boundary_residual.py owns focused synthetic
  parity and route-separation tests.

No core PCNO API or physical boundary-policy implementation is changed.
