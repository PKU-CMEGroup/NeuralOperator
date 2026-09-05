# B5 solver-free intervention comparison

Date: 2026-09-05. Identity: `B5_BUMP_SOLVER_FREE_COMPARISON_20260905A`.

The owner requests a Bump comparison analogous to NACA0012 using interventions
that require no trusted solver. AutoDL remains the selected resource. This
amendment makes that comparison executable and supersedes the earlier four-arm
design envelope and historical-phenotype prerequisite. The frozen Stage 0B
result remains parent evidence; it is not an intervention comparison.

The fresh Clean model establishes the baseline. If it is already accurate and
stable, the comparison measures no-harm and remaining accuracy gains. Neither
blow-up nor a favorable intervention ranking is a prerequisite for reporting.
Historical test, prospective populations, and solver calls remain excluded.

## Common experiment

- Use the existing 256/44 train/development partition and exact prepared-shard
  manifest `5d5373fdcc682544bf330fba6d54fe65509dabe936baee7443c71a8c0c8d9fa7`.
- Refit the maintained normalization on all 256 training trajectories only.
  Use first-order conservative state `[rho,rho_v1,rho_v2,energy]`.
- Preserve the historical 16 selection keys in code. Report on the other 28
  development trajectories. No development data enter optimization, noise
  calibration, projection fitting, or checkpoint selection.
- Fresh PCNO: modes 8, Fourier periods `(6,2)`, five width-128 layers,
  projection width 128, physical node-type one-hot channels, Mach conditioning,
  and a zero output head. All overlapping initial parameters are paired.
- Before and after each physical transition apply the maintained causal nodal
  boundary map: fixed freestream inflow, interior-derived slip wall and outflow,
  with three-hop maximum sources, `rho_inf=1.4`, `p_inf=1`, `gamma=1.4`.
  This map is shared across arms and is not an exact DG boundary replay.
- Use normal-node proxy-weighted state-normalized MSE, batch one, AdamW
  (`lr=1e-3`, weight decay `1e-5`, gradient clip 1), 16,384 updates, seed
  `20260718`. Warm up for 328 updates from `1e-4`, then cosine decay to
  `2e-5` at the final update. Select the fixed terminal checkpoint.
- Each 256-update epoch visits every training geometry once, using the existing
  independently shuffled per-trajectory window queues. All arms share the same
  clean target `(key,t+1)` stream. Prefixes use preceding states and truncate
  only when earlier history does not exist.
- Use FP32 master parameters and BF16 autocast, with TF32 disabled. Key noise
  and exposure draws by experiment, seed, arm/purpose, and update or case/call.
  Different training losses have different compute; report calls, parameters,
  time, and memory. Equal updates do not mean equal compute.

## Deployed comparisons

| System | Training information and deployed transition |
| --- | --- |
| `CLEAN` | Clean one-step loss; online terminal weights. |
| `CLEAN_EMA` | The same Clean training run; terminal EMA weights, decay 0.995. No second training run. |
| `IID_RECOVERY` | Equal clean/recovery losses with the stored clean successor as both targets; online terminal weights. |
| `CURRICULUM_EMA_PREFIX_K13` | Train the online map on detached EMA-generated prefixes ending at time `t`, targeting stored `u_(t+1)`. Exposure probability is zero through 10% of training, ramps linearly to 0.5 at 40%, then stays 0.5; requested depth is uniform in `{1,2,3}`. Deploy EMA weights. |
| `PREFIX_ERROR_CORRECTOR_K13` | Freeze the fresh Clean map `F`. Train a separate identity-initialized PCNO `C` on `F^k(u_(t+1-k)) -> u_(t+1)`, with uniform requested `k in {1,2,3}`, plus an equally weighted `C(u)=u` identity loss. Deploy boundary-closed `C o F` with corrected-state feedback; two network calls. |
| `PCNO_PDEREFINER_K3_VPRED` | Conditional residual diffusion with a shared PCNO velocity predictor, candidate residual and four one-hot noise-level channels added to the common 12-input layout. Reuse the audited four-step DDPM equations. Deploy terminal EMA weights, four calls per physical step. |
| `MAPPED_TRAIN_PCA_R7` and `MAPPED_TRAIN_PCA_R32` | Apply the respective mapped training PCA projection after Clean, close boundaries again, and feed back the corrected state. Both ranks are fixed comparisons; neither is selected from rollout. |

EMA averages learned parameters and copies normalization and other fixed buffers
exactly. The refiner's eight extra input columns start at zero; all common
weights match Clean at initialization. Its output is DDPM velocity rather than
an independently trained `C o F` correction.

Recovery calibration uses eight equally spaced one-step pairs on each of the
frozen 32 training audit trajectories. Match post-boundary perturbation RMS in
state-normalized proxy coordinates to the fresh Clean model's error on those
pairs. Density and pressure get multiplicative log-normal noise; velocities
get additive Gaussian noise, sharing one scalar standard deviation. Three
deterministic scale-matching updates reuse the same normal draws; accept a 5%
matching tolerance. This is calibration, not rollout-based scale tuning.

For PDE-Refiner, measure Clean error in residual-normalized coordinates on the
same train bank. Set `beta_min` to its squared RMS and use
`beta_i=beta_min^(1-i/3)`, `i=0,1,2,3`. Require the measured RMS in `(0,1)`;
do not clip a failing calibration into the allowed interval. Training samples
one of four levels uniformly. Inference uses levels `3,2,1,0`, keyed Gaussian
draws, no clipping, and a boundary closure after the final physical output.

## Projection across different meshes

The existing temporal POD audit fits each trajectory separately. The new
operational baseline fits no basis from development trajectory values.

For each known evaluation geometry, choose a single training donor using Mach
and lower/upper wall profiles at 65 fixed horizontal coordinates. Standardize
features by training-only standard deviations (floor `1e-6`) and minimize
squared Mach distance plus mean squared wall-profile distance; numeric key
order breaks ties. Map nodes to normalized channel coordinates using those
wall profiles. Remap donor snapshots using four nearest neighbors of the same
physical node type and inverse-squared-distance weights; exact coincident
nodes copy directly. Fit centered PCA to the remapped donor's frames `0--78`
in target proxy-weighted normalized conservative coordinates.

This is a nearest-condition mapped training-PCA baseline. It differs from
NACA's shared-mesh global path PCA and has remapping and donor-selection error.
Record donor, distance, effective rank, clean-input projection damage, and
shock fidelity. Its result cannot establish a universal PCA ranking.

## Outcomes and interpretation

Start every evaluation at reference frame 0 and roll out 79 calls. Report
clean one-step error on eight fixed windows, errors at calls 1/20/40/60/79,
and AUC defined as the arithmetic mean of relative error at all 79 calls.
Relative L2 divides proxy-weighted state-normalized error energy by the
corresponding target energy without subtracting a time mean. Report equal-node
error as a robustness view at the five endpoints.

For each endpoint report front centroid/Chamfer distance, overlap, shock
strength/thickness log error, smooth-region high-pass energy, boundary error,
and admissibility. Retain correction dose and clean-state identity error.
Proxy integral errors are not physical conservation evidence. Record finite
but inadmissible states and continue; record nonfinite failure and its first
call. Incomplete rollouts have no fabricated finite H79/AUC and count as losses.

Predictions, frozen before comparison outcomes:

1. IID recovery reduces roughness but risks smearing sharp transported fronts.
2. Prefix exposure better matches endogenous errors and may outperform IID;
   Clean-EMA separates this from averaging alone.
3. The identity-anchored explicit corrector may improve coherent error without
   the geometry restriction of mapped PCA; this advantage fails if its
   identity error or shock damage offsets its rollout gain.
4. PDE-Refiner is a strong candidate given NACA, with four-call cost and
   stochastic variability explicitly reported; its ranking need not transfer.
5. Mapped rank-7 PCA is expected to damage transported structures; rank 32 may
   improve front fidelity without proportionate improvement in global error.

This pilot is exploratory open-development evidence. Confirm at seeds
`20260812/20260813` only after reporting the pilot and its controls; do not
automatically select a favorable subset or open historical test. A useful arm
should improve paired median H79 and AUC by at least 10%, win jointly on at
least 60% of cases, and increase mean clean one-step error by at most 5%.
Require no additional incomplete or inadmissible cases. Flag material structure
harm if the paired median front-centroid, strength-log-error, thickness-log-
error or boundary-RMS ratio exceeds 1.10 (denominator floor `1e-8`). These
rules summarize efficacy; an honest null/no-harm outcome still completes the
pilot comparison.

## Implementation and execution

Run `scripts/time_dependent_no/run_pcno_bump_corrective.py smoke` and then
`run`, supplying `--data-root`, a fresh `--output`, and `--device cuda`.
`utility/time_dependent_no/pcno_bump_corrective.py` owns the mathematical
mechanisms; `tests/time_dependent_no/test_pcno_bump_corrective.py` checks their
targets, gradients, RNGs, projection, access restrictions and composition.

Before launch, pass synthetic CPU tests and a full-native-mesh GPU smoke.
Bind the exact uploaded source archive, imported source files, split, data
manifest, accessed array hashes, normalization, boundary policies and terminal
checkpoints. Keep all generated outputs ignored. No manuscript upload or Git
push is part of this run. Preserve historical attempts and their original
contracts.
