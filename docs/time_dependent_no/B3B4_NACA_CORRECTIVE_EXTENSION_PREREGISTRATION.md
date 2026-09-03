# B3/B4 NACA Corrective-Mechanism Extension

Updated: 2026-09-02

Experiment ID: `B3B4_NACA_CM_EXT_20260902A`

Status: owner-directed implementation and train/development execution. A
train-only SU2 displaced-restart pilot must pass before the paired relabeling
bank is generated. Prospective and sealed populations remain closed.

## 1. Scientific Role

The closed five-arm successor is evidence, not the end of the PDE study. This
extension fills three mechanism gaps without changing the parent result:

1. literature-grounded model-prefix exposure with a curriculum;
2. a paired recovery-versus-dynamics-relabeling comparison on identical
   displaced BDF2 inputs; and
3. a learned iterative operational mechanism based on PDE-Refiner.

The qualified thesis is:

> One-step error constrains the learned update on supervised states but cannot
> certify displaced-input behavior without additional assumptions. Remaining
> in a train-defined controlled region is required for a data-only rollout
> guarantee, not universally necessary for every successful model and not
> sufficient without accurate on-path dynamics.

The train-defined path and its distances remain model-independent. Solver
defect is an offline diagnostic and never defines ID, OOD, or the tube.

## 2. Parent Evidence And Fixed Scope

- Preserve `B3B4_NACA_CM_20260901A` and its artifacts byte-for-byte.
- Reuse its fixed SU2 problem, complete BDF2 state, PCNO width/modes, train and
  development roles, optimizer family, seeds `17/29/43`, horizons, and
  evaluator semantics.
- Keep the prospective frames `1511--1750` and sealed frames `1755--1994`
  closed.
- The recovery input scale remains fixed from train-only parent error:
  overall state-normalized RMS `2.593012258e-3`, with per-field standard
  deviations
  `[2.351053, 1.794329, 4.045914, 2.355419, 1.718743]e-3`.
- This extension does not tune that scale using rollout outcomes.
- The observed radial bands are called mesh-aligned rippling. Flat native-quad
  rendering and symmetric-log colors accentuate them; their architectural
  cause is not established.

## 3. Mechanism Matrix

| Contrast | Input law | Target or deployed operation | Question |
| --- | --- | --- | --- |
| Existing `DETACHED_PUSHFORWARD` | Always-on one-prefix online-model input | Stored clean future | How harsh was the closed stress test? |
| `MP_PDE_PUSHFORWARD_M01` | Original MP-PDE stochastic depth `m in {0,1}` after one clean epoch | Terminal stored clean future | Does the source-faithful clean/exposed mixture avoid the stress-test failure? |
| `CURRICULUM_EMA_PUSHFORWARD_K13` | Ramped EMA-generated prefix depth `1--3` | Terminal stored clean future | Does smoother, deeper model-owned exposure broaden useful control? |
| `PAIRED_RECOVERY` | Frozen displaced BDF2 bank | Clean future from the undisplaced path | What is gained by suppressing the displacement? |
| `DYNAMICS_RELABEL` | The identical frozen displaced BDF2 bank | One-step SU2 continuation from that pair | What is gained by learning the trusted displaced response? |
| `PCNO_PDEREFINER_K3_VPRED` | DDPM-noised candidate residual at four scheduler levels | Shared-PCNO velocity prediction, reversed in four calls | Can a learned iterative corrector improve rollout at four calls per step? |
| Existing `PATH_PROJECTION` | Raw prediction | Train-path piecewise-linear projection | What does direct empirical return buy? |

`MP_PDE_PUSHFORWARD_M01` is the literal released MP-PDE comparator. The
curriculum/EMA/depth-3 arm follows the stronger protocol reported in the
PDE-Refiner study and is named separately; it is not presented as the literal
MP-PDE implementation.

Literature fidelity is bound to MP-PDE commit
`8415c5f6b6044749582bd9ebff7ecd8045ad81b2`, PDEArena commit
`78a8b03d50115d8bb24ce9f04efa5b920fcd4369`, and the latter's pinned
`diffusers==0.17.1` DDPM implementation. Exact public source-file byte counts
and SHA256 values are recorded in the JSON contract. The local scheduler uses
the released `fixed_small` variance rule and is checked at all four timesteps
against frozen reference vectors before scientific execution.

## 4. Train-Only SU2 Relabeling Pilot

The numerical transition is a map on the complete BDF2 pair, not on a single
state. Use train centers `956`, `1075`, and `1193`. For each center, create one
independent Gaussian history-pair draw with the fixed per-field scale and test
both signs, plus a zero-displacement control. Repeat one displaced case and add
one auxiliary-column probe, for eleven train-only one-step executions.

Generate directions with a CPU `torch.Generator` seeded by `20260902` and
`torch.randn(..., dtype=torch.float32)`, using one ascending-center stream over
all centers `956--1193`. Each draw has shape `[2,N,5]` in previous/current
history order and is multiplied by the per-field state-normalized scale. The
pilot consumes the selected centers from this stream, so its directions are
identical to their later query-bank entries. Add the displacement in float32 to
the same float32 clean states presented to PCNO.

The restart writer must template the corresponding verified native trajectory
files, preserve the header, field order, coordinates, and all twelve
auxiliary/native-only fields, and replace exactly the five evolved fields in
both history files. Round-trip hashes and changed-field lists are recorded.
`Nu_Tilde` sign is reported but is not an input rejection rule: the clean wall
value is zero, so unmasked Gaussian corruption makes negative samples
unavoidable under the registered law.

Preserve the absolute native indices: a center `n` stages histories `n-1,n`,
sets `RESTART_ITER=n+1` and `TIME_ITER=n+2`, and expects successor `n+1` under
a distinct output stem. At center `1075`, also run the same five-field BDF2
pair for sign `+1` while swapping the auxiliary columns of native templates
`1074` and `1075` between the two history slots. If the five evolved successor
fields are not bitwise identical, auxiliary fields are solver-relevant and the
full bank stops pending a new consistent-state contract. Record, but do not
gate on, full-file equality.

The pilot passes only if:

1. every input and output is finite, with positive density and ideal-gas
   pressure, and SU2 exits successfully with exactly one expected restart;
2. each zero control has aggregate and per-field state-normalized RMS
   discrepancy at most one percent of the corresponding injected-noise scale;
3. the realized per-field displacement RMS is within five percent of its
   registered value;
4. a repeated displaced case and the auxiliary-column probe are bitwise
   invariant in their declared comparisons; the repeat requires identical full
   native output bytes and evolved arrays, while history rows need only pass the
   convergence contract; and
5. the median trusted response exceeds four times the largest zero-control
   replay discrepancy.

Compute each zero-control discrepancy against the stored native-float64 clean
successor, after writing the clean inputs through the authoritative float32
model-facing states. Compute the trusted response as the state-normalized
equal-entry RMS between each signed output and its same-center zero-control
output. The median is taken over the six center/sign responses.

No failed solver case is silently dropped or replaced. A failed gate stops the
full relabeling bank and is reported as restart/transition insufficiency, not as
a scientific failure of dynamics relabeling.

## 5. Paired Displaced-State Bank

If the pilot passes, generate exactly one Gaussian history-pair direction for
every train center `956--1193`, with deterministic seed `20260902`, and retain
both antithetic signs. This gives `476` paired displaced inputs. Any
input-admissibility redraw is deterministic, occurs before SU2 execution, and
is logged; a solver failure stops the bank.

For an undisplaced pair `z_n`, displacement `eta`, clean successor
`u_(n+1)`, and fixed SU2 BDF2 map `Phi`, train:

```text
PAIRED_RECOVERY:  z_n + eta -> u_(n+1)
DYNAMICS_RELABEL: z_n + eta -> Phi(z_n + eta).
```

Both arms use the same center order and sign schedule, with each sign used in
exactly 50 of 100 epochs. Both average `0.5` clean loss and `0.5` displaced
loss. Thus the comparison changes only the target information.

## 6. Pushforward Contracts

For BDF2 pair `x`, prefix depth `m`, and model `Psi`, all prefix calls are under
`no_grad`; only the terminal call is supervised against the aligned stored
clean future.

`MP_PDE_PUSHFORWARD_M01` uses `m=0` in epoch 1 and samples
`m uniformly from {0,1}` once per minibatch thereafter. Prefixes use the
current online model. There is no added noise, EMA generator, intermediate
loss, or inference-time corrector.

`CURRICULUM_EMA_PUSHFORWARD_K13` uses

```text
p_e = 0.5 * min(1, e / 10).
```

With probability `1-p_e`, use clean depth `m=0`; otherwise sample
`m uniformly from {1,2,3}`. Prefixes use an EMA copy with decay `0.995` in
evaluation mode. Update EMA after each optimizer step. Report both online and
EMA one-step/rollout results; the EMA deployment is primary, and a matched
`CLEAN_EMA` control isolates the weight-averaging effect.

Sample one requested depth per minibatch in both pushforward arms. At the first
three train centers, cap each example by the clean prefix history actually
available inside the train role and record requested and realized depths.

Log depth counts, prefix displacement by depth and field, pre-clipping gradient
norms, EMA/online separation, model calls, wall time, and peak memory.

## 7. PCNO PDE-Refiner Contract

Use the released PDE-Refiner DDPM/velocity-prediction formulation with one
shared PCNO. The candidate variable is the normalized next-state residual
already used by the baseline. Append the candidate and a four-channel one-hot
scheduler index to the two BDF2 histories; retain the same PCNO layers and
Fourier modes. Only the lifting layer grows, by `9*128=1152` weights.
Initialize from the same seeded PCNO parameter law as the parent learned arms
and set the output head exactly to zero; do not transfer a clean checkpoint
whose residual output has a different velocity-prediction meaning.

Sample scheduler time `t uniformly from {0,1,2,3}` per example. Corrupt clean
candidate `y` as

```text
y_t = sqrt(alpha_bar_t) y + sqrt(1-alpha_bar_t) epsilon
```

and predict

```text
v_t = sqrt(alpha_bar_t) epsilon - sqrt(1-alpha_bar_t) y.
```

The train-only parent error in this same normalized-residual coordinate has
equal-entry RMS `sigma_min=0.05748670867`; therefore set the minimum DDPM beta
to `sigma_min^2`. Use the published exponential four-level beta schedule and
do not tune it from rollout.

Inference starts from Gaussian noise and makes four reverse scheduler calls,
corresponding to three refinements after the initial generation. Only the
final candidate enters the BDF2 recurrence. Track and deploy EMA weights with
decay `0.995`. Select checkpoints using fixed-seed development one-step
refinement error at the first registered sampler seed `101`, never rollout.
Evaluate at three frozen inference seeds,
reuse the same noise tapes in paired response diagnostics, and report
variation, four-call latency, peak memory, intermediate path distances, and
validity.

This is a PCNO adaptation of the released PDE-Refiner algorithm, not a claim
of byte-for-byte reproduction of its conditioned U-Net.

## 8. Evaluation And Predictions

Use the parent development anchors, horizons `1/8/35/104/208`, full traces,
state/path/high-frequency/boundary diagnostics, and cost accounting. Recompute
common metrics with one extension evaluator; never splice scalar summaries from
different evaluator versions.

Freeze these qualitative predictions before development rollout:

1. both new pushforward arms should reduce the abrupt gradient/displacement
   pathology of the always-on arm; deeper curriculum exposure may still harm
   clean dynamics;
2. paired recovery should return more strongly toward the clean path, while
   relabeling should better match the measured SU2 displaced response; their
   rollout ranking is problem-dependent;
3. PDE-Refiner should change the candidate-error/high-frequency landscape but
   may be limited by PCNO's ability to represent injected fine-scale noise; and
4. no rollout gain supports the proposed mechanism unless its declared
   mediator changes without unacceptable path or validity harm.

## 9. Execution Gates

1. Pass focused CPU tests for restart writing, BDF2 alignment, target semantics,
   prefix schedules, stopped gradients, EMA updates, refinement steps,
   stochastic replay, recurrence, and cost counts.
2. Pass an independent code-to-intent audit.
3. Freeze source, parent, data, calibration, pilot, query-bank, training, and
   evaluator hashes.
4. Run full-resolution resource smokes before scientific training.
5. Use only train and development roles.
6. Close and independently verify the extension result packet before changing
   the manuscript's empirical claims.
7. Request separate owner decisions for prospective and sealed access.

Primary sources: MP-PDE, <https://arxiv.org/abs/2202.03376>; PDE-Refiner,
<https://arxiv.org/abs/2308.05732>.
