# W26-L2 Representation Routing Preregistration

Status: registered implementation pilot; GPU dispatch held while active D094
B1-C2 replication occupies the project queue.

Date: 2026-08-24

## Question and claim boundary

Can one fixed, bounded, invertible representation change reduce the known
three-call moving-front wake of the current PCNO without changing the physical
task, architecture, population, optimizer recipe, or recurrence?

This is empirical representation engineering. A positive result would not by
itself establish an optimizer mechanism, spectral-bias mechanism, or general
long-horizon PDE claim. A negative result closes only this treatment.

The deprecated Fractal-White-FNO project repeatedly suggested that paired
spectral preprocessing/postprocessing can alter endpoint error. Its strongest
retained Burgers setting smoothed the backbone input and inverted that change at
the output. It did not establish why: evaluation reused a test population
during training, recurrence was absent, coordinates and normalization were
flawed, and some controls mixed wrapper numerics with the treatment.

W26-L2 is a controlled routing task with an exact analytic population, a known
phase-sensitive recurrent wake, and maintained shock and smooth diagnostics. It
requires no external data or sealed population.

## Frozen task

- Native `64 x 32` structured cell averages.
- Existing 48 training and 48 held cases: step, pulse, smooth-tanh, and
  smooth-sine families over the registered anchors and phases.
- Exact physical increment target and physical-space relative loss.
- Three physical recurrent calls. Each physical residual is added to the
  physical state before preprocessing the next call.
- Primary stratum: held phase `0.875`, step and pulse, recurrent call 2.

## Frozen representation arms

All arms share the same initialized backbone within a seed. Coordinates,
quadrature, graph geometry, Fourier tensors, targets, and loss are unchanged.
Only the scalar state input and scalar residual output may pass through the
fixed wrapper.

1. `native`: exact bypass; no DCT is evaluated.
2. `dct_pre_half_smooth`: preprocess the scalar input with the fixed operator
   below, while the backbone predicts the physical residual directly.
3. `dct_coupled_half_smooth`: preprocess the scalar input and decode the
   backbone residual with the inverse operator before physical loss/recurrence.

For the two nonnative arms, use the existing radial cosine transfer `H(q)`,
equal to one through `q=8`, a cosine transition for `8<q<16`, and zero from
`q=16`. Freeze `G(q)=0.5+0.5H(q)`, so `0.5<=G<=1` and `1<=G^-1<=2`.
The backbone receives `DCT^-1(G DCT(u))`; only the coupled arm maps its output
back with `DCT^-1(G^-1 DCT(r_tilde))`.

The coupled treatment is invertible up to floating-point roundoff and neither
treatment adds learned parameters. A unit-gain DCT encode/decode is a required
numerical and autograd closure test, not a trained arm. The bounded inverse
avoids transferring the old one-dimensional alpha or large inverse gain into a
new basis and domain.

## Backbone, stages, and budget

The routing screen uses the full maintained PCNO because it has the target
wake: width 128, four blocks, decoder width 128, `k_max=8`, AdamW at `1e-3`
to `1e-5` under the existing cosine schedule, final-update selection, and seed
1701. The no-gradient model is conditional follow-up, not part of the screen.

- M0: all three reduced CPU smoke arms for two updates. Non-scientific.
- M1: all three full-PCNO arms for 5,000 updates at seed 1701. Exploratory;
  estimated near 17 RTX 5090 GPU-minutes, subject to a fresh timing smoke.
- M2: only after a passing M1, fresh 20,000-update runs for all three arms at
  seeds 1701, 1702, and 1703. Ceiling approximately 3.5 RTX 5090 GPU-hours.

M1/M2 may run only after active D094 compute clears or on an independent idle
GPU. Historical W26-L2 checkpoints are not controls because current core PCNO
source differs; every arm must be retrained under the new identity.

## Metrics and routing gate

Primary metric: mean recurrent call-2 relative increment L2 on held phase
`0.875`, pooled over step and pulse.

Also retain teacher/recurrent increment and next-state L2; active-front, wake,
and ahead errors; fresh/propagated energy and cross term; overshoot,
undershoot, oscillatory mass/lobes, total variation, front and integral errors;
smooth controls; and physical DCT-band summaries. Analytic truth is the sole
ground truth.

The best nonnative arm advances from M1 only if:

1. primary error is at least 5% lower than native;
2. the other nonnative arm is still reported, so input-only and paired behavior
   are not conflated;
3. smooth-sine call-2 next-state L2 is at most 5% worse than native;
4. no discontinuous structure metric develops a new material failure.

M2 supports a bounded result only if primary improvement has the same sign at
all three paired seeds and structure controls remain admissible. Report raw
paired values; three seeds do not support a broad significance claim.

## Provenance and next route

Each run retains canonical config/population digests, Git and environment
state, executable/provenance hashes, paired initial-backbone hash, terminal
checkpoint, parseable history and recurrent rows, physical predictions, and a
hash-covering manifest.

After positive M2 evidence, the next information-adding test is the clean
20-call Burgers recurrence reconstructed from the old trajectories with reused
IDs quarantined. D094 stays separate: first decide whether its retained path
displacement is generated by learned residual response or mostly carried by the
residual map's identity term.
