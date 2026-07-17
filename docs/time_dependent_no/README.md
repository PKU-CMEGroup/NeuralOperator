# Time-Dependent Neural Operators

This branch contains the summer 2026 time-dependent neural-operator work inside
the PKU-CME `NeuralOperator` codebase. The current weekly focus is Idea 2.1, the
solver-facing representation diagnostic: use fast 1D Euler experiments to
separate state coordinates, predicted quantity, supervision graph, and
enforcement mechanism before transferring a supported method to CPG-style and
dynamic 2D shock settings. The accepted ordering and stop rules are in
`RESEARCH_DIRECTION_DECISION.md`.

The tracked code is intentionally lean. Historical one-off probes and scaffold scripts were removed from the active tree after their conclusions were recorded in `MECHANISTIC_DIAGNOSTIC_TRACKER.md`; recover them from git history if an old result must be reproduced exactly.

## Active Code

Reusable branch utilities live in `utility/time_dependent_no/`:

- `fv.py`: PDE-agnostic finite-volume geometry, gather/scatter, and conservative update helpers.
- `euler1d.py`: 1D Euler primitive/conservative conversion, fluxes, geometry, and batch helpers.
- `euler1d_data.py`: collaborator-compatible 1D Euler dataset loading and batching.
- `euler1d_models.py`: FNO target heads, deprecated CPG-style pilots, and the corrected solver-level 1D CPGNet adaptation.
- `euler1d_targets.py`: state/residual, flux, and interface target adapters.
- `euler2d.py`: CPG HDF5 schema inspection, primitive/conservative conversion, node-type helpers, and graph-frame materialization.
- `euler2d_synthetic.py`: deterministic CPG-style synthetic fixture for CPU tests.
- `euler2d_metrics.py`: rollout, positivity, conservation, shock-proxy, boundary, and compact-summary diagnostics.
- `euler2d_fixture.py`: end-to-end no-model fixture diagnostics.
- `errors.py`: small NumPy error helpers used by diagnostics.

Active command-line entry points live in `scripts/time_dependent_no/`:

```bash
python scripts/time_dependent_no/euler1d_weno_hllc_rk3_dataset.py --help
python scripts/time_dependent_no/euler1d_weno_hllc_ader_dataset.py --help
python scripts/time_dependent_no/train_euler1d_target_ladder.py --help
python scripts/time_dependent_no/analyze_euler1d_target_ladder.py --help
python scripts/time_dependent_no/diagnose_euler1d_generated_state_consistency.py --help
python scripts/time_dependent_no/diagnose_euler1d_scale_spectra.py --help
python scripts/time_dependent_no/probe_euler1d_conservative_dissipation.py --help
python scripts/time_dependent_no/generate_euler1d_rollout_animations.py --help
python scripts/time_dependent_no/run_euler_fixture_diagnostics.py
python scripts/time_dependent_no/diagnose_cpg_rollout_mechanisms.py --help
python scripts/time_dependent_no/visualize_official_cpg_rollout.py --help
python scripts/time_dependent_no/visualize_cpg_shock_overlays.py --help
python scripts/time_dependent_no/rollout_pcno_preprocessed.py --help
```

`rollout_pcno_preprocessed.py` is the corrected PCNO evaluation path. It assumes the collaborator-compatible preprocessing contract: HDF5 trajectories are converted to per-trajectory arrays, reconstructed into the PCNO Euler `.npz` format, and then rolled out. The retired raw-HDF5 PCNO adapter path should not be used for checkpoint evaluation.

Tests live in `tests/time_dependent_no/`.

## Current Dataset State

The copied dataset folder label was `forward_300`, but extracted files identify the supersonic bump dataset:

- 300 train trajectories and 20 test trajectories;
- 80 HDF5 time steps per trajectory;
- roughly 19k to 23k nodes per trajectory;
- expected CPG keys are present;
- extracted test cases contain `Bump.jl`, `Bump.msh`, `Bump.inp`, `Mach.txt`, `params.txt`, and VTU snapshots.

See `docs/time_dependent_no/BUMP_300_DATASET_AUDIT.md` and `docs/time_dependent_no/CPG_EULER_DATASET_CONTRACT.md` for stable schema facts. Exact AutoDL paths remain in ignored local context.

## Current Diagnostic Readout

CPGNet learns accurate teacher-forced one-step updates but fails in autoregressive rollout, mainly around shock-local phase, shape, amplitude, and stability rather than simple one-step underfitting. PCNO with the corrected preprocessing path initially follows the solution better visually but develops Fourier-style ripples that can trigger rollout crash.

The conservative-coordinate FNO matrix, exact ADER cumulative-face-impulse
export, and first direct/joint flux-supervision screen are complete. The label
closes correctly, and gauge-canonical joint supervision can fit the identifiable
face field. At 64/16/16 stride-1 scale, however, it reduces active-flux MSE while
worsening held-out state error and failing every paired raw rollout earlier than
the state-loss-only conservative flux head. This exact face-value loss is
stopped before full scale; it is a label-valid decoded-state and recurrent-
stability failure, not an implementation or closure failure.

The matched short-unroll comparison is also complete. At 64/16/16 scale,
four-step autoregressive fine-tuning improved the state-loss-only conservative
flux head's test one-step relative L2 from `0.00461` to `0.00344`, mean raw
survival from `0.519` to `0.759`, and completion from 0/16 to 2/16. A matched
smooth training-only admissibility barrier produced essentially the same result
and did not reduce the 14 nonpositive terminations, so it failed its attribution
gate. Inference remained the exact conservative flux update without a limiter
or floor.

The promoted 384/64/64 confirmation used the 316,739-parameter FNO, 240,000
one-step optimizer updates, then 46,560 four-step recurrent updates. Its selected
test checkpoint reaches one-step relative L2 `0.001305`, mean raw survival
`0.97734`, and 57/64 completed 20-call rollouts at mean initial effective CFL
`3.84`. The seven remaining raw failures are all nonpositive proposals. This
passes the scale gate but misses the strict 90% completion gate by one
trajectory. A frozen-checkpoint extension then exposes the horizon limit: only
1/64 cases completes 50 raw calls (mean survival `0.566`), and 0/64 completes
100 calls. The 20-call row is therefore a strong short-horizon result, not a
medium-horizon or unconstrained-positivity solution.

The strict next-state/residual target control is now complete through full scale,
three initialization seeds, and 100-call evaluation. Direct next conservative
state fails its preregistered tiny-fit gate on both declared seeds, while the
information-equivalent conservative residual passes, isolating an output-
centering/identity-bypass optimization effect. The 316,419-parameter residual
FNO has three-seed mean one-step relative L2 `0.001136`; pooled completion is
192/192, 187/192, and 182/192 at 20, 50, and 100 raw calls without an inference
limiter, floor, or positive transform. All ten seed-case terminations at 100
calls are raw pressure failures.

The matched 316,806-parameter projected-residual ablation factors the increment
into a volume-zero spatial field plus one learned boundary budget and closes to
that budget at about `1e-7`. It improves pooled 100-call completion to 187/192,
but the mean state-error ratio is `1.016`; among common completers it is 9.8%
worse in state L2, 5.9% better in conserved-total error, and 4.8% worse in shock
MAE. Only the preregistered stability-noninferiority gate passes. Residual is the
strong fixed-setting accuracy baseline; projection remains a useful structural
ablation, not evidence of general target superiority or unconditional
positivity.

A matched 64/16/16 generated-state exposure pilot then tested whether the
projected residual's accuracy deficit was caused by seeing states that were too
close to the data manifold. The control used four-step BPTT; the intervention
first rolled eight detached model steps and supervised the following four. The
50 one-step histories match exactly and the recurrent optimizer-update counts
differ by less than 1%. Burn-in reduces 100-call final state L2 by 36%,
conserved-total error by 46%, and shock MAE by 6%; it wins the paired common-
endpoint state comparison on 14/16 cases. It also reduces 50-call state L2 by
23% and wins 15/16 paired cases. The preregistered gate nevertheless fails:
20-call shock MAE is 38% worse, and mean 100-call survival falls from `0.9650`
to `0.9394` even though both rows complete 14/16 cases. Burn-in rescues one
clean-control pressure failure but creates a different failure at call 58 and
moves another from call 50 to 47. This is a partial distribution-exposure
result, not a full-scale promotion of projected residual.

The same gate is now complete for plain residual. Generated burn-in improves
final state L2 by 4.6% at 20 calls and 32% at 50 calls; at 100 calls it lowers
the common-endpoint state error by 51.4%, wins 14/16 paired cases, and raises
completion from 15/16 to 16/16. It also lowers conserved-total error at every
horizon. The formal gate still fails because the legacy single-argmax shock
MAE is 45.6% worse at 20 calls. Focused replay shows that two dominant
contributions are front-strength rank changes, while one case has a genuine
spurious/displaced-front regression. This is strong evidence for later-state
exposure, but the formal clean-versus-generated gate remains failed.

The matched teacher-offset control is now complete. It uses the identical 8+4
windows and recurrent update count but starts supervision from the exact
reference state after the eight-step offset. Teacher offset is 10.9% worse than
clean at 20 calls and statistically indistinguishable at 50/100 calls; it
finishes 14/16 H100 cases versus 15/16 clean. Generated exposure, by contrast,
reduces common-endpoint state error relative to teacher by 13.9%, 33.9%, and
50.8% at 20/50/100 calls, wins 15/16 H100 cases, and finishes all 16. This
supports a fixed-setting causal claim that off-manifold generated states add
value beyond later physical-time sampling. D023 now shows that this is not a
large local PDE-map improvement on matched generated states. At prefix depth
eight, generated/teacher fixed-scale conservative errors are `0.992` to the
original next state and `1.017` to the same-state solver continuation; the
correction-to-trajectory cosine is only `0.074`. Clean is 2.6% closer to the
solver continuation than generated, and generated is 8.0% worse in the
shock-region continuation defect. The solver replay baseline is negligible
and no replay needs retry or fallback. Thus the preregistered result is mixed:
generated-state exposure improves closed-loop dynamics without making the
one-step map materially more solver-consistent on this state bank. Full-scale
promotion remains paused.

The frozen scale diagnostic now identifies the state-loss-only flux failure as
recurrent high-frequency growth, not deficient teacher-forced spectral fit.
A paired conservative Laplacian-flux probe then produces no material H50
stability gain: small coefficients worsen state and Nyquist-tail error, while
larger coefficients shorten survival. D025 is now complete and negative for
the tested global relative-interface parameterization. At 64/16/16 scale,
frame-zero weighting cuts selected test one-step relative L2 from `0.0154` to
`0.00791` and sharply improves initial shock/front error, but H50 completion
remains 0/16. A training-only admissibility barrier at weight `0.1` also gives
0/16 and does not improve mean survival. Four-step training fails only after a
generated state has already become inadmissible, while the raw FNO output is
still finite. This is a recurrent conditioning/admissibility failure, not an
obvious decoder implementation bug or a receptive-field limitation. Do not
promote this row to full scale or repair it with an inference limiter.

An identifiable boundary-exchange auxiliary has also completed its matched
projected-residual gate. RMS-normalized weights `0.01` and `0.1` reduce the
one-step boundary-exchange error by 19.8% and 56.6% and the H20 conserved-total
error by 36.7% and 72.5%. Both preserve 16/16 completion, but they worsen
one-step state error by 15.2% and 33.7%, H20 state error by 14.6% and 19.8%,
and shock-position error by 24.8% and 20.8%. This is a clean structural Pareto
result, not evidence that the auxiliary adds missing information: in 1D the net
exchange is already recoverable from the endpoint balance. Stop coefficient
search and keep plain residual as the accuracy baseline for stride/resolution
transfer.

The first strict stride-transfer gate is now complete. A separately trained
stride-2 residual FNO uses the same 316,419 parameters and beats repeated
stride-1 composition by 25.0%, 36.2%, and 48.6% in common-case state error at
physical frames 20, 50, and 100, while completing all 16 H100 rollouts. This is
direct evidence that a global fixed-step neural operator can learn a useful
larger-step propagator at effective-CFL maxima up to about 13.3. It is not a
claim of CFL-free or timestep-conditioned inference: direct frame-2 error is
1.216 times small-step composition, and a different model is trained per
stride.

Weight-only stride-1 to stride-2 continuation repairs that initial jump and
lowers final-target training floors by more than 40%. It also improves H100
state error and raises the minimum pressure from 0.00193 to 0.0452. The gain is
not uniform: H50 state and shock errors are 1.095 and 1.565 times the cold
stride-2 row. Preserve cold stride 2 as the large-step accuracy baseline and
continuation as a fit/stability tradeoff ablation. The conditional
total-exposure control is stopped by successive halving.

Frozen-checkpoint resolution transfer is now complete. Native-grid zero-shot
one-step errors at 128 and 512 cells are 5.5 to 8.3 times the corresponding
256-cell values, and the 512-cell H20 shock metric worsens by 2.10 times for
stride 1 and 2.99 times for cold stride 2. The mismatch persists across input
times. Along separately evolved paired native trajectories, update labels
differ by 28% to 58% of the coarse increment norm, so the current numerical
residual target is materially grid dependent. This is not an exact same-input
commutator test, but it is evidence against mesh invariance of the present
target contract rather than evidence of an FFT implementation failure.

The larger-step comparison does transfer. Cold stride 2 beats stride 1 at
H50/H100 on both off-grid resolutions; its H100 common-case state-error ratios
are 0.474 at 128 cells and 0.523 at 512 cells without lower common-case
completion. The result is a stable large-step advantage without native
resolution transfer.

The follow-up restriction-consistent gate is complete. One shared 316,419-
parameter residual FNO, trained with the same total sample presentations as one
single-grid row, stays within `1.296x` one-step and `1.418x` H20/H50/H100 state
error of all three 128/256/512-cell oracles. It never loses completion relative
to an oracle and passes the H20 shock and conservation gates. Relative to the
frozen native-nx256 checkpoint, its off-grid one-step error falls by 68.1% at
nx128 and 78.3% at nx512 on common restriction-consistent truth. This shows
that a shared FNO can learn one identifiable cell-average flow map across these
resolutions without an explicit cell-width channel.

It is not a native-solver-invariance result. On independently evolved native
coarse trajectories, the shared model's one-step error is 5.27 times its
restriction-consistent value at nx128 and 3.65 times at nx256; the native
targets differ by several percent even at frame zero. The row is classified as
`shared_restriction_operator_without_native_solver_equivalence`. It also loses
the same three pressure-limited cases by H100 on every resolution. Do not add
cell width or more modes merely to repair D029; next isolate a training-only
tail-stability intervention on the already well-fitted residual baseline.

Target-family exploration remains an empirical program: first run a strict
matched screen, then give surviving families an equal-budget best-engineered
comparison. Classify negative results as label, fit/optimization,
generalization/closure, or recurrent-stability failures before rejecting a
family. `RESEARCH_DIRECTION_DECISION.md` owns the complete gates, target set,
and progressive-stride protocol.

The old 1D rows labeled CPGNet used a generic directed residual head and are
deprecated. The corrected `cpg_interface` baseline reconstructs positive
directed interface states, forms one Rusanov flux per face, applies the exact
finite-volume update, and evaluates raw recurrence without a cell-state
limiter or hidden floor. Its CPU implementation gate passes; the benchmark GPU
run and parameter-matched controls are complete. The deep mp28 row is the frozen
strong reference; the controls support receptive-field depth rather than
parameter count as the main gain.

Pending 2D diagnostics should still measure ripple energy and shock-front geometry using the common rollout artifact contract: `predicteds`, `targets`, `pos`, `edges`, and `node_type` when available. They are background constraints for the bump transfer rather than the immediate blocker for the 1D target selector.

## Not Ported Or Active

The raw `cpggnspdes` training scripts are not vendored into this branch. The CPGNet interface-latent probe code, state-drift/perturbation/time-alignment scripts, early smoke scripts, and raw-HDF5 PCNO adapter path were one-off research tools and are no longer active tracked code.

FNO, PCNO, and MPCNO baselines should use implementations already present in this repository or collaborator-provided preprocessing artifacts. Do not port baseline code from earlier data-assimilation experiment repositories.
