# Mechanistic Diagnostic Tracker

Date: 2026-07-04
Evidence frozen through: 2026-07-23
Consolidated: 2026-07-26
Status: Frozen evidence ledger; not an experiment queue

This file preserves the historical experiment contracts, results, and stopping
decisions. Status words and forward-looking language inside dated entries record
what was true at that point in the campaign; they do not authorize current
work. The current queue is report-only. Current method-design and authorization
precedence is `RESEARCH_DIRECTION_DECISION.md`, then `HANDOFF.md`. Retired
one-off implementations are recoverable from pre-cleanup commit `729091b`.

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D001 | M0 | Recompute current summaries from rollout arrays | CPGNet one-step bs2 | bump test full | paper RMSE, per-time RMSE, VPT | MUST | DONE | Full 20 trajectories analyzed on AutoDL under `artifacts/time_dependent_no/cpg_mechanistic_diagnostic_20260704_full/`; reproduced user-provided AR summary (`rho=0.200242`, `pres=0.384017`). |
| D002 | M0 | Recompute current summaries from rollout arrays | CPGNet two-stage bs2 | bump test full | paper RMSE, per-time RMSE, VPT | MUST | DONE | Full 20 trajectories analyzed on AutoDL; reproduced user-provided AR summary (`rho=0.158637`, `pres=0.293593`). |
| D003 | M1 | Teacher-forced vs autoregressive error by time | CPGNet one-step and two-stage | bump test full | TF RMSE, AR RMSE, AR/TF ratio | MUST | DONE | Per-time teacher-forced evaluator run on AutoDL. TF remains small over time; AR/TF ratios are ~22-26x for one-step and ~16-28x for two-stage depending on variable. |
| D004 | M1 | State-distribution drift | CPGNet one-step and two-stage | bump test/train stats | z-score drift, range violations | MUST | DONE | Sampled train stats from 30 train trajectories at stride 10. Final predictions almost never leave sampled train min/max ranges except tiny pressure fractions; gross range-OOD drift is not the main explanation. |
| D005 | M1 | Perturbation amplification | CPGNet one-step and two-stage | selected bump test | amplification factor, excess target RMSE | MUST | DONE | Sampled AutoDL probe over 5 trajectories, 4 start frames, two perturbation scales, and 5 channel modes. One-step is mostly locally damped; two-stage is more velocity-sensitive, but induced excess target RMSE is tiny versus full AR errors. |
| D006 | M2 | Shock position and region decomposition | CPGNet one-step and two-stage | bump test full | shock-front IoU/F1, Chamfer/front distance, centroid, near-shock/smooth error, thickness, strength | MUST | DONE | Full q=0.85/0.90/0.95 shock diagnostics run on AutoDL. Shock-local error dominates smooth error, but best-shift alignment explains only a small fraction, so failure is shock-local shape/amplitude/stability plus phase, not pure displacement. |
| D007 | M2 | Rollout animations and overlays | CPGNet one-step and two-stage | selected bump test | qualitative phase/smear/drift labels | MUST | DONE | Generated 32 selected-subset pressure/shock overlay PNGs plus JSON/Markdown gallery under `artifacts/time_dependent_no/cpg_shock_overlay_gallery_20260704/`. |
| D008 | M3 | Equal-node physics diagnostics | CPGNet one-step and two-stage | bump test full | conservation drift, positivity, TV proxy | MUST | DONE | Equal-node normal-node total mismatch, positivity, and clamped-boundary leakage computed for full split. No positivity failures; boundary error is exactly zero under clamped rollout. |
| D009 | M3 | Approximate-weight physics diagnostics | CPGNet one-step and two-stage | bump test | weighted conservation drift | NICE | LEGACY CLOSED / NOT RUN | Original block: mesh/cell weights or a validated geometric approximation were unavailable, so diagnostics intentionally reported equal-node totals only. The campaign closed without this run; equal-node or approximate bump sums are not physical conservation evidence. |
| D010 | M4 | Direct state-predictor control | Direct GNN or simplest available comparable model | bump train/test | one-step/rollout error, perturbation, shock-region, and structure diagnostics | NICE | LEGACY CLOSED / NOT RUN | Original block: wait for a dominant defect before training a control. The campaign closed without this run, and it is not current authorized work. |
| D011 | M5 | PCNO diagnostic replay | PCNO completed checkpoint via corrected preprocessing contract | selected bump test | AR RMSE, shock masks, positivity, boundary leakage, animations | MUST | DONE | D019 completed for trajectories 0, 6, 11, 13, 17 using the corrected HDF5-to-npy-to-reconstructed-npz path. The retired raw-HDF5 graph adapter should not be used for model conclusions. |
| D012 | M6 | Correlation-time and geometric rollout aggregation | CPGNet one-step and two-stage | bump test full | high-correlation time, geometric relative error aggregation | MUST | LEGACY CLOSED / NOT RUN | The original plan was to add APEBench/PDE-Refiner-style temporal metrics to existing raw-array reports. The campaign closed without this standalone aggregation, and it is not current authorized work. |
| D013 | M6 | Scale/spectral residual diagnostics | 1D residual FNO, flux FNO, and CPGNet mp28 first; then 2D CPGNet/PCNO | frozen 1D split, then bump full/subset | divergence-active flux spectra, first/second differences, characteristic and shock/smooth splits, pre-failure high-frequency growth; graph-native bins in 2D | MUST | DONE | The frozen 1D flux result and frozen serious-PCNO bump result both identify recurrent high-frequency growth. In 2D, rollout/teacher smooth high-band energy is `13.26x` at the late call, rollout growth is `45.28x`, and error-direction perturbation gain is `1.094`; the learned spectral branch is much smoother than the local branches. The repeated paired-response extension historically selected `local_pointwise_control`; that selector is closed and authorizes no current experiment. Use graph-native evidence, not interpolation-to-grid FFT. |
| D014 | M6 | Effective-CFL / receptive-field audit | CPGNet first, then PCNO | bump test full/subset | shock-front motion per step, median-edge-length units, message-passing/hop coverage, correlation with hard trajectories | MUST | DONE (partial) | Line 2 reports exact architectural current-state radius 13 and zero of 800 endpoint-sampled characteristic rows outside support. Unresolved DG substeps remain unbounded. A finite-hop obstruction does not apply to full PCNO because every spectral branch has global dependence. |
| D015 | M7 | Recurrent/unrolled stabilization control | CPGNet or PCNO after D013/D014 | bump train/test | TF error, AR error, VPT, shock metrics, scale residuals | NICE | DONE (bounded pilot) | One detached depth-one generated-state-exposure continuation retained 5/5 admissible completion and improved selected rollout error only `2.3%` over clean, below the `10%` promotion gate; later epochs regressed. Do not launch the full confirmation or a weight sweep. |
| D016 | M6 | Interface-state latent instrumentation | CPGNet one-step and two-stage bs2 | selected bump test frames | `reconstruct_prims`, one-sided trace metrics, LLF central/dissipation split, induced FV update, wave-type strata | MUST | DONE | Historical probe completed. The one-off implementation was retired from the active tree after the result was recorded; recover from git history only if exact reproduction is needed. |
| D017 | M6 | Interface-state latent selected-frame run | CPGNet one-step and two-stage bs2 | trajectories 0, 6, 11, 13, 17; frames 0, 20, 40, 58, 78 | admissibility, trace-likeness, flux/update match, dissipation localization, speed projection, sampled edge table | MUST | DONE | AutoDL run completed under `artifacts/time_dependent_no/cpg_interface_latent_diagnostic_20260705_full/`; latents are admissible but not physical one-sided traces, induced update matches model delta but not true update exactly, and dissipation is only weakly shock-localized. |
| D018 | M6 | Interface-latent mechanism probe | CPGNet one-step and two-stage bs2 | selected trajectories/frames; teacher-forced and autoregressive state sources | physical projection sensitivity, constrained inverse flux fit, TF-vs-AR latent drift | MUST | DONE | AutoDL run completed under `artifacts/time_dependent_no/cpg_interface_mechanism_probe_20260705_full/`; physical projections do not preserve learned flux/update, constrained physical inverse fits remain poor, and AR mode greatly increases learned-update error against the target next state. |
| D019 | M5 | PCNO corrected preprocessed rollout replay | PCNO Euler checkpoint through collaborator-compatible preprocessing | trajectories 0, 6, 11, 13, 17 | AR RMSE, positivity, velocity blow-up, GIF gallery | MUST | DONE | AutoDL selected replay completed under ignored corrected PCNO rollout artifacts. Visual readout: PCNO initially tracks shock position better than CPGNet, but Fourier-style ripples grow and can trigger long-rollout crash; pressure mean RMSE across selected trajectories is about 2.13 and velocity errors can overflow. |
| D020 | M6 | 1D Euler effective-CFL / receptive-field intervention | corrected CPGNet h128, mp12 versus mp28 | 384/64/64, stride 4, frame 80 | one-step fit, raw completion, survival, CFL correlations, shock, conservation | MUST | DONE | mp28 completed 64/64 raw test rollouts versus 34/64 for mp12; first-rollout-step error fell 88.6% and the initial-CFL/error Pearson correlation fell from 0.90 to -0.12. The initial depth/capacity confound was later closed by the mp12/h193 and mp28/h85 controls, which support hop coverage as the primary mechanism. |
| D021 | Idea 2.1 | Staged target-family optimization and solver screen | coordinate-selected FNO: next state, residual, state-loss-only flux, direct cumulative impulse, and joint supervision | frozen 1D ADER split, then stride ladder | label/endpoint/boundary closure, tiny-set fit, training floor, seed variance, supervised-objective/decoded error, shock/smooth curves, direct horizon, raw rollout, physics, transfer | MUST | DONE | The bounded target, exposure, stride, and resolution program is complete through D031. The 64/24/4 conservative-variable residual FNO remains the strong fixed-setting baseline. Rejected target, constraint, and continuation rows are closed; future Line-1 work is limited to frozen evaluation and the bounded theory package. |
| D022 | Idea 2.1 | Separate later-time sampling from generated-state exposure | plain-residual FNO: clean 0+4, teacher-offset 8+4, generated 8+4 | fixed 64/16/16 split, seed 20260708 | matched one-step history/update count, H20/H50/H100 raw rollout, common-endpoint error, conservation, top-two front position/strength | MUST | DONE | Teacher offset gives no H50/H100 state benefit over clean. Generated exposure beats teacher by 13.9%, 33.9%, and 50.8% at H20/H50/H100 and completes 16/16 versus 14/16 at H100. It still regresses top-two front position versus clean at H20/H50, so full-scale promotion was paused at that gate and later superseded by the completed D023-D031 closeout. |
| D023 | Idea 2.1 | Solver-consistency diagnostic from generated states | learned residual versus WENO-HLLC-ADER advancement initialized from the same generated state | all 16 D022 test cases, starts 0:10:80, prefix depths 0/2/4/8 | learned-vs-reference next-state defect, truth-next defect, shock/smooth and characteristic components, error-versus-prefix depth | MUST | DONE | At depth eight, generated/teacher error is 0.992 to original truth but 1.017 to the same-state solver continuation; correction alignment is only 0.074. Clean is closest to the solver. The result is mixed and rejects a large local PDE-map explanation for the rollout gain. |
| D024 | M6 | Frozen-checkpoint conservative-dissipation probe | state-loss-only flux FNO plus small local interior diffusive face flux | full frozen split | H20/H50 survival and positivity, front position/strength, shock width, TV excess, conservation and boundary exchange | MUST | DONE | The paired five-coefficient probe gives no material H50 stability gain. Small diffusion leaves completion unchanged while worsening state/tail error; larger diffusion shortens survival even when modes 25--64 decrease. Boundary correction is exactly zero. |
| D025 | Idea 2.1 | Global interface-latent FNO pilot | face-grid FNO with two relative directed traces, shared Rusanov or central decoder, exact FV update | tiny fit then 64/16/16 | one-step fit, raw H20/H50 rollout, positivity, conservation, top-two front geometry, decoder ablation | MUST | DONE (1D pilot) | The 317,126-parameter Rusanov model reaches selected test one-step relative L2 `0.00786` after frame-zero weighting and a training-only barrier, but every 16-case H50 rollout still becomes inadmissible within five calls. Short unrolling, temporal reweighting, and barrier weight `0.1` fail the `0.10` survival gate. Do not promote this parameterization to full scale or four-step training. |
| D026 | Idea 2.1 | Identifiable boundary-exchange supervision | projected-residual FNO plus RMS-normalized net solver boundary-impulse loss | matched 64/16/16 stride-1 gate, H20 | one-step state, boundary exchange, raw rollout, shock, conservation, closure | MUST | DONE | Weights `0.1` and `0.01` improve boundary-exchange and conserved-total errors but worsen one-step, H20 state, and shock errors. Both retain 16/16 completion but fail the joint-accuracy gate. Stop without a full seed sweep. |
| D027 | Idea 2.1 | Cold stride-2 transfer gate | plain-residual FNO, fixed stride 2, compared with composed frozen stride-1 model | matched 64/16/16, H20 selection then frame-100 replay | native fit, same-frame H20/H50/H100 state, survival, shock, conservation | MUST | DONE (partial) | Direct stride 2 improves common-case H20/H50/H100 state error by 25.0%/36.2%/48.6% and completes 16/16 at H100, but its direct frame-2 error is 1.216 times stride-1 composition and misses the 1.15 gate. This historically routed the now-completed D028 continuation control; D027 authorizes no current stride action. |
| D028 | Idea 2.1 | Stride-1 to stride-2 continuation gate | plain-residual FNO initialized from the frozen stride-1 weights, fresh stride-2 optimizer | matched D027 split/schedule, H20 selection then frame-100 replay | final-target fit floor, frame-2 defect, H20/H50/H100 state, survival, shock, conservation | MUST | DONE (partial) | Continuation repairs frame 2, lowers one-step/recurrent training floors by 42%/44%, and improves H100 state and pressure margin, but H50 state is 1.095 times cold and fails the 1.05 gate. Do not run the conditional total-exposure control or claim a uniformly better solver. |
| D029 | Idea 2.1 | Frozen cross-resolution transfer gate | frozen plain-residual stride-1 and cold stride-2 FNOs trained at 256 cells | identical 512 physical cases at 128/256/512 cells; frozen 16-case test split | native-grid one-step, frame-2/H20/H50/H100 raw rollout, shock, conservation, completion, solver restriction gap | MUST | DONE (partial) | Neither checkpoint passes native-map resolution transfer: off-grid one-step error is 5.5--8.3 times nx256, high-resolution shock metrics regress, and cold stride 2 loses one case off-grid. The larger-step advantage itself transfers bidirectionally: stride 2 beats stride-1 composition at H20/H50/H100 on nx128 and nx512 with equal same-grid completion. |
| D030 | Idea 2.1 | Restriction-consistent shared-resolution gate | one shared 64/24/4 residual FNO versus equal-presentation single-resolution oracles and the frozen native-nx256 baseline | exact-cell-average nx512 reference conservatively restricted to nx256/nx128; matched 64/16/16 split | label commutation, per-grid one-step, H20/H50/H100 raw rollout, shock, conservation, completion, native-solver diagnostic | MUST | DONE (partial) | The primary representation and frozen-baseline usefulness gates pass. The shared row stays within `1.418x` same-grid-oracle state error and `1.296x` one-step error, with no completion loss. It does not reproduce independently evolved native coarse-grid maps and still loses three pressure-limited cases by H100. Classify as `shared_restriction_operator_without_native_solver_equivalence`. |
| D031 | Line 1 | Full-split fixed-stride flow-map frontier | 64/24/4 conservative-variable residual FNO; cold strides 1/2/4/8 plus two stride-8 repeats | 384/64/64; seed 20260707 for all strides and 20260708--09 for stride 8; H32 selection | direct/composed H32/H64/H96 error, raw completion, shock, conservation, sample-presentation provenance | MUST | DONE | D032 corrected a prose horizon-labeling error in the initial closeout. At physical H96, stride 8 has pooled mean fixed-scale conservative relative L2 `0.01798` and 192/192 completion versus stride 4 at `0.02645`. The first stride-8 jump and shock-position error remain worse than stride-4 composition. This is a right-censored operating-envelope result, not a universal optimum or CFL limit. The six-run cap is exhausted. |
| D032 | Line 1A | Dense frozen reliability and checkpoint-candidate audit | selected plus epoch-50/60 candidates at strides 1/2/4/8 | frozen D031 split and checkpoints; H8--H96 | error-budget survival, integrated relative error, increment error, admissibility, shock, conserved-total mismatch, selection sensitivity | MUST | DONE | Zero learned training. Endpoint-error winners move from stride 2 at H8 to stride 4 at H16 and stride 8 at H32/H64/H96; time-mean winners are stride 8, stride 4, stride 4, stride 8, stride 8. Stride-8 repeats give H96 `0.01767--0.01821` with 192/192 completion. Epoch-60 stride 1 is much more accurate but loses one additional H96 completion; candidates remain validation-only and primary selections are immutable. The legacy `conservative_budget_relative_l2` field is a domain-total prediction/truth mismatch, not boundary-flux closure. |
| D033 | Line 1A | Multi-stride same-state solver decomposition and rescue test | extend D023 to frozen stride-1/2/4/8 models | D031 test states, truth states, and on-policy generated states | learned-versus-solver defect, truth defect, semigroup paths, per-physical-time amplification, oracle rescue, macro closure | MUST | DONE | Zero learned training and raw conservative recurrence. At frame 32, truth-state one-call error rises with stride while on-policy accumulated error falls, directly exposing the harder-map/fewer-calls tradeoff. One reference replacement changes H96 error by only about `-5.8%` to `+1.6%` and rescues no completion. The first max-only replay gate stopped before model evaluation; the documented mean/p99/max gate passed on the clean rerun. |
| D034 | Line 1A | Direct full-rollout timing and Pareto closure | frozen D031 models and documented reference solver on the designated local Line-1 host | same hardware, batch 1 and amortized batches | warm-up, synchronization, preprocessing/transfer boundaries, latency, throughput, accuracy-matched wall time | MUST | DONE | On the fixed 16-case H96 subset, stride-2/4/8 host-to-host batch-1 times are `60.40/30.75/15.78` ms and accuracy-matched speedups are `14.55x/65.53x/127.71x`. Stride-8 batch-16 throughput is 991 cases/s versus 0.277 for the matched reference. These are contract-specific measurements against the documented Python solver, not production-CFD claims. |
| D035 | Line 1B | Frozen error-geometry and visualization closeout | selected seed-20260707 stride-1/2/4/8 checkpoints under exact raw conservative recurrence | 64 frozen test cases through H96; truth-state starts 0:8:88 | L1/L2, shock/smooth, translation oracle, budget survival, phase-resolved same-state defect, physical descriptors, GIF gallery | MUST | DONE | Zero learned training. Truth-state one-call error rises with stride while defect per physical time falls at all 12 starts. At H96 stride 8 is best on global L1/L2, shock/smooth, and translation-aligned error; stride 2 is best on top-two-front position. The descriptor router scores `0.328` versus a `0.547` majority baseline and fails every gate, so adaptive routing is stopped. |
| D036 | Line 1B | Frozen ripple and roughness conflict | selected stride-1/2/4/8 checkpoints plus the two frozen stride-8 repeats | 64 frozen test cases at common H8--H96 endpoints | global error, Hann-windowed error spectra, away-front derivative error and TV, threshold-sensitivity survival | MUST | DONE | Zero learned training. At H96 the primary stride-8 row has `0.668x` stride-4 global L2 but `1.197x` away-front second-derivative error and `1.094x` away-front TV ratio; all paired-bootstrap intervals exclude equality. Under the diagnostic smooth-TV budget `1.5`, median first exceedance is H68 for stride 4 versus H48--H56 for the three stride-8 seeds. This confirms a metric-dependent error/ripple tradeoff, not a universal shorter physical lifetime. |
| D037 | Line 3 | Dynamic finite-volume conservative-correction oracle preflight | canonical Mach-1.1 shock--isentropic-vortex; consumed by the later D044 dynamic baseline | 250x100/500x200/1000x400 primary ladder plus one pinned boundary-matched SharpClaw 250x100 state run | reference convergence and saved-time provenance; independent state agreement; impulse identifiability and contraction; conservation, locality, update norm, raw admissibility, shock/vortex anti-smearing | MUST | DONE (CONTRACT CLOSED; CONSUMED BY D044) | The v3 audit closes the benchmark and direct same-primary-solver impulse contracts with all checks true. Primary state ratios are `0.484166/0.481540/0.397123`; SharpClaw final/time-mean errors `0.0315534/0.0156538` pass envelopes `0.0466208/0.0233192`; full/divergence-active/cycle/boundary impulse ratios are `0.463428/0.463362/0.463482/0.464003`. SharpClaw is state-only and the cycle field is discretization-specific. D044 later froze the perturbation split, neural baseline, D013, and rejecting oracle result; do not read those outcomes back into this reference-only row. |
| D038 | Line 1B | Frozen modal-error evolution | D031 stride-1/2/4/8 checkpoints plus both stride-8 repeats | 64 frozen test cases; H8--H96 rollout endpoints and truth starts 0:5:90 | truth-normalized error by mode, error-energy share, spectral-shape moments, pooled teacher-update bands | MUST | DONE | Zero learned training. Stride-8 teacher-update error exceeds stride 4 in every band, with a `2.465--2.744x` excess in modes 17--24 across its three seeds. At H8 the primary stride-8 total modal error is `1.218x` stride 4 and all 101 truth-resolved modes are worse. At H32 total modal error is already `0.917x`, while centroid, normalized `k^4` shape, and tail share remain `1.203/1.327/1.857x`; all three seeds preserve the rougher shape. At H96 total modal error is `0.522x`, but the global shape ordering is not seed-stable. This supports broadband large-step defect injection followed by a fewer-call advantage and localized late roughness, not global high-frequency blow-up. |
| D039 | Line 1B | Frozen operating-envelope closeout | registered D032--D038 source tables; no checkpoint replay | primary 64-case H8/H16/H32/H64/H96 endpoints; D034 fixed 16-case timing subset | cross-artifact provenance, metric-specific winners, paired case bootstrap, sustained crossover, global/local/runtime Pareto | MUST | DONE | Zero training, checkpoint evaluation, or solver calls. All ten provenance/mechanism invariants pass, including exact equality of 72 duplicated D036/D038 summary rows. Endpoint-error winners are stride 2 at H8, stride 4 at H16, and stride 8 at H32/H64/H96; only H16, H64, and H96 separate the winner from its runner-up under the paired 95% case-bootstrap interval. These are post-selection case intervals, not seed uncertainty or confirmatory winner intervals. Stride 8 first and sustainably beats stride 4 in endpoint error at H32, but only sustainably in time-mean error at H56. At H96 stride 8 wins global error, reliability at budget `0.05`, and latency, while stride 2 wins front position and away-front derivative/TV metrics. The stage is theory-ready but the optimum remains metric-dependent and right-censored. |
| L1-OOD | Line 1 audit | Register existing mild-support OOD artifacts without rerun | frozen primary stride-1/2/4/8 set plus two stride-8 checkpoint substitutions | 32 OOD cases, H32/H64/H96; historical regime labels absent from rows | aggregate error, completion, seed sensitivity, provenance completeness | MUST | DONE (PROVENANCE-LIMITED) | At H96 primary stride-2/4/8 errors are `0.07938/0.04287/0.04890` with `31/32`, `32/32`, and `32/32` completion; stride-8 repeats are `0.04074/0.03611`, both `32/32`. Stride 4 and all observed stride-8 seeds beat the single stride-2 row, but stride 8 does not beat stride 4 for every seed and only stride 8 was repeated. Data hash and old evaluator hash agree across directories. The OOD NPZ is absent and historical rows omit regime labels/generator hash, so the provenance gap is frozen and block-specific labels remain inferred and non-claim-eligible; no evaluator rerun is authorized. |
| XLINE-001 | Lines 1/3 | Frozen rollout-instability attribution | D031 FNO frontier, D041 bump PCNO failures, and D044 six-case D013 PCNO stable control; no training | FNO: at most four small-stride failures, same-case large-stride paths, and four descriptor-matched stable controls; bump: three failures plus three descriptor-matched completions; regimes remain separate | error/ripple/admissibility lifetimes; vector same-state decomposition; direct/composed and crossed-state paths; per-time learned/reference gain; first precursors; at most two intervention types per regime | MUST | CLOSED / NOT RUN | The registered attribution was never executed and closed as mixed/unresolved under the final queue freeze. Existing D033/D035--D038 evidence is retained without regeneration. The bump/DG branch has no validated restart, all cohorts are shock-bearing, and solver-versus-model decomposition and discontinuity necessity remain unresolved. No restart preflight, regeneration, or method action is currently authorized. |
| D040 | Line 2 | CPGNet legal-boundary validity closeout | public release checkpoint under oracle and causal nodal boundaries; one legal-boundary-trained checkpoint | same 20 release-bundle test trajectories, 79 calls | evaluator/checkpoint/dataset identity, normal/all/boundary RMSE, distance strata, positivity, q0.90 shock metrics | MUST | DONE | Exact archived-evaluator oracle parity passes. Legal-boundary training lowers frozen-legal normal-node RMSE by 34--45%, improves all four variables on 19/20 trajectories, and remains finite/positive, but stays 1.6--2.4x worse than oracle. This is one-seed release-bundle evidence, not paper identity, exact DG replay, conservation evidence, or a general learned-solver result. Stop Line 2 without the matched decoder ablation. |
| D041 | Line 3 | Official serious-PCNO holdout, branch counterfactual, and legal-boundary routing | frozen clean 19,155,720-parameter conservative-residual PCNO; pointwise gain `0.75`; full causal nodal policy; fixed inflow | all 20 held-out bump trajectories, 79 raw calls | one-step, mixed-prefix/completed/common endpoint, survival, positivity, graph high-pass, shock geometry, failure localization, batch-1 latency | MUST | DONE | Baseline completes 17/20 H79 runs after 20/20 at H20. Pointwise attenuation is strongly harmful. Full nodal remapping improves completion but fails accuracy/anti-smearing. Exact fixed inflow passes the tight baseline-contract gate and delays both inflow failures by 16 calls, but completion remains 17/20. This historically routed only a matched boundary contract; Line 2 later closed without it. It never authorized learned stabilization or a physical conservation claim. |
| D042 | Line 3 | Fixed compact local-basis capacity counterfactual | 289-column geometry-only FPS/Wendland-C2 partition-of-unity projection versus the saved PCNO Fourier controls | five full-resolution held-out bump trajectories at call 1 | locality, rank, residual RMSE, shock-separated graph high-pass, raw decoded admissibility and anti-smearing | MUST | DONE; REJECTED | The local span is genuinely compact, full-rank, and halves smooth-region high-pass projection error, but residual RMSE remains `0.989/0.982x` the current Fourier control and all 10 decoded proxy/case rows fail anti-smearing/admissibility; six have nonpositive raw pressure/internal energy. Reject this fixed span without a radius, center-count, or kernel sweep. This is a projection-capacity result, not training or rollout evidence. |
| D043 | Line 3 | Official-checkpoint paired branch-cancellation audit | frozen SHA-256 `2bb5ee3c...` conservative-residual PCNO; exact paired error-direction response | trajectories 16 and 60, call 10, all four hidden layers | smooth-region graph-high-pass energy identity under equal-node and reconstructed proxy weights; response gain and roughness | MUST | DONE | All 16 layer/case/proxy cancellation fractions are at most `0.00430` and the medians are negative, so the repeated paired-response screen falsifies strong inter-branch cancellation. The pointwise response is still roughest in all four layers and dominates RMS gain in three, but D041 already shows uniform attenuation is destructive. This was a historical selector toward localized control; the later route closed without promotion. It is not causal attribution or a Jacobian result. |
| D044 | Line 3 | Frozen dynamic-FV residual PCNO, physical rollout, D013, and constrained-oracle route | one seed-20260718 19,155,720-parameter conservative-residual PCNO on the frozen 135-case shock--vortex family | grouped 84/24/27 split; all 24 validation cases for H60 physical evaluation; predeclared six-case D013/oracle cohort; 27 test cases sealed | raw completion/error, front/shock/vortex/smooth hierarchy, physical totals and boundary exchange, branch/basis/recurrence mechanism, 20%-support/10%-update oracle at calls 10/30/60, Line-4 handoff flags | MUST | DONE; LOCAL CORRECTOR REJECTED | Epoch 44 gives 24/24 raw admissible H60 completion and physical-volume state error `0.00834190`, beating all three train-only controls on 24/24 cases. D013 is `unresolved`: every predeclared screen is false, the Gram contract is well-conditioned, and the paired branch selector is composite. The 18-row constrained oracle passes all structural gates but reaches only `0.05135/0.04164` median state/high-pass reduction, so it rejects a learned local correction. The strengthened Line-4 handoff recorded `line4_training_truth_authorized=true` only as a historical artifact field for the frozen family, supplied no front candidate, and kept transition training false. Current Line 4 is stopped and test remains sealed. |
| D045 | Line 3 | State-loss-only shared-face-impulse contract and tiny-fit gate | one 19,210,028-parameter PCNO face decoder on the frozen full-resolution shock--vortex mesh | four immutable training pairs; corrected smoke, 800-update attempt, and one 3,200-update exposure retry; validation smoke only; 27 test cases sealed | provenance and mesh/graph equality, exact decode balance, tiny-fit state error/loss ratio, admissibility, precision attribution, reference-impulse non-use | MUST | DONE; TINY-FIT FAILED; SERIOUS STOPPED | The antisymmetric interior decoder and current-state boundary heads pass focused tests and close against predicted boundary exchange. Best state error/loss ratio improves from `0.00166965/0.0730244` at 800 updates to `0.000955675/0.0254377` at 3,200. The absolute error gate passes but the required 100-fold loss reduction fails; precision replay is nearly unchanged. No serious face row, divergence-active sweep, physical-conservation claim, or test evaluation is authorized. The Line-4 handoff flags remain unchanged. |
| D046 | Line 3 | Canonical divergence-active/minimum-norm face-target preflight | zero-training weighted projection on the frozen full-resolution finite-volume mesh | four fixed train plus four fixed position-OOD validation cases at calls 1/30/60; 24 rows; strength-OOD test forbidden | artifact and split provenance, reference/shard decode closure, independent state-plus-boundary reconstruction, compatibility, cycle removal, weighted norm, wall exchange, sparse-solve status | MUST | DONE; PREFLIGHT FAILED; TRAINING STOPPED | The corrected 24-row run failed only canonical/reference-state closure: maximum `4.94497e-7` versus the frozen `1e-8` gate. Reference closure is `1.23427e-12`; shard closure `9.36905e-6`; independent field disagreement `1.61785e-8`; compatibility `2.94020e-12`; cycle divergence `5.18524e-10`; norm ratio `0.999440`; wall leakage `3.10460e-20`; all solve codes are accepted. Reference cycle energy is only `0.1129--0.6691%` (median `0.2646%`). No supervised tiny fit, tolerance retry, test access, or Line-4 flag change is authorized. |
| D047 | Line 3 | Fixed-mesh direct canonical-projector preflight | one anchored float64 sparse factorization per connected component, reused across all right-hand sides; zero training | exact D046 24-row train/position-OOD-validation cohort; calls 1/30/60; strength-OOD test forbidden | unchanged D046 gates plus explicit compatibility projection, reduced-system residual, factor reuse, finiteness, and no-jitter provenance | MUST | DONE; ALL GATES PASSED | All 24 rows pass. Canonical closure is `1.09848e-9--3.40147e-9`, improving every D046 row by `144.97--157.64x`; independent disagreement is at most `8.42560e-10`, compatibility projection `1.85905e-14`, and reduced residual `1.13351e-12`. One 1,130,356-nnz factor is reused. This confirmed the iterative-projector diagnosis and historically routed the now-completed, failed D048 four-pair tiny fit. No serious run, test access, conservation claim, or Line-4 flag change followed. |
| D048 | Line 3 | Direct canonical-face supervised tiny-fit gate | D045's 19,210,028-parameter full-resolution shared-face PCNO with direct `W_f^{-1}` interior/boundary supervision | exact D045 four train pairs; 3,200 updates; validation smoke is not selection; test forbidden | label closure/provenance, native full/interior/boundary error and loss ratio, decoded-state error, admissibility, wall structure, finite gradients | MUST | DONE; TINY-FIT FAILED; BRANCH STOPPED | Label and code provenance pass and 15 epochs meet every native face-space gate, but no epoch is admissible or reaches the decoded-state gate. Best decoded error is `0.474958`; final is `0.500998`, versus `0.00628209` persistence. Final replay finds `1001--1581x` relative divergence amplification and 0/4 admissible states. Stop this exact canonical face-value objective without retry, mixture, serious run, test access, or conservation claim. |
| D049 | Line 3 | Graph-band finite-volume divergence-conditioning audit | zero-training validated 250x100 physical perturbations plus analytic 250x100/500x200/1000x400 geometry controls | one accepted dynamic reference; three seeds; calls 1/6/12; no test/model access | band frequency/gain, canonical/cycle closure, admissibility and shock distortion, fixed-`Delta t` refinement scaling | MUST | DONE; ALL GATES PASSED | Median low/mid/high gains are `0.34903/0.97641/2.25444`, minimum high/low is `6.3866`, cycle gain is `1.24e-10`, and all six gates pass. Fixed-step flux gain doubles on analytic refinements, which carry no fine-grid shock truth. This supports discrete-divergence conditioning but does not identify Gibbs or a learned branch. |
| D050 | Line 3 | Frozen D044 residual-to-face lift and boundary-headroom preflight | legal predicted-total minimum-norm x-boundary lift plus separate future-reference boundary oracle | six D013 validation trajectories at calls 1/10/30/60; no checkpoint execution; test sealed | legal reconstruction/closure/wall structure, oracle budget/state headroom, admissibility and anti-smearing | MUST | DONE; ARTIFACT FAILED; ROUTE STOPPED | The legal lift reconstructs D044 to `5.37e-12` with zero non-x exchange, but the frozen oracle-closure field was misbound to intentional target projection (`0.07610`) and formally fails. No rerun is allowed. Descriptive oracle H60 state reduction is only `7.41%` versus `15%` despite essentially exact budget repair, so the route would not promote even with corrected metric storage. |
| D051 | Line 3 | Paired D044/coarse-CFD error--cost closure | frozen D044 validation rollout plus same-host WENO5-HLLC-SSPRK3 float32 coarse grids | D013 six-case pilot on 25x10/50x20/125x50/250x100; deterministic maximum-two-grid extension to all 24 validation cases; test sealed | conservative remapping floor, H60 state/shock/smooth error, own boundary balance, synchronized batch-1 cost, sequential and amortized throughput, paired case bootstrap | MUST | DONE; MATCHED ERROR DESCRIPTIVE; STRICT COST/CONSERVATION CLAIMS FAIL | The pilot selects 25x10 and 250x100. Full 250x100 CFD has H60 error `0.00808583` versus PCNO `0.00834190`, but the case-bootstrap difference includes zero and shock strength/thickness are worse. It costs `13.53` s versus calibrated PCNO `0.449` s, while PCNO p95/median is `11.70`, so strict timing fails. Float32 own-balance residual also misses `5e-5` (`2.99e-4` maximum). No production-CFD, physical-conservation, or superiority claim follows. |
| D060 | Line 3 | Matched serious stride-2 physical gate | D044 architecture and training contract with only `step_stride=2`; frozen epoch-34 checkpoint | all 24 position-OOD validation trajectories at physical H60; six D013 cases; strength OOD sealed | raw completion/state, shock/vortex/smooth hierarchy, totals, direct-versus-composed, same-host latency, parameter strata | MUST | DONE; PARTIAL MACRO-STEP GAIN; PROMOTION FAILED | D060 completes 24/24 and improves H60 state error on every case (`0.00729369`, `0.87434x` D044); direct/composed, strength, thickness, vortex, total, and descriptive timing gates pass. It fails the targeted joint capability: six-case D013 high-pass RMS is `1.0255x` rather than `<=0.8x`, all-24 endpoint high-pass energy worsens, and front-centroid distance is `1.4351x`. Fewer calls reduce accumulated state L2 but do not reduce the high-frequency source. Stop without retry, another stride/seed, test access, or add-on. |
| D061 | Line 3 | Frozen multirate rollout-blend headroom | completed D044 stride-1 and D060 stride-2 serious raw validation trajectories; fixed equal blend plus 21-point truth-informed scalar oracle | six D013 cases at matched frames 2/10/30/60; 24 rows; no checkpoint execution; test sealed | aligned source contract, state/high-pass complementarity, raw admissibility, shock/vortex/total anti-smearing, disagreement localization, summed parent cost | MUST | DONE; HEADROOM FAILED; ROUTE STOPPED | All source checks close exactly and every blend is raw-admissible. At H60 the oracle reaches only `8.14%/10.10%` median state/high-pass reduction versus `10%/20%`, is jointly nonworse in 0/6, and passes anti-smearing in 0/6. D060 owns lower state error in 6/6 while D044 owns lower high-pass RMS in 6/6; averaging interpolates rather than dominates and damages shock/vortex metrics. Disagreement localization passes (`0.650` Spearman, `57.81%` top-20 capture), but does not establish alpha selection or realizability. Do not train the shared-backbone row. |
| D062 | Line 3 | Front-fitted conservative-remap capacity oracle | frozen D060 stride-2 raw validation trajectories; target-informed row-wise front phase plus compact two-sided strength fit | six D013 cases at calls 15/30; 12 rows; no checkpoint execution; test sealed | exact row-total preservation, raw admissibility, state/front-curve/high-pass headroom, graph-front/shock/vortex anti-smearing, correction size | MUST | DONE; CAPACITY ORACLE FAILED; EXACT CHART STOPPED | All 12 rows are raw-admissible and conserve row/component totals to `2.69e-15`. At H60 the primary oracle improves front-curve MAE by `62.82%`, but median state error worsens by `41.88%`, high-pass RMS worsens by `828.08%`, joint nonworse is 0/6, and every structure count misses 5/6. The phase-only row already contains the failure. Do not train this front chart or change its extractor after results. |
| L4A-001 | Line 4A | Zero-training representation and closure preflight | identity/POD/oracle-front POD, current-state front-speed augmentation, and full-rank oracle-remap attribution | frozen 512-case nx256 Line-1 dataset; 384/64/64 split; stride-compatible frames 0:8:96 | hierarchical reconstruction, conditional-future ambiguity, fixed-size history, chart conditioning, favorable-cohort grouped bootstrap | MUST | DONE; STOP | Rank-43 POD blurs fronts; the oracle chart does not improve closure over POD; history remains materially useful; current-state speeds fail; and the existing remap fails state, recall, precision, and thickness gates even on 243 favorable snapshots from all 64 validation trajectories. Stop before learned representations. D044 has now emitted the bounded training-truth handoff with no inheritable front candidate; it does not by itself authorize transition training. |
| L4A-002 | Line 4A | Frozen-2D representation and closure gate | identity, matched rank-5000 physical-volume POD, one generic 25x10 spatial-token autoencoder, and one matched conservative-moment spatial-token candidate; no front variables or remap | D044's frozen 84/24 train/position-OOD-validation trajectories and 61 saved states on 250x100; engineering smoke used four spread train plus two endpoint-position validation cases at frames 0/15/30/45/60; 27 strength-OOD trajectories sealed | handoff completeness, reconstruction hierarchy, raw admissibility, physical totals, conditional-future ambiguity, one-step history control, decoder perturbation gain and doubled-query-resolution scaling, cost and intervention ledger | MUST | DONE; ENGINEERING SMOKE FAILED; CANDIDATE STOPPED | Strengthened preflight hashed all 78 staged arrays, loaded endpoints, and read no test arrays. The first actual 800-update matched smoke completed all 10 paired validation rows (summary SHA-256 `e742ccf4...`). The conservative-moment row beat generic L2 on 10/10 rows (`0.005430` versus `0.007605`) and enforced near-exact token/global budgets, but retained only `0.539` mean shock strength and broadened thickness to `2.672x`; generic retained `0.463` and broadened to `3.540x`. Both miss the `0.0021` reconstruction and 5% strength/thickness gates by large margins. The structured row's lower smooth high-pass energy is not ripple reduction because fronts remain blurred. Its conditional-future ambiguity is only `5.74%` below generic, one-code history ratios are approximately one, and doubled-resolution gain is stable at `1.002`; no POD comparison can rescue the failed physical gate. The earlier real-shard shape integration attempt stopped before model construction and zero optimizer steps. Do not continue the same loss/decoder, run rank-5000 POD or a serious representation job, train a transition, access test, or filter. The second smoke was never spent, and the campaign closed that allowance; it conveys no current authorization. |
| L4A-003 | Line 4A | Frozen-decoder code-reachability oracle | exact L4A-002 conservative checkpoint and fixed decoder/four moment channels; per-state L-BFGS fit of only 16 free channels | same two position-OOD validation trajectories and frames 0/15/30/45/60; 10 states; strength-OOD test sealed | inherited reconstruction/front/admissibility gates, overshoot, training-scale code displacement, cost and intervention ledger | MUST | DONE; DECODER MANIFOLD REJECTED; FAMILY STOPPED | Per-state fitting lowers mean L2 from `0.005430` to `0.003226` and thickness from `2.672x` to `2.109x`, but nine of 10 rows still miss `0.0021`, mean strength worsens from `0.539` to `0.490`, maximum overshoot increases, and every fitted code lies outside the one-scale training neighborhood (RMS `1.528--4.419`). All raw states are admissible, but lower smooth high-pass energy is not ripple reduction because fronts remain broad and weak. Summary SHA-256 begins `e9577b43`; cost is `0.00308` GPU-hour. Encoder remediation is ineligible. Stop serious training, POD rescue, transition, test access, and filtering for this family. This per-state oracle is capacity evidence only, not forecast evidence or a universal rejection of latent representations. |
| L4A-004 | Line 4A | Conservative local-Haar capacity preflight | same 25x10x20 state and four exact token means; 16 deterministic training-selected discontinuous mode/direction atoms per token; no front variables | same four training and two position-OOD validation trajectories at frames 0/15/30/45/60; 10 validation states; strength-OOD test sealed | exact L4A-003 target/control binding, latent rank and digest, reconstruction/front/admissibility hierarchy, token/global moments, overshoot, cost and intervention ledger | MUST | DONE; CAPACITY REJECTED; CHART STOPPED | Preflight rehashed all 78 arrays, matched L4A-003 targets bitwise, used only 20 declared training states, and read no test array. Haar improves amortized-encoder L2 on 10/10 rows (`0.005430` to `0.004461`), strength from `0.539` to `0.830`, thickness from `2.672x` to `1.554x`, and IoU from `0.390` to `0.500`; admissibility, overshoot, rank, token moments, and global budgets pass. But only one row reaches `0.0021`, mean strength remains 17.0% weak, thickness 55.4% broad, and Haar loses L2 to the privileged fitted-code control on 10/10 rows. Summary SHA-256 begins `f91b78d5`; cost is `0.00276` CPU-hour. This supports only that discontinuous decoder regularity helps front fidelity; it rejects sufficiency of this fixed 16-detail chart. Stop without a level, atom, lattice, or dictionary sweep; no closure, serious training, transition, test access, filtering, geometry-transfer, or neural-operator claim is authorized. |

## 2026-07-20 D043 Paired Branch-Cancellation Audit

D043 repeats the frozen two-case branch trace on the official selected clean
checkpoint, SHA-256
`2bb5ee3ca831a6ffc498f01e309ae7ea957b2df0f411023fb3812919a6732964`,
epoch 4, configuration digest
`5cd903ca0b0af43651d1b72280e55a89d63c93fff9b8b17e90fc85e5b0494418`.
Two earlier local bundles are superseded: one used the pre-v2 classifier and
one used the epoch-39 parent checkpoint rather than the selected clean
checkpoint. Neither enters the mechanism claim.

The primary statistic is the graph-high-pass energy identity for the finite
branch response to an accepted fraction of the observed rollout-error
direction. It is not an infinitesimal Jacobian or a causal ablation. Every one
of the eight layer/case rows is valid under both proxy measures. Cancellation
fractions range from `-0.10534` to `0.00430` under equal-node weights and from
`-0.10159` to `0.00429` under reconstructed weights; their medians are
`-0.02153` and `-0.01903`. All 16 values are below the frozen `0.05`
falsification threshold, with relative energy-identity residual at most
`1.98e-7`. The absolute teacher/rollout outputs independently show no strong
cancellation, but remain noncausal context.

The pointwise response is nevertheless the roughest branch in all four layers
for both cases and has the largest RMS response in three layers; the
differential branch leads at layer zero. Combine this with D041: the pointwise
branch is a plausible ripple carrier, but globally scaling it by `0.75`
destroys the learned map. Reject both the strong-cancellation explanation and
a pointwise gain sweep. The top-level mechanism remains composite/unresolved,
not pure Gibbs, pure pointwise, or spectral-only.

## 2026-07-20 D042 Fixed Compact Local-Basis Counterfactual

D042 tests one no-training capacity counterfactual on five full-resolution
call-1 bump residuals. The 289-column geometry-only FPS/Wendland-C2
partition-of-unity basis covers every node, has maximum partition error
`4.44e-16`, effective rank 289/289 under both proxy measures, and median node
support `2.73--2.80%`. It reduces shock-separated smooth-region graph-high-pass
projection error to `0.474/0.477x` the saved Fourier control.

That spectral-looking improvement is not a solver improvement. Median
conservative-residual RMSE is still `0.989/0.982x` the current Fourier control
and `0.9987x` the coordinate-span Fourier control. All ten decoded
trajectory/proxy rows fail the raw anti-smearing/admissibility gate: six have
nonpositive pressure/internal energy, and pressure-front centroid error is
`8.46--15.88` median-edge lengths. Reject this exact fixed local span and do
not sweep its radius, center count, or kernel. The result falsifies only the
claim that compact support alone is sufficient; it does not falsify adaptive,
discontinuous, multiresolution, or learned local bases, nor does it establish
a universal Gibbs mechanism, rollout gain, conservation, or flux accuracy.

## 2026-07-20 D041 Official PCNO Holdout And Boundary Routing

D041 freezes the selected conservative-residual checkpoint (SHA-256
`2bb5ee3ca831a6ffc498f01e309ae7ea957b2df0f411023fb3812919a6732964`)
and evaluates all 20 held-out bump trajectories for 79 raw autonomous calls.
The model has 19,155,720 parameters, `kmax=8`, five width-128 layers, and a
`6 x 2` Fourier domain. No row uses future-reference boundary values, clipping,
primitive floors, smoothing, or a limiter.

| Frozen row | one-step proxy relative L2 | H79 completion | mean survival | completed-case final proxy error | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| PCNO `model_all_nodes` | 0.006663 | 17/20 | 0.905696 | 0.043668 | official strict baseline |
| Pointwise tail gain `0.75` | 0.024931 | 13/20 | 0.771519 | 0.278870 | reject attenuation and gain sweep |
| Full causal nodal policy | 0.009509 | 19/20 | 0.982911 | 0.069585 | stability sensitivity; fails accuracy/anti-smearing |
| Fixed freestream inflow only | 0.006663 | 17/20 | 0.925949 | 0.043521 | passes matched boundary-contract repair gate |

The official baseline completes all 20 H20 rollouts with mean error `0.022257`.
Its three terminations are negative-pressure proposals: trajectories `05` and
`19` first fail at inflow nodes, while `12` first fails at a wall node. Median
front-region scaled error-energy share is `99.6%` at call 1, `82.2%` at call 20,
and `68.0%` at call 40. Relative to call 1, median smooth-region graph high-pass
energy grows `14.1x`, `460x`, `3064x`, `6923x`, and `18206x` at calls
5/10/20/40/79. A failure call need not contain a global error spike: termination
is often a local admissibility overshoot after recurrent spread.

The exact gain-one replay differs from the uninstrumented path by at most
`7.63e-6` absolute and `1.38e-8` relative L2. The `0.75` pointwise-tail row has
`3.74x` worse one-step error and `7.21x` worse common-case H79 state error, with
worse high-pass, thickness, and strength metrics. Therefore the earlier
five-case paired-response selector was directional sensitivity, not causal
attribution. D043 subsequently falsifies strong inter-branch cancellation in
the paired error response: the pointwise branch is rough and functionally
coupled, but its harmful global attenuation is not evidence that beneficial
cancellation must be preserved. Reject the pointwise-only hypothesis and any
branch-gain sweep.

The full Line-2 causal nodal policy demonstrates that boundary recurrence can
change failure timing: completion rises to 19/20. It nevertheless fails its
predeclared H20 gate with median candidate/baseline ratios `1.159` for state,
`1.227` for shock thickness, and `1.112` for shock strength. Its approximate
wall/outflow remapping mismatches the dataset boundary states and is rejected as
the next matched PCNO contract.

Fixing only the prescribed freestream inflow leaves H20 state/high-pass/front/
thickness/strength ratios at `0.99984/0.99893/1.00000/1.00000/0.99985`, extends
both inflow-triggered failures by 16 calls, and improves the mean common-
completer H79 state ratio to `0.99663`. The failures migrate to high-percentile
graph-high-pass wall or interior nodes, while completion remains 17/20. Fixed
inflow is therefore a legal, low-distortion baseline contract repair for the
next serious PCNO training/evaluation pair; the present frozen result is still
a checkpoint-mismatch sensitivity and does not justify an expensive retrain by
itself.

Mechanism decision: current evidence supports composite recurrent amplification
seeded by shock/front approximation and modulated by boundary recurrence, with
local conservative-state admissibility overshoot causing termination. Pure
Gibbs, pointwise-only, and boundary-only explanations are falsified; spectral
basis conditioning remains a contributing mechanism rather than an isolated
cause. Do not reopen CPGNet, pointwise, smoothing, noise, exposure, or full
wall/outflow-policy sweeps. The next method-facing gate is D037's frozen oracle
on the now-validated dynamic finite-volume testbed. None of these bump rows supports a
physical conservation, shared-face flux, or coarse-CFD claim.

## 2026-07-20 D040 CPGNet Legal-Boundary Closeout

D040 closes the release-bundle validity question before making mechanism
claims. All three rows use the same 20 selected test trajectories, 79 calls,
dataset SHA-256 `d97abde1...`, and archived closeout-evaluator SHA-256
`97a9c960...`. The public
checkpoint SHA-256 begins `46f8d59e`; the legal-trained checkpoint begins
`1f0f8860`. The exact archived closeout-evaluator oracle replay differs from
the earlier oracle aggregate by at most `8.5e-7`. The release-evaluation runtime files match the
pinned public reference hashes, but paper dataset/checkpoint identity and
checkout git metadata remain unverified. The historical training manifest
omits the public `utils/lossCompute.py` and `utils/noise.py` pins.

The primary evaluator aggregate is the arithmetic mean of per-trajectory RMSE,
not a node-count-weighted pool. Vectors are `[rho, v1, v2, pres]`:

| Checkpoint and boundary contract | normal-node RMSE | all-node RMSE | raw boundary RMSE |
| --- | --- | --- | --- |
| Public, next-reference oracle | `[0.177032, 0.073653, 0.082204, 0.319612]` | `[0.173991, 0.072386, 0.080788, 0.314103]` | `[0.048065, 0.024960, 0.042416, 0.080638]` |
| Public, causal nodal sensitivity | `[0.563023, 0.263983, 0.242344, 1.137027]` | `[0.564728, 0.266693, 0.238618, 1.143770]` | `[0.610301, 0.331888, 0.086356, 1.331108]` |
| Legal-boundary trained, causal nodal rollout | `[0.368161, 0.157947, 0.133379, 0.753202]` | `[0.372082, 0.159912, 0.131358, 0.761662]` | `[0.466753, 0.210027, 0.066296, 0.970038]` |

Legal-boundary training reduces the frozen-sensitivity normal-node RMSE by
`34.6%/40.2%/45.0%/33.8%` and improves all four variables on 19/20
trajectories. The single exception is small but real, so do not claim
trajectory-wise dominance. All 60 trajectory rollouts are finite and have zero
nonpositive normal-node density or pressure. The legal-trained checkpoint
remains `2.08x/2.14x/1.62x/2.36x` worse than oracle by variable.

Run-level q0.90 shock values aggregate each trajectory's time mean:

| Contract | IoU | F1 | Chamfer | centroid | front pressure RMSE | smooth pressure RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Oracle | 0.613632 | 0.750400 | 0.012354 | 0.011142 | 0.869246 | 0.093225 |
| Frozen legal sensitivity | 0.435916 | 0.582985 | 0.084411 | 0.257594 | 2.348944 | 0.774713 |
| Legal trained | 0.489592 | 0.630570 | 0.025252 | 0.020826 | 1.813738 | 0.353543 |

Legal training beats the frozen sensitivity on centroid and Chamfer for 20/20
trajectories and on front/smooth pressure RMSE for 19/20, but on IoU/F1 for
only 11/20. Its worst post-policy boundary absolute error is `19.65`, versus
`15.64` for the frozen sensitivity, so the boundary tail does not uniformly
improve.

The legal training artifact uses all 300 release-bundle train trajectories,
seed `20260718`, 15 teacher-forced epochs plus five multistep epochs, the
released 12-layer architecture and recorded normal-node state-loss call, and no
inference floor or post-update limiter. The final completed epoch was selected without a
validation split. Training took 24.26 hours. Train Mach `2.604--3.397` contains
test Mach `2.660--3.294`; no grouped parameter or geometry holdout was run.
The causal nodal policy uses freestream inflow plus interior-extrapolated slip
wall and supersonic outflow. It does not replay the exact DG boundary stencil.

This run retained the released architecture, recorded normal-node state-loss
contract and noise scales, and nominal two-stage schedule under causal
boundaries; the exact imported loss and noise implementations were unpinned, so
it was not an exact replay of released training. Recorded deviations include deterministic
single-process ordering, separate teacher-forced step backpropagation, and
sequential multistep microbatches. The two recorded local source hashes match
archival commit `4654225e`, but the manifest omitted the local provenance and
Euler utilities, so this is partial archival linkage rather than a complete
training-source closure.

The saved training history predates the hardened gradient/state audit. It
contains no per-update gradient norms or optimizer-state finiteness record.
Both normalizers reached their `300000` accumulation-call cap in teacher-forced
epoch 13, before multistep recurrence began. Teacher-forced raw pressure reached
`-3.55618`; multistep epoch minima were at least `0.25118`, and the evaluated
rollouts were positive. Positivity is therefore observed for this checkpoint,
not an architectural invariant. Likewise, the historical v1 mesh audit used
endpoint temporal evidence; the current v2 code adds midpoint sampling,
all-frame static-graph checks, and all-frame causal-outflow checks, but those
hardening results are not retroactive evidence for the completed run. The same
applies to the current exact stencil/audit binding, complete local-source
closure, stage-2 recurrent-outflow gate, and physical admissible-prefix
accounting. Future boundary truth was not used as a rollout input or recurrent
state in legal training, but full truth targets did contribute to supervised
targets and released output-normalizer statistics.

Locked ignored artifacts:

- `cpg_phase1d_oracle_current_primary20_20260720a/`;
- `cpg_phase1d_frozen_legal_matched_primary20_20260720a/`;
- `cpg_phase1d_legal_train_full_seed20260718_20260718a/`;
- `cpg_phase1d_legal_trained_eval_primary20_seed20260718_20260718a/`; and
- `cpg_phase1d_legal_boundary_closeout_primary20_20260720a/`.

The closeout diagnostic records both evaluator-compatible mean-per-trajectory
and node-time-weighted RMSE, per-trajectory/time CSVs, shock distributions,
positivity, and artifact inputs. Its source SHA-256 is `a88b370f...`.

Result-to-claim decision: future-reference boundaries materially assist the
released row, and the recorded causal-boundary training configuration recovers
a useful but still substantially weaker release-bundle rollout. This does not establish a
paper-table reproduction, compared-baseline parity, exact conservation,
integrated characteristic coverage, conditional-manifold behavior,
memorization, or a general learned solver. The frozen legal row is a
sensitivity counterfactual, not a fairly trained autonomous baseline. D014
found no endpoint-sampled finite-radius violation, while DG-substep coverage
remains unresolved. The graph-to-control-volume/face mapping required for
physical balance is absent. Because no decoder mechanism was isolated, omit
the three-way matched decoder ablation and stop further Line-2 training.

## 2026-07-20 D039 Frozen Operating-Envelope Closeout

D039 is a deterministic synthesis of the registered D032--D038 tables. It adds
no learned training, checkpoint evaluation, solver call, dataset, or checkpoint
selection. The audit verifies the four selected checkpoint hashes, the exact
64-case split, 101 common saved frames, active H96 waves, the H32 harder-map/
fewer-calls ordering, seed-consistent stride-8 teacher-modal deficit, and exact
agreement of all 72 duplicated D036/D038 common-endpoint summary rows.

The metric-specific map makes the crossover and conflict explicit. Endpoint
global error selects strides `2/4/8/8/8` at H8/H16/H32/H64/H96; time-mean error
selects `8/4/4/8/8`. The aggregate H8 stride-2 endpoint edge over stride 4 and
H32 stride-8 edge over stride 4 have paired case-bootstrap intervals crossing
equality, whereas the H16, H64, and H96 endpoint winners do not. Stride 8 versus
stride 4 crosses sustainably at H32 for endpoint error and H56 for time-mean
error. At H96, stride 8 minimizes global error (`0.01767`) and host-to-host
latency (`15.78` ms), while stride 2 minimizes front-position error and the
away-front second-derivative and TV-deviation metrics. Strides 2/4/8 all remain
raw-admissible; reliability at error budget `0.05` uniquely selects stride 8.

The ignored evidence root is
`artifacts/time_dependent_no/line1_frozen_closeout_20260720/`. It contains the
machine-readable operating envelope, paired uncertainty and crossover tables,
the provenance report, and PDF/PNG operating-envelope and Pareto figures. This
closes empirical Line 1. The next Line-1 action is theory and reporting, not a
new stride, target, constraint, or architecture experiment.

## 2026-07-19 D038 Frozen Modal-Error Evolution

D038 adds no training, checkpoint selection, recurrence repair, or new data. It
replays the six D031 checkpoints through the same raw conservative recurrence.
The modal field is the fixed-scale conservative state for rollout and the
fixed-scale conservative increment for teacher forcing. Each field has its
spatial mean removed, receives a Hann window, and is transformed with an
orthonormal real FFT. Powers are averaged over the 64 held-out cases and three
conservative channels. A truth-normalized modal ratio is displayed only when
that mode's mean truth power exceeds `1e-8` of total nonzero-mode truth power;
the error-energy-share heatmap does not use this denominator.

Teacher forcing uses starts `0:5:90`. Pooling modal powers over those common
truth starts gives:

| Mode band | Stride 4 relative update error | Stride-8 range | Stride-8 / stride-4 range |
| --- | ---: | ---: | ---: |
| 1--4 | 0.005162 | 0.006138--0.006698 | 1.189--1.298 |
| 5--16 | 0.007769 | 0.014219--0.015271 | 1.830--1.966 |
| 17--24 | 0.017596 | 0.043372--0.048291 | 2.465--2.744 |
| 25--64 | 0.047859 | 0.062595--0.068319 | 1.308--1.427 |
| 65--Nyquist | 0.074912 | 0.099231--0.103653 | 1.325--1.384 |

Thus the larger flow map has a seed-consistent broadband one-call
approximation deficit. The strongest relative contrast lies at modes 17--24,
next to the 24-mode spectral-layer cutoff, but modes above that cutoff are not
evidence that the spectral branch alone created the error: pointwise paths and
nonlinearities can generate them.

The primary stride-8/stride-4 rollout contrast separates total modal amplitude
from frequency shape:

| Endpoint | Total modal relative-error ratio | Error-centroid ratio | RMS-mode ratio | normalized `k^4` shape ratio | tail-share ratio | all stride-8 seeds rougher by every shape metric |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| H8 | 1.218 | 1.093 | 1.098 | 1.167 | 1.417 | yes |
| H32 | 0.917 | 1.203 | 1.181 | 1.327 | 1.857 | yes |
| H96 | 0.522 | 1.017 | 1.034 | 1.085 | 1.211 | no |

The normalized `k^4` factor is
`sqrt(sum_k k^4 P_error(k) / sum_k P_error(k))`; it measures spectral shape,
not the physical-space D036 second derivative. At H8 every one of the 101
truth-resolved modes is worse for primary stride 8. At H64 and H96 every
truth-resolved primary stride-8 mode is lower in amplitude than stride 4, so
the later global-error advantage is genuine rather than a low-mode average
hiding larger absolute modal errors. What remains is a relatively rougher
shape for the primary seed. That global shape ordering is not preserved by both
repeat seeds at H96, whereas D036's localized away-front derivative/TV
ordering is preserved by all three. The late visual ripple is therefore
localized and window/mask-sensitive; do not call it a global spectral
instability.

Mechanistically, the evidence supports two stages. A larger fixed timestep
requires a more phase-dispersive, broadband one-call map around moving shocks,
so approximation phase/amplitude mismatch injects alternating defects. Fewer
calls then reduce the number of recurrent defect injections and eventually
win in total error. This is consistent with, but does not by itself prove, the
linear-advection multiplier argument
`exp(-i k a Delta t) - 1`: its phase varies faster with mode as `Delta t`
grows. It does not support a universal neural CFL or a claim that the FNO
spectral branch alone causes ringing.

The final ignored evidence root is
`artifacts/time_dependent_no/line1_frozen_modal_spectra_20260719/`.
`mode_spectra.csv` has 41,088 unique rows and SHA256
`134f532bfa5baaf40257c55384d3210db4b35f1d85bbcd2085071935f9d448af`.
The canonical rollout batch size is one. The new scalar table adds denser
teacher starts to D036; every overlapping scalar row is numerically identical.
The `analysis/` directory contains common-scale PDF/PNG heatmaps, modal band
and shape tables, the stride-8/stride-4 mode table, and the report.

## 2026-07-20 D037 Dynamic Finite-Volume Reference Closeout

D037 has reached benchmark and data-contract closure, not method closure. The
primary implementation is
`utility/time_dependent_no/shock_vortex_fv.py`, with generation and audit
entry points under `scripts/time_dependent_no/`. It uses raw conservative
SSPRK3 recurrence, primitive WENO5-JS reconstruction, HLLC fluxes, linear
x-extrapolation, y symmetry, and no accepted-state clipping or positive floor.
Its stored contract includes cell volumes, face measures, unit normals,
owner/neighbor orientation, accepted-step times and rejection counts, and
cumulative accepted-substep face impulses restricted to the common 250x100
mesh. Initial states are certified conservative cell averages, with
shock-crossing cells split exactly and full fine-grid tensor Gauss--Legendre
quadrature checked at doubled order.

The 250x100, 500x200, and 1000x400 runs reach `t=0.6` in 822, 1642, and
3279 accepted steps. Every run has zero rejected attempts and zero invalid-face
fallbacks; maximum interval balance-closure error is `9.38609e-13`. The
three predeclared self-convergence checks pass: final full-state, time-mean
full-state, and final centerline-density successive ratios are `0.484166`,
`0.481540`, and `0.397123`. The medium/fine final shock locations coincide,
and their benchmark-window vortex-core density relative difference is
`1.15817e-4`.

The authoritative ignored result is
`shock_vortex_fv_convergence_cellavg_sharpclaw_matchedbc_20260720c`. It records
schema `shock_vortex_fv_convergence_audit_v3`, status
`benchmark_contract_closed`, all checks true, and both benchmark and direct
reference-impulse closure booleans true. The pinned Clawpack 5.9.0 build
`py311h3d4ca6a_1` SharpClaw run uses the matched linear primitive-variable x
boundary and reflecting y wall. Its initial state differs by `3.18165e-16`;
final and time-mean state errors `0.0315534/0.0156538` are below frozen envelopes
`0.0466208/0.0233192`. Final shock-position difference is zero and vortex-core
density relative difference is `0.00215749`.

After conservatively restricting medium/fine states and impulses to the common
250x100 mesh, the primary ladder's full-face-vector, divergence-active, cycle,
and boundary impulse-error ratios are `0.463428`, `0.463362`, `0.463482`, and
`0.464003`, with every componentwise ratio at most `0.464399`. Maximum
reconstruction,
cycle-divergence, weighted-orthogonality, and decoded-transition residuals are
`2.22363e-17`, `2.71952e-9`, `3.29544e-15`, and `1.31988e-17`. The v3 audit
uses a `1e-11` decomposition solve tolerance without changing the `1e-8` cycle
gate. It applies relative contraction to all x-boundary components and y-wall
normal momentum, while requiring analytically zero y-wall mass, tangential
momentum, and energy errors and actual exchanges to stay below `1e-14`.
SharpClaw exports states only, not face impulses. Direct full-impulse closure is
therefore a same-primary-solver common-mesh label result, not validation of
native fine-grid face vectors; the large 2D cycle component remains
discretization-specific and non-unique.

Preserve the v2 ignored artifact
`shock_vortex_fv_convergence_cellavg_sharpclaw_matchedbc_20260720b` as
superseded fail-closed provenance. Its `failed_convergence` status reflects
one-ULP saved-time/config mismatches, an insufficient `1e-10` LSMR solve
tolerance for the unchanged `1e-8` cycle gate, and relative ratios applied to
analytically zero wall components. Its state, independent-envelope, and
impulse-contraction evidence already passed; it is not a physical-convergence
failure. The pinned public [Pyro](https://github.com/python-hydro/pyro2) 4.5.0
CTU/HLLC run remains a historical low-order mismatch: final/time-mean state
errors `0.200259/0.092201` exceeded `0.047538/0.025736`, without identifying
which discretization was more accurate.

The truth-informed correction scaffold was historically implemented at
`utility/time_dependent_no/conservative_correction_oracle.py` and exercised by
`tests/time_dependent_no/test_conservative_correction_oracle.py`. Both paths
were retired from the active tree after closeout and are recoverable at
pre-cleanup commit `729091b`. The scaffold enforced interior cancellation,
excluded boundary corrections, searched only raw density/pressure-admissible
states, and accepted a frozen anti-smearing callback. The run gate below records
the historical ordering and is not current authorization.

At that historical gate, the next eligible action was to predeclare that
perturbation family and split, then fit one serious conservative-residual global
baseline. Benchmark closure did not establish neural quality or oracle
headroom. The contract required the frozen 20%-support/10%-update oracle before
any learned correction; even a pass would have established constrained
representational headroom only, not learnability, inference-time localization,
or causal attribution.

## 2026-07-19 D036 Frozen Ripple And Roughness Conflict

D036 adds no training and changes no selected checkpoint. The D031 primary
stride-1/2/4/8 checkpoints and both frozen stride-8 repeats are replayed under
the same raw conservative recurrence through H96. The state-only D031 NPZ has
no solver face-flux impulse, so the existing D013 diagnostic now makes flux
metrics conditional while retaining every state, spectrum, and admissibility
metric. The final claim input is
`d036_scale_spectra_smooth/per_sample_metrics.csv`; its SHA256 and all six
checkpoint hashes are recorded in the ignored reports.

The visual ripple concern is real but coexists with the global-error result.
At H96, primary stride 8 versus stride 4 gives:

| Metric | Stride 4 | Stride 8 | Ratio | Paired bootstrap 95% ratio |
| --- | ---: | ---: | ---: | ---: |
| Fixed-scale conservative relative L2 | 0.02646 | **0.01766** | 0.668 | [0.571, 0.780] |
| Smooth-region conservative RMSE | 0.03217 | **0.02377** | 0.739 | [0.647, 0.848] |
| Smooth-region first-derivative error RMS | 1.724 | 1.854 | 1.075 | [1.007, 1.148] |
| Smooth-region second-derivative error RMS | 484.1 | 579.4 | 1.197 | [1.147, 1.244] |
| Smooth-region predicted/truth TV | 1.729 | 1.891 | 1.094 | [1.069, 1.120] |
| Error energy in modes 65--Nyquist | 1.735% | 2.077% | 1.197 | [1.050, 1.377] |

Thus stride 8 has smaller state error even away from the two detected fronts,
but its remaining smooth-region error is rougher. It has higher smooth
second-derivative error on 54/64 cases and higher smooth TV ratio on 59/64.
Across all three stride-8 seeds, H96 global L2 stays in
`0.01766--0.01820`, smooth second-derivative error is `531.5--646.8`, and
smooth TV ratio is `1.834--1.974`; the latter two ranges remain above the
single stride-4 values.

Threshold lifetime is explicitly diagnostic. For smooth-region predicted /
truth TV budgets `1.25/1.5/2.0`, stride 8 reaches the threshold earlier than
stride 4 under restricted-mean and horizon-survival summaries. At budget
`1.5`, median first exceedance is H68 for stride 4 and H56/H56/H48 for the
three stride-8 seeds; H96 survivors are 14/64 versus 6/64, 7/64, and 3/64.
The thresholds are a sensitivity grid, not physical admissibility limits.
Raw stride-4 and stride-8 survival is still 64/64 at H96, so do not rewrite
this as proof that stride 8 becomes physically invalid first.

Publication PNG/PDF plots and the paired, common-endpoint, and threshold CSVs
are under ignored root
`artifacts/time_dependent_no/line1_frozen_ripple_20260719/analysis/`. The
result does not justify truth-free extrapolation beyond H96: once reference
truth and active-wave occupancy are absent, excess variation cannot be
distinguished reliably from correct dynamics. A longer reference trajectory
would require separate authorization and a new domain-occupancy contract.

## 2026-07-19 D035 Frozen Error-Geometry And Visualization Closeout

D035 adds no learned training and does not change any selected checkpoint. It
replays the frozen seed-`20260707` stride-1/2/4/8 models with decoded
conservative recurrence retained exactly between calls: no clipping, floor,
limiter, projection, or other inference repair is applied. Every quantitative
curve uses common saved-frame endpoints, and a failed raw path remains missing
after failure. Input and script hashes are recorded in the ignored report.

At H96 (`t=0.48`), the mean held-out errors are:

| Stride | Calls | Global L1 | Global L2 | Translation-oracle L2 | Shock-region L2 | Smooth-region L2 | Top-two position MAE | Complete |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 96 | 0.06566 | 0.12706 | 0.06029 | 0.32021 | 0.07699 | 0.11218 | 62/64 |
| 2 | 48 | 0.01731 | 0.03008 | 0.02437 | 0.06511 | 0.02064 | **0.06232** | 64/64 |
| 4 | 24 | 0.01300 | 0.02645 | 0.01825 | 0.07233 | 0.01375 | 0.06528 | 64/64 |
| 8 | 12 | **0.00996** | **0.01767** | **0.01520** | **0.04638** | **0.01015** | 0.09488 | 64/64 |

The translation oracle chooses one integer global shift within the fixed
predeclared window to minimize conservative L2 on a fixed interior. It is a
post-hoc phase diagnostic, not an inference correction, and cannot align
multiple waves independently. On that same interior it removes mean error
fractions `0.554/0.172/0.262/0.108` for strides 1/2/4/8. Stride 8 remains best
after alignment, so its advantage is not explained by one correctable global
phase shift. Stride 2 nevertheless has the best explicit front-position MAE;
the empirical sweet spot is metric dependent.

The dense same-state extension evaluates all 12 starts from frame 0 through 88
in increments of eight. At every start, truth-state one-call error increases
strictly from stride 1 to 8, while the same error divided by physical step
decreases strictly. Four-point log-error/log-stride slopes range from `0.488`
to `0.696`. This is an empirical scaling across four separately trained fixed-
step models, not an approximation order. At frame zero the learned-error and
truth-increment slopes are `0.488` and `0.487`, while increment-relative error
stays within `0.0163--0.0183`. Larger maps are harder in absolute one-call
error but have lower measured defect per unit physical time throughout the
observed trajectory phase.

The error-budget view gives the same operational conclusion without averaging
late errors. For budget `0.01`, median first-event times are
`0.04/0.15/0.22/0.28` for strides 1/2/4/8. For budget `0.05`, stride 1 reaches
its median event at `t=0.28`; the other strides do not reach a median event by
H96, where reliable counts are `4/64`, `52/64`, `59/64`, and `64/64`.

Casewise H96 winners among strides 2/4/8 are `14/15/35` cases. A post-hoc
per-case oracle lowers mean error from the population stride-8 value `0.01767`
to `0.01403`, a `20.61%` diagnostic gap. The preregistered leave-one-out
nearest-centroid router built from six initial physical descriptors has
accuracy `0.328`, balanced accuracy `0.368`, and accuracy difference
`-0.219` with paired-bootstrap interval `[-0.406, -0.0469]` relative to the
fold-majority control. It fails all signal gates. Do not build an adaptive
timestep controller from these descriptors.

Publication PNG/PDF figures, five selected-case GIF/NPZ rollouts, final-frame
overlays, pressure-error space-time maps, CSVs, contracts, and reports are
under ignored root
`artifacts/time_dependent_no/line1_frozen_visualization_20260719/`. The durable
entry point is
`scripts/time_dependent_no/visualize_euler1d_flow_map_frontier.py`, with focused
tests in
`tests/time_dependent_no/test_euler1d_flow_map_visualization.py`. This closes
the requested frozen visualization and error-geometry analysis. It does not
bracket a universal optimum: stride 8 is still the tested right boundary, and
the 101-frame dataset supports active-wave comparison through H96 rather than
a late-time plateau.

## 2026-07-15 Integrated Solver-Flux Result

The validated ADER artifact contains 512 trajectories, 101 saved frames, 256
cells, and 257 owner-oriented faces. Its accepted-substep impulses close the
state transition before serialization to about `7.1e-14`; float32 decoded
closure is at most about `2.90e-5`. The state arrays match the original dataset
bitwise and no fallback was used. This rules out a mislabeled-flux or endpoint
closure failure for the tested rows.

The original tiny absolute gate was too strict for the shared FNO: even
state-only reached minimum decoded train relative L2 about `0.0060`, not the
declared `0.002`. The failed gate was preserved. A repeat seed reproduced the
ordering, a 100-epoch `3e-4` control improved all native losses, and the matched
raw/gauge-canonical by joint-weight controls identified gauge-canonical joint
weight `0.1` as the only solver-flux row that kept tiny-set state fit within
`1.10x` while improving identifiable flux fit.

That candidate did not survive the 64/16/16 successive-halving gate. Against
the state-loss-only flux head, its held-out active-flux MSE improved from
`1.0006e-3` to `3.0068e-4`, but one-step relative L2 worsened from `0.004612` to
`0.006226` and mean raw survival from `0.51875` to `0.300`. Every one of the 16
paired test trajectories failed earlier under joint supervision. State-only
survived 10.375 saved steps on average; joint survived 6.0. All terminations
were raw nonpositive states, with pressure responsible in 15/16 joint cases.

Final-prefix relative, shock, and conserved-total errors are not directly
comparable because the joint prefixes are shorter. At each common valid saved
step, joint relative error was already higher; at each joint trajectory's last
valid step, its error exceeded state-only for all 16 cases. The failure is
therefore classified as a decoded-state/generalization tradeoff plus recurrent
stability failure, not label invalidity or inability to optimize the native
flux target.

In this 1D chain, projecting out the one constant-flux null mode makes face flux
uniquely recoverable from its divergence. Gauge-canonical face-flux MSE is thus
an inverse-divergence reweighting of the state increment. It emphasizes global,
low-frequency residual modes and can reduce active face error while weakening
the local/high-frequency update accuracy required at shocks. Do not promote
this exact loss to the full split. The next bounded diagnostic is a good
state-loss-only flux model followed by short unrolled training, with a smooth
training-only admissibility barrier as a matched second-stage intervention.

## 2026-07-15 Short-Unroll Stability Result

The 64/16/16 stability screen used the same 316,739-parameter FNO and identical
50-epoch one-step histories for both rows. Each row then used ten epochs of a
four-step differentiable conservative rollout, a fresh AdamW optimizer at
`3e-5`, and rollout-based validation checkpoint selection. No limiter, floor,
or positive transform was used in training recurrence or inference.

Unroll-only passed the predeclared gate. Relative to the reproduced state-only
checkpoint, test one-step relative L2 changed from `0.004612` to `0.003438`,
mean survival from `0.51875` to `0.759375` (`1.464x`), and completion from 0/16
to 2/16. It ran longer on 14/16 paired trajectories, with a median gain of five
saved steps. At every case's common endpoint its relative error was lower; the
paired survival improvement has a trajectory bootstrap 95% interval of about
`[0.159, 0.306]`. The epoch-60 unroll checkpoint was selected.

The weight-`0.1` admissibility row reached one-step relative L2 `0.003418`,
survival `0.75625`, completion 2/16, and 14 nonpositive terminations. Compared
with unroll-only, its survival ratio is `0.996`; four trajectories are longer,
eight tied, and four shorter. It therefore fails both barrier-attribution
conditions: no `1.10x` survival gain and no reduction in nonpositive failures.
The barrier contributes about 4.5% of state loss only in stage epoch 1 and less
than 0.1% thereafter. This is a negative result for this loss specification, not
for all differentiable invariant-domain penalties.

Result-to-claim classification is `partial`: short recurrent training is a
supported stability intervention on this fixed midscale 1D Euler screen, but
14/16 test trajectories still terminate early and no target-family superiority,
long-horizon robustness, or cross-stride claim is established. Promote only the
unroll-only row to 384/64/64.

## 2026-07-15 Full-Scale Conservative FNO Result

The promoted row used the 316,739-parameter FNO on the complete 512-trajectory,
101-frame, 256-cell cumulative-impulse dataset. The fixed split was 384/64/64,
with 38,400/6,400/6,400 stride-1 one-step pairs. Fifty one-step epochs supplied
240,000 optimizer updates; ten four-step recurrent epochs supplied another
46,560 optimizer updates. No input noise, limiter, floor, positive transform, or
admissibility barrier was active. Runtime was about 1 hour 54 minutes.

The epoch-59 checkpoint was selected by validation survival, then one-step loss.
On test it reaches one-step relative L2 `0.001305`, mean raw survival `0.97734`,
and 57/64 completed 20-call rollouts. Mean valid length is 19.55 calls. All seven
terminations are raw nonpositive proposals: three pressure failures and four
density failures; no nonfinite failure occurred. Mean rollout relative L2 is
`0.01444`, final-prefix relative L2 `0.03097`, shock-position MAE `0.00928`, and
final conserved-total error `0.00373`. Mean initial effective CFL is `3.84` and
the maximum is `5.64`.

The original scale-confirmation and 20-call promotion gates pass. The strict
strong-baseline gate requires at least 90% completion and fails by one case:
57/64 is `0.890625`, while 58/64 would pass. The causal within-run evidence is
the fixed validation set: the best one-step checkpoint has survival `0.635` and
5/64 completion, while the selected recurrent checkpoint has survival `0.984`
and 56/64 completion. A midscale-to-full comparison uses a different held-out
split and must not be presented as a paired data-scale effect.

Failure onset is late rather than an obvious high-CFL subgroup. Failed cases
have lower mean errors than completed cases through about call 8, then separate
rapidly after calls 10--12. The three pressure failures have much larger shock
error than the four density failures, indicating at least two residual modes:
shock-front/pressure breakdown and late density undershoot or ripple. The
result-to-claim classification remains `partial`: this is strong evidence that
a recurrently trained, global, exactly conservative FNO can learn this fixed
large-step 1D Euler map without inference-time positivity repair, but seven raw
positivity failures prevent a general no-constraint claim. Long-horizon,
target-family, seed, stride, resolution, and 2D transfer evidence is absent.

## 2026-07-15 Long-Horizon Flux-Checkpoint Result

The saved epoch-59 full-scale flux checkpoint was evaluated without retraining
at 20, 50, and 100 raw calls on the same 64 test trajectories. The 20-call
metrics reproduce exactly: 57/64 completion and mean survival `0.97734`. At 50
calls, only 1/64 completes and mean survival is `0.565625`; 63 cases terminate
on raw nonpositive proposals. At 100 calls, 0/64 completes, mean survival is
`0.28297`, and all cases have terminated by call 51. There are no nonfinite
terminations.

This fails the predeclared 50-call gate of survival at least `0.85` and
completion at least `0.50`. The full-scale flux row is therefore classified as
a strong short-horizon result and a medium-horizon recurrent-stability failure.
It does not support a claim that training accuracy can replace positivity or
stability mechanisms over arbitrary horizons.

Termination time at the 100-call request has median 27 and range 12--51. Density
becomes nonpositive in 56/64 cases and pressure in 13/64, with five cases showing
both; thus the later failure population is predominantly a density-undershoot
mode rather than the pressure-heavy earliest failures. Initial effective CFL is
not a sufficient explanation: its correlation with valid length is about
`-0.36` at this horizon, and the seven cases that already failed by call 20 had
essentially no initial-CFL association. The next stability intervention should
therefore expose the model to longer recurrent distributions before adding a
generic CFL-conditioned or whole-sample limiter.

## 2026-07-15 Strict State-Target Result

Direct next conservative state and conservative residual labels both pass the
float32 closure gate. The direct target is an uncentered map; the residual has
zero output as the identity update. With the same eight-case, 50-epoch tiny-fit
budget, direct state misses both thresholds on seed `20260707` (minimum train
relative L2 `0.01658`, normalized loss `4.33e-4`) and the declared repeat seed
`20260708` (`0.01711`, `4.84e-4`). Residual passes on the first seed at
`0.00648` and `8.08e-5`. Direct state is stopped as an optimization/centering
failure, not a label or rollout failure.

The 316,419-parameter residual head then advanced to the 64/16/16 matched screen.
After the frozen 50+10 schedule it reaches test one-step relative L2 `0.002649`,
mean survival `1.0`, 16/16 completion, zero nonpositive terminations, final
rollout relative L2 `0.04895`, and shock-position MAE `0.0240`. The matched
316,739-parameter state-loss-only flux head has `0.003438`, `0.7594`, 2/16,
14 nonpositive terminations, truncated final-prefix L2 `0.1547`, and shock MAE
`0.05184`.

Residual survives longer on 14/16 paired cases and ties on the two flux
completions. It has lower relative error on all 16 cases at their common
endpoint; the mean endpoint-error ratio is `0.316`. The paired survival gain is
`0.2406`, with trajectory-bootstrap 95% interval about `[0.134, 0.369]`. This
passes the strict midscale gate and promotes residual to 384/64/64. The causal
claim is still `partial`: this establishes a target-parameterization effect on
one split and seed, but residual does not enforce facewise conservation and has
not yet passed seed, stride, or resolution transfer.

## 2026-07-15 Full-Scale Residual and 100-Call Result

The promoted 316,419-parameter residual FNO used the same 512-trajectory ADER
artifact, 384/64/64 split, 50 one-step plus ten four-step recurrent epochs,
conservative coordinates, and unconstrained inference contract as the flux
comparison. Runtime was `6014 s` (about 1 hour 40 minutes). Epoch 60 was selected.
On test, one-step relative L2 is `0.001123`; all 64 trajectories complete 20
calls with final relative L2 `0.01215`, shock-position MAE `0.00428`, and
conserved-total error `0.00316`. Recurrent training improves the best validation
20-call error by `39.3%` relative to the best one-step-only checkpoint while
preserving full survival.

Without retraining, 62/64 cases complete 50 calls (survival `0.990`) and 61/64
complete 100 (survival `0.97781`). Completed-case 100-call final error has mean
`0.0551`, median `0.0482`, and p90 `0.104`; aggregate conserved-total error is
`0.00927`. No density or nonfinite termination occurs. The three pressure
failures occur at calls 33, 37, and 91 near interior moving waves, after local
velocity overshoot and pressure/internal-energy collapse. Valid length has
essentially zero truth-CFL correlation (`r` about `0.01`).

Against the matched flux checkpoint, residual is lower-error on 58/64 cases at
their 20-call common endpoint; the paired mean difference is `-0.0189` with
bootstrap 95% interval about `[-0.0304, -0.00975]`. At the longer common
endpoint it wins 63/64 and has median error ratio `0.165`. The legacy pressure-
argmax shock mean is affected by multi-wave rank switching: focused replay shows
the large residual outliers retain both fronts at the correct locations but
mis-rank their gradient amplitudes. Report top-k wave matching and amplitude
error alongside the scalar argmax metric.

Result-to-claim remains `partial`. This is a strong unconstrained baseline on one
fixed split/seed and a decisive target-parameterization result against the
matched state-loss-only face-flux head. It is not structurally facewise
conservative and does not establish seed, stride, resolution, or 2D transfer.

## 2026-07-04 Diagnostic Update

Artifacts created under ignored `artifacts/time_dependent_no/`:

| Artifact | Purpose |
| --- | --- |
| `cpg_mechanistic_subset_20260704/` | Local compact subset pulled from AutoDL: trajectories 0, 6, 11, 13, 17 for one-step and two-stage, with `predicteds`, `targets`, `pos`, `edges`, and `node_type`. |
| `cpg_mechanistic_diagnostic_20260704_subset/` | First selected-subset diagnostic and key findings. |
| `cpg_mechanistic_diagnostic_20260704_full/` | Full 20-trajectory AR rollout diagnostics for one-step and two-stage CPGNet. |
| `cpg_teacher_forced_per_time_20260704/` | Per-time teacher-forced RMSE for both bs2 checkpoints. |
| `cpg_state_drift_20260704_full/` | Sampled train-stat range/z-score drift diagnostic for full rollouts. |
| `cpg_perturbation_amplification_20260704/` | Controlled perturbation amplification probe for one-step and two-stage CPGNet. |
| `cpg_shock_overlay_gallery_20260704/` | Selected-subset pressure/shock overlay gallery for qualitative phase/shape readout. |
| `cpg_cloud_diagnostics_20260704_summary.md` | Compact interpretation of the full cloud diagnostics. |


## 2026-07-04 Literature-Informed Update

The diagnostic plan now incorporates lessons from four neural-operator failure-mode papers: spectral FNO analysis, APEBench, PDE-Refiner, and Recurrent Neural Operators. The main additions are:

- Treat one-step accuracy and autoregressive stability as separate quantities in every report.
- Add correlation-time and geometric rollout aggregation so long-horizon behavior is visible even when final-step averages are noisy.
- Add graph-native scale/spectral residual diagnostics to test whether low-amplitude or high-frequency shock-local residuals drive rollout failure.
- Add an effective-CFL/receptive-field audit to compare shock-front motion against graph spacing and model propagation depth.
- Defer recurrent/unrolled control experiments until the scale/resolution and propagation diagnostics identify a concrete target.

Blockers recorded at that time:

- Approximate/mesh-weighted conservation was blocked until geometric weights
  could be validated; D009 later closed without that run.
- Corrected PCNO replay was complete through the collaborator-compatible
  preprocessing path. Subsequent PCNO conclusions were required to use that
  path, not the retired raw-HDF5 adapter.
- The 1D scale/spectral diagnostic and D025 pilot were complete. The
  graph-native 2D extension was pending then and later closed through D013;
  D025 remained closed.
- New 2D CPG/PCNO recurrent controls were blocked on D014 at that time; D015
  later completed only the bounded pilot recorded above.

## 2026-07-05 Interface-Latent Run Result

D017 completed on AutoDL for the selected trajectories/frames. Compact outputs are under ignored `artifacts/time_dependent_no/cpg_interface_latent_diagnostic_20260705_full/`, including `summary.json`, `per_frame_summary.csv`, `per_wave_type_summary.csv`, `sampled_edges.csv`, and `aggregate_analysis.md`.

Main evidence: decoded interface states remain density/pressure-admissible but are not physical one-sided traces; pressure boundedness between adjacent node states is near zero and owner-closer fractions are only about 0.52. The LLF/FV path is internally consistent with the model delta, but its true-update error remains nontrivial, especially around shocks. Dissipation is enriched near shocks only weakly: top-decile dissipative edges are about 17% target shock-front edges versus an 8.6% base shock-edge fraction. Frame-motion/wave-speed diagnostics are noisy and do not support a single too-fast or too-slow explanation.

## 2026-07-05 Interface-Latent Priority Update

Idea 2.2 is now the next diagnostic priority before Idea 2.1 target-ladder training. The new script should be run first on the selected hard/representative trajectories and frames for both bs2 checkpoints. Its immediate purpose is to decide whether learned `reconstruct_prims` behaves like physical one-sided traces, space-time averaged predictors, flux coordinates, hidden dissipation, or a wrong wave-speed / shock-shape mechanism.

Local implementation status:

- The exact one-off implementation was retired from the active tree during cleanup after the D017/D018 results were recorded.
- Full D017 evidence has been generated under ignored interface-latent diagnostic artifacts.

## 2026-07-05 Interface Mechanism Probe Result

D018 completed on AutoDL for the same selected trajectories/frames as D017, now in both teacher-forced and autoregressive state-source modes. Compact outputs are under ignored `artifacts/time_dependent_no/cpg_interface_mechanism_probe_20260705_full/`, including `summary.json`, `per_frame_summary.csv`, `projection_summary.csv`, `inverse_fit_summary.csv`, `report.md`, and `aggregate_analysis.md`.

Main evidence: replacing learned interface states with physical candidates or componentwise-bounded projections changes the induced model update substantially. The best physical projection (`expanded_clip`) still has update-vs-model energy relative L2 about 1.7-1.9 and learned-flux relative L2 about 7.8-7.9. Constrained inverse fits inside strict local owner-neighbor boxes almost never match learned fluxes within 10%; one-jump-expanded boxes improve the median residual but still leave relative L2 medians around 1.4-3.9 and high p90 residuals. AR-mode inputs increase learned true-update energy relative L2 from about 0.075-0.080 teacher-forced to about 0.79-0.99 on frames >0, while the internal learned update still matches the model delta at roughly 1e-6.

Interpretation: `reconstruct_prims` is not just a strangely scaled physical trace. It behaves as a functional nonphysical flux-control coordinate; projection to plausible physical states destroys the model update, and the learned flux is generally not close to a local physical LLF flux manifold. This strengthens the case for Idea 2.1 target-ladder training with explicit physical/interface/flux targets rather than supervising PCNO-FV from raw CPGNet latents.

## 2026-07-05 Corrected PCNO Preprocessed Rollout Result

After collaborator feedback, D019 uses the corrected PCNO replay path: HDF5-to-npy conversion and reconstruction before loading the Euler PCNO `.npz` contract. The active script is `scripts/time_dependent_no/rollout_pcno_preprocessed.py`; the raw-HDF5 graph adapter was retired from the active tree.

Corrected selected-rollout artifacts are under ignored `artifacts/time_dependent_no/`, including `pcno_corrected_rollout_20260705_selected/` and `pcno_corrected_animation_gallery_20260706_selected/`.

Main evidence: PCNO initially places the shock front more accurately than CPGNet, but nonphysical ripple artifacts grow over time and can trigger rollout crash. In the selected corrected replay, pressure mean RMSE is about 2.13 and final pressure RMSE about 3.48; velocity errors can reach overflow scale. The next useful diagnostics are direct ripple-energy and shock-front-position metrics, not time-lag curves.
## 2026-07-06 Active-Tree Cleanup

The tracked `time_dependent_no` surface was reduced to reusable utilities, active rollout/visualization diagnostics, corrected PCNO replay, and compact documentation. One-off inspection, smoke, state-drift, perturbation, time-alignment, raw-HDF5 PCNO adapter, and CPGNet interface-latent probe implementations were removed from the active tree after their conclusions were recorded here. Use git history for exact reproduction; do not treat retired paths as current entry points.
## 2026-07-09 Idea 2.1 Priority Update

The active weekly objective is now Idea 2.1: solver-facing target diagnostics. Use 1D Euler as the fast pilot to compare target parameterizations before transferring only the useful stabilized variants to CPGNet-style and 2D bump runs.

The target ladder was kept compact: one nonzero training-noise level was chosen
from the follow-up batch, then the FNO stride-4 selector ran over
`limited_residual`, `limited_flux`, and `positive_limited_interface` with noise
`0` and the selected nonzero noise. Seed confirmation was limited to the best
one or two variants.

This target-ladder work also supports later Idea 2.2. By making models predict residuals, fluxes, or interface states through explicit adapters, the learned quantities become inspectable as physical traces, flux corrections, dissipation controls, or nonphysical update coordinates.

## 2026-07-09 1D Euler Noise Follow-up Result

The `euler1d_noise_followups_v2` batch completed on AutoDL. Compact local analysis was written under ignored `artifacts/time_dependent_no/euler1d_noise_followups_v2_analysis_20260709/`.

Main evidence: for `FNO + limited_residual + stride 4`, three-seed final rollout L2 was `0.0861 +/- 0.0329` with no noise, `0.0754 +/- 0.0173` with noise `0.003`, and `0.0755 +/- 0.0055` with noise `0.02`. Noise `0.003` gave the best mean rollout/conservation without pressure-floor hugging. Noise `0.02` gave much better shock MAE (`0.0058 +/- 0.0015`) but had worse one-step error and minimum pressure at the floor.

Limiter control: raw conservative residual did not crash in the small stride-4 control and beat limited residual at noise `0` for one seed, but at noise `0.02` limited residual had better final rollout and shock metrics. Interpret this as evidence that the current limiter may be conservative/diffusive, not that admissibility control is unnecessary.

Stride-2 check: zero-noise stride 2 had nonpositive raw-pressure counts and final L2 `0.1318`; noise `0.02` improved final L2 to `0.0962`, conservation to `0.0286`, and shock MAE to `0.0061`. Smaller stride does not automatically stabilize rollout because it increases the number of learned operator applications.

Decision: use noise `0.003` as the default nonzero setting for the first stabilized target selector. Keep noise `0.02` as a shock-stability stress setting, especially when a target family is shock-unstable after the `0`/`0.003` selector.

## 2026-07-09 1D Euler Stabilized Target Selector Launch

Implemented the shared `ConservativeUpdateLimiter`, `limited_flux`, and `positive_limited_interface` target adapters. `limited_flux` and `positive_limited_interface` use samplewise limiting so a single limiter coefficient scales the whole finite-volume update for each trajectory sample, preserving interior face-pair conservation accounting. Existing `limited_residual` keeps cellwise limiting for backward compatibility with the prior residual experiments.

Local verification before launch: `uvx ruff check` passed for the touched Python files, `tests/time_dependent_no` passed, and tiny FNO CPU smoke runs completed for both new targets. Remote CPU smoke runs on the real 1D Euler dataset also completed for `limited_flux` and `positive_limited_interface`.

The compact AutoDL selector `euler1d_target_selector_v1` was launched on 2026-07-09 at 23:08 CST. Matrix: FNO, stride 4, 40 epochs, 384 train cases, 64 test cases, targets `limited_residual`, `limited_flux`, and `positive_limited_interface`, with training noise `0` and `0.003`. The first run (`limited_residual`, noise `0`) reached epoch 2/40 with finite training and test metrics. Output directories use the relative pattern `artifacts/time_dependent_no/target_selector_v1_fno_*`; the latest log path is recorded in `artifacts/time_dependent_no/euler1d_target_selector_v1_latest_log.txt`.
## 2026-07-10 1D Euler Target Selector V1 Result

The compact FNO stride-4 stabilized-target selector completed all six runs. Ignored local analysis artifacts are under `artifacts/time_dependent_no/target_selector_v1_analysis_20260710/`, with downloaded lightweight metrics under `artifacts/time_dependent_no/target_selector_v1_metrics_20260710/`.

Main evidence: one-step relative L2 was similar across targets (`0.0059` to `0.0081`), but rollout separated sharply. `limited_residual` with noise `0.003` had the best final rollout L2 (`0.0933`), shock MAE (`0.0175`), and conservation error (`0.0457`), improving over zero-noise `limited_residual` (`0.1231`, `0.0305`, `0.0663`). However, it touched the pressure floor in a few rollout states, so it remains a cautionary baseline rather than a clean structure-preserving solution.

Current flux/interface target forms failed the selector. `limited_flux` final rollout L2 was `0.6138` at noise `0` and `0.4182` at noise `0.003`; `positive_limited_interface` was `1.3574` at noise `0` and `0.3594` at noise `0.003`. Both families had large shock MAE, pressure-floor hugging, nonpositive raw pressure after conservative decoding, and high limiter activation during rollout, despite stable one-step metrics.

Interpretation: conservation-form decoding alone is not enough. The current absolute flux and absolute interface-state parameterizations are structurally conservative but dynamically wrong under autoregressive distribution shift. The limiter prevents immediate crash but becomes an emergency clamp. Next target work should prioritize physical base flux plus bounded correction, direct macro-step face-flux supervision from the generator, and bounded interface corrections around local/Riemann base states.
## 2026-07-10 Physical Flux-Correction Target Launch

Implemented `physical_flux_correction` for the 1D Euler target ladder: the network predicts a bounded correction around the current-state Rusanov face flux, then the finite-volume update and shared samplewise conservative admissibility limiter decode the next state. The target exposes correction-scale diagnostics: mean/max absolute correction over bound and saturation fraction.

Local verification: focused solver-target tests passed, `tests/time_dependent_no` passed, script help/smoke checks passed, and ruff passed with the established E402 ignore for script entry points.

Remote smoke on the real 1D dataset passed. The zero-correction/base-Rusanov audit at stride 4 was poor (`one_step_l2` about `0.198`, final rollout L2 about `0.818`, shock MAE about `0.410`, and rollout limiter activation about `0.999`), so the learned correction must do real macro-step work rather than lightly polishing a good classical step.

The first full scale-1 launch was interrupted after early live diagnostics showed test relative L2 near `0.10` and correction saturation around `0.18-0.20` with no teacher-forced limiter activation. The decision at that point was to run a short bound-scale probe over correction scales `1`, `2`, and `4` before spending the full training budget. Its ignored outputs used the relative pattern `artifacts/time_dependent_no/physical_flux_scale_probe_v1_*`, with log `artifacts/time_dependent_no/logs/euler1d_physical_flux_scale_probe_v1.log`.
## 2026-07-10 Physical Flux-Correction Scale Probe Result

The short scale probe over correction scales `1`, `2`, and `4` completed on AutoDL. Ignored local analysis artifacts are under `artifacts/time_dependent_no/physical_flux_scale_probe_v1_analysis_20260710/`, with lightweight downloaded metrics under `artifacts/time_dependent_no/physical_flux_scale_probe_v1_metrics_20260710/`.

Main evidence: all three physical-flux-correction scale settings were rejected. Scale `1` had one-step L2 `0.0991` and final rollout L2 `1.1100`; scale `2` had one-step L2 `0.0850` and final rollout L2 `0.8259`; scale `4` had one-step L2 `0.0805` and final rollout L2 `0.8568`. All reached pressure floor, had nonpositive raw pressure counts, and had rollout limiter activation about `0.95-0.96` with minimum theta `0`.

Interpretation: increasing the correction bound improves one-step fit but does not stabilize rollout. Scale `1` is correction-bound-limited, while scale `4` largely removes correction saturation but activates the teacher-forced limiter heavily. The base audit showed the core issue: one explicit Rusanov flux over the stride-4 macro step is a poor large-timestep anchor, so the network must cancel and replace the base flux rather than learn a small physical correction.

Decision: do not run the full noise `0`/`0.003` selector for this exact target family. Next flux-target work should prioritize direct macro-step time-integrated face-flux supervision or a stable/data-derived macro flux base.

## 2026-07-13 Corrected Solver-Level CPGNet Gate

The earlier `CPGNetEuler1DHead` rows are now deprecated: that generic directed target head did not execute the paper's solver-level interface-state/FV recurrence. The corrected solver-level implementation is `CPGNetEuler1D` with the exclusive `cpg_interface` target. It uses physical ghost nodes, directed geometry-only edge encoding, 12 unshared message-passing layers, target-node interface reconstruction, positive density/pressure interface decoders, one shared oriented Rusanov flux per face, and an exact finite-volume update. Left inflow is anchored to the case state; the right-wall exterior interface is the reflected owner-side prediction. Exact 1D geometry replaces the release model's learned positive geometry factor.

No post-update cell-state floor or admissibility limiter is used in the corrected recurrence. Invalid raw density/pressure terminates and is counted by rollout diagnostics. Training now supports the paper-compatible two-stage schedule: one-step standardized next-state loss followed by three-step fully differentiable autoregressive fine-tuning with reduced learning rate and additive Gaussian primitive-input noise. The existing FNO path retains its admissibility-preserving log-normal density/pressure noise. Checkpoint selection prioritizes completed admissible validation rollouts by final error; if every candidate fails the horizon, one-step validation loss selects the best-fit failure for diagnosis rather than silently freezing on epoch 1.

Local verification: all `tests/time_dependent_no` tests passed (`55 passed`), and a small synthetic CPU gate with the additive CPG noise path completed without nonpositive raw states. With a 16-hidden-channel, 2-layer smoke model trained for five one-step epochs plus one three-step autoregressive epoch, one-step relative L2 was `0.00177` and four-step final-rollout L2 was `0.00696`; final conservation error was `0.00129`. These numbers validate the implementation path only and are not benchmark evidence.

This solver-level launch gate was closed by D020. The h128/mp28 run supplied the
competitive-fit, raw-recurrence reference; the later shock-tail diagnostic and
mp12/h193 and mp28/h85 controls completed. This gate leaves no current CPG work
authorized.

## 2026-07-14 1D Euler Receptive-Field Result

The matched h128 intervention completed on AutoDL with the same data split,
stride-4 operator, seed, noise, optimizer, 15 one-step epochs, and five
three-step unrolled epochs. The only architectural change was unshared directed
message-passing depth, although that also increased parameter count.

| Variant | Parameters | Best Epoch | One-Step Loss | One-Step Rel L2 | Completion | Mean Survival | Rollout Mean L2 | Final L2 | Shock MAE | Conservation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CPGNet h128/mp12 | 1,500,931 | 3 | 3.49e-03 | 3.34e-02 | 34/64 | 0.695 | 2.14e-01* | 2.68e-01* | 1.68e-01* | 2.75e-02* |
| CPGNet h128/mp28 | 3,350,275 | 20 | 6.75e-05 | 6.14e-03 | 64/64 | 1.000 | 2.06e-02 | 2.98e-02 | 2.57e-02 | 1.03e-02 |
| FNO 64/24/4, limited residual | 316,419 | 79 | 6.43e-05 | 5.10e-03 | 64/64 | 1.000 | 2.16e-02 | 3.72e-02 | 2.51e-03 | 1.89e-02 |

`*` mp12 aggregates use variable-length valid prefixes because 30 raw
rollouts terminate; they are not full-horizon estimates.

Mechanistic evidence:

- mp28 lowered the initial-rollout error on all 64 paired test cases. The mean
  fell from 0.1366 to 0.01557, with paired bootstrap mean difference 95% CI
  [-0.1326, -0.1095].
- All 30 mp12 failures were rescued by mp28. mp12 completion was 10% for cases
  with truth effective CFL in [20, 24), and zero above 24; mp28 completed every
  bin through the observed maximum 28.44.
- Initial effective CFL strongly predicted mp12 first-step error
  (Pearson/Spearman 0.901/0.900) but not mp28 error (-0.120/-0.197).
- Against FNO, mp28 is slightly worse at early steps but accumulates error more
  slowly. Its final L2 is 19.9% lower (paired p=0.020), while mean rollout L2
  differs by only 4.4% and is not significant (paired p=0.199).
- mp28 conservation error is 45.7% lower than FNO and wins on 75% of cases.
  Both methods complete all raw rollouts without active limiting.
- mp28 shock median is 0.00234, close to FNO's 0.00176, but its tail is much
  worse: p95 0.0747 and maximum 0.5785 versus FNO p95 0.00684 and maximum
  0.00801. Three cases dominate the mp28 mean shock error.
- Saved-checkpoint replay of those three cases shows a real but
  argmax-amplified failure: mp28 retains a strong nearly stationary pressure
  gradient near the original discontinuity while the target front moves.
  Predicted second/first gradient ratios are about 0.65-0.93, so a largest-front
  metric switches between two comparable fronts, but the persistent ghost
  front is not a metric artifact.

At this checkpoint, the result-to-claim gate was `partial` with high internal
confidence. Independent Codex review was not performed because unpublished
results were not sent to an external tool without approval; this was a
historical review-status note, not an open queue item. The supported statement
is narrow: on this
fixed 1D Euler stride-4 dataset, a CPGNet whose hop depth covers the observed
macro-step domain of dependence can learn a stable feed-forward macro flow map,
whereas mp12 underfits and fails high-CFL cases. The result does not yet prove
that hop depth rather than added capacity causes the full gain, nor does it
establish timestep/resolution transfer or robust shock tracking.

Completion update, 2026-07-15:

- The mp12/h193 and mp28/h85 controls are complete. Width does not recover the
  shallow model and the narrow deep model retains the gain, supporting hop depth
  as the primary mechanism. Freeze h128/mp28 and stop the CPG architecture
  sweep.
- The four-way FNO coordinate matrix is complete. Conservative input, loss, and
  recurrence with fixed physical scaling are the provisional D021 contract.
- The 512-case ADER cumulative-impulse dataset and serialized closure checks are
  complete. The exact face-value supervision screen then reached the midscale
  stop documented above; it is no longer queued or waiting.
- The matched stability screen is complete. Unroll-only passed its midscale
  promotion gate; the weight-`0.1` barrier failed attribution.
- The frozen full-scale unroll-only confirmation is complete at 57/64 raw
  completion and survival `0.97734`. It passes scale and 20-call promotion gates
  but misses the strict 90% completion gate by one trajectory.
- The frozen flux checkpoint then fails the 50-call extension at 1/64
  completion. Direct next state fails tiny fit on both declared seeds. The
  completed full-scale residual checkpoint reaches 64/64, 62/64, and 61/64 at
  20/50/100 calls and wins 63/64 longer common-endpoint comparisons. At that
  point D021 remained open for seed confirmation, conservation-compatible
  residual projection, stride, and resolution gates; those gates later
  completed through D031.

## 2026-07-16 Three-Seed Residual Projection Result

The matched full-scale comparison used the frozen 512-trajectory split,
conservative coordinates, 50 one-step plus ten four-step recurrent epochs, and
20/50/100 raw calls. The residual FNO has 316,419 parameters; the projected
variant has 316,806. Neither uses noise, a limiter, a floor, a positive
transform, or an admissibility loss. The projected decoder separates a
volume-zero cell increment from one learned three-component boundary budget;
its decoded increment closes to that budget at about `1.0e-7` absolute error.

| Seed | One-step L2 | 100-call completed | 100-call final L2 | Conserved-total error | Shock MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20260707 | 0.001133 | 62/64 | 0.04730 | 0.00720 | 0.00435 |
| 20260708 | 0.001060 | 61/64 | 0.05940 | 0.00834 | 0.00622 |
| 20260709 | 0.001113 | 64/64 | 0.06136 | 0.00904 | 0.00355 |

Projection raises pooled completion from 187/192 to 189/192 at 50 calls and
from 182/192 to 187/192 at 100 calls, so the preregistered stability-
noninferiority gate passes. The other family gates fail: only one seed improves
100-call state error; the three-seed mean state-error ratio is `1.016`; only one
seed improves conservation at both 50 and 100 calls; and the mean conservation
ratios are `0.953` and `0.871`, short of the prespecified robust criterion.
Both new seeds also miss the broad frame-20 gate because shock MAE is worse.

Variable-length prefix averages are optimistic for different models on
different cases. Among the 181 seed-case pairs completed by both methods at 100
calls, projection is 9.8% worse in final state L2, 5.9% better in conserved-total
error, and 4.8% worse in shock MAE. At each trajectory's last common valid step,
its pooled state L2 is 2.5% worse. All projected failures are raw nonpositive-
pressure events on the same hard case family as residual; the parameterization
changes which seeds enter that failure basin rather than eliminating it.

Classification is `partial`: the low-dimensional boundary-budget factorization
is an exact and useful conservation coordinate and gives a real stability
signal, but it is not a seed-stable accuracy or shock improvement. Keep plain
residual as the strong accuracy baseline and projected residual as a structural
ablation. The next bounded intervention is generated burn-in plus detached
short BPTT; do not reopen full face-value supervision or add an inference
limiter.

## 2026-07-16 Generated-State Burn-In Pilot

The matched 64/16/16 pilot used the 316,806-parameter projected-residual FNO,
seed `20260708`, the fixed conservative-coordinate contract, and no noise,
limiter, floor, positive transform, or admissibility loss. Both rows used 50
one-step epochs and four supervised recurrent steps. The control used no
burn-in for ten recurrent epochs (`7,760` updates); the intervention rolled
eight detached model steps before supervision for eleven epochs (`7,832`
updates, `1.009x`). Their one-step histories match exactly and both select
recurrent epoch 58. Total wall time, including sanity and three horizons per
row, was about 41 minutes.

The generated states are a materially harder training distribution without
being inadmissible. Burn-in relative L2 falls from `0.01057` to `0.00897` over
the second stage; no burn-in sample has nonpositive density or pressure, and
the observed minima are `0.0831` and `0.0675`. The supervised four-step train
loss starts about an order of magnitude above the clean-start control and then
falls by about 47%. This confirms that the intervention is active and that
clean four-step BPTT underexposes the model to later accumulated errors.

| Calls | Clean completed | Burn-in completed | Survival clean / burn-in | State-L2 ratio | Conserved-total ratio | Shock ratio | Paired burn-in wins |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | 16/16 | 16/16 | 1.000 / 1.000 | 1.021 | 0.924 | 1.382 | 8/16 |
| 50 | 15/16 | 15/16 | 0.9988 / 0.9950 | 0.768 | 0.659 | 1.068 | 15/16 |
| 100 | 14/16 | 14/16 | 0.9650 / 0.9394 | 0.638 | 0.537 | 0.936 | 14/16 |

The predeclared primary gate fails on 20-call shock regression and the 0.01
100-call survival tolerance. The aggregate failure is not broad. Burn-in
rescues case 21, whose clean pressure becomes nonpositive at call 96, but case
234 changes from completed to a density/pressure failure at call 58 and case
238's pressure failure advances from call 50 to 47. The 20-call shock increase
is also concentrated: case 238 explains about 56% of the mean increase, and
cases 183 and 348 bring the concentration to about 85%.

Classification is `partial recurrent-distribution improvement with rare-case
stability regression`. Later-state exposure strongly reduces medium/long-
horizon error on most cases, but eight burn-in calls do not sample every later
failure basin and are not a positivity mechanism. Do not promote projected
residual to a full-scale accuracy row. The matched plain-residual exposure gate
below is the relevant accuracy-baseline continuation; defer a pressure/internal-
energy penalty until its mechanism controls resolve the active failure mode.

## 2026-07-16 Plain-Residual Generated-State Exposure Result

The matched 64/16/16 plain-residual gate used the 316,419-parameter FNO, seed
`20260708`, conservative coordinates, no noise or constraint, 50 identical
one-step epochs, and four supervised recurrent steps. The clean control used
ten recurrent epochs and 7,760 optimizer updates. The intervention used eight
detached generated steps followed by four-step BPTT for eleven epochs and 7,832
updates (`1.009x`). Generated prefix error fell from `0.01008` to
`0.00922`, with no nonpositive prefix states.

| Calls | Clean completed | Burn-in completed | Survival clean / burn-in | Common-endpoint state ratio | Conserved-total ratio | Legacy shock ratio | Burn-in wins |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | 16/16 | 16/16 | 1.000 / 1.000 | 0.954 | 0.923 | 1.456 | 9/16 |
| 50 | 15/16 | 16/16 | 0.9975 / 1.000 | 0.678 | 0.592 | 1.145 | 11/16 |
| 100 | 15/16 | 16/16 | 0.9675 / 1.000 | 0.486 | 0.526 | 1.066 | 14/16 |

The formal primary gate fails only the 20-call legacy shock criterion. Three
cases explain nearly all of that mean increase. Cases 114 and 21 contain
important front-strength/rank changes; case 238 is a real short-horizon
regression with an extra or displaced strong pressure front. Do not erase the
failed gate after this adjudication. Subsequent runs report both the legacy
argmax and separated top-two position/strength metrics.

Classification is `strong later-state-exposure signal with unresolved causal
source and one real shock regression`. The current generated-burn objective
supervises the reference next state after a perturbed model prefix. For a
generated error `e`, the consistent PDE target is the reference solver
advanced from the generated state, which differs to first order by the solver
Jacobian acting on `e`. The present label can therefore train a trajectory
correction or denoiser. D022 holds the later-time window fixed and replaces the
generated prefix with the exact offset state; D023 then advances generated
states with the reference solver. Full-scale promotion and a pressure penalty
wait for those controls.

## 2026-07-16 Teacher-Offset Mechanism Result

D022 used the same 316,419-parameter residual FNO, 64/16/16 split, seed
`20260708`, 50 one-step epochs, and four supervised recurrent steps as the
plain-residual gate. Teacher offset and generated burn-in both use the same
eight-step-offset windows, eleven recurrent epochs, and 7,832 optimizer
updates. Teacher starts from the exact reference state at the offset; generated
uses eight detached model calls. One-step histories match exactly. Teacher
prefix L2 and nonpositive fraction are both zero.

| Calls | Teacher / clean state ratio | Generated / teacher state ratio | Generated / teacher state-ratio 95% interval | Generated wins vs teacher | Completion clean / teacher / generated |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | 1.109 | 0.861 | [0.782, 0.948] | 13/16 | 16 / 16 / 16 |
| 50 | 1.017 | 0.661 | [0.470, 0.854] | 13/16 | 15 / 16 / 16 |
| 100 | 1.006 | 0.492 | [0.348, 0.669] | 15/16 | 15 / 14 / 16 |

Teacher offset is significantly worse than clean at H20 and statistically tied
at H50/H100. It postpones case 21's pressure failure from call 49 to 51 but
adds a case-114 pressure failure at call 89. Generated exposure completes every
case and cuts conserved-total error relative to teacher by 24%, 46%, and 50%.
The selected teacher checkpoint has one-step relative L2 `0.002660`, better
than generated's `0.002886` despite much worse H50/H100 behavior. This is
another direct inversion between one-step and rollout ranking.

The causal fixed-setting conclusion is that generated off-manifold state
exposure contributes beyond later physical-time sampling. The method-level
result remains partial. Relative to clean, generated top-two front-position
error is 17.8% worse at H20 and 10.2% worse at H50; trajectory-bootstrap
intervals exclude parity. At H100 that front-position difference reverses in
mean but is not significant. Also, the generated-state objective uses the
original reference next state, not the reference solver advanced from the
generated state. D023 must quantify that operator-consistency gap before
full-scale promotion.

The local result-to-claim verdict at this stage was `partial` with high internal confidence:
the narrow D022 mechanism claim is supported, while a PDE-consistent burn-in
method claim is not. Independent Codex review was not performed because
unpublished results were not sent to an external tool without approval; this
was not left as an active task.

## 2026-07-16 D023 Frozen Solver-Consistency Protocol

D023 freezes the clean, teacher-offset, and generated-burn-in residual FNOs.
The generated checkpoint constructs one shared state bank over all 16 held-out
cases, start frames `0,10,...,80`, and prefix depths `0,2,4,8`. Each state is
advanced one saved interval by the dataset WENO-HLLC-ADER solver with its CFL
substeps, retry logic, shock flattening, HLLE troubled-face fallback, inflow,
and reflective wall. All three learned maps are evaluated on that identical
state. Prefix depth zero controls for replaying float32 serialized snapshots.

The primary readout is fixed-physical-scale conservative RMSE to (a) the
same-state solver continuation and (b) the original stored next state. It also
records primitive error, shock/smooth splits, three linearized characteristic
families, separated top-two pressure-front position/strength, and the cosine
between the learned correction from the solver continuation and the direction
back to the original trajectory. Ratios use 2,000 bootstrap replicates over
held-out case IDs, keeping start frames clustered within each case.

At prefix depth eight, call the result PDE-map improvement only if the
generated/teacher ratios to both targets are at most `0.90`. Call it trajectory
correction or denoising only if the original-trajectory ratio is at most
`0.90`, the solver-continuation ratio is at least `0.95`, and the generated
correction-alignment cosine is positive. A solver ratio at most `0.90` with a
truth ratio at least `0.95` is PDE consistency without trajectory recovery;
anything else is mixed or inconclusive. This diagnostic identifies the
existing objective's mechanism and does not by itself justify full-scale or
multi-seed promotion.

## 2026-07-16 D023 Solver-Consistency Result

The 576 same-state solver advances and 1,728 learned-map evaluations completed
in `56.9 s`. Prefix-zero replay RMSE is `1.17e-7` (maximum `4.02e-7`), versus
`1.90e-2` after an eight-step generated prefix, and none of the solver advances
uses a retry or first-order fallback. Serialization or replay failure therefore
does not explain the result.

| Prefix depth | Generated / teacher truth RMSE | 95% interval | Generated / teacher solver RMSE | 95% interval |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 1.147 | [1.086, 1.215] | 1.147 | [1.080, 1.223] |
| 2 | 1.037 | [1.011, 1.069] | 1.019 | [0.997, 1.045] |
| 4 | 1.011 | [0.996, 1.027] | 1.024 | [1.007, 1.044] |
| 8 | 0.992 | [0.984, 0.999] | 1.017 | [1.006, 1.028] |

Generated burn-in gradually removes its clean-state one-step disadvantage and
is 0.8% better than teacher against the original trajectory at depth eight,
but it is 1.7% worse against the actual continuation from that state. Clean is
2.6% closer to the solver than generated. The depth-eight generated solver
defect is 6.3% worse than teacher in the shock region and 8.0% worse than clean;
its truth-front strength error is also 5.9% worse than teacher. The generated
correction from the solver continuation has mean alignment `0.074` toward the
original trajectory (median `0.054`) and norm `0.407` of the solver-to-truth
defect. This is weak trajectory bias, not a dominant denoising map.

The preregistered classification is `mixed_or_inconclusive`. D022 still shows
that exposure to generated states changes closed-loop behavior, but D023 rules
out the simple explanation that it materially improves the local PDE map on
those states. A small systematic modified-equation effect can compound over
100 calls. The completed D013 and D024 results below identify its spectral
signature and reject blind local dissipation as the remedy.

## 2026-07-16 D013 Scale/Spectral Result

The exact frozen-checkpoint run covered the full residual and state-loss-only
flux FNOs, corrected CPGNet mp28, and the D022 clean, teacher-offset, and
generated-exposure residual models. The full residual model completes 61/64
raw H100 rollouts. The flux model completes 0/64 by H60; first failures occur
at call 13 and last failures at call 52, with mean valid survival 28.30 calls.

Teacher-forced fit does not expose a high-frequency disadvantage for the flux
head. Relative to residual, its state error is 1.031, modes-25--64 error is
0.982, second-difference error is 1.013, and its divergence-active update
high-band projection gain is essentially identical (0.9862 versus 0.9860).
The instability appears only under recurrence. From eight calls before failure
to the invalid proposal, flux-model state error grows 10.23 times,
modes-25--64 error 7.45 times, Nyquist-tail error 27.43 times, first-difference
error 23.84 times, and second-difference error 26.23 times. At the same
case/frame, the failure has 10.37 times the residual model's high-band error,
254.7 times its Nyquist-tail error, and 154.2 times its second-difference
error. Error starts shock-local but spreads into the smooth region by failure.

The preregistered classification is
`recurrent_high_frequency_growth_not_one_step_fit`. D022 generated exposure
reduces state error while increasing the error spectral centroid, so it is not
a clean learned-viscosity mechanism. This result justified D024 as a narrow
causal probe, not blanket smoothing as a method.

## 2026-07-16 D024 Conservative-Dissipation Result

D024 freezes the full stride-1 flux FNO and adds only the interior correction
`-kappa (dx / dt) (U_R - U_L)`; the added boundary flux is exactly zero.
The five paired coefficients are 0, 0.0025, 0.005, 0.01, and 0.02. No model is
retrained and no limiter or floor is used.

| Kappa | Mean valid calls | H20 complete | H50 complete | H20 state error | H20 high-band error | H20 Nyquist-tail error |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 28.28 | 57/64 | 1/64 | 0.02022 | 0.00418 | 0.00943 |
| 0.0025 | 27.86 | 57/64 | 0/64 | 0.02234 | 0.00440 | 0.01202 |
| 0.005 | 28.06 | 57/64 | 1/64 | 0.02206 | 0.00420 | 0.01162 |
| 0.01 | 27.30 | 57/64 | 0/64 | 0.02222 | 0.00409 | 0.01249 |
| 0.02 | 26.23 | 55/64 | 0/64 | 0.02416 | 0.00397 | 0.01562 |

For the selected small coefficient 0.005, the paired mean-survival change is
-0.22 calls with a 95% interval spanning zero; H20 state, tail, and shock-width
errors are 1.091, 1.231, and 1.045 times the no-correction control. Larger
diffusion slightly lowers the broad modes-25--64 metric but amplifies the
Nyquist tail, worsens front/width metrics, and shortens survival. The
classification is `no_material_stability_gain`: blind current-state
Laplacian viscosity does not repair the learned nonlinear recurrence.

## 2026-07-16 D025 Global Implicit-Interface Pilot

D025 adds one global face-grid FNO target with six outputs per face. Two
directed traces use multiplicative exponential density/pressure corrections
and sound-speed-scaled additive velocity corrections around the local physical
states. The exterior traces obey the fixed inflow and reflective wall
conditions. One shared Rusanov or central flux then feeds the exact finite-
volume update. Training uses decoded state loss only; recurrence is raw, with
no interface label, limiter, state floor, or positivity penalty.

The matched 8/4/4 tiny runs use 317,126 parameters, 100 one-step epochs, and
the same seed and split.

| Decoder | Minimum train relative L2 | Minimum validation relative L2 | Selected test one-step relative L2 | Raw H20 completion | Failure |
| --- | ---: | ---: | ---: | ---: | --- |
| Rusanov | 0.00734 | 0.01328 | 0.01429 | 0/4 | all invalid on call 2 |
| Central | 0.00823 | 0.01351 | 0.03069 | 0/4 | all invalid on call 3 |

Thus the Rusanov row passes the preregistered fit gate, while central narrowly
misses the selected-checkpoint test threshold of 0.03. Neither is a viable
tiny-set recurrent model. A per-frame frozen analysis localizes the mismatch:
Rusanov conservative relative error is 0.130 at frame 0 versus 0.0087 over
frames 20--99; central is 0.157 versus 0.0186. The frame-0 error is dominated
by shock cells. Both models remain admissible on all 400 truth-state
evaluations, but recurrent tail/high-band error grows within two to three
calls before pressure or density failure. Rusanov's divergence-active flux
MSE is lower than central's in every time bin despite the absence of direct
flux supervision.

The 64/16/16 run localizes a genuine recurrent failure. Its 50-epoch one-step
stage reaches train/validation relative L2 `0.00559/0.00544`, but four-step
training overflows at sequence step three after the generated input has already
become inadmissible (`min rho=-0.919`, `min p=-300.8`). The network output is
still finite (`max |raw|=49.5`); decoding the invalid state produces a finite
prediction of order `1e35`. Skipping a zero-weight barrier fixed one real
trainer bug, but the identical rerun proves that the remaining failure is not
that implementation defect.

A matched two-step curriculum is finite, so three recovery controls were run
on the same 64/16/16 split and 50-call validation horizon:

| Recovery control | Selected test one-step relative L2 | Test mean H50 survival | H50 completion |
| --- | ---: | ---: | ---: |
| Uniform time sampling, barrier `0` | 0.01538 | 0.04875 | 0/16 |
| Frame-zero weight `15`, barrier `0` | 0.00791 | 0.04625 | 0/16 |
| Frame-zero weight `15`, barrier `0.1` | 0.00786 | 0.04750 | 0/16 |

Frame weighting hits its intended mechanism: it lowers initial-call
conservative error from `0.1113` to `0.0304`, shock RMSE from `0.572` to
`0.151`, and top-two front-position error from `0.0683` to `0.0135`. It does
not move the admissibility threshold: all cases still fail on calls two through
four. The barrier lowers its recurrent training value from `0.00443` to about
`1e-7` and slightly improves call-one/two errors, but mean valid calls move only
from `2.3125` to `2.375`; 0/16 complete.

The classification is `well_fitted_but_recurrently_inadmissible`. This is a
negative result for the tested state-loss-only relative-trace parameterization,
not a proof that interface-latent operators or neural operators cannot work.
The likely design liabilities are the underidentified six-trace-to-three-update
map and a zero-output anchor equal to an explicit Rusanov macro-step at effective
CFL about four to six. The FNO has global receptive field, so CPGNet locality is
not the leading explanation. Stop this row at successive halving; do not add an
inference limiter or sweep more barrier/viscosity coefficients.

## 2026-07-16 D026 Boundary-Exchange Supervision Result

D026 tests the smallest identifiable solver-flux quantity that is not affected
by the interior face gauge: the three-component net boundary exchange over one
saved interval. The target is derived directly from accepted-substep,
owner-oriented solver impulses. The projected-residual decoder predicts the
same exchange through its learned boundary budget, and the auxiliary uses a
zero-centered per-channel RMS normalization. Inference remains the exact raw
projected-residual update, with no limiter, floor, positive transform, or
post-hoc projection.

The matched gate uses the 316,806-parameter FNO, 64/16/16 cases, seed
`20260707`, 50 one-step plus ten four-step recurrent epochs, and 20 raw calls.
The zero-weight row is the frozen matched projected-residual control.

| Boundary weight | One-step relative L2 | Boundary-exchange relative L2 | H20 state L2 | Conserved-total error | Shock-position MAE | H20 completion |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.002646 | 0.01892 | 0.04377 | 0.01175 | 0.02789 | 16/16 |
| 0.01 | 0.003048 | 0.01518 | 0.05016 | 0.00744 | 0.03481 | 16/16 |
| 0.1 | 0.003538 | 0.00821 | 0.05242 | 0.00323 | 0.03370 | 16/16 |

Relative to the control, weight `0.01` improves boundary exchange by 19.8%
and conserved-total error by 36.7%, but worsens one-step, H20 state, and shock
errors by 15.2%, 14.6%, and 24.8%. Weight `0.1` improves the two structural
metrics by 56.6% and 72.5%, but worsens the three accuracy metrics by 33.7%,
19.8%, and 20.8%. Both runs preserve full H20 survival and decoder closure near
`1e-7`.

The auxiliary therefore reaches its intended coordinate but exposes a
conservation--accuracy Pareto tradeoff rather than a joint improvement. In 1D,
the net exchange is already identifiable from the endpoint state balance, so
this auxiliary primarily reweights existing information. Moreover, the
projected decoder distributes budget correction globally, while the dominant
remaining errors are shock-local. The supported claim is narrow: direct
boundary supervision can reduce budget error, but it is not a better joint
accuracy/stability objective for this fixed projected-residual setting. The
campaign stopped the weight search and full three-seed promotion, then proceeded
to the predeclared stride and resolution gates for the plain residual baseline.

## 2026-07-16 D027 Cold Stride-2 Gate

D027 tests whether the frozen plain-residual FNO can learn a two-saved-frame
operator directly before adding a continuation mechanism. The gate keeps the
316,419-parameter 64/24/4 FNO, 64/16/16 cases, model seed `20260708`, split
seed `20260707`, conservative coordinates, fixed physical scaling, batch size
8, learning rate `3e-4`, 50 one-step plus ten four-step recurrent epochs, and
raw inference. Only `step_stride` changes from 1 to 2. Checkpoints are selected
at physical frame 20, exactly as for the frozen stride-1 control. The selected
checkpoint is then replayed without retraining to frames 50 and 100, so the
direct model uses 10/25/50 calls and is compared with 20/50/100 calls of the
frozen stride-1 control.

Before launch, the cold row is declared viable without continuation only if:

- selected native one-step relative L2 is at most `0.005`;
- mean direct frame-2 error from the initial condition is at most 1.15 times
  the frozen stride-1 two-call composition error at the same frame;
- common-case state-error ratios at frames 20, 50, and 100 are at most 1.10;
- frame-100 completion does not decrease and mean survival drops by at most
  0.01; and
- frame-20 shock-position and conserved-total error ratios are at most 1.15.

This first cold screen uses the same epoch schedule, which gives about 1% fewer
one-step and 4% fewer recurrent sample presentations because stride-2 windows
are shorter. If the row fails and activates continuation, the final comparison
must instead match final-target sample presentations exactly, add the
total-exposure-matched cold control, compose the saved stride-1 model, transfer
weights only, and reset optimizer and scheduler state. Do not infer a
continuation-sensitive optimization effect from validation or rollout alone;
that claim additionally requires a lower final-target training-loss floor than
both cold controls.

### D027 result

The corrected run selected recurrent epoch 59 and finished in 938.8 s. Its
native test one-step relative L2 is 0.004139, below the absolute 0.005 fit
threshold but 1.61 times the stride-1 value. From the common initial condition,
the direct frame-2 error is 0.01487 versus 0.01223 for two stride-1 calls, a
ratio of 1.216; only 3/16 cases favor the direct jump. This misses the
predeclared 1.15 initial-jump gate.

The longer same-frame comparison strongly favors direct stride 2:

| Physical frame | Stride-1 calls | Stride-2 calls | Common cases | State-error ratio | Direct wins | Completion, stride 1 / 2 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | 20 | 10 | 16 | 0.750 | 15/16 | 16/16 / 16/16 |
| 50 | 50 | 25 | 15 | 0.638 | 13/15 | 15/16 / 16/16 |
| 100 | 100 | 50 | 15 | 0.514 | 11/15 | 15/16 / 16/16 |

Trajectory-bootstrap 95% intervals for the three common-case ratios are
[0.616, 0.866], [0.430, 0.827], and [0.337, 0.744]. At frame 20, direct
stride 2 also improves conserved-total error from 0.00924 to 0.00722
(ratio 0.781) and shock-position MAE from 0.03903 to 0.03557 (ratio
0.911). All 16 direct rollouts remain finite and admissible through frame 100,
although minimum pressure falls to 0.00193, so the result is not a broad
positivity guarantee. The mean truth effective-CFL maximum is about 9.89,
with maximum 13.32.

The classification is large_step_capacity_with_initial_jump_defect. A global
fixed-step FNO can learn a useful stride-2 coarse propagator and substantially
outperform repeated small-step composition over medium horizons. The formal
gate remains partial because the rare nonsmooth initial jump is worse. Follow
the preregistered continuation route: initialize the stride-2 model from the
frozen stride-1 weights, reset optimizer state, and keep the final stride-2
training presentations and all other contracts fixed. Only if that arm improves
the failed initial-jump metric should it advance to the total-exposure-matched
cold control and a continuation-sensitive optimization claim.

## 2026-07-16 D028 Stride Continuation Gate

D028 changes only initialization relative to the corrected D027 cold row. The
316,419-parameter residual FNO is initialized from the rollout-selected
stride-1 epoch-59 weights. Model family, target, architecture, coordinate and
normalization contract, case IDs, model and split seeds, stride-2 labels, batch
size, learning rate, weight decay, 50+10 final-stage schedule, H20 checkpoint
selection, and raw inference are unchanged. The loader checks those contracts,
requires a smaller source stride, verifies identical train/validation/test case
IDs and input-normalizer buffers, loads model weights only, and constructs a
fresh optimizer afterward.

The historical-checkpoint CUDA smoke passed. After one stride-2 epoch,
continuation gives training loss 3.78e-4 and H20 validation error 0.0506 with
full survival, compared with 8.77e-3 and 0.1948 for cold epoch one.

Before the full launch, continuation is declared successful at this
successive-halving stage only if:

- mean direct frame-2 error is at most 1.15 times stride-1 composition, repairing
  the only failed D027 gate;
- selected native one-step error is at most 1.05 times the cold stride-2 value;
- H20/H50/H100 common-case state error is at most 1.05 times the cold stride-2
  value at each horizon;
- H100 completion does not decrease and survival drops by at most 0.01; and
- H20 shock-position and conserved-total errors are at most 1.15 times the cold
  stride-2 values.

A pass justifies the total-exposure-matched cold stride-2 control. Only a lower
final-target training-loss floor than both the original and total-exposure cold
controls supports a continuation-sensitive optimization claim. A rollout gain
without that fit-floor gain is instead a curriculum-induced generalization or
stability result.

### D028 result

The matched run selected recurrent epoch 59 and finished in 909.7 s. Weight
provenance confirms source stride 1, source epoch 59, identical model and split
seeds, and no optimizer-state loading.

Continuation fixes the D027 initial-jump defect. Native test one-step relative
L2 improves from 0.004139 to 0.003623 (ratio 0.875). Mean direct frame-2 error
falls from 0.01487 to 0.01058, which is 0.866 times the two-call stride-1
composition error and 0.712 times cold stride 2. It wins 13/16 cases against
composition and all 16 against cold. The one-step training-loss floor falls
from 4.82e-5 to 2.80e-5 (ratio 0.581); the recurrent floor falls from 4.44e-5
to 2.49e-5 (ratio 0.562).

The solver comparison is nonuniform:

| Physical frame | State ratio, continuation / cold | Continuation wins | Conservation ratio | Shock ratio | Completion, cold / continuation |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | 1.000 | 7/16 | 0.986 | 0.515 | 16/16 / 16/16 |
| 50 | 1.095 | 7/16 | 1.079 | 1.565 | 16/16 / 16/16 |
| 100 | 0.888 | 12/16 | 0.981 | 1.217 | 16/16 / 16/16 |

The paired-bootstrap 95% state-ratio intervals are [0.924, 1.105],
[0.861, 1.537], and [0.725, 1.046]. Continuation also raises the minimum
pressure through frame 100 from 0.00193 to 0.0452, indicating a materially
safer trajectory on this split. Nevertheless, the point H50 state ratio exceeds
the preregistered 1.05 limit, and its H50 shock regression is substantial.

The classification is continuation_repairs_fit_but_not_uniform_rollout.
Warm initialization clearly accelerates and deepens final-target optimization,
repairs the rare initial discontinuity, and moves the model to a safer-pressure
trajectory. It does not yield a uniformly better fixed-step solver. Because the
successive-halving solver gate fails, do not spend the conditional
total-exposure cold run and do not claim continuation-sensitive optimization;
the missing total-exposure control remains an explicit claim boundary. Preserve
cold stride 2 as the large-step accuracy baseline and continuation as a
fit/stability tradeoff ablation.

## 2026-07-16 D029 Frozen Cross-Resolution Gate

D029 asks whether the two frozen 316,419-parameter FNOs trained only at 256
cells define useful zero-shot operators at 128 and 512 cells. It changes no
weights, normalizers, timestep, physical cases, saved times, or test IDs. The
input, loss, and recurrent coordinates remain conservative with fixed physical
scaling. Inference is native-grid and interpolation-free. The stride-1 model
uses 1/20/50/100 calls to reach frames 1/20/50/100; cold stride 2 uses
1/10/25/50 calls to reach frames 2/20/50/100.

This is a discrete-operator transfer test, not an assertion that the predicted
cell residual is itself resolution invariant. The FNO keeps 24 learned Fourier
modes at every mesh, so the 512-cell evaluation tests interpolation and operator
transfer without granting additional learned shock bandwidth. Native reference
differences are reported after conservative finite-volume restriction from
512 to 256 and 256 to 128; they contextualize, but are not subtracted from,
model error.

Before looking at off-grid model results, the gate requires:

- exact identity of left/right states, domains, discontinuity positions, and
  saved times across all 512 cases;
- uniform-grid endpoint reconstruction within `2e-6` and enough grid modes for
  the frozen 24-mode FNO; and
- nx256 replay of the frozen one-step and rollout metrics to numerical print
  precision.

A checkpoint has usable zero-shot resolution transfer only if, on both nx128
and nx512 relative to its own nx256 replay:

- native one-step relative L2 and H20/H50/H100 final state error are each at
  most `1.5x`;
- completion loses at most one of 16 cases at each horizon and mean survival
  falls by at most `0.02`; and
- H20 shock-position and conserved-total errors are each at most `1.5x`.

The D027 larger-step advantage transfers only if cold stride 2 also has no
larger H20/H50/H100 final state error than stride-1 composition on each new
mesh and does not reduce completion. Passing only the lower resolution is a
coarse-grid interpolation result, not bidirectional mesh robustness. Failure at
512 with stable completion will be classified separately from recurrent
instability because fixed spectral bandwidth may impose an accuracy floor near
shocks.

### D029 result

The implementation contract passes. The current generator configuration and
seed reproduce all 512 serialized physical case parameters exactly. All three
datasets have identical saved times and reconstruct their uniform domains within
`6e-8`. The nx256 replay reproduces every D027 one-step, rollout, completion,
shock, and conservation value to printed precision. This rules out checkpoint
restoration, FFT shape handling, split drift, or metric drift as explanations.

Neither frozen checkpoint passes the preregistered native-grid transfer gate:

| Checkpoint | Grid | One-step ratio | H20 state ratio | H50 state ratio | H100 state ratio | H100 completion, off / nx256 | H20 shock ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stride 1 | 128 | 5.692 | 1.345 | 0.984 | 0.961 | 15/15 | 0.651 |
| stride 1 | 512 | 8.340 | 1.306 | 1.077 | 1.017 | 15/15 | 2.098 |
| cold stride 2 | 128 | 5.527 | 1.735 | 1.168 | 0.910 | 15/16 | 0.656 |
| cold stride 2 | 512 | 5.619 | 1.419 | 1.127 | 1.057 | 15/16 | 2.994 |

State ratios use cases that complete on both the off-grid and nx256 runs;
completion is evaluated separately. Stride 1 remains remarkably stable: the
same case 21 loses pressure positivity at call 49--50 on every resolution.
Cold stride 2 exposes a resolution-sensitive margin: case 190 completes all 50
calls at nx256 with minimum pressure 0.00193, but becomes nonpositive at calls
19 and 18 on nx128 and nx512. Thus the off-grid completion loss is small but
mechanistically real, not a mixed-prefix accounting artifact.

High-resolution shock degradation is also real rather than only primary-front
detector switching. At H20, stride-1 top-two position error rises from 0.0129
to 0.0274 and strength error from 0.344 to 0.732; cold stride 2 rises from
0.0172 to 0.0353 and from 0.353 to 0.827. Conserved-total error stays near its
nx256 value. The fixed 24-mode network therefore transfers the bulk field and
global budget much better than fine-grid shock geometry and strength.

The important positive result is that the D027 larger-step advantage transfers
on both new meshes:

| Grid | H20 stride-2 / stride-1 | H50 ratio | H100 ratio | H100 completion, stride 2 / 1 |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 0.968 | 0.762 | 0.474 | 15/15 |
| 256 | 0.750 | 0.638 | 0.514 | 16/15 |
| 512 | 0.815 | 0.643 | 0.523 | 15/15 |

The H50 and H100 paired-bootstrap intervals exclude one on both off-grid
meshes. The H20 interval narrowly includes one at nx128 but excludes one at
nx512. This supports a resolution-robust reduction in recurrent composition
error from the learned larger step, even though neither frozen network matches
the native off-grid one-step map well.

The label audit explains much of the negative transfer result. Along paired
native trajectories, the conservative one-frame increment mismatch between
nx128 and restricted nx256 is 2.70% of the next-state norm and 57.9% of the
coarse increment norm. Between nx256 and restricted nx512 it is 2.91% and
47.5%. For stride 2 the corresponding pairs are 3.85%/44.8% and 3.03%/28.3%.
The initial condition is also center sampled rather than an exact cut-cell
average; its restricted state mismatch reaches 5.66% on average for 256 to 128
and 2.09% for 512 to 256. The frozen FNO is therefore asked to reproduce a
different grid-dependent numerical flow map. Its cell features contain state
and position but no explicit cell width or timestep, and the fixed 24-mode
checkpoint does not exploit the added 512-cell shock resolution. The bandwidth
explanation is consistent with the result but is not isolated by D029.

The classification is
`stable_large_step_advantage_without_native_resolution_transfer`. Do not call
the current residual FNO mesh invariant, but do retain cold stride 2 as a strong
large-step baseline. Before stride 4 or another constraint sweep, construct a
restriction-consistent state/residual contract and test one shared
multi-resolution FNO with matched sample presentations. First alternate
uniform-resolution batches without adding architecture features; only then
ablate explicit cell-width/timestep channels or higher spectral bandwidth.

## 2026-07-16 D030 Restriction-Consistent Shared-Resolution Gate

D030 asks whether the D029 failure comes primarily from changing the numerical
flow map with the grid. Generate one 512-cell WENO-HLLC-ADER reference with
exact conservative cell averages at the initial discontinuity, then obtain the
256- and 128-cell trajectories only by finite-volume restriction. The primary
truth is this common restriction-consistent family. Separately generated native
coarse-solver trajectories remain a diagnostic and are never substituted for
the primary ground truth.

The shared row keeps the D027 stride-1 contract: a 316,419-parameter 64/24/4
FNO, conservative input/loss/recurrent coordinates, fixed physical scaling,
model seed `20260708`, split seed `20260707`, 64/16/16 cases, batch size 8,
learning rate `3e-4`, 50 one-step epochs, ten four-step recurrent epochs, H20
rollout selection, and raw inference. Each shared epoch has the same total
one-step or recurrent sample presentations as one single-resolution row; those
presentations are divided as evenly as possible among homogeneous 128-, 256-,
and 512-cell batches. Equal-presentation single-resolution models are fit as
same-grid oracle controls. No cell-width/timestep feature, additional Fourier
mode, limiter, positivity transform, viscosity, or smoothing is allowed.

Before full training:

- physical case parameters and saved times must match exactly across all three
  files;
- conservative restriction of the 512-cell states must reproduce both coarse
  files with global conservative relative L2 at most `1e-6` on the selected
  cases;
- restriction must commute with stride-1 and stride-2 increments in the same
  global norm to the same tolerance; per-case/time maxima remain diagnostics;
  and
- a 2-epoch 8/4/4 sanity row must remain finite, write a restorable checkpoint,
  and lower its training loss from the zero-update initialization.

The shared representation gate passes only if, on restriction-consistent test
truth at every resolution:

- one-step and H20/H50/H100 final state error are each at most `1.5x` the
  corresponding same-grid oracle;
- completion loses at most one of 16 cases and mean survival falls by at most
  `0.02` relative to that oracle; and
- H20 shock-position and conserved-total errors are each at most `1.5x` the
  oracle.

Improvement over the frozen native-nx256 checkpoint is a separate usefulness
gate: on restriction-consistent nx128 and nx512 truth, shared one-step error
must fall by at least 25%, H20 state error must not increase, and completion
must not decrease. Native-coarse evaluation is reported without imposing this
target-fit gate. Passing the primary gate but failing native-coarse equivalence
will be classified as `shared_restriction_operator_without_native_solver_equivalence`.
If same-grid oracles fit but the shared row fails, route next to explicit cell
width or higher bandwidth. If the oracles themselves fail, diagnose target fit
before changing the shared architecture.

Preflight amendment before any model result: primitive float32 serialization on
the 12-case synthetic smoke gives state global relative errors below `2.4e-8`
and stride-1 update global relative errors below `8.0e-7`, but a per-case/time
ratio reaches `3.29e-6` when its update denominator is small. The contract
therefore uses the global conservative relative norm at the original `1e-6`
tolerance and records the unstable local maximum separately. This changes the
norm definition, not the tolerance or any learned-model gate.

Execution amendment chosen before inspecting any learned metric: PowerShell
buffered the remote trainer's stdout, so stopped timing attempts were later
found to have written partial histories even though no epoch line had reached
the monitor. None of those metric values was read while choosing the changes;
the decisions used only wall time, GPU utilization, isolated phase benchmarks,
and equivalence tests. Training remains batch 8 with the same sample
presentations. Teacher-forced evaluation uses batch 128,
and the raw H20 checkpoint-selection rollout batches the same 16 cases within
each resolution. A focused equivalence test matches the original casewise
completion, survival, and rollout-error summaries. Final H20/H50/H100 research
evaluation remains casewise with the full shock, conservation, and positivity
diagnostics; this amendment changes execution overhead, not selection data or
the model/training contract. The same execution audit found repeated
GPU-to-CPU scalar reads inside every generic training batch. D030 therefore
uses the trainer's opt-in deferred-metric path, reducing detached loss and
relative-error tensors once per epoch and replacing its remaining host-side
finite-loss branch with a CUDA-stream asynchronous assertion. One-step and
recurrent equivalence tests match the original optimizer updates exactly and
the reported metrics within floating-point summation tolerance; a focused
nonfinite test still aborts, and the default trainer path is unchanged.
An exact-shape AutoDL benchmark rejected `torch.compile`: Inductor does not
generate complex FFT kernels here and was 1.8 times slower than eager
forward/backward. Eager optimization steps are already about 5.6 ms; the
remaining overhead came from the generic validation diagnostics. D030
checkpoint selection therefore also defers its state loss and relative-error
reduction to one synchronization per resolution. A focused test matches the
generic evaluator, while the final paper-facing evaluator remains unchanged.

### Result

The full AutoDL row completed under the ignored artifact root
`artifacts/time_dependent_no/d030_restriction_consistent_f64_20260716/`.
The fine file contains 512 trajectories, 101 saved frames, and 512 cells. The
shared and oracle rows all use the same 316,419-parameter 64/24/4 FNO, 6,400
one-step sample presentations per epoch, 6,208 recurrent-window presentations
per recurrent epoch, and the frozen 64/16/16 split. The shared row selected
epoch 58; the nx128/nx256/nx512 oracles selected epochs 56/60/60.

The exact-data gate is substantially tighter than required. Maximum analytic
initialization error is `2.93e-10`. State restriction and stride-1/stride-2
increment commutation are at machine precision; the largest reported global
relative error is `1.63e-15`, versus the preregistered `1e-6` tolerance.

On the common restriction-consistent evaluator, the shared row has one-step
relative L2 `0.004331/0.004040/0.004664` at nx128/nx256/nx512, versus
`0.003342/0.003524/0.003619` for the corresponding single-grid oracles. The
shared/oracle final-state results are:

| Grid | Horizon | Shared error | Oracle error | Ratio | Shared complete | Oracle complete |
|---|---:|---:|---:|---:|---:|---:|
| nx128 | 20 | 0.050765 | 0.054775 | 0.927 | 16/16 | 15/16 |
| nx128 | 50 | 0.102588 | 0.106591 | 0.962 | 14/16 | 13/16 |
| nx128 | 100 | 0.262138 | 0.244347 | 1.073 | 13/16 | 13/16 |
| nx256 | 20 | 0.048955 | 0.051311 | 0.954 | 16/16 | 15/16 |
| nx256 | 50 | 0.096231 | 0.083663 | 1.150 | 14/16 | 14/16 |
| nx256 | 100 | 0.254927 | 0.179770 | 1.418 | 13/16 | 13/16 |
| nx512 | 20 | 0.049461 | 0.053850 | 0.918 | 16/16 | 15/16 |
| nx512 | 50 | 0.096530 | 0.095497 | 1.011 | 14/16 | 14/16 |
| nx512 | 100 | 0.255026 | 0.182126 | 1.400 | 13/16 | 13/16 |

The worst state and one-step ratios are `1.418` and `1.296`; completion is
never lower than the oracle, the worst mean-survival deficit is `0.0075`, and
the worst H20 shock-position and conserved-total ratios are `1.123` and
`1.138`. The primary representation gate therefore passes. Against the frozen
native-nx256 checkpoint on restriction-consistent nx128/nx512 truth, shared
one-step error falls by 68.1%/78.3%, H20 error falls by 8.9%/13.3%, and
completion remains 16/16, so the separate usefulness gate also passes.

The native-coarse diagnostic does not pass equivalence. Moving the shared row
from restriction-consistent to independently evolved native truth multiplies
one-step error by `5.27x` at nx128 and `3.65x` at nx256, while changing it by
only `1.002x` at nx512. Native paired-grid state gaps are already several
percent and include frame-zero discretization differences. This supports target
inconsistency as a major cause of D029, rather than an FFT or shared-capacity
failure, but it does not establish one model for multiple numerical flow maps.

Raw recurrent stability remains unresolved. The same held-out cases 190, 238,
and 418 terminate on negative pressure across all three shared-grid rollouts at
approximately calls 21--25, 35--39, and 60--66. D030 therefore supports
resolution sharing for one identifiable restricted flow map, not positivity-free
H100 reliability. Do not add cell width or more modes merely to repair D029.
Route the next stability experiment to a separately controlled post-fit
tail-risk/admissibility stage on the strong residual baseline; do not repeat the
rejected blind-viscosity or interface-latent rows unchanged.

## 2026-07-18 D013 2D PCNO Ripple And D015 Exposure Result

The active 2D baseline is the 19,155,720-parameter PCNO conservative-variable
residual model, not the historical positive-primitive checkpoint. It uses the
closed 270/30 trajectory split, fixed normalization, `kmax=8`, five width-128
layers, a `6 x 2` Fourier domain, reconstructed vertex-lumped diagnostic proxy
weights, legal `model_all_nodes` boundaries, raw recurrence, and rollout-based
selection. Its selected clean continuation has 5/5 admissible 20-call
completion, mean rollout relative L2 `0.024285`, and one-step validation
relative L2 `0.005402`. These proxy weights are not validated physical control
volumes, so this result makes no physical conservation or flux claim.

Frozen D013 ran on five selected validation trajectories under
`artifacts/time_dependent_no/pcno_euler2d_d013_clean_20260718a/`. All 20/20
calls completed without clipping, floors, smoothing, a limiter, or reference
boundary replacement. The predeclared screen classifies the dominant measured
mechanism as recurrent amplification: median late rollout/teacher smooth-region
high-band energy is `13.26x`, first-to-late rollout high-band growth is
`45.28x`, and error-direction perturbation gain is `1.094`. The front union
contains `92.65%` of scaled error energy, while local translation removes zero
median pressure error; this is shock-local strength/thickness/ripple error, not
a pure phase shift. The learned spectral branch carries large amplitude but has
only `0.0074x` the roughness of the rougher local branch. Exact branch replay is
within `7.63e-6`, and disabling any final branch worsens error on the traced
cases. These observations reject “ringing is born in the spectral branch” as
the primary explanation for this checkpoint.

The basis is nevertheless vulnerable to Gibbs-like approximation and discrete
coefficient leakage. Under the reconstructed proxy, the original-basis Gram
matrix has median rank fraction `0.744`, condition number `8.46e9`, and
off-diagonal Frobenius ratio `1.816`. The no-training five-case counterfactual
under `artifacts/time_dependent_no/pcno_euler2d_basis_contract_clean_20260718a/`
uses the occupied `3 x 1` coordinate span. It restores full numerical rank,
lowers median condition number to `3.37e4`, off-diagonal ratio to `0.455`, and
cross-band leakage from `0.445` to `0.247`. It also lowers the PCNO uniform-formula
pressure reconstruction RMSE from `1.158` to `0.797`. However, the best
mass-orthogonal pressure projection slightly worsens from `0.466` to `0.497`,
with substantial shock overshoot remaining. Thus the doubled domain contributes
coefficient leakage, but changing the domain alone does not remove the
finite-basis shock limit and is not a demonstrated rollout fix.

Line 2's D014 interface was consumed rather than recomputed. It records an
exact architectural current-state support radius of 13 and zero of 800
endpoint-sampled directed characteristic cones outside that support. This does
not bound unresolved DG substeps. PCNO's full architecture is globally
dependent through its spectral branches, so its relevant tests are basis,
quadrature, approximation, and nonlocal-state-use tests rather than a finite-hop
obstruction.

D015 therefore tested exactly one routed intervention: detached depth-one raw
generated-state exposure with a 0.5 clean one-step anchor, initialized from the
selected clean checkpoint. The five-epoch, 1,024-presentation pilot under
`artifacts/time_dependent_no/pcno_euler2d_generated_exposure_w05_pilot_20260718/`
retains 5/5 admissible completion. Its selected first epoch gives rollout error
`0.023736` and validation error `0.005387`, only a `2.26%` rollout improvement;
three cases improve and two regress. Later rollout errors range from `0.028999`
to `0.047350`. This fails the predeclared `10%` promotion gate. Do not run the
full matched confirmation, combine exposure with noise, or begin an exposure
weight sweep. Retain the clean checkpoint as the serious baseline and keep the
mechanism classification separate from this intervention's negative result.

## 2026-07-19 D013 Branch-Control CPU Preflight

No model was trained and no rollout was launched in this preflight. The PCNO
trace now supports frozen gains in `[0, 1]` for the spectral, pointwise, and
differential branches at each layer. Gain one follows the exact existing path,
so it provides a replay control; smaller gains are bounded diagnostic
counterfactuals, not a new trained architecture or evidence of causality.

The reusable Euler utility now provides a differentiable graph-neighbor
high-pass diagnostic, a reference-only pressure-jump shock exclusion mask, and
a directional high-pass response gain with a one-sided cap penalty. This is a
training preflight only: it neither filters the recurrent state nor changes the
current trainer. On the bump artifact, all spatial weights remain diagnostic
proxies and cannot support physical conservation or finite-volume flux claims.

The historical paired-response audit had a frozen selector before its results
were examined. At least two repeated trajectory/call rows had to agree, and the same
branch must have both the largest finite-response RMS gain and the largest
edge-to-node roughness in at least 60% of traced layers in every accepted row.
It then routes exactly one of spectral-contract repair, local pointwise
control, or local differential control; disagreement routes to
`composite_or_unresolved`, and fewer than two rows route to
`insufficient_repeated_rows`. This selector was designed to route one subsequent
experiment only; it
is not an infinitesimal Jacobian estimate or causal branch attribution.

The focused CPU suite passes all 15 PCNO residual and ripple-diagnostic tests,
including exact gain-one replay, bounded-gain validation, reference-shock
exclusion, differentiability, and repeated-row selector behavior. A learned
local correction was not authorized by this preflight. The historical gate
required the audit to select a local route and the predeclared oracle
constrained-decomposition test to show that a local, bounded, admissible
correction could materially reduce the residual error.

## 2026-07-19 D013 Paired Branch-Response Audit

The frozen paired-response audit is complete under ignored artifact root
`artifacts/time_dependent_no/pcno_euler2d_paired_response_clean_20260719a/`.
It uses the same selected clean continuation checkpoint (SHA256 beginning
`2bb5ee3c`), full-manifest digest beginning `5d5373fd`, five historical
validation trajectories `16/60/128/141/235`, and Line-2 interface digest
beginning `d3f0eb8d`. A strict 19,155,720-parameter load and all 15 focused CPU
tests passed before the RTX 5060 Ti evaluation. All five trajectories complete
20/20 raw calls without a floor, clip, limiter, smoothing, or reference
boundary replacement.

The original mechanism screen is stable under this replay: late
rollout/teacher smooth high-band energy is `13.17x`, first-to-late rollout
growth is `45.71x`, perturbation gain is `1.0948`, and `92.65%` of scaled error
energy lies in the front union. Small numerical differences from the first
D013 bundle do not change its `recurrent_amplification` classification.

The new predeclared selector has two complete call-10 rows, one each for
trajectories 16 and 60. In both rows the pointwise branch has the largest
finite-response RMS gain in 3/4 layers and the largest edge-to-node response
roughness in 4/4 layers. It therefore passes the required 60% layer threshold
and routes `local_pointwise_control`. Layer zero remains informative rather
than contradictory: the differential branch has the largest RMS response
there, while pointwise remains the roughest response. This is a finite response
along 25% of the observed rollout-error direction, not an infinitesimal
Jacobian, causal branch attribution, or evidence that attenuating the pointwise
branch will improve rollout.

The route does not authorize a learned correction on the bump bundle. The
required oracle decomposition must enforce conservation, locality, bounded
norm, and admissibility, while the current reconstructed vertex weights are
only diagnostic proxies and the artifact still lacks validated control-volume
volumes and oriented physical faces. A bump-only proxy correction cannot pass
that conservation gate. The next eligible Line-3 step at that point was to
freeze the narrow dynamic perturbation split and serious global baseline, then
run the predeclared D037 oracle decomposition. The contract forbade
pointwise-gain, smoothing, and learned-correction sweeps there.

## 2026-07-21 D044 Dynamic FV-PCNO And Oracle Closeout

The serious full-resolution run is complete on the frozen 135-case
shock--vortex family. Its 19,155,720-parameter residual PCNO predicts
conservative-variable state residuals, uses training-only primitive noise, and
selects raw H60 validation rather than one-step loss. Epoch 44 is selected. On
all 24 validation cases, mean physical-volume state error is `0.00028009` at
call 1 and `0.00834190` at call 60, with 24/24 finite and admissible completion.
No future-reference boundary, floor, clip, smoother, limiter, or decode/reencode
projection is used. H60 persistence, nearest-training-trajectory, and four-
neighbor train-only parameter/time interpolation errors are
`0.0732949/0.0231132/0.0231196`; PCNO wins all 24 paired cases against each.

The model is not conservative by construction. It predicts no face exchange,
and the H60 normalized physical-total mismatch relative to reference boundary
exchange is `0.0513185`, worse than the train-manifold controls. The separate
physical evaluator stores validated volumes, oriented faces, normals, measures,
mesh/graph mappings, and reference impulses, but those fields do not convert a
state-residual model into a flux model.

The frozen six-case D013 bundle completes every H60 trajectory and consumes the
Line-2 D014 report. Smooth-region high-pass RMS appears on the first call and
grows with interaction time under both teacher forcing and rollout. At call 60,
median rollout state error is `0.0087148` versus teacher-forced `0.0009734`, but
smooth high-pass RMS is only `0.0053520` versus `0.0048877`; the deep graph-band
energy ratio is `1.0458`. Thus recurrence strongly grows total state error, but
the measured high-frequency birth is mostly shared with later teacher-forced
calls rather than a uniquely autoregressive instability.

The actual PCNO Fourier Gram condition number is `1.000003`, its off-diagonal
Frobenius ratio is `4.80e-7`, and mass orthogonalization changes reconstruction
RMSE by essentially one. The spectral response is the smoothest branch.
Pointwise dominates paired RMS response in all four layers, while differential
is roughest in three; no branch wins both criteria. Disabling any final-layer
branch at call 60 worsens error in both deep trajectories. Strong paired error-
response cancellation is falsified. All aliasing, quadrature/geometry, front-
phase, and recurrent-amplification threshold screens are false, so the mechanism
remains `unresolved`; these observations do not prove a Gibbs or local-branch
cause.

The truth-informed constrained oracle then evaluates exactly the predeclared
six trajectories at calls 10/30/60. All 18 rows remain raw-admissible, conserve
the interior correction to at most `4.75e-19`, use no more than 20% of physical
faces and a 10% global-update norm, and pass every anti-smearing check. Median
state and smooth-high-pass reductions are only `0.05135` and `0.04164`, below
the `0.15/0.20` gates. The learned sparse local correction branch is rejected;
zero-correction optima are retained rather than hidden.

The authoritative Line-3/Line-4 report is
`line3_to_line4_handoff_s20260718_20260721b.json`, SHA-256 beginning
`4b56baef`. It reconstructs the full source-artifact mapping from the family
audit and matches it to the shard contract. It records
`line4_training_truth_authorized=true` only for this frozen family,
`line4_front_candidate_available=false`, and
`line4_transition_training_authorized=false`. The 27-case strength-OOD test
split remains sealed.

## 2026-07-21 D045 Shared-Face Tiny-Fit Closeout

D045 implemented only the first predeclared structured-target row. Its reusable
code was historically stored at
`utility/time_dependent_no/pcno_face_impulse.py`; that path was retired from the
active tree after closeout and is recoverable at pre-cleanup commit `729091b`.
The implementation validated the frozen family manifest, bound a training-case
source artifact by SHA-256, reconstructed the physical-face PCNO graph, and
byte-compared its graph and mesh-mapping arrays. The loader read no cumulative
reference face impulses. The model predicts one
owner-oriented interior impulse with an antisymmetric shared head, uses a
current-state-only boundary head, restricts reflecting-wall exchange to y
momentum, and decodes through physical cell volumes. Zero output initialization
is exact persistence. Training and rollout remain raw, conservative-coordinate,
and state-loss-only, without a future boundary, floor, clip, limiter, or
smoother.

The first smoke artifact,
`pcno_face_impulse_smoke_s20260718_20260721a`, is retained as failed provenance:
it exposed a bfloat16 indexed-assignment dtype bug before an optimizer step. A
focused regression fixes the accumulation dtype. The corrected artifact,
`pcno_face_impulse_smoke_s20260718_20260721b`, passes with 19,210,028 parameters,
peak allocated memory 1,428,881,920 bytes, admissible validation, and no
reference-impulse supervision.

The two registered four-pair attempts are:

| Artifact | Optimizer updates | Best mean state relative error | Best loss ratio | Joint gate |
|---|---:|---:|---:|---|
| `pcno_face_impulse_tinyfit_s20260718_20260721a` | 800 | `0.00166965` | `0.0730244` | fail |
| `pcno_face_impulse_tinyfit_s20260718_20260721b` | 3,200 | `0.000955675` | `0.0254377` | fail |

The second row changes only presentations of the same four immutable pairs. It
passes the absolute relative-error threshold `0.001` but misses the required
loss ratio `0.01`. It uses 1,584,159,744 peak allocated bytes and finishes in
185.9 seconds, so downsampling is neither a memory requirement nor acceptable
for a serious baseline. A read-only last-checkpoint replay gives mean relative
errors `0.00123513` for full bfloat16, `0.00121919` for a bfloat16 backbone with
float32 face heads, and `0.00127140` for full float32. Precision is not the
leading measured limitation.

Interior cancellation and the predicted-boundary balance identity hold to
numerical precision, but the learned boundary exchange is not thereby the
physical reference exchange. This is an optimization/identifiability failure
of the direct state-loss-only shared-face parameterization at its entry gate,
not evidence against every face, flux, or interface target. The serious row is
stopped; the strength-OOD test split remains sealed. Any later divergence-
active/minimum-norm row requires separate authorization and should start with a
zero-training canonical-projection preflight. The authoritative D044 Line-4
handoff and all three coordination flags remain unchanged.

## 2026-07-21 D046 Canonical-Target Preflight Registration

D046 is registered before any reference-impulse row is inspected. The exact
24-row train/validation cohort and tolerances are in
`RESEARCH_DIRECTION_DECISION.md`. The diagnostic will retain the accepted
reference boundary exchange, replace the interior reference face field by its
minimum-`W_f^{-1}`-norm divergence-active representative, and independently
reconstruct that representative from the reference state increment plus the
same boundary exchange. It will also evaluate closure against the actual
float32 state shards used by PCNO training.

The test split, a learned model, loss weights, and the full reference cycle
field are outside this preflight. Passing every gate permits only one immutable
four-pair supervised tiny fit; failing any gate stops the row without a
tolerance retry. The Line-4 handoff remains unchanged.

Attempt `fv_divergence_active_target_preflight_20260721b` solved all 24 rows in
110.3 seconds and failed only the `1e-8` canonical/reference closure gate, at a
maximum `1.82559e-8`. Inspection found that the implementation had reversed
the two registered constructions: the projection of the full reference field
was called canonical, while the state-constrained minimum-norm solution was
treated as the independent check. The mathematical contract defines the latter
as canonical. Attempt b is preserved as failed provenance. One corrected rerun
may swap those roles only; all inputs, solver settings, thresholds, and stop
rules remain fixed.

The corrected attempt `fv_divergence_active_target_preflight_20260721c` solved
all 24 rows in 111.4 seconds and again failed exactly one frozen gate. The
state-plus-boundary canonical solve has reference-state closure
`1.67298e-7--4.94497e-7`, with maximum `4.94497e-7` against the required
`1e-8`. Train and position-OOD validation ranges overlap, so this is not a
split-specific failure. Reference-impulse closure is at most `1.23427e-12`;
canonical closure against the published float32 shard increments is at most
`9.36905e-6`; the independently projected and state-constructed face fields
disagree by at most `1.61785e-8`; compatibility and cycle-divergence residuals
are at most `2.94020e-12` and `5.18524e-10`; the canonical/full weighted-norm
ratio is at most `0.999440`; forbidden reflecting-wall exchange is at most
`3.10460e-20`; and all sparse solves have accepted stop codes.

The accepted reference cycle component contains only `0.1129--0.6691%` of the
interior `W_f^{-1}` energy (median `0.2646%`), while the canonical/full norm
ratio is `0.996677--0.999440`. This supports practical identifiability of the
divergence-active field and validates the state, geometry, orientation,
boundary, and cumulative-impulse consistency checks used here. It does not
override the preregistered proof-level closure requirement. D046 therefore
stops without a supervised tiny fit, tolerance relaxation, another LSMR retry,
or test access. A reusable direct-factorization projector would be a distinct,
separately preregistered numerical-method question, not an automatic retry or
training authorization. The Line-4 handoff flags remain unchanged.

## 2026-07-22 D047 Direct-Projector Registration

D047 is registered before implementing or executing a direct solve on the
frozen data. It reuses D046's exact 24 rows, source digests, boundary impulses,
and numerical gates. The only changed object is the canonical construction:
after explicitly removing and reporting each connected component's arithmetic-
mean incompatibility, a once-factored anchored float64 weighted graph
Laplacian produces `W_i B_i^T phi`. The lowest global cell index fixes the
potential gauge; it cannot affect the face field. No diagonal jitter,
regularization, refinement, alternative anchor, or tolerance/solver sweep is
allowed.

In addition to every D046 gate, the compatibility projection and reduced-
system residual must each be at most `1e-10` relative, all factors and outputs
must be finite, and the same factor objects must serve every variable and row.
Only synthetic CPU tests may run before the single frozen-data execution. A
pass does not itself authorize training; it only permits a separately frozen
four-pair tiny-fit contract. A numerical failure ends this target branch. The
test split and all Line-4 coordination flags remain untouched.

The completed artifact `fv_direct_canonical_target_preflight_20260722a`
passes every registered check. Its 24 canonical closures span
`1.09848e-9--3.40147e-9`, versus `1.67298e-7--4.94497e-7` for D046, and every
matched improvement is `144.97--157.64x`. Maximum direct compatibility-
projection and reduced-system residuals are `1.85905e-14` and `1.13351e-12`;
the independently projected face field differs by at most `8.42560e-10`.
Factorization count, reuse, finiteness, provenance, split, and test-nonaccess
checks all pass. D047 therefore confirms the specific numerical diagnosis but
makes no learned or conservation claim.

D048 was frozen as the only promoted action. It repeats D045's exact four
training pairs and 3,200-update full-resolution architecture/optimizer budget,
but directly supervises D047's canonical interior field and accepted boundary
exchange through equal-weight interior/boundary relative `W_f^{-1}` loss. The
decoded state is an outcome, not a loss term. A common epoch must reach native
loss ratio `0.01`, full/interior/boundary relative error `0.10`, decoded-state
relative L2 `0.001`, and 4/4 admissibility. There is no retry, state-loss
mixture, downsampling, test access, or automatic serious-run promotion.

## 2026-07-22 D048 Canonical-Face Tiny-Fit Closeout

The one authorized artifact,
`pcno_canonical_face_tinyfit_s20260718_20260722a`, uses the exact four D045
training pairs, 19,210,028 parameters, and 3,200 updates. The passed D047
summary, label set, trainer, face decoder, and projector are digest-bound. The
worst float64 canonical/reference closure is `2.77918e-9`, the one-time
float32-label closure is `3.34619e-5`, and the test access list is empty.

The face objective fits: 15 epochs jointly pass loss ratio and full/interior/
boundary face error, and the minimum native values are `0.00707672` loss ratio,
`0.0950046/0.0953408` full/interior error, and `0.0201020` boundary error. But
none of the 50 saved epochs has any admissible tiny-pair prediction. The best
decoded-state error is `0.474958`; the final error is `0.500998`, versus
`0.00628209` for zero-output persistence.

The final-checkpoint descriptive replay distinguishes target optimization from
update conditioning. Face error is only `0.09772--0.09877`, while decoded cell-
increment error is `98.91--155.90` times the true increment. The error field has
weighted divergence gain about 2, but the very smooth canonical target has gain
only `0.001264--0.001999`, so relative amplification is `1001--1581x`.
Density, internal energy, and pressure become strongly negative. This is direct
evidence for a face-loss/discrete-divergence mismatch on these four training
pairs; calling it Fourier Gibbs or attributing it to one PCNO branch would go
beyond the saved evidence.

D048 therefore fails and stops this exact canonical face-value objective. Do
not add a state-loss rescue, retune the objective, launch a serious row, access
the 27 strength-OOD test cases, or make a conservation claim. D044 remains the
strong 2D residual baseline. A future face-form row, if separately authorized,
must first show a well-conditioned `H(div)`-aligned error contract or a
deterministic face lifting of an accurate cell residual. All Line-4 flags remain
unchanged.

## 2026-07-22 D049 Divergence-Conditioning Closeout

Artifact `fv_divergence_conditioning_d049_20260722a` contains three canonical
horizons, nine physical band modes, 198 signed perturbation rows, and nine
geometry-only resolution rows. All six preregistered checks pass. On the
validated 250x100 mesh, median low/mid/high normalized frequency and decoded
gain are `0.01523/0.11917/0.63531` and
`0.34903/0.97641/2.25444`; seed standard deviations of gain are below
`0.0032`. The minimum high/low ratio is `6.3866`, and the high band is at least
`369.06x` the largest physical canonical-target component gain. Canonical
closure is at most `6.61e-9`; the reference cycle gain is at most `1.24e-10`
and its full-field energy fraction is `0.0837--0.2074%`.

At fixed macro `Delta t=0.01`, analytic Cartesian flux-decoder gain is
`3.20137/6.40303/12.80620` at 250x100/500x200/1000x400, while impulse gain
stays near `2.8284`. The latter two meshes carry no dynamic states or face
truth. The result confirms norm-dependent divergence conditioning and brackets
D048's documented error gain between the mid/high bands. It does not prove
Gibbs causation, find the learned source, or authorize a basis/face/loss sweep.

## 2026-07-22 D050 Residual-To-Face Registration

D050 consumes only the six frozen D013 validation trajectories at calls
`1/10/30/60`, the passed D049 summary, and the digest-bound D044 evaluation and
handoff. It executes no model and opens no strength-OOD test trajectory. The
legal control distributes the PCNO-predicted total cell integral over x
boundaries by a deterministic minimum-`W_f^-1` allocation and applies the D047
direct interior lift. It uses no future reference but is an algebraic boundary
closure without a physical flux claim. The oracle control substitutes accepted
cumulative reference boundary impulses solely for headroom attribution.

Structural gates are `1e-8` for legal state reconstruction and both lift
closures, `1e-14` for forbidden legal wall exchange, plus raw admissibility at
every endpoint. Headroom gates are median H60 state/budget reductions
`0.15/0.80`, maximum correction/update norm `0.10`, and maximum median H60
front-chamfer, shock-strength, shock-thickness, or smooth-error ratio `1.05`.
Passing both sets permits only design of a legal current-state boundary-budget
preflight; it does not authorize a learned boundary, face model, or serious
run. All Line-4 flags remain frozen.

## 2026-07-22 D050 Residual-To-Face Closeout

The single artifact `pcno_residual_face_lift_d050_20260722a` writes all 24
paired and 72 variant rows with source, checkpoint, trajectory, D049, and
handoff digests. No model executes and no test trajectory opens. The legal
algebraic row reconstructs D044 to at most `5.37e-12`, its stored target
residual is at most `7.58e-11`, forbidden wall exchange is exactly zero, and
every raw/legal/oracle endpoint is admissible.

D050 formally fails because the implementation populated the frozen
`oracle_lift_closure` metric with the intentional residual between the decoded
compatibility-projected field and the original incompatible D044 cell total.
It reaches `0.07610` against `1e-8`; that is not the reduced-system solve
residual the gate intended. The implementation now stores reduced-system
closure and target-projection magnitude separately, but the frozen artifact is
neither rerun nor relabeled.

The descriptive oracle headroom independently misses promotion. Median H60
budget defect falls from `0.05206` to `4.04e-12`, median front/shock/smooth
ratios remain at most one, and correction/update norm is at most `0.05467`.
Median H60 state error falls only from `0.008673` to `0.008055`, a `7.41%`
reduction versus `15%`; individual reductions span `5.09--9.54%`. Stop the
residual-to-face/boundary route without a retry, new training row, test access,
or conservation claim. D044 remains the flagship residual baseline, D049
remains valid as conditioning evidence, and all Line-4 flags remain frozen.

## 2026-07-22 D051 Coarse-CFD Error--Cost Registration

D051 closes the one explicit D044 baseline omission without training another
model. It consumes `pcno_shock_vortex_physical_val_s20260718_20260721b`, the
selected epoch-44 checkpoint SHA-256 `c5e468c7045...`, and a fresh same-host
timing-only replay under the unchanged CUDA/bfloat16/no-intervention contract.
Accuracy always comes from the frozen 24-case validation artifact; the timing
replay cannot replace targets or select a checkpoint. The 27 strength-OOD test
trajectories remain sealed.

The immutable CFD candidates are `25x10`, `50x20`, `125x50`, and `250x100`.
All use the existing WENO5--HLLC--SSPRK3 solver in float32 with adaptive CFL,
raw accepted states, linear x extrapolation, reflecting y boundaries, and no
future-reference boundary, floor, clip, limiter, or smoother. The D013 cohort
`sv_e00/e06/e11 x y00/y08` screens all four. The grid closest in log runtime to
the calibrated PCNO H60 cost and the grid closest in log H60 error to PCNO on
the same six cases are extended to all 24 validation trajectories; if they
coincide, one deterministic combined-distance runner-up is added. No more than
two grids receive full evaluation.

Coarse initial cell averages are exact uniform block restrictions of the D044
250x100 initial state. Predictions return to 250x100 only by conservative
piecewise-constant prolongation. Restrict-then-prolong reference truth is the
declared oracle remapping-error floor; it is not a CFD prediction. Solver cost
includes synchronized evolution, CFL reductions, SSPRK stages, retries,
retained save states, and the solver's own boundary sums, while excluding
initialization, host export, remapping, metrics, and artifact I/O. PCNO uses the
matching synchronized forward boundary and multiplies one-call latency by 60.

Strict cost wording additionally requires PCNO per-call p95/median at most
`2`, an observed CFD/PCNO cost ratio at most `2`, complete admissible rollouts,
and maximum coarse own-boundary balance error at most `5e-5`. An observed
error-matched row requires a symmetric H60 error ratio at most `1.25`.
Paired 95% intervals resample the 24 validation cases only and are not seed
uncertainty. Failure of either match gate leaves that comparison descriptive;
it does not authorize grid interpolation, additional CFD resolutions, PCNO
training, test access, a production-CFD claim, or any Line-4 flag change.

## 2026-07-22 D051 Coarse-CFD Error--Cost Closeout

Artifact `pcno_shock_vortex_coarse_cfd_d051_20260722a` contains 60 native
endpoint bundles and digest-valid tables: 3,600 call rows, 360 endpoint rows,
and two paired full-validation comparisons. The fixed pilot selects `25x10`
for closest measured cost and `250x100` for closest H60 error. All evaluated
rollouts are finite and admissible with no rejected step or reconstruction
fallback. The strength-OOD test split remains unopened.

The `25x10` row is not cost matched: median H60 core time is `1.938` s versus
the calibrated PCNO median estimate `0.449` s, a `4.31x` ratio, and H60 error is
`0.53185` versus `0.0083419`. Its remapping floor is only `0.022814`; error is
near that floor at call 1 but grows to `23.3x` the floor by H60. Thus its failure
is coarse dynamical/discretization error, not an artifact of prolongation.

The `250x100` row is error matched: mean H60 error is `0.00808583`, `3.07%`
below PCNO, with symmetric ratio `1.0317`. The paired CFD-minus-PCNO mean is
`-0.0002561`, but its case-bootstrap 95% interval `[-0.0008367, 0.0003231]`
contains zero and PCNO wins 10/24 cases. Aggregate equality hides a physical
tradeoff. CFD improves front IoU/chamfer and vortex-core error, while PCNO has
lower shock-strength and shock-thickness log error and lower smooth-region
scaled error; CFD has lower smooth high-pass energy. This is consistent with
PCNO retaining sharper shocks while carrying more ripple energy, not with one
method uniformly dominating.

Strict error--cost validity is false. The fresh PCNO timing replay has median
one-call latency `0.007485` s but p95 `0.087588` s, a `11.70x` ratio versus the
registered `2x` gate. Descriptively, 250x100 CFD costs `13.53` s against PCNO's
`0.449` s calibrated median H60 estimate and `2.225` s frozen observed median;
these are implementation measurements, not a production-CFD speedup claim.
The float32 coarse solver also misses its own `5e-5` balance gate, at
`6.43e-5` for 25x10 and `2.99e-4` for 250x100. Do not retune precision or the
tolerance post hoc. D051 therefore closes only the matched-error and
descriptive-cost question, makes no physical-conservation claim, opens no new
grid or training row, and changes no Line-4 flag.

The result-to-claim verdict is `partial`: validation-family matched-error
competitiveness is supported, while strict speed, physical conservation,
general CFD superiority, seed uncertainty, and strength-OOD behavior are not.
The post-run source change only narrows the claim-boundary wording from
"physical boundary balance" to "measured own-boundary residual"; it does not
change or rerun the digest-bound artifact.

## 2026-07-22 D052 Frozen Branch-Gain Attribution Closeout

D052 reuses the six D013 validation trajectories at calls `1/10/30/60` and
adds no training, test access, clipping, floor, smoothing, limiter, or future-
reference boundary. The code-frozen selector applies one-sided gain `1.00 ->
0.99` independently to each of four layers and the spectral, pointwise, and
differential branches under both teacher-forced and rollout-state inputs. A
row requires at least `0.10` smooth-region high-pass improvement elasticity,
no state/front/strength/thickness elasticity below `-0.05`, and admissibility.
A branch must pass at least three of four calls on five of six cases in both
input regimes before it can route one zero-training capacity test.

Artifact `pcno_shock_vortex_branch_sensitivity_d052_20260722a` completes all
48 source/case/call rows. Exact traced-model replay differs by at most
`4.76837e-7`, and every base and attenuated output is finite and admissible.
No branch passes even one complete row, so every source/branch has zero passing
cases. Median teacher/rollout smooth-high-pass elasticities are
`-0.0208/-0.0050` spectral, `-0.0118/-0.0120` pointwise, and
`0.0324/0.0249` differential. The weak differential ripple tendency is not a
safe correction: its median teacher state/front elasticities are
`-0.1680/-0.2999`, and its rollout front elasticity is `-0.3426`. Pointwise
attenuation likewise trades occasional ripple reduction against shock-strength
regression; spectral attenuation does not support a Gibbs-specific repair.

The formation traces separate ordinary recurrence from selective ripple-band
amplification. At call 60, median rollout state error is `0.00871484` versus
teacher-forced `0.000973433`, an `8.95x` ratio, while smooth-region high-pass
error is `0.00535199` versus `0.00488767`, only `1.095x`. The spectral branch
remains smoother than the local branches and no hidden-state high-band
explosion appears uniquely under rollout. D052 therefore returns
`composite_or_unresolved` and routes `stop_line3_architecture_tinkering`.
This is frozen off-path sensitivity, not a trained ablation or proof of causal
branch independence. Do not launch a gain, basis, smoothing, or local-branch
sweep from it.

## 2026-07-22 D053 Exact Rollout-Error Source Registration

At registration, D053 was the sole permitted Line-3 diagnostic at that
historical stage. It trained no model and used only D052's six saved validation
rollouts plus the same digest-bound D044
checkpoint and shards. For every call `1--60`, let `G` be the legal raw PCNO
map, `u_t` the reference state, and `uhat_t` the rollout state. Record the exact
identity

`G(uhat_t) - u_(t+1) = [G(uhat_t) - G(u_t)] + [G(u_t) - u_(t+1)]`

as propagated input error plus fresh teacher-forced defect. Measure both terms,
their cosine/cross energy, shock/smooth support, and their linear graph-high-
pass images using validated cell-volume weights. The call-1 propagated term is
the zero control. Full-field and high-pass reconstruction residuals must be at
most `1e-6` relative, all 360 rows must be present, and all teacher predictions
must remain raw and admissible.

At calls 30 and 60 define the propagated magnitude share as
`||p||/(||p||+||d||)`. A source is propagation dominated only if both full-
field and smooth-high-pass shares are at least `0.65` on at least five of six
cases at both calls; it is fresh-defect dominated only if both are at most
`0.35` under the same repetition rule. All other outcomes are mixed. Under that
historical contract, only the first outcome could route one matched short
generated-state-exposure capacity test; the second routed
target/representation diagnosis without recurrence training; a split or mixed
result authorized no learned method. D053 cannot
change D044, open strength OOD, or support a conservation or flux claim.

## 2026-07-22 D053 Attempt-A Replay-Binding Failure

Attempt `pcno_shock_vortex_error_source_d053_20260722a` writes all 360 rows,
keeps every raw teacher and rollout proposal admissible, and closes the
additive full/high-pass identities to `4.91408e-15`. It is not a scientific
selector result because the call-1 zero-propagation control fails. The code
compared D052's saved prior-process rollout output with a newly replayed
teacher output at the identical current. Their full-field propagated norms are
only `1.65e-7--1.84e-7`, but their shares are `1.97e-5--2.31e-5` full and
`3.52e-4--4.31e-4` high-pass, above the frozen `1e-6` zero-control gate.

Attempt A was preserved as fail-closed implementation provenance. At that
historical gate, one corrected attempt B was authorized without changing any scientific threshold: evaluate
both `G(uhat_t)` and `G(u_t)` in the same process, reuse one output when the
saved currents are bit-identical, and record the maximum absolute replay
difference from D052's saved proposal. That replay difference must be at most
`1e-5`. The saved rollout currents remain frozen, so this is not a regenerated
recurrence or a second scientific trial. Any other failure stops D053.

## 2026-07-22 D053 Corrected Closeout

Corrected artifact `pcno_shock_vortex_error_source_d053_20260722b` passes every
contract check on all 360 rows. Maximum additive-identity residual is
`4.93441e-15`; maximum replay difference from D052 is `4.76837e-7` versus the
fixed `1e-5` gate; the call-1 propagated share is exactly zero; no smooth-mask
fallback occurs; and all teacher and rollout proposals are raw, finite, and
admissible.

The result is a repeated split mechanism. At calls 30/60, median full-field
propagated magnitude shares are `0.88995/0.89043` with six-case ranges
`0.87827--0.90709` and `0.88227--0.92324`. The corresponding smooth-high-pass
shares are only `0.15619/0.30733`, with every case at or below `0.34958`.
Full-field propagation gains relative to the incoming error are about
`0.9878/1.0026`; smooth-high-pass gains are only `0.1889/0.4056`.

The energy identity makes the distinction sharper. At call 30, propagated,
fresh-defect, and cross contributions to full error are
`0.9159/0.0140/0.0710`, while smooth-high-pass contributions are
`0.0326/0.9475/0.0228`. At call 60 they are
`0.9099/0.0136/0.0746` full and `0.1669/0.8472/-0.0035` smooth high-pass.
Thus recurrence carries the large, mostly shock-supported state error nearly
neutrally, while the smooth-region ripple is primarily regenerated by the
teacher-forced one-step defect rather than amplified from the incoming ripple.

D053 returns `mixed_or_split` and `no_learned_method`. Generated-state exposure
is not selected as a ripple remedy because the high-pass response is
contractive; a target-only change is not selected because the dominant full
state error is propagated. Combined with D052, this stops gain, local-basis,
smoothing, local-corrector, and generated-exposure branches under the current
contract. D044 remains the flagship residual baseline. Keep strength OOD
sealed and do not spend serious-run or seed-confirmation compute until a new,
explicitly authorized target/representation hypothesis has its own falsifier.

## 2026-07-22 D054 Fresh-Defect Locality Registration

At registration, D054 was the sole permitted Line-3 action at that historical
stage. It was a zero-training necessary-condition test for one new representation hypothesis: keep the
successful global conservative-residual PCNO path, but factor its target into
a global component and a current-state shock-conditioned local detail. This is
not a retry of D042's fixed geometry-only basis or D044's inference-time local
correction. It analyzes D053's teacher-forced fresh defect before any model or
rollout is changed.

Use exactly D052/D053's six validation trajectories and calls `1--60`; keep the
27 strength-OOD cases sealed. Scale conservative components by the frozen D044
training scales, use validated physical cell volumes, apply the same linear
graph high-pass and D053 two-hop smooth mask, and define one causal seed from
the current input state's top-decile pressure jumps. Record graph-hop energy
rings, component shares, neighbor correlation, and two upper bounds. The
truth-informed bound may select the best 20% of smooth nodes. The legal sensor
bound is the fixed four-hop current-state seed halo and may use neither the
target nor rollout state.

At both calls 30 and 60, at least five of six cases must meet each repeated
gate: the 20% truth oracle captures at least 70% of smooth high-pass fresh-
defect energy; the causal halo captures at least 50% while covering at most
25% of interior nodes. All 360 rows, masks, raw teacher admissibility, and D053
fresh-norm replay within `2e-4` relative are contract gates. Under the historical
contract, passing both conditions would have authorized only drafting one matched tiny-fit contract for a zero-
initialized shock-conditioned detail target. Oracle-only locality rejects a
simple shock gate; failure of the oracle rejects the bounded local-target
route. No serious run, test access, basis/sensor-radius sweep, conservation
claim, or Line-4 flag change is authorized by D054.

## 2026-07-22 D054 Fresh-Defect Locality Closeout

Artifact `pcno_shock_vortex_fresh_defect_locality_d054_20260722a` completes all
360 rows with every contract check true. Summary/row SHA-256 values begin
`2cd8a79a` and `94e99dc1`. The truth-selected 20% smooth-node oracle passes all
six cases at both late calls, capturing median `0.98741` at call 30 and
`0.92536` at call 60. The current-state four-hop pressure-jump halo passes zero
cases: median capture is only `0.18751/0.20290`, although its interior support
is `0.16526/0.20885`.

The failure is not evidence that the fresh defect is diffuse. At calls 30/60,
median energy beyond eight hops is `0.55838/0.49105`, while the sparse oracle
still captures more than 90%. Smooth-neighbor cosine is negative
(`-0.27573/-0.21278`), and x-momentum carries median `0.62481/0.58079` of the
scaled high-pass energy. The defect is sparse and oscillatory but not confined
to the current shock halo. D054 therefore returns
`localized_but_not_causally_shock_localizable` and rejects the simple
shock-gated detail target without training. Do not widen the halo or sweep the
sensor after seeing this result.

## 2026-07-22 D055 Frozen-Proposal Self-Sensor Registration

At registration, D055 was the sole permitted Line-3 action at that historical
stage. It tested one fixed legal score,
not a sensor family: the nodewise norm of the graph high-pass of the frozen
PCNO proposed conservative update `(G(u_t)-u_t)/scale`. Exclude exactly the
two-hop halo of the current input's top-decile pressure-jump seed, rank the
remaining interior nodes, and retain the top 20% of those eligible nodes. The
selection uses no target, future boundary, rollout state, or branch sweep.

Use the six D052 validation trajectories and only calls `1/10/30/60`; keep
strength OOD sealed. D053 fresh norms and D054 truth-oracle capture must replay
within `2e-4` relative, every teacher proposal must be raw-admissible, and the
selected support must remain at most 20% of the interior. At both calls 30 and
60, at least five of six cases must capture at least 50% of D053 smooth-high-
pass fresh-defect energy. Under the historical contract, a pass would have
authorized only writing one matched tiny-fit contract. Failure rejects the
proposal-sensor-gated local-detail route. No
score, exclusion-radius, support, or threshold sweep; no training, test access,
conservation claim, D044 change, or Line-4 flag change is authorized by D055.

## 2026-07-22 D055 Proposal Self-Sensor Closeout

Artifact `pcno_shock_vortex_proposal_sensor_d055_20260722a` completes all 24
rows with every contract check true. Summary/row SHA-256 values are
`894ecd23...` and `66e493f9...`; maximum D053/D054 replay discrepancy is
`1.84322e-6`. The legal score passes all six cases at both late calls. Median
fresh smooth-high-pass energy capture is `0.91224` at call 30 and `0.80212` at
call 60, with ranges `0.90494--0.91803` and `0.75534--0.82258`. Median interior
support is only `0.17304/0.16866`.

The proposal score recovers median `0.92347/0.87245` of the truth-oracle energy
at calls 30/60. Its smooth-node Spearman correlation with fresh-defect high-
pass amplitude is `0.87482/0.73318`. Thus the frozen proposal exposes where
the later one-step ripple is likely to occur even though current shock
location does not. This is localization evidence only: it does not show that a
bounded admissible correction of the right sign is realizable or learnable.
D055 returned `proposal_self_sensor_candidate` and historically routed the
now-completed D056 contract below, not a training run.

## 2026-07-22 D056 Causal-Support Correction-Realizability Contract

At registration, D056 was the sole permitted Line-3 action at that historical
stage and remained zero-training. On
the same six validation trajectories and calls `1/10/30/60`, reconstruct the
exact D055 support from current state plus frozen proposal. Within that already
fixed support, truth may define an upper-bound state-correction direction.
Project each conservative component to zero validated-volume integral over the
support, cap its physical-volume/component-scaled norm at 10% of the frozen
global update norm, and use a fixed line search only to enforce raw
admissibility and minimize teacher state error. The correction is a globally
balanced state-space oracle, not a face flux or local conservation claim.

Every row must retain at most 20% interior support, at most 10% update norm,
volume-integral correction at most `1e-10`, raw density/pressure/internal-energy
admissibility, and no more than 5% worsening in front position, shock strength,
shock thickness, vortex core, or smooth high-pass energy. At calls 30 and 60,
median state-error reduction must reach 15%, median smooth-high-pass reduction
20%, and at least five of six cases must be nonworse on both. Under the
historical contract, a pass would have authorized implementation of one
zero-initialized tiny detail-head fit with the global
D044 model frozen. Failure rejects that learned detail route. D056 cannot train,
open test OOD, tune support/norm/line search, claim flux conservation, or change
Line-4 flags.

## 2026-07-22 D056 Causal-Support Correction-Realizability Closeout

Artifact `pcno_shock_vortex_proposal_correction_oracle_d056_20260722a`
completes all 24 rows and passes every structural contract. Summary/row
SHA-256 values are `90f5256a...` and `016e1ec5...`; maximum D055 replay error
is `1.00394e-5`, maximum absolute volume-integral correction is
`3.61508e-17`, every state is raw-admissible, and every anti-smearing check
passes. Interior support spans `0.16767--0.17474`.

The material gates fail. At calls 30/60, median state-error reduction is only
`0.07185/0.07010` versus the required `0.15`. Median smooth-high-pass
reduction is `0.23763/-0.03592`; all six call-30 cases are jointly nonworse,
but only two of six call-60 cases are. This is not caused by the 10% norm cap:
the unscaled balanced truth direction itself uses median
`0.02361/0.05783` of the frozen global-update norm. All call-30 rows accept
the full direction, while the median applied call-60 ratio is only `0.02750`
because the front/high-pass safeguards restrict the line search.

D055 therefore found a legal locator for fresh high-pass energy, but that
sparse support does not contain enough safely correctable state error. At the
late endpoint, a support-limited balanced adjustment can also add graph jumps
at the support boundary. D056 returns
`bounded_causal_support_correction_insufficient` and rejects the learned local
detail head without training. Do not widen the support, relax balance or
anti-smearing, raise the update cap, or sweep the sensor. This result does not
reject end-to-end changes to the global map; any new Line-3 learned proposal
must explicitly address D053's propagated shock-supported state error and its
fresh smooth-region high-pass defect together, with a new predeclared
falsifier. Strength OOD remains sealed and all Line-4 flags remain unchanged.

## 2026-07-22 D057 Joint-Objective Gradient-Compatibility Contract

At registration, D057 was the sole permitted Line-3 action at that historical
stage. It took zero optimizer steps and never changed the D044 checkpoint. The new hypothesis was that the global
map must be trained against both parts of D053's split mechanism rather than
repaired afterward. At the frozen checkpoint define three full-model losses:
the existing clean physical-volume/component-scaled state MSE; graph-high-pass
MSE of the teacher-forced defect on the intersection of the reference-current
and target two-hop smooth masks; and state MSE after one detached raw generated
call. The latter is the existing depth-one exposure contract, not a longer
unroll. No input noise is sampled in this deterministic audit; D044 itself
remains the noise-trained parent.

Use exactly D048's four immutable training pairs to form one aggregate
float32 gradient for each loss. Normalize those three gradients separately and
sum them to define the sole candidate direction; this fixes relative scaling
without a loss-weight sweep. Then compute directional cosines for each loss on
the six D013 validation trajectories independently at calls 30 and 60. The 27
strength-OOD trajectories remain sealed. Record full and layerwise norms,
pairwise Gram/cosine matrices, candidate-direction cosines, raw generated-state
admissibility, mask mass, checkpoint/configuration/data digests, device, peak
memory, and an explicit zero-optimizer-step ledger.

Every training directional cosine must be at least `0.05`. At each of calls 30
and 60, at least five of six validation cases must have cosine at least `0.02`
for all three losses. All gradients and losses must be finite, every generated
input must remain raw-admissible, every smooth mask must have positive physical
volume, and the exact split/pair/call contract must close. Under the historical
contract, passing would have authorized only a separately frozen short
full-resolution continuation contract; it did not authorize that run. Failure rejects naive equal-gradient scalarization,
not all multiobjective optimization. D057 may not train, update buffers, use
AMP, access test OOD, tune thresholds or loss definitions, alter D044, claim
conservation, or change Line-4 flags.

## 2026-07-22 D057 Joint-Objective Gradient-Compatibility Closeout

Artifact `pcno_shock_vortex_joint_objective_gradients_d057_20260722a`
completes its four training pairs and 12 validation rows with every structural
check true. Summary/row SHA-256 values are `72ed8121...` and `c78187ea...`.
The ledger records 48 backward calls, no optimizer creation or step, identical
before/after parameter-state digests, no test access, and 1.991 GiB peak CUDA
allocation. All generated inputs are raw-admissible and every smooth mask has
positive physical volume.

The fixed training direction passes: its cosines with clean state, smooth
high-pass, and generated-state gradients are `0.90133/0.41047/0.87180`.
Clean and generated gradients are nearly identical (`0.98776` cosine), while
smooth high-pass is weakly opposed to them (`-0.01961/-0.08409`). Validation
transfer is not repeated. Only four of six cases pass jointly at call 30 and
two of six at call 60, below five. High-pass transfer itself is positive on all
12 rows (`0.1555--0.5833`); clean/generated transfer fails by geometry group.
At call 60 all three `y00` cases have negative clean and generated cosines,
whereas the `y08` cases retain positive generated cosines and two retain
positive clean cosines.

D057 returns `naive_equal_gradient_scalarization_incompatible`; do not run its
continuation or change its three weights. The result does not show that the
mechanism-matched objectives are intrinsically incompatible. It shows that a
direction estimated from D048's four-pair bank is not geometry-group robust.

## 2026-07-22 D058 Geometry-Group MGDA Contract

At registration, D058 was the sole permitted Line-3 action at that historical
stage and again took zero optimizer steps. It used all 84 full-resolution training trajectories, equally weighted at
calls 30 and 60. For each training `y_index` group `1--7`, independently
average clean-state and detached-generated-state gradients over all 12 training
strengths and both calls; unit-normalize those two aggregates, sum them, and
renormalize to one state/recurrence task. Separately average and unit-normalize
the smooth-high-pass gradient over the complete 84-case cohort. These seven
geometry tasks plus one high-pass task are fixed; no validation gradient enters
their construction.

Choose their single common direction by the minimum-norm convex combination on
the eight-task simplex, using exactly 500 deterministic Frank-Wolfe iterations
from the lexicographically first task and the analytic segment minimizer. No
task deletion, alternative grouping, loss coefficient, seed, or solver sweep
is allowed. The final Frank-Wolfe gap must be at most `1e-6`, the direction
must be finite and nonzero, and its cosine with every training task must be at
least `0.02`. On the unchanged six D013 validation cases at calls 30 and 60,
at least five of six cases per call must have cosine at least `0.02` for clean,
smooth-high-pass, and generated losses simultaneously.

All full-resolution data, split/digest bindings, positive smooth mass, raw
generated-state admissibility, parameter-state equality, zero-step ledger, and
test nonaccess are structural gates. Under the historical contract, a pass would
have authorized only drafting one short full-resolution geometry-group MGDA
continuation contract. Failure stops this
joint-objective route. D058 cannot train, use AMP/noise in the audit, access
strength OOD, alter the objectives/groups/calls after inspection, claim
conservation, modify D044, or change Line-4 flags.

## 2026-07-22 D058 Geometry-Group MGDA Closeout

Artifact `pcno_shock_vortex_group_robust_gradients_d058_20260722a` completes
the exact 84-trajectory training audit and 12 validation rows with every
structural check true. Summary/row SHA-256 values are
`45fe90196ce0c8c7d9b6d7e192b47e4404e9aea2b0260206aa1866be7116ddc0`
and `57ccc449273e1b9465dcaeb7a72c91135dc26ed70f4e2af978954ccfb6c41045`.
The ledger records 288 backward calls, zero optimizer creation or steps,
identical before/after parameter digests, no AMP, noise, or test access,
`153.708` wall seconds, and 2.875 GiB peak CUDA allocation. A batched
`node_type [B,N]` versus `node_mask [B,N,1]` preflight mismatch was fixed
and regression-tested before the canonical run; the failed technical attempt
created no artifact.

The 500-step Frank-Wolfe solve closes to gap `5.55e-17`. Every training-task
cosine is positive and at least `0.563997`, so the complete training split is
not the failure. All six call-30 validation cases pass jointly. At call 60 only
three of six pass, below the frozen five-case gate. High-pass and generated-
state cosines remain above `0.02` on all six rows. The repeated failure is
clean state loss on all lower boundary-nearest `y00` cases, whose cosines are
`-0.01409-- -0.00916`; all upper `y08` clean cosines are positive
(`0.03534--0.11023`). Relative to D057, the new direction raises median clean
and generated transfer at both calls, while retaining positive but weaker
high-pass transfer.

D058 therefore returns `geometry_group_joint_objective_insufficient` and
stops the current joint-objective continuation without training. The result
rejects this single static full-model direction as a safe long-horizon,
position-OOD continuation. It does not reject all multiobjective optimization,
finite-step retraining, or neural operators. No weight, grouping, call, or
Frank-Wolfe variant may be tried post hoc; strength OOD stays sealed and Line 4
is unchanged. A later Line-3 proposal must be a genuinely new
target/representation or solver-coupling hypothesis with its own necessary-
condition and cost falsifier.

## 2026-07-22 D059 Direct-Stride-2 PCNO Tiny-Fit Contract

At registration, D059 was the sole permitted Line-3 action at that historical
stage. D053 shows that smooth-region
high-pass error is freshly injected by the one-call defect while the dominant
state error is carried nearly neutrally. Rather than add another loss or branch,
D059 changes only the fixed macro-map from one saved interval to two. A direct
stride-2 model needs 30 rather than 60 recurrent calls to reach `t=0.6`;
whether its harder target can be fit is tested before any serious run.

Use the unchanged full-resolution 250x100 shards, 84/24/27 grouped split, seed
and split seed `20260718`, 19,155,720-parameter conservative-residual PCNO
(`k_max=8`, five width-128 layers, `fc_dim=128`), AdamW at `1e-3`,
`1e-5` weight decay, unit gradient clip, constant schedule, and BF16. Set
`step_stride=2`, batch size one, zero input noise, zero generated-state
exposure, four presentations per epoch, and at most 50 epochs with early stop.
The immutable pair bank is `(sv_e08_y07,8)`, `(sv_e03_y03,7)`,
`(sv_e03_y01,44)`, and `(sv_e05_y07,40)`, generated by the existing
seeded sampler before launch. No downsampling, initialization checkpoint, or
test state is allowed.

Pass requires best tiny-bank relative L2 at most `0.01`, loss ratio at most
`0.01`, finite gradients, 100% raw admissibility, and one complete raw
30-call position-OOD validation smoke with no future boundary, floor, clip,
limiter, smoothing, or projection. All model/data/split/normalization/code
digests and the explicit pair bank must be saved. The run budget is one attempt
and at most 0.05 GPU-hour. Under the historical contract, a pass would have
authorized only drafting a matched serious stride-2 contract; it did not
authorize that training. Failure stops direct
stride 2 without a warm start, threshold retry, stride-4 fallback, or
architecture/loss change.

## 2026-07-22 D059 Direct-Stride-2 Tiny-Fit Closeout

Artifact `pcno_shock_vortex_stride2_tinyfit_d059_20260722a` passes the exact
gate at epoch 35 in `69.4304` seconds, or 0.0193 GPU-hour. Summary, split,
normalization, and metrics SHA-256 values are `40a15670...`,
`b2d2fc78...`, `4c931c81...`, and `a29ac681...`. The artifact binds the
19,155,720-parameter model, full 84/24/27 split, four preregistered pairs,
manifest/configuration/normalization digests, and trainer/adapter/core source
hashes.

Initial tiny-bank relative L2/loss are `0.0124691/0.0409940`. Best values are
`0.00115519/0.000400830`, giving loss ratio `0.00977779` and 100% raw
admissibility. The fixed position-OOD smoke `sv_e01_y08` completes all 30 raw
calls with final relative L2 `0.0524226`, minimum density `0.854982`,
minimum pressure `0.780739`, and no inference intervention. No strength-OOD
state is loaded.

Stride 2 is almost exactly scale-consistent with the earlier stride-1 tiny fit:
its four residual component scales are `1.95--1.98x`, its initial/best state
errors are `2.083/1.994x`, and its relative loss reduction is slightly
stronger (`0.009778` versus `0.009972`). D059 therefore verifies target
fitability without revealing a new optimization pathology. The smoke case and
stride-1 tiny fit use different validation trajectories, so their rollout
errors are not comparable. D059 historically permitted drafting the
now-completed D060 contract only; it is not
evidence that stride 2 improves rollout, ripple, shocks, conservation, or OOD.

## D060 Matched Serious Stride-2 Result (promotion failed 2026-07-22)

The historically authorized one-seed run changed only D044's `step_stride` from
one to two and its validation rollout length from 60 to 30 calls. It froze seed/split seed
`20260718`, 50 epochs, 1,024 training and 256 validation presentations per
epoch, batch four, all 24 position-OOD validation rollouts every five epochs,
the same architecture/optimizer/BF16 settings, training-only primitive noise
`0.003`, zero generated exposure, cold initialization, raw model-all-node
recurrence, and the full-resolution data contract. Estimated budget is at most
2.5 GPU-hours and 1 GiB retained storage.

Selection used only all-24 raw H60 validation. Promotion required 24/24 finite,
admissible completion; at least 10% lower mean H60 physical-volume state error
than D044; at least 20% lower six-case D013 H60 smooth-high-pass RMS; no more
than 5% regression in shock position, strength, thickness, vortex-core error,
or physical-total mismatch; direct frame-2 error at most 1.15 times two
composed D044 calls; and measured H60 wall time at most 0.65 times D044 on the
same host. The contract required mixed-prefix, completed-case, common-endpoint,
and raw failure statistics. The 27 strength-OOD cases remained sealed.

The frozen row completed all 50 epochs as
`pcno_shock_vortex_stride2_serious_d060_s20260718_20260722a`. It selects epoch
34 and checkpoint SHA-256
`95e6c180a3298c4b38662d8c6cb77301c0e7fd0286e9a53571bddc487b8379c9`.
Summary/metrics/split SHA-256 values are `b92d8bef4dab...`, `231027ecb52b...`,
and `1be17494eaac...`; configuration and data-manifest digests are
`0064571b5414...` and `f8d228ae3c6e...`. Recorded training wall time is
1,247.44 seconds and peak allocated GPU memory is approximately 4.58 GiB.

The frozen physical evaluator completes 24/24 cases with mean H60 state error
`0.00729369`, giving a D060/D044 ratio `0.87434`; D060 wins all 24 paired cases.
Direct frame 2 is `1.03246x` two composed D044 calls. Mean shock-strength,
shock-thickness, vortex-core, and physical-total-mismatch ratios are
`0.78994/0.86185/0.34240/0.55000`, so these registered components pass.

Promotion fails on the intended ripple mechanism and front position. The six-
case D013 H60 median rollout smooth-high-pass RMS is `0.00548836`, versus
D044's `0.00535199`, a `1.02548x` ratio rather than `<=0.8x`. The all-24
endpoint graph-high-pass energy is worse in every paired case; its mean energy
ratio is `1.36252` and mean RMS ratio `1.16727`. Mean front-centroid distance is
`0.0260282` versus `0.0181368` (`1.43511x`), while symmetric chamfer is nearly
unchanged (`0.99522x`). The centroid ratio is systematically `2.74982x` on
`y00` and `0.74853x` on `y08`, so aggregate position error hides opposite
geometry-conditioned behavior. Physical/D013 summary SHA-256 values begin
`50ad14177f1e...` and `2f2b5d8c6414...`; paired D044 values begin
`ded10f92fb6e...` and `cf8815020303...`.

Matched-time diagnostics explain the partial gain. The state-error ratio starts
at `1.032` at `t=0.02`, crosses below one by `t=0.08`, reaches `0.708` near
`t=0.26`, and ends at `0.874`. On the D013 cohort, H60 teacher-forced error is
`1.432x` D044 but raw rollout error is `0.854x`; teacher-forced and rollout
high-pass RMS are still `1.033x` and `1.025x`. A harder local stride-2 map is
offset by half as many recurrent compositions in state L2, while the high-pass
source remains. Epoch 44's lower one-step validation error and higher H60 error
than epoch 34 independently show why recurrent selection is required.

Median synchronized forward time times the required call count gives an H60
ratio `0.480`, but D044's batch-1 p95 outlier makes the strict cost comparison
descriptive. Timing cannot rescue the failed physical conjunction. Strength OOD
remains sealed. No retry, extra seed, stride 4, warm start, loss change, method
add-on, Line-4 flag change, or data-assimilation work is authorized.

## D061 Frozen Multirate Rollout-Blend Headroom Contract (historically authorized 2026-07-22)

At registration, D061 was the sole permitted Line-3 action at that historical
stage and was frozen-artifact synthesis, not a training run or method add-on. It used the completed D044 stride-1 and D060 stride-2
raw validation artifacts for exactly the six D013 trajectories. Align stride-1
frames `2/10/30/60` with stride-2 calls `1/5/15/30`. Evaluate the two parent
paths, one legal fixed `alpha=0.5` convex blend, and one truth-informed global
convex oracle on the fixed 21-point grid `alpha=0,0.05,...,1`. The oracle may
use the aligned target only to select alpha and is never an autonomous result.

Promotion was conjunctive. All 24 case/call rows and all four variants had to be
finite and raw-admissible. At H60, oracle state error must improve by at least
10% over the better parent in the median case; smooth-region graph-high-pass
RMS must improve by at least 20% over D060; oracle state and high-pass must both
be nonworse than both parents on at least 5/6 cases; and front centroid,
symmetric chamfer, shock strength, shock thickness, vortex-core error, and
reference-boundary physical-total mismatch must stay within 5% of the
metricwise better parent on at least 5/6 cases. Separately report whether the
target-free direct/composed disagreement reaches median late-call Spearman
`>=0.5` and whether its top-20% nodes capture at least 50% of better-parent
error energy; this localization readout does not choose oracle alpha.

The run budget was zero GPU-hours, no checkpoint execution, at most 0.5 GiB
retained output, no strength-OOD or test access, and no threshold retry. Under
the historical contract, a full pass would have authorized only drafting one
shared-backbone multirate tiny-fit contract;
failure rejects this two-rate blend as the next stabilization branch. Neither
outcome establishes a cheaper solver, model-predicted face flux, or physical
conservation.

## D061 Frozen Multirate Rollout-Blend Closeout (failed 2026-07-22)

Artifact `pcno_shock_vortex_multirate_headroom_d061_20260722a` completes all 24
registered rows, retains six trajectory bundles in 50 MiB, and reads no test
case or checkpoint. Summary, proposal-row, and variant-table SHA-256 values are
`01ec0b2d890f...`, `64fe6b9e1a2d...`, and `8d98d03d1ef2...`. Every matched
state, time, normalization, geometry, volume, face, orientation, and boundary-
exchange source check closes exactly, and all parent, equal-blend, and oracle
states are raw-admissible.

The oracle misses every scientific promotion component. At H60 its median
state-error reduction over the better parent is `8.14%` rather than `>=10%`,
and its median smooth-region graph-high-pass RMS reduction from D060 is
`10.10%` rather than `>=20%`. It is jointly nonworse than both parents in 0/6
cases and passes the metricwise anti-smearing envelope in 0/6. Shock strength,
shock thickness, and vortex-core error each fail in 5/6 cases; front centroid
fails in 3/6, while chamfer and physical-total mismatch each fail once. The
truth-selected H60 alpha lies in `0.30--0.45`.

The parent roles are complementary but not composable by state averaging:
D060 has lower H60 state error on 6/6 cases, while composed D044 has lower
smooth high-pass RMS on 6/6. The legal fixed `alpha=0.5` control reduces median
state error by `7.16%` relative to the better parent and high-pass RMS by
`12.02%` relative to D060, but remains rougher than D044 and violates the same
front hierarchy. Its estimated two-checkpoint H60 cost is `1.479x` D044 and
`3.086x` D060. Early oracle gains at calls 1/5 are larger, but decay by calls
15/30 as the recurrent paths diverge.

Direct/composed disagreement is nevertheless informative: its late-call
median Spearman correlation with better-parent node error is `0.650`, and its
top-20% nodes capture `57.81%` of that error energy. This independently passes
the descriptive localization readout, not alpha selection or correction
realizability. Together with D055/D056, it shows that finding troubled nodes is
not the current bottleneck. Reject the exact two-rate state-blend and do not
train the shared-backbone tiny fit, retry thresholds, or reinterpret the legal
equal blend as a promoted solver. A successor must preserve discontinuity
phase and strength nonlinearly rather than average phase-displaced states, and
must receive a new oracle contract before training.

The stride-aware D013 selector repair also replays D060's saved sensitivity
rows without checkpoint execution. Calls `1/5/15/30` now close the intended
four-time contract and return `composite_or_unresolved` with zero qualifying
spectral, pointwise, or differential branches for both teacher-forced and
rollout sources. The former `incomplete_frozen_contract` label was bookkeeping,
not mechanism evidence.

## D062 Front-Fitted Conservative-Remap Oracle Contract (historically authorized 2026-07-23)

At registration, D062 was the sole permitted Line-3 action at that historical
stage and took zero optimizer steps. It used only the completed D060 stride-2 raw validation artifacts for the frozen six D013
trajectories at calls 15 and 30 (physical frames 30 and 60). Reconstruct the
audited identity-mapped 250x100 tensor grid. In every row, extract one shock
coordinate `x_s(y)` from the absolute pressure jump on x-faces within the
fixed interval `shock_x +/- 0.10`; use the jump-weighted centroid of the
maximizing face and its immediate in-window neighbors. Do not smooth, clip,
regularize, or hand-correct either predicted or target curve.

The phase-only attribution row uses the target-informed curve displacement to
define a row-wise, domain-anchored piecewise-linear map through
`(x_min,x_min)`, `(x_pred,x_target)`, and `(x_max,x_max)`. Push every source
cell integral through that map using exact interval overlaps. The primary
phase-plus-strength row starts from this conservative warp and adds one
four-component, row-zero-total two-sided contrast: `+a` left of the target
curve and `-a V_left(y)/V_right(y)` on its right. Choose the four entries of
`a` by physical-volume least squares to the target. This leaves exactly four
fitted strength degrees of freedom while preserving each row/component total.
No target value enters the D060 recurrence, and both transformed rows are
post-hoc capacity oracles rather than autonomous forecasts.

All 12 source rows, identity-map, row-major coordinate, uniform-volume, face-
geometry, checkpoint/configuration/data/geometry digests, raw recurrence, and
legal-boundary contracts must close. Baseline, phase-only, and primary states
must be finite and raw-admissible. For both transformed rows, the maximum
row/component conservative-total defect, scaled by the larger of the source
row total and component-scale row volume, must be at most `1e-10`. No floor,
clip, limiter, smoothing, projection, boundary replacement, or failed-row
omission is allowed.

Scientific promotion is conjunctive on the six H60 rows. The primary oracle
must reduce median physical-volume state error by at least 15% and median
row-front MAE by at least 50% relative to D060. Median smooth-region graph-
high-pass RMS must not increase, and at least five cases must be jointly
nonworse in state and front MAE while keeping high-pass RMS within 5%. Front
centroid, symmetric Chamfer, shock-strength log error, shock-thickness log
error, and vortex-core density error must each stay within 5% of D060 on at
least five cases. Report phase-only attribution, displacement in cell units,
warp-Jacobian range, the fitted left/right offsets, correction-to-error and
correction-to-model-update norms, and physical-total/reference-boundary
metrics; these reports do not relax a failed gate.

The run budget was zero GPU-hours, no checkpoint execution, at most 0.25 GiB
retained output, no strength-OOD or test access, and no threshold or extractor
retry. Under the historical contract, a full pass would have authorized only
drafting one tiny-fit contract for this
front chart. Failure rejects the exact row-graph `x_s(y)` plus two-sided-
constant-strength chart as the next method; it does not reject every possible
level-set, discontinuous-coordinate, or shock-fitting representation. Neither
outcome establishes an autonomous phase predictor, predicted physical flux,
or a conservative neural solver.

## D062 Front-Fitted Conservative-Remap Oracle Closeout (failed 2026-07-23)

Artifact `pcno_shock_vortex_front_fitted_oracle_d062_20260723a` completes the
predeclared 12 validation rows and six trajectory bundles in 31.5 MiB. It uses
zero GPU-hours, executes no checkpoint, reads neither strength OOD nor test,
and binds implementation SHA-256 `ea7cce025359...` to the exact `+/-0.10`
front window and kink-split finite-volume pushforward. Summary, row-table, and
variant-table SHA-256 values are `9ea02b14eff1...`, `7708fcd85b8b...`, and
`c7dedf04f062...`. Independent replay reproduces every front curve, remapped
state, fitted coefficient, state metric, and promotion field exactly.

The source, geometry, admissibility, and conservation gates pass. All baseline,
phase-only, and primary states are finite and raw-admissible, and the maximum
scaled row/component total defect is `2.6914e-15`. The scientific conjunction
fails. On the six H60 cases, the primary target-informed oracle has median
per-case state reduction `-41.88%` rather than `>=15%`, front-curve MAE
reduction `62.82%` rather than `>=50%`, and smooth high-pass reduction
`-828.08%` rather than nonnegative. It is jointly nonworse in 0/6 cases.
Front centroid and symmetric Chamfer pass in 0/6, shock strength in 2/6, shock
thickness in 1/6, and vortex-core error in 2/6.

The four strength coefficients are not the cause. The H60 phase-only row
already improves front-curve MAE by `74.89%` while worsening state error by
`43.52%`; its ratio of median high-pass RMS to D060 is `9.626x`. Adding the
strength contrast changes the median state error only from `0.0100037` to
`0.0098950` and the high-pass RMS from `0.0992257` to `0.0991205`, while
front-curve MAE becomes worse than phase-only. Exact conservation therefore
does not make this coordinate warp physically accurate or smooth.

The failure is systematically stratified rather than an aggregate surprise.
The two `e00` cases have no row displacement above one cell and only
`1.035--1.141x` high-pass factors. The `e06/e11` cases contain two to five
rows above one cell, maximum displacements `2.81--10.54` cells, and adjacent
row jumps up to `10.51` cells; their high-pass factors are
`5.91--15.66x`. At the worst rows, predicted and target argmaxes switch
between separated, comparably strong pressure-jump branches. The unsmoothed
independent-row scalar `x_s(y)` therefore lacks a stable front-identity
contract under shock--vortex interaction. Aligning that scalar can improve its
own target-informed MAE while shearing the 2D field and damaging the physical
hierarchy.

Reject the exact row-argmax front coordinate plus global two-sided-strength
chart. Do not train a front predictor, retry the window, smooth the extracted
curve after seeing the failure, add strength degrees of freedom, access test,
or start data assimilation. This result does not reject every level-set,
multi-chart, or shock-fitting representation; it says that any future
reopening must first predeclare how front identity and transverse regularity
remain well defined through interacting jumps, then pass a new zero-training
capacity oracle. No successor experiment is authorized by D062.
