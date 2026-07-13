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

The completed 1D Euler target ladder compared solver-facing parameterizations
under standardized next-primitive-state supervision, with rollout diagnostics
for relative error, conservation, shock position, positivity, and limiter
behavior. It established useful failure controls but did not directly supervise
flux/interface labels or test a fully conservative-coordinate model. Noise and
positivity remain stabilizer controls, not reasons to repeat rejected target
sweeps.

The next causal sequence is: cross primitive/conservative inputs with
primitive/conservative loss while holding the conservative-residual output
fixed; export and validate exact macro-integrated face flux; compare residual,
state-loss-only flux, direct-flux, and joint supervision; then add dense
auxiliary labels one family at a time. Every finalist must pass a documented
one-step, direct-horizon, and 50-call raw-rollout capability protocol before
more elaborate invariant-domain correction, face routing, front tracking, INR,
or adaptive-time engineering is justified.

The old 1D rows labeled CPGNet used a generic directed residual head and are
deprecated. The corrected `cpg_interface` baseline reconstructs positive
directed interface states, forms one Rusanov flux per face, applies the exact
finite-volume update, and evaluates raw recurrence without a cell-state
limiter or hidden floor. Its CPU implementation gate passes; the benchmark GPU
run is pending.

Pending 2D diagnostics should still measure ripple energy and shock-front geometry using the common rollout artifact contract: `predicteds`, `targets`, `pos`, `edges`, and `node_type` when available. They are background constraints for the bump transfer rather than the immediate blocker for the 1D target selector.

## Not Ported Or Active

The raw `cpggnspdes` training scripts are not vendored into this branch. The CPGNet interface-latent probe code, state-drift/perturbation/time-alignment scripts, early smoke scripts, and raw-HDF5 PCNO adapter path were one-off research tools and are no longer active tracked code.

FNO, PCNO, and MPCNO baselines should use implementations already present in this repository or collaborator-provided preprocessing artifacts. Do not port baseline code from earlier data-assimilation experiment repositories.
