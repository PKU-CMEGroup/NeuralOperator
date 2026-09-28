# Completed Corrective-Mechanism Studies

Updated: 2026-09-28

This index routes completed evidence to its role in the paper.
[EXPERIMENT_TRACKER.md](EXPERIMENT_TRACKER.md) owns exact results, contracts,
artifact identities and execution limitations. This page is not an experiment
queue. The abstract and Sections 1--3 are locked; Section 4 is next.
New research waits for discussion of the consolidated draft with the mentor.

## Evidence Routing

| Paper role | Retained study or run |
| --- | --- |
| Clean competence and the recovery/composition discrepancy | `kf_fixed_data_20260916a`; independent confirmation below |
| Response range, direction, clean-error cost and eventual violation of the physical envelope | `kf_reached_response_20260917a`; retrospective analysis with sampled ground-truth qualification |
| Exposure versus temporal gradients | `kf_unroll_pair_20260918a`; matched detached/full K=4 |
| Recovery versus faithful dynamics targets | `kf_target_pair_20260918a`; identical displaced inputs, different targets and information costs |
| Next-state refinement and conditional generation | `kf_refiner_20260919c/evaluation`, `kf_acdm_20260920b/evaluation_v2`; fixed-PCNO adaptations with recorded acquisition and physical-fidelity limits |
| Response prior, amplitude cap and architecture restriction | `kf_remaining_coverage_20260921a`, `kf_spectral_control_20260921b`; selected components, not whole-family verdicts |
| Response-preserving design, forecasts and fitting-order repeats | `kf_bias_design_20260922b`, `kf_bias_forecast_20260922a`, `kf_bias_replication_20260922a`; three adaptation pairs from one parent |
| Independent population and frozen prospective controls | `kf_confirmation64_20260923c`; nine frozen maps and 64 evaluation-only trajectories |

The twelve-map representative comparison is development evidence. Independent
confirmation covers the selected nine central/design maps, not all twelve.
Training data and fitted models were held fixed for that confirmation.

DySLIM and Thermalizer remain conceptual representatives in the mechanism
discussion. The transient training pool does not support an invariant-law
ranking; no stationary-data study was conducted. ODEs provide controlled
illustrations, NACA provides problem-specific projection evidence, and Bump
provides a contrasting PDE case. Select supporting results through the
[paper plan](../../paper/PAPER_PLAN.md) and exact records in the experiment tracker.

## Interpretation And Use

The main study concerns autonomous rollouts from in-distribution initial
conditions under fixed, sufficiently rich data. Solver relabeling adds target
information and must be distinguished from recovery training. Component
adaptations do not establish a universal ranking of the original methods.
Retrospective diagnostics and forecasts frozen before evaluation retain their
separate roles; deployment does not assume access to a trusted solver.

Closed additive-bias, blend and prefix-switch attempts remain in the tracker.
They do not authorize new strengths or schedules. Historical protected
populations remain closed, and the fresh confirmation does not open them.
[PROJECT_PLAN.md](PROJECT_PLAN.md) owns completion criteria;
[HANDOFF.md](HANDOFF.md) owns the current action.

## Historical Recipes

The [completed-study plan snapshot](history/EXPERIMENT_PLAN_completed_2026-09-28.md)
preserves the preceding 832-line plan byte-for-byte, including recipes,
contracts, expectations and then-current future instructions. Those instructions
are historical evidence. Resolve its relative links against the original
`docs/time_dependent_no/` directory. Its 60,386 bytes have SHA-256
`5a28af8c65d903240b450a2e4c76ed8ab1ba3acf6ea9ea5e428e6e1d45b1b8f8`.
