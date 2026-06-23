# Failure Diagnostics

Aggregate rollout RMSE is not enough for hard 2D Euler benchmarks. Each baseline readout should identify which physical or numerical structure is broken.

## Required Diagnostic Families

- Rollout error: RMSE, relative L2, valid prediction time.
- Shock diagnostics: graph-gradient shock proxy, near-shock versus smooth-region error, shock centroid distance, smearing/thickness proxy, strength ratio.
- Conservation diagnostics: total mass, momentum, and energy drift; prefer mesh weights when available.
- Positivity diagnostics: minimum density and pressure, nonpositive counts and fractions.
- Boundary diagnostics: boundary-local error and leakage under clamped-boundary versus free/prescribed-boundary rollout.
- Geometry/regime diagnostics: error by Mach and geometry parameter bins when metadata is available.

## Method Gate

Do not add a new method family until baseline diagnostics name one dominant defect:

| Dominant defect | First method direction | Required control |
| --- | --- | --- |
| conservation drift | conservative flux aggregation or flux residual | raw state predictor |
| positivity violation | positivity-preserving output or projection | post-hoc clipping |
| shock smearing | learned reconstruction / flux-aware update | multistep/noise-trained baseline |
| shock displacement | shock-aware loss or DA shock correction | interpolation and scalar shift/gain |
| boundary leakage | boundary-conditioned flux or residual head | boundary-mask baseline |
| rollout compounding | multistep/noise training or analysis cadence | one-step equal-capacity baseline |

## Reporting Template

Use this structure for experiment readouts:

```text
Question -> Hypothesis -> Diagnostic -> Result -> Interpretation -> Next question
```
