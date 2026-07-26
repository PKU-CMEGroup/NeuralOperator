# cpgGNSpdes Reference Audit

Status: Frozen provenance for the completed bounded adaptation

The reference repository was inspected locally before this branch was created.
This page records that bounded source audit; it is not an implementation queue.
Keep the checkout path in private `LOCAL_CONTEXT.md`, not in committed
documentation.

Reference state at prior inspection:

```text
main...origin/main
HEAD b127e5e add codes
```

## Summary

The reference repo is a compact research-code release, not a turn-key benchmark package. It defines the HDF5 schema, boundary convention, training loop, rollout behavior, and structure-preserving model idea, but the dataset is not included in the clone.

The public code was used as implementation evidence for the branch's completed
bounded CPG adaptation. It is not proof of an unreleased paper dataset or
checkpoint contract and must not be vendored wholesale.

## Core Mechanism Captured In The Bounded Adaptation

The main model family is `modelEdgeUpd.Simulator`:

1. Convert primitive variables `[rho, v1, v2, pres]` to conservative variables.
2. Normalize solution and Mach features.
3. Encode geometry from coordinates and one-hot node type.
4. Run edge-enriched message passing.
5. Decode positive primitive interface states on directed edges.
6. Evaluate a Local Lax-Friedrichs / Rusanov flux from paired reconstructed states.
7. Aggregate a graph finite-volume-style conservative update.
8. Convert updated conservative variables back to primitive variables.

This is the paper-relevant idea: learn interface reconstruction and use a Riemann-flux update, instead of directly predicting next node states.

## Reproduction Risks

- The dataset path `dataset/data_downsampled/train.h5` and `test.h5` is absent from the clone.
- Training scripts call `/usr/bin/shutdown` after training and call `.cuda()` directly.
- Dataset paths, checkpoint paths, epochs, noise scales, and device behavior are hard-coded.
- Several model files hard-code `DELTA_T = 0.025`, which does not match all paper datasets.
- Boundary nodes are clamped to target states during training and rollout, so released rollout does not test free boundary maintenance.
- Conservation is finite-volume-style but not automatically exact if directed learned edge factors are asymmetric.
- Positivity is only partially enforced: decoded edge density/pressure are positive, but updated node density/pressure can still violate positivity.
- The public reader does not obviously expose physical cell areas, edge lengths, face normals, or cell volumes.

## Frozen Branch Policy

Do not run the reference training scripts as-is on AutoDL, HPC, or shared
machines. The required bounded components were reimplemented in this branch
with explicit configuration, safe device handling, and diagnostic logging.
Further reproduction or expansion requires a new authorization; this frozen
audit does not authorize another port, training run, or mechanism sweep.

