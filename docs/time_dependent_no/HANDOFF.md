# Handoff

Use this page when transferring work between human collaborators or AI agents.

## Current Objective

Establish the CPG Euler benchmark pipeline inside PKU-CME `NeuralOperator` on branch `time-dependent-no`.

## Do Next

1. Run CPU tests for `tests/time_dependent_no`.
2. Fill private `LOCAL_CONTEXT.md` with the AutoDL dataset root.
3. Inspect `train.h5` and `test.h5` on AutoDL.
4. Decide the first adapter: PCNO/MPCNO native point-cloud path or FNO remeshing sanity.

## Do Not Do Yet

- Do not run raw `cpggnspdes` training scripts.
- Do not launch large training before schema inspection is saved.
- Do not push or merge to main without human approval.
- Do not commit private paths, credentials, raw data, checkpoints, or generated heavy artifacts.
