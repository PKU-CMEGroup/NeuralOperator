# Handoff

Use this page when transferring work between human collaborators or AI agents.

## Current Objective

Establish the CPG Euler benchmark pipeline inside PKU-CME `NeuralOperator` on branch `time-dependent-no`.

## Current Data State

The first copied HDF5 dataset has been inspected. The folder label is `forward_300`, but extracted case files indicate it is the supersonic bump dataset. The exact AutoDL path is in ignored `LOCAL_CONTEXT.md`; committed schema facts are in `docs/time_dependent_no/BUMP_300_DATASET_AUDIT.md`.

## Do Next

1. Add a real-data smoke script that loads one trajectory/frame from `train.h5` through `make_cpg_graph_frame` and prints shape/range summaries.
2. Decide the first adapter path: PCNO/MPCNO native graph/point-cloud path versus FNO remeshing sanity.
3. If starting PCNO/MPCNO, audit the existing `pcno`/`mpcno` expected inputs against CPG fields.
4. If starting FNO, first implement remesh/inverse-remesh sanity and show preprocessing error is smaller than model error.

## Do Not Do Yet

- Do not run raw `cpggnspdes` training scripts.
- Do not launch large training before a no-training real-data adapter smoke passes.
- Do not push or merge to main without human approval.
- Do not commit private paths, credentials, raw data, checkpoints, extracted VTU files, or generated heavy artifacts.
