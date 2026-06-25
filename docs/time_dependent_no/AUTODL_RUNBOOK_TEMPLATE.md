# AutoDL Runbook Template

This committed file is intentionally generic. Put real hostnames, SSH commands, credentials, dataset paths, and user-specific environment details in untracked `LOCAL_CONTEXT.md` or `docs/time_dependent_no/PRIVATE_AUTODL.md`.

## Local CPU Sanity

```bash
python -m pytest tests/time_dependent_no
python scripts/time_dependent_no/run_euler_fixture_diagnostics.py
```

## AutoDL Environment Notes

The branch utilities are NumPy-first. HDF5 inspection requires `h5py`:

```bash
python -m pip install h5py
```

Prefer installing dependencies in a project environment for training runs. Installing `h5py` into the base environment is acceptable only for quick schema inspection.

## Dataset Schema Inspection

Replace `/path/to/train.h5` with the private AutoDL dataset path from local context.

```bash
python scripts/time_dependent_no/inspect_cpg_euler_dataset.py /path/to/train.h5 \
  --max-trajectories 2 \
  --metadata-only \
  --output artifacts/time_dependent_no/train_schema_metadata.json
```

A copied folder labeled `forward_300` was inspected on 2026-06-25 and appears to be the supersonic bump dataset. See `docs/time_dependent_no/BUMP_300_DATASET_AUDIT.md`.

## Artifact Rules

- Save generated JSON, figures, and small summaries under `artifacts/time_dependent_no/`.
- Do not commit raw datasets, checkpoints, rollout arrays, extracted VTU files, or large logs.
- Commit only cleaned reports or small schema summaries when they are useful for collaboration and contain no private paths.
