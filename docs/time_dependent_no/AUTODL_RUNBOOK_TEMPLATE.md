# AutoDL Runbook Template

This committed file is intentionally generic. Put real hostnames, SSH commands, credentials, dataset paths, and user-specific environment details in untracked `LOCAL_CONTEXT.md` or `docs/time_dependent_no/PRIVATE_AUTODL.md`.

## Local CPU Sanity

```bash
python -m pytest tests/time_dependent_no
python scripts/time_dependent_no/run_euler_fixture_diagnostics.py
```

## Dataset Schema Inspection

Replace `/path/to/train.h5` with the private AutoDL dataset path from local context.

```bash
python scripts/time_dependent_no/inspect_cpg_euler_dataset.py /path/to/train.h5 \
  --max-trajectories 2 \
  --output artifacts/time_dependent_no/train_schema.json
```

## Artifact Rules

- Save generated JSON, figures, and small summaries under `artifacts/time_dependent_no/`.
- Do not commit raw datasets, checkpoints, rollout arrays, or large logs.
- Commit only cleaned reports or small schema summaries when they are useful for collaboration and contain no private paths.
