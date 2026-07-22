#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  printf 'usage: %s STRIDE SEED DATASET OUTPUT_ROOT\n' "$0" >&2
  exit 2
fi

stride="$1"
seed="$2"
dataset="$3"
output_root="$4"

case "$stride" in
  1|2|4|8) ;;
  *)
    printf 'unsupported frontier stride: %s\n' "$stride" >&2
    exit 2
    ;;
esac

if [[ ! "$seed" =~ ^[0-9]+$ ]]; then
  printf 'seed must be a nonnegative integer\n' >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

python_bin="${PYTHON_BIN:-python}"
if ! command -v "$python_bin" >/dev/null 2>&1; then
  printf 'Python is not available: %s\n' "$python_bin" >&2
  exit 2
fi
if [[ ! -f "$dataset" ]]; then
  printf 'dataset does not exist: %s\n' "$dataset" >&2
  exit 2
fi

run_dir="${output_root}/s${stride}_seed${seed}"
log_dir="${output_root}/logs"
mkdir -p "$run_dir" "$log_dir"
if [[ -e "${run_dir}/RUNNING" || -e "${run_dir}/SUCCESS" ]]; then
  printf 'run directory is already active or complete: %s\n' "$run_dir" >&2
  exit 3
fi

manifest="${run_dir}/launch_manifest.json"
DATASET="$dataset" RUN_DIR="$run_dir" STRIDE="$stride" SEED="$seed" \
  "$python_bin" - "$manifest" <<'PY'
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


repo = Path.cwd()
dataset = Path(os.environ["DATASET"])
source_paths = [
    Path("scripts/time_dependent_no/train_euler1d_target_ladder.py"),
    Path("scripts/time_dependent_no/evaluate_euler1d_flow_map_frontier.py"),
    Path("scripts/time_dependent_no/launch_euler1d_large_step_frontier.sh"),
    Path("baselines/fno.py"),
    Path("utility/time_dependent_no/euler1d.py"),
    Path("utility/time_dependent_no/euler1d_data.py"),
    Path("utility/time_dependent_no/euler1d_models.py"),
    Path("utility/time_dependent_no/euler1d_targets.py"),
]
payload = {
    "dataset": str(dataset),
    "dataset_sha256": sha256(dataset),
    "git_head": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip(),
    "git_status": subprocess.check_output(
        ["git", "status", "--short"], text=True
    ).splitlines(),
    "hostname": platform.node(),
    "python": sys.version,
    "run_dir": os.environ["RUN_DIR"],
    "seed": int(os.environ["SEED"]),
    "source_sha256": {
        str(path): sha256(repo / path)
        for path in source_paths
    },
    "stride": int(os.environ["STRIDE"]),
    "training_contract": {
        "architecture": "FNO residual 64/24/4 fc128 pad0",
        "batch_size": 8,
        "checkpoint_selection_frame": 32,
        "one_step_presentations": 1_920_000,
        "one_step_validation_milestones": 50,
        "recurrent_call_depth": 4,
        "recurrent_window_presentations": 372_480,
        "recurrent_validation_milestones": 10,
        "split": [384, 64, 64],
        "split_seed": 20260707,
    },
}
Path(sys.argv[1]).write_text(
    json.dumps(payload, indent=2, sort_keys=True),
    encoding="utf-8",
)
PY

printf '%s\n' "$$" > "${run_dir}/RUNNING"
log_path="${log_dir}/s${stride}_seed${seed}.log"
set +e
PYTHONPATH="$repo_root" CUDA_VISIBLE_DEVICES=0 "$python_bin" \
  scripts/time_dependent_no/train_euler1d_target_ladder.py \
  --data-path "$dataset" \
  --output-dir "$run_dir" \
  --model fno \
  --target residual \
  --epochs 50 \
  --one-step-presentations 1920000 \
  --unroll-epochs 10 \
  --unroll-window-presentations 372480 \
  --unroll-steps 4 \
  --unroll-lr-factor 0.1 \
  --batch-size 8 \
  --lr 0.0003 \
  --weight-decay 0.00001 \
  --grad-clip 1 \
  --seed "$seed" \
  --split-seed 20260707 \
  --train-cases 384 \
  --val-cases 64 \
  --test-cases 64 \
  --step-stride "$stride" \
  --rollout-final-frame 32 \
  --input-coordinates conservative \
  --loss-coordinates conservative \
  --recurrent-coordinates conservative \
  --input-normalization fixed_physical \
  --loss-normalization fixed_physical \
  --target-supervision state \
  --positive-transform none \
  --input-noise-std 0 \
  --initial-frame-weight 1 \
  --fno-width 64 \
  --fno-modes 24 \
  --fno-layers 4 \
  --fno-fc-dim 128 \
  --fno-pad-ratio 0 \
  --device cuda \
  --gpu 0 \
  --torch-threads 1 \
  --save-checkpoints \
  --save-candidate-checkpoints \
  --fail-fast 2>&1 | tee "$log_path"
status=${PIPESTATUS[0]}
set -e

rm -f "${run_dir}/RUNNING"
if [[ $status -eq 0 ]]; then
  printf 'ok\n' > "${run_dir}/SUCCESS"
else
  printf '%s\n' "$status" > "${run_dir}/FAILED"
fi
exit "$status"
