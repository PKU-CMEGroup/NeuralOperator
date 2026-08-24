#!/bin/bash
#SBATCH -o drivaerml_preprocess_%j.out
#SBATCH --qos=low
#SBATCH -p C064M0256G
#SBATCH -J drivaerml_preprocess
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=100:00:00

set -euo pipefail

source ~/.bashrc
conda activate "${CONDA_ENV:-myconda}"

# 无论从仓库根目录还是 scripts/drivaerml 目录提交，都切换到仓库根目录运行。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON:-python}"
LOG_DIR="${LOG_DIR:-log}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/drivaerml_preprocess.log}"

mkdir -p "${LOG_DIR}" data/drivaerml/preprocess

echo "Preprocessing data/drivaerml/boundary_*.vtp"
echo "Log: ${LOG_FILE}"

"${PYTHON_BIN}" scripts/drivaerml/preprocess.py 2>&1 | tee "${LOG_FILE}"
