#!/bin/bash
#SBATCH -o MPCNO_hifi3d_preprocess_%A_%a.out
#SBATCH --qos=low
#SBATCH -p C064M1024G
#SBATCH -J MPCNO_hifi3d_preprocess
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --array=0-5
#SBATCH --time=100:00:00

set -euo pipefail

# Preprocess HiFi3D VTP meshes for PCNO/M-PCNO.
#
# Default behavior under Slurm:
#   sbatch scripts/hifi3d/mpcno_preprocess_data.sh
# runs six array tasks, one per dataset, and writes one full cache per dataset.
#
# Useful overrides:
#   REPO_ROOT=/lustre/home/2200010815/neuralop/NeuralOperator
#   DATA_ROOT=${REPO_ROOT}/data/HiFi3D
#   OUTPUT_DIR=${REPO_ROOT}/data/hifi3d_processed/cache
#   MESH_TYPE=cell_centered       # cell_centered or vertex_centered
#   ADJACENT_TYPE=edge            # node, edge, or face
#   N_EACH=0                      # 0 means full dataset
#   SEED=0
#   CONDA_ENV=geometry
#   PYTHON=/lustre/home/2200010815/software/miniconda3/envs/geometry/bin/python
#   DATASET=BlendedNet            # process only one dataset

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO_ROOT="/lustre/home/2200010815/neuralop/NeuralOperator"
DEFAULT_PYTHON="/lustre/home/2200010815/software/miniconda3/envs/geometry/bin/python"

if [[ -n "${REPO_ROOT:-}" ]]; then
    REPO_ROOT="$(cd "${REPO_ROOT}" && pwd)"
elif [[ -f "${DEFAULT_REPO_ROOT}/scripts/hifi3d/preprocess_hifi3d.py" ]]; then
    REPO_ROOT="${DEFAULT_REPO_ROOT}"
elif [[ -f "${SUBMIT_DIR}/scripts/hifi3d/preprocess_hifi3d.py" ]]; then
    REPO_ROOT="$(cd "${SUBMIT_DIR}" && pwd)"
elif [[ -f "${SCRIPT_DIR}/preprocess_hifi3d.py" ]]; then
    REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
else
    echo "Error: could not infer NeuralOperator repository root." >&2
    echo "Submit from the repository root or set REPO_ROOT=/path/to/NeuralOperator." >&2
    exit 2
fi
cd "${REPO_ROOT}"

DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data/HiFi3D}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/data/hifi3d_processed/cache}"
MESH_TYPE="${MESH_TYPE:-cell_centered}"
ADJACENT_TYPE="${ADJACENT_TYPE:-edge}"
N_EACH="${N_EACH:-0}"
SEED="${SEED:-0}"
CONDA_ENV="${CONDA_ENV:-geometry}"
PYTHON_BIN="${PYTHON:-${DEFAULT_PYTHON}}"

DATASETS=(
    "AirCraft"
    "BlendedNet"
    "DrivAerML"
    "DrivAerNet++"
    "DrivAerStar"
    "NACA-CRM"
)

load_conda_env() {
    if [[ -x "${PYTHON_BIN}" ]]; then
        return 0
    fi

    if command -v module >/dev/null 2>&1; then
        module load conda || true
    fi

    if command -v conda >/dev/null 2>&1; then
        # shellcheck disable=SC1091
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate "${CONDA_ENV}"
    elif command -v source >/dev/null 2>&1; then
        source activate "${CONDA_ENV}"
    fi

    if [[ ! -x "${PYTHON_BIN}" ]]; then
        PYTHON_BIN="$(command -v python)"
    fi
}

validate_settings() {
    if [[ "${MESH_TYPE}" != "cell_centered" && "${MESH_TYPE}" != "vertex_centered" ]]; then
        echo "Error: MESH_TYPE must be cell_centered or vertex_centered, got ${MESH_TYPE}" >&2
        exit 2
    fi

    case "${ADJACENT_TYPE}" in
        node|edge|face) ;;
        *)
            echo "Error: ADJACENT_TYPE must be node, edge, or face, got ${ADJACENT_TYPE}" >&2
            exit 2
            ;;
    esac

    if [[ ! "${N_EACH}" =~ ^[0-9]+$ ]]; then
        echo "Error: N_EACH must be a non-negative integer, got ${N_EACH}" >&2
        exit 2
    fi

    if [[ ! -d "${DATA_ROOT}" ]]; then
        echo "Error: DATA_ROOT not found: ${DATA_ROOT}" >&2
        exit 2
    fi

    if [[ ! -x "${PYTHON_BIN}" ]]; then
        echo "Error: Python executable not found or not executable: ${PYTHON_BIN}" >&2
        echo "Set PYTHON=/path/to/python or CONDA_ENV=${CONDA_ENV}." >&2
        exit 2
    fi
}

process_dataset() {
    local dataset="$1"
    local mesh_dir="${DATA_ROOT}/${dataset}_20000"
    local file_count

    if [[ ! -d "${mesh_dir}" ]]; then
        echo "Error: mesh directory not found: ${mesh_dir}" >&2
        return 2
    fi

    file_count="$(find "${mesh_dir}" -maxdepth 1 -type f -name "*.vtp" | wc -l | tr -d " ")"
    if [[ "${file_count}" -eq 0 ]]; then
        echo "Error: no .vtp files found under ${mesh_dir}" >&2
        return 2
    fi

    mkdir -p "${OUTPUT_DIR}"

    echo "============================================================"
    echo "HiFi3D MPCNO preprocessing"
    echo "Repository: ${REPO_ROOT}"
    echo "Data root: ${DATA_ROOT}"
    echo "Dataset: ${dataset}"
    echo "Input dir: ${mesh_dir}"
    echo "Input files: ${file_count}"
    echo "Output dir: ${OUTPUT_DIR}"
    echo "Output name: ${dataset}_full"
    echo "Mesh type: ${MESH_TYPE}"
    echo "Adjacent type: ${ADJACENT_TYPE}"
    echo "N_EACH: ${N_EACH} (0 means full dataset)"
    echo "Seed: ${SEED}"
    echo "Conda env: ${CONDA_ENV}"
    echo "Python: ${PYTHON_BIN}"
    echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================================"

    "${PYTHON_BIN}" "${REPO_ROOT}/scripts/hifi3d/preprocess_hifi3d.py" \
        --data_root "${DATA_ROOT}" \
        --datasets "${dataset}" \
        --n_each "${N_EACH}" \
        --seed "${SEED}" \
        --output_dir "${OUTPUT_DIR}" \
        --output_name "${dataset}_full" \
        --mesh_type "${MESH_TYPE}" \
        --adjacent_type "${ADJACENT_TYPE}"

    echo "Finished ${dataset}: $(date '+%Y-%m-%d %H:%M:%S')"
}

main() {
    load_conda_env
    validate_settings

    export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
    export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
    export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

    if [[ -n "${DATASET:-}" ]]; then
        if [[ -n "${SLURM_ARRAY_TASK_ID:-}" && "${SLURM_ARRAY_TASK_ID}" != "0" ]]; then
            echo "DATASET=${DATASET} is set; skipping duplicate array task ${SLURM_ARRAY_TASK_ID}."
            exit 0
        fi
        process_dataset "${DATASET}"
        exit 0
    fi

    if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
        if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= ${#DATASETS[@]} )); then
            echo "Error: SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} is outside 0-$((${#DATASETS[@]} - 1))" >&2
            exit 2
        fi
        process_dataset "${DATASETS[${SLURM_ARRAY_TASK_ID}]}"
    else
        for dataset in "${DATASETS[@]}"; do
            process_dataset "${dataset}"
        done
    fi
}

main "$@"
