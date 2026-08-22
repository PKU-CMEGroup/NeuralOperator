#!/bin/bash
#SBATCH -o drivaerml_pcfno_train_%j.out
#SBATCH --qos=low
#SBATCH -J drivaerml_pcfno
#SBATCH -p GPU80G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --time=100:00:00

set -euo pipefail

source ~/.bashrc
conda activate "${CONDA_ENV:-myconda}"

# 统一从仓库根目录运行，使数据、日志和模型路径不依赖 sbatch 的提交位置。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON:-python}"
DATA_DIR="${DATA_DIR:-data/drivaerml}"
PREPROCESS_DIR="${PREPROCESS_DIR:-${DATA_DIR}/preprocess}"
LOG_DIR="${LOG_DIR:-log}"
MODEL_DIR="${MODEL_DIR:-model}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/pcfno_train.log}"
MODEL_NAME="${MODEL_NAME:-${MODEL_DIR}/pcfno_drivaerml}"

mkdir -p "${LOG_DIR}" "${MODEL_DIR}"

echo "Training PCFNO"
echo "Data: ${DATA_DIR}"
echo "Preprocessed arrays: ${PREPROCESS_DIR}"
echo "Model: ${MODEL_NAME}"
echo "Log: ${LOG_FILE}"

"${PYTHON_BIN}" scripts/drivaerml/pcfno_train.py \
    --data_dir "${DATA_DIR}" \
    --preprocess_dir "${PREPROCESS_DIR}" \
    --y_fields CpMeanTrim,wallShearStressMeanTrim \
    --statistics_chunk_size 200000 \
    --train_sample_size 16384 \
    --test_sample_size 16384 \
    --sample_weight_correction measure \
    --n_train 400 \
    --n_test 80 \
    --num_workers 4 \
    --geointegral \
    --k_max 16 \
    --layer_sizes 64,64,64,64 \
    --fc_dim 128 \
    --bsz 1 \
    --ep 200 \
    --base_lr 5e-4 \
    --weight_decay 1e-4 \
    --to_divide_factor 20 \
    --save_every 100 \
    --model_name "${MODEL_NAME}" \
    2>&1 | tee "${LOG_FILE}"
