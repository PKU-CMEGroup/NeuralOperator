#!/bin/bash
#SBATCH -o out/PCNO_train_log_%A_%a.out
#SBATCH --qos=low
#SBATCH -J PCNO_train
#SBATCH -p GPU40G
#SBATCH --nodes=1 
#SBATCH --ntasks=6
#SBATCH --gres=gpu:1
#SBATCH --time=100:00:00
#SBATCH --array=0-3  # 4个任务对应4个K_MAX值

# ========== params ==========
KERNEL_TYPE="stokes" # "sp_laplace" or "dp_laplace" or "modified_dp_laplace" or "adjoint_dp_laplace" or "stokes"

GRAD="True"
GEO="True"
GEOINTEGRAL="True"

N_TRAIN=8000
N_TEST=1000
N_TWO_CIRCLES_TEST=1000

TO_DIVIDE_FACTOR=20.0
BATCH_SIZE=8
LAYERS=(64 64 64 64 64 64)
ACT="gelu"

K_MAX_VALUES=(8 16 32 64)
NORMAL_PROD="False"
NUM_GRAD=3 # 1 或 3
# =============================

# 根据数组索引选择K_MAX
INDEX=$SLURM_ARRAY_TASK_ID
K_MAX=${K_MAX_VALUES[$INDEX]}

LAYER_SIZES_STR=$(IFS=,; echo "${LAYERS[*]}")
LOG_DIR="log/1_1_5_2d_${KERNEL_TYPE}/${NUM_GRAD}_${LAYER_SIZES_STR}_${ACT}/"
mkdir -p ${LOG_DIR}

source activate pytorch 

python pcno_curve_geo_test.py \
    --grad $GRAD \
    --geo $GEO \
    --geointegral $GEOINTEGRAL \
    --n_train $N_TRAIN \
    --n_test $N_TEST \
    --n_two_circles_test $N_TWO_CIRCLES_TEST \
    --to_divide_factor $TO_DIVIDE_FACTOR \
    --kernel_type $KERNEL_TYPE \
    --k_max $K_MAX \
    --normal_prod $NORMAL_PROD \
    --num_grad $NUM_GRAD \
    --layer_sizes $LAYER_SIZES_STR \
    --act $ACT \
    --bsz $BATCH_SIZE \
    > ${LOG_DIR}/N${N_TRAIN}_Ntest${N_TEST},${N_TWO_CIRCLES_TEST}_k${K_MAX}_L10_bsz${BATCH_SIZE}_factor${TO_DIVIDE_FACTOR}_grad${GRAD}_geo${GEO}_geoint${GEOINTEGRAL}.log