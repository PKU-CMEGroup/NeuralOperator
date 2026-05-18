#!/bin/bash
#SBATCH -o out/PCNO_train_1layer_log_%A.out 
#SBATCH --qos=low
#SBATCH -J linear
#SBATCH -p GPU80G
#SBATCH --nodes=1 
#SBATCH --ntasks=6
#SBATCH --gres=gpu:1
#SBATCH --time=100:00:00
#SBATCH --array=0-3

# ========== params ==========
KERNEL_TYPE="dp_laplace"

GRAD="True"
GEO="True"
GEOINTEGRAL="True"

N_TRAIN=8000
N_TEST=1000
N_TWO_CIRCLES_TEST=1000

TO_DIVIDE_FACTOR=20.0
BATCH_SIZE=8
LAYERS=(64 64)
ACT="none"
GEO_ACT='softsign'

# 新增 beta 范围参数
BETA_LOW=0.5
BETA_HIGH=1.0

K_MAX_VALUES=(8 16 32 64)
K_MAX_COUNT=${#K_MAX_VALUES[@]}

INDEX=$SLURM_ARRAY_TASK_ID
K_MAX=${K_MAX_VALUES[$INDEX]}
# =============================


LAYER_SIZES_STR=$(IFS=,; echo "${LAYERS[*]}")
LOG_DIR="log_linear/1_1_5_2d_${KERNEL_TYPE}/beta${BETA_LOW}-${BETA_HIGH}/${LAYER_SIZES_STR}_${ACT}/"
mkdir -p ${LOG_DIR}

source activate pytorch 

python mpcno_curve_beta_linear_test.py \
    --kernel_type $KERNEL_TYPE \
    --grad $GRAD \
    --geo $GEO \
    --geointegral $GEOINTEGRAL \
    --n_train $N_TRAIN \
    --n_test $N_TEST \
    --n_two_circles_test $N_TWO_CIRCLES_TEST \
    --to_divide_factor $TO_DIVIDE_FACTOR \
    --k_max $K_MAX \
    --layer_sizes $LAYER_SIZES_STR \
    --act $ACT \
    --geo_act $GEO_ACT \
    --bsz $BATCH_SIZE \
    --beta_low $BETA_LOW \
    --beta_high $BETA_HIGH \
    > ${LOG_DIR}/k${K_MAX}_L10_bsz${BATCH_SIZE}_factor${TO_DIVIDE_FACTOR}_grad${GRAD}_geo${GEO}_geoint${GEOINTEGRAL}_beta${BETA_LOW}-${BETA_HIGH}.log
