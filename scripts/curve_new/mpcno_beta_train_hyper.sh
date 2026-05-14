#!/bin/bash
#SBATCH -o out/MPCNO_hyper_%A.out
#SBATCH --qos=low
#SBATCH -J MPCNO_hyper
#SBATCH -p GPU40G
#SBATCH --nodes=1 
#SBATCH --ntasks=6
#SBATCH --gres=gpu:1
#SBATCH --time=100:00:00
#SBATCH --array=0-3 

# ========== params ==========
KERNEL_TYPE="fourier_param-10"

GRAD="True"
GEO="True"
GEOINTEGRAL="False"

N_TRAIN=8000
N_TEST=1000
N_TWO_CIRCLES_TEST=1000

TO_DIVIDE_FACTOR=20.0
BATCH_SIZE=8
LAYERS=(256 256)
ACT="none"
GEO_ACT='softsign'
PROJ_ACT='gelu'

# 新增 beta 范围参数
BETA_LOW=1.0
BETA_HIGH=1.0

K_MAX_VALUES=(4 8 16 32)
K_MAX_COUNT=${#K_MAX_VALUES[@]}

INDEX=$SLURM_ARRAY_TASK_ID
K_MAX=${K_MAX_VALUES[$INDEX]}
# =============================


LAYER_SIZES_STR=$(IFS=,; echo "${LAYERS[*]}")
PROJ_LAYER_SIZES_STR=$(IFS=,; echo "${PROJ_LAYERS[*]}")
LOG_DIR="log/beta_MPCNO_hyper/${KERNEL_TYPE}_beta${BETA_LOW}-${BETA_HIGH}/${LAYER_SIZES_STR}_${ACT}_${N_TRAIN}/${PROJ_LAYER_SIZES_STR}_${PROJ_ACT}"
mkdir -p ${LOG_DIR}

source activate pytorch 

python mpcno_curve_beta_test_hyper.py \
    --kernel_type $KERNEL_TYPE \
    --geointegral $GEOINTEGRAL \
    --n_train $N_TRAIN \
    --n_test $N_TEST \
    --n_two_circles_test $N_TWO_CIRCLES_TEST \
    --to_divide_factor $TO_DIVIDE_FACTOR \
    --k_max $K_MAX \
    --layer_sizes $LAYER_SIZES_STR \
    --act $ACT \
    --bsz $BATCH_SIZE \
    --data_path "mpcno_curve_data_1.0_1.0_5_beta(1.0, 1.0)_2d_fourier_param_k_max_8_beta_random_panel_single.npz" \
    --two_circles_data_path "mpcno_curve_data_1.0_1.0_5_beta(1.0, 1.0)_2d_fourier_param_k_max_8_beta_random_panel_two_curves.npz" \
    > ${LOG_DIR}/k${K_MAX}_L10_bsz${BATCH_SIZE}_factor${TO_DIVIDE_FACTOR}_grad${GRAD}_geo${GEO}_geoint${GEOINTEGRAL}_beta${BETA_LOW}-${BETA_HIGH}.log
