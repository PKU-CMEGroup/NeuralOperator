#!/bin/bash
#SBATCH -o out/MPCNO_nograd_%A.out
#SBATCH --qos=high
#SBATCH -J MPCNO_nograd
#SBATCH -p GPU80G
#SBATCH --nodes=1 
#SBATCH --ntasks=6
#SBATCH --gres=gpu:1
#SBATCH --time=100:00:00
#SBATCH --array=0-3 

# ========== params ==========
KERNEL_TYPE="fourier_twoparam-8"

GRAD="False"
GEO="False"
GEOINTEGRAL="False"

N_TRAIN=8000
N_TEST=1000
N_TWO_CIRCLES_TEST=1000

TO_DIVIDE_FACTOR=20.0
BATCH_SIZE=8
LAYERS=(256 256)
PROJ_LAYERS=(128 128 128)
ACT="none"
GEO_ACT='softsign'
PROJ_ACT='gelu'

# 新增 beta 范围参数
BETA_LOW=0.5
BETA_HIGH=1.0

K_MAX_VALUES=(4 8 16 32)
K_MAX_COUNT=${#K_MAX_VALUES[@]}

INDEX=$SLURM_ARRAY_TASK_ID
K_MAX=${K_MAX_VALUES[$INDEX]}
# =============================


LAYER_SIZES_STR=$(IFS=,; echo "${LAYERS[*]}")
PROJ_LAYER_SIZES_STR=$(IFS=,; echo "${PROJ_LAYERS[*]}")
LOG_DIR="log/beta_MPCNO_nograd/${KERNEL_TYPE}_beta${BETA_LOW}-${BETA_HIGH}/${LAYER_SIZES_STR}_${ACT}_${N_TRAIN}/${PROJ_LAYER_SIZES_STR}_${PROJ_ACT}"
mkdir -p ${LOG_DIR}

source activate pytorch 

python mpcno_curve_beta_test_nograd.py \
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
    --proj_layer_sizes $PROJ_LAYER_SIZES_STR \
    --act $ACT \
    --geo_act $GEO_ACT \
    --proj_act $PROJ_ACT \
    --bsz $BATCH_SIZE \
    --data_path "mpcno_curve_data_1.0_1.0_5_beta(0.5, 1.0)_2d_fourier_twoparam_k_max_8_beta_random_panel_single.npz" \
    --two_circles_data_path "mpcno_curve_data_1.0_1.0_5_beta(0.5, 1.0)_2d_fourier_twoparam_k_max_8_beta_random_panel_two_curves.npz" \
    > ${LOG_DIR}/k${K_MAX}_L10_bsz${BATCH_SIZE}_factor${TO_DIVIDE_FACTOR}_grad${GRAD}_geo${GEO}_geoint${GEOINTEGRAL}_beta${BETA_LOW}-${BETA_HIGH}.log
