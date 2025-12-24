#!/bin/bash
#SBATCH -o out/PCNO_train_log.out
#SBATCH --qos=low
#SBATCH -J PCNO_train
#SBATCH -p GPU40G
#SBATCH --nodes=1 
#SBATCH --ntasks=6
#SBATCH --gres=gpu:1
#SBATCH --time=100:00:00

# ========== params ==========
KERNEL_TYPE="dp_laplace"

GRAD="True"
GEO="True"
GEOINTEGRAL="False"

N_TRAIN=2000
N_TEST=1000
N_TWO_CIRCLES_TEST=1000

BATCH_SIZE=32
LAYERS=(128 128 128 128 128)
ACT="gelu"

K_MAX=16
NORMAL_PROD="True"
NUM_GRAD=3
# =============================


LAYER_SIZES_STR=$(IFS=,; echo "${LAYERS[*]}")
LOG_DIR="log/1_1_5_2d_${KERNEL_TYPE}/${LAYER_SIZES_STR}_${ACT}/"
mkdir -p ${LOG_DIR}

source activate pytorch 

python pcno_curve_geo_test.py \
    --grad $GRAD \
    --geo $GEO \
    --geointegral $GEOINTEGRAL \
    --n_train $N_TRAIN \
    --n_test $N_TEST \
    --n_two_circles_test $N_TWO_CIRCLES_TEST \
    --kernel_type $KERNEL_TYPE \
    --k_max $K_MAX \
    --normal_prod $NORMAL_PROD \
    --num_grad $NUM_GRAD \
    --layer_sizes $LAYER_SIZES_STR \
    --act $ACT \
    --bsz $BATCH_SIZE \
    > ${LOG_DIR}/k${K_MAX}_L10_bsz${BATCH_SIZE}_grad${GRAD}_geo${GEO}_geoint${GEOINTEGRAL}.log
