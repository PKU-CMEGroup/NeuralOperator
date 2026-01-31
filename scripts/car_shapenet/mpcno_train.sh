#!/bin/bash
#SBATCH -o MPCNO_train.out
#SBATCH --qos=low
#SBATCH -J MPCNO_train
#SBATCH -p GPU80G
#SBATCH --nodes=1 
#SBATCH --ntasks=6
#SBATCH --gres=gpu:1
#SBATCH --time=100:00:00

module load conda
source activate pytorch 

GRAD="True"
GEO="True"
GEOINTEGRAL="True"


TO_DIVIDE_FACTOR=1.0
BATCH_SIZE=8
LAYERS=(64 64 64 64 64 64 64)
ACT="gelu"

GEO_ACT='soft_identity'

K_MAX=8
# =============================


LAYER_SIZES_STR=$(IFS=,; echo "${LAYERS[*]}")
LOG_DIR="log/${LAYER_SIZES_STR}_${ACT}/"
mkdir -p ${LOG_DIR}


python mpcno_car_test.py \
    --grad $GRAD \
    --geo $GEO \
    --geointegral $GEOINTEGRAL \
    --to_divide_factor $TO_DIVIDE_FACTOR \
    --k_max $K_MAX \
    --layer_sizes $LAYER_SIZES_STR \
    --act $ACT \
    --geo_act $GEO_ACT \
    --bsz $BATCH_SIZE \
    > ${LOG_DIR}/k${K_MAX}_L10_bsz${BATCH_SIZE}_factor${TO_DIVIDE_FACTOR}_grad${GRAD}_geo${GEO}_geoint${GEOINTEGRAL}_geoact_${GEO_ACT}.log

