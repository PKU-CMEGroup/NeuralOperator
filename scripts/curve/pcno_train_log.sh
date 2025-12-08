#!/bin/bash
#SBATCH -o out/PCNO_train_log.out
#SBATCH --qos=low
#SBATCH -J PCNO_train
#SBATCH -p GPU40G
#SBATCH --nodes=1 
#SBATCH --ntasks=6
#SBATCH --gres=gpu:1
#SBATCH --time=100:00:00

# ========== 参数设置 ==========
KERNEL_TYPE="sp_laplace"
K_MAX=32
N_TRAIN=9000
N_TEST=1000
TWO_CIRCLES_TEST="True"
NORMAL_PROD="False"
TPYE="1_1_5_2d"
# 动态生成日志目录
if [ "$TWO_CIRCLES_TEST" = "True" ]; then
    LOG_DIR="log/${TPYE}_${KERNEL_TYPE}_two_circles"
else
    LOG_DIR="log/${TPYE}_${KERNEL_TYPE}"
fi

# 创建日志目录
mkdir -p ${LOG_DIR}

# =============================


source activate pytorch 

python pcno_curve_geo_test.py \
    --grad True \
    --geo False \
    --lap False \
    --n_train $N_TRAIN \
    --n_test $N_TEST \
    --kernel_type $KERNEL_TYPE \
    --k_max $K_MAX \
    --two_circles_test $TWO_CIRCLES_TEST \
    --normal_prod $NORMAL_PROD \
    > ${LOG_DIR}/k${K_MAX}_L10_layer2_noact_new.log

python pcno_curve_geo_test.py \
    --grad True \
    --geo True \
    --lap False \
    --n_train $N_TRAIN \
    --n_test $N_TEST \
    --kernel_type $KERNEL_TYPE \
    --k_max $K_MAX \
    --two_circles_test $TWO_CIRCLES_TEST \
    --normal_prod $NORMAL_PROD \
    > ${LOG_DIR}/k${K_MAX}_L10_layer2_noact_geo3wx_new.log

python pcno_curve_geo_test.py \
    --grad False \
    --geo True \
    --lap False \
    --n_train $N_TRAIN \
    --n_test $N_TEST \
    --kernel_type $KERNEL_TYPE \
    --k_max $K_MAX \
    --two_circles_test $TWO_CIRCLES_TEST \
    --normal_prod $NORMAL_PROD \
    > ${LOG_DIR}/k${K_MAX}_L10_layer2_noact_nograd_geo3wx_new.log

python pcno_curve_geo_test.py \
    --grad False \
    --geo False \
    --lap False \
    --n_train $N_TRAIN \
    --n_test $N_TEST \
    --kernel_type $KERNEL_TYPE \
    --k_max $K_MAX \
    --two_circles_test $TWO_CIRCLES_TEST \
    --normal_prod $NORMAL_PROD \
    > ${LOG_DIR}/k${K_MAX}_L10_layer2_noact_nograd_new.log