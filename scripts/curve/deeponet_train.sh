#!/bin/bash
#SBATCH -o out/TRANSOLVER_train_log.out
#SBATCH --qos=low
#SBATCH -J TRANSOLVER_train
#SBATCH -p GPU40G
#SBATCH --nodes=1 
#SBATCH --ntasks=6
#SBATCH --gres=gpu:1
#SBATCH --time=100:00:00

# ========== params ==========

KERNEL_TYPE="sp_laplace"  #'sp_laplace', 'dp_laplace', 'adjoint_dp_laplace', 'stokes', 'modified_dp_laplace',  'exterior_laplace_neumann', 'weighted_sp_laplace'

N_TRAIN=2000
N_TEST=1000
N_TWO_CIRCLES_TEST=0
BSZ=32
LR=5e-4

LOG_DIR="log/1_1_5_2d_${KERNEL_TYPE}/deeponet/"
mkdir -p ${LOG_DIR}

python deeponet_curve_test.py \
    --kernel_type $KERNEL_TYPE \
    --n_train $N_TRAIN \
    --n_test $N_TEST \
    --n_two_circles_test $N_TWO_CIRCLES_TEST \
    --bsz $BSZ \
    --base_lr $LR \
    --act gelu \
    --latent_dim 128 \
    --point_encoder_dims 128,128 \
    --branch_dims 128,128 \
    --trunk_dims 128,128 \
    > ${LOG_DIR}/bsz${BSZ}_lr${LR}_sin_mulx.log