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

KERNEL_TYPE="dp_laplace"  #'sp_laplace', 'dp_laplace', 'adjoint_dp_laplace', 'stokes', 'modified_dp_laplace',  'exterior_laplace_neumann'


LOG_DIR="log/1_1_5_2d_${KERNEL_TYPE}_panel/transolver/"
mkdir -p ${LOG_DIR}

python transolver_curve_test.py \
    --kernel_type $KERNEL_TYPE \
    > ${LOG_DIR}/transolver.log