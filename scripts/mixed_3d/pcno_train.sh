#!/bin/bash
#SBATCH -o logs/PCNO_train.out
#SBATCH --qos=low
#SBATCH -J PCNO_train
#SBATCH -p GPU80G
#SBATCH --nodes=1 
#SBATCH --ntasks=6
#SBATCH --gres=gpu:1
#SBATCH --time=100:00:00

module load conda
source activate pytorch
python pcno_mixed_3d_test.py    --grad True \
                                --geo True \
                                --geointegral False \
                                --num_grad 3 \
                                --k_max 8 \
                                --batch_size 8 \
                                --epochs 500 \
                                --scale 0.0 \
                                --normal_prod False \
                                --to_divide_factor 1.0 \
                                > logs/PCNO_mixed_3d_grad_geo.log
