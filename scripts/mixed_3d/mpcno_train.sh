#!/bin/bash
#SBATCH -o logs/MPCNO_train_vertex_centered.out
#SBATCH --qos=low
#SBATCH -J MPCNO_train_vertex_centered
#SBATCH -p GPU80G
#SBATCH --nodes=1 
#SBATCH --ntasks=16
#SBATCH --gres=gpu:1
#SBATCH --time=100:00:00

module load conda
source activate pytorch
python mpcno_mixed_3d_train.py  --grad True \
                                --geo True \
                                --geointegral True \
                                --k_max 16 \
                                --batch_size 5 \
                                --epochs 500 \
                                --n_train 2000 \
                                --n_test 500 \
                                --to_divide_factor 1.0 \
                                --mesh_type "vertex_centered" \
                                > logs/MPCNO_mixed_3d_vertex_centered.log
