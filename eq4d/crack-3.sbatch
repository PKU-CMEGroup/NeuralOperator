#!/bin/bash

#SBATCH --job-name=crack_dim_3 
#SBATCH --output=crack_dim_3

#SBATCH -p cpu 
#SBATCH --cpus-per-task=2      # 申请 1 cpu

##SBATCH -p gpu
##SBATCH --gres=gpu:1          # 申请 1 gpu

#SBATCH -N 1
#SBATCH -t 96:00:00

source activate fno

echo crack_3

python -u crack.py 3 >> log.crack_dim_3

