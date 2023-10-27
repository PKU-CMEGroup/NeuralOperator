#!/bin/bash

#SBATCH --job-name=crack_dim_4 
#SBATCH --output=crack_dim_4

#SBATCH -p cpu 
#SBATCH --cpus-per-task=4      # 申请 1 cpu

##SBATCH -p gpu
##SBATCH --gres=gpu:1          # 申请 1 gpu

#SBATCH -N 1
#SBATCH -t 96:00:00

source activate fno

echo crack_4

python -u crack.py 4 >> log.crack_dim_4


