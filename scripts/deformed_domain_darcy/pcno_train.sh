#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH -p GPU80G
#SBATCH --qos=low
#SBATCH -J Darcy_pcno
#SBATCH --nodes=1 
#SBATCH --ntasks=6
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

module load conda
source ~/.bashrc
conda activate pytorch 
python pcno_deformed_darcy.py --train_type mixed  --n_train 500 --Lx 2.0 --Ly 2.0 --lr_ratio 5 > log/test.log