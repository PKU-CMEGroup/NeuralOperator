#!/bin/bash
#SBATCH -o MGNO_train.out
#SBATCH --qos=low
#SBATCH -J MGNO_train
#SBATCH -p GPU80G
#SBATCH --nodes=1 
#SBATCH --ntasks=6
#SBATCH --gres=gpu:1
#SBATCH --time=100:00:00

module load conda
source activate pytorch 
python mgno_darcy_test.py > MGNO_darcy.log

