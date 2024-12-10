#!/bin/bash
#SBATCH -o FNO_train.out
#SBATCH --qos=low
#SBATCH -J FNO_train
#SBATCH -p GPU80G
#SBATCH --nodes=1 
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=100:00:00

module load conda
source activate pytorch 
python fno_burgers_test.py > FNO_burgers.log
