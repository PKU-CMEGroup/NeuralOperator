#!/bin/bash
#SBATCH -o PCNO_laplace_train.out
#SBATCH --qos=low
#SBATCH -J laplace
#SBATCH -p GPU80G
#SBATCH --nodes=1 
#SBATCH --ntasks=6
#SBATCH --gres=gpu:1
#SBATCH --time=100:00:00

module load conda
source activate pytorch 
python pcno_test.py --problem_type "Laplace"  --train_sp_L "together" --feature_SDF "True"  > PCNO_laplace_test.log

