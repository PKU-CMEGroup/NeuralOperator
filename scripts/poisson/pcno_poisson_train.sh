#!/bin/bash
#SBATCH -o PCNO_poisson_train.out
#SBATCH --qos=low
#SBATCH -J poisson
#SBATCH -p GPU80G
#SBATCH --nodes=1 
#SBATCH --ntasks=6
#SBATCH --gres=gpu:1
#SBATCH --time=100:00:00

module load conda
source activate pytorch 
python pcno_test.py --problem_type "Poisson"  --train_sp_L "independently" --feature_SDF "True"  > PCNO_poisson_test.log

