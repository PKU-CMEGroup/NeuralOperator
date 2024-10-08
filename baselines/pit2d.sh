#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH -p GPU80G
#SBATCH --qos=low
#SBATCH -J Darcy
#SBATCH --nodes=1 
#SBATCH --ntasks=6
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00

module load conda
source activate pytorch 
python pit2d_darcy_test_ds.py > Darcy_pit_ds.log
