#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH -p GPU80G
#SBATCH --qos=low
#SBATCH -J Darcy
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=16
#SBATCH --gres=gpu:1


module load conda
source activate pytorch 
python FFT_2D.py > Darcy.log 
