#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH -p GPU80G
#SBATCH --qos=low
#SBATCH -J Darcy
#SBATCH --nodes=1 
#SBATCH --ntasks=6
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

module load conda
source activate pytorch 
python Transformer2d_darcy_test.py > Transformer_Darcy.log
# python fno2d_airfoil_test.py > Airfoil.log
