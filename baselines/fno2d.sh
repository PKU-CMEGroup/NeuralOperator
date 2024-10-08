#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH -p GPU80G
#SBATCH --qos=low
#SBATCH -J FNO
#SBATCH --nodes=1 
#SBATCH --ntasks=6
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00

module load conda
source activate pytorch 
python fno2d_darcy_test.py > fno_Darcy.log
python fno2d_airfoil_test.py > fno_Airfoil.log
