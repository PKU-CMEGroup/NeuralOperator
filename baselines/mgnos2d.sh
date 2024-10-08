#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH -p GPU80G
#SBATCH --qos=low
#SBATCH -J mgnos
#SBATCH --nodes=1 
#SBATCH --ntasks=6
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00

module load conda
source activate pytorch 

python mgnos2d_airfoil_test.py > mgnos_Airfoil.log
python mgnos2d_darcy_test.py > mgnos_Darcy.log