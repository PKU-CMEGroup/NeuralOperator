#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH -p GPU80G
#SBATCH --qos=low
#SBATCH -J GeoFNO
#SBATCH --nodes=1 
#SBATCH --ntasks=6
#SBATCH --gres=gpu:1
#SBATCH --time=100:00:00

module load conda
source activate pytorch 
python geokno_car_test.py > geokno1d_car.log

python geokno1d_airfoil_test.py > geokno1d_airfoil.log
python geokno1d_darcy_test.py > geokno1d_darcy.log
