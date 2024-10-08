#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH -p GPU80G
#SBATCH --qos=low
#SBATCH -J GeoFNO
#SBATCH --nodes=1 
#SBATCH --ntasks=6
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00

module load conda
source activate pytorch 
python geofno2d_darcy_test.py > geofno2d_Darcy.log
python geofno2d_airfoil_test.py > geofno2d_Airfoil.log

