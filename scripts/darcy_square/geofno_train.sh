#!/bin/bash
#SBATCH -o GeoFNO_train.out
#SBATCH --qos=low
#SBATCH -J GeoFNO_train
#SBATCH -p GPU80G
#SBATCH --nodes=1 
#SBATCH --ntasks=6
#SBATCH --gres=gpu:1
#SBATCH --time=100:00:00

module load conda
source activate pytorch 
python geofno_darcy_test.py > GeoFNO_darcy.log
