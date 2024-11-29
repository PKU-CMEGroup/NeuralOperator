#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH -p GPU80G
#SBATCH --qos=low
#SBATCH -J GeoKNO
#SBATCH --nodes=1 
#SBATCH --ntasks=6
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00

source activate pytorch
python uu_eq_ng.py > uu_eq_ng.log
python uu_eq_wg.py > uu_eq_wg.log
python uu_un_ng.py > uu_un_ng.log
python uu_un_wg.py > uu_un_wg.log