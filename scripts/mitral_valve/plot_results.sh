#!/bin/bash
#SBATCH -o PCNO_plot_results.out
#SBATCH --qos=low
#SBATCH -J Mitral_Valve
#SBATCH -p GPU80G
#SBATCH --nodes=1 
#SBATCH --ntasks=6
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00

module load conda
source activate pytorch 
python plot_results.py > plot_results.log

