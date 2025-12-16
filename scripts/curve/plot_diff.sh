#!/bin/bash
#SBATCH -o out/data_plot.out 
#SBATCH --qos=low
#SBATCH -J data_plot
#SBATCH -p C064M0256G
#SBATCH --nodes=1 
#SBATCH --ntasks=12
#SBATCH --time=100:00:00

source ~/.bashrc
conda activate myconda
python plot_difference_of_points.py