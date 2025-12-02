#!/bin/bash
#SBATCH -o out/PCNO_quality_test.out
#SBATCH --qos=low
#SBATCH -J PCNO_quality_test
#SBATCH --nodes=1 
#SBATCH --ntasks=2
#SBATCH --time=100:00:00

source activate pytorch 
python quality_test.py  > ../../data/curve/quality_test/1_1_5_2d_log_panel_new.log