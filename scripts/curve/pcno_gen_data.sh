#!/bin/bash
#SBATCH -o out/PCNO_gen_data.out
#SBATCH --qos=low
#SBATCH -J PCNO_gen_data
#SBATCH --nodes=1 
#SBATCH --ntasks=2
#SBATCH --time=100:00:00

source activate pytorch 
# python generate_curves_data_panel.py > ../../data/curve/quality_test/1_1_5_2d_stokes_panel.log
python generate_curves_data_batch_pre.py  > ../../data/curve/quality_test/1_1_5_5_grad_log_pre.log
# python add_normal.py
