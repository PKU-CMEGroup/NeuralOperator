#!/bin/bash
#SBATCH -o PCNO_plot_results.out
#SBATCH --qos=low
#SBATCH -J PCNO_plot_results
#SBATCH -p GPU80G
#SBATCH --nodes=1 
#SBATCH --ntasks=6
#SBATCH --gres=gpu:1
#SBATCH --time=100:00:00

module load conda
source activate pytorch 

echo "===== Laplace Results ====="
python pcno_plot_results.py --problem_type "Laplace"  --train_inv_L_scale "False" --feature_SDF "True"  > PCNO_plot_results.log
echo "===== Poisson Results ====="
python pcno_plot_results.py --problem_type "Poisson"  --train_inv_L_scale "False" --feature_SDF "True"  >> PCNO_plot_results.log
