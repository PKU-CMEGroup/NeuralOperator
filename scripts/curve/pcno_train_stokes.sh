#!/bin/bash
#SBATCH -o out/PCNO_train_stokes.out
#SBATCH --qos=low
#SBATCH -J PCNO_train
#SBATCH -p GPU40G
#SBATCH --nodes=1 
#SBATCH --ntasks=6
#SBATCH --gres=gpu:1
#SBATCH --time=100:00:00

source activate pytorch 
python pcno_curve_geo_test.py --grad True --geo False --lap False --k_max 16 > log/1_1_5_2d_stokes_panel/k16_L10_layer2_noact.log
python pcno_curve_geo_test.py --grad True --geo True --lap False --k_max 16  > log/1_1_5_2d_stokes_panel/k16_L10_layer2_noact_geo3wx.log
python pcno_curve_geo_test.py --grad False --geo True --lap False --k_max 16  > log/1_1_5_2d_stokes_panel/k16_L10_layer2_noact_nograd_geo3wx.log
python pcno_curve_geo_test.py --grad False --geo False --lap False --k_max 16  > log/1_1_5_2d_stokes_panel/k16_L10_layer2_noact_nograd.log

python pcno_curve_geo_test.py --grad True --geo False --lap False --k_max 16  --normal_prod True > log/1_1_5_2d_stokes_panel/k16_L10_layer2_noact_np.log
python pcno_curve_geo_test.py --grad True --geo True --lap False --k_max 16  --normal_prod True  > log/1_1_5_2d_stokes_panel/k16_L10_layer2_noact_np_geo3wx.log
python pcno_curve_geo_test.py --grad False --geo True --lap False --k_max 16  --normal_prod True  > log/1_1_5_2d_stokes_panel/k16_L10_layer2_noact_np_nograd_geo3wx.log
python pcno_curve_geo_test.py --grad False --geo False --lap False --k_max 16  --normal_prod True  > log/1_1_5_2d_stokes_panel/k16_L10_layer2_noact_np_nograd.log