#!/bin/sh

for n_data in 64 128 256 512 1024 2048 4096 8192 16384; do
    for k_max in 16 32 64 128; do
        for d_f in 16 32 64 128; do
            for n_fno_layers in 3 4 5; do
                for downsample_ratio in 1 2 4 8; do
               
                    scommand="sbatch --job-name=KS_n_data_${n_data}_k_max_${k_max}_d_f_${d_f}_n_fno_layers_${n_fno_layers}_downsample_ratio_${downsample_ratio}
 --output=output/KS_n_data_${n_data}_k_max_${k_max}_d_f_${d_f}_n_fno_layers_${n_fno_layers}_downsample_ratio_${downsample_ratio}.out 
fno_KS.sbatch $n_data $k_max $d_f $n_fno_layers $downsample_ratio"
                    
                    echo "submit command: $scommand"
                    
                    $scommand
                done
            done
        done
    done 
done

