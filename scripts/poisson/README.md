# Data Information

## Data Download
Download Poisson problem data from 
- PKU drive: https://disk.pku.edu.cn/link/ARFDAF6E987AF9481FAA70D3DB32E9DDA1
- Name of the data file: poisson.zip

## Flies in the data
The dataset contains four types of geometries, each with solutions for:
- Laplace equation (boundary conditions only, no source term)
- Poisson equation (both boundary conditions and source term)

Each dataset contains 2048 geometries with two types of boundary conditions (high-frequency and low-frequency functions):

1. **lowfreq**:  
   Domain boundary defined by low-frequency modes.

2. **highfreq**:  
   Domain boundary defined by high-frequency modes.

3. **double**:  
   Domain formed by merging two lowfreq domains.

4. **hole**:  
   Domain with a central hole.

# Running the Script

## PCNO
To **preprocess** the data before training, run with the `preprocess_data` argument in `pcno_preprocess.sh`:
```bash
python pcno_test.py --problem_type "preprocess_data"
```
Note: Adjust the number of preprocessing samples in `pcno_test.py` by modifying `ndata_list = [256, 256, 256, 256]` for each geometry type


To **training** for Laplacian equation, run the script `pcno_laplace_train.sh` with customized parameters. You can adjust the number of training and test samples in the python file. 
```bash
python pcno_test.py --problem_type "Laplace"  --train_sp_L "together" --feature_SDF "True"  > PCNO_laplace_test.log
```

For Poisson equation, you can run the script `pcno_poisson_train.sh` with customized parameters. 
```bash
python pcno_test.py --problem_type "Poisson"  --train_sp_L "together" --feature_SDF "True"  > PCNO_poisson_test.log
```


To **postprocess** the data, run `pcno_laplace_train.sh`. For example:
```bash
python pcno_plot_results.py --problem_type "Laplace"  --train_sp_L "False" --feature_SDF "True"  > PCNO_plot_results.log
```

## BNO

## GNOBNO
