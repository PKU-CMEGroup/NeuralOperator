# Data Information
## Data Download
Download the advection diffusion boundary value problem data from:
- PKU drive: https://disk.pku.edu.cn/link/AR082433050E2C4D4E9B31E54728B02B4A
  - Name of the data file: adv_diff_bvp.zip
- File Location: AnyShare://Neural-Operator-Data/adv_diff_bvp.zip

In the downloaded folder, there is a `Data.ipynb` file which contains more detailed introduction to the data.


# Training Tips for Parameter sp_L
When training, you can select one of the following options for the parameter `train_sp_L` from `False`, `together`, and `independently`:
- **`False`**: This option indicates that the parameter `sp_L` will not be trained.
- **`together`**: Selecting this option means that `sp_L` will be trained simultaneously with other parameters using the same optimizer.
- **`independently`**: When choosing this option, `sp_L` will be trained independently in a separate optimizer. The learning rate for this independent training is calculated as `base_lr * lr_ratio`.

In general, keeping `L` fixed during training can lead to a more stable training process. However, training `sp_L` provides the model with a greater chance of generalization. It is recommended to set `train_sp_L` to `independently` and use an appropriate `lr_ratio`. A ratio of 10 is often a suitable choice. If significant fluctuations occur during training and the model performance consistently deteriorates, it is advisable to decrease the `lr_ratio` or reevaluate the selection of `train_sp_L`.


# Example Command

## Data Preprocessing
Before running the training or other code, it's necessary to preprocess the data. The first time you run the program, execute the following command in the terminal:
`python pcno_adv_test.py "preprocess_data"`

Once the data preprocessing is completed successfully, you can proceed with the subsequent steps and run the training or other related code as described below.

## Training
To run a training example, you can use the following command in the terminal:

`python pcno_adv_test.py  --train_distribution 'mixed' --n_train 1000 --train_sp_L 'False'`

Replace the values of `--train_distribution`, `--n_train`, and `--train_sp_L` according to your specific requirements.