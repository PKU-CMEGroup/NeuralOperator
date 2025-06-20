# Data Information
## Data Download
Download the advection diffusion boundary value problem data from:
- PKU drive: https://disk.pku.edu.cn/link/AA03F972E9AEE6477790DBC9C21C5FA3B1
- Name of the data file: adv_diff_bvp.zip

In the downloaded folder, there is a `Data.ipynb` file which contains more detailed introduction to the data.


# Training Tips for length scale parameter L
When training, you can select one of the following options for the parameter `train_inv_L_scale` from `False`, `together`, and `independently`:
- **`False`**: This option indicates that the length scale parameter `L` will not be trained.
- **`together`**: Selecting this option means that `L` will be trained simultaneously with other parameters using the same optimizer.
- **`independently`**: When choosing this option, `L` will be trained independently in a separate optimizer. The learning rate for this independent training is calculated as `base_lr * lr_ratio`.

In general, keeping `L` fixed during training can lead to a more stable training process. However, training `L` provides the model with a greater chance of generalization. It is recommended to set `train_inv_L_scale` to `independently` and use an appropriate `lr_ratio`. A ratio of 10 is often a suitable choice. If significant fluctuations occur during training and the model performance consistently deteriorates, it is advisable to decrease the `lr_ratio` or reevaluate the selection of `train_inv_L_scale`.


# Example Command

## Data Preprocessing
Before running the training or other code, it's necessary to preprocess the data. The first time you run the program, execute the following command in the terminal:
`python pcno_adv_test.py "preprocess_data"`

Once the data preprocessing is completed successfully, you can proceed with the subsequent steps and run the training or other related code as described below.

## Training
To run a training example, you can use the following command in the terminal:

`python pcno_adv_test.py  --train_distribution 'mixed' --n_train 1000 --train_inv_L_scale 'False'`

Replace the values of `--train_distribution`, `--n_train`, and `--train_inv_L_scale` according to your specific requirements.



# Parameters

| Name             | Type    | Default Value | Choices                              | Description                                                                                                                                                                                                        |
| ----------------- | ------- | ------------- | ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `--train_distribution`   | `str`   | `mixed`       | `uniform, exponential, linear, mixed`        | Specifies the distribution of training data           |
| `--equal_weight` | `str`   |  `False`        | `True`, `False`           | Specify whether to use equal weight   - `True`: Point cloud density - `False`: Uniform density|
| `--n_train`      | `int`   | `1000`        | `500`, `1000`, `1500`                | Number of training samples to use|
| `--n_test`       | `int`   | `200`         |              | Number of testing samples to use|
| `--train_inv_L_scale`   | `str`   | `False`       | `False`, `together`, `independently` | Specifies whether the spatial length scale is trained.|
| `--L`           | `float` | `15.0`         |                                      | Initial value of the spatial length scale Ly.          |
| `--lr_ratio`     | `float` | `10`          |                                      | Learning rate ratio of main parameters and L parameters when train_inv_L_scale is set to `independently`. |
| `--batch_size`     | `int` | `8`          |                                      | Batch size. |