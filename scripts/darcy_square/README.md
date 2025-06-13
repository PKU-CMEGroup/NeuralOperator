# Data Information

Download darcy equation data from 

Google drive
https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-


Darcy_421.zip
piececonst_r421_N1024_smooth1.mat
piececonst_r421_N1024_smooth2.mat

+ The data should be downloaded into the `NeuralOperator/data/darcy_square/` folder.

+ The shape of the input `coeff` and output `sol` are both `(2048,421,421)` in total.

# Run the Script
Change the `downsample_ratio` from `1,2,3,5` to get resolutions `421,211,141,85` respectively. After selecting a `downsample_ratio`, you can follow the steps below to train the model.

## Preprocess data
To preprocess the data before training, run the script `pcno_darcy_test.py` with the preprocess_data argument:
```shell
python pcno_darcy_test.py "preprocess_data"
```
The preprocessed data is saved into the `NeuralOperator/data/darcy_square/` folder.
>[!TIP]
>If the preprocessed data is too large, you can replace the original code by the following code to preprocess only the first 1,000 and the last 200 data, which is used as training and testing data respectively.

```python
indices = np.concatenate((np.arange(0, 1000), np.arange(2048 - 200, 2048)))
data_in = np.vstack((data1["coeff"], data2["coeff"]))[indices, 0::downsample_ratio, 0::downsample_ratio] # shape: 1200,421,421 
data_out = np.vstack((data1["sol"], data2["sol"]))[indices, 0::downsample_ratio, 0::downsample_ratio]    # shape: 1200,421,421
```

## Train
Run the script `pcno_darcy_test.py` without argument:
```shell
python pcno_darcy_test.py
```
>[!TIP]
>The `checkpoint.pth` saved `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict` and `current_epoch`. If the training process is too long, you can first train partly and use the `checkpoint` to train successively.