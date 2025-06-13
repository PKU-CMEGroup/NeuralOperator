# Data Information

Download Burgers equation data from 

Google drive
https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-


Burgers_R10.zip
burgers_data_R10.mat

+ The data should be downloaded into the `NeuralOperator/data/burgers/` folder.

+ The shape of the input `a` and output `u` are both `(2048,8192)`.

# Run the Script
Change the `downsample_ratio` from `4,8,16,32` to get resolutions `2048,1024,512,256` respectively. After selecting a `downsample_ratio`, you can follow the steps below to train the model.

## Preprocess data
To preprocess the data before training, run the script `pcno_burgers_test.py` with the preprocess_data argument:
```shell
python pcno_burgers_test.py "preprocess_data"
```
The preprocessed data is saved into the `NeuralOperator/data/burgers/` folder.

## Train
Run the script `pcno_burgers_test.py` without argument:
```shell
python pcno_burgers_test.py
```
