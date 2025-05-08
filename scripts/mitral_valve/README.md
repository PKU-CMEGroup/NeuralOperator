# Data Information

## Data Download
Download mitral valve problem data from 
- PKU drive: https://disk.pku.edu.cn/link/AAD73AE293F7AF43F4B32BFA59CC9BA07E
- Name of the data file: mitral_valve.zip



# Running the Script
To preprocess the data before training, run the script with the preprocess_data argument:
```bash
python pcno_mitral_valve_test.py  "preprocess_data"
```

You can run the script with customized parameters. For example:
```bash
python pcno_mitral_valve_test.py  --train_sp_L together 
```


You can postprocess and visualize the results. For example:
```bash
python plot_results.py
```
