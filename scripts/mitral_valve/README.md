# Running the Script
To preprocess the data before training, run the script with the preprocess_data argument:
```bash
python pcno_mitral_valve_test.py  "preprocess_data"
```

You can run the script with customized parameters. For example:
```bash
python pcno_mitral_valve_test.py  --train_inv_L_scale together 
```


You can postprocess and visualize the results. For example:
```bash
python plot_results.py
```
