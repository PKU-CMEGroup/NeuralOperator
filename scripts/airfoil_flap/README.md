# Data Information

Download airfoil with flap data from 

PKU drive
https://disk.pku.edu.cn/link/AR64D4A8DC03C44E83832290D85B89322D
Name: airfoil_flap.zip

File Location: AnyShare://Neural-Operator-Data/airfoil_flap.zip


# Running the Script
To preprocess the data before training, run the script with the preprocess_data argument:
```bash
python pcno_airfoilflap_test.py preprocess_data
```

You can run the script with customized parameters. For example:
```
python pcno_airfoilflap_test.py --train_type flap --feature_type pressure --n_train 500 --Lx 1.5 --Ly 0.7 --lr_ratio 5
```


# Parameters

| Name             | Type    | Default Value | Choices                              | Description                                                                                                                                                                                                        |
| ---------------- | ------- | ------------- | ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `--train_type`   | `str`   | `mixed`       | `standard`, `flap`, `mixed`          | Specifies the type of training data:   - `standard`: Data from the airfoil without a flap.  - `flap`: Data from the airfoil with a - `mixed`: A balanced combination of standard and flap data.                    |
| `--feature_type` | `str`   | `mach`        | `pressure`, `mach`, `both`           | Determines the feature type used as the output:      - `pressure`: Outputs the pressure field.     - `mach`: Outputs the Mach number  - `both`: Outputs both pressure and Mach fields.                             |
| `--n_train`      | `int`   | `1000`        | `500`, `1000`, `1500`                | Number of training samples to use.                                                                                                                                                                                 |
| `--train_sp_L`   | `str`   | `False`       | `False`, `together`, `independently` | Specifies whether the spatial length scales (`Lx`, `Ly`) are trained:  - `False`: Do not train the spatial length scales. - `together`: Train `Lx` and `Ly`  - `independently`: Train `Lx` and `Ly` independently. |
| `--Lx`           | `float` | `1.0`         |                                      | Initial value of the spatial length scale Lx.                                                                                                                                                                      |
| `--Ly`           | `float` | `0.5`         |                                      | Initial value of the spatial length scale Ly.                                                                                                                                                                      |
| `--lr_ratio`     | `float` | `10`          |                                      | Learning rate ratio of main parameters and L parameters when train_sp_L is set to `independently`.                                                                                                                 |
---

The output file names will include the parameter values for traceability. For example:

```bash
PCNO_airfoilplap_mixed_mach_n1000_Lx1.0_Ly0.5_20250101_123030.pth
```