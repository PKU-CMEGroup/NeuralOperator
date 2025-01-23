# Data Information

Download airfoil with flap data from 

PKU drive
https://disk.pku.edu.cn/link/AR64D4A8DC03C44E83832290D85B89322D
Name: airfoil_flap.zip

File Location: AnyShare://Neural-Operator-Data/airfoil_flap.zip


## Flies in the data
This dataset contains samples for two types of airfoil simulations: single-airfoil (Airfoil) and airfoil-with-flap (Airfoil + Flap).
<pre style="white-space: pre-wrap;"><code>NeuralOperator/
airfoil_flap/

├── Airfoil_data/(samples for single-airfoil)
│   ├── airfoil_mesh/
│       ├── (Data only at boundary)
│   ├── fluid_mesh/
│       ├── elems_00000.npy
│       ├── elems_00001.npy
│       ├── ......
│       ├── features_00000.npy
│       ├── features_00001.npy
│       ├── ......
│       ├── nodes_00000.npy
│       ├── nodes_00001.npy
│       ├── ......

├── Airfoil_flap_data/(samples for airfoil-with-flap)
│   ├── airfoil_mesh/
│       ├── (Data only at boundary)
│   ├── fluid_mesh/
│       ├── elems_00000.npy
│       ├── elems_00001.npy
│       ├── ......
│       ├── features_00000.npy
│       ├── features_00001.npy
│       ├── ......
│       ├── nodes_00000.npy
│       ├── nodes_00001.npy
│       ├── ......

├── Airfoil_residual.npy
├── Airfoil_flap_residual.npy
├── Data.ipynb
</code></pre>

- **`nodes_xxxxx.npy`**: These files contain the coordinates of the mesh nodes. The nodes represent the discretized points in the airfoil or airfoil-flap geometry. The shape is **(nnodes, 2)** 
- **`elems_xxxxx.npy`**: These files define the connectivity between the nodes, specifying which nodes form the triangle elements of the mesh. The shape is **(nelems, 3)**


For instance, visualizations of the mesh are shown below.
![Airfoil Mesh Diagram](./figure/readme/sample_mesh_A.png)
![Airfoil with Flap Mesh Diagram](./figure/readme/sample_mesh_AF.png)


- **`features_xxxxx.npy`**: These files store the features for each element of the mesh, such as pressure or Mach number at the respective locations. The shape of the data is **(nnodes, 3)**, where:
  - **0**: Pressure
  - **1**: Mach number
  - **2**: Indicator (0: interior node, 1: airfoil node, 2: farfield node)

For instance, visualizations of the pressure are shown below.
![Airfoil Diagram](./figure/readme/sample_A.png)
![Airfoil with Flap Diagram](./figure/readme/sample_AF.png)


# Running the Script
To preprocess the data before training, run the script with the preprocess_data argument:
```bash
python pcno_airfoilflap_test.py preprocess_data
```

You can run the script with customized parameters. For example:
```
python pcno_airfoilflap_test.py --train_type flap --feature_type pressure --n_train 500 --lr_ratio 5
```


# Parameters

| Name             | Type    | Default Value | Choices                              | Description                                                                                                                                                                                                        |
| ---------------- | ------- | ------------- | ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `--train_type`   | `str`   | `mixed`       | `standard`, `flap`, `mixed`          | Specifies the type of training data:   - `standard`: Data generated from the single-airfoil configuration.   - `flap`: Data generated from the airfoil-with-flap configuration.   - mixed: A balanced combination of both standard and flap data.                     |
| `--feature_type` | `str`   | `pressure`        | `pressure`, `mach`         | Determines the feature type used as the output:      - `pressure`: Outputs the pressure field.     - `mach`: Outputs the Mach number|
| `--n_train`      | `int`   | `1000`        | `500`, `1000`, `1500`                | Number of training samples to use|
| `--n_test`       | `int`   | `400`         |              | Number of testing samples to use|
| `--train_sp_L`   | `str`   | `False`       | `False`, `together`, `independently` | Specifies the training mode for the spatial length scales (Lx, Ly):  - `False`: Do not train the spatial length scales.  - `together`: Train (Lx, Ly) using the same learning rate as the main parameters.  - `independently`: Train (Lx, Ly) using an independent learning rate |
| `--lr_ratio`     | `float` | `10`          |                                      | Learning rate ratio of L-parameters to main parameters when train_sp_L is set to `independently`. |
---

The output file names will include some parameter values for traceability. For example:

```bash
PCNO_airfoil_flap_mixed_n1000_20250101_123030.pth
```