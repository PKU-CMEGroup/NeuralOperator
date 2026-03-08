# Mixed_3d Test

This benchmark involves 3D models of cars and aircraft:

- **Cars** (~5,000 models): Derived from [DrivAerNet++](https://dataverse.harvard.edu/dataverse/DrivAerNet).
- **Aircraft** (~6,000 models): Generated using NASA's OpenVSP. These are randomized variations of standard designs like fighter jets and commercial airliners obtained from the [OpenVSP Airshow](https://airshow.openvsp.org/).

Details:

- **Meshes**: All surface meshes are converted to triangular meshes and decimated to about 40,000 elements and isotropically scaled to fit within the bounding box $[-1, 1]^3$.
- **Features**: Surface pressure coefficients ($C_p$) was calculated using the panel method with a freestream inflow condition $v_{\infty} = [1,\,0,\,0]$.

## Download Dataset

[Click to download](https://disk.pku.edu.cn/link/AA581EA0843A6441E8BC6CDBE18DE5EA15)
file name：mixed_3d_add_elem_features.zip

## Get Started

- **Step 1**: Preprocess all data

```bash
sbatch mpcno_preprocess_data.sh
```

**Parameters**:

| Name | Type | Default | Choices | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--n_each` | `int` | `2000` | | Number of each categories to preprocess. |
| `--mesh_type` | `str` | `vertex_centered` | `cell_centered`, `vertex_centered` | Representation of the features (e.g., `cell_centered` stores features at cell centers, while `vertex_centered` stores features at mesh vertices) |

- **Setp 2**: Reduce data, random shuffle and save the first n_train data and the last n_test data to reduce data size for training

```bash
sbatch mpcno_reduce_data.sh
```

**Parameters**:

| Name | Type | Default | Choices | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--n_train` | `int` | `2000` | | Number of samples to use for training. |
| `--n_test` | `int` | `500` | | Number of samples to use for testing. |
| `--mesh_type` | `str` | `vertex_centered` | `cell_centered`, `vertex_centered` | Representation of the features (e.g., `cell_centered` stores features at cell centers, while `vertex_centered` stores features at mesh vertices) |

- **Setp 3**: Training

```bash
# Training with a single GPU
sbatch mpcno_mixed_3d_train.sh

# Training with multiple GPUs
sbatch mpcno_mixed_3d_parallel_train.sh
```

**Parameters**:

| Name | Type | Default | Choices | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--grad` | `str` | `True` | `True`, `False` | Whether to use gradient information in the model. |
| `--geo` | `str` | `True` | `True`, `False` | Whether to use geometric features. |
| `--geointegral` | `str` | `True` | `True`, `False` | Whether to use geometric integral layers. |
| `--to_divide_factor` | `float` | `1.0` | | Scaling factor to divide the node weights. |
| `--k_max` | `int` | `16` | | Maximum frequency (modes) for the Fourier layers. |
| `--batch_size` | `int` | `5` | | Batch size for training. |
| `--epochs` | `int` | `500` | | Number of training epochs. |
| `--n_train` | `int` | `2000` | | Number of samples to use for training. |
| `--n_test` | `int` | `500` | | Number of samples to use for testing. |
| `--act` | `str` | `gelu` | | Activation function for the model (e.g., `gelu`, `relu`). |
| `--geo_act` | `str` | `softsign` | `softsign`, `soft_identity` | Activation function for geometric layers. |
| `--layer_sizes` | `str` | `64,64,64,64,64,64` | | Comma-separated hidden layer dimensions (e.g., "64,64,64"). |
| `--mesh_type` | `str` | `vertex_centered` | `cell_centered`, `vertex_centered` | Representation of the features (e.g., `cell_centered` stores features at cell centers, while `vertex_centered` stores features at mesh vertices) |
