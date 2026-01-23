# Curve Benchmark

This benchmark involves random 1D closed curves in 2D space, used for testing (M)PCNO on various boundary integral operators.

# Data Information
The data is self-generated using the provided scripts. It involves solving boundary integral equations on random polar curves.

## Data Generation & Preprocessing
To generate the curve data and save it in the format required for PCNO, run:

```bash
python generate_curves_data_panel.py
```

This will generate a `.npz` file in `../../data/curve/`. You can configure the `kernel_type` (e.g., `dp_laplace`, `sp_laplace`, `stokes`) and other settings inside the `if __name__ == "__main__":` block of the script.

Alternatively, you can use the shell script for batch generation:
```bash
bash generate_data.sh
```

# Training
To train the model, use the provided training scripts. The main entry point is `mpcno_curve_test.py`.

## Example Command
```bash
python mpcno_curve_test.py --kernel_type 'dp_laplace' --n_train 2000 --n_test 1000 --layer_sizes "64,64,64,64,64,64"
```

Or run the pre-configured shell script:
```bash
bash mpcno_train.sh
```

# Parameters

| Name | Type | Default | Choices | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--grad` | `str` | `True` | `True`, `False` | Whether to use gradient information in the model. |
| `--geo` | `str` | `True` | `True`, `False` | Whether to use geometric features. |
| `--geointegral` | `str` | `True` | `True`, `False` | Whether to use geometric integral layers. |
| `--kernel_type` | `str` | `sp_laplace` | `sp_laplace`, `dp_laplace`, `adjoint_dp_laplace`, `stokes`, `modified_dp_laplace`,  `exterior_laplace_neumann` | The physical kernel type used for the problem. |
| `--to_divide_factor` | `float` | `20.0` | | Scaling factor to divide the node weights. |
| `--k_max` | `int` | `16` | | Maximum frequency (modes) for the Fourier layers. |
| `--bsz` | `int` | `32` | | Batch size for training. |
| `--ep` | `int` | `500` | | Number of training epochs. |
| `--n_train` | `int` | `2000` | | Number of samples to use for training. |
| `--n_test` | `int` | `1000` | | Number of samples to use for testing. |
| `--n_two_circles_test` | `int` | `0` | | Number of samples for testing the two-circle interaction case. |
| `--layer_sizes` | `str` | `64,64,64,64,64,64` | | Comma-separated hidden layer dimensions (e.g., "64,64,64"). |
| `--act` | `str` | `gelu` | | Activation function for the model (e.g., `gelu`, `relu`). |
| `--geo_act` | `str` | `softsign` | `softsign`, `soft_identity` | Activation function for geometric layers. |
