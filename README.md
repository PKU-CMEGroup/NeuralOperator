# NeuralOperator

<img src="docs/neural_operator.png" width="800" />

NeuralOperator is

* An educational **Neural Operator** library. 
The goal is to provide students with a light-weighted code to explore this area 
and interactive lectures with amazing [Jupyter Notebook](https://jupyter.org/).
* A benchmark repository originally designed to test point cloud neural operator (PCNO) and other **neural operators**. 
The goal is to provide reseachers with access to various parametric partial differential equations and associated datasets, 
enabling researchers to quickly and easily develop and test novel surrogate models.

## Code Structure
* Utilities for all neural networks, such as optimizers and normalizers are in the *utility* folder.
* State-of-the-art neural operators, including `fno.py`, are in the *baselines* folder.
* The Point Cloud Neural Operator (PCNO) and its related utility files are in the *pcno* folder. If you plan to design a new neural operator, consider starting a dedicated folder similar to *pcno*.
* Datasets should be downloaded into the *data* folder. Each subfolder contains one dataset, such as `darcy_square` for the Darcy flow problem in the unit square.
* Test scripts are in the *scripts* folder. Each subfolder contains scripts for various neural operators applied to a specific dataset, with the folder name matching the corresponding dataset subfolder in the *data* folder.

<pre style="white-space: pre-wrap;">
<code>NeuralOperator/

├── utility/
│   ├── adam.py
│   ├── losses.py
│   ├── normalizer.py

├── baselines/
│   ├── (various state-of-the-art neural operators, such as fno.py)

├── pcno/
│   ├── pcno.py
│   ├── geo_utility.py

├── tests/
│   ├── __init__.py
│   ├── (various test files, such as pcno_test.py)

├── data/
│   ├── (various data folders, such as darcy_square)

├── scripts/
│   ├── (various test script folders, such as darcy_square)
</code>
</pre>


## Tutorial
Let's start! (⚠️ under construction)



* Overview
    * [Surrogate Modeling](docs/surrogate_modeling.ipynb) 
* Neural operator
    * [Fourier Neural Operator](docs/fno.ipynb)
    * [Point Cloud Neural Operator](docs/pcno.ipynb)
* Example
    * [Advection-Diffusion Boundary Value Problem](scripts/adv_diff_bvp/README.md)
    * [Darcy Flow Problem on Square Domain](scripts/darcy_square/README.md)
    * [Darcy Flow Problem on Deformed Domain](scripts/darcy_deformed_domain/README.md)
    * [Airfoil](scripts/airfoil/README.md)
    * [Airfoil with flap](scripts/airfoil_flap/README.md)
    * [ShapeNet Car](scripts/car_shapenet/README.md)
    * [Ahmed Body](scripts/ahmed_body/README.md)
    * [Parachute Dynamics](scripts/parachute/README.md)
      


## Submit an issue
You are welcome to submit an issue for any questions related to NeuralOperator. 

<!-- ## Here are some research papers using NeuralOperator
1. Daniel Zhengyu Huang "[Iterated Kalman Methodology For Inverse Problems / Unscented Kalman Inversion](https://arxiv.org/pdf/2102.01580.pdf)." -->



