# Layer = [128,128], noact 

### 1. Double layer potential Kernel Results (Rel. L2 Loss)

| Configuration | Seed 1 | Seed 2 | Average |
| :--- | :---: | :---: | :---: |
| Base | 0.1873 | 0.1940 | 0.1907 |
| Grad | 0.0706 | 0.0733 | 0.0719 |
| Geo | 0.0493 | 0.0528 | 0.0510 |
| Grad + Geo | **0.0478** | **0.0503** | **0.0491** |



### 2. Single layer potential Kernel Results (Rel. L2 Loss)

#### k=16

| Configuration | no np |
| :--- | :---: |
| Base | 0.00298 |
| Grad | 0.00297 |
| Geo | 0.00292 | 
| Grad + Geo | **0.00289** | 
| Base (np)|0.00349 |
| Grad (np)|0.00336 |
| Geo (np)|0.00336 |
| Grad + Geo (np)|0.00329 |

#### k=8

| Configuration | Seed 1 | Seed 2 |
| :--- | :---: | :---: |
| Base | 0.00614 | 0.01128 |
| Grad | 0.00614 | 0.01127 |
| Geo | **0.00588** | **0.01081** |
| Grad + Geo | 0.00588 | 0.01083 |


### 3. Stokes Kernel Results (Rel. L2 Loss)

| Configuration | Seed 1 |
| :--- | :---: |
| Base | 0.0226 |
| Grad | 0.0225 |
| Geo | 0.00742 |
| Grad + Geo | 0.00750 |
| Base (np)| 0.0215 |
| Grad (np)| 0.0188 |
| Geo (np)| 0.00741 |
| Grad + Geo (np)| **0.00737** |



### 4. Modified double layer potential Kernel Results (Rel. L2 Loss)

| Configuration | $k = 8$ |  $k = 16$ |  $k = 32$ |
| :--- | :---: | :---: | :---: |
| Base | 0.1679 | 0.0761 | 0.0298 |
| Grad | 0.1573 | 0.0758 | 0.0296 |
| Geo | 0.1189 | 0.0570 | 0.0243 |
| Grad + Geo | **0.0590** | 0.0311 | **0.0230** |
| Base (np)|- | 0.0700 |- |
| Grad (np)|- | 0.0684 |- |
| Geo (np)| -| 0.0497 | -|
| Grad + Geo (np)| | **0.0212** | -|





# Layer = [128,128], act = 'gelu'

### 1. Double layer potential Kernel Results (Rel. L2 Loss)
![](figures/dp_laplace_curve.png)

### 2. Single layer potential Kernel Results (Rel. L2 Loss)
![](figures/sp_laplace_curve.png)

### 3. Stokes Kernel Results (Rel. L2 Loss)
![](figures/stokes_curve.png)



# Gradient plots
$$
x, \frac{\partial x}{\partial x},  \frac{\partial^2 x}{\partial x^2}, \frac{\partial^3 x}{\partial x^3}
$$
![](figures/grad_feature_0.png)
$$
y, \frac{\partial y}{\partial x},  \frac{\partial^2 y}{\partial x^2}, \frac{\partial^3 y}{\partial x^3}
$$
![](figures/grad_feature_1.png)
$$
n_x, \frac{\partial n_x}{\partial x},  \frac{\partial^2 n_x}{\partial x^2}, \frac{\partial^3 n_x}{\partial x^3}
$$
![](figures/grad_feature_2.png)

$$
n_y, \frac{\partial n_y}{\partial x},  \frac{\partial^2 n_y}{\partial x^2}, \frac{\partial^3 n_y}{\partial x^3}
$$
![](figures/grad_feature_3.png)