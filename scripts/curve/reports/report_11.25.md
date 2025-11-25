# Data generation

## 1. Previous method

### curve
$$
\begin{align*}
r(\theta) &= r_0 + \sum_{k=1}^{k_\text{curve}} (\frac{a_k}{k}\cos(\theta) + \frac{b_k}{k}\sin(\theta)), \quad r_0, a_k, b_k \sim U(-1,1)\\
r &= \tanh (r) + 1.5\\
\text{nodes} &= \text{nodes} + \varepsilon \text{RBFfield(nodes)}\\
\theta &= \text{linspace}(0,2\pi,N,\text{endpoint} = \text{False})
\end{align*}
$$

### feature
$$
\begin{align*}
f(\theta) &= \sum_{k=1}^{k_\text{curve}} (\frac{a_k}{k}\cos(\theta) + \frac{b_k}{k}\sin(\theta))\\
f &= \tanh (f) + 0.5 \sin(f)\\
\theta &= \text{linspace}(0,2\pi,N,\text{endpoint} = \text{False})
\end{align*}
$$

### integral

$$
g(x_i) = \sum_{\substack{j=1 \\ j \neq i}}^{N} k(x_i,x_j)f(x_j) w_j
$$


## 2. Current method

### curve
$$
\begin{align*}
r(\theta) &= r_0 + \sum_{k=1}^{k_\text{curve}} (\frac{a_k}{k}\cos(\theta) + \frac{b_k}{k}\sin(\theta)), \quad r_0, a_k, b_k \sim U(-1,1)\\
r &= \tanh (r) + 1.5\\
\end{align*}$$
```
for _ in range(10):
	nodes = nodes + epsilon/10*RBFfield(nodes)
```
$$
\theta = \text{linspace}(0,2\pi,N,\text{endpoint} = \text{False})
$$


### feature
$$
\begin{align*}
f &= \text{RBFfield(nodes)} \\
f &= \tanh(f) + 0.5\sin(f) + 0.2f
\end{align*}$$

### integral
Assuming curve $\Gamma$ and function $f$ is linear on every segment $\Gamma_j$, we have
$$
\begin{align*}
g(x) &= \int_\Gamma k(x,y)f(y)dy \\
&= \sum_j \int_{\Gamma_j} k(x,y)f_j dy \\
&= \sum_j w_j(x)f_j  \\
&= ...
\end{align*}
$$


# Test results

### 1. Grad Log Kernel Results (Rel. L2 Loss)

| Configuration | Seed 1 | Seed 2 | Average |
| :--- | :---: | :---: | :---: |
| Base | 0.1873 | 0.1940 | 0.1907 |
| Grad | 0.0706 | 0.0733 | 0.0719 |
| Geo | 0.0493 | 0.0528 | 0.0510 |
| Grad + Geo | **0.0478** | **0.0503** | **0.0491** |



### 2. Log Kernel Results (Rel. L2 Loss)

| Configuration | Seed 1 |
| :--- | :---: |
| Base | 0.00349 |
| Grad | 0.00336 |
| Geo | 0.00336 |
| Grad + Geo | **0.00329** |


### 3. Stokes Kernel Results (Rel. L2 Loss)

| Configuration | Seed 1 |
| :--- | :---: |
| Base | 0.0215 |
| Grad | 0.0188 |
| Geo | 0.00741 |
| Grad + Geo | **0.00737** |

