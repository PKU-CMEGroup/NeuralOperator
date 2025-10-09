# 1. Data Generation

## 1.1 Curve
\[
r(t)= \tanh\!\Big( r_0 + \sum_{k=1}^{k_{\text{curve}}} \big(a_k \sin(k t) + b_k \cos(k t)\big) \Big) + 1.5
\]
\[
a_k,b_k \sim \mathcal{U}(-0.5,0.5),\quad t_k = 2k\pi/N
\]

## 1.2 Feature Function
\[
f_0(t)= \sum_{i=1}^{k_{\text{feat}}} A_i\sin(i t)+B_i\cos(i t), \qquad
A_i = \frac{u_i}{\sqrt{i}},\; u_i\sim \mathcal{U}(0.5,1.5)
\]
\[
f(t)= \tanh(f_0(t)) + 0.5 \sin(f_0(t))
\]

## 1.3 Integral Operator
\[
g(x_i)= \sum_{j\ne i} K(x_i,x_j)\, f(x_j)\, \Delta s_j
\]
Kernels:
\[
K_{\log}(x,y)= \log(\|x-y\|+\varepsilon),\quad
K_{\text{inv}}(x,y)= \frac{1}{\|x-y\|+\varepsilon},\quad \varepsilon=10^{-6}
\]

![Data Generation Illustration](figures/data.png)
Task: map \((r(t), f) \mapsto g\).  
Default config: \(N=500,\ k_{\text{curve}}=k_{\text{feat}}=3,\ K=K_{\log}\).

---

# 2. PCNO Experiments

## 2.1 Standard PCNO

L = 5

| Setting | Rel Train L2 | Rel Test L2 |
|---------|--------------|-------------|
| k=8 | 0.00179 | 0.07337 |
| norm node_measure, k=8 | 0.00184 | 0.06468 |
| norm node_measure, k=16 | 0.00131 | 0.08581 |

L = 6

| Setting | Rel Train L2 | Rel Test L2 |
|---------|--------------|-------------|
| k=8 | 0.00179 | 0.04639 |
| norm node_measure, k=8 | 0.00087 | 0.03393 |
| norm node_measure, k=16 | 0.00152 | 0.05660 |

L = 10

| Setting | Rel Train L2 | Rel Test L2 |
|---------|--------------|-------------|
| norm node_measure, k=8 | 0.00200 | 0.02157 |
| norm node_measure, k=16 | 0.00150 | 0.03702 |
## 2.2 Shallow Variant
We use $\text{len(layers)}=2$ (only 1 spconv layer), with normalized node measure.

L = 5

| k | Rel Train L2 | Rel Test L2 |
|---|--------------|-------------|
| 8 | 0.00361 | 0.03189 |
| 16 | 0.00189 | 0.03235 |

L = 6

| k | Rel Train L2 | Rel Test L2 |
|---|--------------|-------------|
| 8 | 0.00225 | 0.00478 |
| 16 | 0.00185 | 0.00500 |


## 2.3 Modes Scaling Test 
We use $L=5$, norm node_measure, $k=16$.

Default depth

| scale | Rel Train L2 | Rel Test L2 |
|-------|--------------|-------------|
| 0.5 | 0.00181 | 0.01932 |
| 1.0 | 0.00215 | 0.01545 |
| 2.0 | 0.00356 | 0.03162 |

Shallow ($\text{len(layers)}=2$)

| scale | Rel Train L2 | Rel Test L2 |
|-------|--------------|-------------|
| 0.5 | 0.00167 | 0.00256 |
| 1.0 | 0.00167 | 0.00219 |
| 2.0 | 0.00190 | 0.00203 |


- Shallow + proper scaling → strong stability (test ~2e-3).
- Larger scale harms deeper model but not shallow one (possible spectral bias difference).

---

# 3. Kernel Reconstruction

Question: effective support length of \(k(x-y)\) over domain \(\Omega\) ~ \(2\,\text{diam}(\Omega)\)?  


## 3.1 Kernel: \(k(z)= \log(|z|+10^{-6})\)

Effect of (L, k)

| L | k | rel_L2_error |
|---|---|--------------|
| 5 | 8  | 0.04160 |
| 5 | 16 | 0.02172 |
| 6 | 8  | 0.04862 |
| 6 | 16 | 0.02347 |

Effect of scale (L=5, k=16)

| scale | rel_L2_error |
|-------|--------------|
| 0.5 | 0.02208 |
| 1.0 | 0.02662 |
| 1.5 | 0.02976 |
| 2.0 | 0.03217 |

Notes:
- More modes (k=16) helps.
- Larger scale slightly degrades accuracy (smoothing vs alias trade-off).

## 3.2 Kernel: \(k(z)= z_1\)

Effect of (L, k)

| L | k | rel_L2_error |
|---|---|--------------|
| 5 | 8  | 0.27537 |
| 5 | 16 | 0.19967 |
| 6 | 8  | 0.00487 |
| 6 | 16 | 5.28e-05 |

Effect of scale (L=5, k=16)

| scale | rel_L2_error |
|-------|--------------|
| 0.5 | 3.86e-06 |
| 1.0 | 2.53e-06 |
| 1.5 | 1.55e-06 |
| 2.0 | 1.56e-06 |

Notes:
- Depth (L=6) is critical for linear kernel.
- With proper scaling, extremely low error achievable (near machine precision floor).

## 3.3 Kernel: \(k(z)= \sin(3 z_1) + \cos(2 z_2)\)

Effect of (L, k)

| L | k | rel_L2_error |
|---|---|--------------|
| 5 | 8  | 0.22333 |
| 5 | 16 | 0.15709 |
| 6 | 8  | 0.00184 |
| 6 | 16 | 1.77e-05 |

Effect of scale (L=5, k=16)

| scale | rel_L2_error |
|-------|--------------|
| 0.5 | 4.83e-06 |
| 1.0 | 3.38e-06 |
| 1.5 | 6.91e-06 |
| 2.0 | 4.75e-06 |
