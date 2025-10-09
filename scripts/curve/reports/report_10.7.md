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