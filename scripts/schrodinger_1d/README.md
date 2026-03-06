# 1D Periodic Schrödinger Equation 
## Problem setup (periodic domain)

Let $L>0$ and consider the one-dimensional periodic domain $\mathbb T_L = [0,L)$. We study the (linear) time-dependent Schrödinger equation

$$
i  \partial_t \psi(x,t)=\Big(-\tfrac12\partial_{xx}+V(x)\Big)\psi(x,t),\qquad x\in[0,L),\ t\ge 0,
$$

with **periodic boundary condition**

$$
\psi(x+L,t)=\psi(x,t)\quad\text{for all }x,t,
$$

and initial data

$$
\psi(x,0)=\psi_0(x)\in L^2(\mathbb T_L).
$$

Here $V:\mathbb T_L\to\mathbb R$ is a given (time-independent) **periodic potential**, $V(x+L)=V(x)$. Define the Hamiltonian

$$
H := -\tfrac12\partial_{xx}+V(x).
$$

For real-valued $V$, $H$ is self-adjoint on a suitable domain (e.g. $H^2(\mathbb T_L)$), and the evolution is unitary:

$$
\|\psi(\cdot,t)\|_{L^2(\mathbb T_L)}=\|\psi_0\|_{L^2(\mathbb T_L)}.
$$

---

## Operator (propagator) viewpoint

The solution can be written as

$$
\psi(\cdot,t)=U(t)\psi_0,\qquad U(t)=e^{-itH}.
$$

The family $\{U(t)\}_{t\in\mathbb R}$ forms a one-parameter unitary group: $U(0)=I$, $U(t+s)=U(t)U(s)$, and $U(t)^*=U(-t)$.

---

## Integral (kernel) form of the solution

Since $U(t)$ is a linear operator on $L^2(\mathbb T_L)$, it can be written as an **integral operator** with kernel $K_t(x,y)$:

$$
\psi(x,t) = (U(t)\psi_0)(x) = \int_0^L K_t(x,y) \psi_0(y) dy.
$$

The kernel $K_t$ is periodic in each variable:

$$
K_t(x+L,y)=K_t(x,y),\qquad K_t(x,y+L)=K_t(x,y).
$$


### 1) Constant potential: $V(x)\equiv V_0$

If $V\equiv V_0\in\mathbb R$, then

$$
H = -\tfrac12\partial_{xx} + V_0 =: H_0 + V_0,
\qquad U(t)=e^{-itH}=e^{-iV_0 t}e^{-itH_0}.
$$

Hence the kernel is translation invariant and differs from the free kernel by a global phase:

$$
K_t(x,y)=e^{-iV_0 t} K^{(0)}_t(x,y)=e^{-iV_0 t} k^{(0)}_t(x-y).
$$

On the torus $\mathbb T_L$, the free kernel admits the Fourier series representation

$$
K^{(0)}_t(x,y)=\frac1L\sum_{n\in\mathbb Z}
\exp\!\Big(i\frac{2\pi n}{L}(x-y) - i \tfrac12\Big(\frac{2\pi n}{L}\Big)^2 t\Big),
$$

so

$$
K_t(x,y)=\frac{e^{-iV_0 t}}{L}\sum_{n\in\mathbb Z}
\exp\!\Big(i\frac{2\pi n}{L}(x-y) - i \tfrac12\Big(\frac{2\pi n}{L}\Big)^2 t\Big).
$$

Equivalently, for any initial data $\psi_0$,

$$
\psi(x,t)=\int_0^L K_t(x,y) \psi_0(y) dy
= e^{-iV_0 t}\int_0^L K^{(0)}_t(x,y) \psi_0(y) dy.
$$

---

### 2) Periodic nonconstant potential: $V(x+L)=V(x)$

For a general periodic, nonconstant $V$, the propagator $U(t)=e^{-itH}$ still has an integral-kernel representation

$$
\psi(x,t) = (U(t)\psi_0)(x)=\int_0^L K_t(x,y)  \psi_0(y)  dy,
$$

but in general $K_t(x,y)$ is **not** translation invariant (it depends on $x$ and $y$ separately, not only on $x-y$).

Since the domain is compact, $H=-\tfrac12\partial_{xx}+V(x)$ has a discrete spectrum with an orthonormal eigenbasis $\{(\phi_n,E_n)\}_{n\ge 0}$:

$$
H\phi_n = E_n\phi_n,\qquad \langle \phi_n,\phi_m\rangle=\delta_{nm}.
$$

The kernel admits the spectral expansion

$$
K_t(x,y)=\sum_{n=0}^\infty e^{-iE_n t}  \phi_n(x)  \overline{\phi_n(y)},
$$

and the solution can be written as

$$
\psi(x,t)=\sum_{n=0}^\infty e^{-iE_n t}  \phi_n(x)  \langle \phi_n,\psi_0\rangle.
$$

# Code Structure

## 1) Generate dataset
```bash
python generate_schrodinger1d_data.py
```
The script supports multiple periodic potentials $V$, including `constant`,  `cosine`, `two_mode`, `lattice`

## 2) Train a 1D neural operator, 
Train a Fourier Neural Operator (FNO) to learn the map $\psi_0 \rightarrow \psi_t$ 
```bash
python fno_train.py
```
Train a Transformer to learn the map $\psi_0 \rightarrow \psi_t$ 
```bash
python transformer_train.py
```

## 3) Test and roll out (autoregressive)
Evaluate the trained model on $\psi_0 \rightarrow \psi_t$, then roll the prediction forward for 100 steps:
```bash
python fno_plot_results.py
python transformer_plot_results.py
```


# Observations

## 1)
## 2)