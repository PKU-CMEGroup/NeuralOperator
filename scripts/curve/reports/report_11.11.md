
# 渐进展开分析

我们考虑局部的积分, 假设流形(以曲线为例)在其上一点$x$附近有弧长参数$y(s)$,且$y(0)=x$.
由曲线的Taylor展开知
$$
y(s) = x + \tau s + \frac{1}{2}\kappa n s^2 + \mathcal{O}(s^3).
$$
那么所要的局部积分为
$$
\begin{align*}
g_\epsilon(x) &= \int_{-\epsilon}^{\epsilon}k(x,y(s))f(y(s)) ds \\
&=\int_{-\epsilon}^{\epsilon}k(x,y(s))(f(x) + \nabla f(x) \cdot (y(s) - x) + \frac{1}{2} (y(s) - x)^T \nabla^2 f(x) (y(s) - x) + \mathcal{O}(s^3)) ds \\
&= \int_{-\epsilon}^{\epsilon}k(x,y(s))(f(x) + \nabla f(x) \cdot (\tau s + \frac{1}{2}\kappa n s^2) + \frac{1}{2} \tau^T \nabla^2 f(x) \tau s^2 + \mathcal{O}(s^3)) ds \\
&= f(x) M_0(x) + \nabla f(x) \cdot \tau M_1(s) + \left( \frac{1}{2} \kappa \nabla f(x) \cdot n + \frac{1}{2} \tau^T \nabla^2 f(x) \tau \right) M_2(x) + \mathcal{o}(\epsilon^3) \\
&= f(x) M_0(x) + \frac{\partial f}{\partial \tau} M_1(s) + \left( \frac{1}{2} \kappa \frac{\partial f}{\partial n} + \frac{1}{2} \frac{\partial^2 f}{\partial \tau^2} \right) M_2(x) + \mathcal{o}(\epsilon^3)
\end{align*}
$$
其中
$$
M_i(x) = \int_{-\epsilon}^{\epsilon}k(x,y(s)) s^i ds
$$应在给定核$k$后只与流形在x处的局部几何确定.

- 对于 Single layer potential,
$$
\begin{align*}
k(x,y) = G(x,y)
&= \log|x-y| \\
&= \log\left| \tau s + \frac{1}{2}\kappa n s^2 + \mathcal{O}(s^3) \right| \\
&= \log|s| + \mathcal{O}(s).
\end{align*}
$$
因此
$$
M_0(x) =  \epsilon \log\epsilon + \mathcal{O}(\epsilon), \quad M_1(x) = \mathcal{O}(\epsilon^3), \quad M_2(x) = \mathcal{O}(\epsilon^3\log\epsilon).
$$

- 对于 Double layer potential，
$$
\begin{align*}
k(x,y) = \frac{\partial}{\partial n_y} G(x,y)
&= \frac{(x-y) \cdot n_y}{|x-y|^2} \\
&= \frac{\big(-\tau s - \tfrac12\kappa n s^2 + \mathcal{O}(s^3)\big) \cdot \big(n - \kappa\,\tau s + \mathcal{O}(s^2)\big)}{\big| \tau s + \tfrac12\kappa n s^2 + \mathcal{O}(s^3)\big|^2} \\
&= \frac{\kappa s^2 - \tfrac12\kappa s^2 + \mathcal{O}(s^3)}{s^2 + \mathcal{O}(s^3)} \\
&= \frac{1}{2}\kappa + \mathcal{O}(s),
\end{align*}
$$因此
$$
M_0(x) = \kappa \epsilon + \mathcal{O}(\epsilon^2), \quad M_1(x) = \mathcal{O}(\epsilon^3), \quad M_2(x) = \mathcal{O}(\epsilon^3).
$$




## 一般 $\mathbb{R}^n$ 中codim = 1 子流形上的推广

令 $\Gamma \subset \mathbb{R}^n$ 为光滑余 1 维闭子流形, 点 $x\in\Gamma$ 处取正交框架 $\{ \tau_1,\dots,\tau_{n-1}, n\}$, 其中 $\tau_i$ 为切向正交基, $n$ 为单位法向。在主方向坐标下的局部参数写为
$$
y(u)= x + \sum_{i=1}^{n-1} \tau_i u_i + \frac{1}{2} n \sum_{i=1}^{n-1}\kappa_i u_i^2 + \mathcal{O}(|u|^3), \qquad u\in U\subset\mathbb{R}^{n-1},
$$
其中 $\kappa_i$ 是主曲率。

因此光滑标量 $f$ 在 $x$ 附近有展开
$$
f(y(u)) = f(x) + \sum_{i}\partial_{\tau_i} f \, u_i + \frac{1}{2}\sum_{i}\partial_{\tau_i\tau_i} f \, u_i^2 + \frac{1}{2}\sum_{i}\kappa_i \partial_n f \, u_i^2 + \mathcal{O}(|u|^3),
$$

设局部积分算子
$$
g_\epsilon(x)= \int_{U_\epsilon} k(x,y(u)) f(y(u)) \, du,
$$
其中 $U_\epsilon=\{u:|u|\le \epsilon\}$。定义各阶“矩” (多重指标 $\alpha$):
$$
M_\alpha(x)= \int_{U_\epsilon} k(x,y(u))\, u^\alpha \, du.
$$
若核与域在 $x$ 附近对称, 则所有奇次矩消失, 二阶矩满足
$$
M_{2,i}(x)= \int_{U_\epsilon} k(x,y(u))\, u_i^2\, du \quad\text{且}\quad M_{2,i} \text{ 与 } i \text{ 无关} \Rightarrow M_{2,i}=M_2.
$$

于是得到推广展开
$$
g_\epsilon(x)= f(x) M_0(x) + \frac{1}{2} M_2(x)\Big(\Delta_\Gamma f(x) + H(x)\partial_n f(x)\Big) + \mathcal{o}(\epsilon^{n+1}).
$$
其中 $\Delta_\Gamma = \sum_i \partial_{\tau_i\tau_i}$ 是流形上的Laplace-Beltrami算子, $H=\sum_i \kappa_i$ 是非归一化的平均曲率。





# Test results

k=16, noact, L=10

| gradient | geo*x |  geo*laplacian | Rel.train error| Rel.test error|
|----------|-------|----------------|----------------|---------------|
|          |       |                |      12.28%    |    13.22%     |
|   Yes    |       |                |      5.54%     |    5.78%      |
|          |  Yes  |                |      4.56%     |    4.94%      |
|          |       |       Yes      |      4.46%     |    5.24%      |
|   Yes    |  Yes  |                |      3.85%     |    4.25%      |
|   Yes    |       |       Yes      |      3.56%     |    4.01%      |
|          |  Yes  |       Yes      |      3.78%     |    4.36%      |
|   Yes    |  Yes  |       Yes      |      3.40%     |    3.93%      |




| gradient | geo*w(x) |  geo*laplacian | Rel.train error| Rel.test error|
|----------|-------|----------------|----------------|---------------|
|          |       |                |      12.28%    |    13.22%     |
|   Yes    |       |                |      5.54%     |    5.78%      |
|          |  Yes  |                |      3.87%     |    4.25%      |
|          |       |       Yes      |      4.46%     |    5.24%      |
|   Yes    |  Yes  |                |      3.57%     |    3.99%      |
|   Yes    |       |       Yes      |      3.56%     |    4.01%      |
|          |  Yes  |       Yes      |      3.46%     |    4.07%      |
|   Yes    |  Yes  |       Yes      |      3.30%     |    3.84%      |


# Split tests

我们将各个特征明确分开，测试到底是什么因素在发挥作用.

```
x1 = speconv(fn, bases_c, bases_s, bases_0, wbases_c, wbases_s, wbases_0)
x2 = w(f)
x3 = gw(self.softsign(compute_gradient(f, directed_edges, edge_gradient_weights)))
geo_weight1 = self.softsign(geow1(geo))
geo_weight2 = self.softsign(geow2(geo))
x_lap = lapw(self.softsign(compute_laplacian(f, directed_edges,edge_gradient_weights)))
x = x1 + geo_weight1*x2 + x3 + geo_weight2*x_lap
```

我们保持只对$f*n$做speconv


| gradient() | w() |  geo*laplacian() | Rel.train error| Rel.test error|
|----------|-------|----------------|----------------|---------------|
|       |   $w(f)$    |                |      16.20% (50ep)    |    16.32% (50ep)     |
|   $x$    |   $w(f)$    |                |      8.40% (50ep)<br>6.38%(500ep)   |    8.25% (50ep)<br>6.64%(500ep)   |
|   $f$    |   $w(f)$    |                |      16.13% (50ep)    |    16.13% (50ep)     |
|   $n$    |   $w(f)$    |                |      15.51% (50ep)    |    15.60% (50ep)      |
|   $f*n$  |   $w(f)$    |                |      8.65% (50ep)    |    8.65%  (50ep)    |
|   $f*n$  |   $w(x)$    |                |      8.72% (50ep)    |    8.84%  (50ep)    |
|          |   $geo*w(f)$    |                |      8.63%(50ep)<br>4.50%(500ep)    |    8.97%  (50ep)<br> 5.08%(500ep)    |
|          |   $geo*w(f)$    |      $geo*lap(f)$          |      8.30%(50ep)<br>4.34%(500ep)    |    8.33%  (50ep)<br> 5.03%(500ep)    |
|          |   $w(f)$    |      $geo*lap(f)$          |      12.31%(50ep)<br>5.96%(500ep)    |    12.39%  (50ep)<br> 6.59%(500ep)    |
|          |   $w(f)$    |      $geo*lap(f*n)$          |      11.30%(50ep)<br>5.55%(500ep)    |    11.64%  (50ep)<br> 6.48%(500ep)    |
|  $f*n$   |  $geo*w(f)$ |      $geo*lap(f)$          |      7.34%(50ep)<br>4.23%(500ep)    |    7.52%  (50ep)<br> 4.86%(500ep)    |